from brainpy.dyn import NeuGroup, TwoEndConn, DynamicalSystem
from brainpy.connect import TwoEndConnector, All2All, One2One
from brainpy.types import Array
from brainpy.initialize import Initializer, init_param
from typing import Union, Dict, Callable
from brainpy.integrators import odeint, JointEq
import brainpy.math as bm
import brainpy as bp

import pdb


class UnitExpCUBA(TwoEndConn):
    def __init__(
            self,
            pre: NeuGroup,
            post: NeuGroup,
            conn: Union[TwoEndConnector, Array, Dict[str, Array]],
            conn_type: str = 'sparse',
            g_max: Union[float, Array, Initializer, Callable] = 1.,
            delay_step: Union[int, Array, Initializer, Callable] = None,
            tau: Union[float, Array] = 8.0,
            name: str = None,
            method: str = 'exp_auto',
    ):
        super(UnitExpCUBA, self).__init__(pre=pre, post=post, conn=conn, name=name)
        self.check_pre_attrs('spike')
        self.check_post_attrs('input', 'V')

        # parameters
        self.tau = tau
        if bm.size(self.tau) != 1:
            raise ValueError(f'"tau" must be a scalar or a Array with size of 1. '
                                             f'But we got {self.tau}')

        # connections and weights
        self.conn_type = conn_type
        if conn_type not in ['sparse', 'dense']:
            raise ValueError(f'"conn_type" must be in "sparse" and "dense", but we got {conn_type}')
        if self.conn is None:
            raise ValueError(f'Must provide "conn" when initialize the model {self.name}')
        if isinstance(self.conn, One2One):
            self.g_max = init_param(g_max, (self.pre.num,), allow_none=False)
            self.weight_type = 'heter' if bm.size(self.g_max) != 1 else 'homo'
        elif isinstance(self.conn, All2All):
            self.g_max = init_param(g_max, (self.pre.num, self.post.num), allow_none=False)
            if bm.size(self.g_max) != 1:
                self.weight_type = 'heter'
                bm.fill_diagonal(self.g_max, 0.)
            else:
                self.weight_type = 'homo'
        else:
            if conn_type == 'sparse':
                self.pre2post = self.conn.require('pre2post')
                self.g_max = init_param(g_max, self.pre2post[1].shape, allow_none=False)
                self.weight_type = 'heter' if bm.size(self.g_max) != 1 else 'homo'
            elif conn_type == 'dense':
                self.g_max = init_param(g_max, (self.pre.num, self.post.num), allow_none=False)
                self.weight_type = 'heter' if bm.size(self.g_max) != 1 else 'homo'
                if self.weight_type == 'homo':
                    self.conn_mat = self.conn.require('conn_mat')
            else:
                raise ValueError(f'Unknown connection type: {conn_type}')

        # variables
        self.g = bm.Variable(bm.zeros(self.post.num))
        self.output_value = bm.Variable(bm.zeros(self.post.num))

        self.delay_step = self.register_delay(f"{self.pre.name}.spike", delay_step, self.pre.spike)

        # function
        self.integral = odeint(lambda g, t: -g / self.tau, method=method)

    def reset(self):
        self.g.value = bm.zeros(self.post.num)
        if self.delay_step is not None:
            self.reset_delay(f"{self.pre.name}.spike", self.pre.spike)

    def update(self, tdi):
        t, dt = tdi.t, tdi.dt
        # delays
        if self.delay_step is None:
            pre_spike = self.pre.spike
        else:
            pre_spike = self.get_delay_data(f"{self.pre.name}.spike", self.delay_step)
            self.update_delay(f"{self.pre.name}.spike", self.pre.spike)

        # post values
        assert self.weight_type in ['homo', 'heter']
        assert self.conn_type in ['sparse', 'dense']
        if isinstance(self.conn, All2All):
            pre_spike = pre_spike.astype(bm.float_)
            if self.weight_type == 'homo':
                post_vs = bm.sum(pre_spike)
                if not self.conn.include_self:
                    post_vs = post_vs - pre_spike
                post_vs = self.g_max / self.tau * post_vs 
            else:
                post_vs = pre_spike @ (self.g_max / self.tau)
        elif isinstance(self.conn, One2One):
            pre_spike = pre_spike.astype(bm.float_)
            post_vs = pre_spike * self.g_max / self.tau
        else:
            if self.conn_type == 'sparse':
                post_vs = bm.pre2post_event_sum(pre_spike,
                                                self.pre2post,
                                                self.post.num,
                                                self.g_max / self.tau)
            else:
                pre_spike = pre_spike.astype(bm.float_)
                if self.weight_type == 'homo':
                    post_vs = (self.g_max / self.tau) * (pre_spike @ self.conn_mat)
                else:
                    post_vs = pre_spike @ (self.g_max / self.tau)

        # updates
        self.g.value = self.integral(self.g.value, t, dt=dt) + post_vs
        self.output_value.value = self.output(self.g)
        self.post.input += self.output_value

    def output(self, g_post):
        return g_post