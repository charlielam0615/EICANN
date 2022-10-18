from typing import Union, Dict, Callable

import brainpy.math as bm
from brainpy.connect import TwoEndConnector, All2All, One2One
from brainpy.dyn.base import NeuGroup, TwoEndConn
from brainpy.initialize import Initializer, init_param
from brainpy.integrators import odeint, JointEq
from brainpy.types import Tensor

__all__ = [
    'DeltaSynapse',
    'ExpCUBA',
    'ExpCOBA',
    'DualExpCUBA',
    'DualExpCOBA',
    'AlphaCUBA',
    'AlphaCOBA',
    'NMDA',
]

class ExpCUBA(TwoEndConn):
    def __init__(
            self,
            pre: NeuGroup,
            post: NeuGroup,
            conn: Union[TwoEndConnector, Tensor, Dict[str, Tensor]],
            conn_type: str = 'sparse',
            g_max: Union[float, Tensor, Initializer, Callable] = 1.,
            delay_step: Union[int, Tensor, Initializer, Callable] = None,
            tau: Union[float, Tensor] = 8.0,
            name: str = None,
            method: str = 'exp_auto',
    ):
        super(ExpCUBA, self).__init__(pre=pre, post=post, conn=conn, name=name)
        self.check_pre_attrs('spike')
        self.check_post_attrs('input', 'V')

        # parameters
        self.tau = tau
        if bm.size(self.tau) != 1:
            raise ValueError(f'"tau" must be a scalar or a tensor with size of 1. '
                                             f'But we got {self.tau}')

        # connections and weights
        self.pre2post = self.conn.require('pre2post')
        self.g_max = init_param(g_max, self.pre2post[1].shape, allow_none=False)

        # variables
        self.g = bm.Variable(bm.zeros(self.post.num))
        self.delay_step = self.register_delay(f"{self.pre.name}.spike",
                                                delay_step,
                                                self.pre.spike)

        # function
        self.integral = odeint(lambda g, t: -g / self.tau, method=method)


    def reset(self):
        self.g.value = bm.zeros(self.post.num)

    def update(self, t, dt):
        # delays
        pre_spike = self.get_delay_data(f"{self.pre.name}.spike", self.delay_step)

        # post values
        post_vs = bm.pre2post_event_sum(pre_spike,
                                        self.pre2post,
                                        self.post.num,
                                        self.g_max)
        # updates
        self.g.value = self.integral(self.g.value, t, dt=dt) + post_vs
        self.post.input += self.output(self.g)

    def output(self, g_post):
        return g_post


class ExpCOBA(ExpCUBA):
    def __init__(
            self,
            pre: NeuGroup,
            post: NeuGroup,
            # connection
            conn: Union[TwoEndConnector, Tensor, Dict[str, Tensor]],
            conn_type: str = 'sparse',
            # connection strength
            g_max: Union[float, Tensor, Initializer, Callable] = 1.,
            # synapse parameter
            tau: Union[float, Tensor] = 8.0,
            E: Union[float, Tensor] = 0.,
            # synapse delay
            delay_step: Union[int, Tensor, Initializer, Callable] = None,
            # others
            method: str = 'exp_auto',
            name: str = None
    ):
        super(ExpCOBA, self).__init__(pre=pre, post=post, conn=conn,
                                    conn_type=conn_type,
                                    g_max=g_max, delay_step=delay_step,
                                    tau=tau, method=method, name=name)

        # parameter
        self.E = E
        if bm.size(self.E) != 1:
            raise ValueError(f'"E" must be a scalar or a tensor with size of 1. '
                                             f'But we got {self.E}')


    def output(self, g_post):
        return g_post * (self.E - self.post.V)
        