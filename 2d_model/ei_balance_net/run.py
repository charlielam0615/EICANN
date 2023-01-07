import brainpy as bp
import numpy as np
import brainpy.math as bm
import matplotlib.pyplot as plt
from input_protocol import input_setup
from visualize_protocol import vis_setup
from ei_balance_network_2d import EINet2D


bp.math.set_platform('cpu')
global_dt = 0.01

# neuron and network parameters
w, h = 32, 32
qI = 0.25
qE = 1 - qI
inh_w, inh_h = int(w*np.sqrt(qI)), int(h*np.sqrt(qI))
num_inh = inh_w * inh_h
exc_w, exc_h = int(w*np.sqrt(qE)), int(h*np.sqrt(qE))
num_exc = exc_w * exc_h
num_ff = num = w*h
prob = 0.25
tau_scale = 10.
tau_E = 2.0 * tau_scale
tau_I = 2.0 * tau_scale
V_reset = 0.
V_threshold = 1.

# EI balance synapse parameters
tau_Ef = 0.5 * tau_scale
tau_If = 0.6 * tau_scale

ei_scale = 1.0 * 0.6
jie = -8. * ei_scale
jii = -6.4 * ei_scale
jee = 4.12 * ei_scale
jei = 8.3 * ei_scale
JIE = jie / bm.sqrt(num_inh*prob)
JII = jii / bm.sqrt(num_inh*prob)
JEE = jee / bm.sqrt(num_exc*prob)
JEI = jei / bm.sqrt(num_exc*prob)
gl = -0.15

# inputs
mu = 0.1  # can be thought of as mean firing rate of input neuron
f_E = 1.0
f_I = 0.


def run(exp_id):
    net = EINet2D(exc_w=exc_w, exc_h=exc_h, inh_w=inh_w, inh_h=inh_h, tau_E=tau_E, tau_I=tau_I, tau_Ef=tau_Ef, tau_If=tau_If,
                  V_threshold=V_threshold, V_reset=V_reset, prob=prob, gl=gl, JEI=JEI, JEE=JEE, JII=JII, JIE=JIE)

    # fetch protocols
    input_specs = input_setup[exp_id]
    vis_func = vis_setup[exp_id]

    # init input
    Einp_scale = num_ff * f_E / bm.sqrt(num_ff)
    Iinp_scale = num_ff * f_I / bm.sqrt(num_ff)
    E_inp, I_inp, duration = input_specs
    E_inp = Einp_scale * E_inp * mu
    I_inp = Iinp_scale * I_inp * mu

    runner = bp.dyn.DSRunner(net,
                             jit=True,
                             monitors=['E2I.g', 'E2E.g', 'I2I.g', 'I2E.g', 'E.input', 'E.spike', 'I.spike', 'E.V'],
                             inputs=[('E.ext_input', E_inp, 'iter', '='),
                                     ('I.ext_input', I_inp, 'iter', '=')],
                             dt=global_dt)
    runner.run(duration)

    vis_func(runner, duration, E_inp)
    plt.show()

    return


if __name__ == '__main__':
    # Available protocols are:
    # 'sin_input': sin input protocol for fast response task
    # 'linear_input': linear input protocol amplitude tracking task
    # 'constant_input': constant input protocol for irregular spike autocorrelation
    # 'check_current_input': constant input and visualize currents
    plt.style.use('ggplot')
    run('check_current_input')


