import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
from input_protocol import input_setup
from visualize_protocol import vis_setup
from ei_balance_network import EINet


bp.math.set_platform('cpu')
global_dt = 0.01

# neuron and network parameters
n_scale = 0.125
size_E, size_Id, size_ff = int(800*n_scale), int(100*n_scale), int(1000*n_scale)
num = size_E + size_Id
num_ff = num
prob = 0.25
tau_scale = 10.
tau_E = 2 * tau_scale
tau_I = 2 * tau_scale
V_reset = 0.
V_threshold = 1.
gl = 0.

# ===== EI Balance Parameters ====
ei_scale = 0.8
tau_Ef = 1.0 * tau_scale
tau_If = 0.6 * tau_scale
jie = -4.8 * ei_scale
jii = -3.8 * ei_scale
jee = 2.5 * ei_scale
jei = 5.0 * ei_scale
JIE = jie / bm.sqrt(size_Id*prob)
JII = jii / bm.sqrt(size_Id*prob)
JEE = jee / bm.sqrt(size_E*prob)
JEI = jei / bm.sqrt(size_E*prob)

# inputs
f_E = 0.1
f_I = 0.0
mu = 1.0  # can be thought of as mean firing rate of input neuron

def run(exp_id):
    net = EINet(num_exc=size_E, num_inh=size_Id, tau_E=tau_E, tau_I=tau_I, tau_Ef=tau_Ef, tau_If=tau_If,
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
    # 'linear_input_save': linear input protocol amplitude tracking task with saving data
    # 'constant_input': constant input protocol for irregular spike autocorrelation
    # 'localized_input': localized bump input
    # 'staircase_input': staircase-like input
    # 'staircase_powerspec_input': staircase-like input for power spectrum analysis

    run('linear_input')
