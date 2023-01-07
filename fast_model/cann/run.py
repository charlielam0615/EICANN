import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
from cann import CANN
from input_protocol import input_setup
from visualize_protocol import vis_setup

bp.math.set_platform('cpu')
global_dt = 0.01

# ==== Neuron parameters =====
n_scale = 1
size_E, size_Ip, size_ff = 750*n_scale, 250*n_scale, 1000*n_scale
num = size_E + size_Ip
num_ff = num
prob = 0.25
tau_scale = 10
tau_E = 2 * tau_scale
tau_I = 2 * tau_scale
V_reset = 0.
V_threshold = 1.
gl = -0.15

# ===== CANN Parameters =====
cann_scale = 1.0
tau_Es = 0.5 * tau_scale
tau_Is = 0.6 * tau_scale
gEE = 4.8 * cann_scale / bm.sqrt(num)
gEIp = 2.675 * cann_scale / bm.sqrt(num)
gIpE = -5.41 * cann_scale / bm.sqrt(num)
gIpIp = -2. * cann_scale / bm.sqrt(num)
shunting_k = 2.0

f_E = 0.1
f_I = 0.
mu = 1.0


def run(exp_id):
    net = CANN(size_E=size_E, size_Ip=size_Ip, tau_E=tau_E, tau_I=tau_I, tau_Es=tau_Es, tau_Is=tau_Is,
               V_reset=V_reset, V_threshold=V_threshold, prob=prob,
               gl=gl, gEE=gEE, gEIp=gEIp, gIpIp=gIpIp, gIpE=gIpE, shunting_k=shunting_k)

    # fetch protocols
    input_specs = input_setup[exp_id]
    vis_func = vis_setup[exp_id]

    # init input
    Einp_scale = num_ff * f_E / bm.sqrt(num_ff)
    Iinp_scale = num_ff * f_I / bm.sqrt(num_ff)
    E_inp, I_inp, duration = input_specs()
    E_inp = Einp_scale * E_inp * mu
    I_inp = Iinp_scale * I_inp * mu

    runner = bp.dyn.DSRunner(net,
                             jit=True,
                             monitors=['Ip.V', 'E.V', 'E.spike', 'Ip.spike',
                                       'E2E_s.g', 'E2I_s.g',
                                       'I2I_s.g', 'I2E_s.g',
                                       ],
                             inputs=[('E.ext_input', E_inp, 'iter', '='),
                                     ('Ip.ext_input', I_inp, 'iter', '=')],
                             dt=global_dt)
    runner(duration)
    vis_func(runner, net, E_inp)
    plt.show()
    return


if __name__ == "__main__":
    # Available protocols are:
    # 'background_input': background input only for checking spontaneous activity
    # 'persistent_input': persistent input for bump holding task
    # 'noisy_input': persistent input for bump holding task
    # 'global_inhibition': input with two bumps for global inhibition test
    # 'tracking_input': tracks a moving input
    # 'compare_speed_input': population readout for convergence rate analysis
    # 'compare_current_input': plot current for convergence rate analysis
    # 'compare_noise_sensitivity_input': compare bump sensitivity to noise

    run('compare_noise_sensitivity_input')