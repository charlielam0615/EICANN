import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
from model import EICANN
from input_protocol import input_setup
from visualize_protocol import vis_setup


bp.math.set_platform('cpu')
global_dt = 0.01

# ==== Neuron parameters =====
n_scale = 2
qI = 0.25
qE = 0.75
w, h = 35*n_scale, 35*n_scale
a = bm.pi/12

E_w, E_h = int(w*np.sqrt(qE)), int(h*np.sqrt(qE))
num_E, size_E = E_w*E_h, (E_h, E_w)
Ip_w, Ip_h = int(w*np.sqrt(qI)), int(h*np.sqrt(qI))
num_Ip, size_Ip = Ip_w*Ip_h, (Ip_h, Ip_w)
Id_w, Id_h = int(w*np.sqrt(qI)), int(h*np.sqrt(qI))
num_Id, size_Id = Id_w*Id_h, (Id_h, Id_w)
num_ff = num = num_E + num_Ip
prob = 0.25
tau_scale = 10
tau_E = 2 * tau_scale
tau_I = 2 * tau_scale
V_reset = 0.
V_threshold = 1.
gl = -0.15

# ===== CANN Parameters =====
cann_scale = 1.0 * 2.5 #/ (bm.sqrt(bm.pi/2)*a)
tau_Es = 15 * tau_scale
tau_Is = 5 * tau_scale
gEE = 113.85 * cann_scale / (num_E*1.0) * 3.8
gEIp = 15.8 * cann_scale / (num_E*prob)
gIpE = -10.7 * cann_scale / (num_Ip*prob)
gIpIp = -3.95 * cann_scale / (num_Ip*prob)
shunting_k = 4.0  # 4.0

# ===== EI Balance Parameters ====
tau_Ef = 0.5 * tau_scale
tau_If = 0.6 * tau_scale
ei_scale = 1.0 * 1.2 #/ (bm.sqrt(bm.pi/2)*a)
jie = -8. * ei_scale
jii = -6.4 * ei_scale
jee = 4.12 * ei_scale
jei = 8.3 * ei_scale
JIE = jie / bm.sqrt(num_Id*prob)
JII = jii / bm.sqrt(num_Id*prob)
JEE = jee / bm.sqrt(num_E*prob)
JEI = jei / bm.sqrt(num_E*prob)

# ======= Input Parameters ======
f_E = 0.1 * 10  #/ (bm.sqrt(bm.pi/2)*a)
f_I = 0.015 * 10 * 0
mu = 1.0


def run(exp_id):
    net = EICANN(size_E=size_E, size_Ip=size_Ip, size_Id=size_Id,
                 tau_E=tau_E, tau_I=tau_I,
                 tau_Es=tau_Es, tau_Is=tau_Is,
                 tau_Ef=tau_Ef, tau_If=tau_If,
                 V_reset=V_reset, V_threshold=V_threshold, prob=prob,
                 JEE=JEE, JEI=JEI, JII=JII, JIE=JIE,
                 gl=gl, gEE=gEE, gEIp=gEIp, gIpIp=gIpIp, gIpE=gIpE, shunting_k=shunting_k, a=a)

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
                             monitors=[
                                       'Id.V', 'Ip.V', 'E.V',
                                       'E.spike', 'Ip.spike', 'Id.spike',
                                       'E2E_s.g', 'E2E_f.g', 'E2I_s.g', 'E2I_f.g',
                                       'I2I_s.g', 'I2I_f.g', 'I2E_s.g', 'I2E_f.g',
                                       'ESI.output_value',
                                       ],
                             inputs=[('E.ext_input', E_inp, 'iter', '='),
                                     ('Id.ext_input', I_inp, 'iter', '='),
                                     ('Ip.ext_input', I_inp, 'iter', '=')],
                             dt=global_dt)
    runner(duration)
    vis_func(runner, net, E_inp)
    plt.show()
    return


if __name__ == "__main__":
    # Available protocols are:
    # 'persistent_input': persistent input for bump holding task
    # 'persistent_spiking_input': persistent input in spikes for bump holding task
    # 'tracking_input': tracks a moving input
    # 'tracking_spiking_input': tracks a moving spiking input
    # 'spiking_camera_input': use spiking camera data as input
    plt.style.use('ggplot')
    run('persistent_spiking_input')