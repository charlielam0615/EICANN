import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from model_nc import (EIContinousAttractor,
                      CosineWeightEIContinousAttractor,
                      RandWeightEIContinousAttractor)
from input_protocol import input_setup
from visualize_protocol import vis_setup
from aggregate_protocol import agg_setup
from analysis_protocol import analy_setup
import warnings
import pdb

warnings.filterwarnings("ignore", category=UserWarning)

bp.math.set_platform('cpu')
global_dt = 0.01

# ==== Neuron parameters =====
n_scale = 1
size_E, size_Ip, size_Id, size_ff = int(750*n_scale), int(250*n_scale), int(250*n_scale), int(1000*n_scale)
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
tau_Es = 15 * tau_scale
tau_Is = 0.6 * tau_scale
gEE = 114. * cann_scale / (size_E*1.0)
gEIp = 16. * cann_scale / (size_E*prob)
gIpE = -11. * cann_scale / (size_Ip*prob)
gIpIp = -4. * cann_scale / (size_Ip*prob)
shunting_k = 1.0

# ===== EI Balance Parameters ====
tau_Ef = 0.5 * tau_scale
tau_If = 0.6 * tau_scale
ei_scale = 1.0
jie = -4.8 * ei_scale
jii = -3.8 * ei_scale
jee = 2.5 * ei_scale
jei = 5.0 * ei_scale
JIE = jie / np.sqrt(size_Id*prob)
JII = jii / np.sqrt(size_Id*prob)
JEE = jee / np.sqrt(size_E*prob)
JEI = jei / np.sqrt(size_E*prob)

# ======= Input Parameters ======
f_E = 0.03
f_I = 0.
mu = 1.0

criterion = V_threshold * JEE * prob - size_Id * JII * prob * JEE * prob + size_Id * JIE * prob * JEI * prob


def run(exp_id, loop_num):
    progress_bar = False if loop_num > 1 else True
    range_ = trange if loop_num > 1 else range

    # fetch protocols
    input_specs = input_setup[exp_id]
    vis_func = vis_setup[exp_id]
    agg_func = agg_setup[exp_id]
    agg_res = []

    # net = CosineWeightEIContinousAttractor(
    #     size_E=size_E, size_Ip=size_Ip, size_Id=size_Id, prob=prob,
    #     tau_E=tau_E, tau_I=tau_I,
    #     tau_Es=tau_Es, tau_Is=tau_Is,
    #     tau_Ef=tau_Ef, tau_If=tau_If,
    #     V_reset=V_reset, V_threshold=V_threshold,
    #     JEE=JEE, JEI=JEI, JII=JII, JIE=JIE,
    #     gl=gl, gEE=gEE, gEIp=gEIp, gIpIp=gIpIp, gIpE=gIpE, shunting_k=shunting_k,
    #     weight_scale_cos_range=(0.5, 1.5),
    #     weight_scale_cos_freq=1.5 / bm.pi,
    # )

    net = RandWeightEIContinousAttractor(
        size_E=size_E, size_Ip=size_Ip, size_Id=size_Id, prob=prob,
        tau_E=tau_E, tau_I=tau_I,
        tau_Es=tau_Es, tau_Is=tau_Is,
        tau_Ef=tau_Ef, tau_If=tau_If,
        V_reset=V_reset, V_threshold=V_threshold,
        JEE=JEE, JEI=JEI, JII=JII, JIE=JIE,
        gl=gl, gEE=gEE, gEIp=gEIp, gIpIp=gIpIp, gIpE=gIpE, shunting_k=shunting_k,
        weight_scale_range=(0.9, 1.1),
        a_scale_range=(0.9, 1.1),
    )

    net.init_conn()

    runner = bp.DSRunner(net,
                         jit=True,
                         monitors=['E.spike', 'E2E_s.g'],
                         # inputs=[('E.ext_input', E_inp, 'iter', '='),
                         #         ('Id.ext_input', I_inp, 'iter', '='),
                         #         ('Ip.ext_input', I_inp, 'iter', '=')],
                         dt=global_dt,
                         progress_bar=progress_bar)

    for i in range_(loop_num):
        Einp_scale = num_ff * f_E / bm.sqrt(num_ff)
        Iinp_scale = num_ff * f_I / bm.sqrt(num_ff)
        E_inp, I_inp, duration = input_specs(loop_ratio=i/loop_num)
        E_inp = Einp_scale * E_inp * mu
        I_inp = Iinp_scale * I_inp * mu

        runner.reset_state()
        net.reset_state()
        runner(duration=duration, inputs=(E_inp, I_inp, I_inp))
        res = vis_func(runner, net, E_inp)
        agg_res = agg_func(agg_res, res)

    return agg_res


if __name__ == "__main__":
    # Available protocols are:
    # 'different_input_position_protocol': plot bump distribution at steady state in inhomogeneous CANN
    # 'sanity_check': raster plot of E spike
    protocol = 'different_input_position_protocol'
    results = run(protocol, loop_num=int(size_E*6))
    np.save('coupled_result.npy', results)
    analy_setup[protocol](results)
