import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import Config

import numpy as np
import brainpy as bp
import brainpy.math as bm


__all__ = ['config']

n_scale = 1
size_E, size_Ip = int(800*n_scale), int(100*n_scale)
num = size_E + size_Ip
num_ff = num
prob = 0.25
tau_scale = 10
cann_scale = 1.25

config = Config(
    global_dt = 0.01,
    stim_a = bm.pi/6,
    conn_a = bm.pi/10,

    # ==== Neuron parameters =====
    n_scale = n_scale,
    size_E=size_E,
    size_Ip=size_Ip,
    num = num,
    num_ff = num_ff,
    prob = prob,
    tau_E = 2 * tau_scale,
    tau_I = 1.5 * tau_scale,
    tau_ref = 5.,
    V_reset = 0.,
    V_threshold = 1.,
    gl = -0.15,

    # ===== CANN Parameters =====
    cann_scale = cann_scale,
    tau_Es = 15 * tau_scale,
    tau_Is = 1.0 * tau_scale,
    gEE = 185. * cann_scale / (size_E*1.0),
    gEIp = 16. * cann_scale / (size_E*prob),
    gIpE = -11. * cann_scale / (size_Ip*prob),
    gIpIp = -4. * cann_scale / (size_Ip*prob),
    shunting_k = 1.0,

    # ===== Input Parameters =====
    f_E = 0.1,
    f_I = 0.,
    mu = 1.0,
)