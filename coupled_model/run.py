import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
from balanced_cann import EICANN
from input_protocol import input_setup
from visualize_protocol import vis_setup

from configs import (
    slow_coupled_config,
    )

config_and_name = {
        slow_coupled_config: "slow_coupled",
        }

config_file = slow_coupled_config
config = config_file.config


bp.math.set_platform('cpu')
global_dt = 0.01

# criterion = V_threshold * JEE * prob - size_Id * JII * prob * JEE * prob + size_Id * JIE * prob * JEI * prob


def run(exp_id):
    net = EICANN(config)

    # fetch protocols
    input_specs = input_setup[exp_id]
    vis_func = vis_setup[exp_id]

    # init input
    Einp_scale = config.num_ff * config.f_E / bm.sqrt(config.num_ff)
    Iinp_scale = config.num_ff * config.f_I / bm.sqrt(config.num_ff)
    E_inp, I_inp, duration, input_duration = input_specs()
    E_inp = Einp_scale * E_inp * config.mu
    I_inp = Iinp_scale * I_inp * config.mu

    runner = bp.DSRunner(net,
                         jit=True,
                         monitors=[
                             # 'Id.V', 'Ip.V',
                             'E.V',
                             'E.spike',
                             # 'Ip.spike', 'Id.spike',
                             'E2E_s.g', 'E2E_f.g', 'E2I_s.g', 'E2I_f.g',
                             'I2I_s.g', 'I2I_f.g', 'I2E_s.g', 'I2E_f.g',
                         ],
                         inputs=[('E.ext_input', E_inp, 'iter', '='),
                                 ('Id.ext_input', I_inp, 'iter', '='),
                                 ('Ip.ext_input', I_inp, 'iter', '=')],
                         dt=global_dt)
    runner(duration)
    vis_func(runner, net, E_inp, duration, input_duration)
    plt.show()
    return


if __name__ == "__main__":
    # Available protocols are:
    # 'persistent_input': persistent input for bump holding task
    # 'balance_check_bump_input: check balance condition in Ip using bump input
    # 'balance_check_flat_input: check balance condition in Ip using flat input
    # 'irregular_check_flat_input': check whether the network activity is irregular
    # 'tracking_input': tracks a moving input
    # 'convergence_rate_population_readout_input': population readout
    # 'convergence_rate_current_input': plot current for convergence rate analysis
    # 'noise_sensitivity_input': compare bump sensitivity to noise
    # 'sudden_change_convergence_input': analyze converging speed and save results
    # 'smooth_moving_lag_input': compute the lag between stimulus and response and save results
    # 'turn_off_with_exicitation_input': shut off the network with excitation

    import time
    start = time.time()
    run('convergence_rate_population_readout_input')
    print(f'Time cost: {time.time() - start:.2f}s')
