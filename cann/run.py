import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
from cann import CANN
from input_protocol import input_setup
from visualize_protocol import vis_setup
from configs import (fast_cann_config,
                     slow_cann_config,
                     turn_off_config,
                     )

config_and_name = {slow_cann_config: "slow_CANN",
                   fast_cann_config: "fast_CANN",
                   turn_off_config: "turn_off_with_excitation",
                   }

config_file = slow_cann_config
config = config_file.config

print(f"Using config: {config_and_name[config_file]}")

bp.math.set_platform('cpu')
global_dt = 0.01

def run(exp_id):
    net = CANN(config)

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
                                   'Ip.V',
                                   'E.V',
                                   'E.spike',
                                   'Ip.spike',
                                   'E2E_s.g',
                                   'E2I_s.g',
                                   'I2I_s.g',
                                   'I2E_s.g',

                                   # for debug purposes
                                #    'E._leak',
                                #    'E._recinp',
                                #    'E._ext',
                                   ],
                         inputs=[('E.ext_input', E_inp, 'iter', '='),
                                 ('Ip.ext_input', I_inp, 'iter', '=')],
                         dt=global_dt)
    runner(duration=duration)
    vis_func(runner, net, E_inp, duration, input_duration)
    plt.show()
    return


if __name__ == "__main__":
    # Available protocols are:
    # 'persistent_input': persistent input for bump holding task, requires "slow_CANN" config
    # 'tracking_input': tracks a moving input, requires "slow_CANN" config
    # 'convergence_rate_population_readout_input': population readout for convergence rate analysis, requires "slow_CANN" config
    # 'convergence_rate_current_input': plot current for convergence rate analysis, requires "slow_CANN" config
    # 'noise_sensitivity_input': compare bump sensitivity to noise
    # 'sudden_change_convergence_input': analyze converging speed for sudden change
    # 'smooth_moving_lag_input': compute the lag between stimulus and response
    # 'turn_off_with_exicitation_input': turn off with excitation input, requires "turn_off_with_excitation" config
    # 'debug_input': unit test for identity between input current calculation and monitoring

    run('convergence_rate_population_readout_input')