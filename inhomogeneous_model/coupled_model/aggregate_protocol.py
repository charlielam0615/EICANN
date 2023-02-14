import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
from functools import partial


def different_input_position_protocol(agg_res, res):
    agg_res.append(res)
    return agg_res


def different_input_position_sanity_check(agg_res, res):
    pass

agg_setup = {
    "different_input_position_protocol": partial(different_input_position_protocol),
    "sanity_check": partial(different_input_position_sanity_check),
}