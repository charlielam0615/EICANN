import brainpy as bp
import brainpy.math as bm


def generate_bump_stimulus(pos, size_n, stim_a):
    def dist(d):
        d = bm.remainder(d, 2 * bm.pi)
        d = bm.where(d > bm.pi, d - 2 * bm.pi, d)
        return d

    x = bm.linspace(-bm.pi, bm.pi, size_n)
    if bm.ndim(pos) == 2:
        x = x[None, ]
    I = 1.0 * bm.exp(-bm.square(dist(x-pos)) / stim_a)
    return I
