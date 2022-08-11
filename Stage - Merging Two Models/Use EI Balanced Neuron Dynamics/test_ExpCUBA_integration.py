import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
from UnitExpCUBA import UnitExpCUBA
import pdb

neu1 = bp.dyn.LIF(1, V_initializer=bm.array([25.]))
neu2 = bp.dyn.LIF(1)
taus = 5.
syn1 = UnitExpCUBA(neu1, neu2, bp.conn.All2All(), g_max=1., tau=taus)
net = bp.dyn.Network(pre=neu1, syn=syn1, post=neu2)
runner = bp.dyn.DSRunner(net, inputs=[('pre.input', 0)], monitors=['pre.V', 'post.V', 'syn.g'], dt=0.01)
runner.run(150.)
fig, gs = bp.visualize.get_figure(2, 1, 3, 8)
fig.add_subplot(gs[0, 0])
plt.plot(runner.mon.ts, runner.mon['pre.V'], label='pre-V')
plt.plot(runner.mon.ts, runner.mon['post.V'], label='post-V')
plt.legend()
fig.add_subplot(gs[1, 0])
plt.plot(runner.mon.ts, runner.mon['syn.g'], label='g')
plt.legend()
plt.show()

print(bm.sum(runner.mon['syn.g']) * 0.01)

