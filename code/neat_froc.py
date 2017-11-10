import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter
import numpy as np

# plot settings
FROC_minX = 0.125 # Mininum value of x-axis of FROC curve
FROC_maxX = 8 # Maximum value of x-axis of FROC curve
bLogPlot = True

fp_points = [0.125, 0.25, 0.5, 1, 2, 4, 8]

ax = plt.gca()
fps_itp = np.linspace(FROC_minX, FROC_maxX, num=10001)

#plot all
plt.plot(fps_itp, sens_itp, label="(%.3f) \t %s" % (overall_score, CADSystemName), lw=2)


xmin = FROC_minX
xmax = FROC_maxX
plt.xlim(xmin, xmax)
plt.ylim(0, 1)
plt.xlabel('Average number of false positives per scan')
plt.ylabel('Sensitivity')
plt.legend(loc='lower right')
plt.title('FROC performance')

if bLogPlot:
    plt.xscale('log', basex=2)
    ax.xaxis.set_major_formatter(FixedFormatter([0.125, 0.25, 0.5, 1, 2, 4, 8]))

# set your ticks manually
ax.xaxis.set_ticks([0.125, 0.25, 0.5, 1, 2, 4, 8])
ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
plt.grid(b=True, which='both')
plt.tight_layout()

plt.savefig(os.path.join(outputDir, "%s-froc.png" % fname), bbox_inches=0, dpi=300)