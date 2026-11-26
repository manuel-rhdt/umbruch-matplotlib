import os
import numpy as np, matplotlib; matplotlib.use("module://umbruch_matplotlib")
import matplotlib.pyplot as plt
from umbruch_matplotlib import rich
fig,axs=plt.subplots(1,2,figsize=(9,3.6),constrained_layout=True)
ax=axs[0]; x=np.linspace(0,10,200)
ax.plot(x,np.exp(-x/4)*np.cos(3*x),label=rich("<i>e</i><sup>−<i>t</i>/τ</sup> cos(3<i>t</i>)"))
ax.plot(x,np.exp(-x/4),"--",label=rich("<b>envelope</b> <span style='color:#c0392b'>(τ = 4)</span>"))
ax.set_title(rich("Damped <i>oscillation</i> — <small>H<sub>2</sub>O sample</small>"))
ax.set_xlabel(rich("time <i>t</i> / ms")); ax.set_ylabel(rich("amplitude <i>A</i><sub>0</sub> (a.u.)"))
ax.legend()
ax=axs[1]
ax.plot(x,np.sin(x)); ax.set_title(rich("Mixed: <b>bold</b>, <i>italic</i>, <u>underline</u>, x<sup>2</sup>+y<sub>i</sub>"),fontsize=9)
ax.set_xlabel(rich("$\\int_0^\\infty e^{-x^2}\\,dx = \\frac{\\sqrt{\\pi}}{2}$ is a rich label"))
ax.set_ylabel(rich("$\\hat{\\rho}_{ij}$ with <code>code</code> font"))
ax.annotate(rich("peak: <b>1.0</b> at <i>t</i> = π/2"),(1.57,1),(4,.6),arrowprops=dict(arrowstyle="->"))
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "rich_text.pdf"))
