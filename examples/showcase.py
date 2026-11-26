import os
import numpy as np, matplotlib; matplotlib.use("module://umbruch_matplotlib")
matplotlib.rcParams["font.family"] = "TeX Gyre Heros"   # ships with umbruch; has italic and bold
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from umbruch_matplotlib import rich
rng=np.random.default_rng(1)
fig=plt.figure(figsize=(13,9),constrained_layout=True)
gs=fig.add_gridspec(3,3)
fig.suptitle(rich("umbruch-matplotlib showcase — <b>rich text</b>, <i>MathML</i>, meshes and hatches"),fontsize=14)

# 1 line plot with rich legend
ax=fig.add_subplot(gs[0,0]); x=np.linspace(0,10,300)
for k,c in zip((1,2,3),("C0","C1","C2")):
    ax.plot(x,np.exp(-x/5)*np.cos(k*x),label=rich(f"<i>k</i> = {k}, e<sup>−<i>t</i>/5</sup>"))
ax.set_title(rich("Damped <i>oscillations</i>")); ax.set_xlabel(rich("time <i>t</i> (ms)")); ax.set_ylabel(rich("<i>V</i><sub>m</sub> (mV)"))
ax.legend(fontsize=7)

# 2 errorbar + math
ax=fig.add_subplot(gs[0,1]); xs=np.arange(1,9)
ax.errorbar(xs,xs**1.5+rng.normal(0,1,8),yerr=1+0.2*xs,fmt="o",capsize=3,label="data")
ax.plot(xs,xs**1.5,"-",label=r"$y=x^{3/2}$")
ax.set_title(r"Power law $y\propto x^{3/2}$"); ax.set_xlabel(r"$\sqrt{x}\;\mathrm{(a.u.)}$"); ax.set_ylabel(r"$\int_0^x t^{1/2}\,dt$")
ax.legend()

# 3 log-log with rich ticks
ax=fig.add_subplot(gs[0,2]); f=np.logspace(-1,3,200)
ax.loglog(f,1/(1+(f/20)**2),label=rich("low-pass <i>f</i><sub>c</sub> = 20 Hz"))
ax.loglog(f,1/(1+(20/f)**2),label=rich("high-pass"))
ax.set_title("Bode magnitude"); ax.set_xlabel(rich("frequency <i>f</i> (Hz)")); ax.grid(True,which="both",alpha=.3); ax.legend(fontsize=8)

# 4 bars with hatches
ax=fig.add_subplot(gs[1,0]); h=["//","\\\\","xx","..","oo"]
for i,(v,ht) in enumerate(zip([3,5,2,4,6],h)): ax.bar(i,v,hatch=ht,color=f"C{i}",edgecolor="k")
ax.set_xticks(range(5)); ax.set_xticklabels([rich(f"<b>{c}</b><sub>{i+1}</sub>") for i,c in enumerate("ABCDE")])
ax.set_title("Hatched bars")

# 5 imshow + colorbar
ax=fig.add_subplot(gs[1,1]); yy,xx=np.mgrid[-3:3:200j,-3:3:200j]
im=ax.imshow(np.sin(xx*yy)*np.exp(-(xx**2+yy**2)/6),extent=(-3,3,-3,3),cmap="RdBu_r",origin="lower")
cb=fig.colorbar(im,ax=ax); cb.set_label(rich("<i>ψ</i>(<i>x</i>,<i>y</i>)"))
ax.set_title("imshow"); ax.set_xlabel("x"); ax.set_ylabel("y")

# 6 contourf + contour labels
ax=fig.add_subplot(gs[1,2]); z=np.exp(-((xx-1)**2+yy**2))-np.exp(-((xx+1)**2+yy**2))
cs=ax.contourf(xx,yy,z,12,cmap="coolwarm"); ax.contour(xx,yy,z,12,colors="k",linewidths=.4)
ax.set_title("Dipole potential"); ax.set_aspect("equal")

# 7 gouraud pcolormesh
ax=fig.add_subplot(gs[2,0]); X,Y=np.meshgrid(np.linspace(0,1,12),np.linspace(0,1,12))
ax.pcolormesh(X,Y,np.sin(6*X)*np.cos(6*Y),shading="gouraud",cmap="viridis")
ax.set_title("Gouraud shading"); ax.set_aspect("equal")

# 8 scatter alpha + annotate
ax=fig.add_subplot(gs[2,1]); p=rng.normal(size=(400,2))
sc=ax.scatter(p[:,0],p[:,1],c=np.hypot(*p.T),s=40,alpha=.55,cmap="plasma")
ax.annotate(rich("outlier: <b>3.1σ</b>"),(2.5,2.2),(0.3,2.6),arrowprops=dict(arrowstyle="->"),fontsize=9)
ax.set_title(rich("Scatter, <i>α</i> = 0.55")); ax.set_xlabel(rich("<i>x</i><sub>1</sub>")); ax.set_ylabel(rich("<i>x</i><sub>2</sub>"))

# 9 fill_between + math
ax=fig.add_subplot(gs[2,2]); t=np.linspace(0,2*np.pi,300)
ax.fill_between(t,np.sin(t),np.sin(t)+.5*np.cos(2*t),alpha=.4,label=r"$\sin t \pm \frac{1}{2}\cos 2t$")
ax.plot(t,np.sin(t),"k"); ax.set_xticks([0,np.pi,2*np.pi]); ax.set_xticklabels(["0",r"$\pi$",r"$2\pi$"])
ax.set_title(r"Band: $\hat\rho_{ij}=\frac{\langle x_i x_j\rangle}{\sigma_i\sigma_j}$",fontsize=10,pad=11); ax.legend(fontsize=8)
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "showcase.pdf"))
