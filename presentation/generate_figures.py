# Fix: use scipy.stats.norm for cdf/ppf to avoid math.erfinv issue.
import os, numpy as np, matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm, t

np.random.seed(0)
OUTDIR="./figs"; os.makedirs(OUTDIR,exist_ok=True)

def setup_fig(aspect_ratio=(16,9),width_in=8):
    w=width_in; h=w*aspect_ratio[1]/aspect_ratio[0]
    fig,ax=plt.subplots(figsize=(w,h),dpi=150); ax.grid(True,alpha=0.2); return fig,ax

# 1) Same mean different shapes
x=np.linspace(-4,4,2000)
fig,ax=setup_fig()
ax.plot(x,norm.pdf(x),label="Normal(0,1)")
ax.plot(x,t.pdf(x,3),label="t(ν=3)")
ax.plot(x,0.5*norm.pdf(x,-1,0.5)+0.5*norm.pdf(x,1,0.5),label="Mixture")
ax.axvline(0,ls="--",c="k"); ax.legend(); ax.set_title("Same mean, different shapes")
f1=os.path.join(OUTDIR,"f01_same_mean_diff_shape.pdf"); fig.savefig(f1); plt.close(fig)

# 2) CDF & quantile mapping
z=np.linspace(-3,3,1000); Fz=norm.cdf(z); tau=0.8; z_tau=norm.ppf(tau)
fig1,ax1=setup_fig(); ax1.plot(z,Fz); ax1.axhline(tau,ls="--"); ax1.axvline(z_tau,ls="--"); ax1.set_ylim(0,1); ax1.set_title("CDF with τ mapping")
taus=np.linspace(1e-4,1-1e-4,1000); q=norm.ppf(taus)
fig2,ax2=setup_fig(); ax2.plot(taus,q); ax2.axvline(tau,ls="--"); ax2.axhline(z_tau,ls="--"); ax2.set_title("Quantile function")
f2=os.path.join(OUTDIR,"f02_cdf_and_quantile_mapping.pdf")
with PdfPages(f2) as pdf: pdf.savefig(fig1); pdf.savefig(fig2)
plt.close(fig1); plt.close(fig2)

# 3) Pinball loss
def rho(d,tau): return np.maximum(tau*d,(tau-1)*d)
d=np.linspace(-2,2,500)
fig,ax=setup_fig()
for tval in [0.1,0.5,0.9]: ax.plot(d,rho(d,tval),label=f"τ={tval}")
ax.legend(); ax.set_title("Pinball loss")
f3=os.path.join(OUTDIR,"f03_pinball_loss_tau_0p1_0p5_0p9.pdf"); fig.savefig(f3); plt.close(fig)

# 4) Huberized pinball
def huberL(d,k): return np.where(abs(d)<=k,0.5*d**2,k*(abs(d)-0.5*k))
def rhok(d,t,k): return np.abs(t-(d<0).astype(float))*huberL(d,k)
def drhok(d,t,k): return np.abs(t-(d<0).astype(float))*np.where(abs(d)<=k,d,k*np.sign(d))
d=np.linspace(-2,2,500); tau=0.9;k=0.2
figA,axA=setup_fig(); axA.plot(d,huberL(d,k),label="Huber L"); axA.plot(d,rhok(d,tau,k),label="ρ^κ_τ"); axA.legend(); axA.set_title("Huberized pinball loss")
figB,axB=setup_fig(); axB.plot(d,drhok(d,tau,k),label="Derivative"); axB.legend(); axB.set_title("Derivative")
f4=os.path.join(OUTDIR,"f04_huber_loss_and_derivative.pdf")
with PdfPages(f4) as pdf: pdf.savefig(figA); pdf.savefig(figB)
plt.close(figA); plt.close(figB)

# 5) Wasserstein quantile overlay
U=np.sort(np.random.normal(0,1,5000)); V=np.sort(np.random.normal(0.6,1.2,5000)); ω=np.linspace(0,1,len(U))
fig,ax=setup_fig(); ax.plot(ω,U,label="F_U^-1"); ax.plot(ω,V,label="F_V^-1"); ax.fill_between(ω,U,V,alpha=0.3); ax.legend(); ax.set_title("Wasserstein via quantiles")
f5=os.path.join(OUTDIR,"f05_wasserstein_quantile_overlay.pdf"); fig.savefig(f5); plt.close(fig)

# 6) Distortion curves
tau=np.linspace(0,1,1000)
def wang(tau,e): return norm.cdf(norm.ppf(tau)+e)
def cvar(tau,e): return np.clip(e*tau,0,1)
def cpw(tau,e): tau=np.clip(tau,1e-6,1-1e-6); return (tau**e)/((tau**e+(1-tau)**e)**(1/e))
fig,ax=setup_fig()
ax.plot(tau,tau,label="Identity")
for e in [-1,0,1]: ax.plot(tau,wang(tau,e),label=f"Wang η={e}")
ax.plot(tau,cpw(tau,0.71),label="CPW 0.71"); ax.plot(tau,cvar(tau,0.2),label="CVaR 0.2")
ax.legend(); ax.set_title("Distortion functions")
f6=os.path.join(OUTDIR,"f06_distortions_beta_curves.pdf"); fig.savefig(f6); plt.close(fig)

[f1,f2,f3,f4,f5,f6]
