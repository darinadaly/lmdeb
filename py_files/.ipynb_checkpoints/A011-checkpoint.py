# A-011
# python3 /users/darinadaly/lmdeb//py_files/A011.py

# %matplotlibinline
#run notebook_setup
"""isort:skip_file"""

# get_ipython().magic('config InlineBackend.figure_format = "retina"')

import os
import logging
import warnings

import matplotlib.pyplot as plt

# Don't use the schmantzy progress bar
os.environ["EXOPLANET_NO_AUTO_PBAR"] = "true"

# Remove when Theano is updated
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Remove when arviz is updated
warnings.filterwarnings("ignore", category=UserWarning)


logger = logging.getLogger("theano.gof.compilelock")
logger.setLevel(logging.ERROR)
logger = logging.getLogger("exoplanet")
logger.setLevel(logging.DEBUG)


plt.style.use("default")
plt.rcParams["savefig.dpi"] = 100
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 16
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Liberation Sans"]
plt.rcParams["font.cursive"] = ["Liberation Sans"]
plt.rcParams["mathtext.fontset"] = "custom"




# DATA ACCESS
lit_period = 0.44559604  # days
lit_t0 = 2450000 + 1868.83933 - 2457000

# Prior on the flux ratio
lit_flux_ratio = (0.258, 0.2)

import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk

tpf = lk.search_targetpixelfile("TIC 183596242", mission=('TESS')).download()
lc = tpf.to_lightcurve(aperture_mask="all")
lc = lc.remove_nans().normalize()

hdr = tpf.hdu[1].header
texp = hdr["FRAMETIM"] * hdr["NUM_FRM"]
texp /= 60.0 * 60.0 * 24.0

x = np.ascontiguousarray(lc.time, dtype=np.float64)
y = np.ascontiguousarray(lc.flux, dtype=np.float64)
mu = np.median(y)
y = (y / mu - 1) * 1e3

plt.plot((x - lit_t0 + 0.5 * lit_period) % lit_period - 0.5 * lit_period, y, ".k")
plt.xlim(-0.5 * lit_period, 0.5 * lit_period)
plt.xlabel("time since primary eclipse [days]")
_ = plt.ylabel("relative flux [ppt]")
plt.show()

ref_date = 2450000
rvs = np.array(
    [
        (4006.50332 + ref_date, -90.330, 161.590),  # (JD, RV_Aa, RV_Ab); from H_alpha lines
        (4007.52761 + ref_date, 96.905, -85.570),
        (4007.57128 + ref_date, 132.301, -144.540),

        #  (4008.47697 + ref_date, 122.606, 0),
        #  (4008.47271 + ref_date, 0, -136.700),

        (4008.52042 + ref_date, 107.616, -113.824),
        (4375.87729 + ref_date, -110.900, 191.210),
        (4727.14578 + ref_date, 94.415, -68.139),
    ]
)
rvs[:, 0] -= 2457000
rvs = rvs[np.argsort(rvs[:, 0])]

x_rv = np.ascontiguousarray(rvs[:, 0], dtype=np.float64)
y1_rv = np.ascontiguousarray(rvs[:, 1], dtype=np.float64)
y2_rv = np.ascontiguousarray(rvs[:, 2], dtype=np.float64)

fold = (rvs[:, 0] - lit_t0 + 0.5 * lit_period) % lit_period - 0.5 * lit_period
plt.plot(fold, rvs[:, 1], ".", label="primary")
plt.plot(fold, rvs[:, 2], ".", label="secondary")
plt.legend(fontsize=10)
plt.xlim(-0.5 * lit_period, 0.5 * lit_period)
plt.ylabel("radial velocity [km / s]")
_ = plt.xlabel("time since primary eclipse [days]")



# PROBABILISTIC MODEL
import multiprocessing as mp
mp.set_start_method("fork")

# import theano
# theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"

import pymc3 as pm
import theano.tensor as tt
import exoplanet as xo

mask = np.ones_like(x, dtype=bool)
with pm.Model() as model:
    # Systemic parameters
    mean_lc = pm.Normal("mean_lc", mu=0.0, sd=5.0)
    mean_rv = pm.Normal("mean_rv", mu=0.0, sd=50.0)
    u1 = xo.QuadLimbDark("u1")
    u2 = xo.QuadLimbDark("u2")

    # Parameters describing the primary
    M1 = pm.Lognormal("M1", mu=0.0, sigma=10.0, testval=0.612)
    R1 = pm.Lognormal("R1", mu=0.0, sigma=20.0, testval=0.596)

    # Secondary ratios
    k = pm.Lognormal("k", mu=0.0, sigma=20.0, testval=0.9403)  # radius ratio; 0.63/0.67 = 0.9403
    q = pm.Lognormal("q", mu=0.0, sigma=20.0, testval=0.98522)  # mass ratio; given=q=0.98522; or 0.6197/0.6314 = 0.9815
    s = pm.Lognormal("s", mu=0.0, sigma=20.0, testval=1.024)  # surface brightness ratio; 7.32/7.15 = 1.024

    # Prior on flux ratio
    pm.Normal(
        "flux_prior",
        mu=lit_flux_ratio[0],
        sigma=lit_flux_ratio[1],
        observed=k ** 2 * s,
    )

    # Parameters describing the orbit
    b = xo.ImpactParameter("b", ror=k, testval=1.5)
    period = pm.Lognormal("period", mu=np.log(lit_period), sigma=1.0)
    t0 = pm.Normal("t0", mu=lit_t0, sigma=1.0)

    # Parameters describing the eccentricity: ecs = [e * cos(w), e * sin(w)]
    ecs = xo.UnitDisk("ecs", testval=np.array([1e-5, 0.0]))
    ecc = pm.Deterministic("ecc", tt.sqrt(tt.sum(ecs ** 2)))
    omega = pm.Deterministic("omega", tt.arctan2(ecs[1], ecs[0]))

    # Build the orbit
    R2 = pm.Deterministic("R2", k * R1)
    M2 = pm.Deterministic("M2", q * M1)
    orbit = xo.orbits.KeplerianOrbit(
        period=period,
        t0=t0,
        ecc=ecc,
        omega=omega,
        b=b,
        r_star=R1,
        m_star=M1,
        m_planet=M2,
    )

    # Track some other orbital elements
    pm.Deterministic("incl", orbit.incl)
    pm.Deterministic("a", orbit.a)

    # Noise model for the light curve
    sigma_lc = pm.InverseGamma(
        "sigma_lc", testval=1.0, **xo.estimate_inverse_gamma_parameters(0.1, 2.0)
    )
    S_tot_lc = pm.InverseGamma(
        "S_tot_lc", testval=2.5, **xo.estimate_inverse_gamma_parameters(1.0, 5.0)
    )
    ell_lc = pm.InverseGamma(
        "ell_lc", testval=2.0, **xo.estimate_inverse_gamma_parameters(1.0, 5.0)
    )
    kernel_lc = xo.gp.terms.SHOTerm(
        S_tot=S_tot_lc, w0=2 * np.pi / ell_lc, Q=1.0 / 3
    )

    # Noise model for the radial velocities
    sigma_rv1 = pm.InverseGamma(
        "sigma_rv1", testval=1.0, **xo.estimate_inverse_gamma_parameters(0.5, 5.0)
    )
    sigma_rv2 = pm.InverseGamma(
        "sigma_rv2", testval=1.0, **xo.estimate_inverse_gamma_parameters(0.5, 5.0)
    )
    S_tot_rv = pm.InverseGamma(
        "S_tot_rv", testval=2.5, **xo.estimate_inverse_gamma_parameters(1.0, 5.0)
    )
    ell_rv = pm.InverseGamma(
        "ell_rv", testval=2.0, **xo.estimate_inverse_gamma_parameters(1.0, 5.0)
    )
    kernel_rv = xo.gp.terms.SHOTerm(
        S_tot=S_tot_rv, w0=2 * np.pi / ell_rv, Q=1.0 / 3
    )

    # Set up the light curve model
    lc = xo.SecondaryEclipseLightCurve(u1, u2, s)


    def model_lc(t):
        return (
                mean_lc
                + 1e3 * lc.get_light_curve(orbit=orbit, r=R2, t=t, texp=texp)[:, 0]
        )


    # Condition the light curve model on the data
    gp_lc = xo.gp.GP(
        kernel_lc, x[mask], tt.zeros(mask.sum()) ** 2 + sigma_lc ** 2, mean=model_lc
    )
    gp_lc.marginal("obs_lc", observed=y[mask])


    # Set up the radial velocity model
    def model_rv1(t):
        return mean_rv + 1e-3 * orbit.get_radial_velocity(t)


    def model_rv2(t):
        return mean_rv - 1e-3 * orbit.get_radial_velocity(t) / q


    # Condition the radial velocity model on the data
    gp_rv1 = xo.gp.GP(
        kernel_rv, x_rv, tt.zeros(len(x_rv)) ** 2 + sigma_rv1 ** 2, mean=model_rv1
    )
    gp_rv1.marginal("obs_rv1", observed=y1_rv)
    gp_rv2 = xo.gp.GP(
        kernel_rv, x_rv, tt.zeros(len(x_rv)) ** 2 + sigma_rv2 ** 2, mean=model_rv2
    )
    gp_rv2.marginal("obs_rv2", observed=y2_rv)

    # Optimize the logp
    map_soln = model.test_point

    # First the RV parameters
    map_soln = xo.optimize(map_soln, [mean_rv, q])
    map_soln = xo.optimize(
        map_soln, [mean_rv, sigma_rv1, sigma_rv2, S_tot_rv, ell_rv]
    )

    # Then the LC parameters
    map_soln = xo.optimize(map_soln, [mean_lc, R1, k, s, b])
    map_soln = xo.optimize(map_soln, [mean_lc, R1, k, s, b, u1, u2])
    map_soln = xo.optimize(map_soln, [mean_lc, sigma_lc, S_tot_lc, ell_lc])
    map_soln = xo.optimize(map_soln, [t0, period])

    # Then all the parameters together
    map_soln = xo.optimize(map_soln)

    model.gp_lc = gp_lc
    model.model_lc = model_lc
    model.gp_rv1 = gp_rv1
    model.model_rv1 = model_rv1
    model.gp_rv2 = gp_rv2
    model.model_rv2 = model_rv2

    model.x = x[mask]
    model.y = y[mask]

with model:
    period, t0, mean = xo.eval_in_model([model.period, model.t0, model.mean_rv])
# period = map_soln["period"]
# t0 = map_soln["t0"]
# mean = map_soln["mean_rv"]

x_fold = (x_rv - t0 + 0.5 * period) % period - 0.5 * period
plt.plot(fold, y1_rv - mean, ".", label="primary")
plt.plot(fold, y2_rv - mean, ".", label="secondary")

x_phase = np.linspace(-0.5 * period, 0.5 * period, 500)
with model:
    y1_mod, y2_mod = xo.eval_in_model(
        [model.model_rv1(x_phase + t0), model.model_rv2(x_phase + t0)],  # map_soln
    )
plt.plot(x_phase, y1_mod - mean, "C0")
plt.plot(x_phase, y2_mod - mean, "C1")

plt.legend(fontsize=10)
plt.xlim(-0.5 * period, 0.5 * period)
plt.ylabel("radial velocity [km / s]")
plt.xlabel("time since primary eclipse [days]")
_ = plt.title("AE For; map model", fontsize=14)
plt.show()

with model:
    gp_pred = xo.eval_in_model(model.gp_lc.predict(), map_soln) + map_soln["mean_lc"]
    lc = xo.eval_in_model(model.model_lc(model.x), map_soln) - map_soln["mean_lc"]

fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(12, 7))

ax1.plot(model.x, model.y, "k.", alpha=0.2)
ax1.plot(model.x, gp_pred, color="C1", lw=1)

ax2.plot(model.x, model.y - gp_pred, "k.", alpha=0.2)
ax2.plot(model.x, lc, color="C2", lw=1)
ax2.set_xlim(model.x.min(), model.x.max())

ax1.set_ylabel("raw flux [ppt]")
ax2.set_ylabel("de-trended flux [ppt]")
ax2.set_xlabel("time [KBJD]")
ax1.set_title("A-011; map model", fontsize=14)
plt.show()

fig.subplots_adjust(hspace=0.05)

fig, ax1 = plt.subplots(1, figsize=(12, 3.5))

x_fold = (model.x - map_soln["t0"]) % map_soln["period"] / map_soln["period"]
inds = np.argsort(x_fold)

ax1.plot(x_fold[inds], model.y[inds] - gp_pred[inds], "k.", alpha=0.2)
ax1.plot(x_fold[inds] - 1, model.y[inds] - gp_pred[inds], "k.", alpha=0.2)
ax2.plot(x_fold[inds], model.y[inds] - gp_pred[inds], "k.", alpha=0.2, label="data!")
ax2.plot(x_fold[inds] - 1, model.y[inds] - gp_pred, "k.", alpha=0.2)

yval = model.y[inds] - gp_pred
bins = np.linspace(0, 1, 75)
num, _ = np.histogram(x_fold[inds], bins, weights=yval)
denom, _ = np.histogram(x_fold[inds], bins)
ax2.plot(0.5 * (bins[:-1] + bins[1:]) - 1, num / denom, ".w")

args = dict(lw=1)

ax1.plot(x_fold[inds], lc[inds], "C2", **args)
ax1.plot(x_fold[inds] - 1, lc[inds], "C2", **args)

ax1.set_xlim(-1, 1)
ax1.set_ylabel("de-trended flux [ppt]")
ax1.set_xlabel("phase")
_ = ax1.set_title("A-011; map model", fontsize=14)
plt.show()






# MCMC
np.random.seed(23642)
with model:
    trace = xo.sample(
        tune=3500,
        draws=3000,
        start=map_soln,
        chains=4,
        initial_accept=0.8,
        target_accept=0.95,
    )
    

    
    
#RESULTS
pm.summary(trace, var_names=["M1", "M2", "R1", "R2", "ecs", "incl", "s"])

import corner

samples = pm.trace_to_dataframe(trace, varnames=["k", "q", "ecs"])
_ = corner.corner(
    samples,
    labels=["$k = R_2 / R_1$", "$q = M_2 / M_1$", "$e\,\cos\omega$", "$e\,\sin\omega$"],
)


samples = pm.trace_to_dataframe(trace, varnames=["R1", "R2", "M1", "M2"])
weights = 1.0 / trace["ecc"]
weights *= len(weights) / np.sum(weights)
fig = corner.corner(samples, weights=weights, plot_datapoints=False, color="C1")
_ = corner.corner(samples, truths=[1.727, 1.503, 2.203, 1.5488], fig=fig)


plt.hist(
    trace["ecc"] * np.sin(trace["omega"]),
    50,
    density=True,
    histtype="step",
    label="$p(e) = e / 2$",
)
plt.hist(
    trace["ecc"] * np.sin(trace["omega"]),
    50,
    density=True,
    histtype="step",
    weights=1.0 / trace["ecc"],
    label="$p(e) = 1$",
)
plt.xlabel("$e\,\sin(\omega)$")
plt.ylabel("$p(e\,\sin\omega\,|\,\mathrm{data})$")
plt.yticks([])
plt.legend(fontsize=12)
plt.show()


plt.figure()
plt.hist(trace["ecc"], 50, density=True, histtype="step", label="$p(e) = e / 2$")
plt.hist(
    trace["ecc"],
    50,
    density=True,
    histtype="step",
    weights=1.0 / trace["ecc"],
    label="$p(e) = 1$",
)
plt.xlabel("$e$")
plt.ylabel("$p(e\,|\,\mathrm{data})$")
plt.yticks([])
plt.xlim(0, 0.015)
_ = plt.legend(fontsize=12)
plt.show()




weights = 1.0 / trace["ecc"]
print(
    "for p(e) = e/2: p(e < x) = 0.9 -> x = {0:.5f}".format(
        corner.quantile(trace["ecc"], [0.9])[0]
    )
)
print(
    "for p(e) = 1:   p(e < x) = 0.9 -> x = {0:.5f}".format(
        corner.quantile(trace["ecc"], [0.9], weights=weights)[0]
    )
)



samples = trace["R1"]

print(
    "for p(e) = e/2: R1 = {0:.3f} ± {1:.3f}".format(np.mean(samples), np.std(samples))
)

mean = np.sum(weights * samples) / np.sum(weights)
sigma = np.sqrt(np.sum(weights * (samples - mean) ** 2) / np.sum(weights))
print("for p(e) = 1:   R1 = {0:.3f} ± {1:.3f}".format(mean, sigma))


#CITATIONS

with model:
    txt, bib = xo.citations.get_citations_for_model()
print(txt)


print("\n".join(bib.splitlines()[:10]) + "\n...")