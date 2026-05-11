using WaveguideQED
using QuantumOptics

ENV["MPLBACKEND"] = get(ENV, "MPLBACKEND", "Agg")
ENV["MPLCONFIGDIR"] = get(ENV, "MPLCONFIGDIR", joinpath(@__DIR__, ".mplconfig"))
mkpath(ENV["MPLCONFIGDIR"])
import PyPlot

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 14
rcParams["font.family"] = "serif"
rcParams["mathtext.fontset"] = "cm"
rcParams["axes.unicode_minus"] = false
try
    rcParams["text.usetex"] = true
catch
end


# ------------------------------------------------------------
# Simple single-photon scattering on a TLS with two waveguides
# Inspired by:
#   WaveguideQED/docs/src/multiplewaveguides_v2.md
#
# Waveguide 1: incoming/transmitted (right-moving) mode
# Waveguide 2: reflected (left-moving) mode
#
# The Hamiltonian is written in a rotating frame, so the photon
# carrier frequency does not appear explicitly.
# What matters is the detuning:
#   Delta = omega_TLS - omega_photon
# ------------------------------------------------------------


# -----------------
# User parameters
# -----------------
gamma = 3.14             # total decay rate into the two waveguides
Delta = gamma/2                # detuning = omega_TLS - omega_photon
sigma = 30* 1/gamma      # temporal width of incoming Gaussian pulse
t0 = 25                  # pulse center
dt = 2                  # time step
tmax = 50.0                # total simulation time
times = 0.0:dt:(tmax - dt)


# -----------------
# Bases and operators
# -----------------
bw = WaveguideBasis(1, 2, times)
be = FockBasis(1)

a = destroy(be)
ad = create(be)
n_tls = number(be) ⊗ identityoperator(bw)

w1 = destroy(bw, 1)
wd1 = create(bw, 1)
w2 = destroy(bw, 2)
wd2 = create(bw, 2)


# -----------------
# Hamiltonian
# -----------------
# In this frame:
# - Delta controls the atom-photon frequency mismatch
# - each propagation direction gets gamma/2
H =
    Delta * n_tls +
    im * sqrt(gamma / 2 / dt) * (ad ⊗ w1 - a ⊗ wd1) +
    im * sqrt(gamma / 2 / dt) * (ad ⊗ w2 - a ⊗ wd2)


# -----------------
# Incoming single-photon pulse
# -----------------
xi(t) = complex(sqrt(2 / sigma) * (log(2) / pi)^(1 / 4) * exp(-2 * log(2) * (t - t0)^2 / sigma^2))

# Incoming photon in waveguide 1, TLS initially in ground state
psi_in = fockstate(be, 0) ⊗ onephoton(bw, 1, xi; norm = true)


# -----------------
# Time evolution
# -----------------
# We store:
# - TLS excited-state population
# - photon population in waveguide 1
# - photon population in waveguide 2
fout(t, psi) = (
    real(expect(n_tls, psi)),
    sum(abs2, OnePhotonView(psi, 1, [1, :])),
    sum(abs2, OnePhotonView(psi, 2, [1, :])),
)

psi_out, tls_pop, wg1_pop, wg2_pop = waveguide_evolution(times, psi_in, H; fout = fout)


# -----------------
# Diagnostics
# -----------------
println("Simulation parameters")
println("gamma = $gamma")
println("Delta = $Delta")
println("sigma = $sigma")
println("t0    = $t0")
println("dt    = $dt")
println("tmax  = $tmax")
println()

println("Final populations")
println("waveguide 1 = $(wg1_pop[end])")
println("waveguide 2 = $(wg2_pop[end])")
println("TLS         = $(tls_pop[end])")
println("total       = $(wg1_pop[end] + wg2_pop[end] + tls_pop[end])")


# -----------------
# Plot
# -----------------
fig, ax = PyPlot.subplots(figsize = (3.4, 3.4), dpi = 300)

step_marker = max(1, Int(round(1 / dt / 10)))

ax.plot(times, wg1_pop, "-", color = "#1d3a5c", lw = 1.1, label = raw"$\mathcal{T}_n$")
ax.plot(times, wg2_pop, "-", color = "#c14b82", lw = 1.1, label = raw"$\mathcal{R}_n$")
ax.plot(times, tls_pop, "-", color = "#ffa600", lw = 1.1, label = raw"$\mathcal{A}_n$")

ax.plot(times[1:step_marker:end], wg1_pop[1:step_marker:end], "o", ms = 3, color = "#1d3a5c")
ax.plot(times[1:step_marker:end], wg2_pop[1:step_marker:end], "o", ms = 3, color = "#c14b82")
ax.plot(times[1:step_marker:end], tls_pop[1:step_marker:end], "o", ms = 3, color = "#ffa600")

ax.set_xlabel(raw"\textbf{Time (a.u.)}")
ax.set_ylabel(raw"\textbf{Probability}")
ax.grid(color = "0.9", linestyle = "-", linewidth = 0.4)
ax.legend(loc = "upper right", frameon = false)

for item in [ax.xaxis.label, ax.yaxis.label]
    item.set_fontsize(15)
end

for item in (ax.get_xticklabels(), ax.get_yticklabels())
    for tick in item
        tick.set_fontsize(14)
    end
end

PyPlot.tight_layout()

pdf_path = joinpath(@__DIR__, "fig2_like_waveguideqed.pdf")
png_path = joinpath(@__DIR__, "fig2_like_waveguideqed.png")

PyPlot.savefig(pdf_path, bbox_inches = "tight")
#PyPlot.savefig(png_path, bbox_inches = "tight")
PyPlot.close(fig)

println()
println("Saved figures to:")
println(pdf_path)
println(png_path)
