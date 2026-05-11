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
# Reflection vs detuning for a single-photon scattering problem
# ------------------------------------------------------------
#
# Same simple model as in test.jl:
# - one TLS
# - two directional waveguides
# - input photon in waveguide 1
# - reflected photon measured in waveguide 2
#
# We repeat the scattering experiment for several detunings
# Delta = omega_TLS - omega_photon
# and store the final reflection probability.


# -----------------
# User parameters
# -----------------
gamma = 3.14             # total decay rate into the two waveguides
sigma = 30* 1/gamma      # temporal width of incoming Gaussian pulse
t0 = 25                  # pulse center
dt = 1/gamma        # time step
tmax = 50.0                # total simulation time

# Detuning sweep
detunings = range(-1 * gamma, 1 * gamma, length = 19)

times = 0.0:dt:(tmax - dt)


# -----------------
# Shared bases and operators
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

xi(t) = complex(sqrt(2 / sigma) * (log(2) / pi)^(1 / 4) * exp(-2 * log(2) * (t - t0)^2 / sigma^2))
psi_in = fockstate(be, 0) ⊗ onephoton(bw, 1, xi; norm = true)


function run_scattering(Delta, gamma, dt, psi_in, n_tls, a, ad, w1, wd1, w2, wd2)
    H =
        Delta * n_tls +
        im * sqrt(gamma / 2 / dt) * (ad ⊗ w1 - a ⊗ wd1) +
        im * sqrt(gamma / 2 / dt) * (ad ⊗ w2 - a ⊗ wd2)

    fout(t, psi) = (
        real(expect(n_tls, psi)),
        sum(abs2, OnePhotonView(psi, 1, [1, :])),
        sum(abs2, OnePhotonView(psi, 2, [1, :])),
    )

    _, tls_pop, wg1_pop, wg2_pop = waveguide_evolution(times, psi_in, H; fout = fout)
    return tls_pop[end], wg1_pop[end], wg2_pop[end]
end


# -----------------
# Sweep
# -----------------
final_tls = zeros(length(detunings))
final_transmission = zeros(length(detunings))
final_reflection = zeros(length(detunings))

#Test: renormalized decay rate
gamma_0 = gamma 
gamma_A = -2/dt * log(abs(cos(sqrt(gamma_0 * dt))))

# Eq.-style theoretical trend used in the preprint notebooks:
# R(Delta) = 1 / (1 + (Delta / (gamma/2))^2)
# Since Delta = omega_TLS - omega_photon, the sign convention does not matter here
# because the Lorentzian is even in Delta.
detuning_theory = range(first(detunings), last(detunings), length = 400)
reflection_theory = 1.0 ./ (1.0 .+ (detuning_theory ./ (gamma_A / 2)).^2)

for (i, Delta) in enumerate(detunings)
    println("Running scattering experiment $i / $(length(detunings)) with Delta/gamma = $(Delta / gamma)")


    tls_end, trans_end, refl_end =
        run_scattering(Delta, gamma_0, dt, psi_in, n_tls, a, ad, w1, wd1, w2, wd2)
    final_tls[i] = tls_end
    final_transmission[i] = trans_end
    final_reflection[i] = refl_end
end


# -----------------
# Save raw data
# -----------------
csv_path = joinpath(@__DIR__, "reflection_vs_detuning.csv")
open(csv_path, "w") do io
    println(io, "Delta,Delta_over_gamma,reflection,transmission,tls_population,total")
    for i in eachindex(detunings)
        total = final_reflection[i] + final_transmission[i] + final_tls[i]
        println(
            io,
            "$(detunings[i]),$(detunings[i] / gamma),$(final_reflection[i]),$(final_transmission[i]),$(final_tls[i]),$(total)",
        )
    end
end


# -----------------
# Plot
# -----------------
fig, ax = PyPlot.subplots(figsize = (3.6, 3.4), dpi = 300)

ax.scatter(
    detunings ./ gamma,
    final_reflection,
    color = "#E69F00",
    s = 22,
    label = raw"$\mathcal{R}$",
    zorder = 3,
)
ax.plot(
    detuning_theory ./ gamma,
    reflection_theory,
    color = "#3E4A89",
    lw = 1.2,
    alpha = 0.9,
    zorder = 1,
    label = raw"$\mathcal{R}^{(\rm th)}$",
)

ax.set_xlabel(raw"\textbf{Detuning} $\Delta / \gamma$")
ax.set_ylabel(raw"\textbf{Probability}")
ax.set_ylim(-0.05, 1.05)
ax.grid(color = "0.9", linestyle = "-", linewidth = 0.4)
ax.legend(loc = "lower center", frameon = false)

for item in [ax.xaxis.label, ax.yaxis.label]
    item.set_fontsize(15)
end

for item in (ax.get_xticklabels(), ax.get_yticklabels())
    for tick in item
        tick.set_fontsize(14)
    end
end

PyPlot.tight_layout()

pdf_path = joinpath(@__DIR__, "reflection_vs_detuning.pdf")
png_path = joinpath(@__DIR__, "reflection_vs_detuning.png")

PyPlot.savefig(pdf_path, bbox_inches = "tight")
#PyPlot.savefig(png_path, bbox_inches = "tight")
PyPlot.close(fig)


# -----------------
# Diagnostics
# -----------------
imax = argmax(final_reflection)
println("Sweep parameters")
println("gamma = $gamma")
println("gamma_0 = $gamma_0")
println("sigma = $sigma")
println("t0    = $t0")
println("dt    = $dt")
println("tmax  = $tmax")
println("number of detunings = $(length(detunings))")
println()
println("Maximum reflection = $(final_reflection[imax]) at Delta/gamma = $(detunings[imax] / gamma)")
println()
println("Saved data to:")
println(csv_path)
println()
println("Saved figures to:")
println(pdf_path)
println(png_path)
