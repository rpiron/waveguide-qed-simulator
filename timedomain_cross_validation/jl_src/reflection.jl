using WaveguideQED
using QuantumOptics

const PROJECT_DIR = normpath(joinpath(@__DIR__, ".."))

const DEFAULT_PARAMS = Dict{String,Any}(
    "gamma_a" => 3.14,
    "gamma_0" => 3.14,
    "sigma_factor" => 30.0,
    "t0" => 25.0,
    "dt" => 2,
    "tmax" => 50.0,
    "detuning_min_over_gamma" => -0.5,
    "detuning_max_over_gamma" => 0.5,
    "n_detunings" => 21,
    "output_dir" => joinpath("results", "csv"),
)

function usage()
    println("""
    Usage:
      julia --project=timedomain_cross_validation timedomain_cross_validation/jl_src/reflection.jl [options]

    Main options:
      --dt VALUE                         Time step used in the time-domain evolution.
      --gamma-a VALUE                    Analytical decay rate used for normalization.
      --gamma-0 VALUE                    Numerical decay rate used in the Hamiltonian.
      --n-detunings VALUE                Number of detuning points.
      --detuning-min-over-gamma VALUE    First detuning in units of gamma_A.
      --detuning-max-over-gamma VALUE    Last detuning in units of gamma_A.
      --output-dir PATH                  CSV output directory.

    Default output:
      results/csv/reflection_profile_jl_dt_<dt>.csv
    """)
end

function option_pair(args, i)
    arg = args[i]
    if occursin("=", arg)
        key, value = split(arg[3:end], "=", limit = 2)
        return replace(key, "-" => "_"), value, i
    end

    key = replace(arg[3:end], "-" => "_")
    i == length(args) && error("Missing value for option $arg")
    return key, args[i + 1], i + 1
end

function parse_cli_args(args)
    params = copy(DEFAULT_PARAMS)
    integer_options = Set(["n_detunings"])
    string_options = Set(["output_dir"])

    i = 1
    while i <= length(args)
        arg = args[i]

        if arg == "-h" || arg == "--help"
            usage()
            exit(0)
        end

        startswith(arg, "--") || error("Unknown positional argument: $arg")
        key, value, i = option_pair(args, i)
        haskey(params, key) || error("Unknown option: --$(replace(key, "_" => "-"))")

        if key in integer_options
            params[key] = parse(Int, value)
        elseif key in string_options
            params[key] = value
        else
            params[key] = parse(Float64, value)
        end

        i += 1
    end

    return params
end

function normalized_output_dir(path)
    isabspath(path) && return normpath(path)
    return normpath(joinpath(PROJECT_DIR, path))
end

function dt_label(dt)
    return replace(string(dt), "/" => "_over_", " " => "")
end

function build_single_photon_problem(times, gamma_A, sigma_factor, t0)
    bw = WaveguideBasis(1, 2, times)
    be = FockBasis(1)

    a = destroy(be)
    ad = create(be)
    n_tls = number(be) ⊗ identityoperator(bw)

    w1 = destroy(bw, 1)
    wd1 = create(bw, 1)
    w2 = destroy(bw, 2)
    wd2 = create(bw, 2)

    sigma = sigma_factor / gamma_A
    xi(t) = complex(
        sqrt(2 / sigma) *
        (log(2) / pi)^(1 / 4) *
        exp(-2 * log(2) * (t - t0)^2 / sigma^2),
    )

    psi_in = fockstate(be, 0) ⊗ onephoton(bw, 1, xi; norm = true)

    return (; psi_in, n_tls, a, ad, w1, wd1, w2, wd2, sigma)
end

function run_scattering(Delta, gamma, dt, times, problem)
    H =
        Delta * problem.n_tls +
        im * sqrt(gamma / 2 / dt) * (problem.ad ⊗ problem.w1 - problem.a ⊗ problem.wd1) +
        im * sqrt(gamma / 2 / dt) * (problem.ad ⊗ problem.w2 - problem.a ⊗ problem.wd2)

    fout(t, psi) = (
        real(expect(problem.n_tls, psi)),
        sum(abs2, OnePhotonView(psi, 1, [1, :])),
        sum(abs2, OnePhotonView(psi, 2, [1, :])),
    )

    _, tls_pop, wg1_pop, wg2_pop = waveguide_evolution(times, problem.psi_in, H; fout = fout)
    return tls_pop[end], wg1_pop[end], wg2_pop[end]
end

function write_reflection_profile(csv_path, rows)
    open(csv_path, "w") do io
        println(io, "Delta,Delta_over_gamma,reflection,transmission,tls_population,total,dt,gamma_A,gamma_0,sigma,t0,tmax")
        for row in rows
            println(
                io,
                join(
                    (
                        row.Delta,
                        row.Delta_over_gamma,
                        row.reflection,
                        row.transmission,
                        row.tls_population,
                        row.total,
                        row.dt,
                        row.gamma_A,
                        row.gamma_0,
                        row.sigma,
                        row.t0,
                        row.tmax,
                    ),
                    ",",
                ),
            )
        end
    end
end

function main(args)
    params = parse_cli_args(args)

    gamma_A = params["gamma_a"]
    gamma_0 = params["gamma_0"]
    sigma_factor = params["sigma_factor"]
    t0 = params["t0"]
    dt = params["dt"]
    tmax = params["tmax"]
    n_detunings = params["n_detunings"]

    dt > 0 || error("dt must be positive.")
    tmax > dt || error("tmax must be larger than dt.")
    n_detunings >= 2 || error("n_detunings must be at least 2.")

    times = 0.0:dt:(tmax - dt)
    problem = build_single_photon_problem(times, gamma_A, sigma_factor, t0)

    detunings = range(
        params["detuning_min_over_gamma"] * gamma_A,
        params["detuning_max_over_gamma"] * gamma_A,
        length = n_detunings,
    )

    rows = Vector{NamedTuple}(undef, length(detunings))

    for (i, Delta) in enumerate(detunings)
        println("Running scattering experiment $i / $(length(detunings)) with Delta/gamma_A = $(Delta / gamma_A)")
        tls_end, trans_end, refl_end = run_scattering(Delta, gamma_0, dt, times, problem)
        total = refl_end + trans_end + tls_end

        rows[i] = (
            Delta = Delta,
            Delta_over_gamma = Delta / gamma_A,
            reflection = refl_end,
            transmission = trans_end,
            tls_population = tls_end,
            total = total,
            dt = dt,
            gamma_A = gamma_A,
            gamma_0 = gamma_0,
            sigma = problem.sigma,
            t0 = t0,
            tmax = tmax,
        )
    end

    output_dir = normalized_output_dir(params["output_dir"])
    mkpath(output_dir)
    csv_path = joinpath(output_dir, "reflection_profile_jl_dt_$(dt_label(dt)).csv")
    write_reflection_profile(csv_path, rows)

    imax = argmax([row.reflection for row in rows])

    println()
    println("Sweep parameters")
    println("gamma_A = $gamma_A")
    println("gamma_0 = $gamma_0")
    println("sigma = $(problem.sigma)")
    println("t0    = $t0")
    println("dt    = $dt")
    println("tmax  = $tmax")
    println("number of detunings = $(length(detunings))")
    println()
    println("Maximum reflection = $(rows[imax].reflection) at Delta/gamma_A = $(rows[imax].Delta_over_gamma)")
    println()
    println("Saved data to:")
    println(csv_path)
end

main(ARGS)
