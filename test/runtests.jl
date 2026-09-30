using ImpactTJ
using ImpactTJ: igf!, SpaceChargeSolver, set_geometry!, solve_bin!, boris, V3, ext_field,
                FourierSeries, fourier_eval, BufferBuilder, fourier_axis_table, axis_eval,
                csr_wake, compile_lattice, gather_sc, generate, rms, emittance
using Beamlines
using Test, Random, LinearAlgebra

const DATA = joinpath(@__DIR__, "data")

std(v) = sqrt(sum((v .- sum(v) / length(v)) .^ 2) / length(v))
# error function without SpecialFunctions (Abramowitz-Stegun 7.1.26 is not accurate enough)
function erf_(x)
    x < 0 && return -erf_(-x)
  if x < 2.5
    s = 0.0; t = x; n = 0
    while abs(t) > 1e-17
      s += t / (2n + 1)
      n += 1
      t *= -x^2 / n
    end
    return 2 / sqrt(π) * s
  else
    # continued fraction for erfc
    f = 0.0
    for k in 60:-1:1
      f = k / 2 / (x + f)
    end
    return 1 - exp(-x^2) / sqrt(π) / (x + f)
  end
end


"Values of `f` (a vector over the statistics `st`) at the times of the reference `ref`."
function at_t(st, ref, f)
  [f[argmin(abs.(st.t .- t))] for t in ref.t]
end
"Maximum deviation relative to the maximum of the reference."
relerr(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))

@testset "ImpactTJ" begin

@testset "constants" begin
  @test ImpactTJ.C_LIGHT_SI == 299792458.0
  @test isapprox(ImpactTJ.EPS0_SI, 8.8541878188e-12; rtol=1e-9)
  @test isapprox(ImpactTJ.MU0_SI * ImpactTJ.EPS0_SI * ImpactTJ.C_LIGHT_SI^2, 1.0; rtol=1e-14)
end

@testset "on-axis tables" begin
  fs = FourierSeries([1.0, 0.3, -0.2, 0.1, 0.05, -0.02, 0.01], 0.2, 0.1)
  bb = BufferBuilder()
  tab = fourier_axis_table(bb, fs, 0.0, 0.2)
  err = 0.0
  for z in range(0.001, 0.199, length=997)
    a = axis_eval(bb.data, tab, z); b = fourier_eval(fs, z)
    err = max(err, abs(a[1] - b[1]) / 1.0, abs(a[2] - b[2]) / 50, abs(a[3] - b[3]) / 2500)
  end
  @test err < 1e-7
  @test all(axis_eval(bb.data, tab, 0.3) .== 0)
end

@testset "field models satisfy Maxwell's equations" begin
  # RF cavity from on-axis field: ∇⋅E = 0 and ∇×E = -∂B/∂t to O(r²) near the axis
  coefs = [0.2, 1.0, 0.3, -0.4, 0.1, 0.05, 0.02]
  cav = RFCavity(L=0.1, rf_frequency=1.3e9, phi0=0.4,
                 field_map=OnAxisFourier(coefs; zstart=0.0, zend=0.1), field_scale=1e7)
  sol = Solenoid(L=0.1, field_map=OnAxisFourier(coefs; zstart=0.0, zend=0.1), field_scale=0.2)
  for ele in (cav, sol)
    cl = compile_lattice(Beamline([Drift(L=0.1), ele]))
    fs = cl.fields
    h = 1e-6; t = 0.3e-9
    x, y, z = 3e-4, -2e-4, 0.15
    E(x, y, z, t) = ext_field(fs, x, y, z, t)[1]
    B(x, y, z, t) = ext_field(fs, x, y, z, t)[2]
    divE = (E(x + h, y, z, t)[1] - E(x - h, y, z, t)[1] + E(x, y + h, z, t)[2] - E(x, y - h, z, t)[2] +
            E(x, y, z + h, t)[3] - E(x, y, z - h, t)[3]) / 2h
    divB = (B(x + h, y, z, t)[1] - B(x - h, y, z, t)[1] + B(x, y + h, z, t)[2] - B(x, y - h, z, t)[2] +
            B(x, y, z + h, t)[3] - B(x, y, z - h, t)[3]) / 2h
    scaleE = norm(E(x, y, z, t)) / 0.01 + 1e-30
    scaleB = norm(B(x, y, z, t)) / 0.01 + 1e-30
    @test abs(divE) < 1e-3 * scaleE + 1e-9
    @test abs(divB) < 1e-3 * scaleB + 1e-15
    # Faraday: (∇×E)_z = -∂Bz/∂t (Bz = 0 for TM), (∇×E)_x = -∂Bx/∂t
    dt = 1e-14
    curlEx = (E(x, y + h, z, t)[3] - E(x, y - h, z, t)[3]) / 2h - (E(x, y, z + h, t)[2] - E(x, y, z - h, t)[2]) / 2h
    dBxdt = (B(x, y, z, t + dt)[1] - B(x, y, z, t - dt)[1]) / 2dt
    @test abs(curlEx + dBxdt) < 1e-3 * (abs(dBxdt) + abs(curlEx)) + 1e-6 * scaleE
  end
  # quadrupole with Enge fringe: ∇×B = 0
  q = Quadrupole(L=0.2, Bn1=10.0, quad_profile=:enge, quad_leff=0.15, x2_limit=0.01, x1_limit=-0.01,
                 y1_limit=-0.01, y2_limit=0.01)
  fs = compile_lattice(Beamline([Drift(L=0.1), q])).fields
  B(x, y, z) = ext_field(fs, x, y, z, 0.0)[2]
  x, y, z, h = 2e-3, 1e-3, 0.12, 1e-6
  curlz = (B(x + h, y, z)[2] - B(x - h, y, z)[2] - B(x, y + h, z)[1] + B(x, y - h, z)[1]) / 2h
  curlx = (B(x, y + h, z)[3] - B(x, y - h, z)[3] - B(x, y, z + h)[2] + B(x, y, z - h)[2]) / 2h
  @test abs(curlz) < 1e-6 * 10
  @test abs(curlx) < 1e-3 * 10
  # skew quadrupole: Bs1 gives Bx = Bs1 x, By = -Bs1 y
  sq = Quadrupole(L=0.2, Bs1=5.0)
  fs = compile_lattice(Beamline([sq])).fields
  Bq = ext_field(fs, 1e-3, 2e-3, 0.1, 0.0)[2]
  @test Bq[1] ≈ 5.0 * 1e-3 atol = 1e-12
  @test Bq[2] ≈ -5.0 * 2e-3 atol = 1e-12
end

@testset "integrated Green function" begin
  for (hx, hy, hz) in ((1.0, 1.0, 1.0), (1.0, 2.0, 0.3), (0.1, 0.1, 3.0))
    G1 = zeros(20, 20, 20); G2 = zeros(20, 20, 20)
    igf!(G1, hx, hy, hz, 0.0; far=1e9)
    igf!(G2, hx, hy, hz, 0.0; far=5.0)
    @test maximum(abs.(G1 .- G2) ./ abs.(G1)) < 1e-6
  end
end

@testset "Poisson solver: Gaussian bunch (second order convergence)" begin
  σ = 1e-3; Q = 1e-9; L = 5σ
  Er(r) = Q * ImpactTJ.COULOMB_K / r^2 * (erf_(r / (sqrt(2) * σ)) - sqrt(2 / π) * r / σ * exp(-r^2 / (2σ^2)))
  function maxerr(n)
    sc = SpaceChargeSolver(n, n, n)
    set_geometry!(sc, (-L, -L, -L), (L, L, L), 1.0, false)
    g = sc.geom
    xs = [g.xmin + (i - 1) * g.hx for i in 1:n]
    ρ = [exp(-(x^2 + y^2 + z^2) / (2σ^2)) for x in xs, y in xs, z in xs]
    sc.rho .= ρ .* (Q / sum(ρ))
    ImpactTJ.reset_fields!(sc)
    solve_bin!(sc, 1, 1.0, 1.0, false)
    maximum(abs(gather_sc(sc.F, g, r, 0.0, 0.0)[1][1] - Er(r)) for r in (0.5σ, σ, 1.5σ, 2σ, 3σ)) / Er(1.5σ)
  end
  e1 = maxerr(32); e2 = maxerr(64)
  @test e2 < 0.015
  @test e1 / e2 > 3.5      # O(h²)
end

@testset "pruned FFT and device Green function paths agree" begin
  n = (16, 24, 20)
  sc = SpaceChargeSolver(n...)
  set_geometry!(sc, (-1e-3, -2e-3, 1e-3), (1e-3, 1e-3, 3e-3), 2.0, false)
  sc.rho .= rand(n...)
  solve_bin!(sc, 1, 2.0, 1.0, true)
  φ1 = copy(sc.phi); φi1 = copy(sc.phiimg)
  sc2 = SpaceChargeSolver(n...); sc2.geom = sc.geom; sc2.rho .= sc.rho
  sc2.pruned = nothing
  sc2.plan_f = ImpactTJ.FFTW.plan_rfft(sc2.pad); sc2.plan_b = ImpactTJ.FFTW.plan_brfft(sc2.prod, 2n[1])
  hx, hy, hzp = sc.geom.hx, sc.geom.hy, 2 * sc.geom.hz
  ImpactTJ._forward!(sc2); ImpactTJ._mul_green_generic!(sc2, 1, hx, hy, hzp); ImpactTJ._inverse!(sc2, sc2.phi)
  ImpactTJ._forward!(sc2); ImpactTJ._mul_image_generic!(sc2, hx, hy, hzp, 2 * sc.geom.zmin)
  ImpactTJ._inverse!(sc2, sc2.phiimg)
  @test maximum(abs.(φ1 .- sc2.phi)) / maximum(abs.(φ1)) < 1e-8
  @test maximum(abs.(φi1 .- sc2.phiimg)) / maximum(abs.(φi1)) < 1e-8
end

@testset "Boris push" begin
  # uniform magnetic field: gyration with exact period and radius (Boris is exact in |p|)
  B = V3(0.0, 0.0, 1.0); E = V3(0.0, 0.0, 0.0)
  qm = -1 / 510998.95
  p = (10.0, 0.0, 0.0)
  γ = sqrt(1 + 100.0)
  ωc = abs(qm) * ImpactTJ.C_LIGHT_SI^2 * 1.0 / γ
  N = 1000
  dt = 2π / ωc / N
  for _ in 1:N
    p = boris(p..., E, B, qm * ImpactTJ.C_LIGHT_SI * dt)
  end
  @test hypot(p[1], p[2]) ≈ 10.0 rtol = 1e-12
  @test p[1] ≈ 10.0 rtol = 1e-4
  # uniform electric field: γβz gain = q E t /(m c)
  p = (0.0, 0.0, 0.0)
  E = V3(0.0, 0.0, -1e6)
  for _ in 1:100
    p = boris(p..., E, V3(0, 0, 0), qm * ImpactTJ.C_LIGHT_SI * 1e-12)
  end
  @test p[3] ≈ -qm * 1e6 * ImpactTJ.C_LIGHT_SI * 1e-10 rtol = 1e-12
end

@testset "distributions" begin
  rng = Xoshiro(1)
  d = BeamDistribution(:gauss; x=PhasePlane(sigma=1e-3, sigma_p=2e-3), y=PhasePlane(sigma=2e-3, sigma_p=1e-3),
                       z=PhasePlane(sigma=1e-4, sigma_p=0.01, p_offset=10.0))
  r, _ = generate(d, 200_000, rng)
  @test std(r[:, 1]) ≈ 1e-3 rtol = 0.01
  @test std(r[:, 4]) ≈ 1e-3 rtol = 0.01
  @test sum(r[:, 6]) / size(r, 1) ≈ 10.0 rtol = 1e-3
  for kind in (:uniform, :waterbag, :kv, :semigauss)
    r, _ = generate(BeamDistribution(kind; x=PhasePlane(sigma=1e-3, sigma_p=1e-3), y=PhasePlane(sigma=1e-3, sigma_p=1e-3),
                                     z=PhasePlane(sigma=1e-3, sigma_p=1e-3)), 100_000, rng)
    @test std(r[:, 1]) ≈ 1e-3 rtol = 0.03
  end
  # photoinjector flat top split into two energy bins: the bins tile the bunch
  d = BeamDistribution((:combine, 1, 1, 2); x=PhasePlane(sigma=1e-3), y=PhasePlane(sigma=1e-3),
                       z=PhasePlane(sigma=1e-3, scale=1e-4, sigma_p=1e-3))
  r1, f1 = generate(d, 10_000, Xoshiro(2); ib=1, nb=2)
  r2, f2 = generate(d, 10_000, Xoshiro(2); ib=2, nb=2)
  @test maximum(r2[:, 5]) <= minimum(r1[:, 5]) + 1e-12
  @test f1 + f2 ≈ 2 rtol = 0.05
end

@testset "CSR converges with the grid" begin
  R = 0.34; γ = 100.0; Lb = 0.1015
  Eavg(N, s0) = begin
    hz = 8e-3 / (N - 1); zmin = -4e-3
    zs = [zmin + (i - 1) * hz for i in 1:N]
    λ = 1e-9 / (sqrt(2π) * 1e-3) .* exp.(-0.5 .* (zs ./ 1e-3) .^ 2)
    ez = csr_wake(λ, zmin + s0, hz, R, γ, Lb)
    sum(λ .* ez) / sum(λ), maximum(abs, ez)
  end
  a, pa = Eavg(256, 0.03); b, pb = Eavg(1024, 0.03)
  @test abs(a - b) < 0.01 * pb
  @test abs(pa - pb) < 0.01 * pb
end

@testset "IMPACT-T translation round trips" begin
  for name in ("sample1", "sample4", "features", "chicane", "dwa_slab")
    inp = read_impactt(joinpath(DATA, name))
    cl1 = compile_lattice(inp.beamline; s_offset=inp.settings.s_offset, s_end=inp.s_end)
    # ImpactTJ → IMPACT-T → ImpactTJ
    dir = mktempdir()
    write_impactt(dir, inp)
    inp2 = read_impactt(dir)
    cl2 = compile_lattice(inp2.beamline; s_offset=inp2.settings.s_offset, s_end=inp2.s_end)
    # ImpactTJ → Julia script
    dir2 = mktempdir()
    f = write_impacttj(dir2, inp)
    code = replace(read(f, String), "run!(sim)" => "")
    m = Module(); Core.eval(m, :(using ImpactTJ, Beamlines))
    Base.include_string(m, code, f)
    cl3 = Core.eval(m, :sim).lattice
    rng = Xoshiro(5)
    for k in 1:500
      x, y = 2e-3 .* (rand(rng, 2) .- 0.5); z = cl1.s_end * rand(rng); t = 1e-9 * rand(rng)
      E1, B1 = ext_field(cl1.fields, x, y, z, t)
      for cl in (cl2, cl3)
        E2, B2 = ext_field(cl.fields, x, y, z, t)
        @test isapprox(E1, E2; rtol=1e-9, atol=1e-9 * (norm(E1) + 1))
        @test isapprox(B1, B2; rtol=1e-9, atol=1e-12)
      end
    end
  end
end

@testset "IO" begin
  inp = read_impactt(joinpath(DATA, "sample1"))
  f = joinpath(mktempdir(), "p.h5")
  write_particles(f, inp.bins, 1.5e-9)
  bins, t = read_particles(f)
  @test t == 1.5e-9
  @test bins[1].r ≈ inp.bins[1].r
  @test bins[1].w ≈ inp.bins[1].w
end

@testset "agreement with IMPACT-T (same initial particles)" begin
  # photoinjector: emission, image charge, gun + solenoids + linac, space charge
  inp = read_impactt(joinpath(DATA, "sample1"))
  sim = Simulation(inp); sim.settings.write_output = false; sim.settings.verbose = false
  run!(sim)
  ref = read_impactt_stats(joinpath(DATA, "sample1", "reference"))
  @test relerr(at_t(sim.stats, ref, sim.stats.ekin), ref.ekin) < 1e-3
  @test relerr(at_t(sim.stats, ref, rms(sim.stats, 1)), rms(ref, 1)) < 1e-3
  @test relerr(at_t(sim.stats, ref, emittance(sim.stats, 1)), emittance(ref, 1)) < 1e-2

  # magnets with fringe fields, misalignments, multipoles and beam manipulation events
  inp = read_impactt(joinpath(DATA, "features"))
  sim = Simulation(inp); sim.settings.write_output = false; sim.settings.verbose = false
  sim.settings.aperture_mode = :bunch
  run!(sim)
  ref = read_impactt_stats(joinpath(DATA, "features", "reference"))
  @test relerr(at_t(sim.stats, ref, rms(sim.stats, 1)), rms(ref, 1)) < 2e-2
  @test relerr(at_t(sim.stats, ref, rms(sim.stats, 3)), rms(ref, 3)) < 2e-2
  @test sim.stats.n_particle[end] == ref.n_particle[end]

  # chicane with space charge and CSR (bend mode). IMPACT-T's CSR integration has an
  # O(h) error at the bend entrances, so only the orbit and emittance are compared.
  inp = read_impactt(joinpath(DATA, "chicane"))
  sim = Simulation(inp); sim.settings.write_output = false; sim.settings.verbose = false
  run!(sim)
  ref = read_impactt_stats(joinpath(DATA, "chicane", "reference"))
  @test relerr(at_t(sim.stats, ref, rms(sim.stats, 1)), rms(ref, 1)) < 0.05
  @test relerr(at_t(sim.stats, ref, emittance(sim.stats, 1)), emittance(ref, 1)) < 0.2
  @test abs(sim.stats.ekin[end] - ref.ekin[end]) < 1e-4 * ref.ekin[end]

  # dielectric lined slab waveguide wakefield
  inp = read_impactt(joinpath(DATA, "dwa_slab"))
  sim = Simulation(inp); sim.settings.write_output = false; sim.settings.verbose = false
  run!(sim)
  ref = read_impactt_stats(joinpath(DATA, "dwa_slab", "reference"))
  @test relerr(at_t(sim.stats, ref, sim.stats.sigma_gamma), ref.sigma_gamma) < 1e-3
  @test relerr(at_t(sim.stats, ref, emittance(sim.stats, 2)), emittance(ref, 2)) < 1e-3
end

@testset "generic (GPU) kernels" begin
  jl = Base.find_package("JLArrays")
  if jl === nothing
    @info "JLArrays not available: skipping the generic backend test"
  else
    @eval using JLArrays
    inp = read_impactt(joinpath(DATA, "sample1"))
    cl = compile_lattice(inp.beamline; s_offset=inp.settings.s_offset)
    b = inp.bins[1]; b.r[:, 5] .= abs.(b.r[:, 5]) .+ 0.01
    proto = JLArrays.JLArray(zeros(1))
    bd = ImpactTJ.to_device(b, proto); fsd = ImpactTJ.to_device(cl.fields, proto)
    sc = SpaceChargeSolver(8, 8, 8); scd = SpaceChargeSolver(8, 8, 8; proto=proto)
    ImpactTJ.drift!(b, 1e-12, false); ImpactTJ.drift!(bd, 1e-12, false)
    ImpactTJ.kick!(b, 1e-10, 1e-12, cl.fields, ImpactTJ.all_ranges(cl.fields), sc, false, true)
    ImpactTJ.kick!(bd, 1e-10, 1e-12, fsd, ImpactTJ.all_ranges(fsd), scd, false, true)
    @test Array(bd.r) == b.r
    # complete simulation (no space charge; events, losses, collimation) on the device
    s1 = Simulation(read_impactt(joinpath(DATA, "features")))
    s2 = Simulation(read_impactt(joinpath(DATA, "features")); device=proto)
    for s in (s1, s2)
      s.settings.write_output = false; s.settings.verbose = false; s.settings.aperture_mode = :bunch
      s.settings.max_steps = 400
      run!(s)
    end
    @test s1.stats.n_particle == s2.stats.n_particle
    @test rms(s1.stats, 1) ≈ rms(s2.stats, 1) rtol = 1e-12
    # device reductions and compaction
    r = randn(1000, 6); st = rand([ImpactTJ.STATE_ALIVE, ImpactTJ.STATE_LOST], 1000)
    v = ImpactTJ._summary_local(JLArrays.JLArray(r), JLArrays.JLArray(st))
    vh = ImpactTJ._summary_local(r, st)
    @test all(isapprox.(v, vh; rtol=1e-12))
    bh = ParticleBin(ImpactTJ.Species("electron"), r, 1.0); bh.state .= st
    bd = ImpactTJ.to_device(bh, proto)
    ImpactTJ.compact!(bh); ImpactTJ.compact!(bd)
    @test Array(bd.r) == bh.r && Array(bd.id) == bh.id
    # space charge (azimuthal deposition, image charge, Green function reuse), structure
    # wake and collimation on the device: no data leaves the device during the run
    function s1run(dev)
      inp = read_impactt(joinpath(DATA, "sample1"))
      s = Simulation(inp.beamline, inp.bins, inp.settings; device=dev,
                     events=vcat(inp.events, [Collimator(z=0.05, xmin=-4e-4, xmax=4e-4, ymin=-4e-4, ymax=4e-4)]),
                     wakes=[WakeRegion(0.0, 10.0, TeslaWake())])
      s.settings.write_output = false; s.settings.verbose = false; s.settings.max_steps = 500
      s.settings.green_tol = 0.05
      run!(s)
    end
    a = s1run(nothing); d = s1run(proto)
    @test a.stats.n_particle == d.stats.n_particle
    @test rms(a.stats, 1) ≈ rms(d.stats, 1) rtol = 1e-7
    @test emittance(a.stats, 3) ≈ emittance(d.stats, 3) rtol = 1e-6
    # dielectric wake
    a = Simulation(read_impactt(joinpath(DATA, "dwa_slab"))); d = Simulation(read_impactt(joinpath(DATA, "dwa_slab")); device=proto)
    for s in (a, d)
      s.settings.write_output = false; s.settings.verbose = false; s.settings.max_steps = 100
      run!(s)
    end
    @test emittance(a.stats, 3) ≈ emittance(d.stats, 3) rtol = 1e-7
  end
end

@testset "MPI" begin
  if Base.find_package("MPI") === nothing || get(ENV, "IMPACTTJ_TEST_MPI", "true") == "false"
    @info "skipping MPI test"
  else
    @eval using MPI
    script = joinpath(@__DIR__, "mpi_test.jl")
    out = read(`$(MPI.mpiexec()) -n 2 $(Base.julia_cmd()) --project=$(Base.active_project()) $script`, String)
    @test occursin("MPI test passed", out)
  end
end

end

