# Simulation driver: time stepping through the lattice.

"""
    Settings(; kwargs...)

Numerical and physics settings of a simulation.

Time stepping
- `dt`: time step [s]. `max_steps`: maximum number of steps.
- `t0`: initial time [s] (the RF phases refer to t = 0).
- `z_stop`: stop when the bunch centroid passes this z [m] (default: end of lattice).

Cathode emission (active if `emission_steps > 0`)
- `emission_steps`, `emission_time` [s]: the emission is done in `emission_steps` steps
  of size `emission_time/emission_steps`.
- `emission_ekin` [eV]: kinetic energy used to convert the laser time profile to the
  length of the bunch behind the cathode.

Space charge
- `grid = (nx, ny, nz)`: mesh size (any size; FFTs are done with FFTW).
- `space_charge = true`: include space charge if the beam charge is nonzero.
- `sc_solver = :mesh` (integrated Green function) or `:nbody` (relativistic
  point-to-point with softening radius `nbody_r0` [m]).
- `image_charge`: include the cathode image charge while the centroid is at
  z < `image_cutoff` [m].
- `sc_3d_start = Inf`: z [m] after which the charge deposition is fully 3D; before, the
  azimuthally symmetric deposition of IMPACT-T is used (lower noise for round beams).
- `green_tol = 0.0`: relative tolerance for reusing the Green function between steps. With
  e.g. 0.02 the mesh spacings are snapped to cached values and the (expensive) Green
  function is recomputed only when the bunch dimensions change by more than 2 %.
- `merge_bins_at = Inf`: z [m] at which all energy bins are merged into one.

Losses
- `domain_radius`: particles with |x| or |y| beyond the smallest of this and the element
  apertures are lost. `domain_length`: particles beyond this z are lost.
- `aperture_mode = :particle`: each particle is tested against the apertures of the
  elements at its own z. `:bunch` reproduces IMPACT-T, which uses the smallest aperture
  of all elements spanned by the bunch for every particle.

Lattice
- `s_offset = 0`: lab z of the start of the Beamline.

Diagnostics
- `diag_interval = 5`: record beam statistics every `diag_interval` steps.
- `diag_fixed_z = true`: statistics projected to the centroid plane (else at fixed time).
- `diag_per_bin = false`: also record statistics of every bin separately (`sim.bin_stats`).
- `output_dir`, `write_output = true`: where output files go.
- `verbose = true`.
"""
Base.@kwdef mutable struct Settings
  dt::Float64
  max_steps::Int = 1_000_000_000
  t0::Float64 = 0.0
  z_stop::Float64 = Inf
  emission_steps::Int = 0
  emission_time::Float64 = 0.0
  emission_ekin::Float64 = 0.0
  grid::NTuple{3,Int} = (32, 32, 32)
  space_charge::Bool = true
  sc_solver::Symbol = :mesh
  nbody_r0::Float64 = 0.0
  image_charge::Bool = false
  image_cutoff::Float64 = Inf
  sc_3d_start::Float64 = Inf
  green_tol::Float64 = 0.0
  merge_bins_at::Float64 = Inf
  domain_radius::Float64 = Inf
  domain_length::Float64 = Inf
  diag_interval::Int = 5
  diag_fixed_z::Bool = true
  diag_per_bin::Bool = false
  output_dir::String = "."
  write_output::Bool = true
  verbose::Bool = true
  seed::Int = 1
  s_offset::Float64 = 0.0
  aperture_mode::Symbol = :particle
end

"""
    Simulation(lattice, bins, settings; events=[], wakes=[], comm=default_comm(), device=nothing)

A simulation. `lattice` is a Beamlines `Beamline` (or an already `CompiledLattice`),
`bins` a vector of `ParticleBin`s (the particles are distributed over MPI ranks if
`comm` has several ranks). `device` is a prototype array (e.g. `CUDA.zeros(1)`) to run on
a GPU.
"""
mutable struct Simulation
  lattice::CompiledLattice
  fields::FieldSet
  settings::Settings
  bins::Vector{ParticleBin}
  queues::Dict{DataType,EventQueue}
  wakes::Vector{WakeRegion}
  sc::SpaceChargeSolver
  stats::BeamStats
  comm::AbstractComm
  t::Float64
  dt::Float64
  step::Int
  distance::Float64
  dzz::Float64
  sc_on::Bool
  dt_fixed::Bool
  image_on::Bool
  round_aperture::Bool
  total_charge0::Float64
  device::Any
  rng::Random.AbstractRNG
  timers::Dict{Symbol,Float64}
  done::Bool
  done_dipoles::Vector{Int}
  bend_stats::BeamStats
  bend_ref::Vector{NTuple{7,Float64}}
  bin_stats::Vector{BeamStats}
  aperture_cache::Tuple{Vector{ApertureSpec},Any}   # last aperture list and its device copy
end

function Simulation(lat, bins::AbstractVector{<:ParticleBin}, settings::Settings;
                    events=Event[], wakes=WakeRegion[], comm::AbstractComm=default_comm(),
                    device=nothing)
  cl = lat isa CompiledLattice ? lat : compile_lattice(lat; s_offset=settings.s_offset)
  proto = isnothing(device) ? zeros(1) : device
  fs = isnothing(device) ? cl.fields : to_device(cl.fields, device)
  bins_local = [scatter_bin(b, comm) for b in bins]
  if !isnothing(device)
    bins_local = [to_device(b, device) for b in bins_local]
  end
  allevents = vcat(Vector{Event}(collect(events)), Vector{Event}(cl.events))
  sc = SpaceChargeSolver(settings.grid...; green_tol=settings.green_tol, proto=proto)
  q0 = sum(abs(total_charge(b, comm)) for b in bins_local; init=0.0)
  rng = Random.Xoshiro(settings.seed + 7919 * rank(comm))
  sim = Simulation(cl, fs, settings, bins_local, event_queues(allevents),
                   vcat(WakeRegion[collect(wakes)...], cl.wakes), sc, BeamStats(), comm,
                   settings.t0, settings.dt, 0, 0.0, 0.0, true, false,
                   settings.image_charge && emission_on(settings), false, q0, device, rng,
                   Dict{Symbol,Float64}(), false, Int[], BeamStats(), NTuple{7,Float64}[],
                   [BeamStats() for _ in bins_local], (ApertureSpec[], nothing))
  return sim
end

function Base.show(io::IO, sim::Simulation)
  print(io, "Simulation(t = $(sim.t) s, step $(sim.step), <z> = $(sim.distance) m, ",
        "$(sum(b -> b.n, bin_summary.(sim.bins, Ref(sim.comm)); init=0)) particles in ",
        "$(length(sim.bins)) bin(s))")
end

macro timed_section(sim, name, ex)
  quote
    local t0 = time_ns()
    local v = $(esc(ex))
    local d = $(esc(sim)).timers
    d[$name] = get(d, $name, 0.0) + (time_ns() - t0) / 1e9
    v
  end
end

emission_on(s::Settings) = s.emission_steps > 0

"βz used by the emission model (from the emission kinetic energy)."
function emission_beta(sim::Simulation)
  mc2 = first(sim.bins).mc2
  γ = 1 + sim.settings.emission_ekin / mc2
  return sqrt(1 - 1 / γ^2)
end

# ---------------------------------------------------------------------------------------
# Losses

@kernel function _loss_ka!(state, @Const(r), rlim_x, rlim_y, zlen, round_ap::Bool, aps, nap::Int, xrad)
  p = @index(Global)
  @inbounds if state[p] == STATE_ALIVE
    x = r[p, 1]; y = r[p, 3]; z = r[p, 5]
    # smallest aperture of the elements containing the particle
    rad = xrad
    for a in 1:nap
      ap = aps[a]
      if z >= ap.zmin && z <= ap.zmax
        rad = min(rad, ap.radius)
      end
    end
    lost = false
    if round_ap
      lost = x * x + y * y >= rad * rad
    else
      lost = (abs(x) >= rad) | (abs(y) >= rad)
    end
    lost |= (z <= 0) & (r[p, 6] < 0)
    lost |= z >= zlen
    if lost
      state[p] = STATE_LOST
    end
  end
end

"Mark lost particles and remove them. Returns the number of particles lost on all ranks."
function apply_losses!(sim::Simulation, zlo, zhi)
  s = sim.settings
  aps = [a for a in sim.lattice.apertures if a.zmax >= zlo && a.zmin <= zhi]
  if s.aperture_mode == :bunch && !isempty(aps)
    # IMPACT-T: the smallest aperture of all elements spanned by the bunch applies to all
    aps = [ApertureSpec(-Inf, Inf, minimum(a -> a.radius, aps))]
  end
  if isnothing(sim.device)
    apsd = aps
  else   # upload the (small) aperture table only when it changes
    if aps != sim.aperture_cache[1] || isnothing(sim.aperture_cache[2])
      sim.aperture_cache = (aps, to_device_vec(aps, sim.device))
    end
    apsd = sim.aperture_cache[2]
  end
  nlost = 0
  for b in sim.bins
    N = size(b.r, 1)
    N == 0 && continue
    backend = get_backend(b.r)
    _loss_ka!(backend)(b.state, b.r, s.domain_radius, s.domain_radius, s.domain_length,
                       sim.round_aperture, apsd, length(aps), s.domain_radius; ndrange=N)
    _sync(backend)
    nlost += compact!(b)
  end
  return allreduce_sum(sim.comm, Float64(nlost))
end

to_device_vec(v::Vector, proto) = (A = Base.typename(typeof(proto)).wrapper; A(v))

@kernel function _collimate_ka!(state, @Const(r), c::Collimator)
  p = @index(Global)
  @inbounds begin
    x = r[p, 1]; y = r[p, 3]
    out = c.round ? (x^2 + y^2 >= c.xmax^2) :
          (x <= c.xmin || x >= c.xmax || y <= c.ymin || y >= c.ymax)
    out && (state[p] = STATE_LOST)
  end
end

function collimate!(sim::Simulation, c::Collimator)
  for b in sim.bins
    N = size(b.r, 1)
    N == 0 && continue
    backend = get_backend(b.r)
    _collimate_ka!(backend)(b.state, b.r, c; ndrange=N)
    _sync(backend)
    compact!(b)
  end
  sim.round_aperture = c.round
end

# ---------------------------------------------------------------------------------------
# Instantaneous beam manipulations

function apply_event!(sim::Simulation, e::Steer)
  for b in sim.bins, c in 1:6
    view(b.r, :, c) .+= e.dr[c]
  end
end

@kernel function _linear_map_ka!(r, R::SMatrix{6,6,Float64}, m::NTuple{6,Float64})
  p = @index(Global)
  @inbounds begin
    u = SVector(ntuple(j -> r[p, j] - m[j], Val(6)))
    v = R * u
    for j in 1:6
      r[p, j] = v[j] + m[j]
    end
  end
end

function apply_event!(sim::Simulation, e::LinearMap)
  for b in sim.bins
    s = bin_summary(b, sim.comm)
    N = size(b.r, 1)
    N == 0 && continue
    backend = get_backend(b.r)
    _linear_map_ka!(backend)(b.r, SMatrix{6,6,Float64}(e.R), s.mean; ndrange=N)
    _sync(backend)
  end
end

function apply_event!(sim::Simulation, e::Heat)
  for b in sim.bins
    if b.r isa Array
      noise = randn(sim.rng, size(b.r, 1))
    else   # random numbers generated on the device
      noise = randn!(similar(b.w))
    end
    view(b.r, :, 6) .+= e.sigma .* noise
  end
end

function apply_event!(sim::Simulation, e::RotateZ)
  c, s = cos(e.angle), sin(e.angle)
  for b in sim.bins
    x = b.r[:, 1]; px = b.r[:, 2]; y = b.r[:, 3]; py = b.r[:, 4]
    b.r[:, 1] .= x .* c .+ y .* s
    b.r[:, 2] .= px .* c .+ py .* s
    b.r[:, 3] .= -x .* s .+ y .* c
    b.r[:, 4] .= -px .* s .+ py .* c
  end
end


function apply_event!(sim::Simulation, e::ParticleOutput)
  name = isempty(e.name) ? @sprintf("particles_z%.6f.h5", e.z) : e.name
  write_particles(joinpath(sim.settings.output_dir, name), sim; every=e.every)
end

function apply_event!(sim::Simulation, e::SliceOutput)
  name = isempty(e.name) ? @sprintf("slices_z%.6f.h5", e.z) : e.name
  write_slices(joinpath(sim.settings.output_dir, name), sim, e.nslices)
end

apply_event!(sim::Simulation, e::Checkpoint) = write_checkpoint(joinpath(sim.settings.output_dir, e.name), sim)

function merge_all_bins!(sim::Simulation)
  length(sim.bins) == 1 && return
  sim.bins = [merge_bins(sim.bins)]
end

# ---------------------------------------------------------------------------------------
# Space charge for all bins

"""
Compute space charge (+ wakes) on the grid for all bins. `lo`, `hi` is the global box and
`summ` the per-bin summaries.
"""
function compute_space_charge!(sim::Simulation, lo, hi, summ, cathode_active::Bool, wakes)
  s = sim.settings
  sc = sim.sc
  nb = length(sim.bins)
  γ1 = sqrt(1 + summ[1].mean[6]^2)
  sim.distance > s.image_cutoff && (sim.image_on = false)   # permanently off
  image = sim.image_on
  set_geometry!(sc, lo, hi, γ1, nb == 1 && !image)
  reset_fields!(sc)
  azimuthal = sim.distance < s.sc_3d_start
  flagpos = cathode_active    # deposit only z > 0 particles
  for (ib, b) in enumerate(sim.bins)
    summ[ib].n == 0 && continue
    b.charge == 0 && continue
    γ = sqrt(1 + summ[ib].mean[6]^2)
    βsign = summ[ib].mean[6] > 0 ? 1.0 : -1.0
    @timed_section sim :deposit begin
      if azimuthal
        deposit_azimuthal!(sc, b, true, sim.comm)
      else
        deposit!(sc, b, flagpos)
        allreduce_sum!(sim.comm, sc.rho)
      end
    end
    if sim.sc_on
      @timed_section sim :poisson solve_bin!(sc, ib, γ, βsign, image)
    end
    for wake in wakes
      if wake isa DielectricWake
        dielectric_wake!(sc, wake, γ)
      else
        g = sc.geom
        λ, xs, ys = line_moments(b, g.zmin, g.hz, g.nz, flagpos, sim.comm)
        ex, ey, ez = wake_fields(wake, λ, xs, ys, g.hz)
        add_slice_fields!(sc, ex, ey, ez)
      end
    end
  end
end

function record_all_stats!(sim::Simulation)
  s = sim.settings
  record_stats!(sim.stats, sim.t, sim.bins, sim.comm; at_fixed_z=s.diag_fixed_z)
  if s.diag_per_bin
    for (ib, b) in enumerate(sim.bins)
      ib > length(sim.bin_stats) && push!(sim.bin_stats, BeamStats())
      record_stats!(sim.bin_stats[ib], sim.t, [b], sim.comm; at_fixed_z=s.diag_fixed_z)
    end
  end
end

# ---------------------------------------------------------------------------------------
# Main loop

"""
    run!(sim; nsteps=typemax(Int))

Track the beam until it leaves the lattice, `z_stop` is reached, all particles are lost or
the maximum number of steps is done.
"""
function run!(sim::Simulation; nsteps::Int=typemax(Int))
  s = sim.settings
  comm = sim.comm
  s.write_output && isroot(comm) && mkpath(s.output_dir)
  βini = emission_beta(sim)
  zend = min(sim.lattice.s_end, s.z_stop)
  if sim.step == 0
    record_all_stats!(sim)
    if s.write_output
      write_particles(joinpath(s.output_dir, "particles_initial.h5"), sim)
    end
    sim.dzz = βini * C_LIGHT_SI * sim.dt
  end
  cathode = emission_on(s)
  step_end = sim.step + nsteps
  twall = time()
  while sim.step < min(s.max_steps, step_end) && !sim.done
    sim.step += 1
    i = sim.step
    # --- events before the step
    for T in (Steer, LinearMap, Heat, RotateZ)
      q = get(sim.queues, T, nothing)
      isnothing(q) && continue
      e = trigger!(q, sim.distance, sim.dzz)
      isnothing(e) || apply_event!(sim, e)
    end
    # time step
    if i <= s.emission_steps
      sim.dt = s.emission_time / s.emission_steps
    elseif !sim.dt_fixed
      sim.dt = s.dt
    end
    q = get(sim.queues, TimeStepChange, nothing)
    if !isnothing(q)
      e = trigger!(q, sim.distance, sim.dzz)
      if !isnothing(e)
        sim.dt = e.dt
        sim.dt_fixed = true
      end
    end
    q = get(sim.queues, SpaceChargeSwitch, nothing)
    if !isnothing(q)
      e = trigger!(q, sim.distance, sim.dzz)
      isnothing(e) || (sim.sc_on = e.on)
    end
    # active wakes: the first structure wake region containing the centroid (IMPACT-T)
    # and all dielectric wake regions containing it
    wakes = WakeModel[]
    for w in sim.wakes
      (sim.distance >= w.zstart && sim.distance <= w.zend) || continue
      if w.model isa DielectricWake
        push!(wakes, w.model)
      elseif !any(x -> !(x isa DielectricWake), wakes)
        push!(wakes, w.model)
      end
    end
    use_bz = any(x -> x isa DielectricWake, wakes)
    dt = sim.dt
    # --- first half drift
    @timed_section sim :drift for b in sim.bins
      drift!(b, dt / 2, cathode)
    end
    sim.t += dt / 2
    # --- bunch extent and centroid
    summ = [bin_summary(b, comm) for b in sim.bins]
    ntot = sum(x -> x.n, summ)
    if ntot < 1
      s.verbose && isroot(comm) && @warn "All particles lost"
      sim.done = true
      break
    end
    lo = [Inf, Inf, Inf]; hi = [-Inf, -Inf, -Inf]
    zsum = 0.0; pzsum = 0.0
    for x in summ
      x.n == 0 && continue
      zlo = x.lo[3] > 0 ? x.lo[3] : (x.hi[3] > 0 ? 0.0 : x.lo[3])
      lo .= min.(lo, (x.lo[1], x.lo[2], zlo))
      hi .= max.(hi, x.hi)
      zsum += x.mean[5] * x.n; pzsum += x.mean[6] * x.n
    end
    zcent = zsum / ntot
    γavg = sqrt(1 + (pzsum / ntot)^2)
    cathode = emission_on(s) && lo[3] <= 0
    sim.distance = zcent
    sim.dzz = sqrt(1 - 1 / γavg^2) * C_LIGHT_SI * dt
    if zcent > zend
      sim.done = true
      break
    end
    # --- bend magnets: switch to bend mode when the bunch head enters a dipole region
    dip = entering_dipole(sim, hi[3])
    if !isnothing(dip)
      track_bend!(sim, dip)
      continue
    end
    # --- losses
    @timed_section sim :losses begin
      q = get(sim.queues, Collimator, nothing)
      if !isnothing(q)
        e = trigger!(q, sim.distance, sim.dzz)
        isnothing(e) || collimate!(sim, e)
      end
      apply_losses!(sim, lo[3], hi[3])
    end
    # --- fields and kick
    ranges = active_ranges(sim.lattice.fields, lo[3], hi[3])   # host copy (same ordering)
    if hi[3] > 0
      use_sc = sim.total_charge0 > 0 && s.space_charge
      if use_sc && s.sc_solver == :mesh
        @timed_section sim :spacecharge compute_space_charge!(sim, lo, hi, summ, cathode, wakes)
        @timed_section sim :kick for b in sim.bins
          kick!(b, sim.t, dt, sim.fields, ranges, sim.sc, true, true, use_bz)
        end
      elseif use_sc && s.sc_solver == :nbody
        @timed_section sim :kick for b in sim.bins
          kick_nbody!(sim, b, dt, ranges)
        end
      else
        @timed_section sim :kick for b in sim.bins
          kick!(b, sim.t, dt, sim.fields, ranges, sim.sc, false, false)
        end
      end
    end
    # --- second half drift (+ emission)
    @timed_section sim :drift for b in sim.bins
      drift!(b, dt / 2, cathode)
      cathode && emission_step!(b, dt, βini)
    end
    sim.t += dt / 2
    # --- diagnostics
    if i % s.diag_interval == 0
      @timed_section sim :diagnostics record_all_stats!(sim)
      if s.verbose && isroot(comm) && i % (20s.diag_interval) == 0
        @printf("step %8d  t = %.6e s  <z> = %.6f m  <E> = %.6f MeV  N = %d\n", i, sim.t,
                zcent, last(sim.stats.ekin) / 1e6, ntot)
      end
    end
    # --- events at the end of the step
    for T in (ParticleOutput, SliceOutput, Checkpoint)
      q = get(sim.queues, T, nothing)
      isnothing(q) && continue
      e = trigger!(q, sim.distance, sim.dzz)
      isnothing(e) || apply_event!(sim, e)
    end
    if sim.distance <= s.merge_bins_at && sim.distance + sim.dzz >= s.merge_bins_at
      merge_all_bins!(sim)
    end
  end
  sim.timers[:wall] = get(sim.timers, :wall, 0.0) + time() - twall
  if s.write_output
    write_particles(joinpath(s.output_dir, "particles_final.h5"), sim)
    write_stats(joinpath(s.output_dir, "stats.h5"), sim)
  end
  return sim
end
