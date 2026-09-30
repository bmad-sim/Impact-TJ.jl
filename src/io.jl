# Output and input files.
#
# Particles: openPMD with the BeamPhysics extension (HDF5), readable by e.g. the Python
# package openPMD-beamphysics (`ParticleGroup(h5=...)`).
# Statistics: one HDF5 file with a dataset per quantity.

const EV_C_UNIT_SI = E_CHARGE_SI / C_LIGHT_SI   # 1 eV/c in kg m/s

function _pmd_root!(f)
  a = HDF5.attributes(f)
  a["openPMD"] = "2.0.0"
  a["openPMDextension"] = "BeamPhysics;SpeciesType"
  a["basePath"] = "/data/%T/"
  a["particlesPath"] = "particles/"
  a["software"] = "ImpactTJ.jl"
end

function _dset!(g, name, v, unitSI, dim)
  g[name] = v
  a = HDF5.attributes(g[name])
  a["unitSI"] = unitSI
  a["unitDimension"] = collect(Float64.(dim))
end

"""
    write_particles(file, sim; every=1)
    write_particles(file, bins, t; every=1)

Write the particles of all bins (gathered from all MPI ranks) to an openPMD-beamphysics
HDF5 file. Every bin is written as a separate species group `particles/bin<i>`; the
species name is stored in the `speciesType` attribute.
"""
write_particles(file::AbstractString, sim::Simulation; every=1) =
  write_particles(file, sim.bins, sim.t; every=every, comm=sim.comm)

function write_particles(file::AbstractString, bins::AbstractVector{<:ParticleBin}, t; every=1,
                         comm::AbstractComm=SerialComm())
  gathered = [gather_bin(b, comm) for b in bins]
  isroot(comm) || return file
  HDF5.h5open(file, "w") do f
    _pmd_root!(f)
    base = HDF5.create_group(f, "data/0/particles")
    for (ib, b) in enumerate(gathered)
      r = Array(b.r)[1:every:end, :]
      w = Array(b.w)[1:every:end] .* every
      n = size(r, 1)
      g = HDF5.create_group(base, length(gathered) == 1 ? "beam" : "bin$ib")
      a = HDF5.attributes(g)
      a["speciesType"] = nameof(b.species)
      a["numParticles"] = n
      a["totalCharge"] = abs(sum(w))
      a["chargeUnitSI"] = 1.0
      a["chargeLive"] = abs(sum(w))
      a["mc2"] = b.mc2
      a["charge_e"] = b.charge
      gp = HDF5.create_group(g, "position")
      for (c, ax) in ((1, "x"), (3, "y"), (5, "z"))
        _dset!(gp, ax, r[:, c], 1.0, (1, 0, 0, 0, 0, 0, 0))
      end
      gm = HDF5.create_group(g, "momentum")
      for (c, ax) in ((2, "x"), (4, "y"), (6, "z"))
        _dset!(gm, ax, r[:, c] .* b.mc2, EV_C_UNIT_SI, (1, 1, -1, 0, 0, 0, 0))
      end
      _dset!(g, "time", fill(Float64(t), n), 1.0, (0, 0, 1, 0, 0, 0, 0))
      _dset!(g, "weight", abs.(w), 1.0, (0, 0, 1, 1, 0, 0, 0))
      g["particleStatus"] = fill(Int32(1), n)
      g["id"] = Array(b.id)[1:every:end]
    end
  end
  return file
end

"""
    read_particles(file) -> (bins, t)

Read particles written by `write_particles` (or any openPMD-beamphysics file with the
particles at a common time).
"""
function read_particles(file::AbstractString)
  bins = ParticleBin[]
  t = 0.0
  HDF5.h5open(file, "r") do f
    base = f["data"]
    step = first(keys(base))
    parts = base[step]["particles"]
    for name in keys(parts)
      g = parts[name]
      a = HDF5.attributes(g)
      sp = Species(read(a["speciesType"]))
      mc2 = haskey(a, "mc2") ? read(a["mc2"]) : massof(sp)
      ch = haskey(a, "charge_e") ? read(a["charge_e"]) : chargeof(sp)
      pos = g["position"]; mom = g["momentum"]
      rd(gg, ax) = read(gg[ax]) .* read(HDF5.attributes(gg[ax])["unitSI"])
      pu = 1 / (mc2 * EV_C_UNIT_SI)   # kg m/s → γβ
      r = hcat(rd(pos, "x"), rd(mom, "x") .* pu, rd(pos, "y"), rd(mom, "y") .* pu,
               rd(pos, "z"), rd(mom, "z") .* pu)
      w = read(g["weight"]) .* sign(ch)
      id = haskey(g, "id") ? Int.(read(g["id"])) : collect(1:size(r, 1))
      haskey(g, "time") && (t = first(read(g["time"])))
      push!(bins, ParticleBin(sp, r, w; mc2=mc2, charge=ch, id=id))
    end
  end
  return bins, t
end

"""
    write_stats(file, sim)

Write the beam statistics of the simulation to an HDF5 file (one dataset per quantity,
matrices with one column per diagnostic step).
"""
function write_stats(file::AbstractString, sim::Simulation)
  isroot(sim.comm) || return file
  st = sim.stats
  HDF5.h5open(file, "w") do f
    _write_stats_group(f, st)
    if !isempty(sim.bend_stats.t)
      g = HDF5.create_group(f, "bend")
      _write_stats_group(g, sim.bend_stats)
      g["reference"] = reduce(hcat, [collect(x) for x in sim.bend_ref])
    end
    a = HDF5.attributes(f)
    a["software"] = "ImpactTJ.jl"
    a["description"] = "sigma: packed upper triangle of the 6x6 central second moments of (x, gbx, y, gby, z, gbz)"
  end
  return file
end

function _write_stats_group(f, st::BeamStats)
  isempty(st.t) && return
  for fld in (:t, :z_mean, :gamma, :ekin, :beta, :r_max, :sigma_gamma, :n_particle, :charge)
    f[String(fld)] = getfield(st, fld)
  end
  for fld in (:mean, :sigma, :max_abs, :moment3, :moment4)
    f[String(fld)] = reduce(hcat, [collect(x) for x in getfield(st, fld)])
  end
  f["emit_x"] = emittance(st, 1); f["emit_y"] = emittance(st, 2); f["emit_z"] = emittance(st, 3)
  f["sigma_x"] = rms(st, 1); f["sigma_y"] = rms(st, 3); f["sigma_z"] = rms(st, 5)
end

"Read statistics written by `write_stats`."
function read_stats(file::AbstractString)
  st = BeamStats()
  HDF5.h5open(file, "r") do f
    for fld in (:t, :z_mean, :gamma, :ekin, :beta, :r_max, :sigma_gamma, :n_particle, :charge)
      setfield!(st, fld, read(f[String(fld)]))
    end
    for (fld, n) in ((:mean, 6), (:sigma, 21), (:max_abs, 6), (:moment3, 6), (:moment4, 6))
      M = read(f[String(fld)])
      setfield!(st, fld, [NTuple{n,Float64}(M[:, k]) for k in axes(M, 2)])
    end
  end
  return st
end

# ---------------------------------------------------------------------------------------
# Slice diagnostics

"""
    slice_analysis(bins, nslices; comm) -> NamedTuple

Slice properties along z: slice center, macroparticle count, current [A], normalized
slice emittances [m], rms energy spread with and without the linear energy-position
correlation within the slice [eV].
"""
function slice_analysis(bins::AbstractVector{<:ParticleBin}, nslices::Int; comm=SerialComm())
  gb = [gather_bin(b, comm) for b in bins]
  r = reduce(vcat, (Array(b.r) for b in gb))
  w = reduce(vcat, (Array(b.w) for b in gb))
  mc2 = first(gb).mc2
  z = r[:, 5]
  zmin, zmax = extrema(z)
  h = (zmax - zmin) / nslices * (1 + 1e-9)
  idx = [min(floor(Int, (zi - zmin) / h) + 1, nslices) for zi in z]
  zc = [zmin + (k - 0.5) * h for k in 1:nslices]
  cnt = zeros(Int, nslices); cur = zeros(nslices); ex = zeros(nslices); ey = zeros(nslices)
  dE = zeros(nslices); dEu = zeros(nslices)
  for k in 1:nslices
    sel = findall(==(k), idx)
    cnt[k] = length(sel)
    length(sel) < 2 && continue
    s = r[sel, :]
    γ = sqrt.(1 .+ s[:, 2] .^ 2 .+ s[:, 4] .^ 2 .+ s[:, 6] .^ 2)
    β = mean(s[:, 6] ./ γ)
    cur[k] = abs(sum(w[sel])) * β * C_LIGHT_SI / h
    em(a, b) = (u = a .- mean(a); v = b .- mean(b); sqrt(max(mean(u .^ 2) * mean(v .^ 2) - mean(u .* v)^2, 0.0)))
    ex[k] = em(s[:, 1], s[:, 2]); ey[k] = em(s[:, 3], s[:, 4])
    E = γ .* mc2
    dE[k] = std_(E)
    zz = s[:, 5] .- mean(s[:, 5])
    c = sum(zz .^ 2) > 0 ? sum(zz .* (E .- mean(E))) / sum(zz .^ 2) : 0.0
    dEu[k] = std_(E .- c .* zz)
  end
  return (z=zc, count=cnt, current=cur, emit_x=ex, emit_y=ey, sigma_E=dE, sigma_E_uncorrelated=dEu)
end

mean(v) = sum(v) / length(v)
std_(v) = (m = mean(v); sqrt(max(sum((v .- m) .^ 2) / length(v), 0.0)))

function write_slices(file::AbstractString, sim::Simulation, nslices::Int)
  sa = slice_analysis(sim.bins, nslices; comm=sim.comm)
  isroot(sim.comm) || return file
  HDF5.h5open(file, "w") do f
    for (k, v) in pairs(sa)
      f[String(k)] = v
    end
    HDF5.attributes(f)["t"] = sim.t
  end
  return file
end

# ---------------------------------------------------------------------------------------
# Checkpoints

"""
    write_checkpoint(file, sim)

Save everything needed to continue the simulation with `read_checkpoint!`.
"""
function write_checkpoint(file::AbstractString, sim::Simulation)
  write_particles(file, sim)
  isroot(sim.comm) || return file
  HDF5.h5open(file, "r+") do f
    g = HDF5.create_group(f, "state")
    g["t"] = sim.t; g["dt"] = sim.dt; g["step"] = sim.step; g["distance"] = sim.distance
    g["dzz"] = sim.dzz; g["sc_on"] = sim.sc_on; g["dt_fixed"] = sim.dt_fixed
    g["image_on"] = sim.image_on; g["round_aperture"] = sim.round_aperture
    g["queue_types"] = [string(T) for T in keys(sim.queues)]
    g["queue_next"] = [q.next for q in values(sim.queues)]
    g["done_dipoles"] = isempty(sim.done_dipoles) ? [0] : sim.done_dipoles
  end
  return file
end

"""
    read_checkpoint!(sim, file)

Restore the particles and the time stepping state saved with `write_checkpoint` into a
simulation built with the same lattice and settings.
"""
function read_checkpoint!(sim::Simulation, file::AbstractString)
  bins, t = read_particles(file)
  sim.bins = [scatter_bin(b, sim.comm) for b in bins]
  HDF5.h5open(file, "r") do f
    g = f["state"]
    sim.t = read(g["t"]); sim.dt = read(g["dt"]); sim.step = read(g["step"])
    sim.distance = read(g["distance"]); sim.dzz = read(g["dzz"]); sim.sc_on = read(g["sc_on"])
    sim.dt_fixed = read(g["dt_fixed"]); sim.image_on = read(g["image_on"])
    sim.round_aperture = read(g["round_aperture"])
    names = read(g["queue_types"]); nexts = read(g["queue_next"])
    for (T, q) in sim.queues
      k = findfirst(==(string(T)), names)
      isnothing(k) || (q.next = nexts[k])
    end
    sim.done_dipoles = filter(>(0), read(g["done_dipoles"]))
  end
  return sim
end
