# Bunch reductions and diagnostics.

"""
    BinSummary

Extent and centroid of a bin (over all ranks): `lo`/`hi` are the min/max of (x, y, z),
`mean` the mean of the six coordinates, `n` the number of live particles.
"""
struct BinSummary
  lo::NTuple{3,Float64}
  hi::NTuple{3,Float64}
  mean::NTuple{6,Float64}
  n::Int
end

function _summary_local(r::Matrix{Float64}, state)
  N = size(r, 1)
  nt = Threads.nthreads()
  parts = Vector{NTuple{13,Float64}}(undef, nt)
  Threads.@threads :static for t in 1:nt
    lo = div(N * (t - 1), nt) + 1
    hi = div(N * t, nt)
    xmn = ymn = zmn = Inf; xmx = ymx = zmx = -Inf
    s1 = s2 = s3 = s4 = s5 = s6 = 0.0; c = 0.0
    @inbounds for p in lo:hi
      state[p] == STATE_ALIVE || continue
      x = r[p, 1]; y = r[p, 3]; z = r[p, 5]
      xmn = min(xmn, x); xmx = max(xmx, x)
      ymn = min(ymn, y); ymx = max(ymx, y)
      zmn = min(zmn, z); zmx = max(zmx, z)
      s1 += x; s2 += r[p, 2]; s3 += y; s4 += r[p, 4]; s5 += z; s6 += r[p, 6]
      c += 1
    end
    parts[t] = (xmn, ymn, zmn, xmx, ymx, zmx, s1, s2, s3, s4, s5, s6, c)
  end
  return reduce(parts) do a, b
    (min(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]), max(a[4], b[4]), max(a[5], b[5]),
     max(a[6], b[6]), ntuple(i -> a[6+i] + b[6+i], 7)...)
  end
end

@inline function _summary_item(p, r, state)
  @inbounds if state[p] == STATE_ALIVE
    x = r[p, 1]; y = r[p, 3]; z = r[p, 5]
    return (x, y, z, x, y, z, x, r[p, 2], y, r[p, 4], z, r[p, 6], 1.0)
  else
    return (Inf, Inf, Inf, -Inf, -Inf, -Inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  end
end
const _SUMMARY_KINDS = (0x01, 0x01, 0x01, 0x02, 0x02, 0x02, ntuple(_ -> 0x00, 7)...)

# generic (GPU) path: one fused device reduction
_summary_local(r::AbstractMatrix, state) =
  device_reduce(_summary_item, _SUMMARY_KINDS, size(r, 1), get_backend(r), r, state)

function bin_summary(b::ParticleBin, comm::AbstractComm)
  v = size(b.r, 1) == 0 ? (Inf, Inf, Inf, -Inf, -Inf, -Inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) :
      _summary_local(b.r, b.state)
  lo = allreduce_min(comm, [v[1], v[2], v[3]])
  hi = allreduce_max(comm, [v[4], v[5], v[6]])
  s = allreduce_sum(comm, ntuple(i -> v[6+i], 7))
  n = s[7]
  m = n > 0 ? ntuple(i -> s[i] / n, 6) : ntuple(_ -> 0.0, 6)
  return BinSummary(Tuple(lo), Tuple(hi), m, round(Int, n))
end

# ---------------------------------------------------------------------------------------
# Beam statistics

"""
Beam statistics recorded during a simulation (one entry per diagnostic step). All units
SI; momenta are γβ. Emittances are normalized rms emittances [m].
"""
Base.@kwdef mutable struct BeamStats
  t::Vector{Float64} = Float64[]            # time [s]
  z_mean::Vector{Float64} = Float64[]       # centroid z [m]
  gamma::Vector{Float64} = Float64[]        # mean γ
  ekin::Vector{Float64} = Float64[]         # mean kinetic energy [eV]
  beta::Vector{Float64} = Float64[]         # β of the mean γ
  r_max::Vector{Float64} = Float64[]        # max radius [m]
  sigma_gamma::Vector{Float64} = Float64[]  # rms γ spread
  mean::Vector{NTuple{6,Float64}} = NTuple{6,Float64}[]     # <x>, <γβx>, ...
  sigma::Vector{NTuple{21,Float64}} = NTuple{21,Float64}[]  # upper triangle of the 6×6 second central moments
  max_abs::Vector{NTuple{6,Float64}} = NTuple{6,Float64}[]  # max |x|, |px|, |y|, |py|, |z-<z>|, |pz|
  moment3::Vector{NTuple{6,Float64}} = NTuple{6,Float64}[]  # cube root of third central moments
  moment4::Vector{NTuple{6,Float64}} = NTuple{6,Float64}[]  # fourth root of fourth central moments
  n_particle::Vector{Int} = Int[]
  charge::Vector{Float64} = Float64[]       # total charge [C]
end

"Index of <uᵢ uⱼ> (1 ≤ i ≤ j ≤ 6) in the packed upper triangle."
@inline sigidx(i, j) = i <= j ? (i - 1) * 6 - div((i - 1) * (i - 2), 2) + (j - i + 1) : sigidx(j, i)

"Normalized rms emittance of plane 1 (x), 2 (y) or 3 (z) of stats entry `k`."
function emittance(st::BeamStats, plane::Int, k::Int)
  a = 2plane - 1; b = 2plane
  s = st.sigma[k]
  return sqrt(max(s[sigidx(a, a)] * s[sigidx(b, b)] - s[sigidx(a, b)]^2, 0.0))
end
emittance(st::BeamStats, plane::Int) = [emittance(st, plane, k) for k in eachindex(st.t)]
rms(st::BeamStats, i::Int) = [sqrt(max(s[sigidx(i, i)], 0.0)) for s in st.sigma]

# Raw moments about a provisional center `c` (the unprojected centroid). Layout of the
# returned sums: 1:21 Σ(u-c)ᵢ(u-c)ⱼ, 22:27 Σ(u-c), 28:33 Σ(u-c)³, 34:39 Σ(u-c)⁴,
# 40:46 max |x|,|px|,|y|,|py|,|z-c|,|pz|, r², 47 Σγ, 48 Σγ², 49 count, 50 Σw.
const NMOM = 50
const MAXRANGE = 40:46

# moment contributions of particle p (layout above; 40:46 hold maxima, the rest sums)
@inline function _moment_item(p, r, state, w, c::NTuple{6,Float64}, zproj::Bool)
  @inbounds begin
    state[p] == STATE_ALIVE || return ntuple(_ -> 0.0, Val(NMOM))
    px = r[p, 2]; py = r[p, 4]; pz = r[p, 6]
    γ = sqrt(1 + px^2 + py^2 + pz^2)
    x = r[p, 1]; y = r[p, 3]; z = r[p, 5]
    if zproj   # project transverse positions to the centroid plane z = zc
      dtc = (z - c[5]) / (pz / γ)
      x -= px / γ * dtc; y -= py / γ * dtc
    end
    u = (x - c[1], px - c[2], y - c[3], py - c[4], z - c[5], pz - c[6])
    uu = ntuple(Val(21)) do k
      i, j = _SIGPAIRS[k]
      u[i] * u[j]
    end
    return (uu..., u..., ntuple(i -> u[i]^3, Val(6))..., ntuple(i -> u[i]^4, Val(6))...,
            abs(x), abs(px), abs(y), abs(py), abs(u[5]), abs(pz), x^2 + y^2,
            γ, γ^2, 1.0, w[p])
  end
end
const _SIGPAIRS = Tuple((i, j) for i in 1:6 for j in i:6)
const _MOMENT_KINDS = ntuple(i -> i in MAXRANGE ? 0x02 : 0x00, NMOM)

function _moments_local(r::Matrix{Float64}, state, w, c::NTuple{6,Float64}, zproj::Bool)
  N = size(r, 1)
  nt = Threads.nthreads()
  parts = Vector{NTuple{NMOM,Float64}}(undef, nt)
  Threads.@threads :static for t in 1:nt
    lo = div(N * (t - 1), nt) + 1
    hi = div(N * t, nt)
    acc = ntuple(_ -> 0.0, Val(NMOM))
    @inbounds for p in lo:hi
      acc = _combt(_MOMENT_KINDS, acc, _moment_item(p, r, state, w, c, zproj))
    end
    parts[t] = acc
  end
  return collect(reduce((a, b) -> _combt(_MOMENT_KINDS, a, b), parts))
end

# generic (GPU) path
_moments_local(r::AbstractMatrix, state, w, c::NTuple{6,Float64}, zproj::Bool) =
  collect(device_reduce(_moment_item, _MOMENT_KINDS, size(r, 1), get_backend(r), r, state, w, c, zproj))

"""
    record_stats!(st, t, bins, comm; at_fixed_z=false)

Compute beam moments of all bins combined and append them to `st`. With `at_fixed_z`,
transverse positions are projected (ballistically) to the plane of the bunch centroid
(IMPACT-T diagnostic flag 2); otherwise statistics are at fixed time.
"""
function record_stats!(st::BeamStats, t, bins, comm::AbstractComm; at_fixed_z::Bool=false)
  tot = zeros(7)
  for b in bins
    s = bin_summary(b, comm)
    for i in 1:6
      tot[i] += s.mean[i] * s.n
    end
    tot[7] += s.n
  end
  ntot = tot[7]
  ntot == 0 && return st
  c = ntuple(i -> tot[i] / ntot, 6)
  sums = zeros(NMOM)
  for b in bins
    size(b.r, 1) == 0 && continue
    q = _moments_local(b.r, b.state, b.w, c, at_fixed_z)
    for i in 1:NMOM
      sums[i] = i in MAXRANGE ? max(sums[i], q[i]) : sums[i] + q[i]
    end
  end
  mx = allreduce_max(comm, sums[MAXRANGE])
  sm = allreduce_sum!(comm, vcat(sums[1:39], sums[47:50]))
  d = ntuple(i -> sm[21+i] / ntot, 6)                 # mean offset from the provisional center
  m0 = ntuple(i -> c[i] + d[i], 6)
  k = 0
  sig = zeros(21)
  for i in 1:6, j in i:6
    k += 1
    sig[k] = sm[k] / ntot - d[i] * d[j]
  end
  m3 = ntuple(6) do i
    e1 = d[i]; e2 = sig[sigidx(i, i)] + d[i]^2; e3 = sm[27+i] / ntot
    cbrt(e3 - 3e1 * e2 + 2e1^3)
  end
  m4 = ntuple(6) do i
    e1 = d[i]; e2 = sig[sigidx(i, i)] + d[i]^2; e3 = sm[27+i] / ntot; e4 = sm[33+i] / ntot
    sqrt(sqrt(max(e4 - 4e1 * e3 + 6e1^2 * e2 - 3e1^4, 0.0)))
  end
  γm = sm[40] / ntot
  mc2 = first(bins).mc2
  push!(st.t, t)
  push!(st.z_mean, m0[5])
  push!(st.gamma, γm)
  push!(st.ekin, (γm - 1) * mc2)
  push!(st.beta, sqrt(max(1 - 1 / γm^2, 0.0)))
  push!(st.r_max, sqrt(mx[7]))
  push!(st.sigma_gamma, sqrt(abs(sm[41] / ntot - γm^2)))
  push!(st.mean, m0)
  push!(st.sigma, Tuple(sig))
  push!(st.max_abs, Tuple(mx[1:6]))
  push!(st.moment3, m3)
  push!(st.moment4, m4)
  push!(st.n_particle, round(Int, sm[42]))
  push!(st.charge, sm[43])
  return st
end
