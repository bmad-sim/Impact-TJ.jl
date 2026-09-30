# Initial particle distributions (IMPACT-T distribution types).
#
# Every distribution is described by the 21 IMPACT-T parameters organised per plane
# (x, y, z). Positions are in meters, momenta in γβ. A distribution that starts behind the
# cathode (z < 0) is emitted with the cathode emission model during the simulation.

"""
    PhasePlane(; sigma=0, sigma_p=0, corr=0, scale=1, p_scale=1, offset=0, p_offset=0)

Parameters of one phase plane (IMPACT-T: sig, sigp, mu, scale, pscale, xmu, pmu).
The meaning of `sigma`, `scale`, ... depends on the distribution type (see
[`BeamDistribution`](@ref)).
"""
Base.@kwdef struct PhasePlane
  sigma::Float64 = 0.0
  sigma_p::Float64 = 0.0
  corr::Float64 = 0.0
  scale::Float64 = 1.0
  p_scale::Float64 = 1.0
  offset::Float64 = 0.0
  p_offset::Float64 = 0.0
end

PhasePlane(v::AbstractVector) = PhasePlane(v[1], v[2], v[3], v[4], v[5], v[6], v[7])
Base.collect(p::PhasePlane) = [p.sigma, p.sigma_p, p.corr, p.scale, p.p_scale, p.offset, p.p_offset]

"""
    BeamDistribution(kind; x, y, z, file=nothing)

Initial distribution. `kind` is one of

- `:uniform`, `:gauss`, `:waterbag`, `:semigauss`, `:kv` — 6D distributions (IMPACT-T 1-5)
  with rms sizes `sigma·scale`, `sigma_p·p_scale` and correlation `corr` per plane.
- `:parabolic_gauss` (10), `:semicircle_gauss` (15) — round transverse profiles,
  longitudinal Gaussian, half-Gaussian γβz.
- `:cylinder_modulated` (27) — uniform cylinder with sinusoidal density modulation
  (amplitude `z.corr`, wavelength `z.scale`), quasi-random (Sobol) sampling.
- `:read` — particles from `file` (openPMD HDF5, or IMPACT-T `partcl.data` text formats);
  the plane offsets are added.
- `(:combine, i, j, k)` — IMPACT-T "ijk" photoinjector distributions: transverse
  `i` (1 uniform ellipse of radius sigma, 2 Gaussian cut at `scale` σ), longitudinal
  `j` (1 flat top of length `z.sigma` with linear ramps `z.scale`, 2 flat top with Gaussian
  ramps, 3 Gaussian cut at `z.scale` σ, 4 flat top with Tukey ramps) and momentum `k`
  (1 3D Gaussian, 2 transverse Gaussian + half Gaussian γβz, 3 x-ray emission model
  E/(E+W)⁴ sin2θ with Emax = `z.sigma_p` and W = `z.p_scale` [eV], 4 Dowell-Schmerge
  three-step model with photon energy `z.sigma_p`, Fermi energy `z.corr` and effective
  work function `z.p_scale` [eV]).
"""
struct BeamDistribution
  kind::Any
  x::PhasePlane
  y::PhasePlane
  z::PhasePlane
  file::Union{Nothing,String}
end

BeamDistribution(kind; x=PhasePlane(), y=PhasePlane(), z=PhasePlane(), file=nothing) =
  BeamDistribution(kind, x, y, z, file)

"IMPACT-T numeric distribution code → `kind`."
function impact_dist_kind(code::Integer)
  code == 1 && return :uniform
  code == 2 && return :gauss
  code == 3 && return :waterbag
  code == 4 && return :semigauss
  code == 5 && return :kv
  code == 10 && return :parabolic_gauss
  code == 15 && return :semicircle_gauss
  code in (16, 166, 167, 24, 25) && return :read
  code == 27 && return :cylinder_modulated
  i = code ÷ 100; j = (code - 100i) ÷ 10; k = code - 100i - 10j
  (1 <= i <= 2 && 1 <= j <= 4 && 1 <= k <= 4) || error("Unknown IMPACT-T distribution type $code")
  return (:combine, i, j, k)
end

function impact_dist_code(kind)
  kind == :uniform && return 1
  kind == :gauss && return 2
  kind == :waterbag && return 3
  kind == :semigauss && return 4
  kind == :kv && return 5
  kind == :parabolic_gauss && return 10
  kind == :semicircle_gauss && return 15
  kind == :read && return 166
  kind == :cylinder_modulated && return 27
  kind isa Tuple && kind[1] == :combine && return 100kind[2] + 10kind[3] + kind[4]
  error("No IMPACT-T code for distribution $kind")
end

# ---------------------------------------------------------------------------------------

_sq(c) = sqrt(1 - c^2)

# (u, v) unit-rms variables → (position, momentum) with correlation `corr`
@inline function _plane(pl::PhasePlane, u, v, possig=pl.sigma * pl.scale, momsig=pl.sigma_p * pl.p_scale)
  s = _sq(pl.corr)
  return pl.offset + possig * u / s, pl.p_offset + momsig * (-pl.corr * u / s + v)
end

function _box_muller(rng)
  x1 = rand(rng); x2 = rand(rng)
  x1 = x1 == 0 ? 1.0e-16 : x1
  r = sqrt(-2log(x1))
  return r * cos(2π * x2), r * sin(2π * x2)
end

"Gaussian pair cut at radius `cut` of the 2D normal (IMPACT-T normVecCut)."
function _box_muller_cut(rng, cut)
  while true
    x1 = rand(rng); x2 = rand(rng)
    x1 = x1 == 0 ? 1.0e-18 : x1
    r = sqrt(-2log(x1))
    r <= cut && return r * cos(2π * x2), r * sin(2π * x2)
  end
end

_normal12(rng) = sum(rand(rng) for _ in 1:12) - 6.0   # IMPACT-T normdv2

"""
    generate(dist, N, rng; mc2=massof(electron), ib=1, nb=1) -> (r, frac)

Sample `N` particles (N×6 matrix). For longitudinal flat-top "combine" distributions the
bunch is cut into `nb` longitudinal slices and only slice `ib` is kept (so the returned
count may be smaller than N); `frac` is the fraction of the requested particles kept.
"""
function generate(d::BeamDistribution, N::Integer, rng::AbstractRNG; mc2=510998.95069, ib=1, nb=1)
  kind = d.kind
  X, Y, Z = d.x, d.y, d.z
  r = zeros(N, 6)
  if kind == :uniform
    for p in 1:N
      for (c, pl) in ((1, X), (3, Y), (5, Z))
        u = (2rand(rng) - 1) * sqrt(3.0); v = (2rand(rng) - 1) * sqrt(3.0)
        r[p, c], r[p, c+1] = _plane(pl, u, v)
      end
    end
  elseif kind == :gauss
    for p in 1:N
      for (c, pl) in ((1, X), (3, Y), (5, Z))
        u, v = _box_muller(rng)
        r[p, c], r[p, c+1] = _plane(pl, u, v)
      end
    end
  elseif kind == :waterbag
    p = 0
    while p < N
      q = ntuple(_ -> 2rand(rng) - 1, 6)
      sum(abs2, q) > 1 && continue
      p += 1
      for (i, (c, pl)) in enumerate(((1, X), (3, Y), (5, Z)))
        r[p, c], r[p, c+1] = _plane(pl, q[2i-1] * sqrt(8.0), q[2i] * sqrt(8.0))
      end
    end
  elseif kind == :kv
    for p in 1:N
      r1 = rand(rng); r2 = 2π * rand(rng); r3 = 2π * rand(rng)
      a = sqrt(r1); b = sqrt(1 - r1)
      r[p, 1], r[p, 2] = _plane(X, 2a * cos(r2), 2a * sin(r2))
      r[p, 3], r[p, 4] = _plane(Y, 2b * cos(r3), 2b * sin(r3))
      u = (2rand(rng) - 1) * sqrt(3.0); v = (2rand(rng) - 1) * sqrt(3.0)
      r[p, 5], r[p, 6] = _plane(Z, u, v)
    end
  elseif kind == :semigauss
    for p in 1:N
      local q
      while true
        q = (2rand(rng) - 1, 2rand(rng) - 1, 2rand(rng) - 1)
        sum(abs2, q) <= 1 && break
      end
      for (i, (c, pl)) in enumerate(((1, X), (3, Y), (5, Z)))
        r[p, c], r[p, c+1] = _plane(pl, q[i] * sqrt(5.0), _normal12(rng))
      end
    end
  elseif kind == :parabolic_gauss || kind == :semicircle_gauss
    for p in 1:N
      u1 = rand(rng); θ = 2π * rand(rng)
      sq = kind == :parabolic_gauss ? sqrt(1 - sqrt(u1)) * sqrt(6.0) : sqrt(1 - u1^(2 / 3)) * sqrt(5.0)
      r[p, 1] = X.offset + X.sigma * X.scale * sq * cos(θ)
      r[p, 3] = Y.offset + Y.sigma * Y.scale * sq * sin(θ)
      r[p, 2] = X.p_offset + X.sigma_p * X.p_scale * randn(rng)
      r[p, 4] = Y.p_offset + Y.sigma_p * Y.p_scale * randn(rng)
      local zt
      while true
        zt = randn(rng)
        abs(zt) <= 4 && break
      end
      r[p, 5] = Z.offset + Z.sigma * zt
      r[p, 6] = Z.p_offset + Z.sigma_p * abs(randn(rng))
    end
  elseif kind == :cylinder_modulated
    _cylinder_modulated!(r, X, Y, Z)
  elseif kind isa Tuple && kind[1] == :combine
    return _combine(d, N, rng, mc2, ib, nb)
  elseif kind == :read
    error("`:read` distributions are loaded with `load_particles`")
  else
    error("Unknown distribution kind $kind")
  end
  return r, 1.0
end

function _combine(d::BeamDistribution, N, rng, mc2, ib, nb)
  _, i, j, k = d.kind
  X, Y, Z = d.x, d.y, d.z
  N0 = N * nb   # the full (all slices) distribution
  # longitudinal
  if j == 3
    zs = [Z.offset + Z.sigma * first(_box_muller_cut(rng, Z.scale)) - Z.scale * Z.sigma for _ in 1:N]
    kept = N
  else
    zrise = j == 2 ? 2Z.scale : Z.scale
    zflat = Z.sigma
    totleng = zflat + 2zrise
    Δ = totleng / nb
    zminbin = -ib * Δ; zmaxbin = -(ib - 1) * Δ
    zmingl = -totleng
    z1 = zmingl + zrise; z2 = z1 + zflat; z3 = 0.0
    zs = Float64[]
    ntried = 0
    while ntried < N0
      zz = -totleng * rand(rng)
      fv = if zz >= zmingl && zz < z1
        j == 1 ? (zz - zmingl) / (z1 - zmingl) :
        j == 2 ? exp(-((zz - z1) / Z.scale)^2) : 0.5 * (1 - cos(π * (zz - zmingl) / (z1 - zmingl)))
      elseif zz >= z1 && zz < z2
        1.0
      elseif zz >= z2 && zz <= z3
        j == 1 ? 1 - (zz - z2) / (z3 - z2) :
        j == 2 ? exp(-((zz - z2) / Z.scale)^2) : 0.5 * (1 - cos(π * (1 - (zz - z2) / (z3 - z2))))
      else
        0.0
      end
      rand(rng) > fv && continue
      ntried += 1
      (zz <= zmaxbin && zz > zminbin) && push!(zs, Z.offset + zz)
    end
    kept = length(zs)
  end
  r = zeros(kept, 6)
  r[:, 5] .= zs
  # transverse positions
  for p in 1:kept
    if i == 1
      u = sqrt(rand(rng)); θ = 2π * rand(rng)
      r[p, 1] = X.offset + X.sigma * u * cos(θ)
      r[p, 3] = Y.offset + Y.sigma * u * sin(θ)
    else
      a, b = _box_muller_cut(rng, X.scale)   # IMPACT-T uses the x cut for both planes
      r[p, 1] = X.offset + X.sigma * a
      r[p, 3] = Y.offset + Y.sigma * b
    end
  end
  # momenta
  if k == 1
    for p in 1:kept
      r[p, 2] = X.p_offset + X.sigma_p * _normal12(rng)
      r[p, 4] = Y.p_offset + Y.sigma_p * _normal12(rng)
      r[p, 6] = Z.p_offset + Z.sigma_p * _normal12(rng)
    end
  elseif k == 2
    for p in 1:kept
      a, b = _box_muller(rng)
      r[p, 2] = X.p_offset + X.sigma_p * a
      r[p, 4] = Y.p_offset + Y.sigma_p * b
      rr = rand(rng); rr = rr == 0 ? 1.0e-18 : rr
      r[p, 6] = Z.p_offset + Z.sigma_p * sqrt(-log(rr))
    end
  elseif k == 3
    emax = Z.sigma_p; wkf = Z.p_scale
    ep = wkf / 3
    fmax = ep / (ep + wkf)^4
    p = 0
    while p < kept
      e1 = rand(rng) * emax
      rand(rng) > e1 / (e1 + wkf)^4 / fmax && continue
      p += 1
      γ = e1 / mc2 + 1
      gb = sqrt(γ^2 - 1)
      θ = acos(-(2rand(rng) - 1)) / 2
      ϕ = 2π * rand(rng)
      r[p, 2] = X.p_offset + gb * sin(θ) * cos(ϕ)
      r[p, 4] = Y.p_offset + gb * sin(θ) * sin(ϕ)
      r[p, 6] = Z.p_offset + gb * cos(θ)
    end
  else # k == 4, Dowell-Schmerge three step model
    Eph = Z.sigma_p; Ef = Z.corr; Ewk = Z.p_scale; Tem = 0.026
    Emax = Ef + Eph + 10Tem; Emin = Ef + Ewk
    Emin < Emax || error("photon energy too small for the three step emission model")
    fmax = (1 / (1 + exp(-Eph / Tem / 2)))^2
    p = 0
    while p < kept
      vr1 = Emin + rand(rng) * (Emax - Emin)
      t3 = (vr1 - Ef) / Tem
      f5 = t3 > 0 ? 1 / (1 + exp(-t3)) : exp(t3) / (1 + exp(t3))
      t6 = (vr1 - Ef - Eph) / Tem
      f8 = t6 > 0 ? exp(-t6) / (1 + exp(-t6)) : 1 / (1 + exp(t6))
      rand(rng) > f5 * f8 / fmax && continue
      c = rand(rng)
      t9 = vr1 * c * c
      t9 < Ef + Ewk && continue
      p += 1
      θ = acos(c); ϕ = 2π * rand(rng)
      pt = sqrt(2vr1 / mc2)
      r[p, 2] = X.p_offset + pt * sin(θ) * cos(ϕ)
      r[p, 4] = Y.p_offset + pt * sin(θ) * sin(ϕ)
      r[p, 6] = Z.p_offset + sqrt(2 * (t9 - Ef - Ewk) / mc2)
    end
  end
  return r, (j == 3 ? 1.0 : kept / N)
end

# Uniform cylinder with longitudinal modulation, quasi-random (Sobol) sampling.
function _cylinder_modulated!(r, X, Y, Z)
  N = size(r, 1)
  sob = SobolSeq6()
  rk = 2π / Z.scale
  xmod = Z.corr
  for p in 1:N
    q = next!(sob)
    a = sqrt(q[1]); c, s = cos(2π * q[2]), sin(2π * q[2])
    r[p, 1] = X.offset + X.sigma * X.scale * a * c
    r[p, 3] = Y.offset + Y.sigma * Y.scale * a * s
    q3 = q[3] == 0 ? 1.0e-18 : q[3]
    g = sqrt(-2log(q3))
    r[p, 2] = X.p_offset + X.sigma_p * X.p_scale * g * cos(2π * q[4])
    r[p, 4] = Y.p_offset + Y.sigma_p * Y.p_scale * g * sin(2π * q[4])
    xz = (2q[5] - 1) * Z.sigma
    ψ = xz
    for _ in 1:500   # solve ψ + m sin(kψ)/k = xz
      f = ψ + xmod * sin(rk * ψ) / rk - xz
      abs(f) < 1.0e-8 * max(abs(Z.sigma), 1.0e-30) && break
      ψ -= f / (1 + xmod * cos(rk * ψ))
    end
    r[p, 5] = Z.offset + ψ
    q6 = q[6] == 0 ? 1.0e-18 : q[6]
    r[p, 6] = Z.p_offset + Z.sigma_p * sqrt(-2log(q6))
  end
  return r
end

# Minimal 6D Sobol sequence (Joe-Kuo direction numbers).
mutable struct SobolSeq6
  v::Matrix{UInt32}
  x::Vector{UInt32}
  n::UInt32
end
function SobolSeq6()
  s = [1, 2, 3, 3, 4, 4]; a = [0, 1, 1, 2, 1, 4]
  m = [[1], [1, 3], [1, 3, 1], [1, 1, 1], [1, 1, 3, 3], [1, 3, 5, 13]]
  v = zeros(UInt32, 32, 6)
  for d in 1:6
    if d == 1
      for i in 1:32
        v[i, d] = UInt32(1) << (32 - i)
      end
    else
      sd = s[d]
      for i in 1:sd
        v[i, d] = UInt32(m[d][i]) << (32 - i)
      end
      for i in sd+1:32
        v[i, d] = v[i-sd, d] ⊻ (v[i-sd, d] >> sd)
        for k in 1:sd-1
          v[i, d] ⊻= ((a[d] >> (sd - 1 - k)) & 1) * v[i-k, d]
        end
      end
    end
  end
  SobolSeq6(v, zeros(UInt32, 6), UInt32(0))
end
function next!(s::SobolSeq6)
  c = trailing_ones(s.n) + 1
  s.n += UInt32(1)
  for d in 1:6
    s.x[d] ⊻= s.v[c, d]
  end
  return [Float64(s.x[d]) / 2.0^32 for d in 1:6]
end
