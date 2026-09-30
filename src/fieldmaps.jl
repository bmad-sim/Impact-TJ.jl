# User facing field map descriptions (host side). They are attached to Beamlines elements
# through the `field_map` / `sol_map` properties and compiled into the isbits field models
# of `fields/elements.jl`.

abstract type FieldMap end

"""
    OnAxisFourier(coefs; zstart, zend, zlen=zend-zstart, zc=zstart+zlen/2)

On-axis field given by the Fourier series a0/2 + Σ aₙcos(2πn(z-zc)/zlen) + bₙsin(...)
(`coefs = [a0, a1, b1, a2, b2, ...]`), defined for zstart ≤ z ≤ zend (element local z).
The series is tabulated once at setup so its evaluation cost is independent of the
number of modes.
"""
struct OnAxisFourier <: FieldMap
  coefs::Vector{Float64}
  zstart::Float64
  zend::Float64
  zlen::Float64
  zc::Float64
end
OnAxisFourier(coefs; zstart, zend, zlen=zend - zstart, zc=zstart + zlen / 2) =
  OnAxisFourier(Vector{Float64}(coefs), zstart, zend, zlen, zc)

"""
    OnAxisData(z0, dz, f; fp, fpp, fppp, periodic=false, interpolation=:linear)

On-axis field (and its first three z-derivatives) on a uniform grid starting at local z0.
Missing derivatives are computed by finite differences. With `periodic = true` the data
is repeated along the element (IMPACT-T traveling wave cell data). `interpolation` is
`:linear` (IMPACT-T) or `:hermite` (4th order).
"""
struct OnAxisData <: FieldMap
  z0::Float64
  dz::Float64
  f::Vector{Float64}
  fp::Vector{Float64}
  fpp::Vector{Float64}
  fppp::Vector{Float64}
  periodic::Bool
  interpolation::Symbol
end

function _fd(f, dz)
  n = length(f)
  d = similar(f)
  n == 1 && (d[1] = 0; return d)
  d[1] = (f[2] - f[1]) / dz; d[n] = (f[n] - f[n-1]) / dz
  for i in 2:n-1
    d[i] = (f[i+1] - f[i-1]) / (2dz)
  end
  return d
end

function OnAxisData(z0, dz, f; fp=nothing, fpp=nothing, fppp=nothing, periodic=false,
                    interpolation=:linear)
  f = Vector{Float64}(f)
  fp = isnothing(fp) ? _fd(f, dz) : Vector{Float64}(fp)
  fpp = isnothing(fpp) ? _fd(fp, dz) : Vector{Float64}(fpp)
  fppp = isnothing(fppp) ? _fd(fpp, dz) : Vector{Float64}(fppp)
  OnAxisData(Float64(z0), Float64(dz), f, fp, fpp, fppp, periodic, interpolation)
end

"""
    RZSolenoidMap(r_max, z0, z1, Br, Bz)

Static solenoid map on a uniform (r, z) grid with r ∈ [0, r_max] and local z ∈ [z0, z1];
`Br`, `Bz` are (nr, nz) matrices [T].
"""
struct RZSolenoidMap <: FieldMap
  rmax::Float64
  z0::Float64
  z1::Float64
  Br::Matrix{Float64}
  Bz::Matrix{Float64}
end

"""
    RZCavityMap(r_max, z0, z1, Ez, Er, Bθ)

TM-mode cavity map on a uniform (r, z) grid (matrices (nr, nz), fields in V/m and T at
peak phase). Time dependence: E ∝ cos(ωt+φ0), Bθ ∝ -sin(ωt+φ0).
"""
struct RZCavityMap <: FieldMap
  rmax::Float64
  z0::Float64
  z1::Float64
  Ez::Matrix{Float64}
  Er::Matrix{Float64}
  Btheta::Matrix{Float64}
end

"""
    CartesianMap(xlim, ylim, zlim, E, B)

3D complex field map on a uniform grid covering `xlim = (xmin, xmax)` etc. (local
coordinates). `E` and `B` are arrays of size (3, nx, ny, nz) of complex values; the
physical field is Re[F e^{i(ωt+φ0)}].
"""
struct CartesianMap <: FieldMap
  xlim::NTuple{2,Float64}
  ylim::NTuple{2,Float64}
  zlim::NTuple{2,Float64}
  E::Array{ComplexF64,4}
  B::Array{ComplexF64,4}
end

"""
    PoleFaceGeometry(k, b, dd1, dd2, enge; gamma_ref, csr=false, s_start, s_end)

Bending magnet pole face description (IMPACT-T dipole data file): faces z = kᵢx + bᵢ,
Enge coefficients `enge` (8 values), fringe shifts `dd1`, `dd2`, the design γ of the
reference particle, the CSR switch and the effective start/end arc positions used by
the CSR calculation.
"""
struct PoleFaceGeometry
  k::NTuple{4,Float64}
  b::NTuple{4,Float64}
  dd1::Float64
  dd2::Float64
  enge::NTuple{8,Float64}
  gamma_ref::Float64
  csr::Bool
  s_start::Float64
  s_end::Float64
end

"""
    PoleFaceGeometry(; angle, rho, e1=0, e2=0, fringe=0.0, enge=default, gamma_ref, csr=false)

Pole faces of a sector-like bend of bending angle `angle` and radius `rho` with entrance
and exit edge angles `e1`, `e2` and fringe region width `fringe` (IMPACT-T manual eq.
40-47).
"""
function PoleFaceGeometry(; angle, rho, e1=0.0, e2=0.0, fringe=0.0,
                          enge=(-0.003183, 2.32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                          gamma_ref, csr=false, s_start=fringe / 2,
                          s_end=fringe / 2 + rho * angle)
  Δ = fringe; R = rho; θ = angle
  b1 = 0.0
  b2 = Δ / cos(e1)
  b3 = Δ / (2cos(e1)) + R * sin(θ) + (R - R * cos(θ)) * tan(θ - e2) - Δ / (2cos(θ - e2))
  b4 = b3 + Δ / cos(θ - e2)
  k1 = -tan(e1); k3 = -tan(θ - e2)
  PoleFaceGeometry((k1, k1, k3, k3), (b1, b2, b3, b4), Δ / 2, Δ / 2, Tuple(Float64.(enge)),
                   gamma_ref, csr, s_start, s_end)
end

# ---------------------------------------------------------------------------------------
# Wakefield models (short range structure wakes)

abstract type WakeModel end

"Bane's analytic wakes of a periodic structure with iris radius `a`, gap `g`, period `L`."
struct BaneWake <: WakeModel
  a::Float64; g::Float64; L::Float64
end
"Hardwired wakes of the ELETTRA backward traveling wave structure (Craievich et al.)."
struct BTWWake <: WakeModel end
"TESLA 1.3 GHz standing wave cavity wake (per cavity length)."
struct TeslaWake <: WakeModel end
"TESLA 3.9 GHz (third harmonic) cavity longitudinal wake."
struct Tesla3p9Wake <: WakeModel end
"""
    TabulatedWake(L, Wz, Wx, Wy)

Wake functions tabulated uniformly on s ∈ [0, L]: longitudinal Wz [V/C/m... as per unit
length], transverse Wx, Wy [V/C/m²].
"""
struct TabulatedWake <: WakeModel
  L::Float64
  Wz::Vector{Float64}
  Wx::Vector{Float64}
  Wy::Vector{Float64}
end
