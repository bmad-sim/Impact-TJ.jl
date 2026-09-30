# External field models.
#
# Every model is an isbits struct holding a `Placement` (where the field lives in the lab
# frame and how the element is misaligned) plus model parameters. `field(e, buf, x, y, z, t)`
# returns the lab-frame (E [V/m], B [T]) at lab position (x, y, z) [m] and time t [s].
# Fields of overlapping elements are summed, so standing and traveling wave structures and
# solenoids superimposed on RF guns are modeled naturally.

const V3 = SVector{3,Float64}
const ZERO3 = V3(0.0, 0.0, 0.0)

"""
Placement of a field element.

- `zmin`, `zmax`: lab z-range outside of which the field is zero.
- `s0`: lab z of the element's local origin (entrance) when perfectly aligned.
- `misaligned`: if true, local coordinates are `R*(r - o)` and fields are rotated back
  with `R'`; else local coordinates are `(x, y, z - s0)`.
"""
struct Placement
  zmin::Float64
  zmax::Float64
  s0::Float64
  misaligned::Bool
  R::SMatrix{3,3,Float64,9}
  o::V3
end

const IDENTITY3 = SMatrix{3,3,Float64,9}(1, 0, 0, 0, 1, 0, 0, 0, 1)

Placement(zmin, zmax, s0) = Placement(zmin, zmax, s0, false, IDENTITY3, V3(0, 0, s0))

rotx(a) = SMatrix{3,3,Float64,9}(1, 0, 0, 0, cos(a), sin(a), 0, -sin(a), cos(a))
roty(a) = SMatrix{3,3,Float64,9}(cos(a), 0, -sin(a), 0, 1, 0, sin(a), 0, cos(a))
rotz(a) = SMatrix{3,3,Float64,9}(cos(a), sin(a), 0, -sin(a), cos(a), 0, 0, 0, 1)

"""
    Placement(zmin, zmax, s0, L; x_offset, y_offset, z_offset, x_rot, y_rot, tilt)

Misaligned placement. The element body (entrance at lab `s0`, length `L`) is rotated by
W = Rz(tilt)·Ry(y_rot)·Rx(x_rot) about its center and then shifted by the offsets
(Bmad/Beamlines convention: positive angles are right handed rotations of the element).
"""
function Placement(zmin, zmax, s0, L; x_offset=0.0, y_offset=0.0, z_offset=0.0,
                   x_rot=0.0, y_rot=0.0, tilt=0.0)
  if x_offset == 0 && y_offset == 0 && z_offset == 0 && x_rot == 0 && y_rot == 0 && tilt == 0
    return Placement(zmin, zmax, s0)
  end
  W = rotz(tilt) * roty(y_rot) * rotx(x_rot)
  R = transpose(W)                              # lab -> local
  c = V3(x_offset, y_offset, s0 + L / 2 + z_offset)  # lab position of element center
  o = c - W * V3(0, 0, L / 2)                   # lab position of local origin
  return Placement(zmin, zmax, s0, true, SMatrix{3,3,Float64,9}(R), o)
end

@inline function to_local(p::Placement, x, y, z)
  if p.misaligned
    return p.R * (V3(x, y, z) - p.o)
  else
    return V3(x, y, z - p.s0)
  end
end

@inline to_lab(p::Placement, F::V3) = p.misaligned ? transpose(p.R) * F : F

@inline in_extent(p::Placement, z) = (z >= p.zmin) & (z <= p.zmax)

abstract type FieldModel end

"Return the model with its time dependent factors evaluated at time `t` (identity by default)."
set_time(m::FieldModel, t) = m
"Whether models of this type have time dependent factors precomputed by `set_time`."
time_dependent(::Type) = false

# ---------------------------------------------------------------------------------------
"""
Cylindrically symmetric RF field from the on-axis Ez(z) (types 101-105 of IMPACT-T).
Off-axis fields use the expansion to third order in r:

    Ez = (e + f r²) cos φ,  Er = -r (e'/2 + f' r²/4) cos φ,  Bθ = r k/c (e/2 + f r²/4) sin φ
    f = -(e'' + k² e)/4,  φ = ωt + θ0,  k = ω/c
"""
struct RFAxisField <: FieldModel
  pl::Placement
  tab::AxisTable
  scale::Float64
  omega::Float64
  phi0::Float64
  sph::Float64     # sin(ωt+φ0) at the current time (set once per step by `update_time!`)
  cph::Float64     # cos(ωt+φ0)
end
RFAxisField(pl, tab, scale, omega, phi0) = RFAxisField(pl, tab, scale, omega, phi0, sin(phi0), cos(phi0))
time_dependent(::Type{RFAxisField}) = true
function set_time(e::RFAxisField, t)
  s, c = sincos(e.omega * t + e.phi0)
  return RFAxisField(e.pl, e.tab, e.scale, e.omega, e.phi0, s, c)
end

@inline function field(e::RFAxisField, buf, x, y, z, t)
  l = to_local(e.pl, x, y, z)
  f0, f1, f2, f3 = axis_eval(buf, e.tab, l[3])
  s = e.scale
  f0 *= s; f1 *= s; f2 *= s; f3 *= s
  k = e.omega / C_LIGHT_SI
  g = -(f2 + f0 * k * k) / 4
  gp = -(f3 + f1 * k * k) / 4
  sφ = e.sph; cφ = e.cph
  lx = l[1]; ly = l[2]
  r2 = lx * lx + ly * ly
  er = -(f1 / 2 + gp * r2 / 4) * cφ
  bt = k / C_LIGHT_SI * (f0 / 2 + g * r2 / 4) * sφ
  E = V3(lx * er, ly * er, (f0 + g * r2) * cφ)
  B = V3(ly * bt, -lx * bt, 0.0)
  return to_lab(e.pl, E), to_lab(e.pl, B)
end

"""
Solenoid field from the on-axis Bz(z) (third order expansion):
    Bx,y = -b' x,y/2 + b''' x,y r²/16,   Bz = b - b'' r²/4
"""
struct SolAxisField <: FieldModel
  pl::Placement
  tab::AxisTable
  scale::Float64
end

@inline function field(e::SolAxisField, buf, x, y, z, t)
  l = to_local(e.pl, x, y, z)
  b0, b1, b2, b3 = axis_eval(buf, e.tab, l[3])
  s = e.scale
  lx = l[1]; ly = l[2]
  r2 = lx * lx + ly * ly
  br = s * (-b1 / 2 + b3 * r2 / 16)
  B = V3(br * lx, br * ly, s * (b0 - b2 * r2 / 4))
  return ZERO3, to_lab(e.pl, B)
end

# ---------------------------------------------------------------------------------------
"Uniform (r, z) grid description. Data node (ir, iz) (0-based) is at buf[off + (iz*nr + ir)*ncomp + 1:ncomp]."
struct RZGrid
  off::Int
  nr::Int          # number of r nodes
  nz::Int          # number of z nodes
  dr::Float64
  z0::Float64      # local z of first node
  dz::Float64
end

@inline function rz_locate(g::RZGrid, r, z)
  u = (z - g.z0) / g.dz
  (u >= 0 && u <= g.nz - 1) || return (-1, 0, 0.0, 0.0)
  v = r / g.dr
  v <= g.nr - 1 || return (-1, 0, 0.0, 0.0)
  iz = min(unsafe_trunc(Int, u), g.nz - 2)
  ir = min(unsafe_trunc(Int, v), g.nr - 2)
  return (iz, ir, u - iz, v - ir)
end

@inline function rz_interp(buf, g::RZGrid, ncomp, c, iz, ir, fz, fr)
  n00 = g.off + (iz * g.nr + ir) * ncomp + c
  n01 = n00 + ncomp                     # ir + 1
  n10 = n00 + g.nr * ncomp              # iz + 1
  n11 = n10 + ncomp
  @inbounds return (buf[n00] * (1 - fz) * (1 - fr) + buf[n10] * fz * (1 - fr) +
                    buf[n11] * fz * fr + buf[n01] * (1 - fz) * fr)
end

"Static solenoid field map Br(r,z), Bz(r,z) (IMPACT-T type 3)."
struct SolRZField <: FieldModel
  pl::Placement
  g::RZGrid       # components: Br, Bz
  scale::Float64
end

@inline function field(e::SolRZField, buf, x, y, z, t)
  l = to_local(e.pl, x, y, z)
  r = sqrt(l[1]^2 + l[2]^2)
  iz, ir, fz, fr = rz_locate(e.g, r, l[3])
  iz < 0 && return ZERO3, ZERO3
  br = e.scale * rz_interp(buf, e.g, 2, 1, iz, ir, fz, fr)
  bz = e.scale * rz_interp(buf, e.g, 2, 2, iz, ir, fz, fr)
  rr = max(r, 1.0e-12)
  B = V3(br * l[1] / rr, br * l[2] / rr, bz)
  return ZERO3, to_lab(e.pl, B)
end

"TM-mode RF field map Ez(r,z), Er(r,z), Bθ(r,z) (e.g. from SUPERFISH; IMPACT-T type 112)."
struct CylRZField <: FieldModel
  pl::Placement
  g::RZGrid       # components: Ez, Er, Bθ
  scale::Float64
  omega::Float64
  phi0::Float64
  sph::Float64
  cph::Float64
end
CylRZField(pl, g, scale, omega, phi0) = CylRZField(pl, g, scale, omega, phi0, sin(phi0), cos(phi0))
time_dependent(::Type{CylRZField}) = true
function set_time(e::CylRZField, t)
  s, c = sincos(e.omega * t + e.phi0)
  return CylRZField(e.pl, e.g, e.scale, e.omega, e.phi0, s, c)
end

@inline function field(e::CylRZField, buf, x, y, z, t)
  l = to_local(e.pl, x, y, z)
  r = sqrt(l[1]^2 + l[2]^2)
  iz, ir, fz, fr = rz_locate(e.g, r, l[3])
  iz < 0 && return ZERO3, ZERO3
  sφ = e.sph; cφ = e.cph
  ez = rz_interp(buf, e.g, 3, 1, iz, ir, fz, fr) * e.scale * cφ
  er = rz_interp(buf, e.g, 3, 2, iz, ir, fz, fr) * e.scale * cφ
  bt = -rz_interp(buf, e.g, 3, 3, iz, ir, fz, fr) * e.scale * sφ
  rr = max(r, 1.0e-12)
  cx = l[1] / rr; cy = l[2] / rr
  E = V3(er * cx, er * cy, ez)
  B = V3(-bt * cy, bt * cx, 0.0)
  return to_lab(e.pl, E), to_lab(e.pl, B)
end

# ---------------------------------------------------------------------------------------
"""
Complex 3D Cartesian field map (IMPACT-T type 111). Each node stores 12 numbers
(Re,Im) of (Ex, Ey, Ez, Bx, By, Bz); the physical field is Re[F e^{i(ωt+θ0)}]·scale.
A static electric field is (E, 0) with ω = 0; a static magnetic field (0, B) with θ0 = -90°.
"""
struct Cart3DField <: FieldModel
  pl::Placement
  off::Int
  nx::Int; ny::Int; nz::Int
  x0::Float64; y0::Float64; z0::Float64
  dx::Float64; dy::Float64; dz::Float64
  scale::Float64
  omega::Float64
  phi0::Float64
  sph::Float64
  cph::Float64
end
Cart3DField(pl, off, nx, ny, nz, x0, y0, z0, dx, dy, dz, scale, omega, phi0) =
  Cart3DField(pl, off, nx, ny, nz, x0, y0, z0, dx, dy, dz, scale, omega, phi0, sin(phi0), cos(phi0))
time_dependent(::Type{Cart3DField}) = true
function set_time(e::Cart3DField, t)
  s, c = sincos(e.omega * t + e.phi0)
  return Cart3DField(e.pl, e.off, e.nx, e.ny, e.nz, e.x0, e.y0, e.z0, e.dx, e.dy, e.dz, e.scale, e.omega,
                     e.phi0, s, c)
end

@inline function field(e::Cart3DField, buf, x, y, z, t)
  l = to_local(e.pl, x, y, z)
  u = (l[1] - e.x0) / e.dx; v = (l[2] - e.y0) / e.dy; w = (l[3] - e.z0) / e.dz
  ((u >= 0) & (u <= e.nx - 1) & (v >= 0) & (v <= e.ny - 1) & (w >= 0) & (w <= e.nz - 1)) ||
    return ZERO3, ZERO3
  i = min(unsafe_trunc(Int, u), e.nx - 2); fx = u - i
  j = min(unsafe_trunc(Int, v), e.ny - 2); fy = v - j
  k = min(unsafe_trunc(Int, w), e.nz - 2); fz = w - k
  sφ = e.sph; cφ = e.cph
  acc = zero(SVector{12,Float64})
  for dk in 0:1, dj in 0:1, di in 0:1
    wt = (di == 0 ? 1 - fx : fx) * (dj == 0 ? 1 - fy : fy) * (dk == 0 ? 1 - fz : fz)
    base = e.off + (((k + dk) * e.ny + (j + dj)) * e.nx + (i + di)) * 12
    @inbounds acc += wt * SVector{12,Float64}(ntuple(q -> buf[base+q], Val(12)))
  end
  s = e.scale
  E = V3(acc[1] * cφ - acc[2] * sφ, acc[3] * cφ - acc[4] * sφ, acc[5] * cφ - acc[6] * sφ) * s
  B = V3(acc[7] * cφ - acc[8] * sφ, acc[9] * cφ - acc[10] * sφ, acc[11] * cφ - acc[12] * sφ) * s
  return to_lab(e.pl, E), to_lab(e.pl, B)
end

# ---------------------------------------------------------------------------------------
"""
Quadrupole (IMPACT-T type 1). Gradient profile `kind`:
0 = hard edge, 1 = Enge fringe (IMPACT-T analytic form, fringe size set by the aperture),
2 = tabulated g(z), g'(z), g''(z). Field to third order:

    Bx = g y - g''(y³ + 3x²y)/12,  By = g x - g''(x³ + 3xy²)/12,  Bz = g' x y

If `rf_omega > 0` the field is multiplied by cos(rf_omega t + rf_phase) (RF quadrupole).
"""
struct QuadField <: FieldModel
  pl::Placement
  g0::Float64       # gradient [T/m]
  kind::Int
  L::Float64        # element length
  Leff::Float64     # effective length (Enge profile)
  dd::Float64       # aperture diameter (Enge profile)
  tab::AxisTable
  rf_omega::Float64
  rf_phase::Float64
  cph::Float64      # cos(rf_omega t + rf_phase) at the current time
end
QuadField(pl, g0, kind, L, Leff, dd, tab, rf_omega, rf_phase) =
  QuadField(pl, g0, kind, L, Leff, dd, tab, rf_omega, rf_phase, cos(rf_phase))
time_dependent(::Type{QuadField}) = true
set_time(e::QuadField, t) = e.rf_omega > 0 ?
  QuadField(e.pl, e.g0, e.kind, e.L, e.Leff, e.dd, e.tab, e.rf_omega, e.rf_phase, cos(e.rf_omega * t + e.rf_phase)) : e

@inline function quad_profile(e::QuadField, buf, z)
  if e.kind == 0
    return e.g0, 0.0, 0.0
  elseif e.kind == 2
    g, gp, gpp, _ = axis_eval(buf, e.tab, z)
    return e.g0 * g, e.g0 * gp, e.g0 * gpp
  else
    c1 = -0.00004; c2 = 4.518219
    dd = e.dd; bb = e.g0
    z10 = (e.L - e.Leff) / 2
    z20 = e.L - z10
    z3 = z10 + 1.5dd
    z4 = z20 - 1.5dd
    if z < z3
      s1 = c1 + c2 * (-(z - z10) / dd); s2 = -c2 / dd
    elseif z > z4
      s1 = c1 + c2 * ((z - z20) / dd); s2 = c2 / dd
    else
      return bb, 0.0, 0.0
    end
    ex = exp(s1)
    den = 1 + ex
    g = bb / den
    tmp1 = ex * s2
    gp = -bb * tmp1 / den^2
    gpp = bb * (-ex * s2 * s2 / den^2 + 2 * tmp1 * tmp1 / den^3)
    return g, gp, gpp
  end
end

@inline function field(e::QuadField, buf, x, y, z, t)
  l = to_local(e.pl, x, y, z)
  g, gp, gpp = quad_profile(e, buf, l[3])
  lx = l[1]; ly = l[2]
  B = V3(g * ly - gpp * (ly^3 + 3 * lx^2 * ly) / 12,
         g * lx - gpp * (lx^3 + 3 * lx * ly^2) / 12,
         gp * lx * ly)
  if e.rf_omega > 0
    B = B * e.cph
  end
  return ZERO3, to_lab(e.pl, B)
end

"Hard edge sextupole (order 2), octupole (3) or decapole (4) (IMPACT-T type 5)."
struct MultipoleField <: FieldModel
  pl::Placement
  order::Int
  strength::Float64   # T/m^order
end

@inline function field(e::MultipoleField, buf, x, y, z, t)
  l = to_local(e.pl, x, y, z)
  a = l[1]; b = l[2]; s = e.strength
  if e.order == 2
    bx = s * a * b; by = s * (a^2 - b^2) / 2
  elseif e.order == 3
    bx = s * (3a^2 * b - b^3) / 6; by = s * (a^3 - 3a * b^2) / 6
  else
    bx = s * (a^3 * b - a * b^3) / 6; by = s * (a^4 - 6a^2 * b^2 + b^4) / 24
  end
  return ZERO3, to_lab(e.pl, V3(bx, by, 0.0))
end

"Linear 3D constant focusing electric field E = -(kx² x, ky² y, kz² z) (IMPACT-T type 2)."
struct ConstFocField <: FieldModel
  pl::Placement
  kx2::Float64; ky2::Float64; kz2::Float64
end

@inline function field(e::ConstFocField, buf, x, y, z, t)
  l = to_local(e.pl, x, y, z)
  return to_lab(e.pl, V3(-e.kx2 * l[1], -e.ky2 * l[2], -e.kz2 * l[3])), ZERO3
end

# ---------------------------------------------------------------------------------------
# Analytic test fields (IMPACT-T type 113)

"Alpha magnet: Bx = 0, By = scale·z, Bz = scale·y (local coordinates)."
struct AlphaMagnetField <: FieldModel
  pl::Placement
  scale::Float64
end
@inline function field(e::AlphaMagnetField, buf, x, y, z, t)
  l = to_local(e.pl, x, y, z)
  # IMPACT-T uses the lab z here; the element local z is used for a placement independent field
  return ZERO3, to_lab(e.pl, V3(0.0, e.scale * (l[3] + e.pl.s0), e.scale * l[2]))
end

"Traveling step wave between meander plates (vertical electric field)."
struct MeanderWaveField <: FieldModel
  pl::Placement
  escale::Float64
  t0::Float64       # wave start time
  t1::Float64       # wave end time
  z0::Float64       # wave start location (lab)
  zslope::Float64   # rising length
  vz::Float64       # wave speed
end
@inline function field(e::MeanderWaveField, buf, x, y, z, t)
  tt = t - e.t0
  tt1 = e.t1 - e.t0
  if tt >= 0 && tt <= tt1
    z0t = e.z0 + e.vz * tt
    z1 = z0t - e.zslope; z2 = z0t + e.zslope
    ey = (z >= z1 && z <= z2) ? e.escale / e.zslope * (z - z0t) : (z > z2 ? e.escale : -e.escale)
  elseif tt < 0
    ey = e.escale
  else
    ey = -e.escale
  end
  return V3(0.0, ey, 0.0), ZERO3
end

"DC field above a sinusoidally rough cathode surface."
struct SurfaceRoughnessField <: FieldModel
  pl::Placement
  escale::Float64
  an::Float64      # relative amplitude
  wkn::Float64     # wave number
end
@inline function field(e::SurfaceRoughnessField, buf, x, y, z, t)
  l = to_local(e.pl, x, y, z)
  d = e.escale * e.an * e.wkn * exp(-e.wkn * l[3])
  s, c = sincos(e.wkn * l[1])
  return V3(d * s, 0.0, e.escale + d * c), ZERO3
end

# ---------------------------------------------------------------------------------------
"""
Pole face geometry of an IMPACT-T bending magnet in its local (x, z) frame. The faces
are the lines z = kᵢ x + bᵢ (i = 1..4); between faces 1,2 and 3,4 are the entrance and
exit fringe regions, between 2 and 3 the uniform field region. The fringe profile is an
Enge function of the distance from face 1 (entrance) or face 3 (exit).
"""
struct PoleFaces
  k::SVector{4,Float64}
  b::SVector{4,Float64}
  dd1::Float64          # entrance Enge shift
  dd2::Float64          # exit Enge shift
  c::SVector{8,Float64} # Enge coefficients
end

@inline function enge(pf::PoleFaces, tmpz, dsign, dd)
  c = pf.c
  s1 = c[1] + tmpz * (c[2] + tmpz * (c[3] + tmpz * (c[4] + tmpz * (c[5] + tmpz * (c[6] + tmpz * (c[7] + tmpz * c[8]))))))
  ds = (c[2] + tmpz * (2c[3] + tmpz * (3c[4] + tmpz * (4c[5] + tmpz * (5c[6] + tmpz * (6c[7] + tmpz * 7c[8])))))) / dd
  s2 = dsign * ds
  s3 = (2c[3] + tmpz * (6c[4] + tmpz * (12c[5] + tmpz * (20c[6] + tmpz * (30c[7] + tmpz * 42c[8]))))) / dd^2
  ex = exp(s1)
  den = 1 + ex
  bb = 1 / den
  tmp1 = ex * s2
  bbp = -tmp1 / den^2
  s4 = ex * (s3 + s2 * s2)
  bbpp = -s4 / den^2 + 2 * tmp1 * tmp1 / den^3
  return bb, bbp, bbpp
end

"""
Field (Bu, Bv, Bz) of a pole-face magnet where `u` is the bend-plane transverse coordinate
and `v` the coordinate along the main field. `B0` is the uniform field and `dd` the full gap.
"""
@inline function poleface_field(pf::PoleFaces, B0, dd, u, v, z)
  z1 = pf.k[1] * u + pf.b[1]; z2 = pf.k[2] * u + pf.b[2]
  z3 = pf.k[3] * u + pf.b[3]; z4 = pf.k[4] * u + pf.b[4]
  if z >= z1 && z <= z2
    k1 = pf.k[1]; b1 = pf.b[1]
    α = atan(k1)
    xx1 = (u + z * k1 - b1 * k1) / (1 + k1 * k1)
    zz1 = k1 * xx1 + b1
    ss = sqrt((xx1 - u)^2 + (zz1 - z)^2)
    bb, bbp, bbpp = enge(pf, -(ss - pf.dd1) / dd, -1.0, dd)
  elseif z > z2 && z < z3
    return 0.0, B0, 0.0
  elseif z >= z3 && z <= z4
    k1 = pf.k[3]; b1 = pf.b[3]
    α = atan(k1)
    xx1 = (u + z * k1 - b1 * k1) / (1 + k1 * k1)
    zz1 = k1 * xx1 + b1
    ss = sqrt((xx1 - u)^2 + (zz1 - z)^2)
    bb, bbp, bbpp = enge(pf, (ss - pf.dd2) / dd, 1.0, dd)
  else
    return 0.0, 0.0, 0.0
  end
  sα, cα = sincos(α)
  return -B0 * bbp * v * sα, B0 * (bb - bbpp / 2 * v * v), B0 * bbp * v * cα
end

"""
Steering dipole with pole face geometry (IMPACT-T type 113 with sub-type 10). Deflects
horizontally (field along y) or, if `vertical`, vertically (field along x).
"""
struct SteeringDipoleField <: FieldModel
  pl::Placement
  faces::PoleFaces
  B0::Float64
  dd::Float64       # full gap
  vertical::Bool
end
@inline function field(e::SteeringDipoleField, buf, x, y, z, t)
  l = to_local(e.pl, x, y, z)
  if e.vertical
    bu, bv, bz = poleface_field(e.faces, e.B0, e.dd, l[2], l[1], l[3])
    B = V3(bv, bu, bz)
  else
    bu, bv, bz = poleface_field(e.faces, e.B0, e.dd, l[1], l[2], l[3])
    B = V3(bu, bv, bz)
  end
  return ZERO3, to_lab(e.pl, B)
end

"""
Bending magnet (IMPACT-T type 4). Tracked in "bend mode": the bunch is followed in the
magnet's entrance Cartesian frame (local x, z) along with a reference particle.
"""
struct DipoleField <: FieldModel
  pl::Placement
  faces::PoleFaces
  B0::Float64          # vertical field By [T]
  dd::Float64          # full gap
end
@inline function field(e::DipoleField, buf, x, y, z, t)
  bu, bv, bz = poleface_field(e.faces, e.B0, e.dd, x, y, z - e.pl.s0)
  return ZERO3, V3(bu, bv, bz)
end
