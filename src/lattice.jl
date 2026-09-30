# Lattice description with Beamlines.jl.
#
# Elements are ordinary Beamlines `LineElement`s. Standard parameter groups are used
# where they exist (length, multipole strengths, RF frequency/phase, misalignments,
# apertures, bend geometry). Field maps and the other IMPACT-T specific data live in the
# `ImpactParams` parameter group, registered with Beamlines when ImpactTJ is loaded, so
# they can be set as element properties, e.g.
#
#     gun = RFCavity(L=0.15, rf_frequency=1.3e9, phi0=0.3,
#                    field_map=OnAxisFourier(coefs; zstart=0, zend=0.15), field_scale=60e6)
#
# The field of an element can extend beyond its body (`field_extent`, local coordinates)
# so overlapping fields (e.g. a solenoid around a gun) are modeled without superposition
# bookkeeping: fields of all elements are summed at every particle position.

"""
    ImpactParams

IMPACT-T specific element parameters.

- `field_map`: `OnAxisFourier`, `OnAxisData`, `RZSolenoidMap`, `RZCavityMap` or
  `CartesianMap`. For RF elements the on-axis map is Ez; for solenoids Bz.
- `field_scale`: multiplier of `field_map` (e.g. peak field in V/m for normalized maps).
- `sol_map`, `sol_scale`: static solenoid on-axis Bz superimposed on the element
  (IMPACT-T SolRF element).
- `field_extent`: `(zmin, zmax)` in local coordinates where the field exists; defaults
  to `(0, L)`.
- `quad_profile`: `:hard`, `:enge` (fringe of IMPACT-T with effective length
  `quad_leff`) or an `OnAxisData` gradient table (normalized, multiplied by `Bn1`).
- `rf_quad_frequency` [Hz], `rf_quad_phase` [rad]: RF quadrupole.
- `const_focus`: `(kx², ky², kz²)` [V/m²] of a linear focusing electric field.
- `analytic_field`: an analytic test field (see `AlphaMagnet`, `MeanderWave`,
  `SurfaceRoughness`, `SteeringDipole`).
- `pole_faces`, `bend_gap`: bending magnet geometry and full gap [m] for `SBend`s.
- `wake`: short range structure wake (`BaneWake`, `TabulatedWake`, ...) applied while the
  bunch centroid is inside the element's field extent.
- `impact_event`: event executed when the bunch centroid passes a `Marker`.
- `body_length`: physical length of the element if different from its length `L` in the
  Beamline (used for elements whose body overlaps following elements, e.g. translated
  IMPACT-T decks). Defaults to `L`.
"""
Base.@kwdef mutable struct ImpactParams <: AbstractParams
  field_map = nothing
  field_scale = 1.0
  sol_map = nothing
  sol_scale = 1.0
  field_extent = nothing
  quad_profile = :hard
  quad_leff = NaN
  rf_quad_frequency = 0.0
  rf_quad_phase = 0.0
  const_focus = nothing
  analytic_field = nothing
  pole_faces = nothing
  bend_gap = 0.0
  wake = nothing
  impact_event = nothing
  body_length = NaN
end

const IMPACT_PROPS = fieldnames(ImpactParams)

function _register_beamlines_params()
  Beamlines.PARAMS_MAP[:ImpactParams] = ImpactParams
  for k in IMPACT_PROPS
    if haskey(Beamlines.PROPERTIES_MAP, k) && Beamlines.PROPERTIES_MAP[k] !== ImpactParams
      error("ImpactTJ property $k conflicts with an existing Beamlines property")
    end
    Beamlines.PROPERTIES_MAP[k] = ImpactParams
  end
end

# Analytic test field specifications -----------------------------------------------------
"Alpha magnet field Bx = 0, By = strength·z, Bz = strength·y [T/m]."
struct AlphaMagnet; strength::Float64; end
"Traveling step field between meander plates."
Base.@kwdef struct MeanderWave; escale::Float64; t_start::Float64; t_end::Float64
  z_start::Float64; rise::Float64; speed::Float64; end
"DC field above a sinusoidally rough surface."
Base.@kwdef struct SurfaceRoughness; escale::Float64; amplitude::Float64; wavenumber::Float64; end
"Steering dipole with pole face geometry; `vertical` deflection if true."
Base.@kwdef struct SteeringDipole; B0::Float64; gap::Float64; faces::PoleFaceGeometry; vertical::Bool = false; end

# ---------------------------------------------------------------------------------------
# Compiled lattice

"Aperture of an element region (IMPACT-T pipe radius)."
struct ApertureSpec
  zmin::Float64
  zmax::Float64
  radius::Float64
end

"A bending magnet tracked in bend mode."
struct DipoleSpec
  index::Int           # position in the lattice (unique id)
  name::String
  s0::Float64          # lab z of the magnet (local frame) origin
  s1::Float64          # end of the element region
  field::DipoleField
  geom::PoleFaceGeometry
end

"Structure wake active while the bunch centroid is in [zstart, zend]."
struct WakeRegion
  zstart::Float64
  zend::Float64
  model::WakeModel
end

struct CompiledLattice
  fields::FieldSet
  dipoles::Vector{DipoleSpec}
  apertures::Vector{ApertureSpec}
  wakes::Vector{WakeRegion}
  events::Vector{Any}
  s_end::Float64
  regions::Vector{Tuple{String,Float64,Float64}}   # (name, zmin, zmax) of every element
end

_get(ele, key, default) = (v = getproperty(ele, key); isnothing(v) ? default : v)

function _impact(ele)
  p = getproperty(ele, :ImpactParams)
  return isnothing(p) ? ImpactParams() : p
end

function _placement(ele, s, L, zmin, zmax)
  al = getproperty(ele, :AlignmentParams)
  isnothing(al) && return Placement(zmin, zmax, s)
  return Placement(zmin, zmax, s, L; x_offset=Float64(al.x_offset), y_offset=Float64(al.y_offset),
                   z_offset=Float64(al.z_offset), x_rot=Float64(al.x_rot), y_rot=Float64(al.y_rot),
                   tilt=Float64(al.tilt))
end

function _axis_table(bb::BufferBuilder, m::OnAxisFourier)
  fourier_axis_table(bb, FourierSeries(m.coefs, m.zlen, m.zc), m.zstart, m.zend)
end
function _axis_table(bb::BufferBuilder, m::OnAxisData)
  span = (length(m.f) - 1) * m.dz
  axis_table(bb, m.z0, m.dz, m.f, m.fp, m.fpp, m.fppp; period=m.periodic ? span : 0.0,
             hermite=m.interpolation == :hermite)
end

function _rz_grid(bb::BufferBuilder, rmax, z0, z1, comps)
  nr, nz = size(first(comps))
  data = Vector{Float64}(undef, nr * nz * length(comps))
  n = 0
  for iz in 1:nz, ir in 1:nr, c in comps
    n += 1
    data[n] = c[ir, iz]
  end
  off = push_data!(bb, data)
  return RZGrid(off, nr, nz, rmax / (nr - 1), z0, (z1 - z0) / (nz - 1))
end

function _smooth_step_solenoid(L, Bsol, aperture)
  # Hard edge solenoids need fringe fields to focus in a time domain code. The field is
  # modeled with tanh edges of width w (≈ aperture radius, at most L/10).
  w = max(min(aperture > 0 ? aperture : L / 10, L / 10), 1.0e-6)
  zs = range(-4w, L + 4w; length=4001)
  f(z) = 0.5Bsol * (tanh(z / w) - tanh((z - L) / w))
  vals = f.(zs)
  OnAxisData(first(zs), step(zs), vals; interpolation=:hermite), (-4w, L + 4w)
end

"""
    compile_lattice(bl::Beamline; s_end=nothing) -> CompiledLattice

Convert a Beamlines `Beamline` into ImpactTJ's field models, apertures, bends, wakes and
events. The lab z coordinate is the Beamlines `s` of each element.
"""
function compile_lattice(bl::Beamline; s_end=nothing, s_offset=0.0)
  bb = BufferBuilder()
  models = FieldModel[]
  dipoles = DipoleSpec[]
  apertures = ApertureSpec[]
  wakes = WakeRegion[]
  events = Any[]
  regions = Tuple{String,Float64,Float64}[]
  s_last = 0.0
  for ele in bl.line
    s = Float64(ele.s) + s_offset
    Lplace = Float64(ele.L)
    kind = String(ele.kind)
    name = String(ele.name)
    ip = _impact(ele)
    L = isnan(ip.body_length) ? Lplace : Float64(ip.body_length)
    ext = isnothing(ip.field_extent) ? (0.0, L) : Tuple(Float64.(ip.field_extent))
    zmin = s + ext[1]; zmax = s + ext[2]
    push!(regions, (name, zmin, zmax))
    s_last = s + Lplace
    pl = _placement(ele, s, L, zmin, zmax)

    # aperture
    ap = getproperty(ele, :ApertureParams)
    elem_rad = 0.0
    if !isnothing(ap) && ap.aperture_active
      rad = minimum(abs.(Float64.((ap.x1_limit, ap.x2_limit, ap.y1_limit, ap.y2_limit))))
      if isfinite(rad) && rad > 0
        elem_rad = rad
        push!(apertures, ApertureSpec(zmin, zmax, rad))
      end
    end

    # events
    isnothing(ip.impact_event) || push!(events, with_z(ip.impact_event, s))

    # wakes
    isnothing(ip.wake) || push!(wakes, WakeRegion(zmin, zmax, ip.wake))

    # bends (bend mode)
    if kind == "SBend"
      isnothing(ip.pole_faces) && error("SBend $name needs `pole_faces` (PoleFaceGeometry)")
      pf = ip.pole_faces
      By = Float64(_get(ele, :Bn0, 0.0))
      faces = PoleFaces(SVector(pf.k), SVector(pf.b), pf.dd1, pf.dd2, SVector(pf.enge))
      push!(dipoles, DipoleSpec(length(regions), name, s, s + L, DipoleField(Placement(zmin, zmax, s), faces, By,
                                                              Float64(ip.bend_gap)), pf))
      continue
    end

    # multipoles
    bm = getproperty(ele, :BMultipoleParams)
    if !isnothing(bm) && kind != "Solenoid"
      for order in 1:4
        bn = Float64(_get(ele, Symbol("Bn$order"), 0.0))
        bs = Float64(_get(ele, Symbol("Bs$order"), 0.0))
        (bn == 0 && bs == 0) && continue
        # skew components are modeled by rotating the multipole
        tiltm = atan(-bs, bn) / (order + 1)
        str = hypot(bn, bs)
        plm = tiltm == 0 ? pl : _placement_tilted(ele, s, L, zmin, zmax, tiltm)
        if order == 1
          qp = ip.quad_profile
          if qp isa OnAxisData
            tab = _axis_table(bb, qp)
            push!(models, QuadField(plm, str, 2, L, L, 0.0, tab, 2π * ip.rf_quad_frequency, ip.rf_quad_phase))
          else
            kindq = qp == :enge ? 1 : 0
            push!(models, QuadField(plm, str, kindq, L, isnan(ip.quad_leff) ? L : ip.quad_leff,
                                    2elem_rad, AxisTable(), 2π * ip.rf_quad_frequency, ip.rf_quad_phase))
          end
        else
          push!(models, MultipoleField(plm, order, str))
        end
      end
    end

    # solenoid without map: smooth hard edge model
    if kind == "Solenoid" && isnothing(ip.field_map)
      Bsol = Float64(_get(ele, :Bsol, 0.0))
      if Bsol != 0
        m, e2 = _smooth_step_solenoid(L, Bsol, elem_rad)
        tab = _axis_table(bb, m)
        push!(models, SolAxisField(_placement(ele, s, L, s + e2[1], s + e2[2]), tab, 1.0))
      end
    end

    # field maps
    fm = ip.field_map
    rf = getproperty(ele, :RFParams)
    ω = 0.0; φ0 = 0.0
    if !isnothing(rf)
      f = Float64(_get(ele, :rf_frequency, 0.0))
      ω = 2π * f
      φ0 = Float64(rf.phi0)
    end
    if fm isa Union{OnAxisFourier,OnAxisData}
      tab = _axis_table(bb, fm)
      if kind == "Solenoid"
        push!(models, SolAxisField(pl, tab, Float64(ip.field_scale)))
      else
        push!(models, RFAxisField(pl, tab, Float64(ip.field_scale), ω, φ0))
      end
    elseif fm isa RZSolenoidMap
      g = _rz_grid(bb, fm.rmax, fm.z0, fm.z1, (fm.Br, fm.Bz))
      push!(models, SolRZField(pl, g, Float64(ip.field_scale)))
    elseif fm isa RZCavityMap
      g = _rz_grid(bb, fm.rmax, fm.z0, fm.z1, (fm.Ez, fm.Er, fm.Btheta))
      push!(models, CylRZField(pl, g, Float64(ip.field_scale), ω, φ0))
    elseif fm isa CartesianMap
      _, nx, ny, nz = size(fm.E)
      data = Vector{Float64}(undef, 12nx * ny * nz)
      n = 0
      for k in 1:nz, j in 1:ny, i in 1:nx
        for c in 1:3
          data[n+1] = real(fm.E[c, i, j, k]); data[n+2] = imag(fm.E[c, i, j, k]); n += 2
        end
        for c in 1:3
          data[n+1] = real(fm.B[c, i, j, k]); data[n+2] = imag(fm.B[c, i, j, k]); n += 2
        end
      end
      off = push_data!(bb, data)
      dx = (fm.xlim[2] - fm.xlim[1]) / (nx - 1); dy = (fm.ylim[2] - fm.ylim[1]) / (ny - 1)
      dz = (fm.zlim[2] - fm.zlim[1]) / (nz - 1)
      push!(models, Cart3DField(pl, off, nx, ny, nz, fm.xlim[1], fm.ylim[1], fm.zlim[1], dx, dy, dz,
                                Float64(ip.field_scale), ω, φ0))
    elseif !isnothing(fm)
      error("Unsupported field map $(typeof(fm)) in element $name")
    end
    if !isnothing(ip.sol_map)
      tab = _axis_table(bb, ip.sol_map)
      push!(models, SolAxisField(pl, tab, Float64(ip.sol_scale)))
    end

    # constant focusing and analytic fields
    if !isnothing(ip.const_focus)
      k2 = Float64.(ip.const_focus)
      push!(models, ConstFocField(pl, k2[1], k2[2], k2[3]))
    end
    af = ip.analytic_field
    if af isa AlphaMagnet
      push!(models, AlphaMagnetField(pl, af.strength))
    elseif af isa MeanderWave
      push!(models, MeanderWaveField(pl, af.escale, af.t_start, af.t_end, af.z_start, af.rise, af.speed))
    elseif af isa SurfaceRoughness
      push!(models, SurfaceRoughnessField(pl, af.escale, af.amplitude, af.wavenumber))
    elseif af isa SteeringDipole
      pf = af.faces
      faces = PoleFaces(SVector(pf.k), SVector(pf.b), pf.dd1, pf.dd2, SVector(pf.enge))
      push!(models, SteeringDipoleField(pl, faces, af.B0, af.gap, af.vertical))
    end
  end
  fs = FieldSet(models, bb.data)
  sort!(apertures, by=a -> a.zmin)
  sort!(events, by=event_z)
  return CompiledLattice(fs, dipoles, apertures, wakes, events,
                         isnothing(s_end) ? s_last : Float64(s_end), regions)
end

function _placement_tilted(ele, s, L, zmin, zmax, tiltm)
  al = getproperty(ele, :AlignmentParams)
  if isnothing(al)
    return Placement(zmin, zmax, s, L; tilt=tiltm)
  end
  return Placement(zmin, zmax, s, L; x_offset=Float64(al.x_offset), y_offset=Float64(al.y_offset),
                   z_offset=Float64(al.z_offset), x_rot=Float64(al.x_rot), y_rot=Float64(al.y_rot),
                   tilt=Float64(al.tilt) + tiltm)
end
