# Field map files (HDF5) and translation of a setup into a self-contained ImpactTJ input
# script (the "new format": a Julia script using Beamlines + HDF5 field maps + openPMD
# particles).

"""
    save_fieldmaps(file, maps::AbstractDict{String})

Write field maps (and tabulated wakes) to an HDF5 file, one group per map.
"""
function save_fieldmaps(file::AbstractString, maps::AbstractDict)
  HDF5.h5open(file, "w") do f
    HDF5.attributes(f)["software"] = "ImpactTJ.jl"
    for (name, m) in maps
      g = HDF5.create_group(f, name)
      a = HDF5.attributes(g)
      a["type"] = string(nameof(typeof(m)))
      if m isa OnAxisFourier
        g["coefs"] = m.coefs
        a["zstart"] = m.zstart; a["zend"] = m.zend; a["zlen"] = m.zlen; a["zc"] = m.zc
      elseif m isa OnAxisData
        a["z0"] = m.z0; a["dz"] = m.dz; a["periodic"] = Int(m.periodic)
        a["interpolation"] = string(m.interpolation)
        g["f"] = m.f; g["fp"] = m.fp; g["fpp"] = m.fpp; g["fppp"] = m.fppp
      elseif m isa RZSolenoidMap
        a["rmax"] = m.rmax; a["z0"] = m.z0; a["z1"] = m.z1
        g["Br"] = m.Br; g["Bz"] = m.Bz
      elseif m isa RZCavityMap
        a["rmax"] = m.rmax; a["z0"] = m.z0; a["z1"] = m.z1
        g["Ez"] = m.Ez; g["Er"] = m.Er; g["Btheta"] = m.Btheta
      elseif m isa CartesianMap
        a["xlim"] = collect(m.xlim); a["ylim"] = collect(m.ylim); a["zlim"] = collect(m.zlim)
        g["E_re"] = real.(m.E); g["E_im"] = imag.(m.E); g["B_re"] = real.(m.B); g["B_im"] = imag.(m.B)
      elseif m isa TabulatedWake
        a["L"] = m.L; g["Wz"] = m.Wz; g["Wx"] = m.Wx; g["Wy"] = m.Wy
      else
        error("cannot save $(typeof(m))")
      end
    end
  end
  return file
end

"""
    load_fieldmaps(file) -> Dict{String,Any}

Read the field maps written by `save_fieldmaps`.
"""
function load_fieldmaps(file::AbstractString)
  out = Dict{String,Any}()
  HDF5.h5open(file, "r") do f
    for name in keys(f)
      g = f[name]
      a = HDF5.attributes(g)
      rd(k) = read(a[k])
      t = rd("type")
      out[name] = if t == "OnAxisFourier"
        OnAxisFourier(read(g["coefs"]); zstart=rd("zstart"), zend=rd("zend"), zlen=rd("zlen"), zc=rd("zc"))
      elseif t == "OnAxisData"
        OnAxisData(rd("z0"), rd("dz"), read(g["f"]); fp=read(g["fp"]), fpp=read(g["fpp"]),
                   fppp=read(g["fppp"]), periodic=rd("periodic") == 1,
                   interpolation=Symbol(rd("interpolation")))
      elseif t == "RZSolenoidMap"
        RZSolenoidMap(rd("rmax"), rd("z0"), rd("z1"), read(g["Br"]), read(g["Bz"]))
      elseif t == "RZCavityMap"
        RZCavityMap(rd("rmax"), rd("z0"), rd("z1"), read(g["Ez"]), read(g["Er"]), read(g["Btheta"]))
      elseif t == "CartesianMap"
        CartesianMap(Tuple(rd("xlim")), Tuple(rd("ylim")), Tuple(rd("zlim")),
                     complex.(read(g["E_re"]), read(g["E_im"])), complex.(read(g["B_re"]), read(g["B_im"])))
      elseif t == "TabulatedWake"
        TabulatedWake(rd("L"), read(g["Wz"]), read(g["Wx"]), read(g["Wy"]))
      else
        error("unknown field map type $t in $file")
      end
    end
  end
  return out
end

# --- script generation ---------------------------------------------------------------

_jl(x::Integer) = string(x)
_jl(x::Real) = isinteger(x) && abs(x) < 1e15 ? string(Float64(x)) : repr(Float64(x))
_jl(x::Bool) = string(x)
_jl(x::Symbol) = repr(x)
_jl(x::AbstractString) = repr(x)
_jl(x::Tuple) = "(" * join(_jl.(x), ", ") * (length(x) == 1 ? ",)" : ")")
_jl(x::Species) = "Species($(repr(nameof(x))))"
_jl(x::PoleFaceGeometry) = "PoleFaceGeometry($(_jl(x.k)), $(_jl(x.b)), $(_jl(x.dd1)), $(_jl(x.dd2)), " *
                           "$(_jl(x.enge)), $(_jl(x.gamma_ref)), $(x.csr), $(_jl(x.s_start)), $(_jl(x.s_end)))"
_jl(x::AlphaMagnet) = "AlphaMagnet($(_jl(x.strength)))"
_jl(x::MeanderWave) = "MeanderWave(escale=$(_jl(x.escale)), t_start=$(_jl(x.t_start)), t_end=$(_jl(x.t_end)), " *
                      "z_start=$(_jl(x.z_start)), rise=$(_jl(x.rise)), speed=$(_jl(x.speed)))"
_jl(x::SurfaceRoughness) = "SurfaceRoughness(escale=$(_jl(x.escale)), amplitude=$(_jl(x.amplitude)), wavenumber=$(_jl(x.wavenumber)))"
_jl(x::SteeringDipole) = "SteeringDipole(B0=$(_jl(x.B0)), gap=$(_jl(x.gap)), faces=$(_jl(x.faces)), vertical=$(x.vertical))"
_jl(x::DielectricWake) = "DielectricWake(geometry=$(repr(x.geometry)), a=$(_jl(x.a)), b=$(_jl(x.b)), eps=$(_jl(x.eps)), Lx=$(_jl(x.Lx)), nkx=$(x.nkx), nky=$(x.nky))"
_jl(x::BaneWake) = "BaneWake($(_jl(x.a)), $(_jl(x.g)), $(_jl(x.L)))"
_jl(x::Union{BTWWake,TeslaWake,Tesla3p9Wake}) = "$(nameof(typeof(x)))()"
_jl(x::Matrix{Float64}) = "[" * join((join(_jl.(x[i, :]), " ") for i in axes(x, 1)), "; ") * "]"

function _jl_event(e::Event)
  T = typeof(e)
  args = String[]
  for f in fieldnames(T)
    v = getfield(e, f)
    push!(args, "$f=$(_jl(v))")
  end
  return "$(nameof(T))(" * join(args, ", ") * ")"
end

"""
    write_impacttj(dir, inp::ImpactTInput; name="run.jl")

Write a translated IMPACT-T deck in the ImpactTJ format: `dir/run.jl` (a Julia script
defining the Beamlines lattice, the settings and the events), `dir/fieldmaps.h5` and
`dir/particles_initial.h5` (openPMD). Run it with `julia run.jl`.
"""
function write_impacttj(dir::AbstractString, inp::ImpactTInput; name="run.jl")
  mkpath(dir)
  maps = Dict{String,Any}()
  lines = String[]
  for (k, ele) in enumerate(inp.beamline.line)
    push!(lines, "  " * _jl_element(ele, k, maps) * ",")
  end
  wakelines = String[]
  for (k, w) in enumerate(inp.wakes)
    m = w.model
    if m isa TabulatedWake
      key = "wake$k"; maps[key] = m
      push!(wakelines, "  WakeRegion($(_jl(w.zstart)), $(_jl(w.zend)), fieldmaps[$(repr(key))]),")
    else
      push!(wakelines, "  WakeRegion($(_jl(w.zstart)), $(_jl(w.zend)), $(_jl(m))),")
    end
  end
  save_fieldmaps(joinpath(dir, "fieldmaps.h5"), maps)
  write_particles(joinpath(dir, "particles_initial.h5"), inp.bins, inp.settings.t0)
  s = inp.settings
  sfields = String[]
  d = Settings(dt=s.dt)
  for f in fieldnames(Settings)
    v = getfield(s, f)
    (f != :dt && v == getfield(d, f)) && continue
    f == :output_dir && continue
    push!(sfields, "  $f = $(_jl(v)),")
  end
  sp = first(inp.bins).species
  open(joinpath(dir, name), "w") do io
    println(io, """
    # ImpactTJ input translated from IMPACT-T.
    using ImpactTJ, Beamlines

    const here = @__DIR__
    fieldmaps = ImpactTJ.load_fieldmaps(joinpath(here, "fieldmaps.h5"))

    # Lattice. Elements start at the IMPACT-T element edges; `body_length` is the physical
    # length of elements whose bodies overlap the following elements.
    line = [""")
    foreach(l -> println(io, l), lines)
    println(io, """
    ]
    beamline = Beamline(line; species_ref=$(_jl(sp)))

    settings = Settings(""")
    foreach(l -> println(io, l), sfields)
    println(io, """
      output_dir = joinpath(here, "output"),
    )

    events = Event[""")
    for e in inp.events
      println(io, "  ", _jl_event(e), ",")
    end
    println(io, """
    ]

    wakes = WakeRegion[""")
    foreach(l -> println(io, l), wakelines)
    println(io, """
    ]

    bins, _ = read_particles(joinpath(here, "particles_initial.h5"))
    sim = Simulation(compile_lattice(beamline; s_offset=settings.s_offset, s_end=$(_jl(inp.s_end))),
                     bins, settings; events=events, wakes=wakes)
    run!(sim)""")
  end
  return joinpath(dir, name)
end

function _jl_element(ele, k, maps)
  kind = String(ele.kind)
  args = String["name=$(repr(String(ele.name)))", "L=$(_jl(Float64(ele.L)))"]
  ip = _impact(ele)
  for o in 0:4
    for (pre, sym) in (("Bn", Symbol("Bn$o")), ("Bs", Symbol("Bs$o")))
      bm = getproperty(ele, :BMultipoleParams)
      isnothing(bm) && break
      v = Float64(_get(ele, sym, 0.0))
      v != 0 && push!(args, "$sym=$(_jl(v))")
    end
  end
  rf = getproperty(ele, :RFParams)
  if !isnothing(rf)
    push!(args, "rf_frequency=$(_jl(Float64(_get(ele, :rf_frequency, 0.0))))", "phi0=$(_jl(Float64(rf.phi0)))")
  end
  al = getproperty(ele, :AlignmentParams)
  if !isnothing(al)
    for f in (:x_offset, :y_offset, :z_offset, :x_rot, :y_rot, :tilt)
      v = Float64(getproperty(al, f))
      v != 0 && push!(args, "$f=$(_jl(v))")
    end
  end
  ap = getproperty(ele, :ApertureParams)
  if !isnothing(ap)
    for f in (:x1_limit, :x2_limit, :y1_limit, :y2_limit)
      push!(args, "$f=$(_jl(Float64(getproperty(ap, f))))")
    end
  end
  for f in fieldnames(ImpactParams)
    v = getfield(ip, f)
    dflt = getfield(ImpactParams(), f)
    (v === dflt || (v isa Number && dflt isa Number && (v == dflt || (isnan(v) && isnan(dflt))))) && continue
    if v isa FieldMap
      key = "e$(k)_$(f)"
      maps[key] = v
      push!(args, "$f=fieldmaps[$(repr(key))]")
    elseif f == :quad_profile && v isa OnAxisData
      key = "e$(k)_quad_profile"
      maps[key] = v
      push!(args, "$f=fieldmaps[$(repr(key))]")
    elseif v isa TabulatedWake
      key = "e$(k)_wake"
      maps[key] = v
      push!(args, "$f=fieldmaps[$(repr(key))]")
    elseif v isa Event
      push!(args, "$f=$(_jl_event(v))")
    else
      push!(args, "$f=$(_jl(v))")
    end
  end
  ctor = kind in ("Drift", "Quadrupole", "Sextupole", "Octupole", "Multipole", "Solenoid", "SBend",
                  "RFCavity", "Marker") ? kind : "LineElement"
  ctor == "LineElement" && pushfirst!(args, "kind=$(repr(kind))")
  return "$ctor(" * join(args, ", ") * ")"
end
