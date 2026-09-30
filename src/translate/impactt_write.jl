# Translation from ImpactTJ to IMPACT-T files, and readers for IMPACT-T output files.

_f(x) = @sprintf("%.15g", x)

"""
    write_impactt_particles(file, bin)

Write a bin in the 9 column IMPACT-T particle format (distribution type 166):
x[m] γβx y[m] γβy z[m] γβz q/m[1/eV] charge[C] id.
"""
function write_impactt_particles(file::AbstractString, b::ParticleBin)
  r = Array(b.r); w = Array(b.w); id = Array(b.id)
  open(file, "w") do io
    println(io, size(r, 1))
    for p in axes(r, 1)
      @printf(io, "%.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %d\n", r[p, 1], r[p, 2], r[p, 3],
              r[p, 4], r[p, 5], r[p, 6], b.charge / b.mc2, w[p], id[p])
    end
  end
  return file
end

function _write_numbers(file, v)
  open(file, "w") do io
    for x in v
      println(io, _f(x))
    end
  end
end

function _write_solrf_fourier(file, ez::Union{Nothing,OnAxisFourier}, bz::Union{Nothing,OnAxisFourier})
  v = Float64[]
  for m in (ez, bz)
    if isnothing(m)
      append!(v, [1.0, 0.0, 0.0, 0.0, 0.0])
    else
      abs(m.zc - (m.zstart + m.zlen / 2)) < 1e-12 ||
        error("IMPACT-T SolRF Fourier data requires zc = zstart + zlen/2")
      append!(v, [length(m.coefs), m.zstart, m.zend, m.zlen]); append!(v, m.coefs)
    end
  end
  _write_numbers(file, v)
end

function _write_discrete(file, blocks)
  open(file, "w") do io
    for m in blocks
      if isnothing(m)
        println(io, "1 0.0 0.0"); println(io, "0.0 0.0 0.0 0.0")
        continue
      end
      n = length(m.f)
      println(io, "$n $(_f(m.z0)) $(_f(m.z0 + (n - 1) * m.dz))")
      for i in 1:n
        println(io, "$(_f(m.f[i])) $(_f(m.fp[i])) $(_f(m.fpp[i])) $(_f(m.fppp[i]))")
      end
    end
  end
end

function _write_t7_cyl(file, m::RZCavityMap)
  nr, nz = size(m.Ez)
  open(file, "w") do io
    println(io, "$(_f(m.z0*100)) $(_f(m.z1*100)) $(nz-1)")
    println(io, "0.0")
    println(io, "0.0 $(_f(m.rmax*100)) $(nr-1)")
    for j in 1:nr, i in 1:nz
      e = hypot(m.Ez[j, i], m.Er[j, i])
      println(io, "$(_f(m.Ez[j,i]/1e6)) $(_f(m.Er[j,i]/1e6)) $(_f(e/1e6)) $(_f(m.Btheta[j,i]/MU0_SI))")
    end
  end
end

function _write_t7_sol(file, m::RZSolenoidMap)
  nr, nz = size(m.Br)
  open(file, "w") do io
    println(io, "0.0 $(_f(m.rmax*100)) $(nr-1)")
    println(io, "$(_f(m.z0*100)) $(_f(m.z1*100)) $(nz-1)")
    for j in 1:nz, i in 1:nr
      println(io, "$(_f(m.Br[i,j])) $(_f(m.Bz[i,j]))")
    end
  end
end

function _write_t7_cart(file, m::CartesianMap)
  _, nx, ny, nz = size(m.E)
  c(z) = "($(_f(real(z))),$(_f(imag(z))))"
  open(file, "w") do io
    println(io, "$(_f(m.xlim[1])) $(_f(m.xlim[2])) $(nx-1)")
    println(io, "$(_f(m.ylim[1])) $(_f(m.ylim[2])) $(ny-1)")
    println(io, "$(_f(m.zlim[1])) $(_f(m.zlim[2])) $(nz-1)")
    for k in 1:nz, j in 1:ny, i in 1:nx
      println(io, join((c(m.E[1, i, j, k]), c(m.E[2, i, j, k]), c(m.E[3, i, j, k]),
                        c(m.B[1, i, j, k]), c(m.B[2, i, j, k]), c(m.B[3, i, j, k])), " "))
    end
  end
end

function _write_dipole(file, g::PoleFaceGeometry)
  v = [g.csr ? 1.0 : 0.0, g.gamma_ref, g.k[1], g.b[1], g.k[2], g.b[2], g.k[3], g.b[3], g.k[4], g.b[4],
       2g.dd1, 2g.dd2, g.enge..., g.s_start, g.s_end]
  _write_numbers(file, v)
end

_rad(ele) = (ap = getproperty(ele, :ApertureParams);
             isnothing(ap) ? 0.0 : (r = minimum(abs.(Float64.((ap.x1_limit, ap.x2_limit, ap.y1_limit, ap.y2_limit)))); isfinite(r) ? r : 0.0))

function _impact_misalign(ele, L)
  al = getproperty(ele, :AlignmentParams)
  isnothing(al) && return zeros(5)
  xr, yr, tl = Float64(al.x_rot), Float64(al.y_rot), Float64(al.tilt)
  W = rotz(tl) * roty(yr) * rotx(xr)
  pc = V3(al.x_offset, al.y_offset, al.z_offset + L / 2)
  d = pc - W * V3(0, 0, L / 2)
  return [d[1], d[2], xr, -yr, tl]
end

"""
    write_impactt(dir, bl, settings, bins; events=[], wakes=[])
    write_impactt(dir, sim::Simulation)

Write an IMPACT-T input deck (ImpactT.in, ImpactT<i>.in for several bins, rfdata*,
1T*.T7 field files and partcl*.data particle files) equivalent to an ImpactTJ setup. The
particles are written explicitly (distribution type 166) so the IMPACT-T run starts from
exactly the same macroparticles.
"""
function write_impactt(dir::AbstractString, bl::Beamline, s::Settings, bins::AbstractVector{<:ParticleBin};
                       events=Event[], wakes=WakeRegion[])
  mkpath(dir)
  lines = String[]
  fid = Ref(0)
  newid(base=0) = (fid[] += 1; base + fid[])
  push!(lines, "! ======= machine description (written by ImpactTJ) =======")
  anyerr = false
  for ele in bl.line
    ip = _impact(ele)
    Lp = Float64(ele.L)
    L = isnan(ip.body_length) ? Lp : Float64(ip.body_length)
    z = Float64(ele.s) + s.s_offset
    kind = String(ele.kind)
    rad = _rad(ele)
    fm = ip.field_map
    rfp = getproperty(ele, :RFParams)
    freq = isnothing(rfp) ? 0.0 : Float64(_get(ele, :rf_frequency, 0.0))
    ph = isnothing(rfp) ? 0.0 : Float64(rfp.phi0) * 180 / π
    mis = _impact_misalign(ele, L)
    any(!=(0), mis) && (anyerr = true)
    if !isnothing(ip.field_extent) && Tuple(Float64.(ip.field_extent)) != (0.0, L)
      ext = Tuple(Float64.(ip.field_extent))
      z += ext[1]; L = ext[2] - ext[1]
    end
    if kind == "SBend"
      id = newid(); _write_dipole(joinpath(dir, "rfdata$id"), ip.pole_faces)
      push!(lines, "$(_f(L)) 10 20 4 $(_f(z)) 0.0 $(_f(Float64(_get(ele, :Bn0, 0.0)))) $id $(_f(ip.bend_gap/2)) /")
    elseif kind == "Quadrupole" || (kind in ("Sextupole", "Octupole", "Multipole") && false)
      g = Float64(_get(ele, :Bn1, 0.0))
      f3 = ip.quad_profile == :enge ? ip.quad_leff : 0.0
      if ip.quad_profile isa OnAxisData
        f3 = newid(100); _write_discrete(joinpath(dir, "rfdata$(Int(f3))"), (ip.quad_profile, nothing))
      end
      push!(lines, "$(_f(L)) 10 20 1 $(_f(z)) $(_f(g)) $(_f(f3)) $(_f(rad)) $(join(_f.(mis), " ")) " *
                   "$(_f(ip.rf_quad_frequency)) $(_f(ip.rf_quad_phase*180/π)) /")
    elseif kind in ("Sextupole", "Octupole", "Multipole")
      order, str = 0, 0.0
      for o in 2:4
        b = Float64(_get(ele, Symbol("Bn$o"), 0.0))
        b != 0 && (order = o; str = b)
      end
      push!(lines, "$(_f(L)) 10 20 5 $(_f(z)) $order $(_f(str)) 0.0 $(_f(rad)) $(join(_f.(mis), " ")) /")
    elseif fm isa Union{OnAxisFourier,OnAxisData} || !isnothing(ip.sol_map)
      ez = kind == "Solenoid" ? nothing : fm
      bz = kind == "Solenoid" ? fm : ip.sol_map
      escale = kind == "Solenoid" ? 0.0 : Float64(ip.field_scale)
      bscale = kind == "Solenoid" ? Float64(ip.field_scale) : Float64(ip.sol_scale)
      if (ez isa Union{Nothing,OnAxisFourier}) && (bz isa Union{Nothing,OnAxisFourier})
        id = newid(); _write_solrf_fourier(joinpath(dir, "rfdata$id"), ez, bz)
      else
        id = newid(1000); _write_discrete(joinpath(dir, "rfdata$id"), (ez, bz))
      end
      push!(lines, "$(_f(L)) 10 20 105 $(_f(z)) $(_f(escale)) $(_f(freq)) $(_f(ph)) $id $(_f(rad)) " *
                   "$(join(_f.(mis), " ")) $(_f(bscale)) /")
    elseif fm isa RZCavityMap
      id = newid(); _write_t7_cyl(joinpath(dir, "1T$id.T7"), fm)
      push!(lines, "$(_f(L)) 10 20 112 $(_f(z)) $(_f(ip.field_scale)) $(_f(freq)) $(_f(ph)) $id $(_f(rad)) /")
    elseif fm isa CartesianMap
      id = newid(); _write_t7_cart(joinpath(dir, "1T$id.T7"), fm)
      push!(lines, "$(_f(L)) 10 20 111 $(_f(z)) $(_f(ip.field_scale)) $(_f(freq)) $(_f(ph)) $id $(_f(rad)) /")
    elseif fm isa RZSolenoidMap
      id = newid(); _write_t7_sol(joinpath(dir, "1T$id.T7"), fm)
      push!(lines, "$(_f(L)) 10 20 3 $(_f(z)) $(_f(ip.field_scale)) $id $(_f(rad)) /")
    elseif !isnothing(ip.const_focus)
      k = ip.const_focus
      push!(lines, "$(_f(L)) 10 20 2 $(_f(z)) $(_f(k[1])) $(_f(k[2])) $(_f(k[3])) $(_f(rad)) /")
    elseif !isnothing(ip.analytic_field)
      af = ip.analytic_field
      if af isa AlphaMagnet
        push!(lines, "$(_f(L)) 10 20 113 $(_f(z)) $(_f(af.strength)) 0 0 0 0 0 0 0 -1 /")
      elseif af isa MeanderWave
        push!(lines, "$(_f(L)) 10 20 113 $(_f(z)) $(_f(af.escale)) $(_f(af.t_start)) $(_f(af.t_end)) 0 $(_f(af.z_start)) $(_f(af.rise)) $(_f(af.speed)) 0 -2 /")
      elseif af isa SurfaceRoughness
        push!(lines, "$(_f(L)) 10 20 113 $(_f(z)) $(_f(af.escale)) 0 0 0 0 $(_f(af.amplitude)) $(_f(af.wavenumber)) 0 -3 /")
      elseif af isa SteeringDipole
        id = newid(); _write_dipole(joinpath(dir, "rfdata$id"), af.faces)
        push!(lines, "$(_f(L)) 10 20 113 $(_f(z)) 0 $(_f(af.B0)) 0 $id $(_f(af.gap/2)) $(af.vertical ? 1 : 0) 0 0 10 /")
      end
    elseif kind == "Solenoid"
      @warn "solenoid $(ele.name) without field map written as a drift"
      push!(lines, "$(_f(L)) 10 20 0 $(_f(z)) $(_f(rad)) /")
    else
      push!(lines, "$(_f(L)) 10 20 0 $(_f(z)) $(_f(rad)) /")
    end
    isnothing(ip.impact_event) || push!(events, with_z(ip.impact_event, Float64(ele.s) + s.s_offset))
  end
  # events
  ev(e::Steer) = "0 1 1 -1 $(_f(e.z)) $(_f(e.z)) $(join(_f.(e.dr), " ")) /"
  ev(e::ParticleOutput) = "0 $(e.every) $(_outnum(e.name, 100)) -2 $(_f(e.z)) $(_f(e.z)) $(_f(e.z)) /"
  ev(e::Checkpoint) = "0 1 1000 -3 $(_f(e.z)) $(_f(e.z)) $(_f(e.z)) /"
  ev(e::TimeStepChange) = "0 1 1 -4 $(_f(e.z)) $(_f(e.z)) $(_f(e.z)) $(_f(e.dt)) /"
  ev(e::SpaceChargeSwitch) = "0 1 1 -8 $(_f(e.z)) $(e.on ? 1 : -1) $(_f(e.z)) /"
  ev(e::SliceOutput) = "0 $(e.nslices) $(_outnum(e.name, 200)) -9 $(_f(e.z)) $(_f(e.z)) $(_f(e.z)) /"
  ev(e::Collimator) = "0 1 1 -11 $(_f(e.z)) $(_f(e.z)) $(_f(e.xmin)) $(_f(e.xmax)) $(_f(e.ymin)) $(_f(e.ymax)) $(e.round ? 11 : 1) /"
  ev(e::Heat) = "0 1 1 -16 $(_f(e.z)) $(_f(e.sigma)) /"
  ev(e::RotateZ) = "0 1 1 -17 $(_f(e.z)) $(_f(e.angle)) /"
  maps = LinearMap[]
  function ev(e::LinearMap)
    push!(maps, e)
    "0 1 1 -12 $(_f(e.z)) /"
  end
  for e in sort(collect(events), by=event_z)
    push!(lines, ev(e))
  end
  if !isempty(maps)
    open(joinpath(dir, "linearmap.in"), "w") do io
      sc = C_LIGHT_SI * s.dt
      for m in maps, i in 1:6
        println(io, join((_f(m.R[i, j] / (isodd(i) ? sc : 1.0) * (isodd(j) ? sc : 1.0)) for j in 1:6), " "))
      end
    end
  end
  for w in wakes
    m = w.model
    if m isa BaneWake
      push!(lines, "0 0 1 -6 $(_f(w.zstart)) $(_f(w.zstart)) $(_f(w.zstart)) $(_f(w.zend)) $(_f(m.a)) $(_f(m.g)) $(_f(m.L)) /")
    elseif m isa BTWWake
      push!(lines, "0 0 1 -6 $(_f(w.zstart)) $(_f(w.zstart)) $(_f(w.zstart)) $(_f(w.zend)) 101 $(_f(w.zstart)) 1 /")
    elseif m isa TeslaWake || m isa Tesla3p9Wake
      a = m isa TeslaWake ? -1.0 : -11.0
      push!(lines, "0 0 1 -6 $(_f(w.zstart)) $(_f(w.zstart)) $(_f(w.zstart)) $(_f(w.zend)) $a 1 1 /")
    elseif m isa DielectricWake
      push!(lines, "0 1 1 -13 $(_f(w.zstart)) $(_f(w.zend)) $(m.geometry == :slab ? 0 : 1) $(_f(m.a)) $(_f(m.b)) $(_f(m.eps)) $(_f(m.Lx)) $(m.nkx) $(m.nky) /")
    elseif m isa TabulatedWake
      id = 300 + length(lines)
      open(joinpath(dir, "fort.$id"), "w") do io
        for i in eachindex(m.Wz)
          println(io, "$(_f(m.Wz[i])) $(_f(m.Wx[i])) $(_f(m.Wy[i]))")
        end
      end
      push!(lines, "0 1 $id -6 $(_f(w.zstart)) $(_f(w.zstart)) $(_f(w.zstart)) $(_f(w.zend)) 0 -1 $(_f(m.L)) /")
    end
  end
  isfinite(s.sc_3d_start) && push!(lines, "0 1 1 -5 $(_f(s.sc_3d_start)) $(_f(s.sc_3d_start)) $(_f(s.sc_3d_start)) /")
  isfinite(s.merge_bins_at) && push!(lines, "0 1 1 -7 $(_f(s.merge_bins_at)) $(_f(s.merge_bins_at)) $(_f(s.merge_bins_at)) /")
  s.sc_solver == :nbody && push!(lines, "0 1 1 -15 0 0 $(_f(s.nbody_r0/(C_LIGHT_SI*s.dt))) /")
  isfinite(s.z_stop) && push!(lines, "0 1 1 -99 $(_f(s.z_stop)) $(_f(s.z_stop)) $(_f(s.z_stop)) /")
  # headers, one input file per bin
  nb = length(bins)
  freq = 1.0e9   # reference frequency: only used to convert current to charge
  for (ib, b) in enumerate(bins)
    q = abs(sum(Array(b.w)))
    write_impactt_particles(joinpath(dir, ib == 1 ? "partcl.data" : "partcl$ib.data"), b)
    fn = joinpath(dir, ib == 1 ? "ImpactT.in" : "ImpactT$ib.in")
    open(fn, "w") do io
      println(io, "! IMPACT-T input written by ImpactTJ.jl")
      println(io, "1 1")
      println(io, "$(_f(s.dt)) $(s.max_steps) $nb")
      println(io, "$(s.seed) $(nparticles(b)) 1 $(anyerr ? 1 : 0) $(s.diag_fixed_z ? 2 : 1) $(s.image_charge ? 1 : 0) $(_f(min(s.image_cutoff, 1e10)))")
      println(io, "$(s.grid[1]) $(s.grid[2]) $(s.grid[3]) 1 $(_f(min(s.domain_radius, 1e10))) $(_f(min(s.domain_radius, 1e10))) $(_f(min(s.domain_length, 1e10)))")
      println(io, "166 0 0 $(s.emission_steps > 0 ? s.emission_steps : -1) $(_f(s.emission_time))")
      println(io, "1 0 0 1 1 0 0")
      println(io, "1 0 0 1 1 0 0")
      println(io, "1 0 0 1 1 0 0")
      println(io, "$(_f(q*freq)) $(_f(s.emission_ekin)) $(_f(b.mc2)) $(_f(b.charge)) $(_f(freq)) $(_f(s.t0))")
      for l in lines
        println(io, l)
      end
    end
  end
  return dir
end

_outnum(name, default) = (m = match(r"fort\.(\d+)", name); isnothing(m) ? default : parse(Int, m.captures[1]))

# ---------------------------------------------------------------------------------------
# IMPACT-T output

"""
    read_impactt_stats(dir; bin=1) -> BeamStats

Read the IMPACT-T statistics output (fort.18, fort.24-28) into a `BeamStats` (SI units;
emittances normalized, momenta γβ). With `bin = 2` the per-bunch files of diagnostic
flag 3 (fort.188, 244, 255, 266, 277, 288) are read.
"""
function read_impactt_stats(dir::AbstractString; bin=1)
  rd(n) = (f = joinpath(dir, "fort.$n"); isfile(f) ? [x for x in (_fortran_numbers(l) for l in eachline(f)) if !isempty(x)] : Vector{Float64}[])
  un = bin == 1 ? Dict(n => n for n in 18:28) : Dict(18 => 188, 24 => 244, 25 => 255, 26 => 266, 27 => 277, 28 => 288)
  f18 = rd(un[18]); f24 = rd(un[24]); f25 = rd(un[25]); f26 = rd(un[26]); f27 = rd(un[27]); f28 = rd(un[28])
  n = minimum(length, (f18, f24, f25, f26))
  st = BeamStats()
  for k in 1:n
    a = f18[k]; x = f24[k]; y = f25[k]; z = f26[k]
    push!(st.t, a[1]); push!(st.z_mean, a[2]); push!(st.gamma, a[3]); push!(st.ekin, a[4] * 1e6)
    push!(st.beta, a[5]); push!(st.r_max, a[6]); push!(st.sigma_gamma, a[7])
    push!(st.mean, (x[3], x[5], y[3], y[5], z[2], z[4]))
    sig = zeros(21)
    sig[sigidx(1, 1)] = x[4]^2; sig[sigidx(2, 2)] = x[6]^2; sig[sigidx(1, 2)] = -x[7]
    sig[sigidx(3, 3)] = y[4]^2; sig[sigidx(4, 4)] = y[6]^2; sig[sigidx(3, 4)] = -y[7]
    sig[sigidx(5, 5)] = z[3]^2; sig[sigidx(6, 6)] = z[5]^2; sig[sigidx(5, 6)] = -z[6]
    push!(st.sigma, Tuple(sig))
    push!(st.max_abs, k <= length(f27) ? Tuple(f27[k][3:8]) : ntuple(_ -> 0.0, 6))
    push!(st.moment3, ntuple(_ -> 0.0, 6)); push!(st.moment4, ntuple(_ -> 0.0, 6))
    push!(st.n_particle, k <= length(f28) ? round(Int, f28[k][5]) : 0)
    push!(st.charge, 0.0)
  end
  return st
end

write_impactt(dir::AbstractString, inp::ImpactTInput) =
  write_impactt(dir, inp.beamline, inp.settings, inp.bins; events=inp.events, wakes=inp.wakes)
