# Translation of IMPACT-T input files (ImpactT.in, rfdata*, 1T*.T7, partcl.data) into
# ImpactTJ objects (Beamlines lattice + Settings + particle bins + events).

"Numbers of a Fortran list-directed record: text after '/' or '!' is ignored, 1.0d-3 allowed."
function _fortran_numbers(line::AbstractString)
  s = split(line, '/'; limit=2)[1]
  s = split(s, '!'; limit=2)[1]
  toks = split(replace(s, ',' => ' '))
  vals = Float64[]
  for t in toks
    t2 = replace(t, r"[dD]" => "e")
    v = tryparse(Float64, t2)
    v === nothing && break
    push!(vals, v)
  end
  return vals
end

"Data records of an ImpactT.in file (comment and blank lines removed)."
function _impact_records(file::AbstractString)
  recs = Vector{Float64}[]
  for line in eachline(file)
    st = strip(line)
    (isempty(st) || startswith(st, "!")) && continue
    v = _fortran_numbers(st)
    isempty(v) && continue
    push!(recs, v)
  end
  return recs
end

_pad(v, n) = vcat(v, zeros(max(0, n - length(v))))

"Raw content of one IMPACT-T input file."
struct ImpactTFile
  npcol::Int; nprow::Int
  dt::Float64; ntstep::Int; nbunch::Int
  seed::Int; np::Int; flagmap::Int; flagerr::Int; flagdiag::Int; flagimg::Int; zimage::Float64
  nx::Int; ny::Int; nz::Int; flagbc::Int; xrad::Float64; yrad::Float64; perdlen::Float64
  flagdist::Int; rstartflg::Int; flagsbstp::Int; nemission::Int; temission::Float64
  distparam::Vector{Float64}
  bcurr::Float64; bkenergy::Float64; bmass::Float64; bcharge::Float64; bfreq::Float64; tini::Float64
  elements::Vector{Vector{Float64}}   # L, nseg, mpstp, type, V1..V24
end

function parse_impactt_file(file::AbstractString)
  r = _impact_records(file)
  length(r) >= 9 || error("$file: not an IMPACT-T input file")
  a = _pad(r[1], 2); b = _pad(r[2], 3); c = _pad(r[3], 7); d = _pad(r[4], 7); e = _pad(r[5], 5)
  dp = vcat(_pad(r[6], 7)[1:7], _pad(r[7], 7)[1:7], _pad(r[8], 7)[1:7])
  f = _pad(r[9], 6)
  els = Vector{Float64}[]
  for rec in r[10:end]
    v = _pad(rec, 29)
    push!(els, v)
    round(Int, v[4]) == -99 && break
  end
  ImpactTFile(round(Int, a[1]), round(Int, a[2]), b[1], round(Int, b[2]), max(round(Int, b[3]), 1),
              round(Int, c[1]), round(Int, c[2]), round(Int, c[3]), round(Int, c[4]), round(Int, c[5]),
              round(Int, c[6]), c[7], round(Int, d[1]), round(Int, d[2]), round(Int, d[3]),
              round(Int, d[4]), d[5], d[6], d[7], round(Int, e[1]), round(Int, e[2]), round(Int, e[3]),
              round(Int, e[4]), e[5], dp, f[1], f[2], f[3], f[4], f[5], f[6], els)
end

# ---------------------------------------------------------------------------------------
# Field files

_rfname(dir, id) = joinpath(dir, "rfdata$(id)")
_t7name(dir, id) = joinpath(dir, "1T$(id).T7")

"First number of every line (IMPACT-T `read1t` format)."
function read_rfdata_numbers(file)
  v = Float64[]
  for line in eachline(file)
    x = _fortran_numbers(line)
    isempty(x) || push!(v, x[1])
  end
  return v
end

"""
    read_rfdata_solrf(file) -> (Ez::OnAxisFourier or nothing, Bz::OnAxisFourier or nothing)

IMPACT-T SolRF Fourier file: [n, zstart, zend, zlen, coefs(n)] for Ez then for Bz.
"""
function read_rfdata_solrf(file)
  v = read_rfdata_numbers(file)
  blk(i0) = begin
    n = round(Int, v[i0+1])
    zs, ze, zl = v[i0+2], v[i0+3], v[i0+4]
    c = v[i0+5:min(i0 + 4 + n, length(v))]
    m = (zl > 0 && ze > zs && any(!=(0), c)) ? OnAxisFourier(c; zstart=zs, zend=ze, zlen=zl, zc=zs + zl / 2) : nothing
    m, i0 + 4 + n
  end
  ez, i1 = blk(0)
  bz = nothing
  length(v) >= i1 + 4 && ((bz, _) = blk(i1))
  return ez, bz
end

"""
    read_rfdata_discrete(file; periodic=false) -> Vector{OnAxisData}

IMPACT-T discrete on-axis data: blocks "n zstart zend" followed by n lines (f, f', f'', f''').
"""
function read_rfdata_discrete(file; periodic_first=false)
  lines = [x for x in (_fortran_numbers(l) for l in eachline(file)) if !isempty(x)]
  out = OnAxisData[]
  i = 1
  blk = 0
  while i <= length(lines)
    h = lines[i]
    n = round(Int, h[1]); z0 = _pad(h, 3)[2]; z1 = _pad(h, 3)[3]
    i += 1
    rows = [_pad(lines[j], 4) for j in i:min(i + n - 1, length(lines))]
    i += n
    blk += 1
    length(rows) == 0 && continue
    f = [r[1] for r in rows]; fp = [r[2] for r in rows]; fpp = [r[3] for r in rows]; fppp = [r[4] for r in rows]
    dz = n > 1 ? (z1 - z0) / (n - 1) : 1.0
    push!(out, OnAxisData(z0, dz, f; fp=fp, fpp=fpp, fppp=fppp,
                          periodic=periodic_first && blk == 1, interpolation=:linear))
  end
  return out
end

"IMPACT-T solenoid field map 1T#.T7 (type 3): r and z ranges in cm, then Br, Bz with r fastest."
function read_t7_solenoid(file)
  lines = [x for x in (_fortran_numbers(l) for l in eachline(file)) if !isempty(x)]
  rmin, rmax, nr = lines[1][1] / 100, lines[1][2] / 100, round(Int, lines[1][3])
  zmin, zmax, nz = lines[2][1] / 100, lines[2][2] / 100, round(Int, lines[2][3])
  Br = zeros(nr + 1, nz + 1); Bz = zeros(nr + 1, nz + 1)
  n = 0
  for l in lines[3:end]
    length(l) < 2 && continue
    j = n ÷ (nr + 1) + 1; i = n % (nr + 1) + 1
    j > nz + 1 && break
    Br[i, j] = l[1]; Bz[i, j] = l[2]
    n += 1
  end
  # IMPACT-T places the first z node at the element entrance
  RZSolenoidMap(rmax - rmin, 0.0, zmax - zmin, Br, Bz)
end

"IMPACT-T cylindrical RF map 1T#.T7 (type 112, SUPERFISH output units: cm, MV/m, A/m)."
function read_t7_cylindrical(file)
  lines = [x for x in (_fortran_numbers(l) for l in eachline(file)) if !isempty(x)]
  zmin, zmax, nz = lines[1][1] / 100, lines[1][2] / 100, round(Int, lines[1][3])
  rmin, rmax, nr = lines[3][1] / 100, lines[3][2] / 100, round(Int, lines[3][3])
  Ez = zeros(nr + 1, nz + 1); Er = zeros(nr + 1, nz + 1); Bt = zeros(nr + 1, nz + 1)
  n = 0
  for l in lines[4:end]
    length(l) < 4 && continue
    j = n ÷ (nz + 1) + 1; i = n % (nz + 1) + 1   # z fastest
    j > nr + 1 && break
    Ez[j, i] = l[1] * 1e6; Er[j, i] = l[2] * 1e6; Bt[j, i] = l[4] * MU0_SI
    n += 1
  end
  RZCavityMap(rmax - rmin, 0.0, zmax - zmin, Ez, Er, Bt)
end

function _parse_complex_line(line)
  vals = ComplexF64[]
  for m in eachmatch(r"\(\s*([^,\)]+)\s*,\s*([^\)]+)\s*\)", line)
    re = parse(Float64, replace(strip(m.captures[1]), r"[dD]" => "e"))
    im = parse(Float64, replace(strip(m.captures[2]), r"[dD]" => "e"))
    push!(vals, complex(re, im))
  end
  if isempty(vals)   # plain real numbers
    vals = ComplexF64.(_fortran_numbers(line))
  end
  return vals
end

"IMPACT-T 3D Cartesian complex field map 1T#.T7 (type 111, SI units)."
function read_t7_cartesian(file)
  io = open(file)
  hdr = Vector{Vector{Float64}}()
  while length(hdr) < 3
    x = _fortran_numbers(readline(io))
    isempty(x) || push!(hdr, x)
  end
  xmin, xmax, nx = hdr[1][1], hdr[1][2], round(Int, hdr[1][3])
  ymin, ymax, ny = hdr[2][1], hdr[2][2], round(Int, hdr[2][3])
  zmin, zmax, nz = hdr[3][1], hdr[3][2], round(Int, hdr[3][3])
  E = zeros(ComplexF64, 3, nx + 1, ny + 1, nz + 1); B = zeros(ComplexF64, 3, nx + 1, ny + 1, nz + 1)
  n = 0
  buf = ComplexF64[]
  N = (nx + 1) * (ny + 1) * (nz + 1)
  for line in eachline(io)
    append!(buf, _parse_complex_line(line))
    while length(buf) >= 6 && n < N
      k = n ÷ ((nx + 1) * (ny + 1)); rem = n - k * (nx + 1) * (ny + 1)
      j = rem ÷ (nx + 1); i = rem - j * (nx + 1)
      E[:, i+1, j+1, k+1] .= buf[1:3]; B[:, i+1, j+1, k+1] .= buf[4:6]
      deleteat!(buf, 1:6)
      n += 1
    end
  end
  close(io)
  CartesianMap((xmin, xmax), (ymin, ymax), (zmin, zmax), E, B)
end

"IMPACT-T bend data file (type 4): CSR flag, γ, k1,b1..k4,b4, fringe widths, Enge coefficients, s range."
function read_rfdata_dipole(file)
  v = _pad(read_rfdata_numbers(file), 22)
  PoleFaceGeometry((v[3], v[5], v[7], v[9]), (v[4], v[6], v[8], v[10]), v[11] / 2, v[12] / 2,
                   Tuple(v[13:20]), v[2], v[1] > 0, v[21], v[22])
end

# ---------------------------------------------------------------------------------------
# Particle files

"""
    read_impactt_particles(file, species; mc2, format=:auto, charge=nothing) -> ParticleBin

Read an IMPACT-T particle file: first line = particle count; then either 6 columns
(x[m], γβx, y[m], γβy, z[m], γβz) or 9 columns (…, q/m, charge per macroparticle, id).
For the 6 column format `charge` (total, C) sets the macroparticle charge.
"""
function read_impactt_particles(file::AbstractString, species::Species=Species("electron");
                                mc2=massof(species), qe=chargeof(species), charge=nothing)
  lines = [x for x in (_fortran_numbers(l) for l in eachline(file)) if !isempty(x)]
  n = round(Int, lines[1][1])
  rows = lines[2:min(n + 1, end)]
  N = length(rows)
  r = zeros(N, 6)
  for (i, l) in enumerate(rows)
    r[i, :] .= _pad(l, 6)[1:6]
  end
  if length(rows[1]) >= 9
    w = [l[8] for l in rows]
    id = [round(Int, l[9]) for l in rows]
    return ParticleBin(species, r, w; mc2=mc2, charge=qe, id=id)
  end
  q = isnothing(charge) ? 0.0 : abs(charge) / N * sign(qe)
  return ParticleBin(species, r, q; mc2=mc2, charge=qe)
end

function _species_from(mass, charge)
  if abs(mass - 510998.95) < 5000 && charge == -1
    return Species("electron")
  elseif abs(mass - 510998.95) < 5000 && charge == 1
    return Species("positron")
  elseif abs(mass - 938.272e6) < 2e6 && charge == 1
    return Species("proton")
  elseif abs(mass - 938.272e6) < 2e6 && charge == -1
    return Species("anti-proton")
  else
    return charge < 0 ? Species("electron") : Species("proton")
  end
end

# ---------------------------------------------------------------------------------------

"""
    ImpactTInput

Result of translating an IMPACT-T input deck: a Beamlines `Beamline`, the `Settings`, the
particle bins, events and wake regions. Use `Simulation(inp)` to create the simulation.
"""
struct ImpactTInput
  beamline::Beamline
  settings::Settings
  bins::Vector{ParticleBin}
  events::Vector{Event}
  wakes::Vector{WakeRegion}
  s_end::Float64
  files::Vector{ImpactTFile}
  warnings::Vector{String}
end

Simulation(inp::ImpactTInput; kwargs...) =
  Simulation(compile_lattice(inp.beamline; s_offset=inp.settings.s_offset, s_end=inp.s_end),
             inp.bins, inp.settings; events=inp.events, wakes=inp.wakes, kwargs...)

deg(x) = x * π / 180

function _misalign!(ele, v, i0, flagerr, L)
  # IMPACT-T misalignment (x, y offsets, rotations about x, y, z around the entrance) to
  # Beamlines alignment (rotation W = Rz(tilt)Ry(y_rot)Rx(x_rot) about the element center).
  flagerr == 1 || return
  dx, dy, ax, ay, az = v[4+i0], v[5+i0], v[6+i0], v[7+i0], v[8+i0]
  (dx == 0 && dy == 0 && ax == 0 && ay == 0 && az == 0) && return
  xr, yr, tl = ax, -ay, az
  W = rotz(tl) * roty(yr) * rotx(xr)
  pc = V3(dx, dy, 0.0) + W * V3(0, 0, L / 2)   # center relative to the aligned entrance
  ele.x_offset = pc[1]; ele.y_offset = pc[2]; ele.z_offset = pc[3] - L / 2
  ele.x_rot = xr; ele.y_rot = yr; ele.tilt = tl
end

"""
    read_impactt(path; seed=nothing, rng=nothing) -> ImpactTInput

Translate an IMPACT-T input deck. `path` is either the `ImpactT.in` file or the directory
containing it (with the rfdata / 1T*.T7 / partcl files and, for several bins,
ImpactT2.in ...).
"""
function read_impactt(path::AbstractString; seed=nothing)
  file = isdir(path) ? joinpath(path, "ImpactT.in") : path
  dir = dirname(abspath(file))
  f1 = parse_impactt_file(file)
  warns = String[]
  files = [f1]
  for ib in 2:f1.nbunch
    push!(files, parse_impactt_file(joinpath(dir, "ImpactT$(ib).in")))
  end
  species = _species_from(f1.bmass, f1.bcharge)
  # ---- settings
  s = Settings(dt=f1.dt, max_steps=f1.ntstep, t0=f1.tini, grid=(f1.nx, f1.ny, f1.nz),
               emission_steps=max(f1.nemission, 0), emission_time=f1.temission,
               emission_ekin=files[end].bkenergy, image_charge=f1.flagimg == 1,
               image_cutoff=f1.zimage, domain_radius=f1.xrad, domain_length=f1.perdlen,
               diag_fixed_z=f1.flagdiag in (2, 3), diag_per_bin=f1.flagdiag == 3,
               seed=isnothing(seed) ? f1.seed : seed)
  f1.flagbc == 1 || push!(warns, "boundary condition flag $(f1.flagbc) not supported; open boundaries used")
  # ---- elements and events
  events = Event[]
  wakes = WakeRegion[]
  phys = Tuple{Float64,Float64,LineElement}[]   # (zedge, L, element)
  s_end = 0.0
  linmap_count = 0
  for v in f1.elements
    L = v[1]; nseg = round(Int, v[2]); mpstp = round(Int, v[3]); typ = round(Int, v[4])
    V = v[5:end]   # V[1] = V1 (zedge) ...
    zedge = V[1]
    s_end = zedge + L
    if typ < 0
      if typ == -1
        push!(events, Steer(z=V[2], dr=(V[3], V[4], V[5], V[6], V[7], V[8])))
      elseif typ == -2
        push!(events, ParticleOutput(z=V[3], every=max(nseg, 1), name="fort.$(mpstp).h5"))
      elseif typ == -3
        push!(events, Checkpoint(z=V[3], name="checkpoint.h5"))
      elseif typ == -4
        push!(events, TimeStepChange(z=V[3], dt=V[4]))
      elseif typ == -5
        s.sc_3d_start = V[3]
      elseif typ == -6
        model = if nseg > 0
          dat = [_fortran_numbers(l) for l in eachline(joinpath(dir, "fort.$(mpstp)"))]
          dat = filter(x -> length(x) >= 3, dat)
          TabulatedWake(V[7], [x[1] for x in dat], [x[2] for x in dat], [x[3] for x in dat])
        elseif V[5] > 100
          BTWWake()
        elseif V[5] < 0
          V[5] > -10 ? TeslaWake() : Tesla3p9Wake()
        else
          BaneWake(V[5], V[6], V[7])
        end
        push!(wakes, WakeRegion(V[3], V[4], model))
      elseif typ == -7
        s.merge_bins_at = V[3]
      elseif typ == -8
        push!(events, SpaceChargeSwitch(z=V[3], on=V[2] > 0))
      elseif typ == -9
        push!(events, SliceOutput(z=V[3], nslices=max(nseg, 1), name="fort.$(mpstp).h5"))
      elseif typ == -11
        push!(events, Collimator(z=V[2], xmin=V[3], xmax=V[4], ymin=V[5], ymax=V[6], round=V[7] > 10))
      elseif typ == -12
        linmap_count += 1
        m = readlines(joinpath(dir, "linearmap.in"))
        m = [x for x in (_fortran_numbers(l) for l in m) if !isempty(x)]
        R = reduce(vcat, [permutedims(_pad(m[6(linmap_count-1)+k], 6)[1:6]) for k in 1:6])
        # IMPACT-T applies the map to positions in units of cΔt: convert to meters
        sc = C_LIGHT_SI * f1.dt
        R = [R[i, j] * (isodd(i) ? sc : 1.0) / (isodd(j) ? sc : 1.0) for i in 1:6, j in 1:6]
        push!(events, LinearMap(z=V[1], R=R))
      elseif typ == -13
        push!(wakes, WakeRegion(V[1], V[2], DielectricWake(geometry=V[3] == 0 ? :slab : :cylinder,
              a=V[4], b=V[5], eps=V[6], Lx=V[7], nkx=round(Int, V[8]), nky=round(Int, V[9]))))
      elseif typ == -15
        s.sc_solver = :nbody
        s.nbody_r0 = V[3] * C_LIGHT_SI * f1.dt   # IMPACT-T uses V3 in units of cΔt
      elseif typ == -16
        push!(events, Heat(z=V[1], sigma=V[2]))
      elseif typ == -17
        push!(events, RotateZ(z=V[1], angle=V[2]))
      elseif typ == -99
        s.z_stop = V[3]
      end
      continue
    end
    ele = _impact_element(typ, L, V, dir, f1, warns)
    for e in (ele isa Vector ? ele : [ele])
      push!(phys, (zedge, L, e))
    end
  end
  isempty(phys) && error("no beamline elements in $file")
  # ---- sequential Beamline: elements start at their zedge; the "placement" length of an
  # element is the distance to the next element start (fields keep their own extent)
  sort!(phys, by=first)   # stable
  line = LineElement[]
  s0 = phys[1][1]
  cur = s0
  for (k, (zedge, L, ele)) in enumerate(phys)
    if zedge > cur + 1e-15
      push!(line, Drift(L=zedge - cur, name="gap$(k)"))
      cur = zedge
    end
    nxt = k < length(phys) ? phys[k+1][1] : zedge + L
    place = nxt < zedge + L ? nxt - zedge : L
    ele.body_length = L
    ele.L = place
    push!(line, ele)
    cur = zedge + place
  end
  s.s_offset = s0
  bl = Beamline(line; species_ref=species)
  # ---- particles
  bins = ParticleBin[]
  rng = Random.Xoshiro(s.seed)
  qref = f1.bfreq
  for (ib, fb) in enumerate(files)
    sp = _species_from(fb.bmass, fb.bcharge)
    Q = fb.bcurr / qref
    dp = fb.distparam
    dist = BeamDistribution(impact_dist_kind(fb.flagdist); x=PhasePlane(dp[1:7]),
                            y=PhasePlane(dp[8:14]), z=PhasePlane(dp[15:21]))
    if dist.kind == :read
      pf = ib == 1 ? joinpath(dir, "partcl.data") : joinpath(dir, "partcl$(ib).data")
      isfile(pf) || (pf = ib == 1 ? joinpath(dir, "partcl.in") : joinpath(dir, "partcl$(ib).in"))
      b = read_impactt_particles(pf, sp; mc2=fb.bmass, qe=fb.bcharge, charge=Q)
      if fb.flagdist == 16 || fb.flagdist == 166
        for (c, k) in ((1, 6), (2, 7), (3, 13), (4, 14), (5, 20), (6, 21))
          b.r[:, c] .+= dp[k]
        end
      end
      fb.flagdist in (24, 25, 167) && push!(warns, "particle file format $(fb.flagdist) read as IMPACT-T 6/9 column format")
      push!(bins, b)
    else
      push!(bins, make_bin(dist, fb.np, sp; charge=Q, mc2=fb.bmass, qe=fb.bcharge, rng=rng,
                           ib=ib, nb=f1.nbunch))
    end
  end
  for w in warns
    @warn w
  end
  return ImpactTInput(bl, s, bins, events, wakes, s_end, files, warns)
end

function _impact_element(typ, L, V, dir, f1, warns)
  flagerr = f1.flagerr
  aperture(ele, r) = (r > 1e-8 && (ele.x1_limit = -r; ele.x2_limit = r; ele.y1_limit = -r; ele.y2_limit = r))
  if typ == 0
    e = Drift(L=L); aperture(e, V[2]); return e
  elseif typ == 1
    e = Quadrupole(L=L, Bn1=V[2])
    fid = V[3]
    if fid > 0 && fid < 100
      e.quad_profile = :enge; e.quad_leff = fid
    elseif fid >= 100
      tab = read_rfdata_discrete(_rfname(dir, round(Int, fid)))[1]
      e.quad_profile = tab
    end
    aperture(e, V[4])
    _misalign!(e, V, 1, flagerr, L)
    flagerr == 1 || (V[9] != 0 && (e.tilt = V[9]))   # skew angle is always applied
    if V[10] > 1e-5
      e.rf_quad_frequency = V[10]; e.rf_quad_phase = deg(V[11])
    end
    return e
  elseif typ == 2
    e = LineElement(kind="ConstFocus", L=L, const_focus=(V[2], V[3], V[4])); aperture(e, V[5])
    return e
  elseif typ == 3
    e = Solenoid(L=L, field_map=read_t7_solenoid(_t7name(dir, round(Int, V[3]))), field_scale=V[2])
    aperture(e, V[4])
    return e
  elseif typ == 4
    geom = read_rfdata_dipole(_rfname(dir, round(Int, V[4])))
    e = SBend(L=L, Bn0=V[3], pole_faces=geom, bend_gap=2V[5])
    return e
  elseif typ == 5
    order = round(Int, V[2])
    V[4] > 1e-5 && push!(warns, "multipole fringe field files are not supported; hard edge used")
    e = order == 2 ? Sextupole(L=L, Bn2=V[3]) : order == 3 ? Octupole(L=L, Bn3=V[3]) :
        Multipole(L=L, Bn4=V[3])
    aperture(e, V[5])
    _misalign!(e, V, 2, flagerr, L)
    return e
  elseif typ in (101, 102, 103, 104)
    coefs = read_rfdata_numbers(_rfname(dir, round(Int, V[5])))
    iseven(length(coefs)) && push!(coefs, 0.0)
    fm = OnAxisFourier(coefs; zstart=0.0, zend=L, zlen=L, zc=L / 2)
    e = RFCavity(L=L, rf_frequency=V[3], phi0=deg(V[4]), field_map=fm, field_scale=V[2])
    aperture(e, V[6])
    if typ == 101
      q1 = Quadrupole(L=0.0, Bn1=V[8], field_extent=(0.0, V[7]))
      q2 = Quadrupole(L=0.0, Bn1=V[10], field_extent=(L - V[9], L))
      q1.name = "dtl_quad1"; q2.name = "dtl_quad2"
      return [e, q1, q2]
    end
    return e
  elseif typ == 105
    fid = round(Int, V[5])
    e = RFCavity(L=L, rf_frequency=V[3], phi0=deg(V[4]))
    if fid > 1000
      d = read_rfdata_discrete(_rfname(dir, fid); periodic_first=true)
      ez = d[1]; bz = length(d) >= 2 ? d[2] : nothing
    else
      ez, bz = read_rfdata_solrf(_rfname(dir, fid))
    end
    (V[2] != 0 && !isnothing(ez)) && (e.field_map = ez; e.field_scale = V[2])
    (V[12] != 0 && !isnothing(bz)) && (e.sol_map = bz; e.sol_scale = V[12])
    aperture(e, V[6])
    _misalign!(e, V, 3, flagerr, L)
    return e
  elseif typ == 110
    error("element type 110 (EMfld) is not used by IMPACT-T")
  elseif typ == 111
    e = RFCavity(L=L, rf_frequency=V[3], phi0=deg(V[4]),
                 field_map=read_t7_cartesian(_t7name(dir, round(Int, V[5]))), field_scale=V[2])
    aperture(e, V[6])
    return e
  elseif typ == 112
    e = RFCavity(L=L, rf_frequency=V[3], phi0=deg(V[4]),
                 field_map=read_t7_cylindrical(_t7name(dir, round(Int, V[5]))), field_scale=V[2])
    aperture(e, V[6])
    return e
  elseif typ == 113
    it = round(Int, V[10])
    af = if it == -1
      AlphaMagnet(V[2])
    elseif it == -2
      MeanderWave(escale=V[2], t_start=V[3], t_end=V[4], z_start=V[6], rise=V[7], speed=V[8])
    elseif it == -3
      SurfaceRoughness(escale=V[2], amplitude=V[7], wavenumber=V[8])
    elseif it == 10
      SteeringDipole(B0=V[3], gap=2V[6], faces=read_rfdata_dipole(_rfname(dir, round(Int, V[5]))),
                     vertical=round(Int, V[7]) > 0)
    else
      error("analytic field type $it not supported")
    end
    e = LineElement(kind="EMFieldAnalytic", L=L, analytic_field=af)
    return e
  else
    error("unknown IMPACT-T element type $typ")
  end
end
