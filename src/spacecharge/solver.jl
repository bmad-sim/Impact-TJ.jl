# Mean-field space-charge solver.
#
# For every particle bin, the charge is deposited on a Cartesian grid spanning the bunch,
# the Poisson equation is solved in the bin's rest frame (z stretched by γ) with open
# boundary conditions (Hockney doubled grid + integrated Green function) and the
# rest-frame potential is turned into lab-frame E and B fields:
#
#   Ex = -γ ∂φ'/∂x,  Ey = -γ ∂φ'/∂y,  Ez = -∂φ'/∂z',  Bx = -β/c Ey,  By = β/c Ex
#
# Image charges of a planar cathode at z = 0 are included with a shifted Green function
# evaluated as a correlation along z (it reuses the FFT of the charge density).
#
# FFT work (CPU): the Green function is even in x, y, z, so its transform on the doubled
# grid is obtained with a DCT-I of size (n+1)³, 8 times cheaper than a (2n)³ FFT.

"Grid geometry (lab frame). Node (i,j,k) (0-based) is at (xmin + i hx, ymin + j hy, zmin + k hz)."
struct GridGeom
  nx::Int; ny::Int; nz::Int
  xmin::Float64; ymin::Float64; zmin::Float64
  hx::Float64; hy::Float64; hz::Float64
end

mutable struct GreenCache
  hx::Float64; hy::Float64; hzp::Float64   # beam-frame spacings the cache was built for
  valid::Bool
  ghat::Array{Float64,3}                    # DCT-I of G, (nx+1, ny+1, nz+1)
end

mutable struct SpaceChargeSolver{A3<:AbstractArray{Float64,3},A4<:AbstractArray{Float64,4},
                                 C3<:AbstractArray{ComplexF64,3}}
  nx::Int; ny::Int; nz::Int
  geom::GridGeom
  rho::A3            # charge per node [C]
  phi::A3            # rest-frame potential
  phiimg::A3         # rest-frame image potential
  F::A4              # (6, nx, ny, nz): lab Ex, Ey, Ez, Bx, By, Bz summed over bins (Bz only from dielectric wakes)
  pad::A3            # (2nx, 2ny, 2nz) zero padded charge (only the first octant is ever written)
  outb::A3           # (2nx, 2ny, 2nz) inverse transform output
  rhat::C3           # (nx+1, 2ny, 2nz) transform of the charge
  prod::C3           # (nx+1, 2ny, 2nz) product work array
  plan_f             # rfft plan of `pad` (generic path)
  plan_b             # unnormalized inverse rfft plan of `prod` (1/M is folded into the Green function)
  pruned::Any        # CPU pruned transforms (see `_pruned_plans`), or nothing
  green_tol::Float64 # relative tolerance for reusing the Green function (grid spacing snapping)
  caches::Vector{GreenCache}   # one per bin
  # image Green function work arrays (CPU)
  himg::Array{Float64,3}       # (nx+1, ny+1, 2nz)
  hhat::Array{ComplexF64,3}    # (nx+1, ny+1, nz+1)
  plan_dct_img
  plan_rfft_img
  plan_dct
  gtmp::Array{Float64,3}       # (nx+1, ny+1, nz+1)
  # threaded deposition scratch grids
  scratch::Vector{Array{Float64,3}}
  # 1D line density work (wakes / CSR)
  line::Vector{Float64}
  # azimuthal deposition work
  rzparts::Vector{Matrix{Float64}}
  ringpts::Vector{Tuple{Int,Int,Int,Float64,Float64}}
  # device path: transformed Green functions per bin, kept when green_tol > 0
  dgreen::Vector{Any}      # (hx, hy, hzp, array) or nothing
end

function SpaceChargeSolver(nx, ny, nz; green_tol=0.0, proto=zeros(1))
  A = Base.typename(typeof(proto)).wrapper
  rho = A(zeros(nx, ny, nz)); phi = A(zeros(nx, ny, nz)); phiimg = A(zeros(nx, ny, nz))
  F = A(zeros(6, nx, ny, nz))
  pad = A(zeros(2nx, 2ny, 2nz))
  outb = A(zeros(2nx, 2ny, 2nz))
  rhat = A(zeros(ComplexF64, nx + 1, 2ny, 2nz))
  prod = A(zeros(ComplexF64, nx + 1, 2ny, 2nz))
  if A === Array
    FFTW.set_num_threads(Threads.nthreads())
    plan_f = nothing; plan_b = nothing
    pruned = _pruned_plans(nx, ny, nz, rhat, prod)
  else
    plan_f = AbstractFFTs.plan_rfft(pad)
    plan_b = AbstractFFTs.plan_brfft(prod, 2nx)
    pruned = nothing
  end
  himg = zeros(nx + 1, ny + 1, 2nz)
  hhat = zeros(ComplexF64, nx + 1, ny + 1, nz + 1)
  gtmp = zeros(nx + 1, ny + 1, nz + 1)
  plan_dct = FFTW.plan_r2r(gtmp, FFTW.REDFT00; flags=FFTW.MEASURE)
  plan_dct_img = FFTW.plan_r2r(himg, FFTW.REDFT00, (1, 2); flags=FFTW.MEASURE)
  plan_rfft_img = FFTW.plan_rfft(himg, 3; flags=FFTW.MEASURE)
  scratch = A === Array ? [zeros(nx, ny, nz) for _ in 1:Threads.nthreads()] : Array{Float64,3}[]
  geom = GridGeom(nx, ny, nz, 0, 0, 0, 1, 1, 1)
  return SpaceChargeSolver(nx, ny, nz, geom, rho, phi, phiimg, F, pad, outb, rhat, prod, plan_f, plan_b, pruned,
                           Float64(green_tol), GreenCache[], himg, hhat, plan_dct_img, plan_rfft_img,
                           plan_dct, gtmp, scratch, zeros(nz), Matrix{Float64}[],
                           Tuple{Int,Int,Int,Float64,Float64}[], Any[])
end

is_cpu(s::SpaceChargeSolver) = s.rho isa Array

# ---------------------------------------------------------------------------------------
# Grid geometry

"""
    set_geometry!(s, lo, hi, γ, snap)

Choose the grid covering the box `lo`..`hi` (lab frame). IMPACT-T expands the box by 1e-5
of its size on each side. If `snap` is true and `s.green_tol > 0`, the spacings are
snapped to those of the cached Green function of bin 1 (beam-frame hz' = γ hz) whenever
they are within the tolerance, and the grid is centered on the box; this lets the Green
function be reused for many steps. On a cache miss the spacings are enlarged by
green_tol/2 to leave room for the bunch to grow or shrink.
"""
function set_geometry!(s::SpaceChargeSolver, lo, hi, γ::Float64, snap::Bool)
  n = (s.nx, s.ny, s.nz)
  h = ntuple(3) do d
    span = hi[d] - lo[d]
    span = span > 0 ? span : max(abs(hi[d]), 1.0e-9) * 1.0e-6 + 1.0e-12
    span * (1 + 2.0e-5) / (n[d] - 1)
  end
  if snap && s.green_tol > 0
    tol = s.green_tol
    hb = (h[1], h[2], h[3] * γ)       # beam-frame spacings
    c = _cached_spacing(s)
    if c !== nothing && all(d -> hb[d] <= c[d] && hb[d] >= c[d] / (1 + tol), 1:3)
      hs = (c[1], c[2], c[3] / γ)
    else
      hs = h .* (1 + tol / 2)
    end
    mins = ntuple(d -> (lo[d] + hi[d]) / 2 - hs[d] * (n[d] - 1) / 2, 3)
    s.geom = GridGeom(s.nx, s.ny, s.nz, mins..., hs...)
  else
    mins = ntuple(d -> lo[d] - 1.0e-5 * max(hi[d] - lo[d], 0.0), 3)
    s.geom = GridGeom(s.nx, s.ny, s.nz, mins..., h...)
  end
  return s.geom
end

# beam-frame spacings of the cached Green function of bin 1 (or nothing)
function _cached_spacing(s::SpaceChargeSolver)
  if is_cpu(s)
    (isempty(s.caches) || !s.caches[1].valid) && return nothing
    c = s.caches[1]
    return (c.hx, c.hy, c.hzp)
  else
    (isempty(s.dgreen) || s.dgreen[1] === nothing) && return nothing
    c = s.dgreen[1]
    return (c[1], c[2], c[3])
  end
end

# ---------------------------------------------------------------------------------------
# Charge deposition

@inline function _cic_index(x, xmin, hinv, n)
  u = (x - xmin) * hinv
  i = unsafe_trunc(Int, floor(u))
  i = clamp(i, 0, n - 2)
  return i, u - i
end

"""
    deposit!(s, bin, cathode)

Cloud-in-cell deposition of the macroparticle charges of `bin` on the grid nodes. With
`cathode = true` only particles in front of the cathode (z > 0) are deposited.
"""
function deposit!(s::SpaceChargeSolver, b::ParticleBin, cathode::Bool)
  g = s.geom
  if is_cpu(s)
    _deposit_cpu!(s, b.r, b.w, b.state, g, cathode)
  else
    fill!(s.rho, 0.0)
    backend = get_backend(s.rho)
    _deposit_ka!(backend)(s.rho, b.r, b.w, b.state, g, cathode; ndrange=size(b.r, 1))
  end
  return s.rho
end

function _deposit_cpu!(s, r, w, state, g::GridGeom, cathode)
  N = size(r, 1)
  nt = length(s.scratch)
  hxi = 1 / g.hx; hyi = 1 / g.hy; hzi = 1 / g.hz
  nx, ny, nz = g.nx, g.ny, g.nz
  Threads.@threads :static for t in 1:nt
    grid = s.scratch[t]
    fill!(grid, 0.0)
    lo = div(N * (t - 1), nt) + 1
    hi = div(N * t, nt)
    @inbounds for p in lo:hi
      state[p] == STATE_ALIVE || continue
      z = r[p, 5]
      (cathode && z <= 0) && continue
      i, fx = _cic_index(r[p, 1], g.xmin, hxi, nx)
      j, fy = _cic_index(r[p, 3], g.ymin, hyi, ny)
      k, fz = _cic_index(z, g.zmin, hzi, nz)
      q = w[p]
      ax = 1 - fx; ay = 1 - fy; az = 1 - fz
      grid[i+1, j+1, k+1] += ax * ay * az * q
      grid[i+2, j+1, k+1] += fx * ay * az * q
      grid[i+1, j+2, k+1] += ax * fy * az * q
      grid[i+2, j+2, k+1] += fx * fy * az * q
      grid[i+1, j+1, k+2] += ax * ay * fz * q
      grid[i+2, j+1, k+2] += fx * ay * fz * q
      grid[i+1, j+2, k+2] += ax * fy * fz * q
      grid[i+2, j+2, k+2] += fx * fy * fz * q
    end
  end
  rho = s.rho
  L = length(rho)
  Threads.@threads :static for t in 1:nt
    lo = div(L * (t - 1), nt) + 1
    hi = div(L * t, nt)
    @inbounds for c in lo:hi
      acc = 0.0
      for q in 1:nt
        acc += s.scratch[q][c]
      end
      rho[c] = acc
    end
  end
  return rho
end

@kernel function _deposit_ka!(rho, @Const(r), @Const(w), @Const(state), g::GridGeom, cathode::Bool)
  p = @index(Global)
  @inbounds if state[p] == STATE_ALIVE && !(cathode && r[p, 5] <= 0)
    i, fx = _cic_index(r[p, 1], g.xmin, 1 / g.hx, g.nx)
    j, fy = _cic_index(r[p, 3], g.ymin, 1 / g.hy, g.ny)
    k, fz = _cic_index(r[p, 5], g.zmin, 1 / g.hz, g.nz)
    q = w[p]
    for dk in 0:1, dj in 0:1, di in 0:1
      wt = (di == 0 ? 1 - fx : fx) * (dj == 0 ? 1 - fy : fy) * (dk == 0 ? 1 - fz : fz)
      Atomix.@atomic rho[i+1+di, j+1+dj, k+1+dk] += wt * q
    end
  end
end

"""
    deposit_azimuthal!(s, bin, cathode)

IMPACT-T's azimuthally symmetric deposition (used until full 3D space charge is switched
on): charge is binned in (r, z) with area weighting and then spread uniformly over 4·ny
azimuthal angles onto the Cartesian grid. This suppresses numerical noise for round beams.
"""
function deposit_azimuthal!(s::SpaceChargeSolver, b::ParticleBin, cathode::Bool, comm::AbstractComm)
  g = s.geom
  nx, ny, nz = g.nx, g.ny, g.nz
  xmax = g.xmin + (nx - 1) * g.hx; ymax = g.ymin + (ny - 1) * g.hy
  rmax = sqrt(max(abs(g.xmin), abs(xmax))^2 + max(abs(g.ymin), abs(ymax))^2)
  hr = rmax / (nx - 1)
  hr2 = hr * hr
  hzi = 1 / g.hz
  if !(b.r isa Array)
    # device arrays: radial binning on the device, only the (r, z) profile is copied back
    rzd = similar(b.w, nx, nz)
    fill!(rzd, 0.0)
    backend = get_backend(rzd)
    _radial_deposit_ka!(backend)(rzd, b.r, b.w, b.state, hr, g.zmin, hzi, nx, nz, cathode;
                                 ndrange=size(b.r, 1))
    allreduce_sum!(comm, rzd)   # all ranks need the full radial profile
    nth = 4 * ny
    fill!(s.rho, 0.0)
    _spread_rings_ka!(backend)(s.rho, rzd, g, hr, nth, xmax, ymax; ndrange=(nx * nth, nz))
    _sync(backend)
    return s.rho
  end
  r = b.r; w = b.w; st = b.state
  nt = Threads.nthreads()
  length(s.rzparts) == nt && size(s.rzparts[1]) == (nx, nz) || (s.rzparts = [zeros(nx, nz) for _ in 1:nt])
  parts = s.rzparts
  N = size(r, 1)
  Threads.@threads :static for t in 1:nt
    rt = parts[t]
    fill!(rt, 0.0)
    lo = div(N * (t - 1), nt) + 1
    hi = div(N * t, nt)
    @inbounds for p in lo:hi
      st[p] == STATE_ALIVE || continue
      z = r[p, 5]
      (cathode && z <= 0) && continue
      rr = sqrt(r[p, 1]^2 + r[p, 3]^2)
      ix = min(unsafe_trunc(Int, rr / hr) + 1, nx - 1)     # 1-based inner node
      ab = (ix * ix * hr2 - rr * rr) / (ix * ix * hr2 - (ix - 1) * (ix - 1) * hr2)
      k, fz = _cic_index(z, g.zmin, hzi, nz)
      q = w[p]
      rt[ix, k+1] += ab * (1 - fz) * q
      rt[ix, k+2] += ab * fz * q
      rt[ix+1, k+1] += (1 - ab) * (1 - fz) * q
      rt[ix+1, k+2] += (1 - ab) * fz * q
    end
  end
  rz = parts[1]
  for t in 2:nt
    rz .+= parts[t]
  end
  return _spread_rings!(s, rz, g, hr, comm)
end

@kernel function _radial_deposit_ka!(rz, @Const(r), @Const(w), @Const(state), hr, zmin, hzi, nx, nz, cathode::Bool)
  p = @index(Global)
  @inbounds if state[p] == STATE_ALIVE && !(cathode && r[p, 5] <= 0)
    rr = sqrt(r[p, 1]^2 + r[p, 3]^2)
    ix = min(unsafe_trunc(Int, rr / hr) + 1, nx - 1)
    hr2 = hr * hr
    ab = (ix * ix * hr2 - rr * rr) / (ix * ix * hr2 - (ix - 1) * (ix - 1) * hr2)
    k, fz = _cic_index(r[p, 5], zmin, hzi, nz)
    q = w[p]
    Atomix.@atomic rz[ix, k+1] += ab * (1 - fz) * q
    Atomix.@atomic rz[ix, k+2] += ab * fz * q
    Atomix.@atomic rz[ix+1, k+1] += (1 - ab) * (1 - fz) * q
    Atomix.@atomic rz[ix+1, k+2] += (1 - ab) * fz * q
  end
end

# device version of `_spread_rings!`: one work item per (ring point, z node)
@kernel function _spread_rings_ka!(rho, @Const(rz), g::GridGeom, hr, nth, xmax, ymax)
  ij, k = @index(Global, NTuple)
  i = (ij - 1) ÷ nth + 1
  j = (ij - 1) % nth + 1
  @inbounds q = rz[i, k] / nth
  if q != 0
    hθ = 2π / nth
    xx = (i - 1) * hr * cos((j - 1) * hθ)
    yy = (i - 1) * hr * sin((j - 1) * hθ)
    if xx >= g.xmin && xx <= xmax && yy >= g.ymin && yy <= ymax
      ix, fx = _cic_index(xx, g.xmin, 1 / g.hx, g.nx)
      jy, fy = _cic_index(yy, g.ymin, 1 / g.hy, g.ny)
      Atomix.@atomic rho[ix+1, jy+1, k] += (1 - fx) * (1 - fy) * q
      Atomix.@atomic rho[ix+2, jy+1, k] += fx * (1 - fy) * q
      Atomix.@atomic rho[ix+1, jy+2, k] += (1 - fx) * fy * q
      Atomix.@atomic rho[ix+2, jy+2, k] += fx * fy * q
    end
  end
end

"Spread the (r, z) charge profile uniformly over 4·ny azimuthal points onto the grid."
function _spread_rings!(s::SpaceChargeSolver, rz::Matrix{Float64}, g::GridGeom, hr, comm)
  nx, ny, nz = g.nx, g.ny, g.nz
  xmax = g.xmin + (nx - 1) * g.hx; ymax = g.ymin + (ny - 1) * g.hy
  allreduce_sum!(comm, rz)   # all ranks need the full radial profile
  nth = 4 * ny
  rz ./= nth
  # grid nodes and weights of the ring points (i, j) inside the Cartesian box
  hθ = 2π / nth
  hxi = 1 / g.hx; hyi = 1 / g.hy
  pts = s.ringpts
  empty!(pts)
  for i in 1:nx, j in 1:nth
    xx = (i - 1) * hr * cos((j - 1) * hθ)
    yy = (i - 1) * hr * sin((j - 1) * hθ)
    (xx >= g.xmin && xx <= xmax && yy >= g.ymin && yy <= ymax) || continue
    ix, fx = _cic_index(xx, g.xmin, hxi, nx)
    jy, fy = _cic_index(yy, g.ymin, hyi, ny)
    push!(pts, (i, ix, jy, fx, fy))
  end
  rho = s.rho isa Array ? s.rho : zeros(nx, ny, nz)
  Threads.@threads for k in 1:nz
    @inbounds begin
      for jj in 1:ny, ii in 1:nx
        rho[ii, jj, k] = 0.0
      end
      for (i, ix, jy, fx, fy) in pts
        q = rz[i, k]
        q == 0 && continue
        rho[ix+1, jy+1, k] += (1 - fx) * (1 - fy) * q
        rho[ix+2, jy+1, k] += fx * (1 - fy) * q
        rho[ix+1, jy+2, k] += (1 - fx) * fy * q
        rho[ix+2, jy+2, k] += fx * fy * q
      end
    end
  end
  rho === s.rho || copyto!(s.rho, rho)
  return s.rho
end

# ---------------------------------------------------------------------------------------
# Green function transforms

"Transform of the direct Green function for beam-frame spacings (hx, hy, hzp)."
function green_hat!(s::SpaceChargeSolver, ib, hx, hy, hzp)
  while length(s.caches) < ib
    push!(s.caches, GreenCache(0, 0, 0, false, zeros(s.nx + 1, s.ny + 1, s.nz + 1)))
  end
  c = s.caches[ib]
  if c.valid && c.hx == hx && c.hy == hy && c.hzp == hzp
    return c.ghat
  end
  igf!(s.gtmp, hx, hy, hzp, 0.0)
  mul!(c.ghat, s.plan_dct, s.gtmp)
  c.hx = hx; c.hy = hy; c.hzp = hzp; c.valid = true
  return c.ghat
end

"""
Transform of the image Green function H(x, y, 2zmin' + m hz') (m = 0..2nz-1): DCT-I in
x, y (even) and a real FFT along z.
"""
function image_green_hat!(s::SpaceChargeSolver, hx, hy, hzp, zminp)
  igf!(s.himg, hx, hy, hzp, 2 * zminp)
  s.himg .= s.plan_dct_img * s.himg
  mul!(s.hhat, s.plan_rfft_img, s.himg)
  return s.hhat
end

@inline _fold(k, n) = k <= n ? k : 2n - k   # 0-based index folding for even sequences

# ---------------------------------------------------------------------------------------
# Poisson solve

"""
    solve_bin!(s, ib, γ, βsign, image)

Solve for the rest-frame potential of the charge in `s.rho` (bin `ib` with Lorentz factor
`γ`) and add the lab-frame fields to `s.F`. If `image`, the field of the image charge in
the cathode plane z = 0 is added as well.
"""
function solve_bin!(s::SpaceChargeSolver, ib::Int, γ::Float64, βsign::Float64, image::Bool)
  g = s.geom
  nx, ny, nz = s.nx, s.ny, s.nz
  hx, hy, hzp = g.hx, g.hy, g.hz * γ
  # transform of the zero padded charge
  _forward!(s)
  # direct potential
  if is_cpu(s)
    ghat = green_hat!(s, ib, hx, hy, hzp)
    _mul_green_cpu!(s.prod, s.rhat, ghat, nx, ny, nz)
  else
    _mul_green_generic!(s, ib, hx, hy, hzp)
  end
  _inverse!(s, s.phi)
  if image
    zminp = g.zmin * γ
    if is_cpu(s)
      hhat = image_green_hat!(s, hx, hy, hzp, zminp)
      _mul_image_cpu!(s.prod, s.rhat, hhat, nx, ny, nz)
    else
      _mul_image_generic!(s, hx, hy, hzp, zminp)
    end
    _inverse!(s, s.phiimg)
  end
  add_lab_fields!(s, γ, βsign, image)
  return nothing
end

# ---------------------------------------------------------------------------------------
# Pruned FFTs of zero padded data (CPU). Only 1/8 of the doubled grid holds charge and only
# 1/8 of the result is needed, so the transforms along x and y skip all-zero lines:
#   forward: rfft x on (2nx, ny, nz) → fft y on (nx+1, 2ny, nz) → fft z on (nx+1, 2ny, 2nz)
#   inverse: the same in reverse order, keeping only the needed planes.
struct PrunedPlans
  xpad::Array{Float64,3}     # (2nx, ny, nz), rows nx+1:2nx always zero
  outr::Array{Float64,3}     # (2nx, ny, nz)
  f1; f2; f3; b3; b2; b1
end

function _pruned_plans(nx, ny, nz, rhat, prod)
  fl = FFTW.MEASURE
  xpad = zeros(2nx, ny, nz); outr = zeros(2nx, ny, nz)
  f1 = FFTW.rFFTWPlan{Float64,FFTW.FORWARD,false,3}(xpad, view(rhat, :, 1:ny, 1:nz), 1, fl, FFTW.NO_TIMELIMIT)
  f2 = FFTW.plan_fft!(view(rhat, :, :, 1:nz), 2; flags=fl)
  f3 = FFTW.plan_fft!(rhat, 3; flags=fl)
  b3 = FFTW.plan_bfft!(prod, 3; flags=fl)
  b2 = FFTW.plan_bfft!(view(prod, :, :, 1:nz), 2; flags=fl)
  b1 = FFTW.rFFTWPlan{ComplexF64,FFTW.BACKWARD,false,3}(view(prod, :, 1:ny, 1:nz), outr, 1, fl,
                                                         FFTW.NO_TIMELIMIT)
  fill!(xpad, 0.0); fill!(outr, 0.0); fill!(rhat, 0.0); fill!(prod, 0.0)
  return PrunedPlans(xpad, outr, f1, f2, f3, b3, b2, b1)
end

function _forward!(s::SpaceChargeSolver)
  nx, ny, nz = s.nx, s.ny, s.nz
  if s.pruned === nothing
    _copy_octant!(s.pad, s.rho, nx, ny, nz)
    mul!(s.rhat, s.plan_f, s.pad)
    return
  end
  P = s.pruned
  _copy_octant!(P.xpad, s.rho, nx, ny, nz)
  C = s.rhat
  mul!(view(C, :, 1:ny, 1:nz), P.f1, P.xpad)
  Threads.@threads for k in 1:nz          # zero padding in y
    @inbounds for j in ny+1:2ny, i in axes(C, 1)
      C[i, j, k] = 0
    end
  end
  P.f2 * view(C, :, :, 1:nz)
  Threads.@threads for k in nz+1:2nz      # zero padding in z
    @inbounds for j in axes(C, 2), i in axes(C, 1)
      C[i, j, k] = 0
    end
  end
  P.f3 * C
  return
end

"Unnormalized inverse transform of `s.prod`; the first octant is written to `dest`."
function _inverse!(s::SpaceChargeSolver, dest)
  nx, ny, nz = s.nx, s.ny, s.nz
  if s.pruned === nothing
    mul!(s.outb, s.plan_b, s.prod)
    _copy_octant!(dest, s.outb, nx, ny, nz)
    return
  end
  P = s.pruned
  Q = s.prod
  P.b3 * Q
  P.b2 * view(Q, :, :, 1:nz)
  mul!(P.outr, P.b1, view(Q, :, 1:ny, 1:nz))
  _copy_octant!(dest, P.outr, nx, ny, nz)
  return
end

"Copy the first nx×ny×nz octant between arrays of different sizes."
function _copy_octant!(dst::Array, src::Array, nx, ny, nz)
  Threads.@threads for k in 1:nz
    @inbounds for j in 1:ny, i in 1:nx
      dst[i, j, k] = src[i, j, k]
    end
  end
  return dst
end
_copy_octant!(dst, src, nx, ny, nz) = (view(dst, 1:nx, 1:ny, 1:nz) .= view(src, 1:nx, 1:ny, 1:nz); dst)

function _mul_green_cpu!(prod, rhat, ghat, nx, ny, nz)
  invM = 1 / (8nx * ny * nz)
  Threads.@threads for kz in 0:2nz-1
    fz = _fold(kz, nz)
    @inbounds for ky in 0:2ny-1
      fy = _fold(ky, ny)
      for kx in 0:nx
        prod[kx+1, ky+1, kz+1] = rhat[kx+1, ky+1, kz+1] * (ghat[kx+1, fy+1, fz+1] * invM)
      end
    end
  end
end

# image potential: φimg = -IFFT( Ĥ(k) ρ̂(kx, ky, -kz) )
function _mul_image_cpu!(prod, rhat, hhat, nx, ny, nz)
  M = 2nz
  invM = 1 / (8nx * ny * nz)
  Threads.@threads for kz in 0:M-1
    kzr = kz == 0 ? 0 : M - kz
    @inbounds for ky in 0:2ny-1
      fy = _fold(ky, ny)
      for kx in 0:nx
        h = kz <= nz ? hhat[kx+1, fy+1, kz+1] : conj(hhat[kx+1, fy+1, M-kz+1])
        prod[kx+1, ky+1, kz+1] = -invM * h * rhat[kx+1, ky+1, kzr+1]
      end
    end
  end
end

# Generic (GPU) path: the Green functions are built on the doubled grid on the device and
# transformed with full FFTs.
function _mul_green_generic!(s, ib, hx, hy, hzp)
  if s.green_tol > 0
    # keep the transformed Green function of every bin on the device for reuse
    while length(s.dgreen) < ib
      push!(s.dgreen, nothing)
    end
    c = s.dgreen[ib]
    if c === nothing || c[1] != hx || c[2] != hy || c[3] != hzp
      G = c === nothing ? similar(s.prod) : c[4]
      green_doubled!(s.outb, s.nx, s.ny, s.nz, hx, hy, hzp, 0.0, false)
      mul!(G, s.plan_f, s.outb)
      G ./= length(s.pad)
      s.dgreen[ib] = c = (hx, hy, hzp, G)
    end
    s.prod .= s.rhat .* c[4]
  else
    green_doubled!(s.outb, s.nx, s.ny, s.nz, hx, hy, hzp, 0.0, false)
    mul!(s.prod, s.plan_f, s.outb)
    s.prod .*= s.rhat ./ length(s.pad)
  end
end

# prod holds the transform of the image Green function on entry
@kernel function _image_product_ka!(prod, @Const(rhat), M, scale)
  i, j, kz = @index(Global, NTuple)
  kzr = kz == 1 ? 1 : M - kz + 2
  @inbounds prod[i, j, kz] = -scale * prod[i, j, kz] * rhat[i, j, kzr]
end

function _mul_image_generic!(s, hx, hy, hzp, zminp)
  green_doubled!(s.outb, s.nx, s.ny, s.nz, hx, hy, hzp, 2zminp, true)
  mul!(s.prod, s.plan_f, s.outb)
  backend = get_backend(s.prod)
  _image_product_ka!(backend)(s.prod, s.rhat, 2 * s.nz, 1 / length(s.pad); ndrange=size(s.prod))
  _sync(backend)
end

# ---------------------------------------------------------------------------------------
# Lab frame fields from the rest-frame potential

@inline function _dfd(φ, i, j, k, n, dim, hinv)
  # -dφ/dξ, second order centered or one sided at the boundary (IMPACT-T stencils)
  @inbounds if dim == 1
    if i == 1
      return hinv * (1.5φ[1, j, k] - 2φ[2, j, k] + 0.5φ[3, j, k])
    elseif i == n
      return hinv * (-0.5φ[n-2, j, k] + 2φ[n-1, j, k] - 1.5φ[n, j, k])
    else
      return 0.5hinv * (φ[i-1, j, k] - φ[i+1, j, k])
    end
  elseif dim == 2
    if j == 1
      return hinv * (1.5φ[i, 1, k] - 2φ[i, 2, k] + 0.5φ[i, 3, k])
    elseif j == n
      return hinv * (-0.5φ[i, n-2, k] + 2φ[i, n-1, k] - 1.5φ[i, n, k])
    else
      return 0.5hinv * (φ[i, j-1, k] - φ[i, j+1, k])
    end
  else
    if k == 1
      return hinv * (1.5φ[i, j, 1] - 2φ[i, j, 2] + 0.5φ[i, j, 3])
    elseif k == n
      return hinv * (-0.5φ[i, j, n-2] + 2φ[i, j, n-1] - 1.5φ[i, j, n])
    else
      return 0.5hinv * (φ[i, j, k-1] - φ[i, j, k+1])
    end
  end
end

@kernel function _lab_fields_ka!(F, @Const(φ), @Const(φi), nx, ny, nz, hxi, hyi, hzi, γ, βc, βci, image::Bool)
  i, j, k = @index(Global, NTuple)
  ex = _dfd(φ, i, j, k, nx, 1, hxi)
  ey = _dfd(φ, i, j, k, ny, 2, hyi)
  ez = _dfd(φ, i, j, k, nz, 3, hzi)
  @inbounds begin
    F[1, i, j, k] += γ * ex
    F[2, i, j, k] += γ * ey
    F[3, i, j, k] += ez
    F[4, i, j, k] -= γ * ey * βc
    F[5, i, j, k] += γ * ex * βc
  end
  if image
    exi = _dfd(φi, i, j, k, nx, 1, hxi)
    eyi = _dfd(φi, i, j, k, ny, 2, hyi)
    ezi = _dfd(φi, i, j, k, nz, 3, hzi)
    @inbounds begin
      F[1, i, j, k] += γ * exi
      F[2, i, j, k] += γ * eyi
      F[3, i, j, k] += ezi
      F[4, i, j, k] -= γ * eyi * βci
      F[5, i, j, k] += γ * exi * βci
    end
  end
end

function add_lab_fields!(s::SpaceChargeSolver, γ, βsign, image)
  g = s.geom
  β = sqrt(max(γ^2 - 1, 0.0)) / γ
  βc = βsign * β / C_LIGHT_SI
  βci = -β / C_LIGHT_SI          # the image charge moves towards -z
  backend = get_backend(s.F)
  _lab_fields_ka!(backend)(s.F, s.phi, s.phiimg, s.nx, s.ny, s.nz, 1 / g.hx, 1 / g.hy,
                           1 / (g.hz * γ), γ, βc, βci, image; ndrange=(s.nx, s.ny, s.nz))
  _sync(backend)
end

reset_fields!(s::SpaceChargeSolver) = fill!(s.F, 0.0)

"CIC interpolation of the lab fields stored in `F` at a lab position (Bz only if `bz`)."
@inline function gather_sc(F, g::GridGeom, x, y, z, bz::Bool=false)
  i, fx = _cic_index(x, g.xmin, 1 / g.hx, g.nx)
  j, fy = _cic_index(y, g.ymin, 1 / g.hy, g.ny)
  k, fz = _cic_index(z, g.zmin, 1 / g.hz, g.nz)
  ex = 0.0; ey = 0.0; ez = 0.0; bx = 0.0; by = 0.0; bzz = 0.0
  for dk in 0:1, dj in 0:1, di in 0:1
    wt = (di == 0 ? 1 - fx : fx) * (dj == 0 ? 1 - fy : fy) * (dk == 0 ? 1 - fz : fz)
    ii = i + 1 + di; jj = j + 1 + dj; kk = k + 1 + dk
    @inbounds begin
      ex += wt * F[1, ii, jj, kk]; ey += wt * F[2, ii, jj, kk]; ez += wt * F[3, ii, jj, kk]
      bx += wt * F[4, ii, jj, kk]; by += wt * F[5, ii, jj, kk]
      bz && (bzz += wt * F[6, ii, jj, kk])
    end
  end
  return V3(ex, ey, ez), V3(bx, by, bzz)
end
