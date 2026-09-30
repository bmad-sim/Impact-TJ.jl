# Wakefields of dielectric lined waveguides (D. Mihalcea, P. Piot et al., PRST-AB 15,
# 081304 (2012)), as in IMPACT-T element -13.
#
# Slab geometry: dielectric of relative permittivity ε between |y| = a and |y| = b, width
# Lx in x; LSM and LSE modes, symmetric and antisymmetric in y. Cylindrical geometry:
# dielectric between r = a and r = b, monopole modes only.
# The mode spectrum is computed once per structure (IMPACT-T recomputes it every step)
# and the Bessel function derivatives are exact (IMPACT-T uses single precision finite
# differences).

"""
    DielectricWake(; geometry=:slab, a, b, eps, Lx=0.0, nkx=1, nky=1)

Wakefield of a dielectric lined waveguide: `a` inner half gap (slab) or radius (cylinder),
`b` outer half gap / radius, `eps` relative permittivity, `Lx` slab width, `nkx`, `nky`
number of modes in x and y (cylinder: `nky` monopole modes).
"""
Base.@kwdef struct DielectricWake <: WakeModel
  geometry::Symbol = :slab
  a::Float64
  b::Float64
  eps::Float64
  Lx::Float64 = 0.0
  nkx::Int = 1
  nky::Int = 1
end

struct DWAModes
  kx::Vector{Float64}
  ky::Array{Float64,3}     # (nkx, nky, 2) (slab) / (1, nky, 1) (cylinder)
  amp::Array{Float64,3}
end

const _DWA_CACHE = Dict{DielectricWake,DWAModes}()
dwa_modes(w::DielectricWake) = get!(() -> _dwa_modes(w), _DWA_CACHE, w)

_f1(x, c1, c2) = c1 / tan(x) - c2 * x
_f2(x, c1, c3) = c1 / tan(x) + c3 / x

function _bisect(f, x1, x2; tol=1e-6)
  f1 = f(x1)
  while abs(x2 - x1) > tol
    cx = (x1 + x2) / 2
    fc = f(cx)
    if f1 * fc < 0
      x2 = cx
    else
      x1 = cx; f1 = fc
    end
  end
  return (x1 + x2) / 2
end

function _amp_slab(lsm::Bool, sym::Bool, a, b, eps, kx, ky)
  d = b - a
  t1 = sym ? sinh(2kx * a) / 2 / kx : cosh(2kx * a) / 2 / kx
  ch = sym ? cosh(kx * a)^2 : sinh(kx * a)^2
  if lsm
    t2 = eps * ch / sin(ky * d)^2
    t3 = d * (1 + eps * kx^2 / ky^2) / 2 - sin(2ky * d) / 4 / ky * (1 - eps * kx^2 / ky^2)
  else
    t2 = ch / sin(ky * d)^2
    t3 = d / 2 * (eps + ky^2 / kx^2) - sin(2ky * d) / 4 / ky * (eps - ky^2 / kx^2)
  end
  return 1 / (t1 + t2 * t3)
end

# cylindrical monopole dispersion function and its ingredients
_R0(s, a, b) = bessely0(s * b) * besselj0(s * a) - besselj0(s * b) * bessely0(s * a)
_R0P(s, a, b) = -bessely0(s * b) * besselj1(s * a) + besselj0(s * b) * bessely1(s * a)   # d/d(sa)
_LFS(s, a, b, eps) = _R0P(s, a, b) + s * a * _R0(s, a, b) / eps / 2

function _dwa_modes(w::DielectricWake)
  a, b, eps = w.a, w.b, w.eps
  if w.geometry == :slab
    kx = [(2i - 1) * π / w.Lx for i in 1:w.nkx]
    ky = zeros(w.nkx, w.nky, 2); amp = zeros(w.nkx, w.nky, 2)
    for i in 1:w.nkx, k in 1:2
      c1 = k == 1 ? 1 / tanh(kx[i] * a) : tanh(kx[i] * a)
      c2 = 1 / kx[i] / (b - a) / eps
      c3 = kx[i] * (b - a)
      for j in 1:w.nky
        lsm = ((j - 1) ÷ 2) * 2 == (j - 1)
        x1 = (j - 1) * π / 2 + 1e-10; x2 = j * π / 2 - 1e-10
        f = lsm ? (x -> _f1(x, c1, c2)) : (x -> _f2(x, c1, c3))
        ky[i, j, k] = _bisect(f, x1, x2; tol=1e-13) / (b - a)
        amp[i, j, k] = _amp_slab(lsm, k == 1, a, b, eps, kx[i], ky[i, j, k])
      end
    end
    return DWAModes(kx, ky, amp)
  else
    ky = zeros(1, w.nky, 1); amp = zeros(1, w.nky, 1)
    sg = 2π / a
    step = sg / 1e3
    s = step
    found = 0
    while found < w.nky
      if _LFS(s, a, b, eps) * _LFS(s + step, a, b, eps) < 0
        found += 1
        root = _bisect(x -> _LFS(x, a, b, eps), s, s + step; tol=step * 1e-12)
        ky[1, found, 1] = root
        h = root * 1e-6
        d = (_LFS(root + h, a, b, eps) - _LFS(root - h, a, b, eps)) / 2h
        amp[1, found, 1] = _R0(root, a, b) / d / a
      end
      s += step
    end
    return DWAModes(Float64[], ky, amp)
  end
end

# λ(i) = Σ_{j≥i} w_j kern(j-i) hz, with the j = i term weighted 1/2
function _dwa_conv(w, kern, hz)
  n = length(w)
  out = zeros(n)
  for i in 1:n
    s = 0.5 * kern[1] * w[i]
    for j in i+1:n
      s += kern[j-i+1] * w[j]
    end
    out[i] = s * hz
  end
  return out
end

"""
    dielectric_wake!(sc, w, γ)

Add the fields of the dielectric wake `w` of the charge on the grid of `sc` (bin with
Lorentz factor γ) to the field grid.
"""
function dielectric_wake!(sc::SpaceChargeSolver, w::DielectricWake, γ)
  g = sc.geom
  nx, ny, nz = g.nx, g.ny, g.nz
  size(sc.F, 1) >= 6 || error("dielectric wakes need a 6 component field grid")
  modes = dwa_modes(w)
  hz = g.hz
  if !is_cpu(sc)
    return _dwa_device!(sc, w, modes)
  end
  Q = sc.rho                 # node charges [C]
  F = sc.F
  xs = [g.xmin + (i - 1) * g.hx for i in 1:nx]
  ys = [g.ymin + (j - 1) * g.hy for j in 1:ny]
  zs = [(k - 1) * hz for k in 1:nz]
  if w.geometry == :slab
    _dwa_slab!(F, Q, w, modes, xs, ys, zs, hz)
  else
    _dwa_cylinder!(F, Q, w, modes, zs, hz)
  end
  return sc
end

function _dwa_slab!(F, Q, w, modes, xs, ys, zs, hz)
  nx, ny, nz = length(xs), length(ys), length(zs)
  begin
    lx = π / modes.kx[1]
    cV = 1 / (EPS0_SI * lx * hz)    # IMPACT-T: dv γ /(lx hz) × ρ/(ε0 dv γ)
    cT = cV / C_LIGHT_SI
    # The fields of every mode are separable: f(x) g(y) h(z). For fixed (symmetry e, kx)
    # the x and y factors are common to all ky modes, so the z factors are summed first.
    for e in 1:2, l in 1:w.nkx
      kx = modes.kx[l]
      cx = cos.(kx .* xs); sx = sin.(kx .* xs)
      chy = cosh.(kx .* ys); shy = sinh.(kx .* ys)
      yw = e == 1 ? chy : shy           # y weight of the source
      lam = zeros(nz)
      Threads.@threads for k in 1:nz
        acc = 0.0
        @inbounds for p in 1:ny, n in 1:nx
          acc += Q[n, p, k] * yw[p] * cx[n]
        end
        lam[k] = acc
      end
      # z profiles of the 6 field patterns summed over the ky modes
      hEx = zeros(nz); hEy = zeros(nz); hEz = zeros(nz); hBx = zeros(nz); hBy = zeros(nz); hBz = zeros(nz)
      for m in 1:w.nky
        lsm = ((m - 1) ÷ 2) * 2 == (m - 1)
        ky = modes.ky[l, m, e]
        kz = sqrt((kx^2 + ky^2) / (w.eps - 1))
        A = modes.amp[l, m, e]
        ls = _dwa_conv(lam, [A * cos(kz * z) for z in zs], hz)
        la = _dwa_conv(lam, [A * sin(kz * z) for z in zs], hz)
        q = (kx^2 + kz^2) / kx / kz
        if lsm
          @. hEx -= la * kx / kz; @. hEy += la * q; @. hEz += ls; @. hBx -= la * kz / kx; @. hBz += ls
        else
          @. hEx += la * kz / kx; @. hEz += ls; @. hBx += la * kx / kz; @. hBy += la * q; @. hBz += ls
        end
      end
      # transverse patterns: e = 1 (symmetric) / e = 2 (antisymmetric) swap cosh and sinh
      yA = e == 1 ? chy : shy     # Ex, Ez, By
      yB = e == 1 ? shy : chy     # Ey, Bx, Bz
      Threads.@threads for k in 1:nz
        @inbounds for j in 1:ny, i in 1:nx
          F[1, i, j, k] += cV * hEx[k] * sx[i] * yA[j]
          F[2, i, j, k] += cV * hEy[k] * cx[i] * yB[j]
          F[3, i, j, k] -= cV * hEz[k] * cx[i] * yA[j]
          F[4, i, j, k] += cT * hBx[k] * cx[i] * yB[j]
          F[5, i, j, k] += cT * hBy[k] * sx[i] * yA[j]
          F[6, i, j, k] -= cT * hBz[k] * sx[i] * yB[j]
        end
      end
    end
  end
  return F
end

function _dwa_cylinder!(F, Q, w, modes, zs, hz)
  nz = length(zs)
  begin
    λ = vec(sum(Q; dims=(1, 2)))
    ez = zeros(nz)
    for m in 1:w.nky
      ky = modes.ky[1, m, 1]
      kz = ky / sqrt(w.eps - 1)
      A = modes.amp[1, m, 1]
      ez .+= _dwa_conv(λ, [A * cos(kz * z) for z in zs], hz)
    end
    cV = 1 / (EPS0_SI * w.eps * π * hz)
    for k in 1:nz
      F[3, :, :, k] .-= cV * ez[k]
    end
  end
  return F
end

# ---------------------------------------------------------------------------------------
# Device (GPU) path: same algorithm, with the mode sums done in kernels. The mode table
# (a few numbers per mode) is the only data sent to the device.

# λ(k) = Σ_{i,j} Q[i,j,k] cos(kx x_i) yw(y_j)  (slab)  or  Σ_{i,j} Q[i,j,k]  (cylinder)
@kernel function _dwa_lam_ka!(lam, @Const(Q), g::GridGeom, kx, sym::Bool, slab::Bool)
  k = @index(Global)
  acc = 0.0
  @inbounds for j in 1:g.ny
    y = g.ymin + (j - 1) * g.hy
    yw = slab ? (sym ? cosh(kx * y) : sinh(kx * y)) : 1.0
    a = 0.0
    for i in 1:g.nx
      xw = slab ? cos(kx * (g.xmin + (i - 1) * g.hx)) : 1.0
      a += Q[i, j, k] * xw
    end
    acc += a * yw
  end
  @inbounds lam[k] = acc
end

# z profiles of the six field patterns summed over the ky modes (slab), or Ez (cylinder)
@kernel function _dwa_prof_ka!(H, @Const(lam), @Const(M), l, e, kx, eps, hz, nz, nky, slab::Bool)
  i = @index(Global)
  hEx = 0.0; hEy = 0.0; hEz = 0.0; hBx = 0.0; hBy = 0.0; hBz = 0.0
  @inbounds for m in 1:nky
    ky = M[l, m, e, 1]; A = M[l, m, e, 2]
    kz = slab ? sqrt((kx^2 + ky^2) / (eps - 1)) : ky / sqrt(eps - 1)
    ls = 0.5 * A * lam[i]; la = 0.0
    for j in i+1:nz
      sn, cs = sincos(kz * (j - i) * hz)
      ls += A * cs * lam[j]
      la += A * sn * lam[j]
    end
    ls *= hz; la *= hz
    if !slab
      hEz += ls
    elseif iseven(m - 1)   # LSM
      q = (kx^2 + kz^2) / kx / kz
      hEx -= la * kx / kz; hEy += la * q; hEz += ls; hBx -= la * kz / kx; hBz += ls
    else                   # LSE
      q = (kx^2 + kz^2) / kx / kz
      hEx += la * kz / kx; hEz += ls; hBx += la * kx / kz; hBy += la * q; hBz += ls
    end
  end
  @inbounds begin
    H[i, 1] = hEx; H[i, 2] = hEy; H[i, 3] = hEz; H[i, 4] = hBx; H[i, 5] = hBy; H[i, 6] = hBz
  end
end

@kernel function _dwa_add_ka!(F, @Const(H), g::GridGeom, kx, sym::Bool, cV, cT, slab::Bool)
  i, j, k = @index(Global, NTuple)
  @inbounds if slab
    x = g.xmin + (i - 1) * g.hx; y = g.ymin + (j - 1) * g.hy
    cx = cos(kx * x); sx = sin(kx * x)
    ch = cosh(kx * y); sh = sinh(kx * y)
    yA = sym ? ch : sh; yB = sym ? sh : ch
    F[1, i, j, k] += cV * H[k, 1] * sx * yA
    F[2, i, j, k] += cV * H[k, 2] * cx * yB
    F[3, i, j, k] -= cV * H[k, 3] * cx * yA
    F[4, i, j, k] += cT * H[k, 4] * cx * yB
    F[5, i, j, k] += cT * H[k, 5] * sx * yA
    F[6, i, j, k] -= cT * H[k, 6] * sx * yB
  else
    F[3, i, j, k] -= cV * H[k, 3]
  end
end

function _dwa_device!(sc::SpaceChargeSolver, w::DielectricWake, modes::DWAModes)
  g = sc.geom
  nz = g.nz
  hz = g.hz
  slab = w.geometry == :slab
  backend = get_backend(sc.F)
  M = copyto!(similar(sc.rho, size(modes.ky)..., 2), cat(modes.ky, modes.amp; dims=4))
  lam = similar(sc.rho, nz)
  H = similar(sc.rho, nz, 6)
  if slab
    lx = π / modes.kx[1]
    cV = 1 / (EPS0_SI * lx * hz)
    cT = cV / C_LIGHT_SI
    for e in 1:2, l in 1:w.nkx
      kx = modes.kx[l]
      _dwa_lam_ka!(backend)(lam, sc.rho, g, kx, e == 1, true; ndrange=nz)
      _dwa_prof_ka!(backend)(H, lam, M, l, e, kx, w.eps, hz, nz, w.nky, true; ndrange=nz)
      _dwa_add_ka!(backend)(sc.F, H, g, kx, e == 1, cV, cT, true; ndrange=(g.nx, g.ny, nz))
    end
  else
    cV = 1 / (EPS0_SI * w.eps * π * hz)
    _dwa_lam_ka!(backend)(lam, sc.rho, g, 0.0, true, false; ndrange=nz)
    _dwa_prof_ka!(backend)(H, lam, M, 1, 1, 0.0, w.eps, hz, nz, w.nky, false; ndrange=nz)
    _dwa_add_ka!(backend)(sc.F, H, g, 0.0, true, cV, 0.0, false; ndrange=(g.nx, g.ny, nz))
  end
  _sync(backend)
  return sc
end
