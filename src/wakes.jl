# Short range structure wakefields.
#
# The wake fields at grid slice i are
#   Ez(i) = -Σ_{j≥i} W_L(s_j - s_i) λ_j h      (self term with weight 1/2)
#   Ex(i) =  Σ_{j>i} W_T(s_j - s_i) x̄_j λ_j h  (and similarly Ey)
# with λ the line charge density [C/m] and x̄ the slice centroid. The sums are done
# directly (O(nz²) with nz ≤ a few hundred slices is negligible).

wake_long(w::BaneWake, s, h) = begin
  s00 = 0.41 * (w.a / w.L)^0.8 * (w.g / w.L)^1.6 * w.a
  Z0_SI * C_LIGHT_SI * exp(-sqrt(s / s00)) / (π * w.a^2)
end
wake_tran(w::BaneWake, s) = begin
  s0 = 0.169 * w.a * (w.a / w.L)^1.17 * (w.g / w.a)^0.38
  4 * Z0_SI * C_LIGHT_SI * s0 / (π * w.a^4) * (1 - (1 + sqrt(s / s0)) * exp(-sqrt(s / s0)))
end

# The 1/√s term is averaged over the slice to remove the singularity at s = 0.
wake_long(::BTWWake, s, h) = 1.0e12 * (1226 * exp(-sqrt(s / 3.0e-4)) +
  0.494 * 2 * (sqrt(s + h / 2) - sqrt(max(s - h / 2, 0.0))) / h + 494 * sqrt(s))
wake_tran(::BTWWake, s) = 1.0e12 * (2.8e4 * (1 - (1 + sqrt(s / 1.2e-4)) * exp(-sqrt(s / 1.2e-4))) +
  1.4e4 * sqrt(s))

wake_long(::TeslaWake, s, h) = 0.78688 * 1.0e12 * 38.1 * (1.165 * exp(-sqrt(s / 3.65e-3)) - 0.165)
wake_tran(::TeslaWake, s) = 0.78688 * 1.0e12 * 121 * (1 - (1 + sqrt(s / 0.92e-3)) * exp(-sqrt(s / 0.92e-3)))

wake_long(::Tesla3p9Wake, s, h) = 0.769324 * 1.0e12 * 229.8 * exp(-sqrt(s / 0.84e-3))
wake_tran(::Tesla3p9Wake, s) = 0.0

function _tab(v, L, s)
  n = length(v)
  h = L / (n - 1)
  i = min(floor(Int, s / h) + 1, n - 1)
  return v[i] + (v[i+1] - v[i]) * (s - (i - 1) * h) / h
end
wake_long(w::TabulatedWake, s, h) = _tab(w.Wz, w.L, s)
wake_tran(w::TabulatedWake, s) = _tab(w.Wx, w.L, s)
wake_tran_y(w::TabulatedWake, s) = _tab(w.Wy, w.L, s)
wake_tran_y(w::WakeModel, s) = wake_tran(w, s)

"""
    line_moments!(λ, xs, ys, bin, zmin, hz, cathode, comm)

Linear deposition of charge, charge·x and charge·y on the z grid (nz nodes from zmin
with spacing hz). Returns line density [C/m] and slice centroids.
"""
function line_moments(b::ParticleBin, zmin, hz, nz, cathode::Bool, comm::AbstractComm)
  b.r isa Array || return _line_moments_device(b, zmin, hz, nz, cathode, comm)
  q = zeros(nz); qx = zeros(nz); qy = zeros(nz)
  r = b.r; w = b.w; st = b.state
  @inbounds for p in axes(r, 1)
    st[p] == STATE_ALIVE || continue
    z = r[p, 5]
    (cathode && z <= 0) && continue
    k, f = _cic_index(z, zmin, 1 / hz, nz)
    wp = w[p]
    q[k+1] += (1 - f) * wp; q[k+2] += f * wp
    qx[k+1] += (1 - f) * wp * r[p, 1]; qx[k+2] += f * wp * r[p, 1]
    qy[k+1] += (1 - f) * wp * r[p, 3]; qy[k+2] += f * wp * r[p, 3]
  end
  allreduce_sum!(comm, q); allreduce_sum!(comm, qx); allreduce_sum!(comm, qy)
  xs = [q[k] != 0 ? qx[k] / q[k] : 0.0 for k in 1:nz]
  ys = [q[k] != 0 ? qy[k] / q[k] : 0.0 for k in 1:nz]
  return q ./ hz, xs, ys
end

@kernel function _line_moments_ka!(Q, @Const(r), @Const(w), @Const(state), zmin, hzi, nz, cathode::Bool)
  p = @index(Global)
  @inbounds if state[p] == STATE_ALIVE && !(cathode && r[p, 5] <= 0)
    k, f = _cic_index(r[p, 5], zmin, hzi, nz)
    wp = w[p]; x = r[p, 1]; y = r[p, 3]
    Atomix.@atomic Q[k+1, 1] += (1 - f) * wp
    Atomix.@atomic Q[k+2, 1] += f * wp
    Atomix.@atomic Q[k+1, 2] += (1 - f) * wp * x
    Atomix.@atomic Q[k+2, 2] += f * wp * x
    Atomix.@atomic Q[k+1, 3] += (1 - f) * wp * y
    Atomix.@atomic Q[k+2, 3] += f * wp * y
  end
end

function _line_moments_device(b::ParticleBin, zmin, hz, nz, cathode::Bool, comm::AbstractComm)
  Q = fill!(similar(b.w, nz, 3), 0.0)
  N = size(b.r, 1)
  backend = get_backend(Q)
  N > 0 && _line_moments_ka!(backend)(Q, b.r, b.w, b.state, zmin, 1 / hz, nz, cathode; ndrange=N)
  _sync(backend)
  allreduce_sum!(comm, Q)
  q = view(Q, :, 1)
  xs = ifelse.(q .!= 0, view(Q, :, 2) ./ q, 0.0)
  ys = ifelse.(q .!= 0, view(Q, :, 3) ./ q, 0.0)
  return q ./ hz, xs, ys
end

"Wake fields (Ex, Ey, Ez) on the nz slices for line density λ and slice centroids."
function wake_fields(model::WakeModel, λ, xs, ys, hz)
  nz = length(λ)
  WL = [wake_long(model, (m) * hz, hz) for m in 0:nz-1]
  WT = [m == 0 ? 0.0 : wake_tran(model, m * hz) for m in 0:nz-1]
  WTy = [m == 0 ? 0.0 : wake_tran_y(model, m * hz) for m in 0:nz-1]
  if !(λ isa Array)   # device: the wake tables (functions of hz only) are uploaded
    W = copyto!(similar(λ, nz, 3), hcat(WL, WT, WTy))
    E = similar(λ, nz, 3)
    backend = get_backend(E)
    _wake_conv_ka!(backend)(E, W, λ, xs, ys, hz, nz; ndrange=nz)
    _sync(backend)
    return view(E, :, 1), view(E, :, 2), view(E, :, 3)
  end
  ex = zeros(nz); ey = zeros(nz); ez = zeros(nz)
  Threads.@threads for i in 1:nz
    ex[i], ey[i], ez[i] = _wake_conv(i, WL, WT, WTy, λ, xs, ys, hz, nz)
  end
  return ex, ey, ez
end

@inline function _wake_conv(i, WL, WT, WTy, λ, xs, ys, hz, nz)
  @inbounds begin
    sz = 0.5 * WL[1] * λ[i]; sx = 0.0; sy = 0.0
    for j in i+1:nz
      m = j - i
      sz += WL[m+1] * λ[j]
      sx += WT[m+1] * λ[j] * xs[j]
      sy += WTy[m+1] * λ[j] * ys[j]
    end
    return sx * hz, sy * hz, -sz * hz
  end
end

@kernel function _wake_conv_ka!(E, @Const(W), @Const(λ), @Const(xs), @Const(ys), hz, nz)
  i = @index(Global)
  ex, ey, ez = _wake_conv(i, view(W, :, 1), view(W, :, 2), view(W, :, 3), λ, xs, ys, hz, nz)
  @inbounds begin
    E[i, 1] = ex; E[i, 2] = ey; E[i, 3] = ez
  end
end

@kernel function _add_slice_ka!(F, @Const(ex), @Const(ey), @Const(ez))
  i, j, k = @index(Global, NTuple)
  @inbounds begin
    F[1, i, j, k] += ex[k]; F[2, i, j, k] += ey[k]; F[3, i, j, k] += ez[k]
  end
end

"Add slice fields (functions of the z index only) to the space-charge field grid."
function add_slice_fields!(s::SpaceChargeSolver, ex, ey, ez)
  backend = get_backend(s.F)
  _add_slice_ka!(backend)(s.F, ex, ey, ez; ndrange=(s.nx, s.ny, s.nz))
  _sync(backend)
  return s.F
end

# ---------------------------------------------------------------------------------------
# 1D coherent synchrotron radiation (CSR) wake including entrance and exit transients
# (integrated Green function method of Mitchell, Qiang, Ryne, NIM-A 715, 119 (2013)).

function _csrA(φh, ssh, xk2)
  bb = 2φh + φh^3 / 3 - 2ssh
  cc = φh^2 + φh^4 / 12 - 2ssh * φh
  yh = (-bb + sqrt(max(bb * bb - 4cc, 0.0))) / 2
  p = φh + yh
  return xk2 * (-(2p + φh^3) / (p^2 + φh^4 / 4) + 1 / ssh)
end

function _csrB(ssh, xk2)
  aa = sqrt(64 + 144 * ssh^2)
  u = cbrt(aa + 12ssh) - cbrt(aa - 12ssh)
  return xk2 * (-4u * (u^2 + 8) / ((u^2 + 4) * (u^2 + 12)))
end

function _csrC(φm, xb, ssh, xk2)
  bb = 2(φm + xb) - 2ssh + φm^3 / 3 + φm^2 * xb
  cc = (φm + xb)^2 + φm^2 * (φm^2 + 4φm * xb) / 12 - 2ssh * (φm + xb)
  yh = (-bb + sqrt(max(bb * bb - 4cc, 0.0))) / 2
  pa = φm + xb + yh
  pb = φm * xb + φm^2 / 2
  return xk2 * (-2(pa + φm * pb) / (pa^2 + pb^2) + 1 / ssh)
end

function _csr_psi(xxh, ζ, ψmax)
  f(ψ) = ψ^4 / 12 + xxh * ψ^3 / 3 + ψ^2 + (2xxh - 2ζ) * ψ - 2ζ * xxh + xxh^2
  df(ψ) = ψ^3 / 3 + xxh * ψ^2 + 2ψ + 2xxh - 2ζ
  lo = -2.0e-9; hi = ψmax * (1 + 2.0e-9)
  flo = f(lo)
  ψ = 0.5 * (lo + hi)
  for _ in 1:200    # safeguarded Newton
    fv = f(ψ)
    abs(fv) < 1.0e-14 && break
    if (fv < 0) == (flo < 0)
      lo = ψ; flo = fv
    else
      hi = ψ
    end
    ψn = ψ - fv / df(ψ)
    ψ = (ψn > lo && ψn < hi) ? ψn : 0.5 * (lo + hi)
    hi - lo < 1.0e-15 * max(ψmax, 1.0) && break
  end
  return ψ
end

function _csrD(xb, ψmax, ssh, xk2)
  ψ = _csr_psi(xb, ssh, ψmax)
  pa = ψ + xb
  pb = ψ * (xb + ψ / 2)
  return xk2 * (-2(pa + ψ * pb) / (pa^2 + pb^2) + 1 / ssh)
end

"""
    csr_wake(λ, zmin, hz, R, γ, Lbend) -> Ez

Longitudinal CSR field on the nodes zmin + (i-1) hz, where the node coordinate is the
path length of the node from the (effective) bend entrance. `λ` is the line charge
density on the nodes, `R` the bend radius, `Lbend` the (effective) bend arc length.

With I(s) the integrated Green function (kernel antiderivative) for a source a distance
s behind the observer, Ez(x) = ∫ λ(x-s) dI(s). The line density is taken piecewise linear
between nodes (consistent with the linear deposition) and the integral is split at the
cell boundaries and at the slippage points where the kernel changes form:

    ∫ₐᵇ λ(x-s) I'(s) ds = λ(x-b) I(b) - λ(x-a) I(a) + λ' ∫ₐᵇ I(s) ds,

with the last integral evaluated by adaptive quadrature. This converges with the grid
where the IMPACT-T scheme (piecewise constant λ, slippage point assigned to the cell
below it) has O(h) errors, largest at the bend entrance where the slippage length is much
smaller than a cell.
"""
function csr_wake(λ::AbstractVector{Float64}, zmin, hz, R, γ, Lbend)
  nx = length(λ)
  if !(λ isa Array)
    ez = fill!(similar(λ), 0.0)
    backend = get_backend(ez)
    _csr_ka!(backend)(ez, λ, zmin, hz, R, γ, Lbend; ndrange=nx)
    _sync(backend)
    return ez
  end
  ez = zeros(nx)
  Threads.@threads for i in 2:nx
    ez[i] = _csr_node(λ, i, zmin, hz, R, γ, Lbend)
  end
  return ez
end

@inline function _csr_node(λ, i, zmin, hz, R, γ, Lbend)
  x = zmin + (i - 1) * hz
  x <= 0 && return 0.0
  return COULOMB_K * _csr_point(λ, zmin, hz, x, R, γ, Lbend)
end

@kernel function _csr_ka!(ez, @Const(λ), zmin, hz, R, γ, Lbend)
  i = @index(Global)
  if i >= 2
    @inbounds ez[i] = _csr_node(λ, i, zmin, hz, R, γ, Lbend)
  end
end

# kernel antiderivative I(s) for regime r (1 = A, 2 = B, 3 = C, 4 = D, 0 = none)
@inline function _csr_I(r, s, p)
  s <= 0 && return 0.0
  ssh = s * p.g3R
  r == 1 && return _csrA(p.φh, ssh, p.xk2)
  r == 2 && return _csrB(ssh, p.xk2)
  r == 3 && return _csrC(p.φmh, p.xbh, ssh, p.xk2)
  r == 4 && return _csrD(p.xbh, p.φmh, ssh, p.xk2)
  return 0.0
end

const _GL3X = (-sqrt(3 / 5), 0.0, sqrt(3 / 5))
const _GL3W = (5 / 9, 8 / 9, 5 / 9)

@inline function _gl3(f, a, b)
  h = (b - a) / 2; c = (a + b) / 2
  return h * (_GL3W[1] * f(c + h * _GL3X[1]) + _GL3W[2] * f(c) + _GL3W[3] * f(c + h * _GL3X[3]))
end

"""
Integral of f over [a, b] where f may vary on the scale of the distance to the points
`cs` (kernel breakpoints, s = 0). Intervals are refined geometrically towards the closest
such point, so each interval is smaller than its distance to it; 3-point Gauss-Legendre is
used on every interval. Depth first traversal with an explicit stack (no recursion or
allocation, so it runs inside GPU kernels).
"""
@inline function _graded_integral(f, a0, b0, cs, smin)
  stack = MVector{2 * _CSR_STACK,Float64}(undef)
  stack[1] = a0; stack[2] = b0
  top = 1
  total = 0.0
  while top > 0
    a = stack[2top-1]; b = stack[2top]
    top -= 1
    δ = Inf
    for c in cs
      d = c < a ? a - c : (c > b ? c - b : 0.0)
      δ = min(δ, d)
    end
    if b - a <= max(δ, smin) || top + 2 > _CSR_STACK
      total += _gl3(f, a, b)
      continue
    end
    # split at the singular point inside, else at the middle
    m = a + (b - a) / 2
    for c in cs
      if c >= a && c <= b
        m = c == a ? a + (b - a) / 2 : (c == b ? b - (b - a) / 2 : c)
        break
      end
    end
    if m <= a || m >= b
      total += _gl3(f, a, b)
      continue
    end
    # push the right half first so the left half is done first
    top += 1; stack[2top-1] = m; stack[2top] = b
    top += 1; stack[2top-1] = a; stack[2top] = m
  end
  return total
end
const _CSR_STACK = 128

@inline function _csr_lin(λ, zmin, hz, z)
  nx = length(λ)
  u = (z - zmin) / hz
  (u < 0 || u > nx - 1) && return 0.0
  k = min(unsafe_trunc(Int, floor(u)), nx - 2)
  @inbounds return λ[k+1] + (u - k) * (λ[k+2] - λ[k+1])
end

@inline function _csr_slope(λ, zmin, hz, z)
  nx = length(λ)
  u = (z - zmin) / hz
  (u < 0 || u > nx - 1) && return 0.0
  k = min(unsafe_trunc(Int, floor(u)), nx - 2)
  @inbounds return (λ[k+2] - λ[k+1]) / hz
end

@inline _csr_regime(inside::Bool, s, xslp, xslpN) =
  inside ? (s > xslp ? 1 : 2) : (s > xslp ? 3 : (s > xslpN ? 4 : 0))

function _csr_point(λ, zmin, hz, x, R, γ, Lb)
  φm = Lb / R
  inside = x <= Lb
  if inside
    xslp = x / 2 / γ^2 + x^3 / (24R^2)
    xslpN = 0.0
    brk = (xslp, Inf)
  else
    xb = x - Lb
    xslp = (R * φm + xb) / 2 / γ^2 + R * φm^3 / 24 * (R * φm + 4xb) / (R * φm + xb)
    xslpN = xb / 2 / γ^2
    brk = (xslpN, xslp)
  end
  p = (g3R=γ^3 / R, xk2=γ / R, φh=x / R * γ, φmh=φm * γ, xbh=(x - Lb) * γ / R)
  S = x - zmin
  sing = (0.0, brk[1], brk[2])
  smin = hz
  for bp in brk
    bp > 0 && (smin = min(smin, bp))
  end
  smin *= 1.0e-4
  # piece boundaries in increasing s: 0, the node cell boundaries s = S - m hz
  # (m = kmax, ..., 1) and the kernel breakpoints inside (0, S), then S (merged on the fly)
  kmax = unsafe_trunc(Int, floor(S / hz))
  m = kmax
  nb = 1
  total = 0.0
  a = 0.0
  while true
    # next boundary candidate
    while m >= 1 && S - m * hz <= 0
      m -= 1
    end
    while nb <= 2 && !(brk[nb] > 0 && brk[nb] < S)
      nb += 1
    end
    sg = m >= 1 ? S - m * hz : Inf
    sb = nb <= 2 ? brk[nb] : Inf
    if sg <= sb && sg < Inf
      b = sg; m -= 1
    elseif sb < Inf
      b = sb; nb += 1
    else
      b = S
    end
    if b - a > 0
      rg = _csr_regime(inside, (a + b) / 2, xslp, xslpN)
      if rg != 0
        Ia = _csr_I(rg, a, p); Ib = _csr_I(rg, b, p)
        slope = _csr_slope(λ, zmin, hz, x - (a + b) / 2)
        total += _csr_lin(λ, zmin, hz, x - b) * Ib - _csr_lin(λ, zmin, hz, x - a) * Ia
        if slope != 0
          total += slope * _graded_integral(s -> _csr_I(rg, s, p), a, b, sing, smin)
        end
      end
    end
    a = b
    b == S && break
  end
  return total
end
