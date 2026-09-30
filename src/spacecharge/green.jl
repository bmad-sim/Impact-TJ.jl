# Integrated Green functions (IGF) for the open-boundary Poisson equation.
#
# The potential of a charge distribution on a uniform grid is φᵢ = Σⱼ Gᵢ₋ⱼ Qⱼ where
# G(r) = 1/(4πε₀ V) ∫_cell dV'/|r - r'| is the Coulomb kernel averaged over a grid cell
# (V = hx hy hz). The cell integral is evaluated in closed form from the antiderivative
#
#   F(x,y,z) = -z²/2 atan(xy/(zr)) - y²/2 atan(xz/(yr)) - x²/2 atan(yz/(xr))
#              + yz ln(x+r) + xz ln(y+r) + xy ln(z+r)
#
# at the cell corners. This is accurate for any grid aspect ratio (important for the long
# or thin bunches of photoinjectors), unlike the point-sampled kernel 1/r.

# ln(a + r) with r ≥ |a|, computed without cancellation for negative a.
@inline function _lnp(a, r)
  a >= 0 ? log(a + r) : log(muladd(r, r, -a * a) / (r - a))
end

@inline function _igf_F(x, y, z)
  r = sqrt(x * x + y * y + z * z)
  return (-z * z / 2 * atan(x * y / (z * r)) - y * y / 2 * atan(x * z / (y * r)) -
          x * x / 2 * atan(y * z / (x * r)) + y * z * _lnp(x, r) + x * z * _lnp(y, r) +
          x * y * _lnp(z, r))
end

"""
    igf!(G, hx, hy, hz, z0; far=5.0)

Fill `G[i+1, j+1, k+1] = G(i hx, j hy, z0 + k hz)` (0-based i, j, k) with the cell-averaged
Coulomb kernel including the factor 1/(4πε₀). `G` must be a CPU array.

Nodes closer than `far`·max(h) to the origin use the exact closed-form cell integral.
Farther nodes use the multipole (Taylor) expansion of the cell average of 1/r to fourth
order in h/r (relative error ≈ 1e-7 at r = 5h, verified in the tests), which avoids both the
expensive transcendental functions and the cancellation of the 8-corner difference.
"""
function igf!(G::Array{Float64,3}, hx, hy, hz, z0; far=5.0)
  n1, n2, n3 = size(G)
  hm = max(hx, hy, hz)
  R = far * hm
  R2 = R^2
  c = COULOMB_K / (hx * hy * hz)
  hx2, hy2, hz2 = hx^2, hy^2, hz^2
  # index box containing all nodes with r ≤ R (exact IGF from shared cell corners there)
  bi = min(n1, floor(Int, R / hx) + 1)
  bj = min(n2, floor(Int, R / hy) + 1)
  kin = [k for k in 1:n3 if abs(z0 + (k - 1) * hz) <= R]
  if !isempty(kin)
    k1, k2 = first(kin), last(kin)
    nk = k2 - k1 + 1
    Fc = Array{Float64,3}(undef, bi + 1, bj + 1, nk + 1)
    Threads.@threads for kk in 0:nk
      z = z0 + (k1 - 1 + kk - 0.5) * hz
      @inbounds for j in 0:bj, i in 0:bi
        Fc[i+1, j+1, kk+1] = _igf_F((i - 0.5) * hx, (j - 0.5) * hy, z)
      end
    end
    Threads.@threads for kk in 1:nk
      k = k1 + kk - 1
      @inbounds for j in 1:bj, i in 1:bi
        G[i, j, k] = c * (Fc[i+1, j+1, kk+1] - Fc[i, j+1, kk+1] - Fc[i+1, j, kk+1] + Fc[i, j, kk+1] -
                          Fc[i+1, j+1, kk] + Fc[i, j+1, kk] + Fc[i+1, j, kk] - Fc[i, j, kk])
      end
    end
  else
    k1, k2 = 1, 0
  end
  Threads.@threads for k in 1:n3
    z = z0 + (k - 1) * hz
    z2 = z * z
    inbox_k = k1 <= k <= k2
    @inbounds for j in 1:n2
      y = (j - 1) * hy
      y2 = y * y
      for i in 1:n1
        (inbox_k && i <= bi && j <= bj) && continue
        x = (i - 1) * hx
        x2 = x * x
        r2 = x2 + y2 + z2
        ir2 = 1 / r2
        ir = sqrt(ir2)
        ir5 = ir * ir2 * ir2
        ir9 = ir5 * ir2 * ir2
        t2 = (hx2 * (3x2 - r2) + hy2 * (3y2 - r2) + hz2 * (3z2 - r2)) * ir5 / 24
        r4 = r2 * r2
        t4 = (hx2^2 * (105x2^2 - 90x2 * r2 + 9r4) + hy2^2 * (105y2^2 - 90y2 * r2 + 9r4) +
              hz2^2 * (105z2^2 - 90z2 * r2 + 9r4)) * ir9 / 1920 +
             (hx2 * hy2 * (105x2 * y2 - 15(x2 + y2) * r2 + 3r4) +
              hx2 * hz2 * (105x2 * z2 - 15(x2 + z2) * r2 + 3r4) +
              hy2 * hz2 * (105y2 * z2 - 15(y2 + z2) * r2 + 3r4)) * ir9 / 576
        G[i, j, k] = COULOMB_K * (ir + t2 + t4)
      end
    end
  end
  return G
end

# ---------------------------------------------------------------------------------------
# Device (GPU) version: the Green function on the doubled grid is computed directly on the
# device, with the same near (exact) / far (expansion) split as `igf!`.

@inline function _igf_node(x, y, z, hx, hy, hz, R2)
  x2 = x * x; y2 = y * y; z2 = z * z
  r2 = x2 + y2 + z2
  if r2 > R2
    ir2 = 1 / r2
    ir = sqrt(ir2)
    ir5 = ir * ir2 * ir2
    ir9 = ir5 * ir2 * ir2
    hx2 = hx * hx; hy2 = hy * hy; hz2 = hz * hz
    t2 = (hx2 * (3x2 - r2) + hy2 * (3y2 - r2) + hz2 * (3z2 - r2)) * ir5 / 24
    r4 = r2 * r2
    t4 = (hx2^2 * (105x2^2 - 90x2 * r2 + 9r4) + hy2^2 * (105y2^2 - 90y2 * r2 + 9r4) +
          hz2^2 * (105z2^2 - 90z2 * r2 + 9r4)) * ir9 / 1920 +
         (hx2 * hy2 * (105x2 * y2 - 15(x2 + y2) * r2 + 3r4) +
          hx2 * hz2 * (105x2 * z2 - 15(x2 + z2) * r2 + 3r4) +
          hy2 * hz2 * (105y2 * z2 - 15(y2 + z2) * r2 + 3r4)) * ir9 / 576
    return COULOMB_K * (ir + t2 + t4)
  else
    xa = x - hx / 2; xb = x + hx / 2
    ya = y - hy / 2; yb = y + hy / 2
    za = z - hz / 2; zb = z + hz / 2
    c = COULOMB_K / (hx * hy * hz)
    return c * (_igf_F(xb, yb, zb) - _igf_F(xa, yb, zb) - _igf_F(xb, ya, zb) + _igf_F(xa, ya, zb) -
                _igf_F(xb, yb, za) + _igf_F(xa, yb, za) + _igf_F(xb, ya, za) - _igf_F(xa, ya, za))
  end
end

@kernel function _green_doubled_ka!(D, nx, ny, nz, hx, hy, hz, z0, image::Bool, R2)
  i, j, k = @index(Global, NTuple)
  fi = i - 1 <= nx ? i - 1 : 2nx - (i - 1)
  fj = j - 1 <= ny ? j - 1 : 2ny - (j - 1)
  fk = image ? k - 1 : (k - 1 <= nz ? k - 1 : 2nz - (k - 1))
  @inbounds D[i, j, k] = _igf_node(fi * hx, fj * hy, z0 + fk * hz, hx, hy, hz, R2)
end

"Fill the doubled (2nx, 2ny, 2nz) array `D` with the (even extended) Green function on the device."
function green_doubled!(D, nx, ny, nz, hx, hy, hz, z0, image::Bool; far=5.0)
  backend = get_backend(D)
  R2 = (far * max(hx, hy, hz))^2
  _green_doubled_ka!(backend)(D, nx, ny, nz, hx, hy, hz, z0, image, R2; ndrange=size(D))
  _sync(backend)
  return D
end
