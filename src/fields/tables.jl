# Field data tables.
#
# All tabulated field data of a lattice is stored in a single flat `Vector{Float64}`
# ("field buffer") and each field element holds only isbits offsets/metadata into it.
# This keeps every field element an isbits struct so that the same kernel code runs on
# CPU threads and GPUs.

"""
On-axis function table f(z) on a uniform grid with its first three derivatives stored at
every node: buf[off + 4i + 1:4] = (f, f', f'', f''') for node i = 0..n-1.

With `hermite=true`, f, f' and f'' are interpolated with cubic Hermite polynomials using
the next derivative as slope (4th order accurate); otherwise linear interpolation is used.
If `period > 0` the data is repeated periodically (used for traveling wave cells).
"""
struct AxisTable
  off::Int
  n::Int
  z0::Float64
  dz::Float64
  period::Float64
  hermite::Bool
end

AxisTable() = AxisTable(0, 0, 0.0, 1.0, 0.0, false)

@inline function _herm(u, dz, f0, d0, f1, d1)
  u2 = u * u
  h00 = (1 + 2u) * (1 - u)^2
  h10 = u * (1 - u)^2
  h01 = u2 * (3 - 2u)
  h11 = u2 * (u - 1)
  return h00 * f0 + h10 * dz * d0 + h01 * f1 + h11 * dz * d1
end

"""
    axis_eval(buf, tab, z) -> (f, f', f'', f''')

Evaluate an on-axis table at local position `z`. Returns zeros outside the table.
"""
@inline function axis_eval(buf, tab::AxisTable, z::Float64)
  u = z - tab.z0
  if tab.period > 0
    eps = 1.0e-15 * tab.period
    nzl = unsafe_trunc(Int, (u - eps) / tab.period)
    u -= nzl * tab.period
  end
  zspan = (tab.n - 1) * tab.dz
  if !(u >= 0.0 && u <= zspan) || tab.n < 2
    return (0.0, 0.0, 0.0, 0.0)
  end
  s = u / tab.dz
  i = min(unsafe_trunc(Int, s), tab.n - 2)
  w = s - i
  j = tab.off + 4i
  @inbounds begin
    f0 = buf[j+1]; a0 = buf[j+2]; b0 = buf[j+3]; c0 = buf[j+4]
    f1 = buf[j+5]; a1 = buf[j+6]; b1 = buf[j+7]; c1 = buf[j+8]
  end
  if tab.hermite
    dz = tab.dz
    return (_herm(w, dz, f0, a0, f1, a1), _herm(w, dz, a0, b0, a1, b1),
            _herm(w, dz, b0, c0, b1, c1), c0 + w * (c1 - c0))
  else
    return (f0 + w * (f1 - f0), a0 + w * (a1 - a0), b0 + w * (b1 - b0), c0 + w * (c1 - c0))
  end
end

# ---------------------------------------------------------------------------------------
# Host-side builder for the field buffer

"Accumulates tabulated field data while a lattice is being compiled."
struct BufferBuilder
  data::Vector{Float64}
end
BufferBuilder() = BufferBuilder(Float64[])

"Append `v` to the buffer and return its 0-based offset."
function push_data!(bb::BufferBuilder, v::AbstractVector{<:Real})
  off = length(bb.data)
  append!(bb.data, v)
  return off
end

"""
    axis_table(bb, z0, dz, f, fp, fpp, fppp; period=0, hermite=true)

Store an on-axis table (values and first three derivatives on a uniform grid starting at
local position `z0`).
"""
function axis_table(bb::BufferBuilder, z0, dz, f, fp, fpp, fppp; period=0.0, hermite=true)
  n = length(f)
  v = Vector{Float64}(undef, 4n)
  for i in 1:n
    v[4i-3] = f[i]; v[4i-2] = fp[i]; v[4i-1] = fpp[i]; v[4i] = fppp[i]
  end
  off = push_data!(bb, v)
  return AxisTable(off, n, Float64(z0), Float64(dz), Float64(period), hermite)
end

"""
    FourierSeries(coefs, zlen, zc)

Truncated Fourier series  f(z) = a0/2 + Σₙ aₙ cos(2πn(z-zc)/zlen) + bₙ sin(2πn(z-zc)/zlen)
with `coefs = [a0, a1, b1, a2, b2, ...]` (IMPACT-T `rfdata` convention).
"""
struct FourierSeries
  coefs::Vector{Float64}
  zlen::Float64
  zc::Float64
end

"Evaluate a Fourier series and its first three derivatives at `z`."
function fourier_eval(fs::FourierSeries, z)
  c = fs.coefs
  f = c[1] / 2; f1 = 0.0; f2 = 0.0; f3 = 0.0
  nmodes = (length(c) - 1) ÷ 2
  kz = 2π / fs.zlen
  ζ = z - fs.zc
  for n in 1:nmodes
    a = c[2n]; b = c[2n+1]
    k = n * kz
    s, co = sincos(k * ζ)
    f += a * co + b * s
    f1 += k * (-a * s + b * co)
    f2 += k^2 * (-a * co - b * s)
    f3 += k^3 * (a * s - b * co)
  end
  return f, f1, f2, f3
end

"""
    fourier_axis_table(bb, fs, zstart, zend; points_per_wavelength=128)

Tabulate a Fourier series on [zstart, zend] finely enough that cubic Hermite
interpolation reproduces the series to ≈1e-9 relative accuracy. Evaluating the table is
O(1) per particle instead of O(number of modes).
"""
function fourier_axis_table(bb::BufferBuilder, fs::FourierSeries, zstart, zend;
                            points_per_wavelength=128)
  nmodes = max((length(fs.coefs) - 1) ÷ 2, 1)
  λmin = fs.zlen / nmodes
  dz_target = λmin / points_per_wavelength
  span = zend - zstart
  n = clamp(ceil(Int, span / dz_target) + 1, 257, 2_000_001)
  dz = span / (n - 1)
  f = zeros(n); f1 = zeros(n); f2 = zeros(n); f3 = zeros(n)
  Threads.@threads for i in 1:n
    f[i], f1[i], f2[i], f3[i] = fourier_eval(fs, zstart + (i - 1) * dz)
  end
  return axis_table(bb, zstart, dz, f, f1, f2, f3; hermite=true)
end
