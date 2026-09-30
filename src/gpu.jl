# Device-side building blocks (reductions, stream compaction) written with
# KernelAbstractions so that no particle or grid data leaves the GPU. Only the few
# reduced numbers needed by the host control logic are copied back.

# ---------------------------------------------------------------------------------------
# Multi-component reductions
#
# `device_reduce(f, kinds, N, args...)` evaluates f(p, args...) :: NTuple{K,Float64} for
# p = 1..N and reduces every component with + (kind 0), min (kind 1) or max (kind 2).
# Pass 1: each work item accumulates a strided subset in registers, the work group then
# combines its items in local memory; pass 2 combines the group results with a single
# work group. The decomposition depends only on N, so results are deterministic.

const _RED_WG = 64         # work group size
const _RED_MAXG = 1024     # maximum number of groups of pass 1

@inline _ident(kind) = kind == 0x00 ? 0.0 : (kind == 0x01 ? Inf : -Inf)
@inline _comb(kind, a, b) = kind == 0x00 ? a + b : (kind == 0x01 ? min(a, b) : max(a, b))
@inline _combt(kinds::NTuple{K,UInt8}, a::NTuple{K,Float64}, b::NTuple{K,Float64}) where {K} =
  ntuple(i -> _comb(kinds[i], a[i], b[i]), Val(K))

# tree step of the work group reduction of the K×WG local array
@inline function _tree_step!(buf, kinds::NTuple{K,UInt8}, li, s) where {K}
  if li <= s
    for k in 1:K
      @inbounds buf[k, li] = _comb(kinds[k], buf[k, li], buf[k, li+s])
    end
  end
end

@kernel function _reduce1_ka!(part, f, kinds::NTuple{K,UInt8}, N, stride, args) where {K}
  # (index assignments are re-evaluated in every section between @synchronize on CPUs)
  li = @index(Local, Linear)
  g = @index(Group, Linear)
  buf = @localmem Float64 (K, _RED_WG)
  acc = ntuple(i -> _ident(kinds[i]), Val(K))
  p = (g - 1) * _RED_WG + li
  while p <= N
    acc = _combt(kinds, acc, f(p, args...))
    p += stride
  end
  for k in 1:K
    @inbounds buf[k, li] = acc[k]
  end
  @synchronize
  _tree_step!(buf, kinds, li, 32)
  @synchronize
  _tree_step!(buf, kinds, li, 16)
  @synchronize
  _tree_step!(buf, kinds, li, 8)
  @synchronize
  _tree_step!(buf, kinds, li, 4)
  @synchronize
  _tree_step!(buf, kinds, li, 2)
  @synchronize
  _tree_step!(buf, kinds, li, 1)
  @synchronize
  if li == 1
    for k in 1:K
      @inbounds part[k, g] = buf[k, 1]
    end
  end
end

@inline _part_column(p, part::AbstractMatrix, ::Val{K}) where {K} = ntuple(k -> @inbounds(part[k, p]), Val(K))

"""
    device_reduce(f, kinds, N, args...) -> NTuple{K,Float64}

Reduce f(p, args...) over p = 1..N on the device of the arrays in `args`. `kinds[k]` is
0x00 (sum), 0x01 (min) or 0x02 (max). `f` must be a GPU compatible function.
"""
function device_reduce(f, kinds::NTuple{K,UInt8}, N::Integer, backend, args...) where {K}
  N == 0 && return ntuple(i -> _ident(kinds[i]), Val(K))
  G = min(cld(N, _RED_WG), _RED_MAXG)
  part = KernelAbstractions.allocate(backend, Float64, K, G)
  _reduce1_ka!(backend, _RED_WG)(part, f, kinds, N, G * _RED_WG, args; ndrange=G * _RED_WG)
  out = KernelAbstractions.allocate(backend, Float64, K, 1)
  _reduce1_ka!(backend, _RED_WG)(out, _part_column, kinds, G, _RED_WG, (part, Val(K)); ndrange=_RED_WG)
  _sync(backend)
  h = Array(out)
  return ntuple(k -> h[k], Val(K))
end

# ---------------------------------------------------------------------------------------
# Order preserving stream compaction of particles
#
# The particles are split into chunks of _CMP_B; pass 1 counts the survivors of every
# chunk, pass 2 scans the counts (one work item per group of chunks, then a short serial
# scan of the group totals) and pass 3 moves the survivors of each chunk to their place.

const _CMP_B = 32

@kernel function _count_alive_ka!(cnt, @Const(state), N)
  c = @index(Global)
  lo = (c - 1) * _CMP_B + 1
  hi = min(c * _CMP_B, N)
  n = 0
  @inbounds for p in lo:hi
    n += state[p] == STATE_ALIVE
  end
  @inbounds cnt[c] = n
end

# exclusive scan of cnt in place, in segments of length S (segment totals in tot)
@kernel function _scan_seg_ka!(cnt, tot, S, n)
  s = @index(Global)
  lo = (s - 1) * S + 1
  hi = min(s * S, n)
  acc = 0
  @inbounds for i in lo:hi
    v = cnt[i]
    cnt[i] = acc
    acc += v
  end
  @inbounds tot[s] = acc
end

@kernel function _scan_tot_ka!(tot, n)
  acc = 0
  @inbounds for i in 1:n
    v = tot[i]
    tot[i] = acc
    acc += v
  end
end

@kernel function _compact_ka!(r2, w2, id2, @Const(r), @Const(w), @Const(id), @Const(state),
                              @Const(cnt), @Const(tot), S, N)
  c = @index(Global)
  lo = (c - 1) * _CMP_B + 1
  hi = min(c * _CMP_B, N)
  @inbounds begin
    q = cnt[c] + tot[(c - 1) ÷ S + 1]
    for p in lo:hi
      if state[p] == STATE_ALIVE
        q += 1
        for j in 1:6
          r2[q, j] = r[p, j]
        end
        w2[q] = w[p]
        id2[q] = id[p]
      end
    end
  end
end

"Indices-free compaction of the live particles of device arrays; returns (r, w, id)."
function device_compact(r, w, id, state, nkeep)
  N = size(r, 1)
  backend = get_backend(r)
  nc = cld(N, _CMP_B)
  S = max(1, ceil(Int, sqrt(nc)))
  ns = cld(nc, S)
  cnt = KernelAbstractions.allocate(backend, Int, nc)
  tot = KernelAbstractions.allocate(backend, Int, ns)
  _count_alive_ka!(backend)(cnt, state, N; ndrange=nc)
  _scan_seg_ka!(backend)(cnt, tot, S, nc; ndrange=ns)
  _scan_tot_ka!(backend)(tot, ns; ndrange=1)
  r2 = similar(r, nkeep, 6); w2 = similar(w, nkeep); id2 = similar(id, nkeep)
  _compact_ka!(backend)(r2, w2, id2, r, w, id, state, cnt, tot, S, N; ndrange=nc)
  _sync(backend)
  return r2, w2, id2
end
