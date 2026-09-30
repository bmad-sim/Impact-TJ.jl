module ImpactTJMPIExt

# MPI parallelization: particles are distributed over the ranks, the space-charge grid is
# replicated and assembled with all-reduces.

using ImpactTJ
using MPI

"MPI communicator for ImpactTJ."
struct MPIComm <: ImpactTJ.AbstractComm
  comm::MPI.Comm
end

ImpactTJ.nranks(c::MPIComm) = MPI.Comm_size(c.comm)
ImpactTJ.rank(c::MPIComm) = MPI.Comm_rank(c.comm)
ImpactTJ.barrier(c::MPIComm) = MPI.Barrier(c.comm)

# GPU-aware MPI (CUDA/ROCm): device buffers are passed to MPI directly
_gpu_aware() = (isdefined(MPI, :has_cuda) && MPI.has_cuda()) ||
               (isdefined(MPI, :has_rocm) && MPI.has_rocm())

function ImpactTJ.allreduce_sum!(c::MPIComm, x::AbstractArray)
  if x isa Array || _gpu_aware()
    MPI.Allreduce!(x, +, c.comm)
  else   # device array without GPU-aware MPI: stage through host memory
    h = Array(x)
    MPI.Allreduce!(h, +, c.comm)
    copyto!(x, h)
  end
  return x
end
ImpactTJ.allreduce_sum(c::MPIComm, x::Number) = MPI.Allreduce(x, +, c.comm)
ImpactTJ.allreduce_sum(c::MPIComm, x::NTuple{N,Float64}) where {N} = Tuple(MPI.Allreduce(collect(x), +, c.comm))
ImpactTJ.allreduce_min(c::MPIComm, x::AbstractVector) = MPI.Allreduce(Vector{Float64}(x), min, c.comm)
ImpactTJ.allreduce_max(c::MPIComm, x::AbstractVector) = MPI.Allreduce(Vector{Float64}(x), max, c.comm)
ImpactTJ.allreduce_min(c::MPIComm, x::Number) = MPI.Allreduce(x, min, c.comm)
ImpactTJ.allreduce_max(c::MPIComm, x::Number) = MPI.Allreduce(x, max, c.comm)
ImpactTJ.bcast(c::MPIComm, x) = MPI.bcast(x, 0, c.comm)

function ImpactTJ.allgather_rows(c::MPIComm, x::AbstractMatrix)
  xh = Matrix{Float64}(x)
  n, k = size(xh)
  counts = MPI.Allgather(Int32(n), c.comm)
  N = Int(sum(counts))
  out = Matrix{Float64}(undef, N, k)
  for j in 1:k
    col = Vector{Float64}(undef, N)
    MPI.Allgatherv!(xh[:, j], MPI.VBuffer(col, counts), c.comm)
    out[:, j] .= col
  end
  return out
end

function __init__()
  ImpactTJ._MPI_HOOK[] = function ()
    MPI.Initialized() || return nothing
    MPI.Finalized() && return nothing
    MPI.Comm_size(MPI.COMM_WORLD) > 1 || return nothing
    return MPIComm(MPI.COMM_WORLD)
  end
end

end
