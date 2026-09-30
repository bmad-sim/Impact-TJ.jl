# Distributed-memory communication layer.
#
# Particles are distributed across MPI ranks; the space-charge mesh is replicated on
# every rank and assembled with an all-reduce. All communication goes through the
# small interface below so the serial build needs no MPI. The MPI implementation lives
# in the `ImpactTJMPIExt` package extension (loaded with `using MPI`).

abstract type AbstractComm end

"Single-process communicator (no MPI)."
struct SerialComm <: AbstractComm end

nranks(::SerialComm) = 1
rank(::SerialComm) = 0
isroot(c::AbstractComm) = rank(c) == 0
allreduce_sum!(::SerialComm, x::AbstractArray) = x
allreduce_sum(::SerialComm, x::Number) = x
allreduce_min(::SerialComm, x) = x
allreduce_max(::SerialComm, x) = x
isroot(::SerialComm) = true
"Concatenate `x` (a N×k matrix) over ranks along dimension 1."
allgather_rows(::SerialComm, x::AbstractMatrix) = x
barrier(::SerialComm) = nothing
bcast(::SerialComm, x) = x

"""
    default_comm()

Returns the communicator used when none is given: the MPI world communicator if the MPI
extension is loaded and MPI is initialized with more than one rank, else `SerialComm()`.
"""
function default_comm()
  c = _MPI_HOOK[]()
  return isnothing(c) ? SerialComm() : c
end
const _MPI_HOOK = Ref{Function}(() -> nothing)

# Element-wise reductions on NTuples (used for fused diagnostics reductions).
allreduce_sum(c::AbstractComm, x::NTuple{N,Float64}) where {N} = Tuple(allreduce_sum!(c, collect(x)))
allreduce_sum(::SerialComm, x::NTuple{N,Float64}) where {N} = x
