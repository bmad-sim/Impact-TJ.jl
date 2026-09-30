# Macroparticle storage.
#
# A `ParticleBin` is IMPACT-T's "bunch/bin": a set of macroparticles of a single species
# whose space-charge field is computed in its own rest frame. Several bins are used to
# model beams with a large energy spread (e.g. just after the cathode) or several species.
#
# Coordinates (structure-of-arrays, one column per coordinate so GPU access is coalesced):
#   r[:,1] = x [m]    r[:,2] = γβx
#   r[:,3] = y [m]    r[:,4] = γβy
#   r[:,5] = z [m]    r[:,6] = γβz
# All particles share the lab time `t` of the simulation.

mutable struct ParticleBin{M<:AbstractMatrix{Float64},V<:AbstractVector{Float64},
                           S<:AbstractVector{UInt8},I<:AbstractVector{Int}}
  species::Species
  mc2::Float64      # rest energy [eV]
  charge::Float64   # charge [e] (signed)
  r::M              # N×6 coordinates
  w::V              # macroparticle charge [C] (signed)
  state::S          # STATE_ALIVE / STATE_LOST
  id::I             # global particle id
end

"""
    ParticleBin(species, r, w; mc2=massof(species), charge=chargeof(species), id=1:N)

Create a bin of macroparticles. `r` is an N×6 matrix (see the file header for the
coordinate definition) and `w` the macroparticle charges in Coulomb (a scalar is
broadcast to every particle).
"""
function ParticleBin(species::Species, r::AbstractMatrix, w; mc2=massof(species),
                     charge=chargeof(species), id=nothing)
  N = size(r, 1)
  size(r, 2) == 6 || error("particle coordinate matrix must be N×6")
  rr = Matrix{Float64}(r)
  ww = w isa Number ? fill(Float64(w), N) : Vector{Float64}(w)
  ids = isnothing(id) ? collect(1:N) : Vector{Int}(id)
  st = fill(STATE_ALIVE, N)
  return ParticleBin(species, Float64(mc2), Float64(charge), rr, ww, st, ids)
end

nparticles(b::ParticleBin) = size(b.r, 1)
"Charge-to-mass ratio q/(mc²) [1/V]."
qm(b::ParticleBin) = b.charge / b.mc2
total_charge(b::ParticleBin, comm::AbstractComm=SerialComm()) = allreduce_sum(comm, sum(b.w))

function Base.show(io::IO, b::ParticleBin)
  print(io, "ParticleBin($(nameof(b.species)), N=$(nparticles(b)))")
end

"Move bin data to the array type of `proto` (e.g. a GPU array)."
function to_device(b::ParticleBin, proto::AbstractArray)
  A = Base.typename(typeof(proto)).wrapper
  ParticleBin(b.species, b.mc2, b.charge, A(b.r), A(b.w), A(b.state), A(b.id))
end

"Remove particles whose state is not alive. Returns the number removed."
function compact!(b::ParticleBin)
  if b.r isa Array
    keep = b.state .== STATE_ALIVE
    nkeep = count(keep)
    nlost = length(keep) - nkeep
    nlost == 0 && return 0
    idx = findall(keep)
    b.r = b.r[idx, :]
    b.w = b.w[idx]
    b.id = b.id[idx]
    b.state = b.state[idx]
    return nlost
  end
  # device arrays: order preserving compaction on the device
  N = size(b.r, 1)
  nkeep = count(==(STATE_ALIVE), b.state)
  nlost = N - nkeep
  nlost == 0 && return 0
  b.r, b.w, b.id = device_compact(b.r, b.w, b.id, b.state, nkeep)
  b.state = fill!(similar(b.state, nkeep), STATE_ALIVE)
  return nlost
end

"Concatenate bins `bs` into one bin (used when bins are merged)."
function merge_bins(bs::AbstractVector{<:ParticleBin})
  b1 = first(bs)
  ParticleBin(b1.species, b1.mc2, b1.charge,
              reduce(vcat, (b.r for b in bs)), reduce(vcat, (b.w for b in bs)),
              reduce(vcat, (b.state for b in bs)), reduce(vcat, (b.id for b in bs)))
end

# ---------------------------------------------------------------------------------------
# Distribution over MPI ranks

"Keep only the rows of `b` that belong to this rank (block distribution)."
function scatter_bin(b::ParticleBin, comm::AbstractComm)
  nranks(comm) == 1 && return b
  N = nparticles(b)
  r0 = rank(comm)
  n = nranks(comm)
  lo = div(N * r0, n) + 1
  hi = div(N * (r0 + 1), n)
  ParticleBin(b.species, b.mc2, b.charge, b.r[lo:hi, :], b.w[lo:hi], b.state[lo:hi], b.id[lo:hi])
end

"Gather all particles of a distributed bin to every rank."
function gather_bin(b::ParticleBin, comm::AbstractComm)
  nranks(comm) == 1 && return b
  rh = Array(b.r); wh = Array(b.w); idh = Array(b.id)
  M = allgather_rows(comm, hcat(rh, wh, Float64.(idh)))
  ParticleBin(b.species, b.mc2, b.charge, M[:, 1:6], M[:, 7], fill(STATE_ALIVE, size(M, 1)),
              round.(Int, M[:, 8]))
end
