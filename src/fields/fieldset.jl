# Collection of all external field models of a lattice.
#
# Field models are grouped by concrete type into a tuple of vectors so that the total
# field is evaluated with a fully type-stable, unrolled loop (works inside GPU kernels).
# Each vector is sorted by `zmin`; before every time step the index range of elements
# that can overlap the bunch is computed on the host so that each particle only tests a
# few candidate elements.

struct FieldSet{T<:Tuple,B<:AbstractVector{Float64}}
  models::T
  buf::B
end

"""
    FieldSet(models::AbstractVector, buf::Vector{Float64})

Group the field models by type (dropping types with no instance) and sort each group by
`zmin`.
"""
function FieldSet(models::AbstractVector, buf::Vector{Float64})
  types = unique(typeof.(models))
  groups = Tuple(sort!(Vector{T}(filter(m -> m isa T, models)), by=m -> m.pl.zmin) for T in types)
  return FieldSet(groups, buf)
end

Adapt.adapt_structure(to, fs::FieldSet) = FieldSet(map(v -> Adapt.adapt(to, v), fs.models), Adapt.adapt(to, fs.buf))

"Move the field set to the array type of `proto`."
function to_device(fs::FieldSet, proto::AbstractArray)
  A = Base.typename(typeof(proto)).wrapper
  FieldSet(map(v -> A(v), fs.models), A(fs.buf))
end

nmodels(fs::FieldSet) = sum(length, fs.models; init=0)

"""
    active_ranges(fs, zlo, zhi) -> NTuple of (lo, hi)

Index ranges (per model group) of the elements whose field extent may overlap [zlo, zhi].
"""
function active_ranges(fs::FieldSet, zlo, zhi)
  return map(fs.models) do v
    vh = v isa Vector ? v : Array(v)   # device vectors: pass the host field set instead
    hi = searchsortedlast(vh, zhi; by=m -> m isa Number ? m : m.pl.zmin)
    lo = 1
    while lo <= hi && vh[lo].pl.zmax < zlo
      lo += 1
    end
    (lo, hi)
  end
end

@kernel function _set_time_ka!(v, t)
  i = @index(Global)
  @inbounds v[i] = set_time(v[i], t)
end

"All-elements ranges."
all_ranges(fs::FieldSet) = map(v -> (1, length(v)), fs.models)

@inline function _group_field(v, rng, buf, x, y, z, t)
  E = ZERO3; B = ZERO3
  for i in rng[1]:rng[2]
    @inbounds m = v[i]
    if in_extent(m.pl, z)
      Ei, Bi = field(m, buf, x, y, z, t)
      E += Ei; B += Bi
    end
  end
  return E, B
end

"""
    ext_field(models, ranges, buf, x, y, z, t) -> (E, B)

Total external field at a lab point from the field models in the given index ranges.
"""
@inline function ext_field(models::Tuple, ranges::Tuple, buf, x, y, z, t)
  E1, B1 = _group_field(first(models), first(ranges), buf, x, y, z, t)
  E2, B2 = ext_field(Base.tail(models), Base.tail(ranges), buf, x, y, z, t)
  return E1 + E2, B1 + B2
end
@inline ext_field(::Tuple{}, ::Tuple{}, buf, x, y, z, t) = (ZERO3, ZERO3)

"""
    update_time!(fs, t)

Evaluate the time dependence (RF phases) of all field models at time `t`. This is done
once per kick instead of once per particle.
"""
function update_time!(fs::FieldSet, t)
  for v in fs.models
    isempty(v) && continue
    time_dependent(eltype(v)) || continue
    if v isa Vector
      @inbounds for i in eachindex(v)
        v[i] = set_time(v[i], t)
      end
    else
      backend = get_backend(v)
      _set_time_ka!(backend)(v, t; ndrange=length(v))
      _sync(backend)
    end
  end
  return fs
end

"Evaluate the total external field at one point (host convenience function)."
function ext_field(fs::FieldSet, x, y, z, t)
  update_time!(fs, t)
  ext_field(fs.models, all_ranges(fs), fs.buf, x, y, z, t)
end
