# Actions triggered when the bunch centroid reaches a longitudinal position `z`.
#
# Events can be given explicitly (`Simulation(...; events=[...])`) or attached to Marker
# elements of the Beamline (`Marker(impact_event=ParticleOutput())`, whose `z` is then the
# marker position). As in IMPACT-T, events of the same type are executed in order of
# increasing z, at most one per time step, when the centroid distance d satisfies
# d ≤ z ≤ d + βcΔt.

abstract type Event end

event_z(e::Event) = e.z
with_z(e::Event, z) = (T = typeof(e); T((f == :z ? Float64(z) : getfield(e, f) for f in fieldnames(T))...))

"Write the particle distribution (every `every`-th particle) to `name` (openPMD HDF5)."
Base.@kwdef struct ParticleOutput <: Event
  z::Float64 = 0.0
  every::Int = 1
  name::String = ""
end

"Write slice diagnostics with `nslices` slices."
Base.@kwdef struct SliceOutput <: Event
  z::Float64 = 0.0
  nslices::Int = 64
  name::String = ""
end

"Save the complete simulation state for a restart."
Base.@kwdef struct Checkpoint <: Event
  z::Float64 = 0.0
  name::String = "checkpoint.h5"
end

"Change the time step to `dt` [s]."
Base.@kwdef struct TimeStepChange <: Event
  z::Float64 = 0.0
  dt::Float64
end

"Switch the space-charge solve on or off (wakefields are still applied)."
Base.@kwdef struct SpaceChargeSwitch <: Event
  z::Float64 = 0.0
  on::Bool
end

"Kick the whole beam by `dr = (dx, dγβx, dy, dγβy, dz, dγβz)`."
Base.@kwdef struct Steer <: Event
  z::Float64 = 0.0
  dr::NTuple{6,Float64}
end

"Remove particles outside [xmin,xmax]×[ymin,ymax] (`round = false`) or outside the circle of radius `xmax` (`round = true`)."
Base.@kwdef struct Collimator <: Event
  z::Float64 = 0.0
  xmin::Float64 = -Inf
  xmax::Float64 = Inf
  ymin::Float64 = -Inf
  ymax::Float64 = Inf
  round::Bool = false
end

"Apply the linear map `R` (6×6, IMPACT-T coordinates) around the bunch centroid."
Base.@kwdef struct LinearMap <: Event
  z::Float64 = 0.0
  R::Matrix{Float64}
end

"Add a random Gaussian γβz spread of rms `sigma` (laser heater)."
Base.@kwdef struct Heat <: Event
  z::Float64 = 0.0
  sigma::Float64
end

"Rotate the beam about the z axis by `angle` [rad]."
Base.@kwdef struct RotateZ <: Event
  z::Float64 = 0.0
  angle::Float64
end

"Executed before the particle push of a time step."
is_pre_event(e::Event) = e isa Union{Steer,LinearMap,Heat,RotateZ,TimeStepChange,SpaceChargeSwitch}
"Executed after the field calculation (collimators) or at the end of a time step."
is_post_event(e::Event) = e isa Union{ParticleOutput,SliceOutput,Checkpoint}

"Queue of events of one type (sorted by z) with the index of the next one to execute."
mutable struct EventQueue
  events::Vector{Event}
  next::Int
end

function event_queues(events)
  types = unique(typeof.(events))
  Dict(T => EventQueue(sort!(Event[e for e in events if e isa T], by=event_z), 1) for T in types)
end

"Return the next event of the queue if it is triggered at centroid distance `d` with step `dz`."
function trigger!(q::EventQueue, d, dz)
  q.next > length(q.events) && return nothing
  e = q.events[q.next]
  if d <= e.z && d + dz >= e.z
    q.next += 1
    return e
  end
  return nothing
end
