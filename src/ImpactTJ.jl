"""
    ImpactTJ

Time-domain 3D tracking of charged particle beams with space charge, wakefields and CSR,
implementing the physics of IMPACT-T (J. Qiang et al., PRST-AB 9, 044204 (2006)).

Lattices are described with Beamlines.jl, physical constants come from
AtomicAndPhysicalConstants.jl. Runs on CPU threads, with MPI (load `MPI`), and on GPUs
through KernelAbstractions.
"""
module ImpactTJ

using AtomicAndPhysicalConstants
using Beamlines
using KernelAbstractions
using Atomix
using StaticArrays
using Adapt
using LinearAlgebra
using Printf
using Random
using SpecialFunctions: besselj0, besselj1, bessely0, bessely1
import AbstractFFTs
import FFTW
import HDF5
import TOML

export ParticleBin, Settings, Simulation, run!, compile_lattice, CompiledLattice,
       BeamDistribution, PhasePlane, generate, make_bin,
       OnAxisFourier, OnAxisData, RZSolenoidMap, RZCavityMap, CartesianMap, PoleFaceGeometry,
       BaneWake, BTWWake, TeslaWake, Tesla3p9Wake, TabulatedWake, DielectricWake, WakeRegion,
       AlphaMagnet, MeanderWave, SurfaceRoughness, SteeringDipole,
       ParticleOutput, SliceOutput, Checkpoint, TimeStepChange, SpaceChargeSwitch, Steer,
       Collimator, LinearMap, Heat, RotateZ,
       BeamStats, emittance, rms, write_particles, read_particles, write_stats, read_stats,
       slice_analysis, write_checkpoint, read_checkpoint!, ext_field,
       read_impactt, write_impactt, read_impactt_stats, read_impactt_particles,
       write_impactt_particles, SerialComm, default_comm, nparticles, Event, ImpactParams,
       write_impacttj, save_fieldmaps, load_fieldmaps, ImpactTInput, autophase!, energy_gain

include("constants.jl")
include("comm.jl")
include("gpu.jl")
include("particles.jl")
include("fields/tables.jl")
include("fields/elements.jl")
include("fields/fieldset.jl")
include("fieldmaps.jl")
include("events.jl")
include("lattice.jl")
include("distributions.jl")
include("spacecharge/green.jl")
include("spacecharge/solver.jl")
include("push.jl")
include("diagnostics.jl")
include("wakes.jl")
include("dwa.jl")
include("simulation.jl")
include("bend.jl")
include("spacecharge/nbody.jl")
include("io.jl")
include("beam.jl")
include("autophase.jl")
include("translate/impactt_read.jl")
include("translate/impactt_write.jl")
include("translate/script_write.jl")

function __init__()
  _register_beamlines_params()
end

end
