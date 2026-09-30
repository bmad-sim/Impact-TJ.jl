# ImpactTJ.jl

Time-domain 3D tracking of charged particle beams with space charge, image charges,
wakefields and CSR, implementing the physics of IMPACT-T
([J. Qiang et al., PRST-AB 9, 044204 (2006)](https://github.com/impact-lbl/IMPACT-T)) in
Julia and integrated with SciBmad: lattices are
[Beamlines.jl](https://github.com/bmad-sim/Beamlines.jl) `Beamline`s, constants come from
AtomicAndPhysicalConstants.jl. Runs on CPU threads, with MPI (`using MPI`) and on GPUs
(KernelAbstractions; pass `device=CUDA.zeros(1)` to `Simulation`, Float64 required).

## Quick start

```julia
using ImpactTJ
inp = read_impactt("path/to/ImpactT.in")   # translate an IMPACT-T deck
sim = Simulation(inp)
run!(sim)                                  # stats in sim.stats, files in settings.output_dir
write_impacttj("new_format", inp)          # IMPACT-T → ImpactTJ (Julia script + HDF5)
write_impactt("old_format", inp)           # ImpactTJ → IMPACT-T
```

See `examples/photoinjector.jl` for a lattice written directly in the native format
(Beamlines elements with `field_map`, `field_scale`, … properties from `ImpactParams`,
`Settings`, events such as `ParticleOutput`, and `autophase!` for RF phasing).

## Formats

- Input: Julia script (Beamlines lattice + `Settings` + events), field maps in HDF5
  (`save_fieldmaps`/`load_fieldmaps`), particles in openPMD-beamphysics HDF5.
- Output: `stats.h5` (moments, full 6×6 second-moment matrix, emittances, …), particles
  in openPMD HDF5 (`read_particles`), slice diagnostics, checkpoints.
- Translators: `read_impactt`/`write_impactt` (input decks, all field file formats,
  particle files), `read_impactt_stats` (fort.18/24–28).

## Validation

Compared with the Fortran IMPACT-T on all six IMPACT-T examples plus chicane (bend mode +
CSR), multipole/misalignment/event, and dielectric-wake decks. Starting from identical
particles the results agree to 5–6 significant digits (e.g. Sample1 photoinjector, full
2.5 m); see `test/`. MPI (1, 2, 4 ranks) reproduces serial results exactly.

## Performance (Apple M1, vs serial Fortran)

4 threads: Sample1 21.6 s vs 162 s; Sample2 19 s vs 205 s; Sample3 12 s vs 118 s; Sample5
9.4 s vs 53 s; Sample6 13 s vs 98 s; chicane 20 s vs 100 s. Single thread: Sample1 31 s,
Sample2 35 s, Sample6 17 s. Algorithmic improvements: tabulated on-axis fields (O(1)
instead of Fourier sums per particle), far-field expansion of the integrated Green
function, DCT for the symmetric Green function, pruned FFTs, RF phases evaluated once per
step, optional Green function reuse (`green_tol`), replicated-grid MPI (no load
balancing needed).

## GPU

With `device=CUDA.zeros(1)` (or any KernelAbstractions backend array) particles, grids
and field models stay on the device for the whole run: push, deposition (3D and
azimuthal), Poisson solve, Green function (cached on the device with `green_tol > 0`),
losses and compaction, beam moments, structure wakes, CSR, dielectric wakes, N-body
space charge and the beam manipulation events all run in kernels. Only reduced numbers
(bunch extent, centroid, moments, particle counts) go back to the host, plus the output
files. With MPI, device buffers are passed to MPI directly when it is GPU aware
(`MPI.has_cuda()`/`has_rocm()`), otherwise they are staged through host memory.

## Differences from IMPACT-T

- CSR: piecewise-linear line density with graded quadrature; IMPACT-T assigns the
  slippage point to the wrong cell, giving O(h) errors at bend entrances (~10–20 % in
  energy loss for typical grids).
- Apertures are checked per particle (`aperture_mode=:bunch` reproduces IMPACT-T, minus
  its fallback to element 1 when the bunch sits in a gap).
- Field maps return zero outside their radius instead of stopping the run; Tesla
  transverse wake uses physical units; dielectric wake uses exact Bessel derivatives.
- Not implemented: Parmela/Elegant particle readers (read as IMPACT-T text format),
  on-axis field printing (-18), integrator switching.

## Status / TODO

The GPU path is verified with the generic JLArrays backend (agreement with the CPU to
1e-7 or better over full runs, differences come from the order of atomic additions) but
not yet on real CUDA hardware. The field solve is replicated on each MPI rank (a distributed
FFT, e.g. PencilFFTs, would help very large grids).
