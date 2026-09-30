# Running an existing IMPACT-T input deck and converting it between formats.
#   julia -t 4 --project translate_impactt.jl /path/to/IMPACT-T/examples/Sample1
using ImpactTJ

deck = get(ARGS, 1, joinpath(@__DIR__, "..", "test", "data", "sample1"))

# 1. Translate ImpactT.in (+ rfdata*, 1T*.T7, partcl.data, ...) and run it directly.
inp = read_impactt(deck)
sim = Simulation(inp)
sim.settings.output_dir = "sample_output"
run!(sim)
println("final kinetic energy: ", sim.stats.ekin[end] / 1e6, " MeV")

# 2. Write the deck in the ImpactTJ format: a Julia script using Beamlines, field maps in
#    HDF5 and the initial particles in openPMD format. Run it with `julia new_format/run.jl`.
write_impacttj("new_format", inp)

# 3. Write an IMPACT-T deck again (e.g. to compare with the Fortran code). The particles are
#    written explicitly so both codes start from identical macroparticles.
write_impactt("impactt_format", inp)

# 4. Read IMPACT-T output files (fort.18, fort.24, ...) for comparisons.
# ref = read_impactt_stats("impactt_format")
