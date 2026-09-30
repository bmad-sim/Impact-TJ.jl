# Run with 2 MPI ranks (from runtests.jl): the distributed result must equal the serial one.
using MPI
MPI.Init()
using ImpactTJ

comm = ImpactTJ.default_comm()
inp = read_impactt(joinpath(@__DIR__, "data", "sample1"))
settings = inp.settings
settings.write_output = false; settings.verbose = false; settings.max_steps = 200

par = Simulation(inp; comm=comm)
run!(par)
ser = Simulation(read_impactt(joinpath(@__DIR__, "data", "sample1")); comm=SerialComm())
ser.settings.write_output = false; ser.settings.verbose = false; ser.settings.max_steps = 200
run!(ser)

ok = ImpactTJ.nranks(comm) == 2 &&
     isapprox(par.stats.ekin, ser.stats.ekin; rtol=1e-10) &&
     isapprox(ImpactTJ.rms(par.stats, 1), ImpactTJ.rms(ser.stats, 1); rtol=1e-8) &&
     par.stats.n_particle == ser.stats.n_particle
if ImpactTJ.isroot(comm)
  println(ok ? "MPI test passed" : "MPI test FAILED")
end
MPI.Finalize()
