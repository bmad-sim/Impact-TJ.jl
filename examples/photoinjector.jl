# A simple S-band photoinjector written directly in the ImpactTJ format:
# 1.5 cell RF gun + solenoid + drift, cathode emission with image charges and 3D space
# charge. Run with e.g.  julia -t 4 --project photoinjector.jl
using ImpactTJ, Beamlines, Random

f_rf = 2.856e9
λ = ImpactTJ.C_LIGHT_SI / f_rf

# On-axis Ez of a 1.5 cell standing wave gun (normalized to 1, scaled by `field_scale`).
Lgun = 0.75λ
zs = range(0, Lgun; length=401)
ez = [cos(2π * z / λ) for z in zs]
gunmap = OnAxisData(first(zs), step(zs), ez; interpolation=:hermite)

@elements begin
  gun = RFCavity(L=Lgun, rf_frequency=f_rf, phi0=0.0, field_map=gunmap,
                 field_scale=100e6, x1_limit=-0.01, x2_limit=0.01, y1_limit=-0.01, y2_limit=0.01)
  d1 = Drift(L=0.1)
  # Solenoid without a field map: smooth hard-edge model (field extends beyond the body)
  sol = Solenoid(L=0.2, Bsol=0.065)          # focal length ≈ 1 m at 4 MeV
  d2 = Drift(L=1.0, x1_limit=-0.02, x2_limit=0.02, y1_limit=-0.02, y2_limit=0.02)
  screen = Marker(impact_event=ParticleOutput(name="screen.h5"))
  d3 = Drift(L=0.5)
end
line = Beamline([gun, d1, sol, d2, screen, d3]; species_ref=Species("electron"))

# 250 pC, 10 ps flat top laser (1 ps rise), 1 mm radius, thermal emittance from 0.5 eV
electron = Species("electron")
βini = sqrt(1 - 1 / (1 + 0.5 / 510998.95)^2)
Lb = 10e-12 * βini * ImpactTJ.C_LIGHT_SI
dist = BeamDistribution((:combine, 1, 1, 2);
  x=PhasePlane(sigma=1e-3, sigma_p=1.0e-3), y=PhasePlane(sigma=1e-3, sigma_p=1.0e-3),
  z=PhasePlane(sigma=Lb, scale=0.1Lb, sigma_p=1.0e-3))
beam = make_bin(dist, 20_000, electron; charge=250e-12, rng=Xoshiro(1))

settings = Settings(dt=1e-12, emission_steps=60, emission_time=12e-12, emission_ekin=0.5,
                    image_charge=true, image_cutoff=0.02, grid=(32, 32, 32),
                    domain_radius=0.1, sc_3d_start=0.0, output_dir="photoinjector_output")

# Phase the gun 20° off crest (replaces IMPACT-T's PhaseOpt.py)
crest = autophase!(line, gun, settings, electron; offset=deg2rad(-20))
println("gun crest phase = ", rad2deg(crest), " deg")

sim = Simulation(line, [beam], settings)
run!(sim)

st = sim.stats
println("final: <z> = ", st.z_mean[end], " m, E = ", st.ekin[end] / 1e6, " MeV, ",
        "εx = ", emittance(st, 1)[end] * 1e6, " µm, σz = ", rms(st, 5)[end] * 1e3, " mm")
