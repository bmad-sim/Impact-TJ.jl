# Automatic RF phasing (replaces IMPACT-T's external PhaseOpt.py script).

"""
    energy_gain(line, settings, species; z_end, ekin0=settings.emission_ekin, z0=0)

Track a single on-axis particle (no space charge) starting at `z0` with kinetic energy
`ekin0` [eV] through `line` until its z exceeds `z_end`; return its final kinetic energy.
"""
function energy_gain(line::Beamline, settings::Settings, species::Species; z_end,
                     ekin0=settings.emission_ekin, z0=0.0, mc2=massof(species))
  γ0 = 1 + ekin0 / mc2
  r = zeros(1, 6)
  r[1, 5] = max(z0, 1e-12)
  r[1, 6] = sqrt(γ0^2 - 1)
  b = ParticleBin(species, r, 0.0; mc2=mc2)
  s = deepcopy(settings)
  s.space_charge = false; s.image_charge = false; s.emission_steps = 0
  s.write_output = false; s.verbose = false; s.z_stop = z_end; s.diag_interval = typemax(Int)
  s.dt = settings.dt
  cl = compile_lattice(line; s_offset=s.s_offset)
  sim = Simulation(cl, [b], s; comm=SerialComm())
  run!(sim)
  isempty(sim.bins) || nparticles(sim.bins[1]) == 0 && return -Inf
  p = sim.bins[1].r
  size(p, 1) == 0 && return -Inf
  return (sqrt(1 + p[1, 2]^2 + p[1, 4]^2 + p[1, 6]^2) - 1) * mc2
end

"""
    autophase!(line, ele, settings, species; offset=0.0, npts=36, ekin0, z0=0)

Set the phase `phi0` of the RF element `ele` of `line` to its crest (maximum energy gain
of an on-axis particle at the element end, tracked from the start of the line) plus
`offset` [rad]. Upstream cavities should be phased first. Returns the crest phase.
"""
function autophase!(line::Beamline, ele, settings::Settings, species::Species; offset=0.0, npts=36,
                    ekin0=settings.emission_ekin, z0=0.0)
  child = only(findchildren(ele, line))   # the element instance in the beamline
  ip = _impact(ele)
  L = isnan(ip.body_length) ? Float64(ele.L) : Float64(ip.body_length)
  z_end = Float64(child.s) + settings.s_offset + L
  f(φ) = (ele.phi0 = φ; energy_gain(line, settings, species; z_end=z_end, ekin0=ekin0, z0=z0))
  φs = range(0, 2π; length=npts + 1)[1:end-1]
  E = f.(φs)
  k = argmax(E)
  # golden section refinement around the best scan point
  a = φs[k] - 2π / npts; b = φs[k] + 2π / npts
  g = (sqrt(5) - 1) / 2
  c = b - g * (b - a); d = a + g * (b - a)
  fc = f(c); fd = f(d)
  for _ in 1:40
    if fc > fd
      b = d; d = c; fd = fc; c = b - g * (b - a); fc = f(c)
    else
      a = c; c = d; fc = fd; d = a + g * (b - a); fd = f(d)
    end
    b - a < 1e-6 && break
  end
  crest = mod2pi((a + b) / 2)
  ele.phi0 = crest + offset
  return crest
end
