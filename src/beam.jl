# Creating particle bins from distributions.

"""
    make_bin(dist, N, species; charge, mc2=massof(species), rng=Xoshiro(1), ib=1, nb=1)

Sample `N` macroparticles from `dist` and return a `ParticleBin` carrying the bunch
charge `charge` [C] (magnitude; the sign is that of the species). For flat-top
photoinjector distributions split into `nb` energy bins, bin `ib` keeps only its slice;
each macroparticle then carries `charge/N`.
"""
function make_bin(dist::BeamDistribution, N::Integer, species::Species; charge::Real,
                  mc2=massof(species), qe=chargeof(species), rng=Random.Xoshiro(1), ib=1, nb=1)
  if dist.kind == :read
    bins, _ = load_particle_file(dist.file, species; mc2=mc2)
    b = only(bins)
    for (c, pl) in ((1, dist.x), (3, dist.y), (5, dist.z))
      b.r[:, c] .+= pl.offset
      b.r[:, c+1] .+= pl.p_offset
    end
    return b
  end
  r, _ = generate(dist, N, rng; mc2=mc2, ib=ib, nb=nb)
  q = abs(charge) / N * sign(qe)
  return ParticleBin(species, r, q; mc2=mc2, charge=qe)
end

"""
    load_particle_file(file, species; mc2)

Load particles from an openPMD HDF5 file or one of the IMPACT-T text formats.
"""
function load_particle_file(file::AbstractString, species::Species; mc2=massof(species))
  if endswith(lowercase(file), ".h5") || endswith(lowercase(file), ".hdf5")
    bins, _ = read_particles(file)
    return bins, nothing
  end
  return [read_impactt_particles(file, species; mc2=mc2)], nothing
end
