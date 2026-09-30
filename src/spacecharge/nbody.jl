# Direct (point-to-point) relativistic space charge.
#
# Every macroparticle feels the exact field of all other macroparticles moving with
# constant velocity (Lorentz-boosted Coulomb field), softened inside radius r0, plus their
# images in the cathode plane z = 0 when the image charge is active. O(N²): meant for
# small numbers of particles or GPUs.

@inline function _pp_field(rx, ry, rz, bx, by, bz, γ, q, r0)
  t1 = γ^2 * (rx * bx + ry * by + rz * bz) / (γ + 1)
  ux = rx + t1 * bx; uy = ry + t1 * by; uz = rz + t1 * bz     # rest frame separation
  u = sqrt(ux * ux + uy * uy + uz * uz)
  u == 0 && return ZERO3, ZERO3
  t2 = γ * (bx * ux + by * uy + bz * uz) / (γ + 1)
  d = u > r0 ? u^3 : r0^3
  c = q * γ / d
  E = V3(c * (ux - t2 * bx), c * (uy - t2 * by), c * (uz - t2 * bz))
  B = V3(c * (by * uz - bz * uy), c * (bz * ux - bx * uz), c * (bx * uy - by * ux)) / C_LIGHT_SI
  return E, B
end

@kernel function _nbody_ka!(r, @Const(state), @Const(src), nsrc::Int, qmcdt, t, models, ranges,
                            @Const(buf), r0, image::Bool)
  p = @index(Global)
  @inbounds if state[p] == STATE_ALIVE && r[p, 5] > 0
    x = r[p, 1]; y = r[p, 3]; z = r[p, 5]
    E, B = ext_field(models, ranges, buf, x, y, z, t)
    Es = ZERO3; Bs = ZERO3
    for j in 1:nsrc
      sx = src[j, 1]; sy = src[j, 2]; sz = src[j, 3]
      γ = src[j, 7]; q = src[j, 8]
      bx = src[j, 4] / γ; by = src[j, 5] / γ; bz = src[j, 6] / γ
      e, b = _pp_field(x - sx, y - sy, z - sz, bx, by, bz, γ, q, r0)
      Es += e; Bs += b
      if image
        e, b = _pp_field(x - sx, y - sy, z + sz, bx, by, -bz, γ, -q, r0)
        Es += e; Bs += b
      end
    end
    E += COULOMB_K * Es; B += COULOMB_K * Bs
    px, py, pz = boris(r[p, 2], r[p, 4], r[p, 6], E, B, qmcdt)
    r[p, 2] = px; r[p, 4] = py; r[p, 6] = pz
  end
end

@kernel function _nbody_src_ka!(src, @Const(r), @Const(w), @Const(state), off)
  p = @index(Global)
  @inbounds begin
    ok = state[p] == STATE_ALIVE && r[p, 5] > 0
    γ = sqrt(1 + r[p, 2]^2 + r[p, 4]^2 + r[p, 6]^2)
    q = off + p
    src[q, 1] = r[p, 1]; src[q, 2] = r[p, 3]; src[q, 3] = r[p, 5]
    src[q, 4] = r[p, 2]; src[q, 5] = r[p, 4]; src[q, 6] = r[p, 6]
    src[q, 7] = γ; src[q, 8] = ok ? w[p] : 0.0   # inactive particles have no charge
  end
end

"""
Source table of all particles (all bins, all ranks). Particles that are lost or behind
the cathode are kept with zero charge (no compaction needed; built on the device).
"""
function nbody_sources(sim::Simulation)
  ntot = sum(b -> size(b.r, 1), sim.bins; init=0)
  proto = first(sim.bins).r
  src = similar(proto, ntot, 8)
  off = 0
  for b in sim.bins
    N = size(b.r, 1)
    N == 0 && continue
    backend = get_backend(b.r)
    _nbody_src_ka!(backend)(src, b.r, b.w, b.state, off; ndrange=N)
    _sync(backend)
    off += N
  end
  nranks(sim.comm) == 1 && return src
  rows = allgather_rows(sim.comm, src)   # host matrix (staged through MPI)
  return src isa Array ? rows : copyto!(similar(src, size(rows)...), rows)
end

function kick_nbody!(sim::Simulation, b::ParticleBin, dt, ranges)
  N = size(b.r, 1)
  N == 0 && return
  src = nbody_sources(sim)
  backend = get_backend(b.r)
  update_time!(sim.fields, sim.t)
  _nbody_ka!(backend)(b.r, b.state, src, size(src, 1), qm(b) * C_LIGHT_SI * dt, sim.t,
                      sim.fields.models, ranges, sim.fields.buf, sim.settings.nbody_r0,
                      sim.image_on; ndrange=N)
  _sync(backend)
end
