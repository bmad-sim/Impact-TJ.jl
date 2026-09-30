# Particle push: leap-frog in time.
#
#   drift dt/2  →  (deposit, field solve)  →  kick dt  →  drift dt/2
#
# The kick is the Boris scheme: half electric kick, exact magnetic rotation, half electric
# kick. All kernels are written with KernelAbstractions and run on CPU threads or GPUs.

const _C = C_LIGHT_SI

@kernel function _drift_ka!(r, @Const(state), cdt, cathode::Bool)
  p = @index(Global)
  @inbounds if state[p] == STATE_ALIVE
    z = r[p, 5]
    if !(cathode && z <= 0)
      px = r[p, 2]; py = r[p, 4]; pz = r[p, 6]
      s = cdt / sqrt(1 + px * px + py * py + pz * pz)
      r[p, 1] += s * px
      r[p, 3] += s * py
      r[p, 5] = z + s * pz
    end
  end
end

"""
    drift!(bin, dt, cathode)

Advance positions by `dt` with the current momenta. If `cathode`, particles still behind
the cathode plane (z ≤ 0) are not moved (they are handled by the emission model).
"""
function drift!(b::ParticleBin, dt, cathode::Bool=false)
  N = size(b.r, 1)
  N == 0 && return
  backend = get_backend(b.r)
  _drift_ka!(backend)(b.r, b.state, _C * dt, cathode; ndrange=N)
  _sync(backend)
end

@kernel function _emission_ka!(r, @Const(state), cdt, βini)
  p = @index(Global)
  @inbounds if state[p] == STATE_ALIVE
    z = r[p, 5]
    if z <= 0 && r[p, 6] >= 0
      z += cdt * βini
      r[p, 5] = z
    end
    # first order emission: a particle that crossed the cathode during this step is
    # re-positioned as if emitted at the crossing time with its own momentum
    if z >= 0 && z - cdt * βini <= 0
      frac = z / βini               # c·(time since crossing)
      px = r[p, 2]; py = r[p, 4]; pz = r[p, 6]
      ig = 1 / sqrt(1 + px * px + py * py + pz * pz)
      r[p, 1] += frac * px * ig
      r[p, 3] += frac * py * ig
      r[p, 5] = frac * pz * ig
    end
  end
end

"""
    emission_step!(bin, dt, βini)

Emission model of IMPACT-T: particles behind the cathode (z ≤ 0) are moved towards it
with the velocity βini·c of the laser time-to-length conversion. Particles crossing z = 0
during the step are placed where they would be had they left the cathode with their own
momentum at the crossing time.
"""
function emission_step!(b::ParticleBin, dt, βini)
  N = size(b.r, 1)
  N == 0 && return
  backend = get_backend(b.r)
  _emission_ka!(backend)(b.r, b.state, _C * dt, βini; ndrange=N)
  _sync(backend)
end

"""
    boris(px, py, pz, E, B, qmcdt)

Boris push of the momentum γβ over a step dt, with `qmcdt = q c dt /(m c²)` [1/V·m].
"""
@inline function boris(px, py, pz, E::V3, B::V3, qmcdt)
  h = 0.5 * qmcdt
  umx = px + h * E[1]; umy = py + h * E[2]; umz = pz + h * E[3]
  γ = sqrt(1 + umx * umx + umy * umy + umz * umz)
  tt = h * _C / γ
  a1 = tt * B[1]; a2 = tt * B[2]; a3 = tt * B[3]
  a4 = 1 + a1 * a1 + a2 * a2 + a3 * a3
  s1 = umx + tt * (umy * B[3] - umz * B[2])
  s2 = umy - tt * (umx * B[3] - umz * B[1])
  s3 = umz + tt * (umx * B[2] - umy * B[1])
  upx = ((1 + a1 * a1) * s1 + (a1 * a2 + a3) * s2 + (a1 * a3 - a2) * s3) / a4
  upy = ((a1 * a2 - a3) * s1 + (1 + a2 * a2) * s2 + (a2 * a3 + a1) * s3) / a4
  upz = ((a1 * a3 + a2) * s1 + (a2 * a3 - a1) * s2 + (1 + a3 * a3) * s3) / a4
  return upx + h * E[1], upy + h * E[2], upz + h * E[3]
end

@kernel function _kick_ka!(r, @Const(state), qmcdt, t, models, ranges, @Const(buf),
                           @Const(F), g::GridGeom, use_sc::Bool, zpos_only::Bool, use_bz::Bool)
  p = @index(Global)
  @inbounds if state[p] == STATE_ALIVE
    x = r[p, 1]; y = r[p, 3]; z = r[p, 5]
    if !(zpos_only && z <= 0)
      E, B = ext_field(models, ranges, buf, x, y, z, t)
      if use_sc
        Es, Bs = gather_sc(F, g, x, y, z, use_bz)
        E += Es; B += Bs
      end
      px, py, pz = boris(r[p, 2], r[p, 4], r[p, 6], E, B, qmcdt)
      r[p, 2] = px; r[p, 4] = py; r[p, 6] = pz
    end
  end
end

"""
    kick!(bin, t, dt, fs, ranges, sc, use_sc, zpos_only)

Momentum update over `dt` at time `t` with external fields of `fs` (index `ranges`) plus,
if `use_sc`, the space-charge fields stored on the grid of `sc`. With `zpos_only`, only
particles in front of the cathode plane (z > 0) are kicked (IMPACT-T convention).
"""
function kick!(b::ParticleBin, t, dt, fs::FieldSet, ranges, sc::SpaceChargeSolver, use_sc::Bool,
               zpos_only::Bool, use_bz::Bool=false)
  N = size(b.r, 1)
  N == 0 && return
  backend = get_backend(b.r)
  qmcdt = qm(b) * _C * dt
  update_time!(fs, t)
  _kick_ka!(backend)(b.r, b.state, qmcdt, t, fs.models, ranges, fs.buf, sc.F, sc.geom, use_sc,
                     zpos_only, use_bz; ndrange=N)
  _sync(backend)
end
