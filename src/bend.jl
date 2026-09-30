# Bend mode.
#
# When the head of the bunch reaches a bending magnet, the bunch is tracked in the
# Cartesian frame of the magnet entrance together with a reference particle that enters on
# axis with the design energy. Space charge (and optionally CSR) is computed each step in
# the frame rotated to the direction of the reference particle. When the reference
# particle has traveled the element length, all coordinates are transformed to the frame
# of the reference particle, which becomes the axis of the following beamline.

mutable struct RefParticle
  r::MVector{6,Float64}   # x, γβx, y, γβy, z, γβz
end

"Dipole whose region the bunch head has just entered (or nothing)."
function entering_dipole(sim::Simulation, zhead)
  for d in sim.lattice.dipoles
    d.index in sim.done_dipoles && continue          # already done
    if zhead >= d.s0 && sim.distance < d.s1
      return d
    end
  end
  return nothing
end

# rotation (x, z) → frame whose z axis is along (px, pz) of the reference particle
@inline _rot_cs(ref) = (n = hypot(ref[2], ref[6]); (ref[6] / n, ref[2] / n))

function _rotate!(b::ParticleBin, cs, ss)
  r = b.r
  x = r[:, 1]; px = r[:, 2]; z = r[:, 5]; pz = r[:, 6]
  r[:, 1] .= x .* cs .- z .* ss
  r[:, 2] .= px .* cs .- pz .* ss
  r[:, 5] .= x .* ss .+ z .* cs
  r[:, 6] .= px .* ss .+ pz .* cs
end
_rotate_back!(b::ParticleBin, cs, ss) = _rotate!(b, cs, -ss)

function _drift_ref!(ref::RefParticle, dt)
  r = ref.r
  ig = 1 / sqrt(1 + r[2]^2 + r[4]^2 + r[6]^2)
  r[1] += C_LIGHT_SI * dt * r[2] * ig
  r[3] += C_LIGHT_SI * dt * r[4] * ig
  r[5] += C_LIGHT_SI * dt * r[6] * ig
end

@kernel function _kick_bend_ka!(r, @Const(state), qmcdt, pf::PoleFaces, B0, dd, cs, ss, @Const(F),
                                g::GridGeom, use_sc::Bool)
  p = @index(Global)
  @inbounds if state[p] == STATE_ALIVE
    x = r[p, 1]; y = r[p, 3]; z = r[p, 5]
    # position in the magnet frame
    xm = cs * x + ss * z
    zm = -ss * x + cs * z
    bu, bv, bz = poleface_field(pf, B0, dd, xm, y, zm)
    E = ZERO3
    B = V3(bu * cs - bz * ss, bv, bu * ss + bz * cs)
    if use_sc
      Es, Bs = gather_sc(F, g, x, y, z)
      E += Es; B += Bs
    end
    px, py, pz = boris(r[p, 2], r[p, 4], r[p, 6], E, B, qmcdt)
    r[p, 2] = px; r[p, 4] = py; r[p, 6] = pz
  end
end

"""
    track_bend!(sim, dip)

Track the bunch through the bending magnet `dip` (IMPACT-T bend mode). On entry the bunch
is at the middle of a time step (after the first half drift).
"""
function track_bend!(sim::Simulation, dip::DipoleSpec)
  s = sim.settings
  comm = sim.comm
  push!(sim.done_dipoles, dip.index)
  geom = dip.geom
  pf = dip.field.faces
  B0 = dip.field.B0
  dd = dip.field.dd
  γin = geom.gamma_ref
  vref = sqrt(1 - 1 / γin^2)
  dt = sim.dt
  zorg = dip.s0; zorg2 = dip.s1
  b1 = sim.bins[1]
  Rbend = b1.mc2 / C_LIGHT_SI * sqrt(γin^2 - 1) / abs(B0)
  # enter the magnet frame
  refs = RefParticle[]
  ibb = length(sim.bins) > 1 ? div(length(sim.bins), 2) : 1
  zc = 0.0
  for (ib, b) in enumerate(sim.bins)
    sm = bin_summary(b, comm)
    b.r[:, 5] .-= zorg
    push!(refs, RefParticle(MVector(0.0, 0.0, 0.0, 0.0, sm.mean[5] - zorg, sqrt(γin^2 - 1))))
    ib == ibb && (zc = sm.mean[5] - zorg)
  end
  zz = zc - 0.5 * vref * dt * C_LIGHT_SI       # path length of the reference particle
  # back half a step
  for (ib, b) in enumerate(sim.bins)
    drift!(b, -dt / 2, false)
    _drift_ref!(refs[ib], -dt / 2)
  end
  sim.t -= dt / 2
  use_sc = sim.total_charge0 > 0 && s.space_charge
  while true
    sim.step += 1
    for (ib, b) in enumerate(sim.bins)
      drift!(b, dt / 2, false)
      _drift_ref!(refs[ib], dt / 2)
    end
    sim.t += dt / 2
    for (ib, b) in enumerate(sim.bins)
      ref = refs[ib]
      γref = sqrt(1 + ref.r[2]^2 + ref.r[6]^2)
      cs, ss = _rot_cs(ref.r)
      _rotate!(b, cs, ss)
      sm = bin_summary(b, comm)
      qmcdt = qm(b) * C_LIGHT_SI * dt
      if use_sc && sm.n > 0
        sc = sim.sc
        set_geometry!(sc, sm.lo, sm.hi, γref, false)
        reset_fields!(sc)
        deposit!(sc, b, false)
        allreduce_sum!(comm, sc.rho)
        solve_bin!(sc, ib, γref, 1.0, false)
        if geom.csr
          g = sc.geom
          λ, _, _ = line_moments(b, g.zmin, g.hz, g.nz, false, comm)
          zwkmin = (g.zmin - sm.mean[5]) + (zz - geom.s_start)
          ez = csr_wake(λ, zwkmin, g.hz, Rbend, γin, geom.s_end - geom.s_start)
          z0 = fill!(similar(ez), 0.0)
          add_slice_fields!(sc, z0, z0, ez)
        end
      end
      # reference particle: magnet field at its position
      bu, bv, bz = poleface_field(pf, B0, dd, ref.r[1], ref.r[3], ref.r[5])
      qref = qm(b) * C_LIGHT_SI * dt
      px, py, pz = boris(ref.r[2], ref.r[4], ref.r[6], ZERO3, V3(bu, bv, bz), qref)
      ref.r[2] = px; ref.r[4] = py; ref.r[6] = pz
      N = size(b.r, 1)
      if N > 0
        backend = get_backend(b.r)
        _kick_bend_ka!(backend)(b.r, b.state, qmcdt, pf, B0, dd, cs, ss, sim.sc.F, sim.sc.geom,
                                use_sc && sm.n > 0; ndrange=N)
        _sync(backend)
      end
      _rotate_back!(b, cs, ss)
    end
    for (ib, b) in enumerate(sim.bins)
      drift!(b, dt / 2, false)
      _drift_ref!(refs[ib], dt / 2)
    end
    sim.t += dt / 2
    rr = refs[ibb].r
    γ = sqrt(1 + rr[2]^2 + rr[6]^2)
    zz += hypot(rr[2], rr[6]) / γ * dt * C_LIGHT_SI
    if sim.step % s.diag_interval == 0
      record_stats!(sim.bend_stats, sim.t, sim.bins, comm; at_fixed_z=false)
      push!(sim.bend_ref, (sim.t, rr[1], rr[2], rr[3], rr[4], rr[5], rr[6]))
    end
    (zz > zorg2 - zorg || sim.step >= s.max_steps) && break
  end
  # exit: go to the frame of the reference particle
  for (ib, b) in enumerate(sim.bins)
    rr = refs[ib].r
    cs, ss = _rot_cs(rr)
    b.r[:, 1] .-= rr[1]; b.r[:, 3] .-= rr[3]; b.r[:, 5] .-= rr[5]
    _rotate!(b, cs, ss)
    b.r[:, 5] .+= zorg2
  end
  return nothing
end
