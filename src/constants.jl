# Physical constants in SI units. All values come from AtomicAndPhysicalConstants.jl
# (which stores ε0 and μ0 in eV-based units, so they are converted here).

const C_LIGHT_SI  = Float64(C_LIGHT)                 # m/s
const E_CHARGE_SI = Float64(E_CHARGE)                # C
const EPS0_SI     = Float64(EPS_0 * E_CHARGE)        # F/m
const MU0_SI      = 1 / (EPS0_SI * C_LIGHT_SI^2)     # H/m
const Z0_SI       = MU0_SI * C_LIGHT_SI              # Ω, impedance of free space
const COULOMB_K   = 1 / (4π * EPS0_SI)               # m/F

# Particle state flags (same values as BeamTracking.jl).
const STATE_ALIVE = UInt8(1)
const STATE_LOST  = UInt8(2)

"Wait for the kernels of a KernelAbstractions backend to finish."
@inline _sync(backend) = applicable(KernelAbstractions.synchronize, backend) ? KernelAbstractions.synchronize(backend) : nothing
