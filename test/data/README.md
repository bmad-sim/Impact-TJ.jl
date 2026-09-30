# Test data

Small IMPACT-T input decks used by the test suite and reference output of the Fortran
IMPACT-T code (<https://github.com/impact-lbl/IMPACT-T>) run on them (`reference/fort.*`).
All decks start from explicit particle files (`partcl.data`, distribution type 166) so
that ImpactTJ and IMPACT-T track identical macroparticles.

- `sample1`: IMPACT-T example Sample1 (photoinjector) reduced to 2000 particles and 600
  steps. `ImpactT.in` and `rfdata1-3` are from the IMPACT-T distribution.
- `sample4`: IMPACT-T example Sample4 (SUPERFISH cavity map and solenoid map); used for
  translation tests only.
- `features`: quadrupoles with fringe fields, misalignments and skew angle, multipoles,
  RF quadrupole, constant focusing element, and the steering, rotation, linear map,
  time step change and collimation events.
- `chicane`: four dipole chicane with space charge and CSR (bend mode).
- `dwa_slab`: dielectric lined slab waveguide wakefield.

The files taken from IMPACT-T are covered by the IMPACT-T license:

IMPACT-T, Copyright (c) 2016, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from
the U.S. Dept. of Energy). All rights reserved. Redistribution and use in source and
binary forms, with or without modification, are permitted provided that the conditions of
the IMPACT-T license (BSD) are met.
