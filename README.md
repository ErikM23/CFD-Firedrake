# Firedrake
This project is part of the diploma thesis "Efficient scalable solvers for incompressible flow problems".

The emphasis is on efficient preconditioning (PCD and LSC) for incompressible Navier-Stokes system. Especially, PCD preconditioning for time-dependent flow described by incompressible Navies-Stokes equations was implemented in Firedrake. The validation was performed on 2 benchmarks: lid-driven cavity and flow around the cylinder. For comparing the performance, weak and strong scaling were used.

The performance of solver with PCD preconditioning for time-dependent flow in 3D was reasonably comparable with solver with LSC preconditioning. Here, LSC preconditioning implemented in PETSc was used. In 3D, both iterative solvers (FGMRES+PCD and FGMRES+LSC) were more efficient as MUMPS solver.


All computations were performed on the cluster Shěhurka - https://cluster.karlin.mff.cuni.cz/
