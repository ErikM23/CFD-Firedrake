from firedrake import*
from firedrake.petsc import PETSc

print = PETSc.Sys.Print

N = 128

mesh = UnitSquareMesh(N, N)

print(mesh.num_cells(), mesh.num_vertices())

V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = V*Q

u, p = TrialFunctions(W)
v, q = TestFunctions(W)

x, y = SpatialCoordinate(mesh)
nu = Constant(1)
f = Constant((0,0))
       
a = nu*inner(grad(u),grad(v))*dx - p*div(v)*dx - div(u)*q*dx 

L = inner(f,v)*dx

V = Constant(1)

bcs = [DirichletBC(W.sub(0), Constant((V, 0)), (4,)), DirichletBC(W.sub(0), Constant((0, 0)), (1, 2, 3))]
       
nullspace = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True)])

appctx = {"nu" : nu, "velocity_space":0}

class MassMatrix(AuxiliaryOperatorPC):
    _prefix = "mass_"
    def form(self, pc, test, trial):
        nu = self.get_appctx(pc)["nu"]
        return (-1/nu *inner(test,trial)*dx, None)


MINRES = {
        "ksp_monitor":None,
        "ksp_type":"minres",
        "ksp_max_it":100000,
        "pc_type":"none",
        }

MUMPS = {
          "mat_type":"aij",
          "ksp_monitor":None,
          "ksp_type":"preonly",
          "pc_type":"lu",
          "pc_factor_mat_solver_type":"mumps",

        }

MINRES_MASS = { 
       "ksp_type": "minres",
       "ksp_monitor":None,
       "pc_type": "fieldsplit",
       "pc_fieldsplit_type":"schur",
       "pc_fieldsplit_schur_fact_type":"diag",
       "fieldsplit_0":{
           "ksp_type":"preonly",
           "pc_type":"hypre",
           },
       "fieldsplit_1":{
           "ksp_type":"preonly",
           "pc_type":"python",
           "pc_python_type":"__main__.MassMatrix",
           "mass_pc_type":"bjacobi",
           "mass_sub_pc_type":"ilu",
           }
       }


def convergence(solver):
        from firedrake.solving_utils import KSPReasons, SNESReasons
        snes=solver.snes
        print("""
KSP iterations: {ksp}; KSP converged reason: {kspreason}""".format(ksp=snes.ksp.getIterationNumber(),kspreason=KSPReasons[snes.ksp.getConvergedReason()]))



w = Function(W)
w.assign(0)

print("Creating solver ...")

problem = LinearVariationalProblem(a, L, w, bcs)
solver = LinearVariationalSolver(problem, solver_parameters=parDIR, appctx=appctx)

print("Solving ...")
solver.solve()

#outfile = File("STOKES_2D.pvd")
#u, p = w.split()
#u.rename("Velocity","Velocity")
#p.rename("Pressure","Pressure")
#outfile.write(u,p)

#convergence(solver)

