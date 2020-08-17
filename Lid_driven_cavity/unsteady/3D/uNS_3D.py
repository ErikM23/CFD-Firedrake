from firedrake import *
from FPCD import*

print = PETSc.Sys.Print

N = 16

M = UnitCubeMesh(N, N, N)

V = VectorFunctionSpace(M, "CG", 2)
Q = FunctionSpace(M, "CG", 1)
Z = V * Q

w = Function(Z)
w0 = Function(Z)
u, p = split(w)
u0, p0 = split(w0)
v, q = TestFunctions(Z)

nu = Constant(0.001)
Re = Constant(1000)


dt = 0.01
t_end = 3
theta = Constant(0.5)


def F_(u,p,v,q) :
            return  ( nu * inner(grad(u),grad(v))*dx + inner(dot(grad(u),u),v)*dx - p*div(v)*dx + q*div(u)*dx)

F = Constant(1.0/dt)*inner((u-u0),v)*dx + theta* F_(u,p,v,q) +(1-theta)*F_(u0,p0,v,q)

V = Constant(1)

bcs = [DirichletBC(Z.sub(0), Constant((V, 0, 0)), 4), DirichletBC(Z.sub(0), Constant((0, 0, 0)), (1, 2, 3, 5, 6))]

nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

appctx = {"nu": nu, "velocity_space": 0}

LSC = {
        "ksp_type": "fgmres",
        "ksp_monitor":None,
        "ksp_gmres_restart":400,
        "ksp_rtol":1e-3,
        "ksp_max_it":400,
        "snes_monitor":None,
        "snes_converged_reason":None,
        "ksp_gmres_modifiedgramschmidt":True,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type":"schur",
        "pc_fieldsplit_schur_fact_type":"upper",
        "fieldsplit_0":{
            "ksp_type":"gmres",
            "ksp_gmres_restart":50,
            "pc_type":"bjacobi",
                        },
        "fieldsplit_1":{
            "ksp_type":"preonly",
            "pc_type":"lsc",
            "lsc_pc_type":"ksp",
            "lsc_pc_ksp_type":"gmres",
            "lsc_pc_ksp_rtol":1e-2,
            "lsc_pc_ksp_max_it":50,
            "lsc_pc_ksp_gmres_restart":50,
            "lsc_pc_pc_type":"hypre",
            "lsc_pc_pc_hypre_type":"boomeramg",  
            "lsc_pc_pc_hypre_boomeramg_strong_threshold": 0.2,       
            "lsc_pc_pc_hypre_boomeramg_P_max":2,
                        }
        }


MUMPS = {
        "snes_monitor": None,
        "snes_max_it":1000,
        "ksp_type": "preonly",
        "mat_type": "aij",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"
            }

PCD = {
        "mat_type": "matfree",
        "snes_monitor": None,
        "ksp_monitor":None,
        "ksp_type": "fgmres",
        "ksp_rtol": 1e-3,
        "ksp_gmres_restart": 600,
        "ksp_max_it":800,
        "ksp_gmres_modifiedgramschmidt": None,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "upper",

        "fieldsplit_0_ksp_type": "gmres",
        "fieldsplit_0_ksp_gmres_restart": 100,
        "fieldsplit_0_ksp_max_it":10,
        "fieldsplit_0_pc_type": "python",
        "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
        "fieldsplit_0_assembled_pc_type": "bjacobi",

        "fieldsplit_1_ksp_type": "gmres",
        "fieldsplit_1_ksp_rtol": 1e-1,
        "fieldsplit_1_ksp_max_it": 10,
        "fieldsplit_1_pc_type": "python",
        "fieldsplit_1_pc_python_type": "FPCD.PCD",

        "fieldsplit_1_pcd_Mp_ksp_type": "richardson",
        "fieldsplit_1_pcd_Mp_ksp_max_it": 5,
        "fieldsplit_1_pcd_Mp_pc_type": "sor",


        "fieldsplit_1_pcd_Kp_ksp_type": "preonly",
        "fieldsplit_1_pcd_Kp_pc_type": "hypre",
        "fieldsplit_1_pcd_Kp_pc_hypre_type":"boomeramg",
        "fieldsplit_1_pcd_Kp_pc_hypre_boomeramg_strong_threshold": 0.2,
        "fieldsplit_1_pcd_Kp_pc_hypre_boomeramg_P_max":2,

        "fieldsplit_1_pcd_Fp_mat_type": "matfree"
        }


def convergence(solver):
    from firedrake.solving_utils import KSPReasons, SNESReasons
    snes = solver.snes
    print("""
            SNES iterations: {snes}; SNES converged reason: {snesreason}
            KSP iterations: {ksp}; KSP converged reason: {kspreason}""".format(snes=snes.getIterationNumber(),snesreason=SNESReasons[snes.getConvergedReason()],
                                                                               ksp=snes.ksp.getIterationNumber(), kspreason=KSPReasons[snes.ksp.getConvergedReason()]))



def create_solver(solver_parameters, appctx):
    problem = NonlinearVariationalProblem(F, w, bcs=bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters, nullspace=nullspace, appctx=appctx)
    return solver


solver = create_solver(solver_parameters=LSC, appctx=appctx)

#outfile = File("NS3D_cavity_time.pvd")

t = 0.0

w.assign(0)
w0.assign(w)
#u0, p0 = w0.split()
#u0.rename("Velocity", "Velocity")
#p0.rename("Pressure", "Pressure")

print("Solving ...")

while t<t_end:
    print("t= ",t)
    solver.solve()
    w0.assign(w)
    u0, p0 = w0.split()
    outfile.write(u0, p0, time=t)
    t += dt



#solve(F == 0, up, bcs=bcs, nullspace=nullspace, solver_parameters=parPCD, appctx=appctx)

#u, p = up.split()
#u.rename("Velocity")
#p.rename("Pressure")

#File("cavity.pvd").write(u, p)
#print(w.function_space().mesh().num_cells())
#convergence(solver)

