from firedrake import *
from FPCD import*
from PCD import*

print = PETSc.Sys.Print

N = 16

M = UnitCubeMesh(N, N, N)

V = VectorFunctionSpace(M, "CG", 2)
W = FunctionSpace(M, "CG", 1)
Z = V * W

up = Function(Z)
u, p = split(up)
v, q = TestFunctions(Z)

nu = Constant(0.01)
Re = Constant(100)
F = (nu * inner(grad(u), grad(v)) * dx + inner(dot(grad(u), u), v) * dx - p * div(v) * dx - div(u) * q * dx )

bcs = [DirichletBC(Z.sub(0), Constant((1, 0, 0)), 4), DirichletBC(Z.sub(0), Constant((0, 0, 0)), (1, 2, 3, 5, 6))]

nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

#appctx = {"Re": Re, "velocity_space": 0}
appctx = {"nu": nu, "velocity_space": 0}

parLSC = {
        "ksp_type": "fgmres",
        "ksp_monitor":None,
        #"ksp_max_it":100000,
        "ksp_gmres_restart":600,#200#600#500
        #"ksp_rtol":1e-6,
        "snes_monitor":None,
        "snes_converged_reason":None,
        "ksp_gmres_modifiedgramschmidt":True,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type":"schur",
        "pc_fieldsplit_schur_fact_type":"upper",
        "fieldsplit_0":{
            "ksp_type":"preonly",
            #"ksp_max_it":2,
            #"ksp_gmres_restart":20, #50
            #"ksp_monitor":None,
            "pc_type":"hypre",
            #"pc_hypre_boomeramg_max_it":3,
                        },
        "fieldsplit_1":{
            "ksp_type":"preonly",#gmres
            "pc_type":"lsc",
            #"pc_lsc_scale_diag":None,
            "lsc_pc_type":"lu",
            "lsc_pc_factor_mat_solver_type":"mumps",
            #"lsc_pc_sub_type":"ilu",
            #"lsc_pc_ksp_type":"richardson",#gmres
            #"lsc_pc_ksp_max_it":2,
            #"lsc_pc_ksp_gmres_restart":50,
            #"lsc_pc_pc_type":"hypre",#bjacobi
            #"lsc_pc_pc_hypre_type":"boomeramg",  
            #"lsc_pc_pc_hypre_boomeramg_strong_threshold": 0.2,       
            #"lsc_pc_pc_hypre_boomeramg_P_max":2,
            #"lsc_pc_ksp_rtol":1e-4,
                        }
        }


parDIR={"snes_monitor": None,
        "snes_max_it":1000,
        "ksp_type": "preonly",
        "mat_type": "aij",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"}

parPCD = {
        "mat_type": "matfree",
        "snes_monitor": None,
        "ksp_monitor":None,
        "ksp_type": "fgmres",
        #"ksp_rtol": 1e-9,
        "ksp_gmres_restart": 600,
        "ksp_gmres_modifiedgramschmidt": None,
        #"ksp_monitor_true_residual": None,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "upper",

        "fieldsplit_0_ksp_type": "preonly",
        #"fieldsplit_0_ksp_gmres_restart": 50,
        "fieldsplit_0_pc_type": "python",
        "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
        "fieldsplit_0_assembled_pc_type": "hypre",
        #"fieldsplit_0_assembled_pc_sub_type": "ilu",

        "fieldsplit_1_ksp_type": "preonly",
        #"fieldsplit_1_ksp_rtol": 1e-4,
        "fieldsplit_1_pc_type": "python",
        "fieldsplit_1_pc_python_type": "FPCD.PCD",

        "fieldsplit_1_pcd_Mp_ksp_type": "richardson",
        "fieldsplit_1_pcd_Mp_ksp_max_it": 5,
        "fieldsplit_1_pcd_Mp_pc_type": "sor",

        "fieldsplit_1_pcd_Kp_ksp_type": "cg",
        "fieldsplit_1_pcd_Kp_ksp_max_it": 5,
        "fieldsplit_1_pcd_Kp_pc_type": "bjacobi",

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
    problem = NonlinearVariationalProblem(F, up, bcs=bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters, nullspace=nullspace, appctx=appctx)
    return solver

up.assign(0)
print('Cells: ', up.function_space().mesh().num_cells())
print('Z dim: ', Z.dim())

solver = create_solver(parLSC, appctx=appctx)

#w.assign(0)

solver.solve()
print('Z dim: ', Z.dim())
print('Cells: ', up.function_space().mesh().num_cells())
#solve(F == 0, up, bcs=bcs, nullspace=nullspace, solver_parameters=parPCD, appctx=appctx)

#u, p = up.split()
#u.rename("Velocity")
#p.rename("Pressure")

#File("cavity.pvd").write(u, p)

convergence(solver)

