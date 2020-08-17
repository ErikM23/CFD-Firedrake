from firedrake import *
from FPCD import*

print = PETSc.Sys.Print

N = 128

M = UnitSquareMesh(N, N)

V = VectorFunctionSpace(M, "CG", 2)
W = FunctionSpace(M, "CG", 1)
Z = V * W

up = Function(Z)
u, p = split(up)
v, q = TestFunctions(Z)

nu = Constant(0.01)
Re = Constant(100)
F = (nu * inner(grad(u), grad(v)) * dx + inner(dot(grad(u), u), v) * dx - p * div(v) * dx - div(u) * q * dx )

bcs = [DirichletBC(Z.sub(0), Constant((1, 0)), 4), DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3))]

nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

appctx = {"nu" : nu, "velocity_space":0}

LSC = {
        "ksp_type": "fgmres",
        "ksp_monitor":None,
        "ksp_gmres_restart":600,
        "snes_monitor":None,
        "snes_converged_reason":None,
        "ksp_gmres_modifiedgramschmidt":True,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type":"schur",
        "pc_fieldsplit_schur_fact_type":"upper",
        "fieldsplit_0":{
            "ksp_type":"preonly",
            "pc_type":"lu",
                        },
        "fieldsplit_1":{
            "ksp_type":"preonly",
            "pc_type":"lsc",
            "lsc_pc_type":"lu",
            "lsc_pc_factor_mat_solver_type":"mumps",
                        }
        }


DIR={
        "snes_monitor": None,
        "snes_max_it":1000,
        "ksp_type": "preonly",
        "mat_type": "aij",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"}

PCD = {
        "mat_type": "matfree",
        "snes_monitor": None,
        "ksp_monitor":None,
        "ksp_type": "fgmres",
        "ksp_gmres_restart": 600,
        "ksp_gmres_modifiedgramschmidt": None,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "upper",

        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "python",
        "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
        "fieldsplit_0_assembled_pc_type": "lu",

        "fieldsplit_1_ksp_type": "preonly",
        "fieldsplit_1_pc_type": "python",
        "fieldsplit_1_pc_python_type": "FPCD.PCD",

        "fieldsplit_1_pcd_Mp_ksp_type": "preonly",
        "fieldsplit_1_pcd_Mp_pc_type": "lu",

        "fieldsplit_1_pcd_Kp_ksp_type": "preonly",
        "fieldsplit_1_pcd_Kp_pc_type": "lu",

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

solver = create_solver(parPCD, appctx=appctx)

solver.solve()
print('Z dim: ', Z.dim())
print('Cells: ', up.function_space().mesh().num_cells())

#u, p = up.split()
#u.rename("Velocity")
#p.rename("Pressure")

#File("cavity.pvd").write(u, p)

convergence(solver)

