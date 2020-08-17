from firedrake import *
from FPCD2 import*
from PCD import*
from petsc4py import PETSc

print = PETSc.Sys.Print

N = 16

M = UnitSquareMesh(N, N)

V = VectorFunctionSpace(M, "CG", 2)
W = FunctionSpace(M, "CG", 1)
Z = V * W

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
    return  (nu * inner(grad(u),grad(v))*dx + inner(dot(grad(u),u),v)*dx - p*div(v)*dx - q*div(u)*dx)


F = Constant(1.0/dt)*inner((u-u0),v)*dx + theta* F_(u,p,v,q) +(1-theta)*F_(u0,p0,v,q)



bcs = [DirichletBC(Z.sub(0), Constant((1, 0)), 4), DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3))]

nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

#appctx = {"Re": Re, "velocity_space": 0}
#appctx = {"nu" : nu, "velocity_space":0}
appctx = {"nu": nu, "velocity_space": 0, "dt": dt, "w": w0, "u": u}

parLSC = {
        "ksp_type": "fgmres",
        "ksp_monitor":None,
        #"ksp_max_it":100000,
        "ksp_gmres_restart":600,#200#600#500
        "ksp_rtol":1e-3,
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
            "pc_type":"lu",
            #"pc_hypre_boomeramg_max_it":3,
                        },
        "fieldsplit_1":{
            "ksp_type":"preonly",#gmres
            "pc_type":"lsc",
            #"pc_lsc_scale_diag":None,
            "lsc_pc_type":"lu",
            "lsc_pc_factor_mat_solver_type":"mumps",
            #"lsc_pc_sub_type":"ksp",
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
        "ksp_rtol": 1e-3,
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
        "fieldsplit_0_assembled_pc_type": "lu",
        "fieldsplit_0_assembled_pc_factor_solver_type": "mumps",

        "fieldsplit_1_ksp_type": "preonly",
        #"fieldsplit_1_ksp_rtol": 1e-4,
        "fieldsplit_1_pc_type": "python",
        "fieldsplit_1_pc_python_type": "FPCD2.PCD",

        "fieldsplit_1_pcd_Mp_ksp_type": "preonly",
        #"fieldsplit_1_pcd_Mp_ksp_max_it": 5,
        "fieldsplit_1_pcd_Mp_pc_type": "lu",

        "fieldsplit_1_pcd_Kp_ksp_type": "preonly",
        #"fieldsplit_1_pcd_Kp_ksp_max_it": 5,
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
    problem = NonlinearVariationalProblem(F, w, bcs=bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters, nullspace=nullspace, appctx=appctx)
    return solver

w.assign(0)
w0.assign(w)

outfile = File("Unsteady2D_NS.pvd")


#u0, p0 = w0.split()
#u0.rename("Velocity", "Velocity")
#p0.rename("Pressure","Pressure")


t = 0.0

solver = create_solver(solver_parameters = parDIR, appctx=appctx)

print("Solving ...")

while t<t_end:
    print("t= ",t)
    solver.solve()
    w0.assign(w)
    #u0, p0 = w0.split()
    #outfile.write(u0, p0, time=t)
    t += dt


print('Cells: ', w.function_space().mesh().num_cells())
print('Z dim: ', Z.dim())

#u, p = up.split()
#u.rename("Velocity")
#p.rename("Pressure")

#File("cavity.pvd").write(u, p)

convergence(solver)
