from firedrake import*
from firedrake.petsc import PETSc
from FPCD import*

print = PETSc.Sys.Print

parDIR={
    "snes_monitor": None,
    "ksp_type": "preonly",
    "mat_type": "aij",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"
    }


parLSC = {
    "ksp_type": "fgmres",
    "ksp_gmres_restart":150,
    "snes_monitor":None,
    "snes_converged_reason":None,
    "ksp_gmres_modifiedgramschmidt":True,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type":"schur",
    "pc_fieldsplit_schur_fact_type":"upper",
    "fieldsplit_0":{
         "ksp_type":"gmres",
         "ksp_gmres_restart":20,
         "pc_type":"bjacobi",
         },
     "fieldsplit_1":{
         "ksp_type":"preonly",
         "pc_type":"lsc",
         "lsc_pc_type":"ksp",
         "lsc_pc_ksp_type":"richardson",
         "lsc_pc_ksp_max_it":2,
         "lsc_pc_pc_type":"hypre",
         "lsc_pc_pc_hypre_type":"boomeramg",
         "lsc_pc_pc_hypre_boomeramg_strong_threshold": 0.2,
         "lsc_pc_pc_hypre_boomeramg_P_max":2,
         }
    }


parPCD = {
    "mat_type":"matfree",
    "ksp_type": "fgmres",
    "ksp_monitor":None,
    "ksp_converged_reason":None,
    "snes_monitor":None,
    "snes_converged_reason":None,
    "ksp_rtol": 1e-3,
    "ksp_gmres_restart":600,
    "ksp_gmres_modifiedgramschmidt":None,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type":"schur",
    "pc_fieldsplit_schur_fact_type":"upper",
    "fieldsplit_0":{
        "ksp_type":"gmres",
        "ksp_max_it":10,
        "ksp_rtol":1e-1,
        "ksp_gmres_restart": 100,
        "pc_type":"python",
        #"pc_side":"right",
        "pc_python_type": "firedrake.AssembledPC",
        "assembled_pc_type": "bjacobi",
        },
    "fieldsplit_1":{
        "ksp_type":"gmres",
        "ksp_rtol":1e-1,
        "ksp_max_it":10,
        "pc_type":"python",
        "pc_python_type":"FPCD.PCD",
        "pcd_Mp" :{
            "ksp_type":"richardson",
            "ksp_rtol":1e-1,
            "ksp_max_it":5,
            "pc_type":"sor",
            },
        "pcd_Kp": {
            "ksp_type":"preonly",
            "pc_type":"hypre",
            "pc_hypre_type":"boomeramg",
            "pc_hypre_boomeramg_strong_threshold": 0.2,
            "pc_hypre_boomeramg_P_max":2,
            },
        "pcd_Fp_mat_type":"matfree"
        }
        
    }

mesh = Mesh("3Dmesh.msh")

V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = V * Q

w = Function(W)
w0 = Function(W)
u, p = split(w)
u0, p0 = split(w0)
v, q = TestFunctions(W)

Re = Constant(100.0)
nu = Constant(0.001)

dt = 0.01
t_end = 5 
theta = Constant(0.5)


def F_(u,p,v,q) :
        return  ( nu*inner(grad(u),grad(v))*dx + inner(dot(grad(u),u),v)*dx - p*div(v)*dx + q*div(u)*dx)
        
    

F = Constant(1.0/dt)*inner((u-u0),v)*dx + theta* F_(u,p,v,q) +(1-theta)*F_(u0,p0,v,q)    
   
J = derivative(F, w)

x, y, z = SpatialCoordinate(mesh)

inflow = as_vector([ 2.25 * y * (0.41-y) * z * (0.41-z) / ( pow(0.205,4) ),0,0])
    
bcs = [DirichletBC(W.sub(0), Constant((0, 0, 0)), (3,4)), DirichletBC(W.sub(0), inflow, 1)]

PCDbc=DirichletBC(Q,0,2)

appctx = {"nu": nu, "velocity_space": 0, "dt": dt, "w": w0, "u": u, "PCDbc": PCDbc}
   
problem = NonlinearVariationalProblem(F, w, bcs = bcs)
    
solver = NonlinearVariationalSolver(problem, solver_parameters=parDIR, appctx=appctx)
    

t = 0.0

w.assign(0)
w0.assign(w)

while t<t_end:
        print("t= ",t)
        solver.solve()
        w0.assign(w)
        t += dt
