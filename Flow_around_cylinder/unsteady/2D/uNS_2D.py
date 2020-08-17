from firedrake import*
from firedrake.petsc import PETSc
from FPCD import*

print = PETSc.Sys.Print

mesh = Mesh("2Dmesh.msh")

V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = V*Q

w = Function(W)
w0 = Function(W)
u, p = split(w)
u0, p0 = split(w0)
v, q = TestFunctions(W)

x, y = SpatialCoordinate(mesh)
Re = Constant(100.0)
nu = Constant(0.001)
f = as_vector([0, 0])

dt = 0.01
t_end = 5 
theta = Constant(0.5)  # Crank-Nicolson

def F_(u,p,v,q) :
    return  (Constant(nu) * inner(grad(u),grad(v))*dx + inner(dot(grad(u),u),v)*dx - p*div(v)*dx - q*div(u)*dx)

# u = new solution, u0 = previous solution
F = Constant(1.0/dt)*inner((u-u0),v)*dx + theta* F_(u,p,v,q) +(1-theta)*F_(u0,p0,v,q)

inflow = as_vector([6.0*y*(0.41-y)/(0.41*0.41),0.0])

noslipBC = DirichletBC(W.sub(0), (0,0), 3)
inflowBC = DirichletBC(W.sub(0), interpolate(inflow,V), 1)
cylinderBC = DirichletBC(W.sub(0), (0,0), 5)

PCDbc=DirichletBC(Q,0,2)

bcs = [inflowBC, noslipBC, cylinderBC]
       
appctx = {"nu": nu, "velocity_space": 0, "dt": dt, "w": w0, "u": u, "PCDbc": PCDbc}


def create_solver(solver_parameters, appctx):
    problem = NonlinearVariationalProblem(F, w, bcs=bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters, appctx=appctx)
    return solver


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
           "ksp_type":"preonly",
           "pc_type":"lu",
           "pc_lsc_factor_mat_solver_type": "mumps",
                      },
       "fieldsplit_1":{
           "ksp_type":"preonly",
           "pc_type":"lsc",
           "lsc_pc_type":"lu",
                      }
       }


parDIR = {
          "snes_monitor": None,
          "ksp_type": "preonly",
          "mat_type": "aij",
          "pc_type": "lu",
          "pc_factor_mat_solver_type": "mumps"
          }



parPCD = {
       "ksp_type":"fgmres",
       "mat_type":"matfree",
       "snes_monitor":None,
       "snes_converged_reason":None,
       "ksp_gmres_restart":800,
       "ksp_gmres_modifiedgramschmidt":None,
       "pc_type": "fieldsplit",
       "pc_fieldsplit_type":"schur",
       "pc_fieldsplit_schur_fact_type":"upper",
       "fieldsplit_0":{
           "ksp_type":"preonly",
           "pc_type":"python",
           "pc_python_type":"firedrake.AssembledPC",
           "assembled_mat_type":"aij",
           "assembled_pc_type":"lu",
           },
       "fieldsplit_1":{
           "ksp_type":"preonly",
           "pc_type":"python",
           "pc_python_type":"FPCD.PCD",

           # MASS MATRIX Mp = p*q*dx
           "pcd_Mp_ksp_type":"preonly",
           "pcd_Mp_pc_type":"lu",
 
           # STIFNESS MATRIX Kp = inner(grad(p),grad(q))*dx
           "pcd_Kp_ksp_type":"preonly",
           "pcd_Kp_pc_type": "lu",

           "pcd_Fp_mat_type":"matfree"
           },
      }  


solver = create_solver(parPCD, appctx=appctx)

t = 0.0

w.assign(0)
w0.assign(w)

print("Solving ...")
while t<t_end:
    print("t= ",t)
    solver.solve()
    w0.assign(w)
    t += dt
