from firedrake import*
from firedrake.petsc import PETSc
from PCD import*
from PCD1 import*
from FPCD import*
from FPCD2 import*
# creating of nonlinear solver for NS flow past the cylinder in 3D

#def create_solver():

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
        #"ksp_monitor":None,
        "ksp_gmres_restart":150,#200#600#500
        #"ksp_rtol":1e-6,
        "snes_monitor":None,
        "snes_converged_reason":None,
        "ksp_gmres_modifiedgramschmidt":True,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type":"schur",
        "pc_fieldsplit_schur_fact_type":"upper",
        "fieldsplit_0":{
             "ksp_type":"gmres",
             #"ksp_max_it":2,
             "ksp_gmres_restart":20, #50
             #"ksp_monitor":None,
             "pc_type":"bjacobi",
             #"mat_type":"aij",
             #"pc_hypre_type":"boomeramg",
             #"pc_hypre_boomeramg_strong_threshold": 0.2,
             },
         "fieldsplit_1":{
             "ksp_type":"preonly",#gmres
             #"ksp_max_it":2,#50#200#300
             #"ksp_max_it":10,
             #"ksp_monitor":None,
             "pc_type":"lsc",
             "lsc_pc_type":"ksp",
             "lsc_pc_ksp_type":"richardson",#gmres
             "lsc_pc_ksp_max_it":2,
             #"lsc_pc_ksp_gmres_restart":50,
             "lsc_pc_pc_type":"hypre",#bjacobi
             "lsc_pc_pc_hypre_type":"boomeramg",
             "lsc_pc_pc_hypre_boomeramg_strong_threshold": 0.2,
             "lsc_pc_pc_hypre_boomeramg_P_max":2,
             #"lsc_pc_ksp_rtol":1e-4,
             }
        }


parPCD1 = {
          "ksp_type": "fgmres",
          "ksp_rtol":1e-2,
          "snes_rtol":1e-5,
          "mat_type":"matfree",
          "snes_monitor":None,
          #"ksp_rtol":1e-2,
          "snes_converged_reason":None,
          #"ksp_monitor":None,
          #"ksp_max_it":1000,
          "ksp_gmres_restart":800,
          #"ksp_rtol":1e-10,
          "ksp_gmres_modifiedgramschmidt":None,
          "pc_type": "fieldsplit",
          "pc_fieldsplit_type":"schur",
          "pc_fieldsplit_schur_fact_type":"full",
          "pc_fieldsplit_schur_precondition":"selfp",
          "fieldsplit_0":{
              "ksp_type":"preonly", #bcgsl
              #"pc_factor_mat_solver_type":"mumps",
              #"ksp_gmres_restart":20,
              #"ksp_rtol":1e-3,#5 #10
              #"ksp_monitor":None,
              #"ksp_converged_reason":True,
              #"ksp_max_it":2,
              "pc_type":"python",
              #"pc_side":"right",
              "pc_python_type":"firedrake.AssembledPC",
              "assembled_pc_type":"lu",
             # "assembled_mg_levels":{
             #     "ksp_type":"chebyshev",
             #     "ksp_max_it":2,
             #     },
              #"assembled_sub_pc_type":"hypre",
              },
          "fieldsplit_1":{
              "ksp_type":"preonly",
              #"ksp_gmres_restart":550,
              #"ksp_monitor":None,
              #"ksp_rtol":1e-2,
              "pc_type":"python",
              "pc_python_type":"FPCD.PCD",
              
              #"pcd_Mp_mat_type":"aij",
              "pcd_Mp_ksp_type":"richardson", #gmres
              #"pcd_Mp_ksp_chebyshev_eigenvalues":"0.5,2.2",
              #"pcd_Mp_ksp_monitor":None,
              "pcd_Mp_ksp_max_it":15,
              "pcd_Mp_pc_type":"sor",
              #"pcd_Mp_pc_sub_type":"ilu",

              #"pcd_Kp_ksp_monitor":None,
              "pcd_Kp_ksp_type":"gmres",
              "pcd_Kp_ksp_max_it":15,
              #"pcd_Kp_ksp_rtol":1e-8,
              #"pcd_Kp_ksp_gmres_restart":400,
              #"pcd_Kp_ksp_gmres_modifiedgramschmidt":True,
              "pcd_Kp_pc_type":"hypre",
              #"pcd_Kp_pc_sub_type":"ilu",
              #"pcd_Kp_mg_levels":{
              #    "ksp_type":"richardson",
              #    "ksp_max_it":2,
              #    },

              "pcd_Fp_mat_type":"matfree"
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
    #"ksp_max_it":800,
    "ksp_gmres_restart":600,
    "ksp_gmres_modifiedgramschmidt":None,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type":"schur",
    "pc_fieldsplit_schur_fact_type":"full",
   # "pc_fieldsplit_schur_precondition": "a11",
    "fieldsplit_0":{
        #"ksp_converged_reason":None,
        #"ksp_monitor":None,
        "ksp_type":"gmres",
        "ksp_max_it":10,
        "ksp_rtol":1e-1,
        "ksp_gmres_restart": 100,
        "pc_type":"python",
        #"pc_side":"right",
        "pc_python_type": "firedrake.AssembledPC",
        "assembled_pc_type": "bjacobi",
        #"assembled_pc_factor_mat_solver_type": "mumps",
        },
    "fieldsplit_1":{
        #"ksp_converged_reason":None,
        #"ksp_monitor":None,
        "ksp_type":"gmres",
        "ksp_rtol":1e-1,
        "ksp_max_it":10,
        "pc_type":"python",
        "pc_python_type":"FPCD2.PCD",
        "pcd_Mp" :{
            "ksp_type":"richardson",
            "ksp_rtol":1e-1,
            "ksp_max_it":5,
            "pc_type":"sor",
            #"ksp_converged_reason":None,
            },
        "pcd_Kp": {
            "ksp_type":"preonly",
            #"ksp_rtol":1e-1,
            #"ksp_max_it": 5,
            #"ksp_gmres_restart":50,
            "pc_type":"hypre",
            "pc_hypre_type":"boomeramg",
            "pc_hypre_boomeramg_strong_threshold": 0.2,
            "pc_hypre_boomeramg_P_max":2,
            #"ksp_converged_reason":None,
            #"pc_type":"telescope",
            #"telescope_pc_type":"ksp",
            #"telescope_ksp_ksp_max_it": 3,
            #"telescope_ksp_ksp_type":"richardson",
            #"telescope_ksp_pc_type": "hypre",
            #"telescope_ksp_pc_hypre_boomeramg_P_max": 4,
            },
        "pcd_Fp_mat_type":"matfree"
        }
        
    }


#mesh = Mesh("3Dmesh.msh")
mesh = Mesh("bench3D_ns.msh")
#hierarchy = MeshHierarchy(c_mesh,2 )
#mesh = hierarchy[2]

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
        
    
    # residual
    #F = ( 1.0 / Re * inner(grad(u), grad(v)) * dx + inner(dot(grad(u), u), v) * dx - p * div(v) * dx + div(u) * q * dx)
F = Constant(1.0/dt)*inner((u-u0),v)*dx + theta* F_(u,p,v,q) +(1-theta)*F_(u0,p0,v,q)    
   
J = derivative(F, w)
#J = (nu*inner(grad(u),grad(v))+inner(dot(grad(u),u0),v)-p*div(v)-q*div(u))*dx
    # coordinates
x, y, z = SpatialCoordinate(mesh)

inflow = as_vector([ 2.25 * y * (0.41-y) * z * (0.41-z) / ( pow(0.205,4) ),0,0])
    
    # BC
bcs = [DirichletBC(W.sub(0), Constant((0, 0, 0)), (3,4)), DirichletBC(W.sub(0), inflow, 1)]

PCDbc=DirichletBC(Q,0,2)
appctx = {"nu": nu, "velocity_space": 0, "dt": dt, "w": w0, "u": u, "PCDbc": PCDbc}
    # nullspace
nullspace = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True)])
   
    # def of problem
problem = NonlinearVariationalProblem(F, w, bcs = bcs)
    
    # def of solver
solver = NonlinearVariationalSolver(problem, solver_parameters=parDIR, appctx=appctx)
    
    #return solver

#outfile = File("NS3D_cylinder_flow_time.pvd")
#solver = create_solver()

t = 0.0

w.assign(0)
w0.assign(w)

u0, p0 = w0.split()

#u0.rename("Velocity", "Velocity")
#p0.rename("Pressure", "Pressure")

print('cells: ',w.function_space().mesh().num_cells())
print('dim: ', W.dim())


print("Solving ...")
while t<t_end:
        print("t= ",t)
        solver.solve()
        w0.assign(w)
        #u0, p0 = w0.split()
        #outfile.write(u0, p0, time=t)
        t += dt


def convergence(solver):
    from firedrake.solving_utils import KSPReasons, SNESReasons
    snes = solver.snes
    print("""
SNES iterations: {snes}; SNES converged reason: {snesreason}
KSP iterations: {ksp}; KSP converged reason: {kspreason}""".format(snes=snes.getIterationNumber(),snesreason=SNESReasons[snes.getConvergedReason()],
                                                                   ksp=snes.ksp.getIterationNumber(), kspreason=KSPReasons[snes.ksp.getConvergedReason()]))

convergence(solver)
print('dim: ', W.dim())
