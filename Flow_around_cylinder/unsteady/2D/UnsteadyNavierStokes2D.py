from firedrake import*
from firedrake.petsc import PETSc
from FPCD3 import*
from GPCD import *
from PCD import*
#c_mesh = Mesh("2Dmesh.msh")

#meshes = MeshHierarchy(c_mesh, 3)

#for mesh in meshes:
#    Vc = mesh.coordinates.function_space()
#    x, y = SpatialCoordinate(mesh)
#    f = Function(Vc).interpolate(mesh.coordinates)

#    ur =0.05*as_vector([(x)/sqrt((x-0.5)**2+(y-0.2)**2), (y)/sqrt((x-0.5)**2+(y-0.2)**2)])#/sqrt((x)**2 + (y)**2)])

#    bc = DirichletBC(Vc, ur, 5)

#    plex = mesh._plex
#    plex.view()
#    bc.apply(f)
#    mesh.coordinates.assign(f)


#mesh = meshes[-1]

print = PETSc.Sys.Print

mesh = Mesh("2Dmesh.msh")

#meshes = MeshHierarchy(c_mesh, 4)

#mesh = meshes[1]


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

#J = derivative(F, w)

inflow = as_vector([6.0*y*(0.41-y)/(0.41*0.41),0.0])

noslipBC = DirichletBC(W.sub(0), (0,0), 3)
inflowBC = DirichletBC(W.sub(0), interpolate(inflow,V), 1)
cylinderBC = DirichletBC(W.sub(0), (0,0), 5)

PCDbc=DirichletBC(Q,0,2)
#print("start")
#print(bcsPCD)
#print("end")
bcs = [inflowBC, noslipBC, cylinderBC]
       
#nullspace = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True)])

appctx = {"nu": nu, "velocity_space": 0, "dt": dt, "w": w0, "u": u, "PCDbc": PCDbc}
#appctx = {"nu": nu, "velocity_space": 0, "bcsPCD": bcsPCD}


def create_solver(solver_parameters, appctx):
    problem = NonlinearVariationalProblem(F, w, bcs=bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters, appctx=appctx)
    return solver


parLSC = {
       "ksp_type": "fgmres",
       #"ksp_monitor":none,
       "ksp_gmres_restart":150,
       #"ksp_rtol":1e-6,
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
           #"mg_levels":{
           #    "ksp_type":"chebyshev",
           #    "ksp_max_it":2,
           #    },
                      },
       "fieldsplit_1":{
           #"pc_fieldsplit_schur_precondition":"selfp",
           "ksp_type":"preonly",
           #"ksp_gmres_restart":100,
           "pc_type":"lsc",
           "lsc_pc_type":"lu",
                      }
       }




parPCD3 = {
        "ksp_type": "fgmres",
        "mat_type":"matfree",
        "snes_monitor":None,
        "snes_converged_reason":None,
        #"ksp_monitor":None,
        "ksp_gmres_restart":750,
        "ksp_gmres_modifiedgramschmidt":None,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type":"schur",
        "pc_fieldsplit_schur_fact_type":"lower",
        "fieldsplit_0":{
            "ksp_type":"richardson", #gmres
            "ksp_max_it":3,#5 #10
            #"ksp_monitor":None,
            #"ksp_converged_reason":True,
            "pc_type":"python",
            #"pc_side":"right",
            "pc_python_type":"firedrake.AssembledPC",
            "assembled_pc_type":"hypre",
            "assembled_mat_type":"aij",
            "assembled_pc_hypre_type":"boomeramg",
            #"assembled_pc_hypre_boomeramg_strong_threshold": 0.2,
            "assembled_pc_hypre_boomeramg_P_max":2,
            },
        "fieldsplit_1":{
            "ksp_type":"preonly",
            "pc_python_type":"FPCD2.PCD",

            "pcd_Mp_ksp_type":"chebyshev", #gmres
            #"pcd_Mp_ksp_monitor":None,
            "pcd_Mp_ksp_max_it":5,
            "pcd_Mp_pc_type":"jacobi", #bjacobi
            #"pcd_Mp_pc_mg_levels":2,
            #"pcd_Mp_pc_mg_cycle_type":"w",
            #"pcd_Mp_ksp_rtol":1e-12,
            #"pcd_Mp_ksp_monitor":None,
            #"pcd_Mp_pc_type":"mg",
            #"pcd_Mp_pc_hypre_type":"boomeramg",
            #"pcd_Mp_pc_hypre_boomeramg_P_max":2,

            "pcd_Kp_ksp_type":"gmres",
            #"pcd_Kp_ksp_max_it":2,
            "pcd_Kp_ksp_gmres_restart":200,#300#400
            #"pcd_Kp_ksp_monitor":None,
            #"pcd_Kp_ksp_converged_reason":None,
            "pcd_Kp_pc_side":"right",
            "pcd_Kp_pc_type": "hypre",
            "pcd_Kp_pc_hypre_type":"boomeramg",
            "pcd_Kp_pc_hypre_boomeramg_strong_threshold": 0.2,
            #"pcd_Kp_pc_hypre_boomeramg_agg_nl": 4,
            #"pcd_Kp_pc_hypre_boomeramg_agg_num_paths": 5,
            "pcd_Kp_pc_hypre_boomeramg_max_levels": 5,
            #"pcd_Kp_pc_hypre_boomeramg_coarsen_type": "HMIS",
            #"pcd_Kp_pc_hypre_boomeramg_interp_type": "ext+i",
            "pcd_Kp_pc_hypre_boomeramg_P_max" :2,
            "pcd_Kp_pc_hypre_boomeramg_truncfactor": 0.3,
            
            "pcd_Fp_mat_type":"matfree"
            }
        }



parDIR = {
          "snes_monitor": None,
          #"snes_converged_reason":None,
          "ksp_type": "preonly",#gmres
          #"ksp_monitor":None,
          "mat_type": "aij",
          "pc_type": "lu",
          "pc_factor_mat_solver_type": "mumps"
          }



parPCD = {
       "ksp_type":"fgmres",
       #"ksp_rtol":1e-4,
       #"ksp_monitor":None,
       #"ksp_max_it":1000,
       #"ksp_rtol":1e-8,
       "mat_type":"matfree",
       #"snes_rtol":1e-5,
       "snes_monitor":None,
       "snes_converged_reason":None,
       "ksp_gmres_restart":800,
       "ksp_gmres_modifiedgramschmidt":None,
       "pc_type": "fieldsplit",
       "pc_fieldsplit_type":"schur",
       "pc_fieldsplit_schur_fact_type":"upper",
       "fieldsplit_0":{
           "ksp_type":"preonly",
           #"ksp_gmres_restart":200,
           #"ksp_max_it":5,
           #"ksp_monitor":None,
           #"ksp_converged_reason":True,
           #"pc_side":"right",
           "pc_type":"python",
           "pc_python_type":"firedrake.AssembledPC",
           "assembled_mat_type":"aij",
           "assembled_pc_type":"lu",#hypre
           #"assembled_pc_ksp_type":"gmres",
           #"assembled_pc_pc_type":"hypre",
           #"assambled_pc_sub_type":"ilu",
           #"assembled_mat_type":"aij",
          # "assembled_pc_hypre_type":"boomeramg",
           #"assembled_pc_hypre_boomeramg_strong_threshold": 0.2,
           #"assembled_pc_hypre_boomeramg_P_max":2,
           },
       "fieldsplit_1":{
           "ksp_type":"preonly",
           #"ksp_max_it":5,
           #"ksp_gmres_restart":300,
           #"ksp_gmres_modifiedgramschmidt":None,
           #"ksp_rtol":1e-2,
           #"ksp_monitor":None,
           "pc_type":"python",
           #"pc_side":"right",
           "pc_python_type":"FPCD3.PCD",

           # MASS MATRIX Mp = p*q*dx
           "pcd_Mp_ksp_type":"preonly",#richardson
           #"pcd_Mp_ksp_gmres_restart":500,
           #"pcd_Mp_ksp_max_it":5,
           #"pcd_Mp_ksp_chebyshev_eigenvalues":"0.5,2.0",
           "pcd_Mp_pc_type":"lu",
 
           # STIFNESS MATRIX Kp = inner(grad(p),grad(q))*dx
           "pcd_Kp_ksp_type":"preonly",
           #"pcd_Kp_ksp_monitor":None,
           #"pcd_Kp_ksp_gmres_restart":100,
           #"pcd_Kp_pc_side":"right",
           #"pcd_Kp_ksp_max_it":10,
           #"pcd_Kp_ksp_monitor":None,
           #"pcd_Kp_ksp_rtol":1e-8,
           #"pcd_Kp_ksp_converged_reason":None,
           "pcd_Kp_pc_type": "lu",#hypre
           #"pcd_Kp_pc_sub_type":"ilu",
           #"pcd_Kp_pc_hypre_type":"boomeramg",
           #"pcd_Kp_pc_hypre_boomeramg_strong_treshold":0.2,
           #"pcd_Kp_pc_hypre_boomeramg_P_max":2,

           "pcd_Fp_mat_type":"matfree"
           },
      }  



def convergence(solver):
    from firedrake.solving_utils import KSPReasons, SNESReasons
    snes = solver.snes
    print("""
SNES iterations: {snes}; SNES converged reason: {snesreason}
KSP iterations: {ksp}; KSP converged reason: {kspreason}""".format(snes=snes.getIterationNumber(),snesreason=SNESReasons[snes.getConvergedReason()],
                                                            ksp=snes.ksp.getIterationNumber(), kspreason=KSPReasons[snes.ksp.getConvergedReason()]))



outfile = File("Cylinder2DunsteadyNS.pvd")
solver = create_solver(parPCD, appctx=appctx)



t = 0.0

w.assign(0)
w0.assign(w)

u0, p0 = w0.split()

u0.rename("Velocity", "Velocity")
p0.rename("Pressure", "Pressure")

print('dim: ', W.dim())
print('cells: ', w.function_space().mesh().num_cells())



print("Solving ...")
while t<t_end:
    print("t= ",t)

    solver.solve()

    w0.assign(w)
    u0, p0 = w0.split()
    
    outfile.write(u0, p0, time=t)
    t += dt

print('dim: ', W.dim())
convergence(solver)
