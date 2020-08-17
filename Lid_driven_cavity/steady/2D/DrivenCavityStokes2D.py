from firedrake import*
from firedrake.petsc import PETSc

print = PETSc.Sys.Print

#c_mesh = Mesh("2Dmesh.msh")

#meshes = MeshHierarchy(c_mesh, 3)

#mesh = meshes[1]

N = 128

mesh = UnitSquareMesh(N, N)
#meshes = MeshHierarchy(c_mesh, 3)

#mesh = meshes[2]

print(mesh.num_cells(), mesh.num_vertices())

V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = V*Q


#w = Function(W)
#u,p=split(w)
u, p = TrialFunctions(W)
v, q = TestFunctions(W)

x, y = SpatialCoordinate(mesh)
nu = Constant(1)
f = Constant((0,0))
       
a = nu*inner(grad(u),grad(v))*dx - p*div(v)*dx - div(u)*q*dx 

#F = nu*inner(grad(u),grad(v))*dx - p*div(v)*dx - div(v)*q*dx

L = inner(f,v)*dx

V = Constant(1)

bcs = [DirichletBC(W.sub(0), Constant((V, 0)), (4,)), DirichletBC(W.sub(0), Constant((0, 0)), (1, 2, 3))]

#inflow = as_vector([6*y*(0.41-y)/(0.41*0.41),0])

#noslipBC = DirichletBC(W.sub(0), (0,0), 3)
#inflowBC = DirichletBC(W.sub(0), interpolate(inflow, V), 1)
#cylinderBC = DirichletBC(W.sub(0), (0,0), 5)

#bcs = [inflowBC, noslipBC, cylinderBC]
       
nullspace = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True)])

appctx = {"nu" : nu, "velocity_space":0}

class MassMatrix(AuxiliaryOperatorPC):
    _prefix = "mass_"
    def form(self, pc, test, trial):
        nu = self.get_appctx(pc)["nu"]
        return (-1/nu *inner(test,trial)*dx, None)


parMin = {
        "ksp_monitor":None,
        "ksp_type":"minres",
        "ksp_max_it":100000,
        "pc_type":"none",
        }

parDIR = {
          "mat_type":"aij",
          "ksp_monitor":None,
          #"ksp_view":None,
          #"ksp_converged_reason":None,
          "ksp_type":"preonly",
          "pc_type":"lu",
          "pc_factor_mat_solver_type":"mumps",

        }

parMass = { 
       #"snes_type":"ksponly", 
       #"mat_type":"nest", 
       "ksp_type": "minres",
       #"ksp_rtol":1e-15,
       "ksp_monitor":None,
       #"ksp_view":None,
       #"ksp_converged_reason":None,
       #"ksp_view":None,
       "pc_type": "fieldsplit",
       "pc_fieldsplit_type":"schur",
       "pc_fieldsplit_schur_fact_type":"diag",
       "fieldsplit_0":{
           "ksp_type":"preonly",
           "pc_type":"hypre",#mg
           #"pc_sub_type":"ilu",
           #"mg_levels":{
                #"ksp_type":"chebyshev",
                #"ksp_max_it":1,
               #}
                    
           #"pc_hypre_type":"boomeramg",
           #"pc_hypre_boomeramg_max_it":5,
           },
       #"pc_fieldsplit_schur_precondition":"selfp",
       "fieldsplit_1":{
           "ksp_type":"preonly",#chebyshev
           #"ksp_max_it":2,
           "pc_type":"python",
           "pc_python_type":"__main__.MassMatrix",
           #"mass_ksp_type": "preonly",
           "mass_pc_type":"bjacobi",
           "mass_sub_pc_type":"ilu",
           #"mass_sub_pc_type":"ilu",#lu
           #"mass_sub_pc_type":"ilu",
           #"mass_ksp_rtol":1e-8,
           #"mass_pc_type":"lu",
           }
       }




parPCD = {
        "mat_type":"matfree",
        "ksp_type": "gmres",
        #"ksp_rtol":1e-8,
        "ksp_monitor":None,
        "ksp_converged_reason":None,
        #"ksp_view":None,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type":"schur",
        "pc_fieldsplit_schur_fact_type":"upper",
        "fieldsplit_0":{
              "ksp_type":"preonly",
              "pc_type":"python",
              "pc_python_type":"firedrake.AssembledPC",
              "assembled_pc_type":"hypre",
              #"assembled_pc_hypre_type":"boomeramg",
              #"assembled_pc_hypre_boomeramg_max_it":5,
              },
         #"pc_fieldsplit_schur_precondition":"selfp",
         "fieldsplit_1":{
             "ksp_type":"preonly",
             #"pc_type":"python",
             "pc_python_type":"firedrake.PCDPC",
             "pcd_Mp_ksp_type":"preonly",
             #"pcd_Mp_ksp_max_it":5,
             "pcd_Mp_pc_type":"lu",

             "pcd_Kp_ksp_type":"preonly",
             #"pcd_Kp_ksp_max_it":5,
             "pcd_Kp_pc_type":"hypre",
             
             "pcd_Fp_mat_type":"matfree",
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
#solve(F==0, w, bcs=bcs, nullspace=nullspace, appctx=appctx, solver_parameters=parMass)

problem = LinearVariationalProblem(a, L, w, bcs)
solver = LinearVariationalSolver(problem, solver_parameters=parDIR, appctx=appctx)

print("Solving ...")
solver.solve()

#convergence(solver)
#outfile = File("STOKES_2D.pvd")
#u, p = w.split()
#u.rename("Velocity","Velocity")
#p.rename("Pressure","Pressure")
#outfile.write(u,p)
convergence(solver)
print('cells: ',w.function_space().mesh().num_cells())
#print('V dim: ',V.dim())
#print('P dim: ',Q.dim())
print('W dim: ',W.dim())
