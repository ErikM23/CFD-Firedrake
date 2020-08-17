from firedrake.preconditioners.base import PCBase
from firedrake.petsc import PETSc
import copy
from firedrake import Function
from firedrake import*

class PCD(PCBase):

    needs_python_pmat = True

    def initialize(self, pc):
        from firedrake import TrialFunction, TestFunction, Function, DirichletBC, dx, \
             Mesh, inner, grad, split, Constant, parameters
        from firedrake.assemble import allocate_matrix, create_assembly_callable
        #from assamble import allocate_matrix, create_assembly_callable
        prefix = pc.getOptionsPrefix() + "pcd_"

        _, P = pc.getOperators()
        context = P.getPythonContext()

        test, trial = context.a.arguments()

        Q = test.function_space()

        #mesh = Mesh("2Dmesh_unsteady2D.msh")
        #Q = FunctionSpace(mesh, "CG", 1)

        self.Q = Q

        p = TrialFunction(Q)
        q = TestFunction(Q)

        nu = context.appctx.get("nu", 1.0)
        self.nu = nu

        # pressure mass matri
        mass = Constant(1.0/self.nu)*p*q*dx
        # stiffness matrix
        stiffness = inner(grad(p), grad(q))*dx
        
        state = context.appctx["state"]

        velid = context.appctx["velocity_space"]
        
        
        #preid = context.appctx["pressure_space"]

        #u0 = split(state)[velid]
        
        #self.bcs = context.appctx.get("bcsPCD")
        #self.P = context.appctx["P"]
        # boundary conditions ------ TO DO with IF ELSE
        #self.bcs = DirichletBC(Q,0,2)

        opts = PETSc.Options()
        
        default = parameters["default_matrix_type"]
        Mp_mat_type = opts.getString(prefix+"Mp_mat_type", default)
        Kp_mat_type = opts.getString(prefix+"Kp_mat_type", default)
        self.Fp_mat_type = opts.getString(prefix+"Fp_mat_type", "matfree")

        Mp = assemble(mass, form_compiler_parameters=context.fc_params,
                      mat_type=Mp_mat_type,
                      options_prefix=prefix + "Mp_")
        
        
        Kp = assemble(stiffness, form_compiler_parameters=context.fc_params,
                      mat_type=Kp_mat_type,
                      options_prefix=prefix + "Kp_")


        Mksp = PETSc.KSP().create(comm=pc.comm)
        Mksp.incrementTabLevel(1, parent=pc)
        Mksp.setOptionsPrefix(prefix + "Mp_")
        Mksp.setOperators(Mp.petscmat)
        Mksp.setUp()
        Mksp.setFromOptions()
        self.Mksp = Mksp

        Kksp = PETSc.KSP().create(comm=pc.comm)
        Kksp.incrementTabLevel(1, parent=pc)
        Kksp.setOptionsPrefix(prefix + "Kp_")
        Kksp.setOperators(Kp.petscmat)
        Kksp.setUp()
        Kksp.setFromOptions()
        self.Kksp = Kksp

        #state = context.appctx["state"]

        #velid = context.appctx["velocity_space"]
        
        u0 = split(state)[velid]
        fp = Constant(1.0/self.nu)*inner(u0,grad(p))*q*dx
        #Constant(self.nu)*inner(grad(p), grad(q))*dx + inner(u0, grad(p))*q*dx

        self.Fp = allocate_matrix(fp, form_compiler_parameters=context.fc_params,
                                  mat_type=self.Fp_mat_type,
                                  options_prefix=prefix + "Fp_")

        self._assemble_Fp = create_assembly_callable(fp, tensor=self.Fp,
                                                     form_compiler_parameters=context.fc_params,
                                                     mat_type=self.Fp_mat_type)
        self._assemble_Fp()

        Fpmat = self.Fp.petscmat
        self.workspace = [Fpmat.createVecLeft() for i in (0, 1)]
        #self.bcs.zero(self.Fp)
        self.tmp = Function(self.Q)
    
    def update(self, pc):
        self._assemble_Fp()


    def apply(self, pc, x, y):
        a, b = self.workspace
        #c = self.context.a.arguments()
        #self.bcs.apply(c)

        #x.copy(a)
        #x.copy(b)

        #tmp = Function(self.Q)
        #with tmp.dat.vec_wo as v:
        #        x.copy(v)
                # Now tmp contains the value from `x`
        #self.bcs.apply(tmp)
        #with tmp.dat.vec_ro as v:
        #        v.copy(x)
                
        
        self.Mksp.solve(x, y)
        y.copy(a)
        #a.scale(1.0/self.nu)
                
        #self.bcs.apply(self.Fp)
        self.Fp.petscmat.mult(a, b)
        # BC
        #self.bcs.apply(x)

        #tmp = Function(self.Q)

        #with self.tmp.dat.vec_wo as v:
        #    b.copy(v)
        # Now tmp contains the value from `x`
        #self.bcs.apply(self.tmp)
        #with self.tmp.dat.vec_ro as v:
        #    v.copy(b)

        self.Kksp.solve(b, a)

        y.axpy(1.0, a)
        y.scale(-1.0)

    def applyTranspose(self, pc, x, y):
        
        #from firedrake.assemble import apply_bcs
        a, b = self.workspace
        print("TRANSPOSE")
        pass
        x.copy(result=b)
        x.copy(result=a)
        #self.bcs.apply(a)
        #a = apply_bcs(a, bcs=self.bcs)
        #self.bcs.apply(a)
        self.Kksp.solveTranspose(a, y)

        self.Fp.petscmat.multTranspose(y, a)
        a.axpy(1.0, x)

        self.Mksp.solveTranspose(a, y)

        y.axpy(1.0,a)
        y.scale(-1.0)
