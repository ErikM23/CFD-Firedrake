from firedrake.preconditioners.base import PCBase
from firedrake.petsc import PETSc
import copy
from firedrake import Function
from firedrake import*


class PCD(PCBase):

    needs_python_pmat = True

    def initialize(self, pc):
        from firedrake import TrialFunction, TestFunction, Function, DirichletBC, dx, \
             assemble, Mesh, inner, grad, split, Constant, parameters
        from firedrake.assemble import allocate_matrix, create_assembly_callable
        #from assamble1 import allocate_matrix, apply_bcs, create_assembly_callable

        prefix = pc.getOptionsPrefix() + "pcd_"

        _, P = pc.getOperators()
        context = P.getPythonContext()

        test, trial = context.a.arguments()

        Q = test.function_space()

        p = TrialFunction(Q)
        q = TestFunction(Q)

        nu = context.appctx["nu"]
        dt = context.appctx["dt"]

        # pressure mass matri
        mass = (1.0/nu)*p*q*dx
        # stiffness matrix
        stiffness = inner(grad(p), grad(q))*dx
        
        velid = context.appctx["velocity_space"]
        #preid = context.appctx["pressure_space"]

        self.bcs = context.appctx["PCDbc"]
        #self.bcs = DirichletBC(Q,0,2)

        opts = PETSc.Options()
        
        default = parameters["default_matrix_type"]
        Mp_mat_type = opts.getString(prefix+"Mp_mat_type", default)
        Kp_mat_type = opts.getString(prefix+"Kp_mat_type", default)
        Fp_mat_type = opts.getString(prefix+"Fp_mat_type", "matfree")

        Mp = assemble(mass, form_compiler_parameters=context.fc_params,
                      mat_type=Mp_mat_type,
                      options_prefix=prefix + "Mp_")
        
        
        Kp = assemble(stiffness, bcs=self.bcs, form_compiler_parameters=context.fc_params,
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
        
        u=context.appctx["u"]
        fp = (1.0/nu)*((1.0/dt)*p + dot(u,grad(p)))*q*dx
        
        Fp = allocate_matrix(fp, form_compiler_parameters=context.fc_params,
                             mat_type=Fp_mat_type,
                             options_prefix=prefix + "Fp_")
        
        self._assemble_Fp = create_assembly_callable(fp, tensor=Fp,
                                                     form_compiler_parameters=context.fc_params,
                                                     mat_type=Fp_mat_type)
        
        self.Fp=Fp
        self._assemble_Fp()
        self.workspace = [Fp.petscmat.createVecLeft() for i in (0, 1)]
        self.tmp = Function(Q)
    
    def update(self, pc):
        #print("PC UPDATE")
        self._assemble_Fp()


    def apply(self, pc, x, y):
        a, b = self.workspace
        self.Mksp.solve(x, y)
        y.copy(result=a)
        self.Fp.petscmat.mult(a, b)

        with self.tmp.dat.vec_wo as v:
            b.copy(v)
        # Now tmp contains the value from `b`
        self.bcs.apply(self.tmp)
        with self.tmp.dat.vec_ro as v:
            v.copy(b)

        self.Kksp.solve(b, a)
        y.axpy(1.0,a)
        y.scale(-1.0)

    def applyTranspose(self, pc, x, y):
        print("TRANSPOSE")
        pass

    def view(self, pc, viewer=None):
        super(PCD, self).view(pc, viewer)
        viewer.printfASCII("Pressure-Convection-Diffusion inverse K^-1 F_p M^-1:\n")
        viewer.printfASCII("Reynolds number in F_p (applied matrix-free) is %s\n" %
                           str(1.0/self.nu))
        viewer.printfASCII("KSP solver for K^-1:\n")
        self.Aksp.view(viewer)
        viewer.printfASCII("KSP solver for M^-1:\n")
        self.Mksp.view(viewer)
