import taichi as ti
import utils

@ti.data_oriented
class MGPCGPoissonSolver:
    def __init__(self, dim, res, n_mg_levels = 4, pre_and_post_smoothing = 2, bottom_smoothing = 50, real = float):

        self.FLUID = utils.FLUID
        self.SOLID = utils.SOLID
        self.AIR = utils.AIR

        # grid parameters
        self.dim = dim
        self.res = res
        self.n_mg_levels = n_mg_levels
        self.pre_and_post_smoothing = pre_and_post_smoothing
        self.bottom_smoothing = bottom_smoothing
        self.real = real

        # rhs of linear system
        self.b = ti.field(dtype=real) # Ax=b

        # grid type
        self.grid_type = [ti.field(dtype=ti.i32) 
                        for _ in range(self.n_mg_levels)]
        self.boundary = [ti.field(dtype=ti.i32) 
                        for _ in range(self.n_mg_levels)]

        # lhs of linear system and its corresponding form in coarse grids
        self.Adiag = [ti.field(dtype = real)
                        for _ in range(self.n_mg_levels)] # A(i,j,k)(i,j,k)
        self.Ax = [ti.Vector.field(dim, dtype = real)
                        for _ in range(self.n_mg_levels)] # Ax=A(i,j,k)(i+1,j,k), Ay=A(i,j,k)(i,j+1,k), Az=A(i,j,k)(i,j,k+1)
        
        # setup sparse simulation data arrays
        self.r = [ti.field(dtype = real)
                        for _ in range(self.n_mg_levels)]       # residual
        self.z = [ti.field(dtype = real)
                        for _ in range(self.n_mg_levels)]       # M^-1 self.r
        self.delta = [ti.field(dtype = real)
                        for _ in range(self.n_mg_levels)]       # local storage for Damped-Jacobi

        self.x = ti.field(dtype=real) # solution
        self.p = ti.field(dtype=real) # conjugate gradient
        self.Ap = ti.field(dtype=real) # matrix-vector product
        self.sum = ti.field(dtype=real) # storage for reductions
        self.alpha = ti.field(dtype=real) # step size
        self.beta = ti.field(dtype=real) # step size

        indices = ti.ijk if self.dim == 3 else ti.ij
        self.grid = ti.root.pointer(indices, [res[_] // 4 for _ in range(dim)]).dense(
            indices, 4).place(self.b, self.x, self.p, self.Ap)

        for l in range(self.n_mg_levels):
            self.grid = ti.root.pointer(indices,
                                        [res[_] // (4 * 2**l) for _ in range(dim)]).dense(
                                            indices,
                                            4).place(self.grid_type[l], self.boundary[l],
                                                    self.Adiag[l], self.Ax[l], self.r[l], self.z[l], self.delta[l])

        ti.root.place(self.alpha, self.beta, self.sum)

    @ti.kernel
    def init_gridtype(self, grid0 : ti.template(), grid : ti.template(), boundary : ti.template()):
        for I in ti.grouped(grid):
            I2 = I * 2

            tot_fluid = 0
            tot_air = 0
            for offset in ti.static(ti.grouped(ti.ndrange(*((0, 2), ) * self.dim))):
                attr = int(grid0[I2 + offset])
                if attr == ti.cast(self.AIR, ti.i32): tot_air += 1
                elif attr == ti.cast(self.FLUID, ti.i32): tot_fluid += 1

            if tot_air > 0: grid[I] = ti.cast(self.AIR, ti.i32)
            elif tot_fluid > 0: grid[I] = ti.cast(self.FLUID, ti.i32)
            else: grid[I] = ti.cast(self.SOLID, ti.i32)

    @ti.kernel
    def init_boundary(self, grid : ti.template(), boundary : ti.template()):
        for I in ti.grouped(boundary):
            if grid[I] == ti.cast(self.FLUID, ti.i32):
                boundary[I] = 0
                I0 = (I - 1) // 2
                for offset in ti.static(ti.grouped(ti.ndrange(*((0, 2), ) * self.dim))):
                    I2 = (I0 + offset) * 2
                    non_fluid = 0
                    for k in ti.static(ti.grouped(ti.ndrange(*((0, 2), ) * self.dim))):
                        non_fluid |= (grid[I2 + k] != ti.cast(self.FLUID, ti.i32))
                    boundary[I] |= non_fluid
    
    @ti.kernel
    def initialize(self):
        for I in ti.grouped(ti.ndrange(* [self.res[_] for _ in range(self.dim)])):
            self.r[0][I] = 0
            self.z[0][I] = 0
            self.Ap[I] = 0
            self.p[I] = 0
            self.x[I] = 0
            self.b[I] = 0

        for l in ti.static(range(self.n_mg_levels)):
            for I in ti.grouped(ti.ndrange(* [self.res[_] // (2**l) for _ in range(self.dim)])):
                self.grid_type[l][I] = 0
                self.boundary[l][I] = 0
                self.Adiag[l][I] = 0
                self.Ax[l][I] = ti.zero(self.Ax[l][I])

    def reinitialize(self, cell_type, strategy):
        self.initialize()
        self.grid_type[0].copy_from(cell_type)
        strategy.build_rhs(self)
        strategy.build_lhs(self, 0)

        for l in range(1, self.n_mg_levels):
            self.init_boundary(self.grid_type[l - 1], self.boundary[l - 1])
            self.init_gridtype(self.grid_type[l - 1], self.grid_type[l], self.boundary[l])
            strategy.build_lhs(self, l)

    @ti.func
    def neighbor_sum(self, Ax, x, I):
        ret = ti.cast(0.0, self.real)
        for i in ti.static(range(self.dim)):
            offset = ti.Vector.unit(self.dim, i)
            ret += Ax[I - offset][i] * x[I - offset] + Ax[I][i] * x[I + offset]
        return ret

    @ti.kernel
    def boundary_smooth(self, l: ti.template(), phase: ti.template()):
        # phase = red/black Gauss-Seidel phase
        for I in ti.grouped(self.r[l]):
            if (I.sum()) & 1 == phase and self.grid_type[l][I] == ti.cast(self.FLUID, ti.i32):
                self.z[l][I] = (self.r[l][I] - self.neighbor_sum(self.Ax[l], self.z[l], I)) / self.Adiag[l][I]

    @ti.kernel
    def damped_jacobi(self, l: ti.template()):
        # Damped Jacobi (w = 2/3)
        for I in ti.grouped(self.r[l]):
            if self.grid_type[l][I] == ti.cast(self.FLUID, ti.i32):
                self.delta[l][I] = ti.cast(2 / 3, self.real) * (self.r[l][I] - self.neighbor_sum(self.Ax[l], self.z[l], I)) / self.Adiag[l][I]

    def smooth(self, l):
        self.damped_jacobi(l)
        self.z[l].copy_from(self.delta[l])

    @ti.func
    def B_w(self, x):
        val = ti.cast(1 / 8, self.real)
        if x == 0 or x == 1: val = ti.cast(3 / 8, self.real)
        return val
        
    @ti.kernel
    def restrict(self, l: ti.template()):
        for I in ti.grouped(self.r[l]):
            if self.grid_type[l][I] == ti.cast(self.FLUID, ti.i32):
                Az = self.Adiag[l][I] * self.z[l][I]
                Az += self.neighbor_sum(self.Ax[l], self.z[l], I)
                res = self.r[l][I] - Az # instant calculate residual to avoid memory-bandwidth

                # (Bu^h)(x) = 1/8u^h(x-3/2h)+3/8u^h(x-h/2)+3/8u^h(x+h/2)+1/8u^h(x+3/2h)=u^2h(x)
                # R = B @ B @ B
                I0 = (I - 1) // 2
                for offset in ti.static(ti.grouped(ti.ndrange(*((0, 2), ) * self.dim))):
                    w = ti.cast(1.0, self.real)
                    for k in ti.static(range(self.dim)):
                        w *= ((I[k] + offset[k]) & 1) / 4 + 1 / 8
                    self.r[l + 1][I0 + offset] += w * res


    @ti.kernel
    def prolongate(self, l: ti.template()):
        for I in ti.grouped(self.z[l + 1]):
            if self.grid_type[l + 1][I] == ti.cast(self.FLUID, ti.i32):      
                # (Bu^h)(x) = 1/8u^h(x-3/2h)+3/8u^h(x-h/2)+3/8u^h(x+h/2)+1/8u^h(x+3/2h)=u^2h(x)
                # P^T = 8B @ B @ B
                I2 = I * 2
                for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3), ) * self.dim))):
                    w = ti.cast(8.0, self.real)
                    for k in ti.static(range(self.dim)): 
                        w *= self.B_w(offset[k])
                    self.z[l][I2 + offset] += w * self.z[l + 1][I]


    def v_cycle(self):
        self.z[0].fill(0.0)
        for l in range(self.n_mg_levels - 1):
            self.smooth(l)
            # perform boundary smoothing after the interior sweep on the downstroke
            for i in range(self.pre_and_post_smoothing << l):
                self.boundary_smooth(l, 0)
                self.boundary_smooth(l, 1)


            self.r[l + 1].fill(0.0)
            self.z[l + 1].fill(0.0)
            self.restrict(l)

        # solve Az = r on the coarse grid
        for i in range(self.bottom_smoothing):
            self.smooth(l)

        for l in reversed(range(self.n_mg_levels - 1)):
            self.prolongate(l)
            # before the interior sweep on the upstroke 
            for i in range(self.pre_and_post_smoothing << l):
                self.boundary_smooth(l, 1)
                self.boundary_smooth(l, 0)
            self.smooth(l)

    def solve(self,
              max_iters=-1,
              verbose=False,
              eps=1e-12,
              abs_tol=1e-12,
              rel_tol=1e-12):

        self.r[0].copy_from(self.b)
        self.reduce(self.r[0], self.r[0])
        initial_rTr = self.sum[None]

        tol = min(abs_tol, initial_rTr * rel_tol)

        # self.r = b - Ax = b    since self.x = 0
        # self.p = self.r = self.r + 0 self.p
        self.v_cycle()

        self.update_p()

        self.reduce(self.z[0], self.r[0])
        old_zTr = self.sum[None]

        # Conjugate gradients
        iter = 0
        while max_iters == -1 or iter < max_iters:
            # self.alpha = rTr / pTAp
            self.compute_Ap()
            self.reduce(self.p, self.Ap)
            pAp = self.sum[None]
            self.alpha[None] = old_zTr / (pAp + eps)

            # self.x = self.x + self.alpha self.p
            self.update_x()

            # self.r = self.r - self.alpha self.Ap
            self.update_r()

            # check for convergence
            self.reduce(self.r[0], self.r[0])
            rTr = self.sum[None]

            if verbose:
                print(f'iter {iter}, |residual|_2={ti.sqrt(rTr)}')

            if rTr < tol:
                break

            # self.z = M^-1 self.r
            self.v_cycle()

            # self.beta = new_rTr / old_rTr
            self.reduce(self.z[0], self.r[0])
            new_zTr = self.sum[None]
            self.beta[None] = new_zTr / (old_zTr + eps)

            # self.p = self.z + self.beta self.p
            self.update_p()
            old_zTr = new_zTr

            iter += 1

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template()):
        self.sum[None] = 0
        for I in ti.grouped(p):
            if self.grid_type[0][I] == ti.cast(self.FLUID, ti.i32):
                self.sum[None] += p[I] * q[I]

    @ti.kernel
    def compute_Ap(self):
        for I in ti.grouped(self.Ap):
            if self.grid_type[0][I] == ti.cast(self.FLUID, ti.i32):
                self.Ap[I] = self.Adiag[0][I] * self.p[I]
                for k in ti.static(range(self.dim)):
                    offset = ti.Vector.unit(self.dim, k)
                    self.Ap[I] += self.Ax[0][I - offset][k] * self.p[I - offset]
                    self.Ap[I] += self.Ax[0][I][k] * self.p[I + offset]

    @ti.kernel
    def update_x(self):
        for I in ti.grouped(self.p):
            if self.grid_type[0][I] == ti.cast(self.FLUID, ti.i32):
                self.x[I] += self.alpha[None] * self.p[I]

    @ti.kernel
    def update_r(self):
        for I in ti.grouped(self.p):
            if self.grid_type[0][I] == ti.cast(self.FLUID, ti.i32):
                self.r[0][I] -= self.alpha[None] * self.Ap[I]

    @ti.kernel
    def update_p(self):
        for I in ti.grouped(self.p):
            if self.grid_type[0][I] == ti.cast(self.FLUID, ti.i32):
                self.p[I] = self.z[0][I] + self.beta[None] * self.p[I]
