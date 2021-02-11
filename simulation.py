import taichi as ti

import utils
from mgpcg import *
from project import *

ti.init(arch=ti.cpu, kernel_profiler=True)

@ti.data_oriented
class FluidSolver:
    def __init__(self,
        dim = 2,
        res = (256, 256),
        cell_size = 6.250e-4,
        rho = 1000.0,
        gravity = [0, -9.8],
        substeps = 1,
        real = float):
        
        self.dim = dim
        self.real = real
        self.res = res
        self.cell_size = cell_size
        
        self.rho = rho
        self.gravity = gravity
        self.substeps = substeps

        # cell_type
        self.cell_type = ti.field(dtype = ti.i32, shape = res)

        self.velocity = [ti.field(dtype=real) for _ in range(self.dim)] # MAC grid
        self.velocity_backup = [ti.field(dtype=real) for _ in range(self.dim)]

        # x/v for marker particles
        self.total_particles = ti.field(dtype=ti.i32, shape())
        self.particle_position = ti.Vector.field(dim, dtype=real)
        self.particle_velocity = ti.Vector.field(dim, dtype=real)

        indices = ti.ijk if self.dim == 3 else ti.ij
        ti.root.dense(ti.i, 
        for d in range(self.dim):
            ti.root.dense(indices, [res[_] + (d == _) for _ in range(self.dim)]).place(self.velocity[d], self.velocity_backup[d])
            

if __name__ == '__main__':
    solver = FluidSolver()
    
    @ti.kernel
    def xxx():
        value = 0
        for I in ti.grouped(solver.velocity[0]):
            value += 1
        print(value)

    xxx()
    
