import taichi as ti
from CGSolver import CGSolver
from MICPCGSolver import MICPCGSolver
from MGPCGSolver_ import MGPCGSolver
from mgpcg import *
from project import *
import numpy as np
from utils import ColorMap, vec2, vec3, clamp
import utils
import random
import time

ti.init(arch=ti.cpu, kernel_profiler=True)

# params in simulation
cell_res = 128
npar = 2

m = cell_res
n = cell_res
w = 10
h = 10 * n / m
grid_x = w / m
grid_y = h / n
pspace_x = grid_x / npar
pspace_y = grid_y / npar

rho = 1000
g = -9.8
substeps = 1

# algorithm = 'FLIP/PIC'
algorithm = 'Euler'
# algorithm = 'APIC'
FLIP_blending = 0.0

wall_low = ti.Vector.field(2, dtype=ti.f32, shape=1000)
wall_upper = ti.Vector.field(2, dtype=ti.f32, shape=1000)
total_wall = ti.field(dtype=ti.i32, shape=())

# params in render
screen_res = (512, 512 * n // m)
bwrR = ColorMap(1.0, .25, 1, .5)
bwrG = ColorMap(1.0, .5, .5, .5)
bwrB = ColorMap(1.0, 1, .25, .5)
color_buffer = ti.Vector.field(3, dtype=ti.f32, shape=screen_res)
gui = ti.GUI("watersim2D", screen_res)

# cell type
cell_type = ti.field(dtype=ti.i32, shape=(m, n))

# velocity field
u = ti.field(dtype=ti.f32, shape=(m + 1, n))
v = ti.field(dtype=ti.f32, shape=(m, n + 1))
vec = [ti.field(dtype=ti.f32, shape=(m + 1, n)), ti.field(dtype=ti.f32, shape=(m, n + 1))]
u_temp = ti.field(dtype=ti.f32, shape=(m + 1, n))
v_temp = ti.field(dtype=ti.f32, shape=(m, n + 1))

# pressure field
p = ti.field(dtype=ti.f32, shape=(m, n))

#pressure solver
preconditioning = 'MG'

MIC_blending = 0.97

mg_level = 4
pre_and_post_smoothing = 2
bottom_smoothing = 50

if preconditioning == None:
    solver = CGSolver(m, n, u, v, cell_type)
elif preconditioning == 'MIC':
    solver = MICPCGSolver(m, n, u, v, cell_type, MIC_blending=MIC_blending)
elif preconditioning == 'MG':
    solver = MGPCGSolver(m,
                         n,
                         u,
                         v,
                         cell_type,
                         multigrid_level=mg_level,
                         pre_and_post_smoothing=pre_and_post_smoothing,
                         bottom_smoothing=bottom_smoothing)
    solver1 = MGPCGPoissonSolver(2, (m, n), mg_level, pre_and_post_smoothing, bottom_smoothing)

# particle x and v
particle_positions = ti.Vector.field(2, dtype=ti.f32, shape=(m, n, npar * npar))
particle_velocities = ti.Vector.field(2,
                                      dtype=ti.f32,
                                      shape=(m, n, npar * npar))
# particle type
particle_type = ti.field(dtype=ti.f32, shape=(m, n, npar * npar))
P_FLUID = 1
P_OTHER = 0

# extrap utils
valid = ti.field(dtype=ti.i32, shape=(m + 1, n + 1))
valid_temp = ti.field(dtype=ti.i32, shape=(m + 1, n + 1))

# save to gif
result_dir = "./results"
video_manager = ti.VideoManager(output_dir=result_dir,
                                framerate=24,
                                automatic_build=False)


def render():
    render_type = 'particles'

    @ti.func
    def map_color(c):
        return vec3(bwrR.map(c), bwrG.map(c), bwrB.map(c))

    @ti.kernel
    def fill_marker():
        for i, j in color_buffer:
            x = int((i + 0.5) / screen_res[0] * w / grid_x)
            y = int((j + 0.5) / screen_res[1] * h / grid_y)

            m = cell_type[x, y]

            if m == utils.SOLID:
                color_buffer[i, j] = vec3(1.0, 0, 0)
            else:
                color_buffer[i, j] = map_color(m * 0.5)

            # if solver1.boundary[1][x, y] == 1:
            #    color_buffer[i, j] = vec3(0, 1.0, 0)

    def render_pixels():
        fill_marker()
        img = color_buffer.to_numpy()
        gui.set_image(img)
        gui.show()

    def render_particles():
        bg_color = 0x112f41
        particle_color = 0x068587
        particle_radius = 1.0

        pf = particle_type.to_numpy()
        np_type = pf.copy()
        np_type = np.reshape(np_type, -1)

        pos = particle_positions.to_numpy()
        np_pos = pos.copy()
        np_pos = np.reshape(pos, (-1, 2))
        np_pos = np_pos[np.where(np_type == P_FLUID)]

        for i in range(np_pos.shape[0]):
            np_pos[i][0] /= w
            np_pos[i][1] /= h

        gui.clear(bg_color)
        gui.circles(np_pos, radius=particle_radius, color=particle_color)
        gui.show()
    
    if render_type == 'particles':
        render_particles()
    else:
        render_pixels()

    # video_manager.write_frame(gui.get_image())

def init():
    # init scene
    @ti.kernel
    def init_dambreak(x: ti.f32, y: ti.f32, x1: ti.f32, x2: ti.f32, y1:ti.f32, y2: ti.f32):
        xn = int(x / grid_x)
        yn = int(y / grid_y)
        total_wall[None] = 1
        wall_low[0] = [x1, y1]
        wall_upper[0] = [x2, y2]
        x1_ = int(x1 / grid_x)
        x2_ = int(x2 / grid_x)
        y1_ = int(y1 / grid_y)
        y2_ = int(y2 / grid_y)

        for i, j in cell_type:
            if i == 0 or i == m - 1 or j == 0 or j == n - 1:
                cell_type[i, j] = utils.SOLID  # boundary
            else:
                if i <= xn and j <= yn:
                    cell_type[i, j] = utils.FLUID
                # elif i >= x1_ and i <= x2_ and j >= y1_ and j <= y2_:
                #    cell_type[i, j] = utils.SOLID
                else:
                    cell_type[i, j] = utils.AIR

    @ti.kernel
    def init_spherefall(xc: ti.f32, yc: ti.f32, r: ti.f32):
        for i, j in cell_type:
            if i == 0 or i == m - 1 or j == 0 or j == n - 1:
                cell_type[i, j] = utils.SOLID  # boundary
            else:
                x = (i + 0.5) * grid_x
                y = (j + 0.5) * grid_y

                phi = (x - xc)**2 + (y - yc) ** 2 - r**2

                if phi <= 0 :
                    cell_type[i, j] = utils.FLUID
                else:
                    cell_type[i, j] = utils.AIR

    #init simulation
    @ti.kernel
    def init_field():
        for i, j in u:
            u[i, j] = 0.0
            
        for i, j in v:
            v[i, j] = 0.0
           
        for i, j in p:
            p[i, j] = 0.0

    @ti.kernel
    def init_particles():
        for i, j, k in particle_positions:
            if cell_type[i, j] == utils.FLUID:
                particle_type[i, j, k] = P_FLUID
                ix = k % 2
                jx = k // 2
                px = i * grid_x + (ix + random.random()) * pspace_x
                py = j * grid_y + (jx + random.random()) * pspace_y

                particle_positions[i, j, k] = vec2(px, py)
                particle_velocities[i, j, k] = vec2(0.0, 0.0)
    
    init_dambreak(4, 4, 4.4, 6, 1, 5)
    # init_spherefall(5,3,2)
    init_field()
    init_particles()


# -------------- Helper Functions -------------------
@ti.func
def is_valid(i, j):
    return i >= 0 and i < m and j >= 0 and j < n


@ti.func
def is_fluid(i, j):
    return is_valid(i, j) and cell_type[i, j] == utils.FLUID


@ti.func
def is_solid(i, j):
    return is_valid(i, j) and cell_type[i, j] == utils.SOLID


@ti.func
def is_air(i, j):
    return is_valid(i, j) and cell_type[i, j] == utils.AIR


@ti.func
def pos_to_stagger_idx(pos, stagger):
    pos[0] = clamp(pos[0], stagger[0] * grid_x,
                   w - 1e-4 - grid_x + stagger[0] * grid_x)
    pos[1] = clamp(pos[1], stagger[1] * grid_y,
                   h - 1e-4 - grid_y + stagger[1] * grid_y)
    p_grid = pos / vec2(grid_x, grid_y) - stagger
    I = ti.cast(ti.floor(p_grid), ti.i32)

    return I, p_grid


@ti.func
def sample_bilinear(x, source_pos, stagger):
    I, p_grid = pos_to_stagger_idx(source_pos, stagger)
    f = p_grid - I
    g = 1 - f

    return x[I] * (g[0] * g[1]) + x[I + vec2(1, 0)] * (f[0] * g[1]) + x[
        I + vec2(0, 1)] * (g[0] * f[1]) + x[I + vec2(1, 1)] * (f[0] * f[1])


@ti.func
def sample_velocity(pos, u, v):
    u_p = sample_bilinear(u, pos, vec2(0, 0.5))
    v_p = sample_bilinear(v, pos, vec2(0.5, 0))

    return vec2(u_p, v_p)


# -------------- Simulation Steps -------------------
@ti.kernel
def apply_gravity(dt: ti.f32):
    for i, j in v:
        v[i, j] += g * dt


@ti.kernel
def enforce_boundary():
    # u solid
    for i, j in u:
        if is_solid(i - 1, j) or is_solid(i, j):
            u[i, j] = 0.0

    # v solid
    for i, j in v:
        if is_solid(i, j - 1) or is_solid(i, j):
            v[i, j] = 0.0


def extrapolate_velocity():
    # reference: https://gitee.com/citadel2020/taichi_demos/blob/master/mgpcgflip/mgpcgflip.py
    @ti.kernel
    def mark_valid_u():
        for i, j in u:
            # NOTE that the the air-liquid interface is valid
            if is_fluid(i - 1, j) or is_fluid(i, j):
                valid[i, j] = 1
            else:
                valid[i, j] = 0

    @ti.kernel
    def mark_valid_v():
        for i, j in v:
            # NOTE that the the air-liquid interface is valid
            if is_fluid(i, j - 1) or is_fluid(i, j):
                valid[i, j] = 1
            else:
                valid[i, j] = 0

    @ti.kernel
    def diffuse_quantity(dst: ti.template(), src: ti.template(),
                         valid_dst: ti.template(), valid: ti.template()):
        for i, j in dst:
            if valid[i, j] == 0:
                s = 0.0
                count = 0
                for m, n in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                    if 1 == valid[i + m, j + n]:
                        s += src[i + m, j + n]
                        count += 1
                if count > 0:
                    dst[i, j] = s / float(count)
                    valid_dst[i, j] = 1

    mark_valid_u()
    for i in range(10):
        u_temp.copy_from(u)
        valid_temp.copy_from(valid)
        diffuse_quantity(u, u_temp, valid, valid_temp)

    mark_valid_v()
    for i in range(10):
        v_temp.copy_from(v)
        valid_temp.copy_from(valid)
        diffuse_quantity(v, v_temp, valid, valid_temp)

# ycc
strategy = PressureProjectStrategy(0, 0, vec)
def solve_pressure(dt):
    scale_A = dt / (rho * grid_x * grid_x)
    scale_b = 1 / grid_x
    strategy.scale_A = scale_A
    strategy.scale_b = scale_b

    vec[0].copy_from(u)
    vec[1].copy_from(v)

    start1 = time.perf_counter() 
    solver1.reinitialize(cell_type, strategy)
    end1 = time.perf_counter()
    # solver.system_init(scale_A, scale_b)

    # solver.solve(50)
    render()


    start2 = time.perf_counter()
    solver1.solve(50, True)
    end2 = time.perf_counter()

    print(f'\033[33minit cost {end1 - start1}s, solve cost {end2 - start2}s\033[0m')

    p.copy_from(solver1.x)


@ti.kernel
def apply_pressure(dt: ti.f32):
    scale = dt / (rho * grid_x)

    for i, j in ti.ndrange(m, n):
        if is_fluid(i - 1, j) or is_fluid(i, j):
            if is_solid(i - 1, j) or is_solid(i, j):
                u[i, j] = 0
            else:
                u[i, j] -= scale * (p[i, j] - p[i - 1, j])

        if is_fluid(i, j - 1) or is_fluid(i, j):
            if is_solid(i, j - 1) or is_solid(i, j):
                v[i, j] = 0
            else:
                v[i, j] -= scale * (p[i, j] - p[i, j - 1])


@ti.kernel
def update_particle_velocities(dt: ti.f32):
    for p in ti.grouped(particle_positions):
        if particle_type[p] == P_FLUID:
            pv = sample_velocity(particle_positions[p], u, v)
            particle_velocities[p] = pv


@ti.kernel
def advect_particles(dt: ti.f32):
    for p in ti.grouped(particle_positions):
        if particle_type[p] == P_FLUID:
            pos = particle_positions[p]
            pv = particle_velocities[p]

            pos += pv * dt

            if pos[0] <= grid_x:  # left boundary
                pos[0] = grid_x
                pv[0] = 0
            if pos[0] >= w - grid_x:  # right boundary
                pos[0] = w - grid_x
                pv[0] = 0
            if pos[1] <= grid_y:  # bottom boundary
                pos[1] = grid_y
                pv[1] = 0
            if pos[1] >= h - grid_y:  # top boundary
                pos[1] = h - grid_y
                pv[1] = 0
            
            '''
            for k in range(total_wall[None]):
                if pos[0] >= wall_low[k][0] - grid_x and pos[0] <= wall_upper[k][0] + grid_x and \
                     pos[1] >= wall_low[k][1] - grid_y and pos[1] <= wall_upper[k][1] + grid_y:
                        if pos[0] <= grid_x:  # left boundary
                            pos[0] = grid_x
                            pv[0] = 0
                        if pos[0] >= w - grid_x:  # right boundary
                            pos[0] = w - grid_x
                            pv[0] = 0
                        if pos[1] <= grid_y:  # bottom boundary
                            pos[1] = grid_y
                            pv[1] = 0
                        if pos[1] >= h - grid_y:  # top boundary
                            pos[1] = h - grid_y
                            pv[1] = 0
            '''


            particle_positions[p] = pos
            particle_velocities[p] = pv

@ti.kernel
def mark_cell():
    for i, j in cell_type:
        if not is_solid(i, j):
            cell_type[i, j] = utils.AIR

    for p in ti.grouped(particle_positions):
        if particle_type[p] == P_FLUID:
            pos = particle_positions[p]
            idx = ti.cast(ti.floor(pos / vec2(grid_x, grid_y)), ti.i32)

            if not is_solid(idx[0], idx[1]):
                cell_type[idx] = utils.FLUID


@ti.func
def backtrace(p, dt):
    # rk2 backtrace
    p_mid = p - 0.5 * dt * sample_velocity(p, u, v)
    p -= dt * sample_velocity(p_mid, u, v)

    return p


@ti.func
def semi_Largrange(x, x_temp, stagger, dt):
    m, n = x.shape
    for i, j in ti.ndrange(m, n):
        pos = (vec2(i, j) + stagger) * vec2(grid_x, grid_y)
        source_pos = backtrace(pos, dt)
        x_temp[i, j] = sample_bilinear(x, source_pos, stagger)


@ti.kernel
def advection_kernel(dt: ti.f32):
    semi_Largrange(u, u_temp, vec2(0, 0.5), dt)
    semi_Largrange(v, v_temp, vec2(0.5, 0), dt)


def advection(dt):
    advection_kernel(dt)
    u.copy_from(u_temp)
    v.copy_from(v_temp)

def onestep(dt):
    apply_gravity(dt)
    enforce_boundary()

    extrapolate_velocity()
    enforce_boundary()

    solve_pressure(dt)
    apply_pressure(dt)
    enforce_boundary()

    extrapolate_velocity()
    enforce_boundary()

    update_particle_velocities(dt)
    advect_particles(dt)
    mark_cell()

    advection(dt)
    enforce_boundary()


def simulation(max_time, max_step):
    dt = 0.01
    t = 0
    step = 1

    while step < max_step and t < max_time: 
        for i in range(substeps):
            onestep(dt)

            pv = particle_velocities.to_numpy()
            max_vel = np.max(np.linalg.norm(pv, 2, axis=1))

            print("step = {}, substeps = {}, time = {}, dt = {}, maxv = {}".
                  format(step, i, t, dt, max_vel))

            t += dt
        
        step += 1


def main():
    init()
    t0 = time.time()
    simulation(4000, 4000)
    t1 = time.time()
    print("simulation elapsed time = {} seconds".format(t1 - t0))
    ti.kernel_profiler_print()
    # video_manager.make_video(gif=True, mp4=True)


if __name__ == "__main__":
    main()
