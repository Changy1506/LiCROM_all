import os
import math

import warp as wp
import warp.sim
#import warp.sim.render

import numpy as np

import polyscope as ps
import polyscope.imgui

import json


from MESHLoader import *
from IntegratorVariationalImplicit import *



from Encoder import *
from Decoder import *


from taichi._lib import core as _ti_core

import taichi as ti
import math
wp.init()

mu = 500000
lamb = 500000
dt = 1. / 200.
gravity = wp.vec3(0, -0.0, 0)
test_number = 0




mesh_file = 'data/Holes/cheese6_200.mesh'
output_dir_root = 'output/Holes/'

mesh_load = MESHLoader(mesh_file, 'cpu')
x = mesh_load.x
tets = mesh_load.tets
faces = mesh_load.faces
mesh_file_name = mesh_load.mesh_file_name

width = 0.25



ti.init(arch=ti.cpu, dynamic_index=False)

output_dir = os.path.join(output_dir_root, 'test_tension0{test_number}_wid-{width}_'+mesh_file_name,'sim_seq_data_mu={mu}_lamb={lamb}').format(test_number=test_number, width=width, mu=mu, lamb=lamb)
output = os.path.join(output_dir,"h5_f_{:010d}.h5")




mouse_pos = []

integrator = IntegratorVariationalImplicit(x, tets, gravity, dt, 1000, None, mu, lamb, output, 1000, faces, mesh_load.masses, 0.0002, beta_damping=0, test_number=-1, warm_start=False, write_every=1)


is_dragging_last = False

drag_x = 0.0
drag_y = 0.0
drag_z = 0.0

last_x = 0.0
last_y = 0.0
last_z = 0.0

@ti.kernel
def apply_drag(particle_v: ti.template()):

    for tid in particle_v:

        
        particle_v[tid][1] = particle_v[tid][1] - 9.8 / 200. * 0.3
        
               

           

idx = 0  
center_x = -0.1
center_y = -0.1
center_z = -0.1     

def callback():
    global is_dragging_last, last_x, last_y, last_z, idx
    global center_x, center_y, center_z
    
    if(integrator.timestep > 700):
         quit()
    #     center_x = center_x + 0.005
         
    #print()
    integrator.top = 0.
    
    apply_drag(integrator.Vel_cur)
   
    integrator.integrate()
    pos_numpy = integrator.V_cur.to_numpy()
    #ps.register_point_cloud("head points", pos_numpy, radius=0.0125)
    ps_ourmesh = ps.register_surface_mesh("my mesh", pos_numpy, integrator.faces - 1, smooth_shade=False)
    
    
    

    
if __name__ == '__main__':
    
       
    ps.init()
    ps.set_user_callback(callback)
    ps.look_at((-2.,1.5,-2), (0,0.5,0))
    ps.set_ground_plane_mode('none')
    ps.show()
    ps.clear_user_callback()

