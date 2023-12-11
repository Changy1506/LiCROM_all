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
from NetMap import *

from SamplePoint import *


from taichi._lib import core as _ti_core

import taichi as ti

wp.init()

mu = 500000
lamb = 500000
dt = 1. / 200.
gravity = wp.vec3(0, -0.0, 0)
test_number = 0



mesh_file_step0 = 'data/Fracture/original_10k.mesh'
mesh_file_step1 = 'data/Fracture/fracture_0_1_20k.mesh'
mesh_file_step2 = 'data/Fracture/fracture_0_2_20k.mesh'
mesh_file_step3 = 'data/Fracture/fracture_0_3_20k.mesh'
mesh_file_step4 = 'data/Fracture/fracture_0_4_20k.mesh'
mesh_file_step5 = 'data/Fracture/fractured_20k.mesh'

output_dir_root = 'output/Fracture/'

    
mesh_load = MESHLoader(mesh_file_step0, 'cpu')
x = mesh_load.x
tets = mesh_load.tets
faces = mesh_load.faces
mesh_file_name = mesh_load.mesh_file_name


mesh_load_step1 = MESHLoader(mesh_file_step1, 'cpu')
x_step1 = mesh_load_step1.x
tets_step1 = mesh_load_step1.tets
faces_step1 = mesh_load_step1.faces
mesh_file_name_step1 = mesh_load_step1.mesh_file_name


mesh_load_step2 = MESHLoader(mesh_file_step2, 'cpu')
x_step2 = mesh_load_step2.x
tets_step2 = mesh_load_step2.tets
faces_step2 = mesh_load_step2.faces
mesh_file_name_step2 = mesh_load_step2.mesh_file_name

mesh_load_step3 = MESHLoader(mesh_file_step3, 'cpu')
x_step3 = mesh_load_step3.x
tets_step3 = mesh_load_step3.tets
faces_step3 = mesh_load_step3.faces
mesh_file_name_step3 = mesh_load_step3.mesh_file_name

mesh_load_step4 = MESHLoader(mesh_file_step4, 'cpu')
x_step4 = mesh_load_step4.x
tets_step4 = mesh_load_step4.tets
faces_step4 = mesh_load_step4.faces
mesh_file_name_step4 = mesh_load_step4.mesh_file_name

mesh_load_step5 = MESHLoader(mesh_file_step5, 'cpu')
x_step5 = mesh_load_step5.x
tets_step5 = mesh_load_step5.tets
faces_step5 = mesh_load_step5.faces
mesh_file_name_step5 = mesh_load_step5.mesh_file_name

width = 0.25




me = 'data/Fracture/epoch=3749-step=307499_enc_cpu.pt'
md = 'data/Fracture/epoch=3749-step=307499_dec_cpu.pt'
mdg = 'data/Fracture/epoch=3749-step=307499_dec_func_grad_cpu.pt'
mm = 'data/Fracture/epoch=3749-step=307499_map_cpu.pt'


net_enc_jit_load = torch.jit.load(me)
net_dec_jit_load = torch.jit.load(md)
net_dec_grad_jit_load = torch.jit.load(mdg)
net_map_jit_load = torch.jit.load(mm)


encoder = Encoder(net_enc_jit_load)
decoder = Decoder(net_dec_jit_load, md, net_dec_grad_jit_load)

net_map = NetMap(net_map_jit_load)


ti.init(arch=ti.cpu, dynamic_index=False)

output_dir = os.path.join(output_dir_root, 'test_tension0{test_number}_wid-{width}_'+mesh_file_name,'sim_seq_data_mu={mu}_lamb={lamb}').format(test_number=test_number, width=width, mu=mu, lamb=lamb)
output = os.path.join(output_dir,"h5_f_{:010d}.h5")




mouse_pos = []



integrator = IntegratorVariationalImplicit(x, tets, gravity, dt, 1000, None, mu, lamb, output, 1000, faces, mesh_load.masses, 8, encoder, decoder, None, net_map,  beta_damping=0, test_number=-1, warm_start=False, write_every=1)


integrator_step1 = IntegratorVariationalImplicit(x_step1, tets_step1, gravity, dt, 1000, None, mu, lamb, output, 1000, faces_step1, mesh_load_step1.masses, 8, encoder, decoder, None, net_map, beta_damping=0, test_number=-1, warm_start=False, write_every=1)

integrator_step2 = IntegratorVariationalImplicit(x_step2, tets_step2, gravity, dt, 1000, None, mu, lamb, output, 1000, faces_step2, mesh_load_step2.masses, 8, encoder, decoder, None, net_map, beta_damping=0, test_number=-1, warm_start=False, write_every=1)

integrator_step3 = IntegratorVariationalImplicit(x_step3, tets_step3, gravity, dt, 1000, None, mu, lamb, output, 1000, faces_step3, mesh_load_step3.masses, 8, encoder, decoder, None, net_map, beta_damping=0, test_number=-1, warm_start=False, write_every=1)

integrator_step4 = IntegratorVariationalImplicit(x_step4, tets_step4, gravity, dt, 1000, None, mu, lamb, output, 1000, faces_step4, mesh_load_step4.masses, 8, encoder, decoder, None, net_map, beta_damping=0, test_number=-1, warm_start=False, write_every=1)

integrator_step5 = IntegratorVariationalImplicit(x_step5, tets_step5, gravity, dt, 1000, None, mu, lamb, output, 1000, faces_step5, mesh_load_step5.masses, 8, encoder, decoder, None, net_map, beta_damping=0, test_number=-1, warm_start=False, write_every=1)
#integrator.convert_to_taichi()



sample_point = SamplePoint(integrator)
sample_point.initIndicesRandom(300, 100)


sample_point_step1 = SamplePoint(integrator_step1)
sample_point_step1.initIndicesRandom(300, 100)
   


sample_point_step2 = SamplePoint(integrator_step2)
sample_point_step2.initIndicesRandom(300, 100)

sample_point_step3 = SamplePoint(integrator_step3)
sample_point_step3.initIndicesRandom(300, 100)

sample_point_step4 = SamplePoint(integrator_step4)
sample_point_step4.initIndicesRandom(300, 100)

sample_point_step5 = SamplePoint(integrator_step5)
sample_point_step5.initIndicesRandom(300, 100)


integrator_step1.integrate()
integrator_step2.integrate()
integrator_step3.integrate()
integrator_step4.integrate()
integrator_step5.integrate()



is_dragging_last = False

drag_x = 0.0
drag_y = 0.0
drag_z = 0.0

last_x = 0.0
last_y = 0.0
last_z = 0.0



idx = 0        

top = 0.75

@ti.kernel
def apply_drag(particle_v: ti.template()):

    for tid in particle_v:

        
        particle_v[tid][1] = particle_v[tid][1] - 9.8 / 200.
        
@ti.kernel
def apply_boundary_bottom(x_0: ti.template(),
                 x_pred: ti.template(),
                 x_pred_out: ti.template(),
                 v_pred_out: ti.template(),
                 weight: ti.template(),
                 length: int):

    for tid in range(length):
        if x_0[tid][0] < -0.45 or x_0[tid][0] > 0.45 or x_0[tid][2] < -0.45 or x_0[tid][2] > 0.45 :  
            weight[tid] = 1.
            v_pred_out[tid] = - x_pred_out[tid] * 20.
            x_pred_out[tid] = x_pred_out[tid] - x_pred_out[tid]
        else:
            x_pred_out[tid] = x_pred[tid]
            
def callback():
    global is_dragging_last, drag_x, drag_y, drag_z, last_x, last_y, last_z, idx, top
    global integrator, integrator_step1, integrator_step2, integrator_step3, integrator_step4
    
    #if(integrator.timestep <= 150):
    #   top -= 0.01    
    
   
    
    integrator.update_vel()
    integrator.weight.fill(1)
    
    apply_drag(integrator.Vel_cur)
    apply_boundary_bottom(integrator.V0_sample, integrator.V_cur, integrator.V_cur, integrator.Vel_cur, integrator.weight, integrator.sum_selected)
    #poke_3
    
    if (integrator.timestep == 250): #remesh
          integrator_step1.vhat_now = integrator.vhat_now
          integrator_step1.xhat = integrator.xhat
          integrator_step1.timestep = integrator.timestep
          integrator_step1.update_vel_resample()
          integrator = integrator_step1
    if (integrator.timestep == 500): #remesh
          integrator_step2.vhat_now = integrator.vhat_now
          integrator_step2.xhat = integrator.xhat
          integrator_step2.timestep = integrator.timestep
          integrator_step2.update_vel_resample()
          integrator = integrator_step2
    if (integrator.timestep == 750): #remesh
          integrator_step3.vhat_now = integrator.vhat_now
          integrator_step3.xhat = integrator.xhat
          integrator_step3.timestep = integrator.timestep
          integrator_step3.update_vel_resample()
          integrator = integrator_step3
    if (integrator.timestep == 1000): #remesh
          integrator_step4.vhat_now = integrator.vhat_now
          integrator_step4.xhat = integrator.xhat
          integrator_step4.timestep = integrator.timestep
          integrator_step4.update_vel_resample()
          integrator = integrator_step4
    if (integrator.timestep == 1250): #remesh
          integrator_step5.vhat_now = integrator.vhat_now
          integrator_step5.xhat = integrator.xhat
          integrator_step5.timestep = integrator.timestep
          integrator_step5.update_vel_resample()
          integrator = integrator_step5      
                
    
    
    
    
                            
    integrator.integrate()
    print("integrator vel cur: ", integrator.Vel_cur.shape)
   
    
    if(integrator.timestep % 5 == 0):
        pos_numpy = integrator.V.to_torch()[:integrator.V0_surface.shape[0],:].numpy()
        ps_ourmesh = ps.register_surface_mesh("my mesh", pos_numpy, integrator.tri_new, edge_width = 0.0, color = (1,1,0))
        
        
        

    
if __name__ == '__main__':
    
       
    ps.init()
    ps.set_user_callback(callback)
    ps.look_at((-2.,1.5,-2), (0,0,0))
    ps.set_ground_plane_mode('none')
    ps.show()
    ps.clear_user_callback()

