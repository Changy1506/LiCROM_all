import os
import math

import warp as wp
import warp.sim
import warp.torch
#import warp.sim.render

import numpy as np

import polyscope as ps
import polyscope.imgui

import torch
from SimulationData import *
from SimulationDataConfig import *
from optimizer import *
import json


from Encoder import *
from Decoder import *

from taichi._lib import core as _ti_core

import scipy.optimize

import taichi as ti


wp.init()

    


        
        



#copied from https://github.com/NVIDIA/warp/blob/main/warp/sim/integrator_euler.py

    
@ti.kernel
def integrate_particles(x: ti.template(),
                        v: ti.template(),
                        f: ti.template(),
                        w: ti.template(),
                        dt: ti.f32,
                        x_new: ti.template(),
                        v_new: ti.template(),
                        length: int):

    for tid in range(length):

        x0 = x[tid]
        v0 = v[tid]
        f0 = f[tid]

        inv_mass = w[tid]
        

        v1 = v0 
        x1 = x0 + v1 * dt

        x_new[tid] = x1
        v_new[tid] = v1



       

@ti.kernel
def eval_tetrahedra(x: ti.template(),
                    X0: ti.template(),
                    v: ti.template(),
                    indices: ti.template(),
                    #pose_torch: ti.types.ndarray(),
                    pose: ti.template(),
                    activation: ti.template(),
                    materials: ti.template(),
                    f: ti.template(),
                    len_f: int,
                    accumulate_tet: int,
                    sum_selected: int):

    for tid in range(accumulate_tet):
        #print(tid, accumulate_tet)

        i = indices[tid][0]
        j = indices[tid][1]
        k = indices[tid][2]
        l = indices[tid][3]
        
        if i >= sum_selected or j >= sum_selected or k >= sum_selected or l >= sum_selected:
           #print("error here", tid, sum_selected, i, j, k, l)
           continue
        
    
    

        act = activation[tid]
    

        k_mu = materials[tid][0]
        k_lambda = materials[tid][1]
        k_damp = materials[tid][2]
        
        k_damp = 0.001 * k_mu
        
        x0 = x[i] + X0[i]
        x1 = x[j] + X0[j]
        x2 = x[k] + X0[k]
        x3 = x[l] + X0[l]
    

        v0 = v[i]
        v1 = v[j]
        v2 = v[k]
        v3 = v[l]
        
        #print(v0, v1, v2, v3, i,j,k,l)

        x10 = x1 - x0
        x20 = x2 - x0
        x30 = x3 - x0

        v10 = v1 - v0
        v20 = v2 - v0
        v30 = v3 - v0
        
        #print(x10)
    
    

        #Ds = wp.mat33(x10, x20, x30)
        Ds = ti.Matrix([[x10[0], x20[0], x30[0]], [x10[1], x20[1], x30[1]], [x10[2], x20[2], x30[2]]]) 
        #Ds = ti.Matrix.cols([x10, x20, x30])
        Dm = pose[tid]
        #print(k_mu, k_lambda)
        
        
        
           
    

        inv_rest_volume = Dm.determinant() * 6.0
        rest_volume = 1.0 / inv_rest_volume

        alpha = 1.0 + k_mu / k_lambda - k_mu / (4.0 * k_lambda)
    

        # scale stiffness coefficients to account for area
        k_mu = k_mu * rest_volume
        k_lambda = k_lambda * rest_volume
        k_damp = k_damp * rest_volume

        # F = Xs*Xm^-1
        F = Ds @ Dm
        
        
        
        #dFdt = wp.mat33(v10, v20, v30) * Dm
        dFdt = ti.Matrix([[v10[0], v20[0], v30[0]], [v10[1], v20[1], v30[1]], [v10[2], v20[2], v30[2]]]) @ Dm
        #dFdt = ti.Matrix.cols([v10, v20, v30]) @ Dm


        col1 = ti.Vector([F[0, 0], F[1, 0], F[2, 0]])
        col2 = ti.Vector([F[0, 1], F[1, 1], F[2, 1]])
        col3 = ti.Vector([F[0, 2], F[1, 2], F[2, 2]])

         
        Ic = col1.dot(col1) + col2.dot(col2) + col3.dot(col3)

        # deviatoric part
        P = F * k_mu * (1.0 - 1.0 / (Ic + 1.0)) + dFdt * k_damp
    
        #print(P)
    
        H = P @ Dm.transpose()
    
        #print(H)

        f1 = ti.Vector([H[0, 0], H[1, 0], H[2, 0]])
        f2 = ti.Vector([H[0, 1], H[1, 1], H[2, 1]])
        f3 = ti.Vector([H[0, 2], H[1, 2], H[2, 2]])

    



        # hydrostatic part
        J = F.determinant()
    
        #print("J = ", J)

        #print(J)
        s = inv_rest_volume / 6.0
        dJdx1 = x20.cross(x30) * s
        dJdx2 = x30.cross(x10) * s
        dJdx3 = x10.cross(x20) * s

        f_volume = (J - alpha + act) * k_lambda
        f_damp = (dJdx1.dot(v1) + dJdx2.dot(v2) + dJdx3.dot(v3)) * k_damp

        f_total = f_volume + f_damp
        
        #print(f_damp, f_volume)
    

        f1 = f1 + dJdx1 * f_total
        f2 = f2 + dJdx2 * f_total
        f3 = f3 + dJdx3 * f_total
        f0 = (f1 + f2 + f3) * (0.0 - 1.0)
    
    
        # apply forces
        if True: #Ds.determinant() > 0: #.2 * (1 / Dm.determinant()):
          if(i < len_f):
             ti.atomic_sub(f[i], f0)
          if(j < len_f):
             ti.atomic_sub(f[j], f1)
          if(k < len_f):
             ti.atomic_sub(f[k], f2)
          if(l < len_f):
             ti.atomic_sub(f[l], f3)
       


def compute_forces(V_pred, V0, Vel_pred, T, tet_poses, tet_activations, tet_materials, particle_f, accumulate_tet, sum_selected):
   
   eval_tetrahedra(V_pred, V0, Vel_pred, T, tet_poses, tet_activations, tet_materials, particle_f, particle_f.shape[0], accumulate_tet, sum_selected)
   








    

@ti.kernel
def update_state_taichi(
    particle_q_0: ti.template(),
    particle_q_1: ti.template(),
    particle_qd_1: ti.template(),
    dt: float,
    length: int):

    for tid in range(length):

        qd_1 = particle_qd_1[tid]
    
        q_0 = particle_q_0[tid]
        q_1 = q_0 + qd_1*dt

        particle_q_1[tid] = q_1  
    
    


@ti.kernel
def copy_field(
    a: ti.template(),
    b: ti.template()):

    for tid in a:

        a[tid] = b[tid]
        
        
@ti.kernel
def sub_field(
    a: ti.template(),
    b: ti.template(),
    c: ti.template()):

    for tid in a:

        a[tid] = b[tid] - c[tid]
        
@ti.kernel
def update_vel(
    a: ti.template(),
    b: ti.template(),
    c: ti.template(),
    dt: float):

    for tid in a:

        a[tid] = (b[tid] - c[tid]) / dt
        
       

def update_state_by_latent(xhat_old, particle_v0, particle_q_out, particle_qd_out, res, dt, decoder, decoder2, net_map, mapped, jac_sample, weight, sum_selected_all, first_iteration = False, vhat_last = None):
     
     
     weight = weight.view(-1,1)
     
     weight = weight.expand(weight.size(0), 3)
     weight = weight.reshape(-1,1)
     #weight = weight / weight
     res_new = res.to_torch(xhat_old.device)
     jac_new = jac_sample #* weight
     res_new = res_new.view(1, -1)
     AA = torch.matmul(jac_new.transpose(1, 0), jac_new)
     BB = jac_new.transpose(1, 0).matmul(res_new.view(-1, 1))
     
     vhat = torch.linalg.solve(AA, BB)
        
     vhat = vhat.view(1,1,-1)
     
     
     
     
     xhat = xhat_old + vhat * dt
     
     if(net_map is not None):
        if(mapped is None):
          x = particle_v0.to_torch(xhat.device)[:sum_selected_all, :].view(1,-1,3)
          x_2 = x.view(-1,x.shape[-1])
          x = net_map.forward(x_2).view(x.shape[0], x.shape[1], -1)
        else:
          x = mapped.clone()
     
     if first_iteration:   
         xhat_all = xhat.expand(xhat.size(0), x.size(1), xhat.size(2))
         x = torch.cat((xhat_all, x), 2)
         x = x.view(x.size(0)*x.size(1), x.size(2))
         q = decoder.forward(x)
         float2d2vec3_k(q.view(-1, 3), particle_q_out, sum_selected_all)
     else:
        dx = (jac_sample @ vhat.view(-1,1)).view(-1,3) * dt
        float2d2vec3_dk(dx.view(-1, 3), particle_q_out, sum_selected_all)

     
     
      
        
     return vhat
            
  




@ti.kernel
def compute_particle_residual(particle_qd_0: ti.template(),
                            particle_q_0: ti.template(),
                            particle_qd_1: ti.template(),
                            particle_f: ti.template(),
                            particle_m: ti.template(),
                            dt: ti.f32,
                            residual: ti.template(),
                            sum_selected: int):

    for tid in range(sum_selected):

        m = particle_m[tid]
        v1 = particle_qd_1[tid]
        v0 = particle_qd_0[tid]
        f = particle_f[tid]
        
        
        
        
        err = (v1-v0)*m * 1. - f*dt
        
    
        residual[tid * 3] = err[0]
        residual[tid * 3 + 1] = err[1]
        residual[tid * 3 + 2] = err[2]









def compute_residual(Vel_cur, V0_sample, Vel_pred, particle_f, particle_mass, dt, residual, sum_selected):
    
    compute_particle_residual(
            Vel_cur,
            V0_sample,
            Vel_pred,
            particle_f,
            particle_mass,
            dt,
            residual,
            sum_selected)
    
        

####################
#initialization
#init tets

@ti.kernel
def init_tetrahedra(x: ti.template(),
                    indices:  ti.template(),
                    k_mu: float, 
                    k_lambda: float, 
                    k_damp: float,
                    pose:  ti.template(),
                    activation:  ti.template(),
                    materials:  ti.template()):
                  
        
     for tid in range(indices.shape[0]):

        p = x[indices[tid][0]]
        q = x[indices[tid][1]]
        r = x[indices[tid][2]]
        s = x[indices[tid][3]]

        qp = q - p
        rp = r - p
        sp = s - p

        #Dm = wp.mat33(qp, rp, sp)
        Dm = ti.Matrix([[qp[0], rp[0], sp[0]], [qp[1], rp[1], sp[1]], [qp[2], rp[2], sp[2]]]) 
        
        inv_Dm = Dm.inverse()
    
        pose[tid] = inv_Dm
        activation[tid] = 0.0
        materials[tid][0] = k_mu
        materials[tid][1] = k_lambda
        materials[tid][2] = k_damp
        


@ti.kernel
def apply_damping(
                  vel_in: ti.template(),
                  vel_out: ti.template()
                  ):
                  
        
        for tid in vel_in:
            vel_in[tid] = vel_in[tid] * 0.99
        
        
        

    
@ti.kernel
def vec32float(
    x: ti.template(),
    xx: ti.template()):

    for tid in x:

        xx[tid * 3] = x[tid][0]
        xx[tid * 3 + 1] = x[tid][1]
        xx[tid * 3 + 2] = x[tid][2]



    
@ti.kernel
def float2vec3(
    x: ti.template(),
    xx: ti.template()):

    for tid in xx:
        xx[tid][0] = x[tid * 3]
        xx[tid][1] = x[tid * 3 + 1]
        xx[tid][2] = x[tid * 3 + 2]





    
    
@ti.kernel
def float2d2vec3(
    x: ti.types.ndarray(),
    xx: ti.template()):

    for tid in xx:

        data = ti.Vector([x[tid,0], x[tid,1], x[tid,2]])
        xx[tid] = data
        
@ti.kernel
def float2d2vec3_k(
    x: ti.types.ndarray(),
    xx: ti.template(),
    length: int):

    for tid in range(length):

        data = ti.Vector([x[tid,0], x[tid,1], x[tid,2]])
        xx[tid] = data

@ti.kernel
def float2d2vec3_dk(
    x: ti.types.ndarray(),
    xx: ti.template(),
    length: int):

    for tid in range(length):

        data = ti.Vector([x[tid,0], x[tid,1], x[tid,2]])
        xx[tid] = xx[tid] + data
        
@ti.kernel        
def update_vel_pred(
    a: ti.template(),
    b: ti.types.ndarray(),
    length: int):
    
    for tid in range(length):

        data = ti.Vector([b[tid,0], b[tid,1], b[tid,2]])
        a[tid] = a[tid] + data
        
@ti.kernel        
def update_vel_pred_all(
    a: ti.template(),
    b: ti.types.ndarray(),
    length: int):
    
    for tid in range(length):

        data = ti.Vector([b[tid,0], b[tid,1], b[tid,2]])
        a[tid] = data

        
@ti.kernel
def update_vel_dampig(
    x_now: ti.types.ndarray(),
    x_last: ti.types.ndarray(),
    xx: ti.template(),
    dt: float,
    length: int):

    for tid in range(length):

        data = ti.Vector([x_now[tid,0], x_now[tid,1], x_now[tid,2]]) - ti.Vector([x_last[tid,0], x_last[tid,1], x_last[tid,2]])
        xx[tid] = data / dt
        
        
@ti.kernel
def copy_part(
    xx: ti.template(),
    x: ti.types.ndarray(),
    length: int):

    for tid in range(length):
        xx[tid] = x[tid]
        
@ti.kernel
def copy_part_2d(
    xx: ti.template(),
    x: ti.types.ndarray(),
    length: int,
    len2: int):

    for tid in range(length):
      for j in ti.static(range(3)):
        xx[tid][j] = x[tid,j]

@ti.kernel
def copy_part_2d_(
    xx: ti.template(),
    x: ti.types.ndarray(),
    length: int,
    len2: int):

    for tid in range(length):
      for j in ti.static(range(4)):
        xx[tid][j] = x[tid,j]


@ti.kernel
def copy_part_3d(
    xx: ti.template(),
    x: ti.types.ndarray(),
    length: int):

    for tid in range(length):
      for j in ti.static(range(3)):
        for k in ti.static(range(3)):
          xx[tid][j,k] = x[tid,j,k]
        

def NewTensor2IndicesOfBaseTensor(tensor_new, tensor_base):
    return (tensor_new.view(-1,1) == tensor_base).int().argmax(dim=1).view(-1, tensor_new.size(1))
    
    
@ti.data_oriented
class IntegratorVariationalImplicit:

    def __init__(self,  V, T, gravity, dt, Nstep, indices_bdry_save, lamb, mu, output, rho, faces, masses, alpha, encoder, decoder, decoder2 = None, net_map = None, beta_damping=0, test_number=-1,warm_start=False, write_every=1):
                
                V_numpy = V
                self.V = ti.Vector.field(n = 3, dtype = ti.f32, shape = V_numpy.shape[0])
                self.V.from_numpy(V_numpy)
                
                
                T_numpy = T
                self.T = ti.Vector.field(n = 4, dtype = int, shape = T_numpy.shape[0])
                self.T.from_numpy(T_numpy)
                
                self.T_reduce = ti.Vector.field(n = 4, dtype = int, shape = T_numpy.shape[0])
                self.T_reduce.fill(0)
                #copy_part_2d_(self.T_reduce, T_reduce_numpy, T_reduce_numpy.shape[0], 4)
                
                
                
                self.V0 = ti.Vector.field(n = 3, dtype = ti.f32, shape = V_numpy.shape[0])
                self.V0.from_numpy(V_numpy)
                
                self.V0_sample = ti.Vector.field(n = 3, dtype = ti.f32, shape = V_numpy.shape[0])
                self.V0_sample.from_numpy(V_numpy)
                
                self.V0_sample_all = ti.Vector.field(n = 3, dtype = ti.f32, shape = V_numpy.shape[0])
                self.V0_sample_all.from_numpy(V_numpy)
                
                self.V_cur = ti.Vector.field(3, dtype=ti.f32, shape=V_numpy.shape[0])
                self.V_cur.fill(0)
                
                self.gravity = gravity
                self.dt = dt
                self.Nstep = Nstep
                self.output = output
                self.indices_bdry_save = indices_bdry_save
                self.warm_start = warm_start
                self.write_every = write_every
                self.alpha = alpha
                
                masses_numpy = masses
                self.masses = ti.field(ti.f32, shape=(masses_numpy.shape[0]))
                self.masses.from_numpy(masses_numpy.reshape((masses_numpy.shape[0])))
                self.inv_mass = ti.field(ti.f32, shape=(masses_numpy.shape[0]))
                
                
                
                self.masses_reduce = ti.field(ti.f32, shape=(masses_numpy.shape[0]))
                self.inv_mass_reduce = ti.field(ti.f32, shape=(masses_numpy.shape[0]))

           
                self.mg = self.masses
                self.beta_damping = beta_damping
                self.test_number = test_number
                
                self.timestep = 0

            
                outpur_dir = os.path.dirname(output)
                config_file = os.path.join(outpur_dir, 'config.h5')
                self.config = SimulationConfig(lamb, mu, self.dt, self.Nstep, self.indices_bdry_save, self.gravity, self.beta_damping, self.test_number)
                self.config.write_to_file(config_file)
                self.faces = faces
                
                faces2 = torch.tensor(faces).clone()
                
                faces2 = torch.unique(faces2.view(-1))- 1
                print("faces2", faces2.shape)
                print("V_cur", self.V0.to_torch().shape)
                
                self.V0_surface = (self.V0.to_torch()).cpu()[faces2,:]
                self.faces_idx = faces2.numpy()
                
                
                self.tri_new = NewTensor2IndicesOfBaseTensor(torch.tensor(faces)-1, faces2).numpy()
                
                
                
                
                
                self.tet_poses = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=self.T.shape[0])
                
                self.tet_activations = ti.field(ti.f32, shape=self.T.shape[0])
                
                self.tet_materials = ti.Vector.field(n = 3, dtype = ti.f32, shape = self.T.shape[0])
                
                self.tet_poses_reduce = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=self.T.shape[0])
                self.tet_poses_reduce.fill(0)
                
                self.tet_activations_reduce = ti.field(ti.f32, shape=self.T.shape[0])
                self.tet_activations_reduce.fill(0)
                
                
                self.tet_materials_reduce = ti.Vector.field(n = 3, dtype = ti.f32, shape = self.T.shape[0])
                self.tet_materials_reduce.fill(0)
                
                
                
                
                
                
                self.decoder = decoder
                self.encoder = encoder
                self.decoder2 = decoder2
                self.net_map = net_map
                
                self.sum_v_cur = 0
                
                print("v0 shape = ", self.V0.shape)
                
                #encoder_input = torch.zeros(1500,3).view(1, -1, 3)
                encoder_input = torch.zeros(1500,3).view(1, -1, 3)
                encoder_input_2 = self.V0_surface[:1500,:].view_as(encoder_input)
                encoder_input = torch.cat((encoder_input_2, encoder_input),2)
                print(encoder_input.shape)
                self.xhat = self.encoder.forward(encoder_input)
                
                self.xhat = self.xhat.detach()
                print("xhat initial = ", self.xhat)
                self.vhat = torch.zeros_like(self.xhat)
                
                
                
                lbllength = self.xhat.size(2)
                x = (self.V0.to_torch()).view(1,-1,3)
                if(self.net_map is not None):
                  x_2 = x.view(-1,x.shape[-1])
                  x = self.net_map.forward(x_2).view(x.shape[0], x.shape[1], -1)
        
                print(x.shape)
                xhat_all = self.xhat.expand(self.xhat.size(0), x.size(1), self.xhat.size(2))
                x = torch.cat((xhat_all, x), 2)
        
                print(x.shape)
                x = x.view(x.size(0)*x.size(1), x.size(2))
                jacobian_part, q = self.decoder.jacobianPartAndFunc(x, lbllength, 'fir')
                
                q = q + self.V0.to_torch().view_as(q)
        
        
                
                float2d2vec3((q.view(-1,3)), self.V_cur)
                #add(self.V_cur, self.V0)
                
                
                
                
                #print(self.tet_materials.shape, self.T.shape)
                init_tetrahedra(self.V_cur, self.T, lamb, mu, 0, self.tet_poses, self.tet_activations, self.tet_materials)
                #print(self.masses.shape) 
                
            
            
    def initialize_after_sample(self):
        print("inside init jac", self.V_cur.shape)
        
        self.particle_f = ti.Vector.field(n = 3, dtype = ti.f32, shape = self.sum_selected)
        self.opt = Optimizer(self.sum_selected*3, "gd", device=self.xhat.device)
       
        print(self.V0_sample.to_torch()[:self.sum_selected,:].shape)
        lbllength = self.xhat.size(2)
        x = (self.V0_sample.to_torch()[:self.sum_selected,:]).view(1,-1,3)
        if(self.net_map is not None):
          x_2 = x.view(-1,x.shape[-1])
          x = self.net_map.forward(x_2).view(x.shape[0], x.shape[1], -1)
               
        #x = (self.V0_sample.to_torch(self.xhat.device)).view(1,-1,3)
        
        print('x shape = ', x.shape)
        xhat_all = self.xhat.expand(self.xhat.size(0), x.size(1), self.xhat.size(2))
        x = torch.cat((xhat_all, x), 2)
        
        print(x.shape)
        x = x.view(x.size(0)*x.size(1), x.size(2))
        jacobian_part, q = self.decoder.jacobianPartAndFunc(x, lbllength, 'fir')
        if(self.decoder2 is not None):
                   jacobian_part2, q2 = self.decoder2.jacobianPartAndFunc(x, lbllength, 'fir')
                   jacobian_part = jacobian_part + jacobian_part2
                   q = q + q2
        
        print(jacobian_part.shape)
        jac = jacobian_part.view(-1, jacobian_part.size(2))
        print(jac.shape)
        self.jac_sample = jac.clone()
        
        
        float2d2vec3(q.view(-1,3),
            self.V_cur)
        
        self.xhat_now = None
        
        
        self.V_pred = ti.Vector.field(3, dtype=ti.f32, shape=self.V_cur.shape[0])
        self.Vel_cur = ti.Vector.field(3, dtype=ti.f32, shape=self.V_cur.shape[0])
        self.Vel_pred_damping = ti.Vector.field(3, dtype=ti.f32, shape=self.V_cur.shape[0])
        self.Vel_before_iteration = ti.Vector.field(3, dtype=ti.f32, shape=self.V_cur.shape[0])
                
        self.Vel_pred = ti.Vector.field(3, dtype=ti.f32, shape=self.sum_selected)
        self.Vel_pred_last = ti.Vector.field(3, dtype=ti.f32, shape=self.sum_selected)
        self.Vel_pred_last.fill(0)
        self.res = ti.Vector.field(3, dtype=ti.f32, shape=self.sum_selected)
        self.V_cur_last = ti.Vector.field(3, dtype=ti.f32, shape=self.sum_selected)
        self.forces = ti.Vector.field(3, dtype=ti.f32, shape=self.sum_selected)
                
        self.xx = ti.Vector.field(3, dtype=ti.f32, shape=self.sum_selected)
                
                
        self.xxx = ti.field(ti.f32, shape=(self.Vel_pred.shape[0] * 3))
        self.decoder_input_all = None
                
                
        self.weight = ti.field(ti.f32, shape=(self.sum_selected))
        self.weight.fill(1)
                
        self.first = True
        self.mapped = None
        self.mapped_surface = None
        self.vhat_now = None
    
    
    
                
            

    def write_to_file(self, time_step):
        pos_numpy = self.V.to_torch()[:self.V0_surface.shape[0],:].numpy()
        
        #ps_ourmesh = ps.register_surface_mesh("my mesh", pos_numpy, integrator.tri_new, edge_width = 0.0, color = (1,1,0))
        
        filename = self.output.format(time_step)
        input_x = pos_numpy
        input_q = pos_numpy
        input_t = np.array([[time_step * self.dt]])
        state = SimulationState(filename, False, input_x, input_q, input_t)
        #state.tets = self.T.numpy()
        #state.masses = self.masses.numpy()
        state.faces = self.tri_new + 1
        state.write_to_file(filename)
    
    
    
    def update_vel(self):
       with wp.ScopedTimer("update Vel"): 
         
        
        self.Vel_pred_last.fill(0)
        self.Vel_pred.fill(0)
        self.res.fill(0)
        self.forces.fill(0)
        self.Vel_pred_damping.fill(0)
        
        if (self.xhat_now is not None):
            
            
            with wp.ScopedTimer("prepare"): 
                lbllength = self.xhat.size(2)
            
                if(self.net_map is not None):
                  if(self.mapped is None):
                   x = (self.V0_sample_all.to_torch(self.xhat.device)[:self.sum_selected,:]).view(1,-1,3)
                   x_2 = x.view(-1,x.shape[-1])
                   x = self.net_map.forward(x_2).view(x.shape[0], x.shape[1], -1)
                   self.mapped = x.clone()
                  else:
                   x = self.mapped.clone()
               #print("OK")
            
           
                xhat_all = self.xhat.expand(self.xhat.size(0), x.size(1), self.xhat.size(2))
            
           
            
                x = torch.cat((xhat_all, x), 2)
                x = x.view(x.size(0)*x.size(1), x.size(2))
            
            
            
            
            
            with wp.ScopedTimer("else"):  
                
                copy_field(self.V_cur_last, self.V_cur)
                
                copy_field(self.V_cur, self.V_pred)
                self.V_pred.fill(0)
                update_vel(self.Vel_cur, self.V_cur, self.V_cur_last, self.dt)
                
                
    def update_vel_resample(self):
       with wp.ScopedTimer("update Vel resample"): 
         
        
        self.Vel_pred_last.fill(0)
        self.Vel_pred.fill(0)
        self.res.fill(0)
        self.forces.fill(0)
        self.Vel_pred_damping.fill(0)
        
        if (self.xhat_now is not None):
            
            
            with wp.ScopedTimer("prepare"): 
                lbllength = self.xhat.size(2)
            
                if(self.net_map is not None):
                  if(self.mapped is None):
                   x = (self.V0_sample_all.to_torch(self.xhat.device)[:self.sum_selected,:]).view(1,-1,3)
                   x_2 = x.view(-1,x.shape[-1])
                   x = self.net_map.forward(x_2).view(x.shape[0], x.shape[1], -1)
                   self.mapped = x.clone()
                  else:
                   x = self.mapped.clone()
               
            
           
                xhat_all = self.xhat.expand(self.xhat.size(0), x.size(1), self.xhat.size(2))
            
           
            
                x = torch.cat((xhat_all, x), 2)
                x = x.view(x.size(0)*x.size(1), x.size(2))
            
            with wp.ScopedTimer("jac part"):  
               jacobian_part, q = self.decoder.jacobianPartAndFunc(x, lbllength, 'fir')
               jac = jacobian_part.view(-1, jacobian_part.size(2))
               q = self.decoder.forward(x)
            
               self.jac_sample = jac.clone()
               #print(self.jac_sample)
            
            
            
            with wp.ScopedTimer("else"):  
                #copy_field(self.V_cur_last, self.V_cur)
                #copy_field(self.V_cur, self.V_pred)
                float2d2vec3_k(q.view(-1,3), self.V_cur, self.sum_selected)
                float2d2vec3_k(torch.matmul(jac, self.vhat_now.view(-1,1)).view(-1,3), self.Vel_cur, self.sum_selected)
                self.V_pred.fill(0)
                #update_vel(self.Vel_cur, self.V_cur, self.V_cur_last, self.dt)            
            
        
        
    def integrate(self):
      with wp.ScopedTimer("simulate"):  
        
        
        
        
        self.xx.fill(0)
        
        
        
        def residual_func(x, dfdx):

                    self.particle_f.fill(0)
                    
                    
                    
                    float2vec3(x, self.xx)
                    sub_field(self.res, self.xx, self.Vel_pred_last)
                    
                    
                    with wp.ScopedTimer("stage1"):
                      
                        self.vhat = update_state_by_latent(self.xhat, self.V0_sample_all, self.V_pred, self.Vel_pred, self.xx, self.dt, self.decoder, self.decoder2, self.net_map, self.mapped, self.jac_sample, self.weight.to_torch(self.xhat.device), self.sum_selected, self.first_iteration, self.vhat_now)
                        #print(self.vhat, self.xx)
                        
                        update_vel(self.Vel_pred_last, self.V_pred, self.V_cur, self.dt)
                        
                        
                        update_vel(self.Vel_pred_damping, self.V_pred, self.V_cur, self.dt)
                        
                        
                        self.first_iteration = False
                        self.xhat = self.xhat + self.vhat * self.dt
                        
                        
                    
                    #print("vhat = ", self.vhat)
                    with wp.ScopedTimer("stage2"):  
                        compute_forces(self.V_pred, self.V0_sample_all, self.Vel_pred_damping, self.T_reduce, self.tet_poses_reduce, self.tet_activations_reduce, self.tet_materials_reduce, self.particle_f, self.accumulate_tet, self.sum_selected) 
                        
                    with wp.ScopedTimer("stage3"):  
                        
                        compute_residual(self.Vel_cur, self.V0_sample_all, self.Vel_pred_last, self.particle_f, self.masses_reduce, self.dt, dfdx, self.sum_selected) 
                        
        
        
        
        
        if (self.timestep % 5 == 0):
              #x = warp.to_torch(self.V0).view(1,-1,3)
              
              if(self.net_map is not None):
                if(self.mapped is None):
                    x = (self.V0_surface).view(1,-1,3)
                    x_2 = x.view(-1,x.shape[-1])
                    x = self.net_map.forward(x_2).view(x.shape[0], x.shape[1], -1)
                    self.mapped_surface = x.clone()
                else:
                    x = self.mapped_surface.clone()
              xhat_all = self.xhat.expand(self.xhat.size(0), x.size(1), self.xhat.size(2))
              x = torch.cat((xhat_all, x), 2)
              x = x.view(x.size(0)*x.size(1), x.size(2))
              q = self.decoder.forward(x)
            
              
              
              float2d2vec3_k(q.view(-1,3) + self.V0_surface, self.V, self.V0_surface.shape[0])
        
        
        apply_damping(self.Vel_cur, self.Vel_cur)
        
        
        
        integrate_particles(self.V_cur, self.Vel_cur, self.forces, self.inv_mass, self.dt, self.V_pred, self.Vel_pred, self.sum_selected)
        
        
        
        x = self.Vel_pred
        
        
        self.xxx.fill(0)
        

       
        vec32float(x, self.xxx)
        
        
        self.xhat_now = self.xhat.clone()
        
        self.first_iteration = True
        
        


        self.opt.solve(
                    x=self.xxx, 
                    grad_func=residual_func,
                    max_iters=3,
                    alpha=self.alpha,
                    report=True)
              
        self.vhat_now = (self.xhat - self.xhat_now) / self.dt
        
        
        
        
        self.timestep += 1
        #if(self.timestep % 5 == 0):
        #   ps.screenshot()
           #self.write_to_file(self.timestep)




