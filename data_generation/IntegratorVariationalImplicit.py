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



from taichi._lib import core as _ti_core

import taichi as ti


import os, psutil
process = psutil.Process(os.getpid())

wp.init()


        
        
@ti.kernel
def apply_boundary_bottom(x_0: ti.template(),
                 x_pred: ti.template(),
                 x_pred_out: ti.template(),
                 v_pred: ti.template()):

    for tid in x_0:
        
        if x_0[tid][1] < 0.15:  
            x_pred_out[tid] = x_0[tid]
            v_pred[tid] = v_pred[tid] - v_pred[tid]
            
        else:
            x_pred_out[tid] = x_pred[tid]
            #print("yes")


#copied from https://github.com/NVIDIA/warp/blob/main/warp/sim/integrator_euler.py

    
@ti.kernel
def integrate_particles(x: ti.template(),
                        v: ti.template(),
                        f: ti.template(),
                        w: ti.template(),
                        dt: ti.f32,
                        x_new: ti.template(),
                        v_new: ti.template()):

    for tid in x:

        x0 = x[tid]
        v0 = v[tid]
        f0 = f[tid]

        inv_mass = w[tid]
        
        #print(inv_mass)
        #print(gravity)

        v1 = v0 + (f0 * inv_mass) *dt
    
        
        x1 = x0 + v1 * dt
        #print(x1, x0, v1)

        x_new[tid] = x1
        v_new[tid] = v1




  
    
       

@ti.kernel
def eval_tetrahedra(x: ti.template(),
                    X0: ti.template(),
                    v: ti.template(),
                    indices: ti.template(),
                    pose: ti.template(),
                    activation: ti.template(),
                    materials: ti.template(),
                    f: ti.template(),
                    len_f: int,
                    first: int):

    for tid in indices:

        i = indices[tid][0]
        j = indices[tid][1]
        k = indices[tid][2]
        l = indices[tid][3]
    

        act = activation[tid]
    

        k_mu = materials[tid][0]
        k_lambda = materials[tid][1]
        k_damp = materials[tid][2]
        #if(first == 1):
        k_damp = 0.001 * k_mu
    

        x0 = x[i]# + X0[i]
        x1 = x[j]# + X0[j]
        x2 = x[k]# + X0[k]
        x3 = x[l]# + X0[l]
    

        v0 = v[i]
        v1 = v[j]
        v2 = v[k]
        v3 = v[l]

        x10 = x1 - x0
        x20 = x2 - x0
        x30 = x3 - x0

        v10 = v1 - v0
        v20 = v2 - v0
        v30 = v3 - v0
    
    

        
        Ds = ti.Matrix([[x10[0], x20[0], x30[0]], [x10[1], x20[1], x30[1]], [x10[2], x20[2], x30[2]]]) 
        
        Dm = pose[tid]
    

        inv_rest_volume = Dm.determinant() * 6.0
        #print(10000 / inv_rest_volume)
        rest_volume = 1 / inv_rest_volume

        alpha = 1.0 + k_mu / k_lambda - k_mu / (4.0 * k_lambda)
    

        # scale stiffness coefficients to account for area
        
        
        
        k_mu = k_mu * rest_volume
        k_lambda = k_lambda * rest_volume
        k_damp = k_damp * rest_volume

        # F = Xs*Xm^-1
        F = Ds @ Dm
        
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
        
        #if J < 0:
        #  print(">>>>>>>>>>>>>>>>>>>>")
    
        #print("J = ", J)

        #print(J)
        s = inv_rest_volume / 6.0
        dJdx1 = x20.cross(x30) * s
        dJdx2 = x30.cross(x10) * s
        dJdx3 = x10.cross(x20) * s

        f_volume = (J - alpha + act) * k_lambda
        f_damp = (dJdx1.dot(v1) + dJdx2.dot(v2) + dJdx3.dot(v3)) * k_damp
        f_total = f_damp + f_volume #+ f_damp
        
        #print(f_damp, f_volume)
    

        f1 = f1 + dJdx1 * f_total
        f2 = f2 + dJdx2 * f_total
        f3 = f3 + dJdx3 * f_total
        f0 = (f1 + f2 + f3) * (0.0 - 1.0)
    
    
        # apply forces
        
        if(i < len_f):
           ti.atomic_sub(f[i], f0)
        if(j < len_f):
           ti.atomic_sub(f[j], f1)
        if(k < len_f):
           ti.atomic_sub(f[k], f2)
        if(l < len_f):
           ti.atomic_sub(f[l], f3)
       


def compute_forces(V_pred, V0, Vel_pred, T, tet_poses, tet_activations, tet_materials, particle_f, first):
   
   eval_tetrahedra(V_pred, V0, Vel_pred, T, tet_poses, tet_activations, tet_materials, particle_f, particle_f.shape[0], first)
   









    
    
@ti.kernel
def update_particle_position(
    particle_q_0: ti.template(),
    particle_q_1: ti.template(),
    particle_qd_1: ti.template(),
    x: ti.template(),
    dt: ti.f32,
    top: ti.f32):

    for tid in x:

        qd_1 = x[tid]
    
        q_0 = particle_q_0[tid]
        q_1 = q_0 + qd_1*dt

        
        qd_1 = (q_1 - q_0) / dt
        x[tid] = qd_1
        
        particle_q_1[tid] = q_1
        particle_qd_1[tid] = qd_1    
        
    
#where we should do the projection

def update_state(particle_q, particle_q_out, particle_qd_out, x, dt, top):

    update_particle_position(
            particle_q,
            particle_q_out,
            particle_qd_out,
            x,
            dt,
            top)


    
@ti.kernel
def compute_particle_residual(particle_qd_0: ti.template(),
                            particle_q_0: ti.template(),
                            particle_qd_1: ti.template(),
                            particle_f: ti.template(),
                            particle_m: ti.template(),
                            dt: ti.f32,
                            residual: ti.template()):

    for tid in particle_m:

        m = particle_m[tid]
        v1 = particle_qd_1[tid]
        v0 = particle_qd_0[tid]
        f = particle_f[tid]
        
        
        
        err = (v1-v0) * m * 1. - f*dt
        
        
    
        residual[tid * 3] = err[0] 
        residual[tid * 3 + 1] = err[1] 
        residual[tid * 3 + 2] = err[2] 
    

def compute_residual(Vel_cur, V0, Vel_pred, particle_f, particle_mass, dt, residual):
    
    compute_particle_residual(
            Vel_cur,
            V0,
            Vel_pred,
            particle_f,
            particle_mass,
            dt,
            residual)
    
        


        
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
            #vel_out[tid] = vel_in[tid] #lawson
            #vel_out[tid] = vel_in[tid] * 0.99#bunny
            vel_out[tid] = vel_in[tid] * 0.99#human
        
        

    
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
def copy_field(
    a: ti.template(),
    b: ti.template()):

    for tid in a:

        a[tid] = b[tid]

@ti.data_oriented
class IntegratorVariationalImplicit:

    def __init__(self,  V, T, gravity, dt, Nstep, indices_bdry_save, lamb, mu, output, rho, faces, masses, alpha, beta_damping=0, test_number=-1,warm_start=False, write_every=1):
                V_numpy = V
                print(V_numpy, V_numpy.shape)
                self.V = ti.Vector.field(n = 3, dtype = ti.f32, shape = V_numpy.shape[0])
                self.V.from_numpy(V_numpy)
                
                
                T_numpy = T
                self.T = ti.Vector.field(n = 4, dtype = int, shape = T_numpy.shape[0])
                self.T.from_numpy(T_numpy)
                
                
                
                
                
                self.V0 = ti.Vector.field(n = 3, dtype = ti.f32, shape = V_numpy.shape[0])
                self.V0.from_numpy(V_numpy)
                
               
                
                self.V_cur = ti.Vector.field(3, dtype=ti.f32, shape=V_numpy.shape[0])
                self.V_cur.from_numpy(V_numpy)
                
                
                
                self.Vel_cur = ti.Vector.field(n = 3, dtype = ti.f32, shape = V_numpy.shape[0])
                self.Vel_cur.fill(0)
                self.V_pred = ti.Vector.field(n = 3, dtype = ti.f32, shape = V_numpy.shape[0])
                self.V_pred.fill(0)
                self.Vel_pred = ti.Vector.field(n = 3, dtype = ti.f32, shape = V_numpy.shape[0])
                self.Vel_pred.fill(0)
                self.forces = ti.Vector.field(3, dtype=ti.f32, shape=V_numpy.shape[0])
                self.forces.fill(0)
                self.particle_f = ti.Vector.field(n = 3, dtype = ti.f32, shape = V_numpy.shape[0])
                self.particle_f.fill(0)
                
                self.gravity = gravity
                self.dt = dt
                self.Nstep = Nstep
                self.output = output
                
                self.warm_start = warm_start
                self.write_every = write_every
                self.alpha = alpha
                
                masses_numpy = masses
                self.masses = ti.field(ti.f32, shape=(masses_numpy.shape[0]))
                self.masses.from_numpy(masses_numpy.reshape((masses_numpy.shape[0])))
                self.inv_mass = ti.field(ti.f32, shape=(masses_numpy.shape[0]))
                self.inv_mass.from_numpy((1 / masses_numpy).reshape((masses_numpy.shape[0])))
                
                self.mg = self.masses
                self.beta_damping = beta_damping
                self.test_number = test_number
                
                self.timestep = 0

            
                outpur_dir = os.path.dirname(output)
                config_file = os.path.join(outpur_dir, 'config.h5')
                self.config = SimulationConfig(lamb, mu, self.dt, self.Nstep, None, self.gravity, self.beta_damping, self.test_number)
                self.config.write_to_file(config_file)
                self.faces = faces
                
                
                
                
                
                
                self.tet_poses = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=self.T.shape[0])
                
                self.tet_activations = ti.field(ti.f32, shape=self.T.shape[0])
                
                self.tet_materials = ti.Vector.field(n = 3, dtype = ti.f32, shape = self.T.shape[0])
                
                
                self.xx = ti.Vector.field(3, dtype=ti.f32, shape=V_numpy.shape[0])
                
                self.xxx = ti.field(ti.f32, shape=(V_numpy.shape[0] * 3))
                
                
                
                init_tetrahedra(self.V_cur, self.T, lamb, mu, 0, self.tet_poses, self.tet_activations, self.tet_materials)
                
                self.opt = Optimizer(V_numpy.shape[0]*3, "gd")
                
            

    

                
            

    def write_to_file(self, time_step):
        filename = self.output.format(time_step)
        input_x = self.V0.to_numpy()
        input_q = self.V_cur.to_numpy()
        input_t = np.array([[time_step * self.dt]])
        state = SimulationState(filename, False, input_x, input_q, input_t)
        state.tets = self.T.to_numpy()
        state.masses = self.masses.to_numpy()
        state.faces = self.faces#.numpy()
        state.forces = self.particle_f.to_numpy()
        state.write_to_file(filename)
        
        
    def integrate(self):
      with wp.ScopedTimer("simulate"):  
        
        
        
        self.V_pred.fill(0)
        self.Vel_pred.fill(0)
        self.forces.fill(0)
        self.first = 1
        
        def residual_func(x, dfdx):

                    self.particle_f.fill(0)
                    
                    
                    self.xx.fill(0)

                    
                    
                    float2vec3(x, self.xx)
                    
                    #update_state(self.V_cur, V_pred, Vel_pred, xx, self.dt)
                    with wp.ScopedTimer("stage1"):  
                        #self.vhat = update_state_by_latent(self.xhat, self.V0_sample_all, self.V_pred, self.Vel_pred, self.xx, self.dt, self.decoder, self.jac_sample, self.masses_reduce.to_torch(self.xhat.device))
                        update_state(self.V_cur, self.V_pred, self.Vel_pred, self.xx, self.dt, self.top)
                    
                    #print("vhat = ", self.vhat)
                    with wp.ScopedTimer("stage2"):  
                        compute_forces(self.V_pred, self.V0, self.Vel_pred, self.T, self.tet_poses, self.tet_activations, self.tet_materials, self.particle_f, self.first) 
                        self.first = 2
                        print("++++++++++++++++++++++++++++++++++++++++++++ memory: ", process.memory_info().rss) 
                    
                    #compute_residual(self.Vel_cur, Vel_pred, self.particle_f, self.masses, self.gravity, self.dt, dfdx) 
                    with wp.ScopedTimer("stage3"):  
                        compute_residual(self.Vel_cur, self.V0, self.xx, self.particle_f, self.masses, self.dt, dfdx) 
        
        #particle prediction
        
        #print("vel cur", self.Vel_cur.shape, self.Vel_cur_all.shape)
        #print(self.Vel_cur)
        
        
        apply_damping(self.Vel_cur, self.Vel_cur)
        
        
       
        apply_boundary_bottom(self.V0, self.V_cur, self.V_cur, self.Vel_cur)
        
        
        
        integrate_particles(self.V_cur, self.Vel_cur, self.forces, self.inv_mass, self.dt, self.V_pred, self.Vel_pred)
        
        
        
        x = self.Vel_pred
        self.xxx.fill(0)
        

        vec32float(self.Vel_cur, self.xxx)
   
        
        self.opt.solve(
                    x=self.xxx, 
                    grad_func=residual_func,
                    max_iters=80,
                    alpha=self.alpha, 
                    report=True)
        

        
        
        copy_field(self.V_cur, self.V_pred)
        copy_field(self.Vel_cur, self.xx)
        
        
        
        
        
        
        #self.write_to_file(self.timestep)
        #print(self.timestep)
        self.timestep += 1
        if(self.timestep % 5 == 0):
            self.write_to_file(self.timestep)
        
        #   ps.screenshot()




