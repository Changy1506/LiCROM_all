import torch
import json

import torch.utils.data


import warp as wp
import warp.sim
import warp.torch

import taichi as ti

from optimizer import *

    
    
@ti.kernel
def set_illegal(
    map_o2n: ti.template(),
    illegal: int):

    for tid in map_o2n:
        map_o2n[tid] = illegal
    

@ti.kernel
def set_legal(
    map_o2n: ti.template(),
    legal: ti.template()):

    for tid in legal:
        legal_id = legal[tid]
        map_o2n[legal_id] = legal_id
    
         
@ti.kernel
def map_from_sorted_oid(
    sorted_original_id: ti.template(),
    map_o2n: ti.template(),
    illegal: int,
    stage: int 
    ): 

    for tid in sorted_original_id:
        if(sorted_original_id[tid][0] != illegal):
             if(sorted_original_id[tid][0] < stage):
                  map_o2n[sorted_original_id[tid][0]][0] = tid
             else:
                  map_o2n[sorted_original_id[tid][0] - stage][0] = tid


@ti.kernel
def calc_sum_from_sorted_oid(
    sorted_original_id: ti.template(),
    sum_inc_1ring: ti.template(),
    illegal: int
    ): 

    for tid in sorted_original_id:
        if(sorted_original_id[tid][0] != illegal):
             sum_inc_1ring[tid][0] = 1
        else:
             sum_inc_1ring[tid][0] = 0
         
       

@ti.kernel
def calculate_new_tet(
    indices_old: ti.template(),
    indices_new: ti.template(),
    map_o2n: ti.template(),
    length: int):

    #tid = wp.tid()
    for tid in indices_old:
    
        i = indices_old[tid][0]
        j = indices_old[tid][1]
        k = indices_old[tid][2]
        l = indices_old[tid][3]
    
        i = map_o2n[i][0]
        j = map_o2n[j][0]
        k = map_o2n[k][0]
        l = map_o2n[l][0]
    
        indices_new[tid][0] = i
        indices_new[tid][1] = j
        indices_new[tid][2] = k
        indices_new[tid][3] = l
    
    
@ti.kernel
def tag_tet(
    indices: ti.template(),
    map_o2n: ti.template(),
    map_tet: ti.template(),
    sum_tet: ti.template(),
    vertex_1ring: ti.template(),
    illegal_vertex: int,
    illegal_tet: int):

    for tid in indices:
    
     i = indices[tid][0]
     j = indices[tid][1]
     k = indices[tid][2]
     l = indices[tid][3]
    
     if (map_o2n[i][0] != illegal_vertex * 3) or (map_o2n[j][0] != illegal_vertex * 3) or (map_o2n[k][0] != illegal_vertex * 3) or (map_o2n[l][0] != illegal_vertex * 3):
       map_tet[tid][0] = tid
       sum_tet[tid][0] = 1
       
       if(map_o2n[i][0] == illegal_vertex * 3):
            vertex_1ring[i][0] = i + illegal_vertex
       else:
            vertex_1ring[i][0] = i
       
       if(map_o2n[j][0] == illegal_vertex * 3):
            vertex_1ring[j][0] = j + illegal_vertex
       else:
            vertex_1ring[j][0] = j
       
       if(map_o2n[k][0] == illegal_vertex * 3):
            vertex_1ring[k][0] = k + illegal_vertex
       else:
            vertex_1ring[k][0] = k
       
       if(map_o2n[l][0] == illegal_vertex * 3):
            vertex_1ring[l][0] = l + illegal_vertex
       else:
            vertex_1ring[l][0] = l
       
     else:
       map_tet[tid][0] = illegal_tet
       sum_tet[tid][0] = 0


@ti.kernel
def calculate_new_tet_ind(
    indices: ti.template(),
    indices_new: ti.template(),
    map_tet: ti.template(),
    accumulate_tet: int):

    for tid in range(accumulate_tet):
      oid = map_tet[tid]
      
      i = indices[oid][0]
      j = indices[oid][1]
      k = indices[oid][2]
      l = indices[oid][3]
    
      indices_new[tid][0] = i
      indices_new[tid][1] = j
      indices_new[tid][2] = k
      indices_new[tid][3] = l
    
    
    

@ti.kernel
def pos_o2n(
    pos_old: ti.template(),
    pos_new: ti.template(),
    map_o2n: ti.template(),
    illegal: int
    ):

    for tid in map_o2n:
      oid = tid
      nid = map_o2n[tid][0]
      if (nid != illegal):
          pos_new[nid] = pos_old[oid]
          
@ti.kernel
def mass_o2n(
    pos_old: ti.template(),
    pos_new: ti.template(),
    map_o2n: ti.template(),
    illegal: int
    ):

    for tid in map_o2n:
      oid = tid
      nid = map_o2n[tid][0]
      if (nid != illegal):
          pos_new[nid][0] = pos_old[oid]
          
          
          
@ti.kernel
def tet_info_o2n(
    pose: ti.template(),
    pose_new: ti.template(),
    activation: ti.template(),
    activation_new: ti.template(),
    materials: ti.template(),
    materials_new: ti.template(),
    map_o2n: ti.template(),
    illegal: int,
    length: int
    ):
    
    for tid in range(length):
    
      nid = tid
      oid = map_o2n[tid]
      if (nid != illegal):
          pose_new[nid] = pose[oid]
          activation_new[nid] = activation[oid]
          materials_new[nid] = materials[oid]



@ti.kernel
def copy_part(
    x: ti.types.ndarray(),
    xx: ti.template(),
    length: int):

    for tid in range(length):
        xx[tid][0] = x[tid,0]
        xx[tid][1] = x[tid,1]
        xx[tid][2] = x[tid,2]
        
        
@ti.kernel
def copy_part_tet(
    x: ti.types.ndarray(),
    xx: ti.template(),
    length: int):

    for tid in range(length):
        xx[tid][0] = x[tid]
        
@ti.kernel
def copy_field_part(
    a: ti.template(),
    b: ti.template(),
    length: int):

    for tid in range(length):

        a[tid] = b[tid]        
        
        
@ti.kernel
def copy_field_part_mass(
    a: ti.template(),
    b: ti.template(),
    length: int):

    for tid in range(length):

        a[tid] = b[tid][0] 

@ti.kernel
def copy_field(
    a: ti.template(),
    b: ti.template()):

    for tid in a:

        a[tid] = b[tid]


class SamplePoint(object):
    def __init__(self, problem, seed = 0):
        self.problem = problem
        self.seed = seed
        
        self.head_index = None
        self.back_index = None
        
        
        
        self.indices_warp = None
                
        self.particle_o2n = ti.Vector.field(n = 1, dtype = int, shape = self.problem.V.shape[0])    
        self.vertex_1ring = ti.Vector.field(n = 1, dtype = int, shape = self.problem.V.shape[0])   
        self.sum_1ring_inc = ti.Vector.field(n = 1, dtype = int, shape = self.problem.V.shape[0])
        
        self.map_tet = None
        self.sum_tet = None
        
        
        self.mass_1ring = ti.Vector.field(n = 1, dtype = ti.f32, shape = self.problem.V.shape[0])
        
        self.T_reduce_old = None
        
        
        self.vertex_pos_1ring = ti.Vector.field(n = 3, dtype = ti.f32, shape = self.problem.V.shape[0])
        self.vertex_pos_sample = ti.Vector.field(n = 3, dtype = ti.f32, shape = self.problem.V.shape[0])
    
    
    
    
    def update_sample_ids(self):
    
        
        if(self.indices_warp is None):
             self.indices_warp = ti.Vector.field(n = 1, dtype = int, shape = self.indices.shape[0])
             print(self.indices_warp.shape, self.indices.shape)
        self.indices_warp.from_torch(self.indices.view(-1,1))
        
        self.particle_o2n.fill(0)
        self.vertex_1ring.fill(0)
        illegal_vertex = int(self.problem.V.shape[0] * 3)
        print(illegal_vertex)
        
        
        #set_illegal(self.particle_o2n, illegal_vertex) 
        self.particle_o2n.fill(illegal_vertex)
        
        #set_illegal(self.vertex_1ring, illegal_vertex) 
        self.vertex_1ring.fill(illegal_vertex)
        
        set_legal(self.particle_o2n, self.indices_warp)
        
        #step2 rearrange tets
        
        
        illegal_vertex = self.problem.V.shape[0]
        illegal_tet = self.problem.T.shape[0] * 2
        
        if(self.map_tet is None):
            self.map_tet = ti.Vector.field(n = 1, dtype = int, shape = self.problem.T.shape[0]) #wp.zeros(shape=(self.problem.T.shape[0]), dtype=int, device=self.problem.V.device)
        if(self.sum_tet is None):
            self.sum_tet = ti.Vector.field(n = 1, dtype = int, shape = self.problem.T.shape[0]) #wp.zeros(shape=(self.problem.T.shape[0]), dtype=int, device=self.problem.V.device)
        self.map_tet.fill(0)
        self.sum_tet.fill(0)
        
        
        tag_tet(self.problem.T, self.particle_o2n, self.map_tet, self.sum_tet, self.vertex_1ring, illegal_vertex, illegal_tet)   
                            
        accumulate_tet = int(torch.sum(self.sum_tet.to_torch("cpu")).cpu().numpy())
        #print("accumulate tet = ", accumulate_tet, illegal_tet)
        
        values, indices = torch.sort(self.map_tet.to_torch("cpu").view(-1))
        
        #print(values)
        self.map_tet.fill(-1)
        #copy_part_tet(values[:accumulate_tet], self.map_tet, accumulate_tet)#warp.array(values[:accumulate_tet].cpu().numpy(), dtype=int, device=self.problem.V.device)
        self.map_tet.from_torch(values.view(-1,1))
        
        self.problem.T_reduce.fill(-1) #= wp.zeros(shape=(accumulate_tet, self.problem.T.shape[1]), dtype=int, device=self.problem.V.device)
        
        calculate_new_tet_ind(
                            self.problem.T,
                            self.problem.T_reduce,
                            self.map_tet,
                            accumulate_tet
                            )   
                            
                            
        
        #T_reduce_old = wp.empty_like(self.problem.T_reduce)
        if(self.T_reduce_old is None):
            self.T_reduce_old = ti.Vector.field(n = 4, dtype = int, shape = self.problem.T_reduce.shape[0])
        copy_field_part(self.T_reduce_old, self.problem.T_reduce, accumulate_tet)
        
        #step3 add 1-ring vertex   
        
        values, indices = torch.sort(self.vertex_1ring.to_torch().view(-1))
        
        #print()
        #self.vertex_1ring.fill(self.problem.V.shape[0] * 3)
        self.vertex_1ring.from_torch(values.view(-1,1)) 
        #copy_part_tet(values.cpu(), self.vertex_1ring, self.vertex_1ring.shape[0])
        
        #print("vertex 1ring", self.vertex_1ring)
        
        #sum_1ring_inc = wp.zeros(shape=(self.problem.V.shape[0]), dtype=int, device=self.problem.V.device)
        self.sum_1ring_inc.fill(0)
        
        
        calc_sum_from_sorted_oid(self.vertex_1ring, self.sum_1ring_inc, self.problem.V.shape[0] * 3)
        
        sum_selected = self.indices.shape[0]                   
        sum_selected_1ring = torch.sum(self.sum_1ring_inc.to_torch()).cpu().numpy()
        
        
        self.problem.sum_selected = int(sum_selected_1ring)
        self.problem.sum_selected_1ring = int(sum_selected_1ring)
        self.problem.accumulate_tet = int(accumulate_tet)
        
        #print("vertex 1ring sorted", values[:sum_selected + 1],values[: sum_selected_1ring + 1])
        
        
        #set_illegal(self.particle_o2n, self.problem.V.shape[0] * 3)
        self.particle_o2n.fill(int(self.problem.V.shape[0] * 3))
       
        map_from_sorted_oid(self.vertex_1ring, self.particle_o2n, self.problem.V.shape[0] * 3, self.problem.V.shape[0])
        
        
        
        #step4 tet indices old to new, setup rest position
        #print(wp.to_torch(particle_o2n)[:33])
        #print("T reduce shape", int(self.problem.T_reduce.shape[0]))
        
        calculate_new_tet(self.T_reduce_old, self.problem.T_reduce, self.particle_o2n, accumulate_tet)
        
        
        
        self.vertex_pos_1ring.fill(0)#wp.zeros(shape=int(sum_selected_1ring), dtype=wp.vec3, device=self.problem.V.device)                    
        
        pos_o2n(self.problem.V0, self.vertex_pos_1ring, self.particle_o2n, self.problem.V.shape[0] * 3)
        
        self.vertex_pos_sample.fill(0)  
        copy_field_part(self.vertex_pos_sample, self.vertex_pos_1ring, sum_selected)
        
        copy_field(self.problem.V0_sample, self.vertex_pos_1ring)
        copy_field(self.problem.V0_sample_all, self.vertex_pos_1ring)
                
        
        
        
        
        
        ############################## 0129 ends here ##########################
        #mass_1ring = wp.zeros(shape = int(sum_selected_1ring), dtype = float, device = self.problem.V.device)
        self.mass_1ring.fill(0)
        mass_o2n(self.problem.masses, self.mass_1ring, self.particle_o2n, self.problem.V.shape[0] * 3)
        print(self.mass_1ring)
        
        #mass_sample = wp.from_torch(wp.to_torch(mass_1ring)[:sum_selected])
        
        self.problem.masses_reduce.fill(0)
        copy_field_part_mass(self.problem.masses_reduce, self.mass_1ring, int(sum_selected_1ring))

       
        tet_info_o2n(
                     self.problem.tet_poses,
                     self.problem.tet_poses_reduce,
                     self.problem.tet_activations,
                     self.problem.tet_activations_reduce,
                     self.problem.tet_materials,
                     self.problem.tet_materials_reduce,
                     self.map_tet,
                     -1,
                     int(accumulate_tet)
                     )
                      
        self.problem.initialize_after_sample()
    
    def initIndicesRandom(self, num_sample_interior, num_sample_bdry, seed=333):
        torch.manual_seed(seed)
        selection_interior = torch.ones((self.problem.V0.shape[0])).to(self.problem.xhat.device)
        
        
        indices_interior = selection_interior.nonzero().view(-1)
        
        
        weight = (self.problem.masses.to_torch()).clone() 
        weight = weight[indices_interior]
        weight = weight / weight
            
        ind_rand = list(torch.utils.data.WeightedRandomSampler((weight).view(-1).cpu().numpy().tolist(), num_sample_interior, replacement = False))
        indices_interior_select = indices_interior[ind_rand]
        self.indices = indices_interior_select#torch.cat((indices_interior_select, indices_bdry_select))
            
        #print("sample point:", self.indices)
        self.indices = self.indices.to(self.problem.xhat.device)
        
        self.update_sample_ids()
        
        
     
