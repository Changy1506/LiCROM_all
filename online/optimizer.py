# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from warp.context import synchronize
import warp as wp

import numpy as np

from taichi._lib import core as _ti_core

import taichi as ti

'''
@ti.kernel
def gd_step(arr_x: ti.template(), 
            arr_dfdx: ti.template(),
            alpha: float):

    for tid in arr_x:
        x = arr_x[tid]
        dfdx = arr_dfdx[tid]
        x = x - dfdx*alpha
        
        #if(tid % 100 == 0):
        #   print(x, arr_x[tid], dfdx, alpha)
        arr_x[tid] = x
'''

@ti.kernel
def gd_step(arr_x: ti.template(), 
            arr_dfdx: ti.template(),
            alpha: float):

    for tid in arr_x:
        x = arr_x[tid]
        dfdx = arr_dfdx[tid]
        x = x - dfdx*alpha
        
        arr_x[tid] = - dfdx*alpha
        
        
@ti.data_oriented
class Optimizer:

    def __init__(self, n, mode, device):
        
        self.n = n
        self.mode = mode
        self.device = device
               
        # allocate space for residual buffers
        #self.dfdx = wp.zeros(n, dtype=float, device=device)
        self.dfdx = ti.field(float, shape=(n))

        


    def solve(self, x, grad_func, max_iters=20, alpha=0.01, report=False):
        

        if (report):

            stats = {}

            # reset stats
            stats["evals"] = 0
            stats["residual"] = []
            

        if (self.mode == "gd"):
            alpha_now = alpha

            for i in range(max_iters):

                # compute residual
                grad_func(x, self.dfdx)

                # gradient step
                #wp.launch(kernel=gd_step, dim=self.n, inputs=[x, self.dfdx, alpha], device=self.device)
                gd_step(x, self.dfdx, alpha_now)
                #return
                
                

                if (report):

                    stats["evals"] += 1
                     
                    #alpha_now *= 0.9
                    
                    r = np.linalg.norm(self.dfdx.to_numpy())
                    stats["residual"].append(r)
                    if(stats["evals"] > 1):
                        if(stats["residual"][stats["evals"]-2] - r < 0): # and stats["residual"][stats["evals"]-2] - r > 0):
                           #alpha_now *= 0.5
                           break
                        #elif stats["residual"][stats["evals"]-2] - r > 0:
                        #   alpha_now *= -1
                        #elif (stats["residual"][stats["evals"]-2] - r < 0):
                        #   stats["residual"][stats["evals"]-1] = stats["residual"][stats["evals"]-2]
                        #   alpha_now *= 0.1
                
                           
                    

       
                
        else:
            raise RuntimeError("Unknown optimizer")

        if (report):
            
            print(stats)

    
