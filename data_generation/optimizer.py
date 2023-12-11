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


@ti.kernel
def gd_step(arr_x: ti.template(), 
            arr_dfdx: ti.template(),
            alpha: float):

    for tid in arr_x:
        x = arr_x[tid]
        dfdx = arr_dfdx[tid]
        x = x - dfdx*alpha
        arr_x[tid] = x



@ti.data_oriented
class Optimizer:

    def __init__(self, n, mode):
        
        self.n = n
        self.mode = mode
       
               
       
        self.dfdx = ti.field(float, shape=(n))

        


    def solve(self, x, grad_func, max_iters=20, alpha=0.01, report=False):
        

        if (report):

            stats = {}

            # reset stats
            stats["evals"] = 0
            stats["residual"] = []
            

        if (self.mode == "gd"):
            alpha_now = alpha

            for i in range(300):

                # compute residual
                grad_func(x, self.dfdx)

                
                gd_step(x, self.dfdx, alpha_now)

                if (report):

                    stats["evals"] += 1
                    
                    r = np.linalg.norm(self.dfdx.to_numpy())
                    stats["residual"].append(r)
                    alpha_now = alpha #min(alpha_now, alpha * 0.17 / (r + 0.0001))
                    if(stats["evals"] != 1):
                        if(stats["residual"][stats["evals"]-2] - r < r * 0.000000001):
                              print(stats["residual"][stats["evals"]-2] - r, r * 0.005)
                              break
                           
                    

       
                
        else:
            raise RuntimeError("Unknown optimizer")

        if (report):
            print(stats)

    
