from ObjLoader import *
import h5py
import os
import numpy as np

class SimulationConfig(object):
    def __init__(self, lamb, mu, dt, Nstep, indices_bdry_save, gravity, beta_damping=0, test_number=-1):
        self.lamb = lamb
        self.mu = mu
        self.dt = dt
        self.Nstep = Nstep
        self.indices_bdry_save = indices_bdry_save
        self.gravity = gravity
        self.beta_damping = beta_damping
        self.test_number = test_number
    
    def write_to_file(self, filename=None):
        if filename:
            self.filename = filename
        print('writng config: ', self.filename)
        dirname = os.path.dirname(self.filename)
        os.umask(0)
        os.makedirs(dirname, 0o777, exist_ok=True)
        with h5py.File(self.filename, 'w') as h5_file:
            dset = h5_file.create_dataset("lambda", data=np.array([[self.lamb]]))
            dset = h5_file.create_dataset("mu", data=np.array([[self.mu]]))
            dset = h5_file.create_dataset("dt", data=np.array([[self.dt]]))
            dset = h5_file.create_dataset("beta_damping", data=np.array([[self.beta_damping]]))
            dset = h5_file.create_dataset("test_number", data=np.array([[self.test_number]]))
            dset = h5_file.create_dataset("Nstep", data=np.array([[self.Nstep]]))
            #dset = h5_file.create_dataset("indices_bdry_save", data=self.indices_bdry_save.view(1,-1).cpu())
            dset = h5_file.create_dataset("gravity", data=self.gravity)
        

