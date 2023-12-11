import meshio
import os
import torch
import h5py
import torch.linalg as LA

import warp as wp
import warp.sim


class MeshH5(object):
    def __init__(self, x=None, tets=None, faces=None, masses=None, volume=None, Dm_inv=None, accumulate_inidces_all=None):
        if x is not None: self.x = x
        if tets is not None: self.tets = tets
        if faces is not None: self.faces = faces
        if masses is not None: self.masses = masses
        if volume is not None: self.volume = volume
        if Dm_inv is not None: self.Dm_inv = Dm_inv
        if accumulate_inidces_all is not None: self.accumulate_inidces_all = accumulate_inidces_all
        
    def read(self, filename):
        with h5py.File(filename, 'r') as h5_file:
            self.x = h5_file['/x'][:]
            self.tets = h5_file['/tets'][:]
            self.faces = h5_file['/faces'][:]
            self.masses = h5_file['/masses'][:]
            self.volume = h5_file['/volume'][:]
            self.Dm_inv = h5_file['/Dm_inv'][:]
            self.accumulate_inidces_all = []
            self.accumulate_inidces_all.append(h5_file['/accumulate_inidces0'][:])
            self.accumulate_inidces_all.append(h5_file['/accumulate_inidces1'][:])
            self.accumulate_inidces_all.append(h5_file['/accumulate_inidces2'][:])
            self.accumulate_inidces_all.append(h5_file['/accumulate_inidces3'][:])

    def write(self, filename):
        x = self.x.detach().cpu().numpy()
        tets = self.tets.detach().cpu().numpy()
        faces = self.faces.detach().cpu().numpy()
        masses = self.masses.detach().cpu().numpy()
        volume = self.volume.detach().cpu().numpy()
        Dm_inv = self.Dm_inv.detach().cpu().numpy()
        accumulate_inidces0 = self.accumulate_inidces_all[0].detach().cpu().numpy()
        accumulate_inidces1 = self.accumulate_inidces_all[1].detach().cpu().numpy()
        accumulate_inidces2 = self.accumulate_inidces_all[2].detach().cpu().numpy()
        accumulate_inidces3 = self.accumulate_inidces_all[3].detach().cpu().numpy()
        with h5py.File(filename, 'w') as h5_file:
            dset = h5_file.create_dataset("x", data=x)
            dset = h5_file.create_dataset("tets", data=tets)
            dset = h5_file.create_dataset("faces", data=faces)
            dset = h5_file.create_dataset("masses", data=masses)
            dset = h5_file.create_dataset("volume", data=volume)
            dset = h5_file.create_dataset("Dm_inv", data=Dm_inv)
            dset = h5_file.create_dataset("accumulate_inidces0", data=accumulate_inidces0)
            dset = h5_file.create_dataset("accumulate_inidces1", data=accumulate_inidces1)
            dset = h5_file.create_dataset("accumulate_inidces2", data=accumulate_inidces2)
            dset = h5_file.create_dataset("accumulate_inidces3", data=accumulate_inidces3)

class MESHLoader(object):
    def __init__(self, mesh_file, device):
        self.mesh_file_name = os.path.splitext(os.path.basename(mesh_file))[0]
        mesh_file_h5 = os.path.join(os.path.dirname(mesh_file), self.mesh_file_name + '.h5')
        if not os.path.exists(mesh_file_h5):
            mesh = meshio.read(mesh_file)
            x = mesh.points
            tets = mesh.cells_dict['tetra']
            self.x = torch.from_numpy(x).to(device)
            self.tets = torch.from_numpy(tets).to(device)
            self.device = device
            self.computeFaces()
            self.computeVolumesAndMasses()
            self.precomputeAccumulateIndicesAll()
            mesh_h5 = MeshH5(self.x, self.tets, self.faces, self.masses, self.volume, self.Dm_inv, self.accumulate_inidces_all)
            mesh_h5.write(mesh_file_h5)
        else:
            mesh_h5 = MeshH5()
            mesh_h5.read(mesh_file_h5)
            self.x = mesh_h5.x
            self.faces = (mesh_h5.faces)
            self.tets =  mesh_h5.tets
            self.masses = mesh_h5.masses
            self.volume = mesh_h5.volume
            
            #self.Dm_inv = torch.from_numpy(mesh_h5.Dm_inv).to(device)
            #self.accumulate_inidces_all = []
            #self.accumulate_inidces_all.append(torch.from_numpy(mesh_h5.accumulate_inidces_all[0]).to(device))
            #self.accumulate_inidces_all.append(torch.from_numpy(mesh_h5.accumulate_inidces_all[1]).to(device))
            #self.accumulate_inidces_all.append(torch.from_numpy(mesh_h5.accumulate_inidces_all[2]).to(device))
            #self.accumulate_inidces_all.append(torch.from_numpy(mesh_h5.accumulate_inidces_all[3]).to(device))
                
    
    def computeDifference(self, V, T):
        D = []
        bs = T.size(0)
        X0or4s = V[T[:,0]]
        X1s = V[T[:,1]]
        X2s = V[T[:,2]]
        X3s = V[T[:,3]]
        fir = (X1s-X0or4s).view(bs, 3, 1)
        sec = (X2s-X0or4s).view(bs, 3, 1)
        thi = (X3s-X0or4s).view(bs, 3, 1)

        D = torch.cat((fir, sec, thi), 2)
        return D
    
    def computeMass(self, rho, vertices, tets, volumes):
        mass = torch.zeros(vertices.size(0), 1).type_as(vertices)
        for idx, tet in enumerate(tets):
            m_ver = rho * volumes[idx,0] / 4.
            '''
            if m_ver < 0.005 * 0.005 * 0.005 :
                m_ver = 0.005 * 0.005 * 0.005
            '''
            mass[tet[0]] += m_ver
            mass[tet[1]] += m_ver
            mass[tet[2]] += m_ver
            mass[tet[3]] += m_ver
        return mass

    def computeVolumesAndMasses(self):
        Dm = self.computeDifference(self.x, self.tets)
        self.volume = (LA.det(Dm) / 6.).view(-1,1,1)
        self.Dm_inv = torch.inverse(Dm)
        self.masses = self.computeMass(1, self.x, self.tets, self.volume)

    def precomputeAccumulateIndicesSingle(self, id):
        nver = self.x.size(0)
        indices_id = [ [] for _ in range(nver)] 
        for idx, tet in enumerate(self.tets):
            indices_id[tet[id]].append(idx)
        max_len = 0
        for idx in range(nver):
            max_len = max(max_len, len(indices_id[idx]))
        dummy_tet_idx = self.tets.size(0)
        sum_zero = 0
        for idx in range(nver):
            #print("len = ",len(indices_id[idx]))
            if(len(indices_id[idx]) == 0):
                sum_zero = sum_zero + 1
            while len(indices_id[idx])<max_len:
                indices_id[idx].append(dummy_tet_idx)
        #print("zero and all: ", sum_zero, nver)
        #print("maxlen: ", max_len)
        indices_id = torch.Tensor(indices_id).type_as(self.x).long()
        return indices_id
    
    def precomputeAccumulateIndicesAll(self):
        self.accumulate_inidces_all = []
        self.accumulate_inidces_all.append(self.precomputeAccumulateIndicesSingle(0))
        self.accumulate_inidces_all.append(self.precomputeAccumulateIndicesSingle(1))
        self.accumulate_inidces_all.append(self.precomputeAccumulateIndicesSingle(2))
        self.accumulate_inidces_all.append(self.precomputeAccumulateIndicesSingle(3))


    def computeFaces(self):
        def facefromTets(tets):
            faces = []
            for tet in tets:
                faces.append([tet[1], tet[3], tet[2]])
                faces.append([tet[0], tet[2], tet[3]])
                faces.append([tet[0], tet[3], tet[1]])
                faces.append([tet[0], tet[1], tet[2]])
            faces = torch.Tensor(faces).long() + 1
            return faces

        def boundaryFacets(faces):
            faces_sorted, index_sorted = torch.sort(faces, axis=1)
            faces_sorted_unq, inverse_indices, counts = torch.unique(faces_sorted, dim=0, return_inverse=True, return_counts=True)
            unique_indices = torch.where(counts[inverse_indices]==1)
            boundary_facets = faces[unique_indices]
            return boundary_facets

        faces = facefromTets(self.tets).to(self.device)
        self.faces = boundaryFacets(faces)
