import shutil
import os
import sys
import random
import numpy as np

import torch
from torch import norm
from torch.utils.data import Dataset

import potpourri3d as pp3d

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src

class HumanSegOrigDataset(Dataset):
    """Human segmentation dataset from Maron et al (not the remeshed version from subsequent work)"""

    def __init__(self, root_dir, train, k_eig=128, use_cache=True, op_cache_dir=None, debug=False):

        self.train = train  # bool
        self.k_eig = k_eig 
        self.root_dir = root_dir
        self.cache_dir = os.path.join(root_dir, "cache")
        self.op_cache_dir = op_cache_dir
        self.n_class = 8

        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.labels_list = []  # per-face labels!!

        self.debug = debug

        # check the cache
        if use_cache:
            train_cache = os.path.join(self.cache_dir, "train.pt")
            test_cache = os.path.join(self.cache_dir, "test.pt")
            load_cache = train_cache if self.train else test_cache
            print("using dataset cache path: " + str(load_cache))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                self.verts_list, self.faces_list, self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list, self.labels_list = torch.load( load_cache)
                return
            print("  --> dataset not in cache, repopulating")


        # Load the meshes & labels

        # Get all the files
        mesh_files = []
        label_files = []

        # Train test split
        if self.train:
    
            # adobe
            mesh_dirpath = os.path.join(self.root_dir, "meshes", "train", "adobe")
            label_dirpath = os.path.join(self.root_dir, "segs", "train", "adobe")
            for fname in os.listdir(mesh_dirpath):
                mesh_fullpath = os.path.join(mesh_dirpath, fname)
                label_fullpath = os.path.join(label_dirpath, fname[:-4] + ".txt")
                mesh_files.append(mesh_fullpath)
                label_files.append(label_fullpath)
            
            # faust
            mesh_dirpath = os.path.join(self.root_dir, "meshes", "train", "faust")
            label_dirpath = os.path.join(self.root_dir, "segs", "train", "faust")
            for fname in os.listdir(mesh_dirpath):
                mesh_fullpath = os.path.join(mesh_dirpath, fname)
                label_fullpath = os.path.join(label_dirpath, "faust_corrected.txt")
                mesh_files.append(mesh_fullpath)
                label_files.append(label_fullpath)
            
            # mit
            mesh_dirpath_patt = os.path.join(self.root_dir, "meshes", "train", "MIT_animation", "meshes_{}", "meshes")
            label_dirpath = os.path.join(self.root_dir, "segs", "train", "mit")
            pose_names = ['bouncing','handstand','march1','squat1', 'crane','jumping', 'march2', 'squat2']
            for pose in pose_names:
                mesh_dirpath = mesh_dirpath_patt.format(pose)
                for fname in os.listdir(mesh_dirpath):
                    mesh_fullpath = os.path.join(mesh_dirpath, fname)
                    label_fullpath = os.path.join(label_dirpath, "mit_{}_corrected.txt".format(pose))
                    mesh_files.append(mesh_fullpath)
                    label_files.append(label_fullpath)
            
            # scape
            mesh_dirpath = os.path.join(self.root_dir, "meshes", "train", "scape")
            label_dirpath = os.path.join(self.root_dir, "segs", "train", "scape")
            for fname in os.listdir(mesh_dirpath):
                mesh_fullpath = os.path.join(mesh_dirpath, fname)
                label_fullpath = os.path.join(label_dirpath, "scape_corrected.txt")
                mesh_files.append(mesh_fullpath)
                label_files.append(label_fullpath)
            
        else:

            # shrec
            mesh_dirpath = os.path.join(self.root_dir, "meshes", "test", "shrec")
            label_dirpath = os.path.join(self.root_dir, "segs", "test", "shrec")
            for iShrec in range(1,21):
                if iShrec == 16 or iShrec == 18: continue # why are these messing from the dataset? so many questions...
                if iShrec == 12:
                    mesh_fname = "12_fix_orientation.off"
                else:
                    mesh_fname = "{}.off".format(iShrec)
                label_fname = "shrec_{}_full.txt".format(iShrec)
                mesh_fullpath = os.path.join(mesh_dirpath, mesh_fname)
                label_fullpath = os.path.join(label_dirpath, label_fname)
                mesh_files.append(mesh_fullpath)
                label_files.append(label_fullpath)

        print("loading {} meshes".format(len(mesh_files)))

        # Load the actual files
        for iFile in range(len(mesh_files[:5]) if self.debug else len(mesh_files)):

            print("loading mesh " + str(mesh_files[iFile]))

            verts, faces = pp3d.read_mesh(mesh_files[iFile])
            labels = np.loadtxt(label_files[iFile]).astype(int)-1

            # to torch
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))
            labels = torch.tensor(np.ascontiguousarray(labels))

            # center and unit scale
            verts = self.normalize_positions(verts)

            self.verts_list.append(verts)
            self.faces_list.append(faces)
            self.labels_list.append(labels)

        for ind, labels in enumerate(self.labels_list[:5] if self.debug else self.labels_list):
            self.labels_list[ind] = labels


    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, idx):
        return self.verts_list[idx]


    def normalize_positions(self, pos, faces=None, method='mean', scale_method='max_rad'):
        # center and unit-scale positions

        if method == 'mean':
            # center using the average point position
            pos = (pos - torch.mean(pos, dim=-2, keepdim=True))
        elif method == 'bbox':
            # center via the middle of the axis-aligned bounding box
            bbox_min = torch.min(pos, dim=-2).values
            bbox_max = torch.max(pos, dim=-2).values
            center = (bbox_max + bbox_min) / 2.
            pos -= center.unsqueeze(-2)
        else:
            raise ValueError("unrecognized method")

        if scale_method == 'max_rad':
            scale = torch.max(norm(pos), dim=-1, keepdim=True).values.unsqueeze(-1)
            pos = pos / scale
        elif scale_method == 'area':
            if faces is None:
                raise ValueError("must pass faces for area normalization")
            coords = pos[faces]
            vec_A = coords[:, 1, :] - coords[:, 0, :]
            vec_B = coords[:, 2, :] - coords[:, 0, :]
            face_areas = torch.norm(torch.cross(vec_A, vec_B, dim=-1), dim=1) * 0.5
            total_area = torch.sum(face_areas)
            scale = (1. / torch.sqrt(total_area))
            pos = pos * scale
        else:
            raise ValueError("unrecognized scale method")
        return pos
