import argparse
import shutil
import random
import os
import re
import math
import numpy as np
import h5py

import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
rootrdir = os.path.dirname(currentdir)
commo_dir = os.path.join(rootrdir,'common')
sys.path.append(commo_dir)

from SimulationData import *

hprefix = 'h5_f'
config_file_pattern = r'h5_f_(\d+)\.h5'
config_file_matcher = re.compile(config_file_pattern)
dir_pattern = r'sim_seq_(.*?)'
dir_matcher = re.compile(dir_pattern)

parser = argparse.ArgumentParser(
    description='pos2disp')
parser.add_argument('-d', help='path to the dataset',
                    type=str, nargs=1, required=True)
parser.add_argument('-type', help='p2d or d2p',
                    type=str, nargs=1, required=True)
args = parser.parse_args()

path = args.d[0]
type = args.type[0]
if type == 'p2d':
    path_changed = path + '_p2d'
elif type == 'd2p':
    path_changed = path + '_d2p'
else:
    exit('invalid type')


def obtainFilesRecursively(path, train_ratio):
    data_list = []
    data_train_list = []
    data_test_list = []
    data_train_dir = []
    data_test_dir = []

    dir_list = os.listdir(path)
    print(dir_list)

    num_sims = 0
    dir_list_sim = []
    for dirname in dir_list:
        if os.path.isdir(os.path.join(path,dirname)):
            #dir_match = dir_matcher.match(
            #    dirname)
            #if dir_match != None:
                num_sims += 1
                dir_list_sim.append(dirname)
    random.seed(0)
    random.shuffle(dir_list_sim)

    train_size = math.ceil(train_ratio * num_sims)
    test_size = num_sims - train_size

    counter = 0
    for dirname in dir_list_sim:
        data_list_local = data_train_list if counter < train_size else data_test_list
        data_dir_local = data_train_dir if counter < train_size else data_test_dir
        data_dir_local.append(os.path.join(path, dirname))
        counter += 1
        for filename in os.listdir(os.path.join(path, dirname)):
            config_file_match = config_file_matcher.match(
                filename)
            if config_file_match is None:
                continue
            # skip files begin
            file_number = int(config_file_match[1])
            # skip files finish
            # print(file_number)
            fullfilename = os.path.join(path, dirname, filename)
            data_list.append(fullfilename)
            data_list_local.append(fullfilename)
        # exit()
    return data_list, data_train_list, data_test_list, data_train_dir, data_test_dir

data_list, _, _, _, _ = obtainFilesRecursively(path, 1.0)
dir_list = set()
for h5_file in data_list:
    dirname_original = os.path.dirname(h5_file)
    dirname = os.path.join(path_changed, os.path.basename(dirname_original))
    os.umask(0)
    os.makedirs(dirname, 0o777, exist_ok=True)
    if not dirname_original in dir_list:
        dir_list.add(dirname_original)
        config_file = os.path.join(dirname_original, 'config.h5')
        if os.path.exists(config_file):
            shutil.copy(config_file, dirname)
    filename = os.path.basename(h5_file) 
    output_filename = os.path.join(dirname, filename)
    state = SimulationState(h5_file)
    if type == 'p2d':
        state.q -= state.x
        print("q shape = ", state.q.shape)
        if hasattr(state, 'f_tensor'):
            state.f_tensor -= np.eye(3)
        delattr(state, 'tets')    
        delattr(state, 'faces') # not writing obj file, which is very slow for large mesh
    elif type == 'd2p':
        config_file = os.path.join(dirname_original, 'config.h5')
        state.q += state.x
        #with h5py.File(config_file, 'r') as h5_file:
        #    faces = h5_file['/faces'][:]
        #    state.faces = faces
        if hasattr(state, 'f_tensor'):
            state.f_tensor += np.eye(3)
    else:
        exit('invalid type')
    
    state.write_to_file(output_filename)
    
