# LiCROM_offline_training

This repository contains the offline training pipeline for [LiCROM](https://arxiv.org/abs/2310.15907)

## Prerequisites
We assume a fresh install of Ubuntu 20.04. For example,

```
docker run --gpus all --shm-size 128G -it --rm -v $HOME:/home/ubuntu ubuntu:20.04
```

Install python and pip:
```
apt-get update
apt install python3-pip
```

## Dependencies
Install python package dependencies through pip:

```
pip install -r requirements.txt
```

## Usage

### Training

```python
python run.py -mode train -d [data directory] -epo [epoch sequence] -lr [learning rate scaling sequence] -batch_size [batch size] -lbl [latent space dimension] -scale_mlp [network width scale]
```

For example 

```python
python run.py -mode train -d data/fracture_p2d -lbl 20 -lr 10 2 0.2 -epo 1000 1000 1000 -batch_size 16 -scale_mlp 20 --gpus 1
```
Sample data can be downloaded from here: https://utoronto-my.sharepoint.com/:u:/g/personal/changyue_chang_mail_utoronto_ca/EbtMgxJQ4TlDp7pzA-y96IkBiQnhcGGuN5dm2AhtFhVT5Q?e=dfk9a5

### Reconstructing Simulation

```python
python run.py -mode reconstruct -m [path to .ckpt file to use]
```

You may also provide any built-in flags for PytorchLightning's [Trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags)

### Data 
Simulation data should be stored in a directory with the following structure. 
For example, 
```
├───sim_data_parent_directory (contain multiple simulation sequences; each entry in this directory is a simulation sequence)
    ├───sim_seq_ + suffix
        ├───h5_f_0000000000.h5
        ├───h5_f_0000000001.h5
        ├───...
        
    ├───....
```
See SimulationState under SimulationDataset.py for the structure of the h5 file.
