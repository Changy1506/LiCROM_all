# LiCROM

This repository contains the source code for [LiCROM](https://arxiv.org/abs/2310.15907). Currently, it only includes the fundamental parts of the implementation without detailed documentation and has been tested solely on my laptop. A more organized and thoroughly tested version will be forthcoming in this repository.

## Usage

### Data generation
The `data_generation` folder includes the code for generating training data, with snapshots being stored in `data_generation/output/`
```
cd data_generation
python main_Holes.py 
```
After obtaining the data, the next step involves converting positions to displacements.

```python
cd ..
cd misc
python pos_disp.py -type p2d -d [data directory] 
```

### Training

Please refer to `README.md` in `training` folder.

### Online Simulation
The `online` folder includes the code for running reduced space dynamics with pre-trained model.
```
python main_Holes.py 
```

