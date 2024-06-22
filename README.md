# Download LiDARHuman26M dataset
- Download the dataset from the link: http://www.lidarhumanmotion.net/lidarcap/.
- Download the weight file from the link: https://pan.baidu.com/s/1J0WEdkVCE4vlfBb1RWZl9w  code: oafj
- Download `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`,`J_regressor_extra.npy` and put it in data directory.

If you want to deal with your own dataset, refer to `datasets/preprocess/lidarcap.py` to process your data into a format can be handled by LiDARCap.

# TRAIN or EVAL
### 1. Modify the info
- Modify `base.yaml`to set `DATASET_DIR` to the path where your dataset is located.
- Update the relevant information for `wandb` in `tools/common.py`.
### 2. Build Environment
```
conda create -n lidar_human python=3.7
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"（或者下载github然后pip install pointnet2_ops_lib/.）
pip install wandb
pip install h5py
pip install tqdm
pip install scipy
pip install opencv-python
pip install pyransac3d
pip install yacs
pip install plyfile
pip install scikit-image
pip install joblib
pip install chumpy
```

### 3. train 
```
python train.py --threads x --gpu x --dataset lidarcap
```

### 4.eval
```
python train.py --threads x --gpu x --dataset lidarcap --ckpt_path best-train-loss21.pth --eval --eval_bs 4 --debug
```

```
# Output reference
LiDARCap >> 01/14 13:02:28 >> Launching on GPUs 5
100%|███| 376/376 [00:13<00:00, 27.09it/s]
EVAL LOSS 0.03766141264907461
100%|██████████| 1502/1502 [00:01<00:00, 972.60it/s] 
poses_to_vertices: 100%|██████████| 188/188 [00:04<00:00, 40.94it/s]
poses_to_vertices: 100%|██████████| 188/188 [00:04<00:00, 46.83it/s]
poses_to_joints: 100%|██████████| 188/188 [00:03<00:00, 57.27it/s]
poses_to_joints: 100%|██████████| 188/188 [00:03<00:00, 57.23it/s]
44.9119471013546
79.38986271619797
66.94589555263519
102.49917954206467
0.8611590795605859
0.9491181896360409
```
