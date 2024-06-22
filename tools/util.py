import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from modules.geometry import rotation_matrix_to_axis_angle
from modules.smpl import SMPL
from tools import path_util
from tools.multiprocess import multi_func
from tqdm import tqdm
import json
import numpy as np
import pickle
import torch

data_folder = '/data'
lidarcap_dataset_folder = '/data/lidarcap'




def get_gt_pose(pose_filename):
    with open(pose_filename) as f:
        content = json.load(f)
        gt_pose = np.array(content['pose'], dtype=np.float32)
        return gt_pose


def get_gt_poses(idx):
    pose_folder = '{}/labels/3d/pose/{}'.format(lidarcap_dataset_folder, idx)
    gt_poses = []
    pose_filenames = list(filter(lambda x: x.endswith(
        '.json'), path_util.get_sorted_filenames_by_index(pose_folder)))

    gt_poses = multi_func(get_gt_pose, 32, len(
        pose_filenames), 'get_gt_poses', True, pose_filenames)
    # for pose_filename in tqdm(pose_filenames):
    #     with open(pose_filename) as f:
    #         content = json.load(f)
    #         gt_pose = np.array(content['pose'], dtype=np.float32)
    #         gt_poses.append(gt_pose)
    gt_poses = np.stack(gt_poses)
    return gt_poses




def get_pred_poses(filename):
    pred_rotmats = np.load(filename).reshape(-1, 24, 3, 3)
    pred_poses = []
    for pred_rotmat in tqdm(pred_rotmats):
        pred_poses.append(rotation_matrix_to_axis_angle(
            torch.from_numpy(pred_rotmat)).numpy().reshape((72, )))
    pred_poses = np.stack(pred_poses)
    return pred_poses



def poses_to_vertices(poses, trans=None):
    poses = poses.astype(np.float32)
    vertices = []

    n = len(poses)
    smpl = SMPL().cuda()
    batch_size = 128
    n_batch = (n + batch_size - 1) // batch_size

    for i in tqdm(range(n_batch), desc='poses_to_vertices'):
        lb = i * batch_size
        ub = (i + 1) * batch_size

        cur_n = min(ub - lb, n - lb)
        cur_vertices = smpl(torch.from_numpy(
            poses[lb:ub]).cuda(), torch.zeros((cur_n, 10)).cuda())
        vertices.append(cur_vertices.cpu().numpy())

    vertices = np.concatenate(vertices, axis=0)
    if trans is not None:
        trans = trans.astype(np.float32)
        vertices += np.expand_dims(trans, 1)
    return vertices


def save_smpl_ply(vertices, filename):
    if type(vertices) == torch.Tensor:
        vertices = vertices.squeeze().cpu().detach().numpy()
    if vertices.ndim == 3:
        assert vertices.shape[0] == 1
        vertices = vertices.squeeze(0)
    model_file = 'data/lidarcap/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
    with open(model_file, 'rb') as f:
        smpl_model = pickle.load(f, encoding='iso-8859-1')
        face_index = smpl_model['f'].astype(np.int64)
    face_1 = np.ones((face_index.shape[0], 1))
    face_1 *= 3
    face = np.hstack((face_1, face_index)).astype(int)
    with open(filename, "wb") as zjy_f:
        np.savetxt(zjy_f, vertices, fmt='%f %f %f')
        np.savetxt(zjy_f, face, fmt='%d %d %d %d')
    ply_header = '''ply
format ascii 1.0
element vertex 6890
property float x
property float y
property float z
element face 13776
property list uchar int vertex_indices
end_header
    '''
    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header)
        f.write(old)


def poses_to_joints(poses):
    poses = poses.astype(np.float32)
    joints = []

    n = len(poses)
    smpl = SMPL().cuda()
    batch_size = 128
    n_batch = (n + batch_size - 1) // batch_size

    for i in tqdm(range(n_batch), desc='poses_to_joints'):
        lb = i * batch_size
        ub = (i + 1) * batch_size

        cur_n = min(ub - lb, n - lb)
        cur_vertices = smpl(torch.from_numpy(
            poses[lb:ub]).cuda(), torch.zeros((cur_n, 10)).cuda())
        cur_joints = smpl.get_full_joints(cur_vertices)
        joints.append(cur_joints.cpu().numpy())
    joints = np.concatenate(joints, axis=0)
    return joints
