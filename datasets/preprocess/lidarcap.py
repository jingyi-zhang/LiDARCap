from ast import parse
from plyfile import PlyData, PlyElement
from typing import List
import argparse
import numpy as np
import json
import os
import re
import sys
import h5py
import torch

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from tools import multiprocess
from modules.smpl import SMPL

smpl = SMPL().cuda()


ROOT_PATH = 'your_raw_data_path'
MAX_PROCESS_COUNT = 64

# img_filenames = []


def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    ply_data = PlyData.read(filename)['vertex'].data
    points = np.array([[x, y, z] for x, y, z in ply_data])
    return points


def save_ply(filename, points):
    points = [(points[i, 0], points[i, 1], points[i, 2])
              for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=False).write(filename)


def get_index(filename):
    basename = os.path.basename(filename)
    return int(os.path.splitext(basename)[0])


def get_sorted_filenames_by_index(dirname, isabs=True):
    if not os.path.exists(dirname):
        return []
    filenames = os.listdir(dirname)
    filenames = sorted(os.listdir(dirname), key=lambda x: get_index(x))
    if isabs:
        filenames = [os.path.join(dirname, filename) for filename in filenames]
    return filenames


def parse_json(json_filename):
    with open(json_filename) as f:
        content = json.load(f)
        beta = np.array(content['beta'], dtype=np.float32)
        pose = np.array(content['pose'], dtype=np.float32)
        trans = np.array(content['trans'], dtype=np.float32)
    return beta, pose, trans


def fix_points_num(points: np.array, num_points: int):
    points = points[~np.isnan(points).any(axis=-1)]

    origin_num_points = points.shape[0]
    if origin_num_points < num_points:
        num_whole_repeat = num_points // origin_num_points
        res = points.repeat(num_whole_repeat, axis=0)
        num_remain = num_points % origin_num_points
        res = np.vstack((res, res[:num_remain]))
    if origin_num_points >= num_points:
        res = points[np.random.choice(origin_num_points, num_points)]
    return res


def foo(id, args):
    id = str(id)
    # cur_img_filenames = get_sorted_filenames_by_index(
    #     os.path.join(ROOT_PATH, 'images', id))

    pose_filenames = get_sorted_filenames_by_index(
        os.path.join(ROOT_PATH, 'labels', '3d', 'pose', id))
    json_filenames = list(filter(lambda x: x.endswith('json'), pose_filenames))
    ply_filenames = list(filter(lambda x: x.endswith('ply'), pose_filenames))

    cur_betas, cur_poses, cur_trans = multiprocess.multi_func(
        parse_json, MAX_PROCESS_COUNT, len(json_filenames), 'Load json files',
        True, json_filenames)
    # cur_vertices = multiprocess.multi_func(
    #     read_ply, MAX_PROCESS_COUNT, len(ply_filenames), 'Load vertices files',
    #     True, ply_filenames)

    depth_filenames = get_sorted_filenames_by_index(
        os.path.join(ROOT_PATH, 'labels', '3d', 'depth', id))
    cur_depths = depth_filenames

    segment_filenames = get_sorted_filenames_by_index(
        os.path.join(ROOT_PATH, 'labels', '3d', 'segment', id))
    cur_point_clouds = multiprocess.multi_func(
        read_ply, MAX_PROCESS_COUNT, len(segment_filenames),
        'Load segment files', True, segment_filenames)

    cur_points_nums = [min(args.npoints, points.shape[0])
                       for points in cur_point_clouds]
    cur_point_clouds = [fix_points_num(
        points, args.npoints) for points in cur_point_clouds]

    poses = []
    betas = []
    trans = []
    # vertices = []
    points_nums = []
    point_clouds = []
    depths = []
    full_joints = []

    assert(args.seqlen != 0)

    n = len(cur_betas)
    # 直接补齐
    while n % args.seqlen != 0:
        # cur_img_filenames.append(cur_img_filenames[-1])
        cur_betas.append(cur_betas[-1])
        cur_poses.append(cur_poses[-1])
        cur_trans.append(cur_trans[-1])
        # cur_vertices.append(cur_vertices[-1])
        cur_point_clouds.append(cur_point_clouds[-1])
        cur_points_nums.append(cur_points_nums[-1])
        cur_depths.append(cur_depths[-1])
        n += 1
    times = n // args.seqlen
    for i in range(times):
        # [lb, ub)
        lb = i * args.seqlen
        ub = lb + args.seqlen
        # img_filenames.append(cur_img_filenames[lb:ub])
        betas.append(np.stack(cur_betas[lb:ub]))
        np_poses = np.stack(cur_poses[lb:ub])
        poses.append(np_poses)
        trans.append(np.stack(cur_trans[lb:ub]))
        # vertices.append(np.stack(cur_vertices[lb:ub]))
        point_clouds.append(np.stack(cur_point_clouds[lb:ub]))
        points_nums.append(np.stack(cur_points_nums[lb:ub]))
        depths.append(cur_depths[lb:ub])

        full_joints.append(smpl.get_full_joints(smpl(torch.from_numpy(
            np_poses).cuda(), torch.zeros((args.seqlen, 10)).cuda())).cpu().numpy())

    # return poses, betas, trans, vertices, point_clouds, points_nums
    return np.stack(poses), np.stack(betas), np.stack(trans), np.stack(point_clouds), np.stack(points_nums), depths, np.stack(full_joints)


def test(args):
    pass


def get_sorted_ids(s):
    if re.match('^([1-9]\d*)-([1-9]\d*)$', s):
        start_index, end_index = s.split('-')
        indexes = list(range(int(start_index), int(end_index) + 1))
    elif re.match('^(([1-9]\d*),)*([1-9]\d*)$', s):
        indexes = [int(x) for x in s.split(',')]
    return sorted(indexes)


def dump(args):

    seq_str = '' if args.seqlen == 0 else 'seq{}_'.format(args.seqlen)
    ids = get_sorted_ids(args.ids)

    whole_poses = np.zeros((0, args.seqlen, 72))
    whole_betas = np.zeros((0, args.seqlen, 10))
    whole_trans = np.zeros((0, args.seqlen, 3))
    # whole_vertices = np.zeros((0, args.seqlen, 6890, 3))
    whole_point_clouds = np.zeros((0, args.seqlen, args.npoints, 3))
    whole_points_nums = np.zeros((0, args.seqlen))
    whole_full_joints = np.zeros((0, args.seqlen, 24, 3))
    whole_depths = []

    for id in ids:
        # poses, betas, trans, vertices, point_clouds, points_nums = foo(
        poses, betas, trans, point_clouds, points_nums, depths, full_joints = foo(
            id, args)

        whole_poses = np.concatenate((whole_poses, np.stack(poses)))
        whole_betas = np.concatenate((whole_betas, np.stack(betas)))
        whole_trans = np.concatenate((whole_trans, np.stack(trans)))
        # whole_vertices = np.concatenate(
        #     (whole_vertices, np.stack(vertices)))
        whole_point_clouds = np.concatenate(
            (whole_point_clouds, np.stack(point_clouds)))
        whole_points_nums = np.concatenate(
            (whole_points_nums, np.stack(points_nums)))
        whole_depths += depths
        whole_full_joints = np.concatenate(
            (whole_full_joints, full_joints))

    whole_filename = args.name + '.hdf5'
    with h5py.File(os.path.join(extras_path, whole_filename), 'w') as f:
        f.create_dataset('pose', data=whole_poses)
        f.create_dataset('shape', data=whole_betas)
        f.create_dataset('trans', data=whole_trans)
        # f.create_dataset('human_vertex', data=whole_vertices)
        f.create_dataset('point_clouds', data=whole_point_clouds)
        f.create_dataset('points_num', data=whole_points_nums)
        f.create_dataset('depth', data=whole_depths)
        f.create_dataset('full_joints', data=whole_full_joints)


if __name__ == '__main__':
    extras_path = 'your_save_path'
    os.makedirs(extras_path, exist_ok=True)

    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers()

    parser_dump = subparser.add_parser('dump')
    parser_dump.add_argument('--seqlen', type=int, default=0)
    parser_dump.add_argument('--npoints', type=int, default=512)
    parser_dump.add_argument('--ids', type=str, required=True)
    parser_dump.add_argument('--name', type=str, required=True)
    parser_dump.set_defaults(func=dump)

    parser_test = subparser.add_parser('test')
    parser_test.set_defaults(func=test)

    args = parser.parse_args()
    args.func(args)
