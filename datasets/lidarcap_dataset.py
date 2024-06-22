import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
import os, random, cv2

from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import pyransac3d as pyrsc
from yacs.config import CfgNode
import torchvision.transforms as transforms

trans = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def pc_normalize(pc):
    pc[..., 0:2] -= np.mean(pc[..., 0:2], axis=1, keepdims=True)
    pc[..., 2] -= np.mean(pc[..., 2])
    pc /= 1.2
    return pc


def pc_normalize_w_raw_z(pc):
    pc[..., 0:2] -= np.mean(pc[..., 0:2], axis=1, keepdims=True)
    # pc[..., 2] -= np.mean(pc[..., 2])
    # pc /= 1.2
    return pc


def augment(points, points_num):  # (T, N, 3), (T, )
    T, N = points.shape[:2]
    augmented_points = points.copy()

    # 放缩
    scale = np.random.uniform(0.9, 1.1)
    augmented_points *= scale

    # 随机丢弃，至少保留50个点
    dropout_ratio = np.clip(0, np.random.random() *
                            (1 - 50 / np.min(points_num)), 0.5)
    drop_idx = np.where(np.random.random((T, N)) <= dropout_ratio)
    augmented_points[drop_idx] = augmented_points[0][0]

    # 高斯噪声
    jittered_points = np.clip(
        0.01 * np.random.randn(*augmented_points.shape), -0.05, 0.05)
    augmented_points += jittered_points

    return augmented_points


def affine(X, matrix):
    if type(X) == np.ndarray:
        res = np.concatenate((X, np.ones((*X.shape[:-1], 1))), axis=-1).T
        res = np.dot(matrix, res).T
    else:
        res = torch.cat((X, torch.ones((*X.shape[:-1], 1)).to(X.device)), axis=-1)
        res = matrix.to(X.device).matmul(res.transpose(1, 2)).transpose(1, 2)
    return res[..., :-1]


class TemporalDataset(Dataset):
    default_cfg = {
        'dataset_path': 'your_hdf5_dataset_path',
        'use_aug': False,
        'use_rot': False,
        'use_straight': False,
        'use_pc_w_raw_z': False,
        'ret_raw_pc': False,
        'seqlen': 16,
        'drop_first_n': 0,
        'add_noice_pc': False,
        'noice_pc_scale': 1.5,
        'set_body_label_all_one': False,
        'noice_pc_rate': 1.0,
        'replace_noice_pc': False,
        'replace_noice_pc_rate': 0.2,
        'random_permutation': False,
        'use_trans_to_normalize': False,
        'replace_pc_strategy': 'random',
        'noise_distribution': 'uniform'
    }

    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        if cfg is not None:
            assert not hasattr(self, 'cfg'), 'cfg for initialization！'
            self.cfg = cfg
            self.cfg.update({k: v for k, v in TemporalDataset.default_cfg.items() if
                             k not in self.cfg})
        else:
            cfg = TemporalDataset.default_cfg.copy()
            cfg.update(kwargs)
            self.cfg = CfgNode(cfg)

        self.update_cfg()

    def update_cfg(self, **kwargs):
        assert all([k in self.cfg for k in kwargs.keys()])
        self.cfg.update(kwargs)

        self.dataset_path = self.cfg.dataset_path
        self.dataset_ids = self.cfg.dataset_ids

        self.length = 0
        self.lidar_to_mocap_RT_flag = True
        for id in self.dataset_ids:
            with h5py.File(os.path.join(self.dataset_path, f'{id}.hdf5'), 'r') as f:
                assert f['pose'][0].shape == (
                72,), f"Dataset：{os.path.join(self.dataset_path, f'{id}.hdf5')}, the pose shape:{f['pose'][0].shape} is wrong！"
                self.length += (len(
                    f['pose']) - self.cfg.drop_first_n) // self.cfg.seqlen
                if 'lidar_to_mocap_RT' not in f:
                    self.lidar_to_mocap_RT_flag = False

        if self.cfg.use_rot or self.cfg.use_straight:
            from util.smpl import SMPL
            self.smpl = SMPL()
            # self.smpl = SMPL()
        else:
            self.lidar_to_mocap_RT_flag = False

    def __del__(self):
        if hasattr(self, 'datas'):
            for data in self.datas:
                data.close()
            print('success close dataset')

    def open_hdf5(self):
        self.datas = []
        self.datas_length = []
        # for id in tqdm(self.dataset_ids, desc="Load Datasets", ncols=60):

        for id in self.dataset_ids:
            f = h5py.File(os.path.join(self.dataset_path, f'{id}.hdf5'), 'r')
            self.datas_length.append(len(f['pose']))
            self.datas.append(f)

    def access_hdf5(self, index):
        seqlen = self.cfg.seqlen
        raw_index = int(index)
        for data, length in zip(self.datas, self.datas_length):
            dataset_max_index = (length - self.cfg.drop_first_n) // seqlen - 1
            if index > dataset_max_index:
                index -= dataset_max_index + 1
            else:
                l = self.cfg.drop_first_n + index * seqlen
                r = l + seqlen
                assert r <= len(data['pose']), 'access_hdf5：unknow fault！'

                pose = data['pose'][l:r]
                betas = data['shape'][l:r]
                trans = data['trans'][l:r]
                if 'masked_point_clouds' in data:
                    human_points = data['masked_point_clouds'][l:r]
                else:
                    human_points = data['point_clouds'][l:r]
                points_num = data['points_num'][l:r]
                full_joints = data['full_joints'][l:r]
                rotmats = data['rotmats'][l:r] if 'rotmats' in data else None
                if self.lidar_to_mocap_RT_flag:
                    if 'lidar_to_mocap_RT' in data:
                        lidar_to_mocap_RT = data['lidar_to_mocap_RT'][l:r]
                    else:
                        assert self.cfg.use_rot is False, f'[ERROR]: data{data} dont have lidar_to_mocap_RT! use_rot option only support dataset has lidar_to_mocap_RT option.'
                        lidar_to_mocap_RT = None
                else:
                    lidar_to_mocap_RT = None

                # body_label = data['body_label'][l:r] if 'body_label' in data else None
                if 'body_label' in data:
                    body_label = data['body_label'][l:r]
                else:
                    body_fake = np.ones(data['point_clouds'].shape[:2])
                    body_label = body_fake[l:r]

                back_pc = data['background_m'][l:r] if 'background_m' in data else None

                twice_noice = data['whole_noise'][
                              l:r] if 'whole_noise' in data else None

                plane_model = data['plane_model'][
                              l:r] if 'plane_model' in data else None

                assert pose.shape == (seqlen, 72) and full_joints.shape == (
                seqlen, 24, 3) and human_points.shape == (seqlen, 512, 3), 'shape is wrong！'

                sample_pc = data['sample_pc'][l:r] if 'sample_pc' in data else None

                boundary_label = data['boundary_label'][
                                 l:r] if 'boundary_label' in data else None

                project_image = data['project_image'][
                                l:r] if 'project_image' in data else None

                return pose, betas, trans, human_points, points_num, full_joints, \
                       rotmats, lidar_to_mocap_RT, body_label, sample_pc, boundary_label \
                    , project_image, back_pc, plane_model, twice_noice
        assert False, f'cant find the dataset whose index：{raw_index}'

    def access_hdf5_dataset(self, dataset_id, dataset_key):
        index = self.dataset_ids.index(dataset_id)
        if index < 0:
            print(f'cant find dataset_id:{dataset_id}！')
            return None
        return self.datas[index][dataset_key]

    def acquire_hdf5_by_index(self, index):
        for i, (data, length) in enumerate(zip(self.datas, self.datas_length)):
            if index > length - 1:
                index -= length
            else:
                return i, data

    def split_list_by_dataset(self, *l):
        left_i = 0
        seqlen = self.cfg.seqlen
        for i, length in enumerate(self.datas_length):
            # ret.append([e[left_i:left_i+length] for e in l])
            dataset_seq_count = (length - self.cfg.drop_first_n) // seqlen
            dataset_length = dataset_seq_count * seqlen
            yield [self.dataset_ids[i], ] + [e[left_i:left_i + dataset_length] for e in
                                             l]
            left_i += dataset_length

        assert all([len(e) == left_i for e in l]), 'assert false:split_list_by_dataset'

    def get_range(self, intial, P, boundary, error_index):
        for i in range(len(error_index)):
            if intial < P < boundary[error_index[i + 1]]:
                range_y = [intial - P, boundary[error_index[i + 1]] - P]
                return range_y
            elif boundary[error_index[i + 1]] < P < intial:
                range_y = [boundary[error_index[i + 1]] - P, intial - P]
                return range_y
            else:
                return [0, 0]

    def add_dis_xy(self, P_x_, P_y_, boundary_x_, boundary_y_):
        error_x_index = np.argsort(np.abs(boundary_x_ - P_x_))
        intial = boundary_y_[error_x_index[0]]
        range_y = self.get_range(intial, P_y_, boundary_y_, error_x_index)
        random_dis = random.uniform(range_y[0], range_y[1])
        return P_y_ + random_dis

    def add_dis_z(self, P_x_, P_y_, P_z_, boundary_x_, boundary_y_, boundary_z_):
        error_x_index = np.argsort(
            ((boundary_x_ - P_x_) ** 2 + (boundary_y_ - P_y_) ** 2) ** 0.5)
        intial = boundary_z_[error_x_index[0]]
        range_y = self.get_range(intial, P_z_, boundary_z_, error_x_index)
        random_dis = random.uniform(range_y[0], range_y[1])
        return P_z_ + random_dis

    def __getitem__(self, index):
        if not hasattr(self, 'datas'):
            self.open_hdf5()

        item = {}
        item['index'] = index
        try:
            pose, betas, trans, human_points, points_num, full_joints, rotmats, \
            lidar_to_mocap_RT, body_label, sample_pc, boundary_label, project_image, \
            back_pc, plane_model, twice_noise = self.access_hdf5(index)

        except NotImplementedError as e:
            print(e)
            print(
                f'[ERROR]access_hdf5 error, index is {index}, hdf5 is {self.cfg.dataset_ids[self.acquire_hdf5_by_index(index)[0]]}')

        if self.cfg.ret_raw_pc:
            item['point_clouds'] = human_points.copy()

        if self.cfg.use_sample:
            human_points = sample_pc

        if self.cfg.add_noice_pc:
            assert body_label is None
            unique_pc = [np.unique(seg, axis=0) for seg in human_points]
            noice_pc = []
            body_label = []
            for e in unique_pc:
                numa = int((512 - e.shape[0]) * self.cfg.noice_pc_rate)
                numb = 512 - numa - e.shape[0]
                noice_pc.append(np.concatenate(
                    (e,
                     e[np.random.choice(np.arange(e.shape[0]), numb)],
                     (np.random.rand(numa, 3) - 0.5) * self.cfg.noice_pc_scale + e.mean(
                         axis=0, keepdims=True),
                     ), axis=0))
                body_label.append(
                    np.concatenate((np.ones(512 - numa), np.zeros(numa)), axis=0))
            human_points = np.stack(noice_pc)
            body_label = np.stack(body_label)

        if self.cfg.set_body_label_all_one:
            body_label = np.ones(human_points.shape[:2])

        if self.cfg.use_trans_to_normalize:
            points = human_points.copy() - trans[:, np.newaxis, :]
            if back_pc is not None:
                back_pc = back_pc - trans[:, np.newaxis, :]
            if twice_noise is not None:
                twice_noise = twice_noise - trans[:, np.newaxis, :]
            # points = human_points.copy() - trans[7:8, np.newaxis, :]
            # points -= np.mean(points[7:8, :, :], axis=1)
        elif self.cfg.use_pc_w_raw_z:
            points = pc_normalize_w_raw_z(human_points.copy())
        else:
            points = pc_normalize(human_points.copy())
            # sample_pc_ = pc_normalize(sample_pc.copy())

        if self.cfg.replace_noice_pc and self.cfg.replace_noice_pc_rate > 0:
            if 'random' == self.cfg.replace_pc_strategy:
                num_of_noise = int(512 * self.cfg.replace_noice_pc_rate)
                noise_label = np.zeros((16, 512), dtype=bool)
                noice_choice = np.random.choice(np.arange(512), num_of_noise,
                                                replace=False)
                noise_label[
                    np.arange(16)[:, np.newaxis], noice_choice[np.newaxis, :]] = True
            elif 'ballquery16' == self.cfg.replace_pc_strategy:
                ballcenters = points[
                    np.arange(16)[:, np.newaxis], np.random.randint(0, 512, (16, 1))]
                noise_label = np.linalg.norm(points - ballcenters,
                                             axis=-1) < self.cfg.replace_noice_pc_rate
            elif 'ballquery1' == self.cfg.replace_pc_strategy:
                ballcenters = points[
                                  np.random.randint(0, 16), np.random.randint(0, 512)][
                              np.newaxis, :]
                noise_label = np.linalg.norm(points - ballcenters,
                                             axis=-1) < self.cfg.replace_noice_pc_rate
            else:
                raise NotImplementedError()

            if 'uniform' == self.cfg.noise_distribution:
                noise = (np.random.rand(16, 512, 3) - 0.5) * np.array(
                    [0.8, 0.8, 1.2]) + np.array([0, -0.23081176, 0])
            elif 'normal' == self.cfg.noise_distribution:
                noise = np.random.randn(16, 512, 3) * self.cfg.replace_noice_pc_rate
                if 'ballquery' in self.cfg.replace_pc_strategy:
                    noise += ballcenters
                else:
                    noise += np.array([0, -0.23081176, 0])
            else:
                raise NotImplementedError()

            # 将人体点云中一些点随机替换为噪音点
            if noise_label.sum() > 0:
                points[noise_label] = noise[noise_label]
                if body_label is None:
                    body_label = np.ones(human_points.shape[:2])
                body_label[noise_label] = 0
            else:
                if body_label is None:
                    body_label = np.ones(human_points.shape[:2])

        if self.cfg.random_permutation:
            permuatation = np.random.permutation(512)
            points = points[np.arange(16)[:, np.newaxis], permuatation[np.newaxis, :]]
            body_label = body_label[
                np.arange(16)[:, np.newaxis], permuatation[np.newaxis, :]]

        item['human_points'] = torch.from_numpy(points).float()
        item['pose'] = torch.from_numpy(pose).float()
        # if project_image is not None:
        #     # item['project_image']=torch.from_numpy(project_image).float()
        #     item['project_image'] = trans(project_image)
        # else:
        #     item['project_image']=None

        if self.cfg.use_aug:
            points_num = points_num
            augmented_points = augment(points, points_num)
            item['human_points'] = torch.from_numpy(augmented_points).float()

        item['points_num'] = torch.from_numpy(points_num).int()
        item['betas'] = torch.from_numpy(betas).float()
        item['trans'] = torch.from_numpy(trans).float()

        if self.cfg.use_boundary:
            if len(boundary_label.shape) == 3:
                human_boundary_points = points * boundary_label
                item['human_points'] = torch.from_numpy(human_boundary_points).float()
            elif len(boundary_label.shape) == 2:
                human_boundary_points = points * boundary_label[..., np.newaxis]
                item['human_points'] = torch.from_numpy(human_boundary_points).float()

        if len(boundary_label.shape) == 3:
            human_boundary_points = points * boundary_label
            item['human_boundary'] = torch.from_numpy(human_boundary_points).float()
        elif len(boundary_label.shape) == 2:
            human_boundary_points = points * boundary_label[..., np.newaxis]
            item['human_boundary'] = torch.from_numpy(human_boundary_points).float()

        if self.cfg.inside_random:
            if len(boundary_label.shape) == 3:
                boundary_label_ = boundary_label.astype(bool).squeeze()
            elif len(boundary_label.shape) == 2:
                boundary_label_ = boundary_label.astype(bool)
            final_random = np.zeros((0, points.shape[1], points.shape[2]))
            for index in range(points.shape[0]):
                # 512 * 3
                boundary_points = points[index][boundary_label_[index]]
                boundary_x, boundary_y, boundary_z, = boundary_points[:, 0], \
                                                      boundary_points[:, 1], \
                                                      boundary_points[:, 2]
                inside_points = points[index][~boundary_label_[index]]
                random_noise = np.zeros_like(inside_points)
                for k in range(inside_points.shape[0]):
                    # 1 * 3
                    point = inside_points[k]
                    P_x, P_y, P_z = point[0], point[1], point[2]
                    P_y = self.add_dis_xy(P_x, P_y, boundary_x, boundary_y)
                    # P_x = self.add_dis_xy(P_y, P_x, boundary_y, boundary_x)
                    P_z = self.add_dis_z(P_x, P_y, P_z, boundary_x, boundary_y,
                                         boundary_z)
                    random_noise[k] = np.array([P_x, P_y, P_z])
                random = np.concatenate((random_noise, boundary_points))
                np.random.shuffle(random)

                #     random_noise_1 = np.random.rand(inside_points.shape[0],inside_points.shape[1]) * 0.15
                #     random_noise_2 = np.random.rand(inside_points.shape[0],inside_points.shape[1]) * 0.15
                #     random_noise = random_noise_2 - random_noise_1
                #     random_noise[:,-1:] = random_noise[:,-1:]*2
                #     random_inside = inside_points + random_noise
                #     try:
                #         bool_ = in_convex_polyhedron(boundary_points, random_inside)
                #         final_inside = inside_points * ~bool_ + random_inside * bool_
                #         random = np.concatenate((final_inside, boundary_points))
                #     except:
                #         random = points[index]
                final_random = np.concatenate((final_random, random[np.newaxis, ...]))

            item['human_points'] = torch.from_numpy(final_random).float()

        if full_joints is not None:
            item['full_joints'] = torch.from_numpy(full_joints).float()
        if rotmats is not None:
            item['rotmats'] = torch.from_numpy(rotmats).float()
        if self.lidar_to_mocap_RT_flag:
            item['lidar_to_mocap_RT'] = torch.from_numpy(lidar_to_mocap_RT).float()
        if body_label is not None:
            item['body_label'] = body_label.astype(np.float64)
        if back_pc is not None:
            # item['back_pc'] = back_pc.astype(np.float64)
            item['back_pc'] = torch.from_numpy(back_pc).float()
        if twice_noise is not None:
            item['twice_noise'] = torch.from_numpy(twice_noise).float()

        if plane_model is not None:
            item['plane_model'] = torch.from_numpy(plane_model).float()

        if self.cfg.concat_info:
            concat = np.concatenate(
                (item['human_points'], item['back_pc'], item['human_boundary']), axis=1)
            item['human_points'] = concat
        return item

    def __len__(self):
        return self.length


def fix_dataset_seqlen(dataset_id):
    import h5py
    output_dataset_name = os.path.join('your_data_path',
                                       f'{dataset_id}_fix_dataset_seqlen.hdf5')
    dataset_name = os.path.join('your_data_path',
                                f'{dataset_id}.hdf5')

    print(f'reading data：{dataset_name}')
    dataset = h5py.File(dataset_name, 'r')

    with h5py.File(output_dataset_name, 'w') as f:
        for k, v in dataset.items():
            print('ori dataset', k, v.shape)
            if len(v.shape) >= 2 and v.shape[1] == 16:
                new_v = v[:].reshape(v.shape[0] * v.shape[1], *v.shape[2:])
                print('new dataset', k, new_v.shape)
                f.create_dataset(k, data=new_v)
    print(f'success change the seqlen of data：{dataset_id}to 0！')
    print(f'new date saved in：{output_dataset_name}')


def collate(batch, _use_shared_memory=True):
    """Puts each data field into a tensor with outer dimension batch size.
    Copied from https://github.com/pytorch in torch/utils/data/_utils/collate.py
    """
    import re
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        assert elem_type.__name__ == 'ndarray'
        # array of string classes and object
        if re.search('[SaUO]', elem.dtype.str) is not None:
            raise TypeError(error_msg.format(elem.dtype))
        batch = [torch.from_numpy(b) for b in batch]
        try:
            return torch.stack(batch, 0)
        except RuntimeError:
            return batch
    elif batch[0] is None:
        return list(batch)
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], str):
        return batch
    elif isinstance(batch[0], dict):
        return {key: collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))
