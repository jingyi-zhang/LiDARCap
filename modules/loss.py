import torch
import torch.nn as nn

from modules.smpl import SMPL
from modules.geometry import axis_angle_to_rotation_matrix


def batch_pc_normalize(pc):
    pc -= pc.mean(1, True)
    return pc / pc.norm(dim=-1, keepdim=True).max(1, True)[0]



class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion_param = nn.MSELoss()
        self.criterion_joints = nn.MSELoss()
        self.criterion_vertices = nn.MSELoss()
        # self.chamfer_loss = ChamferLoss()
        self.smpl = SMPL()

    def forward(self, **kw):
        B, T = kw['human_points'].shape[:2]
        gt_pose = kw['pose']
        gt_rotmats = axis_angle_to_rotation_matrix(
            gt_pose.reshape(-1, 3)).reshape(B, T, 24, 3, 3)

        gt_full_joints = kw['full_joints'].reshape(B, T, 24, 3)

        details = {}

        if 'pred_rotmats' in kw:
            # L_{\theta}
            pred_rotmats = kw['pred_rotmats'].reshape(B, T, 24, 3, 3)
            loss_param = self.criterion_param(pred_rotmats, gt_rotmats)
            details['loss_param'] = loss_param

            # L_{J_{SMPL}}
            pred_human_vertices = self.smpl(
                pred_rotmats.reshape(-1, 24, 3, 3), torch.zeros((B * T, 10)).cuda())
            pred_smpl_joints = self.smpl.get_full_joints(
                pred_human_vertices).reshape(B, T, 24, 3)
            loss_smpl_joints = self.criterion_joints(
                pred_smpl_joints, gt_full_joints)
            details['loss_smpl_joints'] = loss_smpl_joints


        if 'pred_full_joints' in kw:
            # L_{J}
            pred_full_joints = kw['pred_full_joints']
            loss_full_joints = self.criterion_joints(
                pred_full_joints, gt_full_joints)
            details['loss_full_joints'] = loss_full_joints

        loss = 0
        for _, v in details.items():
            loss += v
        details['loss'] = loss
        return loss, details

