from tools import util
import metric


def eval(name):
    for idx in [7, 24, 29, 41]:
        pred_poses = util.get_pred_poses(name, idx)
        gt_poses = util.get_gt_poses(idx)
        pred_poses = pred_poses[:len(gt_poses)]
        metric.output_metric(pred_poses, gt_poses)


if __name__ == '__main__':
    name = input()
    eval(name)
