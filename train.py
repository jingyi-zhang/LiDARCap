import argparse
import h5py
import metric
import numpy as np
import os
import torch
import wandb

from config import DATASET_DIR
from datasets.lidarcap_dataset import collate, TemporalDataset
from modules.geometry import rotation_matrix_to_axis_angle
from modules.regressor import Regressor
from modules.loss import Loss
from tools import common, crafter, multiprocess
from tools.util import save_smpl_ply
from tqdm import tqdm
torch.set_num_threads(1)

class MyTrainer(crafter.Trainer):
    def forward_backward(self, inputs):
        output = self.net(inputs)
        loss, details = self.loss_func(**output)
        loss.backward()
        return details

    def forward_val(self, inputs):
        output = self.net(inputs)
        loss, details = self.loss_func(**output)
        return details

    def forward_net(self, inputs):
        output = self.net(inputs)
        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # bs
    parser.add_argument('--bs', type=int, default=8,
                        help='input batch size for training (default: 24)')
    parser.add_argument('--eval_bs', type=int, default=16,
                        help='input batch size for evaluation')
    # threads
    parser.add_argument('--threads', type=int, default=4,
                        help='Number of threads (default: 4)')
    # gpu
    parser.add_argument('--gpu', type=int,
                        default=[0], help='-1 for CPU', nargs='+')
    # lr
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    # epochs
    parser.add_argument('--epochs', type=int, default=200,
                        help='Traning epochs (default: 200)')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Traning epochs (default: 100)')
    # dataset
    parser.add_argument("--dataset", type=str, required=True)
    # debug
    parser.add_argument('--debug', action='store_true', help='For debug mode')
    # eval or visual
    parser.add_argument('--eval', default=False, action='store_true',
                        help='evaluation the trained model')

    parser.add_argument('--visual', default=False, action='store_true',
                        help='visualization the result ply')

    # extra things, ignored
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='the saved ckpt needed to be evaluated or visualized')

    # wandb
    parser.add_argument('--project', type=str, default='test-project')
    parser.add_argument('--entity', type=str, default='lidar_human')

    args = parser.parse_args()

    if args.debug:
        os.environ['WANDB_MODE'] = 'dryrun'
    # else:
    #     common.login()
    wandb.init(project=args.project, entity=args.entity)
    wandb.config.update(args, allow_val_change=True)
    config = wandb.config

    iscuda = common.torch_set_gpu(config.gpu)
    common.make_reproducible(iscuda, 0)
    wandbid = [x for x in wandb.run.dir.split('/') if wandb.run.id in x][-1]

    # model save models in training
    model_dir = os.path.join('output', wandbid, 'model')
    os.makedirs(model_dir, exist_ok=True)

    dataset_name = args.dataset
    from yacs.config import CfgNode
    cfg = CfgNode.load_cfg(open('base.yaml'))
    # Load training and validation data
    if args.eval:
        test_dataset = TemporalDataset(cfg.TestDataset)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=config.eval_bs, num_workers=config.threads, pin_memory=True, collate_fn=collate)
        loader = {'Test': test_loader}

    else:
        train_dataset = TemporalDataset(dataset=dataset_name, train=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.bs, shuffle=True, num_workers=config.threads, pin_memory=True, collate_fn=collate)
        valid_dataset = TemporalDataset(dataset=dataset_name, train=False)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=config.bs, shuffle=False, num_workers=config.threads, pin_memory=True, collate_fn=collate)
        loader = {'Train': train_loader, 'Valid': valid_loader}

    net = Regressor()
    loss = Loss()

    if args.ckpt_path is not None:
        save_model = torch.load(args.ckpt_path)['state_dict']
        model_dict = net.state_dict()
        state_dict = {k: v for k, v in save_model.items()
                      if k in model_dict.keys()}
        model_dict.update(state_dict)
        net.load_state_dict(model_dict)

    # Define optimizer
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad],
                                 lr=config.lr, weight_decay=1e-4)
    sc = {'factor': 0.9, 'patience': 1, 'threshold': 0.01, 'min_lr': 0.00000003}
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=sc['factor'], patience=sc['patience'],
        threshold_mode='rel', threshold=sc['threshold'], min_lr=sc['min_lr'],
        verbose=True)

    # Instance tr
    wandbid = args.ckpt_path.split('.')[0] if args.eval else None
    train = MyTrainer(net, loader, loss, optimizer)
    if iscuda:
        train = train.cuda()

    if args.eval:
        if args.visual:

            visual_dir = os.path.join('visual', wandbid, dataset_name)
            os.makedirs(visual_dir, exist_ok=True)
            final_loss, pred_rotmats, pred_vertices = train(
                epoch=1, train=False, test=True, visual=True)
            n = len(pred_vertices)
            filenames = [os.path.join(
                visual_dir, '{}.ply'.format(i + 1)) for i in range(n)]
            multiprocess.multi_func(save_smpl_ply, 32, len(
                pred_vertices), 'saving ply', False, pred_vertices, filenames)

        else:
            final_loss, pred_rotmats = train(
                epoch=1, train=False, visual=False, test=True)
        print('EVAL LOSS', final_loss['loss'])

        pred_poses = []
        for pred_rotmat in tqdm(pred_rotmats):
            pred_poses.append(rotation_matrix_to_axis_angle(torch.from_numpy(pred_rotmat.reshape(-1, 3, 3))).numpy().reshape((-1, 72)))
        pred_poses = np.stack(pred_poses)

        test_dataset_filename = os.path.join(
            DATASET_DIR, '{}_test.hdf5'.format(dataset_name))
        test_data = h5py.File(test_dataset_filename, 'r')
        gt_poses = test_data['pose'][:]
        metric.output_metric(pred_poses.reshape(-1, 72), gt_poses.reshape(-1, 72))

    else:
        # Training loop
        mintloss = float('inf')
        minvloss = float('inf')
        for epoch in range(1, config.epochs + 1):
            print('')

            train_loss_dict = train(epoch)
            val_loss_dict = train(epoch, train=False)
            train_loss_log = {'train' + k: v for k,
                              v in train_loss_dict.items()}
            val_loss_log = {'val' + k: v for k, v in val_loss_dict.items()}
            epoch_logs = train_loss_log
            epoch_logs.update(val_loss_log)
            wandb.log(epoch_logs)

            # save model in this epoch
            # if this model is better, then save it as best
            if train_loss_dict['loss'] <= mintloss:
                mintloss = train_loss_dict['loss']
                best_save = os.path.join(model_dir, 'best-train-loss.pth')
                torch.save({'state_dict': net.state_dict()}, best_save)
                common.hint(f"Saving best train loss model at epoch {epoch}")
            if val_loss_dict['loss'] <= minvloss:
                minvloss = val_loss_dict['loss']
                best_save = os.path.join(model_dir, 'best-valid-loss.pth')
                torch.save({'state_dict': net.state_dict()}, best_save)
                common.hint(f"Saving best valid loss model at epoch {epoch}")

            common.clean_summary(wandb.run.summary)
            wandb.run.summary["best_train_loss"] = mintloss
            wandb.run.summary["best_valid_loss"] = minvloss

            # scheduler
            scheduler.step(train_loss_dict['loss'])
