from tqdm import tqdm
from collections import defaultdict

import wandb
import os
import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def mean(lis): return sum(lis) / len(lis)


class Crafter(nn.Module):
    """ Helper class to train/valid a deep network.
        Overload this class `forward_backward` for your actual needs.

    Usage: 
        train/valid = Trainer/Valider(net, loader, loss, optimizer)
        for epoch in range(n_epochs):
            train()/valid()
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def iscuda(self):
        return next(self.net.parameters()).device != torch.device('cpu')

    def todevice(self, x):
        if isinstance(x, dict):
            return {k: self.todevice(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return [self.todevice(v) for v in x]

        if self.iscuda():
            if isinstance(x, str):
                return x
            else:
                return x.contiguous().cuda(non_blocking=True)
        else:
            return x.cpu()

    def __call__(self, epoch):
        raise NotImplementedError()




class Trainer(Crafter):
    def __init__(self, net, loader, loss, optimizer):
        Crafter.__init__(self, net)
        self.loader = loader
        self.loss_func = loss
        self.optimizer = optimizer

    def __call__(self, epoch, train=True, test=False, visual=False):

        if train:
            self.net.train()
            key = 'Train'
        elif test:
            self.net.eval()
            key = 'Test'
        else:
            self.net.eval()
            key = 'Valid'

        stats = defaultdict(list)

        loader = self.loader[key]
        # loader.dataset.epoch += 1

        bar = tqdm(loader, bar_format="{l_bar}{bar:3}{r_bar}", ncols=110)

        if test:
            rotmats = []
            vertices = []
            from modules.smpl import SMPL, get_smpl_vertices
            smpl = SMPL().cuda()
            for bi, inputs in enumerate(bar):
                inputs = self.todevice(inputs)
                output = self.forward_net(inputs)
                _, details = self.loss_func(**output)
                pred_rotmats = output['pred_rotmats']
                B, T = pred_rotmats.shape[:2]
                rotmats.append(pred_rotmats.cpu().detach().numpy())

                for k, v in details.items():
                    if type(v) is not dict:
                        if isinstance(v, torch.Tensor):
                            stats[k].append(v.detach().cpu().numpy())
                        else:
                            stats[k].append(v)
                if visual:
                    pred_vertices = get_smpl_vertices(output['trans'].reshape(B * T, 3), pred_rotmats.reshape(B * T, 24, 3, 3), output['betas'].reshape(B * T, 10), smpl)
                    for index in range(pred_vertices.shape[0]):
                        vertices.append(
                            pred_vertices[index].squeeze().cpu().detach().numpy())
                #del inputs, details

            rotmats = np.concatenate(rotmats, axis=0)
            final_loss = {k: mean(v) for k, v in stats.items()}

            if visual:
                return final_loss, rotmats, vertices
            return final_loss, rotmats

        bar.set_description(f'{key} {epoch:02d}')
        for iter, inputs in enumerate(bar):
            inputs = self.todevice(inputs)
            # compute gradient and do model update
            if train:
                self.optimizer.zero_grad()
                details = self.forward_backward(inputs)
                self.optimizer.step()
            else:
                details = self.forward_val(inputs)

            for k, v in details.items():
                if type(v) is not dict:
                    if isinstance(v, torch.Tensor):
                        stats[k].append(v.detach().cpu().numpy())
                    else:
                        stats[k].append(v)
            # if is training, print median stats on terminal
            if train:
                N = len(stats['loss']) // 10 + 1
                loss = stats['loss']
                bar.set_postfix(loss=f'{mean(loss[:N]):06.06f} -> '
                                        f'{mean(loss[-N:]):06.06f} '
                                        f'({mean(loss):06.06f})')

            if not train and (iter + 1) == len(loader):
                bar.set_postfix(loss=f'{mean(stats["loss"]):06.06f}')

            first_step = epoch == 1 and iter == 0 and train
            log_step = (
                iter + 1) % wandb.config.log_interval == 0 and iter != 0 and train
            if first_step or log_step:
                # Use trained image numbers as step
                step = (epoch - 1) * len(loader.dataset) \
                    + (iter + 1) * loader.batch_size
                logs = {k: mean(v) for k, v in stats.items()}
                logs['lr'] = self.optimizer.param_groups[0]['lr']
                logs['global_step'] = step
                wandb.log(logs)
                # del logs
            # del inputs, details


        final_loss = {k: mean(v) for k, v in stats.items()}

        return final_loss

    def forward_backward(self, inputs):
        raise NotImplementedError()

    def forward_net(self, inputs):
        raise NotImplementedError()

    def forward_val(self, inputs):
        raise NotImplementedError()
