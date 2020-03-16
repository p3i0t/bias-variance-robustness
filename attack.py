import hydra
from omegaconf import DictConfig

import os

import torch
import utils
import torch.nn.functional as F
from utils import AverageMeter


def clamp(X, clip_min, clip_max):
    return torch.max(torch.min(X, clip_max), clip_min)


def attack_pgd(model, x, y, eps, eps_iter, attack_iters, restarts):
    """
    Perform PGD attack on one mini-batch.
    :param model: pytorch model.
    :param x: x of minibatch.
    :param y: y of minibatch.
    :param eps: L-infinite norm budget.
    :param eps_iter: step size for each iteration.
    :param attack_iters: number of iterations.
    :param restarts:  number of restart times
    :return: best adversarial perturbations delta in all restarts
    """
    assert x.device == y.device
    max_loss = torch.zeros_like(y).float()
    max_delta = torch.zeros_like(x)

    for i in range(restarts):
        delta = torch.zeros_like(x).uniform_(-eps, eps)
        delta.data = clamp(delta, utils.clip_min - x, utils.clip_max - x)
        delta.requires_grad = True

        for _ in range(attack_iters):
            logits = model(x + delta)
            # index = torch.where(output.max(1)[1] == y)
            index = torch.where(logits.argmax(dim=1) == y)  # get the correct predictions, pgd performed only on them
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(logits, y)
            loss.backward()

            # select & update
            grad = delta.grad.detach()
            delta_update = (delta[index] + eps_iter * torch.sign(grad[index])).clamp_(-eps, eps)
            delta_update = clamp(delta_update, utils.clip_min - x[index], utils.clip_max - x[index])

            # write back
            delta.data[index] = delta_update
            delta.grad.zero_()

        all_loss = F.cross_entropy(model(x + delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def eval_epoch(model, data_loader, args, adversarial=False, save=False):
    """Self-implemented PGD evaluation"""
    eps = eval(args.epsilon) / utils.std
    eps_iter = eval(args.pgd_epsilon_iter) / utils.std
    attack_iters = 40
    restarts = 1

    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('Acc')
    model.eval()
    adv_list = []
    for i, (x, y) in enumerate(data_loader):
        x, y = x.to(args.device), y.to(args.device)
        if adversarial is True:
            delta = attack_pgd(model, x, y, eps, eps_iter, attack_iters, restarts)
        else:
            delta = 0.

        with torch.no_grad():
            adv_x = x + delta
            logits = model(adv_x)
            if save is True:
                adv_list.append(adv_x)
            loss = F.cross_entropy(logits, y)

            loss_meter.update(loss.item(), x.size(0))
            acc = (logits.argmax(dim=1) == y).float().mean().item()
            acc_meter.update(acc, x.size(0))

    if save is True:
        save_dir = hydra.utils.to_absolute_path(args.data_dir)
        adv_set = torch.cat(adv_list, dim=0)
        torch.save(adv_set, os.path.join(save_dir, 'advset_{}_{}.pt'.format(args.classifier_name, args.model_type)))
    return loss_meter.avg, acc_meter.avg

