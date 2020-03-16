import hydra
from omegaconf import DictConfig
import logging

import torch
from torch.optim import SGD, lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from utils import cal_parameters, get_dataset, AverageMeter
import utils

from models import resnet18
from attack import eval_epoch

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logger = logging.getLogger(__name__)


def clamp(X, clip_min, clip_max):
    return torch.max(torch.min(X, clip_max), clip_min)


def train_epoch(classifier, data_loader, args, optimizer, scheduler=None):
    """
    Run one epoch.
    :param classifier: torch.nn.Module representing the classifier.
    :param data_loader: dataloader
    :param args:
    :param optimizer:
    :param scheduler:
    :return: mean of loss, mean of accuracy of this epoch.
    """
    classifier.train()

    # ajust according to std.
    eps = eval(args.epsilon) / utils.std
    eps_iter = eval(args.epsilon_iter) / utils.std

    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('Acc')

    for batch_idx, (x, y) in enumerate(data_loader):
        x, y = x.to(args.device), y.to(args.device)
        # start with uniform noise
        delta = torch.zeros_like(x).uniform_(-eps, eps)
        delta.requires_grad_()
        delta = clamp(delta, utils.clip_min - x, utils.clip_max - x)

        loss = F.cross_entropy(classifier(x + delta), y)
        grad_delta = torch.autograd.grad(loss, delta)[0].detach()  # get grad of noise

        # update delta with grad
        delta = (delta + torch.sign(grad_delta) * eps_iter).clamp_(-eps, eps)
        delta = clamp(delta, utils.clip_min - x, utils.clip_max - x)

        # real forward
        logits = classifier(x + delta)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        loss_meter.update(loss.item(), x.size(0))
        acc = (logits.argmax(dim=1) == y).float().mean().item()
        acc_meter.update(acc, x.size(0))

    return loss_meter.avg, acc_meter.avg


@hydra.main(config_path='configs/fast_fgsm_config.yaml')
def run(args: DictConfig) -> None:
    # cuda_available = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # device = "cuda" if cuda_available and args.device == 'cuda' else "cpu"

    classifier = eval(args.classifier_name)(args.width, args.n_classes).to(args.device)
    logger.info('Classifier: {}, width: {}, # parameters: {}'
                .format(args.classifier_name, args.width, cal_parameters(classifier)))

    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    train_data = get_dataset(data_name=args.dataset, data_dir=data_dir, train=True, crop_flip=True)
    test_data = get_dataset(data_name=args.dataset, data_dir=data_dir, train=False, crop_flip=False)

    test_loader = DataLoader(dataset=test_data, batch_size=args.n_batch_test, shuffle=False)

    optimizer = SGD(classifier.parameters(), lr=args.lr_max,
                    momentum=args.momentum, weight_decay=args.weight_decay)

    def run_forward(scheduler):
        optimal_loss = 1e5
        for epoch in range(1, args.n_epochs + 1):
            loss, acc = train_epoch(classifier, train_loader, args, optimizer, scheduler=scheduler)
            if loss < optimal_loss:
                optimal_loss = loss
                torch.save(classifier.state_dict(), checkpoint)
            logger.info('Epoch {}, lr: {:.4f}, loss: {:.4f}, acc: {:.4f}'.format(epoch, scheduler.get_lr()[0], loss, acc))

    if args.adv_generation:
        checkpoint = '{}_w{}_at_fast.pth'.format(args.classifier_name, args.width)
        train_loader = DataLoader(dataset=train_data, batch_size=args.n_batch_train, shuffle=True)
        lr_steps = args.n_epochs * len(train_loader)
        scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=args.lr_min, max_lr=args.lr_max,
                                          step_size_up=lr_steps/2, step_size_down=lr_steps/2)

        run_forward(scheduler)

        clean_loss, clean_acc = eval_epoch(classifier, test_loader, args, adversarial=False)
        adv_loss, adv_acc = eval_epoch(classifier, test_loader, args, adversarial=True, save=True)
        logger.info('Clean loss: {:.4f}, acc: {:.4f}'.format(clean_loss, clean_acc))
        logger.info('Adversarial loss: {:.4f}, acc: {:.4f}'.format(adv_loss, adv_acc))

    else:
        n = len(train_data)
        split_size = n // args.n_split
        lengths = [split_size] * (args.n_split - 1) + [n % split_size + split_size]
        datasets_list = random_split(train_data, lengths=lengths)

        for split_id, dataset in enumerate(datasets_list):
            checkpoint = '{}_w{}_split{}_at_fast.pth'.format(args.classifier_name, args.width, split_id)
            logger.info('Running on subset {}, size: {}'.format(split_id + 1, len(dataset)))
            train_loader = DataLoader(dataset=dataset, batch_size=args.n_batch_train, shuffle=True)

            lr_steps = args.n_epochs * len(train_loader)
            scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=args.lr_min, max_lr=args.lr_max,
                                              step_size_up=lr_steps/2, step_size_down=lr_steps/2)

            run_forward(scheduler)

            clean_loss, clean_acc = eval_epoch(classifier, test_loader, args, adversarial=False)
            adv_loss, adv_acc = eval_epoch(classifier, test_loader, args, adversarial=True)
            logger.info('Clean loss: {:.4f}, acc: {:.4f}'.format(clean_loss, clean_acc))
            logger.info('Adversarial loss: {:.4f}, acc: {:.4f}'.format(adv_loss, adv_acc))


if __name__ == '__main__':
    run()
