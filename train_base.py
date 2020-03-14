import hydra
from omegaconf import DictConfig
import logging

import torch
from torch.optim import SGD, lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from utils import cal_parameters, get_dataset, AverageMeter

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

    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('Acc')

    for batch_idx, (x, y) in enumerate(data_loader):
        x, y = x.to(args.device), y.to(args.device)

        # real forward
        logits = classifier(x)
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


@hydra.main(config_path='configs/base_config.yaml')
def run(args: DictConfig) -> None:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    classifier = eval(args.classifier_name)(args.width, args.n_classes).to(args.device)
    logger.info('Classifier: {}, width: {}, # parameters: {}'
                .format(args.classifier_name, args.width, cal_parameters(classifier)))

    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    train_data = get_dataset(data_name=args.dataset, data_dir=data_dir, train=True, crop_flip=True)
    test_data = get_dataset(data_name=args.dataset, data_dir=data_dir, train=False, crop_flip=False)

    test_loader = DataLoader(dataset=test_data, batch_size=args.n_batch_test, shuffle=False)
    n = len(train_data)
    split_size = n // args.n_split
    lengths = [split_size] * (args.n_split - 1) + [n % split_size + split_size]
    datasets_list = random_split(train_data, lengths=lengths)

    for split_id, dataset in enumerate(datasets_list):
        checkpint = '{}_w{}_split{}.pth'.format(args.classifier_name, args.width, split_id)
        logger.info('Running on subset {}, size: {}'.format(split_id + 1, len(dataset)))
        train_loader = DataLoader(dataset=dataset, batch_size=args.n_batch_train, shuffle=True)

        if args.inference is True:
            classifier.load_state_dict(torch.load(checkpint))
            logger.info('Load classifier from checkpoint')
        else:
            optimizer = SGD(classifier.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            optimal_loss = 1e5
            for epoch in range(1, args.n_epochs + 1):
                if epoch in args.schedule:
                    args.lr *= args.gamma
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = args.lr

                loss, acc = train_epoch(classifier, train_loader, args, optimizer)
                if loss < optimal_loss:
                    optimal_loss = loss
                    torch.save(classifier.state_dict(), checkpint)
                logger.info('Epoch {}, loss: {:.4f}, acc: {:.4f}'.format(epoch,loss, acc))

        clean_loss, clean_acc = eval_epoch(classifier, test_loader, args, adversarial=False)
        adv_loss, adv_acc = eval_epoch(classifier, test_loader, args, adversarial=True)
        logger.info('Clean loss: {:.4f}, acc: {:.4f}'.format(clean_loss, clean_acc))
        logger.info('Adversarial loss: {:.4f}, acc: {:.4f}'.format(adv_loss, adv_acc))


if __name__ == '__main__':
    run()
