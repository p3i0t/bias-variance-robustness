import hydra
from omegaconf import DictConfig
import logging
from functools import reduce

import torch
from torch.optim import SGD, lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
from utils import cal_parameters, get_dataset, AverageMeter
import utils

from models import resnet18
from attack import eval_epoch

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logger = logging.getLogger(__name__)


def kl_discrete(p1, p2):
    return (p1 * torch.log(p1 / p2)).sum(dim=1)


def eval_risk_bias_variance(classifier_list, dataloader, args):
    risk_meter = AverageMeter('risk')
    variance_meter = AverageMeter('variance')
    bias_meter = AverageMeter('bias')
    acc_meter = AverageMeter('acc')

    for batch_id, (x, y) in enumerate(dataloader):
        x, y = x.to(args.device), y.to(args.device)
        logits_list = [classifier(x) for classifier in classifier_list]
        acc_list = [(logits.argmax(dim=1) == y).float().mean().item() for logits in logits_list]
        risk_list = [F.cross_entropy(logits, y) for logits in logits_list]
        mean_risk = reduce(lambda a, b: a + b, risk_list) / len(risk_list)

        pi = reduce(lambda a, b: a + b, logits_list) / len(logits_list)  # average of log-probs, i.e. logts
        pi = pi.softmax(dim=1)  # normalization with softmax

        kl_list = [kl_discrete(pi, logits.softmax(dim=1)) for logits in logits_list]
        mean_variance = reduce(lambda a, b: a + b, kl_list) / len(kl_list)

        mean_bias = mean_risk - mean_variance

        risk_meter.update(mean_risk.mean().item(), x.size(0))
        variance_meter.update(mean_variance.mean().item(), x.size(0))
        bias_meter.update(mean_bias.mean().item(), x.size(0))
        acc_meter.update(sum(acc_list)/len(acc_list), x.size(0))
    return risk_meter.avg, bias_meter.avg, variance_meter.avg, acc_meter


@hydra.main(config_path='configs/eval_config.yaml')
def run(args: DictConfig) -> None:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    data_dir = hydra.utils.to_absolute_path(args.data_dir)

    clean_test_data = get_dataset(data_name=args.dataset, data_dir=data_dir, train=False, crop_flip=False)
    #advset_at = TensorDataset(torch.load(os.path.join(data_dir, 'advset_{}_at_fast.pt'.format(args.classifier_name))))
    advset_clean = TensorDataset(torch.load(os.path.join(data_dir, 'advset_{}_clean.pt'.format(args.classifier_name))))

    clean_loader = DataLoader(dataset=clean_test_data, batch_size=args.n_batch_test, shuffle=False)
    advset_loader = DataLoader(dataset=advset_clean, batch_size=args.n_batch_test, shuffle=False)

    results_dict = dict()
    for width in args.width_list:
        classifier_list = []
        for split_id in range(args.n_split):
            classifier = eval(args.classifier_name)(width, args.n_classes).to(args.device)
            logger.info('Classifier: {}, width: {}, # parameters: {}'
                        .format(args.classifier_name, width, cal_parameters(classifier)))
            checkpoint = '{}_w{}_split{}.pth'.format(args.classifier_name, width, split_id)
            classifier.load_state_dict(torch.load(checkpoint))
            classifier_list.append(classifier)

        results_dict['clean_on_clean'] = eval_risk_bias_variance(classifier_list, clean_loader, args)
        results_dict['clean_on_adv'] = eval_risk_bias_variance(classifier_list, advset_loader, args)

        del classifier_list

        classifier_list = []
        for split_id in range(args.n_split):
            classifier = eval(args.classifier_name)(width, args.n_classes).to(args.device)
            logger.info('Classifier: {}, width: {}, # parameters: {}'
                        .format(args.classifier_name, width, cal_parameters(classifier)))
            checkpoint = '{}_w{}_split{}_at_fast.pth'.format(args.classifier_name, width, split_id)
            classifier.load_state_dict(torch.load(checkpoint))
            classifier_list.append(classifier)

        results_dict['adv_on_clean'] = eval_risk_bias_variance(classifier_list, clean_loader, args)
        results_dict['adv_on_adv'] = eval_risk_bias_variance(classifier_list, advset_loader, args)

    torch.save(results_dict, 'adv_eval_width_results.pt')



