import argparse
from pathlib import Path
import importlib.util
from evaluation import evaluate_semseg, evaluate_anomaly
import torch
import math


def import_module(path):
    spec = importlib.util.spec_from_file_location("module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def max_softmax(logits_data):
    softmax_tensor = torch.nn.functional.softmax(logits_data, dim=1)
    score_tensor = 1 - torch.max(softmax_tensor, dim=1).values
    return score_tensor


def max_logit(logits_data):
    score_tensor = -torch.max(logits_data, dim=1).values
    return score_tensor


def entropy(logits_data):
    probs = torch.nn.functional.softmax(logits_data, dim=1)
    score_tensor = -(probs * torch.log(probs) / math.log(19)).sum(dim=1)
    return score_tensor


parser = argparse.ArgumentParser(description='Detector train')
parser.add_argument('config', type=str, help='Path to configuration .py file')
parser.add_argument('--profile', dest='profile', action='store_true', help='Profile one forward pass')

if __name__ == '__main__':
    args = parser.parse_args()
    conf_path = Path(args.config)
    conf = import_module(args.config)

    class_info = conf.dataset_val.class_info

    model = conf.model.cuda()

    for loader, name in conf.eval_loaders:
        if name == "anomaly":
            softmax_ap, softmax_auroc = evaluate_anomaly(model, loader, max_softmax)
            logit_ap, logit_auroc = evaluate_anomaly(model, loader, max_logit)
            entropy_ap, entropy_auroc = evaluate_anomaly(model, loader, entropy)
            print('softmax: ')
            print(f'ap: {softmax_ap}')
            print(f'auroc: {softmax_auroc}')
            print('logit: ')
            print(f'ap: {logit_ap}')
            print(f'auroc: {logit_auroc}')
            print('entropy: ')
            print(f'ap: {entropy_ap}')
            print(f'auroc: {entropy_auroc}')
        else:
            iou, per_class_iou = evaluate_semseg(model, loader, class_info, observers=conf.eval_observers)
            print(f'{name}: {iou:.2f}')
