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
            pass
            # softmax_ap, softmax_auroc, logit_ap, logit_auroc, entropy_ap, entropy_auroc = evaluate_anomaly(model, loader)
            # print('softmax: ')
            # print(f'ap: ', end="")
            # print(softmax_ap)
            # print(f'auroc: ', end="")
            # print(softmax_auroc)
            # print('logit: ')
            # print(f'ap: ', end="")
            # print(logit_ap)
            # print(f'auroc: ', end="")
            # print(logit_auroc)
            # print('entropy: ')
            # print(f'ap: ', end="")
            # print(entropy_ap)
            # print(f'auroc: ', end="")
            # print(entropy_auroc)
        else:
            iou, per_class_iou = evaluate_semseg(model, loader, class_info, observers=conf.eval_observers)
            print(f'{name}: {iou:.2f}')
            break
