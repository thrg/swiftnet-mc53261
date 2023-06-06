import contextlib

import numpy as np
import torch
from tqdm import tqdm
from time import perf_counter
import math

__all__ = ['compute_errors', 'get_pred', 'evaluate_semseg']


def compute_errors(conf_mat, class_info, verbose=True):
    num_correct = conf_mat.trace()
    num_classes = conf_mat.shape[0]
    total_size = conf_mat.sum()
    avg_pixel_acc = num_correct / total_size * 100.0
    TPFP = conf_mat.sum(1)
    TPFN = conf_mat.sum(0)
    FN = TPFN - conf_mat.diagonal()
    FP = TPFP - conf_mat.diagonal()
    class_iou = np.zeros(num_classes)
    class_recall = np.zeros(num_classes)
    class_precision = np.zeros(num_classes)
    per_class_iou = []
    if verbose:
        print('Errors:')
    for i in range(num_classes):
        TP = conf_mat[i, i]
        class_iou[i] = (TP / (TP + FP[i] + FN[i])) * 100.0
        if TPFN[i] > 0:
            class_recall[i] = (TP / TPFN[i]) * 100.0
        else:
            class_recall[i] = 0
        if TPFP[i] > 0:
            class_precision[i] = (TP / TPFP[i]) * 100.0
        else:
            class_precision[i] = 0

        class_name = class_info[i]
        per_class_iou += [(class_name, class_iou[i])]
        if verbose:
            print('\t%s IoU accuracy = %.2f %%' % (class_name, class_iou[i]))
    avg_class_iou = class_iou.mean()
    avg_class_recall = class_recall.mean()
    avg_class_precision = class_precision.mean()
    if verbose:
        print('IoU mean class accuracy -> TP / (TP+FN+FP) = %.2f %%' % avg_class_iou)
        print('mean class recall -> TP / (TP+FN) = %.2f %%' % avg_class_recall)
        print('mean class precision -> TP / (TP+FP) = %.2f %%' % avg_class_precision)
        print('pixel accuracy = %.2f %%' % avg_pixel_acc)
    return avg_pixel_acc, avg_class_iou, avg_class_recall, avg_class_precision, total_size, per_class_iou


def get_pred(logits, labels, conf_mat):
    _, pred = torch.max(logits.data, dim=1)
    pred = pred.byte().cpu()
    pred = pred.numpy().astype(np.int32)
    true = labels.numpy().astype(np.int32)
    calculate_conf_matrix(pred.reshape(-1), true.reshape(-1), conf_mat)


def mt(sync=False):
    if sync:
        torch.cuda.synchronize()
    return 1000 * perf_counter()


# def softmax(x):
#     """Compute softmax values for each sets of scores in x."""
#     if isinstance(x, list):
#         x = np.array(x)
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum()


def max_softmax(logits_data, threshold):
    usual_pred = torch.argmax(logits_data, dim=1)

    softmax_tensor = torch.nn.functional.softmax(logits_data, dim=1)
    max_tensor = torch.max(softmax_tensor, dim=1).values
    anomaly_tensor = (max_tensor < threshold).float()

    for i, img in enumerate(usual_pred):
        for j, row in enumerate(img):
            for k, pixel in enumerate(row):
                if pixel == 2 and anomaly_tensor[i][j][k] == 0:
                    anomaly_tensor[i][j][k] = 2.0

    return anomaly_tensor


def max_logit(logits_data, threshold):
    usual_pred = torch.argmax(logits_data, dim=1)

    max_tensor = torch.max(logits_data, dim=1).values
    anomaly_tensor = (max_tensor < threshold).float()

    for i, img in enumerate(usual_pred):
        for j, row in enumerate(img):
            for k, pixel in enumerate(row):
                if pixel == 2 and anomaly_tensor[i][j][k] == 0:
                    anomaly_tensor[i][j][k] = 2.0

    return anomaly_tensor


def entropy(logits_data, threshold):
    usual_pred = torch.argmax(logits_data, dim=1)

    softmax_tensor = torch.nn.functional.softmax(logits_data, dim=1)
    max_data = torch.zeros(len(usual_pred), len(usual_pred[0]), len(usual_pred[0][0]))

    for i, img in enumerate(softmax_tensor):
        for j, pred_class in enumerate(img):
            for k, row in enumerate(pred_class):
                for l, pixel in enumerate(row):
                    max_data[i][k][l] -= pixel * math.log(pixel, 3)

    max_tensor = torch.FloatTensor(max_data)
    anomaly_tensor = (max_tensor > threshold).float()

    for i, img in enumerate(usual_pred):
        for j, row in enumerate(img):
            for k, pixel in enumerate(row):
                if pixel == 2 and anomaly_tensor[i][j][k] == 0:
                    anomaly_tensor[i][j][k] = 2.0

    return anomaly_tensor


def evaluate_semseg(model, data_loader, class_info, observers=(), anomaly_loader_type=None):
    model.eval()
    managers = [torch.no_grad()] + list(observers)
    with contextlib.ExitStack() as stack:
        for ctx_mgr in managers:
            stack.enter_context(ctx_mgr)
        conf_mat = np.zeros((model.num_classes, model.num_classes), dtype=np.uint64)
        for step, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            batch['original_labels'] = batch['original_labels'].numpy().astype(np.uint32)
            logits, additional = model.do_forward(batch, batch['original_labels'].shape[1:3])
            pred = torch.argmax(logits.data, dim=1).byte().cpu().numpy().astype(np.uint32)
            print(logits.data.shape)
            print(pred.shape)
            if anomaly_loader_type == "softmax":
                # pred = list(map(max_softmax, torch.unbind(logits.data, 0)))
                pred = max_softmax(logits.data, 0.5).byte().cpu().numpy().astype(np.uint32)
                print(pred.shape)
                # pred = torch.stack(pred, 0)
            elif anomaly_loader_type == "logit":
                pred = max_logit(logits.data, 0.5).byte().cpu().numpy().astype(np.uint32)
            elif anomaly_loader_type == "entropy":
                pred = entropy(logits.data, 0.5).byte().cpu().numpy().astype(np.uint32)
            else:
                pred = torch.argmax(logits.data, dim=1).byte().cpu().numpy().astype(np.uint32)
            for o in observers:
                o(pred, batch, additional)
            calculate_conf_matrix(pred.flatten(), batch["original_labels"].flatten(), conf_mat, model.num_classes)
        print('')
        pixel_acc, iou_acc, recall, precision, _, per_class_iou = compute_errors(conf_mat, class_info, verbose=True)
    model.train()
    return iou_acc, per_class_iou


def calculate_conf_matrix(predicted, target, conf_mat, num_classes=12):
    idx = target != 255
    target = target[idx]
    predicted = predicted[idx]
    x = predicted + num_classes * target
    if num_classes == 19:
        bincount_2d = np.bincount(
            x.astype(np.int32), minlength=num_classes ** 2
        )[:361]
    else:
        bincount_2d = np.bincount(
            x.astype(np.int32), minlength=num_classes ** 2
        )
    if bincount_2d.size != num_classes ** 2:
        print(bincount_2d.size, num_classes ** 2)
        import pdb; pdb.set_trace()
    conf = bincount_2d.reshape((num_classes, num_classes))
    conf_mat += conf.astype(np.uint64)
