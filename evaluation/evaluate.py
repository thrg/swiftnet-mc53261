import contextlib

import numpy as np
import torch
from tqdm import tqdm
from time import perf_counter
from sklearn.metrics import average_precision_score, roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay
import math
import matplotlib.pyplot as plt

__all__ = ['compute_errors', 'get_pred', 'evaluate_semseg', 'evaluate_anomaly']


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


def max_softmax(logits_data):
    softmax_tensor = torch.nn.functional.softmax(logits_data, dim=1)
    score_tensor = 1 - torch.max(softmax_tensor, dim=1).values
    return score_tensor


def max_logit(logits_data):
    score_tensor = -torch.max(logits_data, dim=1).values
    return score_tensor


def entropy(logits_data):
    probs = torch.nn.functional.softmax(logits_data, dim=1)
    print(probs)
    score_tensor = -(probs * torch.log(probs) / math.log(19)).sum(dim=1)
    return score_tensor


def evaluate_anomaly(model, data_loader):
    model.eval()
    managers = [torch.no_grad()]
    softmax_ap = []
    softmax_auroc = []
    logit_ap = []
    logit_auroc = []
    entropy_ap = []
    entropy_auroc = []

    softmax_scores = []
    logit_scores = []
    entropy_scores = []

    with contextlib.ExitStack() as stack:
        for ctx_mgr in managers:
            stack.enter_context(ctx_mgr)
        for step, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            gt = batch['original_labels'].numpy().astype(np.uint32)
            new_gt = gt[gt != 2]
            logits, additional = model.do_forward(batch, batch['original_labels'].shape[1:3])

            score = max_softmax(logits.data).cpu().numpy()
            score = score[gt != 2]
            softmax_scores.extend(score.tolist())
            if 0 in new_gt and 1 in new_gt:
                softmax_ap.append(average_precision_score(new_gt, score))
                roc_display = RocCurveDisplay.from_predictions(new_gt, score)
                roc_display.plot()
                plt.savefig(f"images/auroc_softmax_{step}")
                plt.close()
                ap_display = PrecisionRecallDisplay.from_predictions(new_gt, score)
                ap_display.plot()
                plt.savefig(f"images/ap_softmax_{step}")
                plt.close()
                softmax_auroc.append(roc_auc_score(new_gt, score))

            score = max_logit(logits.data).cpu().numpy()
            score = score[gt != 2]
            logit_scores.extend(score.tolist())
            if 0 in new_gt and 1 in new_gt:
                logit_ap.append(average_precision_score(new_gt, score))
                roc_display = RocCurveDisplay.from_predictions(new_gt, score)
                roc_display.plot()
                plt.savefig(f"images/auroc_logit_{step}")
                plt.close()
                ap_display = PrecisionRecallDisplay.from_predictions(new_gt, score)
                ap_display.plot()
                plt.savefig(f"images/ap_logit_{step}")
                plt.close()
                logit_auroc.append(roc_auc_score(new_gt, score))

            score = entropy(logits.data).cpu().numpy()
            score = score[gt != 2]
            entropy_scores.extend(score.tolist())
            if 0 in new_gt and 1 in new_gt:
                entropy_ap.append(average_precision_score(new_gt, score))
                roc_display = RocCurveDisplay.from_predictions(new_gt, score)
                roc_display.plot()
                plt.savefig(f"images/auroc_entropy_{step}")
                plt.close()
                ap_display = PrecisionRecallDisplay.from_predictions(new_gt, score)
                ap_display.plot()
                plt.savefig(f"images/ap_entropy_{step}")
                plt.close()
                entropy_auroc.append(roc_auc_score(new_gt, score))

        print('')
    model.train()

    softmax_scores = np.array(softmax_scores)
    logit_scores = np.array(logit_scores)
    entropy_scores = np.array(entropy_scores)

    plt.hist(softmax_scores)
    plt.xlabel('Vrijednost anomalije piksela')
    plt.ylabel('Broj piksela')
    plt.savefig(f"images/hist_softmax_anomaly")
    plt.close()

    plt.hist(logit_scores)
    plt.xlabel('Vrijednost anomalije piksela')
    plt.ylabel('Broj piksela')
    plt.savefig(f"images/hist_logit_anomaly")
    plt.close()

    plt.hist(entropy_scores)
    plt.xlabel('Vrijednost anomalije piksela')
    plt.ylabel('Broj piksela')
    plt.savefig(f"images/hist_entropy_anomaly")
    plt.close()

    softmax_ap = np.array(softmax_ap)
    softmax_auroc = np.array(softmax_auroc)
    logit_ap = np.array(logit_ap)
    logit_auroc = np.array(logit_auroc)
    entropy_ap = np.array(entropy_ap)
    entropy_auroc = np.array(entropy_auroc)

    softmax_mean_ap = np.mean(softmax_ap)
    softmax_mean_auroc = np.mean(softmax_auroc)
    logit_mean_ap = np.mean(logit_ap)
    logit_mean_auroc = np.mean(logit_auroc)
    entropy_mean_ap = np.mean(entropy_ap)
    entropy_mean_auroc = np.mean(entropy_auroc)

    return softmax_mean_ap, softmax_mean_auroc, logit_mean_ap, logit_mean_auroc, entropy_mean_ap, entropy_mean_auroc


def evaluate_semseg(model, data_loader, class_info, observers=()):
    model.eval()
    managers = [torch.no_grad()] + list(observers)

    bins = np.linspace(0, 1, 20)
    logit_bins = np.linspace(-100, 100, 20)
    s_OD_h_total = np.zeros((1, 19))
    l_OD_h_total = np.zeros((1, 19))
    e_OD_h_total = np.zeros((1, 19))
    # s_OD_h_total = np.zeros(20)
    # l_OD_h_total = np.zeros(20)
    # e_OD_h_total = np.zeros(20)

    with contextlib.ExitStack() as stack:
        for ctx_mgr in managers:
            stack.enter_context(ctx_mgr)
        conf_mat = np.zeros((model.num_classes, model.num_classes), dtype=np.uint64)
        for step, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            batch['original_labels'] = batch['original_labels'].numpy().astype(np.uint32)
            logits, additional = model.do_forward(batch, batch['original_labels'].shape[1:3])
            pred = torch.argmax(logits.data, dim=1).byte().cpu().numpy().astype(np.uint32)

            # score = max_softmax(logits.data).cpu().numpy()
            # score = score[batch['original_labels'] != 2]
            # OD_h, _ = np.histogram(score, bins)
            # s_OD_h_total += OD_h
            #
            # score = max_logit(logits.data).cpu().numpy()
            # score = score[batch['original_labels'] != 2]
            # OD_h, _ = np.histogram(score, logit_bins)
            # l_OD_h_total += OD_h

            score = entropy(logits.data).cpu().numpy()
            score = score[batch['original_labels'] != 2]
            print(score)
            OD_h, _ = np.histogram(score, bins)
            print(OD_h)
            e_OD_h_total += OD_h

            # score = max_softmax(logits.data).cpu().numpy()
            # score = score[batch['original_labels'] != 2]
            # OD_h, _ = np.histogram(score, 20)
            # s_OD_h_total += OD_h
            #
            # score = max_logit(logits.data).cpu().numpy()
            # score = score[batch['original_labels'] != 2]
            # OD_h, _ = np.histogram(score, 20)
            # l_OD_h_total += OD_h
            #
            # score = entropy(logits.data).cpu().numpy()
            # score = score[batch['original_labels'] != 2]
            # OD_h, _ = np.histogram(score, 20)
            # e_OD_h_total += OD_h

            for o in observers:
                o(pred, batch, additional)
            calculate_conf_matrix(pred.flatten(), batch["original_labels"].flatten(), conf_mat, model.num_classes)
        print('')
        pixel_acc, iou_acc, recall, precision, _, per_class_iou = compute_errors(conf_mat, class_info, verbose=True)
    model.train()

    # print(s_OD_h_total[0])
    # plt.plot(bins[1:], s_OD_h_total[0])
    # plt.xlabel('Vrijednost anomalije piksela')
    # plt.ylabel('Broj piksela')
    # plt.savefig(f"images/hist_softmax_normal")
    # plt.close()
    #
    # print(l_OD_h_total[0])
    # plt.plot(logit_bins[1:], l_OD_h_total[0])
    # plt.xlabel('Vrijednost anomalije piksela')
    # plt.ylabel('Broj piksela')
    # plt.savefig(f"images/hist_logit_normal")
    # plt.close()

    print(e_OD_h_total[0])
    plt.plot(bins[1:], e_OD_h_total[0])
    plt.xlabel('Vrijednost anomalije piksela')
    plt.ylabel('Broj piksela')
    plt.savefig(f"images/hist_entropy_normal")
    plt.close()

    # plt.stairs(s_OD_h_total)
    # plt.xlabel('Vrijednost anomalije piksela')
    # plt.ylabel('Broj piksela')
    # plt.savefig(f"images/hist_softmax_normal")
    # plt.close()
    #
    # plt.stairs(l_OD_h_total)
    # plt.xlabel('Vrijednost anomalije piksela')
    # plt.ylabel('Broj piksela')
    # plt.savefig(f"images/hist_logit_normal")
    # plt.close()
    #
    # plt.stairs(e_OD_h_total)
    # plt.xlabel('Vrijednost anomalije piksela')
    # plt.ylabel('Broj piksela')
    # plt.savefig(f"images/hist_entropy_normal")
    # plt.close()

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
