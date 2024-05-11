import numpy as np
import torch
import matplotlib.pyplot as plt

def remove_self_connection(gc):
    """
    gc: [d, d, t] or [d, d]
    """
    if 2 == len(gc.shape):
        gc = gc[:, :, np.newaxis]
    idx = np.array(tuple(np.ndindex(gc.shape)))
    idx.resize(gc.shape + (3,))
    flat = gc[idx[:, :, :, 0] != idx[: ,:, :, 1]]
    return flat

def two_classify_metrics(pred, true):
    """
    
    """
    with torch.no_grad():
        #
        if isinstance(pred, np.ndarray) and isinstance(true, np.ndarray):
            pos = np.sort(pred[np.where(true==1)])[::-1]
            neg = np.sort(pred[np.where(true==0)])[::-1]
        else:
            raise ValueError("Unaccepted value type: " + str(pred.__class__) + " and " + str(true.__class__))
        
        if (np.isnan(pred).any()):
            raise ValueError("pred gc contains nan")

        cur = np.inf
        if pos.shape[0]: cur = max(cur, pos[0]) # 
        if neg.shape[0]: cur = max(cur, neg[0])
        tp, fp, tn, fn = 0, 0, neg.shape[0], pos.shape[0]
        f1, acc, fpr, tpr = 2 * tp / (2 * tp + fp + fn), (tp + tn) / (tp + fp + tn + fn), fp / (tn + fp) ,tp / (tp + fn)
        precision, recall = tp / (tp + fp) if tp + fp else 1, tp / (tp + fn)
        roc_fpr, roc_tpr = [fpr], [tpr]
        prc_precision, prc_recall = [precision], [recall]
        best_f1, f1_trd = f1, cur
        best_acc, acc_trd = acc, cur

        i, j = 0, 0
        while i < pos.shape[0] or j < neg.shape[0]:
            upd = -np.inf
            if i < pos.shape[0]: upd = max(upd, pos[i])
            if j < neg.shape[0]: upd = max(upd, neg[j])
            cur = min(cur, upd)

            while i < pos.shape[0] and cur <= pos[i]: 
                i, tp, fn = i + 1, tp + 1, fn - 1
            while j < neg.shape[0] and cur <= neg[j]:
                j, tn, fp = j + 1, tn - 1, fp + 1

            f1, acc, fpr, tpr = 2 * tp / (2 * tp + fp + fn), (tp + tn) / (tp + fp + tn + fn), fp / (tn + fp) ,tp / (tp + fn)
            precision, recall = tp / (tp + fp) if tp + fp else 1, tp / (tp + fn)
            if f1 > best_f1: 
                best_f1, f1_trd = f1, cur
            if acc > best_acc:
                best_acc, acc_trd = acc, cur
            roc_fpr.append(fpr)
            roc_tpr.append(tpr)
            prc_precision.append(precision)
            prc_recall.append(recall)

        auroc = 0
        for (lx, ly, x, y) in zip(roc_fpr[:-1], roc_tpr[:-1], roc_fpr[1:], roc_tpr[1:]):
            auroc += 0.5 * (x - lx) * (ly + y)
        auprc = 0
        for (lx, ly, x, y) in zip(prc_recall[:-1], prc_precision[:-1], prc_recall[1:], prc_precision[1:]):
            auprc += 0.5 * (x - lx) * (ly + y)

        return (best_f1, f1_trd), (best_acc, acc_trd), (auroc, roc_fpr, roc_tpr), (auprc, prc_recall, prc_precision)
