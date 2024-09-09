import torch
import numpy as np
from sklearn.metrics import (
    r2_score,
    matthews_corrcoef,
    roc_auc_score,
    accuracy_score,
    precision_score,
    root_mean_squared_error,
)
from sklearn.metrics import ndcg_score as _ndcg_score
from scipy.stats import pearsonr as _pearsonr, spearmanr as _spearmanr, rankdata
from functools import partial
from models import MLPHead, ABYSSALModel


def train_epoch_baseline(model, optimizer, criterion, train_loader, device):
    total_loss = 0
    preds = []
    targets = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        targets.append(target)
        preds.append(output)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"batch {batch_idx} loss: {loss.item()}")

    avg_loss = total_loss / len(train_loader)
    return avg_loss, torch.cat(targets), torch.cat(preds)


def train_epoch(model, optimizer, criterion, train_loader, device):
    total_loss = 0
    preds = []
    targets = []
    for batch_idx, (data, target, seq_lens) in enumerate(train_loader):
        # print(seq_lens)

        wt, mt = data
        padded_seq_len = wt["input_ids"].shape[1]
        wt = wt.to(device)
        mt = mt.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        mask = (torch.arange(padded_seq_len)[None, :] < seq_lens[:, None]).to(device)

        output = model(wt, mt, mask)
        targets.append(target)
        preds.append(output)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"batch {batch_idx} loss: {loss.item()}")

    avg_loss = total_loss / len(train_loader)
    return avg_loss, torch.cat(targets), torch.cat(preds)


def val_epoch(model, criterion, test_loader, device):
    all_outputs = []
    targets = []
    total_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            all_outputs.append(output)
            targets.append(target)
    avg_loss = total_loss / len(test_loader)
    return avg_loss, torch.cat(targets), torch.cat(all_outputs)


# Pearson correlation
def pearsonr(true, pred):
    return _pearsonr(true, pred)[0]


# Spearman rank correlation
def spearmanr(true, pred):
    return _spearmanr(true, pred)[0]


# Stability-based Spearman correlation
def stab_spearman(true, pred, THRESHOLD):
    mask = true < THRESHOLD
    stab_spearman = _spearmanr(true[mask], pred[mask])[0]
    return stab_spearman


# Normalized Discounted Cumulative Gain (NDCG)
def ndcg_score(true, pred, K=None):
    true_ranking = rankdata(-true, method="min")
    return _ndcg_score([true_ranking], [-pred], k=K)


# Detection precision score
def det_precision_score(true, pred, THRESHOLD, K=None):
    if K is None:
        ids = np.argpartition(pred, -K)[:K]
        sample_weight = [1.0 if i in ids else 0.0 for i in range(len(true))]
    else:
        sample_weight = None
    true = true < THRESHOLD
    pred = pred < THRESHOLD
    detpr = precision_score(true, pred, sample_weight=sample_weight)
    return detpr


# Updated compute_metrics function
def compute_metrics(true, pred, THRESHOLD=-0.5, K=30):
    metrics = dict()

    regression_metrics = {
        "R2": r2_score,
        "RMSE": root_mean_squared_error,
        "Pearson": pearsonr,
        "Spearman": spearmanr,
        "StabSpearman": partial(stab_spearman, THRESHOLD=THRESHOLD),
    }

    classification_metrics = {
        "MCC": matthews_corrcoef,
        "AUC": roc_auc_score,
        "ACC": accuracy_score,
    }

    other_metrics = {
        "DetPr": partial(det_precision_score, THRESHOLD=THRESHOLD, K=K),
        "nDCG": partial(ndcg_score, K=K),
    }

    for k, metric in (regression_metrics | other_metrics).items():
        metrics[k] = metric(true, pred)

    true = true < THRESHOLD

    for k, metric in classification_metrics.items():
        # The mutation is considered stabilizing if predicted DDG < THRESHOLD=-0.5.
        # That means that the lower the prediction of the model the more mutation is likely to be stabilizing.
        # Hence, to correctly calculate AUC score we must invert the predictions of the model:
        pred_ = pred < THRESHOLD if k != "AUC" else -pred
        try:
            metrics[k] = metric(true, pred_)
        except:
            metrics[k] = 0.0

    return metrics


def add_tensorboard_metrics(
    writer, epoch, train_loss, valid_loss, train_metrics, valid_metrics
):
    writer.add_scalar("Training Loss", train_loss, epoch)
    writer.add_scalar("Validation Loss", valid_loss, epoch)
    writer.add_scalars(
        "Validation metrics",
        {
            "R2": valid_metrics["R2"],
            "RMSE": valid_metrics["RMSE"],
            "Pearson": valid_metrics["Pearson"],
            "Spearman": valid_metrics["Spearman"],
            "StabSpearman": valid_metrics["StabSpearman"],
            "DetPr": valid_metrics["DetPr"],
            "nDCG": valid_metrics["nDCG"],
            "MCC": valid_metrics["MCC"],
            "AUC": valid_metrics["AUC"],
            "ACC": valid_metrics["ACC"],
        },
        epoch,
    )

    writer.add_scalars(
        "Training metrics",
        {
            "R2": train_metrics["R2"],
            "RMSE": train_metrics["RMSE"],
            "Pearson": train_metrics["Pearson"],
            "Spearman": train_metrics["Spearman"],
            "StabSpearman": train_metrics["StabSpearman"],
            "DetPr": train_metrics["DetPr"],
            "nDCG": train_metrics["nDCG"],
            "MCC": train_metrics["MCC"],
            "AUC": train_metrics["AUC"],
            "ACC": train_metrics["ACC"],
        },
        epoch,
    )


def build_baseline_model(model_cfg):
    return MLPHead(
        in_channels=model_cfg.in_channels,
        dim_hidden=model_cfg.hidden_size,
        dropout=model_cfg.dropout,
        norm_layer=torch.nn.BatchNorm1d,
    )


def build_abyssal_model(model_cfg):
    return ABYSSALModel(
        esm_model_name=model_cfg.esm_model_name, embed_dim=model_cfg.hidden_size
    )
