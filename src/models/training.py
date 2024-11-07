"""
Code to define training and testing loops for our RDL models.
"""
import copy
import math
import numpy as np
import torch
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

# Global constants for node classification tasks
tune_metric = "roc_auc"
higher_is_better = True

def train(model, device, optimizer, loader_dict, task, loss_fn, entity_table) -> float:
    """
    Training the model.
    """
    model.train()

    loss_accum = count_accum = 0
    for batch in tqdm(loader_dict["train"]):
        batch = batch.to(device)

        optimizer.zero_grad()
        pred = model(
            batch,
            task.entity_table,
        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred

        loss = loss_fn(pred.float(), batch[entity_table].y.float())
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)

    return loss_accum / count_accum


@torch.no_grad()
def test(loader: NeighborLoader, model, task, device) -> np.ndarray:
    """
    Testing the model.
    """
    model.eval()

    pred_list = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(
            batch,
            task.entity_table,
        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
    return torch.cat(pred_list, dim=0).numpy()

def training_run(model, device, optimizer, task, loader_dict, val_table, loss_fn, entity_table, epochs = 10):
    """
    Training loop of training, running on validation set, and then returning best set of weights.
    """
    state_dict = None
    best_val_metric = -math.inf if higher_is_better else math.inf
    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, optimizer, loader_dict, task, loss_fn, entity_table)
        val_pred = test(loader_dict["val"], model, task, device)
        val_metrics = task.evaluate(val_pred, val_table)
        print(f"Epoch: {epoch:02d}, Train loss: {train_loss}, Val metrics: {val_metrics}")

        if (higher_is_better and val_metrics[tune_metric] > best_val_metric) or (
            not higher_is_better and val_metrics[tune_metric] < best_val_metric
        ):
            best_val_metric = val_metrics[tune_metric]
            state_dict = copy.deepcopy(model.state_dict())
    return state_dict

def eval_model(model, loader_dict, split, task, device, table):
    """
    Evaluate a model on a specific split of a task.
    """
    pred = test(loader_dict[split], model, task, device)
    metrics = None
    if split == "test":
        metrics = task.evaluate(pred)
    else:
        metrics = task.evaluate(pred, table)
    print(f"Best {split} metrics: {metrics}")
