"""
Training code for GVP on estimation of model accuracy data. Parts of this script are modified from https://github.com/drorlab/gvp-pytorch/blob/main/run_atom3d.py.
"""

import argparse
import logging
import os
import time
import datetime
from tqdm import tqdm
from functools import partial
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from atom3d.datasets import LMDBDataset
from atom3d.util import metrics
from gvp import atom3d

def get_metrics():
    def _correlation(metric, targets, predict, ids=None, glob=True):
        if glob: return metric(targets, predict)
        _targets, _predict = defaultdict(list), defaultdict(list)
        for _t, _p, _id in zip(targets, predict, ids):
            _targets[_id].append(_t)
            _predict[_id].append(_p)
        return np.mean([metric(_targets[_id], _predict[_id]) for _id in _targets])
        
    correlations = {
        'pearson': partial(_correlation, metrics.pearson),
        'kendall': partial(_correlation, metrics.kendall),
        'spearman': partial(_correlation, metrics.spearman)
    }
    mean_correlations = {f'mean {k}' : partial(v, glob=False) \
                            for k, v in correlations.items()}
    
    return {**correlations, **mean_correlations}

def train_loop(model, loader, optimizer, device):
    model.train()

    loss_all = 0
    total = 0
    for data in tqdm(loader):
        data = data.to(device)
        optimizer.zero_grad()
        try:
            output = model(data)
            loss = F.mse_loss(output, data.label)
            loss.backward()
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e): raise(e)
            torch.cuda.empty_cache()
            print('Skipped batch due to OOM', flush=True)
            continue

        loss_all += loss.item() * data.num_graphs
        total += data.num_graphs
        optimizer.step()
    return np.sqrt(loss_all / total)


@torch.no_grad()
def test(model, loader, device):
    model.eval()

    losses = []
    total = 0

    y_true = []
    y_pred = []
    ids = []

    print_frequency = 10

    for it, data in enumerate(loader):
        data = data.to(device)
        output = model(data)
        loss = F.mse_loss(output, data.label)
        losses.append(loss.item())
        # loss_all += loss.item() * data.num_graphs
        # total += data.num_graphs
        y_true.extend([x.item() for x in data.label])
        y_pred.extend(output.tolist())
        ids.extend(data.id)
        
        if it % print_frequency == 0:
            print(f'iter {it}, loss {np.mean(losses)}')

    test_df = pd.DataFrame(
        np.array([ids, y_true, y_pred]).T,
        columns=['target', 'true', 'pred'],
        )
    
    metrics = get_metrics()
    results = {}
    for name, func in metrics.items():
        func = partial(func, ids=ids)
        value = func(y_true, y_pred)
        print(f"{name}: {value}")
        results[name] = value

    return np.mean(losses), results, test_df

def plot_corr(y_true, y_pred, plot_dir):
    plt.clf()
    sns.scatterplot(y_true, y_pred)
    plt.xlabel('Actual -log(K)')
    plt.ylabel('Predicted -log(K)')
    plt.savefig(plot_dir)

def save_weights(model, weight_dir):
    torch.save(model.state_dict(), weight_dir)

def train(args, device, log_dir, rep=None, test_mode=False):
    # logger = logging.getLogger('lba')
    # logger.basicConfig(filename=os.path.join(log_dir, f'train_{split}_cv{fold}.log'),level=logging.INFO)

    train_dataset = LMDBDataset(os.path.join(args.data_dir, 'train'), transform=atom3d.PSRTransform())
    val_dataset = LMDBDataset(os.path.join(args.data_dir, 'val'), transform=atom3d.PSRTransform())
    # test_dataset = LMDBDataset(os.path.join(args.data_dir, 'test'), transform=atom3d.PSRTransform())
    test_dataset = LMDBDataset(os.path.join(args.data_dir, 'test'), transform=atom3d.PSRTransform())
    
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4)

    model = atom3d.PSRModel().to(device)

    best_val_loss = 999
    best_rp = 0
    best_rs = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.num_epochs+1):
        start = time.time()
        train_loss = train_loop(model, train_loader, optimizer, device)
        val_loss, metrics, _ = test(model, val_loader, device)
        if metrics['spearman'] > best_rs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                }, os.path.join(log_dir, f'best_weights_{args.name}.pt'))
            best_rs = metrics['spearman']
        elapsed = (time.time() - start)
        print('Epoch: {:03d}, Time: {:.3f} s'.format(epoch, elapsed))
        print('\tTrain RMSE: {:.7f}, Val RMSE: {:.7f}, Per-target Spearman R: {:.7f}, Global Spearman R: {:.7f}'.format(
            train_loss, val_loss, metrics['mean spearman'], metrics['spearman']))

    if test_mode:
        cpt = torch.load(os.path.join(log_dir, f'best_weights_{args.name}.pt'))
        model.load_state_dict(cpt['model_state_dict'])
        print('Loading checkpoint from best epoch:', cpt['epoch'])
        train_file = os.path.join(log_dir, f'out-{args.name}.best.train.pt')
        val_file = os.path.join(log_dir, f'out-{args.name}.best.val.pt')
        test_file = os.path.join(log_dir, f'out-{args.name}.best.test.pt')
        cpt = torch.load(os.path.join(log_dir, f'best_weights_{args.name}.pt'))
        model.load_state_dict(cpt['model_state_dict'])
        print('Evaluating train...')
        _, metrics, results_train = test(model, train_loader, device)
        torch.save(results_train.to_dict('list'), train_file)
        print('Evaluating val...')
        _, metrics, results_val = test(model, val_loader, device)
        torch.save(results_val.to_dict('list'), val_file)
        print('Evaluating test...')
        print(len(test_dataset))
        _, metrics, results_test = test(model, test_loader, device)
        torch.save(results_test.to_dict('list'), test_file)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--precomputed', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = args.log_dir

    if args.mode == 'train':
        if log_dir is None:
            log_dir = os.path.join('logs', args.name)
        else:
            log_dir = os.path.join('logs', log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        train(args, device, log_dir)
        
    elif args.mode == 'test':
        seed = 2021
        log_dir = os.path.join('logs', f'{args.name}_test')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        np.random.seed(seed)
        torch.manual_seed(seed)
        train(args, device, log_dir, test_mode=True)
