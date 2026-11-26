"""DPA-inspired parameter diversity diagnostic (NOT full DPA reproduction).
Runs on CPU with PyTorch, using synthetic non-biometric data only.
"""
import argparse
import csv
import copy
from pathlib import Path
import torch
from torch import nn


def set_seed(seed):
    torch.manual_seed(seed)


def make_data(n=640, dim=16, classes=4):
    # Each class is a synthetic Gaussian cluster, no face images or identities.
    centers = torch.randn(classes, dim) * 2.0
    y = torch.randint(0, classes, (n,))
    x = centers[y] + 0.8 * torch.randn(n, dim)
    return x, y


def train_model(model, x, y, epochs=30, lr=0.02):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    model.eval()
    return model


def make_model(dim=16, classes=4):
    return nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, classes))


def diversify(base, alpha):
    """Interpolate pretrained parameters with random initialization.
    Diagnostic approximation only: not paper DPO training procedure.
    """
    out = copy.deepcopy(base)
    fresh = make_model()
    with torch.no_grad():
        for a, b in zip(out.parameters(), fresh.parameters()):
            a.mul_(1 - alpha).add_(alpha * b)
    return out.eval()


def agreement(a, b, x):
    with torch.no_grad():
        return (a(x).argmax(1) == b(x).argmax(1)).float().mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='results/dpa_parameter_diagnostic.csv')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=30)
    args = p.parse_args()
    set_seed(args.seed)
    x, y = make_data()
    x_train, y_train = x[:512], y[:512]
    x_test, y_test = x[512:], y[512:]
    base = train_model(make_model(), x_train, y_train, args.epochs)
    with torch.no_grad():
        base_acc = (base(x_test).argmax(1) == y_test).float().mean().item()
    rows = []
    for alpha in [0., 0.05, 0.10, 0.20, 0.40, 0.80]:
        variant = diversify(base, alpha)
        with torch.no_grad():
            acc = (variant(x_test).argmax(1) == y_test).float().mean().item()
        rows.append({'random_init_fraction': alpha,
                     'variant_accuracy': round(acc, 4),
                     'agreement_with_base': round(agreement(base, variant, x_test), 4),
                     'base_accuracy': round(base_acc, 4)})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Baseline accuracy:', round(base_acc, 4))
    for r in rows:
        print(r)
    print('Saved:', out.resolve())
    print('NOTE: parameter-diversity diagnostic only; not DPO+HMA or attack ASR.')


if __name__ == '__main__':
    main()
