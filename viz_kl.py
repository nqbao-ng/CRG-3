
import argparse, os, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import DiGemo
from trainer import seed_everything, train_or_eval_model
# get_data_loaders nằm trong run.py; import trực tiếp để khỏi viết lại
from run import get_data_loaders


# ----------------------------- args -----------------------------
def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no_cuda", action="store_true", default=False)
    p.add_argument("--gpu", default="0", type=str)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--modals", default="tva")
    p.add_argument("--dataset", default="IEMOCAP")
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--win", nargs="+", type=int, default=[17, 17])
    p.add_argument("--heter_n_layers", nargs="+", type=int, default=[5, 5, 5])
    p.add_argument("--dropout_1", type=float, default=0.05)
    p.add_argument("--dropout_2", type=float, default=0.2)
    p.add_argument("--loss_type", default="distil")
    p.add_argument("--gammas", nargs="+", type=float, default=[1.0, 0.4, 1.0])
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--temp", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=2020)
    p.add_argument("--no_intra", action="store_true", default=False)
    p.add_argument("--fusion_method", default="gated")
    p.add_argument("--no_residual", action="store_true", default=False)
    p.add_argument("--no_graph", action="store_true", default=False)
    # rieng cho script nay:
    p.add_argument("--data_path", required=True, help="duong dan file feature .pkl")
    p.add_argument("--valid_ratio", type=float, default=0.1)
    p.add_argument("--out", default="kl_curve")
    return p.parse_args()


@torch.no_grad()
def measure_kl_per_branch(model, loader, modals, temp, cuda):

    model.eval()
    kl_fn = nn.KLDivLoss(reduction="batchmean")
    tot = {"t": 0.0, "v": 0.0, "a": 0.0}
    nb = 0
    for data in loader:
        textf, visuf, acouf, qmask, umask, label_emotion = (
            [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        )
        dia_lengths = []
        for j in range(umask.size(1)):
            dia_lengths.append((umask[:, j] == 1).nonzero().tolist()[-1][0] + 1)

        fused_logit, t_logit, v_logit, a_logit, _ = model(
            textf, visuf, acouf, umask, qmask, dia_lengths
        )
        teacher = F.softmax(fused_logit / temp, -1)
        if "t" in modals:
            tot["t"] += kl_fn(F.log_softmax(t_logit / temp, -1), teacher).item()
        if "v" in modals:
            tot["v"] += kl_fn(F.log_softmax(v_logit / temp, -1), teacher).item()
        if "a" in modals:
            tot["a"] += kl_fn(F.log_softmax(a_logit / temp, -1), teacher).item()
        nb += 1
    return {k: v / max(nb, 1) for k, v in tot.items()}


def main():
    args = build_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cuda = (not args.no_cuda) and torch.cuda.is_available()

    ds = args.dataset.upper()
    if ds.startswith("IEMOCAP") and "4" not in ds:
        embedding_dims, n_classes = [1024, 342, 1582], 6
    elif "MELD" in ds:
        embedding_dims, n_classes = [1024, 342, 300], 7
    else:
        raise ValueError("Chinh embedding_dims/n_classes cho dataset cua ban tai day")

    seed_everything(args.seed)
    model = DiGemo(args, embedding_dims, n_classes)
    if cuda:
        model = model.cuda()

    loss_emo = nn.NLLLoss()
    loss_kl = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)

    with open(args.data_path, "rb") as f:
        _ = f 
    train_loader, valid_loader, test_loader = get_data_loaders(
        args.data_path, ds, args.batch_size, args.valid_ratio, 0, False
    )

    hist = {"t": [], "v": [], "a": []}
    for epoch in range(1, args.epochs + 1):
        train_or_eval_model(
            model, loss_emo, loss_kl, train_loader, cuda, args.modals,
            optimizer=optimizer, train=True, loss_type=args.loss_type,
            gammas=args.gammas, temp=args.temp, seed=args.seed,
        )
        kl = measure_kl_per_branch(model, train_loader, args.modals, args.temp, cuda)
        for k in hist:
            hist[k].append(kl[k])
        print(f"epoch {epoch:3d} | KL_t {kl['t']:.4f}  KL_v {kl['v']:.4f}  KL_a {kl['a']:.4f}")

    np.savez(f"{args.out}.npz", **{k: np.array(v) for k, v in hist.items()})
    plot_curve(hist, ds, f"{args.out}.pdf")
    print(f"saved {args.out}.pdf and {args.out}.npz")


def plot_curve(hist, ds, out_pdf):
    plt.rcParams["font.family"] = "DejaVu Serif"
    ep = np.arange(1, len(hist["t"]) + 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(ep, hist["v"], color="#6FB07F", lw=2, label="Visual branch")
    ax.plot(ep, hist["a"], color="#E08A3C", lw=2, label="Acoustic branch")
    ax.plot(ep, hist["t"], color="#3C6FE0", lw=2, label="Textual branch")
    ax.set_xlabel("Training epoch", fontsize=12)
    ax.set_ylabel(r"KL divergence  KL($\hat{Y}^{\tau}\,\Vert\,\hat{Y}^{\tau}_{m}$)", fontsize=12)
    ax.set_title(f"Teacher-student divergence during self-distillation ({ds})", fontsize=12)
    ax.legend(frameon=False, fontsize=11)
    ax.grid(alpha=0.25, ls="--")
    ax.set_xlim(1, len(ep)); ax.set_ylim(0, None)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_pdf.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()