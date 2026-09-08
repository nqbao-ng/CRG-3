import logging
import os
import numpy as np
import pickle as pk
import datetime
import torch.nn as nn
import torch.optim as optim
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import time
from utils import AutomaticWeightedLoss
from model import DiGemo
from sklearn.metrics import confusion_matrix, classification_report
from trainer import train_or_eval_model, seed_everything
from dataloader import (
    IEMOCAPDataset_BERT,
    MELDDataset_BERT,
    CMUMOSEIDataset7
)
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter
import argparse
from plot import plot_tsne, plot_confusion_matrix


parser = argparse.ArgumentParser()

parser.add_argument("--no_cuda", action="store_true", default=False, help="does not use GPU")

parser.add_argument("--gpu", default="0", type=str, help="GPU ids")

parser.add_argument("--port", default="15301", help="MASTER_PORT")

parser.add_argument("--lr", type=float, default=0.00001, metavar="LR", help="learning rate")

parser.add_argument("--l2", type=float, default=0.0001, metavar="L2", help="L2 regularization weight")

parser.add_argument("--batch_size", type=int, default=16, metavar="BS", help="batch size")

parser.add_argument("--epochs", type=int, default=100, metavar="E", help="number of epochs")

parser.add_argument("--tensorboard", action="store_true", default=False, help="Enables tensorboard log")

parser.add_argument("--modals", default="tva", help="modals: tva, tv, ta, va")

parser.add_argument("--dataset", default="IEMOCAP6", help="dataset to train and test IEMOCAP6/MELD")

parser.add_argument("--hidden_dim", type=int, default=512, help="hidden_dim")

parser.add_argument("--win", nargs="+", type=int, default=[17, 17], help="[win_p, win_f], -1 denotes all nodes")

parser.add_argument("--heter_n_layers", nargs="+", type=int, default=[4, 4, 4], help="heter_n_layers")

parser.add_argument("--dropout_1", type=float, default=0.1, metavar="DR1", help="dropout rate into TransFormer")

parser.add_argument("--dropout_2", type=float, default=0.2, metavar="DR2", help="dropout rate into GCN")

parser.add_argument("--loss_type", default="distil", help="distil/wo_distil/auto")

parser.add_argument("--gammas", nargs="+", type=float, default=[1.0, 1.0, 1.0], help="[task_loss, uni_ce_loss, kl_loss]")

parser.add_argument("--num_heads", type=int, default=8, metavar="H", help="number of heads of trans layers")

parser.add_argument("--temp", type=float, default=1.0, help="temp of KL loss")

parser.add_argument("--seed", type=int, default=2020, help="seed")

parser.add_argument("--no_intra", action="store_true", default=False, help="does not use Trans based contextual modeling")

parser.add_argument("--fusion_method", default="gated", help="fusion method: gated/concat/add/mean/max")
    
parser.add_argument("--no_residual", action="store_true", default=False, help="does not use residual graph")

parser.add_argument("--no_graph", action="store_true", default=False, help="does not use cross graph")

args = parser.parse_args()

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = args.port
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
world_size = torch.cuda.device_count()
os.environ["WORLD_SIZE"] = str(world_size)

MELD_path = "./features/meld_multi_features.pkl"
IEMOCAP_path = "./features/iemocap_multi_features.pkl"
CMUMOSEI7_path = "./features/cmumosei7_multi_features.pkl"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_ddp(local_rank):
    try:
        if not dist.is_initialized():
            torch.cuda.set_device(local_rank)
            os.environ["RANK"] = str(local_rank)
            dist.init_process_group(backend="nccl", init_method="env://")
        else:
            logger.info("Distributed process group already initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize distributed process group: {e}")
        raise


def get_data_loaders(path, dataset_class, batch_size, valid_ratio, num_workers, pin_memory, sampler_seed=2026):
    full_trainset = dataset_class(path, train=True)
    
    size = len(full_trainset)
    indices = list(range(size))

    split = max(1, int(valid_ratio * size))
    valid_indices = indices[:split]
    train_indices = indices[split:]

    train_subset = Subset(full_trainset, train_indices)
    valid_subset = Subset(full_trainset, valid_indices)

    train_sampler = DistributedSampler(
        train_subset,
        shuffle=True,
        seed=sampler_seed
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=full_trainset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    valid_loader = DataLoader(
        valid_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=full_trainset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    testset = dataset_class(path, train=False)
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=testset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    train_ids = [full_trainset.keys[i] for i in train_indices]
    valid_ids = [full_trainset.keys[i] for i in valid_indices]
    assert set(train_ids).isdisjoint(valid_ids)

    return train_loader, valid_loader, test_loader , train_sampler

def training(local_rank, seeds):
    
    print(f"Running main(**args) on rank {local_rank}.")
    init_ddp(local_rank) 
    seed_test_accs, seed_test_f1s = [], []
    for seed in seeds:
        args.seed = seed

        today = datetime.datetime.now()
        name_ = args.modals + "_" + args.dataset

        cuda = torch.cuda.is_available() and not args.no_cuda
        if args.tensorboard:
            writer = SummaryWriter()

        if args.dataset == "IEMOCAP":
            embedding_dims = [1024, 342, 1582]
            n_classes_emo = 6
        elif args.dataset == "IEMOCAP4":
            embedding_dims = [1024, 512, 100]
            n_classes_emo = 4
        elif args.dataset == "MELD":
            embedding_dims = [1024, 342, 300]
            n_classes_emo = 7
        elif args.dataset == "CMUMOSEI7":
            embedding_dims = [1024, 35, 384]
            n_classes_emo = 7

        seed_everything(args.seed)
        model = DiGemo(args, embedding_dims, n_classes_emo)

        model = model.to(local_rank)
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )

        loss_function_emo = nn.NLLLoss()
        loss_function_kl = nn.KLDivLoss(reduction='batchmean')

        if args.loss_type == "auto":
            awl = AutomaticWeightedLoss(3)
            optimizer = optim.AdamW(
                [
                    {
                        "params": model.parameters()
                    },
                    {
                        "params": awl.parameters(),
                        "weight_decay": 0
                    },
                ],
                lr=args.lr,
                weight_decay=args.l2,
                amsgrad=True,
            )
        else:
            awl = None
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2, amsgrad=True)

        if args.dataset == "MELD":
            train_loader, valid_loader, test_loader, train_sampler = get_data_loaders(
                path=MELD_path,
                dataset_class=MELDDataset_BERT,
                valid_ratio=0.1,
                batch_size=args.batch_size,
                num_workers=0,
                pin_memory=False,
                sampler_seed=args.seed
            )
        elif args.dataset == "IEMOCAP":
            train_loader, valid_loader, test_loader, train_sampler = get_data_loaders(
                path=IEMOCAP_path,
                dataset_class=IEMOCAPDataset_BERT,
                valid_ratio=0.1,
                batch_size=args.batch_size,
                num_workers=0,
                pin_memory=False,
                sampler_seed=args.seed
            )
        
        elif args.dataset == "CMUMOSEI7":
            train_loader, valid_loader, test_loader, train_sampler = get_data_loaders(
                path=CMUMOSEI7_path,
                dataset_class=CMUMOSEIDataset7,
                valid_ratio=0.1,
                batch_size=args.batch_size,
                num_workers=0,
                pin_memory=False,
                sampler_seed=args.seed
            )

        else:
            print("There is no such dataset")

        best_valid_f1 = -float("inf")
        best_epoch = -1
        save_dir = "checkpoints"
        if local_rank == 0:
            os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"best_model_{args.dataset}_{args.seed}.pth")

        for epoch in range(args.epochs):

            train_sampler.set_epoch(epoch)

            start_time = time.time()

            train_loss, _, _, train_acc_emo, train_f1_emo, _, _, _ = train_or_eval_model(
                model,
                loss_function_emo,
                loss_function_kl,
                train_loader,
                cuda,
                args.modals,
                optimizer,
                True,
                args.loss_type,
                args.gammas,
                args.temp,
                awl,
                args.seed
            )
            if local_rank == 0:
                valid_loss, valid_label_emo, valid_pred_emo, valid_acc_emo, valid_f1_emo, _, _, _ = train_or_eval_model(
                    model,
                    loss_function_emo,
                    loss_function_kl,
                    valid_loader,
                    cuda,
                    args.modals,
                    None,
                    False,
                    args.loss_type,
                    args.gammas,
                    args.temp,
                    awl,
                    args.seed
                )

                test_loss, test_label_emo, test_pred_emo, test_acc_emo, test_f1_emo, _, test_initial_feats, test_extracted_feats = train_or_eval_model(
                    model,
                    loss_function_emo,
                    loss_function_kl,
                    test_loader,
                    cuda,
                    args.modals,
                    None,
                    False,
                    args.loss_type,
                    args.gammas,
                    args.temp,
                    awl,
                    args.seed
                )

                print(
                    "epoch: {}, train_loss: {}, train_acc_emo: {}, train_f1_emo: {}, valid_loss: {}, valid_acc_emo: {}, valid_f1_emo: {}, "
                    "test_loss: {}, test_acc_emo: {}, test_f1_emo: {}, total time: {} sec"
                    .format(
                        epoch + 1,
                        train_loss,
                        train_acc_emo,
                        train_f1_emo,
                        valid_loss,
                        valid_acc_emo,
                        valid_f1_emo,
                        test_loss,
                        test_acc_emo,
                        test_f1_emo,
                        round(time.time() - start_time, 2),
                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    ))
                
                if valid_f1_emo > best_valid_f1:
                    best_valid_f1 = valid_f1_emo
                    best_epoch = epoch + 1

                    checkpoint = {
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'args': args.copy(),
                        'valid_f1': best_valid_f1,
                        'test_f1_at_best_dev_epoch': test_f1_emo,
                        'test_acc_at_best_dev_epoch': test_acc_emo,
                        'epoch': best_epoch
                    }
                    torch.save(checkpoint, save_path)


                if (epoch + 1) % 10 == 0:
                    np.set_printoptions(suppress=True)
                    print(classification_report(valid_label_emo, valid_pred_emo, digits=4, zero_division=0))
                    print(confusion_matrix(valid_label_emo, valid_pred_emo))
                    print("-" * 100)

                if args.tensorboard:
                    writer.add_scalar("valid: accuracy", valid_acc_emo, epoch)
                    writer.add_scalar("valid: fscore", valid_f1_emo, epoch)
                    writer.add_scalar("test: accuracy", test_acc_emo, epoch)
                    writer.add_scalar("test: fscore", test_f1_emo, epoch)
                    writer.add_scalar("train: accuracy", train_acc_emo, epoch)
                    writer.add_scalar("train: fscore", train_f1_emo, epoch)

            dist.barrier()

            if epoch == 1:
                allocated_memory = torch.cuda.memory_allocated()
                reserved_memory = torch.cuda.memory_reserved()
                print(f"Allocated Memory: {allocated_memory / 1024**2:.2f} MB")
                print(f"Reserved Memory: {reserved_memory / 1024**2:.2f} MB")
                print(f"All Memory: {(allocated_memory + reserved_memory) / 1024**2:.2f} MB")

        if args.tensorboard:
            writer.close()
        if local_rank == 0:
            checkpoint = torch.load(save_path, map_location=f"cuda:{local_rank}", weights_only=False)
            model.module.load_state_dict(checkpoint['model_state_dict'])

            test_loss, test_label_emo, test_pred_emo, test_acc_emo, test_f1_emo, _, test_initial_feats, test_extracted_feats = train_or_eval_model(
                model.module,
                loss_function_emo,
                loss_function_kl,
                test_loader,
                cuda,
                args.modals,
                None,
                False,
                args.loss_type,
                args.gammas,
                args.temp,
                awl,
                args.seed
            )

            print("-" * 100)
            print(
                f"Best dev checkpoint: epoch {best_epoch}, dev WF1={best_valid_f1:.2f}"
            )
            print(
                f"FINAL TEST (best-dev checkpoint): loss={test_loss}, "
                f"Acc={test_acc_emo:.2f}, WF1={test_f1_emo:.2f}"
            )
            print(classification_report(test_label_emo, test_pred_emo, digits=4, zero_division=0))
            print(confusion_matrix(test_label_emo, test_pred_emo))

            seed_test_accs.append(test_acc_emo)
            seed_test_f1s.append(test_f1_emo)

            log_path = "results/log_results.txt"
            os.makedirs("results", exist_ok=True)
            with open(log_path, "a") as f:
                f.write(
                    f"Seed: {args.seed}  BestEpoch: {best_epoch}  "
                    f"DevF1: {best_valid_f1:.4f}  "
                    f"TestAcc: {test_acc_emo:.4f}  TestF1: {test_f1_emo:.4f}\n"
                )

            plot_tsne(test_initial_feats, test_label_emo, args.dataset, f'initial_features_{args.seed}')
            plot_tsne(test_extracted_feats, test_label_emo, args.dataset, f'extracted_features_{args.seed}')

        dist.barrier()
    
    if local_rank == 0 and seed_test_f1s:
        print("=" * 100)
        print(
            f"FINAL over {len(seed_test_f1s)} seeds | "
            f"Test Acc: {np.mean(seed_test_accs):.2f} ± {np.std(seed_test_accs):.2f} | "
            f"Test WF1: {np.mean(seed_test_f1s):.2f} ± {np.std(seed_test_f1s):.2f}"
        )
        print("=" * 100)


def main(local_rank, seeds):
    try:
        training(local_rank, seeds)
    finally:
        # Release NCCL resources on normal completion and on Python exceptions.
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    print(args)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("not args.no_cuda:", not args.no_cuda)
    n_gpus = torch.cuda.device_count()
    print(f"Use {n_gpus} GPUs")
    seeds = [260, 9161, 1833, 3216, 3620]
    mp.spawn(fn=main, args=(seeds,), nprocs=n_gpus)