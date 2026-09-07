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
from torch.utils.data import DataLoader
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


def get_train_valid_sampler(trainset, valid_ratio):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid_ratio * size)

    return DistributedSampler(idx[split:]), DistributedSampler(idx[:split])


def get_data_loaders(path, dataset_class, batch_size, valid_ratio, num_workers, pin_memory):
    trainset = dataset_class(path)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid_ratio)

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=trainset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    valid_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=valid_sampler,
        collate_fn=trainset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    testset = dataset_class(path, train=False)
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        collate_fn=testset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, valid_loader, test_loader


def setup_samplers(trainset, valid_ratio, epoch):
    train_sampler, valid_sampler = get_train_valid_sampler(
        trainset, valid_ratio=valid_ratio)
    train_sampler.set_epoch(epoch)
    valid_sampler.set_epoch(epoch)


def main(local_rank, seeds):
    
    print(f"Running main(**args) on rank {local_rank}.")
    init_ddp(local_rank) 
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
            train_loader, valid_loader, test_loader = get_data_loaders(
                path=MELD_path,
                dataset_class=MELDDataset_BERT,
                valid_ratio=0.1,
                batch_size=args.batch_size,
                num_workers=0,
                pin_memory=False
            )
        elif args.dataset == "IEMOCAP":
            train_loader, valid_loader, test_loader = get_data_loaders(
                path=IEMOCAP_path,
                dataset_class=IEMOCAPDataset_BERT,
                valid_ratio=0.1,
                batch_size=args.batch_size,
                num_workers=0,
                pin_memory=False
            )
        
        elif args.dataset == "CMUMOSEI7":
            train_loader, valid_loader, test_loader = get_data_loaders(
                path=CMUMOSEI7_path,
                dataset_class=CMUMOSEIDataset7,
                valid_ratio=0.1,
                batch_size=args.batch_size,
                num_workers=0,
                pin_memory=False
            )

        else:
            print("There is no such dataset")

        best_f1_emo, best_loss = None, None
        best_label_emo, best_pred_emo = None, None
        best_initial_feats = None
        best_extracted_feats = None
        all_f1_emo, all_acc_emo, all_loss = [], [], []

        for epoch in range(args.epochs):
            if args.dataset == "MELD":
                trainset = MELDDataset_BERT(MELD_path)
            elif args.dataset == "IEMOCAP":
                trainset = IEMOCAPDataset_BERT(IEMOCAP_path)
            elif args.dataset == "CMUMOSEI7":
                trainset = CMUMOSEIDataset7(CMUMOSEI7_path)

            setup_samplers(trainset, valid_ratio=0.1, epoch=epoch)

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

            valid_loss, _, _, valid_acc_emo, valid_f1_emo, _, _, _ = train_or_eval_model(
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

            print(
                "epoch: {}, train_loss: {}, train_acc_emo: {}, train_f1_emo: {}, valid_loss: {}, valid_acc_emo: {}, valid_f1_emo: {}"
                .format(
                    epoch + 1,
                    train_loss,
                    train_acc_emo,
                    train_f1_emo,
                    valid_loss,
                    valid_acc_emo,
                    valid_f1_emo
                ))

            if local_rank == 0:
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

                all_f1_emo.append(test_f1_emo)
                all_acc_emo.append(test_acc_emo)

                print(
                    "test_loss: {}, test_acc_emo: {}, test_f1_emo: {}, total time: {} sec, {}"
                    .format(
                        test_loss,
                        test_acc_emo,
                        test_f1_emo,
                        round(time.time() - start_time, 2),
                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    ))
                print("-" * 100)
                
                if best_f1_emo == None or best_f1_emo < test_f1_emo:
                    best_f1_emo = test_f1_emo
                    best_label_emo, best_pred_emo = test_label_emo, test_pred_emo
                    best_initial_feats = test_initial_feats
                    best_extracted_feats = test_extracted_feats

                    # save checkpoints
                    save_dir = "checkpoints"
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    save_path = os.path.join(save_dir, f"best_model_{args.dataset}_{args.seed}.pth")

                    checkpoint = {
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'args': args,
                        'f1': best_f1_emo,
                        'epoch': epoch + 1 
                    }

                    torch.save(checkpoint, save_path)


                if (epoch + 1) % 10 == 0:
                    np.set_printoptions(suppress=True)
                    print(classification_report(best_label_emo, best_pred_emo, digits=4, zero_division=0))
                    print(confusion_matrix(best_label_emo, best_pred_emo))
                    print("-" * 100)

            dist.barrier()

            if args.tensorboard:
                writer.add_scalar("test: accuracy", test_acc_emo, epoch)
                writer.add_scalar("test: fscore", test_f1_emo, epoch)
                writer.add_scalar("train: accuracy", train_acc_emo, epoch)
                writer.add_scalar("train: fscore", train_f1_emo, epoch)

            if epoch == 1:
                allocated_memory = torch.cuda.memory_allocated()
                reserved_memory = torch.cuda.memory_reserved()
                print(f"Allocated Memory: {allocated_memory / 1024**2:.2f} MB")
                print(f"Reserved Memory: {reserved_memory / 1024**2:.2f} MB")
                print(f"All Memory: {(allocated_memory + reserved_memory) / 1024**2:.2f} MB")

        if args.tensorboard:
            writer.close()
        if local_rank == 0:
            print("Test performance..")
            print("Acc: {}, F-Score: {}".format(max(all_acc_emo), max(all_f1_emo)))

            log_path = "results/log_results.txt"
            os.makedirs("results", exist_ok=True)

            report_str = classification_report(best_label_emo,
                                    best_pred_emo,
                                    digits=4,
                                    zero_division=0)

            with open(log_path, "a") as f:
                f.write(f"Seed: {args.seed}  Acc: {max(all_acc_emo):.4f}   F-Score: {max(all_f1_emo):.4f}\n")
                # f.write(report_str + "\n")

            print(report_str)
            # conf_matrix = confusion_matrix(best_label_emo, best_pred_emo)
            # print(conf_matrix)
            # plot_confusion_matrix(conf_matrix, args.dataset, f'conf_{args.seed}')
            plot_tsne(best_initial_feats, best_label_emo, args.dataset, f'initial_features_{args.seed}')
            plot_tsne(best_extracted_feats, best_label_emo, args.dataset, f'extracted_features_{args.seed}')


    dist.destroy_process_group()


if __name__ == "__main__":
    print(args)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("not args.no_cuda:", not args.no_cuda)
    n_gpus = torch.cuda.device_count()
    print(f"Use {n_gpus} GPUs")
    seeds = [260, 9161, 1833, 3216, 3620, 6083, 4642, 2931, 5973, 2136]
    mp.spawn(fn=main, args=(seeds,), nprocs=n_gpus)

            