import numpy as np, random
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from utils import AutomaticWeightedLoss


# seed = 2025


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_or_eval_model(
    model,
    loss_function_emo,
    loss_function_kl,
    dataloader,
    cuda,
    modals,
    optimizer=None,
    train=False,
    loss_type="",
    gammas=[1.0, 1.0, 1.0],
    temp=1.0,
    awl=None,
    seed=None
):
    losses, preds_emo, labels_emo = [], [], []
    vids = []
    initial_feats, fused_features = [], []

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything(seed)
    for iter, data in enumerate(dataloader):

        if train:
            optimizer.zero_grad()

        textf, visuf, acouf, qmask, umask, label_emotion = ([d.cuda() for d in data[:-1]] if cuda else data[:-1])

        dia_lengths, label_emotions, initial_cats = [], [], [] 
        for j in range(umask.size(1)):
            dia_lengths.append((umask[:, j] == 1).nonzero().tolist()[-1][0] + 1)
            label_emotions.append(label_emotion[:dia_lengths[j], j])
            initial_cats.append(torch.cat([
                textf[:dia_lengths[j], j, :],
                visuf[:dia_lengths[j], j, :],
                acouf[:dia_lengths[j], j, :]
            ], dim=-1))
        label_emo = torch.cat(label_emotions)
        initial_feat = torch.cat(initial_cats)

        fused_logit, t_logit, v_logit, a_logit, fused_feature = model(textf, visuf, acouf, umask, qmask, dia_lengths)

        fused_prob = F.log_softmax(fused_logit, -1)
        kl_all_prob = F.softmax(fused_logit / temp, -1)
        loss_emo = loss_function_emo(fused_prob, label_emo)

        loss_t, loss_v, loss_a = 0.0, 0.0, 0.0
        kl_loss_t, kl_loss_v, kl_loss_a = 0.0, 0.0, 0.0
        if 't' in modals:
            t_prob = F.log_softmax(t_logit, -1)
            loss_t = loss_function_emo(t_prob, label_emo)

            kl_t_prob = F.log_softmax(t_logit / temp, -1)
            kl_loss_t = loss_function_kl(kl_t_prob, kl_all_prob)

        if 'v' in modals:
            v_prob = F.log_softmax(v_logit, -1)
            loss_v = loss_function_emo(v_prob, label_emo)

            kl_v_prob = F.log_softmax(v_logit / temp, -1)
            kl_loss_v = loss_function_kl(kl_v_prob, kl_all_prob)

        if 'a' in modals:
            a_prob = F.log_softmax(a_logit, -1)
            loss_a = loss_function_emo(a_prob, label_emo)

            kl_a_prob = F.log_softmax(a_logit / temp, -1)
            kl_loss_a = loss_function_kl(kl_a_prob, kl_all_prob)

        if loss_type == "auto":
            loss = awl([loss_emo, (loss_t + loss_v + loss_a), (kl_loss_t + kl_loss_v + kl_loss_a)])
            pass  
        elif loss_type == "wo_distil":
            loss = loss_emo
        elif loss_type == 'distil':
            loss = gammas[0] * loss_emo + gammas[1] * (loss_t + loss_v + loss_a) + gammas[2] * (kl_loss_t + kl_loss_v + kl_loss_a)
        else:
            NotImplementedError

        preds_emo.append(torch.argmax(fused_prob, 1).cpu().numpy())
        labels_emo.append(label_emo.cpu().numpy())
        initial_feats.append(initial_feat.cpu().numpy())
        losses.append(loss.item())

        if train:
            loss.backward()
            optimizer.step()

        fused_features.append(fused_feature.cpu().detach().numpy())

    if preds_emo != []:
        preds_emo = np.concatenate(preds_emo)
        labels_emo = np.concatenate(labels_emo)
        initial_feats = np.concatenate(initial_feats)
        fused_features = np.concatenate(fused_features)

    vids += data[-1]
    labels_emo = np.array(labels_emo)
    initial_feats = np.array(initial_feats)
    fused_features = np.array(fused_features)
    preds_emo = np.array(preds_emo)
    vids = np.array(vids)

    fused_features = np.array(fused_features)

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_acc_emo = round(accuracy_score(labels_emo, preds_emo) * 100, 2)
    avg_f1_emo = round(f1_score(labels_emo, preds_emo, average="weighted") * 100, 2)

    return avg_loss, labels_emo, preds_emo, avg_acc_emo, avg_f1_emo, vids, initial_feats, fused_features



