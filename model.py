import torch
import torch.nn as nn
from module import TransformerBasedContext, GatedFusion, CrossModalGraph
from utils import flatten_batch


class DiGemo(nn.Module):

    def __init__(self, args, embedding_dims, n_classes_emo):
        super().__init__()
        self.no_cuda = args.no_cuda
        self.win_p = args.win[0]
        self.win_f = args.win[1]
        self.modals = args.modals
        self.fusion_method = args.fusion_method
        self.no_residual = args.no_residual
        self.no_graph = args.no_graph
        self.no_intra = args.no_intra

        # # Conv layers
        # self.conv_t = nn.Conv1d(embedding_dims[0], args.hidden_dim, kernel_size=1, padding=0, bias=False) 
        # self.conv_v = nn.Conv1d(embedding_dims[1], args.hidden_dim, kernel_size=1, padding=0, bias=False)
        # self.conv_a = nn.Conv1d(embedding_dims[2], args.hidden_dim, kernel_size=1, padding=0, bias=False)

        self.proj_t = nn.Linear(embedding_dims[0], args.hidden_dim, bias=False)
        self.proj_v = nn.Linear(embedding_dims[1], args.hidden_dim, bias=False)
        self.proj_a = nn.Linear(embedding_dims[2], args.hidden_dim, bias=False)

        # speaker embedding
        if n_classes_emo == 6 or n_classes_emo == 4:
            self.n_speakers = 2
        if n_classes_emo == 7:
            self.n_speakers = 9

        self.speaker_embeddings = nn.Embedding(self.n_speakers + 1, args.hidden_dim, padding_idx=self.n_speakers)

        self.enhance_t = TransformerBasedContext(
            args.hidden_dim,
            args.hidden_dim,
            args.num_heads,
            args.dropout_1
        )

        self.enhance_v = TransformerBasedContext(
            args.hidden_dim,
            args.hidden_dim,
            args.num_heads,
            args.dropout_1
        )

        self.enhance_a = TransformerBasedContext(
            args.hidden_dim,
            args.hidden_dim,
            args.num_heads,
            args.dropout_1
        )

        # Heter graph
        self.graph_tv = CrossModalGraph(args.hidden_dim, args.heter_n_layers[0], args.no_cuda)
        self.graph_ta = CrossModalGraph(args.hidden_dim, args.heter_n_layers[1], args.no_cuda)
        self.graph_va = CrossModalGraph(args.hidden_dim, args.heter_n_layers[2], args.no_cuda)

        # residual module
        self.residual_t = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Dropout(args.dropout_2)
        )

        self.residual_v = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Dropout(args.dropout_2)
        )
        
        self.residual_a = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Dropout(args.dropout_2)
        )

        if self.fusion_method == 'gated':
            self.gated_fusion = GatedFusion(args.hidden_dim)
        elif self.fusion_method == 'concat':
            self.reduce_cat = nn.Linear(3 * args.hidden_dim, args.hidden_dim, bias=False)

        # Cls Layers
        self.t_cls_layer = nn.Linear(args.hidden_dim, n_classes_emo)
        self.v_cls_layer = nn.Linear(args.hidden_dim, n_classes_emo)
        self.a_cls_layer = nn.Linear(args.hidden_dim, n_classes_emo)
        self.fusion_cls_layer = nn.Linear(args.hidden_dim, n_classes_emo)
        

    def forward(self, feature_t, feature_v, feature_a, umask, qmask, dia_lengths):
        '''
        feature: (L, B, D)
        umask: (L, B)
        qmask: (L, B, n_speakers)
        '''

        # # Conv layer
        # if 't' in self.modals:
        #     feature_t = self.conv_t(feature_t.permute(1, 2, 0)).transpose(1, 2) # (B, L, D)
        # if 'v' in self.modals:
        #     feature_v = self.conv_v(feature_v.permute(1, 2, 0)).transpose(1, 2)
        # if 'a' in self.modals:
        #     feature_a = self.conv_a(feature_a.permute(1, 2, 0)).transpose(1, 2)

        if 't' in self.modals:
            feature_t = self.proj_t(feature_t.transpose(0, 1)) # (B, L, D)
        if 'v' in self.modals:
            feature_v = self.proj_v(feature_v.transpose(0, 1))
        if 'a' in self.modals:
            feature_a = self.proj_a(feature_a.transpose(0, 1))

        # Speaker emb
        spk_idx = torch.argmax(qmask, -1).transpose(0, 1) # (B, L)
        origin_spk_ixd = spk_idx
        for i, x in enumerate(dia_lengths):
            insert = self.n_speakers * (torch.ones(origin_spk_ixd[i].size(0) - x).int())    
            if not self.no_cuda:
                insert.cuda()
            spk_idx[i, x:] = insert
        spk_embeddings = self.speaker_embeddings(spk_idx) # (B, L, D)

        umask = umask.transpose(0, 1) # (B, L)

        if 't' in self.modals and not self.no_intra:
            feature_t = self.enhance_t(feature_t, umask, spk_embeddings) # (B, L, D)
        if 'v' in self.modals and not self.no_intra:
            feature_v = self.enhance_v(feature_v, umask, spk_embeddings)
        if 'a' in self.modals and not self.no_intra:
            feature_a = self.enhance_a(feature_a, umask, spk_embeddings)

        if 't' in self.modals:
            feature_t = flatten_batch(feature_t, dia_lengths, self.no_cuda)
        if 'v' in self.modals:
            feature_v = flatten_batch(feature_v, dia_lengths, self.no_cuda)
        if 'a' in self.modals:
            feature_a = flatten_batch(feature_a, dia_lengths, self.no_cuda)
        
        if not self.no_graph:

            # Graph learning
            edge_index = None
            if 't' in self.modals and 'v' in self.modals:
                (v_to_t_graph_out, t_to_v_graph_out), edge_index = self.graph_tv(
                    (feature_t, feature_v), 
                    dia_lengths,
                    self.win_p,
                    self.win_f, 
                    edge_index
                )
            if 't' in self.modals and 'a' in self.modals:
                (a_to_t_graph_out, t_to_a_graph_out), edge_index = self.graph_ta(
                    (feature_t, feature_a), 
                    dia_lengths,
                    self.win_p,
                    self.win_f, 
                    edge_index
                )
            if 'v' in self.modals and 'a' in self.modals:
                (a_to_v_graph_out, v_to_a_graph_out), edge_index = self.graph_va(
                    (feature_v, feature_a), 
                    dia_lengths,
                    self.win_p,
                    self.win_f, 
                    edge_index
                )

            # Residual 
            if self.modals == 'tva':
                t_graph_out = v_to_t_graph_out + a_to_t_graph_out  # (N, d)
                v_graph_out = t_to_v_graph_out + a_to_v_graph_out 
                a_graph_out = t_to_a_graph_out + v_to_a_graph_out 
                if not self.no_residual:
                    t_graph_out = t_graph_out + self.residual_t(feature_t)
                    v_graph_out = v_graph_out + self.residual_v(feature_v)
                    a_graph_out = a_graph_out + self.residual_a(feature_a)
                h_list = [t_graph_out, v_graph_out, a_graph_out]
            else:
                if 't' in self.modals and 'v' in self.modals:
                    t_graph_out = v_to_t_graph_out 
                    v_graph_out = t_to_v_graph_out 
                    if not self.no_residual:
                        t_graph_out = t_graph_out + self.residual_t(feature_t)
                        v_graph_out = v_graph_out + self.residual_v(feature_v)
                    a_graph_out = None
                    h_list = [t_graph_out, v_graph_out]
                    
                elif 't' in self.modals and 'a' in self.modals:
                    t_graph_out = a_to_t_graph_out 
                    a_graph_out = t_to_a_graph_out 
                    if not self.no_residual:
                        t_graph_out = t_graph_out + self.residual_t(feature_t)
                        a_graph_out = a_graph_out + self.residual_a(feature_a)
                    v_graph_out = None
                    h_list = [t_graph_out, a_graph_out]
                elif 'v' in self.modals and 'a' in self.modals:
                    v_graph_out = a_to_v_graph_out 
                    a_graph_out = v_to_a_graph_out 
                    if not self.no_residual:
                        v_graph_out = v_graph_out + self.residual_v(feature_v)
                        a_graph_out = a_graph_out + self.residual_a(feature_a)
                    t_graph_out = None
                    h_list = [v_graph_out, a_graph_out]
        else:
            h_list = [feature_t, feature_v, feature_a]

        if self.fusion_method == 'gated':
            fused_feature = self.gated_fusion(h_list)
        elif self.fusion_method == 'mean':
            fused_feature = torch.sum(torch.stack(h_list), dim=0) / len(h_list)
        elif self.fusion_method == 'concat':
            fused_feature = self.reduce_cat(torch.cat(h_list, dim=-1))
        elif self.fusion_method == 'add':
            fused_feature = torch.sum(torch.stack(h_list), dim=0)
        elif self.fusion_method == 'max':
            fused_feature = torch.max(torch.stack(h_list), dim=0).values
        
        
        
        # Cls
        if not self.no_graph:
            t_logit, v_logit, a_logit = None, None, None
            if t_graph_out is not None:
                t_logit = self.t_cls_layer(t_graph_out) # (N, nclass)
            if v_graph_out is not None:
                v_logit = self.v_cls_layer(v_graph_out)
            if a_graph_out is not None:
                a_logit = self.a_cls_layer(a_graph_out)
            fused_logit = self.fusion_cls_layer(fused_feature)
        else:
            t_logit = self.t_cls_layer(feature_t)
            v_logit = self.v_cls_layer(feature_v)
            a_logit = self.a_cls_layer(feature_a)
            fused_logit = self.fusion_cls_layer(fused_feature)

        return fused_logit, t_logit, v_logit, a_logit, fused_feature
    