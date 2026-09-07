import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from itertools import permutations, product


class DynamicGating(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        z = F.sigmoid(self.W(x))
        return z * x 


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, model_dim, dropout=0.1):
        super().__init__()
        assert model_dim % num_heads == 0
        self.dim_per_head = model_dim // num_heads
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.proj_q = nn.Linear(model_dim, model_dim)
        self.proj_k = nn.Linear(model_dim, model_dim)
        self.proj_v = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(model_dim, model_dim)

    def forward(self, q, k, v, mask=None):
        '''
        input shape: (B, L, D)
        output shape: (B, L, D)
        '''
        batch_size = q.size(0)
        
        q = self.proj_q(q).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2) # (B, H, L, d)
        k = self.proj_k(k).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        v = self.proj_v(v).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)

        q = q / math.sqrt(self.dim_per_head) 
        attn_scores = torch.matmul(q, k.transpose(2, 3)) # (B, H, L, L)

        if mask is not None: # (B, L)
            mask = (mask == 0)
            mask = mask.unsqueeze(1).unsqueeze(2).expand_as(attn_scores) # (B, 1, 1, L) -> (B, H, L, L)
            attn_scores = attn_scores.masked_fill(mask, -1e10)
        
        attn_scores = F.softmax(attn_scores, dim=-1)
        drop_attn_scores = self.dropout(attn_scores)
        output = torch.matmul(drop_attn_scores, v).transpose(1, 2).contiguous().view(batch_size, -1, self.model_dim) # (B, L, D)
        return self.proj_out(output)
    

class FeedForwardNet(nn.Module):

    def __init__(self, model_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(model_dim, ff_dim)
        self.linear_2 = nn.Linear(ff_dim, model_dim)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.actv = nn.GELU(approximate='tanh')
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        mid = self.dropout_1(self.actv(self.linear_1(self.layer_norm(x))))
        output = self.dropout_2(self.actv(self.linear_2(mid)))
        return output + x


class PositionalEncoding(nn.Module):

    def __init__(self, model_dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, model_dim) # (max_len, D)
        position = torch.arange(0, max_len).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp((torch.arange(0, model_dim, 2, dtype=torch.float) * -(math.log(10000.0) / model_dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, D)

        self.register_buffer('pe', pe) 

    def forward(self, x): 
        '''
        input shape: (B, L, D)
        '''
        L = x.size(1)
        pos_emb = self.pe[:, :L, :] 
        x += pos_emb
        return x


class TransformerLayer(nn.Module):

    def __init__(self, model_dim, ff_dim, num_heads, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(num_heads, model_dim, dropout)
        self.feed_forward = FeedForwardNet(model_dim, ff_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, mask):
        context = self.self_attn(input, input, input, mask)
        output = self.dropout(context) + input
        return self.feed_forward(output)
    

class TransformerBasedContext(nn.Module):

    def __init__(self, hidden_dim, ff_dim, num_heads, dropout):
        super().__init__()
        self.gate = DynamicGating(hidden_dim)
        self.pos_emb = PositionalEncoding(hidden_dim)
        self.trans_layer = TransformerLayer(hidden_dim, ff_dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, speaker_emb):
        x = self.gate(x)
        x += speaker_emb
        x = self.pos_emb(x)
        x = self.dropout(x)
        x = self.trans_layer(x, mask)
        return x
            
                  
class GatedFusion(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, h_list): 
        h_list = [h.unsqueeze(0) for h in h_list] #(1, N, D)
        h = torch.cat(h_list, dim=0)
        h_proj = torch.cat([self.W(h) for h in h_list], dim=0)
        g = F.softmax(h_proj, dim=0) 
        h_out = torch.sum(h * g, dim=0) 
        return h_out


class EdgeWeightGen(nn.Module):

    def __init__(self, hidden_dim=512, shared_dim=128):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, shared_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(2 * shared_dim, shared_dim),
            nn.LeakyReLU(),
            nn.Linear(shared_dim, 1)
        )

    def forward(self, scr_embeddings, target_embeddings):
        '''
        scr: (E, D), target: (E, D)
        return: edge_weight (E, )
        '''
        shared_src = F.normalize(self.proj(scr_embeddings), dim=-1)
        shared_target = F.normalize(self.proj(target_embeddings), dim=-1)

        emb_cat = torch.cat([shared_src, shared_target], dim=-1) # (E, 2D)
        edge_weight = F.sigmoid(self.mlp(emb_cat).squeeze(-1)) # (E, )

        return edge_weight


class CrossModalGraphLayer(nn.Module):
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W1 = nn.Linear(in_features, out_features, bias=False)
        self.W2 = nn.Linear(2 * out_features, out_features, bias=False)

    def forward(self, input, adj):
        proj_input = self.W1(input)
        neighboor = torch.spmm(adj, proj_input)
         
        updated_feature = F.leaky_relu(self.W2(torch.cat([input + neighboor, input * neighboor], dim=-1)))
        return updated_feature


class CrossModalGraph(nn.Module):

    def __init__(self, hidden_dim, num_layers, no_cuda):
        super().__init__()
        self.num_layers = num_layers
        self.no_cuda = no_cuda

        self.edge_weight_gen = EdgeWeightGen(hidden_dim=hidden_dim, shared_dim=hidden_dim // 4)

        self.graph_layers = nn.ModuleList([
            CrossModalGraphLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])


    def forward(self, feature_tuple, dia_lens, win_p, win_f, edge_index=None):
        num_modals = len(feature_tuple)
        feature = torch.cat(feature_tuple, dim=0)
       
        if edge_index is None:
            edge_index = self.build_edge_index(num_modals, dia_lens, win_p, win_f)
        
        src, target = edge_index[0], edge_index[1]
        edge_weight = self.edge_weight_gen(feature[src], feature[target])

        adj_weight = self.build_adj_matrix(
            edge_index,
            edge_weight=edge_weight,
            num_nodes=feature.size(0),
            no_cuda=self.no_cuda)
        for i in range(self.num_layers):
            feature = self.graph_layers[i](feature, adj_weight)
        feat_tuple = torch.chunk(feature, num_modals, dim=0)

        return feat_tuple, edge_index


    def build_adj_matrix(self, edge_index, edge_weight=None, num_nodes=100, no_cuda=False):
        if edge_weight is not None:
            edge_weight = edge_weight.squeeze()
        else:
            edge_weight = torch.ones(edge_index.size(1)).cuda() if not no_cuda else torch.ones(edge_index.size(1))

        adj_sparse = torch.sparse_coo_tensor(edge_index, edge_weight, size=(num_nodes, num_nodes))
        adj = adj_sparse.to_dense()
        row_sum = torch.sum(adj, dim=1)
        d_inv_sqrt = torch.pow(row_sum, -0.5)
        d_inv_sqrt[d_inv_sqrt == float("inf")] = 0
        d_inv_sqrt_mat = torch.diag_embed(d_inv_sqrt)
        gcn_fact = torch.matmul(d_inv_sqrt_mat, torch.matmul(adj, d_inv_sqrt_mat))

        if not no_cuda and torch.cuda.is_available():
            gcn_fact = gcn_fact.cuda()

        return gcn_fact


    def build_edge_index(self, num_modal, dia_lens, win_p, win_f):
        index_inter = []
        all_dia_len = sum(dia_lens)
        all_nodes = list(range(all_dia_len * num_modal))
        nodes_uni = [None] * num_modal

        for m in range(num_modal):
            nodes_uni[m] = all_nodes[m * all_dia_len:(m + 1) * all_dia_len]

        start = 0
        for dia_len in dia_lens:
            for m, n in permutations(range(num_modal), 2):

                for j, node_m in enumerate(nodes_uni[m][start:start + dia_len]):
                    if win_p == -1 and win_f == -1:
                        nodes_n = nodes_uni[n][start:start + dia_len]
                    elif win_p == -1:
                        nodes_n = nodes_uni[n][start:min(start + dia_len, start + j + win_f + 1)]
                    elif win_f == -1:
                        nodes_n = nodes_uni[n][max(start, start + j - win_p):start + dia_len]
                    else:
                        nodes_n = nodes_uni[n][max(start, start + j - win_p):min(start + dia_len, start + j + win_f + 1)]
                    index_inter.extend(list(product([node_m], nodes_n)))
            start += dia_len
        edge_index = (torch.tensor(index_inter).permute(1, 0).cuda() if not self.no_cuda else torch.tensor(index_inter).permute(1, 0))

        return edge_index



