

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm

from torch.distributions import Categorical

import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.distributions import Categorical


class PhysicsGNN(nn.Module):
    
    def __init__(self, node_features_dim, hidden_dim, num_heads, num_layers, dropout=0.05, num_nodes=20):
        super(PhysicsGNN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.hidden_dim_with_heads = hidden_dim * num_heads
        self.num_nodes = num_nodes


        self.node_type_dim = 16
        self.node_embedding = nn.Embedding(num_nodes, self.node_type_dim)
        

        self.raw_feature_encoder = nn.Sequential(
            nn.Linear(node_features_dim + self.node_type_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        

        self.input_proj = nn.Linear(hidden_dim, self.hidden_dim_with_heads)


        self.layers = nn.ModuleList()
        self.layers.append(GATConv(hidden_dim, hidden_dim, heads=num_heads))
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(self.hidden_dim_with_heads, hidden_dim, heads=num_heads))

        self.gru = nn.GRUCell(
            input_size=self.hidden_dim_with_heads,
            hidden_size=self.hidden_dim_with_heads
        )
        

        self.output_norm = nn.LayerNorm(self.hidden_dim_with_heads)

        self._cached_edge_index = {}

    def _get_batched_edge_index(self, edge_index_static, B_T, N, device):
        if B_T in self._cached_edge_index:
            return self._cached_edge_index[B_T]
        
        num_edges = edge_index_static.shape[1]
        edge_index_batch = edge_index_static.repeat(1, B_T).to(device)
        offsets = torch.arange(0, B_T, device=device) * N
        offsets_repeated = torch.repeat_interleave(offsets, num_edges)
        edge_index_batch = edge_index_batch + offsets_repeated
        
        self._cached_edge_index[B_T] = edge_index_batch
        return edge_index_batch

    def _gat_forward(self, x_encoded, edge_index, B_T, N):
        
        x_flat = x_encoded.view(B_T * N, -1)
        edge_index_batch = self._get_batched_edge_index(edge_index, B_T, N, x_encoded.device)
        
        residual_base = self.input_proj(x_flat)
        
        h = x_flat
        for i in range(self.num_layers):
            h_prev_layer = h
            h = self.layers[i](h, edge_index_batch)
            h = F.elu(h)
            h = F.layer_norm(h, h.shape[-1:])

            if h_prev_layer.shape == h.shape:
                h = h + h_prev_layer
            if i < self.num_layers - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)
        

        h = h + residual_base
        h = F.layer_norm(h, h.shape[-1:])
        
        return h.view(B_T, N, -1)

    def forward(self, x, edge_index, h_prev=None):
        """
        前向传播（支持时序动力学）
        
        Args:
            x: 输入状态 [B*T, N, F] 或 [B, T, N, F]
            edge_index: 边索引
            h_prev: 上一时刻隐藏状态 [B, N, D] 或 None
            
        Returns:
            h_nodes: 节点表征 [B*T, N, D] 或 [B, N, D]
            h_global: 图级表征
            h_current: 当前隐藏状态（用于下一时刻）
        """

        if x.dim() == 4:

            B, T, N, feat_dim = x.shape
            return self._forward_temporal(x, edge_index, h_prev)
        else:

            B_T, N, feat_dim = x.shape
            

            node_ids = torch.arange(N, device=x.device)
            node_emb = self.node_embedding(node_ids)

            node_emb = node_emb.unsqueeze(0).expand(B_T, -1, -1)
            

            x_with_emb = torch.cat([x, node_emb], dim=-1)
            

            x_encoded = self.raw_feature_encoder(x_with_emb)
            

            h_spatial = self._gat_forward(x_encoded, edge_index, B_T, N)
            

            if h_prev is not None:


                h_prev_flat = h_prev.view(-1, self.hidden_dim_with_heads)
                h_spatial_flat = h_spatial.view(-1, self.hidden_dim_with_heads)
                

                if h_prev_flat.shape[0] == h_spatial_flat.shape[0]:
                    h_nodes_flat = self.gru(h_spatial_flat, h_prev_flat)
                    h_nodes = h_nodes_flat.view(B_T, N, -1)
                else:
                    h_nodes = h_spatial
            else:
                h_nodes = h_spatial
            

            h_mean = h_nodes.mean(dim=1)
            h_max = h_nodes.max(dim=1)[0]
            h_global = torch.cat([h_mean, h_max], dim=-1)
            

            h_nodes = self.output_norm(h_nodes)

            return h_nodes, h_global, h_nodes

    def _forward_temporal(self, x, edge_index, h_prev=None):
        """
        时序前向传播：逐时间步处理
        
        Args:
            x: [B, T, N, F]
            edge_index: 边索引
            h_prev: [B, N, D] 或 None
        """
        B, T, N, feat_dim = x.shape
        

        node_ids = torch.arange(N, device=x.device)
        node_emb = self.node_embedding(node_ids)
        node_emb = node_emb.unsqueeze(0).expand(B, -1, -1)
        
        h_nodes_all = []
        h_current = h_prev
        
        for t in range(T):
            x_t = x[:, t, :, :]
            

            x_with_emb = torch.cat([x_t, node_emb], dim=-1)
            

            x_encoded = self.raw_feature_encoder(x_with_emb)
            

            h_spatial = self._gat_forward(x_encoded, edge_index, B, N)
            

            if h_current is not None:
                h_spatial_flat = h_spatial.view(B * N, -1)
                h_current_flat = h_current.view(B * N, -1)
                h_nodes_t = self.gru(h_spatial_flat, h_current_flat).view(B, N, -1)
            else:
                h_nodes_t = h_spatial
            
            h_nodes_all.append(h_nodes_t)
            h_current = h_nodes_t
        

        h_nodes = torch.stack(h_nodes_all, dim=1)
        h_nodes_flat = h_nodes.view(B * T, N, -1)
        

        h_mean = h_nodes_flat.mean(dim=1)
        h_max = h_nodes_flat.max(dim=1)[0]
        h_global = torch.cat([h_mean, h_max], dim=-1)
        
        return h_nodes_flat, h_global, h_current


class ContextEncoder(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(ContextEncoder, self).__init__()
        self.output_dim = output_dim
        

        self.attn_gate = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False)
        )
        

        hidden = output_dim * 2
        self.fc_shared = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.fc_mu = nn.Linear(hidden, output_dim)
        self.fc_logvar = nn.Linear(hidden, output_dim)
        

        self.layer_norm = nn.LayerNorm(output_dim)
    
    def reparameterize(self, mu, logvar):
        
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:

            return mu
    
    def forward(self, h_nodes):
        """
        前向传播
        
        Args:
            h_nodes: [B*T, N, D] 节点表征
            
        Returns:
            z: [B*T, D_ctx] 采样的潜变量
            mu: [B*T, D_ctx] 均值
            logvar: [B*T, D_ctx] 对数方差
        """

        scores = self.attn_gate(h_nodes)
        

        attn_weights = F.softmax(scores, dim=1)
        

        context_vec = (h_nodes * attn_weights).sum(dim=1)
        

        h = self.fc_shared(context_vec)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        

        logvar = torch.clamp(logvar, min=-5.0, max=10)
        
        z = self.reparameterize(mu, logvar)
        z = self.layer_norm(z)
        
        return z, mu, logvar

class ActorCriticPolicy(nn.Module):
    
    def __init__(
        self,
        node_dim=None,
        action_dim=None,
        hidden_dim=64,
        num_heads=4,
        use_context=True,
        context_dim=None,
        dropout=0.1,
        bc_dropout=0.3,
        use_bc=True,
        fusion_method='node_level',
        node_state_dim=None,
        **kwargs
    ):
        super().__init__()

        actual_node_dim = node_dim if node_dim is not None else node_state_dim
        if actual_node_dim is None:
             raise ValueError("ActorCriticPolicy: Must provide `node_dim` or `node_state_dim`.")
        
        self.use_context = use_context
        self.use_bc = use_bc
        self.fusion_method = fusion_method
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.node_dim = actual_node_dim
        self.context_dim = context_dim if context_dim is not None else 0


        if use_context and context_dim is not None:

            self.z_projector = nn.Linear(context_dim, node_dim)
            self.post_context_projector = nn.Linear(context_dim, hidden_dim)
        else:
            self.z_projector = None
            self.post_context_projector = None
        

        self.node_context_proj = nn.Sequential(
            nn.Linear(self.node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        

        self.node_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        

        if use_bc:
            self.bc_dropout = nn.Dropout(bc_dropout)


        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
            nn.LogSoftmax(dim=-1)
        )

    def forward_logits(self, h_nodes, context=None, raw_states=None):
        """Return pre-softmax actor logits for anchored residual decomposition."""
        B, N, D_node = h_nodes.shape

        if context is not None and context.dim() == 3:
            context = context.squeeze(1)

        use_post_pool = self.fusion_method in {"post_pooling", "post-pooling", "post_pool"}

        if (not use_post_pool) and context is not None and self.z_projector is not None:
            z_bias = self.z_projector(context)
            z_bias = z_bias.unsqueeze(1).expand(-1, N, -1)
            h_enriched = h_nodes + z_bias
        else:
            h_enriched = h_nodes

        h_fused = self.node_context_proj(h_enriched)
        if self.use_bc:
            h_fused = self.bc_dropout(h_fused)

        attn_scores = self.node_attention(h_fused)
        attn_weights = F.softmax(attn_scores, dim=1)
        h_graph = (h_fused * attn_weights).sum(dim=1)

        if use_post_pool and context is not None and self.post_context_projector is not None:
            h_graph = h_graph + self.post_context_projector(context)

        logits = h_graph
        for layer in self.actor_head:
            if isinstance(layer, nn.LogSoftmax):
                break
            logits = layer(logits)
        return logits

    def forward(self, h_nodes, context=None, raw_states=None):
        """
        前向传播（节点级先融合）
        
        Args:
            h_nodes: [B, N, D_node] 节点表征
            context: [B, D_ctx] 或 [B, 1, D_ctx] 隐性向量 z
            raw_states: 未使用，保留接口兼容
            
        Returns:
            log_probs: [B, action_dim] 动作对数概率
            None: 已删除 aux_logits
        """
        B, N, D_node = h_nodes.shape
        
        if context is not None and context.dim() == 3:
            context = context.squeeze(1)


        use_post_pool = self.fusion_method in {"post_pooling", "post-pooling", "post_pool"}


        if (not use_post_pool) and context is not None and self.z_projector is not None:

            z_bias = self.z_projector(context)
            z_bias = z_bias.unsqueeze(1).expand(-1, N, -1)
            

            h_enriched = h_nodes + z_bias
        else:
            h_enriched = h_nodes
        

        h_fused = self.node_context_proj(h_enriched)
        

        if self.use_bc:
            h_fused = self.bc_dropout(h_fused)
        

        attn_scores = self.node_attention(h_fused)
        attn_weights = F.softmax(attn_scores, dim=1)
        

        h_graph = (h_fused * attn_weights).sum(dim=1)

        if use_post_pool and context is not None and self.post_context_projector is not None:
            h_graph = h_graph + self.post_context_projector(context)
        

        log_probs = self.actor_head(h_graph)
        
        return log_probs, None


class ResidualLogitHead(nn.Module):
    """Context-only additive residual head Delta(z) for Gate 1 anchoring."""
    def __init__(self, context_dim, action_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, context):
        if context is not None and context.dim() == 3:
            context = context.squeeze(1)
        return self.net(context)


class CrossAttention(nn.Module):
    
    
    def __init__(self, hidden_dim, prompt_dim, num_heads=4):
        super(CrossAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.prompt_dim = prompt_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        

        if prompt_dim != hidden_dim:
            self.prompt_projection = nn.Linear(prompt_dim, hidden_dim)
        else:
            self.prompt_projection = nn.Identity()
        

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, graph_embed, prompt_embed):
        """
        Args:
            graph_embed: 图节点嵌入 [batch_size, num_nodes, hidden_dim]
            prompt_embed: 动态上下文嵌入 [batch_size, num_nodes, prompt_dim] (经过expand)
        Returns:
            output: 注意力输出 [batch_size, num_nodes, hidden_dim]
        """
        batch_size, num_nodes, _ = graph_embed.shape
        

        prompt_embed_projected = self.prompt_projection(prompt_embed)
        

        Q = self.q_proj(graph_embed)
        

        K = self.k_proj(prompt_embed_projected)
        V = self.v_proj(prompt_embed_projected)
        

        if prompt_embed.shape[1] == 1:

            Q = Q.reshape(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
            

            K = K.reshape(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.expand(-1, -1, num_nodes, -1)
            

            V = V.reshape(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.expand(-1, -1, num_nodes, -1)
            
        else:

            Q = Q.reshape(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.reshape(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.reshape(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        

        attention = F.softmax(scores, dim=-1)
        

        output = torch.matmul(attention, V)
        

        output = output.transpose(1, 2).contiguous().reshape(batch_size, num_nodes, -1)
        

        if output.shape[-1] != self.hidden_dim:

            if output.shape[-1] < self.hidden_dim:

                padding_size = self.hidden_dim - output.shape[-1]
                if padding_size > 0:
                    padding = torch.zeros(batch_size, num_nodes, padding_size, device=output.device)
                    output = torch.cat([output, padding], dim=-1)
            elif output.shape[-1] > self.hidden_dim:

                output = output[:, :, :self.hidden_dim]
        

        output = self.out_proj(output)
        
        return output


class DecoupledIntegratedGradientsAttribution:
    
    def __init__(self, model_forward, n_steps=50):
        """
        Args:
            model_forward (callable): 
                一个包装函数, 接收 'state' [1, N, F], 
                返回 action_log_probs [1, ActionSpace]
            n_steps (int): 积分步数
        """
        self.model_forward = model_forward
        self.n_steps = n_steps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def attribute(self, state, baseline, target_action_id):
        """
        对单个时间步进行归因
        
        Args:
            state (Tensor): [1, num_nodes, node_dim]
            baseline (Tensor): [1, num_nodes, node_dim]
            target_action_id (int): 目标动作的整数ID
        
        Returns:
            attribution_scores (Tensor): [num_nodes] 每个节点的归因分数
        """
        state = state.to(self.device)
        baseline = baseline.to(self.device)
        
        num_nodes = state.size(1)
        

        alphas = torch.linspace(0, 1, self.n_steps + 1).to(self.device)
        
        integrated_gradients = torch.zeros(num_nodes, device=self.device)
        
        for i in range(self.n_steps):
            alpha = alphas[i]
            

            interpolated_state = baseline + alpha * (state - baseline)
            interpolated_state.requires_grad_(True)
            

            log_probs = self.model_forward(interpolated_state)
            

            target_log_prob = log_probs[0, target_action_id]
            

            if interpolated_state.grad is not None:
                interpolated_state.grad.zero_()
            
            target_log_prob.backward()
            

            node_gradients = interpolated_state.grad.sum(dim=[0, 2])
            integrated_gradients += node_gradients
        

        integrated_gradients = integrated_gradients / self.n_steps
        

        state_diff = (state - baseline).sum(dim=[0, 2])
        attribution_scores = integrated_gradients * state_diff
        
        return attribution_scores.detach()
    
    def _compute_nll_discrepancy(self, state, edge_index, h_prev, actions):
        """
        计算NLL Discrepancy
        直接从tggnn获取action_probs
        
        Args:
            state: [num_nodes, node_dim]
            edge_index: [2, num_edges]
            h_prev: 隐藏状态或None
            actions: [action_dim] one-hot encoded
        
        Returns:
            nll: 标量tensor
        """
        gnn_input_state = state.unsqueeze(0)
        

        _, _, _, action_probs = self.model(
            gnn_input_state,
            edge_index,
            h_prev
        )
        

        action_dist = Categorical(probs=action_probs + 1e-9)
        action_indices = torch.argmax(actions, dim=-1).long()
        log_probs = action_dist.log_prob(action_indices)
        nll = -log_probs
        
        return nll
    

def get_context_aware_model_components(node_features_dim, action_space_size, config):
    

    context_dim = config.get('context_dim', 16)
    hidden_dim = config.get('hidden_dim', 32)
    num_heads = config.get('num_heads', 2)
    num_nodes = config.get('num_nodes', 20)
    

    gnn_output_dim = hidden_dim * num_heads


    tggnn = PhysicsGNN(
        node_features_dim=node_features_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=config.get('num_gat_layers', 1),
        dropout=config.get('dropout', 0.3),
        num_nodes=num_nodes
    )
    

    context_encoder = ContextEncoder(
        input_dim=node_features_dim, 
        output_dim=context_dim
    )


    system_policy = ActorCriticPolicy(
        node_dim=gnn_output_dim,
        context_dim=context_dim,
        action_dim=action_space_size,
        hidden_dim=hidden_dim,
        num_heads=num_heads
    )
    
    return tggnn, context_encoder, system_policy


def initialize_model_params(model):
    for name, p in model.named_parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            if 'bias' in name:
                nn.init.zeros_(p)
            else:
                nn.init.uniform_(p, -0.1, 0.1)


