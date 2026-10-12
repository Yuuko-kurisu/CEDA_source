

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm
import os
import json
import random
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
from .models import initialize_model_params


from .models import PhysicsGNN, ContextEncoder, ActorCriticPolicy, ResidualLogitHead, initialize_model_params, DecoupledIntegratedGradientsAttribution


from .dataloader import ExpertTrajectoryDataset, create_dataloader, create_train_val_dataloaders
from .dataloader import get_state_action_dims_from_dataset, collate_variable_length_trajectories
import logging


logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class FocalLoss(nn.Module):
    """
    Focal Loss: 解决类别不平衡问题
    论文: Focal Loss for Dense Object Detection (Lin et al., 2017)
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        gamma: 聚焦参数，降低易分类样本的权重 (默认 2.0)
        alpha: 类别权重张量 [num_classes] (可选)
        reduction: 'mean', 'sum' 或 'none'
        label_smoothing: 标签平滑系数 (默认 0.0)
    """
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None, 
                 reduction: str = 'mean', label_smoothing: float = 0.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, log_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            log_probs: [N, C] 对数概率 (log_softmax 输出)
            targets: [N] 目标类别索引
        """
        num_classes = log_probs.size(1)
        

        if self.label_smoothing > 0:

            one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)

            smooth_targets = one_hot * (1 - self.label_smoothing) + self.label_smoothing / num_classes

            ce_loss = -(smooth_targets * log_probs).sum(dim=1)

            probs = torch.exp(log_probs)
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            focal_weight = (1 - pt) ** self.gamma
            loss = focal_weight * ce_loss
        else:

            probs = torch.exp(log_probs)
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            focal_weight = (1 - pt) ** self.gamma
            log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            loss = -focal_weight * log_pt
        

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CEDATPipeline:
    
    
    @staticmethod
    def set_all_seeds(seed: int = 42):
        """
        设置所有随机数种子，确保实验可重复性
        
        Args:
            seed: 随机数种子
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"  [Seed] 全局随机数种子已设置: {seed}")
    

    @staticmethod
    def clear_gpu_memory():
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    
    def __init__(self, config, seed: int = 42):
        """
        初始化 CEDA-T Pipeline
        
        Args:
            config: 配置字典
            seed: 随机数种子 (默认 42)
        """

        self.seed = seed
        CEDATPipeline.set_all_seeds(seed)
        

        CEDATPipeline.clear_gpu_memory()
        

        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Initializing CEDA-T Pipeline (V8.2.1) | Device: {self.device} | Seed: {seed}")
        

        self.tggnn = None
        self.context_encoder = None
        self.gate1_raw_x_encoder = None
        self.system_policy = None
        self.physical_policy = None
        self.residual_head = None
        self.gate1_metrics = {}
        self.global_optimizer = None
        self.dataset = None
        self.dataloader = None
        self.results = {}
        

        self.orth_proj = None
        

        self.exp_id = config.get('exp_id', 'default')
        

        if 'free_bits' not in config:
             raise ValueError("[CRITICAL] 'free_bits' is missing in config! "
                              "Posterior collapse protection is mandatory in V10.8. "
                              "Please update your YAML config.")
    
    
    def initialize_components(self, ckg_path: str, trajectories_path: str, collate_fn=None):
        """
        初始化所有组件
        
        Args:
            ckg_path: CKG JSON path
            trajectories_path: Expert trajectories JSON path
            collate_fn: Optional custom collate function for public dataset isolation
        """
        logger.info("=== 1. 初始化组件 (V4.1) ===")
        

        use_balanced = self.config.get('use_balanced_sampling', True)
        val_split = self.config.get('val_split', 0.2)
        num_workers = self.config.get('num_workers', 0)
        
        self.dataloader, self.val_dataloader, self.dataset = create_train_val_dataloaders(
            trajectories_path=trajectories_path,
            ckg_path=ckg_path,
            batch_size=self.config.get('batch_size', 16),
            val_split=val_split,
            use_balanced_sampling=use_balanced,
            seed=self.seed,
            num_workers=num_workers,
            subset_ratio=self.config.get('subset_ratio', 1.0),
            data_scale_factor=self.config.get('data_scale_factor', 1.0),
            collate_fn=collate_fn
        )
        

        self.node_feature_dim = 3
        self.num_nodes = self.dataset.num_nodes             
        self.action_space_size = self.dataset.action_space_size 


        self.edge_index = self.dataset.edge_index.to(self.device)
        
        logger.info(f"  [Dataset] Nodes: {self.num_nodes}, Actions: {self.action_space_size}, FeatDim: {self.node_feature_dim}")
        

        context_dim = self.config.get('context_dim', 32)
        hidden_dim = self.config.get('hidden_dim', 32)
        num_heads = self.config.get('num_heads', 4)
        gnn_output_dim = hidden_dim * num_heads 


        self.tggnn = PhysicsGNN(
            node_features_dim=self.node_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=self.config.get('num_gat_layers', 2),
            dropout=self.config.get('dropout', 0.1),
            num_nodes=self.num_nodes
        ).to(self.device)
        

        self.context_encoder = ContextEncoder(
            input_dim=self.node_feature_dim,
            output_dim=context_dim
        ).to(self.device)


        self.system_policy = ActorCriticPolicy(
            node_dim=gnn_output_dim,
            context_dim=context_dim,
            action_dim=self.action_space_size,
            hidden_dim=hidden_dim,
            use_context=self.config.get('use_context', True),
            use_bc=self.config.get('use_bc', True),
            fusion_method=self.config.get('fusion_method', 'node_level'),
            dropout=self.config.get('dropout', 0.1),
            bc_dropout=self.config.get('bc_dropout', 0.5)
        ).to(self.device)
        
        if self.config.get('gate1_anchored', False):
            self.physical_policy = ActorCriticPolicy(
                node_dim=gnn_output_dim,
                context_dim=None,
                action_dim=self.action_space_size,
                hidden_dim=hidden_dim,
                use_context=False,
                use_bc=self.config.get('use_bc', True),
                fusion_method='node_level',
                dropout=self.config.get('dropout', 0.1),
                bc_dropout=self.config.get('bc_dropout', 0.5)
            ).to(self.device)
            self.residual_head = ResidualLogitHead(
                context_dim=context_dim,
                action_dim=self.action_space_size,
                hidden_dim=hidden_dim,
                dropout=self.config.get('dropout', 0.1)
            ).to(self.device)
            if self.config.get('gate1_residual_raw_x', False):
                self.gate1_raw_x_encoder = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(self.num_nodes * self.node_feature_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.config.get('dropout', 0.1)),
                    nn.Linear(hidden_dim, context_dim)
                ).to(self.device)

        initialize_model_params(self.tggnn)
        initialize_model_params(self.context_encoder)
        initialize_model_params(self.system_policy)
        if self.physical_policy is not None:
            initialize_model_params(self.physical_policy)
        if self.residual_head is not None:
            initialize_model_params(self.residual_head)
        if self.gate1_raw_x_encoder is not None:
            initialize_model_params(self.gate1_raw_x_encoder)


        if not torch.cuda.is_available():
             if self.config.get('allow_cpu', False):
                 logger.warning(" [WARNING] GPU not detected, but 'allow_cpu' is True. Proceeding with CPU (SLOW!).")
                 self.device = torch.device('cpu')
             else:

                 raise RuntimeError("[FATAL] No GPU detected! Terminating to prevent CPU degradation. Set 'allow_cpu': True to override.")


        amp_enabled = (self.device.type == 'cuda') and not self.config.get('disable_amp', False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        logger.info(f' AMP Mixed Precision Enabled: {self.scaler.is_enabled()}')

        logger.info("  [Model] V4.1 端到端架构初始化完成")
    
    def _compute_attributions(self, h_nodes, context, log_probs):
        """
        计算归因分数
        
        Args:
            h_nodes: GNN 节点嵌入
            context: 上下文特征
            log_probs: 策略输出
            
        Returns:
            dict: 包含归因信息的字典
        """

        attribution_dict = {}
        
        if self.config.get('attribution_method') == 'integrated_gradients':

            from captum.attr import IntegratedGradients
            
            ig = IntegratedGradients(self.system_policy)
            

            if h_nodes is not None:
                attributions_gnn = ig.attribute(h_nodes, target=log_probs.argmax(dim=-1))
                attribution_dict['gnn_attr'] = attributions_gnn
            

            if context is not None:
                attributions_context = ig.attribute(context, target=log_probs.argmax(dim=-1))
                attribution_dict['context_attr'] = attributions_context
        
        else:

            if h_nodes is not None:
                h_nodes.requires_grad_(True)
            if context is not None:
                context.requires_grad_(True)
            

            log_probs_grad, _ = self.system_policy(h_nodes, context)
            

            target = log_probs_grad.argmax(dim=-1)
            loss = F.nll_loss(log_probs_grad, target)
            loss.backward()
            
            if h_nodes is not None and h_nodes.grad is not None:
                attribution_dict['gnn_attr'] = h_nodes.grad.abs()
            
            if context is not None and context.grad is not None:
                attribution_dict['context_attr'] = context.grad.abs()
        
        return attribution_dict


    def _gate1_encode_context(self, states_flat):

        if self.config.get('gate1_residual_raw_x', False) and self.gate1_raw_x_encoder is not None:
            z = self.gate1_raw_x_encoder(states_flat)
            mu = torch.zeros_like(z)
            logvar = torch.zeros_like(z)
            return z, mu, logvar

        z, mu, logvar = self.context_encoder(states_flat)
        return z, mu, logvar

    def _gate1_forward_total(self, states):
        B, T, N, F_dim = states.shape
        h_phy, _, _ = self.tggnn(states, self.edge_index)
        if not self.config.get('use_physics_gnn', True):
            h_phy = torch.zeros_like(h_phy)
        h_phy_flat = h_phy.reshape(B * T, N, -1)
        phy_logits = self.physical_policy.forward_logits(h_phy_flat, None)
        states_flat = states.view(B * T, N, F_dim)
        z, mu, logvar = self._gate1_encode_context(states_flat)
        delta_logits = self.residual_head(z)
        total_log_probs = F.log_softmax(phy_logits + delta_logits, dim=-1).view(B, T, -1)
        return total_log_probs, phy_logits.view(B, T, -1), delta_logits.view(B, T, -1), mu.view(B, T, -1), logvar.view(B, T, -1)

    def _gate1_accuracy_on_loader(self, loader, target_key='actions_exp'):
        self.tggnn.eval()
        self.context_encoder.eval()
        if self.gate1_raw_x_encoder is not None:
            self.gate1_raw_x_encoder.eval()
        self.physical_policy.eval()
        self.residual_head.eval()
        total_correct = 0
        total_steps = 0
        with torch.no_grad():
            for batch in loader:
                states = batch['states'].to(self.device)
                targets = batch.get(target_key, batch['actions']).to(self.device)
                masks = batch['masks'].to(self.device)
                log_probs, _, _, _, _ = self._gate1_forward_total(states)
                preds = log_probs.argmax(dim=-1)
                total_correct += ((preds == targets) & masks).sum().item()
                total_steps += masks.sum().item()
        return total_correct / max(1, total_steps)

    def evaluate_accuracy_gate1(self, verbose=True, use_train_loader=False, update_last_metrics=True):
        loader = self.dataloader if use_train_loader else getattr(self, 'val_dataloader', self.dataloader)
        acc = self._gate1_accuracy_on_loader(loader, target_key='actions_exp')
        if update_last_metrics:
            self.last_eval_metrics = {'gate1_accuracy': acc}
        if verbose:
            logger.info(f"  [Gate1 Eval] Anchored NTP Accuracy: {acc:.4f}")
        self.tggnn.train()
        self.context_encoder.train()
        self.physical_policy.train()
        self.residual_head.train()
        return acc

    def run_training_gate1_anchored(self):
        logger.info("=" * 50)
        logger.info("Gate 1: Anchored pi_sys + residual Delta(z) training")
        logger.info("=" * 50)
        if self.physical_policy is None or self.residual_head is None:
            raise RuntimeError("gate1_anchored=True requires physical_policy and residual_head initialization")

        train_loader = self.dataloader
        stage_a_epochs = int(self.config.get('gate1_stageA_epochs', 5))
        stage_b_epochs = int(self.config.get('gate1_stageB_epochs', 5))
        lr_a = float(self.config.get('gate1_lr_stageA', self.config.get('lr_policy', 1e-3)))
        lr_b = float(self.config.get('gate1_lr_stageB', self.config.get('lr_policy', 1e-3)))
        beta_kl = float(self.config.get('beta_kl', 0.0))
        free_bits = float(self.config.get('free_bits', 0.0))
        nonres_delta_l2 = float(self.config.get('gate1_nonres_delta_l2', 0.01))

        opt_a = optim.Adam(list(self.tggnn.parameters()) + list(self.physical_policy.parameters()), lr=lr_a)
        stage_a_last = {}
        for epoch in range(stage_a_epochs):
            self.tggnn.train()
            self.physical_policy.train()
            total_loss = 0.0
            total_correct_sys = 0
            total_sys = 0
            residual_correct_exp = 0
            residual_correct_sys = 0
            residual_total = 0
            for batch in tqdm(train_loader, desc=f"Gate1 StageA {epoch+1}/{stage_a_epochs}"):
                states = batch['states'].to(self.device)
                actions_sys = batch.get('actions_sys', batch['actions']).to(self.device)
                actions_exp = batch.get('actions_exp', batch['actions']).to(self.device)
                residual_mask = batch.get('residual_mask', torch.zeros_like(actions_sys, dtype=torch.float32)).to(self.device)
                masks = batch['masks'].to(self.device)
                B, T, N, _ = states.shape

                opt_a.zero_grad()
                h_phy, _, _ = self.tggnn(states, self.edge_index)
                if not self.config.get('use_physics_gnn', True):
                    h_phy = torch.zeros_like(h_phy)
                logits = self.physical_policy.forward_logits(h_phy.reshape(B*T, N, -1), None).view(B, T, -1)
                log_probs = F.log_softmax(logits, dim=-1)
                active_log_probs = log_probs[masks]
                active_sys = actions_sys[masks]
                loss = F.nll_loss(active_log_probs, active_sys)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.tggnn.parameters()) + list(self.physical_policy.parameters()), max_norm=1.0)
                opt_a.step()

                preds = log_probs.argmax(dim=-1)
                total_correct_sys += ((preds == actions_sys) & masks).sum().item()
                total_sys += masks.sum().item()
                res_bool = (residual_mask > 0.5) & masks
                if res_bool.any():
                    residual_correct_exp += ((preds == actions_exp) & res_bool).sum().item()
                    residual_correct_sys += ((preds == actions_sys) & res_bool).sum().item()
                    residual_total += res_bool.sum().item()
                total_loss += loss.item()
            stage_a_last = {
                'stageA_epoch': epoch + 1,
                'stageA_loss': total_loss / max(1, len(train_loader)),
                'agree_phy_sys': total_correct_sys / max(1, total_sys),
                'residual_agree_phy_exp': residual_correct_exp / max(1, residual_total),
                'residual_agree_phy_sys': residual_correct_sys / max(1, residual_total),
                'residual_total': residual_total,
            }
            logger.info(f"[Gate1 StageA] {stage_a_last}")

        for module in (self.tggnn, self.physical_policy):
            module.eval()
            for p in module.parameters():
                p.requires_grad = False

        residual_encoder = self.gate1_raw_x_encoder if self.gate1_raw_x_encoder is not None else self.context_encoder
        residual_encoder_params = list(residual_encoder.parameters())
        opt_b = optim.Adam(residual_encoder_params + list(self.residual_head.parameters()), lr=lr_b)
        stage_b_last = {}
        for epoch in range(stage_b_epochs):
            self.context_encoder.train()
            if self.gate1_raw_x_encoder is not None:
                self.gate1_raw_x_encoder.train()
            self.residual_head.train()
            total_loss = 0.0
            total_nll = 0.0
            total_kl = 0.0
            total_correct = 0
            total_active = 0
            res_delta_sum = 0.0
            res_delta_n = 0
            nonres_delta_sum = 0.0
            nonres_delta_n = 0
            for batch in tqdm(train_loader, desc=f"Gate1 StageB {epoch+1}/{stage_b_epochs}"):
                states = batch['states'].to(self.device)
                actions_exp = batch.get('actions_exp', batch['actions']).to(self.device)
                residual_mask = batch.get('residual_mask', torch.zeros_like(actions_exp, dtype=torch.float32)).to(self.device)
                masks = batch['masks'].to(self.device)
                B, T, N, F_dim = states.shape

                opt_b.zero_grad()
                with torch.no_grad():
                    h_phy, _, _ = self.tggnn(states, self.edge_index)
                    if not self.config.get('use_physics_gnn', True):
                        h_phy = torch.zeros_like(h_phy)
                    phy_logits = self.physical_policy.forward_logits(h_phy.reshape(B*T, N, -1), None)

                states_flat = states.view(B*T, N, F_dim)
                z, mu, logvar = self._gate1_encode_context(states_flat)
                delta_logits = self.residual_head(z)
                log_probs = F.log_softmax(phy_logits + delta_logits, dim=-1).view(B, T, -1)
                active_log_probs = log_probs[masks]
                active_exp = actions_exp[masks]
                loss_nll = F.nll_loss(active_log_probs, active_exp)

                mu_bt = mu.view(B, T, -1)
                logvar_bt = logvar.view(B, T, -1)
                kl = -0.5 * (1 + logvar_bt - mu_bt.pow(2) - logvar_bt.exp()).sum(dim=-1)
                loss_kl_raw = (kl * masks).sum() / (masks.sum() + 1e-8)
                loss_kl = torch.max(loss_kl_raw, torch.tensor(free_bits, device=self.device)) if free_bits > 0 else loss_kl_raw

                delta_norm = delta_logits.view(B, T, -1).norm(dim=-1)
                nonres_bool = (residual_mask <= 0.5) & masks
                loss_nonres = (delta_norm[nonres_bool].pow(2).mean() if nonres_bool.any() else torch.tensor(0.0, device=self.device))
                loss = loss_nll + beta_kl * loss_kl + nonres_delta_l2 * loss_nonres
                loss.backward()
                torch.nn.utils.clip_grad_norm_(residual_encoder_params + list(self.residual_head.parameters()), max_norm=1.0)
                opt_b.step()

                preds = log_probs.argmax(dim=-1)
                total_correct += ((preds == actions_exp) & masks).sum().item()
                total_active += masks.sum().item()
                res_bool = (residual_mask > 0.5) & masks
                if res_bool.any():
                    res_delta_sum += delta_norm[res_bool].sum().item()
                    res_delta_n += res_bool.sum().item()
                if nonres_bool.any():
                    nonres_delta_sum += delta_norm[nonres_bool].sum().item()
                    nonres_delta_n += nonres_bool.sum().item()
                total_loss += loss.item()
                total_nll += loss_nll.item()
                total_kl += loss_kl_raw.item()

            stage_b_last = {
                'stageB_epoch': epoch + 1,
                'stageB_loss': total_loss / max(1, len(train_loader)),
                'stageB_nll': total_nll / max(1, len(train_loader)),
                'stageB_kl_raw': total_kl / max(1, len(train_loader)),
                'train_accuracy': total_correct / max(1, total_active),
                'delta_norm_residual_mean': res_delta_sum / res_delta_n if res_delta_n else None,
                'delta_norm_nonresidual_mean': nonres_delta_sum / nonres_delta_n if nonres_delta_n else None,
                'residual_delta_n': res_delta_n,
                'nonresidual_delta_n': nonres_delta_n,
            }
            logger.info(f"[Gate1 StageB] {stage_b_last}")

        for module in (self.tggnn, self.physical_policy):
            for p in module.parameters():
                p.requires_grad = True

        val_acc = self.evaluate_accuracy_gate1(verbose=False, update_last_metrics=True)
        self.gate1_metrics = {**stage_a_last, **stage_b_last, 'val_accuracy': val_acc}
        self.results.update(self.gate1_metrics)
        logger.info(f"[Gate1 Complete] {self.gate1_metrics}")

    def stage4_attribution_gate1(self, max_samples=None):
        print("\n=== Gate 1 Residual CCI Attribution ===")
        self.tggnn.eval()
        self.context_encoder.eval()
        if self.gate1_raw_x_encoder is not None:
            self.gate1_raw_x_encoder.eval()
        self.physical_policy.eval()
        self.residual_head.eval()

        target_dataset = self.dataset
        if max_samples and max_samples < len(target_dataset):
            from torch.utils.data import Subset
            rng = np.random.default_rng(self.seed)
            indices = rng.choice(len(target_dataset), max_samples, replace=False)
            target_dataset = Subset(target_dataset, indices)
            logger.info(f"[Gate1 CCI] Sampling {max_samples} trajectories for attribution")

        attr_loader = DataLoader(target_dataset, batch_size=1, shuffle=False, collate_fn=collate_variable_length_trajectories)
        R_gnn_list = []
        R_context_list = []
        total_ctx = 0.0
        total_phy = 0.0
        trajectory_limit = self.config.get('num_trajectories', None)

        for i, batch in enumerate(attr_loader):
            if trajectory_limit is not None and i >= trajectory_limit:
                break
            states = batch['states'].to(self.device)
            seq_len = int(batch['seq_len'][0].item())
            _, _, N_nodes, F_dim = states.shape
            traj_r_context = []
            traj_r_gnn = []
            for t in range(seq_len):
                state_t = states[:, t, :, :].view(1, N_nodes, F_dim)
                with torch.no_grad():
                    h_base, _, _ = self.tggnn(state_t.unsqueeze(1), self.edge_index)
                    h_base = h_base.reshape(1, N_nodes, -1)
                    phy_logits = self.physical_policy.forward_logits(h_base, None)
                    z_base, _, _ = self._gate1_encode_context(state_t)
                    delta_logits = self.residual_head(z_base)
                    log_base = F.log_softmax(phy_logits + delta_logits, dim=-1)
                    probs_base = torch.exp(log_base)

                cci_context_t = np.zeros(N_nodes)
                cci_gnn_t = np.zeros(N_nodes)
                for node_idx in range(N_nodes):
                    with torch.no_grad():
                        state_masked = state_t.clone()
                        state_masked[0, node_idx, :] = 0.0

                        z_masked, _, _ = self._gate1_encode_context(state_masked)
                        delta_masked = self.residual_head(z_masked)
                        log_ctx_masked = F.log_softmax(phy_logits + delta_masked, dim=-1)
                        cci_context_t[node_idx] = max(0.0, F.kl_div(log_ctx_masked, probs_base, reduction='sum', log_target=False).item())

                        h_masked, _, _ = self.tggnn(state_masked.unsqueeze(1), self.edge_index)
                        h_masked = h_masked.reshape(1, N_nodes, -1)
                        phy_logits_masked = self.physical_policy.forward_logits(h_masked, None)
                        log_phy_masked = F.log_softmax(phy_logits_masked + delta_logits, dim=-1)
                        cci_gnn_t[node_idx] = max(0.0, F.kl_div(log_phy_masked, probs_base, reduction='sum', log_target=False).item())

                traj_r_context.append(cci_context_t)
                traj_r_gnn.append(cci_gnn_t)
                total_ctx += float(np.sum(cci_context_t))
                total_phy += float(np.sum(cci_gnn_t))
            R_context_list.append(traj_r_context)
            R_gnn_list.append(traj_r_gnn)

        channel_share = total_ctx / (total_ctx + total_phy + 1e-12)
        self.results.update({
            'R_gnn': R_gnn_list,
            'R_context': R_context_list,
            'CCI_kl': R_context_list,
            'gate1_channel_share_context': channel_share,
            'gate1_channel_share_physics': 1.0 - channel_share,
            'node_names_ordered': self.dataset.node_column_order,
        })
        logger.info(f"[Gate1 CCI] context_share={channel_share:.4f}, context_sum={total_ctx:.6f}, physics_sum={total_phy:.6f}")
        return R_gnn_list, R_context_list
    def evaluate_accuracy(self, verbose=True, use_train_loader=False, update_last_metrics=True):
        if self.config.get('gate1_anchored', False):
            return self.evaluate_accuracy_gate1(verbose=verbose, use_train_loader=use_train_loader, update_last_metrics=update_last_metrics)
        """
        评估当前的 NTP (Next Token Prediction) 准确率
        使用独立验证集，或训练集（用于诊断）
        """
        self.tggnn.eval()
        self.context_encoder.eval()
        self.system_policy.eval()
        
        total_correct = 0
        total_steps = 0
        

        if use_train_loader:
            val_loader = self.dataloader
        else:
            val_loader = getattr(self, 'val_dataloader', self.dataloader)
        
        with torch.no_grad():
            for batch in val_loader:
                states = batch['states'].to(self.device)
                actions = batch['actions'].to(self.device)
                masks = batch['masks'].to(self.device)
                
                B, T, N, F_dim = states.shape
                

                h_phy, h_global, _ = self.tggnn(states, self.edge_index)
                

                if not self.config.get('use_physics_gnn', True):
                    h_phy = torch.zeros_like(h_phy)
                

                states_flat = states.view(B*T, N, F_dim)
                mu, logvar, z = self.context_encoder(states_flat)
                D_ctx = z.shape[-1]
                z = z.view(B, T, D_ctx)
                

                h_phy_flat = h_phy.reshape(B*T, N, -1)
                z_flat = z.reshape(B*T, -1)
                
                log_probs_flat, _ = self.system_policy(h_phy_flat, z_flat)
                log_probs = log_probs_flat.view(B, T, -1)
                

                preds = log_probs.argmax(dim=-1)
                

                correct = ((preds == actions) * masks).sum().item()
                total = masks.sum().item()
                
                total_correct += correct
                total_steps += total
                

                if 'cm_accum' not in locals():
                    action4_tp = 0
                    action4_pred = 0
                    action4_total = 0

                    num_classes = self.action_space_size
                    cm_accum = torch.zeros((num_classes, num_classes), dtype=torch.long, device='cpu')
                
                active_preds = preds[masks]
                active_actions = actions[masks]
                

                mask_pred4 = (active_preds == 4)
                mask_true4 = (active_actions == 4)
                
                action4_pred += mask_pred4.sum().item()
                action4_total += mask_true4.sum().item()
                action4_tp += (mask_true4 & mask_pred4).sum().item()


                flat_indices = active_actions * num_classes + active_preds
                counts = torch.bincount(flat_indices, minlength=num_classes**2)
                cm_accum += counts.cpu().view(num_classes, num_classes)

        acc = total_correct / total_steps if total_steps > 0 else 0.0
        

        a4_prec = action4_tp / max(1, action4_pred) if 'action4_pred' in locals() else 0.0
        a4_recall = action4_tp / max(1, action4_total) if 'action4_total' in locals() else 0.0
        
        if update_last_metrics:
            self.last_eval_metrics = {
                'accuracy': acc,
                'action4_precision': a4_prec,
                'action4_recall': a4_recall,
                'action4_pred_count': action4_pred if 'action4_pred' in locals() else 0,
                'confusion_matrix': cm_accum.numpy() if 'cm_accum' in locals() else None
            }

            self.results['val_acc_final'] = acc
            if 'action4_pred' in locals():

                 audit_id = self.config.get('audit_action_id', 4) 
                 self.results[f'val_recall_cls_{audit_id}'] = a4_recall
                 self.results[f'val_precision_cls_{audit_id}'] = a4_prec
        
        if verbose:
            logger.info(f"  [Eval] Current NTP Accuracy: {acc:.4f}")
            logger.info(f"  [Eval] Action 4 Precision: {a4_prec:.1%} ({action4_tp}/{max(1, action4_pred)})")
                
        self.tggnn.train()
        self.context_encoder.train()
        self.system_policy.train()
        return acc

    def run_training(self):
        if self.config.get('gate1_anchored', False):
            return self.run_training_gate1_anchored()
        logger.info("=" * 50)
        logger.info("Stage 3: Training CEDA-T Policy (V8.5 加固版)")
        logger.info("=" * 50)
        

        train_loader = self.dataloader
        if not train_loader: return
        

        action_counts = {}
        total_actions = 0
        for batch in train_loader:
            actions = batch['actions']
            masks = batch['masks']
            active_actions = actions[masks].cpu().numpy()
            for a in active_actions:
                action_counts[int(a)] = action_counts.get(int(a), 0) + 1
                total_actions += 1
        
        logger.info(f' 动作分布:')
        for a_id in sorted(action_counts.keys()):
            freq = action_counts[a_id] / total_actions
            logger.info(f"  Action {a_id}: {action_counts[a_id]} ({freq:.2%})")
        

        num_classes = self.action_space_size
        class_weights = torch.ones(num_classes, device=self.device)
        if action_counts:
            max_count = max(action_counts.values())
            for a_id, count in action_counts.items():
                if a_id < num_classes:

                    class_weights[a_id] = np.sqrt(max_count / (count + 1))

            class_weights = class_weights / class_weights.mean()
            

            logger.info(f' Class weights (sqrt): {class_weights.cpu().numpy()}')
        

        first_batch = next(iter(train_loader))
        states_sample = first_batch['states']
        logger.info(f' Feature stats:')
        logger.info(f"  States shape: {states_sample.shape}")
        logger.info(f"  States mean: {states_sample.mean():.4f}")
        logger.info(f"  States std: {states_sample.std():.4f}")
        logger.info(f"  States min: {states_sample.min():.4f}, max: {states_sample.max():.4f}")
        

        val_loader = getattr(self, 'val_dataloader', None)
        if val_loader:
            val_batch = next(iter(val_loader))
            val_states = val_batch['states']
            print(f' Train Mean={states_sample.mean():.3f}, Val Mean={val_states.mean():.3f}')
            print(f' Train Std={states_sample.std():.3f}, Val Std={val_states.std():.3f}')
            

            mean_diff = abs(states_sample.mean() - val_states.mean())
            std_diff = abs(states_sample.std() - val_states.std())
            if mean_diff > 0.5 or std_diff > 1.0:
                logger.warning(f' Distribution shift detected! MeanDiff={mean_diff:.3f}, StdDiff={std_diff:.3f}')
        
        if states_sample.std() < 0.01:
            logger.warning("  [WARNING] Features have very low variance! Consider re-scaling.")
        

        all_params = list(self.tggnn.parameters()) + \
                    list(self.context_encoder.parameters()) + \
                    list(self.system_policy.parameters())
        

        if self.orth_proj is not None:
            all_params += list(self.orth_proj.parameters())
        
        optimizer = optim.Adam(
            all_params,
            lr=self.config.get('lr_policy', 1e-3),
            weight_decay=self.config.get('weight_decay_policy', 1e-5)
        )


        num_epochs = self.config.get('epochs_policy', 200)
        beta_kl_target = self.config.get('beta_kl', 0.01)
        gamma_orth_target = self.config.get('gamma_orth', 0.1)
        

        two_stage_enabled = self.config.get('two_stage', False)
        stage1_epochs = self.config.get('stage1_epochs', 8) if two_stage_enabled else 0
        accumulation_steps = self.config.get('accumulation_steps', 1)
        

        kl_warmup_start = stage1_epochs if two_stage_enabled else self.config.get('kl_warmup_start', 30)
        kl_warmup_end = self.config.get('kl_warmup_end', kl_warmup_start + 10)
        

        total_steps = num_epochs * len(train_loader)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=self.config.get('lr_policy', 1e-3) * 3,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=10.0,
            final_div_factor=100.0
        )
        
        logger.info(f' Two-Stage Training: {two_stage_enabled}, Stage1 Epochs: {stage1_epochs}')
        logger.info(f" OneCycleLR: max_lr={self.config.get('lr_policy', 0.001) * 3}, total_steps={total_steps}")
        logger.info(f' Gradient Accumulation Steps: {accumulation_steps}')
        

        best_val_acc = 0.0
        patience_counter = 0
        early_stop_patience = self.config.get('patience', 15)
        best_model_state = None
        

        prediction_counts = {}
        

        quiet = self.config.get('quiet', False)
        
        self.tggnn.train()
        self.context_encoder.train()
        self.system_policy.train()
        
        if not quiet:
            logger.info(f' Epochs={num_epochs}, Beta_KL_target={beta_kl_target}, Gamma_Orth_target={gamma_orth_target}')
            logger.info(f" KL Warmup: epoch {kl_warmup_start}-{kl_warmup_end}, LR={self.config.get('lr_policy', 0.001)}")
            logger.info(f' Early Stopping patience={early_stop_patience}')


        for epoch in range(num_epochs):
            total_loss = 0.0
            total_nll = 0.0
            total_kl = 0.0
            total_orth = 0.0
            total_correct = 0
            total_samples = 0
            

            class_tp = {}
            class_total = {}
            

            sd_density_sum = 0.0


            is_stage1 = two_stage_enabled and epoch < stage1_epochs
            if is_stage1:
                for param in self.context_encoder.parameters():
                    param.requires_grad = False
                current_beta = 0.0
                current_gamma_orth = 0.0
                if epoch == 0:
                    logger.info(f' Physics-Only Training (Epochs 0-{stage1_epochs - 1})')
            else:
                for param in self.context_encoder.parameters():
                    param.requires_grad = True

                if two_stage_enabled:
                    stage2_progress = min(1.0, (epoch - stage1_epochs) / max(1, num_epochs - stage1_epochs - 5))
                    current_beta = beta_kl_target * stage2_progress
                    current_gamma_orth = gamma_orth_target * stage2_progress
                    if epoch == stage1_epochs:
                        logger.info(f' Decoupling Training Begins (Epochs {stage1_epochs}-{num_epochs - 1})')
                else:

                    if epoch < kl_warmup_start:
                        current_beta = 0.0
                    elif epoch < kl_warmup_end:
                        warmup_duration = max(1, kl_warmup_end - kl_warmup_start)
                        progress = min(1.0, (epoch - kl_warmup_start) / warmup_duration)
                        current_beta = beta_kl_target * (0.5 * (1 - np.cos(np.pi * progress)))
                    else:
                        current_beta = beta_kl_target
                    current_gamma_orth = gamma_orth_target * min(1.0, epoch / max(1, kl_warmup_end))
            

            accum_step = 0            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:

                states = batch['states'].to(self.device)
                actions = batch['actions'].to(self.device)
                masks = batch['masks'].to(self.device)
                sd_signals = batch['sd_signals'].to(self.device)
                
                B, T, N, F_dim = states.shape
                num_active = masks.sum()
                
                optimizer.zero_grad()
                

                if self.tggnn.training and random.random() < 0.3:
                    if F_dim >= 2:
                        states = states.clone()
                        states[..., -1] = 0.0
                

                node_drop_rate = self.config.get('node_drop_rate', 0.1)
                if self.tggnn.training and node_drop_rate > 0:
                    states = states.clone() if not states.requires_grad else states.detach().clone()
                    drop_mask = torch.rand(N, device=self.device) < node_drop_rate
                    states[:, :, drop_mask, :] = 0.0
                

                _amp_ctx = torch.cuda.amp.autocast(enabled=self.scaler.is_enabled())
                _amp_ctx.__enter__()
                if True:
                

                    h_phy, h_global, _ = self.tggnn(states, self.edge_index)
                

                    if not self.config.get('use_physics_gnn', True):
                        h_phy = torch.zeros_like(h_phy)
                

                    phy_bottleneck_drop = self.config.get('phy_bottleneck_drop', 0.0)
                    phy_bottleneck_start = self.config.get('phy_bottleneck_start', 0)
                    if phy_bottleneck_drop > 0 and self.tggnn.training and epoch >= phy_bottleneck_start:
                        h_phy = F.dropout(h_phy, p=phy_bottleneck_drop, training=True)
                

                    states_flat = states.view(B*T, N, F_dim)
                    mu, logvar, z = self.context_encoder(states_flat)
                

                    D_ctx = z.shape[-1]
                    z = z.view(B, T, D_ctx)
                    mu = mu.view(B, T, D_ctx)
                    logvar = logvar.view(B, T, D_ctx)
                

                    h_phy_flat = h_phy.reshape(B*T, N, -1)
                    z_flat = z.reshape(B*T, -1)
                
                    log_probs_flat, _ = self.system_policy(h_phy_flat, z_flat)
                    log_probs = log_probs_flat.view(B, T, -1)

                
                    active_log_probs = log_probs[masks]
                    active_actions = actions[masks]
                    active_sd = sd_signals[masks]
                

                    focal_gamma = self.config.get('focal_gamma', 2.0)
                    label_smoothing = self.config.get('label_smoothing', 0.1)
                    focal_loss_fn = FocalLoss(
                        gamma=focal_gamma, 
                        alpha=class_weights, 
                        reduction='none',
                        label_smoothing=label_smoothing
                    )
                    per_sample_loss = focal_loss_fn(active_log_probs, active_actions)
                

                    sd_weight = torch.where(
                        active_sd == 1, 
                        torch.tensor(3.0, device=self.device), 
                        torch.tensor(1.0, device=self.device)
                    )
                    loss_nll = (per_sample_loss * sd_weight).mean()
                

                    predictions = active_log_probs.argmax(dim=-1)
                    correct = (predictions == active_actions).sum().item()
                    total_correct += correct
                

                    for pred in predictions.cpu().numpy():
                        prediction_counts[int(pred)] = prediction_counts.get(int(pred), 0) + 1
                

                    for a_id in action_counts.keys():
                        mask_class = (active_actions == a_id)
                        if mask_class.sum() > 0:
                            class_total[a_id] = class_total.get(a_id, 0) + mask_class.sum().item()
                            class_tp[a_id] = class_tp.get(a_id, 0) + ((predictions == a_id) & mask_class).sum().item()
                
                    total_samples += active_actions.numel()
                

                    kl_elementwise = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

                    kl_elementwise = torch.clamp(kl_elementwise, min=-100, max=100)
                    kl_per_step = kl_elementwise.sum(dim=-1)
                    loss_kl_raw = (kl_per_step * masks).sum() / (num_active + 1e-8)
                

                    if epoch < kl_warmup_start:
                        kl_weight = 0.0
                    elif epoch < kl_warmup_end:

                        warmup_duration = max(1, kl_warmup_end - kl_warmup_start)
                        progress = min(1.0, (epoch - kl_warmup_start) / warmup_duration)
                        kl_weight = beta_kl_target * (0.5 * (1 - np.cos(np.pi * progress)))
                    else:
                        kl_weight = beta_kl_target
                

                    free_bits = self.config.get('free_bits', 0.0)
                    if free_bits > 0.0:
                        loss_kl = torch.max(loss_kl_raw, torch.tensor(free_bits, device=self.device))
                    else:
                        loss_kl = loss_kl_raw
                

                    l1_lambda = self.config.get('l1_lambda', 0.0)
                    if l1_lambda > 0:

                        l1_loss = l1_lambda * z.abs().mean()
                    else:
                        l1_loss = torch.tensor(0.0, device=self.device)
                

                    if self.config.get('use_physics_gnn', True):

                        h_phy_reshaped = h_phy.view(B, T, N, -1)
                        h_phy_pooled = h_phy_reshaped.mean(dim=2).detach()
                    
                        D_gnn = h_phy_pooled.shape[-1]
                        D_ctx = z.shape[-1]
                    

                        if D_gnn != D_ctx:
                            if self.orth_proj is None or self.orth_proj.in_features != D_ctx or self.orth_proj.out_features != D_gnn:
                                self.orth_proj = nn.Linear(D_ctx, D_gnn).to(self.device)

                                nn.init.xavier_uniform_(self.orth_proj.weight)
                                nn.init.zeros_(self.orth_proj.bias)
                                logger.info(f"  [Orth] 初始化投影层: z ({D_ctx}D) -> ({D_gnn}D)")
                            z_aligned = self.orth_proj(z)
                        else:
                            z_aligned = z
                    

                        h_phy_norm = F.normalize(h_phy_pooled, p=2, dim=-1)
                        z_norm = F.normalize(z_aligned, p=2, dim=-1)
                    

                        orth_sim = F.cosine_similarity(h_phy_norm, z_norm, dim=-1)
                        loss_orth = (orth_sim.abs() * masks).sum() / (num_active + 1e-8)
                    else:
                        loss_orth = torch.tensor(0.0, device=self.device)


                    loss = loss_nll + current_beta * loss_kl + current_gamma_orth * loss_orth + l1_loss
                    _amp_ctx.__exit__(None, None, None)
                

                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.error(f"[FATAL] Loss Collapse at Epoch {epoch}, Batch ?")
                        logger.error(f"  NLL: {loss_nll.item()}, KL: {loss_kl.item()}, Orth: {loss_orth.item()}")
                        logger.error(f"  Beta: {current_beta}, Gamma: {current_gamma_orth}")
                

                accum_step += 1
                self.scaler.scale(loss / accumulation_steps).backward()
                
                if accum_step % accumulation_steps == 0:

                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                    

                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
                    

                    scheduler.step()
                

                total_loss += loss.item()
                total_nll += loss_nll.item()
                total_kl += loss_kl.item()
                total_orth += loss_orth.item()
                

                active_sd = sd_signals[masks]
                sd_ratio = (active_sd != 0).float().mean().item()
                sd_density_sum += sd_ratio
                
                pbar.set_postfix({
                    'NLL': f"{loss_nll.item():.4f}", 
                    'KL': f"{loss_kl.item():.4f}", 
                    'Acc': f"{correct/max(1, active_actions.numel()):.2%}"
                })
            

            avg_nll = total_nll / len(train_loader)
            avg_kl = total_kl / len(train_loader)
            avg_sd = sd_density_sum / len(train_loader)
            epoch_acc = total_correct / max(1, total_samples)
            

            if (epoch + 1) % 10 == 0:
                val_acc_metric = self.evaluate_accuracy(verbose=False, update_last_metrics=True)

                train_eval_acc = self.evaluate_accuracy(verbose=False, use_train_loader=True, update_last_metrics=False)
                gap = epoch_acc - val_acc_metric if val_acc_metric else 0.0
                
                top_class_id = max(prediction_counts, key=prediction_counts.get) if prediction_counts else 0
                top_class_ratio = prediction_counts.get(top_class_id, 0) / max(1, sum(prediction_counts.values()))
                

                val_a4_prec = self.last_eval_metrics.get('action4_precision', 0.0)
                val_a4_recall = self.last_eval_metrics.get('action4_recall', 0.0)
                
                logger.info(f"[VAL_AUDIT] Ep: {epoch+1} | Train_Acc: {epoch_acc:.2f} | Val_Acc: {val_acc_metric:.2f} | Val_A4_Prec: {val_a4_prec:.1%} | Val_A4_Rec: {val_a4_recall:.1%} | Gap: {gap:.2f}")


                cm = self.last_eval_metrics.get('confusion_matrix')
                if cm is not None and not quiet:
                     logger.info(f"  [VAL_CM] Breakdown:\n{cm}")
                

                if not quiet:
                    logger.info(f"[Top-Class Dist] Action {top_class_id}: {top_class_ratio:.1%}")
                    

                    avg_kl_this_epoch = total_kl / max(1, len(train_loader))
                    if epoch >= kl_warmup_start and avg_kl_this_epoch < 0.01:
                        logger.warning(f"[POSTERIOR COLLAPSE] KL={avg_kl_this_epoch:.4f} < 0.01! Z 路径可能未激活，考虑增加 context_dim 或减小 beta")
                    
                    if gap > 0.2:
                        logger.warning("[CRITICAL OVERFIT] Gap > 0.2! 过拟合严重，请检查正则化配置。")
                    if top_class_ratio > 0.5:
                        logger.warning(f' Top-Class Ratio > 50%! 模型可能在盲猜多数类。')
                    

                    minority_classes = [3, 4]
                    minority_recalls = []
                    for mc in minority_classes:
                        if class_total.get(mc, 0) > 0:
                            recall = class_tp.get(mc, 0) / class_total[mc]
                            minority_recalls.append(recall)
                            logger.info(f"[Minority Recall] Action {mc}: {recall:.1%} ({class_tp.get(mc, 0)}/{class_total[mc]})")
                        else:
                            logger.info(f"[Minority Recall] Action {mc}: N/A (无样本)")
                    

                    action4_pred_count = prediction_counts.get(4, 0)
                    action4_tp = class_tp.get(4, 0)
                    action4_precision = action4_tp / max(1, action4_pred_count)
                    logger.info(f' Precision: {action4_precision:.1%} (TP={action4_tp}/Pred={action4_pred_count})')
                    
                    if action4_precision < 0.15 and prediction_counts.get(4, 0) > 100:
                        logger.warning(f' Action 4 Precision < 15%! Z 路径可能在盲猜，考虑减小 beta')
                    
                    avg_minority_recall = np.mean(minority_recalls) if minority_recalls else 0.0
                    

                    if avg_minority_recall < 0.1:
                        boost_factor = 1.5
                        for mc in minority_classes:
                            if mc < len(class_weights):
                                class_weights[mc] *= boost_factor
                        logger.warning(f"[GRADIENT BOOST] Minority Recall < 10%! 临时加大 Action 3/4 权重 x{boost_factor}")
                

                if val_acc_metric > best_val_acc:
                    best_val_acc = val_acc_metric
                    patience_counter = 0

                    best_model_state = {
                        'tggnn': self.tggnn.state_dict(),
                        'context_encoder': self.context_encoder.state_dict(),
                        'system_policy': self.system_policy.state_dict()
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= early_stop_patience:
                        logger.warning(f' Val_Acc 在 {early_stop_patience} 个 Epoch 内无提升，提前结束训练')

                        if best_model_state:
                            self.tggnn.load_state_dict(best_model_state['tggnn'])
                            self.context_encoder.load_state_dict(best_model_state['context_encoder'])
                            self.system_policy.load_state_dict(best_model_state['system_policy'])
                            logger.info(f' 已恢复最佳模型 (Val_Acc: {best_val_acc:.4f})')
                        break
                

                prediction_counts = {}
            else:

                logger.info(f"Epoch {epoch+1}/{num_epochs} | NLL: {avg_nll:.4f} | KL: {avg_kl:.4f} | Acc: {epoch_acc:.4f} | Beta: {kl_weight:.4f}")
        

        if best_model_state:
            self.tggnn.load_state_dict(best_model_state['tggnn'])
            self.context_encoder.load_state_dict(best_model_state['context_encoder'])
            self.system_policy.load_state_dict(best_model_state['system_policy'])
            logger.info(f' 训练结束，已恢复最佳模型 (Val_Acc: {best_val_acc:.4f})')


        final_nll = total_nll / len(train_loader) if len(train_loader) > 0 else 0.0
        final_kl = total_kl / len(train_loader) if len(train_loader) > 0 else 0.0
        final_orth = total_orth / len(train_loader) if len(train_loader) > 0 else 0.0
        final_acc = total_correct / max(1, total_samples)
        
        self.results.update({
            'final_nll': final_nll,
            'final_kl': final_kl,
            'final_orth': final_orth,
            'train_accuracy': final_acc,
            'beta_kl_used': beta_kl_target,
            'gamma_orth_used': gamma_orth_target,
            'epochs_trained': num_epochs
        })
        
        logger.info(f"[Training Complete] NLL: {final_nll:.4f}, KL: {final_kl:.4f}, Acc: {final_acc:.4f}")


    def stage4_attribution(self, max_samples=None):
        if self.config.get('gate1_anchored', False):
            return self.stage4_attribution_gate1(max_samples=max_samples)
        """
        阶段 4: CCI 归因 (KL-based Causal Cognitive Influence)
        使用 KL 散度作为因果干预度量，符合论文定义
        """
        print("\n=== 3. CCI 归因分析 (V7.0 KL-based) ===")
        self.tggnn.eval()
        self.context_encoder.eval()
        self.system_policy.eval()
        

        target_dataset = self.dataset
        if max_samples and max_samples < len(target_dataset):
            from torch.utils.data import Subset
            indices = np.random.choice(len(target_dataset), max_samples, replace=False)
            target_dataset = Subset(target_dataset, indices)
            logger.info(f"[SpeedUp] Sampling {max_samples} trajectories for attribution (Original: {len(self.dataset)})")

        attr_loader = DataLoader(
            target_dataset, batch_size=1, shuffle=False, 
            collate_fn=collate_variable_length_trajectories
        )
        
        R_gnn_list = []
        R_context_list = []
        CCI_kl_list = []
        
        N = self.num_nodes
        D_ctx = self.config.get('context_dim', 32)
        

        trajectory_limit = self.config.get('num_trajectories', None)
        if self.config.get('dry_run', False):
            trajectory_limit = 2
        

        total_kl_mag = 0.0
        debug_limit = 5
        is_dry_run = self.config.get('dry_run', False)

        for i, batch in enumerate(attr_loader):

            if trajectory_limit is not None and i >= trajectory_limit:
                break
                
            states = batch['states'].to(self.device)
            actions = batch['actions'].to(self.device)
            seq_len = batch['seq_len'][0].item()
            
            B, T, N_nodes, F_dim = states.shape
            
            traj_r_gnn = []
            traj_r_context = []
            traj_cci_kl = []
            
            for t in range(seq_len):
                target_action = actions[0, t].item()
                state_t = states[:, t, :, :].view(1, N_nodes, F_dim)


                with torch.no_grad():

                    h_nodes_real, h_global, _ = self.tggnn(state_t.unsqueeze(1), self.edge_index)
                    

                    if not self.config.get('use_physics_gnn', True):
                        h_nodes_real = torch.zeros_like(h_nodes_real)
                    

                    phy_drop_attr = self.config.get('phy_bottleneck_drop', 0.0)
                    if phy_drop_attr > 0:
                        h_nodes_real = F.dropout(h_nodes_real, p=phy_drop_attr, training=True)
                    

                    state_flat = state_t.view(1, N_nodes, F_dim)
                    mu, logvar, z_real_flat = self.context_encoder(state_flat)
                    z_real = z_real_flat.view(1, 1, -1)
                    

                    log_probs_base, _ = self.system_policy(h_nodes_real, z_real)
                    probs_base = torch.exp(log_probs_base)
                    score_base = log_probs_base[0, target_action].item()


                cci_context_t = np.zeros(N_nodes)
                
                for node_idx in range(N_nodes):
                    with torch.no_grad():

                        state_masked = state_t.clone()
                        state_masked[0, node_idx, :] = 0.0
                        

                        state_masked_flat = state_masked.view(1, N_nodes, F_dim)
                        _, _, z_masked_flat = self.context_encoder(state_masked_flat)
                        z_masked = z_masked_flat.view(1, 1, -1)
                        

                        log_probs_masked, _ = self.system_policy(h_nodes_real, z_masked)

                        
                        kl_div = F.kl_div(
                            log_probs_masked, 
                            probs_base, 
                            reduction='sum',
                            log_target=False
                        ).item()
                        
                        cci_context_t[node_idx] = kl_div
                
                traj_r_context.append(cci_context_t)
                traj_cci_kl.append(cci_context_t.copy())


                if i < debug_limit and t == 0:
                    max_kl = np.max(cci_context_t)
                    mean_kl = np.mean(cci_context_t)
                    top_3_idx = np.argsort(cci_context_t)[-3:][::-1]
                    
                    print(f"\n[CCI Case {i}] Action: {target_action}")
                    print(f"  Base Score (LogProb): {score_base:.4f}")
                    print(f"  KL Range: [0, {max_kl:.6f}]")
                    print(f"  Mean KL: {mean_kl:.6f}")
                    print(f"  Top-3 CCI (Node Indices): {top_3_idx}")
                    print(f"  Top-3 KL Values: {cci_context_t[top_3_idx]}")
                    
                    if max_kl < 1e-5:
                        print("  [WARNING] CCI is near zero - Context may not influence decisions!")
                
                total_kl_mag += np.mean(cci_context_t)


                cci_gnn_t = np.zeros(N_nodes)
                
                for node_idx in range(N_nodes):
                    with torch.no_grad():

                        state_masked = state_t.clone()
                        state_masked[0, node_idx, :] = 0.0
                        

                        h_nodes_masked, _, _ = self.tggnn(state_masked.unsqueeze(1), self.edge_index)
                        

                        log_probs_masked, _ = self.system_policy(h_nodes_masked, z_real)
                        

                        kl_div = F.kl_div(
                            log_probs_masked,
                            probs_base,
                            reduction='sum',
                            log_target=False
                        ).item()
                        
                        cci_gnn_t[node_idx] = kl_div
                
                traj_r_gnn.append(cci_gnn_t)

            R_context_list.append(traj_r_context)
            R_gnn_list.append(traj_r_gnn)
            CCI_kl_list.append(traj_cci_kl)
            
            if (i + 1) % 100 == 0:
                print(f"  已处理 {i+1}/{len(attr_loader)} 条轨迹")
            
            if is_dry_run and i >= 10:
                print("  [Dry Run] Limit reached, stopping attribution early.")
                break
        
        avg_kl = total_kl_mag / (len(attr_loader) * seq_len) if len(attr_loader) > 0 else 0
        print(f"\n[Global Stats] Average CCI (KL) Magnitude: {avg_kl:.6f}") 
        print("  CCI 归因分析完成。")
        
        self.results.update({
            'R_gnn': R_gnn_list,
            'R_context': R_context_list,
            'CCI_kl': CCI_kl_list,
            'node_names_ordered': self.dataset.node_column_order
        })
        return R_gnn_list, R_context_list
    def save_model(self, save_dir=None):
        """
        保存训练好的模型
        
        Args:
            save_dir: 保存目录路径（可选）
        """
        if save_dir is None:

            save_dir = self.config.get('save_dir', 'checkpoints')
        

        os.makedirs(save_dir, exist_ok=True)
        

        model_path = os.path.join(save_dir, 'ceda_model.pth')
        

        checkpoint = {
            'tggnn_state_dict': self.tggnn.state_dict(),
            'context_encoder_state_dict': self.context_encoder.state_dict(),
            'system_policy_state_dict': self.system_policy.state_dict(),
            'config': self.config,
            'num_nodes': self.num_nodes,
            'node_feature_dim': self.node_feature_dim,
            'action_space_size': self.action_space_size
        }
        
        torch.save(checkpoint, model_path)
        logger.info(f' 模型已保存到: {model_path}')
        
        return model_path
    def load_model(self, model_path):
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        

        if 'config' in checkpoint:
            saved_conf = checkpoint['config']

            check_keys = ['hidden_dim', 'context_dim', 'num_heads', 'num_gat_layers']
            for k in check_keys:
                if k in saved_conf and k in self.config:
                    if saved_conf[k] != self.config[k]:
                        logger.error(f" Configuration Mismatch for '{k}': Saved={saved_conf[k]}, Current={self.config[k]}")
                        logger.error(' 建议: 1. 更新 config 以匹配权重; 或 2. 设置 force_train=True 重新训练')
                        raise ValueError(f"Model config mismatch: {k}")

        try:

            if 'tggnn_state_dict' in checkpoint:
                self.tggnn.load_state_dict(checkpoint['tggnn_state_dict'])
            

            if 'system_policy_state_dict' in checkpoint:

                self.system_policy.load_state_dict(checkpoint['system_policy_state_dict'])
            elif 'policy_state_dict' in checkpoint:

                self.system_policy.load_state_dict(checkpoint['policy_state_dict'])
            else:
                logger.warning(' Warning: Policy state dict not found in checkpoint')

            if self.context_encoder is not None:
                if 'context_encoder_state_dict' in checkpoint:

                    self.context_encoder.load_state_dict(checkpoint['context_encoder_state_dict'])
                elif 'context_state_dict' in checkpoint:

                    self.context_encoder.load_state_dict(checkpoint['context_state_dict'])
                else:
                     logger.warning(' Warning: Context encoder state dict not found in checkpoint')

        except RuntimeError as e:
            logger.error(f' 权重加载失败: 架构可能已变更。错误详情: {e}')
            logger.error(' 建议: 设置 force_train=True 以重新训练模型')
            raise e
        
        logger.info("Model loaded successfully")
        
        return checkpoint.get('config', {})
    
    def save_results(self, output_dir: str = None, sub_dir: str = None):
        """
        增强版结果保存
        支持实验路径隔离和配置快照
        
        Args:
            output_dir: 基础输出目录（可选）
            sub_dir: 子目录（如 'beta_0.01/seed_42'），用于实验隔离
        """
        if output_dir is None:
            output_dir = self.config.get('output_dir', 'results/attribution')
        

        output_path = Path(output_dir)
        if sub_dir:
            output_path = output_path / sub_dir
        else:

            output_path = output_path / self.exp_id / f"seed_{self.seed}"
        
        output_path.mkdir(parents=True, exist_ok=True)
        

        results_path = output_path / "attribution_results.pt"
        torch.save(self.results, results_path)
        logger.info(f' 归因结果已保存到: {results_path}')
        

        config_snapshot = {
            'config': self.config,
            'seed': self.seed,
            'exp_id': self.exp_id,
            'pipeline_version': 'V8.2.1',
        }
        config_path = output_path / "config_snapshot.json"
        with open(config_path, 'w', encoding='utf-8') as f:

            serializable_config = {}
            for k, v in config_snapshot.items():
                if isinstance(v, dict):
                    serializable_config[k] = {
                        sk: str(sv) if not isinstance(sv, (int, float, str, bool, list, dict, type(None))) else sv
                        for sk, sv in v.items()
                    }
                else:
                    serializable_config[k] = v
            json.dump(serializable_config, f, ensure_ascii=False, indent=2)
        logger.info(f'⚙ 配置快照已保存到: {config_path}')
        

        if 'metrics' in self.results or 'CCI_kl' in self.results:
            metrics_summary = {
                'L1_imitation': self.results.get('imitation_metrics', {}),
                'L2_information': {
                    'final_kl': self.results.get('final_kl', None),
                    'final_orth': self.results.get('final_orth', None),
                },
                'L3_diagnosis': self.results.get('diagnosis_metrics', {}),
            }
            metrics_path = output_path / "metrics_summary.json"
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_summary, f, ensure_ascii=False, indent=2)
            logger.info(f' 指标摘要已保存到: {metrics_path}')
        

        if 'llm_rules' in self.results:
            rules_path = output_path / "llm_rules.json"
            with open(rules_path, 'w', encoding='utf-8') as f:
                json.dump(self.results['llm_rules'], f, ensure_ascii=False, indent=2)
            logger.info(f' LLM 规则已保存到: {rules_path}')
        
        return str(output_path)

    
    def run_ceda_t(self, mode='full', model_path=None, force_train=False):
        """
        运行 CEDA-T Pipeline
        
        Args:
            mode: 'full', 'train', 'load', 'eval'
            model_path: 指定模型路径
            force_train: 是否强制重训
        """

        if model_path is None:
            save_dir = self.config.get('save_dir', 'checkpoints')
            model_path = os.path.join(save_dir, 'ceda_model.pth')


        if mode == 'full' and not force_train:
            if os.path.exists(model_path):
                logger.info(f"⚡ 检测到已有模型: {model_path}")
                logger.info("⚡ 自动切换为 'load' 模式 (跳过训练)...")
                mode = 'load'
            else:
                logger.info(f"Running CEDA-T Pipeline in mode: {mode} (Model not found, starting training)")
        else:
            logger.info(f"Running CEDA-T Pipeline in mode: {mode}")
        

        if mode in ['full', 'train']:
            self.run_training()


            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            self.save_model(save_dir=os.path.dirname(model_path))
            
        elif mode in ['load', 'eval']:
            if os.path.exists(model_path):
                self.load_model(model_path)
            else:
                logger.warning(f'  模型文件不存在: {model_path}，无法加载！')

                if mode == 'load':
                    logger.error(' Critical: Load mode requested but model missing.')
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        

        if mode in ['full', 'load'] and self.config.get('enable_attribution', False):
            try:
                logger.info("开始归因分析...")
                attr_max = self.config.get('attr_max_samples', None)
                self.stage4_attribution(max_samples=attr_max)
                self.save_results()
            except Exception as e:
                logger.warning(f'  归因分析失败（跳过）: {e}')
                import traceback
                traceback.print_exc()
        

        accuracy = self.evaluate_accuracy()
        self.results['accuracy'] = accuracy
        
        logger.info("CEDA-T Pipeline completed")
        
        return self.results


