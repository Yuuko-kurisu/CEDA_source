"""Evaluation metrics and diagnostic reports."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import time

class CEDAMetrics:
    
    
    DEBUG_MODE = False
    DEBUG_LOG_PATH = "metrics_detail_debug.log"
    
    @staticmethod
    def set_debug_mode(enable: bool, log_path: str = None):
        
        CEDAMetrics.DEBUG_MODE = enable
        if log_path:
            CEDAMetrics.DEBUG_LOG_PATH = log_path
        
        if enable:

            with open(CEDAMetrics.DEBUG_LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(f"\n\n{'='*80}\n")
                f.write(f"Metrics Debug Session Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*80}\n")
            print(f' [Metrics] Debug mode ENABLED. Details will be saved to {CEDAMetrics.DEBUG_LOG_PATH}')

    @staticmethod
    def _log(msg: str):
        
        if not CEDAMetrics.DEBUG_MODE:
            return
        

        try:
            with open(CEDAMetrics.DEBUG_LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(msg + "\n")
        except Exception as e:
            print(f' Failed to write to debug log: {e}')


    @staticmethod
    def calc_accuracy(pred_probs: np.ndarray, true_actions: np.ndarray) -> float:
        
        if len(pred_probs) == 0 or len(true_actions) == 0:
            return 0.0
        preds = np.argmax(pred_probs, axis=1)
        correct_mask = (preds == true_actions)
        acc = float(np.mean(correct_mask))

        if CEDAMetrics.DEBUG_MODE:
            n_samples = len(true_actions)
            n_correct = np.sum(correct_mask)
            msg = f"\n[Metric: Accuracy] Detail Report\n"
            msg += f"  - Total Samples: {n_samples}\n"
            msg += f"  - Correct Predictions: {n_correct}\n"
            msg += f"  - Accuracy Calculation: {n_correct} / {n_samples} = {acc:.4f}\n"
            

            wrong_indices = np.where(~correct_mask)[0][:5]
            if len(wrong_indices) > 0:
                msg += "  - Sample Wrong Predictions (Index | True -> Pred):\n"
                for idx in wrong_indices:
                    msg += f"    Idx {idx}: True {true_actions[idx]} -> Pred {preds[idx]} (Conf: {pred_probs[idx, preds[idx]]:.4f})\n"
            CEDAMetrics._log(msg)

        return acc

    @staticmethod
    def calc_ntp(pred_probs: np.ndarray, true_actions: np.ndarray) -> float:
        
        if len(pred_probs) == 0 or len(true_actions) == 0:
            return 0.0
        n_samples = len(true_actions)
        correct_probs = pred_probs[np.arange(n_samples), true_actions.astype(int)]
        ntp = float(np.mean(correct_probs))

        if CEDAMetrics.DEBUG_MODE:
            msg = f"\n[Metric: NTP] Detail Report\n"
            msg += f"  - Definition: Mean probability assigned to the ground truth action.\n"
            msg += f"  - Stats of GT Probabilities:\n"
            msg += f"    Min: {np.min(correct_probs):.4f}\n"
            msg += f"    Max: {np.max(correct_probs):.4f}\n"
            msg += f"    Median: {np.median(correct_probs):.4f}\n"
            msg += f"  - Final NTP: {ntp:.4f}\n"
            CEDAMetrics._log(msg)

        return ntp

    @staticmethod
    def calc_tail_ntp(pred_probs: np.ndarray, true_actions: np.ndarray, tail_percent: float = 0.1) -> float:
        
        if len(pred_probs) == 0 or len(true_actions) == 0:
            return 0.0
        n_samples = len(true_actions)
        correct_probs = pred_probs[np.arange(n_samples), true_actions.astype(int)]
        
        sorted_probs = np.sort(correct_probs)
        k = max(1, int(n_samples * tail_percent))
        tail_probs = sorted_probs[:k]
        tail_ntp = float(np.mean(tail_probs))

        if CEDAMetrics.DEBUG_MODE:
            msg = f"\n[Metric: Tail NTP @ {tail_percent*100}%] Detail Report\n"
            msg += f"  - Considering bottom {k} samples out of {n_samples}.\n"
            msg += f"  - Worst Case Probability: {sorted_probs[0]:.6f}\n"
            msg += f"  - Tail Range (0% - {tail_percent*100}%): [{sorted_probs[0]:.4f}, {sorted_probs[k-1]:.4f}]\n"
            msg += f"  - Final Tail NTP: {tail_ntp:.4f}\n"
            CEDAMetrics._log(msg)

        return tail_ntp

    @staticmethod
    def calc_avg_nll(pred_probs: np.ndarray, true_actions: np.ndarray, eps: float = 1e-8) -> float:
        if len(pred_probs) == 0 or len(true_actions) == 0:
            return 0.0
        n_samples = len(true_actions)
        correct_probs = pred_probs[np.arange(n_samples), true_actions.astype(int)]
        correct_probs = np.clip(correct_probs, eps, 1.0)
        nll = -np.log(correct_probs)
        return float(np.mean(nll))


    @staticmethod
    def calc_ndcg_at_k(attribution_dict: Dict[str, float], gt_keys: List[str], k: int) -> float:
        
        if not attribution_dict or not gt_keys or k <= 0:
            return 0.0
        

        sorted_items = sorted(attribution_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        top_k = sorted_items[:k]
        gt_set = set(gt_keys)
        

        dcg = 0.0
        dcg_details = []
        for i, (item_key, score) in enumerate(top_k):
            rel = 1.0 if item_key in gt_set else 0.0
            gain = rel / np.log2(i + 2)
            dcg += gain
            dcg_details.append({
                'rank': i + 1,
                'node': item_key,
                'score': score,
                'is_hit': rel > 0,
                'gain': gain
            })
            

        n_rel = min(k, len(gt_set))
        idcg = sum(1.0 / np.log2(i + 2) for i in range(n_rel))
        
        ndcg = float(dcg / idcg) if idcg > 0 else 0.0


        if CEDAMetrics.DEBUG_MODE:
            msg = f"\n🔍 [Metric: NDCG@{k}] Calculation Basis\n"
            msg += f"  1. Ground Truth ({len(gt_set)} nodes):\n"
            msg += f"     {list(gt_set)}\n"
            msg += f"  2. Top-{k} Model Predictions & Gains:\n"
            msg += f"     {'Rank':<5} | {'Node Name':<20} | {'Score':<10} | {'Status':<6} | {'Gain (rel/log2)':<15}\n"
            msg += f"     {'-'*70}\n"
            for d in dcg_details:
                status = "✅ HIT" if d['is_hit'] else "❌"
                msg += f"     {d['rank']:<5} | {d['node']:<20} | {d['score']:.4f}     | {status:<6} | {d['gain']:.4f}\n"
            msg += f"  3. Formula:\n"
            msg += f"     DCG  = Sum(Gains) = {dcg:.4f}\n"
            msg += f"     IDCG = Ideal Sum (for {n_rel} relevant items) = {idcg:.4f}\n"
            msg += f"     NDCG = DCG / IDCG = {ndcg:.4f}\n"
            CEDAMetrics._log(msg)

        return ndcg

    @staticmethod
    def calc_precision_at_k(attribution_dict: Dict[str, float], gt_keys: List[str], k: int) -> float:
        
        if not attribution_dict or k <= 0:
            return 0.0
        
        sorted_items = sorted(attribution_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        top_k_keys = [item[0] for item in sorted_items[:k]]
        gt_set = set(gt_keys)
        
        hits = sum(1 for key in top_k_keys if key in gt_set)
        precision = float(hits / k)

        if CEDAMetrics.DEBUG_MODE:
            msg = f"\n🔍 [Metric: Precision@{k}] Calculation Basis\n"
            msg += f"  - Ground Truth: {len(gt_set)} nodes\n"
            msg += f"  - Top-{k} Candidates checked: {top_k_keys}\n"
            msg += f"  - Hits found: {hits}\n"
            msg += f"  - Result: {hits}/{k} = {precision:.4f}\n"
            CEDAMetrics._log(msg)

        return precision

    @staticmethod
    def calc_recall_at_k(attribution_dict: Dict[str, float], gt_keys: List[str], k: int) -> float:

        if not attribution_dict or not gt_keys or k <= 0:
            return 0.0
        
        sorted_items = sorted(attribution_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        top_k_keys = [item[0] for item in sorted_items[:k]]
        gt_set = set(gt_keys)
        
        hits = sum(1 for key in top_k_keys if key in gt_set)
        return float(hits / len(gt_set)) if len(gt_set) > 0 else 0.0

    @staticmethod
    def calc_auc_roc(attribution_dict: Dict[str, float], gt_keys: List[str]) -> float:
        
        if not attribution_dict:
            return 0.0
        
        gt_set = set(gt_keys)
        y_true = []
        y_score = []
        
        for item_key, score in attribution_dict.items():
            y_true.append(1 if item_key in gt_set else 0)
            y_score.append(abs(score))
        
        n_pos = sum(y_true)
        n_neg = len(y_true) - n_pos
        
        if n_pos == 0 or n_neg == 0:
            return 0.0
        
        sorted_pairs = sorted(zip(y_score, y_true), reverse=True)
        
        rank_sum = 0
        for i, (score, label) in enumerate(sorted_pairs):
            if label == 1:
                rank_sum += (i + 1)
        
        auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        auc = float(auc)

        if CEDAMetrics.DEBUG_MODE:
            msg = f"\n🔍 [Metric: AUC-ROC] Calculation Basis\n"
            msg += f"  - Positive Samples (GT): {n_pos}\n"
            msg += f"  - Negative Samples: {n_neg}\n"
            msg += f"  - Rank Sum of Positives: {rank_sum}\n"
            msg += f"  - Formula (Mann-Whitney U): ({rank_sum} - {n_pos}*({n_pos}+1)/2) / ({n_pos}*{n_neg})\n"
            msg += f"  - Result: {auc:.4f}\n"
            CEDAMetrics._log(msg)

        return auc

    @staticmethod
    def calc_map(attribution_dict: Dict[str, float], gt_keys: List[str]) -> float:
        
        if not attribution_dict or not gt_keys:
            return 0.0

        gt_set = set(gt_keys)
        sorted_items = sorted(attribution_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        hits = 0
        precision_sum = 0.0

        for rank, (item_key, _) in enumerate(sorted_items, start=1):
            if item_key in gt_set:
                hits += 1
                precision_sum += hits / rank

        return float(precision_sum / len(gt_set)) if gt_set else 0.0

    @staticmethod
    def calc_auprc(attribution_dict: Dict[str, float], gt_keys: List[str]) -> float:
        
        if not attribution_dict or not gt_keys:
            return 0.0

        gt_set = set(gt_keys)
        sorted_items = sorted(attribution_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        y_true = np.array([1 if key in gt_set else 0 for key, _ in sorted_items], dtype=float)
        if y_true.sum() == 0:
            return 0.0

        tp = np.cumsum(y_true)
        fp = np.cumsum(1.0 - y_true)
        recall = tp / max(float(y_true.sum()), 1e-12)
        precision = tp / np.maximum(tp + fp, 1e-12)

        recall = np.concatenate([[0.0], recall])
        precision = np.concatenate([[1.0], precision])
        return float(np.trapz(precision, recall))

    @staticmethod
    def calc_sparsity(attribution_values: np.ndarray, threshold: float = 1e-4) -> float:
        if len(attribution_values) == 0:
            return 0.0
        n_zero = np.sum(np.abs(attribution_values) < threshold)
        return float(n_zero / len(attribution_values))

    @staticmethod
    def calc_gini_coefficient(values: np.ndarray) -> float:
        if len(values) == 0 or np.sum(np.abs(values)) == 0:
            return 0.0
        values = np.abs(values)
        values = np.sort(values)
        n = len(values)
        gini = (2 * np.sum((np.arange(1, n + 1) * values))) / (n * np.sum(values)) - (n + 1) / n
        return float(gini)


    @staticmethod
    def calc_all_imitation_metrics(pred_probs: np.ndarray, true_actions: np.ndarray) -> Dict[str, float]:

        return {
            'Accuracy': CEDAMetrics.calc_accuracy(pred_probs, true_actions),
            'NTP': CEDAMetrics.calc_ntp(pred_probs, true_actions),
            'Tail_NTP': CEDAMetrics.calc_tail_ntp(pred_probs, true_actions),
            'Avg_NLL': CEDAMetrics.calc_avg_nll(pred_probs, true_actions)
        }

    @staticmethod
    def calc_all_diagnosis_metrics(attribution_dict: Dict[str, float], 
                                   gt_keys: List[str], 
                                   k_values: List[int] = [1, 3, 5, 10, 20]) -> Dict[str, float]:
        metrics = {}
        num_nodes = len(attribution_dict) if attribution_dict else 0
        valid_k_values = [k for k in k_values if k <= num_nodes] if num_nodes > 0 else k_values
        

        if CEDAMetrics.DEBUG_MODE:
            CEDAMetrics._log(f"\n\n{'#'*30} New Diagnosis Evaluation Batch {'#'*30}")
            CEDAMetrics._log(f"Processing {len(valid_k_values)} K-values for {len(gt_keys)} GT nodes.")

        for k in valid_k_values:
            metrics[f'NDCG@{k}'] = CEDAMetrics.calc_ndcg_at_k(attribution_dict, gt_keys, k)
        
        for k in valid_k_values:
            metrics[f'P@{k}'] = CEDAMetrics.calc_precision_at_k(attribution_dict, gt_keys, k)
        
        metrics['AUC_ROC'] = CEDAMetrics.calc_auc_roc(attribution_dict, gt_keys)
        
        if attribution_dict:
            values = np.array(list(attribution_dict.values()))
            metrics['Sparsity'] = CEDAMetrics.calc_sparsity(values)
            metrics['Gini'] = CEDAMetrics.calc_gini_coefficient(values)
        else:
            metrics['Sparsity'] = 0.0
            metrics['Gini'] = 0.0
        
        return metrics
    

    @staticmethod
    def calc_mutual_information_kde(z: np.ndarray, h_phy: np.ndarray, 
                                     num_bins: int = 20) -> float:
        """
        基于 KDE 的互信息上界估计 I(Z; H_phy)
        用于验证 IB 约束是否在压低信息传递
        
        Args:
            z: [N, D_ctx] 隐性向量
            h_phy: [N, D_gnn] 物理表征（建议预先 PCA 降维）
            num_bins: 直方图分桶数
        
        Returns:
            mi_estimate: 互信息估计值 (bits)
        """
        from scipy.stats import entropy
        

        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        h_phy = np.nan_to_num(h_phy, nan=0.0, posinf=0.0, neginf=0.0)
        

        if z.ndim == 1:
            z_1d = z
        else:

            z_1d = z.mean(axis=1) if z.shape[0] > z.shape[1] else z[:, 0]
        
        if h_phy.ndim == 1:
            h_1d = h_phy
        else:
            h_1d = h_phy.mean(axis=1) if h_phy.shape[0] > h_phy.shape[1] else h_phy[:, 0]
        

        n_samples = min(len(z_1d), len(h_1d))
        if n_samples == 0:
            return 0.0
        z_1d = z_1d[:n_samples]
        h_1d = h_1d[:n_samples]
        

        z_1d = np.nan_to_num(z_1d, nan=0.0)
        h_1d = np.nan_to_num(h_1d, nan=0.0)
        

        joint_hist, _, _ = np.histogram2d(z_1d, h_1d, bins=num_bins)
        joint_prob = joint_hist / (joint_hist.sum() + 1e-10)
        

        p_z = joint_prob.sum(axis=1)
        p_h = joint_prob.sum(axis=0)
        

        h_z = entropy(p_z + 1e-10)
        h_h = entropy(p_h + 1e-10)
        h_joint = entropy(joint_prob.flatten() + 1e-10)
        
        mi = h_z + h_h - h_joint
        return float(max(0, mi))
    
    @staticmethod
    def calc_z_entropy(z: np.ndarray, num_bins: int = 20) -> float:
        """
        计算隐变量 z 的熵估计 H(Z)
        
        Args:
            z: [N, D_ctx] 隐性向量
            num_bins: 直方图分桶数
        
        Returns:
            entropy_estimate: 熵估计值 (bits)
        """
        from scipy.stats import entropy
        
        if z.ndim == 1:
            z_1d = z
        else:
            z_1d = z.mean(axis=1)
        
        hist, _ = np.histogram(z_1d, bins=num_bins, density=True)
        hist = hist / (hist.sum() + 1e-10)
        return float(entropy(hist + 1e-10))
    

    @staticmethod
    def calc_jaccard_stability(topk_set_1: set, topk_set_2: set) -> float:
        """
        计算两次归因 Top-K 节点集合的 Jaccard 相似度
        用于衡量 CCI 归因的稳定性
        
        Args:
            topk_set_1: 第一次运行的 Top-K 节点索引集合
            topk_set_2: 第二次运行的 Top-K 节点索引集合
        
        Returns:
            jaccard: 交并比 [0, 1]
        """
        if not topk_set_1 and not topk_set_2:
            return 1.0
        
        intersection = len(topk_set_1 & topk_set_2)
        union = len(topk_set_1 | topk_set_2)
        return float(intersection / union) if union > 0 else 0.0
    
    @staticmethod
    def calc_multi_run_stability(attribution_runs: List[Dict[str, float]], 
                                  k: int = 5) -> Tuple[float, float]:
        """
        计算多次运行的归因稳定性统计
        
        Args:
            attribution_runs: 多次运行的归因结果列表 (List of {node_name: score})
            k: Top-K 参数
        
        Returns:
            mean_jaccard: 平均 Jaccard 相似度
            std_jaccard: Jaccard 标准差
        """
        if len(attribution_runs) < 2:
            return 1.0, 0.0
        

        topk_sets = []
        for attr_dict in attribution_runs:
            sorted_items = sorted(attr_dict.items(), key=lambda x: abs(x[1]), reverse=True)
            topk_keys = set([item[0] for item in sorted_items[:k]])
            topk_sets.append(topk_keys)
        

        jaccards = []
        for i in range(len(topk_sets)):
            for j in range(i + 1, len(topk_sets)):
                sim = CEDAMetrics.calc_jaccard_stability(topk_sets[i], topk_sets[j])
                jaccards.append(sim)
        
        mean_jaccard = float(np.mean(jaccards))
        std_jaccard = float(np.std(jaccards))

        if CEDAMetrics.DEBUG_MODE:
             msg = f"\n[Metric: Stability@K={k}] Detail Report\n"
             msg += f"  - Analyzed {len(attribution_runs)} independent runs.\n"
             msg += f"  - Total Pairwise Comparisons: {len(jaccards)}\n"
             msg += f"  - Pairwise Jaccard Scores: {[f'{v:.2f}' for v in jaccards]}\n"
             msg += f"  - Result: {mean_jaccard:.4f} ± {std_jaccard:.4f}\n"
             CEDAMetrics._log(msg)
             
        return mean_jaccard, std_jaccard
    

    @staticmethod
    def get_metrics_for_experiment(exp_name: str) -> List[str]:
        """
        根据实验编号返回所需指标列表
        
        Args:
            exp_name: 实验编号 (E1, E2, E3, ...)
        
        Returns:
            metric_names: 需要计算的指标名称列表
        """
        experiment_metrics = {
            'E1_IB_Legitimacy': ['MI_z_h', 'H_z', 'NDCG@5', 'KL_divergence'],
            'E2_Stability': ['Jaccard@5', 'Jaccard@10', 'NDCG@5'],
            'E3_Pseudo_Expert': ['context_std', 'NDCG@5', 'Accuracy'],
            'E4_Quantitative': ['NDCG@1', 'NDCG@3', 'NDCG@5', 'NDCG@10', 'AUC_ROC', 'P@5'],
            'E5_Ablation': ['NTP', 'Accuracy', 'NDCG@5', 'Tail_NTP'],
            'E6_Baseline_Comparison': ['NDCG@5', 'AUC_ROC', 'P@5', 'Jaccard@5'],
            'E7_Noise_Robustness': ['NDCG@5', 'Accuracy', 'context_std'],
        }
        return experiment_metrics.get(exp_name, ['NDCG@5'])


class MetricExplainer:
    
    
    @staticmethod
    def explain_diagnosis(attribution_dict, gt_keys, k=5, method_name="Unknown"):
        """
        解释归因诊断指标
        
        Args:
            attribution_dict: {node_name: score} 归因字典
            gt_keys: list of ground truth node names
            k: Top-K
            method_name: 方法名称
            
        Returns:
            report: 文本报告字符串
        """
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append(f"[🔍 Metric Inspector] Method: {method_name}")
        report_lines.append("=" * 70)
        

        report_lines.append(f"\n1️⃣  Ground Truth (Total {len(gt_keys)}):")
        report_lines.append(f"   {gt_keys}")
        

        sorted_items = sorted(attribution_dict.items(), key=lambda x: x[1], reverse=True)
        top_k_items = sorted_items[:k]
        
        report_lines.append(f"\n2️⃣  Top-{k} Predictions:")
        hits = []
        misses = []
        
        for rank, (node_name, score) in enumerate(top_k_items, 1):
            is_hit = node_name in gt_keys
            status = "✅ HIT!" if is_hit else "❌ Miss"
            report_lines.append(f"   Rank {rank}: {node_name} (Score: {score:.4f}) -> {status}")
            
            if is_hit:
                hits.append((rank, node_name, score))
            else:
                misses.append((rank, node_name, score))
        

        report_lines.append(f"\n3️⃣  Statistics:")
        report_lines.append(f"   Found {len(hits)}/{len(gt_keys)} GT nodes in Top-{k}")
        report_lines.append(f"   Precision@{k}: {len(hits)/k:.4f}")
        report_lines.append(f"   Recall@{k}: {len(hits)/len(gt_keys):.4f}" if len(gt_keys) > 0 else "   Recall@{k}: N/A")
        

        if hits:
            report_lines.append(f"\n4️⃣  NDCG@{k} Breakdown:")
            dcg = 0.0
            for rank, node_name, score in hits:
                gain = 1.0 / np.log2(rank + 1)
                dcg += gain
                report_lines.append(f"   Rank {rank} ({node_name}): gain = 1/log2({rank}+1) = {gain:.4f}")
            

            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt_keys), k)))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            
            report_lines.append(f"   DCG@{k} = {dcg:.4f}")
            report_lines.append(f"   IDCG@{k} = {idcg:.4f}")
            report_lines.append(f"   NDCG@{k} = {ndcg:.4f}")
        
        report_lines.append("=" * 70)
        
        return "\n".join(report_lines)
    
    @staticmethod
    def explain_imitation(pred_probs, true_actions, batch_indices=None):
        """
        解释模仿学习性能
        
        Args:
            pred_probs: (N, num_actions) 预测概率
            true_actions: (N,) 真实动作
            batch_indices: (N,) 样本索引（可选）
            
        Returns:
            report: 文本报告字符串
        """
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("[🔍 Imitation Inspector]")
        report_lines.append("=" * 70)
        
        pred_actions = pred_probs.argmax(dim=-1)
        correct_mask = (pred_actions == true_actions)
        
        num_correct = correct_mask.sum().item()
        num_total = len(true_actions)
        accuracy = num_correct / num_total if num_total > 0 else 0.0
        
        report_lines.append(f"\n📊 Overall:")
        report_lines.append(f"   Accuracy: {accuracy:.4f} ({num_correct}/{num_total})")
        

        ntp_values = pred_probs[torch.arange(len(true_actions)), true_actions]
        mean_ntp = ntp_values.mean().item()
        
        report_lines.append(f"   Mean NTP (confidence on true action): {mean_ntp:.4f}")
        

        if batch_indices is not None:
            report_lines.append(f"\n✅ Correct Predictions (sample 5):")
            correct_indices = torch.where(correct_mask)[0][:5]
            for idx in correct_indices:
                report_lines.append(f"   Sample {batch_indices[idx]}: pred={pred_actions[idx]}, true={true_actions[idx]}, conf={ntp_values[idx]:.4f}")
            
            report_lines.append(f"\n❌ Wrong Predictions (sample 5):")
            wrong_indices = torch.where(~correct_mask)[0][:5]
            for idx in wrong_indices:
                report_lines.append(f"   Sample {batch_indices[idx]}: pred={pred_actions[idx]}, true={true_actions[idx]}, conf={ntp_values[idx]:.4f}")
        
        report_lines.append("=" * 70)
        
        return "\n".join(report_lines)


def calc_accuracy(pred_probs, true_actions):
    return CEDAMetrics.calc_accuracy(pred_probs, true_actions)

def calc_ntp(pred_probs, true_actions):
    return CEDAMetrics.calc_ntp(pred_probs, true_actions)

def calc_ndcg_at_k(attribution_dict, gt_keys, k):
    return CEDAMetrics.calc_ndcg_at_k(attribution_dict, gt_keys, k)

def calc_precision_at_k(attribution_dict, gt_keys, k):
    return CEDAMetrics.calc_precision_at_k(attribution_dict, gt_keys, k)

def calc_auc_roc(attribution_dict, gt_keys):
    return CEDAMetrics.calc_auc_roc(attribution_dict, gt_keys)

def calc_map(attribution_dict, gt_keys):
    return CEDAMetrics.calc_map(attribution_dict, gt_keys)

def calc_auprc(attribution_dict, gt_keys):
    return CEDAMetrics.calc_auprc(attribution_dict, gt_keys)

def calc_sparsity(attribution_values, threshold=1e-4):
    return CEDAMetrics.calc_sparsity(attribution_values, threshold)


