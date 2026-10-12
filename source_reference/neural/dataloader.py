

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
import json
from pathlib import Path
import os
from typing import List, Dict, Tuple, Optional
import warnings
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence
import logging


logger = logging.getLogger(__name__)
logger.propagate = False
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ExpertTrajectoryDataset(Dataset):
    SYSTEM_SIMPLE_ACTIONS = ("system_simple_restart", "system_simple_replace")

    """
    专家轨迹数据集
    
    特点：
    1.  **Strict Node Contract**: 强制校验 trajectory metadata 中的 column_order 与 CKG 一致。
    2.  **Compact Format Support**: 支持 state['s'] (列表) 和 state['i'] (索引)。
    3.  **SD Signal**: 提取 step['performance']['sd'] 作为偏差信号。
    4.  **Normalization**: 使用 sensor_data.csv 拟合 scaler，并未 's' 列表进行转换。
    """

    def __init__(
        self,
        trajectories_path: str,
        ckg_path: str,
        max_seq_len: Optional[int] = None,
        scaler: Optional[StandardScaler] = None,
        action_vocab: Optional[Dict[str, int]] = None,
        data_scale_factor: float = 1.0
    ):
        self.trajectories_path = trajectories_path
        self.ckg_path = ckg_path
        self.max_seq_len = max_seq_len
        self.normalize = True
        self._external_scaler = scaler
        self._external_action_vocab = action_vocab
        self.data_scale_factor = data_scale_factor


        self._initialize_node_order()


        cache_path = Path(self.trajectories_path).with_suffix('.pkl')
        if cache_path.exists() and False:

             pass


        self._load_and_validate_trajectories()


        self._initialize_scaler()


        self._build_action_vocab()


        if False and cache_path.exists():

            source_mtime = os.path.getmtime(self.trajectories_path)
            cache_mtime = os.path.getmtime(cache_path)
            
            if source_mtime > cache_mtime:
                print(f"[INFO] Source data is newer than cache. Reloading...")
                should_load = False
            else:
                should_load = True
                
            if should_load:
                try:
                    import pickle
                    print(f"[INFO] Loading cached dataset from {cache_path}")
                    with open(cache_path, 'rb') as f:
                        self.processed_data = pickle.load(f)
                        print(f"[SUCCESS] Loaded {len(self.processed_data)} trajectories from cache.")
                except Exception as e:
                    print(f"[WARN] Cache load failed ({e}), reprocessing...")
                    self.processed_data = self._process_all_trajectories()
                    self._save_cache(cache_path)
            else:
                self.processed_data = self._process_all_trajectories()
                self._save_cache(cache_path)
        else:
            self.processed_data = self._process_all_trajectories()


    def _save_cache(self, path):
        try:
            import pickle
            print(f"[INFO] Saving dataset cache to {path}")
            with open(path, 'wb') as f:
                pickle.dump(self.processed_data, f)
        except Exception as e:
            print(f"[WARN] Failed to save cache: {e}")
        

        self._verify_sd_density()

        logger.info(f"[SUCCESS] Dataset built: {len(self.processed_data)} trajectories")

    def _initialize_node_order(self):
        
        if not os.path.exists(self.ckg_path):
            raise FileNotFoundError(f"CKG文件不存在: {self.ckg_path}")
        
        with open(self.ckg_path, 'r', encoding='utf-8') as f:
            self.ckg = json.load(f)
        

        self.column_mapping = self.ckg.get('column_to_node_mapping', {})
        if not self.column_mapping and 'nodes' in self.ckg:
            logger.info(" CKG missing mapping field. Synthesizing identity mapping from 'nodes' list...")
            for node_item in self.ckg['nodes']:
                if isinstance(node_item, str):
                    node_id = node_item
                    self.column_mapping[node_id] = {"id": node_id, "type": "Unknown"}
                elif isinstance(node_item, dict):
                    node_id = node_item.get('id', node_item.get('node_id'))
                    if node_id:
                        self.column_mapping[node_id] = {"id": node_id, "type": node_item.get('type', 'Unknown')}


        if 'node_order' in self.ckg:
             self.node_column_order = self.ckg['node_order']
        else:

             self.node_column_order = sorted(list(self.column_mapping.keys()))
             
        self.num_nodes = len(self.node_column_order)
        self.node_feature_dim = 1
        

        self.node_id_to_column = {v['id']: k for k, v in self.column_mapping.items()}
        self.column_to_int_index = {col_name: i for i, col_name in enumerate(self.node_column_order)}
        
        self.node_id_to_int_index = {
            v['id']: self.column_to_int_index[k]
            for k, v in self.column_mapping.items()
            if k in self.column_to_int_index
        }
        

        self._build_edge_index()

    def _build_edge_index(self):
        
        print("[INFO] Analyzing CKG edge structure...")
        edges = self.ckg.get('edges') 
        source_indices = []
        target_indices = []
        total_edges_found = 0
        
        def process_edge_data(edge_data):
            if isinstance(edge_data, dict):
                source_id = edge_data.get('source')
                target_id = edge_data.get('target')
                if source_id in self.node_id_to_int_index and target_id in self.node_id_to_int_index:
                    source_indices.append(self.node_id_to_int_index[source_id])
                    target_indices.append(self.node_id_to_int_index[target_id])
                    return 1
            return 0
            
        if isinstance(edges, dict):
            for edge_type, edge_list in edges.items():
                if isinstance(edge_list, list):
                    for edge_data in edge_list:
                        total_edges_found += process_edge_data(edge_data)
                        
        self.edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)
        print(f"[SUCCESS] Successfully built edge_index: {self.edge_index.shape} ({total_edges_found} edges)")

    def _load_and_validate_trajectories(self):
        
        if not os.path.exists(self.trajectories_path):
             raise FileNotFoundError(f"轨迹文件不存在: {self.trajectories_path}")
             
        print(f"[INFO] Loading trajectories: {self.trajectories_path}")
        with open(self.trajectories_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.raw_trajectories = data.get('trajectories', [])
        metadata = data.get('metadata', {})
        

        traj_column_order = metadata.get('column_order')
        

        actual_input_dim = 0
        if self.raw_trajectories and len(self.raw_trajectories[0].get('trajectory_steps', [])) > 0:
            first_s = self.raw_trajectories[0]['trajectory_steps'][0].get('state', {}).get('s', [])
            actual_input_dim = len(first_s)

        if not traj_column_order:

             traj_column_order = metadata.get('node_id_order', [])
             if not traj_column_order:
                 raise ValueError("CRITICAL: Trajectory JSON metadata missing 'column_order'. Cannot verify node alignment.")
        

        if traj_column_order != self.node_column_order:
            raise ValueError('Node ID order mismatch between trajectory metadata and CKG')
            
        print(f"[SUCCESS] Node Index Contract Verified: Dataset dimension is {self.num_nodes}")

    def _initialize_scaler(self):
        
        if self._external_scaler is not None:
            self.scaler = self._external_scaler
            print(' Using external Scaler (val mode)')
            return
        
        base_data_path = Path(self.trajectories_path).parent
        sensor_data_path = base_data_path / "sensor_data.csv"
        
        if not os.path.exists(sensor_data_path):
             raise FileNotFoundError(f"sensor_data.csv not found: {sensor_data_path}")
             
        print(f"[INFO] Loading {sensor_data_path} for normalizer fitting...")
        sensor_df = pd.read_csv(sensor_data_path)
        
        missing_cols = [c for c in self.node_column_order if c not in sensor_df.columns]
        if missing_cols:
            raise ValueError(f"sensor_data.csv missing cols: {missing_cols}")
            
        self.scaler = StandardScaler().fit(sensor_df[self.node_column_order])
        

        zero_std_mask = self.scaler.scale_ == 0.0
        if zero_std_mask.any():
            self.scaler.scale_[zero_std_mask] = 1.0
            print(f"[WARN] Found zero-variance columns. Fixed normalizer to prevent NaN. Replaced {zero_std_mask.sum()} scales.")
            
        print(f"[SUCCESS] Normalizer fitted. Means: {self.scaler.mean_[:3]}, Stds: {self.scaler.scale_[:3]}")
    def _build_action_vocab(self):
        
        if self._external_action_vocab is not None:
            self.action_vocab = self._external_action_vocab
            self.action_space_size = len(self.action_vocab)
            print(f' Using external Action Vocab (val mode): {self.action_space_size} actions')
            return
        
        print("  Building root intent action vocab...")
        all_root_intent_strings = set()
        all_root_intent_strings.update(self.SYSTEM_SIMPLE_ACTIONS)
        for traj in self.raw_trajectories:
            for step in traj.get('trajectory_steps', []):
                for action_key in ('action', 'a_exp', 'a_sys', 'system_action'):
                    action_str = step.get(action_key, {}).get('intent_id')
                    if action_str:
                        all_root_intent_strings.add(self._root_intent(action_str))
                inferred_sys = self._infer_system_intent(step)
                if inferred_sys:
                    all_root_intent_strings.add(self._root_intent(inferred_sys))
        
        self.action_vocab = {action_str: i for i, action_str in enumerate(sorted(list(all_root_intent_strings)))}
        self.action_space_size = len(self.action_vocab)
        print(f"  Vocab built: {self.action_space_size} root actions")

    @staticmethod
    def _root_intent(action_str: str) -> str:
        return action_str.split("_step_")[0] if "_step_" in action_str else action_str

    def _infer_system_intent(self, step: Dict) -> str:
        """Deterministic pi_sys labels for old trajectories without explicit a_sys."""
        explicit = step.get('a_sys') or step.get('system_action')
        if isinstance(explicit, dict) and explicit.get('intent_id'):
            return explicit['intent_id']

        action_id = step.get('action', {}).get('intent_id', '')
        if action_id.startswith('edge_system_') or action_id in self.SYSTEM_SIMPLE_ACTIONS:
            return action_id

        trigger_info = step.get('trigger_info', {}) or {}
        if 'source_high' in trigger_info or 'target_high' in trigger_info:
            target_high = bool(trigger_info.get('target_high', False))
            return 'edge_system_reactive_replace' if target_high else 'edge_system_routine_restart'


        return 'system_simple_replace'

    def _process_all_trajectories(self) -> List[Dict]:
        
        processed_list = []
        self.sd_stats = {'1': 0, '0': 0, '-1': 0, 'total_steps': 0}
        
        for traj_idx, traj in enumerate(self.raw_trajectories):
            try:
                processed = self._process_single_trajectory(traj)
                if processed:

                    processed['trajectory_id'] = traj.get('trajectory_id', f'traj_{traj_idx}')

                    processed['context_info'] = self._extract_context_info(traj)
                    processed_list.append(processed)
            except Exception as e:
                print(f"[WARN] Failed to process trajectory {traj_idx}: {e}")
                continue
                
        return processed_list

    def _process_single_trajectory(self, trajectory: Dict) -> Optional[Dict]:
        
        steps = trajectory.get('trajectory_steps', [])
        if not steps:
            return None

        states_list = []
        actions_exp_list = []
        actions_sys_list = []
        residual_list = []
        sd_list = []
        
        for step in steps:

            state_data = step.get('state', {})
            

            if 's' in state_data:

                state_vec = np.array(state_data['s'], dtype=np.float32)

                if len(state_vec) != self.num_nodes:
                    raise ValueError(f"State vector length {len(state_vec)} != num_nodes {self.num_nodes}")
            elif 'sensor_readings' in state_data:

                readings = state_data['sensor_readings']
                state_vec = np.array([readings.get(col, 0.0) for col in self.node_column_order], dtype=np.float32)
            else:
                continue
                
            states_list.append(state_vec)
            

            action_str = (step.get('a_exp') or step.get('action') or {}).get('intent_id')
            sys_action_str = self._infer_system_intent(step)
            root_intent = self._root_intent(action_str)
            root_sys_intent = self._root_intent(sys_action_str)
            
            if root_intent in self.action_vocab and root_sys_intent in self.action_vocab:
                action_exp_id = self.action_vocab[root_intent]
                action_sys_id = self.action_vocab[root_sys_intent]
                actions_exp_list.append(action_exp_id)
                actions_sys_list.append(action_sys_id)
                explicit_residual = step.get('r_t', step.get('residual_label', None))
                residual_list.append(int(explicit_residual) if explicit_residual is not None else int(action_exp_id != action_sys_id))
            else:

                states_list.pop()
                continue
                

            perf = step.get('performance', {})
            sd_val = perf.get('sd', 0)
            sd_list.append(sd_val)
            

            self.sd_stats[str(sd_val)] += 1
            self.sd_stats['total_steps'] += 1

        if not actions_exp_list:
            return None


        states_np = np.array(states_list, dtype=np.float32)
        actions_exp_np = np.array(actions_exp_list, dtype=np.int64)
        actions_sys_np = np.array(actions_sys_list, dtype=np.int64)
        residual_np = np.array(residual_list, dtype=np.float32)
        sd_np = np.array(sd_list, dtype=np.float32)
        
        current_seq_len = len(states_np)


        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            states_normalized = self.scaler.transform(states_np)
        

        if self.data_scale_factor != 1.0:
            states_normalized = states_normalized / self.data_scale_factor
            

        if self.max_seq_len and current_seq_len > self.max_seq_len:
            states_normalized = states_normalized[:self.max_seq_len]
            actions_exp_np = actions_exp_np[:self.max_seq_len]
            actions_sys_np = actions_sys_np[:self.max_seq_len]
            residual_np = residual_np[:self.max_seq_len]
            sd_np = sd_np[:self.max_seq_len]
            current_seq_len = self.max_seq_len
        

        position_encoding = np.arange(current_seq_len, dtype=np.float32) / max(current_seq_len - 1, 1)

        position_encoding = np.tile(position_encoding[:, np.newaxis], (1, self.num_nodes))
        

        delta_features = np.zeros_like(states_normalized)
        if current_seq_len > 1:
            delta_features[1:] = np.diff(states_normalized, axis=0)
            delta_features[0] = delta_features[1] if current_seq_len > 1 else 0.0
        

        states_with_features = np.stack([
            states_normalized, 
            position_encoding, 
            delta_features
        ], axis=-1)
            
        return {
            'states': torch.tensor(states_with_features, dtype=torch.float32),
            'actions': torch.tensor(actions_exp_np, dtype=torch.long),
            'actions_exp': torch.tensor(actions_exp_np, dtype=torch.long),
            'actions_sys': torch.tensor(actions_sys_np, dtype=torch.long),
            'residual_mask': torch.tensor(residual_np, dtype=torch.float32),
            'sd_signals': torch.tensor(sd_np, dtype=torch.float32),
            'seq_len': current_seq_len
        }

    def _extract_context_info(self, trajectory: Dict) -> Dict:
        
        triggering_info = trajectory.get('triggering_blind_spot', {})

        return {
            'equipment_type': triggering_info.get('equipment_type', 'Unknown'),
            'condition_variable': triggering_info.get('condition_variable', 'Unknown'),
            'threshold_value': triggering_info.get('threshold_value', 0.0),
            'blind_spot_type': triggering_info.get('type', 'Unknown')
        }

    def _verify_sd_density(self):
        
        total = self.sd_stats['total_steps']
        if total == 0:
            logger.warning("[WARN] Dataset empty, cannot verify SD density")
            return
            
        ratio_1 = self.sd_stats['1'] / total
        logger.info(f"[INFO] SD Signal Statistics: Total Steps={total}, SD=1: {self.sd_stats['1']} ({ratio_1:.2%})")
        
        if ratio_1 < 0.05:
            logger.warning(f"[RISK] SD=1 ratio ({ratio_1:.2%}) < 5% threshold! CCI experiments may lack valid positive samples.")
        elif ratio_1 < 0.15:
            logger.info(f"[NOTE] SD=1 ratio ({ratio_1:.2%}) is low but within acceptable imbalance range.")
        else:
            logger.info("[SUCCESS] SD Signal distribution is healthy.")

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]
    
    def compute_sample_weights(self) -> List[float]:
        

        action_counts = {}
        total_actions = 0
        for item in self.processed_data:
            actions = item['actions'].numpy()
            for a in actions:
                action_counts[int(a)] = action_counts.get(int(a), 0) + 1
                total_actions += 1
        

        action_weights = {}
        for a_id, count in action_counts.items():
            freq = count / total_actions
            action_weights[a_id] = 1.0 / freq
        

        weights = []
        for item in self.processed_data:
            actions = item['actions'].numpy()
            if len(actions) > 0:

                traj_weight = np.mean([action_weights.get(int(a), 1.0) for a in actions])
            else:
                traj_weight = 1.0
            weights.append(traj_weight)
        
        logger.info(f' 采样权重计算完成: 轨迹数={len(weights)}, 权重范围=[{min(weights):.2f}, {max(weights):.2f}]')
        return weights

def collate_variable_length_trajectories(batch: List[Dict]) -> Dict:
    
    
    states_list = [item['states'] for item in batch]
    actions_list = [item['actions'] for item in batch]
    actions_exp_list = [item.get('actions_exp', item['actions']) for item in batch]
    actions_sys_list = [item.get('actions_sys', item['actions']) for item in batch]
    residual_list = [item.get('residual_mask', torch.zeros_like(item['actions'], dtype=torch.float32)) for item in batch]
    sd_list = [item['sd_signals'] for item in batch]
    seq_lens = torch.tensor([item['seq_len'] for item in batch], dtype=torch.long)
    
    max_len = seq_lens.max().item()
    

    states_batch = pad_sequence(states_list, batch_first=True, padding_value=0.0)
    

    states_batch = states_batch * 3.0
    

    actions_batch = pad_sequence(actions_list, batch_first=True, padding_value=0)
    actions_exp_batch = pad_sequence(actions_exp_list, batch_first=True, padding_value=0)
    actions_sys_batch = pad_sequence(actions_sys_list, batch_first=True, padding_value=0)
    residual_batch = pad_sequence(residual_list, batch_first=True, padding_value=0.0)
    

    sd_batch = pad_sequence(sd_list, batch_first=True, padding_value=0.0)
    

    batch_size = len(batch)
    masks_batch = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for i, length in enumerate(seq_lens):
        masks_batch[i, :length] = True
        
    return {
        'states': states_batch,
        'actions': actions_batch,
        'actions_exp': actions_exp_batch,
        'actions_sys': actions_sys_batch,
        'residual_mask': residual_batch,
        'sd_signals': sd_batch,
        'seq_len': seq_lens,
        'masks': masks_batch,
        'trajectory_ids': [item['trajectory_id'] for item in batch],
        'contexts': [item['context_info'] for item in batch]
    }

def create_dataloader(
    trajectories_path: str,
    ckg_path: str,
    batch_size: int = 16,
    shuffle: bool = True,
    max_seq_len: Optional[int] = None,
    use_balanced_sampling: bool = False
) -> Tuple[DataLoader, ExpertTrajectoryDataset]:
    """
    Helper to create DataLoader
    
    Args:
        use_balanced_sampling: 启用 WeightedRandomSampler 进行平衡采样
    """
    from torch.utils.data import WeightedRandomSampler
    
    dataset = ExpertTrajectoryDataset(
        trajectories_path=trajectories_path,
        ckg_path=ckg_path,
        max_seq_len=max_seq_len
    )
    
    sampler = None
    if use_balanced_sampling:
        weights = dataset.compute_sample_weights()
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
        logger.info(f' 启用 WeightedRandomSampler 平衡采样')
        shuffle = False
    

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=collate_variable_length_trajectories,
        num_workers=0,
        pin_memory=True
    )
    
    return dataloader, dataset


def create_train_val_dataloaders(
    trajectories_path: str,
    ckg_path: str,
    batch_size: int = 16,
    val_split: float = 0.2,
    max_seq_len: Optional[int] = None,
    use_balanced_sampling: bool = False,
    seed: int = 42,
    num_workers: int = 0,
    subset_ratio: float = 1.0,
    data_scale_factor: float = 1.0,
    collate_fn=None
) -> Tuple[DataLoader, DataLoader, ExpertTrajectoryDataset]:
    """
    创建带有训练/验证划分的 DataLoader
    
    Args:
        val_split: 验证集比例 (默认 0.2)
        seed: 随机种子（保证划分可复现）
        num_workers: DataLoader workers (默认 0 以兼容 Windows)
        subset_ratio: 数据子集比例 (0.0 ~ 1.0)
        data_scale_factor: Feature downscaling for public datasets
        collate_fn: Optional custom collate function. If None, uses default with 3.0x scaling.
        
    Returns:
        train_loader, val_loader, dataset
    """
    from torch.utils.data import WeightedRandomSampler, Subset
    import numpy as np
    

    dataset = ExpertTrajectoryDataset(
        trajectories_path=trajectories_path,
        ckg_path=ckg_path,
        max_seq_len=max_seq_len,
        data_scale_factor=data_scale_factor
    )
    

    n_samples = len(dataset)
    indices = list(range(n_samples))
    np.random.seed(seed)
    np.random.shuffle(indices)


    if subset_ratio < 1.0:
        num_subset = int(n_samples * subset_ratio)
        indices = indices[:num_subset]
        logger.info(f' Using {subset_ratio * 100}% of data ({num_subset}/{n_samples} samples)')
        n_samples = len(indices)
    
    split_idx = int(n_samples * (1 - val_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    logger.info(f' Train/Val Split: {len(train_indices)} / {len(val_indices)} (seed={seed})')
    

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    

    train_sampler = None
    train_shuffle = True
    if use_balanced_sampling:

        all_weights = dataset.compute_sample_weights()
        subset_weights = [all_weights[i] for i in train_indices]
        train_sampler = WeightedRandomSampler(
            weights=subset_weights,
            num_samples=len(subset_weights),
            replacement=True
        )
        train_shuffle = False
        logger.info(f' 启用 WeightedRandomSampler 平衡采样 (训练集)')
    

    actual_collate_fn = collate_fn if collate_fn is not None else collate_variable_length_trajectories
    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=train_shuffle if train_sampler is None else False,
        sampler=train_sampler,
        collate_fn=actual_collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=actual_collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, dataset

def get_state_action_dims_from_dataset(dataset):
    
    return dataset.node_feature_dim, dataset.action_space_size, dataset.num_nodes