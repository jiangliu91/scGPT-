import argparse
import json
import os
from pathlib import Path
import sys
import time
import warnings
import pandas as pd
import pickle
import torch
import scanpy as sc
import seaborn as sns
import numpy as np
import wandb
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# 导入 scgpt 核心库
sys.path.insert(0, "../")
import scgpt as scg
from scgpt_local.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import (
    masked_mse_loss,
    criterion_neg_log_bernoulli,
)
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, add_file_handler
from typing import Tuple, Dict, Optional, List, Any
import yaml
from types import SimpleNamespace  # 用于将字典转换为对象访问

# 设置 scanpy 图形参数，并禁用特定警告
sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')


# --- 参数解析函数：统一管理所有配置参数 ---
def parse_arguments():
    """
    解析命令行参数和 YAML 配置文件，并统一处理所有配置项的默认值和推断值。
    这是所有模块配置参数的唯一来源。
    """
    parser = argparse.ArgumentParser(description="Fine-tuning scGPT for Cell-type Annotation")
    parser.add_argument("--config_file", type=str, default="config.yaml", help="Path to the YAML configuration file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--dataset_name", type=str, default="ms", help="Dataset name (e.g., 'ms').")  # 添加默认值
    parser.add_argument("--do_train", action='store_true', help="Whether to perform training.")
    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to pre-trained scGPT model directory or .pt file.")  # 默认None
    parser.add_argument("--mask_ratio", type=float, default=0.0,
                        help="Masking ratio for MLM (0.0 for cell-type classification).")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--n_bins", type=int, default=5000, help="Number of bins for binned input.")
    parser.add_argument("--max_seq_len", type=int, default=1000, help="max_seq_len.")
    parser.add_argument("--MVC", action="store_true", help="Enable Masked Value Prediction for Cell Embedding (MVC).")
    parser.add_argument("--ecs_thres", type=float, default=0.0,
                        help="Elastic Cell Similarity objective threshold (0.0 to disable).")
    parser.add_argument("--dab_weight", type=float, default=0.0,
                        help="Weight for Domain Adaptation by Reverse Backpropagation (DAB).")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--layer_size", type=int, default=512, help="Embedding and hidden dimension of the model.")
    parser.add_argument("--nlayers", type=int, default=12,
                        help="Number of TransformerEncoderLayer in TransformerEncoder.")
    parser.add_argument("--nhead", type=int, default=8, help="Number of heads in nn.MultiheadAttention.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")
    parser.add_argument("--schedule_ratio", type=float, default=0.9, help="Ratio of epochs for learning rate schedule.")
    parser.add_argument("--save_eval_interval", type=int, default=1, help="Epochs interval to save model and evaluate.")
    parser.add_argument("--fast_transformer", action="store_true", help="Use fast transformer implementation.")
    parser.add_argument("--pre_norm", action="store_true", help="Apply pre-normalization in Transformer.")
    parser.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision (AMP).")
    parser.add_argument("--include_zero_gene", action='store_true',
                        help="If True, include zero genes among hvgs in training.")
    parser.add_argument("--freeze", action="store_true", help="Freeze pre-decoder weights during fine-tuning.")
    parser.add_argument("--DSBN", action="store_true", help="Enable Domain-specific Batch Normalization (DSBN).")
    parser.add_argument("--total_embs", type=str, default="both",
                        help="Source for total_embs: 'src', 'values', or 'both'.")
    parser.add_argument("--MLM", action="store_true", help="Enable Masked Language Modeling objective.")
    parser.add_argument("--CLS", action='store_true', help="Enable Cell Type Classification objective.")
    parser.add_argument("--ADV", action="store_true", help="Enable Adversarial training for batch correction.")
    parser.add_argument("--CCE", action="store_true", help="Enable Contrastive Cell Embedding objective.")
    parser.add_argument("--ECS_enabled", action='store_true', help="Enable Elastic Cell Similarity objective (ECS).")
    parser.add_argument("--DAB_enabled", action='store_true',
                        help="Enable Domain Adaptation by Reverse Backpropagation (DAB).")
    parser.add_argument("--INPUT_BATCH_LABELS", action="store_true", help="Use batch labels as input features.")

    cmd_args = parser.parse_args()
    final_config = vars(cmd_args)  # 从命令行参数开始构建配置字典

    # 加载 YAML 配置文件并与命令行参数合并 (命令行参数优先)
    if cmd_args.config_file and Path(cmd_args.config_file).exists():
        with open(cmd_args.config_file, 'r') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                for key, yaml_value in yaml_config.items():
                    try:
                        argparse_default = parser.get_default(key)
                        if final_config.get(key) == argparse_default:
                            final_config[key] = yaml_value
                    except AttributeError:
                        final_config[key] = yaml_value  # 如果 argparse 没有定义此参数，则直接添加 YAML 中的值
            print(f"Loaded config from '{cmd_args.config_file}'. Values may be overridden by command-line arguments.")
    else:
        print(
            f"Warning: Config file '{cmd_args.config_file}' not found. Using only command-line defaults or specified values.")

    # 将最终配置转换为 SimpleNamespace 对象，以便通过属性访问
    config_obj = SimpleNamespace(**final_config)

    # --- 统一设置所有模型和数据模块依赖的常量/参数 ---
    # 这些值在原始代码中是硬编码的，现在统一由 parse_arguments() 来确定
    if config_obj.n_bins is not None:
        config_obj.input_style = "binned"
        config_obj.output_style = "binned"
        config_obj.input_emb_style = "category"  # binned input typically uses category embedding
        # 兼容性检查
        if config_obj.input_emb_style == "scaling":
            raise ValueError("input_emb_style `scaling` is not supported for binned input.")
        config_obj.mask_value = config_obj.n_bins + 1  # Mask value for binned data
        config_obj.pad_value = config_obj.n_bins  # Pad value for binned data
        config_obj.n_input_bins = config_obj.n_bins + 2  # Total bins for input (including special tokens for category)
    else:
        config_obj.input_style = "log1p"  # Default to log1p if not binned
        config_obj.output_style = "log1p"
        config_obj.input_emb_style = "continuous"  # log1p/normed_raw typically uses continuous embedding
        # 兼容性检查
        if config_obj.input_emb_style == "category":
            raise ValueError("input_emb_style `category` is not supported for log1p or normed_raw input.")
        config_obj.mask_value = -1  # Mask value for continuous data
        config_obj.pad_value = -2  # Pad value for continuous data
        config_obj.n_input_bins = config_obj.n_bins  # n_bins is None here, but still kept for consistency in definition

    # 硬编码的特殊 token (模型和数据模块都会用到)
    config_obj.pad_token = "<pad>"
    config_obj.special_tokens = [config_obj.pad_token, "<cls>", "<eoc>"]

    # 模型内部的硬编码风格参数
    config_obj.cell_emb_style = "cls"
    config_obj.mvc_decoder_style = "inner product"

    # 对抗训练延迟 epochs (原始代码中为 0)
    config_obj.adv_E_delay_epochs = 0
    config_obj.adv_D_delay_epochs = 0

    # 对抗训练学习率 (原始代码中为 1e-3)
    config_obj.lr_ADV = 1e-3

    # Fast Transformer 后端 (原始代码中为 "flash")
    config_obj.fast_transformer_backend = "flash"

    # 学习率调度器间隔 (原始代码中为 1)
    config_obj.schedule_interval = 1

    # 评估批次大小 (原始代码中通常与 batch_size 相同)
    config_obj.eval_batch_size = config_obj.batch_size

    # DAB 是否使用独立优化器 (根据 DAB_enabled 和 dab_weight 推断，保持与原始逻辑一致)
    config_obj.DAB_separate_optim = True if config_obj.DAB_enabled and config_obj.dab_weight > 1 else False
    config_obj.per_seq_batch_sample = False
    # 根据 ECS_enabled 和 DAB_enabled 调整阈值 (与原代码相同)
    if hasattr(config_obj, 'ECS_enabled') and not config_obj.ECS_enabled:
        config_obj.ecs_thres = 0.0
    if hasattr(config_obj, 'DAB_enabled') and not config_obj.DAB_enabled:
        config_obj.dab_weight = 0.0

    return config_obj


# --- SeqDataset 类：用于处理模型输入的数据集 ---
class SeqDataset(Dataset):
    """
    一个简单的 PyTorch Dataset，用于从字典中获取数据。
    与原代码相同，兼容 LightningDataModule。
    """

    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


# --- ScGPTDataModule 类：数据加载、预处理和 DataLoader 管理 ---
class ScGPTDataModule(pl.LightningDataModule):
    """
    ScGPT 数据模块，负责数据的加载、预处理、划分和 DataLoader 的创建。
    所有配置参数直接从 config 对象获取，确保一致性。
    """

    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.config = config
        # 保存超参数，这会在 checkpoint 中保存 config 的所有属性
        self.save_hyperparameters(vars(config))

        # --- 直接从 config 对象获取所有已确定的数据相关参数，避免重复推断 ---
        self.dataset_name = config.dataset_name
        self.load_model = config.load_model
        self.n_bins = config.n_bins
        self.max_seq_len = config.max_seq_len
        self.include_zero_gene = config.include_zero_gene

        self.input_emb_style = config.input_emb_style
        self.input_style = config.input_style
        self.output_style = config.output_style

        self.pad_token = config.pad_token
        self.special_tokens = config.special_tokens
        self.mask_value = config.mask_value
        self.pad_value = config.pad_value
        self.n_input_bins = config.n_input_bins  # 确保从config获取，避免不一致

        # 初始化内部状态变量，将在 prepare_data 和 setup 阶段填充
        self.adata: Optional[sc.AnnData] = None
        self.vocab: Optional[GeneVocab] = None
        self.id2type: Optional[Dict[int, str]] = None
        self.num_types: Optional[int] = None
        self.num_batch_types: Optional[int] = None
        self.input_layer_key: Optional[str] = None
        self.gene_ids: Optional[np.ndarray] = None

        self.preprocessor: Optional[Preprocessor] = None
        self.test_adata_split: Optional[sc.AnnData] = None
        self.test_outputs_buffer = []
        self.train_data_pt: Optional[Dict[str, torch.Tensor]] = None
        self.valid_data_pt: Optional[Dict[str, torch.Tensor]] = None
        self.test_data_pt: Optional[Dict[str, torch.Tensor]] = None

    def prepare_data(self):
        """
        在所有 GPU 上运行一次，用于下载和处理数据。
        加载原始 AnnData 文件，进行初步合并和标签分配。
        """
        if self.dataset_name == "ms":
            data_dir = Path("../data/ms")  # 硬编码数据路径
            adata_train_raw = sc.read(data_dir / "c_data.h5ad")
            adata_test_raw = sc.read(data_dir / "filtered_ms_adata.h5ad")

            # 统一 celltype 命名并转换为 categorical
            adata_train_raw.obs["celltype"] = adata_train_raw.obs[
                "Factor Value[inferred cell type - authors labels]"].astype("category")
            adata_test_raw.obs["celltype"] = adata_test_raw.obs[
                "Factor Value[inferred cell type - authors labels]"].astype("category")

            # 设置批次ID (硬编码为 "0" 和 "1")
            adata_train_raw.obs["batch_id"] = adata_train_raw.obs["str_batch"] = "0"
            adata_test_raw.obs["batch_id"] = adata_test_raw.obs["str_batch"] = "1"

            # 设置 gene_name 为 var.index，确保基因名称一致性
            adata_train_raw.var.set_index(adata_train_raw.var["gene_name"], inplace=True)
            adata_test_raw.var.set_index(adata_test_raw.var["gene_name"], inplace=True)

            data_is_raw = False  # 硬编码
            filter_gene_by_counts = False  # 硬编码

            # 合并数据集
            self.adata = adata_train_raw.concatenate(adata_test_raw, batch_key="str_batch")

            # 构建 batch_id 和 celltype_id 标签
            batch_id_labels = self.adata.obs["str_batch"].astype("category").cat.codes.values
            self.adata.obs["batch_id"] = batch_id_labels
            celltype_id_labels = self.adata.obs["celltype"].astype("category").cat.codes.values
            self.adata.obs["celltype_id"] = celltype_id_labels

            # 获取 celltype 映射和数量
            self.id2type = dict(enumerate(self.adata.obs["celltype"].astype("category").cat.categories))
            self.num_types = len(np.unique(celltype_id_labels))
            self.num_batch_types = len(set(self.adata.obs["batch_id"].tolist()))

            # 基因名称列表
            self.adata.var["gene_name"] = self.adata.var.index.tolist()

            # 初始化预处理器，其参数与原代码一致
            self.preprocessor = Preprocessor(
                use_key="X", filter_gene_by_counts=filter_gene_by_counts, filter_cell_by_counts=False,
                normalize_total=1e4, result_normed_key="X_normed", log1p=data_is_raw,
                result_log1p_key="X_log1p", subset_hvg=False,
                hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
                binning=self.n_bins, result_binned_key="X_binned",
            )

            # 设置输入层键，基于 config 中的 input_style
            self.input_layer_key = {
                "normed_raw": "X_normed", "log1p": "X_normed", "binned": "X_binned",
            }[self.input_style]

            # 词汇表加载或创建 (更健壮的逻辑)
            # 尝试从指定模型路径的vocab.json加载，否则从数据构建新vocab
            if self.load_model and Path(self.load_model).is_dir():
                vocab_file = Path(self.load_model) / "vocab.json"
                if vocab_file.exists():
                    self.vocab = GeneVocab.from_file(vocab_file)
                    # 确保特殊 token 在词汇表中
                    for s in self.special_tokens:
                        if s not in self.vocab:
                            self.vocab.append_token(s)
                    scg.logger.info(f"Loaded vocabulary from {vocab_file}.")
                else:
                    scg.logger.warning(f"Vocab file not found at {vocab_file}. Building new vocabulary from data.")
                    self.vocab = GeneVocab(self.adata.var["gene_name"].tolist(), self.special_tokens)
            else:
                scg.logger.info('No valid model directory or vocab file specified. Building new vocabulary from data.')
                self.vocab = GeneVocab(self.adata.var["gene_name"].tolist(), self.special_tokens)

            # 根据词汇表过滤基因
            self.adata.var["id_in_vocab"] = [1 if gene in self.vocab else -1 for gene in self.adata.var["gene_name"]]
            self.adata = self.adata[:, self.adata.var["id_in_vocab"] >= 0]
            scg.logger.info(
                f"Matched {np.sum(self.adata.var['id_in_vocab'] >= 0)}/{len(self.adata.var['id_in_vocab'])} genes in vocabulary of size {len(self.vocab)}.")

    def setup(self, stage: Optional[str] = None):
        """
        在每个 GPU 上运行一次，用于设置训练、验证和测试数据集。
        执行数据预处理、数据集划分和 tokenization 逻辑。
        """
        # 确保 prepare_data 已经运行
        if self.adata is None:
            self.prepare_data()

        # 划分训练集和测试集（基于原始的 batch_id）
        adata_train_split = self.adata[self.adata.obs["str_batch"] == "0"].copy()
        self.test_adata_split = self.adata[self.adata.obs["str_batch"] == "1"].copy()

        # 对训练和测试集进行预处理
        self.preprocessor(adata_train_split, batch_key=None)
        self.preprocessor(self.test_adata_split, batch_key=None)

        if stage == "fit" or stage is None:
            all_counts_train = (
                adata_train_split.layers[self.input_layer_key].toarray()
                if issparse(adata_train_split.layers[self.input_layer_key])
                else adata_train_split.layers[self.input_layer_key]
            )
            genes_in_adata = adata_train_split.var["gene_name"].tolist()
            self.gene_ids = np.array(self.vocab(genes_in_adata), dtype=int)

            celltypes_labels_train = adata_train_split.obs["celltype_id"].tolist()
            batch_ids_train = adata_train_split.obs["batch_id"].tolist()

            # 划分训练集和验证集
            (
                train_data_raw, valid_data_raw,
                train_celltype_labels, valid_celltype_labels,
                train_batch_labels, valid_batch_labels,
            ) = train_test_split(
                all_counts_train, celltypes_labels_train, batch_ids_train,
                test_size=0.1, shuffle=True, random_state=self.config.seed,
            )

            # Tokenize 和 Pad 训练数据
            tokenized_train = tokenize_and_pad_batch(
                train_data_raw, self.gene_ids, max_len=self.max_seq_len, vocab=self.vocab,
                pad_token=self.pad_token, pad_value=self.pad_value, append_cls=True,
                include_zero_gene=self.include_zero_gene,
            )
            # Tokenize 和 Pad 验证数据
            tokenized_valid = tokenize_and_pad_batch(
                valid_data_raw, self.gene_ids, max_len=self.max_seq_len, vocab=self.vocab,
                pad_token=self.pad_token, pad_value=self.pad_value, append_cls=True,
                include_zero_gene=self.include_zero_gene,
            )

            # 转换为 PyTorch Tensor
            self.train_data_pt = {
                "gene_ids": tokenized_train["genes"], "values": tokenized_train["values"],
                "target_values": tokenized_train["values"],
                "batch_labels": torch.from_numpy(np.array(train_batch_labels)).long(),
                "celltype_labels": torch.from_numpy(np.array(train_celltype_labels)).long(),
            }
            self.valid_data_pt = {
                "gene_ids": tokenized_valid["genes"], "values": tokenized_valid["values"],
                "target_values": tokenized_valid["values"],
                "batch_labels": torch.from_numpy(np.array(valid_batch_labels)).long(),
                "celltype_labels": torch.from_numpy(np.array(valid_celltype_labels)).long(),
            }

            scg.logger.info(
                f"Train set samples: {self.train_data_pt['gene_ids'].shape[0]}, feature length: {self.train_data_pt['gene_ids'].shape[1]}")
            scg.logger.info(
                f"Valid set samples: {self.valid_data_pt['gene_ids'].shape[0]}, feature length: {self.valid_data_pt['gene_ids'].shape[1]}")

            # DEBUG: 检查训练集和验证集标签范围
            scg.logger.debug(
                f"DEBUG: Train labels min: {self.train_data_pt['celltype_labels'].min()}, max: {self.train_data_pt['celltype_labels'].max()}, num_types: {self.num_types}")
            scg.logger.debug(
                f"DEBUG: Valid labels min: {self.valid_data_pt['celltype_labels'].min()}, max: {self.valid_data_pt['celltype_labels'].max()}, num_types: {self.num_types}")
            assert self.train_data_pt['celltype_labels'].min() >= 0 and self.train_data_pt[
                'celltype_labels'].max() < self.num_types, "Train labels out of range!"
            assert self.valid_data_pt['celltype_labels'].min() >= 0 and self.valid_data_pt[
                'celltype_labels'].max() < self.num_types, "Valid labels out of range!"

            # DEBUG: 检查训练集和验证集数据值范围（用于嵌入层）
            scg.logger.debug(
                f"DEBUG: Train values min: {self.train_data_pt['values'].min()}, max: {self.train_data_pt['values'].max()}, n_input_bins/vocab_size: {self.n_input_bins if self.input_emb_style == 'category' else len(self.vocab)}")
            scg.logger.debug(
                f"DEBUG: Valid values min: {self.valid_data_pt['values'].min()}, max: {self.valid_data_pt['values'].max()}, n_input_bins/vocab_size: {self.n_input_bins if self.input_emb_style == 'category' else len(self.vocab)}")
            # 确保 values 在有效索引范围内
            max_allowed_index = self.n_input_bins - 1 if self.input_emb_style == 'category' else len(
                self.vocab) - 1  # gene_ids for embedding layer
            if self.config.per_seq_batch_sample:  # SubsetsBatchSampler会影响batch_labels，但gene_ids和values不受影响
                pass
            else:  # 如果使用DataLoader，可能会有默认的索引转换，这里是通用检查
                # 注意：masked_values 会包含 mask_value 和 pad_value，它们可能超出正常范围，但会在模型中特殊处理
                # 此处检查的是原始的tokenized_train/valid['values']，在掩码前
                if self.input_emb_style == 'category':
                    assert self.train_data_pt['values'].min() >= self.pad_value or self.train_data_pt[
                        'values'].min() >= 0, "Values must be valid or pad_value"
                    assert self.train_data_pt[
                               'values'].max() <= self.n_input_bins - 1, "Values max index out of range for category embedding."
                    assert self.valid_data_pt[
                               'values'].max() <= self.n_input_bins - 1, "Values max index out of range for category embedding."
                else:  # continuous
                    # 对于连续值，范围检查意义不大，更关注是否有NaN/inf
                    pass  # values here are raw float values, min/max depends on expression range.

        if stage == "test" or stage is None:
            all_counts_test = (
                self.test_adata_split.layers[self.input_layer_key].toarray()
                if issparse(self.test_adata_split.layers[self.input_layer_key])
                else self.test_adata_split.layers[self.input_layer_key]
            )
            celltypes_labels_test = self.test_adata_split.obs["celltype_id"].tolist()
            batch_ids_test = self.test_adata_split.obs["batch_id"].tolist()

            tokenized_test = tokenize_and_pad_batch(
                all_counts_test, self.gene_ids, max_len=self.max_seq_len, vocab=self.vocab,
                pad_token=self.pad_token, pad_value=self.pad_value, append_cls=True,
                include_zero_gene=self.include_zero_gene,
            )
            self.test_data_pt = {
                "gene_ids": tokenized_test["genes"], "values": tokenized_test["values"],
                "target_values": tokenized_test["values"],
                "batch_labels": torch.from_numpy(np.array(batch_ids_test)).long(),
                "celltype_labels": torch.from_numpy(np.array(celltypes_labels_test)).long(),
            }
            # DEBUG: 检查测试集标签范围
            scg.logger.debug(
                f"DEBUG: Test labels min: {self.test_data_pt['celltype_labels'].min()}, max: {self.test_data_pt['celltype_labels'].max()}, num_types: {self.num_types}")
            assert self.test_data_pt['celltype_labels'].min() >= 0 and self.test_data_pt[
                'celltype_labels'].max() < self.num_types, "Test labels out of range!"
            # DEBUG: 检查测试集数据值范围
            scg.logger.debug(
                f"DEBUG: Test values min: {self.test_data_pt['values'].min()}, max: {self.test_data_pt['values'].max()}, n_input_bins/vocab_size: {self.n_input_bins if self.input_emb_style == 'category' else len(self.vocab)}")
            if self.input_emb_style == 'category':
                assert self.test_data_pt['values'].min() >= self.pad_value or self.test_data_pt[
                    'values'].min() >= 0, "Test values must be valid or pad_value"
                assert self.test_data_pt[
                           'values'].max() <= self.n_input_bins - 1, "Test values max index out of range for category embedding."

    def train_dataloader(self) -> DataLoader:
        masked_values_train = random_mask_value(
            self.train_data_pt["values"], mask_ratio=self.config.mask_ratio,
            mask_value=self.mask_value, pad_value=self.pad_value,
        )
        self.train_data_pt["values"] = masked_values_train

        dataset = SeqDataset(self.train_data_pt)
        if self.config.per_seq_batch_sample:
            subsets = []
            batch_labels_array = self.train_data_pt["batch_labels"].numpy()
            for batch_label in np.unique(batch_labels_array):
                batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
                subsets.append(batch_indices)
            return DataLoader(
                dataset=dataset, batch_sampler=SubsetsBatchSampler(
                    subsets, self.config.batch_size, intra_subset_shuffle=True,
                    inter_subset_shuffle=True, drop_last=False,
                ), num_workers=min(len(os.sched_getaffinity(0)), self.config.batch_size // 2),
                pin_memory=True,
            )
        else:
            return DataLoader(
                dataset=dataset, batch_size=self.config.batch_size, shuffle=True,
                drop_last=False, num_workers=min(len(os.sched_getaffinity(0)), self.config.batch_size // 2),
                pin_memory=True,
            )

    def val_dataloader(self) -> DataLoader:
        masked_values_valid = random_mask_value(
            self.valid_data_pt["values"], mask_ratio=self.config.mask_ratio,
            mask_value=self.mask_value, pad_value=self.pad_value,
        )
        self.valid_data_pt["values"] = masked_values_valid

        dataset = SeqDataset(self.valid_data_pt)
        return DataLoader(
            dataset=dataset, batch_size=self.config.eval_batch_size, shuffle=False,
            drop_last=False, num_workers=min(len(os.sched_getaffinity(0)), self.config.eval_batch_size // 2),
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        input_values_test = random_mask_value(
            self.test_data_pt["values"], mask_ratio=self.config.mask_ratio,
            mask_value=self.mask_value, pad_value=self.pad_value,
        )
        self.test_data_pt["values"] = input_values_test

        dataset = SeqDataset(self.test_data_pt)
        return DataLoader(
            dataset=dataset, batch_size=self.config.eval_batch_size, shuffle=False,
            drop_last=False, num_workers=min(len(os.sched_getaffinity(0)), self.config.eval_batch_size // 2),
            pin_memory=True,
        )


# --- ScGPTModel 类：模型定义、训练/验证/测试逻辑 ---
class ScGPTModel(pl.LightningModule):
    def __init__(self, config: SimpleNamespace, vocab: GeneVocab, num_types: int, num_batch_types: int, id2type: Dict,
                 data_module_instance: 'ScGPTDataModule'):
        super().__init__()
        # 保存超参数，这会在 checkpoint 中保存 config 的所有属性
        self.save_hyperparameters(vars(config))
        self.test_outputs_buffer = []
        self.config = config
        self.vocab = vocab
        self.num_types = num_types
        self.num_batch_types = num_batch_types
        self.id2type = id2type
        self.data_module_instance = data_module_instance

        # 从 config 中提取模型初始化所需的参数
        embsize = config.layer_size
        nhead = config.nhead
        d_hid = config.layer_size
        nlayers = config.nlayers
        n_layers_cls = getattr(config, 'n_layers_cls', 3)  # 从config获取，如果不存在则为默认值

        # 加载预训练模型配置 (如果有)
        if config.load_model is not None:
            model_dir_for_config = Path(config.load_model)
            if model_dir_for_config.is_file():  # 如果是.pt文件，取其父目录
                model_dir_for_config = model_dir_for_config.parent
            model_config_file = model_dir_for_config / "args.json"

            if model_config_file.exists():
                with open(model_config_file, "r") as f:
                    model_configs_loaded = json.load(f)
                embsize = model_configs_loaded.get("embsize", embsize)
                nhead = model_configs_loaded.get("nheads", nhead)
                d_hid = model_configs_loaded.get("d_hid", d_hid)
                nlayers = model_configs_loaded.get("nlayers", nlayers)
                n_layers_cls = model_configs_loaded.get("n_layers_cls", n_layers_cls)
                scg.logger.info(f"Loaded model configs from {model_config_file}, overriding default settings.")

        # 根据 config.total_embs 设置 values 和 src 选择
        values_choose = 1 if config.total_embs in ['values', 'both'] else 0
        src_choose = 1 if config.total_embs in ['src', 'both'] else 0

        # --- 直接从 config 获取所有模型初始化所需的参数，确保一致性 ---
        # 避免重复推断或使用局部变量，保证与 parse_arguments 一致
        input_emb_style = config.input_emb_style
        mask_value = config.mask_value
        pad_value = config.pad_value
        n_input_bins = config.n_input_bins  # 确保从config获取

        self.model = TransformerModel(
            len(vocab), embsize, nhead, d_hid, nlayers, nlayers_cls=n_layers_cls,
            n_cls=self.num_types if config.CLS else 1, vocab=vocab, dropout=config.dropout,
            pad_token=config.pad_token,  # 确保传递字符串 "<pad>"
            pad_value=config.pad_value,  # 直接使用 config 中的 pad_value
            do_mvc=config.MVC, do_dab=config.DAB_enabled,
            use_batch_labels=config.INPUT_BATCH_LABELS, num_batch_labels=self.num_batch_types,
            domain_spec_batchnorm=config.DSBN, input_emb_style=config.input_emb_style,  # 直接使用 config 中的 input_emb_style
            n_input_bins=config.n_input_bins,  # 直接使用 config 中的 n_input_bins
            cell_emb_style=config.cell_emb_style,
            mvc_decoder_style=config.mvc_decoder_style, ecs_threshold=config.ecs_thres,
            explicit_zero_prob=config.MLM and config.include_zero_gene,
            use_fast_transformer=config.fast_transformer,
            fast_transformer_backend=config.fast_transformer_backend,
            pre_norm=config.pre_norm, values_choose=values_choose, src_choose=src_choose,
        )

        # 加载预训练模型权重 (这里处理的是 .pt 文件)
        if config.load_model is not None:
            model_load_path = Path(config.load_model)
            model_file_to_load = None

            if model_load_path.is_dir():
                model_file_to_load = model_load_path / "best_model.pt"
            elif model_load_path.is_file() and str(model_load_path).endswith(".pt"):
                model_file_to_load = model_load_path
            else:
                scg.logger.warning(
                    f"Invalid load_model path or type: {config.load_model}. Not loading pre-trained weights.")

            if model_file_to_load and model_file_to_load.exists():
                try:
                    # 将模型移到正确设备后加载权重
                    self.model.load_state_dict(torch.load(model_file_to_load, map_location=self.device))
                    scg.logger.info(f"Loading all model params from {model_file_to_load}")
                except Exception as e:
                    scg.logger.warning(
                        f"Could not load full model state dict from {model_file_to_load}: {e}. Attempting partial load.")
                    model_dict = self.model.state_dict()
                    pretrained_dict = torch.load(model_file_to_load, map_location=self.device)
                    pretrained_dict = {
                        k: v
                        for k, v in pretrained_dict.items()
                        if k in model_dict and v.shape == model_dict[k].shape
                    }
                    for k, v in pretrained_dict.items():
                        scg.logger.info(f"Loading params {k} with shape {v.shape}")
                    model_dict.update(pretrained_dict)
                    self.model.load_state_dict(model_dict)
            else:
                scg.logger.warning(
                    f"Model file {model_file_to_load} not found. Starting with randomly initialized weights.")

        # 冻结预解码器权重
        if config.freeze:
            pre_freeze_param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            for name, para in self.model.named_parameters():
                if "encoder" in name and "transformer_encoder" not in name:  # 确保不会冻结TransformerEncoder内部的某些部分
                    para.requires_grad = False
            post_freeze_param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            scg.logger.info(f"Total Pre freeze Params: {pre_freeze_param_count}")
            scg.logger.info(f"Total Post freeze Params: {post_freeze_param_count}")
            self.log("info/pre_freeze_param_count", pre_freeze_param_count, prog_bar=False, logger=True)
            self.log("info/post_freeze_param_count", post_freeze_param_count, prog_bar=False, logger=True)

        self.discriminator = None
        if config.ADV:
            self.discriminator = AdversarialDiscriminator(d_model=embsize, n_cls=self.num_batch_types)

        # 定义损失函数
        self.criterion_mlm = masked_mse_loss
        self.criterion_zero_prob = criterion_neg_log_bernoulli
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_dab = nn.CrossEntropyLoss()
        self.criterion_adv = nn.CrossEntropyLoss()

        # 直接从 config 获取掩码值和填充值
        self.mask_value = config.mask_value
        self.pad_value = config.pad_value

        self.automatic_optimization = not (config.ADV or config.DAB_enabled)

    def forward(self, input_gene_ids, input_values, src_key_padding_mask, batch_labels=None,
                CLS=False, CCE=False, MVC=False, ECS=False, do_sample=False):
        return self.model(
            input_gene_ids, input_values, src_key_padding_mask=src_key_padding_mask,
            batch_labels=batch_labels if self.config.INPUT_BATCH_LABELS or self.config.DSBN else None,
            CLS=CLS, CCE=CCE, MVC=MVC, ECS=ECS, do_sample=do_sample,
        )

    def training_step(self, batch, batch_idx):
        # --- 修正：更健壮地获取优化器和调度器 ---
        # self.optimizers() 返回单个优化器对象或优化器列表
        # 总是将其转换为列表，以便后续解包或索引
        all_optimizers = self.optimizers()
        if not isinstance(all_optimizers, list):
            all_optimizers = [all_optimizers]

        opt_model = all_optimizers[0]  # 主模型优化器总是第一个

        # 获取所有调度器，同样确保是列表形式
        all_schedulers = self.lr_schedulers()
        if not isinstance(all_schedulers, list):
            all_schedulers = [all_schedulers]

        sch_model = all_schedulers[0] if all_schedulers else None  # 主模型调度器

        # 如果存在判别器优化器，它是列表中的下一个
        opt_D = None
        if self.config.ADV and len(all_optimizers) > 1:
            opt_D = all_optimizers[1]

        # 如果存在 DAB 独立优化器，且 ADV 也存在，则 DAB 优化器可能是第三个
        # 注意：这里需要与 configure_optimizers 中返回的顺序严格对应
        opt_dab_idx = -1  # 初始化一个无效索引
        if self.config.DAB_separate_optim and self.config.dab_weight > 1:
            if self.config.ADV:  # 如果ADV也开着，DAB可能是第三个优化器
                opt_dab_idx = 2
            else:  # 如果ADV没开，DAB是第二个优化器
                opt_dab_idx = 1

        opt_DAB = all_optimizers[opt_dab_idx] if opt_dab_idx != -1 and len(all_optimizers) > opt_dab_idx else None

        input_gene_ids, input_values, target_values, batch_labels, celltype_labels = (
            batch["gene_ids"], batch["values"], batch["target_values"],
            batch["batch_labels"], batch["celltype_labels"]
        )
        src_key_padding_mask = input_gene_ids.eq(self.vocab[self.config.pad_token])
        masked_positions = input_values.eq(self.mask_value)

        if self.automatic_optimization:
            with torch.cuda.amp.autocast(enabled=self.config.amp):
                output_dict = self(
                    input_gene_ids, input_values, src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels, CLS=self.config.CLS, CCE=self.config.CCE, MVC=self.config.MVC,
                    ECS=self.config.ECS_enabled, do_sample=self.config.MLM and self.config.include_zero_gene,
                )
                loss = 0.0
                logs = {}

                if self.config.MLM:
                    loss_mse = self.criterion_mlm(output_dict["mlm_output"], target_values, masked_positions)
                    loss += loss_mse
                    logs["train/mse"] = loss_mse.item()
                if self.config.MLM and self.config.include_zero_gene:
                    loss_zero_log_prob = self.criterion_zero_prob(output_dict["mlm_zero_probs"], target_values,
                                                                  masked_positions)
                    loss += loss_zero_log_prob
                    logs["train/nzlp"] = loss_zero_log_prob.item()
                if self.config.CLS:
                    loss_cls = self.criterion_cls(output_dict["cls_output"], celltype_labels)
                    loss += loss_cls
                    logs["train/cls"] = loss_cls.item()
                    error_rate = 1 - (output_dict["cls_output"].argmax(1) == celltype_labels).float().mean()
                    logs["train/err"] = error_rate.item()
                if self.config.CCE:
                    loss_cce = 10 * output_dict["loss_cce"]
                    loss += loss_cce
                    logs["train/cce"] = loss_cce.item()
                if self.config.MVC:
                    loss_mvc = self.criterion_mlm(output_dict["mvc_output"], target_values, masked_positions)
                    loss += loss_mvc
                    logs["train/mvc"] = loss_mvc.item()
                if self.config.MVC and self.config.include_zero_gene:
                    loss_mvc_zero_log_prob = self.criterion_zero_prob(output_dict["mvc_zero_probs"], target_values,
                                                                      masked_positions)
                    loss += loss_mvc_zero_log_prob
                    logs["train/mvc_nzlp"] = loss_mvc_zero_log_prob.item()
                if self.config.ECS_enabled:
                    loss_ecs = 10 * output_dict["loss_ecs"]
                    loss += loss_ecs
                    logs["train/ecs"] = loss_ecs.item()
                if self.config.DAB_enabled:
                    loss_dab = self.criterion_dab(output_dict["dab_output"], batch_labels)
                    loss += self.config.dab_weight * loss_dab
                    logs["train/dab"] = loss_dab.item()

                self.log_dict(logs, prog_bar=True, logger=True)
                return loss
        else:  # 手动优化模式
            # 主模型优化器步骤
            opt_model.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.config.amp):
                output_dict = self(
                    input_gene_ids, input_values, src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels, CLS=self.config.CLS, CCE=self.config.CCE, MVC=self.config.MVC,
                    ECS=self.config.ECS_enabled, do_sample=self.config.MLM and self.config.include_zero_gene,
                )
                loss_main = 0.0
                logs = {}

                if self.config.MLM: loss_main += self.criterion_mlm(output_dict["mlm_output"], target_values,
                                                                    masked_positions); logs[
                    "train/mse"] = self.criterion_mlm(output_dict["mlm_output"], target_values, masked_positions).item()
                if self.config.MLM and self.config.include_zero_gene: loss_main += self.criterion_zero_prob(
                    output_dict["mlm_zero_probs"], target_values, masked_positions); logs[
                    "train/nzlp"] = self.criterion_zero_prob(output_dict["mlm_zero_probs"], target_values,
                                                             masked_positions).item()
                if self.config.CLS: loss_main += self.criterion_cls(output_dict["cls_output"], celltype_labels); logs[
                    "train/cls"] = self.criterion_cls(output_dict["cls_output"], celltype_labels).item(); logs[
                    "train/err"] = (1 - (output_dict["cls_output"].argmax(1) == celltype_labels).float().mean()).item()
                if self.config.CCE: loss_main += 10 * output_dict["loss_cce"]; logs["train/cce"] = (
                            10 * output_dict["loss_cce"]).item()
                if self.config.MVC: loss_main += self.criterion_mlm(output_dict["mvc_output"], target_values,
                                                                    masked_positions); logs[
                    "train/mvc"] = self.criterion_mlm(output_dict["mvc_output"], target_values, masked_positions).item()
                if self.config.MVC and self.config.include_zero_gene: loss_main += self.criterion_zero_prob(
                    output_dict["mvc_zero_probs"], target_values, masked_positions); logs[
                    "train/mvc_nzlp"] = self.criterion_zero_prob(output_dict["mvc_zero_probs"], target_values,
                                                                 masked_positions).item()
                if self.config.ECS_enabled: loss_main += 10 * output_dict["loss_ecs"]; logs["train/ecs"] = (
                            10 * output_dict["loss_ecs"]).item()
                if self.config.DAB_enabled: loss_dab = self.criterion_dab(output_dict["dab_output"],
                                                                          batch_labels); loss_main += self.config.dab_weight * loss_dab;
                logs["train/dab"] = loss_dab.item()

                if self.config.ADV and self.current_epoch > self.config.adv_E_delay_epochs:
                    loss_adv_E = -self.criterion_adv(self.discriminator(output_dict["cell_emb"]), batch_labels)
                    loss_main += loss_adv_E
                    logs["train/adv_E"] = loss_adv_E.item()

            self.manual_backward(loss_main)
            if self.config.amp: self.scaler.unscale_(opt_model)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(opt_model)
            self.scaler.update()

            # 判别器优化器步骤
            if self.config.ADV:
                if self.current_epoch > self.config.adv_D_delay_epochs:
                    opt_D.zero_grad()
                    with torch.cuda.amp.autocast(enabled=self.config.amp):
                        loss_adv_D = self.criterion_adv(self.discriminator(output_dict["cell_emb"].detach()),
                                                        batch_labels)
                    self.manual_backward(loss_adv_D)
                    if self.config.amp: self.scaler.unscale_(opt_D)
                    self.scaler.step(opt_D)
                    self.scaler.update()
                    logs["train/adv_D"] = loss_adv_D.item()
                else:
                    logs["train/adv_D"] = 0.0

            # DAB独立优化器步骤 (如果存在)
            if opt_DAB:  # 如果DAB优化器存在
                opt_DAB.zero_grad()
                with torch.cuda.amp.autocast(enabled=self.config.amp):
                    # 重新计算DAB损失 (可能需要重新运行forward获取dab_output)
                    # 或者如果output_dict['dab_output']是之前计算的，直接用
                    # 为了避免重新forward，通常直接使用之前的损失或output_dict
                    # 注意：如果DAB损失参与loss_main，这里需要确保它不被重复计算或梯度清零
                    # PyTorch Lightning的manual_optimization下，每个优化器负责自己的梯度
                    # 这里假设DAB_separate_optim意味着dab_weight * loss_dab没有被loss_main反向传播
                    # 但你之前的代码是：loss = loss + dab_weight * loss_dab
                    # 如果 dab_weight * loss_dab 已经在 loss_main 里了，那这个优化器就不该独立 step
                    # 除非 configure_optimizers 明确将 DAB 参数从主优化器中排除
                    # 鉴于原始逻辑，如果DAB_separate_optim为True，则DAB_weight可能就是0，或者DAB_separate_optim仅仅代表独立优化器
                    # 但在 ScGPTModel.__init__ 中，DAB_separate_optim 是基于 dab_weight > 1
                    # 这意味着 DAB_separate_optim 时，DAB 损失仍然在 loss_main 中
                    # 这是一个潜在的逻辑冲突：如果 loss_dab 已经被 loss_main 反向传播，就不应该再有独立优化器
                    # 通常，如果DAB是独立优化器，那么loss_main不应该包含DAB损失。
                    # 为了遵循原始代码，我将保留 DAB 损失在 loss_main 中，并移除这里的独立 DAB 优化器逻辑
                    # 因为 DAB_separate_optim 通常意味着 DAB 损失不计入主损失，而是单独优化其梯度。
                    # 检查一下 config.DAB_separate_optim 是如何与 configure_optimizers 配合的。
                    # 根据你的 configure_optimizers: optimizers.append(optimizer_dab)
                    # 但是 training_step 中的 loss_main 仍然包含 loss_dab = self.config.dab_weight * loss_dab
                    # 这表示 DAB 损失的梯度会被 opt_model 反向传播。
                    # 所以，如果 DAB_separate_optim 意味着独立优化器，那 DAB 损失就不该加到 loss_main 里。
                    # 这点需要澄清原始 scGPT 设计。
                    # 暂时保持现有 DAB 损失在 loss_main 的计算，这意味着 DAB 优化器不应独立存在于此处。
                    # 如果 DAB_separate_optim 表示 DAB 优化器只在训练器中注册，但其损失不贡献给主模型梯度，
                    # 那么 loss_main 就不应该加上 dab_weight * loss_dab。
                    # 鉴于此，这里的 `opt_DAB` 逻辑很可能与原代码意图不符或存在冗余。
                    # **我将移除此处的 `opt_DAB` 步进逻辑，除非你的 `configure_optimizers`
                    #   明确将 `DAB` 模型的参数从主优化器中排除，且 `loss_main` 不包含 `DAB` 损失。**
                    pass  # 移除独立的opt_DAB步进，避免重复梯度计算

            # Log metrics
            self.log_dict(logs, prog_bar=True, logger=True)

            # 手动步进学习率调度器
            # 需要根据 configure_optimizers 的返回顺序来步进
            # 假设 all_schedulers[0] 是主模型调度器
            if sch_model:
                sch_model.step()

            # 如果存在其他调度器 (ADV, DAB)，也在这里手动步进
            if self.config.ADV and len(all_schedulers) > 1:  # ADV 调度器通常是第二个
                all_schedulers[1].step()  # 假设 ADV 调度器是 all_schedulers[1]

            # 如果 DAB_separate_optim 为 True 且有对应的调度器
            if self.config.DAB_separate_optim and len(all_schedulers) > (2 if self.config.ADV else 1):
                # 如果 ADV 存在，DAB 调度器可能是第三个 (索引2)，否则是第二个 (索引1)
                dab_scheduler_idx = 2 if self.config.ADV else 1
                all_schedulers[dab_scheduler_idx].step()


    def validation_step(self, batch, batch_idx):
        input_gene_ids, input_values, target_values, batch_labels, celltype_labels = (
            batch["gene_ids"], batch["values"], batch["target_values"],
            batch["batch_labels"], batch["celltype_labels"]
        )
        src_key_padding_mask = input_gene_ids.eq(self.vocab[self.config.pad_token])

        with torch.cuda.amp.autocast(enabled=self.config.amp):
            output_dict = self(
                input_gene_ids, input_values, src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if self.config.INPUT_BATCH_LABELS or self.config.DSBN else None,
                CLS=self.config.CLS, CCE=False, MVC=False, ECS=False, do_sample=False,
            )
            loss_cls = self.criterion_cls(output_dict["cls_output"], celltype_labels)
            error_rate = 1 - (output_dict["cls_output"].argmax(1) == celltype_labels).float().mean()
            loss_dab = 0.0
            if self.config.DAB_enabled:
                loss_dab = self.criterion_dab(output_dict["dab_output"], batch_labels)

        self.log("valid/loss_cls", loss_cls, prog_bar=True, logger=True)
        self.log("valid/err", error_rate, prog_bar=True, logger=True)
        if self.config.DAB_enabled:
            self.log("valid/dab", loss_dab, prog_bar=True, logger=True)
            self.log("valid/sum_cls_dab", loss_cls + self.config.dab_weight * loss_dab, prog_bar=True, logger=True)
        return {"val_loss": loss_cls}

    def test_step(self, batch, batch_idx):
        """
        Defines a single testing step.
        CRITICAL MODIFICATION: Instead of returning, append results to a local buffer.
        """
        input_gene_ids, input_values, target_values, batch_labels, celltype_labels = (
            batch["gene_ids"], batch["values"], batch["target_values"],
            batch["batch_labels"], batch["celltype_labels"]
        )
        src_key_padding_mask = input_gene_ids.eq(self.vocab[self.config.pad_token])

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.config.amp):
                output_dict = self(
                    input_gene_ids, input_values, src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if self.config.INPUT_BATCH_LABELS or self.config.DSBN else None,
                    CLS=self.config.CLS, CCE=False, MVC=False, ECS=False, do_sample=False,
                )
                predictions = output_dict["cls_output"].argmax(1)  # Get predicted class IDs

        # --- KEY CHANGE: Append results to a local buffer instead of returning ---
        # Move tensors to CPU here, as they are collected locally per process.
        # This is safe because all_gather will then collect these CPU tensors.
        self.test_outputs_buffer.append({
            "predictions": predictions.cpu(),
            "labels": celltype_labels.cpu()
        })
        # Note: No return statement is needed here as we're manually collecting
        # PyTorch Lightning will call on_test_epoch_end even if test_step doesn't return
        # any value, as long as the dataloader yields batches.

    def on_test_epoch_end(self):  # 移除 outputs 参数，因为我们手动从 self.test_outputs_buffer 聚合
        """
        Executed at the end of the test epoch.
        Performs manual aggregation of test results from all processes, calculates metrics,
        and generates visualizations.
        """
        # 检查本地缓冲区是否有数据
        if not self.test_outputs_buffer:
            scg.logger.warning(
                "Local test_outputs_buffer is empty. Skipping aggregation and metrics calculation. "
                "This might happen if a DataLoader yields no batches locally on this process/GPU, "
                "or if test_step did not append any results."
            )
            return

        # 连接当前进程（GPU）收集到的所有批次数据
        try:
            local_predictions = torch.cat([x["predictions"] for x in self.test_outputs_buffer])
            local_labels = torch.cat([x["labels"] for x in self.test_outputs_buffer])
        except Exception as e:
            scg.logger.error(f"Failed to concatenate local test outputs: {e}. Skipping aggregation.")
            self.test_outputs_buffer.clear()  # 清除缓冲区以防止内存累积
            return

        # 使用 self.all_gather() 收集所有进程的预测和标签
        try:
            gathered_predictions_list = self.all_gather(local_predictions)
            gathered_labels_list = self.all_gather(local_labels)
        except Exception as e:
            scg.logger.error(f"Failed to perform all_gather for test outputs: {e}. Skipping metrics calculation.")
            self.test_outputs_buffer.clear()  # 清除缓冲区以防止内存累积
            return

        # 将聚合后的张量列表连接成单个 NumPy 数组
        if isinstance(gathered_predictions_list, list):
            all_predictions = torch.cat(gathered_predictions_list).cpu().numpy()
            all_labels = torch.cat(gathered_labels_list).cpu().numpy()
        else:  # 单个张量的情况，通常发生在非分布式（单GPU）运行中
            all_predictions = gathered_predictions_list.cpu().numpy()
            all_labels = gathered_labels_list.cpu().numpy()

        # 清除本地缓冲区，防止内存累积（为下一次测试或训练做准备）
        self.test_outputs_buffer.clear()

        # 健壮性检查：处理最终聚合后仍然为空的情况
        if all_predictions.size == 0 or all_labels.size == 0:
            scg.logger.warning(
                "on_test_epoch_end received empty aggregated outputs after all_gather. Skipping metrics calculation and plotting."
            )
            return

        # 计算分类评估指标
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average="macro")
        recall = recall_score(all_labels, all_predictions, average="macro")
        macro_f1 = f1_score(all_labels, all_predictions, average="macro")

        # 记录到标准日志
        scg.logger.info(
            f"Test Results: Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, "
            f"Macro F1: {macro_f1:.3f}"
        )

        # 记录到 WandB (通过 self.log_dict)
        results = {
            "test/accuracy": accuracy,
            "test/precision": precision,
            "test/recall": recall,
            "test/macro_f1": macro_f1,
        }
        self.log_dict(results, logger=True)

        # 保存原始预测结果和标签到 .pkl 文件
        save_dict = {
            "predictions": all_predictions,
            "labels": all_labels,
            "results": results,
            "id_maps": self.id2type
        }
        save_path = Path(self.trainer.log_dir) / "test_results.pkl"
        try:
            with open(save_path, "wb") as f:
                pickle.dump(save_dict, f)
        except Exception as e:
            scg.logger.error(f"Failed to save test_results.pkl: {e}")

        # --- UMAP 可视化 ---
        adata_test_raw = self.data_module_instance.test_adata_split.copy()
        adata_test_raw.obs["predictions"] = [self.id2type[p] for p in all_predictions]

        # 确保 AnnData 中有 UMAP 坐标
        if 'X_umap' not in adata_test_raw.obsm:
            # 检查用于 neighbors 的数据是否存在且有效
            use_rep_for_neighbors = None
            if 'X_pca' in adata_test_raw.obsm:
                use_rep_for_neighbors = 'X_pca'
            else:
                # 确保 adata.X 是稀疏矩阵或 NumPy 数组，如果需要的话可以转换为稠密
                if issparse(adata_test_raw.X):
                    adata_test_raw.X = adata_test_raw.X.toarray()

            # 检查 NaN/Inf
            if np.isnan(adata_test_raw.X).any() or np.isinf(adata_test_raw.X).any():
                scg.logger.error(
                    "adata_test_raw.X contains NaN or Inf values before neighbors calculation! Skipping UMAP plot.")
                return  # 提前退出以避免错误

            try:
                if use_rep_for_neighbors:
                    sc.pp.neighbors(adata_test_raw, use_rep=use_rep_for_neighbors)
                else:
                    sc.pp.neighbors(adata_test_raw)
                sc.tl.umap(adata_test_raw)
            except Exception as e:
                scg.logger.error(f"UMAP calculation FAILED: {e}. Skipping UMAP plot.")
                return  # 如果计算失败，则跳过绘图

        # 最终检查 UMAP 坐标
        if 'X_umap' not in adata_test_raw.obsm:
            scg.logger.error(
                "X_umap still not found in adata_test_raw.obsm after calculation attempt. Skipping UMAP plot.")
            return  # 提前退出
        if np.isnan(adata_test_raw.obsm['X_umap']).any() or np.isinf(adata_test_raw.obsm['X_umap']).any():
            scg.logger.error("Computed X_umap contains NaN or Inf values. Skipping UMAP plot.")
            return  # 提前退出

        palette_ = plt.rcParams["axes.prop_cycle"].by_key()["color"] * 3
        celltypes_unique = sorted(list(adata_test_raw.obs["celltype"].unique()))
        palette_dict = {c: palette_[i] for i, c in enumerate(celltypes_unique)}

        # 绘制 UMAP 图
        plt.figure(figsize=(12, 6), dpi=300)  # 为两个面板预留空间
        try:
            sc.pl.umap(
                adata_test_raw,
                color=["celltype", "predictions"],
                palette=palette_dict,
                show=False,
            )
            umap_path = Path(self.trainer.log_dir) / "results_umap.png"
            plt.gcf().savefig(umap_path, dpi=300, bbox_inches='tight')
            # 上传 UMAP 图到 WandB
            self.logger.experiment.log(
                {"test/cell_umap": wandb.Image(str(umap_path), caption=f"Predictions macro F1: {macro_f1:.3f}")}
            )
        except Exception as e:
            scg.logger.error(f"Failed to plot or save UMAP: {e}. Skipping UMAP plot.")
        finally:
            plt.close(plt.gcf())  # 确保关闭当前 Figure

        # --- Confusion Matrix ---
        celltypes_for_cm = sorted(list(self.id2type.values()))
        cm_labels_ids = sorted(list(self.id2type.keys()))

        try:
            cm = confusion_matrix(all_labels, all_predictions, labels=cm_labels_ids)
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # 归一化
            cm_df = pd.DataFrame(cm, index=celltypes_for_cm, columns=celltypes_for_cm)

            fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
            sns.heatmap(cm_df, annot=True, fmt=".1f", cmap="Blues", ax=ax)
            ax.set_title("Confusion Matrix (Normalized by True Label)")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            cm_path = Path(self.trainer.log_dir) / "confusion_matrix.png"
            fig.savefig(cm_path, dpi=300, bbox_inches='tight')
            # 上传混淆矩阵图到 WandB
            self.logger.experiment.log({"test/confusion_matrix": wandb.Image(str(cm_path), caption="Confusion Matrix")})
        except Exception as e:
            scg.logger.error(f"Failed to plot or save Confusion Matrix: {e}. Skipping.")
        finally:
            plt.close(fig)  # 确保关闭 Figure

    def configure_optimizers(self):
        """
        配置优化器和学习率调度器。
        原函数：main 函数中优化器和调度器的定义部分。
        改动：返回 PyTorch Lightning 期望的优化器和调度器列表。
        原因：Lightning 会自动调用此方法来获取优化器和调度器。
              对于手动优化，需要返回一个包含所有优化器的列表，以及一个包含所有调度器的字典。
        """
        # 主模型优化器和调度器
        optimizer_model = torch.optim.Adam(
            self.model.parameters(), lr=self.config.lr, eps=1e-4 if self.config.amp else 1e-8
        )
        scheduler_model = torch.optim.lr_scheduler.StepLR(
            optimizer_model, self.config.schedule_interval, gamma=self.config.schedule_ratio
        )
        optimizers = [optimizer_model]
        schedulers = [
            {
                'scheduler': scheduler_model,
                'interval': 'epoch',  # 每个 epoch 更新一次
                'frequency': 1,
            }
        ]

        # DAB 独立优化器 (如果启用且单独优化)
        if self.config.DAB_enabled and self.config.dab_weight > 1:  # 沿用原代码的DAB_separate_optim逻辑
            optimizer_dab = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
            scheduler_dab = torch.optim.lr_scheduler.StepLR(
                optimizer_dab, self.config.schedule_interval, gamma=self.config.schedule_ratio
            )
            optimizers.append(optimizer_dab)
            schedulers.append(
                {
                    'scheduler': scheduler_dab,
                    'interval': 'epoch',
                    'frequency': 1,
                }
            )

        # ADV 优化器 (如果启用)
        if self.config.ADV:
            optimizer_E = torch.optim.Adam(self.model.parameters(), lr=self.config.lr_ADV)
            scheduler_E = torch.optim.lr_scheduler.StepLR(
                optimizer_E, self.config.schedule_interval, gamma=self.config.schedule_ratio
            )
            optimizers.append(optimizer_E)
            schedulers.append(
                {
                    'scheduler': scheduler_E,
                    'interval': 'epoch',
                    'frequency': 1,
                }
            )

            optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.config.lr_ADV)
            scheduler_D = torch.optim.lr_scheduler.StepLR(
                optimizer_D, self.config.schedule_interval, gamma=self.config.schedule_ratio
            )
            optimizers.append(optimizer_D)
            schedulers.append(
                {
                    'scheduler': scheduler_D,
                    'interval': 'epoch',
                    'frequency': 1,
                }
            )

        # 如果启用了 AMP，初始化 GradScaler
        if self.config.amp:
            self.scaler = torch.cuda.amp.GradScaler()

        return optimizers, schedulers

    def on_train_epoch_start(self):
        """
        每个训练 epoch 开始时执行。
        原函数：prepare_data 中掩码日志的打印。
        改动：将与训练周期相关的动态掩码日志移到这里。
        原因：`on_train_epoch_start` 是 LightningModule 的一个钩子，适合执行每个 epoch 开始时的操作。
              在这里，我们可以打印掩码信息，就像原代码那样。
        """
        # 注意：实际的随机掩码操作在 train_dataloader 中完成
        # 这里仅用于日志输出
        scg.logger.info(
            f"random masking at epoch {self.current_epoch:3d}, mask ratio: {self.config.mask_ratio:.4f}"
        )


# --- 主运行函数 (Lightning Trainer 的入口) ---
def main():
    """
    程序的主入口点，负责解析参数、设置日志、初始化数据模块和模型模块，并启动训练。
    改动：重写 main 函数，使用 PyTorch Lightning 的 Trainer 协调整个训练流程。
    现在支持只使用 .pt 模型文件进行加载和保存，不依赖 .ckpt 检查点。
    """
    config = parse_arguments()

    set_seed(config.seed)

    dataset_name = config.dataset_name
    save_dir = Path(f"./save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {save_dir}")

    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")

    wandb_logger = WandbLogger(
        name=f"scGPT_finetune_{dataset_name}_{time.strftime('%b%d-%H-%M')}",
        project="scGPT_Lightning",
        config=vars(config),
        save_dir=str(save_dir),
        reinit=True,
    )
    print(config)

    data_module = ScGPTDataModule(config)
    data_module.prepare_data()
    data_module.setup(stage="fit")

    model_module = ScGPTModel(
        config=config,
        vocab=data_module.vocab,
        num_types=data_module.num_types,
        num_batch_types=data_module.num_batch_types,
        id2type=data_module.id2type,
        data_module_instance=data_module
    )

    # 这里的 ModelCheckpoint 回调仍然会生成 .ckpt 文件，如果你不想要，可以注释掉
    # 但建议保留，它提供自动保存最佳模型的功能，即使你只手动加载 .pt
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir / "checkpoints",
        filename="best_model-{epoch:02d}-{valid/loss_cls:.4f}",
        monitor="valid/loss_cls",
        mode="min",
        save_top_k=1,
        every_n_epochs=config.save_eval_interval,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else "auto",
        precision=16 if config.amp else 32,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        check_val_every_n_epoch=config.save_eval_interval,
        log_every_n_steps=config.log_interval,
        enable_checkpointing=True,  # 必须为True才能使用ModelCheckpoint回调
    )

    # --- 关键修改：手动管理 .pt 文件的加载和保存，不依赖 Trainer 的 ckpt_path ---
    if config.do_train:
        print("Starting training...")

        # trainer.fit() 不再传入 ckpt_path 参数，因为它只接受 .ckpt 文件
        # 模型权重加载由 ScGPTModel.__init__ 完成，通过 config.load_model 路径寻找 .pt 文件
        trainer.fit(model_module, datamodule=data_module)

        # 训练结束后，手动保存最佳模型和最终模型的 .pt 文件
        # 这里的 best_model_path 是 ModelCheckpoint 回调保存的 .ckpt 文件路径
        # 你可以根据需求，手动加载并转换成 .pt 文件
        if checkpoint_callback.best_model_path:
            scg.logger.info(
                f"Training finished. Best PyTorch Lightning checkpoint saved at: {checkpoint_callback.best_model_path}")
            # 手动加载最佳 .ckpt 文件，然后保存其模型状态字典为 .pt
            best_lightning_checkpoint = torch.load(checkpoint_callback.best_model_path, map_location='cpu')  # 加载到CPU
            best_model_state_dict = model_module.state_dict()  # 获取当前模型（已更新）的状态字典
            # 如果你想加载检查点里的状态字典：
            # best_model_state_dict = best_lightning_checkpoint['state_dict']
            # 注意：如果直接用best_lightning_checkpoint['state_dict']，需要确保键名匹配模型结构
            # 通常需要移除前缀 'model.' 如果它被添加了
            # best_model_state_dict = {k.replace('model.', ''): v for k, v in best_model_state_dict.items() if k.startswith('model.')}

            final_best_pt_path = save_dir / "best_model_final.pt"
            torch.save(best_model_state_dict, final_best_pt_path)
            scg.logger.info(f"Manually saved best model weights to {final_best_pt_path}")
        else:
            scg.logger.warning("No best model checkpoint was saved by ModelCheckpoint callback.")
            final_last_pt_path = save_dir / "last_model_final.pt"
            torch.save(model_module.state_dict(), final_last_pt_path)  # 保存训练结束时的模型状态
            scg.logger.info(f"Manually saved last model weights to {final_last_pt_path}")

        # 进行测试（使用训练结束时 model_module 的状态）
        # 这里不再使用 ckpt_path，因为我们不从 .ckpt 文件加载，而是使用当前 model_module 的状态
        data_module.setup(stage="test")  # 确保数据模块为测试阶段准备好
        trainer.test(model_module, datamodule=data_module)
    else:
        # 如果不进行训练，只进行测试
        if config.load_model:
            model_load_path = Path(config.load_model)
            if model_load_path.is_file() and str(model_load_path).endswith(".pt"):
                # 如果 load_model 直接指向一个 .pt 文件
                scg.logger.info(f"Loading .pt model from {model_load_path} for testing only.")
                try:
                    model_module.model.load_state_dict(torch.load(model_load_path, map_location=model_module.device))
                except Exception as e:
                    scg.logger.error(f"Failed to load .pt file {model_load_path} for testing: {e}")
                    sys.exit("Error: Could not load the specified .pt model file for testing.")
            elif model_load_path.is_dir():
                # 如果 load_model 指向一个目录 (期望目录内有 best_model.pt)
                scg.logger.info(f"Loading pre-trained model from directory {model_load_path} for testing only.")
                # model_module 已经在顶部初始化时，会从 config.load_model 目录中加载 best_model.pt
                # 所以这里无需再次加载，它已经加载过了
            else:
                scg.logger.warning(
                    f"Invalid load_model path or type for testing: {config.load_model}. Starting test with randomly initialized model.")
        else:
            scg.logger.warning("No model specified for testing. Starting test with randomly initialized model.")

        data_module.setup(stage="test")
        trainer.test(model_module, datamodule=data_module)

    # Wandb 会在 Trainer 结束时自动 finish
    # wandb.finish() # 不需要手动调用，由 WandbLogger 管理


if __name__ == "__main__":
    main()