import argparse
import copy
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import warnings
import pandas as pd
import pickle
import torch
from anndata import AnnData
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
from typing import List, Tuple, Dict, Union, Optional
sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

import yaml
from types import SimpleNamespace

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tuning scGPT for Cell-type Annotation")
    parser.add_argument("--config_file", type=str, default="config.yaml",
                        help="Path to the YAML configuration file.")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility.")
    parser.add_argument("--dataset_name", type=str, help="Dataset name (e.g., 'ms').")
    parser.add_argument("--do_train", type=bool, help="Whether to perform training.")
    parser.add_argument("--load_model", type=str, help="Path to pre-trained scGPT model directory.")
    parser.add_argument("--mask_ratio", type=float, help="Masking ratio for MLM (0.0 for cell-type classification).")
    parser.add_argument("--epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--n_bins", type=int, help="Number of bins for binned input.")
    parser.add_argument("--max_seq_len", type=int, help="max_seq_len.")
    parser.add_argument("--MVC", action="store_true", help="Enable Masked Value Prediction for Cell Embedding (MVC).")
    parser.add_argument("--ecs_thres", type=float, help="Elastic Cell Similarity objective threshold (0.0 to disable).")
    parser.add_argument("--dab_weight", type=float, help="Weight for Domain Adaptation by Reverse Backpropagation (DAB).")
    parser.add_argument("--lr", type=float, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, help="Batch size for training.")
    parser.add_argument("--layer_size", type=int, help="Embedding and hidden dimension of the model.")
    parser.add_argument("--nlayers", type=int, help="Number of TransformerEncoderLayer in TransformerEncoder.")
    parser.add_argument("--nhead", type=int, help="Number of heads in nn.MultiheadAttention.")
    parser.add_argument("--dropout", type=float, help="Dropout probability.")
    parser.add_argument("--schedule_ratio", type=float, help="Ratio of epochs for learning rate schedule.")
    parser.add_argument("--save_eval_interval", type=int, help="Epochs interval to save model and evaluate.")
    parser.add_argument("--fast_transformer", action="store_true", help="Use fast transformer implementation.")
    parser.add_argument("--pre_norm", action="store_true", help="Apply pre-normalization in Transformer.")
    parser.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision (AMP).")
    parser.add_argument("--include_zero_gene", type=bool, help="If True, include zero genes among hvgs in training.")
    parser.add_argument("--freeze", action="store_true", help="Freeze pre-decoder weights during fine-tuning.")
    parser.add_argument("--DSBN", action="store_true", help="Enable Domain-specific Batch Normalization (DSBN).")
    parser.add_argument("--total_embs", type=str, help="Source for total_embs: 'src', 'values', or 'both'.")
    # 训练目标开关
    parser.add_argument("--MLM", action="store_true", help="Enable Masked Language Modeling objective.")
    parser.add_argument("--CLS", type=bool, help="Enable Cell Type Classification objective.")
    parser.add_argument("--ADV", action="store_true", help="Enable Adversarial training for batch correction.")
    parser.add_argument("--CCE", action="store_true", help="Enable Contrastive Cell Embedding objective.")
    parser.add_argument("--ECS_enabled", type=bool, help="Enable Elastic Cell Similarity objective (ECS).") # 使用新名称避免歧义
    parser.add_argument("--DAB_enabled", type=bool, help="Enable Domain Adaptation by Reverse Backpropagation (DAB).") # 使用新名称避免歧义
    parser.add_argument("--INPUT_BATCH_LABELS", action="store_true", help="Use batch labels as input features.")


    cmd_args = parser.parse_args()

    config_dict = {}
    if cmd_args.config_file and Path(cmd_args.config_file).exists():
        with open(cmd_args.config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
    else:
        print(f"Warning: Config file '{cmd_args.config_file}' not found. Using only command-line defaults or specified values.")

    final_config = {}

    if config_dict:
        final_config.update(config_dict)

    cmd_arg_dest_names = [action.dest for action in parser._actions if action.dest]

    for dest_name in cmd_arg_dest_names:
        cmd_value = getattr(cmd_args, dest_name, None)

        is_boolean_flag = any(isinstance(a, (argparse._StoreTrueAction, argparse._StoreFalseAction)) and a.dest == dest_name for a in parser._actions)

        if is_boolean_flag:

            if f'--{dest_name}' in sys.argv or (f'--no-{dest_name}' in sys.argv if hasattr(cmd_args, f'no_{dest_name}') else False):
                final_config[dest_name] = cmd_value
            elif dest_name not in final_config:
                final_config[dest_name] = cmd_value
        else:
            if cmd_value is not None:
                final_config[dest_name] = cmd_value
            elif dest_name not in final_config and cmd_value is None:
                final_config[dest_name] = cmd_value

    for action in parser._actions:
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            if action.dest not in final_config:
                final_config[action.dest] = action.default if action.default is not None else False
        elif action.dest not in final_config and action.default is not None:
            final_config[action.dest] = action.default

    config_obj = SimpleNamespace(**final_config)

    if hasattr(config_obj, 'ECS_enabled') and not config_obj.ECS_enabled:
        config_obj.ecs_thres = 0.0
    if hasattr(config_obj, 'DAB_enabled') and not config_obj.DAB_enabled:
        config_obj.dab_weight = 0.0

    return config_obj



def main():
    config = parse_arguments()
    dataset_name = config.dataset_name
    save_dir = Path(f"./save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save to {save_dir}")
    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")

    run = wandb.init(
        config=vars(config),
        project="scGPT",
        reinit=True,
        settings=wandb.Settings(start_method="fork"),
        dir=save_dir
    )
    print(config)

    set_seed(config.seed)


    # settings for input and preprocessing
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    mask_ratio = config.mask_ratio
    mask_value = "auto"  # for masked values, now it should always be auto

    include_zero_gene = config.include_zero_gene  # if True, include zero genes among hvgs in the training
    max_seq_len = config.max_seq_len
    n_bins = config.n_bins

    # input/output representation
    input_style = "binned"  # "normed_raw", "log1p", or "binned"
    output_style = "binned"  # "normed_raw", "log1p", or "binned"

    # settings for training
    MLM = config.MLM  # now configurable via command line
    CLS = config.CLS  # now configurable via command line
    ADV = config.ADV  # now configurable via command line
    CCE = config.CCE  # now configurable via command line
    MVC = config.MVC  # Masked value prediction for cell embedding, configurable via command line
    ECS = config.ecs_thres > 0  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    DAB = config.DAB_enabled # Domain adaptation by reverse backpropagation, now configurable via command line
    INPUT_BATCH_LABELS = config.INPUT_BATCH_LABELS # now configurable via command line
    input_emb_style = "continuous"  # "category" or "continuous" or "scaling"
    cell_emb_style = "cls"  # "avg-pool" or "w-pool" or "cls"
    adv_E_delay_epochs = 0  # delay adversarial training on encoder for a few epochs
    adv_D_delay_epochs = 0
    mvc_decoder_style = "inner product"
    ecs_threshold = config.ecs_thres
    dab_weight = config.dab_weight

    explicit_zero_prob = MLM and include_zero_gene  # whether explicit bernoulli for zeros
    do_sample_in_train = False and explicit_zero_prob  # sample the bernoulli in training

    per_seq_batch_sample = False

    # settings for optimizer
    lr = config.lr  # TODO: test learning rate ratio between two tasks
    lr_ADV = 1e-3  # learning rate for discriminator, used when ADV is True
    batch_size = config.batch_size
    eval_batch_size = config.batch_size
    epochs = config.epochs
    schedule_interval = 1 # original fixed value

    # settings for the model
    fast_transformer = config.fast_transformer
    fast_transformer_backend = "flash"  # "linear" or "flash"
    embsize_initial = config.layer_size # Use a temporary var for initial config before model load
    d_hid_initial = config.layer_size # Use a temporary var for initial config before model load
    nlayers_initial = config.nlayers # Use a temporary var for initial config before model load
    nhead_initial = config.nhead # Use a temporary var for initial config before model load
    dropout = config.dropout  # dropout probability

    # logging
    log_interval = 100  # iterations
    save_eval_interval = config.save_eval_interval  # epochs
    do_eval_scib_metrics = True


    assert input_style in ["normed_raw", "log1p", "binned"]
    assert output_style in ["normed_raw", "log1p", "binned"]
    assert input_emb_style in ["category", "continuous", "scaling"]
    if input_style == "binned":
        if input_emb_style == "scaling":
            raise ValueError("input_emb_style `scaling` is not supported for binned input.")
    elif input_style == "log1p" or input_style == "normed_raw":
        if input_emb_style == "category":
            raise ValueError(
                "input_emb_style `category` is not supported for log1p or normed_raw input."
            )

    if input_emb_style == "category":
        mask_value = n_bins + 1
        pad_value = n_bins  # for padding gene expr values
        n_input_bins = n_bins + 2
    else:
        mask_value = -1
        pad_value = -2
        n_input_bins = n_bins


    if ADV and DAB:
        raise ValueError("ADV and DAB cannot be both True.")
    DAB_separate_optim = True if DAB > 1 else False # Original logic, DAB here refers to the DAB_flag




    if dataset_name == "ms":
        data_dir = Path("../data/ms") # Hardcoded, you can make this a command line arg if needed
        adata = sc.read(data_dir / "c_data.h5ad")
        adata_test = sc.read(data_dir / "filtered_ms_adata.h5ad")
        adata.obs["celltype"] = adata.obs["Factor Value[inferred cell type - authors labels]"].astype("category")
        adata_test.obs["celltype"] = adata_test.obs["Factor Value[inferred cell type - authors labels]"].astype("category")
        adata.obs["batch_id"]  = adata.obs["str_batch"] = "0"
        adata_test.obs["batch_id"]  = adata_test.obs["str_batch"] = "1"
        adata.var.set_index(adata.var["gene_name"], inplace=True)
        adata_test.var.set_index(adata.var["gene_name"], inplace=True)
        data_is_raw = False # Hardcoded
        filter_gene_by_counts = False # Hardcoded
        adata_test_raw = adata_test.copy()
        adata = adata.concatenate(adata_test, batch_key="str_batch")

    # make the batch category column
    batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
    adata.obs["batch_id"] = batch_id_labels
    celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
    celltypes = adata.obs["celltype"].unique()
    num_types = len(np.unique(celltype_id_labels))
    id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
    adata.obs["celltype_id"] = celltype_id_labels
    adata.var["gene_name"] = adata.var.index.tolist()


    if config.load_model is not None:
        model_dir = Path(config.load_model)
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"
        vocab_file = model_dir / "vocab.json"

        vocab = GeneVocab.from_file(vocab_file)
        shutil.copy(vocab_file, save_dir / "vocab.json")
        for s in special_tokens: # special_tokens defined in In[4]
            if s not in vocab:
                vocab.append_token(s)

        adata.var["id_in_vocab"] = [
            1 if gene in vocab else -1 for gene in adata.var["gene_name"]
        ]
        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        logger.info(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )
        adata = adata[:, adata.var["id_in_vocab"] >= 0]

        # model configs are loaded from args.json, these still override config values in original script
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        logger.info(
            f"Resume model from {model_file}, the model args will override the "
            f"config {model_config_file}."
        )
        embsize = model_configs["embsize"]
        nhead = model_configs["nheads"]
        d_hid = model_configs["d_hid"]
        nlayers = model_configs["nlayers"]
        n_layers_cls = model_configs["n_layers_cls"] # Not in config, but derived from loaded model


    # set up the preprocessor, use the args to config the workflow
    preprocessor = Preprocessor(
        use_key="X",  # the key in adata.layers to use as raw data
        filter_gene_by_counts=filter_gene_by_counts,  # step 1, hardcoded
        filter_cell_by_counts=False,  # step 2, hardcoded
        normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum, hardcoded
        result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        log1p=data_is_raw,  # 4. whether to log1p the normalized data, hardcoded
        result_log1p_key="X_log1p",
        subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes, hardcoded
        hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger", # hardcoded
        binning=n_bins,  # 6. whether to bin the raw data and to what number of bins, uses n_bins from config
        result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    )


    adata_test_split = adata[adata.obs["str_batch"] == "1"].copy() # Renamed to avoid confusion with original adata_test_raw
    adata_train_split = adata[adata.obs["str_batch"] == "0"].copy() # Renamed

    preprocessor(adata_train_split, batch_key=None)
    preprocessor(adata_test_split, batch_key=None)


    input_layer_key = {  # the values of this map coorespond to the keys in preprocessing
        "normed_raw": "X_normed",
        "log1p": "X_normed",
        "binned": "X_binned",
    }[input_style]

    all_counts = (
        adata_train_split.layers[input_layer_key].toarray() # Use adata_train_split here for training data
        if issparse(adata_train_split.layers[input_layer_key])
        else adata_train_split.layers[input_layer_key]
    )
    genes = adata.var["gene_name"].tolist() # genes from the full concatenated adata

    celltypes_labels = adata_train_split.obs["celltype_id"].tolist()  # make sure count from 0
    celltypes_labels = np.array(celltypes_labels)

    batch_ids = adata_train_split.obs["batch_id"].tolist()
    num_batch_types = len(set(adata.obs["batch_id"].tolist())) # Total batch types from full adata
    batch_ids = np.array(batch_ids)

    (
        train_data,
        valid_data,
        train_celltype_labels,
        valid_celltype_labels,
        train_batch_labels,
        valid_batch_labels,
    ) = train_test_split(
        all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True, random_state=config.seed # Added random_state for reproducibility
    )


    gene_ids = np.array(vocab(genes), dtype=int)




    tokenized_train = tokenize_and_pad_batch(
        train_data,
        gene_ids,
        max_len=max_seq_len, # max_seq_len defined in In[4]
        vocab=vocab,
        pad_token=pad_token, # pad_token defined in In[4]
        pad_value=pad_value, # pad_value defined in In[5] based on n_bins
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=include_zero_gene, # include_zero_gene defined in In[4]
    )
    tokenized_valid = tokenize_and_pad_batch(
        valid_data,
        gene_ids,
        max_len=max_seq_len, # max_seq_len defined in In[4]
        vocab=vocab,
        pad_token=pad_token, # pad_token defined in In[4]
        pad_value=pad_value, # pad_value defined in In[5] based on n_bins
        append_cls=True,
        include_zero_gene=include_zero_gene, # include_zero_gene defined in In[4]
    )
    logger.info(
        f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
        f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
    )
    logger.info(
        f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
        f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
    )



    def prepare_data(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
        masked_values_train = random_mask_value(
            tokenized_train["values"],
            mask_ratio=mask_ratio, # From In[4]
            mask_value=mask_value, # From In[5]
            pad_value=pad_value, # From In[5]
        )
        masked_values_valid = random_mask_value(
            tokenized_valid["values"],
            mask_ratio=mask_ratio, # From In[4]
            mask_value=mask_value, # From In[5]
            pad_value=pad_value, # From In[5]
        )
        # The 'epoch' variable is global in the original notebook context
        print(
            f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
            f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
        )

        input_gene_ids_train, input_gene_ids_valid = (
            tokenized_train["genes"],
            tokenized_valid["genes"],
        )
        input_values_train, input_values_valid = masked_values_train, masked_values_valid
        target_values_train, target_values_valid = (
            tokenized_train["values"],
            tokenized_valid["values"],
        )

        tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
        tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()

        tensor_celltype_labels_train = torch.from_numpy(train_celltype_labels).long()
        tensor_celltype_labels_valid = torch.from_numpy(valid_celltype_labels).long()

        if sort_seq_batch:  # TODO: update to random pick seq source in each traning batch
            train_sort_ids = np.argsort(train_batch_labels)
            input_gene_ids_train = input_gene_ids_train[train_sort_ids]
            input_values_train = input_values_train[train_sort_ids]
            target_values_train = target_values_train[train_sort_ids]
            tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
            tensor_celltype_labels_train = tensor_celltype_labels_train[train_sort_ids]

            valid_sort_ids = np.argsort(valid_batch_labels)
            input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
            input_values_valid = input_values_valid[valid_sort_ids]
            target_values_valid = target_values_valid[valid_sort_ids]
            tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]
            tensor_celltype_labels_valid = tensor_celltype_labels_valid[valid_sort_ids]

        train_data_pt = {
            "gene_ids": input_gene_ids_train,
            "values": input_values_train,
            "target_values": target_values_train,
            "batch_labels": tensor_batch_labels_train,
            "celltype_labels": tensor_celltype_labels_train,
        }
        valid_data_pt = {
            "gene_ids": input_gene_ids_valid,
            "values": input_values_valid,
            "target_values": target_values_valid,
            "batch_labels": tensor_batch_labels_valid,
            "celltype_labels": tensor_celltype_labels_valid,
        }

        return train_data_pt, valid_data_pt


    # dataset
    class SeqDataset(Dataset):
        def __init__(self, data: Dict[str, torch.Tensor]):
            self.data = data

        def __len__(self):
            return self.data["gene_ids"].shape[0]

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.data.items()}


    # data_loader
    def prepare_dataloader(
        data_pt: Dict[str, torch.Tensor],
        batch_size: int, # From In[4]
        shuffle: bool = False,
        intra_domain_shuffle: bool = False,
        drop_last: bool = False,
        num_workers: int = 0,
    ) -> DataLoader:
        if num_workers == 0:
            num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)

        dataset = SeqDataset(data_pt)

        if per_seq_batch_sample: # From In[4]
            # find the indices of samples in each seq batch
            subsets = []
            batch_labels_array = data_pt["batch_labels"].numpy()
            for batch_label in np.unique(batch_labels_array):
                batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
                subsets.append(batch_indices)
            data_loader = DataLoader(
                dataset=dataset,
                batch_sampler=SubsetsBatchSampler(
                    subsets,
                    batch_size,
                    intra_subset_shuffle=intra_domain_shuffle,
                    inter_subset_shuffle=shuffle,
                    drop_last=drop_last,
                ),
                num_workers=num_workers,
                pin_memory=True,
            )
            return data_loader

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader


    # ## Step 3: Load the pre-trained scGPT model


    if config.total_embs == 'values':
        values =1
        src = 0
    elif config.total_embs == 'src':
        values =0
        src = 1
    else :
        values =1
        src = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        # 打印当前使用的GPU的名称
        print(f"当前使用的GPU: {torch.cuda.get_device_name(0)}")
        # 打印当前使用的GPU的索引
        print(f"当前CUDA设备索引: {torch.cuda.current_device()}")
    else:
        print("正在使用CPU进行训练。")
    ntokens = len(vocab)  # size of vocabulary
    model = TransformerModel(
        ntokens,
        embsize, # From In[8]
        nhead, # From In[8]
        d_hid, # From In[8]
        nlayers, # From In[8]
        nlayers_cls=n_layers_cls, # From In[8]
        n_cls=num_types if CLS else 1, # CLS from In[4]
        vocab=vocab,
        dropout=dropout, # From In[4]
        pad_token=pad_token, # From In[4]
        pad_value=pad_value, # From In[5]
        do_mvc=MVC, # From In[4]
        do_dab=DAB, # From In[4]
        use_batch_labels=INPUT_BATCH_LABELS, # From In[4]
        num_batch_labels=num_batch_types, # From In[10]
        domain_spec_batchnorm=config.DSBN, # From config
        input_emb_style=input_emb_style, # From In[4]
        n_input_bins=n_input_bins, # From In[5]
        cell_emb_style=cell_emb_style, # From In[4]
        mvc_decoder_style=mvc_decoder_style, # From In[4]
        ecs_threshold=ecs_threshold, # From In[4]
        explicit_zero_prob=explicit_zero_prob, # From In[4]
        use_fast_transformer=fast_transformer, # From In[4]
        fast_transformer_backend=fast_transformer_backend, # From In[4]
        pre_norm=config.pre_norm, # From config
        values_choose=values,
        src_choose=src,

    )
    if config.load_model is not None:
        try:
            model.load_state_dict(torch.load(model_file))
            logger.info(f"Loading all model params from {model_file}")
        except:
            # only load params that are in the model and match the size
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_file)
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            for k, v in pretrained_dict.items():
                logger.info(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

    pre_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

    # Freeze all pre-decoder weights
    for name, para in model.named_parameters():
        print("-"*20)
        print(f"name: {name}")
        if config.freeze and "encoder" in name and "transformer_encoder" not in name: # config.freeze from arguments
        # if config.freeze and "encoder" in name:
            print(f"freezing weights for: {name}")
            para.requires_grad = False

    post_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

    logger.info(f"Total Pre freeze Params {(pre_freeze_param_count )}")
    logger.info(f"Total Post freeze Params {(post_freeze_param_count )}")
    wandb.log(
            {
                "info/pre_freeze_param_count": pre_freeze_param_count,
                "info/post_freeze_param_count": post_freeze_param_count,
            },
    )

    model.to(device)
    wandb.watch(model)

    discriminator = None # Initialize discriminator here
    if ADV: # ADV from In[4]
        discriminator = AdversarialDiscriminator(
            d_model=embsize, # From In[8]
            n_cls=num_batch_types, # From In[10]
        ).to(device)



    criterion = masked_mse_loss
    criterion_cls = nn.CrossEntropyLoss()
    criterion_dab = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, eps=1e-4 if config.amp else 1e-8 # lr from In[4], config.amp from config
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, schedule_interval, gamma=config.schedule_ratio # schedule_interval from In[4], config.schedule_ratio from config
    )
    if DAB_separate_optim: # From In[5]
        optimizer_dab = torch.optim.Adam(model.parameters(), lr=lr) # lr from In[4]
        scheduler_dab = torch.optim.lr_scheduler.StepLR(
            optimizer_dab, schedule_interval, gamma=config.schedule_ratio # schedule_interval from In[4], config.schedule_ratio from config
        )
    if ADV: # ADV from In[4]
        criterion_adv = nn.CrossEntropyLoss()  # consider using label smoothing
        optimizer_E = torch.optim.Adam(model.parameters(), lr=lr_ADV) # lr_ADV from In[4]
        scheduler_E = torch.optim.lr_scheduler.StepLR(
            optimizer_E, schedule_interval, gamma=config.schedule_ratio # schedule_interval from In[4], config.schedule_ratio from config
        )
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_ADV) # lr_ADV from In[4]
        scheduler_D = torch.optim.lr_scheduler.StepLR(
            optimizer_D, schedule_interval, gamma=config.schedule_ratio # schedule_interval from In[4], config.schedule_ratio from config
        )

    scaler = torch.cuda.amp.GradScaler(enabled=config.amp) # config.amp from config



    def train(model: nn.Module, loader: DataLoader) -> None:
        """
        Train the model for one epoch.
        """
        model.train()
        (
            total_loss,
            total_mse,
            total_cls,
            total_cce,
            total_mvc,
            total_ecs,
            total_dab,
            total_adv_E,
            total_adv_D,
            total_zero_log_prob,
            total_mvc_zero_log_prob,
        ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        total_error = 0.0
        start_time = time.time()

        num_batches = len(loader)
        for batch, batch_data in enumerate(loader):
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token]) # pad_token from In[4]
            with torch.cuda.amp.autocast(enabled=config.amp): # config.amp from config
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if INPUT_BATCH_LABELS or config.DSBN else None, # INPUT_BATCH_LABELS from In[4], config.DSBN from config
                    CLS=CLS, # CLS from In[4]
                    CCE=CCE, # CCE from In[4]
                    MVC=MVC, # MVC from In[4]
                    ECS=ECS, # ECS from In[4]
                    do_sample=do_sample_in_train, # do_sample_in_train from In[4]
                    #generative_training=False
                )

                masked_positions = input_values.eq(mask_value)  # the postions to predict, mask_value from In[5]
                loss = 0.0
                metrics_to_log = {}
                if MLM: # MLM from In[4]
                    loss_mse = criterion(
                        output_dict["mlm_output"], target_values, masked_positions
                    )
                    loss = loss + loss_mse
                    metrics_to_log = {"train/mse": loss_mse.item()}
                if explicit_zero_prob: # explicit_zero_prob from In[4]
                    loss_zero_log_prob = criterion_neg_log_bernoulli(
                        output_dict["mlm_zero_probs"], target_values, masked_positions
                    )
                    loss = loss + loss_zero_log_prob
                    metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
                if CLS: # CLS from In[4]
                    loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)
                    loss = loss + loss_cls
                    metrics_to_log.update({"train/cls": loss_cls.item()})

                    error_rate = 1 - (
                        (output_dict["cls_output"].argmax(1) == celltype_labels)
                        .sum()
                        .item()
                    ) / celltype_labels.size(0)
                if CCE: # CCE from In[4]
                    loss_cce = 10 * output_dict["loss_cce"]
                    loss = loss + loss_cce
                    metrics_to_log.update({"train/cce": loss_cce.item()})
                if MVC: # MVC from In[4]
                    loss_mvc = criterion(
                        output_dict["mvc_output"], target_values, masked_positions
                    )
                    loss = loss + loss_mvc
                    metrics_to_log.update({"train/mvc": loss_mvc.item()})
                if MVC and explicit_zero_prob: # MVC and explicit_zero_prob from In[4]
                    loss_mvc_zero_log_prob = criterion_neg_log_bernoulli(
                        output_dict["mvc_zero_probs"], target_values, masked_positions
                    )
                    loss = loss + loss_mvc_zero_log_prob
                    metrics_to_log.update({"train/mvc_nzlp": loss_mvc_zero_log_prob.item()})
                if ECS: # ECS from In[4]
                    loss_ecs = 10 * output_dict["loss_ecs"]
                    loss = loss + loss_ecs
                    metrics_to_log.update({"train/ecs": loss_ecs.item()})
                if DAB: # DAB from In[4]
                    # try weighting and separate optimizer
                    loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
                    loss = loss + dab_weight * loss_dab # dab_weight from In[4]
                    metrics_to_log.update({"train/dab": loss_dab.item()})

            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                    error_if_nonfinite=False if scaler.is_enabled() else True,
                )
                if len(w) > 0:
                    logger.warning(
                        f"Found infinite gradient. This may be caused by the gradient "
                        f"scaler. The current scale is {scaler.get_scale()}. This warning "
                        "can be ignored if no longer occurs after autoscaling of the scaler."
                    )
            scaler.step(optimizer)
            scaler.update()

            if ADV: # ADV from In[4]
                # rerun the model for adversarial training
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if INPUT_BATCH_LABELS or config.DSBN else None, # INPUT_BATCH_LABELS from In[4], config.DSBN from config
                    CLS=CLS, # CLS from In[4]
                    CCE=CCE, # CCE from In[4]
                    MVC=MVC, # MVC from In[4]
                    ECS=ECS, # ECS from In[4]
                    do_sample=do_sample_in_train, # do_sample_in_train from In[4]
                    #generative_training=False
                )

                # TRAINING DISCRIMINATOR
                loss_adv_D = criterion_adv(
                    discriminator(output_dict["cell_emb"].detach()), batch_labels
                )
                if epoch > adv_D_delay_epochs: # adv_D_delay_epochs from In[4]
                    discriminator.zero_grad()
                    loss_adv_D.backward()
                    optimizer_D.step()

                # TRAINING ENCODER
                loss_adv_E = -criterion_adv(
                    discriminator(output_dict["cell_emb"]), batch_labels
                )
                # NOTE: the loss is negative here because we want to maximize
                # the cross_entropy_loss, in other words, disguise against the discriminator
                if epoch > adv_E_delay_epochs: # adv_E_delay_epochs from In[4]
                    model.zero_grad()
                    discriminator.zero_grad()
                    loss_adv_E.backward()
                    optimizer_E.step()

            wandb.log(metrics_to_log)

            total_loss += loss.item()
            total_mse += loss_mse.item() if MLM else 0.0 # MLM from In[4]
            total_cls += loss_cls.item() if CLS else 0.0 # CLS from In[4]
            total_cce += loss_cce.item() if CCE else 0.0 # CCE from In[4]
            total_mvc += loss_mvc.item() if MVC else 0.0 # MVC from In[4]
            total_ecs += loss_ecs.item() if ECS else 0.0 # ECS from In[4]
            total_dab += loss_dab.item() if DAB else 0.0 # DAB from In[4]
            total_adv_E += loss_adv_E.item() if ADV else 0.0 # ADV from In[4]
            total_adv_D += loss_adv_D.item() if ADV else 0.0 # ADV from In[4]
            total_zero_log_prob += loss_zero_log_prob.item() if explicit_zero_prob else 0.0 # explicit_zero_prob from In[4]
            total_mvc_zero_log_prob += (
                loss_mvc_zero_log_prob.item() if MVC and explicit_zero_prob else 0.0 # MVC, explicit_zero_prob from In[4]
            )
            total_error += error_rate
            if batch % log_interval == 0 and batch > 0: # log_interval from In[4]
                lr_current = scheduler.get_last_lr()[0] # renamed lr to lr_current to avoid conflict with global lr
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval # log_interval from In[4]
                cur_loss = total_loss / log_interval
                cur_mse = total_mse / log_interval
                cur_cls = total_cls / log_interval if CLS else 0.0 # CLS from In[4]
                cur_cce = total_cce / log_interval if CCE else 0.0 # CCE from In[4]
                cur_mvc = total_mvc / log_interval if MVC else 0.0 # MVC from In[4]
                cur_ecs = total_ecs / log_interval if ECS else 0.0 # ECS from In[4]
                cur_dab = total_dab / log_interval if DAB else 0.0 # DAB from In[4]
                cur_adv_E = total_adv_E / log_interval if ADV else 0.0 # ADV from In[4]
                cur_adv_D = total_adv_D / log_interval if ADV else 0.0 # ADV from In[4]
                cur_zero_log_prob = (
                    total_zero_log_prob / log_interval if explicit_zero_prob else 0.0 # explicit_zero_prob from In[4]
                )
                cur_mvc_zero_log_prob = (
                    total_mvc_zero_log_prob / log_interval
                    if MVC and explicit_zero_prob
                    else 0.0 # MVC, explicit_zero_prob from In[4]
                )
                cur_error = total_error / log_interval
                # ppl = math.exp(cur_loss)
                logger.info(
                    f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                    f"lr {lr_current:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.2f} | "
                    + (f"mse {cur_mse:5.2f} | mre {cur_error:5.2f} |" if MLM else "") # MLM from In[4]
                    + (f"cls {cur_cls:5.2f} | " if CLS else "") # CLS from In[4]
                    + (f"err {cur_error:5.2f} | " if CLS else "") # CLS from In[4]
                    + (f"cce {cur_cce:5.2f} |" if CCE else "") # CCE from In[4]
                    + (f"mvc {cur_mvc:5.2f} |" if MVC else "") # MVC from In[4]
                    + (f"ecs {cur_ecs:5.2f} |" if ECS else "") # ECS from In[4]
                    + (f"dab {cur_dab:5.2f} |" if DAB else "") # DAB from In[4]
                    + (f"adv_E {cur_adv_E:5.2f} |" if ADV else "") # ADV from In[4]
                    + (f"adv_D {cur_adv_D:5.2f} |" if ADV else "") # ADV from In[4]
                    + (f"nzlp {cur_zero_log_prob:5.2f} |" if explicit_zero_prob else "") # explicit_zero_prob from In[4]
                    + (
                        f"mvc_nzlp {cur_mvc_zero_log_prob:5.2f} |"
                        if MVC and explicit_zero_prob
                        else "" # MVC, explicit_zero_prob from In[4]
                    )
                )
                total_loss = 0
                total_mse = 0
                total_cls = 0
                total_cce = 0
                total_mvc = 0
                total_ecs = 0
                total_dab = 0
                total_adv_E = 0
                total_adv_D = 0
                total_zero_log_prob = 0
                total_mvc_zero_log_prob = 0
                total_error = 0
                start_time = time.time()


    def define_wandb_metrcis():
        wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
        wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
        wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
        wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
        wandb.define_metric("test/avg_bio", summary="max")

    def evaluate(model: nn.Module, loader: DataLoader, return_raw: bool = False) -> float:
        """
        Evaluate the model on the evaluation data.
        """
        model.eval()
        total_loss = 0.0
        total_error = 0.0
        total_dab = 0.0
        total_num = 0
        predictions = []
        with torch.no_grad():
            for batch_data in loader:
                input_gene_ids = batch_data["gene_ids"].to(device)
                input_values = batch_data["values"].to(device)
                target_values = batch_data["target_values"].to(device)
                batch_labels = batch_data["batch_labels"].to(device)
                celltype_labels = batch_data["celltype_labels"].to(device)

                src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
                with torch.cuda.amp.autocast(enabled=config.amp):
                    output_dict = model(
                        input_gene_ids,
                        input_values,
                        src_key_padding_mask=src_key_padding_mask,
                        batch_labels=batch_labels if INPUT_BATCH_LABELS or config.DSBN else None,
                        CLS=CLS,  # evaluation does not need CLS or CCE
                        CCE=False,
                        MVC=False,
                        ECS=False,
                        do_sample=do_sample_in_train,
                        # generative_training = False,
                    )
                    output_values = output_dict["cls_output"]
                    loss = criterion_cls(output_values, celltype_labels)

                    if DAB:
                        loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)

                total_loss += loss.item() * len(input_gene_ids)
                accuracy = (output_values.argmax(1) == celltype_labels).sum().item()
                total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
                total_dab += loss_dab.item() * len(input_gene_ids) if DAB else 0.0
                total_num += len(input_gene_ids)
                preds = output_values.argmax(1).cpu().numpy()
                predictions.append(preds)

        wandb.log(
            {
                "valid/mse": total_loss / total_num,
                "valid/err": total_error / total_num,
                "valid/dab": total_dab / total_num,
                "valid/sum_mse_dab": (total_loss + dab_weight * total_dab) / total_num, # dab_weight from In[4]
                "epoch": epoch, # epoch variable is global in original notebook context
            },
        )

        if return_raw:
            return np.concatenate(predictions, axis=0)

        return total_loss / total_num, total_error / total_num

    # ## Step 4: Finetune scGPT with task-specific objectives



    best_val_loss = float("inf")
    best_avg_bio = 0.0 # This variable doesn't seem to be used anywhere later for actual model selection
    best_model = None
    define_wandb_metrcis()

    for epoch in range(1, epochs + 1): # epochs from In[4]
        epoch_start_time = time.time()
        train_data_pt, valid_data_pt = prepare_data(sort_seq_batch=per_seq_batch_sample) # per_seq_batch_sample from In[4]
        train_loader = prepare_dataloader(
            train_data_pt,
            batch_size=batch_size, # batch_size from In[4]
            shuffle=True, # Changed to True for proper training shuffling
            intra_domain_shuffle=True, # Original was True
            drop_last=False,
        )
        valid_loader = prepare_dataloader(
            valid_data_pt,
            batch_size=eval_batch_size, # eval_batch_size from In[4]
            shuffle=False,
            intra_domain_shuffle=False,
            drop_last=False,
        )

        if config.do_train: # config.do_train from config
            train(
                model,
                loader=train_loader,
            )
        val_loss, val_err = evaluate(
            model,
            loader=valid_loader,
        )
        elapsed = time.time() - epoch_start_time
        logger.info("-" * 89)
        logger.info(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss/mse {val_loss:5.4f} | err {val_err:5.4f}" # Added dab to log
        )
        logger.info("-" * 89)

        # Original code uses val_loss for best_model selection
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            best_model_epoch = epoch
            logger.info(f"Best model with score {best_val_loss:5.4f}")

        scheduler.step()
        if DAB_separate_optim: # DAB_separate_optim from In[5]
            scheduler_dab.step()
        if ADV: # ADV from In[4]
            scheduler_D.step()
            scheduler_E.step()



    def test(model: nn.Module, adata: DataLoader) -> float:
        all_counts = (
            adata.layers[input_layer_key].toarray()
            if issparse(adata.layers[input_layer_key])
            else adata.layers[input_layer_key]
        )

        celltypes_labels = adata.obs["celltype_id"].tolist()  # make sure count from 0
        celltypes_labels = np.array(celltypes_labels)

        batch_ids = adata.obs["batch_id"].tolist()
        batch_ids = np.array(batch_ids)

        tokenized_test = tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=max_seq_len,
            vocab=vocab,
            pad_token=pad_token,
            pad_value=pad_value,
            append_cls=True,  # append <cls> token at the beginning
            include_zero_gene=include_zero_gene,
        )

        input_values_test = random_mask_value(
            tokenized_test["values"],
            mask_ratio=mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
        )

        test_data_pt = {
            "gene_ids": tokenized_test["genes"],
            "values": input_values_test,
            "target_values": tokenized_test["values"],
            "batch_labels": torch.from_numpy(batch_ids).long(),
            "celltype_labels": torch.from_numpy(celltypes_labels).long(),
        }

        test_loader = DataLoader(
            dataset=SeqDataset(test_data_pt),
            batch_size=eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=min(len(os.sched_getaffinity(0)), eval_batch_size // 2),
            pin_memory=True,
        )

        model.eval()
        predictions = evaluate(
            model,
            loader=test_loader,
            return_raw=True,
        )

        # compute accuracy, precision, recall, f1
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        accuracy = accuracy_score(celltypes_labels, predictions)
        precision = precision_score(celltypes_labels, predictions, average="macro")
        recall = recall_score(celltypes_labels, predictions, average="macro")
        macro_f1 = f1_score(celltypes_labels, predictions, average="macro")

        logger.info(
            f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, "
            f"Macro F1: {macro_f1:.3f}"
        )

        results = {
            "test/accuracy": accuracy,
            "test/precision": precision,
            "test/recall": recall,
            "test/macro_f1": macro_f1,
        }

        return predictions, celltypes_labels, results


    predictions, labels, results = test(best_model, adata_test_split)
    adata_test_raw.obs["predictions"] = [id2type[p] for p in predictions]

    # plot
    palette_ = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    palette_ = plt.rcParams["axes.prop_cycle"].by_key()["color"] + plt.rcParams["axes.prop_cycle"].by_key()["color"] + \
               plt.rcParams["axes.prop_cycle"].by_key()["color"]
    palette_ = {c: palette_[i] for i, c in enumerate(celltypes)}

    with plt.rc_context({"figure.figsize": (6, 4), "figure.dpi": (300)}):
        sc.pl.umap(
            adata_test_raw,
            color=["celltype", "predictions"],
            palette=palette_,
            show=False,
        )
        plt.savefig(save_dir / "results.png", dpi=300)

    save_dict = {
        "predictions": predictions,
        "labels": labels,
        "results": results,
        "id_maps": id2type
    }
    with open(save_dir / "results.pkl", "wb") as f:
        pickle.dump(save_dict, f)

    results["test/cell_umap"] = wandb.Image(
        str(save_dir / "results.png"),
        caption=f"predictions macro f1 {results['test/macro_f1']:.3f}",
    )
    wandb.log(results)

    from sklearn.metrics import confusion_matrix
    celltypes = list(celltypes)
    for i in set([id2type[p] for p in predictions]):
        if i not in celltypes:
            celltypes.remove(i)
    cm = confusion_matrix(labels, predictions)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm = pd.DataFrame(cm, index=celltypes[:cm.shape[0]], columns=celltypes[:cm.shape[1]])
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues")
    plt.savefig(save_dir / "confusion_matrix.png", dpi=300)

    results["test/confusion_matrix"] = wandb.Image(
        str(save_dir / "confusion_matrix.png"),
        caption=f"confusion matrix",
    )


    # save the model into the save_dir
    torch.save(best_model.state_dict(), save_dir / "model.pt")

    wandb.finish()



if __name__ == "__main__":
    main()