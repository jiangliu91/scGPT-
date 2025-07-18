# scGPT-peeling-test

本项目旨在研究 `scGPT` 模型在细胞类型分类任务中的**输入表征的消融研究**以及**关键超参数的系统性灵敏度分析**。我们通过引入 **YAML 配置文件**和 **Weights & Biases (W&B) Sweep** 来自动化实验流程和超参数调优。

## 1. 输入表征消融研究

通过修改模型中用于组合 **基因身份（src** 和 **基因表达水平（values)** 嵌入的方式，以确定它们对细胞类型分类任务的相对贡献。

**可调参数**：`total_embs`
* **可选值**：`"src"` (仅使用基因身份嵌入), `"values"` (仅使用基因表达水平嵌入), `"both"` (两者都使用)。
* **修改方法**：通过 **`config.yaml` 文件**或**命令行参数 `--total_embs`** 进行配置。

## 2. 关键超参数系统性灵敏度分析

对以下关键超参数进行了系统性灵敏度分析，以评估它们对模型性能的影响：

### a. 零表达基因的影响

分析了包含或排除零表达基因对模型性能的影响。

**可调参数**：`include_zero_gene`
* **可选值**：`True` / `False`。
* **修改方法**：通过 **`config.yaml` 文件**或**命令行参数 `--include_zero_gene`** 进行配置。

### b. 表达值离散化的影响

我们研究了不同表达值离散化粒度对模型性能的影响。

**可调参数**：`n_bins`
* **可选值**：整数值，例如 `50`, `100`, `300`, `1000`。
* **修改方法**：通过 **`config.yaml` 文件**或**命令行参数 `--n_bins`** 进行配置。

### c. 输入序列长度的影响

我们评估了输入序列长度对模型性能的影响。

**可调参数**：`max_seq_len`
* **可选值**：整数值，例如 `50`, `200`, `3000`。
* **修改方法**：通过 **`config.yaml` 文件**或**命令行参数 `--max_seq_len`** 进行配置。

### d. 训练轮次的影响

评估了不同训练轮次对模型性能的影响。

**可调参数**：`epochs`
* **可选值**：整数值，例如 `10`, `30`, `100`。
* **修改方法**：通过 **`config.yaml` 文件**或**命令行参数 `--epochs`** 进行配置。

---

## 如何运行代码

本项目使用 Weights & Biases (W&B) Sweep 进行超参数调优。请按照以下步骤运行项目代码：

### 1. 环境准备

1.  **创建 Conda 环境**：
    ```bash
    conda env create -f environment.yml -n scgpt
    conda activate scgpt
    ```
2.  **配置 W&B 账户**:
    如果尚未登录，运行以下命令并根据提示操作：
    ```bash
    wandb login
    ```

### 2. 配置参数

所有可调参数的默认值都存储在 `config.yaml` 文件中。你可以直接修改此文件来设置默认值，也可以通过命令行参数在运行时覆盖它们。
### 3. 运行超参数 Sweep

创建或修改 'tutorials/sweep_config.yaml' 文件，指定要调优的参数及其搜索范围,在 tutorials/ 目录下运行：
```bash
wandb sweep sweep_config.yaml 
```

复制并运行上一步生成的 wandb agent 命令。你可以在同一台机器上启动多个代理来并行运行实验。


### 4. 单次运行脚本 (可选)

如果你不想进行 Sweep，只想用特定参数运行单次实验，可以直接通过命令行覆盖 config.yaml 中的默认值：
```bash
python Annotation_peeling_test.py --n_bins 51 --max_seq_len 3001 --epochs 10 --total_embs both
```
### 5.新增使用Lighting封装程序
新增使用Lighting封装的程序：'Annotation_peeling_lighting.py'，模块化模型：
```bash
python Annotation_peeling_lighting.py
```

