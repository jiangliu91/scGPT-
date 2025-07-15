# scGPT-peeling-test

本项目旨在于研究 `scGPT` 模型在细胞类型分类任务中的**输入表征的消融研究**以及**关键超参数的系统性灵敏度分析**。


## 1. 输入表征消融研究

通过修改模型中的 `total_embs` 变量，以确定**基因身份（src）**与**基因表达水平（values）**对细胞类型分类任务的相对贡献。

**修改方法**：
在model/model.py代码中调整 `total_embs` 变量，以探索不同的输入表征设置。

---

## 2. 关键超参数系统性灵敏度分析

对以下关键超参数进行了系统性灵敏度分析，以评估它们对模型性能的影响：

### a. 零表达基因的影响

分析了包含或排除零表达基因对模型性能的影响。

**修改方法**：
在tutorials/Tutorial_Annotation.py中设置 `include_zero_gene` 变量（布尔值: `True` / `False`）。

### b. 表达值离散化的影响

我们研究了不同表达值离散化粒度对模型性能的影响。

**修改方法**：
* 在tutorials/Tutorial_Annotation.py中调整 `n_bins` 变量（整数），以改变离散化的箱数。

### c. 输入序列长度的影响

我们评估了输入序列长度对模型性能的影响。

**修改方法**：
* 在tutorials/Tutorial_Annotation.py中修改 `max_seq_len` 变量（整数），以设置最大序列长度。

---

## 如何运行代码 

请按照以下步骤运行项目代码：

### 1. 环境准备

conda env create -f environment.yml -n scgpt

### 2. 运行脚本

打开你的终端或命令行，直接运行 `tutorials/Tutorial_Annotation.py` 脚本：

```bash
python tutorials/Tutorial_Annotation.py
