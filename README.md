# scGPT-peeling test
针对输⼊表征的消融研究，修改model中的 total_embs = value，
以确定基因⾝份（src）与基因表达⽔平（values）对细胞类型分类任务的相对贡献。
并且进行了关键超参数的系统性灵敏度分析
对于零表达基因的影响修改变量: include_zero_gene (布尔值: True / False)。
对于表达值离散化的影响修改变量：n_bins (整数)。
对于输⼊序列⻓度的影响修改变量: max_seq_len (整数)。
