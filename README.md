# README

### 介绍


清华大学2022人工神经网络课程小组作业仓库。本仓库利用jittor框架复现了论文 **[Prefix-tuning: Optimizing continuous prompts for generation](https://arxiv.org/abs/2101.00190)** 的实验结果。仓库同时提供了pytorch和jittor版的代码。请先进入`./Pytorch`或`./Jittor`中的一个并执行下面步骤。

### Quick Start


**环境配置：**

```cpp
pip install -r requirements.txt
```

**运行finetune版：**

```cpp
bash train_finetune.sh
```

**运行prefix-tuning版：**

```cpp
bash train_prefix.sh
```

### Usage


baseline.yaml提供了以下可调参数：

```python
dataset: webnlg # 数据集:[webnlg, e2e, animal, person]
pretrained: gpt2-medium

save_ckpt: ./temp.pth
batch_size: 5
max_epoch: 5
lr: 5e-5
max_length: 256
warmup_steps: 0
output_dir: ./outputs/output.txt

non_prefix_layers: [] # 不添加前缀参数的层数，仅在prefix-tuning方法下有用
# 生成方式
decode_strategy: top-p
temperature: 1.0
top_p: 0.9
top_k: 40
```
