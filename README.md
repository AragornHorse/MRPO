# MRPO
<div align="center">

[**Wei Han**](https://scholar.google.com.hk/citations?user=67WVzncAAAAJ&hl=zh-CN),
[**Yuanxing Liu**](https://scholar.google.com.hk/citations?user=jgSM9f0AAAAJ&hl=zh-CN&oi=ao),
[**Mingda Li**](https://scholar.google.com/citations?user=k-BROPQAAAAJ&hl=zh-CN),
[**Ruiyu Xiao**](https://www.semanticscholar.org/author/Ruiyu-Xiao/1491639098),
[**Weinan Zhang**](https://scholar.google.com.hk/citations?hl=zh-CN&user=DBLdEf4AAAAJ),
[**Ting Liu**](https://scholar.google.com.hk/citations?user=zyMJ1V0AAAAJ&hl=zh-CN&oi=ao)

</div>

Training code for the ICML 2026 paper 
**MRPO: Magnitude-Regularized Policy Optimization via L1 Constraints**


## 🔖 Introduction
Reinforcement learning (RL) for large language models (LLMs) relies on imperfect reward supervision, necessitating constraints on policy updates to prevent overfitting.
Nevertheless, the widely adopted KL constraint over-penalizes actions with low reference probabilities and lacks the sparsity to discard marginal policy shifts.
In contrast, the L1-norm offers a distinct mechanism that is more tolerant of low-probability actions yet strictly suppresses minor probability perturbations.
Motivated by this, we propose Magnitude-Regularized Policy Optimization (MRPO), which enforces an L1-norm constraint on policy updates.
We demonstrate that MRPO permits substantial probability boosts for low-probability actions and induces sparse updates, ensuring invariance to noise that preserves the top-ranking order.
Furthermore, MRPO guarantees convergence in general RL settings and achieves a tighter approach to optimality than KL-based methods in single-step scenarios.
Empirically, MRPO delivers exceptional results across diverse scenarios, notably doubling the performance gains of GRPO in preference alignment, outperforming DAPO in mathematical reasoning, and surpassing DPO in offline settings using only binary rewards.

## 📑 Download packages
+ trl
+ datasets
+ peft

## 🚀Training
You can use the provided `train_mrpo_token.py` for training. Alternatively, you can follow the commands in `trl` (https://github.com/huggingface/trl) and train with our `MRTrainer` implemented in `mr_trainer_token.py`.

## Cite
```
@inproceedings{
han2026mrpo,
title={{MRPO}: Magnitude-Regularized Policy Optimization via L1 Constraints},
author={Wei Han and Yuanxing Liu and Mingda Li and Ruiyu Xiao and Weinan Zhang and Ting Liu},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=e2xQL4BWrY}
}
```
