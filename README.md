
# Meta-Transfer PINN for Seismic Wavefield Modeling
**"Meta-Transfer Learning for Efficient Initialization of Neural Network Wavefield Solutions"**  
Zhijun Cheng, Shijun Cheng, Xiang Wang, and Guojun Mao  

---

## 🧠 Overview

This repository provides the code and data to reproduce the results of **Meta-Transfer PINN**, a physics-informed neural network framework that accelerates seismic wavefield modeling by combining:

- **Meta-learning** (Model-Agnostic Meta-Learning, MAML)
- **Transfer learning**
- **Parameter averaging strategies**

Compared with traditional PINN and Meta-PINN, our method significantly reduces meta-training time while maintaining or even improving convergence speed.

![LOGO](https://github.com/chengzhijun-czj/MTPINN/blob/main/meta-transfer-learning.png)
---

# 📁 Project structure
This repository is organized as follows:

* :open_file_folder: **code**: python library containing routines for MTPINN;
* :open_file_folder: **dataset**: folder to store dataset.

---
## 🔍 Key Features

- ✅ **Fast Adaptation** to new seismic velocity models using a meta-learned initialization.
- 🧩 **Reduced Meta-Training Cost** by avoiding costly nested optimization loops.
- 🔁 **Transfer-and-Average Strategy** for lightweight, scalable learning.
- 🌍 Tested on a layer model and the overthrust models.

---

## 📖 Paper Abstract

> We propose a meta-transfer learning strategy for physics-informed neural networks (PINNs) to improve seismic wavefield modeling. By fusing transfer learning with MAML, we reduce the dual-loop computational overhead of standard meta-learning. we optimize the meta-model by performing fast gradient updating for a single velocity model on the support set, and then employing a parameter averaging strategy for multiple velocity models on the query set, and the resulting initialization is used for regular training of the new velocity model. This significantly reduces the computational cost of meta-training while preserving convergence speed. Experiments on layered, overthrust, and diverse velocity models confirm the effectiveness and efficiency of our approach.

---

