
# Meta-Transfer PINN for Seismic Wavefield Modeling
**"Meta-Transfer Learning for Efficient Initialization of Neural Network Wavefield Solutions"**  
- Zhijun Cheng<sup>1</sup>, Xiang Wang<sup>1</sup>, Guojun Mao <sup>1</sup>, Weijian Mao<sup>2</sup> and Shijun Cheng<sup>3</sup>,

 <small><sup>1</sup> Fujian University of Technology, Fuzhou 350118, China
 
 <small><sup>2</sup> Innovation Academy for Precision Measurement Science and Technology, Chinese Academy of Sciences, Wuhan 430077, China
 
 <small><sup>3</sup> King Abdullah University of Science and Technology, Thuwal 23955-6900, Saudi Arabia
---

## 🧠 Overview

This repository provides the code and data to reproduce the results of **Meta-Transfer PINN**, a physics-informed neural network framework that accelerates seismic wavefield modeling by combining:

- **Meta-learning** (Model-Agnostic Meta-Learning, MAML)
- **Transfer learning**
- **Parameter averaging strategies**

Compared with traditional PINN and Meta-PINN, our method significantly reduces meta-training time while maintaining or even improving convergence speed.

![LOGO](https://github.com/chengzhijun-czj/MTPINN/blob/main/meta-transfer-learning.png)

---

## 📁 Project structure
This repository is organized as follows:

* :open_file_folder: **code**: python library containing routines for MTPINN;
* :open_file_folder: **dataset**: folder to store dataset.
* :open_file_folder: **trained_model**: meta-trained model for Meta-PINN and Meta-transfer PINN.

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
## 🚀 Running code 

For metatransfer-train, you can directly run:
```
python metatransfertrain.py
```

For meta-test, you can directly run:
```
python metatest.py
```
**Note:** When you run demo for meta-test, you need open the `code/metatest.py` file to specify the path for meta initialization model. Here, we provide meta-training models for Meta-transfer PINN and Meta-PINN in trained_model file, you can directly load meta-initialization model to perform meta-test.

If you need to compare with a randomly initialized network, you can set the configuration value of `args.use_meta` in the `code/metatest.py` file to `False`,
and then run:
```
python metatest.py
```


