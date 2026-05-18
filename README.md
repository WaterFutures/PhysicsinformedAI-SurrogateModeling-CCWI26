# A Practical Introduction to Physics-informed Deep Learning for Surrogate Models of Drinking Water Systems

Short course (tutorial) presented at WDSA/CCWI 2026, Paphos, Cyprus.

[Slides](slides/)

### Abstract

Rapid urban population growth, among others, makes drinking water system analysis an
increasingly challenging task. Often, simulators such as EPANET are used in various tasks, such
as planning, state estimation, event diagnosis, etc. However, such simulators are
computationally expensive and scale poorly in time and space, challenging the development of
digital twins for improved system analysis. In this context, recent work highlights the potential of
AI methods for building efficient surrogate models of the simulator that can be used as part of a
digital twin.
In this training course, we provide the participants with the necessary knowledge of using
physics-informed Deep Learning to develop high-fidelity surrogate models. The course will
include many hands-on examples in Python and be split into three parts.
1. Recap of Machine Learning (ML) basics and foundations of (physics-informed) Deep Learning.
A special focus will be on pre-processing strategies, evaluation metrics and procedures,
incorporation of different types of physics, and hyper-parameter tuning.
2. Data generation for training and evaluating ML models, such as deep neural networks. We will
discuss and demonstrate the selection and creation of appropriate benchmarks for building and
evaluating the ML models. Most importantly, we will discuss scenario generation in Python,
including how to break correlations and incorporate parameter uncertainties to ensure high
fidelity of the final ML models.
3. A case study on a physics-informed Graph Neural Network (GNN) based hydraulic surrogate
model. After an introduction to Graph Neural Networks (GNNs), we present and explain a
hydraulic surrogate model based on physics-informed GNNs. In this context, we will discuss and
demonstrate how such a surrogate model can be used in a digital twin for a variety of different
tasks, such as state estimation from sparse sensor readings, and network rehabilitation.

### Hands-On Sessions

1. Physics-Informed Neural Networks [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WaterFutures/PhysicsinformedAI-SurrogateModeling-CCWI26/blob/main/PINNs/1.+PINN%3A+Water+Quality.ipynb)
2. Physics-Informed Neural Networks for Inverse Problems [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WaterFutures/PhysicsinformedAI-SurrogateModeling-CCWI26/blob/main/PINNs/2.+PINN%3A+Inverse+Problem.ipynb)
3. Case Study [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WaterFutures/PhysicsinformedAI-SurrogateModeling-CCWI26/blob/main/case_study/case_study.ipynb)

### How to cite

If you are using / referring to these materials in your own work, please cite 

Artelt, A., Ashraf, I., Hermes, L. (2026). A Practical Introduction to Physics-informed Deep Learning for Surrogate Models of Drinking Water Systems (Short course).
*4th International Joint Conference on Water Distribution Systems Analysis and Computing and Control in the Water Industry (WDSA/CCWI 2026)*, Paphos, Cyprus (May).
