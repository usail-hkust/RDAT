# Robust Spatiotemporal Traffic Forecasting with Reinforced Dynamic Adversarial Training



> Machine learning-based forecasting models are commonly used in Intelligent Transportation Systems (ITS) to predict traffic patterns and provide city-wide services. However, these models are susceptible to adversarial attacks, which can lead to inaccurate predictions and negative consequences such as congestion and delays. 
Improving the adversarial robustness of these models is crucial in ITS. In this paper, we propose a novel framework for incorporating adversarial training into spatiotemporal traffic forecasting tasks. 
We demonstrate that traditional adversarial training methods in static domains cannot be directly applied to traffic forecasting tasks, as they fail to effectively defend against dynamic adversarial attacks. We propose a reinforcement learning-based method to learn the optimal node selection strategy for adversarial examples, which improves defense against dynamic attacks and reduces overfitting. 
Additionally, we introduce a self-knowledge distillation regularization to overcome the "forgetting issue" caused by constantly changing adversarial nodes during training.   
We evaluate our approach on two real-world traffic datasets and demonstrate its superiority over other baselines. 
Our method effectively enhances the adversarial robustness of spatiotemporal traffic forecasting models. 

This repository includes:
- Code for the RDAT in our study.


## Environment 
* [PyTorch](https://pytorch.org/) (tested on 1.8.0)
* [mmcv](https://github.com/open-mmlab/mmcv)


## Datasets
We use the PeMS-Bay and PeMS-D4 datasets ([link](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX)). 

## Usage
### Adversarial Robustoness
To train and evaluate a baseline model, run the following commands:
```
# PeMS-BAY
# pre-train policy net
python train.py configs/PeMS/PeMS-train0.7-val0.1-test0.2-in12out12-STpgd0.1nodes0.5eps-random-none_dropout-gwnet-AT_policy_atten_epoch100-steps10-constant1e-03-start_index_learn-baseline_saliency.yaml -d 0  #device id
# AT
python train.py configs/PeMS/PeMS-train0.7-val0.1-test0.2-in12out12-STpgd0.1nodes0.5eps-random-none_dropout-gwnet-AT_policy_atten_epoch100-steps10-constant1e-03-start_index_learn-baseline_saliency-offine-reg0.4-Exps.yaml -d 0  #device id

# PeMS-D4
# pre-train policy net
python train.py configs/PeMSD4/PeMSD4-train0.7-val0.1-test0.2-in12out12-STpgd0.1nodes0.5eps-none_dropout-gwnet-AT_policy_atten_epoch100-steps30-constant1e-03-start_index_learn-baseline_saliency-offine.yaml -d 0  #device id
# AT
python train.py configs/PeMSD4/PeMSD4-train0.7-val0.1-test0.2-in12out12-STpgd0.1nodes0.5eps-random-none_dropout-gwnet-AT_policy_atten_epoch100-steps30-constant1e-03-start_index_learn-baseline_saliency-offine-dist0.4.yaml -d 0  #device id


```




## Acknowledgement
We thank the authors for the following repositories for code reference:
[Adversarial Long-Tail](https://github.com/wutong16/Adversarial_Long-Tail), etc.

