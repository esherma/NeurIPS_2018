# Identification and Estimation of Causal Effects from Dependent Data

This repository contains the code to run the experiments in [this](https://papers.nips.cc/paper/8153-identification-and-estimation-of-causal-effects-from-dependent-data) paper, published at the 31st annual conference on Advances in Neural Information Processing Systems (NeurIPS 2018). The primary novelty is the use of the Gibbs sampler-based algorithm the [Auto-G-Computation Algorithm](https://arxiv.org/abs/1709.01577).

The Auto-G-Computation algorithm is used for generating networks with hidden variables in [DataGen.py](https://github.com/esherma/NeurIPS_2018/blob/master/DataGen.py).

To generate ground truth networks (using Auto-G with hidden variables) consult [Auto-G-GT.ipynb](https://github.com/esherma/NeurIPS_2018/blob/master/Auto-G-GT.ipynb). It will be necessary to change the names of files/folders for each network size. The appropriate folders to store results (400_3/, 800_3/, etc.)



Results from the paper are stored in the [Results folder](https://github.com/esherma/NeurIPS_2018/tree/master/Results/aws/NIPS18_Camera_Ready) (analyses must be calculated using Analysis.ipynb by changing the appropriate file/folder names). The files stored in the results folder are merely the outputs of the data generating process, the ground truth effects, and the calculated effects.