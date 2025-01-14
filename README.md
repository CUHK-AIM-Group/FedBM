# FedBM
Code for our paper: "FedBM: Stealing Knowledge from Pre-trained Language Models for Heterogeneous Federated Learning, MedIA 2025, under review".

Federated learning (FL) has shown great potential in medical image computing since it provides a decentralized learning paradigm that allows multiple clients to train a model collaboratively without privacy leakage. However, current studies have shown that data heterogeneity incurs local learning bias in classifiers and feature extractors of client models during local training, leading to the performance degradation of a federation system. To address these issues, we propose a novel framework called Federated Bias eliMinating (FedBM) to get rid of local learning bias in heterogeneous federated learning (FL), which mainly consists of two modules, i.e., Linguistic Knowledge-based Classifier Construction (LKCC) and Concept-guided Global Distribution Estimation (CGDE). Specifically, LKCC exploits class concepts, prompts and pre-trained language models (PLMs) to obtain concept embeddings. These embeddings are used to estimate the latent concept distribution of each class in the linguistic space. Based on the theoretical derivation, we can rely on these distributions to pre-construct a high-quality classifier for clients to achieve classification optimization, which is frozen to avoid classifier bias during local training. CGDE samples probabilistic concept embeddings from the latent concept distributions to learn a conditional generator to capture the input space of the global model. Three regularization terms are introduced to improve the quality and utility of the generator. The generator is shared by all clients and produces pseudo data to calibrate updates of local feature extractors. Extensive comparison experiments and ablation studies on public datasets demonstrate the superior performance of FedBM over state-of-the-arts and confirm the effectiveness of each module, respectively.

<div align=center>
<img width="800" src="imgs/framework.png" alt="FL"/>
</div>

## Running
### Dependencies
```
pip install -r requirements.txt
```
### Scripts
- [x] download [Kvasir dataset](https://drive.google.com/file/d/1fzIIiZZYnpDtetjkdQOhSDTtjBkPbFCU/view?usp=sharing) and [OCT-8](https://drive.google.com/file/d/13Mm2TybL44jC2dMCh4flz0zXGE1VYGms/view?usp=sharing) datasets and put them into the dir './data/'.
- [x]  Train DEeR framework on OCT-8 dataset.
```
python main.py --exp_name='exp1' \
    --data_dir='./data/Retinal_OCT-C8/' \
    --node_num=12 \
    --iid=0 \
    --dirichlet_alpha=0.1 \
    --local_model='CLIP' \
    --dataset='OCT' \
    --T=50 \
    --E=5 \
    --lr=0.01 \
    --num_classes=8 \
    --lora_r=8 \
    --device=0 \
    --method='LoRA' \
    --is_DP=1 \
    --C=0.1 \
    --epsilon=0.1 \
```
- [x] Train DEeR framework on Kvasir dataset.
```
python main.py --exp_name='exp1' \
    --data_dir='./data/kvasir-dataset-v2-processed-224/' \
    --node_num=12 \
    --iid=0 \
    --dirichlet_alpha=0.1 \
    --local_model='CLIP' \
    --dataset='Kvasir' \
    --T=50 \
    --E=5 \
    --lr=0.01 \
    --num_classes=8 \
    --lora_r=8 \
    --device=0 \
    --method='FedLoRA' \
    --is_DP=1 \
    --C=0.3 \
    --epsilon=6 \
```

## Contact

  Meilu Zhu (meiluzhu2-c@my.cityu.edu.hk)
