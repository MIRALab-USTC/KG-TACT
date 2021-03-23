# Topology-Aware Correlations Between Relations for Inductive Link Prediction in Knowledge Graphs

This repository contains the code and the datasets of **Topology-Aware Correlations Between Relations for Inductive Link Prediction in Knowledge Graphs**. Jiajun Chen, Huarui He, Feng Wu, Jie Wang. AAAI 2021. [[arXiv](https://arxiv.org/pdf/2103.03642.pdf)]

## Dependencies
The code is based on Python 3.7. You can use the following command to create a environment and enter it.
```shell script
conda create --name TACT-env python=3.7
source activate TACT-env
```
All the required packages can be installed by running 
```shell script
pip install -r requirements.txt
```
To test the code, run the following commands.

```shell script
cd code/AUC-PR
bash run_once.sh WN18RR_v1 8 1 2 0.01 0.01 demo_test 10 4
```

Notice that, for the first time you run the code, it would take some time to sample the subgraph. 

## Reproduce the Results

### Usage

```
bash {run_once.sh | run_five.sh} <dataset>  <gamma: margin in the loss function> \
<negative_sample_size> <enclosing_subgraph_hop_number> <learning_rate> <weight_decay> \ 
<experiment_id> <max_epoch> <gpu_id> 
```

- `{ | }`: Mutually exclusive items. Choose one from them.
- `< >`: Placeholder for which you must supply a value.

To reproduce the results, run the following commands. 

```shell script
#################################### AUC-PR ####################################
cd code/AUC-PR
bash run_five.sh WN18RR_v1 8 1 2 0.01 0.01 demo 10 0
bash run_five.sh WN18RR_v2 8 1 2 0.01 0.01 demo 10 0
bash run_five.sh WN18RR_v3 8 1 2 0.01 0.01 demo 10 0
bash run_five.sh WN18RR_v4 8 1 2 0.01 0.01 demo 10 0

bash run_five.sh fb237_v1 16 1 2 0.01 0.01 demo 10 0
bash run_five.sh fb237_v2 16 1 2 0.01 0.01 demo 10 0
bash run_five.sh fb237_v3 16 1 2 0.01 0.01 demo 10 0
bash run_five.sh fb237_v4 16 1 2 0.01 0.01 demo 10 0

bash run_five.sh nell_v1 10 1 2 0.01 0.01 demo 10 0
bash run_five.sh nell_v2 10 1 2 0.01 0.01 demo 10 0
bash run_five.sh nell_v3 10 1 2 0.01 0.01 demo 10 0
bash run_five.sh nell_v4 10 1 2 0.01 0.1 demo 10 0

#################################### Ranking #############################
cd code/Ranking
bash run_five.sh WN18RR_v1 8 8 2 0.01 0.01 demo 10 0
bash run_five.sh WN18RR_v2 8 8 2 0.01 0.01 demo 10 0
bash run_five.sh WN18RR_v3 8 8 2 0.01 0.01 demo 10 0
bash run_five.sh WN18RR_v4 8 8 2 0.01 0.01 demo 10 0

bash run_five.sh fb237_v1 16 8 2 0.005 0.01 demo 10 0
bash run_five.sh fb237_v2 16 8 2 0.005 0.01 demo 10 0
bash run_five.sh fb237_v3 16 8 2 0.005 0.01 demo 10 0
bash run_five.sh fb237_v4 16 8 2 0.005 0.01 demo 10 0

bash run_five.sh nell_v1 10 8 2 0.01 0.01 demo 10 0
bash run_five.sh nell_v2 10 8 2 0.01 0.01 demo 10 0
bash run_five.sh nell_v3 10 8 2 0.01 0.01 demo 10 0
bash run_five.sh nell_v4 16 8 2 0.008 0.01 demo 5 0
```

**Remark**:  We run each experiment five times and report the mean results.

## Citation
If you find this code useful, please consider citing the following paper.
```shell script
@article{DBLP:journals/corr/abs-2103-03642,
  author    = {Jiajun Chen and
               Huarui He and
               Feng Wu and
               Jie Wang},
  title     = {Topology-Aware Correlations Between Relations for Inductive Link Prediction
               in Knowledge Graphs},
  journal   = {CoRR},
  volume    = {abs/2103.03642},
  year      = {2021},
  url       = {https://arxiv.org/abs/2103.03642},
  archivePrefix = {arXiv},
  eprint    = {2103.03642},
  timestamp = {Mon, 15 Mar 2021 17:30:55 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2103-03642.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Acknowledgement

We refer to the code of [GraIL](https://github.com/kkteru/grail). Thanks for their contributions.
