# Reinforced agglomerative clustering
To overcome the greediness of traditional linkage criteria in agglomerative clustering, 
we proposed a reinforcement learning approach to learn a non-greedy merge policy by 
modeling agglomerative clustering as Markov Decision Process.


[Agglomerative clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering) is a "bottom up" approach of hierarchical clustering, 
where each observation starts in its own cluster, and pairs of clusters 
are merged as one moves up the hierarchy. Agglomerative clustering is a sequential decision problem, which comes with
the problem that a decision made earlier affects the later result. But traditional linkage criteria fail to handle
this problem by simply measuring similarity of clusters in current phase. This motivated us to model the clustering
as Markov Decision Process and solve it with reinforcement learning. The agent should learn a non-greedy merge policy
so that each merge operation is chosen for a better long term discounted reward.

The state is defined as feature representation of current clustering. We use pooling to aggregate the feature of all 
clusters. The action is defined as merging cluster i and cluster j. We use Q-learning to compute the value of a
state-action pair. In training, the reward is computed by the ground truth label of images. And at test time,
we test the agent in a different domain to see how it can generalize.


## Installation
1. Download mnist dataset
```
cd dataset/ & bash download_data.sh
```
2. Install all the dependencies 
```
pip install -r requirements.txt
```

## Usage
1. Train
```
python main.py --train
```
2. Test
```
python main.py --test [MODEL_DIR]
```