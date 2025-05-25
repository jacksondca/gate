import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd

from torch.utils.data import Dataset, DataLoader

from cluster.io import read_fvecs


class ContrastiveDataset(Dataset):
    def __init__(self, embadding_data, eps_data, query_data, pos_sample, neg_sample, filter_num, sample_num):
        self.filter_num = filter_num
        self.sample_num = sample_num

        self.ep_graph_features = torch.Tensor(embadding_data)
        self.ep_node_features = torch.Tensor(eps_data)
        self.query_node_features = torch.Tensor(query_data)

        self.random_graph_features = torch.randn_like(self.ep_graph_features)
        self.random_node_features = torch.randn_like(self.ep_node_features)
        
        # pairing
        query2ep_pos = pos_sample
        query2ep_neg = neg_sample

        ep2query_pos = [[] for _ in range(len(embadding_data))]
        ep2query_neg = [[] for _ in range(len(embadding_data))]

        dummy_query_ids = []

        for query_id, (poses, negs) in enumerate(zip(pos_sample, neg_sample)):
            if len(negs) > self.filter_num or len(negs) < 10:
                dummy_query_ids.append(query_id)
                continue

            for ep_id in poses:
                ep2query_pos[ep_id].append(query_id)


        for query_id, negs in enumerate(neg_sample):
            for ep_id in negs:
                if ep_id in dummy_query_ids:
                    continue

                ep2query_neg[ep_id].append(query_id)
        
        self.ep2query_neg = ep2query_neg
        self.ep2query_pos = ep2query_pos

        
        pos_pairs = []
        for ep_id, poses in enumerate(ep2query_pos):
            for pos in poses:
                pos_pairs.append((ep_id, pos))
        self.pos_pairs = pos_pairs

        pairs = []
        for ep_id, (poses, negs) in enumerate(zip(ep2query_pos, ep2query_neg)):
            random.shuffle(negs)
            random.shuffle(poses)
            negs = negs[: self.sample_num]
            poses = poses[: self.sample_num]
            for neg in negs:
                for pos in poses:
                    pairs.append((ep_id, pos, neg))

        self.pairs = pairs

    def __len__(self):
        # return len(self.pos_pairs)
        return len(self.pairs)

    def __getitem__(self, idx):
        ep_id, pos_query_id, neg_query_id = self.pairs[idx]

        graph_feature = self.ep_graph_features[ep_id]
        node_feature = self.ep_node_features[ep_id]

        random_graph_feature = self.random_graph_features[ep_id]
        random_node_feature = self.random_node_features[ep_id]

        pos_feature = self.query_node_features[pos_query_id]
        neg_feature = self.query_node_features[neg_query_id]

        return node_feature, graph_feature, pos_feature, neg_feature


def get_dataset(params_dict):
    x_data = pd.read_csv(params_dict['EMBADDING_FILE_PATH'], index_col=0)
    embadding_data = [x_data.loc[i].tolist() for i in range(len(x_data))]

    eps_data = read_fvecs(params_dict['EP_FILE_PATH']).tolist()

    query_data = read_fvecs(params_dict['QUERY_FILE_PATH']).tolist()

    def read_sample(file):
        with open(file, 'r') as fp:
            sample = []
            for line in fp.readlines():
                sample.append(list(map(int, line.split())))
            return sample
    
    pos_sample = read_sample(params_dict['POS_SAMPLE_PATH'])
    neg_sample = read_sample(params_dict['NEG_SAMPLE_PATH'])

    return ContrastiveDataset(embadding_data, eps_data, query_data, pos_sample, neg_sample, params_dict['FILTER_NUM'], params_dict['SAMPLE_NUM'])
