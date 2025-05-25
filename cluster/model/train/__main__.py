import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from cluster.model.train.InfoNCE import InfoNCE
from cluster.model.train.model import Model
from cluster.io import read_fvecs, read_kmeans_file, write_fvecs
from cluster.model.train.utils import get_model_name
from cluster.model.train.dataset import ContrastiveDataset, get_dataset
from cluster.model.train.params import device, sift_query1k_dataset_params, deep10m_query1b_dataset_params, deep10m_query1k_dataset_params, \
                                        gist_query1k_dataset_params, \
                                        fashion_mnist_query1k_kmeans8_8_dataset_params, \
                                        tiny5m_query1k_kmeans16_32_dataset_params, \
                                        text2image10m_query1k_kmeans16_32_dataset_params, \
                                        laion3m_query1k_kmeans16_16_dataset_params, \
                                        sift_query1k_rq4_dataset_params
from cluster.model.train.infer import infer


# model
params = sift_query1k_rq4_dataset_params
MODEL_PATH = params['MODEL_PATH']


def train(model_params):
    model = Model(model_params['input_dim'], model_params['hidden_dim'], model_params['output_dim'], model_params['dropout']).to(device)

    # criterion = InfoNCE(temperature=params['temperature']).to(device)
    # criterion = nn.TripletMarginLoss(margin=100, p=2.0).to(device)
    criterion = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y)).to(device)
    # criterion = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: -torch.sum(x * y, dim=1)).to(device)
    optimizer = optim.Adam(list(model.parameters()), lr=model_params['lr'])

    print(f'{model_params=}')
    dataset = get_dataset(params)

    dataloader = DataLoader(dataset, batch_size=model_params['batch_size'], shuffle=True)
    
    max_acc, max_opt_hop_count = -1e9, -1e9
    for epoch in tqdm(range(model_params['epochs']), desc='traning'):
        model.train()

        running_loss = 0.0
        for node_feature_batch, graph_feature_batch, pos_batch, neg_batch in tqdm(dataloader):
            node_feature_batch = node_feature_batch.to(device)
            graph_feature_batch = graph_feature_batch.to(device)
            pos_batch = pos_batch.to(device)
            neg_batch = neg_batch.to(device)

            embaddings = model(node_feature_batch, graph_feature_batch)

            loss = criterion(embaddings, pos_batch, neg_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        cur_acc, _, _, _, opt_hop_count = infer(params, model)

        max_opt_hop_count = max(max_opt_hop_count, opt_hop_count)
        if max_acc < cur_acc:
            max_acc = cur_acc
            model_path = f"{MODEL_PATH}/{get_model_name(model_params)}.pth"
            torch.save(model, model_path)
        
        print(f"\nEpoch [{epoch+1}/{model_params['epochs']}], Loss: {avg_loss:.4f}, Acc: {cur_acc:.4f}, Max Acc: {max_acc:.4f}, opt: {opt_hop_count}, Max Opt hop-count: {max_opt_hop_count}")


if __name__ == '__main__':
    params_path = params['JSON_PATH']
    with open(params_path, 'r') as f:
        model_params = json.load(f)
    train(model_params)
    