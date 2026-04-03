
import os
import numpy as np
import pickle
import torch
from sklearn.decomposition import PCA
import torch.nn as nn 
from math import sqrt
import torch.nn.functional as F


class LLMEmbeddingMapper(nn.Module):
    def __init__(self, llm_dim, hidden_size, dropout_prob=0.2):
        super().__init__()
        self.proj1 = nn.Linear(llm_dim, llm_dim // 2)
        self.proj2 = nn.Linear(llm_dim // 2, hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, llm_emb):
        mapped = self.proj1(llm_emb)
        mapped = self.act(mapped)
        mapped = self.dropout(mapped)
        mapped = self.proj2(mapped)
        return mapped

#Item-Level Alignment
class ItemContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction='none')  

    def forward(self, id_embeddings, llm_embeddings, item_ids,t=None):
        if t is not None:
            temperature=t
        else:
            temperature=self.temperature

        id_emb = F.normalize(id_embeddings, dim=1)
        llm_emb = F.normalize(llm_embeddings, dim=1)
        
        sim_matrix = torch.matmul(id_emb, llm_emb.T) / temperature
        N = sim_matrix.size(0)
        
        same_item_mask = (item_ids.unsqueeze(0) == item_ids.unsqueeze(1)).float()
        same_item_mask = same_item_mask - torch.eye(N, device=sim_matrix.device)
        sim_matrix = sim_matrix.masked_fill(same_item_mask.bool(), float('-inf'))
        
        labels = torch.arange(N, device=sim_matrix.device)
        loss_per_sample = self.criterion(sim_matrix, labels)  # [N]
        
        
        return loss_per_sample.mean()

#Feature-Level Alignment
class BTLoss(nn.Module):

    def __init__(self, gamma=0.01):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, id_embeddings, llm_embeddings):
        N, D = id_embeddings.shape
        
        id_norm = (id_embeddings - id_embeddings.mean(0)) / (id_embeddings.std(0) + 1e-5)
        llm_norm = (llm_embeddings - llm_embeddings.mean(0)) / (llm_embeddings.std(0) + 1e-5)
        
        c = torch.mm(id_norm.T, llm_norm) / N
        
        c_diff = (c - torch.eye(D, device=c.device)).pow(2)
        
        on_diag = torch.diagonal(c_diff).sum()
        off_diag = c_diff.sum() - on_diag
        
        loss = on_diag + self.gamma * off_diag
        
        return loss

# load LLM Embeddings
def llm_embeddings(args):
    data_dir = os.path.join("llm_emb", args.data_name)
    itm_emb_path = os.path.join(data_dir, "itm_emb_np.npy")
    

    llm_item_emb = np.load(itm_emb_path)
    emb_dim = llm_item_emb.shape[1]
    
    llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, emb_dim)), axis=0)
    llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, emb_dim))], axis=0)
    
    device = torch.device("cuda" if args.cuda_condition else "cpu")
    llm_item_emb = torch.tensor(llm_item_emb, dtype=torch.float32).to(device)
    
    return llm_item_emb

# Principal Component Analysis (PCA) for Dimension Reduction
def llm_embeddings_pca(args, n_components=None):
    if n_components is None:
        n_components = args.hidden_size
    
    save_dir = os.path.join("llm_emb", args.data_name)
    pca_item_path = os.path.join(save_dir, "pca_itm_emb_np.npy")
    
    if os.path.exists(pca_item_path) :
        pca_item_emb_np = np.load(pca_item_path)
        
        device = torch.device("cuda" if args.cuda_condition and torch.cuda.is_available() else "cpu")
        
        pca_item_emb = torch.tensor(pca_item_emb_np, dtype=torch.float32).to(device)
        
        print(f"PCA embeddings loaded from file: Item embeddings: {pca_item_path}\n")
        return pca_item_emb
    
    llm_item_emb= llm_embeddings(args)
    
    device = llm_item_emb.device
    item_emb_np = llm_item_emb.cpu().numpy()
    
    pca_item = PCA(n_components=n_components, random_state=args.seed)
    
    pca_item_emb_np = pca_item.fit_transform(item_emb_np)
    
    os.makedirs(save_dir, exist_ok=True)
    np.save(pca_item_path, pca_item_emb_np)
    
    pca_item_emb = torch.tensor(pca_item_emb_np, dtype=torch.float32).to(device)
    
    return pca_item_emb