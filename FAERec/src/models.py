# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from modules import Encoder, LayerNorm
from llm import llm_embeddings,llm_embeddings_pca
from llm import LLMEmbeddingMapper
import torch.distributions as dist
import numpy as np
import torch.nn.functional as F


class SASRecModel(nn.Module):

    def __init__(self, args):
        super(SASRecModel, self).__init__()
        self.device = torch.device("cuda" if args.cuda_condition else "cpu")
        
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.args = args
        self.gate_type = self.args.gate_type
        self.criterion = nn.BCELoss(reduction='none').to(self.device)
        self.apply(self.init_weights)
        self.pca_item_emb=llm_embeddings_pca(args)
        self._init_all_items_with_pca(args)
        self._init_llm_embeddings(args)

    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_embeddings(position_ids)

        id_seqs = self.item_embeddings(sequence)
        id_seqs += position_embeddings
        id_seqs = self.dropout(id_seqs)
        
        llm_seqs = self.llm_item_emb[sequence]
        llm_seqs = self.llm_mapper(llm_seqs)
        llm_seqs += position_embeddings
        llm_seqs = self.dropout(llm_seqs)

        sequence_emb = self._dimension_gating(id_seqs, llm_seqs)

        sequence_emb = self.LayerNorm(sequence_emb)
        return sequence_emb

    # Adaptive gate fusion
    def _dimension_gating(self, id_seqs, llm_seqs):
        concat_emb = torch.cat([id_seqs, llm_seqs], dim=-1)
        gate = self.gate_network(concat_emb)
        sequence_emb = gate * id_seqs + (1 - gate) * llm_seqs
        return sequence_emb

    def transformer_encoder(self, input_ids, perturbed=False):
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                perturbed,
                                                output_all_encoded_layers=True)

        sequence_output = item_encoded_layers[-1]
        return sequence_output

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _init_all_items_with_pca(self, args):
        all_items = torch.arange(1, args.item_size, dtype=torch.long, device=self.device)
        valid_mask = all_items < self.pca_item_emb.size(0)
        valid_items = all_items[valid_mask]
        pca_emb_to_use = self.pca_item_emb[valid_items].to(self.item_embeddings.weight.device)
        with torch.no_grad():
            self.item_embeddings.weight[valid_items] = pca_emb_to_use
        print(f"Initialized {valid_items.numel()} items with PCA embeddings (total {args.item_size} items)")
        
    def _init_llm_embeddings(self, args):
        self.llm_item_emb = llm_embeddings(args)
        self.llm_dim = self.llm_item_emb.shape[1]
        
        self.llm_mapper = LLMEmbeddingMapper(llm_dim=self.llm_dim, hidden_size=args.hidden_size,
                                             dropout_prob=0.3).to(self.device)
        self.llm_item_emb = self.llm_item_emb.to(self.device)

        self.gate_network = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_size*2, args.hidden_size),
            nn.Sigmoid()
        ).to(self.device)
        
        with torch.no_grad():
            if hasattr(self.gate_network[-2], 'bias'):
                self.gate_network[-2].bias.zero_()

class FMLPRecModel(nn.Module):

    def __init__(self, args):
        super(FMLPRecModel, self).__init__()
        self.device = torch.device("cuda" if args.cuda_condition else "cpu")
        print(f"模型使用设备: {self.device}")
        
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.args = args
        self.gate_type = self.args.gate_type
        self.criterion = nn.BCELoss(reduction='none').to(self.device)
        self.apply(self.init_weights)
        self.pca_item_emb=llm_embeddings_pca(args)
        self._init_all_items_with_pca(args)
        self._init_llm_embeddings(args)

    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_embeddings(position_ids)

        id_seqs = self.item_embeddings(sequence)
        id_seqs += position_embeddings
        id_seqs = self.dropout(id_seqs)
        
        llm_seqs = self.llm_item_emb[sequence]
        llm_seqs = self.llm_mapper(llm_seqs)
        llm_seqs += position_embeddings
        llm_seqs = self.dropout(llm_seqs)

        sequence_emb = self._dimension_gating(id_seqs, llm_seqs)

        sequence_emb = self.LayerNorm(sequence_emb)
        return sequence_emb

    def _dimension_gating(self, id_seqs, llm_seqs):
        concat_emb = torch.cat([id_seqs, llm_seqs], dim=-1)
        gate = self.gate_network(concat_emb)
        sequence_emb = gate * id_seqs + (1 - gate) * llm_seqs
        return sequence_emb

    def transformer_encoder(self, input_ids, perturbed=False):
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                perturbed,
                                                output_all_encoded_layers=True)

        sequence_output = item_encoded_layers[-1]
        return sequence_output

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _init_all_items_with_pca(self, args):
        all_items = torch.arange(1, args.item_size, dtype=torch.long, device=self.device)
        valid_mask = all_items < self.pca_item_emb.size(0)
        valid_items = all_items[valid_mask]
        pca_emb_to_use = self.pca_item_emb[valid_items].to(self.item_embeddings.weight.device)
        with torch.no_grad():
            self.item_embeddings.weight[valid_items] = pca_emb_to_use
        print(f"已用PCA嵌入初始化 {valid_items.numel()} 个物品（共 {args.item_size} 个物品）")
        
    def _init_llm_embeddings(self, args):
        self.llm_item_emb = llm_embeddings(args)
        self.llm_dim = self.llm_item_emb.shape[1]
        
        self.llm_mapper = LLMEmbeddingMapper(llm_dim=self.llm_dim, hidden_size=args.hidden_size,
                                             dropout_prob=0.3).to(self.device)
        self.llm_item_emb = self.llm_item_emb.to(self.device)

        print(f"使用门控类型: {self.gate_type}")
        self.gate_network = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_size*2, args.hidden_size),
            nn.Sigmoid()
        ).to(self.device)
        
        with torch.no_grad():
            if hasattr(self.gate_network[-2], 'bias'):
                self.gate_network[-2].bias.zero_()


class LRULayer(nn.Module):
    """Linear Recurrent Unit Layer"""
    def __init__(self, args, r_min=0.8, r_max=0.99):
        super().__init__()
        from modules import LayerNorm
        
        self.embed_size = args.hidden_size
        self.hidden_size = 2 * args.hidden_size
        self.use_bias = True

        # Initialize nu, theta, gamma (ring initialization)
        u1 = torch.rand(self.hidden_size)
        u2 = torch.rand(self.hidden_size)
        nu_log = torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))
        theta_log = torch.log(u2 * torch.tensor(np.pi) * 2)
        diag_lambda = torch.exp(torch.complex(-torch.exp(nu_log), torch.exp(theta_log)))
        gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))
        self.params_log = nn.Parameter(torch.vstack((nu_log, theta_log, gamma_log)))

        # Initialize B, C matrices in complex space
        self.in_proj = nn.Linear(self.embed_size, self.hidden_size, bias=self.use_bias).to(torch.cfloat)
        self.out_proj = nn.Linear(self.hidden_size, self.embed_size, bias=self.use_bias).to(torch.cfloat)
        self.out_vector = nn.Identity()
        
        # Dropout and layer norm
        self.dropout = nn.Dropout(p=args.hidden_dropout_prob)
        self.layer_norm = LayerNorm(self.embed_size)

    def lru_parallel(self, i, h, lamb, mask, B, L, D):
        """Recursive parallelization for efficient computation"""
        l = 2 ** i
        h = h.reshape(B * L // l, l, D)
        mask_ = mask.reshape(B * L // l, l)
        h1, h2 = h[:, :l // 2], h[:, l // 2:]

        if i > 1:
            lamb = torch.cat((lamb, lamb * lamb[-1]), 0)
        h2 = h2 + lamb * h1[:, -1:] * mask_[:, l // 2 - 1:l // 2].unsqueeze(-1)
        h = torch.cat([h1, h2], axis=1)
        return h, lamb

    def forward(self, x, mask):
        # Compute lambda and gamma
        nu, theta, gamma = torch.exp(self.params_log).split((1, 1, 1))
        lamb = torch.exp(torch.complex(-nu, theta))
        
        # Project input to complex space and apply gamma normalization
        h = self.in_proj(x.to(torch.cfloat)) * gamma
        
        # Recursive parallelization
        log2_L = int(np.ceil(np.log2(h.size(1))))
        B, L, D = h.size(0), h.size(1), h.size(2)
        for i in range(log2_L):
            h, lamb = self.lru_parallel(i + 1, h, lamb, mask, B, L, D)
        
        # Output projection with residual connection
        x = self.dropout(self.out_proj(h).real) + self.out_vector(x)
        return self.layer_norm(x)


class LRUBlock(nn.Module):
    """LRU Block with Position-wise Feed-Forward Network"""
    def __init__(self, args):
        super().__init__()
        from modules import Intermediate
        
        self.lru_layer = LRULayer(args)
        self.feed_forward = Intermediate(args)  # Reuse existing PFFN from SASRec
    
    def forward(self, x, mask):
        x = self.lru_layer(x, mask)
        x = self.feed_forward(x)
        return x


class LRURecModel(nn.Module):
    """
    Enhanced LRURec with:
    1. PCA-based item embedding initialization
    2. Dual embedding (ID + LLM) with dimension gating fusion
    """

    def __init__(self, args):
        super(LRURecModel, self).__init__()
        from modules import LayerNorm
        
        # Device setup
        self.device = torch.device("cuda" if args.cuda_condition else "cpu")
        print(f"模型使用设备: {self.device}")
        
        # Core components
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        # LRU blocks
        self.lru_blocks = nn.ModuleList([
            LRUBlock(args) for _ in range(args.num_hidden_layers)
        ])

        self.criterion = nn.BCELoss(reduction='none').to(self.device)
        
        # ========== Step 1: Initialize with standard method ==========
        self.apply(self.init_weights)
        
        # ========== Step 2: Override with PCA initialization ==========
        self.pca_item_emb = llm_embeddings_pca(args)
        self._init_all_items_with_pca(args)
        
        # ========== Step 3: Initialize LLM embeddings and gating ==========
        self._init_llm_embeddings(args)

    def get_sequence_embedding(self, input_ids):
        """
        Get fused sequence embeddings using dimension gating mechanism
        
        Returns:
            sequence_emb: [batch_size, seq_len, hidden_size]
        """
        # ========== ID Embedding Branch ==========
        id_seqs = self.item_embeddings(input_ids)
        id_seqs = self.dropout(id_seqs)
        
        # ========== LLM Embedding Branch ==========
        llm_seqs = self.llm_item_emb[input_ids]
        llm_seqs = self.llm_mapper(llm_seqs)
        llm_seqs = self.dropout(llm_seqs)
        
        # ========== Dimension Gating Fusion ==========
        sequence_emb = self._dimension_gating(id_seqs, llm_seqs)
        
        # Layer normalization
        sequence_emb = self.LayerNorm(sequence_emb)
        
        return sequence_emb

    def _dimension_gating(self, id_seqs, llm_seqs):
        """Dimension-wise gating: gate = σ(W·[id; llm])"""
        concat_emb = torch.cat([id_seqs, llm_seqs], dim=-1)
        gate = self.gate_network(concat_emb)
        sequence_emb = gate * id_seqs + (1 - gate) * llm_seqs
        return sequence_emb

    def forward_lru(self, input_ids):
        """LRU forward with recursive parallelization and dual embedding fusion"""
        # Get mask
        mask = (input_ids > 0)
        
        # ========== Get fused embeddings (ID + LLM with gating) ==========
        sequence_emb = self.get_sequence_embedding(input_ids)
        
        # ========== Left padding to power of 2 for recursive parallelization ==========
        seq_len = sequence_emb.size(1)
        log2_L = int(np.ceil(np.log2(seq_len)))
        padded_len = 2 ** log2_L
        
        if padded_len > seq_len:
            sequence_emb = F.pad(sequence_emb, (0, 0, padded_len - seq_len, 0, 0, 0))
            mask = F.pad(mask, (padded_len - seq_len, 0, 0, 0))
        
        # ========== LRU blocks with PFFN ==========
        x = sequence_emb
        for lru_block in self.lru_blocks:
            x = lru_block(x, mask)
        
        # ========== Remove padding ==========
        sequence_output = x[:, -seq_len:]
        sequence_output = sequence_output.contiguous()
        
        return sequence_output

    def transformer_encoder(self, input_ids):
        """Interface compatibility method for SASRecTrainer"""
        return self.forward_lru(input_ids)

    def _init_all_items_with_pca(self, args):
        """
        Initialize item embeddings using PCA-reduced LLM embeddings
        """
        all_items = torch.arange(1, args.item_size, dtype=torch.long, device=self.device)
        valid_mask = all_items < self.pca_item_emb.size(0)
        valid_items = all_items[valid_mask]
        pca_emb_to_use = self.pca_item_emb[valid_items].to(self.item_embeddings.weight.device)
        
        with torch.no_grad():
            self.item_embeddings.weight[valid_items] = pca_emb_to_use
        
        print(f"已用PCA嵌入初始化 {valid_items.numel()} 个物品（共 {args.item_size} 个物品）")

    def _init_llm_embeddings(self, args):
        """
        Initialize LLM embeddings and dimension gating network
        """
        # Load LLM embeddings
        self.llm_item_emb = llm_embeddings(args)
        self.llm_dim = self.llm_item_emb.shape[1]
        
        # LLM dimension mapper
        self.llm_mapper = LLMEmbeddingMapper(
            llm_dim=self.llm_dim, 
            hidden_size=args.hidden_size,
            dropout_prob=0.3
        ).to(self.device)
        
        self.llm_item_emb = self.llm_item_emb.to(self.device)

        # ========== Dimension Gating Network ==========
        print(f"使用门控类型: dimension gating")
        
        self.gate_network = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_size * 2, args.hidden_size),
            nn.Sigmoid()
        ).to(self.device)
        
        # Initialize bias to zero for stable training
        with torch.no_grad():
            if hasattr(self.gate_network[-2], 'bias'):
                self.gate_network[-2].bias.zero_()

    def init_weights(self, module):
        """
        Initialize the weights with truncated normal for LRU
        NOTE: LRULayer is skipped to preserve ring initialization
        """
        import math
        from modules import LayerNorm
        
        # ========== Skip LRULayer (params_log already initialized) ==========
        if isinstance(module, LRULayer):
            return
        
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if hasattr(module.weight, 'is_complex') and module.weight.is_complex():
                # Complex parameter initialization (truncated normal)
                with torch.no_grad():
                    std = self.args.initializer_range
                    mean = 0
                    lower, upper = -0.04, 0.04
                    l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
                    u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.
                    
                    module.weight.real.uniform_(2 * l - 1, 2 * u - 1)
                    module.weight.imag.uniform_(2 * l - 1, 2 * u - 1)
                    module.weight.real.erfinv_()
                    module.weight.imag.erfinv_()
                    module.weight.real.mul_(std * math.sqrt(2.))
                    module.weight.imag.mul_(std * math.sqrt(2.))
                    module.weight.real.add_(mean)
                    module.weight.imag.add_(mean)
            else:
                # Real parameter initialization (truncated normal)
                with torch.no_grad():
                    std = self.args.initializer_range
                    mean = 0
                    lower, upper = -0.04, 0.04
                    l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
                    u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.
                    
                    module.weight.uniform_(2 * l - 1, 2 * u - 1)
                    module.weight.erfinv_()
                    module.weight.mul_(std * math.sqrt(2.))
                    module.weight.add_(mean)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
