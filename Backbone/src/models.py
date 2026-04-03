
import torch
import torch.nn as nn
from modules import Encoder, LayerNorm
import numpy as np
import torch.nn.functional as F

class SASRecModel(nn.Module):

    def __init__(self, args):
        super(SASRecModel, self).__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        self.criterion = nn.BCELoss(reduction='none')
        self.apply(self.init_weights)

    # Positional Embedding
    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # model same as SASRec
    def transformer_encoder(self, input_ids, perturbed=False):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
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


class FMLPRecModel(nn.Module):

    def __init__(self, args):
        super(FMLPRecModel, self).__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        self.criterion = nn.BCELoss(reduction='none')
        self.apply(self.init_weights)

    # Positional Embedding
    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # model same as SASRec
    def transformer_encoder(self, input_ids, perturbed=False):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                perturbed,
                                                output_all_encoded_layers=True)

        sequence_output = item_encoded_layers[-1]

        return sequence_output

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()



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

    def __init__(self, args):
        super(LRURecModel, self).__init__()
        from modules import LayerNorm
        
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        # LRU blocks
        self.lru_blocks = nn.ModuleList([
            LRUBlock(args) for _ in range(args.num_hidden_layers)
        ])

        self.criterion = nn.BCELoss(reduction='none')
        self.apply(self.init_weights)

    def forward_lru(self, input_ids):
        """LRU forward with recursive parallelization"""
        # Get embeddings
        mask = (input_ids > 0)
        sequence_emb = self.item_embeddings(input_ids)
        sequence_emb = self.LayerNorm(self.dropout(sequence_emb))
        
        # Left padding to the power of 2 for recursive parallelization
        seq_len = sequence_emb.size(1)
        log2_L = int(np.ceil(np.log2(seq_len)))
        padded_len = 2 ** log2_L
        
        if padded_len > seq_len:
            sequence_emb = F.pad(sequence_emb, (0, 0, padded_len - seq_len, 0, 0, 0))
            mask = F.pad(mask, (padded_len - seq_len, 0, 0, 0))
        
        # LRU blocks with PFFN
        x = sequence_emb
        for lru_block in self.lru_blocks:
            x = lru_block(x, mask)
        
        # Remove padding
        sequence_output = x[:, -seq_len:]
        sequence_output = sequence_output.contiguous()
        
        return sequence_output

    def transformer_encoder(self, input_ids):
        """Interface compatibility method for SASRecTrainer"""
        return self.forward_lru(input_ids)

    def init_weights(self, module):
        """Initialize the weights with truncated normal for LRU"""
        import math
        from modules import LayerNorm
        
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if hasattr(module.weight, 'is_complex') and module.weight.is_complex():
                # Complex parameter initialization
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
                module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()