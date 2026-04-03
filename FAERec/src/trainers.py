
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import Adam
from utils import recall_at_k, ndcg_k, get_metric
import torch.distributions as dist
from llm import ItemContrastiveLoss,BTLoss
import torch.nn.functional as F
import math
import torch.nn as nn

class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader,
                 args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model

        if self.cuda_condition:
            self.model.cuda()
        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.item_contrast_loss = ItemContrastiveLoss(temperature=self.args.item_tou).to(self.device)

        if not self.args.do_eval:
            #The popularity here is derived solely from the training set, preventing data leakage.
            self.item_popularity = torch.ones(args.item_size, device=self.device, dtype=torch.float32)
            for item_id, count in args.item_popularity.items():
                if item_id < args.item_size:
                    self.item_popularity[item_id] = count

            self.tail_items_array = np.array(list(args.tail_items), dtype=np.int64)

        self.bt_loss = BTLoss(gamma=0.005).to(self.device)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort=full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort=full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_full_sort_score(self, epoch, answers, pred_list, user_list):
        k_list = [5, 10, 20]
        recall, ndcg = [], []
        for k in k_list:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))

        scores = [
            recall[0], ndcg[0],    
            recall[1], ndcg[1],   
            recall[2], ndcg[2],   
        ]

        def recallk_ndcgk(mask, k):
            if mask.sum() == 0:
                return 0.0, 0.0
            sub_ans = answers[mask]
            sub_rec = pred_list[mask]
            return recall_at_k(sub_ans, sub_rec, k), ndcg_k(sub_ans, sub_rec, k)

        item_answers = np.array([
            item
            for ans_list in answers
            for item in ans_list
        ])
        head_item_mask = np.array([item in self.args.head_items for item in item_answers])
        tail_item_mask = ~head_item_mask

        recall_hi5, ndcg_hi5 = recallk_ndcgk(head_item_mask, 5)
        recall_hi10, ndcg_hi10 = recallk_ndcgk(head_item_mask, 10)
        recall_hi20, ndcg_hi20 = recallk_ndcgk(head_item_mask, 20)

        recall_ti5, ndcg_ti5 = recallk_ndcgk(tail_item_mask, 5)
        recall_ti10, ndcg_ti10 = recallk_ndcgk(tail_item_mask, 10)
        recall_ti20, ndcg_ti20 = recallk_ndcgk(tail_item_mask, 20)

        post_fix_k5 = {
            "Epoch": epoch,
            "Overall_HIT@5": f"{recall[0]:.4f}", "Overall_NDCG@5": f"{ndcg[0]:.4f}",
            "Head_Item_HIT@5": f"{recall_hi5:.4f}", "Head_Item_NDCG@5": f"{ndcg_hi5:.4f}",
            "Tail_Item_HIT@5": f"{recall_ti5:.4f}", "Tail_Item_NDCG@5": f"{ndcg_ti5:.4f}",
        }
        
        post_fix_k10 = {
            "Epoch": epoch,
            "Overall_HIT@10": f"{recall[1]:.4f}", "Overall_NDCG@10": f"{ndcg[1]:.4f}",
            "Head_Item_HIT@10": f"{recall_hi10:.4f}", "Head_Item_NDCG@10": f"{ndcg_hi10:.4f}",
            "Tail_Item_HIT@10": f"{recall_ti10:.4f}", "Tail_Item_NDCG@10": f"{ndcg_ti10:.4f}",
        }
        
        post_fix_k20 = {
            "Epoch": epoch,
            "Overall_HIT@20": f"{recall[2]:.4f}", "Overall_NDCG@20": f"{ndcg[2]:.4f}",
            "Head_Item_HIT@20": f"{recall_hi20:.4f}", "Head_Item_NDCG@20": f"{ndcg_hi20:.4f}",
            "Tail_Item_HIT@20": f"{recall_ti20:.4f}", "Tail_Item_NDCG@20": f"{ndcg_ti20:.4f}",
        }
        # metrics for group analysis
        if self.args.print_interaction_metrics:

            item_interaction_count = self.args.item_cnt
            ranges = [
                (1, 4, "1-4"),
                (5, 9, "5-9"),
                (10, 19, "10-19"),
                (20, 49, "20-49"),
                (50, float('inf'), "50+")
            ]
            
            item_interaction_metrics = {}
            for min_count, max_count, label in ranges:
                range_mask = np.array([
                    min_count <= item_interaction_count.get(item, 0) <= max_count 
                    for item in item_answers
                ])
                recall_10, ndcg_10 = recallk_ndcgk(range_mask, 10)
                recall_20, ndcg_20 = recallk_ndcgk(range_mask, 20)
                item_interaction_metrics[label] = {
                    'recall@10': recall_10, 'ndcg@10': ndcg_10,
                    'recall@20': recall_20, 'ndcg@20': ndcg_20
                }


            k10_lines = [f"Epoch: {epoch}"]  
            for label, metrics in item_interaction_metrics.items():
                k10_lines.append(
                    f"  Item_{label}_HIT@10: {metrics['recall@10']:.4f}, Item_{label}_NDCG@10: {metrics['ndcg@10']:.4f}"
                )
            post_fix_item_ranges_k10 = "\n".join(k10_lines)

            k20_lines = [f"Epoch: {epoch}"]  
            for label, metrics in item_interaction_metrics.items():
                k20_lines.append(
                    f"  Item_{label}_HIT@20: {metrics['recall@20']:.4f}, Item_{label}_NDCG@20: {metrics['ndcg@20']:.4f}"
                )
            post_fix_item_ranges_k20 = "\n".join(k20_lines)

        post_fix = f"{str(post_fix_k5)}\n{str(post_fix_k10)}\n{str(post_fix_k20)}"

        if self.args.print_interaction_metrics:
            post_fix += f"\n\n@10 :\n{post_fix_item_ranges_k10}\n\n@20 :\n{post_fix_item_ranges_k20}"

        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(post_fix + '\n')
        
        return scores, post_fix
        

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)
    
    def load(self, file_name):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        saved_state_dict = torch.load(file_name, map_location=device)
        current_state_dict = self.model.state_dict()
        
        filtered_state_dict = {
            k: v for k, v in saved_state_dict.items() 
            if k in current_state_dict and v.shape == current_state_dict[k].shape
        }
        
        missing_keys = [k for k in current_state_dict if k not in filtered_state_dict]
        unexpected_keys = [k for k in saved_state_dict if k not in filtered_state_dict]
        print(f"Missing keys ignored when loading weights: {missing_keys}")
        print(f"Unexpected keys ignored when loading weights: {unexpected_keys}")
        
        current_state_dict.update(filtered_state_dict)
        self.model.load_state_dict(current_state_dict)

    def cross_entropy(self, seq_out, pos_ids, neg_ids, input_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size)
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        
        valid_samples = (input_ids.sum(dim=1) > 0).float()  # [batch]
        
        valid_samples = valid_samples.unsqueeze(1).expand(-1, seq_len).contiguous().view(-1)
        istarget = (pos_ids > 0).view(batch_size * seq_len).float()
        
        final_mask = istarget * valid_samples
        sum_mask = torch.sum(final_mask)
        
        if sum_mask > 0:
            loss = torch.sum(
                - torch.log(torch.sigmoid(pos_logits) + 1e-24) * final_mask -
                torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * final_mask
            ) / sum_mask
        else:
            loss = torch.tensor(0.0, device=seq_out.device, requires_grad=True)
        
        return loss
    
    def predict_full(self, seq_out):
        test_item_emb = self.model.item_embeddings.weight
        item_num = test_item_emb.size(0)
        
        llm_emb = self.model.llm_item_emb[:]
        llm_emb = self.model.llm_mapper(llm_emb)
        
        test_item_emb = test_item_emb + llm_emb
        test_item_emb=self.model.LayerNorm(test_item_emb)
        
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class SASRecTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader,
                 args):
        super(SASRecTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            args
        )

    def iteration(self, epoch, dataloader, full_sort=True, train=True):

        str_code = "train" if train else "valid"
        if train:

            self.model.train()
            rec_avg_loss = 0.0
            rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            self.epoch=epoch
            for i, (rec_batch) in rec_cf_data_iter:
                '''
                rec_batch shape: key_name x batch_size x feature_dim
                cl_batches shape: 
                    list of n_views x batch_size x feature_dim tensors
                '''
                # 0. batch_data will be sent into the device(GPU or CPU)
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                user_ids, input_ids, target_pos, target_neg, _ = rec_batch

                batch_size = input_ids.size(0)

                # ---------- recommendation task ---------------#
                sequence_output = self.model.transformer_encoder(input_ids)

                rec_loss = self.cross_entropy(sequence_output, target_pos, target_neg,input_ids)
                joint_loss = rec_loss

                # combine items in a batch
                combined_items = torch.cat([input_ids, target_neg], dim=1)  # [batch_size, seq_len + resample_num]
                
                # Curriculum Learning Scheduler
                item_weight, fcl_weight = self.get_curriculum_weights(epoch,self.args.warmup_epochs,self.args.period,self.args.w_min,self.args.w_max)

                # Dual-level Alignment
                item_cl = self.cl_loss(combined_items)
                fcl = self.fcl_loss(combined_items)

                joint_loss += self.args.cl_weight*(item_weight * item_cl + fcl_weight * fcl)
                
                self.optim.zero_grad()
                joint_loss.backward()
                self.optim.step()
                rec_avg_loss += rec_loss.item()

            post_fix = {
            "epoch": epoch,
            "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_cf_data_iter)),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')
        else:
            rec_data_iter = tqdm(enumerate(dataloader),
                                 desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                 total=len(dataloader),
                                 bar_format="{l_bar}{r_bar}")
            self.model.eval()
            pred_list = None
            if full_sort:
                answer_list = None
                pred_list_batches = []
                answer_list_batches = []
                user_list_batches = []
                
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch

                    if answers.dim() == 2:
                        valid_mask = (answers > 0).any(dim=1)  # [batch_size]
                    else:
                        valid_mask = answers > 0  # [batch_size]
                    
                    if not valid_mask.any():
                        print('This batch consists entirely of short sequences and is not evaluated.')
                        continue
                    
                    user_ids = user_ids[valid_mask]
                    input_ids = input_ids[valid_mask]
                    target_pos = target_pos[valid_mask]
                    target_neg = target_neg[valid_mask]
                    answers = answers[valid_mask]

                    recommend_output = self.model.transformer_encoder(input_ids)
                    recommend_output = recommend_output[:, -1, :]
                    
                    rating_pred = self.predict_full(recommend_output)  # [batch_size, item_size]

                    batch_user_index = user_ids.cpu().numpy()
                    train_matrix_batch = self.args.train_matrix[batch_user_index].tocoo()
                    
                    if train_matrix_batch.nnz > 0:  
                        rows = torch.from_numpy(train_matrix_batch.row).long().to(self.device)
                        cols = torch.from_numpy(train_matrix_batch.col).long().to(self.device)
                        rating_pred[rows, cols] = 0


                    _, batch_pred_list = torch.topk(rating_pred, k=20, dim=1, largest=True)
                    
                    batch_pred_list = batch_pred_list.cpu().numpy()
                    answers_cpu = answers.cpu().numpy()
                    user_ids_cpu = user_ids.cpu().numpy()

                    pred_list_batches.append(batch_pred_list)
                    answer_list_batches.append(answers_cpu)
                    user_list_batches.append(user_ids_cpu)
                
                pred_list = np.concatenate(pred_list_batches, axis=0)
                answer_list = np.concatenate(answer_list_batches, axis=0)
                user_list = np.concatenate(user_list_batches, axis=0)
                
                return self.get_full_sort_score(epoch, answer_list, pred_list, user_list)

    def cl_loss(self, input_ids):
        mask = (input_ids > 0).float().view(-1, 1)
        
        id_emb = self.model.item_embeddings(input_ids)
        llm_emb = self.model.llm_item_emb[input_ids]
        llm_emb = self.model.llm_mapper(llm_emb)
        
        flat_id_emb = id_emb.view(-1, self.args.hidden_size) * mask
        flat_llm_emb = llm_emb.view(-1, self.args.hidden_size) * mask
        
        valid_mask = mask.squeeze(-1).bool()
        valid_id_emb = flat_id_emb[valid_mask]
        valid_llm_emb = flat_llm_emb[valid_mask]
        valid_item_ids = input_ids.view(-1)[valid_mask]
        
        if valid_id_emb.size(0) > 0:
            return self.item_contrast_loss(valid_id_emb, valid_llm_emb, 
                                        valid_item_ids)
        else:
            return torch.tensor(0.0, device=self.device)
        
    def fcl_loss(self, input_ids):
        mask = (input_ids > 0).float().view(-1, 1)
        
        id_emb = self.model.item_embeddings(input_ids)
        llm_emb = self.model.llm_item_emb[input_ids]
        llm_emb = self.model.llm_mapper(llm_emb)
        
        flat_id_emb = id_emb.view(-1, self.args.hidden_size) * mask
        flat_llm_emb = llm_emb.view(-1, self.args.hidden_size) * mask
        
        valid_mask = mask.squeeze(-1).bool()
        valid_id_emb = flat_id_emb[valid_mask]
        valid_llm_emb = flat_llm_emb[valid_mask]
        valid_item_ids = input_ids.view(-1)[valid_mask]

        if valid_id_emb.size(0) > 1:
            item_pops = self.item_popularity[valid_item_ids]
            
            median_pop = torch.median(item_pops)
            high_pop_mask = item_pops >= median_pop
            low_pop_mask = ~high_pop_mask
            
            total_loss = torch.tensor(0.0, device=self.device)

            if high_pop_mask.sum() > 1:
                high_id_emb = valid_id_emb[high_pop_mask]
                high_llm_emb = valid_llm_emb[high_pop_mask]
                total_loss += self.bt_loss(high_id_emb, high_llm_emb)

            if low_pop_mask.sum() > 1:
                low_id_emb = valid_id_emb[low_pop_mask]
                low_llm_emb = valid_llm_emb[low_pop_mask]
                total_loss += self.bt_loss(low_id_emb, low_llm_emb)
            
            return total_loss
        else:
            return torch.tensor(0.0, device=self.device)
    
    def get_curriculum_weights(self, epoch, warmup_epochs=0, period=10,
                         weight_min=0.7, weight_max=1.0,
                         temp_min=0.05, temp_max=0.5):
        #The warmup_epochs is always set to 0.
        if epoch < warmup_epochs:
            return 1.0, 0.0
        
        relative_epoch = epoch - warmup_epochs
        
        cos_value = (1 + math.cos(2 * math.pi * relative_epoch / period)) / 2
        
        infonce_weight = (weight_max - weight_min) * cos_value + weight_min
        fcl_weight = 1.0 - infonce_weight

        
        return infonce_weight, fcl_weight