
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import Adam
from utils import recall_at_k, ndcg_k, get_metric
import scipy.sparse as sp


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
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        self.optim = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
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
        self.model.load_state_dict(torch.load(file_name, map_location=device))

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

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
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

            for i, rec_batch, in rec_cf_data_iter:
                '''
                rec_batch shape: key_name x batch_size x feature_dim
                cl_batches shape: 
                    list of n_views x batch_size x feature_dim tensors
                '''
                # 0. batch_data will be sent into the device(GPU or CPU)
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_ids, target_pos, target_neg, _ = rec_batch

                # ---------- recommendation task ---------------#
                sequence_output = self.model.transformer_encoder(input_ids)

                rec_loss = self.cross_entropy(sequence_output, target_pos, target_neg,input_ids)
                joint_loss = rec_loss
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
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch

                    if answers.dim() == 2:
                        valid_mask = (answers > 0).any(dim=1)  
                    else:
                        valid_mask = answers > 0  
                    
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
                    
                    rating_pred = self.predict_full(recommend_output)   

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