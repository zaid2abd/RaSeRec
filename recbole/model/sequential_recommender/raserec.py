# -*- coding: utf-8 -*-
# @Time    : 2025/1/1
# @Author  : Xinping Zhao
# @Email   : zhaoxinping@stu.hit.edu.cn

"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""

import torch, heapq, scipy, faiss, random
from faiss import normalize_L2
from torch import nn
import numpy as np
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, CrossMultiHeadAttention, FeedForward, activation_layer, MLPLayers
from recbole.model.loss import BPRLoss
import torch.nn.functional as F

from recbole.model.sequential_recommender.meta_alpha_gate import MetaAlphaFusionGate

class RaSeRec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(RaSeRec, self).__init__(config, dataset)

        self.len_lower_bound = config["len_lower_bound"] if "len_lower_bound" in config else -1
        self.len_upper_bound = config["len_upper_bound"] if "len_upper_bound" in config else -1
        self.len_bound_reverse = config["len_bound_reverse"] if "len_bound_reverse" in config else True
        self.nprobe = config['nprobe']
        self.dropout_rate = config['dropout_rate']
        self.topk = config['top_k']
        self.alpha = config['alpha']
        self.low_popular = 100
        self.beta = config['beta']
        self.attn_tau = config['attn_tau']
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.hidden_dropout_prob = 0.0
        self.attn_dropout_prob = 0.0
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.tau = config['tau']
        self.sim = config['sim']
        self.batch_size = config['train_batch_size']
        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
        self.aug_nce_fct = nn.CrossEntropyLoss()
        self.sem_aug_nce_fct = nn.CrossEntropyLoss()
        
      
        # إضافة Meta-α Fusion Gate
        self.meta_fusion_gate = MetaAlphaFusionGate(
            hidden_size=self.hidden_size,
            dropout_prob=self.hidden_dropout_prob
        )
        
        # إضافة optimizer منفصل للبوابة
        gate_lr = config['gate_learning_rate'] if 'gate_learning_rate' in config else 0.01
        self.gate_optimizer = torch.optim.Adam(
            self.meta_fusion_gate.parameters(),
            lr=gate_lr  # استخدام معدل تعلم مخصص للبوابة
        )
        print(f"🔧 Gate optimizer created with learning rate: {gate_lr}")

        # إضافة معلمات للتدريب التدريجي
        self.gate_warmup_epochs = config['gate_warmup_epochs'] if 'gate_warmup_epochs' in config else 5
        self.current_epoch = 0

        # إضافة معلمات لجدولة معدل التعلم
        self.gate_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.gate_optimizer, 
            mode='max', 
            factor=0.5, 
            patience=3,
            verbose=True
        )

        # بعد إنشاء meta_fusion_gate:
        print("🔍 Checking Meta-α Fusion Gate parameters:")
        for name, param in self.meta_fusion_gate.named_parameters():
            print(f"  {name}: requires_grad={param.requires_grad}, shape={param.shape}")

        # معامل تنظيم البوابة
        self.gate_regularization_weight = config['gate_reg_weight'] if 'gate_reg_weight' in config else 0.01

        # إعداد تحميل checkpoint مع التعامل مع الأجزاء الجديدة
        self.strict_loading = False

        # parameters initialization
        self.apply(self._init_weights)
        # precached knowledge
        self.dataset = dataset

    def precached_knowledge(self):
        length_threshold = 1
        seq_emb_knowledge, tar_emb_knowledge, user_id_list = None, None, None
        item_seq_all = None
        item_seq_len_all = None
        for batch_idx, interaction in enumerate(self.dataset):
            interaction = interaction.to("cuda")
            if self.len_lower_bound != -1 or self.len_upper_bound != -1:
                if self.len_lower_bound != -1 and self.len_upper_bound != -1:
                    look_up_indices = (interaction[self.ITEM_SEQ_LEN]>=self.len_lower_bound) * (interaction[self.ITEM_SEQ_LEN]<=self.len_upper_bound)
                elif self.len_upper_bound != -1:
                    look_up_indices = interaction[self.ITEM_SEQ_LEN]<self.len_upper_bound
                else:
                    look_up_indices = interaction[self.ITEM_SEQ_LEN]>self.len_lower_bound
                if self.len_bound_reverse:
                    look_up_indices = ~look_up_indices
            else:
                look_up_indices = interaction[self.ITEM_SEQ_LEN]>-1
            item_seq = interaction[self.ITEM_SEQ][look_up_indices]
            if item_seq_all==None:
                item_seq_all = item_seq
            else:
                item_seq_all = torch.cat((item_seq_all, item_seq), dim=0)
            item_seq_len = interaction[self.ITEM_SEQ_LEN][look_up_indices]
            item_seq_len_list = list(interaction[self.ITEM_SEQ_LEN][look_up_indices].detach().cpu().numpy())
            if isinstance(item_seq_len_all, list):
                item_seq_len_all.extend(item_seq_len_list)
            else:
                item_seq_len_all = item_seq_len_list
            seq_output = self.forward(item_seq, item_seq_len)
            tar_items = interaction[self.POS_ITEM_ID][look_up_indices]
            tar_items_emb = self.item_embedding(tar_items)
            user_id_cans = list(interaction[self.USER_ID][look_up_indices].detach().cpu().numpy())
            if isinstance(seq_emb_knowledge, np.ndarray):
                seq_emb_knowledge = np.concatenate((seq_emb_knowledge, seq_output.detach().cpu().numpy()), 0)
            else:
                seq_emb_knowledge = seq_output.detach().cpu().numpy()
            
            if isinstance(tar_emb_knowledge, np.ndarray):
                tar_emb_knowledge = np.concatenate((tar_emb_knowledge, tar_items_emb.detach().cpu().numpy()), 0)
            else:
                tar_emb_knowledge = tar_items_emb.detach().cpu().numpy()
            
            if isinstance(user_id_list, list):
                user_id_list.extend(user_id_cans)
            else:
                user_id_list = user_id_cans
        self.user_id_list = user_id_list
        self.item_seq_all = item_seq_all
        self.item_seq_len_all = item_seq_len_all
        self.seq_emb_knowledge = seq_emb_knowledge
        self.tar_emb_knowledge = tar_emb_knowledge
        # faiss
        d = 64  
        nlist = 128
        seq_emb_knowledge_copy = np.array(seq_emb_knowledge, copy=True)
        normalize_L2(seq_emb_knowledge_copy)
        seq_emb_quantizer = faiss.IndexFlatL2(d) 
        self.seq_emb_index = faiss.IndexIVFFlat(seq_emb_quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT) 
        self.seq_emb_index.train(seq_emb_knowledge_copy)
        self.seq_emb_index.add(seq_emb_knowledge_copy)    
        self.seq_emb_index.nprobe=self.nprobe

        tar_emb_knowledge_copy = np.array(tar_emb_knowledge, copy=True)
        normalize_L2(tar_emb_knowledge_copy)
        tar_emb_quantizer = faiss.IndexFlatL2(d) 
        self.tar_emb_index = faiss.IndexIVFFlat(tar_emb_quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT) 
        self.tar_emb_index.train(tar_emb_knowledge_copy)
        self.tar_emb_index.add(tar_emb_knowledge_copy) 
        self.tar_emb_index.nprobe=self.nprobe

    def precached_knowledge_val(self, val_dataset):
        length_threshold = 1
        seq_emb_knowledge, tar_emb_knowledge, user_id_list = None, None, None
        item_seq_len_all = None
        for batch_idx, interaction in enumerate(self.dataset):
            interaction = interaction.to("cuda")
            if self.len_lower_bound != -1 or self.len_upper_bound != -1:
                if self.len_lower_bound != -1 and self.len_upper_bound != -1:
                    look_up_indices = (interaction[self.ITEM_SEQ_LEN]>=self.len_lower_bound) * (interaction[self.ITEM_SEQ_LEN]<=self.len_upper_bound)
                elif self.len_upper_bound != -1:
                    look_up_indices = interaction[self.ITEM_SEQ_LEN]<self.len_upper_bound
                else:
                    look_up_indices = interaction[self.ITEM_SEQ_LEN]>self.len_lower_bound
                if self.len_bound_reverse:
                    look_up_indices = ~look_up_indices
            else:
                look_up_indices = interaction[self.ITEM_SEQ_LEN]>-1
            item_seq = interaction[self.ITEM_SEQ][look_up_indices]
            item_seq_len = interaction[self.ITEM_SEQ_LEN][look_up_indices]
            item_seq_len_list = list(interaction[self.ITEM_SEQ_LEN][look_up_indices].detach().cpu().numpy())
            if isinstance(item_seq_len_all, list):
                item_seq_len_all.extend(item_seq_len_list)
            else:
                item_seq_len_all = item_seq_len_list
            seq_output = self.forward(item_seq, item_seq_len)
            tar_items = interaction[self.POS_ITEM_ID][look_up_indices]
            tar_items_emb = self.item_embedding(tar_items)
            user_id_cans = list(interaction[self.USER_ID][look_up_indices].detach().cpu().numpy())
            if isinstance(seq_emb_knowledge, np.ndarray):
                seq_emb_knowledge = np.concatenate((seq_emb_knowledge, seq_output.detach().cpu().numpy()), 0)
            else:
                seq_emb_knowledge = seq_output.detach().cpu().numpy()
            if isinstance(tar_emb_knowledge, np.ndarray):
                tar_emb_knowledge = np.concatenate((tar_emb_knowledge, tar_items_emb.detach().cpu().numpy()), 0)
            else:
                tar_emb_knowledge = tar_items_emb.detach().cpu().numpy()
            if isinstance(user_id_list, list):
                user_id_list.extend(user_id_cans)
            else:
                user_id_list = user_id_cans
        length_threshold = 1
        for batch_idx, batched_data in enumerate(val_dataset):
            interaction, history_index, swap_row, swap_col_after, swap_col_before = batched_data
            interaction = interaction.to("cuda")
            item_seq = interaction[self.ITEM_SEQ]
            item_seq_len = interaction[self.ITEM_SEQ_LEN]
            item_seq_len_list = list(interaction[self.ITEM_SEQ_LEN].detach().cpu().numpy())
            if isinstance(item_seq_len_all, list):
                item_seq_len_all.extend(item_seq_len_list)
            else:
                item_seq_len_all = item_seq_len_list

            seq_output = self.forward(item_seq, item_seq_len)
            tar_items = interaction[self.POS_ITEM_ID]
            tar_items_emb = self.item_embedding(tar_items)
            user_id_cans = list(interaction[self.USER_ID].detach().cpu().numpy())
            if isinstance(seq_emb_knowledge, np.ndarray):
                seq_emb_knowledge = np.concatenate((seq_emb_knowledge, seq_output.detach().cpu().numpy()), 0)
            else:
                seq_emb_knowledge = seq_output.detach().cpu().numpy()
            if isinstance(tar_emb_knowledge, np.ndarray):
                tar_emb_knowledge = np.concatenate((tar_emb_knowledge, tar_items_emb.detach().cpu().numpy()), 0)
            else:
                tar_emb_knowledge = tar_items_emb.detach().cpu().numpy()
            if isinstance(user_id_list, list):
                user_id_list.extend(user_id_cans)
            else:
                user_id_list = user_id_cans
        self.user_id_list = user_id_list
        self.item_seq_len_all = item_seq_len_all

        self.seq_emb_knowledge = seq_emb_knowledge
        self.tar_emb_knowledge = tar_emb_knowledge
        # faiss
        d = 64  
        nlist = 128
        seq_emb_knowledge_copy = np.array(seq_emb_knowledge, copy=True)
        normalize_L2(seq_emb_knowledge_copy)
        seq_emb_quantizer = faiss.IndexFlatL2(d) 
        self.seq_emb_index = faiss.IndexIVFFlat(seq_emb_quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT) 
        self.seq_emb_index.train(seq_emb_knowledge_copy)
        self.seq_emb_index.add(seq_emb_knowledge_copy)    
        self.seq_emb_index.nprobe=self.nprobe

        tar_emb_knowledge_copy = np.array(tar_emb_knowledge, copy=True)
        normalize_L2(tar_emb_knowledge_copy)
        tar_emb_quantizer = faiss.IndexFlatL2(d) 
        self.tar_emb_index = faiss.IndexIVFFlat(tar_emb_quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT) 
        self.tar_emb_index.train(tar_emb_knowledge_copy)
        self.tar_emb_index.add(tar_emb_knowledge_copy) 
        self.tar_emb_index.nprobe=self.nprobe
    
    def presetting_ram(self):
        dropout_rate = self.dropout_rate
        n_heads = self.n_heads
        self.seq_tar_ram = CrossMultiHeadAttention(
            n_heads=n_heads,
            hidden_size=self.hidden_size,
            hidden_dropout_prob=dropout_rate,
            attn_dropout_prob=dropout_rate,
            layer_norm_eps=self.layer_norm_eps,
            attn_tau=self.attn_tau
        ).to("cuda")
        self.seq_tar_ram_1 = CrossMultiHeadAttention(
            n_heads=n_heads,
            hidden_size=self.hidden_size,
            hidden_dropout_prob=dropout_rate,
            attn_dropout_prob=dropout_rate,
            layer_norm_eps=self.layer_norm_eps,
            attn_tau=self.attn_tau
        ).to("cuda")
        self.seq_tar_ram_fnn = FeedForward(self.hidden_size, self.inner_size, dropout_rate, self.hidden_act, self.layer_norm_eps).to("cuda")


        self.tar_seq_ram = CrossMultiHeadAttention(
            n_heads=n_heads,
            hidden_size=self.hidden_size,
            hidden_dropout_prob=dropout_rate,
            attn_dropout_prob=dropout_rate,
            layer_norm_eps=self.layer_norm_eps,
            attn_tau=self.attn_tau
        ).to("cuda")
        self.tar_seq_ram_1 = CrossMultiHeadAttention(
            n_heads=n_heads,
            hidden_size=self.hidden_size,
            hidden_dropout_prob=dropout_rate,
            attn_dropout_prob=dropout_rate,
            layer_norm_eps=self.layer_norm_eps,
            attn_tau=self.attn_tau
        ).to("cuda")
        self.tar_seq_ram_fnn = FeedForward(self.hidden_size, self.inner_size, dropout_rate, self.hidden_act, self.layer_norm_eps).to("cuda")
       
        self.seq_tar_ram_position_embedding_retrieval = nn.Embedding(self.topk, self.hidden_size).to("cuda")

        self.tar_seq_ram_position_embedding_retrieval = nn.Embedding(self.topk, self.hidden_size).to("cuda")

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            # module.weight.data = self.truncated_normal_(tensor=module.weight.data, mean=0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def truncated_normal_(self, tensor, mean=0, std=0.09):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
    
    def get_bi_attention_mask(self, item_seq):
        """Generate bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # bidirectional mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        
        return output  # [B H]

    def seq_augmented(self, seq_output, batch_user_id, batch_seq_len, mode="train"):
        # تأكد من أن seq_output يحتفظ بالمشتقات
        seq_output = seq_output.requires_grad_(True)
        
        torch_retrieval_seq_embs1, torch_retrieval_tar_embs1, torch_retrieval_seq_embs2, torch_retrieval_tar_embs2 = self.retrieve_seq_tar(seq_output, batch_user_id, batch_seq_len, topk=self.topk, mode=mode)

        # تصفية وترجيح الذكريات المسترجعة
        filtered_seq_embs1, filtered_tar_embs1, relevance_scores1 = self.filter_and_weight_memories(
            seq_output, torch_retrieval_seq_embs1, torch_retrieval_tar_embs1
        )
        
        filtered_seq_embs2, filtered_tar_embs2, relevance_scores2 = self.filter_and_weight_memories(
            seq_output, torch_retrieval_seq_embs2, torch_retrieval_tar_embs2
        )

        # augmentation مع الذكريات المصفاة
        seq_output_saug = self.seq_tar_ram(seq_output.unsqueeze(1), filtered_seq_embs1, filtered_tar_embs1)
        seq_output_saug = self.seq_tar_ram_fnn(seq_output_saug)
        seq_output_saug = self.seq_tar_ram_1(seq_output_saug.unsqueeze(1), filtered_seq_embs1, filtered_tar_embs1) 
        
        seq_output_taug = self.tar_seq_ram(seq_output.unsqueeze(1), filtered_tar_embs2, filtered_seq_embs2)
        seq_output_taug = self.tar_seq_ram_fnn(seq_output_taug)
        seq_output_taug = self.tar_seq_ram_1(seq_output_taug.unsqueeze(1), filtered_tar_embs2, filtered_seq_embs2)
        
        # دمج الذكريات من كلا المصدرين
        if filtered_seq_embs1.dim() == 3 and filtered_seq_embs2.dim() == 3:
            retrieved_memories = torch.cat([filtered_seq_embs1, filtered_seq_embs2], dim=1)
        else:
            retrieved_memories = filtered_seq_embs1

        # التأكد من أبعاد صحيحة قبل الدمج
        if seq_output_saug.dim() > 2:
            seq_output_saug = seq_output_saug.squeeze(1)
        if seq_output_taug.dim() > 2:
            seq_output_taug = seq_output_taug.squeeze(1)
        
        # تأكد من أن المدخلات تحتفظ بالمشتقات
        seq_output_saug = seq_output_saug.requires_grad_(True)
        seq_output_taug = seq_output_taug.requires_grad_(True)
        
        # إضافة معلومات الصلة إلى البوابة
        seq_output_enhanced, gate_info = self.meta_fusion_gate(
            user_repr=seq_output,
            channel1_repr=seq_output_saug,
            channel2_repr=seq_output_taug,
            retrieved_memories=retrieved_memories
        )
        
        # إضافة معلومات الصلة إلى gate_info
        gate_info['memory_relevance'] = torch.mean(relevance_scores1, dim=1)
        
        # حفظ معلومات البوابة للاستخدام في حساب الخسارة
        self.last_gate_info = gate_info
        
        # جمع إحصائيات α, β
        if not hasattr(self, 'alpha_stats'):
            self.alpha_stats = []
            self.beta_stats = []
            self.relevance_stats = []
        
        # حفظ قيم α, β, relevance للتحليل
        self.alpha_stats.extend(gate_info['adaptive_alpha'].detach().cpu().numpy().flatten())
        self.beta_stats.extend(gate_info['adaptive_beta'].detach().cpu().numpy().flatten())
        self.relevance_stats.extend(gate_info['memory_relevance'].detach().cpu().numpy().flatten())
        
        return seq_output_enhanced

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        
        # تحديث رقم الحقبة الحالية
        if not hasattr(self, 'current_epoch'):
            self.current_epoch = 0

        # استخدام النموذج الأصلي في الحقب الأولى للتدريب المسبق
        use_original_model = self.current_epoch < self.gate_warmup_epochs
        
        pos_items = interaction[self.POS_ITEM_ID]
        batch_user_id = list(interaction[self.USER_ID].detach().cpu().numpy())
        batch_seq_len = list(item_seq_len.detach().cpu().numpy())
    
        # aug
        if use_original_model:
            # استخراج تمثيلات القنوات
            torch_retrieval_seq_embs1, torch_retrieval_tar_embs1, torch_retrieval_seq_embs2, torch_retrieval_tar_embs2 = self.retrieve_seq_tar(seq_output, batch_user_id, batch_seq_len, topk=self.topk)
            
            seq_output_saug = self.seq_tar_ram(seq_output.unsqueeze(1), torch_retrieval_seq_embs1, torch_retrieval_tar_embs1)
            seq_output_saug = self.seq_tar_ram_fnn(seq_output_saug)
            seq_output_saug = self.seq_tar_ram_1(seq_output_saug.unsqueeze(1), torch_retrieval_seq_embs1, torch_retrieval_tar_embs1)
            if seq_output_saug.dim() > 2:
                seq_output_saug = seq_output_saug.squeeze(1)
            
            seq_output_taug = self.tar_seq_ram(seq_output.unsqueeze(1), torch_retrieval_tar_embs2, torch_retrieval_seq_embs2)
            seq_output_taug = self.tar_seq_ram_fnn(seq_output_taug)
            seq_output_taug = self.tar_seq_ram_1(seq_output_taug.unsqueeze(1), torch_retrieval_tar_embs2, torch_retrieval_seq_embs2)
            if seq_output_taug.dim() > 2:
                seq_output_taug = seq_output_taug.squeeze(1)
            
            # استخدام المعاملات الثابتة كما في النموذج الأصلي
            alpha = self.alpha
            beta = self.beta
            seq_output_aug = alpha * seq_output + (1 - alpha) * (beta * seq_output_saug + (1 - beta) * seq_output_taug)
        else:
            # استخدام البوابة التكيفية
            seq_output_aug = self.seq_augmented(seq_output, batch_user_id, batch_seq_len)

        seq_output_aug = torch.where((item_seq_len > self.low_popular).unsqueeze(-1).repeat(1, 64), seq_output, seq_output_aug)

    
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output_aug * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output_aug * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output_aug, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)

        # إضافة تنظيم البوابة فقط بعد مرحلة التدريب المسبق
        if not use_original_model and hasattr(self, 'last_gate_info') and self.last_gate_info is not None:
            gate_reg_loss = self.compute_gate_regularization(self.last_gate_info)
            
            # تحديث البوابة بشكل صريح
            self.gate_optimizer.zero_grad()
            
            # تحديث مباشر للمعاملات بدلاً من backward
            with torch.no_grad():
                # تحريك alpha نحو 0.7
                target_alpha = 0.7
                current_alpha = torch.sigmoid(self.meta_fusion_gate.alpha_param).item()
                alpha_step = 0.05 * (target_alpha - current_alpha)  # زيادة معدل التحديث من 0.01 إلى 0.05
                
                # تحريك beta نحو 0.3
                target_beta = 0.3
                current_beta = torch.sigmoid(self.meta_fusion_gate.beta_param).item()
                beta_step = 0.05 * (target_beta - current_beta)  # زيادة معدل التحديث من 0.01 إلى 0.05
                
                # تطبيق التحديث
                new_alpha = self.meta_fusion_gate.alpha_param + alpha_step
                new_beta = self.meta_fusion_gate.beta_param + beta_step
                
                self.meta_fusion_gate.alpha_param.copy_(new_alpha)
                self.meta_fusion_gate.beta_param.copy_(new_beta)

            # طباعة معلومات تشخيصية
            if hasattr(self, 'debug_counter'):
                self.debug_counter += 1
            else:
                self.debug_counter = 0
                
            if self.debug_counter % 100 == 0:
                alpha_param = torch.sigmoid(self.meta_fusion_gate.alpha_param).item()
                beta_param = torch.sigmoid(self.meta_fusion_gate.beta_param).item()
                print(f"🔧 Gate params: α={alpha_param:.4f}, β={beta_param:.4f}, steps: α={alpha_step:.4f}, β={beta_step:.4f}")
            
            total_loss = loss + self.gate_regularization_weight * gate_reg_loss
            return total_loss
        else:
            return loss

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size
    
        z = torch.cat((z_i, z_j), dim=0)
    
        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp
    
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
    
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)
    
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def decompose(self, z_i, z_j, origin_z, batch_size):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size
    
        z = torch.cat((z_i, z_j), dim=0)
    
        # pairwise l2 distace
        sim = torch.cdist(z, z, p=2)
    
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
    
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        alignment = positive_samples.mean()

        # pairwise l2 distace
        sim = torch.cdist(origin_z, origin_z, p=2)
        mask = torch.ones((batch_size, batch_size), dtype=bool)
        mask = mask.fill_diagonal_(0)
        negative_samples = sim[mask].reshape(batch_size, -1)
        uniformity = torch.log(torch.exp(-2 * negative_samples).mean())
        
        return alignment, uniformity
    
    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def retrieve_seq_tar(self, qeuries, batch_user_id, batch_seq_len, topk=5, mode="train"):
        qeuries_cpu = qeuries.detach().cpu().numpy()
        normalize_L2(qeuries_cpu)
        _, I1 = self.seq_emb_index.search(qeuries_cpu, 4*topk)
        I1_filtered = []
        for i, I_entry in enumerate(I1):
            current_user = batch_user_id[i]
            current_length = batch_seq_len[i]
            filtered_indices = [idx for idx in I_entry if self.user_id_list[idx] != current_user or (self.user_id_list[idx] == current_user and self.item_seq_len_all[idx] < current_length)]
            I1_filtered.append(filtered_indices[:topk])
        I1_filtered = np.array(I1_filtered)
        if mode=="train":
            retrieval_seq1 = self.seq_emb_knowledge[I1_filtered]
            retrieval_tar1 = self.tar_emb_knowledge[I1_filtered]
            retrieval_seq2 = self.seq_emb_knowledge[I1_filtered]
            retrieval_tar2 = self.tar_emb_knowledge[I1_filtered]
        else:
            retrieval_seq1 = self.seq_emb_knowledge[I1_filtered]
            retrieval_tar1 = self.tar_emb_knowledge[I1_filtered]
            retrieval_seq2 = self.seq_emb_knowledge[I1_filtered]
            retrieval_tar2 = self.tar_emb_knowledge[I1_filtered]
        return torch.tensor(retrieval_seq1).to("cuda"), torch.tensor(retrieval_tar1).to("cuda"), torch.tensor(retrieval_seq2).to("cuda"), torch.tensor(retrieval_tar2).to("cuda")

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        batch_user_id = list(interaction[self.USER_ID].detach().cpu().numpy())
        batch_seq_len = list(item_seq_len.detach().cpu().numpy())
        # aug
        seq_output_aug = self.seq_augmented(seq_output, batch_user_id, batch_seq_len, mode="test")
        seq_output_aug = torch.where((item_seq_len > self.low_popular).unsqueeze(-1).repeat(1, 64), seq_output, seq_output_aug)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output_aug, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores

    def filter_and_weight_memories(self, query_repr, retrieved_seq_embs, retrieved_tar_embs):
        """
        تصفية وترجيح الذكريات المسترجعة بناءً على مدى صلتها بالاستعلام
        
        Args:
            query_repr: تمثيل المستخدم الحالي [batch_size, hidden_size]
            retrieved_seq_embs: تمثيلات المستخدمين المسترجعة [batch_size, K, hidden_size]
            retrieved_tar_embs: تمثيلات العناصر المستهدفة المسترجعة [batch_size, K, hidden_size]
        
        Returns:
            filtered_seq_embs: تمثيلات المستخدمين المصفاة والمرجحة
            filtered_tar_embs: تمثيلات العناصر المستهدفة المصفاة والمرجحة
            relevance_scores: درجات الصلة للذكريات المسترجعة
        """
        batch_size, K, hidden_size = retrieved_seq_embs.shape
        
        # حساب درجات التشابه بين الاستعلام وكل ذاكرة مسترجعة
        query_expanded = query_repr.unsqueeze(1).expand(-1, K, -1)  # [batch_size, K, hidden_size]
        
        # استخدام تشابه الجيب التمام للحصول على درجات بين 0 و 1
        relevance_scores = F.cosine_similarity(query_expanded, retrieved_seq_embs, dim=2)  # [batch_size, K]
        
        # تطبيق softmax للحصول على أوزان تجمع إلى 1
        attention_weights = F.softmax(relevance_scores / 0.1, dim=1).unsqueeze(2)  # [batch_size, K, 1]
        
        # ترجيح الذكريات بأوزان الانتباه
        filtered_seq_embs = retrieved_seq_embs * attention_weights
        filtered_tar_embs = retrieved_tar_embs * attention_weights
        
        return filtered_seq_embs, filtered_tar_embs, relevance_scores

    def compute_gate_regularization(self, gate_info):
        """حساب خسارة تنظيم للبوابة لتشجيع قيم متوازنة"""
        # تشجيع alpha على أن تكون أكبر من 0.5 (تفضيل تمثيل المستخدم الأصلي)
        alpha_reg = torch.mean(torch.relu(0.6 - gate_info['adaptive_alpha']))
        
        # تشجيع beta على أن تكون أقل من 0.5 (تفضيل القناة الثانية)
        beta_reg = torch.mean(torch.relu(gate_info['adaptive_beta'] - 0.4))
        
        # الخسارة الإجمالية للتنظيم
        total_reg = alpha_reg + beta_reg
        
        return total_reg

    def load_state_dict(self, state_dict, strict=True):
        """تحميل checkpoint مع تجاهل الأجزاء المفقودة الجديدة"""
        # فصل الأجزاء الخاصة بـ Meta-α Fusion Gate
        meta_gate_keys = [k for k in state_dict.keys() if 'meta_fusion_gate' in k]
        
        # إذا لم توجد مفاتيح Meta-α Fusion Gate في الـ checkpoint
        if not meta_gate_keys:
            print("⚠️ Warning: Meta-α Fusion Gate not found in the checkpoint.")
            print("→ Initializing Meta-α Fusion Gate with random weights.")
            
            # إزالة المفاتيح المفقودة من التحقق الصارم
            missing_keys = [k for k in self.state_dict().keys() if 'meta_fusion_gate' in k]
            for key in missing_keys:
                if key in self.state_dict():
                    state_dict[key] = self.state_dict()[key]
        
        # تحميل باقي الأوزان
        return super().load_state_dict(state_dict, strict=False)
            
    def print_gate_statistics(self, epoch):
        """طباعة إحصائيات البوابة المتعلمة"""
        if hasattr(self, 'alpha_stats') and len(self.alpha_stats) > 0:
            import numpy as np
            
            alpha_mean = np.mean(self.alpha_stats)
            beta_mean = np.mean(self.beta_stats)
            relevance_mean = np.mean(self.relevance_stats) if hasattr(self, 'relevance_stats') and len(self.relevance_stats) > 0 else 0.0
            
            # القيم الحالية للمعاملات
            alpha_value = torch.sigmoid(self.meta_fusion_gate.alpha_param).item()
            beta_value = torch.sigmoid(self.meta_fusion_gate.beta_param).item()
            
            print(f"-------------------")
            print(f"Epoch {epoch} Gate Statistics:")
            print(f"Current gate params: α={alpha_value:.4f}, β={beta_value:.4f}")
            print(f"Applied values: α mean={alpha_mean:.4f}, β mean={beta_mean:.4f}")
            print(f"Memory relevance mean={relevance_mean:.4f}")
            print(f"Samples={len(self.alpha_stats)}")
            print(f"-------------------")
            
            # إعادة تعيين للـ epoch التالي
            self.alpha_stats = []
            self.beta_stats = []
            if hasattr(self, 'relevance_stats'):
                self.relevance_stats = []
