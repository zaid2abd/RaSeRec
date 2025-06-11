# # Collaborative-based Pre-training Stage
# python run_seq.py \
#     --dataset='beauty' \
#     --gpu_id=1  \
#     --metrics="['Recall', 'NDCG', 'MRR']" \
#     --valid_metric="MRR@10" \
#     --train_batch_size=1024 \
#     --lmd=0.1 \
#     --lmd_sem=0.1 \
#     --model='DuoRec' \
#     --contrast='us_x' \
#     --sim='dot' \
#     --tau=1 \
#     --nproc=2 \
#     --epochs=100 \
#     --data_path="./recbole/dataset"


# # Retrieval-Augmented Fine-tuning Stage
# python run_seq.py \
#     --dataset='beauty' \
#     --nprobe=1 \
#     --attn_tau=1.0 \
#     --dropout_rate=0.5 \
#     --alpha=0.5 \
#     --beta=1.0 \
#     --top_k=20 \
#     --metrics="['Recall', 'NDCG']" \
#     --valid_metric="Recall@10" \
#     --stopping_step=10 \
#     --train_batch_size=1024 \
#     --model='RaSeRec' \
#     --sim='dot' \
#     --tau=1 \
#     --nproc=2 \
#     --epochs=100 \
#     --data_path="./recbole/dataset" \
#     --pre_training_ckt="./log/DuoRec/beauty/bs256-lmd0.1-sem0.1-us_x-Sep-26-2024_19-07-19-lr0.001-l20-tau1-dot-DPh0.5-DPa0.5/model.pth" \
#     --gate_reg_weight=0.01 \
#     --meta_fusion_enabled=true \


# Retrieval-Augmented Fine-tuning Stage with Learnable Meta-α Fusion Gate
python run_seq.py \
    --dataset='beauty' \
    --nprobe=1 \
    --attn_tau=1.0 \
    --dropout_rate=0.5 \
    --alpha=0.5 \
    --beta=0.5 \
    --top_k=30 \
    --metrics="['Recall', 'NDCG']" \
    --valid_metric="Recall@10" \
    --stopping_step=15 \
    --train_batch_size=1024 \
    --model='RaSeRec' \
    --sim='dot' \
    --tau=1 \
    --nproc=2 \
    --epochs=150 \
    --data_path="./recbole/dataset" \
    --pre_training_ckt="./log/DuoRec/beauty/bs256-lmd0.1-sem0.1-us_x-Sep-26-2024_19-07-19-lr0.001-l20-tau1-dot-DPh0.5-DPa0.5/model.pth" \
    --gate_reg_weight=0.005 \
    --meta_fusion_enabled=true \
    --gate_warmup_epochs=5 \
    
