import os
import re
import pandas as pd

# Define the path to the DuoRec logs
duorec_log_root = './log/DuoRec/beauty'

# Regex to extract MRR@10 and Recall@10
mrr_pattern = re.compile(r'mrr@10\s*:\s*([0-9.]+)', re.IGNORECASE)
recall_pattern = re.compile(r'recall@10\s*:\s*([0-9.]+)', re.IGNORECASE)

# Store results
results = []

# Traverse subdirectories and extract scores
for folder in os.listdir(duorec_log_root):
    folder_path = os.path.join(duorec_log_root, folder)
    if not os.path.isdir(folder_path):
        continue
    log_file = os.path.join(folder_path, 'log.txt')
    if not os.path.exists(log_file):
        log_file = os.path.join(folder_path, 'recbole.log')
    if not os.path.exists(log_file):
        continue

    with open(log_file, 'r', encoding='utf-8') as f:
        log_content = f.read()

    mrr_matches = mrr_pattern.findall(log_content)
    recall_matches = recall_pattern.findall(log_content)

    if mrr_matches and recall_matches:
        max_mrr = max(float(x) for x in mrr_matches)
        max_recall = max(float(x) for x in recall_matches)
        results.append({
            'folder': folder,
            'MRR@10': max_mrr,
            'Recall@10': max_recall
        })

# Analyze results
df = pd.DataFrame(results)
df_sorted = df.sort_values(by=['Recall@10', 'Recall@10'], ascending=False)

# Get best checkpoint info
best_folder = df_sorted.iloc[0]['folder'] if not df_sorted.empty else None
best_path = os.path.join(duorec_log_root, best_folder) if best_folder else "No valid folder found"

# Output
print(df_sorted.to_string(index=False))
print("\nBest checkpoint path:")
print(best_path)
