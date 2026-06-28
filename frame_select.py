import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import random
import re
from twfinch import FINCH
from encoder import CLIPDoubleTower
from dataset import load_video_1fps
import os
import json
from transformers import CLIPTokenizer
import logging
from sklearn.cluster import KMeans
import sys
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
import math
import gc
import torch.multiprocessing as mp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MemoryQueryExpander(nn.Module):
    def __init__(self, d_model=768): 
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, E_q, E_k):
        """
        E_q: [1, d_model] 
        E_k: [N, d_model]
        """
        Q = self.query_proj(E_q)      # [1, d_model]
        K = self.key_proj(E_k)        # [N, d_model]
        V = self.value_proj(E_k)      # [N, d_model]

        scaling = Q.size(-1) ** 0.5
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / scaling # [1, N]
        attention_probs = F.softmax(attention_scores, dim=-1) 

        E_memo = torch.matmul(attention_probs, V) # [1, d_model]

        combined = torch.cat([E_q, E_memo], dim=-1) # [1, 2 * d_model]
        E_q_prime = self.fusion(combined)
        
        return self.norm(E_q_prime + E_q) # [1, d_model]


def finch_cluster(pooled_image, text_guide_idx, simlarity_score, max_frames_to_select=64):

    pooled_image = pooled_image.detach().cpu().numpy()
    c, num_clust, _ = FINCH(pooled_image, verbose=False, tw_finch=True)
    classification = torch.tensor(
        c[:, np.argsort(num_clust)[-2] if len(num_clust) > 1 else -1]
    )

    device = text_guide_idx.device if isinstance(text_guide_idx, torch.Tensor) else torch.device("cpu")
    if not isinstance(text_guide_idx, torch.Tensor):
        text_guide_idx = torch.tensor(text_guide_idx, dtype=torch.long, device=device)
    if not isinstance(simlarity_score, torch.Tensor):
        simlarity_score = torch.tensor(simlarity_score, dtype=torch.float32, device=device)
    classification = classification.to(device)
    unique_classes = torch.unique(classification)
    selected_indexs = []

    if len(unique_classes) <= 1:
        k = min(max_frames_to_select, len(simlarity_score))
        topk = torch.topk(simlarity_score, k=k)
        best_frames = text_guide_idx[topk.indices].tolist()
        selected_indexs.extend(best_frames)
    else:
        per_class_quota = {cls.item(): 1 for cls in unique_classes}
        remaining = max_frames_to_select - len(unique_classes)
        class_sizes = {cls.item(): (classification == cls).sum().item() for cls in unique_classes}
        sorted_classes = sorted(class_sizes.items(), key=lambda x: x[1], reverse=True)
        
        while remaining > 0:
            updated = False
            for cls, size in sorted_classes:
                if per_class_quota[cls] < size:
                    per_class_quota[cls] += 1
                    remaining -= 1
                    updated = True
                    if remaining <= 0: break
            if not updated: break

        for cls in unique_classes:
            cls_mask = (classification == cls)
            indices = cls_mask.nonzero(as_tuple=True)[0]
            
            cls_scores = simlarity_score[indices]
            num = per_class_quota[cls.item()]
            topk = torch.topk(cls_scores, k=min(num, len(indices)))
            
            selected_global_idxs = text_guide_idx[indices][topk.indices]
            selected_indexs.extend(selected_global_idxs.tolist())


    selected_indexs_tensor = torch.tensor(selected_indexs, dtype=torch.long, device=device)
    return selected_indexs_tensor


def text_guided_frames_tokens(E_q_prime, images, clip_model, batch_size=128):
    image_pooled_list = []
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i+batch_size]
        _, image_pooled = clip_model.get_pooled_features(None, batch_imgs)
        image_pooled_list.append(image_pooled)
    image_pooled = torch.cat(image_pooled_list, dim=0)  # [N, D]

    question_tokens = F.normalize(E_q_prime, p=2, dim=-1)
    image_tokens = F.normalize(image_pooled, p=2, dim=-1)

    cos = F.cosine_similarity(question_tokens, image_tokens, dim=-1)
    return cos, question_tokens, image_tokens, image_pooled


def build_memory_graph(dense_feats, K=None, alpha=0.5):

    dense_feats = F.normalize(dense_feats, p=2, dim=-1)
    N = dense_feats.shape[0]
    min_K, max_K = 8, 32
    if K is None:
        K = int(np.clip(np.sqrt(N), min_K, max_K))
    else:
        K = int(K)
    if N <= K:
        eye = torch.eye(N).to(dense_feats.device)
        return dense_feats, eye

    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(dense_feats.detach().cpu().numpy())
    centers = torch.tensor(kmeans.cluster_centers_).to(dense_feats.device)

    graph_nodes = []
    node_timesteps = []

    for k in range(K):
        cluster_mask = (labels == k)
        cluster_feats = dense_feats[cluster_mask]
        cluster_indices = torch.where(torch.tensor(cluster_mask))[0]

        if len(cluster_feats) == 0:
            continue

        distances = torch.norm(cluster_feats - centers[k], dim=1)
        best_idx = torch.argmin(distances)
        graph_nodes.append(cluster_feats[best_idx])
        node_timesteps.append(cluster_indices.float().mean().item())

    graph_nodes = torch.stack(graph_nodes)


    sort_idx = torch.argsort(torch.tensor(node_timesteps))
    sorted_nodes = graph_nodes[sort_idx]

    sim_adj = torch.matmul(sorted_nodes, sorted_nodes.T)
    sim_adj = F.relu(sim_adj)

    K_real = len(sorted_nodes)
    temp_adj = torch.zeros((K_real, K_real)).to(dense_feats.device)
    for i in range(K_real - 1):
        temp_adj[i, i+1] = 1.0 
        temp_adj[i+1, i] = 1.0 
    temp_adj = temp_adj + torch.eye(K_real).to(dense_feats.device) 

    final_adj = (1 - alpha) * temp_adj + alpha * sim_adj
    row_sum = final_adj.sum(dim=1, keepdim=True) + 1e-6
    final_adj = final_adj / row_sum

    return sorted_nodes, final_adj


def graph_reasoning(query_feat, graph_nodes, adj_matrix, steps=2):

    seed = torch.matmul(graph_nodes, query_feat.T).squeeze() # [K]
    seed = F.softmax(seed, dim=0) 
    
    current_activation = seed.clone()
    restart_prob = 0.6 
    
    for _ in range(steps):
        propagated = torch.matmul(adj_matrix, current_activation)
        current_activation = (1 - restart_prob) * propagated + restart_prob * seed
        
    return current_activation


def worker(rank, world_size, shard_paths, output_path, MODEL_NAME, video_root):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    clip_model = CLIPDoubleTower(model_name=MODEL_NAME)
    clip_model.clip.to(device)
    clip_model.device = device
    clip_model.clip.eval()
    torch.set_grad_enabled(False)
    shard_path = shard_paths[rank]
    with open(shard_path, 'r') as f:
        shard = json.load(f)
    part_path = f"{output_path}.part{rank}.json"
    processed_ids = set()
    if os.path.exists(part_path):
        with open(part_path, 'r') as f:
            try:
                processed_data = json.load(f)
                processed_ids = set(d['question_id'] for d in processed_data if 'question_id' in d)
            except Exception:
                processed_data = []
    else:
        processed_data = []
    total = len(shard)
    for idx, entry in enumerate(shard):
        data = entry["data"]
        qid = data.get('question_id', None)
        if qid is not None and qid in processed_ids:
            continue
        logging.info(f"[GPU {rank}] Processing {idx+1}/{total}: {data.get('video_name', '')}")
        question = data['question']
        video_path = os.path.join(video_root, data['video_name'])
        granularity_score = data['granularity']
        video_duration = int(round(data['duration']))
        print(f"[GPU {rank}] Video duration (number of frames): {video_duration}")
        granularity_score = float(granularity_score)
        LAMBDA = 1e-4
        alpha = 1 - 0.5 * granularity_score * math.exp(-LAMBDA * video_duration)
        keywords = data['entity_keywords']
        E_q, _ = clip_model.get_pooled_features(question, None)  # [1, d_model]
        E_k_list = []
        for kw in keywords:
            kw_feat, _ = clip_model.get_pooled_features(kw, None)
            E_k_list.append(kw_feat)
        E_k = torch.cat(E_k_list, dim=0)  # [N, d_model]
        expander = MemoryQueryExpander(d_model=768).to(device)
        E_q_prime = expander(E_q.to(device), E_k.to(device)) 

        frames, sizes, frames_id = load_video_1fps(video_path)
        cos, question_tokens, image_tokens, image_pooled = text_guided_frames_tokens(
            E_q_prime, frames, clip_model=clip_model
        )
        graph_nodes, adj_matrix = build_memory_graph(image_tokens, K=None, alpha=0.5)
        node_importance = graph_reasoning(question_tokens, graph_nodes, adj_matrix)
        mem_sim_all = torch.matmul(image_tokens, graph_nodes.T)
        weighted_scores = torch.sum(mem_sim_all * node_importance, dim=1)
        final_scores = alpha * torch.tensor(cos, device=weighted_scores.device) + (1 - alpha) * weighted_scores
        num_top = 150
        topk = torch.topk(final_scores, k=num_top)
        top_indices = topk.indices
        finch_clust_idx = finch_cluster(image_pooled[top_indices], top_indices, final_scores[top_indices], 64)
        selected_indices = finch_clust_idx.tolist()
        if len(selected_indices) < 64:
            already = set(selected_indices)
            pad_candidates = [i for i in torch.topk(final_scores, k=len(final_scores)).indices.tolist() if i not in already]
            need = 64 - len(selected_indices)
            selected_indices += pad_candidates[:need]
        selected_frame_ids = [frames_id[i] for i in selected_indices]
        sorted_frame_ids = sorted([float(i) for i in selected_frame_ids])
        data['frame_idx'] = sorted_frame_ids
        del frames, sizes, frames_id, image_pooled, cos, selected_frame_ids
        torch.cuda.empty_cache()

        processed_data.append(data)
        with open(part_path, 'w') as f:
            json.dump(processed_data, f, indent=4)
        if idx % 50 == 0:
            gc.collect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    label_path = "./VideoMME/val_qa.json"
    video_root = './VideoMME/videos'
    output_path = "./VideoMME/val_qa_selected.json"
    MODEL_NAME = "openai/clip-vit-large-patch14-336"
    with open(label_path,'r') as f:
        datas = json.load(f)

    world_size = torch.cuda.device_count()
    shard_dir = f"{output_path}.shards"
    os.makedirs(shard_dir, exist_ok=True)
    shard_paths = []
    for rank in range(world_size):
        shard_path = os.path.join(shard_dir, f"shard{rank}.json")
        if not os.path.exists(shard_path):
            shard = []
            for idx in range(rank, len(datas), world_size):
                shard.append({
                    "data": datas[idx],
                })
            with open(shard_path, "w") as f:
                json.dump(shard, f, indent=4)
        shard_paths.append(shard_path)
    del datas
    gc.collect()

    mp.spawn(worker, args=(world_size, shard_paths, output_path, MODEL_NAME, video_root), nprocs=world_size)

    merged = []
    for rank in range(world_size):
        part_path = f"{output_path}.part{rank}.json"
        with open(part_path) as f:
            merged.extend(json.load(f))
        os.remove(part_path)
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=4)

