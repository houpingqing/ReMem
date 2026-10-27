import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import random
import re
from twfinch import FINCH
from encoder import CLIPDoubleTower
from decord import VideoReader, cpu
from PIL import Image
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
import argparse
import subprocess
import tempfile
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

    if len(pooled_image) < 2:
        return text_guide_idx

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


def text_guided_frames_tokens(E_q_prime, images, clip_model, batch_size=8):
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


def encode_video_1fps(video_path, clip_model, batch_size=8):
    """Encode sampled frames in batches so long videos do not fill host RAM."""
    # Decord can crash while constructing VideoReader on malformed or very
    # long Windows MP4s. In FFmpeg/PyAV mode, obtain only stream metadata via
    # PyAV and never construct a Decord reader.
    decoder_mode = os.environ.get('REMEM_VIDEO_DECODER', '').lower()
    vr = None
    if decoder_mode in ('ffmpeg', 'pyav'):
        import av
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            rate = stream.average_rate or stream.base_rate
            fps = float(rate) if rate else 0.0
            frame_count = int(stream.frames or 0)
            if frame_count <= 0 and stream.duration and stream.time_base and fps > 0:
                frame_count = max(1, int(round(float(stream.duration * stream.time_base) * fps)))
        if fps <= 0 or frame_count <= 0:
            raise ValueError(f'Cannot read video metadata: {video_path}')
        frame_ids = [min(int(i * fps), frame_count - 1) for i in range(max(1, int(frame_count / fps)))]
    else:
        # Decord's multithreaded seek occasionally crashes on long Video-MME
        # MP4s on Windows. A single decoder thread keeps sampled frame IDs.
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        fps = vr.get_avg_fps()
        if fps <= 0 or len(vr) == 0:
            raise ValueError(f'Cannot read video frames: {video_path}')
        frame_ids = [min(int(i * fps), len(vr) - 1) for i in range(max(1, int(len(vr) / fps)))]
    pooled = []
    last_tenth = -1
    if decoder_mode == 'ffmpeg':
        # Keep exact Decord frame ordinals while isolating native decoding in
        # an FFmpeg process. Some Video-MME MP4s crash in-process decoders on Windows.
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        probe = subprocess.run(
            [ffmpeg_exe, '-hide_banner', '-i', video_path],
            capture_output=True, text=True, timeout=30,
        )
        video_stream = next((line for line in probe.stderr.splitlines() if 'Video:' in line), '')
        size = re.search(r'\b(\d{2,5})x(\d{2,5})\b', video_stream)
        if size is None:
            raise ValueError(f'Cannot determine video dimensions: {video_path}')
        width, height = map(int, size.groups())
        frame_bytes = width * height * 3
        with tempfile.TemporaryDirectory(prefix='remem-ffmpeg-') as temp_dir:
            filter_script = os.path.join(temp_dir, 'select.txt')
            with open(filter_script, 'w', encoding='utf-8') as script:
                script.write('select=' + '+'.join(f'eq(n\\,{index})' for index in frame_ids))
            process = subprocess.Popen(
                [ffmpeg_exe, '-hide_banner', '-loglevel', 'error',
                 '-i', video_path, '-filter_script:v', filter_script, '-vsync', '0',
                 '-an', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            try:
                for start in range(0, len(frame_ids), batch_size):
                    images = []
                    for _ in frame_ids[start:start + batch_size]:
                        buffer = bytearray()
                        while len(buffer) < frame_bytes:
                            piece = process.stdout.read(frame_bytes - len(buffer))
                            if not piece:
                                break
                            buffer.extend(piece)
                        if len(buffer) != frame_bytes:
                            raise ValueError(f'FFmpeg returned only {len(buffer)}/{frame_bytes} bytes for {video_path}')
                        images.append(Image.frombytes('RGB', (width, height), bytes(buffer)))
                    _, features = clip_model.get_pooled_features(None, images)
                    pooled.append(features)
                    done = min(start + batch_size, len(frame_ids))
                    tenth = (10 * done) // len(frame_ids)
                    if tenth > last_tenth or done == len(frame_ids):
                        print(f"  {os.path.basename(video_path)}: encoded {done}/{len(frame_ids)} frames ({100 * done // len(frame_ids)}%)", flush=True)
                        last_tenth = tenth
                if process.wait(timeout=60) != 0:
                    raise RuntimeError(f'FFmpeg exited with code {process.returncode}: {video_path}')
            finally:
                process.stdout.close()
                if process.poll() is None:
                    process.kill()
                    process.wait()
        return torch.cat(pooled, dim=0), frame_ids
    if decoder_mode == 'pyav' or len(frame_ids) >= 2000:
        # Decord's Windows get_batch can crash on Video-MME MP4s. Decode the
        # same frame ordinals sequentially with FFmpeg/PyAV instead.
        import av

        target = 0
        images = []
        with av.open(video_path) as container:
            for frame_number, frame in enumerate(container.decode(video=0)):
                if target >= len(frame_ids):
                    break
                if frame_number != frame_ids[target]:
                    continue
                images.append(frame.to_image())
                target += 1
                if len(images) == batch_size or target == len(frame_ids):
                    _, features = clip_model.get_pooled_features(None, images)
                    pooled.append(features)
                    images.clear()
                    tenth = (10 * target) // len(frame_ids)
                    if tenth > last_tenth or target == len(frame_ids):
                        print(f"  {os.path.basename(video_path)}: encoded {target}/{len(frame_ids)} frames ({100 * target // len(frame_ids)}%)", flush=True)
                        last_tenth = tenth
        if target != len(frame_ids):
            raise ValueError(f'FFmpeg decoded only {target}/{len(frame_ids)} sampled frames: {video_path}')
        return torch.cat(pooled, dim=0), frame_ids
    for start in range(0, len(frame_ids), batch_size):
        batch_ids = frame_ids[start:start + batch_size]
        images = [Image.fromarray(array) for array in vr.get_batch(batch_ids).asnumpy()]
        _, features = clip_model.get_pooled_features(None, images)
        pooled.append(features)
        done = min(start + batch_size, len(frame_ids))
        tenth = (10 * done) // len(frame_ids)
        if tenth > last_tenth or done == len(frame_ids):
            print(f"  {os.path.basename(video_path)}: encoded {done}/{len(frame_ids)} frames ({100 * done // len(frame_ids)}%)", flush=True)
            last_tenth = tenth
    return torch.cat(pooled, dim=0), frame_ids


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


def worker(rank, world_size, shard_paths, output_path, MODEL_NAME, video_root, device_choice, num_frames):
    if device_choice == 'cuda':
        torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if device_choice == 'cuda' else "cpu")
    clip_model = CLIPDoubleTower(model_name=MODEL_NAME)
    clip_model.clip.to(device)
    clip_model.device = device
    clip_model.clip.eval()
    torch.set_grad_enabled(False)
    shard_path = shard_paths[rank]
    with open(shard_path, 'r', encoding='utf-8') as f:
        shard = json.load(f)
    part_path = f"{output_path}.part{rank}.json"
    processed_ids = set()
    if os.path.exists(part_path):
        with open(part_path, 'r', encoding='utf-8') as f:
            try:
                processed_data = json.load(f)
                processed_ids = set(d['question_id'] for d in processed_data if 'question_id' in d)
            except Exception:
                processed_data = []
    else:
        processed_data = []
    total = len(shard)
    print(f"[{device}] {len(processed_ids)}/{total} questions already completed", flush=True)
    cached_video_path = None
    image_pooled = None
    frames_id = None
    for idx, entry in enumerate(shard):
        data = entry["data"]
        qid = data.get('question_id', None)
        if qid is not None and qid in processed_ids:
            continue
        print(f"[{device}] Question {idx+1}/{total} ({qid}): {data.get('video_name', '')}", flush=True)
        question = data['question']
        # The released annotations use Windows separators. Normalize them
        # before joining on Linux servers while preserving the original path
        # semantics on Windows.
        video_name = str(data['video_name']).replace('\\', os.sep).replace('/', os.sep)
        video_path = os.path.join(video_root, video_name)
        granularity_score = data['granularity']
        video_duration = int(round(data['duration']))
        print(f"[{device}] Video duration: {video_duration} seconds", flush=True)
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
        expander = MemoryQueryExpander(d_model=E_q.shape[-1]).to(device)
        E_q_prime = expander(E_q.to(device), E_k.to(device)) 

        if video_path != cached_video_path:
            image_pooled, frames_id = encode_video_1fps(video_path, clip_model)
            cached_video_path = video_path
        question_tokens = F.normalize(E_q_prime, p=2, dim=-1)
        image_tokens = F.normalize(image_pooled, p=2, dim=-1)
        cos = F.cosine_similarity(question_tokens, image_tokens, dim=-1)
        graph_nodes, adj_matrix = build_memory_graph(image_tokens, K=None, alpha=0.5)
        node_importance = graph_reasoning(question_tokens, graph_nodes, adj_matrix)
        mem_sim_all = torch.matmul(image_tokens, graph_nodes.T)
        weighted_scores = torch.sum(mem_sim_all * node_importance, dim=1)
        final_scores = alpha * cos.to(weighted_scores.device) + (1 - alpha) * weighted_scores
        num_top = 150
        topk = torch.topk(final_scores, k=min(num_top, len(final_scores)))
        top_indices = topk.indices
        finch_clust_idx = finch_cluster(image_pooled[top_indices], top_indices, final_scores[top_indices], num_frames)
        selected_indices = finch_clust_idx.tolist()
        if len(selected_indices) > num_frames:
            selected_indices = sorted(selected_indices, key=lambda i: float(final_scores[i]), reverse=True)[:num_frames]
        if len(selected_indices) < num_frames:
            already = set(selected_indices)
            pad_candidates = [i for i in torch.topk(final_scores, k=len(final_scores)).indices.tolist() if i not in already]
            need = num_frames - len(selected_indices)
            selected_indices += pad_candidates[:need]
        selected_frame_ids = [frames_id[i] for i in selected_indices]
        sorted_frame_ids = sorted([float(i) for i in selected_frame_ids])
        data['frame_idx'] = sorted_frame_ids
        del cos, selected_frame_ids
        if device_choice == 'cuda':
            torch.cuda.empty_cache()

        processed_data.append(data)
        checkpoint_tmp = f"{part_path}.tmp"
        with open(checkpoint_tmp, 'w', encoding='utf-8') as f:
            f.write(json.dumps(processed_data, ensure_ascii=False))
        os.replace(checkpoint_tmp, part_path)
        print(f"[{device}] Completed {len(processed_data)}/{total} questions", flush=True)
        if idx % 50 == 0:
            gc.collect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', default='./VideoMME/val_qa_full.json')
    parser.add_argument('--video-root', default='./VideoMME/videos')
    parser.add_argument('--output', default='./VideoMME/val_qa_selected_full_32f.json')
    local_clip = './models/clip-vit-large-patch14-336'
    parser.add_argument('--clip-model', default=local_clip if os.path.isdir(local_clip) else 'openai/clip-vit-large-patch14-336')
    parser.add_argument('--limit', type=int, default=None, help='Process only the first N questions for a smoke test')
    parser.add_argument('--num-frames', type=int, default=32, help='Target number of selected frames per question')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    args = parser.parse_args()
    if args.num_frames < 1:
        parser.error('--num-frames must be positive')
    label_path, video_root, output_path, MODEL_NAME = args.labels, args.video_root, args.output, args.clip_model
    if not os.path.isfile(label_path):
        parser.error(f'Annotation file not found: {label_path}')
    if not os.path.isdir(video_root):
        parser.error(f'Video directory not found: {video_root}')
    with open(label_path,'r', encoding='utf-8') as f:
        datas = json.load(f)
    if args.limit is not None:
        if args.limit < 1:
            parser.error('--limit must be positive')
        datas = datas[:args.limit]

    device_choice = args.device
    if device_choice == 'auto':
        free_mib = 0
        if torch.cuda.is_available():
            try:
                report = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                    text=True, timeout=5,
                )
                free_mib = min(int(line.strip()) for line in report.splitlines() if line.strip())
            except Exception:
                free_mib = torch.cuda.mem_get_info()[0] // 1024**2
        device_choice = 'cuda' if free_mib >= 5120 else 'cpu'
    if device_choice == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA was requested but CUDA-enabled PyTorch is unavailable')
    logging.info('Frame selection device: %s', device_choice)
    world_size = torch.cuda.device_count() if device_choice == 'cuda' else 1
    shard_dir = f"{output_path}.shards"
    os.makedirs(shard_dir, exist_ok=True)
    shard_paths = []
    for rank in range(world_size):
        shard_path = os.path.join(shard_dir, f"shard{rank}.json")
        shard = []
        for idx in range(rank, len(datas), world_size):
            shard.append({"data": datas[idx]})
        with open(shard_path, "w", encoding='utf-8') as f:
            json.dump(shard, f, indent=4, ensure_ascii=False)
        shard_paths.append(shard_path)
    del datas
    gc.collect()

    mp.spawn(worker, args=(world_size, shard_paths, output_path, MODEL_NAME, video_root, device_choice, args.num_frames), nprocs=world_size)

    merged = []
    for rank in range(world_size):
        part_path = f"{output_path}.part{rank}.json"
        with open(part_path, encoding='utf-8') as f:
            merged.extend(json.load(f))
        os.remove(part_path)
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(merged, f, indent=4, ensure_ascii=False)
    print(f"Selected frames for {len(merged)} questions: {output_path}", flush=True)

