"""
构建文档库向量索引（RAG2，带切片）
知乎语料 → 切片（500字+100交叠）→ 编码 → 存 FAISS 索引
"""
import json
import sys
import io
import re
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# --- Config ---
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
SAMPLE_SIZE = 5000  # 取前 5000 条长文本，约为 2w 个块
MIN_LENGTH = 1000

BASE_DIR = Path(__file__).parent.parent
INDEX_DIR = BASE_DIR / "data"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# --- Load dataset ---
cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--suolyer--zhihu"
snapshot_dirs = list(cache_dir.glob("snapshots/*/"))
if not snapshot_dirs:
    raise RuntimeError("Dataset not found")
snapshot_dir = snapshot_dirs[0]

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

print("加载数据集中...")
all_answers = load_jsonl(snapshot_dir / "validation.json") + load_jsonl(snapshot_dir / "test.json")
print(f"总计 {len(all_answers)} 条")

long_answers = [a for a in all_answers if len(a.get("content", "")) > MIN_LENGTH]
print(f">1000 字的有 {len(long_answers)} 条")

samples = long_answers[:SAMPLE_SIZE]
print(f"取前 {len(samples)} 条\n")


# --- Chunking ---
def split_text(text, chunk_size=500, overlap=100):
    """按段落→句子→字符递归切片"""
    if len(text) <= chunk_size:
        return [text]

    chunks = []

    # 先按段落切
    paras = re.split(r'\n\n+', text)
    current = []

    for para in paras:
        para = para.strip()
        if not para:
            continue
        current_len = sum(len(p) for p in current)
        if current_len + len(para) + 2 <= chunk_size or not current:
            current.append(para)
        else:
            # 合并当前段落组为一个块
            chunks.append("\n\n".join(current))

            # overlap：从上一个块尾部回溯
            overlap_chars = []
            overlap_len = 0
            for p in reversed(current):
                if overlap_len + len(p) + 2 <= overlap:
                    overlap_chars.insert(0, p)
                    overlap_len += len(p) + 2
                else:
                    break

            current = overlap_chars + [para]

    if current:
        chunks.append("\n\n".join(current))

    # 如果块还是太长（单个段落超长），按句子二次切分
    final_chunks = []
    for c in chunks:
        if len(c) <= chunk_size:
            final_chunks.append(c)
        else:
            # 按句号/问号/感叹号分句
            sents = re.split(r'(?<=[。！？])', c)
            sub = []
            for s in sents:
                if sum(len(x) for x in sub) + len(s) <= chunk_size:
                    sub.append(s)
                else:
                    if sub:
                        final_chunks.append("".join(sub))
                    sub = [s]
            if sub:
                final_chunks.append("".join(sub))

    return final_chunks


# --- Chunk all samples ---
print("切片中...")
all_chunks = []
chunk_meta = []

for idx, item in enumerate(samples):
    content = item.get("content", "")
    title = item.get("title", "")
    url = item.get("url", "")

    # 超长截断（不截断会导致切片过多，且边角内容质量低）
    if len(content) > 5000:
        content = content[:5000]

    chunks = split_text(content, CHUNK_SIZE, CHUNK_OVERLAP)

    for ci, chunk_text in enumerate(chunks):
        all_chunks.append(chunk_text)
        chunk_meta.append({
            "chunk_id": f"chunk_{idx:04d}_{ci:02d}",
            "answer_idx": idx,
            "chunk_index": ci,
            "source_title": title[:80],
            "source_url": url,
            "text_preview": chunk_text[:100],
        })

print(f"切片完成: {len(samples)} 条 → {len(all_chunks)} 个块")
print(f"平均每条约 {len(all_chunks)/len(samples):.1f} 个块")
print(f"块长度分布: 最短={min(len(c) for c in all_chunks)}, 最长={max(len(c) for c in all_chunks)}")

# --- Encode ---
print(f"\n加载模型 {MODEL_NAME}...")
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(MODEL_NAME, device="cpu")

print(f"编码 {len(all_chunks)} 个块...")
vectors = model.encode(all_chunks, show_progress_bar=True, normalize_embeddings=True)
print(f"向量形状: {vectors.shape}")

# --- Build & save FAISS index ---
import faiss
import numpy as np

dim = vectors.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(vectors.astype(np.float32))

index_bytes = faiss.serialize_index(index)
index_path = INDEX_DIR / "doc.index"
with open(index_path, "wb") as f:
    f.write(index_bytes)
print(f"\nFAISS 索引已保存: {index_path} ({index_path.stat().st_size / 1024 / 1024:.1f}MB)")

# --- Save metadata ---
meta = {
    "model": MODEL_NAME,
    "dim": dim,
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "total_chunks": len(all_chunks),
    "total_answers": len(samples),
    "chunks": chunk_meta,
}
meta_path = INDEX_DIR / "doc_meta.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f"元数据已保存: {meta_path}")

# Also save raw answer lookup (answer_idx → original content)
raw_answers = []
for item in samples:
    content = item.get("content", "")
    if len(content) > 5000:
        content = content[:5000]
    raw_answers.append({
        "title": item.get("title", ""),
        "content": content,
        "url": item.get("url", ""),
    })
raw_path = INDEX_DIR / "doc_raw.json"
with open(raw_path, "w", encoding="utf-8") as f:
    json.dump(raw_answers, f, ensure_ascii=False, indent=2)

# --- Quick test ---
print("\n=== 快速验证 ===")
from sklearn.metrics.pairwise import cosine_similarity

test_queries = [
    "程序员裸辞后怎么调整心态",
    "考研英语怎么复习",
]
for q in test_queries:
    q_vec = model.encode([q], normalize_embeddings=True)
    scores, idxs = index.search(q_vec.astype(np.float32), 3)
    print(f"\n查询: {q}")
    for rank, (sid, sc) in enumerate(zip(idxs[0], scores[0])):
        chunk = chunk_meta[sid]
        print(f"  [{rank+1}] {chunk['source_title'][:40]} | "
              f"{chunk['text_preview'][:60]} (score={sc:.3f})")
