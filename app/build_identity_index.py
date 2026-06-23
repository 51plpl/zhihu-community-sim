"""
构建身份库向量索引（RAG1）
加载 1000 条身份画像 → 构建 search_text → 编码 → 存 FAISS 索引
"""
import json
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# --- Config ---
MODEL_NAME = "BAAI/bge-small-zh-v1.5"  # 轻量中文嵌入模型
BASE_DIR = Path(__file__).parent.parent
INDEX_DIR = BASE_DIR / "data"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

IDENTITIES_DIR = BASE_DIR / "identities"
ENTRIES_DIR = IDENTITIES_DIR / "entries"

# --- Load identities ---
print("加载身份画像...")
entries = sorted(ENTRIES_DIR.glob("*.json"))
print(f"找到 {len(entries)} 条")

personas = []
search_texts = []
for path in entries:
    with open(path, "r", encoding="utf-8") as f:
        entry = json.load(f)
    persona = entry["persona"]

    # 拼成搜索文本：包含所有检索入口字段
    parts = [
        persona.get("demographics", {}).get("occupation", ""),
        persona.get("demographics", {}).get("industry", ""),
        "兴趣: " + ", ".join(persona.get("interest_topics", [])),
        "经历: " + ", ".join(persona.get("life_experiences", [])),
        "触发话题: " + ", ".join(persona.get("discussion_triggers", [])),
        "常答话题: " + ", ".join(
            t["topic"] for t in persona.get("answer_probability_topics", [])
        ),
    ]
    search_text = "\n".join(p for p in parts if p)
    search_texts.append(search_text)
    personas.append({
        "id": entry["id"],
        "source_title": entry["source"]["title"],
        "impression": persona.get("impression", ""),
        "search_text": search_text,
    })

print(f"搜索文本示例（第一条前 200 字）:\n{search_texts[0][:200]}\n")

# --- Load embedding model ---
print(f"加载模型 {MODEL_NAME}...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(MODEL_NAME, device="cpu")
print(f"模型维度: {model.get_sentence_embedding_dimension()}")

# --- Encode ---
print(f"编码 {len(search_texts)} 条身份...")
# bge 模型要求 query 加 instruction prefix，但 document 不需要
vectors = model.encode(search_texts, show_progress_bar=True, normalize_embeddings=True)
print(f"向量形状: {vectors.shape}")

# --- Build & save FAISS index ---
import faiss
import numpy as np

dim = vectors.shape[1]
index = faiss.IndexFlatIP(dim)  # Inner Product = cosine similarity (已归一化)
index.add(vectors.astype(np.float32))

# FAISS 的 C++ 后端不支持含中文路径，用 bytes buffer 绕开
index_bytes = faiss.serialize_index(index)
index_path = INDEX_DIR / "identity.index"
with open(index_path, "wb") as f:
    f.write(index_bytes)
print(f"FAISS 索引已保存: {index_path}")

# --- Save metadata (ID ↔ search_text mapping) ---
meta = {
    "model": MODEL_NAME,
    "dim": dim,
    "total": len(personas),
    "personas": personas,
}
with open(INDEX_DIR / "identity_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f"元数据已保存: {INDEX_DIR / 'identity_meta.json'}")

# --- Quick test ---
print("\n=== 快速验证 ===")
test_queries = [
    "程序员裸辞后怎么调整心态",
    "考研英语怎么复习",
    "如何评价新款手机",
]
for q in test_queries:
    q_vec = model.encode([q], normalize_embeddings=True)
    scores, idxs = index.search(q_vec.astype(np.float32), 3)
    print(f"\n查询: {q}")
    for rank, (sid, sc) in enumerate(zip(idxs[0], scores[0])):
        p = personas[sid]
        print(f"  [{rank+1}] {p['impression'][:60]} (score={sc:.3f})")
