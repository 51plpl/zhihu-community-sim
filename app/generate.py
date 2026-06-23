"""
生成 pipeline：RAG1 找人 → RAG2 带视角搜案例 → 组装 prompt → 调 DeepSeek 生成

用法：
  python app/generate.py "你的问题"
  python app/generate.py "你的问题" --count 3
  python app/generate.py "你的问题" --identity 2 --chunks 2
"""
import json
import sys
import io
import random
import argparse
import os
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).parent.parent
INDEX_DIR = BASE_DIR / "data"
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not API_KEY:
    raise SystemExit("请设置环境变量 DEEPSEEK_API_KEY，例如: export DEEPSEEK_API_KEY=sk-xxx")
MODEL = "deepseek-chat"


# ==================== Load indexes ====================

def load_index(name):
    """加载 FAISS index（绕过中文路径问题）"""
    import faiss
    import numpy as np
    path = INDEX_DIR / f"{name}.index"
    with open(path, "rb") as f:
        data = f.read()
    # FAISS deserialize 需要 numpy array
    return faiss.deserialize_index(np.frombuffer(data, dtype="uint8"))


def load_meta(name):
    path = INDEX_DIR / f"{name}_meta.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_raw():
    path = INDEX_DIR / "doc_raw.json"
    default = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


print("加载索引中...")
identity_index = load_index("identity")
identity_meta = load_meta("identity")
doc_index = load_index("doc")
doc_meta = load_meta("doc")
doc_raw = load_raw()
print(f"身份库: {identity_meta['total']} 条")
print(f"文档库: {doc_meta['total_chunks']} 个块 ({doc_meta['total_answers']} 条回答)")


# ==================== BM25 index (hybrid retrieval) ====================

print("构建 BM25 索引...")
import jieba
from rank_bm25 import BM25Okapi
import numpy as np

# Build BM25 over raw answers (for keyword matching)
bm25_raw_texts = [item.get("content", "")[:5000] for item in doc_raw]
bm25_tokenized = [list(jieba.cut(t)) for t in bm25_raw_texts]
bm25_index = BM25Okapi(bm25_tokenized)
# Build answer_idx → chunk_idxs lookup
answer_to_chunks = {}
for ci, chunk in enumerate(doc_meta["chunks"]):
    aidx = chunk["answer_idx"]
    answer_to_chunks.setdefault(aidx, []).append(ci)
print(f"BM25 索引: {len(bm25_raw_texts)} 条回答\n")


# ==================== Embedding model ====================

print("加载嵌入模型...")
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(identity_meta["model"], device="cpu")


def encode(texts):
    return model.encode(texts, normalize_embeddings=True)


# ==================== Style library ====================


def load_style_library():
    styles_dir = BASE_DIR / 'styles' / 'entries'
    entries = []
    for path in sorted(styles_dir.glob('*.json')):
        with open(path, 'r', encoding='utf-8') as f:
            entries.append(json.load(f))
    return entries

style_entries = load_style_library()


def pick_style():
    entry = random.choice(style_entries)
    lang = entry.get('language', {})
    parts = []
    for k, label in [('sentence_rhythm', '句式'), ('vocabulary_level', '用词'), ('emotion_intensity', '语气')]:
        v = lang.get(k, '')
        if v:
            parts.append(f'{label}: {v}')
    exprs = lang.get('favorite_expressions', [])
    if exprs:
        parts.append(f"爱用: {' '.join(exprs[:4])}")
    return ' | '.join(parts)


print(f'风格库: {len(style_entries)} 条')


def search_identities(query, top_k=3, exclude_ids=None):
    q_vec = encode([query])
    scores, idxs = identity_index.search(q_vec.astype("float32"), top_k * 3)
    results = []
    for sid, sc in zip(idxs[0], scores[0]):
        p = identity_meta["personas"][sid]
        if exclude_ids and p["id"] in exclude_ids:
            continue
        results.append({
            "score": float(sc),
            "persona": p,
        })
        if len(results) >= top_k:
            break
    return results


# ==================== RAG2: Find documents ====================

def search_documents(query, top_k=3):
    """Hybrid retrieval: dense + BM25, fused via RRF"""
    K = 60

    # Dense search
    q_vec = encode([query])
    dense_scores, dense_idxs = doc_index.search(q_vec.astype("float32"), top_k * 3)

    # BM25 search
    tokenized_query = list(jieba.cut(query))
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_top = np.argsort(bm25_scores)[-top_k * 3:][::-1]

    # RRF fusion
    rrf = {}
    for rank, cid in enumerate(dense_idxs[0]):
        rrf[int(cid)] = rrf.get(int(cid), 0) + 1 / (rank + K)
    for rank, aidx in enumerate(bm25_top):
        if bm25_scores[aidx] <= 0:
            continue
        for cid in answer_to_chunks.get(int(aidx), []):
            rrf[cid] = rrf.get(cid, 0) + 1 / (rank + K)

    sorted_chunks = sorted(rrf.items(), key=lambda x: -x[1])[:top_k]

    results = []
    for cid, score in sorted_chunks:
        info = doc_meta["chunks"][cid]
        raw = doc_raw[info["answer_idx"]] if info["answer_idx"] < len(doc_raw) else None
        results.append({
            "score": round(float(score), 4),
            "source_title": info["source_title"],
            "chunk_text": raw["content"] if raw else "",
            "full_title": raw["title"] if raw else "",
        })
    return results


def build_persona_context(persona):
    """从身份数据拼出供 RAG2 检索用的上下文"""
    d = persona.get("demographics", {})
    parts = [
        d.get("occupation", ""),
        d.get("industry", ""),
        "兴趣: " + ", ".join(persona.get("interest_topics", [])),
        "经历: " + ", ".join(persona.get("life_experiences", [])),
    ]
    return " | ".join(p for p in parts if p)


# ==================== Prompt assembly ====================


def build_prompt(question, persona, chunks):
    """组装最终 prompt，参考用户提供的约束结构"""
    p = persona

    # Persona description
    d = p.get("demographics", {})
    parts = []
    age = d.get("age_range", "")
    gender = d.get("gender", "")
    occ = d.get("occupation", "")
    edu = d.get("education", "")
    industry = d.get("industry", "")
    city = d.get("city_tier", "")
    if age and gender and occ:
        parts.append(f"{age}的{gender}{occ}")
    elif occ:
        parts.append(occ)
    if edu:
        parts.append(edu)
    if industry:
        parts.append(f"{industry}行业")
    if city:
        parts.append(f"生活在{city}")
    experiences = p.get("life_experiences", [])
    if experiences:
        parts.append("经历过" + "、".join(experiences[:3]))
    interests = p.get("interest_topics", [])
    if interests:
        parts.append("平时关注" + "、".join(interests[:5]))

    # Values & worldview
    values = p.get("value_tendencies", [])
    if values:
        parts.append("、" .join(values[:3]))
    wv = p.get("worldview", {})
    wv_parts = []
    re = wv.get("rational_vs_emotional", 0)
    if re:
        wv_parts.append("偏理性" if re < 40 else "偏感性" if re > 60 else "理性感性各半")
    op = wv.get("optimistic_vs_pessimistic", 0)
    if op:
        wv_parts.append("偏乐观" if op > 60 else "偏悲观" if op < 40 else "")
    if wv_parts:
        parts.append("、".join(w for w in wv_parts if w))

    # Motivation & sensitive topics
    motivation = p.get("motivation", [])
    if motivation:
        parts.append("来这里主要是" + "、".join(motivation[:3]))
    sensitive = p.get("sensitive_topics", [])
    if sensitive:
        parts.append("容易激动的话题：" + "、".join(sensitive[:3]))

    # Knowledge profile
    kp = p.get("knowledge_profile", {})
    kp_parts = []
    exp_fields = kp.get("expertise_fields", [])
    if exp_fields:
        kp_parts.append("擅长" + "、".join(exp_fields[:3]))
    cite = kp.get("likes_citing_data", -1)
    if cite >= 70:
        kp_parts.append("喜欢用数据说话")
    elif cite >= 0:
        pass  # neutral, skip
    pe = kp.get("likes_personal_experience", -1)
    if pe >= 70:
        kp_parts.append("喜欢用亲身经历说话")
    if kp_parts:
        parts.append("，".join(kp_parts))

    thinking = p.get("thinking_style", [])
    if thinking:
        parts.append("、" .join(thinking[:3]))
    persona_text = "，".join(parts)
    impression = p.get("impression", "")
    if impression and persona_text:
        persona_text += "。" + impression

    # Reference texts (200 chars each for more diverse语感 samples)
    ref_lines = []
    for i, c in enumerate(chunks[:3], 1):
        ref_lines.append(f"--- 参考 {i} ---")
        ref_lines.append(c["chunk_text"][:200])
        ref_lines.append("")
    ref_text = "\n".join(ref_lines)

    # Style hint
    style_hint = pick_style()

    prompt = f"""你在知乎看到一个问题：

{question}

写一个回答，达到知乎的感觉，仿佛是真实的回答，给题主提供需要的情绪价值。

------------------------

【角色约束】

你是一个{persona_text}

你的表达来自真实生活经验，而不是课堂总结或条目式讲解。

写作时参考以下风格感觉：{style_hint}

------------------------

【写作硬约束】

必须满足：

- 不使用"首先/其次/最后"
- 不使用教科书式分点结构
- 每一段只推进一个核心信息点
- 不要把逻辑解释得过于完整或对称
- 不要写成清单、指南、模板化建议
- 允许适度冗余和自然转折

以下是参考的知乎文本示例，可以用于把握相关感觉：

{ref_text}"""

    return prompt


# ==================== Generate ====================

def generate_answer(question, identity_count=2, chunks_per_person=2, temperature=0.7, exclude_ids=None, temp_range=(0.6, 0.95)):
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

    identities = search_identities(question, top_k=identity_count, exclude_ids=exclude_ids)
    print(f"找到 {len(identities)} 个潜在回答者\n")

    answers = []

    for rank, hit in enumerate(identities, 1):
        # Random temperature per persona for more diversity
        t = random.uniform(*temp_range) if isinstance(temp_range, (list, tuple)) else temperature

        print(f"── 回答者 {rank} (temp={t:.2f}) ──")
        print(f"   印象: {hit['persona']['impression'][:60]}")
        print(f"   匹配分: {hit['score']:.3f}")

        # Load full persona
        pid = hit["persona"]["id"]
        entry_path = BASE_DIR / "identities" / "entries" / f"{pid}.json"
        if entry_path.exists():
            with open(entry_path, "r", encoding="utf-8") as f:
                full_persona = json.load(f)["persona"]
        else:
            full_persona = hit["persona"]

        # Search docs with persona perspective
        persona_context = build_persona_context(full_persona)
        query = f"{question} {persona_context}"
        chunks = search_documents(query, top_k=chunks_per_person)
        print(f"   搜索到 {len(chunks)} 个参考片段")

        for c in chunks:
            print(f"     - {c['source_title'][:50]} (score={c['score']:.3f})")

        # Build prompt and generate
        prompt = build_prompt(question, full_persona, chunks)
        # Truncate if too long
        if len(prompt) > 12000:
            prompt = prompt[:12000] + "\n\n[参考内容过长已截断]\n"

        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=t,
            max_tokens=2000,
        )

        answer_text = resp.choices[0].message.content
        answers.append({
            "persona_id": pid,
            "persona_impression": full_persona.get("impression", ""),
            "persona_demo": f"{full_persona['demographics'].get('age_range')} {full_persona['demographics'].get('gender')} {full_persona['demographics'].get('occupation')}",
            "answer": answer_text,
            "token_usage": resp.usage.total_tokens if resp.usage else 0,
        })

        print(f"   生成完成 ({resp.usage.total_tokens if resp.usage else 0} tokens)")
        print()

    return answers


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(description="Self-Space 知乎风格回答生成")
    parser.add_argument("question", help="要回答的问题")
    parser.add_argument("--count", "-n", type=int, default=2,
                        help="生成几条回答 (default: 2)")
    parser.add_argument("--chunks", "-c", type=int, default=2,
                        help="每人参考几条案例 (default: 2)")
    parser.add_argument("--temperature", "-t", type=float, default=0.7,
                        help="生成温度 (default: 0.7)")
    args = parser.parse_args()

    print(f"\n问题: {args.question}")
    print(f"生成 {args.count} 条回答, 每人参考 {args.chunks} 条案例\n")
    print("=" * 60)

    answers = generate_answer(
        args.question,
        identity_count=args.count,
        chunks_per_person=args.chunks,
        temperature=args.temperature,
    )

    print("=" * 60)
    print(f"\n=== 共 {len(answers)} 条回答 ===\n")

    for i, a in enumerate(answers, 1):
        print(f"[回答 {i}] {a['persona_demo']}")
        print(f"    印象: {a['persona_impression']}")
        print(f"    {a['answer']}")
        print(f"    (tokens: {a['token_usage']})")
        print()


if __name__ == "__main__":
    main()
