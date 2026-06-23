"""
从知乎语料库随机抽 1000 条（>1000 字），批量做 identity-extract 分析，入库建身份画像库。
"""
import json
import random
import time
import sys
import io
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# --- Config ---
API_KEY = "sk-f9ef570af10f4c3abae000b226ed3619"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"
SAMPLE_SIZE = 1000
MAX_WORKERS = 10
MIN_LENGTH = 1000

# --- Paths ---
BASE_DIR = Path(__file__).parent.parent
IDENTITIES_DIR = BASE_DIR / "identities"
ENTRIES_DIR = IDENTITIES_DIR / "entries"
ENTRIES_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = IDENTITIES_DIR / "index.jsonl"

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

# Filter and sample
long_answers = [a for a in all_answers if len(a.get("content", "")) > MIN_LENGTH]
print(f">1000 字的有 {len(long_answers)} 条")

random.shuffle(long_answers)
samples = long_answers[:SAMPLE_SIZE]
print(f"随机抽取 {len(samples)} 条\n")

# --- Identity-extract prompt ---
SYSTEM_PROMPT = """你是一名社区用户画像生成器（Community Persona Generator）。

你的任务是根据给定文本，推测作者可能具备的身份特征、人生经历、兴趣领域、价值倾向、认知风格和社区行为特征。

目标不是还原真实作者，而是生成一个合理、丰满、适合作为社区成员模拟的虚拟人物画像。

## 规则
1. 允许推测和补全信息。
2. 允许根据文本特征进行合理脑补。
3. 不要求真实。
4. 不要求与原作者完全一致。
5. 目标是生成一个可信且具有鲜明特点的人。
6. 如果存在多个合理解释，可以随机选择一种。
7. 允许加入少量随机性，使相似文本不会总是得到相同画像。
8. 不要输出分析过程。
9. 不要解释原因。
10. 直接输出JSON。
11. 各字段尽量填写，不要大量留空。
12. 画像需要能够用于后续LLM角色扮演和社区模拟。
13. **所有内容使用中文输出，包括字段值、列表项、示例等。**

## 输出格式
{
  "persona": {
    "demographics": {
      "age_range": "",
      "gender": "",
      "education": "",
      "occupation": "",
      "industry": "",
      "city_tier": "",
      "marital_status": "",
      "children": "",
      "income_level": ""
    },
    "life_stage": [],
    "life_experiences": [],
    "interest_topics": [],
    "discussion_triggers": [],
    "sensitive_topics": [],
    "value_tendencies": [],
    "worldview": {
      "rational_vs_emotional": 0,
      "optimistic_vs_pessimistic": 0
    },
    "thinking_style": [],
    "knowledge_profile": {
      "expertise_fields": [],
      "general_knowledge_level": 0,
      "likes_citing_data": 0,
      "likes_personal_experience": 0
    },
    "community_behavior": {
      "activity_level": 0,
      "likes_answering": 0,
      "likes_commenting": 0,
      "likes_debating": 0,
      "likes_contrarian": 0,
      "likes_long_answers": 0,
      "likes_short_answers": 0,
      "likes_storytelling": 0,
      "likes_explaining": 0,
      "likes_teaching": 0,
      "likes_showing_expertise": 0,
      "likes_sharing_experience": 0,
      "likes_emotional_expression": 0,
      "likes_logical_analysis": 0,
      "likes_hot_topics": 0,
      "likes_niche_topics": 0
    },
    "motivation": [],
    "social_needs": {
      "seeking_recognition": 0,
      "seeking_agreement": 0,
      "seeking_influence": 0,
      "seeking_help_others": 0,
      "seeking_self_expression": 0
    },
    "answer_probability_topics": [
      {"topic": "", "score": 0}
    ],
    "impression": ""
  }
}

## 字段说明
- **interest_topics**: 长期关注的话题。示例：["心理学", "公务员考试", "AI Agent", "游戏运营", "教育"]
- **discussion_triggers**: 容易主动发言的话题。示例：["考公", "AI取代程序员", "游戏运营争议", "男女关系"]
- **sensitive_topics**: 容易引发情绪或强烈观点的话题。
- **thinking_style**: 示例：["经验主义", "机制分析", "数据导向", "故事叙述", "情绪表达", "价值判断"]
- **motivation**: 示例：["帮助别人", "展示专业能力", "表达观点", "获得认同", "反驳他人", "分享经历"]
- **worldview**（0-100，0=完全偏向左侧，100=完全偏向右侧）：
  - rational_vs_emotional: 0=理性驱动，100=情感驱动
  - optimistic_vs_pessimistic: 0=悲观，100=乐观
- **impression**: 用一句话概括这个人的整体感觉。不是总结身份特征，而是读这段文本时的"人味儿"印象。示例："一个被社会毒打过但还没躺平的理想主义者"、"冷静克制的理中客，但偶尔流露优越感"、"热情爱分享的过来人\""""

# --- Load existing index for ID numbering ---
existing_ids = []
if INDEX_PATH.exists():
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                existing_ids.append(json.loads(line)["id"])

existing_entries = list(ENTRIES_DIR.glob("idy_*.json"))
today = datetime.now().strftime("%Y%m%d")

max_num = 0
for eid in existing_ids:
    try:
        num = int(eid.split("_")[-1])
        max_num = max(max_num, num)
    except (IndexError, ValueError):
        pass
for ep in existing_entries:
    try:
        parts = ep.stem.split("_")
        num = int(parts[-1])
        max_num = max(max_num, num)
    except (IndexError, ValueError):
        pass
from threading import Lock
counter = max_num + 1
counter_lock = Lock()


# --- Analyze single answer ---
def analyze_one(item):
    global counter
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120)
    title = item.get("title", "")
    content = item.get("content", "")
    url = item.get("url", "")

    # Truncate very long content
    if len(content) > 5000:
        content = content[:5000] + "\n\n[（因长度限制，后续内容已截断）]"

    user_prompt = f"## 标题\n{title}\n\n## 正文\n{content}"

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.5,
                max_tokens=3000,
                response_format={"type": "json_object"},
            )
            result_text = resp.choices[0].message.content
            result = json.loads(result_text)
            persona = result.get("persona", result)
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
                continue
            return None, str(e)

    with counter_lock:
        entry_id = f"idy_{today}_{counter:04d}"
        counter += 1

    entry = {
        "id": entry_id,
        "source": {
            "title": title[:80],
            "url": url,
            "is_self": False,
        },
        "persona": persona,
        "analyzed_at": datetime.now().isoformat(),
        "token_usage": resp.usage.total_tokens if resp.usage else 0,
    }

    # Write entry file
    entry_path = ENTRIES_DIR / f"{entry_id}.json"
    with open(entry_path, "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)

    # Append to index
    index_entry = {
        "id": entry_id,
        "source_title": title[:80],
        "impression": persona.get("impression", ""),
        "analyzed_at": entry["analyzed_at"],
    }
    with open(INDEX_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(index_entry, ensure_ascii=False) + "\n")

    return entry_id, title[:40], resp.usage.total_tokens if resp.usage else 0


# --- Run batch ---
print(f"开始批量分析 {len(samples)} 条（并发 {MAX_WORKERS} 线程）...\n")
start_time = time.time()

success = 0
fail = 0
total_tokens = 0

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(analyze_one, item): item for item in samples}
    for i, future in enumerate(as_completed(futures), 1):
        result = future.result()
        if result and result[0]:
            success += 1
            total_tokens += result[2]
            print(f"[{i}/{SAMPLE_SIZE}] ✓ {result[1]} -> {result[0]}")
        else:
            fail += 1
            print(f"[{i}/{SAMPLE_SIZE}] ✗ 失败: {result[1] if result else 'unknown'}")

        # Progress summary every 100
        if i % 100 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed * 60
            print(f"  → 进度 {i}/{SAMPLE_SIZE} | 成功 {success} 失败 {fail} | "
                  f"已用 {elapsed:.0f}s | 速率 {rate:.0f}条/分")

elapsed = time.time() - start_time
print(f"\n=== 完成 ===")
print(f"成功: {success}, 失败: {fail}")
print(f"总耗时: {elapsed:.0f}s ({elapsed/60:.1f}min), 平均: {elapsed/max(success,1):.1f}s/条")
print(f"总 tokens: {total_tokens}")
print(f"平均 tokens/条: {total_tokens//max(success,1)}")
print(f"索引位置: {INDEX_PATH}")
print(f"条目目录: {ENTRIES_DIR}")
