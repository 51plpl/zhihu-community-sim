"""
从知乎语料库随机抽 100 条（>1000 字），批量做 style-extract 分析，入库建风格素材库。
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
SAMPLE_SIZE = 100
MAX_WORKERS = 5  # 并发数
MIN_LENGTH = 1000

# --- Paths ---
BASE_DIR = Path(__file__).parent.parent
STYLES_DIR = BASE_DIR / "styles"
ENTRIES_DIR = STYLES_DIR / "entries"
ENTRIES_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = STYLES_DIR / "index.jsonl"

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

# --- Style-extract prompt (JSON output) ---
SYSTEM_PROMPT = """你是一位专业的写作风格分析师。分析文章的写作风格，输出结构化 JSON。

分析以下 12 个维度，每个维度用一句话概括特征：

## 语言维度
1. vocabulary_level: 口语化/书面/混合？用词偏日常还是专业？
2. sentence_rhythm: 短句为主还是长句？节奏急促还是舒缓？
3. favorite_expressions: 反复出现的口头禅、过渡语、语气词（提取 3-5 个）
4. punctuation_habits: 破折号多还是省略号多？逗号密度？感叹号使用频率？
5. person_perspective: 第一/第二/第三人称？是否切换？
6. emotion_intensity: 冷静克制/中等/热烈煽情？

## 结构维度
7. opening_pattern: 故事/提问/数据/场景/直接亮观点？
8. paragraph_rhythm: 段落长短交替规律？关键句是否独立成段？
9. argument_logic: 归纳型（案例→观点）/ 演绎型（观点→论证）/ 并列型？
10. transition_style: 连接词/口语化/直接跳转？
11. closing_pattern: 金句/开放提问/行动号召/故事回环？
12. title_pattern: 反常识/数字/痛点/悬念/直给？

## 输出格式

只输出 JSON，不要其他文字。JSON 结构：
{
  "language": {
    "vocabulary_level": "一句话概括",
    "sentence_rhythm": "一句话概括",
    "favorite_expressions": ["表达1", "表达2", "表达3"],
    "punctuation_habits": "一句话概括",
    "person_perspective": "一句话概括",
    "emotion_intensity": "一句话概括"
  },
  "structure": {
    "opening_pattern": "一句话概括",
    "paragraph_rhythm": "一句话概括",
    "argument_logic": "一句话概括",
    "transition_style": "一句话概括",
    "closing_pattern": "一句话概括",
    "title_pattern": "一句话概括"
  }
}"""

# --- Load existing index for ID numbering ---
existing_ids = []
if INDEX_PATH.exists():
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                existing_ids.append(json.loads(line)["id"])

existing_entries = list(ENTRIES_DIR.glob("sty_*.json"))
today = datetime.now().strftime("%Y%m%d")

# Determine starting number
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
counter = max_num + 1

# --- Analyze single answer ---
def analyze_one(item):
    global counter
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=60)
    title = item.get("title", "")
    content = item.get("content", "")
    url = item.get("url", "")

    user_prompt = f"## 标题\n{title}\n\n## 正文\n{content}"

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=1500,
                response_format={"type": "json_object"},
            )
            result_text = resp.choices[0].message.content
            analysis = json.loads(result_text)
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
                continue
            return None, str(e)

    # Build entry
    entry_id = f"sty_{today}_{counter:03d}"
    entry = {
        "id": entry_id,
        "source": {
            "title": title,
            "url": url,
            "is_self": False,
        },
        "language": {
            "vocabulary_level": analysis.get("language", {}).get("vocabulary_level", ""),
            "sentence_rhythm": analysis.get("language", {}).get("sentence_rhythm", ""),
            "favorite_expressions": analysis.get("language", {}).get("favorite_expressions", []),
            "punctuation_habits": analysis.get("language", {}).get("punctuation_habits", ""),
            "person_perspective": analysis.get("language", {}).get("person_perspective", ""),
            "emotion_intensity": analysis.get("language", {}).get("emotion_intensity", ""),
        },
        "structure": {
            "opening_pattern": analysis.get("structure", {}).get("opening_pattern", ""),
            "paragraph_rhythm": analysis.get("structure", {}).get("paragraph_rhythm", ""),
            "argument_logic": analysis.get("structure", {}).get("argument_logic", ""),
            "transition_style": analysis.get("structure", {}).get("transition_style", ""),
            "closing_pattern": analysis.get("structure", {}).get("closing_pattern", ""),
            "title_pattern": analysis.get("structure", {}).get("title_pattern", ""),
        },
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
        "analyzed_at": entry["analyzed_at"],
    }
    with open(INDEX_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(index_entry, ensure_ascii=False) + "\n")

    counter += 1
    return entry_id, title[:40]

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
            total_tokens += result[0].get("token_usage", 0) if isinstance(result[0], dict) else 0
            print(f"[{i}/{SAMPLE_SIZE}] ✓ {result[1]} -> {result[0]}")
        else:
            fail += 1
            print(f"[{i}/{SAMPLE_SIZE}] ✗ 失败: {result[1] if result else 'unknown'}")

elapsed = time.time() - start_time
print(f"\n=== 完成 ===")
print(f"成功: {success}, 失败: {fail}")
print(f"总耗时: {elapsed:.0f}s, 平均: {elapsed/max(success,1):.1f}s/条")
print(f"总 tokens: {total_tokens}")
print(f"索引位置: {INDEX_PATH}")
print(f"条目目录: {ENTRIES_DIR}")
