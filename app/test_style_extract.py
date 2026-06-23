"""
从知乎语料库取一篇 ~2000 字的回答，调用 DeepSeek API 做 style-extract 分析
"""
import json
import random
import os
from pathlib import Path
from openai import OpenAI

# --- Config ---
API_KEY = "sk-f9ef570af10f4c3abae000b226ed3619"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"  # DeepSeek V4 latest

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

all_answers = load_jsonl(snapshot_dir / "validation.json")

# --- Pick a ~2000 char answer ---
candidates = [a for a in all_answers if 1500 <= len(a.get("content", "")) <= 3000 and len(a.get("title", "")) > 5]
if not candidates:
    candidates = all_answers

sample = random.choice(candidates)
title = sample["title"]
content = sample["content"]
url = sample.get("url", "")

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print(f"=== 选中的回答 ===")
print(f"标题: {title}")
print(f"链接: {url}")
print(f"字数: {len(content)}")
print(f"\n--- 正文预览 ---")
print(content[:500] + "\n...")
print("=" * 60)

# --- Style-extract prompt ---
system_prompt = """你是一位专业的写作风格分析师。你的任务是从一篇文章中提取作者的写作风格特征。

请按以下 12 个维度逐一分析，每个维度用一句话概括。

## 语言维度（6维）

1. **词汇风格** (vocabulary_level)：口语化 / 书面 / 混合？用词偏日常还是专业？
2. **句式节奏** (sentence_rhythm)：短句为主还是长句？节奏急促还是舒缓？
3. **常用表达** (favorite_expressions)：反复出现的口头禅、过渡语、语气词（提取 3-5 个）
4. **标点习惯** (punctuation_habits)：破折号多还是省略号多？逗号密度？感叹号使用频率？
5. **人称视角** (person_perspective)：第一/第二/第三人称？是否切换？
6. **情绪强度** (emotion_intensity)：冷静克制 / 中等 / 热烈煽情？

## 结构维度（6维）

7. **开头方式** (opening_pattern)：故事/提问/数据/场景/直接亮观点？
8. **段落节奏** (paragraph_rhythm)：段落长短交替规律？关键句是否独立成段？
9. **论证逻辑** (argument_logic)：归纳型（案例→观点）/ 演绎型（观点→论证）/ 并列型？
10. **过渡方式** (transition_style)：连接词/口语化/直接跳转？
11. **结尾方式** (closing_pattern)：金句/开放提问/行动号召/故事回环？
12. **标题模式** (title_pattern)：反常识/数字/痛点/悬念/直给？

## 输出格式

请用中文输出，格式如下：

## 风格分析报告

**来源：** 《标题》 by 作者（如有）

### 语言特征
- **词汇风格：** ...
- **句式节奏：** ...
- **常用表达：** ...
- **标点习惯：** ...
- **人称视角：** ...
- **情绪强度：** ...

### 结构特征
- **开头方式：** ...
- **段落节奏：** ...
- **论证逻辑：** ...
- **过渡方式：** ...
- **结尾方式：** ...
- **标题模式：** ...

"""

user_prompt = f"""请分析以下知乎回答的写作风格：

## 标题
{title}

## 正文
{content}"""

# --- Call API ---
print("\n正在调用 DeepSeek API 进行分析...\n")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    temperature=0.3,
    max_tokens=2000,
)

result = response.choices[0].message.content

# Save to file
output_path = Path(__file__).parent / "style_analysis_result.md"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"# 风格分析报告\n\n")
    f.write(f"**标题：** {title}\n\n")
    f.write(f"**字数：** {len(content)}\n\n")
    f.write(f"**链接：** {url}\n\n")
    f.write("---\n\n")
    f.write(result)
    f.write(f"\n\n---\nToken 使用: {response.usage.total_tokens if response.usage else 'N/A'}\n")

print(f"分析结果已保存到: {output_path}")
print(f"Token 使用: {response.usage.total_tokens if response.usage else 'N/A'}")
