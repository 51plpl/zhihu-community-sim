"""
测试 identity-extract：从知乎回答提取虚拟身份画像
"""
import json
import sys
import io
from pathlib import Path
from openai import OpenAI

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

API_KEY = "sk-f9ef570af10f4c3abae000b226ed3619"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

# --- Load dataset ---
cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--suolyer--zhihu"
snapshot_dir = list(cache_dir.glob("snapshots/*/"))[0]

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

all_answers = load_jsonl(snapshot_dir / "validation.json") + load_jsonl(snapshot_dir / "test.json")

# Pick a content-rich answer
target = [a for a in all_answers if a.get("title", "").startswith("程序员裸辞")][0]
title = target["title"]
content = target["content"]
url = target.get("url", "")

print(f"标题: {title}")
print(f"字数: {len(content)}")
print(f"链接: {url}\n")

# --- Identity extraction prompt ---
system_prompt = """你是一名社区用户画像生成器（Community Persona Generator）。

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
- **impression**: 用一句话概括这个人的整体感觉。不是总结身份特征，而是读这段文本时的"人味儿"印象。示例："一个被社会毒打过但还没躺平的理想主义者"、"冷静克制的理中客，但偶尔流露优越感"、"热情爱分享的过来人"""

user_prompt = f"## 标题\n{title}\n\n## 正文\n{content}"

# --- Call API ---
print("调用 DeepSeek API 生成身份画像...\n")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
resp = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    temperature=0.5,
    max_tokens=3000,
    response_format={"type": "json_object"},
)

result = json.loads(resp.choices[0].message.content)
persona = result.get("persona", result)

# Save
output_path = Path(__file__).parent / "identity_test_output.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(persona, f, ensure_ascii=False, indent=2)

# Pretty print
print(json.dumps(persona, ensure_ascii=False, indent=2))
print(f"\nToken 使用: {resp.usage.total_tokens if resp.usage else 'N/A'}")
print(f"已保存到: {output_path}")
