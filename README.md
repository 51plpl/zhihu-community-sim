# zhihu-community-sim

输入一个问题，生成多条不同身份的知乎风格回答。每条回答来自一个不同的虚拟身份，有各自的职业、经历和写法。

## 安装

```bash
pip install -r requirements.txt
```

## 设置 API Key

```bash
# Windows
set DEEPSEEK_API_KEY=sk-your-key-here

# Mac / Linux
export DEEPSEEK_API_KEY=sk-your-key-here
```

需要 DeepSeek API key，[点这里申请](https://platform.deepseek.com/)。

## 快速开始

```bash
# 启动 Web 界面
python -m app.backend

# 访问 http://127.0.0.1:8080
```

首次请求会加载模型和索引，需要约 30 秒。之后正常速度。

## 命令行

```bash
python app/generate.py "你的问题" --count 5
```

## 文件结构

```
├── app/
│   ├── generate.py        # 生成 pipeline
│   ├── backend.py          # Web 后端
│   └── static/             # Web 前端
├── data/
│   ├── identity.index      # 身份库 FAISS 索引
│   ├── identity_meta.json  # 身份元数据
│   ├── doc.index           # 文档库 FAISS 索引
│   ├── doc_meta.json       # 文档元数据
│   └── doc_raw.json        # 原始知乎回答
├── identities/entries/     # 848 条虚拟身份（70% 18-35 岁）
└── styles/entries/         # 99 条风格分析
```


## 注意

身份库中的 persona 由 LLM 从真实知乎回答中自动提取，均为虚构画像，不指向任何真实个人。
