# 🌿 Metanour · AI 老中医养生功法陪伴系统

基于 RAG 检索增强生成的中医养生智能问答系统，融合体质自测与功法推荐，面向大学生与上班族的日常养生科普。

> ⚠️ 本项目仅供养生科普学习，不构成医疗诊断建议，如有不适请及时就医。

## ✨ 功能特性

- **智能问答**：基于中医养生语料库的 RAG 问答，检索准确率 90%（10 条评测集）
- **体质自测**：9 道题快速判断体质倾向（气虚/阳虚/阴虚/平和）
- **功法推荐**：根据体质匹配适合的传统养生功法（八段锦、五禽戏等）
- **饮食建议**：针对体质给出饮食宜忌

## 🛠 技术栈

| 模块 | 技术选型 |
|------|---------|
| 向量检索 | ChromaDB + BGE-small-zh-v1.5（中文语义 Embedding） |
| 大模型生成 | DeepSeek Chat API |
| Web 界面 | Streamlit |
| 数据处理 | Python + jieba 分词 |
| 版本管理 | Git + GitHub |

## 📁 项目结构

```
tcm-ai/
├── data/
│   ├── raw/           # 原始语料（7 份，含经典、功法、体质、穴位、食疗）
│   ├── processed/     # 清洗分块后的语料（chunks.jsonl，17 块）
│   └── eval/          # 评测集（eval_set.jsonl，10 条）
├── scripts/
│   ├── clean_text.py  # 语料清洗与分块脚本
│   └── eval_rag.py    # RAG 检索准确率评测脚本
├── app/
│   ├── rag.py         # RAG 核心（命令行版）
│   └── web_app.py     # Streamlit Web 界面
├── requirements.txt   # 依赖清单
└── .env               # API Key 配置（不入库）
```

## 🚀 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. 配置 API Key

在项目根目录创建 `.env` 文件：
```
DEEPSEEK_API_KEY=sk-你的DeepSeek密钥
```

### 3. 构建语料库

```bash
python scripts/clean_text.py
```

### 4. 运行 Web 界面

```bash
streamlit run app/web_app.py
```

浏览器自动打开 `http://localhost:8501`

### 5. 运行评测

```bash
python scripts/eval_rag.py
```

## 📊 评测结果

在 10 条人工标注的评测问题上：

| 分类 | 准确率 |
|------|--------|
| 经典 | 100% (4/4) |
| 功法 | 50% (1/2) |
| 中医常识 | 100% (4/4) |
| **总体** | **90% (9/10)** |

## 🗺 路线图

- [x] RAG 核心问答（向量检索 + 大模型生成）
- [x] Web 界面（Streamlit）
- [x] 体质自测 + 功法推荐
- [ ] 多轮问诊辨证
- [ ] 养生打卡养成
- [ ] 轻社群功能

## 📄 许可证

本项目仅供学习研究使用，语料版权归原作者所有。
