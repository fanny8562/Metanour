# AI 老中医养生功法陪伴系统

面向养生科普的中医智能问答系统：以 RAG 检索增强为核心，提供体质自测、
多轮问诊辨证、辨证-功法匹配（八段锦/五禽戏/易筋经/六字诀）、养生打卡
与轻社群，形成"辨证 → 方案 → 坚持养成"闭环。

定位：研究导向项目，用于留学申请实践经历与开源沉淀；免费普惠，不做
诊疗与疗效承诺。

## 技术栈
- 后端：Python / FastAPI
- RAG：LangChain + ChromaDB + LLM API
- 数据：JSONL 语料库 + 评测集
- 存储：SQLite → PostgreSQL（后续）

## 目录结构
- data/raw       原始语料（不入库）
- data/processed 清洗分块后的语料（JSONL）
- data/eval      评测集（JSONL）
- scripts        清洗/构建脚本
- app            应用代码（第二阶段起）