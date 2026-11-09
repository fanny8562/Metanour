r"""
clean_text.py - 语料清洗与分块脚本
遍历 data/raw/ 下所有 .txt，清洗后按固定长度分块，输出 JSONL 到 data/processed/chunks.jsonl

用法（在 D:\tcm-ai 目录下，且 (.venv) 已激活）：
    python scripts\clean_text.py
"""
import json
import re
from pathlib import Path

# ===== 配置 =====
RAW = Path("data/raw")
OUT = Path("data/processed")
CHUNK_SIZE = 600      # 每块约 600 字
OVERLAP = 80          # 相邻两块重叠 80 字，避免切断语义

# 文件名前缀 → 分类
CATEGORY_MAP = {
    "ctext_": "经典",
    "pe_": "功法",
    "tcm_": "中医常识",
}


def detect_category(filename: str) -> str:
    """根据文件名前缀判断分类"""
    for prefix, cat in CATEGORY_MAP.items():
        if filename.startswith(prefix):
            return cat
    return "未分类"


def clean_text(text: str) -> str:
    """清洗文本：去 markdown 符号、去 URL、合并空白"""
    # 去掉 markdown 引用符号 "> "
    text = re.sub(r"^>\s*", "", text, flags=re.M)
    # markdown 链接 [文字](url) -> 保留文字
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # 去掉独立 URL
    text = re.sub(r"https?://\S+", "", text)
    # 合并多余空行和空白
    text = re.sub(r"\n\s*\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP):
    """按固定长度分块，相邻块带 overlap 重叠"""
    chunks = []
    i = 0
    while i < len(text):
        piece = text[i:i + size]
        chunks.append(piece)
        if i + size >= len(text):
            break
        i += size - overlap
    return chunks


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out_file = OUT / "chunks.jsonl"
    total = 0

    with open(out_file, "w", encoding="utf-8") as f:
        for txt in sorted(RAW.glob("*.txt")):
            # 跳过 _README 说明文件
            if txt.name.startswith("_"):
                continue

            raw = txt.read_text(encoding="utf-8")
            # 第一行是 "# 来源：xxx"
            lines = raw.split("\n", 1)
            source_url = lines[0].lstrip("# ").replace("来源：", "").strip()
            body = lines[1] if len(lines) > 1 else ""

            body = clean_text(body)
            category = detect_category(txt.name)
            chunks = chunk_text(body)

            for idx, ch in enumerate(chunks):
                rec = {
                    "id": f"{txt.stem}_{idx:03d}",
                    "source_file": txt.name,
                    "source_url": source_url,
                    "title": txt.stem,
                    "category": category,
                    "chunk_index": idx,
                    "text": ch,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += 1

            print(f"[{txt.name}] 分类={category} -> {len(chunks)} 块")

    print(f"\n完成：共 {total} 块，已写入 {out_file}")


if __name__ == "__main__":
    main()
