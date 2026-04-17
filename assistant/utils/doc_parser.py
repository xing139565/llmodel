import os
import re
import docx
import pandas as pd
import json

def clean_text(text: str) -> str:
    # 去掉连续空格，只保留一个
    text = re.sub(r'[ \t]+', ' ', text)
    # 去掉多余空行，最多保留一个空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def format_routes_markdown(node, level=1):
    md_lines = []
    prefix = "#" * level
    name = node.get("name", node.get("title", "未命名菜单"))
    path = node.get("path", "无路径")
    
    md_lines.append(f"{prefix} 菜单名称: {name} | 路由路径: {path}")
    
    if "children" in node and isinstance(node["children"], list):
        for child in node["children"]:
            md_lines.extend(format_routes_markdown(child, min(level + 1, 6)))
    return md_lines

def parse_txt(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # 尝试智能探测：如果是披着 .md/.txt 外衣的纯 JSON 配置信息，转化为语义结构
    try:
        data = json.loads(content)
        # 如果是路由性质的数组
        if isinstance(data, list) and len(data) > 0 and ("name" in data[0] or "path" in data[0] or "alias" in data[0] or "children" in data[0]):
            lines = ["本文件是系统各个菜单、路由的层级结构配置：\n"]
            for item in data:
                lines.extend(format_routes_markdown(item, 1))
            return "\n\n".join(lines)
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return content

def parse_docx(file_path: str) -> str:
    try:
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        print(f"Error parsing docx {file_path}: {e}")
        return ""

def parse_excel(file_path: str) -> str:
    try:
        xls = pd.read_excel(file_path, sheet_name=None)
        texts = []
        for sheet_name, df in xls.items():
            df = df.dropna(how="all").dropna(axis=1, how="all")
            if df.empty:
                continue

            texts.append(f"--- Sheet: {sheet_name} ---")

            # 只取前 100 行，避免巨量噪声
            preview_df = df.head(100).astype(str)
            texts.append(preview_df.to_string(index=False))
        return "\n".join(texts)
    except Exception as e:
        print(f"Error parsing excel {file_path}: {e}")
        return ""

def parse_json(file_path: str) -> str:
    return parse_txt(file_path)

def parse_file(file_path: str) -> dict:
    source = os.path.basename(file_path)
    # 分类取自父文件夹的名称
    category = os.path.basename(os.path.dirname(file_path))
    ext = os.path.splitext(source)[1].lower()
    
    text = ""
    if ext in ['.txt', '.md']:
        text = parse_txt(file_path)
    elif ext == '.docx':
        text = parse_docx(file_path)
    elif ext in ['.xlsx', '.xls']:
        text = parse_excel(file_path)
    elif ext == '.json':
        text = parse_json(file_path)
    else:
        return None
        
    text = clean_text(text)
    
    if not text:
        return None
        
    return {
        "source": source,
        "category": category,
        "path": file_path,
        "text": text
    }

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list:
    seps = ["\n\n", "\n", "。", "！", "？", "；", "，", " "]
    return _recursive_split(text, seps, chunk_size, chunk_overlap)

def _recursive_split(text: str, separators: list, chunk_size: int, chunk_overlap: int) -> list:
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    # Find the best separator
    separator_idx = -1
    for i, sep in enumerate(separators):
        if sep == "":
            separator_idx = i
            break
        if sep in text:
            separator_idx = i
            break

    if separator_idx == -1:
        # No separator found, forcefully split by characters
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunks.append(text[i:i + chunk_size])
        return chunks

    separator = separators[separator_idx]
    
    splits = text.split(separator)
    docs = []
    
    for i, s in enumerate(splits):
        if i < len(splits) - 1:
            docs.append(s + separator)
        else:
            if s:
                docs.append(s)

    chunks = []
    current_doc = []
    current_len = 0

    for d in docs:
        d_len = len(d)
        if current_len + d_len > chunk_size and current_len > 0:
            chunk_str = "".join(current_doc).strip()
            if chunk_str:
                chunks.append(chunk_str)
            
            # Start a new chunk, but overlap from existing current_doc
            while current_len > chunk_overlap and len(current_doc) > 0:
                removed_doc = current_doc.pop(0)
                current_len -= len(removed_doc)

        current_doc.append(d)
        current_len += d_len

    chunk_str = "".join(current_doc).strip()
    if chunk_str:
        chunks.append(chunk_str)

    # Post processing for oversized chunks
    final_chunks = []
    next_separators = separators[separator_idx + 1:] if separator_idx + 1 < len(separators) else []
    
    for c in chunks:
        if len(c) > chunk_size:
            if next_separators:
                final_chunks.extend(_recursive_split(c, next_separators, chunk_size, chunk_overlap))
            else:
                for i in range(0, len(c), chunk_size - chunk_overlap):
                    final_chunks.append(c[i:i + chunk_size])
        else:
            final_chunks.append(c)

    return final_chunks
