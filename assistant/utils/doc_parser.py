import os
import re
import docx

def clean_text(text: str) -> str:
    # 去掉连续空格，只保留一个
    text = re.sub(r'[ \t]+', ' ', text)
    # 去掉多余空行，最多保留一个空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def parse_txt(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def parse_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

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
