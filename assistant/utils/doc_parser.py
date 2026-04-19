import os
import re
import docx
import pandas as pd
import json

def clean_text(text: str) -> str:
    """清理文本，去除多余的空格和空行"""
    # 将多个连续的空格或制表符替换为一个小小的单个空格
    text = re.sub(r'[ \t]+', ' ', text)
    # 把长长的一大段连续空行（3个以上），合并成最多只留一个空行，保持排版整洁
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 去除首尾多余的空白字符
    return text.strip()

def format_routes_markdown(node, level=1, parent_names=None):
    """专门用来把那种复杂的 JSON 菜单树，翻译成大模型能看懂的层级 Markdown 文字"""
    if parent_names is None:
        parent_names = []
        
    md_lines = []
    prefix = "#" * level # 根据层级决定几个井号（比如 ## 代表第二级菜单）
    name = node.get("name", node.get("title", "未命名菜单")) # 找菜单的名字
    path = node.get("path", "无路径") # 这是这个菜单对应的网页链接路径
    remark = str(node.get("remark") or "").strip() # 备注信息
    alias = str(node.get("alias") or "").strip() # 别名信息
    
    # 拼装出最重要的描述：当前是哪一层、叫什么名字、路径是啥
    desc = f"{prefix} 菜单名称: {name} | 路由路径: {path}"
    
    # 如果有上级，就把它的“老爸老妈老鼻祖”都带上，这样大模型就不会搞混同名菜单
    if parent_names:
        desc += f" | 上级归属: {' > '.join(parent_names)}"
        
    # 过滤掉像 "null" 或者 "none" 这种毫无意义的系统填充词
    extras = []
    if remark and remark.lower() not in ["null", "none"]:
        extras.append(remark)
    if alias and alias.lower() not in ["menu", "null", "none", name.lower(), path.lower()]:
        extras.append(alias)
        
    if extras:
        desc += f" | 页面功能/别名: {' / '.join(extras)}"
    
    md_lines.append(desc)
    
    # 如果它下面还有“儿子”（子菜单），就套娃（递归）进去继续翻译
    if "children" in node and isinstance(node["children"], list):
        current_parents = parent_names + [name]
        for child in node["children"]:
             # 层级最多加到 6 级，不然大模型可能就懵了
            md_lines.extend(format_routes_markdown(child, min(level + 1, 6), current_parents))
    return md_lines

def parse_txt(file_path: str) -> str:
    """读取普通的 TXT 或者 MD 文本文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # ==== 下面是一段“智能探测”魔法 ====
    # 有时候虽然文件是 .txt 或 .md，但里面实际装的是 JSON 代码。系统会试着把它当 JSON 打开
    try:
        data = json.loads(content)
        # 看看这串 JSON 长得像不像我们系统的菜单路由（寻找特有的关键词）
        if isinstance(data, list) and len(data) > 0 and ("name" in data[0] or "path" in data[0] or "alias" in data[0] or "children" in data[0]):
            lines = ["本文件是系统各个菜单、路由的层级结构配置：\n"]
            # 是菜单配置的话，就把它翻译成大模型更爱读的 Markdown 格式
            for item in data:
                lines.extend(format_routes_markdown(item, 1))
            return "\n\n".join(lines)
        # 是普通 JSON，就美化排版后返回
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        # 如果不是 JSON，报错没关系，就当成普通纯文字原样还回去
        return content

def parse_docx(file_path: str) -> str:
    """读取 Word 文档（.docx），把里面每一段字全揪出来拼到一起"""
    try:
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        print(f"解析 Word 出错了，文件：{file_path}，错误：{e}")
        return ""

def parse_excel(file_path: str) -> str:
    """简单粗暴地读取 Excel 表格（只取前100行防爆炸）"""
    try:
        # 借助 pandas 库来读 Excel，一句话搞定所有分页 (Sheet)
        xls = pd.read_excel(file_path, sheet_name=None)
        texts = []
        for sheet_name, df in xls.items():
            # 扔掉全空的行和列
            df = df.dropna(how="all").dropna(axis=1, how="all")
            if df.empty:
                continue

            texts.append(f"--- 标签页: {sheet_name} ---")

            # 只取前 100 行转化为文字，要是全读，文本太大大模型会看晕的！
            preview_df = df.head(100).astype(str)
            texts.append(preview_df.to_string(index=False))
        return "\n".join(texts)
    except Exception as e:
        print(f"解析 Excel 出错了，文件：{file_path}，错误：{e}")
        return ""

def parse_json(file_path: str) -> str:
    """解析 JSON 文件，其实底层也是按 txt 读的"""
    return parse_txt(file_path)

def parse_file(file_path: str) -> dict:
    """
    【大总管方法】给它一个文件路径，它自动判断是什么后缀，然后叫对应的方法去解析，
    最后打包成一个包含来源、分类、内容的整齐字典返回去。
    """
    source = os.path.basename(file_path) # 比如拿到 "规章制度.docx" 这个文件名
    # 分类取自父文件夹的名称，比如存在 "docs/行政/" 里面，分类就是 "行政"
    category = os.path.basename(os.path.dirname(file_path)) 
    ext = os.path.splitext(source)[1].lower() # 拿到后缀，比如 ".docx"
    
    text = ""
    # 根据后缀找对口的解析器
    if ext in ['.txt', '.md']:
        text = parse_txt(file_path)
    elif ext == '.docx':
        text = parse_docx(file_path)
    elif ext in ['.xlsx', '.xls']:
        text = parse_excel(file_path)
    elif ext == '.json':
        text = parse_json(file_path)
    else:
        # 不认识的文件后缀，直接摇头拒收
        return None
        
    text = clean_text(text) # 把解析出来的文字洗个澡，去掉多余空格
    
    if not text:
        return None
        
    # 打包成标准包裹返回
    return {
        "source": source,
        "category": category,
        "path": file_path,
        "text": text,
        "ext": ext
    }


# =========================================================
# 下面是非常核心的【文本切块 (Chunking) 系统】
# 上面把书读成了长篇大论，下面就是我们要把长篇大论剪成一小块一小块的“知识切片”
# =========================================================

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100, ext: str = "") -> list:
    """
    切块总指挥：每次大概切 500 个字的一块，首尾留 100 个字的重叠（防断句断了魂）
    """
    if ext == ".md":
        # 如果是 Markdown 文件，就用特制的带标题记忆的聪明刀法
        return markdown_semantic_chunker(text, chunk_size, chunk_overlap)
        
    # 不然的话，就按各种标点符号（段落、句号、逗号等）来强行切
    seps = ["\n\n", "\n", "。", "！", "？", "；", "，", " "]
    return _recursive_split(text, seps, chunk_size, chunk_overlap)


def markdown_semantic_chunker(text: str, chunk_size: int, chunk_overlap: int) -> list:
    """
    【智能 Markdown 切片机】（高级玩法）
    因为 Markdown 里面有标题（# 一级 ## 二级...），咱们切出来的每一段小碎片，
    如果单看可能不知道在讲啥。所以要在每个碎片开头加上它所属的各级老祖宗标题。
    就像贴了一个标签：【上下文：公司制度 > 报账规范 > 餐饮报账】
    这样大模型就不会把 "公司制度" 里的 "补贴500" 和 "部门团建" 的 "补贴500" 给搞混了。
    """
    lines = text.split("\n")
    chunks = []
    
    current_content = []
    current_headers = {} # 记忆当前这块碎片属于哪些老祖宗标题
    current_depth = 0
    
    def _build_parent_context(heads, depth):
        """用这个小方法把老祖宗标题拼成一行文字【上下文：xxx > yyy】"""
        parents = [heads[d] for d in sorted(heads.keys()) if d < depth]
        if not parents:
            return ""
        return "【上下文：" + " > ".join(parents) + "】\n"

    for line in lines:
        # 正则表达式，认出 "# 某某某" 这种标题
        header_match = re.match(r'^(#{1,6})\s+(.*)', line)
        if header_match:
            depth = len(header_match.group(1)) # 看看是几级标题 (几个 #)
            header_text = header_match.group(2).strip() # 标题写了啥
            
            # 既然碰到了新标题，那上一段标题下的内容就该收网（打包成块）了
            if current_content:
                block_text = "\n".join(current_content).strip()
                if block_text:
                    context = _build_parent_context(current_headers, current_depth)
                    combined_text = context + block_text
                    # 如果这一大段字实在太多（超过 500 字），就拜托 `_recursive_split` 帮忙切细一点
                    if len(combined_text) > chunk_size:
                        sub_chunks = _recursive_split(combined_text, ["\n\n", "\n", "。", "！", "？", "；", "，", " "], chunk_size, chunk_overlap)
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append(combined_text)
            
            # 开启新内容收集，同时在这块内容上挂上自己这个新标题的标识
            current_content = [line]
            current_depth = depth
            
            current_headers[depth] = header_text
            # 清除比自己级别更深（如数字更大）的过期标题记录
            keys_to_remove = [k for k in list(current_headers.keys()) if k > depth]
            for k in keys_to_remove:
                del current_headers[k]
                
        else:
            # 不是标题，就是普通正文内容，攒起来先
            current_content.append(line)
            
    # 文件到底了，把最后攒在手里的这把牌也扔出去
    if current_content:
        block_text = "\n".join(current_content).strip()
        if block_text:
            context = _build_parent_context(current_headers, current_depth)
            combined_text = context + block_text
            if len(combined_text) > chunk_size:
                sub_chunks = _recursive_split(combined_text, ["\n\n", "\n", "。", "！", "？", "；", "，", " "], chunk_size, chunk_overlap)
                chunks.extend(sub_chunks)
            else:
                chunks.append(combined_text)
                
    # 要是这篇 Markdown 压根就没写任何标题标记，就只能用下面强切割的保底手段了
    if not chunks:
        seps = ["\n\n", "\n", "。", "！", "？", "；", "，", " "]
        return _recursive_split(text, seps, chunk_size, chunk_overlap)
        
    return chunks

def _recursive_split(text: str, separators: list, chunk_size: int, chunk_overlap: int) -> list:
    """
    【递归刀法（通用切块法则）】
    这可是个功夫活：给你一堆段落、句号、逗号用来当劈柴的缝隙。
    它会先试着从段落那里劈，如果劈出来的木柴还是比 500 字（chunk_size）大；
    就把这块硬木柴丢回去，让它用句号进一步劈... 直到每一块都不超过限定大小。
    切的时候还会让相邻切片首尾重叠一定字数（chunk_overlap），防止前言不搭后语。
    """
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text] # 连 500 字都不到，别砍了，原样送走

    # 找咱们用来作为切割基准点的分隔符
    separator_idx = -1
    for i, sep in enumerate(separators):
        if sep == "":
            separator_idx = i
            break
        if sep in text:
            separator_idx = i
            break

    # 真没找到合适的符号（比如全英文没句号），就残酷地“生硬截断”
    if separator_idx == -1:
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunks.append(text[i:i + chunk_size])
        return chunks

    separator = separators[separator_idx] # 这个就是我们决定要下刀的分隔符
    
    splits = text.split(separator)
    docs = []
    
    # 劈完之后，把分隔符（比如句号）贴回句子尾巴上，别乱丢了
    for i, s in enumerate(splits):
        if i < len(splits) - 1:
            docs.append(s + separator)
        else:
            if s:
                docs.append(s)

    chunks = []
    current_doc = []
    current_len = 0

    # 像拼俄罗斯方块一样，把短句子一个个塞进去，塞满就打包发货一块
    for d in docs:
        d_len = len(d)
        if current_len + d_len > chunk_size and current_len > 0:
            chunk_str = "".join(current_doc).strip()
            if chunk_str:
                chunks.append(chunk_str)
            
            # 发完货新起一块时，悄悄把上一块末尾的内容也带上一点，作为“承上启下”的作用
            while current_len > chunk_overlap and len(current_doc) > 0:
                removed_doc = current_doc.pop(0)
                current_len -= len(removed_doc)

        current_doc.append(d)
        current_len += d_len

    # 打包发掉最后攒的边角料
    chunk_str = "".join(current_doc).strip()
    if chunk_str:
        chunks.append(chunk_str)

    # 万一有些极端长句子依然远远超出大小
    # 就拿刚才没用上的刀具（更细的分隔符）叫它重新给自己做个二次手术。
    final_chunks = []
    next_separators = separators[separator_idx + 1:] if separator_idx + 1 < len(separators) else []
    
    for c in chunks:
        if len(c) > chunk_size:
            if next_separators:
                # 自己去找自己的下一级切刀
                final_chunks.extend(_recursive_split(c, next_separators, chunk_size, chunk_overlap))
            else:
                # 实在切不动了，直接字数生切
                for i in range(0, len(c), chunk_size - chunk_overlap):
                    final_chunks.append(c[i:i + chunk_size])
        else:
            final_chunks.append(c)

    return final_chunks

