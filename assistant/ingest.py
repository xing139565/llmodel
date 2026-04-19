import os
import argparse
import logging
from datetime import datetime
from sentence_transformers import SentenceTransformer
import chromadb
import pickle
import jieba

# 从我们刚才写好注释的 doc_parser 里请来“切割大师”和“解析大师”
from utils.doc_parser import parse_file, chunk_text

# ==========================================
# 1. 设置工作日志 (打卡记录本)
# ==========================================
log_filename = f"logs/ingest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'), # 存到文件里
        logging.StreamHandler() # 同时也打印在屏幕上给你看
    ]
)
logger = logging.getLogger(__name__)

def main():
    # ==========================================
    # 2. 接收启动命令的参数
    # ==========================================
    parser = argparse.ArgumentParser(description="知识库向量化导入（把人类文字翻译成机器坐标并存起来）")
    # 必须指定是全量重建 (rebuild) 还是增量更新 (incremental)
    parser.add_argument('--mode', choices=['rebuild', 'incremental'], required=True, help="导入模式: rebuild (全量重建) 或 incremental (增量更新)")
    args = parser.parse_args()

    docs_dir = "data/docs" # 我们放原始书籍资料的仓库文件夹
    
    if args.mode == 'incremental':
        logger.warning("增量更新(incremental)第一版仅留接口结构，当前版本将执行与 rebuild 类似的简单动作。推荐使用 --mode rebuild")

    logger.info("开始扫描文档...")
    
    # ==========================================
    # 3. 翻箱倒柜，找出所有要读的文件
    # ==========================================
    all_files = []
    # os.walk 会像探险家一样，走遍 docs_dir 里的每一个子文件夹
    for root, _, files in os.walk(docs_dir):
        for f in files:
            if not f.startswith("~"):  # 跳过那种带~符号的系统临时缓存文件（通常是你正打开着的Word）
                all_files.append(os.path.join(root, f))
                
    logger.info(f"发现文档 {len(all_files)} 份")
    
    # ==========================================
    # 4. 让“图书管理员”(parse_file) 开始读书
    # ==========================================
    parsed_docs = []
    success_count = 0
    fail_count = 0
    fail_logs = []
    
    for file_path in all_files:
        try:
            doc_data = parse_file(file_path) # 把文件交给他去读
            if doc_data:
                parsed_docs.append(doc_data) # 如果读懂了，存进大名单里
                success_count += 1
            else:
                fail_count += 1 # 如果读不懂（可能是乱码或里面压根没字空文件）
                fail_logs.append(f"{file_path} - 无效格式或空文件")
        except Exception as e:
            fail_count += 1
            fail_logs.append(f"{file_path} - Error: {str(e)}")

    logger.info(f"成功解析 {success_count} 份, 失败 {fail_count} 份")
    if fail_count > 0:
        for fl in fail_logs:
            logger.error(f"失败记录: {fl}")

    # ==========================================
    # 5. 把长篇大论切成“知识小碎片” (chunk)
    # ==========================================
    all_chunks = []      # 放切割下来所有文字碎片的篮子
    all_metadatas = []   # 放卡片小票标签（比如：出自哪本书、第几块）的篮子
    all_ids = []         # 放这些碎片唯一身份证号的篮子
    
    total_chunk_count = 0
    for doc in parsed_docs:
        # 召唤切割大师：把这篇文章切成每块大约 500 字的小碎片
        chunks = chunk_text(doc["text"], chunk_size=500, chunk_overlap=100, ext=doc.get("ext", ""))
        for i, text_chunk in enumerate(chunks):
            # 给每块碎片写上贴心的小票备注
            metadata = {
                "source": doc["source"],     # 文件名
                "category": doc["category"], # 所属分类（它所在的文件夹名）
                "chunk_index": i,            # 这是被切出来的第几块
                "file_path": str(doc["path"])# 原始文件在电脑里的什么地方
            }
            # 办个长长的身份证号，防止撞名：类别::文件名::第几块
            chunk_id = f"{doc['category']}::{doc['source']}::{i}"
            
            all_chunks.append(text_chunk)
            all_metadatas.append(metadata)
            all_ids.append(chunk_id)
            total_chunk_count += 1
            
    logger.info(f"切分 chunk 总数 {total_chunk_count}")

    if total_chunk_count == 0:
        logger.info("没有可向量化的内容，流程结束。")
        return

    # ==========================================
    # 6. 开启“灵魂翻译机” (Embedding Model)
    # ==========================================
    logger.info("正在加载 embedding 模型 BAAI/bge-small-zh-v1.5 ...")
    # 这个模型可厉害了，专门负责把我们人类看得懂的中文句子，翻译成电脑好比较的一长串数学坐标
    embedding_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    
    logger.info("开始生成 embeddings ...")
    # model.encode 就好像把一箩筐的汉字纸条倒进翻译机里，瞬间吐出对应的坐标账本
    embeddings = embedding_model.encode(all_chunks).tolist()
    
    # ==========================================
    # 7. 把翻译好的成果，存进“魔法大书柜” (ChromaDB) 里
    # ==========================================
    logger.info("准备写入 ChromaDB ...")
    chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
    
    # 如果启动命令上写了 rebuild (重建)，就把以前书柜里的旧卡片全扔了清空！
    if args.mode == 'rebuild':
        try:
            chroma_client.delete_collection("docs")
            logger.info("已删除旧的 docs collection (清空旧魔法柜子)")
        except Exception:
            pass
            
    # 拉开一个叫 "docs" 的大抽屉（如果之前没这个抽屉，就现场给你造一个）
    collection = chroma_client.get_or_create_collection(name="docs")
    
    # 怕柜子一次吃太多数据撑坏了，我们按每次喂 500 张卡片的速度分批推进去
    batch_size = 500
    total_batches = (len(all_ids) + batch_size - 1) // batch_size
    for i in range(0, len(all_ids), batch_size):
        end_idx = i + batch_size
        collection.add(
            documents=all_chunks[i:end_idx],     # 送进去的原文字碎片
            embeddings=embeddings[i:end_idx],    # 送进去的翻译坐标数字
            metadatas=all_metadatas[i:end_idx],  # 绑定标签信息（方便找回是哪本书的）
            ids=all_ids[i:end_idx]               # 送进去专属身份证号
        ) # 一推就存进数据库了！
        logger.info(f"已写入 batch {i//batch_size + 1}/{total_batches}")

    # ==========================================
    # 8. 额外附赠：为主力库打辅助的“传统关键词搜索引擎库”（BM25）
    # ==========================================
    logger.info("开始提取中文分词构建 BM25 数据集...")
    # jieba 切词就像切菜，会把 "我爱北京天安门" 切成 "我", "爱", "北京", "天安门"
    # 用老式的词匹配搜索（BM25算法用到），作为高级 AI 语义搜索的保底强力替补。
    tokenized_corpus = [jieba.lcut(chunk) for chunk in all_chunks]
    
    bm25_data = {
        "chunks": all_chunks,
        "metadatas": all_metadatas,
        "ids": all_ids,
        "tokenized_corpus": tokenized_corpus
    }
    
    # pickle 这个小工具，能把 Python 乱七八糟的变量结构“打包装箱”成一个实实在在的文件存下
    with open("data/bm25_corpus.pkl", "wb") as f:
        pickle.dump(bm25_data, f)
    logger.info("已成功保存 BM25 数据集: data/bm25_corpus.pkl")

    logger.info("写入向量库完成，魔法图书馆开门营业啦！你现在可以去用 chat_cli 跟 AI 提问了。")

if __name__ == "__main__":
    main()
