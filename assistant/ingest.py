import os
import argparse
import logging
from datetime import datetime
from sentence_transformers import SentenceTransformer
import chromadb

from utils.doc_parser import parse_file, chunk_text

# 设置日志
log_filename = f"logs/ingest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="知识库向量化导入")
    parser.add_argument('--mode', choices=['rebuild', 'incremental'], required=True, help="导入模式: rebuild (全量重建) 或 incremental (增量更新)")
    args = parser.parse_args()

    docs_dir = "data/docs"
    
    if args.mode == 'incremental':
        logger.warning("增量更新(incremental)第一版仅留接口结构，当前版本将执行与 rebuild 类似的简单动作。推荐使用 --mode rebuild")

    logger.info("开始扫描文档...")
    
    all_files = []
    for root, _, files in os.walk(docs_dir):
        for f in files:
            if not f.startswith("~"):  # 跳过临时文件
                all_files.append(os.path.join(root, f))
                
    logger.info(f"发现文档 {len(all_files)} 份")
    
    parsed_docs = []
    success_count = 0
    fail_count = 0
    fail_logs = []
    
    for file_path in all_files:
        try:
            doc_data = parse_file(file_path)
            if doc_data:
                parsed_docs.append(doc_data)
                success_count += 1
            else:
                fail_count += 1
                fail_logs.append(f"{file_path} - 无效格式或空文件")
        except Exception as e:
            fail_count += 1
            fail_logs.append(f"{file_path} - Error: {str(e)}")

    logger.info(f"成功解析 {success_count} 份, 失败 {fail_count} 份")
    if fail_count > 0:
        for fl in fail_logs:
            logger.error(f"失败记录: {fl}")

    # 切分 chunks
    all_chunks = []
    all_metadatas = []
    all_ids = []
    
    total_chunk_count = 0
    for doc in parsed_docs:
        chunks = chunk_text(doc["text"], chunk_size=500, chunk_overlap=100)
        for i, text_chunk in enumerate(chunks):
            metadata = {
                "source": doc["source"],
                "category": doc["category"],
                "chunk_index": i,
                "file_path": str(doc["path"])
            }
            chunk_id = f"{doc['source']}-{i}"
            
            all_chunks.append(text_chunk)
            all_metadatas.append(metadata)
            all_ids.append(chunk_id)
            total_chunk_count += 1
            
    logger.info(f"切分 chunk 总数 {total_chunk_count}")

    if total_chunk_count == 0:
        logger.info("没有可向量化的内容，流程结束。")
        return

    logger.info("正在加载 embedding 模型 BAAI/bge-small-zh-v1.5 ...")
    embedding_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    
    logger.info("开始生成 embeddings ...")
    embeddings = embedding_model.encode(all_chunks).tolist()
    
    logger.info("准备写入 ChromaDB ...")
    chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
    
    if args.mode == 'rebuild':
        try:
            chroma_client.delete_collection("docs")
            logger.info("已删除旧的 docs collection")
        except Exception:
            pass
            
    collection = chroma_client.get_or_create_collection(name="docs")
    
    batch_size = 500
    total_batches = (len(all_ids) + batch_size - 1) // batch_size
    for i in range(0, len(all_ids), batch_size):
        end_idx = i + batch_size
        collection.add(
            documents=all_chunks[i:end_idx],
            embeddings=embeddings[i:end_idx],
            metadatas=all_metadatas[i:end_idx],
            ids=all_ids[i:end_idx]
        )
        logger.info(f"已写入 batch {i//batch_size + 1}/{total_batches}")

    logger.info("写入向量库完成")

if __name__ == "__main__":
    main()
