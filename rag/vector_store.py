from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from utils.config_handler import chroma_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
import os
import pickle
import time
import hashlib


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=chroma_conf["persist_directory"],
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        要计算文件的MD5做去重
        :return: None
        """

        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                # 创建文件
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False            # md5 没处理过

            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True     # md5 处理过

                return False            # md5 没处理过

        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)

            if read_path.endswith("pdf"):
                return pdf_loader(read_path)

            return []

        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        for path in allowed_files_path:
            # 获取文件的MD5
            md5_hex = get_file_md5_hex(path)

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue

            try:
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                split_document: list[Document] = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                # 将内容存入向量库
                self.vector_store.add_documents(split_document)

                # 记录这个已经处理好的文件的md5，避免下次重复加载
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path} 内容加载成功")
            except Exception as e:
                # exc_info为True会记录详细的报错堆栈，如果为False仅记录报错信息本身
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue

    def get_hybrid_retriever(self):
        """返回混合检索器：向量检索 + BM25 关键字检索（使用序列化缓存）"""
        bm25_retriever = self._get_cached_bm25_retriever()
        vector_retriever = self.get_retriever()

        hybrid_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.5, 0.5],
        )

        return hybrid_retriever

    # ── BM25 缓存相关 ──────────────────────────────────────────────

    def _bm25_cache_path(self) -> str:
        return get_abs_path("bm25_cache.pkl")

    def _bm25_cache_key_path(self) -> str:
        return get_abs_path("bm25_cache_key.txt")

    def _compute_bm25_cache_key(self) -> str:
        """用所有源文件的 MD5 拼接成一个缓存键，文件变了键就变"""
        data_path = get_abs_path(chroma_conf["data_path"])
        allowed_files = listdir_with_allowed_type(
            data_path, tuple(chroma_conf["allow_knowledge_file_type"])
        )
        allowed_files = sorted(allowed_files)

        hasher = hashlib.md5()
        for filepath in allowed_files:
            file_md5 = get_file_md5_hex(filepath)
            if file_md5:
                hasher.update(file_md5.encode())
        return hasher.hexdigest()

    def _get_cached_bm25_retriever(self):
        """尝试从磁盘加载缓存的 BM25 检索器，无效则重新构建并缓存"""
        cache_path = self._bm25_cache_path()
        key_path = self._bm25_cache_key_path()

        # 检查缓存是否有效（文件存在且源文件未变化）
        if os.path.exists(cache_path) and os.path.exists(key_path):
            with open(key_path, "r", encoding="utf-8") as f:
                cached_key = f.read().strip()
            if cached_key == self._compute_bm25_cache_key():
                logger.info("[BM25缓存]命中缓存，从磁盘加载")
                with open(cache_path, "rb") as f:
                    retriever = pickle.load(f)
                retriever.k = chroma_conf["k"]
                return retriever

        # 缓存无效，重新构建
        logger.info("[BM25缓存]未命中，重新构建 BM25 索引")
        t_start = time.time()

        all_docs = self._get_all_documents()
        retriever = BM25Retriever.from_documents(
            documents=all_docs,
            k=chroma_conf["k"],
        )
        retriever.k = chroma_conf["k"]

        elapsed = time.time() - t_start
        logger.info(f"[BM25缓存]索引构建完成，耗时 {elapsed:.2f}s，文档数 {len(all_docs)}")

        # 序列化到磁盘
        with open(cache_path, "wb") as f:
            pickle.dump(retriever, f)
        with open(key_path, "w", encoding="utf-8") as f:
            f.write(self._compute_bm25_cache_key())
        logger.info(f"[BM25缓存]已序列化到 {cache_path}")

        return retriever

    def _get_all_documents(self) -> list[Document]:
        """从知识库文件中读取所有文档并分片（含计时）"""
        t_start = time.time()

        data_path = get_abs_path(chroma_conf["data_path"])
        allowed_files = listdir_with_allowed_type(
            data_path, tuple(chroma_conf["allow_knowledge_file_type"])
        )

        all_docs = []
        for path in allowed_files:
            if path.endswith("txt"):
                all_docs.extend(txt_loader(path))
            elif path.endswith("pdf"):
                all_docs.extend(pdf_loader(path))

        split_docs = self.spliter.split_documents(all_docs)

        elapsed = time.time() - t_start
        logger.info(f"[文档加载]共 {len(split_docs)} 个分片，耗时 {elapsed:.2f}s")

        return split_docs


if __name__ == '__main__':
    vs = VectorStoreService()

    vs.load_document()

    retriever = vs.get_retriever()

    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-"*20)


