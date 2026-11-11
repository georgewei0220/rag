import hashlib
import sys
import uuid
from pathlib import Path

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

import config
from store import ensure_collection, get_client, get_embedder
import re

_CJK = r"\u4e00-\u9fff"
_SP = re.compile(rf"(?<=[{_CJK}，、）」])[ \t\u3000]+(?=[{_CJK}（「])")
_NL = re.compile(rf"(?<=[{_CJK}，、])\n(?=[{_CJK}（「])")
_MULTI = re.compile(r"[ \t]{2,}")

def clean_text(text: str) -> str:
    text = _NL.sub("", text)      # 句子中間的換行：前後都是中文就接起來
    text = _SP.sub("", text)      # 中文字之間的空白：移除
    text = _MULTI.sub(" ", text)  # 其他連續空白：壓成一個
    return text.strip()

def file_hash(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()

def already_ingested(client, doc_id: str) -> bool:
    n = client.count(
        collection_name=config.COLLECTION,
        count_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
        exact=True,
    ).count
    return n > 0


def ingest_pdf(path: Path, client, embedder, splitter) -> int:
    doc_id = file_hash(path)
    if already_ingested(client, doc_id):
        print(f"[跳過] {path.name}（已匯入）")
        return 0

    pages = PDFPlumberLoader(str(path)).load()          # 一頁一個 Document
    for p in pages:
        p.page_content = clean_text(p.page_content)
    chunks = splitter.split_documents(pages)            # metadata（含頁碼）會保留
    chunks = [c for c in chunks if c.page_content.strip()]
    if not chunks:
        print(f"[警告] {path.name} 抽不到文字（可能是掃描檔）")
        return 0

    # 每頁內的 chunk 順序
    counters: dict[int, int] = {}
    points = []
    texts = [c.page_content for c in chunks]
    vectors = []
    for i in range(0, len(texts), config.BATCH):
        vectors += embedder.embed_documents(texts[i : i + config.BATCH])

    for c, vec in zip(chunks, vectors):
        page = int(c.metadata.get("page", 0)) + 1       # 轉成 1 起算
        idx = counters.get(page, 0)
        counters[page] = idx + 1
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{page}:{idx}"))
        points.append(
            PointStruct(
                id=point_id,
                vector=vec,
                payload={
                    "doc_id": doc_id,
                    "source_file": path.name,
                    "page": page,
                    "chunk_index": idx,
                    "text": c.page_content,
                },
            )
        )

    client.upsert(collection_name=config.COLLECTION, points=points)
    print(f"[完成] {path.name}：{len(pages)} 頁 → {len(points)} 個 chunk")
    return len(points)


def main():
    config.DOCS_DIR.mkdir(exist_ok=True)
    pdfs = sorted(p for p in config.DOCS_DIR.glob("*.pdf") if not p.name.startswith("._"))
    if not pdfs:
        print(f"{config.DOCS_DIR} 裡沒有 PDF")
        sys.exit(1)

    client = get_client()
    ensure_collection(client)
    embedder = get_embedder()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=config.SEPARATORS,
    )

    total = sum(ingest_pdf(p, client, embedder, splitter) for p in pdfs)
    print(f"\n本次新增 {total} 個 chunk；collection 目前共 "
          f"{client.count(config.COLLECTION, exact=True).count} 個")


if __name__ == "__main__":
    main()
