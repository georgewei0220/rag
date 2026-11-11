from pathlib import Path
 
BASE = Path(__file__).parent
DOCS_DIR = BASE / "docs"            # 放要匯入的 PDF
QDRANT_PATH = str(BASE / "qdrant_data")   # 本機模式；之後改伺服器模式只要換 client
COLLECTION = "reports"
EMBED_MODEL = "bge-m3"
VECTOR_SIZE = 1024
 
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
SEPARATORS = ["\n\n", "\n", "。", "；", "，", " ", ""]
BATCH = 32