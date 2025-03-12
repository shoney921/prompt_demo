import os
import ssl
from dotenv import load_dotenv

# SSL 및 환경 설정
load_dotenv()
ssl._create_default_https_context = ssl._create_unverified_context

# 모델 설정
DEFAULT_MODEL = "models/gemini-1.5-pro"

# RAG 설정
RAG_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "vector_store_path": "data/vector_store",
    "documents_path": "data/documents",  # 문서 저장 경로
    "supported_formats": [".txt", ".pdf", ".docx", ".md"]  # 지원하는 파일 형식
}

# 필요한 디렉토리 생성
os.makedirs(RAG_CONFIG["vector_store_path"], exist_ok=True)
os.makedirs(RAG_CONFIG["documents_path"], exist_ok=True) 