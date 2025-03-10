import os
import ssl
from dotenv import load_dotenv

# SSL 및 환경 설정
load_dotenv()
ssl._create_default_https_context = ssl._create_unverified_context

# 모델 설정
DEFAULT_MODEL = "models/gemini-1.5-pro" 