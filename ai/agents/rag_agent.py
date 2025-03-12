import os
from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredMarkdownLoader
from ai.config import RAG_CONFIG, DEFAULT_MODEL

class RAGTool:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model=DEFAULT_MODEL)
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAG_CONFIG["chunk_size"],
            chunk_overlap=RAG_CONFIG["chunk_overlap"],
            length_function=len,
        )
        self.loader_map = {
            ".txt": TextLoader,
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".md": UnstructuredMarkdownLoader
        }
    
    def load_documents(self) -> List[Document]:
        """문서 디렉토리에서 모든 지원되는 문서를 로드합니다."""
        documents = []
        docs_path = RAG_CONFIG["documents_path"]
        
        for filename in os.listdir(docs_path):
            file_path = os.path.join(docs_path, filename)
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext in self.loader_map:
                try:
                    loader = self.loader_map[file_ext](file_path)
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Warning: {filename} 로딩 중 오류 발생 - {str(e)}")
                    
        return documents

    def add_document(self, file_path: str) -> bool:
        """새로운 문서를 추가합니다."""
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.loader_map:
            return False
            
        try:
            # 문서를 documents 디렉토리로 복사
            import shutil
            filename = os.path.basename(file_path)
            dest_path = os.path.join(RAG_CONFIG["documents_path"], filename)
            shutil.copy2(file_path, dest_path)
            
            # 벡터 스토어 업데이트
            loader = self.loader_map[file_ext](dest_path)
            documents = loader.load()
            self.initialize_vector_store(documents)
            return True
        except Exception as e:
            print(f"Error adding document: {str(e)}")
            return False

    def initialize_vector_store(self, documents: List[Document]):
        """문서를 벡터 스토어에 초기화하고 저장합니다."""
        texts = self.text_splitter.split_documents(documents)
        self.vector_store = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=RAG_CONFIG["vector_store_path"]
        )

    def _run(self, query: str) -> str:
        """검색된 문서를 기반으로 응답을 생성합니다."""
        if not self.vector_store:
            return "문서가 초기화되지 않았습니다. 먼저 문서를 로드해주세요."
            
        relevant_docs = self.vector_store.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        return f"""관련 문서 검색 결과:
        
        {context}"""