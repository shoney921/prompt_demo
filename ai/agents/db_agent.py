from langchain_experimental.tools import PythonREPLTool
from ai.agents.base_agent import BaseSubAgent
from langchain.agents import AgentExecutor
from langchain_core.language_models import BaseLanguageModel
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import (
    QuerySQLDatabaseTool,
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool
)
from langchain.chains import create_sql_query_chain
import os
from dotenv import load_dotenv

class DBAgent(BaseSubAgent):
    def __init__(self, llm: BaseLanguageModel):
        # 먼저 db 설정을 초기화합니다
        load_dotenv()
        
        mysql_config = {
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'host': os.getenv('DB_HOST'),
            'database': "ecommerce_db"
        }
        
        # 데이터베이스 이름을 포함한 연결 문자열 생성
        connection_string = f"mysql+pymysql://{mysql_config['user']}:{mysql_config['password']}@{mysql_config['host']}/{mysql_config['database']}"
        # 메타데이터 리플렉션을 비활성화하여 캐싱 문제 방지
        self.db = SQLDatabase.from_uri(connection_string, metadata=None)
        
        # SQL 체인 설정 업데이트
        self.db_chain = create_sql_query_chain(llm, self.db)
        
        # 그 다음 부모 클래스 초기화를 호출합니다
        super().__init__(llm)
        
        self.setup_tools()
        
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            handle_parsing_errors=True,
            verbose=True
        )

    def setup_tools(self):
        self.tools = [
            QuerySQLDatabaseTool(
                db=self.db,
                name="MySQL_쿼리실행",
                description="MySQL 데이터베이스에 SQL 쿼리를 실행합니다.",
                # 쿼리 실행 전에 마크다운 포맷팅 제거
                preprocessor=lambda q: q.replace('```sql', '').replace('```', '').strip()
            ),
            QuerySQLCheckerTool(
                db=self.db,
                llm=self.llm,
                name="MySQL_쿼리검증",
                description="SQL 쿼리의 안전성을 검증하고 개선사항을 제안합니다."
            ),
            InfoSQLDatabaseTool(
                db=self.db,
                name="MySQL_테이블정보",
                description="특정 테이블의 스키마 정보를 조회합니다. 테이블 이름을 입력으로 받습니다."
            ),
            ListSQLDatabaseTool(
                db=self.db,
                name="MySQL_테이블목록",
                description="데이터베이스의 모든 테이블 목록을 조회합니다."
            )
        ]
    
    def run(self, query: str) -> str:
        """자연어 쿼리를 실행하고 결과를 반환합니다."""
        try:
            # 여러 SQL 문을 세미콜론으로 분리하여 개별 실행
            if isinstance(query, str):
                # 마크다운 포맷팅 제거
                query = query.replace('```sql', '').replace('```', '').strip()
                
                # 여러 SQL 문을 분리
                queries = [q.strip() for q in query.split(';') if q.strip()]
                
                results = []
                for single_query in queries:
                    if single_query:
                        result = self.agent_executor.invoke({"input": single_query})
                        results.append(result["output"])
                
                return "\n".join(results)
                
        except Exception as e:
            return f"에러 발생: {str(e)}"

# 테스트 코드
def test_db_agent():
    from langchain_google_genai import ChatGoogleGenerativeAI
    from ai.config import DEFAULT_MODEL 

    llm = ChatGoogleGenerativeAI(model=DEFAULT_MODEL)
    db_agent = DBAgent(llm)
    
    test_queries = [
        # "이커머스 관련된 데이터베이스 부터 만들어주고, 이커머스에 필요한 테이블들 생성해줘"
        "ecommerce_db 데이터베이스 안의 모든 테이블에 적절한 데이터들을 채워줘"
        # # 기본적인 쿼리 검증
        # "SELECT * FROM tb_product 이 쿼리의 문제점을 분석해줘",
        
        # # 성능 개선 제안
        # "SELECT p.*, c.category_name FROM tb_product p LEFT JOIN tb_category c ON p.category_id = c.id WHERE p.price > 1000 이 쿼리의 성능을 개선할 방법을 제안해줘",
        
        # # 보안 검사
        # "UPDATE tb_product SET price = price * 1.1 이 쿼리의 안전성을 검사해줘",
        
        # # 복잡한 쿼리 분석
        # """
        # SELECT 
        #     p.product_name,
        #     COUNT(*) as order_count,
        #     SUM(o.quantity * p.price) as total_revenue
        # FROM tb_product p
        # JOIN tb_order_item o ON p.id = o.product_id
        # GROUP BY p.product_name
        # HAVING total_revenue > 1000000
        # 이 쿼리의 문제점과 개선방안을 분석해줘
        # """
    ]
    
    for query in test_queries:
        try:
            print(f"\n질문: {query}")
            result = db_agent.run(query)
            print(f"응답: {result}")
        except Exception as e:
            print(f"에러 발생: {str(e)}")

if __name__ == "__main__":
    test_db_agent()
