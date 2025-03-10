from sqlalchemy import Column, Integer, String, DateTime, Boolean
from database import Base
from datetime import datetime

class User(Base):
    __tablename__ = "tb_user"

    user_id = Column(Integer, primary_key=True, index=True)
    user_no = Column(String(50), unique=True, index=True)
    user_name = Column(String(100))
    user_profile_id = Column(String(200), nullable=True)
    user_profile_bg = Column(String(200), nullable=True)
    last_visited_datetime = Column(DateTime, nullable=True)
    activate_yn = Column(Boolean, default=True)
    role_id = Column(Integer)
    
    # 감사 필드
    created_by = Column(String(50))
    created_datetime = Column(DateTime, default=datetime.utcnow)
    updated_by = Column(String(50), nullable=True)
    updated_datetime = Column(DateTime, nullable=True, onupdate=datetime.utcnow)
    # 필요한 다른 필드들도 추가할 수 있습니다 