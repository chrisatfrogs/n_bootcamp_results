import uuid
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import (Column, Integer, String)
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()
engine = create_engine('sqlite:///.\\turns\\turns.db')

def create_session():
    Session = sessionmaker(bind=engine)
    session = Session()
    return session

class Turn(Base):
    __tablename__ = 'turns'

    turn_id = Column(Integer(), primary_key=True)
    turn_name = Column(String(50), unique=True)
    turn_uuid = Column(String(50), index=True)
    turn_description = Column(String(255))

    def __repr__(self):
        return f"Turn(turn_name={self.turn_name}, turn_description={self.turn_description})"
    
    def create_turns():
        Base.metadata.create_all(engine)
        session = create_session()
        turns = ['turnx88nf', 'turnx90', 'turnx92nf', 'turnx93','turnx96nf','turnx97', 'turnx98nf', 'turnx100nf', 'turnx101', 'turnx102nf', 'turnx103', 'turnx104nf', 'turnx106nf']
        for turn in turns:
            session.add(Turn(turn_name=turn, turn_uuid=str(uuid.uuid4()), turn_description=""))
        session.commit()


