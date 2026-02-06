'''
이번 예시에서는 sqlite를 통해 Chinook_Sqlite.sql를 사용한다. 실습 실행 전에 sqlite와 sql 파일을 다운받아야 한다.

```bash
curl -s https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql | sqlite3 Chinook.db

```

`Chinook.db`를 코드를 실행할 디렉터리에 옮긴다.

'''

from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI

# 사용할 db 경로로 수정
db = SQLDatabase.from_uri('sqlite:///Chinook.db')
print(db.get_usable_table_names())

llm = ChatOpenAI(model='gpt-4o', temperature=0)

# 질문을 SQL 쿼리로 변환
write_query = create_sql_query_chain(llm, db)

# SQL 쿼리 실행
execute_query = QuerySQLDatabaseTool(db=db)

# combined chain = write_query | execute_query
combined_chain = write_query | execute_query

# 체인 실행
result = combined_chain.invoke({'question': '직원(employee)은 모두 몇 명인가요?'})

print(result)
