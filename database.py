import sqlite3
import os
from datetime import datetime

#=======================================当前数据库路径设置的是相对地址，所以需要在Research_Copilot目录下运行该文件，而不是PythonProject2下===============================================

#数据库文件路径
DB_PATH="resources/research_copilot.db"

def get_db_connection():
  """获取数据库连接，并设置row_factory为sqlite3.Row，以便返回字典形式的结果"""
  conn=sqlite3.connect(DB_PATH,check_same_thread=False)
  conn.row_factory=sqlite3.Row    #正常情况下，返回的数据是元组格式，这一行可以使得结果为字典格式，用row['thread_id']来取值
  return conn

def init_db():
  """初始化数据库：创建sessions表（如果不存在）"""
  os.makedirs(os.path.dirname(DB_PATH),exist_ok= True)
  conn=get_db_connection()
  try :
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS sessions(
  thread_id TEXT PRIMARY KEY,
  title TEXT DEFAULT '新会话',
  created_at TEXT,
  updated_at TEXT
);
      """
    )
    conn.commit()
  finally:
    conn.close()

#-------------四个核心功能---------------
def create_session(thread_id:str,title:str="新会话")->None:
  """创建一个新的会话"""
  now=datetime.now().isoformat()  #时间对象转换为 ISO 8601 格式的字符串表示
  conn=get_db_connection()
  try:
    conn.execute(
      "INSERT INTO sessions (thread_id,title,created_at,updated_at) VALUES(?,?,?,?)",
      (thread_id,title,now,now),
    )
    conn.commit()
  finally:
    conn.close()


def list_sessions()->list[dict]:
  """
  获取所有会话，按最后更新时间（updated_at）降序排序。
  :return: 会话列表，每个元素为字典，包含thread_id,title,created_at,updated_at
  """
  conn=get_db_connection()
  try:
    rows=conn.execute(
      "SELECT thread_id,title,created_at,updated_at FROM sessions ORDER BY updated_at DESC"
    ).fetchall()  #fetchall() 获取所有查询结果行
    #将Row对象转换为普通字典
    return [dict(row) for row in rows]
  finally:
    conn.close()


def delete_session(thread_id:str)->bool:
  """
  删除指定会话
  :param thread_id:
  :return: 是否删除成功
  """
  conn=get_db_connection()
  try:
    cursor=conn.execute("DELETE FROM sessions WHERE thread_id=?",(thread_id,))
    conn.commit()
    return cursor.rowcount>0  #受影响行数>0 表示删除了记录
  finally:
    conn.close()


def update_session_title(thread_id:str,title:str)->bool:
  """
  更新会话的标题
  :param thread_id:
  :param title:
  :return: 是否更新成功
  """
  now=datetime.now().isoformat()
  conn=get_db_connection()
  try:
    cursor=conn.execute(
      "UPDATE sessions SET title=?,updated_at=? WHERE thread_id=?",
      (title,now,thread_id),
    )
    conn.commit()
    return cursor.rowcount>0
  finally:
    conn.close()
