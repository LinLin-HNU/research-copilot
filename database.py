import sqlite3
import os
from datetime import datetime

#=======================================当前数据库路径设置的是相对地址，所以需要在Research_Copilot目录下运行该文件，而不是PythonProject2下===============================================

#数据库文件路径
DB_PATH="resources/research_copilot.db"

def get_db_connection():
  """获取数据库连接，并设置row_factory为sqlite3.Row，以便返回字典形式的结果"""
  conn=sqlite3.connect(DB_PATH,check_same_thread=False,timeout=10)
  conn.execute("PRAGMA busy_timeout=10000")
  conn.row_factory=sqlite3.Row    #正常情况下，返回的数据是元组格式，这一行可以使得结果为字典格式，用row['thread_id']来取值
  return conn

def init_db():
  """初始化数据库：创建sessions表（如果不存在）"""
  os.makedirs(os.path.dirname(DB_PATH),exist_ok= True)
  conn=get_db_connection()
  try :
    conn.execute("PRAGMA journal_mode=WAL")
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
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS request_metrics(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id TEXT NOT NULL UNIQUE,
        thread_id TEXT NOT NULL,
        request_kind TEXT NOT NULL,
        route_category TEXT,
        success INTEGER NOT NULL,
        download_ms REAL,
        parse_ms REAL,
        index_ms REAL,
        routing_ms REAL,
        retrieval_ms REAL,
        generation_ms REAL,
        total_ms REAL NOT NULL,
        evidence_blocks INTEGER NOT NULL DEFAULT 0,
        evidence_chars INTEGER NOT NULL DEFAULT 0,
        output_chars INTEGER NOT NULL DEFAULT 0,
        input_tokens INTEGER,
        output_tokens INTEGER,
        error_type TEXT,
        created_at TEXT NOT NULL
      );
      """
    )
    conn.execute(
      "CREATE INDEX IF NOT EXISTS idx_request_metrics_created_at "
      "ON request_metrics(created_at DESC)"
    )
    conn.commit()
  finally:
    conn.close()


def record_request_metric(metric: dict) -> None:
  """Persist non-content request telemetry and retain the latest 10,000 rows."""
  now=datetime.now().isoformat()
  conn=get_db_connection()
  try:
    cursor=conn.execute(
      """
      INSERT INTO request_metrics(
        request_id,thread_id,request_kind,route_category,success,
        download_ms,parse_ms,index_ms,routing_ms,retrieval_ms,generation_ms,total_ms,
        evidence_blocks,evidence_chars,output_chars,input_tokens,output_tokens,error_type,created_at
      ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
      """,
      (
        metric["request_id"], metric["thread_id"], metric["request_kind"],
        metric.get("route_category"), int(metric["success"]),
        metric.get("download_ms"), metric.get("parse_ms"), metric.get("index_ms"),
        metric.get("routing_ms"), metric.get("retrieval_ms"), metric.get("generation_ms"),
        metric["total_ms"], metric.get("evidence_blocks", 0), metric.get("evidence_chars", 0),
        metric.get("output_chars", 0), metric.get("input_tokens"), metric.get("output_tokens"),
        metric.get("error_type"), now,
      ),
    )
    if cursor.lastrowid % 100 == 0:
      conn.execute(
        "DELETE FROM request_metrics WHERE id IN ("
        "SELECT id FROM request_metrics ORDER BY id DESC LIMIT -1 OFFSET 10000"
        ")"
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
