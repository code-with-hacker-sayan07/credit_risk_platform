import sqlite3
import re
import pandas as pd
from src.utils.logger import setup_logger
from src.utils.config import DB_PATH

logger = setup_logger("query_runner")

class SafeQueryRunner:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path

    def is_query_safe(self, sql: str) -> bool:
        """Verifies if the SQL statement is read-only and free of dangerous keywords."""
        sanitized = sql.strip().lower()
        
        # Must start with SELECT or WITH
        if not (sanitized.startswith("select") or sanitized.startswith("with")):
            return False
            
        # Check for modifications
        unsafe_keywords = [
            r"\binsert\b", r"\bupdate\b", r"\bdelete\b", r"\bdrop\b", 
            r"\bcreate\b", r"\balter\b", r"\breplace\b", r"\btruncate\b", 
            r"\bgrant\b", r"\brevoke\b"
        ]
        
        for keyword in unsafe_keywords:
            if re.search(keyword, sanitized):
                logger.warning(f"Rejected SQL query due to safety check: Contains {keyword}")
                return False
                
        return True

    def execute_query(self, sql: str) -> dict:
        """Executes a read-only SQL query against the SQLite database."""
        # Clean potential markdown wrapping
        sql = sql.replace("```sql", "").replace("```", "").strip()
        
        if not self.is_query_safe(sql):
            return {
                "status": "error",
                "sql": sql,
                "data": [],
                "columns": [],
                "message": "Security Error: Only read-only SELECT queries are allowed."
            }

        try:
            logger.info(f"Executing SQL query:\n{sql}")
            conn = sqlite3.connect(str(self.db_path))
            
            # Execute query into dataframe
            df = pd.read_sql_query(sql, conn)
            conn.close()
            
            # Cap output to protect performance
            total_records = len(df)
            df_display = df.head(100)
            
            # Convert columns and rows
            columns = list(df_display.columns)
            data = df_display.to_dict(orient="records")
            
            msg = f"Query executed successfully. Returned {total_records} rows."
            if total_records > 100:
                msg += " (Showing first 100 rows for display performance)."
                
            return {
                "status": "success",
                "sql": sql,
                "data": data,
                "columns": columns,
                "message": msg
            }
            
        except Exception as e:
            logger.error(f"SQL Execution Error: {str(e)}")
            return {
                "status": "error",
                "sql": sql,
                "data": [],
                "columns": [],
                "message": f"Database Error: {str(e)}"
            }
