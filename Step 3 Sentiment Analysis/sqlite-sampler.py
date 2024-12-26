import sqlite3
import json
from typing import Dict, List, Optional
import os
import seaborn as sns

os.chdir(os.path.dirname(os.path.realpath(__file__)))

def sample_sqlite_for_llm(db_path: str, 
                         max_rows_per_table: int = 3,
                         include_tables: Optional[List[str]] = None) -> str:
    """
    Creates a comprehensive yet concise representation of a SQLite database
    suitable for LLM understanding.
    
    Args:
        db_path: Path to the SQLite database
        max_rows_per_table: Maximum number of sample rows per table
        include_tables: List of specific tables to include (None for all)
    
    Returns:
        String containing database schema and sample data in a structured format
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all tables or specific tables if provided
    if include_tables:
        tables = include_tables
    else:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]
    
    database_info = {
        "database_name": db_path,
        "tables": []
    }
    
    for table in tables:
        # Get table schema
        cursor.execute(f"PRAGMA table_info('{table}');")
        columns = cursor.fetchall()
        
        table_info = {
            "table_name": table,
            "columns": [
                {
                    "name": col[1],
                    "type": col[2],
                    "nullable": not col[3],
                    "primary_key": bool(col[5])
                }
                for col in columns
            ],
            "sample_rows": []
        }
        
        # Get foreign key information
        cursor.execute(f"PRAGMA foreign_key_list('{table}');")
        foreign_keys = cursor.fetchall()
        if foreign_keys:
            table_info["foreign_keys"] = [
                {
                    "column": fk[3],
                    "references_table": fk[2],
                    "references_column": fk[4]
                }
                for fk in foreign_keys
            ]
        
        # Get sample rows
        try:
            cursor.execute(f"SELECT * FROM '{table}' LIMIT {max_rows_per_table};")
            rows = cursor.fetchall()
            column_names = [description[0] for description in cursor.description]
            
            for row in rows:
                sample_row = {}
                for col_name, value in zip(column_names, row):
                    # Convert special types to strings if needed
                    if isinstance(value, (bytes, bytearray)):
                        value = "<binary data>"
                    sample_row[col_name] = value
                table_info["sample_rows"].append(sample_row)
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM '{table}';")
            table_info["total_rows"] = cursor.fetchone()[0]
            
        except sqlite3.Error as e:
            table_info["error"] = str(e)
        
        database_info["tables"].append(table_info)
    
    conn.close()
    
    # Format the output in a clean, readable way
    def format_table_info(table_info: Dict) -> str:
        output = []
        output.append(f"\nTable: {table_info['table_name']}")
        output.append(f"Total rows: {table_info.get('total_rows', 'Unknown')}")
        
        # Columns
        output.append("\nColumns:")
        for col in table_info["columns"]:
            pk_str = " (Primary Key)" if col["primary_key"] else ""
            null_str = " NULL" if col["nullable"] else " NOT NULL"
            output.append(f"- {col['name']}: {col['type']}{null_str}{pk_str}")
        
        # Foreign Keys
        if "foreign_keys" in table_info:
            output.append("\nForeign Keys:")
            for fk in table_info["foreign_keys"]:
                output.append(
                    f"- {fk['column']} → {fk['references_table']}"
                    f"({fk['references_column']})"
                )
        
        # Sample Rows
        if table_info["sample_rows"]:
            output.append(f"\nSample Rows ({len(table_info['sample_rows'])} of {table_info['total_rows']}):")
            for row in table_info["sample_rows"]:
                output.append(json.dumps(row, indent=2))
        
        if "error" in table_info:
            output.append(f"\nError: {table_info['error']}")
            
        return "\n".join(output)
    
    final_output = f"Database: {database_info['database_name']}\n"
    final_output += "=" * 50
    for table in database_info["tables"]:
        final_output += format_table_info(table)
        final_output += "\n" + "=" * 50
    
    return final_output

# Example usage:
if __name__ == "__main__":
    # Replace with your database path
    db_path = "rapper_sentiments.db"
    
    # Sample specific tables
    # output = sample_sqlite_for_llm(db_path, include_tables=["users", "orders"])
    
    # Sample all tables
    output = sample_sqlite_for_llm(db_path, max_rows_per_table=3)
    print(output)
