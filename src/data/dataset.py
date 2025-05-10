import os 
import sqlite3
import pandas as pd
import numpy as np

IMAGE_EXTENSION = "png"

def load_all_character_tags(db_path: str, 
                        table_name: str="character_tags") -> list[str]:
    """
    Load character_tags from sqlite3 database 

    The database should have `name` column represent name of characters.
    """

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"""
    SELECT name FROM {table_name}
    ORDER BY id;
    """)

    rows = cursor.fetchall()

    return [row[0] for row in rows]

def load_records(root_path: str, db_path: str, table_name: str="illusts") -> pd.DataFrame:
    """
    Load all records from database. 

    Args:
        root_path: absolute path of root directory of dataset. See 'Dataset Structure' sector of README.md
        db_path: absolute or relative path of database file.
    """

    # If `db_path` is relative, then convert it to absolute.
    if not os.path.isabs(db_path):
        db_path = os.path.join(root_path, db_path)

    conn = sqlite3.connect(db_path)
    
    # Read all record from database by DataFrame
    df = pd.read_sql_query(f"""
    SELECT filename, sanity_level, total_view, total_bookmarks, tags, tag_character, date 
        (CASE WHEN illust_ai_type = 2 
            OR tags LIKE '%AIイラスト%' 
            OR tags LIKE '%Diffusion%' 
            OR tags LIKE '%Novel%' 
            OR tags LIKE '%midjourney%'
        THEN 1 ELSE 0 END) as ai_flag
    FROM {table_name}
    ORDER BY filename;
    """, conn)

    # Convert `filename` to `image_path`. 
    # e.g., "1234567_p0" -> "{root_path}/12/1234567_p0.png"
    df['image_path'] = df['filename'].apply(lambda x: os.path.join(root_path, x[:2], f"{x}.{IMAGE_EXTENSION}"))
    
    # Drop `filename` column 
    del df['filename']

    # Convert TIMESTAMP to datetime and truncate to date only.
    df['date'] = pd.to_datetime(df['date']).dt.date

    # Rename columns. match to model output name.
    df.rename(columns={
        "sanity_level": "rating_prediction",
        "ai_flag": "ai_prediction",
    })

    return df