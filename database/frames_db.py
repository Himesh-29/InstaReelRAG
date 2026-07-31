import os
import sqlite3
from config.logger import setup_logger

logger = setup_logger("FramesDB")

DEFAULT_FRAMES_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frames.db")

def init_frames_db(db_path: str = DEFAULT_FRAMES_DB_PATH):
    """
    Initializes a separate SQLite database for storing compressed WebP thumbnail blobs
    without bloating the main metadata database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS frame_blobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_shortcode TEXT NOT NULL,
            frame_idx INTEGER NOT NULL,
            webp_bytes BLOB NOT NULL,
            UNIQUE(video_shortcode, frame_idx)
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_frame_blobs_shortcode ON frame_blobs(video_shortcode)")
    conn.commit()
    conn.close()

def save_frame_blob(video_shortcode: str, frame_idx: int, webp_bytes: bytes, db_path: str = DEFAULT_FRAMES_DB_PATH):
    """Saves or updates a WebP thumbnail blob in frames.db."""
    init_frames_db(db_path)
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO frame_blobs (id, video_shortcode, frame_idx, webp_bytes)
            VALUES (
                (SELECT id FROM frame_blobs WHERE video_shortcode = ? AND frame_idx = ?),
                ?, ?, ?
            )
        """, (video_shortcode, frame_idx, video_shortcode, frame_idx, webp_bytes))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Error saving frame blob to frames.db for '{video_shortcode}': {e}")

def get_frame_blob(video_shortcode: str, frame_idx: int, db_path: str = DEFAULT_FRAMES_DB_PATH) -> bytes:
    """Retrieves the WebP thumbnail binary bytes for a given video shortcode and frame index."""
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT webp_bytes FROM frame_blobs WHERE video_shortcode = ? AND frame_idx = ?", (video_shortcode, frame_idx))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
    except Exception as e:
        logger.warning(f"Error retrieving frame blob from frames.db for '{video_shortcode}': {e}")
        return None
