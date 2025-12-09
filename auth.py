# auth.py  - User Authentication using PBKDF2-SHA256 (no bcrypt issues)

import sqlite3
from passlib.hash import pbkdf2_sha256

DB_NAME = "users.db"


def init_auth_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def create_user(username: str, password: str) -> bool:
    """
    Create a new user with hashed password using PBKDF2-SHA256.
    Returns True on success, False if username already exists.
    """
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    # hash password with pbkdf2_sha256
    hashed = pbkdf2_sha256.hash(password)

    try:
        cur.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, hashed)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # username already exists
        return False
    finally:
        conn.close()


def verify_user(username: str, password: str) -> bool:
    """
    Verify given username & password against stored hash.
    Returns True if valid, else False.
    """
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute(
        "SELECT password FROM users WHERE username = ?",
        (username,)
    )
    row = cur.fetchone()
    conn.close()

    if row is None:
        return False

    stored_hash = row[0]
    return pbkdf2_sha256.verify(password, stored_hash)
