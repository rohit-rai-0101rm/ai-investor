import os
from urllib.parse import quote

import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

load_dotenv()

# ===== MONITORING & OBSERVABILITY (Phase 1 fix) =====
# Keyed by database name (get_engine supports overriding it) rather than a
# single bare global, so the multi-database parameter still works correctly -
# in practice this app only ever uses one database, so this dict holds at
# most one entry. Same singleton pattern as the embeddings/vector store fix
# in routes/chat.py: SQLAlchemy Engines are meant to be created once and
# reused (they own a connection pool internally) - the old code called
# create_engine() fresh on every get_engine() call, paying a full fresh
# TCP+TLS+auth handshake to Neon on every single query (~5.5s, confirmed by
# direct measurement) instead of reusing an already-open pooled connection.
_engines: dict[str, Engine] = {}


def get_engine(database: str | None = None):
    """
    Get (or lazily create) a cached PostgreSQL engine.

    Args:
        database: Database name. Defaults to POSTGRES_DATABASE from .env.
    """
    if database is None:
        database = os.getenv("POSTGRES_DATABASE")

    if database in _engines:
        return _engines[database]

    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD", "")

    # URL-encode credentials to handle special characters (e.g., @ in password)
    encoded_user = quote(user, safe="")
    encoded_password = quote(password, safe="")

    # ===== OLD CODE (Azure PostgreSQL always requires SSL) =====
    # connection_string = (
    #     f"postgresql+psycopg2://"
    #     f"{encoded_user}:{encoded_password}@{host}:{port}/{database}"
    #     "?sslmode=require"
    # )

    # ===== FREE ALTERNATIVE (local Postgres has no SSL; "prefer" works for both
    # local dev and SSL-requiring free-tier hosts like Neon/Supabase) =====
    sslmode = os.getenv("POSTGRES_SSLMODE", "prefer")
    connection_string = (
        f"postgresql+psycopg2://"
        f"{encoded_user}:{encoded_password}@{host}:{port}/{database}"
        f"?sslmode={sslmode}"
    )

    engine = create_engine(connection_string)
    _engines[database] = engine
    return engine


def create_database() -> None:
    """
    Create the target database if it does not exist.
    
    Uses psycopg2 directly with autocommit to bypass SQLAlchemy transaction wrapping.
    """
    target_db = os.getenv("POSTGRES_DATABASE")
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD", "")

    try:
        # ===== OLD CODE (Azure PostgreSQL always requires SSL) =====
        # conn = psycopg2.connect(
        #     host=host,
        #     port=port,
        #     database="postgres",
        #     user=user,
        #     password=password,
        #     sslmode="require"
        # )

        # ===== FREE ALTERNATIVE (local Postgres has no SSL) =====
        conn = psycopg2.connect(
            host=host,
            port=port,
            database="postgres",
            user=user,
            password=password,
            sslmode=os.getenv("POSTGRES_SSLMODE", "prefer")
        )
        # Enable autocommit mode before executing CREATE DATABASE
        conn.autocommit = True
        
        cursor = conn.cursor()
        try:
            # Check if database exists
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (target_db,)
            )
            
            if not cursor.fetchone():
                print(f"Database '{target_db}' does not exist. Creating...")
                cursor.execute(f"CREATE DATABASE {target_db}")
                print(f"Database '{target_db}' created successfully.")
            else:
                print(f"Database '{target_db}' already exists.")
        finally:
            cursor.close()
            conn.close()
    except Exception as exc:
        print(f"Failed to create database: {exc}")
        raise
