import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

async def get_connection():
    """Returns a connection to the PostgreSQL database."""
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is not set")

    try:
        conn = await asyncpg.connect(DATABASE_URL)
        return conn
    except asyncpg.InvalidCatalogNameError as e:
        raise Exception("Database catalog does not exist. Check DATABASE_URL.") from e
    except asyncpg.CannotConnectNowError as e:
        raise Exception("Cannot connect to database — server may be down.") from e
    except asyncpg.PostgresError as e:
        raise Exception(f"PostgreSQL error: {str(e)}") from e
    except Exception as e:
        raise Exception(f"Database connection failed: {str(e)}") from e


async def init_db():
    """Creates all tables on startup if they don't exist."""
    conn = None
    try:
        conn = await get_connection()

        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                joined_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
            )
            """
        )

        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS questions (
                id SERIAL PRIMARY KEY,
                topic VARCHAR(100) NOT NULL,
                difficulty VARCHAR(20) NOT NULL,
                question TEXT NOT NULL,
                correct_answer TEXT NOT NULL,
                choices TEXT,
                created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
            )
            """
        )

        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS quiz_results (
                id SERIAL PRIMARY KEY,
                student_id INT NOT NULL REFERENCES students(id),
                question_id INT REFERENCES questions(id),
                topic VARCHAR(100) NOT NULL,
                question TEXT NOT NULL,
                student_answer TEXT NOT NULL,
                correct_answer TEXT NOT NULL,
                is_correct BOOLEAN NOT NULL,
                time_taken INT,
                feedback TEXT,
                submitted_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
            )
            """
        )

        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS performance_history (
                id SERIAL PRIMARY KEY,
                student_id INT NOT NULL REFERENCES students(id),
                topic VARCHAR(100) NOT NULL,
                total_attempts INT NOT NULL,
                correct_attempts INT NOT NULL,
                average_accuracy DECIMAL(5,2) NOT NULL,
                last_updated TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
                UNIQUE(student_id, topic)
            )
            """
        )

    finally:
        if conn:
            await conn.close()



