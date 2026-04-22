"""PostgreSQL + pgvector database layer for face recognition pipeline."""

import psycopg2
from psycopg2.extras import execute_values
from config import DATABASE_URL


def get_connection():
    """Return a new database connection."""
    return psycopg2.connect(DATABASE_URL)


def create_tables():
    """Create all required tables (idempotent)."""
    conn = get_connection()
    cur = conn.cursor()
    # Use pgvector if available, otherwise fall back to TEXT for embeddings
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        embedding_type = "VECTOR(512)"
    except Exception:
        conn.rollback()
        embedding_type = "TEXT"
        print("pgvector not available — storing embeddings as TEXT (cosine similarity runs in Python).")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS person_embeddings (
            id SERIAL PRIMARY KEY,
            person_id INTEGER REFERENCES persons(id),
            embedding {embedding_type},
            photo_path VARCHAR(500),
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS stream_chunks (
            id SERIAL PRIMARY KEY,
            stream_id VARCHAR(100),
            chunk_index INTEGER,
            started_at TIMESTAMP,
            ended_at TIMESTAMP,
            processed_at TIMESTAMP
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS speaker_sightings (
            id SERIAL PRIMARY KEY,
            chunk_id INTEGER REFERENCES stream_chunks(id),
            person_id INTEGER REFERENCES persons(id),
            spoke_from TIMESTAMP,
            spoke_until TIMESTAMP,
            duration_seconds INTEGER,
            avg_confidence FLOAT,
            detection_count INTEGER
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("Tables created successfully.")


# ── Persons ──────────────────────────────────────────────────

def insert_person(name):
    """Insert a person and return their id."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO persons (name) VALUES (%s) RETURNING id;",
        (name,),
    )
    person_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return person_id


def get_person_by_name(name):
    """Return person row by name, or None."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM persons WHERE name = %s;", (name,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def get_or_create_person(name):
    """Return person id, creating the person if they don't exist."""
    row = get_person_by_name(name)
    if row:
        return row[0]
    return insert_person(name)


# ── Embeddings ───────────────────────────────────────────────

def insert_embedding(person_id, embedding, photo_path):
    """Store a 512d embedding for a person."""
    conn = get_connection()
    cur = conn.cursor()
    embedding_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"
    cur.execute(
        "INSERT INTO person_embeddings (person_id, embedding, photo_path) "
        "VALUES (%s, %s, %s);",
        (person_id, embedding_str, photo_path),
    )
    conn.commit()
    cur.close()
    conn.close()


def get_all_embeddings():
    """Return list of (person_id, person_name, embedding) tuples."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT pe.person_id, p.name, pe.embedding
        FROM person_embeddings pe
        JOIN persons p ON p.id = pe.person_id;
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    results = []
    for person_id, name, emb_str in rows:
        # pgvector returns embedding as a string like "[0.1,0.2,...]"
        emb = [float(x) for x in emb_str.strip("[]").split(",")]
        results.append((person_id, name, emb))
    return results


# ── Stream chunks ────────────────────────────────────────────

def insert_chunk(stream_id, chunk_index, started_at, ended_at):
    """Insert a stream chunk record and return its id."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO stream_chunks (stream_id, chunk_index, started_at, ended_at) "
        "VALUES (%s, %s, %s, %s) RETURNING id;",
        (stream_id, chunk_index, started_at, ended_at),
    )
    chunk_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return chunk_id


def mark_chunk_processed(chunk_id):
    """Set processed_at timestamp on a chunk."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE stream_chunks SET processed_at = NOW() WHERE id = %s;",
        (chunk_id,),
    )
    conn.commit()
    cur.close()
    conn.close()


# ── Speaker sightings ───────────────────────────────────────

def insert_sighting(chunk_id, person_id, spoke_from, spoke_until,
                    duration_seconds, avg_confidence, detection_count):
    """Insert a speaker sighting record."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO speaker_sightings "
        "(chunk_id, person_id, spoke_from, spoke_until, duration_seconds, "
        "avg_confidence, detection_count) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s);",
        (chunk_id, person_id, spoke_from, spoke_until,
         duration_seconds, avg_confidence, detection_count),
    )
    conn.commit()
    cur.close()
    conn.close()


# ── Query helpers (for API) ──────────────────────────────────

def get_all_persons():
    """Return list of all persons as dicts."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT p.id, p.name, p.created_at,
               COUNT(pe.id) AS embedding_count
        FROM persons p
        LEFT JOIN person_embeddings pe ON pe.person_id = p.id
        GROUP BY p.id, p.name, p.created_at
        ORDER BY p.id;
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {"id": r[0], "name": r[1], "created_at": r[2].isoformat(), "embedding_count": r[3]}
        for r in rows
    ]


def delete_person(person_id):
    """Delete a person and their embeddings. Returns True if found."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM person_embeddings WHERE person_id = %s;", (person_id,))
    cur.execute("DELETE FROM persons WHERE id = %s RETURNING id;", (person_id,))
    deleted = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return deleted is not None


def get_sightings_by_stream(stream_id):
    """Return all sightings for a given stream_id."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT ss.id, COALESCE(p.name, 'Unknown'), ss.person_id,
               ss.spoke_from, ss.spoke_until, ss.duration_seconds,
               ss.avg_confidence, ss.detection_count,
               sc.stream_id, sc.chunk_index
        FROM speaker_sightings ss
        JOIN stream_chunks sc ON sc.id = ss.chunk_id
        LEFT JOIN persons p ON p.id = ss.person_id
        WHERE sc.stream_id = %s
        ORDER BY ss.spoke_from;
    """, (stream_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {
            "id": r[0], "person_name": r[1], "person_id": r[2],
            "spoke_from": r[3].isoformat(), "spoke_until": r[4].isoformat(),
            "duration_seconds": r[5], "avg_confidence": r[6],
            "detection_count": r[7], "stream_id": r[8], "chunk_index": r[9],
        }
        for r in rows
    ]


def get_all_stream_ids():
    """Return list of all stream sessions with metadata."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT sc.stream_id,
               MIN(sc.started_at) AS started,
               MAX(sc.ended_at) AS ended,
               COUNT(DISTINCT sc.id) AS chunks,
               COUNT(ss.id) AS sightings
        FROM stream_chunks sc
        LEFT JOIN speaker_sightings ss ON ss.chunk_id = sc.id
        GROUP BY sc.stream_id
        ORDER BY started DESC;
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {
            "stream_id": r[0],
            "started_at": r[1].isoformat() if r[1] else None,
            "ended_at": r[2].isoformat() if r[2] else None,
            "chunk_count": r[3],
            "sighting_count": r[4],
        }
        for r in rows
    ]


if __name__ == "__main__":
    create_tables()
