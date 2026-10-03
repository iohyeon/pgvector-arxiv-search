-- pgvector-arxiv-search schema
-- Design notes: docs/DESIGN.md

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- papers: one row per arXiv paper (metadata + processing state)
-- authors/categories are kept as arrays for display and GIN filtering;
-- the normalized author relation below is the source of truth for author filters.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS papers (
    id               BIGSERIAL PRIMARY KEY,
    arxiv_id         TEXT        NOT NULL UNIQUE,          -- version-less id, e.g. 2610.01981
    title            TEXT        NOT NULL,
    abstract         TEXT        NOT NULL,
    authors          TEXT[]      NOT NULL DEFAULT '{}',
    categories       TEXT[]      NOT NULL DEFAULT '{}',
    primary_category TEXT,
    published        DATE        NOT NULL,
    updated          DATE,
    pdf_url          TEXT,
    pdf_path         TEXT,
    status           TEXT        NOT NULL DEFAULT 'metadata'
                     CHECK (status IN ('metadata', 'embedded', 'failed')),
    has_fulltext     BOOLEAN     NOT NULL DEFAULT FALSE,
    error            TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS papers_categories_gin ON papers USING GIN (categories);
CREATE INDEX IF NOT EXISTS papers_published_idx  ON papers (published DESC);

-- ---------------------------------------------------------------------------
-- authors: deduplicated by normalized name (lowercase, no accents/dots)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS authors (
    id              BIGSERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    normalized_name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS paper_authors (
    paper_id  BIGINT   NOT NULL REFERENCES papers(id)  ON DELETE CASCADE,
    author_id BIGINT   NOT NULL REFERENCES authors(id) ON DELETE CASCADE,
    position  SMALLINT NOT NULL,                       -- 1-based author order
    PRIMARY KEY (paper_id, author_id)
);

CREATE INDEX IF NOT EXISTS paper_authors_author_idx ON paper_authors (author_id);

-- ---------------------------------------------------------------------------
-- chunks: retrieval unit. Text + vector live in the same row so a search
-- never needs a second lookup to show the matched passage.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    id              BIGSERIAL PRIMARY KEY,
    paper_id        BIGINT      NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    chunk_index     INTEGER     NOT NULL,
    section         TEXT        NOT NULL,                -- 'abstract', '3 Method', ...
    page            INTEGER,                             -- NULL for the abstract
    content         TEXT        NOT NULL,
    token_count     INTEGER     NOT NULL,                -- model tokenizer, no special tokens
    embedding       vector(384) NOT NULL,                -- all-MiniLM-L6-v2, L2-normalized
    embedding_model TEXT        NOT NULL,
    tsv             tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    UNIQUE (paper_id, chunk_index)
);

-- ANN index. vector_cosine_ops must match the <=> operator used in queries.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin   ON chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS chunks_paper_idx ON chunks (paper_id);

-- ---------------------------------------------------------------------------
-- keep papers.updated_at honest
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION touch_updated_at() RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS papers_touch_updated_at ON papers;
CREATE TRIGGER papers_touch_updated_at
    BEFORE UPDATE ON papers
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();
