-- Run once in Neon SQL Editor (https://console.neon.tech)
CREATE EXTENSION IF NOT EXISTS vector;

-- file_registry is created automatically by the app on startup.
-- LangChain PGVector creates its own tables when documents are indexed.
