# Deploy: Neon + Render

## Platform roles

| Platform | Use for this project | Why |
|----------|----------------------|-----|
| **Neon** | PostgreSQL + **pgvector** | Vector store + `file_registry` tables |
| **Render** | **Gradio Python app** (Docker) | Long-running process, heavy deps (torch, embeddings) |

---

## 1. Neon (database)

1. Create a project at [console.neon.tech](https://console.neon.tech).
2. Open **SQL Editor** and run:

   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

   (Or run `scripts/neon_init.sql`.)

3. Copy the **pooled** connection string (`postgresql://...?sslmode=require`).
4. Keep it secret — you will paste it into Render as `DATABASE_URL`.

---

## 2. Render (Gradio app)

### Option A — Blueprint (recommended)

1. Push this repo to GitHub (without `.env`, `.venv`, or `uploaded_files/`).
2. [Render Dashboard](https://dashboard.render.com) → **New** → **Blueprint**.
3. Connect the repo; Render reads `render.yaml`.
4. Set **secret** env vars when prompted:
   - `DATABASE_URL` → Neon pooled URL
   - `GROQ_API_KEY` → your Groq key
5. Deploy. First build may take **10–20 minutes** (torch + sentence-transformers).
6. Open the Render URL (e.g. `https://rag-gradio.onrender.com`).

### Option B — Manual Web Service

1. **New Web Service** → connect repo.
2. **Runtime:** Docker  
3. **Dockerfile path:** `./Dockerfile`  
4. **Instance type:** at least **Starter (512MB)**; **Standard (2GB+)** is safer for embeddings.
5. **Environment variables** (same as `.env.example`):

   | Key | Value |
   |-----|--------|
   | `DATABASE_URL` | Neon connection string |
   | `GROQ_API_KEY` | Groq API key |
   | `GRADIO_SERVER_NAME` | `0.0.0.0` |
   | `GRADIO_SHARE` | `false` |

   Render injects `PORT` automatically.

### Uploads on Render

The app stores files under `uploaded_files/` on disk. Render’s filesystem is **ephemeral** (files can disappear on redeploy). For production, enable **S3** in env vars or attach a [Render persistent disk](https://render.com/docs/disks).

---

## Local vs production env

**Local** (unchanged):

```bash
cd E:\RagApp
.\.venv\Scripts\activate
python -m RagApplication
```

**Production** (Render sets these; you can mirror locally):

```env
DATABASE_URL=postgresql://...@...neon.tech/neondb?sslmode=require
GROQ_API_KEY=...
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SHARE=false
```

---

## Code changes made for deployment

- `config.py` — reads `DATABASE_URL` / `POSTGRES_URL` (Neon).
- `ui.py` — binds `GRADIO_SERVER_NAME` and `PORT` (Render).
- `Dockerfile` + `render.yaml` — container deploy on Render.

No retrieval, embedding, Groq, or Gradio flow logic was changed.

---

## Checklist before going live

- [ ] `.env` is **not** committed (rotate keys if it ever was).
- [ ] Neon `vector` extension enabled.
- [ ] `DATABASE_URL` and `GROQ_API_KEY` set on Render.
- [ ] GitHub repo pushed without `.venv` / large files.
- [ ] First deploy finished; open Render URL and upload a test PDF.
