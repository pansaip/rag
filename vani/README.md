# vani - Vanilla RAG

Vanilla RAG is an independent document-chat application.

## Structure

- `api/` - FastAPI application, PDF extraction, chunking, embeddings, and Qdrant search.
- `data/qdrant/` - Persistent local Qdrant collection (ignored by Git).
- `requirements.txt` - Python dependencies.
- `app/` - Next.js App Router page and global styles.
- `lib/` - typed API client for the FastAPI backend.

## Configure Gemini and run the API

Create a local `.env` from the safe template once, then add your Gemini API key and an available model ID. This guarded copy preserves an existing `.env`:

```powershell
Set-Location .\vani
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

Get a key from Google AI Studio. Choose an exact model ID currently available to your key from the [Gemini models list](https://ai.google.dev/gemini-api/docs/models), then set `GEMINI_API_KEY` and `GEMINI_MODEL` in `.env`. Keep the key in this backend file; never use a `NEXT_PUBLIC_` variable for it.

In one PowerShell terminal, install backend dependencies manually and start FastAPI:

```powershell
Set-Location .\vani
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload --env-file .env
```

Uvicorn reads the ignored `.env` file. No Gemini API call happens until a user selects **AI Answer** and submits a question. API docs are available at `http://127.0.0.1:8000/docs` while FastAPI runs.

## Run the frontend

Use Node.js 20.9 or newer. In a second PowerShell terminal, install the frontend dependencies once and start Next.js:

```powershell
Set-Location .\vani
npm.cmd install
$env:NEXT_PUBLIC_API_BASE_URL = "http://127.0.0.1:8000"
npm.cmd run dev
```

The API base URL defaults to `http://127.0.0.1:8000`, so setting the environment variable is optional. The frontend runs at `http://localhost:3000`.

## Test document search

With both services running, open `http://localhost:3000` and upload a PDF no larger than 10 MB. Choose the uploaded document or **All documents**, then select **AI Answer** for a grounded answer with source markers and citation excerpts, or **Search Sources** to inspect retrieved chunks and similarity scores without an LLM answer. Uploaded document names and metadata are kept in the current browser session; indexed chunks remain in local Qdrant.
