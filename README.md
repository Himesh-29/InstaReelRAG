# InstaReelRAG

InstaReelRAG is an advanced, production-grade multimodal Retrieval-Augmented Generation (RAG) system designed to ingest, watch, listen to, and index Instagram Reels. Whether you are tracking desk setups, tech gear reviews, coding tutorials, or creative inspiration, InstaReelRAG lets you search and converse with your saved video library in natural language—backed by verifiable source citations and visual previews.

---

## Why I Built This

Ever watched an Instagram Reel showcasing a stunning desk setup, a custom mechanical keyboard, or a must-have productivity app—only to lose track of it weeks later? 

Traditional bookmarking only saves URLs. InstaReelRAG goes deeper by combining computer vision (CLIP), speech-to-text (OpenAI Whisper), lexical search (SQLite FTS5 BM25), and dense vector embeddings (ChromaDB) into a unified local search engine. You can ask complex questions like *"What 4K monitor arm did @setupsai recommend in that Reel about cable management?"* and get an instant, cited answer.

---

## Complete Project Workflow

```
┌────────────────────────────────────────────────────────────────────────┐
│                      1. ANONYMOUS IG INGESTION                         │
│  Public Profile Scraper ──► Zero-GraphQL Fallback ──► Skip Static Pins │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │  .mp4 video & caption
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   2. MULTIMODAL FEATURE EXTRACTION                     │
│  ┌─────────────────────────────┐    ┌──────────────────────────────┐   │
│  │     AUDIO TRANSCRIPTION     │    │   VISION & deduplication     │   │
│  │ ffmpeg (subprocess.run)     │    │ OpenCV over-sampling         │   │
│  │ OpenAI Whisper API (.mp3)   │    │ CLIP (clip-ViT-B-32 on GPU)  │   │
│  └──────────────┬──────────────┘    │ WebP Micro-Thumbnails (480px)│   │
│                 │                   └──────────────┬───────────────┘   │
└─────────────────┼──────────────────────────────────┼───────────────────┘
                  │                                  │
                  ▼                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 3. DUAL-DATABASE & BLOB ARCHIVING                      │
│  • instareelrag.db (SQLite)  ──► Metadata, Captions, Shortcodes        │
│  • frames.db (SQLite)        ──► Zero-Loose-File WebP Binary Blobs     │
│  • chromadb/ (ChromaDB)      ──► Dense Embeddings (all-MiniLM-L6-v2)   │
│  • videos_fts (SQLite BM25)  ──► C-Level Okapi BM25 Lexical Index      │
│  • MEDIA ERADICATION         ──► Auto-delete .mp4, .mp3, .webp, .txt   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                4. HYBRID RETRIEVAL & RE-RANKING PIPELINE               │
│  • Query Transformer ──► Resolves conversational coreference           │
│  • Hybrid Search     ──► Combines BM25 Lexical + ChromaDB Semantic     │
│  • Cross-Encoder     ──► GPU-accelerated re-ranking (ms-marco-MiniLM)  │
│  • O(1) Lookup       ──► Fetches metadata only for top candidates      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      5. GRADIO CHAT INTERFACE                          │
│  Grounded LLM Answer + Collapsible Citations Accordion (Scores & URLs) │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Key Features Incorporated

- **1. Zero-GraphQL Anonymous Scraping**: Scrapes public creator profiles without login, cookies, or 2FA checkpoints using Instagram's `web_profile_info` endpoint with mobile app headers (`X-IG-App-ID`), while skipping static photo carousels (`is_video == True`).
- **2. GPU-Accelerated Multimodal Feature Extraction**: Uses CUDA-enabled CLIP (`clip-ViT-B-32`) to deduplicate up to 10 visually unique scenes per Reel, and OpenAI Whisper via `subprocess.run` ffmpeg audio extraction for speech transcription.
- **3. WebP Micro-Thumbnail Compression & `frames.db` Archiving**: Downscales frame images to 480px WebP quality 65 (~10 KB each, a 30x storage savings) and archives them as binary blobs in `database/frames.db` without filesystem clutter.
- **4. Sequential Streaming Ingestion & 100% Media Eradication**: Checks candidates against the database before downloading any media, then downloads, processes, and eradicates `.mp4` videos, `.mp3` audio, `.webp` thumbnails, and companion `.txt`/`.json`/`.xz` files sequentially one by one—capping peak disk usage at just 1 video file.
- **5. O(1) Unified Hybrid Search & Re-Ranking**: Merges SQLite FTS5 Okapi BM25 keyword matching with ChromaDB semantic embeddings (`all-MiniLM-L6-v2`) and a CUDA Cross-Encoder re-ranker (`ms-marco-MiniLM-L-6-v2`), ranking via lightweight UUIDs with O(1) payload lookups.
- **6. Conversational Coreference Resolution**: Uses an LLM `QueryTransformer` to rewrite follow-up questions (*"How much does that keyboard cost?"*) into standalone search queries.
- **7. Per-Run Timestamped Logging with Auto-Rotation**: Generates timestamped log files in `logs/` (`run_YYYY-MM-DD_HH-MM-SS.log`) with automatic pruning of older logs based on `max_log_files`.

---

## Crucial Architectural Design Decisions & Core Technical Resolutions

- **1. Anonymous Zero-GraphQL Ingestion**: Standard Instagram GraphQL API queries trigger rate limits and 2FA checkpoints. We bypassed this by scraping public `web_profile_info` endpoints with mobile app headers (`X-IG-App-ID`) and filtering for videos (`is_video == True`) upfront.
- **2. Micro-Thumbnail Storage (480px WebP + Blob DB)**: Storing 10 JPEG frames per Reel (~3 MB/video) causes massive disk bloat. We compress frames to 480px WebP (~10 KB each, 30x smaller) and archive them directly as binary blobs in `database/frames.db`, wiping all temporary `.webp`, `.mp4`, and `.mp3` files from disk after ingestion.
- **3. O(1) Targeted Hybrid Retrieval**: Semantic vector search misses exact model numbers (`"LG 27GP850"`), while keyword search misses conceptual synonyms. We combine SQLite FTS5 Okapi BM25 with ChromaDB dense vectors and a CUDA Cross-Encoder, using lightweight UUIDs during ranking and fetching document payloads in O(1) time only for the final top candidates.
- **4. Persistent Model Residency with Transient VRAM Eviction**: Reloading PyTorch models per Reel kills throughput, but retaining intermediate tensors causes out-of-memory errors. We load AI models once into GPU VRAM for the entire run, while explicitly deleting temporary Reel dataset tensors (`del current_embedding, last_embedding`) and clearing CUDA cache after each Reel.
- **5. Conversational Coreference Resolution & Grounded Citations**: Multi-turn chat questions lack context for stateless vector search. An LLM-powered `QueryTransformer` rewrites conversational follow-ups into standalone queries, and answers are paired with a collapsible Citations Accordion showing exact relevance scores, Instagram URLs, and spoken transcripts.
- **6. Sequential Streaming Ingestion (O(1) Disk Footprint)**: Batch downloading Reels upfront wastes network bandwidth and disk space if videos are already indexed. We check shortcodes against the database before downloading any media, then download, process, and eradicate Reels sequentially one by one—capping peak disk usage at just 1 video file regardless of library size.

---

## Empirical Findings While Training

Benchmarked during a live 181-Reel continuous ingestion run with read-only database inspections taken mid-flight (zero interruption to the running pipeline):

- **1. Pipeline Throughput**: End-to-end processing (scrape + download + GPU CLIP dedup + WebP compress + SQLite/ChromaDB index + disk eradicate) averages **12.77s per Reel** (P50: 10s, max: 121s), sustaining **~282 Reels/hour** on consumer NVIDIA hardware.
- **2. 19.6% Frame Redundancy Culled by CLIP**: Out of 1,810 theoretical frame slots (10 per video), CLIP cosine-similarity deduplication retained only 1,455 unique frames (avg 8.04 / video), eliminating 355 visually redundant frames and saving both storage and downstream compute.
- **3. WebP Blob Size Distribution**: 63.8% of archived frame blobs land in the 10-30 KB nominal range, 29.1% are detail-rich (> 30 KB), and only 7.1% are lightweight (< 10 KB). Average blob size is 23.3 KB with a total binary payload of 33.04 MB for 1,455 frames.
- **4. True O(1) Disk Footprint Confirmed**: Active directory snapshots during ingestion consistently showed exactly 1 `.mp4`, 1 `.mp3`, and up to 10 `.webp` files on disk at any moment—the single Reel currently being processed. Total transient media footprint: 2.38 MB.
- **5. 100% SQLite B-Tree Utilization**: Both `instareelrag.db` (49 pages, 0.19 MB) and `frames.db` (8,783 pages, 34.31 MB) operate at 100.0% page utilization with zero freelist fragmentation, confirming efficient sequential write patterns.
- **6. Storage Budget Breakdown**: `frames.db` consumes 92.7% of total application storage (34.31 MB), followed by transient downloads (6.4%), relational metadata + FTS5 index (0.5%), and logs (0.3%). Total footprint for 181 Reels: **37.00 MB**.
- **7. Zero Rate-Limit Events**: Across 181 consecutive sequential downloads, zero 429/rate-limit errors or Instagram anti-scraping checkpoints were triggered, validating the polite jitter delay and `web_profile_info` endpoint strategy.
- **8. Log Rotation Operational**: The timestamped log system produced a single 129 KB log file (1,271 structured lines) for the entire 181-Reel run, confirming per-run isolation and auto-rotation readiness under `max_log_files` policy.

---

## Getting Started

### 1. Environment Setup
Create and activate your Python virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install Dependencies
Install the package in editable mode. 

**For standard / CPU setup:**
```powershell
pip install -e .
```

**For NVIDIA GPU Acceleration (CUDA 12.1+ Recommended):**
```powershell
pip install -e ".[gpu]" --extra-index-url https://download.pytorch.org/whl/cu121
```

### 3. Configure API Keys
Copy `.env.example` to `.env` and add your API keys:
```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENAI_API_KEY=sk-proj-your-openai-key-here
```
*(Note: Instagram login is optional—public profiles are scraped anonymously by default)*

---

## CLI Commands

### Ingest Instagram Reels
Scrape, transcribe, extract frames, and index Reels from any public Instagram creator:
```powershell
python main.py ingest --channel setupsai --max_posts 5
```
*What happens under the hood:*
1. Fetches channel profile and skips non-video posts.
2. Downloads the `.mp4` video and extracts `.mp3` audio.
3. Transcribes spoken narration via OpenAI Whisper API.
4. Extracts 10 distinct visual scenes via GPU CLIP cosine deduplication.
5. Compresses thumbnails to 480px WebP and archives them into `frames.db`.
6. Indexes captions and transcripts into ChromaDB and SQLite FTS5 BM25.
7. Automatically wipes all temporary `.mp4`, `.mp3`, `.webp`, and `.txt` files from disk.

### Launch the RAG Assistant UI
Start the web-based interactive Gradio chat application:
```powershell
python main.py chat
```
Open your browser at `http://127.0.0.1:9876` to chat with your saved Reels library.

---

## Project Structure

```
InstaReelRAG/
├── config/
│   ├── config.json         # Configurable AI models, thresholds, and log settings
│   └── logger.py           # Timestamped per-run logger with auto-rotation
├── database/
│   ├── metadata_db.py      # SQLAlchemy ORM (Channels, Videos, ImageFrames, FTS5)
│   ├── frames_db.py        # SQLite binary blob archive for WebP micro-thumbnails
│   └── vector_store.py     # ChromaDB vector store wrapper (CUDA-enabled)
├── processing/
│   └── video_processor.py  # FFMPEG audio extraction, Whisper API, GPU CLIP deduplication
├── rag/
│   ├── generator.py        # Grounded LLM answer generator with citation builder
│   ├── query_transform.py  # Conversational coreference resolution
│   ├── reranker.py         # Local Cross-Encoder re-ranker (CUDA-enabled)
│   └── retriever.py        # O(1) Hybrid search engine (BM25 + Semantic Vector)
├── scraper/
│   └── ig_scraper.py       # Zero-GraphQL anonymous Instagram profile scraper
├── main.py                 # CLI entrypoint for ingest and chat commands
├── pyproject.toml          # Project dependencies & CUDA GPU optional groups
└── README.md               # Project documentation
```

---

## License
This project is open-source and available under the MIT License.
