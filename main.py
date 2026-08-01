import argparse
import os
import uuid
from config import setup_logger

logger = setup_logger()

def ingest(channel_name: str, max_posts: int):
    logger.info(f"=== Starting Ingestion for channel '@{channel_name}' (max posts: {max_posts}) ===")
    
    from database.metadata_db import init_db, Video, Channel, ImageFrame
    from rag.retriever import HybridRetriever
    from scraper.ig_scraper import IGScraper
    from processing.video_processor import VideoProcessor
    from tqdm import tqdm
    
    # 1. Init DB, Scraper, and HybridRetriever
    db_session = init_db()
    scraper = IGScraper()
    video_proc = VideoProcessor()
    retriever = HybridRetriever(db_session=db_session)
    
    # 2. Get or create Channel
    channel = db_session.query(Channel).filter_by(username=channel_name).first()
    if not channel:
        channel = Channel(username=channel_name)
        db_session.add(channel)
        db_session.commit()
        
    # 3. Fetch profile post candidates sequentially without downloading all files upfront
    logger.info(f"Fetching profile metadata sequentially for: @{channel_name}")
    post_candidates = scraper.get_profile_posts_metadata(channel_name, max_posts=max_posts)
    logger.info(f"Found {len(post_candidates)} video post candidates from @{channel_name}.")
    
    docs_to_add = []
    metas_to_add = []
    ids_to_add = []
    
    processed_count = 0
    for candidate in tqdm(post_candidates, desc="Processing Reels sequentially"):
        if processed_count >= max_posts:
            break
            
        # Check if already in DB BEFORE downloading .mp4!
        shortcode = getattr(candidate, "shortcode", None)
        if not shortcode:
            continue
        existing_video = db_session.query(Video).filter_by(shortcode=shortcode).first()
        if existing_video:
            logger.info(f"Video '{shortcode}' already in database. Skipping download & processing.")
            continue
            
        # Download just this single Reel sequentially
        post = scraper.download_single_post(candidate, channel_name)
        if not post:
            continue
        processed_count += 1
            
        logger.info(f"Processing new video '{post['shortcode']}'...")
        video = Video(
            channel_id=channel.id,
            shortcode=post['shortcode'],
            caption=post['caption'],
            timestamp=post['timestamp'],
            local_video_path=post['video_path']
        )
        db_session.add(video)
        db_session.commit() # Commit to get ID
        
        # 4. Extract Frames
        frames_dir = os.path.join(scraper.download_dir, channel_name, "frames")
        frame_paths = video_proc.extract_frames(post['video_path'], frames_dir)
        
        for p in frame_paths:
            frame = ImageFrame(video_id=video.id, local_image_path=p)
            db_session.add(frame)
        
        # 5. Extract Audio & Transcribe (Optional, if OpenAI key is present)
        transcript = ""
        audio_path = post['video_path'].replace(".mp4", ".mp3")
        if os.environ.get("OPENAI_API_KEY"):
            if video_proc.extract_audio(post['video_path'], audio_path):
                transcript = video_proc.transcribe_audio(audio_path)
                
        # 5.1 Immediately clean up heavy .mp4, .mp3, and .webp files after processing and DB archiving
        from config import get_config
        if get_config()["processing"].get("delete_media_after_processing", False):
            import shutil
            # Include video, audio, extracted WebP frames, and any companion files (.txt, .xz, .json)
            base_no_ext = os.path.splitext(post['video_path'])[0]
            companion_files = [f for f in os.listdir(os.path.dirname(post['video_path'])) if f.startswith(os.path.basename(base_no_ext))]
            companion_paths = [os.path.join(os.path.dirname(post['video_path']), f) for f in companion_files]
            
            for file_path in [post['video_path'], audio_path] + frame_paths + companion_paths:
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logger.warning(f"Could not remove temporary file {file_path}: {e}")
            # Try to remove empty frames directory if clean
            if os.path.exists(frames_dir) and not os.listdir(frames_dir):
                try:
                    os.rmdir(frames_dir)
                except Exception:
                    pass
            logger.info(f"Cleaned up temporary audio, video, and frame (.webp) files for '{post['shortcode']}' to keep downloads folder empty.")
                
        # 6. Prepare for Vector Store
        # Combine caption and transcript
        full_text = f"Caption: {post['caption']}\n\nTranscript: {transcript}"
        
        if len(full_text.strip()) > 15: # Only add if there is meaningful text
            docs_to_add.append(full_text)
            metas_to_add.append({
                "video_id": video.id,
                "shortcode": post['shortcode'],
                "url": post['url'],
                "channel": channel_name
            })
            ids_to_add.append(str(uuid.uuid4()))
            
        # Free GPU VRAM cache after processing each reel so memory never piles up across thousands of reels
        try:
            from config import ECOSYSTEM
            ECOSYSTEM.clear_gpu_cache()
        except Exception:
            pass
            
    db_session.commit()
    
    # 7. Add to HybridRetriever (indexes into both ChromaDB VectorStore and SQLite FTS5 BM25 in a single call)
    if docs_to_add:
        retriever.add_documents(docs_to_add, metas_to_add, ids_to_add)
        logger.info(f"=== Successfully indexed {len(docs_to_add)} new documents into ChromaDB and SQLite FTS5 BM25 ===")
    else:
        logger.info("=== Ingestion complete. No new documents to add ===")

def chat_ui(server_name: str = "0.0.0.0", server_port: int = 7860):
    print(f"Launching Gradio Chat Interface on {server_name}:{server_port}...")
    import gradio as gr
    from database.metadata_db import init_db
    from rag.retriever import HybridRetriever
    from rag.reranker import LocalReranker
    from rag.generator import ContextAnswerGenerator
    from rag.query_transform import QueryTransformer
    
    db_session = init_db()
    
    retriever = HybridRetriever(db_session=db_session)
    has_docs = retriever.ensure_indexed()
        
    reranker = LocalReranker()
    generator = ContextAnswerGenerator()
    query_transformer = QueryTransformer()
    
    def chat_function(message, history):
        if not has_docs:
            return "No documents found in the database. Please run the ingest command first!"
            
        logger.info(f"User Chat Input: '{message}'")
        
        # 1. Rephrase Query
        optimized_query = query_transformer.rephrase_query(message, history)
        if optimized_query != message:
            logger.info(f"Optimized Query: '{optimized_query}'")
        
        # 2. Retrieve top IDs (using hybrid_top_k from config.json)
        hybrid_results = retriever.hybrid_search(optimized_query)
        logger.info(f"Hybrid search retrieved {len(hybrid_results)} candidate documents.")
        
        # 3. Targeted O(1) lookup: fetch metadata ONLY for the retrieved IDs
        retrieved_ids = [res['id'] for res in hybrid_results]
        targeted_docs = retriever.vector_store.collection.get(ids=retrieved_ids)
        
        # Create lookup map for fast attaching
        doc_map = {}
        for i, doc_id in enumerate(targeted_docs['ids']):
            doc_map[doc_id] = {
                "metadata": targeted_docs['metadatas'][i],
                "content": targeted_docs['documents'][i]
            }
            
        # Attach metadata and content to results
        for res in hybrid_results:
            if res['id'] in doc_map:
                res['metadata'] = doc_map[res['id']]['metadata']
                if 'content' not in res:
                    res['content'] = doc_map[res['id']]['content']
                
        # 4. Rerank (using rerank_top_k from config.json)
        reranked = reranker.rerank(optimized_query, hybrid_results)
        logger.info(f"Reranking completed: top {len(reranked)} documents selected.")
        
        # 5. Generate Answer
        answer = generator.generate_answer(message, reranked)
        logger.info("Answer generated successfully.")
        
        # 6. Build Citations Accordion
        citations_html = "\n\n<details>\n<summary>📚 <b>View Retrieved Context (Citations)</b></summary>\n\n"
        for i, doc in enumerate(reranked):
            url = doc['metadata'].get('url', 'Unknown URL')
            score = doc.get('rerank_score', 0)
            content = doc.get('content', '')
            
            citations_html += f"**Source {i+1}** (Score: {score:.2f}) - [Link]({url})\n"
            citations_html += f"```text\n{content}\n```\n\n"
            
        citations_html += "</details>"
            
        return answer + citations_html

    demo = gr.ChatInterface(
        fn=chat_function,
        title="InstaReelRAG Assistant",
        description="Ask questions about the scraped Instagram setups and tech gear!",
    )
    demo.launch(server_name=server_name, server_port=server_port, share=False, theme="soft")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InstaReelRAG CLI")
    subparsers = parser.add_subparsers(dest="command")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Scrape and process an Instagram profile")
    ingest_parser.add_argument("--channel", type=str, required=True, help="Instagram username (e.g., setupsai)")
    ingest_parser.add_argument("--max_posts", type=int, default=5, help="Max posts to scrape")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Launch the Gradio Chat UI")
    chat_parser.add_argument("--server-name", type=str, default="0.0.0.0", help="Host interface to bind to")
    chat_parser.add_argument("--server-port", type=int, default=7860, help="Port to bind to")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        ingest(args.channel, args.max_posts)
    elif args.command == "chat":
        chat_ui(server_name=args.server_name, server_port=args.server_port)
    else:
        parser.print_help()
