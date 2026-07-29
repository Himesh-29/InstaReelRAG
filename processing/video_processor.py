import os
import cv2
from openai import OpenAI
from dotenv import load_dotenv
from config.logger import setup_logger

load_dotenv()
logger = setup_logger("VideoProcessor")

class VideoProcessor:
    def __init__(self):
        # We use standard OpenAI client for Whisper API
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.clip_model = None

    def extract_frames(self, video_path: str, output_dir: str, num_frames: int = None, similarity_threshold: float = None) -> list[str]:
        """
        Extracts frames and uses CLIP for semantic deduplication to keep only unique, important frames.
        """
        from config import get_config
        config = get_config()["processing"]
        if num_frames is None:
            num_frames = config["max_frames"]
        if similarity_threshold is None:
            similarity_threshold = config["similarity_threshold"]

        if not os.path.exists(video_path):
            logger.warning(f"Video not found: {video_path}")
            return []
            
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []
            
        # 1. Over-sample candidate frames (e.g., 5x the requested amount to find the best distinct ones)
        candidate_count = min(num_frames * 5, total_frames)
        frame_indices = [int(i * total_frames / (candidate_count + 1)) for i in range(1, candidate_count + 1)]
        
        video_filename = os.path.basename(video_path).split('.')[0]
        
        # Load CLIP model lazily once per instance to prevent GPU VRAM leak/accumulation across reels
        from sentence_transformers import SentenceTransformer, util
        from PIL import Image
        import torch
        if self.clip_model is None:
            from config import get_device
            device = get_device()
            logger.info(f"Loading CLIP model '{config['clip_model']}' on device: {device.upper()}...")
            self.clip_model = SentenceTransformer(config["clip_model"], device=device)
        
        accepted_paths = []
        last_embedding = None
        
        for idx, frame_idx in enumerate(frame_indices):
            if len(accepted_paths) >= num_frames:
                break # We have enough distinct frames
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Convert OpenCV BGR frame to RGB for PIL/CLIP
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # 2. Get CLIP Embedding
            current_embedding = self.clip_model.encode(pil_img, convert_to_tensor=True)
            
            # 3. Deduplicate: compare with the last accepted frame
            is_unique = True
            if last_embedding is not None:
                similarity = util.cos_sim(current_embedding, last_embedding).item()
                if similarity >= similarity_threshold:
                    is_unique = False # Too similar, skip it
                    
            if is_unique:
                # 4. Resize to max width 480px (maintaining aspect ratio) for ultra-compact storage
                h, w = frame.shape[:2]
                if w > 480:
                    scale = 480.0 / float(w)
                    resized_frame = cv2.resize(frame, (480, int(h * scale)), interpolation=cv2.INTER_AREA)
                else:
                    resized_frame = frame

                # 5. Save as WebP (.webp) with quality 65 (~10 KB per image instead of 300 KB JPEG)
                out_path = os.path.join(output_dir, f"{video_filename}_frame_{idx}.webp")
                cv2.imwrite(out_path, resized_frame, [int(cv2.IMWRITE_WEBP_QUALITY), 65])
                
                # 6. Also store in dedicated frames.db SQLite database for zero-loose-file archiving
                try:
                    from database.frames_db import save_frame_blob
                    with open(out_path, "rb") as f:
                        save_frame_blob(video_filename, idx, f.read())
                except Exception as db_err:
                    logger.debug(f"Could not archive WebP blob to frames.db: {db_err}")

                accepted_paths.append(out_path)
                last_embedding = current_embedding
                
        cap.release()
        # Explicitly delete temporary Reel dataset/frame CUDA tensors while keeping self.clip_model loaded
        try:
            del last_embedding
            del current_embedding
        except NameError:
            pass
        from config import clear_gpu_memory
        clear_gpu_memory()
        return accepted_paths

    def extract_audio(self, video_path: str, output_audio_path: str) -> bool:
        """Extracts audio from video using imageio-ffmpeg binary (no system PATH required)."""
        if not os.path.exists(video_path):
            return False
            
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            ffmpeg_exe = "ffmpeg"

        # Using subprocess.run with argument array to avoid Windows cmd.exe quote parsing errors on paths with spaces
        import subprocess
        logger.info(f"Extracting audio from '{video_path}' to '{output_audio_path}'...")
        try:
            cmd = [
                ffmpeg_exe,
                "-i", video_path,
                "-q:a", "0",
                "-map", "a",
                output_audio_path,
                "-y",
                "-loglevel", "error"
            res = subprocess.run(cmd, check=False)
            if res.returncode != 0:
                # Windows unsigned integer overflow for negative return codes (4294967274 is -22 / EINVAL, no audio stream)
                if res.returncode in [4294967274, -22]:
                    logger.info("Video contains no audio stream (silent Reel). Skipping audio extraction.")
                else:
                    logger.error(f"ffmpeg audio extraction failed with return code {res.returncode}")
                return False
            logger.info("Audio extraction successful.")
            return True
        except Exception as e:
            logger.error(f"Failed to extract audio with ffmpeg: {e}")
            return False

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribes audio using OpenAI Whisper API."""
        if not os.path.exists(audio_path):
            logger.warning(f"Audio file '{audio_path}' not found for transcription.")
            return ""
            
        logger.info(f"Transcribing audio file '{audio_path}' with OpenAI Whisper API...")
        try:
            with open(audio_path, "rb") as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file, 
                    response_format="text"
                )
            logger.info(f"Whisper transcription completed successfully ({len(transcript)} characters).")
            return transcript
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""
