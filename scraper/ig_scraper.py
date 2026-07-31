import os
import instaloader
from datetime import datetime
import time
import random
from config.logger import setup_logger

logger = setup_logger("IGScraper")

class IGScraper:
    def __init__(self, download_dir: str = "./downloads"):
        self.L = instaloader.Instaloader(
            download_videos=True,
            download_video_thumbnails=False,
            download_geotags=False,
            download_comments=False,
            save_metadata=False,
            compress_json=False,
            dirname_pattern=download_dir + "/{profile}"
        )
        self.download_dir = download_dir
        
        # Load session headers dynamically from config.json to bypass GraphQL 403 blocks
        from config import get_config
        ig_config = get_config().get("instagram", {})
        self.headers = ig_config.get("headers", {
            "X-IG-App-ID": "936619743392459",
            "X-ASBD-ID": "198387",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        })
        self.L.context._session.headers.update(self.headers)
        
        # Optionally login to avoid rate limits; otherwise scrape anonymously
        ig_user = os.environ.get("INSTAGRAM_USERNAME")
        ig_pass = os.environ.get("INSTAGRAM_PASSWORD")
        if ig_user:
            session_loaded = False
            try:
                self.L.load_session_from_file(ig_user)
                logger.info(f"Loaded Instagram session for '{ig_user}' from session file (bypassing 2FA/login).")
                session_loaded = True
            except Exception:
                logger.debug(f"No valid session file found for '{ig_user}'. Attempting direct password login...")
                
            if not session_loaded and ig_pass:
                try:
                    self.L.login(ig_user, ig_pass)
                    try:
                        self.L.save_session_to_file()
                    except Exception:
                        pass
                    logger.info("Logged in to Instagram and saved session to file.")
                except Exception as e:
                    logger.warning(f"Login failed: {e}. Tip: For 2FA accounts, run 'instaloader --login={ig_user}' in your terminal first to save a session file.")
        else:
            logger.info("Scraping Instagram anonymously (no login credentials provided).")

    def _get_profile_and_posts(self, username: str, max_posts: int = 10):
        """
        Fetches an Instagram profile and up to max_posts posts.
        Falls back to the 'web_profile_info' API endpoint if standard GraphQL returns 403/ProfileNotExistsException/KeyError.
        """
        try:
            profile = instaloader.Profile.from_username(self.L.context, username)
            posts = []
            for p in profile.get_posts():
                if p.is_video:
                    posts.append(p)
                    if len(posts) >= max_posts:
                        break
            return profile, posts
        except Exception as e:
            logger.warning(f"Standard GraphQL query failed for '{username}' ({type(e).__name__}). Trying fallback 'web_profile_info' endpoint...")
            url = f"https://www.instagram.com/api/v1/users/web_profile_info/?username={username}"
            fallback_headers = {
                "Referer": f"https://www.instagram.com/{username}/",
                "Accept": "application/json, text/plain, */*",
            }
            fallback_headers.update(self.headers)
            try:
                resp = self.L.context._session.get(url, headers=fallback_headers)
                if resp.status_code == 200:
                    data = resp.json()
                    user_data = data.get("data", {}).get("user")
                    if user_data:
                        logger.info(f"Successfully retrieved profile '{username}' using fallback web_profile_info endpoint!")
                        profile = instaloader.Profile(self.L.context, user_data)
                        edges = user_data.get("edge_owner_to_timeline_media", {}).get("edges", [])
                        posts = []
                        for edge in edges:
                            node = edge.get("node")
                            if node:
                                p = instaloader.Post(self.L.context, node)
                                if p.is_video:
                                    posts.append(p)
                                    if len(posts) >= max_posts:
                                        break
                        return profile, posts
            except Exception as fallback_e:
                logger.error(f"Fallback endpoint also failed: {fallback_e}")
            logger.error("Instagram anti-scraping blocked anonymous access. Tip: Set INSTAGRAM_USERNAME and INSTAGRAM_PASSWORD in .env to scrape with a logged-in account.")
            raise e

    def get_profile_posts_metadata(self, username: str, max_posts: int = 10) -> list:
        """Fetches profile video post candidate objects without downloading media files."""
        profile, posts = self._get_profile_and_posts(username, max_posts)
        return posts

    def download_single_post(self, post, username: str) -> dict | None:
        """Downloads a single video post sequentially and returns its metadata dictionary."""
        if not getattr(post, "is_video", False):
            return None
        logger.info(f"Downloading video {post.shortcode}...")
        max_retries = 10
        for attempt in range(max_retries):
            try:
                self.L.download_post(post, target=username)
                break
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower() or attempt == max_retries - 1:
                    wait_time = (2 ** attempt) * 10
                    logger.warning(f"Download error or rate limit ({e}). Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    break

        video_filename = f"{post.date_utc:%Y-%m-%d_%H-%M-%S}_UTC.mp4"
        local_video_path = os.path.join(self.download_dir, username, video_filename)

        if os.path.exists(local_video_path):
            time.sleep(random.uniform(2.0, 5.0))
            return {
                "shortcode": post.shortcode,
                "caption": post.caption or "",
                "timestamp": post.date_utc,
                "video_path": local_video_path,
                "url": f"https://www.instagram.com/reel/{post.shortcode}/"
            }
        else:
            logger.warning(f"Expected video file not found at {local_video_path}")
            return None

    def scrape_profile(self, username: str, max_posts: int = 10) -> list[dict]:
        """Scrapes the most recent posts from a profile and downloads videos/captions."""
        posts = self.get_profile_posts_metadata(username, max_posts)
        posts_data = []
        for post in posts[:max_posts]:
            data = self.download_single_post(post, username)
            if data:
                posts_data.append(data)
        return posts_data
