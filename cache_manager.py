"""
Cache management utilities for storing and retrieving processed data.
"""
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image

from config import SUPPORTED_IMAGE_EXTENSIONS, PROCESSED_IMAGES_DIR, UNDEFINED_TABLES_DIR
from image_processor import ImageProcessor
from data_processor import DataProcessor
from logger import logger


class CacheManager:
    """Handles cache operations for storing and retrieving processed data."""

    def __init__(self, cache_path: str, image_folder: str):
        """
        Initialize cache manager.

        Args:
            cache_path: Path to cache file
            image_folder: Path to image folder
        """
        self.cache_path = cache_path
        self.image_folder = image_folder
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        """
        Load cache from file.

        Returns:
            Cache dictionary
        """
        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                logger.info(f"Loaded cache with {len(cache)} entries")
                return cache
        except FileNotFoundError:
            logger.info(
                f"Cache file '{self.cache_path}' not found. Initializing a new cache.")
            return {}
        except json.JSONDecodeError:
            logger.warning(
                f"Cache file '{self.cache_path}' is empty or not valid JSON. Initializing a new cache.")
            return {}

    def save_cache(self) -> None:
        """Save cache to file."""
        try:
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False,
                          indent=4, sort_keys=False)
            logger.debug("Cache saved successfully")
        except IOError as e:
            logger.error(
                f"Error: Could not write to file {self.cache_path}. Reason: {e}")
            raise

    def get_cached_paths(self) -> set:
        """
        Get set of cached image paths.

        Returns:
            Set of cached image paths
        """
        return set(value['image_path'] for value in self.cache.values())

    def add_to_cache(self, id_value: str, data: Dict[str, Any], image_path: str) -> None:
        """
        Add entry to cache.

        Args:
            id_value: ID value
            data: Data dictionary
            image_path: Path to source image
        """
        data["image_path"] = image_path
        self.cache[id_value] = data
        logger.debug(f"Added {id_value} to cache")

    def get_from_cache(self, id_value: str) -> Optional[Dict[str, Any]]:
        """
        Get entry from cache.

        Args:
            id_value: ID value to retrieve

        Returns:
            Cached data or None if not found
        """
        return self.cache.get(id_value)

    def is_in_cache(self, image_path: str) -> bool:
        """
        Check if image path is already cached.

        Args:
            image_path: Path to check

        Returns:
            True if cached, False otherwise
        """
        return image_path in self.get_cached_paths()

    def get_cache_keys(self) -> List[str]:
        """
        Get list of cache keys.

        Returns:
            List of cache keys
        """
        return list(self.cache.keys())

    def setup_directories(self) -> tuple:
        """
        Setup necessary directories for processing.

        Returns:
            Tuple of (processed_dir, undefined_dir)
        """
        script_dir = Path(__file__).parent
        processed_dir = script_dir / PROCESSED_IMAGES_DIR
        undefined_dir = Path(self.image_folder) / UNDEFINED_TABLES_DIR

        processed_dir.mkdir(exist_ok=True)
        undefined_dir.mkdir(exist_ok=True)

        return str(processed_dir), str(undefined_dir)

    def process_new_images(self, model) -> None:
        """
        Process new images and add to cache.

        Args:
            model: Gemini model for processing
        """
        logger.info("Looking through images...")

        processed_dir, undefined_dir = self.setup_directories()
        cached_paths = self.get_cached_paths()

        files_to_remove = []
        files_to_move = []

        # Process images that haven't been cached
        for filename in os.listdir(self.image_folder):
            image_path = os.path.join(self.image_folder, filename)

            # Skip if not an image file or already cached
            if (not self._is_image_file(filename) or
                os.path.isdir(image_path) or
                    image_path in cached_paths):
                continue

            # Process image
            try:
                img = Image.open(image_path)
                processed_img = ImageProcessor.process_image(img)
                processed_path = os.path.join(processed_dir, filename)
                processed_img.save(processed_path)

                # Send to model for processing
                logger.info(f"Sending {filename} to bot...")
                response = model.generate_content([processed_img])
                content = DataProcessor.clean_json_response(response.text)

                if not content:
                    logger.warning(
                        f"Skipping {filename} - no valid JSON response")
                    files_to_move.append(image_path)
                    files_to_remove.append(processed_path)
                    continue

                if not content['table']:
                    logger.warning(f"Empty table in {filename}")
                    files_to_move.append(image_path)
                    files_to_remove.append(processed_path)
                    if content['id'] in self.cache:
                        continue

                # Add to cache
                self.add_to_cache(content['id'], content, image_path)

            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                files_to_move.append(image_path)
                files_to_remove.append(processed_path)

        # Move problematic files
        for file_path in files_to_move:
            shutil.move(file_path, undefined_dir)

        for file_path in files_to_remove:
            if os.path.exists(file_path):
                os.remove(file_path)

        # Save updated cache
        self.save_cache()
        logger.info("Done looking through images!")

    def _is_image_file(self, filename: str) -> bool:
        """
        Check if file is a supported image file.

        Args:
            filename: Filename to check

        Returns:
            True if supported image file, False otherwise
        """
        return Path(filename).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
