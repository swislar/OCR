"""
Refactored Gemini API client for OCR processing.
"""
import os
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Optional, Dict, Any

from config import DEFAULT_MODEL_NAME, DEFAULT_CACHE_PATH, DEFAULT_IMAGE_FOLDER, SYSTEM_INSTRUCTION
from cache_manager import CacheManager
from cost_tracker import CostTracker
from similarity_matcher import SimilarityMatcher
from logger import logger


class GeminiFlash:
    """Refactored Gemini API client for OCR processing."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        cache_path: str = DEFAULT_CACHE_PATH,
        image_folder: str = DEFAULT_IMAGE_FOLDER,
        cache_refresh: bool = False
    ):
        """
        Initialize GeminiFlash client.

        Args:
            model_name: Name of the Gemini model to use
            cache_path: Path to cache file
            image_folder: Path to image folder
            cache_refresh: Whether to refresh cache on initialization
        """
        self._setup_api()
        self._model_name = model_name
        self._cache_manager = CacheManager(cache_path, image_folder)
        self._cost_tracker = CostTracker()
        self._setup_models()

        if cache_refresh:
            self._refresh_cache()

    def _setup_api(self) -> None:
        """Setup Google API configuration."""
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set. Please set it.")
        genai.configure(api_key=api_key)

    def _setup_models(self) -> None:
        """Setup Gemini models."""
        self._model = genai.GenerativeModel(
            model_name=self._model_name,
            system_instruction=SYSTEM_INSTRUCTION
        )
        self._id_model = genai.GenerativeModel(model_name=self._model_name)

    def get_similar_id(self, target_id: str) -> Optional[Dict[str, Any]]:
        """
        Find similar ID using fuzzy matching and Gemini verification.

        Args:
            target_id: Target ID to find

        Returns:
            Matched data dictionary or None if not found
        """
        # Try fuzzy matching first
        if self._cache_manager.cache:
            match_result = SimilarityMatcher.find_best_match(
                target_id, self._cache_manager.cache)
            if match_result:
                return match_result[1]  # Return the data dictionary

        # Try Gemini matching if fuzzy matching failed
        logger.info("Verifying with Gemini...")
        if self._cache_manager.cache:
            return self._gemini_match_id(target_id)

        return None

    def _gemini_match_id(self, target_id: str) -> Optional[Dict[str, Any]]:
        """
        Find similar ID using Gemini model.

        Args:
            target_id: Target ID to match

        Returns:
            Matched data dictionary or None if not found
        """
        id_list = self._cache_manager.get_cache_keys()
        prompt = SimilarityMatcher.create_id_matching_prompt(
            target_id, id_list)

        try:
            response = self._id_model.generate_content([prompt])

            # Track usage
            self._cost_tracker.add_id_matching_usage(
                response.usage_metadata.prompt_token_count,
                response.usage_metadata.candidates_token_count
            )

            derived_id = response.text.strip()
            logger.debug(f"Derived_id: {derived_id}")

            if derived_id in self._cache_manager.cache:
                return self._cache_manager.cache[derived_id]
            else:
                return None

        except Exception as e:
            logger.error(f"An error occurred during Gemini ID matching: {e}")
            return None

    def _refresh_cache(self) -> None:
        """Refresh cache by processing new images."""
        self._cache_manager.process_new_images(self._model)

    def estimate_cost(self) -> None:
        """Print cost estimation report."""
        self._cost_tracker.print_cost_report(include_image_processing=True)

    @property
    def cache(self) -> Dict[str, Any]:
        """Get cache dictionary."""
        return self._cache_manager.cache

    @property
    def cost_tracker(self) -> CostTracker:
        """Get cost tracker instance."""
        return self._cost_tracker
