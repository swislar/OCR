"""
Similarity matching utilities for ID comparison.
"""
import re
import regex
from rapidfuzz import fuzz
from typing import List, Tuple, Optional, Dict, Any

from config import SIMILARITY_THRESHOLD
from data_processor import DataProcessor
from logger import logger


class SimilarityMatcher:
    """Handles similarity matching for ID comparison."""

    @staticmethod
    def similarity_score(code_1: str, code_2: str) -> float:
        """
        Calculate similarity score between two codes.

        Args:
            code_1: First code string
            code_2: Second code string

        Returns:
            Similarity score (0-100)
        """
        clean_code_1 = re.sub(r'[^a-zA-Z0-9]', '', code_1)
        clean_code_2 = re.sub(r'[^a-zA-Z0-9]', '', code_2)
        return fuzz.ratio(clean_code_1, clean_code_2)

    @staticmethod
    def similarity_score_stripped(code_1: str, code_2: str) -> float:
        """
        Calculate similarity score between two codes with brackets removed.

        Args:
            code_1: First code string
            code_2: Second code string

        Returns:
            Similarity score (0-100)
        """
        code_1 = regex.sub(r'\((?:[^()]|(?R))*\)', "", code_1)
        code_2 = regex.sub(r'\((?:[^()]|(?R))*\)', "", code_2)
        clean_code_1 = re.sub(r'[^a-zA-Z0-9]', '', code_1)
        clean_code_2 = re.sub(r'[^a-zA-Z0-9]', '', code_2)
        return fuzz.ratio(clean_code_1, clean_code_2)

    @staticmethod
    def find_best_match(target_id: str, cache: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any], float]]:
        """
        Find the best matching ID from cache using fuzzy matching.

        Args:
            target_id: Target ID to match
            cache: Cache dictionary containing ID mappings

        Returns:
            Tuple of (matched_id, data_dict, similarity_score) or None if no match
        """
        if not cache:
            return None

        best_matches = []

        for id_value, dict_object in cache.items():
            clean_id = DataProcessor.clean_id(id_value)
            similarity_score = SimilarityMatcher.similarity_score(
                clean_id, target_id)
            stripped_similarity_score = SimilarityMatcher.similarity_score_stripped(
                clean_id, target_id)

            best_matches.append((
                id_value,
                dict_object,
                similarity_score,
                stripped_similarity_score
            ))

        # Sort by regular similarity score first
        best_matches.sort(key=lambda x: x[2], reverse=True)

        if best_matches[0][2] >= SIMILARITY_THRESHOLD:
            logger.info(
                f"Found match with similarity score: {best_matches[0][2]}")
            logger.debug(
                f"Original ID: {best_matches[0][0]}, Clean ID: {DataProcessor.clean_id(best_matches[0][0])}")
            return best_matches[0][0], best_matches[0][1], best_matches[0][2]

        # Try stripped similarity score
        best_matches.sort(key=lambda x: x[3], reverse=True)

        if best_matches[0][3] >= SIMILARITY_THRESHOLD:
            logger.info(
                f"Found match with stripped similarity score: {best_matches[0][3]}")
            logger.debug(
                f"Original ID: {best_matches[0][0]}, Clean ID: {DataProcessor.clean_id(best_matches[0][0])}")
            return best_matches[0][0], best_matches[0][1], best_matches[0][3]

        # No match found
        logger.warning(
            f"No match found for {target_id}. Best similarity score: {best_matches[0][2]}")
        logger.debug(
            f"Best candidate: {best_matches[0][0]}, Clean ID: {DataProcessor.clean_id(best_matches[0][0])}")
        return None

    @staticmethod
    def create_id_matching_prompt(target_id: str, id_list: List[str]) -> str:
        """
        Create a prompt for Gemini ID matching.

        Args:
            target_id: Target ID to match
            id_list: List of available IDs

        Returns:
            Formatted prompt string
        """
        return f"""
        You are a precise and silent ID matching tool. Your purpose is to find the single best match for a target ID from a list.

        Your instructions are absolute:
        1.  You will analyze the 'ID to match' and compare it against the 'ID list'.
        3.  If the result is a clear, unambiguous match, you will return the original ID from the list.
        4.  If there is no clear match, you will return the exact string 'NA'.

        Your output must be ONLY the matched ID or the string 'NA'. Do not provide any explanation, preamble, justification, or any text other than the result.

        ---
        Example 1:
        ID list = ['FF(G)/EF1152', 'FF(G)1152 (VIRTEX-4: XQ4VFX100)', 'CLIENT-789-C']
        ID to match = 'FF1152'
        Matched ID from list = ?

        FF(G)/EF1152

        Example 2:
        ID list = ['FF(G)/EF1152', 'FF(G)1152 (VIRTEX-4: XQ4VFX100)', 'CLIENT-789-C']
        ID to match = 'FF1152 (XQ4VFX100)'
        Matched ID from list = ?

        FF(G)1152 (VIRTEX-4: XQ4VFX100)

        Example 3 (No Match):
        ID list = ['PROD-A-999', 'PROD-B-101', 'DEV-C-500']
        ID to match = 'cus'
        Matched ID from list = ?

        NA
        ---

        Here is your task:

        ID list = {id_list}
        ID to match = {target_id}
        Matched ID from list = ?
        """
