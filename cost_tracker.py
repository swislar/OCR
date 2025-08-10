"""
Cost tracking utilities for API usage monitoring.
"""
from typing import Dict, Any

from config import (
    INPUT_TOKEN_COST_PER_MILLION, OUTPUT_TOKEN_COST_PER_MILLION
)
from logger import logger


class CostTracker:
    """Tracks API usage costs for different operations."""

    def __init__(self):
        """Initialize cost tracker."""
        self.image_processing_stats = {
            'prompt_tokens': 0,
            'candidates_tokens': 0,
            'query_count': 0
        }
        self.id_matching_stats = {
            'prompt_tokens': 0,
            'candidates_tokens': 0,
            'query_count': 0
        }

    def add_image_processing_usage(self, prompt_tokens: int, candidates_tokens: int) -> None:
        """
        Add image processing API usage.

        Args:
            prompt_tokens: Number of prompt tokens used
            candidates_tokens: Number of candidate tokens used
        """
        self.image_processing_stats['prompt_tokens'] += prompt_tokens
        self.image_processing_stats['candidates_tokens'] += candidates_tokens
        self.image_processing_stats['query_count'] += 1

    def add_id_matching_usage(self, prompt_tokens: int, candidates_tokens: int) -> None:
        """
        Add ID matching API usage.

        Args:
            prompt_tokens: Number of prompt tokens used
            candidates_tokens: Number of candidate tokens used
        """
        self.id_matching_stats['prompt_tokens'] += prompt_tokens
        self.id_matching_stats['candidates_tokens'] += candidates_tokens
        self.id_matching_stats['query_count'] += 1

    def calculate_image_processing_cost(self) -> Dict[str, float]:
        """
        Calculate image processing costs.

        Returns:
            Dictionary with cost breakdown
        """
        input_cost = (self.image_processing_stats['prompt_tokens'] *
                      INPUT_TOKEN_COST_PER_MILLION / 1_000_000)
        output_cost = (self.image_processing_stats['candidates_tokens'] *
                       OUTPUT_TOKEN_COST_PER_MILLION / 1_000_000)
        total_cost = input_cost + output_cost

        avg_cost = (total_cost / self.image_processing_stats['query_count']
                    if self.image_processing_stats['query_count'] > 0 else 0)

        return {
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': total_cost,
            'avg_cost': avg_cost,
            'query_count': self.image_processing_stats['query_count']
        }

    def calculate_id_matching_cost(self) -> Dict[str, float]:
        """
        Calculate ID matching costs.

        Returns:
            Dictionary with cost breakdown
        """
        input_cost = (self.id_matching_stats['prompt_tokens'] *
                      INPUT_TOKEN_COST_PER_MILLION / 1_000_000)
        output_cost = (self.id_matching_stats['candidates_tokens'] *
                       OUTPUT_TOKEN_COST_PER_MILLION / 1_000_000)
        total_cost = input_cost + output_cost

        avg_cost = (total_cost / self.id_matching_stats['query_count']
                    if self.id_matching_stats['query_count'] > 0 else 0)

        return {
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': total_cost,
            'avg_cost': avg_cost,
            'query_count': self.id_matching_stats['query_count']
        }

    def get_total_cost(self) -> float:
        """
        Get total cost across all operations.

        Returns:
            Total cost in USD
        """
        img_cost = self.calculate_image_processing_cost()['total_cost']
        id_cost = self.calculate_id_matching_cost()['total_cost']
        return img_cost + id_cost

    def print_cost_report(self, include_image_processing: bool = True) -> None:
        """
        Print detailed cost report.

        Args:
            include_image_processing: Whether to include image processing costs
        """
        logger.info("====================")

        if include_image_processing:
            img_costs = self.calculate_image_processing_cost()
            logger.info("IMG PROCESSING")
            logger.info(f"    Total cost - {img_costs['query_count']} Queries")
            logger.info(f"         US$ {img_costs['total_cost']:.8f}")
            logger.info("")
            logger.info("     Average cost per query")
            logger.info(f"         US$ {img_costs['avg_cost']:.8f}")
            logger.info("--------------------")
        else:
            img_costs = {'total_cost': 0}

        id_costs = self.calculate_id_matching_cost()
        logger.info("ID MATCHING")
        logger.info(f"    Total cost - {id_costs['query_count']} Queries")
        logger.info(f"         US$ {id_costs['total_cost']:.8f}")
        logger.info("")
        logger.info("     Average cost per query")
        logger.info(f"         US$ {id_costs['avg_cost']:.8f}")
        logger.info("--------------------")

        total_cost = img_costs['total_cost'] + id_costs['total_cost']
        logger.info("TOTAL COST")
        logger.info(f"     US$ {total_cost:.8f}")
        logger.info("====================")

    def reset_stats(self) -> None:
        """Reset all usage statistics."""
        self.image_processing_stats = {
            'prompt_tokens': 0,
            'candidates_tokens': 0,
            'query_count': 0
        }
        self.id_matching_stats = {
            'prompt_tokens': 0,
            'candidates_tokens': 0,
            'query_count': 0
        }
        logger.info("Cost tracking statistics reset")
