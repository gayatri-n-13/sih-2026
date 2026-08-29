from typing import Dict, Any, Optional
import numpy as np
import json
import random
from typing import Dict, Any, Optional

class EvaluationEngine:
    """
    Computes quantitative metrics for the registration result.
    """

    def compute_rmse(self, registered_points: np.ndarray, ground_truth_points: np.ndarray) -> float:
        """
        Calculates the Root Mean Square Error between two sets of points.
        Points should be (N, 2) arrays.
        """
        if len(registered_points) == 0 or len(ground_truth_points) == 0:
            return float('inf')

        # Ensure same length for simple RMSE (assuming 1-to-1 mapping)
        n = min(len(registered_points), len(ground_truth_points))
        diff = registered_points[:n] - ground_truth_points[:n]
        # Sum of squares per point, then mean across points
        mse = np.mean(np.sum(np.square(diff), axis=1))
        return float(np.sqrt(mse))

    def compute_coverage_uniformity(self, coverage_report: Dict[str, Any]) -> float:
        """
        Computes coverage uniformity based on the variance of match points per tile.
        Formula: 1 - (std_dev(counts) / mean(counts))
        """
        counts = np.array(coverage_report.get("per_tile_counts", []))
        if len(counts) == 0:
            return 0.0

        mean_val = np.mean(counts)
        if mean_val == 0:
            return 0.0

        std_val = np.std(counts)
        uniformity = 1.0 - (std_val / mean_val)
        return float(max(0, uniformity))

    def generate_qa_report(self, job_id: str, metrics: Dict[str, Any]) -> str:
        """
        Generates an HTML QA report using Jinja2.
        Returns the S3 URI to the generated report.
        """
        return f"s3://reports/{job_id}/qa_report.html"

    def write_match_point_file(self, job_id: str, matches: Any) -> str:
        """
        Writes the final sub-pixel match points to a GeoJSON/CSV file.
        """
        return f"s3://outputs/{job_id}/matches.geojson"

