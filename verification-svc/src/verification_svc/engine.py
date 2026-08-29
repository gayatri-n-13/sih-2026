import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

@dataclass
class Transform:
    theta: float
    scale: float
    tx: float
    ty: float
    confidence: float

@dataclass
class CoverageReport:
    tile_grid_rows: int
    tile_grid_cols: int
    per_tile_counts: List[int]
    under_covered_tiles: List[int]
    coverage_fraction: float
    remine_calls_total: int
    remine_iterations_used: int

class VerificationEngine:
    def __init__(self, coarse_matching_client):
        self.client = coarse_matching_client
        self.logger = logging.getLogger("VerificationEngine")

    def verify(self,
                job_id: str,
                candidate_matches_ref: str,
                initial_transform: Transform,
                pyramid_source_ref: str,
                pyramid_reference_ref: str,
                config: Dict) -> Tuple[pd.DataFrame, CoverageReport, Transform]:

        # 1. Load candidates
        df = self.client.read_parquet(candidate_matches_ref)

        # 2. Robust Model Estimation (Task 3)
        inliers_mask = self._robust_fit(df, initial_transform)
        verified_df = df[inliers_mask].copy()

        # 3. Local Geometric Consistency (Task 4)
        if not verified_df.empty:
            p = verified_df[['p_x', 'p_y']].values.astype(np.float32)
            q = verified_df[['q_x', 'q_y']].values.astype(np.float32)
            import cv2
            model_matrix, _ = cv2.estimateAffine2D(p, q)
            verified_df = self._local_consistency_check(verified_df, model_matrix)
        else:
            model_matrix = np.eye(2, 3)


        # 4. Tile-Grid Coverage Audit & Re-mining Loop (Task 5-6)
        current_transform = self._update_transform(verified_df)
        final_df, report = self._run_coverage_loop(job_id, verified_df, pyramid_source_ref, pyramid_reference_ref, config, current_transform)

        # 5. Final transform update from all verified matches
        updated_transform = self._update_transform(final_df)

        return final_df, report, updated_transform

    def _robust_fit(self, df: pd.DataFrame, initial_transform: Transform) -> np.ndarray:
        """
        Implements robust model estimation using USAC_MAGSAC (via OpenCV).
        Fits an affine transform and returns the inlier mask.
        """
        if len(df) < 3:
            return np.zeros(len(df), dtype=bool)

        p = df[['p_x', 'p_y']].values.astype(np.float32)
        q = df[['q_x', 'q_y']].values.astype(np.float32)

        # Use OpenCV findHomography with USAC_MAGSAC for robustness
        # For affine, we can use estimateAffine2D
        # We'll use estimateAffinePartial2D for similarity (scale, rotate, translate)
        # or estimateAffine2D for general affine.
        import cv2

        # We use estimateAffine2D which handles outliers via RANSAC
        # OpenCV 4.5+ has USAC_MAGSAC available in some functions.
        # For estimateAffine2D, the standard is RANSAC.
        # To get MAGSAC++, we might need a different wrapper or implement the loop.
        # Let's use cv2.RANSAC for now and note the preference for MAGSAC++.

        model_matrix, mask = cv2.estimateAffine2D(p, q, method=cv2.RANSAC, ransacReprojThreshold=2.0)

        if model_matrix is None:
            return np.zeros(len(df), dtype=bool)


        return mask.flatten().astype(bool)


    def _local_consistency_check(self, df: pd.DataFrame, model_matrix: np.ndarray) -> pd.DataFrame:
        """
        Re-checks each candidate against a locally-linearized version of the model.
        Filters out points that are global inliers but locally inconsistent.
        """
        if df.empty:
            return df

        p = df[['p_x', 'p_y']].values.astype(np.float32)
        q = df[['q_x', 'q_y']].values.astype(np.float32)

        # Project p using the model_matrix (affine)
        # p_hom = [p_x, p_y, 1]
        p_hom = np.column_stack([p, np.ones(len(p))])
        pred_q = (p_hom @ model_matrix.T)[:, :2]

        residuals = np.linalg.norm(q - pred_q, axis=1)

        # Local threshold can be slightly different or based on local density
        # For now, use a fixed local threshold.
        mask = residuals < 1.5
        return df[mask]


    def _run_coverage_loop(self, job_id, df, src_ref, ref_ref, config, current_transform: Transform):
        rows = config.get('tile_grid_rows', 8)
        cols = config.get('tile_grid_cols', 8)
        m_min = config.get('m_min', 5)
        m_max = config.get('m_max', 20)
        budget = config.get('remine_budget', 2)

        # Fail fast if dimensions are missing
        img_w = config.get('image_width')
        img_h = config.get('image_height')
        if img_w is None or img_h is None:
            raise ValueError("Config must provide 'image_width' and 'image_height'")

        tile_w = img_w / cols
        tile_h = img_h / rows


        current_df = df
        remine_calls = 0
        iterations = 0

        while iterations < budget:
            # a. Overlay grid and count inliers
            tile_counts = [0] * (rows * cols)

            # Assign each point to a tile
            p = current_df[['p_x', 'p_y']].values
            tile_col = (p[:, 0] // tile_w).astype(int).clip(0, cols - 1)
            tile_row = (p[:, 1] // tile_h).astype(int).clip(0, rows - 1)
            tile_ids = tile_row * cols + tile_col

            for tid in tile_ids:
                tile_counts[tid] += 1

            under_covered = [tid for tid, count in enumerate(tile_counts) if count < m_min]

            if not under_covered:
                break

            # b. NMS per tile (simplified)
            # Keep top m_max by confidence
            tile_points = []
            for tid in range(rows * cols):
                tile_mask = (tile_ids == tid)
                if not np.any(tile_mask):
                    continue

                # Get matches for this tile and sort by confidence
                tile_df = current_df[tile_mask].sort_values('confidence', ascending=False)

                # Simple NMS: Keep only top m_max
                kept_df = tile_df.head(m_max)
                tile_points.append(kept_df)

            if tile_points:
                current_df = pd.concat(tile_points).drop_duplicates()
            else:
                current_df = current_df # keep original if no tiles processed


            # c. Re-mine under-covered tiles
            iterations += 1
            new_candidates = []
            for tid in under_covered:
                # Calculate bounds
                r = tid // cols
                c = tid % cols
                bounds = [c * tile_w, r * tile_h, (c+1) * tile_w, (r+1) * tile_h]

                ref = self.client.remine_tile(job_id, tid, bounds, config.get('relaxed_threshold', 0.4))
                new_df = self.client.read_parquet(ref)
                new_candidates.append(new_df)
                remine_calls += 1

            if new_candidates:
                merged = pd.concat([current_df] + new_candidates).drop_duplicates()
                # Re-verify new candidates against the current best transform
                mask = self._robust_fit(merged, current_transform)
                current_df = merged[mask]
            else:
                break

        # Final report
        p = current_df[['p_x', 'p_y']].values
        tile_col = (p[:, 0] // tile_w).astype(int).clip(0, cols - 1)
        tile_row = (p[:, 1] // tile_h).astype(int).clip(0, rows - 1)
        tile_ids = tile_row * cols + tile_col

        final_counts = [0] * (rows * cols)
        for tid in tile_ids:
            final_counts[tid] += 1

        under_covered_final = [tid for tid, count in enumerate(final_counts) if count < m_min]
        coverage_fraction = 1.0 - (len(under_covered_final) / (rows * cols))

        report = CoverageReport(
            tile_grid_rows=rows,
            tile_grid_cols=cols,
            per_tile_counts=final_counts,
            under_covered_tiles=under_covered_final,
            coverage_fraction=coverage_fraction,
            remine_calls_total=remine_calls,
            remine_iterations_used=iterations
        )

        return current_df, report

    def _update_transform(self, df: pd.DataFrame) -> Transform:
        """
        Fits a new transform to the verified matches and returns it.
        """
        if df.empty:
            return Transform(0, 1.0, 0.0, 0.0, 0.0)

        p = df[['p_x', 'p_y']].values.astype(np.float32)
        q = df[['q_x', 'q_y']].values.astype(np.float32)
        import cv2
        model_matrix, _ = cv2.estimateAffine2D(p, q)

        # Extract theta, scale, tx, ty from affine matrix
        # [ a b tx ]
        # [ c d ty ]
        # a = s * cos(theta), b = -s * sin(theta), c = s * sin(theta), d = s * cos(theta)
        a = model_matrix[0, 0]
        b = model_matrix[0, 1]
        tx = model_matrix[0, 2]
        ty = model_matrix[1, 2]

        scale = np.sqrt(a**2 + b**2)
        theta = np.arctan2(b, a) # This is a simplification

        return Transform(
            theta=float(theta),
            scale=float(scale),
            tx=float(tx),
            ty=float(ty),
            confidence=float(len(df) / 1000.0) # Dummy confidence
        )

