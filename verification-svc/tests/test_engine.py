import pytest
import numpy as np
import pandas as pd
from verification_svc.engine import VerificationEngine, Transform
from verification_svc.mocks import CoarseMatchingMock

@pytest.fixture
def mock_client():
    return CoarseMatchingMock()

@pytest.fixture
def engine(mock_client):
    return VerificationEngine(mock_client)

def test_robust_outlier_rejection(engine, mock_client):
    """
    Synthetic outlier-rejection accuracy test:
    Generate candidate set with known inlier/outlier ratio and transform.
    """
    # 100 matches, 30% inliers
    inlier_ratio = 0.3
    ref, job_id = mock_client.generate_initial_candidates(num_matches=100, inlier_ratio=inlier_ratio)
    df = mock_client.read_parquet(ref)

    initial_transform = Transform(theta=0, scale=1.0, tx=10.0, ty=20.0, confidence=1.0)
    inliers_mask = engine._robust_fit(df, initial_transform)

    num_inliers = np.sum(inliers_mask)
    print(f"\n[Outlier Rejection] Injected Ratio: {inlier_ratio}, Recovered Inliers: {num_inliers}/100")
    assert 20 <= num_inliers <= 40, f"Expected ~30 inliers, got {num_inliers}"

def test_coverage_uniformity(engine, mock_client):
    """
    Coverage-uniformity test:
    Generate DELIBERATELY CLUSTERED candidate set; assert coverage improves after re-mining.
    """
    # Create clustered data: all in one corner [0, 0, 200, 200]
    num_matches = 100
    p_x = np.random.uniform(0, 200, num_matches)
    p_y = np.random.uniform(0, 200, num_matches)
    q_x = p_x + 10.0
    q_y = p_y + 20.0

    df = pd.DataFrame({'p_x': p_x, 'p_y': p_y, 'q_x': q_x, 'q_y': q_y,
                       'source_method': 'classical', 'confidence': 1.0, 'pyramid_level': 0})

    job_id = "test_cluster_job"
    ref = "s3://mock-bucket/cluster_candidates.parquet"
    mock_client.generated_files[ref] = df

    config = {
        'tile_grid_rows': 4,
        'tile_grid_cols': 4,
        'm_min': 5,
        'm_max': 20,
        'remine_budget': 2,
        'image_width': 1000,
        'image_height': 1000,
        'relaxed_threshold': 0.4
    }

    # Initial coverage check
    rows, cols = config['tile_grid_rows'], config['tile_grid_cols']
    tile_w, tile_h = config['image_width']/cols, config['image_height']/rows
    p = df[['p_x', 'p_y']].values
    tile_ids = ((p[:, 1] // tile_h).astype(int).clip(0, rows-1) * cols +
                (p[:, 0] // tile_w).astype(int).clip(0, cols-1))
    initial_counts = np.zeros(rows * cols)
    for tid in tile_ids: initial_counts[tid] += 1
    initial_cov = 1.0 - (np.sum(initial_counts < config['m_min']) / (rows * cols))
    initial_cov_var = np.var(initial_counts)

    # Run verification
    initial_transform = Transform(theta=0, scale=1.0, tx=10.0, ty=20.0, confidence=1.0)
    final_df, report = engine._run_coverage_loop(job_id, df, "src", "ref", config)

    final_cov = report.coverage_fraction
    final_cov_var = np.var(report.per_tile_counts)

    print(f"\n[Coverage Uniformity] Initial Cov: {initial_cov:.2f} (Var: {initial_cov_var:.2f})")
    print(f"[Coverage Uniformity] Final Cov: {final_cov:.2f} (Var: {final_cov_var:.2f})")
    print(f"[Coverage Uniformity] Remine Calls: {report.remine_calls_total}")

    assert final_cov > initial_cov, f"Coverage should improve. Initial: {initial_cov}, Final: {final_cov}"
    assert final_cov_var < initial_cov_var, f"Variance should decrease. Initial: {initial_cov_var}, Final: {final_cov_var}"

def test_remining_budget(engine, mock_client):
    """
    Re-mining budget test:
    Verify the loop terminates after the configured iteration count.
    """
    # Empty dataset to force remining
    df = pd.DataFrame(columns=['p_x', 'p_y', 'q_x', 'q_y', 'source_method', 'confidence', 'pyramid_level'])

    budget = 3
    config = {
        'tile_grid_rows': 2,
        'tile_grid_cols': 2,
        'm_min': 100, # Impossible to satisfy
        'm_max': 20,
        'remine_budget': budget,
        'image_width': 1000,
        'image_height': 1000,
        'relaxed_threshold': 0.4
    }

    final_df, report = engine._run_coverage_loop("budget_job", df, "src", "ref", config)

    print(f"\n[Budget Test] Configured Budget: {budget}, Actual Iterations: {report.remine_iterations_used}")
    assert report.remine_iterations_used <= budget, f"Loop exceeded budget. Budget: {budget}, Actual: {report.remine_iterations_used}"

def test_full_verify_pipeline(engine, mock_client):
    """
    End-to-end test of the verify method to cover missing lines in engine.py.
    """
    # Generate candidates
    ref, job_id = mock_client.generate_initial_candidates(num_matches=200, inlier_ratio=0.4)

    initial_transform = Transform(theta=0, scale=1.0, tx=10.0, ty=20.0, confidence=1.0)
    config = {
        'tile_grid_rows': 4,
        'tile_grid_cols': 4,
        'm_min': 5,
        'm_max': 20,
        'remine_budget': 1,
        'image_width': 1000,
        'image_height': 1000,
        'relaxed_threshold': 0.4
    }

    final_df, report, updated_transform = engine.verify(
        job_id=job_id,
        candidate_matches_ref=ref,
        initial_transform=initial_transform,
        pyramid_source_ref="src_ref",
        pyramid_reference_ref="ref_ref",
        config=config
    )

    assert not final_df.empty
    assert updated_transform.tx != 0 or updated_transform.ty != 0
    assert report.coverage_fraction >= 0
