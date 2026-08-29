import pytest
import numpy as np
from evaluation_svc.app.metrics import EvaluationEngine

def test_compute_rmse_perfect():
    engine = EvaluationEngine()
    pts1 = np.array([[0, 0], [1, 1], [2, 2]])
    pts2 = np.array([[0, 0], [1, 1], [2, 2]])
    assert engine.compute_rmse(pts1, pts2) == 0.0

def test_compute_rmse_known():
    engine = EvaluationEngine()
    # diffs: [0,0], [1,0], [0,1] -> squares: 0, 1, 1 -> mean: 2/3 -> sqrt(2/3) approx 0.816
    pts1 = np.array([[0, 0], [1, 0], [0, 1]])
    pts2 = np.array([[0, 0], [0, 0], [0, 0]])
    expected = np.sqrt(2/3)
    assert pytest.approx(engine.compute_rmse(pts1, pts2)) == expected

def test_coverage_uniformity_perfect():
    engine = EvaluationEngine()
    # All tiles have exactly 10 matches
    report = {"per_tile_counts": [10] * 100}
    assert engine.compute_coverage_uniformity(report) == 1.0

def test_coverage_uniformity_terrible():
    engine = EvaluationEngine()
    # One tile has 100 matches, others have 0
    counts = [0] * 100
    counts[0] = 100
    report = {"per_tile_counts": counts}
    # mean = 1.0, std = sqrt((100^2 + 0*99)/100) = sqrt(100) = 10.0
    # uniformity = 1 - (10/1) = -9.0 -> clamped to 0.0
    assert engine.compute_coverage_uniformity(report) == 0.0

def test_coverage_uniformity_mixed():
    engine = EvaluationEngine()
    # Constant 10, but one tile has 20.
    counts = [10] * 99
    counts[0] = 20
    report = {"per_tile_counts": counts}
    # mean = (99*10 + 20)/100 = 10.1
    # variance = (99*(10-10.1)^2 + (20-10.1)^2)/100 = (99*0.01 + 9.9^2)/100 = (0.99 + 98.01)/100 = 99/100 = 0.99
    # std = sqrt(0.99) approx 0.9949
    # uniformity = 1 - (0.9949 / 10.1) approx 0.901
    res = engine.compute_coverage_uniformity(report)
    assert 0.9 < res < 1.0
