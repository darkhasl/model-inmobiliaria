import json
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.evaluate import evaluate_model
@pytest.mark.usefixtures("run_pipeline")
def test_metrics_generation():
    evaluate_model()
    
    with open('../data/scores/metrics.json') as f:
        metrics = json.load(f)
        
    assert 'mae' in metrics
    assert metrics['r2'] <= 1.0