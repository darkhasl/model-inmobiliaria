import json
import pytest
from src.models.evaluate import evaluate_model

def test_metrics_generation():
    evaluate_model()
    
    with open('data/scores/metrics.json') as f:
        metrics = json.load(f)
        
    assert 'mae' in metrics
    assert metrics['r2'] <= 1.0