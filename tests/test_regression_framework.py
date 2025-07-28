"""
Tests for the performance regression framework.

Validates baseline management, regression detection, and reporting functionality.
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime

import sys
from pathlib import Path

# Add the results directory to Python path
results_path = Path(__file__).parent / "results" / "performance"
sys.path.insert(0, str(results_path))

from regression_framework import (
    PerformanceBaseline,
    PerformanceRegression,
    create_performance_regression_framework
)


@pytest.fixture
def temp_results_dir():
    """Create temporary directory for test results"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_performance_result(temp_results_dir):
    """Create sample performance test result file"""
    result_data = {
        "test_name": "test_embedding_performance",
        "timestamp": datetime.now().timestamp(),
        "system_info": {
            "cpu_count": 8,
            "memory_gb": 16.0,
            "python_version": "3.12.2",
            "platform": "darwin"
        },
        "statistics": {
            "count": 10,
            "mean": 150.5,
            "median": 145.0,
            "min": 120.0,
            "max": 200.0,
            "stdev": 25.3,
            "p95": 190.0,
            "p99": 195.0
        },
        "measurements": [
            {"label": "embedding_0", "elapsed_ms": 145.2, "timestamp": datetime.now().timestamp()},
            {"label": "embedding_1", "elapsed_ms": 152.1, "timestamp": datetime.now().timestamp()},
        ],
        "metadata": {"entity_count": 10}
    }
    
    result_file = temp_results_dir / "test_embedding_performance_123456.json"
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    return result_file


class TestPerformanceBaseline:
    """Test performance baseline management"""
    
    def test_baseline_creation(self, temp_results_dir):
        """Test creating new baseline manager"""
        baseline_file = temp_results_dir / "baselines.json"
        baseline = PerformanceBaseline(baseline_file)
        
        assert not baseline_file.exists()
        assert baseline.baselines is not None
        assert "embedding_performance" in baseline.baselines
    
    def test_set_baseline(self, temp_results_dir):
        """Test setting performance baseline"""
        baseline_file = temp_results_dir / "baselines.json"
        baseline = PerformanceBaseline(baseline_file)
        
        baseline.set_baseline("test_category", "mean", 100.0, 0.1)
        
        assert "test_category" in baseline.baselines
        assert "mean" in baseline.baselines["test_category"]
        assert baseline.baselines["test_category"]["mean"]["baseline_value"] == 100.0
        assert baseline.baselines["test_category"]["mean"]["tolerance"] == 0.1
    
    def test_save_and_load_baselines(self, temp_results_dir):
        """Test saving and loading baselines"""
        baseline_file = temp_results_dir / "baselines.json"
        
        # Create and save baseline
        baseline1 = PerformanceBaseline(baseline_file)
        baseline1.set_baseline("test_category", "mean", 100.0, 0.1)
        baseline1.save_baselines()
        
        assert baseline_file.exists()
        
        # Load baseline in new instance
        baseline2 = PerformanceBaseline(baseline_file)
        assert "test_category" in baseline2.baselines
        assert baseline2.baselines["test_category"]["mean"]["baseline_value"] == 100.0
    
    def test_regression_detection_no_regression(self, temp_results_dir):
        """Test regression detection when performance is within range"""
        baseline_file = temp_results_dir / "baselines.json"
        baseline = PerformanceBaseline(baseline_file)
        
        baseline.set_baseline("test_category", "mean", 100.0, 0.1)  # ±10%
        
        # Test value within range
        is_regression, message = baseline.check_regression("test_category", "mean", 105.0)
        assert not is_regression
        assert "WITHIN_RANGE" in message
    
    def test_regression_detection_with_regression(self, temp_results_dir):
        """Test regression detection when performance degrades"""
        baseline_file = temp_results_dir / "baselines.json"
        baseline = PerformanceBaseline(baseline_file)
        
        baseline.set_baseline("test_category", "mean", 100.0, 0.1)  # ±10%
        
        # Test value that represents regression (>10% worse)
        is_regression, message = baseline.check_regression("test_category", "mean", 120.0)
        assert is_regression
        assert "REGRESSION" in message
        assert "20.0%" in message
    
    def test_regression_detection_improvement(self, temp_results_dir):
        """Test detection of performance improvements"""
        baseline_file = temp_results_dir / "baselines.json"
        baseline = PerformanceBaseline(baseline_file)
        
        baseline.set_baseline("test_category", "mean", 100.0, 0.1)
        
        # Test improved performance
        is_regression, message = baseline.check_regression("test_category", "mean", 80.0)
        assert not is_regression
        assert "IMPROVEMENT" in message
        assert "20.0%" in message
    
    def test_regression_detection_no_baseline(self, temp_results_dir):
        """Test regression detection when no baseline exists"""
        baseline_file = temp_results_dir / "baselines.json"
        baseline = PerformanceBaseline(baseline_file)
        
        is_regression, message = baseline.check_regression("nonexistent", "mean", 100.0)
        assert not is_regression
        assert "No baseline set" in message


class TestPerformanceRegression:
    """Test performance regression analysis"""
    
    def test_regression_framework_creation(self, temp_results_dir):
        """Test creating regression framework"""
        framework = create_performance_regression_framework(temp_results_dir)
        assert isinstance(framework, PerformanceRegression)
        assert framework.results_dir == temp_results_dir
    
    def test_analyze_performance_results(self, temp_results_dir, sample_performance_result):
        """Test analyzing performance results"""
        framework = PerformanceRegression(temp_results_dir)
        
        analysis = framework.analyze_performance_results(sample_performance_result)
        
        assert analysis["test_name"] == "test_embedding_performance"
        assert "timestamp" in analysis
        assert "regressions" in analysis
        assert "improvements" in analysis
        assert "no_baseline" in analysis
        
        # Should have no baseline initially
        assert len(analysis["no_baseline"]) == 2  # mean and p95
    
    def test_set_baselines_from_results(self, temp_results_dir, sample_performance_result):
        """Test setting baselines from test results"""
        framework = PerformanceRegression(temp_results_dir)
        
        framework.set_baselines_from_results()
        
        # Check that baselines were set
        baselines = framework.baseline_manager.baselines
        assert "test_embedding_performance" in baselines
        assert "mean" in baselines["test_embedding_performance"]
        assert "p95" in baselines["test_embedding_performance"]
        
        # Verify baseline values match the sample data
        assert baselines["test_embedding_performance"]["mean"]["baseline_value"] == 150.5
        assert baselines["test_embedding_performance"]["p95"]["baseline_value"] == 190.0
    
    def test_tolerance_assignment(self, temp_results_dir):
        """Test that appropriate tolerances are assigned"""
        framework = PerformanceRegression(temp_results_dir)
        
        # Test known test types
        assert framework._get_tolerance_for_test("single_embedding_performance") == 0.15
        assert framework._get_tolerance_for_test("payload_search_performance") == 0.2
        assert framework._get_tolerance_for_test("semantic_search_performance") == 0.25
        assert framework._get_tolerance_for_test("unknown_test") == 0.2  # default
    
    def test_regression_report_generation(self, temp_results_dir, sample_performance_result):
        """Test generating regression report"""
        framework = PerformanceRegression(temp_results_dir)
        
        # Set baselines first
        framework.set_baselines_from_results()
        
        # Generate report
        report = framework.generate_regression_report()
        
        assert "Performance Regression Report" in report
        assert "Summary" in report
        assert "Tests analyzed:" in report
        assert "Baseline Status" in report
    
    def test_regression_detection_workflow(self, temp_results_dir):
        """Test complete regression detection workflow"""
        framework = PerformanceRegression(temp_results_dir)
        
        # Create initial baseline result
        baseline_result = {
            "test_name": "test_performance",
            "timestamp": datetime.now().timestamp(),
            "statistics": {"mean": 100.0, "p95": 150.0}
        }
        
        baseline_file = temp_results_dir / "baseline_result.json"
        with open(baseline_file, 'w') as f:
            json.dump(baseline_result, f)
        
        # Set baselines
        framework.set_baselines_from_results()
        
        # Create new result with regression
        regression_result = {
            "test_name": "test_performance", 
            "timestamp": datetime.now().timestamp(),
            "statistics": {"mean": 130.0, "p95": 200.0}  # Regression
        }
        
        regression_file = temp_results_dir / "regression_result.json"
        with open(regression_file, 'w') as f:
            json.dump(regression_result, f)
        
        # Analyze for regressions
        analysis = framework.analyze_performance_results(regression_file)
        
        # Should detect regressions (30% increase exceeds 20% tolerance)
        assert len(analysis["regressions"]) > 0
        
        regression = analysis["regressions"][0]
        assert "REGRESSION" in regression["message"]
        assert regression["value"] == 130.0


class TestIntegration:
    """Integration tests for regression framework"""
    
    @pytest.mark.asyncio
    async def test_real_performance_data_integration(self, temp_results_dir):
        """Test integration with real performance test data structure"""
        # Create realistic performance data
        performance_data = {
            "test_name": "single_embedding_performance",
            "timestamp": datetime.now().timestamp(),
            "system_info": {
                "cpu_count": 8,
                "memory_gb": 16.0,
                "python_version": "3.12.2",
                "platform": "darwin"
            },
            "statistics": {
                "count": 30,
                "mean": 287.5,  # Matches our actual performance docs
                "median": 280.0,
                "min": 150.0,
                "max": 445.0,
                "stdev": 65.2,
                "p95": 445.0,
                "p99": 445.0
            },
            "measurements": [],
            "metadata": {"entity_count": 30}
        }
        
        # Save performance data
        result_file = temp_results_dir / "single_embedding_performance_12345.json"
        with open(result_file, 'w') as f:
            json.dump(performance_data, f, indent=2)
        
        # Test framework operations
        framework = PerformanceRegression(temp_results_dir)
        framework.set_baselines_from_results()
        
        # Verify baseline establishment
        baselines = framework.baseline_manager.baselines
        assert "single_embedding_performance" in baselines
        assert baselines["single_embedding_performance"]["mean"]["baseline_value"] == 287.5
        assert baselines["single_embedding_performance"]["mean"]["tolerance"] == 0.15  # 15% for embedding
        
        # Test regression detection with target performance
        is_regression, message = framework.baseline_manager.check_regression(
            "single_embedding_performance", "mean", 500.0  # Target limit
        )
        
        # 500ms vs 287.5ms baseline = 73.9% increase, should be regression (>15% tolerance)
        assert is_regression
        assert "REGRESSION" in message
    
    def test_framework_with_empty_results_dir(self, temp_results_dir):
        """Test framework behavior with no existing results"""
        framework = PerformanceRegression(temp_results_dir)
        
        # Should handle empty directory gracefully
        framework.set_baselines_from_results()
        report = framework.generate_regression_report()
        
        assert "No performance results found" in report
    
    def test_framework_with_invalid_json(self, temp_results_dir):
        """Test framework behavior with corrupted result files"""
        # Create invalid JSON file
        invalid_file = temp_results_dir / "invalid_result.json"
        with open(invalid_file, 'w') as f:
            f.write("{ invalid json content")
        
        framework = PerformanceRegression(temp_results_dir)
        
        # Should handle invalid files gracefully
        framework.set_baselines_from_results()
        report = framework.generate_regression_report()
        
        assert isinstance(report, str)
        assert "Performance Regression Report" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])