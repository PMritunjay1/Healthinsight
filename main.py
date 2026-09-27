"""
IEEE Multi-Agent Medical Pipeline — Single Entry Point
=======================================================
Usage:
    python run_pipeline.py              # Run all splits + stats + graphs
    python run_pipeline.py --eval       # Run evaluation only
    python run_pipeline.py --graphs     # Run graph generation only
    python run_pipeline.py --stats      # Recompute statistics only
"""
import asyncio
import sys
import logging
from backend.evaluation import Evaluator, compute_statistics
from backend.datasets import DEV_CASES, VALIDATION_CASES, BENCHMARK_CASES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Pipeline")


async def run_evaluation():
    """Run all 3 dataset splits through the pipeline."""
    splits = [
        ("dev_results", DEV_CASES),
        ("val_results", VALIDATION_CASES),
        ("benchmark_results", BENCHMARK_CASES),
    ]
    for name, cases in splits:
        evaluator = Evaluator()
        await evaluator.run_experiment(name, cases)

    compute_statistics()
    logger.info("All evaluation splits complete. Statistics saved.")


def run_graphs():
    """Generate all 8 IEEE-grade graphs from real data."""
    from backend.graphs import (
        _load_all_splits, _validate_records,
        plot_accuracy, plot_latency, plot_confidence, plot_agreement,
        plot_ablation, plot_reliability_curve, plot_error_distribution,
        plot_agent_contribution
    )

    data = _load_all_splits()
    data = _validate_records(data)

    plot_accuracy(data)
    plot_latency(data)
    plot_confidence(data)
    plot_agreement(data)
    plot_ablation(data)
    plot_reliability_curve(data)
    plot_error_distribution(data)
    plot_agent_contribution(data)

    logger.info(f"All 8 graphs generated from {len(data)} records.")


async def run_all():
    """Full pipeline: evaluation → statistics → graphs."""
    await run_evaluation()
    run_graphs()
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    args = set(sys.argv[1:])

    if "--eval" in args:
        asyncio.run(run_evaluation())
    elif "--graphs" in args:
        run_graphs()
    elif "--stats" in args:
        compute_statistics()
    else:
        asyncio.run(run_all())
