"""
KiVA Evaluation Script
This script evaluates model submissions on the KiVA dataset and generates visualization plots.

Usage:
    python evaluate.py --submission_path path/to/submission.json --plots_dir ../plots
"""

import json
import os
from collections import Counter
from typing import Any

from config import create_config_from_args
from utils.helper import (
    LEVEL_KIVA_DB_KEY,
    LEVEL_KIVA_FUNCTIONS_COMPOSITIONALITY_DB_KEY,
    LEVEL_KIVA_FUNCTIONS_DB_KEY,
    TRANSFORMATIONS_FOR_COMPOSITE_GROUP,
    TRANSFORMATIONS_FOR_SIMPLE_GROUP,
    plot_tags,
    radar_plot_pt,
)


def load_submission_data(submission_path: str) -> list[dict[str, str]]:
    """
    Load submission data from JSON file.

    Args:
        submission_path: Path to the submission.json file

    Returns:
        List of dictionaries with 'id' and 'answer' keys
    """
    try:
        with open(submission_path) as f:
            submission_data = json.load(f)
        print(f"Loaded {len(submission_data)} predictions from {submission_path}")
        return submission_data
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Submission file not found: {submission_path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in submission file: {submission_path}") from e


def load_validation_data(metadata_path: str) -> dict[str, Any]:
    """
    Load trial data from JSON file.

    Args:
        data_dir: Directory containing the JSON file
        data_type: Type of data to load ('validation', 'unit', or 'train')

    Returns:
        Dictionary containing trial data
    """
    with open(metadata_path) as f:
        trials = json.load(f)
    print(f"Loaded {len(trials)} trials")
    return trials


def calc_top1(answers_list: list[dict[str, str]], db_dict: dict[str, Any]) -> float:
    """Calculate overall top-1 accuracy."""
    correct_count = 0
    total_count = 0
    answers_by_id = {item["id"]: item["answer"] for item in answers_list}

    for trial_id, ground_truth_info in db_dict.items():
        if trial_id in answers_by_id:
            total_count += 1
            if answers_by_id[trial_id] == ground_truth_info["correct"]:
                correct_count += 1

    return correct_count / total_count if total_count > 0 else 0.0


def calc_kiva_cat_accuracies(
    answers_list: list[dict[str, str]], db_dict: dict[str, Any]
) -> dict[str, float]:
    """
    Calculate accuracies based on LEVEL_CATEGORIES and TRANSFORMATION_CATEGORIES.

    Args:
        answers_list: List of dictionaries with 'id' and 'answer'
        db_dict: Dictionary containing the database (e.g., val_trials)

    Returns:
        Dictionary with accuracies for different categories
    """
    results_counts = Counter()
    answers_by_id = {item["id"]: item["answer"] for item in answers_list}

    for trial_id, ground_truth_info in db_dict.items():
        if trial_id not in answers_by_id:
            continue

        # Extract information from ground truth
        level = ground_truth_info.get("level", "")
        transformation_domain = ground_truth_info.get("transformation_domain", "")
        correct_answer = ground_truth_info["correct"]
        predicted_answer = answers_by_id[trial_id]

        is_correct = 1 if predicted_answer == correct_answer else 0

        # Count for overall KiVA
        results_counts[("kiva-overall", "correct")] += is_correct
        results_counts[("kiva-overall", "total")] += 1

        # Count for level-specific categories
        if level:
            level_key = f"{level}_overall"
            results_counts[(level_key, "correct")] += is_correct
            results_counts[(level_key, "total")] += 1

            # Count for transformation-specific categories within each level
            if transformation_domain:
                transform_key = f"{level}_{transformation_domain}"
                results_counts[(transform_key, "correct")] += is_correct
                results_counts[(transform_key, "total")] += 1

    # Calculate final accuracies
    final_accuracies = {}
    for key, _ in results_counts.items():
        if key[1] == "total":  # Only process "total" keys to avoid duplicates
            category = key[0]
            correct = results_counts.get((category, "correct"), 0)
            total = results_counts.get((category, "total"), 0)
            final_accuracies[category] = correct / total if total > 0 else 0.0

    return dict(sorted(final_accuracies.items()))


def evaluate_submission(
    submission_data: list[dict[str, str]], val_trials: dict[str, Any]
) -> dict[str, float]:
    """
    Evaluate submission data against validation trials.

    Args:
        submission_data: List of predictions with 'id' and 'answer'
        val_trials: Dictionary containing validation trial data

    Returns:
        Dictionary containing evaluation results
    """
    if not val_trials:
        print("Warning: 'val_trials' is empty. Skipping evaluation.")
        return {}

    # Calculate overall accuracy
    overall_accuracy = calc_top1(submission_data, val_trials)

    # Calculate category-specific accuracies
    cat_accuracies = calc_kiva_cat_accuracies(submission_data, val_trials)

    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Overall KiVA Score: {cat_accuracies.get('kiva-overall', 0.0):.4f}")

    return {"overall_accuracy": overall_accuracy, "cat_accuracies": cat_accuracies}


def print_results_by_category(cat_accuracies: dict[str, float]) -> None:
    """
    Print detailed results organized by category.

    Args:
        cat_accuracies: Dictionary with category-specific accuracy scores
    """
    print("\n--- Results by Category ---")

    if not cat_accuracies:
        print("No results available to print.")
        return

    print("\nBy Level:")
    print(f"  KiVA Score: {cat_accuracies.get(LEVEL_KIVA_DB_KEY + '_overall', 0.0):.4f}")
    print(
        f"  KiVA-functions Score: "
        f"{cat_accuracies.get(f'{LEVEL_KIVA_FUNCTIONS_DB_KEY}_overall', 0.0):.4f}"
    )
    print(
        f"  KiVA-functions-compositionality Score: "
        f"{cat_accuracies.get(f'{LEVEL_KIVA_FUNCTIONS_COMPOSITIONALITY_DB_KEY}_overall', 0.0):.4f}"
    )

    print("\nBy Transformation Category within Each Level:")
    print("  KiVA:")
    for trans_name in TRANSFORMATIONS_FOR_SIMPLE_GROUP:
        key = f"kiva_{trans_name}"
        print(f"    {key}: {cat_accuracies.get(key, 0.0):.4f}")

    print("  KiVA-functions:")
    for trans_name in TRANSFORMATIONS_FOR_SIMPLE_GROUP:
        key = f"{LEVEL_KIVA_FUNCTIONS_DB_KEY}_{trans_name}"
        print(f"    {key}: {cat_accuracies.get(key, 0.0):.4f}")

    print("  KiVA-functions-compositionality:")
    for trans_name in TRANSFORMATIONS_FOR_COMPOSITE_GROUP:
        key = f"{LEVEL_KIVA_FUNCTIONS_COMPOSITIONALITY_DB_KEY}_{trans_name}"
        print(f"    {key}: {cat_accuracies.get(key, 0.0):.4f}")


def generate_radar_plots(cat_accuracies: dict[str, float], plots_dir: str) -> None:
    """
    Generate radar plots for different KiVA levels.

    Args:
        cat_accuracies: Dictionary with category-specific accuracy scores
        plots_dir: Directory to save plots
    """
    print("\n--- Generating Radar Plots ---")

    if not cat_accuracies:
        print("No results available for plotting.")
        return

    os.makedirs(plots_dir, exist_ok=True)

    # Radar Plot for KiVA
    kiva_labels_radar = TRANSFORMATIONS_FOR_SIMPLE_GROUP
    kiva_scores_radar = {
        "Submission": [
            cat_accuracies.get(f"{LEVEL_KIVA_DB_KEY}_{trans}", 0.0) for trans in kiva_labels_radar
        ],
        "Random": [0.33] * len(kiva_labels_radar),  # Random baseline at 33%
    }
    kiva_radar_path = os.path.join(plots_dir, "kiva_radar.svg")
    radar_plot_pt(kiva_scores_radar, kiva_labels_radar, "KiVA", ["Random"], kiva_radar_path)

    # Radar Plot for KiVA-functions
    kiva_functions_labels_radar = TRANSFORMATIONS_FOR_SIMPLE_GROUP
    kiva_functions_scores_radar = {
        "Submission": [
            cat_accuracies.get(f"{LEVEL_KIVA_FUNCTIONS_DB_KEY}_{trans}", 0.0)
            for trans in kiva_functions_labels_radar
        ],
        "Random": [0.33] * len(kiva_functions_labels_radar),
    }
    kiva_functions_radar_path = os.path.join(plots_dir, "kiva_functions_radar.svg")
    radar_plot_pt(
        kiva_functions_scores_radar,
        kiva_functions_labels_radar,
        "KiVA-functions",
        ["Random"],
        kiva_functions_radar_path,
    )

    # Radar Plot for KiVA-functions-compositionality
    kiva_comp_labels_radar = TRANSFORMATIONS_FOR_COMPOSITE_GROUP
    kiva_comp_scores_radar = {
        "Submission": [
            cat_accuracies.get(f"{LEVEL_KIVA_FUNCTIONS_COMPOSITIONALITY_DB_KEY}_{trans}", 0.0)
            for trans in kiva_comp_labels_radar
        ],
        "Random": [0.33] * len(kiva_comp_labels_radar),
    }
    kiva_comp_radar_path = os.path.join(plots_dir, "kiva_compositionality_radar.svg")
    radar_plot_pt(
        kiva_comp_scores_radar,
        kiva_comp_labels_radar,
        "KiVA-functions-compositionality",
        ["Random"],
        kiva_comp_radar_path,
    )

    print(f"Radar plots saved to {plots_dir}")


def generate_bar_plots(cat_accuracies: dict[str, float], plots_dir: str) -> None:
    """
    Generate bar plots for different KiVA levels.

    Args:
        cat_accuracies: Dictionary with category-specific accuracy scores
        plots_dir: Directory to save plots
    """
    print("\n--- Generating Bar Plots ---")

    if not cat_accuracies:
        print("No results available for plotting.")
        return

    os.makedirs(plots_dir, exist_ok=True)

    # Bar Plot for KiVA
    kiva_bar_exp_results = {
        trans: cat_accuracies.get(f"{LEVEL_KIVA_DB_KEY}_{trans}", 0.0)
        for trans in TRANSFORMATIONS_FOR_SIMPLE_GROUP
    }
    kiva_bar_tags_map = dict.fromkeys(TRANSFORMATIONS_FOR_SIMPLE_GROUP, "default")
    kiva_bar_path = os.path.join(plots_dir, "kiva_bar.svg")
    plot_tags(kiva_bar_exp_results, kiva_bar_tags_map, "KiVA", kiva_bar_path)

    # Bar Plot for KiVA-functions
    kiva_functions_bar_exp_results = {
        trans: cat_accuracies.get(f"{LEVEL_KIVA_FUNCTIONS_DB_KEY}_{trans}", 0.0)
        for trans in TRANSFORMATIONS_FOR_SIMPLE_GROUP
    }
    kiva_functions_bar_tags_map = dict.fromkeys(TRANSFORMATIONS_FOR_SIMPLE_GROUP, "default")
    kiva_functions_bar_path = os.path.join(plots_dir, "kiva_functions_bar.svg")
    plot_tags(
        kiva_functions_bar_exp_results,
        kiva_functions_bar_tags_map,
        "KiVA-functions",
        kiva_functions_bar_path,
    )

    # Bar Plot for KiVA-functions-compositionality
    kiva_comp_bar_exp_results = {
        trans: cat_accuracies.get(f"{LEVEL_KIVA_FUNCTIONS_COMPOSITIONALITY_DB_KEY}_{trans}", 0.0)
        for trans in TRANSFORMATIONS_FOR_COMPOSITE_GROUP
    }
    kiva_comp_bar_tags_map = dict.fromkeys(TRANSFORMATIONS_FOR_COMPOSITE_GROUP, "default")
    kiva_comp_bar_path = os.path.join(plots_dir, "kiva_compositionality_bar.svg")
    plot_tags(
        kiva_comp_bar_exp_results,
        kiva_comp_bar_tags_map,
        "KiVA-functions-compositionality",
        kiva_comp_bar_path,
    )

    print(f"Bar plots saved to {plots_dir}")


def main(
    submission_path: str,
    metadata_path: str,
    plots_dir: str = "../plots",
) -> None:
    """
    Main evaluation pipeline.

    Args:
        submission_path: Path to submission.json file
        plots_dir: Directory to save generated plots
        metadata_path: Path to metadata.json file
    """
    print("=== KiVA Evaluation Pipeline ===")

    # Load data
    submission_data = load_submission_data(submission_path)
    val_trials = load_validation_data(metadata_path)

    # Evaluate submission
    results = evaluate_submission(submission_data, val_trials)

    if not results:
        print("Evaluation failed or no results available.")
        return

    cat_accuracies = results["cat_accuracies"]

    # Print detailed results
    print_results_by_category(cat_accuracies)

    # Generate visualizations
    generate_radar_plots(cat_accuracies, plots_dir)
    generate_bar_plots(cat_accuracies, plots_dir)

    print("\n=== Evaluation Complete ===")
    print(f"Final KiVA Score: {cat_accuracies.get('kiva-overall', 0.0):.4f}")


def run_evaluation_analysis(args, test_dataset_name: str) -> None:
    """
    Run evaluation analysis on the generated submission file.

    Args:
        args: Command line arguments containing output_dir
        test_dataset_name: Name of the test dataset to determine data type
    """
    if not args.output_dir:
        print("⚠️  No output directory available. Skipping evaluation analysis.")
        return

    submission_path = f"{args.output_dir}/submission.json"
    if not os.path.exists(submission_path):
        print(f"⚠️  Submission file not found: {submission_path}. Skipping evaluation analysis.")
        return

    print("\n🔍 Running evaluation analysis...")

    # Determine data type based on test dataset name
    config = create_config_from_args(args, for_task="test")

    # Load submission data
    with open(submission_path) as f:
        submission_data = json.load(f)
    print(f"📄 Loaded {len(submission_data)} predictions from submission file")

    # Load ground truth data
    ground_truth_data = load_validation_data(config.metadata_path)

    # Run evaluation
    results = evaluate_submission(submission_data, ground_truth_data)
    category_accuracies = results["cat_accuracies"]

    # Print detailed results
    print_results_by_category(category_accuracies)

    # Create plots directory in output_dir
    plots_dir = f"{args.output_dir}/plots"
    os.makedirs(plots_dir, exist_ok=True)

    # Generate visualizations
    generate_radar_plots(category_accuracies, plots_dir)
    generate_bar_plots(category_accuracies, plots_dir)

    print("\n✅ Evaluation analysis complete!")
    print(f"📊 Plots saved to: {plots_dir}")
    print(f"🎯 Final KiVA Score: {category_accuracies.get('kiva-overall', 0.0):.4f}")
