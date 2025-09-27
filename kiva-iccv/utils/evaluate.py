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

import matplotlib.pyplot as plt
import numpy as np
from config import Config
from utils.helper import (
    LEVEL_KIVA_DB_KEY,
    LEVEL_KIVA_FUNCTIONS_COMPOSITIONALITY_DB_KEY,
    LEVEL_KIVA_FUNCTIONS_DB_KEY,
    TRANSFORMATIONS_FOR_COMPOSITE_GROUP,
    TRANSFORMATIONS_FOR_SIMPLE_GROUP,
    plot_tags,
    radar_factory,
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
) -> tuple[dict[str, float], dict[str, int]]:
    """
    Calculate accuracies based on LEVEL_CATEGORIES and TRANSFORMATION_CATEGORIES.

    Args:
        answers_list: List of dictionaries with 'id' and 'answer'
        db_dict: Dictionary containing the database (e.g., val_trials)

    Returns:
        Tuple of (accuracies dictionary, sample counts dictionary)
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

    # Calculate final accuracies and sample counts
    final_accuracies = {}
    sample_counts = {}
    for key, _ in results_counts.items():
        if key[1] == "total":  # Only process "total" keys to avoid duplicates
            category = key[0]
            correct = results_counts.get((category, "correct"), 0)
            total = results_counts.get((category, "total"), 0)
            final_accuracies[category] = correct / total if total > 0 else 0.0
            sample_counts[category] = total

    return dict(sorted(final_accuracies.items())), dict(sorted(sample_counts.items()))


def evaluate_submission(
    submission_data: list[dict[str, str]], val_trials: dict[str, Any]
) -> dict[str, Any]:
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

    # Calculate category-specific accuracies and sample counts
    cat_accuracies, sample_counts = calc_kiva_cat_accuracies(submission_data, val_trials)

    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Overall KiVA Score: {cat_accuracies.get('kiva-overall', 0.0):.4f}")

    return {
        "overall_accuracy": overall_accuracy,
        "cat_accuracies": cat_accuracies,
        "sample_counts": sample_counts,
    }


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


def create_labels_with_percentages(
    transformation_list: list[str], level_key: str, sample_counts: dict[str, int]
) -> list[str]:
    """
    Create labels with transformation names and sample percentages.

    Args:
        transformation_list: List of transformation names
        level_key: Level key (e.g., 'kiva', 'kiva-functions')
        sample_counts: Dictionary with sample counts for each category

    Returns:
        List of formatted labels with percentages
    """
    # Get total samples across ALL levels (kiva-overall represents all samples)
    total_samples = sample_counts.get("kiva-overall", 0)

    labels_with_percentages = []
    for trans in transformation_list:
        trans_key = f"{level_key}_{trans}"
        trans_samples = sample_counts.get(trans_key, 0)
        percentage = (trans_samples / total_samples * 100) if total_samples > 0 else 0.0
        label_with_percentage = f"{trans}\n({percentage:.1f}%)"
        labels_with_percentages.append(label_with_percentage)

    return labels_with_percentages


def generate_combined_radar_plot(
    cat_accuracies: dict[str, float], plots_dir: str, sample_counts: dict[str, int]
) -> None:
    """
    Generate combined radar plot with three subplots in one row.

    Args:
        cat_accuracies: Dictionary with category-specific accuracy scores
        plots_dir: Directory to save plots
        sample_counts: Dictionary with sample counts for each category
    """
    print("\n--- Generating Combined Radar Plot ---")

    if not cat_accuracies:
        print("No results available for plotting.")
        return

    os.makedirs(plots_dir, exist_ok=True)

    # Create figure with proper size
    fig = plt.figure(figsize=(24, 8))
    fig.subplots_adjust(wspace=0.4, hspace=0.3, top=0.85, bottom=0.05)

    # Data for each subplot with percentage labels
    subplot_data = [
        {
            "title": "KiVA",
            "labels": create_labels_with_percentages(
                TRANSFORMATIONS_FOR_SIMPLE_GROUP, LEVEL_KIVA_DB_KEY, sample_counts
            ),
            "scores": {
                "Submission": [
                    cat_accuracies.get(f"{LEVEL_KIVA_DB_KEY}_{trans}", 0.0)
                    for trans in TRANSFORMATIONS_FOR_SIMPLE_GROUP
                ],
                "Random": [0.33] * len(TRANSFORMATIONS_FOR_SIMPLE_GROUP),
            },
        },
        {
            "title": "KiVA-functions",
            "labels": create_labels_with_percentages(
                TRANSFORMATIONS_FOR_SIMPLE_GROUP, LEVEL_KIVA_FUNCTIONS_DB_KEY, sample_counts
            ),
            "scores": {
                "Submission": [
                    cat_accuracies.get(f"{LEVEL_KIVA_FUNCTIONS_DB_KEY}_{trans}", 0.0)
                    for trans in TRANSFORMATIONS_FOR_SIMPLE_GROUP
                ],
                "Random": [0.33] * len(TRANSFORMATIONS_FOR_SIMPLE_GROUP),
            },
        },
        {
            "title": "KiVA-functions-compositionality",
            "labels": create_labels_with_percentages(
                TRANSFORMATIONS_FOR_COMPOSITE_GROUP,
                LEVEL_KIVA_FUNCTIONS_COMPOSITIONALITY_DB_KEY,
                sample_counts,
            ),
            "scores": {
                "Submission": [
                    cat_accuracies.get(
                        f"{LEVEL_KIVA_FUNCTIONS_COMPOSITIONALITY_DB_KEY}_{trans}", 0.0
                    )
                    for trans in TRANSFORMATIONS_FOR_COMPOSITE_GROUP
                ],
                "Random": [0.33] * len(TRANSFORMATIONS_FOR_COMPOSITE_GROUP),
            },
        },
    ]

    # Create each subplot
    for i, data in enumerate(subplot_data):
        # Register radar projection for this number of variables
        theta = radar_factory(len(data["labels"]), frame="polygon")

        # Create subplot with radar projection
        ax = fig.add_subplot(1, 3, i + 1, projection="radar")

        # Configure axis
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylim(0, 1)

        # Plot data
        for method, score in data["scores"].items():
            if method == "Random":
                ax.plot(theta, score, color="red", linestyle="dashed", label=method)
            else:
                ax.plot(theta, score, color="blue", label=method)
                ax.fill(theta, score, facecolor="blue", alpha=0.15, label="_nolegend_")

        # Set labels and title
        ax.set_title(
            data["title"],
            size=14,
            position=(0.5, 1.1),
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.set_varlabels(data["labels"])

        # Add legend only to the first subplot
        if i == 0:
            ax.legend(prop={"size": 12}, loc="upper right", bbox_to_anchor=(1.3, 1.0))

    # Save combined plot
    combined_radar_path = os.path.join(plots_dir, "combined_radar.svg")
    plt.savefig(combined_radar_path, bbox_inches="tight")
    plt.show()

    print(f"Combined radar plot saved to {combined_radar_path}")


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


def generate_combined_bar_plot(
    cat_accuracies: dict[str, float], plots_dir: str, sample_counts: dict[str, int]
) -> None:
    """
    Generate combined bar plot with three subplots in one row.

    Args:
        cat_accuracies: Dictionary with category-specific accuracy scores
        plots_dir: Directory to save plots
        sample_counts: Dictionary with sample counts for each category
    """
    print("\n--- Generating Combined Bar Plot ---")

    if not cat_accuracies:
        print("No results available for plotting.")
        return

    os.makedirs(plots_dir, exist_ok=True)

    # Create figure with 3 subplots in one row
    fig, axes = plt.subplots(figsize=(24, 6), nrows=1, ncols=3)
    fig.subplots_adjust(wspace=0.3, hspace=0.3, top=0.85, bottom=0.2)

    # Data for each subplot with percentage labels
    subplot_data = [
        {
            "title": "KiVA",
            "labels": create_labels_with_percentages(
                TRANSFORMATIONS_FOR_SIMPLE_GROUP, LEVEL_KIVA_DB_KEY, sample_counts
            ),
            "values": [
                cat_accuracies.get(f"{LEVEL_KIVA_DB_KEY}_{trans}", 0.0)
                for trans in TRANSFORMATIONS_FOR_SIMPLE_GROUP
            ],
        },
        {
            "title": "KiVA-functions",
            "labels": create_labels_with_percentages(
                TRANSFORMATIONS_FOR_SIMPLE_GROUP, LEVEL_KIVA_FUNCTIONS_DB_KEY, sample_counts
            ),
            "values": [
                cat_accuracies.get(f"{LEVEL_KIVA_FUNCTIONS_DB_KEY}_{trans}", 0.0)
                for trans in TRANSFORMATIONS_FOR_SIMPLE_GROUP
            ],
        },
        {
            "title": "KiVA-functions-compositionality",
            "labels": create_labels_with_percentages(
                TRANSFORMATIONS_FOR_COMPOSITE_GROUP,
                LEVEL_KIVA_FUNCTIONS_COMPOSITIONALITY_DB_KEY,
                sample_counts,
            ),
            "values": [
                cat_accuracies.get(f"{LEVEL_KIVA_FUNCTIONS_COMPOSITIONALITY_DB_KEY}_{trans}", 0.0)
                for trans in TRANSFORMATIONS_FOR_COMPOSITE_GROUP
            ],
        },
    ]

    # Create each subplot
    for i, (ax, data) in enumerate(zip(axes, subplot_data, strict=False)):
        # Create bars
        x_positions = np.arange(len(data["labels"]))
        ax.bar(x_positions, data["values"], color="#6495ED", width=0.8)

        # Configure axis
        ax.set_xticks(x_positions)
        ax.set_xticklabels(data["labels"], rotation=45, ha="right", fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.set_ylabel("Accuracy")
        ax.set_title(data["title"], fontsize=14)

        # Add random baseline
        ax.axhline(y=0.33, color="black", linestyle="dashed", alpha=0.7)

        # Add legend only to the first subplot
        if i == 0:
            handles = [
                plt.Rectangle((0, 0), 1, 1, color="#6495ED"),
                plt.Line2D([0], [0], color="black", linestyle="dashed"),
            ]
            ax.legend(
                handles,
                ["8-shot Frequency", "Random Level (33%)"],
                loc="upper left",
                bbox_to_anchor=(0.01, 0.99),
                fontsize=10,
            )

        # Adjust margins
        ax.margins(x=0.05)

    # Save combined plot
    combined_bar_path = os.path.join(plots_dir, "combined_bar.svg")
    plt.savefig(combined_bar_path, bbox_inches="tight")
    plt.show()

    print(f"Combined bar plot saved to {combined_bar_path}")


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
    sample_counts = results["sample_counts"]
    generate_combined_radar_plot(cat_accuracies, plots_dir, sample_counts)
    generate_combined_bar_plot(cat_accuracies, plots_dir, sample_counts)

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
    config = Config.from_args(args, for_task="test")

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
    sample_counts = results["sample_counts"]
    generate_combined_radar_plot(category_accuracies, plots_dir, sample_counts)
    generate_combined_bar_plot(category_accuracies, plots_dir, sample_counts)

    print("\n✅ Evaluation analysis complete!")
    print(f"📊 Plots saved to: {plots_dir}")
    print(f"🎯 Final KiVA Score: {category_accuracies.get('kiva-overall', 0.0):.4f}")
