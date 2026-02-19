import json

from pathlib import Path

from collections import defaultdict

from typing import Dict, Any, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from cockatoo_ml.logger.context import data_processing_logger as logger


def _parse_eval_metrics(eval_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    # parse eval metrics and group them by metric type
    grouped_metrics = defaultdict(dict)
    
    for key, value in eval_data.items():
        # skip non-metric keys
        if not isinstance(value, (int, float)):
            continue
        
        # remove eval prefix
        metric_key = key.replace('eval_', '')
        
        # skip runtime and performance metrics
        if metric_key in ('runtime', 'samples_per_second', 'steps_per_second', 'loss'):
            continue
        
        # parse metric type and label (ie 'f1' or 'f1_scam')
        parts = metric_key.split('_', 1)
        metric_type = parts[0]
        label = parts[1] if len(parts) > 1 else 'overall'
        
        grouped_metrics[metric_type][label] = value
    
    return dict(grouped_metrics)


def _create_metric_chart(metric_type: str, metric_data: Dict[str, float], output_path: Path, figsize: tuple = (12, 6)) -> None:
    # create bar chart for a specific metric type with labels on x axis and scores on y axis
    
    # sort labels with 'overall' first if it exists, then alphabetically
    labels = sorted(metric_data.keys(), key=lambda x: (x != 'overall', x))
    values = [metric_data[label] for label in labels]
    
    # create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # create bar chart with color gradient
    colors = plt.cm.viridis([(v - min(values)) / (max(values) - min(values)) if max(values) > min(values) else 0.5 for v in values])
    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=1.2)
    
    # add value labels on top of bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # customize chart
    ax.set_ylabel(metric_type.upper(), fontsize=12, fontweight='bold')
    ax.set_xlabel('Labels', fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_type.upper()} Scores by Label', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # rotate x labels if there are many labels
    if len(labels) > 5:
        plt.xticks(rotation=45, ha='right')
    
    # tight layout to prevent label cutoff
    plt.tight_layout()
    
    # save figure
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_eval_metrics(eval_data: Dict[str, Any], experiment_name: str, model_name: str, epoch_num: Optional[int] = None, output_dir: str = 'graph') -> Path:
    # generate and save matplotlib plots for eval metrics
    # creates individual charts for each metric type (f1, precision, recall, etc.)
    # plots organized as: {output_dir}/{experiment_name}-epoch{epoch_num}/
    # filenames: {metric_type}_{model_name}_epoch{epoch_num}.png
    
    # create output directory with proper structure
    if epoch_num is not None:
        exp_dir = f"{experiment_name}-epoch{epoch_num}"
    else:
        exp_dir = experiment_name
    
    output_path = Path(output_dir) / exp_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # parse and group metrics
    grouped_metrics = _parse_eval_metrics(eval_data)
    
    if not grouped_metrics:
        logger.warning(f"No metrics found to plot in eval_data")
        return output_path
    
    # create individual charts for each metric type
    for metric_type, metric_data in grouped_metrics.items():
        if epoch_num is not None:
            filename = f"{metric_type}_{model_name}_epoch{epoch_num}.png"
        else:
            filename = f"{metric_type}_{model_name}.png"
        
        chart_path = output_path / filename
        _create_metric_chart(metric_type, metric_data, chart_path)
        logger.info(f"✓ Saved plot: {chart_path}")
    
    logger.info(f"All evaluation plots saved to: {output_path.absolute()}")
    
    return output_path


def load_and_plot_eval_data(json_file_path: str, experiment_name: str, model_name: str, epoch_num: Optional[int] = None, output_dir: str = 'graph') -> Path:
    # load json file and plot data
    
    json_path = Path(json_file_path)
    
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")
    
    with open(json_path, 'r') as f:
        eval_data = json.load(f)
    
    return plot_eval_metrics(eval_data, experiment_name, model_name, epoch_num, output_dir)
