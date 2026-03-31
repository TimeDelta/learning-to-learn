#!/usr/bin/env python3
"""
Analyze pairwise inverted behavior among N MLflow metrics using three methods:

1. local_slope
   - Computes a local linear regression slope over a moving window
   - Detects intervals where one metric trends upward while the other trends downward

2. monotonic_segments
   - Compresses each metric into piecewise monotonic segments
   - Finds overlapping regions where one segment is increasing and the other decreasing

3. anti_correlation
   - Computes rolling Pearson correlation between pairs of metrics
   - Detects intervals where correlation is strongly negative

Example:
python analyze_inverted_behavior_mlflow.py \
    --tracking-uri http://localhost:5000 \
    --run-id FULL_RUN_ID \
    --metrics trainer_loss_fitness trainer_loss_wl_structural trainer_node_type_loss \
              trainer_total_loss trainer_loss_adjacency_recon trainer_loss_attribute_recon \
    --smoothing-window 5 \
    --slope-window 11 \
    --min-abs-slope 0.001 \
    --monotonic-tolerance 0.0005 \
    --correlation-window 21 \
    --negative-correlation-threshold -0.5 \
    --min-interval-length 5 \
    --output-prefix inverted_behavior
"""

import argparse
import itertools
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

# ============================================================
# Dataclasses
# ============================================================


@dataclass
class LocalSlopeInterval:
    metric_a: str
    metric_b: str
    direction: str
    start_step: int
    end_step: int
    interval_length_points: int
    mean_slope_a: float
    mean_slope_b: float
    mean_abs_slope_a: float
    mean_abs_slope_b: float
    method: str = "local_slope"


@dataclass
class MonotonicSegment:
    metric_name: str
    direction: str
    start_step: int
    end_step: int
    start_value: float
    end_value: float
    delta_value: float
    length_points: int


@dataclass
class MonotonicOverlapInterval:
    metric_a: str
    metric_b: str
    direction: str
    start_step: int
    end_step: int
    overlap_length_steps: int
    segment_a_start_step: int
    segment_a_end_step: int
    segment_b_start_step: int
    segment_b_end_step: int
    delta_a: float
    delta_b: float
    method: str = "monotonic_segments"


@dataclass
class AntiCorrelationInterval:
    metric_a: str
    metric_b: str
    start_step: int
    end_step: int
    interval_length_points: int
    mean_correlation: float
    min_correlation: float
    method: str = "anti_correlation"


# ============================================================
# MLflow metric loading
# ============================================================


def fetch_metric_history_for_run(
    run_id: str,
    metric_name: str,
    tracking_uri: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch full metric history for one MLflow metric in one run.

    Returns columns:
        step, value
    """
    mlflow_client = MlflowClient(tracking_uri=tracking_uri)
    metric_history = mlflow_client.get_metric_history(run_id, metric_name)

    if not metric_history:
        raise ValueError(
            f"No history found for metric '{metric_name}' in run '{run_id}'. "
            "Double-check that you are using the full MLflow run_id, not the shortened UI label."
        )

    metric_dataframe = pd.DataFrame(
        [{"step": metric_record.step, "value": metric_record.value} for metric_record in metric_history]
    )

    metric_dataframe = (
        metric_dataframe.sort_values("step").drop_duplicates(subset=["step"], keep="last").reset_index(drop=True)
    )

    return metric_dataframe


def build_aligned_metric_dataframe(
    run_id: str,
    metric_names: List[str],
    tracking_uri: Optional[str] = None,
) -> pd.DataFrame:
    """
    Align all requested metrics onto the union of all observed steps.

    Returns columns:
        step, <metric_1>, <metric_2>, ...
    """
    aligned_metric_dataframe: Optional[pd.DataFrame] = None

    for metric_name in metric_names:
        one_metric_dataframe = fetch_metric_history_for_run(
            run_id=run_id,
            metric_name=metric_name,
            tracking_uri=tracking_uri,
        ).rename(columns={"value": metric_name})

        if aligned_metric_dataframe is None:
            aligned_metric_dataframe = one_metric_dataframe
        else:
            aligned_metric_dataframe = aligned_metric_dataframe.merge(
                one_metric_dataframe,
                on="step",
                how="outer",
            )

    if aligned_metric_dataframe is None:
        raise ValueError("No metric data could be loaded.")

    aligned_metric_dataframe = aligned_metric_dataframe.sort_values("step").reset_index(drop=True)
    return aligned_metric_dataframe


# ============================================================
# Preprocessing
# ============================================================


def apply_optional_smoothing(
    aligned_metric_dataframe: pd.DataFrame,
    metric_names: List[str],
    smoothing_window: int,
) -> pd.DataFrame:
    """
    Apply centered moving average smoothing independently to each metric.
    """
    smoothed_metric_dataframe = aligned_metric_dataframe.copy()

    if smoothing_window <= 1:
        return smoothed_metric_dataframe

    for metric_name in metric_names:
        smoothed_metric_dataframe[metric_name] = (
            smoothed_metric_dataframe[metric_name].rolling(window=smoothing_window, center=True, min_periods=1).mean()
        )

    return smoothed_metric_dataframe


def interpolate_metrics_on_union_steps(
    metric_dataframe: pd.DataFrame,
    metric_names: List[str],
) -> pd.DataFrame:
    """
    Interpolate missing metric values on the union-of-steps grid.

    This makes pairwise windowed analyses more stable.
    """
    interpolated_dataframe = metric_dataframe.copy()

    for metric_name in metric_names:
        interpolated_dataframe[metric_name] = interpolated_dataframe[metric_name].interpolate(
            method="linear", limit_direction="both"
        )

    return interpolated_dataframe


# ============================================================
# Method 1: Local linear regression slope
# ============================================================


def compute_local_linear_regression_slope(
    step_series: pd.Series,
    value_series: pd.Series,
    window_size: int,
) -> np.ndarray:
    """
    Compute local slope by fitting a line inside a moving window.

    For each center index t, fit:
        y = beta_0 + beta_1 * step
    and return beta_1.
    """
    if window_size < 2:
        raise ValueError("window_size must be >= 2 for local regression slope.")

    step_values = step_series.to_numpy(dtype=float)
    metric_values = value_series.to_numpy(dtype=float)
    local_slope_values = np.full(len(metric_values), np.nan, dtype=float)

    half_window_size = window_size // 2

    for center_index in range(len(metric_values)):
        start_index = max(0, center_index - half_window_size)
        end_index = min(len(metric_values), center_index + half_window_size + 1)

        window_steps = step_values[start_index:end_index]
        window_values = metric_values[start_index:end_index]

        valid_mask = np.isfinite(window_steps) & np.isfinite(window_values)
        window_steps = window_steps[valid_mask]
        window_values = window_values[valid_mask]

        if len(window_values) < 2:
            continue

        centered_steps = window_steps - window_steps.mean()
        denominator = np.sum(centered_steps**2)

        if denominator <= 0:
            continue

        slope_estimate = np.sum(centered_steps * window_values) / denominator
        local_slope_values[center_index] = slope_estimate

    return local_slope_values


def compute_all_local_slopes(
    aligned_metric_dataframe: pd.DataFrame,
    metric_names: List[str],
    slope_window: int,
) -> pd.DataFrame:
    """
    Compute local linear regression slopes for each metric.
    """
    local_slope_dataframe = pd.DataFrame({"step": aligned_metric_dataframe["step"].values})

    for metric_name in metric_names:
        local_slope_dataframe[f"{metric_name}__local_slope"] = compute_local_linear_regression_slope(
            step_series=aligned_metric_dataframe["step"],
            value_series=aligned_metric_dataframe[metric_name],
            window_size=slope_window,
        )

    return local_slope_dataframe


def summarize_local_slope_interval(
    pair_slope_dataframe: pd.DataFrame,
    metric_a: str,
    metric_b: str,
    start_index: int,
    end_index: int,
    direction_label: str,
    min_interval_length: int,
) -> Optional[LocalSlopeInterval]:
    if end_index < start_index:
        return None

    interval_dataframe = pair_slope_dataframe.iloc[start_index : end_index + 1].copy()

    if len(interval_dataframe) < min_interval_length:
        return None

    slope_column_a = f"{metric_a}__local_slope"
    slope_column_b = f"{metric_b}__local_slope"

    if direction_label == "A_up_B_down":
        readable_direction = f"{metric_a}_up__{metric_b}_down"
    else:
        readable_direction = f"{metric_a}_down__{metric_b}_up"

    return LocalSlopeInterval(
        metric_a=metric_a,
        metric_b=metric_b,
        direction=readable_direction,
        start_step=int(interval_dataframe["step"].iloc[0]),
        end_step=int(interval_dataframe["step"].iloc[-1]),
        interval_length_points=int(len(interval_dataframe)),
        mean_slope_a=float(interval_dataframe[slope_column_a].mean()),
        mean_slope_b=float(interval_dataframe[slope_column_b].mean()),
        mean_abs_slope_a=float(interval_dataframe[slope_column_a].abs().mean()),
        mean_abs_slope_b=float(interval_dataframe[slope_column_b].abs().mean()),
    )


def detect_pairwise_local_slope_inversions(
    local_slope_dataframe: pd.DataFrame,
    metric_a: str,
    metric_b: str,
    min_abs_slope: float,
    min_interval_length: int,
) -> List[LocalSlopeInterval]:
    """
    Detect intervals where local slopes have opposite sign with sufficient magnitude.
    """
    slope_column_a = f"{metric_a}__local_slope"
    slope_column_b = f"{metric_b}__local_slope"

    pair_slope_dataframe = (
        local_slope_dataframe[["step", slope_column_a, slope_column_b]].dropna().reset_index(drop=True)
    )

    if pair_slope_dataframe.empty:
        return []

    slope_a = pair_slope_dataframe[slope_column_a]
    slope_b = pair_slope_dataframe[slope_column_b]

    a_up_b_down_mask = (slope_a > min_abs_slope) & (slope_b < -min_abs_slope)
    a_down_b_up_mask = (slope_a < -min_abs_slope) & (slope_b > min_abs_slope)

    direction_label_series = pd.Series(index=pair_slope_dataframe.index, dtype="object")
    direction_label_series[a_up_b_down_mask] = "A_up_B_down"
    direction_label_series[a_down_b_up_mask] = "A_down_B_up"

    detected_intervals: List[LocalSlopeInterval] = []
    current_direction_label: Optional[str] = None
    current_start_index: Optional[int] = None

    for row_index, direction_label in direction_label_series.items():
        if pd.isna(direction_label):
            if current_direction_label is not None and current_start_index is not None:
                interval_summary = summarize_local_slope_interval(
                    pair_slope_dataframe=pair_slope_dataframe,
                    metric_a=metric_a,
                    metric_b=metric_b,
                    start_index=current_start_index,
                    end_index=row_index - 1,
                    direction_label=current_direction_label,
                    min_interval_length=min_interval_length,
                )
                if interval_summary is not None:
                    detected_intervals.append(interval_summary)

                current_direction_label = None
                current_start_index = None
            continue

        if current_direction_label is None:
            current_direction_label = direction_label
            current_start_index = row_index
        elif direction_label != current_direction_label:
            interval_summary = summarize_local_slope_interval(
                pair_slope_dataframe=pair_slope_dataframe,
                metric_a=metric_a,
                metric_b=metric_b,
                start_index=current_start_index,
                end_index=row_index - 1,
                direction_label=current_direction_label,
                min_interval_length=min_interval_length,
            )
            if interval_summary is not None:
                detected_intervals.append(interval_summary)

            current_direction_label = direction_label
            current_start_index = row_index

    if current_direction_label is not None and current_start_index is not None:
        interval_summary = summarize_local_slope_interval(
            pair_slope_dataframe=pair_slope_dataframe,
            metric_a=metric_a,
            metric_b=metric_b,
            start_index=current_start_index,
            end_index=len(pair_slope_dataframe) - 1,
            direction_label=current_direction_label,
            min_interval_length=min_interval_length,
        )
        if interval_summary is not None:
            detected_intervals.append(interval_summary)

    return detected_intervals


# ============================================================
# Method 2: Piecewise monotonic segments
# ============================================================


def extract_monotonic_segments(
    step_series: pd.Series,
    value_series: pd.Series,
    metric_name: str,
    monotonic_tolerance: float,
    min_segment_length: int,
) -> List[MonotonicSegment]:
    """
    Extract piecewise monotonic segments using sign-consistent first differences.

    Rules:
    - positive delta > tolerance => increasing
    - negative delta < -tolerance => decreasing
    - otherwise => flat/ignored in direction transitions
    """
    clean_dataframe = (
        pd.DataFrame({"step": step_series.values, "value": value_series.values}).dropna().reset_index(drop=True)
    )

    if len(clean_dataframe) < 2:
        return []

    delta_values = clean_dataframe["value"].diff()

    direction_labels: List[str] = []
    for delta_value in delta_values:
        if pd.isna(delta_value):
            direction_labels.append("undefined")
        elif delta_value > monotonic_tolerance:
            direction_labels.append("increasing")
        elif delta_value < -monotonic_tolerance:
            direction_labels.append("decreasing")
        else:
            direction_labels.append("flat")

    clean_dataframe["direction"] = direction_labels

    extracted_segments: List[MonotonicSegment] = []

    current_direction: Optional[str] = None
    current_start_index: Optional[int] = None

    for row_index in range(1, len(clean_dataframe)):
        row_direction = clean_dataframe.loc[row_index, "direction"]

        if row_direction == "flat":
            continue

        if current_direction is None:
            current_direction = row_direction
            current_start_index = row_index - 1
            continue

        if row_direction != current_direction:
            segment_end_index = row_index - 1
            if current_start_index is not None:
                segment_dataframe = clean_dataframe.iloc[current_start_index : segment_end_index + 1]

                if len(segment_dataframe) >= min_segment_length:
                    extracted_segments.append(
                        MonotonicSegment(
                            metric_name=metric_name,
                            direction=current_direction,
                            start_step=int(segment_dataframe["step"].iloc[0]),
                            end_step=int(segment_dataframe["step"].iloc[-1]),
                            start_value=float(segment_dataframe["value"].iloc[0]),
                            end_value=float(segment_dataframe["value"].iloc[-1]),
                            delta_value=float(segment_dataframe["value"].iloc[-1] - segment_dataframe["value"].iloc[0]),
                            length_points=int(len(segment_dataframe)),
                        )
                    )

            current_direction = row_direction
            current_start_index = row_index - 1

    if current_direction is not None and current_start_index is not None:
        segment_dataframe = clean_dataframe.iloc[current_start_index:]

        if len(segment_dataframe) >= min_segment_length:
            extracted_segments.append(
                MonotonicSegment(
                    metric_name=metric_name,
                    direction=current_direction,
                    start_step=int(segment_dataframe["step"].iloc[0]),
                    end_step=int(segment_dataframe["step"].iloc[-1]),
                    start_value=float(segment_dataframe["value"].iloc[0]),
                    end_value=float(segment_dataframe["value"].iloc[-1]),
                    delta_value=float(segment_dataframe["value"].iloc[-1] - segment_dataframe["value"].iloc[0]),
                    length_points=int(len(segment_dataframe)),
                )
            )

    return extracted_segments


def detect_pairwise_monotonic_segment_overlaps(
    segments_a: List[MonotonicSegment],
    segments_b: List[MonotonicSegment],
    metric_a: str,
    metric_b: str,
    min_overlap_length: int,
) -> List[MonotonicOverlapInterval]:
    """
    Find overlapping intervals where one segment is increasing and the other decreasing.
    """
    overlap_intervals: List[MonotonicOverlapInterval] = []

    for segment_a in segments_a:
        for segment_b in segments_b:
            if segment_a.direction == segment_b.direction:
                continue

            overlap_start_step = max(segment_a.start_step, segment_b.start_step)
            overlap_end_step = min(segment_a.end_step, segment_b.end_step)

            if overlap_end_step <= overlap_start_step:
                continue

            overlap_length_steps = overlap_end_step - overlap_start_step
            if overlap_length_steps < min_overlap_length:
                continue

            direction_label = f"{metric_a}_{segment_a.direction}__{metric_b}_{segment_b.direction}"

            overlap_intervals.append(
                MonotonicOverlapInterval(
                    metric_a=metric_a,
                    metric_b=metric_b,
                    direction=direction_label,
                    start_step=int(overlap_start_step),
                    end_step=int(overlap_end_step),
                    overlap_length_steps=int(overlap_length_steps),
                    segment_a_start_step=int(segment_a.start_step),
                    segment_a_end_step=int(segment_a.end_step),
                    segment_b_start_step=int(segment_b.start_step),
                    segment_b_end_step=int(segment_b.end_step),
                    delta_a=float(segment_a.delta_value),
                    delta_b=float(segment_b.delta_value),
                )
            )

    overlap_intervals.sort(key=lambda interval: (interval.metric_a, interval.metric_b, interval.start_step))
    return overlap_intervals


# ============================================================
# Method 3: Windowed anti-correlation
# ============================================================


def compute_rolling_correlation(
    value_series_a: pd.Series,
    value_series_b: pd.Series,
    window_size: int,
) -> pd.Series:
    """
    Compute centered rolling Pearson correlation.
    """
    correlation_series = value_series_a.rolling(
        window=window_size,
        center=True,
        min_periods=max(3, window_size // 2),
    ).corr(value_series_b)

    return correlation_series


def summarize_anti_correlation_interval(
    pair_correlation_dataframe: pd.DataFrame,
    metric_a: str,
    metric_b: str,
    start_index: int,
    end_index: int,
    min_interval_length: int,
) -> Optional[AntiCorrelationInterval]:
    if end_index < start_index:
        return None

    interval_dataframe = pair_correlation_dataframe.iloc[start_index : end_index + 1].copy()

    if len(interval_dataframe) < min_interval_length:
        return None

    return AntiCorrelationInterval(
        metric_a=metric_a,
        metric_b=metric_b,
        start_step=int(interval_dataframe["step"].iloc[0]),
        end_step=int(interval_dataframe["step"].iloc[-1]),
        interval_length_points=int(len(interval_dataframe)),
        mean_correlation=float(interval_dataframe["rolling_correlation"].mean()),
        min_correlation=float(interval_dataframe["rolling_correlation"].min()),
    )


def detect_pairwise_anti_correlation_intervals(
    aligned_metric_dataframe: pd.DataFrame,
    metric_a: str,
    metric_b: str,
    correlation_window: int,
    negative_correlation_threshold: float,
    min_interval_length: int,
) -> Tuple[pd.DataFrame, List[AntiCorrelationInterval]]:
    """
    Detect contiguous intervals where rolling correlation is below threshold.
    """
    pair_correlation_dataframe = (
        pd.DataFrame(
            {
                "step": aligned_metric_dataframe["step"].values,
                metric_a: aligned_metric_dataframe[metric_a].values,
                metric_b: aligned_metric_dataframe[metric_b].values,
            }
        )
        .dropna()
        .reset_index(drop=True)
    )

    if pair_correlation_dataframe.empty:
        return pd.DataFrame(), []

    pair_correlation_dataframe["rolling_correlation"] = compute_rolling_correlation(
        value_series_a=pair_correlation_dataframe[metric_a],
        value_series_b=pair_correlation_dataframe[metric_b],
        window_size=correlation_window,
    )

    pair_correlation_dataframe = pair_correlation_dataframe.dropna().reset_index(drop=True)

    if pair_correlation_dataframe.empty:
        return pair_correlation_dataframe, []

    negative_correlation_mask = pair_correlation_dataframe["rolling_correlation"] < negative_correlation_threshold

    anti_correlation_intervals: List[AntiCorrelationInterval] = []
    current_start_index: Optional[int] = None

    for row_index, is_negative_correlation in negative_correlation_mask.items():
        if is_negative_correlation:
            if current_start_index is None:
                current_start_index = row_index
        else:
            if current_start_index is not None:
                interval_summary = summarize_anti_correlation_interval(
                    pair_correlation_dataframe=pair_correlation_dataframe,
                    metric_a=metric_a,
                    metric_b=metric_b,
                    start_index=current_start_index,
                    end_index=row_index - 1,
                    min_interval_length=min_interval_length,
                )
                if interval_summary is not None:
                    anti_correlation_intervals.append(interval_summary)
                current_start_index = None

    if current_start_index is not None:
        interval_summary = summarize_anti_correlation_interval(
            pair_correlation_dataframe=pair_correlation_dataframe,
            metric_a=metric_a,
            metric_b=metric_b,
            start_index=current_start_index,
            end_index=len(pair_correlation_dataframe) - 1,
            min_interval_length=min_interval_length,
        )
        if interval_summary is not None:
            anti_correlation_intervals.append(interval_summary)

    return pair_correlation_dataframe, anti_correlation_intervals


# ============================================================
# Master analysis
# ============================================================


def analyze_all_metric_pairs(
    run_id: str,
    metric_names: List[str],
    tracking_uri: Optional[str],
    smoothing_window: int,
    slope_window: int,
    min_abs_slope: float,
    monotonic_tolerance: float,
    monotonic_min_segment_length: int,
    correlation_window: int,
    negative_correlation_threshold: float,
    min_interval_length: int,
) -> Dict[str, pd.DataFrame]:
    """
    Run all three pairwise analyses.
    """
    aligned_metric_dataframe = build_aligned_metric_dataframe(
        run_id=run_id,
        metric_names=metric_names,
        tracking_uri=tracking_uri,
    )

    smoothed_metric_dataframe = apply_optional_smoothing(
        aligned_metric_dataframe=aligned_metric_dataframe,
        metric_names=metric_names,
        smoothing_window=smoothing_window,
    )

    interpolated_metric_dataframe = interpolate_metrics_on_union_steps(
        metric_dataframe=smoothed_metric_dataframe,
        metric_names=metric_names,
    )

    local_slope_dataframe = compute_all_local_slopes(
        aligned_metric_dataframe=interpolated_metric_dataframe,
        metric_names=metric_names,
        slope_window=slope_window,
    )

    all_local_slope_intervals: List[LocalSlopeInterval] = []
    all_monotonic_segments: List[MonotonicSegment] = []
    all_monotonic_overlap_intervals: List[MonotonicOverlapInterval] = []
    all_anti_correlation_intervals: List[AntiCorrelationInterval] = []
    all_pair_correlation_dataframes: List[pd.DataFrame] = []

    metric_name_to_segments: Dict[str, List[MonotonicSegment]] = {}
    for metric_name in metric_names:
        extracted_segments = extract_monotonic_segments(
            step_series=interpolated_metric_dataframe["step"],
            value_series=interpolated_metric_dataframe[metric_name],
            metric_name=metric_name,
            monotonic_tolerance=monotonic_tolerance,
            min_segment_length=monotonic_min_segment_length,
        )
        metric_name_to_segments[metric_name] = extracted_segments
        all_monotonic_segments.extend(extracted_segments)

    for metric_a, metric_b in itertools.combinations(metric_names, 2):
        local_slope_intervals = detect_pairwise_local_slope_inversions(
            local_slope_dataframe=local_slope_dataframe,
            metric_a=metric_a,
            metric_b=metric_b,
            min_abs_slope=min_abs_slope,
            min_interval_length=min_interval_length,
        )
        all_local_slope_intervals.extend(local_slope_intervals)

        monotonic_overlap_intervals = detect_pairwise_monotonic_segment_overlaps(
            segments_a=metric_name_to_segments[metric_a],
            segments_b=metric_name_to_segments[metric_b],
            metric_a=metric_a,
            metric_b=metric_b,
            min_overlap_length=min_interval_length,
        )
        all_monotonic_overlap_intervals.extend(monotonic_overlap_intervals)

        pair_correlation_dataframe, anti_correlation_intervals = detect_pairwise_anti_correlation_intervals(
            aligned_metric_dataframe=interpolated_metric_dataframe,
            metric_a=metric_a,
            metric_b=metric_b,
            correlation_window=correlation_window,
            negative_correlation_threshold=negative_correlation_threshold,
            min_interval_length=min_interval_length,
        )

        if not pair_correlation_dataframe.empty:
            pair_correlation_dataframe = pair_correlation_dataframe.copy()
            pair_correlation_dataframe["metric_a"] = metric_a
            pair_correlation_dataframe["metric_b"] = metric_b
            all_pair_correlation_dataframes.append(pair_correlation_dataframe)

        all_anti_correlation_intervals.extend(anti_correlation_intervals)

    return {
        "aligned_metrics": interpolated_metric_dataframe,
        "local_slopes": local_slope_dataframe,
        "local_slope_intervals": pd.DataFrame([asdict(interval) for interval in all_local_slope_intervals]),
        "monotonic_segments": pd.DataFrame([asdict(segment) for segment in all_monotonic_segments]),
        "monotonic_overlap_intervals": pd.DataFrame([asdict(interval) for interval in all_monotonic_overlap_intervals]),
        "pairwise_correlations": pd.concat(all_pair_correlation_dataframes, ignore_index=True)
        if all_pair_correlation_dataframes
        else pd.DataFrame(),
        "anti_correlation_intervals": pd.DataFrame([asdict(interval) for interval in all_anti_correlation_intervals]),
    }


# ============================================================
# CLI
# ============================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze pairwise inverted behavior among MLflow metrics.")
    parser.add_argument("--tracking-uri", type=str, default="http://127.0.0.1:5000")
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--metrics", type=str, nargs="+", required=True)

    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=7,
        help="Centered moving average smoothing window. Use 1 for no smoothing.",
    )
    parser.add_argument(
        "--slope-window",
        type=int,
        default=21,
        help="Window size for local linear regression slopes.",
    )
    parser.add_argument(
        "--min-abs-slope",
        type=float,
        default=0.01,
        help="Minimum absolute local slope magnitude for inversion detection.",
    )
    parser.add_argument(
        "--monotonic-tolerance",
        type=float,
        default=0.02,
        help="Tolerance for treating first differences as increasing/decreasing.",
    )
    parser.add_argument(
        "--monotonic-min-segment-length",
        type=int,
        default=5,
        help="Minimum point count for monotonic segments.",
    )
    parser.add_argument(
        "--correlation-window",
        type=int,
        default=30,
        help="Window size for rolling correlation.",
    )
    parser.add_argument(
        "--negative-correlation-threshold",
        type=float,
        default=-0.25,
        help="Threshold below which rolling correlation counts as anti-correlation.",
    )
    parser.add_argument(
        "--min-interval-length",
        type=int,
        default=8,
        help="Minimum length for detected intervals.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="inverted_behavior",
        help="Prefix for output CSV files.",
    )

    return parser.parse_args()


def save_output_dataframe(output_dataframe: pd.DataFrame, output_path: str) -> None:
    output_dataframe.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def print_summary_table(output_dataframe: pd.DataFrame, title: str, max_rows: int = 20) -> None:
    print(f"\n=== {title} ===")
    if output_dataframe.empty:
        print("No results.")
    else:
        print(output_dataframe.head(max_rows).to_string(index=False))


def main() -> None:
    args = parse_arguments()

    analysis_results = analyze_all_metric_pairs(
        run_id=args.run_id,
        metric_names=args.metrics,
        tracking_uri=args.tracking_uri,
        smoothing_window=args.smoothing_window,
        slope_window=args.slope_window,
        min_abs_slope=args.min_abs_slope,
        monotonic_tolerance=args.monotonic_tolerance,
        monotonic_min_segment_length=args.monotonic_min_segment_length,
        correlation_window=args.correlation_window,
        negative_correlation_threshold=args.negative_correlation_threshold,
        min_interval_length=args.min_interval_length,
    )

    output_prefix = args.output_prefix

    save_output_dataframe(analysis_results["aligned_metrics"], f"{output_prefix}_aligned_metrics.csv")
    save_output_dataframe(analysis_results["local_slopes"], f"{output_prefix}_local_slopes.csv")
    save_output_dataframe(analysis_results["local_slope_intervals"], f"{output_prefix}_local_slope_intervals.csv")
    save_output_dataframe(analysis_results["monotonic_segments"], f"{output_prefix}_monotonic_segments.csv")
    save_output_dataframe(
        analysis_results["monotonic_overlap_intervals"], f"{output_prefix}_monotonic_overlap_intervals.csv"
    )
    save_output_dataframe(analysis_results["pairwise_correlations"], f"{output_prefix}_pairwise_correlations.csv")
    save_output_dataframe(
        analysis_results["anti_correlation_intervals"], f"{output_prefix}_anti_correlation_intervals.csv"
    )

    print_summary_table(analysis_results["local_slope_intervals"], "Local slope inversion intervals")
    print_summary_table(analysis_results["monotonic_overlap_intervals"], "Monotonic overlap intervals")
    print_summary_table(analysis_results["anti_correlation_intervals"], "Anti-correlation intervals")


if __name__ == "__main__":
    main()
