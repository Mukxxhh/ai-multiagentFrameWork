"""
FastAPI backend for ClarityLens Analytics.
Supports file uploads, direct text analysis, follow-up Q&A, and PDF reporting.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config.settings import settings
from utils.file_parser import FileParser
from utils.report_generator import ReportGenerator
from utils.visualizer import VisualizationGenerator

try:
    from agents.insights_agents import answer_data_question, run_analysis_pipeline

    AI_ANALYSIS_AVAILABLE = True
except ImportError:
    answer_data_question = None
    run_analysis_pipeline = None
    AI_ANALYSIS_AVAILABLE = False


app = FastAPI(
    title="ClarityLens Analytics API",
    description="Adaptive agentic system for numerical and non-numerical data analysis",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

file_parser = FileParser()
visualizer = VisualizationGenerator(output_dir=settings.output_dir)
report_generator = ReportGenerator(template_dir="./templates", output_dir=settings.output_dir)
jobs: Dict[str, Dict[str, Any]] = {}
analysis_cache: Dict[str, str] = {}
ANALYSIS_CACHE_VERSION = "metrics_v3"

app.mount("/output", StaticFiles(directory=settings.output_dir), name="output")


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int
    message: str
    created_at: str
    updated_at: str
    result: Optional[Dict[str, Any]] = None
    pdf_path: Optional[str] = None
    processing: Optional[Dict[str, Any]] = None


class TextAnalysisRequest(BaseModel):
    content: str = Field(..., min_length=1)
    title: Optional[str] = "direct_input.txt"
    goal: Optional[str] = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)


PIPELINE_STAGES = [
    {
        "key": "input_received",
        "label": "Input Received",
        "icon": "fa-inbox",
        "description": "We received your data and are checking its format.",
    },
    {
        "key": "format_detection",
        "label": "Format Detection",
        "icon": "fa-file-waveform",
        "description": "We are identifying whether your input is numerical, mixed, or unstructured.",
    },
    {
        "key": "structure_mapping",
        "label": "Structure Mapping",
        "icon": "fa-diagram-project",
        "description": "We are converting important parts into a structured analysis view.",
    },
    {
        "key": "metric_discovery",
        "label": "Metric Discovery",
        "icon": "fa-magnifying-glass-chart",
        "description": "We are selecting the best metrics and comparisons to explain visually.",
    },
    {
        "key": "visualization_planning",
        "label": "Visualization Planning",
        "icon": "fa-chart-line",
        "description": "We are generating charts designed for non-technical understanding.",
    },
    {
        "key": "insight_reasoning",
        "label": "Insight Reasoning",
        "icon": "fa-brain",
        "description": "We are reasoning over the data to produce key findings and recommendations.",
    },
    {
        "key": "response_writing",
        "label": "Response Writing",
        "icon": "fa-pen-nib",
        "description": "We are writing a clear response that translates analysis into business language.",
    },
    {
        "key": "report_packaging",
        "label": "Report Packaging",
        "icon": "fa-file-pdf",
        "description": "We are packaging your final response and report.",
    },
]


def allowed_file(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in settings.allowed_extensions_list


def ollama_running() -> bool:
    try:
        host = settings.ollama_base_url.replace("http://", "").replace("https://", "").split("/")[0]
        hostname, port = host.split(":")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((hostname, int(port)))
        sock.close()
        return result == 0
    except Exception:
        return False


def compute_fingerprint(content: bytes, source_name: str, goal: Optional[str] = None) -> str:
    digest = hashlib.sha256()
    digest.update(ANALYSIS_CACHE_VERSION.encode("utf-8", errors="ignore"))
    digest.update(source_name.encode("utf-8", errors="ignore"))
    if goal:
        digest.update(goal.encode("utf-8", errors="ignore"))
    digest.update(content)
    return digest.hexdigest()


def serialize_parsed_data(parsed_data: Dict[str, Any]) -> Dict[str, Any]:
    serializable = dict(parsed_data)
    data = parsed_data.get("data")
    if isinstance(data, pd.DataFrame):
        serializable["data_records"] = data.head(1000).to_dict(orient="records")
        serializable["data_columns"] = list(data.columns)
        serializable["data"] = None
    return serializable


def restore_parsed_data(serialized: Dict[str, Any]) -> Dict[str, Any]:
    restored = dict(serialized or {})
    records = restored.pop("data_records", None)
    if records is not None:
        restored["data"] = pd.DataFrame(records)
    else:
        restored.setdefault("data", None)
    return restored


def build_chart_urls(visualizations: Dict[str, Any]) -> Dict[str, Any]:
    clean_visualizations = {
        "charts": [],
        "has_charts": visualizations.get("has_charts", False),
        "chart_count": visualizations.get("chart_count", 0),
    }
    for chart in visualizations.get("charts", []):
        image_path = chart.get("image_path", "")
        filename = os.path.basename(image_path) if image_path else ""
        clean_visualizations["charts"].append(
            {
                "type": chart.get("type", "unknown"),
                "title": chart.get("title", "Chart"),
                "caption": chart.get("caption", ""),
                "figure_json": chart.get("figure_json"),
                "image_path": image_path,
                "image_url": f"/output/{filename}" if filename else "",
                "chart_id": chart.get("chart_id", 0),
            }
        )
    return clean_visualizations


def build_preview_cards(parsed_data: Dict[str, Any], statistics: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    metadata = parsed_data.get("metadata", {})
    stats = statistics or {}
    previews = [
        {"label": "Detected input", "value": str(parsed_data.get("file_type", "unknown")).replace("_", " ")},
        {
            "label": "Likely analysis mode",
            "value": "numerical" if parsed_data.get("has_numerical_data") else "qualitative",
        },
    ]
    if metadata.get("rows"):
        previews.append({"label": "Rows", "value": str(metadata.get("rows"))})
    if metadata.get("columns"):
        previews.append({"label": "Columns", "value": str(len(metadata.get("columns", [])))})
    metrics = stats.get("analysis_ready_metrics", [])
    if metrics:
        previews.append({"label": "Likely focus metrics", "value": ", ".join(metrics[:3])})
    text_preview = (parsed_data.get("text_content") or "").strip()
    if text_preview and not parsed_data.get("has_numerical_data"):
        previews.append({"label": "Structured extraction preview", "value": text_preview[:120] + ("..." if len(text_preview) > 120 else "")})
    return previews[:5]


def build_project_metrics() -> List[Dict[str, str]]:
    return [
        {
            "name": "Accuracy",
            "formula": "(TP + TN) / (TP + TN + FP + FN)",
            "input_data": "All predictions with true labels, counting true positives, true negatives, false positives, and false negatives.",
            "how_it_works": "Measures the overall fraction of correct predictions across every class decision.",
            "quality_signal": "Useful when classes are balanced. High accuracy alone is not enough if one class is much more common than the other.",
        },
        {
            "name": "Precision",
            "formula": "TP / (TP + FP)",
            "input_data": "Only predicted-positive cases, comparing correct positive predictions against false alarms.",
            "how_it_works": "Tells us how trustworthy a positive prediction is.",
            "quality_signal": "A high value means the model makes fewer false positive mistakes. Important when false alarms are costly.",
        },
        {
            "name": "Recall",
            "formula": "TP / (TP + FN)",
            "input_data": "All actual-positive cases, comparing captured positives against missed positives.",
            "how_it_works": "Shows how much of the real positive class the model successfully finds.",
            "quality_signal": "A high value means the model misses fewer important cases. Important when missing positives is risky.",
        },
        {
            "name": "F1-Score",
            "formula": "2 x (Precision x Recall) / (Precision + Recall)",
            "input_data": "Derived from precision and recall after computing TP, FP, and FN.",
            "how_it_works": "Combines precision and recall into one balanced score.",
            "quality_signal": "A strong F1-score means the model balances false alarms and missed detections well.",
        },
        {
            "name": "ROC-AUC",
            "formula": "Area under the ROC curve across thresholds",
            "input_data": "Predicted probabilities or confidence scores plus true binary labels.",
            "how_it_works": "Evaluates how well the model separates positive and negative cases over many decision thresholds.",
            "quality_signal": "Closer to 1.0 is stronger separation, around 0.5 is near-random, and below 0.5 suggests the ranking is poor.",
        },
    ]


def _normalize_binary_target(series: pd.Series) -> Optional[pd.Series]:
    clean = series.dropna()
    if clean.empty:
        return None

    unique_values = pd.unique(clean)
    if len(unique_values) != 2:
        return None

    normalized = clean.astype(str).str.strip().str.lower()
    mapping_sets = [
        ({"0", "1"}, {"0": 0, "1": 1}),
        ({"false", "true"}, {"false": 0, "true": 1}),
        ({"no", "yes"}, {"no": 0, "yes": 1}),
        ({"n", "y"}, {"n": 0, "y": 1}),
    ]
    normalized_unique = set(normalized.unique())
    for allowed, mapping in mapping_sets:
        if normalized_unique.issubset(allowed):
            encoded = normalized.map(mapping)
            return encoded.reindex(series.index)

    try:
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_clean = numeric.dropna()
        if set(pd.unique(numeric_clean)).issubset({0, 1}):
            return numeric.astype(float)
    except Exception:
        pass

    # Stable mapping for arbitrary two-class labels.
    ordered = sorted(str(value) for value in unique_values)
    mapping = {ordered[0]: 0, ordered[1]: 1}
    encoded = series.astype(str).str.strip().map(mapping)
    return encoded.reindex(series.index)


def _infer_binary_target_column(data: pd.DataFrame) -> Optional[str]:
    preferred_tokens = ["target", "label", "class", "outcome", "returned", "return", "churn", "fraud", "default"]

    candidates = []
    for column in data.columns:
        encoded = _normalize_binary_target(data[column])
        if encoded is None:
            continue
        score = 1.0
        lower = str(column).lower()
        if any(token in lower for token in preferred_tokens):
            score += 3
        non_null_ratio = encoded.notna().mean()
        score += non_null_ratio
        candidates.append((score, column))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


def _prepare_model_features(data: pd.DataFrame, target_column: str) -> pd.DataFrame:
    features = data.drop(columns=[target_column]).copy()

    # Drop near-identifier columns to avoid unstable fits.
    drop_columns = []
    for column in features.columns:
        series = features[column]
        non_null = series.dropna()
        if non_null.empty:
            drop_columns.append(column)
            continue
        unique_ratio = non_null.nunique() / max(len(non_null), 1)
        if unique_ratio > 0.98 and series.dtype == object:
            drop_columns.append(column)
    if drop_columns:
        features = features.drop(columns=drop_columns, errors="ignore")

    for column in list(features.columns):
        if pd.api.types.is_datetime64_any_dtype(features[column]):
            features[column] = features[column].astype("int64") / 10**9
        elif features[column].dtype == object:
            parsed = pd.to_datetime(features[column], errors="coerce")
            if parsed.notna().mean() >= 0.8:
                features[column] = parsed.astype("int64") / 10**9

    categorical_cols = [col for col in features.columns if features[col].dtype == object]
    features = pd.get_dummies(features, columns=categorical_cols, dummy_na=True, drop_first=True)
    features = features.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Remove constant columns after encoding.
    constant_cols = [col for col in features.columns if features[col].nunique(dropna=False) <= 1]
    if constant_cols:
        features = features.drop(columns=constant_cols, errors="ignore")

    # Standardize numeric columns to reduce fitting instability.
    for column in features.columns:
        series = features[column].astype(float)
        std = float(series.std())
        mean = float(series.mean())
        if std > 0:
            features[column] = (series - mean) / std
        else:
            features[column] = 0.0

    return features


def _roc_auc_score_manual(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    positive_mask = y_true == 1
    negative_mask = y_true == 0
    n_pos = int(positive_mask.sum())
    n_neg = int(negative_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    positive_ranks = ranks[positive_mask]
    auc = (positive_ranks.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def compute_project_metric_results_from_dataset(parsed_data: Dict[str, Any]) -> List[Dict[str, str]]:
    data = parsed_data.get("data")
    if not isinstance(data, pd.DataFrame) or data.empty or len(data) < 30:
        return []

    target_column = _infer_binary_target_column(data)
    if not target_column:
        return []

    target = _normalize_binary_target(data[target_column])
    if target is None:
        return []

    features = _prepare_model_features(data, target_column)
    if features.empty:
        return []

    dataset = features.copy()
    dataset["_target"] = target
    dataset = dataset.dropna(subset=["_target"])
    if len(dataset) < 30:
        return []

    y = dataset["_target"].astype(int)
    X = dataset.drop(columns=["_target"])
    if X.empty:
        return []

    rng = np.random.default_rng(42)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    split_index = max(int(len(indices) * 0.8), 1)
    train_idx = indices[:split_index]
    test_idx = indices[split_index:]
    if len(test_idx) == 0:
        return []

    X_train = sm.add_constant(X.iloc[train_idx], has_constant="add")
    X_test = sm.add_constant(X.iloc[test_idx], has_constant="add")
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        return []

    try:
        model = sm.GLM(y_train, X_train, family=sm.families.Binomial()).fit()
        probabilities = np.clip(model.predict(X_test).astype(float), 0.0, 1.0)
    except Exception:
        return []

    predictions = (probabilities >= 0.5).astype(int)
    y_true = y_test.to_numpy(dtype=int)
    tp = int(((predictions == 1) & (y_true == 1)).sum())
    tn = int(((predictions == 0) & (y_true == 0)).sum())
    fp = int(((predictions == 1) & (y_true == 0)).sum())
    fn = int(((predictions == 0) & (y_true == 1)).sum())

    total = tp + tn + fp + fn
    if total == 0:
        return []

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    roc_auc = _roc_auc_score_manual(y_true, probabilities.to_numpy(dtype=float))

    metric_values = [
        ("Accuracy", accuracy),
        ("Precision", precision),
        ("Recall", recall),
        ("F1-Score", f1_score),
        ("ROC-AUC", roc_auc),
    ]

    results = []
    for metric_name, score_0_to_1 in metric_values:
        if score_0_to_1 is None:
            continue
        if score_0_to_1 >= 0.95:
            verdict = f"Excellent result from a baseline model using '{target_column}' as the target."
        elif score_0_to_1 >= 0.90:
            verdict = f"Strong result from a baseline model using '{target_column}' as the target."
        elif score_0_to_1 >= 0.80:
            verdict = f"Good result from a baseline model using '{target_column}' as the target."
        elif score_0_to_1 >= 0.70:
            verdict = f"Moderate result from a baseline model using '{target_column}' as the target."
        else:
            verdict = f"Weak baseline result for target '{target_column}'. Feature engineering or a different model may help."

        results.append(
            {
                "name": metric_name,
                "raw_value": f"{score_0_to_1:.4f}",
                "display_value": f"{score_0_to_1 * 100:.2f}%" if metric_name != "ROC-AUC" else f"{score_0_to_1:.4f}",
                "verdict": verdict,
                "source": "computed_from_dataset",
                "target_column": target_column,
            }
        )

    return results


def extract_project_metric_results(parsed_data: Dict[str, Any]) -> List[Dict[str, str]]:
    text_sources = [
        parsed_data.get("text_content", ""),
        json.dumps(parsed_data.get("data_records", []), ensure_ascii=True) if parsed_data.get("data_records") else "",
    ]
    source_text = "\n".join(part for part in text_sources if part)

    metric_patterns = [
        ("Accuracy", [r"accuracy"]),
        ("Precision", [r"precision"]),
        ("Recall", [r"recall", r"sensitivity", r"true positive rate"]),
        ("F1-Score", [r"f1[\s-]?score", r"\bf1\b"]),
        ("ROC-AUC", [r"roc[\s-]?auc", r"auc[\s-]?roc", r"\bauc\b"]),
    ]

    results: List[Dict[str, str]] = []
    for metric_name, aliases in metric_patterns:
        raw_value = None
        for alias in aliases:
            pattern = re.compile(
                rf"(?is)\b{alias}\b[^0-9%]{{0,25}}(?P<value>\d+(?:\.\d+)?)\s*(?P<percent>%?)"
            )
            match = pattern.search(source_text)
            if match:
                value = match.group("value")
                percent = match.group("percent")
                raw_value = f"{value}{percent}" if percent else value
                break

        if raw_value is None:
            continue

        numeric_match = re.search(r"\d+(?:\.\d+)?", raw_value)
        numeric_value = float(numeric_match.group(0)) if numeric_match else None
        is_percent = "%" in raw_value
        score_0_to_1 = None
        if numeric_value is not None:
            score_0_to_1 = numeric_value / 100 if is_percent else numeric_value
            if metric_name != "ROC-AUC" and score_0_to_1 is not None and score_0_to_1 <= 1:
                normalized_value = f"{score_0_to_1 * 100:.2f}%"
            elif is_percent:
                normalized_value = f"{numeric_value:.2f}%"
            else:
                normalized_value = f"{numeric_value:.4f}" if metric_name == "ROC-AUC" else f"{numeric_value:.2f}"
        else:
            normalized_value = raw_value

        if score_0_to_1 is None:
            verdict = "Metric was found, but it could not be interpreted numerically."
        elif score_0_to_1 >= 0.95:
            verdict = "Excellent result."
        elif score_0_to_1 >= 0.90:
            verdict = "Strong result."
        elif score_0_to_1 >= 0.80:
            verdict = "Good result, but there is still room to improve."
        elif score_0_to_1 >= 0.70:
            verdict = "Moderate result. Review model errors before presenting it as robust."
        elif metric_name == "ROC-AUC" and abs(score_0_to_1 - 0.5) < 0.05:
            verdict = "Near-random class separation."
        else:
            verdict = "Weak result. The model likely needs tuning, better data, or threshold changes."

        results.append(
            {
                "name": metric_name,
                "raw_value": raw_value,
                "display_value": normalized_value,
                "verdict": verdict,
            }
        )

    if results:
        return results
    return compute_project_metric_results_from_dataset(parsed_data)


def set_stage(
    job_id: str,
    stage_key: str,
    progress: int,
    message: Optional[str] = None,
    live_note: Optional[str] = None,
    preview_cards: Optional[List[Dict[str, str]]] = None,
) -> None:
    stages = []
    active_found = False
    stage_description = ""
    for stage in PIPELINE_STAGES:
        status = "waiting"
        if stage["key"] == stage_key:
            status = "active"
            active_found = True
            stage_description = stage["description"]
        elif not active_found:
            status = "done"
        stages.append({**stage, "status": status})

    jobs[job_id]["progress"] = progress
    jobs[job_id]["message"] = message or stage_description
    jobs[job_id]["updated_at"] = datetime.now().isoformat()
    jobs[job_id]["processing"] = {
        "current_stage": stage_key,
        "current_label": next((stage["label"] for stage in PIPELINE_STAGES if stage["key"] == stage_key), ""),
        "current_description": stage_description,
        "live_note": live_note or stage_description,
        "stages": stages,
        "preview_cards": preview_cards or jobs[job_id].get("processing", {}).get("preview_cards", []),
    }


def fallback_analysis(
    parsed_data: Dict[str, Any],
    statistics: Dict[str, Any],
    goal: Optional[str] = None,
) -> Dict[str, Any]:
    metadata = parsed_data.get("metadata", {})
    file_type = parsed_data.get("file_type", "unknown")
    has_numerical = parsed_data.get("has_numerical_data", False)
    rows = metadata.get("rows", 0)
    cols = metadata.get("columns", [])
    text_preview = (parsed_data.get("text_content") or "").strip()[:600]
    mode = "numerical" if has_numerical else "non_numerical"
    recommended_metrics = statistics.get("analysis_ready_metrics", [])
    insights = []
    recommendations = []

    if has_numerical:
        insights.append("The input contains measurable patterns that can be explained through comparisons, distributions, and segment performance.")
        if rows:
            insights.append(f"The dataset includes {rows} rows across {len(cols)} fields, giving enough scale for chart-based interpretation.")
        if recommended_metrics:
            insights.append(f"The strongest analysis-ready metrics are: {', '.join(recommended_metrics[:4])}.")
        for col, stats in list(statistics.get("numerical_summary", {}).items())[:2]:
            try:
                insights.append(
                    f"{col} ranges from {stats.get('min', 0):.2f} to {stats.get('max', 0):.2f}, "
                    f"with a typical value near {stats.get('mean', 0):.2f}."
                )
            except Exception:
                continue
        recommendations.extend(
            [
                "Review the leading charts first to identify the highest-performing and weakest segments.",
                "Use the follow-up question box to test specific hypotheses about trends, drivers, or anomalies.",
            ]
        )
        summary = (
            f"This dataset was processed as structured numerical data. The system identified the most meaningful metrics, "
            "generated explanatory charts, and prepared the results for plain-language interpretation."
        )
        overall_response = (
            "The analysis completed with local summary logic because advanced narrative reasoning was not available for this run."
        )
    else:
        if text_preview:
            insights.append("The input is primarily qualitative, so the analysis focuses on themes, intent, concerns, and recommendations.")
            insights.append("The content can be converted into a structured understanding of topics, entities, risks, and opportunities.")
        recommendations.extend(
            [
                "Use follow-up questions to probe deeper into risks, priorities, and hidden themes in the content.",
                "Review the structured interpretation before sharing the findings with non-technical stakeholders.",
            ]
        )
        summary = (
            "This input was processed as non-numerical content. The system prepared it for qualitative "
            "analysis and can summarize findings, recommendations, and follow-up answers."
        )
        overall_response = (
            "The input was prepared for qualitative interpretation, but richer narrative reasoning was not available for this run."
        )

    if goal:
        recommendations.insert(0, f"Align the next analysis pass to the user goal: {goal}")

    return {
        "analysis_mode": mode,
        "summary": summary,
        "insights": insights,
        "recommendations": recommendations,
        "overall_response": overall_response,
        "structured_view": {
            "input_type": file_type,
            "rows": rows,
            "columns": cols[:10],
            "analysis_ready_metrics": recommended_metrics[:5],
            "project_evaluation_metrics": [metric["name"] for metric in build_project_metrics()],
        },
        "suggested_questions": [
            "Which chart should I look at first?",
            "What is the biggest risk in this data?",
            "What are the strongest signals I should share with stakeholders?",
        ],
        "visual_story": [
            "Start with the broad distribution charts to understand spread and extremes.",
            "Use comparison charts to see which segments perform best or worst.",
        ],
    }


async def run_job(
    job_id: str,
    parsed_data: Dict[str, Any],
    original_name: str,
    goal: Optional[str] = None,
    fingerprint: Optional[str] = None,
) -> None:
    try:
        jobs[job_id]["status"] = "processing"
        set_stage(
            job_id,
            "input_received",
            8,
            live_note="Your input has entered the analysis journey.",
            preview_cards=build_preview_cards(parsed_data),
        )

        set_stage(
            job_id,
            "format_detection",
            18,
            live_note="We are determining the shape of the input so the right analysis path can be used.",
            preview_cards=build_preview_cards(parsed_data),
        )

        set_stage(
            job_id,
            "structure_mapping",
            28,
            live_note="We are mapping useful fields, text blocks, and context into a structured view.",
            preview_cards=build_preview_cards(parsed_data),
        )

        data = parsed_data.get("data")
        visualizations = {"charts": [], "has_charts": False, "chart_count": 0}
        statistics: Dict[str, Any] = {}

        set_stage(
            job_id,
            "metric_discovery",
            38,
            live_note="We are looking for the strongest signals, segments, and measurable patterns.",
            preview_cards=build_preview_cards(parsed_data),
        )

        if parsed_data.get("has_numerical_data") and isinstance(data, pd.DataFrame):
            visualizations = visualizer.generate_visualizations(data, os.path.splitext(original_name)[0])
            statistics = visualizer.generate_summary_stats(data)

        set_stage(
            job_id,
            "visualization_planning",
            52,
            live_note="We are preparing visual explanations that make the patterns easier to understand.",
            preview_cards=build_preview_cards(parsed_data, statistics),
        )

        set_stage(
            job_id,
            "insight_reasoning",
            66,
            live_note="We are comparing your strongest signals and looking for unusual patterns.",
            preview_cards=build_preview_cards(parsed_data, statistics),
        )

        analysis_result = None
        if ollama_running() and AI_ANALYSIS_AVAILABLE and run_analysis_pipeline:
            try:
                analysis_result = run_analysis_pipeline(
                    parsed_data=parsed_data,
                    model_name=settings.ollama_model,
                    user_goal=goal,
                    base_url=settings.ollama_base_url,
                )
            except Exception as exc:
                print(f"Ollama analysis failed: {exc}")

        if analysis_result is None:
            analysis_result = fallback_analysis(parsed_data, statistics, goal=goal)

        set_stage(
            job_id,
            "response_writing",
            82,
            live_note="We are turning the analysis into findings, recommendations, and a clear narrative.",
            preview_cards=build_preview_cards(parsed_data, statistics),
        )

        report_data_for_pdf = {
            "metadata": parsed_data.get("metadata", {}),
            "summary": analysis_result.get("summary", ""),
            "insights": analysis_result.get("insights", []),
            "recommendations": analysis_result.get("recommendations", []),
            "statistics": statistics,
            "visualizations": visualizations,
            "file_type": parsed_data.get("file_type", "unknown"),
            "has_numerical_data": parsed_data.get("has_numerical_data", False),
            "overall_response": analysis_result.get("overall_response", ""),
            "structured_view": analysis_result.get("structured_view", {}),
            "project_metrics": build_project_metrics(),
            "project_metric_results": extract_project_metric_results(parsed_data),
        }
        pdf_path = report_generator.generate_report(report_data_for_pdf, original_name)

        set_stage(
            job_id,
            "report_packaging",
            94,
            live_note="We are packaging the final response and preparing the downloadable report.",
            preview_cards=build_preview_cards(parsed_data, statistics),
        )

        report_data = {
            "metadata": parsed_data.get("metadata", {}),
            "summary": analysis_result.get("summary", ""),
            "insights": analysis_result.get("insights", []),
            "recommendations": analysis_result.get("recommendations", []),
            "statistics": statistics,
            "visualizations": build_chart_urls(visualizations),
            "file_type": parsed_data.get("file_type", "unknown"),
            "has_numerical_data": parsed_data.get("has_numerical_data", False),
            "analysis_mode": analysis_result.get("analysis_mode", "unknown"),
            "overall_response": analysis_result.get("overall_response", ""),
            "structured_view": analysis_result.get("structured_view", {}),
            "project_metrics": build_project_metrics(),
            "project_metric_results": extract_project_metric_results(parsed_data),
            "suggested_questions": analysis_result.get("suggested_questions", []),
            "visual_story": analysis_result.get("visual_story", []),
            "question_history": [],
        }

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["message"] = "Analysis complete!"
        jobs[job_id]["updated_at"] = datetime.now().isoformat()
        jobs[job_id]["processing"] = {
            "current_stage": "completed",
            "current_label": "Completed",
            "current_description": "Your analysis is ready to explore.",
            "live_note": "Your insights, charts, and report are ready.",
            "stages": [{**stage, "status": "done"} for stage in PIPELINE_STAGES],
            "preview_cards": build_preview_cards(parsed_data, statistics),
        }
        jobs[job_id]["result"] = report_data
        jobs[job_id]["pdf_path"] = pdf_path
        jobs[job_id]["parsed_data"] = serialize_parsed_data(parsed_data)
        if fingerprint:
            jobs[job_id]["fingerprint"] = fingerprint
            analysis_cache[fingerprint] = job_id

        result_path = os.path.join(settings.output_dir, f"{job_id}_result.json")
        with open(result_path, "w", encoding="utf-8") as file_handle:
            json.dump(jobs[job_id], file_handle, default=str, ensure_ascii=True)

    except Exception as exc:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = f"Error: {exc}"
        jobs[job_id]["updated_at"] = datetime.now().isoformat()


def ensure_job_loaded(job_id: str) -> Dict[str, Any]:
    if job_id in jobs:
        return jobs[job_id]

    result_path = os.path.join(settings.output_dir, f"{job_id}_result.json")
    if os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as file_handle:
            jobs[job_id] = json.load(file_handle)
        return jobs[job_id]

    raise HTTPException(status_code=404, detail="Job not found")


def get_cached_job(fingerprint: str) -> Optional[Dict[str, Any]]:
    cached_job_id = analysis_cache.get(fingerprint)
    if not cached_job_id:
        return None
    try:
        cached_job = ensure_job_loaded(cached_job_id)
    except HTTPException:
        analysis_cache.pop(fingerprint, None)
        return None
    if cached_job.get("status") == "completed":
        return cached_job
    return None


def create_job(source_name: str, source_type: str, size_mb: float, goal: Optional[str] = None) -> Dict[str, Any]:
    job_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Input received and queued for adaptive analysis",
        "created_at": now,
        "updated_at": now,
        "filename": source_name,
        "source_type": source_type,
        "file_size_mb": size_mb,
        "goal": goal,
        "processing": {
            "current_stage": "input_received",
            "current_label": "Input Received",
            "current_description": PIPELINE_STAGES[0]["description"],
            "live_note": "Your input is queued and ready for analysis.",
            "stages": [{**stage, "status": "waiting"} for stage in PIPELINE_STAGES],
            "preview_cards": [],
        },
    }
    return jobs[job_id]


@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "name": "ClarityLens Analytics API",
        "version": "2.0.0",
        "ollama_model": settings.ollama_model,
        "supported_formats": settings.allowed_extensions_list,
        "supports_direct_text_input": True,
        "supports_follow_up_questions": True,
    }


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "ollama_url": settings.ollama_base_url,
        "ollama_model": settings.ollama_model,
        "ollama_running": ollama_running(),
    }


@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard() -> str:
    dashboard_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "dashboard.html")
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as file_handle:
            return file_handle.read()
    return "<h1>Dashboard not found</h1><p>Please ensure static/dashboard.html exists</p>"


@app.post("/upload", response_model=JobStatus)
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> JobStatus:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    if not allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Supported formats: {settings.allowed_extensions}",
        )

    contents = await file.read()
    file_size_mb = len(contents) / (1024 * 1024)
    if file_size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB",
        )

    fingerprint = compute_fingerprint(contents, file.filename)
    cached_job = get_cached_job(fingerprint)
    if cached_job:
        cached_job["updated_at"] = datetime.now().isoformat()
        return JobStatus(**cached_job)

    job = create_job(file.filename, "file", file_size_mb)
    job_id = job["job_id"]
    extension = file.filename.rsplit(".", 1)[-1].lower()
    file_path = os.path.join(settings.upload_dir, f"{job_id}.{extension}")

    with open(file_path, "wb") as file_handle:
        file_handle.write(contents)

    parsed_data = file_parser.parse(file_path)
    background_tasks.add_task(run_job, job_id, parsed_data, file.filename, None, fingerprint)
    return JobStatus(**jobs[job_id])


@app.post("/analyze-text", response_model=JobStatus)
async def analyze_text(request: TextAnalysisRequest, background_tasks: BackgroundTasks) -> JobStatus:
    content = request.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="No text content provided")

    size_mb = len(content.encode("utf-8")) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=400,
            detail=f"Input too large. Maximum size: {settings.max_file_size_mb}MB",
        )

    source_name = request.title or "direct_input.txt"
    fingerprint = compute_fingerprint(content.encode("utf-8"), source_name, request.goal)
    cached_job = get_cached_job(fingerprint)
    if cached_job:
        cached_job["updated_at"] = datetime.now().isoformat()
        return JobStatus(**cached_job)

    job = create_job(source_name, "text", size_mb, goal=request.goal)
    job_id = job["job_id"]
    parsed_data = file_parser.parse_text_input(content, source_name=source_name)
    background_tasks.add_task(run_job, job_id, parsed_data, source_name, request.goal, fingerprint)
    return JobStatus(**jobs[job_id])


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str) -> JobStatus:
    job = ensure_job_loaded(job_id)
    return JobStatus(**job)


@app.get("/result/{job_id}")
async def get_analysis_result(job_id: str) -> Dict[str, Any]:
    job = ensure_job_loaded(job_id)
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed. Current status: {job['status']}")
    return job.get("result", {})


@app.post("/query/{job_id}")
async def query_job(job_id: str, request: QueryRequest) -> Dict[str, Any]:
    job = ensure_job_loaded(job_id)
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis must complete before querying it")

    parsed_data = restore_parsed_data(job.get("parsed_data", {}))
    prior_result = job.get("result", {})

    if ollama_running() and AI_ANALYSIS_AVAILABLE and answer_data_question:
        try:
            response = answer_data_question(
                question=request.question,
                parsed_data=parsed_data,
                prior_result=prior_result,
                model_name=settings.ollama_model,
                base_url=settings.ollama_base_url,
            )
            answer_text = response.get("answer", "")
        except Exception as exc:
            answer_text = (
                "The follow-up analysis could not complete successfully for this question. "
                f"Internal error: {exc}"
            )
    else:
        answer_text = (
            "Advanced follow-up reasoning is currently unavailable for this analysis session."
        )

    question_entry = {
        "question": request.question,
        "answer": answer_text,
        "created_at": datetime.now().isoformat(),
    }
    job.setdefault("result", {}).setdefault("question_history", []).append(question_entry)
    job["updated_at"] = datetime.now().isoformat()

    result_path = os.path.join(settings.output_dir, f"{job_id}_result.json")
    with open(result_path, "w", encoding="utf-8") as file_handle:
        json.dump(job, file_handle, default=str, ensure_ascii=True)

    return question_entry


@app.post("/chart-explain/{job_id}/{chart_id}")
async def explain_chart(job_id: str, chart_id: int) -> Dict[str, Any]:
    job = ensure_job_loaded(job_id)
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis must complete before explaining charts")

    result = job.get("result", {})
    charts = result.get("visualizations", {}).get("charts", [])
    chart = next((item for item in charts if item.get("chart_id") == chart_id), None)
    if not chart:
        raise HTTPException(status_code=404, detail="Chart not found")

    if ollama_running() and AI_ANALYSIS_AVAILABLE and answer_data_question:
        prompt = (
            f"Explain this chart for a non-technical user.\n\n"
            f"Chart title: {chart.get('title', '')}\n"
            f"Chart type: {chart.get('type', '')}\n"
            f"Caption: {chart.get('caption', '')}\n"
            f"Overall analysis summary: {result.get('summary', '')}\n"
            f"Key findings: {json.dumps(result.get('insights', []), ensure_ascii=True)}\n"
            "Describe what the chart shows, why it matters, and what the user should notice first."
        )
        parsed_data = restore_parsed_data(job.get("parsed_data", {}))
        try:
            explanation = answer_data_question(
                question=prompt,
                parsed_data=parsed_data,
                prior_result=result,
                model_name=settings.ollama_model,
                base_url=settings.ollama_base_url,
            ).get("answer", "")
        except Exception as exc:
            explanation = f"Chart explanation is temporarily unavailable. Internal error: {exc}"
    else:
        explanation = chart.get("caption", "This chart highlights one of the important patterns in the data.")

    return {"chart_id": chart_id, "title": chart.get("title", ""), "explanation": explanation}


@app.get("/download/{job_id}")
async def download_report(job_id: str) -> FileResponse:
    job = ensure_job_loaded(job_id)
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed. Current status: {job['status']}")

    pdf_path = job.get("pdf_path")
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF report not found")

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=os.path.basename(pdf_path),
    )


@app.delete("/job/{job_id}")
async def delete_job(job_id: str) -> Dict[str, str]:
    job = ensure_job_loaded(job_id)

    result_path = os.path.join(settings.output_dir, f"{job_id}_result.json")
    pdf_path = job.get("pdf_path")

    if pdf_path and os.path.exists(pdf_path):
        os.remove(pdf_path)
    if os.path.exists(result_path):
        os.remove(result_path)

    for filename in os.listdir(settings.upload_dir):
        if filename.startswith(job_id):
            upload_path = os.path.join(settings.upload_dir, filename)
            if os.path.isfile(upload_path):
                os.remove(upload_path)

    jobs.pop(job_id, None)
    return {"message": "Job deleted successfully"}


@app.get("/jobs")
async def list_jobs() -> Dict[str, Any]:
    return {
        "jobs": list(jobs.values()),
        "total": len(jobs),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
