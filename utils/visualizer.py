"""
Visualization generator with smarter metric selection and chart narratives.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder


class VisualizationGenerator:
    """Generate understandable charts for non-technical users."""

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_visualizations(self, data: pd.DataFrame, file_name: str, max_charts: int = 6) -> Dict[str, Any]:
        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            return {"charts": [], "has_charts": False, "chart_count": 0}

        charts: List[Dict[str, Any]] = []
        numeric_cols = self._rank_numeric_columns(data)
        categorical_cols = self._rank_categorical_columns(data)
        datetime_cols = self._detect_datetime_columns(data)

        if datetime_cols and numeric_cols and len(charts) < max_charts:
            charts.extend(
                self._generate_time_series_charts(
                    data,
                    datetime_cols=datetime_cols,
                    numeric_cols=numeric_cols,
                    max_remaining=max_charts - len(charts),
                )
            )

        if numeric_cols and len(charts) < max_charts:
            charts.extend(
                self._generate_distribution_charts(
                    data, numeric_cols=numeric_cols, max_remaining=max_charts - len(charts)
                )
            )

        if len(numeric_cols) >= 2 and len(charts) < max_charts:
            charts.extend(
                self._generate_relationship_charts(
                    data, numeric_cols=numeric_cols, max_remaining=max_charts - len(charts)
                )
            )

        if categorical_cols and numeric_cols and len(charts) < max_charts:
            charts.extend(
                self._generate_category_comparison_charts(
                    data,
                    categorical_cols=categorical_cols,
                    numeric_cols=numeric_cols,
                    max_remaining=max_charts - len(charts),
                )
            )

        for index, chart in enumerate(charts):
            chart_path = os.path.join(self.output_dir, f"{file_name}_chart_{index + 1}.png")
            self._save_chart_image(chart["figure"], chart_path)
            chart["image_path"] = chart_path
            chart["chart_id"] = index + 1
            chart["figure_json"] = json.loads(json.dumps(chart["figure"].to_plotly_json(), cls=PlotlyJSONEncoder))

        return {"charts": charts, "has_charts": bool(charts), "chart_count": len(charts)}

    def generate_summary_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            return {}

        stats = {
            "basic_info": {
                "total_rows": len(data),
                "total_columns": len(data.columns),
                "memory_usage_mb": data.memory_usage(deep=True).sum() / (1024 * 1024),
            },
            "missing_values": data.isnull().sum().to_dict(),
            "numerical_summary": {},
            "categorical_summary": {},
            "analysis_ready_metrics": self._rank_numeric_columns(data),
        }

        numerical_cols = self._rank_numeric_columns(data)
        if numerical_cols:
            stats["numerical_summary"] = data[numerical_cols].describe().to_dict()

        for col in self._rank_categorical_columns(data)[:5]:
            stats["categorical_summary"][col] = {
                "unique_values": data[col].nunique(),
                "top_value": data[col].mode().iloc[0] if len(data[col].mode()) > 0 else None,
                "value_counts": data[col].astype(str).value_counts().head(5).to_dict(),
            }

        return stats

    def _generate_time_series_charts(
        self,
        data: pd.DataFrame,
        datetime_cols: List[str],
        numeric_cols: List[str],
        max_remaining: int,
    ) -> List[Dict[str, Any]]:
        charts = []
        date_col = datetime_cols[0]
        metric = numeric_cols[0]
        temp = data[[date_col, metric]].dropna().copy()
        temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
        temp = temp.dropna()
        if temp.empty:
            return charts

        grouped = temp.groupby(pd.Grouper(key=date_col, freq="ME"))[metric].mean().reset_index()
        if grouped.empty:
            return charts

        fig = px.line(
            grouped,
            x=date_col,
            y=metric,
            title=f"{metric} Trend Over Time",
            markers=True,
        )
        fig.update_layout(template="plotly_white", height=430, margin=dict(l=40, r=20, t=60, b=40))
        charts.append(
            {
                "type": "time_trend",
                "title": f"{metric} Trend Over Time",
                "caption": f"This chart shows how {metric} changes over time, making it easier to spot sustained rises, drops, or unusual shifts.",
                "figure": fig,
            }
        )
        return charts[:max_remaining]

    def _generate_distribution_charts(self, data: pd.DataFrame, numeric_cols: List[str], max_remaining: int) -> List[Dict[str, Any]]:
        charts = []
        for col in numeric_cols[: min(2, max_remaining)]:
            fig = px.histogram(
                data,
                x=col,
                title=f"Distribution of {col}",
                nbins=30,
                marginal="box",
                color_discrete_sequence=["#0f766e"],
            )
            fig.update_layout(template="plotly_white", height=420, margin=dict(l=40, r=20, t=60, b=40))
            charts.append(
                {
                    "type": "distribution",
                    "title": f"Distribution of {col}",
                    "caption": f"This view shows where most {col} values are concentrated and whether there are extreme values that deserve attention.",
                    "figure": fig,
                }
            )
        return charts

    def _generate_relationship_charts(self, data: pd.DataFrame, numeric_cols: List[str], max_remaining: int) -> List[Dict[str, Any]]:
        charts = []
        if max_remaining <= 0:
            return charts

        corr = data[numeric_cols[: min(6, len(numeric_cols))]].corr()
        heatmap = px.imshow(
            corr,
            title="Relationship Heatmap",
            color_continuous_scale="Tealrose",
            aspect="auto",
        )
        heatmap.update_layout(template="plotly_white", height=480, margin=dict(l=40, r=20, t=60, b=40))
        charts.append(
            {
                "type": "relationship_heatmap",
                "title": "Relationship Heatmap",
                "caption": "This chart highlights which important metrics tend to move together, helping users see possible drivers and dependencies.",
                "figure": heatmap,
            }
        )

        if len(numeric_cols) >= 2 and max_remaining >= 2:
            x_col, y_col = numeric_cols[0], numeric_cols[1]
            scatter = px.scatter(
                data,
                x=x_col,
                y=y_col,
                title=f"{x_col} vs {y_col}",
                trendline="ols",
                color_discrete_sequence=["#be123c"],
            )
            scatter.update_layout(template="plotly_white", height=420, margin=dict(l=40, r=20, t=60, b=40))
            charts.append(
                {
                    "type": "relationship_scatter",
                    "title": f"{x_col} vs {y_col}",
                    "caption": f"This comparison helps users see whether higher {x_col} tends to align with higher or lower {y_col}.",
                    "figure": scatter,
                }
            )

        return charts[:max_remaining]

    def _generate_category_comparison_charts(
        self,
        data: pd.DataFrame,
        categorical_cols: List[str],
        numeric_cols: List[str],
        max_remaining: int,
    ) -> List[Dict[str, Any]]:
        charts = []
        category_col = categorical_cols[0]
        metric_col = numeric_cols[0]

        grouped = (
            data[[category_col, metric_col]]
            .dropna()
            .groupby(category_col)[metric_col]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        if grouped.empty:
            return charts

        fig = px.bar(
            grouped,
            x=category_col,
            y=metric_col,
            title=f"Average {metric_col} by {category_col}",
            color=metric_col,
            color_continuous_scale="Sunsetdark",
        )
        fig.update_layout(
            template="plotly_white",
            height=430,
            margin=dict(l=40, r=20, t=60, b=80),
            xaxis_tickangle=-35,
            coloraxis_showscale=False,
        )
        charts.append(
            {
                "type": "category_comparison",
                "title": f"Average {metric_col} by {category_col}",
                "caption": f"This chart compares the typical {metric_col} across the leading {category_col} groups so users can quickly spot the strongest and weakest segments.",
                "figure": fig,
            }
        )
        return charts[:max_remaining]

    def _rank_numeric_columns(self, data: pd.DataFrame) -> List[str]:
        scored = []
        for column in data.select_dtypes(include=["number"]).columns:
            lower = column.lower()
            series = data[column].dropna()
            if series.empty:
                continue

            score = 1.0
            unique_ratio = series.nunique() / max(len(series), 1)
            if any(token in lower for token in ["id", "zip", "postal", "lat", "lon", "lng", "phone", "code"]):
                score -= 5
            if unique_ratio > 0.98:
                score -= 1.5
            if series.nunique() <= 2:
                score -= 0.5
            score += min(series.std() / max(abs(series.mean()), 1), 3)
            score += min(series.notna().mean(), 1)
            scored.append((score, column))

        scored.sort(reverse=True)
        ordered = [column for score, column in scored if score > 0]
        return ordered or data.select_dtypes(include=["number"]).columns.tolist()[:4]

    def _rank_categorical_columns(self, data: pd.DataFrame) -> List[str]:
        candidates = []
        for column in data.select_dtypes(include=["object", "category", "bool"]).columns:
            series = data[column].dropna().astype(str)
            if series.empty:
                continue
            unique_count = series.nunique()
            if unique_count <= 1:
                continue
            score = 1.0
            if unique_count <= 12:
                score += 2
            elif unique_count <= 30:
                score += 1
            avg_len = series.str.len().mean()
            if avg_len < 40:
                score += 0.5
            candidates.append((score, column))
        candidates.sort(reverse=True)
        return [column for _, column in candidates]

    def _detect_datetime_columns(self, data: pd.DataFrame) -> List[str]:
        detected = []
        for column in data.columns:
            if pd.api.types.is_datetime64_any_dtype(data[column]):
                detected.append(column)
                continue
            if data[column].dtype == object:
                sample = data[column].dropna().head(100)
                if sample.empty:
                    continue
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().mean() >= 0.7:
                    detected.append(column)
        return detected

    def _save_chart_image(self, fig: Any, path: str) -> None:
        fig.write_image(path, engine="kaleido")
