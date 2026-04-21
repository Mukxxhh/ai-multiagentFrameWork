"""
Adaptive multi-agent analytics pipeline powered directly by Ollama.
This avoids brittle framework-specific LLM adapters while preserving
the product's multi-agent reasoning model.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pandas as pd
from ollama import Client


def _safe_json_loads(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except Exception:
        return {"raw": raw}


class OllamaAgentOrchestrator:
    """Runs router, analyst, storyteller, and Q&A roles against Ollama."""

    def __init__(self, model_name: str = "llama3.1", base_url: str = "http://localhost:11434"):
        self.client = Client(host=base_url)
        self.model_name = model_name
        self.base_url = base_url

    def analyze_file(self, parsed_data: Dict[str, Any], user_goal: Optional[str] = None) -> Dict[str, Any]:
        mode = self._detect_mode(parsed_data)
        context = self._prepare_context(parsed_data, user_goal=user_goal)

        analyst = self._chat_json(
            system_prompt=(
                "You are a senior data analyst and insights strategist. "
                "Your job is to deeply brainstorm the supplied data and explain what matters in plain business language."
            ),
            user_prompt=(
                "Return strict JSON with these keys:\n"
                "executive_summary: string\n"
                "key_findings: array of 5 to 7 strings\n"
                "recommendations: array of 4 to 6 strings\n"
                "overall_response: string\n"
                "structured_view: object\n"
                "suggested_questions: array of 3 to 5 strings\n"
                "visual_story: array of 3 to 5 strings\n\n"
                f"Detected mode: {mode}\n"
                f"Audience: non-technical users who need clarity and decision support.\n"
                f"Priority focus: explain what matters, what is unusual, what should be watched, and what to do next.\n\n"
                f"Input context:\n{context}\n\n"
                "Important requirements:\n"
                "- Speak to non-technical users.\n"
                "- If the data is numerical, focus on trends, drivers, anomalies, segmentation, and business meaning.\n"
                "- If the data is non-numerical, convert the unstructured content into a structured understanding using themes, entities, topics, risks, and opportunities.\n"
                "- Do not mention Ollama, models, or technical implementation.\n"
                "- Avoid weak observations like simply repeating column names unless they matter.\n"
                "- Prefer interpretation over description."
            ),
            temperature=0.25,
        )

        final = {
            "analysis_mode": mode,
            "summary": analyst.get("executive_summary") or "",
            "insights": analyst.get("key_findings") or [],
            "recommendations": analyst.get("recommendations") or [],
            "overall_response": analyst.get("overall_response") or "",
            "structured_view": analyst.get("structured_view") or {},
            "suggested_questions": analyst.get("suggested_questions") or [],
            "visual_story": analyst.get("visual_story") or [],
            "routing": {
                "analysis_mode": mode,
                "audience": "non-technical users",
                "decision_lens": "plain-language explanation and actionability",
            },
        }

        return self._normalize_result(final, mode)

    def answer_question(
        self,
        question: str,
        parsed_data: Dict[str, Any],
        prior_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = self._prepare_context(parsed_data)
        prior = json.dumps(prior_result or {}, ensure_ascii=True)[:8000]

        response = self._chat_text(
            system_prompt=(
                "You answer user questions about analyzed data. "
                "Use only the provided context and be clear for non-technical readers."
            ),
            user_prompt=(
                f"Question:\n{question}\n\n"
                f"Input context:\n{context}\n\n"
                f"Previous analysis:\n{prior}\n\n"
                "Answer in three sections:\n"
                "Direct Answer:\n"
                "Evidence:\n"
                "What To Look At Next:\n"
                "Do not mention technical implementation details."
            ),
            temperature=0.2,
        )
        return {"question": question, "answer": response}

    def _chat_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> Dict[str, Any]:
        response = self.client.chat(
            model=self.model_name,
            format="json",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": temperature},
        )
        content = response["message"]["content"]
        return _safe_json_loads(content)

    def _chat_text(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": temperature},
        )
        return response["message"]["content"].strip()

    def _detect_mode(self, parsed_data: Dict[str, Any]) -> str:
        data = parsed_data.get("data")
        has_numeric = parsed_data.get("has_numerical_data", False)
        text_content = (parsed_data.get("text_content") or "").strip()
        if isinstance(data, pd.DataFrame) and not data.empty:
            if has_numeric:
                return "numerical"
            return "structured_non_numerical"
        if has_numeric and text_content:
            return "mixed"
        if has_numeric:
            return "numerical"
        if text_content:
            return "unstructured"
        return "unknown"

    def _prepare_context(self, parsed_data: Dict[str, Any], user_goal: Optional[str] = None) -> str:
        metadata = parsed_data.get("metadata", {})
        data = parsed_data.get("data")
        text_content = parsed_data.get("text_content") or ""

        parts = [
            f"Detected mode: {self._detect_mode(parsed_data)}",
            f"Input type: {parsed_data.get('file_type', 'unknown')}",
            f"Input name: {metadata.get('file_name', 'direct_input')}",
        ]
        if user_goal:
            parts.append(f"User goal: {user_goal}")
        if metadata.get("rows"):
            parts.append(f"Rows: {metadata['rows']}")
        if metadata.get("columns"):
            parts.append(f"Columns: {', '.join(str(col) for col in metadata['columns'][:25])}")

        if isinstance(data, pd.DataFrame) and not data.empty:
            numeric_cols = self._select_analysis_metrics(data)
            category_cols = [col for col in data.columns if col not in numeric_cols][:8]
            parts.append("Data preview:\n" + data.head(10).to_string())

            if numeric_cols:
                describe = data[numeric_cols].describe().round(3).to_string()
                parts.append("Important numerical summary:\n" + describe)

            if category_cols:
                category_summaries = []
                for column in category_cols[:5]:
                    counts = data[column].astype(str).value_counts(dropna=False).head(5).to_dict()
                    category_summaries.append(f"{column}: {counts}")
                parts.append("Top categories:\n" + "\n".join(category_summaries))

            missing = data.isna().sum()
            missing_lines = [f"{col}: {int(val)}" for col, val in missing[missing > 0].head(10).items()]
            if missing_lines:
                parts.append("Missing values:\n" + "\n".join(missing_lines))

        if text_content:
            parts.append("Text excerpt:\n" + text_content[:5000])

        return "\n\n".join(parts)

    def _select_analysis_metrics(self, data: pd.DataFrame) -> List[str]:
        numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
        scored: List[tuple[float, str]] = []
        for column in numeric_cols:
            score = 1.0
            lower = column.lower()
            series = data[column].dropna()
            unique_ratio = series.nunique() / max(len(series), 1)
            if any(token in lower for token in ["id", "zip", "postal", "phone", "lat", "lon", "lng", "code"]):
                score -= 4
            if unique_ratio > 0.95:
                score -= 1.5
            if series.nunique() <= 2:
                score -= 0.5
            if abs(series.mean()) > 0:
                score += min(float(series.std() / max(abs(series.mean()), 1)), 3)
            score += min(series.notna().mean(), 1)
            scored.append((score, column))
        scored.sort(reverse=True)
        selected = [column for score, column in scored if score > 0]
        return selected[:6] or numeric_cols[:4]

    def _normalize_result(self, result: Dict[str, Any], mode: str) -> Dict[str, Any]:
        normalized_mode = result.get("analysis_mode", mode)
        if normalized_mode not in {"numerical", "mixed", "structured_non_numerical", "unstructured", "unknown"}:
            normalized_mode = mode
        return {
            "analysis_mode": normalized_mode,
            "summary": str(result.get("summary", "")).strip(),
            "insights": self._normalize_list(result.get("insights"), minimum=4),
            "recommendations": self._normalize_list(result.get("recommendations"), minimum=3),
            "overall_response": str(result.get("overall_response", "")).strip(),
            "structured_view": result.get("structured_view") if isinstance(result.get("structured_view"), dict) else {},
            "suggested_questions": self._normalize_list(result.get("suggested_questions"), minimum=0),
            "visual_story": self._normalize_list(result.get("visual_story"), minimum=0),
            "routing": result.get("routing", {}),
        }

    def _normalize_list(self, value: Any, minimum: int = 0) -> List[str]:
        if isinstance(value, list):
            cleaned = [str(item).strip() for item in value if str(item).strip()]
        elif isinstance(value, str) and value.strip():
            cleaned = [line.strip("- ").strip() for line in value.splitlines() if line.strip()]
        else:
            cleaned = []
        if len(cleaned) < minimum:
            return cleaned
        return cleaned


def run_analysis_pipeline(
    parsed_data: Dict[str, Any],
    model_name: str = "llama3.1",
    user_goal: Optional[str] = None,
    base_url: str = "http://localhost:11434",
) -> Dict[str, Any]:
    orchestrator = OllamaAgentOrchestrator(model_name=model_name, base_url=base_url)
    return orchestrator.analyze_file(parsed_data, user_goal=user_goal)


def answer_data_question(
    question: str,
    parsed_data: Dict[str, Any],
    prior_result: Optional[Dict[str, Any]] = None,
    model_name: str = "llama3.1",
    base_url: str = "http://localhost:11434",
) -> Dict[str, Any]:
    orchestrator = OllamaAgentOrchestrator(model_name=model_name, base_url=base_url)
    return orchestrator.answer_question(question=question, parsed_data=parsed_data, prior_result=prior_result)
