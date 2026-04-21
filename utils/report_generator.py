"""
PDF report generator for analysis results using ReportLab.
"""
from datetime import datetime
from typing import Any, Dict
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


class ReportGenerator:
    """Generate PDF reports from analysis results using ReportLab."""

    def __init__(self, template_dir: str = "./templates", output_dir: str = "./output"):
        self.template_dir = template_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_report(self, analysis_data: Dict[str, Any], file_name: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"report_{os.path.splitext(file_name)[0]}_{timestamp}.pdf"
        pdf_path = os.path.join(self.output_dir, pdf_filename)

        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=24,
            textColor=colors.HexColor("#2c3e50"),
            spaceAfter=30,
            alignment=TA_CENTER,
        )
        heading_style = ParagraphStyle(
            "CustomHeading",
            parent=styles["Heading2"],
            fontSize=14,
            textColor=colors.HexColor("#34495e"),
            spaceBefore=20,
            spaceAfter=10,
            borderPadding=5,
            borderColor=colors.HexColor("#3498db"),
            borderWidth=2,
            leftIndent=10,
        )
        body_style = ParagraphStyle(
            "CustomBody",
            parent=styles["Normal"],
            fontSize=11,
            leading=16,
            textColor=colors.HexColor("#333333"),
            alignment=TA_JUSTIFY,
        )
        insight_style = ParagraphStyle(
            "InsightStyle",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#2c3e50"),
            leftIndent=20,
            spaceBefore=8,
            spaceAfter=8,
            backColor=colors.HexColor("#ecf0f1"),
            borderPadding=10,
        )

        story = []
        story.append(Paragraph(f"Analysis Report: {file_name}", title_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
        story.append(Spacer(1, 30))

        metadata = analysis_data.get("metadata", {})
        story.append(Paragraph("File Information", heading_style))
        story.append(Spacer(1, 10))
        metadata_table_data = [
            ["Property", "Value"],
            ["File Name", str(metadata.get("file_name", "N/A"))],
            ["File Type", str(analysis_data.get("file_type", "unknown")).upper()],
            ["File Size", f"{metadata.get('file_size_mb', 0):.2f} MB"],
        ]
        if metadata.get("rows"):
            metadata_table_data.append(["Total Rows", str(metadata.get("rows"))])
        if metadata.get("columns"):
            metadata_table_data.append(["Total Columns", str(len(metadata.get("columns", [])))])
        metadata_table_data.append(
            ["Contains Numerical Data", "Yes" if analysis_data.get("has_numerical_data") else "No"]
        )

        metadata_table = Table(metadata_table_data, colWidths=[2 * inch, 4 * inch])
        metadata_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 1), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                    ("TOPPADDING", (0, 0), (-1, -1), 12),
                    ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#ddd")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
                ]
            )
        )
        story.append(metadata_table)
        story.append(Spacer(1, 20))

        story.append(Paragraph("Executive Summary", heading_style))
        story.append(Spacer(1, 10))
        summary = analysis_data.get("summary", "No summary available.")
        story.append(Paragraph(str(summary).replace("\n", "<br/>"), body_style))
        story.append(Spacer(1, 20))

        story.append(Paragraph("Key Insights", heading_style))
        story.append(Spacer(1, 10))
        insights = analysis_data.get("insights", [])
        if insights:
            for index, insight in enumerate(insights, 1):
                clean_insight = str(insight).replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(f"<b>{index}.</b> {clean_insight}", insight_style))
                story.append(Spacer(1, 5))
        else:
            story.append(Paragraph("No specific insights were generated for this input.", body_style))
        story.append(Spacer(1, 20))

        structured_view = analysis_data.get("structured_view", {})
        if structured_view:
            story.append(Paragraph("Structured Interpretation", heading_style))
            story.append(Spacer(1, 10))
            for key, value in list(structured_view.items())[:8]:
                clean_key = str(key).replace("_", " ").title()
                if isinstance(value, list):
                    clean_value = ", ".join(str(item) for item in value[:8])
                else:
                    clean_value = str(value)
                story.append(Paragraph(f"<b>{clean_key}:</b> {clean_value}", body_style))
                story.append(Spacer(1, 6))
            story.append(Spacer(1, 14))

        metric_results = analysis_data.get("project_metric_results", [])
        if metric_results:
            story.append(Paragraph("Evaluation Metric Results", heading_style))
            story.append(Spacer(1, 10))
            metrics_table_data = [["Metric", "Detected Result", "Interpretation"]]
            for metric in metric_results[:5]:
                metrics_table_data.append(
                    [
                        str(metric.get("name", "N/A")),
                        str(metric.get("display_value", "Not available")),
                        str(metric.get("verdict", "No interpretation available")),
                    ]
                )
            metrics_table = Table(metrics_table_data, colWidths=[1.5 * inch, 1.5 * inch, 3.0 * inch])
            metrics_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#8e44ad")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                        ("TOPPADDING", (0, 0), (-1, -1), 10),
                        ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#ddd")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f5fb")]),
                    ]
                )
            )
            story.append(metrics_table)
            story.append(Spacer(1, 20))

        statistics = analysis_data.get("statistics", {})
        if statistics and statistics.get("basic_info"):
            story.append(Paragraph("Statistical Summary", heading_style))
            story.append(Spacer(1, 10))

            basic_info = statistics.get("basic_info", {})
            stats_table_data = [
                ["Metric", "Value"],
                ["Total Rows", str(basic_info.get("total_rows", "N/A"))],
                ["Total Columns", str(basic_info.get("total_columns", "N/A"))],
                ["Memory Usage", f"{basic_info.get('memory_usage_mb', 0):.2f} MB"],
            ]
            stats_table = Table(stats_table_data, colWidths=[3 * inch, 3 * inch])
            stats_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#27ae60")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                        ("TOPPADDING", (0, 0), (-1, -1), 10),
                        ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#ddd")),
                    ]
                )
            )
            story.append(stats_table)
            story.append(Spacer(1, 20))

            for col_name, column_stats in list(statistics.get("numerical_summary", {}).items())[:5]:
                story.append(Paragraph(f"<b>{col_name}</b>", styles["Heading4"]))
                rows = [["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]]
                try:
                    rows.append(
                        [
                            f"{column_stats.get('count', 0):.1f}",
                            f"{column_stats.get('mean', 0):.2f}",
                            f"{column_stats.get('std', 0):.2f}",
                            f"{column_stats.get('min', 0):.2f}",
                            f"{column_stats.get('25%', 0):.2f}",
                            f"{column_stats.get('50%', 0):.2f}",
                            f"{column_stats.get('75%', 0):.2f}",
                            f"{column_stats.get('max', 0):.2f}",
                        ]
                    )
                except Exception:
                    continue

                table = Table(rows, colWidths=[0.75 * inch] * 8)
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                            ("FONTSIZE", (0, 0), (-1, -1), 8),
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#ddd")),
                        ]
                    )
                )
                story.append(table)
                story.append(Spacer(1, 10))

        visualizations = analysis_data.get("visualizations", {})
        if visualizations.get("has_charts"):
            story.append(PageBreak())
            story.append(Paragraph("Visualizations", heading_style))
            story.append(Spacer(1, 10))
            for chart in visualizations.get("charts", []):
                chart_path = chart.get("image_path", "")
                if chart_path and os.path.exists(chart_path):
                    story.append(Paragraph(f"<b>{chart.get('title', 'Chart')}</b>", styles["Heading3"]))
                    story.append(Spacer(1, 5))
                    try:
                        story.append(Image(chart_path, width=5 * inch, height=3 * inch))
                        if chart.get("caption"):
                            story.append(Spacer(1, 6))
                            story.append(Paragraph(str(chart.get("caption")), body_style))
                        story.append(Spacer(1, 15))
                    except Exception:
                        story.append(Paragraph(f"[Chart could not be embedded: {chart.get('title', 'Chart')}]", body_style))

        story.append(Spacer(1, 20))
        story.append(Paragraph("Recommendations", heading_style))
        story.append(Spacer(1, 10))
        recommendations = analysis_data.get("recommendations", [])
        if recommendations:
            for index, recommendation in enumerate(recommendations, 1):
                clean_rec = str(recommendation).replace("<", "&lt;").replace(">", "&gt;")
                rec_style = ParagraphStyle(
                    f"RecStyle{index}",
                    parent=styles["Normal"],
                    fontSize=10,
                    leading=14,
                    textColor=colors.HexColor("#856404"),
                    backColor=colors.HexColor("#fff3cd"),
                    leftIndent=20,
                    borderPadding=10,
                )
                story.append(Paragraph(f"<b>Recommendation {index}:</b> {clean_rec}", rec_style))
                story.append(Spacer(1, 8))
        else:
            story.append(Paragraph("No specific recommendations available.", body_style))

        overall_response = analysis_data.get("overall_response")
        if overall_response:
            story.append(Spacer(1, 20))
            story.append(Paragraph("Overall Response", heading_style))
            story.append(Spacer(1, 10))
            story.append(Paragraph(str(overall_response).replace("\n", "<br/>"), body_style))

        footer_style = ParagraphStyle(
            "Footer",
            parent=styles["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#7f8c8d"),
            alignment=TA_CENTER,
        )
        story.append(Spacer(1, 40))
        story.append(Paragraph("-" * 50, footer_style))
        story.append(Paragraph("This report was automatically generated by ClarityLens Analytics", footer_style))
        story.append(Paragraph("Built for clear, human-friendly data understanding", footer_style))

        doc.build(story)
        return pdf_path
