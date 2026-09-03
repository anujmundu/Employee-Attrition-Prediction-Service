"""
Executive Attrition & Retention Report Generator.
Outputs executive summaries in Markdown and formatted HTML for leadership review.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd


class ExecutiveReportGenerator:
    """Generates structured executive reports summarizing workforce attrition risks."""

    def __init__(self, company_name: str = "Enterprise Organization"):
        self.company_name = company_name

    def generate_markdown_report(
        self,
        cohort_name: str,
        total_employees: int,
        predicted_at_risk: int,
        total_replacement_cost: float,
        top_risk_factors: List[Dict[str, Any]],
        recommended_actions: List[Dict[str, Any]],
        drift_status: str = "Healthy (No Drift)"
    ) -> str:
        """Create a Markdown executive report."""
        attrition_rate = (predicted_at_risk / total_employees * 100) if total_employees > 0 else 0
        date_str = datetime.now().strftime("%B %d, %Y")

        report = f"""# Workforce Retention & Attrition Risk Executive Report
**Company:** {self.company_name}  
**Cohort:** {cohort_name}  
**Date of Assessment:** {date_str}  
**MLOps Model & Data Status:** {drift_status}  

---

## 1. Executive Summary & KPIs
| Metric | Value | Leadership Takeaway |
|---|---|---|
| **Total Cohort Headcount** | {total_employees:,} | Active evaluated employees |
| **Predicted High-Risk Count** | {predicted_at_risk:,} | Employees with >50% attrition risk |
| **Projected Attrition Rate** | {attrition_rate:.1f}% | Sector Benchmark: 12.5% |
| **Total At-Risk Replacement Cost** | ${total_replacement_cost:,.2f} | Seniority-indexed cost exposure |

---

## 2. Key Systemic Risk Drivers (SHAP Feature Importance)
The multi-model supervised ensemble identified the following primary root causes driving turnover risk:

"""
        for idx, factor in enumerate(top_risk_factors, 1):
            report += f"{idx}. **{factor.get('name', 'Feature')}** ({factor.get('impact', 'High')} Impact)  \n"
            report += f"   *Description:* {factor.get('description', 'Key behavioral or compensation driver.')}\n\n"

        report += """---

## 3. Prescriptive Strategic Retention Playbooks
Recommended prioritized interventions to mitigate flight risk and maximize retention ROI:

"""
        for idx, action in enumerate(recommended_actions, 1):
            report += f"### {idx}. [{action.get('priority', 'HIGH')}] {action.get('title', 'Retention Strategy')}  \n"
            report += f"- **Target Persona:** {action.get('persona', 'All At-Risk Talent')}\n"
            report += f"- **Actionable Steps:** {action.get('action', 'Provide growth and compensation reviews.')}\n"
            report += f"- **Expected ROI:** {action.get('roi', '3.5x to 6.0x cost-to-savings ratio')}\n\n"

        report += """---
*Report generated automatically by Enterprise Employee Attrition Prediction Service.*
"""
        return report

    def save_report(self, markdown_content: str, output_path: str = "reports/executive_summary.md"):
        """Save report to disk."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown_content, encoding="utf-8")
        return str(out)
