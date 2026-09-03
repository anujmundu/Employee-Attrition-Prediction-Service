# AI Ethics, Fairness, and Model Governance

## 1. Governance Principles
Workforce attrition predictive models carry operational and ethical responsibilities. The service enforces strict fairness checks, explainability mandates, and human-in-the-loop safeguards.

---

## 2. Fairness & Anti-Bias Standards (AIF360 Alignment)
- **Protected Attributes**: Models are explicitly tested against demographic disparities (Age, Gender, Marital Status) to prevent disparate impact.
- **Disparate Impact Ratio (DIR)**: Monitored to ensure acceptance rates across demographic slices remain within the `[0.80, 1.25]` regulatory threshold (Four-Fifths Rule).
- **Equality of Opportunity**: True Positive Rates across protected classes are continuously evaluated using the AIF360 fairness evaluation benchmark dataset.

---

## 3. Explainability & Right-to-Explanation
- **No Black-Box Decisions**: Predictions must be accompanied by TreeSHAP local attribution explanations.
- **Actionable Levers**: The platform focuses on organizational and workplace factors that leadership can constructively influence (compensation reviews, workload rebalancing, promotion pathways) rather than unmodifiable personal attributes.

---

## 4. Human-in-the-Loop Protocol
- Predictive outputs are **advisory decision-support tools** for People Operations and HR business partners, not automated personnel actions.
- Retention recommendations require HR and management discretion prior to implementation.
