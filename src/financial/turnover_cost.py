from src.core.config import TURNOVER_COST_RATES


def determine_seniority_tier(job_role: str, monthly_income: float, years_at_company: int) -> str:
    """Classifies employee into a seniority level for financial modeling."""
    role_lower = str(job_role).lower()
    
    if any(k in role_lower for k in ["vp", "director", "chief", "executive", "principal", "managing"]):
        return "EXECUTIVE"
    if any(k in role_lower for k in ["senior", "lead", "manager", "architect", "supervisor"]):
        return "SENIOR"
    if monthly_income > 9000 or years_at_company >= 6:
        return "SENIOR"
    if monthly_income < 3500 and years_at_company <= 2:
        return "ENTRY"
    return "MID"


def calculate_turnover_financials(
    monthly_income: float,
    job_role: str = "Specialist",
    years_at_company: int = 3,
    attrition_probability: float = 0.5,
    intervention_cost: float = 2500.0,
    expected_risk_reduction: float = 0.40,
) -> dict:
    """
    Computes turnover cost, expected financial loss, and ROI of retention intervention.
    
    Financial Model:
    - Annual Salary = Monthly Income * 12
    - Replacement Cost = Annual Salary * Seniority Multiplier (0.5x to 1.75x)
      (accounting for recruitment, vacancy loss, onboarding, training & institutional knowledge)
    - Expected Financial Loss = Replacement Cost * Attrition Probability
    - Potential Retained Value = Expected Loss * expected_risk_reduction
    - Net ROI = (Potential Retained Value - Intervention Cost) / Intervention Cost * 100%
    """
    annual_salary = float(monthly_income * 12)
    tier = determine_seniority_tier(job_role, monthly_income, years_at_company)
    multiplier = TURNOVER_COST_RATES.get(tier, 0.75)
    
    replacement_cost = round(annual_salary * multiplier, 2)
    expected_loss = round(replacement_cost * attrition_probability, 2)
    
    potential_savings = round(expected_loss * expected_risk_reduction, 2)
    net_benefit = round(potential_savings - intervention_cost, 2)
    roi_percent = round((net_benefit / max(intervention_cost, 1.0)) * 100.0, 1)
    
    return {
        "annual_salary": annual_salary,
        "seniority_tier": tier,
        "cost_multiplier": multiplier,
        "replacement_cost": replacement_cost,
        "expected_loss_at_risk": expected_loss,
        "estimated_intervention_cost": intervention_cost,
        "potential_retained_savings": potential_savings,
        "net_retention_roi_percent": roi_percent,
    }


class TurnoverCostCalculator:
    """Class wrapper for computing seniority-indexed turnover replacement cost."""

    def calculate_cost(
        self,
        job_role: str,
        department: str = "Research & Development",
        monthly_income: float = 5000.0,
        years_at_company: int = 3,
        attrition_probability: float = 0.5,
        intervention_cost: float = 2500.0
    ) -> dict:
        tier = determine_seniority_tier(job_role, monthly_income, years_at_company)
        fin = calculate_turnover_financials(
            monthly_income=monthly_income,
            job_role=job_role,
            years_at_company=years_at_company,
            attrition_probability=attrition_probability,
            intervention_cost=intervention_cost
        )
        fin["department"] = department
        fin["cost_breakdown"] = {
            "hiring_cost": round(fin["replacement_cost"] * 0.35, 2),
            "lost_productivity_cost": round(fin["replacement_cost"] * 0.45, 2),
            "onboarding_training_cost": round(fin["replacement_cost"] * 0.20, 2)
        }
        return fin

