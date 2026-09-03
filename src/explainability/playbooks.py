def generate_retention_playbook(employee_data: dict, top_risk_drivers: list = None) -> list:
    """
    Generates tailored, actionable HR retention playbooks based on an employee's
    risk profile and feature attributions.
    """
    playbook = []
    
    # 1. OverTime / Burnout
    if employee_data.get("OverTime") == "Yes" or employee_data.get("WorkLifeBalance", 3) <= 2:
        playbook.append({
            "pillar": "Work-Life Balance & Burnout",
            "urgency": "HIGH",
            "issue_detected": "Severe overtime commitments and compromised work-life harmony.",
            "action_item": "Cap weekly overtime hours, redistribute critical sprint tasks, and mandate alternate Friday recovery days.",
            "estimated_budget": 500,
            "projected_risk_reduction_pct": 35,
        })
        
    # 2. Compensation & Market Parity
    monthly_income = float(employee_data.get("MonthlyIncome", 5000))
    salary_hike = float(employee_data.get("PercentSalaryHike", 14))
    if monthly_income < 4500 or salary_hike < 13:
        playbook.append({
            "pillar": "Compensation & Recognition",
            "urgency": "HIGH",
            "issue_detected": "Compensation or annual hike is below competitive industry benchmarks.",
            "action_item": "Initiate off-cycle salary benchmarking review with a targeted 10-15% market adjustment or retention spot bonus.",
            "estimated_budget": round(monthly_income * 0.12 * 6, 2),
            "projected_risk_reduction_pct": 40,
        })
        
    # 3. Career Progression & Promotion Stagnation
    years_since_promo = float(employee_data.get("YearsSinceLastPromotion", 0))
    years_at_company = float(employee_data.get("YearsAtCompany", 1))
    if years_since_promo >= 3 or (years_at_company >= 4 and years_since_promo >= 2):
        playbook.append({
            "pillar": "Career Growth & Mobility",
            "urgency": "MEDIUM",
            "issue_detected": f"Stagnation detected: {int(years_since_promo)} years without formal promotion or role expansion.",
            "action_item": "Establish a defined 6-month promotion pathway with clear technical/leadership milestones and executive sponsor pairing.",
            "estimated_budget": 1200,
            "projected_risk_reduction_pct": 30,
        })
        
    # 4. Long Commute / Remote Flexibility
    distance = float(employee_data.get("DistanceFromHome", 5))
    if distance >= 15 and employee_data.get("Department") != "Remote":
        playbook.append({
            "pillar": "Workplace Flexibility",
            "urgency": "MEDIUM",
            "issue_detected": f"Extensive daily commute ({int(distance)} km) contributing to daily fatigue.",
            "action_item": "Approve a hybrid or remote-first arrangement (2-3 work-from-home days/week) or transit stipend.",
            "estimated_budget": 400,
            "projected_risk_reduction_pct": 20,
        })
        
    # 5. Management & Environment Satisfaction
    job_sat = float(employee_data.get("JobSatisfaction", 3))
    env_sat = float(employee_data.get("EnvironmentSatisfaction", 3))
    years_manager = float(employee_data.get("YearsWithCurrManager", 2))
    if job_sat <= 2 or env_sat <= 2 or years_manager <= 1:
        playbook.append({
            "pillar": "Leadership & Culture",
            "urgency": "HIGH",
            "issue_detected": "Low environmental or job satisfaction with potential manager-employee friction.",
            "action_item": "Schedule an off-the-record skip-level 1-on-1 with Department VP to uncover team friction points; explore internal team transfer if needed.",
            "estimated_budget": 0,
            "projected_risk_reduction_pct": 28,
        })
        
    # Default preventative intervention if no severe flag triggered
    if not playbook:
        playbook.append({
            "pillar": "Talent Development & Retention",
            "urgency": "LOW",
            "issue_detected": "Employee in healthy retention zone; proactive engagement recommended.",
            "action_item": "Provide specialized technical learning budget ($1,500) and invite to high-visibility cross-functional initiatives.",
            "estimated_budget": 1500,
            "projected_risk_reduction_pct": 15,
        })
        
    return playbook
