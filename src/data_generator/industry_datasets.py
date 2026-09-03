import random
import numpy as np
import pandas as pd
from pathlib import Path
from src.core.config import DATASETS_DIR, MASTER_BENCHMARK_PATH, RAW_DATA_PATH

random.seed(42)
np.random.seed(42)

INDUSTRY_PROFILES = [
    {
        "id": "01_tech_software",
        "name": "Technology & Software Engineering",
        "departments": ["Engineering", "Product", "Data Science"],
        "roles": ["Software Engineer", "DevOps Engineer", "Product Manager", "QA Lead", "Tech Lead"],
        "education_fields": ["Computer Science", "Information Technology", "Mathematics"],
        "income_range": (5500, 18000),
        "overtime_prob": 0.40,
        "base_attrition_rate": 0.21,
        "travel": ["Travel_Rarely", "Non-Travel", "Travel_Frequently"],
        "feedback_positive": [
            "Great tech stack and modern microservices.",
            "Love the remote flexibility and autonomous culture.",
            "High engineering standards and fast CI/CD pipelines.",
            "Good equity upside and generous learning budget."
        ],
        "feedback_negative": [
            "On-call rotations are exhausting and burning out the team.",
            "Salary benchmark is below competing FAANG offers.",
            "Endless technical debt and lack of refactoring time.",
            "Lack of clear staff engineer promotion criteria."
        ]
    },
    {
        "id": "02_healthcare_nursing",
        "name": "Healthcare & Clinical Services",
        "departments": ["Inpatient Nursing", "Intensive Care", "Emergency Medicine"],
        "roles": ["Registered Nurse", "ICU Specialist", "Clinical Coordinator", "Nurse Practitioner"],
        "education_fields": ["Nursing", "Medical Science", "Healthcare Admin"],
        "income_range": (4200, 11000),
        "overtime_prob": 0.65,
        "base_attrition_rate": 0.26,
        "travel": ["Non-Travel", "Travel_Rarely"],
        "feedback_positive": [
            "Deeply rewarding patient care and strong peer camaraderie.",
            "Solid hospital benefits and retirement match.",
            "High team trust during critical medical shifts."
        ],
        "feedback_negative": [
            "Understaffed shifts causing severe clinical burnout.",
            "Mandatory weekend overtime is ruining family balance.",
            "Emotional fatigue and unsupportive floor leadership."
        ]
    },
    {
        "id": "03_finance_banking",
        "name": "Finance & Investment Banking",
        "departments": ["Investment Banking", "Wealth Management", "Risk & Quantitative Analysis"],
        "roles": ["Financial Analyst", "Associate Director", "Portfolio Manager", "Risk Auditor"],
        "education_fields": ["Finance", "Economics", "Applied Mathematics", "Business"],
        "income_range": (7500, 24000),
        "overtime_prob": 0.58,
        "base_attrition_rate": 0.18,
        "travel": ["Travel_Frequently", "Travel_Rarely"],
        "feedback_positive": [
            "Exceptional annual performance bonuses and prestige.",
            "Fast-paced deal execution and sharp financial minds.",
            "Unrivaled networking and institutional exposure."
        ],
        "feedback_negative": [
            "80-hour work weeks leaving zero work-life balance.",
            "Hyper-competitive culture and brutal performance ranking.",
            "Strict non-compete clauses and high manager scrutiny."
        ]
    },
    {
        "id": "04_retail_customer_service",
        "name": "Retail & Consumer Operations",
        "departments": ["Store Operations", "Merchandising", "Customer Experience"],
        "roles": ["Store Supervisor", "Merchandiser", "Customer Lead", "Inventory Specialist"],
        "education_fields": ["High School", "Business", "Marketing", "Other"],
        "income_range": (2200, 5200),
        "overtime_prob": 0.35,
        "base_attrition_rate": 0.32,
        "travel": ["Non-Travel"],
        "feedback_positive": [
            "Friendly store team and flexible day shift scheduling.",
            "Good staff discounts and customer interactions.",
            "Clear entry-level progression to shift lead."
        ],
        "feedback_negative": [
            "Hourly wage is barely keeping up with cost of living.",
            "Irregular shift scheduling makes second jobs impossible.",
            "High customer friction with little management backing."
        ]
    },
    {
        "id": "05_consulting_services",
        "name": "Management Consulting & Advisory",
        "departments": ["Strategy", "Operations Consulting", "Digital Transformation"],
        "roles": ["Associate Consultant", "Engagement Manager", "Strategy Principal"],
        "education_fields": ["MBA", "Economics", "Engineering", "Business Administration"],
        "income_range": (7000, 21000),
        "overtime_prob": 0.55,
        "base_attrition_rate": 0.24,
        "travel": ["Travel_Frequently", "Travel_Rarely"],
        "feedback_positive": [
            "Steep learning curve and direct C-suite exposure.",
            "Top-tier training programs and firm brand prestige.",
            "Intellectually stimulating strategic problems."
        ],
        "feedback_negative": [
            "Travel fatigue from Monday-to-Thursday client flights.",
            "Up-or-out promotion model creates toxic pressure.",
            "Unrealistic client deadlines with shifting scopes."
        ]
    },
    {
        "id": "06_sales_enterprise",
        "name": "Enterprise B2B Sales",
        "departments": ["Enterprise Sales", "Commercial Accounts", "Business Development"],
        "roles": ["Account Executive", "Sales Director", "BDR Lead", "Solutions Specialist"],
        "education_fields": ["Marketing", "Business", "Communications"],
        "income_range": (5000, 19000),
        "overtime_prob": 0.42,
        "base_attrition_rate": 0.28,
        "travel": ["Travel_Frequently", "Travel_Rarely"],
        "feedback_positive": [
            "Uncapped commission accelerators on mega deals.",
            "High energy team and great quarterly sales kickoffs.",
            "Strong inbound lead pipeline and marketing support."
        ],
        "feedback_negative": [
            "Unattainable quota increases after territory reshuffles.",
            "Stressful end-of-quarter pressure and pipeline micromanagement.",
            "High base salary disparity among newly hired peers."
        ]
    },
    {
        "id": "07_manufacturing_industrial",
        "name": "Manufacturing & Heavy Industrial",
        "departments": ["Plant Operations", "Quality Assurance", "Plant Maintenance"],
        "roles": ["Assembly Lead", "Plant Engineer", "Safety Inspector", "Operations Supervisor"],
        "education_fields": ["Mechanical Engineering", "Industrial Technology", "Technical Trade"],
        "income_range": (3500, 8500),
        "overtime_prob": 0.48,
        "base_attrition_rate": 0.16,
        "travel": ["Non-Travel", "Travel_Rarely"],
        "feedback_positive": [
            "Predictable shift routines and strong union benefits.",
            "Pride in physical machinery and production output.",
            "High job security for certified technicians."
        ],
        "feedback_negative": [
            "Physical strain and outdated factory ventilation.",
            "Rigid hierarchy with no voice for shop-floor feedback.",
            "Repetitive daily tasks with slow promotion cycles."
        ]
    },
    {
        "id": "08_remote_distributed",
        "name": "Remote-First Distributed Teams",
        "departments": ["Cloud Infrastructure", "Customer Success", "Remote Operations"],
        "roles": ["Remote Systems Admin", "Async Project Manager", "Virtual Support Lead"],
        "education_fields": ["Computer Science", "Information Systems", "Liberal Arts"],
        "income_range": (4800, 14000),
        "overtime_prob": 0.25,
        "base_attrition_rate": 0.19,
        "travel": ["Non-Travel"],
        "feedback_positive": [
            "Zero commute saves two hours every single day.",
            "High autonomy to structure deep work around life.",
            "Modern asynchronous Slack and Notion workflows."
        ],
        "feedback_negative": [
            "Feeling isolated and disconnected from team culture.",
            "Timezone lag creates fragmented communication blocks.",
            "Blurred boundaries between home life and working late."
        ]
    },
    {
        "id": "09_startup_venture",
        "name": "Early-Stage Tech Startups",
        "departments": ["Growth", "Founding Engineering", "Operations"],
        "roles": ["Growth Hacker", "Full Stack Generalist", "Operations Lead"],
        "education_fields": ["Computer Science", "Entrepreneurship", "Design"],
        "income_range": (4000, 12000),
        "overtime_prob": 0.62,
        "base_attrition_rate": 0.30,
        "travel": ["Travel_Rarely", "Non-Travel"],
        "feedback_positive": [
            "Massive ownership and direct impact on company survival.",
            "Fast decision making without corporate red tape.",
            "Significant early equity pool allocation."
        ],
        "feedback_negative": [
            "Constantly shifting roadmap and runway anxiety.",
            "Wearing ten hats leads to chaotic priority whiplash.",
            "Below-market cash salary hoping for an uncertain exit."
        ]
    },
    {
        "id": "10_executive_leadership",
        "name": "Executive & Senior Leadership",
        "departments": ["Corporate Strategy", "General Management", "Finance Leadership"],
        "roles": ["VP of Operations", "Managing Director", "Chief of Staff", "Senior Director"],
        "education_fields": ["MBA", "Doctorate", "Economics", "Law"],
        "income_range": (14000, 32000),
        "overtime_prob": 0.50,
        "base_attrition_rate": 0.12,
        "travel": ["Travel_Frequently", "Travel_Rarely"],
        "feedback_positive": [
            "Direct stewardship of enterprise vision and P&L.",
            "Executive equity vesting and comprehensive bonuses.",
            "High governance authority and top-tier resources."
        ],
        "feedback_negative": [
            "Board friction and misaligned activist investor targets.",
            "Total accountability for macroeconomic headwinds.",
            "Extreme isolation at the top of the organization."
        ]
    },
    {
        "id": "11_education_academia",
        "name": "Higher Education & Research Institutes",
        "departments": ["Academic Faculty", "Institutional Research", "Admissions"],
        "roles": ["Assistant Professor", "Postdoctoral Fellow", "Research Scientist", "Academic Advisor"],
        "education_fields": ["Life Sciences", "Physics", "Humanities", "Education"],
        "income_range": (3600, 9500),
        "overtime_prob": 0.38,
        "base_attrition_rate": 0.17,
        "travel": ["Travel_Rarely", "Non-Travel"],
        "feedback_positive": [
            "Academic freedom and intellectual mentorship.",
            "Generous university campus sabbaticals and tenure perks.",
            "Joy of guiding next-generation researchers."
        ],
        "feedback_negative": [
            "Tenure-track publication pressure and grant rejection stress.",
            "Heavy administrative overhead choking research time.",
            "Stagnant public university compensation scales."
        ]
    },
    {
        "id": "12_logistics_supply_chain",
        "name": "Logistics & Global Supply Chain",
        "departments": ["Fleet Logistics", "Distribution Hubs", "Procurement"],
        "roles": ["Fleet Coordinator", "Supply Planner", "Logistics Specialist", "Warehouse Manager"],
        "education_fields": ["Supply Chain Management", "Business", "Engineering"],
        "income_range": (3800, 9200),
        "overtime_prob": 0.52,
        "base_attrition_rate": 0.23,
        "travel": ["Travel_Rarely", "Travel_Frequently"],
        "feedback_positive": [
            "Dynamic problem solving across global freight lanes.",
            "Clear KPIs and modern tracking technologies.",
            "Solid overtime pay multipliers during peak holiday season."
        ],
        "feedback_negative": [
            "Vendor delays and port backlogs create constant crises.",
            "Night shifts during holiday peaks are grueling.",
            "Little recognition when supply chains run smoothly."
        ]
    },
    {
        "id": "13_call_center_bpo",
        "name": "Customer Support & BPO Centers",
        "departments": ["Tier 2 Support", "Inbound Helpdesk", "Customer Success"],
        "roles": ["Helpdesk Specialist", "Technical Support Lead", "Escalation Agent"],
        "education_fields": ["Communications", "Information Technology", "General Studies"],
        "income_range": (2300, 4800),
        "overtime_prob": 0.35,
        "base_attrition_rate": 0.36,
        "travel": ["Non-Travel"],
        "feedback_positive": [
            "Supportive floor team and structured daily schedules.",
            "Fast feedback loops and helpful knowledge base tools.",
            "Performance bonuses for high customer satisfaction."
        ],
        "feedback_negative": [
            "Strict Average Handle Time (AHT) metrics monitor every minute.",
            "High volume of frustrated customers taking emotional toll.",
            "Very limited salary growth without changing companies."
        ]
    },
    {
        "id": "14_legal_compliance",
        "name": "Legal & Corporate Compliance",
        "departments": ["Corporate Legal", "Regulatory Affairs", "Compliance Audit"],
        "roles": ["Legal Counsel", "Compliance Analyst", "Contracts Officer", "Regulatory Specialist"],
        "education_fields": ["Law", "Public Policy", "Finance"],
        "income_range": (6800, 22000),
        "overtime_prob": 0.45,
        "base_attrition_rate": 0.15,
        "travel": ["Travel_Rarely", "Non-Travel"],
        "feedback_positive": [
            "High intellectual rigor and enterprise protection mandate.",
            "Prestigious counsel positions with solid base pay.",
            "Professional autonomy in legal opinion drafting."
        ],
        "feedback_negative": [
            "Tedious regulatory reporting cycles with strict deadlines.",
            "High liability anxiety and risk aversion overhead.",
            "Uncompromising billable/audit tracking expectations."
        ]
    },
    {
        "id": "15_public_sector_gov",
        "name": "Public Sector & Government Agencies",
        "departments": ["Public Administration", "Civil Services", "Policy Research"],
        "roles": ["Program Analyst", "Policy Coordinator", "Civil Officer", "Grants Manager"],
        "education_fields": ["Public Administration", "Political Science", "Economics"],
        "income_range": (3900, 9800),
        "overtime_prob": 0.15,
        "base_attrition_rate": 0.09,
        "travel": ["Non-Travel", "Travel_Rarely"],
        "feedback_positive": [
            "Unmatched job stability, state pension, and job security.",
            "Predictable 9-to-5 working hours with zero weekend calls.",
            "Mission to serve public citizens and civil welfare."
        ],
        "feedback_negative": [
            "Glacial bureaucratic processes and rigid hierarchy.",
            "Pay grades strictly capped regardless of individual merit.",
            "Outdated technology systems and procurement bottlenecks."
        ]
    },
    {
        "id": "16_creative_media",
        "name": "Creative Agency & Digital Media",
        "departments": ["Content Production", "Brand Design", "Media Strategy"],
        "roles": ["Art Director", "Content Strategist", "Senior Designer", "Copy Lead"],
        "education_fields": ["Visual Arts", "Design", "Communications", "Marketing"],
        "income_range": (3800, 11500),
        "overtime_prob": 0.44,
        "base_attrition_rate": 0.25,
        "travel": ["Travel_Rarely", "Non-Travel"],
        "feedback_positive": [
            "Vibrant, energetic creative peers and dynamic projects.",
            "Freedom to pitch novel aesthetic and storytelling ideas.",
            "Portfolio-building campaigns for global brand clients."
        ],
        "feedback_negative": [
            "Client revisions at 11 PM before morning campaign pitches.",
            "Creative burnout under unrelenting delivery schedules.",
            "Lower retainer margins leading to delayed bonuses."
        ]
    },
    {
        "id": "17_hospitality_tourism",
        "name": "Hospitality & Resort Tourism",
        "departments": ["Guest Relations", "Event Management", "Hotel Operations"],
        "roles": ["Guest Service Lead", "Event Manager", "Food & Beverage Director"],
        "education_fields": ["Hospitality Management", "Business", "Communications"],
        "income_range": (2600, 6800),
        "overtime_prob": 0.46,
        "base_attrition_rate": 0.31,
        "travel": ["Non-Travel", "Travel_Rarely"],
        "feedback_positive": [
            "Dynamic resort setting and diverse international guests.",
            "Discounts across global hotel and restaurant chains.",
            "Lively, customer-facing teamwork and event energy."
        ],
        "feedback_negative": [
            "Working every single weekend and public holiday.",
            "Seasonal off-peak wage downturns create instability.",
            "High physical demands standing for full 10-hour shifts."
        ]
    },
    {
        "id": "18_energy_utilities",
        "name": "Energy, Oil & Renewable Utilities",
        "departments": ["Field Engineering", "Grid Operations", "Renewable Asset Mgmt"],
        "roles": ["Field Operations Tech", "Power Grid Engineer", "Asset Manager", "Safety Officer"],
        "education_fields": ["Electrical Engineering", "Geology", "Mechanical Engineering"],
        "income_range": (5200, 16000),
        "overtime_prob": 0.40,
        "base_attrition_rate": 0.14,
        "travel": ["Travel_Frequently", "Travel_Rarely"],
        "feedback_positive": [
            "Top-tier compensation and generous hazard pay allowances.",
            "Critical infrastructure impact keeping communities powered.",
            "Cutting-edge transition towards renewable wind and solar."
        ],
        "feedback_negative": [
            "Rotational remote field work away from family for weeks.",
            "Rigid regulatory paperwork and severe safety pressure.",
            "Commodity price down-cycles trigger periodic hiring freezes."
        ]
    },
    {
        "id": "19_pharma_biotech",
        "name": "Pharmaceuticals & Biotechnology",
        "departments": ["R&D Discovery", "Regulatory Compliance", "Clinical Operations"],
        "roles": ["Principal Scientist", "Clinical Trial Manager", "Biostatistician", "Toxicologist"],
        "education_fields": ["Biochemistry", "Molecular Biology", "Pharmacy", "Medicine"],
        "income_range": (6200, 19500),
        "overtime_prob": 0.34,
        "base_attrition_rate": 0.15,
        "travel": ["Travel_Rarely", "Travel_Frequently"],
        "feedback_positive": [
            "Transformative science curing life-threatening diseases.",
            "State-of-the-art laboratory instrumentation and grants.",
            "Generous patent royalty incentives and publication prestige."
        ],
        "feedback_negative": [
            "Failed Phase 3 clinical trials wipe out years of effort.",
            "Overly bureaucratic FDA compliance documentation.",
            "Slow corporate consolidation during pharma mergers."
        ]
    },
    {
        "id": "20_hybrid_workforce",
        "name": "Hybrid Corporate Enterprise",
        "departments": ["Human Resources", "Shared Services", "Strategic Planning"],
        "roles": ["HR Business Partner", "Talent Acquisition Lead", "Operations Analyst"],
        "education_fields": ["Human Resources", "Business Admin", "Psychology"],
        "income_range": (4500, 13500),
        "overtime_prob": 0.30,
        "base_attrition_rate": 0.18,
        "travel": ["Travel_Rarely", "Non-Travel"],
        "feedback_positive": [
            "Balanced 3-days in-office / 2-days remote arrangement.",
            "Modern headquarters with collaborative breakout spaces.",
            "Clear enterprise benefits and annual performance reviews."
        ],
        "feedback_negative": [
            "Return-to-office mandates causing commute resentment.",
            "Hybrid parity issues where in-office staff get promoted faster.",
            "Endless Zoom meetings while physically sitting in the office."
        ]
    }
]


def generate_industry_dataset(profile: dict, n_samples: int = 500) -> pd.DataFrame:
    """Generates a realistic synthetic HR dataset tailored to an industry profile."""
    records = []
    
    for i in range(n_samples):
        age = int(np.clip(np.random.normal(37, 8), 21, 62))
        years_at_company = int(np.clip(np.random.exponential(4), 0, min(age - 20, 30)))
        total_working_years = int(np.clip(years_at_company + np.random.exponential(5), years_at_company, age - 18))
        years_in_role = int(min(years_at_company, max(0, int(np.random.normal(years_at_company * 0.6, 2)))))
        years_with_manager = int(min(years_at_company, max(0, int(np.random.normal(years_in_role * 0.8, 1.5)))))
        years_since_promotion = int(min(years_at_company, max(0, int(np.random.exponential(2.5)))))
        
        # Satisfaction levels (1-4 scale)
        job_sat = int(np.clip(np.random.choice([1, 2, 3, 4], p=[0.12, 0.22, 0.42, 0.24]), 1, 4))
        env_sat = int(np.clip(np.random.choice([1, 2, 3, 4], p=[0.14, 0.24, 0.40, 0.22]), 1, 4))
        work_life = int(np.clip(np.random.choice([1, 2, 3, 4], p=[0.16, 0.28, 0.38, 0.18]), 1, 4))
        perf_rating = int(np.random.choice([3, 4], p=[0.84, 0.16]))
        salary_hike = int(np.random.randint(11, 26))
        training_times = int(np.random.randint(1, 6))
        
        # Categoricals
        dept = random.choice(profile["departments"])
        role = random.choice(profile["roles"])
        edu_field = random.choice(profile["education_fields"])
        travel = random.choice(profile["travel"])
        marital = random.choice(["Single", "Married", "Divorced"])
        overtime = "Yes" if random.random() < profile["overtime_prob"] else "No"
        
        # Monthly Income based on profile range and experience
        base_low, base_high = profile["income_range"]
        exp_factor = (total_working_years / 35.0)
        income = int(base_low + (base_high - base_low) * (0.2 + 0.8 * exp_factor) * random.uniform(0.85, 1.2))
        
        distance = int(np.clip(np.random.exponential(8), 1, 45))
        if profile["id"] == "08_remote_distributed":
            distance = 0
            
        education = int(np.clip(np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.15, 0.45, 0.25, 0.10]), 1, 5))
        num_companies = int(np.clip(np.random.poisson(2.5), 0, 9))
        
        # Calculate attrition probability based on key risk factors
        logit = -1.8  # baseline offset
        if overtime == "Yes":
            logit += 0.95
        if job_sat <= 2:
            logit += 0.80 * (3 - job_sat)
        if env_sat <= 2:
            logit += 0.65 * (3 - env_sat)
        if work_life <= 2:
            logit += 0.70 * (3 - work_life)
        if years_since_promotion >= 5:
            logit += 0.55
        if distance > 20:
            logit += 0.40
        if marital == "Single":
            logit += 0.35
        if years_with_manager == 0 and years_at_company > 2:
            logit += 0.45  # manager change friction
            
        prob = 1.0 / (1.0 + np.exp(-logit))
        # Blend with profile base rate
        blended_prob = 0.6 * prob + 0.4 * profile["base_attrition_rate"]
        blended_prob = min(max(blended_prob, 0.04), 0.88)
        
        attrition = "Yes" if random.random() < blended_prob else "No"
        
        feedback = random.choice(profile["feedback_negative"]) if attrition == "Yes" or job_sat <= 2 else random.choice(profile["feedback_positive"])
        
        records.append({
            "Age": age,
            "Attrition": attrition,
            "Department": dept,
            "JobRole": role,
            "EducationField": edu_field,
            "Education": education,
            "DistanceFromHome": distance,
            "EnvironmentSatisfaction": env_sat,
            "JobSatisfaction": job_sat,
            "WorkLifeBalance": work_life,
            "MaritalStatus": marital,
            "MonthlyIncome": income,
            "NumCompaniesWorked": num_companies,
            "OverTime": overtime,
            "BusinessTravel": travel,
            "PerformanceRating": perf_rating,
            "PercentSalaryHike": salary_hike,
            "TrainingTimesLastYear": training_times,
            "YearsAtCompany": years_at_company,
            "TotalWorkingYears": total_working_years,
            "YearsInCurrentRole": years_in_role,
            "YearsSinceLastPromotion": years_since_promotion,
            "YearsWithCurrManager": years_with_manager,
            "Industry": profile["name"],
            "EmployeeFeedback": feedback,
        })
        
    return pd.DataFrame(records)


def generate_all_datasets():
    """Generates all 20 industry datasets and compiles the master benchmark dataset."""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    all_dfs = []
    
    print(f"Generating 20 specialized industry datasets in {DATASETS_DIR}...")
    for idx, profile in enumerate(INDUSTRY_PROFILES, 1):
        df = generate_industry_dataset(profile, n_samples=400)
        file_path = DATASETS_DIR / f"{profile['id']}.csv"
        df.to_csv(file_path, index=False)
        all_dfs.append(df)
        print(f"  [{idx}/20] Saved {profile['name']} ({len(df)} rows) -> {file_path.name}")
        
    # Also integrate existing raw.csv if available
    if RAW_DATA_PATH.exists():
        raw_df = pd.read_csv(RAW_DATA_PATH)
        # Harmonize columns
        if "Industry" not in raw_df.columns:
            raw_df["Industry"] = "IBM HR General Benchmark"
        if "JobRole" not in raw_df.columns:
            raw_df["JobRole"] = "Research Scientist"
        if "OverTime" not in raw_df.columns:
            raw_df["OverTime"] = "No"
        if "BusinessTravel" not in raw_df.columns:
            raw_df["BusinessTravel"] = "Travel_Rarely"
        if "TotalWorkingYears" not in raw_df.columns:
            raw_df["TotalWorkingYears"] = raw_df["YearsAtCompany"] + 3
        if "YearsInCurrentRole" not in raw_df.columns:
            raw_df["YearsInCurrentRole"] = (raw_df["YearsAtCompany"] * 0.7).astype(int)
        if "YearsSinceLastPromotion" not in raw_df.columns:
            raw_df["YearsSinceLastPromotion"] = 1
        if "YearsWithCurrManager" not in raw_df.columns:
            raw_df["YearsWithCurrManager"] = (raw_df["YearsAtCompany"] * 0.6).astype(int)
        if "PerformanceRating" not in raw_df.columns:
            raw_df["PerformanceRating"] = 3
        if "PercentSalaryHike" not in raw_df.columns:
            raw_df["PercentSalaryHike"] = 14
        if "TrainingTimesLastYear" not in raw_df.columns:
            raw_df["TrainingTimesLastYear"] = 3
        if "EmployeeFeedback" not in raw_df.columns:
            raw_df["EmployeeFeedback"] = "Standard corporate work environment."
            
        all_dfs.append(raw_df)
        
    master_df = pd.concat(all_dfs, ignore_index=True)
    master_df.to_csv(MASTER_BENCHMARK_PATH, index=False)
    print(f"\nSuccessfully generated Master Benchmark dataset with {len(master_df)} rows across {len(INDUSTRY_PROFILES) + 1} domains!")
    print(f"Saved to {MASTER_BENCHMARK_PATH}")
    return master_df


if __name__ == "__main__":
    generate_all_datasets()
