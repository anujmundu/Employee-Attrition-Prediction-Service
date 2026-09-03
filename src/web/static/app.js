const app = {
  DEPARTMENT_ROLES_MAP: {
    "Research & Development": [
      "Research Scientist",
      "Laboratory Technician",
      "Manufacturing Director",
      "Healthcare Representative",
      "Research Director",
      "Manager"
    ],
    "Sales": [
      "Sales Executive",
      "Sales Representative",
      "Account Executive",
      "BDR Lead",
      "Solutions Specialist",
      "Manager"
    ],
    "Human Resources": [
      "Human Resources",
      "HR Business Partner",
      "Talent Acquisition Lead",
      "Operations Analyst",
      "Manager"
    ],
    "Engineering": [
      "Software Engineer",
      "DevOps Engineer",
      "Tech Lead",
      "QA Lead",
      "Product Manager"
    ],
    "Healthcare & Clinical": [
      "Registered Nurse",
      "Nurse Practitioner",
      "ICU Specialist",
      "Clinical Coordinator",
      "Therapist"
    ],
    "Finance & Banking": [
      "Financial Analyst",
      "Portfolio Manager",
      "Risk Auditor",
      "Associate Director"
    ],
    "Management Consulting": [
      "Associate Consultant",
      "Engagement Manager",
      "Strategy Principal"
    ],
    "Retail & Operations": [
      "Store Supervisor",
      "Merchandiser",
      "Customer Lead",
      "Inventory Specialist"
    ],
    "Logistics & Supply Chain": [
      "Logistics Specialist",
      "Warehouse Manager",
      "Supply Planner",
      "Fleet Coordinator"
    ],
    "Customer Support & BPO": [
      "Technical Support Lead",
      "Escalation Agent",
      "Helpdesk Specialist",
      "Virtual Support Lead"
    ],
    "Legal & Compliance": [
      "Legal Counsel",
      "Compliance Analyst",
      "Regulatory Specialist",
      "Contracts Officer"
    ],
    "Creative Agency & Media": [
      "Content Strategist",
      "Senior Designer",
      "Art Director",
      "Copy Lead"
    ],
    "Public Sector & Gov": [
      "Policy Coordinator",
      "Program Analyst",
      "Grants Manager",
      "Civil Officer"
    ],
    "Manufacturing & Industrial": [
      "Plant Engineer",
      "Operations Supervisor",
      "Assembly Lead",
      "Safety Inspector"
    ]
  },

  state: {
    kpis: {},
    roster: [],
    datasets: [],
    benchmarks: {},
    currentModalEmployee: null,
    batchRecords: [],
    charts: {},
  },

  async init() {
    console.log("Initializing RetainAI Enterprise UI...");
    const deptSelect = document.getElementById("inp-dept");
    if (deptSelect) {
      deptSelect.addEventListener("change", () => this.handleDepartmentChange());
    }
    this.handleDepartmentChange();
    this.initCharts();
    await this.fetchKPIs();
    await this.fetchRecentRoster();
    await this.fetchDatasets();
    await this.fetchBenchmarks();
  },

  handleDepartmentChange(selectedRole = null) {
    const deptSelect = document.getElementById("inp-dept");
    const roleSelect = document.getElementById("inp-role");
    if (!deptSelect || !roleSelect) return;

    const currentDept = deptSelect.value;
    const roles = this.DEPARTMENT_ROLES_MAP[currentDept] || this.DEPARTMENT_ROLES_MAP["Research & Development"] || [];

    roleSelect.innerHTML = roles.map(role => 
      `<option value="${role}" ${selectedRole === role ? 'selected' : ''}>${role}</option>`
    ).join("");
  },

  switchTab(tabId) {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    
    // Find active button
    const btn = Array.from(document.querySelectorAll(".tab-btn")).find(b => b.getAttribute("onclick").includes(tabId));
    if (btn) btn.classList.add("active");
    
    const target = document.getElementById(`tab-${tabId}`);
    if (target) target.classList.add("active");
    
    if (tabId === "predict-single") {
      this.handleDepartmentChange();
    }
    if (tabId === "mlops") {
      this.fetchDriftStatus();
    }
  },

  async refreshAll() {
    const btn = document.getElementById("refresh-btn");
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> Syncing...`;
    await this.fetchKPIs();
    await this.fetchRecentRoster();
    btn.disabled = false;
    btn.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polyline points="23 4 23 10 17 10"></polyline>
        <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
      </svg> Sync Data`;
  },

  // -------------------------------------------------------------
  // Data Fetchers
  // -------------------------------------------------------------
  async fetchKPIs() {
    try {
      const res = await fetch("/v1/kpis");
      if (!res.ok) return;
      const data = await res.json();
      this.state.kpis = data;

      document.getElementById("kpi-total-analyzed").innerText = (data.total_predictions || 0).toLocaleString();
      document.getElementById("kpi-avg-risk").innerText = `${((data.average_attrition_risk || 0) * 100).toFixed(1)}%`;
      document.getElementById("kpi-high-risk").innerText = (data.high_risk_count || 0).toLocaleString();
      document.getElementById("kpi-loss-at-risk").innerText = `$${Math.round(data.total_loss_at_risk || 0).toLocaleString()}`;
      document.getElementById("kpi-trust-score").innerText = `${(data.average_trust_score || 97.8).toFixed(1)}%`;
    } catch (e) {
      console.warn("Error fetching KPIs:", e);
    }
  },

  async fetchRecentRoster() {
    try {
      const res = await fetch("/v1/recent-predictions?limit=50");
      if (!res.ok) return;
      const rows = await res.json();
      this.state.roster = rows;
      this.renderRoster(rows);
      this.updateChartsWithRoster(rows);
    } catch (e) {
      console.warn("Error fetching roster:", e);
    }
  },

  async fetchDatasets() {
    try {
      // Fetch 20 Online Downloaded Datasets
      const resOnline = await fetch("/v1/online-datasets");
      if (resOnline.ok) {
        const dataOnline = await resOnline.json();
        this.state.onlineDatasets = dataOnline.datasets || [];
      }
      
      // Fetch 20 Industry Benchmarks
      const resIndustry = await fetch("/v1/datasets");
      if (resIndustry.ok) {
        const dataIndustry = await resIndustry.json();
        this.state.industryDatasets = dataIndustry.datasets || [];
      }
      
      this.state.currentDatasetView = "online";
      this.renderCurrentDatasets();
    } catch (e) {
      console.warn("Error fetching datasets:", e);
    }
  },

  setDatasetView(view) {
    this.state.currentDatasetView = view;
    const btnOnline = document.getElementById("btn-show-online");
    const btnIndustry = document.getElementById("btn-show-industry");
    
    if (view === "online") {
      btnOnline.className = "btn btn-primary btn-sm";
      btnIndustry.className = "btn btn-outline btn-sm";
    } else {
      btnOnline.className = "btn btn-outline btn-sm";
      btnIndustry.className = "btn btn-primary btn-sm";
    }
    this.renderCurrentDatasets();
  },

  renderCurrentDatasets() {
    const container = document.getElementById("datasets-grid");
    if (!container) return;

    if (this.state.currentDatasetView === "online") {
      const list = this.state.onlineDatasets || [];
      container.innerHTML = list.map(d => `
        <div class="dataset-card">
          <div>
            <span class="badge badge-low" style="margin-bottom: 0.35rem;">${d.category || 'Real Open Data'}</span>
            <h4 style="font-size: 0.95rem;">${d.name}</h4>
            <div class="dataset-badge-row">
              <span class="badge badge-minimal">${(d.row_count || 0).toLocaleString()} records</span>
              <span class="badge badge-moderate">${d.column_count || 35} features</span>
              <span class="text-muted" style="font-size: 0.7rem;">${d.file_size_kb || 0} KB</span>
            </div>
            <p style="font-size: 0.75rem; color: var(--text-secondary); margin: 0.4rem 0;">
              ${d.description || 'Public employee attrition benchmark.'}
            </p>
            <div style="font-size: 0.7rem; color: var(--accent-cyan); margin-bottom: 0.75rem; word-break: break-all;">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:middle;"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path></svg>
              GitHub: <strong>${d.source_repo || 'open-source'}</strong>
            </div>
          </div>
          <div style="display: flex; gap: 0.4rem;">
            <button class="btn btn-outline btn-sm btn-block" onclick="app.previewOnlineDataset('${d.id}')">
              Preview 10 Rows
            </button>
          </div>
        </div>
      `).join("");
    } else {
      const list = this.state.industryDatasets || [];
      container.innerHTML = list.map(d => `
        <div class="dataset-card">
          <div>
            <h4 style="font-size: 0.95rem;">${d.name}</h4>
            <div class="dataset-badge-row">
              <span class="badge badge-low">${d.row_count} records</span>
              <span class="badge badge-moderate">${Math.round(d.base_attrition_rate * 100)}% base turnover</span>
            </div>
            <div class="dataset-roles">
              <strong>Key Roles:</strong> ${d.roles.slice(0, 3).join(", ")}
            </div>
          </div>
          <div>
            <button class="btn btn-outline btn-sm btn-block" onclick="app.previewDataset('${d.id}')">
              Preview 10 Rows
            </button>
          </div>
        </div>
      `).join("");
    }
  },

  async previewOnlineDataset(datasetId) {
    try {
      const res = await fetch(`/v1/online-datasets/${datasetId}/sample?n=10`);
      if (!res.ok) return;
      const sample = await res.json();

      const previewCard = document.getElementById("dataset-preview-card");
      previewCard.style.display = "block";
      document.getElementById("dataset-preview-title").innerText = `Preview: ${datasetId.replace(/_/g, " ").toUpperCase()} (Downloaded from GitHub)`;

      if (!sample || sample.length === 0) return;
      const headers = Object.keys(sample[0]).slice(0, 8);

      let html = `<table class="data-table"><thead><tr>${headers.map(h => `<th>${h}</th>`).join("")}</tr></thead><tbody>`;
      html += sample.map(row => `<tr>${headers.map(h => `<td>${row[h]}</td>`).join("")}</tr>`).join("");
      html += `</tbody></table>`;

      document.getElementById("dataset-preview-table").innerHTML = html;
      previewCard.scrollIntoView({ behavior: "smooth" });
    } catch (e) {
      console.warn("Error previewing online dataset:", e);
    }
  },

  async previewDataset(datasetId) {
    try {
      const res = await fetch(`/v1/datasets/${datasetId}/sample?n=10`);
      if (!res.ok) return;
      const sample = await res.json();

      const previewCard = document.getElementById("dataset-preview-card");
      previewCard.style.display = "block";
      document.getElementById("dataset-preview-title").innerText = `Preview: ${datasetId.replace(/_/g, " ").toUpperCase()}`;

      if (!sample || sample.length === 0) return;
      const headers = Object.keys(sample[0]).slice(0, 8);

      let html = `<table class="data-table"><thead><tr>${headers.map(h => `<th>${h}</th>`).join("")}</tr></thead><tbody>`;
      html += sample.map(row => `<tr>${headers.map(h => `<td>${row[h]}</td>`).join("")}</tr>`).join("");
      html += `</tbody></table>`;

      document.getElementById("dataset-preview-table").innerHTML = html;
      previewCard.scrollIntoView({ behavior: "smooth" });
    } catch (e) {
      console.warn("Error previewing dataset:", e);
    }
  },

  closeDatasetPreview() {
    document.getElementById("dataset-preview-card").style.display = "none";
  },

  async fetchBenchmarks() {
    try {
      const res = await fetch("/v1/model-benchmarks");
      if (!res.ok) return;
      const data = await res.json();
      this.state.benchmarks = data;
      this.renderBenchmarks(data);
    } catch (e) {
      console.warn("Error fetching benchmarks:", e);
    }
  },

  async fetchDriftStatus() {
    try {
      const res = await fetch("/v1/drift-status");
      if (!res.ok) return;
      const data = await res.json();
      this.renderDriftStatus(data);
    } catch (e) {
      console.warn("Error fetching drift status:", e);
    }
  },

  // -------------------------------------------------------------
  // Roster Renderers
  // -------------------------------------------------------------
  renderRoster(rows) {
    const tbody = document.getElementById("roster-tbody");
    if (!tbody) return;
    if (!rows || rows.length === 0) {
      tbody.innerHTML = `<tr><td colspan="9" class="text-center text-muted" style="padding: 2rem;">No employee predictions recorded yet. Run a single or batch prediction above!</td></tr>`;
      return;
    }

    tbody.innerHTML = rows.map(r => {
      const tierClass = `badge-${(r.risk_tier || "low").toLowerCase()}`;
      const empId = r.employee_id || `EMP-${r.id}`;
      return `
        <tr>
          <td><strong>${empId}</strong></td>
          <td>
            <div>${r.job_role || "Specialist"}</div>
            <span class="text-muted" style="font-size:0.75rem;">${r.department || "Corporate"}</span>
          </td>
          <td>$${Math.round(r.monthly_income || 5000).toLocaleString()}/mo</td>
          <td><strong>${((r.probability || 0) * 100).toFixed(1)}%</strong></td>
          <td><span class="badge ${tierClass}">${r.risk_tier || "LOW"}</span></td>
          <td><span style="color: ${(r.trust_score || 95) >= 80 ? 'var(--accent-emerald)' : 'var(--accent-amber)'}">${r.trust_score || 95}%</span></td>
          <td><span class="text-muted">${r.cluster_id !== undefined ? `Cluster #${r.cluster_id}` : '--'}</span></td>
          <td>$${Math.round(r.expected_loss || 0).toLocaleString()}</td>
          <td>
            <button class="btn btn-outline btn-sm" onclick='app.openEmployeeInspect(${JSON.stringify(r).replace(/'/g, "&apos;")})'>
              Inspect & Retain
            </button>
          </td>
        </tr>
      `;
    }).join("");
  },

  filterRoster() {
    const search = (document.getElementById("roster-search") ? document.getElementById("roster-search").value : "").toLowerCase();
    const tier = document.getElementById("roster-filter-tier") ? document.getElementById("roster-filter-tier").value : "ALL";

    const filtered = (this.state.roster || []).filter(r => {
      const matchSearch = (r.job_role || "").toLowerCase().includes(search) || 
                          (r.department || "").toLowerCase().includes(search) ||
                          (r.employee_id || "").toLowerCase().includes(search);
      const matchTier = tier === "ALL" || r.risk_tier === tier;
      return matchSearch && matchTier;
    });

    this.renderRoster(filtered);
  },

  renderBenchmarks(benchmarks) {
    const tbody = document.getElementById("benchmark-tbody");
    if (!benchmarks || Object.keys(benchmarks).length === 0 || benchmarks.message) {
      tbody.innerHTML = `<tr><td colspan="7" class="text-center text-muted">No model benchmarks recorded yet. Run src/train_all_models.py to evaluate models.</td></tr>`;
      return;
    }

    tbody.innerHTML = Object.entries(benchmarks).map(([name, m]) => `
      <tr>
        <td><strong>${name}</strong></td>
        <td><span class="badge badge-minimal">${m.roc_auc || '--'}</span></td>
        <td>${m.f1_score || '--'}</td>
        <td>${m.accuracy || '--'}</td>
        <td>${m.precision || '--'}</td>
        <td>${m.recall || '--'}</td>
        <td><span class="text-muted">${m.brier_score || '--'}</span></td>
      </tr>
    `).join("");
  },

  renderDriftStatus(report) {
    const pill = document.getElementById("drift-summary-pill");
    const tbody = document.getElementById("drift-features-tbody");

    if (report.drift_detected) {
      pill.className = "status-pill status-alert";
      pill.innerText = `CRITICAL DRIFT ALERT (${report.drifted_features_count} Features Shifted)`;
    } else {
      pill.className = "status-pill status-healthy";
      pill.innerText = `Traffic Healthy (No Critical Drift)`;
    }

    document.getElementById("drift-monitored-text").innerText = `Monitored: ${report.total_features_monitored || 0} features | Composite Score: ${report.composite_drift_score || 0}`;

    if (!report.feature_details || report.feature_details.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" class="text-center text-muted">Awaiting sufficient production inference requests.</td></tr>`;
      return;
    }

    tbody.innerHTML = report.feature_details.map(f => `
      <tr>
        <td><strong>${f.feature}</strong></td>
        <td><span class="badge badge-low">${f.type}</span></td>
        <td>${f.metric}</td>
        <td>${f.score}</td>
        <td>${f.p_value !== null ? f.p_value : 'N/A'}</td>
        <td>
          <span class="badge ${f.drift_detected ? 'badge-critical' : 'badge-minimal'}">
            ${f.drift_detected ? 'DRIFT DETECTED' : 'STABLE'}
          </span>
        </td>
      </tr>
    `).join("");
  },

  // -------------------------------------------------------------
  // Chart.js Visualizations
  // -------------------------------------------------------------
  initCharts() {
    const ctxRisk = document.getElementById("riskTierChart").getContext("2d");
    this.state.charts.riskTier = new Chart(ctxRisk, {
      type: "doughnut",
      data: {
        labels: ["Minimal", "Low", "Moderate", "High", "Critical"],
        datasets: [{
          data: [15, 25, 20, 10, 5],
          backgroundColor: ["#10b981", "#06b6d4", "#f59e0b", "#fb923c", "#f43f5e"],
          borderWidth: 0,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: "bottom", labels: { color: "#9ca3af", boxWidth: 12 } } },
        cutout: "70%",
      }
    });

    const ctxPersona = document.getElementById("personaChart").getContext("2d");
    this.state.charts.persona = new Chart(ctxPersona, {
      type: "bar",
      data: {
        labels: ["Veterans", "Commercial", "At-Risk Juniors", "Balanced Tech"],
        datasets: [{
          label: "Personnel Count",
          data: [12, 18, 14, 22],
          backgroundColor: "#6366f1",
          borderRadius: 6,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: "#9ca3af" }, grid: { display: false } },
          y: { ticks: { color: "#9ca3af" }, grid: { color: "rgba(255,255,255,0.05)" } }
        }
      }
    });

    const ctxDept = document.getElementById("deptChart").getContext("2d");
    this.state.charts.dept = new Chart(ctxDept, {
      type: "bar",
      data: {
        labels: ["Engineering", "R&D", "Sales", "Clinical", "Operations"],
        datasets: [{
          label: "Turnover Loss at Risk ($)",
          data: [180000, 240000, 310000, 150000, 120000],
          backgroundColor: "rgba(244, 63, 94, 0.75)",
          borderRadius: 6,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: "#9ca3af" }, grid: { display: false } },
          y: {
            ticks: {
              color: "#9ca3af",
              callback: (val) => `$${val / 1000}k`
            },
            grid: { color: "rgba(255,255,255,0.05)" }
          }
        }
      }
    });
  },

  updateChartsWithRoster(rows) {
    if (!rows || rows.length === 0) return;

    // Update Tier Counts
    const tiers = { "MINIMAL": 0, "LOW": 0, "MODERATE": 0, "HIGH": 0, "CRITICAL": 0 };
    const personas = { 0: 0, 1: 0, 2: 0, 3: 0 };
    const depts = {};

    rows.forEach(r => {
      if (r.risk_tier && tiers[r.risk_tier] !== undefined) tiers[r.risk_tier]++;
      if (r.cluster_id !== undefined && personas[r.cluster_id] !== undefined) personas[r.cluster_id]++;
      const d = r.department || "General";
      depts[d] = (depts[d] || 0) + (r.expected_loss || 0);
    });

    if (this.state.charts.riskTier) {
      this.state.charts.riskTier.data.datasets[0].data = Object.values(tiers);
      this.state.charts.riskTier.update();
    }

    if (this.state.charts.persona) {
      this.state.charts.persona.data.datasets[0].data = Object.values(personas);
      this.state.charts.persona.update();
    }

    if (this.state.charts.dept) {
      const dLabels = Object.keys(depts);
      const dVals = Object.values(depts);
      if (dLabels.length > 0) {
        this.state.charts.dept.data.labels = dLabels;
        this.state.charts.dept.data.datasets[0].data = dVals;
        this.state.charts.dept.update();
      }
    }
  },

  // -------------------------------------------------------------
  // Single Prediction Form Handler
  // -------------------------------------------------------------
  async handleSinglePredict(event) {
    event.preventDefault();
    const payload = {
      Age: parseInt(document.getElementById("inp-age").value),
      Department: document.getElementById("inp-dept").value,
      JobRole: document.getElementById("inp-role").value,
      MonthlyIncome: parseFloat(document.getElementById("inp-income").value),
      OverTime: document.getElementById("inp-overtime").value,
      DistanceFromHome: parseInt(document.getElementById("inp-distance").value),
      JobSatisfaction: parseInt(document.getElementById("inp-jobsat").value),
      WorkLifeBalance: parseInt(document.getElementById("inp-wlb").value),
      YearsAtCompany: parseInt(document.getElementById("inp-yearscompany").value),
      YearsSinceLastPromotion: parseInt(document.getElementById("inp-yearspromo").value),
      EmployeeFeedback: document.getElementById("inp-feedback").value,
    };

    try {
      const res = await fetch("/v1/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json();
        alert(`Prediction error: ${err.detail || "Server error"}`);
        return;
      }

      const result = await res.json();
      this.renderSingleResult(payload, result);
      await this.fetchKPIs();
      await this.fetchRecentRoster();
    } catch (e) {
      alert(`Network error: ${e.message}`);
    }
  },

  renderSingleResult(input, result) {
    const card = document.getElementById("single-result-card");
    card.style.display = "block";

    const fin = result.financials;
    const tierClass = `badge-${result.risk_tier.toLowerCase()}`;

    card.innerHTML = `
      <div class="card-header-flex">
        <div>
          <h3>Inference Intelligence Output</h3>
          <p class="card-sub">${input.JobRole} | ${input.Department}</p>
        </div>
        <span class="badge ${tierClass}">${result.risk_tier} RISK (${(result.attrition_probability * 100).toFixed(1)}%)</span>
      </div>

      <div class="kpi-grid" style="grid-template-columns: 1fr 1fr; margin-top: 1rem;">
        <div class="kpi-card">
          <div class="kpi-details">
            <span class="kpi-label">Replacement Cost</span>
            <span class="kpi-value" style="font-size: 1.3rem;">$${Math.round(fin.replacement_cost).toLocaleString()}</span>
            <span class="kpi-trend">Turnover impact</span>
          </div>
        </div>
        <div class="kpi-card">
          <div class="kpi-details">
            <span class="kpi-label">Expected Loss at Risk</span>
            <span class="kpi-value" style="font-size: 1.3rem; color: var(--accent-rose);">$${Math.round(fin.expected_loss_at_risk).toLocaleString()}</span>
            <span class="kpi-trend">Probability-weighted</span>
          </div>
        </div>
      </div>

      <div style="margin-top: 1.25rem;">
        <h4 style="font-size: 0.9rem; margin-bottom: 0.5rem;">Data Trust & Anomaly Shield</h4>
        <div style="background: rgba(0,0,0,0.25); padding: 0.85rem; border-radius: 8px; font-size: 0.8rem;">
          <div>Trust Score: <strong>${result.data_trust_score}%</strong> (${result.data_trust_status})</div>
          <div>Autoencoder Reconstruction MSE: <strong>${result.reconstruction_error}</strong></div>
          <div>Persona Archetype: <strong>${result.persona_name}</strong></div>
        </div>
      </div>

      <div style="margin-top: 1.25rem;">
        <h4 style="font-size: 0.9rem; margin-bottom: 0.5rem;">Key Risk Drivers (SHAP Sensitivity)</h4>
        <div style="display: flex; flex-direction: column; gap: 0.4rem;">
          ${(result.explanations.top_risk_drivers || []).map(d => `
            <div style="font-size: 0.8rem; background: rgba(244,63,94,0.1); border-left: 3px solid var(--accent-rose); padding: 0.4rem 0.6rem; border-radius: 4px;">
              ${d.description}
            </div>
          `).join("")}
          ${(!result.explanations.top_risk_drivers || result.explanations.top_risk_drivers.length === 0) ? '<span class="text-muted" style="font-size: 0.8rem;">No high-severity risk drivers detected.</span>' : ''}
        </div>
      </div>

      <div style="margin-top: 1.25rem;">
        <h4 style="font-size: 0.9rem; margin-bottom: 0.5rem;">Prescriptive Retention Actions</h4>
        ${(result.retention_playbook || []).map(p => `
          <div class="playbook-item">
            <div class="playbook-header">
              <span class="badge ${p.urgency === 'HIGH' ? 'badge-critical' : 'badge-low'}">${p.urgency} URGENCY</span>
              <span class="text-muted" style="font-size: 0.75rem;">Budget: ~$${p.estimated_budget}</span>
            </div>
            <div class="playbook-action"><strong>${p.pillar}:</strong> ${p.action_item}</div>
            <div class="playbook-meta">
              <span>Projected Risk Reduction: <strong>-${p.projected_risk_reduction_pct}%</strong></span>
            </div>
          </div>
        `).join("")}
      </div>
    `;
  },

  // -------------------------------------------------------------
  // Employee Inspector Modal & What-If Sandbox
  // -------------------------------------------------------------
  openEmployeeInspect(emp) {
    this.state.currentModalEmployee = emp;
    const modal = document.getElementById("employee-modal");
    modal.style.display = "flex";

    document.getElementById("modal-emp-name").innerText = `${emp.job_role || "Specialist"} (${emp.employee_id || `EMP-${emp.id}`})`;
    document.getElementById("modal-emp-sub").innerText = `${emp.department || "General"} • $${Math.round(emp.monthly_income || 5000).toLocaleString()}/mo`;

    const rawData = emp.data_json ? JSON.parse(emp.data_json) : emp;
    const income = rawData.MonthlyIncome || emp.monthly_income || 5000;
    const overtime = rawData.OverTime || "No";
    const wlb = rawData.WorkLifeBalance || 3;
    const promo = rawData.YearsSinceLastPromotion || 1;

    document.getElementById("modal-content").innerHTML = `
      <div class="kpi-grid" style="grid-template-columns: repeat(3, 1fr);">
        <div class="kpi-card">
          <div class="kpi-details">
            <span class="kpi-label">Attrition Probability</span>
            <span class="kpi-value" id="modal-prob">${((emp.probability || 0) * 100).toFixed(1)}%</span>
            <span class="badge badge-${(emp.risk_tier || 'low').toLowerCase()}" id="modal-tier-badge">${emp.risk_tier || 'LOW'}</span>
          </div>
        </div>
        <div class="kpi-card">
          <div class="kpi-details">
            <span class="kpi-label">Replacement Cost</span>
            <span class="kpi-value">$${Math.round(emp.replacement_cost || 0).toLocaleString()}</span>
            <span class="kpi-trend">Turnover impact</span>
          </div>
        </div>
        <div class="kpi-card">
          <div class="kpi-details">
            <span class="kpi-label">Expected Loss</span>
            <span class="kpi-value" id="modal-loss" style="color: var(--accent-rose);">$${Math.round(emp.expected_loss || 0).toLocaleString()}</span>
            <span class="kpi-trend">Weighted exposure</span>
          </div>
        </div>
      </div>

      <!-- What-If Sandbox Sliders -->
      <div class="whatif-sandbox">
        <h4 style="margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9"></path><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path></svg>
          Interactive What-If Retention Simulator
        </h4>
        <p class="card-sub" style="margin-bottom: 1rem;">Adjust retention levers below to evaluate live probability recalibration and projected cost savings.</p>

        <div class="whatif-slider-group">
          <label>
            <span>Monthly Salary Adjustment: <strong id="lbl-mod-income">$${income}</strong></span>
            <span class="text-muted">Base: $${income}</span>
          </label>
          <input type="range" id="slider-income" min="${Math.round(income * 0.8)}" max="${Math.round(income * 1.5)}" step="100" value="${income}" oninput="document.getElementById('lbl-mod-income').innerText = '$' + this.value; app.runLiveSimulation();" />
        </div>

        <div class="whatif-slider-group">
          <label>
            <span>Overtime Status:</span>
            <select id="select-mod-overtime" onchange="app.runLiveSimulation()">
              <option value="No" ${overtime === 'No' ? 'selected' : ''}>No (Capped 40 hrs/wk)</option>
              <option value="Yes" ${overtime === 'Yes' ? 'selected' : ''}>Yes (Active Overtime)</option>
            </select>
          </label>
        </div>

        <div class="whatif-slider-group">
          <label>
            <span>Work-Life Balance Rating: <strong id="lbl-mod-wlb">${wlb}/4</strong></span>
          </label>
          <input type="range" id="slider-wlb" min="1" max="4" step="1" value="${wlb}" oninput="document.getElementById('lbl-mod-wlb').innerText = this.value + '/4'; app.runLiveSimulation();" />
        </div>

        <!-- Simulation Feedback -->
        <div id="sim-result-box" style="margin-top: 1rem; padding: 0.75rem; background: rgba(0,0,0,0.3); border-radius: 8px; font-size: 0.85rem; display: none;"></div>
      </div>
    `;
  },

  async runLiveSimulation() {
    const emp = this.state.currentModalEmployee;
    if (!emp) return;

    const rawData = emp.data_json ? JSON.parse(emp.data_json) : emp;
    const newIncome = parseFloat(document.getElementById("slider-income").value);
    const newOvertime = document.getElementById("select-mod-overtime").value;
    const newWlb = parseInt(document.getElementById("slider-wlb").value);

    const modifications = {
      MonthlyIncome: newIncome,
      OverTime: newOvertime,
      WorkLifeBalance: newWlb,
    };

    try {
      const res = await fetch("/v1/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          baseline_employee: rawData,
          modifications: modifications,
        })
      });

      if (!res.ok) return;
      const sim = await res.json();

      // Update Modal displays live
      document.getElementById("modal-prob").innerText = `${(sim.simulated_probability * 100).toFixed(1)}%`;
      document.getElementById("modal-loss").innerText = `$${Math.round(sim.simulated_loss_at_risk).toLocaleString()}`;
      
      const badge = document.getElementById("modal-tier-badge");
      badge.className = `badge badge-${sim.simulated_risk_tier.toLowerCase()}`;
      badge.innerText = sim.simulated_risk_tier;

      const simBox = document.getElementById("sim-result-box");
      simBox.style.display = "block";
      const savings = sim.projected_cost_savings;
      const delta = (sim.probability_delta * 100).toFixed(1);

      simBox.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <span>Risk Delta: <strong style="color: ${delta <= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)'}">${delta}%</strong></span>
          <span>Projected Loss Saved: <strong style="color: var(--accent-emerald)">$${Math.round(savings).toLocaleString()}</strong></span>
        </div>
      `;
    } catch (e) {
      console.warn("Simulation error:", e);
    }
  },

  closeModal() {
    document.getElementById("employee-modal").style.display = "none";
  },

  // -------------------------------------------------------------
  // Batch CSV Scoring
  // -------------------------------------------------------------
  async handleBatchUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    const dropzone = document.getElementById("dropzone");
    dropzone.innerHTML = `<h3>Uploading & Scoring ${file.name}...</h3><p class="text-muted">Running multi-model inference, trust shields, and financial modeling...</p>`;

    try {
      const res = await fetch("/v1/batch-predict", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        alert("Failed to process batch CSV.");
        return;
      }

      const data = await res.json();
      this.state.batchRecords = data.records;
      this.renderBatchResults(data);
      await this.fetchKPIs();
      await this.fetchRecentRoster();
    } catch (e) {
      alert(`Upload error: ${e.message}`);
    }
  },

  async loadSampleBatch() {
    try {
      const res = await fetch("/v1/datasets/01_tech_software/sample?n=50");
      if (!res.ok) return;
      const sample = await res.json();

      // Convert to CSV
      const headers = Object.keys(sample[0]);
      let csv = headers.join(",") + "\n";
      csv += sample.map(r => headers.map(h => `"${r[h]}"`).join(",")).join("\n");

      const blob = new Blob([csv], { type: "text/csv" });
      const formData = new FormData();
      formData.append("file", blob, "sample_tech_workforce.csv");

      const res2 = await fetch("/v1/batch-predict", {
        method: "POST",
        body: formData,
      });

      if (!res2.ok) return;
      const data = await res2.json();
      this.state.batchRecords = data.records;
      this.renderBatchResults(data);
      await this.fetchKPIs();
      await this.fetchRecentRoster();
    } catch (e) {
      console.warn("Error loading sample batch:", e);
    }
  },

  renderBatchResults(data) {
    document.getElementById("batch-results-container").style.display = "block";
    document.getElementById("batch-total-count").innerText = data.total_records;
    document.getElementById("batch-avg-risk").innerText = `${(data.average_attrition_risk * 100).toFixed(1)}%`;
    document.getElementById("batch-high-risk-count").innerText = data.high_risk_count;
    document.getElementById("batch-total-loss").innerText = `$${Math.round(data.total_loss_at_risk).toLocaleString()}`;

    const headers = ["EmployeeID", "Department", "JobRole", "MonthlyIncome", "Attrition_Probability", "Risk_Tier", "Data_Trust_Score", "Expected_Loss_At_Risk"];
    const records = data.records.slice(0, 25);

    let html = `<table class="data-table"><thead><tr>${headers.map(h => `<th>${h}</th>`).join("")}</tr></thead><tbody>`;
    html += records.map(r => `
      <tr>
        <td><strong>${r.EmployeeID || "EMP"}</strong></td>
        <td>${r.Department}</td>
        <td>${r.JobRole}</td>
        <td>$${Math.round(r.MonthlyIncome).toLocaleString()}</td>
        <td><strong>${((r.Attrition_Probability || 0) * 100).toFixed(1)}%</strong></td>
        <td><span class="badge badge-${(r.Risk_Tier || 'low').toLowerCase()}">${r.Risk_Tier}</span></td>
        <td>${r.Data_Trust_Score}%</td>
        <td>$${Math.round(r.Expected_Loss_At_Risk || 0).toLocaleString()}</td>
      </tr>
    `).join("");
    html += `</tbody></table>`;

    document.getElementById("batch-table-container").innerHTML = html;
  },

  downloadBatchCSV() {
    if (!this.state.batchRecords || this.state.batchRecords.length === 0) return;
    const headers = Object.keys(this.state.batchRecords[0]);
    let csv = headers.join(",") + "\n";
    csv += this.state.batchRecords.map(r => headers.map(h => `"${r[h]}"`).join(",")).join("\n");

    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `scored_retention_workforce_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }
};

window.addEventListener("DOMContentLoaded", () => {
  app.init();
});
