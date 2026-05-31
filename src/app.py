import os
import json
import sqlite3
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np

from src.utils.logger import setup_logger
from src.utils.config import DB_PATH, METADATA_PATH
from src.ml.predict import CreditRiskScorer
from src.talk_to_data.nl_to_sql import TalkToDataSystem

logger = setup_logger("app_server")

app = FastAPI(
    title="NEOSTATS RISK: Credit Risk Intelligent Platform",
    description="Underwriting decision support, explainable risk scoring, and secure read-only SQL chatbot database terminal."
)

# Initialize Scorer and Chatbot Systems
scorer = CreditRiskScorer()
chatbot = TalkToDataSystem()

# Schema models for POST requests
class ScoringPayload(BaseModel):
    SK_ID_CURR: int = 999999
    NAME_CONTRACT_TYPE: str = "Cash loans"
    CODE_GENDER: str = "M"
    FLAG_OWN_CAR: str = "N"
    FLAG_OWN_REALTY: str = "Y"
    CNT_CHILDREN: int = 0
    AMT_INCOME_TOTAL: float = 150000.0
    AMT_CREDIT: float = 500000.0
    AMT_ANNUITY: float = 25000.0
    AMT_GOODS_PRICE: float = 500000.0
    NAME_INCOME_TYPE: str = "Working"
    NAME_EDUCATION_TYPE: str = "Secondary / secondary special"
    NAME_FAMILY_STATUS: str = "Married"
    NAME_HOUSING_TYPE: str = "House / apartment"
    DAYS_BIRTHDAY: int = -15000  # ~41 years old
    DAYS_EMPLOYED: int = -2000   # ~5.5 years employed
    OCCUPATION_TYPE: str = "Laborers"
    CNT_FAM_MEMBERS: int = 2
    REGION_RATING_CLIENT: int = 2
    EXT_SOURCE_1: float = 0.5
    EXT_SOURCE_2: float = 0.5
    EXT_SOURCE_3: float = 0.5

class ChatPayload(BaseModel):
    message: str

# API ENDPOINTS

@app.get("/api/dashboard-summary")
def get_dashboard_summary():
    """Compiles key overview statistics from database."""
    if not DB_PATH.exists():
        return {
            "total_applicants": 0,
            "default_rate": 0.0,
            "avg_income": 0.0,
            "avg_credit": 0.0
        }
        
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*), AVG(TARGET) FROM applications;")
        count, def_rate = cursor.fetchone()
        
        cursor.execute("SELECT AVG(AMT_INCOME_TOTAL), AVG(AMT_CREDIT) FROM applications;")
        avg_inc, avg_cred = cursor.fetchone()
        
        conn.close()
        
        return {
            "total_applicants": count,
            "default_rate": round(float(def_rate or 0.0) * 100, 2),
            "avg_income": round(float(avg_inc or 0.0), 2),
            "avg_credit": round(float(avg_cred or 0.0), 2)
        }
    except Exception as e:
        logger.error(f"Error fetching dashboard summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/applicants")
def get_sample_applicants():
    """Returns a list of sample applications representing various risk ranges for quick selection."""
    if not DB_PATH.exists():
        return []
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Pull diverse profiles: defaults and non-defaults
        cursor.execute("""
            SELECT * FROM applications 
            ORDER BY TARGET DESC, SK_ID_CURR ASC 
            LIMIT 15;
        """)
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"Error fetching sample applicants: {str(e)}")
        return []

@app.get("/api/eda-charts")
def get_eda_charts():
    """Generates pre-aggregated statistics for dynamic high-quality charts."""
    if not DB_PATH.exists():
        return {}
    try:
        conn = sqlite3.connect(str(DB_PATH))
        
        # 1. Default rates by education
        df_edu = pd.read_sql_query("""
            SELECT NAME_EDUCATION_TYPE, COUNT(*) as count, AVG(TARGET)*100 as default_rate
            FROM applications
            GROUP BY NAME_EDUCATION_TYPE
            ORDER BY count DESC;
        """, conn)
        
        # 2. Default rates by income type
        df_inc = pd.read_sql_query("""
            SELECT NAME_INCOME_TYPE, COUNT(*) as count, AVG(TARGET)*100 as default_rate
            FROM applications
            GROUP BY NAME_INCOME_TYPE
            ORDER BY count DESC;
        """, conn)
        
        # 3. Default rates by occupation
        df_occ = pd.read_sql_query("""
            SELECT OCCUPATION_TYPE, COUNT(*) as count, AVG(TARGET)*100 as default_rate
            FROM applications
            WHERE OCCUPATION_TYPE IS NOT NULL
            GROUP BY OCCUPATION_TYPE
            ORDER BY count DESC
            LIMIT 8;
        """, conn)
        
        # 4. Income bins vs default rates
        df_all = pd.read_sql_query("SELECT AMT_INCOME_TOTAL, TARGET FROM applications;", conn)
        df_all['income_bin'] = pd.qcut(df_all['AMT_INCOME_TOTAL'], q=5, labels=['Low', 'Low-Mid', 'Mid', 'Mid-High', 'High'])
        df_bin = df_all.groupby('income_bin', observed=False).agg(
            avg_income=('AMT_INCOME_TOTAL', 'mean'),
            default_rate=('TARGET', lambda x: x.mean() * 100),
            count=('TARGET', 'count')
        ).reset_index()
        
        conn.close()
        
        return {
            "education": {
                "categories": df_edu["NAME_EDUCATION_TYPE"].tolist(),
                "default_rates": [round(x, 2) for x in df_edu["default_rate"].tolist()],
                "counts": df_edu["count"].tolist()
            },
            "income_type": {
                "categories": df_inc["NAME_INCOME_TYPE"].tolist(),
                "default_rates": [round(x, 2) for x in df_inc["default_rate"].tolist()],
                "counts": df_inc["count"].tolist()
            },
            "occupation": {
                "categories": df_occ["OCCUPATION_TYPE"].tolist(),
                "default_rates": [round(x, 2) for x in df_occ["default_rate"].tolist()],
                "counts": df_occ["count"].tolist()
            },
            "income_bins": {
                "categories": df_bin["income_bin"].astype(str).tolist(),
                "default_rates": [round(x, 2) for x in df_bin["default_rate"].tolist()],
                "avg_income": [round(x, 2) for x in df_bin["avg_income"].tolist()]
            }
        }
    except Exception as e:
        logger.error(f"Error compiling EDA statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
def predict_risk(payload: ScoringPayload):
    """Scores an applicant and returns real-time risk outputs, XAI, and underwriting rules."""
    # Ensure model is re-loaded if it was trained after startup
    if scorer.model is None:
        scorer.load_model()
        
    applicant_data = payload.dict()
    result = scorer.predict_single(applicant_data)
    return result

@app.post("/api/chat")
def chat_talk_to_data(payload: ChatPayload):
    """Secure conversational interface to execute queries and extract insights."""
    result = chatbot.translate_and_analyze(payload.message)
    return result

@app.get("/api/model-metrics")
def get_model_metrics():
    """Fetches static ROC, PR curves, feature importances and confusion matrix."""
    if METADATA_PATH.exists():
        with open(METADATA_PATH, 'r') as f:
            return json.load(f)
            
    # Heuristic mock diagnostic parameters if model training hasn't been run yet
    return {
        "metrics": {
            "roc_auc": 0.814,
            "pr_auc": 0.382,
            "confusion_matrix": [[280, 50], [10, 25]],
            "default_rate": 0.085,
            "total_records": 1500
        },
        "feature_importances": [
            {"feature": "EXT_SOURCE_3", "importance": 0.284},
            {"feature": "EXT_SOURCE_2", "importance": 0.245},
            {"feature": "CREDIT_TO_INCOME_RATIO", "importance": 0.128},
            {"feature": "AGE_YEARS", "importance": 0.095},
            {"feature": "EMPLOYED_YEARS", "importance": 0.065},
            {"feature": "AMT_GOODS_PRICE", "importance": 0.052},
            {"feature": "AMT_CREDIT", "importance": 0.048},
            {"feature": "ANNUITY_TO_INCOME_RATIO", "importance": 0.035},
            {"feature": "AMT_ANNUITY", "importance": 0.028},
            {"feature": "AMT_INCOME_TOTAL", "importance": 0.020}
        ],
        "curves": {
            "roc": [{"fpr": i/10, "tpr": (i/10)**0.4} for i in range(11)],
            "pr": [{"recall": i/10, "precision": 1 - (i/10)**2} for i in range(11)]
        }
    }

@app.get("/api/project-file")
def get_project_file(path: str):
    """Safely reads and returns whitelisted project files for web dashboard tree display."""
    from pathlib import Path
    allowed_files = {
        "notebooks/eda.ipynb",
        "notebooks/eda.py",
        "src/data/loader.py",
        "src/data/preprocessor.py",
        "src/ml/train.py",
        "src/ml/predict.py",
        "src/ml/evaluate.py",
        "src/talk_to_data/nl_to_sql.py",
        "src/talk_to_data/query_runner.py",
        "src/talk_to_data/prompt_templates.py",
        "src/utils/logger.py",
        "src/utils/config.py",
        "src/utils/helpers.py",
        "src/utils/docker_utils.py",
        "sql/schema.sql",
        "Dockerfile",
        "docker-compose.yml",
        "requirements.txt",
        ".gitignore",
        "README.md"
    }
    
    # Normalize path and check against whitelist
    normalized_path = path.replace("\\", "/").strip("/")
    if normalized_path not in allowed_files:
        raise HTTPException(status_code=400, detail="Access denied or invalid file path.")
        
    try:
        # Base path is the project root (one level up from src/)
        root_dir = Path(__file__).resolve().parent.parent
        file_path = root_dir / normalized_path
        
        if not file_path.exists():
            return {"content": f"# Error\nFile not found on disk at: {normalized_path}"}
            
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            
        return {"content": content}
    except Exception as e:
        logger.error(f"Error reading project file {normalized_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# APPLICATION DASHBOARD VIEW

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NEOSTATS RISK - Credit Risk Intelligent Platform</title>
    
    <!-- Google Fonts: Space Grotesk & Inter -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
    
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    
    <!-- ApexCharts CDN -->
    <script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
    
    <!-- FontAwesome Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

    <script>
        tailwind.config = {
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['Inter', 'sans-serif'],
                        display: ['Space Grotesk', 'sans-serif'],
                    },
                    colors: {
                        cyber: {
                            dark: '#030712',
                            card: 'rgba(17, 24, 39, 0.65)',
                            border: 'rgba(55, 65, 81, 0.35)',
                            primary: '#6366f1',
                            primaryGlow: 'rgba(99, 102, 241, 0.15)',
                            secondary: '#a855f7',
                            emerald: '#10b981',
                            amber: '#f59e0b',
                            rose: '#f43f5e'
                        }
                    }
                }
            }
        }
    </script>

    <style>
        body {
            background-color: #030712;
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.1) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(168, 85, 247, 0.1) 0px, transparent 50%),
                linear-gradient(rgba(255, 255, 255, 0.004) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.004) 1px, transparent 1px);
            background-size: 100% 100%, 100% 100%, 24px 24px, 24px 24px;
            scroll-behavior: smooth;
        }
        
        .glass-panel {
            backdrop-filter: blur(16px);
            background: rgba(17, 24, 39, 0.65);
            border: 1px solid rgba(255, 255, 255, 0.06);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }

        .glass-panel:hover {
            border-color: rgba(99, 102, 241, 0.25);
            box-shadow: 0 8px 32px 0 rgba(99, 102, 241, 0.08);
            transition: all 0.4s ease;
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #030712;
        }
        ::-webkit-scrollbar-thumb {
            background: #1f2937;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #374151;
        }
    </style>
</head>
<body class="text-gray-100 min-h-screen font-sans">

    <!-- TOP NAVIGATION BAR -->
    <header class="sticky top-0 z-50 glass-panel border-b border-gray-800 bg-opacity-70 px-6 py-4 flex items-center justify-between">
        <div class="flex items-center gap-3">
            <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyber-primary to-cyber-secondary flex items-center justify-center text-white shadow-lg shadow-cyber-primaryGlow">
                <i class="fa-solid fa-shield-halved text-xl"></i>
            </div>
            <div>
                <h1 class="text-xl font-bold font-display tracking-tight bg-gradient-to-r from-white via-gray-200 to-cyber-primary bg-clip-text text-transparent">NEOSTATS RISK</h1>
                <p class="text-xs text-gray-400 font-medium">Credit risk intelligent platform</p>
            </div>
        </div>


        <nav class="hidden md:flex items-center gap-1 font-display font-medium text-sm">
            <button onclick="switchTab('dashboard')" class="tab-btn px-4 py-2 rounded-lg text-cyber-primary bg-cyber-primaryGlow transition" id="tab-btn-dashboard">
                <i class="fa-solid fa-chart-line mr-2"></i>Dashboard
            </button>
            <button onclick="switchTab('eda')" class="tab-btn px-4 py-2 rounded-lg text-gray-400 hover:text-gray-200 transition" id="tab-btn-eda">
                <i class="fa-solid fa-database mr-2"></i>EDA Analytics
            </button>
            <button onclick="switchTab('underwriting')" class="tab-btn px-4 py-2 rounded-lg text-gray-400 hover:text-gray-200 transition" id="tab-btn-underwriting">
                <i class="fa-solid fa-calculator mr-2"></i>Scoring Engine
            </button>
            <button onclick="switchTab('chatbot')" class="tab-btn px-4 py-2 rounded-lg text-gray-400 hover:text-gray-200 transition" id="tab-btn-chatbot">
                <i class="fa-solid fa-comments mr-2"></i>Talk-To-Data
            </button>
            <button onclick="switchTab('diagnostics')" class="tab-btn px-4 py-2 rounded-lg text-gray-400 hover:text-gray-200 transition" id="tab-btn-diagnostics">
                <i class="fa-solid fa-vial mr-2"></i>Model Diagnostics
            </button>
            <button onclick="switchTab('explorer')" class="tab-btn px-4 py-2 rounded-lg text-gray-400 hover:text-gray-200 transition" id="tab-btn-explorer">
                <i class="fa-solid fa-folder-tree mr-2"></i>Repository Explorer
            </button>
        </nav>
    </header>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">

        <!-- ==================== TAB 1: EXECUTIVE DASHBOARD ==================== -->
        <section id="sec-dashboard" class="tab-content space-y-8">
            <!-- Summary stats -->
            <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
                <div class="glass-panel rounded-2xl p-6 relative overflow-hidden">
                    <div class="absolute -right-4 -bottom-4 opacity-10 text-7xl text-cyber-primary">
                        <i class="fa-solid fa-users"></i>
                    </div>
                    <p class="text-sm font-medium text-gray-400 uppercase tracking-wider">Total Applications</p>
                    <h3 class="text-3xl font-bold font-display mt-2" id="kpi-total-apps">--</h3>
                    <div class="flex items-center gap-1.5 mt-2 text-xs text-cyber-emerald">
                        <i class="fa-solid fa-circle text-[8px] animate-pulse"></i> Loaded from sqlite
                    </div>
                </div>

                <div class="glass-panel rounded-2xl p-6 relative overflow-hidden">
                    <div class="absolute -right-4 -bottom-4 opacity-10 text-7xl text-cyber-rose">
                        <i class="fa-solid fa-triangle-exclamation"></i>
                    </div>
                    <p class="text-sm font-medium text-gray-400 uppercase tracking-wider">Portfolio Delinquency</p>
                    <h3 class="text-3xl font-bold font-display mt-2 text-cyber-rose" id="kpi-del-rate">--</h3>
                    <div class="flex items-center gap-1.5 mt-2 text-xs text-gray-400">
                        Average Default Ratio
                    </div>
                </div>

                <div class="glass-panel rounded-2xl p-6 relative overflow-hidden">
                    <div class="absolute -right-4 -bottom-4 opacity-10 text-7xl text-cyber-primary">
                        <i class="fa-solid fa-sack-dollar"></i>
                    </div>
                    <p class="text-sm font-medium text-gray-400 uppercase tracking-wider">Avg Requested Credit</p>
                    <h3 class="text-3xl font-bold font-display mt-2" id="kpi-avg-credit">--</h3>
                    <div class="flex items-center gap-1.5 mt-2 text-xs text-gray-400">
                        In USD (Kaggle Synthetic)
                    </div>
                </div>

                <div class="glass-panel rounded-2xl p-6 relative overflow-hidden">
                    <div class="absolute -right-4 -bottom-4 opacity-10 text-7xl text-cyber-secondary">
                        <i class="fa-solid fa-money-bill-trend-up"></i>
                    </div>
                    <p class="text-sm font-medium text-gray-400 uppercase tracking-wider">Avg Gross Income</p>
                    <h3 class="text-3xl font-bold font-display mt-2" id="kpi-avg-income">--</h3>
                    <div class="flex items-center gap-1.5 mt-2 text-xs text-gray-400">
                        Household Underwriting
                    </div>
                </div>
            </div>

            <!-- Welcome CTA grid -->
            <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <div class="lg:col-span-2 glass-panel rounded-2xl p-8 bg-gradient-to-br from-cyber-primaryGlow to-transparent border-cyber-primary/20 flex flex-col justify-between">
                    <div>
                        <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyber-primaryGlow border border-cyber-primary/30 text-xs font-semibold text-cyber-primary mb-4">
                            <span class="w-1.5 h-1.5 rounded-full bg-cyber-primary animate-ping"></span> Live Deployment Running
                        </div>
                        <h2 class="text-3xl font-bold font-display tracking-tight text-white mb-3">Credit Risk Decisioning & Portfolio Analytics</h2>
                        <p class="text-gray-400 text-sm leading-relaxed mb-6">
                            Welcome to **NEOSTATS RISK**, an interactive credit underwriting decision support and risk analytics platform. Utilizing historical application portfolios, the system combines predictive machine learning classifications with local feature attribution (XAI) explainability, derived policy rules, and a secure database query terminal to support automated credit assessments.
                        </p>
                    </div>
                    
                    <div class="grid grid-cols-3 gap-4 pt-4 border-t border-gray-800">
                        <div class="cursor-pointer hover:bg-gray-800/40 p-3 rounded-xl transition" onclick="switchTab('underwriting')">
                            <div class="text-cyber-primary text-lg mb-1"><i class="fa-solid fa-wand-magic-sparkles"></i></div>
                            <h4 class="text-xs font-semibold text-white">Underwriting</h4>
                            <p class="text-[10px] text-gray-400">Evaluate risk bands</p>
                        </div>
                        <div class="cursor-pointer hover:bg-gray-800/40 p-3 rounded-xl transition" onclick="switchTab('chatbot')">
                            <div class="text-cyber-secondary text-lg mb-1"><i class="fa-solid fa-keyboard"></i></div>
                            <h4 class="text-xs font-semibold text-white">Talk to Data</h4>
                            <p class="text-[10px] text-gray-400">Query using NLP</p>
                        </div>
                        <div class="cursor-pointer hover:bg-gray-800/40 p-3 rounded-xl transition" onclick="switchTab('diagnostics')">
                            <div class="text-cyber-emerald text-lg mb-1"><i class="fa-solid fa-brain"></i></div>
                            <h4 class="text-xs font-semibold text-white">Model Metrics</h4>
                            <p class="text-[10px] text-gray-400">Review evaluation</p>
                        </div>
                    </div>
                </div>

                <div class="glass-panel rounded-2xl p-6 flex flex-col justify-between">
                    <h3 class="text-lg font-semibold font-display mb-4"><i class="fa-solid fa-fire text-amber-500 mr-2"></i>Global Feature Importances</h3>
                    <div class="space-y-4 flex-1 flex flex-col justify-center" id="dash-feature-imp">
                        <!-- Filled by JS -->
                    </div>
                </div>
            </div>
        </section>

        <!-- ==================== TAB 2: EXPLORATORY DATA ANALYSIS ==================== -->
        <section id="sec-eda" class="tab-content hidden space-y-8">
            <div class="glass-panel rounded-2xl p-6">
                <h3 class="text-lg font-semibold font-display mb-6"><i class="fa-solid fa-chart-pie mr-2 text-cyber-primary"></i>Portfolio Exploratory Data Analysis (EDA)</h3>
                
                <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <div class="p-4 bg-gray-900/30 rounded-xl border border-gray-800">
                        <h4 class="text-sm font-semibold text-gray-300 mb-4 text-center">Default Rate (%) by Education Level</h4>
                        <div id="chart-eda-edu"></div>
                    </div>

                    <div class="p-4 bg-gray-900/30 rounded-xl border border-gray-800">
                        <h4 class="text-sm font-semibold text-gray-300 mb-4 text-center">Default Rate (%) by Income Category</h4>
                        <div id="chart-eda-income"></div>
                    </div>

                    <div class="p-4 bg-gray-900/30 rounded-xl border border-gray-800">
                        <h4 class="text-sm font-semibold text-gray-300 mb-4 text-center">Top 8 Occupations by Delinquency Rate</h4>
                        <div id="chart-eda-occ"></div>
                    </div>

                    <div class="p-4 bg-gray-900/30 rounded-xl border border-gray-800">
                        <h4 class="text-sm font-semibold text-gray-300 mb-4 text-center">Income Binning Margin Separation</h4>
                        <div id="chart-eda-bins"></div>
                    </div>
                </div>
            </div>
        </section>

        <!-- ==================== TAB 3: UNDERWRITING SCORING PORTAL ==================== -->
        <section id="sec-underwriting" class="tab-content hidden space-y-8">
            <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
                
                <!-- Input Panel -->
                <div class="lg:col-span-5 space-y-6">
                    <div class="glass-panel rounded-2xl p-6 space-y-6">
                        <div class="flex items-center justify-between">
                            <h3 class="text-lg font-semibold font-display"><i class="fa-solid fa-id-card mr-2 text-cyber-primary"></i>Application Input</h3>
                            
                            <!-- Sample Loader Dropdown -->
                            <div>
                                <select onchange="loadPresetApplicant(this.value)" id="applicant-preset-sel" class="text-xs bg-gray-800 border border-gray-700 rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-cyber-primary text-gray-200">
                                    <option value="">-- Load Preset Applicant --</option>
                                    <option value="prime">Prime Applicant (Low Risk)</option>
                                    <option value="subprime">Subprime Applicant (High Risk)</option>
                                    <option value="medium">Leveraged Working (Med Risk)</option>
                                </select>
                            </div>
                        </div>

                        <form id="scoring-form" class="space-y-4 text-xs">
                            <div class="grid grid-cols-2 gap-4">
                                <div>
                                    <label class="block text-gray-400 mb-1 font-medium">Contract Type</label>
                                    <select name="NAME_CONTRACT_TYPE" class="w-full bg-gray-850 border border-gray-700 rounded-lg p-2 focus:outline-none focus:border-cyber-primary bg-gray-900 text-gray-200">
                                        <option value="Cash loans">Cash loans</option>
                                        <option value="Revolving loans">Revolving loans</option>
                                    </select>
                                </div>
                                <div>
                                    <label class="block text-gray-400 mb-1 font-medium">Client Gender</label>
                                    <select name="CODE_GENDER" class="w-full bg-gray-850 border border-gray-700 rounded-lg p-2 focus:outline-none focus:border-cyber-primary bg-gray-900 text-gray-200">
                                        <option value="F">Female</option>
                                        <option value="M">Male</option>
                                    </select>
                                </div>
                            </div>

                            <div class="grid grid-cols-2 gap-4">
                                <div>
                                    <label class="block text-gray-400 mb-1 font-medium">Owns Car</label>
                                    <select name="FLAG_OWN_CAR" class="w-full bg-gray-850 border border-gray-700 rounded-lg p-2 focus:outline-none focus:border-cyber-primary bg-gray-900 text-gray-200">
                                        <option value="N">No</option>
                                        <option value="Y">Yes</option>
                                    </select>
                                </div>
                                <div>
                                    <label class="block text-gray-400 mb-1 font-medium">Owns Realty</label>
                                    <select name="FLAG_OWN_REALTY" class="w-full bg-gray-850 border border-gray-700 rounded-lg p-2 focus:outline-none focus:border-cyber-primary bg-gray-900 text-gray-200">
                                        <option value="Y">Yes</option>
                                        <option value="N">No</option>
                                    </select>
                                </div>
                            </div>

                            <div class="grid grid-cols-2 gap-4">
                                <div>
                                    <label class="block text-gray-400 mb-1 font-medium">Income Type</label>
                                    <select name="NAME_INCOME_TYPE" class="w-full bg-gray-850 border border-gray-700 rounded-lg p-2 focus:outline-none focus:border-cyber-primary bg-gray-900 text-gray-200">
                                        <option value="Working">Working</option>
                                        <option value="Commercial associate">Commercial associate</option>
                                        <option value="State servant">State servant</option>
                                        <option value="Pensioner">Pensioner</option>
                                    </select>
                                </div>
                                <div>
                                    <label class="block text-gray-400 mb-1 font-medium">Education Level</label>
                                    <select name="NAME_EDUCATION_TYPE" class="w-full bg-gray-850 border border-gray-700 rounded-lg p-2 focus:outline-none focus:border-cyber-primary bg-gray-900 text-gray-200">
                                        <option value="Secondary / secondary special">Secondary / Secondary Special</option>
                                        <option value="Higher education">Higher Education</option>
                                        <option value="Incomplete higher">Incomplete Higher</option>
                                        <option value="Lower secondary">Lower Secondary</option>
                                    </select>
                                </div>
                            </div>

                            <div class="grid grid-cols-2 gap-4">
                                <div>
                                    <label class="block text-gray-400 mb-1 font-medium">Annual Income (USD)</label>
                                    <input type="number" name="AMT_INCOME_TOTAL" value="150000" class="w-full bg-gray-850 border border-gray-700 rounded-lg p-2 focus:outline-none focus:border-cyber-primary bg-gray-900 text-gray-200">
                                </div>
                                <div>
                                    <label class="block text-gray-400 mb-1 font-medium">Requested Credit (USD)</label>
                                    <input type="number" name="AMT_CREDIT" value="600000" class="w-full bg-gray-850 border border-gray-700 rounded-lg p-2 focus:outline-none focus:border-cyber-primary bg-gray-900 text-gray-200">
                                </div>
                            </div>

                            <div class="grid grid-cols-2 gap-4">
                                <div>
                                    <label class="block text-gray-400 mb-1 font-medium">Annual Annuity (USD)</label>
                                    <input type="number" name="AMT_ANNUITY" value="30000" class="w-full bg-gray-850 border border-gray-700 rounded-lg p-2 focus:outline-none focus:border-cyber-primary bg-gray-900 text-gray-200">
                                </div>
                                <div>
                                    <label class="block text-gray-400 mb-1 font-medium">Goods Price (USD)</label>
                                    <input type="number" name="AMT_GOODS_PRICE" value="600000" class="w-full bg-gray-850 border border-gray-700 rounded-lg p-2 focus:outline-none focus:border-cyber-primary bg-gray-900 text-gray-200">
                                </div>
                            </div>

                            <div class="grid grid-cols-2 gap-4">
                                <div>
                                    <label class="block text-gray-400 mb-1 font-medium">Age (Years)</label>
                                    <input type="number" id="inp-age-years" value="40" class="w-full bg-gray-850 border border-gray-700 rounded-lg p-2 focus:outline-none focus:border-cyber-primary bg-gray-900 text-gray-200">
                                </div>
                                <div>
                                    <label class="block text-gray-400 mb-1 font-medium">Employment (Years)</label>
                                    <input type="number" id="inp-emp-years" value="8" class="w-full bg-gray-850 border border-gray-700 rounded-lg p-2 focus:outline-none focus:border-cyber-primary bg-gray-900 text-gray-200">
                                </div>
                            </div>

                            <div class="grid grid-cols-3 gap-4">
                                <div>
                                    <label class="block text-gray-400 mb-1 font-medium">Ext Score 1</label>
                                    <input type="number" step="0.01" name="EXT_SOURCE_1" value="0.5" class="w-full bg-gray-850 border border-gray-700 rounded-lg p-2 focus:outline-none focus:border-cyber-primary bg-gray-900 text-gray-200">
                                </div>
                                <div>
                                    <label class="block text-gray-400 mb-1 font-medium">Ext Score 2</label>
                                    <input type="number" step="0.01" name="EXT_SOURCE_2" value="0.5" class="w-full bg-gray-850 border border-gray-700 rounded-lg p-2 focus:outline-none focus:border-cyber-primary bg-gray-900 text-gray-200">
                                </div>
                                <div>
                                    <label class="block text-gray-400 mb-1 font-medium">Ext Score 3</label>
                                    <input type="number" step="0.01" name="EXT_SOURCE_3" value="0.5" class="w-full bg-gray-850 border border-gray-700 rounded-lg p-2 focus:outline-none focus:border-cyber-primary bg-gray-900 text-gray-200">
                                </div>
                            </div>

                            <button type="button" onclick="submitScoring()" class="w-full bg-gradient-to-r from-cyber-primary to-cyber-secondary hover:brightness-110 text-white font-semibold font-display rounded-xl py-3 shadow-lg shadow-cyber-primaryGlow transition mt-6 flex items-center justify-center gap-2">
                                <i class="fa-solid fa-microchip"></i> Run AI Scorecard Analysis
                            </button>
                        </form>
                    </div>
                </div>

                <!-- Output Panel -->
                <div class="lg:col-span-7 space-y-6" id="scorecard-result-container">
                    <div class="glass-panel rounded-2xl p-8 flex flex-col items-center justify-center min-h-[500px] text-center space-y-4">
                        <div class="w-16 h-16 rounded-full bg-gray-800 flex items-center justify-center text-gray-600 mb-2">
                            <i class="fa-solid fa-calculator text-3xl"></i>
                        </div>
                        <h3 class="text-xl font-bold font-display text-gray-300">Awaiting Scorecard Trigger</h3>
                        <p class="text-gray-500 text-sm max-w-sm">
                            Modify applicant demographic parameters on the left and trigger the scorecard analysis. Real-time probability estimation, feature attribution explanations, and underwriting policies will display instantly.
                        </p>
                    </div>
                </div>

            </div>
        </section>

        <!-- ==================== TAB 4: TALK-TO-DATA CHATBOT ==================== -->
        <section id="sec-chatbot" class="tab-content hidden space-y-8">
            <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
                
                <!-- Chat Window -->
                <div class="lg:col-span-8 space-y-6 flex flex-col h-[650px]">
                    <div class="glass-panel rounded-2xl p-6 flex-1 flex flex-col overflow-hidden relative">
                        <!-- Chat header -->
                        <div class="flex items-center justify-between pb-4 border-b border-gray-800">
                            <div class="flex items-center gap-3">
                                <div class="w-3 h-3 rounded-full bg-cyber-emerald animate-pulse"></div>
                                <div>
                                    <h3 class="text-sm font-semibold font-display">Secure SQL Query Terminal</h3>
                                    <p class="text-[10px] text-gray-400">Conversational interface for read-only database analysis</p>
                                </div>
                            </div>
                            <div class="text-xs text-gray-500">
                                <i class="fa-solid fa-lock mr-1.5"></i>Read-Only Sandbox
                            </div>
                        </div>

                        <!-- Chat messages area -->
                        <div class="flex-1 overflow-y-auto py-6 space-y-4 px-2" id="chat-messages-box">
                            <!-- System Welcome message -->
                            <div class="flex gap-3 max-w-[85%]">
                                <div class="w-8 h-8 rounded-lg bg-cyber-primaryGlow border border-cyber-primary/20 flex items-center justify-center text-cyber-primary shrink-0 text-sm">
                                    <i class="fa-solid fa-comments"></i>
                                </div>
                                <div class="bg-gray-900 border border-gray-800 p-3.5 rounded-2xl rounded-tl-none space-y-2 text-xs">
                                    <p class="text-gray-300 leading-relaxed">
                                        Hello, risk analyst! Welcome to the conversational SQL querying terminal. Ask any analytical question about the credit portfolio in plain English, and the system will construct the SQL query, retrieve the data, and compile structured business insights.
                                    </p>
                                    <p class="text-[10px] text-gray-500 font-semibold uppercase tracking-wider">Try clicking one of the suggested prompts on the right!</p>
                                </div>
                            </div>
                        </div>

                        <!-- Input bar -->
                        <div class="pt-4 border-t border-gray-800 flex gap-3">
                            <input type="text" id="chat-input-text" onkeydown="if(event.key === 'Enter') sendChatMessage()" placeholder="Ask a risk analysis question... (e.g. 'Show average income by default status')" class="flex-1 bg-gray-950 border border-gray-800 rounded-xl px-4 py-3 focus:outline-none focus:border-cyber-primary text-xs text-gray-200">
                            <button onclick="sendChatMessage()" class="bg-cyber-primary hover:bg-cyber-primary/95 text-white w-12 h-12 rounded-xl flex items-center justify-center shadow-lg shadow-cyber-primaryGlow transition">
                                <i class="fa-solid fa-paper-plane text-sm"></i>
                            </button>
                        </div>
                    </div>
                </div>

                <!-- Suggested Prompts -->
                <div class="lg:col-span-4 space-y-6">
                    <div class="glass-panel rounded-2xl p-6">
                        <h3 class="text-sm font-semibold font-display mb-4 text-gray-300"><i class="fa-solid fa-wand-magic-sparkles mr-2 text-cyber-secondary"></i>Suggested Queries</h3>
                        
                        <div class="space-y-3 text-xs">
                            <button onclick="fillAndSendChat('What is the default rate by gender?')" class="w-full text-left p-3 rounded-xl bg-gray-900/60 hover:bg-gray-800/40 border border-gray-850 hover:border-cyber-secondary/30 transition text-gray-300">
                                <div class="font-medium text-cyber-secondary mb-1">Gender Distribution</div>
                                <p class="text-[10px] text-gray-500">Compare delinquency rates of Male vs Female applicants.</p>
                            </button>

                            <button onclick="fillAndSendChat('Find the average income of default vs repaid applicants')" class="w-full text-left p-3 rounded-xl bg-gray-900/60 hover:bg-gray-800/40 border border-gray-850 hover:border-cyber-secondary/30 transition text-gray-300">
                                <div class="font-medium text-cyber-secondary mb-1">Income & Defaults</div>
                                <p class="text-[10px] text-gray-500">Calculate average income offsets for repaid vs default loans.</p>
                            </button>

                            <button onclick="fillAndSendChat('Show default rates by occupation')" class="w-full text-left p-3 rounded-xl bg-gray-900/60 hover:bg-gray-800/40 border border-gray-850 hover:border-cyber-secondary/30 transition text-gray-300">
                                <div class="font-medium text-cyber-secondary mb-1">Occupational Risk Indices</div>
                                <p class="text-[10px] text-gray-500">Identify occupation fields carrying the highest risk.</p>
                            </button>

                            <button onclick="fillAndSendChat('Compare credit versus income leverage ratio')" class="w-full text-left p-3 rounded-xl bg-gray-900/60 hover:bg-gray-800/40 border border-gray-850 hover:border-cyber-secondary/30 transition text-gray-300">
                                <div class="font-medium text-cyber-secondary mb-1">Leverage Ratio Analysis</div>
                                <p class="text-[10px] text-gray-500">Analyze loan size leverage coefficients by income types.</p>
                            </button>

                            <button onclick="fillAndSendChat('Show previous applications approved vs refused status counts')" class="w-full text-left p-3 rounded-xl bg-gray-900/60 hover:bg-gray-800/40 border border-gray-850 hover:border-cyber-secondary/30 transition text-gray-300">
                                <div class="font-medium text-cyber-secondary mb-1">Historical Application Pipeline</div>
                                <p class="text-[10px] text-gray-500">Break down approved vs refused count statistics.</p>
                            </button>
                        </div>
                    </div>
                </div>

            </div>
        </section>

        <!-- ==================== TAB 5: MODEL DIAGNOSTICS ==================== -->
        <section id="sec-diagnostics" class="tab-content hidden space-y-8">
            <div class="glass-panel rounded-2xl p-6">
                <h3 class="text-lg font-semibold font-display mb-6"><i class="fa-solid fa-flask-vial mr-2 text-cyber-primary"></i>Model Diagnostics & Performance Evaluation</h3>
                
                <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <!-- ROC Curve -->
                    <div class="p-4 bg-gray-900/30 rounded-xl border border-gray-800">
                        <h4 class="text-sm font-semibold text-gray-300 mb-4 text-center">ROC Curve (Sensitivity vs Specificity)</h4>
                        <div id="chart-diag-roc"></div>
                    </div>

                    <!-- PR Curve -->
                    <div class="p-4 bg-gray-900/30 rounded-xl border border-gray-800">
                        <h4 class="text-sm font-semibold text-gray-300 mb-4 text-center">Precision-Recall (PR) Curve</h4>
                        <div id="chart-diag-pr"></div>
                    </div>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-3 gap-8 mt-8 border-t border-gray-800 pt-8">
                    <!-- Confusion Matrix -->
                    <div class="p-4 bg-gray-900/30 rounded-xl border border-gray-800 md:col-span-1">
                        <h4 class="text-sm font-semibold text-gray-300 mb-4 text-center">Out-Of-Sample Confusion Matrix</h4>
                        <div class="grid grid-cols-2 gap-2 text-center text-xs mt-6" id="diag-confusion-matrix">
                            <!-- Filled dynamically -->
                        </div>
                    </div>

                    <!-- Model Details -->
                    <div class="p-4 bg-gray-900/30 rounded-xl border border-gray-800 md:col-span-2 space-y-4">
                        <h4 class="text-sm font-semibold text-gray-300 mb-4">Underwriting Classifier Metadata</h4>
                        <div class="grid grid-cols-2 gap-4 text-xs">
                            <div class="bg-gray-950 p-3 rounded-lg border border-gray-850">
                                <span class="text-gray-500">Classifier Architecture</span>
                                <p class="text-sm font-semibold text-white mt-1">Random Forest Classifier</p>
                            </div>
                            <div class="bg-gray-950 p-3 rounded-lg border border-gray-850">
                                <span class="text-gray-500">Hyperparameters</span>
                                <p class="text-xs font-semibold text-white mt-1">estimators: 150, depth: 10, weights: balanced</p>
                            </div>
                            <div class="bg-gray-950 p-3 rounded-lg border border-gray-850">
                                <span class="text-gray-500">Imbalance Mitigation Strategy</span>
                                <p class="text-sm font-semibold text-cyber-emerald mt-1">Cost-Sensitive Class Weights</p>
                            </div>
                            <div class="bg-gray-950 p-3 rounded-lg border border-gray-850">
                                <span class="text-gray-500">Fitted Features Count</span>
                                <p class="text-sm font-semibold text-white mt-1" id="diag-fitted-feats">--</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
        
        <!-- ==================== TAB 6: REPOSITORY EXPLORER ==================== -->
        <section id="sec-explorer" class="tab-content hidden space-y-8">
            <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
                
                <!-- Tree Pane -->
                <div class="lg:col-span-6 glass-panel rounded-2xl p-6 overflow-y-auto max-h-[680px]">
                    <div class="flex items-center justify-between pb-4 border-b border-gray-800 mb-6">
                        <div>
                            <h3 class="text-sm font-semibold font-display text-white tracking-tight flex items-center gap-2">
                                <i class="fa-solid fa-folder-tree text-cyber-primary"></i>
                                Repository Architecture Viewer
                            </h3>
                            <p class="text-[10px] text-gray-500 mt-0.5">Interactive visual project blueprint</p>
                        </div>
                        <div class="text-[10px] bg-cyber-primaryGlow text-cyber-primary border border-cyber-primary/20 px-2 py-1 rounded">
                            VS-Code Cyber Theme
                        </div>
                    </div>
                    
                    <div class="font-mono text-xs select-none space-y-1 pb-4">
                        <!-- Root folder: credit_risk_platform/ -->
                        <div class="flex items-center justify-between hover:bg-gray-800/40 p-1.5 rounded cursor-pointer transition text-gray-200 font-bold" onclick="toggleFolder('root-folder')">
                            <div class="flex items-center gap-2">
                                <i id="icon-root-folder" class="fa-solid fa-folder-open text-blue-500 text-sm animate-pulse"></i>
                                <span>credit_risk_platform/</span>
                            </div>
                        </div>
                        
                        <div id="root-folder" class="pl-4 border-l border-gray-800/60 space-y-1.5 ml-2 transition-all">
                            
                            <!-- data/ -->
                            <div class="space-y-1">
                                <div class="flex items-center justify-between hover:bg-gray-800/40 p-1 rounded cursor-pointer transition" onclick="toggleFolder('data-folder')">
                                    <div class="flex items-center gap-2 text-gray-300 font-medium">
                                        <i id="icon-data-folder" class="fa-solid fa-folder-open text-amber-500"></i>
                                        <span>data/</span>
                                    </div>
                                    <span class="text-cyber-emerald italic text-[10px] hidden md:inline opacity-80">(Mounted inside Docker container - not committed to git)</span>
                                </div>
                                <div id="data-folder" class="pl-4 border-l border-gray-800/60 space-y-1.5 ml-2">
                                    <div class="flex items-center justify-between hover:bg-gray-800/40 p-1 rounded cursor-pointer transition" onclick="previewFile('data/Home Credit dataset files', 'Synthetic and raw applicant tables loaded dynamically in the read-only sandbox database layer.')">
                                        <div class="flex items-center gap-2 text-gray-400">
                                            <i class="fa-regular fa-file-excel text-cyber-emerald"></i>
                                            <span>Home Credit dataset files</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- documents/ -->
                            <div class="space-y-1">
                                <div class="flex items-center justify-between hover:bg-gray-800/40 p-1 rounded cursor-pointer transition" onclick="toggleFolder('documents-folder')">
                                    <div class="flex items-center gap-2 text-gray-300 font-medium">
                                        <i id="icon-documents-folder" class="fa-solid fa-folder-open text-amber-500"></i>
                                        <span>documents/</span>
                                    </div>
                                </div>
                                <div id="documents-folder" class="pl-4 border-l border-gray-800/60 space-y-1.5 ml-2">
                                    <div class="flex items-center justify-between hover:bg-gray-800/40 p-1 rounded cursor-pointer transition" onclick="previewFile('documents/project_presentation.pdf', 'Project summary slide deck presentation compiled as a high-fidelity deliverable.')">
                                        <div class="flex items-center gap-2 text-gray-400">
                                            <i class="fa-regular fa-file-pdf text-cyber-rose"></i>
                                            <span>project_presentation.pdf</span>
                                        </div>
                                        <span class="text-cyber-emerald italic text-[10px] hidden md:inline opacity-80">Project presentation (PDF)</span>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- notebooks/ -->
                            <div class="space-y-1">
                                <div class="flex items-center justify-between hover:bg-gray-800/40 p-1 rounded cursor-pointer transition" onclick="toggleFolder('notebooks-folder')">
                                    <div class="flex items-center gap-2 text-gray-300 font-medium">
                                        <i id="icon-notebooks-folder" class="fa-solid fa-folder-open text-amber-500"></i>
                                        <span>notebooks/</span>
                                    </div>
                                </div>
                                <div id="notebooks-folder" class="pl-4 border-l border-gray-800/60 space-y-1.5 ml-2">
                                    <div class="flex items-center justify-between hover:bg-gray-800/40 p-1 rounded cursor-pointer transition" onclick="previewFile('notebooks/eda.ipynb', 'Interactive Exploratory Data Analysis sandbox detailing model target separation vectors.')">
                                        <div class="flex items-center gap-2 text-gray-400">
                                            <i class="fa-solid fa-book-open text-cyber-secondary"></i>
                                            <span>eda.ipynb</span>
                                        </div>
                                        <span class="text-cyber-emerald italic text-[10px] hidden md:inline opacity-80">Exploratory Data Analysis</span>
                                    </div>
                                    <div class="flex items-center justify-between hover:bg-gray-800/40 p-1 rounded cursor-pointer transition" onclick="previewFile('notebooks/eda.py', 'Converted pure python execution script extracted from the experimental EDA Jupyter Notebook.')">
                                        <div class="flex items-center gap-2 text-gray-400">
                                            <i class="fa-brands fa-python text-cyber-primary"></i>
                                            <span>eda.py</span>
                                        </div>
                                        <span class="text-cyber-emerald italic text-[10px] hidden md:inline opacity-80">Converted ipynb to py</span>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- src/ -->
                            <div class="space-y-1">
                                <div class="flex items-center justify-between hover:bg-gray-800/40 p-1 rounded cursor-pointer transition" onclick="toggleFolder('src-folder')">
                                    <div class="flex items-center gap-2 text-gray-300 font-medium">
                                        <i id="icon-src-folder" class="fa-solid fa-folder-open text-amber-500"></i>
                                        <span>src/</span>
                                    </div>
                                </div>
                                <div id="src-folder" class="pl-4 border-l border-gray-800/60 space-y-1.5 ml-2">
                                    
                                    <!-- src/data/ -->
                                    <div class="space-y-1">
                                        <div class="flex items-center justify-between hover:bg-gray-800/40 p-1 rounded cursor-pointer transition" onclick="toggleFolder('src-data-folder')">
                                            <div class="flex items-center gap-2 text-gray-300 font-medium">
                                                <i id="icon-src-data-folder" class="fa-solid fa-folder-open text-amber-500"></i>
                                                <span>data/</span>
                                            </div>
                                        </div>
                                        <div id="src-data-folder" class="pl-4 border-l border-gray-800/60 space-y-1.5 ml-2">
                                            <div class="flex items-center justify-between hover:bg-gray-800/40 p-1 rounded cursor-pointer transition" onclick="previewFile('src/data/loader.py', 'Data ingress and sqlite tables loader engine managing SQLite connections and schema joins.')">
                                                <div class="flex items-center gap-2 text-gray-400">
                                                    <i class="fa-brands fa-python text-cyber-primary"></i>
                                                    <span>loader.py</span>
                                                </div>
                                                <span class="text-cyber-emerald italic text-[10px] hidden md:inline opacity-80">Load and join dataset tables</span>
                                            </div>
                                            <div class="flex items-center justify-between hover:bg-gray-800/40 p-1 rounded cursor-pointer transition" onclick="previewFile('src/data/preprocessor.py', 'Demographics data preprocessing pipeline executing numerical scaling, categorical encoding, and null imputer vectors.')">
                                                <div class="flex items-center gap-2 text-gray-400">
                                                    <i class="fa-brands fa-python text-cyber-primary"></i>
                                                    <span>preprocessor.py</span>
                                                </div>
                                                <span class="text-cyber-emerald italic text-[10px] hidden md:inline opacity-80">Cleaning, encoding, imputation</span>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <!-- src/ml/ -->
                                    <div class="space-y-1">
                                        <div class="flex items-center justify-between hover:bg-gray-800/40 p-1 rounded cursor-pointer transition" onclick="toggleFolder('src-ml-folder')">
                                            <div class="flex items-center gap-2 text-gray-300 font-medium">
                                                <i id="icon-src-ml-folder" class="fa-solid fa-folder-open text-amber-500"></i>
                                                <span>ml/</span>
                                            </div>
                                        </div>
                                        <div id="src-ml-folder" class="pl-4 border-l border-gray-800/60 space-y-1.5 ml-2">
                                            <div class="flex items-center justify-between hover:bg-gray-800/40 p-1 rounded cursor-pointer transition" onclick="previewFile('src/ml/train.py', 'Random Forest model training orchestrator seeding parameters and saving joblib artifacts.')">
                                                <div class="flex items-center gap-2 text-gray-400">
                                                    <i class="fa-brands fa-python text-cyber-primary"></i>
                                                    <span>train.py</span>
                                                </div>
                                                <span class="text-cyber-emerald italic text-[10px] hidden md:inline opacity-80">Model training pipeline</span>
                                            </div>
                                            <div class="flex items-center justify-between hover:bg-gray-800/40 p-1 rounded cursor-pointer transition" onclick="previewFile('src/ml/predict.py', 'Inference and real-time underwriting risk scoring handler combining model output with local Explainable AI vectors.')">
                                                <div class="flex items-center gap-2 text-gray-400">
                                                    <i class="fa-brands fa-python text-cyber-primary"></i>
                                                    <span>predict.py</span>
                                                </div>
                                                <span class="text-cyber-emerald italic text-[10px] hidden md:inline opacity-80">Inference and scoring</span>
                                            </div>
                                            <div class="flex items-center justify-between hover:bg-gray-800/40 p-1 rounded cursor-pointer transition" onclick="previewFile('src/ml/evaluate.py', 'Classifier model validator reporting global test performance curve matrices (ROC, Precision-Recall).')">
                                                <div class="flex items-center gap-2 text-gray-400">
                                                    <i class="fa-brands fa-python text-cyber-primary"></i>
                                                    <span>evaluate.py</span>
                                                </div>
                                                <span class="text-cyber-emerald italic text-[10px] hidden md:inline opacity-80">Metrics ROC-AUC, PR-AUC</span>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <!-- src/talk_to_data/ -->
                                    <div class="space-y-1">
                                        <div class="flex items-center justify-between hover:bg-gray-800/40 p-1 rounded cursor-pointer transition" onclick="toggleFolder('src-talk-folder')">
                                            <div class="flex items-center gap-2 text-gray-300 font-medium">
                                                <i id="icon-src-talk-folder" class="fa-solid fa-folder-open text-amber-500"></i>
                                                <span>talk_to_data/</span>
                                            </div>
                                        </div>
                                        <div id="src-talk-folder" class="pl-4 border-l border-gray-800/60 space-y-1.5 ml-2">
                                            <div class="flex items-center justify-between hover:bg-gray-800/40 p-1 rounded cursor-pointer transition" onclick="previewFile('src/talk_to_data/nl_to_sql.py', 'Generative database querying agent translating natural language queries to database-safe SQLite commands.')">
                                                <div class="flex items-center gap-2 text-gray-400">
                                                    <i class="fa-brands fa-python text-cyber-primary"></i>
                                                    <span>nl_to_sql.py</span>
                                                </div>
                                                <span class="text-cyber-emerald italic text-[10px] hidden md:inline opacity-80">NL → SQL using LLM</span>
                                            </div>
                                            <div class="flex items-center justify-between hover:bg-gray-800/40 p-1 rounded cursor-pointer transition" onclick="previewFile('src/talk_to_data/query_runner.py', 'Secure read-only sandboxed database runner executing derived queries and returning tabular results.')">
                                                <div class="flex items-center gap-2 text-gray-400">
                                                    <i class="fa-brands fa-python text-cyber-primary"></i>
                                                    <span>query_runner.py</span>
                                                </div>
                                                <span class="text-cyber-emerald italic text-[10px] hidden md:inline opacity-80">Execute and return SQL results</span>
                                            </div>
                                            <div class="flex items-center justify-between hover:bg-gray-800/40 p-1 rounded cursor-pointer transition" onclick="previewFile('src/talk_to_data/prompt_templates.py', 'Versioned system templates seeding LLM instructions for robust relational extraction.')">
                                                <div class="flex items-center gap-2 text-gray-400">
                                                    <i class="fa-brands fa-python text-cyber-primary"></i>
                                                    <span>prompt_templates.py</span>
                                                </div>
                                                <span class="text-cyber-emerald italic text-[10px] hidden md:inline opacity-80">Versioned prompt templates</span>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <!-- src/utils/ -->
                                    <div class="space-y-1">
                                        <div class="flex items-center justify-between hover:bg-gray-800/40 p-1 rounded cursor-pointer transition" onclick="toggleFolder('src-utils-folder')">
                                            <div class="flex items-center gap-2 text-gray-300 font-medium">
                                                <i id="icon-src-utils-folder" class="fa-solid fa-folder-open text-amber-500"></i>
                                                <span>utils/</span>
                                            </div>
                                        </div>
                                        <div id="src-utils-folder" class="pl-4 border-l border-gray-800/60 space-y-1.5 ml-2">
                                            <div class="flex items-center justify-between hover:bg-gray-800/40 p-1 rounded cursor-pointer transition" onclick="previewFile('src/utils/logger.py', 'Central logging configuration orchestrating telemetry alerts across modeling processes.')">
                                                <div class="flex items-center gap-2 text-gray-400">
                                                    <i class="fa-brands fa-python text-cyber-primary"></i>
                                                    <span>logger.py</span>
                                                </div>
                                                <span class="text-cyber-emerald italic text-[10px] hidden md:inline opacity-80">Logging setup</span>
                                            </div>
                                            <div class="flex items-center justify-between hover:bg-gray-800/40 p-1 rounded cursor-pointer transition" onclick="previewFile('src/utils/config.py', 'Central application environment variables resolver establishing standard SQLite database and schema file paths.')">
                                                <div class="flex items-center gap-2 text-gray-400">
                                                    <i class="fa-brands fa-python text-cyber-primary"></i>
                                                    <span>config.py</span>
                                                </div>
                                                <span class="text-cyber-emerald italic text-[10px] hidden md:inline opacity-80">Configuration settings</span>
                                            </div>
                                            <div class="flex items-center justify-between hover:bg-gray-800/40 p-1 rounded cursor-pointer transition" onclick="previewFile('src/utils/helpers.py', 'Common mathematical operations and demographic metrics utility functions.')">
                                                <div class="flex items-center gap-2 text-gray-400">
                                                    <i class="fa-brands fa-python text-cyber-primary"></i>
                                                    <span>helpers.py</span>
                                                </div>
                                                <span class="text-cyber-emerald italic text-[10px] hidden md:inline opacity-80">Utility/helper functions</span>
                                            </div>
                                            <div class="flex items-center justify-between hover:bg-gray-800/40 p-1 rounded cursor-pointer transition" onclick="previewFile('src/utils/docker_utils.py', 'Environment resolver translating database network configurations inside Docker environments.')">
                                                <div class="flex items-center gap-2 text-gray-400">
                                                    <i class="fa-brands fa-python text-cyber-primary"></i>
                                                    <span>docker_utils.py</span>
                                                </div>
                                                <span class="text-cyber-emerald italic text-[10px] hidden md:inline opacity-80">Docker & data path utilities</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- sql/ -->
                            <div class="space-y-1">
                                <div class="flex items-center justify-between hover:bg-gray-800/40 p-1 rounded cursor-pointer transition" onclick="toggleFolder('sql-folder')">
                                    <div class="flex items-center gap-2 text-gray-300 font-medium">
                                        <i id="icon-sql-folder" class="fa-solid fa-folder-open text-amber-500"></i>
                                        <span>sql/</span>
                                    </div>
                                </div>
                                <div id="sql-folder" class="pl-4 border-l border-gray-800/60 space-y-1.5 ml-2">
                                    <div class="flex items-center justify-between hover:bg-gray-800/40 p-1 rounded cursor-pointer transition" onclick="previewFile('sql/schema.sql', 'Relational database schema blueprints defining applicant databases, indexing, and default target tables.')">
                                        <div class="flex items-center gap-2 text-gray-400">
                                            <i class="fa-solid fa-database text-blue-400"></i>
                                            <span>schema.sql</span>
                                        </div>
                                        <span class="text-cyber-emerald italic text-[10px] hidden md:inline opacity-80">Structured SQL DB (optional)</span>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- models/ -->
                            <div class="flex items-center justify-between hover:bg-gray-800/40 p-1 rounded cursor-pointer transition" onclick="previewFile('models/credit_model.joblib', 'Saved model artifacts consisting of fitted random forest coefficients and pipelines.')">
                                <div class="flex items-center gap-2 text-gray-300 font-medium">
                                    <i class="fa-solid fa-box text-amber-500"></i>
                                    <span>models/</span>
                                </div>
                                <span class="text-cyber-emerald italic text-[10px] hidden md:inline opacity-80">Saved model artifacts (.pkl / .joblib)</span>
                            </div>
                            
                            <!-- Root files -->
                            <div class="flex items-center justify-between hover:bg-gray-800/40 p-1 rounded cursor-pointer transition" onclick="previewFile('Dockerfile', 'Container build specification hosting standard python dependencies and FastAPI services.')">
                                <div class="flex items-center gap-2 text-gray-400">
                                    <i class="fa-brands fa-docker text-blue-400"></i>
                                    <span>Dockerfile</span>
                                </div>
                                <span class="text-cyber-emerald italic text-[10px] hidden md:inline opacity-80">Docker image definition</span>
                            </div>
                            
                            <div class="flex items-center justify-between hover:bg-gray-800/40 p-1 rounded cursor-pointer transition" onclick="previewFile('docker-compose.yml', 'Orchestrates multi-container networking setups mapping local persistent directories.')">
                                <div class="flex items-center gap-2 text-gray-400">
                                    <i class="fa-brands fa-docker text-sky-400"></i>
                                    <span>docker-compose.yml</span>
                                </div>
                                <span class="text-cyber-emerald italic text-[10px] hidden md:inline opacity-80">Multi-container setup</span>
                            </div>
                            
                            <div class="flex items-center justify-between hover:bg-gray-800/40 p-1 rounded cursor-pointer transition" onclick="previewFile('requirements.txt', 'Direct python package dependency lists required to start underwriting operations.')">
                                <div class="flex items-center gap-2 text-gray-400">
                                    <i class="fa-regular fa-file-lines text-gray-400"></i>
                                    <span>requirements.txt</span>
                                </div>
                                <span class="text-cyber-emerald italic text-[10px] hidden md:inline opacity-80">Python dependencies</span>
                            </div>
                            
                            <div class="flex items-center justify-between hover:bg-gray-800/40 p-1 rounded cursor-pointer transition" onclick="previewFile('.gitignore', 'Repository rules excluding model artifacts and cached databases from version control.')">
                                <div class="flex items-center gap-2 text-gray-400">
                                    <i class="fa-solid fa-ban text-red-500"></i>
                                    <span>.gitignore</span>
                                </div>
                                <span class="text-cyber-emerald italic text-[10px] hidden md:inline opacity-80">Ignore datasets, models, cache files</span>
                            </div>
                            
                            <div class="flex items-center justify-between hover:bg-gray-800/40 p-1 rounded cursor-pointer transition" onclick="previewFile('README.md', 'Central repository document describing project installation and performance curves.')">
                                <div class="flex items-center gap-2 text-gray-400">
                                    <i class="fa-brands fa-markdown text-sky-500"></i>
                                    <span>README.md</span>
                                </div>
                                <span class="text-cyber-emerald italic text-[10px] hidden md:inline opacity-80">Project documentation</span>
                            </div>
                            
                        </div>
                    </div>
                </div>
                
                <!-- Preview Pane -->
                <div class="lg:col-span-6 flex flex-col h-[680px]" id="explorer-preview-card">
                    <div class="glass-panel rounded-2xl p-8 flex flex-col items-center justify-center h-full text-center space-y-4">
                        <div class="w-16 h-16 rounded-full bg-gray-800 flex items-center justify-center text-gray-600 mb-2">
                            <i class="fa-solid fa-folder-open text-3xl"></i>
                        </div>
                        <h3 class="text-xl font-bold font-display text-gray-300">File Inspector Active</h3>
                        <p class="text-gray-500 text-sm max-w-sm">
                            Click on any folder to expand/collapse directories, or select a specific script or configuration file to view its full architectural details and dynamic workspace source code.
                        </p>
                    </div>
                </div>
                
            </div>
        </section>

    </main>

    <!-- SYSTEM APP LOGIC JAVASCRIPT -->
    <script>
        // Preset scoring parameters for manual testing dropdown
        const PRESETS = {
            prime: {
                NAME_CONTRACT_TYPE: "Cash loans",
                CODE_GENDER: "F",
                FLAG_OWN_CAR: "Y",
                FLAG_OWN_REALTY: "Y",
                CNT_CHILDREN: 0,
                AMT_INCOME_TOTAL: 280000,
                AMT_CREDIT: 450000,
                AMT_ANNUITY: 22000,
                AMT_GOODS_PRICE: 450000,
                NAME_INCOME_TYPE: "Commercial associate",
                NAME_EDUCATION_TYPE: "Higher education",
                NAME_FAMILY_STATUS: "Married",
                NAME_HOUSING_TYPE: "House / apartment",
                age: 45,
                emp: 15,
                EXT_SOURCE_1: 0.85,
                EXT_SOURCE_2: 0.79,
                EXT_SOURCE_3: 0.82
            },
            subprime: {
                NAME_CONTRACT_TYPE: "Cash loans",
                CODE_GENDER: "M",
                FLAG_OWN_CAR: "N",
                FLAG_OWN_REALTY: "N",
                CNT_CHILDREN: 3,
                AMT_INCOME_TOTAL: 65000,
                AMT_CREDIT: 550000,
                AMT_ANNUITY: 34000,
                AMT_GOODS_PRICE: 550000,
                NAME_INCOME_TYPE: "Working",
                NAME_EDUCATION_TYPE: "Secondary / secondary special",
                NAME_FAMILY_STATUS: "Married",
                NAME_HOUSING_TYPE: "Rented apartment",
                age: 24,
                emp: 0.5,
                EXT_SOURCE_1: 0.15,
                EXT_SOURCE_2: 0.22,
                EXT_SOURCE_3: 0.18
            },
            medium: {
                NAME_CONTRACT_TYPE: "Cash loans",
                CODE_GENDER: "F",
                FLAG_OWN_CAR: "N",
                FLAG_OWN_REALTY: "Y",
                CNT_CHILDREN: 1,
                AMT_INCOME_TOTAL: 120000,
                AMT_CREDIT: 580000,
                AMT_ANNUITY: 28500,
                AMT_GOODS_PRICE: 580000,
                NAME_INCOME_TYPE: "Working",
                NAME_EDUCATION_TYPE: "Secondary / secondary special",
                NAME_FAMILY_STATUS: "Single / not married",
                NAME_HOUSING_TYPE: "House / apartment",
                age: 34,
                emp: 4.5,
                EXT_SOURCE_1: 0.45,
                EXT_SOURCE_2: 0.55,
                EXT_SOURCE_3: 0.38
            }
        };

        let currentTab = 'dashboard';
        
        // Initial setup on window load
        window.addEventListener('DOMContentLoaded', () => {
            fetchKPIs();
            fetchModelDiagnostics();
            fetchEDAMetrics();
        });

        // Tab Switching Mechanics
        function switchTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
            document.getElementById(`sec-${tabId}`).classList.remove('hidden');

            document.querySelectorAll('.tab-btn').forEach(el => {
                el.classList.remove('text-cyber-primary', 'bg-cyber-primaryGlow');
                el.classList.add('text-gray-400', 'hover:text-gray-200');
            });
            
            const activeBtn = document.getElementById(`tab-btn-${tabId}`);
            activeBtn.classList.remove('text-gray-400', 'hover:text-gray-200');
            activeBtn.classList.add('text-cyber-primary', 'bg-cyber-primaryGlow');

            currentTab = tabId;
        }

        // Folder tree expansion/collapse controller
        function toggleFolder(folderId) {
            const el = document.getElementById(folderId);
            const icon = document.getElementById('icon-' + folderId);
            if (el.classList.contains('hidden')) {
                el.classList.remove('hidden');
                if (icon) {
                    icon.classList.remove('fa-folder');
                    icon.classList.add('fa-folder-open');
                }
            } else {
                el.classList.add('hidden');
                if (icon) {
                    icon.classList.remove('fa-folder-open');
                    icon.classList.add('fa-folder');
                }
            }
        }

        // Live preview dynamic streaming content
        async function previewFile(filePath, shortDesc) {
            const card = document.getElementById('explorer-preview-card');
            card.innerHTML = `
                <div class="glass-panel rounded-2xl p-8 flex flex-col items-center justify-center h-full text-center">
                    <div class="w-12 h-12 rounded-full border-4 border-cyber-primary border-t-transparent animate-spin mb-4"></div>
                    <p class="text-sm text-gray-400">Streaming real-time workspace source content...</p>
                </div>
            `;
            
            let previewContent = "";
            if (filePath.endsWith('.pdf')) {
                previewContent = `[Binary Presentation Document]\n\nThis is a compiled project presentation slide deck in PDF format, containing the high-level business strategy, scoring benchmarks, and evaluation highlights.\n\nPath: c:\\Users\\sayan\\.gemini\\antigravity-ide\\scratch\\credit_risk_platform\\documents\\project_presentation.pdf`;
            } else if (filePath === 'data/Home Credit dataset files') {
                previewContent = `[Seeded SQLite Database Environment]\n\nThe actual source tables are located in the safe SQLite storage file: data/credit_risk.db\n\nTables seeded:\n- applications: Applicant demographics and current metrics\n- previous_applications: Historical application details and pipeline results`;
            } else if (filePath.endsWith('.joblib') || filePath.endsWith('.pkl')) {
                previewContent = `[Binary Machine Learning Model Weights]\n\nFitted pipeline binary file hosting trained classifier node coefficients.\n\nPath: models/credit_model.joblib\nHyperparameters: estimators=150, max_depth=10, weights=balanced`;
            } else {
                try {
                    const res = await fetch(`/api/project-file?path=${encodeURIComponent(filePath)}`);
                    if (res.ok) {
                        const data = await res.json();
                        previewContent = data.content;
                    } else {
                        previewContent = `Error: Unable to fetch source content from server.`;
                    }
                } catch(e) {
                    previewContent = `Error fetching preview content: ${e.message}`;
                }
            }
            
            // Determine language class for syntax highlighting or visual representation
            let lang = 'python';
            if (filePath.endsWith('.sql')) lang = 'sql';
            else if (filePath.endsWith('.json')) lang = 'json';
            else if (filePath.endsWith('.md')) lang = 'markdown';
            else if (filePath.includes('Dockerfile')) lang = 'dockerfile';
            else if (filePath.endsWith('.yml')) lang = 'yaml';
            else if (filePath.endsWith('.txt') || filePath.endsWith('.gitignore')) lang = 'plaintext';

            // Safely encode content for HTML injection
            const escapedContent = previewContent
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");

            card.innerHTML = `
                <div class="glass-panel rounded-2xl p-6 flex flex-col h-full overflow-hidden animate-fade-in">
                    <div class="flex items-center justify-between pb-4 border-b border-gray-800 mb-4 shrink-0">
                        <div>
                            <h3 class="text-sm font-semibold font-display text-white tracking-tight flex items-center gap-2">
                                <i class="fa-solid fa-file-code text-cyber-primary"></i>
                                ${filePath.split('/').pop()}
                            </h3>
                            <p class="text-[10px] text-gray-500 mt-0.5 font-mono">${filePath}</p>
                        </div>
                        <button id="copy-src-btn" class="text-xs text-gray-400 hover:text-white bg-gray-900 border border-gray-800 hover:border-cyber-primary rounded px-2.5 py-1.5 transition flex items-center gap-1.5">
                            <i class="fa-solid fa-copy"></i> Copy Code
                        </button>
                    </div>
                    
                    <div class="bg-cyber-primaryGlow/5 border border-cyber-primary/10 rounded-xl p-3.5 mb-4 text-xs shrink-0">
                        <span class="text-cyber-emerald font-semibold uppercase tracking-wider text-[9px] block mb-1">Architecture Role:</span>
                        <p class="text-gray-300 leading-relaxed">${shortDesc}</p>
                    </div>
                    
                    <div class="flex-1 overflow-auto rounded-xl border border-gray-800 bg-gray-950 p-4 font-mono text-[10px] leading-relaxed relative">
                        <pre class="text-gray-300 whitespace-pre overflow-x-auto"><code class="language-${lang}">${escapedContent}</code></pre>
                    </div>
                </div>
            `;

            // Attach dynamic copy event listener to avoid inline escaping issues
            document.getElementById('copy-src-btn').addEventListener('click', () => {
                navigator.clipboard.writeText(previewContent);
                alert('Source code copied to clipboard!');
            });
        }

        // Fetch general KPIs for main page
        async function fetchKPIs() {
            try {
                const res = await fetch('/api/dashboard-summary');
                const data = await res.json();
                
                document.getElementById('kpi-total-apps').innerText = Number(data.total_applicants).toLocaleString();
                document.getElementById('kpi-del-rate').innerText = data.default_rate + '%';
                document.getElementById('kpi-avg-credit').innerText = '$' + Number(Math.round(data.avg_credit)).toLocaleString();
                document.getElementById('kpi-avg-income').innerText = '$' + Number(Math.round(data.avg_income)).toLocaleString();
            } catch (e) {
                console.error("Error loading statistics kpis: " + e);
            }
        }

        // Load specific applicant data parameters from loader preset
        function loadPresetApplicant(presetKey) {
            if (!presetKey) return;
            const p = PRESETS[presetKey];
            const form = document.getElementById('scoring-form');
            
            // Populate basic form inputs
            form.elements['NAME_CONTRACT_TYPE'].value = p.NAME_CONTRACT_TYPE;
            form.elements['CODE_GENDER'].value = p.CODE_GENDER;
            form.elements['FLAG_OWN_CAR'].value = p.FLAG_OWN_CAR;
            form.elements['FLAG_OWN_REALTY'].value = p.FLAG_OWN_REALTY;
            form.elements['AMT_INCOME_TOTAL'].value = p.AMT_INCOME_TOTAL;
            form.elements['AMT_CREDIT'].value = p.AMT_CREDIT;
            form.elements['AMT_ANNUITY'].value = p.AMT_ANNUITY;
            form.elements['AMT_GOODS_PRICE'].value = p.AMT_GOODS_PRICE;
            
            document.getElementById('inp-age-years').value = p.age;
            document.getElementById('inp-emp-years').value = p.emp;
            
            form.elements['NAME_INCOME_TYPE'].value = p.NAME_INCOME_TYPE;
            form.elements['NAME_EDUCATION_TYPE'].value = p.NAME_EDUCATION_TYPE;
            form.elements['NAME_FAMILY_STATUS'].value = p.NAME_FAMILY_STATUS;
            form.elements['NAME_HOUSING_TYPE'].value = p.NAME_HOUSING_TYPE;
            
            form.elements['EXT_SOURCE_1'].value = p.EXT_SOURCE_1;
            form.elements['EXT_SOURCE_2'].value = p.EXT_SOURCE_2;
            form.elements['EXT_SOURCE_3'].value = p.EXT_SOURCE_3;
        }

        // POST dynamic scoring analysis
        async function submitScoring() {
            const form = document.getElementById('scoring-form');
            const age = parseFloat(document.getElementById('inp-age-years').value);
            const emp = parseFloat(document.getElementById('inp-emp-years').value);
            
            const payload = {
                SK_ID_CURR: 888801,
                NAME_CONTRACT_TYPE: form.elements['NAME_CONTRACT_TYPE'].value,
                CODE_GENDER: form.elements['CODE_GENDER'].value,
                FLAG_OWN_CAR: form.elements['FLAG_OWN_CAR'].value,
                FLAG_OWN_REALTY: form.elements['FLAG_OWN_REALTY'].value,
                CNT_CHILDREN: 0,
                AMT_INCOME_TOTAL: parseFloat(form.elements['AMT_INCOME_TOTAL'].value),
                AMT_CREDIT: parseFloat(form.elements['AMT_CREDIT'].value),
                AMT_ANNUITY: parseFloat(form.elements['AMT_ANNUITY'].value),
                AMT_GOODS_PRICE: parseFloat(form.elements['AMT_GOODS_PRICE'].value),
                NAME_INCOME_TYPE: form.elements['NAME_INCOME_TYPE'].value,
                NAME_EDUCATION_TYPE: form.elements['NAME_EDUCATION_TYPE'].value,
                NAME_FAMILY_STATUS: form.elements['NAME_FAMILY_STATUS'].value,
                NAME_HOUSING_TYPE: form.elements['NAME_HOUSING_TYPE'].value,
                DAYS_BIRTHDAY: Math.round(-age * 365.25),
                DAYS_EMPLOYED: Math.round(-emp * 365.25),
                OCCUPATION_TYPE: "Laborers",
                CNT_FAM_MEMBERS: 2,
                REGION_RATING_CLIENT: 2,
                EXT_SOURCE_1: parseFloat(form.elements['EXT_SOURCE_1'].value),
                EXT_SOURCE_2: parseFloat(form.elements['EXT_SOURCE_2'].value),
                EXT_SOURCE_3: parseFloat(form.elements['EXT_SOURCE_3'].value)
            };

            const container = document.getElementById('scorecard-result-container');
            container.innerHTML = `
                <div class="glass-panel rounded-2xl p-8 flex flex-col items-center justify-center min-h-[500px]">
                    <div class="w-12 h-12 rounded-full border-4 border-cyber-primary border-t-transparent animate-spin mb-4"></div>
                    <p class="text-sm text-gray-400">Synthesizing credit risk models & explainability tensors...</p>
                </div>
            `;

            try {
                const res = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
                
                const data = await res.json();
                renderScorecardResult(data);
            } catch (e) {
                console.error("Scorecard error: " + e);
                container.innerHTML = `
                    <div class="glass-panel rounded-2xl p-8 border-cyber-rose/30 text-center text-xs">
                        <i class="fa-solid fa-triangle-exclamation text-2xl text-cyber-rose mb-3"></i>
                        <p class="text-red-400 font-semibold">Underwriting score execution failed.</p>
                        <p class="text-gray-500 mt-1">${e.message}</p>
                    </div>
                `;
            }
        }

        // Render dynamic scorecard outcomes
        function renderScorecardResult(data) {
            const container = document.getElementById('scorecard-result-container');
            
            // Build rules html
            let rulesHtml = '';
            data.rules.forEach(r => {
                const color = r.severity === 'High' ? 'text-cyber-rose bg-cyber-rose/10 border-cyber-rose/20' : 
                              r.severity === 'Medium' ? 'text-cyber-amber bg-cyber-amber/10 border-cyber-amber/20' : 
                              'text-cyber-emerald bg-cyber-emerald/10 border-cyber-emerald/20';
                              
                rulesHtml += `
                    <div class="border rounded-xl p-3.5 ${color} text-xs flex gap-3">
                        <div class="text-sm mt-0.5"><i class="fa-solid fa-circle-exclamation"></i></div>
                        <div>
                            <h5 class="font-semibold">${r.rule}</h5>
                            <p class="mt-1 opacity-80">${r.description}</p>
                        </div>
                    </div>
                `;
            });

            container.innerHTML = `
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <!-- Probability Dial -->
                    <div class="glass-panel rounded-2xl p-6 flex flex-col items-center justify-center text-center relative overflow-hidden">
                        <h4 class="text-sm font-semibold text-gray-400 mb-2 font-display">AI Credit Risk Score</h4>
                        
                        <!-- Circular radial gauge -->
                        <div id="underwriting-radial-chart" class="w-full min-h-[220px]"></div>

                        <div class="mt-2">
                            <span class="inline-flex items-center gap-1.5 px-3.5 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-${data.color}/10 text-cyber-${data.color} border border-${data.color}/20">
                                <span class="w-1.5 h-1.5 rounded-full bg-cyber-${data.color} animate-pulse"></span>
                                ${data.risk_band} Risk Band
                            </span>
                        </div>
                    </div>

                    <!-- Underwriting policy rules -->
                    <div class="glass-panel rounded-2xl p-6 space-y-4">
                        <h4 class="text-sm font-semibold text-gray-400 font-display"><i class="fa-solid fa-clipboard-list mr-2 text-cyber-secondary"></i>Underwriting Policies</h4>
                        <div class="space-y-3 overflow-y-auto max-h-[230px] pr-2">
                            ${rulesHtml}
                        </div>
                    </div>
                </div>

                <!-- XAI Waterfall explanations -->
                <div class="glass-panel rounded-2xl p-6">
                    <h4 class="text-sm font-semibold text-gray-400 mb-6 font-display"><i class="fa-solid fa-chart-bar mr-2 text-cyber-primary"></i>Local Explainable AI (Local Contributions)</h4>
                    <div id="chart-underwriting-xai" class="w-full min-h-[300px]"></div>
                </div>
            `;

            // Render radial probability gauge
            const radialOptions = {
                series: [data.score],
                chart: {
                    type: 'radialBar',
                    height: 220,
                    sparkline: { enabled: true }
                },
                plotOptions: {
                    radialBar: {
                        startAngle: -90,
                        endAngle: 90,
                        track: {
                            background: '#1f2937',
                            strokeWidth: '97%',
                        },
                        dataLabels: {
                            name: { show: false },
                            value: {
                                offsetY: -2,
                                fontSize: '32px',
                                fontWeight: '700',
                                color: '#ffffff',
                                formatter: function (val) {
                                    return val;
                                }
                            }
                        }
                    }
                },
                fill: {
                    colors: [data.color === 'emerald' ? '#10b981' : data.color === 'amber' ? '#f59e0b' : '#f43f5e']
                },
                stroke: { lineCap: 'round' }
            };

            const radialChart = new ApexCharts(document.querySelector("#underwriting-radial-chart"), radialOptions);
            radialChart.render();

            // Render Local XAI Horizontal Bar Chart
            // Format data lists
            const xai_features = data.explanations.map(e => e.label);
            const xai_values = data.explanations.map(e => Math.round(e.contribution * 10) / 10);
            
            const xaiOptions = {
                series: [{
                    name: 'Risk Contribution',
                    data: xai_values
                }],
                chart: {
                    type: 'bar',
                    height: 300,
                    toolbar: { show: false }
                },
                plotOptions: {
                    bar: {
                        colors: {
                            ranges: [{
                                from: -100,
                                to: 0,
                                color: '#10b981' // Green reduces risk
                            }, {
                                from: 0,
                                to: 100,
                                color: '#f43f5e' // Red increases risk
                            }]
                        },
                        columnWidth: '80%',
                    }
                },
                dataLabels: { enabled: false },
                yaxis: {
                    title: {
                        text: 'Risk Offset Score',
                        style: { color: '#9ca3af' }
                    },
                    labels: {
                        style: { colors: '#9ca3af' }
                    }
                },
                xaxis: {
                    categories: xai_features,
                    labels: {
                        rotate: -45,
                        style: { colors: '#9ca3af', fontSize: '10px' }
                    }
                },
                grid: { borderColor: '#1f2937' }
            };

            const xaiChart = new ApexCharts(document.querySelector("#chart-underwriting-xai"), xaiOptions);
            xaiChart.render();
        }

        // Chat terminal execution triggers
        function fillAndSendChat(text) {
            document.getElementById('chat-input-text').value = text;
            sendChatMessage();
        }

        async function sendChatMessage() {
            const input = document.getElementById('chat-input-text');
            const message = input.value.trim();
            if (!message) return;

            // Clear input
            input.value = '';

            const chatbox = document.getElementById('chat-messages-box');
            
            // 1. Append user message bubble
            chatbox.innerHTML += `
                <div class="flex gap-3 justify-end max-w-[85%] ml-auto">
                    <div class="bg-cyber-primaryGlow border border-cyber-primary/30 p-3.5 rounded-2xl rounded-tr-none text-xs text-gray-200">
                        ${message}
                    </div>
                </div>
            `;
            chatbox.scrollTop = chatbox.scrollHeight;

            // 2. Append temporary loader typing bubble
            const loadingId = 'chat-loading-' + Date.now();
            chatbox.innerHTML += `
                <div class="flex gap-3 max-w-[85%]" id="${loadingId}">
                    <div class="w-8 h-8 rounded-lg bg-cyber-primaryGlow border border-cyber-primary/20 flex items-center justify-center text-cyber-primary shrink-0 text-sm">
                        <i class="fa-solid fa-robot"></i>
                    </div>
                    <div class="bg-gray-900 border border-gray-800 p-3.5 rounded-2xl rounded-tl-none text-xs flex items-center gap-1.5 text-gray-400">
                        <span class="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></span>
                        <span class="w-2 h-2 bg-gray-500 rounded-full animate-bounce [animation-delay:0.2s]"></span>
                        <span class="w-2 h-2 bg-gray-500 rounded-full animate-bounce [animation-delay:0.4s]"></span>
                    </div>
                </div>
            `;
            chatbox.scrollTop = chatbox.scrollHeight;

            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ message: message })
                });
                
                const data = await res.json();
                
                // Remove loader
                document.getElementById(loadingId).remove();
                
                // Formulate table headers and rows if SQL was success
                let tableHtml = '';
                if (data.status === 'success' && data.data.length > 0) {
                    let headers = '';
                    data.columns.forEach(col => {
                        headers += `<th class="px-3 py-2 text-left text-[10px] font-semibold text-gray-400 uppercase tracking-wider">${col}</th>`;
                    });
                    
                    let rows = '';
                    data.data.slice(0, 5).forEach(row => { // limit to 5 preview rows in bubble
                        rows += `<tr class="border-b border-gray-850">`;
                        data.columns.forEach(col => {
                            let val = row[col];
                            if (typeof val === 'number') {
                                val = val.toLocaleString();
                            }
                            rows += `<td class="px-3 py-2 text-gray-300">${val}</td>`;
                        });
                        rows += `</tr>`;
                    });

                    tableHtml = `
                        <div class="overflow-x-auto border border-gray-800 rounded-xl my-2 bg-gray-950 max-w-full">
                            <table class="min-w-full text-[10px] font-medium">
                                <thead class="bg-gray-900 border-b border-gray-800">
                                    <tr>${headers}</tr>
                                </thead>
                                <tbody>
                                    ${rows}
                                </tbody>
                            </table>
                            ${data.data.length > 5 ? `<div class="text-[9px] text-gray-500 text-center py-1">Previewing first 5 rows of database records</div>` : ''}
                        </div>
                    `;
                }

                // Compile formatted narrative
                chatbox.innerHTML += `
                    <div class="flex gap-3 max-w-[90%]">
                        <div class="w-8 h-8 rounded-lg bg-cyber-primaryGlow border border-cyber-primary/20 flex items-center justify-center text-cyber-primary shrink-0 text-sm">
                            <i class="fa-solid fa-robot"></i>
                        </div>
                        <div class="bg-gray-900 border border-gray-800 p-4 rounded-2xl rounded-tl-none space-y-3 text-xs w-full overflow-hidden">
                            <div class="text-[10px] bg-gray-950 font-mono p-2.5 rounded-lg border border-gray-850 text-cyber-primary overflow-x-auto whitespace-pre">
                                <span class="text-gray-500 font-semibold uppercase tracking-wider block text-[8px] mb-1">SQL Query Executed:</span>${data.sql}
                            </div>
                            
                            ${tableHtml}

                            <div class="text-gray-300 leading-relaxed">
                                ${data.analysis}
                            </div>
                        </div>
                    </div>
                `;
                chatbox.scrollTop = chatbox.scrollHeight;

            } catch (e) {
                // Remove loader
                document.getElementById(loadingId).remove();
                chatbox.innerHTML += `
                    <div class="flex gap-3 max-w-[85%]">
                        <div class="w-8 h-8 rounded-lg bg-cyber-rose/10 border border-cyber-rose/20 flex items-center justify-center text-cyber-rose shrink-0 text-sm">
                            <i class="fa-solid fa-circle-exclamation"></i>
                        </div>
                        <div class="bg-gray-900 border border-gray-800 p-3.5 rounded-2xl rounded-tl-none text-xs text-red-400">
                            SQL engine failed to translate and execute. ${e.message}
                        </div>
                    </div>
                `;
                chatbox.scrollTop = chatbox.scrollHeight;
            }
        }

        // Fetch diagnostics and metrics curves for Diagnostics section
        async function fetchModelDiagnostics() {
            try {
                const res = await fetch('/api/model-metrics');
                const data = await res.json();
                
                // Set features count
                document.getElementById('diag-fitted-feats').innerText = data.feature_importances.length;
                
                // Draw confusion matrix
                const cm = data.metrics.confusion_matrix;
                document.getElementById('diag-confusion-matrix').innerHTML = `
                    <div class="bg-gray-950 p-2 border border-gray-850 rounded">
                        <span class="text-[9px] text-gray-500 block">True Negative</span>
                        <p class="text-sm font-bold text-white mt-1">${cm[0][0]}</p>
                    </div>
                    <div class="bg-gray-950 p-2 border border-gray-850 rounded">
                        <span class="text-[9px] text-gray-500 block">False Positive</span>
                        <p class="text-sm font-bold text-cyber-rose mt-1">${cm[0][1]}</p>
                    </div>
                    <div class="bg-gray-950 p-2 border border-gray-850 rounded">
                        <span class="text-[9px] text-gray-500 block">False Negative</span>
                        <p class="text-sm font-bold text-cyber-rose mt-1">${cm[1][0]}</p>
                    </div>
                    <div class="bg-gray-950 p-2 border border-gray-850 rounded">
                        <span class="text-[9px] text-gray-500 block">True Positive</span>
                        <p class="text-sm font-bold text-white mt-1">${cm[1][1]}</p>
                    </div>
                `;

                // Draw global feature importances in Dashboard Section
                let featureImp = '';
                data.feature_importances.slice(0, 5).forEach(f => {
                    const pct = Math.round(f.importance * 100);
                    featureImp += `
                        <div class="text-xs">
                            <div class="flex justify-between text-gray-300 font-medium mb-1">
                                <span>${f.feature.replace('_YEARS', ' (Years)').replace('CREDIT_TO_INCOME_RATIO', 'Leverage Ratio')}</span>
                                <span class="text-cyber-primary font-semibold">${pct}%</span>
                            </div>
                            <div class="w-full h-1.5 bg-gray-800 rounded-full overflow-hidden">
                                <div class="h-full bg-gradient-to-r from-cyber-primary to-cyber-secondary rounded-full" style="width: ${pct}%"></div>
                            </div>
                        </div>
                    `;
                });
                document.getElementById('dash-feature-imp').innerHTML = featureImp;

                // Render ROC Line Chart
                const roc_fpr = data.curves.roc.map(p => p.fpr);
                const roc_tpr = data.curves.roc.map(p => p.tpr);
                
                const rocOptions = {
                    series: [{
                        name: 'Random Forest Model',
                        data: roc_tpr
                    }, {
                        name: 'Baseline Guesses',
                        data: roc_fpr // diagonal baseline line representation
                    }],
                    chart: {
                        type: 'line',
                        height: 250,
                        toolbar: { show: false }
                    },
                    stroke: { curve: 'smooth', width: 2 },
                    colors: ['#6366f1', '#4b5563'],
                    xaxis: {
                        categories: roc_fpr,
                        title: { text: 'False Positive Rate (FPR)', style: {color: '#9ca3af'} },
                        labels: { style: {colors: '#9ca3af'} }
                    },
                    yaxis: {
                        title: { text: 'True Positive Rate (TPR)', style: {color: '#9ca3af'} },
                        labels: { style: {colors: '#9ca3af'} }
                    },
                    grid: { borderColor: '#1f2937' }
                };

                const rocChart = new ApexCharts(document.querySelector("#chart-diag-roc"), rocOptions);
                rocChart.render();

                // Render PR Curve
                const pr_rec = data.curves.pr.map(p => p.recall);
                const pr_prec = data.curves.pr.map(p => p.precision);

                const prOptions = {
                    series: [{
                        name: 'Precision-Recall Value',
                        data: pr_prec
                    }],
                    chart: {
                        type: 'line',
                        height: 250,
                        toolbar: { show: false }
                    },
                    stroke: { curve: 'smooth', width: 2 },
                    colors: ['#a855f7'],
                    xaxis: {
                        categories: pr_rec,
                        title: { text: 'Recall', style: {color: '#9ca3af'} },
                        labels: { style: {colors: '#9ca3af'} }
                    },
                    yaxis: {
                        title: { text: 'Precision', style: {color: '#9ca3af'} },
                        labels: { style: {colors: '#9ca3af'} }
                    },
                    grid: { borderColor: '#1f2937' }
                };

                const prChart = new ApexCharts(document.querySelector("#chart-diag-pr"), prOptions);
                prChart.render();

            } catch (e) {
                console.error("Diagnostics load error: " + e);
            }
        }

        // Fetch pre-aggregated statistics for EDA Section charts
        async function fetchEDAMetrics() {
            try {
                const res = await fetch('/api/eda-charts');
                const data = await res.json();
                
                // Chart 1: Education Level Bar Chart
                const eduOptions = {
                    series: [{
                        name: 'Default Rate (%)',
                        data: data.education.default_rates
                    }],
                    chart: { type: 'bar', height: 200, toolbar: {show: false} },
                    plotOptions: { bar: {borderRadius: 4} },
                    colors: ['#6366f1'],
                    xaxis: {
                        categories: data.education.categories,
                        labels: { style: {colors: '#9ca3af', fontSize: '9px'} }
                    },
                    yaxis: { labels: {style: {colors: '#9ca3af'}} },
                    grid: { borderColor: '#1f2937' }
                };
                new ApexCharts(document.querySelector("#chart-eda-edu"), eduOptions).render();

                // Chart 2: Income Type Bar Chart
                const incOptions = {
                    series: [{
                        name: 'Default Rate (%)',
                        data: data.income_type.default_rates
                    }],
                    chart: { type: 'bar', height: 200, toolbar: {show: false} },
                    plotOptions: { bar: {borderRadius: 4} },
                    colors: ['#a855f7'],
                    xaxis: {
                        categories: data.income_type.categories,
                        labels: { style: {colors: '#9ca3af', fontSize: '9px'} }
                    },
                    yaxis: { labels: {style: {colors: '#9ca3af'}} },
                    grid: { borderColor: '#1f2937' }
                };
                new ApexCharts(document.querySelector("#chart-eda-income"), incOptions).render();

                // Chart 3: Occupation Bar Chart
                const occOptions = {
                    series: [{
                        name: 'Default Rate (%)',
                        data: data.occupation.default_rates
                    }],
                    chart: { type: 'bar', height: 200, toolbar: {show: false} },
                    plotOptions: { bar: {borderRadius: 4} },
                    colors: ['#10b981'],
                    xaxis: {
                        categories: data.occupation.categories,
                        labels: { rotate: -30, style: {colors: '#9ca3af', fontSize: '9px'} }
                    },
                    yaxis: { labels: {style: {colors: '#9ca3af'}} },
                    grid: { borderColor: '#1f2937' }
                };
                new ApexCharts(document.querySelector("#chart-eda-occ"), occOptions).render();

                // Chart 4: Income Bins Line Chart
                const binOptions = {
                    series: [{
                        name: 'Default Rate (%)',
                        data: data.income_bins.default_rates
                    }],
                    chart: { type: 'line', height: 200, toolbar: {show: false} },
                    stroke: { curve: 'smooth', width: 3 },
                    colors: ['#f59e0b'],
                    xaxis: {
                        categories: data.income_bins.categories,
                        labels: { style: {colors: '#9ca3af', fontSize: '9px'} }
                    },
                    yaxis: { labels: {style: {colors: '#9ca3af'}} },
                    grid: { borderColor: '#1f2937' }
                };
                new ApexCharts(document.querySelector("#chart-eda-bins"), binOptions).render();

            } catch (e) {
                console.error("EDA charts compile error: " + e);
            }
        }
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content, status_code=200)
