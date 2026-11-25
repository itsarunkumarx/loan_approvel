from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import pickle
import io
import os

app = FastAPI(
    title="Loan Approval Prediction & Batch Evaluation API",
    description="Machine Learning service to predict individual and batch loan approval statuses with applicant name support",
    version="2.1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "logistic_regression_model.pkl")
DASHBOARD_FILE = os.path.join(BASE_DIR, "dashboard.html")
SAMPLE_CSV_FILE = os.path.join(BASE_DIR, "loan_approval_1000.csv")

# Load model from .pkl
model = None
try:
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    print("SUCCESS: Loaded logistic_regression_model.pkl")
except Exception as e:
    print(f"WARNING: Could not load model pickle: {e}")

EMP_MAPPING = {
    "Salaried": 0,
    "Self-Employed": 1,
    "Unemployed": 2
}

class LoanApplicant(BaseModel):
    Name: str = Field(default="Applicant", example="John Doe")
    Age: int = Field(..., ge=18, le=100, example=35)
    Income: float = Field(..., ge=0, example=85000.0)
    Loan_Amount: float = Field(..., ge=1000, example=250000.0)
    Credit_Score: int = Field(..., ge=300, le=900, example=720)
    Employment_Type: str = Field(..., example="Salaried", description="Salaried, Self-Employed, or Unemployed")

class PredictionResponse(BaseModel):
    name: str
    approved: bool
    probability: float
    risk_score: str
    status: str
    log_odds: float
    model_source: str

def parse_num(val, default):
    if pd.isna(val) or val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    clean_str = str(val).replace("$", "").replace(",", "").strip()
    try:
        return float(clean_str)
    except Exception:
        return default

def find_column(df, candidates):
    cols_lower = {str(c).lower().replace(" ", "_").replace("-", "_"): c for c in df.columns}
    for cand in candidates:
        cand_norm = cand.lower().replace(" ", "_").replace("-", "_")
        if cand_norm in cols_lower:
            return cols_lower[cand_norm]
    return None

def score_dataframe(df_raw: pd.DataFrame):
    df = df_raw.copy()
    
    # 1. Identify Name Column
    col_name = find_column(df, [
        "Name", "name", "Full_Name", "full_name", "Applicant_Name", "applicant_name",
        "Customer_Name", "customer_name", "Client_Name", "Person_Name", "Borrower_Name"
    ])
    
    col_first_name = find_column(df, ["First_Name", "first_name", "fname"])
    col_last_name = find_column(df, ["Last_Name", "last_name", "lname", "surname"])

    if col_name:
        df["Applicant_Name"] = df[col_name].fillna("Unnamed Applicant").astype(str).str.strip()
    elif col_first_name and col_last_name:
        df["Applicant_Name"] = (df[col_first_name].fillna("").astype(str).str.strip() + " " + df[col_last_name].fillna("").astype(str).str.strip()).str.strip()
    elif col_first_name:
        df["Applicant_Name"] = df[col_first_name].fillna("Unnamed Applicant").astype(str).str.strip()
    else:
        df["Applicant_Name"] = [f"Applicant #{i+1:04d}" for i in range(len(df))]

    # 2. Identify or generate Applicant ID
    col_id = find_column(df, [
        "Applicant_ID", "applicant_id", "ID", "id", "Application_ID", "application_id",
        "Loan_ID", "loan_id", "Customer_ID", "customer_id", "User_ID", "user_id"
    ])
    if col_id:
        df["Applicant_ID"] = df[col_id].fillna("").astype(str).str.strip()
    else:
        df["Applicant_ID"] = [f"APP-{i+1:04d}" for i in range(len(df))]

    # 3. Identify numeric / ML feature columns
    col_age = find_column(df, ["Age", "age", "applicant_age", "borrower_age"])
    col_income = find_column(df, ["Income", "income", "annual_income", "salary", "monthly_income", "applicant_income"])
    col_loan = find_column(df, ["Loan_Amount", "loan_amount", "loan", "amount", "loan_amt", "requested_amount"])
    col_credit = find_column(df, ["Credit_Score", "credit_score", "score", "cibil", "cibil_score", "fico"])
    col_emp = find_column(df, ["Employment_Type", "employment_type", "employment", "job", "job_type", "work_status"])

    missing_cols = []
    if not col_income: missing_cols.append("Income")
    if not col_credit: missing_cols.append("Credit_Score")
    if not col_loan: missing_cols.append("Loan_Amount")

    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded CSV is missing required columns: {', '.join(missing_cols)}. Please verify your CSV header."
        )

    clean_age = df[col_age].apply(lambda x: parse_num(x, 40.0)).astype(float) if col_age else pd.Series([40.0] * len(df))
    clean_income = df[col_income].apply(lambda x: parse_num(x, 85000.0)).astype(float)
    clean_loan = df[col_loan].apply(lambda x: parse_num(x, 250000.0)).astype(float)
    clean_credit = df[col_credit].apply(lambda x: parse_num(x, 610.0)).astype(float)

    def map_emp(val):
        if pd.isna(val) or val is None:
            return 0  # Salaried default
        s = str(val).strip().lower()
        if "self" in s:
            return 1
        elif "unemploy" in s:
            return 2
        return 0  # Salaried

    clean_emp = df[col_emp].apply(map_emp).astype(int) if col_emp else pd.Series([0] * len(df))

    # Features DataFrame
    X = pd.DataFrame({
        "Age": clean_age,
        "Income": clean_income,
        "Loan_Amount": clean_loan,
        "Credit_Score": clean_credit,
        "Employment_Type": clean_emp
    })

    if model is not None:
        probs = model.predict_proba(X)[:, 1]
    else:
        coef = np.array([1.61064728e-02, 2.28136828e-05, -1.69063369e-06, 8.29314143e-03, -4.44354720e-01])
        intercept = -8.12945928
        logits = intercept + np.dot(X.values, coef)
        probs = 1.0 / (1.0 + np.exp(-logits))

    preds = (probs >= 0.5).astype(int)

    # Attach prediction columns
    df["Model_Decision"] = ["APPROVED" if p == 1 else "REJECTED" for p in preds]
    df["Approval_Probability"] = [round(float(p) * 100, 1) for p in probs]
    
    def get_risk(p):
        if p >= 0.70:
            return "Low Risk"
        elif p >= 0.50:
            return "Moderate Risk"
        else:
            return "High Risk"
            
    df["Risk_Tier"] = [get_risk(p) for p in probs]

    # Standardized normalized display fields
    df["Display_Age"] = clean_age.astype(int)
    df["Display_Income"] = clean_income
    df["Display_Loan_Amount"] = clean_loan
    df["Display_Credit_Score"] = clean_credit.astype(int)
    
    raw_emp_series = df[col_emp] if col_emp else pd.Series(["Salaried"] * len(df))
    df["Display_Employment"] = raw_emp_series.fillna("Salaried").astype(str)

    return df, probs, preds

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    if os.path.exists(DASHBOARD_FILE):
        return FileResponse(DASHBOARD_FILE)
    return HTMLResponse("<h1>Dashboard not found</h1>")

@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard_alias():
    if os.path.exists(DASHBOARD_FILE):
        return FileResponse(DASHBOARD_FILE)
    return HTMLResponse("<h1>Dashboard not found</h1>")

@app.get("/sample-csv")
def download_sample_csv():
    # If a sample with names exists, return that, otherwise return base sample
    sample_with_names = os.path.join(BASE_DIR, "sample_applicants_with_names.csv")
    if os.path.exists(sample_with_names):
        return FileResponse(sample_with_names, media_type="text/csv", filename="sample_applicants_with_names.csv")
    elif os.path.exists(SAMPLE_CSV_FILE):
        return FileResponse(SAMPLE_CSV_FILE, media_type="text/csv", filename="sample_loan_applicants.csv")
    raise HTTPException(status_code=404, detail="Sample CSV not found")

@app.post("/predict", response_model=PredictionResponse)
def predict_loan(applicant: LoanApplicant):
    emp_type = applicant.Employment_Type.strip()
    if emp_type not in EMP_MAPPING:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Employment_Type '{emp_type}'. Allowed values: {list(EMP_MAPPING.keys())}"
        )

    emp_code = EMP_MAPPING[emp_type]
    X = pd.DataFrame([{
        "Age": applicant.Age,
        "Income": applicant.Income,
        "Loan_Amount": applicant.Loan_Amount,
        "Credit_Score": applicant.Credit_Score,
        "Employment_Type": emp_code
    }])

    if model is not None:
        prob = float(model.predict_proba(X)[0][1])
        approved = bool(model.predict(X)[0] == 1)
        logit = float(np.log(prob / (1.0 - prob + 1e-12)))
        source = "logistic_regression_model.pkl"
    else:
        coef = np.array([1.61064728e-02, 2.28136828e-05, -1.69063369e-06, 8.29314143e-03, -4.44354720e-01])
        intercept = -8.12945928
        feats = np.array([applicant.Age, applicant.Income, applicant.Loan_Amount, applicant.Credit_Score, emp_code])
        logit = float(intercept + np.dot(feats, coef))
        prob = float(1.0 / (1.0 + np.exp(-logit)))
        approved = bool(prob >= 0.5)
        source = "Fallback Weights"

    risk = "Low Risk" if prob >= 0.7 else ("Moderate Risk" if prob >= 0.5 else "High Risk")

    return {
        "name": applicant.Name,
        "approved": approved,
        "probability": round(prob, 4),
        "risk_score": risk,
        "status": "Approved" if approved else "Rejected",
        "log_odds": round(logit, 4),
        "model_source": source
    }

@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    if not file.filename.endswith((".csv", ".txt")):
        raise HTTPException(status_code=400, detail="Only CSV files (.csv) are currently supported.")

    content = await file.read()
    try:
        df_raw = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {str(e)}")

    if len(df_raw) == 0:
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty.")

    evaluated_df, probs, preds = score_dataframe(df_raw)

    total = len(evaluated_df)
    approved_count = int((preds == 1).sum())
    rejected_count = int((preds == 0).sum())
    approval_rate = round((approved_count / total) * 100, 1)

    # Average metrics for approved cohort
    approved_rows = evaluated_df[evaluated_df["Model_Decision"] == "APPROVED"]
    col_credit = find_column(evaluated_df, ["Credit_Score", "credit_score", "score"])
    col_income = find_column(evaluated_df, ["Income", "income", "annual_income"])
    
    avg_approved_credit = round(float(approved_rows[col_credit].mean()), 0) if (len(approved_rows) > 0 and col_credit) else 0
    avg_approved_income = round(float(approved_rows[col_income].mean()), 0) if (len(approved_rows) > 0 and col_income) else 0

    has_real_names = bool(find_column(df_raw, [
        "Name", "name", "Full_Name", "full_name", "Applicant_Name", "applicant_name",
        "Customer_Name", "customer_name", "Client_Name", "Person_Name", "First_Name"
    ]))

    records = evaluated_df.fillna("").to_dict(orient="records")

    return {
        "filename": file.filename,
        "total_applicants": total,
        "approved_count": approved_count,
        "rejected_count": rejected_count,
        "approval_rate": approval_rate,
        "avg_approved_credit": avg_approved_credit,
        "avg_approved_income": avg_approved_income,
        "has_names": has_real_names,
        "records": records
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
