import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

st.set_page_config(page_title="Diabetes Predictor", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }
.stApp { background: #0f1117; }

div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label {
    font-size: 12px !important;
    color: #9ca3af !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
div[data-testid="stNumberInput"] > div {
    background: #12151f !important;
    border: 0.5px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
}
div[data-testid="stNumberInput"] input {
    color: #e8eaf0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 14px !important;
    text-align: center !important;
    background: transparent !important;
}
div[data-testid="stNumberInput"] button {
    color: #6b7280 !important;
    background: transparent !important;
}
div[data-testid="stNumberInput"] button:hover {
    color: #fff !important;
    background: rgba(255,255,255,0.06) !important;
}
div[data-testid="stSelectbox"] > div {
    background: #12151f !important;
    border: 0.5px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    color: #e8eaf0 !important;
}
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 2rem !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.03em !important;
    width: 100%;
}
div[data-testid="stButton"] > button:hover { opacity: 0.85 !important; }
div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #6366f1, #10b981) !important;
    border-radius: 99px !important;
}
div[data-testid="stProgress"] > div {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 99px !important;
    height: 8px !important;
}
hr { border-color: rgba(255,255,255,0.07) !important; }
</style>
""", unsafe_allow_html=True)

# ── MODEL ────────────────────────────────────────────────────────────────────
@st.cache_resource
def train_model():
    df = pd.read_csv("data/diabetes.csv")
    cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols] = df[cols].replace(0, df[cols].median())
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(class_weight='balanced', probability=True))
    ])
    param_grid = {"svm__kernel": ["linear", "rbf"], "svm__C": [0.1, 1, 10]}
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='recall')
    grid.fit(X_train, y_train)
    return grid.best_estimator_, X.columns

model, features = train_model()

# ── TITLE ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-size:52px;font-weight:700;color:#fff;
           display:flex;align-items:center;gap:10px;margin-bottom:1rem'>
  <svg width='20' height='20' viewBox='0 0 24 24' fill='none'
    stroke='#6366f1' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'>
    <path d='M22 12h-4l-3 9L9 3l-3 9H2'/>
  </svg>
  Diabetes Prediction
</h1>
""", unsafe_allow_html=True)

# ── INSTRUCTION BANNER ───────────────────────────────────────────────────────
st.markdown("""
<div style='background:#1a1d27;border:0.5px solid rgba(255,255,255,0.08);
            border-radius:12px;padding:1rem 1.25rem;margin-bottom:1.5rem;
            display:flex;gap:14px;align-items:flex-start'>
  <svg style='flex-shrink:0;margin-top:2px' width='16' height='16' viewBox='0 0 24 24'
    fill='none' stroke='#6366f1' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>
    <circle cx='12' cy='12' r='10'/><line x1='12' y1='8' x2='12' y2='12'/>
    <line x1='12' y1='16' x2='12.01' y2='16'/>
  </svg>
  <div>
    <p style='font-size:13px;font-weight:600;color:#a5b4fc;margin:0 0 6px'>How to use this tool</p>
    <ul style='margin:0;padding-left:1.1rem;font-size:12.5px;color:#9ca3af;line-height:1.9'>
      <li>Enter realistic medical values for accurate prediction</li>
      <li><b>Glucose</b> is the most important factor (normal: 70–140 mg/dL)</li>
      <li><b>Blood Pressure</b> should be resting diastolic value (60–120 mmHg)</li>
      <li><b>Insulin</b> and <b>Skin Thickness</b> can be left as 0 if unknown</li>
      <li><b>Height</b> and <b>Weight</b> are used to automatically calculate BMI</li>
      <li><b>DPF</b> represents family history risk (higher = more risk)</li>
      <li><b>Age</b> should be in years</li>
      <li>This tool provides an estimate and is not a medical diagnosis</li>
    </ul>
  </div>
</div>
""", unsafe_allow_html=True)

# ── MAIN COLUMNS ─────────────────────────────────────────────────────────────
left, right = st.columns([2.2, 1], gap="large")

with left:

    # ── INPUTS ───────────────────────────────────────────────────────────────
    st.markdown("""
    <p style='font-size:11px;font-weight:600;letter-spacing:.1em;
              text-transform:uppercase;color:#6b7280;margin-bottom:.5rem'>
      Inputs
    </p>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        pregnancies = st.number_input("Pregnancies",    min_value=0.0, step=1.0,  format="%.0f")
        glucose     = st.number_input("Glucose (mg/dL)",        min_value=0.0, step=1.0,  format="%.0f")
        bp          = st.number_input("Blood Pressure (mmHg)", min_value=0.0, step=1.0,  format="%.0f")
        skin        = st.number_input("Skin Thickness (mm)", min_value=0.0, step=1.0,  format="%.0f")
    with c2:
        insulin     = st.number_input("Insulin (μU/mL)",        min_value=0.0, step=5.0,  format="%.0f")
        dpf         = st.number_input("DPF (Diabetes Pedigree Function)",            min_value=0.0, step=0.01, format="%.2f")
        age         = st.number_input("Age",            min_value=0.0, step=1.0,  format="%.0f")

    # ── BMI SECTION ──────────────────────────────────────────────────────────
st.markdown("""
<p style='font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
          color:#6b7280;margin-top:1.25rem;margin-bottom:.5rem'>
  BMI Calculator
</p>""", unsafe_allow_html=True)

# HEIGHT ROW
h1, h2 = st.columns([1, 2], gap="medium")

with h1:
    height_unit = st.selectbox("Height unit", ["cm", "ft/in"])

with h2:
    if height_unit == "cm":
        height_cm = st.number_input("Height (cm)", min_value=0.0, max_value=300.0,
                                   step=0.5, format="%.1f")
    else:
        ft_col, in_col = st.columns(2)
        with ft_col:
            height_ft = st.number_input("Height (ft)", min_value=0, max_value=8, step=1)
        with in_col:
            height_in = st.number_input("Height (in)", min_value=0, max_value=11, step=1)
        height_cm = (height_ft * 12 + height_in) * 2.54

# WEIGHT ROW
w1, w2 = st.columns([1, 2], gap="medium")

with w1:
    weight_unit = st.selectbox("Weight unit", ["kg", "lbs"])

with w2:
    if weight_unit == "kg":
        weight_kg = st.number_input("Weight (kg)", min_value=0.0, max_value=500.0,
                                   step=0.5, format="%.1f")
    else:
        weight_lbs = st.number_input("Weight (lbs)", min_value=0.0, max_value=1100.0,
                                    step=0.5, format="%.1f")
        weight_kg  = weight_lbs * 0.453592

# Compute BMI
if height_cm > 0 and weight_kg > 0:
    bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)

    if bmi < 18.5:
        bmi_cat, bmi_color = "Underweight", "#facc15"
    elif bmi < 25:
        bmi_cat, bmi_color = "Normal weight", "#10b981"
    elif bmi < 30:
        bmi_cat, bmi_color = "Overweight", "#f97316"
    else:
        bmi_cat, bmi_color = "Obese", "#ef4444"

    st.markdown(f"""
    <div style='background:#12151f;border:0.5px solid rgba(255,255,255,0.1);
                border-radius:10px;padding:.85rem 1.1rem;margin-top:.75rem;
                display:flex;align-items:center;gap:16px'>
      <div>
        <div style='font-size:11px;color:#6b7280;font-weight:600;
                    text-transform:uppercase;letter-spacing:.08em;margin-bottom:2px'>
          Computed BMI
        </div>
        <div style='font-size:24px;font-weight:700;color:#e8eaf0;
                    font-family:"DM Mono",monospace;line-height:1'>
          {bmi}
        </div>
      </div>
      <div style='height:40px;width:0.5px;background:rgba(255,255,255,0.1)'></div>
      <div>
        <div style='font-size:11px;color:#6b7280;font-weight:600;
                    text-transform:uppercase;letter-spacing:.08em;margin-bottom:2px'>
          Category
        </div>
        <div style='font-size:14px;font-weight:600;color:{bmi_color}'>{bmi_cat}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    bmi = 0.0
    st.markdown("""
    <div style='background:#12151f;border:0.5px solid rgba(255,255,255,0.07);
                border-radius:10px;padding:.85rem 1.1rem;margin-top:.75rem;
                font-size:13px;color:#4b5563'>
      Enter height and weight to compute BMI
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
st.divider()

# ── OUTPUT ───────────────────────────────────────────────────────────────
st.markdown("""
<p style='font-size:11px;font-weight:600;letter-spacing:.1em;
          text-transform:uppercase;color:#6b7280;margin-bottom:.75rem'>
  Output
</p>""", unsafe_allow_html=True)

inputs = [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
valid  = any(v != 0 for v in inputs)

if not valid:
    st.markdown("""
    <div style='background:rgba(99,102,241,0.08);border:0.5px solid rgba(99,102,241,0.3);
                border-radius:8px;padding:12px 16px;color:#a5b4fc;font-size:13px'>
      Enter values above to enable prediction
    </div>""", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom:.75rem'></div>", unsafe_allow_html=True)

if st.button("🔍 Predict", disabled=not valid):
    input_df   = pd.DataFrame([inputs], columns=features)
    prediction = model.predict(input_df)[0]
    prob       = model.predict_proba(input_df)[0][1]
    pct        = round(prob * 100, 1)

    if prediction == 1:
        st.markdown(f"""
        <div style='background:rgba(239,68,68,0.1);border:0.5px solid rgba(239,68,68,0.35);
                    border-radius:10px;padding:1rem 1.25rem;
                    display:flex;align-items:center;gap:14px;margin-bottom:1rem'>
          <span style='font-size:26px'>⚠️</span>
          <div>
            <div style='font-size:16px;font-weight:600;color:#ef4444'>Diabetic</div>
            <div style='font-size:13px;color:#9ca3af;margin-top:2px'>{pct}% probability score</div>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background:rgba(16,185,129,0.1);border:0.5px solid rgba(16,185,129,0.35);
                    border-radius:10px;padding:1rem 1.25rem;
                    display:flex;align-items:center;gap:14px;margin-bottom:1rem'>
          <span style='font-size:26px'>✅</span>
          <div>
            <div style='font-size:16px;font-weight:600;color:#10b981'>Not Diabetic</div>
            <div style='font-size:13px;color:#9ca3af;margin-top:2px'>{pct}% probability score</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<p style='font-size:12px;color:#6b7280;margin-bottom:6px'>Probability</p>",
                unsafe_allow_html=True)
    st.progress(int(prob * 100))
# ── MODEL DETAILS ─────────────────────────────────────────────────────────────
with right:
    chips = "".join(
        f'<span style="background:rgba(255,255,255,0.06);border:0.5px solid rgba(255,255,255,0.1);'
        f'border-radius:20px;padding:3px 10px;font-size:11px;color:#9ca3af;'
        f'font-family:monospace">{f}</span>'
        for f in features
    )
    st.markdown(f"""
    <div style='background:#1a1d27;border:0.5px solid rgba(255,255,255,0.08);
                border-radius:14px;padding:1.25rem'>
      <p style='font-size:11px;font-weight:600;letter-spacing:.1em;
                text-transform:uppercase;color:#6b7280;margin-bottom:1rem'>
        Model Details
      </p>
      <div style='display:flex;justify-content:space-between;padding:8px 0;
                  border-bottom:0.5px solid rgba(255,255,255,0.05);font-size:13px'>
        <span style='color:#6b7280'>Algorithm</span>
        <span style='color:#e8eaf0;font-family:monospace;font-size:12px'>SVM</span>
      </div>
      <div style='display:flex;justify-content:space-between;padding:8px 0;
                  border-bottom:0.5px solid rgba(255,255,255,0.05);font-size:13px'>
        <span style='color:#6b7280'>Scaling</span>
        <span style='color:#e8eaf0;font-family:monospace;font-size:12px'>StandardScaler</span>
      </div>
      <div style='display:flex;justify-content:space-between;padding:8px 0;
                  border-bottom:0.5px solid rgba(255,255,255,0.05);font-size:13px'>
        <span style='color:#6b7280'>Tuning</span>
        <span style='color:#e8eaf0;font-family:monospace;font-size:12px'>GridSearchCV</span>
      </div>
      <div style='display:flex;justify-content:space-between;padding:8px 0;
                  border-bottom:0.5px solid rgba(255,255,255,0.05);font-size:13px'>
        <span style='color:#6b7280'>Metric</span>
        <span style='color:#e8eaf0;font-family:monospace;font-size:12px'>Recall</span>
      </div>
      <div style='margin-top:14px'>
        <span style='font-size:12px;color:#6b7280;font-weight:500'>Features</span>
        <div style='display:flex;flex-wrap:wrap;gap:6px;margin-top:8px'>{chips}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)