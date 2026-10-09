# streamlit_app.py
# ------------------------------------------------------------------
# Preemptive Cancer Risk Screening — questionnaire-based ML web app
# Breast & Prostate models (Logistic Regression, threshold-tuned).
# ------------------------------------------------------------------
import streamlit as st
import joblib

# Importing ml_utils registers MultiMethodFeatureSelector so joblib can
# rebuild the saved pipelines. Must happen BEFORE the models are loaded.
import ml_utils  # noqa: F401
from encoding import encode_breast, encode_prostate

# ---------- page setup ----------
st.set_page_config(page_title="Cancer Risk Screening", page_icon="🎗️", layout="centered")

# ---------- load models once (cached across reruns) ----------
@st.cache_resource
def load_models():
    breast   = joblib.load("models/breast_cancer_lr_model.pkl")
    prostate = joblib.load("models/prostate_cancer_lr_model.pkl")
    return breast, prostate

breast_model, prostate_model = load_models()

# ---------- shared disclaimer ----------
DISCLAIMER = (
    "⚠️ **This tool is for educational screening support only.** It does **not** "
    "provide a medical diagnosis. Results are a risk estimate from a machine-learning "
    "model. Always consult a qualified healthcare professional for medical advice."
)

# ---------- sidebar navigation ----------
st.sidebar.title("🎗️ Cancer Risk Screening")
page = st.sidebar.radio("Choose a screening tool:", ["Home", "Breast Cancer", "Prostate Cancer"])

# ---------- pages ----------
def home_page():
    st.title("Preemptive Cancer Risk Screening")
    st.write(
        "This application estimates the risk of **breast** or **prostate** cancer "
        "from a short questionnaire, using machine-learning models trained on clinical data."
    )
    st.info(DISCLAIMER)
    st.write("👈 Select a screening tool from the sidebar to begin.")

def breast_page():
    st.title("🎗️ Breast Cancer Risk Screening")
    st.warning(DISCLAIMER)
    st.caption("Answer the questions below, then click Predict. Some questions appear based on your answers.")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", 20, 90, value=45)
        menarche = st.number_input("Age at first menstrual period", 8, 17, value=13)
        children = st.number_input("Number of children", 0, 5, value=0, help="Enter 5 for 5 or more.")
    with col2:
        family_history = st.radio("Any close relative diagnosed with breast cancer?", ["No", "Yes"], horizontal=True)
        hyperplasia    = st.radio("Ever diagnosed with breast hyperplasia?", ["No", "Yes"], horizontal=True)
        tobacco        = st.radio("Do you use tobacco?", ["No", "Yes"], horizontal=True)
        alcohol        = st.radio("Do you drink alcohol?", ["No", "Yes"], horizontal=True)

    # --- conditional: menopause age only if menopausal ---
    is_menopausal = st.radio("Have you reached menopause?", ["No", "Yes"], horizontal=True)
    menopause_age = 0
    if is_menopausal == "Yes":
        menopause_age = st.number_input("Age at menopause", 30, 65, value=50)

    # --- conditional: childbirth details only if she has children ---
    age_first_birth = 0
    breastfeeding_months = 0
    if children > 0:
        age_first_birth      = st.number_input("Age at first childbirth", 12, 50, value=28)
        breastfeeding_months = st.number_input("Total months breastfeeding (all children combined)", 0, 72, value=6)

    if st.button("🔍 Predict Risk", type="primary"):
        answers = dict(
            tobacco=(tobacco == "Yes"),
            alcohol=(alcohol == "Yes"),
            hyperplasia=(hyperplasia == "Yes"),
            family_history=(family_history == "Yes"),
            children=children,
            age=age,
            menarche=menarche,
            is_menopausal=(is_menopausal == "Yes"),
            menopause_age=menopause_age,
            age_first_birth=age_first_birth,
            breastfeeding_months=breastfeeding_months,
        )
        X = encode_breast(answers)
        pred  = breast_model.predict(X)[0]
        proba = breast_model.predict_proba(X)[0][1]

        st.subheader("Result")
        st.metric("Estimated cancer risk", f"{proba*100:.1f}%")
        if pred == 1:
            st.error("🔴 Higher-risk result — clinical follow-up (e.g. seeing a doctor) is recommended.")
        else:
            st.success("🟢 Lower-risk result based on the answers provided.")
        st.caption("This is a screening estimate, not a diagnosis — see the disclaimer above.")

def prostate_page():
    st.title("🎗️ Prostate Cancer Risk Screening")
    st.warning(DISCLAIMER)
    st.caption("Enter values from a recent blood test (PSA panel) plus basic details, then click Predict.")

    with st.form("prostate_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age (years)", 40, 90, value=63)
            bmi = st.number_input("BMI (body mass index)", 15.0, 45.0, value=27.0, step=0.1)
            psa = st.number_input("PSA level (ng/mL)", 0.0, 50.0, value=11.0, step=0.1,
                                  help="Prostate-Specific Antigen — from a blood test.")
        with col2:
            free_psa = st.number_input("Free PSA ratio", 0.0, 1.0, value=0.16, step=0.01,
                                       help="Free PSA divided by total PSA. Lower can indicate higher risk.")
            testo = st.number_input("Testosterone level (ng/dL)", 100.0, 1000.0, value=476.0, step=1.0)
            alp = st.number_input("Alkaline phosphatase (U/L)", 20.0, 200.0, value=97.0, step=0.1)

        submitted = st.form_submit_button("🔍 Predict Risk")

    if submitted:
        answers = dict(psa=psa, alp=alp, testosterone=testo,
                       free_psa_ratio=free_psa, age=age, bmi=bmi)
        X = encode_prostate(answers)
        pred  = prostate_model.predict(X)[0]
        proba = prostate_model.predict_proba(X)[0][1]

        st.subheader("Result")
        st.metric("Estimated malignancy risk", f"{proba*100:.1f}%")
        if pred == 1:
            st.error("🔴 Higher-risk result — clinical follow-up (e.g. seeing a doctor) is recommended.")
        else:
            st.success("🟢 Lower-risk result based on the values provided.")
        st.caption("This is a screening estimate, not a diagnosis — see the disclaimer above.")

# ---------- router ----------
if page == "Home":
    home_page()
elif page == "Breast Cancer":
    breast_page()
else:
    prostate_page()