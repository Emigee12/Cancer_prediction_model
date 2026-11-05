# api.py
# ------------------------------------------------------------------
# CancerPredict — REST API backend
#
# Serves the custom frontend (frontend/) and exposes prediction endpoints
# that run the saved Logistic Regression models in models/.
#
# Run with:  python api.py       ->  http://localhost:8000
# ------------------------------------------------------------------
from contextlib import asynccontextmanager
from pathlib import Path
import sys

import joblib
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import ml_utils  # noqa: F401
from encoding import encode_breast, encode_prostate

# Windows consoles default to cp1252 and crash on non-ASCII characters
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
MODELS_DIR = BASE_DIR / "models"

breast_model = None
prostate_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global breast_model, prostate_model
    breast_model = joblib.load(MODELS_DIR / "breast_cancer_lr_model.pkl")
    prostate_model = joblib.load(MODELS_DIR / "prostate_cancer_lr_model.pkl")
    print("✅ Models loaded: breast + prostate")
    yield


app = FastAPI(
    title="CancerPredict API",
    description="Preemptive breast & prostate cancer risk estimation from questionnaire data.",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================
class BreastAnswers(BaseModel):
    """Plain questionnaire answers for the breast risk assessment."""
    age: int = Field(..., ge=20, le=90)
    menarche: int = Field(..., ge=8, le=17)
    children: int = Field(..., ge=0, le=5)
    tobacco: bool
    alcohol: bool
    hyperplasia: bool
    family_history: bool
    is_menopausal: bool
    menopause_age: int = Field(0, ge=0, le=65)
    age_first_birth: int = Field(0, ge=0, le=50)
    breastfeeding_months: int = Field(0, ge=0, le=72)


class ProstateAnswers(BaseModel):
    """Clinical / questionnaire answers for the prostate risk assessment."""
    age: int = Field(..., ge=40, le=90)
    bmi: float = Field(..., ge=15.0, le=45.0)
    psa: float = Field(..., ge=0.0, le=50.0)
    free_psa_ratio: float = Field(..., ge=0.0, le=1.0)
    testosterone: float = Field(..., ge=100.0, le=1000.0)
    alp: float = Field(..., ge=20.0, le=200.0)


class PredictionResponse(BaseModel):
    """What the frontend renders for the user."""
    risk: float                 # probability of malignancy, 0.0 - 1.0
    verdict: str                # 'lower' | 'elevated' | 'higher'
    message: str                # plain-English interpretation
    threshold: float            # the tuned decision threshold used
    features_used: list[str]    # features that actually drive the score


# ============================================================
# PREDICTION HELPERS
# ============================================================
def _interpret(risk: float, threshold: float, higher_msg: str, lower_msg: str):
    """Turn a probability into a three-band verdict the UI can colour-code."""
    if risk >= 0.60:
        verdict = "higher"
        message = higher_msg
    elif risk >= threshold:
        verdict = "elevated"
        message = ("Your estimated risk is above the model's decision threshold. "
                   "Consider discussing this with a healthcare professional.")
    else:
        verdict = "lower"
        message = lower_msg
    return verdict, message


def _predict(model, frame, threshold, higher_msg, lower_msg):
    """Run a model on an encoded feature frame and build the response."""
    prediction = int(model.predict(frame)[0])
    risk = float(model.predict_proba(frame)[0][1])

    verdict, message = _interpret(risk, threshold, higher_msg, lower_msg)
    if prediction == 1 and verdict == "lower":
        verdict = "elevated"  # threshold-tuned decision overrides the band

    # Which features the fitted selector actually uses
    selector = model.estimator_.named_steps["selector"]
    features_used = list(selector.top_features_)

    return PredictionResponse(
        risk=round(risk, 4),
        verdict=verdict,
        message=message,
        threshold=round(float(threshold), 4),
        features_used=features_used,
    )


# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/api/health")
def health():
    """Simple status check used by the frontend to confirm the API is live."""
    return {"status": "ok", "models": ["breast", "prostate"]}


@app.post("/api/predict/breast", response_model=PredictionResponse)
def predict_breast(answers: BreastAnswers):
    """Estimate breast cancer risk from questionnaire answers."""
    frame = encode_breast(answers.model_dump())
    return _predict(
        breast_model,
        frame,
        threshold=breast_model.best_threshold_,
        higher_msg=("Your estimated risk is high. Please arrange a clinical "
                    "evaluation as soon as possible."),
        lower_msg=("Your estimated risk is comparatively low based on the "
                   "answers provided. Continue with routine screening."),
    )


@app.post("/api/predict/prostate", response_model=PredictionResponse)
def predict_prostate(answers: ProstateAnswers):
    """Estimate prostate cancer risk from questionnaire + blood-test values."""
    frame = encode_prostate(answers.model_dump())
    return _predict(
        prostate_model,
        frame,
        threshold=prostate_model.best_threshold_,
        higher_msg=("Your estimated risk is high. Please discuss PSA follow-up "
                    "testing with a urologist."),
        lower_msg=("Your estimated risk is comparatively low based on the "
                   "values provided. Continue with routine screening."),
    )


# ============================================================
# FRONTEND SERVING
# ============================================================
@app.get("/")
def landing():
    """Serve the single-page frontend."""
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 50)
    print("  CancerPredict — starting server")
    print("  Open:  http://localhost:8000")
    print("=" * 50 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
