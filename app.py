import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_classif, chi2, f_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, recall_score, f1_score,
                             roc_auc_score, roc_curve, auc)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# STEP 1: LOAD AND CLEAN DATA
# ============================================================
print("🔍 Loading breast cancer dataset...")
df = pd.read_csv('data/breast_cancer_preprocessed.csv')

leakage_cols = ['birads', 'year', 'histologicalclass']
df = df.drop(columns=[c for c in leakage_cols if c in df.columns])
df = df.dropna()

print(f"✅ Dataset cleaned. Shape: {df.shape}")
print(f"   Columns dropped : {leakage_cols}")
print(f"   Cancer distribution:\n{df['cancer'].value_counts(normalize=True).round(3)}\n")


# ============================================================
# STEP 2: SPLIT DATA FIRST  (prevents leakage)
# ============================================================
X = df.drop('cancer', axis=1)
y = df['cancer']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 Train set : {len(X_train)} samples | Test set : {len(X_test)} samples")
print(f"   Train cancer rate : {y_train.mean():.2%} | "
      f"Test cancer rate  : {y_test.mean():.2%}\n")


# ============================================================
# STEP 3: FEATURE SELECTION  (training data only)
# ============================================================
print("🧠 Running Feature Selection on training data only...")

X_train_num = X_train.copy()
for col in X_train_num.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_train_num[col] = le.fit_transform(X_train_num[col])

mi_scores      = mutual_info_classif(X_train_num, y_train, random_state=42)
mi_rank        = pd.Series(mi_scores, index=X_train.columns).rank(ascending=False)

chi2_scores, _ = chi2(X_train_num, y_train)
chi2_rank      = pd.Series(chi2_scores, index=X_train.columns).rank(ascending=False)

f_scores, _    = f_classif(X_train_num, y_train)
f_rank         = pd.Series(f_scores, index=X_train.columns).rank(ascending=False)

avg_rank           = (mi_rank + chi2_rank + f_rank) / 3
feature_importance = avg_rank.sort_values()
top_10_features    = feature_importance.head(10).index.tolist()

print("\n📈 Top 10 Features by Statistical Rank:")
for i, feat in enumerate(top_10_features, 1):
    print(f"   {i}. {feat}  (Avg Rank: {feature_importance[feat]:.2f})")

X_train_sel = X_train[top_10_features]
X_test_sel  = X_test[top_10_features]
print(f"\n✅ Selected features: {top_10_features}\n")


# ============================================================
# STEP 4: SMOTE for LR  (LR uses MinMax-scaled data from
#         preprocessing — no extra scaling needed)
# ============================================================
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_sel, y_train)

print(f"⚖️  SMOTE applied for LR — balanced class distribution:")
print(pd.Series(y_train_bal).value_counts().to_string())

# Shared CV strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ════════════════════════════════════════════════════════════
#  MODEL 1 — LOGISTIC REGRESSION
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODEL 1 — LOGISTIC REGRESSION")
print("=" * 60)

lr_model = LogisticRegression(max_iter=1000, random_state=42)

cv_lr = cross_val_score(lr_model, X_train_bal, y_train_bal,
                        cv=cv, scoring='roc_auc')
print(f"\n🔁 5-Fold CV ROC-AUC : {cv_lr.mean():.4f} ± {cv_lr.std():.4f}")
print(f"   Individual folds  : {[round(s, 4) for s in cv_lr]}")

lr_model.fit(X_train_bal, y_train_bal)
y_pred_lr = lr_model.predict(X_test_sel)
y_prob_lr = lr_model.predict_proba(X_test_sel)[:, 1]

acc_lr     = accuracy_score(y_test, y_pred_lr)
rec_lr     = recall_score(y_test, y_pred_lr)
f1_lr      = f1_score(y_test, y_pred_lr)
roc_auc_lr = roc_auc_score(y_test, y_prob_lr)

print(f"\n📊 LR — Test Performance:")
print(f"   Accuracy             : {acc_lr:.4f}")
print(f"   Recall (Sensitivity) : {rec_lr:.4f}  ← CRITICAL FOR CANCER SCREENING")
print(f"   F1-Score             : {f1_lr:.4f}")
print(f"   ROC-AUC              : {roc_auc_lr:.4f}")
print(f"\n   Classification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['No Cancer', 'Cancer']))

# LR Plots
plt.figure(figsize=(7, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Cancer', 'Cancer'], yticklabels=['No Cancer', 'Cancer'])
plt.title('Logistic Regression — Confusion Matrix', fontweight='bold')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('lr_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
plt.figure(figsize=(7, 5))
plt.plot(fpr_lr, tpr_lr, color='darkorange', lw=2,
         label=f'LR  (AUC = {roc_auc_lr:.3f})')
plt.plot([0, 1], [0, 1], 'navy', lw=1.5, linestyle='--', label='Random')
plt.fill_between(fpr_lr, tpr_lr, alpha=0.1, color='darkorange')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Logistic Regression', fontweight='bold')
plt.legend(loc='lower right'); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('lr_roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()

coef_df = pd.DataFrame({
    'Feature': top_10_features, 'Coefficient': lr_model.coef_[0]
}).sort_values('Coefficient')
colors_coef = ['#e74c3c' if c > 0 else '#3498db' for c in coef_df['Coefficient']]
plt.figure(figsize=(8, 6))
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors_coef)
plt.axvline(0, color='black', linewidth=0.8)
plt.title('LR — Feature Coefficients\n'
          '(Red = raises cancer risk  |  Blue = lowers cancer risk)',
          fontweight='bold')
plt.xlabel('Coefficient Value')
plt.tight_layout()
plt.savefig('lr_feature_coefficients.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ LR plots saved.")


# ════════════════════════════════════════════════════════════
#  MODEL 2 — K-NEAREST NEIGHBORS (KNN)
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODEL 2 — K-NEAREST NEIGHBORS (KNN)")
print("=" * 60)

# Scale FIRST → then SMOTE (KNN is distance-based)
scaler_knn       = StandardScaler()
X_train_knn_sc   = scaler_knn.fit_transform(X_train_sel)
X_test_knn_sc    = scaler_knn.transform(X_test_sel)

X_train_knn_bal, y_train_knn_bal = smote.fit_resample(X_train_knn_sc, y_train)

print(f"\n⚖️  SMOTE on scaled data — balanced distribution:")
print(pd.Series(y_train_knn_bal).value_counts().to_string())

print("\n🔍 Finding optimal K via 5-Fold CV (scoring = Recall)...")
k_range     = range(1, 21)
cv_scores_k = []

for k in k_range:
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=k),
                             X_train_knn_bal, y_train_knn_bal,
                             cv=cv, scoring='recall')
    cv_scores_k.append(scores.mean())

best_k = list(k_range)[np.argmax(cv_scores_k)]
print(f"   ✅ Best K : {best_k}  (CV Recall : {max(cv_scores_k):.4f})")

cv_knn = cross_val_score(KNeighborsClassifier(n_neighbors=best_k),
                         X_train_knn_bal, y_train_knn_bal,
                         cv=cv, scoring='roc_auc')
print(f"   5-Fold CV ROC-AUC : {cv_knn.mean():.4f} ± {cv_knn.std():.4f}")
print(f"   Individual folds  : {[round(s, 4) for s in cv_knn]}")

knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train_knn_bal, y_train_knn_bal)

y_pred_knn = knn_model.predict(X_test_knn_sc)
y_prob_knn = knn_model.predict_proba(X_test_knn_sc)[:, 1]

acc_knn     = accuracy_score(y_test, y_pred_knn)
rec_knn     = recall_score(y_test, y_pred_knn)
f1_knn      = f1_score(y_test, y_pred_knn)
roc_auc_knn = roc_auc_score(y_test, y_prob_knn)

print(f"\n📊 KNN — Test Performance:")
print(f"   Accuracy             : {acc_knn:.4f}")
print(f"   Recall (Sensitivity) : {rec_knn:.4f}  ← CRITICAL FOR CANCER SCREENING")
print(f"   F1-Score             : {f1_knn:.4f}")
print(f"   ROC-AUC              : {roc_auc_knn:.4f}")
print(f"\n   Classification Report:")
print(classification_report(y_test, y_pred_knn, target_names=['No Cancer', 'Cancer']))

# KNN Plots
plt.figure(figsize=(7, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', cmap='Greens',
            xticklabels=['No Cancer', 'Cancer'], yticklabels=['No Cancer', 'Cancer'])
plt.title('KNN — Confusion Matrix', fontweight='bold')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('knn_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)
plt.figure(figsize=(7, 5))
plt.plot(fpr_knn, tpr_knn, color='green', lw=2,
         label=f'KNN K={best_k}  (AUC = {roc_auc_knn:.3f})')
plt.plot([0, 1], [0, 1], 'navy', lw=1.5, linestyle='--', label='Random')
plt.fill_between(fpr_knn, tpr_knn, alpha=0.1, color='green')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve — KNN Model', fontweight='bold')
plt.legend(loc='lower right'); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('knn_roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(9, 5))
plt.plot(k_range, cv_scores_k, marker='o', color='green', lw=2, label='CV Recall per K')
plt.axvline(best_k, color='red', linestyle='--', label=f'Best K = {best_k}')
plt.title('Elbow Method — Optimal K Selection (Recall-Based)', fontweight='bold')
plt.xlabel('K Value'); plt.ylabel('Cross-Validated Recall')
plt.xticks(k_range); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('knn_elbow_method.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ KNN plots saved.")


# ════════════════════════════════════════════════════════════
#  MODEL 3 — SUPPORT VECTOR MACHINE (SVM)
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODEL 3 — SUPPORT VECTOR MACHINE (SVM)")
print("=" * 60)

# Scale FIRST → then SMOTE  (same correct order as KNN)
scaler_svm      = StandardScaler()
X_train_svm_sc  = scaler_svm.fit_transform(X_train_sel)
X_test_svm_sc   = scaler_svm.transform(X_test_sel)

X_train_svm_bal, y_train_svm_bal = smote.fit_resample(X_train_svm_sc, y_train)

print(f"\n⚖️  SMOTE on scaled data — balanced distribution:")
print(pd.Series(y_train_svm_bal).value_counts().to_string())

# Find optimal C via Cross-Validation (optimise for Recall)
print("\n🔍 Finding optimal C via 5-Fold CV (scoring = Recall)...")
c_values    = [0.1, 0.5, 1.0, 5.0, 10.0]
cv_scores_c = []

for c in c_values:
    svm_temp = SVC(kernel='rbf', C=c, gamma='scale',
                   probability=True, random_state=42)
    scores   = cross_val_score(svm_temp, X_train_svm_bal, y_train_svm_bal,
                               cv=cv, scoring='recall')
    cv_scores_c.append(scores.mean())
    print(f"   C = {c:<5}  →  CV Recall : {scores.mean():.4f}")

best_c = c_values[np.argmax(cv_scores_c)]
print(f"\n   ✅ Best C : {best_c}  (CV Recall : {max(cv_scores_c):.4f})")

cv_svm = cross_val_score(
    SVC(kernel='rbf', C=best_c, gamma='scale', probability=True, random_state=42),
    X_train_svm_bal, y_train_svm_bal, cv=cv, scoring='roc_auc'
)
print(f"   5-Fold CV ROC-AUC : {cv_svm.mean():.4f} ± {cv_svm.std():.4f}")
print(f"   Individual folds  : {[round(s, 4) for s in cv_svm]}")

# Train final SVM with best C
# No class_weight='balanced' — SMOTE already handled the imbalance
svm_model = SVC(
    kernel='rbf',
    C=best_c,
    gamma='scale',
    probability=True,
    random_state=42
)
svm_model.fit(X_train_svm_bal, y_train_svm_bal)

y_pred_svm = svm_model.predict(X_test_svm_sc)
y_prob_svm = svm_model.predict_proba(X_test_svm_sc)[:, 1]

acc_svm     = accuracy_score(y_test, y_pred_svm)
rec_svm     = recall_score(y_test, y_pred_svm)
f1_svm      = f1_score(y_test, y_pred_svm)
roc_auc_svm = roc_auc_score(y_test, y_prob_svm)

print(f"\n📊 SVM — Test Performance:")
print(f"   Accuracy             : {acc_svm:.4f}")
print(f"   Recall (Sensitivity) : {rec_svm:.4f}  ← CRITICAL FOR CANCER SCREENING")
print(f"   F1-Score             : {f1_svm:.4f}")
print(f"   ROC-AUC              : {roc_auc_svm:.4f}")
print(f"\n   Classification Report:")
print(classification_report(y_test, y_pred_svm, target_names=['No Cancer', 'Cancer']))

# SVM Plots
plt.figure(figsize=(7, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', cmap='Purples',
            xticklabels=['No Cancer', 'Cancer'], yticklabels=['No Cancer', 'Cancer'])
plt.title('SVM — Confusion Matrix\n(RBF Kernel, Tuned C)', fontweight='bold')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('svm_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
plt.figure(figsize=(7, 5))
plt.plot(fpr_svm, tpr_svm, color='purple', lw=2,
         label=f'SVM C={best_c}  (AUC = {roc_auc_svm:.3f})')
plt.plot([0, 1], [0, 1], 'navy', lw=1.5, linestyle='--', label='Random')
plt.fill_between(fpr_svm, tpr_svm, alpha=0.1, color='purple')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve — SVM Model', fontweight='bold')
plt.legend(loc='lower right'); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('svm_roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()

# C selection plot (mirrors KNN elbow plot)
plt.figure(figsize=(8, 5))
plt.plot(c_values, cv_scores_c, marker='o', color='purple', lw=2, label='CV Recall per C')
plt.axvline(best_c, color='red', linestyle='--', label=f'Best C = {best_c}')
plt.title('C Parameter Search — SVM (Recall-Based)', fontweight='bold')
plt.xlabel('C Value'); plt.ylabel('Cross-Validated Recall')
plt.xscale('log'); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('svm_c_selection.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ SVM plots saved.")


# ════════════════════════════════════════════════════════════
#  FINAL COMPARISON — All Three Models
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  FINAL MODEL COMPARISON — Breast Cancer Prediction")
print("=" * 65)
print(f"  {'Metric':<22} | {'Log. Reg':^12} | {'KNN (K='+str(best_k)+')':^12} | {'SVM (C='+str(best_c)+')':^12}")
print(f"  {'-'*22}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
print(f"  {'Accuracy':<22} | {acc_lr:^12.4f} | {acc_knn:^12.4f} | {acc_svm:^12.4f}")
print(f"  {'Recall (Sensitivity)':<22} | {rec_lr:^12.4f} | {rec_knn:^12.4f} | {rec_svm:^12.4f}")
print(f"  {'F1-Score':<22} | {f1_lr:^12.4f} | {f1_knn:^12.4f} | {f1_svm:^12.4f}")
print(f"  {'ROC-AUC':<22} | {roc_auc_lr:^12.4f} | {roc_auc_knn:^12.4f} | {roc_auc_svm:^12.4f}")
print(f"  {'CV ROC-AUC':<22} | {cv_lr.mean():^12.4f} | {cv_knn.mean():^12.4f} | {cv_svm.mean():^12.4f}")
print("=" * 65)
print("  💡 For cancer screening → prioritise the model with")
print("     highest RECALL. Missing a cancer is worse than a false alarm.")
print("=" * 65)

# Combined ROC Curve — all three models
plt.figure(figsize=(8, 6))
plt.plot(fpr_lr,  tpr_lr,  color='darkorange', lw=2,
         label=f'Logistic Regression  (AUC = {roc_auc_lr:.3f})')
plt.plot(fpr_knn, tpr_knn, color='green',      lw=2,
         label=f'KNN K={best_k}              (AUC = {roc_auc_knn:.3f})')
plt.plot(fpr_svm, tpr_svm, color='purple',     lw=2,
         label=f'SVM C={best_c}             (AUC = {roc_auc_svm:.3f})')
plt.plot([0, 1], [0, 1], 'navy', lw=1.5, linestyle='--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison — All Three Models\n(Breast Cancer Prediction)',
          fontweight='bold')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('comparison_all_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Combined ROC comparison saved → comparison_all_roc_curves.png")
print("\n🎉 Breast Cancer modelling complete — LR, KNN, and SVM all done!")