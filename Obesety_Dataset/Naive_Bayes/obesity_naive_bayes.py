"""
=============================================================================
Trabalho Final - Inteligência Artificial (PUCRS)
Dataset: Estimation of Obesity Levels Based On Eating Habits and Physical Condition
Modelo Obrigatório: Naive Bayes

Antes de rodar, instale as dependências:
    pip install ucimlrepo scikit-learn pandas numpy matplotlib seaborn
=============================================================================
Bibliotecas: scikit-learn, pandas, numpy, matplotlib, seaborn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_fscore_support
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="colorblind")

OUTPUT_DIR = "output_naive_bayes"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 1. CARREGAMENTO DO DATASET
# =============================================================================
print("=" * 70)
print("1. CARREGAMENTO DO DATASET")
print("=" * 70)

# Importa direto do UCI Machine Learning Repository
# Instale antes: pip install ucimlrepo
dataset = fetch_ucirepo(id=544)
X = dataset.data.features.copy()
y = dataset.data.targets.copy().values.ravel()

print(f"Shape das features: {X.shape}")
print(f"Total de amostras: {len(y)}")
print(f"Valores ausentes: {X.isnull().sum().sum()}")
print(f"\nClasses (NObeyesdad):")
for cls, cnt in pd.Series(y).value_counts().sort_index().items():
    print(f"  {cls}: {cnt} ({cnt/len(y)*100:.1f}%)")

# =============================================================================
# 2. ANÁLISE EXPLORATÓRIA (EDA)
# =============================================================================
print("\n" + "=" * 70)
print("2. ANÁLISE EXPLORATÓRIA")
print("=" * 70)

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
print(f"Variáveis categóricas ({len(cat_cols)}): {cat_cols}")
print(f"Variáveis numéricas  ({len(num_cols)}): {num_cols}")

print("\nEstatísticas descritivas (numéricas):")
print(X[num_cols].describe().round(2).to_string())

# Distribuição das classes
fig, ax = plt.subplots(figsize=(10, 5))
order = sorted(pd.Series(y).unique())
sns.countplot(x=y, order=order, ax=ax)
ax.set_title("Distribuição das Classes de Obesidade", fontsize=14, fontweight="bold")
ax.set_xlabel("Nível de Obesidade")
ax.set_ylabel("Contagem")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_distribuicao_classes.png", dpi=150)
plt.close()
print("  -> Gráfico salvo: 01_distribuicao_classes.png")

# Correlação
fig, ax = plt.subplots(figsize=(10, 8))
corr = X[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Matriz de Correlação - Variáveis Numéricas", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_correlacao.png", dpi=150)
plt.close()
print("  -> Gráfico salvo: 02_correlacao.png")

# Boxplots
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
key_nums = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O"]
for ax, col in zip(axes.ravel(), key_nums):
    sns.boxplot(x=y, y=X[col], order=order, ax=ax)
    ax.set_title(col, fontsize=12)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
plt.suptitle("Distribuição de Variáveis Numéricas por Classe", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_boxplots.png", dpi=150)
plt.close()
print("  -> Gráfico salvo: 03_boxplots.png")

# =============================================================================
# 3. PRÉ-PROCESSAMENTO
# =============================================================================
print("\n" + "=" * 70)
print("3. PRÉ-PROCESSAMENTO")
print("=" * 70)

le_dict = {}
X_encoded = X.copy()
for col in cat_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col])
    le_dict[col] = le
    print(f"  Label Encoding - {col}: {list(le.classes_)}")

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
class_names = le_target.classes_
print(f"\nClasses codificadas: {list(enumerate(class_names))}")

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\nTreino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mm_scaler = MinMaxScaler()
X_train_mm = mm_scaler.fit_transform(X_train)
X_test_mm = mm_scaler.transform(X_test)

# =============================================================================
# 4. NAIVE BAYES - MODELO BASE (GaussianNB default)
# =============================================================================
print("\n" + "=" * 70)
print("4. NAIVE BAYES - MODELO BASE (GaussianNB, var_smoothing=1e-9)")
print("=" * 70)

gnb_base = GaussianNB()
gnb_base.fit(X_train_scaled, y_train)
y_pred_base = gnb_base.predict(X_test_scaled)

acc_base = accuracy_score(y_test, y_pred_base)
print(f"\nAcurácia no teste: {acc_base:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_base, target_names=class_names))

fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_base, display_labels=class_names, ax=ax,
    cmap="Blues", xticks_rotation=45, colorbar=False
)
ax.set_title("Matriz de Confusão - GaussianNB (Base)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_cm_gaussiannb_base.png", dpi=150)
plt.close()
print("  -> Gráfico salvo: 04_cm_gaussiannb_base.png")

# =============================================================================
# 5. VARIAÇÕES DE HIPERPARÂMETROS (mínimo 3 variações)
# =============================================================================
print("\n" + "=" * 70)
print("5. VARIAÇÕES DE HIPERPARÂMETROS")
print("=" * 70)

# --- 5.1 var_smoothing no GaussianNB ---
print("\n--- 5.1 GaussianNB: variação de var_smoothing ---")
var_smoothing_values = [1e-12, 1e-9, 1e-6, 1e-3, 1e-1, 1.0]
results_vs = []

for vs in var_smoothing_values:
    model = GaussianNB(var_smoothing=vs)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="accuracy")
    results_vs.append({
        "var_smoothing": vs, "accuracy_test": acc, "f1_weighted_test": f1,
        "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std()
    })
    print(f"  var_smoothing={vs:.0e} | Acc={acc:.4f} | F1={f1:.4f} | CV={cv_scores.mean():.4f}+/-{cv_scores.std():.4f}")

df_vs = pd.DataFrame(results_vs)

fig, ax = plt.subplots(figsize=(10, 5))
x_labels = [f"{v:.0e}" for v in var_smoothing_values]
ax.plot(x_labels, df_vs["accuracy_test"], "o-", label="Acurácia (Teste)", linewidth=2, markersize=8)
ax.plot(x_labels, df_vs["f1_weighted_test"], "s--", label="F1-Score (Teste)", linewidth=2, markersize=8)
ax.plot(x_labels, df_vs["cv_mean"], "^:", label="CV Accuracy (Média)", linewidth=2, markersize=8)
ax.fill_between(range(len(x_labels)),
    df_vs["cv_mean"] - df_vs["cv_std"], df_vs["cv_mean"] + df_vs["cv_std"],
    alpha=0.15, label="CV ± 1 std")
ax.set_xlabel("var_smoothing")
ax.set_ylabel("Score")
ax.set_title("GaussianNB: Impacto do var_smoothing", fontsize=14, fontweight="bold")
ax.legend()
ax.set_ylim(0.3, 1.0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_var_smoothing.png", dpi=150)
plt.close()
print("  -> Gráfico salvo: 05_var_smoothing.png")

# --- 5.2 Variantes Naive Bayes ---
print("\n--- 5.2 Variante de Naive Bayes: Gaussian vs Multinomial vs Complement ---")
models_nb = {
    "GaussianNB (default)": GaussianNB(),
    "MultinomialNB (alpha=1.0)": MultinomialNB(alpha=1.0),
    "MultinomialNB (alpha=0.1)": MultinomialNB(alpha=0.1),
    "MultinomialNB (alpha=10)": MultinomialNB(alpha=10.0),
    "ComplementNB (alpha=1.0)": ComplementNB(alpha=1.0),
    "ComplementNB (alpha=0.1)": ComplementNB(alpha=0.1),
}
results_variants = []
for name, model in models_nb.items():
    if "Gaussian" in name:
        Xtr, Xte = X_train_scaled, X_test_scaled
    else:
        Xtr, Xte = X_train_mm, X_test_mm
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring="accuracy")
    results_variants.append({
        "Modelo": name, "Accuracy": acc, "Precision": prec, "Recall": rec,
        "F1-Score": f1, "CV Mean": cv_scores.mean(), "CV Std": cv_scores.std()
    })
    print(f"  {name:35s} | Acc={acc:.4f} | F1={f1:.4f} | CV={cv_scores.mean():.4f}+/-{cv_scores.std():.4f}")

df_variants = pd.DataFrame(results_variants)

fig, ax = plt.subplots(figsize=(12, 6))
x_pos = np.arange(len(df_variants))
width = 0.25
ax.bar(x_pos - width, df_variants["Accuracy"], width, label="Accuracy", color="#4C72B0")
ax.bar(x_pos, df_variants["F1-Score"], width, label="F1-Score", color="#55A868")
ax.bar(x_pos + width, df_variants["CV Mean"], width, label="CV Mean", color="#C44E52")
ax.set_xticks(x_pos)
ax.set_xticklabels(df_variants["Modelo"], rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Score")
ax.set_title("Comparação de Variantes Naive Bayes", fontsize=14, fontweight="bold")
ax.legend()
ax.set_ylim(0.3, 1.0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_comparacao_variantes.png", dpi=150)
plt.close()
print("  -> Gráfico salvo: 06_comparacao_variantes.png")

# --- 5.3 alpha no MultinomialNB ---
print("\n--- 5.3 MultinomialNB: variação de alpha (Laplace smoothing) ---")
alpha_values = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
results_alpha = []
for alpha in alpha_values:
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train_mm, y_train)
    y_pred = model.predict(X_test_mm)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_mm, y_train, cv=cv, scoring="accuracy")
    results_alpha.append({
        "alpha": alpha, "accuracy_test": acc, "f1_weighted_test": f1,
        "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std()
    })
    print(f"  alpha={alpha:>7.3f} | Acc={acc:.4f} | F1={f1:.4f} | CV={cv_scores.mean():.4f}+/-{cv_scores.std():.4f}")

df_alpha = pd.DataFrame(results_alpha)

fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogx(df_alpha["alpha"], df_alpha["accuracy_test"], "o-", label="Acurácia (Teste)", linewidth=2, markersize=8)
ax.semilogx(df_alpha["alpha"], df_alpha["f1_weighted_test"], "s--", label="F1-Score (Teste)", linewidth=2, markersize=8)
ax.semilogx(df_alpha["alpha"], df_alpha["cv_mean"], "^:", label="CV Accuracy (Média)", linewidth=2, markersize=8)
ax.fill_between(df_alpha["alpha"],
    df_alpha["cv_mean"] - df_alpha["cv_std"], df_alpha["cv_mean"] + df_alpha["cv_std"],
    alpha=0.15, label="CV ± 1 std")
ax.set_xlabel("alpha (escala log)")
ax.set_ylabel("Score")
ax.set_title("MultinomialNB: Impacto do alpha (Laplace Smoothing)", fontsize=14, fontweight="bold")
ax.legend()
ax.set_ylim(0.3, 1.0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_alpha_multinomial.png", dpi=150)
plt.close()
print("  -> Gráfico salvo: 07_alpha_multinomial.png")

# =============================================================================
# 6. MELHOR MODELO - ANÁLISE DETALHADA
# =============================================================================
print("\n" + "=" * 70)
print("6. MELHOR MODELO NAIVE BAYES - ANÁLISE DETALHADA")
print("=" * 70)

best_vs_row = df_vs.loc[df_vs["accuracy_test"].idxmax()]
best_vs = best_vs_row["var_smoothing"]
print(f"\nMelhor var_smoothing: {best_vs:.0e} (Acc={best_vs_row['accuracy_test']:.4f})")

best_gnb = GaussianNB(var_smoothing=best_vs)
best_gnb.fit(X_train_scaled, y_train)
y_pred_best = best_gnb.predict(X_test_scaled)

print("\nClassification Report (melhor GaussianNB):")
print(classification_report(y_test, y_pred_best, target_names=class_names))

# Confusion Matrix normalizada
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_best, display_labels=class_names, ax=ax,
    cmap="Blues", xticks_rotation=45, colorbar=False, normalize="true"
)
ax.set_title(f"Matriz de Confusão Normalizada - GaussianNB (var_smoothing={best_vs:.0e})",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_cm_melhor_modelo_normalizada.png", dpi=150)
plt.close()
print("  -> Gráfico salvo: 08_cm_melhor_modelo_normalizada.png")

# Métricas por classe
prec_pc, rec_pc, f1_pc, sup_pc = precision_recall_fscore_support(y_test, y_pred_best)
df_perclass = pd.DataFrame({
    "Classe": class_names, "Precision": prec_pc,
    "Recall": rec_pc, "F1-Score": f1_pc, "Suporte": sup_pc
})

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(class_names))
w = 0.25
ax.bar(x - w, df_perclass["Precision"], w, label="Precision")
ax.bar(x, df_perclass["Recall"], w, label="Recall")
ax.bar(x + w, df_perclass["F1-Score"], w, label="F1-Score")
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Score")
ax.set_title("Métricas por Classe - Melhor GaussianNB", fontsize=14, fontweight="bold")
ax.legend()
ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_metricas_por_classe.png", dpi=150)
plt.close()
print("  -> Gráfico salvo: 09_metricas_por_classe.png")

# =============================================================================
# 7. ANÁLISE DE OVERFITTING/UNDERFITTING
# =============================================================================
print("\n" + "=" * 70)
print("7. ANÁLISE DE OVERFITTING / UNDERFITTING")
print("=" * 70)

train_sizes, train_scores, val_scores = learning_curve(
    GaussianNB(var_smoothing=best_vs),
    X_train_scaled, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring="accuracy", n_jobs=-1
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Treino", linewidth=2)
ax.plot(train_sizes, val_scores.mean(axis=1), "s-", label="Validação", linewidth=2)
ax.fill_between(train_sizes,
    train_scores.mean(axis=1) - train_scores.std(axis=1),
    train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
ax.fill_between(train_sizes,
    val_scores.mean(axis=1) - val_scores.std(axis=1),
    val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
ax.set_xlabel("Tamanho do conjunto de treino")
ax.set_ylabel("Acurácia")
ax.set_title("Curva de Aprendizado - GaussianNB", fontsize=14, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/10_learning_curve.png", dpi=150)
plt.close()

train_acc = accuracy_score(y_train, best_gnb.predict(X_train_scaled))
test_acc = accuracy_score(y_test, y_pred_best)
print(f"  Acurácia no treino:  {train_acc:.4f}")
print(f"  Acurácia no teste:   {test_acc:.4f}")
print(f"  Gap (treino - teste): {train_acc - test_acc:.4f}")
if train_acc - test_acc > 0.05:
    print("  -> Indicativo de leve OVERFITTING")
elif test_acc < 0.6:
    print("  -> Indicativo de UNDERFITTING")
else:
    print("  -> Modelo com boa generalização (gap pequeno)")
print("  -> Gráfico salvo: 10_learning_curve.png")

# =============================================================================
# 8. TABELA RESUMO FINAL
# =============================================================================
print("\n" + "=" * 70)
print("8. TABELA RESUMO FINAL")
print("=" * 70)

print("\n--- Resumo: var_smoothing no GaussianNB ---")
print(df_vs.to_string(index=False, float_format="%.4f"))

print("\n--- Resumo: Comparação entre variantes Naive Bayes ---")
print(df_variants.to_string(index=False, float_format="%.4f"))

print("\n--- Resumo: alpha no MultinomialNB ---")
print(df_alpha.to_string(index=False, float_format="%.4f"))

# Salvar tabelas
df_vs.to_csv(f"{OUTPUT_DIR}/tabela_var_smoothing.csv", index=False)
df_variants.to_csv(f"{OUTPUT_DIR}/tabela_variantes_nb.csv", index=False)
df_alpha.to_csv(f"{OUTPUT_DIR}/tabela_alpha_multinomial.csv", index=False)
df_perclass.to_csv(f"{OUTPUT_DIR}/tabela_metricas_por_classe.csv", index=False)

print("\n" + "=" * 70)
print("EXECUÇÃO FINALIZADA! Resultados salvos em:", OUTPUT_DIR)
print("=" * 70)
print("\nArquivos gerados:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  {f}")
