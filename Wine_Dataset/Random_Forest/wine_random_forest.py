"""
=============================================================================
Trabalho Final - Inteligência Artificial (PUCRS)
Dataset: Wine Quality (UCI id=186)
Modelo: Random Forest (Floresta Aleatória)

Random Forest é um ensemble de árvores de decisão treinadas com:
  - Bootstrap sampling: cada árvore vê uma amostra aleatória do treino
  - Feature randomness: cada split usa apenas um subconjunto aleatório de features
O resultado final é a votação majoritária de todas as árvores.

Antes de rodar, instale as dependências:
    pip install ucimlrepo scikit-learn pandas numpy matplotlib seaborn
=============================================================================
"""

import matplotlib
matplotlib.use("Agg")  # backend não interativo — evita conflito com n_jobs=-1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay,
    precision_recall_fscore_support
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="colorblind")

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 1. CARREGAMENTO DO DATASET
# =============================================================================
print("=" * 70)
print("1. CARREGAMENTO DO DATASET")
print("=" * 70)

dataset = fetch_ucirepo(id=186)
X = dataset.data.features.copy()
y = dataset.data.targets.copy().values.ravel()

print(f"Shape das features: {X.shape}")
print(f"Total de amostras: {len(y)}")
print(f"Valores ausentes: {X.isnull().sum().sum()}")
print(f"\nDistribuição da qualidade (target):")
for cls, cnt in pd.Series(y).value_counts().sort_index().items():
    print(f"  Quality {cls}: {cnt} ({cnt/len(y)*100:.1f}%)")

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

order = sorted(pd.Series(y).unique())

# Distribuição das classes
fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(x=y, order=order, ax=ax)
ax.set_title("Distribuição das Classes de Qualidade do Vinho", fontsize=14, fontweight="bold")
ax.set_xlabel("Qualidade (3=pior, 9=melhor)")
ax.set_ylabel("Contagem")
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_distribuicao_classes.png", dpi=150)
plt.close()
print("  -> Gráfico salvo: 01_distribuicao_classes.png")

# Correlação
fig, ax = plt.subplots(figsize=(12, 9))
df_full = X[num_cols].copy()
df_full["quality"] = y
corr = df_full.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
            annot_kws={"size": 8})
ax.set_title("Matriz de Correlação - Features e Quality", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_correlacao.png", dpi=150)
plt.close()
print("  -> Gráfico salvo: 02_correlacao.png")

# Boxplots
key_features = ["alcohol", "volatile acidity", "sulphates", "citric acid",
                "residual sugar", "density"]
key_features = [f for f in key_features if f in num_cols]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, col in zip(axes.ravel(), key_features):
    sns.boxplot(x=y, y=X[col], order=order, ax=ax)
    ax.set_title(col, fontsize=12)
    ax.set_xlabel("Quality")
plt.suptitle("Distribuição de Variáveis-chave por Qualidade", fontsize=14, fontweight="bold")
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
class_names = [str(c) for c in le_target.classes_]
print(f"\nClasses codificadas: {list(enumerate(class_names))}")

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\nTreino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")
print("Nota: Random Forest é invariante ao escalonamento de features (usa")
print("      comparações ordinais nos splits). StandardScaler não é necessário.")

feature_names = list(X_encoded.columns)

# =============================================================================
# 4. RANDOM FOREST - MODELO BASE
# =============================================================================
print("\n" + "=" * 70)
print("4. RANDOM FOREST - MODELO BASE (n_estimators=100, parâmetros padrão)")
print("=" * 70)

rf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_base.fit(X_train, y_train)
y_pred_base = rf_base.predict(X_test)

acc_base  = accuracy_score(y_test, y_pred_base)
acc_train = accuracy_score(y_train, rf_base.predict(X_train))
print(f"\nAcurácia no treino: {acc_train:.4f}")
print(f"Acurácia no teste:  {acc_base:.4f}")
print(f"Profundidade média das árvores: {np.mean([t.get_depth() for t in rf_base.estimators_]):.1f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_base, target_names=class_names, zero_division=0))

fig, ax = plt.subplots(figsize=(9, 7))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_base, display_labels=class_names, ax=ax,
    cmap="Blues", colorbar=False
)
ax.set_title("Matriz de Confusão - Random Forest Base (100 árvores)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Predição")
ax.set_ylabel("Real")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_cm_rf_base.png", dpi=150)
plt.close()
print("  -> Gráfico salvo: 04_cm_rf_base.png")

# =============================================================================
# 5. VARIAÇÕES DE HIPERPARÂMETROS
# =============================================================================
print("\n" + "=" * 70)
print("5. VARIAÇÕES DE HIPERPARÂMETROS")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- 5.1 n_estimators (número de árvores) ---
# Mais árvores = ensemble mais estável, mas retorno marginal diminui.
# Abaixo de ~50 árvores a variância do ensemble ainda é alta.
print("\n--- 5.1 Random Forest: variação de n_estimators (número de árvores) ---")
n_est_values = [5, 10, 25, 50, 100, 200, 400]
results_nest = []

for n in n_est_values:
    model = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    acc_tr = accuracy_score(y_train, model.predict(X_train))
    acc_te = accuracy_score(y_test,  model.predict(X_test))
    f1  = f1_score(y_test, model.predict(X_test), average="weighted")
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    results_nest.append({
        "n_estimators": n, "accuracy_train": acc_tr, "accuracy_test": acc_te,
        "f1_weighted_test": f1, "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std()
    })
    print(f"  n_estimators={n:>3} | TrainAcc={acc_tr:.4f} | TestAcc={acc_te:.4f} | "
          f"F1={f1:.4f} | CV={cv_scores.mean():.4f}+/-{cv_scores.std():.4f}")

df_nest = pd.DataFrame(results_nest)

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(df_nest["n_estimators"], df_nest["accuracy_train"], "o-",
        label="Acurácia (Treino)", linewidth=2, markersize=8, color="#1f77b4")
ax.plot(df_nest["n_estimators"], df_nest["accuracy_test"],  "s-",
        label="Acurácia (Teste)",  linewidth=2, markersize=8, color="#ff7f0e")
ax.plot(df_nest["n_estimators"], df_nest["cv_mean"],        "^--",
        label="CV (Média)",       linewidth=2, markersize=8, color="#2ca02c")
ax.fill_between(df_nest["n_estimators"],
    df_nest["cv_mean"] - df_nest["cv_std"],
    df_nest["cv_mean"] + df_nest["cv_std"],
    alpha=0.15, color="#2ca02c", label="CV ± 1 std")
ax.set_xlabel("n_estimators (número de árvores)")
ax.set_ylabel("Acurácia")
ax.set_title("Random Forest: Impacto do Número de Árvores\n"
             "(poucas árvores → instável; muitas → retorno marginal decrescente)",
             fontsize=13, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_variacao_n_estimators.png", dpi=150)
plt.close()
print("  -> Gráfico salvo: 05_variacao_n_estimators.png")

# --- 5.2 max_depth (profundidade máxima de cada árvore) ---
# Sem restrição (None): cada árvore cresce até separar perfeitamente o treino
# → overfitting individual, mas o ensemble compensa pela votação majoritária.
# Profundidades menores regularizam mais cada árvore.
print("\n--- 5.2 Random Forest: variação de max_depth ---")
depth_values = [2, 4, 6, 8, 10, 15, 20, None]
results_depth = []

for depth in depth_values:
    model = RandomForestClassifier(n_estimators=100, max_depth=depth,
                                   random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    acc_tr = accuracy_score(y_train, model.predict(X_train))
    acc_te = accuracy_score(y_test,  model.predict(X_test))
    f1  = f1_score(y_test, model.predict(X_test), average="weighted")
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    label = str(depth) if depth is not None else "None"
    results_depth.append({
        "max_depth": label, "accuracy_train": acc_tr, "accuracy_test": acc_te,
        "f1_weighted_test": f1, "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std()
    })
    print(f"  max_depth={label:>4} | TrainAcc={acc_tr:.4f} | TestAcc={acc_te:.4f} | "
          f"F1={f1:.4f} | CV={cv_scores.mean():.4f}+/-{cv_scores.std():.4f}")

df_depth = pd.DataFrame(results_depth)

fig, ax = plt.subplots(figsize=(11, 5))
x_pos = np.arange(len(df_depth))
ax.plot(x_pos, df_depth["accuracy_train"], "o-", label="Acurácia (Treino)", linewidth=2, markersize=8, color="#1f77b4")
ax.plot(x_pos, df_depth["accuracy_test"],  "s-", label="Acurácia (Teste)",  linewidth=2, markersize=8, color="#ff7f0e")
ax.plot(x_pos, df_depth["cv_mean"],         "^--", label="CV (Média)",      linewidth=2, markersize=8, color="#2ca02c")
ax.fill_between(x_pos,
    df_depth["cv_mean"] - df_depth["cv_std"],
    df_depth["cv_mean"] + df_depth["cv_std"],
    alpha=0.15, color="#2ca02c", label="CV ± 1 std")
ax.set_xticks(x_pos)
ax.set_xticklabels(df_depth["max_depth"])
ax.set_xlabel("max_depth")
ax.set_ylabel("Acurácia")
ax.set_title("Random Forest: Impacto da Profundidade Máxima das Árvores",
             fontsize=13, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_variacao_max_depth.png", dpi=150)
plt.close()
print("  -> Gráfico salvo: 06_variacao_max_depth.png")

# --- 5.3 max_features (features consideradas em cada split) ---
# Controla a diversidade entre as árvores:
#   'sqrt'  → raiz quadrada do total (padrão para classificação) — mais diversidade
#   'log2'  → log2 do total — ainda menos features por split
#   None    → todas as features → árvores similares, menos diversidade no ensemble
# Menos features por split = árvores mais diversas = ensemble mais robusto.
print("\n--- 5.3 Random Forest: variação de max_features ---")
mf_options = ["sqrt", "log2", None, 0.3, 0.5, 0.8]
results_mf = []

for mf in mf_options:
    label = str(mf) if mf is not None else "None (todas)"
    model = RandomForestClassifier(n_estimators=100, max_features=mf,
                                   random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    acc_tr = accuracy_score(y_train, model.predict(X_train))
    acc_te = accuracy_score(y_test,  model.predict(X_test))
    f1   = f1_score(y_test, model.predict(X_test), average="weighted")
    prec = precision_score(y_test, model.predict(X_test), average="weighted", zero_division=0)
    rec  = recall_score(y_test, model.predict(X_test), average="weighted")
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    n_feats = int(np.sqrt(X_train.shape[1])) if mf == "sqrt" else \
              int(np.log2(X_train.shape[1])) if mf == "log2" else \
              int(mf * X_train.shape[1]) if isinstance(mf, float) else X_train.shape[1]
    results_mf.append({
        "max_features": label, "n_feats_por_split": n_feats,
        "accuracy_train": acc_tr, "accuracy_test": acc_te,
        "Precision": prec, "Recall": rec,
        "F1-Score": f1, "CV Mean": cv_scores.mean(), "CV Std": cv_scores.std()
    })
    print(f"  max_features={label:<14} (~{n_feats} feats) | TrainAcc={acc_tr:.4f} | "
          f"TestAcc={acc_te:.4f} | F1={f1:.4f} | CV={cv_scores.mean():.4f}+/-{cv_scores.std():.4f}")

df_mf = pd.DataFrame(results_mf)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
x_pos = np.arange(len(df_mf))
width = 0.28
ax.bar(x_pos - width, df_mf["accuracy_test"], width, label="Accuracy (Teste)", color="#4C72B0")
ax.bar(x_pos,         df_mf["F1-Score"],      width, label="F1-Score",         color="#55A868")
ax.bar(x_pos + width, df_mf["CV Mean"],       width, label="CV Mean",          color="#C44E52")
ax.errorbar(x_pos + width, df_mf["CV Mean"], yerr=df_mf["CV Std"],
            fmt="none", color="black", capsize=4, linewidth=1.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(df_mf["max_features"], rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Score")
ax.set_title("Acurácia por max_features", fontsize=12, fontweight="bold")
ax.legend(fontsize=8)
ax.set_ylim(0.4, 0.85)

ax = axes[1]
ax.bar(x_pos, df_mf["n_feats_por_split"], color="#9467bd", alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(df_mf["max_features"], rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Features usadas por split")
ax.set_title("Nº de features por split\n(menos = árvores mais diversas)",
             fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")

plt.suptitle("Random Forest: Impacto do max_features (diversidade do ensemble)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_variacao_max_features.png", dpi=150)
plt.close()
print("  -> Gráfico salvo: 07_variacao_max_features.png")

# =============================================================================
# 6. MELHOR MODELO - ANÁLISE DETALHADA
# =============================================================================
print("\n" + "=" * 70)
print("6. MELHOR MODELO RANDOM FOREST - ANÁLISE DETALHADA")
print("=" * 70)

# Usa o melhor n_estimators e max_depth encontrados por CV
best_nest_row  = df_nest.loc[df_nest["cv_mean"].idxmax()]
best_depth_row = df_depth.loc[df_depth["cv_mean"].idxmax()]
best_mf_row    = df_mf.loc[df_mf["CV Mean"].idxmax()]

best_n     = int(best_nest_row["n_estimators"])
best_depth = None if best_depth_row["max_depth"] == "None" else int(best_depth_row["max_depth"])
best_mf_str = best_mf_row["max_features"]
if best_mf_str == "None (todas)":
    best_mf = None
elif best_mf_str in ("sqrt", "log2"):
    best_mf = best_mf_str
else:
    best_mf = float(best_mf_str)

print(f"\nMelhores hiperparâmetros encontrados:")
print(f"  n_estimators = {best_n}  (CV={best_nest_row['cv_mean']:.4f})")
print(f"  max_depth    = {best_depth_row['max_depth']}  (CV={best_depth_row['cv_mean']:.4f})")
print(f"  max_features = {best_mf_str}  (CV={best_mf_row['CV Mean']:.4f})")

best_rf = RandomForestClassifier(
    n_estimators=best_n, max_depth=best_depth, max_features=best_mf,
    random_state=42, n_jobs=-1
)
best_rf.fit(X_train, y_train)
y_pred_best = best_rf.predict(X_test)

print("\nClassification Report (melhor Random Forest):")
print(classification_report(y_test, y_pred_best, target_names=class_names, zero_division=0))

# Confusion Matrix normalizada
fig, ax = plt.subplots(figsize=(9, 7))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_best, display_labels=class_names, ax=ax,
    cmap="Blues", colorbar=False, normalize="true"
)
ax.set_title(
    f"Matriz de Confusão Normalizada - Random Forest\n"
    f"(n={best_n}, max_depth={best_depth_row['max_depth']}, max_features={best_mf_str})",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_cm_melhor_modelo_normalizada.png", dpi=150)
plt.close()
print("  -> Gráfico salvo: 08_cm_melhor_modelo_normalizada.png")

# Métricas por classe + importância das features
prec_pc, rec_pc, f1_pc, sup_pc = precision_recall_fscore_support(
    y_test, y_pred_best, zero_division=0
)
df_perclass = pd.DataFrame({
    "Classe": class_names, "Precision": prec_pc,
    "Recall": rec_pc, "F1-Score": f1_pc, "Suporte": sup_pc
})

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
x = np.arange(len(class_names))
w = 0.25
ax.bar(x - w, df_perclass["Precision"], w, label="Precision")
ax.bar(x,     df_perclass["Recall"],    w, label="Recall")
ax.bar(x + w, df_perclass["F1-Score"],  w, label="F1-Score")
ax.set_xticks(x)
ax.set_xticklabels(class_names)
ax.set_xlabel("Qualidade do Vinho")
ax.set_ylabel("Score")
ax.set_title("Métricas por Classe", fontsize=12, fontweight="bold")
ax.legend()
ax.set_ylim(0, 1.05)
for i, sup in enumerate(df_perclass["Suporte"]):
    ax.text(i, 0.02, f"n={sup}", ha="center", fontsize=9, color="gray")

ax = axes[1]
importances = best_rf.feature_importances_
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)
colors = ["#e74c3c" if v >= feat_imp.quantile(0.75) else "#4C72B0" for v in feat_imp]
feat_imp.plot(kind="barh", ax=ax, color=colors)
ax.set_title("Importância das Features (Random Forest)\n(vermelho = top 25%)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Importância média (Gini)")
ax.grid(True, alpha=0.3, axis="x")

plt.suptitle("Melhor Random Forest — Análise Detalhada", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_metricas_e_importancia.png", dpi=150)
plt.close()
print("  -> Gráfico salvo: 09_metricas_e_importancia.png")

# =============================================================================
# 7. ANÁLISE DE OVERFITTING / UNDERFITTING
# =============================================================================
print("\n" + "=" * 70)
print("7. ANÁLISE DE OVERFITTING / UNDERFITTING")
print("=" * 70)

train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(n_estimators=best_n, max_depth=best_depth,
                           max_features=best_mf, random_state=42, n_jobs=-1),
    X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring="accuracy", n_jobs=-1
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Treino",    linewidth=2)
ax.plot(train_sizes, val_scores.mean(axis=1),   "s-", label="Validação", linewidth=2)
ax.fill_between(train_sizes,
    train_scores.mean(axis=1) - train_scores.std(axis=1),
    train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
ax.fill_between(train_sizes,
    val_scores.mean(axis=1) - val_scores.std(axis=1),
    val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
ax.set_xlabel("Tamanho do conjunto de treino")
ax.set_ylabel("Acurácia")
ax.set_title(
    f"Curva de Aprendizado - Random Forest\n"
    f"(n={best_n}, max_depth={best_depth_row['max_depth']}, max_features={best_mf_str})",
    fontsize=13, fontweight="bold"
)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/10_learning_curve.png", dpi=150)
plt.close()

train_acc = accuracy_score(y_train, best_rf.predict(X_train))
test_acc  = accuracy_score(y_test,  y_pred_best)
print(f"  Acurácia no treino:   {train_acc:.4f}")
print(f"  Acurácia no teste:    {test_acc:.4f}")
print(f"  Gap (treino - teste): {train_acc - test_acc:.4f}")
if train_acc - test_acc > 0.10:
    print("  -> Indicativo de OVERFITTING moderado/forte")
elif train_acc - test_acc > 0.05:
    print("  -> Indicativo de leve OVERFITTING")
elif test_acc < 0.5:
    print("  -> Indicativo de UNDERFITTING")
else:
    print("  -> Modelo com boa generalização")
print("  -> Gráfico salvo: 10_learning_curve.png")

print("\n--- Demonstração: poucas árvores (instável) vs muitas (estável) ---")
for n_demo in [5, 10, best_n, 400]:
    model = RandomForestClassifier(n_estimators=n_demo, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    acc_tr = accuracy_score(y_train, model.predict(X_train))
    acc_te = accuracy_score(y_test,  model.predict(X_test))
    print(f"  n_estimators={n_demo:>3} | TrainAcc={acc_tr:.4f} | TestAcc={acc_te:.4f} | Gap={acc_tr-acc_te:+.4f}")

# =============================================================================
# 8. TABELA RESUMO FINAL
# =============================================================================
print("\n" + "=" * 70)
print("8. TABELA RESUMO FINAL")
print("=" * 70)

print("\n--- Resumo: variação de n_estimators ---")
print(df_nest.to_string(index=False, float_format="%.4f"))

print("\n--- Resumo: variação de max_depth ---")
print(df_depth.to_string(index=False, float_format="%.4f"))

print("\n--- Resumo: variação de max_features ---")
print(df_mf.to_string(index=False, float_format="%.4f"))

df_nest.to_csv(f"{OUTPUT_DIR}/tabela_variacao_n_estimators.csv", index=False)
df_depth.to_csv(f"{OUTPUT_DIR}/tabela_variacao_max_depth.csv", index=False)
df_mf.to_csv(f"{OUTPUT_DIR}/tabela_variacao_max_features.csv", index=False)
df_perclass.to_csv(f"{OUTPUT_DIR}/tabela_metricas_por_classe.csv", index=False)

print("\n" + "=" * 70)
print("EXECUÇÃO FINALIZADA! Resultados salvos em:", OUTPUT_DIR)
print("=" * 70)
print("\nArquivos gerados:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  {f}")