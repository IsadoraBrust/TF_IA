"""
=============================================================================
Trabalho Final - Inteligência Artificial (PUCRS)
Dataset: Estimation of Obesity Levels Based On Eating Habits and Physical Condition
Modelo: Decision Tree (Árvore de Decisão)

Antes de rodar, instale as dependências:
    pip install ucimlrepo scikit-learn pandas numpy matplotlib seaborn
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
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

# Árvore de Decisão não requer escalonamento, mas mantemos para consistência
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Nota: Árvores de Decisão são invariantes ao escalonamento de features.")
print("StandardScaler aplicado apenas para manter consistência com os demais modelos.")

feature_names = list(X_encoded.columns)

# =============================================================================
# 4. DECISION TREE - MODELO BASE
# =============================================================================
print("\n" + "=" * 70)
print("4. DECISION TREE - MODELO BASE (criterion=gini, sem restrição de profundidade)")
print("=" * 70)

dt_base = DecisionTreeClassifier(random_state=42)
dt_base.fit(X_train, y_train)
y_pred_base = dt_base.predict(X_test)

acc_base = accuracy_score(y_test, y_pred_base)
print(f"\nProfundidade da árvore: {dt_base.get_depth()}")
print(f"Número de folhas:       {dt_base.get_n_leaves()}")
print(f"Acurácia no teste:      {acc_base:.4f}")
print(f"Acurácia no treino:     {accuracy_score(y_train, dt_base.predict(X_train)):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_base, target_names=class_names))

fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_base, display_labels=class_names, ax=ax,
    cmap="Blues", xticks_rotation=45, colorbar=False
)
ax.set_title("Matriz de Confusão - Decision Tree Base (sem restrição)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_cm_dt_base.png", dpi=150)
plt.close()
print("  -> Gráfico salvo: 04_cm_dt_base.png")

# =============================================================================
# 5. VARIAÇÕES DE HIPERPARÂMETROS (mínimo 3 variações)
# =============================================================================
print("\n" + "=" * 70)
print("5. VARIAÇÕES DE HIPERPARÂMETROS")
print("=" * 70)

# --- 5.1 max_depth ---
print("\n--- 5.1 Decision Tree: variação de max_depth ---")
depth_values = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, None]
results_depth = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for depth in depth_values:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred_tr = model.predict(X_train)
    y_pred_te = model.predict(X_test)
    acc_tr = accuracy_score(y_train, y_pred_tr)
    acc_te = accuracy_score(y_test, y_pred_te)
    f1 = f1_score(y_test, y_pred_te, average="weighted")
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    n_leaves = model.get_n_leaves()
    results_depth.append({
        "max_depth": str(depth) if depth is not None else "None",
        "accuracy_train": acc_tr,
        "accuracy_test": acc_te,
        "f1_weighted_test": f1,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "n_leaves": n_leaves
    })
    depth_label = str(depth) if depth is not None else "None"
    print(f"  max_depth={depth_label:>4} | TrainAcc={acc_tr:.4f} | TestAcc={acc_te:.4f} | "
          f"F1={f1:.4f} | CV={cv_scores.mean():.4f}+/-{cv_scores.std():.4f} | folhas={n_leaves}")

df_depth = pd.DataFrame(results_depth)

fig, ax = plt.subplots(figsize=(12, 5))
x_labels = df_depth["max_depth"].tolist()
x_pos = np.arange(len(x_labels))
ax.plot(x_pos, df_depth["accuracy_train"], "o-", label="Acurácia (Treino)", linewidth=2, markersize=8, color="#1f77b4")
ax.plot(x_pos, df_depth["accuracy_test"],  "s-", label="Acurácia (Teste)",  linewidth=2, markersize=8, color="#ff7f0e")
ax.plot(x_pos, df_depth["cv_mean"],        "^:", label="CV Accuracy (Média)", linewidth=2, markersize=8, color="#2ca02c")
ax.fill_between(x_pos,
    df_depth["cv_mean"] - df_depth["cv_std"],
    df_depth["cv_mean"] + df_depth["cv_std"],
    alpha=0.15, color="#2ca02c", label="CV ± 1 std")
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels)
ax.set_xlabel("max_depth")
ax.set_ylabel("Score")
ax.set_title("Decision Tree: Impacto da Profundidade Máxima\n(Overfitting em profundidades altas)",
             fontsize=13, fontweight="bold")
ax.legend()
ax.set_ylim(0.3, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_variacao_max_depth.png", dpi=150)
plt.close()
print("  -> Gráfico salvo: 05_variacao_max_depth.png")

# --- 5.2 criterion ---
print("\n--- 5.2 Decision Tree: variação de criterion (função de impureza) ---")
criterions = ["gini", "entropy", "log_loss"]
results_crit = []

for crit in criterions:
    for depth in [3, 5, 7, 10, None]:
        model = DecisionTreeClassifier(criterion=crit, max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        results_crit.append({
            "criterion": crit,
            "max_depth": str(depth) if depth is not None else "None",
            "Accuracy": acc, "Precision": prec, "Recall": rec,
            "F1-Score": f1, "CV Mean": cv_scores.mean(), "CV Std": cv_scores.std()
        })
        depth_label = str(depth) if depth is not None else "None"
        print(f"  criterion={crit:>8} | max_depth={depth_label:>4} | "
              f"Acc={acc:.4f} | F1={f1:.4f} | CV={cv_scores.mean():.4f}+/-{cv_scores.std():.4f}")

df_crit = pd.DataFrame(results_crit)

fig, ax = plt.subplots(figsize=(12, 5))
depth_labels = ["3", "5", "7", "10", "None"]
x_pos = np.arange(len(depth_labels))
colors = {"gini": "#4C72B0", "entropy": "#55A868", "log_loss": "#C44E52"}
width = 0.28
offsets = [-width, 0, width]
for i, crit in enumerate(criterions):
    sub = df_crit[df_crit["criterion"] == crit]
    ax.plot(x_pos + offsets[i], sub["Accuracy"].values, "o-",
            label=f"criterion={crit}", linewidth=2, markersize=8, color=list(colors.values())[i])
ax.set_xticks(x_pos)
ax.set_xticklabels(depth_labels)
ax.set_xlabel("max_depth")
ax.set_ylabel("Acurácia no Teste")
ax.set_title("Decision Tree: Comparação de Critério de Impureza (Gini × Entropy × Log Loss)",
             fontsize=13, fontweight="bold")
ax.legend()
ax.set_ylim(0.3, 1.0)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_comparacao_criterion.png", dpi=150)
plt.close()
print("  -> Gráfico salvo: 06_comparacao_criterion.png")

# --- 5.3 min_samples_split e min_samples_leaf ---
print("\n--- 5.3 Decision Tree: variação de min_samples_split e min_samples_leaf ---")
min_split_values = [2, 5, 10, 20, 50, 100]
min_leaf_values  = [1, 2, 5, 10, 20, 50]
results_min = []

for ms in min_split_values:
    model = DecisionTreeClassifier(min_samples_split=ms, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    results_min.append({
        "parametro": "min_samples_split", "valor": ms,
        "accuracy_test": acc, "f1_weighted_test": f1,
        "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std()
    })
    print(f"  min_samples_split={ms:>3} | Acc={acc:.4f} | F1={f1:.4f} | "
          f"CV={cv_scores.mean():.4f}+/-{cv_scores.std():.4f}")

for ml in min_leaf_values:
    model = DecisionTreeClassifier(min_samples_leaf=ml, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    results_min.append({
        "parametro": "min_samples_leaf", "valor": ml,
        "accuracy_test": acc, "f1_weighted_test": f1,
        "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std()
    })
    print(f"  min_samples_leaf ={ml:>3} | Acc={acc:.4f} | F1={f1:.4f} | "
          f"CV={cv_scores.mean():.4f}+/-{cv_scores.std():.4f}")

df_min = pd.DataFrame(results_min)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, param in zip(axes, ["min_samples_split", "min_samples_leaf"]):
    sub = df_min[df_min["parametro"] == param]
    ax.plot(sub["valor"], sub["accuracy_test"], "o-", label="Acurácia (Teste)",
            linewidth=2, markersize=8, color="#ff7f0e")
    ax.plot(sub["valor"], sub["cv_mean"], "s--", label="CV Accuracy (Média)",
            linewidth=2, markersize=8, color="#2ca02c")
    ax.fill_between(sub["valor"],
        sub["cv_mean"] - sub["cv_std"], sub["cv_mean"] + sub["cv_std"],
        alpha=0.15, color="#2ca02c")
    ax.set_xlabel(param)
    ax.set_ylabel("Score")
    ax.set_title(f"Impacto de {param}", fontsize=12, fontweight="bold")
    ax.legend()
    ax.set_ylim(0.3, 1.0)
    ax.grid(True, alpha=0.3)
plt.suptitle("Decision Tree: Regularização via min_samples", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_variacao_min_samples.png", dpi=150)
plt.close()
print("  -> Gráfico salvo: 07_variacao_min_samples.png")

# =============================================================================
# 6. MELHOR MODELO - ANÁLISE DETALHADA
# =============================================================================
print("\n" + "=" * 70)
print("6. MELHOR MODELO DECISION TREE - ANÁLISE DETALHADA")
print("=" * 70)

best_depth_row = df_depth.loc[df_depth["accuracy_test"].idxmax()]
best_depth_val = None if best_depth_row["max_depth"] == "None" else int(best_depth_row["max_depth"])
print(f"\nMelhor max_depth: {best_depth_row['max_depth']} (Acc={best_depth_row['accuracy_test']:.4f})")

best_dt = DecisionTreeClassifier(max_depth=best_depth_val, random_state=42)
best_dt.fit(X_train, y_train)
y_pred_best = best_dt.predict(X_test)

print(f"Profundidade real da árvore: {best_dt.get_depth()}")
print(f"Número de folhas:            {best_dt.get_n_leaves()}")
print("\nClassification Report (melhor Decision Tree):")
print(classification_report(y_test, y_pred_best, target_names=class_names))

# Confusion Matrix normalizada
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_best, display_labels=class_names, ax=ax,
    cmap="Blues", xticks_rotation=45, colorbar=False, normalize="true"
)
ax.set_title(
    f"Matriz de Confusão Normalizada - Decision Tree (max_depth={best_depth_row['max_depth']})",
    fontsize=13, fontweight="bold"
)
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

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Métricas por classe (barras)
ax = axes[0]
x = np.arange(len(class_names))
w = 0.25
ax.bar(x - w, df_perclass["Precision"], w, label="Precision")
ax.bar(x,     df_perclass["Recall"],    w, label="Recall")
ax.bar(x + w, df_perclass["F1-Score"],  w, label="F1-Score")
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Score")
ax.set_title("Métricas por Classe", fontsize=12, fontweight="bold")
ax.legend()
ax.set_ylim(0, 1.1)

# Importância das features
ax = axes[1]
importances = best_dt.feature_importances_
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)
feat_imp.plot(kind="barh", ax=ax, color="#4C72B0")
ax.set_title("Importância das Features", fontsize=12, fontweight="bold")
ax.set_xlabel("Importância (Gini)")
ax.grid(True, alpha=0.3, axis="x")

plt.suptitle(f"Melhor Decision Tree (max_depth={best_depth_row['max_depth']})",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_metricas_e_importancia.png", dpi=150)
plt.close()
print("  -> Gráfico salvo: 09_metricas_e_importancia.png")

# =============================================================================
# 7. ANÁLISE DE OVERFITTING/UNDERFITTING
# =============================================================================
print("\n" + "=" * 70)
print("7. ANÁLISE DE OVERFITTING / UNDERFITTING")
print("=" * 70)

train_sizes, train_scores, val_scores = learning_curve(
    DecisionTreeClassifier(max_depth=best_depth_val, random_state=42),
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
ax.set_title(f"Curva de Aprendizado - Decision Tree (max_depth={best_depth_row['max_depth']})",
             fontsize=14, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/10_learning_curve.png", dpi=150)
plt.close()

train_acc = accuracy_score(y_train, best_dt.predict(X_train))
test_acc  = accuracy_score(y_test,  y_pred_best)
print(f"  Acurácia no treino:   {train_acc:.4f}")
print(f"  Acurácia no teste:    {test_acc:.4f}")
print(f"  Gap (treino - teste): {train_acc - test_acc:.4f}")
if train_acc - test_acc > 0.10:
    print("  -> Indicativo de OVERFITTING moderado/forte")
elif train_acc - test_acc > 0.05:
    print("  -> Indicativo de leve OVERFITTING")
elif test_acc < 0.6:
    print("  -> Indicativo de UNDERFITTING")
else:
    print("  -> Modelo com boa generalização (gap pequeno)")
print("  -> Gráfico salvo: 10_learning_curve.png")

print("\n--- Demonstração: overfitting (profundidade alta) vs underfitting (profundidade baixa) ---")
for depth_demo in [1, 3, 5, best_depth_val, None]:
    model = DecisionTreeClassifier(max_depth=depth_demo, random_state=42)
    model.fit(X_train, y_train)
    acc_tr = accuracy_score(y_train, model.predict(X_train))
    acc_te = accuracy_score(y_test,  model.predict(X_test))
    label  = str(depth_demo) if depth_demo is not None else "None"
    print(f"  max_depth={label:>4} | TrainAcc={acc_tr:.4f} | TestAcc={acc_te:.4f} | "
          f"Gap={acc_tr - acc_te:+.4f} | folhas={model.get_n_leaves()}")

# =============================================================================
# 8. TABELA RESUMO FINAL
# =============================================================================
print("\n" + "=" * 70)
print("8. TABELA RESUMO FINAL")
print("=" * 70)

print("\n--- Resumo: variação de max_depth ---")
print(df_depth.to_string(index=False, float_format="%.4f"))

print("\n--- Resumo: comparação de criterion ---")
print(df_crit.to_string(index=False, float_format="%.4f"))

print("\n--- Resumo: variação de min_samples ---")
print(df_min.to_string(index=False, float_format="%.4f"))

df_depth.to_csv(f"{OUTPUT_DIR}/tabela_variacao_max_depth.csv", index=False)
df_crit.to_csv(f"{OUTPUT_DIR}/tabela_comparacao_criterion.csv", index=False)
df_min.to_csv(f"{OUTPUT_DIR}/tabela_variacao_min_samples.csv", index=False)
df_perclass.to_csv(f"{OUTPUT_DIR}/tabela_metricas_por_classe.csv", index=False)

print("\n" + "=" * 70)
print("EXECUÇÃO FINALIZADA! Resultados salvos em:", OUTPUT_DIR)
print("=" * 70)
print("\nArquivos gerados:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  {f}")
