"""
=============================================================================
Trabalho Final - Inteligência Artificial (PUCRS)
Dataset: Wine Quality (UCI id=186)
Modelo Obrigatório: KNN (K-Nearest Neighbors)

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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay,
    precision_recall_fscore_support
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="colorblind")

OUTPUT_DIR = "output_knn"
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

# Distribuição das classes
fig, ax = plt.subplots(figsize=(10, 5))
order = sorted(pd.Series(y).unique())
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

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features escalonadas com StandardScaler (media=0, std=1)")
print("  -> Essencial para KNN, que usa distancia euclidiana")

# =============================================================================
# 4. KNN - MODELO BASE
# =============================================================================
print("\n" + "=" * 70)
print("4. KNN - MODELO BASE (k=5, weights=uniform, metric=minkowski)")
print("=" * 70)

knn_base = KNeighborsClassifier()
knn_base.fit(X_train_scaled, y_train)
y_pred_base = knn_base.predict(X_test_scaled)

acc_base = accuracy_score(y_test, y_pred_base)
print(f"\nAcuracia no teste: {acc_base:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_base, target_names=class_names, zero_division=0))

fig, ax = plt.subplots(figsize=(9, 7))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_base, display_labels=class_names, ax=ax,
    cmap="Blues", colorbar=False
)
ax.set_title("Matriz de Confusao - KNN Base (k=5)", fontsize=14, fontweight="bold")
ax.set_xlabel("Predicao")
ax.set_ylabel("Real")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_cm_knn_base.png", dpi=150)
plt.close()
print("  -> Grafico salvo: 04_cm_knn_base.png")

# =============================================================================
# 5. VARIACOES DE HIPERPARAMETROS
# =============================================================================
print("\n" + "=" * 70)
print("5. VARIACOES DE HIPERPARAMETROS")
print("=" * 70)

# --- 5.1 n_neighbors ---
print("\n--- 5.1 KNN: variacao de n_neighbors (k) ---")
k_values = [1, 3, 5, 7, 11, 15, 21, 31, 51, 71, 101]
results_k = []
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    acc_tr = accuracy_score(y_train, y_pred_train)
    acc_te = accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test, average="weighted")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="accuracy")
    results_k.append({
        "k": k, "accuracy_train": acc_tr, "accuracy_test": acc_te,
        "f1_weighted_test": f1, "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std()
    })
    print(f"  k={k:>3} | TrainAcc={acc_tr:.4f} | TestAcc={acc_te:.4f} | F1={f1:.4f} | "
          f"CV={cv_scores.mean():.4f}+/-{cv_scores.std():.4f}")

df_k = pd.DataFrame(results_k)

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(df_k["k"], df_k["accuracy_train"], "o-", label="Acuracia (Treino)", linewidth=2,
        markersize=8, color="#1f77b4")
ax.plot(df_k["k"], df_k["accuracy_test"], "s-", label="Acuracia (Teste)", linewidth=2,
        markersize=8, color="#ff7f0e")
ax.plot(df_k["k"], df_k["cv_mean"], "^--", label="CV Accuracy (Media)", linewidth=2,
        markersize=8, color="#2ca02c")
ax.fill_between(df_k["k"],
    df_k["cv_mean"] - df_k["cv_std"], df_k["cv_mean"] + df_k["cv_std"],
    alpha=0.15, color="#2ca02c", label="CV +/- 1 std")
ax.set_xlabel("k (n_neighbors)")
ax.set_ylabel("Acuracia")
ax.set_title("KNN: Impacto do k - Overfitting (k pequeno) vs Underfitting (k grande)",
             fontsize=13, fontweight="bold")
ax.legend(loc="best")
ax.set_xscale("log")
ax.set_xticks(k_values)
ax.set_xticklabels(k_values)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_variacao_k.png", dpi=150)
plt.close()
print("  -> Grafico salvo: 05_variacao_k.png")

# --- 5.2 weights ---
print("\n--- 5.2 KNN: variacao de weights ---")
weights_options = ["uniform", "distance"]
results_w = []
for w in weights_options:
    for k in [3, 5, 11, 21, 51]:
        model = KNeighborsClassifier(n_neighbors=k, weights=w)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="accuracy")
        results_w.append({
            "weights": w, "k": k,
            "accuracy_test": acc, "f1_weighted_test": f1,
            "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std()
        })
        print(f"  weights={w:>8} | k={k:>3} | Acc={acc:.4f} | F1={f1:.4f} | "
              f"CV={cv_scores.mean():.4f}+/-{cv_scores.std():.4f}")

df_w = pd.DataFrame(results_w)

fig, ax = plt.subplots(figsize=(10, 5))
for w in weights_options:
    sub = df_w[df_w["weights"] == w]
    ax.plot(sub["k"], sub["accuracy_test"], "o-", label=f"weights={w}",
            linewidth=2, markersize=10)
ax.set_xlabel("k (n_neighbors)")
ax.set_ylabel("Acuracia no teste")
ax.set_title("KNN: Impacto do parametro weights", fontsize=14, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_variacao_weights.png", dpi=150)
plt.close()
print("  -> Grafico salvo: 06_variacao_weights.png")

# --- 5.3 metric ---
print("\n--- 5.3 KNN: variacao da metrica de distancia ---")
metrics_options = ["euclidean", "manhattan", "chebyshev", "minkowski"]
results_m = []
for m in metrics_options:
    model = KNeighborsClassifier(n_neighbors=11, metric=m)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="accuracy")
    results_m.append({
        "metric": m, "Accuracy": acc, "Precision": prec, "Recall": rec,
        "F1-Score": f1, "CV Mean": cv_scores.mean(), "CV Std": cv_scores.std()
    })
    print(f"  metric={m:>10} | Acc={acc:.4f} | F1={f1:.4f} | "
          f"CV={cv_scores.mean():.4f}+/-{cv_scores.std():.4f}")

df_m = pd.DataFrame(results_m)

fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(df_m))
width = 0.25
ax.bar(x_pos - width, df_m["Accuracy"], width, label="Accuracy", color="#4C72B0")
ax.bar(x_pos, df_m["F1-Score"], width, label="F1-Score", color="#55A868")
ax.bar(x_pos + width, df_m["CV Mean"], width, label="CV Mean", color="#C44E52")
ax.set_xticks(x_pos)
ax.set_xticklabels(df_m["metric"])
ax.set_ylabel("Score")
ax.set_title("KNN: Comparacao de Metricas de Distancia (k=11)",
             fontsize=14, fontweight="bold")
ax.legend()
ax.set_ylim(0.3, 0.8)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_variacao_metric.png", dpi=150)
plt.close()
print("  -> Grafico salvo: 07_variacao_metric.png")

# =============================================================================
# 6. MELHOR MODELO
# =============================================================================
print("\n" + "=" * 70)
print("6. MELHOR MODELO KNN - ANALISE DETALHADA")
print("=" * 70)

best_w_row = df_w.loc[df_w["accuracy_test"].idxmax()]
best_k = int(best_w_row["k"])
best_w = best_w_row["weights"]
print(f"\nMelhor combinacao: k={best_k}, weights={best_w} "
      f"(Acc={best_w_row['accuracy_test']:.4f})")

best_knn = KNeighborsClassifier(n_neighbors=best_k, weights=best_w)
best_knn.fit(X_train_scaled, y_train)
y_pred_best = best_knn.predict(X_test_scaled)

print("\nClassification Report (melhor KNN):")
print(classification_report(y_test, y_pred_best, target_names=class_names, zero_division=0))

fig, ax = plt.subplots(figsize=(9, 7))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_best, display_labels=class_names, ax=ax,
    cmap="Blues", colorbar=False, normalize="true"
)
ax.set_title(f"Matriz de Confusao Normalizada - KNN (k={best_k}, weights={best_w})",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_cm_melhor_modelo_normalizada.png", dpi=150)
plt.close()
print("  -> Grafico salvo: 08_cm_melhor_modelo_normalizada.png")

prec_pc, rec_pc, f1_pc, sup_pc = precision_recall_fscore_support(y_test, y_pred_best, zero_division=0)
df_perclass = pd.DataFrame({
    "Classe": class_names, "Precision": prec_pc,
    "Recall": rec_pc, "F1-Score": f1_pc, "Suporte": sup_pc
})

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(class_names))
w = 0.25
ax.bar(x - w, df_perclass["Precision"], w, label="Precision")
ax.bar(x, df_perclass["Recall"], w, label="Recall")
ax.bar(x + w, df_perclass["F1-Score"], w, label="F1-Score")
ax.set_xticks(x)
ax.set_xticklabels(class_names)
ax.set_xlabel("Qualidade do Vinho")
ax.set_ylabel("Score")
ax.set_title("Metricas por Classe - Melhor KNN", fontsize=14, fontweight="bold")
ax.legend()
ax.set_ylim(0, 1.05)
for i, sup in enumerate(df_perclass["Suporte"]):
    ax.text(i, 0.02, f"n={sup}", ha="center", fontsize=9, color="gray")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_metricas_por_classe.png", dpi=150)
plt.close()
print("  -> Grafico salvo: 09_metricas_por_classe.png")

# =============================================================================
# 7. OVERFITTING / UNDERFITTING
# =============================================================================
print("\n" + "=" * 70)
print("7. ANALISE DE OVERFITTING / UNDERFITTING")
print("=" * 70)

train_sizes, train_scores, val_scores = learning_curve(
    KNeighborsClassifier(n_neighbors=best_k, weights=best_w),
    X_train_scaled, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring="accuracy", n_jobs=-1
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Treino", linewidth=2)
ax.plot(train_sizes, val_scores.mean(axis=1), "s-", label="Validacao", linewidth=2)
ax.fill_between(train_sizes,
    train_scores.mean(axis=1) - train_scores.std(axis=1),
    train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
ax.fill_between(train_sizes,
    val_scores.mean(axis=1) - val_scores.std(axis=1),
    val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
ax.set_xlabel("Tamanho do conjunto de treino")
ax.set_ylabel("Acuracia")
ax.set_title(f"Curva de Aprendizado - KNN (k={best_k}, weights={best_w})",
             fontsize=14, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/10_learning_curve.png", dpi=150)
plt.close()

train_acc = accuracy_score(y_train, best_knn.predict(X_train_scaled))
test_acc = accuracy_score(y_test, y_pred_best)
print(f"  Acuracia no treino:  {train_acc:.4f}")
print(f"  Acuracia no teste:   {test_acc:.4f}")
print(f"  Gap (treino - teste): {train_acc - test_acc:.4f}")
if train_acc - test_acc > 0.10:
    print("  -> Indicativo de OVERFITTING moderado/forte")
elif train_acc - test_acc > 0.05:
    print("  -> Indicativo de leve OVERFITTING")
elif test_acc < 0.5:
    print("  -> Indicativo de UNDERFITTING")
else:
    print("  -> Modelo com boa generalizacao")
print("  -> Grafico salvo: 10_learning_curve.png")

print("\n--- Demonstracao: overfitting (k=1) vs underfitting (k=101) ---")
for k_demo in [1, best_k, 101]:
    model = KNeighborsClassifier(n_neighbors=k_demo, weights=best_w)
    model.fit(X_train_scaled, y_train)
    acc_tr = accuracy_score(y_train, model.predict(X_train_scaled))
    acc_te = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"  k={k_demo:>3} | TrainAcc={acc_tr:.4f} | TestAcc={acc_te:.4f} | "
          f"Gap={acc_tr-acc_te:+.4f}")

# =============================================================================
# 8. TABELAS RESUMO
# =============================================================================
print("\n" + "=" * 70)
print("8. TABELA RESUMO FINAL")
print("=" * 70)

print("\n--- Resumo: variacao de k ---")
print(df_k.to_string(index=False, float_format="%.4f"))
print("\n--- Resumo: variacao de weights ---")
print(df_w.to_string(index=False, float_format="%.4f"))
print("\n--- Resumo: variacao de metrica ---")
print(df_m.to_string(index=False, float_format="%.4f"))

df_k.to_csv(f"{OUTPUT_DIR}/tabela_variacao_k.csv", index=False)
df_w.to_csv(f"{OUTPUT_DIR}/tabela_variacao_weights.csv", index=False)
df_m.to_csv(f"{OUTPUT_DIR}/tabela_variacao_metric.csv", index=False)
df_perclass.to_csv(f"{OUTPUT_DIR}/tabela_metricas_por_classe.csv", index=False)

print("\n" + "=" * 70)
print("EXECUCAO FINALIZADA! Resultados salvos em:", OUTPUT_DIR)
print("=" * 70)
print("\nArquivos gerados:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  {f}")
