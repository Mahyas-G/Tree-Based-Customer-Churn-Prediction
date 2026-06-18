import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_preprocess(path: str = "Telco-Customer-Churn.csv"):
    print("=" * 65)
    print("  Step 1: Data Preparation")
    print("=" * 65)

    df = pd.read_csv(path)
    
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    df = df.fillna(df.median(numeric_only=True))
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    df.drop(columns=["customerID"], inplace=True)
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    assert X_train.isnull().sum().sum() == 0
    assert X_test.isnull().sum().sum() == 0

    return X_train, X_test, y_train, y_test, feature_names

def decision_tree_analysis(X_train, X_test, y_train, y_test, feature_names):
    print("\n" + "=" * 65)
    print("  Step 2: Decision Tree")
    print("=" * 65)

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, dt.predict_proba(X_test)[:, 1])
    print(f"\n[Default Tree]  Accuracy: {acc:.4f}  |  AUC: {auc:.4f}")
    print(f"Tree Depth: {dt.get_depth()}  |  Number of Leaves: {dt.get_n_leaves()}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Stay", "Churn"]))

    fig, ax = plt.subplots(figsize=(22, 8))
    plot_tree(
        dt, max_depth=3, feature_names=feature_names,
        class_names=["Stay", "Churn"], filled=True,
        impurity=True, rounded=True, fontsize=9, ax=ax
    )
    ax.set_title("Decision Tree Structure (First 3 Levels)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "decision_tree_structure.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[✓] Tree structure image saved: decision_tree_structure.png")

    depths = range(1, 21)
    train_accs, cv_accs = [], []
    for d in depths:
        m = DecisionTreeClassifier(max_depth=d, random_state=42)
        m.fit(X_train, y_train)
        train_accs.append(accuracy_score(y_train, m.predict(X_train)))
        cv_accs.append(cross_val_score(m, X_train, y_train, cv=5, scoring="accuracy").mean())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(depths, train_accs, "o-", label="Train", color="#2196F3", linewidth=2)
    ax.plot(depths, cv_accs, "s-", label="CV", color="#F44336", linewidth=2)
    best_d = depths[np.argmax(cv_accs)]
    ax.axvline(best_d, color="green", linestyle="--", alpha=0.7,
               label=f"Best Depth = {best_d}")
    ax.set_xlabel("max_depth", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Effect of max_depth on Decision Tree Performance", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "max_depth_effect.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] max_depth plot saved  |  Best Depth: {best_d}")

    return dt, best_d

def tree_pruning_analysis(X_train, X_test, y_train, y_test, best_depth):
    print("\n" + "=" * 65)
    print("  Step 3: Tree Pruning")
    print("=" * 65)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    split_vals = [2, 5, 10, 20, 50, 100]
    split_train, split_cv = [], []
    for v in split_vals:
        m = DecisionTreeClassifier(max_depth=best_depth, min_samples_split=v, random_state=42)
        m.fit(X_train, y_train)
        split_train.append(accuracy_score(y_train, m.predict(X_train)))
        split_cv.append(cross_val_score(m, X_train, y_train, cv=5, scoring="accuracy").mean())

    axes[0].plot(split_vals, split_train, "o-", color="#2196F3", label="Train", linewidth=2)
    axes[0].plot(split_vals, split_cv, "s-", color="#F44336", label="CV", linewidth=2)
    axes[0].set_xlabel("min_samples_split", fontsize=11)
    axes[0].set_ylabel("Accuracy", fontsize=11)
    axes[0].set_title("Effect of min_samples_split", fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    leaf_vals = [1, 2, 5, 10, 20, 50]
    leaf_train, leaf_cv = [], []
    for v in leaf_vals:
        m = DecisionTreeClassifier(max_depth=best_depth, min_samples_leaf=v, random_state=42)
        m.fit(X_train, y_train)
        leaf_train.append(accuracy_score(y_train, m.predict(X_train)))
        leaf_cv.append(cross_val_score(m, X_train, y_train, cv=5, scoring="accuracy").mean())

    axes[1].plot(leaf_vals, leaf_train, "o-", color="#2196F3", label="Train", linewidth=2)
    axes[1].plot(leaf_vals, leaf_cv, "s-", color="#F44336", label="CV", linewidth=2)
    axes[1].set_xlabel("min_samples_leaf", fontsize=11)
    axes[1].set_ylabel("Accuracy", fontsize=11)
    axes[1].set_title("Effect of min_samples_leaf", fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Decision Tree Pruning", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pruning_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[✓] Pruning plot saved: pruning_analysis.png")

    best_score = 0
    best_split = split_vals[0]
    best_leaf = leaf_vals[0]
    
    for split in split_vals:
        for leaf in leaf_vals:
            m = DecisionTreeClassifier(
                max_depth=best_depth,
                min_samples_split=split,
                min_samples_leaf=leaf,
                random_state=42
            )
            score = cross_val_score(m, X_train, y_train, cv=5, scoring="accuracy").mean()
            if score > best_score:
                best_score = score
                best_split = split
                best_leaf = leaf

    pruned_dt = DecisionTreeClassifier(
        max_depth=best_depth,
        min_samples_split=best_split,
        min_samples_leaf=best_leaf,
        random_state=42
    )
    pruned_dt.fit(X_train, y_train)
    
    test_acc = accuracy_score(y_test, pruned_dt.predict(X_test))
    
    print(f"\n[Optimal Pruned Tree]")
    print(f"  max_depth={best_depth} | min_samples_split={best_split} | min_samples_leaf={best_leaf}")
    print(f"  Test Accuracy: {test_acc:.4f}")

    return pruned_dt

def random_forest_analysis(X_train, X_test, y_train, y_test, feature_names):
    print("\n" + "=" * 65)
    print("  Step 4: Random Forest")
    print("=" * 65)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    print(f"\n[Random Forest]  Accuracy: {acc:.4f}  |  AUC: {auc:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Stay','Churn'])}")

    importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=True)
    top15 = importances[-15:]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top15)))
    top15.plot(kind="barh", ax=ax, color=colors)
    ax.set_title("Top 15 Important Features - Random Forest", fontsize=14)
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[✓] Feature importance plot saved: feature_importance.png")

    return rf

def adaboost_analysis(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 65)
    print("  Step 5: AdaBoost")
    print("=" * 65)

    ada = AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=42)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, ada.predict_proba(X_test)[:, 1])
    print(f"\n[AdaBoost]  Accuracy: {acc:.4f}  |  AUC: {auc:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Stay','Churn'])}")

    return ada

def compare_models(models: dict, X_train, y_train, X_test, y_test):
    print("\n" + "=" * 65)
    print("  Final Comparison of Three Algorithms")
    print("=" * 65)

    results = {}
    for name, model in models.items():
        y_pred  = model.predict(X_test)
        y_prob  = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "auc":      roc_auc_score(y_test, y_prob),
            "y_pred":   y_pred,
            "y_prob":   y_prob,
        }
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        results[name]["cv_mean"] = cv.mean()
        results[name]["cv_std"]  = cv.std()

    print(f"\n{'Model':<30} {'Accuracy':>8} {'AUC':>8} {'CV Mean':>14}")
    print("-" * 65)
    for name, r in results.items():
        print(f"{name:<30} {r['accuracy']:>8.4f} {r['auc']:>8.4f} "
              f"{r['cv_mean']:>8.4f} ± {r['cv_std']:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    names = list(results.keys())
    accs  = [results[n]["accuracy"] for n in names]
    aucs  = [results[n]["auc"] for n in names]
    x = np.arange(len(names))
    bars1 = ax.bar(x - 0.2, accs, 0.35, label="Accuracy",  color=["#2196F3","#4CAF50","#FF9800"])
    bars2 = ax.bar(x + 0.2, aucs, 0.35, label="AUC",   color=["#1565C0","#2E7D32","#E65100"])
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0.7, 0.95)
    ax.set_ylabel("Value", fontsize=11)
    ax.set_title("Accuracy and AUC Comparison", fontsize=13)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for b in list(bars1) + list(bars2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.002,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    colors_roc = ["#2196F3", "#4CAF50", "#FF9800"]
    for (name, r), c in zip(results.items(), colors_roc):
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        ax.plot(fpr, tpr, color=c, linewidth=2,
                label=f"{name}  (AUC={r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    best_name = max(results, key=lambda n: results[n]["auc"])
    cm = confusion_matrix(y_test, results[best_name]["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stay", "Churn"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix\n({best_name})", fontsize=13)

    plt.suptitle("Comprehensive Comparison of Classification Algorithms", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("\n[✓] Model comparison plot saved: model_comparison.png")

    best_name = max(results, key=lambda n: results[n]["auc"])
    print(f"\n[✓] Best model based on AUC: {best_name}  (AUC = {results[best_name]['auc']:.4f})")

    return results

if __name__ == "__main__":

    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess(
        "Telco-Customer-Churn.csv"
    )

    dt_base, best_depth = decision_tree_analysis(X_train, X_test, y_train, y_test, feature_names)

    dt_pruned = tree_pruning_analysis(X_train, X_test, y_train, y_test, best_depth)

    rf = random_forest_analysis(X_train, X_test, y_train, y_test, feature_names)

    ada = adaboost_analysis(X_train, X_test, y_train, y_test)

    models = {
        "Decision Tree (pruned)": dt_pruned,
        "Random Forest":          rf,
        "AdaBoost":               ada,
    }
    results = compare_models(models, X_train, y_train, X_test, y_test)

    print("\n" + "=" * 65)
    print("  All output files saved:")
    print("    decision_tree_structure.png")
    print("    max_depth_effect.png")
    print("    pruning_analysis.png")
    print("    feature_importance.png")
    print("    model_comparison.png")
    print("=" * 65)
