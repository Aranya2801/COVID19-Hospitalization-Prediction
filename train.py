#!/usr/bin/env python3
"""
COVID-19 Hospitalization Prediction — Full Training Pipeline
=============================================================
Run: python train.py --data data/raw/covid19_hospitalization_data.csv
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
from sklearn.calibration import calibration_curve

sys.path.insert(0, str(Path(__file__).parent))
from src.models.predictor import COVID19HospitalizationPredictor


# ── Styling ──────────────────────────────────────────────────────────────
PALETTE = {
    'primary':   '#0D47A1',
    'secondary': '#1565C0',
    'accent':    '#FF6F00',
    'danger':    '#B71C1C',
    'success':   '#1B5E20',
    'bg':        '#F8F9FA',
}

def set_style():
    plt.rcParams.update({
        'figure.facecolor': PALETTE['bg'],
        'axes.facecolor':   PALETTE['bg'],
        'axes.edgecolor':   '#CCCCCC',
        'axes.grid':        True,
        'grid.color':       '#DDDDDD',
        'grid.linestyle':   '--',
        'grid.alpha':       0.6,
        'font.family':      'DejaVu Sans',
        'font.size':        11,
    })

set_style()


def plot_roc_pr(metrics: dict, save_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Model Performance — ROC & Precision-Recall Curves',
                 fontsize=15, fontweight='bold', y=1.02)

    # ROC
    ax = axes[0]
    fpr = metrics['roc_curve']['fpr']
    tpr = metrics['roc_curve']['tpr']
    auc = metrics['roc_auc']
    ax.plot(fpr, tpr, color=PALETTE['primary'], lw=2.5,
            label=f'Ensemble (AUC = {auc:.4f})')
    ax.plot([0,1],[0,1],'--', color='gray', alpha=0.7, label='Random')
    ax.fill_between(fpr, tpr, alpha=0.08, color=PALETTE['primary'])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)

    # PR
    ax = axes[1]
    pr  = metrics['pr_curve']['precision']
    rec = metrics['pr_curve']['recall']
    ap  = metrics['avg_precision']
    ax.plot(rec, pr, color=PALETTE['accent'], lw=2.5,
            label=f'Ensemble (AP = {ap:.4f})')
    ax.fill_between(rec, pr, alpha=0.08, color=PALETTE['accent'])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)

    plt.tight_layout()
    path = save_dir / 'roc_pr_curves.png'
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_calibration(metrics: dict, save_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    fp = metrics['calibration_curve']['frac_pos']
    mp = metrics['calibration_curve']['mean_pred']
    ax.plot([0,1],[0,1],'--', color='gray', lw=1.5, label='Perfect calibration')
    ax.plot(mp, fp, 'o-', color=PALETTE['primary'], lw=2.5,
            markersize=8, label='Calibrated Ensemble')
    ax.fill_between(mp, fp, mp, alpha=0.08, color=PALETTE['danger'])
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title('Probability Calibration Curve', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    brier = metrics['brier_score']
    ax.text(0.05, 0.92, f'Brier Score: {brier:.4f}',
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    path = save_dir / 'calibration_curve.png'
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_importance(model: COVID19HospitalizationPredictor,
                             df_val: pd.DataFrame, save_dir: Path):
    explanation = model.explain(df_val.head(500))
    features = [f for f, _ in explanation['top_features'][:20]]
    importances = [v for _, v in explanation['top_features'][:20]]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [PALETTE['danger'] if i < 5 else PALETTE['primary']
              if i < 10 else PALETTE['secondary'] for i in range(len(features))]
    bars = ax.barh(range(len(features)), importances, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
    ax.set_title('Top 20 Features — SHAP Global Importance',
                 fontsize=13, fontweight='bold')
    for bar, val in zip(bars, importances):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    path = save_dir / 'shap_feature_importance.png'
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_risk_distribution(df: pd.DataFrame, probs: np.ndarray, save_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Probability histogram
    ax = axes[0]
    ax.hist(probs[df['hospitalized']==0], bins=40, alpha=0.65,
            color=PALETTE['success'], label='Not Hospitalized', density=True)
    ax.hist(probs[df['hospitalized']==1], bins=40, alpha=0.65,
            color=PALETTE['danger'], label='Hospitalized', density=True)
    ax.axvline(0.40, color=PALETTE['accent'], linestyle='--', lw=2, label='Decision threshold (0.40)')
    ax.set_xlabel('Predicted Hospitalization Probability', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Risk Score Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

    # Risk tier pie
    ax = axes[1]
    tiers = pd.cut(probs, bins=[0, 0.20, 0.40, 0.65, 1.01],
                   labels=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])
    counts = tiers.value_counts()
    colors_pie = [PALETTE['success'], '#FFC107', PALETTE['accent'], PALETTE['danger']]
    wedges, texts, autotexts = ax.pie(
        counts, labels=counts.index, autopct='%1.1f%%',
        colors=colors_pie, startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight('bold')
    ax.set_title('Risk Tier Distribution', fontsize=13, fontweight='bold')

    plt.tight_layout()
    path = save_dir / 'risk_distribution.png'
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(metrics: dict, save_dir: Path):
    cm = np.array(metrics['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not Hospitalized', 'Hospitalized'],
                yticklabels=['Not Hospitalized', 'Hospitalized'],
                linewidths=0.5, linecolor='white',
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix (Threshold = 0.40)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = save_dir / 'confusion_matrix.png'
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='COVID-19 Hospitalization Prediction Training')
    parser.add_argument('--data',    default='data/raw/covid19_hospitalization_data.csv')
    parser.add_argument('--target',  default='hospitalized')
    parser.add_argument('--val',     type=float, default=0.20)
    parser.add_argument('--outdir',  default='models/saved')
    parser.add_argument('--plots',   default='docs/images')
    args = parser.parse_args()

    plot_dir = Path(args.plots)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading data: {args.data}")
    df = pd.read_csv(args.data)
    print(f"Shape: {df.shape} | Hospitalized: {df[args.target].mean():.1%}\n")

    model = COVID19HospitalizationPredictor(model_dir=args.outdir)
    model.fit(df, target=args.target, validation_split=args.val)

    # Evaluate on a held-out test split
    from sklearn.model_selection import train_test_split
    _, df_test = train_test_split(df, test_size=0.15,
                                  stratify=df[args.target], random_state=42)
    metrics = model.evaluate(df_test, target=args.target)

    print("\nGenerating plots...")
    probs = model.predict_proba(df_test)[:, 1]
    plot_roc_pr(metrics, plot_dir)
    plot_calibration(metrics, plot_dir)
    plot_feature_importance(model, df_test, plot_dir)
    plot_risk_distribution(df_test, probs, plot_dir)
    plot_confusion_matrix(metrics, plot_dir)

    model.save("covid19_predictor")

    # Save metrics
    clean_metrics = {k: v for k, v in metrics.items()
                     if k not in ('roc_curve','pr_curve','calibration_curve')}
    with open(Path(args.outdir) / 'evaluation_metrics.json', 'w') as f:
        json.dump(clean_metrics, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  Training complete!")
    print(f"  Final AUC : {metrics['roc_auc']:.4f}")
    print(f"  Avg-Prec  : {metrics['avg_precision']:.4f}")
    print(f"  Brier     : {metrics['brier_score']:.4f}")
    print(f"{'='*55}\n")


if __name__ == '__main__':
    main()
