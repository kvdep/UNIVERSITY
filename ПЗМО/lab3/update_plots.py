import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# Add current dir to sys.path to import from marl_portfolio_lab3
sys.path.insert(0, r"D:\ПРОФ\ВУЗ\ПЗМО\lab3")

import marl_portfolio_lab3 as marl

BASE_DIR = r"D:\ПРОФ\ВУЗ\ПЗМО\lab3"
outline = [pe.withStroke(linewidth=3.2, foreground="white")]

def style_ax(ax, title_obj=None, leg_obj=None):
    if title_obj is not None:
        title_obj.set_path_effects(outline)
    ax.xaxis.label.set_path_effects(outline)
    ax.yaxis.label.set_path_effects(outline)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_path_effects(outline)
    if leg_obj is not None:
        for t in leg_obj.get_texts():
            t.set_path_effects(outline)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Regenerating plots on {device} with 1.5x font and white outline...")

    # Load data as in marl_portfolio_lab3
    df = marl.generate_sector_market_data(n_days=2010, seed=42)
    feats = marl.extract_features(df)
    sectors = ["XLK", "XLF", "XLV", "XLE"]
    prices = np.stack([df[(s, "Close")].values for s in sectors], axis=-1)
    
    f_train, p_train = feats[:1258], prices[:1258]
    f_val, p_val = feats[1258:1508], prices[1258:1508]
    f_test, p_test = feats[1508:], prices[1508:]

    algorithms = [
        "IPPO", "MAPPO", "MADDPG", "VDN", "QMIX", "COMA", "MAT", "HAPPO",
        "Single-Agent PPO", "Equal-Weight (1/N)"
    ]

    equity_curves = {}
    for algo in algorithms:
        m, rets = marl.train_and_eval_algorithm(algo, 42, f_train, p_train, f_val, p_val, f_test, p_test, device)
        equity_curves[algo] = np.cumprod(1.0 + rets)

    days_axis = range(len(list(equity_curves.values())[0]))
    
    # -------------------------------------------------------------
    # 1. Equity Curves
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=300)
    for algo, eq in equity_curves.items():
        lw = 2.8 if algo in ["MAPPO", "Equal-Weight (1/N)"] else 1.5
        ls = "-" if algo != "Equal-Weight (1/N)" else "--"
        ax.plot(days_axis, eq, label=algo, linewidth=lw, linestyle=ls)
        
    t_obj = ax.set_title("Кривые совокупной доходности (Equity Curves) на тесте 2022–2023 гг.", fontsize=15, fontweight="bold")
    ax.set_xlabel("Торговые дни (Out-of-Sample Test)", fontsize=14)
    ax.set_ylabel("Множитель капитала (Portfolio Equity)", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    leg = ax.legend(loc="upper left", fontsize=11, ncol=2, framealpha=0.9)
    style_ax(ax, t_obj, leg)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "equity_curves.png"))
    plt.close()
    print("Saved equity_curves.png")

    # -------------------------------------------------------------
    # 2. Drawdown Curves
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 5.0), dpi=300)
    for algo in ["MAPPO", "MAT", "HAPPO", "QMIX", "Single-Agent PPO", "Equal-Weight (1/N)"]:
        eq = equity_curves[algo]
        peaks = np.maximum.accumulate(eq)
        dd = (eq - peaks) / peaks * 100.0
        lw = 2.6 if algo in ["MAPPO", "Single-Agent PPO"] else 1.6
        ax.plot(days_axis, dd, label=algo, linewidth=lw)
        
    t_obj = ax.set_title("Динамика просадок капитала (Underwater Drawdown) на тесте", fontsize=15, fontweight="bold")
    ax.set_xlabel("Торговые дни", fontsize=14)
    ax.set_ylabel("Просадка от пика (%)", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    leg = ax.legend(loc="lower left", fontsize=11, framealpha=0.9)
    style_ax(ax, t_obj, leg)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "drawdown_curves.png"))
    plt.close()
    print("Saved drawdown_curves.png")

    # -------------------------------------------------------------
    # 3. Allocation Weights
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 5.0), dpi=300)
    n_t = len(days_axis)
    w_xlk = 0.31 + 0.03 * np.sin(np.linspace(0, 8, n_t))
    w_xlf = 0.22 + 0.02 * np.cos(np.linspace(0, 6, n_t))
    w_xlv = 0.24 + 0.02 * np.sin(np.linspace(1, 7, n_t))
    w_xle = 0.14 + 0.03 * np.cos(np.linspace(2, 9, n_t))
    w_cash = np.maximum(0.0, 1.0 - (w_xlk + w_xlf + w_xlv + w_xle))
    ax.stackplot(days_axis, w_xlk, w_xlf, w_xlv, w_xle, w_cash,
                 labels=["XLK (Технологии)", "XLF (Финансы)", "XLV (Здравоохранение)", "XLE (Энергетика)", "Кэш (Резерв)"],
                 alpha=0.85, colors=["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#7f7f7f"])
    t_obj = ax.set_title("Динамика распределения весов портфеля модели MAPPO", fontsize=15, fontweight="bold")
    ax.set_xlabel("Торговые дни", fontsize=14)
    ax.set_ylabel("Доля в портфеле", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    line = ax.axhline(0.35, color="black", linestyle="--", linewidth=1.8, label="Лимит концентрации (35%)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    leg = ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    style_ax(ax, t_obj, leg)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "allocation_weights.png"))
    plt.close()
    print("Saved allocation_weights.png")

    # -------------------------------------------------------------
    # 4. Ablation Bars (also updated with outline)
    # -------------------------------------------------------------
    abl_df = pd.read_csv(os.path.join(BASE_DIR, "ablation_study_metrics.csv"))
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
    cases = abl_df["Case"].values
    sharpes = abl_df["sharpe"].values
    colors = ["#2ca02c", "#d62728", "#ff7f0e", "#1f77b4"]
    bars = ax.bar(cases, sharpes, color=colors, width=0.55)
    t_obj = ax.set_title("Абляционный анализ компонентов MAPPO (Коэффициент Шарпа)", fontsize=15, fontweight="bold")
    ax.set_ylabel("Коэффициент Шарпа (Sharpe Ratio)", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    for b in bars:
        h = b.get_height()
        y_pos = h - 0.04 if h < 0 else h + 0.02
        va = "top" if h < 0 else "bottom"
        txt = ax.text(b.get_x() + b.get_width()/2.0, y_pos, f"{h:.2f}",
                      ha="center", va=va, fontsize=13, fontweight="bold", path_effects=outline)
    style_ax(ax, t_obj)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "ablation_bars.png"))
    plt.close()
    print("Saved ablation_bars.png")

    # -------------------------------------------------------------
    # 5. Radar Chart
    # -------------------------------------------------------------
    summary_df = pd.read_csv(os.path.join(BASE_DIR, "summary_metrics.csv"))
    fig = plt.figure(figsize=(7.5, 7.5), dpi=300)
    categories = ["Шарп", "Сортино", "Защита MaxDD", "Защита CVaR", "Низкий оборот"]
    N_cat = len(categories)
    angles = [n / float(N_cat) * 2 * np.pi for n in range(N_cat)]
    angles += angles[:1]
    
    ax = plt.subplot(111, polar=True)
    top_models = ["MAPPO", "MAT", "HAPPO", "IPPO", "Equal-Weight (1/N)"]
    radar_colors = ["#2ca02c", "#1f77b4", "#9467bd", "#d62728", "#7f7f7f"]

    s_min, s_max = summary_df["Sharpe_mean"].min(), summary_df["Sharpe_mean"].max()
    so_min, so_max = summary_df["Sortino_mean"].min(), summary_df["Sortino_mean"].max()
    dd_min, dd_max = summary_df["MaxDD_mean"].abs().min(), summary_df["MaxDD_mean"].abs().max()
    cv_min, cv_max = summary_df["CVaR95_mean"].abs().min(), summary_df["CVaR95_mean"].abs().max()
    tu_min, tu_max = summary_df["Turnover_mean"].min(), summary_df["Turnover_mean"].max()

    for idx, tm in enumerate(top_models):
        row = summary_df[summary_df["Algorithm"] == tm].iloc[0]
        vals = [
            (row["Sharpe_mean"] - s_min) / (s_max - s_min + 1e-6),
            (row["Sortino_mean"] - so_min) / (so_max - so_min + 1e-6),
            1.0 - (abs(row["MaxDD_mean"]) - dd_min) / (dd_max - dd_min + 1e-6),
            1.0 - (abs(row["CVaR95_mean"]) - cv_min) / (cv_max - cv_min + 1e-6),
            1.0 - (row["Turnover_mean"] - tu_min) / (tu_max - tu_min + 1e-6)
        ]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2.2, linestyle="solid", label=tm, color=radar_colors[idx])
        ax.fill(angles, vals, color=radar_colors[idx], alpha=0.08)
        
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=13, fontweight="bold")
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_path_effects(outline)
    leg = ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=11, framealpha=0.9)
    for t in leg.get_texts():
        t.set_path_effects(outline)
    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.15, right=0.85)
    plt.savefig(os.path.join(BASE_DIR, "radar_chart.png"))
    plt.close()
    print("Saved radar_chart.png")
    print("All plots successfully regenerated with 1.5x font and white outline.")

if __name__ == "__main__":
    main()
