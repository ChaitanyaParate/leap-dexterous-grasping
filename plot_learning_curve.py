"""
plot_learning_curve.py
Generates an accurate 2-panel learning curve from monitor CSV files and
EvalCallback evaluations.npz. Replaces the previous TensorBoard-based version.

Panel 1 (top)   : Rolling-mean episode reward across both training phases.
Panel 2 (bottom): EvalCallback success rate (%) — available for resume phase only.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

MONITOR_FRESH   = "results/logs/monitor.csv"
MONITOR_RESUME  = "results/logs/monitor_resume.csv.monitor.csv"
EVAL_NPZ        = "results/logs/evaluations.npz"
OUT_PATH        = "results/learning_curve.png"

RESUME_START_STEP = 2_677_048   # checkpoint used to start resume_train.py
ROLL_WINDOW       = 60          # episode rolling-mean window
SMOOTH_WEIGHT     = 0.90        # EMA weight for reward curve

# ── colour palette (dark background) ────────────────────────────────────────
BG       = "#0F1117"
PANEL_BG = "#16171F"
C_FRESH  = "#4C9BE8"   # blue  — fresh run
C_RESUME = "#F4845F"   # coral — resume run
C_SUCC   = "#2ECC71"   # green — success rate
C_FINAL  = "#E8D44D"   # gold  — final eval marker
GRAY     = "#555566"


def load_monitor(path):
    """Return DataFrame with columns [r, l] from a Stable-Baselines Monitor CSV."""
    df = pd.read_csv(path, skiprows=1, names=["r", "l", "t"])
    df["r"] = pd.to_numeric(df["r"], errors="coerce")
    df["l"] = pd.to_numeric(df["l"], errors="coerce")
    df = df.dropna(subset=["r", "l"])
    return df


def rolling(values, window):
    """Centred rolling mean (edges use available data)."""
    series = pd.Series(values)
    return series.rolling(window, min_periods=1, center=True).mean().values


def ema(values, weight=0.90):
    last = float(values[0])
    out = []
    for v in values:
        last = last * weight + (1 - weight) * float(v)
        out.append(last)
    return np.array(out)


def main():
    os.makedirs("results", exist_ok=True)

    # ── load episode data ────────────────────────────────────────────────────
    df_fresh  = load_monitor(MONITOR_FRESH)
    df_resume = load_monitor(MONITOR_RESUME)

    # Cumulative step counts inside each run
    steps_fresh  = np.cumsum(df_fresh["l"].values)          # 0 → ~2.0 M
    steps_resume = RESUME_START_STEP + np.cumsum(df_resume["l"].values)  # → ~3.7 M

    rew_fresh  = df_fresh["r"].values
    rew_resume = df_resume["r"].values

    roll_fresh  = rolling(rew_fresh,  ROLL_WINDOW)
    roll_resume = rolling(rew_resume, ROLL_WINDOW)

    # ── load eval checkpoints (resume phase only) ────────────────────────────
    ev = np.load(EVAL_NPZ)
    eval_steps   = ev["timesteps"]                    # shape (40,)
    eval_rewards = ev["results"].mean(axis=1)         # mean over 5 episodes
    eval_success = ev["successes"].mean(axis=1) * 100 # % success over 5 episodes

    # ── figure ───────────────────────────────────────────────────────────────
    fig, (ax_r, ax_s) = plt.subplots(
        2, 1, figsize=(13, 8),
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08},
        sharex=True
    )
    fig.patch.set_facecolor(BG)
    for ax in (ax_r, ax_s):
        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRAY)
        ax.tick_params(colors="white", labelsize=9)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")

    # ── Panel 1 : episode reward ─────────────────────────────────────────────
    # raw (faint)
    ax_r.plot(steps_fresh,  rew_fresh,  alpha=0.08, color=C_FRESH,  linewidth=0.6)
    ax_r.plot(steps_resume, rew_resume, alpha=0.08, color=C_RESUME, linewidth=0.6)
    # rolling mean
    ax_r.plot(steps_fresh,  roll_fresh,  color=C_FRESH,  linewidth=2.2,
              label="Phase 1 — Fresh run (0 – 2.7 M steps)")
    ax_r.plot(steps_resume, roll_resume, color=C_RESUME, linewidth=2.2,
              label="Phase 2 — Resume run (2.7 – 3.7 M steps)")

    # eval reward dots (mid-training, 5 eps, normalised)
    ax_r.scatter(eval_steps, eval_rewards, color=C_SUCC, s=18, zorder=5,
                 alpha=0.7, label="EvalCallback reward (5 eps, norm)")

    # phase separator
    ax_r.axvline(RESUME_START_STEP, color=GRAY, linestyle="--", linewidth=1.2)
    ax_r.text(RESUME_START_STEP + 40_000, ax_r.get_ylim()[0] if ax_r.get_ylim()[0] > -3000 else -2500,
              "Resume\nstart", color=GRAY, fontsize=8, va="bottom")

    # final eval annotation (honest: 85%, 100 eps, raw reward)
    ax_r.axhline(2846.96, color=C_FINAL, linestyle=":", linewidth=1.4, alpha=0.9)
    ax_r.text(200_000, 2846.96 + 120,
              "Final eval mean reward: 2847 (100 eps, raw, deterministic)",
              color=C_FINAL, fontsize=8)

    ax_r.set_ylabel("Episode Reward", fontsize=11, color="white")
    ax_r.set_title(
        "LEAP Hand Grasping — PPO Learning Curve  |  Two-Phase Training (~3.7 M Steps)",
        fontsize=13, color="white", pad=10
    )
    ax_r.legend(fontsize=8.5, facecolor="#1C1C2E", labelcolor="white",
                edgecolor=GRAY, loc="upper left")

    # ── Panel 2 : success rate ───────────────────────────────────────────────
    ax_s.plot(eval_steps, eval_success, color=C_SUCC, linewidth=2.2,
              marker="o", markersize=4, label="EvalCallback success % (5 eps, mid-training)")
    ax_s.fill_between(eval_steps, eval_success, alpha=0.12, color=C_SUCC)

    # final eval 85% line
    ax_s.axhline(85, color=C_FINAL, linestyle=":", linewidth=1.4, alpha=0.9)
    ax_s.text(eval_steps[0], 86.5, "Final deterministic eval: 85%  (100 episodes)",
              color=C_FINAL, fontsize=8)

    # phase separator
    ax_s.axvline(RESUME_START_STEP, color=GRAY, linestyle="--", linewidth=1.2)

    ax_s.set_ylabel("Success Rate (%)", fontsize=11, color="white")
    ax_s.set_xlabel("Total Training Timesteps", fontsize=11, color="white")
    ax_s.set_ylim(0, 115)
    ax_s.set_yticks([0, 20, 40, 60, 80, 100])

    # note about eval availability
    ax_s.text(0.01, 0.05,
              "Note: EvalCallback data available for Phase 2 only. "
              "Phase 1 mid-training success not logged.",
              transform=ax_s.transAxes, color=GRAY, fontsize=7.5, va="bottom")

    ax_s.legend(fontsize=8.5, facecolor="#1C1C2E", labelcolor="white",
                edgecolor=GRAY, loc="lower right")

    # ── x-axis formatting ────────────────────────────────────────────────────
    ax_s.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1_000_000:.1f}M")
    )

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Learning curve saved to {OUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
