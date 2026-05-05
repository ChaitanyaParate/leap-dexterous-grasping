"""
plot_learning_curve.py
Reads TensorBoard event files and produces a learning curve plot
saved to results/learning_curve.png.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("Installing tensorboard...")
    os.system("pip install tensorboard -q")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tb_scalars(logdir, tag="rollout/ep_rew_mean"):
    """Load scalar values from a TensorBoard event directory."""
    ea = EventAccumulator(logdir, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        print(f"  Warning: tag '{tag}' not found in {logdir}")
        return np.array([]), np.array([])
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def smooth(values, weight=0.85):
    """Exponential moving average smoothing."""
    last = values[0]
    smoothed = []
    for v in values:
        last = last * weight + (1 - weight) * v
        smoothed.append(last)
    return np.array(smoothed)


def main():
    os.makedirs("results", exist_ok=True)

    # Locate TensorBoard runs
    tb_base = "results/tensorboard"
    runs = {
        "Run 1 (0–1.5M steps)": os.path.join(tb_base, "PPO_1"),
        "Run 2 (1.5M–3.5M steps, resume)": os.path.join(tb_base, "PPO_resume_0"),
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#4C9BE8", "#F4845F"]
    offset = 0  # step offset for stitching runs

    for (label, logdir), color in zip(runs.items(), colors):
        if not os.path.exists(logdir):
            print(f"Skipping {logdir} — not found")
            continue
        print(f"Loading: {logdir}")
        steps, values = load_tb_scalars(logdir, "rollout/ep_rew_mean")
        if len(steps) == 0:
            continue

        # Stitch runs together
        stitched_steps = steps + offset
        offset = stitched_steps[-1]

        # Raw and smoothed
        ax.plot(stitched_steps, values, alpha=0.2, color=color, linewidth=0.8)
        ax.plot(stitched_steps, smooth(values), color=color, linewidth=2.2, label=label)

    # Annotations
    ax.axvline(x=1_500_000, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(1_510_000, 200, "Resume\n(Z=0.16)", fontsize=9, color="gray")

    ax.axhline(y=1000, color="#2ECC71", linestyle=":", linewidth=1.2, alpha=0.8)
    ax.text(100_000, 1020, "Success threshold (~1000)", fontsize=9, color="#2ECC71")

    # Final result annotation
    ax.annotate(
        "96% Success Rate\n(1632 mean reward)",
        xy=(3_000_000, 1632),
        xytext=(2_400_000, 1200),
        arrowprops=dict(arrowstyle="->", color="white", lw=1.5),
        fontsize=10, color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#2C3E50", edgecolor="#4C9BE8")
    )

    # Style
    ax.set_facecolor("#1C1C2E")
    fig.patch.set_facecolor("#1C1C2E")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#555")
    ax.spines["left"].set_color("#555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

    ax.set_xlabel("Total Timesteps", fontsize=12)
    ax.set_ylabel("Mean Episode Reward", fontsize=12)
    ax.set_title("LEAP Hand Grasping — PPO Learning Curve (3.5M Steps)", fontsize=14, pad=15)
    ax.legend(fontsize=10, facecolor="#2C3E50", labelcolor="white", edgecolor="#555")
    ax.set_xlim(left=0)

    plt.tight_layout()
    out_path = "results/learning_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nLearning curve saved to {out_path}")


if __name__ == "__main__":
    main()
