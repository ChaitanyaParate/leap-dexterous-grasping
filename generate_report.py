"""
generate_report.py
Generates the 2-page PDF technical report for Task A: Learning-Based Dexterous Grasping.
"""
from fpdf import FPDF
import os

LEARNING_CURVE = "results/learning_curve.png"
OUTPUT = "report.pdf"


class ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, "Task A: Learning-Based Dexterous Grasping  |  RRC IIIT-H Research Internship", align="C")
        self.ln(4)
        self.set_draw_color(180, 180, 180)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 6, f"Page {self.page_no()}/2", align="C")

    def section_title(self, title):
        self.ln(4)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(30, 80, 160)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(30, 80, 160)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def body_text(self, text, indent=0):
        self.set_font("Helvetica", size=9.5)
        self.set_text_color(30, 30, 30)
        self.set_x(10 + indent)
        self.multi_cell(190 - indent, 5.5, text)
        self.ln(1)

    def bullet(self, text, indent=4):
        self.set_font("Helvetica", size=9.5)
        self.set_text_color(30, 30, 30)
        self.set_x(10 + indent)
        self.cell(5, 5.5, "-")
        self.multi_cell(185 - indent, 5.5, text)

    def kv_row(self, key, value):
        self.set_font("Helvetica", "B", 9.5)
        self.set_x(14)
        self.cell(55, 5.5, key)
        self.set_font("Helvetica", size=9.5)
        self.multi_cell(125, 5.5, value)


def build():
    pdf = ReportPDF(format="A4")
    pdf.set_margins(10, 14, 10)
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()

    # --- Title ---------------------------------------------------------------
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(20, 50, 120)
    pdf.cell(0, 10, "Learning-Based Dexterous Grasping with the LEAP Hand", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 6, "PPO Training in MuJoCo  |  True Success Rate: 85%  (100 deterministic episodes)", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(3)
    pdf.set_draw_color(20, 50, 120)
    pdf.set_line_width(0.6)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.set_line_width(0.2)
    pdf.ln(5)

    # --- 1. Observation Space ------------------------------------------------
    pdf.section_title("1. Observation Space (39 values)")
    pdf.body_text(
        "The policy receives a 39-dimensional observation vector at each timestep, providing "
        "full proprioceptive state of the hand and the object's pose in world frame:"
    )
    pdf.bullet("16 joint positions (qpos[:16])  --  current angles of all 16 LEAP Hand actuators (rad)")
    pdf.bullet("16 joint velocities (qvel[:16])  --  angular velocity of each joint (rad/s)")
    pdf.bullet("3 object position (xpos)  --  cube centroid in world XYZ (m)")
    pdf.bullet("4 object quaternion (xquat)  --  cube orientation as unit quaternion [w, x, y, z]")
    pdf.body_text(
        "Justification: Joint positions and velocities give the policy full knowledge of its own "
        "configuration for closed-loop control. The object's world-frame pose is the minimal "
        "sufficient representation for computing reach and grasp. No visual input is used, making "
        "this a compact, low-dimensional formulation amenable to fast PPO convergence."
    )

    # --- 2. Action Space ----------------------------------------------------
    pdf.section_title("2. Action Space (16 values)")
    pdf.body_text(
        "The action space is a 16-dimensional continuous Box in [-1, 1], one dimension per actuator. "
        "Actions are applied as delta control (incremental position targets):"
    )
    pdf.body_text("    ctrl[t+1] = clip(ctrl[t] + action * max_delta,  joint_low,  joint_high)", indent=6)
    pdf.body_text(
        "where max_delta = 0.05 rad/step. This delta formulation is critical: absolute position control "
        "allowed the network to command a joint to its full range in a single timestep, causing "
        "physically unrealistic 'flinging'. Delta control enforces smooth, human-like finger movements. "
        "The reduced max_delta (0.05 vs 0.1) was introduced after observing that faster finger sweeps "
        "created lateral contact forces that pushed the cube out of grasp."
    )

    # --- 3. Deviations from Suggested Training Structure --------------------
    pdf.section_title("3. Deviations from Suggested Training Structure")
    pdf.body_text(
        "The task suggests: continuous action space (PD position targets), observations including "
        "joint positions/velocities + object pose, and reward components: distance to object, "
        "contact reward, lift height, action penalty. Our implementation follows this baseline "
        "but with two deliberate, justified deviations:"
    )
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.set_x(10)
    pdf.cell(0, 5.5, "Deviation 1: Delta Control instead of Absolute PD Position Targets", new_x="LMARGIN", new_y="NEXT")
    pdf.body_text(
        "The suggested structure uses absolute PD position targets as actions. We use DELTA "
        "position targets: ctrl[t+1] = ctrl[t] + action * max_delta. Both encode the same "
        "continuous joint control, but absolute targets allowed the network to command a joint "
        "from 0 to its full limit in one timestep (0.01s), causing infinite effective acceleration "
        "and physically unrealistic 'flinging'. Delta control enforces smooth, realistic movements.", indent=4
    )
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.set_x(10)
    pdf.cell(0, 5.5, "Deviation 2: Approach Reward Removed in Final Design", new_x="LMARGIN", new_y="NEXT")
    pdf.body_text(
        "The suggested structure includes a 'distance to object' reward component. We included "
        "this initially (r_approach = 10.0 * max(0, 0.2 - mean_dist)) but removed it after 2M "
        "training steps. Once the policy matured, the approach reward continuously drove all 4 "
        "fingertips toward the cube center even while already grasping, creating lateral contact "
        "forces that pushed the cube out of reach. The agent had already internalized how to find "
        "the cube, making the approach reward counterproductive. Removing it stabilized grasps.", indent=4
    )
    pdf.ln(2)

    # --- 4. Reward Function -------------------------------------------------
    pdf.section_title("4. Reward Function (Final Converged Design)")
    pdf.body_text("The final reward combines 8 components per timestep:")
    pdf.ln(1)
    pdf.bullet("Approach Reward:      r_approach = 10.0 * max(0, 0.2 - mean_dist)  [early training only; removed after 2M steps]")
    pdf.bullet("Contact Reward:       r_contact = 0.5 * min(n_hand_contacts, 4)    [excludes table/floor contacts]")
    pdf.bullet("Lift Reward:          r_lift = 10.0/step  if  obj_Z > 0.085m  AND  n_contacts >= 2  else 0.0")
    pdf.bullet("Corner Airborne:      r_air = 3.0 * max(0, min_corner_Z - 0.055)   [all 8 cube corners in world frame]")
    pdf.bullet("XY Drift Penalty:     r_drift = -15.0 * max(0, |obj_xy - spawn_xy| - 0.03)")
    pdf.bullet("Drop Penalty:         r_drop = -20.0  when cube falls below threshold after having been above it")
    pdf.bullet("Spawn-Z Tether:       r_tether = -20.0 * max(0, 0.085 - obj_Z)")
    pdf.bullet("Time Penalty:         r_time = -0.5/step")
    pdf.bullet("Action Penalty:       r_action = -0.01 * sum(action^2)")
    pdf.ln(1)
    pdf.body_text(
        "The decisive fix was setting lift reward to ZERO below the threshold. Any positive reward below "
        "Z=0.085m gave the agent a profitable 'hovering' exploit -- holding the cube at Z=0.084m for "
        "1000 steps earned more reward than risking a full lift. Zero reward below threshold combined "
        "with a time penalty (-0.5/step) made hovering a net-negative strategy, forcing a committed lift."
    )

    # --- 4. Hyperparameters -------------------------------------------------
    pdf.section_title("4. Training Hyperparameters")
    params = [
        ("Algorithm", "PPO (Stable Baselines 3)"),
        ("Policy network", "MlpPolicy -- 2 hidden layers of 256 units, tanh activation"),
        ("n_steps / batch_size", "2048  /  64"),
        ("n_epochs", "10 gradient steps per rollout"),
        ("learning_rate", "Linear decay: 3e-4 -> 0 over training"),
        ("gamma / gae_lambda", "0.99  /  0.95"),
        ("clip_range", "0.15  (tightened from default 0.2 for stability)"),
        ("ent_coef / vf_coef", "0.01  /  0.5"),
        ("Total timesteps", "~3,000,000  (2M fresh + 1M resume fine-tuning)"),
        ("Physics substeps", "5 per policy step  (timestep = 0.002 s)"),
        ("Cube spawn height", "Z = 0.085 m  (at success threshold, eliminates scooping forces)"),
        ("max_delta", "0.05 rad/step  (halved for smoother, non-lateral finger motion)"),
        ("VecNormalize", "Enabled -- normalizes observations and rewards across the run"),
    ]
    for k, v in params:
        pdf.kv_row(k, v)

    # --- PAGE 2 -------------------------------------------------------------
    pdf.add_page()

    # --- 5. Learning Curve --------------------------------------------------
    pdf.section_title("5. Learning Curve Analysis")
    if os.path.exists(LEARNING_CURVE):
        pdf.image(LEARNING_CURVE, x=12, w=184)
        pdf.ln(2)
    pdf.body_text(
        "Training progressed through four phases: (1) Exploration (0-300k steps): random grasps, "
        "zero success. (2) Lift discovery (300k-600k): success rate jumped from 0% to 38% once the "
        "agent accidentally crossed Z=0.085m and discovered the +10/step reward stream. (3) Policy "
        "refinement (600k-1M): success climbed to 83%, agent mastered stable multi-finger contact. "
        "(4) Convergence (1M-2M): success rate plateaued at 92-100%, reward ~4200/episode. "
        "Fine-tuning resume runs added drop penalty and drift penalty, stabilizing hold duration."
    )

    # --- 6. What Worked / Failed --------------------------------------------
    pdf.section_title("6. What Worked")
    pdf.bullet("Top-down hand orientation (quat='1 0 0 0'): essential for generating upward Z-force. The fixed-base hand has no wrist; only a downward-facing palm allows fingertips to oppose gravity.")
    pdf.bullet("Zero reward below threshold + time penalty: the decisive combination that eliminated hovering. The agent's only profitable strategy became crossing Z=0.085m and holding.")
    pdf.bullet("Cube spawned at threshold height (Z=0.085m): eliminated the 'scooping' motion. Fingers only need to close, not sweep upward, removing the main source of lateral push forces.")
    pdf.bullet("Delta control (max_delta=0.05 rad/step): eliminated simulator instability and produced smooth, physically realistic grasps.")
    pdf.bullet("Linear LR decay (3e-4 -> 0): prevented catastrophic forgetting during long fine-tuning runs.")

    pdf.section_title("7. What Failed")
    pdf.bullet("Velocity-based action penalty: qvel spikes from MuJoCo contact events made this highly unstable and impossible to tune.")
    pdf.bullet("Naive contact reward (all contacts): the agent exploited the 4 resting contacts between cube and table, learning to 'hover near' rather than grasp.")
    pdf.bullet("Continuous lift reward below threshold (lift_height * 50.0): made hovering at Z=0.084m worth ~2800 points/episode -- more profitable than risking a full lift.")
    pdf.bullet("Approach reward in late training: once the policy matured, it caused all 4 fingertips to push inward simultaneously, creating lateral forces that flung the cube away.")
    pdf.bullet("Early episode termination on success: the agent discovered it could earn more reward by hovering indefinitely than by completing the task and ending the episode early.")

    pdf.section_title("8. Difficulties & Lessons")
    pdf.bullet("Reward hacking is the dominant failure mode in dense reward RL. Every positive reward component must be analyzed for exploitability. The agent will always find the optimal mathematical strategy, even if physically nonsensical.")
    pdf.bullet("Physical constraint verification is essential before defining success thresholds. The original Z > 0.25m target was physically impossible (palm at Z=0.16m). All task parameters must be verified against the URDF kinematics.")
    pdf.bullet("Curriculum spawning (cube at Z=0.085m) was critical. Starting from table height (Z=0.075m) required a 'scooping' motion that consistently pushed the cube out of reach due to lateral contact forces.")

    # --- Results Summary ----------------------------------------------------
    pdf.ln(3)
    pdf.set_fill_color(235, 245, 255)
    pdf.set_draw_color(30, 80, 160)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "  Final Evaluation Results (100 randomized episodes, deterministic policy)", new_x="LMARGIN", new_y="NEXT", fill=True, border=1)
    pdf.set_font("Helvetica", size=9.5)
    results = [
        ("Mean Episode Reward", "2846.96  +/-  3305.17"),
        ("Mean Episode Length", "994.01  /  1000 steps"),
        ("True Success Rate", "85%  (Z > 0.090 m for 20 consecutive steps)"),
        ("Total Training Steps", "~3,000,000"),
        ("Eval Policy", "Deterministic (no exploration noise)"),
    ]
    for k, v in results:
        pdf.set_x(14)
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.cell(60, 5.5, k)
        pdf.set_font("Helvetica", size=9.5)
        pdf.cell(0, 5.5, v, ln=True)

    pdf.output(OUTPUT)
    print(f"Report saved to {OUTPUT}")


if __name__ == "__main__":
    build()
