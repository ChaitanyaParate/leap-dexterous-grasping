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
    pdf.set_font("Helvetica", size=9.5)
    pdf.cell(0, 6, "PPO Training in MuJoCo  |  True Success Rate: 0%  (Proxy Metric: 96%)", new_x="LMARGIN", new_y="NEXT", align="C")
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
        "where max_delta = 0.1 rad/step. This delta formulation is critical: absolute position control "
        "allowed the network to command a joint from 0 to its limit in a single 0.01s step, causing "
        "infinite effective acceleration and physically unrealistic 'flinging'. Delta control enforces "
        "smooth, human-like finger movements and prevents the simulator from becoming numerically unstable."
    )

    # --- 3. Reward Function -------------------------------------------------
    pdf.section_title("3. Reward Function")
    pdf.body_text("The dense reward combines four components at each timestep:")
    pdf.ln(1)
    pdf.bullet("Approach Reward:  r_approach = 10.0 * max(0, 0.2 - mean_dist_fingertips_to_cube)")
    pdf.bullet("Contact Reward:   r_contact = 0.5 * min(n_hand_contacts, 4)   [excludes table/floor contacts]")
    pdf.bullet("Lift Reward:      r_lift = 50.0 * max(0, obj_z - 0.075) * [n_contacts >= 2]   [grasping-gated]")
    pdf.bullet("Action Penalty:   r_penalty = -0.01 * sum(action^2)")
    pdf.ln(1)
    pdf.body_text("Total reward:  R = r_approach + r_contact + r_lift + r_penalty")
    pdf.body_text(
        "Key design decisions: (1) The lift reward is grasped-gated -- multiplied by zero unless at least 2 "
        "fingertips are touching the cube -- preventing the policy from simply tossing the cube upward. "
        "(2) Contact detection explicitly excludes contacts between the cube and table/floor, so the agent "
        "cannot farm contact reward by letting the cube rest on the table. (3) The approach reward saturates "
        "at 0.2m to avoid an infinitely large gradient pulling all fingers toward the same point."
    )

    # --- 4. Hyperparameters -------------------------------------------------
    pdf.section_title("4. Training Hyperparameters")
    params = [
        ("Algorithm", "PPO (Stable Baselines 3 v2.3.2)"),
        ("Policy network", "MlpPolicy -- 2 hidden layers of 256 units, tanh activation"),
        ("n_steps / batch_size", "2048  /  64"),
        ("n_epochs", "10 gradient steps per rollout"),
        ("learning_rate", "3e-4 (Adam optimizer)"),
        ("gamma / gae_lambda", "0.99  /  0.95"),
        ("clip_range", "0.15  (tightened from default 0.2 for stability at high reward)"),
        ("ent_coef / vf_coef", "0.01  /  0.5"),
        ("Total timesteps", "2,500,000  (1.5M fresh + 1M resume)"),
        ("Physics substeps", "5 per policy step  (timestep = 0.002 s)"),
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
        "The learning curve shows three distinct phases: (1) Exploration (0-300k steps): the agent "
        "discovers fingertip-to-cube proximity rewards. (2) Grasping discovery (300k-1.5M): the policy "
        "learns to make stable multi-finger contact and slightly lift the cube to farm reward, peaking "
        "at convergence around 1.5M steps. (3) Over-exploration and Regression (1.5M-2.5M): during the "
        "resume run, the constant high learning rate (3e-4) caused aggressive updates to the already "
        "converged policy, leading to catastrophic forgetting and steady regression in mean reward from "
        "~1600 down to ~1000. Evaluating the peak model (saved dynamically by EvalCallback at ~1.5M) "
        "is necessary to capture the best policy state before regression."
    )

    # --- 6. Critical Analysis of Success Metric -----------------------------
    pdf.section_title("6. Critical Analysis of Success Metric")
    pdf.body_text(
        "The proxy metric initially reported a 96% success rate, but the rigorous physical criterion "
        "(cube lifted > 0.25m and sustained for 10 steps) reports exactly 0%. Here is why they differ "
        "and what this means:"
    )
    pdf.bullet("Proxy Flaw: The initial evaluation script falsely defined success as ep_reward > 500 combined with early termination. If the agent grabbed the cube, farmed 800 reward by hovering it slightly off the ground, and then flung it out of bounds (triggering early termination and a -50 penalty), it finished with 750 reward and <500 steps, falsely counting as a 'success'.")
    pdf.bullet("Reward Hacking: The lift reward formula (50.0 * max(0, z - 0.075)) allowed the agent to achieve massive cumulative reward simply by holding the cube at Z=0.12m for the entire 500-step episode. The policy never needed to reach the true 0.25m threshold to maximize expected return.")
    pdf.bullet("OOB Boundary Flaw: The out-of-bounds check (dist_xy > 0.1m) was too tight given the spawn radius. The agent was heavily penalized for lateral manipulation, forcing it into conservative, low-height hovers rather than dynamic lifts.")

    # --- 7. What Worked -----------------------------------------------------
    pdf.section_title("7. What Worked")
    pdf.bullet("Top-down hand orientation (quat='1 0 0 0'): essential for generating upward Z-force against gravity. The fixed-base hand has no wrist joint; only a downward-facing palm allows fingertips to sweep under the cube.")
    pdf.bullet("Delta control (max_delta=0.1 rad/step): eliminated all simulator instability and produced smooth, physically realistic grasps.")
    pdf.bullet("Grasping-gated lift reward: completely stopped the 'fling' exploit where the agent would bat the cube upward.")
    pdf.bullet("Height calibration (Z=0.16m): the 'Goldilocks' height gave fingers clearance to close around the cube without getting stuck under the table.")

    # --- 8. What Failed -----------------------------------------------------
    pdf.section_title("8. What Failed")
    pdf.bullet("Sideways orientation: physically impossible to lift -- no upward force vector.")
    pdf.bullet("Velocity-based action penalty: qvel spikes from MuJoCo contacts made this highly unstable.")
    pdf.bullet("Naive contact reward (all contacts): the agent exploited the 4 resting contacts between cube and table.")
    pdf.bullet("High constant learning rate: 3e-4 was too aggressive for fine-tuning past 1.5M steps, causing regression.")

    # --- 9. What I'd Do Differently -----------------------------------------
    pdf.section_title("9. What I Would Do Differently")
    pdf.bullet("Learning rate schedule: anneal learning_rate from 3e-4 to 1e-5 over training to prevent policy degradation.")
    pdf.bullet("Fix Reward Structure: Change the lift reward to be exponential or sparse upon crossing the true Z=0.25m threshold, preventing the agent from farming reward at low hover heights.")
    pdf.bullet("Widen OOB Boundary: Increase dist_xy threshold to 0.2m to allow realistic lateral grasping adjustments without triggering premature failure.")
    pdf.bullet("Curriculum learning: start with the cube directly under the fingertips and gradually increase randomization range.")

    # --- Results Summary ----------------------------------------------------
    pdf.ln(3)
    pdf.set_fill_color(235, 245, 255)
    pdf.set_draw_color(30, 80, 160)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "  Final Evaluation Results (100 randomized episodes, deterministic policy)", new_x="LMARGIN", new_y="NEXT", fill=True, border=1)
    pdf.set_font("Helvetica", size=9.5)
    results = [
        ("Mean Episode Reward", "1541.28  +/-  499.10"),
        ("Mean Episode Length", "441.86  /  500 steps"),
        ("True Success Rate", "0%  (Proxy metric was 96%)"),
        ("Total Training Steps", "2,500,000"),
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
