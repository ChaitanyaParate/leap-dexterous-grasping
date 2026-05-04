# Training and Reward Tuning Log

This log documents the iterative process of tuning the dense reward function and debugging policy exploits for the LEAP Hand grasping task.

## Run 1: The "Jitter" Problem
*   **Initial Setup:** PPO training with Approach (+), Contact (+), Lift (+), and Velocity Penalty (-).
*   **Observed Behavior:** The agent learned to either freeze completely or flail its fingers violently.
*   **Diagnosis:** The high-frequency physics jitter inherent in MuJoCo contacts caused massive, unpredictable spikes in `qvel` (joint velocities). The velocity penalty became highly unstable, heavily punishing the agent anytime it touched the cube.

## Run 2: Action Penalties and The "Hovering" Hack
*   **Changes Applied:** Removed the velocity penalty. Replaced it with a small action regularization penalty (`-0.001 * sum(ctrl^2)`).
*   **Observed Behavior:** The violent movements persisted (flinging the cube). Eventually, the agent learned to hover its fingers directly above the cube, making a fist without actually grasping it.
*   **Diagnosis:** 
    1.  **Infinite Acceleration:** The environment used *Absolute Position Control*. The neural network learned it could command a joint to move from fully open to fully closed in a single timestep (0.01s), causing infinite physics acceleration and violent flinging.
    2.  **Reward Hacking:** The contact reward simply counted ANY contacts involving the cube. Because the cube was resting on the table, it inherently had 4 contact points. The agent realized that touching the cube might tilt it (losing table contacts) or knock it out of bounds. The optimal strategy was to let the cube sit still on the table and hover nearby to farm the "approach" reward.

## Run 3 (Final): Delta Control and True Contact 
*   **Changes Applied:**
    1.  **Delta Control:** Changed action space to `[-1, 1]` and applied bounded relative actions (`max_delta = 0.1` rad/step). This physically enforced smooth, human-like finger movements.
    2.  **True Contact Detection:** Explicitly filtered out MuJoCo contacts involving `table_top` and `floor`. The agent is now only rewarded when a component of the hand touches the cube.
    3.  **Out-of-Bounds Penalty:** Added a strict 10cm horizontal boundary and `z < 0` check. If the cube is dropped or flung, the episode terminates early with a `-50.0` penalty.
*   **Result:** **Success**. The policy converged to a physically robust, stable tripod grasp. The final model achieved an Eval Mean Reward > +1350 and an Explained Variance of 0.83, indicating a near-perfect understanding of the physical task constraints.
