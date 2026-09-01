"""Render a figure from an edge-client run log (the <tag>.json the robot saves).

Usage:
    python plot_run.py mp2_open_straight.json [out.png]

Left panel: predicted action chunk (8 waypoints) vs the goal direction.
Middle:     guidance staleness over the run (how old the server's guidance was).
Right:      edge-inference latency histogram.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

path = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else path.rsplit(".", 1)[0] + ".png"
with open(path) as f:
    run = json.load(f)

traj = np.array(run["final_traj"])
goal = run["goal"]
k = np.array(run["stats"]["k"])
edge_ms = np.array(run["stats"]["edge_ms"])
t_wall = np.array(run["stats"]["t_wall"])

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Plot in screen-intuitive frame: forward = up, robot's right = plot right.
axes[0].plot([0.0, *(-traj[:, 1])], [0.0, *traj[:, 0]], "o-", color="tab:blue",
             label="action chunk")
gnorm = 10.0 / max(abs(goal[0]), abs(goal[1]), 1e-9)
axes[0].plot(-goal[1] * gnorm, goal[0] * gnorm, "r*", markersize=15,
             label="goal direction")
axes[0].set_title(f"{run['tag']}  (goal x_fwd={goal[0]:.0f}, y_left={goal[1]:.0f})")
axes[0].set_xlabel("y (left <- -> right)")
axes[0].set_ylabel("x (forward)")
axes[0].axis("equal")
axes[0].legend()

axes[1].plot(t_wall, k)
axes[1].set_title(f"Guidance staleness (mean {k.mean():.2f} s)")
axes[1].set_xlabel("wall time (s)")
axes[1].set_ylabel("k (s)")

axes[2].hist(edge_ms, bins=30)
axes[2].set_title(f"Edge latency: mean {edge_ms.mean():.0f} ms "
                  f"= {1000 / edge_ms.mean():.1f} Hz")
axes[2].set_xlabel("ms per pass")

plt.tight_layout()
plt.savefig(out, dpi=90)
print("saved", out)
