"""Bounded insertion reward, independent of simulator bindings."""
import numpy as np


def insertion_reward(head_pos, socket_pos, socket_height=0.0215, success_height=0.005):
    goal = np.asarray(socket_pos).copy()
    goal[2] += success_height
    dist = float(np.linalg.norm(np.asarray(head_pos) - goal))
    pos_score = float(np.exp(-dist * dist / (2.0 * 0.03**2)))
    delta = np.asarray(head_pos) - np.asarray(socket_pos)
    xy_dist = float(np.linalg.norm(delta[:2]))
    z_error = abs(float(delta[2]))
    engaged = z_error < socket_height * 0.5 and xy_dist < 0.05
    inserted = z_error < success_height and xy_dist < 0.05
    reward = 0.5 * pos_score + 0.3 * engaged + 0.2 * inserted
    return float(reward), {
        "rew_pos_score": pos_score, "rew_engaged": float(engaged),
        "rew_inserted": float(inserted), "head_to_goal_mm": dist * 1000.0,
    }
