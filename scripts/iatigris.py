# IA-TIGRIS with Camera Frustum Sensing + Bezier Belief Maps

import numpy as np
import math
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple, Dict, List
import time
import os

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import torch

import numpy as np
from scipy.signal import savgol_filter


def entropy(p: float) -> float:
    if p <= 1e-6 or p >= 1 - 1e-6:
        return 0.0
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


class BeliefMap:
    def __init__(self, prob_map: np.ndarray):
        self.p = prob_map.copy()
        self.H, self.W = prob_map.shape
        self._entropy_cache = self.entropy_map()

    def entropy_map(self):
        v = np.vectorize(entropy)
        return v(self.p)

    def bayes_update(self, p, z, tpr=0.9, tnr=0.9):
        if z == 1:
            num = tpr * p
            den = tpr * p + (1 - tnr) * (1 - p)
        else:
            num = (1 - tpr) * p
            den = (1 - tpr) * p + tnr * (1 - p)
        return num / max(den, 1e-9)


@dataclass
class Node:
    state: Tuple[float, float, float] 
    parent: 'Node' = None
    cost: float = 0.0
    info: float = 0.0
    delta: Dict[int, float] = field(default_factory=dict)


class IATigris:
    def __init__(self,
                 belief: BeliefMap,
                 budget: float,
                 step_size: float = 8.0,
                 edge_resolution: float = 2.0,
                 cam_range: float = 20.0,
                 cam_fov: float = math.pi / 3):

        self.belief = belief
        self.budget = budget
        self.step_size = step_size
        self.edge_resolution = edge_resolution
        self.cam_range = cam_range
        self.cam_fov = cam_fov
        self.nodes: List[Node] = []


    def informed_sample(self):
        H = self.belief._entropy_cache
        flat = H.flatten()
        total = flat.sum()

        if total <= 1e-12 or not np.isfinite(total):
            x = np.random.uniform(0, self.belief.W)
            y = np.random.uniform(0, self.belief.H)
            yaw = np.random.uniform(-math.pi, math.pi)
            return (x, y, yaw)

        probs = flat / total
        idx = np.random.choice(len(flat), p=probs)

        y = idx // self.belief.W
        x = idx % self.belief.W
        yaw = np.random.uniform(-math.pi, math.pi)
        return (x, y, yaw)

    def nearest(self, state):
        return min(self.nodes, key=lambda n: math.hypot(n.state[0]-state[0], n.state[1]-state[1]))


    def steer(self, from_state, to_state):
        x, y, _ = from_state
        angle = math.atan2(to_state[1]-y, to_state[0]-x)
        steps = max(1, int(self.step_size / self.edge_resolution))
        pts = []
        for i in range(1, steps+1):
            nx = x + i*self.edge_resolution*math.cos(angle)
            ny = y + i*self.edge_resolution*math.sin(angle)
            pts.append((nx, ny, angle))
        return pts


    def frustum_cells(self, x, y, yaw):
        cells = []
        for j in range(self.belief.H):
            for i in range(self.belief.W):
                dx = i - x
                dy = j - y
                r = math.hypot(dx, dy)
                if r > self.cam_range or r < 1e-6:
                    continue
                ang = math.atan2(dy, dx)
                if abs((ang - yaw + math.pi) % (2*math.pi) - math.pi) <= self.cam_fov/2:
                    cells.append((i, j))
        return cells


    def edge_information(self, parent: Node, states):
        gain = 0.0
        delta = dict(parent.delta)

        for (x, y, yaw) in states:
            for (i, j) in self.frustum_cells(x, y, yaw):
                key = j * self.belief.W + i
                p_old = delta.get(key, self.belief.p[j, i])
                z = 1 if p_old > 0.5 else 0
                p_new = self.belief.bayes_update(p_old, z)
                gain += entropy(p_old) - entropy(p_new)
                delta[key] = p_new

        return gain, delta


    def add_node(self, x_rand):
        parent = self.nearest(x_rand)
        states = self.steer(parent.state, x_rand)
        step_cost = self.step_size
        if parent.cost + step_cost > self.budget:
            return None

        gain, delta = self.edge_information(parent, states)
        new_state = states[-1]

        node = Node(new_state, parent, parent.cost+step_cost, parent.info+gain, delta)
        self.nodes.append(node)
        return node


    def plan(self, start, iterations=500, reuse=True):
        if not self.nodes or not reuse:
            self.nodes = [Node(start)]

        for _ in range(iterations):
            self.add_node(self.informed_sample())

        return self.best_path()

    def best_path(self):
        best = max(self.nodes, key=lambda n: n.info)
        path = []
        while best:
            path.append(best.state)
            best = best.parent
        return path[::-1]



def belief_after_node(belief: BeliefMap, node: Node):
    p = belief.p.copy()
    for key, val in node.delta.items():
        j = key // belief.W
        i = key % belief.W
        p[j, i] = val
    return p


def visualize_all(belief: BeliefMap, planner: IATigris, best_node: Node):
    path = []
    n = best_node
    while n:
        path.append(n.state)
        n = n.parent
    path = path[::-1]

    updated_belief = belief_after_node(belief, best_node)
    entropy_updated = np.vectorize(entropy)(updated_belief)

    plt.figure(figsize=(18, 8))

    plt.subplot(2, 3, 1)
    plt.title("Initial Belief")
    plt.imshow(belief.p, cmap='hot')

    plt.subplot(2, 3, 2)
    plt.title("Initial Entropy")
    plt.imshow(belief._entropy_cache, cmap='viridis')

    plt.subplot(2, 3, 3)
    plt.title("Tree + Best Path")
    plt.imshow(belief.p, cmap='gray')
    for n in planner.nodes:
        if n.parent:
            plt.plot([n.state[0], n.parent.state[0]],
                     [n.state[1], n.parent.state[1]],
                     color='lime', alpha=0.15)
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    plt.plot(xs, ys, '-r', linewidth=2)

    plt.subplot(2, 3, 4)
    plt.title("Belief After Executing Best Path")
    plt.imshow(updated_belief, cmap='hot')

    plt.subplot(2, 3, 5)
    plt.title("Entropy After Path")
    plt.imshow(entropy_updated, cmap='viridis')

    plt.subplot(2, 3, 6)
    plt.title("Trajectory (orientation)")
    plt.imshow(updated_belief, cmap='gray')
    for (x, y, yaw) in path:
        plt.arrow(x, y, 4*math.cos(yaw), 4*math.sin(yaw), color='cyan')

    plt.tight_layout()
    plt.show()


def execute_and_replan(belief: BeliefMap, planner: IATigris, start, steps=4, iters=400):
    current = start
    final_path = []
    for k in range(steps):
        if belief._entropy_cache.sum() < 1e-6:
            break
        path = planner.plan(current, iterations=iters, reuse=True)
        best = max(planner.nodes, key=lambda n: n.info)


        belief.p = belief_after_node(belief, best)
        belief._entropy_cache = belief.entropy_map()

        if len(path) > 1:
            current = path[-1]

        for point in path:
            final_path.append(point)
        planner.nodes = [Node(current)]
    return final_path


def visualize_heatmap_path(belief, path, arrow_scale=4, save_path=None):

    plt.figure(figsize=(8, 8))
    plt.imshow(belief.p, cmap='hot', origin='lower')
    plt.colorbar(label='Belief Probability')
    plt.title("Belief Heatmap with Path")

    xs = [p[0] for p in path]
    ys = [p[1] for p in path]

    plt.scatter(xs[0], ys[0], color='lime', s=100, label='Start', zorder=5)
    plt.scatter(xs[-1], ys[-1], color='magenta', s=100, label='End', zorder=5)

    plt.plot(xs, ys, '-r', linewidth=2, label='Path')

    for (x, y, theta) in path:
        plt.arrow(x, y, arrow_scale * math.cos(theta), arrow_scale * math.sin(theta),
                  color='cyan', head_width=2, head_length=2)

    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def smooth_trajectory_savgol(traj, window_length=11, polyorder=3):

    traj = np.asarray(traj)

    x_smooth = savgol_filter(traj[:, 0], window_length, polyorder)
    y_smooth = savgol_filter(traj[:, 1], window_length, polyorder)

    return np.column_stack((x_smooth, y_smooth))



def generate_path_from_image(img):


    belief_map = np.array(img, dtype=np.float32)
    belief_map = belief_map / belief_map.sum() if belief_map.sum() > 0 else belief_map

    belief = BeliefMap(belief_map)
    planner = IATigris(
        belief,
        budget=500,
        cam_range=25,
        cam_fov=math.pi / 3
    )

    start = (
        random.uniform(0, belief_map.shape[1] / 2),
        random.uniform(0, belief_map.shape[0] / 2),
        random.uniform(-np.pi, np.pi),
    )
    start_time = time.time()

    final_path = execute_and_replan(belief, planner, start, 4)
    final_path = np.array(final_path)

    final_path[:, :2] = smooth_trajectory_savgol(final_path[:, :2], window_length=8)
    path_planned_time = time.time() - start_time



    return (path_planned_time, path_to_tensor(final_path))

def path_to_tensor(path_np, device='cuda'):

    path_tensor = torch.from_numpy(path_np).float().to(device)

    if path_tensor.ndim == 2:  # single trajectory [T,3]
        x, y, theta = path_tensor[:,0], path_tensor[:,1], path_tensor[:,2]
        new_tensor = torch.stack([x, y, torch.sin(theta), torch.cos(theta)], dim=1)
    elif path_tensor.ndim == 3:  # batch of trajectories [B,T,3]
        x, y, theta = path_tensor[:,:,0], path_tensor[:,:,1], path_tensor[:,:,2]
        new_tensor = torch.stack([x, y, torch.sin(theta), torch.cos(theta)], dim=2)
    else:
        raise ValueError("path_np must be 2D [T,3] or 3D [B,T,3]")

    return new_tensor



def find_largest_image_index(base_dir):
    import os
    import re

    pattern = re.compile(r"^image_(\d+)$")
    max_index = -1

    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path):
            match = pattern.match(name)
            if match:
                max_index = max(max_index, int(match.group(1)))

    return max_index


