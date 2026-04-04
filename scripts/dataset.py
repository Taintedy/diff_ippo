import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt




def resample_trajectory_xytheta(traj, target_len):
    """
    Resample (x, y, theta) trajectory to fixed length using arc-length.

    traj: (T, 3)
    target_len: int
    returns: (target_len, 3)
    """

    if len(traj) == 1:
        return np.repeat(traj, target_len, axis=0)

    xy = traj[:, :2]
    theta = traj[:, 2]

    deltas = np.diff(xy, axis=0)
    dists = np.linalg.norm(deltas, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(dists)])
    total_length = arc[-1]

    if total_length == 0:
        return np.repeat(traj[:1], target_len, axis=0)

    new_arc = np.linspace(0, total_length, target_len)

    new_xy = np.zeros((target_len, 2), dtype=np.float32)
    new_xy[:, 0] = np.interp(new_arc, arc, xy[:, 0])
    new_xy[:, 1] = np.interp(new_arc, arc, xy[:, 1])

    theta_unwrapped = np.unwrap(theta)
    new_theta = np.interp(new_arc, arc, theta_unwrapped)

    new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi

    return np.column_stack([new_xy, new_theta]).astype(np.float32)

def xytheta_to_xy_sincos(traj):
    """
    traj: (T, 3)
    returns: (T, 4)
    """
    x, y, theta = traj[:, 0], traj[:, 1], traj[:, 2]
    return np.column_stack([
        x,
        y,
        np.sin(theta),
        np.cos(theta)
    ]).astype(np.float32)


class BeliefTrajectoryDataset(Dataset):
    """
    Input : belief map (1, H, W)
    Output: trajectory (T, 4) -> x, y, sinθ, cosθ
    """

    def __init__(
        self,
        root_dir,
        image_size=(256, 256),
        trajectory_len=100,
        normalize_belief=True
    ):
        self.root_dir = root_dir
        self.image_size = image_size
        self.trajectory_len = trajectory_len
        self.normalize_belief = normalize_belief

        self.samples = sorted(
            [
                os.path.join(root_dir, d)
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ],
            key=lambda x: int(os.path.basename(x).split('_')[-1])
        )

        if len(self.samples) == 0:
            raise RuntimeError("No dataset folders found!")

    def __len__(self):
        return len(self.samples)

    def _load_belief(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to load belief map: {path}")

        img = cv2.resize(img, self.image_size)
        img = img.astype(np.float32)

        if self.normalize_belief:

            img /= 255

        return torch.from_numpy(img).unsqueeze(0)

    def _load_trajectory(self, path):
        traj = np.load(path).astype(np.float32)

        traj = resample_trajectory_xytheta(traj, self.trajectory_len)
        traj = xytheta_to_xy_sincos(traj)

        return torch.from_numpy(traj)

    def __getitem__(self, idx):
        folder = self.samples[idx]

        belief = self._load_belief(os.path.join(folder, "belief_map.png"))
        trajectory = self._load_trajectory(os.path.join(folder, "path.npy"))

        return belief, trajectory



def visualize_sample(belief, trajectory, arrow_step=5, title=None):
    """
    belief: (1, H, W)
    trajectory: (T, 4) -> x, y, sinθ, cosθ
    """

    belief = belief.squeeze().cpu().numpy()
    traj = trajectory.cpu().numpy()

    x, y = traj[:, 0], traj[:, 1]
    theta = np.arctan2(traj[:, 2], traj[:, 3])

    plt.figure(figsize=(6, 6))
    plt.imshow(belief, cmap="gray", origin="lower")

    plt.plot(x, y, 'r-', linewidth=2)

    plt.quiver(
        x[::arrow_step],
        y[::arrow_step],
        np.cos(theta[::arrow_step]),
        np.sin(theta[::arrow_step]),
        color="blue",
        scale=30
    )

    plt.scatter(x[0], y[0], c='green', label="start")
    plt.scatter(x[-1], y[-1], c='black', label="end")

    if title:
        plt.title(title)

    plt.legend()
    plt.axis("off")
    plt.show()


def visualize_batch(beliefs, trajectories, arrow_step=10, max_cols=4, title=None):
    """
    beliefs: (B, 1, H, W)
    trajectories: (B, T, 4) -> x, y, sinθ, cosθ
    """

    beliefs = beliefs.cpu().numpy()
    trajectories = trajectories.cpu().numpy()

    B = beliefs.shape[0]
    cols = min(B, max_cols)
    rows = int(np.ceil(B / cols))

    plt.figure(figsize=(4 * cols, 4 * rows))

    for i in range(B):
        plt.subplot(rows, cols, i + 1)

        belief = beliefs[i, 0]
        traj = trajectories[i]

        x, y = traj[:, 0], traj[:, 1]
        theta = np.arctan2(traj[:, 2], traj[:, 3])

        plt.imshow(belief, cmap="gray", origin="lower")
        plt.plot(x, y, 'r-', linewidth=2)

        plt.quiver(
            x[::arrow_step],
            y[::arrow_step],
            np.cos(theta[::arrow_step]),
            np.sin(theta[::arrow_step]),
            color="blue",
            scale=30
        )

        plt.scatter(x[0], y[0], c='green', s=20)
        plt.scatter(x[-1], y[-1], c='black', s=20)

        plt.grid()

        plt.title(f"Sample {i}")

    if title:
        plt.suptitle(title)

    plt.tight_layout()

    plt.show()


DATASET_ROOT_DIR = ""
if __name__ == "__main__":

    import os

    num_workers = min(8, os.cpu_count() // 2)
    print(num_workers)
    dataset = BeliefTrajectoryDataset(
        root_dir=DATASET_ROOT_DIR,
        image_size=(256, 256),
        trajectory_len=128
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    import time
    for i in range(10):
        start = time.time()
        beliefs, trajectories = next(iter(loader))
        print(time.time() - start)
    visualize_batch(
        beliefs,
        trajectories,
        arrow_step=10,
        title="Training Batch Visualization"
    )


