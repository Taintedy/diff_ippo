import torch
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.figsize": (8,6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "lines.linewidth": 2
})

def compute_visibility(trajectories, H, W, fov_deg=60.0, max_range=25.0, alpha=5.0, beta=1.0):
    device = trajectories.device
    B, T, _ = trajectories.shape

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    yy = yy[None]
    xx = xx[None]

    visibility = torch.zeros((B,H,W), device=device)
    fov_rad = math.radians(fov_deg)/2.0

    for t in range(T):
        x = trajectories[:,t,0][:,None,None]
        y = trajectories[:,t,1][:,None,None]
        theta = torch.atan2(trajectories[:,t,2], trajectories[:,t,3])[:,None,None]

        dx = xx - x
        dy = yy - y
        dist = torch.sqrt(dx*dx + dy*dy)
        angle = torch.atan2(dy, dx)
        angle_diff = torch.atan2(torch.sin(angle - theta), torch.cos(angle - theta))
        range_w = torch.sigmoid(beta * (max_range - dist))
        angle_w = torch.sigmoid(alpha * (fov_rad - angle_diff.abs()))
        v_t = range_w * angle_w
        visibility = visibility + v_t - visibility*v_t

    return visibility

def expected_detection_normalized(beliefs, trajectories,  fov, max_range, alpha=5.0, beta=1.0):
    if beliefs.ndim == 4:
        beliefs = beliefs[:,0]
    visibility = compute_visibility(trajectories, beliefs.shape[1], beliefs.shape[2], fov_deg=fov, max_range=max_range, alpha=alpha, beta=beta)
    ed = (beliefs * visibility).sum(dim=(1,2))
    total_belief = beliefs.sum(dim=(1,2))
    return (ed / total_belief).detach().cpu().numpy(), visibility

def coverage_normalized(visibility):
    return visibility.mean(dim=(1,2)).detach().cpu().numpy()

def path_length(trajectories):
    pos = trajectories[:,:,0:2]
    diff = pos[:,1:] - pos[:,:-1]
    dist = torch.sqrt((diff**2).sum(dim=-1))
    return dist.sum(dim=1).detach().cpu().numpy()

def detection_curve(belief, trajectory,  fov, max_range, alpha=5.0, beta=1.0):
    H,W = belief.shape
    belief = belief.unsqueeze(0)
    trajectory = trajectory.unsqueeze(0)
    distances, probs = [], []
    for t in range(1, trajectory.shape[1]+1):
        vis = compute_visibility(trajectory[:,:t], H, W, fov_deg=fov, max_range=max_range, alpha=alpha, beta=beta)[0]
        p = (belief[0] * vis).sum().item() / belief.sum().item()  # normalized
        d = path_length(trajectory[:,:t])[0]
        distances.append(d)
        probs.append(p)
    return distances, probs

def plot_detection_curve(belief, trajectory, fov, max_range, alpha=5.0, beta=1.0):
    dist, prob = detection_curve(belief, trajectory, fov=fov, max_range=max_range, alpha=alpha, beta=beta)
    plt.figure()
    plt.plot(dist, prob, label="Expected Detection")
    plt.xlabel("Distance (pixels)")
    plt.ylabel("Normalized Detection")
    plt.title("Normalized Expected Detection vs Distance")
    plt.ylim(0,1)
    plt.grid(True)
    plt.show()

def coverage_curve(belief, trajectory,  fov, max_range, alpha=5.0, beta=1.0):

    H, W = belief.shape
    distances, covs = [], []

    trajectory = trajectory.unsqueeze(0)  

    for t in range(1, trajectory.shape[1]+1):
        vis = compute_visibility(trajectory[:,:t], H, W, fov_deg=fov, max_range=max_range, alpha=alpha, beta=beta)[0]  # take batch 0
        cov = vis.mean().item() 
        d = path_length(trajectory[:,:t])[0]
        distances.append(d)
        covs.append(cov)
    return distances, covs

def plot_coverage_curve(belief, trajectory, fov, max_range, alpha=5.0, beta=1.0):
    dist, cov = coverage_curve(belief, trajectory, fov=fov, max_range=max_range, alpha=alpha, beta=beta)
    plt.figure()
    plt.plot(dist, cov, label="Coverage")
    plt.xlabel("Distance (pixels)")
    plt.ylabel("Normalized Coverage")
    plt.title("Normalized Coverage vs Distance")
    plt.ylim(0,1)
    plt.grid(True)
    plt.show()

def plot_trajectory(belief, trajectory):
    plt.figure()
    plt.imshow(belief.cpu(), cmap="viridis")
    x = trajectory[:,0].cpu()
    y = trajectory[:,1].cpu()
    plt.plot(x, y, color="red", linewidth=1.5)
    plt.scatter(x[0], y[0], color="white", label="start")
    plt.title("Trajectory on Belief Map")
    plt.legend()
    plt.show()








def redundancy_metric(trajectories, H, W, fov_deg=60.0, max_range=25.0, alpha=10.0, beta=10.0):

    B = trajectories.shape[0]
    vis = compute_visibility(trajectories, H, W, fov_deg, max_range, alpha, beta)

    soft_or = 1 - torch.prod(1 - vis, dim=0)

    total_vis = vis.sum()
    redundant = (vis.sum(dim=0) - soft_or).sum()

    redundancy = redundant / (total_vis + 1e-8)

    return redundancy.item()


def exploration_efficiency(expected_detection, lengths):

    return (expected_detection / lengths).detach().cpu().numpy()


def max_detection_region(beliefs, visibility):

    detected_prob = beliefs * visibility

    max_h = detected_prob.max(dim=1)[0]   
    max_hw = max_h.max(dim=1)[0]          
    return max_hw.detach().cpu().numpy()


def spatial_spread(trajectories):

    final_pos = trajectories[:,-1,0:2] 
    B = final_pos.shape[0]
    if B < 2:
        return 0.0
    dists = torch.sqrt(((final_pos[:,None,:] - final_pos[None,:,:])**2).sum(-1))
    return (dists.sum() - dists.trace()) / (B*(B-1))


def trajectory_smoothness(trajectories):

    pos = trajectories[:,:,0:2]  
    vec = pos[:,1:] - pos[:,:-1]  
    norms = torch.sqrt((vec**2).sum(-1, keepdim=True))
    vec_normed = vec / (norms + 1e-6)
    dot = (vec_normed[:,1:] * vec_normed[:,:-1]).sum(-1)
    dot = torch.clamp(dot, -1.0, 1.0)
    angles = torch.acos(dot) 
    return angles.mean().item() * 180 / math.pi


def evaluate_search_planner(beliefs, trajectories, plot=True, verbos=True, fov=60, max_range=25, alpha=5.0, beta=1.0):

    B,H,W = beliefs.shape[0], beliefs.shape[2], beliefs.shape[3]
    
    ed, visibility = expected_detection_normalized(beliefs, trajectories, fov, max_range, alpha=alpha, beta=beta)  # lists/arrays of shape [B]
    
    cov = coverage_normalized(visibility)  
    
    lengths = path_length(trajectories)  
    
    red = redundancy_metric(trajectories, H, W, fov_deg=fov, max_range=max_range, alpha=alpha, beta=beta)
    
    eff = exploration_efficiency(torch.tensor(ed), torch.tensor(lengths))
    
    max_det = max_detection_region(beliefs[:,0], visibility)
    
    smooth = trajectory_smoothness(trajectories)
    
    if verbos:
        print("\nEvaluation Metrics (normalized)")
        print("--------------------------------")
        for i,(p,l,c) in enumerate(zip(ed, lengths, cov)):
            print(f"Traj {i}: Norm ExpectedDetect={p:.4f}, Length={l:.1f}, Norm Coverage={c:.4f}")
        print(f"Redundancy: {red}")
        print(f"Exploration efficiency: {eff}")
        print(f"Max detection probability: {max_det}")
        print(f"Trajectory smoothness (rad): {smooth}")
    
    if plot:
        plot_detection_curve(beliefs[0,0], trajectories[0], fov=fov, max_range=max_range, alpha=alpha, beta=beta)
        plot_coverage_curve(beliefs[0,0], trajectories[0], fov=fov, max_range=max_range, alpha=alpha, beta=beta)
        plot_trajectory(beliefs[0,0], trajectories[0])
    
    metrics_dict = {
        "per_trajectory": [
            {
                "norm_expected_detection": float(ed[i]),
                "path_length": float(lengths[i]),
                "norm_coverage": float(cov[i])
            }
            for i in range(B)
        ],
        "redundancy": red,
        "exploration_efficiency": eff,
        "max_detection_probability": max_det,
        "trajectory_smoothness_rad": smooth
    }
    
    return metrics_dict






def aggregate_metrics(metrics_list):

    if len(metrics_list) == 0:
        return {}

    num_trajectories = len(metrics_list[0]["per_trajectory"])

    norm_expected_detection = [[] for _ in range(num_trajectories)]
    path_length = [[] for _ in range(num_trajectories)]
    norm_coverage = [[] for _ in range(num_trajectories)]

    redundancy = []
    exploration_efficiency = []
    max_detection_probability = []
    trajectory_smoothness_rad = []

    for m in metrics_list:
        for i, traj_metrics in enumerate(m["per_trajectory"]):
            norm_expected_detection[i].append(traj_metrics["norm_expected_detection"])
            path_length[i].append(traj_metrics["path_length"])
            norm_coverage[i].append(traj_metrics["norm_coverage"])
        redundancy.append(m["redundancy"])
        exploration_efficiency.append(m["exploration_efficiency"])
        max_detection_probability.append(m["max_detection_probability"])
        trajectory_smoothness_rad.append(m["trajectory_smoothness_rad"])

    mean_per_trajectory = []
    for i in range(num_trajectories):
        mean_per_trajectory.append({
            "norm_expected_detection": float(np.mean(norm_expected_detection[i])),
            "path_length": float(np.mean(path_length[i])),
            "norm_coverage": float(np.mean(norm_coverage[i]))
        })

    mean_metrics = {
        "per_trajectory": mean_per_trajectory,
        "redundancy": float(np.mean(redundancy)),
        "exploration_efficiency": float(np.mean(exploration_efficiency)),
        "max_detection_probability": float(np.mean(max_detection_probability)),
        "trajectory_smoothness_rad": float(np.mean(trajectory_smoothness_rad))
    }

    return mean_metrics