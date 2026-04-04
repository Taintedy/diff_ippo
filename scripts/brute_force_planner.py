import torch

class BruteForcePlanner:
    """
    Subdivide a belief map into n boxes and fill each box with a zig-zag trajectory.
    Each point is repeated 4 times with one direction each: up, down, left, right.
    """

    def __init__(self, belief_map, stride=8, device='cpu'):
        if belief_map.ndim == 2:
            belief_map = belief_map.unsqueeze(0)
        self.belief = belief_map.to(device).float()
        self.H, self.W = belief_map.shape[1], belief_map.shape[2]
        self.stride = stride
        self.device = device

    def plan(self, num_agents):
        box_widths = [self.W // num_agents] * num_agents
        remainder = self.W % num_agents
        box_widths[-1] += remainder

        trajs_list = []

        x_start = 0
        for w in box_widths:
            x_end = x_start + w
            xs = torch.arange(x_start, x_end, self.stride, device=self.device)
            ys = torch.arange(0, self.H, self.stride, device=self.device)

            traj_points = []
            for i, y in enumerate(ys):
                row_xs = xs if i % 2 == 0 else torch.flip(xs, dims=[0])
                for x in row_xs:
                    # Each point has 4 separate directions
                    traj_points.append([x.float(), y.float(), 0.0, -1.0])  # up
                    traj_points.append([x.float(), y.float(), 0.0, 1.0])   # down
                    traj_points.append([x.float(), y.float(), -1.0, 0.0])  # left
                    traj_points.append([x.float(), y.float(), 1.0, 0.0])   # right

            trajs_list.append(torch.tensor(traj_points, device=self.device))

            x_start = x_end

        # Pad trajectories to same length
        T_max = max(len(t) for t in trajs_list)
        trajs = torch.zeros((num_agents, T_max, 4), device=self.device)
        for a in range(num_agents):
            traj = trajs_list[a]
            L = traj.shape[0]
            trajs[a, :L, :] = traj
            if L < T_max:
                trajs[a, L:, :] = traj[-1]

        return trajs

