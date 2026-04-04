import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
from .model import ConditionalTemporalUnet
from diffusers import DDPMScheduler
import math


class DiffIPPPlanner():

    def __init__(self, model=None, num_train_timesteps=150, belief_dims=(256,256), visibility_calculation_type='vect', path_to_model=None, device="cpu", ):
        self.model = None
        self.device = device
        if model is not None:
            self.model = model
        elif path_to_model is not None:
            self.model = ConditionalTemporalUnet(
                belief_dim=256,
                dim_mults=(1, 2, 4, 8),
            )
            self.model.load_state_dict(torch.load(path_to_model))
        
        if self.model is None:
            print("model should be defined")
            return
        
        self.model.to(self.device)
        self.model.eval()


        self.num_train_timesteps = num_train_timesteps
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_timesteps, beta_schedule="scaled_linear", beta_start=1e-7, beta_end=0.02,
        )

        self.visibility_calculation_type = visibility_calculation_type

        if self.visibility_calculation_type == 'loop':
            y = torch.arange(belief_dims[0], device=self.device)
            x = torch.arange(belief_dims[1], device=self.device)
            self.yy, self.xx = torch.meshgrid(y, x, indexing="ij")  # (H,W)

            self.yy = self.yy[None]  # (1,H,W)
            self.xx = self.xx[None]
        elif self.visibility_calculation_type == 'vect':
            y = torch.arange(belief_dims[0], device=self.device)
            x = torch.arange(belief_dims[1], device=self.device)
            self.yy, self.xx = torch.meshgrid(y, x, indexing="ij")  # (H,W)

            self.yy = self.yy[None, None]  # (1,1,H,W)
            self.xx = self.xx[None, None]

        print("Planner created")



    def covariance_trace_loss(
        self,
        belief_before: torch.Tensor,
        belief_after: torch.Tensor,
        mode: str = "variance",
        normalize: bool = True,
        eps: float = 1e-8,
    ):
        """
        Differentiable covariance trace loss.

        Args:
            belief_before: (H, W) or (B, H, W), values in [0,1]
            belief_after:  (H, W) or (B, H, W), values in [0,1]
            mode: 'variance' or 'probability'
            normalize: if True, normalizes by trace_before
        Returns:
            loss (scalar tensor)
        """

        belief_before = belief_before.float()
        belief_after = belief_after.float()

        if mode == "variance":
            trace_before = belief_before.sum(dim=(-2, -1))
            trace_after = belief_after.sum(dim=(-2, -1))

        elif mode == "probability":
            var_before = belief_before * (1.0 - belief_before)
            var_after = belief_after * (1.0 - belief_after)

            trace_before = var_before.sum(dim=(-2, -1))
            trace_after = var_after.sum(dim=(-2, -1))

        else:
            raise ValueError("mode must be 'variance' or 'probability'")

        if normalize:
            loss = trace_after / (trace_before + eps)
        else:
            loss = trace_after

        return loss.mean()


    def visualize_visibility_with_trajectory_batch(
        self,
        beliefs,        
        trajectories,   
        fov_deg=60.0,
        max_range=10.0,
        alpha=10.0,
        beta=10.0,
        plot=True,      
    ):
        """
        Computes visibility for entire batch.
        
        If plot=True:
            Plots B x 3 images:
            [belief | visibility | belief*visibility]
        Else:
            Returns tensors only (FAST).
        """


        if beliefs.ndim == 4:
            beliefs = beliefs[:, 0]  # (B,H,W)

        device = beliefs.device
        B, H, W = beliefs.shape
        T = trajectories.shape[1]



        visibility = torch.zeros((B, H, W), device=device)
        fov_rad = math.radians(fov_deg) * 0.5

        for t in range(T):
            x_t = trajectories[:, t, 0][:, None, None]
            y_t = trajectories[:, t, 1][:, None, None]

            theta = torch.atan2(
                trajectories[:, t, 2],
                trajectories[:, t, 3]
            )[:, None, None]

            dx = self.xx - x_t
            dy = self.yy - y_t

            dist = torch.sqrt(dx**2 + dy**2)
            angle = torch.atan2(dy, dx)

            angle_diff = torch.atan2(
                torch.sin(angle - theta),
                torch.cos(angle - theta)
            )

            range_w = torch.sigmoid(beta * (max_range - dist))
            angle_w = torch.sigmoid(alpha * (fov_rad - angle_diff.abs()))

            v_t = range_w * angle_w


            visibility = visibility + v_t - visibility * v_t 
        belief_visibility = beliefs * visibility

        if not plot:
            belief_visibility = belief_visibility.unsqueeze(1) 
            return visibility, belief_visibility


        fig, axs = plt.subplots(
            B, 3,
            figsize=(12, 4 * B),
            squeeze=False
        )

        beliefs_np = beliefs.cpu()
        visibility_np = visibility.cpu()
        belief_vis_np = belief_visibility.cpu()
        traj_np = trajectories.cpu()

        for b in range(B):
            imgs = [
                (beliefs_np[b], "Belief"),
                (visibility_np[b], "Visibility"),
                (belief_vis_np[b], "Belief × Visibility"),
            ]

            for i, (img, title) in enumerate(imgs):
                ax = axs[b, i]
                ax.imshow(img)
                ax.set_title(f"{title} (batch {b})")
                ax.axis("off")

                traj = traj_np[b]
                ax.plot(traj[:, 0], traj[:, 1], "r-o", linewidth=2, markersize=4)

                for p in traj:
                    ax.arrow(
                        p[0], p[1],
                        p[3] * 5,   
                        p[2] * 5,   
                        color="red",
                        head_width=2,
                        length_includes_head=True,
                    )

        plt.tight_layout()
        plt.show()
        belief_visibility = belief_visibility.unsqueeze(1) 
        return visibility, belief_visibility


    def belief_diff_loss_vect(
        self,
        beliefs,        
        trajectories,   
        fov_deg=60.0,
        max_range=10.0,
        alpha=10.0,
        beta=10.0,
    ):



        if beliefs.ndim == 4:
            beliefs = beliefs[:, 0] 

        B, H, W = beliefs.shape
        T = trajectories.shape[1]
        device = beliefs.device



        x_t = trajectories[..., 0][:, :, None, None]  
        y_t = trajectories[..., 1][:, :, None, None]

        sin_t = trajectories[..., 2][:, :, None, None]
        cos_t = trajectories[..., 3][:, :, None, None]


        dx = self.xx - x_t  
        dy = self.yy - y_t

        dist_sq = dx**2 + dy**2
        max_range_sq = max_range ** 2


        range_w = torch.sigmoid(beta * (max_range_sq - dist_sq))


        dot = dx * cos_t + dy * sin_t
        dist = torch.sqrt(dist_sq + 1e-6) 
        cos_angle_diff = dot / dist

        fov_rad = math.radians(fov_deg) * 0.5
        cos_fov = math.cos(fov_rad)


        angle_w = torch.sigmoid(alpha * (cos_angle_diff - cos_fov))

        v_t = range_w * angle_w  


        visibility = 1 - torch.prod(1 - v_t, dim=1)  

        belief_visibility = beliefs * visibility
        belief_visibility = belief_visibility.unsqueeze(1)

        return visibility, belief_visibility



    def denormalize_path(self, trajectory, max_val_x, max_val_y):
        tmp_traj = trajectory
        offset = torch.tensor([0.5, 0.5, 0.0, 0.0], dtype=tmp_traj.dtype).to(trajectory.device)
        tmp_traj += offset
        tmp_traj[:, :, 0] = tmp_traj[:, :, 0] * (max_val_x)
        tmp_traj[:, :, 1] = tmp_traj[:, :, 1] * (max_val_y)
        return tmp_traj

    def create_start_condition(self, start_pose, number_of_trajectories, x_size, y_size):
        start_pose_tensor = torch.as_tensor(
            start_pose,
            device=self.device,
            dtype=torch.float32,
        )

        x, y, yaw = start_pose_tensor

        start_condition = torch.stack([
            x,
            y,
            torch.sin(yaw),
            torch.cos(yaw),
        ]).unsqueeze(0).repeat(number_of_trajectories, 1)


        offset = torch.tensor(
            [
                x_size / 2.0,
                y_size / 2.0,
                0.0,
                0.0
            ],
            dtype=torch.float32,
            device=self.device,
        )

        start_condition = start_condition - offset

        start_condition[:, 0] = start_condition[:, 0] / x_size
        start_condition[:, 1] = start_condition[:, 1] / y_size
        
        return start_condition.to(self.device)



    def masked_laplacian_smoothing_xy(self, trajs, mask, alpha=0.5, iters=5):

        x = trajs.clone()

        if mask.shape[-1] == 1:
            pos_mask = mask.expand(-1, -1, 2)
        else:
            pos_mask = mask[..., :2]

        for _ in range(iters):
            prev_xy = x[:, :-2, :2]
            mid_xy  = x[:, 1:-1, :2]
            next_xy = x[:, 2:, :2]

            lap_xy = prev_xy + next_xy - 2 * mid_xy
            x[:, 1:-1, :2] = mid_xy + alpha * lap_xy

            x[..., :2] = pos_mask * trajs[..., :2] + (1 - pos_mask) * x[..., :2]

        return x



    def trajectory_cost(self, traj, smooth_w=0.1):

        dxy  = traj[:, 1:, :2] - traj[:, :-1, :2]
        ddxy = traj[:, 2:, :2] - 2*traj[:, 1:-1, :2] + traj[:, :-2, :2]

        length_cost = (dxy ** 2).sum(dim=(-1, -2))
        smooth_cost = (ddxy ** 2).sum(dim=(-1, -2))

        return length_cost + smooth_w * smooth_cost


    def smooth_heatmap_batch(self, batch_tensor, kernel_size=3, sigma=1.0):
        """
        Fast Gaussian smoothing using separable convolution.
        
        Args:
            batch_tensor: Tensor of shape [B, C, H, W]
            kernel_size: Size of Gaussian kernel (must be odd)
            sigma: Standard deviation of Gaussian
        
        Returns:
            Smoothed heatmaps of same shape
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        x = torch.arange(kernel_size, dtype=torch.float32, device=batch_tensor.device) - kernel_size // 2
        gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        
        gauss_1d = gauss_1d.view(1, 1, kernel_size, 1)  
        gauss_1d_t = gauss_1d.view(1, 1, 1, kernel_size) 
        
        padding = kernel_size // 2
        

        intermediate = F.conv2d(batch_tensor, 
                            gauss_1d.repeat(batch_tensor.shape[1], 1, 1, 1),
                            padding=(padding, 0),
                            groups=batch_tensor.shape[1])
        
        smoothed = F.conv2d(intermediate,
                        gauss_1d_t.repeat(batch_tensor.shape[1], 1, 1, 1),
                        padding=(0, padding),
                        groups=batch_tensor.shape[1])
        
        return smoothed

    def plan(
        self,
        belief_map,
        start_pose=None,
        number_of_trajectories=1,
        horizon=144,
        start_time=0,
        smooth_iteration=50,
        guidence_step=5,
        explore_guidence_scale=0.5,
        smooth_guidence_scale=0.00001,
        explore_loss_type="cov_trace",
        smooth_heatmap=False,
        heatmap_smooth_kernel=100,
        heatmap_smooth_sigma=20,
    ):
        device = self.device
        vis_loss_fn = nn.HuberLoss()
        do_optimization = explore_guidence_scale != 0 or smooth_guidence_scale != 0
        if belief_map.dim() == 3:
            belief_map = belief_map.unsqueeze(0).repeat(
                number_of_trajectories,
                1,
                1,
                1
            )

        elif belief_map.dim() == 2:
            belief_map = belief_map.unsqueeze(0).unsqueeze(1).repeat(
                number_of_trajectories,
                1,
                1,
                1
            )


        if smooth_heatmap:
            belief_map = self.smooth_heatmap_batch(belief_map, heatmap_smooth_kernel, heatmap_smooth_sigma).to(self.device)


        print(belief_map.shape)

        if number_of_trajectories == 0:
            number_of_trajectories = belief_map.shape[0]

        if start_time == 0 or start_time >= self.num_train_timesteps:
            start_time = self.num_train_timesteps - 1

        max_t = self.noise_scheduler.timesteps[0].item()
        offset = max_t - start_time

        has_start = start_pose is not None


        mask = torch.zeros(
            (number_of_trajectories, horizon, 4),
            device=device,
        )

        if has_start:
            mask[:, 0, :] = 1.0
            start_condition = self.create_start_condition(
                start_pose,
                number_of_trajectories,
                belief_map.shape[2],
                belief_map.shape[3],
            )
            start_condition = start_condition.to(device)

            start_condition = start_condition.unsqueeze(1).expand(-1, horizon, -1)


        sample = torch.randn(
            (number_of_trajectories, horizon, 4),
            device=device,
        )

        if has_start:
            sample[:, 0, :] = start_condition[:, 0, :]


        timesteps = self.noise_scheduler.timesteps[offset:]
        timestep_tensor = torch.tensor(timesteps, device=device)

        noise_cache = torch.randn(
            (len(timesteps), *sample.shape),
            device=device,
        )


        for i, t in enumerate(timestep_tensor):

            with torch.no_grad(), torch.autocast(device_type="cuda"):

                t_batch = torch.full(
                    (sample.shape[0],),
                    t,
                    device=device,
                    dtype=torch.long,
                )

                residual = self.model(sample, belief_map, t_batch)            


            if do_optimization:
                
                sample = sample.detach().requires_grad_()

                opt_ready = False
                cond_grad = 0
                pred_original_guiding = self.noise_scheduler.step(
                    residual,
                    t,
                    sample
                ).pred_original_sample
                denormed_sample = self.denormalize_path(pred_original_guiding, 
                                                        belief_map.shape[2], 
                                                        belief_map.shape[3])
                vis_loss = 0
                if i%guidence_step == 0:
                    if explore_guidence_scale != 0:
                        
                        flatten_samples = torch.flatten(denormed_sample, end_dim=1)
                        if self.visibility_calculation_type == "loop":
                            visibility, vis_belief = self.visualize_visibility_with_trajectory_batch(
                                    belief_map[0].to(device),
                                    flatten_samples[None, :, :].to(device),
                                    max_range=25, alpha=5, beta=1, plot=False
                                )
                        elif self.visibility_calculation_type == 'vect':
                            visibility, vis_belief = self.belief_diff_loss_vect(
                                    belief_map[0].to(device),
                                    flatten_samples[None, :, :].to(device),
                                    max_range=25, alpha=5, beta=1
                                )
                            
                        if explore_loss_type == "huber":
                            vis_loss = vis_loss_fn(vis_belief, belief_map) * explore_guidence_scale
                        elif explore_loss_type == "cov_trace":
                            delta_bel = belief_map - vis_belief
                            vis_loss = explore_guidence_scale * self.covariance_trace_loss(belief_map, delta_bel)
                        opt_ready |= True
                    
                    smooth_loss = 0
                    if smooth_guidence_scale != 0:

                        smooth_loss = self.trajectory_cost(denormed_sample).mean() * smooth_guidence_scale
                        opt_ready |= True


                    total_loss = vis_loss + smooth_loss
                    if opt_ready:
                        cond_grad -= torch.autograd.grad(total_loss, sample)[0]

                        alpha_bar = self.noise_scheduler.alphas_cumprod[i]
                        sample = (
                            sample.detach() + cond_grad * alpha_bar.sqrt()
                        )


            with torch.no_grad():
                pred_original = self.noise_scheduler.step(
                    residual,
                    t,
                    sample
                ).pred_original_sample
            if has_start:

                pred_guided = (
                    start_condition * mask +
                    pred_original * (1 - mask)
                )
            else:
                pred_guided = pred_original


            if has_start and smooth_iteration > 0:
                pred_guided[:, :10] = self.masked_laplacian_smoothing_xy(
                    pred_guided[:, :10],
                    mask[:, :10],
                    alpha=0.5,
                    iters=smooth_iteration,
                )


            if i < len(timestep_tensor) - 1:

                sample = self.noise_scheduler.add_noise(
                    pred_guided,
                    noise_cache[i] * (1 - mask),
                    t,
                )

            else:
                sample = pred_guided


        final_paths = self.denormalize_path(
            sample,
            belief_map.shape[2],
            belief_map.shape[3],
        ).detach()
        return final_paths