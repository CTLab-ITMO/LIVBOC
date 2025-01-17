import math

import cv2
from utils import get_sdf
from scipy.ndimage import binary_fill_holes
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import torch
from skopt import forest_minimize
from skopt.space import Categorical, Real, Integer
fixed_attempts = 0
all_attempts = 0
last_pred = None
def plot_img(img, title=None):
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()  # Convert PyTorch tensor to NumPy array

    # If the image has more than 2 dimensions (e.g., HWC or CHW), handle accordingly
    if img.ndim == 3:
        if img.shape[0] == 3:  # Assume CHW format for PyTorch images
            img = np.transpose(img, (1, 2, 0))  # Convert to HWC
        elif img.shape[-1] != 3:  # Ensure last dimension is color channels
            raise ValueError("Unexpected image shape. Expected last dimension to be color channels.")
    plt.imshow(img, cmap='gray' if img.ndim == 2 else None)
    if title:
        plt.title(title)
    plt.axis('off')  # Turn off axis for better visualization
    plt.show()

class Contour_path_init():
    def __init__(self, pred, gt, format='[bs x c x 2D]', quantile_interval=800, nodiff_thres=0.5861487371810189):
        self.path_id = 0
        self.attempts = all_attempts
        if all_attempts <3:
            self.attempts = 0
        self.quantile_interval = quantile_interval
        self.nodiff_thres = nodiff_thres

        # Ensure pred and gt are on the GPU
        if isinstance(pred, torch.Tensor):
            self.pred = pred.cuda()  # Move to GPU if it is a tensor
        else:
            self.pred = torch.tensor(pred).cuda()  # Convert to tensor and move to GPU if not already

        if isinstance(gt, torch.Tensor):
            self.gt = gt.cuda()  # Move to GPU if it is a tensor
        else:
            self.gt = torch.tensor(gt).cuda()  # Convert to tensor and move to GPU if not already

        # Extract the first image from the batch
        pred_img = self.pred[0]
        gt_img = self.gt[0]

        # If the images have multiple channels (e.g., RGB), select one channel
        if pred_img.shape[0] == 3:
            pred_img = pred_img.permute(1, 2, 0)  # Convert (C, H, W) to (H, W, C)
            gt_img = gt_img.permute(1, 2, 0)
        self.gt_img = gt_img

        if format == '[bs x c x 2D]':
            # Move the difference calculation to the GPU
            self.map = ((self.pred[0] - self.gt[0])**2).sum(0)
            self.reference_gt = self.gt[0].cpu().numpy().transpose(1, 2,
                                                                   0)  # Keep this part on CPU for NumPy compatibility
        elif format == '[2D x c]':
            # Move the difference calculation to the GPU
            self.map = (torch.abs(self.pred - self.gt)).sum(-1)
            self.reference_gt = self.gt[0].cpu().numpy()  # Keep this part on CPU for NumPy compatibility
        else:
            raise ValueError("Invalid format specified.")

        self.loss = torch.tensor([1e10], device= self.pred.device)
        self.format = format
    def ensure_distinct_and_multiple_of_three(self, points):
        # Ensure points are on the GPU
        points = points.squeeze(dim=1).cuda() if points.is_cuda == False else points.squeeze(dim=1)

        num_points = points.shape[0]
        all_points = []
        targ_num_points =  int(num_points/2.1**math.log10(num_points)) + (3-int(2.1**math.log10(num_points))%2) # min(int(num_points**0.6) + (3-int(num_points**0.6)%2), 84)
        if targ_num_points <24:
            targ_num_points = 24
        print(num_points, targ_num_points, '================')
        if num_points > targ_num_points:
            total_num = 0
            skip = -(num_points // -targ_num_points)
            for i in range(num_points):
                if (i % skip == 0 and len(all_points) < targ_num_points) or points.shape[
                    0] - i <= targ_num_points - len(all_points):
                    all_points.append(points[i])
                    total_num = total_num + 1
            points = torch.stack(all_points)

        if points.shape[0] < 12:
            old_points_num = points.shape[0]
            needed_points_num = 12 - old_points_num
            for i in range(1, needed_points_num + 1):
                first_point = points[0]
                last_point = points[-1]
                new_point = last_point + (first_point - last_point) * i / (needed_points_num + 1)
                points = torch.cat((points, new_point.unsqueeze(0)), dim=0)

        # Ensure the number of points is a multiple of 3
        elif points.shape[0] % 3 != 0:
            points = points[:-(points.shape[0] % 3)]

        # Return points on the GPU, as a FloatTensor and flipped
        return points.flip(dims=[0]).to(torch.float32)

    def find_border_points(self, component):
        # Find contours (border points) using OpenCV
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Stack the contours into a single array
        border_points = np.vstack(contours)

        # Convert border points to a PyTorch tensor and move it to the GPU
        border_points = torch.FloatTensor(border_points)
        if torch.cuda.is_available():
            border_points = border_points.cuda()

        # Ensure the number of points is distinct and a multiple of three
        border_points = self.ensure_distinct_and_multiple_of_three(border_points)

        return border_points

    def calc_mean_color(self, component_binary=None, k=7, masked_pixels=None, mask=None, attempts=0):
        if masked_pixels is None:
            # Extract the mask region from the ground truth image (make sure it's on the GPU)
            masked_pixels = self.gt_img[component_binary == 1].cuda() if not self.gt_img.is_cuda else self.gt_img[
                component_binary == 1]
        print('len(masked_pixels)', len(masked_pixels))

        if len(masked_pixels.shape) > 2:
            if mask is not None:
                masked_pixels = masked_pixels[mask == 1]
            masked_pixels = masked_pixels.reshape(-1, 3)

        # Check if there are any valid pixels
        if len(masked_pixels) == 0:
            return np.array([0, 0, 0])  # Return black if no pixels found

        min_color = masked_pixels.min(dim=0).values
        max_color = masked_pixels.max(dim=0).values
        print(min_color, max_color, torch.all(torch.abs(max_color - min_color) < 0.01).item())

        # If color range is very small, return the mean of masked pixels
        if torch.all(torch.abs(max_color - min_color) < 0.01):
            return [
                masked_pixels.mean(dim=0).cpu().numpy().astype(np.float64)]  # Move to CPU for compatibility with NumPy

        # Use KMeans clustering (must move tensor to CPU for scikit-learn compatibility)
        kmeans = KMeans(n_clusters=min(k, len(masked_pixels)), random_state=0)
        kmeans.fit(masked_pixels.cpu().numpy())  # Convert to NumPy and perform clustering

        # Find the largest cluster
        counts = np.bincount(kmeans.labels_)
        most_frequent_cluster_idx = np.argmax(counts)
        most_frequent_color = [kmeans.cluster_centers_[most_frequent_cluster_idx]]

        return most_frequent_color

    def optimize_params_bayesian(self, num_iter=20, format='[bs x c x 2D]', attempts=0):
        def calculate_loss(quantile_interval, nodiff_thres, kernel_size, opacity, self, format='[bs x c x 2D]'):
            # Ensure tensors are on the correct device (GPU or CPU)
            device = self.pred.device

            if format == '[bs x c x 2D]':
                map = torch.abs(self.pred[0] - self.gt[0]).sum(0)**2
            elif format == '[2D x c]':
                map = (torch.abs(self.pred - self.gt)).sum(-1)
            else:
                raise ValueError("Unsupported format")

            print("calculating loss...", quantile_interval.item(), nodiff_thres.item(), kernel_size)

            # Apply thresholding directly on the GPU tensor
            map[map < nodiff_thres] = 0

            quantile_interval_vals = torch.linspace(0., 1., int(quantile_interval), device=device)

            quantized_interval = torch.quantile(map.flatten().to(torch.float32), quantile_interval_vals)
            quantized_interval = torch.unique(quantized_interval)
            quantized_interval = sorted(quantized_interval[1:-1])

            def torch_digitize(input_tensor, bins):
                # Ensure bins are sorted for the torch searchsorted function
                bins = torch.tensor(bins, device=input_tensor.device, dtype=input_tensor.dtype)
                return torch.bucketize(input_tensor, bins, right=False)

            map = torch_digitize(map, quantized_interval)

            map = map.clamp(0, 255).to(torch.uint8)
            # Apply Morphological Operations (Erosion followed by Dilation)
            kernel = np.ones((np.abs(kernel_size), np.abs(kernel_size)), dtype=np.uint8)

            if kernel_size > 0:
                if kernel_size < 0:
                    map = cv2.dilate(map.cpu().numpy(), kernel, iterations=1)
                    map = cv2.erode(map, kernel, iterations=1)
                else:
                    map = cv2.erode(map.cpu().numpy(), kernel, iterations=1)
                    map = cv2.dilate(map, kernel, iterations=1)

            idcnt = {}
            map = torch.tensor(map, dtype=torch.float32,
                               device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            for idi in sorted(torch.unique(map)):
                idcnt[idi.item()] = (map == idi).sum()

            idcnt.pop(min(idcnt.keys()))

            loss = torch.tensor(100000000000000., device=device)
            i = 1
            while i > 0 and len(idcnt) > 1:
                is_first_path = (torch.min(self.pred) == 1).item()
                is_white_background = False
                i -= 1
                target_id = max(idcnt, key=idcnt.get)

                _, component, cstats, ccenter = cv2.connectedComponentsWithStats(
                    (map == target_id).to(torch.uint8).cpu().numpy(), connectivity=4
                )

                csize = [ci[-1] for ci in cstats[1:]]
                target_cid = csize.index(max(csize)) + 1

                component_binary = (component == target_cid).astype(np.uint8)
                loss, mean_color, pred_filled = self.calc_l2(component_binary, opacity, attempts)
                component_binary = binary_fill_holes(component_binary).astype(np.uint8)
                if loss < self.loss:
                    self.component_binary = component_binary
                    self.loss = loss
                    self.best_color = mean_color
                    self.next_pred = pred_filled
                    print("New best color obtained:", self.best_color)
                    break
                if self.loss < 1e10:
                    break
                idcnt.pop(target_id)

            return loss.item()

        diff_map_thres = torch.abs(self.pred[0] - self.gt[0]).sum(0)**2# ((self.pred[0] - self.gt[0]) ** 2).sum(0)
        # Define the parameter space
        space = [
            Integer(10,2700, name='quantile_interval'),
            Real(diff_map_thres.min().item()/2,diff_map_thres.max().item(), prior='uniform', name='nodiff_thres'),
            Categorical([-9, -7, -5, -3, -2, 0, 2, 3, 5, 7, 9], name='kernel_size'),
        ]

        # Objective function to minimize
        def objective(params):
            quantile_interval, nodiff_thres, kernel_size = params
            opacity = 1.0

            # Move parameters to the GPU if available
            quantile_interval = torch.tensor(quantile_interval, device='cuda' if torch.cuda.is_available() else 'cpu')
            nodiff_thres = torch.tensor(nodiff_thres, device='cuda' if torch.cuda.is_available() else 'cpu')
            opacity = torch.tensor(opacity, device='cuda' if torch.cuda.is_available() else 'cpu')

            # Calculate loss and return it
            return calculate_loss(quantile_interval, nodiff_thres, kernel_size, opacity, self, format)

        # Run Bayesian Optimization with forest_minimize
        res = forest_minimize(
            objective,  # Objective function
            space,  # Parameter space
            n_calls=num_iter,  # Number of iterations
            verbose=False,  # Display optimization progress
            base_estimator="ET",  # Use Extra Trees as the estimator
        )

        # Retrieve the best found parameters
        best_quantile_interval = torch.tensor(res.x[0], device='cuda' if torch.cuda.is_available() else 'cpu')
        best_nodiff_thres = torch.tensor(res.x[1], device='cuda' if torch.cuda.is_available() else 'cpu')
        best_opacity = torch.tensor(1.0, device='cuda' if torch.cuda.is_available() else 'cpu')

        # Print the best parameters
        print(f"Best quantile_interval: {best_quantile_interval}")
        print(f"Best nodiff_thres: {best_nodiff_thres}")
        print(f"Best opacity: {best_opacity}")

        # Assign the best found values back to the object's properties
        self.opacity = best_opacity
        self.quantile_interval = best_quantile_interval
        self.nodiff_thres = best_nodiff_thres
        torch.cuda.empty_cache()

    def calc_l2(self, mask, opacity, attempts):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Apply binary closing to fill the gaps
        filled_mask = torch.tensor(
            binary_fill_holes(mask),
            dtype=torch.float32
        ).to(device)

        # Get the signed distance field weights
        sdf_weights = get_sdf(filled_mask.cpu().numpy(), method='skfmm', dx=1.0, normalize='lo1')
        sdf_weights_tensor = torch.tensor(sdf_weights, dtype=torch.float32, device=device, requires_grad=False)

        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device, requires_grad=False)
        masked_weights = mask_tensor * sdf_weights_tensor
        masked_weights2 = filled_mask * sdf_weights_tensor

        # Filtering based on the mean and max conditions
        mean = masked_weights.mean()
        max_val = masked_weights.max()
        masked_weights[(masked_weights < mean / 2) | (masked_weights > 0.8 * max_val + 0.2 * mean)] = 0
        masked_weights[masked_weights != 0] = 1

        mean = masked_weights2.mean()
        max_val = masked_weights2.max()
        global fixed_attempts
        if attempts>1:
            fixed_attempts = 1
        if attempts>2:
            fixed_attempts = 2
        if masked_weights.sum()==0 :#and fixed_attempts == 1:
            masked_weights = masked_weights2.clone()
            masked_weights[(masked_weights < mean / 2) | (masked_weights > 0.99 * max_val + 0.01 * mean)] = 0
            masked_weights[masked_weights != 0] = 1

        if masked_weights.sum()==0 :#and fixed_attempts == 2:
            masked_weights = masked_weights2
            masked_weights[masked_weights != 0] = 1
            if masked_weights.sum()<15:
                return torch.tensor([1e10], device=device), 0, 0
        gt_tensor = torch.tensor(self.gt, dtype=torch.float32, device=device, requires_grad=False)
        pred_tensor = torch.tensor(self.pred, dtype=torch.float32, device=device, requires_grad=False)
        mean_color = torch.tensor(
            self.calc_mean_color(masked_pixels=gt_tensor[0].permute(1, 2, 0)[masked_weights == 1],
                                 mask=mask, attempts=attempts), device=device
        )[0]

        print("Non-zero count in masked_weights:", (masked_weights != 0).sum().item())
        print('calc_mean_color: ', mean_color)

        try:
            mean_color_expanded = mean_color.view(1, 3, 1, 1)  # Shape: (1, 3, 1, 1)
        except:
            return torch.tensor([1e10], device=device), 0, 0

        mean_color_filled = mask_tensor[None, :, :] * mean_color_expanded  # Shape: (1, 3, H, W)
        mask_expanded = mask_tensor[None, :, :]  # Expand mask to (1, 1, H, W)
        real_mean_color_filled = filled_mask[None, :, :] * mean_color_expanded  # Shape: (1, 3, H, W)
        real_mask_expanded = filled_mask[None, :, :]  # Expand mask to (1, 1, H, W)
        pred_filled = pred_tensor * (1 - mask_expanded * opacity) + mean_color_filled * opacity
        diff_tensor = ((gt_tensor - pred_filled) ** 2).sum((0, 1))
        real_pred_filled = pred_tensor * (1 - real_mask_expanded * opacity) + real_mean_color_filled * opacity
        # plot_img(real_pred_filled[0])
        # Calculate overlap loss
        prev_loss = ((gt_tensor - pred_tensor) ** 2).sum((0, 1))
        overlap_loss = torch.sum(diff_tensor[(diff_tensor > prev_loss) & (prev_loss < 0.1)]) #* 10

        if diff_tensor.sum() >= prev_loss.sum(): #and self.loss >= torch.tensor([1e10], device=device):
            return torch.tensor([1e10], device=device), 0, 0
        # diff_tensor[diff_tensor>0.01] = 1
        # Compute loss with additional penalties
        loss = diff_tensor.sum() + overlap_loss.sum()*0.02 + 0.01 * torch.sum(mask_tensor == 0)
        print("Loss:", loss.item(), diff_tensor.sum().item(), overlap_loss.sum().item(),
              torch.sum(mask_tensor == 0).item())
        torch.cuda.empty_cache()
        return loss, mean_color, real_pred_filled
    def __call__(self):
        global all_attempts
        global last_pred
        self.attempts = all_attempts
        if self.attempts < 7:
            self.attempts = 0
        i = 0
        if last_pred is None:
            while self.loss >= 1e10:
                with torch.no_grad():
                    self.optimize_params_bayesian(format=self.format, attempts=self.attempts)
                    torch.cuda.empty_cache()
                if self.attempts <7:
                    self.attempts+=1
                i+=1
                if i ==8:
                    break
        else:
            self.component_binary = last_pred
        if i == 8:
            last_pred = self.component_binary
            self.opacity = 0
        component_binary = self.component_binary
        border_points = self.find_border_points(component_binary)
        mean_color = [torch.tensor(self.best_color[0]), torch.tensor(self.best_color[1]),
                      torch.tensor(self.best_color[2]), torch.tensor(self.opacity)]
        all_attempts = self.attempts
        #======================================================

        self.pred = self.next_pred
        self.loss = 1e10
        print('self.path_id : ', self.path_id )
        self.path_id+=1
        return {
            'center': [None, None],
            'border_points': border_points,
            'mean_color': mean_color
        }


class linear_decay_lrlambda_f(object):
    def __init__(self, decay_every, decay_ratio):
        self.decay_every = decay_every
        self.decay_ratio = decay_ratio

    def __call__(self, n):
        decay_time = n//self.decay_every
        decay_step = n %self.decay_every
        lr_s = self.decay_ratio**decay_time
        lr_e = self.decay_ratio**(decay_time+1)
        r = decay_step/self.decay_every
        lr = lr_s * (1-r) + lr_e * r
        return lr