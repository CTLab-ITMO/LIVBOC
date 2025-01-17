import torch
import numpy as np

def curvature_smoothness_loss(points):
    """
    Curvature-based smoothness loss penalizes high curvature by computing the second derivative.
    Args:
        points: Tensor of shape (N, 2), where N is the number of points.
    Returns:
        curvature_loss: Scalar tensor representing the curvature-based smoothness loss.
    """
    # Calculate second derivative to penalize curvature
    curvature_loss = torch.sum((points[2:] - 2 * points[1:-1] + points[:-2]) ** 2)
    return curvature_loss



def local_angle_penalty(points, ideal_angle=torch.pi, reduction='mean'):
    """
    Compute a local angle penalty loss for a tensor of points.

    Args:
        points (torch.Tensor): A tensor of shape (N, 2), where each row is (x, y).
        ideal_angle (float): The ideal angle (in radians) between consecutive segments. Default is pi (straight line).
        reduction (str): Specifies the reduction to apply to the output: 'mean', 'sum', or 'none'.

    Returns:
        torch.Tensor: The computed local angle penalty loss.
    """
    # Calculate direction vectors between consecutive points
    vectors = points[1:] - points[:-1]  # Shape: (N-1, 2)

    # Normalize the direction vectors
    norms = torch.norm(vectors, dim=1, keepdim=True) + 1e-6
    unit_vectors = vectors / norms

    # Compute dot products between consecutive unit vectors to get cos(theta)
    cos_angles = (unit_vectors[:-1] * unit_vectors[1:]).sum(dim=1).clamp(-1.0, 1.0)  # Shape: (N-2)
    cos_angles = cos_angles.clamp(-1.0 + 1e-6, 1.0 - 1e-6)

    # Compute angles (theta) between consecutive segments
    angles = torch.acos(cos_angles)  # Shape: (N-2)

    # Penalize deviations from the ideal angle
    angle_differences = torch.abs(angles - ideal_angle)  # Shape: (N-2)
    penalty_weight = 0.01  # Adjust based on your application
    # print(angle_differences.mean().item())
    # Apply reduction
    if reduction == 'mean':
        return angle_differences.mean()*penalty_weight
    elif reduction == 'sum':
        return angle_differences.sum()*penalty_weight
    elif reduction == 'none':
        return angle_differences*penalty_weight
    else:
        raise ValueError("Invalid reduction type. Choose 'mean', 'sum', or 'none'.")

def smoothness_loss(x_list, smoothness_weight=1.0, scale=1e-1):  # x[ npoints,2]
    smoothness_loss = 0.
    for x in x_list:
        smoothness_loss += curvature_smoothness_loss(x)

    loss = smoothness_weight * smoothness_loss
    # print("smoothness_loss: ", smoothness_loss.item())
    return loss / (len(x_list)**2)

if __name__ == "__main__":
    # Test cases for the curvature-based smoothness loss
    x = torch.tensor([[0,0], [1,1], [2,1], [0.5,0]], dtype=torch.float32)
    scale = 1
    y = smoothness_loss([x], scale)
    print(y)

    x = torch.tensor([[0,0], [1,1], [2,1], [2.,0]], dtype=torch.float32)
    scale = 1
    y = smoothness_loss([x], scale)
    print(y)
