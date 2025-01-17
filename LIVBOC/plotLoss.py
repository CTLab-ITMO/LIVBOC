import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# Define a function to load and transform an image
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to 256x256 or your preferred size
        transforms.ToTensor()  # Convert image to tensor
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)  # Add batch dimension

# Define a function to calculate Euclidean loss
def euclidean_distance_loss(img1, img2):
    return torch.sqrt(torch.sum((img1 - img2) ** 2))

# Function to compute Euclidean loss across iterations
def compute_losses(image_folder, reference_image):
    losses = []
    for file in sorted(os.listdir(image_folder)):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, file)
            current_image = load_image(image_path).to(device)
            loss = euclidean_distance_loss(current_image, reference_image).item()
            losses.append(loss)
    return losses

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the reference image (the image to compare against)
reference_image_path = "path_to_reference_image.jpg"
reference_image = load_image(reference_image_path).to(device)

# Load images from each iteration folder
live_images_folder = "path_to_live_images"
my_method_images_folder = "C:\\Users\\asus\\PycharmProjects\\text2vecImg\\LIVE-Layerwise-Image-Vectorization\\LIVE\\log_contour_enh\\172628448941_dora8888\\video-png\\"

# Compute losses for both methods
live_losses = compute_losses(live_images_folder, reference_image)
my_method_losses = compute_losses(my_method_images_folder, reference_image)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(live_losses, label="LIVE Method", color="blue", linestyle='--')
plt.plot(my_method_losses, label="My Method", color="red", linestyle='-')
plt.xlabel("Iteration")
plt.ylabel("Euclidean Distance Loss")
plt.title("Comparison of Euclidean Loss Across Iterations")
plt.legend()
plt.grid(True)
plt.show()
