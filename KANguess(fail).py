import torch
from efficient_kan import KAN  # Import your KAN model
from PIL import Image
import torchvision.transforms as transforms

# Initialize the model
model = KAN([28 * 28, 64, 10])  # Ensure this matches the architecture used during training

# Load the trained weights
# model.load_state_dict(torch.load('kan_mnist_model.pth'))
model.load_state_dict(torch.load('/absolute/path/to/kan_mnist_model.pth'))

model.eval()  # Set the model to evaluation mode

# Define the same transformations used during training
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize image to 28x28
    transforms.Grayscale(),       # Convert to grayscale if not already
    transforms.ToTensor(),        # Convert image to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize
])

# Load and preprocess the image
image_path = 'handwrittenExamples/8hw.heic'
image = Image.open(image_path)
image = transform(image)
image = image.unsqueeze(0)  # Add batch dimension

# Make predictions
with torch.no_grad():
    output = model(image)  # Forward pass
    _, predicted = torch.max(output, 1)  # Get the predicted digit

# Convert tensor to digit
predicted_digit = predicted.item()
print(f'The predicted digit is: {predicted_digit}')
