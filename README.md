# KAN-MNIST: Kolmogorov-Arnold Networks for MNIST Classification

This repository demonstrates the application of **Kolmogorov-Arnold Networks (KAN)** for the classification of handwritten digits from the MNIST dataset.The original implementation of KAN is available [here](https://github.com/KindXiaoming/pykan).
<!--
## Project Overview

The MNIST dataset is a benchmark dataset in the machine learning community, containing 70,000 images of handwritten digits (0-9). Our approach utilizes Kolmogorov-Arnold Networks to build a classification model. This project provides an implementation of KAN applied to the MNIST dataset, exploring how KANs can perform digit classification efficiently.

### Key Features:
- **Custom Implementation of KANs:** We built KANs from scratch for classification purposes.
- **Efficient Network Training:** The model is optimized using PyTorch and achieves a good balance between training speed and accuracy.
- **Interactive Visualizations:** The repository includes tools to visualize the learned representations and performance of the model on the MNIST data.

## Approach

1. **Dataset Preparation:**  
   The MNIST dataset is downloaded, preprocessed, and normalized to suit the input requirements of the Kolmogorov-Arnold Network.

2. **KAN Implementation:**  
   We implemented the KAN architecture, following the Kolmogorov-Arnold representation theorem, which decomposes functions into simpler components for efficient computation.

3. **Training the Model:**  
   The model is trained using PyTorch. It minimizes the cross-entropy loss while optimizing the weights of the KAN layers. The model training process is configurable to fine-tune hyperparameters like learning rate, batch size, and epochs.

4. **Evaluation:**  
   Post-training, the model is evaluated on the test set to measure its accuracy in classifying handwritten digits. Visualization of results helps in understanding model performance.

5. **Saving the Model:**  
   The trained model is saved to a file (`kan_mnist_model.pth`), which can later be loaded for inference or further fine-tuning.

You're right! Since the MNIST dataset is being directly downloaded from `torchvision`, you don't need to give it as a reference in the sense of a separate external link. Instead, you can clarify in the README that the dataset is automatically fetched by PyTorch when the code is executed, eliminating the need for manual intervention.

I'll adjust the README to reflect this:

---

## Dataset

The MNIST dataset used in this project is automatically downloaded using PyTorch's `torchvision.datasets` package. It requires no manual download, and the necessary preprocessing (such as normalization) is handled by the code. When the project is made to run, the dataset will be stored in the `data/` folder.

The code for loading the dataset is as follows:

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
```

The dataset is fetched directly from PyTorch's utilities.

## Project Structure

- `KANguess.py`: Contains the implementation of the Kolmogorov-Arnold Network.
- `kan_mnist.py`: Script to train and evaluate the model on the MNIST dataset.
- `efficient_kan.py`: A more optimized version of the KAN implementation.
- `data/`: Folder containing the MNIST dataset (downloaded on first run).
- `handwrittenExamples/`: Folder containing some examples of handwritten digits for testing.
- `kan_env/`: Virtual environment configuration for the project.
- `kan_mnist_model.pth`: The pre-trained model file, saved after training.

  KAN-MNIST/
│
├── handwrittenExamples/   # Directory containing example images for prediction
│   └── m8.png             # Example image
│
├── efficient_kan.py       # Definition of the KAN model
├── KANguess.py            # Script to make predictions on new images
├── train.py               # Script to train the KAN model
├── requirements.txt       # List of required packages
└── README.md              # Project documentation

## Running the Project Locally

To run this project on your local machine, follow the steps below:

### Prerequisites
Ensure you have the following installed on your machine:
- Python 3.8 or higher
- PyTorch
- Git

### Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/CheetahCodes21/KAN-MNIST.git
   cd KAN-MNIST
   ```

2. **Create a Virtual Environment:**
   It's recommended to use a virtual environment to manage dependencies.

   ```bash
   python3 -m venv kan_env
   source kan_env/bin/activate  # On Mac/Linux
   kan_env\Scripts\activate  # On Windows
   ```

3. **Install Dependencies:**
   After activating the virtual environment, install the required Python packages using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, if you don't have a `requirements.txt` file, install the dependencies manually:
   
   ```bash
   pip install torch torchvision matplotlib
   ```

4. **Run the Training Script:**
   To train the Kolmogorov-Arnold Network on the MNIST dataset, run the following command:

   ```bash
   python kan_mnist.py
   ```

   This will begin the training process, and the model's progress will be displayed in the terminal.

5. **Evaluate the Model:**
   After training is complete, you can evaluate the model by running the test script:

   ```bash
   python kan_mnist.py --test
   ```

6. **Use the Pretrained Model:**
   If you want to use the pre-trained model to classify digits without retraining, you can load the model from `kan_mnist_model.pth`:

   ```bash
   python kan_mnist.py --load_model
   ```

## Visualizing Results

We provide a set of visualizations that display the performance of the network. You can run the visualization script to view the network's predictions on the test data:

```bash
python visualize_results.py
```

## References

- [Kolmogorov-Arnold Representation Theorem](https://www.sciencedirect.com/science/article/pii/S0893608021000289)
- [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/pdf/2404.19756)
- [efficient-kan](https://github.com/Blealtan/efficient-kan)
-->

## Overview

The MNIST dataset is a benchmark dataset in the machine learning community, containing 70,000 images of handwritten digits (0-9). This project explores how **Kolmogorov-Arnold Networks (KANs)** can be applied to build an efficient and effective classification model.

### Key Features
- **Custom KAN Implementation:** We built the KAN architecture from scratch.
- **Optimized Training:** PyTorch is used for fast training with a balance of speed and accuracy.
- **Interactive Visualizations:** Explore the learned representations and performance metrics through provided visual tools.

## Approach

1. **Dataset Preparation**  
   The MNIST dataset is automatically downloaded via PyTorch's `torchvision.datasets`. It is preprocessed and normalized to meet the input requirements for KAN.

2. **KAN Implementation**  
   The KAN architecture is implemented following the Kolmogorov-Arnold representation theorem, which decomposes complex functions into simpler components for efficient computation.

3. **Training the Model**  
   The model is trained using PyTorch, minimizing cross-entropy loss while optimizing KAN layer weights. Hyperparameters such as learning rate, batch size, and epochs can be configured.

4. **Evaluation**  
   After training, the model is evaluated on the test set for accuracy in classifying digits. Visualizations are provided for better understanding.

5. **Saving the Model**  
   The trained model is saved to `kan_mnist_model.pth` for future inference or fine-tuning.

## Dataset

The MNIST dataset is automatically downloaded using PyTorch's `torchvision.datasets` package. No manual download is required; the dataset is stored in the `data/` folder upon running the code.

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
```

## Project Structure

```
KAN-MNIST/
├── handwrittenExamples/   # Example images for prediction
│   └── m8.png             # Sample image for testing
├── efficient_kan.py       # Optimized KAN model
├── KANguess.py            # Prediction script for new images
├── train.py               # Training script for KAN
├── requirements.txt       # Required packages
└── README.md              # Documentation
```

## Running the Project Locally

### Prerequisites

Ensure the following are installed:
- Python 3.8+
- PyTorch
- Git

### Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/CheetahCodes21/KAN-MNIST.git
   cd KAN-MNIST
   ```

2. **Create a Virtual Environment:**

   ```bash
   python3 -m venv kan_env
   source kan_env/bin/activate  # On Mac/Linux
   kan_env\Scripts\activate     # On Windows
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Training Script:**

   ```bash
   python kan_mnist.py
   ```

5. **Evaluate the Model:**

   ```bash
   python kan_mnist.py --test
   ```

6. **Use the Pretrained Model:**

   ```bash
   python kan_mnist.py --load_model
   ```

## Model Architecture

The KAN model is defined in [`efficient_kan.py`](https://github.com/CheetahCodes21/KAN-MNIST/blob/main/efficient_kan.py). The architecture used in this project is a custom neural network with the following layers:


- Custom linear layers `KANLinear` that incorporate B-splines for interpolation.
- The `KAN` model is initialized with a list of hidden layer sizes, and it constructs a series of `KANLinear` layers based on this list.

### Example Initialization

```python
model = KAN([28 * 28, 64, 10])  # Example with input size 28*28, one hidden layer of size 64, and output size 10
```

## Making Predictions

The [`KANguess.py`](https://github.com/CheetahCodes21/KAN-MNIST/blob/main/KANguess.py) script is used to make predictions on new images. It loads the trained model, preprocesses the input image, and outputs the predicted digit.

1. Place your image in the [`handwrittenExamples`](https://github.com/CheetahCodes21/KAN-MNIST/tree/main/handwrittenExamples) directory.
2. Update the `image_path` variable in [`KANguess.py`](https://github.com/CheetahCodes21/KAN-MNIST/blob/main/KANguess.py) to point to your image.
3. Run the script:

The `KANguess.py` script is used to make predictions on new images. It loads the trained model, preprocesses the input image, and outputs the predicted digit. To use this script for predictions:

```bash
python KANguess.py
```

Make sure the input image is a grayscale image of a handwritten digit (similar to the MNIST dataset format) for accurate results.

## References

- [Kolmogorov-Arnold Representation Theorem](https://www.sciencedirect.com/science/article/pii/S0893608021000289)
- [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/pdf/2404.19756)
- [efficient-kan](https://github.com/Blealtan/efficient-kan)
