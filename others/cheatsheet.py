import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# You must paste your PointerNetwork class blueprint here so Python knows the shape of the brain!
class PointerNetwork(nn.Module):
    def __init__(self):
        super(PointerNetwork, self).__init__()
        # The Hardware: 2 inputs (dx, dy) -> 16 -> 16 -> 1 output (angle)
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)
        
    def forward(self, x):
        # The Wiring: How data moves through the hardware
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model_name = input("Model to load: ")

# 1. Print the brain
model = PointerNetwork()

# 2. Load the trained memories (sliders) into the brain
model.load_state_dict(torch.load(f"models/{model_name}.pth", weights_only=True))
model.eval() # Tell the brain it's testing time, not training time

# 3. Ask it a question
# Let's say your mouse is exactly 100 pixels right and 100 pixels down from the center
dx = 100.0
dy = 100.0

# Convert the raw numbers into a PyTorch Tensor
input_tensor = torch.tensor([[dx, dy]], dtype=torch.float32)

# Get the prediction (and turn the tensor back into a normal Python float)
with torch.no_grad(): # Don't do calculus, just guess
    predicted_radian = model(input_tensor).item()

print(f"The AI points at: {predicted_radian} radians")
print(f"Against perfect score of {np.arctan2(dx, dy)}")