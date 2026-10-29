import torch
import onnxscript
import torch.nn as nn

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


model_name = input("Enter model name to export: ")

# 1. Print the brain
model = PointerNetwork()

# 2. Load the trained memories (sliders) into the brain
model.load_state_dict(torch.load(f"models/{model_name}.pth", weights_only=True))

model.eval()
torch.onnx.export(
    model, 
    torch.tensor([[1.0, 1.0]]), 
    f"{model_name}.onnx",
    input_names=["coordinates"],
    output_names=["predicted_angle"],
    external_data=False
)

