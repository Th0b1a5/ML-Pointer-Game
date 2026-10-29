import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. THE BLUEPRINT (The Class)
# This is just the Fusion360 schematic. It doesn't DO anything yet.
# ==========================================
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

# ==========================================
# 2. THE DATA LOADER BLUEPRINT
# ==========================================
class PointerDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.inputs = torch.tensor(self.data[['dx', 'dy']].values, dtype=torch.float32)
        self.targets = torch.tensor(self.data['ang'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# ==========================================
# 3. SETUP & INITIALIZATION (Printing the Drone)
# ==========================================
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

print("Loading data...")
dataset = PointerDataset("dataset/pointer_data.csv")
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

print("Printing the Brain...")
model = PointerNetwork()  # We just printed the blueprint into reality!

criterion = nn.MSELoss()  # The Loss Function (Reality Check)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # The Weight Adjuster

# ==========================================
# 4. THE TRAINING LOOP (The Test Flight)
# ==========================================
epochs = 250
print(f"Starting Training for {epochs} Epochs...")

for epoch in range(epochs):
    running_loss = 0.0
    
    for batch_inputs, batch_targets in data_loader:
        # Step 1: Clear the slate
        optimizer.zero_grad()
        
        # Step 2: The AI guesses the angle
        predictions = model(batch_inputs)
        
        # Step 3: Calculate the error (Compare guess to reality)
        # .unsqueeze(1) turns [32] into a column of 32 rows so shapes match
        loss = criterion(predictions, batch_targets.unsqueeze(1))
 
        # Step 4: Calculus (Find the slopes)
        loss.backward()
        
        # Step 5: Tweak the sliders
        optimizer.step()
        
        running_loss += loss.item()
        
    # Print the average error for this epoch
    avg_loss = running_loss / len(data_loader)
    print(f"Epoch {epoch+1} | Average Error (Loss): {avg_loss:.4f}")

# Save the brain!
#name = input("Model name: ")
name = "250pass1m"
torch.save(model.state_dict(), f"models/{name}.pth")
print("Training Complete. Brain Saved.")