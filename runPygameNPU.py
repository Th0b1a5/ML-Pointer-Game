import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pygame as pg
import random as ran
import time
import math
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
device = "NPU" # CHANGE THIS TO CHANGE TARGET PROCESSING DEVICE, "CPU" or "GPU" or "NPU" (if available)

xWidth = 1000
yWidth = 1000
lenght = 400

xCenter = xWidth // 2
yCenter = yWidth // 2

xWidth = min(xWidth, 1000)
yWidth = min(yWidth, 1000)

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

def drawMan(size, cursorX, cursorY, ai_angle, real_angle, currXsize = 25, currYsize = 25):
    pg.draw.rect(screen, (0,255,0), (cursorX - (currXsize  / 2), cursorY - (currYsize  / 2), currXsize, currYsize))
    pg.draw.circle(screen, (255,0,0), (xCenter, yCenter), size)

    # AI line
    pg.draw.line(screen, (200,0,0), (xCenter, yCenter), (xCenter + (math.cos(ai_angle) * lenght), yCenter + math.sin(ai_angle) * lenght), 5)
    pg.draw.line(screen, (0,200,0), (xCenter, yCenter), (xCenter + (math.cos(real_angle) * lenght * 1.6), yCenter + math.sin(real_angle) * lenght * 1.6), 5)

if __name__ == '__main__':
    import openvino as ov
    core = ov.Core()

    available_device = core.available_devices
    device = device if device in available_device else "CPU"
    print(f"Available {available_device} devices, using {device} device")

    model_name = input("Model to load: ")
    print("(normal - nor) for normal (random - ran) for random")
    setting = input("Select the current mode of the runtime: ")

    logs = pd.DataFrame(columns=["ai", "real", "diff"])
    print(logs)

    pg.init()
    screen = pg.display.set_mode((xWidth, yWidth))
    pg.display.set_caption("Pointer game")
    font = pg.font.SysFont("Arial", 38)

    # 1. Print the brain
    model = PointerNetwork()

    # 2. Load the trained memories (sliders) into the brain
    model.load_state_dict(torch.load(f"models/{model_name}.pth", weights_only=True))
    model.eval() # Tell the brain it's testing time, not training time

    last_dx = None
    last_dy = None
    wait = None
    running = True

    ex_input = input_tensor = torch.tensor([[1, 2]], dtype=torch.float32)
    vino_model = ov.convert_model(model, example_input=ex_input)
    vino_model.reshape([1, 2])
    npu_model = core.compile_model(vino_model, "NPU")

    while running:
        if setting == "nor" or setting == "normal":
            dx, dy = pg.mouse.get_pos()
            dx = dx - xCenter
            dy = dy - yCenter
            wait = 0.02
        else:
            dx = ran.uniform(-xCenter, xCenter)
            dy = ran.uniform(-yCenter, yCenter)
            wait = 1.4

        if dx == last_dx and dy == last_dy:
            last_dx = dx
            last_dy = dy
            pass
        else:
            
            # Convert and pust it thru the network
            input_tensor = torch.tensor([[dx, dy]], dtype=torch.float32)
            with torch.torch.inference_mode(): # Don't do calculus, just guess
                predicted_radian = npu_model(input_tensor)[0].item()

            predicted_degrees = math.degrees(predicted_radian)
            perfect_radians = np.arctan2(dy, dx)
            perfect_degrees = math.degrees(perfect_radians)
            diffrence_degrees = abs(predicted_degrees - perfect_degrees)

            print(f"AI: {predicted_radian} radians, {predicted_degrees}")
            print(f"Real: {perfect_radians} radians, {perfect_degrees}")

            values = [predicted_degrees, perfect_degrees, diffrence_degrees]
            logs.loc[len(logs)] = values
            print(f"Logged: {values}")
            
            last_dx = dx
            last_dy = dy

        screen.fill((255,255,255))
        drawMan(25, dx + xCenter, dy + yCenter, predicted_radian, perfect_radians)

        prediction_text = font.render(f"AI: {str(predicted_degrees)}", True, (40, 44, 52))
        perfect_text = font.render(f"Real: {str(perfect_degrees)}", True, (30, 54, 50))
        diffrence_text = font.render(f"Diff: {str(diffrence_degrees)}", True, (120,30,30))
        screen.blit(prediction_text, (0,0))
        screen.blit(perfect_text, (0,42))
        screen.blit(diffrence_text, (500,0))
        pg.display.update()

        for event in pg.event.get():
            if event.type == pg.QUIT:
                logs.to_csv(f"performance/{model_name}.csv", index=False)
                #print(logs)
                running = False

        time.sleep(wait)
