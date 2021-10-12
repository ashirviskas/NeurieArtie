import numpy as np
import torch
from PIL import Image


class ImageGen:
    def __init__(self):
        pass

    def get_grid(self, scale=5, steps=500, offset_x=0, offset_y=0):
        start_x = -scale + offset_x
        end_x = scale + offset_x

        start_y = -scale + offset_y
        end_y = scale + offset_y
        x = np.linspace(start_x, end_x, steps)
        y = np.linspace(start_y, end_y, steps)
        x, y = np.meshgrid(x, y)
        xy = np.vstack((x.flatten(), y.flatten())).T
        xy = torch.tensor(xy).type(torch.float)
        return xy

    def get_image(self, result, steps=500):
        img_arr = result.cpu().detach().numpy().reshape((steps, steps, 3))
        img_arr = (img_arr * 255).astype(np.uint8)
        img = Image.fromarray(img_arr)
        return img

    def generate_image(self, nn, scale=5, steps=500, offset_x=0, offset_y=0):
        grid = self.get_grid(scale=scale, steps=steps, offset_x=offset_x, offset_y=offset_y)
        result = nn(grid)
        img = self.get_image(result, steps)
        return img
