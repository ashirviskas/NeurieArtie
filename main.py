from image_gen import ImageGen
from nn_gen import BasicNet
import time


def main():
    img_g = ImageGen()

    nn = BasicNet(neurons=16)
    scale = 5
    offset_x = -5.0
    for i in range(160):
        img = img_g.generate_image(nn, steps=1000, scale=scale, offset_x=offset_x)
        # img.show()
        img.save(f'vid_gen/{i:03}.png')
        scale -= scale * 0.01
        offset_x -= offset_x * 0.02

    # print(result)


if __name__ == '__main__':
    main()
