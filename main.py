from image_gen import ImageGen
from nn_gen import BasicNet
import time


def moovy():
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


def piccy():
    img_g = ImageGen()
    scale = 0.5
    offset_x = scale - scale / 1.618
    offset_y = scale - scale / 1.618
    for i in range(10):
        nn = BasicNet(neurons=32)
        img = img_g.generate_image(nn, steps=2000, scale=scale, offset_x=offset_x, offset_y=offset_y)
        # img.show()
        img.save(f'img_gen/{time.time()}.png')


    # print(result)


if __name__ == '__main__':
    # moovy()
    piccy()
