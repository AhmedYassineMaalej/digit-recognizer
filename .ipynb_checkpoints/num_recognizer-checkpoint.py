import os, sys
from math import sqrt

import numpy as np
import pygame
from tensorflow.keras.models import load_model

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model = load_model(r"C:\Users\MSI\Desktop\Code\Python\Number Recognizer\my_model.h5")


pygame.init()
info = pygame.display.Info()

screen_width = info.current_w
screen_height = info.current_h


window_width = screen_width // 3

red = (255, 0, 0)

pixel_dim = window_width // 28
window_width = pixel_dim * 28


class Pixel:
    def __init__(self, row:int , col:int) -> None:
        self.row = row
        self.col = col
        self.activation = 0
        self.center_x, self.center_y = self.get_center()
    

    def get_center(self) -> tuple[int, int]:
        self.x = self.col * pixel_dim 
        self.y = self.row * pixel_dim 

        center_y = self.y + pixel_dim // 2
        center_x = self.x + pixel_dim // 2

        return (center_x, center_y)


    def get_dis(self, mouse_pos:tuple[int, int]):
        mouse_x, mouse_y = mouse_pos
        x_dis = mouse_x - self.center_x
        y_dis = mouse_y - self.center_y

        return sqrt(x_dis ** 2 + y_dis ** 2)


    def activate(self, mouse_pos):
        dis = self.get_dis(mouse_pos)
        self.activation = max(self.activation, int(f(dis)))

    def draw(self):
        color = (self.activation,) * 3
        pygame.draw.rect(screen, color, (self.x, self.y, pixel_dim, pixel_dim))


def f(x):
    return  - 0.001 * (x ** 4) + 255

def draw_pixels():
    for pixel in pixels:
        pixel.draw()

def activate_pixels(mouse_pos):
    for pixel in pixels:
        pixel.activate(mouse_pos)


screen = pygame.display.set_mode((window_width, window_width))

image = [[Pixel(row, col) for row in range(28)] for col in range(28)]

pixels = [image[i][j] for i in range(28) for j in range(28)]


def main():
    
    mouse_pressed = False
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()


            elif ev.type == pygame.MOUSEBUTTONDOWN:
                mouse_pressed = True
            
            elif ev.type == pygame.MOUSEBUTTONUP:
                mouse_pressed = False

            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_RETURN:
                pygame.quit()
                return [[image[i][j].activation for i in range(28)] for j in range(28)]

            elif ev.type == pygame.MOUSEMOTION and mouse_pressed:
                mouse_pos = pygame.mouse.get_pos()
                activate_pixels(mouse_pos)
        
        draw_pixels()
        pygame.display.update()


data = main()


data_arr = np.array(data)
data_arr = np.array([data_arr])
prediction = model.predict(data_arr)
print("the number you wrote is: ", end="")
print(np.argmax(prediction[0]))

sys.exit()