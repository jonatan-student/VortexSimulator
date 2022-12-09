import pygame
import glob
import sys
from pathlib import Path
import os

pygame.init()
# get the path/directory
folder_dir = 'Figures'
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
step = 0
images = Path(folder_dir).glob('*.png')



GRAPHS = [pygame.image.load(os.path.join(image)) for image in images]

run = True
step = 0
TimeScale = 10
while run == True:
    #create white Background
    SCREEN.fill((255,255,255))
    SCREEN.blit(GRAPHS[step//TimeScale], (100,100))

    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    step +=1
    if step >= len(GRAPHS*TimeScale):
        step = 0
    pygame.display.update()
            