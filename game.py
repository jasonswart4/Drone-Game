from functions import *
import matplotlib.pyplot as plt
import numpy as np
from random import uniform
import pickle
# Initialising the game
character_size = [64, 64]
screen_size = [1500, 790]
game = Game(screen_size, character_size)

targets = [[0,20], [400,30], [-10,20], [50,10], [0,10]]
#targets = [[uniform(-350,350), uniform(-350,350)] for _ in range(200)] # randomly generated targets
drones = [Drone(0,[[0,0,0],[0,0,0],[0,0,0]],targets)]
scores = [0]
max_score = 1e6
game.break_angle = 1e6
game.max_counter = 1e6
game.max_dist = 2000
drones[0].dt = 0.005

run_test(game, drones, max_score)
plt.figure(1)
plt.plot(drones[0].thrust[0])
plt.plot(drones[0].thrust[1])
plt.show()

'''
data = drones[0].data

with open('data.pkl', 'wb') as file:
    pickle.dump(data, file)
'''