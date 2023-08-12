#import tensorflow.keras as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import pygame

import tensorflow.keras as tf
from tensorflow.keras import layers

class Joystick:
    def __init__(self, joystick_id=0):
        self.joystick = pygame.joystick.Joystick(joystick_id)
        self.joystick.init()
        self.num_axes = self.joystick.get_numaxes()

    def get_axis_values(self):
        pygame.event.pump()  # Update the event queue
        axis_values = [self.joystick.get_axis(i) for i in range(self.num_axes)]
        return axis_values

def createNet(structure, activF):
    '''
    name = tf.Sequential()
    name.add(tf.layers.Input(shape=(structure[0])))
    for i in range(len(structure) - 1):
        name.add(tf.layers.Dense(structure[i + 1], activation=activF))
    '''

    name = tf.Sequential()
    name.add(layers.Dense(structure[1], activation=activF, input_shape=(structure[0],)))
    for i in range(len(structure) - 1):
        name.add(layers.Dense(structure[i + 1], activation=activF))

    name.compile(loss='mean_squared_error', optimizer='adam')
    return name


def mutateNet(net, r):
    w = net.get_weights()
    w_new = w
    #for i in np.arange(0, len(w) - 1, 2): # num of matrices for NN is always (nLayers-1)*2. Between each layer and the next, there is a matrix. And each matrix has 1 bias layer.
    for i in [random.choice([2*i for i in range(int(len(w)/2))])]:
        for j in range(len(w[i][:, 0])):
            for k in range(len(w[i][0, :])):
                # w_new[i][j,k] = np.random.choice([w[i][j,k],w[i][j,k]+random.uniform(-.5,.5)],1,p=[1-r,r])
                #w_new[i][j,k] = np.random.choice([w[i][j,k], np.random.randn(), w[i][j,k]+random.uniform(-.5,.5)], 1, p=[1-r,r*1/10,r*9/10])
                #w_new[i][j, k] = np.random.choice([w[i][j, k], w[i][j, k] + random.uniform(-.5, .5)], 1, p=[1 - r, r])
                w_new[i][j, k] = np.random.choice([w[i][j, k], np.random.randn()], 1, p=[1 - r, r])

    mutant = tf.models.clone_model(net)
    mutant.set_weights(w_new)
    return mutant

'''
def mutateBias(net, r):
    w = net.get_weights()
    w_new = w
    for i in np.arange(1, len(w), 2):
        for j in range(len(w[i])):
            # w_new[i][j] = np.random.choice([w[i][j,k],w[i][j,k]+random.uniform(-.5,.5)],1,p=[1-r,r])
            w_new[i][j] = np.random.choice([w[i][j], w[i][j] + random.uniform(-.5, .5)], 1, p=[1 - r, r])

    mutant = tf.models.clone_model(net)
    mutant.set_weights(w_new)
    return mutant
    '''
def set0(net):
    w = net.get_weights()
    w_new = w
    for i in np.arange(0, len(w) - 1, 2):
        for j in range(len(w[i][:, 0])):
            for k in range(len(w[i][0, :])):
                w_new[i][j, k] = 0

    model = tf.models.clone_model(net)
    model.set_weights(w_new)
    return model


def suvat(s0, v0, a0, dt):
    s = s0 + v0 * dt + 0.5 * a0 * dt ** 2
    v = v0 + a0 * dt
    return [s, v]


class Drone:
    def __init__(self, number, state, targets):
        self.number = number
        [self.position, self.velocity, self.acceleration] = state
        self.brain = createNet([6,8,8,2], 'tanh')
        self.targets = targets
        self.iTarget = 0
        self.close_counter = 0
        self.dt = 0.05
        self.dist = Drone.get_dist(self)
        self.max_thrust = 60
        self.radius = .215/2
        self.I = 0.266
        self.mass = 1
        self.g = -100
        self.alive = True
        self.thrust = [[],[]]

    def mutate(self, r):
        self.brain = mutateNet(self.brain, r)
    def get_dist(self):
        return ((self.targets[self.iTarget][0] - self.position[0])**2 + (self.targets[self.iTarget][1] - self.position[1])**2)**0.5
    def update_state(self): #Physics update
        #net_inputs = [[element for sublist in [self.position,self.velocity] for element in sublist]]
        net_inputs = [element for sublist in [self.position,self.velocity] for element in sublist]
        for i in range(2): net_inputs[i] = self.targets[self.iTarget][i] - net_inputs[i]
        net_inputs = np.array([net_inputs])

        # add noise
        #for i in range(len(net_inputs[0])):
        #    net_inputs[0][i] = net_inputs[0][i] + np.random.rand() * net_inputs[0][i] * 0.0001

        #[[L_thrust, R_thrust]] = self.mass / 2 * (-self.g) + self.max_thrust*self.brain.predict(net_inputs, verbose=0)# percentage of thrust for tanh
        [[L_thrust, R_thrust]] = self.brain(net_inputs)*5
        #L_thrust = abs(L_thrust)
        #R_thrust = abs(R_thrust)
        L_thrust = min(self.mass / 2 * (-self.g) + L_thrust + np.random.rand()*0.0001, self.max_thrust)
        R_thrust = min(self.mass / 2 * (-self.g) + R_thrust + np.random.rand()*0.0001, self.max_thrust)

        #xtar = np.tanh((self.targets[0][0] - self.position[0]))
        #ytar = np.tanh((self.targets[0][1] - self.position[1]))

        #R_thrust += - self.position[2] - self.velocity[2] + 0.1*(self.velocity[0] - self.velocity[1])
        #L_thrust += + self.position[2] + self.velocity[2] - 0.1*(self.velocity[0] - self.velocity[1])

        self.thrust[0].append(L_thrust)
        self.thrust[1].append(R_thrust)

        F = (L_thrust + R_thrust)
        M = (-L_thrust + R_thrust) * self.radius

        self.acceleration[2] = M / self.I
        [self.position[2], self.velocity[2]] = suvat(self.position[2], self.velocity[2], self.acceleration[2], self.dt)

        # x
        Fx = -F * np.sin(self.position[2])
        self.acceleration[0] = Fx / self.mass
        [self.position[0], self.velocity[0]] = suvat(self.position[0], self.velocity[0], self.acceleration[0], self.dt)

        # y
        Fy = F * np.cos(self.position[2]) + self.g * self.mass
        self.acceleration[1] = Fy / self.mass
        [self.position[1], self.velocity[1]] = suvat(self.position[1], self.velocity[1], self.acceleration[1], self.dt)
    def kill(self):
        self.alive = False
    def set_state(self,state):
        [self.position, self.velocity, self.acceleration] = state

class Game():
    def __init__(self, screen_size, character_size):
        [self.w, self.h] = screen_size
        [self.char_w, self.char_h] = character_size
        pygame.init()
        self.win = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("AI drone")
        self.im_drone = pygame.transform.scale(pygame.image.load('drone.png'), (self.char_w, self.char_h))
        self.im_target = pygame.transform.scale(pygame.image.load("red_dot.png"), (200, 200))
        self.scores = [-1e6]
        self.r = 1
        self.break_angle = 1e6
        self.r_auto = True
        self.best_weights = 0
        self.targets = [0,0]
        self.drones = 0
        self.max_counter = 1e8
        self.max_dist = 1e8

    def update(self, drones):
        self.win.fill([255, 255, 255])
        for i in range(len(drones)):
            if drones[i].alive:
                [target_x, target_y] = drones[i].targets[drones[i].iTarget]
                [x_new, y_new, th_new] = [drones[0].position[j].numpy() for j in range(3)]
                rotated_image = pygame.transform.rotate(self.im_drone, th_new * 180 / 3.14159265359)
                self.win.blit(rotated_image, (
                (self.w - (self.char_w * abs(np.cos(th_new)) + self.char_h * abs(np.sin(th_new)))) / 2 + x_new,
                (self.h - (self.char_w * abs(np.sin(th_new)) + self.char_h * abs(np.cos(th_new)))) / 2 - y_new))
                self.win.blit(self.im_target, ((self.w - 200) / 2 + target_x, (self.h - 200) / 2 - target_y))

    def pause(self):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    plt.plot(self.scores[1:])
                    plt.show()
                elif event.key == pygame.K_r:
                    self.r_auto = not self.r_auto
                    print("Auto mutation rate is set to ", self.r_auto)
                elif event.key == pygame.K_UP:
                    self.break_angle += 5
                    print("Break angle = ", self.break_angle)
                elif event.key == pygame.K_DOWN:
                    self.break_angle -= 5
                    print("Break angle = ", self.break_angle)
                elif event.key == pygame.K_LEFT:
                    if not self.r_auto:
                        self.r -= 0.0001
                        self.r = min(1,self.r)
                        print("r = ", self.r)
                elif event.key == pygame.K_RIGHT:
                    if not self.r_auto:
                        self.r += 0.0001
                        self.r = min(1,self.r)
                        print("r = ", self.r)
                elif event.key == pygame.K_m:
                    if not self.r_auto:
                        self.r = float(input("set mutation rate"))
                        self.r = min(1,self.r)
                        print("r = ", self.r)
    def quit_game(self):
        run = True
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    run = False
        return run
    def quit(self):
        pygame.quit()



def run_test(game, drones, max_score):
    # need to try only reward for going closer
    run = True
    pi = np.pi
    dist = [[drones[i].get_dist()] for i in range(len(drones))]
    theta = [[abs(drones[i].position[2])] for i in range(len(drones))]
    omega = [[0] for i in range(len(drones))]
    v = [[0] for i in range(len(drones))]

    penalty = [[0] for i in range(len(drones))]
    #vx = [[0] for i in range(len(drones))]
    dist0 = dist[0][0]
    counter = 0
    while run:
        for i in range(len(drones)):
            if drones[i].alive:
                game.pause()
                #if drones[i].get_dist() > 5:
                dist[i].append(drones[i].get_dist())
                #else:
                omega[i].append(abs(drones[i].velocity[2]))
                v[i].append((drones[i].velocity[1]**2 + drones[i].velocity[1]**2)**0.5)
                theta.append(abs(drones[i].position[2]))

                #vx[i].append(drones[i].velocity[0])

                drones[i].update_state()
                '''
                if max_score < 0:
                    if dist[i][-1] > dist[i][-2]:
                        drones[i].kill()
                '''
                if abs(drones[i].position[2]) > game.break_angle*pi/180:
                    drones[i].kill()
                elif drones[i].get_dist() > game.max_dist:
                    drones[i].kill()

                if dist[i][-1] < 8:
                    drones[i].close_counter += 1
                    if drones[i].close_counter > 400:
                        drones[i].iTarget += 1
                        dist[i] = []
                        counter = 0
                else:
                    if v[i][-1] < 1:
                        penalty[i].append(dist[i][-1])
                    else:
                        penalty[i].append(0)
                    drones[i].close_counter = 0

        game.update(drones)
        pygame.display.update()

        if all([not drones[i].alive for i in range(len(drones))]):
            run = False
        else:
            run = game.quit_game()
        counter += 1
        #print(counter)
        if counter > game.max_counter:
            run = False

    #scores = [drones[i].iTarget + 1 / (1 + np.mean(dist[i]) ) + 1 / (1 + np.mean(theta[i]) + np.mean(omega[i]) + np.mean(v[i]) ) for i in range(len(drones))]
    #scores = [drones[i].iTarget + 1 / (1 + np.mean(dist[i]) ) - np.mean(penalty[i]) for i in range(len(drones))]
    scores = [drones[i].iTarget + 1 / (1 + np.mean(dist[i])) for i in range(len(drones))]

    #scores = [1 / (1 + np.mean(dist[i])) for i in range(len(drones))]

    #scores = [len(dist[i]) / (1+np.mean(dist[i]) + np.mean(omega[i])) for i in range(len(drones))]
    #scores = [1 / (1 + np.mean(dist[i])) * 1 / (1 + np.mean(theta[i]) + np.mean(omega[i]) + np.mean(v[i])) for i in
    #          range(len(drones))]

    max_index, max_score = max(enumerate(scores), key=lambda x: x[1])

    if game.r_auto:
        game.r = min(1,0.0000084*np.mean(dist[max_index])**2)

    #plt.plot(vx[0])
    #plt.show()

    return [max_index, max_score]
