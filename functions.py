import random
import numpy as np
import matplotlib.pyplot as plt
import pygame

class Joystick:
    def __init__(self, joystick_id=0):
        self.joystick = pygame.joystick.Joystick(joystick_id)
        self.joystick.init()
        self.num_axes = self.joystick.get_numaxes()

    def get_axis_values(self):
        pygame.event.pump()  # Update the event queue
        axis_values = [self.joystick.get_axis(i) for i in range(self.num_axes)]
        return axis_values

def suvat(s0, v0, a0, dt):
    s = s0 + v0 * dt + 0.5 * a0 * dt ** 2
    v = v0 + a0 * dt
    return [s, v]


class Drone:
    def __init__(self, number, state, targets):
        self.number = number
        [self.position, self.velocity, self.acceleration] = state
        self.targets = targets
        self.iTarget = 0
        self.close_counter = 0
        self.dt = 0.05
        self.dist = Drone.get_dist(self)
        self.max_thrust = 100
        self.radius = .215/2
        self.I = 0.266
        self.mass = 1
        self.g = -100
        self.alive = True
        self.thrust = [[],[]]

    def get_dist(self):
        return ((self.targets[self.iTarget][0] - self.position[0])**2 + (self.targets[self.iTarget][1] - self.position[1])**2)**0.5
    
    def update_state(self): #Physics update
        target = self.targets[self.iTarget]
        targetX = target[0]
        targetY = target[1]

        # add noise
        #for i in range(len(net_inputs[0])):
        #    net_inputs[0][i] = net_inputs[0][i] + np.random.rand() * net_inputs[0][i] * 0.0001

        L_thrust = self.mass / 2 * (-self.g)
        R_thrust = self.mass / 2 * (-self.g)

        y_correction = targetY-self.position[1] - self.velocity[1]
        x_correction = targetX-self.position[0]
        angle_correction = 100*self.position[2]
        angVel_correction = 100*self.velocity[2]
        xVel_correction = 2*self.velocity[0]

        R_thrust += y_correction - x_correction + xVel_correction - angle_correction - angVel_correction
        L_thrust += y_correction + x_correction - xVel_correction + angle_correction + angVel_correction

        R_thrust = min(max(0,R_thrust), self.max_thrust)
        L_thrust = min(max(0,L_thrust), self.max_thrust)
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
        self.im_drone = pygame.transform.scale(pygame.image.load(r'images\drone.png'), (self.char_w, self.char_h))
        self.im_target = pygame.transform.scale(pygame.image.load(r'images\red_dot.png'), (200, 200))
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
                [x_new, y_new, th_new] = [drones[0].position[j] for j in range(3)]
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
    dist0 = dist[0][0]
    counter = 0
    while run:
        for i in range(len(drones)):
            if drones[i].alive:
                game.pause()

                dist[i].append(drones[i].get_dist())
                omega[i].append(abs(drones[i].velocity[2]))
                v[i].append((drones[i].velocity[1]**2 + drones[i].velocity[1]**2)**0.5)
                theta.append(abs(drones[i].position[2]))

                drones[i].update_state()

                if abs(drones[i].position[2]) > game.break_angle*pi/180:
                    drones[i].kill()
                elif abs(drones[i].position[0]) > game.w/2:
                    drones[i].kill()
                elif abs(drones[i].position[1]) > game.h/2:
                    drones[i].kill()

                if dist[i][-1] < 8:
                    drones[i].close_counter += 1
                    if drones[i].close_counter > 400:
                        drones[i].iTarget += 1
                        dist[i] = []
                        counter = 0
                else:
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