import gymnasium as gym
import numpy as np
import pygame
import random

class FlappyBirdPygameEnv(gym.Env):
    def __init__(self, width=288, height=512):
        super(FlappyBirdPygameEnv, self).__init__()
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Flappy Bird RL")

        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 200, 0)
        self.RED = (200, 0, 0)
        

        self.bird_x = 50
        self.bird_y = 250
        self.bird_velocity = 0
        self.gravity = 1
        self.flap_strength = -8
        
        self.pipe_x = self.width
        self.pipe_gap_y = random.randint(150, 350)
        self.pipe_gap_size = 100
        self.pipe_width = 50
        self.pipe_speed = 4

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=np.array([0, -10, 0, -300], dtype=np.float32),
            high=np.array([self.height, 10, self.width, 300], dtype=np.float32),
            dtype=np.float32
        )
        
        self.clock = pygame.time.Clock()
        self.done = False
    
    def reset(self):
        self.bird_y = 250
        self.bird_velocity = 0
        self.pipe_x = self.width
        self.pipe_gap_y = random.randint(150, 350)
        self.done = False
        return self._get_state()
    
    def step(self, action):
        # Apply action
        if action == 1:
            self.bird_velocity = self.flap_strength
        else:
            self.bird_velocity += self.gravity
        
        # Update positions
        self.bird_y += self.bird_velocity
        self.pipe_x -= self.pipe_speed
        
        # Pipe reset
        if self.pipe_x < -self.pipe_width:
            self.pipe_x = self.width
            self.pipe_gap_y = random.randint(150, 350)
        
        # Check collision (Top, Bottom, Pipes)
        reward = 1
        if self.bird_y < 0 or self.bird_y > self.height:
            reward = -100
            self.done = True
        elif self.pipe_x < self.bird_x < self.pipe_x + self.pipe_width:
            if not (self.pipe_gap_y - self.pipe_gap_size//2 < self.bird_y < self.pipe_gap_y + self.pipe_gap_size//2):
                reward = -100
                self.done = True
        
        return self._get_state(), reward, self.done, {}
    
    def _get_state(self):
        dx_to_pipe = self.pipe_x - self.bird_x
        dy_to_gap = self.pipe_gap_y - self.bird_y
        return np.array([self.bird_y, self.bird_velocity, dx_to_pipe, dy_to_gap], dtype=np.float32)
    
    def render(self, mode="human"):
        self.screen.fill(self.WHITE)
        
        # Draw pipe (Two rectangles)
        pygame.draw.rect(self.screen, self.GREEN, (self.pipe_x, 0, self.pipe_width, self.pipe_gap_y - self.pipe_gap_size//2))
        pygame.draw.rect(self.screen, self.GREEN, (self.pipe_x, self.pipe_gap_y + self.pipe_gap_size//2, self.pipe_width, self.height))
        
        # Draw the bird (Circle)
        pygame.draw.circle(self.screen, self.RED, (self.bird_x, int(self.bird_y)), 10)
        
        pygame.display.flip()
        self.clock.tick(30)
    
    def close(self):
        pygame.quit()