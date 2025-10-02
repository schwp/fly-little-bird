import gymnasium as gym
import numpy as np

class FlappyBirdLogicEnv(gym.Env):
    def __init__(self):
        super(FlappyBirdLogicEnv, self).__init__()

        self.action_space = gym.spaces.Discrete(2) # Flap (1) or do nothing (0)

        # bird_Y, bird_velocity, pipe_X, pipe_gap_Y
        self.observation_space = gym.spaces.Box(
            low=np.array([0, -10, 0, -300], dtype=np.float32),
            high=np.array([512, 10, 288, 300], dtype=np.float32)
        )

        self.reset()


    def reset(self):
        self.bird_Y = 250
        self.bird_velocity = 0
        self.pipe_X = 288
        self.pipe_gap_Y = np.random.randint(100, 400)
        self.done = False
        return self._get_state()
    
    def step(self, action):
        if action == 1:
            self.bird_velocity = -8
        else:
            self.bird_velocity += 1

        self.bird_Y += self.bird_velocity
        self.pipe_X -= 4

        if self.pipe_X < -50:
            self.pipe_X = 288
            self.pipe_gap_Y = np.random.randint(100, 400)

        reward = 1
        if self.bird_Y < 0 or self.bird_Y > 512:
            self.done = True
            reward = -100
        elif self.pipe_X < 50 and \
            not (self.pipe_gap_Y - 50 < self.bird_Y < self.pipe_gap_Y + 50):
            self.done = True
            reward = -100

        return self._get_state(), reward, self.done, {}
    
    def _get_state(self):
        return np.array([
                self.bird_Y,
                self.bird_velocity,
                self.pipe_X,
                self.pipe_gap_Y - self.bird_Y
            ], dtype=np.float32)
    
    def render(self):
        print(f"Bird Y={self.bird_Y}, Pipe X={self.pipe_X}, Gap Y={self.pipe_gap_Y}")
