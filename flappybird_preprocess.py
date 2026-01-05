import cv2
import gymnasium as gym
import numpy as np

class FlappyBirdPreprocess(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = self.env.render()
        return self._process_frame(frame), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        frame = self.env.render()
        return self._process_frame(frame), info

    def _process_frame(self, frame):
        if frame is None:
            return np.zeros((84, 84), dtype=np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84))
        return resized

