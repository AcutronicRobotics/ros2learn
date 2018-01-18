import gym
import gym_gazebo
import time
env = gym.make('GazeboModularScaraArm4And3DOF-v1')
env.reset()

for i in range(10):
    env.randomizeCorrect()
    print("Reset!")
    for _ in range(200):
        env.render()
        print("action space sample: ", env.action_space.sample())
        observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
        print("reward: ", reward)
