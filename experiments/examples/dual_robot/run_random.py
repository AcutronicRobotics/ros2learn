import gym
import gym_gazebo
import time
env = gym.make('GazeboModularScaraArm4And3DOF-v1')
env.reset()

# Check the env limits:
# print(env.action_space) # Box(3,)
# print(env.observation_space) # Box(9,)

for i in range(10):
    env.randomizeCorrect()
    time.sleep(2)
    # env.reset()

    print("Reset!")
    for _ in range(200):
        env.render()
        print("action space sample: ", env.action_space.sample())
        observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
        print("reward: ", reward)
