import gym
import gym_gazebo
env = gym.make('HansArticulated-v1')
env.reset()

# Check the env limits:
# print(env.action_space) # Box(3,)
# print(env.observation_space) # Box(9,)

for i in range(10):
    env.reset()
    print("Reset!")
    for _ in range(200):
        observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
        print("reward: ", reward)
