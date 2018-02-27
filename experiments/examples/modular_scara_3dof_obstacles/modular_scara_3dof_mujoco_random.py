import roboschool, gym
from OpenGL import GLU

env = gym.make('RoboschoolScaraMove-v1')
# env.render("human")
env.reset()
env.render()

# Check the env limits:
# print(env.action_space) # Box(3,)
# print(env.observation_space) # Box(9,)

for i in range(100):
    env.reset()
    print("Reset!")
    for _ in range(50):
        observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
        print("reward: ", reward)
