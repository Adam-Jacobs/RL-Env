from game_environments import FindPath
# import numpy as np


env = FindPath(start_position=[0, 0])
print(env.current_observation)
print(env.current_state_num)

print(env.step(1))
print(env.current_state_num)

print(env.step(3))
print(env.current_state_num)

print(env.step(2))
print(env.current_state_num)

print(env.step(0))
print(env.current_state_num)


