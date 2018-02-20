import math
from env import env


data_dir = 'dataset'
sampling_size = 20
clustering_env = env.Env(data_dir, sampling_size, reward='local_purity')
state, _ = clustering_env.reset(seed=0, steps=15)
print(state.cluster_assignments)
# labels                            [ 8, 3, 6, 7, 8, 1, 3, 1, 7, 6 ]
assert state.cluster_assignments == [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]

action = env.Action(0, 3)
reward, state, purity = clustering_env.step(action)
assert purity == 1
assert state.cluster_assignments == [[0, 3], [1], [2], [4], [5], [6], [7], [8], [9]]
assert reward == 0

action = env.Action(0, 1)
reward, state, purity = clustering_env.step(action)
assert purity == 0.9
assert state.cluster_assignments == [[0, 1, 3], [2], [4], [5], [6], [7], [8], [9]]
assert math.isclose(reward, -1/3)

action = env.Action(1, 7)
reward, state, purity = clustering_env.step(action)
assert purity == 0.9
assert state.cluster_assignments == [[0, 1, 3], [2, 9], [4], [5], [6], [7], [8]]
assert math.isclose(reward, 0)