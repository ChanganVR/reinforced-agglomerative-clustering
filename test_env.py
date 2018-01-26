import math
from env import env


data_dir = 'dataset'
sampling_size = 10
clustering_env = env.Env(data_dir, sampling_size, reward='local_purity')
clustering_env.set_seed(0)
state, _, purity = clustering_env.reset()
assert purity == 1
# labels                            [ 9,   3,   8,   9,   4,   5,   4,   5,   3,   8 ]
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