from scipy.misc import imshow
from env.env import Env, load_cifar

label_dict, classes = load_cifar('train', 'dataset')
imshow(label_dict[1][0])
print()