import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

with open('ppo-test.pkl','rb') as f:
    out = pkl.load(f)


mean = out['returned_episode_means'].mean(-1).reshape(-1)

plt.plot(mean)
plt.show()
