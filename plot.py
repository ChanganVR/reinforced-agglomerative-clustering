import matplotlib.pyplot as plt

all_idx_episode = []
all_train_purity = []
all_val_purity = []
all_test_purity = []
with open('results/S_10_step_5.log','r') as f:
    for i_line, line in enumerate(f):
        if (i_line < 10) or ((i_line-10)%4 != 0):
            continue

        line_split = line.strip().split()

        all_idx_episode.append(int(line_split[4]))
        all_train_purity.append(float(line_split[7][:-1]))
        all_val_purity.append(float(line_split[10][:-1]))
        all_test_purity.append(float(line_split[13][:-1]))


plt.plot(all_idx_episode, all_train_purity, color='red')
plt.plot(all_idx_episode, all_val_purity, color='blue')
plt.plot(all_idx_episode, all_test_purity, color='green')

plt.draw()
plt.show()