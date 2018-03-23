import matplotlib.pyplot as plt


def ravg(seq, window):
    return [sum(seq[x-window+1:x+1])/window if x >= window-1 else sum(seq[:x+1])/(x+1) for x in range(len(seq))]


def plot_log(log_name, grid, window=1):

    all_idx_episode = []
    all_train_purity = []
    all_val_purity = []
    all_test_purity = []
    with open('results/{}/output.log'.format(log_name),'r') as f:
        for i_line, line in enumerate(f):
            if (i_line < 10) or ((i_line-10)%4 != 0):
                continue

            line_split = line.strip().split()

            all_idx_episode.append(int(line_split[4]))
            all_train_purity.append(float(line_split[7][:-1]))
            all_val_purity.append(float(line_split[10][:-1]))
            all_test_purity.append(float(line_split[13][:-1]))

    plt.plot(all_idx_episode, ravg(all_train_purity, window), color='red')
    plt.plot(all_idx_episode, ravg(all_val_purity, window), color='blue')
    plt.plot(all_idx_episode, ravg(all_test_purity, window), color='green')

shared_ax = None
# log_names = ['2018-03-22 23:49:27']
log_names = ['2018-03-23 04:47:35']
grids = [[1,1,1]]
for i in range(1):
    plot_log(log_names[i], grids[i], window=1)

plt.draw()
plt.show()