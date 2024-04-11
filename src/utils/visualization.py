import matplotlib.pyplot as plt
import numpy as np

def plot_state_value_function(q_values, title="State Value Function"):
    keys = list(q_values.keys())
    dxs = [key[0] for key in keys]
    dys = [key[1] for key in keys]

    n_x = len(set(dxs))
    n_y = len(set(dys))

    x_min = min(dxs)
    x_max = max(dxs)
    y_min = min(dys)
    y_max = max(dys)

    arr = np.zeros((n_x, n_y, 2))
    
    for key, value in q_values.items():
        arr[key[0] - x_min, key[1] - y_min] = value

    plt.figure(figsize=(20, 10))
    plt.suptitle(title)

    plt.subplot(1, 4, 1)
    plt.imshow(arr[:, :, 0], extent=(x_min, x_max, y_min, y_max))
    plt.colorbar()
    plt.xlabel("dx")
    plt.ylabel("dy")
    plt.title("Q-values when not flapping")
    
    plt.subplot(1, 4, 2)
    plt.imshow(arr[:, :, 1], extent=(x_min, x_max, y_min, y_max))
    plt.colorbar()
    plt.xlabel("dx")
    plt.ylabel("dy")
    plt.title("Q-values when flapping")

    plt.subplot(1, 4, 3)
    plt.imshow(np.max(arr, axis=2), extent=(x_min, x_max, y_min, y_max))
    plt.colorbar()
    plt.xlabel("dx")
    plt.ylabel("dy")
    plt.title("Optimal Value Function")

    plt.subplot(1, 4, 4)
    plt.imshow(np.argmax(arr, axis=2), extent=(x_min, x_max, y_min, y_max))
    plt.xlabel("dx")
    plt.ylabel("dy")
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.ax.set_yticklabels(["Not Flapping", "Flapping"])
    plt.title("Optimal Policy")
    
    plt.show()
