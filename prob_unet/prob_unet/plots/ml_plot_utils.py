from pathlib import Path

import matplotlib.pyplot as plt


def output_training_figures(path_output, count, input_data, target_data, output_data):
    if count in [0, 1, 2, 5, 10, 50, 100, 1000]:
        n_rows = input_data.shape[0]
        n_cols = 3
        fig = plt.figure(figsize=(12, 4 * n_rows))
        for i in range(n_rows):
            ax1 = fig.add_subplot(n_rows, n_cols, i * n_cols + 1)
            ax1.set_title("Input")
            ax1.pcolormesh(input_data[i, 0, :, :].cpu().detach().numpy(), vmin=-1, vmax=1)
            ax2 = fig.add_subplot(n_rows, n_cols, i * n_cols + 2)
            ax2.set_title("Target")
            ax2.pcolormesh(target_data[i, 0, :, :].cpu().detach().numpy(), vmin=-1, vmax=1)
            ax3 = fig.add_subplot(n_rows, n_cols, i * n_cols + 3)
            ax3.set_title("Output")
            ax3.pcolormesh(output_data[i, 0, :, :].cpu().detach().numpy(), vmin=-1, vmax=1)
        plt.tight_layout()
        plt.savefig(Path(path_output, f"training_visualization_{count:06d}.png"))
        plt.close(fig)
