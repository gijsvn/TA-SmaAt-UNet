import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

matplotlib.use("Agg")

# Set consistent font sizes for better readability
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 19,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 17,
    "legend.title_fontsize": 17,
    "figure.titlesize": 22,
})

SCALING = 47.83  # Denormalization factor to convert precipitation back to mm/5 min

def plot_losses(
        epochs: list[int],
        train_losses: list[float],
        val_losses: list[float] | None=None,
        save_path: str | None=None,
        y_scale: str = "linear",
    ) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, marker='s', label="Train")
    if val_losses is not None:
        plt.plot(epochs, val_losses, marker='s', label="Validation")
    plt.yscale(y_scale)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Monitoring")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path) if save_path is not None else plt.show()
    plt.close()

def visualize_precipitation_maps(
        precipitation_maps: np.ndarray,
        row_labels: list[str]|None=None,
        column_labels: list[str]|None=['Ground Truth', 'Prediction', 'Persistence'],
        suptitle: str | None=None,
        save_path: str | None=None,
        dpi: int=200
    ) -> None:

    precipitation_maps = np.asarray(precipitation_maps)

    # Set vmin/vmax in mm/h
    vmax = 20
    vmin = 0

    nrows = precipitation_maps.shape[1]
    ncols = precipitation_maps.shape[0]

    # Scale figure size with number of columns/rows
    fig, axesgrid = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4*ncols, 2.25*nrows),
        subplot_kw={"projection": ccrs.PlateCarree()},
        squeeze=False,  # Keeps axesgrid 2D even if nrows/ncols is 1
    )

    if row_labels != None:
        for ax, row in zip(axesgrid[:, 0], row_labels):
            ax.annotate(
                row, 
                xy=(-0.1, 0.5), 
                xycoords='axes fraction',
                ha='right', 
                va='center',
                rotation=90
            )

    if column_labels != None:
        for ax, col in zip(axesgrid[0], column_labels):
            ax.set_title(col, pad=10)

    extent = [3.38, 7.84, 50.82, 53.48] # lat/lon bounding box for current dataset
    for i in range(nrows):
        for j in range(ncols):
            ax = axesgrid[i, j]

            # Create simple land/ocean background for better visualization
            ax.add_feature(cfeature.LAND, facecolor="#8ad86e")
            ax.add_feature(cfeature.OCEAN, facecolor="#7098d5")
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            ax.set_extent(extent, crs=ccrs.PlateCarree())

            # Scale normalized precipitation maps to mm/h for visualization
            arr = precipitation_maps[j, i, :, :] * SCALING * 12

            im = ax.imshow(
                arr,
                extent=extent,
                vmin=vmin, 
                vmax=vmax,
                alpha=.75,
                transform=ccrs.PlateCarree()
            )

            ax.set_axis_off()

    if suptitle is not None:
        fig.suptitle(suptitle)

    fig.tight_layout(rect=[0, 0, 1, 0.95] if suptitle else None)
    fig.subplots_adjust(left=0.05, hspace=0.02, wspace=0.02)
    if im is not None:
        cbar = fig.colorbar(im, ax=fig.axes, shrink=1.0)
        cbar.ax.set_ylabel("mm/h")
    
    plt.savefig(save_path, dpi=dpi) if save_path is not None else plt.show()
    plt.close()