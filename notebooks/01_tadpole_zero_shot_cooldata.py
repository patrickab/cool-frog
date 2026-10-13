# %% [markdown]
# # Tadpole zero-shot on CoolData
#
# Encode CoolData fields with the frozen, pretrained Tadpole-B autoencoder, decode them, and measure the error.

# %%
from pathlib import Path

from cooldata.pyvista_flow_field_dataset import PyvistaFlowFieldDataset
from cooldata.voxel_flow_field_dataset import VoxelFlowFieldDataset, VoxelFlowFieldDatasetConfig
from einops import rearrange
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from tadpole.model.autoencoder import TadpoleAutoencoder
import torch
import torch.nn.functional as F

DATA_DIR = Path("../data/cooldata")
N_SAMPLES = 1
# domain 0.5 x 0.1 x 0.02 m
# -> aspect ratio 25:5:1
# -> multiply by 16 to allow Tadpole to downsample by 16 and still have at least 1 voxel
FACTOR_ACCURACY = 1
FACTOR_DOWNSAMPLING = 16
FACTOR = FACTOR_ACCURACY * FACTOR_DOWNSAMPLING
RESOLUTION = (25*FACTOR, 5*FACTOR, 1*FACTOR)
CHANNELS = ["Vx [m/s]", "Vy [m/s]", "Vz [m/s]", "p [Pa]", "T [K]"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# %% [markdown]
# ## 1. Data: CFD mesh -> voxel grid

# %%
dataset_pyvista = PyvistaFlowFieldDataset.load_from_huggingface(data_dir=DATA_DIR, num_samples=N_SAMPLES)
dataset_voxel = VoxelFlowFieldDataset(
    cache_dir=DATA_DIR / f"voxels_{RESOLUTION[0]}x{RESOLUTION[1]}x{RESOLUTION[2]}",
    config=VoxelFlowFieldDatasetConfig(pyvista_dataset=dataset_pyvista, resolution=RESOLUTION),
)

y = torch.stack([s.Y for s in dataset_voxel.samples])  # (B, X, Y, Z, C) in physical units
mask = torch.stack([s.mask for s in dataset_voxel.samples])  # (B, X, Y, Z), True = fluid
x_phys = rearrange(y, "b x y z c -> b c x y z")  # Tadpole layout (B, C, X, Y, Z)
m = mask[:, None]  # (B, 1, X, Y, Z), broadcasts over channels
print("x_phys:", tuple(x_phys.shape), "= (samples, channels, X, Y, Z)")

# %% [markdown]
# ## 2. Preprocess
#
# Tadpole was pretrained on values in [-1, 1]. Per sample and channel:
#
# 1. Min-max scale the fluid values to [-1, 1].
# 2. Fill solid voxels with neighbouring fluid values. A constant would create fake jumps at every wall.

# %%
maxfloat = torch.finfo(torch.float32).max
lo = torch.where(m, x_phys, maxfloat).amin(dim=(2, 3, 4), keepdim=True)
hi = torch.where(m, x_phys, -maxfloat).amax(dim=(2, 3, 4), keepdim=True)


def to_unit(tensor_cooldata: torch.Tensor) -> torch.Tensor:
    """Scale physical values to [-1, 1] per sample and channel."""
    return 2 * (tensor_cooldata - lo) / (hi - lo) - 1


def to_phys(tensor_cooldata_normalized: torch.Tensor) -> torch.Tensor:
    """Map [-1, 1] values back to physical units."""
    return (tensor_cooldata_normalized + 1) * (hi - lo) / 2 + lo


def fill_solids(t: torch.Tensor) -> torch.Tensor:
    """Grow fluid values into solid voxels by repeated 3x3x3 neighbour averaging."""
    known = m.expand_as(t).float()
    v = t * known
    while (known == 0).any():
        num, den = F.avg_pool3d(v, 3, 1, 1), F.avg_pool3d(known, 3, 1, 1)
        new = (known == 0) & (den > 0)
        v = torch.where(new, num / den.clamp_min(1e-6), v)
        known = torch.where(new, 1.0, known)
    return v


x_unit = to_unit(x_phys)  # ground truth in model units
x = fill_solids(x_unit)  # model input
print("model input:", tuple(x.shape), f"range [{x.min():.2f}, {x.max():.2f}]")

# %% [markdown]
# ## 3. Load pretrained Tadpole-B

# %%
HF_REPO = "thuerey-group/Tadpole"
autoencoder = (
    TadpoleAutoencoder(
        size="B",
        weight_encoder=hf_hub_download(HF_REPO, "tadpole_b_encoder.safetensors"),
        weight_decoder=hf_hub_download(HF_REPO, "tadpole_b_decoder.safetensors"),

        # deterministic, no sampling
        latent_type="sample",

        # zero-shot: weights never change
        encoder_ft_state="frozen",
        decoder_ft_state="frozen",
        
        encoder_crop_size=None,
    )
    .to(DEVICE)
    .eval()
)

# %% [markdown]
# ## 4. Encode -> latent -> decode
#
# Tadpole treats every channel as its own single-channel 3D field, so samples and channels are stacked into one batch.

# %%
fields = rearrange(x, "B C X Y Z -> (B C) 1 X Y Z").to(DEVICE)
with torch.no_grad():
    latent = autoencoder.encoder(fields, "mode")  # "mode" = mean of the latent Gaussian, no sampling noise
    decoded = autoencoder.decoder(latent)
rec = rearrange(decoded.cpu(), "(B C) 1 X Y Z -> B C X Y Z", B=N_SAMPLES)

print("fields ", tuple(fields.shape), "= (samples x channels, 1, X, Y, Z)")
print("latent ", tuple(latent.shape), "= (samples x channels, features, X/16, Y/16, Z/16)")
print("decoded", tuple(decoded.shape))

# %% [markdown]
# ## 5. Error
#
# NRMSE per sample and channel over fluid voxels: RMSE / std(ground truth).
# 0 is perfect, 1 is as bad as predicting the mean.


# %%
def nrmse(pred: torch.Tensor, ref: torch.Tensor, fluid: torch.Tensor) -> float:
    return ((pred - ref)[fluid].pow(2).mean().sqrt() / ref[fluid].std()).item()


err = pd.DataFrame(
    [[100 * nrmse(rec[b, c], x_unit[b, c], mask[b]) for c in range(len(CHANNELS))] for b in range(N_SAMPLES)],
    columns=CHANNELS,
    index=[f"sample {b}" for b in range(N_SAMPLES)],
)
err.loc["mean"] = err.mean()
print("NRMSE [%]")
print(err.round(1).to_string())

# %% [markdown]
# ## 6. Ground truth vs reconstruction
#
# Top view at half channel height, solids in gray. Flow enters left, exits right.

# %%
EXTENT = [0, 500, 0, 100]  # axis labels in mm: [x_left, x_right, y_bottom, y_top]
Z_MID = RESOLUTION[2] // 2  # horizontal slice at half channel height


def show(  # noqa: PLR0913
    ax: plt.Axes,
    vol: np.ndarray,
    msk: np.ndarray,
    title: str,
    *,
    cmap: str = "plasma",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Plot the mid-height slice of a (X, Y, Z) volume, solids in gray."""
    img = np.where(msk[:, :, Z_MID], vol[:, :, Z_MID], np.nan).T
    colors = plt.get_cmap(cmap).copy()
    colors.set_bad("0.6")
    im = ax.imshow(img, origin="lower", extent=EXTENT, cmap=colors, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)


b = 0
rec_phys = to_phys(rec)
fig, axes = plt.subplots(len(CHANNELS), 3, figsize=(15, 1.3 * len(CHANNELS)), layout="constrained")
for c, name in enumerate(CHANNELS):
    gt, pr, msk = x_phys[b, c].numpy(), rec_phys[b, c].numpy(), mask[b].numpy()
    vmin, vmax = gt[msk].min(), gt[msk].max()
    show(axes[c, 0], gt, msk, f"ground truth | {name}", vmin=vmin, vmax=vmax)
    show(axes[c, 1], pr, msk, f"reconstruction | {name}", vmin=vmin, vmax=vmax)
    show(axes[c, 2], np.abs(pr - gt), msk, f"|error| | {name}", cmap="magma")
fig.suptitle(f"sample {b}")
plt.show()