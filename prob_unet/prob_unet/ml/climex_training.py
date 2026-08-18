from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data as td
from torchmetrics.image import StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure

from prob_unet.datasets import climex_for_prob_unet
from prob_unet.ml.probabilistic_unet import ProbabilisticUnet
from prob_unet.plots.ml_plot_utils import output_training_figures


def climex_get_data_loader(config, split_name="train", batch_size=None):
    dataset = climex_for_prob_unet.ClimexDataset(
        path_daily_data=config.path_daily_output,
        path_daily_coarse_data=config.path_daily_coarse_output,
        path_neighbors=config.path_neighbors,
        train_years=[1961],
        validation_years=[],
        test_years=[],
        num_neighbors=2,
        active_split_name=split_name,
        debug_max_sample=32)
    if batch_size is None:
        batch_size = config.training_batch_size
    if config.device == "cuda":
        data_loader = td.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    else:
        data_loader = td.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader


def climex_training(config):
    prob_unet = ProbabilisticUnet(
        in_channels=1, out_channels=1, num_latent_dimensions=config.num_latent_dimensions,
        depth=config.unet_depth, depth_of_latent_injection=config.depth_of_latent_injection,
        initial_nb_of_hidden_channels=config.initial_nb_of_hidden_channels,
        kernel_size=config.unet_kernel_size, resolution_increase_layers=2)
    prob_unet.to(config.device)
    if not config.restart_training:
        pth_files = sorted(list(Path(config.path_model_output).glob(f"{config.experiment_name}_prob_unet_*.pth")))
        if pth_files:
            if int(pth_files[-1].stem.split("_")[-1]) == config.num_epochs:
                prob_unet.load_state_dict(torch.load(pth_files[-1]))
                return prob_unet
    mse_loss = nn.MSELoss()
    ssim_loss = MultiScaleStructuralSimilarityIndexMeasure(data_range=(-1.0, 1.0), kernel_size=7).to(config.device)
    optimizer = optim.Adam(prob_unet.parameters(), lr=config.learning_rate)
    prob_unet.train()

    data_loader_train = climex_get_data_loader(config=config, split_name="train")

    losses = []
    kl_losses = []
    reconstruction_losses = []
    for epoch in range(config.num_epochs):
        for item in data_loader_train:
            optimizer.zero_grad()
            input_data = item["input_first_block"].unsqueeze(1).to(config.device)  # Add channel dimension
            target_data = item["target"].unsqueeze(1).to(config.device)  # Add channel dimension
            output = prob_unet(input_data, target_data=target_data, mode="train")  # Add channel dimension
            output_training_figures(path_output=config.path_model_output, count=len(losses),
                                    input_data=input_data, target_data=target_data, output_data=output)
            loss_terms = {}
            weights = []
            if config.mse_weight > 0:
                loss_terms["mse_loss"] = mse_loss(output, target_data)
                weights.append(config.mse_weight)
            if config.ssim_weight > 0:
                loss_terms["ssim_loss"] = 1 - ssim_loss(output, target_data)
                weights.append(config.ssim_weight)
            loss_terms["kl_loss"] = prob_unet.kl_divergence()
            weights.append(config.kl_weight)
            loss = sum(loss_terms[term] * weight for term, weight in zip(loss_terms.keys(), weights))
            loss.backward()
            optimizer.step()
            kl_losses.append(loss_terms["kl_loss"].item())
            losses.append(loss.item())
        output_path = Path(config.path_model_output, f"{config.experiment_name}_prob_unet_epoch_{epoch+1:03d}.pth")
        torch.save(prob_unet.state_dict(), output_path)
    return prob_unet
    