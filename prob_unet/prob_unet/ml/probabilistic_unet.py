from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Standard dual convolutional block used in U-Net.
    (Conv2d -> ReLU -> Conv2d -> ReLU)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2  # Keeps spatial dimensions identical
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Encoder(nn.Module):
    """
    General convolutional encoder used for both Prior and Posterior networks.
    Outputs the mu and log-variance for the latent space.
    """
    def __init__(self, in_channels: int, hidden_channels: int, latent_dim: int):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # Pools spatial dimensions down to 1x1 
        )
        self.fc = nn.Linear(hidden_channels * 4, 2 * latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.convs(x)
        features = features.view(features.size(0), -1)
        return self.fc(features)


class ProbabilisticUnet(nn.Module):
    """
    Probabilistic U-Net for climate data downscaling.
    Supports learning two distinct latent distributions (prior and posterior).
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        num_latent_dimensions: int,
        depth_of_latent_injection: int,
        depth: int, 
        initial_nb_of_hidden_channels: int, 
        kernel_size: int, 
        resolution_increase_layers: int
    ):
        super().__init__()
        self.num_latent_dimensions = num_latent_dimensions
        self.depth_of_latent_injection = depth_of_latent_injection
        
        # Track distribution parameters for the KL Divergence loss component
        self.posterior_mu = None
        self.posterior_logvar = None
        self.prior_mu = None
        self.prior_logvar = None

        # ========================================================
        # 1. Prior and Posterior Networks
        # ========================================================
        self.prior = Encoder(in_channels, initial_nb_of_hidden_channels, num_latent_dimensions)
        
        # The posterior sees both the coarse input AND the high-res target.
        # We will dynamically downsample the target to match the coarse input in the forward pass.
        self.posterior = Encoder(in_channels + out_channels, initial_nb_of_hidden_channels, num_latent_dimensions)

        # ========================================================
        # 2. Base U-Net Structure
        # ========================================================
        self.unet_down = nn.ModuleList()
        self.unet_up = nn.ModuleList()
        
        c_in = in_channels
        c_out = initial_nb_of_hidden_channels
        self.down_channels = []
        
        # Downward Path
        for _ in range(depth):
            self.unet_down.append(DoubleConv(c_in, c_out, kernel_size))
            self.down_channels.append(c_out)
            c_in = c_out
            c_out *= 2
            
        # Bottom of the "U"
        self.bottom = DoubleConv(c_in, c_out, kernel_size)
        c_in = c_out
        
        # Upward Path (to original input resolution)
        for i in range(depth):
            skip_c = self.down_channels[-(i+1)]
            out_c = skip_c
            if depth + resolution_increase_layers - i - 1 == self.depth_of_latent_injection:
                latent_c = num_latent_dimensions
            else:
                latent_c = 0
            self.unet_up.append(nn.ModuleList([
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                DoubleConv(c_in + skip_c + latent_c, out_c, kernel_size)
            ]))
            c_in = out_c

        # ========================================================
        # 3. Super-Resolution Layers (Post U-Net)
        # ========================================================
        self.res_increase = nn.ModuleList()
        for i in range(resolution_increase_layers):
            if resolution_increase_layers - i - 1 == self.depth_of_latent_injection:
                c_in += num_latent_dimensions  # Account for latent vector concatenation at this layer
            self.res_increase.append(nn.ModuleList([
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                DoubleConv(c_in, c_in, kernel_size)
            ]))

        # ========================================================
        # 4. Final Combination block (U-Net Features + Latent Vector Z)
        # ========================================================
        if self.depth_of_latent_injection == 0:
            self.fcomb = nn.Sequential(
                nn.Conv2d(c_in + num_latent_dimensions, c_in, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_in, out_channels, kernel_size=1)
            )
        else:
            self.fcomb = nn.Sequential(
                nn.Conv2d(c_in, c_in, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_in, out_channels, kernel_size=1)
            )

    def _get_prior_params(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        params = self.prior(x)
        return torch.chunk(params, 2, dim=1)

    def _get_posterior_params(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Downsample y (target) to match x (input) spatial dimensions for the encoder
        y_down = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
        xy = torch.cat([x, y_down], dim=1)
        params = self.posterior(xy)
        return torch.chunk(params, 2, dim=1)

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_divergence(self) -> torch.Tensor:
        """
        Computes the KL divergence between posterior and prior from the last training pass.
        Call this alongside your reconstruction loss (e.g. MSE/L1) to compute total loss.
        """
        if self.posterior_mu is None or self.prior_mu is None:
            raise RuntimeError("Must execute forward in 'train' mode to compute KL divergence.")
            
        var_prior = torch.exp(self.prior_logvar)
        var_post = torch.exp(self.posterior_logvar)

        kl = 0.5 * torch.sum(
            self.prior_logvar - self.posterior_logvar
            + (var_post + (self.posterior_mu - self.prior_mu) ** 2) / var_prior - 1,
            dim=1 
        )
        return kl.mean()

    def forward(
        self, 
        input_data: torch.Tensor, 
        target_data: Optional[torch.Tensor] = None, 
        mode: str = "train", 
        latent_coord: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        # -----------------------------------------------
        # 1. Feature Extraction (U-Net & Res Layers)
        # -----------------------------------------------
        skip_connections = []
        x = input_data
        
        # Down
        for down_layer in self.unet_down:
            x = down_layer(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, 2)
            
        # Bottom
        x = self.bottom(x)

        # -----------------------------------------------
        # 2. Latent Distribution Sampling
        # -----------------------------------------------
        if mode == "train":
            if target_data is None:
                raise ValueError("target_data must be provided in 'train' mode.")
                
            # Compute Prior (For KL Divergence calculation later)
            self.prior_mu, self.prior_logvar = self._get_prior_params(input_data)
            
            # Compute Posterior & Sample Z
            self.posterior_mu, self.posterior_logvar = self._get_posterior_params(input_data, target_data)
            z = self._reparameterize(self.posterior_mu, self.posterior_logvar)
            
        elif mode == "inference":
            prior_mu, prior_logvar = self._get_prior_params(input_data)
            if latent_coord is not None:
                # Use manually specified latent coordinates
                std = torch.exp(0.5 * prior_logvar)
                z = prior_mu + latent_coord.to(input_data.device) * std
            else:
                # Sample from prior randomly
                z = self._reparameterize(prior_mu, prior_logvar)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # ToDo: allow introduction of latent space at different depths
        # Up
        for i, up_layer in enumerate(self.unet_up):
            upsample, conv = up_layer
            x = upsample(x)
            skip = skip_connections[-(i+1)]
            x = torch.cat([x, skip], dim=1)
            x = conv(x)

        # Resolution increase
        for res_layer in self.res_increase:
            upsample, conv = res_layer
            x = upsample(x)
            x = conv(x)

        unet_features = x

        # -----------------------------------------------
        # 3. Combining features with latent vector
        # -----------------------------------------------
        # Broadcast `z` to spatial shape of the final feature map
        z_broadcast = z.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, unet_features.size(2), unet_features.size(3))
        
        combined_features = torch.cat([unet_features, z_broadcast], dim=1)
        return self.fcomb(combined_features)
