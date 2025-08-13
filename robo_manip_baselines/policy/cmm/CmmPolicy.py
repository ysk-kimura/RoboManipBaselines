import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.ops.misc import FrozenBatchNorm2d


class CmmPolicy(nn.Module):
    """MLP policy with ResNet backbone and cross-modal fusion layer."""

    def __init__(
        self,
        state_dim,
        action_dim,
        num_images,
        n_obs_steps,
        n_action_steps,
        hidden_dim_list,
        state_feature_dim,
        cross_modal_dim=1024,
    ):
        super().__init__()

        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps

        self.state_feature_extractor = nn.Sequential(
            nn.Linear(state_dim * n_obs_steps, state_feature_dim),
            nn.ReLU(),
        )

        resnet_model = resnet18(
            weights=ResNet18_Weights.DEFAULT, norm_layer=FrozenBatchNorm2d
        )
        self.image_feature_extractor = nn.Sequential(
            *list(resnet_model.children())[:-1]
        )
        self.image_feature_dim = resnet_model.fc.in_features

        combined_input_dim = (
            state_feature_dim + num_images * n_obs_steps * self.image_feature_dim
        )
        self.cross_modal_layer = nn.Sequential(
            nn.Linear(combined_input_dim, cross_modal_dim),
            nn.ReLU(),
            nn.Linear(cross_modal_dim, cross_modal_dim),
            nn.ReLU(),
        )

        linear_dim_list = (
            [cross_modal_dim] + hidden_dim_list + [action_dim * n_action_steps]
        )
        linear_layers = []
        for i in range(len(linear_dim_list) - 1):
            linear_layers.append(nn.Linear(linear_dim_list[i], linear_dim_list[i + 1]))
            if i < len(linear_dim_list) - 2:
                linear_layers.append(nn.ReLU())
        self.linear_layer_seq = nn.Sequential(*linear_layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, state_seq, images_seq):
        batch_size, _, _, C, H, W = images_seq.shape

        state_seq = state_seq.reshape(batch_size, -1)
        images_seq = images_seq.reshape(batch_size, -1, C, H, W)

        state_feature = self.state_feature_extractor(state_seq)

        image_features = []
        for i in range(images_seq.shape[1]):
            feat = self.image_feature_extractor(images_seq[:, i])
            feat = feat.view(batch_size, -1)
            image_features.append(feat)
        image_features = torch.cat(image_features, dim=1)

        combined_feature = torch.cat([state_feature, image_features], dim=1)
        fused_feature = self.cross_modal_layer(combined_feature)

        action_seq = self.linear_layer_seq(fused_feature)
        action_seq = action_seq.reshape(batch_size, self.n_action_steps, -1)

        return action_seq
