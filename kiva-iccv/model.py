import torch
import torch.nn as nn
from torchvision import models


class SiameseAnalogyNetwork(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        freeze_encoder: bool = False,
        transformation_net: bool = False,
    ):
        super().__init__()
        resnet = models.resnet18(weights="IMAGENET1K_V1")
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.projection = nn.Linear(resnet.fc.in_features, embedding_dim)

        self.transformation_net = (
            nn.Sequential(
                nn.Linear(
                    embedding_dim * 2, embedding_dim
                ),  # Input is a concatenation of two embeddings
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim),
            )
            if transformation_net
            else None
        )

    def forward(
        self,
        ex_before: torch.Tensor,
        ex_after: torch.Tensor,
        test_before: torch.Tensor,
        choice_a: torch.Tensor,
        choice_b: torch.Tensor,
        choice_c: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        # Determine the batch size from one of the tensors
        batch_size = ex_before.size(0)

        # Reshape all tensors into a single batch, preserving the C, H, W dimensions
        # and stacking along a new dimension (dim=0)
        all_tensors = torch.cat(
            [ex_before, ex_after, test_before, choice_a, choice_b, choice_c], dim=0
        )

        # Process the combined batch
        all_embeddings = self.projection(self.encoder(all_tensors).flatten(1))

        # Split the embeddings back based on the batch size
        t_ex_before = all_embeddings[0 * batch_size : 1 * batch_size]
        t_ex_after = all_embeddings[1 * batch_size : 2 * batch_size]
        t_test_before = all_embeddings[2 * batch_size : 3 * batch_size]
        t_choice_a = all_embeddings[3 * batch_size : 4 * batch_size]
        t_choice_b = all_embeddings[4 * batch_size : 5 * batch_size]
        t_choice_c = all_embeddings[5 * batch_size : 6 * batch_size]

        # Calculate the transformation vectors
        if self.transformation_net:
            t_example = self.transformation_net(torch.cat([t_ex_before, t_ex_after], dim=1))
            t_choice_a_vec = self.transformation_net(torch.cat([t_test_before, t_choice_a], dim=1))
            t_choice_b_vec = self.transformation_net(torch.cat([t_test_before, t_choice_b], dim=1))
            t_choice_c_vec = self.transformation_net(torch.cat([t_test_before, t_choice_c], dim=1))
        else:
            t_example = t_ex_after - t_ex_before
            t_choice_a_vec = t_choice_a - t_test_before
            t_choice_b_vec = t_choice_b - t_test_before
            t_choice_c_vec = t_choice_c - t_test_before

        return t_example, t_choice_a_vec, t_choice_b_vec, t_choice_c_vec
