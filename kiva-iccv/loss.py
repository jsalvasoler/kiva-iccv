import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveAnalogyLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        model_outputs: tuple,
        correct_indices: torch.Tensor,
    ) -> torch.Tensor:
        # Unpack the vectors produced by the network
        t_example, t_choice_a, t_choice_b, t_choice_c = model_outputs

        # Compute similarities between example transformation and each choice transformation
        sim_a = torch.cosine_similarity(t_example, t_choice_a, dim=1)
        sim_b = torch.cosine_similarity(t_example, t_choice_b, dim=1)
        sim_c = torch.cosine_similarity(t_example, t_choice_c, dim=1)

        # Get the positive (correct) similarity score
        batch_size = correct_indices.shape[0]
        positive_sim = torch.gather(
            torch.stack([sim_a, sim_b, sim_c], dim=1), 1, correct_indices.unsqueeze(1)
        ).squeeze(1)

        # Get the two negative similarity scores more efficiently
        # Create a mask for incorrect choices
        batch_indices = torch.arange(batch_size, device=correct_indices.device)
        mask = torch.ones(batch_size, 3, dtype=torch.bool, device=correct_indices.device)
        mask[batch_indices, correct_indices] = False

        # Get the two negative similarities using the mask
        all_sims = torch.stack([sim_a, sim_b, sim_c], dim=1)  # (batch_size, 3)
        negative_sims = all_sims[mask].view(batch_size, 2)  # (batch_size, 2)
        sim_negative1 = negative_sims[:, 0]
        sim_negative2 = negative_sims[:, 1]

        # Compute contrastive loss
        loss1 = torch.clamp(self.margin - (positive_sim - sim_negative1), min=0)
        loss2 = torch.clamp(self.margin - (positive_sim - sim_negative2), min=0)
        return torch.mean(loss1 + loss2)


class StandardTripletAnalogyLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)

    def forward(
        self,
        model_outputs: tuple,
        correct_indices: torch.Tensor,
    ) -> torch.Tensor:
        # Unpack the vectors produced by the network
        t_example, t_choice_a, t_choice_b, t_choice_c = model_outputs
        choices = torch.stack([t_choice_a, t_choice_b, t_choice_c], dim=1)

        batch_size = t_example.shape[0]
        total_loss = 0.0

        for i in range(batch_size):
            # The example transformation is the "anchor"
            anchor = t_example[i]

            # The correct choice's transformation is the "positive"
            correct_idx = correct_indices[i]
            positive = choices[i, correct_idx]

            # The two incorrect choices are the "negatives"
            for neg_idx in range(3):
                if neg_idx != correct_idx:
                    negative = choices[i, neg_idx]
                    total_loss += self.triplet_loss(anchor, positive, negative)

        # We calculate two loss terms per sample in the batch
        return total_loss / (batch_size * 2)


class SoftmaxAnalogyLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        """
        temperature < 1.0 sharpens distributions, >1.0 smooths them.
        Defaults to 0.07, like in InfoNCE/SimCLR.
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        model_outputs: tuple,
        correct_indices: torch.Tensor,
    ) -> torch.Tensor:
        # Unpack the vectors produced by the network
        t_example, t_choice_a, t_choice_b, t_choice_c = model_outputs

        # Compute cosine similarities for each choice
        sim_a = torch.cosine_similarity(t_example, t_choice_a, dim=1)
        sim_b = torch.cosine_similarity(t_example, t_choice_b, dim=1)
        sim_c = torch.cosine_similarity(t_example, t_choice_c, dim=1)

        # Stack into (batch_size, 3)
        sims = torch.stack([sim_a, sim_b, sim_c], dim=1)

        # Scale by temperature (like in contrastive learning)
        sims = sims / self.temperature

        # Apply cross-entropy loss directly
        loss = F.cross_entropy(sims, correct_indices)

        return loss
