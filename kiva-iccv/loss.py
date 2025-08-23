import torch
import torch.nn as nn


class ContrastiveAnalogyLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        t_example: torch.Tensor,
        t_choice_a: torch.Tensor,
        t_choice_b: torch.Tensor,
        t_choice_c: torch.Tensor,
        correct_idx: torch.Tensor,
    ) -> torch.Tensor:
        # Compute similarities between example transformation and each choice transformation
        sim_a = torch.cosine_similarity(t_example, t_choice_a, dim=1)
        sim_b = torch.cosine_similarity(t_example, t_choice_b, dim=1)
        sim_c = torch.cosine_similarity(t_example, t_choice_c, dim=1)

        # Get the positive (correct) similarity score
        batch_size = correct_idx.shape[0]
        positive_sim = torch.gather(
            torch.stack([sim_a, sim_b, sim_c], dim=1), 1, correct_idx.unsqueeze(1)
        ).squeeze(1)

        # Get the two negative similarity scores
        negative_sims = []
        for i in range(batch_size):
            neg_indices = [j for j in range(3) if j != correct_idx[i]]
            neg_sim_1 = torch.stack([sim_a[i], sim_b[i], sim_c[i]])[neg_indices[0]]
            neg_sim_2 = torch.stack([sim_a[i], sim_b[i], sim_c[i]])[neg_indices[1]]
            negative_sims.append([neg_sim_1, neg_sim_2])

        negative_sims = torch.stack(negative_sims)  # Shape: (batch_size, 2)
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
