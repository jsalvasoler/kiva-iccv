import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformationEncoder(nn.Module):
    """
    An encoder that takes two images and processes them as a single sequence
    to explicitly model the transformation between them.
    """

    def __init__(self, encoder_name: str = "vit_small_patch16_224", pretrained: bool = True):
        super().__init__()
        self.vit = timm.create_model(encoder_name, pretrained=pretrained)

        # The ViT's forward_features method is what we need to deconstruct.
        # It typically handles patch embedding, adding CLS token, pos_embed, and transformer blocks.

        # We only need one CLS token for the combined sequence representation
        self.cls_token = self.vit.cls_token

        # The original positional embedding is for 1 CLS token + 196 patches.
        # We need a new pos_embed for 1 CLS token + 196 (img1) + 196 (img2) patches.
        # We can construct this by concatenating the original pos_embeds.
        # TODO: idea for improvement: Consider positional embedding interpolation i
        # nstead of naive duplication.
        original_pos_embed = self.vit.pos_embed
        self.pos_embed = nn.Parameter(
            torch.cat(
                [
                    original_pos_embed[:, :1, :],  # CLS token's pos_embed
                    original_pos_embed[:, 1:, :],  # Patches from img1
                    original_pos_embed[:, 1:, :],  # Patches from img2 (re-used)
                ],
                dim=1,
            )
        )

        # Segment embeddings to disambiguate patches from image A vs B
        self.segment_embed = nn.Embedding(3, self.vit.num_features)
        # 0 = CLS, 1 = image A, 2 = image B

        # We will re-use the patch embedding layer and the transformer blocks
        self.patch_embed = self.vit.patch_embed
        self.blocks = self.vit.blocks
        self.norm = self.vit.norm

        # The output dimension is the embedding dimension of the ViT
        self.num_features = self.vit.num_features

        # Precompute segment IDs pattern (constant for a given model)
        # We need to know the number of patches to create the pattern
        num_patches = self.patch_embed.num_patches
        seg_ids_pattern = torch.cat(
            [
                torch.zeros(1, dtype=torch.long),  # CLS
                torch.ones(num_patches, dtype=torch.long),  # img1
                2 * torch.ones(num_patches, dtype=torch.long),  # img2
            ]
        )  # (1 + 2*num_patches,)

        # Register as buffer so it moves with the model to GPU/CPU
        self.register_buffer("seg_ids_pattern", seg_ids_pattern)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        B = img1.size(0)

        # 1. Create patch embeddings for each image separately
        # (B, C, H, W) -> (B, num_patches, embed_dim)
        patches1 = self.patch_embed(img1)  # (B, N, D)
        patches2 = self.patch_embed(img2)  # (B, N, D)

        # 2. Concatenate the patch sequences
        # (B, 196, 384) + (B, 196, 384) -> (B, 392, 384)
        combined_patches = torch.cat([patches1, patches2], dim=1)  # (B, 2N, D)

        # 3. Prepend the CLS token
        # (B, 1, 384) + (B, 392, 384) -> (B, 393, 384)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, combined_patches], dim=1)  # (B, 1+2N, D)

        # 4. Add the positional embeddings
        x = x + self.pos_embed

        # 5. Add segment embeddings to disambiguate patches from image A vs B
        # Expand the precomputed pattern for the batch
        seg_ids = self.seg_ids_pattern.unsqueeze(0).expand(B, -1)  # (B, 1+2N)

        # Lookup and add segment embeddings
        x = x + self.segment_embed(seg_ids)

        # 6. Pass through the transformer blocks
        x = self.blocks(x)
        x = self.norm(x)

        # 7. Return the embedding of the [CLS] token, which now represents the transformation
        return x[:, 0]


class TransformationEncoderDINOv3(nn.Module):
    """
    A specialized encoder for DINOv3 models that takes two images and processes them
    as a single sequence to explicitly model the transformation between them.

    DINOv3 models use rotary position embeddings (RoPE) and register tokens,
    which requires special handling compared to standard ViT models.
    """

    def __init__(self, encoder_name: str = "vit_small_patch16_dinov3", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(encoder_name, pretrained=pretrained)

        # Store important attributes
        self.num_features = self.backbone.num_features
        self.patch_embed = self.backbone.patch_embed
        self.cls_token = self.backbone.cls_token
        self.blocks = self.backbone.blocks
        self.norm = self.backbone.norm
        self.rope = self.backbone.rope  # Rotary position embeddings

        # DINOv3 models typically have register tokens (extra tokens beyond CLS + patches)
        # From our analysis: 201 tokens = 1 CLS + 196 patches + 4 register tokens
        num_patches = self.patch_embed.num_patches  # 196 for 224x224 with patch_size=16

        # Segment embeddings to disambiguate patches from image A vs B
        # 0 = CLS, 1 = register tokens, 2 = image A patches, 3 = image B patches
        self.segment_embed = nn.Embedding(4, self.num_features)

        # Precompute segment IDs pattern for the combined sequence
        # We need to handle: CLS + register_tokens + patches_img1 + patches_img2
        # The exact number of register tokens depends on the model, but we can infer it
        # by running a forward pass and checking the output size

        # DINOv3 has 4 register tokens (https://github.com/facebookresearch/dinov3/issues/124)
        self.num_register_tokens = 4

        # Create segment ID pattern: [CLS, registers, img1_patches, img2_patches]
        seg_ids_pattern = torch.cat(
            [
                torch.zeros(1, dtype=torch.long),  # CLS token
                torch.ones(self.num_register_tokens, dtype=torch.long),  # Register tokens
                2 * torch.ones(num_patches, dtype=torch.long),  # Image 1 patches
                3 * torch.ones(num_patches, dtype=torch.long),  # Image 2 patches
            ]
        )

        # Register as buffer so it moves with the model to GPU/CPU
        self.register_buffer("seg_ids_pattern", seg_ids_pattern)

        # We need to handle register tokens - let's extract them from the backbone
        # This is a bit tricky since they're not directly exposed
        # We'll create our own register tokens initialized similarly
        if hasattr(self.backbone, "register_tokens"):
            self.register_tokens = self.backbone.register_tokens
        else:
            # Create register tokens if not available
            self.register_tokens = nn.Parameter(
                torch.zeros(1, self.num_register_tokens, self.num_features)
            )
            # Initialize with small random values
            nn.init.trunc_normal_(self.register_tokens, std=0.02)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        B = img1.size(0)

        # 1. Create patch embeddings for each image separately
        # DINOv3 patch_embed returns (B, H, W, D) format, need to flatten to (B, N, D)
        patches1 = self.patch_embed(img1)  # (B, 14, 14, 384)
        patches2 = self.patch_embed(img2)  # (B, 14, 14, 384)

        # Flatten spatial dimensions: (B, H, W, D) -> (B, H*W, D)
        patches1 = patches1.flatten(1, 2)  # (B, 196, 384)
        patches2 = patches2.flatten(1, 2)  # (B, 196, 384)

        # 2. Prepare tokens: CLS + register_tokens + patches_img1 + patches_img2
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, 384)
        register_tokens = self.register_tokens.expand(B, -1, -1)  # (B, 4, 384)

        # 3. Concatenate all tokens
        # Shape: (B, 1 + 4 + 196 + 196, 384) = (B, 397, 384)
        x = torch.cat([cls_token, register_tokens, patches1, patches2], dim=1)

        # 4. Add segment embeddings to disambiguate different token types
        seg_ids = self.seg_ids_pattern.unsqueeze(0).expand(B, -1)  # (B, 397)
        x = x + self.segment_embed(seg_ids)

        # 5. Apply rotary position embeddings through the transformer blocks
        # DINOv3 handles RoPE internally in each attention layer
        for block in self.blocks:
            x = block(x)

        # 6. Apply final normalization
        x = self.norm(x)

        # 7. Return the embedding of the [CLS] token, which represents the transformation
        return x[:, 0]


def get_encoder(encoder_name: str) -> tuple[nn.Module, int]:
    """
    Dynamically gets a pretrained model and removes its final layer.
    Returns both the encoder and the number of input features for the projection layer.
    """
    model = timm.create_model(encoder_name, pretrained=True)

    if encoder_name.startswith("resnet"):
        encoder = nn.Sequential(*list(model.children())[:-1])
        num_features = model.fc.in_features
        return encoder, num_features

    if encoder_name.startswith("vit"):
        # Check if it's a DINOv3 model
        if "dinov3" in encoder_name.lower():

            class DINOv3Encoder(nn.Module):
                def __init__(self, dinov3_model):
                    super().__init__()
                    self.dinov3 = dinov3_model

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    # forward_features returns (B, num_tokens, embed_dim)
                    # For DINOv3: num_tokens = 1 (CLS) + 4 (registers) + 196 (patches) = 201
                    # We extract the class token (first token) which has shape (B, embed_dim)
                    features = self.dinov3.forward_features(
                        x
                    )  # (B, 201, 384) for vit_small_patch16_dinov3
                    return features[:, 0]  # Extract class token: (B, embed_dim)

            return DINOv3Encoder(model), model.num_features
        else:
            # Standard ViT models
            class ViTEncoder(nn.Module):
                def __init__(self, vit_model):
                    super().__init__()
                    self.vit = vit_model

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    # forward_features returns (B, num_patches + 1, embed_dim)
                    # We extract the class token (first token) which has shape (B, embed_dim)
                    features = self.vit.forward_features(
                        x
                    )  # (B, 197, 384) for vit_small_patch16_224
                    return features[:, 0]  # Extract class token: (B, embed_dim)

            return ViTEncoder(model), model.num_features

    raise ValueError(f"Encoder name '{encoder_name}' not supported.")


class SiameseAnalogyNetwork(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        freeze_encoder: bool = False,
        encoder_name: str = "vit_small_patch16_224",
    ):
        super().__init__()

        # Use appropriate encoder based on model type
        if encoder_name.startswith("vit"):
            if "dinov3" in encoder_name.lower():
                # Use specialized DINOv3 TransformationEncoder
                self.encoder = TransformationEncoderDINOv3(encoder_name, pretrained=True)
                encoder_output_dim = self.encoder.num_features
                self.use_transformation_encoder = True
            else:
                # Use standard ViT TransformationEncoder
                self.encoder = TransformationEncoder(encoder_name)
                encoder_output_dim = self.encoder.num_features
                self.use_transformation_encoder = True
        else:
            # Use standard encoder for ResNet and other models
            self.encoder, encoder_output_dim = get_encoder(encoder_name)
            self.use_transformation_encoder = False

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # The projection head remains the same
        self.projection = nn.Sequential(
            nn.Linear(encoder_output_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
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
        if self.use_transformation_encoder:
            # Use the new TransformationEncoder approach for ViT models
            # Encode each transformation pair directly
            # The output of the encoder is now the "transformation vector"
            t_example_raw = self.encoder(ex_before, ex_after)
            t_choice_a_raw = self.encoder(test_before, choice_a)
            t_choice_b_raw = self.encoder(test_before, choice_b)
            t_choice_c_raw = self.encoder(test_before, choice_c)

            # Project all raw transformation embeddings
            all_raw_embeddings = torch.cat(
                [t_example_raw, t_choice_a_raw, t_choice_b_raw, t_choice_c_raw], dim=0
            )

            all_embeddings = self.projection(all_raw_embeddings)

            # Split them back
            batch_size = ex_before.size(0)
            t_example, t_choice_a_vec, t_choice_b_vec, t_choice_c_vec = torch.split(
                all_embeddings, batch_size, dim=0
            )
        else:
            # Optimized approach for ResNet models - process in parallel without concatenation
            batch_size = ex_before.size(0)

            # Stack tensors for parallel processing (no memory copy)
            input_stack = torch.stack(
                [ex_before, ex_after, test_before, choice_a, choice_b, choice_c], dim=1
            )

            # Reshape to process all images in parallel
            input_batch = input_stack.view(-1, *input_stack.shape[2:])  # (batch_size * 6, C, H, W)

            # Process all embeddings in one forward pass
            all_embeddings = self.projection(self.encoder(input_batch).flatten(1))

            # Reshape back to separate the 6 different image types
            all_embeddings = all_embeddings.view(
                batch_size, 6, -1
            )  # (batch_size, 6, embedding_dim)

            # Extract embeddings for each image type
            t_ex_before = all_embeddings[:, 0]  # (batch_size, embedding_dim)
            t_ex_after = all_embeddings[:, 1]
            t_test_before = all_embeddings[:, 2]
            t_choice_a = all_embeddings[:, 3]
            t_choice_b = all_embeddings[:, 4]
            t_choice_c = all_embeddings[:, 5]

            # Calculate the transformation vectors using simple subtraction
            t_example = t_ex_after - t_ex_before
            t_choice_a_vec = t_choice_a - t_test_before
            t_choice_b_vec = t_choice_b - t_test_before
            t_choice_c_vec = t_choice_c - t_test_before

        # Batch normalize all vectors at once for efficiency
        all_vectors = torch.stack(
            [t_example, t_choice_a_vec, t_choice_b_vec, t_choice_c_vec], dim=1
        )
        all_vectors_normalized = F.normalize(all_vectors, p=2, dim=-1)

        # Extract normalized vectors
        t_example = all_vectors_normalized[:, 0]
        t_choice_a_vec = all_vectors_normalized[:, 1]
        t_choice_b_vec = all_vectors_normalized[:, 2]
        t_choice_c_vec = all_vectors_normalized[:, 3]

        return t_example, t_choice_a_vec, t_choice_b_vec, t_choice_c_vec
