import torch
import torch.nn as nn
from torchtyping import TensorType
from transformers.models.conditional_detr.modeling_conditional_detr import (
    ConditionalDetrEncoder,
    ConditionalDetrDecoder,
    ConditionalDetrMLPPredictionHead,
    build_position_encoding,
)
from transformers.models.conditional_detr import ConditionalDetrConfig

from lcmr.grammar import Scene
from lcmr.modeler import Modeler
from lcmr.utils.guards import typechecked, batch_dim, reduced_height_dim, reduced_width_dim, channel_dim

# Copied detr_modeler.py, changed "Detr.*" classes to "ConditionalDetr.*" classes
# TODO: consider merging DETRModeler and ConditionalDETRModeler, add a parameter to select variant

# TODO: add params related to shape and background color


@typechecked
class ConditionalDETRModeler(Modeler):
    def __init__(self, config: ConditionalDetrConfig = ConditionalDetrConfig(), encoder_feature_dim: int = 2048, prediction_head_layers: int = 3):
        super().__init__()

        self.input_projection = nn.Conv2d(encoder_feature_dim, config.d_model, kernel_size=1)
        self.query_position_embedding = nn.Embedding(config.num_queries, config.d_model)
        self.position_encoding = build_position_encoding(config)

        self.encoder = ConditionalDetrEncoder(config)
        self.decoder = ConditionalDetrDecoder(config)

        # prediction heads
        def make_head(output_dim: int):
            return ConditionalDetrMLPPredictionHead(
                input_dim=config.d_model, hidden_dim=config.d_model, output_dim=output_dim, num_layers=prediction_head_layers
            )

        self.to_translation = make_head(output_dim=2)
        self.to_scale = make_head(output_dim=2)
        self.to_color = make_head(output_dim=3)
        self.to_confidence = make_head(output_dim=1)
        self.to_angle = make_head(output_dim=2)
        self.to_fourier_shape = make_head(output_dim=32)
        self.to_background_color = make_head(output_dim=3)

    def forward(self, x: TensorType[batch_dim, reduced_height_dim, reduced_width_dim, channel_dim, torch.float32]) -> Scene:
        device = next(self.parameters()).device
        batch_size, num_channels, height, width = x.shape

        # First, prepare the position embeddings
        pixel_mask = torch.ones(((batch_size, height, width)), device=device)
        object_queries = self.position_encoding(x, pixel_mask)

        # Second, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        features = self.input_projection(x)

        # Third, flatten the feature map + position embeddings of shape NxCxHxW to NxCxHW, and permute it to NxHWxC
        # In other words, turn their shape into (batch_size, sequence_length, hidden_size)
        flattened_features = features.flatten(2).permute(0, 2, 1)
        object_queries = object_queries.flatten(2).permute(0, 2, 1)
        flattened_mask = pixel_mask.flatten(1)

        # Fourth, sent flattened_features + flattened_mask + position embeddings through encoder
        # flattened_features is a Tensor of shape (batch_size, heigth*width, hidden_size)
        # flattened_mask is a Tensor of shape (batch_size, heigth*width)

        encoder_outputs = self.encoder(
            inputs_embeds=flattened_features,
            attention_mask=flattened_mask,
            object_queries=object_queries,
        )

        # Fifth, sent query embeddings + object_queries through the decoder (which is conditioned on the encoder output)
        query_position_embedding = self.query_position_embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        queries = torch.zeros_like(query_position_embedding)

        # Decoder outputs consists of (dec_features, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            inputs_embeds=queries,
            attention_mask=None,
            object_queries=object_queries,
            query_position_embeddings=query_position_embedding,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=flattened_mask,
        )
        hidden_state = decoder_outputs.last_hidden_state

        # Finally project transformer outputs to scene
        translation = torch.sigmoid(self.to_translation(hidden_state)).unsqueeze(1)
        scale = torch.sigmoid(self.to_scale(hidden_state)).unsqueeze(1)
        color = torch.sigmoid(self.to_color(hidden_state)).unsqueeze(1)
        confidence = torch.sigmoid(self.to_confidence(hidden_state)).unsqueeze(1)
        rotation_vec = torch.tanh(self.to_angle(hidden_state))
        rotation_vec = nn.functional.normalize(rotation_vec, dim=-1)
        angle = torch.atan2(rotation_vec[..., 0, None], rotation_vec[..., 1, None]).unsqueeze(1)

        fourier_shape = self.to_fourier_shape(hidden_state).unsqueeze(1).unflatten(-1, (8, 4))
        fourier_shape[..., 0, 0] = 1.0
        fourier_shape[..., 0, 3] = -torch.sigmoid(fourier_shape[..., 0, 3])
        fourier_shape[..., 0, 1:3] = 0.0
        fourier_shape[..., 1:, :] = torch.tanh(fourier_shape[..., 1:, :])

        background_color = torch.sigmoid(self.to_background_color(hidden_state.mean(dim=-2)))

        return Scene.from_tensors_sparse(
            translation=translation,
            scale=scale,
            color=color,
            confidence=confidence,
            angle=angle,
            fourierCoefficients=fourier_shape,
            objectShape=torch.ones(confidence.shape, dtype=torch.uint8) * 2,
            backgroundColor=background_color,
            device=device,
        )
