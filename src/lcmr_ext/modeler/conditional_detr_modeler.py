import torch
import torch.nn as nn
from lcmr.grammar import Scene
from lcmr.grammar.shapes import Shape2D
from lcmr.modeler import Modeler
from lcmr.utils.guards import batch_dim, channel_dim, reduced_height_dim, reduced_width_dim, typechecked
from torchtyping import TensorType
from transformers.models.conditional_detr.configuration_conditional_detr import ConditionalDetrConfig
from transformers.models.conditional_detr.modeling_conditional_detr import ConditionalDetrDecoder, ConditionalDetrEncoder, ConditionalDetrMLPPredictionHead, build_position_encoding

from lcmr_ext.modeler.efd_module import EfdModule
from lcmr_ext.modeler.modeler_config import ModelerConfig

# Copied detr_modeler.py, changed "Detr.*" classes to "ConditionalDetr.*" classes
# TODO: consider merging DETRModeler and ConditionalDETRModeler, add a parameter to select variant

# TODO: add params related to shape and background color


@typechecked
class ConditionalDETRModeler(Modeler):
    def __init__(self, config: ModelerConfig, detr_config: ConditionalDetrConfig = ConditionalDetrConfig()):
        super().__init__()

        self.use_single_scale = config.use_single_scale
        self.num_queries = detr_config.num_queries

        # nn.GroupNorm(1, config.encoder_feature_dim)
        self.input_projection = nn.Sequential(nn.BatchNorm2d(config.encoder_feature_dim), nn.Conv2d(config.encoder_feature_dim, detr_config.d_model, kernel_size=1, bias=False))
        self.query_position_embedding = nn.Embedding(detr_config.num_queries, detr_config.d_model)
        self.position_encoding = build_position_encoding(detr_config)

        self.encoder = ConditionalDetrEncoder(detr_config)
        self.decoder = ConditionalDetrDecoder(detr_config)

        # prediction heads
        def make_head(output_dim: int):
            return ConditionalDetrMLPPredictionHead(input_dim=detr_config.d_model, hidden_dim=detr_config.d_model, output_dim=output_dim, num_layers=config.prediction_head_layers)

        self.to_translation = make_head(output_dim=2)
        self.to_scale = make_head(output_dim=(1 if self.use_single_scale else 2))
        self.to_color = make_head(output_dim=3)
        if config.use_confidence:
            self.to_confidence = make_head(output_dim=1)
        else:
            self.to_confidence = None
        self.to_angle = make_head(output_dim=2)
        self.to_background_color = make_head(output_dim=3)

        self.to_efd = None
        efd_module_config = config.efd_module_config
        if efd_module_config != None:
            efd_module_config.input_dim = detr_config.d_model
            efd_module_config.hidden_dim = detr_config.d_model
            efd_module_config.num_layers = config.prediction_head_layers
            self.to_efd = EfdModule(efd_module_config)

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
        # flattened_features is a Tensor of shape (batch_size, height*width, hidden_size)
        # flattened_mask is a Tensor of shape (batch_size, height*width)

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
        if self.use_single_scale:
            scale = scale.expand(-1, -1, -1, 2)

        color = torch.sigmoid(self.to_color(hidden_state)).unsqueeze(1)

        if self.to_confidence != None:
            confidence = torch.sigmoid(self.to_confidence(hidden_state)).unsqueeze(1)
        else:
            confidence = torch.ones((batch_size, 1, self.num_queries, 1), dtype=torch.float32, device=device)

        rotation_vec = torch.tanh(self.to_angle(hidden_state))
        rotation_vec = nn.functional.normalize(rotation_vec, dim=-1).unsqueeze(1)
        #angle = torch.atan2(rotation_vec[..., 0, None], rotation_vec[..., 1, None]).unsqueeze(1)

        efd = self.to_efd(hidden_state).unsqueeze(1) if self.to_efd != None else None

        background_color = torch.sigmoid(self.to_background_color(hidden_state.mean(dim=-2)))

        objectShape = Shape2D.EFD_SHAPE.value if self.to_efd != None else Shape2D.DISK.value

        return Scene.from_tensors_sparse(
            translation=translation,
            scale=scale,
            color=color,
            confidence=confidence,
            rotation_vec=rotation_vec,
            efd=efd,
            objectShape=torch.ones((batch_size, 1, self.num_queries, 1), dtype=torch.uint8, device=device) * objectShape,
            backgroundColor=background_color,
            device=device,
        )
