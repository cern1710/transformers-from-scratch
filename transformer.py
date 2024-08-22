import torch
from torch import nn, optim

import positional_encoding as pe
import encoder as enc
import scheduler as sch
import decoder as dec

class TransformerPredictor(nn.Module):
    def __init__(self, input_dim: int, model_dim: int,
                 num_classes: int, num_heads: int, num_layers: int,
                 learning_rate: float, warmup: int, max_iter: int,
                 dropout: float, input_dropout: float):
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.warmup = warmup
        self.max_iter = max_iter
        self.dropout = dropout
        self.input_dropout = input_dropout
        self._init_model()

    def _init_model(self):
        # Simple layer mapping input dim -> model dim
        self.input_network = nn.Sequential(
            nn.Dropout(self.input_dropout),
            nn.Linear(self.input_dim, self.model_dim)
        )

        # Positional encoding ffor sequences
        self.positional_encoding = pe.PositionalEncoding(self.model_dim)

        # Transformer architecture
        self.encoder = enc.TransformerEncoder(
            num_layers=self.num_layers,
            input_dim=self.model_dim,
            ff_dim=self.model_dim * 2,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
        self.decoder = dec.TransformerDecoder(
            num_layers=self.num_layers,
            input_dim=self.model_dim,
            ff_dim=self.model_dim * 2,
            num_heads=self.num_heads,
            dropout=self.dropout
        )

        # Classifier per sequence
        self.output_network = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.model_dim, self.num_classes)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None,
                include_positional_encoding: bool = True,
                include_decoder: bool = False):
        x = self.input_network(x)
        if include_positional_encoding:
            x = self.positional_encoding(x)
        if include_decoder:
            encoder_output = self.encoder(x, mask=mask)
            x = self.decoder(x, encoder_output, self_mask=mask, cross_mask=mask)
        else:
            x = self.encoder(x, mask=mask)
        x = self.output_network(x)
        return x

    @torch.no_grad()
    def get_attention_maps(self, x: torch.Tensor, mask: torch.Tensor = None,
                           include_positional_encoding: bool = True,
                           include_decoder: bool = False):
        x = self.input_network(x)
        if include_positional_encoding:
            x = self.positional_encoding(x)

        encoder_output = self.encoder.get_attention_maps(x, mask=mask)

        if include_decoder:
            decoder_output = self.decoder.get_attention_maps(
                x, encoder_output[-1], self_mask=mask, cross_mask=mask
            )
            return decoder_output
        else:
            return encoder_output

    def configure_optimizer(self):
        optimizer = optim.Adam(self.parameters())
        lr_scheduler = sch.ExponentialDecayScheduler(
            optimizer=optimizer,
            warmup=self.warmup,
            max_iter=self.max_iter
        )
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]