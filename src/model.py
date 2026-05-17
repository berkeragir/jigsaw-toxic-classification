"""Shared-trunk multi-label DistilBERT classifier.

Replaces v6's per-label MLP ModuleList with a single trunk that feeds one
6-way linear classifier. Labels co-occur heavily (>70% of positive rows
carry 2+ labels), so the trunk learns "what is toxic" once and the
classifier specializes per label.
"""

import torch.nn as nn
from transformers import DistilBertModel


class MultiLabelDistilBert(nn.Module):
    """DistilBERT -> shared trunk -> 6-way linear classifier.

    Args:
        num_labels: number of classification labels (6 for Jigsaw).
        dropout: dropout inside the trunk.
        use_embeddings_only: sanity-check mode; skips the transformer stack
            and uses only the [CLS] embedding vector. ~3-4x faster, ~0.30-0.40
            PR AUC ceiling.
        freeze_layers: number of bottom DistilBERT transformer layers to
            freeze. v7 default is 2 — lower layers encode general linguistic
            features that don't need to specialize.
        head_hidden_size: width of the shared trunk.
        verbose: print the freeze decision on init.
    """

    def __init__(
        self,
        num_labels: int = 6,
        dropout: float = 0.1,
        use_embeddings_only: bool = False,
        freeze_layers: int = 0,
        head_hidden_size: int = 384,
        verbose: bool = True,
    ):
        super().__init__()
        self.use_embeddings_only = use_embeddings_only
        self.freeze_layers = freeze_layers

        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        bert_hidden_size = self.bert.config.hidden_size  # 768

        if use_embeddings_only:
            for p in self.bert.transformer.parameters():
                p.requires_grad = False
            if verbose:
                print("  Transformer fully frozen (embeddings-only sanity mode)")
        elif freeze_layers > 0:
            num_layers = len(self.bert.transformer.layer)
            layers_to_freeze = min(freeze_layers, num_layers)
            for i in range(layers_to_freeze):
                for p in self.bert.transformer.layer[i].parameters():
                    p.requires_grad = False
            if verbose:
                print(
                    f"  Frozen layers 0..{layers_to_freeze-1}, "
                    f"training layers {layers_to_freeze}..{num_layers-1}"
                )
        elif verbose:
            print("  All layers trainable")

        self.trunk = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(bert_hidden_size, head_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(head_hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        if self.use_embeddings_only:
            pooled = self.bert.embeddings(input_ids)[:, 0, :]
        else:
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled = out.last_hidden_state[:, 0, :]  # [CLS]
        return self.classifier(self.trunk(pooled))

    def trunk_features(self, input_ids, attention_mask):
        """Return the 384-d trunk representation (used by v8 conditional heads)."""
        if self.use_embeddings_only:
            pooled = self.bert.embeddings(input_ids)[:, 0, :]
        else:
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled = out.last_hidden_state[:, 0, :]
        return self.trunk(pooled)
