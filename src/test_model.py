import torch

from src.model import (
    MultimodalClassifier
)

model = MultimodalClassifier(
    num_classes=7
)

batch_size = 4

input_ids = torch.randint(
    0,
    100,
    (batch_size, 77)
)

attention_mask = torch.ones(
    batch_size,
    77
)

pixel_values = torch.randn(
    batch_size,
    3,
    224,
    224
)

output = model(

    input_ids,

    attention_mask,

    pixel_values
)

print(
    output.shape
)