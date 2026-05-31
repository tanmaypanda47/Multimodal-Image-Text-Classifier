import torch
import torch.nn as nn

from transformers import CLIPModel


class MultimodalClassifier(nn.Module):

    def __init__(
        self,
        num_classes
    ):

        super().__init__()

        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        # Freeze CLIP initially
        for param in self.clip.parameters():

            param.requires_grad = False

        self.classifier = nn.Sequential(

            nn.Linear(
                1024,
                512
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                512,
                256
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                256,
                num_classes
            )
        )

    def forward(

        self,

        input_ids,

        attention_mask,

        pixel_values
    ):

        image_features = self.clip.get_image_features(
            pixel_values=pixel_values
        )

        text_features = self.clip.get_text_features(

            input_ids=input_ids,

            attention_mask=attention_mask
        )

        combined = torch.cat(

            [
                image_features,
                text_features
            ],

            dim=1
        )

        logits = self.classifier(
            combined
        )

        return logits