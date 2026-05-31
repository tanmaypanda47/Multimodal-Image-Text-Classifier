import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

from transformers import CLIPProcessor


class FashionDataset(Dataset):

    def __init__(
        self,
        csv_file,
        image_dir,
        processor,
        min_samples=100
    ):

        self.processor = processor
        self.image_dir = image_dir

        print("Loading metadata...")

        self.df = pd.read_csv(
            csv_file,
            on_bad_lines="skip"
        )

       

        self.df = self.df[
            [
                "id",
                "productDisplayName",
                "masterCategory"
            ]
        ]


        self.df = self.df.dropna()



        category_counts = self.df[
            "masterCategory"
        ].value_counts()

        valid_categories = category_counts[
            category_counts >= min_samples
        ].index

        self.df = self.df[
            self.df[
                "masterCategory"
            ].isin(
                valid_categories
            )
        ]

        print(
            f"\nKeeping categories with >= {min_samples} samples"
        )

        print(
            f"Remaining Categories: {len(valid_categories)}"
        )



        categories = sorted(
            self.df[
                "masterCategory"
            ].unique()
        )

        self.label_map = {

            category: idx

            for idx, category
            in enumerate(categories)
        }

        self.idx_to_label = {

            idx: category

            for category, idx
            in self.label_map.items()
        }

        self.df["label"] = self.df[
            "masterCategory"
        ].map(
            self.label_map
        )

 

        valid_rows = []

        print(
            "\nChecking image files..."
        )

        for _, row in self.df.iterrows():

            image_path = os.path.join(

                image_dir,

                f"{row['id']}.jpg"
            )

            if os.path.exists(
                image_path
            ):

                valid_rows.append(
                    row
                )

        self.df = pd.DataFrame(
            valid_rows
        ).reset_index(
            drop=True
        )



        self.num_classes = len(
            self.label_map
        )

        print(
            f"\nLoaded Samples: {len(self.df)}"
        )

        print(
            f"Classes: {self.num_classes}"
        )

        print(
            "\nCategory Distribution:"
        )

        print(
            self.df[
                "masterCategory"
            ].value_counts()
        )

    def __len__(self):

        return len(
            self.df
        )

    def __getitem__(
        self,
        idx
    ):

        row = self.df.iloc[idx]

        image_path = os.path.join(

            self.image_dir,

            f"{row['id']}.jpg"
        )

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )

        text = str(
            row[
                "productDisplayName"
            ]
        )

        label = int(
            row[
                "label"
            ]
        )

        encoding = self.processor(

            text=[text],

            images=image,

            return_tensors="pt",

            padding="max_length",

            truncation=True,

            max_length=77
        )

        return {

            "input_ids":
                encoding[
                    "input_ids"
                ].squeeze(0),

            "attention_mask":
                encoding[
                    "attention_mask"
                ].squeeze(0),

            "pixel_values":
                encoding[
                    "pixel_values"
                ].squeeze(0),

            "label":
                torch.tensor(
                    label,
                    dtype=torch.long
                )
        }