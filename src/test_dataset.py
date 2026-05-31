from transformers import CLIPProcessor

from src.dataset import FashionDataset

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

dataset = FashionDataset(

    csv_file="data/styles.csv",

    image_dir="data/images",

    processor=processor
)

print("\nDataset Size:")
print(len(dataset))

print("\nNumber of Classes:")
print(dataset.num_classes)

sample = dataset[0]

print("\nKeys:")
print(sample.keys())

print("\nInput IDs:")
print(sample["input_ids"].shape)

print("\nImage:")
print(sample["pixel_values"].shape)

print("\nLabel:")
print(sample["label"])