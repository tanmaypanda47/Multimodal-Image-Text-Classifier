import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    random_split
)

from transformers import CLIPProcessor

from sklearn.metrics import (
    accuracy_score,
    f1_score
)

from tqdm import tqdm

from src.dataset import FashionDataset
from src.model import MultimodalClassifier




BATCH_SIZE = 8

EPOCHS = 5

LEARNING_RATE = 1e-4

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"\nUsing Device: {DEVICE}"
)



processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

dataset = FashionDataset(

    csv_file="data/styles.csv",

    image_dir="data/images",

    processor=processor
)



train_size = int(
    0.8 * len(dataset)
)

val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(

    dataset,

    [
        train_size,
        val_size
    ]
)

print(
    f"\nTrain Size: {train_size}"
)

print(
    f"Validation Size: {val_size}"
)



train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0
)

val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0
)



model = MultimodalClassifier(
    dataset.num_classes
)

model = model.to(
    DEVICE
)



criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE
)



best_f1 = 0

for epoch in range(EPOCHS):

    print(
        f"\nEpoch {epoch+1}/{EPOCHS}"
    )

    model.train()

    train_loss = 0

    loop = tqdm(
        train_loader
    )

    for batch in loop:

        input_ids = batch[
            "input_ids"
        ].to(
            DEVICE
        )

        attention_mask = batch[
            "attention_mask"
        ].to(
            DEVICE
        )

        pixel_values = batch[
            "pixel_values"
        ].to(
            DEVICE
        )

        labels = batch[
            "label"
        ].to(
            DEVICE
        )

        optimizer.zero_grad()

        outputs = model(

            input_ids,

            attention_mask,

            pixel_values
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        loop.set_postfix(
            loss=loss.item()
        )



    model.eval()

    predictions = []

    targets = []

    with torch.no_grad():

        for batch in tqdm(
            val_loader
        ):

            input_ids = batch[
                "input_ids"
            ].to(
                DEVICE
            )

            attention_mask = batch[
                "attention_mask"
            ].to(
                DEVICE
            )

            pixel_values = batch[
                "pixel_values"
            ].to(
                DEVICE
            )

            labels = batch[
                "label"
            ].to(
                DEVICE
            )

            outputs = model(

                input_ids,

                attention_mask,

                pixel_values
            )

            preds = torch.argmax(

                outputs,

                dim=1
            )

            predictions.extend(

                preds.cpu().numpy()
            )

            targets.extend(

                labels.cpu().numpy()
            )

    accuracy = accuracy_score(

        targets,

        predictions
    )

    f1 = f1_score(

        targets,

        predictions,

        average="weighted"
    )

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    print(
        f"F1 Score: {f1:.4f}"
    )



    if f1 > best_f1:

        best_f1 = f1

        torch.save(

            model.state_dict(),

            "model/best_model.pth"
        )

        print(
            "\nBest Model Saved"
        )

print(
    "\nTraining Complete"
)