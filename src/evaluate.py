import torch
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import (
    DataLoader,
    random_split
)

from transformers import CLIPProcessor

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.dataset import FashionDataset
from src.model import MultimodalClassifier




BATCH_SIZE = 8

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

_, val_dataset = random_split(

    dataset,

    [
        train_size,
        val_size
    ]
)

val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False
)



model = MultimodalClassifier(
    dataset.num_classes
)

model.load_state_dict(

    torch.load(
        "model/best_model.pth",
        map_location=DEVICE
    )
)

model = model.to(
    DEVICE
)

model.eval()

print(
    "\nModel Loaded Successfully"
)



predictions = []
targets = []

with torch.no_grad():

    for batch in val_loader:

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

print("\n" + "=" * 60)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print("=" * 60)



class_names = [

    dataset.idx_to_label[i]

    for i in range(
        dataset.num_classes
    )
]

report = classification_report(

    targets,

    predictions,

    target_names=class_names
)

print(
    "\nClassification Report\n"
)

print(
    report
)

with open(
    "classification_report.txt",
    "w"
) as f:

    f.write(
        report
    )

print(
    "\nSaved classification_report.txt"
)



cm = confusion_matrix(

    targets,

    predictions
)

plt.figure(
    figsize=(10,8)
)

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=class_names,

    yticklabels=class_names
)

plt.title(
    "Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png"
)

plt.show()

print(
    "\nSaved confusion_matrix.png"
)