import torch
from PIL import Image

from transformers import CLIPProcessor

from src.model import MultimodalClassifier
from src.dataset import FashionDataset




DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using Device: {DEVICE}")



processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)



dataset = FashionDataset(
    csv_file="data/styles.csv",
    image_dir="data/images",
    processor=processor,
    min_samples=100
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

model.to(DEVICE)
model.eval()

print("Model Loaded Successfully")




def predict(
    image,
    product_title
):

    if isinstance(
        image,
        str
    ):

        image = Image.open(
            image
        ).convert(
            "RGB"
        )

    else:

        image = image.convert(
            "RGB"
        )

    encoding = processor(

        text=[product_title],

        images=image,

        return_tensors="pt",

        padding=True,

        truncation=True
    )

    input_ids = encoding[
        "input_ids"
    ].to(
        DEVICE
    )

    attention_mask = encoding[
        "attention_mask"
    ].to(
        DEVICE
    )

    pixel_values = encoding[
        "pixel_values"
    ].to(
        DEVICE
    )

    with torch.no_grad():

        outputs = model(

            input_ids,

            attention_mask,

            pixel_values
        )

        probs = torch.softmax(
            outputs,
            dim=1
        )

    top_probs, top_idx = torch.topk(

        probs,

        k=min(
            3,
            dataset.num_classes
        )
    )

    predictions = []

    for prob, idx in zip(
        top_probs[0],
        top_idx[0]
    ):

        predictions.append({

            "category":
                dataset.idx_to_label[
                    idx.item()
                ],

            "confidence":
                round(
                    prob.item() * 100,
                    2
                )
        })

    return predictions




if __name__ == "__main__":

    result = predict(

        "data/images/10000.jpg",

        "Nike Running Shoes"
    )

    print(result)