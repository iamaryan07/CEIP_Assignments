"""Bonus A -- GradCAM visualisation on the fine-tuned ResNet-18.

Overlays a heatmap of which pixels drove each classification, for 3+
hand-picked validation examples (one correct, plus the spatial-leakage and
error-analysis scripts feed it harder, misclassified examples too).

Run: python -m src.gradcam
"""
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from . import config, utils
from .transfer_model import build_resnet18
from .transforms import resnet_transforms

IMAGE_SIZE = 224


def load_model(device):
    model = build_resnet18(pretrained=False)
    model.load_state_dict(torch.load(config.CHECKPOINT_DIR / "resnet18_finetuned.pt", map_location=device))
    model.to(device).eval()
    return model


def run_gradcam_on_image(model, device, image_path, true_class_idx=None):
    transform = resnet_transforms(train=False, image_size=IMAGE_SIZE)
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_idx)])[0]

    rgb_img = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32) / 255.0
    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return {
        "overlay": overlay,
        "rgb_img": (rgb_img * 255).astype(np.uint8),
        "pred_idx": pred_idx,
        "pred_class": config.EUROSAT_CLASSES[pred_idx],
        "confidence": confidence,
        "true_class": config.EUROSAT_CLASSES[true_class_idx] if true_class_idx is not None else None,
    }


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = utils.get_device()
    model = load_model(device)

    df = pd.read_csv(config.DATA_DIR / "eurosat_index.csv")
    val_df = df[df["split"] == "val"]

    samples = val_df.groupby("class_name", group_keys=False).sample(
        n=1, random_state=config.SEED
    ).sample(4, random_state=config.SEED)

    fig, axes = plt.subplots(2, len(samples), figsize=(4 * len(samples), 8))
    for i, row in enumerate(samples.itertuples()):
        result = run_gradcam_on_image(model, device, row.path, row.class_idx)
        axes[0, i].imshow(result["rgb_img"])
        axes[0, i].set_title(f"True: {result['true_class']}")
        axes[0, i].axis("off")
        axes[1, i].imshow(result["overlay"])
        correctness = "correct" if result["pred_class"] == result["true_class"] else "WRONG"
        axes[1, i].set_title(f"Pred: {result['pred_class']} ({result['confidence']:.2f}) [{correctness}]")
        axes[1, i].axis("off")

    fig.suptitle("GradCAM -- which pixels drove each ResNet-18 prediction")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "gradcam" / "gradcam_examples.png", dpi=150)
    print("Saved GradCAM grid to", config.FIGURES_DIR / "gradcam" / "gradcam_examples.png")


if __name__ == "__main__":
    main()
