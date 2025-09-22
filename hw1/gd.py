import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import VGG13_BN_Weights, vgg13_bn
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def save_img(image, path):
    # Push to CPU, convert from (1, 3, H, W) into (H, W, 3)
    image = image[0].permute(1, 2, 0)
    image = image.clamp(min=0, max=1)
    image = (image * 255).cpu().detach().numpy().astype(np.uint8)
    # opencv expects BGR (and not RGB) format
    cv.imwrite(path, image[:, :, ::-1])


def main():
    model = vgg13_bn(VGG13_BN_Weights.IMAGENET1K_V1).to(DEVICE)
    print(model)
    for label in [0, 12, 954]:
        image = torch.randn(1, 224, 224, 3).to(DEVICE)
        image = (image * 8 + 128) / 255  # background color = 128,128,128
        image = image.permute(0, 3, 1, 2)
        image.requires_grad_()
        image = gradient_descent(image, model, lambda tensor: tensor[0, label].mean(),)
        save_img(image, f"./img_{label}.jpg")
        out = model(image)
        print(f"ANSWER_FOR_LABEL_{label}: {out.softmax(1)[0, label].item()}")


# DO NOT CHANGE ANY OTHER FUNCTIONS ABOVE THIS LINE FOR THE FINAL SUBMISSION


def normalize_and_jitter(img, step=32):
    # You should use this as data augmentation and normalization,
    # convnets expect values to be mean 0 and std 1
    dx, dy = np.random.randint(-step, step - 1, 2)
    return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(
        img.roll(dx, -1).roll(dy, -2)
    )


# Done: Implement the gradient descent function
def gradient_descent(input, model, loss, iterations=256):
    # We perform gradient ascent on the input to maximize the provided loss
    model.eval()
    # Freeze model parameters to avoid computing/storing their gradients
    for p in model.parameters():
        p.requires_grad = False

    step_size = 0.01

    img = input
    if not img.requires_grad:
        img.requires_grad_()

    for iter in tqdm(range(iterations)):
        # Forward with normalization and jitter for better visualization
        img_aug = normalize_and_jitter(img)
        out = model(img_aug)
        objective = loss(out)

        # Clear previous gradients
        if img.grad is not None:
            img.grad.zero_()
        model.zero_grad(set_to_none=True)

        # Compute gradients w.r.t. the original image
        objective.backward()

        with torch.no_grad():
            grad = img.grad
            # Normalize gradient for stable updates (RMS-style)
            grad_scale = grad.abs().mean()
            if grad_scale > 0:
                grad = grad / grad_scale
            # Gradient ascent step
            img.add_(step_size * grad)

            if iter % 20 == 0:
                # Same-shape blur (no downsampling)
                img[:] = F.avg_pool2d(img, kernel_size=3, stride=1, padding=1)

            # Keep the image in valid range expected by save/visualization
            img.clamp_(0.0, 1.0)

    return img


def forward_and_return_activation(model, input, module):
    """
    This function is for the extra credit. You may safely ignore it.
    Given a module in the middle of the model (like `model.features[20]`),
    it will return the intermediate activations.
    Try setting the modeul to `model.features[20]` and the loss to `tensor[0, ind].mean()`
    to see what intermediate activations activate on.
    """
    features = []

    def hook(model, input, output):
        features.append(output)

    handle = module.register_forward_hook(hook)
    model(input)
    handle.remove()

    return features[0]


if __name__ == "__main__":
    main()
