import torch
import torchvision.transforms as T

from .models.clip import CLIP

from .models.clip import IMAGE_SIZE

MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]

preprocess_crop = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(MEAN, STD)
])

def compute_features_from_crops(model: CLIP, crops_preprocessed: torch.Tensor):
    with torch.no_grad():
        crops_features = (
            model.encode_image(crops_preprocessed.cuda())
            # Last layer + send to CPU
            [:, 0, :].cpu()
        )

    return crops_features.numpy()
