from PIL import Image
import numpy as np
import torch

repo = "isl-org/ZoeDepth"
model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=False)
pretrained_dict = torch.hub.load_state_dict_from_url('https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_NK.pt', map_location='cpu')
model_zoe_nk.load_state_dict(pretrained_dict['model'], strict=False)
for b in model_zoe_nk.core.core.pretrained.model.blocks:
    b.drop_path = torch.nn.Identity()


def estimate_metric_depth(src):
  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
  zoe = model_zoe_nk.to(DEVICE)

  # Local file
  image = Image.open(src).convert("RGB")  # load
  depth_numpy = zoe.infer_pil(image, pad_input=False)  # as numpy
  depth_numpy = np.transpose(depth_numpy)

  return depth_numpy


src = "Z:\ML_proj\\3D-BoundingBox\eval\image_2\\000002.png"

depth = estimate_metric_depth(src)

# print(image.size)
# width, height = im.size
print('min: ', depth.min())
print('max: ', depth.max())