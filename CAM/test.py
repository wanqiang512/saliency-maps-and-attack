from matplotlib import pyplot as plt
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image

x = read_image("0.jpg")
img = x
x = (resize(x, (224, 224)) / 255.0).unsqueeze(0)
x = normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

from torchvision.models import resnet18
# from torchcam.methods import GradCAMpp
#
model = resnet18(pretrained=True).eval()
out = model(x)
# cam = GradCAMpp(model=model)
# grayscale_cam = cam(out.argmax(dim=1).item(), out)
# print(grayscale_cam[0])
# import matplotlib.pyplot as plt
#
# # Visualize the raw CAM
# plt.imshow(grayscale_cam[0].squeeze().numpy());
# plt.axis('off');
# plt.tight_layout();
# plt.show()

from GradCAMplusplus import GradCamplusplus

cam = GradCamplusplus(model)
test = cam.get_gradient(x, "layer4", out.argmax(dim=1).item())
print(test)
# Visualize the raw CAM
plt.imshow(test[0].squeeze().numpy());
plt.axis('off');
plt.tight_layout();
plt.show()

from torchcam.utils import  overlay_mask
# Resize the CAM and overlay it
result = overlay_mask(to_pil_image(img), to_pil_image(test[0].squeeze(0), mode='F'), alpha=0.5)
# Display it
plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()
