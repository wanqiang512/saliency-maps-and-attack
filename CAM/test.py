from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image

x = read_image("test.jpg").unsqueeze(0)
x = resize(x, (224, 224)) / 255.0
x = normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

from torchvision.models import resnet50
from torchcam.methods import GradCAMpp

model = resnet50(pretrained=True).eval()
cam = GradCAMpp(model=model, target_layer="layer4")
out = model(x)
grayscale_cam = cam(out.argmax(dim=1).item(), out)
print(grayscale_cam[0])
import matplotlib.pyplot as plt

# Visualize the raw CAM
plt.imshow(grayscale_cam[0].squeeze().numpy());
plt.axis('off');
plt.tight_layout();
plt.show()

from GradCAMplusplus import GradCamplusplus

cam = GradCamplusplus(model)
test = cam.get_gradient(x, "layer4", out.argmax(dim=1).item())
print(test)
