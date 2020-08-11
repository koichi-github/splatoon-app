import torch
import onnx
from onnx_coreml import convert

from models import *
from utils.utils import *
from utils.datasets import *

num_out = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Darknet("config/yolov3-custom.cfg", img_size=416).to(device)
model = ResNet18(num_out).to(device)
# model.load_state_dict(torch.load('checkpoints/yolov3_ckpt_99.pth', map_location=device))
model.load_state_dict(torch.load('../CNN/custom/checkpoints/special/model_out2_epoch9.pth', map_location=device))

dummy_input = torch.FloatTensor(1, 3, 256, 256)
torch.onnx.export(model, dummy_input, 'special.proto', verbose=True)

model_onnx = onnx.load('special.proto')

coreml_model = convert(
    model_onnx,
    mode="classifier",
    class_labels=[0, 1],
    # preprocessing_args={"image_scale": 1./255},
    image_input_names=['image'],
    image_output_names=['output'],
    minimum_ios_deployment_target='13'
)
coreml_model.save('special.mlmodel')

# import coremltools
# import coremltools.proto.FeatureTypes_pb2 as ft 

# spec = coremltools.utils.load_spec("special.mlmodel")

# input = spec.description.input[0]
# input.type.imageType.colorSpace = ft.ImageFeatureType.RGB
# input.type.imageType.height = 256
# input.type.imageType.width = 256

# coremltools.utils.save_spec(spec, "special.mlmodel")