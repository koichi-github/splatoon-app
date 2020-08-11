import torch
import onnx
from onnx_coreml import convert
from models import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load('checkpoints/ckpt_e10.pth', map_location=device))

dummy_input = torch.FloatTensor(1, 1, 28, 28)
torch.onnx.export(model, dummy_input, 'classification.proto', verbose=True)

model_onnx = onnx.load('classification.proto')

coreml_model = convert(
    model_onnx,
    image_input_names=['input'],
    image_output_names=['output'],
    minimum_ios_deployment_target='13'
)
coreml_model.save('classification.mlmodel')