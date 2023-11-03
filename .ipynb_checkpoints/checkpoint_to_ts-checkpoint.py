
import os

import torch
import torch.nn as nn

import config
import legacy_model as pytorch_model
import med_quant_model as quant_model


def build_model(checkpoint_path):
    model = quant_model.MyModel('convnext_small',384)
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_folder_path = os.path.join(project_path, "scripts/models")
    checkpoint_path = os.path.join(models_folder_path, checkpoint_path)

    checkpoint = torch.load(checkpoint_path,
                            map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint)
    model.eval()

    return model


def convert_to_torch_script(model, input_shape):
    example = torch.rand(input_shape)
    traced_script_module = torch.jit.trace(model, example, strict=False)
    return traced_script_module

# After converting to torch script, if we want to use the lite interpreter
# which is what we use on the mobile side to consume the model, we need to
# save the model with the following function. This function will save the
# model with the extension .ptl
def lite_interpreter_support(model):
    model._save_for_lite_interpreter("mobile_model.ptl") 


def convert(checkpoint_path):

    model = build_model(checkpoint_path)
    print("Model loaded successfully")
    # ts_model = convert_to_torch_script(model, (16, 1, 15360))
    # ts_model.save("torch_script_model.pt")
    # lite_interpreter_support(ts_model)


convert("model_int8_384 (2).pth")