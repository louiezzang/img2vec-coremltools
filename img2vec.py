import sys
import re
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision

import coremltools as ct
from PIL import Image


class Img2Vec(nn.Module):
    def __init__(self, prediction_model):
        super().__init__()
        # Pretrained model for extracting embedding.
        self.pt_model = torchvision.models.resnet18(pretrained=True)

        # Finetuned model for predict class label.
        self.ft_model = torchvision.models.resnet18(pretrained=True)
        for param in self.ft_model.parameters():
            param.requires_grad = False

        ft_model_num_classes = len(prediction_model["idx2label_map"])
        ft_model_num_features = self.ft_model.fc.in_features
        self.ft_model.fc = nn.Linear(ft_model_num_features, ft_model_num_classes)
        self.ft_model.load_state_dict(prediction_model["model_state_dict"])

        self.pt_model.eval()
        self.ft_model.eval()

        # Embedding layer from the pretrained model.
        self.layer = self.pt_model._modules.get("avgpool")

    def forward(self, x):
        my_embedding = torch.zeros(1, 512, 1, 1)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        h = self.layer.register_forward_hook(copy_data)
        _ = self.pt_model(x)
        prediction = self.ft_model(x)

        h.remove()

        return prediction.squeeze(0), torch.tensor(my_embedding.numpy()[0, :, 0, 0])


def load_released_model(model_file, torch_device=None):
    model = torch.load(model_file, map_location=torch_device) if torch_device else torch.load(model_file)

    # In case we load a DDP model checkpoint to a non-DDP model.
    for state_dict_key, state_dict in model.items():
        model_state_dict = OrderedDict()
        pattern = re.compile('module.')
        for k, v in state_dict.items():
            if isinstance(k, str) and re.search("module", k):
                model_state_dict[re.sub(pattern, '', k)] = v
            else:
                model_state_dict = model[state_dict_key]
        model[state_dict_key] = model_state_dict

    return model


if __name__ == "__main__":
    # Download class labels for resnet pretrained model.
    # import urllib
    # label_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    # class_labels = urllib.request.urlopen(label_url).read().decode("utf-8").splitlines()
    # assert len(class_labels) == 1000
    # print(class_labels)

    ft_pred_model_bin = load_released_model("resnet18_ft_model-20220615133619.pth.tar", torch.device("cpu"))
    ft_model_idx2label_map = dict(sorted(ft_pred_model_bin["idx2label_map"].items()))
    ft_model_class_labels = list(ft_model_idx2label_map.values())
    print(f"class labels of finetuned prediction model: {ft_model_class_labels}")

    dummy_input = torch.rand(1, 3, 224, 224)

    img2vec = Img2Vec(ft_pred_model_bin)
    # pred, emb = img2vec(dummy_input)
    # print(emb)

    # Changed to JIT Tracer.
    traced_model = torch.jit.trace(img2vec, dummy_input)
    pred, emb = traced_model(dummy_input)
    # print(emb)

    # Converts to Coremltools compatible model.
    cml_model = ct.convert(traced_model,
                           inputs=[ct.ImageType(name="image", color_layout="RGB", scale=1.0/255.0/0.226,
                                                bias=(-0.485/0.226, -0.456/0.226, -0.406/0.226),
                                                shape=dummy_input.shape)],
                           classifier_config=ct.ClassifierConfig(ft_model_class_labels),
                           source="pytorch"
                           )

    # Change the output names.
    spec = cml_model.get_spec()
    ct.utils.rename_feature(spec, spec.description.output[0].name, "pred", rename_outputs=True)
    ct.utils.rename_feature(spec, spec.description.output[1].name, "emb", rename_outputs=True)
    cml_model = ct.models.MLModel(spec)

    cml_model.save("img2vec.mlmodel")

    IS_MACOS = sys.platform == "darwin"

    test_img = Image.open("sample_img.webp")
    test_img = test_img.resize((224, 224))

    if IS_MACOS:
        loaded_model = ct.models.MLModel("img2vec.mlmodel")
        pred_output = loaded_model.predict({"image": test_img})
        print("=" * 50)
        # print(pred_output)

        for k, v in pred_output.items():
            print(f"{k}: {v}")
