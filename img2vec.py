import sys

import torch
import torch.nn as nn
import torchvision

import coremltools as ct
from PIL import Image


class Img2Vec(nn.Module):
    def __init__(self):
        super().__init__()
        # Pretrained model for extracting embedding.
        self.pt_model = torchvision.models.resnet18(pretrained=True)
        # Finetuned model for predict class label.
        self.ft_model = torchvision.models.resnet18(pretrained=True)

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


if __name__ == "__main__":
    # Download class labels.
    import urllib
    label_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    class_labels = urllib.request.urlopen(label_url).read().decode("utf-8").splitlines()
    assert len(class_labels) == 1000
    print(class_labels)

    dummy_input = torch.rand(1, 3, 224, 224)

    img2vec = Img2Vec()
    # pred, emb = img2vec(dummy_input)
    # print(emb)

    traced_model = torch.jit.trace(img2vec, dummy_input)
    pred, emb = traced_model(dummy_input)
    # print(emb)

    cml_model = ct.convert(traced_model,
                           inputs=[ct.ImageType(name="image", color_layout="RGB", scale=1.0/255.0/0.226,
                                                bias=(-0.485/0.226, -0.456/0.226, -0.406/0.226),
                                                shape=dummy_input.shape)],
                           classifier_config=ct.ClassifierConfig(class_labels),
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
