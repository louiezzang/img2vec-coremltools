import sys

import torch
import torch.nn as nn
import torchvision

import coremltools as ct
from PIL import Image


class Img2Vec(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.eval()
        self.layer = self.model._modules.get("avgpool")

    def forward(self, x):
        my_embedding = torch.zeros(1, 512, 1, 1)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        h = self.layer.register_forward_hook(copy_data)
        prediction = self.model(x)

        h.remove()

        # return self.model(x)
        # return torch.zeros(1, 512, 1, 1).view(-1, 512).squeeze(0)
        # return my_embedding.view(-1, 512).squeeze(0)
        return torch.tensor(my_embedding.numpy()[0, :, 0, 0]), prediction.squeeze(0)


if __name__ == "__main__":
    example_input = torch.rand(1, 3, 224, 224)

    img2vec = Img2Vec()
    emb, pred = img2vec(example_input)
    # print(emb)

    traced_model = torch.jit.trace(img2vec, example_input)
    out, pred = traced_model(example_input)
    # print(out)

    # input_tensor = ct.TensorType(name="my_input", shape=example_input.shape)
    # cml_model = ct.convert(traced_model, inputs=[input_tensor], source="pytorch")

    cml_model = ct.convert(traced_model,
                           inputs=[ct.ImageType(name="image", color_layout="RGB", scale=1.0/255.0/0.226,
                                                bias=(-0.485/0.226, -0.456/0.226, -0.406/0.226),
                                                shape=example_input.shape)],
                           # outputs=[ct.TensorType(name="emb", shape=(512,), dtype=np.float16),
                           #          ct.TensorType(name="pred", shape=(1000,), dtype=np.float16)
                           #          ],
                           source="pytorch"
                           )

    # cml_model = ct.convert(traced_model, inputs=[ct.ImageType(shape=example_input.shape)])
    # print(cml_model)

    cml_model.save("img2vec.mlmodel")

    IS_MACOS = sys.platform == "darwin"

    test_img = Image.open("./LW3EZTS_0001_1.webp")
    test_img = test_img.resize((224, 224))

    if IS_MACOS:
        loaded_model = ct.models.MLModel("img2vec.mlmodel")
        pred_output = loaded_model.predict({"image": test_img})
        print("=" * 50)
        print(pred_output)

        for k, v in pred_output.items():
            print(k, v.shape)
