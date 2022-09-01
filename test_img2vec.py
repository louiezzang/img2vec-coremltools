import sys

import coremltools as ct
from PIL import Image


IS_MACOS = sys.platform == "darwin"

test_img = Image.open("sample_img.webp")
test_img = test_img.resize((224, 224))

if IS_MACOS:
    loaded_model = ct.models.MLModel("img2vec.mlmodel")
    pred_output = loaded_model.predict({"image": test_img})
    print("=" * 50)
    for k, v in pred_output.items():
        print(f"{k}: {v}")
        print("-" * 50)
