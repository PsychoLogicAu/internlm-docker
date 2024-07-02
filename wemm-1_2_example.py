import os
import torch
from PIL import Image
from transformers import AutoModel, AutoModelForCausalLM, GenerationConfig

checkpoint = 'feipengma/WeMM'  # the path to the model
model = AutoModel.from_pretrained(
    checkpoint, torch_dtype=torch.bfloat16, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     checkpoint, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
# wemm.cuda()
# wemm.eval()

# load query string from the text file /data/query.txt, prepend it with "<ImageHere>" which appears to be required for this model
with open("/data/query.txt", "r") as f:
    query = f"<ImageHere> {f.read()}"

print("Query:", query)


image_extensions = ["jpeg", "jpg", "JPG", "png", "bmp"]
# iterate over the image files in "/data/images" directory, and add the image path to the img_path_list
img_path_list = []
for root, dirs, files in os.walk("/data/images"):
    for file in files:
        if file.split(".")[-1] in image_extensions:
            img_path_list.append(os.path.join(root, file))

# Create the directory /data/output/ if required
if not os.path.exists("/data/output"):
    os.makedirs("/data/output")

for img_path in img_path_list:
    output_path = f"/data/output/{img_path.split('/')[-1].split('.')[0]}.txt"

    # If the output file already exists, skip the image
    if os.path.exists(output_path):
        continue

    pred = model.mm_generate(img_path, query)
