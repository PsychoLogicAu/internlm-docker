# test.py
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import os

# checkpoint = 'openbmb/MiniCPM-V-2_6'
checkpoint = 'openbmb/MiniCPM-V-2_6-int4'
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True,
                                  # sdpa or flash_attention_2, no eager
                                  attn_implementation='sdpa',
                                  torch_dtype=torch.bfloat16,
                                  # load_in_8bit=True
                                  )
model = model.eval()#.cuda()
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)


# load query string from the text file /data/query.txt
with open("/data/query.txt", "r") as f:
    query = f"{f.read()}"

print("Query:", query)

image_extensions = ["jpeg", "jpg", "JPG", "png", "bmp"]
# iterate over the image files in "/data/images" directory, and add the image path to the img_path_list
img_path_list = []
for root, dirs, files in os.walk("/data/images"):
    for file in files:
        if file.split(".")[-1] in image_extensions:
            img_path_list.append(os.path.join(root, file))

output_dir = f"/data/output/{checkpoint}"
# Create the output directory if required
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for img_path in img_path_list:
    output_path = f"{output_dir}/{img_path.split('/')[-1].split('.')[0]}.txt"

    # If the output file already exists, skip the image
    if os.path.exists(output_path):
        continue

    image = Image.open(img_path).convert('RGB')

    msgs = [{'role': 'user', 'content': [image, query]}]

    response = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer
    )

    print(f"{img_path}: {response}")

    # Write the response to the output file
    with open(output_path, "w") as f:
        f.write(response)
