
import os
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

# int4 quantized model
checkpoint = 'openbmb/MiniCPM-Llama3-V-2_5-int4'
# checkpoint = 'openbmb/MiniCPM-Llama3-V-2_5'
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,  torch_dtype=torch.float16, trust_remote_code=True) #device_map="auto",
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

# Create the directory /data/output/ if required
if not os.path.exists("/data/output"):
    os.makedirs("/data/output")

for img_path in img_path_list:
    output_path = f"/data/output/{img_path.split('/')[-1].split('.')[0]}.txt"

    # If the output file already exists, skip the image
    if os.path.exists(output_path):
        continue

    image = Image.open(img_path).convert('RGB')
    msgs = [{'role': 'user', 'content': query}]

    response = model.chat(
        image=image,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True,  # if sampling=False, beam_search will be used by default
        temperature=0.7,
        # system_prompt='' # pass system_prompt if needed
    )

    print(f"{img_path}: {response}")

    # Write the response to /data/output/{filename}.txt
    with open(output_path, "w") as f:
        f.write(response)
