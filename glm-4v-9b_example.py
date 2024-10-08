import os
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"
checkpoint = "THUDM/glm-4v-9b"
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)

# load query string from the text file /data/query.txt
with open("/data/query.txt", "r") as f:
    query = f.read()

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
    inputs = tokenizer.apply_chat_template([{"role": "user", "image": image, "content": query}],
                                           add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                           return_dict=True)  # chat mode

    inputs = inputs.to(device)

    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0])
        print(f"{img_path}: {response}")

        # Write the response to /data/output/{filename}.txt
        with open(output_path, "w") as f:
            f.write(response)
