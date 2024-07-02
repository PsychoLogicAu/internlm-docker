import os
import torch
import auto_gptq
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from auto_gptq.modeling import BaseGPTQForCausalLM

auto_gptq.modeling._base.SUPPORTED_MODELS = ["internlm"]
torch.set_grad_enabled(False)


class InternLMXComposer2QForCausalLM(BaseGPTQForCausalLM):
    layers_block_name = "model.layers"
    outside_layer_modules = [
        'vit', 'vision_proj', 'model.tok_embeddings', 'model.norm', 'output',
    ]
    inside_layer_modules = [
        ["attention.wqkv.linear"],
        ["attention.wo.linear"],
        ["feed_forward.w1.linear", "feed_forward.w3.linear"],
        ["feed_forward.w2.linear"],
    ]

# init model and tokenizer
model = InternLMXComposer2QForCausalLM.from_quantized(
    'internlm/internlm-xcomposer2-7b-4bit', trust_remote_code=True, device="cuda:0").eval()
tokenizer = AutoTokenizer.from_pretrained(
    'internlm/internlm-xcomposer2-7b-4bit', trust_remote_code=True)

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

    images = []
    image = Image.open(img_path).convert("RGB")
    image = model.vis_processor(image)
    images.append(image)
    image = torch.stack(images)
    with torch.cuda.amp.autocast():
        response, history = model.chat(
            tokenizer, query=query, image=image, history=[], do_sample=False)
    print(f"{img_path}: {response}")

    # Write the response to /data/output/{filename}.txt
    with open(output_path, "w") as f:
        f.write(response)

# My Favorite Animal: The Panda
# The panda, also known as the giant panda, is one of the most beloved animals in the world. These adorable creatures are native to China and can be found in the wild in a few select locations, but they are more commonly seen in captivity at zoos or wildlife reserves.
# Pandas have a distinct black-and-white coloration that makes them instantly recognizable. They are known for their love of bamboo, which they eat almost exclusively. In fact, pandas spend up to 14 hours a day eating, with the majority of their diet consisting of bamboo. Despite this seemingly unbalanced diet, pandas are actually quite healthy and have a low body fat percentage, thanks to their ability to digest bamboo efficiently.
# In addition to their unique eating habits, pandas are also known for their playful personalities. They are intelligent and curious creatures, often engaging in activities like playing with toys or climbing trees. However, they do not typically exhibit these behaviors in the wild, where they are solitary creatures who prefer to spend their time alone.
# One of the biggest threats to the panda's survival is habitat loss due to deforestation. As a result, many pandas now live in captivity, where they are cared for by dedicated staff and provided with enrichment opportunities to keep them engaged and stimulated. While it is important to protect these animals from extinction, it is also crucial to remember that they are still wild creatures and should be treated with respect and care.
# Overall, the panda is an amazing animal that has captured the hearts of people around the world. Whether you see them in the wild or in captivity, there is no denying the charm and allure of these gentle giants.
