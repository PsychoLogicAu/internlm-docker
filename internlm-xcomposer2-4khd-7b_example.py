import os
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForCausalLM

torch.set_grad_enabled(False)

# init model and tokenizer
# model = AutoModel.from_pretrained('internlm/internlm-xcomposer2-4khd-7b', torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
checkpoint = 'internlm/internlm-xcomposer2-4khd-7b'
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, device_map="auto", load_in_8bit=True, torch_dtype=torch.float16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

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

    # image = Image.open(img_path).convert('RGB')

    with torch.cuda.amp.autocast():
        response, his = model.chat(tokenizer, query=query, image=img_path, hd_num=55, history=[
        ], do_sample=False, num_beams=3)

        print(f"{img_path}: {response}")

        # Write the response to /data/output/{filename}.txt
        with open(output_path, "w") as f:
            f.write(response)

'''
###############
# First Round
###############

# query1 = '<ImageHere>Illustrate the fine details present in the image'
image = './example.webp'
with torch.cuda.amp.autocast():
    response, history = model.chat(tokenizer, query=query, image=image, hd_num=55, history=[
    ], do_sample=False, num_beams=3)

    print(response)
# The image is a vibrant and colorful infographic that showcases 7 graphic design trends that will dominate in 2021. The infographic is divided into 7 sections, each representing a different trend.
# Starting from the top, the first section focuses on "Muted Color Palettes", highlighting the use of muted colors in design.
# The second section delves into "Simple Data Visualizations", emphasizing the importance of easy-to-understand data visualizations.
# The third section introduces "Geometric Shapes Everywhere", showcasing the use of geometric shapes in design.
# The fourth section discusses "Flat Icons and Illustrations", explaining how flat icons and illustrations are being used in design.
# The fifth section is dedicated to "Classic Serif Fonts", illustrating the resurgence of classic serif fonts in design.
# The sixth section explores "Social Media Slide Decks", illustrating how slide decks are being used on social media.
# Finally, the seventh section focuses on "Text Heavy Videos", illustrating the trend of using text-heavy videos in design.
# Each section is filled with relevant images and text, providing a comprehensive overview of the 7 graphic design trends that will dominate in 2021.

###############
# Second Round
###############
query1 = 'what is the detailed explanation of the third part.'
with torch.cuda.amp.autocast():
    response, _ = model.chat(tokenizer, query=query1, image=image,
                             hd_num=55, history=his, do_sample=False, num_beams=3)
print(response)
# The third part of the infographic is about "Geometric Shapes Everywhere". It explains that last year, designers used a lot of
# flowing and abstract shapes in their designs. However, this year, they have been replaced with rigid, hard-edged geometric
# shapes and patterns. The hard edges of a geometric shape create a great contrast against muted colors.
'''
