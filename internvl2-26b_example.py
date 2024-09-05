import numpy as np
import os
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

checkpoint = 'OpenGVLab/InternVL2-26B'

# Path to the cached model directory
local_checkpoint_path = f'/data/model/{checkpoint}'

model = AutoModel.from_pretrained(
    checkpoint,
    torch_dtype=torch.float16,
    device_map="auto",
    # torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    # load_in_8bit=True,
    ).eval()#.cuda()
model.save_pretrained(local_checkpoint_path)

# Ensure the offload folder exists
offload_folder = "/data/offload"
os.makedirs(offload_folder, exist_ok=True)

with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True)

# Load the checkpoint with CPU offloading
model = load_checkpoint_and_dispatch(
    model,
    checkpoint=local_checkpoint_path,
    device_map="auto",  # Automatically offload layers to CPU/GPU as needed
    offload_folder=offload_folder,  # Folder where the offloaded layers will be stored
    offload_state_dict=True,  # Offload the state dict to CPU
    dtype="float16"  # Use mixed precision (fp16) to save memory
)

'''
model = AutoModel.from_pretrained(
    checkpoint,
    torch_dtype=torch.float16,
    device_map="auto",
    # torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    # load_in_8bit=True,
    ).eval()#.cuda()
'''

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

    # set the max number of tiles in `max_num`
    max_tiles = 6
    pixel_values = load_image(img_path, max_num=max_tiles).to(torch.float16).cuda() # torch.bfloat16

    generation_config = dict(
        num_beams=1,
        max_new_tokens=1024,
        do_sample=False,
    )

    # single-image single-round conversation (单图单轮对话)
    query = f"<image>\n{query}"
    response = model.chat(tokenizer, pixel_values, query, generation_config)

    print(f"{img_path}: {response}")

    # Write the response to /data/output/{filename}.txt
    with open(output_path, "w") as f:
        f.write(response)

    '''
    # batch inference, single image per sample (单图批处理)
    pixel_values1 = load_image('./examples/image1.jpg', max_num=6).to(torch.bfloat16).cuda()
    pixel_values2 = load_image('./examples/image2.jpg', max_num=6).to(torch.bfloat16).cuda()
    num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
    pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

    questions = ['<image>\nDescribe the image in detail.'] * len(num_patches_list)
    responses = model.batch_chat(tokenizer, pixel_values,
                                num_patches_list=num_patches_list,
                                questions=questions,
                                generation_config=generation_config)
    for question, response in zip(questions, responses):
        print(f'User: {question}')
        print(f'Assistant: {response}')
    '''

    '''
    # video multi-round conversation (视频多轮对话)
    def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])
        return frame_indices

    def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        pixel_values_list, num_patches_list = [], []
        transform = build_transform(input_size=input_size)
        frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list


    video_path = './examples/red-panda.mp4'
    # pixel_values, num_patches_list = load_video(video_path, num_segments=32, max_num=1)
    pixel_values, num_patches_list = load_video(video_path, num_segments=8, max_num=1)
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    question = video_prefix + 'What is the red panda doing?'
    # Frame1: <image>\nFrame2: <image>\n...\nFrame31: <image>\n{question}
    response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                num_patches_list=num_patches_list,
                                history=None, return_history=True)
    print(f'User: {question}')
    print(f'Assistant: {response}')

    question = 'Describe this video in detail. Don\'t repeat.'
    response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                num_patches_list=num_patches_list,
                                history=history, return_history=True)
    print(f'User: {question}')
    print(f'Assistant: {response}')
    '''