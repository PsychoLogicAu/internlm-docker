from share4v.model.builder import load_pretrained_model
from share4v.mm_utils import get_model_name_from_path
from share4v.eval.run_share4v import eval_model

model_path = "Lin-Chen/ShareGPT4V-7B"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

model_path = "Lin-Chen/ShareGPT4V-7B"

prompt = "What is the most common catchphrase of the character on the right?"
image_file = "examples/breaking_bad.png"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

eval_model(args)
