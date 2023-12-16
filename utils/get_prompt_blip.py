import argparse
import os

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BlipForQuestionAnswering,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

mylabel2ids = {
    'hat': [1],
    'sunglasses': [3],
    'upper-clothes': [4],
    #'hair': [2],
    'skirt': [5],
    'pants': [6],
    'dress': [7],
    'belt': [8],
    'shoes': [9, 10],
    'bag': [16],
    'scarf': [17]
}


def is_necessary(g, garments):
    if 'dress' not in garments:
        if g == 'upper-clothes':
            return True
        if (g == 'pants') and ('skirt' not in g):
            return True
        if (g == 'skirt') and ('pants' not in g):
            return True
    return False


def ask(q, img_path):
    print('Question: {}'.format(q))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    blip_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
    blip_model.to(device)

    inputs = blip_processor(
        images=[Image.open(img_path)],
        text=q,
        padding=True,
        return_tensors="pt",
    ).to(device, torch.float16)

    generated_ids = blip_model.generate(**inputs, max_new_tokens=5)
    answer = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print('Answer:', answer)
    return answer


def clean_prompt(prompt):
    while ('  ' in prompt):
        prompt = prompt.replace('  ', ' ')
    while (' ,' in prompt):
        prompt = prompt.replace(' ,', ',')
    return prompt


def get_prompt_segments(img_path, feature_extractor, model):
    image = open(img_path, 'rb')

    gender = ask('Is this person a man or a woman', img_path)
    if 'woman' in gender:
        gender = 'woman'
    elif 'man' in gender:
        gender = 'man'
    else:
        gender = 'person'
    prompt = 'a sks {}'.format(gender)
    garments = get_garments(image, feature_extractor, model)
    haircolor = ask('What is the hair color of this person?', img_path)
    hairstyle = ask('What is the hair style of this person?', img_path)
    face = ask('Describe the facial appearance of this person.', img_path)
    prompt = prompt + ', {} {} hair, {}'.format(haircolor, hairstyle, face)
    for g in garments:
        has_g = is_necessary(g, garments
                            ) or ('yes' in ask('Is this person wearing {}?'.format(g), img_path))
        if has_g:
            kind = ask('What {} is the person wearing?'.format(g), img_path)
            if (g in kind) or (g == 'upper-clothes'):
                g = ''
            color = ask('What is the color of the {} {}?'.format(kind, g), img_path)
            style = ask('What is the style of the {} {}?'.format(kind, g), img_path)
            if style in kind or style in g:
                style = ''
            if color in kind or color in g:
                color = ''
            prompt = prompt + ', sks {} {} {} {}'.format(color, style, kind, g)
    has_beard = ask('Do this person has facial hair?', img_path)
    if 'yes' in has_beard:
        beard = ask('How is the facial hair of this person?', img_path)
        if beard != 'none':
            prompt = prompt + ', {} beard'.format(beard)
    pose = ask('Describe the pose of this person.', img_path)
    prompt = prompt + ', {}'.format(pose)
    prompt = clean_prompt(prompt)
    return prompt, gender


def get_garments(img_path, feature_extractor, model):
    image = np.array(Image.open(img_path))
    alpha = image[..., 3:] > 250
    image = (image[..., :3] * alpha).astype(np.uint8)
    inputs = feature_extractor(images=image, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    image = np.array(Image.open(img_path).resize((128, 128)))
    alpha = image[..., 3:] > 250
    image = image[..., :3] * alpha
    seg = outputs.logits[0].argmax(dim=0)
    result = dict()
    for label in mylabel2ids:
        label_mask = np.zeros_like(alpha[..., 0])
        for id in mylabel2ids[label]:
            label_mask |= (seg == id).cpu().numpy()
        label_mask &= alpha[..., 0]
        #print(label, label_mask.sum())
        if label_mask.sum() == 0:
            continue
        result[label] = label_mask
    return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, required=True, help="input image")
    parser.add_argument('--out_path', type=str, required=True, help="output path")
    opt = parser.parse_args()

    print(f'[INFO] Generating text prompt for {opt.in_dir}...')

    model = SegformerForSemanticSegmentation.from_pretrained(
        "matei-dorian/segformer-b5-finetuned-human-parsing"
    ).cuda()
    feature_extractor = SegformerImageProcessor.from_pretrained(
        "matei-dorian/segformer-b5-finetuned-human-parsing"
    )

    with open(opt.out_path, 'w') as f:
        for file in os.listdir(opt.in_dir):
            if file.endswith(".png"):
                img_path = os.path.join(opt.in_dir, file)
                prompt, gender = get_prompt_segments(img_path, feature_extractor, model)
                print(f'[INFO] generated prompt: {prompt}, estimated category: {gender}')
                f.write(f'{prompt}|{gender} \n')
