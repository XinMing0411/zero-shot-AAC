import os
import json
import torch,math
import logging
import argparse
import numpy as np
from typing import List, Tuple, Union

from models.caption_model import *

def get_uniform_ball_noise(input_shape,device, radius=0.1):
    uniform_noise_ball = torch.randn(input_shape, device=device)  # normal distribution
    uniform_noise_sphere = torch.nn.functional.normalize(uniform_noise_ball, dim=-1)
    u = torch.rand(input_shape[0], device=device)  # unified distribution
    u = u ** (1. / input_shape[1])
    uniform_noise_ball = (uniform_noise_sphere.T * u * radius).T
    return uniform_noise_ball

def noise_injection(x, variance=0.001,uniform_noise=False, dont_norm=False):

    if variance == 0.0:
        return x
    std = math.sqrt(variance)
    if not dont_norm:
        x = torch.nn.functional.normalize(x, dim=-1)
    if uniform_noise:
        x = x + get_uniform_ball_noise(x.shape, radius=std)
    else:
        x = x + (torch.randn(x.shape) * std).to(x.device)  # todo by some conventions multivraiance noise should be devided by sqrt of dim
        
    return torch.nn.functional.normalize(x, dim=-1)

def criterion_improver(mode):
    assert mode in ("loss", "acc", "score")
    best_value = np.inf if mode == "loss" else 0

    def comparator(x, best_x):
        return x < best_x if mode == "loss" else x > best_x

    def inner(x):
        nonlocal best_value

        if comparator(x, best_value):
            best_value = x
            return True
        return False
    return inner

def genlogger(outputfile, level="INFO"):
    formatter = logging.Formatter(
        "[ %(levelname)s : %(asctime)s ] - %(message)s")
    logger = logging.getLogger(__name__ + "." + outputfile)
    logger.setLevel(getattr(logging, level))
    # Dump log to file
    filehandler = logging.FileHandler(outputfile)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    # logger.addHandler(stdhandler)
    return logger

def post_processing(captions):
    output_caption = []
    for item in captions:
        output_caption.append(str(item["caption"].lower()))

    return output_caption

def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)

def load_model(config_path: str, epoch_or_latest: Union[str, int] = '_latest'):
    with open(config_path) as f:
        config = json.load(f)
    parser = argparse.ArgumentParser()
    parser.set_defaults(**config)
    args = parser.parse_args()
    if type(epoch_or_latest) is int:
        epoch_or_latest = f"-{epoch_or_latest:03d}"
    model_path = os.path.join(args.out_dir, f"{args.prefix}{epoch_or_latest}.pt")
    if args.only_prefix:
        model = ClapCaptionPrefix(args.prefix_length)
    else:
        model = ClapCaptionModel(args.prefix_length)
    if os.path.isfile(model_path):
        print(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print(f"{model_path} is not exist")
    return model, parser

def eval_prediction(key2refs, key2pred, scorers, pretokenized=False):
    if not pretokenized:
        refs4eval = {}
        for key, refs in key2refs.items():
            refs4eval[key] = []
            for idx, ref in enumerate(refs):
                refs4eval[key].append({
                    "audio_id": key,
                    "id": idx,
                    "caption": ref
                })
            # print(refs4eval[key])

        preds4eval = {}
        for key, preds in key2pred.items():
            preds4eval[key] = []
            for idx, pred in enumerate(preds):
                preds4eval[key].append({
                    "audio_id": key,
                    "id": idx,
                    "caption": pred
                })

        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

        tokenizer = PTBTokenizer()
        key2refs = tokenizer.tokenize(refs4eval)
        key2pred = tokenizer.tokenize(preds4eval)
       
    output = {}
    for scorer in scorers:
        score, scores = scorer.compute_score(key2refs, key2pred)
        output[scorer.method()] = score
    return output

def sound_effect_choice(prefix, sound_effect_embeddings, choice_num):

    similarity = prefix.__matmul__(sound_effect_embeddings.t())
    similarity_softmax = F.softmax(similarity.detach().cpu(), dim=-1)
    num,index = torch.topk(similarity_softmax,choice_num,dim=-1)
    
    return index


def entities_process(detected_entities: List[str],  # [man, dog, park]
                    mask_probability) -> List[str]:
    process_entities = []
    for i in range(len(detected_entities)):

        detected_entity = detected_entities[i]                # processing the i-th entity
        if mask_probability != 0:
            random_prob = random.random()
            if random_prob < mask_probability:         # mask
                pass
            else:                                              # remain
                process_entities.append(detected_entity)

        else: # entities with any process
            return detected_entities
    
    return process_entities

def compose_discrete_prompts(
    tokenizer,
    process_entities: List[str],) -> torch.Tensor:

    prompt_head = 'There are'
    prompt_tail = ' in this audio.'

    if len(process_entities) == 0: # without entities
        discrete_prompt =  prompt_head + ' something' + prompt_tail
    else:
        discrete_prompt = ''
        for entity in process_entities: # gpt2 in transformer encoder ' ' + word into one token by default
            discrete_prompt += ' ' + entity + ','     # ' person, dog, park,'
        discrete_prompt = discrete_prompt[:-1]        # ' person, dog, park'
        discrete_prompt = prompt_head + discrete_prompt + prompt_tail # 'There are person, dog, park in image.'

    entities_tokens = torch.tensor(tokenizer.encode(discrete_prompt))   # (discrete_prompt_length, ) 

    return entities_tokens

def parse_entities(
    tokenizer,
    detected_entities,      # [[man, dog, park, ...], len = batch size
    mask_probability
) -> List[torch.Tensor]:
    # List[(n_seq1, ), (n_seq2, ), ...]


    process_entities = entities_process(detected_entities,mask_probability)

    return compose_discrete_prompts(tokenizer, process_entities)

def padding_captions(hard_prompts: List[torch.Tensor],
                     hard_prompts_length:List[int]):
    max_length = max(hard_prompts_length)

    out_hard_prompts = list()
    for hard_prompt in hard_prompts:
        padding = max_length - hard_prompt.shape[0]
        if padding >= 0:
            out_hard_prompts.append(torch.cat((hard_prompt, torch.zeros(padding, dtype=torch.int64) - 1)))
        elif padding < 0:
            out_hard_prompts.append(hard_prompt[:max_length])
    
    out_hard_prompts =  torch.stack(out_hard_prompts)
    mask = out_hard_prompts.ge(0)  # mask is zero where we out of sequence
    out_hard_prompts[~mask] = 0

    mask = mask.float()

    return out_hard_prompts, mask