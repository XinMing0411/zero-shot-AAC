from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import os
from tqdm import tqdm, trange
import torch
from custom_types import *
from train import ClipCocoDataset, ClipCaptionModel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Callable, Type, Iterator, Dict, Set, Optional, Any, Sized
from enum import Enum
import datetime
import torch.nn.functional as F


IS_WINDOWS = sys.platform == 'win32'
get_trace = getattr(sys, 'gettrace', None)
DEBUG = get_trace is not None and get_trace() is not None

# if DEBUG:
#     seed = 99
#     torch.manual_seed(seed)
    # np.random.seed(seed)

# N = type(None)
# V = np.array
# ARRAY = np.ndarray
# ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
# VS = Union[Tuple[V, ...], List[V]]
# VN = Union[V, N]
# VNS = Union[VS, N]
# T = torch.Tensor
# TS = Union[Tuple[T, ...], List[T]]
# TN = Optional[T]
# TNS = Union[Tuple[TN, ...], List[TN]]
# TSN = Optional[TS]
# TA = Union[T, ARRAY]


# D = torch.device
# CPU = torch.device('cpu')


# def get_device(device_id: int) -> D:
#     if not torch.cuda.is_available():
#         return CPU
#     device_id = min(torch.cuda.device_count() - 1, device_id)
#     return torch.device(f'cuda:{device_id}')


# CUDA = get_device

# from pycocoevalcap.cider.cider import Cider
# from pycocotools.coco import COCO
# from PIL import Image
# import matplotlib.pyplot as plt


# def image_to_display(img) -> ARRAY:
#     if type(img) is str:
#         img = Image.open(str(img))
#     if type(img) is not V:
#         img = V(img)
#     return img


# def imshow(img, title: Optional[str] = None):
#     img = image_to_display(img)
#     plt.imshow(img)
#     plt.axis("off")
#     if title is not None:
#         plt.title(title)
#     plt.show()
#     plt.close('all')


class ClipCocoDatasetWithImages(ClipCocoDataset):

    def __getitem__(self, item):
        tokens, mask, prefix, caption = super(ClipCocoDatasetWithImages, self).__getitem__(item)
        # item = self.get_ret_item(item)
        image_id = int(self.image_ids[item])
        image_path = f"./data/coco/train2014/COCO_train2014_{image_id:012d}.jpg"
        if not os.path.isfile(image_path):
            image_path = f"./data/coco/val2014/COCO_val2014_{image_id:012d}.jpg"
        return tokens, mask, prefix, caption, image_path

    def __init__(self,  data_path: str,  prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix: bool = False):
        super(ClipCocoDatasetWithImages, self).__init__(data_path, prefix_length, gpt2_type,
                                                        normalize_prefix=normalize_prefix)
        self.image_root = []
        self.images_names = []


def generate_beam(model: ClipCaptionModel, tokenizer, beam_size: int = 5, prompt=None, embed=None,
                  entry_length=67, temperature=1., stop_token: str = '.'):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]

    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated,output_hidden_states=True)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]                
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy() 
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token: str = '.',
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated,output_hidden_states=True)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item() or next_token.item() == 764:
                    break


            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]


def add_embedding_from_text(add_in: str, prefix_embed: T, tokenizer, model: ClipCaptionModel, where: int):
    device = prefix_embed.device
    tokens = torch.tensor(tokenizer.encode(add_in)).to(device)
    token_embedding = model.get_embedding(tokens).unsqueeze(0)
    if where == -1 or where == prefix_embed.shape[1]:
        prefix_list = (prefix_embed, token_embedding)
    elif where == 0:
        prefix_list = (token_embedding, prefix_embed)
    else:
        prefix_list = (prefix_embed[:, :where], token_embedding, prefix_embed[:, where:])
    prefix_new = torch.cat(prefix_list, dim=1)
    return prefix_new


def generate_text(prefix_embed: T, tokenizer, model: ClipCaptionModel, use_beam: bool) -> str:
    if use_beam:
        generated_text = generate_beam(model, tokenizer, embed=prefix_embed, beam_size=5)[0]
    else:
        generated_text = generate2(model, tokenizer, embed=prefix_embed)
    return generated_text


def re_caption(add_in : str, prefix_embed: T, tokenizer, model: ClipCaptionModel,
               where: int, use_beam: bool = True) -> str:
    prefix_new = add_embedding_from_text(add_in, prefix_embed, tokenizer, model, where)
    return generate_text(prefix_new, tokenizer, model, use_beam)


def remove_token(prefix_embed: T, tokenizer, model: ClipCaptionModel, embeddings,
                 where: List[int], use_beam: bool = True):
    prefix_new = [prefix_embed[:, i] for i in range(prefix_embed.shape[1]) if i not in where]
    prefix_new = torch.stack(prefix_new, dim=1)
    sim = torch.einsum('pd,nd->pn', nnf.normalize(prefix_new[0], 2, 1), embeddings)
    sim_arg = sim.argmax(-1)
    prefix_sent = tokenizer.decode(sim_arg)
    generated_text = generate_text(prefix_new, tokenizer, model, use_beam=use_beam)
    return generated_text, prefix_sent


def try_all_places(add_in : str, prefix_embed: T, tokenizer, model: ClipCaptionModel, use_beam: bool = True) -> List[str]:
    out = []
    for i in range(prefix_embed.shape[1]):
        out.append(re_caption(add_in, prefix_embed, tokenizer, model, i, use_beam))
    return out


def get_prefix_tokens(prefix_embed, embeddings, tokenizer) -> str:
    sim = torch.einsum('pd,nd->pn', nnf.normalize(prefix_embed[0], 2, 1), embeddings)
    sim_arg = sim.argmax(-1).cpu().numpy()
    prefix_tokens = [tokenizer.decode(token) for token in sim_arg]
    prefix_sentence = "".join(prefix_tokens),
    # prefix_tokens = tokenizer.decode(sim_arg)
    # print(prefix_sentence)
    return prefix_sentence


def train(dataset: ClipCocoDataset, model: ClipCaptionModel, batch_size: int, device):
    model = model.to(device)
    model.eval()
    tokenizer = dataset.tokenizer
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    embeddings = model.gpt.get_input_embeddings().weight.data
    embeddings = nnf.normalize(embeddings, 2, 1)
    for idx, (tokens, mask, prefix, caption, images) in tqdm(enumerate(train_dataloader)):
        tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
        for jj in range(1, tokens.size(0)):
            found = False
            for item in ("19906", "320200", "341061", "400728", "444467"):
                if item in images[jj - 1]:
                    found = True
                    break
            if not found:
                continue
            prefix_embed = model.clip_project(prefix[jj - 1:jj]).reshape(1, dataset.prefix_length, -1)
            prefix_sent = get_prefix_tokens(prefix_embed, embeddings, tokenizer)
            try:
                generated_text_beam = generate_beam(model, tokenizer, embed=prefix_embed, beam_size=5)
                generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)
            except BaseException:
                continue
                print("probability tensor contains either `inf`, `nan` or element < 0")
            if DEBUG:
                image_caption = f"\nGT: {caption[jj-1]}\n\nClipCap: {generated_text_prefix}"
                print(prefix_sent)
                # imshow(images[jj - 1], image_caption)
            else:
                print("-=(%0d)=-" % jj)
                print("Caption:")
                print(caption[jj-1])
                print(">>>>> Generate from prefix")
                print(generated_text_beam[0])
        # user_input = input("\nto exit type x\n")
        # if user_input == "x":
        #     break
    return 0


def main():
    batch_size = 5
    num_epochs = 10
    prefix_length = 10
    model = ClipCaptionModel(prefix_length)
    device = CPU
    model.load_state_dict(torch.load("./checkpoints/oscar_split-007.pt", map_location=device))
    dataset = ClipCocoDatasetWithImages("./data/coco/oscar_split_train.pkl", prefix_length, normalize_prefix=False)

    # generated_text2 = generate_beam(model, GPT2Tokenizer.from_pretrained('gpt2'), prompt="Toronto Raptors")
    with torch.no_grad():
        train(dataset, model, batch_size, device)





################ New Sampling #########################
@torch.no_grad()
def magic_search(model, tokenizer, audio_embeds, clap,  input_ids=None,prompt=None, embed=None, beam_width=15, alpha=0.1, decoding_len=35, beta=0.2, 
    clip_text_max_len=60,stop_token= '.'):#, add_token_level_score=False):
    clap.eval()
    device = next(model.parameters()).device
    stop_token_index = tokenizer.encode(stop_token)[0]
    if embed is not None:
        input_ids = embed
    else:
        if tokens is None:
            tokens = torch.tensor(tokenizer.encode(prompt))
            tokens = tokens.unsqueeze(0).to(device)
            input_ids = model.gpt.transformer.wte(tokens)

    prefix_len = input_ids.size()[1]
    past_key_values, last_hidden_states, logits = None, None, None
    generated = [item for item in input_ids.tolist()]
    input_ids_for_class = None

    #Extract Audio embeddings
    # image_embeds = clip.compute_image_representation_from_image_instance(image_instance)

    start_time = datetime.datetime.now()

    # the maximum supported length of generation for SimCTG is 256
    # to support longer generated length, you can re-train the SimCTG model with longer sequences
    decoding_len = decoding_len - prefix_len
    for step in range(decoding_len):
        input_ids, past_key_values, last_hidden_states, logits, input_ids_for_class = \
        PlugAndPlayContrastiveDecodingOneStepFast(
            model, 
            input_ids, 
            prefix_len,
            beam_width, 
            alpha, 
            beta, 
            tokenizer,
            audio_embeds, 
            clap, 
            clip_text_max_len,
            past_key_values,
            last_hidden_states,
            logits,
            first_step=step==0,
            input_ids_for_class=input_ids_for_class,
        )
        if  input_ids.item() == stop_token_index:
            break
    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time)
    execution_time = time_diff.total_seconds() * 1000

    output_text = tokenizer.decode(input_ids_for_class[0])
    return output_text
    

def PlugAndPlayContrastiveDecodingOneStepFast(model, input_ids, prefix_len, beam_width, alpha, beta, 
    simctg_tokenizer, audio_embeds, clap, clip_text_max_len, past_key_values, last_hidden_states, 
    logit_for_next_step, first_step=False, input_ids_for_class=None):#, add_token_level_score=False):
    '''
        model: the generation model, e.g., gpt2
        input_ids: 1 x seqlen
    '''

    if first_step:
        output = model.gpt(inputs_embeds=input_ids, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
        past_key_values = output.past_key_values
        last_hidden_states = output.hidden_states[-1]    # [B, S, E]
        logit_for_next_step = output.logits[:, -1, :]    # [B, V]
    bsz, seqlen, embed_dim = last_hidden_states.size()
    print(last_hidden_states.shape)
    next_probs = F.softmax(logit_for_next_step, dim = -1)
    _, top_k_ids = torch.topk(logit_for_next_step, dim = -1, k = beam_width)
    top_k_probs = torch.gather(next_probs, dim = 1, index=top_k_ids)

    # compute the new hidden
    past_key_values = enlarge_past_key_values(past_key_values, beam_width)
    output = model.gpt(
        input_ids=top_k_ids.view(-1, 1) ,
        attention_mask=torch.ones_like(top_k_ids.view(-1, 1)),
        past_key_values=past_key_values,
        output_hidden_states=True,
        use_cache=True,
    )
    # print(output.hidden_states[0].shape)
    past_key_values = output.past_key_values
    logits = output.logits[:, -1, :]
    next_hidden = output.hidden_states[-1]
    print(next_hidden.shape)
    context_hidden = last_hidden_states.unsqueeze(1).expand(-1, beam_width, -1, -1).reshape(bsz*beam_width, seqlen, embed_dim)
    
    if input_ids_for_class ==None:
        input_ids_for_class_ = top_k_ids.view(-1, 1).clone()
    else:
        input_ids_for_class_ = torch.cat([
            input_ids_for_class.unsqueeze(1).expand(-1, beam_width, -1).reshape(bsz*beam_width, -1),
            top_k_ids.view(-1, 1)
            ], dim=-1
        )

    batch_text_list = []

    for one_input_id in input_ids_for_class_:
        one_text = simctg_tokenizer.decode(one_input_id) 
        batch_text_list.append(one_text)
    batch_score = compute_audio_text_similarity_via_raw_text(clap,audio_embeds, batch_text_list)

    selected_idx,_ = plug_and_play_fast_ranking(
        context_hidden, 
        next_hidden, 
        top_k_ids, 
        top_k_probs, 
        alpha, 
        beta, 
        batch_score,
        beam_width,
    )       

    # prepare for the next step
    next_id = top_k_ids[range(len(top_k_ids)), selected_idx].unsqueeze(-1)
    next_hidden = torch.stack(torch.split(next_hidden.squeeze(dim=1), beam_width))
    next_hidden = next_hidden[range(bsz), selected_idx, :]
    last_hidden_states = torch.cat([last_hidden_states, next_hidden.unsqueeze(1)], dim=1)
    past_key_values = select_past_key_values(past_key_values, beam_width, selected_idx)
    logits = torch.stack(torch.split(logits, beam_width))[range(bsz), selected_idx, :]
    if input_ids_for_class ==None:
        input_ids_for_class = next_id
    else:
        input_ids_for_class = torch.cat([input_ids_for_class, next_id], dim=-1)
    return next_id, past_key_values, last_hidden_states, logits, input_ids_for_class

def enlarge_past_key_values(past_key_values, beam_width):
    new_key_values = []
    for layer in past_key_values:
        items = []
        for item in layer:
            bsz, num_head, seq_len, esz = item.size()
            item = item.unsqueeze(1).expand(-1, beam_width, -1, -1, -1).reshape(bsz*beam_width, num_head, seq_len, esz)    # [bsz*beam, num_head, seq_len, esz]
            items.append(item)
        new_key_values.append(items)
    return new_key_values

def select_past_key_values(past_key_values, beam_width, selected_idx):
    '''select_idx: [B]'''
    new_key_values = []
    for layer in past_key_values:
        items = []
        for item in layer:
            bsz_and_beam, num_head, seq_len, esz = item.size()
            bsz = int(bsz_and_beam//beam_width)
            item = torch.stack(torch.split(item, beam_width, dim=0))    # [B, K, num_head, seq_len, esz] 
            item = item[range(bsz), selected_idx, :, :, :]   # [B, num_head, seq_len, esz]
            items.append(item)
        new_key_values.append(items)
    return new_key_values

# ========== fast plug and play version ========= #
def plug_and_play_fast_ranking(
    context_hidden, 
    next_hidden, 
    next_top_k_ids, 
    next_top_k_probs, 
    alpha, 
    beta, 
    batch_class_score,
    beam_width,
    prefix_length = 1
):
    '''
        context_hidden: beam_width x context_len x embed_dim
        next_hidden: beam_width x 1 x embed_dim
        next_top_k_ids: beam_width x 1
        batch_class_score: beam_width x 1
    '''
    _, context_len, embed_dim = context_hidden.size()
    # print(context_hidden[:,prefix_length-1:,:])
    norm_context_hidden = context_hidden[:,prefix_length-1:,:] / context_hidden[:,prefix_length-1:,:].norm(dim=2, keepdim=True)
    # print(norm_context_hidden.shape)
    norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)
    cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1,2)).squeeze(-1)
    scores, _ = torch.max(cosine_matrix, dim = -1)
    # print(scores.shape)
    # scores = torch.stack(torch.split(scores.view(-1), beam_width))
    # scores = scores.softmax(dim=-1).log()
    # print(batch_class_score.shape)
    # batch_class_score = torch.stack(torch.split(batch_class_score.view(-1), beam_width))
    # # print(batch_class_score.shape)
    # batch_class_score = batch_class_score.softmax(dim=-1).log().view(-1)
    

    next_top_k_probs = next_top_k_probs.view(-1)
    scores = (1.0 - alpha) * next_top_k_probs - alpha * scores + beta * batch_class_score.view(-1)
    scores = torch.stack(torch.split(scores, beam_width))
    selected_idx = scores.max(dim=-1)[1]
    return selected_idx,scores

def compute_audio_text_similarity_via_embeddings(clap,audio_embeds, text_embeds):
    '''
        image_embeds: 1 x embed_dim
        text_embeds: len(text_list) x embed_dim
    '''
    audio_embeds = audio_embeds / audio_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    logits_per_text = torch.matmul(text_embeds, audio_embeds.t()) / clap.temp 
    logits_per_audio = logits_per_text.T
    # return logits_per_audio
    # print(logits_per_audio.shape)
    return logits_per_audio.softmax(dim=1).log() # 1 x len(text_list)

def compute_audio_text_similarity_via_raw_text(clap,audio_embeds, text_list):
    text_embeds = clap.encode_text(text_list)
    return compute_audio_text_similarity_via_embeddings(clap,audio_embeds, text_embeds)

def ComputeMagicScore(model, generated,magic_width,input_token,tokenizer,clap,audio_embeds,alpha, beta,prefix_length=1):
    outputs = model.gpt(inputs_embeds=generated,output_hidden_states=True)
    past_key_values = outputs.past_key_values
    last_hidden_states = outputs.hidden_states[-1]
    bsz, seqlen, embed_dim = last_hidden_states.size()

    #根据输出magic选出候选tokens
    _, top_k_ids = torch.topk(outputs.logits[:, -1, :], dim = -1, k = magic_width)
    next_probs = F.softmax(outputs.logits[:, -1, :], dim = -1).log()
    top_k_probs = torch.gather(next_probs, dim = 1, index=top_k_ids)

    #根据候选tokens计算下一时刻的hidden stats, 计算term 2
    past_key_values = enlarge_past_key_values(past_key_values, magic_width)
    next_token_embed = model.gpt.transformer.wte(top_k_ids.squeeze()).view(bsz*magic_width, 1, -1)
    next_output = model.gpt(inputs_embeds=next_token_embed, attention_mask=torch.ones(bsz*magic_width,1).to(top_k_probs.device), past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
    next_hidden = next_output.hidden_states[-1]
    context_hidden = last_hidden_states.unsqueeze(1).expand(-1, magic_width, -1, -1).reshape(bsz*magic_width, seqlen, embed_dim)

    #计算CLAP Score
    if input_token ==None:
        input_token = top_k_ids.view(-1, 1).clone()
        # prefix_length = generated.shape[1]
        # print(prefix_length)
    else:
        input_token = torch.cat([
            input_token.unsqueeze(1).expand(-1, magic_width, -1).reshape(bsz*magic_width, -1),
            top_k_ids.view(-1, 1)
            ], dim=-1
        )
    batch_text_list = []
    for one_input_id in input_token:
        one_text = tokenizer.decode(one_input_id) 
        batch_text_list.append(one_text)
    batch_score = compute_audio_text_similarity_via_raw_text(clap,audio_embeds, batch_text_list)

    #计算最终Score
    _,score = plug_and_play_fast_ranking(
        context_hidden, 
        next_hidden, 
        top_k_ids, 
        top_k_probs, 
        alpha, 
        beta, 
        batch_score,
        magic_width,prefix_length)

    return score.unsqueeze(1), top_k_ids


def generate_beam_magic(model: ClipCaptionModel,clap, tokenizer,audio_embeds,beam_size: int = 5, prompt=None, embed=None,
                  entry_length=20, temperature=1., stop_token: str = '.', magic_width=25,alpha=0.1, beta=0.2):

    # print(alpha)
    # print(beta)
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    # print("--------------")
    # print(stop_token_index)
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        # prefix_length = generated.shape[1]
        prefix_length = 1
        for i in range(entry_length):
            logits, logits_ids= ComputeMagicScore(model, generated,magic_width,tokens,tokenizer,clap,audio_embeds,alpha, beta,prefix_length)
            # print(logits_ids)
            logits = logits[:, -1, :]/ (temperature if temperature > 0 else 1.0)
            # logits = logits.softmax(-1).log()
            if scores is None:
                scores, index = logits.topk(beam_size, -1)
                # print(index.shape)
                next_tokens = logits_ids[torch.arange(logits_ids.size(0)).unsqueeze(1), index]
                # next_tokens
                # next_tokens = logits_ids[index]
                # print(next_tokens.shape)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
                # print(scores.shape)
            else:
                logits_ids = logits_ids.view(-1)
                logits[is_stopped] = -float(np.inf)
                # print(logits.shape)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_index = scores_sum_average.view(-1).topk(beam_size, -1)
                # print(next_tokens)
                next_tokens_source = next_index // scores_sum.shape[1]
                # print(next_tokens_source)
                seq_lengths = seq_lengths[next_tokens_source]

                # next_tokens = next_index % scores_sum.shape[1]
                # print(next_tokens)
                # print(next_tokens)
                # logits_ids(logits_ids)
                next_tokens = logits_ids[next_index]
                # print(next_index)
                # print(next_tokens)
                # next_tokens = logits_ids[next_tokens]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                # print(tokens.shape)
                # print(next_tokens.shape)
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                # print(scores)
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            # print(is_stopped)
            # print(tokens)
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy() 
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


if __name__ == '__main__':
    exit(main())
