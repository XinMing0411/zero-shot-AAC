import pickle
import sys
import os
import torch
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from transformers import GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup,AutoTokenizer
sys.path.append(os.getcwd())
from utils import *

class ClapDataset(Dataset):
    def __len__(self) -> int:
        return len(self.all_data)

    def pad_tokens(self, caption):
        if caption[-1] != '.':
            caption= caption+'.'

        tokens = torch.tensor(self.tokenizer.encode(caption), dtype=torch.int64)
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        
        tokens, mask = self.pad_tokens(self.all_data[item]["caption"])

        if self.use_audio_embedding: # Tradition Audio Captioning
            prefix = self.all_data[item]["audio_embedding"]
        elif self.use_related_text: # Zero-shot Audio Captioning
            random_index =  torch.randint(0, self.all_data[item]["related_embeddings"].size(0), (1,))
            prefix = self.all_data[item]["related_embeddings"][random_index]
        else:
            prefix =  self.all_data[item]["text_embedding"]

        if self.normalize_prefix:
            prefix = F.normalize(prefix, dim=-1)

        return tokens, mask, prefix

    def __init__(self, data_path: list,  prefix_length: int, use_related_text=False,
                 normalize_prefix=False, use_audio_embedding=False, percentage=1.0):

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        self.use_audio_embedding = use_audio_embedding
        self.use_related_text = use_related_text
        self.max_seq_len = 25
        all_data = list()

        for dp in data_path:
            with open(dp, 'rb') as f:
                while True:
                    try:
                        item = pickle.load(f)
                        if type(item) is list:
                            all_data = all_data + item
                        else:
                            if len(item['caption'].split())>=8 and len(item['caption'].split())<=20:
                                all_data.append(item)
                            else:
                                # all_data.append(item)
                                pass
                    except EOFError:
                        break
                       

        sys.stdout.flush()
        self.all_data = all_data
        
        if percentage <=1.0:
            num = int(percentage*len(self.all_data))
            self.all_data = random.sample(self.all_data ,num)
        else:
            self.all_data = random.sample(self.all_data ,int(percentage))

        print("Data size is %0d" % len(self.all_data))

class ClapTestDataset(Dataset): 
    def __len__(self) -> int:
        return len(self.all_data)

    def post_processing(self, captions):
        output_caption = []
        for item in captions:
            output_caption.append(str(item["caption"].lower()))
        return output_caption

    def __getitem__(self, item: int):
        
        prefix = self.all_data[item]["audio_embedding"]
        audio_id = self.all_data[item]["audio_id"]
        if self.normalize_prefix:
            prefix = F.normalize(prefix, dim=-1)

        return audio_id, prefix

    def __init__(self, data_path: str,normalize_prefix=False):
        self.normalize_prefix = normalize_prefix
        # self.tokenizer = GPT2Tokenizer.from_pretrained('/home/zhangyiming/clap/GPTModel/')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data))
        sys.stdout.flush()
        self.all_data = all_data
        self.max_seq_len = 25

class ClapDataset_Mistral(Dataset):
    def __len__(self) -> int:
        return len(self.all_data)

    def pad_tokens(self, caption):
        if caption[-1] == '.':
            caption= caption[:-1]
        tokens = self.tokenizer(caption,return_tensors="pt")['input_ids'].squeeze(0)
        # tokens = torch.tensor(, dtype=torch.int64)
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        
        tokens, mask = self.pad_tokens(self.all_data[item]["caption"])

        if self.use_audio_embedding: # Tradition Audio Captioning
            prefix = self.all_data[item]["audio_embedding"]
        elif self.use_related_text: # Zero-shot Audio Captioning
            random_index =  torch.randint(0, self.all_data[item]["related_embeddings"].size(0), (1,))
            prefix = self.all_data[item]["related_embeddings"][random_index]
        else:
            prefix =  self.all_data[item]["text_embedding"]

        if self.normalize_prefix:
            prefix = F.normalize(prefix, dim=-1)

        return tokens, mask, prefix

    def __init__(self, data_path: list,  prefix_length: int, use_related_text=False,
                 normalize_prefix=False, use_audio_embedding=False, percentage=1.0):

        # self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1",add_bos_token=False,add_eos_token=True)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        self.use_audio_embedding = use_audio_embedding
        self.use_related_text = use_related_text
        self.max_seq_len = 25
        all_data = list()

        for dp in data_path:
            with open(dp, 'rb') as f:
                all_data = all_data + pickle.load(f)

        sys.stdout.flush()
        self.all_data = all_data
        
        if percentage <=1.0:
            num = int(percentage*len(self.all_data))
            self.all_data = random.sample(self.all_data ,num)
        else:
            self.all_data = random.sample(self.all_data ,int(percentage))

        print("Data size is %0d" % len(self.all_data))

class ClapTestDataset_Mistral(Dataset): 
    def __len__(self) -> int:
        return len(self.all_data)

    def post_processing(self, captions):
        output_caption = []
        for item in captions:
            output_caption.append(str(item["caption"].lower()))
        return output_caption

    def __getitem__(self, item: int):
        
        prefix = self.all_data[item]["audio_embedding"]
        audio_id = self.all_data[item]["audio_id"]
        if self.normalize_prefix:
            prefix = F.normalize(prefix, dim=-1)

        return audio_id, prefix

    def __init__(self, data_path: str,normalize_prefix=False):
        self.normalize_prefix = normalize_prefix
        # self.tokenizer = GPT2Tokenizer.from_pretrained('/home/zhangyiming/clap/GPTModel/')
        # self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1",add_bos_token=False,add_eos_token=True)
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data))
        sys.stdout.flush()
        self.all_data = all_data
        self.max_seq_len = 25


class ClapDataset_Mistral_multilingual(Dataset):
    def __len__(self) -> int:
        return len(self.all_data)

    def pad_tokens(self, caption):
        if caption[-1] == '.' or caption[-1] =='。':
            caption= caption[:-1]
        tokens = self.tokenizer(caption,return_tensors="pt")['input_ids'].squeeze(0)
        # tokens = torch.tensor(, dtype=torch.int64)
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        
        # tokens, mask = self.pad_tokens(self.all_data[item]["caption"])
        random_key = random.choice(list(self.lang.keys()))
        random_value = self.lang[random_key]
        if random_key == 'en':
            # print(random_value + self.all_data[item]["caption"])
            tokens, mask = self.pad_tokens(random_value + self.all_data[item]["caption"])
            # print(self.tokenizer.decode(tokens))
        elif random_key == 'zh':
            # print(random_value + self.all_data[item]["cn_caption"])
            tokens, mask = self.pad_tokens(random_value + self.all_data[item]["cn_caption"])
            # print(self.tokenizer.decode(tokens))
        else:
            # print(random_value + self.all_data[item]["fr_caption"])
            tokens, mask = self.pad_tokens(random_value + self.all_data[item]["fr_caption"])
            # print(self.tokenizer.decode(tokens))
        
        if self.use_audio_embedding: # Tradition Audio Captioning
            prefix = self.all_data[item]["audio_embedding"]
        elif self.use_related_text: # Zero-shot Audio Captioning
            random_index =  torch.randint(0, self.all_data[item]["related_embeddings"].size(0), (1,))
            prefix = self.all_data[item]["related_embeddings"][random_index]
        else:
            prefix =  self.all_data[item]["text_embedding"]

        if self.normalize_prefix:
            prefix = F.normalize(prefix, dim=-1)

        return tokens, mask, prefix

    def __init__(self, data_path: list,  prefix_length: int, use_related_text=False,
                 normalize_prefix=False, use_audio_embedding=False, percentage=1.0):

        # self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.lang = {
            'en':'<en> ',
            'zh':'<zh> ',
            'fr':'<fr> ',
        }
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1",add_bos_token=False,add_eos_token=True)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        self.use_audio_embedding = use_audio_embedding
        self.use_related_text = use_related_text
        self.max_seq_len = 40
        all_data = list()

        for dp in data_path:
            with open(dp, 'rb') as f:
                all_data = all_data + pickle.load(f)

        sys.stdout.flush()
        self.all_data = all_data
        
        # if percentage <=1.0:
        #     num = int(percentage*len(self.all_data))
        #     self.all_data = random.sample(self.all_data ,num)
        # else:
        #     self.all_data = random.sample(self.all_data ,int(percentage))

        print("Data size is %0d" % len(self.all_data))

class ClapTestDataset_Mistral_multilingual(Dataset): 
    def __len__(self) -> int:
        return len(self.all_data)

    def post_processing(self, captions):
        output_caption = []
        for item in captions:
            output_caption.append(str(item["caption"].lower()))
        return output_caption

    def __getitem__(self, item: int):
        
        prefix = self.all_data[item]["audio_embedding"]
        audio_id = self.all_data[item]["audio_id"]
        if self.normalize_prefix:
            prefix = F.normalize(prefix, dim=-1)

        return audio_id, prefix

    def __init__(self, data_path: str,normalize_prefix=False):
        self.normalize_prefix = normalize_prefix
        # self.tokenizer = GPT2Tokenizer.from_pretrained('/home/zhangyiming/clap/GPTModel/')
        # self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1",add_bos_token=False,add_eos_token=True)
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data))
        sys.stdout.flush()
        self.all_data = all_data
        self.max_seq_len = 40

class ClapDataset_withHardPrompt(Dataset):
    def __len__(self) -> int:
        return len(self.all_data)

    def pad_tokens(self, caption):
        if caption[-1] != '.':
            caption= caption+'.'

        tokens = torch.tensor(self.tokenizer.encode(caption), dtype=torch.int64)
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        
        tokens, mask = self.pad_tokens(self.all_data[item]["caption"])

        if self.use_audio_embedding: # Tradition Audio Captioning
            prefix = self.all_data[item]["audio_embedding"]
        elif self.use_related_text: # Zero-shot Audio Captioning
            random_index =  torch.randint(0, self.all_data[item]["related_embeddings"].size(0), (1,))
            prefix = self.all_data[item]["related_embeddings"][random_index]
        else:
            prefix =  self.all_data[item]["text_embedding"]
        
        sound_effects_index = sound_effect_choice(prefix, self.sound_effect_embeddings, self.sound_effect_num).squeeze(0)
        # selected_labels = self.sound_effect_labels[for index in sound_effects_index].squeeze(1)
        selected_labels = [self.sound_effect_labels[i].lower() for i in list(sound_effects_index)]
        hard_prompt = parse_entities(self.tokenizer,selected_labels,self.mask_probability)

        if self.normalize_prefix:
            prefix = F.normalize(prefix, dim=-1)

        return tokens, mask, prefix,hard_prompt,len(hard_prompt)

    def __init__(self, data_path: list,  prefix_length: int, sound_effect_path: str,use_related_text=False,
                 normalize_prefix=False, use_audio_embedding=False, sound_effect_num=3,percentage=1.0,mask_probability=0):

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        self.use_audio_embedding = use_audio_embedding
        self.use_related_text = use_related_text
        self.max_seq_len = 25
        self.sound_effect_num = sound_effect_num
        self.mask_probability = mask_probability
        all_data = list()

        with open(sound_effect_path, 'rb') as sound_effect:
            audioset_labels = pickle.load(sound_effect)
        id2label = dict()
        label2id = dict()
        self.sound_effect_embeddings = list()
        self.sound_effect_labels = list()
        for audioset_label in audioset_labels:
            id2label[audioset_label['label_id']] = audioset_label['label']
            label2id[audioset_label['label']] = audioset_label['label_id']
            self.sound_effect_embeddings.append(audioset_label['label_embedding'])
            self.sound_effect_labels.append(audioset_label['label'])
        self.sound_effect_embeddings = torch.cat(self.sound_effect_embeddings,dim=0)
        for dp in data_path:
            # print(dp)
            with open(dp, 'rb') as f:
                while True:
                    try:
                        item = pickle.load(f)
                        if type(item) is list:
                            # print("yes!!!!!")
                            all_data = all_data + item
                        else:
                            all_data.append(item)
                            # if len(item['caption'].split())>=8 and len(item['caption'].split())<=20:
                            #     all_data.append(item)
                            # else:
                            #     pass
                    except EOFError:
                        break
                       

        sys.stdout.flush()
        self.all_data = all_data
        
        if percentage <=1.0:
            num = int(percentage*len(self.all_data))
            self.all_data = random.sample(self.all_data ,num)
        else:
            self.all_data = random.sample(self.all_data ,int(percentage))

        print("Data size is %0d" % len(self.all_data))

class ClapTestDataset_withHardPrompt(Dataset): 
    def __len__(self) -> int:
        return len(self.all_data)

    def post_processing(self, captions):
        output_caption = []
        for item in captions:
            output_caption.append(str(item["caption"].lower()))
        return output_caption

    def __getitem__(self, item: int):
        
        prefix = self.all_data[item]["audio_embedding"]
        audio_id = self.all_data[item]["audio_id"]
        sound_effects_index = sound_effect_choice(prefix, self.sound_effect_embeddings, self.sound_effect_num).squeeze(0)
        # selected_labels = self.sound_effect_labels[for index in sound_effects_index].squeeze(1)
        selected_labels = [self.sound_effect_labels[i].lower() for i in list(sound_effects_index)]
        hard_prompt = parse_entities(self.tokenizer,selected_labels,self.mask_probability)
        # import pdb;pdb.set_trace()
        if self.normalize_prefix:
            prefix = F.normalize(prefix, dim=-1)

        return audio_id, prefix,hard_prompt,len(hard_prompt)

    def __init__(self, data_path: str,normalize_prefix=False,sound_effect_path=str,sound_effect_num=int):
        self.normalize_prefix = normalize_prefix
        self.sound_effect_num = sound_effect_num
        self.mask_probability = 0
        # self.tokenizer = GPT2Tokenizer.from_pretrained('/home/zhangyiming/clap/GPTModel/')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        with open(sound_effect_path, 'rb') as sound_effect:
            audioset_labels = pickle.load(sound_effect)
        id2label = dict()
        label2id = dict()
        self.sound_effect_embeddings = list()
        self.sound_effect_labels = list()
        for audioset_label in audioset_labels:
            id2label[audioset_label['label_id']] = audioset_label['label']
            label2id[audioset_label['label']] = audioset_label['label_id']
            self.sound_effect_embeddings.append(audioset_label['label_embedding'])
            self.sound_effect_labels.append(audioset_label['label'])
        self.sound_effect_embeddings = torch.cat(self.sound_effect_embeddings,dim=0)
        # self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data))
        sys.stdout.flush()
        self.all_data = all_data
        self.max_seq_len = 25

class ClapDataset_Mistral_multilingual_withHardPrompt(Dataset):
    def __len__(self) -> int:
        return len(self.all_data)

    def pad_tokens(self, caption):
        if caption[-1] == '.' or caption[-1] =='。':
            caption= caption[:-1]
        tokens = self.tokenizer(caption,return_tensors="pt")['input_ids'].squeeze(0)
        # tokens = torch.tensor(, dtype=torch.int64)
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        
        # tokens, mask = self.pad_tokens(self.all_data[item]["caption"])
        random_key = random.choice(list(self.lang.keys()))
        random_value = self.lang[random_key]
        if random_key == 'en':
            # print(random_value + self.all_data[item]["caption"])
            tokens, mask = self.pad_tokens(random_value + self.all_data[item]["caption"])
            # print(self.tokenizer.decode(tokens))
        elif random_key == 'zh':
            # print(random_value + self.all_data[item]["cn_caption"])
            tokens, mask = self.pad_tokens(random_value + self.all_data[item]["cn_caption"])
            # print(self.tokenizer.decode(tokens))
        else:
            # print(random_value + self.all_data[item]["fr_caption"])
            tokens, mask = self.pad_tokens(random_value + self.all_data[item]["fr_caption"])
            # print(self.tokenizer.decode(tokens))
        
        if self.use_audio_embedding: # Tradition Audio Captioning
            prefix = self.all_data[item]["audio_embedding"]
        elif self.use_related_text: # Zero-shot Audio Captioning
            random_index =  torch.randint(0, self.all_data[item]["related_embeddings"].size(0), (1,))
            prefix = self.all_data[item]["related_embeddings"][random_index]
        else:
            prefix =  self.all_data[item]["text_embedding"]

        if self.normalize_prefix:
            prefix = F.normalize(prefix, dim=-1)

        sound_effects_index = sound_effect_choice(prefix, self.sound_effect_embeddings, self.sound_effect_num).squeeze(0)
        selected_labels = [self.sound_effect_labels[i].lower() for i in list(sound_effects_index)]
        hard_prompt = parse_entities(self.tokenizer,selected_labels,self.mask_probability)
        
        return tokens, mask, prefix, hard_prompt, len(hard_prompt)

    def __init__(self, data_path: list,  prefix_length: int, sound_effect_path: str, use_related_text=False,
                 normalize_prefix=False, use_audio_embedding=False, sound_effect_num=3, percentage=1.0, mask_probability=0):
        # self.sound_effect_num = sound_effect_num
        # self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.lang = {
            'en':'<en> ',
            'zh':'<zh> ',
            'fr':'<fr> ',
        }
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1",add_bos_token=False,add_eos_token=True)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        self.use_audio_embedding = use_audio_embedding
        self.use_related_text = use_related_text
        self.max_seq_len = 40
        self.sound_effect_num = sound_effect_num
        self.mask_probability = mask_probability
        all_data = list()

        with open(sound_effect_path, 'rb') as sound_effect:
            audioset_labels = pickle.load(sound_effect)
        id2label = dict()
        label2id = dict()
        self.sound_effect_embeddings = list()
        self.sound_effect_labels = list()
        for audioset_label in audioset_labels:
            id2label[audioset_label['label_id']] = audioset_label['label']
            label2id[audioset_label['label']] = audioset_label['label_id']
            self.sound_effect_embeddings.append(audioset_label['label_embedding'])
            self.sound_effect_labels.append(audioset_label['label'])
        self.sound_effect_embeddings = torch.cat(self.sound_effect_embeddings,dim=0)
        
        for dp in data_path:
            with open(dp, 'rb') as f:
                all_data = all_data + pickle.load(f)

        sys.stdout.flush()
        self.all_data = all_data
        
        # if percentage <=1.0:
        #     num = int(percentage*len(self.all_data))
        #     self.all_data = random.sample(self.all_data ,num)
        # else:
        #     self.all_data = random.sample(self.all_data ,int(percentage))

        print("Data size is %0d" % len(self.all_data))

class ClapTestDataset_Mistral_multilingual_withHardPrompt(Dataset): 
    def __len__(self) -> int:
        return len(self.all_data)

    def post_processing(self, captions):
        output_caption = []
        for item in captions:
            output_caption.append(str(item["caption"].lower()))
        return output_caption

    def __getitem__(self, item: int):
        
        prefix = self.all_data[item]["audio_embedding"]
        audio_id = self.all_data[item]["audio_id"]
        if self.normalize_prefix:
            prefix = F.normalize(prefix, dim=-1)
        
        sound_effects_index = sound_effect_choice(prefix, self.sound_effect_embeddings, self.sound_effect_num).squeeze(0)
        selected_labels = [self.sound_effect_labels[i].lower() for i in list(sound_effects_index)]
        hard_prompt = parse_entities(self.tokenizer,selected_labels,self.mask_probability)

        return audio_id, prefix,hard_prompt,len(hard_prompt)

    def __init__(self, data_path: str,normalize_prefix=False,sound_effect_path=str,sound_effect_num=int):
        self.normalize_prefix = normalize_prefix
        self.sound_effect_num = sound_effect_num
        self.mask_probability = 0
        with open(sound_effect_path, 'rb') as sound_effect:
            audioset_labels = pickle.load(sound_effect)
        id2label = dict()
        label2id = dict()
        self.sound_effect_embeddings = list()
        self.sound_effect_labels = list()
        for audioset_label in audioset_labels:
            id2label[audioset_label['label_id']] = audioset_label['label']
            label2id[audioset_label['label']] = audioset_label['label_id']
            self.sound_effect_embeddings.append(audioset_label['label_embedding'])
            self.sound_effect_labels.append(audioset_label['label'])
        self.sound_effect_embeddings = torch.cat(self.sound_effect_embeddings,dim=0)
        # self.tokenizer = GPT2Tokenizer.from_pretrained('/home/zhangyiming/clap/GPTModel/')
        # self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1",add_bos_token=False,add_eos_token=True)
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data))
        sys.stdout.flush()
        self.all_data = all_data
        self.max_seq_len = 40

def collate(batch):
    batch_size = len(batch)
    # args = batch[0][0]
    if len(batch[0])== 5:
        tokens, mask, prefix,hard_prompt,hard_prompt_length = zip(*batch)
        tokens = torch.stack(tokens, dim=0)
        mask =  torch.stack(mask)
    else:
        audio_id, prefix,hard_prompt,hard_prompt_length = zip(*batch)
    
    prefix = torch.stack(prefix)
    padding_hard_prompt,hard_prompts_masks = padding_captions(hard_prompt,hard_prompt_length)
    if len(batch[0])== 5:
        return tokens, mask, prefix, padding_hard_prompt,hard_prompts_masks
    else:
        return audio_id, prefix,padding_hard_prompt,hard_prompts_masks

if __name__ == '__main__':
    #     dataset = ClapDataset(args.data, args.prefix_length, normalize_prefix=args.normalize_prefix,
    #                        use_audio_embedding=args.use_audio_embedding, use_related_text=args.use_related_text, percentage=args.percentage)
    # data = ['/mnt/fast/nobackup/scratch4weeks/yz02417/zero-shot-AC/data/clotho/train/clap_embedding/ZS/data_related.pkl',]
    data = '/mnt/fast/nobackup/scratch4weeks/yz02417/zero-shot-AC/data/clotho/test/clap_embedding/ZS/data.pkl'

    prefix_length = 10
    sound_effect_path = '/mnt/fast/nobackup/scratch4weeks/yz02417/zero-shot-AC/data/audioset_label.pkl'
    normalize_prefix = True
    use_audio_embedding = False
    use_related_text = True
    percentage = 1.0
    sound_effect_num = 4
    mask_probability = 0.4
    dataset = ClapTestDataset_withHardPrompt(data_path=data,normalize_prefix=True,sound_effect_path=sound_effect_path,sound_effect_num=sound_effect_num)
    # dataset = ClapDataset_withHardPrompt(data_path=data,prefix_length=prefix_length,sound_effect_path=sound_effect_path,
    #                                      use_audio_embedding=use_audio_embedding,use_related_text=use_audio_embedding,sound_effect_num=sound_effect_num,
    #                                      mask_probability=mask_probability,percentage=percentage,normalize_prefix=normalize_prefix)
    print(dataset[0])
    # train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True,collate_fn=collate)
    # for data in train_dataloader:

    #     tokens, mask, prefix, padding_hard_prompt,hard_prompts_masks = data
    #     # audio_id, prefix, padding_hard_prompt,hard_prompts_masks = data
    #     # print(audio_id)
    #     print(padding_hard_prompt)
        # break