import torch
import torch.nn as nn
import random
import pickle
from typing import  Optional
from transformers import  GPT2LMHeadModel
from .mapper import *
from enum import Enum
class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'

class ClapCaptionModel(nn.Module):

    def sound_effect_choice(self, prefix, sound_effect_embeddings, choice_num):

        similarity = prefix.__matmul__(sound_effect_embeddings.t())
        similarity_softmax = F.softmax(similarity.detach().cpu(), dim=-1)
        num,index = torch.topk(similarity_softmax,choice_num,dim=-1)
        
        return sound_effect_embeddings[index].squeeze(1)

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
 
        embedding_text = self.gpt.transformer.wte(tokens)
        embedding_cat, mask = self.clap_to_gpt(prefix, embedding_text, mask)

        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)

        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        
        return out, out.logits[:, self.sound_effect_num+self.prefix_length - 1: -1]

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: MappingType = 'mlp',sound_effect_embeddings:torch.Tensor = None, sound_effect_num: Optional[int] = 0,only_prefix: Optional[bool]=False,mask_probability:Optional[float] = 0):
        super(ClapCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.sound_effect_embeddings = sound_effect_embeddings
        self.sound_effect_num = sound_effect_num
        self.only_prefix = only_prefix
        self.mask_probability = mask_probability
        # with open('/mnt/fast/nobackup/scratch4weeks/yz02417/DCASE/pickles/decoder_config.pkl','rb') as f:
        #     config = pickle.load(f)
        
        # self.gpt = GPT2LMHeadModel(config)
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if mapping_type == 'mlp':
            self.clap_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length))
        else:
            self.clap_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                                     clip_length, num_layers)


        if self.sound_effect_embeddings is not None:
            self.sound_effect_project =MLP((prefix_size, (self.gpt_embedding_size) // 2, self.gpt_embedding_size))

    def clap_to_gpt(self,prefix: torch.Tensor, embedding_text:Optional[torch.Tensor]= None, mask: Optional[torch.Tensor] = None):

        prefix_projections = self.clap_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        if embedding_text is not None:
            embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        else:
            embedding_cat = prefix_projections

        if self.sound_effect_embeddings is not None:
            sound_effects = self.sound_effect_choice(prefix, self.sound_effect_embeddings, self.sound_effect_num)
            sound_effects_projections = self.sound_effect_project(sound_effects).view(-1, self.sound_effect_num, self.gpt_embedding_size)
            embedding_cat = torch.cat((sound_effects_projections, embedding_cat), dim=1)

            if mask is not None:
                mask = torch.cat((torch.ones((prefix.shape[0], self.sound_effect_num)).to(prefix.device),mask),dim=-1)

        return embedding_cat, mask

    def train(self, mode: bool = True):
        super(ClapCaptionModel, self).train(mode)
        if self.only_prefix:
            self.gpt.eval()
            # print("Train only prefix")
        return self  
class ClapCaptionPrefix(ClapCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clap_project.parameters()

    def train(self, mode: bool = True):
        super(ClapCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self

class ClapCaptionCrossattention(ClapCaptionModel):

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,num_layers: int = 8,
                mapping_type: MappingType = 'mlp',sound_effect_embeddings:torch.Tensor = None, sound_effect_num: Optional[int] = 0,only_prefix: Optional[bool]=False):
        super(ClapCaptionCrossattention, self).__init__(prefix_length,clip_length,prefix_size,num_layers,mapping_type,sound_effect_embeddings,sound_effect_num,only_prefix)

        if self.sound_effect_embeddings is not None:
            self.sound_effect_project = nn.MultiheadAttention(prefix_size, 4, batch_first=True)
    
    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
 
        embedding_text = self.gpt.transformer.wte(tokens)
        embedding_cat, mask = self.clap_to_gpt(prefix, embedding_text, mask)

        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)

        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        
        return out, out.logits[:, self.prefix_length - 1: -1]
    
    def clap_to_gpt(self, prefix: torch.Tensor, embedding_text:Optional[torch.Tensor]= None, mask: Optional[torch.Tensor] = None):

        if self.sound_effect_embeddings is not None:

            sound_effects = self.sound_effect_choice(prefix, self.sound_effect_embeddings, self.sound_effect_num)
            prefix, _ = self.sound_effect_project(prefix,sound_effects,sound_effects)
        
        prefix_projections = self.clap_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)

        if embedding_text is not None:
            embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        else:
            embedding_cat = prefix_projections

        return embedding_cat, mask
    
    def train(self, mode: bool = True):
        super(ClapCaptionCrossattention, self).train(mode)
        if self.only_prefix:
            self.gpt.eval()
            # print("Train only prefix")
        return self

class ClapCaptionCrossattention_v2(ClapCaptionModel):

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,num_layers: int = 8,
                mapping_type: MappingType = 'mlp',sound_effect_embeddings:torch.Tensor = None, sound_effect_num: Optional[int] = 0,only_prefix: Optional[bool]=False,mask_probability: Optional[float]=0.25):
        super(ClapCaptionCrossattention_v2, self).__init__(prefix_length,clip_length,prefix_size,num_layers,mapping_type,sound_effect_embeddings,sound_effect_num,only_prefix,mask_probability)

        if self.sound_effect_embeddings is not None:
            self.sound_effect_project = nn.MultiheadAttention(prefix_size, 4, batch_first=True)
    
    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
 
        embedding_text = self.gpt.transformer.wte(tokens)
        embedding_cat, mask = self.clap_to_gpt(prefix, embedding_text, mask)

        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)

        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        
        return out, out.logits[:, self.prefix_length - 1: -1]
    
    def clap_to_gpt(self, prefix: torch.Tensor, embedding_text:Optional[torch.Tensor]= None, mask: Optional[torch.Tensor] = None):

        if self.sound_effect_embeddings is not None:

            sound_effects = self.sound_effect_choice(prefix, self.sound_effect_embeddings, self.sound_effect_num)
            if self.training:
                temp_mask = (torch.rand(prefix.shape[0], 1, self.sound_effect_num) < self.mask_probability).to(prefix.device)
                has_true = (torch.sum(temp_mask,-1) == self.sound_effect_num)
                if has_true.any():
                    for index in has_true.nonzero(as_tuple=False):
                        temp_mask[index[0],0,random.randint(0,3)] = False
                temp_mask = temp_mask.repeat_interleave(4,0)
                # import pdb; pdb.set_trace()
            else:
                temp_mask = None
            x = self.sound_effect_project(prefix,sound_effects,sound_effects,attn_mask=temp_mask)
            # pdb.set_trace()
            prefix = x[0] + prefix
            # print(prefix.shape)
            # prefix = F.normalize(prefix, dim=-1)
        
        prefix_projections = self.clap_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)

        if embedding_text is not None:
            embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        else:
            embedding_cat = prefix_projections

        return embedding_cat, mask
    
    def train(self, mode: bool = True):
        super(ClapCaptionCrossattention_v2, self).train(mode)
        if self.only_prefix:
            for name, param in self.gpt.named_parameters():
                param.requires_grad = False
            # self.gpt.eval()
            # print("Train only prefix")
        return self

class ClapCaption_Mistralai(nn.Module):

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,num_layers: int = 8,
                mapping_type: MappingType = 'mlp',sound_effect_embeddings:torch.Tensor = None, sound_effect_num: Optional[int] = 0,only_prefix: Optional[bool]=False,islang: Optional[int] = 0):
        # super(ClapCaption_Mistralai,self).__init__(config)
        super(ClapCaption_Mistralai, self).__init__()
        from transformers import  MistralForCausalLM
        from transformers import  AutoModelForCausalLM, BitsAndBytesConfig,AutoConfig
        from peft import prepare_model_for_kbit_training,LoraConfig, get_peft_model

        self.prefix_length = prefix_length
        self.sound_effect_embeddings = sound_effect_embeddings
        self.sound_effect_num = sound_effect_num
        self.only_prefix = only_prefix
        self.islang = islang
        bnb_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_use_double_quant=True,
                                        bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16)

        self.LMmodel = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", quantization_config=bnb_config)
        self.lm_embedding_size = self.LMmodel.model.embed_tokens.weight.shape[1]
        # self.LMmodel.gradient_checkpointing_enable()
        self.LMmodel = prepare_model_for_kbit_training(self.LMmodel,use_gradient_checkpointing=False)
        config = LoraConfig(r=8,lora_alpha=16, target_modules=["q_proj", "k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","lm_head",],
                            bias="none",lora_dropout=0.05,  task_type="CAUSAL_LM",)
        self.LMmodel = get_peft_model(self.LMmodel, config)

        # self.lm_embedding_size = self.LMmodel.model.embed_tokens.weight.shape[1]

        if mapping_type == 'mlp':
            self.clap_project = MLP((prefix_size, (self.lm_embedding_size * prefix_length) // 2,
                                     self.lm_embedding_size * prefix_length))
        else:
            self.clap_project = TransformerMapper(prefix_size, self.lm_embedding_size, prefix_length,
                                                                     clip_length, num_layers)


        if self.sound_effect_embeddings is not None:
            self.sound_effect_project = nn.MultiheadAttention(prefix_size, 4, batch_first=True)
    
    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
 
        embedding_text = self.LMmodel.base_model.model.model.embed_tokens(tokens)
        embedding_cat, mask = self.clap_to_gpt(prefix, embedding_text, mask)

        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)

        out = self.LMmodel(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        
        return out, out.logits[:, self.prefix_length + self.islang - 1: -1]
    
    def clap_to_gpt(self, prefix: torch.Tensor, embedding_text:Optional[torch.Tensor]= None, mask: Optional[torch.Tensor] = None):

        if self.sound_effect_embeddings is not None:

            sound_effects = self.sound_effect_choice(prefix, self.sound_effect_embeddings, self.sound_effect_num)
            prefix = self.sound_effect_project(prefix,sound_effects,sound_effects)[0] + prefix
            # prefix = F.normalize(prefix, dim=-1)
        
        prefix_projections = self.clap_project(prefix).view(-1, self.prefix_length, self.lm_embedding_size)

        if embedding_text is not None:
            embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        else:
            embedding_cat = prefix_projections

        return embedding_cat, mask
    def sound_effect_choice(self, prefix, sound_effect_embeddings, choice_num):

        similarity = prefix.__matmul__(sound_effect_embeddings.t())
        similarity_softmax = F.softmax(similarity.detach().cpu(), dim=-1)
        num,index = torch.topk(similarity_softmax,choice_num,dim=-1)
        
        return sound_effect_embeddings[index].squeeze(1)
    # def train(self, mode: bool = True):
    #     super(ClapCaption_Mistralai, self).train(mode)
    #     if self.only_prefix:
    #         self.gpt.eval()
    #         print("Train only prefix")
    #     return self

class ClapCaption_prompt(ClapCaptionModel):

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,num_layers: int = 8,
                mapping_type: MappingType = 'mlp',only_prefix: Optional[bool]=False,only_soft_prompt: Optional[bool]=False):
        super(ClapCaption_prompt, self).__init__(prefix_length,clip_length,prefix_size,num_layers,mapping_type,only_prefix = only_prefix)
        self.only_soft_prompt = only_soft_prompt
    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, padding_hard_prompt: torch.Tensor,
                 mask: Optional[torch.Tensor] = None, hard_prompts_masks: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
 
        embedding_text = self.gpt.transformer.wte(tokens)
        embedding_hard_prompt = self.gpt.transformer.wte(padding_hard_prompt)
        embedding_cat, mask = self.clap_to_gpt(prefix, embedding_hard_prompt,embedding_text, mask,hard_prompts_masks)

        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)

        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        if not self.only_soft_prompt:
            return out, out.logits[:, hard_prompts_masks.shape[1]+self.prefix_length - 1: -1]
        else:
            return out, out.logits[:, self.prefix_length - 1: -1]
    
    def clap_to_gpt(self, prefix: torch.Tensor, embedding_hard_prompt: torch.Tensor,embedding_text:Optional[torch.Tensor]= None,
                    mask: Optional[torch.Tensor] = None,hard_prompts_masks:Optional[torch.Tensor] = None):
        
        prefix_projections = self.clap_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        if not self.only_soft_prompt:
            prefix_projections = torch.cat((embedding_hard_prompt,prefix_projections), dim=1)
           
        if embedding_text is not None:
            embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
            if not self.only_soft_prompt:
                mask = torch.cat((hard_prompts_masks, mask), dim=1)
        else:
            embedding_cat = prefix_projections

        return embedding_cat, mask
    
    def train(self, mode: bool = True):
        super(ClapCaption_prompt, self).train(mode)
        if self.only_prefix:
            # for name, param in self.gpt.named_parameters():
            #     param.requires_grad = False
            self.gpt.eval()
            # print("Train only prefix")
        return self

class ClapCaption_Mistralai_prompt(nn.Module):

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,num_layers: int = 8,
                mapping_type: MappingType = 'mlp',only_prefix: Optional[bool]=False, only_soft_prompt: Optional[bool]=False,islang: Optional[int] = 0):
        # super(ClapCaption_Mistralai,self).__init__(config)
        super(ClapCaption_Mistralai_prompt, self).__init__()
        from transformers import  MistralForCausalLM
        from transformers import  AutoModelForCausalLM, BitsAndBytesConfig,AutoConfig
        from peft import prepare_model_for_kbit_training,LoraConfig, get_peft_model

        self.prefix_length = prefix_length
        self.only_soft_prompt = only_soft_prompt
        # self.sound_effect_num = sound_effect_num
        self.only_prefix = only_prefix
        self.islang = islang
        bnb_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_use_double_quant=True,
                                        bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16)

        self.LMmodel = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", quantization_config=bnb_config)
        self.lm_embedding_size = self.LMmodel.model.embed_tokens.weight.shape[1]
        # self.LMmodel.gradient_checkpointing_enable()
        self.LMmodel = prepare_model_for_kbit_training(self.LMmodel,use_gradient_checkpointing=False)
        config = LoraConfig(r=8,lora_alpha=16, target_modules=["q_proj", "k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","lm_head",],
                            bias="none",lora_dropout=0.05,  task_type="CAUSAL_LM",)
        self.LMmodel = get_peft_model(self.LMmodel, config)

        # self.lm_embedding_size = self.LMmodel.model.embed_tokens.weight.shape[1]

        if mapping_type == 'mlp':
            self.clap_project = MLP((prefix_size, (self.lm_embedding_size * prefix_length) // 2,
                                     self.lm_embedding_size * prefix_length))
        else:
            self.clap_project = TransformerMapper(prefix_size, self.lm_embedding_size, prefix_length,
                                                                     clip_length, num_layers)


        # if self.sound_effect_embeddings is not None:
        #     self.sound_effect_project = nn.MultiheadAttention(prefix_size, 4, batch_first=True)
    
    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, padding_hard_prompt: torch.Tensor,
                mask: Optional[torch.Tensor] = None,hard_prompts_masks: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
 
        embedding_text = self.LMmodel.base_model.model.model.embed_tokens(tokens)
        embedding_hard_prompt = self.LMmodel.base_model.model.model.embed_tokens(padding_hard_prompt)
        embedding_cat, mask = self.clap_to_gpt(prefix, embedding_hard_prompt,embedding_text, mask,hard_prompts_masks)

        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)

        out = self.LMmodel(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        
        if not self.only_soft_prompt:
            return out, out.logits[:, hard_prompts_masks.shape[1]+self.prefix_length + self.islang - 1: -1]
        else:
            return out, out.logits[:, self.prefix_length + self.islang - 1: -1]
        return out, out.logits[:, self.prefix_length + self.islang - 1: -1]
    
    def clap_to_gpt(self, prefix: torch.Tensor, embedding_hard_prompt: torch.Tensor, embedding_text:Optional[torch.Tensor]= None, 
                    mask: Optional[torch.Tensor] = None, hard_prompts_masks:Optional[torch.Tensor] = None):
        
        prefix_projections = self.clap_project(prefix).view(-1, self.prefix_length, self.lm_embedding_size)
        if not self.only_soft_prompt:
            prefix_projections = torch.cat((embedding_hard_prompt,prefix_projections), dim=1)
            if hard_prompts_masks is not None:
                mask = torch.cat((hard_prompts_masks, mask), dim=1)
        if embedding_text is not None:
            embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)

        else:
            embedding_cat = prefix_projections

        return embedding_cat, mask
    def sound_effect_choice(self, prefix, sound_effect_embeddings, choice_num):

        similarity = prefix.__matmul__(sound_effect_embeddings.t())
        similarity_softmax = F.softmax(similarity.detach().cpu(), dim=-1)
        num,index = torch.topk(similarity_softmax,choice_num,dim=-1)
        
        return sound_effect_embeddings[index].squeeze(1)
    # def train(self, mode: bool = True):
    #     super(ClapCaption_Mistralai, self).train(mode)
    #     if self.only_prefix:
    #         self.gpt.eval()
    #         print("Train only prefix")
    #     return self
