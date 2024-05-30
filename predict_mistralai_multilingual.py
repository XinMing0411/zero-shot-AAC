import sys
from torch.utils.data import Dataset, DataLoader
from custom_types import *
import torch
import json
import re
import stanza
import yaml
import argparse, pickle
from gpt2_prefix_eval import generate2,generate_beam_magic
import os.path
from tqdm import tqdm

sys.path.append(os.getcwd())
from models.caption_model import *
from dataset.dataset import *
from utils import *

lang = {'en':'<en>',
        'zh':'<zh>',
        'fr':'<fr>',}
def eval_prediction(key2refs, key2pred, scorers,lang='en'):

    nlp = stanza.Pipeline(lang, processors='tokenize',tokenize_no_ssplit=True)
    punctuation = "\【.*?】+|\《.*?》+|\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\，。=？、：“”‘’￥……()《》【】～]"
    pred4eval = {}
    for key, preds in key2pred.items():
        preds = preds[0]
        preds = re.sub(punctuation,'',preds).lower()
        preds_doc = nlp(preds).sentences[0]
        word_tokens = [token.text for token in preds_doc.tokens]
        pred4eval[key] = [' '.join(word_tokens)]

    refs4eval = {}
    for key, refs in key2refs.items():
        refs4eval[key] = []
        for idx, ref in enumerate(refs):
            ref = re.sub(punctuation,'',ref).lower()
            ref_doc = nlp(ref).sentences[0]
            word_tokens = [token.text for token in ref_doc.tokens]
            refs4eval[key].append(' '.join(word_tokens))
    
    output = {}
    for scorer in scorers:
        score, scores = scorer.compute_score(refs4eval, pred4eval)
        output[scorer.method()] = score
    return output

def post_processing_multilingual(items):
    caption = []
    caption_zh = []
    caption_fr =[]
    for item in items['caption']:
        caption.append(str(item["caption"].lower()))
       
    for item in items['cn_caption']:
        caption_zh.append(str(item["caption"].lower()))
        
    for item in items['fr_caption']:
        caption_fr.append(str(item["caption"].lower()))
    return caption,caption_zh,caption_fr

def post_processing(captions):
    output_caption = []
    for item in captions:
        if item["caption"][-1] != ".":
            caption = item["caption"]+"."
        else:
            caption = item["caption"]
        output_caption.append(str(caption.lower()))
    return output_caption

def make_preds(dataset, model: ClapCaption_Mistralai, tokenizer, args):
    model = model.to(args.device)
    model.eval()
    key2pred_en = {}
    key2pred_zh = {}
    key2pred_fr = {}
    key2refs_en = {}
    key2refs_zh = {}
    key2refs_fr = {}
    with open(args.test_data, 'rb') as f:
        all_data = pickle.load(f)

    for item in range(len(all_data)):
        captions,captions_zh,captions_fr = post_processing_multilingual(all_data[item])
        key2refs_en[all_data[item]["audio_id"]] = captions
        key2refs_zh[all_data[item]["audio_id"]] = captions_zh
        key2refs_fr[all_data[item]["audio_id"]] = captions_fr
    test_dataloader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False,num_workers=4,collate_fn=collate)
            
    with tqdm(total=len(test_dataloader), ncols=100,ascii=True) as pbar:
        if args.magic:
            from retrieval.models.ase_model import ASE
            with open("retrieval/settings/extract.yaml", "r") as f:
                config = yaml.safe_load(f)
            clap = ASE(config).to(args.device)
            clap.eval()
            clap_state_dict = torch.load("/mnt/fast/nobackup/scratch4weeks/yz02417/zero-shot-AC/clap_model/HTSAT-BERT-ZS.pt", map_location=args.device)
            clap.load_state_dict(clap_state_dict["model"])
        for idx, (audio_id, prefix, padding_hard_prompt,hard_prompts_masks) in enumerate(test_dataloader):

            prefix = prefix.to(args.device, dtype=torch.float32)
            padding_hard_prompt = padding_hard_prompt.to(args.device)
            embedding_hard_prompt =model.LMmodel.base_model.model.model.embed_tokens(padding_hard_prompt)

            with torch.no_grad():
                tokens_en = tokenizer(lang['en'],return_tensors="pt")['input_ids'].reshape(1,-1).repeat(prefix.shape[0], 1)
                embedding_text_en = model.LMmodel.base_model.model.model.embed_tokens(tokens_en)
                prefix_embed_en, _ = model.clap_to_gpt(prefix,embedding_hard_prompt,embedding_text_en)

                tokens_zh = tokenizer(lang['zh'],return_tensors="pt")['input_ids'].reshape(1,-1).repeat(prefix.shape[0], 1)
                embedding_text_zh = model.LMmodel.base_model.model.model.embed_tokens(tokens_zh)
                prefix_embed_zh, _ = model.clap_to_gpt(prefix,embedding_hard_prompt,embedding_text_zh)

                tokens_fr = tokenizer(lang['fr'],return_tensors="pt")['input_ids'].reshape(1,-1).repeat(prefix.shape[0], 1)
                embedding_text_fr = model.LMmodel.base_model.model.model.embed_tokens(tokens_fr)
                prefix_embed_fr, _ = model.clap_to_gpt(prefix,embedding_hard_prompt,embedding_text_fr)

            if args.isbeam:
                attention_mask_en = torch.ones(prefix_embed_en.shape[:-1]).long().to(embedding_text_en.device)
                generated_text_prefix_en = tokenizer.batch_decode(model.LMmodel.generate(inputs_embeds=prefix_embed_en,attention_mask=attention_mask_en,do_sample=False,max_length=60,eos_token_id=2,pad_token_id=2),skip_special_tokens=True)
                
                attention_mask_zh = torch.ones(prefix_embed_zh.shape[:-1]).long().to(embedding_text_zh.device)
                generated_text_prefix_zh = tokenizer.batch_decode(model.LMmodel.generate(inputs_embeds=prefix_embed_zh,attention_mask=attention_mask_zh,do_sample=False,max_length=60,eos_token_id=2,pad_token_id=2),skip_special_tokens=True)

                attention_mask_fr = torch.ones(prefix_embed_fr.shape[:-1]).long().to(embedding_text_fr.device)
                generated_text_prefix_fr = tokenizer.batch_decode(model.LMmodel.generate(inputs_embeds=prefix_embed_fr,attention_mask=attention_mask_fr,do_sample=False,max_length=60,eos_token_id=2,pad_token_id=2),skip_special_tokens=True)
            else:
                generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed_en)

            for index in range(len(audio_id)):
                key2pred_en[audio_id[index]] = [generated_text_prefix_en[index].lower()]
                key2pred_zh[audio_id[index]] = [generated_text_prefix_zh[index].lower()]
                key2pred_fr[audio_id[index]] = [generated_text_prefix_fr[index].lower()]

            pbar.update()


    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor

    scorers = [Bleu(n=4), Rouge(), Cider(),Meteor()]

    scores_output_en = eval_prediction(key2refs_en, key2pred_en, scorers,lang='en')
    scores_output_zh = eval_prediction(key2refs_zh, key2pred_zh, scorers,lang='zh')
    scores_output_fr = eval_prediction(key2refs_fr, key2pred_fr, scorers,lang='fr')

    with open(os.path.join(args.test_dir,'scores_en.txt'), "w") as f:
        spider = 0
        for name, score in scores_output_en.items():
            if name == "Bleu":
                for n in range(4):
                    f.write("Bleu-{}: {:6.4f}\n".format(n + 1, score[n]))
            else:
                f.write("{}: {:6.4f}\n".format(name, score))
                if name in ["CIDEr", "SPICE"]:
                    spider += score

        f.write("SPIDEr: {:6.4f}\n".format(spider / 2))
    with open(os.path.join(args.test_dir,'scores_zh.txt'), "w") as f:
        spider = 0
        for name, score in scores_output_zh.items():
            if name == "Bleu":
                for n in range(4):
                    f.write("Bleu-{}: {:6.4f}\n".format(n + 1, score[n]))
            else:
                f.write("{}: {:6.4f}\n".format(name, score))
                if name in ["CIDEr", "SPICE"]:
                    spider += score

        f.write("SPIDEr: {:6.4f}\n".format(spider / 2))
    with open(os.path.join(args.test_dir,'scores_fr.txt'), "w") as f:
        spider = 0
        for name, score in scores_output_fr.items():
            if name == "Bleu":
                for n in range(4):
                    f.write("Bleu-{}: {:6.4f}\n".format(n + 1, score[n]))
            else:
                f.write("{}: {:6.4f}\n".format(name, score))
                if name in ["CIDEr", "SPICE"]:
                    spider += score

        f.write("SPIDEr: {:6.4f}\n".format(spider / 2))

    pred_data = []
    for key, pred in key2pred_en.items():
        pred_data.append({
            "filename": key,
            "caption": "".join(pred[0]),
            'zh_caption':''.join(key2pred_zh[key][0]),
            'fr_caption':''.join(key2pred_fr[key][0])
        })
    json.dump({"predictions": pred_data},open(os.path.join(args.test_dir,'output.txt') , "w"), indent=4)

def main():
    print('start....')

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1",add_bos_token=False,add_eos_token=False)
    print('loaded tokenizer')

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str)
    parser.add_argument('--isbeam',  action='store_true')
    parser.add_argument('--magic',  action='store_true')
    parser.add_argument('--test_data', default='/home/zhangyiming/clap/clap_embedding_wavcaps/clotho/evaluation/embedding_temp.pkl')
    args = parser.parse_args()
    new_args = argparse.Namespace(**json.load(open(os.path.join(args.test_dir,"params.json"), 'r')))

    args.__dict__.update(new_args.__dict__)

    if args.use_sound_effect:
        with open(args.sound_effect, 'rb') as sound_effect:
            audioset_labels = pickle.load(sound_effect)

        sound_effect_embeddings = list()
        for audioset_label in audioset_labels:

            sound_effect_embeddings.append(audioset_label['label_embedding'])
        sound_effect_embeddings = torch.cat(sound_effect_embeddings,dim=0).to(args.device)
    else:
        sound_effect_embeddings = None

    dataset = ClapTestDataset_Mistral_multilingual_withHardPrompt(args.test_data,normalize_prefix=args.normalize_prefix,sound_effect_path=args.sound_effect,sound_effect_num=args.sound_effect_num)
    prefix_dim = 1024 if args.is_rn else 512
    args.mapping_type = {'mlp':'mlp', 'transformer':'transformer'}[args.mapping_type]
    model = ClapCaption_Mistralai_prompt(args.prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,num_layers=args.num_layers,
                                    mapping_type=args.mapping_type,  only_prefix=args.only_prefix,only_soft_prompt=args.only_soft_prompt,islang=3)

    model.load_state_dict(torch.load(os.path.join(args.test_dir,"best.pth"),map_location="cpu"))  

    make_preds(dataset,model,tokenizer,args)


if __name__ == '__main__':

    main()