import sys
import re
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
from custom_types import *
import torch
import json
import yaml
import argparse, pickle
from gpt2_prefix_eval import  generate_beam,generate2,get_prefix_tokens,magic_search,generate_beam_magic
import os.path
from tqdm import tqdm

sys.path.append(os.getcwd())
from models.caption_model import *
from dataset.dataset import *
from utils import *

params = {
    'beta':0.2,
    'alpha':0.1,
}
def map2memory(audio_embed,text_features):
    audio_embed = torch.tensor(audio_embed).to(audio_embed.device)
    sim = audio_embed@text_features.T.float()
    sim = (sim*100).softmax(dim=-1)
    prefix_embedding = sim@text_features.float()
    prefix_embedding /= prefix_embedding.norm(dim=-1,keepdim=True)
    return prefix_embedding
def construct_support_memory(text_json):
    # with open(text_json, 'rb') as f:
    #     data = pickle.load(f)
    all_data = list()
    for dp in text_json:
    # print(dp)
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
                            pass
                except EOFError:
                    break
    temp_data = []
    for item in all_data:
        temp_data.append(item['text_embedding'])
    text_features = temp_data    

    text_features = torch.cat(text_features,dim=0)
    text_features /= text_features.norm(dim=-1,keepdim=True).float()    
    return text_features

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

def post_processing(captions):
    output_caption = []
    for item in captions:
        if item["caption"][-1] != ".":
            caption = item["caption"]+"."
        else:
            caption = item["caption"]
        # print(caption)
        output_caption.append(str(caption.lower()))
    return output_caption


def make_preds(dataset, model: ClapCaptionModel, tokenizer, args):
    model = model.to(args.device)
    model.eval()
    key2pred = {}
    key2pred_prefix = {}
    key2refs = {}
    with open(args.test_data, 'rb') as f:
        all_data = pickle.load(f)

    for item in range(len(all_data)):
        captions = post_processing(all_data[item]["caption"])
        key2refs[all_data[item]["audio_id"]] = captions
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False,collate_fn=collate)
    embeddings = model.gpt.get_input_embeddings().weight.data
    embeddings = nnf.normalize(embeddings, 2, 1)

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
            embedding_hard_prompt = model.gpt.transformer.wte(padding_hard_prompt)
            # prefix = map2memory(prefix,text_features)
            with torch.no_grad():
                prefix_embed, _ = model.clap_to_gpt(prefix,embedding_hard_prompt)
                prefix_sent = get_prefix_tokens(prefix_embed, embeddings, tokenizer)

            if args.magic:
                generated_text_prefix = generate_beam_magic(model,clap, tokenizer,audio_embeds=prefix, embed=prefix_embed, beam_size=3,alpha=args.alpha, beta=args.beta,magic_width=args.magic_width)[0]
            elif args.isbeam:
                generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed,beam_size=3)[0]
            else:
                generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)

            key2pred[audio_id[0]] = [generated_text_prefix.lower()]
            key2pred_prefix[audio_id[0]] = [prefix_sent[0]]
            pbar.update()


    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.spice.spice import Spice
    # from fense.fense import Fense

    # scorers = [Bleu(n=4), Rouge(), Cider(),Meteor(),Spice(),Fense(penalty=0)]
    scorers = [Bleu(n=4), Rouge(), Cider(),Meteor(),Spice()]
    scores_output = eval_prediction(key2refs, key2pred, scorers)

    with open(os.path.join(args.test_dir,'scores.txt'), "w") as f:
        spider = 0
        for name, score in scores_output.items():
            if name == "Bleu":
                for n in range(4):
                    f.write("Bleu-{}: {:6.4f}\n".format(n + 1, score[n]))
            else:
                f.write("{}: {:6.4f}\n".format(name, score))
                if name in ["CIDEr", "SPICE"]:
                    spider += score

        f.write("SPIDEr: {:6.4f}\n".format(spider / 2))
    pred_data = []
    for key, pred in key2pred.items():
        pred_data.append({
            "filename": key,
            "caption": "".join(pred[0]),
            "prefix": "".join(key2pred_prefix[key][0]),
        })
    json.dump({"predictions": pred_data},open(os.path.join(args.test_dir,'output.txt') , "w"), indent=4)

def main():
    print('start....')
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print('loaded tokenizer')

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str)
    parser.add_argument('--isbeam',  action='store_true')
    parser.add_argument('--magic',  action='store_true')
    parser.add_argument('--test_data', default='/home/zhangyiming/clap/clap_embedding_wavcaps/clotho/evaluation/embedding_temp.pkl')
    args = parser.parse_args()
    new_args = argparse.Namespace(**json.load(open(os.path.join(args.test_dir,"params.json"), 'r')))

    args.__dict__.update(new_args.__dict__)
    args.__dict__.update(params)

    if args.use_sound_effect:
        with open(args.sound_effect, 'rb') as sound_effect:
            audioset_labels = pickle.load(sound_effect)

        sound_effect_embeddings = list()
        for audioset_label in audioset_labels:

            sound_effect_embeddings.append(audioset_label['label_embedding'])
        sound_effect_embeddings = torch.cat(sound_effect_embeddings,dim=0).to(args.device)
    else:
        sound_effect_embeddings = None

    dataset = ClapTestDataset_withHardPrompt(args.test_data,normalize_prefix=args.normalize_prefix,sound_effect_path=args.sound_effect,sound_effect_num=args.sound_effect_num)
    prefix_dim = 1024 if args.is_rn else 512
    args.mapping_type = {'mlp':'mlp', 'transformer':'transformer'}[args.mapping_type]
    if not hasattr(args, 'only_soft_prompt'):
        model = ClapCaption_prompt(args.prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,num_layers=args.num_layers,
                                            mapping_type=args.mapping_type, only_prefix=args.only_prefix,only_soft_prompt=False)
    else:
        model = ClapCaption_prompt(args.prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,num_layers=args.num_layers,
                                            mapping_type=args.mapping_type, only_prefix=args.only_prefix,only_soft_prompt=args.only_soft_prompt)
    model.load_state_dict(torch.load(os.path.join(args.test_dir,"best.pth"),map_location="cpu"))  

    make_preds(dataset,model,tokenizer,args)


if __name__ == '__main__':

    main()