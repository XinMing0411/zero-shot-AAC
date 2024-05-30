#!/usr/bin/env python3
import torch
import pickle
import sys
import os
# Add the root directory to the Python path
sys.path.append(os.getcwd())
from retrieval.models.ase_model import ASE
import json
import yaml
from tqdm import tqdm
import argparse
import json
from re import sub
import torch.nn.functional as F

def text_preprocess(sentence):
    # for clap encoder to get clap embedding
    # # transform to lower case
    sentence = sentence.lower()

    # remove any forgotten space before punctuation and double space
    sentence = sub(r'\s([,.!?;:"](?:\s|$))', r'\1', sentence).replace('  ', ' ')

    # remove punctuations
    sentence = sub('[(,.!?;:|*\")]', ' ', sentence).replace('  ', ' ')

    return sentence

def Extract_embeddings(model,text_data):

    model.eval()
    num_captions_per_audio = text_data["num_captions_per_audio"]
    text_data = text_data["data"]
    out_data = []
    with torch.no_grad(), tqdm(total=len(text_data), ncols=100,ascii=True) as pbar:
        for i in range(len(text_data)):
            if num_captions_per_audio  == 1:
                sentence = text_data[i]["caption"].strip()
                text_embed = model.encode_text(text_preprocess(sentence)).cpu()
                item = {"caption":sentence, "text_embedding":text_embed,"text_id":i}
                out_data.append(item)
            elif num_captions_per_audio == 5:
                for j in range(1, 6):
                    sentence = text_data[i]["caption_{}".format(j)].strip()
                    text_embed = model.encode_text(text_preprocess(sentence)).cpu()
                    item = {"caption":sentence, "text_embedding":text_embed,"text_id":i*5+j}
                    out_data.append(item)
            pbar.update()
    
    return out_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/mnt/fast/nobackup/scratch4weeks/yz02417/zero-shot-AC/setting/extract_data.yaml", type=str,help="Setting files")
    parser.add_argument('--dataset_path', type=str, help="input path files")  
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load the Model
    model = ASE(config)
    model = model.to(config['device'])
    state_dict = torch.load(config["pretrain_path"], map_location=config['device'])
    model.load_state_dict(state_dict["model"])
    model.eval()

    #fsd_path
    with open(os.path.join(args.dataset_path,"sb_final.json"), 'r', encoding='utf-8') as file:
        alldata = json.load(file)
    # with open(os.path.join(args.dataset_path,"chatgpt.txt"), 'r') as file:
        # alldata = file.readlines()
        # alldata = pickle.load(file)
    
    out_data = Extract_embeddings(model,alldata)
    with open(os.path.join(args.dataset_path,"sb_data.pkl"), 'wb') as f:
        pickle.dump(out_data,f)


if __name__ == '__main__':
    main()
    # python data_handing/embeddings_generator_gpt.py --config /mnt/fast/nobackup/scratch4weeks/yz02417/zero-shot-AC/setting/extract_data.yaml --dataset_path /mnt/fast/nobackup/scratch4weeks/yz02417/zero-shot-AC/data/clotho/train/
    # python data_handing/embeddings_generator_gpt.py --config /mnt/fast/nobackup/scratch4weeks/yz02417/zero-shot-AC/setting/extract_data.yaml --dataset_path /mnt/fast/nobackup/scratch4weeks/yz02417/zero-shot-AC/data/audiocaps/train/