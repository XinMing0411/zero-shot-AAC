#!/usr/bin/env python3
import torch
import yaml
import sys
import os
# Add the root directory to the Python path
sys.path.append(os.getcwd())
from retrieval.models.ase_model import ASE
import pickle

from tqdm import tqdm
import argparse
import pandas as pd
from re import sub


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
    out_data = []
    with torch.no_grad(), tqdm(total=len(text_data), ncols=100,ascii=True) as pbar:
        for i in range(len(text_data)):
            sentence = text_data[i]
            
            text_embed = model.encode_text(text_preprocess(sentence)).cpu()
            item = {"label":sentence, "label_embedding":text_embed,"label_id":i}

            # print(item)
            out_data.append(item)
 
            pbar.update()
    
    return out_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="settings/extract_data.yaml", type=str,help="Setting files")
    parser.add_argument('--dataset_path', type=str, help="output path files")  
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load the Model
    model = ASE(config)
    model = model.to(config['device'])
    state_dict = torch.load(config["pretrain_path"], map_location=config['device'])
    model.load_state_dict(state_dict["model"])
    model.eval()

    df = pd.read_csv("/mnt/fast/datasets/audio/audioset/meta_data/class_labels_indices.csv")
    out_data = Extract_embeddings(model,list(df["display_name"]))

    with open(os.path.join(args.dataset_path,"audioset_label.pkl"), 'wb') as f:
        pickle.dump(out_data,f)


if __name__ == '__main__':
    main()
