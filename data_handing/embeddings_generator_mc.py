import torch,random
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
import csv
import pandas as pd
# input_csv = '/mnt/fast/nobackup/scratch4weeks/yz02417/zero-shot-AC/data/MC/musiccaps-public.csv'


def text_preprocess(sentence):
    # for clap encoder to get clap embedding
    # # transform to lower case
    sentence = sentence.lower()

    # remove any forgotten space before punctuation and double space
    sentence = sub(r'\s([,.!?;:"](?:\s|$))', r'\1', sentence).replace('  ', ' ')

    # remove punctuations
    sentence = sub('[(,.!?;:|*\")]', ' ', sentence).replace('  ', ' ')

    return sentence

def Extract_embeddings(model,df):

    model.eval()
    # num_captions_per_audio = text_data["num_captions_per_audio"]
    # text_data = text_data["data"]
    out_data = []
    with torch.no_grad(), tqdm(total=len(df), ncols=100,ascii=True) as pbar:
        for index, row in df.iterrows():
            # print(row['ytid'])
            for sentence in row['caption'].split('. '):
                if  sentence.split()>20 or sentence.split()<5:
                    continue
                sentence = sentence.strip()
                text_embed = model.encode_text(text_preprocess(sentence)).cpu()
                item = {"caption":sentence, "text_embedding":text_embed}
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

    #mc_path
    df = pd.read_csv(os.path.join(args.dataset_path,'musiccaps-public.csv'))

    out_data = Extract_embeddings(model,df)
    with open(os.path.join(args.dataset_path,"mc_data.pkl"), 'wb') as f:
        pickle.dump(out_data,f)


if __name__ == '__main__':
    main()