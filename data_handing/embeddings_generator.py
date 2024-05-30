#!/usr/bin/env python3
import torch,random
import librosa
import yaml
import sys
import os
# Add the root directory to the Python path
sys.path.append(os.getcwd())
from retrieval.models.ase_model import ASE
import pickle
import json

from tqdm import tqdm
import argparse, math
import json
import pandas as pd
from re import sub
import torch.nn.functional as F


def text_preprocess(sentence):

    # transform to lower case
    sentence = sentence.lower()

    # remove any forgotten space before punctuation and double space
    sentence = sub(r'\s([,.!?;:"](?:\s|$))', r'\1', sentence).replace('  ', ' ')

    # remove punctuations
    sentence = sub('[(,.!?;:|*\")]', ' ', sentence).replace('  ', ' ')

    return sentence

def Extract_embeddings(model,data_path,text_or_not,config):

    h5file_df = pd.read_csv(os.path.join(data_path,"wav.csv"), sep="\t")
    h5file_dict = dict(zip(h5file_df["audio_id"], h5file_df["file_name"]))
    caption_info = json.load(open(os.path.join(data_path,"text.json"), "r"))["audios"]

    model.eval()
    out_data = []
    with torch.no_grad(), tqdm(total=len(caption_info), ncols=100,
                                ascii=True) as pbar:
        for audio_caps in caption_info:

            if not os.path.exists(h5file_dict[audio_caps["audio_id"]]):
                print(audio_caps["audio_id"])
            audio_data, _ = librosa.load(h5file_dict[audio_caps["audio_id"]], sr=config["audio_args"]["sr"], mono=True) # sample rate should be 48000
            audio_data = torch.tensor(audio_data).to(config["device"])

            if audio_data.shape[0]<=0:
                continue
            if config["audio_args"]["max_length"]!=0:

                if audio_data.shape[-1] >config["audio_args"]["max_length"]*config["audio_args"]["sr"]:
                    audio_data = audio_data[:config["audio_args"]["max_length"]*config["audio_args"]["sr"]]
                else:
                    pad_length = config["audio_args"]["max_length"]*config["audio_args"]["sr"] - audio_data.shape[-1]
                    audio_data = F.pad(audio_data, [0, pad_length], "constant", 0.0)

            audio_data = audio_data.reshape(1, -1)
            
            audio_embed = model.encode_audio(torch.tensor(audio_data).to(config["device"])).cpu()
            if text_or_not == True:

                for caption in audio_caps["captions"]:
                    text_embed = model.encode_text(text_preprocess(caption["caption"])).cpu()
                    # print(caption["caption"])
                    out_data.append({"audio_embedding":audio_embed, "caption":caption["caption"], "text_embedding":text_embed,"audio_id":audio_caps["audio_id"]})
            else:

                out_data.append({"audio_embedding":audio_embed, "caption":audio_caps["captions"], "text_embedding":0,"audio_id":audio_caps["audio_id"]})
            pbar.update()
    
    return out_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="settings/extract_data.yaml", type=str,help="Setting files")
    parser.add_argument('--dataset_path', type=str, help="input path files")  
    parser.add_argument('--out_path', type=str, help="input path files")  
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load the Model
    model = ASE(config)
    model = model.to(config['device'])
    state_dict = torch.load(config["pretrain_path"], map_location=config['device'])
    model.load_state_dict(state_dict["model"])
    model.eval()

    
    for split in ["train","val","test"]:
    # for split in ["test"]:
        data_path = os.path.join(args.dataset_path,split)
        print(f"---Extract the embeddings of {split} set---")
        out_data = Extract_embeddings(model,data_path,split=="train",config)
        with open(os.path.join(args.out_path,split,"clap_embedding","ZS","data.pkl"), 'wb') as f:
            pickle.dump(out_data,f)


if __name__ == '__main__':
    main()
    # python data_handing/embeddings_generator.py --config /mnt/fast/nobackup/scratch4weeks/yz02417/zero-shot-AC/setting/extract_data.yaml --dataset_path /mnt/fast/nobackup/scratch4weeks/yz02417/zero-shot-AC/data/clotho --out_path /mnt/fast/nobackup/scratch4weeks/yz02417/zero-shot-AC/data/clotho
    # python data_handing/embeddings_generator.py --config /mnt/fast/nobackup/scratch4weeks/yz02417/zero-shot-AC/setting/extract_data.yaml --dataset_path /mnt/fast/nobackup/scratch4weeks/yz02417/zero-shot-AC/data/audiocaps --out_path /mnt/fast/nobackup/scratch4weeks/yz02417/zero-shot-AC/data/audiocaps/