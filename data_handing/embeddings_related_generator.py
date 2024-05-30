import os
import torch
import pickle
import argparse
from tqdm import tqdm
from typing import List
import torch.nn.functional as F
import numpy as np
def load_data(raw_path):
    all_data = list()
    with open(raw_path, 'rb') as f:
        all_data = all_data + pickle.load(f)

    all_captions = [raw_data['text_embedding'].cpu() for raw_data in all_data]
    unique_capions = torch.cat(list(set(all_captions)),dim=0).to('cuda')

    return F.normalize(unique_capions,dim=-1),all_data

def process_data(valid_text_embs,all_data,topnumber):
    for item in all_data:
        text_embs = F.normalize(item['text_embedding'], dim=-1).to('cuda')
        ids = torch.cosine_similarity(text_embs, valid_text_embs).topk(topnumber)[1]
        related_embs = valid_text_embs[ids.cpu()].cpu()

        item['text_embedding'] = item['text_embedding'].cpu()
        item['related_embeddings'] = related_embs

        yield item

def save_data_to_hdf5(processed_data_gen, output_path, total_items):
    # with pickle.File(output_path, 'w') as h5f:  
    with open(output_path, 'ab') as file:
        for i, item in enumerate(tqdm(processed_data_gen, total=total_items)):
            pickle.dump(item,file)
            # if i == 5:
            #     break
            # grp = h5f.create_group(str(i))
            # grp.create_dataset('text_embedding', data=item['text_embedding'])
            # grp.create_dataset('related_embeddings', data=item['related_embeddings'])
  
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help="input path files")  
    parser.add_argument('--output_path', type=str, help="output path files")  
    parser.add_argument('--topnumber', type=int, default=5)  
    args = parser.parse_args()
    valid_text_embs, all_data = load_data(args.input_path)
    processed_data_gen = process_data(valid_text_embs,all_data,args.topnumber)
    total_items = len(all_data)

    # output_hdf5_path = args.output_path + "/chatgpt_data_related.pkl"
    save_data_to_hdf5(processed_data_gen, args.output_path, total_items)
    # save_data(processed_data_gen, args.output_path, total_items)
    # extract_related_generator(args.input_path, args.output_path)

if __name__ =='__main__':
    main()