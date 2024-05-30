import pickle
import os
from tqdm import tqdm
input_dir_chatgpt = "/mnt/fast/nobackup/scratch4weeks/yz02417/zero-shot-AC/data/chatgpt/chatgpt_data.pkl"
with open(input_dir_chatgpt, 'rb') as file:
        all_data_chatgpt = pickle.load(file)

input_dir_clotho = "/mnt/fast/nobackup/scratch4weeks/yz02417/zero-shot-AC/data/clotho/train/clap_embedding/ZS/data_related.pkl"
with open(input_dir_clotho, 'rb') as file:
        all_data_clotho = pickle.load(file)

Ngrm_1 = list()
Ngrm_2 = list()

for data in all_data_clotho:
    if data['caption'][-1]=='.':
        words = data['caption'][:-1].lower().split()
    else:
        words = data['caption'].lower().split()
    Ngrm_1.extend(words)
    Ngrm_2.extend(zip(words, words[1:]))

Ngrm_1_set = set(Ngrm_1)
Ngrm_2_set = set(Ngrm_2)
output = list()
with tqdm(total=len(all_data_chatgpt), ncols=100,ascii=True) as pbar:
    for data in all_data_chatgpt:
        if data['caption'][-1]=='.':
            words = data['caption'][:-1].lower().split()
        else:
            words = data['caption'].lower().split()
        
        difference_1 = set(words) - Ngrm_1_set
        difference_2 = set(zip(words, words[1:])) - Ngrm_2_set
        # print(words)
        # print(set(words))
        # print(set(zip(words, words[1:])))
        # print(difference_1)
        if ('phone','vibrates') in Ngrm_2_set:
            print(Ngrm_2_set)
            break
        if ('phone','vibrates') in set(zip(words, words[1:])):
            print(set(zip(words, words[1:])))
            break
        # if not difference_1 and not difference_2:
        if not difference_1:
            output.append(data.copy())
        pbar.update()  
output_dir =  "/mnt/fast/nobackup/scratch4weeks/yz02417/zero-shot-AC/data/chatgpt/chatgpt_data_processsed_easy.pkl"
with open(output_dir, 'wb') as file:
        pickle.dump(output,file)