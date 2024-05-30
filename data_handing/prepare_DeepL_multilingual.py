import argparse
import os

import deepl
import pickle
import numpy as np
from tqdm import tqdm
import time
# client = openai.OpenAI()

def generate_multilingual(selected_text,translator):

    fr_generated_text = translator.translate_text(selected_text, target_lang="FR").text
    cn_generated_text = translator.translate_text(selected_text, target_lang="ZH").text
    
    return cn_generated_text,fr_generated_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",  type=str,help="The data path of text embddedings") 
    args = parser.parse_args()
    auth_key = "Replace with your key"  # Replace with your key
    translator = deepl.Translator(auth_key)
    with open(args.data_path, 'rb') as f:
        all_data = pickle.load(f)
    out_alldata = []
    with tqdm(total=len(all_data), ncols=100,ascii=True) as pbar:
        for data in all_data:
            selected_text = data["caption"]
            out_data = data.copy()
            if isinstance(selected_text,str):
                cn_generated_text,fr_generated_text = generate_multilingual(selected_text,translator)
            else: 
                cn_generated_text = []
                fr_generated_text = []
                for text in selected_text:
                    new_text = text.copy()
                    cn_text,fr_text = generate_multilingual(new_text['caption'],translator)
                    new_text['caption'] = cn_text
                    cn_generated_text.append(new_text.copy())
                    new_text['caption'] = fr_text
                    fr_generated_text.append(new_text.copy())
            
            out_data['cn_caption'] = cn_generated_text
            out_data['fr_caption'] = fr_generated_text
            out_alldata.append(out_data)

            pbar.update()
    with open(os.path.join(os.path.dirname(args.data_path),'data_multingual.pkl'), 'wb') as file:
        pickle.dump(out_alldata, file)

if __name__ == '__main__':
    main()