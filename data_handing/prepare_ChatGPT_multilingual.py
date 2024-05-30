import argparse
import os
from openai import OpenAI
import openai
import pickle
import numpy as np
from tqdm import tqdm
import time
def generate_audio_captions(prompt):

    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",    # specify the model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return str(e)

def generate_multilingual(selected_text):
    cn_prompt = (
                "You are an English to Chinese translator,and I will give you a sentence that you need to translate into Chinese."
                "Ensure that the meaning is the same, the grammar is accurate, and the semantics are fluent and natural." 
                "You only need to reply to the translated sentence, don't write an explanation."
                "\n\nSentence:\n" + "\n".join(selected_text) + "\n\nThe translated sentence:"
            )
    fr_prompt = (
                "You are an English to French translator,and I will give you a sentence that you need to translate into French."
                "Ensure that the meaning is the same, the grammar is accurate, and the semantics are fluent and natural." 
                "You only need to reply to the translated sentence, don't write an explanation."
                "\n\nExamples:\n" + "\n".join(selected_text) + "\n\nNew Captions:"
            )
    cn_generated_text = generate_audio_captions(cn_prompt)
    fr_generated_text = generate_audio_captions(fr_prompt)
    return cn_generated_text,fr_generated_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",  type=str,help="The data path of text embddedings") 
    args = parser.parse_args()
    with open(args.data_path, 'rb') as f:
        all_data = pickle.load(f)
    out_alldata = []
    with tqdm(total=len(all_data), ncols=100,ascii=True) as pbar:
        for data in all_data:
            # end2 = time.time()
            selected_text = data["caption"]
            selected_text = "A person is very carefully rapping a gift for someone else."
            if isinstance(selected_text,str):
                cn_generated_text,fr_generated_text = generate_multilingual(selected_text)
    
            else: 
                cn_generated_text = []
                fr_generated_text = []
                for text in selected_text:
                    cn_text,fr_text = generate_multilingual(text['caption'])

                    text['caption'] = cn_text
                    cn_generated_text.append(text.copy())
                    text['caption'] = fr_text
                    fr_generated_text.append(text.copy())
            
            out_data = data.copy()
            out_data['cn_caption'] = cn_generated_text
            out_data['fr_caption'] = fr_generated_text
            # print(out_data)
            out_alldata.append(out_data)

            pbar.update()
    with open(os.path.join(os.path.dirname(args.data_path),'data_multingual.pkl'), 'wb') as file:
        pickle.dump(out_alldata, file)

if __name__ == '__main__':
    main()