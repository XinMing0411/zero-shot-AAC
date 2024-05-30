import argparse
import os
# from openai import OpenAI
import openai
import random
import pickle
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

def clap_score(embedding1, embedding2):
    embedding1 = embedding1.unsqueeze(1)
    embedding2 = embedding2.unsqueeze(1).transpose(0, 1)
    return F.cosine_similarity(embedding1, embedding2, dim=2).squeeze()

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

def generate_examples(text_embeddings, threshold, num_samples=5):
    selected_samples = []
    selected_indices = []

    first_index = random.randint(0, len(text_embeddings) - 1)
    selected_indices.append(first_index)
    selected_samples = text_embeddings[first_index]

    while len(selected_samples) < num_samples:
        index = random.randint(0, len(text_embeddings) - 1)
        candidate = text_embeddings[index]
        
        if torch.max(clap_score(candidate, selected_samples)) <= threshold:
            selected_indices.append(index)
            selected_samples = torch.cat((selected_samples,text_embeddings[first_index]), dim=0)


    return selected_indices

def load_text_embeddings(data_path):

    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)

    embeddings_list = list()
    text_list = list()

    for item in all_data:
        embeddings_list.append(item["text_embedding"])
        text_list.append(item["caption"])

    return embeddings_list,text_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",  type=str,help="The data path of text embddedings")
    parser.add_argument('--number', type=int, help="the number of generated captions")  
    parser.add_argument('--out_path', type=str, help="output path files")  
    args = parser.parse_args()

    generated_captions = list()
    embeddings_list,text_list = load_text_embeddings(args.data_path)
    with tqdm(total=int(args.number//20), ncols=100,ascii=True) as pbar:
        for iteration in range(int(args.number//20)):
            selected_indices = generate_examples(embeddings_list,threshold=0.3, num_samples=5)
            selected_example = [text_list[i] for i in selected_indices]

            prompt = (
                "Generate 20 sentences describing the content of the audio. "
                "Each sentence should be no more than 25 words and no less than 8 words." 
                "Each sentence should be focus solely on the audio aspect. "
                "Do not include words describing visual objects, such as size, shape, color, etc. "
                "Each sentence should describe one or several audio events. "
                "Each sentence should be in plain text without numbering."
                "I will give you five examples:"
                "\n\nExamples:\n" + "\n".join(selected_example) + "\n\nNew Captions:"
            )
            try:
                generated_text = generate_audio_captions(prompt)
                generated_captions.extend(generated_text.strip().split('\n'))
            except:
                continue
            pbar.update()
    with open(os.path.join(args.out_path,'chatgpt.pkl'), 'wb') as file:
        pickle.dump(generated_captions, file)

if __name__ == '__main__':
    main()
