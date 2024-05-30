import os
os.environ['HF_TOKEN_PATH'] ='/mnt/fast/nobackup/scratch4weeks/yz02417/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/fast/nobackup/scratch4weeks/yz02417/huggingface'
os.environ['HF_HOME'] = '/mnt/fast/nobackup/scratch4weeks/yz02417/huggingface'
import torch
import numpy as np
import random
import time
import torch.nn.functional as F
from torch.utils.data import  DataLoader
from transformers import AdamW,get_cosine_schedule_with_warmup
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json

from pprint import pformat
sys.path.append(os.getcwd())
from models.caption_model import *
from dataset.dataset import *
from utils import *
from huggingface_hub import login
login(token="your huggingface_hub",add_to_git_credential=True)

def train(dataset: ClapDataset_Mistral_multilingual, model: ClapCaptionModel, args,valdataset,
          lr: float = 1e-5, warmup_steps: int = 500, output_dir: str = ".", output_prefix: str = ""):

    lang = {'en':'<en>',
            'zh':'<zh>',
            'fr':'<fr>',}
    
    device = torch.device(args.device)
    batch_size = args.bs
    epochs = args.epochs
    key2refs = {}

    with open(args.valdata, 'rb') as f:
        all_data = pickle.load(f)

    for item in range(len(all_data)):
        captions = post_processing(all_data[item]["caption"])
        key2refs[all_data[item]["audio_id"]] = captions

    output_dir = os.path.join(output_dir,time.strftime("%b-%d-%H-%M-%S",time.localtime()))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = genlogger(os.path.join(output_dir,"train_caption.log"))
    logger.info('Args:\n%s', pformat(vars(args)))
    args.output_dir = output_dir
    with open(os.path.join(output_dir,'params.json'), 'w') as f:
        json.dump(vars(args), f)

    crtrn_imprvd = criterion_improver("score")
    model = model.to(device)
    logger.info('Model:\n%s', pformat(vars(model)))
    model.train()
    optimizer = AdamW(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,num_workers=8,collate_fn=collate)
    test_dataloader = DataLoader(valdataset, batch_size=32, shuffle=False, drop_last=False,num_workers=8,collate_fn=collate)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps= args.warmup * len(train_dataloader), num_training_steps=epochs * len(train_dataloader))
    logger.info("{:^10}\t{:^10}\t{:^10}\t{:^10}".format(
    "Epoch", "Train loss", "Val score", "Learning rate"))

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1",add_bos_token=False,add_eos_token=False)
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        model.train()
        loss_history = []
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        for idx, (tokens, mask, prefix,padding_hard_prompt,hard_prompts_masks) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, mask, prefix,padding_hard_prompt,hard_prompts_masks = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32),padding_hard_prompt.to(device),hard_prompts_masks.to(device)            
            prefix = noise_injection(prefix,args.noise_variance)
            outputs,logits = model(tokens, prefix, padding_hard_prompt,mask,hard_prompts_masks)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens[:,3:].flatten(), ignore_index=0)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            loss_history.append(loss.item())
            progress.update()
        progress.close()

        
        
        model.eval()
        key2pred = {}
        
        with torch.no_grad(),tqdm(total=len(test_dataloader), ncols=100,ascii=True) as pbar:
            for idx, (audio_id, prefix, padding_hard_prompt,hard_prompts_masks) in enumerate(test_dataloader):
                prefix = prefix.to(device, dtype=torch.float32)
                padding_hard_prompt = padding_hard_prompt.to(device)
                embedding_hard_prompt =model.LMmodel.base_model.model.model.embed_tokens(padding_hard_prompt)
                with torch.no_grad():
                    tokens = tokenizer(lang['en'],return_tensors="pt")['input_ids'].reshape(1,-1).repeat(prefix.shape[0], 1)
                    embedding_text = model.LMmodel.base_model.model.model.embed_tokens(tokens)
                    prefix_embed, _ = model.clap_to_gpt(prefix,embedding_hard_prompt,embedding_text)

                attention_mask = torch.ones(prefix_embed.shape[:-1]).long().to(prefix_embed.device)
                generated_text_prefix = tokenizer.batch_decode(model.LMmodel.generate(inputs_embeds=prefix_embed,attention_mask=attention_mask,do_sample=False,max_length=60,eos_token_id=2,pad_token_id=2),skip_special_tokens=True)
                
                for index in range(len(audio_id)):
                    key2pred[audio_id[index]] = [generated_text_prefix[index].lower()]
                pbar.update()

        from pycocoevalcap.cider.cider import Cider
        scorers = [Cider()]
        scores_output = eval_prediction(key2refs, key2pred, scorers)
        score = scores_output["CIDEr"]
        lr = optimizer.param_groups[0]["lr"]
        train_loss = np.mean(loss_history)
        output_str = f"{epoch:^10}\t{train_loss:^10.3g}\t{score:^10.3g}\t{lr:^10.3g}"
        logger.info(output_str)

        if crtrn_imprvd(score):
            torch.save(model.state_dict(), os.path.join(output_dir,"best.pth"))
        torch.save(model.state_dict(), os.path.join(output_dir,"last.pth"))
    return model

def main():
    parser = argparse.ArgumentParser()

    #data parameter
    parser.add_argument('--data', nargs = '+',type=str,default=['/home/zhangyiming/clap/clap_embedding/clotho/development/embedding.pkl'])
    parser.add_argument('--valdata', default='/home/zhangyiming/clap/clap_embedding/audiocaps/evaluation/embedding.pkl')
    parser.add_argument('--out_dir', default='./checkpoints/clotho/base/in')
    parser.add_argument('--sound_effect', default='/mnt/fast/nobackup/scratch4weeks/yz02417/zero-shot-AC/data/audioset_label.pkl')
    parser.add_argument('--device',  default='cuda:0')
    parser.add_argument('--prefix', default='coco_prefix', help='prefix for saved filenames')
    parser.add_argument('--ckpt_file', type = str)

    #training parameter
    parser.add_argument('--bs', type=int, default=40)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--warmup',type=int, default=1)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default= 5e-6)
    parser.add_argument('--weight_decay', type=float, default=0)

    #model parameter
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--sound_effect_num', type=int, default=0)
    parser.add_argument('--noise_variance', type=float, default=0)
    parser.add_argument('--mapping_type', type=str, default='mlp', help='mlp/transformer')
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--only_soft_prompt',dest="only_soft_prompt",action='store_true')
    parser.add_argument('--use_sound_effect', dest='use_sound_effect', action='store_true')
    # parser.add_argument('--use_cross_attention', dest='use_cross_attention', action='store_true')
    parser.add_argument('--mask_probability', type=float, default=0)

    #dataset parameter
    parser.add_argument('--percentage', type=float, default=1.0)
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--use_audio_embedding', dest='use_audio_embedding', action='store_true')
    parser.add_argument('--use_related_text', dest='use_related_text', action='store_true')
 
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dataset = ClapDataset_Mistral_multilingual_withHardPrompt(args.data, args.prefix_length, normalize_prefix=args.normalize_prefix,
                           use_audio_embedding=args.use_audio_embedding, use_related_text=args.use_related_text, percentage=args.percentage,
                           mask_probability=args.mask_probability,sound_effect_num=args.sound_effect_num,sound_effect_path=args.sound_effect)
    testdataset = ClapTestDataset_Mistral_multilingual_withHardPrompt(args.valdata,normalize_prefix=args.normalize_prefix,sound_effect_path=args.sound_effect,sound_effect_num=args.sound_effect_num)

    prefix_dim = 1024 if args.is_rn else 512
    args.mapping_type = {'mlp':'mlp', 'transformer':'transformer'}[args.mapping_type]

    model = ClapCaption_Mistralai_prompt(args.prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,num_layers=args.num_layers,
                                    mapping_type=args.mapping_type,  only_prefix=args.only_prefix,only_soft_prompt=args.only_soft_prompt,islang=3)


    if args.ckpt_file != None:
        model.load_state_dict(torch.load(args.ckpt_file,map_location="cpu"))  
    train(dataset, model, args, output_dir=args.out_dir, output_prefix = args.prefix, valdataset = testdataset)

if __name__ == '__main__':
    main()