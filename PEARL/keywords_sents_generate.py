import argparse
import json
import os
import torch
import numpy as np
from tqdm import tqdm
from unsloth import FastLanguageModel

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default='profession')
parser.add_argument("--model_name", type=str, default='llama31-4bit')
parser.add_argument("--K", type=int, default=50)
parser.add_argument("--iter", type=str, default=None)
parser.add_argument("--screen", action='store_true')
parser.add_argument("--aaai_num", type=str, default=1)
parser.add_argument('--gen_len', type=int, default=512)
parser.add_argument('--part_of_dataset', type=float, default=0.33)
parser.add_argument("--num_keywords", type=int, default=60)  # how many keywords in one doc
parser.add_argument("--niter", type=int, default=50)

args = parser.parse_args()
print(args)
model_base = '/mnt/nvm_data/xjn/OLineModel'
if args.model_name == 'llama3-4bit':
    model_name = model_base + '/llama-3-8b-bnb-4bit'
elif args.model_name == 'qwen2-4bit':
    model_name = model_base + "/Qwen2-7B-bnb-4bit"
elif args.model_name == 'llama3':
    model_name = model_base + '/llama-3-8b'
elif args.model_name == 'qwen2':
    model_name = model_base + "/Qwen2-7B"
elif args.model_name == 'llama3-Instruct':
    model_name = model_base + '/llama-3-8b-Instruct'
elif args.model_name == 'qwen2-Instruct':
    model_name = model_base + "/Qwen2-7B-Instruct"
elif args.model_name == 'llama3-Instruct-4bit':
    model_name = model_base + '/llama-3-8b-Instruct-bnb-4bit'
elif args.model_name == 'qwen2-Instruct-4bit':
    model_name = model_base + "/Qwen2-7B-Instruct-bnb-4bit"
elif args.model_name == 'llama31':
    model_name = model_base + '/llama-31-8b'
elif args.model_name == 'llama31-4bit':
    model_name = model_base + '/llama-31-8b-bnb-4bit'
elif args.model_name == 'llama31-Instruct':
    model_name = model_base + '/llama-31-8b-Instruct'
elif args.model_name == 'llama31-Instruct-4bit':
    model_name = model_base + '/llama-31-8b-Instruct-bnb-4bit'
dataset_name = args.dataset_name

if args.iter is None:
    s = f'info_{args.niter}_{args.num_keywords}'
else:
    s = 'info_' + args.model_name + '_' + args.iter + '_' + str(args.niter) + '_' + str(args.num_keywords) + '_' + str(args.K) + '_' + str(args.gen_len) + '_' + str(args.part_of_dataset)
print(f'####### using model: {args.model_name} ## using voca-txt: {s} #######')
word_list = np.loadtxt('./data/' + dataset_name + f'/{s}/voca.txt', delimiter='\t', dtype='str', usecols=1)
with open(os.path.join('.', 'data', dataset_name, 'classes.txt'), mode='r', encoding='utf-8') as f:
    class_names = "".join(f.readlines()).strip().split("\n")
with open(os.path.join('.', 'data', dataset_name, 'dataset.txt'), mode='r', encoding='utf-8') as f:
    dataset_len = len(f.readlines())
w2w_sim = np.loadtxt('./data/' + dataset_name + f'/{s}/model/k{len(class_names)}.{args.aaai_num}.pw_z')
idx = np.argpartition(w2w_sim, -args.K, axis=1)[:, -args.K:]
arr = word_list[idx]
keyword_set = {class_names[i]: arr[i] for i in range(len(class_names))}
print('current dataset:', dataset_name)
print('pw_z: ', f'k{len(class_names)}.{args.aaai_num}.pw_z')
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,  # 模型路径
    max_seq_length=1024,  # 可以设置为任何值内部做了自适应处理
    dtype=torch.bfloat16,  # 数据类型使用float16
    load_in_4bit=True if '4bit' in args.model_name else False,  # 使用4bit量化来减少内存使用
)
alpaca_prompt = """The following is a description of the task, accompanied by input that provides further background information. Write a response that appropriately completes the request.
### Instruction:
{}

### Input:
{}

### Response:
{}"""

# EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # 选择任何大于0的数字！建议使用8、16、32、64、128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", ],
    lora_alpha=16,
    lora_dropout=0,  # 支持任何值，但等于0时经过优化
    bias="none",  # 支持任何值，但等于"none"时经过优化
    use_gradient_checkpointing="unsloth",  # True或"unsloth"适用于非常长的上下文
    random_state=3407,
    use_rslora=False,  # 支持排名稳定的LoRA
    loftq_config=None,  # 和LoftQ
)
FastLanguageModel.for_inference(model)  # 启用原生推理速度快2倍
result = {"text": [], "label": []}
num = int(dataset_len / len(class_names) * args.part_of_dataset)
# num = 10
batch_num = 150
its = num / batch_num

print('gen_len:', args.gen_len)
for it in range(int(its) + 1):
    gen_num = batch_num if batch_num * (it + 1) < num else num - batch_num * it
    if gen_num == 0:
        break
    loop = tqdm(class_names, total=len(class_names))
    for name in loop:
        if 'News' in dataset_name:
            instruction = f"Please use the style of internet press release to generate a natural language text in English using the words in the given word list as far as possible. It is required to reflect that the theme of the news is {name}. The word list is as follows:"
        elif dataset_name == 'hobby' or dataset_name == 'profession':
            instruction = f"Please use the style of internet community comments to generate a natural language text in English using the words in the given word list as far as possible. It is required to reflect that the commenter\'s {args.dataset_name} is {name}. The word list is as follows:"
        else:
            instruction = f"Please generate a natural language text in English using the words in the given word list as far as possible. It is required to reflect that the theme of the text is {name}. The word list is as follows:"
        inputs = tokenizer(
            [
                alpaca_prompt.format(
                    # Instruction
                    instruction,
                    # input
                    ", ".join(keyword_set[name]),
                    # output
                    "",
                )
            ], return_tensors="pt", add_special_tokens=True).to("cuda")
        outputs = model.generate(**inputs,
                                 max_new_tokens=args.gen_len,
                                 # max_length=200,
                                 use_cache=True,
                                 num_return_sequences=gen_num,
                                 pad_token_id=tokenizer.pad_token_id,
                                 top_k=100,
                                 top_p=0.8,
                                 temperature=0.95,
                                 do_sample=True,
                                 )
        result['label'].extend([name] * gen_num)
        decode_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_texts = [text.split("Response:")[1].strip() if "Response:\n" in text else text for text in decode_text]
        result['text'].extend(output_texts)
        loop.set_description(f'Iter [{it}/{int(its)}]'+ f'##gen num [{gen_num*(it+1) if gen_num == batch_num else batch_num*it+gen_num}/{num}]')

if args.iter == None:
    with open(os.path.join('.', 'data', dataset_name, f'df_keywords_gen_{dataset_name}_{args.model_name}_1_{args.niter}_{args.num_keywords}_{str(args.K)}_{str(args.gen_len)}_{str(args.part_of_dataset)}.json'), 'w') as f:
        json.dump(result, f)
        print(f'####### generate: df_keywords_gen_{dataset_name}_{args.model_name}_1_{args.niter}_{args.num_keywords}_{str(args.K)}_{str(args.gen_len)}_{str(args.part_of_dataset)}.json #######')
else:
    with open(os.path.join('.', 'data', dataset_name, f'df_keywords_gen_{dataset_name}_{args.model_name}_{str(int(args.iter)+1)}_{args.niter}_{args.num_keywords}_{str(args.K)}_{str(args.gen_len)}_{str(args.part_of_dataset)}.json'), 'w') as f:
        json.dump(result, f)
        print(f'####### generate: df_keywords_gen_{dataset_name}_{args.model_name}_{str(int(args.iter)+1)}_{args.niter}_{args.num_keywords}_{str(args.K)}_{str(args.gen_len)}_{str(args.part_of_dataset)}.json #######')

