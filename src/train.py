import sys
sys.path.append('../OpenPrompt')

import time
import os
import re
import copy
import argparse
import itertools
from tqdm import tqdm

import torch
import math
import numpy as np
import yaml

import prompt_gen_module
from sklearn.model_selection import train_test_split

from openprompt.pipeline_base import PromptForGeneration
from openprompt.prompts.generation_verbalizer import GenerationVerbalizer
from openprompt.data_utils import PROCESSORS
from openprompt.data_utils.utils import InputExample
from openprompt import PromptDataLoader
from openprompt.prompts import SoftTemplate

from openprompt.utils.crossfit_metrics import evaluate as crossfit_evaluate
from openprompt.utils.crossfit_metrics import METRICS


'''
                                                    Model settings
'''
parser_cfg = argparse.ArgumentParser("")

parser_cfg.add_argument("--cfg", type=str)
args_cfg, _ = parser_cfg.parse_known_args()

with open(args_cfg.cfg) as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

parser = argparse.ArgumentParser("")
parser.add_argument("--cfg", type=str)
parser.add_argument("--seed", type=int, default=cfg['train']['seed'])
parser.add_argument("--model", type=str, default=cfg['model']['model'], help="We test both t5 and t5-lm in this scripts, the corresponding tokenizerwrapper will be automatically loaded.")
parser.add_argument("--model_name_or_path", default=cfg['model']['model_name_or_path'])
parser.add_argument("--plm_eval_mode", type=bool, default=cfg['model']['plm_eval_mode'], help="whether to turn off the dropout in the freezed model.")
parser.add_argument("--use_cuda", type=bool, default=cfg['model']['use_cuda'])
parser.add_argument("--model_parallelize", type=bool, default=cfg['model']['model_parallelize'])
parser.add_argument("--tune_plm", type=bool, default=cfg['model']['tune_plm'])
parser.add_argument("--verbalizer", type=str, default=cfg['model']['verbalizer'])
parser.add_argument("--template", type=str, default=cfg['model']['template'])
parser.add_argument("--template_id", type=int, default=cfg['model']['template_id'])
parser.add_argument("--data_dir", type=str, default=cfg['dataset']['data_dir']) # sometimes, huggingface datasets can not be automatically downloaded due to network issue, please refer to 0_basic.py line 15 for solutions. 
parser.add_argument("--dataset",type=str, default=cfg['dataset']['dataset_name'])
parser.add_argument("--data_processor",type=str,default=cfg['dataset']['data_processor'])
parser.add_argument("--max_steps", type=int, default=cfg['train']['num_training_steps'])
parser.add_argument("--batch_tr", type=int, default=cfg['train']['batch_size'])
parser.add_argument("--batch_te", type=int, default=cfg['test']['batch_size'])
parser.add_argument("--eval_on_test", type=bool, default=cfg['test']['eval_on_test'])
parser.add_argument("--lr_soft_prompt", type=float, default=cfg['train']['lr_soft_prompt'])
parser.add_argument("--lr_cq_prompt", type=float, default=cfg['train']['lr_cq_prompt'])
parser.add_argument("--eval_every_steps", type=int, default=cfg['train']['eval_every_steps'])
parser.add_argument("--optimizer", type=str, default=cfg['train']['optimizer'])
parser.add_argument("--num_codes", type=int, default=cfg['train']['num_codes'])
parser.add_argument("--early_stop", type=int, default=cfg['train']['early_stop'])
parser.add_argument("--num_soft_tokens", type=int, default=cfg['prompt']['num_soft_tokens'])
parser.add_argument("--num_cq_tokens", type=int, default=cfg['prompt']['num_cq_tokens'])
parser.add_argument("--init_from_vocab", type=bool, default=cfg['prompt']['init_from_vocab'])
parser.add_argument("--result_path", type=str, default=cfg['result']['result_path'])
parser.add_argument("--comment", type=str, default="")
parser.add_argument("--temp", type=float, default=cfg['CQ']['temp'])
parser.add_argument("--num_codebook_samples", type=int, default=cfg['CQ']['num_codebook_samples'])
parser.add_argument("--commitment_cost", type=int, default=cfg['CQ']['commitment_cost'])
parser.add_argument("--identifier", type=str, default=cfg['CQ']['identifier'])

args = parser.parse_args()

if args.num_codes == -1:
    args.num_codes = args.num_cq_tokens*10

content_write = ""
content_write += f"config file: {args.cfg}\t"
content_write += f"seed: {args.seed}\t"
content_write += f"model: {args.model}\t"
content_write += f"model_name_or_path: {args.model_name_or_path}\t"
content_write += f"use_cuda: {args.use_cuda}\t"
content_write += f"model_parallelize: {args.model_parallelize}\t"
content_write += f"plm_eval_mode: {args.plm_eval_mode}\t"
content_write += f"tune_plm: {args.tune_plm}\t"
content_write += f"verbalizer: {args.verbalizer}\t"
content_write += f"init_from_vocab: {args.init_from_vocab}\t"
content_write += f"eval_every_steps: {args.eval_every_steps}\t"
content_write += f"lr_soft_prompt: {args.lr_soft_prompt}\t"
content_write += f"lr_cq_prompt: {args.lr_cq_prompt}\t"
content_write += f"batch tr: {args.batch_tr}\t"
content_write += f"batch te: {args.batch_te}\t"
content_write += f"optimizer: {args.optimizer}\t"
content_write += f"num_soft_tokens: {args.num_soft_tokens}\t"
content_write += f"num_cq_tokens: {args.num_cq_tokens}\t"
content_write += f"codebook vectors: {args.num_codes}\t"
content_write += f"early stopping patience: {args.early_stop}\t"
content_write += f"number of codebook samples: {args.num_codebook_samples}\t"
content_write += f"identifier: {args.identifier}\t"
content_write += f"comment: {args.comment}"
content_write += "\n"

print("="*20)
print("Configuration:")
print(content_write.replace('\t','\n->'))


#seed for reproduciblity
from openprompt.utils.reproduciblity import set_seed
set_seed(args.seed)



'''
                                        Initialize data-specifc items
'''
dataset = {}
Processor = PROCESSORS[args.data_processor]
dataset['train'] = Processor().get_train_examples(args.data_dir)
dataset['validation'] = Processor().get_dev_examples(args.data_dir)
dataset['test'] = Processor().get_test_examples(args.data_dir)
class_labels =Processor().get_labels()
label_tokens = [cfg['dataset']['label_tokens']]
max_seq_l = cfg['dataset']['max_seq_l']
metric_set = cfg['dataset']['metric_set']
dataset_decoder_max_length = cfg['dataset']['dataset_decoder_max_length']

print(f"\nTrain len:{len(dataset['train'])}; Valid len:{len(dataset['validation'])}; Test len:{len(dataset['test'])}")




'''
                                                    Model
'''

# use lm-adapted version or t5-v1.1 checkpoint. Note that the originial t5 checkpoint has been pretrained 
# on part of GLUE dataset, thus should not be used.
from openprompt.plms.seq2seq import T5TokenizerWrapper, T5LMTokenizerWrapper
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from openprompt.plms import load_plm


# pre-trained LM such as T5, GPT, etc.
plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)    


prompt_generator = None
if args.num_cq_tokens > 0:
    prompt_generator = prompt_gen_module.PromptGenerator


# template
args.template = os.path.normpath(os.path.join(os.getcwd(), args.template))
mytemplate = SoftTemplate(model=plm, 
                        tokenizer=tokenizer, 
                        num_soft_tokens=args.num_soft_tokens, 
                        initialize_from_vocab=args.init_from_vocab,
                        label_tokens=label_tokens,
                        num_cq_tokens=args.num_cq_tokens,
                        prompt_generator=prompt_generator,
                        task_tokens=[f"{args.dataset}"],
                        num_codes=args.num_codes,
                        temp = args.temp,
                        commitment_cost= args.commitment_cost,
                        num_codebook_samples=args.num_codebook_samples).from_file(args.template, choice=args.template_id)


# verbalizer
args.verbalizer = os.path.normpath(os.path.join(os.getcwd(), args.verbalizer))
myverbalizer = GenerationVerbalizer(tokenizer, classes=class_labels, is_rule=True).from_file(args.verbalizer)


# prompt model: plug-in everything in a complete network
prompt_model = PromptForGeneration(plm=plm,template=mytemplate, freeze_plm=(not args.tune_plm), plm_eval_mode=args.plm_eval_mode)


# shift prompt model to gpu
if args.use_cuda:
    prompt_model=  prompt_model.cuda()

    if args.model_parallelize:
        prompt_model.parallelize()


'''
                                                    data loaders
'''
train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, verbalizer=myverbalizer, tokenizer=tokenizer, # be sure to add verbalizer 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=dataset_decoder_max_length,  # be sure to use larger decoder_max_length for teacher forcing.
    batch_size=args.batch_tr,shuffle=True, teacher_forcing=True, predict_eos_token=True,  # be sure to use teacher_forcing and predict_eos_token=True
    truncate_method="tail")


validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, verbalizer=myverbalizer, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=dataset_decoder_max_length, 
    batch_size=args.batch_te,shuffle=False, teacher_forcing=False, predict_eos_token=False, # predict_eos_token=True or False are both ok 
    truncate_method="tail")


if args.eval_on_test: #false for SuperGLUE datasets
    test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, verbalizer=myverbalizer, tokenizer=tokenizer, 
        tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=dataset_decoder_max_length, 
        batch_size=args.batch_te,shuffle=False, teacher_forcing=False, predict_eos_token=False, # predict_eos_token=True or False are both ok 
        truncate_method="tail")


generation_arguments = {
    "max_length": dataset_decoder_max_length,
}





'''
                                                    evaluation function
'''
if args.eval_every_steps == -1:
    args.eval_every_steps = math.ceil(len(dataset["train"])/args.batch_tr)
    print(f"\n\n\n We will do epoch wise evaluation, i.e., at steps: {args.eval_every_steps}")
else:
    print(f"\n\n\n We will evaluate at steps: {args.eval_every_steps}")

def evaluate(prompt_model, dataloader, dataset, cluster_mode=False, phase='val'):
    
    prompt_model.eval()
    predictions = []
    ground_truths = []

    for step, inputs in enumerate(dataloader):
        if cluster_mode and step > 100:
            break
        if args.use_cuda:
            inputs = inputs.cuda()
        _, output_sentence = prompt_model.generate(inputs, **generation_arguments, verbose=False)
        
        predictions.extend(output_sentence)

        if ('meteor' in metric_set) or ('bleu' in metric_set):
            inputs['tgt_text'] = [k.split("-$$-") for k in inputs['tgt_text']]

        ground_truths.extend(inputs['tgt_text'])

    assert len(predictions)==len(ground_truths), (len(predictions), len(ground_truths))
    
    predictions = [prediction.strip() for prediction in predictions]
    
    if args.dataset == 'conll2003':
        from conll_text2bio import getonebatchresult
        inp_sentences = [e.meta['sentence'] for e in dataset]
        print(f"predictions {predictions[0:1]}, \nground_truths {ground_truths[0:1]}")
        ground_truths, predictions = getonebatchresult(inp_sentences, ground_truths, predictions)
        # shown one example
        print(f"predictions {predictions[0:1]}, \nground_truths {ground_truths[0:1]}")

    elif ('meteor' not in metric_set) and ('bleu' not in metric_set):
        ground_truths = [ground_truth.strip() for ground_truth in ground_truths]
        # shown one example
        print(f"predictions {predictions[0]}, ground_truths {ground_truths[0]}")

    else:
        # shown one example
        print(f"predictions {predictions[0]}, ground_truths {ground_truths[0]}")

    scores = dict()
    for metric in metric_set:
        score =  crossfit_evaluate(predictions, ground_truths, metric=metric)
        if args.dataset == 'conll2003':
            score = score['overall_f1']
            scores[metric] = score
        else:
            scores[metric] = score

    torch.cuda.empty_cache()
    return scores



'''
                                                        optimizer

    [Note:when lr is 0.3 with adafactor, it is the same as the configuration of https://arxiv.org/abs/2104.08691]
'''

from transformers import  AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers.optimization import Adafactor, AdafactorSchedule

tot_step = args.max_steps

optimizer_soft = None
optimizer_cq = None


if args.optimizer.lower() == "adafactor":
    soft_optimizer_parameters = [{'params': [p for name, p in prompt_model.template.named_parameters() if (('raw_embedding' not in name) and ('PromptGen' not in name))]}]     # remove the raw_embedding manually from the optimization
    optimizer_soft = Adafactor(soft_optimizer_parameters,  
                            lr=args.lr_soft_prompt,
                            relative_step=False,
                            scale_parameter=False,
                            warmup_init=False)
    scheduler_soft = get_constant_schedule_with_warmup(optimizer_soft, num_warmup_steps=0) # when num_warmup_steps is 0, it is the same as the configuration of https://arxiv.org/abs/2104.08691

    if args.num_cq_tokens > 0:
        cq_optimizer_parameters = [{'params': [p for name, p in prompt_model.template.PromptGen.named_parameters() if (('raw_embedding' not in name))]}]

        optimizer_cq = Adafactor(cq_optimizer_parameters,  
                                lr=args.lr_cq_prompt,
                                relative_step=False,
                                scale_parameter=False,
                                warmup_init=False)

        scheduler_cq = get_constant_schedule_with_warmup(optimizer_cq, num_warmup_steps=0) # when num_warmup_steps is 0, it is the same as the configuration of https://arxiv.org/abs/2104.08691




'''
                                                        training

'''
from early_stopping import EarlyStopping
delta = 1e-5
save_code_name = f"{args.dataset}_sd{args.seed}_s{args.num_soft_tokens}_v{args.num_cq_tokens}_lS{args.lr_soft_prompt}_lV{args.lr_cq_prompt}_bTr{args.batch_tr}_bTe{args.batch_te}_nC{args.num_codes}_nCS{args.num_codebook_samples}_mS{args.max_steps}_evS{args.eval_every_steps}_eS{args.early_stop}_t{args.temp}_cC{args.commitment_cost}_sE:{args.identifier}"
early_stopping = EarlyStopping(patience=args.early_stop, delta=delta, verbose=True, path=f"../ckpts/{save_code_name}.ckpt", only_save_prompt_params=True)

log_path = f"../logs/{save_code_name}.txt"
mode = 'a' if os.path.exists(log_path) else 'w'

with open(log_path, mode) as f:
    f.write("\n\n\n\n\n")
    f.write("="*50+"\n")
    f.write("Configuration:\n")
    f.write(content_write.replace('\t','\n->')+"\n")


# variables to keep track of training process
best_val_acc = 0

best_val_score_dict = dict()
loss_list = []
best_val_acc_list = []

glb_step = 1
best_glb_step = 0

val_step = 0
best_val_step = 0

tot_train_time = 0

leave_training = False

best_prompt_model = None

# epochs
for epoch in range(1000000):

    #print(f"Begin epoch {epoch}")

    for step, inputs in enumerate(train_dataloader):

        #number of backward pass
        glb_step += 1

        if args.use_cuda:
            inputs = inputs.cuda()

        tot_train_time -= time.time()

        prompt_model.train()


        print('\n\ngradient step: ', glb_step)

        loss = prompt_model(inputs)

        if args.num_cq_tokens > 0:
            print("-->Entropy loss:", loss.item())
            print("-->Aux loss:", prompt_model.template.PromptGen.loss_aux.item())
            loss += prompt_model.template.PromptGen.loss_aux
            print("-->Total loss:", loss.item())
            print("Best val till now: ", best_val_acc, " at step: ", best_glb_step)

        loss.backward()

        if optimizer_soft is not None:
            optimizer_soft.step()
            optimizer_soft.zero_grad()
            scheduler_soft.step()

        if optimizer_cq is not None:
            optimizer_cq.step()
            optimizer_cq.zero_grad()
            scheduler_cq.step()

        tot_train_time += time.time()

        if glb_step % args.eval_every_steps == 0:

            #number of validations
            val_step += 1

            print('\n\n\n validating...')
            torch.cuda.empty_cache()
            val_score_dict = evaluate(prompt_model, validation_dataloader, dataset['validation'])
            val_acc = sum(val_score_dict.values())/len(val_score_dict)
            
            print(f"\n\t\t\t\t[val acc at step {glb_step}: {val_acc}]\n")

            if val_acc > best_val_acc + delta:
                best_val_acc_list.append(f"{glb_step}:{round(val_acc,4)}")
                best_glb_step = glb_step

                best_val_step = val_step
                best_val_acc = val_acc
                best_val_score_dict = val_score_dict
                best_prompt_model = copy.deepcopy(prompt_model)

            with open(log_path, 'a') as f:
                f.write(f"\t+ Entropy loss: {loss.item()}\n")
                if args.num_cq_tokens > 0:
                    f.write(f"\t+ commitment_cost: {prompt_model.template.PromptGen.loss_aux.item()}\n")
                f.write(f"\t+ Val acc at step {glb_step}: {val_acc}\n")
                f.write(f"\t+ Best val till now: {best_val_acc} at step: {best_glb_step}\n\n")
           
            #early stopping
            early_stopping(val_acc, best_glb_step, prompt_model)

            if early_stopping.early_stop:
                leave_training = True
                print("Early stopping...")
                break

        if glb_step > args.max_steps:
            leave_training = True
            break
    
    if leave_training:
        break



'''
                                                        testing

'''                                                   
test_acc = 0
val_chck = None
test_score_dict = dict()
val_score_dict = dict()
if args.eval_on_test:
    print("testing...")
    del prompt_model
    best_prompt_model = best_prompt_model.cuda()
    best_prompt_model.parallelize()
    val_acc_to_check = evaluate(best_prompt_model, validation_dataloader, dataset['validation'])
    val_acc_to_check = sum(val_acc_to_check.values())/len(val_acc_to_check)
    test_score_dict = evaluate(best_prompt_model, test_dataloader, dataset['test'], phase='test')
    test_acc = sum(test_score_dict.values())/len(test_score_dict)
    print("Test accuracy:", test_acc)




'''
                                                        save results

''' 
print('Best score is...')
print(f"best train step: {best_glb_step} | best val step: {best_val_step} | Best Valid Acc: {best_val_acc:.3f} | Num Soft: {args.num_soft_tokens} | Num CQ: {args.num_cq_tokens}")
print(f"saved in {args.result_path}")

import os
import datetime

mode = 'a' if os.path.exists(args.result_path) else 'w'
with open(args.result_path, mode) as f:
    if args.comment != "":
        f.write(f"---> data: {args.dataset} | date-time: {datetime.datetime.now()} | Num Soft: {args.num_soft_tokens} | Num CQ: {args.num_cq_tokens} | Num Codes: {args.num_codes} | Seed: {args.seed}| lr_soft: {args.lr_soft_prompt} | lr_cq: {args.lr_cq_prompt} | val_chck: {val_chck} | Comment: {args.comment}\n")
    else:
        f.write(f"---> data: {args.dataset} | date-time: {datetime.datetime.now()} | Num Soft: {args.num_soft_tokens} | Num CQ: {args.num_cq_tokens} | Num Codes: {args.num_codes} | Seed: {args.seed} | lr_soft: {args.lr_soft_prompt} | lr_cq: {args.lr_cq_prompt} | val_chck: {val_chck} |\n")
    f.write(f"best train step: {best_glb_step} | best val step: {best_val_step} | Val dict: {best_val_score_dict} | Test dict: {test_score_dict}\n")
    f.write(f"Best Valid Acc: {best_val_acc:.3f}% | Test Acc: {test_acc:.3f}%\n")
    f.write(f"Performance trajectory: {best_val_acc_list}\n")
    f.write(f"file_name: {save_code_name}\n\n")
