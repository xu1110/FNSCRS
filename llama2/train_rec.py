import argparse
import math
import os
import sys
import time
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import numpy as np
import torch
# import wandb
# from accelerate import Accelerator
# from accelerate.utils import set_seed
from loguru import logger
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, LlamaForSequenceClassification
# from transformers import BitsAndBytesConfig
from transformers.trainer_pt_utils import get_parameter_names
from dataset_llama import DatasetForRec

from dataloader_llama import LLaMaDataCollatorForRec
from kg_llama import KGForLLaMa
from metric import RecMetric
from utils import load_jsonl_data, simple_collate
from LlamaModels import LlamaForSequenceClassification
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
from LlamaModels import LlamaForCausalLM
from final_evaluation import FinalEvaluator



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, default='redial_rec', help="Where to store the final model.")
    
    parser.add_argument('--log_dir', type=str, default = 'redial_rec')
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    # data
    parser.add_argument("--dataset", type=str, default='./data/redial_rec', help="A file containing all data.")
    parser.add_argument("--other_dataset", type=str)
    parser.add_argument("--kg_dataset", type=str, default='./kg_data/redial')
    parser.add_argument("--shot", type=float, default=1)
    parser.add_argument("--tokenizer", type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument("--context_max_length", type=int, default=160)
    parser.add_argument("--entity_dim", type=int, default=-1)
    parser.add_argument('--num_workers', type=int, default=0)
    # model
    parser.add_argument("--model", type=str, default='meta-llama/Llama-2-7b-chat-hf')
    # optim
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16,
                        help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument('--max_grad_norm', type=float, default=1)
    parser.add_argument('--dtype', default='bfp16')
    parser.add_argument('--lora_rank', type = int, default=4)
    parser.add_argument('--mode', default='multi_turn')
    parser.add_argument('--pad_side', default='right')
    # parser.add_argument('--repeated_item_removed', type=str, default = 'True')
    parser.add_argument("--repeated_item_removed", action='store_true')
    parser.add_argument('--BARCOR', type=str, default = 'False')
    
    args = parser.parse_args()
    return args


@torch.no_grad()
def evaluation(dataloader):
    model.eval()

    loss_list = []
    for batch in tqdm(dataloader):
        batch = data_collator(batch)
        outputs = model(**batch['input'])
        loss_list.append(float(outputs['loss']))
        logits = outputs['logits'][:, item_ids]
        ranks = torch.topk(logits, k=50, dim=-1).indices
        preds = item_ids[ranks].tolist()
        labels = batch['input']['labels'].tolist()
        evaluator.evaluate(preds, labels)

    report = evaluator.report()
    report['loss'] = np.mean(loss_list)
    return report


def learning():
    # training info
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0

    # save model with best metric
    metric, mode = 'recall@50', 1
    assert mode in (-1, 1)
    if mode == 1:
        best_metric = 0
    else:
        best_metric = float('inf')
    best_metric_dir = os.path.join(args.output_dir, 'best')

    # train loop
    for epoch in range(args.num_train_epochs):
        train_loss = []

        for step, batch in enumerate(train_dataloader):
            batch = data_collator(batch)
            outputs = model(**batch['input'])
            loss = outputs['loss']

            # print(batch)
            loss.backward()
            # accelerator.backward(loss)
            train_loss.append(float(loss))
            # print(outputs['logits'])
            # optim step
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                # if args.max_grad_norm is not None:
                #     accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.max_grad_norm, norm_type=2)
                # print(optimizer.state_dict())
                optimizer.step()
                # print(optimizer.state_dict())
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                completed_steps += 1
                if run:
                    run.log({'loss': np.mean(train_loss) * args.gradient_accumulation_steps})

            if completed_steps >= args.max_train_steps:
                break

        # metric
        train_loss = np.mean(train_loss) * args.gradient_accumulation_steps
        logger.info(f'epoch {epoch} train loss {train_loss}')

        # evaluation
        # dev
        report = evaluation(valid_dataloader)
        valid_report = {}
        for k, v in report.items():
            valid_report[f'valid/{k}'] = v
        valid_report['epoch'] = epoch
        logger.info(f'{valid_report}')
        if run:
            run.log(valid_report)
        evaluator.reset_metric()

        if valid_report[f'valid/{metric}'] * mode > best_metric * mode:
            model.save_pretrained(best_metric_dir)
            best_metric = valid_report[f'valid/{metric}']
            logger.info(f'new best model with {metric}')

        # test
        report = evaluation(test_dataloader)
        test_report = {}
        for k, v in report.items():
            test_report[f'test/{k}'] = v
        test_report['epoch'] = epoch
        logger.info(f'{test_report}')
        if run:
            run.log(test_report)
        evaluator.reset_metric()
    final_dir = os.path.join(args.output_dir, 'final')
    model.save_pretrained(final_dir)
    logger.info(f'save final model')
    final_evaluator = FinalEvaluator(best_metric_dir, args)
    final_evaluator.evaluate(logger)



if __name__ == '__main__':


    args = parse_args()
    config = vars(args)
    
    # If passed along, set the training seed now.
    # if args.seed is not None:
    #     set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the accelerator. We will let the accelerator handle device placement for us.
    # accelerator = Accelerator(device_placement=False, fp16=args.fp16)
    # device = accelerator.device
    device = 'cuda'
    # Make one log on every process with the configuration for debugging.
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger.remove()
    # logger.add(sys.stderr, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    if args.log_dir == '':
        logger.add(f'log/{local_time}.log', level='DEBUG')
    else:
        logger.add(f'log/{args.log_dir}.log', level='DEBUG')
    # logger.info(accelerator.state)
    logger.info(config)


    # wandb
    run = None
    # if args.use_wandb:
    #     name = args.name if args.name else local_time
    #     run = wandb.init(entity=args.entity, project=args.project, config=config, name=name)

    # kg
    kg = KGForLLaMa(kg_dataset=args.kg_dataset, debug=args.debug).get_kg_info()
    item_ids = torch.as_tensor(kg['item_ids'], device=device)
    # model
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, padding_side = args.pad_side, token = "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")# 
    tokenizer.pad_token_id = 0

    if args.dtype == 'fp32':
        model = LlamaForSequenceClassification.from_pretrained(args.model, num_labels = kg['num_entities'], entity_dim = args.entity_dim,device_map='auto',torch_dtype=torch.float32, token = "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")# , quantization_config = bnb_config
    elif args.dtype == 'fp16':
        model = LlamaForSequenceClassification.from_pretrained(args.model, num_labels = kg['num_entities'], entity_dim = args.entity_dim,device_map='auto',torch_dtype=torch.float16, token = "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")# , quantization_config = bnb_config
    elif args.dtype == 'bfp16':
        model = LlamaForSequenceClassification.from_pretrained(args.model, num_labels = kg['num_entities'], entity_dim = args.entity_dim,device_map='auto',torch_dtype=torch.bfloat16, token = "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")# , quantization_config = bnb_config
        
    logger.info(f'dtype: {args.dtype}')
    
    target_modules = []

    for name, param in model.named_modules():
        # print(name)
        if 'score' not in name and 'layernorm' not in name:
            for layer_num in range(0,32):
                if '.{}.'.format(str(layer_num)) in name and name not in ["model.layers.{}".format(str(layer_num)), "model.layers.{}.self_attn".format(str(layer_num)), "model.layers.{}.mlp".format(str(layer_num)), "model.layers.{}.self_attn.rotary_emb".format(str(layer_num)), "model.layers.{}.mlp.act_fn".format(str(layer_num))]:
                # if '.{}.'.format(str(layer_num)) in name and ('self_attn.v_proj' in name or 'self_attn.q_proj' in name):
                    target_modules.append(name)

    peft_config = LoraConfig(
                        task_type="SEQ_CLS",
                        r=args.lora_rank,
                        lora_alpha=32,
                        lora_dropout=0.01,
                        target_modules=['v_proj', 'q_proj']
                        )

    
    model = get_peft_model(model, peft_config)
    
    model.print_trainable_parameters()
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f'tunable para:{name},rank{args.lora_rank}')

    if args.dtype == 'fp16':
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, eps = 1e-4)# 
    else:
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)# , eps = 1e-4

    # data
    train_data_file = os.path.join(args.dataset, 'train_data_processed.jsonl')
    train_data_list = load_jsonl_data(train_data_file)
    train_dataset = DatasetForRec(
        data_list=train_data_list, entity2id=kg['entity2id'],
        tokenizer=tokenizer, repeated_item_removed=args.repeated_item_removed, context_max_length=args.context_max_length,
        debug=args.debug, shot=args.shot, mode = args.mode
    )

    valid_data_file = os.path.join(args.dataset, 'valid_data_processed.jsonl')
    valid_data_list = load_jsonl_data(valid_data_file)
    valid_dataset = DatasetForRec(
        data_list=valid_data_list, entity2id=kg['entity2id'],
        tokenizer=tokenizer, repeated_item_removed=args.repeated_item_removed, context_max_length=args.context_max_length,
        debug=args.debug, mode = args.mode
    )

    test_data_file = os.path.join(args.dataset, 'test_data_processed.jsonl')
    test_data_list = load_jsonl_data(test_data_file)
    test_dataset = DatasetForRec(
        data_list=test_data_list, entity2id=kg['entity2id'],
        tokenizer=tokenizer, repeated_item_removed=args.repeated_item_removed, context_max_length=args.context_max_length,
        debug=args.debug, mode = args.mode
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=simple_collate,
        shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=simple_collate,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=simple_collate,
    )
    # train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
    #     train_dataloader, valid_dataloader, test_dataloader
    # )
    data_collator = LLaMaDataCollatorForRec(
        device=device, debug=args.debug, use_amp=True,
        tokenizer=tokenizer, context_max_length=args.context_max_length, mode = args.mode
    )

    # step, epoch, batch size
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = args.per_device_train_batch_size * 1 * args.gradient_accumulation_steps

    # lr_scheduler
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_update_steps_per_epoch, args.max_train_steps)
    # lr_scheduler = accelerator.prepare(lr_scheduler)

    # evaluator
    evaluator = RecMetric()

    learning()
