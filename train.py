import os
import glob
import argparse
import json
import torch
import logging

import numpy as np
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from attrdict import AttrDict
from tensorboardX import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR

from torch.utils.data import DataLoader
from evaluate import Evaluation
from utils.eval_utils import CheckpointManager, load_checkpoint

from utils.general_utils import (
    init_logger,
    MODEL_CLASSES,
    TOKENIZER_CLASSES,
    DATASET_CLASSES,
    DATA_PROCESSORS,
    MODEL_PROCESSORS,
    TRAIN_ARGS,
    GPU_KEYS,
)


logger = logging.getLogger(__name__)


class CustomChatModel(object):
    
    def __init__(self, args):
        self.args = args
        
        self._logger = logging.getLogger(__name__)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda:" + str(self.args.gpu_ids[0]) if torch.cuda.is_available() else "cpu")
        print(self.device)
        
        init_logger()
        logger.info("Training/evaluation parameters {}".format(self.args))
        self.build_model()
        self.build_dataloader()
        self.setup_training()
        self.evaluation = Evaluation(args, self.tokenizer, self.processor, self.model_processor)
    
    def build_dataloader(self):
        # =============================================================================
        #   SETUP DATASET, DATALOADER
        # =============================================================================
        
        self.processor = DATA_PROCESSORS[self.args.task](self.args, self.tokenizer)
        self.model_processor = MODEL_PROCESSORS[self.args.model](self.args, self.tokenizer, self.device, self.processor,
                                                                 None)
        self.train_dataset = DATASET_CLASSES[self.args.model](self.args, self.tokenizer, self.processor,
                                                              self.model_processor, mode="train")
        
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.cpu_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_dataset.collate_fn
        )
        print(
            """
            # -------------------------------------------------------------------------
            #   BUILD DATALOADER DONE
            # -------------------------------------------------------------------------
            """
        )
    
    def build_model(self):
        
        self.tokenizer = TOKENIZER_CLASSES[args.model].from_pretrained(args.backbone)
        with self.tokenizer.as_target_tokenizer():
            self.tokenizer = self.tokenizer
        
        ret_orig_num_tokens = len(self.tokenizer.question_encoder)
        ret_num_added_tokens = 0
        gen_orig_num_tokens = len(self.tokenizer.generator)
        gen_num_added_tokens = 0
        
        add_tokens = gen_orig_num_tokens + gen_num_added_tokens if "rag" not in self.args.model else {
            "retriever": ret_orig_num_tokens + ret_num_added_tokens,
            "generator": gen_orig_num_tokens + gen_num_added_tokens}
        self.model = MODEL_CLASSES[args.model](self.args, self.tokenizer, add_tokens, self.device)
        self.keys_for_device = GPU_KEYS[args.model]
        self.model.to(self.device)
        
        # Use Multi-GPUs
        if -1 not in self.args.gpu_ids and len(self.args.gpu_ids) > 1:
            self.model = nn.DataParallel(self.model, self.args.gpu_ids)
        
        print(
            """
            # -------------------------------------------------------------------------
            #   BUILD MODEL DONE
            # -------------------------------------------------------------------------
            """
        )
    
    def setup_training(self):
        
        # =============================================================================
        #   optimizer / scheduler
        # =============================================================================
        
        self.iterations = len(self.train_dataset) // self.args.virtual_batch_size
        
        self.optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate, correct_bias=True)
        
        if self.args.scheduler == "lambda":
            lr_lambda = lambda epoch: self.args.lr_decay ** epoch
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        elif self.args.scheduler == "warmup":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps,
                num_training_steps=self.iterations * self.args.num_epochs
            )
        elif self.args.scheduler == "lambdalr":
            self.scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lambda epoch: 0.95 ** epoch, last_epoch=-1)
        
        # =============================================================================
        #   checkpoint_manager / tensorboard summary_writer
        # =============================================================================
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        prefix = self.args.tb_prefix
        
        if self.args.save_dirpath == 'checkpoints/':
            self.save_dirpath = os.path.join(self.args.root_dirpath, self.args.task, self.args.model_type,
                                             "%s/" % timestamp, self.args.save_dirpath)
        else:
            self.save_dirpath = self.args.save_dirpath + prefix

        tb_prefix = self.args.model_type + "_" + self.args.tb_prefix
        self.tb_writer = SummaryWriter(log_dir=f'./runs/{self.args.task}/' + timestamp + tb_prefix, comment=tb_prefix)
        
        self.checkpoint_manager = CheckpointManager(self.model, self.optimizer, self.save_dirpath)
        
        if self.args.load_pthpath == "":
            self.start_epoch = 1
        else:
            # "path/to/checkpoint_xx.pth" -> xx
            self.start_epoch = int(self.args.load_pthpath.split("_")[-1][:-4])
            self.start_epoch += 1
            model_state_dict, optimizer_state_dict = load_checkpoint(self.args.load_pthpath)
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(model_state_dict)
            else:
                self.model.load_state_dict(model_state_dict)
            self.optimizer.load_state_dict(optimizer_state_dict)
            self.previous_model_path = self.args.load_pthpath
            print("Loaded model from {}".format(self.args.load_pthpath))
        
        print(
            """
            # -------------------------------------------------------------------------
            #   SETUP TRAINING DONE
            # -------------------------------------------------------------------------
            """
        )
    
    def train(self):
        
        start_time = datetime.now().strftime('%H:%M:%S')
        print("Start train model at %s" % start_time)
        
        train_begin = datetime.utcnow()
        global_iteration_step = 0
        early_stop_count = self.args.early_stop_count
        best_valid_bleu = 0.
        
        print("Total number of iterations per epoch: %d" % self.iterations)
        print("Training...")
        lm_loss, lm_cnt = 0, 0
        kn_loss, kn_cnt = 0, 0
        ps_loss, ps_cnt = 0, 0
        
        for epoch in range(self.start_epoch, self.args.num_epochs + 1):
            self.model.train()
            tqdm_batch_iterator = tqdm(self.train_dataloader)
            accu_loss, accu_cnt = 0, 0
            accu_batch = 0
            
            for batch_idx, batch in enumerate(tqdm_batch_iterator):
                
                for b_k in batch:
                    
                    if b_k in self.keys_for_device:
                        batch[b_k] = batch[b_k].to(self.device)
                
                forward_args = TRAIN_ARGS[self.args.model](batch)
                
                output = self.model(**forward_args)
                lm_loss, knowledge_loss, persona_loss = output["lm_loss"], output["knowledge_loss"], output[
                    "persona_loss"]
                
                loss = torch.sum(
                    torch.stack([lm_loss.clone() * self.args.lm_coef, knowledge_loss.clone() * self.args.kn_coef,
                                 persona_loss.clone() * self.args.ps_coef], axis=-1)) / self.args.virtual_batch_size
                lm_loss += lm_loss.clone().item()
                kn_loss += knowledge_loss.clone().item()
                ps_loss += persona_loss.clone().item()
                
                if lm_cnt != 0:
                    self.tb_writer.add_scalar('LM_loss', lm_loss / lm_cnt, lm_cnt)
                    self.tb_writer.add_scalar('Knoweldge_loss', kn_loss / kn_cnt, kn_cnt)
                    self.tb_writer.add_scalar('Persona_loss', ps_loss / ps_cnt, ps_cnt)
                
                loss.backward()
                accu_loss += loss.item()
                accu_cnt += 1
                lm_cnt += 1
                kn_cnt += 1
                ps_cnt += 1
                accu_batch += batch["input_ids"].shape[0]
                
                if (self.args.virtual_batch_size == accu_batch) or (
                        batch_idx == (len(self.train_dataset) // self.args.batch_size)):  # last batch
                    
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_gradient_norm)
                    self.optimizer.step()
                    
                    if self.args.scheduler == "warmup":
                        self.scheduler.step()
                    
                    self.optimizer.zero_grad()
                    
                    accu_batch = 0
                    global_iteration_step += 1
                    description = "[{}][Epoch: {:3d}][Iter: {:6d}][Loss: {:6f}][lr: {:7f}]".format(
                        datetime.utcnow() - train_begin,
                        epoch,
                        global_iteration_step, accu_loss / accu_cnt,
                        self.optimizer.param_groups[0]['lr'])
                    
                    tqdm_batch_iterator.set_description(description)
                    
                    self.tb_writer.add_scalar('Training_loss', accu_loss / accu_cnt, global_iteration_step)
                
                if self.args.eval_during_training:
                    if (global_iteration_step % self.args.eval_steps == 0) and (global_iteration_step != 0):
                        print("Evaluation at %d steps" % global_iteration_step)
                        print('Evaluating...')
                        
                        metrics = self.evaluation.evaluate(self.model, epoch, 'valid')
                        print(metrics)
                
            
            # -------------------------------------------------------------------------
            #   ON EPOCH END  (checkpointing and validation)
            # -------------------------------------------------------------------------
            print("Evaluation after %d epoch" % epoch)
            print('Evaluating...')
            metrics_valid = self.evaluation.evaluate(self.model, epoch, 'valid')
            
            self.tb_writer.add_scalar('Valid_Knowledge_Accuracy', metrics_valid["k_acc"], epoch)
            self.tb_writer.add_scalar('Valid_Knowledge_Hit@2', metrics_valid["hit@2"], epoch)
            self.tb_writer.add_scalar('Valid_Knowledge_Hit@5', metrics_valid["hit@5"], epoch)
            self.tb_writer.add_scalar('Valid_Persona_Accuracy', metrics_valid["p_acc"], epoch)
            self.tb_writer.add_scalar('Valid_BLEU', metrics_valid["bleu"], epoch)
            self.tb_writer.add_scalar('Valid_ROUGE1', metrics_valid["rouge1"], epoch)
            self.tb_writer.add_scalar('Valid_ROUGE2', metrics_valid["rouge2"], epoch)
            self.tb_writer.add_scalar('Valid_ROUGEL', metrics_valid["rougel"], epoch)
            self.tb_writer.add_scalar('Valid_CharF1', metrics_valid["charf1"], epoch)
            self.tb_writer.add_scalar('Valid_UniF1', metrics_valid["unif1"], epoch)
            

            torch.cuda.empty_cache()
            if best_valid_bleu < metrics_valid["bleu"]:
                # remove previous checkpoint
                for ckpt in glob.glob(os.path.join(self.checkpoint_manager.ckpt_dirpath, "*.pth")):
                    os.remove(ckpt)
                
                self.checkpoint_manager.step(epoch)
                self.previous_model_path = os.path.join(self.checkpoint_manager.ckpt_dirpath,
                                                        "checkpoint_%d.pth" % (epoch))
                
                self._logger.info(self.previous_model_path)
                best_valid_bleu = metrics_valid["bleu"]
                early_stop_count = self.args.early_stop_count  # reset early stop count
            
            else:
                if self.args.scheduler == "lambda":
                    self.scheduler.step()  # learning rate decay
                
                # early stopping
                early_stop_count -= 1
                
                if early_stop_count == 0:
                    break
        
        model_state_dict, optimizer_state_dict = load_checkpoint(
            glob.glob(os.path.join(self.checkpoint_manager.ckpt_dirpath, "*.pth"))[0])
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)
    
    def evaluate(self):
        start_time = datetime.now().strftime('%H:%M:%S')
        self.eval_result_writer = open(os.path.join(self.checkpoint_manager.ckpt_dirpath, "eval_results.txt"), "w",
                                       encoding='utf-8')
        print("Start evaluating model at %s" % start_time)
        
        epoch = 0
        print("Evaluation after %d epoch" % epoch)

        self.start_epoch = int(self.args.load_pthpath.split("_")[-1][:-4])
        self.start_epoch += 1
        model_state_dict, optimizer_state_dict = load_checkpoint(self.args.load_pthpath)
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)
        self.previous_model_path = self.args.load_pthpath
        print("Loaded model from {}".format(self.args.load_pthpath))

        self.model.eval()
        metrics = self.evaluation.evaluate(self.model, epoch, "valid")
        print(metrics)
        
        torch.cuda.empty_cache()
        
    def inference(self):
        print(glob.glob(os.path.join(self.checkpoint_manager.ckpt_dirpath, "*.pth")))
        
        model_state_dict, optimizer_state_dict = load_checkpoint(
            glob.glob(os.path.join(self.checkpoint_manager.ckpt_dirpath, "*.pth"))[0])
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)
        
        print("Saving Evaluation Results in ",
              os.path.join(self.checkpoint_manager.ckpt_dirpath, "eval_results.txt"))
        
        print("Total number of iterations per epoch: %d" % self.iterations)
        print("Inference..")
        
        self.model.eval()
        self.evaluation.inference(self.model, 'inf')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Customized Chat (PyTorch)")
    arg_parser.add_argument("--config_dir", dest="config_dir", type=str, default="config", help="Config Directory")
    arg_parser.add_argument("--config_file", dest="config_file", type=str, default="bart-base",
                            help="Config json file")
    arg_parser.add_argument("--task", dest="task", type=str, default="customchat", help="Task")
    arg_parser.add_argument("--mode", dest="mode", type=str, default="train")
    arg_parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=6.25e-5, help="learning rate")
    arg_parser.add_argument("--valid_dataset", dest="valid_dataset", type=str, default="")
    arg_parser.add_argument("--task_desc", dest="task_desc", type=str, default="")
    arg_parser.add_argument("--root_dirpath", dest="root_dirpath", type=str, default="", help="Root directory path")
    arg_parser.add_argument("--load_pthpath", dest="load_pthpath", type=str, default="", help="Checkpoint path")
    
    parsed_args = arg_parser.parse_args()
    
    # Read from config file and make args
    with open(os.path.join(parsed_args.config_dir, "{}.json".format(parsed_args.config_file))) as f:
        args = AttrDict(json.load(f))
        args.update({"root_dirpath": parsed_args.root_dirpath})
        args.update({"mode": parsed_args.mode})
    
    # print("Training/evaluation parameters {}".format(args))
    cc_model = CustomChatModel(args)
    if parsed_args.mode == 'train':
        cc_model.train()
    elif parsed_args.mode == "inf":
        cc_model.inference()
    else:
        cc_model.evaluate()
