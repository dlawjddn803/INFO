import os
import json
import logging
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from utils.general_utils import DATASET_CLASSES, GPU_KEYS
from datasets import load_metric
from rouge_score import rouge_scorer

from torchmetrics import Accuracy

logger = logging.getLogger(__name__)
from transformers import (
    RagConfig
)

map_config = {
    'rag-tok-ct': RagConfig,
}


def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()

class Evaluation(object):
    def __init__(self, args, tokenizer, data_processor, model_processor):
        
        self.args = args
        self.tokenizer = tokenizer
        self.processor = data_processor
        self.model_processor = model_processor
        self.device = torch.device("cuda:" + str(self.args.gpu_ids[0]) if torch.cuda.is_available() else "cpu")
        self.keys_for_device = GPU_KEYS[args.model]

        self.valid_dataset = DATASET_CLASSES[self.args.model](self.args, self.tokenizer, self.processor,
                                                            self.model_processor, mode="valid")
        self.valid_dataloader = DataLoader(
            self.valid_dataset,
            batch_size=self.args.valid_batch_size,
            num_workers=self.args.cpu_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=self.valid_dataset.collate_fn
        )
        
        self.test_dataset = DATASET_CLASSES[self.args.model](self.args, self.tokenizer, self.processor,
                                                             self.model_processor, mode="test")
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.args.valid_batch_size,
            num_workers=self.args.cpu_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=self.test_dataset.collate_fn
        )
        self.data_map = {
            "valid": self.valid_dataloader,
            "test": self.test_dataloader,
        }
        self.dataset_map = {
            "valid": self.valid_dataset,
            "test": self.test_dataset,
        }
        if self.args.inference_path != "":
            self.inf_dataset = DATASET_CLASSES[self.args.model](self.args, self.tokenizer, self.processor,
                                                                 self.model_processor, mode="inf")
            self.inf_dataloader = DataLoader(
                self.inf_dataset,
                batch_size=self.args.valid_batch_size,
                num_workers=self.args.cpu_workers,
                shuffle=False,
                drop_last=False,
                collate_fn=self.inf_dataset.collate_fn
            )
            self.data_map["inf"] = self.inf_dataloader
            self.dataset_map["inf"] = self.inf_dataset
        
        self.bert_config = map_config[self.args.model].from_pretrained(
            self.args.backbone,
        )
        
        self.p_accuracy = Accuracy(num_classes=2, multiclass=True).to(self.device)
        self.k_accuracy = Accuracy(num_classes=10).to(self.device)
        self.chrf_metric = load_metric("chrf")
        self.rouge = load_metric('rouge')
        # self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bleu_metric = load_metric("sacrebleu")
        self.f1_p = load_metric("f1")
        self.f1_uni = load_metric("chrf")

    def evaluate(self, model, epoch, typ):
        metrics = self.evaluate_rag(model, epoch, typ)
        return metrics
    
    def inference(self, model, typ):
        self.inference_rag(model, typ)

    def evaluate_rag(self, model, epoch, typ='valid'):
        self.p_accuracy.reset()
        self.k_accuracy.reset()
    
        bleu = 0
        charf = 0
        rouge1 = 0
        rouge2 = 0
        rougel = 0
        h2 = 0
        h5 = 0
        f1_persona = 0
        unif1 = 0
    
        logger.info("Starting Evaluation %s" % typ)
        model.eval()
        with torch.no_grad():
            os.makedirs(os.path.join(os.path.dirname(self.args.save_dirpath), self.args.tb_prefix),
                        exist_ok=True)
            with open(os.path.join(os.path.dirname(self.args.save_dirpath), self.args.tb_prefix,
                                   typ + f'_qualitative_results_{epoch}.json'), 'w',
                      newline='') as fw:
                tqdm_batch_iterator = tqdm(self.data_map[typ])
                qual_outputs = []
                for batch_idx, batch in enumerate(tqdm_batch_iterator):
                    for b_k in batch:
                        if b_k in self.keys_for_device:
                            batch[b_k] = batch[b_k].to(self.device)
                    gold = batch["reply"]
                    k_label = batch["knowledge_grounding"]
                    p_label = batch["persona_grounding"]
                    persona_pred, k_index, r2_indices, r5_indices, pred= model.evaluate(batch)
                    
                    text_pred = pred
                    text_target = self.tokenizer.decode(gold[0], skip_special_tokens=True)
                    
                    if text_target != "":
                        self.k_accuracy.update(k_index[0], k_label)
                        self.p_accuracy.update(persona_pred, p_label)
                        f1_persona += self.f1_p.compute(predictions=persona_pred[0], references=p_label[0])["f1"]
                        if k_label.item() in r2_indices[0].detach().tolist():
                            h2 += 1
                        if k_label.item() in r5_indices[0].detach().tolist():
                            h5 += 1
                        
                        bleu += self.bleu_metric.compute(predictions=[text_pred], references=[[text_target]])['score']
                        charf += self.chrf_metric.compute(predictions=[text_pred], references=[[text_target]])['score']
                        # r = self.rouge.score(text_pred, text_target)
                        r = self.rouge.compute(predictions=[text_pred], references=[text_target])
                        # rouge1 += r['rouge1'].fmeasure
                        # rouge2 += r['rouge2'].fmeasure
                        # rougel += r['rougeL'].fmeasure
                        rouge1 += r['rouge1'].mid.fmeasure
                        rouge2 += r['rouge2'].mid.fmeasure
                        rougel += r['rougeL'].mid.fmeasure
                        unif1 += self.f1_uni.compute(predictions=[text_pred], references=[[text_target]], word_order=1, char_order=0)['score']
                        qual_output = {
                            "dialogID": batch["dialogID"][0],
                            "landmark_link": batch["landmark_link"][0],
                            "dialog": batch["raw_dialog"][0],
                            "knowledge_pred": batch["raw_knowledge_cand"][0][k_index[0][0]],
                            "knowledge_pred_index": [k_index[0].detach().tolist()],
                            "knowledge_pred_index_r2": [r2_indices.detach().tolist()],
                            "knowledge_pred_index_r5": [r5_indices.detach().tolist()],
                            "knowledge_grounding": batch["raw_knowledge_cand"][0][k_label[0]],
                            "persona_pred": persona_pred.detach().tolist(),
                            "persona_grounding": p_label.detach().tolist(),
                            "persona_candidates": batch["raw_persona_cand"],
                            "predicted_utterance": text_pred,
                            "ground_truth_utterance": text_target
                        
                        }
                        qual_outputs.append(qual_output)
            
                json.dump({
                    "qualitative_results": qual_outputs
                }, fw, indent=2)
            
                # metric on all batches using custom accumulation
                knowledge_acc_f = self.k_accuracy.compute()
                persona_acc_f = self.p_accuracy.compute()
                
                bleu4_f = bleu / len(qual_outputs)
                rouge1_f = rouge1 / len(qual_outputs)
                rouge2_f = rouge2 / len(qual_outputs)
                rougel_f = rougel / len(qual_outputs)
                charf_f = charf / len(qual_outputs)
                h2_f = h2 / len(qual_outputs)
                h5_f = h5 / len(qual_outputs)
                pf1 = f1_persona / len(qual_outputs)
                unif1_f = unif1 / len(qual_outputs)
            
                metrics = {
                    "k_acc": knowledge_acc_f.item(),
                    "p_acc": persona_acc_f.item(),
                    "p_f1": pf1,
                    "hit@2": h2_f,
                    "hit@5": h5_f,
                    "bleu": bleu4_f,
                    "rouge1": rouge1_f,
                    "rouge2": rouge2_f,
                    "rougel": rougel_f,
                    "charf1": charf_f,
                    "unif1": unif1_f,
                
                }
            
                logging.info(
                    '%s Knowledge Accuracy: %2.5f | Persona Accuracy: %2.5f | Persona F1: %2.5f |BLEU : %2.5f | ROUGE1 : %2.5f | ROUGE2 : %2.5f | ROUGEL : %2.5f | CHARF1 : %2.5f | UniF1 : %2.5f'
                    % (typ, knowledge_acc_f.item(), persona_acc_f.item(), pf1, bleu4_f,
                       rouge1_f, rouge2_f, rougel_f, charf_f, unif1_f))
        
            return metrics

    
    def inference_rag(self, model, typ='inf'):
    
        logger.info("Starting Inference %s" % typ)
        model.eval()
        with torch.no_grad():
            os.makedirs(os.path.join(os.path.dirname(self.args.save_dirpath), self.args.tb_prefix),
                        exist_ok=True)
            with open(os.path.join(os.path.dirname(self.args.save_dirpath), self.args.tb_prefix,
                                   typ + f'_qualitative_results_{0}.json'), 'w',
                      newline='') as fw:
                tqdm_batch_iterator = tqdm(self.data_map[typ])
                qual_outputs = []
                for batch_idx, batch in enumerate(tqdm_batch_iterator):
                    for b_k in batch:
                        if b_k in self.keys_for_device:
                            batch[b_k] = batch[b_k].to(self.device)
                    persona_pred, k_index, r2_indices, r5_indices, pred = model.inference(batch)
                    text_pred = pred
                
                    qual_output = {
                        "dialogID": batch["dialogID"][0],
                        "landmark_link": batch["landmark_link"][0],
                        "dialog": batch["raw_dialog"][0],
                        "knowledge_pred": batch["raw_knowledge_cand"][0][k_index[0][0]],
                        "knowledge_pred_index": [k_index[0].detach().tolist()],
                        "persona_pred": persona_pred.detach().tolist(),
                        "persona_candidates": batch["raw_persona_cand"],
                        "predicted_utterance": text_pred,
                    
                    }
                    qual_outputs.append(qual_output)
            
                json.dump({
                    "qualitative_results": qual_outputs
                }, fw, indent=2)

            
def eval_file(path, gpu_ids):
    with open(path, "r") as f:
        data = json.load(f)["qualitative_results"]
    # move the metric to device you want computations to take place
    device = "cuda:"+str(gpu_ids)
    p_accuracy = Accuracy(num_classes=2, multiclass=True).to(device)
    k_accuracy = Accuracy(num_classes=10).to(device)
    bleu_metric = load_metric("sacrebleu")
    bleu1_metric = load_metric("bleu")
    
    chrf = load_metric("chrf")
    # rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge = load_metric('rouge')
    berts = load_metric('bertscore')
    f1_p = load_metric("f1")
    f1_uni = load_metric("chrf")

    rouge1 = 0
    rouge2 = 0
    rougel = 0
    charf1 = 0
    bleu = 0
    bleu1 = 0
    bert_score = 0
    bleurt_score = 0
    f1_persona = 0
    unif1 = 0
    for qi, qual in enumerate(data):
        text_pred = qual["predicted_utterance"][0]
        text_target = qual["ground_truth_utterance"]
        if "wgt" in path:
            p_index = torch.tensor(qual["persona_pred"][0]).to(device)
            p_label = torch.tensor(qual["persona_grounding"]).to(device)
            
            k_index = torch.tensor(qual["knowledge_pred_index"][0]).to(device)
            k_label = torch.tensor([qual["knowledge_answer_index"]]).to(device)
        else:
            p_index = torch.tensor([0]).to(device)
            p_label = torch.tensor([0]).to(device)
    
            k_index = torch.tensor([0]).to(device)
            k_label = torch.tensor([0]).to(device)
        bleu += bleu_metric.compute(predictions=[text_pred], references=[[text_target]])['score']
        charf1 += chrf.compute(predictions=[text_pred], references = [[text_target]], word_order=2)['score']
        r = rouge.compute(predictions=[text_pred], references=[text_target])
        # r = rouge.score(text_pred, text_target)
        f1_persona += f1_p.compute(predictions=p_index, references=p_label)["f1"]
        unif1 += f1_uni.compute(predictions=[text_pred], references=[[text_target]], word_order=1, char_order=0)[
            'score']
        bleu1 += bleu1_metric.compute(predictions=[text_pred.split(" ")], references=[[text_target.split(" ")]], max_order=1)["bleu"]
        
        k_accuracy.update(k_index.detach(), k_label.detach())
        p_accuracy.update(p_index, p_label)
        # rouge1 += r['rouge1'].fmeasure
        # rouge2 += r['rouge2'].fmeasure
        # rougel += r['rougeL'].fmeasure
        rouge1 += r['rouge1'].mid.fmeasure
        rouge2 += r['rouge2'].mid.fmeasure
        rougel += r['rougeL'].mid.fmeasure
        bert_score += berts.compute(predictions=[text_pred], references=[text_target], lang="en")["f1"][0]

    knowledge_acc_f = k_accuracy.compute()
    persona_acc_f = p_accuracy.compute()
    
    f1_persona_f = f1_persona / len(data)
    rouge1_f = rouge1 / len(data)
    rouge2_f = rouge2 / len(data)
    rougel_f = rougel / len(data)
    bleu4_f = bleu / len(data)
    charf1_f = charf1 / len(data)
    bert_score_f = bert_score / len(data)
    bleurt_score_f = bleurt_score / len(data)
    unif1_f = unif1 / len(data)
    bleu1_f = bleu1 / len(data)
    metrics = {
        "k_acc": knowledge_acc_f.item(),
        "p_acc": persona_acc_f.item(),
        "p_f1" : f1_persona_f,
        "bleu": bleu4_f,
        "bleu1": bleu1_f,
        "rouge1": rouge1_f,
        "rouge2": rouge2_f,
        "rougel": rougel_f,
        "charf1": charf1_f,
        "bert_score": bert_score_f,
        "bleurt_score": bleurt_score_f,
        "uni_f1": unif1_f,
        

    }
    print(metrics)


if __name__ == "__main__":
    qual_path = "valid_qualitative_results_5.json"
    eval_file(qual_path, 2)