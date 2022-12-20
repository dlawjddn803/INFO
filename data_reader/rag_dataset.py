import os
import logging
import torch
from torch.utils.data import Dataset

from itertools import chain
from tqdm import tqdm
from .data_processor import CustomChatModelProcessor, SPECIAL_TOKENS


logger = logging.getLogger(__name__)


class RagDataset(Dataset):
    def __init__(self, args, tokenizer, processor, model_processor, mode):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.processor = processor
        self.mode = mode
        self.model_processor = model_processor
        
        # make cached directory under the task directory
        os.makedirs(os.path.join(args.data_dir, args.task, "cached"), exist_ok=True)
        
        if "/" in args.backbone:
            backbone = args.backbone.replace("/", "-")
            cached_file_name = f"{args.model}-{args.task_desc}-{mode}-{backbone}"
        else:
            backbone = args.backbone
            cached_file_name = f"{args.model}-{args.task_desc}-{mode}-{backbone}"
        
        cached_features_file = os.path.join(args.data_dir, args.task, "cached", cached_file_name)
        
        self.features = self.model_processor.cache_load_examples(cached_features_file, self.mode)
        print(f"Total number of features for {mode}ing: {len(self.features)}")
        self.pad_token = self.tokenizer.question_encoder.pad_token_id
        print(f"Pad Token index: {self.pad_token}")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        cur_example = self.features[index]
        feature = dict()
        feature["input_ids"] = torch.tensor(cur_example.input_ids).long()
        # print(self.tokenizer.question_encoder.batch_decode(cur_example.input_ids))
        feature["input_attn_mask"] = torch.tensor(cur_example.input_attn_mask).long()
        feature["input_eos"] = torch.tensor(cur_example.input_eos).long()
        feature["dialog"] = torch.tensor(cur_example.dialog).long()
        feature["decoder_input_ids"] = torch.tensor(cur_example.decoder_input_ids).long()
        feature["token_type_ids"] = torch.tensor(cur_example.token_type_ids).long()
        feature["lm_labels"] = torch.tensor(cur_example.lm_labels).long()
        feature["persona_candidates"] = torch.tensor(cur_example.persona_candidates).long()
        feature["persona_candidates_attn_mask"] = torch.tensor(cur_example.persona_candidates_attn_mask).long()
        feature["persona_can_idx"] = torch.tensor(cur_example.persona_can_idx).long()
        feature["persona_grounding"] = torch.tensor(cur_example.persona_grounding).long()
        feature["knowledge_candidates"] = torch.tensor(cur_example.knowledge_candidates).long()
        feature["knowledge_candidates_attn_mask"] = torch.tensor(cur_example.knowledge_candidates_attn_mask).long()
        feature["knowledge_can_idx"] = torch.tensor(cur_example.knowledge_can_idx).long()
        feature["knowledge_grounding"] = torch.tensor(cur_example.knowledge_grounding).long()
        feature["mc_token_ids"] = torch.tensor(cur_example.mc_token_ids).long()
        feature["reply"] = torch.tensor(cur_example.reply).long()
        feature["dialogID"] = cur_example.dialogID
        feature["raw_dialog"] = cur_example.raw_dialog
        feature["raw_knowledge_cand"] = cur_example.raw_knowledge_cand
        feature["raw_persona_cand"] = cur_example.raw_persona_cand
        feature["landmark_link"] = cur_example.landmark_link
        
        return feature
    
    def collate_fn(self, batch):
        merged_batch = {key: [d[key] for d in batch] for key in batch[0]}
        
        max_enc_l = max(len(x) for x in merged_batch["input_ids"])
        max_l = max(len(x) for x in merged_batch["decoder_input_ids"])
        max_l_reply = max(len(x) for x in merged_batch["reply"])
        max_l_dialog = max(len(x) for x in merged_batch["dialog"])
        
        for key in merged_batch:
            if key in ["decoder_input_ids", "lm_labels", "token_type_ids"]:
                for batch_idx, features in enumerate(merged_batch[key]):
                    pad_token = self.pad_token
                    pad_features = pad_token + torch.zeros(max_l - len(features)).long()
                    merged_batch[key][batch_idx] = torch.cat((features, pad_features), axis=0)
                merged_batch[key] = torch.stack(merged_batch[key], axis=0)
            elif key in ["input_ids", "input_attn_mask"]:
                for batch_idx, features in enumerate(merged_batch[key]):
                    pad_token = self.pad_token
                    pad_features = pad_token + torch.zeros(max_enc_l - len(features)).long()
                    merged_batch[key][batch_idx] = torch.cat((features, pad_features), axis=0)
                merged_batch[key] = torch.stack(merged_batch[key], axis=0)
            elif key in ["reply"]:
                for batch_idx, features in enumerate(merged_batch[key]):
                    pad_token = self.pad_token
                    pad_features = pad_token + torch.zeros(max_l_reply - len(features)).long()
                    merged_batch[key][batch_idx] = torch.cat((features, pad_features), axis=0)
                merged_batch[key] = torch.stack(merged_batch[key], axis=0)
            elif key in ["dialog"]:
                for batch_idx, features in enumerate(merged_batch[key]):
                    pad_token = self.pad_token
                    pad_features = pad_token + torch.zeros(max_l_dialog - len(features)).long()
                    merged_batch[key][batch_idx] = torch.cat((features, pad_features), axis=0)
                merged_batch[key] = torch.stack(merged_batch[key], axis=0)
            elif key in ["input_eos",
                         "persona_candidates", "persona_candidates_attn_mask",
                         "persona_can_idx", "persona_grounding",
                         "knowledge_candidates", "knowledge_candidates_attn_mask",
                         "knowledge_can_idx", "knowledge_grounding",
                         "mc_token_ids",
                         ]:
                for batch_idx, features in enumerate(merged_batch[key]):
                    merged_batch[key][batch_idx] = torch.tensor(features)
                merged_batch[key] = torch.stack(merged_batch[key], axis=0)
            else:
                pass
        
        return merged_batch


class RagProcessor(CustomChatModelProcessor):
    
    def get_encoded_data(self, dre_data, mode):
        
        encoded_data = []
        for data in tqdm(dre_data):
            encoded_dial = []
            for utt_id, utt in enumerate(data.utterances):
                history = utt["history"]
                
                persona_ground_enc = [1 if item == True else 0 for item in utt["persona_answer"]]
                knowledge_ground_enc = utt["knowledge_answer"]
                
                enc = {}
                
                enc["dialog_id"] = data.dial_id
                enc["raw_dialog"] = " ".join(utt["history"][:-1])
                enc["utterance_id"] = utt_id
                enc["history"] = history
                enc["persona_grounding"] = persona_ground_enc
                enc["raw_persona_cand"] = utt["persona_candidates"]
                enc["knowledge_grounding"] = knowledge_ground_enc
                enc["raw_knowledge_cand"] = utt["knowledge_candidates"]
                enc["landmark_link"] = data.landmark_link
                enc['raw_knowledge'] = [sentence for sentence in data.knowledge]
                enc['raw_history'] = utt["history"]
                
                encoded_dial.append(enc)
            encoded_data.extend(encoded_dial)
        
        return encoded_data
    
    
    def pad_multiple_candidates(self, candidates, special_token, bos, eos):
        all_cand_input_ids = []
        all_cand_attn_mask = []
        for cand in candidates:
            cand_input_ids = [bos] + cand + [special_token] + [eos]
            cand_attn_mask = len(cand_input_ids) * [1]
            candidate_padding_length = self.args.max_paragraph_len - len(cand_input_ids)
            
            if candidate_padding_length > 0:
                # Must Check
                cand_input_ids = cand_input_ids + [self.tokenizer.question_encoder.pad_token_id] * (
                        self.args.max_paragraph_len - len(cand_input_ids))
                cand_attn_mask = cand_attn_mask + [self.tokenizer.question_encoder.pad_token_id] * (
                        self.args.max_paragraph_len - len(cand_attn_mask))
            else:
                cand_input_ids = cand_input_ids[:self.args.max_paragraph_len]
                cand_attn_mask = cand_attn_mask[:self.args.max_paragraph_len]
            
            assert len(cand_attn_mask) == self.args.max_paragraph_len
            assert len(cand_input_ids) == self.args.max_paragraph_len

            all_cand_input_ids.append(cand_input_ids)
            all_cand_attn_mask.append(cand_attn_mask)
            
        return all_cand_input_ids, all_cand_attn_mask
    
    def get_encoded_features(self, example, typ):
        cls = self.tokenizer.question_encoder.cls_token_id
        sep = self.tokenizer.question_encoder.sep_token_id
        bos = self.tokenizer.generator.bos_token_id
        eos = self.tokenizer.generator.eos_token_id
        machine_st, human_st, persona_st, knowledge_st = [to[1] for to in self.tokenizer(SPECIAL_TOKENS)["input_ids"]]
        
        if typ != "inf":
            history, reply = example["history"][:-1], example["history"][-1]
        else:
            history, reply = example["history"][:-1], ""
        history = [[human_st if i % 2 == 0 else machine_st] + self.tokenizer(hist)["input_ids"][1:-1] for i, hist in
                   enumerate(history)]
        knowledge_can_enc = [self.tokenizer(sentence)["input_ids"][1:-1] for sentence in example["raw_knowledge_cand"]]
        persona_can_enc = [self.tokenizer(sentence)["input_ids"][1:-1] for sentence in example["raw_persona_cand"]]
        
        with self.tokenizer.as_target_tokenizer():
            reply = self.tokenizer(reply)["input_ids"][1:-1]
            
        reply_tti = [machine_st] * (len(reply) + 1)  # machine
        knowledge_hint =[self.tokenizer(example["landmark_link"][example["landmark_link"].index("/wiki/")+6:])["input_ids"][1:-1]] if self.args.knowledge_hint else []
        
        if len(history) != 1:
            history = [list(chain(*history))]
        if typ in ["train"]:
            enc_sequence = [[cls]] + knowledge_hint + [[sep]] + history
            dec_sequence = [bos] + reply + ([eos] if self.args.with_eos else [])
            dialog = [[cls]] + history
        else:
            enc_sequence = [[cls]] + knowledge_hint + [[sep]] + history
            dec_sequence = [bos] + [machine_st]
            reply_tti = [machine_st]
            dialog = [[cls]] + history
        all_persona_candidates, all_persona_candidates_attn_mask = self.pad_multiple_candidates(
            persona_can_enc, persona_st, cls, sep)
        all_knowledge_candidates, all_knowledge_candidates_attn_mask = self.pad_multiple_candidates(
            knowledge_can_enc, knowledge_st, cls, sep)
        
        instance = {}
        instance["input_ids"] = list(chain(*enc_sequence))
        instance["input_attn_mask"] = [1] * len(list(chain(*enc_sequence)))
        instance["input_eos"] = len(list(chain(*enc_sequence))) - 1
        instance["dialog"] = list(chain(*dialog))
        instance["raw_dialog"] = example["raw_dialog"]
        instance["decoder_input_ids"] = dec_sequence
        instance["token_type_ids"] = reply_tti
        
        if self.args.lm_labels:
            if len(dec_sequence) > 1:
                instance["lm_labels"] = dec_sequence
            else:
                instance["lm_labels"] = []
        
        instance["persona_candidates"] = all_persona_candidates
        instance["knowledge_candidates"] = all_knowledge_candidates
        instance["persona_candidates_attn_mask"] = all_persona_candidates_attn_mask
        instance["knowledge_candidates_attn_mask"] = all_knowledge_candidates_attn_mask
        instance["raw_knowledge_cand"] = example["raw_knowledge_cand"]
        instance["raw_persona_cand"] = example["raw_persona_cand"]
        
        instance["persona_can_idx"] = [len([cls] + can + [persona_st] + [sep]) - 1 for can in
                                       persona_can_enc]
        instance["persona_grounding"] = example["persona_grounding"]
        instance["knowledge_can_idx"] = [len([cls] + can + [knowledge_st] + [sep]) - 1 for can in
                                         knowledge_can_enc]
        instance["knowledge_grounding"] = example["knowledge_grounding"]
        instance["mc_token_ids"] = 0
        instance["dialogID"] = example["dialog_id"]
        instance["reply"] = reply
       
        instance["landmark_link"] = example["landmark_link"]
        
        assert len(instance["decoder_input_ids"]) == len(instance["lm_labels"])
        
        return instance
    
    def convert_data_to_features(self, encoded_data, typ):
        """Creates examples for the training and dev sets. / json files"""
        features = []
        for i, data_e in tqdm(enumerate(encoded_data)):
            inputs = self.get_encoded_features(data_e, typ)
            # if i % 1000 == 0:
            # 	logger.info(data_e)
            
            features.append(InputFeatures(
                input_ids=inputs["input_ids"],
                input_attn_mask=inputs["input_attn_mask"],
                input_eos=inputs["input_eos"],
                dialog=inputs["dialog"],
                raw_dialog=inputs["raw_dialog"],
                decoder_input_ids=inputs["decoder_input_ids"],
                token_type_ids=inputs["token_type_ids"],
                lm_labels=inputs["lm_labels"],
                persona_candidates=inputs["persona_candidates"],
                persona_candidates_attn_mask=inputs["persona_candidates_attn_mask"],
                persona_can_idx=inputs["persona_can_idx"],
                persona_grounding=inputs["persona_grounding"],
                knowledge_candidates=inputs["knowledge_candidates"],
                knowledge_candidates_attn_mask=inputs["knowledge_candidates_attn_mask"],
                knowledge_can_idx=inputs["knowledge_can_idx"],
                knowledge_grounding=inputs["knowledge_grounding"],
                mc_token_ids=inputs["mc_token_ids"],
                dialogID=inputs["dialogID"],
                reply=inputs["reply"],
                raw_knowledge_cand=inputs["raw_knowledge_cand"],
                raw_persona_cand=inputs["raw_persona_cand"],
                landmark_link=inputs["landmark_link"]
            ))
        
        return features


class InputFeatures(object):
    """A single set of features of data."""
    
    def __init__(self, input_ids, input_attn_mask, input_eos, dialog, raw_dialog, decoder_input_ids, token_type_ids,
                 lm_labels,
                 persona_candidates, persona_candidates_attn_mask,
                 persona_can_idx, persona_grounding,
                 knowledge_candidates, knowledge_candidates_attn_mask, knowledge_can_idx, knowledge_grounding,
                 mc_token_ids, dialogID, reply,
                 raw_persona_cand, raw_knowledge_cand, landmark_link):
        self.input_ids = input_ids
        self.input_attn_mask = input_attn_mask
        self.input_eos = input_eos
        self.dialog = dialog
        self.decoder_input_ids = decoder_input_ids
        self.token_type_ids = token_type_ids
        self.lm_labels = lm_labels
        self.persona_candidates = persona_candidates
        self.persona_candidates_attn_mask = persona_candidates_attn_mask
        self.persona_can_idx = persona_can_idx
        self.persona_grounding = persona_grounding
        self.knowledge_candidates = knowledge_candidates
        self.knowledge_candidates_attn_mask = knowledge_candidates_attn_mask
        self.knowledge_can_idx = knowledge_can_idx
        self.knowledge_grounding = knowledge_grounding
        self.mc_token_ids = mc_token_ids
        self.dialogID = dialogID
        self.reply = reply
        self.raw_dialog = raw_dialog
        self.raw_knowledge_cand = raw_knowledge_cand
        self.raw_persona_cand = raw_persona_cand
        self.landmark_link = landmark_link
