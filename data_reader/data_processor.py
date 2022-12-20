import os
import six
import torch
import json
import logging
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


SPECIAL_TOKENS = [" ", " ", " ", " "]


class DataProcessor(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer


class CustomChatProcessor(DataProcessor):
    def read_data_files(self, path, typ):
        dial_data = []
        datas = self.get_data(typ)
        for di, data in enumerate(datas):
            dial_data.append(self.create_input_examples(data, typ))
        
        return dial_data
    
    def get_data(self, typ):
        print("Reading %s" % typ)
        if typ == "train":
            with open(os.path.join(self.args.data_dir, "train_focus.json"), "r", encoding="utf-8") as reader:
                examples = json.load(reader)["data"]
        elif typ == "valid":
            with open(os.path.join(self.args.data_dir, "valid_focus.json"), "r", encoding="utf-8") as reader:
                examples = json.load(reader)["data"]
        elif typ == "inf":
            with open(os.path.join(self.args.inference_path), "r", encoding="utf-8") as reader:
                examples = json.load(reader)["data"]
        else:
            examples = []
        return examples
    
    def create_input_examples(self, data, typ):
        dialog_id = data["dialogID"]
        utterance = data["utterance"]
        landmark_link = data["landmark_link"]
        if typ != "inf":
            persona = data["persona"]
            knowledge = data["knowledge"]
        else:
            persona = [[] for _ in range(5)]
            knowledge = ["" for _ in range(10)]
        
        utterances = []
        for utt_id, utt in enumerate(utterance):
            # print(utt.keys())
            utt_info = dict()
            utterance_id = utt_id+1
            persona_can = utt["persona_candidate"]
            if len(persona_can) > 5:
                persona_can = persona_can[:5]
            persona_ground = utt["persona_grounding"] if typ != "inf" else [[] for _ in range(5)]
            if len(persona_ground) > 5:
                persona_ground = persona_ground[:5]
            knowledge_can = utt["knowledge_candidates"] if typ != "inf" else utt["knowledge_candidates"]
            knowledge_answer = utt["knowledge_answer_index"] if typ != "inf" else [0]
            utt_info["utterance_id"] = utterance_id
            utt_info["utterance"] = utt["dialogue"+str(utt_id+1)][-2:]
            utt_info["history"] = utt["dialogue"+str(utt_id+1)][-(2*self.args.max_history):]
            utt_info["persona_candidates"] = persona_can
            utt_info["persona_answer"] = persona_ground
            utt_info["knowledge_candidates"] = knowledge_can
            utt_info["knowledge_answer"] = knowledge_answer
            
            utterances.append(utt_info)

        d = InputExample(
        dial_id=dialog_id,
        landmark_link=landmark_link,
        dialog=utterance,
        utterances=utterances,
        persona=persona,
        knowledge=knowledge)

        return d


class InputExample(object):
    def __init__(self, dial_id, dialog, landmark_link, utterances, persona, knowledge, sentencized_kg = None 
        ):
        self.dial_id = dial_id
        self.landmark_link = landmark_link
        self.dialog = dialog
        self.utterances = utterances
        self.persona = persona
        self.knowledge = knowledge
        self.sentencized_kg  = sentencized_kg
        
class CustomChatModelProcessor(object):
    
    def __init__(self, args, tokenizer, device, data_processor, retriever=None):
        self.args = args
        self.tokenizer = tokenizer
        self.processor = data_processor
        self.retriever = retriever
    
    # for training
    def cache_load_examples(self, cached_features_file, typ):
        # Load data features from cache or dataset file

        if not os.path.exists(cached_features_file):
            print("Creating features from dataset file at %s%s" % (self.args.data_dir, self.args.task))
            try:
                cc_data = self.processor.read_data_files(
                    os.path.join(self.args.data_dir, self.args.task), typ)
                encoded_data = self.get_encoded_data(cc_data, typ)
                features = self.convert_data_to_features(encoded_data, typ)
                
                print("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)
            except ValueError:
                print("For mode, only train, valid, test, inf is available")
                raise
        else:
            print("Cached Features already exists in %s", cached_features_file)
            features = torch.load(cached_features_file)
        
        return features

    
    def get_encoded_data(self, qa_data, mode):
        raise NotImplementedError
    
    def convert_data_to_features(self, qa_data, typ):
        raise NotImplementedError
