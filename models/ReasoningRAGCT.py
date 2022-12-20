import torch
from torch import nn

from torch.nn import Sigmoid, Softmax

from transformers import (
    BertConfig,
    BertModel,
    BertTokenizer,
    BertPreTrainedModel,
    RagConfig,
    RagSequenceForGeneration,
    RagTokenForGeneration,
    RagRetriever
)
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss


class PolyEncoder(BertPreTrainedModel):
    def __init__(self, args, config, bert, device):
        super().__init__(config)
        self.config = config
        self.args = args
        self.bert = bert.to(device)
        self.poly_m = self.args.poly_m
        self.poly_code_embeddings = nn.Embedding(self.poly_m, self.config.hidden_size)
        # https://github.com/facebookresearch/ParlAI/blob/master/parlai/agents/transformer/polyencoder.py#L355
        torch.nn.init.normal_(self.poly_code_embeddings.weight, self.config.hidden_size ** -0.5)
    
    def dot_attention(self, q, k, v):
        # q: [bs, poly_m, dim] or [bs, res_cnt, dim]
        # k=v: [bs, length, dim] or [bs, poly_m, dim]
        attn_weights = torch.matmul(q, k.transpose(2, 1))  # [bs, poly_m, length]
        attn_weights = F.softmax(attn_weights, -1)
        output = torch.matmul(attn_weights, v)  # [bs, poly_m, dim]
        return output
    
    def forward(self, context_input_ids, context_input_masks,
                responses_input_ids, responses_input_masks, labels=None):
        if labels is not None:
            responses_input_ids = responses_input_ids[:, 0, :].unsqueeze(1)
            responses_input_masks = responses_input_masks[:, 0, :].unsqueeze(1)
        
        batch_size, res_cnt, seq_length = responses_input_ids.shape  # res_cnt is 1 during training
        
        ctx_out = self.bert(context_input_ids, context_input_masks)  # [bs, length, dim]
        ctx_out = ctx_out[0]
        
        poly_code_ids = torch.arange(self.poly_m, dtype=torch.long).to(context_input_ids.device)
        poly_code_ids = poly_code_ids.unsqueeze(0).expand(batch_size, self.poly_m)
        poly_codes = self.poly_code_embeddings(poly_code_ids)  # [bs, poly_m, dim]
        embs = self.dot_attention(poly_codes, ctx_out, ctx_out)  # [bs, poly_m, dim]
        
        # response encoder
        responses_input_ids = responses_input_ids.view(-1, seq_length)
        responses_input_masks = responses_input_masks.view(-1, seq_length)
        
        cand_emb = self.bert(responses_input_ids, responses_input_masks)  # [bs, dim]
        cand_emb = cand_emb[0][:, 0, :]
        cand_emb = cand_emb.view(batch_size, res_cnt, -1)  # [bs, res_cnt, dim]
        
        # merge
        if labels is not None:
            cand_emb = cand_emb.permute(1, 0, 2)  # [1, bs, dim]
            cand_emb = cand_emb.expand(batch_size, batch_size, cand_emb.shape[2])  # [bs, bs, dim]
            ctx_emb = self.dot_attention(cand_emb, embs, embs).squeeze()  # [bs, bs, dim]
            dot_product = (ctx_emb * cand_emb).sum(-1)  # [bs, bs]
            mask = torch.eye(batch_size).to(context_input_ids.device)  # [bs, bs]
            loss = F.log_softmax(dot_product, dim=-1) * mask
            loss = (-loss.sum(dim=1)).mean()
            return loss
        else:
            ctx_emb = self.dot_attention(cand_emb, embs, embs)  # [bs, res_cnt, dim]
            dot_product = (ctx_emb * cand_emb).sum(-1)
            return dot_product


map_model = {
    'rag-seq': RagSequenceForGeneration,
    'rag-tok': RagTokenForGeneration,
    'rag-tok-ct': RagTokenForGeneration,
}
model_mapper = {
    "rag-seq": "rag",
    "rag-tok": "rag",
    "rag-tok-ct": "rag",
}

encoder_mapper = {
    "poly": PolyEncoder
}
class ReasoningRAGCT(nn.Module):
    def __init__(self, args, tokenizer, num_tokens_to_add, device):
        super(ReasoningRAGCT, self).__init__()
        self.args = args
        self.config = RagConfig.from_pretrained(args.backbone)
        self.device = device
        self.tokenizer = tokenizer
        
        self.bert_config = BertConfig.from_pretrained(self.args.bert_model)
        self.bert_model = BertModel.from_pretrained(self.args.bert_model).to(self.device)
        self.retriever = RagRetriever.from_pretrained(self.args.backbone,
                                                      index_name="custom",
                                                      passages_path=self.args.knowledge_dataset_path,
                                                      index_path=self.args.knowledge_index_path,
                                                      )
        
        self.backbone_model = map_model[self.args.model].from_pretrained(self.args.backbone, retriever=self.retriever, output_retrieved=True).to(self.device)
        self.kn_poly_encoder = encoder_mapper[self.args.encoder](self.args, self.bert_config, self.bert_model, self.device)
        self.pe_poly_encoder = encoder_mapper[self.args.encoder](self.args, self.bert_config, self.bert_model, self.device)
        self.pe_controller = self.bert_model
        self.pe_predictor = nn.Linear(self.bert_model.config.hidden_size, self.args.persona_num)
        self.sigmoid = Sigmoid()
        self.softmax = Softmax()
        
        self.kn_loss_fct = CrossEntropyLoss()
        self.pe_loss_fct = BCEWithLogitsLoss(pos_weight=torch.tensor(6))
    
    def build_ce_input(self, bsz, cand_num, input_ids, cand_input_ids, cand_attn_mask):
        cross_input_ids = []
        cross_attn_mask = []
        cross_token_type_ids = []
        max_seq_len = max([len(s) for s in input_ids])
        for bs in range(bsz):
            input_ids_bs = input_ids[bs, :]
            input_ids_bs = input_ids_bs[input_ids_bs != self.tokenizer.question_encoder.pad_token_id].tolist()
            attn_mask_bs = len(input_ids_bs) * [1]
            token_type_ids = len(input_ids_bs) * [0]
            
            candidate_padding_length = max_seq_len - len(input_ids_bs)
            if candidate_padding_length > 0:
                # Must Check
                new_input_ids_bs_f = input_ids_bs + [self.tokenizer.question_encoder.pad_token_id] * (
                        max_seq_len- len(input_ids_bs))
                attn_mask_bs_f = attn_mask_bs + [self.tokenizer.question_encoder.pad_token_id] * (
                        max_seq_len - len(attn_mask_bs))
                token_type_ids_f = token_type_ids + [self.tokenizer.question_encoder.pad_token_id] * (
                        max_seq_len - len(token_type_ids))
            else:
                new_input_ids_bs_f = input_ids_bs[:max_seq_len]
                attn_mask_bs_f = attn_mask_bs[:max_seq_len]
                token_type_ids_f = token_type_ids[:max_seq_len]

            assert len(new_input_ids_bs_f) == max_seq_len
            assert len(attn_mask_bs_f) == max_seq_len
            assert len(token_type_ids_f) == max_seq_len

            input_ids = torch.tensor(new_input_ids_bs_f).long().to(self.device)
            attn_mask = torch.tensor(attn_mask_bs_f).long().to(self.device)
            token_type_ids = torch.tensor(token_type_ids_f).long().to(self.device)

            input_ids = input_ids.repeat([cand_num, 1])
            attn_mask = attn_mask.repeat([cand_num, 1])
            token_type_ids = token_type_ids.repeat([cand_num, 1])
            
            cand_input_ids_bs = cand_input_ids[bs, :, 1:]
            cand_attn_mask_bs = cand_attn_mask[bs, :, 1:]
            cand_token_type_ids = torch.ones([cand_num, cand_input_ids_bs.size()[1]], dtype=int).to(self.device)
            
            new_input_ids_bs = torch.cat([input_ids, cand_input_ids_bs], dim=1)
            new_attn_mask_bs = torch.cat([attn_mask, cand_attn_mask_bs], dim=1)
            new_token_type_ids_bs = torch.cat([token_type_ids, cand_token_type_ids], dim=1)
            
            cross_input_ids.append(new_input_ids_bs)
            cross_attn_mask.append(new_attn_mask_bs)
            cross_token_type_ids.append(new_token_type_ids_bs)
    
        cross_input_ids = torch.stack(cross_input_ids).to(self.device)
        cross_attn_mask = torch.stack(cross_attn_mask).to(self.device)
        cross_token_type_ids = torch.stack(cross_token_type_ids).to(self.device)
        
        return cross_input_ids, cross_attn_mask, cross_token_type_ids
    
    def forward(self, input_ids, input_attn_mask, decoder_input_ids,
                persona_input_ids, persona_attn_mask,
                knowledge_input_ids, knowledge_attn_mask,
                persona_grounding, knowledge_grounding,
                lm_labels,
                ):
                
        outputs = {}
        bsz = persona_input_ids.size()[0]
        kn_logits = self.kn_poly_encoder(context_input_ids=input_ids,
                                         context_input_masks=input_attn_mask,
                                         responses_input_ids=knowledge_input_ids,
                                         responses_input_masks=knowledge_attn_mask, )

        kn_loss = self.kn_loss_fct(kn_logits, knowledge_grounding)
        kn_pred_idx = torch.argmax(kn_logits, 1)
        kn_sel_input_ids = knowledge_input_ids[torch.arange(knowledge_input_ids.size(0)), kn_pred_idx]

        pe_logits = self.pe_poly_encoder(context_input_ids=input_ids,
                                         context_input_masks=input_attn_mask,
                                         responses_input_ids=persona_input_ids,
                                         responses_input_masks=persona_attn_mask)

        pe_loss = self.pe_loss_fct(pe_logits, persona_grounding.float())
        
        
        persona_num_logits = self.pe_controller(
            input_ids=input_ids,
            attention_mask=input_attn_mask,
            return_dict=True,
        )["pooler_output"]
        persona_num_pred_logit = self.pe_predictor(persona_num_logits)
        persona_num_pred = self.softmax(persona_num_pred_logit)
        
        persona_num = torch.argmax(persona_num_pred, 1).detach().tolist()
        pe_sel_input_ids_b = []
        
        for bs in range(bsz):
            if persona_num[bs] != 0:
                pe_pred_idx = torch.topk(pe_logits[bs, :], persona_num[bs])[1]
                pe_sel_input_ids_b_f = persona_input_ids[bs, :, :][pe_pred_idx]
                pe_sel_input_ids_b_f = pe_sel_input_ids_b_f[
                    pe_sel_input_ids_b_f != self.tokenizer.question_encoder.pad_token_id].detach().tolist()
            else:
                pe_sel_input_ids_b_f = []
            # padding again
            candidate_padding_length = self.args.max_paragraph_len - len(pe_sel_input_ids_b_f)
            if candidate_padding_length > 0:
                # Must Check
                pe_sel_input_ids_b_f = pe_sel_input_ids_b_f + [self.tokenizer.question_encoder.pad_token_id] * (
                        self.args.max_paragraph_len - len(pe_sel_input_ids_b_f))
            else:
                pe_sel_input_ids_b_f = pe_sel_input_ids_b_f[:self.args.max_paragraph_len]
            
            assert len(pe_sel_input_ids_b_f) == self.args.max_paragraph_len
            pe_sel_input_ids_b.append(torch.tensor(pe_sel_input_ids_b_f).long())
        
        pe_sel_input_ids = torch.stack(pe_sel_input_ids_b).to(self.device)
        
        # Add Selected Persona and Knowledge
        new_input = torch.cat((input_ids[:, -312:], pe_sel_input_ids, kn_sel_input_ids), 1)

        question_hidden_states = self.backbone_model.question_encoder(new_input)[0]

        docs_dict = self.retriever(new_input.detach().cpu().numpy(), question_hidden_states.detach().cpu().numpy(),
                                   return_tensors="pt")
        docs_dict = {dd: docs_dict[dd].to(self.device) for dd in docs_dict}
        doc_scores = torch.bmm(
            question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)).squeeze(1)
        
        gen_outputs = self.backbone_model(
            context_input_ids=docs_dict["context_input_ids"],
            context_attention_mask=docs_dict["context_attention_mask"],
            doc_scores=doc_scores,
            decoder_input_ids=decoder_input_ids,
            labels=decoder_input_ids,
            output_retrieved=True,

        )
        
        lm_loss = torch.mean(gen_outputs["loss"])

        outputs["knowledge_loss"] = kn_loss
        outputs["persona_loss"] = pe_loss
        outputs["lm_loss"] = lm_loss

        return outputs
        
    
    def evaluate(self, batch):
        input_ids = batch["input_ids"]
        input_attn_mask = batch["input_attn_mask"]
        decoder_input_ids = batch["decoder_input_ids"]
        persona_input_ids = batch["persona_candidates"]
        persona_attn_mask = batch["persona_candidates_attn_mask"]
        knowledge_input_ids = batch["knowledge_candidates"]
        knowledge_attn_mask = batch["knowledge_candidates_attn_mask"]
        persona_grounding = batch["persona_grounding"]
        knowledge_grounding = batch["knowledge_grounding"]
        outputs = {}
        kn_logits = self.kn_poly_encoder(context_input_ids=input_ids,
                                         context_input_masks=input_attn_mask,
                                         responses_input_ids=knowledge_input_ids,
                                         responses_input_masks=knowledge_attn_mask, )

        pe_logits = self.pe_poly_encoder(context_input_ids=input_ids,
                                         context_input_masks=input_attn_mask,
                                         responses_input_ids=persona_input_ids,
                                         responses_input_masks=persona_attn_mask)

        if self.args.encoder == "bi":
            kn_logits = kn_logits.unsqueeze(0)
            pe_logits = pe_logits.unsqueeze(0)
        pe_loss = self.pe_loss_fct(pe_logits, persona_grounding.float())
        kn_loss = self.kn_loss_fct(kn_logits, knowledge_grounding)
        kn_pred_idx = torch.argmax(kn_logits, 1)

        kn_sel_input_ids = knowledge_input_ids[torch.arange(knowledge_input_ids.size(0)), kn_pred_idx]
        
        persona_num_logits = self.pe_controller(
            input_ids=input_ids,
            attention_mask=input_attn_mask,
            return_dict=True,
        )["pooler_output"]
        persona_num_pred_logit = self.pe_predictor(persona_num_logits)
        persona_num_pred = self.softmax(persona_num_pred_logit)
        
        persona_num = torch.argmax(persona_num_pred, 1).detach().tolist()
        bsz = persona_input_ids.size()[0]
        pe_sel_input_ids_b = []
        pe_sel_index = []
        for bs in range(bsz):
            pe_empty_index = torch.zeros([self.args.persona_num])
            if persona_num[bs] != 0:
                pe_pred_idx = torch.topk(pe_logits[bs, :], persona_num[bs])[1]
                pe_empty_index[pe_pred_idx] = 1
                pe_sel_input_ids_b_f = persona_input_ids[bs, :, :][pe_pred_idx]
                pe_sel_input_ids_b_f = pe_sel_input_ids_b_f[
                    pe_sel_input_ids_b_f != self.tokenizer.question_encoder.pad_token_id].detach().tolist()
            else:
                pe_sel_input_ids_b_f = []
                
            pe_sel_index.append(pe_empty_index)
            candidate_padding_length = self.args.max_paragraph_len - len(pe_sel_input_ids_b_f)
            if candidate_padding_length > 0:
                # Must Check
                pe_sel_input_ids_b_f = pe_sel_input_ids_b_f + [self.tokenizer.question_encoder.pad_token_id] * (
                        self.args.max_paragraph_len - len(pe_sel_input_ids_b_f))
            else:
                pe_sel_input_ids_b_f = pe_sel_input_ids_b_f[:self.args.max_paragraph_len]
            
            assert len(pe_sel_input_ids_b_f) == self.args.max_paragraph_len
            pe_sel_input_ids_b.append(torch.tensor(pe_sel_input_ids_b_f).long())
        
        pe_sel_input_ids = torch.stack(pe_sel_input_ids_b).to(self.device)
        pe_sel_indices = torch.stack(pe_sel_index).to(self.device)
        persona_pred = pe_sel_indices

        new_input = torch.cat((input_ids[:, -312:], pe_sel_input_ids, kn_sel_input_ids), 1)
        question_hidden_states = self.backbone_model.question_encoder(new_input)[0]
        docs_dict = self.retriever(new_input.detach().cpu().numpy(), question_hidden_states.detach().cpu().numpy(),
                                   return_tensors="pt")
        docs_dict = {dd: docs_dict[dd].to(self.device) for dd in docs_dict}
        
        doc_scores = torch.bmm(
            question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)).squeeze(1)

        gen_outputs = self.backbone_model(
            context_input_ids=docs_dict["context_input_ids"],
            context_attention_mask=docs_dict["context_attention_mask"],
            doc_scores=doc_scores,
            decoder_input_ids=decoder_input_ids,
            labels=decoder_input_ids,
            output_retrieved=True,
        )
        generated = self.backbone_model.generate(
            context_input_ids=docs_dict["context_input_ids"],
            context_attention_mask=docs_dict["context_attention_mask"],
            doc_scores=doc_scores )

        try:
            print("input")
            print(self.tokenizer.question_encoder.batch_decode(new_input, skip_special_tokens=True,
                                                               clean_up_tokenization_spaces=True))
            print("generated answers")
            print(self.tokenizer.question_encoder.batch_decode(generated, skip_special_tokens=True,
                                                               clean_up_tokenization_spaces=True))
        except UnicodeEncodeError:
            pass
        pred_gen = self.tokenizer.generator.batch_decode(generated, skip_special_tokens=True)
        lm_loss = torch.mean(gen_outputs["loss"])
        
        
        outputs["knowledge_loss"] = kn_loss
        outputs["persona_loss"] = pe_loss
        outputs["lm_loss"] = lm_loss
        
        r1_indices = torch.topk(kn_logits, 1)[1]
        r2_indices = torch.topk(kn_logits, 2)[1]  # R 2 @ 100
        r5_indices = torch.topk(kn_logits, 5)[1]  # R 5 @ 100
        
        return persona_pred, r1_indices, r2_indices, r5_indices, pred_gen
    
    def inference(self, batch):
        input_ids = batch["input_ids"]
        input_attn_mask = batch["input_attn_mask"]
        decoder_input_ids = batch["decoder_input_ids"]
        persona_input_ids = batch["persona_candidates"]
        persona_attn_mask = batch["persona_candidates_attn_mask"]
        knowledge_input_ids = batch["knowledge_candidates"]
        knowledge_attn_mask = batch["knowledge_candidates_attn_mask"]
        
        outputs = {}
        kn_logits = self.kn_poly_encoder(context_input_ids=input_ids,
                                         context_input_masks=input_attn_mask,
                                         responses_input_ids=knowledge_input_ids,
                                         responses_input_masks=knowledge_attn_mask, )
        kn_pred_idx = torch.argmax(kn_logits, 1)
        pe_logits = self.pe_poly_encoder(context_input_ids=input_ids,
                                         context_input_masks=input_attn_mask,
                                         responses_input_ids=persona_input_ids,
                                         responses_input_masks=persona_attn_mask)

        kn_sel_input_ids = knowledge_input_ids[torch.arange(knowledge_input_ids.size(0)), kn_pred_idx]

        persona_num_logits = self.pe_controller(
            input_ids=input_ids,
            attention_mask=input_attn_mask,
            return_dict=True,
        )["pooler_output"]
        persona_num_pred_logit = self.pe_predictor(persona_num_logits)
        persona_num_pred = self.softmax(persona_num_pred_logit)

        persona_num = torch.argmax(persona_num_pred, 1).detach().tolist()
        bsz = persona_input_ids.size()[0]
        pe_sel_input_ids_b = []
        pe_sel_index = []
        for bs in range(bsz):
            pe_empty_index = torch.zeros([self.args.persona_num])
            if persona_num[bs] != 0:
                pe_pred_idx = torch.topk(pe_logits[bs, :], persona_num[bs])[1]
                pe_empty_index[pe_pred_idx] = 1
                pe_sel_input_ids_b_f = persona_input_ids[bs, :, :][pe_pred_idx]
                pe_sel_input_ids_b_f = pe_sel_input_ids_b_f[
                    pe_sel_input_ids_b_f != self.tokenizer.question_encoder.pad_token_id].detach().tolist()
            else:
                pe_sel_input_ids_b_f = []
    
            pe_sel_index.append(pe_empty_index)
            # padding again
            candidate_padding_length = self.args.max_paragraph_len - len(pe_sel_input_ids_b_f)
            if candidate_padding_length > 0:
                # Must Check
                pe_sel_input_ids_b_f = pe_sel_input_ids_b_f + [self.tokenizer.question_encoder.pad_token_id] * (
                        self.args.max_paragraph_len - len(pe_sel_input_ids_b_f))
            else:
                pe_sel_input_ids_b_f = pe_sel_input_ids_b_f[:self.args.max_paragraph_len]
    
            assert len(pe_sel_input_ids_b_f) == self.args.max_paragraph_len
            pe_sel_input_ids_b.append(torch.tensor(pe_sel_input_ids_b_f).long())

        pe_sel_input_ids = torch.stack(pe_sel_input_ids_b).to(self.device)
        pe_sel_indices = torch.stack(pe_sel_index).to(self.device)
        persona_pred = pe_sel_indices

        new_input = torch.cat((input_ids, pe_sel_input_ids, kn_sel_input_ids), 1)

        question_hidden_states = self.backbone_model.question_encoder(new_input)[0]
        docs_dict = self.retriever(new_input.detach().cpu().numpy(), question_hidden_states.detach().cpu().numpy(),
                                   return_tensors="pt")
        docs_dict = {dd: docs_dict[dd].to(self.device) for dd in docs_dict}
        doc_scores = torch.bmm(
            question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)).squeeze(1)
        gen_outputs = self.backbone_model(
            context_input_ids=docs_dict["context_input_ids"],
            context_attention_mask=docs_dict["context_attention_mask"],
            doc_scores=doc_scores,
            decoder_input_ids=decoder_input_ids,
            labels=decoder_input_ids

        )
        generated = self.backbone_model.generate(
            context_input_ids=docs_dict["context_input_ids"],
            context_attention_mask=docs_dict["context_attention_mask"],
            doc_scores=doc_scores, )

        print(pe_sel_indices, 'sel indices')

        try:
            print("input")
            print(self.tokenizer.question_encoder.batch_decode(new_input, skip_special_tokens=True,
                                                               clean_up_tokenization_spaces=True))
            print("generated answers")
            print(self.tokenizer.question_encoder.batch_decode(generated, skip_special_tokens=True,
                                                               clean_up_tokenization_spaces=True))
        except UnicodeEncodeError:
            pass

        pred_gen = self.tokenizer.generator.batch_decode(generated, skip_special_tokens=True)
        lm_loss = torch.mean(gen_outputs["loss"])

        outputs["lm_loss"] = lm_loss
        
        r1_indices = torch.topk(kn_logits, 1)[1]
        r2_indices = torch.topk(kn_logits, 2)[1]  # R 2 @ 100
        r5_indices = torch.topk(kn_logits, 5)[1]  # R 5 @ 100

        return persona_pred, r1_indices, r2_indices, r5_indices, pred_gen
