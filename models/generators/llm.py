'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from models.generators.generator import Generator
import warnings
from utils import prepare_labels
import random
import os
import json
import gc
random.seed(42)


class LLM(Generator):
    def __init__(
        self,
        model_name=None,
        batch_size=1,
        max_new_tokens=1,
        max_doc_len=100,
        max_length=None,
        prompt=None,
        quantization=None,  # 더 이상 사용되지 않지만, 혹시 모를 호환성을 위해 남겨둠
        gguf_file=None,     # gguf도 사용하지 않을 경우 제거 가능
        attn_implementation="flash_attention_2",
        local_path=False
    ):
        # 기본 Generator 초기화
        Generator.__init__(self, model_name=model_name, batch_size=batch_size)
        
        # A100이 아니면 flash_attention_2 대신 sdpa 사용 (원본 로직 유지)
        if not "A100" in torch.cuda.get_device_name(torch.cuda.current_device):
            attn_implementation = "sdpa"

        self.max_length = max_length
        self.max_doc_len = max_doc_len
        self.quantization = None  # quantization은 사용 안 함
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt

        # ★ PEFT 관련 로직 제거, regular model만 로드
        warnings.warn(f"Loading regular (non-quantized) model: {model_name}.")
        tokenizer_name = self.model_name
        
        # 토크나이저 로드
        print(tokenizer_name)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, gguf_file=gguf_file)
        except:
            config_dict = os.path.join(tokenizer_name, 'config.json')
            with open(config_dict, 'r') as f:
                config = json.load(f)
            tokenizer_name = config['_name_or_path']
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, gguf_file=gguf_file)

        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.bos_token

        # 모델 로드 (원본 모델, bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map='auto',
            gguf_file=gguf_file,
            trust_remote_code=True,
        )

        # bfloat16으로 변환
        self.model = self.model.bfloat16()
        self.model.eval()

        # 필요 시 설정
        self.model.config.pretraining_tp = 1

    def get_response(self):
        return '\nResponse:\n'

    def get_response_template_ids(self):
        response_template = self.get_response()
        return self.tokenizer.encode(response_template, add_special_tokens=False)

    def prediction_step(self, model, model_input, label_ids=None):
        output = model(**model_input, labels=label_ids)
        return output.logits, output.loss

    def generate(self, instr_tokenized):
        input_ids = instr_tokenized['input_ids'].to(self.model.device)
        attention_mask = instr_tokenized['attention_mask'].to(self.model.device)
        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
        )

        prompt_len = instr_tokenized['input_ids'].size(1)
        generated_ids = output_ids[:, prompt_len:]
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return decoded

    def __del__(self):
        gc.collect()
        torch.cuda.empty_cache()

    def collate_fn(self, examples, eval=False, **kwargs):
        ignore_index = -100
        q_ids = [e['q_id'] for e in examples]

        input_ids_list = [e["tokenized_input"]["input_ids"][0] for e in examples]
        attention_mask_list = [e["tokenized_input"]["attention_mask"][0] for e in examples]

        label = [e['label'] if isinstance(e['label'], str) else e['label'] for e in examples]
        query = [e['query'] for e in examples]
        ranking_label = [e['ranking_label'] for e in examples] if 'ranking_label' in examples[0] else [None]*len(examples)
        instr = [e["formatted_instruction"] for e in examples]

        max_length = max(len(ids) for ids in input_ids_list)

        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        input_ids_tensor = torch.stack([
            torch.cat([torch.full((max_length - len(ids),), pad_token_id, dtype=torch.long),
                       torch.tensor(ids, dtype=torch.long)])
            for ids in input_ids_list
        ])

        attention_mask_tensor = torch.stack([
            torch.cat([
                torch.full((max_length - len(mask),), 0, dtype=torch.long),
                torch.tensor(mask, dtype=torch.long)
            ])
            for mask in attention_mask_list
        ])

        model_input = {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
        }
        data_dict = {}

        if not eval:
            response_token_ids = self.get_response_template_ids()
            label_ids = prepare_labels(model_input['input_ids'], response_token_ids[1:], ignore_index=ignore_index)
            data_dict['label_ids'] = label_ids

        data_dict.update({
            'model_input': model_input,
            'q_id': q_ids,
            'query': query,
            'instruction': instr,
            'label': label,
            'ranking_label': ranking_label,
        })

        return data_dict

    def format_instruction(self, sample):
        question = sample['query']
        if 'doc' in sample:
            docs = ''
            for i, doc in enumerate(sample['doc']):
                doc = ' '.join(doc.split()[:self.max_doc_len])
                docs += f"Document {i+1}: {doc}\n"
            compiled_prompt = self.compile_prompt(self.prompt.system, self.prompt.user, question, docs)
        else:
            compiled_prompt = self.compile_prompt(self.prompt.system_without_docs, self.prompt.user_without_docs, question)
        return compiled_prompt + self.get_response()
