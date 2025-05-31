import json
import os
import torch
from tqdm.auto import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM


'''
    prompts = [
        f"""
        Original Passage:
        {passage}

        Question:
        {question}

        Please create an independent document according to the following requirements:
        1) Utilize known facts (parametric knowledge) related to the question,
        2) Seamlessly combine with the original passage by removing redundant or unnecessary sentences. No additional explanations are allowed,
        3) All content must be written smoothly and concisely in **English**.
        """
        for passage in passages
    ]
'''
def load_model_and_tokenizer(model_name="meta-llama/Llama-3.1-8B-Instruct"):
    print("[Info] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("[Info] pad_token was not set. Setting pad_token to eos_token.")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True  
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    print("[Info] Model and tokenizer loaded successfully.")
    return tokenizer, model

def load_doc_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_top5_passages(doc_data, qid):
    if qid not in doc_data:
        return []
    sorted_list = sorted(doc_data[qid], key=lambda x: x["rank"])
    return [entry["content_translated"] for entry in sorted_list[:5]]

def load_mkqa_ko(base_path="datasets"):
    ds_path = os.path.join(base_path, "mkqa_zh_cn_train")
    ds = load_from_disk(ds_path)
    return {str(ex["id"]): ex["content"] for ex in ds}

def rewrite_passages_in_batch(passages, question, tokenizer, model, device):
    prompts = [
        f"""
        原始段落：
        {passage}

        问题：
        {question}

        请根据以下要求创建一个独立的文档：
        1) 利用与问题相关的已知事实（参数知识），
        2) 自然地与原始段落结合，删除重复或不必要的句子。绝对不允许添加额外的解释，
        3) 所有内容必须用**简洁流畅的中文**编写。
        """
        for passage in passages
    ]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    input_length = inputs.input_ids.shape[1] 

    output_ids = model.generate(
        **inputs,
        temperature=0.3,
        max_new_tokens=200,
        repetition_penalty=1.2,
        top_p=0.95,
        do_sample=True
    )

    generated_ids = output_ids[:, input_length:]
    
    rewritten_passages = [
        tokenizer.decode(generated, skip_special_tokens=True).strip()
        for generated in generated_ids
    ]
    
    return rewritten_passages

if __name__ == "__main__":
    tokenizer, model = load_model_and_tokenizer()
    json_path = "zh_zh_top5_with_content.json"
    doc_data = load_doc_data(json_path)
    id2question = load_mkqa_ko(base_path="datasets")

    results = {}
    qids = list(doc_data.keys())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for qid in tqdm(qids, desc="Processing QIDs"):
        question_text = id2question.get(qid, "")
        if not question_text:
            print(f"[Warning] QID {qid} has no associated question. Skipping.")
            continue

        top5 = get_top5_passages(doc_data, qid)
        if not top5:
            print(f"[Warning] QID {qid} has no passages. Skipping.")
            continue

        rewritten_passages = rewrite_passages_in_batch(top5, question_text, tokenizer, model, device)
        
        
        results[qid] = [
            {"rewritten_passage": rewritten}
            for rewritten in rewritten_passages
        ]

    output_file = "optimized_unify_zh.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[Done] Saved {len(results)} QIDs with re-written passages to {output_file}")
