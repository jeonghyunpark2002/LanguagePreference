import json
import os
import glob
from tqdm import tqdm
from datasets import load_from_disk
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
import pickle
import random
from langdetect import detect
from collections import defaultdict

#NLLB error -> transformers 4.37.1 downgrade 

from models.rerankers.reranker import Reranker

class CrossEncoder(Reranker):
    def __init__(self, model_name=None, max_len=512):
        self.model_name = model_name
        self.max_len = max_len
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16
        ).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, max_length=self.max_len)
        self.model.eval()

    def collate_fn(self, examples):
        question = [e['query'] for e in examples]
        doc = [e['doc'] for e in examples]
        q_id = [e['q_id'] for e in examples]
        d_id = [e['d_id'] for e in examples]
        inp_dict = self.tokenizer(
            question, 
            doc, 
            padding="max_length", 
            truncation='only_second', 
            max_length=self.max_len, 
            return_tensors='pt'
        )
        inp_dict['q_id'] = q_id
        inp_dict['d_id'] = d_id
        return inp_dict

    def __call__(self, kwargs):
        for k,v in kwargs.items():
            if isinstance(v, torch.Tensor):
                kwargs[k] = v.to('cuda')
        score = self.model(**{
            k:v for k,v in kwargs.items() 
            if k not in ['q_id', 'd_id']
        }).logits
        return {
            "score": score,
            "q_id": kwargs['q_id'],
            "d_id": kwargs['d_id']
        }

        
device = "cuda" if torch.cuda.is_available() else "cpu"

init_args1 = {
  "model_name": "BAAI/bge-reranker-v2-m3",
}
init_args2 = {
  "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}
init_args3 = {
  "model_name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
}

TARGET_LANG = "kor_Hang" 
initial_rank_file = "runs/ar_ko_initial_rank.json"
translated_passages_file = f"ar_{TARGET_LANG}_translated_passages.json"
reranked_results_file = f"ar_{TARGET_LANG}_reranked_results.json"


reranker = CrossEncoder(
    model_name=init_args3['model_name'], 
    max_len=256
)

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16, 
    device_map="auto"
).to(device)

LANGDETECT_TO_NLLB = {
    "en": "eng_Latn",   
    "ko": "kor_Hang",   
    "ar": "arb_Arab",   
    "zh": "zho_Hans",   
    "fi": "fin_Latn",   
    "fr": "fra_Latn",   
    "de": "deu_Latn",   
    "ja": "jpn_Jpan",   
    "it": "ita_Latn",   
    "pt": "por_Latn",   
    "ru": "rus_Cyrl",   
    "es": "spa_Latn",   
    "th": "tha_Thai",   
}
translation_cache = {}

def batch_translate_with_model(
    texts, 
    src_lang="eng_Latn", 
    tgt_lang="eng_Latn", 
    batch_size=16
):
    tokenizer.src_lang = src_lang
    translated_texts = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
                max_length=256,
                num_beams=5
            )

        for output_ids in outputs:
            translated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            translated_texts.append(translated_text)

    return translated_texts

def auto_translate_passages(
    doc_ids,
    contents, 
    tgt_lang, 
    batch_size=16, 
    pretranslated_dict=None
):
    results = [None] * len(contents)
    grouped_texts = defaultdict(list)

    for i, (doc_id, text) in enumerate(zip(doc_ids, contents)):
        if pretranslated_dict and doc_id in pretranslated_dict:
            results[i] = pretranslated_dict[doc_id]
            translation_cache[text] = pretranslated_dict[doc_id]
            continue

        if text in translation_cache:
            results[i] = translation_cache[text]
            continue

        try:
            lang_code = detect(text)  
        except:
            lang_code = "en"
        src_lang = LANGDETECT_TO_NLLB.get(lang_code, "eng_Latn")
        
        grouped_texts[src_lang].append((i, text))

    for src_lang, idx_text_pairs in grouped_texts.items():
        original_texts = [t[1] for t in idx_text_pairs]
        translated_batch = batch_translate_with_model(
            original_texts,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            batch_size=batch_size
        )
        for (idx, orig), trans in zip(idx_text_pairs, translated_batch):
            results[idx] = trans
            translation_cache[orig] = trans

    return results


with open(initial_rank_file, "r", encoding="utf-8") as f:
    ko_initial_data = json.load(f)

base_path = "datasets"
cache_dir = "cache"
os.makedirs(cache_dir, exist_ok=True)

content_cache = {}

def get_cache_file_for_prefix(prefix):
    return os.path.join(cache_dir, f"{prefix}_content_cache.pickle")

def load_or_create_cache(prefix):
    cache_file = get_cache_file_for_prefix(prefix)
    if os.path.exists(cache_file):
        print(f"Loading content cache for '{prefix}' from disk...")
        with open(cache_file, "rb") as f:
            content_cache[prefix] = pickle.load(f)
    else:
        print(f"No cache found for '{prefix}'. It will be created during runtime.")
        content_cache[prefix] = {}

def save_cache(prefix):
    cache_file = get_cache_file_for_prefix(prefix)
    with open(cache_file, "wb") as f:
        pickle.dump(content_cache[prefix], f)
    print(f"Content cache for '{prefix}' saved to {cache_file}")

def get_dataset_folder(prefix):
    if prefix.startswith("kilt-100w"):
        return "kilt-100w_full"
    else:
        return prefix + "_train"

def load_dataset_for_prefix(prefix):
    folder_name = get_dataset_folder(prefix)
    dataset_path = os.path.join(base_path, folder_name)
    ds = load_from_disk(dataset_path)
    index_map = {}
    for i in range(len(ds)):
        wid = str(ds[i]["id"])
        if wid not in index_map:
            index_map[wid] = ds[i]["content"]
    return index_map

def get_content_from_doc_id(doc_id):
    prefix, wid = doc_id.rsplit("_", 1)
    wid = wid.strip()

    if prefix not in content_cache:
        load_or_create_cache(prefix)
        if not content_cache[prefix]:
            print(f"Loading and indexing dataset for prefix: {prefix}")
            content_cache[prefix] = load_dataset_for_prefix(prefix)
            save_cache(prefix)

    index_map = content_cache[prefix]
    return index_map.get(wid, "")


pretranslated_dict = {}

pattern = f"*{TARGET_LANG}*translated_passages.json"
found_files = glob.glob(pattern)
if found_files:
    print(f"Found existing translation files: {found_files}")
    for file_path in found_files:
        print(f"Loading {file_path} ...")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for qid, doc_list in data.items():
                for doc_item in doc_list:
                    doc_id = doc_item["doc_id"]
                    translated_text = doc_item["content_translated"]
                    pretranslated_dict[doc_id] = translated_text
else:
    print(f"No existing files found for pattern: {pattern}")



all_passages = {}
for query_id, docs in tqdm(ko_initial_data.items(), desc="Loading and translating"):
    doc_ids = [d["doc_id"] for d in docs]
    contents = [get_content_from_doc_id(d["doc_id"]) for d in docs]

    all_translated = all(doc_id in pretranslated_dict for doc_id in doc_ids)
    if all_translated:
        query_passages = []
        for d in docs:
            doc_id = d["doc_id"]
            translated_text = pretranslated_dict[doc_id]
            item = {
                "doc_id": doc_id,
                "initial_rank": d["rank"],
                "content_translated": translated_text
            }
            query_passages.append(item)
    else:
        translated_contents = auto_translate_passages(
            doc_ids=doc_ids,
            contents=contents,
            tgt_lang=TARGET_LANG,
            batch_size=16,
            pretranslated_dict=pretranslated_dict
        )
        query_passages = []
        for d, orig, ko_t in zip(docs, contents, translated_contents):
            item = {
                "doc_id": d["doc_id"],
                "initial_rank": d["rank"],
                "content_translated": ko_t
            }
            query_passages.append(item)
            pretranslated_dict[d["doc_id"]] = ko_t

    all_passages[query_id] = query_passages

with open(translated_passages_file, "w", encoding="utf-8") as f:
    json.dump(all_passages, f, ensure_ascii=False, indent=2)
print(f"Translated passages saved to {translated_passages_file}")



mkqa_path = os.path.join(base_path, "mkqa_en_train")
mkqa_ds = load_from_disk(mkqa_path)
mkqa_id2query = {}
for i in range(len(mkqa_ds)):
    qid = str(mkqa_ds[i]["id"])
    mkqa_id2query[qid] = qid  

final_results = {}
for query_id, passages in tqdm(all_passages.items(), desc="Re-ranking"):
    if query_id not in mkqa_id2query:
        continue
    query_text = mkqa_id2query[query_id]
    rerank_inputs = []
    for p in passages:
        rerank_inputs.append({
            "query": query_text,
            "doc": p["content_translated"],
            "q_id": query_id,
            "d_id": p["doc_id"]
        })
    
    scores = []
    batch_size = 256
    for i in range(0, len(rerank_inputs), batch_size):
        batch = rerank_inputs[i:i+batch_size]
        inp = reranker.collate_fn(batch)
        with torch.no_grad():
            out = reranker(inp)
        batch_scores = out["score"].cpu().float().tolist()
        q_ids = out["q_id"]
        d_ids = out["d_id"]
        for q, d, s in zip(q_ids, d_ids, batch_scores):
            scores.append((d, s))
    
    scores_dict = {d: s for (d, s) in scores}
    ranked_passages = sorted(passages, key=lambda x: scores_dict[x["doc_id"]], reverse=True)

    for rank_idx, p in enumerate(ranked_passages, start=1):
        p["new_rank"] = rank_idx
        p["score"] = scores_dict[p["doc_id"]]

    final_results[query_id] = ranked_passages

with open(reranked_results_file, "w", encoding="utf-8") as f:
    json.dump(final_results, f, ensure_ascii=False, indent=2)
print(f"Reranked results saved to {reranked_results_file}")


query_scores = {}
for query_id, res in final_results.items():
    improvement_sum = 0
    max_possible_sum = 0

    for p_info in res:
        init_rank = p_info["initial_rank"]
        new_rank = p_info["new_rank"]

        improved_i = max(init_rank - new_rank, 0)
        max_improve_i = init_rank - 1

        improvement_sum += improved_i
        max_possible_sum += max_improve_i

    if max_possible_sum > 0:
        score = (improvement_sum / max_possible_sum) * 100
    else:
        score = 0.0
    
    query_scores[query_id] = score

scores = list(query_scores.values())
overall_preference_score = sum(scores) / len(scores) if scores else 0.0

print(f"Query-wise Preference Scores (% of max possible improvement) :")
for qid, score in query_scores.items():
    print(f"  Query {qid}: {score:.2f}/100")

print(f"\nOverall Preference Score: {overall_preference_score:.2f}/100 (for {TARGET_LANG})")
