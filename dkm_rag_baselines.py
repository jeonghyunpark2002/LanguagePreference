import json
import os
import torch
from tqdm.auto import tqdm  
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score as sk_f1_score
import string
import regex
import numpy as np
from rouge import Rouge
from collections import Counter

def simple_accuracy(preds, labels):
    return float((preds == labels).mean())

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = float(sk_f1_score(y_true=labels, y_pred=preds))
    return {
        "accuracy": acc,
        "f1": f1,
    }

def pearson_and_spearman(preds, labels):
    pearson_corr = float(pearsonr(preds, labels)[0])
    spearman_corr = float(spearmanr(preds, labels)[0])
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
    }

def normalize(s: str) -> str:
    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_single(prediction, ground_truth, tokenfun=lambda x: x.split()):
    prediction_tokens = tokenfun(normalize(prediction))
    ground_truth_tokens = tokenfun(normalize(ground_truth))
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def ngrams(s, n=3):
    exclude = set(string.punctuation)
    s = ''.join(ch if ch not in exclude else " " for ch in s.lower())
    tokens = []
    for w in s.split():
        length = len(w)
        if length < n:
            tokens.append(w)
        else:
            for i in range(length - n + 1):
                tokens.append(w[i:i+n])
    return tokens

def rouge_wrapper(prediction, ground_truth):
    try:
        result = rouge.get_scores(prediction, ground_truth, avg=True)
        return result["rouge-1"]["f"], result["rouge-2"]["f"], result["rouge-l"]["f"]
    except:
        return 0.0, 0.0, 0.0

def rouge_score_single(prediction, ground_truths):
    ground_truths = [x for x in ground_truths if len(x) > 0]
    if len(prediction) == 0 or len(ground_truths) == 0:
        return 0.0, 0.0, 0.0
    scores = [rouge_wrapper(prediction, gt) for gt in ground_truths]
    rouge1 = max(s[0] for s in scores)
    rouge2 = max(s[1] for s in scores)
    rougel = max(s[2] for s in scores)
    return rouge1, rouge2, rougel

def rouge_score(predictions, references):
    rouge1, rouge2, rougel = [], [], []
    for ground_truths, prediction in zip(references, predictions):
        r1, r2, rl = rouge_score_single(prediction, ground_truths)
        rouge1.append(r1)
        rouge2.append(r2)
        rougel.append(rl)
    return np.mean(rouge1), np.mean(rouge2), np.mean(rougel)

def f1_score_custom(predictions, references, tokenfun=lambda x: x.split()):
    f1_vals, precision_vals, recall_vals = [], [], []
    for ground_truths, prediction in zip(references, predictions):      
        f1_, precision_, recall_ = [
            max(values) for values in zip(*[f1_single(prediction, gt, tokenfun)
                                            for gt in ground_truths])
        ]
        f1_vals.append(f1_)
        precision_vals.append(precision_)
        recall_vals.append(recall_)
    return np.mean(f1_vals), np.mean(precision_vals), np.mean(recall_vals)

def em_single(prediction, ground_truth):
    return float(normalize(prediction) == normalize(ground_truth))

def exact_match_score(predictions, references):
    results = []
    for ground_truths, prediction in zip(references, predictions):
        em_candidates = [em_single(prediction, gt) for gt in ground_truths]
        results.append(max(em_candidates))
    return np.mean(results)

def match_single(prediction, ground_truth):
    return float(normalize(ground_truth) in normalize(prediction))

def match_score(predictions, references):
    results = []
    for ground_truths, prediction in zip(references, predictions):
        match_candidates = [match_single(prediction, gt) for gt in ground_truths]
        results.append(max(match_candidates))
    return np.mean(results)

class RAGMetrics:
    @staticmethod
    def compute(predictions, references, questions=None):
        rouge1, rouge2, rougel = rouge_score(predictions, references)
        f1_val, precision_val, recall_val = f1_score_custom(predictions, references)
        f1_char3gram, precision_char3gram, recall_char3gram = f1_score_custom(predictions, references, ngrams)
        M = match_score(predictions, references)
        EM = exact_match_score(predictions, references)

        return {
            "M": M,
            "EM": EM,
            "F1": f1_val,
            "Precision": precision_val,
            "Recall": recall_val,
            "Recall_char3gram": recall_char3gram,
            "Rouge-1": rouge1,
            "Rouge-2": rouge2,
            "Rouge-L": rougel,
        }
        
def get_top5_passages1(qid, doc_data):
    if qid not in doc_data:
        return []
    all_passages = doc_data[qid]
    top_passages = []
    for p in all_passages:
        if p["rank"] <= 5:
            top_passages.append(p["content_translated"])
    return top_passages

def get_top5_passages(qid, doc_data):
    if qid not in doc_data:
        return []
    
    all_passages = doc_data[qid]
    
    top_passages = [p["rewritten_passage"] for p in all_passages]
    
    return top_passages


'''
def build_system_message_with_docs(passages):
    docs_str = "\n".join([f"- {p}" for p in passages])
    system_msg = (
        "이제부터 너는 내 유능한 비서야. "
        "네 역할은 내가 제공한 문서에서 관련 정보를 찾아내고 "
        "내 질문에 최대한 짧게 답하는 거야. 한국어로 답해 줘.\n\n"
        f"Background:\n{docs_str}"
    )
    return system_msg


def build_system_message_with_docs(passages):
    docs_str = "\n".join([f"- {p}" for p in passages])
    system_msg = (
        "You are a helpful assistant. "
        "Your task is to extract relevant information from provided documents and "
        "to answer to questions as short as possible. "
        "Please reply in English.\n\n"
        f"Background:\n{docs_str}"
    )
    return system_msg
'''

def build_system_message_with_docs(passages):
    docs_str = "\n".join([f"- {p}" for p in passages])
    system_msg = (
        "你是一名乐于助人的助手。"
        "你的任务是从提供的文档中提取相关信息，并尽可能简短地回答问题。"
        "请用简体中文回复。\n\n"
        f"Background:\n{docs_str}"
    )
    return system_msg



def build_user_message(question):
    return f"Question：{question}"



device = 0 if torch.cuda.is_available() else -1


with open("zh_zh_top5_with_content.json", "r", encoding="utf-8") as f:
    zh_top5_content_data = json.load(f)

with open("optimized_unify_zh.json", "r", encoding="utf-8") as f:
    optimized_unify_data = json.load(f)


base_path = "datasets"
mkqa_path = os.path.join(base_path, "mkqa_zh_cn_train")
mkqa_ds = load_from_disk(mkqa_path)

mkqa_id2question = {}
mkqa_id2label = {}
for i in range(len(mkqa_ds)):
    example = mkqa_ds[i]
    qid = str(example["id"])  
    mkqa_id2question[qid] = example["content"]
    mkqa_id2label[qid] = example["label"]

references = []
for i in range(len(mkqa_ds)):
    example = mkqa_ds[i]
    qid = str(example["id"])
    labels = mkqa_id2label[qid]
    references.append(labels)

rouge = Rouge()

def combine_top5_passages(qid, doc1_data, doc2_data):
    passages1 = get_top5_passages1(qid, doc1_data)  
    passages2 = get_top5_passages(qid, doc2_data)  

    combined_passages = [
        f"{p1} + {p2}" for p1, p2 in zip(passages1, passages2)
    ]

    return combined_passages[:5]  

    

print("\n[1/4] Running Qwen model...")
qwen_tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    trust_remote_code=True
)
qwen_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16
)
qwen_pipeline = pipeline(
    "text-generation",
    model=qwen_model,
    tokenizer=qwen_tokenizer,
    device_map="auto"
)

predictions_qwen = []

for i in tqdm(range(len(mkqa_ds)), desc="Processing Qwen"):
    qid = str(mkqa_ds[i]["id"])
    question = mkqa_id2question[qid]

    top5_passages_combined = combine_top5_passages(qid, zh_top5_content_data, optimized_unify_data)

    system_msg = build_system_message_with_docs(top5_passages_combined)
    user_msg = build_user_message(question)

    ans_qwen = qwen_pipeline(system_msg + "\n\n" + user_msg)
    predictions_qwen.append(ans_qwen[0]["generated_text"])

metrics_qwen = RAGMetrics.compute(predictions_qwen, references)
print("Qwen - Character 3-gram Recall:", metrics_qwen["Recall_char3gram"])

del qwen_model, qwen_pipeline
torch.cuda.empty_cache()



print("\n[2/4] Running Aya model...")
aya_tokenizer = AutoTokenizer.from_pretrained("CohereForAI/aya-expanse-8b")
aya_model = AutoModelForCausalLM.from_pretrained("CohereForAI/aya-expanse-8b", torch_dtype=torch.float16)
aya_pipeline = pipeline(
    "text-generation",
    model=aya_model,
    tokenizer=aya_tokenizer,
    device_map="auto",
)
predictions_aya = []

for i in tqdm(range(len(mkqa_ds)), desc="Processing Aya"):
    qid = str(mkqa_ds[i]["id"])
    question = mkqa_id2question[qid]

    top5_passages_combined = combine_top5_passages(qid, zh_top5_content_data, optimized_unify_data)
    system_msg = build_system_message_with_docs(top5_passages_combined)
    user_msg = build_user_message(question)
    
    ans_aya = aya_pipeline(system_msg + "\n\n" + user_msg)
    predictions_aya.append(ans_aya[0]["generated_text"])

metrics_aya = RAGMetrics.compute(predictions_aya, references)
print("Aya - Character 3-gram Recall:", metrics_aya["Recall_char3gram"])

del aya_model, aya_pipeline
torch.cuda.empty_cache()



print("\n[3/4] Running PHI-4 model...")
phi4_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-4")
phi4_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-4", torch_dtype=torch.float16)
phi4_pipeline = pipeline(
    "text-generation",
    model=phi4_model,
    tokenizer=phi4_tokenizer,
    device_map="auto",
)
predictions_phi4 = []

for i in tqdm(range(len(mkqa_ds)), desc="Processing PHI-4"):
    qid = str(mkqa_ds[i]["id"])
    question = mkqa_id2question[qid]

    top5_passages_combined = combine_top5_passages(qid, zh_top5_content_data, optimized_unify_data)
    system_msg = build_system_message_with_docs(top5_passages_combined)
    user_msg = build_user_message(question)

    ans_phi4 = phi4_pipeline(system_msg + "\n\n" + user_msg)
    predictions_phi4.append(ans_phi4[0]["generated_text"])

metrics_phi4 = RAGMetrics.compute(predictions_phi4, references)
print("PHI-4 - Character 3-gram Recall:", metrics_phi4["Recall_char3gram"])

del phi4_model, phi4_pipeline
torch.cuda.empty_cache()


print("\n[4/4] Running Llama 3.1-8B-Instruct model...")
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
llama_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16
)
llama_pipeline = pipeline(
    "text-generation",
    model=llama_model,
    tokenizer=llama_tokenizer,
    device_map="auto"
)
predictions_llama = []

for i in tqdm(range(len(mkqa_ds)), desc="Processing Llama"):
    qid = str(mkqa_ds[i]["id"])
    question = mkqa_id2question[qid]

    top5_passages_combined = combine_top5_passages(qid, zh_top5_content_data, optimized_unify_data)
    system_msg = build_system_message_with_docs(top5_passages_combined)
    user_msg = build_user_message(question)

    ans_llama = llama_pipeline(system_msg + "\n\n" + user_msg)
    predictions_llama.append(ans_llama[0]["generated_text"])

metrics_llama = RAGMetrics.compute(predictions_llama, references)
print("Llama-3.1-8B-Instruct - Character 3-gram Recall:", metrics_llama["Recall_char3gram"])

del llama_model, llama_pipeline
torch.cuda.empty_cache()

print("\nDone! All models have been processed sequentially.")

