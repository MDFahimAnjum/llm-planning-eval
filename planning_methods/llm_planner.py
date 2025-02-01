from utils.constants import TEMPLATE, INST_CODELLAMA_GEN,TEMPLATE_CORR, INST_CODELLAMA_ITER_CORR, INST_CUSTOM_GEN
from utils.normalize_sql import normalize_sql
from func_timeout import func_timeout
from copy import deepcopy
from transformers import GenerationConfig

import numpy as np
import json
import torch
from utils.functions import swap_memory

def prepare_prompt(example, retriever_gen,prompt_method=0,headstr="SELECT"):
    # prepare prompt
    if retriever_gen is None:
        model_inp = TEMPLATE.format(example["db_id"], example["schema"], example["question"])
        prompt = model_inp
    else:
        demos = retriever_gen.retrieve(example["question"])
        model_inp = "\n\n".join([TEMPLATE.format(ex["db_id"], ex["schema"], ex["question"]) + ex["sql"] for ex in demos])
        prompt = model_inp + "\n\n" + TEMPLATE.format(example["db_id"], example["schema"], example["question"])
    # add instruction
    if prompt_method == 0:
        prompt = INST_CODELLAMA_GEN.format(prompt) + " " + headstr
    elif prompt_method == 2:
        prompt = INST_CUSTOM_GEN.format(prompt) + " " + headstr
    return prompt

# Prompt the generator for 0-shot correction
def prepare_prompt_correction(example, answer_sql,prompt_method=0):
    prompt = TEMPLATE_CORR.format(example["db_id"], example["schema"], example["question"], answer_sql)
    if prompt_method == 0:
        prompt = INST_CODELLAMA_ITER_CORR.format(prompt) + " SELECT"
    return prompt

def generate_completions(generator,prompt,config,device_swap):
    # move model to GPU
    if device_swap: 
        swap_memory(generator.model, device="cuda", verbose=False)
    # generate completions
    responses = generator.generate(prompt, config)
    # move model back to CPU
    if device_swap:
        swap_memory(generator.model, device="cpu", verbose=False)   
    return responses

# experimental extraction. Works so far.
def extract_completionsV2(responses, prompt, headstr):
    # extract completions: collect text after prompt, add back any leading query (headstr) and before the next section (\n\n) if exists
    sql_completions = list(set([normalize_sql((headstr + " " + r.split(prompt)[-1]).split("\n\n")[0]) \
                                for r in responses if (headstr + " " + r.split(prompt)[-1]).split("\n\n")[0] != ""]))
    return sql_completions

def extract_completions(responses,prompt_method=0):
    # extract completions: collect text after [-- SQL:] or [/INST] and before the next section (\n\n) if exists
    if prompt_method == 0:
        sql_completions = list(set([normalize_sql(r.split(" [/INST] ")[-1].split("\n\n")[0]) for r in responses if r.split(" [/INST] ")[-1].split("\n\n")[0] != ""]))
    elif prompt_method == 1:
        sql_completions = list(set([normalize_sql(r.split("-- SQL:\n")[-1].split("\n\n")[0]) for r in responses if r.split("-- SQL:\n")[-1].split("\n\n")[0] != ""]))
    else:
        raise ValueError("Invalid method")
    return sql_completions

def evaluate_completion(evaluator, example, sql_completions, evaluation_config, retriever_eval=None, device_swap=False):
    if device_swap: # move model to GPU
        swap_memory(evaluator.model, device="cuda", verbose=False)

    if retriever_eval is None:
        scores = evaluator.score(example["db_id"], example["question"], sql_completions, evaluation_config)
    else:
        scores = evaluator.score_fewshot(example["db_id"], example["question"], sql_completions, retriever_eval, evaluation_config)

    if device_swap: # move model to GPU
        swap_memory(evaluator.model, device="cpu", verbose=False)
    
    return scores

def log_example(log, example, sql_completions=None, scores=None, candidates_scores=None):
    example_log = deepcopy(example)
    if sql_completions is not None:
        example_log["top_n"] = sql_completions
    if scores is not None:
        example_log["scores"] = scores
    if candidates_scores is not None:
        example_log["candidates"] = candidates_scores
    log.append(example_log)

# rerank completions and return best completion
def rerank(example, generator, evaluator, retriever_gen, retriever_eval, generation_config, evaluation_config, log, device_swap=False, prompt_method=0):
    # load configs
    config = json.load(open(generation_config))
    evaluation_config = json.load(open(evaluation_config))

    # prepare prompt
    prompt = prepare_prompt(example, retriever_gen,prompt_method)
    
    # generate completions
    responses = generate_completions(generator,prompt,config,device_swap)

    # extract completions
    #sql_completions = extract_completions(responses,prompt_method)
    sql_completions = extract_completionsV2(responses, prompt, headstr="SELECT")

    # evaluate completions
    scores = evaluate_completion(evaluator, example, sql_completions, evaluation_config, retriever_eval, device_swap)
    
    # log
    log_example(log, example, sql_completions, scores=scores)

    # return best completion
    return sql_completions[np.argmin(scores)].replace("\n", " ")


# greedy completion. Only one completion is generated and returned
def greedy(example, generator, evaluator, retriever_gen, retriever_eval, generation_config, evaluation_config, log, device_swap=False,prompt_method=0):
    # load configs
    config = json.load(open(generation_config))

    # prepare prompt
    prompt = prepare_prompt(example, retriever_gen,prompt_method)

    # generate completions
    responses = generate_completions(generator,prompt,config,device_swap)

    # extract completions
    #sql_completions = extract_completions(responses,prompt_method)
    sql_completions = extract_completionsV2(responses, prompt, headstr="SELECT")

    # log
    log_example(log, example, sql_completions)

    # return completion
    return sql_completions[0].replace("\n", " ")



def iter_correction(example, generator, evaluator, retriever_gen, retriever_eval, generation_config, evaluation_config, log, device_swap=False, prompt_method=0):
    # load configs
    config = json.load(open(generation_config))
    evaluation_config = json.load(open(evaluation_config))

    # Step 1: Prompt the generator and sample initial plans.
    prompt = prepare_prompt(example, retriever_gen,prompt_method)
    
    # generate completions
    responses = generate_completions(generator,prompt,config,device_swap)

    # extract completions
    sql_completions = extract_completions(responses,prompt_method)

    # Planning iteration setup.
    current_score = 18 #0
    patience = 0
    candidates_scores = {}
    answer_sql = ""

    for t in range(10):
        # Step 2: Score the current batch of plans.
        scores = evaluate_completion(evaluator, example, sql_completions, evaluation_config, retriever_eval, device_swap)

        # Step 3: Find the plan with highest score. Scores are negated for min heap implementation in tree search.
        best_score = min(scores)

        # Step 4: Check termination conditions and replace the old plan with the currently best one, if any.
        if best_score < -0.99:
            answer_sql = sql_completions[np.argmin(scores)]
            current_score = best_score
            candidates_scores[answer_sql] = best_score
            break
        elif best_score >= current_score:
            patience += 1
            if patience >= 3:
                break
        else:
            answer_sql = sql_completions[np.argmin(scores)]
            current_score = best_score
            candidates_scores[answer_sql] = best_score
            patience = 0


        # Step 5: Prompt the generator for 0-shot correction. Sample a new batch of plans.
        prompt = prepare_prompt_correction(example, answer_sql,prompt_method)

        # generate completions
        responses = generate_completions(generator,prompt,config,device_swap)

        # extract completions
        sql_completions = extract_completions(responses,prompt_method)

    answer = answer_sql.replace("\n", " ")

    # log
    log_example(log, example, candidates_scores=candidates_scores)

    return answer