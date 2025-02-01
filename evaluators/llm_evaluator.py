from utils.constants import TEMPLATE_EVAL, TEMPLATE_EVAL_RES, INST_CODELLAMA_EVAL, INST_CODELLAMA_EVAL_RES, INST_CODELLAMA_EVAL_DB

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import numpy as np
import torch
import sqlite3


class LLMEvaluator():
    # yes_token_indx: 
    # The index of the token in the vocabulary that corresponds to the "Yes" text.
    # CodeLlama-Instruct: "No" 1939 "Yes" 3869
    # TinyLlama: "Yes" 3869

    def __init__(self, model_name_or_dir, db_path, device="cuda",yes_token_indx=None):
        # load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_dir,
                use_fast=False
            )
        except:
            # load tokenizer without use_fast=False
            # for stable-code-3b model
            self.tokenizer = AutoTokenizer.from_pretrained( model_name_or_dir )
            Warning("Tokenizer is not loaded with use_fast=False")

        # load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )#.to(device)
        self.model.eval()

        # set other params
        self.db_path = db_path
        self.device = device
        if yes_token_indx:
            self.yes_token_indx = yes_token_indx
            self.validate_yes_token()
        else:
            self.yes_token_indx = self.get_yes_token()

    def get_yes_token(self):
        return int( self.tokenizer.encode("Yes")[-1])
    
    def validate_yes_token(self):
        assert self.get_yes_token() == self.yes_token_indx

    def score(self, db_id, question, candidates, evaluation_config):
        # load db
        db_path=f'{self.db_path}/{db_id}/{db_id}.sqlite'
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        scores = []
        for cand_sql in candidates:
            result = ""

            try:
                cursor.execute(cand_sql)
            except:
                if evaluation_config["check_exec"]:
                    scores.append(1)
                    continue
                else:
                    result = "ERROR"

            if evaluation_config["use_exec_res"]:
                if result != "ERROR":
                    result = [(c[0], []) for c in cursor.description]
                    rows = []
                    for i in range(5):
                        row = cursor.fetchone()
                        if row is None:
                            break
                        rows.append(row)

                    if i == 0:
                        result = "None"
                    else:
                        for values in rows:
                            for c, v in zip(result, values):
                                c[1].append((v[:128] + "..." if type(v) == str and len(v) > 128 else str(v)))

                        result = "-- " + "\n-- ".join([c[0].lower() + ": " + ", ".join(c[1]) for c in result])
                # create prompt
                batch = self.tokenizer(
                    INST_CODELLAMA_EVAL_RES.format(
                        TEMPLATE_EVAL_RES.format(question, cand_sql, result)
                    ),
                    return_tensors="pt", 
                    add_special_tokens=False
                )
            else:
                # create prompt
                batch = self.tokenizer(
                    INST_CODELLAMA_EVAL.format(
                        TEMPLATE_EVAL.format(question, cand_sql)
                    ),
                    return_tensors="pt", 
                    add_special_tokens=False
                )
            # move to device    
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # forward pass
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                # negative of softmax as score. so, lower is better
                scores.append(
                    - torch.nn.functional.softmax(
                        outputs["logits"][:, -1, :], dim=-1
                    ).flatten()[self.yes_token_indx].item()
                )

        return scores


    def score_fewshot(self, db_id, question, candidates, retriever, evaluation_config):
        demos = retriever.retrieve(question)

        conn = sqlite3.connect(f'{self.db_path}/{db_id}/{db_id}.sqlite')
        cursor = conn.cursor()

        scores = []
        for cand_sql in candidates:
            result = ""

            try:
                cursor.execute(cand_sql)
            except:
                if evaluation_config["check_exec"]:
                    scores.append(1)
                    continue
                else:
                    result = "ERROR"

            if evaluation_config["use_exec_res"]:
                if result != "ERROR":
                    result = [(c[0], []) for c in cursor.description]
                    rows = []
                    for i in range(5):
                        row = cursor.fetchone()
                        if row is None:
                            break
                        rows.append(row)

                    if i == 0:
                        result = "None"
                    else:
                        for values in rows:
                            for c, v in zip(result, values):
                                c[1].append((v[:128] + "..." if type(v) == str and len(v) > 128 else str(v)))

                        result = "-- " + "\n-- ".join([c[0].lower() + ": " + ", ".join(c[1]) for c in result])
                
                prompt_strs = []
                for d in demos:
                    prompt_strs.append(
                        TEMPLATE_EVAL_RES.format(d["question"], d["sql"], d["exec_res"]) + "\n-- Answer: Yes"
                    )
                    prompt_strs.append(
                        TEMPLATE_EVAL_RES.format(d["question"], d["neg_sql"], d["neg_exec_res"]) + "\n-- Answer: No"
                    )
                prompt_strs.append(
                    TEMPLATE_EVAL_RES.format(question, cand_sql, result)
                )
                
                batch = self.tokenizer(
                    INST_CODELLAMA_EVAL_RES.format(
                        "\n\n".join(prompt_strs)
                    ),
                    return_tensors="pt", 
                    add_special_tokens=False
                )
            else:
                prompt_strs = []
                for d in demos:
                    prompt_strs.append(
                        TEMPLATE_EVAL.format(d["question"], d["sql"]) + "\n-- Answer: Yes"
                    )
                    prompt_strs.append(
                        TEMPLATE_EVAL.format(d["question"], d["neg_sql"]) + "\n-- Answer: No"
                    )
                prompt_strs.append(
                    TEMPLATE_EVAL.format(question, cand_sql)
                )

                batch = self.tokenizer(
                    INST_CODELLAMA_EVAL.format(
                        "\n\n".join(prompt_strs)
                    ),
                    return_tensors="pt", 
                    add_special_tokens=False
                )

            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
            
                scores.append(
                    - torch.nn.functional.softmax(
                        outputs["logits"][:, -1, :], dim=-1
                    ).flatten()[self.yes_token_indx].item()
                )

        return scores


class LLMLoraEvaluator(LLMEvaluator):

    def __init__(self, model_name_or_dir, peft_model_dir, db_path, device="cuda",yes_token_indx=3869):
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_dir,
            use_fast=False
        )
        # load model
        # first, set up the quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,  
            llm_int8_threshold=6.0, 
            llm_int8_enable_fp32_cpu_offload=True  
        )
        # then, load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_dir,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )#.to(device)
        
        # load the PEFT model
        self.model = PeftModel.from_pretrained(
            self.model,
            peft_model_dir
        )
        self.model.eval()

        # set other params
        self.db_path = db_path
        self.device = device
        if yes_token_indx:
            self.yes_token_indx = yes_token_indx
            self.validate_yes_token()
        else:
            self.yes_token_indx = self.get_yes_token()