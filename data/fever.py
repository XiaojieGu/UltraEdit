from typing import Dict
import random
import torch

from data.base import BaseDataset



class FEVERDataset(BaseDataset):

    def __getitem__(self, idx) -> Dict[str, Dict[str, torch.LongTensor]]:
        row = self.data[idx]
        prompt = row["prompt"]
        equiv_prompt = random.choice(row["equiv_prompt"])
        unrel_prompt = row["unrel_prompt"]
        alt = row["alt"]
        ans = row["unrel_ans"]

        return {
            "edit_tuples": self.tok_tuples(prompt, alt),
            "equiv_tuples": self.tok_tuples(equiv_prompt, alt),
            "unrel_tuples": self.tok_tuples(unrel_prompt, ans)
        }


    def tok_tuples(
        self,
        prompt: str,
        answer: str
    ) -> Dict[str, torch.LongTensor]:

        answer = " " + answer
            
        tok_prompt = self.tok(
            prompt,
            return_tensors="pt",
        )
        tok_answer = self.tok(
            answer,
            return_tensors="pt",
            add_special_tokens=False
        )

        tok_tuples = {
            key: torch.cat((value, tok_answer[key][:, :-1]), -1)
            for key, value in tok_prompt.items()
        }
        tok_tuples["labels"] = torch.cat((
            torch.full(tok_prompt["input_ids"].shape, -100)[:, 1:],
            tok_answer["input_ids"]
        ), -1)

        return tok_tuples