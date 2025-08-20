import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.helpers import seed_everything


class BaseLLM:
    def __init__(self, model_id: str = None, seed: int = None):
        # Set variables
        self.model_id = model_id
        self.seed = seed

        # Instantiate model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
        )

        # Convert to left padding if necessary
        if self.tokenizer.padding_side != "left":
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_completion(
        self,
        prompt: str,
        generation_kwargs: dict,
    ) -> list[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        if self.seed is not None:
            seed_everything(self.seed)
        outputs = self.model.generate(**inputs, **generation_kwargs)
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded_outputs

    def get_response(self, query: str, generation_kwargs: dict) -> str:
        # Format with the HF chat API
        conversation = [
            {"role": "user", "content": f"{query}"},
        ]
        inputs = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.to(self.model.device)
        outputs = self.model.generate(**inputs, **generation_kwargs)
        decoded_outputs = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Only keep the response
        response_only = decoded_outputs.partition(query)[2].strip()
        return response_only

    def convert_to_tok_ids(
        self, tokens: list[str] | list[list[str]]
    ) -> list[int] | list[list[int]]:
        # Backward compatible with tokens as list[str]
        if isinstance(tokens[0], str):
            tokens = [tokens]
        tok_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in tokens]
        assert all(len(t) == len(t_ids) for t, t_ids in zip(tokens, tok_ids)), (
            "One of the tokens could not be converted to a SINGLE token ID."
            + "This is not currently supported for next token probabilities.\n"
            + f"{tokens=}, {tok_ids=}"
        )
        return tok_ids

    def get_next_token_probs(
        self, prompt: str | list[str], next_tokens: list[str] | list[list[str]]
    ) -> list[dict[str, float]]:
        # Backward compatible with prompt as str
        assert isinstance(prompt, str) or (len(prompt) == len(next_tokens))
        if isinstance(prompt, str):
            prompt = [prompt]
            next_tokens = [next_tokens]
        assert (
            self.tokenizer.padding_side == "left"
        ), "This function assumes left padding so it can use -1 to get the next token probability."
        tok_ids = self.convert_to_tok_ids(next_tokens)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        tokenized_prompt = self.tokenizer(
            prompt, padding=True, truncation=False, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            output_logits = self.model(**tokenized_prompt)["logits"]
        assert output_logits.shape[0] == len(prompt)
        next_token_logits = output_logits[:, -1, :]
        # NOTE: Compute softmax over ALL tokens in the vocab
        # and then restrict to only the tokens in next_tokens
        all_next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        subset_next_token_probs = torch.gather(
            input=all_next_token_probs,
            dim=-1,
            index=torch.tensor(tok_ids).to(self.model.device),
        )
        next_token_probs = (
            subset_next_token_probs / subset_next_token_probs.sum(dim=-1).reshape(-1, 1)
        ).tolist()
        next_token_probs = [
            dict(zip(nt, nt_probs))
            for nt, nt_probs in zip(next_tokens, next_token_probs)
        ]
        return next_token_probs

    def get_completion_log_probs(self, prompt: str, completion: str) -> float:
        """
        get_completion_log_probs(llm, prompt, completion) takes a prompt and a completion and returns the log probability of the completion given the prompt.
        Note that we normalize according to the number of tokens in the completion.
        """
        # Tokenize the full text
        full_text = prompt + " " + completion
        tokenized_full_text = self.tokenizer(full_text, return_tensors="pt").to(
            self.model.device
        )
        full_text_ids = tokenized_full_text.input_ids
        tokenized_prompt_ids = (
            self.tokenizer(prompt, return_tensors="pt").to(self.model.device).input_ids
        )
        # Pass the full text through the model
        with torch.no_grad():
            output = self.model(**tokenized_full_text)
            logits = output["logits"]
        # Get the log probabilities of the completion tokens
        completion_start = tokenized_prompt_ids.shape[1]
        completion_logits = logits[0, completion_start - 1 : -1]
        completion_log_probs = torch.nn.functional.log_softmax(
            completion_logits, dim=-1
        )  # Note the use of log_softmax here
        completion_token_ids = full_text_ids[0, completion_start:]
        # Sum the log probabilities for just the tokens observed in the completion
        total_log_prob = 0.0
        for i, token_id in enumerate(completion_token_ids):
            token_log_prob = completion_log_probs[i, token_id]
            total_log_prob += token_log_prob.item()
        # Normalize by the number of tokens in the completion
        total_log_prob = total_log_prob / len(completion_token_ids)
        return total_log_prob
