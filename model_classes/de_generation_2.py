from pathlib import Path
from typing import Union, List

import torch
import torch.nn.functional as F
#from transformers import T5TokenizerFast
#from generation.gpt2_generation import GPT2Generation
#from model_3 import T5PromptTuningLM

from utils import utils
from utils.generation_utils import top_k_top_p_filtering

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

class DExpertsGenerationMod: 
    STOP_TOKEN = "<|endoftext|>"

    def __init__(
        self, 
        expert_model,
        antiexpert_model,
        device,
        seed: int = 42,
    ):
        # Set up device
        self.device = device
        n_gpu = 1
        utils.set_seed(seed, n_gpu)

        self.base_model = expert_model.to(self.device)
        
        if antiexpert_model:
            #self.antiexpert = GPT2LMHeadModel.from_pretrained(antiexpert_model).to(self.device)
            self.antiexpert = antiexpert_model.to(self.device)
        else:
            self.antiexpert = None
            
            
        '''
        if expert_model:
            #self.expert = GPT2LMHeadModel.from_pretrained(expert_model).to(self.device)
            self.expert = expert_model.to(self.device)
        else:
            self.expert = None
        '''
        
        
        #self.tokenizer = tokenizer
        #assert self.tokenizer.eos_token_id == self.tokenizer.pad_token_id

    def __repr__(self):
        return f'<DExpertsGenerator model_name_or_path="{self.model}">'

    def generate(self,
                 input_ids,
                 decoder_input_ids = torch.zeros([1,1]),
                 max_len: int = 20,
                 sample: bool = True,
                 filter_p: float = 1,
                 k: int = 0,
                 p: float = 1.0,
                 temperature: float = 1.0,
                 alpha: float = 0.0,
                 **model_kwargs):
        #if isinstance(prompt, str):
            #prompt = [prompt]

        #encodings_dict = self.tokenizer.encode(prompt, truncation=True, padding=True, return_tensors='pt')

        input_ids = input_ids.to(self.device)
        
        batch_size, input_seq_len = input_ids.shape
        
        attention_mask = torch.ones([batch_size,input_seq_len]).long().to(self.device)

        #position_ids = attention_mask.cumsum(dim=1) - 1 #remove later
        decoder_input_ids=decoder_input_ids.long().to(self.device)
        
        
        unfinished_sents = torch.ones(batch_size, dtype=torch.long, device=self.device)
        
        #print("decoder inputs = ", decoder_input_ids)

        self.base_model.eval()
        #if self.expert:
            #self.expert.eval()
        if self.antiexpert:
            self.antiexpert.eval()
            
        with torch.no_grad():
            for step in range(max_len):
                # base model prediction
                
                base_m_op = self.base_model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, use_cache=False, **model_kwargs)
                base_logits = base_m_op.logits
                
                expert_logits = base_logits
                
                # expert prediction
                
                '''
                if self.expert:

                    expert_logits = base_logits
                else:
                    expert_logits = base_logits
                    
                '''
                
                # antiexpert prediction
                if self.antiexpert:
                    antiexpert_m_op = self.antiexpert(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, use_cache=False, **model_kwargs)
                    antiexpert_logits = antiexpert_m_op.logits
                    
                #print("bs logits = ", base_logits, base_past)
                
                if filter_p < 1.0:
                    base_logits = top_k_top_p_filtering(base_logits, top_p=filter_p)
                
                # DExperts
                alpha = torch.tensor(alpha).to(self.device)
                
                #print("base logs = ", base_logits)
                
                #print("base logs shape = ", base_logits.shape)
                
                #print("antiexp logs = ", antiexpert_logits)
                
                #print("antiexp logs shape = ", antiexpert_logits.shape)
                
                
                ensemble_logits = base_logits + alpha * (expert_logits - antiexpert_logits)
                
                #print("ens logs = ", ensemble_logits)
                
                #print("ens logs shape = ", ensemble_logits.shape)
                
                # in the first decoding step, we want to use the 'real' last position for each sentence
                if step == 0:
                    #last_non_masked_idx = torch.sum(attention_mask, dim=1) - 1
                    last_non_masked_idx = self.base_model.n_tokens + decoder_input_ids.shape[1] -1
                    next_token_logits = ensemble_logits[range(batch_size), last_non_masked_idx, :]
                    #next_token_logits = ensemble_logits[:, -1, :]
                else:
                    next_token_logits = ensemble_logits[:, -1, :]

                if sample:
                    # Temperature (higher temperature => more likely to sample low probability tokens)
                    if temperature != 1.0:
                        next_token_logits = next_token_logits / temperature
                    if k > 0 or p < 1.0:
                        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=k, top_p=p)
                    # Sample
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    # Greedy decoding
                    next_tokens = torch.argmax(next_token_logits, dim=-1)

                # either append a padding token here if <EOS> has been seen or append next token
                tokens_to_add = next_tokens * unfinished_sents   # changed 2

                # this updates which sentences have not seen an EOS token so far
                # if one EOS token was seen the sentence is finished
                #eos_in_sents = tokens_to_add == self.tokenizer.eos_token_id
                #unfinished_sents.mul_((~eos_in_sents).long())

                # stop when there is an EOS in each sentence
                if unfinished_sents.max() == 0:
                    break

                # Update input_ids, attention_mask and position_ids
                decoder_input_ids = torch.cat([decoder_input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                #print("inside dec inp ids = ", decoder_input_ids)
                #attention_mask = torch.cat([attention_mask, attention_mask.new_ones((batch_size, 1))], dim=1)
                #position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)

        #decoded_outputs = [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in input_ids[:, input_seq_len:]]
        return decoder_input_ids