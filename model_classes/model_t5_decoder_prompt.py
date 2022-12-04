import os
from pathlib import Path

from transformers import T5ForConditionalGeneration
import torch
import torch.nn as nn


class T5PromptTuningMixin:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        decoder_soft_prompt_path = None,
        n_tokens = None,
        initialize_from_vocab = True,
        random_range = 0.5,
        device=None,
        **kwargs,
    ):
        
        cls.device=device
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Make sure to freeze Tranformers model
        for param in model.parameters():
            param.requires_grad = False

        if decoder_soft_prompt_path is not None:
            model.set_decoder_soft_prompt_embeds(decoder_soft_prompt_path)
        elif n_tokens is not None:
            print("Initializing decoder soft prompt...")
            model.initialize_decoder_soft_prompt(
                n_tokens=n_tokens,
                initialize_from_vocab=initialize_from_vocab,
                random_range=random_range,
            )

        return model
    
    def set_device(self, device):
        self.device = device

        
    def set_decoder_soft_prompt_embeds(
        self,
        soft_prompt_path,
    ) :
        """
        Args:
            soft_prompt_path: torch soft prompt file path

        """
        self.decoder_soft_prompt = torch.load(
            soft_prompt_path, map_location=torch.device("cpu")
        )
        self.n_tokens = self.decoder_soft_prompt.num_embeddings
        print(f"Set decoder soft prompt! (n_tokens: {self.n_tokens})")

        
    def initialize_decoder_soft_prompt(
        self,
        n_tokens = 20,
        initialize_from_vocab = True,
        random_range = 0.5,
    ) :
        self.n_tokens = n_tokens
        if initialize_from_vocab:
            init_prompt_value = self.decoder.embed_tokens.weight[:n_tokens].clone().detach()
        else:
            init_prompt_value = torch.FloatTensor(2, 10).uniform_(
                -random_range, random_range
            )
        self.decoder_soft_prompt = nn.Embedding(n_tokens, self.config.d_model)
        # Initialize weight
        self.decoder_soft_prompt.weight = nn.parameter.Parameter(init_prompt_value)

    def _cat_learned_embedding_to_encoder_input(self, input_ids) :
        
        
        inputs_embeds = self.encoder.embed_tokens(input_ids.to(self.device))
        

        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        # [batch_size, n_tokens, n_embd]
        learned_embeds = self.encoder_soft_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)

        inputs_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)

        return inputs_embeds
    
    def _cat_learned_embedding_to_decoder_input(self, input_ids) :
        
        
        inputs_embeds = self.decoder.embed_tokens(input_ids.to(self.device))
        

        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)
            

        # [batch_size, n_tokens, n_embd]
        learned_embeds = self.decoder_soft_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)
        

        inputs_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)
        

        return inputs_embeds
    
    
    def get_embedding_to_input_decoder(self, input_ids):
        inputs_embeds = self.decoder.embed_tokens(input_ids.to(self.device))

        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        # [batch_size, n_tokens, n_embd]
        learned_embeds = self.soft_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)

        inputs_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)

        return inputs_embeds

    def _extend_labels(self, labels, ignore_index=-100) :
            
        if len(list(labels.shape)) == 1:
            labels = labels.unsqueeze(0)

        n_batches = labels.shape[0]
        return torch.cat(
            [
                torch.full((n_batches, self.n_tokens), ignore_index).to(self.device),
                labels.to(self.device),
            ],
            dim=1,
        )
    
    def _extend_encod_ids(self, en_inputs, pad_token=1) :
        if len(list(en_inputs.shape)) == 1:
            en_inputs = en_inputs.unsqueeze(0)

        n_batches = en_inputs.shape[0]
        return torch.cat(
            [
                 
                torch.full((n_batches, self.n_tokens), pad_token,  dtype=torch.long).to(self.device),
                en_inputs,
            ],
            dim=1,
        )

    def _extend_attention_mask(self, attention_mask):

        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        n_batches = attention_mask.shape[0]
        return torch.cat(
            [torch.full((n_batches, self.n_tokens), 1, dtype=torch.long).to(self.device), attention_mask.to(self.device)],
            dim=1,
        )
    def _extend_attention_mask_encode(self, attention_mask):

        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        n_batches = attention_mask.shape[0]
        return torch.cat(
            [torch.full((n_batches, self.n_tokens), 0, dtype=torch.long).to(self.device), attention_mask.to(self.device)],
            dim=1,
        )
    

    def save_soft_prompt(self, path: str, filename: str = "soft_prompt.model"):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.decoder_soft_prompt, os.path.join(path, filename))
        # print(f"Saved soft prompt: {os.path.join(path, filename)}")
        
    
    def generate(self, *args, **kwargs):
        # This fixes CUDA for some reason
        #print("device = ", self.device)
        kwargs['input_ids'] = kwargs['input_ids'].to(self.device)
        
        kwargs['attention_mask']=self._extend_attention_mask_encode(torch.ones([1,kwargs['input_ids'].shape[1]]).long().to(self.device)).to(self.device)
        
        kwargs['input_ids'] = self._extend_encod_ids(kwargs['input_ids']).to(self.device)
        
            
        #print("the shape input embeds = ", kwargs['inputs_embeds'].shape)

        return super().generate( *args, **kwargs)

        
    
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        inputs_embeds=None,
        decoder_input_ids=None,
        decoder_inputs_embeds=None,
        encoder_outputs=None,
        use_cache=None,
        labels=None,
        return_dict=None,
        *args,
        **kwargs
    ):
        
        
        
        if input_ids is not None :
            #inputs_embeds = self._cat_learned_embedding_to_encoder_input(input_ids).to(self.device)
            
            input_ids = self._extend_encod_ids(input_ids).to(self.device)
            
            #print("input ids = ",input_ids, "\n", "shape input ids = ", input_ids.shape)
            #print("the shape input embeds = ", inputs_embeds.shape, "\n")
            
            #print("dec input ids before = ",decoder_input_ids.shape)
            
            #print("dec input ids after = ",decoder_inputs_embeds.shape)
                
            inputs_embeds=None
            
        if decoder_input_ids is not None :
            decoder_inputs_embeds = self._cat_learned_embedding_to_decoder_input(decoder_input_ids).to(self.device)
            #print("decoder input ids = ",decoder_input_ids, "\n", "shape input ids = ", decoder_input_ids.shape)
            #print("came to decoder embedding")
            decoder_input_ids=None
            
       
            
        if labels is not None:
            labels = self._extend_labels(labels).to(self.device)

        
            
        if attention_mask is not None and input_ids is not None  :
                
            attention_mask = self._extend_attention_mask_encode(attention_mask).to(self.device)
            #decoder_attention_mask=attention_mask
            
            #print("attention mask after = ",attention_mask.shape)
            
        if decoder_attention_mask is not None   :
            decoder_attention_mask = self._extend_attention_mask(decoder_attention_mask).to(self.device)
            
        elif decoder_inputs_embeds is not None :
            decoder_attention_mask = self._extend_attention_mask(torch.ones([1,decoder_inputs_embeds.shape[1]-self.n_tokens]).long().to(self.device)).to(self.device)
            #print("dec attn msk 2 = ", decoder_attention_mask)
          
 

        # Drop most of the args for now
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            decoder_input_ids=decoder_input_ids,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
            *args,
            **kwargs
        )


class T5PromptTuningLM(T5PromptTuningMixin, T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

