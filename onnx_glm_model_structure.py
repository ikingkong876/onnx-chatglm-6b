import torch
from transformers.modeling_outputs import BaseModelOutputWithPast


class GLMWithLMhead(torch.nn.Module):
    """ Creation of a class to combine the decoder and the lm head """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, position_ids, attention_mask, past_key_values):
        transformer_outputs = self.transformer_forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

        hidden_states = transformer_outputs[0]

        lm_logits = self.model.lm_head(hidden_states).permute(1, 0, 2)

        past_key_values_array = torch.stack([x for x in transformer_outputs[1]]).contiguous()
        return lm_logits, past_key_values_array

    def transformer_forward(self, input_ids, position_ids, attention_mask, past_key_values, use_cache, return_dict):
        inputs_embeds = self.model.transformer.word_embeddings(input_ids)
        hidden_states = inputs_embeds.transpose(0, 1)
        presents = () if use_cache else None

        for i, layer in enumerate(self.model.transformer.layers):
            layer_past = past_key_values[i]
            layer_ret = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_value=layer_past,
                use_cache=use_cache,
            )

            hidden_states = layer_ret[0]

            if use_cache:
                presents = presents + (layer_ret[1],)

            # Final layer norm.
        hidden_states = self.model.transformer.final_layernorm(hidden_states)

        return hidden_states, presents
