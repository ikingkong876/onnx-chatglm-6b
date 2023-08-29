import os
import time
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
import onnxruntime
from typing import Optional, Tuple
import numpy as np
from onnx_glm_exporter import transformers_onnx_pipeline
from modeling_chatglm import ChatGLMForConditionalGeneration, PreTrainedModel


class GLMModelForOnnxGeneration(ChatGLMForConditionalGeneration):
    def __init__(
            self, onnx_model_path: str, model_path="", config=None, fp16=False, threads: int = 0
    ):
        self.fp16 = fp16
        if config is None:
            self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        super(ChatGLMForConditionalGeneration, self).__init__(self.config)
        self.max_sequence_length = self.config.max_sequence_length
        self.position_encoding_2d = self.config.position_encoding_2d
        sess_options = onnxruntime.SessionOptions()
        if threads:
            sess_options.intra_op_num_threads = threads
        self.session = onnxruntime.InferenceSession(onnx_model_path, sess_options,
                                                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.t = torch.zeros(1).to('cuda:0')  # 给整个模块加一个torch的变量，方便获取device信息，如果不加这一行，to(device)会报错

    @classmethod
    def from_pretrained(cls, model_name_path: str, fp16=False, threads=0):
        if fp16:
            onnx_path = os.path.join(model_name_path, "onnx_fp16/model.onnx")
        else:
            onnx_path = os.path.join(model_name_path, "onnx/model.onnx")
        if not os.path.exists(onnx_path):
            transformers_onnx_pipeline(model_name_path)
        return cls(onnx_path, model_path=model_name_path, threads=threads, fp16=fp16)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        if past_key_values is None:
            n_layer = self.config.num_layers
            n_head = self.config.num_attention_heads
            n_embd = self.config.hidden_size
            embed_size_per_head = int(n_embd / n_head)
            past_key_values_array = np.zeros([n_layer, 2, 0, 1, n_head, embed_size_per_head], dtype=np.float32)
            if self.fp16:
                past_key_values_array = past_key_values_array.astype(np.float16)
            attention_mask = attention_mask.cpu().numpy()
            position_ids = position_ids.cpu().numpy()
            ort_inputs = {
                "input_ids": input_ids.cpu().numpy(),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values_array,
            }
            logits, key_value_tensor = self.session.run(None, ort_inputs)

            return CausalLMOutputWithPast(
                loss=None,
                logits=torch.HalfTensor(logits).to(input_ids.device),
                past_key_values=torch.HalfTensor(key_value_tensor).to(input_ids.device),
            )
        else:
            device_type = str(input_ids.device).split(":")[0]
            binding = self.session.io_binding()

            binding.bind_input(
                name="input_ids",
                device_type=device_type,
                device_id=0,
                element_type=np.int64,
                shape=input_ids.shape,
                buffer_ptr=input_ids.data_ptr()
            )
            binding.bind_input(
                name="position_ids",
                device_type=device_type,
                device_id=0,
                element_type=np.int64,
                shape=position_ids.shape,
                buffer_ptr=position_ids.data_ptr()
            )

            binding.bind_input(
                name="attention_mask",
                device_type=device_type,
                device_id=0,
                element_type=bool,
                shape=attention_mask.shape,
                buffer_ptr=attention_mask.data_ptr()
            )

            past_key_values_array = past_key_values
            binding.bind_input(
                name="past_key_values",
                device_type=device_type,
                device_id=0,
                element_type=np.float16 if self.fp16 else np.float,
                shape=past_key_values_array.shape,
                buffer_ptr=past_key_values_array.data_ptr()
            )

            logits_tensor = torch.empty(
                (input_ids.shape[0], input_ids.shape[1], 130528),
                dtype=torch.half if self.fp16 else torch.float, device=input_ids.device
            )
            binding.bind_output(
                name=self.session.get_outputs()[0].name,
                device_type=device_type,
                device_id=0,
                element_type=np.float16 if self.fp16 else np.float,
                shape=logits_tensor.shape,
                buffer_ptr=logits_tensor.data_ptr())
            key_value_tensor = torch.empty(
                (28, 2, past_key_values_array.shape[2] + 1, past_key_values_array.shape[3], 32, 128),
                dtype=torch.half if self.fp16 else torch.float, device=input_ids.device
            ).contiguous()
            binding.bind_output(
                name=self.session.get_outputs()[1].name,
                device_type=device_type,
                device_id=0,
                element_type=np.float16 if self.fp16 else np.float,
                shape=key_value_tensor.shape,
                buffer_ptr=key_value_tensor.data_ptr())

            self.session.run_with_iobinding(binding)

            return CausalLMOutputWithPast(
                loss=None,
                logits=logits_tensor,
                past_key_values=key_value_tensor.to(input_ids.device),
            )

    @staticmethod
    def _reorder_cache(
            past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past
        )

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past: Optional[torch.Tensor] = None,
            past_key_values: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            **kwargs
    ):
        batch_size, seq_length = input_ids.shape
        MASK, gMASK = self.config.mask_token_id, self.config.gmask_token_id
        seqs = input_ids.tolist()
        mask_positions, use_gmasks = [], []
        for seq in seqs:
            mask_token = gMASK if gMASK in seq else MASK
            use_gmask = mask_token == gMASK
            mask_positions.append(seq.index(mask_token))
            use_gmasks.append(use_gmask)

        # only last token for input_ids if past is not None
        if past is not None or past_key_values is not None:
            last_token = input_ids[:, -1].unsqueeze(-1)
            if attention_mask is not None and attention_mask.dtype == torch.bool:
                attention_mask = attention_mask[:, :, -1:]
            else:
                attention_mask = torch.zeros(
                    size=(input_ids.shape[0], 1, 1, 1), dtype=torch.bool, device=input_ids.device
                ).contiguous()
            if position_ids is not None:
                position_ids = position_ids[..., -1:]
            else:
                context_lengths = [seq.index(self.config.bos_token_id) for seq in seqs]
                if self.position_encoding_2d:
                    position_ids = torch.tensor(
                        [[mask_position, seq_length - context_length] for mask_position, context_length in
                         zip(mask_positions, context_lengths)], dtype=torch.long, device=input_ids.device).unsqueeze(-1)
                else:
                    position_ids = torch.tensor([mask_position for mask_position in mask_positions], dtype=torch.long,
                                                device=input_ids.device).unsqueeze(-1)

            if past is None:
                past = past_key_values

            return {
                "input_ids": last_token.contiguous(),
                "past_key_values": past,
                "position_ids": position_ids.contiguous(),
                "attention_mask": attention_mask.contiguous()
            }
        else:
            attention_mask = self.get_masks(
                input_ids,
                device=input_ids.device
            )
            position_ids = self.get_position_ids(
                input_ids,
                device=input_ids.device,
                mask_positions=mask_positions,
                use_gmasks=use_gmasks
            )

            return {
                "input_ids": input_ids.contiguous(),
                "past_key_values": past,
                "position_ids": position_ids.contiguous(),
                "attention_mask": attention_mask.contiguous()
            }


if __name__ == '__main__':
    print("onnx模型开始加载...")
    start = time.time()
    model = GLMModelForOnnxGeneration.from_pretrained(model_name_path='F:\\BERT\\chatglm-6b', fp16=True,
                                                      threads=16).half().to('cuda:0')
    tokenizer = AutoTokenizer.from_pretrained('F:\\BERT\\chatglm-6b', trust_remote_code=True)
    print(f"onnx模型加载完毕！用时{round((time.time() - start) * 1000, 3)}ms")
    print("onnx模型开始推理...")
    print(
        tokenizer.batch_decode(
            model.generate(
                input_ids=tokenizer(['10月25日,金固股份曾发布公告,公司实控人孙锋峰、孙金国、孙利群及一致行动人孙曙虹已累计质押3.39亿股,约占所持公股份总数的93.57%,这句话有几类要素?',
                                     '1945年2月，在雅尔达会议上，美国、英国为换取苏联出兵东北，减少美国的牺牲，未经中国国民政府同意赋予苏联大连国际化、苏联在大连港特殊权力、苏联租用旅顺港设立海军基地以及苏联在中东铁路和南满铁路特权。这句话中提到的雅尔达会议是什么时候召开的?'],
                                    return_tensors="pt", truncation=True, padding=True, max_length=2000
                                    ).input_ids.to(model.device),
                max_length=100, do_sample=False, temperature=1.), skip_special_tokens=True)
    )
    start = time.time()
    for i in range(10):
        model.chat(
            tokenizer,
            '10月25日,金固股份曾发布公告,公司实控人孙锋峰、孙金国、孙利群及一致行动人孙曙虹已累计质押3.39亿股,约占所持公股份总数的93.57%,这句话有几类要素?',
            max_length=100, do_sample=False, temperature=1.
        )
    print(f"onnx模型推理完毕！用时{round((time.time() - start) * 1000 / 10, 3)}ms")
    print(
        model.chat(
            tokenizer,
            '10月25日,金固股份曾发布公告,公司实控人孙锋峰、孙金国、孙利群及一致行动人孙曙虹已累计质押3.39亿股,约占所持公股份总数的93.57%,这句话有几类要素?',
            max_length=100, do_sample=False, temperature=1.
        )
    )
    print(
        model.chat(
            tokenizer,
            '1945年2月，在雅尔达会议上，美国、英国为换取苏联出兵东北，减少美国的牺牲，未经中国国民政府同意赋予苏联大连国际化、苏联在大连港特殊权力、苏联租用旅顺港设立海军基地以及苏联在中东铁路和南满铁路特权。这句话中提到的雅尔达会议是什么时候召开的?',
            max_length=100, do_sample=False, temperature=1.
        )
    )
