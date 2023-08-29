import os
import time

import onnx
import torch
import numpy as np
import onnxruntime

from onnx_glm_model_structure import (
    GLMWithLMhead,
)
from onnxconverter_common import float16
from onnxruntime.quantization import (
    quantize_dynamic,
    QuantType
)
from modeling_chatglm import (
    ChatGLMForConditionalGeneration,
)


def generate_onnx_representation(model, onnx_name_path=None) -> str:
    os.makedirs(os.path.join(model.config._name_or_path, "onnx"), exist_ok=True)
    if onnx_name_path is None:
        onnx_name_path = os.path.join(model.config._name_or_path, "onnx", "model.onnx")
    n_layer = model.config.num_layers
    n_head = model.config.num_attention_heads
    n_embd = model.config.hidden_size
    embed_size_per_head = int(n_embd / n_head)
    model_wrapper = GLMWithLMhead(model)
    model_wrapper.eval()
    past_key_values = torch.randn([n_layer, 2, 0, 1, n_head, embed_size_per_head])

    test_inputs = {
        "input_ids": torch.LongTensor([[5, 9, 8, 63837, 10, 15, 63855, 6, 63958,
                                        67012, 66080, 64434, 102831, 6, 63900, 64900, 66030, 63840,
                                        65309, 67863, 64993, 63824, 65309, 63958, 64090, 63824, 65309,
                                        64294, 64611, 63868, 67606, 65143, 63840, 65309, 109843, 70005,
                                        117807, 80757, 13, 7, 13, 18, 92729, 6, 80148,
                                        92294, 64429, 66080, 83377, 18, 13, 7, 15, 25,
                                        125673, 68584, 63832, 111669, 68214, 31, 130001, 130004]]),
        "position_ids": torch.LongTensor([[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                                            34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                            51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 60],
                                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]]),
        "attention_mask": torch.BoolTensor([[[[False, ] * 62] * 61 + [[False] * 61 + [True]]]]),
        "past_key_values": past_key_values,
    }
    print("onnx dummpy输入配置")
    dummy_inputs = tuple([x for x in test_inputs.values()])
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "position_ids": {0: "batch_size", 2: "seq_len"},
        "attention_mask": {0: "batch_size", 2: "seq_len", 3: "seq_len"},
        "past_key_values": {2: "past_seq_len", 3: "batch_size"},
    }
    print("onnx转换中...")
    # 导出
    torch.onnx.export(
        model_wrapper,
        args=dummy_inputs,
        f=onnx_name_path,
        input_names=[
            "input_ids",
            "position_ids",
            "attention_mask",
            "past_key_values",
        ],
        verbose=True,
        output_names=["logits", "output_past_key_values"],
        dynamic_axes=dynamic_axes,
    )
    return onnx_name_path


def create_test_input(config, fp16=False, type='np'):
    n_layer = config.num_layers
    n_head = config.num_attention_heads
    n_embd = config.hidden_size
    embed_size_per_head = int(n_embd / n_head)
    inputs = {"input_ids": np.array([[5, 9, 8, 63837, 10, 15, 63855, 6, 63958,
                                      67012, 66080, 64434, 102831, 6, 63900, 64900, 66030, 63840,
                                      65309, 67863, 64993, 63824, 65309, 63958, 64090, 63824, 65309,
                                      64294, 64611, 63868, 67606, 65143, 63840, 65309, 109843, 70005,
                                      117807, 80757, 13, 7, 13, 18, 92729, 6, 80148,
                                      92294, 64429, 66080, 83377, 18, 13, 7, 15, 25,
                                      125673, 68584, 63832, 111669, 68214, 31, 130001, 130004]], dtype=np.int64),
              "position_ids": np.array([[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                          17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                                          34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                          51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 60],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]], dtype=np.int64),
              "attention_mask": np.zeros([1, 1, 62, 62], dtype=bool),
              "past_key_values": np.random.normal(
                  size=(n_layer, 2, 0, 1, n_head, embed_size_per_head)
              ).astype(np.float16 if fp16 else np.float32)}
    inputs["attention_mask"][..., -1] = True
    if type == 'pt':
        inputs['input_ids'] = torch.from_numpy(inputs['input_ids'])
        inputs['position_ids'] = torch.from_numpy(inputs['position_ids'])
        inputs['attention_mask'] = torch.from_numpy(inputs['attention_mask'])
        inputs['past_key_values'] = torch.torch.HalfTensor(inputs['past_key_values']) if fp16 else torch.from_numpy(inputs['past_key_values'])
    return inputs


def test_torch_inference(model):
    print("torch加载测试...")
    print("torch模型开始加载...")
    start = time.time()
    model_wrapper = GLMWithLMhead(model)
    model_wrapper.eval()
    print(f"torch模型加载完毕！用时{round((time.time() - start) * 1000, 3)}ms")
    torch_inputs = {k: torch.from_numpy(v) for k, v in create_test_input(model.config).items()}
    start = time.time()
    with torch.no_grad():
        output = model_wrapper(**torch_inputs)
    print("#" * 10 + "\ntorch格式输出：\n")
    for o in output:
        print(o)
        print(o.shape)
    print(f"torch推断用时{round((time.time() - start) * 1000, 3)}ms")


def test_onnx_inference(onnx_name_path: str, config):
    print("onnx加载测试...")
    print("onnx模型开始加载...")
    start = time.time()
    session = onnxruntime.InferenceSession(onnx_name_path, providers=['CPUExecutionProvider'])
    print(f"onnx模型加载完毕！用时{round((time.time() - start) * 1000, 3)}ms")
    ort_inputs = create_test_input(config)
    start = time.time()
    output = session.run(None, ort_inputs)
    print("#" * 10 + "\nonnx格式输出：\n")
    for o in output:
        print(o)
        print(o.shape)
    print(f"onnx推断用时{round((time.time() - start) * 1000, 3)}ms")


def transformers_onnx_pipeline(model_name_path: str = "chatglm-6b"):
    model = ChatGLMForConditionalGeneration.from_pretrained(
        model_name_path, trust_remote_code=True).float().eval()
    test_torch_inference(model)
    onnx_path = generate_onnx_representation(model)
    print(onnx_path)
    test_onnx_inference(onnx_path, model.config)


if __name__ == '__main__':
    model_path = 'F:\\BERT\\chatglm-6b'
    transformers_onnx_pipeline(model_path)
    onnx_model_path = os.path.join(model_path, 'onnx/model.onnx')
    os.makedirs(os.path.join(model_path, 'onnx_fp16'), exist_ok=True)
    fp16_model = float16.convert_float_to_float16_model_path(onnx_model_path)
    onnx.save(fp16_model, os.path.join(model_path, 'onnx_fp16', 'model.onnx'), save_as_external_data=True)
