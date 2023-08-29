# onnx-chatglm-6b
参考：
ChatGLM-MNN[https://github.com/wangzhaode/ChatGLM-MNN]
fastT5[https://github.com/Ki6an/fastT5]
将chatglm-6b的模型结构修改掉，并成功将pytorch权重转化为onnx，支持float32和float16两种数据类型，支持batch推理，具体加速效果，各位看官可以自己测试看看。
