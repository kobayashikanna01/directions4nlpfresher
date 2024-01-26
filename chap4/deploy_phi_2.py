import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
# 如果你将Phi-2的代码和参数存在了/path/to/save/phi_2，可以使用：
# model = AutoModelForCausalLM.from_pretrained("/path/to/save/phi_2", torch_dtype="auto", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("/path/to/save/phi_2", trust_remote_code=True)

# 如果你需要将模型部署到多张GPU上，请按照如下方式：
# pip install accelerate
# 启动python时指定显卡编号，例如在命令行输入CUDA_VISIBLE_DEVICES="0,1,2,3" python
# 进入python之后：
# >>> model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True, device_map="auto")

# 如果显存不够，可以加载bf16编码的模型，几乎不会影响效果，还能节约显存、加快推理速度
# >>> model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

inputs = tokenizer('''def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=200)
# 尝试修改生成模型的超参数
# outputs = model.generate(**inputs,
#                          max_length=200,
#                          repetition_penalty=1.05,
#                          temperature=0.3,
#                          top_p=0.5)

text = tokenizer.batch_decode(outputs)[0]
print(text)
