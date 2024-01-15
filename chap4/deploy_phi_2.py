import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
# 如果你将Phi-2的代码和参数存在了/path/to/save/phi_2，可以使用：
# model = AutoModelForCausalLM.from_pretrained("/path/to/save/phi_2", torch_dtype="auto", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("/path/to/save/phi_2", trust_remote_code=True)

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
