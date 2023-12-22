# NLP/LLM入门推荐

### 1. 对于初学者

  **请尽量按顺序阅读**
* 网课Stanford CS224N: Natural Language Processing with Deep Learning
  * 网站：[CS224N](https://web.stanford.edu/class/cs224n/)
  * 视频：[Winter 2021 - Youtube](https://www.youtube.com/playlist?list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ) [Winter 2021 - B站](https://www.bilibili.com/video/BV18Y411p79k/)
  * 课程网站上有最近一次开课的Slides，但是还没有对应的视频，建议先看视频和2021冬季的Slides，再自行决定是否看最新的Slides。 
* 网课Stanford CS230: Deep Learning
  * [Video](https://cs230.stanford.edu/lecture/)
  * 可以优先看Lecture 1-3, 5-7, 8
* 阅读：
  * [What is Word Embedding?](https://en.wikipedia.org/wiki/Word_embedding)
  * [What is Tokenization?](https://www.kaggle.com/code/satishgunjal/tokenization-in-nlp)
  * 选读：Word2Vec [[论文](https://arxiv.org/abs/1310.4546)] [[Wiki](https://en.wikipedia.org/wiki/Word2vec)]
  * 回答：
    * 如何用向量表征一个词、甚至一个短语、一个句子？
    * 我们也可以用一个系数的、很高维的向量来表示一个句子。例如中文有2万个常用词，那么我们可以使用2万维的向量表示一个句子，2万维的每个位置都对应一个具体的词。例如句子“我爱北京天安门”，我们可以将 我、爱、北京、天安门 这四个词对应的维度设置为1，其余维度设置为0来表示整个句子。这种表示方式，相比于embedding，有哪些缺点？
* 论文：Attention Is All You Need [[arxiv](https://arxiv.org/pdf/1706.03762.pdf)]
  * 大语言模型的基石：attention机制与Transformer结构
  * 看完后希望你能回答：
    * self-attention的K、Q、V是什么？
    * 多头注意力机制的计算是如何实现的，以hidden state为4096维、有32个attention head为例？
    * 为什么要过Softmax
* 论文：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
  * [[论文](https://arxiv.org/abs/1810.04805)]
  * 回答：从Encoder和Decoder的角度，BERT和最原始的Transformer（Attention Is All You Need中的）有什么区别？

### 2. 快速上手Python编程

* 尝试在自己的机器上部署Python3（不要安装Python2）
* 了解主函数、输入输出函数的使用
* 尝试Python的两种执行方式：交互式执行、提交式执行
* 了解基本数据结构：
  * 掌握list、dict、set、tuple的特性和使用方法
  * 掌握函数的定义
  * 掌握class的定义、__init__、初步掌握class的继承
* 掌握json文件的读写，json格式字符串与list、dict对象之间的互相转化（尤其注意dict对象的KEY为数字或KEY为字符串时，该对象与json格式的字符串互相转换之间的不一致性）
* 测试：用Python编写程序，实现BFS走迷宫([链接](https://github.com/kobayashikanna01/directions4nlpfresher/blob/main/chap2/bfs_test.py))
