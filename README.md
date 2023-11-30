# NLP/LLM入门推荐
----
1. 对于初学者
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
