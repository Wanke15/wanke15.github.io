#### 1. Skip gram 跳字模型
 - (1) 核心思想：中心词预测所有背景词
 <img src="http://zh.gluon.ai/_images/skip-gram.svg">
 
 - (2) 原始优化目标函数：
   - 中心词向量： V
   - 背景词向量： U
  
   - 给定中心词生成背景词的条件概率可以通过对向量内积做softmax运算得到：
  
      P(w_o \mid w_c) = \frac{\text{exp}(\boldsymbol{u}_o^\top \boldsymbol{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\boldsymbol{u}_i^\top \boldsymbol{v}_c)}
  
   - 假设给定中心词的情况下背景词的生成相互独立，当背景窗口大小为m时，跳字模型的似然函数即给定任一中心词生成所有背景词的概率:
  
      \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)})
      
   - 损失函数：
      
      \sum_{t=1}^{T} \sum_{-m \leq j \leq m,\ j \neq 0} \text{log}\, P(w^{(t+j)} \mid w^{(t)})
  
   - V_c的梯度：
      
      \begin{split}\begin{aligned} \frac{\partial \text{log}\, P(w_o \mid w_c)}{\partial \boldsymbol{v}_c} &= \boldsymbol{u}_o - \frac{\sum_{j \in \mathcal{V}} \exp(\boldsymbol{u}_j^\top \boldsymbol{v}_c)\boldsymbol{u}_j}{\sum_{i \in \mathcal{V}} \exp(\boldsymbol{u}_i^\top \boldsymbol{v}_c)}\\ &= \boldsymbol{u}_o - \sum_{j \in \mathcal{V}} \left(\frac{\text{exp}(\boldsymbol{u}_j^\top \boldsymbol{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\boldsymbol{u}_i^\top \boldsymbol{v}_c)}\right) \boldsymbol{u}_j\\ &= \boldsymbol{u}_o - \sum_{j \in \mathcal{V}} P(w_j \mid w_c) \boldsymbol{u}_j. \end{aligned}\end{split}
      
 - (3) 负采样
   - 背景：
 
 在原始的损失函数中，每次更新权重都有一个softmax计算，包含词典大小数目的项的累加，对于较大的词典来说，计算开销可能过大。
   
   - 思路：
 
 出现在中心词context中的词概率尽量大(类别为1),非context词(噪声词)概率尽量小(类别为0)。从而把原来单词表大小的多分类问题转换成的近似二分类问题，
 两个类别的概率可以通过logistic regression的计算方式得到，输入为中心词和背景词词向量的内积。
   
   
   - 新的损失函数：
   
   假设同时含有正类样本和负类样本的事件P, N_1, \ldots, N_K相互独立，负采样将以上需要最大化的仅考虑正类样本的联合概率改写为

   \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),

   其中条件概率被近似表示为

   P(w^{(t+j)} \mid w^{(t)}) =P(D=1\mid w^{(t)}, w^{(t+j)})\prod_{k=1,\ w_k \sim P(w)}^K P(D=0\mid w^{(t)}, w_k).

   设文本序列中时间步t的词w^{(t)}在词典中的索引为i_t，噪声词w_k在词典中的索引为h_k。有关以上条件概率的对数损失为

   \begin{split}\begin{aligned} -\log P(w^{(t+j)} \mid w^{(t)}) =& -\log P(D=1\mid w^{(t)}, w^{(t+j)}) - \sum_{k=1,\ w_k \sim P(w)}^K \log P(D=0\mid w^{(t)}, w_k)\\ =&- \log\, \sigma\left(\boldsymbol{u}_{i_{t+j}}^\top \boldsymbol{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\left(1-\sigma\left(\boldsymbol{u}_{h_k}^\top \boldsymbol{v}_{i_t}\right)\right)\\ =&- \log\, \sigma\left(\boldsymbol{u}_{i_{t+j}}^\top \boldsymbol{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\sigma\left(-\boldsymbol{u}_{h_k}^\top \boldsymbol{v}_{i_t}\right). \end{aligned}\end{split}

   现在，训练中每一步的梯度计算开销不再与词典大小相关，而与K线性相关。当K取较小的常数时，负采样在每一步的梯度计算开销较小。
