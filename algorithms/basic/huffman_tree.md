## 1. 背景
最近在做搜索场景下的“猜你喜欢”，由于还在初始阶段，所以没有用复杂的算法。主要是基于query和加购商品的共现做i2i、矩阵分解、item2vec。在搞item2vec时又重温了word2vec算法，之前都是直接用gensim，
对相关的理论也是知道原理，然后一带而过。这次希望能够对细节有更深入的理解

## 2. Word2vec 要点
word2vec 是一个单隐层的神经网络，有两种具体的结构实现：
 - 中心词预测上下文(Skip Gram) 
 - 上下文预测中心词(CBOW)
 
理论上讲用分类算法softmax即可进行训练优化，但在文本和商品领域动辄上百万、千万的数量级场景下，其训练计算效率就会面临巨大挑战。为了提高计算效率，提出了两种计算优化方法：
 - hierarchy softmax
 - negative sampling
 
 ## 3. Hierarchy softmax
 理解 Hierarchy softmax 的第一步是理解霍夫曼树。这里实现了构建霍夫曼树和树的前序遍历算法：
 ```python
from graphviz import Digraph

from collections import Counter
from operator import itemgetter


class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __str__(self):
        return "{}".format(self.val)

    def __eq__(self, other):
        if self.val == other.val:
            if self.left == other.left and self.right == other.right:
                return True
            else:
                return False
        else:
            return False


class WeightedTreeNode(TreeNode):
    def __init__(self, val, left=None, right=None, weight=1):
        super().__init__(val, left, right)
        self.weight = weight


class HuffmanTree(object):
    def __init__(self):
        pass

    @staticmethod
    def char_counter(_text):
        # 叶子节点的初始权重：出现频率
        stats = Counter(_text)
        return stats

    def _merge(self, nodes):
        # 权重最低的节点
        minimal_node = nodes.pop()
        # 权重次低的节点
        second_minimal_node = nodes.pop()
        # 生成的中间节点
        new_node = WeightedTreeNode("{}+{}".format(minimal_node.val, second_minimal_node.val),
                                    left=minimal_node, right=second_minimal_node,
                                    weight=minimal_node.weight + second_minimal_node.weight)
        nodes.append(new_node)
        # 依节点权重排序
        nodes.sort(key=lambda e: (e.weight, e.val), reverse=True)
        # 递归结束条件：只剩根节点
        if len(nodes) == 1:
            return
        else:
            # 递归调用
            self._merge(nodes)

    def build(self, text):
        stats = self.char_counter(text)
        # 创建所有的叶子节点
        all_nodes = [WeightedTreeNode(kv[0], weight=kv[1]) for kv in sorted(stats.items(), key=itemgetter(1), reverse=True)]
        self._merge(all_nodes)
        root = all_nodes[0]
        return root


def visit(_node, _res):
    _res.append(_node)

# 前序遍历
def pre_order(_tree: TreeNode, res: list):
    if _tree is not None:
        visit(_tree, res)
        pre_order(_tree.left, res)
        pre_order(_tree.right, res)


if __name__ == '__main__':
    # 测试文本
    texts = "I love China and I love living in China".split()
    
    # 霍夫曼树
    huffman_tree = HuffmanTree()
    tree = huffman_tree.build(texts)

    # 可视化
    def plot(node: WeightedTreeNode, graph: Digraph):
        graph.node(node.val, "{}:{}".format(node.val, node.weight))
        if node.left is not None:
            graph.node(node.left.val, "{}:{}".format(node.left.val, node.left.weight))
            graph.edge(node.val, node.left.val)
            plot(node.left, graph)
        if node.right is not None:
            graph.node(node.right.val, "{}:{}".format(node.right.val, node.right.weight))
            graph.edge(node.val, node.right.val)
            plot(node.right, graph)

    dot = Digraph(comment='HuffmanTree')
    plot(tree, dot)
    dot.view()
 ```
 <img src="./assets/huffman_tree_with_weight.png">
