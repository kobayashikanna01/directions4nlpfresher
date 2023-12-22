import os
import sys
import random

"""
要求：修改bfs_visit函数的执行部分，不要修改其输入参数，实现用BFS算法走迷宫。
    计算从（0，0）走到（m-1，n-1）所需的最小步数。
    只允许从「通路」移动到「通路」，不可以从「通路」移动到「墙」。
    每步只允许向正上、正下、正左、正右移动1步，不可以跳格或者斜向移动。
    不可以走出迷宫。
    禁止递归调用。
输入：
    m：int类型，迷宫矩阵的行数。
    n：int类型，迷宫矩阵的列数。
    d：list类型，描述迷宫的状态，一个拥有m个元素的list，每个元素是一个包含n个int数字的list；
    迷宫中只有0和1两种数字，0表示通路，1表示墙，确保（0，0）和（m-1，n-1）一定是0。
输出：
    一个数字，如果迷宫能走通，请输出走通所需的最少步数；如果不能走通，请输出-1。
"""
def bfs_visit(
    m:int,
    n:int,
    d:list[list[int]]
) -> int:
    return -1

# 下面的主函数仅供参考，可以随意修改进行本地测试。
# 在线评测时只会加载bfs_visit函数。
if __name__ == '__main__':
    test_num = 10
    while test_num > 0:
        n = random.randint(10, 30)
        m = random.randint(10, 30)
        d = [[0 for j in range(n)] for i in range(m)]
        sampling = [[(i, j) for j in range(n)] for i in range(m)]
        sampling = sampling[1:][:-1]
        k = random.randint(int(len(sampling) * 0.2), int(len(sampling) * 0.8))
        random.shuffle(sampling)
        for i, j in sampling[:k]:
            d[i][j] = 1

        print(m, n, bfs_visit(m, n, d) )

        test_num -= 1
