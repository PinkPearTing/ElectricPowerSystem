from collections import defaultdict


def count_unconnected_components(matrix):
    n = len(matrix)
    graph = defaultdict(list)

    # 构建图
    for a, b in matrix:
        graph[a].append(b)
        graph[b].append(a)

    visited = set()
    components = 0

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        for neighbor in graph[node]:
            dfs(neighbor)

    # 遍历图，计算连通分量数量
    for node in graph:
        if node not in visited:
            components += 1
            dfs(node)

    return components


# 示例用法
matrix = [[1, 2],
          [2, 3],
          [4, 5]]
num_components = count_unconnected_components(matrix)
print(f"存在 {num_components} 个不相连的子图。")
