from collections import defaultdict


def count_unconnected_components(matrix):
    n = len(matrix)
    graph = defaultdict(list)

    # 构建图
    for a, b in matrix:
        graph[a].append(b)
        graph[b].append(a)

    visited = set()
    components = []

    def dfs(node, component):
        if node in visited:
            return
        visited.add(node)
        component.append(node)
        for neighbor in graph[node]:
            dfs(neighbor, component)

    # 遍历图,分割连通分量
    for node in graph:
        if node not in visited:
            component = []
            dfs(node, component)
            components.append(component)

    return components


# 示例用法
matrix = [[0, 1], [1, 2], [3, 4], [5, 6]]
components = count_unconnected_components(matrix)
print("不相连的子图分割结果:")
for i, component in enumerate(components):
    print(f"子图 {i+1}: {component}")