# Define the graph as an adjacency list
graph = {
    '5': ['3', '7'],
    '3': ['2', '4'],
    '7': ['8'],
    '2': [],
    '4': ['8'],
    '8': []
}

# DFS function
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    
    if start not in visited:
        print(start, end=" ")
        visited.add(start)
        
        for neighbour in graph[start]:
            dfs(graph, neighbour, visited)

# Driver code
print("Following is the Depth First Search:")
dfs(graph, '5')