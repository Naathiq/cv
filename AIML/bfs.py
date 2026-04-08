from collections import deque

# Define graph
graph = {
    '5': ['3', '7'],
    '3': ['2', '4'],
    '7': ['8'],
    '2': [],
    '4': ['8'],
    '8': []
}

def bfs(graph, start):
    visited = set()              # to keep track of visited nodes
    queue = deque([start])       # initialize queue

    while queue:
        node = queue.popleft()   # dequeue

        if node not in visited:
            print(node, end=" ")
            visited.add(node)

            # add neighbors to queue
            for neighbour in graph[node]:
                if neighbour not in visited:
                    queue.append(neighbour)

# Driver code
print("Following is the Breadth-First Search:")
bfs(graph, '5')