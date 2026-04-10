import heapq

def h(a, b):
    pos = {'A':(0,0),'B':(1,0),'C':(2,0),'D':(0,1),'E':(1,1),
           'F':(2,1),'G':(0,2),'H':(1,2),'I':(2,2)}
    x1,y1 = pos[a]; x2,y2 = pos[b]
    return ((x1-x2)**2 + (y1-y2)**2)**0.5

def mba_star(start, goal, max_mem):
    graph = {
        'A':[('B',1),('D',1)],
        'B':[('A',1),('C',1),('E',1)],
        'C':[('B',1),('F',1)],
        'D':[('A',1),('E',1),('G',1)],
        'E':[('B',1),('D',1),('F',1),('H',1)],
        'F':[('C',1),('E',1),('I',1)],
        'G':[('D',1),('H',1)],
        'H':[('E',1),('G',1),('I',1)],
        'I':[('F',1),('H',1)]
    }

    pq = [(0,start)]
    cost = {start:0}
    parent = {start:None}
    mem = {start}

    while pq:
        _,cur = heapq.heappop(pq)
        if cur == goal: break

        for nxt,w in graph[cur]:
            new = cost[cur] + w
            if nxt not in cost or new < cost[nxt]:
                cost[nxt] = new
                heapq.heappush(pq,(new+h(nxt,goal),nxt))
                parent[nxt] = cur
                mem.add(nxt)

                if len(mem) > max_mem:
                    worst = max(mem, key=lambda x: cost.get(x,999)+h(x,goal))
                    mem.remove(worst)

    path=[]
    while goal:
        path.append(goal)
        goal = parent[goal]
    return path[::-1]

print(mba_star('A','I',5))