from collections import deque
from typing import List, Any


class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:

        adj = [[] for _ in range(numCourses)]
        in_degree = [0] * numCourses

        for crs, pre in prerequisites:
            adj[pre].append(crs)
            in_degree[crs] += 1

        queue = deque()

        for i in range(numCourses):
            if in_degree[i] == 0:
                queue.append(i)

        order = []
        completed = 0

        while queue:
            crs = queue.popleft()
            completed += 1
            order.append(crs)

            #c: list[Any]
            for c in adj[crs]:
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)

        return order if completed == numCourses else []

numCourses=3
prerequisites=[[0,1]]

solution = Solution()
print(solution.findOrder(numCourses, prerequisites))