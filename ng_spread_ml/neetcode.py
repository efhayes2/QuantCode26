from typing import List


class Solution:
    def __init__(self):
        self.grid = None

    def numIslands(self, grid: List[List[str]]) -> int:

        rows = len(grid)
        cols = len(grid[0])
        self.grid = grid
        count = 0

        def dfs(r, c):
            self.grid[r][c] = 0
            if r < 0 or r > rows - 1 or c < 0 or c > cols - 1:
                return


            if grid[r+1][c] == '1':
                    dfs(r + 1, c)
            if grid[r-1][c] == '1':
                dfs(r - 1, c)
            if grid[r][c+1] == '1':
                dfs(r, c + 1)
            if grid[r][c-1] == '1':
                dfs(r, c - 1)

        for i in range(rows):
            for j in range(cols):
                if self.grid[i][j] == '1':
                    count += 1
                    dfs(i, j)

        return count

grid = [
    ["0","1","1","1","0"],
    ["0","1","0","1","0"],
    ["1","1","0","0","0"],
    ["0","0","0","0","0"]
  ]
s = Solution()
print(s.numIslands(grid))
