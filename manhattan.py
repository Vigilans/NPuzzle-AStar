from AStar import Board, AStar

def manhattan(a: Board, b: Board):
    distance = 0
    for x in range(5):
        for y in range(5):
            x1, y1 = Board.get_pose(a.board[y][x])
            x2, y2 = Board.get_pose(b.board[y][x])
            distance += abs(x2 - x1) + abs(y2 - y1)
    return distance

if __name__ == "__main__":
    astar1 = AStar("manhattan") # Use cpp's built-in manhattan
    astar2 = AStar(manhattan) # Use python version
    start = Board.scrambled(30, True)[-1]
    result1 = astar1.run(start)
    result2 = astar2.run(start)
    print(result1[1:])
    print(result2[1:])
    assert (result1[:-1] == result2[:-1])
