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
    astar1 = AStar("manhattan")
    astar2 = AStar(manhattan)
    start = Board.scrambled(30, True)[-1]
    print(astar1.run(start)[1:])
    print(astar2.run(start)[1:])
