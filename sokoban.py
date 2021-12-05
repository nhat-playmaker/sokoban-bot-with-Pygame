import sys
import collections
import numpy as np
import heapq
import time
import pygame
from itertools import chain


class PriorityQueue:
    """Define a PriorityQueue data structure that will be used"""

    def __init__(self):
        self.Heap = []
        self.Count = 0

    def push(self, item, priority):
        entry = (priority, self.Count, item)
        heapq.heappush(self.Heap, entry)
        self.Count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.Heap)
        return item

    def isEmpty(self):
        return len(self.Heap) == 0

"""Load puzzles and define the rules of sokoban"""

def transferToGameState(layout):
    ### Load the maze from txt file to array ###
    layout = [x.replace('\n', '') for x in layout]
    layout = [','.join(layout[i]) for i in range(len(layout))]
    layout = [x.split(',') for x in layout]
    maxColsNum = max([len(x) for x in layout])

    for row in range(len(layout)):
        for col in range(len(layout[row])):
            # free space
            if layout[row][col] == ' ':
                layout[row][col] = 0
            # wall
            elif layout[row][col] == '#':
                layout[row][col] = 1
            # player
            elif layout[row][col] == '&':
                layout[row][col] = 2
            # box
            elif layout[row][col] == 'B':
                layout[row][col] = 3
            # goal
            elif layout[row][col] == '.':
                layout[row][col] = 4
            # box on goal
            elif layout[row][col] == 'X':
                layout[row][col] = 5
            # player on goal
            elif layout[row][col] == 'W':
                layout[row][col] = 6

        colsNum = len(layout[row])

        if colsNum < maxColsNum:
            layout[row].extend([1 for _ in range(maxColsNum - colsNum)])

    return np.array(layout)


def PosOfPlayer(gameState):
    """Return the position of agent"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 2) | (gameState == 6)))[0]


def PosOfBoxes(gameState):
    """Return the positions of boxes"""
    return tuple(tuple(x) for x in np.argwhere(
        (gameState == 3) | (gameState == 5)))


def PosOfWalls(gameState):
    """Return the positions of walls"""
    return tuple(tuple(x) for x in np.argwhere(gameState == 1))


def PosOfGoals(gameState):
    """Return the positions of goals"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 4) | (gameState == 5) | (gameState == 6)))


def isEndState(posBox, posGoals):
    """Check if all boxes are on the goals (i.e. pass the game)"""
    return sorted(posBox) == sorted(posGoals)

def isLegalAction(action, posPlayer, posBox, posWalls):
    yPlayer, xPlayer = posPlayer
    if action[-1].isupper(): # the move was a push
        # Get the box current position
        y1, x1 = yPlayer + 2 * action[0], xPlayer + 2 * action[1]
    else:
        # Get the player current position
        y1, x1 = yPlayer + action[0], xPlayer + action[1]
    # using set for faster lookup
    posSet = set(chain(posBox, posWalls))
    return (y1, x1) not in posSet

def legalActions(posPlayer, posBox, posWalls):
    allActions = [[-1,0,'u','U'],[1,0,'d','D'],[0,-1,'l','L'],[0,1,'r','R']]
    yPlayer, xPlayer = posPlayer
    legalActions = []
    for action in allActions:
        y1, x1 = yPlayer + action[0], xPlayer + action[1]
        if (y1, x1) in posBox: # the move is a push
            action.pop(2) # drop the little letter
        else:
            action.pop(3) # drop the upper letter
        if isLegalAction(action, posPlayer, posBox, posWalls):
            legalActions.append(action)
        else:
            continue
    return tuple(tuple(x) for x in legalActions)

def updateState(posPlayer, posBox, action):
    yPlayer, xPlayer = posPlayer # the current position of player
    newPosPlayer = [yPlayer + action[0], xPlayer + action[1]] # the position of player after action
    posBox = [list(x) for x in posBox]
    if action[-1].isupper(): # if pushing, update the position of box
        # box current pos is newPos of the player
        posBox.remove(newPosPlayer)
        # move the box 1 block further
        posBox.append([yPlayer + 2 * action[0], xPlayer + 2 * action[1]])
    posBox = tuple(tuple(x) for x in posBox)
    newPosPlayer = tuple(newPosPlayer)
    return newPosPlayer, posBox

def isFailed(posBox, posGoals, posWalls):
    """This function used to observe if the state is potentially failed, then prune the search"""
    rotatePattern = [[0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [2, 5, 8, 1, 4, 7, 0, 3, 6],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8][::-1],
                     [2, 5, 8, 1, 4, 7, 0, 3, 6][::-1]]
    flipPattern = [[2, 1, 0, 5, 4, 3, 8, 7, 6],
                   [0, 3, 6, 1, 4, 7, 2, 5, 8],
                   [2, 1, 0, 5, 4, 3, 8, 7, 6][::-1],
                   [0, 3, 6, 1, 4, 7, 2, 5, 8][::-1]]
    allPattern = rotatePattern + flipPattern

    for box in posBox:
        if box not in posGoals:
            board = [(box[0] - 1, box[1] - 1),
                     (box[0] - 1, box[1]),
                     (box[0] - 1, box[1] + 1),
                     (box[0], box[1] - 1),
                     (box[0], box[1]),
                     (box[0], box[1] + 1),
                     (box[0] + 1, box[1] - 1),
                     (box[0] + 1, box[1]),
                     (box[0] + 1, box[1] + 1)]

            for pattern in allPattern:
                newBoard = [board[i] for i in pattern]

                if newBoard[1] in posWalls and newBoard[5] in posWalls:
                    return True

                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posWalls:
                    return True

                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posBox:
                    return True

                elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox:
                    return True

                elif newBoard[1] in posBox and newBoard[6] in posBox and newBoard[2] in posWalls and newBoard[
                    3] in posWalls and newBoard[8] in posWalls:
                    return True
    return False


"""Implement all approcahes"""

def breadthFirstSearch():
    """Implement breadthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    startState = (beginPlayer, beginBox)  # e.g. ((2, 2), ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5)))
    frontier = collections.deque([[startState]])  # store states
    actions = collections.deque([[0]])  # store actions
    exploredSet = set()
    while frontier:
        node = frontier.popleft()
        node_action = actions.popleft()
        if isEndState(node[-1][-1]):
            print(','.join(node_action[1:]).replace(',', ''))
            break
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            for action in legalActions(node[-1][0], node[-1][1]):
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)
                if isFailed(newPosBox):
                    continue
                frontier.append(node + [(newPosPlayer, newPosBox)])
                actions.append(node_action + [action[-1]])


def depthFirstSearch():
    """Implement depthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    startState = (beginPlayer, beginBox)
    frontier = collections.deque([[startState]])
    exploredSet = set()
    actions = [[0]]
    while frontier:
        node = frontier.pop()
        node_action = actions.pop()
        if isEndState(node[-1][-1]):
            print(','.join(node_action[1:]).replace(',', ''))
            break
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            for action in legalActions(node[-1][0], node[-1][1]):
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)
                if isFailed(newPosBox):
                    continue
                frontier.append(node + [(newPosPlayer, newPosBox)])
                actions.append(node_action + [action[-1]])


def heuristic(posBox, posGoals):
    """A heuristic function to calculate the overall distance between the else boxes and the else goals"""
    distance = 0

    ## Find the box on goal => Heuristic value = 0
    completes = set(posGoals) & set(posBox)

    sortposBox = list(set(posBox).difference(completes))
    sortposGoals = list(set(posGoals).difference(completes))

    # print('B',sortposBox)
    # print('G',sortposGoals)
    #
    # print(len(sortposGoals))
    #
    for i in range(len(sortposBox)):
        distance += (abs(sortposBox[i][0] - sortposGoals[i][0])) + (abs(sortposBox[i][1] - sortposGoals[i][1]))

    # for i in range(len(sortposBox)):
    #     min = (abs(sortposBox[i][0] - sortposGoals[0][0])) + (abs(sortposBox[i][1] - sortposGoals[0][1]))
    #     for j in range(len(sortposGoals)):
    #         if min > (abs(sortposBox[i][0] - sortposGoals[j][0])) + (abs(sortposBox[i][1] - sortposGoals[j][1])):
    #             min = (abs(sortposBox[i][0] - sortposGoals[j][0])) + (abs(sortposBox[i][1] - sortposGoals[j][1]))
    #     distance += min

    return distance


def cost(actions):
    """A cost function"""
    return len([x for x in actions if x.islower()])

def aStarSearch():
    """Implement aStarSearch approach"""
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    print('F:', beginPlayer)

    ## The initial state is the position of player and the list of Box positions
    start_state = (beginPlayer, beginBox)

    ## Save the state
    frontier = PriorityQueue()
    frontier.push([start_state], heuristic(beginBox, posGoals))

    ## Save the action
    actions = PriorityQueue()
    actions.push([0], heuristic(beginBox, posGoals))

    ## Initialize the visited step which we passed
    exploredSet = set()

    while frontier:

        # curr_time = time.time()
        # if curr_time-time_start > 10:
        #     return []

        node = frontier.pop()
        node_action = actions.pop()

        if isEndState(node[-1][-1], posGoals):
            # print(','.join(node_action[1:]).replace(',', ''))
            # break
            return node_action[1:]

        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            Cost = cost(node_action[1:])

            for action in legalActions(node[-1][0], node[-1][1], posWalls):
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)
                if isFailed(newPosBox, posGoals, posWalls):
                    continue
                Heuristic = heuristic(newPosBox, posGoals)
                frontier.push(node + [(newPosPlayer, newPosBox)], Heuristic + Cost)
                actions.push(node_action + [action[-1]], Heuristic + Cost)


"""Read command"""
def readCommand(argv):
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option('-l', '--level', dest='sokobanLevels',
                      help='level of game to play', default='level1.txt')
    parser.add_option('-m', '--method', dest='agentMethod',
                      help='research method', default='bfs')
    args = dict()
    options, _ = parser.parse_args(argv)
    with open('microcomos/' + options.sokobanLevels, "r") as f:
        layout = f.readlines()
    args['layout'] = layout
    args['method'] = options.agentMethod
    return args

'''
    Pygame process 
'''
def p_player(x, y):
    screen.blit(worker, (x * 32, y * 32))

def p_wall(x, y):
    screen.blit(wall, (x * 32, y * 32))

def p_box(x, y):
    screen.blit(box, (x * 32, y * 32))

def p_floor(x, y):
    screen.blit(floor, (x * 32, y * 32))

def p_goal(x, y):
    screen.blit(dock, (x * 32, y * 32))

def p_box_goal(x, y):
    screen.blit(box_docked, (x * 32, y * 32))

def p_worker_goal(x, y):
    screen.blit(worker_dock, (x * 32, y * 32))

def p_maze():
    for r in range(len(maze)):
        for c in range(len(maze[r])):
            if maze[r][c] == '#':
                p_wall(c, r)
            if maze[r][c] == '@':
                p_player(c, r)
            if maze[r][c] == 'B':
                p_box(c, r)
            if maze[r][c] == '&':
                p_player(c, r)
            if maze[r][c] == ' ':
                p_floor(c, r)
            if maze[r][c] == '.':
                p_goal(c, r)
            if maze[r][c] == 'X':
                p_box_goal(c, r)
            if maze[r][c] == 'W':
                p_worker_goal(c, r)

def move(x, y, direction):
    global posPlayer

    _x = x_box = x
    _y = y_box = y

    if direction == 'r' or direction == 'R':
        _y = y + 1
        y_box = _y + 1
    elif direction == 'l' or direction == 'L':
        _y = y - 1
        y_box = _y - 1
    elif direction == 'u' or direction == 'U':
        _x = x - 1
        x_box = _x - 1
    elif direction == 'd' or direction == 'D':
        _x = x + 1
        x_box = _x + 1

    print(_x, _y)

    ## Case 1: empty way
    if maze[_x][_y] == ' ':
        if maze[x][y] == 'W':
            maze[x][y] = '.'
        else:
            maze[x][y] = ' '
        maze[_x][_y] = '&'
        posPlayer = (_x, _y)

    # Case 2: Destination
    elif maze[_x][_y] == '.':
        if maze[x][y] == 'W':
            maze[x][y] = '.'
        else:
            maze[x][y] = ' '
        maze[_x][_y] = 'W'
        posPlayer = (_x, _y)

    # Case 3: Box or Box in destination
    elif maze[_x][_y] == 'B' or maze[_x][_y] == 'X':
        # Can move
        if maze[x_box][y_box] == ' ' or maze[x_box][y_box] == '.':
            if maze[x_box][y_box] == ' ':
                maze[x_box][y_box] = 'B'
            elif maze[x_box][y_box] == '.':
                maze[x_box][y_box] = 'X'

            if maze[_x][_y] == 'B':
                maze[_x][_y] = '&'
            elif maze[_x][_y] == 'X':
                maze[_x][_y] = 'W'

            if maze[x][y] == 'W':
                maze[x][y] = '.'
            else:
                maze[x][y] = ' '
            posPlayer = (_x, _y)

if __name__ == '__main__':
    time_start = time.time()

    ## Initialize global variables
    posPlayer = posWalls = posGoals = ()

    ## Argument command parser
    layout, method = readCommand(sys.argv[1:]).values()

    ## Transfer the 2-d matrix
    gameState = transferToGameState(layout)

    print(gameState)

    posWalls = PosOfWalls(gameState)
    posGoals = PosOfGoals(gameState)
    posPlayer = PosOfPlayer(gameState)

    if method == 'astar':
        res = aStarSearch()
    elif method == 'dfs':
        depthFirstSearch()
    elif method == 'bfs':
        breadthFirstSearch()
    else:
        raise ValueError('Invalid method.')
    time_end = time.time()
    print('Runtime of %s: %.2f second.' % (method, time_end - time_start))

    print(res)

    ## Maze saving
    maze = []
    for i in range(len(layout)):
        t = []
        for j in range(len(layout[i])):
            if layout[i][j] != '\n':
                t.append(layout[i][j])
        maze.append(t)

    print(maze)

    ''' Pygame initialize '''
    pygame.init()

    screen = pygame.display.set_mode((800, 640))

    # Title and Icons
    pygame.display.set_caption('Sokoban Group 3')
    icon = pygame.image.load('images/box.png')
    pygame.display.set_icon(icon)

    # Load image
    box = pygame.image.load('images/box.png')
    box_docked = pygame.image.load('images/box_docked.png')
    dock = pygame.image.load('images/dock.png')
    floor = pygame.image.load('images/floor.png')
    wall = pygame.image.load('images/wall.png')
    worker = pygame.image.load('images/worker.png')
    worker_dock = pygame.image.load('images/worker_dock.png')

    res_idx = 0
    running = True
    while running:
        screen.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

                x = posPlayer[0]
                y = posPlayer[1]

                if event.key == pygame.K_SPACE:
                    if res_idx > len(res):
                        running = False
                        break

                    c = res[res_idx]
                    move(x, y, c)
                    res_idx += 1

        p_maze()

        pygame.display.update()
