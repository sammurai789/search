# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

class Node:


    def __init__(self, state, parent=None, action=None, path_cost=0):
        "Create a search tree Node, derived from a parent by an action."
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node %s>" % (self.state,)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        "List the nodes reachable in one step from this node."
        return [self.child_node(problem, action)
                for action in problem.getSuccessors(self.state)]

    def child_node(self, problem, action):
        "[Figure 3.10]"
        next = action[0]
        return Node(next, self, action[1], self.path_cost+action[2])

    def solution(self):
        "Return the sequence of actions to go from the root to this node."
        return [node.action for node in self.path()[1:]]

    def path(self):
        "Return a list of nodes forming the path from the root to this node."
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)



def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

     print("Start:", problem.getStartState())
     print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
     print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "* YOUR CODE HERE *"

    node = Node(problem.getStartState())

    if problem.isGoalState(problem.getStartState()):
        return node.solution()
    stack = util.Stack()
    stack.push(node)
    checked = set()
    while not stack.isEmpty():
        node = stack.pop()

        if problem.isGoalState(node.state):
            return node.solution()
        checked.add(node.state)
        for child in node.expand(problem):
            if child.state not in checked :
                stack.push(child)
    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "* YOUR CODE HERE *"

    node = Node(problem.getStartState())

    if problem.isGoalState(problem.getStartState()):
        print(node.state)
        return node.solution()
    queue = util.Queue()
    queue.push(node)
    checked = set()
    while not queue.isEmpty():
        node = queue.pop()
        if problem.isGoalState(node.state):
            # print("check node :", node.state)
            return node.solution()
        checked.add(node.state)
        for child in node.expand(problem) :
            if child.state not in checked  and (child not in queue.list):
                queue.push(child)
    # print(len(queue))
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "* YOUR CODE HERE *"

    node = Node(problem.getStartState())
    if problem.isGoalState(problem.getStartState()): return node.solution()
    pq = util.PriorityQueue()

    pq.update(node, node.path_cost)
    checked = set()
    while not pq.isEmpty():
        node = pq.pop()
        if problem.isGoalState(node.state): return node.solution()
        checked.add(node.state)
        for child in node.expand(problem):
            if (child.state not in checked ) :
                pq.update(child, child.path_cost)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "* YOUR CODE HERE *"
    node = Node(problem.getStartState())
    if problem.isGoalState(problem.getStartState()): return node.solution()
    pq = util.PriorityQueue()
    pq.update(node, node.path_cost + heuristic(node.state, problem))
    explored = set()
    while not pq.isEmpty():
        node = pq.pop()
        if problem.isGoalState(node.state): return node.solution()
        explored.add(node.state)
        for child in node.expand(problem):
            if (child.state not in explored) and (child not in pq.heap):
                pq.update(child, child.path_cost + heuristic(child.state, problem))


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch