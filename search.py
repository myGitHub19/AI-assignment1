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
from game import Directions
from typing import List

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


def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    Returns a list of actions that reaches the goal.
    """
    # Stack for frontier
    frontier = util.Stack()
    
    # Each element on the stack: (current_state, path_taken_so_far)
    start_state = problem.getStartState()
    frontier.push((start_state, []))

    # Keep track of visited states
    visited = set()

    while not frontier.isEmpty():
        current_state, path = frontier.pop()

        # If we reach goal, return the path
        if problem.isGoalState(current_state):
            return path

        # If not visited, expand
        if current_state not in visited:
            visited.add(current_state)

            # Get successors
            for successor, action, step_cost in problem.getSuccessors(current_state):
                if successor not in visited:
                    # Append new action to existing path
                    new_path = path + [action]
                    frontier.push((successor, new_path))

    # If no solution found
    return []


def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Queue for frontier
    frontier = util.Queue()

    # Each element in the queue: (current_state, path_taken_so_far)
    start_state = problem.getStartState()
    frontier.push((start_state, []))

    # Keep track of visited states
    visited = set()

    while not frontier.isEmpty():
        current_state, path = frontier.pop()

        # If we reach the goal, return the path
        if problem.isGoalState(current_state):
            return path

        # If not visited, expand
        if current_state not in visited:
            visited.add(current_state)

            # Get successors
            for successor, action, step_cost in problem.getSuccessors(current_state) or []:
                if successor not in visited:
                    # Append new action to existing path
                    new_path = path + [action]
                    frontier.push((successor, new_path))

    # If no solution found
    return []


def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # Priority queue for frontier
    frontier = util.PriorityQueue()

    # Each element in the priority queue: (current_state, path_taken_so_far, total_cost)
    start_state = problem.getStartState()
    frontier.push((start_state, [], 0), 0)

    # Keep track of visited states and their costs
    visited = {}

    while not frontier.isEmpty():
        current_state, path, cost = frontier.pop()

        # If we reach the goal, return the path
        if problem.isGoalState(current_state):
            return path

        # If not visited or we found a cheaper path
        if current_state not in visited or cost < visited[current_state]:
            visited[current_state] = cost

            # Get successors
            for successor, action, step_cost in problem.getSuccessors(current_state) or []:
                new_cost = cost + step_cost
                if successor not in visited or new_cost < visited.get(successor, float('inf')):
                    # Append new action to existing path
                    new_path = path + [action]
                    frontier.push((successor, new_path, new_cost), new_cost)

    # If no solution found
    return []


def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # Priority queue for frontier
    frontier = util.PriorityQueue()

    # Each element in the priority queue: (current_state, path_taken_so_far, total_cost)
    start_state = problem.getStartState()
    frontier.push((start_state, [], 0), 0)

    # Keep track of visited states and their costs
    visited = {}

    while not frontier.isEmpty():
        current_state, path, cost = frontier.pop()

        # If we reach the goal, return the path
        if problem.isGoalState(current_state):
            return path

        # If not visited or we found a cheaper path
        if current_state not in visited or cost < visited[current_state]:
            visited[current_state] = cost

            # Get successors
            for successor, action, step_cost in problem.getSuccessors(current_state) or []:
                new_cost = cost + step_cost
                if successor not in visited or new_cost < visited.get(successor, float('inf')):
                    # Append new action to existing path
                    new_path = path + [action]
                    heuristic_cost = new_cost + heuristic(successor, problem)
                    frontier.push((successor, new_path, new_cost), heuristic_cost)

    # If no solution found
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
