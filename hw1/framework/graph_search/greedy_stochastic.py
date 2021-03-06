from .graph_problem_interface import *
from .best_first_search import BestFirstSearch
from typing import Optional
import numpy as np


class GreedyStochastic(BestFirstSearch):
    def __init__(self, heuristic_function_type: HeuristicFunctionType,
                 T_init: float = 1.0, N: int = 5, T_scale_factor: float = 0.95):
        # GreedyStochastic is a graph search algorithm. Hence, we use close set.
        super(GreedyStochastic, self).__init__(use_close=True)
        self.heuristic_function_type = heuristic_function_type
        self.T = T_init
        self.N = N
        self.T_scale_factor = T_scale_factor
        self.solver_name = 'GreedyStochastic (h={heuristic_name})'.format(
            heuristic_name=heuristic_function_type.heuristic_name)

    def _init_solver(self, problem: GraphProblem):
        super(GreedyStochastic, self)._init_solver(problem)
        self.heuristic_function = self.heuristic_function_type(problem)

    def _open_successor_node(self, problem: GraphProblem, successor_node: SearchNode):
        """
        TODO: implement this method! - done
        """

        if self.close.has_state(successor_node.state):
            already_found_node_with_same_state = self.close.get_node_by_state(successor_node.state)
            if successor_node.expanding_priority < already_found_node_with_same_state.expanding_priority:
                self.close.remove_node(already_found_node_with_same_state)
                self.open.push_node(successor_node)

        elif self.open.has_state(successor_node.state):
            already_found_node_with_same_state = self.open.get_node_by_state(successor_node.state)
            if already_found_node_with_same_state.expanding_priority > successor_node.expanding_priority:
                self.open.extract_node(already_found_node_with_same_state)
                self.open.push_node(successor_node)

        else:
            self.open.push_node(successor_node)

    def _calc_node_expanding_priority(self, search_node: SearchNode) -> float:
        """
        TODO: implement this method! done
        Remember: `GreedyStochastic` is greedy.
        """

        return self.heuristic_function.estimate(search_node.state)

    def _extract_next_search_node_to_expand(self) -> Optional[SearchNode]:
        """
        Extracts the next node to expand from the open queue,
         using the stochastic method to choose out of the N
         best items from open.
        TODO: implement this method! - done
        Use `np.random.choice(...)` whenever you need to randomly choose
         an item from an array of items given a probabilities array `p`.
        You can read the documentation of `np.random.choice(...)` and
         see usage examples by searching it in Google.
        Notice: You might want to pop min(N, len(open) items from the
                `open` priority queue, and then choose an item out
                of these popped items. The other items have to be
                pushed again into that queue.
        """

        if self.open.is_empty():
            return None

        # Get set of items we will randomly choose between
        n_items = min(len(self.open), self.N)
        best_nodes = [self.open.pop_next_node() for _ in range(n_items)]

        # Calculate the probability of using each item
        priorities = np.array([node.expanding_priority for node in best_nodes])
        alpha = min(priorities)
        if alpha != 0.0:
            probabilities = (priorities / alpha) ** (-1.0/self.T)
            probabilities = probabilities / sum(probabilities)

            # Select random node
            node_to_expand = np.random.choice(best_nodes, p=probabilities)
        else:
            # priority = 0 -> node = goal, return immediately
            node_to_expand = best_nodes[np.argmin(priorities)]

        # Clean up function
        for node in best_nodes:
            if node != node_to_expand:
                self.open.push_node(node)
        
        self.close.add_node(node_to_expand)
        self.T *= self.T_scale_factor
        return node_to_expand
