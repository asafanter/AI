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
        TODO: implement this method!
        Use `np.random.choice(...)` whenever you need to randomly choose
         an item from an array of items given a probabilities array `p`.
        You can read the documentation of `np.random.choice(...)` and
         see usage examples by searching it in Google.
        Notice: You might want to pop min(N, len(open) items from the
                `open` priority queue, and then choose an item out
                of these popped items. The other items have to be
                pushed again into that queue.
        """

        best_nodes = self.__getBestNodes()

        best_N_priorities = [node.expanding_priority for node in best_nodes]

        X = np.array(best_N_priorities)
        asaf = np.argmin(X)

        T = 1
        alpha = min(X-100)

        s = sum([(xi / alpha) ** - (1 / T) for xi in X])

        P = 4

    def __getBestNodes(self):

        length = self.open.__len__() if self.open.__len__() < self.N else self.N
        best_nodes = []
        for _ in range(length):
            best_nodes.append(self.open.pop_next_node())

        return best_nodes