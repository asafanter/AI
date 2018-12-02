from framework.graph_search import *
from framework.ways import *
from .map_problem import MapProblem
from .deliveries_problem_input import DeliveriesProblemInput
from .relaxed_deliveries_problem import RelaxedDeliveriesState, RelaxedDeliveriesProblem

from typing import Set, FrozenSet, Optional, Iterator, Tuple, Union


class StrictDeliveriesState(RelaxedDeliveriesState):
    """
    An instance of this class represents a state of the strict
     deliveries problem.
    This state is basically similar to the state of the relaxed
     problem. Hence, this class inherits from `RelaxedDeliveriesState`.

    TODO:
        If you believe you need to modify the state for the strict
         problem in some sense, please go ahead and do so.
    """
    pass


class StrictDeliveriesProblem(RelaxedDeliveriesProblem):
    """
    An instance of this class represents a strict deliveries problem.
    """

    name = 'StrictDeliveries'

    def __init__(self, problem_input: DeliveriesProblemInput, roads: Roads,
                 inner_problem_solver: GraphProblemSolver, use_cache: bool = True):
        super(StrictDeliveriesProblem, self).__init__(problem_input)
        self.initial_state = StrictDeliveriesState(
            problem_input.start_point, frozenset(), problem_input.gas_tank_init_fuel)
        self.inner_problem_solver = inner_problem_solver
        self.roads = roads
        self.use_cache = use_cache
        self._init_cache()

    def _init_cache(self):
        self._cache = {}
        self.nr_cache_hits = 0
        self.nr_cache_misses = 0

    def _insert_to_cache(self, key, val):
        if self.use_cache:
            self._cache[key] = val

    def _get_from_cache(self, key):
        if not self.use_cache:
            return None
        if key in self._cache:
            self.nr_cache_hits += 1
        else:
            self.nr_cache_misses += 1
        return self._cache.get(key)

    def expand_state_with_costs(self, state_to_expand: GraphProblemState) -> Iterator[Tuple[GraphProblemState, float]]:
        """
        TODO: implement this method!
        This method represents the `Succ: S -> P(S)` function of the strict deliveries problem.
        The `Succ` function is defined by the problem operators as shown in class.
        The relaxed problem operators are defined in the assignment instructions.
        It receives a state and iterates over the successor states.
        Notice that this is an *Iterator*. Hence it should be implemented using the `yield` keyword.
        For each successor, a pair of the successor state and the operator cost is yielded.
        """
        assert isinstance(state_to_expand, StrictDeliveriesState)

        for junction in self.possible_stop_points - state_to_expand.dropped_so_far:
            if junction == state_to_expand.current_location:
                continue

            # Gets shortest path from here to there without going through any additional gas stations or unvisited delivery points
            dist = self.get_distance(state_to_expand.current_location, junction, state_to_expand.dropped_so_far)

            if dist > state_to_expand.fuel:
                continue

            if junction in self.gas_stations:
                successor = StrictDeliveriesState(junction, state_to_expand.dropped_so_far, self.gas_tank_capacity)

            else:
                successor = StrictDeliveriesState(junction, state_to_expand.dropped_so_far | frozenset([junction]),
                                                   state_to_expand.fuel - dist)

            yield successor, dist

    def is_goal(self, state: GraphProblemState) -> bool:
        """
        This method receives a state and returns whether this state is a goal.
        TODO: implement this method!
        """
        assert isinstance(state, StrictDeliveriesState)

        return state.dropped_so_far == self.drop_points and state.fuel >= 0
    
    def get_distance(self, start: Junction, end: Junction, allowed: FrozenSet[Junction]) -> float:
        dist = self._get_from_cache((start.index, end.index))
        if dist is None:
            # use inner_roads to avoid going through additional stop points on the way
            inner_roads = dict()
            blacklist = self.drop_points - allowed - frozenset([end]) # Not allowed to visit gas stations or unvisited delivery points
            for j in self.roads.junctions():
                links = [l for l in j.links if self.roads[l.target] not in blacklist]
                inner_roads[j.index] = Junction(j.index, j.lat, j.lon, links)

            path = self.inner_problem_solver.solve_problem(MapProblem(Roads(inner_roads), start.index, end.index))
            dist = path.final_search_node.cost
            self._insert_to_cache((start.index, end.index), dist)
        return dist

