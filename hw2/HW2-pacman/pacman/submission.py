import random, util
from game import Agent, Directions
import numpy

class OriginalReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current GameState (pacman.py) and the proposed action
    and returns a number, where higher numbers are better.
    """
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    return scoreEvaluationFunction(successorGameState)

#     ********* Reflex agent- sections a and b *********
class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current GameState (pacman.py) and the proposed action
    and returns a number, where higher numbers are better.
    """
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    return betterEvaluationFunction(successorGameState)


#     ********* Evaluation functions *********

def scoreEvaluationFunction(gameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
  """
  return gameState.getScore()

######################################################################################
# b: implementing a better heuristic function
def betterEvaluationFunction(gameState):
  """

  The betterEvaluationFunction takes in a GameState (pacman.py) and should return a number, where higher numbers are better.

  A GameState specifies the full game state, including the food, capsules, agent configurations and more.
  Following are a few of the helper methods that you can use to query a GameState object to gather information about
  the present state of Pac-Man, the ghosts and the maze:

  gameState.getLegalActions():
  gameState.getPacmanState():
  gameState.getGhostStates():
  gameState.getNumAgents():
  gameState.getScore():
  The GameState class is defined in pacman.py and you might want to look into that for other helper methods.
  """
  pos = gameState.getPacmanPosition()

# calculate ghost bonus
  ghost_dists = [util.manhattanDistance(pos, ghost_pos) for ghost_pos in gameState.getGhostPositions()]
  closest_ghost_id = ghost_dists.index(min(ghost_dists)) + 1
  closest_ghost_state = gameState.getGhostState(closest_ghost_id)
  closest_ghost_dist = min(ghost_dists)

  ghost_bonus = ((1/2) ** (closest_ghost_dist - 1))
  if closest_ghost_state.scaredTimer > 0 and closest_ghost_state.scaredTimer < 40:
    ghost_bonus *= -200
  else:
    ghost_bonus *= 500

# calculate food bonus
  food = gameState.getFood()
  food_coords = [(x,y) for x in range(food.width) for y in range(food.height) if food[x][y] == True]
  food_dists = [util.manhattanDistance(pos, food_coord) for food_coord in food_coords]
  if len(food_dists) == 0:
      food_bonus = 0
  else:
    food_bonus = min(food_dists) * 10

  total_bonus = gameState.getScore() - ghost_bonus - food_bonus
  #print("ghost bonus = {0}, food bonus = {1} ({3}), total = {2} ({4})".format(ghost_bonus, food_bonus, total_bonus, None, pos))
  #print("Ghost State = {}, Ghost Distance = {} - {}, Ghost Bonus = {}".format(closest_ghost_state.scaredTimer, pos, closest_ghost_state.getPosition(), ghost_bonus))
  #print(gameState.getNumAgents())

  return total_bonus
#     ********* MultiAgent Search Agents- sections c,d,e,f*********

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & both ExpectimaxAgents.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'betterEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# c: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """
    return self.miniMax(self.index, gameState, self.depth)[1]

  def miniMax(self, agent_id, state, depth):
    if state.isWin() or state.isLose() or depth == 0:
        return self.evaluationFunction(state), Directions.STOP

    legal_moves = state.getLegalActions(agent_id)
    children = [(state.generateSuccessor(agent_id, move), move) for move in legal_moves]
    if len(children) == 0:
      print("not children!")

    next_agent = (agent_id+1) % state.getNumAgents()
    next_depth = depth - (1 if next_agent == self.index else 0)

    if agent_id == self.index:
        cur_max = -numpy.inf
        best = Directions.STOP
        for child, move in children:
          value, _ = self.miniMax(next_agent, child, next_depth)
          if value > cur_max:
            cur_max = value
            best = move
        return cur_max, best

    else:
        cur_min = numpy.inf
        best = Directions.STOP
        for child, move in children:
          value, _ = self.miniMax(next_agent, child, next_depth)
          if value < cur_min:
            cur_min = value
            best = move
        return cur_min, best

######################################################################################
# d: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_CODE
    # Returns value, state pair
    return self.alphaBeta(gameState, self.index, -numpy.inf, numpy.inf, self.depth)[1]
    # END_YOUR_CODE

  def alphaBeta(self, gameState, agent, alpha, beta, depth):
    # Maybe needs some tweaking? None arg should be safe...
    if gameState.isWin() or gameState.isLose() or depth == 0:
      return self.evaluationFunction(gameState), Directions.STOP

    actions = gameState.getLegalActions(agent)
    children = [(gameState.generateSuccessor(agent, action), action) for action in actions]

    next_agent = (agent+1) % gameState.getNumAgents()
    next_depth = depth - (1 if next_agent == self.index else 0)

    if agent == self.index:
      cur_max = -numpy.inf
      best = Directions.STOP
      for child, action in children:
        value, _ = self.alphaBeta(child, next_agent, alpha, beta, next_depth)
        if value > cur_max:
          cur_max = value
          alpha = max(alpha, cur_max)
          best = action
        if cur_max >= beta:
          return numpy.inf, None
      return cur_max, best
    else:
      cur_min = numpy.inf
      best = Directions.STOP
      for child, action in children:
        value, _ = self.alphaBeta(child, next_agent, alpha, beta, next_depth)
        if value < cur_min:
          cur_min = value
          beta = min(beta, cur_min)
          best = action
        if cur_min <= alpha:
          return -numpy.inf, None
      return cur_min, best



######################################################################################
# e: implementing random expectimax

class RandomExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as choosing uniformly at random from their legal moves.
    """

    # BEGIN_YOUR_CODE
    return self.expectimax(gameState, self.index, self.depth)[1]

    # END_YOUR_CODE

  def expectimax(self, gameState, agent, depth):
    if gameState.isWin() or gameState.isLose() or depth == 0:
      return self.evaluationFunction(gameState), Directions.STOP
    
    actions = gameState.getLegalActions(agent)
    children = [(gameState.generateSuccessor(agent, action), action) for action in actions]

    next_agent = (agent+1) % gameState.getNumAgents()
    next_depth = depth - (1 if next_agent == 0 else 0)

    if self.probablistic(agent):
      value = 0
      for child, _ in children:
        value += self.expectimax(child, next_agent, next_depth)[0]
      return value / len(children), Directions.STOP
    if agent == self.index:
      cur_max = -numpy.inf
      best = Directions.STOP
      for child, action in children:
        value, _ = self.expectimax(child, next_agent, next_depth)
        if value > cur_max:
          cur_max = value
          best = action
      return cur_max, best
    # no else because all other agents are probabalistic

  def probablistic(self, agent):
     return self.index != agent

######################################################################################
# f: implementing directional expectimax

class DirectionalExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as using the DirectionalGhost distribution to choose from their legal moves.
    """

    # BEGIN_YOUR_CODE
    return self.d_expectimax(gameState, self.index, self.depth)[1]
    # END_YOUR_CODE

  def d_expectimax(self, gameState, agent, depth):
    if gameState.isWin() or gameState.isLose() or depth == 0:
      return self.evaluationFunction(gameState), Directions.STOP

    next_agent = (agent+1) % gameState.getNumAgents()
    next_depth = depth - (1 if next_agent == 0 else 0)

    if self.probablistic(agent):
      value = 0
      children = self.prob_children(gameState, agent)
      for child, prob in children:
        value += (prob * self.d_expectimax(child, next_agent, next_depth)[0])
        return value, Directions.STOP
    if agent == self.index:
      actions = gameState.getLegalActions(agent)
      children = [(gameState.generateSuccessor(agent, action), action) for action in actions]
      cur_max = -numpy.inf
      best = Directions.STOP
      for child, action in children:
        value, _ = self.d_expectimax(child, next_agent, next_depth)
        if value > cur_max:
          cur_max = value
          best = action
      return cur_max, best
    # no else because all other agents are probabalistic

  def probablistic(self, agent):
     return self.index != agent

  def prob_children(self, gameState, agent):
    from ghostAgents import DirectionalGhost
    ghost = DirectionalGhost(agent)
    dist = ghost.getDistribution(gameState)

    return [(gameState.generateSuccessor(agent, dir), prob) for dir, prob in dist.items()]
    



######################################################################################
# I: implementing competition agent

class CompetitionAgent(MultiAgentSearchAgent):
  """
    Your competition agent
  """

  def getAction(self, gameState):
    """
      Returns the action using self.depth and self.evaluationFunction

    """

    # BEGIN_YOUR_CODE
    raise Exception("Not implemented yet")
    # END_YOUR_CODE



