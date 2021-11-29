from quickpomdps import QuickPOMDP
from julia import Pkg

Pkg.add(["POMDPs", "POMDPSimulators", "POMDPPolicies", "POMDPModelTools", "Distributions", "QMDP"])

from julia.POMDPs import solve, pdf
from julia.QMDP import QMDPSolver
from julia.POMDPSimulators import stepthrough
from julia.POMDPPolicies import alphavectors
from julia.POMDPModelTools import Uniform, Deterministic, SparseCat


class MafiaState:
    def __init__(self, game_phase: str, player_dirichlets: tuple, alive_players: tuple):
        self.game_phase = game_phase  # string
        self.player_dirichlets = player_dirichlets  # tuple of ints. length num_villagers+1
        self.alive_players = alive_players  # tuple of booleans. length num_villagers+1

    # state object needs to be hashable to work with julia POMDPs
    def __key(self):
        return tuple(val for _, val in sorted(self.__dict__.items()))

    def __eq__(self, y):
        return isinstance(y, self.__class__) and self.__key() == y.__key()

    def __hash__(self):
        return hash(self.__key())


# number of villagers. so if n=3, player 0 is the mafia, and players 1-4 are the villagers.
# we can designate the agent as player 1 (arbitrarily).
num_villagers = 3

mafia_player = 0

game_phases = ["discussion", "voting"]

# todo: defend & accuse actions should be expanded; one action per player. aka "defend player 1", "defend player 2", ...
actions = ["self mafia", "self villager", "defend player", "accuse player", "do nothing"]

observations = ["observe x", "observe y"]  # TODO

initial_state = MafiaState("discussion", tuple([0] * (num_villagers + 1)), tuple([1] * (num_villagers + 1)))

"""
TODO: figure out how to pass in the state space to solver without having to enumerate through all possible MafiaStates
(cuz there are a lot!!)
* we might be able to do this via the Generative POMDP interface
  * http://juliapomdp.github.io/POMDPs.jl/v0.5/generative/
  
below is just a random list of dummy states to test out that the python/julia stuff works
"""
states = [MafiaState("discussion", tuple([0] * (num_villagers + 1)), tuple([True] * (num_villagers + 1))),
          MafiaState("voting", tuple([1] * (num_villagers + 1)), tuple([0] + [True] * num_villagers)),
          MafiaState("discussion", tuple([2] * (num_villagers + 1)), tuple([False] * (num_villagers + 1)))]


def transition(s: MafiaState, a):
    # TODO
    return Uniform(states)


def observation(s: MafiaState, a, sp: MafiaState):
    # TODO
    return Uniform(observations)


def reward(s: MafiaState, a, sp: MafiaState):
    if not s.alive_players[mafia_player]:
        return 1
    if s.alive_players[mafia_player] and all(not s.alive_players[player] for player in range(1, num_villagers+1)):
        return -1
    return 0


def is_terminal(s: MafiaState):
    # mafia player killed or all villagers killed
    return (not s.alive_players[mafia_player]) or \
           (s.alive_players[mafia_player] and all(not s.alive_players[player] for player in range(1, num_villagers+1)))


m = QuickPOMDP(
    # gen=gen,
    states=states,
    actions=actions,
    discount=0.95,
    isterminal=is_terminal,
    transition=transition,
    observations=observations,
    observation=observation,
    reward=reward,
    initialstate=Uniform(states)
)

solver = QMDPSolver()
policy = solve(solver, m)

print('alpha vectors:')
for v in alphavectors(policy):
    print(v)

print()

rsum = 0.0
for step in stepthrough(m, policy, max_steps=10):
    print('s:', step.s)
    print('a:', step.a)
    print('o:', step.o, '\n')
    rsum += step.r

print('Undiscounted reward was', rsum)
