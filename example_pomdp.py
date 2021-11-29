from quickpomdps import QuickPOMDP
from julia import Pkg

Pkg.add(["POMDPs", "POMDPSimulators", "POMDPPolicies", "POMDPModelTools", "Distributions", "QMDP"])

from julia.POMDPs import solve, pdf
from julia.QMDP import QMDPSolver
from julia.POMDPSimulators import stepthrough
from julia.POMDPPolicies import alphavectors
from julia.POMDPModelTools import Uniform, Deterministic, SparseCat


class Direction:
    def __init__(self, direction):
        self.direction = direction

    # state object needs to be hashable to work with QuickPOMDP
    def __key(self):
        return tuple(val for _, val in sorted(self.__dict__.items()))

    def __eq__(self, y):
        return isinstance(y, self.__class__) and self.__key() == y.__key()

    def __hash__(self):
        return hash(self.__key())


def transition(s: Direction, a):
    if a == "listen":
        return Deterministic(s)  # tiger stays behind the same door
    else:  # a door is opened
        return Uniform([Direction("left"), Direction("right")])  # reset


def observation(s: Direction, a, sp: Direction):
    if a == "listen":
        if sp == Direction("left"):
            return SparseCat(["left", "right"], [0.85, 0.15])  # sparse categorical distribution
        else:
            return SparseCat(["right", "left"], [0.85, 0.15])
    else:
        return Uniform(["left", "right"])


def reward(s: Direction, a, sp: Direction):
    if a == "listen":
        return -1.0
    elif s.direction == a:  # the tiger was found
        return -100.0
    else:  # the tiger was escaped
        return 10.0


m = QuickPOMDP(
    states=[Direction("left"), Direction("right")],
    actions=["left", "right", "listen"],
    discount=0.95,
    # isterminal=is_terminal,
    # obstype=Float64,
    transition=transition,
    observations=["left", "right"],
    observation=observation,
    reward=reward,
    initialstate=Uniform([Direction("left"), Direction("right")])
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
