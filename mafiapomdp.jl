import Pkg
Pkg.add(["POMDPs", "POMDPSimulators", "POMDPPolicies", "POMDPModelTools", "Distributions", "QMDP", "BasicPOMCP"])

import QuickPOMDPs: QuickPOMDP
import POMDPs: solveit 
import POMDPModelTools: ImplicitDistribution, Uniform, Deterministic
import POMDPPolicies: alphavectors
import POMDPSimulators: stepthrough
import Distributions: Normal
import QMDP: QMDPSolver
import BasicPOMCP: POMCPSolver

@enum GAME_PHASE discussion1=1 discussion2=2 voting=3

# agent is player 1 (julia is 1-indexed)
# mafia_player is a player from 2:4 (inclusive range)
# rest are villagers

struct MafiaState # todo add suspicion/accusation matrix ?
    mafia_player::Int
    game_phase::GAME_PHASE
    alive_players::Tuple{Bool, Bool, Bool, Bool} # alive_players[n]: whether player n is alive or not
end

pomdp = QuickPOMDP(
    actions = [-1, 0, 1], # todo change this
    obstype = Float64, # todo change this
    discount = 0.95,

     # todo use rng instead of rand
    transition = function (s, a)
        ImplicitDistribution() do rng
            next_game_phase = mod(Int(s.game_phase), 3) + 1 # increment the game phase
            # todo: alive_players should change depending on the action
            return MafiaState(s.mafia_player, GAME_PHASE(next_game_phase), map(rand, (Bool, Bool, Bool, Bool)))
        end
    end,

    observation = (a, sp) -> Normal(sp.mafia_player, 0.15), # todo change this

    reward = function (s, a, sp)
        if !sp.alive_players[sp.mafia_player] # mafia dead
            return 1.0
        elseif findall(x->x, sp.alive_players) == [sp.mafia_player] # only mafia alive
            return -1.0
        else
            return 0.0
        end
    end,

    # initial state always has game_phase=discussion1 & all players alive.
    # todo use rng instead of rand()
    initialstate = ImplicitDistribution(rng -> MafiaState(rand(2:4), discussion1, (true, true, true, true))),

    isterminal = s -> !s.alive_players[s.mafia_player] || (findall(x->x, s.alive_players) == [s.mafia_player])
)

solver = POMCPSolver()
planner = solve(solver, pomdp)

# the below (sometimes, not always) fails after the first iteration for some reason
# with error message "ERROR: LoadError: ArgumentError: range must be non-empty"
for (s, a, o) in stepthrough(pomdp, planner, "s,a,o", max_steps=10)
    println("State was $s,")
    println("action $a was taken,")
    println("and observation $o was received.\n")
end