import Pkg
Pkg.add(["POMDPs", "POMDPSimulators", "POMDPPolicies", "POMDPModelTools", "Distributions", "QMDP", "BasicPOMCP", "QuickPOMDPs"])

import QuickPOMDPs: QuickPOMDP
import POMDPs: solve
import POMDPModelTools: ImplicitDistribution, Uniform, Deterministic
import POMDPPolicies: alphavectors
import POMDPSimulators: stepthrough
import Distributions: Normal
import QMDP: QMDPSolver
import BasicPOMCP: POMCPSolver

@enum GamePhase discussion1=1 discussion2=2 voting=3 night=4

# agent is player 1 (julia is 1-indexed)
# mafia_player is a player from 2:5 (inclusive range)
# rest are villagers

struct MafiaState
    mafia_player::Int
    game_phase::GamePhase
    alive_players::Tuple{Bool, Bool, Bool, Bool, Bool} # alive_players[n]: whether player n is alive or not
end

@enum Action begin
    accuse1 = 1
    accuse2 = 2
    accuse3 = 3
    accuse4 = 4
    accuse5 = 5
    claimvillager = 6
    donothing = 7
    vote1 = 8
    vote2 = 9
    vote3 = 10
    vote4 = 11
    vote5 = 12
end

discussion_actions = [accuse1, accuse2, accuse3, accuse4, accuse5, claimvillager, donothing]
vote_actions = [vote1, vote2, vote3, vote4, vote5]

struct MafiaObservation
    # after discussion we observe dialogue, after voting we observe other votes
    player_actions::Tuple{Action, Action, Action, Action, Action}
    # after night we just observe alive players
    alive_players::Tuple{Bool, Bool, Bool, Bool, Bool}
end

pomdp = QuickPOMDP(
    actions = instances(Action), # TODO can we make legal actions a function of state

    obstype = MafiaObservation,

    discount = 0.95,

    transition = function (s, a)
        ImplicitDistribution() do rng
            next_game_phase = mod(Int(s.game_phase), 4) + 1 # increment the game phase
            # TODO alive_players should change depending on the action & game phase
            alive_players = rand(rng, [true, false], (1,5))
            return MafiaState(s.mafia_player, GamePhase(next_game_phase), tuple(alive_players...))
        end
    end,

    observation = function (a, sp)
        ImplicitDistribution() do rng
            # sp.game_phase=discussion2 -> prev state was discussion1
            # sp.game_phase=voting -> prev state was discussion2
            # so in either case observation type is dialogue
            if sp.game_phase == discussion2 || sp.game_phase == voting
                # TODO
                dialogue = rand(rng, discussion_actions, (1, 5))
                return MafiaObservation(tuple(dialogue...), sp.alive_players)
            # sp.game_phase = night -> prev state was voting
            # so we observe votes
            elseif sp.game_phase == night
                # TODO
                votes = rand(rng, vote_actions, (1, 5))
                return MafiaObservation(tuple(votes...), sp.alive_players)
            # sp.game_phase = discussion1 -> prev state was night
            # so we observe nothing
            else 
                return MafiaObservation((donothing, donothing, donothing, donothing, donothing), sp.alive_players)
            end
        end
    end,

    reward = function (s, a, sp)
        if !sp.alive_players[sp.mafia_player] # mafia dead
            return 1.0
        elseif findall(x->x, sp.alive_players) == [sp.mafia_player] # only mafia alive
            return -1.0
        else # penalize actions that are illegal during different phases of the game
            if s.game_phase == discussion1 || s.game_phase == discussion2 # We can accuse someone who is alive, claim villager, or do nothing during discussion
                allowed_actions = [discussion_actions[i] for i in 1:7 if i > 5 || s.alive_players[i]]
            elseif s.game_phase == voting # we can vote for someone who is still alive during voting
                allowed_actions = [vote_actions[i] for i in 1:5 if s.alive_players[i]] 
            else # At night we can only do nothing.
                allowed_actions = [donothing] 
            end
            if a in allowed_actions
                return 0.0
            else
                return -5.0
            end
        end
    end,

    # initial state always has game_phase=discussion1 & all players alive.
    initialstate = ImplicitDistribution(rng -> MafiaState(rand(rng, 2:5), discussion1, (true, true, true, true, true))),

    # we are dead || mafia is dead || only mafia is alive
    isterminal = s -> !s.alive_players[1] || !s.alive_players[s.mafia_player] || (findall(x->x, s.alive_players) == [s.mafia_player])
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