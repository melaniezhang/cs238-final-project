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
    killed_last_round::Int
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

function generate_votes(s, player_actions, rng)
    a = player_actions[1]
    for i in 2:5
        if s.alive_players[i] || s.killed_last_round == i
            valid_actions_for_i = [vote_actions[j] for j in 1:5 if j != i && (s.alive_players[j] || s.killed_last_round == j)]
            if i == s.mafia_player
                # Mafia strategy: if voted on by the agent, vote randomly, otherwise follow the agent
                if a in valid_actions_for_i
                    player_actions[i] = a
                else
                    player_actions[i] = rand(rng, valid_actions_for_i)
                end
            else
                # Villager strategy: if voted on by the agent, vote randomly, otherwise bandwagon 50% and random 50%
                if a in valid_actions_for_i && rand(rng) < 0.5
                    player_actions[i] = a
                else
                    player_actions[i] = rand(rng, valid_actions_for_i)
                end
            end
        end
    end
    votes_for_each_player = [0, 0, 0, 0, 0]
    for i in 1:5
        i_vote = Integer(player_actions[i]) - 7
        if i_vote > 0
            votes_for_each_player[i_vote] += 1
        end
    end
    return votes_for_each_player
end

pomdp = QuickPOMDP(
    actions = instances(Action),

    obstype = MafiaObservation,

    discount = 0.95,

    transition = function (s, a)
        ImplicitDistribution() do rng
            next_game_phase = mod(Int(s.game_phase), 4) + 1 # increment the game phase
            alive_players = collect(s.alive_players)
            killed = 0
            if s.game_phase == voting
                votes_for_each_player = generate_votes(s, [a, donothing, donothing, donothing, donothing], rng)
                voted_out = argmax(votes_for_each_player)
                most_votes = votes_for_each_player[voted_out]
                if count(i->i==most_votes, votes_for_each_player) == 1
                    alive_players[voted_out] = false
                    killed = voted_out
                else # tie -> nobody dies
                end
            elseif s.game_phase == night
                # mafia chooses someone to kill at random
                can_kill = [p for p in 1:5 if (p != s.mafia_player) && s.alive_players[p]]
                killed = rand(rng, can_kill)
                alive_players[killed] = false
            end
            return MafiaState(s.mafia_player, GamePhase(next_game_phase), tuple(alive_players...), killed)
        end
    end,

    observation = function (a, sp)
        ImplicitDistribution() do rng
            # println(a, ' ', sp.alive_players, ' ', sp.game_phase, ' ', sp.killed_last_round)
            player_actions = [donothing, donothing, donothing, donothing, donothing]
            # sp.game_phase=discussion2 -> prev state was discussion1
            # sp.game_phase=voting -> prev state was discussion2
            # so in either case observation type is dialogue
            if sp.game_phase == discussion2 || sp.game_phase == voting
                random_dialogue = rand(rng, discussion_actions, 4)
                player_actions[1] = a
                for i in 2:5
                    if sp.alive_players[i]
                        valid_actions_for_i = [discussion_actions[j] for j in 1:7 if (j > 5) || (j != i && sp.alive_players[j])]
                        if i == sp.mafia_player
                            # Mafia strategy: if accused by the agent, accuse back, otherwise act randomly
                            if Integer(a) == i
                                player_actions[i] = accuse1
                            else
                                player_actions[i] = rand(rng, valid_actions_for_i)
                            end
                        else
                            # Villager strategy: if accused by the agent, defend themselves, otherwise bandwagon 50% and random 50%
                            if Integer(a) == i
                                player_actions[i] = claimvillager
                            else
                                if a in valid_actions_for_i && rand(rng) < 0.5
                                    player_actions[i] = a
                                else
                                    player_actions[i] = rand(rng, valid_actions_for_i)
                                end
                            end
                        end
                    end
                end
            # sp.game_phase = night -> prev state was voting
            # so we observe votes
            elseif sp.game_phase == night
                player_actions[1] = a
                
                consistent = false
                tries = 0
                while !consistent
                    votes_for_each_player = generate_votes(sp, player_actions, rng)
                    voted_out = argmax(votes_for_each_player)
                    most_votes = votes_for_each_player[voted_out]
                    if count(i->i==most_votes, votes_for_each_player) != 1
                        voted_out = 0
                    end
                    if voted_out == sp.killed_last_round
                        consistent = true
                    else
                        # println("Expecting to kill ", sp.killed_last_round, " but got votes ", player_actions)
                        tries += 1
                        if tries > 500
                            println("ERROR: Impossible to generate votes consistent with currently living players")
                            break
                        end
                    end
                end
            # sp.game_phase = discussion1 -> prev state was night
            # so we observe nothing
            else
            end
            # println(player_actions, '\n')
            return MafiaObservation(tuple(player_actions...), sp.alive_players)
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
    initialstate = ImplicitDistribution(rng -> MafiaState(rand(rng, 2:5), discussion1, (true, true, true, true, true), 0)),

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