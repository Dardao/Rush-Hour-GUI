"""Return measured in moved squares, including the automatic final exit."""
from .rl import STEP_REWARD, SUCCESS_REWARD, REPEAT_PENALTY, exit_distance

def square_transition(distance, terminal, gamma=0.99, repeated=False):
    if distance < 1 or not 0 < gamma <= 1:
        raise ValueError('positive distance and 0 < gamma <= 1 required')
    discount = gamma ** distance
    costs = distance if gamma == 1 else (1-discount)/(1-gamma)
    reward = STEP_REWARD * costs
    if terminal: reward += gamma ** (distance-1) * SUCCESS_REWARD
    if repeated: reward += gamma ** (distance-1) * REPEAT_PENALTY
    return reward, discount

def transition(puzzle, nxt, action, seen, gamma=0.99):
    terminal = puzzle.is_solved(nxt)
    distance = int(action)%5+1 + (exit_distance(puzzle,nxt) if terminal else 0)
    return square_transition(distance, terminal, gamma, nxt in seen)

def successful_return(total_squares, gamma=0.99):
    return square_transition(total_squares, True, gamma)[0]
