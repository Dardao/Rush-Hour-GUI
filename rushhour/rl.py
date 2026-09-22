"""Epsilon-greedy behavior with clipped V-trace actor-critic updates."""
import numpy as np
from .model import policy, rollout
from .core import MAX_MOVES, permute_vehicles

GAMMA = 0.99
LIMIT = MAX_MOVES
STEP_REWARD = -0.01
REPEAT_PENALTY = -0.03
SUCCESS_REWARD = 1.0


def epsilon_at(epoch, epochs):
    decay_epochs = max(2, int(np.ceil(epochs * 0.8)))
    return float(1.0 + (0.05 - 1.0) * min(1.0, max(0, epoch - 1) / (decay_epochs - 1)))


def epsilon_action(scores, mask, rng, epsilon):
    legal = np.flatnonzero(mask)
    best = int(legal[np.argmax(scores[legal])])
    action = int(rng.choice(legal)) if epsilon > 0 and rng.random() < epsilon else best
    mu = epsilon / len(legal) + ((1 - epsilon) if action == best else 0)
    return action, mu


def vtrace_targets(transitions, bootstrap):
    # Each tuple: actor id, x, mask, action, r, done, V(s), pi(a|s), mu(a|s).
    next_values = bootstrap.copy()
    corrected_next = bootstrap.copy()
    targets, advantages = [], []
    for i, _, _, _, r, done, value, pi, mu in reversed(transitions):
        discount = 0.0 if done else GAMMA
        rho = min(1.0, pi / mu)
        advantage = rho * (r + discount * corrected_next[i] - value)
        target = value + rho * (r + discount * next_values[i] - value) + discount * rho * (corrected_next[i] - next_values[i])
        targets.append(target); advantages.append(advantage)
        next_values[i] = value
        corrected_next[i] = target
    return np.asarray(targets[::-1], np.float32), np.asarray(advantages[::-1], np.float32)


def reward(nxt, seen, solved=False, distance=1, exit_distance=0):
    success = solved or nxt is None
    return (SUCCESS_REWARD if success else 0) + STEP_REWARD * (distance + exit_distance) + (REPEAT_PENALTY if nxt is not None and nxt in seen else 0)


def exit_distance(puzzle, state):
    return max(0, 6 - state[0] - puzzle.cars[0].length) + 1 if state is not None else 0


def transition_reward(puzzle, nxt, action, seen):
    solved = puzzle.is_solved(nxt)
    return reward(nxt, seen, solved, int(action) % 5 + 1,
                  exit_distance(puzzle, nxt) if solved else 0)


def replay_rewards(puzzle, path):
    state = puzzle.start; seen = {state}
    last = SUCCESS_REWARD + STEP_REWARD * exit_distance(puzzle, state) if puzzle.is_solved(state) else None
    total = last if last is not None else 0.0
    for action in path:
        nxt = puzzle.move(state, action)
        last = transition_reward(puzzle, nxt, action, seen); total += last
        if puzzle.is_solved(nxt): break
        state = nxt; seen.add(state)
    return last, total


def replay_successes(net, records, rng, max_records=30, batch_size=256, stopped=lambda: False):
    """Auxiliary self-imitation: positive observed-return advantage only.

    This is deliberately separate from the fresh V-trace objective. Successful
    replay is selection-biased, so it is not treated as an on-policy update.
    """
    if not records:
        return [], 0, 0
    selected = rng.choice(len(records), size=min(max_records, len(records)), replace=False)
    rows, used = [], 0
    for index in selected:
        if stopped(): break
        original, original_path = records[int(index)]
        if any(not 0 <= int(a) < len(original.cars)*10 for a in original_path): continue
        puzzle = permute_vehicles(original, rng)
        slots = {c.label: i for i, c in enumerate(puzzle.cars)}
        path = [slots[original.cars[int(a)//10].label]*10+int(a)%10 for a in original_path]
        state, seen, episode, solved = puzzle.start, {puzzle.start}, [], False
        for action in path:
            if action not in puzzle.actions(state):
                break
            mask = np.zeros(140, bool); mask[puzzle.actions(state)] = True
            nxt = puzzle.move(state, action)
            r = transition_reward(puzzle, nxt, action, seen)
            episode.append((puzzle.encode(state), mask, action, r))
            solved = puzzle.is_solved(nxt)
            if solved: break
            state = nxt; seen.add(state)
        if not solved or not episode: continue
        used += 1
        discounted, with_returns = 0.0, []
        for x, mask, action, r in reversed(episode):
            discounted = r + GAMMA * discounted
            with_returns.append((x, mask, action, discounted))
        rows.extend(reversed(with_returns))
    rng.shuffle(rows)
    losses = []
    for start in range(0, len(rows), batch_size):
        if stopped(): break
        batch = rows[start:start+batch_size]
        x = np.stack([r[0] for r in batch])
        returns = np.asarray([r[3] for r in batch], np.float32)
        values = net.forward(x)[:, 140]
        positive = np.maximum(returns-values, 0)
        if not positive.any(): continue
        losses.append(net.actor_critic(x, np.asarray([r[2] for r in batch]),
                                       np.maximum(returns, values), np.stack([r[1] for r in batch]),
                                       lr=0.0001, entropy_weight=0, policy_advantages=positive))
    return losses, used, len(rows)


def episodes(net, puzzles, learning, rng, stopped=lambda: False, live=None, limit=LIMIT, greedy=False, epsilon=1.0):
    """Validation actors may be displayed, but never contribute gradients."""
    if not 0 <= epsilon <= 1 or (any(learning) and epsilon == 0):
        raise ValueError('Training epsilon must be > 0 and <= 1')
    if greedy and any(learning):
        raise ValueError('Greedy evaluation cannot update weights')
    states = [p.start for p in puzzles]
    seen = [{s} for s in states]
    paths = [[] for _ in puzzles]
    status = ['solved' if p.is_solved(s) else 'running' for p, s in zip(puzzles, states)]
    totals = np.zeros(len(puzzles))
    last_rewards = [None] * len(puzzles)
    for i, p in enumerate(puzzles):
        if status[i] == "solved":
            last_rewards[i], totals[i] = replay_rewards(p, [])
    losses = []
    if stopped(): return None
    if live:
        live({'states': list(states), 'paths': [list(p) for p in paths], 'statuses': list(status), 'step': 0, 'last_rewards': list(last_rewards), 'total_rewards': totals.tolist()})
    for chunk in range(0, limit, 16):
        transitions = []
        for step in range(chunk, min(chunk+16, limit)):
            if stopped(): return None
            active = [i for i, s in enumerate(status) if s == 'running']
            if not active: break
            x = np.stack([puzzles[i].encode(states[i]) for i in active])
            masks = np.zeros((len(active), 140), bool)
            for row, i in enumerate(active): masks[row, puzzles[i].actions(states[i])] = True
            predictions = net.forward(x)
            probs = policy(predictions, masks)
            for row, i in enumerate(active):
                if not masks[row].any():
                    status[i] = 'blocked'; continue
                action, mu = epsilon_action(predictions[row, :140], masks[row], rng, 0.0 if greedy else epsilon)
                nxt = puzzles[i].move(states[i], action)
                solved = puzzles[i].is_solved(nxt)
                r = transition_reward(puzzles[i], nxt, action, seen[i]); totals[i] += r
                last_rewards[i] = r
                paths[i].append(action)
                done = solved or step+1 == limit
                if learning[i]: transitions.append((i, x[row], masks[row], action, r, done, float(predictions[row, 140]), float(probs[row, action]), mu))
                if solved:
                    if nxt is not None: states[i] = nxt
                    status[i] = 'solved'
                else:
                    repeated = nxt in seen[i]
                    states[i] = nxt; seen[i].add(nxt)
                    if greedy and repeated: status[i] = 'loop'
                    elif done: status[i] = 'limit'
            if live:
                live({'states': list(states), 'paths': [list(p) for p in paths], 'statuses': list(status), 'step': step+1, 'last_rewards': list(last_rewards), 'total_rewards': totals.tolist()})
        if transitions:
            bootstrap = np.zeros(len(puzzles))
            active = [i for i, s in enumerate(status) if learning[i] and s == 'running']
            if active:
                bootstrap[active] = net.forward(np.stack([puzzles[i].encode(states[i]) for i in active]))[:, 140]
            targets, advantages = vtrace_targets(transitions, bootstrap)
            loss = net.actor_critic(np.stack([t[1] for t in transitions]),
                                   np.array([t[3] for t in transitions]), targets,
                                   np.stack([t[2] for t in transitions]), policy_advantages=advantages)
            losses.append(loss)
        if all(s != 'running' for s in status): break
    return paths, status, totals, losses


def evaluate(net, puzzles, stopped=lambda: False):
    paths, status = rollout(net, puzzles, stopped=stopped)
    totals = [replay_rewards(p, path)[1] for p, path in zip(puzzles, paths)]
    return status, float(np.mean(totals)) if totals else 0.0
