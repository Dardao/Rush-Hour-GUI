# Rush Hour v0.3.4: minimum-move verification

## What this experiment measures

The starting v0.3.3 checkpoint solved all 2500 original boards. This experiment checks **displayed squares moved, including the automatic red-car exit**, against the 2500 accepted app references. Easy001 uses the user-confirmed 13; the remaining 2499 values use the previously established APK mapping. All board names and geometries were compared with the GUI resource before joining the reference table.

The app reference file is under `evaluation/`. Exploration and training load only `resources/boards.json` (initial geometry) and self-generated experience. No external solution actions, teacher weights, minimum values, BFS, or shortest-path oracle are used by the learner. The reference values are read after evaluation to report equality, excess squares, and efficiency. A reference match is not a new independent exhaustive proof of optimality.

All 2500 boards participate in training. Success here does not establish generalization to unseen boards.

## Training method

This continues the previous archive exploration plus AWR-style self-imitation actor-critic. During exploration, the simulator can restore previously observed states. Random legal actions generate actual transitions; the archive replaces an observed route when a cheaper observed route reaches the same state. Parent paths can recombine observed transitions. There are no unseen-successor sweeps or goal-distance heuristics. Successful self-generated paths are replayed for return-weighted actor updates and value regression.

This is not a claim that plain epsilon-greedy actor-critic alone achieved these results. Archive resets and self-imitation are integral to the experiment. Evaluation uses neither: only one neural legal-action argmax per visited state, with repeat-state termination and a 300-action limit.

## Discount units

The old learner already used gamma = 0.99 once per action. An action can move 1–5 squares, so action-based discounting is not invariant to splitting a slide into shorter slides.

For a transition of d squares, the revised learner uses:

    continuation_discount = gamma ** d
    reward = -0.01 * sum(gamma ** j for j in range(d))
    if solved: reward += gamma ** (d - 1)

The last transition includes the automatic exit distance. A revisit penalty, when applicable, is placed at the end of the transition. Retained successful paths have loops erased, so this penalty does not occur in the successful-path comparisons.

The Monte Carlo recursion is G = reward + continuation_discount * next_G. For any loop-free successful route of D total squares:

    G(D) = -0.01 * (1 - gamma ** D) / (1 - gamma) + gamma ** (D - 1)

This depends only on squares, not action segmentation. At gamma = 1 it becomes 1 - 0.01 * D. For 0 < gamma <= 1, shortening a successful path strictly increases this return. A correct objective alone does not guarantee the optimizer discovers or learns every shortest path.

## Comparisons

1. Baseline v0.3.3: frozen weights, all original starts.
2. Square-only control: 20 more epochs with square-based discount and the same retained experiences; online refresh can retain improvements produced by the policy.
3. Matched action/square conditions: identical starting weights, seed, merged self-experience, 60-epoch schedule, temperature, and optimizer configuration. Only the discount unit differs at initialization; online improvements and subsequent focus sampling may then diverge.
4. Additional exploration and fine-tuning: reported separately, never attributed solely to the discount change.

Checkpoint selection prefers more solved boards, then fewer total squares among successes. It never uses the app reference values. All evaluation histories are retained, including temporary regressions. Comparisons use one training seed and are descriptive, not a statistical claim of one algorithm's superiority.

## Reproduction

Python 3.10 or newer and NumPy are required. C++ exploration requires g++ with C++17. Run commands from this directory. Output folders must not already exist.

    python -m unittest discover -s tests -v
    python tools/evaluate_checkpoint.py --model models/baseline-v033.npz --out evaluation/recheck-baseline
    python tools/run_exploration.py --mode archive-refine --seed 43 --budget 2000000 --refinement-steps 100000 --out experiments/reproduce-explore100k
    python tools/merge_experience.py --inputs resources/baseline-self-experience.json experiments/reproduce-explore100k/experience.json --out resources/reproduce-merged.json
    python tools/train_archive_rl.py --experience resources/reproduce-merged.json --out experiments/reproduce-square60 --epochs 60 --lr .0002 --resume models/baseline-v033.npz --discount-unit square --focus-failures --online-refresh

`best.npz` is selected by solved count, then total moves. `final.npz` is the last epoch and can be worse. Loading the model in the existing GUI uses the expanded RL checkpoint format; this research package does not replace the GUI application.

## Further exploration and stopping

The 100k-refinement training schedule was initially 160 epochs. After a full-solve checkpoint reproduced the shorter experiences, that branch was interrupted and its selected checkpoint was continued using deeper exploration. The same approach was used for the planned 80-epoch 1m-refinement branch. The `stopped.json` files record completed evaluations and the selected checkpoint; the planned epoch count is not reported as completed training.

Two independent full-board archive-refinement runs used seeds 44 and 45, with 1,000,000 sampled steps after the first success on each board. An additional allocation used only two observable quantities: differences in best path costs between those seeds, and the 125 largest visited-state archives. Their union contained 127 boards. Each received 10,000,000 extra refinement steps in an independently seeded simulator worker. `selection.json` contains the exact selection and confirms that app references were not loaded by the allocation tool.

    python tools/refine_uncertain.py --runs experiments/explore-1m-seed44 experiments/explore-1m-seed45 --top-large 125 --steps 10000000 --workers 6 --seed 1042 --out experiments/reproduce-refine-uncertain

The final learner's `--stop-at-experience` condition stops once neural greedy reproduces every retained self-generated path cost. It does not load app minima. Reference comparison is a separate post-hoc evaluation step. The best checkpoint continues to be selected by solved count and total squares, independent of the reference table.
