# v0.2.10 validation

- Offscreen rendered synthetic 300-epoch histories at window widths 1600 and 2048; inspected the 1600 render for a single-row, unclipped legend.
- All three plot widgets measured 165 px high, with equal 86 px plotting areas at both widths.
- This release changes graph layout only; no new training performance claim.

# v0.2.9 validation

- 17 unit tests passed, including epsilon=0 deterministic argmax, epsilon=1 uniform legal selection, epsilon=.2 empirical probabilities, illegal action exclusion, and epoch schedule endpoints.
- V-trace on-policy reduction and off-policy ratio hand calculations passed.
- GUI worker smoke passed: train exploration, frozen greedy evaluation, identical success metrics, efficiency aggregation and plot layout.
- Validation weights and optimizer state unchanged; no reference solution access.
- Long-run performance improvement not verified.

# v0.2.8 validation

- GUI worker smoke passed: all-visible efficiency denominator, failed episodes zero, reward graph still uses exploration, identical evaluation conditions.
- Checked three equal graph widths sharing the former plot region.
- Known two-problem fixture (one optimal solved, one failed) yields mean 0.5.
- Rendered three success series and efficiency line with distinct values; inspected layout.

# v0.2.7 validation

- 14 unit tests passed.
- GUI worker smoke: visible-page scope; validation waits during exploration; all 100 boards reset for greedy evaluation.
- Checked every evaluation action against frozen-model maximum legal score.
- Evaluation weights, biases, optimizer moments and step counter unchanged.
- Final displayed train/validation successes match graph metrics; exploration reward graph matches the first phase mean.

# v0.2.6 validation

- Maximum reward renders below Total reward; Easy001 +0.87 and Easy002 +0.89 confirmed in 1600x1000 GUI.
- Existing metadata and animation verification passed. Training code unchanged.

# v0.2.5 validation

- 14 unit tests passed: 3-square move -0.03, repeated 3-square move -0.06, terminal travel costs, Easy001 13 squares -> total 0.87.
- GUI worker check passed: live cumulative training rewards average exactly equals the epoch graph; training/validation scope and phases retained.
- Initial, intermediate, terminal reward state and reset paths implemented; validation weights remain unchanged.
- 2,500 Minimum possible values and exit animation checks retained. GUI reward text rendered and inspected at 1600x1000.

# v0.2.4 validation

- All 2,500 metadata values checked: Easy001=13; remaining 2,499 match resource distance sums without shifting or duplicating the last row.
- All loaded Puzzle.solution fields remain empty.
- Easy001 display accounting checked: 9 + 3 + 1 = 13.
- Qt exit animation midpoint, completion, no retrigger on repeated solved updates, and page reset checked.
- 1600x1000 GUI render inspected: Minimum possible fits in right information area; red car disappears on completion.
- 13 existing unit tests and the GUI worker smoke check passed (current page scope, phase separation, graph count agreement).

# v0.2.3 validation

- 13 unit tests passed, including blocked/open exits, immediate success reward and final board, zero-move initial success, and the full 300-move cap.
- Qt offscreen worker test: current-page scope, separate training/validation phases, and graph/display count agreement.
- Legacy reference replay helpers retain explicit-exit semantics for import/test compatibility; RL, validation and inference use Puzzle.is_solved (clear exit path).
- Checkpoint comparison allows float32 serialization rounding (rtol 1e-7).

## Historical validation

# v0.2.2 validation

- 10 unit tests passed.
- GUI page-2 regression passed: validation remains at initial state during training;
  training boards retain final state and path throughout the subsequent validation phase.
- Both reported success counts match the final displayed statuses for their respective groups.
- Validation uses greedy actions with no learning transitions and thus no optimizer updates.
- Graphs record completed epochs only. Training is online stochastic episode success;
  validation is frozen-model greedy success, not an identical evaluation protocol.

# v0.2.1 validation

- Current-page GUI regression passed on page 2 (puzzles 101–200): worker input,
  logged puzzle names, training episode count and validation count match that page only.
  Train/validation labels are present before training and after page refresh.
- The 10 unit tests also passed unchanged.

## Earlier v0.2.0 baseline checks


- 10 unit tests passed: existing core/model tests plus reward values, positive-advantage
  policy update, validation-only episodes causing no update, legal RL trajectories and
  live callbacks, cancellation, and full v2 parameter/Adam checkpoint round-trip.
- RL test puzzles expose a solution property that raises if accessed. Training succeeds
  without reading it. Legacy supervised utilities remain tested separately; GUI RL does not call them.
- Qt offscreen smoke test passed: worker-thread RL training, live 100-board display,
  epoch metrics, charts and rendering. Curves have no symbols or area fill.
- Supplied APK: all 2,500 imported puzzles have empty solution tuples. Topology split:
  2,034 training / 466 validation puzzles. One complete real-data RL epoch passed.
- That single epoch produced mean training episode reward -2.0236; greedy evaluation
  solved 1/100 training and 0/100 validation puzzles. The mixed first-100 exploratory
  display solved 13/100. Different sets and action-selection rules explain these differences.
  These are functional smoke-test results, not evidence of convergence or improved performance.
- Initial boards only are retained by the APK importer; reference moves are not parsed
  into training data. Neither RL training nor greedy evaluation uses reference answers.
- CPU float32 implementation tested in Linux. Native macOS and physical NAND hardware
  have not been tested. No long-run solver success target has been validated.
- Checkpoints include weights and Adam state but not exploration RNG or history;
  exact resumed-run reproducibility is not claimed.
