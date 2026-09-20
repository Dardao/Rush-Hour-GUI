import json
from pathlib import Path
import time
import numpy as np
from PySide6.QtCore import Qt, QRectF, QPointF, QTimer, QThread, Signal
from PySide6.QtGui import QColor, QPainter, QFont, QImage, QPen, QPainterPath
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSpinBox, QFileDialog, QMessageBox, QScrollArea, QDialog, QCheckBox)
from .core import load_apk, split, MAX_MOVES, display_moves
from .model import Network, rollout
from .rl import episodes, evaluate, replay_rewards, SUCCESS_REWARD, STEP_REWARD, epsilon_at

# Stable original vehicle identities; flat body colors matched to the supplied
# Classic app screenshots (the APK's older sprite sheet uses different shades).
VEHICLE_COLORS = {
    'x': '#e53935',  # red target
    'a': '#9cdb19', 'b': '#ff941c', 'c': '#13b9ed',
    'd': '#ed3887', 'e': '#a51ce0', 'f': '#7de3d2',
    'g': '#237ea3', 'h': '#edcf8b', 'i': '#e3e923',
    'j': '#a85418', 'k': '#18a322',
    'o': '#ffcf37', 'p': '#d94cda', 'q': '#493be0', 'r': '#43b6a8',
}
STATUS = {'ready': '#40536b', 'running': '#56bafa', 'solved': '#67dca4',
          'loop': '#f8b65a', 'limit': '#ee6c7a', 'blocked': '#ee6c7a'}


def paint_board(painter, p, state, rect, exit_progress=0):
    painter.fillRect(rect, QColor('#202d40'))
    cell = rect.width() / 6
    painter.setPen(QColor('#334259'))
    for k in range(7):
        painter.drawLine(int(rect.x()+k*cell), int(rect.y()), int(rect.x()+k*cell), int(rect.bottom()))
        painter.drawLine(int(rect.x()), int(rect.y()+k*cell), int(rect.right()), int(rect.y()+k*cell))
    painter.save()
    painter.setClipRect(rect)
    for i, (car, pos) in enumerate(zip(p.cars, state)):
        if i == 0 and exit_progress:
            pos += (6 - pos) * exit_progress
        x, y = (pos, car.lane) if car.horizontal else (car.lane, pos)
        w, h = (car.length, 1) if car.horizontal else (1, car.length)
        box = QRectF(rect.x()+x*cell+1.5, rect.y()+y*cell+1.5, w*cell-3, h*cell-3)
        painter.setPen(Qt.PenStyle.NoPen)
        color = QColor(VEHICLE_COLORS.get(car.label.lower(), '#9aa4b2'))
        painter.setBrush(color)
        painter.drawRoundedRect(box, 3, 3)
        if cell >= 15:
            luminance = 0.2126 * color.redF() + 0.7152 * color.greenF() + 0.0722 * color.blueF()
            painter.setPen(QColor('#142031' if luminance > 0.52 else '#ffffff'))
            painter.drawText(box, Qt.AlignmentFlag.AlignCenter, car.label.upper())
    painter.restore()
    painter.setPen(QColor('#ff626e'))
    painter.drawText(QRectF(rect.right()-2, rect.y()+2*cell, 12, cell), Qt.AlignmentFlag.AlignCenter, '›')


class Boards(QWidget):
    clicked = Signal(int)

    def __init__(self):
        super().__init__()
        self.puzzles, self.states, self.paths, self.statuses, self.final = [], [], [], [], []
        self.step = 0
        self.learning = None
        self.last_rewards, self.total_rewards = [], []
        self.fit = True
        self.exit_started = {}
        self.exit_timer = QTimer(self)
        self.exit_timer.setInterval(30)
        self.exit_timer.timeout.connect(self.animate_exits)
        self.setToolTip("moves: 이동 칸 수 + 해결 시 출구까지 거리와 통과 1칸. Minimum possible: Easy001=13, 나머지 APK 값.")
        self.setMinimumWidth(820)
        self.setMinimumHeight(480)

    def set_puzzles(self, puzzles):
        self.exit_started.clear()
        self.exit_timer.stop()
        train_ids = {id(p) for p in split(puzzles)[0]}
        self.learning = [id(p) in train_ids for p in puzzles]
        self.puzzles = puzzles
        self.states = [p.start for p in puzzles]
        self.last_rewards = [None] * len(puzzles)
        self.total_rewards = [0.0] * len(puzzles)
        self.paths = [[] for p in puzzles]
        self.statuses = ['ready'] * len(puzzles)
        self.final = self.statuses.copy()
        self.step = 0
        self.update()

    def set_paths(self, paths, final):
        self.exit_started.clear()
        self.exit_timer.stop()
        self.states = [p.start for p in self.puzzles]
        self.last_rewards = [None] * len(self.puzzles)
        self.total_rewards = [0.0] * len(self.puzzles)
        self.paths, self.final = paths, final
        self.statuses = ['running'] * len(self.puzzles)
        self.step = 0
        self.update()

    def advance(self):
        changed = False
        for i, p in enumerate(self.puzzles):
            path = self.paths[i]
            if self.step < len(path) and self.statuses[i] == 'running':
                nxt = p.move(self.states[i], path[self.step])
                if nxt is not None:
                    self.states[i] = nxt
                changed = True
            if self.step + 1 >= len(path) and self.statuses[i] == 'running':
                self.statuses[i] = self.final[i]
        self.step += 1
        for i, p in enumerate(self.puzzles):
            self.last_rewards[i], self.total_rewards[i] = replay_rewards(p, self.paths[i][:self.step])
        self.sync_exits()
        self.update()
        return changed

    def sync_exits(self):
        now = time.monotonic()
        for i, status in enumerate(self.statuses):
            if status == 'solved':
                self.exit_started.setdefault(i, now)
            else:
                self.exit_started.pop(i, None)
        if any(now - start < 0.55 for start in self.exit_started.values()):
            self.exit_timer.start()

    def exit_progress(self, i):
        start = self.exit_started.get(i)
        return min(1.0, (time.monotonic() - start) / 0.55) if start is not None else 0.0

    def animate_exits(self):
        self.update()
        if all(self.exit_progress(i) >= 1 for i in self.exit_started):
            self.exit_timer.stop()

    def resizeEvent(self, event):
        if not self.fit:
            self.setMinimumHeight(int(self.width()/10 + 95)*10)
        super().resizeEvent(event)

    def set_fit(self, fit):
        self.fit = fit
        self.setMinimumHeight(480 if fit else int(self.width()/10 + 95)*10)
        self.update()

    def mousePressEvent(self, event):
        pitch = self.width()/10
        row_height = self.height()/10 if self.fit else pitch+95
        i = int(event.position().y()//row_height)*10 + int(event.position().x()//pitch)
        if i < len(self.puzzles):
            self.clicked.emit(i)

    def paintEvent(self, event):
        q = QPainter(self)
        q.setRenderHint(QPainter.RenderHint.Antialiasing)
        q.setFont(QFont('Arial', 8))
        pitch = self.width()/10
        row_height = self.height()/10 if self.fit else pitch+95
        for i in range(100):
            x, y = (i%10)*pitch, (i//10)*row_height
            frame = QRectF(x+3, y+3, pitch-6, row_height-6)
            q.setPen(QColor(STATUS[self.statuses[i]] if i < len(self.puzzles) else '#263447'))
            q.setBrush(QColor('#172234'))
            q.drawRoundedRect(frame, 6, 6)
            if i >= len(self.puzzles):
                continue
            p = self.puzzles[i]
            q.setPen(QColor('#b8c7da'))
            side = min(pitch*0.52, row_height-10) if self.fit else pitch-18
            if self.fit:
                paint_board(q, p, self.states[i], QRectF(x+7, y+5, side, side), self.exit_progress(i))
            else:
                q.drawText(QRectF(x+7, y+5, pitch-14, 16), Qt.AlignmentFlag.AlignLeft, p.name)
                paint_board(q, p, self.states[i], QRectF(x+9, y+24, side, side), self.exit_progress(i))
            used = display_moves(p, self.paths[i][:self.step], self.statuses[i] == 'solved', self.states[i])
            last = self.last_rewards[i] if i < len(self.last_rewards) else None
            total = self.total_rewards[i] if i < len(self.total_rewards) else 0.0
            reward_info = f'Reward: {last:+.2f}' if last is not None else 'Reward: —'
            reward_info += f'\nTotal reward: {total:+.2f}'
            maximum = f'{SUCCESS_REWARD + STEP_REWARD * p.minimum_possible:+.2f}' if p.minimum_possible is not None else '—'
            reward_info += f'\nMaximum reward: {maximum}'
            minimum = f'Minimum possible: {p.minimum_possible if p.minimum_possible is not None else "—"}'
            q.setPen(QColor('#aebed2' if self.statuses[i] == 'ready' else STATUS[self.statuses[i]]))
            if self.fit:
                info = (f'{p.name} '+(('[학습]' if self.learning[i] else '[검증]') if self.learning is not None else '')
                        + f'\n{used} moves · {self.statuses[i]}\n{minimum}\n{reward_info}')
                box = QRectF(x+side+12, y+5, pitch-side-16, row_height-10)
                flags = Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap
                font = QFont('Arial', 8)
                for size in (8, 7.5, 7, 6.5, 6, 5.5, 5, 4.5, 4):
                    font.setPointSizeF(size); q.setFont(font)
                    if q.boundingRect(box, flags, info).height() <= box.height(): break
                q.drawText(box, flags, info)
                q.setFont(QFont('Arial', 8))
            else:
                q.drawText(QRectF(x+7, y+pitch+7, pitch-14, 80), Qt.AlignmentFlag.AlignLeft,
                           f'{used} moves · {self.statuses[i]}\n{minimum}\n{reward_info}')



class Heatmaps(QWidget):
    def __init__(self, net):
        super().__init__()
        self.setMinimumHeight(165)
        self.setMaximumHeight(190)
        self.images = []
        self.set_network(net)

    def set_network(self, net):
        self.images = []
        for w in [net.w[0], net.w[1], net.w[2][:, :140], net.w[2][:, 140:]]:
            scale = max(float(np.abs(w).max()), 1e-8)
            z = w / scale
            rgb = np.zeros((*w.shape, 3), np.uint8)
            rgb[:, :, 0] = (35 + 210*np.maximum(z, 0)).astype(np.uint8)
            rgb[:, :, 1] = (40 + 95*(1-abs(z))).astype(np.uint8)
            rgb[:, :, 2] = (55 + 195*np.maximum(-z, 0)).astype(np.uint8)
            self.images.append((QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy(), scale))
        self.update()

    def paintEvent(self, event):
        q = QPainter(self)
        q.setFont(QFont('Arial', 10))
        names = ['INPUT → H1   224 × 128', 'H1 → H2   128 × 64', 'POLICY   64 × 140', 'VALUE   64 × 1']
        width = self.width()/4
        for i, ((im, scale), name) in enumerate(zip(self.images, names)):
            x = i*width
            q.setPen(QColor('#d7e4f4'))
            q.drawText(QRectF(x+8, 0, width-16, 23), Qt.AlignmentFlag.AlignLeft, name)
            q.drawImage(QRectF(x+8, 26, width-16, 112), im)
            q.setPen(QColor('#8fa6c0'))
            q.drawText(QRectF(x+8, 141, width-16, 22), Qt.AlignmentFlag.AlignLeft, f'blue − / red +   ±{scale:.3f}')


class AccuracyPlot(QWidget):
    def __init__(self, success=False, efficiency=False):
        super().__init__()
        self.success = success
        self.efficiency = efficiency
        self.history = []
        self.setMinimumSize(210, 165)
        self.setMaximumHeight(190)
        self.setToolTip('정답 없는 강화학습: 학습 중 탐험 episode의 평균 누적 보상.\n검증 보상과는 종료 조건이 달라 직접 비교하지 않습니다.')
        if efficiency:
            self.setToolTip('현재 페이지 전체 greedy 평가 평균. 성공: Minimum possible / moves, 실패: 0.\n비율이 1을 초과하면 그대로 표시하고 로그에 별도로 기록합니다. 기준값 누락 시 평균은 표시하지 않습니다.')
        if success:
            self.setToolTip('Train explore: 학습 중 확률적 탐험 성공률.\nTrain greedy / Val greedy: 학습 종료 후 동일한 고정 모델·종료 조건으로 평가한 성공률.')

    def clear(self):
        self.history.clear()
        self.update()

    def append(self, metrics):
        if self.success:
            explore = metrics['exploration_solved'] / metrics['exploration_evaluated'] if metrics['exploration_evaluated'] else None
            self.history.append((metrics['epoch'], explore, metrics['train_success_rate'], metrics['validation_success_rate']))
        else:
            key = 'efficiency_mean' if self.efficiency else 'episode_reward'
            self.history.append((metrics['epoch'], metrics[key]))
        self.update()

    def paintEvent(self, event):
        q = QPainter(self)
        q.setRenderHint(QPainter.RenderHint.Antialiasing)
        q.setFont(QFont('Arial', 9))
        q.setPen(QColor('#d7e4f4'))
        title = 'EFFICIENCY · all evaluated' if self.efficiency else ('SOLVE SUCCESS (%)' if self.success else 'RL EPISODE REWARD')
        q.drawText(QRectF(8, 0, self.width()-16, 22), Qt.AlignmentFlag.AlignLeft, title)
        legends = [('Train explore', '#67dca4'), ('Train greedy', '#62b5ff'), ('Val greedy', '#f8b65a')] if self.success else [('All greedy' if self.efficiency else 'Train explore', '#67dca4')]
        q.setFont(QFont('Arial', 8))
        labels = []
        for col, (name, color) in enumerate(legends):
            v = self.history[-1][col+1] if self.history else None
            value = (f' {v:.1%}' if self.success else f' {v:.3f}') if v is not None else ' —'
            labels.append((name+value, color))
        # Keep every legend on one row without taking height from the plot.
        gap = 10
        left = 8 if self.success else 43
        available = self.width()-left-8
        natural = sum(q.fontMetrics().horizontalAdvance(label) for label, _ in labels) + gap*(len(labels)-1)
        q.save()
        q.translate(left, 21)
        q.scale(min(1.0, available/max(1, natural)), 1.0)
        x = 0
        for label, color in labels:
            width = q.fontMetrics().horizontalAdvance(label)
            q.setPen(QColor(color))
            q.drawText(QRectF(x, 0, width+1, 16), Qt.AlignmentFlag.AlignLeft, label)
            x += width+gap
        q.restore()
        area = QRectF(43, 48, self.width()-58, self.height()-79)
        low, high = (0, 1) if self.success or self.efficiency else (-4, 1)
        if self.efficiency:
            high = max(1.0, max((row[1] for row in self.history if row[1] is not None), default=1.0))
        for value in (low, (low+high)/2, high):
            y = area.bottom()-area.height()*(value-low)/(high-low)
            q.setPen(QColor('#334259'))
            q.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
            q.setPen(QColor('#8fa6c0'))
            q.drawText(QRectF(0, y-8, 35, 16), Qt.AlignmentFlag.AlignRight, f'{value*100:.0f}' if self.success else f'{value:g}')
        last = self.history[-1][0] if self.history else 1
        xmax = max(2, last)
        q.drawText(QRectF(area.left(), area.bottom()+3, 35, 17), Qt.AlignmentFlag.AlignLeft, '1')
        q.drawText(QRectF(area.right()-35, area.bottom()+3, 35, 17), Qt.AlignmentFlag.AlignRight, str(xmax))
        q.drawText(QRectF(area.left(), area.bottom()+3, area.width(), 20), Qt.AlignmentFlag.AlignCenter, 'Epoch')
        if not self.history:
            q.drawText(area, Qt.AlignmentFlag.AlignCenter, '학습을 시작하면 표시됩니다')
        for col, (_, color) in enumerate(legends, 1):
            path = QPainterPath()
            started = False
            for j, row in enumerate(self.history):
                if row[col] is None:
                    started = False
                    continue
                point = QPointF(area.left()+area.width()*(row[0]-1)/(xmax-1), area.bottom()-area.height()*(row[col]-low)/(high-low))
                if not started: path.moveTo(point)
                else: path.lineTo(point)
                started = True
            q.setPen(QPen(QColor(color), 2))
            q.setBrush(Qt.BrushStyle.NoBrush)
            q.drawPath(path)


class Detail(QWidget):
    def __init__(self, p, state, solved=False):
        super().__init__()
        self.p, self.state = p, state
        self.solved = solved
        self.setMinimumSize(390, 390)

    def paintEvent(self, e):
        q = QPainter(self)
        q.setRenderHint(QPainter.RenderHint.Antialiasing)
        side = min(self.width(), self.height()) - 30
        paint_board(q, self.p, self.state, QRectF(10, 10, side, side), 1 if self.solved else 0)


def efficiency_metrics(puzzles, paths, statuses):
    raw, values, moves = [], [], []
    for p, path, status in zip(puzzles, paths, statuses):
        actual = display_moves(p, path, status == 'solved')
        moves.append(actual)
        ratio = (p.minimum_possible / actual if p.minimum_possible is not None and actual > 0 else None) if status == 'solved' else 0.0
        raw.append(ratio)
        values.append(ratio)
    return {'efficiency_mean': float(np.mean(values)) if values and all(v is not None for v in values) else None,
            'efficiency_values': values, 'efficiency_raw_ratios': raw,
            'evaluation_moves': moves, 'efficiency_count': len(puzzles),
            'efficiency_above_one': sum(v is not None and v > 1 for v in raw),
            'efficiency_rule': 'solved=minimum/actual; failure=0; mean over all visible puzzles'}


class Worker(QThread):
    progress = Signal(object)
    result = Signal(object)
    error = Signal(str)

    def __init__(self, mode, net=None, puzzles=None, epochs=10, source=None, live_delay=0):
        super().__init__()
        self.mode, self.net, self.puzzles, self.epochs, self.source = mode, net, puzzles, epochs, source
        self.live_delay = live_delay

    def run(self):
        try:
            if self.mode == 'import':
                self.result.emit(('import', load_apk(self.source)))
                return
            if self.mode == 'infer':
                paths, states = rollout(self.net, self.puzzles, stopped=self.isInterruptionRequested)
                self.result.emit(('infer', (paths, states)))
                return
            train, valid = split(self.puzzles)
            if not train:
                raise ValueError('현재 화면에 학습 문제가 없습니다. 전체 또는 학습 필터로 이동하세요.')
            rng = np.random.default_rng(42)
            logdir = Path('runs'); logdir.mkdir(exist_ok=True)
            logpath = logdir / f'rl-{time.time_ns()}.jsonl'
            monitor = self.puzzles
            train_ids = {id(p) for p in train}
            with logpath.open('w') as log:
                for epoch in range(self.epochs):
                    epsilon = epsilon_at(epoch + 1, self.epochs)
                    board_states = [p.start for p in monitor]
                    board_paths = [[] for p in monitor]
                    board_statuses = ['ready' for p in monitor]
                    board_last = [None for p in monitor]
                    board_totals = [0.0 for p in monitor]
                    indices = {id(p): i for i, p in enumerate(monitor)}
                    def callback(group, phase):
                        def live(frame):
                            for j, p in enumerate(group):
                                i = indices[id(p)]
                                board_states[i] = frame['states'][j]
                                board_paths[i] = frame['paths'][j]
                                board_statuses[i] = frame['statuses'][j]
                                board_last[i] = frame['last_rewards'][j]
                                board_totals[i] = frame['total_rewards'][j]
                            self.progress.emit({'live': {'states': list(board_states),
                                'paths': list(board_paths), 'statuses': list(board_statuses),
                                'step': frame['step'], 'last_rewards': list(board_last), 'total_rewards': list(board_totals)}, 'puzzles': monitor, 'phase': phase, 'epsilon': epsilon if phase == '학습 탐험' else 0.0,
                                'learning': [id(p) in train_ids for p in monitor], 'epoch': epoch+1})
                            if self.live_delay: self.msleep(self.live_delay)
                        return live
                    training = episodes(self.net, train, [True]*len(train), rng,
                                        self.isInterruptionRequested, callback(train, '학습 탐험'), epsilon=epsilon)
                    if training is None or self.isInterruptionRequested(): break
                    _, training_statuses, rewards, losses = training
                    exploration_statuses = training_statuses
                    evaluation_net = self.net.copy()
                    evaluation = episodes(evaluation_net, monitor, [False]*len(monitor), rng,
                                          self.isInterruptionRequested, callback(monitor, '전체 greedy 평가'), greedy=True)
                    if evaluation is None or self.isInterruptionRequested(): break
                    _, evaluation_statuses, _, _ = evaluation
                    training_statuses = [status for p, status in zip(monitor, evaluation_statuses) if id(p) in train_ids]
                    validation_statuses = [status for p, status in zip(monitor, evaluation_statuses) if id(p) not in train_ids]
                    efficiency = efficiency_metrics(monitor, evaluation[0], evaluation_statuses)
                    display_solved = board_statuses.count('solved')
                    metric = {'epoch': epoch+1, 'algorithm': 'epsilon-greedy V-trace actor-critic', 'epsilon': epsilon, 'epsilon_start': 1.0, 'epsilon_end': 0.05, 'epsilon_decay_fraction': 0.8, 'vtrace_rho_clip': 1.0, 'vtrace_c_clip': 1.0,
                              'loss': float(np.mean(losses)) if losses else 0.0,
                              'episode_reward': float(np.mean(rewards)),
                              'train_solved': training_statuses.count('solved'),
                              'train_evaluated': len(training_statuses),
                              'train_success_rate': training_statuses.count('solved')/len(training_statuses),
                              'validation_success_rate': (validation_statuses.count('solved')/len(validation_statuses) if validation_statuses else None),
                              'solved': validation_statuses.count('solved'), 'evaluated': len(validation_statuses),
                              'display_solved': display_solved, 'display_count': len(monitor),
                              'episodes': len(rewards), 'scope': 'visible_page',
                              'train_metric': 'frozen_greedy_success',
                              'exploration_solved': exploration_statuses.count('solved'),
                              'exploration_evaluated': len(exploration_statuses),
                              'evaluation_scope': 'all_visible_puzzles',
                              'validation_metric': 'frozen_greedy_success',
                              'puzzle_names': [p.name for p in monitor], 'seed': 42,
                              'gamma': 0.99, 'reward_version': 'distance_exit_v1', 'square_reward': -0.01, 'exit_cost_included': True, 'repeat_penalty': -0.03,
                              'success_reward': 1.0, 'limit': MAX_MOVES, 'success_condition': 'clear_exit_path', 'n_step': 16}
                    metric.update(efficiency)
                    log.write(json.dumps(metric)+'\n'); log.flush()
                    self.progress.emit({'metrics': metric, 'net': self.net.copy()})
            self.result.emit(('train', self.net.copy()))
        except Exception as exc:
            self.error.emit(str(exc))


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Rush Hour · Neural Array Lab v0.2.10 · RL')
        self.resize(1600, 1000)
        self.net = Network()
        self.puzzles = []
        self.settings_path = Path(__file__).resolve().parents[1] / 'data' / 'preferences.json'
        self.worker = None
        self.page = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.setInterval(220)
        root = QWidget(); self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        title = QLabel('NEURAL ARRAY LAB    /    RUSH HOUR')
        title.setStyleSheet('font-size: 22px; font-weight: 700; color: #76d9cb; padding: 6px;')
        layout.addWidget(title)
        top = QHBoxLayout(); layout.addLayout(top)
        self.heat = Heatmaps(self.net); top.addWidget(self.heat, 3)
        self.success_plot = AccuracyPlot(success=True); top.addWidget(self.success_plot, 1)
        self.accuracy_plot = AccuracyPlot(); top.addWidget(self.accuracy_plot, 1)
        self.efficiency_plot = AccuracyPlot(efficiency=True); top.addWidget(self.efficiency_plot, 1)
        bar = QHBoxLayout(); layout.addLayout(bar)
        self.locked = []
        def button(text, action, lock=True):
            b = QPushButton(text); b.clicked.connect(action); bar.addWidget(b)
            if lock: self.locked.append(b)
            return b
        button('APK 가져오기', self.import_apk)
        button('강화학습', self.train)
        button('신경망 추론', self.infer)
        button('정지', self.stop, False)
        button('저장', self.save)
        button('불러오기', self.load)
        button('모델 초기화', self.reset)
        bar.addWidget(QLabel('Epochs'))
        self.epochs = QSpinBox(); self.epochs.setRange(1, 1000); self.epochs.setValue(10); bar.addWidget(self.epochs)
        nav = QHBoxLayout(); layout.addLayout(nav)
        self.filter = QComboBox(); self.filter.addItems(['전체', '학습 그룹', '검증 그룹'])
        self.filter.currentIndexChanged.connect(self.change_filter); nav.addWidget(self.filter)
        self.prev = QPushButton('‹ 이전 100'); self.prev.clicked.connect(lambda: self.change_page(-1)); nav.addWidget(self.prev)
        self.next = QPushButton('다음 100 ›'); self.next.clicked.connect(lambda: self.change_page(1)); nav.addWidget(self.next)
        self.page_label = QLabel(); nav.addWidget(self.page_label)
        nav.addStretch(); nav.addWidget(QLabel('재생 간격 ms'))
        speed = QSpinBox(); speed.setRange(30, 2000); speed.setValue(220); speed.valueChanged.connect(self.timer.setInterval); nav.addWidget(speed)
        self.info = QLabel('APK 가져오기를 선택하세요. 기본 화면은 Easy001~Easy100입니다.')
        self.info.setWordWrap(True); layout.addWidget(self.info)
        self.metrics_label = QLabel('224 → 128 → 64 → [Policy 140 | Value 1]   ·   ε-greedy 1.00 → 0.05 · V-trace Actor–Critic · rewards only · no reference solutions')
        self.metrics_label.setWordWrap(True); layout.addWidget(self.metrics_label)
        self.boards = Boards(); self.boards.clicked.connect(self.detail)
        fit = QCheckBox('100개 한 화면'); fit.setChecked(True); fit.toggled.connect(self.boards.set_fit); nav.addWidget(fit)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(self.boards); layout.addWidget(scroll, 1)
        self.show_page()
        QTimer.singleShot(0, self.restore_apk)

    def restore_apk(self):
        if not self.settings_path.exists():
            return
        try:
            path = json.loads(self.settings_path.read_text())['apk_path']
            if not Path(path).is_file():
                raise ValueError('이전에 선택한 APK를 찾을 수 없습니다. 다시 선택하세요.')
            self.info.setText('이전에 선택한 APK를 불러오는 중…')
            self.launch(Worker('import', source=path))
        except (OSError, ValueError, KeyError) as exc:
            self.info.setText(str(exc))

    def selected(self):
        if self.filter.currentIndex() == 0: return self.puzzles
        return split(self.puzzles)[self.filter.currentIndex()-1]

    def change_filter(self):
        self.page = 0; self.show_page()

    def change_page(self, amount):
        self.page = max(0, min((len(self.selected())-1)//100, self.page+amount)); self.show_page()

    def show_page(self):
        self.timer.stop()
        group = self.selected()
        self.boards.set_puzzles(group[self.page*100:(self.page+1)*100])
        self.page_label.setText(f'{self.page+1} / {max(1, (len(group)+99)//100)}   ·   {len(group)} puzzles')

    def busy(self, active):
        for widget in self.locked + [self.filter, self.prev, self.next, self.epochs]:
            widget.setEnabled(not active)

    def launch(self, worker):
        self.timer.stop(); self.worker = worker; self.busy(True)
        worker.progress.connect(self.progress)
        worker.result.connect(self.result)
        worker.error.connect(lambda text: QMessageBox.critical(self, 'Error', text))
        worker.finished.connect(self.finished)
        worker.start()

    def finished(self):
        self.busy(False)
        if self.worker:
            self.worker.deleteLater(); self.worker = None

    def import_apk(self):
        path, _ = QFileDialog.getOpenFileName(self, '로컬 APK 선택 (실행하지 않음)', '', 'APK (*.apk)')
        if path:
            self.info.setText('APK 데이터 검사 중…')
            self.launch(Worker('import', source=path))

    def train(self):
        if not self.puzzles:
            QMessageBox.information(self, 'APK 필요', '먼저 APK 가져오기로 문제를 불러오세요.')
            return
        visible = list(self.boards.puzzles)
        training, validation = split(visible)
        if not training:
            QMessageBox.information(self, '학습 문제 없음', '현재 화면은 검증 문제만 있습니다. 전체 또는 학습 필터로 이동하세요.')
            return
        self.info.setText(f'현재 화면 {len(visible)}개만 강화학습 · 학습 {len(training)} / 검증 {len(validation)} · 검증은 업데이트 제외')
        self.accuracy_plot.clear(); self.success_plot.clear(); self.efficiency_plot.clear()
        self.launch(Worker('train', self.net.copy(), visible, self.epochs.value(), live_delay=30))

    def infer(self):
        if self.boards.puzzles:
            self.info.setText(f'Greedy inference · legal mask · 반복 또는 {MAX_MOVES}회 행동에서 종료')
            self.launch(Worker('infer', self.net.copy(), list(self.boards.puzzles)))

    def stop(self):
        self.timer.stop()
        if self.worker:
            self.worker.requestInterruption()
            self.info.setText('현재 작업 단위 종료 후 정지합니다.')

    def tick(self):
        if not self.boards.advance(): self.timer.stop()

    def progress(self, payload):
        if 'live' in payload:
            self.timer.stop()
            frame = payload['live']
            self.boards.puzzles = payload['puzzles']
            self.boards.states = frame['states']
            self.boards.paths = frame['paths']
            self.boards.statuses = frame['statuses']
            self.boards.final = list(frame['statuses'])
            self.boards.step = max((len(path) for path in frame['paths']), default=0)
            self.boards.last_rewards = frame['last_rewards']
            self.boards.total_rewards = frame['total_rewards']
            self.boards.learning = payload['learning']
            self.boards.sync_exits()
            self.boards.update()
            self.page_label.setText(f"RL epoch {payload['epoch']} · {payload['phase']} · ε={payload['epsilon']:.3f} · {frame['step']}/{MAX_MOVES}회 행동 · 학습 / 검증")
            return
        if 'message' in payload:
            self.metrics_label.setText(payload['message']); return
        self.net = payload['net']; self.heat.set_network(self.net)
        m = payload['metrics']
        self.accuracy_plot.append(m); self.success_plot.append(m); self.efficiency_plot.append(m)
        validation_rate = f"{m['validation_success_rate']:.1%}" if m['evaluated'] else '해당 없음'
        self.metrics_label.setText(f"Epoch {m['epoch']} · ε={m['epsilon']:.3f} | 학습 greedy 해결 {m['train_solved']}/{m['train_evaluated']} ({m['train_success_rate']:.1%})"
                            f" | 검증 greedy 해결 {m['solved']}/{m['evaluated']} ({validation_rate})"
                            f" | 탐험 평균 보상 {m['episode_reward']:.3f} | RL loss {m['loss']:.4f}"
                            f" | 화면 성공 {m['display_solved']}/{m['display_count']} | 학습 episodes {m['episodes']}")

    def result(self, payload):
        mode, result = payload
        if mode == 'import':
            self.accuracy_plot.clear(); self.success_plot.clear(); self.efficiency_plot.clear()
            self.puzzles, digest = result
            Path('data').mkdir(exist_ok=True)
            Path('data/provenance.json').write_text(json.dumps({'sha256': digest, 'count': len(self.puzzles)}, indent=2))
            try:
                self.settings_path.parent.mkdir(parents=True, exist_ok=True)
                self.settings_path.write_text(json.dumps({'apk_path': str(Path(self.worker.source).resolve())}, indent=2))
            except OSError as exc:
                QMessageBox.warning(self, '경로 저장 실패', f'문제는 불러왔지만 다음 실행을 위한 경로 저장에 실패했습니다: {exc}')
            self.filter.blockSignals(True)
            self.filter.setCurrentIndex(0)
            self.filter.blockSignals(False)
            self.page = 0; self.show_page()
            self.info.setText(f'APK {len(self.puzzles):,}개 로드 / SHA256 {digest[:16]}… / 원본 업로드·복사 없음')
        elif mode == 'infer':
            paths, statuses = result
            self.boards.set_paths(paths, statuses); self.timer.start()
            self.metrics_label.setText(f'Greedy solved {statuses.count("solved")}/{len(statuses)} · loop {statuses.count("loop")} · max {MAX_MOVES} actions')
        else:
            self.net = result; self.heat.set_network(self.net)
            self.info.setText('강화학습 종료 · 화면·성공률=학습/검증 모두 고정 모델 greedy 평가 · 보상 그래프=학습 탐험 · 저장으로 RL 가중치 보관')

    def save(self):
        path, _ = QFileDialog.getSaveFileName(self, '가중치 저장', 'checkpoints/model.npz', 'NumPy weights (*.npz)')
        if path:
            try: self.net.save(path)
            except Exception as exc: QMessageBox.critical(self, 'Save', str(exc))

    def load(self):
        path, _ = QFileDialog.getOpenFileName(self, '가중치 불러오기', '', 'NumPy weights (*.npz)')
        if path:
            try:
                self.net = Network.load(path); self.heat.set_network(self.net); self.show_page()
                self.accuracy_plot.clear(); self.success_plot.clear(); self.efficiency_plot.clear()
                self.info.setText('RL 가중치와 optimizer 로드 완료 · 다음 실행의 그래프/탐험 난수는 새로 시작합니다.')
            except Exception as exc: QMessageBox.critical(self, 'Load', str(exc))

    def reset(self):
        if QMessageBox.question(self, '초기화', '현재 가중치를 초기화할까요? 저장 파일은 유지됩니다.') == QMessageBox.StandardButton.Yes:
            self.net = Network(); self.heat.set_network(self.net); self.show_page()
            self.accuracy_plot.clear(); self.success_plot.clear(); self.efficiency_plot.clear()

    def detail(self, i):
        p, state = self.boards.puzzles[i], self.boards.states[i]
        dialog = QDialog(self); dialog.setWindowTitle(p.name + ' · current state')
        layout = QVBoxLayout(dialog); layout.addWidget(Detail(p, state, self.boards.statuses[i] == 'solved'))
        layout.addWidget(QLabel(f'Minimum possible: {p.minimum_possible if p.minimum_possible is not None else "—"}'))
        last = self.boards.last_rewards[i]
        total = self.boards.total_rewards[i]
        layout.addWidget(QLabel((f'Reward: {last:+.2f}' if last is not None else 'Reward: —') + f'\nTotal reward: {total:+.2f}'))
        out = self.net.forward(p.encode(state)[None])[0]
        legal = p.actions(state)
        scores = out[legal]; prob = np.exp(scores - scores.max()); prob /= prob.sum()
        top = sorted(zip(legal, prob), key=lambda x: -x[1])[:5]
        lines = [f'Value: {out[140]:.3f} expected discounted reward', 'Top legal actions:']
        for a, pr in top:
            car, code = divmod(a, 10)
            direction = ('L' if code < 5 else 'R') if p.cars[car].horizontal else ('U' if code < 5 else 'D')
            lines.append(f'{p.cars[car].label.upper()} {direction} {code%5+1} : {pr:.1%}')
        label = QLabel('\n'.join(lines)); layout.addWidget(label); dialog.exec()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            self.info.setText('작업 정지 요청 중입니다. 종료 후 창을 닫아주세요.')
            event.ignore()
        else:
            event.accept()


def configure_style(app):
    app.setStyle('Fusion')
    app.setStyleSheet('''QWidget {background:#111b2b; color:#dce7f5; font-size:12px;}
        QPushButton,QComboBox,QSpinBox {background:#24364b; border:1px solid #3d566d; border-radius:5px; padding:6px;}
        QPushButton:hover {background:#355575;} QPushButton:disabled {color:#62738a;}
        QScrollArea {border:none;}''')


def main():
    app = QApplication.instance() or QApplication([])
    configure_style(app)
    window = Window(); window.show()
    app.exec()
