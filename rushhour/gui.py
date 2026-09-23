import json
from pathlib import Path
import time
from dataclasses import replace
import numpy as np
from PySide6.QtCore import Qt, QRectF, QPointF, QTimer, QThread, Signal
from PySide6.QtGui import QColor, QPainter, QFont, QImage, QPen, QPainterPath
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSpinBox, QFileDialog, QMessageBox, QScrollArea, QDialog, QCheckBox, QSlider, QTabWidget, QProgressBar)
from .core import load_apk, split_all, split_role, permute_vehicles, MAX_MOVES, display_moves
from .model import Network
from .expert_policy import PolicyNetwork
from .checkpoints import load_network
from .boarddata import load_initial_boards
from .expanded_rl import ExpandedRL
from .training import curriculum_pool, canonical_frame, canonical_actions, SuccessReplay, diagnostic_split
from .rl import episodes, replay_successes, replay_rewards, SUCCESS_REWARD, STEP_REWARD, epsilon_at

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
        self.roles = []
        self.all_trained = False
        self.role_overrides = None
        self.action_sources = []
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
        self.roles = ['train' if self.all_trained else (self.role_overrides.get(p.name,'unused') if self.role_overrides is not None else split_role(p)) for p in puzzles]
        self.action_sources = ['ready'] * len(puzzles)
        self.learning = [role == 'train' for role in self.roles]
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
                role = {'train': '학습', 'validation': '검증', 'test': '테스트', 'unused':'미사용'}.get(self.roles[i], '') if i < len(self.roles) else ''
                if self.action_sources[i]=='teacher':role += ' · 교사 상태'
                if self.action_sources[i] in ('random','greedy'):role += ' · '+('탐험' if self.action_sources[i]=='random' else 'greedy')
                info = (f'{p.name} '+(f'[{role}]' if role else '')
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
        if getattr(net, 'policy_only', False):
            matrices = net.w
            self.names = [f'{"POLICY" if i == len(matrices)-1 else f"LAYER {i+1}"}   {w.shape[0]} × {w.shape[1]}' for i,w in enumerate(matrices)]
        else:
            matrices = [net.w[0], net.w[1], net.w[2][:, :140], net.w[2][:, 140:]]
            self.names = [f'{name}   {w.shape[0]} × {w.shape[1]}' for name,w in zip(['INPUT → H1','H1 → H2','POLICY','VALUE'],matrices)]
        for w in matrices:
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
        width = self.width()/len(self.images)
        for i, ((im, scale), name) in enumerate(zip(self.images, self.names)):
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
            self.setToolTip('전체 학습+검증 greedy 평가 평균 (테스트 제외). 성공: Minimum possible / moves, 실패: 0.\n비율이 1을 초과하면 그대로 표시하고 로그에 별도로 기록합니다. 기준값 누락 시 평균은 표시하지 않습니다.')
        if success:
            self.setToolTip('Train explore: 현재 curriculum 학습 집합의 탐험 성공률.\nTrain greedy / Val greedy: 매 epoch 전체 학습/검증 집합의 고정 모델 평가. 페이지 이동과 무관합니다.')

    def clear(self):
        self.history.clear()
        self.update()

    def append(self, metrics):
        sl=getattr(self,'supervised_kind',None)
        if sl:
            keys={'solve':['train_success_rate','validation_success_rate'],'loss':['loss'],'accuracy':['action_accuracy']}[sl]
            self.history.append(tuple([metrics['epoch']]+[metrics[k] for k in keys]));self.update();return
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
        sl=getattr(self,'supervised_kind',None)
        percentage=self.success or sl=='accuracy'
        title = 'EFFICIENCY · train + val' if self.efficiency else ('SOLVE SUCCESS (%)' if self.success else 'RL EPISODE REWARD')
        if sl:title={'solve':'SOLVE SUCCESS (%)','loss':'SUPERVISED LOSS','accuracy':'TEACHER ACTION ACCURACY (%)'}[sl]
        q.drawText(QRectF(8, 0, self.width()-16, 22), Qt.AlignmentFlag.AlignLeft, title)
        legends = [('Train explore', '#67dca4'), ('Train greedy', '#62b5ff'), ('Val greedy', '#f8b65a')] if self.success else [('Train + val' if self.efficiency else 'Train explore', '#67dca4')]
        if sl:legends={'solve':[('Train greedy','#62b5ff'),('Val greedy','#f8b65a')],'loss':[('Batch CE','#f8b65a')],'accuracy':[('Batch accuracy','#67dca4')]}[sl]
        q.setFont(QFont('Arial', 8))
        labels = []
        for col, (name, color) in enumerate(legends):
            v = self.history[-1][col+1] if self.history else None
            value = (f' {v:.1%}' if percentage else f' {v:.3f}') if v is not None else ' —'
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
        elif not self.success:
            low = min(low, float(np.floor(min((row[1] for row in self.history if row[1] is not None), default=low))))
        if sl=='loss':low,high=0,max(1,max((r[1] for r in self.history),default=1)*1.1)
        elif sl:low,high=0,1
        for value in (low, (low+high)/2, high):
            y = area.bottom()-area.height()*(value-low)/(high-low)
            q.setPen(QColor('#334259'))
            q.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
            q.setPen(QColor('#8fa6c0'))
            q.drawText(QRectF(0, y-8, 35, 16), Qt.AlignmentFlag.AlignRight, f'{value*100:.0f}' if percentage else f'{value:.3g}')
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
            'efficiency_rule': 'solved=minimum/actual; failure=0; mean over all train and validation puzzles'}


class Worker(QThread):
    progress = Signal(object)
    result = Signal(object)
    error = Signal(str)

    def __init__(self, mode, net=None, puzzles=None, epochs=10, source=None, live_delay=0, limit=MAX_MOVES,
                 diagnostic=False, diagnostic_count=4, augment=True, seed=42, replay_max=30):
        super().__init__()
        self.diagnostic = diagnostic
        self.diagnostic_count, self.augment, self.seed = diagnostic_count, augment, seed
        self.artifact_dir = None
        self.replay_max = replay_max
        self.last_update_emit = 0.0
        self.phase_done=0;self.phase_total=1
        self.mode, self.net, self.puzzles, self.epochs, self.source = mode, net, puzzles, epochs, source
        self.live_delay, self.limit = live_delay, limit
        self.epoch, self.phase, self.epsilon = 0, '', 0.0

    def learning_update(self, kind, loss, count):
        now = time.monotonic()
        if now-self.last_update_emit < .2 and not self.live_delay:return
        self.last_update_emit = now
        self.progress.emit(dict(update=kind,loss=float(loss),transitions=count,epoch=self.epoch,update_count=self.net.t,net=self.net.copy()))
        if self.live_delay:self.msleep(self.live_delay)

    def start_phase(self, phase, message):
        self.phase = phase
        self.progress.emit({'phase_start': phase, 'epoch': self.epoch, 'message': message})

    def callback(self, originals, augmented=None):
        last_emit = [0.0]
        def live(frame):
            now = time.monotonic()
            terminal = all(s != 'running' for s in frame['statuses'])
            if frame['step'] and not terminal and now-last_emit[0] < .08:
                return
            last_emit[0] = now
            if augmented is not None:
                frame = canonical_frame(originals, augmented, frame)
            self.progress.emit({'live': frame, 'puzzles': originals, 'phase': self.phase,
                                'epoch': self.epoch, 'epsilon': self.epsilon if self.phase == 'explore' else 0.0,
                                'completed':self.phase_done+sum(s!='running' for s in frame['statuses']), 'total':self.phase_total})
            if self.live_delay: self.msleep(self.live_delay)
        return live

    def evaluate_groups(self, net, puzzles, rng):
        paths, statuses = [], []
        for start in range(0, len(puzzles), 100):
            self.phase_done=start;self.phase_total=len(puzzles)
            group = puzzles[start:start+100]
            result = episodes(net, group, [False]*len(group), rng,
                              self.isInterruptionRequested, self.callback(group),
                              limit=self.limit, greedy=True)
            if result is None: return None
            paths.extend(result[0]); statuses.extend(result[1])
        return paths, statuses

    def run(self):
        try:
            if self.mode == 'import':
                self.result.emit(('import', load_apk(self.source)))
                return
            if self.mode == 'supervised':
                from .supervised import run_supervised
                run_supervised(self)
                return
            rng = np.random.default_rng(self.seed)
            if self.mode in ('infer', 'test', 'all'):
                self.start_phase(self.mode, '고정 모델 greedy 평가 · 가중치 업데이트 없음')
                result = self.evaluate_groups(self.net, self.puzzles, rng)
                if result is not None:
                    metric = efficiency_metrics(self.puzzles, *result)
                    metric.update({'mode': self.mode, 'puzzle_names': [p.name for p in self.puzzles],
                                   'solved': result[1].count('solved'), 'count': len(self.puzzles),
                                   'training_scope': getattr(self.net, 'training_scope', 'RL topology split'),
                                   'statuses': result[1], 'paths': result[0]})
                    if self.mode in ('test', 'all'):
                        Path('runs').mkdir(exist_ok=True)
                        Path(f'runs/{self.mode}-{time.time_ns()}.json').write_text(json.dumps(metric))
                    self.result.emit((self.mode, metric))
                return
            if getattr(self.net, 'policy_only', False):
                raise ValueError('교사 학습 모델은 추론용입니다. 강화학습은 새 RL 모델로 시작하세요.')
            train, valid, test = diagnostic_split(self.puzzles, self.diagnostic_count) if self.diagnostic else split_all(self.puzzles)
            if not train: raise ValueError('학습 집합이 비어 있습니다.')
            # Keep app minima in UI/evaluation objects only. The learner receives
            # initial geometry and zero reference actions, never answer metadata.
            no_answers = getattr(self.net,'no_answer_training',False)
            original_train = train
            if no_answers:train=[replace(p,solution=(),minimum_possible=None,minimum_source='') for p in train]
            replay = SuccessReplay(train,capacity=len(train) if no_answers else 500)
            evaluated = original_train + valid if self.diagnostic else [p for p in self.puzzles if split_role(p) != 'test']
            roles = {p.name: role for role, group in [('train', train), ('validation', valid), ('test', test)] for p in group}
            Path('runs').mkdir(exist_ok=True)
            run_id = time.time_ns()
            log_path = Path(f'runs/rl-{run_id}.jsonl')
            best_score, best_epoch = (-1, -1.0), None
            if self.diagnostic or no_answers:
                self.artifact_dir = Path(f'checkpoints/diagnostic-{run_id}')
                self.artifact_dir.mkdir(parents=True)
                self.net.save(self.artifact_dir/'initial.npz')
            complete_log=[]
            with log_path.open('w') as log:
                manifest={'type': 'split_manifest', 'roles': roles,
                                      'split_rule': 'diagnostic_fixed_cohort' if self.diagnostic else 'topology_sha256_mod10_v1',
                                      'diagnostic_count': self.diagnostic_count if self.diagnostic else None,
                                      'permutation_augmentation': self.augment,
                                      'fresh_initialization': self.net.t == 0, 'initial_adam_step': self.net.t, 'seed': self.seed,
                                      'external_answers':False,'teacher_weights':False,'minimum_used_for_training':not(no_answers or self.diagnostic),
                                      'architecture':[224,self.net.w[0].shape[1],self.net.w[1].shape[1],141]}
                complete_log.append(manifest);log.write(json.dumps(manifest)+'\n')
                for self.epoch in range(1, self.epochs+1):
                    if self.isInterruptionRequested(): break
                    self.epsilon = epsilon_at(self.epoch, self.epochs)
                    eligible, ceiling, fallback = (train, None, False) if self.diagnostic or no_answers else curriculum_pool(train, self.epoch, self.epochs)
                    order = rng.permutation(len(eligible))
                    chosen = [eligible[int(i)] for i in order]
                    stage = f'≤{ceiling}' if ceiling is not None else '전체'
                    self.start_phase('explore', f'Epoch {self.epoch} · 학습 {len(chosen)}개 · curriculum {stage} · ε={self.epsilon:.3f}')
                    rewards, losses, exploration = [], [], []
                    for start in range(0, len(chosen), 100):
                        self.phase_done=start;self.phase_total=len(chosen)
                        originals = chosen[start:start+100]
                        augmented = [permute_vehicles(p, rng) for p in originals] if self.augment else originals
                        result = episodes(self.net, augmented, [True]*len(augmented), rng,
                                          self.isInterruptionRequested, self.callback(originals, augmented),
                                          limit=self.limit, epsilon=self.epsilon,on_update=lambda loss,n:self.learning_update('V-trace',loss,n))
                        if result is None: break
                        paths, statuses, totals, fresh_losses = result
                        exploration.extend(statuses); rewards.extend(totals); losses.extend(fresh_losses)
                        for p, q, path, status in zip(originals, augmented, paths, statuses):
                            if status == 'solved': replay.add(p, canonical_actions(p, q, path))
                    if self.isInterruptionRequested(): break
                    self.start_phase('learn',f'Epoch {self.epoch} · 스스로 성공한 경험 복습 · 보관 {len(replay)}문제')
                    replay_losses, replay_count, replay_steps = replay_successes(
                        self.net, replay.records(), rng, stopped=self.isInterruptionRequested, augment=self.augment,
                        max_records=self.replay_max,on_update=lambda loss,n:self.learning_update('성공 경험 복습',loss,n))
                    if self.isInterruptionRequested(): break
                    frozen = self.net.copy()
                    self.start_phase('evaluate', f'Epoch {self.epoch} · 전체 학습 {len(train)} + 검증 {len(valid)} greedy 평가 · 테스트 별도')
                    evaluation = self.evaluate_groups(frozen, evaluated, rng)
                    if evaluation is None or self.isInterruptionRequested(): break
                    paths, statuses = evaluation
                    train_status = [s for p, s in zip(evaluated, statuses) if roles[p.name] == 'train']
                    val_status = [s for p, s in zip(evaluated, statuses) if roles[p.name] == 'validation']
                    metric = {'epoch': self.epoch, 'algorithm': 'epsilon-greedy V-trace + positive-advantage self-imitation',
                              'epsilon': self.epsilon, 'seed': self.seed, 'gamma': .99,
                              'loss': float(np.mean(losses)) if losses else 0.,
                              'replay_loss': float(np.mean(replay_losses)) if replay_losses else 0.,
                              'episode_reward': float(np.mean(rewards)) if rewards else 0.,
                              'train_solved': train_status.count('solved'), 'train_evaluated': len(train_status),
                              'train_success_rate': train_status.count('solved')/len(train_status),
                              'solved': val_status.count('solved'), 'evaluated': len(val_status),
                              'validation_success_rate': val_status.count('solved')/len(val_status) if val_status else None,
                              'display_solved': statuses.count('solved'), 'display_count': len(statuses),
                              'episodes': len(exploration), 'exploration_evaluated': len(exploration),
                              'exploration_solved': exploration.count('solved'),
                              'split_counts': {'train': len(train), 'validation': len(valid), 'test': len(test)},
                              'scope': f'diagnostic_train_{len(train)}' if self.diagnostic else 'all_curriculum_eligible_train_once', 'batch_size': 100,
                              'evaluation_scope': 'selected_train_and_validation' if self.diagnostic else 'all_train_and_validation', 'test_evaluated': 0,
                              'evaluation_statuses': statuses,
                              'puzzle_names': [p.name for p in evaluated], 'training_names': [p.name for p in chosen],
                              'curriculum_maximum': ceiling, 'curriculum_eligible': len(eligible), 'curriculum_fallback': fallback,
                              'permutation_augmentation': self.augment, 'replay_buffer_size': len(replay),
                              'replay_episodes': replay_count, 'replay_transitions': replay_steps,
                              'replay_capacity': replay.capacity, 'replay_max_per_epoch': self.replay_max,
                              'reward_version': 'distance_exit_v1', 'square_reward': -.01,
                              'repeat_penalty': -.03, 'success_reward': 1., 'exit_cost_included': True,
                              'limit': self.limit, 'n_step': 16}
                    metric.update(efficiency_metrics(evaluated, paths, statuses))
                    if self.artifact_dir:
                        training_efficiency = efficiency_metrics(train, paths[:len(train)], statuses[:len(train)])['efficiency_mean']
                        observed_reward = float(np.mean([replay_rewards(p,path)[1] for p,path in zip(evaluated,paths) if roles[p.name]=='train']))
                        score = (metric['train_solved'], observed_reward)
                        if score > best_score:
                            best_score, best_epoch = score, self.epoch
                            self.net.save(self.artifact_dir / 'best.npz')
                        metric['best_epoch'] = best_epoch
                        metric['checkpoint_selection'] = 'training solved count, then observed greedy reward; no minimum or holdout selection'
                        metric['checkpoint_score_reward'] = observed_reward
                    complete_log.append(metric);log.write(json.dumps(metric)+'\n'); log.flush()
                    self.progress.emit({'metrics': metric, 'net': self.net.copy()})
            temporary=log_path.with_suffix('.complete.tmp');temporary.write_text(''.join(json.dumps(r)+'\n' for r in complete_log));temporary.replace(log_path)
            if self.artifact_dir:
                self.net.save(self.artifact_dir / 'final.npz')
                (self.artifact_dir / 'run.json').write_text(json.dumps({
                    'log_path': str(log_path), 'best_epoch': best_epoch,
                    'seed': self.seed, 'epochs_requested': self.epochs,
                    'diagnostic_count': len(train), 'permutation_augmentation': self.augment,
                    'interrupted': self.isInterruptionRequested()}, indent=2))
            self.result.emit(('train', self.net.copy()))
        except Exception as exc:
            self.error.emit(str(exc))


class Window(QMainWindow):
    def __init__(self, learning_mode='rl'):
        super().__init__()
        self.learning_mode=learning_mode
        self.setWindowTitle('Rush Hour · Neural Array Lab v0.3.2')
        self.resize(1600, 1000)
        self.net = PolicyNetwork((224,512,512,140),seed=42) if learning_mode=='supervised' else ExpandedRL(seed=42)
        self.puzzles = []
        self.monitor_cache = {}
        self.monitor_phase = ""
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
        self.train_button = button('지도학습 계속' if learning_mode=='supervised' else '선택 범위 RL 계속', self.train)
        button('현재 100개 추론', self.infer)
        button('전체 2500개 추론', self.infer_all)
        self.test_button = button('테스트 평가', self.test)
        button('정지', self.stop, False)
        button('저장', self.save)
        button('불러오기', self.load)
        button('모델 초기화', self.reset)
        bar.addWidget(QLabel('Epochs'))
        self.epochs = QSpinBox(); self.epochs.setRange(1, 10000); self.epochs.setValue(70 if learning_mode=='supervised' else 1000); bar.addWidget(self.epochs)
        experiment = QHBoxLayout(); layout.addLayout(experiment)
        experiment.addWidget(QLabel('학습 범위'))
        self.diagnostic_size = QComboBox()
        for label, count in [('Easy001–004 · 학습 4개', 4), ('Easy001–100 중 학습 20개', 20), ('Easy001–100 중 학습 88개', 88),('전체 2500개 학습 · 검증 없음',2500),('2034학습 / 214검증 / 252테스트',2034)]:
            self.diagnostic_size.addItem(label, count)
        self.diagnostic_size.setCurrentIndex(3)
        experiment.addWidget(self.diagnostic_size)
        self.augmentation = QCheckBox('차량 순서 증강 (비교용)')
        self.augmentation.setToolTip('진단 실험에만 적용. 해제하면 탐험과 성공 replay 모두 원래 차량 순서를 사용합니다.')
        experiment.addWidget(self.augmentation)
        self.augmentation.setVisible(learning_mode=='rl')
        self.replay_max=QSpinBox();self.replay_max.setRange(0,2500);self.replay_max.setValue(30)
        replay_label=QLabel('성공 복습 / epoch');replay_label.setVisible(learning_mode=='rl');self.replay_max.setVisible(learning_mode=='rl')
        experiment.addWidget(replay_label);experiment.addWidget(self.replay_max)
        experiment.addWidget(QLabel('Seed'))
        self.experiment_seed = QSpinBox(); self.experiment_seed.setRange(0, 999999); self.experiment_seed.setValue(42)
        experiment.addWidget(self.experiment_seed)
        diagnostic_button = QPushButton('무작위 가중치로 새 지도학습' if learning_mode=='supervised' else '무작위 가중치로 새 RL'); diagnostic_button.clicked.connect(self.train_four)
        experiment.addWidget(diagnostic_button); experiment.addStretch()
        self.locked.extend([self.diagnostic_size, self.augmentation, self.experiment_seed, diagnostic_button, self.replay_max])
        phases=QHBoxLayout();layout.addLayout(phases)
        self.phase_labels={}
        for key,label in ([('prepare','교사 데이터'),('learn','학습'),('evaluate','평가')] if learning_mode=='supervised' else [('explore','탐험'),('learn','학습'),('evaluate','평가')]):
            indicator=QLabel(label);indicator.setMinimumWidth(92);indicator.setAlignment(Qt.AlignmentFlag.AlignCenter);phases.addWidget(indicator);self.phase_labels[key]=indicator
        self.phase_progress=QProgressBar();self.phase_progress.setRange(0,100);self.phase_progress.setValue(0);phases.addWidget(self.phase_progress,2)
        self.learning_status=QLabel('가중치 업데이트 0회');phases.addWidget(self.learning_status)
        self.lesson_label=QLabel('교사 행동과 모델 예측을 학습 중 비교합니다.' if learning_mode=='supervised' else '정답 경로 미사용 · 탐험 중 16행동마다 학습하고 성공 경험을 복습합니다.')
        self.lesson_label.setWordWrap(True);layout.addWidget(self.lesson_label)
        nav = QHBoxLayout(); layout.addLayout(nav)
        self.filter = QComboBox(); self.filter.addItems(['전체', '학습 그룹', '검증 그룹', '테스트 그룹'])
        self.filter.currentIndexChanged.connect(self.change_filter); nav.addWidget(self.filter)
        self.prev = QPushButton('‹ 이전 100'); self.prev.clicked.connect(lambda: self.change_page(-1)); nav.addWidget(self.prev)
        self.page_slider = QSlider(Qt.Orientation.Horizontal); self.page_slider.setRange(0, 0)
        self.page_slider.setToolTip('100개 단위 모니터 페이지. 학습 대상과 그래프 집계는 바뀌지 않습니다.')
        self.page_slider.setMinimumWidth(180); self.page_slider.valueChanged.connect(self.slider_page); nav.addWidget(self.page_slider, 1)
        self.next = QPushButton('다음 100 ›'); self.next.clicked.connect(lambda: self.change_page(1)); nav.addWidget(self.next)
        self.page_label = QLabel(); nav.addWidget(self.page_label)
        nav.addWidget(QLabel('화면 지연 ms'))
        self.speed = QSpinBox(); self.speed.setRange(0, 2000); self.speed.setValue(0); self.speed.setToolTip('0=최고 속도. 값을 올리면 탐험과 평가 화면을 천천히 보여주며 학습 시간도 늘어납니다.'); nav.addWidget(self.speed)
        self.info = QLabel('APK 가져오기를 선택하세요. 기본 화면은 Easy001~Easy100입니다.')
        self.info.setWordWrap(True); layout.addWidget(self.info)
        self.model_label = QLabel(); self.model_label.setWordWrap(True); layout.addWidget(self.model_label)
        self.metrics_label = QLabel('224 → 128 → 64 → [Policy 140 | Value 1]   ·   ε-greedy 1.00 → 0.05 · V-trace Actor–Critic · rewards only · no reference solutions')
        self.metrics_label.setWordWrap(True); layout.addWidget(self.metrics_label)
        self.boards = Boards(); self.boards.clicked.connect(self.detail)
        fit = QCheckBox('100개 한 화면'); fit.setChecked(True); fit.toggled.connect(self.boards.set_fit); nav.addWidget(fit)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(self.boards); layout.addWidget(scroll, 1)
        self.diagnostic_size.currentIndexChanged.connect(self.scope_changed)
        for plot,kind in [(self.success_plot,'solve'),(self.accuracy_plot,'loss'),(self.efficiency_plot,'accuracy')]:
            plot.supervised_kind=kind if learning_mode=='supervised' else None
            if learning_mode=='supervised':plot.setToolTip('실제 지도학습 기록. loss와 행동 정확도는 가중치가 변하는 동안의 batch 평균입니다. greedy는 epoch 후 고정 모델 평가입니다.')
        self.sync_model_view()
        self.set_phase('idle')
        self.show_page()
        QTimer.singleShot(0, self.restore_apk)

    def sync_model_view(self):
        self.learning_status.setText(f'가중치 업데이트 {self.net.t:,}회')
        expert = getattr(self.net, 'policy_only', False)
        self.boards.all_trained = False
        self.heat.set_network(self.net)
        self.filter.blockSignals(True)
        for i,label in enumerate(['전체', '학습 그룹', '검증 그룹', '테스트 그룹']):
            self.filter.setItemText(i, label)
        self.filter.blockSignals(False)
        if expert:
            shape=' → '.join(map(str,self.net.widths))
            self.model_label.setText(f'지도학습 · {shape} · 교사 행동으로 학습 · {self.diagnostic_size.currentText()}')
            self.metrics_label.setText('무작위 가중치로 새 지도학습 / 지도학습 계속 · 평가에서는 교사 경로를 사용하지 않습니다.')
        else:
            self.model_label.setText(f'강화학습 · 224 → {self.net.w[0].shape[1]} → {self.net.w[1].shape[1]} → [Policy 140 | Value 1] · 교사 가중치/정답 경로 미사용')
            self.metrics_label.setText('무작위 가중치로 새 RL / 선택 범위 RL 계속 · 탐험 → 학습 → 고정 greedy 평가')
        self.busy(self.worker is not None)

    def set_phase(self, phase):
        for key,label in self.phase_labels.items():
            label.setStyleSheet('padding:6px;border-radius:5px;background:'+('#247766;color:white;font-weight:bold;' if key==phase else '#1b2a3e;color:#8fa6c0;'))
        self.current_phase=phase

    def scope_changed(self):
        self.sync_model_view()
        self.show_page()
        self.busy(self.worker is not None)

    def restore_apk(self):
        if not self.settings_path.exists():
            initial = Path(__file__).resolve().parents[1] / 'resources' / 'initial_boards.json'
            if initial.exists():
                try:
                    self.puzzles, _ = load_initial_boards(initial)
                    self.show_page()
                    self.info.setText('초기 보드 2500개 로드 · 학습 범위를 선택하고 무작위 가중치로 새 학습을 시작하세요.')
                except (OSError, ValueError, KeyError) as exc:
                    self.info.setText(f'기본 문제 로드 실패: {exc}')
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
        try:return diagnostic_split(self.puzzles,self.diagnostic_size.currentData())[self.filter.currentIndex()-1]
        except ValueError:return []

    def change_filter(self):
        self.page = 0; self.show_page()

    def change_page(self, amount):
        self.page = max(0, min((len(self.selected())-1)//100, self.page+amount)); self.show_page()

    def slider_page(self, page):
        if page != self.page:
            self.page = page
            self.show_page()

    def show_page(self):
        self.timer.stop()
        try:
            groups=diagnostic_split(self.puzzles,self.diagnostic_size.currentData())
            self.boards.role_overrides={p.name:role for role,ps in zip(['train','validation','test'],groups) for p in ps}
        except ValueError:self.boards.role_overrides=None
        group = self.selected()
        pages = max(1, (len(group)+99)//100)
        self.page = min(self.page, pages-1)
        self.page_slider.blockSignals(True); self.page_slider.setRange(0, pages-1); self.page_slider.setValue(self.page); self.page_slider.blockSignals(False)
        self.boards.set_puzzles(group[self.page*100:(self.page+1)*100])
        self.page_label.setText(f'{self.page+1}/{pages} · {self.page*100+1 if group else 0}–{min((self.page+1)*100,len(group))} / {len(group)}')
        self.prev.setEnabled(self.page > 0); self.next.setEnabled(self.page < pages-1)
        self.render_monitor()

    def busy(self, active):
        for widget in self.locked + [self.epochs]:
            widget.setEnabled(not active)
        if not active and self.diagnostic_size.currentData() in (4,2500):
            self.test_button.setEnabled(False)

    def launch(self, worker):
        self.timer.stop(); self.worker = worker; self.busy(True)
        worker.progress.connect(self.progress)
        worker.result.connect(self.result)
        worker.error.connect(lambda text: QMessageBox.critical(self, 'Error', text))
        worker.finished.connect(self.finished)
        worker.start()

    def finished(self):
        self.set_phase('idle')
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
        count=self.diagnostic_size.currentData()
        training, validation, test = diagnostic_split(self.puzzles,count)
        if not training:
            QMessageBox.information(self, '학습 문제 없음', '전체 데이터의 학습 집합이 비어 있습니다.')
            return
        self.info.setText(f'전체 {len(self.puzzles):,}개 · 학습 {len(training)} / 검증 {len(validation)} / 테스트 {len(test)} · 100개씩 처리 · 슬라이더는 모니터 위치만 변경')
        self.accuracy_plot.clear(); self.success_plot.clear(); self.efficiency_plot.clear()
        self.launch(Worker('supervised' if self.learning_mode=='supervised' else 'train', self.net.copy(), list(self.puzzles), self.epochs.value(),live_delay=self.speed.value(),diagnostic=True,diagnostic_count=count,augment=self.augmentation.isChecked(),seed=self.experiment_seed.value(),replay_max=self.replay_max.value()))

    def train_four(self):
        count, seed = self.diagnostic_size.currentData(), self.experiment_seed.value()
        try:
            train, valid, test = diagnostic_split(self.puzzles, count)
        except ValueError as exc:
            QMessageBox.information(self, 'APK 필요', str(exc))
            return
        # Preserve the current in-memory model before replacing it with a fresh one.
        backup = Path(f'checkpoints/before-diagnostic-{time.time_ns()}.npz')
        try:
            backup.parent.mkdir(parents=True, exist_ok=True)
            self.net.save(str(backup))
        except Exception as exc:
            QMessageBox.critical(self, '백업 실패', str(exc))
            return
        self.net = PolicyNetwork((224,512,512,140),seed) if self.learning_mode=='supervised' else ExpandedRL(seed=seed)
        self.learning_status.setText('가중치 업데이트 0회')
        self.sync_model_view()
        self.monitor_cache.clear()
        self.filter.setCurrentIndex(0)
        self.page = 0
        self.show_page()
        self.accuracy_plot.clear(); self.success_plot.clear(); self.efficiency_plot.clear()
        self.info.setText(f'진단 새 학습 · 학습 {len(train)}개 / 검증 {len(valid)}개 · 이전 모델: {backup}')
        self.launch(Worker('supervised' if self.learning_mode=='supervised' else 'train', self.net.copy(), list(self.puzzles), self.epochs.value(),
                           diagnostic=True, diagnostic_count=count, augment=self.augmentation.isChecked(), seed=seed,live_delay=self.speed.value(),replay_max=self.replay_max.value()))

    def infer(self):
        if self.boards.puzzles:
            self.info.setText(f'Greedy inference · legal mask · 반복 또는 {MAX_MOVES}회 행동에서 종료')
            self.launch(Worker('infer', self.net.copy(), list(self.boards.puzzles), live_delay=self.speed.value()))

    def infer_all(self):
        if not self.puzzles: return
        self.filter.setCurrentIndex(0)
        self.info.setText(f'전체 {len(self.puzzles)}개 고정 모델 greedy 추론 · 100개씩 처리 · 슬라이더로 모니터')
        self.launch(Worker('all', self.net.copy(), list(self.puzzles), live_delay=self.speed.value()))

    def test(self):
        group = diagnostic_split(self.puzzles,self.diagnostic_size.currentData())[2]
        if not group: return
        self.filter.setCurrentIndex(3)
        self.launch(Worker('test', self.net.copy(), group, live_delay=self.speed.value()))

    def render_monitor(self):
        for i, p in enumerate(self.boards.puzzles):
            row = self.monitor_cache.get(p.name)
            if row is None: continue
            self.boards.states[i] = row['state']
            self.boards.paths[i] = row['path']
            self.boards.statuses[i] = row['status']
            self.boards.last_rewards[i] = row['last']
            self.boards.total_rewards[i] = row['total']
            self.boards.action_sources[i] = row.get('selected_by','ready')
            if row['status'] == 'solved': self.boards.exit_started[i] = row['solved_at']
        self.boards.final = list(self.boards.statuses)
        self.boards.step = max((len(p) for p in self.boards.paths), default=0)
        self.boards.sync_exits(); self.boards.update()

    def stop(self):
        self.timer.stop()
        if self.worker:
            self.worker.requestInterruption()
            self.info.setText('현재 작업 단위 종료 후 정지합니다.')

    def tick(self):
        if not self.boards.advance(): self.timer.stop()

    def progress(self, payload):
        if 'update' in payload:
            self.net=payload['net'];self.heat.set_network(self.net);self.set_phase('learn')
            self.learning_status.setText(f"가중치 업데이트 {payload['update_count']:,}회")
            self.lesson_label.setText(f"Epoch {payload['epoch']} · {payload['update']} · 실제 업데이트 {payload['transitions']}개 전이 · loss {payload['loss']:.4f}")
            return
        if 'lesson' in payload:
            self.net=payload['net'];self.heat.set_network(self.net);self.set_phase('learn')
            for row in payload['lesson']:
                self.monitor_cache[row['puzzle'].name]=dict(state=row['state'],path=list(row['path']),status='running',last=None,total=0.0,solved_at=None,selected_by='teacher')
            self.render_monitor()
            self.phase_progress.setRange(0,payload['total']);self.phase_progress.setValue(payload['done'])
            self.learning_status.setText(f"가중치 업데이트 {payload['update_count']:,}회")
            self.lesson_label.setText(f"{payload['sample']} · 교사 행동 {payload['target']} · 업데이트 전 모델 예측 {payload['predicted']} · 교사 행동 확률 {payload['probability']:.1%}")
            self.metrics_label.setText(f"Epoch {payload['epoch']} · 실제 batch 학습 {payload['done']:,}/{payload['total']:,} · CE loss {payload['loss']:.4f} · 행동 정확도 {payload['accuracy']:.1%}")
            return
        if 'phase_start' in payload:
            self.set_phase(payload['phase_start'])
            self.phase_progress.setRange(0,0 if payload['phase_start']=='learn' and self.learning_mode=='rl' else 100);self.phase_progress.setValue(0)
            if payload['phase_start']=='evaluate':self.lesson_label.setText('평가 · 신경망의 최고 점수 행동만 선택 · 교사 경로 사용 및 가중치 업데이트 없음')
            self.monitor_phase = payload['phase_start']
            if payload['phase_start']!='learn' or self.learning_mode=='supervised':self.monitor_cache.clear()
            self.show_page()
            self.info.setText(payload['message'] + ' · 다른 페이지도 슬라이더로 확인 가능 · ready=이번 단계 대기/대상 아님')
            return
        if 'live' in payload:
            self.set_phase('evaluate' if payload['phase'] in ('infer','test','all') else payload['phase'])
            self.phase_progress.setRange(0,payload.get('total',100));self.phase_progress.setValue(payload.get('completed',0))
            self.timer.stop()
            frame = payload['live']
            for j, p in enumerate(payload['puzzles']):
                old = self.monitor_cache.get(p.name, {})
                solved_at = old.get('solved_at') if old.get('status') == 'solved' else time.monotonic()
                self.monitor_cache[p.name] = {'state': frame['states'][j], 'path': frame['paths'][j],
                    'status': frame['statuses'][j], 'last': frame['last_rewards'][j],
                    'total': frame['total_rewards'][j], 'solved_at': solved_at,
                    'selected_by':frame.get('selected_by',['ready']*len(payload['puzzles']))[j]}
            self.render_monitor()
            first, count = payload['puzzles'][0].name, len(payload['puzzles'])
            phase = {'explore': '학습 탐험', 'evaluate': '학습+검증 평가', 'infer': '페이지 추론', 'test': '테스트 평가', 'all': '전체 문제 추론'}[payload['phase']]
            self.metrics_label.setText(f"Epoch {payload['epoch']} · {phase} · ε={payload['epsilon']:.3f} · 현재 처리 {first} 등 {count}개 · {frame['step']}/{MAX_MOVES} actions")
            return
        if 'message' in payload:
            self.metrics_label.setText(payload['message']); return
        self.net = payload['net']; self.heat.set_network(self.net)
        m = payload['metrics']
        self.accuracy_plot.append(m); self.success_plot.append(m); self.efficiency_plot.append(m)
        if m.get('supervised'):
            self.metrics_label.setText(f"Epoch {m['epoch']} · 교사 행동 학습 loss {m['loss']:.4f} · 행동 정확도 {m['action_accuracy']:.1%} · 학습 greedy {m['train_solved']}/{m['train_evaluated']} · 검증 greedy {m['solved']}/{m['evaluated']} · Efficiency {m['efficiency_mean']:.3f}")
            return
        validation_rate = f"{m['validation_success_rate']:.1%}" if m['evaluated'] else '해당 없음'
        curriculum = f"≤{m['curriculum_maximum']}" if m['curriculum_maximum'] is not None else '전체'
        self.metrics_label.setText(f"Epoch {m['epoch']} · ε={m['epsilon']:.3f} | 학습 greedy 해결 {m['train_solved']}/{m['train_evaluated']} ({m['train_success_rate']:.1%})"
                            f" | 검증 greedy 해결 {m['solved']}/{m['evaluated']} ({validation_rate})"
                            f" | 탐험 평균 보상 {m['episode_reward']:.3f} | RL loss {m['loss']:.4f}"
                            f" | 전체 평가 성공 {m['display_solved']}/{m['display_count']} | curriculum {curriculum}"
                            f" | replay {m['replay_episodes']}회/{m['replay_buffer_size']}개 | 학습 episodes {m['episodes']}")

    def result(self, payload):
        mode, result = payload
        if mode == 'import':
            self.accuracy_plot.clear(); self.success_plot.clear(); self.efficiency_plot.clear()
            self.puzzles, digest = result
            self.monitor_cache.clear()
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
        elif mode in ('infer', 'test', 'all'):
            name = {'test': '테스트', 'infer': '현재 페이지', 'all': '전체 문제'}[mode]
            efficiency = f"{result['efficiency_mean']:.3f}" if result['efficiency_mean'] is not None else '—'
            self.metrics_label.setText(f"{name} greedy 해결 {result['solved']}/{result['count']} · Efficiency {efficiency}")
            self.info.setText(f'{name} 평가 완료 · 고정 모델 · 가중치 업데이트 없음')
        else:
            self.net = result; self.heat.set_network(self.net)
            self.info.setText(f'학습 종료 · best.npz / final.npz 자동 저장: {self.worker.artifact_dir}' if self.worker.diagnostic else '강화학습 종료 · 전체 분할 학습/검증 · 테스트 학습 미사용 · 차량 순서 증강 · 성공 replay · curriculum 적용')

    def save(self):
        path, _ = QFileDialog.getSaveFileName(self, '가중치 저장', f'checkpoints/{self.learning_mode}-model.npz', 'NumPy weights (*.npz)')
        if path:
            try: self.net.save(path)
            except Exception as exc: QMessageBox.critical(self, 'Save', str(exc))

    def load(self):
        path, _ = QFileDialog.getOpenFileName(self, '가중치 불러오기', '', 'NumPy weights (*.npz)')
        if path:
            try:
                candidate = load_network(path)
                if bool(getattr(candidate,'policy_only',False)) != (self.learning_mode=='supervised'):
                    raise ValueError('이 모델은 다른 학습 방식의 탭에서 불러오세요.')
                self.net = candidate; self.sync_model_view(); self.monitor_cache.clear(); self.show_page()
                self.accuracy_plot.clear(); self.success_plot.clear(); self.efficiency_plot.clear()
                self.info.setText('지도학습 모델 로드 완료 · 학습/추론 가능 · 이전 형식의 모델은 Adam 상태가 새로 시작됩니다.' if getattr(self.net, 'policy_only', False) else 'RL 가중치와 optimizer 로드 완료 · 다음 실행의 그래프/탐험 난수는 새로 시작합니다.')
            except Exception as exc: QMessageBox.critical(self, 'Load', str(exc))

    def reset(self):
        if QMessageBox.question(self, '초기화', '현재 가중치를 초기화할까요? 저장 파일은 유지됩니다.') == QMessageBox.StandardButton.Yes:
            self.net = PolicyNetwork((224,512,512,140),self.experiment_seed.value()) if self.learning_mode=='supervised' else ExpandedRL(seed=self.experiment_seed.value()); self.sync_model_view(); self.monitor_cache.clear(); self.show_page()
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
        scores = out[legal]; prob = np.exp(scores - scores.max()) if legal else np.array([])
        if legal: prob /= prob.sum()
        top = sorted(zip(legal, prob), key=lambda x: -x[1])[:5]
        lines = (['교사 학습 정책 · value head 없음'] if getattr(self.net, 'policy_only', False) else [f'Value: {out[140]:.3f} expected discounted reward']) + ['Top legal actions:']
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


class TabbedWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Rush Hour · Neural Array Lab v0.3.2')
        self.resize(1700,1080)
        self.tabs=QTabWidget();self.setCentralWidget(self.tabs)
        self.supervised=Window('supervised');self.rl=Window('rl')
        for panel,title in [(self.supervised,'지도학습'),(self.rl,'강화학습')]:
            panel.setWindowFlags(Qt.WindowType.Widget)
            self.tabs.addTab(panel,title)
        self.tabs.setCurrentIndex(0)

    def closeEvent(self,event):
        running=[p for p in [self.supervised,self.rl] if p.worker and p.worker.isRunning()]
        if running:
            for p in running:p.stop()
            event.ignore()
        else:event.accept()


def configure_style(app):
    app.setStyle('Fusion')
    app.setStyleSheet('''QWidget {background:#111b2b; color:#dce7f5; font-size:12px;}
        QPushButton,QComboBox,QSpinBox {background:#24364b; border:1px solid #3d566d; border-radius:5px; padding:6px;}
        QPushButton:hover {background:#355575;} QPushButton:disabled {color:#62738a;}
        QSlider::groove:horizontal {height:5px; background:#334259; border-radius:2px;}
        QSlider::handle:horizontal {width:15px; margin:-5px 0; background:#76d9cb; border-radius:5px;}
        QSlider::sub-page:horizontal {background:#438b85; border-radius:2px;}
        QTabBar::tab {padding:10px 35px;background:#1b2a3e;} QTabBar::tab:selected {background:#247766;color:white;}
        QProgressBar {border:1px solid #3d566d;border-radius:4px;text-align:center;} QProgressBar::chunk {background:#247766;}
        QScrollArea {border:none;}''')


def main():
    app = QApplication.instance() or QApplication([])
    configure_style(app)
    window = TabbedWindow(); window.show()
    app.exec()
