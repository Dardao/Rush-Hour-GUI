"""Board animation reused from v0.3.2; monitor only, no training input."""
import time
from PySide6.QtCore import Qt,QRectF,QTimer,Signal
from PySide6.QtGui import QColor,QPainter,QFont
from PySide6.QtWidgets import QWidget
from .core import split_role,display_moves
from .rl import replay_rewards,SUCCESS_REWARD,STEP_REWARD

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
        self.all_trained = True
        self.role_overrides = None
        self.action_sources = []
        self.last_rewards, self.total_rewards = [], []
        self.fit = True
        self.exit_started = {}
        self.exit_timer = QTimer(self)
        self.exit_timer.setInterval(30)
        self.exit_timer.timeout.connect(self.animate_exits)
        self.setToolTip("moves: 이동 칸 수 + 해결 시 자동 탈출 비용. 실제 신경망 평가 경로를 재생합니다.")
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
            
            minimum = f'Minimum possible: {p.minimum_possible if p.minimum_possible is not None else "—"}'
            q.setPen(QColor('#aebed2' if self.statuses[i] == 'ready' else STATUS[self.statuses[i]]))
            if self.fit:
                role = {'train': '학습', 'validation': '검증', 'test': '테스트', 'unused':'미사용'}.get(self.roles[i], '') if i < len(self.roles) else ''
                if self.action_sources[i]=='teacher':role += ' · 교사 상태'
                if self.action_sources[i] in ('random','greedy'):role += ' · '+('탐험' if self.action_sources[i]=='random' else 'greedy')
                info = (f'{p.name} '+(f'[{role}]' if role else '')
                        + f'\n{used} moves · {self.statuses[i]}\n{reward_info}')
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
                           f'{used} moves · {self.statuses[i]}\n{reward_info}')



