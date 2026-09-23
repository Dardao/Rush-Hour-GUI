"""Three-panel SL plots: faint independent runs, bold epoch means."""
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QColor, QPainter, QPen, QPainterPath, QFont
from PySide6.QtWidgets import QWidget
from .seed_sweep import epoch_means


class SLPlot(QWidget):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.histories = {}
        self.setMinimumSize(240, 165)
        self.setMaximumHeight(190)
        self.setToolTip('연한 선: 개별 seed · 진한 실선: 같은 epoch의 산술평균.\n진행 중에는 해당 epoch까지 기록된 seed만 평균에 참여합니다.\n행동 정확도와 loss는 학습 batch 평균, 해결률과 efficiency는 epoch 후 greedy 평가입니다.')

    def clear(self):
        self.histories.clear(); self.update()

    def append(self, metrics):
        self.histories.setdefault(metrics.get('seed', 0), []).append({k: metrics.get(k) for k in ('epoch', 'train_success_rate', 'validation_success_rate', 'action_accuracy', 'loss', 'train_efficiency_mean', 'validation_efficiency_mean')})
        self.update()

    def paintEvent(self, event):
        q = QPainter(self); q.setRenderHint(QPainter.RenderHint.Antialiasing)
        q.fillRect(self.rect(), QColor('#ffffff'))
        q.setFont(QFont('Arial', 9)); q.setPen(QColor('#172033'))
        title, series = {
            'accuracy': ('SL TRAIN ACCURACY (%)', [('train_success_rate', 'Solve', '#2563eb'), ('action_accuracy', 'Action', '#64748b'), ('validation_success_rate', 'Val solve', '#a855f7')]),
            'loss': ('TRAINING LOSS', [('loss', 'Loss', '#dd7622')]),
            'efficiency': ('SOLVING EFFICIENCY', [('train_efficiency_mean', 'Train', '#059669'), ('validation_efficiency_mean', 'Val', '#a855f7')]),
        }[self.kind]
        means = epoch_means(self.histories)
        series = [(k,n,c) for k,n,c in series if not means or any(r[k] is not None for r in means)]
        q.drawText(QRectF(8, 2, self.width()-16, 20), title)
        q.setFont(QFont('Arial', 8))
        # Compact color legend with an explicit number of contributing seeds.
        x = 8
        for key, name, color in series:
            q.setPen(QColor(color)); q.drawText(QRectF(x, 23, 80, 16), name)
            x += q.fontMetrics().horizontalAdvance(name) + 12
        q.setPen(QColor('#64748b'))
        n = means[-1]['n_seeds'] if means else 0
        q.drawText(QRectF(8, 40, self.width()-16, 16), f'Bold: mean | faint: seeds | n(last epoch)={n}')
        area = QRectF(40, 64, self.width()-53, self.height()-94)
        values = [r[k] for rows in self.histories.values() for r in rows for k,_,_ in series if r.get(k) is not None]
        high = max(1, max(values, default=1)*1.05) if self.kind != 'accuracy' else 1
        xmax = max(2, max((r['epoch'] for r in means), default=2))
        for v in (0, high/2, high):
            y = area.bottom()-area.height()*v/high
            q.setPen(QColor('#e2e8f0')); q.drawLine(QPointF(area.left(),y), QPointF(area.right(),y))
            q.setPen(QColor('#64748b'))
            q.drawText(QRectF(0,y-8,35,16), Qt.AlignmentFlag.AlignRight, f'{v*100:.0f}' if self.kind=='accuracy' else f'{v:.2g}')
        q.drawText(QRectF(area.left(),area.bottom()+3,25,17), '1')
        q.drawText(QRectF(area.right()-30,area.bottom()+3,30,17), Qt.AlignmentFlag.AlignRight,str(xmax))
        q.drawText(QRectF(area.left(),area.bottom()+3,area.width(),17),Qt.AlignmentFlag.AlignCenter,'Epoch')
        def draw(rows, key, color, alpha, width):
            path = QPainterPath(); started = False
            q.setBrush(Qt.BrushStyle.NoBrush)
            for row in rows:
                v = row.get(key)
                if v is None: started=False; continue
                point=QPointF(area.left()+area.width()*(row['epoch']-1)/(xmax-1),area.bottom()-area.height()*v/high)
                if started: path.lineTo(point)
                else: path.moveTo(point)
                started=True
            c=QColor(color); c.setAlphaF(alpha); q.setPen(QPen(c,width)); q.drawPath(path)
            if len(rows)==1 and rows[0].get(key) is not None:
                q.drawEllipse(point,1.5,1.5)
        for key,_,color in series:
            for rows in self.histories.values(): draw(rows,key,color,.18,1)
        for key,_,color in series: draw(means,key,color,1,2.4)
