# Rush Hour RL Minimum Study v0.3.4

이 패키지는 기존 v0.3.3 강화학습 모델을 이어 학습하고, 실제 신경망의 풀이가 앱의 `Minimum possible`에 도달하는지 검증하는 실험입니다.

- 결과와 한계: `RESULTS.md`
- 보상식·대조 실험·재현 방법: `METHODOLOGY.md`
- 전체 문제 비교: `minimum-report.html` (브라우저에서 열기, 모드/결과/이름 검색, 100개 단위 페이지)
- 모델의 풀이 경로와 문제별 수치: `evaluation/`
- 학습 모델: `models/`

Python 3.10 이상을 사용하세요. 모델 평가와 학습에는 NumPy만 필요합니다. 탐험을 새로 실행하려면 C++17을 지원하는 g++가 필요합니다. 이 폴더는 연구 패키지이며 기존 GUI의 대체 설치본이 아닙니다.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python tools/evaluate_checkpoint.py --model models/rl-minimum-best.npz --out evaluation/my-recheck
```

출력 폴더는 새 이름을 사용해야 합니다. 평가 프로그램은 모델과 초기 맵만 있는 별도 임시 폴더에서 실행됩니다. 평가가 끝난 후에만 최소값 표와 비교합니다. 원래 시작점에서 합법 행동 중 신경망 점수가 가장 큰 행동을 선택하며, 반복 상태 또는 300번 행동에 도달하면 실패 처리합니다. 마지막 빨간차 탈출 비용은 이동 수에 포함됩니다.

`models/rl-minimum-best.npz`는 기존 확장형 RL 체크포인트 형식을 유지합니다. 기존 GUI의 **강화학습 탭에서 모델 불러오기**로 사용할 수 있습니다. GUI의 누적 보상 표시와 `Maximum reward`는 기존의 비할인 보상 기준이며, 이번 연구의 할인 학습 목표는 별도로 계산합니다.

학습에는 정답 행동 경로, APK 최소값, 지도학습 가중치를 주지 않았습니다. 탐험 과정에서는 방문 상태 보관·복원과 실제 관찰한 경로의 비용 개선을 사용하며, 자신이 발견한 성공 경로를 AWR 방식으로 다시 학습합니다. 일반적인 epsilon-greedy만 사용한 실험은 아닙니다. 2500개 모두 학습에 사용하므로 새로운 차량 구성에 대한 일반화 시험도 아닙니다.
