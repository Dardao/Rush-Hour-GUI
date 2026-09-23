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

### 지도학습 Seed 범위 반복

지도학습 탭에서 **반복 학습 Seed 시작 / 종료 (포함)**를 설정하고
**Seed 범위 새 지도학습**을 누릅니다. 기본값은 1–30입니다. 기존 단일 seed
학습/이어학습 버튼도 사용할 수 있습니다. 선택한 학습 범위와 Epochs는 모든
seed에 동일하게 적용되고, 매 seed의 모델 가중치와 Adam 상태는 새로 초기화됩니다.
데이터 분할은 seed에 따라 변경되지 않습니다. 현재 모델은 시작 전에 백업됩니다.

세 그래프는 accuracy(학습 문제 해결률 및 교사 행동 정확도), loss,
solving efficiency입니다. 검증 집합이 있으면 검증 해결률과 efficiency도 표시합니다.
연한 선은 개별 seed, 진한 실선은 같은 epoch에 기록된 seed들의 산술평균입니다.
진행 중에는 epoch별 참여 seed 수가 다를 수 있으며, 마지막 epoch의 참여 수가
각 패널에 표시됩니다. **그래프 PNG 저장**으로 현재 세 패널을 저장합니다.

각 seed의 기존 `runs/supervised-*.jsonl` 및 `checkpoints/supervised-*/`는
별도로 저장됩니다. `runs/supervised-sweep-*/batch.json`에는 seed별 파일 경로와
완료/중단 상태가, `mean.csv`에는 epoch별 평균과 `n_seeds`가 저장됩니다.
집계 파일은 각 seed가 종료될 때 갱신됩니다. 정지는 현재 seed를 중단하고 다음
seed를 실행하지 않습니다. 완료된 epoch만 평균에 포함하며 누락값을 0으로 채우지 않습니다.
전체 2,500문제 학습 설정에서는 검증 지표가 비어 있습니다.
