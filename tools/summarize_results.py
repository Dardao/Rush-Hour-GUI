"""Build the research summary from completed independent evaluations."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def main():
    groups=[('기존 v0.3.3','evaluation/baseline-minimum-comparison.json'),('할인 방식만 변경 · 20 epoch','evaluation/square-only-20/minimum-comparison.json'),('행동 단위 할인 · 추가 경험 · 60 epoch','evaluation/action-100k-60/minimum-comparison.json'),('칸 단위 할인 · 추가 경험 · 60 epoch','evaluation/square-100k-60/minimum-comparison.json'),('학습률 조정 + 100k 탐험','evaluation/square-100k-best/minimum-comparison.json'),('1m 탐험 경험 추가','evaluation/square-1m-best/minimum-comparison.json'),('최종 추가 탐험·학습','evaluation/final/minimum-comparison.json')]
    comparisons=[dict(condition=label,**read(p)['summary']) for label,p in groups]
    final=read('evaluation/final/minimum-comparison.json');neural=read('evaluation/final/neural-evaluation.json');stage=read('experiments/square-final-60/summary.json')
    (ROOT/'evaluation/experiment-comparison.json').write_text(json.dumps(comparisons,indent=2,ensure_ascii=False))
    lines=['# Minimum possible 검증 결과','',f"최종 저장 모델: **{final['summary']['solved']}/2500개 해결**, **{final['summary']['minimum_match']}/2500개가 앱 최소값과 일치**. 마지막 빨간차 탈출 비용까지 포함한 칸 수를 비교했습니다.",'','| 조건 | 해결 | 최소값 일치 | 해결 문제 평균 칸 수 | 전체 평균 Efficiency |','|---|---:|---:|---:|---:|']
    for x in comparisons:lines.append(f"| {x['condition']} | {x['solved']}/2500 | {x['minimum_match']}/2500 | {x['mean_solved_moves']:.4f} | {x['efficiency']*100:.4f}% |")
    lines+=['','실패한 문제는 Efficiency를 0으로 계산합니다. 일부만 해결한 조건의 평균 이동 수는 해결된 문제만의 평균이므로, 전체 해결 조건보다 작더라도 더 우수하다는 뜻이 아닙니다. 대조 실험의 학습률은 0.0002이며, 후속 100k 실험은 0.001을 사용했습니다.','', '## 최종 모드별 결과','','| 모드 | 해결 | 최소값 일치 | 평균 이동 수 |','|---|---:|---:|---:|']
    for mode,x in final['by_mode'].items():lines.append(f"| {mode} | {x['solved']}/625 | {x['minimum_match']}/625 | {x['mean_solved_moves']:.4f} |")
    lines+=['','## 해석','','- 기존 모델도 할인율 0.99를 사용했습니다. 이번에는 행동 단위 할인을 이동 칸 단위 할인으로 맞췄습니다.','- 할인 방식만 바꿔 20에포크를 추가한 실험은 최소값 일치 311개에서 개선되지 않았습니다. 새로운 짧은 경로 경험을 발견하고 재학습하는 과정이 필요했습니다.','- 신경망 구조는 224 → 512 → 512 → 141로 유지했습니다. 최소값 표와 외부 정답 행동 경로를 학습에 주지 않았습니다.','- 탐험은 방문 상태 보관·복원과 실제 관찰한 더 저렴한 경로 보관을 포함하며, 학습은 AWR 방식의 자기 경험 재학습입니다. 일반적인 epsilon-greedy만으로 달성했다는 주장은 아닙니다.','- 2500개 모두 학습 대상입니다. 새 차량 구성에 대한 일반화 결과가 아닙니다.','- 앱에 기록된 최소값과의 일치 여부를 검증했습니다. 독립적인 전수 최단 경로 탐색으로 최적성을 다시 증명한 것은 아닙니다.','', '## 검증 및 재현','',f"모델 SHA-256: `{neural['model_sha256']}`",'',f"독립 평가: 가중치 고정 `{neural['weights_frozen']}`, 경험 파일 접근 `{neural['experience_files_available']}`, 최단 경로 oracle 접근 `{neural['oracle_available']}`.",'',f"선택된 이어 학습 경로: 기존 v0.3.3 → 100k 경험 학습의 45 epoch 체크포인트 → 1m 경험 학습의 30 epoch 체크포인트 → 최종 {stage['epochs_completed']} epoch. 이 과정 외의 비교 실험은 최종 모델의 학습 계보에 포함되지 않습니다.",'','전체 문제별 비교는 `minimum-report.html`과 `evaluation/final/minimum-comparison.json`에 있습니다. `tools/evaluate_checkpoint.py`로 독립 평가를 다시 실행할 수 있습니다.']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
