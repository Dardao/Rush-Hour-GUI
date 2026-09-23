"""Create a standalone, searchable 100-problem-per-page result viewer."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TEMPLATE=r'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rush Hour · Minimum possible 검증</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#0e1928;color:#e2edf6}body{margin:32px auto;max-width:1180px;padding:0 24px}h1{color:#63dfc4;font-size:27px}p{line-height:1.65;color:#abbcd1}section{display:flex;gap:16px;flex-wrap:wrap;margin:25px 0}article{background:#1a2c40;border:1px solid #30465d;border-radius:12px;padding:20px;flex:1;min-width:200px}b{font-size:29px;display:block;color:#63dfc4;margin:8px 0}label{margin:10px 20px 10px 0;display:inline-block}select,input,button{background:#1a2c40;color:#edf5fc;padding:9px;border:1px solid #405d77;border-radius:5px}input[type=range]{padding:0;width:250px}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}th,td{text-align:right;padding:10px;border-bottom:1px solid #294159}th{color:#a8c2dd}th:first-child,td:first-child{text-align:left}.good{color:#63dfc4}.bad{color:#ffc773}small{color:#a6b9cd}footer{margin-top:28px;color:#90a6bb;font-size:13px}
</style><h1>Rush Hour · Minimum possible 검증</h1>
<p>저장된 신경망의 greedy 풀이를 앱에 기록된 최소 이동 수와 비교했습니다. 이동 수에는 마지막 빨간차 탈출 비용이 포함됩니다. 평가 중에는 정답 경로·탐색·학습을 사용하지 않습니다.</p>
<section id="summary"></section>
<label>모드 <select id="mode"><option value="">전체</option><option>Easy</option><option>Medium</option><option>Hard</option><option>Expert</option></select></label>
<label>결과 <select id="result"><option value="all">전체</option><option value="match">최소값 일치</option><option value="gap">최소값 초과</option><option value="fail">미해결</option></select></label>
<label>문제 찾기 <input id="search" placeholder="예: Easy001"></label>
<div><button id="prev">이전 100개</button> <input id="page" type="range" min="0" value="0"> <button id="next">다음 100개</button> <small id="range"></small></div>
<table><thead><tr><th>문제</th><th>Minimum possible</th><th>기존 moves</th><th>추가 학습 moves</th><th>초과 칸</th><th>Efficiency</th><th>할인 보상</th><th>평가 결과</th></tr></thead><tbody id="rows"></tbody></table>
<footer>γ = 0.99 / 칸 · 성공 +1 · 칸당 −0.01 · 모든 2500개는 학습 대상입니다. 새 차량 구성에 대한 일반화 평가가 아닙니다.<br>최소값은 평가에만 사용했습니다. 학습은 상태 보관·복원 탐험과 자기 경험 재학습을 포함합니다.</footer>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.querySelector('#data').textContent),S=D.final.summary,B=D.baseline.summary;
const old=new Map(D.baseline.rows.map(x=>[x.name,x]));const $=s=>document.getElementById(s);
$('summary').innerHTML=`<article>해결<b>${S.solved} / 2500</b><small>기존 ${B.solved} / 2500</small></article><article>앱 최소값 일치<b>${S.minimum_match} / 2500</b><small>기존 ${B.minimum_match} / 2500</small></article><article>평균 이동 수 · 해결 문제<b>${S.mean_solved_moves.toFixed(2)}</b><small>기존 ${B.mean_solved_moves.toFixed(2)} · 앱 최소 평균 ${S.mean_minimum.toFixed(2)}</small></article><article>평균 Efficiency<b>${(100*S.efficiency).toFixed(2)}%</b><small>기존 ${(100*B.efficiency).toFixed(2)}% · 미해결은 0</small></article>`;
function render(reset=false){if(reset)$('page').value=0;const filter=$('result').value;
const data=D.final.rows.filter(x=>x.name.startsWith($('mode').value)&&x.name.toLowerCase().includes($('search').value.toLowerCase())&&(filter==='all'||(filter==='match'&&x.minimum_match)||(filter==='gap'&&x.gap>0)||(filter==='fail'&&x.status!=='solved')));
$('page').max=Math.max(0,Math.ceil(data.length/100)-1);const page=Math.min(Number($('page').value),Number($('page').max)),start=page*100,end=Math.min(start+100,data.length);$('page').value=page;$('range').textContent=data.length?`${start+1}–${end} / ${data.length}개`:'해당 문제 없음';
$('rows').replaceChildren(...data.slice(start,end).map(x=>{const tr=document.createElement('tr');const values=[x.name,x.minimum_possible,old.get(x.name).moves,x.moves,x.gap??'—',(100*x.efficiency).toFixed(1)+'%',x.discounted_return?.toFixed(4)??'—',x.minimum_match?'최소값 일치':x.status==='solved'?'해결 · 초과':'미해결'];values.forEach((v,i)=>{const td=document.createElement('td');td.textContent=v;if(i===7)td.className=x.minimum_match?'good':'bad';tr.append(td)});return tr}));$('prev').disabled=page===0;$('next').disabled=page===Number($('page').max)}
for(const k of ['mode','result','search'])$(k).addEventListener('input',()=>render(true));$('page').addEventListener('input',()=>render());$('prev').onclick=()=>{$('page').value=Number($('page').value)-1;render()};$('next').onclick=()=>{$('page').value=Number($('page').value)+1;render()};render();
</script></html>'''
def main():
    a=argparse.ArgumentParser();a.add_argument('--comparison',required=True);a.add_argument('--out',required=True);args=a.parse_args()
    data={'baseline':json.loads((ROOT/'evaluation/baseline-minimum-comparison.json').read_text()),'final':json.loads(Path(args.comparison).read_text())}
    for group in data.values():
        for r in group['rows']:r.pop('path',None)
    Path(args.out).write_text(TEMPLATE.replace('__DATA__',json.dumps(data,separators=(',',':')).replace('<','\\u003c')))
if __name__=='__main__':main()
