"""Read initial boards and recorded minima; this file contains no solutions."""
import json
from pathlib import Path
from .core import Puzzle,Vehicle

def load_initial_boards(path):
    data=json.loads(Path(path).read_text());puzzles=[]
    for row in data['puzzles']:
        if set(row)!={'name','difficulty','cars','start','minimum_possible'}:
            raise ValueError('Unsupported initial-board schema')
        p=Puzzle(row['name'],row['difficulty'],tuple(Vehicle(**c) for c in row['cars']),tuple(row['start']),
                 source='Audited APK initial board',minimum_possible=row['minimum_possible'],
                 minimum_source='user-confirmed screen' if row['name']=='Easy001' else 'Audited APK metadata')
        p.board(p.start);puzzles.append(p)
    if len(puzzles)!=2500 or len({p.name for p in puzzles})!=2500:
        raise ValueError('Expected 2500 distinct puzzle names')
    return puzzles,data['apk_sha256']
