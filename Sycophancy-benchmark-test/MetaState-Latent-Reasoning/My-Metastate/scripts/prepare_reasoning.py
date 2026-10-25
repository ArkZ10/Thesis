"""CPU-only synthetic reasoning pilot: build or validate. No model training.
Run with .env/bin/python scripts/prepare_reasoning.py {build,validate}.
"""
import argparse
import ast
from collections import Counter
import hashlib
import importlib.metadata
import json
import operator
import os
from pathlib import Path
import random
import tempfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
from prepare_sciq import encode, tokenizer, write_json, write_lines, read_lines, require, sha

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/reasoning-pilot'
SEED = 42
SIZES = {'train': 256, 'development': 64, 'test': 64}
FAMILIES = ('packs_left', 'shared_remainder', 'two_deliveries', 'rows_changed')
INSTRUCTION = 'Give only the final answer as an integer.'


def evaluate(expression):
    """Independent, restricted arithmetic evaluator; never executes source text."""
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.FloorDiv: operator.floordiv}
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        require(isinstance(node, ast.BinOp) and type(node.op) in ops, 'Invalid expression')
        a, b = visit(node.left), visit(node.right)
        if isinstance(node.op, ast.FloorDiv):
            require(b > 0 and a % b == 0, 'Division must be exact')
        return ops[type(node.op)](a, b)
    return visit(ast.parse(expression, mode='eval').body)


def problem(family, rng):
    a, b = sorted((rng.randint(3, 19), rng.randint(3, 19)))
    if family == 'packs_left':
        c = rng.randint(1, a*b-1)
        question = f'A store has {a} packs of pencils with {b} pencils in each pack. It sells {c} pencils. How many pencils remain?'
        stages = [sum(b for _ in range(a)), sum(b for _ in range(a))-c]
        expressions = [f'{a}*{b}', f'({a}*{b})-{c}']
        params = [a,b,c]
    elif family == 'shared_remainder':
        c = rng.randint(2, 9)
        total = a*b
        extra = rng.randint(1, total//c)
        removed = total - extra*c
        question = f'There are {a} boxes containing {b} beads each. After {removed} beads are removed, the remaining beads are shared equally among {c} children. How many beads does each child receive?'
        stages = [sum(b for _ in range(a)), total-removed, len(list(range(total-removed))[::c])]
        expressions = [f'{a}*{b}', f'({a}*{b})-{removed}', f'(({a}*{b})-{removed})//{c}']
        params = [a,b,removed,c]
    elif family == 'two_deliveries':
        c,d = sorted((rng.randint(3,19), rng.randint(3,19)))
        # Canonicalize delivery order as well as commutative multiplication.
        (a,b),(c,d) = sorted(((a,b),(c,d)))
        question = f'A shop receives {a} cartons with {b} bottles each, then {c} cartons with {d} bottles each. How many bottles arrive in total?'
        stages = [sum(b for _ in range(a)),sum(d for _ in range(c)),sum([b]*a+[d]*c)]
        expressions = [f'{a}*{b}', f'{c}*{d}', f'({a}*{b})+({c}*{d})']
        params = [a,b,c,d]
    else:
        c = rng.randint(1,a-1)
        d = rng.randint(1,9)
        question = f'A hall has {a} rows with {b} chairs each. Workers remove {c} entire rows and add {d} individual chairs. How many chairs are in the hall now?'
        stages = [sum(b for _ in range(a)),sum(b for _ in range(c)),sum([b]*(a-c)+[1]*d)]
        expressions = [f'{a}*{b}', f'{c}*{b}', f'({a}*{b})-({c}*{b})+{d}']
        params = [a,b,c,d]
    key = json.dumps([family,params],separators=(',',':'))
    qid = 'arithmetic-' + hashlib.sha256(key.encode()).hexdigest()[:16]
    return {'id':qid+'-neutral','question_id':qid,'problem_key':key,'family':family,
            'parameters':params,'question':question,'condition':'neutral',
            'source_dataset':'local-synthetic-arithmetic-v1',
            'messages':[{'role':'user','content':question+'\n\n'+INSTRUCTION}],
            'target_answer':str(stages[-1]),
            'verification_steps':[{'expression':e,'value':v} for e,v in zip(expressions,stages)]}


def generate():
    rng = random.Random(SEED)
    pools = {s:[] for s in SIZES}
    seen = set()
    for family in FAMILIES:
        unique = []
        while len(unique) < sum(SIZES.values())//len(FAMILIES):
            row = problem(family,rng)
            if row['problem_key'] in seen:
                continue
            seen.add(row['problem_key']); unique.append(row)
        rng.shuffle(unique)
        offset = 0
        for split,size in SIZES.items():
            for row in unique[offset:offset+size//4]:
                pools[split].append(dict(row,split=split))
            offset += size//4
    for rows in pools.values():
        rng.shuffle(rows)
    return pools


def materialize(path,t):
    from datasets import Dataset, DatasetDict
    pools = generate()
    training = {}
    for split,rows in pools.items():
        write_lines(path/'processed'/f'{split}.jsonl',rows)
        arrays = [encode(t,r['messages'][0]['content'],r['target_answer']) for r in rows]
        require(all(len(r['input_ids']) <= 256 for r in arrays),'Sequence exceeds 256 tokens; no truncation allowed')
        write_lines(path/'token_arrays'/f'{split}.jsonl',arrays)
        training[split] = Dataset.from_list(arrays)
    # Explicit split names: validation is development; test remains held out.
    DatasetDict({'train':training['train'],'validation':training['development'],
                 'test':training['test']}).save_to_disk(str(path/'tokenized'))
    for split in ('train','development'):
        subset = [next(r for r in pools[split] if r['family']==f) for f in FAMILIES]
        write_lines(path/'tiny'/f'{split}.jsonl',subset)
    return pools


def validate(path,t):
    from datasets import load_from_disk
    ids,keys,questions = set(),set(),set()
    dataset = load_from_disk(str(path/'tokenized'))
    lengths=[]
    for split,size in SIZES.items():
        rows = read_lines(path/'processed'/f'{split}.jsonl')
        arrays = read_lines(path/'token_arrays'/f'{split}.jsonl')
        require(len(rows)==len(arrays)==size, f'Wrong count for {split}')
        require(Counter(r['family'] for r in rows)=={f:size//4 for f in FAMILIES},'Unbalanced families')
        saved = dataset['validation' if split=='development' else split]
        require(len(saved)==size,'HF dataset count mismatch')
        for i,(row,arr) in enumerate(zip(rows,arrays)):
            for value,seen in [(row['id'],ids),(row['problem_key'],keys),(row['question'],questions)]:
                require(value not in seen,'Duplicate across splits'); seen.add(value)
            require(row['split']==split and row['condition']=='neutral','Invalid split/condition')
            for step in row['verification_steps']:
                require(evaluate(step['expression'])==step['value'],'Arithmetic check failed')
            require(row['target_answer']==str(row['verification_steps'][-1]['value']),'Target mismatch')
            require(row['messages']==[{'role':'user','content':row['question']+'\n\n'+INSTRUCTION}], 'Metadata leaked into prompt')
            require(arr==encode(t,row['messages'][0]['content'],row['target_answer']), 'Tokenization/masking mismatch')
            require(dict(saved[i])==arr,'HF serialization mismatch')
            lengths.append(len(arr['input_ids']))
        tiny=path/'tiny'/f'{split}.jsonl'
        if tiny.exists():
            small=read_lines(tiny)
            require(len(small)==4 and all(r in rows for r in small),'Invalid tiny subset')
    with tempfile.TemporaryDirectory() as temp:
        rebuilt=Path(temp)
        materialize(rebuilt,t)
        for original in path.rglob('*.jsonl'):
            require(sha(original)==sha(rebuilt/original.relative_to(path)),f'Nonreproducible: {original}')
    return {'status':'passed','counts':SIZES,'min_sequence_tokens':min(lengths),
            'max_sequence_tokens':max(lengths),'unique_questions':len(questions),
            'independent_arithmetic_checks':True,'split_disjointness':True,
            'assistant_boundary_and_prompt_masking':True,'rebuild_jsonl_byte_identical':True,
            'hf_arrays_match_jsonl':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['build','validate'])
    args=parser.parse_args()
    t=tokenizer()
    if args.command=='build':
        require(not DEST.exists(),f'Refusing to overwrite existing data: {DEST}')
        materialize(DEST,t)
        write_json(DEST/'manifest.json',{'schema_version':1,'seed':SEED,'counts':SIZES,
            'source':'Deterministically generated locally; no external dataset',
            'script_sha256':sha(Path(__file__)),'tokenizer':json.loads((ROOT/'setup/backbone.json').read_text()),
            'packages':{p:importlib.metadata.version(p) for p in ('datasets','transformers','tokenizers')},
            'hf_split_mapping':{'train':'training','validation':'development','test':'held-out test; do not use for tuning'},
            'scope':'Neutral arithmetic pilot. Belief variants and TruthfulQA excluded.',
            'split_policy':'Canonical base problems deduplicated before stratified seeded splitting. Templates and operand ranges shared across splits; not a test of unseen task families.',
            'training_policy':'Only target_answer and end-of-turn supervised; verification_steps metadata never tokenized. Fixed masked internal updates require a new trainer; do not pass this unchanged to old phase3 trainer.',
            'files':{str(p.relative_to(DEST)):sha(p) for p in sorted(DEST.rglob('*.jsonl'))}})
    manifest=json.loads((DEST/'manifest.json').read_text())
    require(manifest['script_sha256']==sha(Path(__file__)),'Generator changed since build')
    for rel,digest in manifest['files'].items():
        require(sha(DEST/rel)==digest,f'File changed: {rel}')
    report=validate(DEST,t)
    write_json(DEST/'validation.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
