import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def read_events(path):
    if not Path(path).exists():return []
    data=Path(path).read_bytes();last=data.rfind(b'\n')+1
    return [json.loads(line) for line in data[:last].splitlines()]


class Events:
    def __init__(self, directory, enabled=True):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.path=self.directory/'events.jsonl';self.enabled=enabled;self.overhead_seconds=0.
        if self.path.exists():
            data=self.path.read_bytes();last=data.rfind(b'\n')+1
            if last<len(data):
                self.path.with_name('events.partial.'+str(time.time_ns())).write_bytes(data[last:]);self.path.write_bytes(data[:last])
        previous=read_events(self.path);self.sequence=previous[-1]['sequence'] if previous else 0
    def emit(self, event_type, **fields):
        if not self.enabled:return
        start=time.perf_counter();self.sequence+=1
        record={'sequence':self.sequence,'timestamp_utc':datetime.now(timezone.utc).isoformat(),'event':event_type,**fields}
        with self.path.open('a') as f:f.write(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
        self.overhead_seconds+=time.perf_counter()-start


def replay(events):
    states={}
    for event in events:
        tid=event.get('trajectory_id')
        if not tid:continue
        if event['event']=='trajectory_initial':states[tid]=list(event['token_ids'])
        elif 'changes' in event:
            if tid not in states:raise ValueError('Delta without trajectory initial state')
            for change in event['changes']:
                if states[tid][change['position']]!=change['old_id']:raise ValueError('Replay old-token mismatch')
                states[tid][change['position']]=change['new_id']
            if 'token_ids' in event and states[tid]!=event['token_ids']:raise ValueError('Replay snapshot mismatch')
        if event.get('tokens_sha256') and tid in states:
            digest=hashlib.sha256(json.dumps(states[tid],separators=(',',':')).encode()).hexdigest()
            if digest!=event['tokens_sha256']:raise ValueError('Replay token digest mismatch')
    return states
