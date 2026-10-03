"""Offline verification record; never invoke a paid model or print env secrets."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    env={**os.environ,'PYTHONUTF8':'1'}
    files=[*ROOT.glob('*.py'),*(ROOT/'benchmark').glob('*.py'),*(ROOT/'benchmark/experiments').glob('*.py'),*(ROOT/'tests').glob('*.py')]
    commands=[
        [sys.executable,'-X','utf8','-m','py_compile',*map(str,files)],
        [sys.executable,'-X','utf8','-m','unittest','discover','-s','tests','-v'],
        ['docker','compose','-f','compose.local.yaml','config','--quiet'],
        ['git','check-ignore','benchmark/experiments/private/audit_verified/gold_annotation.csv','benchmark/experiments/answers_v0_naive.jsonl','benchmark/experiments/mapping.private.json'],
    ]
    result={'created_at':datetime.now(timezone.utc).isoformat(),'python':sys.version,'checks':[],
            'live_api_smoke':'NOT RUN: automatic approval denied external paper transfer and paid calls',
            'docker_build':'NOT RUN: engine unavailable, firmware virtualization disabled',
            'synthetic_notice':'Fake model usage 11/7 in test output is a fixture, not API usage or benchmark evidence'}
    for command in commands:
        run=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120)
        result['checks'].append({'command':command,'exit_code':run.returncode,'stdout':run.stdout,'stderr':run.stderr})
    import database
    from fastapi.testclient import TestClient
    from app import app
    with tempfile.TemporaryDirectory(prefix='research-startup-') as temp:
        with patch.object(database,'DB_PATH',str(Path(temp)/'startup.db')):
            with TestClient(app) as client:
                statuses={path:client.get(path).status_code for path in ('/','/openapi.json','/history')}
    result['isolated_http_smoke']=statuses
    result['passed']=all(c['exit_code']==0 for c in result['checks']) and all(code==200 for code in statuses.values())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({'passed':result['passed'],'http':statuses,'output':str(args.output)}))
    if not result['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
