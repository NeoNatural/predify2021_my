"""Sequential, resumable coordinator; abort on failures rather than skip candidates."""
import sys,json,subprocess,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.conv_development_20261104 import candidates,OUT
D=ROOT/'docs/conv_development_20261104';D.mkdir(exist_ok=True);status=D/'run_status.json'
def save(**x):status.write_text(json.dumps(x,indent=2),encoding='utf8')
def run(args,ds,phase,candidate=None):
 storage_config=D/'storage_settings.json'
 if storage_config.exists():
  expected=Path(json.loads(storage_config.read_text(encoding='utf8'))['development_root'])
  if not expected.is_dir() or OUT.resolve()!=expected.resolve():
   save(stage='external_storage_unavailable',complete=False,dataset=ds,candidate=candidate);raise SystemExit('External SSD unavailable or output mapping changed; refusing fallback to C drive')
 storage_path=OUT.resolve() if OUT.exists() else ROOT
 if shutil.disk_usage(storage_path).free < 12*1024**3:
  save(stage='blocked_disk_space',complete=False,dataset=ds,candidate=candidate);raise SystemExit('Less than 12 GiB free; stopped before next stage')
 save(stage=phase,dataset=ds,candidate=candidate,updated=time.time(),complete=False)
 for attempt in range(2):
  with (D/'run.log').open('a',encoding='utf8') as f:
   f.write(f'\nSTAGE {phase} dataset={ds} candidate={candidate} attempt={attempt+1} time={time.time()}\n');f.flush()
   result=subprocess.run([sys.executable,'-X','faulthandler','-u',*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  if result.returncode==0:break
  code=result.returncode & 0xffffffff
  if code==0xc0000409 and attempt==0:
   event=dict(stage=phase,dataset=ds,candidate=candidate,returncode=result.returncode,hex_code=hex(code),attempt=attempt+1,time=time.time())
   with (D/'native_crash_retries.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(event)+'\n')
   save(stage='retry_native_crash',dataset=ds,candidate=candidate,complete=False,**{'attempt':attempt+1})
   time.sleep(3)
   continue
  save(stage='failed',dataset=ds,candidate=candidate,returncode=result.returncode,hex_code=hex(code),attempt=attempt+1,complete=False);raise SystemExit(result.returncode)
for ds in ('imagenet','things'):
 if not (OUT/ds/'diagnose/complete.json').exists():run(['analysis/conv_development_20261104.py','diagnose','--dataset',ds],ds,'diagnosis')
 for cfg in candidates():
  cid=cfg['id'];folder=OUT/ds/cid
  if not (folder/'complete.json').exists():run(['analysis/conv_development_20261104.py','screen','--dataset',ds,'--candidate',cid],ds,'screen',cid)
  if not (folder/'priming_contrasts.csv').exists():run(['analysis/evaluate_conv_development_20261104.py','--dataset',ds,'--candidate',cid],ds,'evaluation',cid)
run(['analysis/rank_conv_development_20261104.py'],'both','ranking')
save(stage='complete',complete=True,updated=time.time())
