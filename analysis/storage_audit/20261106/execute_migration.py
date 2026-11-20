"""Verified relocation of the three approved experiments, preserving original paths."""
from pathlib import Path
import csv
import datetime
import hashlib
import json
import os
import shutil
import stat
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DESTROOT = Path('E:/Predify_my归档/predify2021_my/Hebbian-CNN/Log')
NAMES = ('native_categories_readouts_20261029', 'fc_population_threshold_20261028', 'hebb_fc_full_20261103')
STATE = HERE / 'migration_execution'


def now():
    return datetime.datetime.now().astimezone().isoformat()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def is_reparse(path):
    return bool(path.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def inventory(folder):
    result = {}
    def fail(error):
        raise error
    for base, dirs, files in os.walk(folder, followlinks=False, onerror=fail):
        for name in dirs + files:
            p = Path(base)/name
            if is_reparse(p):
                raise RuntimeError('Unexpected reparse point: '+str(p))
        for name in files:
            p = Path(base)/name
            s = p.stat()
            result[str(p.relative_to(folder))] = (s.st_size, s.st_mtime_ns)
    return result


def junction(link, target):
    def quote(s):
        return "'"+str(s).replace("'", "''")+"'"
    command = "$ErrorActionPreference='Stop'; New-Item -ItemType Junction -Path "+quote(link)+" -Target "+quote(target)+" | Out-Null"
    subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command], check=True, capture_output=True)


def source_hashes():
    found = {}
    for folder in ('analysis', 'predify_experiments', 'predify2021', 'configs', 'Hebbian-CNN'):
        # Log is handled separately, so never follow its external junctions here.
        for base, dirs, files in os.walk(ROOT/folder, followlinks=False):
            dirs[:] = [d for d in dirs if d not in ('Log', '__pycache__') and not is_reparse(Path(base)/d)]
            for n in files:
                p = Path(base)/n
                if p.suffix in ('.py', '.toml') and p.resolve() != Path(__file__).resolve():
                    found[str(p.relative_to(ROOT))] = sha(p)
    return found


def main():
    STATE.mkdir(exist_ok=True)
    DESTROOT.mkdir(parents=True, exist_ok=True)
    original = source_hashes()
    atomic_json(STATE/'original_source_hashes.json', original)
    total_bytes = 0
    started = time.monotonic()
    for name in NAMES:
        src = ROOT/'Hebbian-CNN/Log'/name
        dst = DESTROOT/name
        backup = src.with_name(src.name+'.migration_local_backup_20261106')
        manifest_path = STATE/(name+'.json')
        if is_reparse(src):
            assert src.resolve() == dst.resolve() and manifest_path.exists(), 'Unexpected existing junction'
            assert json.loads(manifest_path.read_text(encoding='utf-8'))['status'] == 'complete'
            continue
        assert src.resolve() == src.absolute() and src.parent.resolve() == (ROOT/'Hebbian-CNN/Log').resolve()
        assert dst.resolve().is_relative_to(DESTROOT.resolve()) and not backup.exists()
        if dst.exists():
            assert not is_reparse(dst), 'Archive destination is a reparse point'
        snapshot = inventory(src)
        assert snapshot, 'Empty source'
        amount = sum(v[0] for v in snapshot.values())
        assert shutil.disk_usage(dst.parent).free > amount + 1024**3, 'Insufficient E: disk space'
        dst.mkdir(exist_ok=True)
        existing = inventory(dst)
        assert set(existing) <= set(snapshot), 'Unexpected destination files'
        manifest = dict(status='copying', source=str(src), destination=str(dst),
                        local_backup=str(backup), started_at=now(), file_count=len(snapshot),
                        bytes=amount, files=[])
        atomic_json(manifest_path, manifest)
        last = time.monotonic(); copied = 0
        print(json.dumps(dict(stage='starting', experiment=name, files=len(snapshot), GiB=round(amount/2**30,3))), flush=True)
        for index, (rel, signature) in enumerate(sorted(snapshot.items()),1):
            a, b = src/rel, dst/rel
            assert a.resolve().is_relative_to(src.resolve()) and b.resolve().is_relative_to(dst.resolve())
            before = a.stat()
            assert (before.st_size,before.st_mtime_ns)==signature, 'Source changed before copy: '+rel
            b.parent.mkdir(parents=True,exist_ok=True)
            if b.exists():
                digest = sha(a)
                assert b.stat().st_size==signature[0] and sha(b)==digest, 'Archive conflict: '+str(b)
            else:
                temp = b.with_name(b.name+'.migration_partial')
                h=hashlib.sha256()
                with a.open('rb') as stream, temp.open('wb') as dest:
                    for block in iter(lambda:stream.read(8*1024**2),b''):
                        h.update(block);dest.write(block)
                    dest.flush();os.fsync(dest.fileno())
                digest=h.hexdigest()
                assert temp.stat().st_size==signature[0] and sha(temp)==digest, 'Copy verification failed: '+rel
                shutil.copystat(a,temp);temp.replace(b)
            after=a.stat()
            assert (after.st_size,after.st_mtime_ns)==signature, 'Source changed during copy: '+rel
            manifest['files'].append(dict(relative_path=rel,bytes=signature[0],mtime_ns=signature[1],sha256=digest))
            copied+=signature[0]
            if time.monotonic()-last>=20 or index==len(snapshot):
                manifest['verified_files']=index;atomic_json(manifest_path,manifest)
                print(json.dumps(dict(stage='verified_copy',experiment=name,done=index,total=len(snapshot),GiB=round(copied/2**30,3),elapsed_s=round(time.monotonic()-started))),flush=True)
                last=time.monotonic()
        assert inventory(src)==snapshot, 'Source tree changed; original retained'
        target_snapshot=inventory(dst)
        assert set(target_snapshot)==set(snapshot) and all(target_snapshot[k][0]==v[0] for k,v in snapshot.items())
        manifest.update(status='verified_before_switch',verified_at=now())
        atomic_json(manifest_path,manifest)
        archive_manifest=DESTROOT/'_migration_manifests_20261106'/(name+'.json')
        atomic_json(archive_manifest,manifest)
        # Rename the exact ordinary source directory locally. Keep it until the junction is working.
        assert not is_reparse(src) and src.resolve()==src.absolute()
        assert backup.parent.resolve()==(ROOT/'Hebbian-CNN/Log').resolve() and not backup.exists()
        src.rename(backup)
        try:
            junction(src,dst)
            assert is_reparse(src) and src.resolve()==dst.resolve(), 'Junction target mismatch'
            for row in (manifest['files'][0],manifest['files'][-1]):
                assert sha(src/row['relative_path'])==row['sha256']
        except BaseException:
            if src.exists() and is_reparse(src):os.rmdir(src)
            if not src.exists():backup.rename(src)
            raise
        manifest.update(status='mapped_source_pending_cleanup')
        atomic_json(manifest_path,manifest);atomic_json(archive_manifest,manifest)
        # Revalidate exact C: backup boundaries and unchanged contents before recursive cleanup.
        assert backup.resolve()==backup.absolute() and not is_reparse(backup)
        assert backup.parent.resolve()==(ROOT/'Hebbian-CNN/Log').resolve()
        assert inventory(backup)==snapshot and src.resolve()==dst.resolve()
        shutil.rmtree(backup)
        note = f'''# 已迁移到 E 盘（2026-11-06）

此实验完整数据位于 `{dst}`。
原路径 `{src}` 是指向该目录的 NTFS 目录联接，现有脚本可继续使用原路径，无需修改。
本次已逐文件核对 SHA-256，清单包含 {len(snapshot):,} 个原始文件、{amount:,} 字节。
本地清单：`{manifest_path}`。
E 盘清单：`{archive_manifest}`。

E 盘须在线。通过原路径新增、修改或删除文件，也会作用到 E 盘数据。
目录联接不是额外备份。空间扫描应识别 ReparsePoint，避免重复计算。
如需恢复到 C 盘：停止使用此实验的任务；仅移除目录联接本身（禁止递归删除其目标）；
按清单从 E 盘复制数据回原目录，逐文件核对 SHA-256；保留 E 盘副本直至复核通过。
'''
        (dst/'ARCHIVE_README.md').write_text(note,encoding='utf-8')
        (STATE/(name+'_ARCHIVE_README.md')).write_text(note,encoding='utf-8')
        manifest.update(status='complete',completed_at=now(),junction_verified=True,local_backup_removed=True)
        atomic_json(manifest_path,manifest);atomic_json(archive_manifest,manifest)
        total_bytes+=amount
        print(json.dumps(dict(stage='complete',experiment=name,files=len(snapshot),GiB=round(amount/2**30,3))),flush=True)
    assert all((ROOT/p).exists() and sha(ROOT/p)==h for p,h in original.items()), 'Existing source script changed'
    summary=dict(status='complete',experiments=list(NAMES),bytes_migrated_this_run=total_bytes,
                 GiB_migrated_this_run=total_bytes/2**30,original_source_hashes_unchanged=True,completed_at=now())
    atomic_json(STATE/'summary.json',summary)
    atomic_json(DESTROOT/'_migration_manifests_20261106/summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
