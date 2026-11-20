"""Copy, SHA-256 verify, journal, then remove only the inventoried geometry files."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil
import time

ROOT = Path(__file__).resolve().parents[2]
AUDIT = Path(__file__).resolve().parent
REL = Path('Hebbian-CNN/Log/tripartite_filtered_cv_20261027')
DEST = Path('E:/Predify_my归档/predify2021_my')
MANIFEST = ROOT / REL / 'geometry_archive_manifest.json'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    inventory = json.loads((AUDIT / 'inventory.json').read_text(encoding='utf-8'))
    files = [f for f in inventory['files'] if
             Path(f['path']).parent in {REL / d / 'formal' for d in ('imagenet', 'things')}
             and Path(f['path']).name.startswith('geometry_') and f['path'].endswith('.npz')]
    assert len(files) == 3072, len(files)
    expected = {f['path'] for f in files}
    actual = {str(p.relative_to(ROOT)) for d in ('imagenet', 'things')
              for p in (ROOT / REL / d / 'formal').glob('geometry_*.npz')}
    journal = AUDIT / 'geometry_archive_journal.ndjson'
    previous = {}
    if journal.exists():
        for line in journal.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            previous[row['relative_path']] = row
    assert actual <= expected, 'Unexpected geometry files; stop for review'
    assert expected - actual <= previous.keys(), 'Missing files without verification journal'
    DEST.mkdir(parents=True, exist_ok=True)
    total = sum(f['bytes'] for f in files)
    needed = sum(f['bytes'] for f in files if not (DEST / f['path']).exists())
    assert shutil.disk_usage(DEST).free > needed + 1024**3, 'Insufficient archive disk space'
    sources = json.loads((ROOT / REL / 'freeze.json').read_text(encoding='utf-8'))['sources']
    source_hashes = {p: sha(ROOT / p) for p in sources}
    manifest = dict(status='in_progress',archive_root=str(DEST),source_root=str(ROOT),
                    created_at=datetime.datetime.now().astimezone().isoformat(),
                    expected_count=len(files),expected_bytes=total,
                    original_source_hashes=source_hashes,files=list(previous.values()))
    def save_manifest():
        temp = MANIFEST.with_suffix('.json.tmp')
        temp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        temp.replace(MANIFEST)
    save_manifest()
    note = '''# 几何数据已迁移归档 / Geometry archive

本目录的正式 geometry_*.npz 已迁移到 E:/Predify_my归档/predify2021_my/ 下的同名相对目录。
只迁移 geometry_*.npz；concept、stimuli、diagnostic 和其他实验记录留在原位。
每个文件均在完整复制并以 SHA-256 校验后，才移除 C 盘副本。
完整清单、进度与哈希见上级实验目录 geometry_archive_manifest.json；status=complete 表示全部完成。

原有脚本完全未修改。原始路径不再直接包含几何文件，后续 AI/人工操作请先阅读此说明。
分析与审计请从项目根目录使用新增的兼容入口：

    python analysis/storage_audit/run_with_geometry_archive.py analysis/filtered_cv_20261027.py representation --dataset imagenet
    python analysis/storage_audit/run_with_geometry_archive.py analysis/audit_filtered_cv_20261027.py
    python analysis/storage_audit/run_with_geometry_archive.py --check

兼容入口仅在当前 Python 进程中重定向此实验 geometry 文件的只读访问和目录检索；不修改原脚本或冻结哈希。
直接启动旧脚本可能在缓存缺失时报告找不到文件；请改用兼容入口。
E 盘离线或归档路径失效时，兼容入口会明确提示归档位置。
恢复原路径：按 manifest 将归档文件复制回原 formal 目录，再逐文件核对 SHA-256。
不要直接重跑提取来恢复：旧提取逻辑会跳过已有 concept .pckl，不会补齐 geometry。
'''
    for d in ('imagenet','things'):
        (ROOT / REL / d / 'formal' / 'GEOMETRY_ARCHIVE_README.md').write_text(note,encoding='utf-8')
    (ROOT / REL / 'GEOMETRY_ARCHIVE_README.md').write_text(note,encoding='utf-8')
    started = time.monotonic(); last = started; done_bytes = 0; records = []
    with journal.open('a',encoding='utf-8') as log:
        for index,item in enumerate(sorted(files,key=lambda f:f['path']),1):
            rel = Path(item['path']); src = ROOT / rel; dst = DEST / rel
            assert src.resolve().is_relative_to(ROOT.resolve())
            assert dst.resolve().is_relative_to(DEST.resolve())
            assert not src.is_symlink() and not dst.is_symlink()
            if src.exists():
                before = src.stat()
                assert before.st_size == item['bytes'], f'Size changed: {src}'
                dst.parent.mkdir(parents=True,exist_ok=True)
                if dst.exists():
                    source_sha = sha(src)
                    assert dst.stat().st_size == before.st_size and sha(dst) == source_sha, f'Archive conflict: {dst}'
                else:
                    tmp = dst.with_suffix('.npz.partial')
                    h = hashlib.sha256()
                    with src.open('rb') as a, tmp.open('wb') as b:
                        for block in iter(lambda:a.read(8*1024**2),b''):
                            h.update(block); b.write(block)
                        b.flush(); os.fsync(b.fileno())
                    source_sha = h.hexdigest()
                    assert tmp.stat().st_size == before.st_size and sha(tmp) == source_sha
                    shutil.copystat(src,tmp); tmp.replace(dst)
                after = src.stat()
                assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns), f'Source changed: {src}'
                row = dict(relative_path=str(rel),bytes=before.st_size,sha256=source_sha,
                           archive_path=str(dst),source_mtime_ns=before.st_mtime_ns)
                log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush();os.fsync(log.fileno())
                # Both absolute paths have been checked above; deletion is limited to this one verified file.
                src.unlink()
            else:
                row = previous[str(rel)]
                assert dst.is_file() and dst.stat().st_size==row['bytes'] and sha(dst)==row['sha256']
            records.append(row);done_bytes+=row['bytes']
            if time.monotonic()-last>=20 or index==len(files):
                print(json.dumps(dict(done=index,total=len(files),gib=round(done_bytes/2**30,3),
                      total_gib=round(total/2**30,3),elapsed_seconds=round(time.monotonic()-started)),ensure_ascii=False),flush=True)
                manifest['files']=records;manifest['completed_count']=index;save_manifest();last=time.monotonic()
    assert all(sha(ROOT/p)==h for p,h in source_hashes.items()), 'Original source changed during archive'
    for d in ('imagenet','things'): assert not list((ROOT/REL/d/'formal').glob('geometry_*.npz'))
    manifest.update(status='complete',files=records,completed_count=len(records),
                    completed_at=datetime.datetime.now().astimezone().isoformat(),
                    archived_bytes=done_bytes,original_sources_unchanged=True)
    save_manifest()
    shutil.copy2(MANIFEST,DEST/'geometry_archive_manifest.json')
    shutil.copy2(ROOT/REL/'GEOMETRY_ARCHIVE_README.md',DEST/'GEOMETRY_ARCHIVE_README.md')
    print('COMPLETE: verified and archived',len(records),'files;',done_bytes,'bytes',flush=True)


if __name__=='__main__':main()
