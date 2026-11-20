"""Run unchanged research scripts with read-only access to archived geometry."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import os
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / 'Hebbian-CNN/Log/tripartite_filtered_cv_20261027'
MANIFEST = EXPERIMENT / 'geometry_archive_manifest.json'


@contextlib.contextmanager
def archive_access(manifest):
    original_open, original_exists, original_glob = Path.open, Path.exists, Path.glob
    records = {os.path.normcase(os.path.abspath(ROOT/r['relative_path'])):r for r in manifest['files']}
    def resolve(path):
        row = records.get(os.path.normcase(os.path.abspath(path)))
        if row and not original_exists(path):
            dest = Path(row['archive_path'])
            if not original_exists(dest):
                raise FileNotFoundError(f'Geometry is archived; connect the E: drive or restore this file: {dest}. See {MANIFEST}')
            if dest.stat().st_size != row['bytes']:
                raise OSError(f'Archive size mismatch: {dest}; see {MANIFEST}')
            return dest
        return path
    def opened(path,mode='r',*args,**kwargs):
        return original_open(resolve(path) if not any(c in mode for c in 'wa+') else path,mode,*args,**kwargs)
    def exists(path):
        return original_exists(resolve(path))
    def glob(path,pattern,*args,**kwargs):
        local = list(original_glob(path,pattern,*args,**kwargs))
        yield from local
        if pattern == 'geometry_*.npz' and path.absolute() in [EXPERIMENT/d/'formal' for d in ('imagenet','things')]:
            seen = {p.name for p in local}
            for key,row in records.items():
                p = ROOT/row['relative_path']
                if p.parent == path.absolute() and p.name not in seen:
                    resolve(p)
                    yield p
    import builtins
    original_builtin_open = builtins.open
    original_io_open = io.open
    def builtin_open(file,mode='r',*args,**kwargs):
        if isinstance(file,(str,os.PathLike)) and not any(c in mode for c in 'wa+'):
            file=resolve(Path(file))
        return original_builtin_open(file,mode,*args,**kwargs)
    def io_open(file,mode='r',*args,**kwargs):
        if isinstance(file,(str,os.PathLike)) and not any(c in mode for c in 'wa+'):
            file=resolve(Path(file))
        return original_io_open(file,mode,*args,**kwargs)
    try:
        Path.open,Path.exists,Path.glob=opened,exists,glob
        builtins.open=builtin_open
        io.open=io_open
        yield
    finally:
        Path.open,Path.exists,Path.glob=original_open,original_exists,original_glob
        builtins.open=original_builtin_open
        io.open=original_io_open


def main():
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
    if manifest['status']!='complete':raise SystemExit('Archive incomplete; inspect '+str(MANIFEST))
    args=sys.argv[1:]
    with archive_access(manifest):
        if args==['--check']:
            for row in manifest['files']:
                p=ROOT/row['relative_path']
                assert p.exists()
                with p.open('rb') as f:assert f.read(4)==b'PK\x03\x04'
            for dataset in ('imagenet','things'):
                assert len(list((EXPERIMENT/dataset/'formal').glob('geometry_*.npz')))==1536
            for rel,expected in manifest['original_source_hashes'].items():
                assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==expected,rel
            print('PASS: 3072 archived files readable; 1536 per dataset; all original frozen source files unchanged.')
            return
        if not args:raise SystemExit('Usage: run_with_geometry_archive.py --check OR script.py [args]')
        script=(ROOT/args[0]).resolve()
        if not script.is_relative_to(ROOT) or script.suffix!='.py':raise SystemExit('Use a Python script inside this project')
        sys.path.insert(0,str(ROOT));sys.argv=[str(script),*args[1:]]
        runpy.run_path(str(script),run_name='__main__')


if __name__=='__main__':main()
