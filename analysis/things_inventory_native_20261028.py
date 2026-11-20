"""Inventory only: no model, fitting, stimulus generation, or split selection."""
from pathlib import Path
import csv
import json
import pickle
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from predify_experiments.things_metadata import imagefolder_index, load_image_sources, _source_for, sha256_file

DATA = Path('C:/Users/liang/Documents/ChatGPT/THINGS数据集测试predify_my/data/THINGS')
OUT = ROOT / 'docs/things_native_inventory_20261028'
CATS = ['animal', 'clothing', 'food', 'plant', 'vehicle']
FIELDS = {'manual': 'Top-down Category (manual selection)',
          'human': 'Bottom-up Category (Human Raters)', 'wordnet': 'Top-down Category (WordNet)'}

def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = DATA / 'metadata/concepts-metadata_things.tsv'
    with metadata.open(encoding='utf-8-sig') as f:
        concepts = list(csv.DictReader(f, delimiter='\t'))
    lookup = {r['uniqueID']: r for r in concepts}
    paths, _ = imagefolder_index(DATA / 'images')
    sources = load_image_sources(DATA / 'metadata/01_image-level')
    history = defaultdict(set)
    history_files = []
    old_path = DATA / 'derived/THINGS_N_Back-1_seed114514.pckl'
    with old_path.open('rb') as f:
        old = pickle.load(f)
    for i, rel in zip(old['out_list'], old['image_relpaths']):
        assert paths[int(i)].relative_to(DATA/'images').as_posix() == rel
        history[rel].add('initial_1000')
    history_files.append(dict(path=str(old_path), sha256=sha256_file(old_path),
                              reference_entries=len(old['image_relpaths']), referenced_images=len(set(old['image_relpaths']))))
    for file in sorted((ROOT/'Hebbian-CNN/Log').rglob('manifest.json')):
        m = json.loads(file.read_text(encoding='utf-8'))
        if str(m.get('dataset', '')).lower() != 'things':
            continue
        used = set()
        for split in ('train', 'dev', 'test', 'trials'):
            for trial in m.get(split, []):
                used.add(int(trial['dataset_index']))
                used.update(int(i) for i in trial.get('previews', []))
                used.update(int(i) for i in trial.get('primes', {}).values() if i is not None)
        for role in m.get('visual_roles', []):
            used.add(int(role['image']))
        # Historical indices are conservative exclusions, even if absent from current trials.
        used.update(int(i) for i in m.get('historical_seen', []))
        used.update(int(i) for i in m.get('historical_target_exclusions', []))
        for i in used:
            history[paths[i].relative_to(DATA/'images').as_posix()].add(str(file.relative_to(ROOT)))
        history_files.append(dict(path=str(file), sha256=sha256_file(file), referenced_images=len(used)))
    print('Hashing', len(paths), 'local images; no experiment execution', flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        digests = list(ex.map(sha256_file, paths))
    history_hashes = {h for p,h in zip(paths,digests) if p.relative_to(DATA/'images').as_posix() in history}
    historical_concepts = {rel.split('/')[0] for rel in history}
    seen = set(); rows=[]
    for i,(p,h) in enumerate(zip(paths,digests)):
        rel=p.relative_to(DATA/'images').as_posix(); uid=rel.split('/')[0]
        source=_source_for(p, DATA/'images', sources)
        duplicate=source!='imagenet' and h in seen
        if source!='imagenet': seen.add(h)
        r=dict(index=i, relative_path=rel, concept=uid, source=source, sha256=h,
               excluded_imagenet=source=='imagenet', excluded_duplicate=duplicate,
               eligible=source!='imagenet' and not duplicate,
               historical_image=rel in history or h in history_hashes,
               historical_concept=uid in historical_concepts,
               history_sources=';'.join(sorted(history.get(rel, []))))
        for name,field in FIELDS.items():
            labels={s.strip() for s in lookup[uid][field].split(',') if s.strip()}
            matches=[c for c in CATS if c in labels]
            r[name+'_categories']=';'.join(matches)
            r[name+'_exclusive']=matches[0] if len(matches)==1 else ''
        rows.append(r)
    by_concept=defaultdict(list)
    for r in rows: by_concept[r['concept']].append(r)
    cr=[]
    for meta in concepts:
        uid=meta['uniqueID']; rr=by_concept[uid]; eligible=[r for r in rr if r['eligible']]
        fresh=[r for r in eligible if not r['historical_image']]
        r=dict(concept=uid, word=meta['Word'], total_images=len(rr),eligible_images=len(eligible),
               fresh_images=len(fresh), historical_concept=uid in historical_concepts)
        for name,field in FIELDS.items():
            r[name+'_original']=meta[field]
            r[name+'_exclusive']=rr[0][name+'_exclusive']
            r[name+'_categories']=rr[0][name+'_categories']
        cr.append(r)
    summaries=[]
    for scheme in FIELDS:
        for c in CATS:
            cc=[r for r in cr if r[scheme+'_exclusive']==c]
            rr=[r for r in rows if r[scheme+'_exclusive']==c]
            summaries.append(dict(scheme=scheme,category=c,concepts=len(cc),raw_images=len(rr),
                excluded_imagenet=sum(r['excluded_imagenet'] for r in rr),
                excluded_duplicate=sum(r['excluded_duplicate'] for r in rr),
                eligible_images=sum(r['eligible'] for r in rr),
                source_unknown_eligible=sum(r['eligible'] and r['source']=='source_unknown' for r in rr),
                historical_eligible=sum(r['eligible'] and r['historical_image'] for r in rr),
                fresh_images=sum(r['eligible'] and not r['historical_image'] for r in rr),
                concepts_eligible_ge3=sum(r['eligible_images']>=3 for r in cc),
                concepts_fresh_ge3=sum(r['fresh_images']>=3 for r in cc),
                pristine_concepts_ge3=sum(not r['historical_concept'] and r['fresh_images']>=3 for r in cc),
                concepts_fresh_ge5=sum(r['fresh_images']>=5 for r in cc),
                pristine_concepts_ge5=sum(not r['historical_concept'] and r['fresh_images']>=5 for r in cc)))
    ambiguity=[r for r in cr if ';' in r['manual_categories']]
    review=[]
    for r in cr:
        if not r['manual_exclusive']: continue
        reasons=[]
        if r['human_exclusive'] != r['manual_exclusive']: reasons.append('human_label_missing_or_different')
        if r['wordnet_exclusive'] != r['manual_exclusive']: reasons.append('wordnet_label_missing_or_different')
        if r['concept'] in {'plant','tree','flower','animal','bird','insect','vehicle','food','clothing','car','boat'}:
            reasons.append('broad_concept_check_hierarchy_overlap')
        if reasons: review.append(dict(**r,review_reasons=';'.join(reasons)))
    write_csv(OUT/'images.csv', rows); write_csv(OUT/'concepts.csv', cr)
    write_csv(OUT/'category_summary.csv', summaries); write_csv(OUT/'ambiguous_concepts.csv', ambiguity)
    write_csv(OUT/'label_review_queue.csv', review)
    audit=dict(date='2026-10-28', inventory_only=True, total_images=len(rows),total_concepts=len(cr),
               source_counts=dict(Counter(r['source'] for r in rows)),
               eligible_images=sum(r['eligible'] for r in rows),
               excluded_duplicate=sum(r['excluded_duplicate'] for r in rows),
               historical_paths=len(history),historical_hashes=len(history_hashes),
               historical_concepts=len(historical_concepts),
               classification='Exact membership tokens in native metadata; exclusive across selected five categories; no ImageNet mapping or pairing',
               manual_ambiguous_concepts=len(ambiguity),manual_ambiguous_images=sum(r['total_images'] for r in ambiguity),
               history_files=history_files, metadata_sha256=sha256_file(metadata),
               limitations=['History includes identifiable manifests and original cohort, not proof of no unlogged use.',
                            'SHA256 deduplication is not perceptual near-duplicate detection.',
                            'Inventory eligibility is not final visual stimulus QC.',
                            'WordNet and human columns are sensitivity analyses, not union-expanded labels.'],
               script_sha256=sha256_file(Path(__file__)))
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    assert len(rows)==26107 and len(cr)==1854
    assert all(r['raw_images']==r['excluded_imagenet']+r['excluded_duplicate']+r['eligible_images'] for r in summaries)
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=='__main__': main()
