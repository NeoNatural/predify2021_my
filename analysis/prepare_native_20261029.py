"""Data-only audit, deterministic native-label sampling and visual QC sheets."""
from pathlib import Path
import csv,json,sys,random
from collections import defaultdict,Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from predify_experiments.native_categories_20261029 import OUT,DOC,PLAN,conditions,IDENTITIES,NOISES
from predify_experiments.readout_init_20261012 import sha,immutable_json
from predify_experiments.tripartite_20261027 import historical_imagenet
from predify_experiments.paper_stimuli import imagenet_supercategory
SEED=20261029
DATA=Path('C:/Users/liang/Documents/ChatGPT/THINGS数据集测试predify_my/data/THINGS')
PLANT_EXCLUDE={'plant','tree','flower','bonsai','bush','hedge','weed'}
IM_EXCLUDE={117:'shell-only exemplars make whole-animal category ambiguous',496:'decorative stocking',676:'animal muzzle',678:'medical brace',691:'breathing apparatus',885:'fabric, not clothing',958:'animal fodder',961:'raw dough',694:'vehicle part',660:'stationary manufactured home',643:'ambiguous mask',836:'overlap with sunglasses',638:'overlap with swimsuit',578:'broad gown overlaps academic/vestment',689:'overlap with skirt',968:'ambiguous cup/drink subject',947:'generic mushrooms include nonedible wild fungi; ambiguous food boundary'}
IMAGE_EXCLUDE={'imagenet':{33359:'illustration rather than clearly visible wearable',18200:'animal obscured by darkness',48346:'food/drink obscured by darkness'},'things':{}}

def write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def read(path):
    with Path(path).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def scan_imagenet():
    from torchvision.datasets import ImageNet
    im=ImageNet('C:/Users/liang/Documents/ImageNet','val');seen=historical_imagenet();history=[]
    for path in sorted((ROOT/'Hebbian-CNN/Log').rglob('manifest.json')):
        if OUT in path.parents:continue
        m=json.loads(path.read_text(encoding='utf8'))
        if m.get('dataset')!='imagenet':continue
        ids=set()
        for split in ('train','dev','test','trials'):
            for t in m.get(split,[]):
                ids.add(int(t['dataset_index']));ids.update(int(i) for i in t.get('previews',[]));ids.update(int(i) for i in t.get('primes',{}).values() if i is not None)
        ids.update(int(i) for i in m.get('historical_seen',[]));seen.update(ids)
        history.append(dict(path=str(path),sha256=sha(path),images=len(ids)))
    used_concepts={im.targets[i] for i in seen};rows=[];mapping=[]
    for l,wnid in enumerate(im.wnids):
        cat=imagenet_supercategory(l)
        if cat not in ('animals','clothing','food','vehicles'):continue
        reason=IM_EXCLUDE.get(l,'native synset: animal / wearable / human food or drink / whole vehicle')
        mapping.append(dict(concept=wnid,label=l,name='; '.join(im.classes[l]),category=cat,included=l not in IM_EXCLUDE,reason=reason,historical_concept=l in used_concepts))
    accepted={r['label']:r for r in mapping if r['included']}
    for i,(p,l) in enumerate(im.samples):
        if l not in accepted:continue
        r=accepted[l];rows.append(dict(index=i,path=p,concept=r['concept'],category=r['category'],eligible=i not in seen and i not in IMAGE_EXCLUDE['imagenet'],historical_concept=l in used_concepts,source='imagenet_validation'))
    write(DOC/'imagenet/category_mapping.csv',mapping)
    return rows,history

def scan_things():
    inv=DOC/'inventory';rr=read(inv/'images.csv');mapping=[];cats={'animal':'animals','clothing':'clothing','food':'food','plant':'plants','vehicle':'vehicles'}
    # Exclude generic umbrella concepts in every category, before looking at scores.
    broad={'animal','bird','insect','fish','food','clothing','vehicle','car','boat','shoe','hat','fruit','vegetable','furniture',
           'crown','seafood','sausage','sauce','apple','grape','lingerie','motorcycle','yacht','bread','melon','poodle','shirt','coat','sweater','jacket','underwear','stockings','truck','cart','buggy','jet','bike','scooter'}|PLANT_EXCLUDE
    for r in read(inv/'concepts.csv'):
        cat=cats.get(r['manual_exclusive']);ok=bool(cat) and r['concept'] not in broad
        mapping.append(dict(concept=r['concept'],category=cat or '',included=ok,reason='exclusive native manual category' if ok else 'generic/overlapping concept or nonexclusive native category',manual=r['manual_original'],human=r['human_original'],wordnet=r['wordnet_original'],historical_concept=r['historical_concept']))
    accepted={r['concept']:r for r in mapping if r['included']};rows=[]
    for r in rr:
        if r['concept'] not in accepted:continue
        rows.append(dict(index=int(r['index']),path=str(DATA/'images'/r['relative_path']),concept=r['concept'],category=accepted[r['concept']]['category'],eligible=r['eligible']=='True' and r['historical_image']=='False' and r['historical_concept']=='False',historical_concept=r['historical_concept']=='True',source=r['source']))
    write(DOC/'things/category_mapping.csv',mapping)
    return rows,json.loads((inv/'audit.json').read_text(encoding='utf8'))['history_files']

def prepare():
    DOC.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'freeze.json').exists():raise RuntimeError('Already frozen; do not resample')
    summaries={}
    for dataset,scanner in [('imagenet',scan_imagenet),('things',scan_things)]:
        previous_concepts=read(DOC/dataset/'concept_review.csv') if (DOC/dataset/'concept_review.csv').exists() else []
        previous_images=read(DOC/dataset/'image_inventory.csv') if (DOC/dataset/'image_inventory.csv').exists() else []
        rows,history=scanner();cats=('animals','clothing','food','vehicles') if dataset=='imagenet' else ('animals','clothing','food','plants','vehicles')
        pools=defaultdict(list)
        for r in rows:
            if r['eligible']:pools[r['concept']].append(r)
        selected=[];concept_rows=[];rng=random.Random(SEED+(dataset=='things'))
        for cat in cats:
            eligible=sorted(c for c,rr in pools.items() if len(rr)>=5 and rr[0]['category']==cat)
            # Prefer pristine concepts, but ImageNet is explicitly an expansion study.
            rng.shuffle(eligible);eligible.sort(key=lambda c:pools[c][0]['historical_concept'])
            if previous_concepts:
                prior={r['concept']:j for j,r in enumerate(previous_concepts)}
                eligible.sort(key=lambda c:prior.get(c,100000))
                chosen=[r['concept'] for r in previous_concepts if r['category']==cat and r['split']!='reserve']
                if chosen:
                    reserve=[c for c in eligible if c not in chosen]
                    restored=[c if c in eligible else reserve.pop(0) for c in chosen]
                    eligible=restored+[c for c in eligible if c not in restored]
            dev=1 if dataset=='things' and cat=='plants' else 4
            if len(eligible)<20+dev:raise RuntimeError(f'{dataset} {cat}: {len(eligible)} < {20+dev}')
            for j,c in enumerate(eligible):
                split='formal' if j<20 else 'development' if j<20+dev else 'reserve'
                fold=j//4 if split=='formal' else -1
                concept_rows.append(dict(concept=c,semantic_group=c,category=cat,split=split,fold=fold,eligible_images=len(pools[c]),historical_concept=pools[c][0]['historical_concept']))
                if split=='reserve':continue
                candidates=list(pools[c]);rng.shuffle(candidates)
                old={int(r['index']):j for j,r in enumerate(previous_images) if r['concept']==c}
                if old:candidates.sort(key=lambda r:old.get(r['index'],100000))
                for role,r in zip(['target']*4+['preview'],candidates[:5]):selected.append(dict(r,split=split,fold=fold,role=role))
        # Fresh file hashes, not reused from the previous inventory.
        hashes=defaultdict(list)
        for r in selected:r['sha256']=sha(r['path']);hashes[r['sha256']].append(r)
        if any(len(v)>1 for v in hashes.values()):raise RuntimeError('Selected byte duplicates require replacement and re-review')
        images={str(r['index']):dict(path=r['path'],concept=r['concept'],category=cats.index(r['category']),sha256=r['sha256'],historical=r['historical_concept'],source=r['source']) for r in selected}
        preview={r['concept']:r['index'] for r in selected if r['role']=='preview'}
        trials=[];devtrials=[];roles=[]
        for r in selected:
            if r['role']!='target':continue
            c=r['concept'];cat=r['category'];split=r['split'];fold=r['fold'];local=random.Random(SEED+r['index'])
            available={v:[q['concept'] for q in concept_rows if q['category']==v and q['split']==split and q['fold']==fold] for v in cats}
            others=[v for v in cats if v!=cat];j=len(trials) if split=='formal' else len(devtrials)
            backgroundcats=others if len(cats)==5 else others+[others[(j//4)%3]]
            used=set();background=[]
            for v in backgroundcats:
                choices=[q for q in available[v] if q not in used]
                if not choices:raise RuntimeError(f'{dataset} {split}: insufficient distinct background concepts')
                q=local.choice(choices);used.add(q);background.append(preview[q])
            local.shuffle(background);slot=2+j%2
            same=[q for q in available[cat] if q!=c]
            # The single plant development concept cannot supply a same-category other-concept control.
            # It is used only as a zero-W/reference diagnostic, never as formal data.
            prime_same=preview[local.choice(same)] if same else preview[c]
            t=dict(dataset_index=r['index'],concept=c,category_index=cats.index(cat),fold=fold,previews=background,slot=slot,primes=dict(exact=r['index'],same_concept=preview[c],same_category=prime_same,different_category=background[slot]),previously_viewed=r['historical_concept'],development_same_category_available=bool(same))
            (trials if split=='formal' else devtrials).append(t)
            for identity in IDENTITIES:
                ids=list(background);ids[slot]=t['primes'][identity]
                for pos,i in enumerate(ids):roles.append(dict(target=r['index'],split=split,fold=fold,identity=identity,position=pos,image=i,concept=images[str(i)]['concept'],role='prime' if pos==slot else 'distractor'))
            roles.append(dict(target=r['index'],split=split,fold=fold,identity='all',position='center',image=r['index'],concept=c,role='target'))
        # Every visual role has one ownership; development cannot leak into any formal fold.
        ownership=defaultdict(set)
        for r in roles:ownership[r['concept']].add((r['split'],r['fold']))
        assert all(len(v)==1 for v in ownership.values())
        formal_targets={t['dataset_index'] for t in trials}
        assert all(r['image'] not in formal_targets or (r['identity']=='exact' and r['image']==r['target']) or r['role']=='target' for r in roles)
        assert Counter(t['fold'] for t in trials)=={f:len(cats)*16 for f in range(5)}
        manifest=dict(dataset=dataset,categories=cats,seed=SEED,trials=trials,development=devtrials,images=images,concept_folds={r['concept']:r['fold'] for r in concept_rows if r['split']=='formal'},conditions=conditions(),noises=NOISES,scope='ImageNet: previously viewed concepts / fresh images; THINGS: pristine native concepts',sampling_plan_sha=sha(PLAN))
        (OUT/dataset).mkdir(parents=True,exist_ok=True)
        (OUT/dataset/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
        write(DOC/dataset/'concept_review.csv',concept_rows);write(DOC/dataset/'image_inventory.csv',selected);write(DOC/dataset/'visual_roles.csv',roles)
        summaries[dataset]=dict(categories=cats,formal_targets=len(trials),formal_concepts=20*len(cats),development_targets=len(devtrials),development_concepts=len(devtrials)//4,selected_images=len(images),historical_concepts=sum(r['historical_concept'] for r in concept_rows if r['split']=='formal'),concept_role_overlap=0,byte_duplicates=0,history_files=history,visual_qc='pending',near_duplicate_qc='pending')
        contacts(dataset,selected)
    (DOC/'data_acceptance_pending.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summaries,ensure_ascii=False,indent=2),flush=True)

def contacts(dataset,selected):
    from PIL import Image,ImageOps,ImageDraw
    dest=DOC/dataset/'contact_sheets';dest.mkdir(parents=True,exist_ok=True)
    by=defaultdict(list)
    for r in selected:by[r['concept']].append(r)
    cc=list(by)
    for page in range((len(cc)+9)//10):
        canvas=Image.new('RGB',(1000,10*125),'white');draw=ImageDraw.Draw(canvas)
        for row,c in enumerate(cc[page*10:page*10+10]):
            for col,r in enumerate(by[c]):
                with Image.open(r['path']) as im:thumb=ImageOps.contain(im.convert('RGB'),(190,94))
                x=col*200;y=row*125;canvas.paste(thumb,(x,y));draw.text((x,y+95),f"{c} {r['index']} {r['role'][0]}",fill='black');draw.text((x,y+109),f"{r['category']} {r['split'][0]} fold{r['fold']}",fill='black')
        canvas.save(dest/f'page_{page:02}.jpg')

if __name__=='__main__':prepare()
