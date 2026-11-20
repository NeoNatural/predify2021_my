"""Perceptual candidate screening; flagged pairs require visual review."""
from pathlib import Path
import sys,json,csv
import numpy as np
from PIL import Image,ImageOps,ImageDraw
from scipy.fft import dctn
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.prepare_native_20261029 import read,write,DOC
from predify_experiments.readout_init_20261012 import sha

def main():
    allpairs=[]
    for dataset in ('imagenet','things'):
        rows=read(DOC/dataset/'image_inventory.csv');bits=[];thumbs=[]
        for r in rows:
            with Image.open(r['path']) as im:
                a=np.asarray(im.convert('L').resize((32,32)),float);d=dctn(a,norm='ortho')[:8,:8].ravel()[1:];bits.append(d>np.median(d))
                thumbs.append(np.asarray(im.convert('RGB').resize((32,32)),float))
        bits=np.array(bits);pairs=[]
        for i in range(len(rows)):
            h=np.count_nonzero(bits[i+1:]!=bits[i],axis=1)
            for k in np.flatnonzero(h<=6):
                j=i+1+int(k);a,b=rows[i],rows[j]
                pairs.append(dict(dataset=dataset,index_a=a['index'],index_b=b['index'],concept_a=a['concept'],concept_b=b['concept'],fold_a=a['fold'],fold_b=b['fold'],split_a=a['split'],split_b=b['split'],hamming=int(h[k]),thumbnail_rmse=float(np.sqrt(np.mean((thumbs[i]-thumbs[j])**2))),path_a=a['path'],path_b=b['path']))
        write(DOC/dataset/'near_duplicate_candidates.csv',pairs);allpairs.extend(pairs)
        dest=DOC/dataset/'duplicate_sheets';dest.mkdir(exist_ok=True)
        for p in range((len(pairs)+9)//10):
            canvas=Image.new('RGB',(700,10*140),'white');draw=ImageDraw.Draw(canvas)
            for row,r in enumerate(pairs[p*10:p*10+10]):
                for col,key in enumerate(('a','b')):
                    with Image.open(r[f'path_{key}']) as im:thumb=ImageOps.contain(im.convert('RGB'),(330,112))
                    canvas.paste(thumb,(col*350,row*140));draw.text((col*350,row*140+113),f"{r[f'concept_{key}']} {r[f'index_{key}']} h={r['hamming']}",fill='black')
            canvas.save(dest/f'page_{p:02}.jpg')
    # Detect copies of previously used ImageNet files, even with different filenames.
    from torchvision.datasets import ImageNet
    from predify_experiments.tripartite_20261027 import historical_imagenet
    from predify_experiments.native_categories_20261029 import OUT
    from concurrent.futures import ThreadPoolExecutor
    im=ImageNet('C:/Users/liang/Documents/ImageNet','val');seen=historical_imagenet()
    for file in (ROOT/'Hebbian-CNN/Log').rglob('manifest.json'):
        if OUT in file.parents:continue
        m=json.loads(file.read_text(encoding='utf8'))
        if m.get('dataset')!='imagenet':continue
        for key in ('train','dev','test','trials'):
            for t in m.get(key,[]):
                seen.add(int(t['dataset_index']));seen.update(int(i) for i in t.get('previews',[]));seen.update(int(i) for i in t.get('primes',{}).values() if i is not None)
        seen.update(int(i) for i in m.get('historical_seen',[]))
    with ThreadPoolExecutor(max_workers=6) as ex:old_hashes=set(ex.map(sha,[im.samples[i][0] for i in sorted(seen)]))
    selected=read(DOC/'imagenet/image_inventory.csv');overlap=[r['index'] for r in selected if r['sha256'] in old_hashes]
    (DOC/'perceptual_screen.json').write_text(json.dumps(dict(passed=False,status='candidate screening complete; pending visual disposition',threshold='63 AC DCT bits, Hamming <= 6',candidates=allpairs,imagenet_historical_images_hashed=len(seen),imagenet_historical_byte_overlap=overlap),indent=2),encoding='utf8')
    assert not overlap,overlap
    print('Candidates',len(allpairs));print(json.dumps([{k:v for k,v in r.items() if not k.startswith('path')} for r in allpairs],indent=2))

if __name__=='__main__':main()
