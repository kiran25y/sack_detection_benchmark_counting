"""Extract native training mask assets for visual review; no model or fitting."""
import argparse, collections, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_api

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();root=Path(a.root);out=Path(a.output);out.mkdir(exist_ok=True)
    if (out/'assets.json').exists():raise ValueError('Exclusive extraction')
    public=root/'research-direction-20261012/ipic_checks_20261030/public_v1/public_train_manifest.jsonl'
    assert sha(public)=='8a70c247df7f304b46cd47b215de23e71c83a1aaeff58def3e5b8522b35e3721'
    native=root/'datasets/D2S-native-v1/annotations/D2S_training.json';coco=json.loads(native.read_text())
    rows={r['image_id']:r for r in map(json.loads,public.read_text().splitlines()) if r['dataset']=='D2S'}
    candidates=collections.defaultdict(list)
    for r in coco['annotations']:
        if r['image_id'] in rows and r['area']>=1000:candidates[r['category_id']].append(r)
    categories={r['id']:r['name'] for r in coco['categories']};assets=[]
    (out/'assets').mkdir(exist_ok=False)
    for category in sorted(candidates):
        chosen=[]
        for ann in sorted(candidates[category],key=lambda r:(r['image_id'],r['id'])):
            row=rows[ann['image_id']];src=root/row['image']
            if any(x['source_image_id']==ann['image_id'] for x in chosen):continue
            seg=ann['segmentation']
            if isinstance(seg,list):seg=mask_api.merge(mask_api.frPyObjects(seg,row['height'],row['width']))
            elif isinstance(seg['counts'],list):seg=mask_api.frPyObjects(seg,row['height'],row['width'])
            mask=mask_api.decode(seg).astype(bool);y,x=np.where(mask)
            if not len(x) or x.min()==0 or y.min()==0 or x.max()==row['width']-1 or y.max()==row['height']-1:continue
            box=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]
            if sha(src)!=row['image_sha256']:raise ValueError('Native asset image changed')
            image=Image.open(src).convert('RGB');im=image.crop(box);m=Image.fromarray(mask.astype('uint8')*255).crop(box)
            asset_id=f'd2s_{category}_{ann["id"]}';impath=out/'assets'/f'{asset_id}.png';mpath=out/'assets'/f'{asset_id}_mask.png'
            im.save(impath);m.save(mpath)
            record=dict(asset_id=asset_id,category=str(category),category_name=categories[category],split='train',
                parent_id='D2S:'+str(ann['image_id']),source_image_id=ann['image_id'],source_image=row['image'],
                source_image_sha256=row['image_sha256'],source_annotation_id=ann['id'],source_annotation_sha256=sha(native),
                source_box_xyxy=box,image='assets/'+impath.name,mask='assets/'+mpath.name,sha256=sha(impath),mask_sha256=sha(mpath),
                visual_review_pending=True,native_mask=True)
            chosen.append(record);assets.append(record)
            if len(chosen)==2:break
        if len(chosen)!=2:raise ValueError('Fewer than two independent source images for category '+str(category))
    assert len(assets)==120
    for start in range(0,len(assets),20):
        canvas=Image.new('RGB',(1600,1200),'white');d=ImageDraw.Draw(canvas)
        for i,r in enumerate(assets[start:start+20]):
            im=Image.open(out/r['image']).convert('RGB');m=Image.open(out/r['mask']).convert('L')
            shown=Image.new('RGB',im.size,(70,70,70));shown.paste(im,mask=m);shown.thumbnail((280,245))
            x=(i%5)*320;y=(i//5)*300;canvas.paste(shown,(x,y+42));d.text((x+3,y+4),r['asset_id']+'\n'+r['category_name'],fill='black')
        canvas.save(out/f'asset_review_{start//20:02}.jpg',quality=95)
    (out/'assets.json').write_text(json.dumps(assets,indent=2))
    (out/'report.json').write_text(json.dumps(dict(state='completed',assets=len(assets),categories=len(categories),
        optimizer_calls=0,model_calls=0,visual_review_pending=True,public_manifest_sha256=sha(public),native_annotation_sha256=sha(native)),indent=2))
if __name__=='__main__':main()
