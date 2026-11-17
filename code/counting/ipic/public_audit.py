"""Native public training admission: full decode, truth identity and matching capacity."""
import argparse,collections,hashlib,json,pathlib,time
import numpy as np
from PIL import Image
from pycocotools import mask as mask_api
from .matching import capacity_check

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def decode_mask(annotation,height,width):
    seg=annotation['segmentation']
    if isinstance(seg,list):rle=mask_api.merge(mask_api.frPyObjects(seg,height,width))
    elif isinstance(seg['counts'],list):rle=mask_api.frPyObjects(seg,height,width)
    else:rle=seg
    value=mask_api.decode(rle).astype(bool)
    if value.shape!=(height,width) or not value.any():raise ValueError('Empty or wrong-shape native mask')
    return value

def visible_anchor(mask):
    y,x=np.nonzero(mask);cx,cy=x.mean(),y.mean();i=np.argmin((x-cx)**2+(y-cy)**2)
    return [int(x[i]),int(y[i])]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    root=pathlib.Path(a.root);out=pathlib.Path(a.output);out.mkdir(exist_ok=True)
    report={'state':'running','fsc_done':0,'d2s_done':0,'capacities':0,'optimizer_steps':0,'gpu_used':False,'errors':[]}
    def save():
        tmp=out/'status.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(out/'status.json')
    cache={};manifest=[]
    def inspect_image(path,width,height,expected=None):
        raw=path.read_bytes();h=hashlib.sha256(raw).hexdigest()
        if expected is not None and h!=expected:raise ValueError('Native image hash changed: '+str(path))
        with Image.open(path) as im:
            if im.size!=(width,height) or int(im.getexif().get(274,1))!=1:raise ValueError('Native size/EXIF inconsistency')
            im.load();pixels=np.asarray(im.convert('RGB'));decoded=hashlib.sha256(pixels.tobytes()).hexdigest()
        return h,decoded
    def capacity(points,width,height):
        pts=np.asarray(points,dtype=np.float64).reshape(-1,2)
        checks=[]
        for side in (768,896,1024):
            w,h=max(1,round(width*side/max(width,height))),max(1,round(height*side/max(width,height)))
            key=(w,h)
            if key not in cache:
                yy,xx=np.meshgrid(np.arange((h+31)//32*8)*4+2,np.arange((w+31)//32*8)*4+2,indexing='ij')
                anchors=np.stack((xx.ravel(),yy.ravel()),1).astype(np.float32)
                cache[key]=(anchors,(anchors[:,0]<w)&(anchors[:,1]<h))
            transformed=(pts+.5)*np.array([w/width,h/height])-.5
            anchors,valid=cache[key];check=capacity_check(anchors,transformed,valid);checks.append(check);report['capacities']+=1
            inverse=(transformed+.5)/np.array([w/width,h/height])-.5
            if len(pts) and np.max(np.abs(inverse-pts))>.5:raise ValueError('Coordinate round-trip')
        return checks
    try:
        clean=root/'manifests/fsc147_train_groupclean.json';native=root/'datasets/FSCD147-author-release/extracted/FSC147_384_V2/annotations'
        rows=json.loads(clean.read_text())['queries'];truth=json.loads((native/'annotation_FSC147_384.json').read_text());splits=json.loads((native/'Train_Test_Val_FSC_147.json').read_text())
        excluded=sorted(set(splits['train'])-{r['query_id'] for r in rows})
        if len(rows)!=3653 or len(splits['train'])!=3659 or len(excluded)!=6:raise ValueError('FSC cohort changed')
        held=json.loads((root/'manifests/fsc147_val.json').read_text())['queries']+json.loads((root/'manifests/fsc147_test.json').read_text())['queries']
        held_hashes={r['image_sha256'] for r in held}
        all_train=json.loads((root/'manifests/fsc147_train.json').read_text())['queries'];all_map={r['query_id']:r for r in all_train}
        if not all(all_map[n]['image_sha256'] in held_hashes for n in excluded):raise ValueError('Unexplained train exclusion')
        categories=dict(line.split(maxsplit=1) for line in (native/'ImageClasses_FSC147.txt').read_text().splitlines() if line.strip())
        report['fsc_six_exclusions']=[{'query_id':n,'reason':'exact image hash also in native held-out split','sha256':all_map[n]['image_sha256']} for n in excluded]
        report['fsc_label_sha256']=digest(native/'annotation_FSC147_384.json');report['fsc_split_sha256']=digest(native/'Train_Test_Val_FSC_147.json');report['fsc_manifest_sha256']=digest(clean)
        report['fsc_boundary_points_retained']=0;save()
        for r in rows:
            if r['points']!=truth[r['query_id']]['points'] or r['gt_count']!=len(r['points']) or r['image_sha256'] in held_hashes:raise ValueError('Native FSC truth/partition changed')
            w,h=r['width'],r['height'];pts=np.asarray(r['points'],float).reshape(-1,2)
            if not np.isfinite(pts).all() or (pts<0).any() or (pts>np.array([w,h])+1e-10).any():raise ValueError('Native FSC point outside continuous extent')
            report['fsc_boundary_points_retained']+=int(np.any(pts>=np.array([w,h]),axis=1).sum())
            # Explicit coordinate chart conversion, not a change to native labels.
            centered=pts-.5;capacity(centered,w,h)
            path=root/r['image'].removeprefix('/research/');eh,dh=inspect_image(path,w,h,r['image_sha256'])
            manifest.append({'dataset':'FSC147','query_id':r['query_id'],'image':str(path.relative_to(root)),
                'width':w,'height':h,'image_sha256':eh,'decoded_sha256':dh,'split':'train','category':categories[r['query_id']],
                'count':len(pts),'native_points':r['points'],'points_pixel_centers':centered.tolist(),
                'native_to_pixel_center_offset':[-.5,-.5],'native_labels_unchanged':True,'exemplars_xyxy':r['exemplars']})
            report['fsc_done']+=1
            if report['fsc_done']%100==0:save()
        d2s=root/'datasets/D2S-native-v1';acq=json.loads((d2s/'status.json').read_text())
        if acq['state']!='completed' or acq['extracted_images']!=4380:raise ValueError('Native D2S acquisition incomplete')
        ann=d2s/'annotations/D2S_training.json';coco=json.loads(ann.read_text());report['d2s_annotation_sha256']=digest(ann)
        if report['d2s_annotation_sha256']!=acq['native_annotations']['D2S_training.json']['sha256']:raise ValueError('D2S annotation identity changed')
        imhash={r['file_name']:r['sha256'] for r in json.loads((d2s/'train_image_hashes.json').read_text())}
        byimage=collections.defaultdict(list)
        for r in coco['annotations']:byimage[r['image_id']].append(r)
        cats={r['id']:r['name'] for r in coco['categories']}
        for im in coco['images']:
            w,h=im['width'],im['height'];path=d2s/'train_images'/im['file_name'];eh,dh=inspect_image(path,w,h,imhash[im['file_name']])
            objects=[];union=np.zeros((h,w),bool)
            for obj in byimage[im['id']]:
                if obj.get('iscrowd',0):raise ValueError('Crowd mask needs a separate incomplete-label rule')
                mask=decode_mask(obj,h,w)
                if (union&mask).any():raise ValueError('Native visible instance masks overlap')
                union|=mask;anchor=visible_anchor(mask)
                if not mask[anchor[1],anchor[0]]:raise ValueError('Anchor outside native visible mask')
                x,y,bw,bh=obj['bbox']
                if not (0<=x<x+bw<=w and 0<=y<y+bh<=h):raise ValueError('Native bbox outside image')
                objects.append({'annotation_id':obj['id'],'category_id':obj['category_id'],'category':cats[obj['category_id']],
                                'point_pixel_center':anchor,'exemplar_xyxy':[x,y,x+bw,y+bh]})
            if not objects:raise ValueError('Unexpected empty native training image')
            capacity([o['point_pixel_center'] for o in objects],w,h)
            manifest.append({'dataset':'D2S','query_id':im['file_name'],'image':str(path.relative_to(root)),'width':w,'height':h,
                'image_sha256':eh,'decoded_sha256':dh,'split':'train','objects':objects,'count_all_products':len(objects),
                'image_id':im['id'],'native_annotation_source':'datasets/D2S-native-v1/annotations/D2S_training.json',
                'point_convention':'visible mask centroid projected to nearest interior integer raster center'})
            report['d2s_done']+=1
            if report['d2s_done']%100==0:save()
        target=out/'public_train_manifest.jsonl'
        with target.open('x') as f:
            for row in manifest:f.write(json.dumps(row)+'\n')
        report.update(state='completed',all_native_training_images_decoded=True,all_three_resize_capacities_passed=True,
                      public_manifest_sha256=digest(target),public_manifest_rows=len(manifest),
                      category_overlap_audit_pending=True,claim='joint public-data foundation; no strict unseen-FSC-category claim',
                      d2s_validation_or_test_used_for_optimization=False);save()
    except Exception as exc:report.update(state='failed',error=repr(exc));save();raise

if __name__=='__main__':main()
