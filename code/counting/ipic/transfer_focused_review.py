"""Strict, model-free import of focused visible-instance annotations."""
import copy, math

FORMAT='ipic_transfer_focused_review_v2'
PACKAGE='ipic_transfer_focused_20261104_v2'

def inside(p,b):return b[0]<=p[0]<b[2] and b[1]<=p[1]<b[3]

def import_focused_review(drafts,review,base_sha256):
 rows=copy.deepcopy(drafts['frames']);blockers=[]
 if review is not None:
  if review.get('format')!=FORMAT or review.get('package_id')!=PACKAGE or review.get('base_sha256')!=base_sha256:raise ValueError('Focused review identity mismatch')
  if not isinstance(review.get('reviewer'),str):raise ValueError('Missing reviewer')
  if set(review.get('frames',{}))!={r['image_id'] for r in rows}:raise ValueError('Focused frame coverage changed')
 for r in rows:
  e=review['frames'][r['image_id']] if review else None
  if e is not None:
   if e.get('decision') not in {'draft','complete','unclear','assistant_complete'} or not isinstance(e.get('notes'),str):raise ValueError('Invalid decision/notes')
   points=e.get('points');base={p['id']:p for p in r['point_records']};seen=set();coords=set()
   if not isinstance(points,list):raise ValueError('Invalid points')
   for p in points:
    if not isinstance(p,dict) or not isinstance(p.get('id'),str) or p['id'] in seen:raise ValueError('Duplicate/invalid point ID')
    seen.add(p['id']);xy=p.get('point')
    if not isinstance(xy,list) or len(xy)!=2 or not all(type(x) in (int,float) and math.isfinite(x) for x in xy) or not inside(xy,r['roi_xyxy']):raise ValueError('Point outside native foreground')
    if tuple(xy) in coords:raise ValueError('Duplicate point coordinates')
    coords.add(tuple(xy))
    if p['id'] in base:
     if p!=base[p['id']]:raise ValueError('Original point changed')
    elif not p['id'].startswith(r['image_id']+':reviewer:') or p.get('status')!='reviewer' or p.get('source')!='reviewer' or not any(inside(xy,x['xyxy']) for x in r['review_regions']):raise ValueError('Invalid reviewer point')
   if not {p['id'] for p in r['point_records'] if p['status']=='accepted'}<=seen:raise ValueError('Locked green point removed')
   checked=e.get('regions_checked')
   if not isinstance(checked,dict) or set(checked)!={x['id'] for x in r['review_regions']} or not all(type(x)is bool for x in checked.values()):raise ValueError('Region coverage changed')
   if r['exhaustive_reviewed']:
    if e['decision']!='assistant_complete' or points!=r['point_records'] or type(e.get('count'))is not int or e['count']!=r['count']:raise ValueError('Complete assistant frame changed')
   elif e['decision']=='assistant_complete':raise ValueError('Unresolved frame cannot claim assistant completion')
   elif e['decision']=='complete':
    if not review['reviewer'].strip() or not isinstance(e.get('reviewed_at'),str) or not e['reviewed_at'] or not all(checked.values()) or type(e.get('count'))is not int or e['count']!=len(points):raise ValueError('Incomplete focused review')
    r.update(points=[p['point'] for p in points],point_records=points,count=e['count'],exhaustive_reviewed=True,points_are_partial=False,annotation_author='assistant_clear_plus_owner_regions:'+review['reviewer'],reviewed_at=e['reviewed_at'],owner_notes=e['notes'],regions_checked=checked)
   else:
    if e['decision']=='unclear' and not e['notes'].strip():raise ValueError('Unclear boundary needs a note')
    r.update(exhaustive_reviewed=False,points_are_partial=True,count=None)
  if not r['exhaustive_reviewed']:blockers.append(r['image_id'])
 return rows,blockers
