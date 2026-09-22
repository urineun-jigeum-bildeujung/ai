"""P2 offline audit, incremental-rebuild planning, and precision-audit outputs."""
from __future__ import annotations
import csv,json,re,uuid
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
from allergen_repository import REF_FIELDS as FIELDS,identity
ROOT=Path(__file__).resolve().parents[2]; P=ROOT/'data/processed'; E=ROOT/'data/eval'

def classify(r):
 s=r['normalized_text']; raw=r['raw_ingredient_text']
 if any(c in raw for c in ('�','&quot','*','_')): return 'OCR_CORRUPTION'
 if re.search(r'\b(and|et|및)\b',s): return 'COMPOUND_PHRASE'
 if any(x in s for x in (' de ',' des ',' protéines',' farine',' huile',' graisse')): return 'MODIFIER_ATTACHMENT'
 if len(s)>80:return 'NON_INGREDIENT_TEXT'
 return 'LANGUAGE_VARIANT' if re.search('[àâçéèêëîïôûùüÿñæœ]',s) else 'UNKNOWN_ALIAS'
def build_manifest(dry_run=True, changed_products=None, pipeline_changed=False):
 data=json.loads((P/'product_allergen_refs_p1.json').read_text()); products=sorted({r['product_id'] for r in data['refs']})
 affected=products if pipeline_changed else sorted(set(changed_products or []))
 now=datetime.now(timezone.utc).replace(microsecond=0).isoformat()
 return {'run_id':str(uuid.uuid4()),'trigger_type':'PIPELINE_VERSION' if pipeline_changed else 'SOURCE_INGREDIENT',
 'previous_dictionary_version':data['dictionary_version'],'target_dictionary_version':data['dictionary_version'],
 'previous_pipeline_version':data['pipeline_version'],'target_pipeline_version':data['pipeline_version'],
 'affected_products':affected,'rebuilt_products':[] if dry_run else affected,'unchanged_products':sorted(set(products)-set(affected)),
 'failed_products':[],'failure_reasons':{},'dry_run':dry_run,'started_at':now,'completed_at':now}
def main():
 data=json.loads((P/'product_allergen_refs_p1.json').read_text()); refs=data['refs']; groups=defaultdict(list)
 for r in refs:groups[identity(r)].append(r)
 dup=[]
 for k,v in groups.items():
  if len(v)>1: dup.append({'identity':k,'original_count':len(v),'kept_row':v[0],'removed_rows':v[1:],
   'source_version_difference':len({x['source_version'] for x in v})>1,'valid_dedupe':len({json.dumps([x.get(z) for z in FIELDS[:-2]],sort_keys=True,ensure_ascii=False) for x in v})==1})
 (P/'allergen_evidence_dedupe_audit_p2.json').write_text(json.dumps({'original_rows':len(refs),'groups':dup},ensure_ascii=False,indent=2))
 causes=defaultdict(Counter); examples=defaultdict(lambda:defaultdict(list)); evals=defaultdict(Counter)
 for r in refs:
  if not r['usable_for_safety']:
   c=classify(r); causes[r['ingredient_source']][c]+=1; examples[r['ingredient_source']][c].append(r['raw_ingredient_text'])
 for source,c in causes.items():
  evals[source]={'unresolved_total':sum(c.values()),'root_causes':dict(c),'examples':{k:v[:5] for k,v in examples[source].items()}}
 (P/'allergen_unresolved_root_causes_p2.json').write_text(json.dumps(evals,ensure_ascii=False,indent=2))
 P.joinpath('allergen_evidence_rebuild_manifest.json').write_text(json.dumps(build_manifest(),ensure_ascii=False,indent=2))
 # Deterministic source-stratified 60-row sample; labels intentionally blank.
 buckets=defaultdict(list)
 for r in refs:
  if r['usable_for_safety']: buckets[r['source_dataset']].append(r)
 rows=[]
 for source in ('OPFF','OEM','GLOBAL'): rows += [(source,r) for r in buckets[source][:20]]
 E.mkdir(exist_ok=True)
 with (E/'allergen_mapping_precision_audit_p2.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['product_id','source_dataset','raw_ingredient_text','normalized_text','segmented_text','matched_text','matched_alias','allergen_code','mapping_method','confidence','dictionary_version','review_label','reviewer_note']);w.writeheader()
  for _,r in rows:w.writerow({'product_id':r['product_id'],'source_dataset':r['source_dataset'],**{k:r.get(k,'') for k in ['raw_ingredient_text','normalized_text','segmented_text','matched_text','matched_alias','allergen_code','mapping_method','confidence','dictionary_version']},'review_label':'','reviewer_note':''})
 print(len(dup),len(rows),evals)
if __name__=='__main__':main()
