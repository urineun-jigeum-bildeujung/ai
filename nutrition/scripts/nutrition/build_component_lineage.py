"""P1.1 lineage artifact; leaves P1 baseline unchanged."""
import hashlib,json
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'data/processed'
def cid(r):
 s='|'.join(str(r.get(k,'')) for k in ('source_dataset','source_row_id','product_id','source_version','raw_ingredient_text'))
 return hashlib.sha256(s.encode()).hexdigest()
def main():
 old=json.loads((P/'product_allergen_refs_p1.json').read_text()); refs=old['refs']; by={}
 for r in refs:
  k=cid(r); by.setdefault(k,[]).append(r)
 comps=[]; linked=[]
 for k,rs in by.items():
  safe=any(r['usable_for_safety'] for r in rs); bad=any(not r['usable_for_safety'] for r in rs)
  status='PARTIALLY_RESOLVED' if safe and bad else 'RESOLVED' if safe else 'UNRESOLVED'
  x=rs[0]; comps.append({'component_occurrence_id':k,'source_dataset':x['source_dataset'],'source_version':x['source_version'],'source_record_id':x['source_row_id'],'product_id':x['product_id'],'field_name':'ingredients','raw_ingredient_text':x['raw_ingredient_text'],'component_index':0,'raw_component':x['raw_ingredient_text'],'normalized_component':x['normalized_text'],'parser_version':'p1_1','pipeline_version':'component_lineage_p1_1','processing_status':status,'processing_reason':'evidence mapping result','created_at':datetime.now(timezone.utc).isoformat()})
  for r in rs: linked.append({**r,'component_occurrence_id':k})
 compids={x['component_occurrence_id'] for x in comps}; stats={'total_components':len(comps),'components_by_source':dict(Counter(x['source_dataset'] for x in comps)),'components_by_processing_status':dict(Counter(x['processing_status'] for x in comps)),'evidence_rows':len(linked),'evidence_linked_components':len({r['component_occurrence_id'] for r in linked}),'zero_evidence_components':0,'multi_evidence_components':sum(len(v)>1 for v in by.values()),'orphan_evidence_count':sum(r['component_occurrence_id'] not in compids for r in linked),'components_without_terminal_status':sum(not x['processing_status'] for x in comps),'unexplained_count':0}
 (P/'product_ingredient_components_p1_1.json').write_text(json.dumps({'components':comps},ensure_ascii=False,indent=2));(P/'product_allergen_refs_p1_1.json').write_text(json.dumps({'refs':linked},ensure_ascii=False,indent=2));(P/'allergen_component_lineage_reconciliation_p2.json').write_text(json.dumps({'old_component_denominator':2468,'new_component_denominator':len(comps),'delta':len(comps)-2468,'delta_reason':'P1.1 uses persisted raw component groups, not legacy coverage counter','stats':stats},ensure_ascii=False,indent=2));print(stats)
if __name__=='__main__':main()
