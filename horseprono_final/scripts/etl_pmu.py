from __future__ import annotations
import argparse, os, sys
from datetime import date
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from app_core.pmu import get_programme, participants_to_df, programme_choices, get_participants
from scripts.import_history import env, upsert_race, upsert_participants
from app_core.db import save_ingestion_run

def main(target_date: date) -> None:
    sb=env(); programme=get_programme(target_date); choices=programme_choices(programme)
    if not choices:
        save_ingestion_run('pmu','empty',0,0,'Programme PMU non interprétable',target_date,target_date); print('Aucune course trouvée'); return
    read=written=0; errors=[]
    for ch in choices:
        try:
            df=participants_to_df(get_participants(target_date,ch['reunion'],ch['course']),target_date,ch['reunion'],ch['course'])
            if df.empty: continue
            read += len(df)
            race=df.iloc[0]
            rr={'external_id':str(race['race_id']),'race_date':target_date.isoformat(),'meeting_number':int(ch['reunion']),'race_number':int(ch['course']),'hippodrome':race.get('hippodrome'),'discipline':race.get('discipline'),'distance_m':int(race['distance']) if race.get('distance') is not None else None,'terrain':race.get('terrain'),'field_size':int(race.get('field_size')) if race.get('field_size') is not None else len(df),'status':'scheduled'}
            rid=upsert_race(sb,rr)
            hs=[]
            for _,r in df.iterrows():
                hs.append({'external_id':None,'horse_name':r.get('horse_name'),'horse_number':int(r['horse_number']),'jockey_name':r.get('jockey'),'trainer_name':r.get('trainer'),'odds':r.get('odds'),'weight_kg':r.get('weight'),'draw':r.get('draw'),'age':None,'sex':None,'recent_form':r.get('recent_form'),'finish_position':r.get('finish_position'),'is_non_runner':False})
            written += upsert_participants(sb,rid,hs)
        except Exception as exc: errors.append(f"R{ch['reunion']}C{ch['course']}: {exc}")
    save_ingestion_run('pmu','success' if not errors else 'completed_with_errors',read,written,'; '.join(errors),target_date,target_date)
    print(f'PMU: {read} lignes lues, {written} partants écrits')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--date',default=date.today().isoformat()); a=p.parse_args(); main(date.fromisoformat(a.date))
