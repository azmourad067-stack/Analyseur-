import argparse, os, time, hashlib, json
from datetime import date, datetime, timedelta
import pandas as pd, requests
from supabase import create_client

API='https://open-pmu-api.vercel.app/api/arrivees'

ALIASES={
'race_date':['race_date','date','date_course'],'external_id':['external_id','race_id','api_race_id'],'meeting_number':['meeting_number','reunion','r'],'race_number':['race_number','course','c'],'hippodrome':['hippodrome','lieu','track'],'discipline':['discipline','type'],'distance_m':['distance_m','distance'],'terrain':['terrain','going'],'field_size':['field_size','partants'],'horse_number':['horse_number','numero','num'],'horse_name':['horse_name','nom_cheval','nom'],'jockey_name':['jockey_name','nom_jockey','jockey','driver'],'trainer_name':['trainer_name','nom_entraineur','entraineur','trainer'],'odds':['odds','cote','cote_finale'],'weight_kg':['weight_kg','poids'],'draw':['draw','corde'],'age':['age'],'sex':['sex'],'recent_form':['recent_form','musique'],'finish_position':['finish_position','place','position'],'is_non_runner':['is_non_runner','non_runner','non_partant']}

def env():
    u=os.getenv('SUPABASE_URL'); k=os.getenv('SUPABASE_SERVICE_KEY') or os.getenv('SUPABASE_KEY')
    if not u or not k: raise RuntimeError('SUPABASE_URL et SUPABASE_SERVICE_KEY sont requis.')
    return create_client(u,k)

def first(d, keys):
    if not isinstance(d,dict): return None
    for k in keys:
        if k in d and d[k] not in ('',None): return d[k]
    return None

def num(x):
    if x is None or x=='': return None
    try: return float(str(x).replace(',','.').replace('%',''))
    except: return None

def integer(x):
    n=num(x); return int(n) if n is not None else None

def parse_date(x):
    if isinstance(x,date): return x.isoformat()
    s=str(x)
    for fmt in ('%Y-%m-%d','%d/%m/%Y','%d-%m-%Y'):
        try:return datetime.strptime(s,fmt).date().isoformat()
        except:pass
    return s

def extract_races(payload):
    if isinstance(payload,dict):
        for k in ('message','races','data','results'):
            if k in payload:
                v=payload[k]
                if isinstance(v,str):
                    try:v=json.loads(v)
                    except:continue
                if isinstance(v,list): return v
                if isinstance(v,dict): return list(v.values())
    return payload if isinstance(payload,list) else []

def normalize_race(r):
    rid=first(r,['external_id','race_id','id','course_id','id_course'])
    rd=parse_date(first(r,['race_date','date','date_course']))
    return {'external_id':str(rid) if rid is not None else None,'race_date':rd,'meeting_number':integer(first(r,['meeting_number','reunion','num_reunion'])),'race_number':integer(first(r,['race_number','course','num_course'])),'hippodrome':first(r,['hippodrome','lieu','hippodrome_nom','track']),'discipline':first(r,['discipline','type','specialite']),'distance_m':integer(first(r,['distance_m','distance'])),'terrain':first(r,['terrain','going']),'field_size':integer(first(r,['field_size','partants','nombre_partants'])),'status':first(r,['status','statut']) or 'finished'}

def normalize_horses(r):
    horses=first(r,['participants','chevaux','horses','partants'])
    if isinstance(horses,dict): horses=list(horses.values())
    return horses if isinstance(horses,list) else []

def normalize_horse(h,r):
    return {'external_id':str(first(h,['external_id','horse_id','id'])) if first(h,['external_id','horse_id','id']) is not None else None,'horse_name':first(h,['horse_name','nom_cheval','nom']),'horse_number':integer(first(h,['horse_number','numero','num'])),'jockey_name':first(h,['jockey_name','nom_jockey','jockey','driver']),'trainer_name':first(h,['trainer_name','nom_entraineur','entraineur','trainer']),'odds':num(first(h,['odds','cote','cote_finale'])),'weight_kg':num(first(h,['weight_kg','poids'])),'draw':integer(first(h,['draw','corde'])),'age':integer(first(h,['age'])),'sex':first(h,['sex','sexe']),'recent_form':first(h,['recent_form','musique']),'finish_position':integer(first(h,['finish_position','place','position'])),'is_non_runner':bool(first(h,['is_non_runner','non_runner','non_partant']) in (True,1,'1','true','True','oui','OUI'))}

def apply_arrivee(r,horses):
    arr=first(r,['arrivee','arrival','classement'])
    if isinstance(arr,str):
        try: arr=json.loads(arr)
        except: arr=None
    if isinstance(arr,list):
        pos={str(v):i+1 for i,v in enumerate(arr)}
        for h in horses:
            n=h.get('horse_number')
            if n is not None and str(n) in pos: h['finish_position']=pos[str(n)]
    details=first(r,['arrivee_details'])
    if isinstance(details,dict):
        for h in horses:
            n=h.get('horse_number')
            d=details.get(str(n)) or details.get(n) if n is not None else None
            if isinstance(d,dict):
                if h.get('finish_position') is None: h['finish_position']=integer(first(d,['position','place','rang']))
                for target,keys in [('horse_name',['nom_cheval','nom']),('jockey_name',['nom_jockey','jockey']),('trainer_name',['nom_entraineur','entraineur']),('sex',['sexe','sex']),('recent_form',['musique']),('draw',['corde']),('age',['age'])]:
                    if h.get(target) in (None,''): h[target]=first(d,keys)
                if h.get('odds') is None: h['odds']=num(first(d,['cote','cotes']))
    return horses

def upsert_race(sb,r):
    q=sb.table('races').upsert(r,on_conflict='external_id').execute()
    if not q.data: raise RuntimeError('Upsert race sans retour')
    return q.data[0]['id']

def upsert_participants(sb,race_id,hs):
    rows=[]
    for h in hs:
        h=dict(h); h['race_id']=race_id
        if h.get('horse_number') is None: continue
        rows.append(h)
    if not rows:return 0
    for i in range(0,len(rows),500): sb.table('participants').upsert(rows[i:i+500],on_conflict='race_id,horse_number').execute()
    return len(rows)

def import_api(sb,start,end,delay):
    run=sb.table('ingestion_runs').insert({'source':'open-pmu-api','status':'running'}).execute().data[0]['id']; read=written=0; errors=[]
    d=start
    while d<=end:
        try:
            resp=requests.get(API,params={'date':d.strftime('%d/%m/%Y')},timeout=45); resp.raise_for_status(); races=extract_races(resp.json()); read+=len(races)
            for raw in races:
                rr=normalize_race(raw)
                if not rr['external_id'] or not rr['race_date']: continue
                hs=[normalize_horse(h,rr) for h in normalize_horses(raw)]; hs=apply_arrivee(raw,hs)
                written+=upsert_participants(sb,upsert_race(sb,rr),hs)
            print(d,len(races))
        except Exception as e: errors.append(f'{d}: {e}'); print(errors[-1])
        d+=timedelta(days=1); time.sleep(delay)
    sb.table('ingestion_runs').update({'finished_at':datetime.utcnow().isoformat(),'status':'completed' if not errors else 'completed_with_errors','rows_read':read,'rows_written':written,'error_message':'\n'.join(errors)[:10000] or None}).eq('id',run).execute()
    return read,written,errors

def import_csv(sb,path):
    df=pd.read_csv(path); cols={c.lower().strip():c for c in df.columns}; out=[]
    for _,row in df.iterrows():
        def get(key):
            for a in ALIASES[key]:
                if a.lower() in cols:return row[cols[a.lower()]]
            return None
        race={k:get(k) for k in ['external_id','race_date','meeting_number','race_number','hippodrome','discipline','distance_m','terrain','field_size']}; race['external_id']=str(race['external_id']); race['race_date']=parse_date(race['race_date']); race['meeting_number']=integer(race['meeting_number']); race['race_number']=integer(race['race_number']); race['distance_m']=integer(race['distance_m']); race['field_size']=integer(race['field_size']); race['status']='finished'
        rid=upsert_race(sb,race); h={k:get(k) for k in ['external_id','horse_name','horse_number','jockey_name','trainer_name','odds','weight_kg','draw','age','sex','recent_form','finish_position','is_non_runner']}; h['race_id']=rid; h['horse_number']=integer(h['horse_number']); h['odds']=num(h['odds']); h['weight_kg']=num(h['weight_kg']); h['draw']=integer(h['draw']); h['age']=integer(h['age']); h['finish_position']=integer(h['finish_position']); h['is_non_runner']=bool(h['is_non_runner'] in (True,1,'1','true','True','oui','OUI')); out.append(h)
    for i in range(0,len(out),500): sb.table('participants').upsert(out[i:i+500],on_conflict='race_id,horse_number').execute()
    return len(out)

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('api'); a.add_argument('--start',required=True); a.add_argument('--end',required=True); a.add_argument('--delay',type=float,default=1.0)
    c=sub.add_parser('csv'); c.add_argument('path')
    x=p.parse_args(); sb=env()
    if x.mode=='api': print('Terminé:',import_api(sb,date.fromisoformat(x.start),date.fromisoformat(x.end),x.delay))
    else: print('Participants importés:',import_csv(sb,x.path))
if __name__=='__main__': main()
