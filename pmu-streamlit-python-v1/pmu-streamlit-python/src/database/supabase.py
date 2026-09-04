import streamlit as st
from supabase import create_client

@st.cache_resource
def client():
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_KEY", st.secrets.get("SUPABASE_PUBLISHABLE_KEY", ""))
    if not url or not key: return None
    return create_client(url, key)

def is_configured(): return client() is not None

def get_courses(target_date=None, limit=100):
    q = client().table("courses").select("*, participants(count)").order("date", desc=True).order("heure")
    if target_date: q=q.eq("date", target_date)
    rows=q.limit(limit).execute().data or []
    for r in rows: r["nb_participants_stockes"]=(r.get("participants") or [{}])[0].get("count",0)
    return rows

def get_course_participants(course_id):
    return client().table("participants").select("*, courses(prix)").eq("course_id", course_id).order("prediction_score", desc=True).execute().data or []

def insert_courses(courses):
    inserted=0; sb=client()
    for c in courses:
        participants=c.pop("participants",[]) or []
        row={k:v for k,v in c.items() if k not in {"meteo","rapports"}}
        row["meteo_json"]=c.get("meteo") or {}
        row["rapports_json"]=c.get("rapports")
        existing=sb.table("courses").select("id").eq("date",row["date"]).eq("reunion_num",row["reunion_num"]).eq("course_num",row["course_num"]).limit(1).execute().data
        if existing: continue
        created=sb.table("courses").insert(row).select("id").execute().data
        if not created: continue
        cid=created[0]["id"]; inserted+=1
        payload=[]
        for p in participants:
            p=dict(p); p["course_id"]=cid
            payload.append(p)
        if payload: sb.table("participants").insert(payload).execute()
    return inserted

def get_performance_stats():
    rows=client().table("participants").select("course_id,numero,nom,cote_ref,classement,prediction_score,courses(date,ordre_arrivee)").gt("prediction_score",0).execute().data or []
    best={}
    for r in rows:
        if not r.get("courses",{}).get("ordre_arrivee"): continue
        cid=r["course_id"]
        if cid not in best or r["prediction_score"]>best[cid]["prediction_score"]: best[cid]=r
    history={}
    for r in best.values():
        d=r["courses"]["date"]; won=(r.get("classement")==1); cote=float(r.get("cote_ref") or 0); ret=cote if won else 0
        h=history.setdefault(d,{"date":d,"total_courses":0,"wins":0,"total_returns":0}); h["total_courses"]+=1; h["wins"]+=int(won); h["total_returns"]+=ret
    cumulative=0; out=[]
    for d,h in sorted(history.items()):
        profit=h["total_returns"]-h["total_courses"]; cumulative+=profit
        out.append({"date":d,"profit":round(profit,2),"cumulative":round(cumulative,2),"winRate":round(h["wins"]/h["total_courses"]*100,1)})
    total=sum(x["total_courses"] for x in history.values()); wins=sum(x["wins"] for x in history.values()); returns=sum(x["total_returns"] for x in history.values())
    return {"global":{"total_courses":total,"wins":wins,"win_rate":round(wins/total*100,1) if total else 0,"roi":round((returns/total-1)*100,1) if total else 0,"total_profit":round(cumulative,2)},"history":out}
