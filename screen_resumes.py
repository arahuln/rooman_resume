import os, json, argparse, re
import requests, pandas as pd, numpy as np
from pathlib import Path
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

U = os.getenv("NVIDIA_URL")
M = os.getenv("NVIDIA_MODEL_NAME")
API_KEY = os.getenv("NVIDIA_API_KEY")
from sentence_transformers import SentenceTransformer



def rd_pdf(p):
    try:
        r=PdfReader(p)
        t=[]
        for i in r.pages: t.append(i.extract_text() or "")
        return "\n".join(t)
    except:
        return ""

def norm(t):
    return re.sub(r'\s+',' ',t).strip()

def ssplit(t):
    s=re.split(r'(?<=[\.!\?])\s+',t)
    return [x.strip() for x in s if x and len(x.strip())>0]

def topk_tfidf(jd_txt,docs,k):
    c=[jd_txt]+docs
    v=TfidfVectorizer(lowercase=True,ngram_range=(1,2),min_df=1)
    X=v.fit_transform(c)
    s=cosine_similarity(X[0:1],X[1:]).ravel()
    idx=list(range(len(docs)))
    idx.sort(key=lambda i: s[i],reverse=True)
    return idx[:min(k,len(idx))],s

def emb_model(n):
    return SentenceTransformer(n)

def emb_sim_doc(m,jd_txt,docs):
    e_j=m.encode([jd_txt],normalize_embeddings=True)[0]
    e_d=m.encode(docs,normalize_embeddings=True)
    s=e_d@e_j
    return s,e_j,e_d

def sent_evidence(m,e_j,txt,kw=None,sent_max=120,thr=0.45,kw_thr=0.55):
    s=ssplit(txt)[:sent_max]
    if not s: return [],{}
    e_s=m.encode(s,normalize_embeddings=True)
    sc=e_s@e_j
    o_idx=np.argsort(-sc)[:5]
    ev=[s[i] for i in o_idx]
    mp={}
    if kw:
        for w in kw:
            e_w=m.encode([w],normalize_embeddings=True)[0]
            z=e_s@e_w
            j=int(np.argmax(z)) if len(z)>0 else 0
            if len(z)>0 and z[j]>=kw_thr: mp[w]=s[j]
            else: mp[w]=""
    return ev,mp

def jdump(x):
    return json.dumps(x,separators=(',',':'))

def ask(api_key,model,sys_msg,user_msg):
    h={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"}
    b={"model":model,"messages":[{"role":"system","content":sys_msg},{"role":"user","content":user_msg}],
       "temperature":0,"response_format":{"type":"json_object"}}
    r=requests.post(U,headers=h,data=json.dumps(b),timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def build_user(jd,res_txt,ev,kw_ev):
    j=jdump(jd)
    e=" || ".join(ev)[:2000]
    k=[]
    for w in jd.get("must_have_keywords",[]):
        a=kw_ev.get(w,"")
        k.append(f"{w}:{a if a else 'miss'}")
    k_str=" ; ".join(k)[:2000]
    p=("Return strict JSON only with keys: score(0-100), passes(bool), reasons(array of strings), "
       "skills:{must:[],nice:[]}, experience_years(number), education_matches(bool), red_flags([]), summary(string). "
       "Use jd.weights. If any jd.must_have_keywords missing in kw_evidence (marked miss), prefer passes=false and score<=60 unless overall evidence contradicts. "
       "Be concise. No extra keys.\n"
       f"jd={j}\nkw_evidence={k_str}\nsentence_evidence={e}\nresume=\n{res_txt[:120000]}\n")
    return p

def score_one(k,jd,res_txt,ev,kw_ev):
    sys="You are a resume-screening engine. Output JSON only. Be strict and deterministic."
    u=build_user(jd,res_txt,ev,kw_ev)
    try:
        y=ask(k,M,sys,u)
    except:
        y='{"score":0,"passes":false,"reasons":["llm_error"],"skills":{"must":[],"nice":[]},"experience_years":0,"education_matches":false,"red_flags":["llm_error"],"summary":""}'
    try:
        o=json.loads(y)
    except:
        o={"score":0,"passes":False,"reasons":["parse_error"],"skills":{"must":[],"nice":[]},
           "experience_years":0,"education_matches":False,"red_flags":["parse_error"],"summary":""}
    s=o.get("score",0)
    if isinstance(s,str):
        try:
            s=float(re.findall(r'\d+\.?\d*',s)[0]); o["score"]=s
        except:
            o["score"]=0
    return o

def load_jd(p):
    with open(p,"r",encoding="utf-8") as f: return json.load(f)

def gate_kw(jd,t):
    mh=[w.lower() for w in jd.get("must_have_keywords",[])]
    z=t.lower()
    miss=[w for w in mh if w not in z]
    return miss

def run(res_dir,jd_path,topk,out,em,alpha,sent_max):
    k=API_KEY
    jd=load_jd(jd_path)
    fs=sorted([p for p in Path(res_dir).glob("*.pdf")])
    docs=[norm(rd_pdf(str(p))) for p in fs]
    jd_txt=jdump(jd)
    idx_tfidf,sim_tfidf=topk_tfidf(jd_txt,docs,topk)
    m=emb_model(em)
    sim_trf,e_j,e_d=emb_sim_doc(m,jd_txt,docs)
    sim_trf=np.array(sim_trf).ravel()
    sim_tfidf=np.array(sim_tfidf) if len(sim_tfidf)==len(docs) else np.zeros(len(docs))
    sim_h=alpha*sim_trf+(1-alpha)*sim_tfidf
    idx=list(range(len(docs)))
    idx.sort(key=lambda i: sim_h[i],reverse=True)
    idx=idx[:min(topk,len(idx))]
    rows=[]
    for i in range(len(docs)):
        p=str(fs[i]) if i<len(fs) else f"doc_{i}.pdf"
        miss=gate_kw(jd,docs[i])
        rows.append({"file":p,"sim_trf":float(sim_trf[i])*100.0,"sim_tfidf":float(sim_tfidf[i])*100.0,"sim_h":float(sim_h[i])*100.0,"kw_missing":",".join(miss)})
    res=[]
    for i in idx:
        ev,kw_ev=sent_evidence(m,e_j,docs[i],jd.get("must_have_keywords",[]),sent_max=sent_max)
        o=score_one(k,jd,docs[i],ev,kw_ev)
        se=" || ".join(ev)
        ke=" ; ".join([f"{w}:{(kw_ev.get(w)[:140] if kw_ev.get(w) else 'miss')}" for w in jd.get("must_have_keywords",[])])
        res.append({"file":str(fs[i]),
                    "score":o.get("score",0),
                    "passes":o.get("passes",False),
                    "reasons":" | ".join(o.get("reasons",[])),
                    "must":", ".join(o.get("skills",{}).get("must",[])),
                    "nice":", ".join(o.get("skills",{}).get("nice",[])),
                    "exp_years":o.get("experience_years",0),
                    "edu_match":o.get("education_matches",False),
                    "red_flags":" | ".join(o.get("red_flags",[])),
                    "summary":o.get("summary",""),
                    "sim_trf":float(sim_trf[i])*100.0,
                    "sim_tfidf":float(sim_tfidf[i])*100.0,
                    "sim_h":float(sim_h[i])*100.0,
                    "sent_evidence":se,
                    "kw_evidence":ke})
    res.sort(key=lambda x:(x["passes"],x["score"],x["sim_h"]),reverse=True)
    df=pd.DataFrame(res)
    if out: df.to_csv(out,index=False)
    print(df.to_string(index=False,max_colwidth=60))

def demo(em):
    jd={"role":"AI Product Engineer","must_have_keywords":["python","pytorch"],"nice_to_have_keywords":["langchain","vector db","docker"],"min_years_experience":0,"location":"Bangalore","education":["B.Tech","B.E."],"weights":{"skills":50,"experience":10,"impact":20,"education":10,"extras":10}}
    r1="Rahul B.Tech CSE. Built LLM RAG with Python, PyTorch, LangChain, FAISS, Docker, AWS ECS. Deployed REST APIs. Vector search and prompt engineering."
    r2="Frontend React TypeScript Redux CSS. UI dashboards and charts. No Python. Basic Node. Some AWS S3."
    m=emb_model(em)
    s,e_j,_=emb_sim_doc(m,jdump(jd),[r1,r2])
    ev1,kw1=sent_evidence(m,e_j,r1,["python","pytorch"])
    ev2,kw2=sent_evidence(m,e_j,r2,["python","pytorch"])
    print("sims:",[float(x) for x in s])
    print("r1 ev:",ev1[:2],"kw:",kw1)
    print("r2 ev:",ev2[:2],"kw:",kw2)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--resumes",type=str,default="./resumes")
    ap.add_argument("--jd",type=str,default="./jd.json")
    ap.add_argument("--topk",type=int,default=15)
    ap.add_argument("--out",type=str,default="ranked.csv")
    ap.add_argument("--embed_model",type=str,default="all-MiniLM-L6-v2")
    ap.add_argument("--alpha",type=float,default=0.8)
    ap.add_argument("--sent_max",type=int,default=120)
    ap.add_argument("--demo",action="store_true")
    a=ap.parse_args()
    if a.demo: demo(a.embed_model)
    else: run(a.resumes,a.jd,a.topk,a.out,a.embed_model,a.alpha,a.sent_max)