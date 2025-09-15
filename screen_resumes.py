import os, json, argparse, re, time, math, random
import requests, pandas as pd
from pathlib import Path
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

U = os.getenv("NVIDIA_URL")
M = os.getenv("NVIDIA_MODEL_NAME")
API_KEY = os.getenv("NVIDIA_API_KEY")

def rd_pdf(p):
    try:
        r=PdfReader(p)
        t=[]
        for i in r.pages: t.append(i.extract_text() or "")
        return "\n".join(t)
    except:
        return ""

def norm(t):
    t=re.sub(r'\s+',' ',t).strip()
    return t

def topk_idx(jd_txt,docs,k):
    c=[jd_txt]+docs
    v=TfidfVectorizer(lowercase=True,ngram_range=(1,2),min_df=1)
    X=v.fit_transform(c)
    s=cosine_similarity(X[0:1],X[1:]).ravel()
    idx=list(range(len(docs)))
    idx.sort(key=lambda i: s[i],reverse=True)
    return idx[:min(k,len(idx))],s

def jdump(x):
    return json.dumps(x,separators=(',',':'))

def ask(api_key,model,sys_msg,user_msg,max_t=0):
    h={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"}
    b={"model":model,"messages":[{"role":"system","content":sys_msg},{"role":"user","content":user_msg}],"temperature":0,"response_format":{"type":"json_object"}}
    r=requests.post(U,headers=h,data=json.dumps(b),timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def build_user(jd,res_txt):
    j=jdump(jd)
    r=res_txt[:120000]
    p=(
    "Return strict JSON only with keys: score(0-100), passes(bool), reasons(array of strings), "
    "skills:{must:[],nice:[]}, experience_years(number), education_matches(bool), red_flags([]), summary(string). "
    "Use weights from jd.weights. If any jd.must_have_keywords missing, set passes=false and score<=60. "
    "Be concise. No extra keys.\n"
    f"jd={j}\nresume=\n{r}\n")
    return p

def score_one(api_key,model,jd,res_txt):
    sys="You are a resume-screening engine. Output JSON only. Be strict and deterministic."
    u=build_user(jd,res_txt)
    try:
        y=ask(api_key,model,sys,u)
    except Exception as e:
        y='{"score":0,"passes":false,"reasons":["llm_error"],"skills":{"must":[],"nice":[]},"experience_years":0,"education_matches":false,"red_flags":["llm_error"],"summary":""}'
    try:
        o=json.loads(y)
    except:
        o={"score":0,"passes":False,"reasons":["parse_error"],"skills":{"must":[],"nice":[]},
           "experience_years":0,"education_matches":False,"red_flags":["parse_error"],"summary":""}
    s=o.get("score",0)
    if isinstance(s,str):
        try: s=float(re.findall(r'\d+\.?\d*',s)[0]); o["score"]=s
        except: o["score"]=0
    return o

def load_jd(p):
    with open(p,"r",encoding="utf-8") as f: return json.load(f)

def gate_quick(jd,t):
    mh=set([w.lower() for w in jd.get("must_have_keywords",[])])
    z=t.lower()
    miss=[w for w in mh if w not in z]
    return miss

def run(res_dir,jd_path,topk,out):
    k=API_KEY
    jd=load_jd(jd_path)
    fs=sorted([p for p in Path(res_dir).glob("*.pdf")])
    docs=[]
    for p in fs:
        x=norm(rd_pdf(str(p)))
        docs.append(x)
    jd_txt=jdump(jd)
    pre_idx,pre_sim=topk_idx(jd_txt,docs,len(docs))  # Process ALL resumes
    rows=[]
    for i in range(len(docs)):
        p=str(fs[i]) if i<len(fs) else f"doc_{i}.pdf"
        miss=gate_quick(jd,docs[i])
        base=0.0
        if i<len(pre_sim): base=pre_sim[i]*100.0
        rows.append({"file":p,"pre_sim":base,"missing_must":",".join(miss)})
    sel=set(pre_idx)
    res=[]
    for i in range(len(docs)):  # Process ALL resumes, not just topk
        o=score_one(k,M,jd,docs[i])
        res.append({"file":str(fs[i]),"score":o.get("score",0),"passes":o.get("passes",False),
                    "reasons":" | ".join(o.get("reasons",[])),
                    "must":", ".join(o.get("skills",{}).get("must",[])),
                    "nice":", ".join(o.get("skills",{}).get("nice",[])),
                    "exp_years":o.get("experience_years",0),
                    "edu_match":o.get("education_matches",False),
                    "red_flags":" | ".join(o.get("red_flags",[])),
                    "summary":o.get("summary",""),
                    "pre_sim":pre_sim[i]*100.0 if i<len(pre_sim) else 0.0})
    res.sort(key=lambda x:(x["passes"],x["score"],x["pre_sim"]),reverse=True)
    df=pd.DataFrame(res)
    if out: df.to_csv(out,index=False)
    print(df.to_string(index=False,max_colwidth=40))

def demo():
    jd={"role":"Backend Engineer","must_have_keywords":["Python","Django","PostgreSQL"],"nice_to_have_keywords":["AWS","Docker"],"min_years_experience":1,"location":"Remote","education":["B.Tech","B.E."],"exclude_titles":["QA"],"weights":{"skills":50,"experience":25,"impact":15,"education":10}}
    r1="Rahul B.Tech CSE. 2 years Python Django, REST, PostgreSQL, AWS ECS, S3, SQS, Docker. Built scheduler and payments."
    r2="Frontend React TypeScript Redux CSS. 3 years. No Python. Some Node. Built UI dashboards."
    k=API_KEY
    i1=score_one(k,M,jd,r1)
    i2=score_one(k,M,jd,r2)
    d=pd.DataFrame([{"file":"r1.txt","score":i1.get("score",0),"passes":i1.get("passes",False),"summary":i1.get("summary","")},
                    {"file":"r2.txt","score":i2.get("score",0),"passes":i2.get("passes",False),"summary":i2.get("summary","")}])
    print(d.to_string(index=False))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--resumes",type=str,default="./resumes")
    ap.add_argument("--jd",type=str,default="./jd.json")
    ap.add_argument("--topk",type=int,default=15)
    ap.add_argument("--out",type=str,default="ranked.csv")
    ap.add_argument("--demo",action="store_true")
    a=ap.parse_args()
    if a.demo: demo()
    else: run(a.resumes,a.jd,a.topk,a.out)
