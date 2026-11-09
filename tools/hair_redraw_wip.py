# 초상화 머리 다시 그리기 - 작업 중(WIP) 스크립트. HANDOFF.md 참고.
# 실행: pip install numpy opencv-python-headless scipy pillow
#       python3 tools/hair_redraw_wip.py overlay   # 설계선을 원본 위에 겹쳐 tools/out/overlay.png
#       python3 tools/hair_redraw_wip.py render    # 시안 tools/out/portrait_new.png, new_crop.png, new_small.png
# intro/portrait.webp 는 덮어쓰지 않음.
import numpy as np, cv2, sys, json
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
import os
REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S=os.path.join(REPO,'tools','out')
os.makedirs(S,exist_ok=True)
rng=np.random.default_rng(7)
src=np.array(Image.open(os.path.join(REPO,'intro','portrait.webp')))
A=src[:,:,3].astype(np.float32)/255.
H,W=A.shape
SS=3

def crspline(pts,n=400):
    p=np.array(pts,float)
    p=np.vstack([2*p[0]-p[1],p,2*p[-1]-p[-2]])
    out=[]
    for i in range(1,len(p)-2):
        P0,P1,P2,P3=p[i-1],p[i],p[i+1],p[i+2]
        for t in np.linspace(0,1,40,endpoint=False):
            t2,t3=t*t,t*t*t
            out.append(0.5*((2*P1)+(-P0+P2)*t+(2*P0-5*P1+4*P2-P3)*t2+(-P0+3*P1-3*P2+P3)*t3))
    out.append(p[-2]); out=np.array(out)
    seg=np.r_[0,np.cumsum(np.hypot(*np.diff(out,axis=0).T))]
    u=np.linspace(0,seg[-1],n)
    return np.c_[np.interp(u,seg,out[:,0]),np.interp(u,seg,out[:,1])]

class Curve:
    def __init__(s,pts): s.c=crspline(pts)
    def __call__(s,t):
        t=np.clip(np.asarray(t,float),0,1)*(len(s.c)-1)
        return np.c_[np.interp(t,np.arange(len(s.c)),s.c[:,0]),np.interp(t,np.arange(len(s.c)),s.c[:,1])]

class Patch:
    # Coons patch: v across strands (L at v=0, R at v=1), t root(0)->tip(1)
    def __init__(s,L,R,Sr,E):
        s.L,s.R,s.S,s.E=Curve(L),Curve(R),Curve(Sr),Curve(E)
        s.P00,s.P10,s.P01,s.P11=s.L(0)[0],s.R(0)[0],s.L(1)[0],s.R(1)[0]
    def __call__(s,v,t):
        v=np.asarray(v,float)[:,None]; t=np.asarray(t,float)[:,None]
        return ((1-v)*s.L(t[:,0])+v*s.R(t[:,0])+(1-t)*s.S(v[:,0])+t*s.E(v[:,0])
                -((1-v)*(1-t)*s.P00+v*(1-t)*s.P10+(1-v)*t*s.P01+v*t*s.P11))
    def outline(s,n=200):
        t=np.linspace(0,1,n)
        return np.vstack([s.S(t),s.R(t),s.E(t)[::-1],s.L(t)[::-1]])

PART_TOP=(386,98); PART_BOT=(356,196)
# 가르마 오른쪽: 오른쪽으로 쓸려 내려오는 앞머리 + 오른쪽 옆머리
M=Patch(L=[PART_BOT,(351,240),(347,290),(350,334)],
        R=[(400,86),(458,85),(513,100),(556,132),(584,178),(596,238),(595,298),(590,345),(584,378)],
        Sr=[PART_BOT,(372,142),PART_TOP,(400,90)],
        E=[(350,332),(395,334),(440,329),(485,331),(522,340),(548,362),(564,388),(574,392),(584,378)])
# 가르마 왼쪽: 왼쪽으로 넘어가 관자놀이를 감싸는 머리
K=Patch(L=[PART_TOP,(338,86),(286,92),(240,112),(204,148),(184,196),(177,250),(177,300),(180,345),(184,378)],
        R=[PART_BOT,(325,236),(292,270),(262,303),(240,340),(226,394)],
        Sr=[PART_TOP,(372,142),PART_BOT],
        E=[(184,378),(192,392),(200,398),(213,402),(226,394)])

if __name__=='__main__' and sys.argv[1]=='overlay':
    base=Image.alpha_composite(Image.new('RGBA',(W,H),(255,255,255,255)),Image.fromarray(src)).convert('RGB'); d=ImageDraw.Draw(base)
    for P,col in [(M,(220,0,0)),(K,(0,0,220))]:
        o=P.outline(); d.line([tuple(x) for x in o],fill=col,width=2)
        for v in np.linspace(0,1,9):
            pts=P(np.full(50,v),np.linspace(0,1,50)); d.line([tuple(x) for x in pts],fill=col,width=1)
    for y in range(0,H,50): d.line([(0,y),(8,y)],fill=(0,160,0)); d.text((10,y-5),str(y),fill=(0,160,0))
    for x in range(0,W,50): d.line([(x,0),(x,8)],fill=(0,160,0)); d.text((x-6,10),str(x),fill=(0,160,0))
    base.crop((100,40,680,560)).resize((1160,1040)).save(S+'/overlay.png')

def sstep(a,b,x):
    x=np.clip((np.asarray(x)-a)/(b-a),0,1); return x*x*(3-2*x)

def old_hair_mask():
    a=cv2.GaussianBlur(A,(0,0),3.5)
    m=(a>0.42).astype(np.uint8)
    yy,xx=np.mgrid[:H,:W]
    lim=np.where((xx>238)&(xx<535),359,380)
    lim=np.where((xx>=340)&(xx<=430),372,lim).astype(np.float32)
    lim=cv2.GaussianBlur(lim,(0,0),12)
    m[yy>lim]=0
    n,lab=cv2.connectedComponents(m)
    m=(lab==lab[150,380]).astype(np.uint8)
    m=ndi.binary_fill_holes(m).astype(np.uint8)
    m=cv2.dilate(m,np.ones((7,7),np.uint8))
    m=m.astype(np.float32)
    fade=1-sstep(lim-14,lim+4,yy)
    return m*fade

class Canvas:
    def __init__(s): s.im=Image.new('L',(W*SS,H*SS),0); s.d=ImageDraw.Draw(s.im)
    def stroke(s,pts,w,ink=255):
        p=np.asarray(pts)*SS
        for i in range(len(p)-1):
            wi=w[i]*SS
            if wi<0.6: continue
            s.d.line([tuple(p[i]),tuple(p[i+1])],fill=ink,width=max(1,int(round(wi))))
            if wi>=2.5:
                r=wi/2; s.d.ellipse([p[i+1][0]-r,p[i+1][1]-r,p[i+1][0]+r,p[i+1][1]+r],fill=ink)
    def alpha(s):
        return np.array(s.im.resize((W,H),Image.Resampling.BOX),np.float32)/255.

def normals(p):
    d=np.gradient(p,axis=0); n=np.c_[-d[:,1],d[:,0]]
    return n/(np.linalg.norm(n,axis=1,keepdims=True)+1e-9)

def make_clumps(n_clumps, jitter=0.5):
    w=rng.uniform(1-jitter,1+jitter,n_clumps); e=np.r_[0,np.cumsum(w)]/w.sum()
    return e

def draw_patch(cv,P,N,clump_edges,sheen,w0=1.25,pinch=0.8,tipcut=(0.06,0.14),shorten=0.05,root_t=0.0,skew=0.0):
    nC=len(clump_edges)-1
    cuts=rng.uniform(*tipcut,nC); shorts=rng.uniform(0,shorten,nC); bright=rng.uniform(0.6,1.25,nC)
    skews=rng.uniform(0.3,1.0,nC)*skew
    vs=np.sort(np.r_[rng.uniform(0,1,N),[0,0.004,0.996,1]])
    for v in vs:
        k=min(np.searchsorted(clump_edges,v,side='right')-1,nC-1)
        c=(clump_edges[k]+clump_edges[k+1])/2; h=(clump_edges[k+1]-clump_edges[k])/2
        o=(v-c)/h
        t0=root_t+(rng.uniform(0,0.05) if rng.random()<0.45 or v in (0,1) else rng.uniform(0.05,0.6))
        t1=1-shorts[k]-cuts[k]*abs(o)**1.3-rng.uniform(0,0.02)
        if v in (0,1): t1=1-shorts[k]*0.5
        if v in (0,1): ve_fix=True
        n=120
        t=np.linspace(t0,t1,n)
        ve=c+(v-c)*(1-pinch*sstep(0.6,1.0,t))+skews[k]*(2*h)*sstep(0.55,1.0,t)
        if v in (0,1): ve=np.full_like(t,v)
        p=P(np.clip(ve,-0.02,1.02),t)
        nr=normals(p)
        amp=rng.uniform(0.2,1.0); f=rng.uniform(1.5,3.5); ph=rng.uniform(0,6.3)
        wob=amp*np.sin(2*np.pi*f*t+ph)+rng.uniform(0.2,0.6)*np.sin(2*np.pi*rng.uniform(6,11)*t+rng.uniform(0,6.3))
        if v in (0,1): wob*=0.3
        p=p+nr*wob[:,None]
        u=(t-t0)/max(t1-t0,1e-6)
        taper=sstep(0,0.08,u)*0.6+0.4
        taper*=1-sstep(0.7,1.0,u)*0.92
        hl=sheen(v,t)*bright[k]
        w=w0*taper*np.clip(1-0.9*hl,0.18,1)*rng.uniform(0.8,1.15)
        if v in (0,1): w=np.maximum(w,1.0*taper)
        cv.stroke(p,w)

def texture(cv,P,N,sheen,w=0.7,ink=255,lo=0.12,hi=0.45,only_sheen=False,vrange=(0,1)):
    for i in range(N):
        v=rng.uniform(*vrange); t0=rng.uniform(0.02,0.9); L=rng.uniform(lo,hi)
        t=np.linspace(t0,min(t0+L,0.97),40)
        if only_sheen and sheen(v,t0+L/2)<0.25: continue
        dv=rng.normal(0,0.004)
        p=P(np.full_like(t,v)+dv*np.linspace(-1,1,len(t)),t)
        nr=normals(p); p=p+nr*(rng.uniform(0.2,1.2)*np.sin(2*np.pi*rng.uniform(1,3)*t+rng.uniform(0,6.3)))[:,None]
        u=np.linspace(0,1,len(t)); ww=w*np.sin(np.pi*u)**0.6*rng.uniform(0.7,1.2)
        cv.stroke(p,ww,ink)

def sheen_M(v,t):
    return 0.95*np.exp(-((t-0.30)/0.075)**2)*(1-0.55*v) + 0.3*np.exp(-((t-0.02)/0.04)**2)
def sheen_K(v,t):
    return 0.9*np.exp(-((t-0.34)/0.08)**2)*(0.5+0.5*v) + 0.3*np.exp(-((t-0.02)/0.04)**2)*v

def brow(cv,pts,thick,inner_left):
    C=Curve(pts)
    for i in range(300):
        s=rng.uniform(0,1)**0.9
        p0=C(s)[0]; th=np.interp(s,[0,0.4,1],thick)
        nn=normals(C(np.clip([s-0.01,s,s+0.01],0,1)))[1]
        off=rng.uniform(-0.5,0.5)*th
        start=p0+nn*off
        d=C(min(s+0.02,1))[0]-C(max(s-0.02,0))[0]; d/=np.linalg.norm(d)
        up=np.array([0,-1.0])
        ang_mix=np.interp(s,[0,0.35,1],[0.75,0.35,0.05])
        dirv=(1-ang_mix)*d+ang_mix*up; dirv/=np.linalg.norm(dirv)
        L=rng.uniform(5,10)*(1-0.4*s)
        q=np.linspace(0,1,8)[:,None]
        bend=nn*(q**2)*rng.uniform(-1,1)
        pts_=start+dirv*L*q+bend
        w=1.25*(1-0.75*q[:,0])*np.interp(s,[0,1],[1,0.75])
        cv.stroke(pts_,w)

def hatch_field(intensity, angle_deg=-35, spacing=2.6):
    yy,xx=np.mgrid[:H*SS,:W*SS]/SS
    th=np.deg2rad(angle_deg)
    ph=((xx*np.cos(th)+yy*np.sin(th))/spacing)%1.0
    I=cv2.resize(intensity,(W*SS,H*SS),interpolation=cv2.INTER_LINEAR)
    ink=(np.abs(ph-0.5)*2 < I).astype(np.float32)
    return cv2.resize(ink,(W,H),interpolation=cv2.INTER_AREA)

def render():
    Mold=old_hair_mask()
    hair=Canvas()
    draw_patch(hair,K,300,make_clumps(9),sheen_K,w0=0.95,pinch=0.75,tipcut=(0.05,0.12),shorten=0.05,skew=-0.35)
    draw_patch(hair,M,560,make_clumps(16),sheen_M,w0=0.95,pinch=0.8,tipcut=(0.06,0.15),shorten=0.07,skew=0.45)
    texture(hair,K,260,sheen_K); texture(hair,M,480,sheen_M)
    Ah=hair.alpha()
    hl=Canvas()
    texture(hl,K,170,sheen_K,w=0.75,lo=0.06,hi=0.22,only_sheen=True,vrange=(0.02,0.98))
    texture(hl,M,300,sheen_M,w=0.75,lo=0.06,hi=0.22,only_sheen=True,vrange=(0.02,0.98))
    texture(hl,K,90,sheen_K,w=0.6,lo=0.1,hi=0.3,vrange=(0.05,0.95))
    texture(hl,M,160,sheen_M,w=0.6,lo=0.1,hi=0.3,vrange=(0.05,0.95))
    Ah=Ah*(1-0.9*hl.alpha())
    fly=Canvas()
    for P,vv in ((M,1.0),(K,0.0)):
        for i in range(14):
            t0=rng.uniform(0.05,0.75); t=np.linspace(t0,t0+rng.uniform(0.06,0.16),30)
            out=(1 if vv==1 else -1)*np.linspace(0,rng.uniform(0.01,0.03),30)**1.5*8
            p=P(np.full_like(t,vv)+out,t)
            fly.stroke(p,0.6*np.sin(np.pi*np.linspace(0,1,30))**0.5+0.1)
    Ah=1-(1-Ah)*(1-fly.alpha())
    # 새 머리 실루엣(가르마 사이 포함)
    sil=np.zeros((H,W),np.uint8)
    for P in (M,K):
        cv2.fillPoly(sil,[np.round(P.outline()).astype(np.int32)],1)
    silf=cv2.GaussianBlur(sil.astype(np.float32),(0,0),1.2)
    brows=Canvas()
    brow(brows,[(341,323),(318,316),(292,313),(268,315),(250,321)],[10,8.5,4],True)
    brow(brows,[(429,323),(452,316),(478,313),(502,315),(520,321)],[10,8.5,4],False)
    Ab=brows.alpha()
    # 앞머리 아래 이마에 드리운 그림자(가는 사선 해칭)
    yy=np.mgrid[:H,:W][0]
    hb=cv2.GaussianBlur(np.roll(Ah,4,axis=0),(0,0),3.5)
    shadow=np.clip(hb*0.75,0,0.4)*(1-np.clip(cv2.GaussianBlur(Ah,(0,0),1.5)*3,0,1))*(1-sstep(370,385,yy))
    As=hatch_field(shadow)
    base=A*(1-Mold)*(1-np.clip(cv2.GaussianBlur(Ah,(0,0),1.5)*1.6,0,1))
    out=1-(1-base)*(1-Ab)*(1-As)
    out=1-(1-out)*(1-Ah)
    np.save(S+'/alpha_new.npy',out)
    rgba=np.zeros((H,W,4),np.uint8); rgba[:,:,:3]=src[:,:,:3]; rgba[:,:,3]=np.clip(out*255+0.5,0,255).astype(np.uint8)
    Image.fromarray(rgba).save(S+'/portrait_new.png')
    flat=Image.alpha_composite(Image.new('RGBA',(W,H),(255,255,255,255)),Image.fromarray(rgba)).convert('RGB')
    flat.save(S+'/new_flat.png')
    flat.resize((384,576),Image.Resampling.LANCZOS).save(S+'/new_small.png')
    flat.crop((120,40,640,500)).resize((1040,920),Image.Resampling.LANCZOS).save(S+'/new_crop.png')
    Image.fromarray((Mold*255).astype(np.uint8)).save(S+'/mold.png')

if __name__=='__main__' and sys.argv[1]=='render':
    render()
