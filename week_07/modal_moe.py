import modal

# ── Image: CUDA base + Python deps + gcc ──────────────────────────────────
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("gcc")
    .pip_install("torch", "transformers")
)

app = modal.App("deepseekv3-moe-operator", image=image)

# ══════════════════════════════════════════════════════════════════════════════
# Embedded source files 
# ══════════════════════════════════════════════════════════════════════════════

GENERATE_PY = r"""
import json, torch, torch.nn as nn, torch.nn.functional as F

CFG = dict(
    hidden_size=16, moe_intermediate_size=8,
    n_routed_experts=8, num_experts_per_tok=3,
    n_shared_experts=1, routed_scaling_factor=2.5,
    seq_len=4, batch_size=1,
)
WEIGHT_SEED = 42
INPUT_SEED  = 99

class DeepseekV3MLP(nn.Module):
    def __init__(self, H, IE):
        super().__init__()
        self.gate_proj = nn.Linear(H, IE, bias=False)
        self.up_proj   = nn.Linear(H, IE, bias=False)
        self.down_proj = nn.Linear(IE, H, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class DeepseekV3TopKRouter(nn.Module):
    def __init__(self, H, NE, K, scale):
        super().__init__()
        self.NE = NE; self.K = K; self.scale = scale
        self.weight = nn.Parameter(torch.empty(NE, H))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
    def forward(self, x):
        scores = F.softmax(x @ self.weight.T, dim=-1)
        tw, ti = torch.topk(scores, self.K, dim=-1, sorted=False)
        tw = tw / tw.sum(dim=-1, keepdim=True) * self.scale
        return ti, tw

class DeepseekV3MoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        H, IE = cfg["hidden_size"], cfg["moe_intermediate_size"]
        self.router = DeepseekV3TopKRouter(
            H, cfg["n_routed_experts"],
            cfg["num_experts_per_tok"], cfg["routed_scaling_factor"])
        self.shared_experts = nn.ModuleList(
            [DeepseekV3MLP(H, IE) for _ in range(cfg["n_shared_experts"])])
        self.routed_experts = nn.ModuleList(
            [DeepseekV3MLP(H, IE) for _ in range(cfg["n_routed_experts"])])
    def forward(self, x):
        B, T, H = x.shape; u = x.view(T, H)
        out = sum(e(u) for e in self.shared_experts)
        ti, tw = self.router(u)
        ro = torch.zeros_like(u)
        for t in range(T):
            for k in range(ti.shape[1]):
                eid = ti[t,k].item(); w = tw[t,k].item()
                ro[t] += w * self.routed_experts[eid](u[t].unsqueeze(0)).squeeze(0)
        return (out + ro).view(B, T, H)

def t2l(t): return t.detach().float().tolist()

def make_input(cfg):
    torch.manual_seed(INPUT_SEED)
    return torch.randn(cfg["batch_size"], cfg["seq_len"], cfg["hidden_size"])

def gen_all(cfg):
    cases = []

    # Block 1: MLP
    torch.manual_seed(WEIGHT_SEED)
    torch.use_deterministic_algorithms(True)
    m = DeepseekV3MLP(cfg["hidden_size"], cfg["moe_intermediate_size"]).eval()
    x = make_input(cfg).view(-1, cfg["hidden_size"])
    with torch.no_grad(): y = m(x)
    cases.append({"block":"DeepseekV3MLP",
        "config":{k:cfg[k] for k in ("hidden_size","moe_intermediate_size")},
        "weights":{"gate_proj":t2l(m.gate_proj.weight),
                   "up_proj":t2l(m.up_proj.weight),
                   "down_proj":t2l(m.down_proj.weight)},
        "input":t2l(x), "output":t2l(y)})

    # Block 2: Router
    torch.manual_seed(WEIGHT_SEED)
    r = DeepseekV3TopKRouter(cfg["hidden_size"], cfg["n_routed_experts"],
                              cfg["num_experts_per_tok"],
                              cfg["routed_scaling_factor"]).eval()
    x = make_input(cfg).view(-1, cfg["hidden_size"])
    with torch.no_grad(): ti, tw = r(x)
    cases.append({"block":"DeepseekV3TopKRouter",
        "config":{k:cfg[k] for k in ("hidden_size","n_routed_experts",
                                      "num_experts_per_tok","routed_scaling_factor")},
        "weights":{"router_weight":t2l(r.weight)},
        "input":t2l(x), "topk_idx":ti.tolist(), "topk_weight":t2l(tw)})

    # Block 3: Full MoE
    torch.manual_seed(WEIGHT_SEED)
    moe = DeepseekV3MoE(cfg).eval()
    x = make_input(cfg)
    with torch.no_grad(): y = moe(x)
    weights = {"router": t2l(moe.router.weight)}
    for i, e in enumerate(moe.shared_experts):
        weights[f"shared_{i}_gate"] = t2l(e.gate_proj.weight)
        weights[f"shared_{i}_up"]   = t2l(e.up_proj.weight)
        weights[f"shared_{i}_down"] = t2l(e.down_proj.weight)
    for i, e in enumerate(moe.routed_experts):
        weights[f"routed_{i}_gate"] = t2l(e.gate_proj.weight)
        weights[f"routed_{i}_up"]   = t2l(e.up_proj.weight)
        weights[f"routed_{i}_down"] = t2l(e.down_proj.weight)
    cases.append({"block":"DeepseekV3MoE", "config":cfg,
        "weights":weights,
        "input":t2l(x.view(-1, cfg["hidden_size"])),
        "output":t2l(y.view(-1, cfg["hidden_size"]))})

    return cases

if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "test_cases.json"
    cases = gen_all(CFG)
    # verify determinism
    assert cases == gen_all(CFG), "Non-deterministic!"
    with open(out, "w") as f: json.dump(cases, f, indent=2)
    print(f"Wrote {len(cases)} test cases -> {out}")
"""

# ── C implementation ─────────────────────────────────────
MOE_C = r"""
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── tiny JSON helpers ─────────────────────────────────────────────────── */
static const char *skip_ws(const char *p){
    while(*p&&(*p==' '||*p=='\n'||*p=='\r'||*p=='\t'))p++; return p;}
static const char *parse_float(const char *p,float *v){
    char b[64];int i=0;
    while(*p&&((*p>='0'&&*p<='9')||*p=='-'||*p=='.'||*p=='e'||*p=='E'||*p=='+'))b[i++]=*p++;
    b[i]=0;*v=(float)atof(b);return p;}
static const char *parse_int(const char *p,int *v){
    char b[32];int i=0;
    while(*p&&((*p>='0'&&*p<='9')||*p=='-'))b[i++]=*p++;
    b[i]=0;*v=atoi(b);return p;}
static const char *parse_float_array(const char *p,float *buf,int *n){
    p=skip_ws(p);assert(*p=='[');p++;*n=0;
    while(1){p=skip_ws(p);
        if(*p==']'){p++;break;}
        if(*p==','){p++;continue;}
        if(*p=='['){int s=0;p=parse_float_array(p,buf+*n,&s);*n+=s;}
        else p=parse_float(p,buf+(*n)++);}
    return p;}
static const char *parse_int_array(const char *p,int *buf,int *n){
    p=skip_ws(p);assert(*p=='[');p++;*n=0;
    while(1){p=skip_ws(p);
        if(*p==']'){p++;break;}
        if(*p==','){p++;continue;}
        if(*p=='['){int s=0;p=parse_int_array(p,buf+*n,&s);*n+=s;}
        else p=parse_int(p,buf+(*n)++);}
    return p;}
static const char *find_key(const char *p,const char *key){
    char needle[128];snprintf(needle,sizeof(needle),"\"%s\"",key);
    const char *f=strstr(p,needle);assert(f);
    f+=strlen(needle);f=skip_ws(f);assert(*f==':');f++;return f;}

/* ── math ──────────────────────────────────────────────────────────────── */
static void linear(const float*A,const float*x,float*y,int O,int I){
    for(int i=0;i<O;i++){float a=0;for(int j=0;j<I;j++)a+=A[i*I+j]*x[j];y[i]=a;}}
static float silu(float x){return x/(1.f+expf(-x));}
static void softmax(float*x,int n){
    float m=-FLT_MAX;for(int i=0;i<n;i++)if(x[i]>m)m=x[i];
    float s=0;for(int i=0;i<n;i++){x[i]=expf(x[i]-m);s+=x[i];}
    for(int i=0;i<n;i++)x[i]/=s;}

/* ── MLP ───────────────────────────────────────────────────────────────── */
typedef struct{int H,IE;float*gate,*up,*down;}MLP;
static void mlp_fwd(const MLP*m,const float*x,float*out){
    float*g=malloc(m->IE*sizeof(float));
    float*u=malloc(m->IE*sizeof(float));
    float*a=malloc(m->IE*sizeof(float));
    linear(m->gate,x,g,m->IE,m->H);
    linear(m->up,x,u,m->IE,m->H);
    for(int i=0;i<m->IE;i++)a[i]=silu(g[i])*u[i];
    linear(m->down,a,out,m->H,m->IE);
    free(g);free(u);free(a);}

/* ── Router ────────────────────────────────────────────────────────────── */
typedef struct{int H,NE,K;float scale,*w;}Router;
static void router_tok(const Router*r,const float*x,int*idx,float*wt){
    float*s=malloc(r->NE*sizeof(float));
    linear(r->w,x,s,r->NE,r->H);softmax(s,r->NE);
    int*picked=calloc(r->NE,sizeof(int));
    for(int k=0;k<r->K;k++){
        float best=-FLT_MAX;int bi=-1;
        for(int i=0;i<r->NE;i++)if(!picked[i]&&s[i]>best){best=s[i];bi=i;}
        idx[k]=bi;wt[k]=best;picked[bi]=1;}
    free(picked);
    float sum=0;for(int k=0;k<r->K;k++)sum+=wt[k];
    for(int k=0;k<r->K;k++)wt[k]=wt[k]/sum*r->scale;
    free(s);}

/* ── MoE ───────────────────────────────────────────────────────────────── */
typedef struct{int H,IE,NS,NR,K;float scale;Router router;MLP*shared,*routed;}MoE;
static void moe_fwd(const MoE*m,const float*x,float*out,int T){
    int H=m->H;
    float*tmp=malloc(H*sizeof(float));
    int*ti=malloc(m->K*sizeof(int));
    float*tw=malloc(m->K*sizeof(float));
    for(int t=0;t<T;t++){
        const float*xt=x+t*H;float*yt=out+t*H;
        memset(yt,0,H*sizeof(float));
        for(int s=0;s<m->NS;s++){mlp_fwd(&m->shared[s],xt,tmp);
            for(int h=0;h<H;h++)yt[h]+=tmp[h];}
        router_tok(&m->router,xt,ti,tw);
        for(int k=0;k<m->K;k++){mlp_fwd(&m->routed[ti[k]],xt,tmp);
            for(int h=0;h<H;h++)yt[h]+=tw[k]*tmp[h];}}
    free(tmp);free(ti);free(tw);}

/* ── test helpers ──────────────────────────────────────────────────────── */
#define MAX_E (1<<20)
#define TOL 1e-4f
static int check(const char*name,const float*g,const float*r,int n){
    float e=0;for(int i=0;i<n;i++){float d=fabsf(g[i]-r[i]);if(d>e)e=d;}
    if(e<TOL){printf("  %-32s PASS  (max_err=%.2e)\n",name,e);return 1;}
    printf("  %-32s FAIL  (max_err=%.2e)\n",name,e);return 0;}
static char*read_file(const char*path){
    FILE*f=fopen(path,"rb");if(!f){fprintf(stderr,"Cannot open %s\n",path);exit(1);}
    fseek(f,0,SEEK_END);long sz=ftell(f);rewind(f);
    char*buf=malloc(sz+1);fread(buf,1,sz,f);buf[sz]=0;fclose(f);return buf;}

/* ── test runners ──────────────────────────────────────────────────────── */
static int run_mlp(const char*j){
    printf("\n[Block 1] DeepseekV3MLP\n");
    int H=0,IE=0,cnt=0;
    const char*p=find_key(j,"hidden_size");p=skip_ws(p);p=parse_int(p,&H);
    p=find_key(j,"moe_intermediate_size");p=skip_ws(p);p=parse_int(p,&IE);
    float*gate=malloc(IE*H*4),*up=malloc(IE*H*4),*down=malloc(H*IE*4);
    p=find_key(j,"gate_proj");p=skip_ws(p);p=parse_float_array(p,gate,&cnt);
    p=find_key(j,"up_proj");p=skip_ws(p);p=parse_float_array(p,up,&cnt);
    p=find_key(j,"down_proj");p=skip_ws(p);p=parse_float_array(p,down,&cnt);
    float*inp=malloc(MAX_E*4),*ref=malloc(MAX_E*4);
    int ni=0,nr=0;
    p=find_key(j,"input");p=skip_ws(p);p=parse_float_array(p,inp,&ni);
    p=find_key(j,"output");p=skip_ws(p);p=parse_float_array(p,ref,&nr);
    int T=ni/H; MLP m={H,IE,gate,up,down};
    float*got=malloc(nr*4);
    for(int t=0;t<T;t++)mlp_fwd(&m,inp+t*H,got+t*H);
    int ok=check("MLP",got,ref,nr);
    free(gate);free(up);free(down);free(inp);free(ref);free(got);return ok;}

static int run_router(const char*j){
    printf("\n[Block 2] DeepseekV3TopKRouter\n");
    int H=0,NE=0,K=0,cnt=0;float scale=0;
    const char*p=find_key(j,"hidden_size");p=skip_ws(p);p=parse_int(p,&H);
    p=find_key(j,"n_routed_experts");p=skip_ws(p);p=parse_int(p,&NE);
    p=find_key(j,"num_experts_per_tok");p=skip_ws(p);p=parse_int(p,&K);
    p=find_key(j,"routed_scaling_factor");p=skip_ws(p);p=parse_float(p,&scale);
    float*rw=malloc(NE*H*4);
    p=find_key(j,"router_weight");p=skip_ws(p);p=parse_float_array(p,rw,&cnt);
    float*inp=malloc(MAX_E*4),*rw2=malloc(MAX_E*4);
    int*ri=malloc(MAX_E*4);int ni=0,nri=0,nrw=0;
    p=find_key(j,"input");p=skip_ws(p);p=parse_float_array(p,inp,&ni);
    p=find_key(j,"topk_idx");p=skip_ws(p);p=parse_int_array(p,ri,&nri);
    p=find_key(j,"topk_weight");p=skip_ws(p);p=parse_float_array(p,rw2,&nrw);
    int T=ni/H; Router r={H,NE,K,scale,rw};
    int*gi=malloc(T*K*4);float*gw=malloc(T*K*4);
    for(int t=0;t<T;t++)router_tok(&r,inp+t*H,gi+t*K,gw+t*K);
    float me=0;int io=1;
    for(int t=0;t<T;t++)for(int k=0;k<K;k++){
        int found=0;
        for(int k2=0;k2<K;k2++)if(ri[t*K+k2]==gi[t*K+k]){
            float e=fabsf(gw[t*K+k]-rw2[t*K+k2]);if(e>me)me=e;found=1;break;}
        if(!found)io=0;}
    int ok=0;
    if(!io){printf("  %-32s FAIL (idx mismatch)\n","Router");ok=0;}
    else if(me<TOL){printf("  %-32s PASS (max_w_err=%.2e)\n","Router",me);ok=1;}
    else{printf("  %-32s FAIL (max_w_err=%.2e)\n","Router",me);ok=0;}
    free(rw);free(inp);free(ri);free(rw2);free(gi);free(gw);return ok;}

static int run_moe(const char*j){
    printf("\n[Block 3] DeepseekV3MoE\n");
    int H=0,IE=0,NS=0,NR=0,K=0,cnt=0;float scale=0;
    const char*p=find_key(j,"hidden_size");p=skip_ws(p);p=parse_int(p,&H);
    p=find_key(j,"moe_intermediate_size");p=skip_ws(p);p=parse_int(p,&IE);
    p=find_key(j,"n_shared_experts");p=skip_ws(p);p=parse_int(p,&NS);
    p=find_key(j,"n_routed_experts");p=skip_ws(p);p=parse_int(p,&NR);
    p=find_key(j,"num_experts_per_tok");p=skip_ws(p);p=parse_int(p,&K);
    p=find_key(j,"routed_scaling_factor");p=skip_ws(p);p=parse_float(p,&scale);
    float*rw=malloc(NR*H*4);
    p=find_key(j,"router");p=skip_ws(p);p=parse_float_array(p,rw,&cnt);
    MLP*sh=malloc(NS*sizeof(MLP));
    for(int s=0;s<NS;s++){char key[64];
        sh[s].H=H;sh[s].IE=IE;
        sh[s].gate=malloc(IE*H*4);sh[s].up=malloc(IE*H*4);sh[s].down=malloc(H*IE*4);
        snprintf(key,64,"shared_%d_gate",s);p=find_key(j,key);p=skip_ws(p);p=parse_float_array(p,sh[s].gate,&cnt);
        snprintf(key,64,"shared_%d_up",s);p=find_key(j,key);p=skip_ws(p);p=parse_float_array(p,sh[s].up,&cnt);
        snprintf(key,64,"shared_%d_down",s);p=find_key(j,key);p=skip_ws(p);p=parse_float_array(p,sh[s].down,&cnt);}
    MLP*ro=malloc(NR*sizeof(MLP));
    for(int r2=0;r2<NR;r2++){char key[64];
        ro[r2].H=H;ro[r2].IE=IE;
        ro[r2].gate=malloc(IE*H*4);ro[r2].up=malloc(IE*H*4);ro[r2].down=malloc(H*IE*4);
        snprintf(key,64,"routed_%d_gate",r2);p=find_key(j,key);p=skip_ws(p);p=parse_float_array(p,ro[r2].gate,&cnt);
        snprintf(key,64,"routed_%d_up",r2);p=find_key(j,key);p=skip_ws(p);p=parse_float_array(p,ro[r2].up,&cnt);
        snprintf(key,64,"routed_%d_down",r2);p=find_key(j,key);p=skip_ws(p);p=parse_float_array(p,ro[r2].down,&cnt);}
    float*inp=malloc(MAX_E*4),*ref=malloc(MAX_E*4);
    int ni=0,nr=0;
    p=find_key(j,"input");p=skip_ws(p);p=parse_float_array(p,inp,&ni);
    p=find_key(j,"output");p=skip_ws(p);p=parse_float_array(p,ref,&nr);
    int T=ni/H;
    MoE moe={H,IE,NS,NR,K,scale,{H,NR,K,scale,rw},sh,ro};
    float*got=calloc(nr,4);moe_fwd(&moe,inp,got,T);
    int ok=check("MoE",got,ref,nr);
    for(int s=0;s<NS;s++){free(sh[s].gate);free(sh[s].up);free(sh[s].down);}
    for(int r2=0;r2<NR;r2++){free(ro[r2].gate);free(ro[r2].up);free(ro[r2].down);}
    free(sh);free(ro);free(rw);free(inp);free(ref);free(got);return ok;}

int main(int argc,char**argv){
    const char*path=(argc>1)?argv[1]:"test_cases.json";
    char*json=read_file(path);
    printf("=== DeepSeekV3 MoE Operator Tests ===\n");
    int passed=0;
    const char*c1=strstr(json,"\"DeepseekV3MLP\"");
    const char*c2=strstr(json,"\"DeepseekV3TopKRouter\"");
    const char*c3=strstr(json,"\"DeepseekV3MoE\"");
    assert(c1&&c2&&c3);
    while(*c1!='{')c1--;
    while(*c2!='{')c2--;
    while(*c3!='{')c3--;
    passed+=run_mlp(c1);
    passed+=run_router(c2);
    passed+=run_moe(c3);
    printf("\n=== Results: %d / 3 passed ===\n",passed);
    free(json);return(passed==3)?0:1;}
"""

# ══════════════════════════════════════════════════════════════════════════════
# Modal function
# ══════════════════════════════════════════════════════════════════════════════
@app.function(gpu="A10G", timeout=300)
def run_moe_tests():
    import subprocess, tempfile, os, sys

    with tempfile.TemporaryDirectory() as td:
        gen_py  = os.path.join(td, "gen.py")
        moe_c   = os.path.join(td, "moe.c")
        moe_bin = os.path.join(td, "moe_test")
        json_f  = os.path.join(td, "test_cases.json")

        # write source files
        with open(gen_py, "w") as f: f.write(GENERATE_PY)
        with open(moe_c,  "w") as f: f.write(MOE_C)

        # ── Step 1: generate test cases ──────────────────────────────────
        print("=" * 55)
        print("Step 1: Generating test cases (PyTorch)")
        print("=" * 55)
        r = subprocess.run(
            [sys.executable, gen_py, json_f],
            capture_output=True, text=True
        )
        print(r.stdout)
        if r.returncode != 0:
            print("ERROR:", r.stderr); return

        # ── Step 2: compile C ─────────────────────────────────────────────
        print("=" * 55)
        print("Step 2: Compiling moe.c")
        print("=" * 55)
        compile_cmd = f"gcc -O2 -std=c11 {moe_c} -o {moe_bin} -lm"
        print(f"[cmd] {compile_cmd}")
        r = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
        if r.returncode != 0:
            print("COMPILE ERROR:\n", r.stderr); return
        print("Compilation successful.")

        # ── Step 3: run tests ─────────────────────────────────────────────
        print("=" * 55)
        print("Step 3: Running C tests")
        print("=" * 55)
        r = subprocess.run(
            [moe_bin, json_f],
            capture_output=True, text=True, timeout=60
        )
        print(r.stdout)
        if r.stderr: print("stderr:", r.stderr)


# ══════════════════════════════════════════════════════════════════════════════
# Local entrypoint
# ══════════════════════════════════════════════════════════════════════════════
@app.local_entrypoint()
def main():
    print("Submitting DeepSeekV3 MoE test job to Modal …")
    run_moe_tests.remote()
