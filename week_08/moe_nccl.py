"""
modal_moe_nccl.py

Week 8 assignment - multi-GPU DeepSeekV3 MoE with NCCL
data parallelism (split tokens across GPUs) + expert parallelism (split routed experts across GPUs)

Run:
    pip install modal
    python3 -m modal setup
    modal run modal_moe_nccl.py
"""

import modal

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("gcc", "g++", "cmake")
    .run_commands(
        "apt-get update && apt-get install -y libnccl-dev libnccl2 || true"
    )
    .pip_install("torch", "transformers")
)

app = modal.App("deepseekv3-moe-nccl-week8", image=image)


# ---- test case generator (pytorch reference) ----

GENERATE_PY = r"""
import json, torch, torch.nn as nn, torch.nn.functional as F, sys, time

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
        self.NE, self.K, self.scale = NE, K, scale
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
            H, cfg["n_routed_experts"], cfg["num_experts_per_tok"],
            cfg["routed_scaling_factor"])
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
                eid, w = ti[t,k].item(), tw[t,k].item()
                ro[t] += w * self.routed_experts[eid](u[t].unsqueeze(0)).squeeze(0)
        return (out + ro).view(B, T, H)

def t2l(t): return t.detach().float().tolist()

def make_input(cfg):
    torch.manual_seed(INPUT_SEED)
    return torch.randn(cfg["batch_size"], cfg["seq_len"], cfg["hidden_size"])

def gen_moe(cfg):
    torch.manual_seed(WEIGHT_SEED)
    torch.use_deterministic_algorithms(True)
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
    return {"block": "DeepseekV3MoE", "config": cfg, "weights": weights,
            "input": t2l(x.view(-1, cfg["hidden_size"])),
            "output": t2l(y.view(-1, cfg["hidden_size"]))}

# two configs: small for quick sanity check, large for actual perf testing
CFGS = {
    "small": dict(hidden_size=16, moe_intermediate_size=8,
                  n_routed_experts=8, num_experts_per_tok=3,
                  n_shared_experts=1, routed_scaling_factor=2.5,
                  seq_len=4, batch_size=1),
    "large": dict(hidden_size=128, moe_intermediate_size=64,
                  n_routed_experts=8, num_experts_per_tok=3,
                  n_shared_experts=2, routed_scaling_factor=2.5,
                  seq_len=256, batch_size=1),
}

if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    for name, cfg in CFGS.items():
        tc = gen_moe(cfg)
        path = f"{out_dir}/test_{name}.json"
        with open(path, "w") as f: json.dump(tc, f)
        print(f"  Wrote {path}  (T={cfg['seq_len']})")
"""


# ---- CUDA + NCCL multi-gpu MoE ----

MOE_NCCL_CU = r"""
/*
 * Multi-GPU DeepSeekV3 MoE with NCCL
 *
 * The idea: split input tokens across GPUs (data parallel) and
 * split routed experts across GPUs (expert parallel).
 * Shared experts + router are just replicated on every GPU.
 *
 * Forward pass follows the 6 phases from lecture:
 *   1) shared expert compute (local, no comm needed)
 *   2) routing (local, replicated router)
 *   3) permutation - all-to-all to send tokens to the right GPU
 *   4) expert compute on received tokens
 *   5) un-permutation - all-to-all to send results back
 *   6) scale + combine: output = shared_out + sum(wk * expert_k(x))
 */

#include <cuda_runtime.h>
#include <nccl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){        \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,        \
    cudaGetErrorString(e));exit(1);}}while(0)
#define NK(x) do{ncclResult_t e=(x);if(e!=ncclSuccess){        \
    fprintf(stderr,"NCCL %s:%d %s\n",__FILE__,__LINE__,        \
    ncclGetErrorString(e));exit(1);}}while(0)

/* --- json parsing (mostly reused from week 07 with minor tweaks) --- */
static const char*S(const char*p){
    while(*p&&(*p==' '||*p=='\n'||*p=='\r'||*p=='\t'))p++;return p;}
static const char*PF(const char*p,float*v){
    char b[64];int i=0;
    while(*p&&((*p>='0'&&*p<='9')||*p=='-'||*p=='.'||*p=='e'||*p=='E'||*p=='+'))
        b[i++]=*p++;b[i]=0;*v=(float)atof(b);return p;}
static const char*PI(const char*p,int*v){
    char b[32];int i=0;
    while(*p&&((*p>='0'&&*p<='9')||*p=='-'))b[i++]=*p++;
    b[i]=0;*v=atoi(b);return p;}
static const char*PFA(const char*p,float*buf,int*n){
    p=S(p);assert(*p=='[');p++;*n=0;
    while(1){p=S(p);if(*p==']'){p++;break;}
    if(*p==','){p++;continue;}
    if(*p=='['){int s=0;p=PFA(p,buf+*n,&s);*n+=s;}
    else p=PF(p,buf+(*n)++);}return p;}
static const char*FK(const char*p,const char*key){
    char nd[128];snprintf(nd,128,"\"%s\"",key);
    const char*f=strstr(p,nd);assert(f);
    f+=strlen(nd);f=S(f);assert(*f==':');f++;return f;}
static char*read_file(const char*path){
    FILE*f=fopen(path,"rb");if(!f){fprintf(stderr,"can't open %s\n",path);exit(1);}
    fseek(f,0,SEEK_END);long sz=ftell(f);rewind(f);
    char*b=(char*)malloc(sz+1);fread(b,1,sz,f);b[sz]=0;fclose(f);return b;}

/* --- cuda kernels --- */

// batched matrix-vector: Y[b,o] = sum_i W[o,i]*X[b,i]
__global__ void linear_k(const float*W,const float*X,float*Y,
                         int B,int O,int I){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    int b=idx/O, o=idx%O;
    if(b>=B) return;
    float acc=0; for(int i=0;i<I;i++) acc+=W[o*I+i]*X[b*I+i];
    Y[b*O+o]=acc;
}

__global__ void silu_mul_k(float*g,const float*u,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float v=g[i]; g[i]=(v/(1.f+expf(-v)))*u[i];
}

__global__ void add_k(float*a,const float*b,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    a[i]+=b[i];
}

// one block per token row
__global__ void softmax_k(float*S,int T,int N){
    int t=blockIdx.x; if(t>=T) return;
    float*r=S+t*N; float m=-FLT_MAX;
    for(int i=0;i<N;i++) m=fmaxf(m,r[i]);
    float s=0; for(int i=0;i<N;i++){r[i]=expf(r[i]-m);s+=r[i];}
    for(int i=0;i<N;i++) r[i]/=s;
}

// greedy top-k per token, then normalize and scale
__global__ void topk_k(const float*Sc,int*idx,float*wt,
                       int T,int N,int K,float sc){
    int t=blockIdx.x; if(t>=T) return;
    const float*r=Sc+t*N; int*ti=idx+t*K; float*tw=wt+t*K;
    int pk[128]={};
    for(int k=0;k<K;k++){
        float best=-FLT_MAX; int bi=0;
        for(int i=0;i<N;i++) if(!pk[i]&&r[i]>best){best=r[i];bi=i;}
        ti[k]=bi; tw[k]=best; pk[bi]=1;
    }
    float s=0; for(int k=0;k<K;k++) s+=tw[k];
    for(int k=0;k<K;k++) tw[k]=tw[k]/s*sc;
}

/* --- GPU MLP (SwiGLU) --- */
typedef struct{ float*dg,*du,*dd; int H,IE; } GpuMLP;

static void mlp_upload(GpuMLP*m,const float*g,const float*u,
                       const float*d,int H,int IE){
    m->H=H; m->IE=IE;
    CK(cudaMalloc(&m->dg,(size_t)IE*H*4));
    CK(cudaMemcpy(m->dg,g,(size_t)IE*H*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&m->du,(size_t)IE*H*4));
    CK(cudaMemcpy(m->du,u,(size_t)IE*H*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&m->dd,(size_t)H*IE*4));
    CK(cudaMemcpy(m->dd,d,(size_t)H*IE*4,cudaMemcpyHostToDevice));
}
static void mlp_free(GpuMLP*m){
    cudaFree(m->dg);cudaFree(m->du);cudaFree(m->dd);}

// run one SwiGLU MLP forward on T tokens
static void mlp_fwd(GpuMLP*m,const float*d_in,float*d_out,
                    int T,cudaStream_t st){
    int H=m->H, IE=m->IE, BK=256;
    float*dg2,*du2;
    CK(cudaMalloc(&dg2,(size_t)T*IE*4));
    CK(cudaMalloc(&du2,(size_t)T*IE*4));
    int n1=T*IE, n2=T*H;
    linear_k<<<(n1+BK-1)/BK,BK,0,st>>>(m->dg,d_in,dg2,T,IE,H);
    linear_k<<<(n1+BK-1)/BK,BK,0,st>>>(m->du,d_in,du2,T,IE,H);
    silu_mul_k<<<(n1+BK-1)/BK,BK,0,st>>>(dg2,du2,n1);
    linear_k<<<(n2+BK-1)/BK,BK,0,st>>>(m->dd,dg2,d_out,T,H,IE);
    cudaFree(dg2); cudaFree(du2);
}

/* --- per-gpu state --- */
typedef struct{
    int rank,W,H,IE,NR,NS,K; float scale;
    float*d_rtr;        // router weights [NR,H] on device
    GpuMLP *shared;     // shared experts (replicated on all GPUs)
    GpuMLP *local_exp;  // this GPU's partition of routed experts
    int LE;             // num local experts = NR / W
    ncclComm_t comm;
    cudaStream_t st;
} Ctx;

/* ---------------------------------------------------------------
 * the main distributed forward pass
 * --------------------------------------------------------------- */
static void moe_fwd_dist(Ctx*c,const float*h_in,float*h_out,int lT){
    int H=c->H, K=c->K, NR=c->NR, W=c->W, LE=c->LE, BK=256;
    int n = lT*H;

    // upload this GPU's local input chunk
    float *d_in, *d_out;
    CK(cudaMalloc(&d_in, (size_t)n*4));
    CK(cudaMemcpy(d_in, h_in, (size_t)n*4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_out,(size_t)n*4));
    CK(cudaMemset(d_out, 0, (size_t)n*4));

    // --- phase 1: shared experts (no communication needed) ---
    // each gpu just runs shared experts on its own tokens
    float*d_tmp; CK(cudaMalloc(&d_tmp,(size_t)n*4));
    for(int s=0;s<c->NS;s++){
        mlp_fwd(&c->shared[s], d_in, d_tmp, lT, c->st);
        add_k<<<(n+BK-1)/BK,BK,0,c->st>>>(d_out, d_tmp, n);
    }
    CK(cudaFree(d_tmp));

    // --- phase 2: routing (also local, router is replicated) ---
    int ns=lT*NR;
    float*d_sc; CK(cudaMalloc(&d_sc,(size_t)ns*4));
    linear_k<<<(ns+BK-1)/BK,BK,0,c->st>>>(c->d_rtr,d_in,d_sc,lT,NR,H);
    softmax_k<<<lT,1,0,c->st>>>(d_sc, lT, NR);

    int*d_idx; float*d_wt;
    CK(cudaMalloc(&d_idx,(size_t)lT*K*4));
    CK(cudaMalloc(&d_wt, (size_t)lT*K*4));
    topk_k<<<lT,1,0,c->st>>>(d_sc, d_idx, d_wt, lT, NR, K, c->scale);
    CK(cudaStreamSynchronize(c->st));
    CK(cudaFree(d_sc));

    // need routing decisions on host to figure out the permutation
    int   *h_idx = (int*)  malloc((size_t)lT*K*4);
    float *h_wt  = (float*)malloc((size_t)lT*K*4);
    CK(cudaMemcpy(h_idx, d_idx, (size_t)lT*K*4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_wt,  d_wt,  (size_t)lT*K*4, cudaMemcpyDeviceToHost));
    CK(cudaFree(d_idx)); CK(cudaFree(d_wt));

    // --- phase 3: permutation (all-to-all) ---
    // figure out which tokens need to go to which GPU based on expert assignment
    // dest gpu for expert e = e / LE
    int max_s = lT * K;   // upper bound on sends per destination
    int msg   = H + 3;    // each message = token_data(H floats) + local_expert_id + weight + original_token_idx

    int *scnt = (int*)calloc(W, 4);  // how many items we send to each gpu

    typedef struct{int tok,el; float wt;} SendItem;
    SendItem **sitems = (SendItem**)malloc((size_t)W*sizeof(SendItem*));
    for(int w=0;w<W;w++) sitems[w]=(SendItem*)malloc((size_t)max_s*sizeof(SendItem));

    for(int t=0;t<lT;t++) for(int k=0;k<K;k++){
        int eid=h_idx[t*K+k];
        int dest=eid/LE;       // which gpu owns this expert
        int el=eid%LE;         // local index on that gpu
        sitems[dest][scnt[dest]++] = (SendItem){t, el, h_wt[t*K+k]};
    }
    free(h_idx); free(h_wt);

    // exchange send counts so everyone knows how much they'll receive
    int *d_sc2, *d_ac;
    CK(cudaMalloc(&d_sc2,(size_t)W*4));
    CK(cudaMalloc(&d_ac, (size_t)W*W*4));
    CK(cudaMemcpy(d_sc2, scnt, (size_t)W*4, cudaMemcpyHostToDevice));
    NK(ncclAllGather(d_sc2, d_ac, W, ncclInt, c->comm, c->st));
    CK(cudaStreamSynchronize(c->st));

    int *h_ac = (int*)malloc((size_t)W*W*4);
    CK(cudaMemcpy(h_ac, d_ac, (size_t)W*W*4, cudaMemcpyDeviceToHost));
    CK(cudaFree(d_sc2)); CK(cudaFree(d_ac));

    // rcnt[w] = number of items gpu w is sending to me
    int *rcnt = (int*)malloc((size_t)W*4);
    for(int w=0;w<W;w++) rcnt[w] = h_ac[w*W + c->rank];
    free(h_ac);

    // pack the send buffer: for each dest gpu, lay out the tokens + metadata
    float *h_inp = (float*)malloc((size_t)n*4);
    CK(cudaMemcpy(h_inp, d_in, (size_t)n*4, cudaMemcpyDeviceToHost));

    size_t a2a_n  = (size_t)W * max_s * msg;
    float *h_send = (float*)calloc(a2a_n, 4);

    for(int w=0;w<W;w++) for(int i=0;i<scnt[w];i++){
        SendItem *it = &sitems[w][i];
        float *slot = &h_send[((size_t)w*max_s + i) * msg];
        memcpy(slot, &h_inp[(size_t)it->tok*H], (size_t)H*4);  // token vector
        slot[H]   = (float)it->el;   // local expert id
        slot[H+1] = it->wt;          // routing weight
        slot[H+2] = (float)it->tok;  // so we know where to put the result back
    }
    free(h_inp);
    for(int w=0;w<W;w++) free(sitems[w]); free(sitems);

    // do the actual all-to-all with NCCL send/recv
    float *d_send, *d_recv;
    CK(cudaMalloc(&d_send, a2a_n*4));
    CK(cudaMalloc(&d_recv, a2a_n*4));
    CK(cudaMemcpy(d_send, h_send, a2a_n*4, cudaMemcpyHostToDevice));
    free(h_send);

    NK(ncclGroupStart());
    for(int w=0;w<W;w++){
        size_t chunk = (size_t)max_s * msg;
        NK(ncclSend(d_send + (size_t)w*chunk, (int)chunk, ncclFloat, w, c->comm, c->st));
        NK(ncclRecv(d_recv + (size_t)w*chunk, (int)chunk, ncclFloat, w, c->comm, c->st));
    }
    NK(ncclGroupEnd());
    CK(cudaStreamSynchronize(c->st));
    CK(cudaFree(d_send));

    // --- phase 4: expert computation ---
    // unpack received tokens, group by local expert, run MLP, store results
    float *h_recv = (float*)malloc(a2a_n*4);
    CK(cudaMemcpy(h_recv, d_recv, a2a_n*4, cudaMemcpyDeviceToHost));
    CK(cudaFree(d_recv));

    int total_recv = 0;
    for(int w=0;w<W;w++) total_recv += rcnt[w];

    // flatten everything into arrays for easier processing
    float *all_tok = (float*)malloc((size_t)total_recv*H*4);
    int   *all_eid = (int*)  malloc((size_t)total_recv*4);
    float *all_wt  = (float*)malloc((size_t)total_recv*4);
    int   *all_orig= (int*)  malloc((size_t)total_recv*4);

    int fi=0;
    for(int w=0;w<W;w++) for(int i=0;i<rcnt[w];i++){
        float *m2 = &h_recv[((size_t)w*max_s + i)*msg];
        memcpy(&all_tok[(size_t)fi*H], m2, (size_t)H*4);
        all_eid[fi] = (int)m2[H];
        all_wt[fi]  = m2[H+1];
        all_orig[fi]= (int)m2[H+2];
        fi++;
    }
    free(h_recv);

    // run each local expert on its batch of tokens
    float *all_res = (float*)calloc((size_t)total_recv*H, 4);
    for(int e=0;e<LE;e++){
        // count how many tokens go to this expert
        int cnt=0;
        for(int i=0;i<total_recv;i++) if(all_eid[i]==e) cnt++;
        if(!cnt) continue;

        // gather them into a batch
        float *batch = (float*)malloc((size_t)cnt*H*4);
        int   *bidx  = (int*)  malloc((size_t)cnt*4);
        int bi=0;
        for(int i=0;i<total_recv;i++) if(all_eid[i]==e){
            memcpy(&batch[(size_t)bi*H], &all_tok[(size_t)i*H], (size_t)H*4);
            bidx[bi++]=i;
        }

        float *d_ei, *d_eo;
        CK(cudaMalloc(&d_ei,(size_t)cnt*H*4));
        CK(cudaMalloc(&d_eo,(size_t)cnt*H*4));
        CK(cudaMemcpy(d_ei, batch, (size_t)cnt*H*4, cudaMemcpyHostToDevice));
        mlp_fwd(&c->local_exp[e], d_ei, d_eo, cnt, c->st);
        CK(cudaStreamSynchronize(c->st));

        float *bout = (float*)malloc((size_t)cnt*H*4);
        CK(cudaMemcpy(bout, d_eo, (size_t)cnt*H*4, cudaMemcpyDeviceToHost));
        for(int j=0;j<cnt;j++)
            memcpy(&all_res[(size_t)bidx[j]*H], &bout[(size_t)j*H], (size_t)H*4);

        CK(cudaFree(d_ei)); CK(cudaFree(d_eo));
        free(batch); free(bidx); free(bout);
    }
    free(all_tok); free(all_eid);

    // --- phase 5: un-permutation (send results back) ---
    // pack results + weight + orig_tok, then all-to-all back to originators
    int rsz = H + 2;  // result vector + weight + orig token id
    size_t r_a2a_n = (size_t)W * max_s * rsz;
    float *h_rs = (float*)calloc(r_a2a_n, 4);

    fi=0;
    for(int w=0;w<W;w++) for(int i=0;i<rcnt[w];i++){
        float *slot = &h_rs[((size_t)w*max_s + i)*rsz];
        memcpy(slot, &all_res[(size_t)fi*H], (size_t)H*4);
        slot[H]   = all_wt[fi];
        slot[H+1] = (float)all_orig[fi];
        fi++;
    }
    free(all_res); free(all_wt); free(all_orig);

    float *d_rs, *d_rr;
    CK(cudaMalloc(&d_rs, r_a2a_n*4));
    CK(cudaMalloc(&d_rr, r_a2a_n*4));
    CK(cudaMemcpy(d_rs, h_rs, r_a2a_n*4, cudaMemcpyHostToDevice));
    free(h_rs);

    NK(ncclGroupStart());
    for(int w=0;w<W;w++){
        size_t chunk = (size_t)max_s * rsz;
        NK(ncclSend(d_rs + (size_t)w*chunk, (int)chunk, ncclFloat, w, c->comm, c->st));
        NK(ncclRecv(d_rr + (size_t)w*chunk, (int)chunk, ncclFloat, w, c->comm, c->st));
    }
    NK(ncclGroupEnd());
    CK(cudaStreamSynchronize(c->st));
    CK(cudaFree(d_rs));

    // --- phase 6: combine shared + weighted routed results ---
    // d_out already has shared expert output from phase 1
    // now we just accumulate the routed part
    float *h_rr = (float*)malloc(r_a2a_n*4);
    CK(cudaMemcpy(h_rr, d_rr, r_a2a_n*4, cudaMemcpyDeviceToHost));
    CK(cudaFree(d_rr));

    float *h_routed = (float*)calloc((size_t)n, 4);
    for(int w=0;w<W;w++) for(int i=0;i<scnt[w];i++){
        float *res = &h_rr[((size_t)w*max_s + i)*rsz];
        float wt   = res[H];
        int   orig = (int)res[H+1];
        for(int h=0;h<H;h++) h_routed[(size_t)orig*H+h] += wt * res[h];
    }
    free(h_rr);

    // add routed contribution on top of shared output
    float *d_rt; CK(cudaMalloc(&d_rt,(size_t)n*4));
    CK(cudaMemcpy(d_rt, h_routed, (size_t)n*4, cudaMemcpyHostToDevice));
    add_k<<<(n+BK-1)/BK,BK,0,c->st>>>(d_out, d_rt, n);
    CK(cudaStreamSynchronize(c->st));
    CK(cudaFree(d_rt)); free(h_routed);

    // done, copy back
    CK(cudaMemcpy(h_out, d_out, (size_t)n*4, cudaMemcpyDeviceToHost));
    CK(cudaFree(d_in)); CK(cudaFree(d_out));
    free(scnt); free(rcnt);
}

/* --- test case parser --- */

typedef struct{float*g,*u,*d;} Wts;
typedef struct{
    int H,IE,NR,NS,K,T; float scale;
    float*rtr; Wts*shared; Wts*routed;
    float*input; float*output;
} TC;

static TC parse_tc(const char*j){
    TC tc; int cnt; const char*p;
    p=FK(j,"hidden_size");           p=S(p); p=PI(p,&tc.H);
    p=FK(j,"moe_intermediate_size"); p=S(p); p=PI(p,&tc.IE);
    p=FK(j,"n_shared_experts");      p=S(p); p=PI(p,&tc.NS);
    p=FK(j,"n_routed_experts");      p=S(p); p=PI(p,&tc.NR);
    p=FK(j,"num_experts_per_tok");   p=S(p); p=PI(p,&tc.K);
    p=FK(j,"routed_scaling_factor"); p=S(p); p=PF(p,&tc.scale);
    p=FK(j,"seq_len");               p=S(p); p=PI(p,&tc.T);

    tc.rtr=(float*)malloc((size_t)tc.NR*tc.H*4);
    p=FK(j,"router"); p=S(p); p=PFA(p,tc.rtr,&cnt);

    tc.shared=(Wts*)malloc((size_t)tc.NS*sizeof(Wts));
    for(int s=0;s<tc.NS;s++){char k[64];
        tc.shared[s].g=(float*)malloc((size_t)tc.IE*tc.H*4);
        tc.shared[s].u=(float*)malloc((size_t)tc.IE*tc.H*4);
        tc.shared[s].d=(float*)malloc((size_t)tc.H*tc.IE*4);
        snprintf(k,64,"shared_%d_gate",s);
        p=FK(j,k);p=S(p);p=PFA(p,tc.shared[s].g,&cnt);
        snprintf(k,64,"shared_%d_up",s);
        p=FK(j,k);p=S(p);p=PFA(p,tc.shared[s].u,&cnt);
        snprintf(k,64,"shared_%d_down",s);
        p=FK(j,k);p=S(p);p=PFA(p,tc.shared[s].d,&cnt);
    }
    tc.routed=(Wts*)malloc((size_t)tc.NR*sizeof(Wts));
    for(int r=0;r<tc.NR;r++){char k[64];
        tc.routed[r].g=(float*)malloc((size_t)tc.IE*tc.H*4);
        tc.routed[r].u=(float*)malloc((size_t)tc.IE*tc.H*4);
        tc.routed[r].d=(float*)malloc((size_t)tc.H*tc.IE*4);
        snprintf(k,64,"routed_%d_gate",r);
        p=FK(j,k);p=S(p);p=PFA(p,tc.routed[r].g,&cnt);
        snprintf(k,64,"routed_%d_up",r);
        p=FK(j,k);p=S(p);p=PFA(p,tc.routed[r].u,&cnt);
        snprintf(k,64,"routed_%d_down",r);
        p=FK(j,k);p=S(p);p=PFA(p,tc.routed[r].d,&cnt);
    }
    int ni=0,no=0;
    tc.input =(float*)malloc((size_t)tc.T*tc.H*4);
    tc.output=(float*)malloc((size_t)tc.T*tc.H*4);
    p=FK(j,"input"); p=S(p); p=PFA(p,tc.input,&ni);
    p=FK(j,"output");p=S(p); p=PFA(p,tc.output,&no);
    return tc;
}

static void free_tc(TC*tc){
    free(tc->rtr);
    for(int s=0;s<tc->NS;s++){free(tc->shared[s].g);free(tc->shared[s].u);free(tc->shared[s].d);}
    for(int r=0;r<tc->NR;r++){free(tc->routed[r].g);free(tc->routed[r].u);free(tc->routed[r].d);}
    free(tc->shared);free(tc->routed);free(tc->input);free(tc->output);
}

/* --- thread worker (one per GPU) --- */

typedef struct{
    int rank, W;
    ncclUniqueId id;
    TC *tc;
    float max_err;
    double fwd_ms;
} TArg;

void* worker(void*a){
    TArg *arg = (TArg*)a;
    int rank=arg->rank, W=arg->W;
    TC *tc = arg->tc;
    CK(cudaSetDevice(rank));

    ncclComm_t comm;
    NK(ncclCommInitRank(&comm, W, arg->id, rank));

    cudaStream_t st;
    CK(cudaStreamCreate(&st));

    int H=tc->H, IE=tc->IE, NR=tc->NR, NS=tc->NS, K=tc->K;
    int LE = NR / W;   // local experts per gpu
    int lT = tc->T / W; // local tokens per gpu

    // router is replicated on every gpu
    float *d_rtr;
    CK(cudaMalloc(&d_rtr, (size_t)NR*H*4));
    CK(cudaMemcpy(d_rtr, tc->rtr, (size_t)NR*H*4, cudaMemcpyHostToDevice));

    // shared experts: also replicated
    GpuMLP *shared = (GpuMLP*)malloc((size_t)NS*sizeof(GpuMLP));
    for(int s=0;s<NS;s++)
        mlp_upload(&shared[s], tc->shared[s].g, tc->shared[s].u,
                   tc->shared[s].d, H, IE);

    // routed experts: each gpu only loads its partition
    // gpu 0 gets experts [0..LE-1], gpu 1 gets [LE..2*LE-1], etc.
    GpuMLP *local_exp = (GpuMLP*)malloc((size_t)LE*sizeof(GpuMLP));
    for(int e=0;e<LE;e++){
        int global_eid = rank*LE + e;
        mlp_upload(&local_exp[e], tc->routed[global_eid].g, tc->routed[global_eid].u,
                   tc->routed[global_eid].d, H, IE);
    }

    Ctx ctx = {rank, W, H, IE, NR, NS, K, tc->scale,
               d_rtr, shared, local_exp, LE, comm, st};

    // each gpu gets its slice of the input
    float *local_in  = tc->input + (size_t)rank * lT * H;
    float *local_out = (float*)calloc((size_t)lT*H, 4);

    // warmup
    moe_fwd_dist(&ctx, local_in, local_out, lT);

    // timed run
    cudaEvent_t ev0, ev1;
    CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));
    CK(cudaEventRecord(ev0, st));
    moe_fwd_dist(&ctx, local_in, local_out, lT);
    CK(cudaEventRecord(ev1, st));
    CK(cudaEventSynchronize(ev1));
    float ms; CK(cudaEventElapsedTime(&ms, ev0, ev1));
    arg->fwd_ms = ms;

    // check against pytorch reference
    float *ref = tc->output + (size_t)rank * lT * H;
    float me = 0;
    for(int i=0;i<lT*H;i++){
        float d2 = fabsf(local_out[i]-ref[i]);
        if(d2>me) me=d2;
    }
    arg->max_err = me;

    free(local_out);
    CK(cudaFree(d_rtr));
    for(int s=0;s<NS;s++) mlp_free(&shared[s]);
    for(int e=0;e<LE;e++) mlp_free(&local_exp[e]);
    free(shared); free(local_exp);
    CK(cudaEventDestroy(ev0)); CK(cudaEventDestroy(ev1));
    ncclCommDestroy(comm);
    CK(cudaStreamDestroy(st));
    return NULL;
}

/* --- main: parse json, spawn gpu threads, check results --- */

int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"Usage: %s <test.json>\n",argv[0]);return 1;}

    int nGPU; CK(cudaGetDeviceCount(&nGPU));
    int W = (nGPU>=2) ? 2 : 1;
    printf("GPUs available: %d, using: %d\n", nGPU, W);

    char *json = read_file(argv[1]);
    TC tc = parse_tc(json);
    printf("Config: H=%d IE=%d NR=%d NS=%d K=%d T=%d scale=%.2f\n",
           tc.H,tc.IE,tc.NR,tc.NS,tc.K,tc.T,tc.scale);

    if(tc.T % W){printf("T must be divisible by num GPUs\n");return 1;}
    if(tc.NR% W){printf("NR must be divisible by num GPUs\n");return 1;}

    ncclUniqueId id; NK(ncclGetUniqueId(&id));

    pthread_t thr[8]; TArg args[8];
    for(int i=0;i<W;i++){
        args[i] = (TArg){i, W, id, &tc, 0, 0};
        pthread_create(&thr[i], NULL, worker, &args[i]);
    }
    for(int i=0;i<W;i++) pthread_join(thr[i],NULL);

    float max_err=0; double max_ms=0;
    for(int i=0;i<W;i++){
        printf("  GPU %d: max_err=%.6e  time=%.3f ms\n",
               i, args[i].max_err, args[i].fwd_ms);
        if(args[i].max_err>max_err) max_err=args[i].max_err;
        if(args[i].fwd_ms>max_ms)   max_ms =args[i].fwd_ms;
    }

    float tol=5e-3f;
    int pass = max_err < tol;
    printf("\n%s  (max_err=%.6e  tol=%.1e  gpu_time=%.3f ms)\n",
           pass?"PASS":"FAIL", max_err, tol, max_ms);

    free_tc(&tc); free(json);
    return pass ? 0 : 1;
}
"""


# ---- pytorch benchmark for perf comparison ----

BENCHMARK_PY = r"""
import torch, torch.nn as nn, torch.nn.functional as F, time

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
        self.NE, self.K, self.scale = NE, K, scale
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
            H, cfg["n_routed_experts"], cfg["num_experts_per_tok"],
            cfg["routed_scaling_factor"])
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
                eid, w = ti[t,k].item(), tw[t,k].item()
                ro[t] += w * self.routed_experts[eid](u[t].unsqueeze(0)).squeeze(0)
        return (out + ro).view(B, T, H)

def bench_one(cfg, device='cuda', n_runs=50):
    torch.manual_seed(WEIGHT_SEED)
    moe = DeepseekV3MoE(cfg).eval().to(device)
    torch.manual_seed(INPUT_SEED)
    x = torch.randn(cfg['batch_size'], cfg['seq_len'], cfg['hidden_size']).to(device)

    # warmup
    for _ in range(10):
        with torch.no_grad(): moe(x)
    if device=='cuda': torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_runs):
        with torch.no_grad(): moe(x)
    if device=='cuda': torch.cuda.synchronize()
    return (time.time()-t0)/n_runs*1000

CFGS = {
    "small": dict(hidden_size=16, moe_intermediate_size=8,
                  n_routed_experts=8, num_experts_per_tok=3,
                  n_shared_experts=1, routed_scaling_factor=2.5,
                  seq_len=4, batch_size=1),
    "large": dict(hidden_size=128, moe_intermediate_size=64,
                  n_routed_experts=8, num_experts_per_tok=3,
                  n_shared_experts=2, routed_scaling_factor=2.5,
                  seq_len=256, batch_size=1),
    "xlarge": dict(hidden_size=512, moe_intermediate_size=256,
                   n_routed_experts=16, num_experts_per_tok=4,
                   n_shared_experts=2, routed_scaling_factor=2.5,
                   seq_len=1024, batch_size=1),
}

if __name__=="__main__":
    print("=" * 55)
    print("PyTorch MoE benchmark")
    print("=" * 55)
    for name, cfg in CFGS.items():
        try:
            ms = bench_one(cfg)
            print(f"  {name:8s}: {ms:8.3f} ms  "
                  f"(T={cfg['seq_len']}, H={cfg['hidden_size']}, "
                  f"NR={cfg['n_routed_experts']})")
        except Exception as e:
            print(f"  {name:8s}: SKIPPED ({e})")
"""


# ---- modal function: generate, compile, test, benchmark ----

@app.function(gpu="A10G:2", timeout=600)
def run_moe_nccl():
    import subprocess, tempfile, os, sys

    with tempfile.TemporaryDirectory() as td:
        gen_py  = os.path.join(td, "gen.py")
        moe_cu  = os.path.join(td, "moe_nccl.cu")
        moe_bin = os.path.join(td, "moe_nccl")
        bench   = os.path.join(td, "bench.py")

        with open(gen_py, "w") as f:  f.write(GENERATE_PY)
        with open(moe_cu, "w") as f:  f.write(MOE_NCCL_CU)
        with open(bench,  "w") as f:  f.write(BENCHMARK_PY)

        # step 1: generate test cases with pytorch
        print("=" * 55)
        print("Step 1: generating test cases")
        print("=" * 55)
        r = subprocess.run([sys.executable, gen_py, td],
                           capture_output=True, text=True)
        print(r.stdout)
        if r.returncode != 0:
            print("gen error:", r.stderr); return

        # step 2: compile cuda+nccl
        print("=" * 55)
        print("Step 2: compiling")
        print("=" * 55)
        cmd = (f"nvcc -O2 -std=c++17 -arch=sm_86 "
               f"{moe_cu} -o {moe_bin} -lnccl -lpthread")
        print(f"  {cmd}")
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if r.returncode != 0:
            print("compile error:\n", r.stderr); return
        print("  ok\n")

        # step 3: run correctness tests
        print("=" * 55)
        print("Step 3: correctness tests (multi-gpu)")
        print("=" * 55)
        for name in ["small", "large"]:
            json_f = os.path.join(td, f"test_{name}.json")
            if not os.path.exists(json_f):
                print(f"  {name}: skipped"); continue
            print(f"\n--- {name} ---")
            r = subprocess.run([moe_bin, json_f],
                               capture_output=True, text=True, timeout=120)
            print(r.stdout)
            if r.stderr: print("stderr:", r.stderr)

        # step 4: pytorch perf comparison
        print("\n" + "=" * 55)
        print("Step 4: pytorch perf comparison")
        print("=" * 55)
        r = subprocess.run([sys.executable, bench],
                           capture_output=True, text=True, timeout=120)
        print(r.stdout)
        if r.stderr: print("stderr:", r.stderr)


@app.local_entrypoint()
def main():
    print("submitting multi-gpu MoE job to Modal...")
    run_moe_nccl.remote()
