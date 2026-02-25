/*
 * MIT License
 *
 * Copyright (c) 2024 [Takuya Inoue]
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("O3", on)
//#pragma GCC optimize("", on)
#include <vector>
#include <time.h>
#include <immintrin.h>
#include <string.h>
#include <unistd.h>

//#define LOCAL

#ifndef LOCAL
#include "io_struct.hpp"
#endif

//#pragma optimize("O3", on)

using namespace std;

typedef unsigned long long ull;
typedef unsigned int uint;

inline int BTR_16(int x){
    return ((x&8)>>3)|((x&4)>>1)|((x&2)<<1)|((x&1)<<3);
}

inline __m512i set32(uint* g){
    return _mm512_setr_epi32(g[0],g[1],g[2],g[3],g[4],g[5],g[6],g[7],
                            g[8],g[9],g[10],g[11],g[12],g[13],g[14],g[15]);
}

inline __m512i set64(uint* g){
    return _mm512_setr_epi64(g[0],g[1],g[2],g[3],g[4],g[5],g[6],g[7]);
}

/*ull modpow(ull x,ull k,ull mod){
    if(k==0)return 1;
    ull ret=modpow(x,k/2,mod);
    ret*=ret; ret%=mod;
    if(k%2==1){
        ret*=x; ret%=mod;
    }
    return ret;
}*/

ull modpow(ull x, ull k, ull mod){
    ull ret=1, temp=x;
    while(k>0){
        if(k&1ull){ ret *= temp; ret %= mod; }
        temp *= temp; temp %= mod;
        k >>= 1;
    }
    return ret;
}

template <int p=7340033,int t=20>
struct CONV{
    alignas(512) uint a[(1<<t)+2];
    alignas(512) uint b[(1<<t)+2];
    ull root;
    uint rh[t],rih[t],rhdeep[t],rihdeep[t];
    //ull im;
    ull p_;
    unsigned int p_ui;
    ull pull;
    ull r2;
    ull ROOT = 1;
    __m512i v_deep[16];
    __m512i rhdeep512[t];
    __m512i rihdeep512[t];
    uint rr_a_precalc[8][t];
    uint rr_c_precalc[8][t];
    uint rr_c_precalc1, rr_c_precalc2, rr_c_precalc3;
    uint inv;
    uint r2inv;
    CONV(){
        int tt=0;
        int x=p-1;
        while(x%2==0){
            tt++;
            x/=2;
        }
        root=5;
        /*while(1){
            root++;
            if(modpow(root,1<<(tt-1),p)==p-1)break;
        }*/
        uint h[t],ih[t];
        for(int i=0;i<t;i++){
            h[i] = modpow(root,((1<<tt)>>(i+2))*3,p);
            h[i]=p-h[i];
            ih[i] = modpow(h[i],p-2,p);
            
            rh[i] = (ull(h[i])<<32)%p;
            rih[i] = (ull(ih[i])<<32)%p;
        }
        p_ = (ull(1)<<32) - modpow(p,(1<<30)-1,ull(1)<<32);
        p_ui = p_;
        pull = ull(p);
        r2 = (ull(1)<<32)%p; r2 *= r2; r2 %= p;
        ROOT = modpow(root,1<<(tt-t),p);
        
        for(int l=0;l<16;l++){
            uint temp[16] = {};
            for(int j=0;j<16;j++){
                temp[j] = modpow(ROOT, ((j*l)&15)<<(t-4), p);
            }
            v_deep[l] = set32(temp);
        }
        uint hdeep[t],ihdeep[t];
        for(int i=0;i<t;i++){
            rhdeep[i] = (ull(modpow(root,(1<<tt)-(1<<(tt-4))+((3<<tt)>>(i+5)),p))<<32)%p;
            rihdeep[i] = (ull(modpow(root,(1<<tt)+(1<<(tt-4))-((3<<tt)>>(i+5)),p))<<32)%p;
        }
        for(int i=0;i<t;i++){
            uint temp[16]={};
            temp[0]=(ull(1)<<32)%p;
            for(int j=0;j<15;j++){
                temp[j+1] = mont(temp[j],rhdeep[i]);
            }
            rhdeep512[i] = set32(temp);
        }
        for(int i=0;i<t;i++){
            uint temp[16]={};
            temp[0]=(ull(1)<<32)%p;
            for(int j=0;j<15;j++){
                temp[j+1] = mont(temp[j],rihdeep[i]);
            }
            rihdeep512[i] = set32(temp);
        }
        
        for(int th=0;th<8;th++)
        {
            for(int i=3;i<t-1;i++){
                rr_a_precalc[th][i] = modpow(ROOT,BTR_16(th)<<(t-2-i),p)*(ull(1)<<32)%p;
                rr_c_precalc[th][i] = modpow(ROOT,(1<<t)-(BTR_16(th)<<(t-2-i)),p)*(ull(1)<<32)%p;
            }
            rr_a_precalc[th][t-1] = modpow(ROOT,BTR_16(th)>>1,p)*(ull(1)<<32)%p;
            rr_c_precalc[th][t-1] = modpow(ROOT,(1<<t)-(BTR_16(th)>>1),p)*(ull(1)<<32)%p;
        }
        rr_c_precalc1 = modpow(ROOT,(1<<t)-(1<<(t-2)),p)*(ull(1)<<32)%p;;
        rr_c_precalc2 = modpow(ROOT,(1<<t)-(1<<(t-3)),p)*(ull(1)<<32)%p;;
        rr_c_precalc3 = modpow(ROOT,(1<<t)-(3<<(t-3)),p)*(ull(1)<<32)%p;;
        inv=modpow(1<<t,p-2,p);
        r2inv=ull(r2)*inv%p;
    }
    inline uint mont(ull x){
        return (x+(unsigned int)(x)*p_ui*pull)>>32;
    }
    inline uint mont(uint y, uint z){
        ull x = ull(y)*z;
        return (x+(unsigned int)(x)*p_ui*pull)>>32;
    }
    __m512i mont(uint* gU, uint* hU){
        __m512i g = set32(gU);
        __m512i h = set32(hU);
        
        __m512i w0 = _mm512_mul_epu32(g,h);
        __m512i ret0 = _mm512_srli_epi64(_mm512_add_epi64(w0,_mm512_mul_epu32(_mm512_mul_epu32(w0,_mm512_set1_epi64(p_ui)),_mm512_set1_epi64(p))),32);
        
        __m512i w1 = _mm512_mul_epu32(_mm512_srli_epi64(g,32),_mm512_srli_epi64(h,32));
        __m512i ret1 = _mm512_add_epi64(w1,_mm512_mul_epu32(_mm512_mul_epu32(w1,_mm512_set1_epi64(p_ui)),_mm512_set1_epi64(p)));
        
        return _mm512_mask_blend_epi32(0b1010101010101010,ret0,ret1);
    }
    __m512i mont(uint* gU, uint hu){
        __m512i g = set32(gU);
        __m512i h = _mm512_set1_epi64(hu);
        
        __m512i w0 = _mm512_mul_epu32(g,h);
        __m512i ret0 = _mm512_srli_epi64(_mm512_add_epi64(w0,_mm512_mul_epu32(_mm512_mul_epu32(w0,_mm512_set1_epi64(p_ui)),_mm512_set1_epi64(p))),32);
        
        __m512i w1 = _mm512_mul_epu32(_mm512_srli_epi64(g,32),h);
        __m512i ret1 = _mm512_add_epi64(w1,_mm512_mul_epu32(_mm512_mul_epu32(w1,_mm512_set1_epi64(p_ui)),_mm512_set1_epi64(p)));
        
        return _mm512_mask_blend_epi32(0b1010101010101010,ret0,ret1);
    }
    __m512i mont(__m512i g, uint hu){
        __m512i h = _mm512_set1_epi64(hu);
        
        __m512i w0 = _mm512_mul_epu32(g,h);
        __m512i ret0 = _mm512_srli_epi64(_mm512_add_epi64(w0,_mm512_mul_epu32(_mm512_mul_epu32(w0,_mm512_set1_epi64(p_ui)),_mm512_set1_epi64(p))),32);
        
        __m512i w1 = _mm512_mul_epu32(_mm512_srli_epi64(g,32),h);
        __m512i ret1 = _mm512_add_epi64(w1,_mm512_mul_epu32(_mm512_mul_epu32(w1,_mm512_set1_epi64(p_ui)),_mm512_set1_epi64(p)));
        
        return _mm512_mask_blend_epi32(0b1010101010101010,ret0,ret1);
    }
    __m512i mont(uint* gU, __m512i h){
        __m512i g = set32(gU);
        
        __m512i w0 = _mm512_mul_epu32(g,h);
        __m512i ret0 = _mm512_srli_epi64(_mm512_add_epi64(w0,_mm512_mul_epu32(_mm512_mul_epu32(w0,_mm512_set1_epi64(p_ui)),_mm512_set1_epi64(p))),32);
        
        __m512i w1 = _mm512_mul_epu32(_mm512_srli_epi64(g,32),_mm512_srli_epi64(h,32));
        __m512i ret1 = _mm512_add_epi64(w1,_mm512_mul_epu32(_mm512_mul_epu32(w1,_mm512_set1_epi64(p_ui)),_mm512_set1_epi64(p)));
        
        return _mm512_mask_blend_epi32(0b1010101010101010,ret0,ret1);
    }
    __m512i mont(__m512i g, __m512i h){
        __m512i w0 = _mm512_mul_epu32(g,h);
        __m512i ret0 = _mm512_srli_epi64(_mm512_add_epi64(w0,_mm512_mul_epu32(_mm512_mul_epu32(w0,_mm512_set1_epi64(p_ui)),_mm512_set1_epi64(p))),32);
        
        __m512i w1 = _mm512_mul_epu32(_mm512_srli_epi64(g,32),_mm512_srli_epi64(h,32));
        __m512i ret1 = _mm512_add_epi64(w1,_mm512_mul_epu32(_mm512_mul_epu32(w1,_mm512_set1_epi64(p_ui)),_mm512_set1_epi64(p)));
        
        return _mm512_mask_blend_epi32(0b1010101010101010,ret0,ret1);
    }
    void DFT_old_partial(int min_i, int max_i, int min_j, int max_j, uint* g, uint* rr){
        for(int i=min_i;i<max_i;i++){
            const int x=1<<(t-1-i);
            for(int j0=min_j,k=min_j>>(t-i);j0<max_j;j0+=(x<<1)){
                for(int j=j0;j<j0+x;j+=16){
                    __m512i u = set32(g+j);
                    __m512i v = mont(g+j+x,rr[i]);
                    __m512i w = _mm512_set1_epi32(p<<1);
                    _mm512_storeu_epi32(g+j+x,_mm512_add_epi32(u,_mm512_sub_epi32(w,v)));
                    _mm512_storeu_epi32(g+j,_mm512_add_epi32(u,v));
                }
                rr[i] = mont(rr[i],rh[__builtin_ctz(++k)]);
            }
        }
    }
    void deepest_all_partial(int min_j, int max_j, __m512i& rr_a512, __m512i& rr_c512){
        int i=min_j;
        //for(int i=min_j;i<max_j;i+=16){
            uint coef0[16];
            _mm512_storeu_epi32(coef0,mont(a+i,rr_a512));
            __m512i sum0 = _mm512_setzero_epi32();
            for(int l=0;l<16;l++){
                sum0 = _mm512_add_epi32( sum0, mont( v_deep[l], coef0[l] ) );
            }
            uint coef1[16];
            _mm512_storeu_epi32(coef1,mont(b+i,rr_a512));
            __m512i sum1 = _mm512_setzero_epi32();
            for(int l=0;l<16;l++){
                sum1 = _mm512_add_epi32( sum1, mont( v_deep[l], coef1[l] ) );
            }
            rr_a512 = mont( rr_a512, rhdeep512[__builtin_ctz((i>>4)+1)] );
            _mm512_storeu_epi32(a+i,mont(sum0,sum1));
            
            __m512i sum = mont( v_deep[0], a[i] );
            for(int l=1;l<16;l++){
                sum = _mm512_add_epi32( sum, mont( v_deep[16-l], a[i+l] ) );
            }
            sum = mont( sum , rr_c512 );
            _mm512_storeu_epi32(a+i,sum);
            rr_c512 = mont( rr_c512, rihdeep512[__builtin_ctz((i>>4)+1)] );
        //}
    }
    void inv_DFT_old_partial(int min_i, int max_i, int min_j, int max_j, uint* g, uint* rr){
        for(int i=max_i-1;i>=min_i;i--){
            const int x=1<<(t-1-i);
            if(i<t-4&&i>=t-8){
                for(int j0=min_j,k=min_j>>(t-i);j0<max_j;j0+=(x<<1)){
                    for(int j=j0;j<j0+x;j+=16){
                        __m512i u = _mm512_add_epi32(set32(g+j),_mm512_sub_epi32(_mm512_set1_epi32(p<<8),set32(g+j+x)));
                        _mm512_storeu_epi32(g+j,_mm512_add_epi32(set32(g+j),set32(g+j+x)));
                        _mm512_storeu_epi32(g+j+x,mont(u,rr[i]));
                    }
                    rr[i]=mont(rr[i],rih[__builtin_ctz(++k)]);
                }
            }
            else {
                for(int j0=min_j,k=min_j>>(t-i);j0<max_j;j0+=(x<<1)){
                    int x_ = max( x>>8 , 16 );
                    for(int j=j0;j<j0+x_;j+=16){
                        __m512i u = _mm512_add_epi32(set32(g+j),_mm512_sub_epi32(_mm512_set1_epi32(p<<8),set32(g+j+x)));
                        _mm512_storeu_epi32(g+j,mont(_mm512_add_epi32(set32(g+j),set32(g+j+x)),(ull(1)<<32)%p));
                        _mm512_storeu_epi32(g+j+x,mont(u,rr[i]));
                    }
                    for(int j=j0+x_;j<j0+x;j+=16){
                        __m512i u = _mm512_add_epi32(set32(g+j),_mm512_sub_epi32(_mm512_set1_epi32(p<<8),set32(g+j+x)));
                        _mm512_storeu_epi32(g+j,_mm512_add_epi32(set32(g+j),set32(g+j+x)));
                        _mm512_storeu_epi32(g+j+x,mont(u,rr[i]));
                    }
                    rr[i]=mont(rr[i],rih[__builtin_ctz(++k)]);
                }
            }
        }
    }
    void inv_DFT_old_partial_refined(int i, int min_j, int max_j, uint* g, uint rr){
        const int x=1<<(t-1-i);
        int j0 = min_j;
        int x_ = max( x>>8 , 16 ); if(i==0)x_ = max_j-min_j;
        for(int j=j0;j<j0+x_;j+=16){
            __m512i u = _mm512_add_epi32(set32(g+j),_mm512_sub_epi32(_mm512_set1_epi32(p<<8),set32(g+j+x)));
            _mm512_storeu_epi32(g+j,mont(_mm512_add_epi32(set32(g+j),set32(g+j+x)),(ull(1)<<32)%p));
            _mm512_storeu_epi32(g+j+x,mont(u,rr));
        }
        for(int j=j0+x_;j<max_j;j+=16){
            __m512i u = _mm512_add_epi32(set32(g+j),_mm512_sub_epi32(_mm512_set1_epi32(p<<8),set32(g+j+x)));
            _mm512_storeu_epi32(g+j,_mm512_add_epi32(set32(g+j),set32(g+j+x)));
            _mm512_storeu_epi32(g+j+x,mont(u,rr));
        }
    }
    void run(){
        //uint rr_a[t], rr_b[t]; for(int i=0;i<t;i++)rr_b[i]=rr_a[i]=(ull(1)<<32)%p;
        //uint rr_c[t]; for(int i=0;i<t;i++)rr_c[i]=(ull(1)<<32)%p;
        //__m512i rr_a512 = _mm512_set1_epi32(uint((ull(1)<<32)%p*(ull(1)<<32)%p));
        //__m512i rr_c512 = _mm512_set1_epi32(uint((ull(1)<<32)%p*(ull(1)<<32)%p));
        constexpr int s_L3=4;
        constexpr int s_L3_old=t-18;
        //constexpr int s_L2=max(t-15,4);
        constexpr int s_L2=max(t-16,4);
        //constexpr int s_L1=t-11;
        constexpr int s_L1=t-12;
        //DFT_old_partial(2,s_L3,0,1<<t,a,rr_a);
        //DFT_old_partial(2,s_L3,0,1<<t,b,rr_b);
#ifndef LOCAL
#pragma omp parallel num_threads(8) proc_bind(close)
#endif
        {
#ifndef LOCAL
#pragma omp for
#endif
            for(int th=0;th<8;th++)
            {
                uint rr_a[t], rr_b[t]; //for(int i=0;i<t;i++)rr_b[i]=rr_a[i]=(ull(1)<<32)%p;
                uint rr_c[t]; //for(int i=0;i<t;i++)rr_c[i]=(ull(1)<<32)%p;
                __m512i rr_a512; // = _mm512_set1_epi32(uint((ull(1)<<32)%p*(ull(1)<<32)%p));
                __m512i rr_c512; // = _mm512_set1_epi32(uint((ull(1)<<32)%p*(ull(1)<<32)%p));
                for(int i=3;i<t;i++){
                    rr_a[i] = rr_b[i] = rr_a_precalc[th][i];
                    rr_c[i] = rr_c_precalc[th][i];
                    //rr_a[i] = rr_b[i] = modpow(ROOT,BTR_16(th)<<(t-2-i),p)*(ull(1)<<32)%p;
                    //rr_c[i] = modpow(ROOT,(1<<t)-(BTR_16(th)<<(t-2-i)),p)*(ull(1)<<32)%p;
                }
                {
                    uint temp[16]={};
                    temp[0] = (ull(1)<<32)%p*(ull(1)<<32)%p;
                    for(int j=0;j<15;j++){
                        temp[j+1] = mont(temp[j],rr_a[t-1]);
                    }
                    rr_a512 = set32(temp);
                }
                {
                    uint temp[16]={};
                    temp[0] = (ull(1)<<32)%p*(ull(1)<<32)%p;
                    for(int j=0;j<15;j++){
                        temp[j+1] = mont(temp[j],rr_c[t-1]);
                    }
                    rr_c512 = set32(temp);
                }
                
                for(int i_L3=(th<<(s_L3-3));i_L3<((th+1)<<(s_L3-3));i_L3++){
                    DFT_old_partial(s_L3,s_L2,i_L3<<(t-s_L3),(i_L3+1)<<(t-s_L3),a,rr_a);
                    DFT_old_partial(s_L3,s_L2,i_L3<<(t-s_L3),(i_L3+1)<<(t-s_L3),b,rr_b);
                    for(int i_L2=0;i_L2<(1<<(s_L2-s_L3));i_L2++){
                        DFT_old_partial(s_L2,s_L1,(i_L3<<(t-s_L3))+(i_L2<<(t-s_L2)),(i_L3<<(t-s_L3))+((i_L2+1)<<(t-s_L2)),a,rr_a);
                        DFT_old_partial(s_L2,s_L1,(i_L3<<(t-s_L3))+(i_L2<<(t-s_L2)),(i_L3<<(t-s_L3))+((i_L2+1)<<(t-s_L2)),b,rr_b);
                        for(int i_L1=0;i_L1<(1<<(s_L1-s_L2));i_L1+=2){
                            DFT_old_partial(s_L1,t-4,(i_L3<<(t-s_L3))+(i_L2<<(t-s_L2))+(i_L1<<(t-s_L1)),
                                            (i_L3<<(t-s_L3))+(i_L2<<(t-s_L2))+((i_L1+1)<<(t-s_L1)),a,rr_a);
                            DFT_old_partial(s_L1,t-4,(i_L3<<(t-s_L3))+(i_L2<<(t-s_L2))+(i_L1<<(t-s_L1)),
                                            (i_L3<<(t-s_L3))+(i_L2<<(t-s_L2))+((i_L1+1)<<(t-s_L1)),b,rr_b);
                            for(int i=(i_L3<<(t-s_L3))+(i_L2<<(t-s_L2))+(i_L1<<(t-s_L1));i<(i_L3<<(t-s_L3))+(i_L2<<(t-s_L2))+((i_L1+1)<<(t-s_L1));i+=16){
                                deepest_all_partial(i,i+16,rr_a512,rr_c512);
                            }
                            inv_DFT_old_partial(s_L1,t-4,(i_L3<<(t-s_L3))+(i_L2<<(t-s_L2))+(i_L1<<(t-s_L1)),
                                                (i_L3<<(t-s_L3))+(i_L2<<(t-s_L2))+((i_L1+1)<<(t-s_L1)),a,rr_c);
                            
                            DFT_old_partial(s_L1,t-4,(i_L3<<(t-s_L3))+(i_L2<<(t-s_L2))+((i_L1+1)<<(t-s_L1)),
                                            (i_L3<<(t-s_L3))+(i_L2<<(t-s_L2))+((i_L1+2)<<(t-s_L1)),a,rr_a);
                            DFT_old_partial(s_L1,t-4,(i_L3<<(t-s_L3))+(i_L2<<(t-s_L2))+((i_L1+1)<<(t-s_L1)),
                                            (i_L3<<(t-s_L3))+(i_L2<<(t-s_L2))+((i_L1+2)<<(t-s_L1)),b,rr_b);
                            for(int i=(i_L3<<(t-s_L3))+(i_L2<<(t-s_L2))+((i_L1+1)<<(t-s_L1));i<(i_L3<<(t-s_L3))+(i_L2<<(t-s_L2))+((i_L1+2)<<(t-s_L1));i+=16){
                                deepest_all_partial(i,i+16,rr_a512,rr_c512);
                            }
                            inv_DFT_old_partial(s_L1,t-4,(i_L3<<(t-s_L3))+(i_L2<<(t-s_L2))+((i_L1+1)<<(t-s_L1)),
                                                (i_L3<<(t-s_L3))+(i_L2<<(t-s_L2))+((i_L1+2)<<(t-s_L1)),a,rr_c);
                            inv_DFT_old_partial(s_L1-1,s_L1,(i_L3<<(t-s_L3))+(i_L2<<(t-s_L2))+(i_L1<<(t-s_L1)),
                                                (i_L3<<(t-s_L3))+(i_L2<<(t-s_L2))+((i_L1+2)<<(t-s_L1)),a,rr_c);
                        }
                        inv_DFT_old_partial(s_L2,s_L1-1,(i_L3<<(t-s_L3))+(i_L2<<(t-s_L2)),(i_L3<<(t-s_L3))+((i_L2+1)<<(t-s_L2)),a,rr_c);
                        if(i_L2&1){
                            inv_DFT_old_partial(s_L2-1,s_L2,(i_L3<<(t-s_L3))+((i_L2-1)<<(t-s_L2)),(i_L3<<(t-s_L3))+((i_L2+1)<<(t-s_L2)),a,rr_c);
                        }
                    }
                    inv_DFT_old_partial(s_L3,s_L2-1,i_L3<<(t-s_L3),(i_L3+1)<<(t-s_L3),a,rr_c);
                    if(i_L3&1){
                        inv_DFT_old_partial(s_L3-1,s_L3,(i_L3-1)<<(t-s_L3),(i_L3+1)<<(t-s_L3),a,rr_c);
                    }
                }
            }
        /*}
        //uint rr_c_gen[t]; for(int i=0;i<t;i++)rr_c_gen[i]=(ull(1)<<32)%p;
        //inv_DFT_old_partial(2,3,0,1<<t,a,rr_c_gen);
#ifndef LOCAL
#pragma omp parallel num_threads(8) proc_bind(spread)
#endif
        {*/
#ifndef LOCAL
#pragma omp barrier
#endif
#ifndef LOCAL
#pragma omp for
#endif
            for(int th=0;th<8;th++){
                if(th<2){
                    uint rr_c = (ull(1)<<32)%p;
                    inv_DFT_old_partial_refined(2,th<<(t-4),(th+1)<<(t-4),a,rr_c);
                }
                else if(th<4){
                    //uint rr_c = modpow(ROOT,(1<<t)-(1<<(t-2)),p)*(ull(1)<<32)%p;
                    uint rr_c = rr_c_precalc1;
                    inv_DFT_old_partial_refined(2,(th+2)<<(t-4),(th+3)<<(t-4),a,rr_c);
                }
                else if(th<6){
                    //uint rr_c = modpow(ROOT,(1<<t)-(1<<(t-3)),p)*(ull(1)<<32)%p;
                    uint rr_c = rr_c_precalc2;
                    inv_DFT_old_partial_refined(2,(th+4)<<(t-4),(th+5)<<(t-4),a,rr_c);
                }
                else {
                    //uint rr_c = modpow(ROOT,(1<<t)-(3<<(t-3)),p)*(ull(1)<<32)%p;
                    uint rr_c = rr_c_precalc3;
                    inv_DFT_old_partial_refined(2,(th+6)<<(t-4),(th+7)<<(t-4),a,rr_c);
                }
            }
/*        }
        //inv_DFT_old_partial(1,2,0,1<<t,a,rr_c_gen);
#ifndef LOCAL
#pragma omp parallel num_threads(8) proc_bind(spread)
#endif
        {*/
#ifndef LOCAL
#pragma omp barrier
#endif
#ifndef LOCAL
#pragma omp for
#endif
            for(int th=0;th<8;th++){
                if(th<4){
                    uint rr_c = (ull(1)<<32)%p;
                    inv_DFT_old_partial_refined(1,th<<(t-4),(th+1)<<(t-4),a,rr_c);
                }
                else {
                    //uint rr_c = modpow(ROOT,(1<<t)-(1<<(t-2)),p)*(ull(1)<<32)%p;
                    uint rr_c = rr_c_precalc1;
                    inv_DFT_old_partial_refined(1,(th+4)<<(t-4),(th+5)<<(t-4),a,rr_c);
                }
            }
/*        }
        //inv_DFT_old_partial(0,1,0,1<<t,a,rr_c_gen);
#ifndef LOCAL
#pragma omp parallel num_threads(8) proc_bind(spread)
#endif
        {*/
            
#ifndef LOCAL
#pragma omp barrier
#endif
#ifndef LOCAL
#pragma omp for
#endif
            for(int th=0;th<8;th++){
                uint rr_c = (ull(1)<<32)%p;
                inv_DFT_old_partial_refined(0,th<<(t-4),(th+1)<<(t-4),a,rr_c);
            }
//        }
        
        
/*#ifndef LOCAL
#pragma omp parallel num_threads(8) proc_bind(close)
#endif
        {*/
/*#ifndef LOCAL
#pragma omp barrier
#endif
#ifndef LOCAL
#pragma omp for
#endif
            for(int th=0;th<8;th++){
                for(int i=th<<(t-3);i<(th+1)<<(t-3);i+=16){
                    __m512i u = mont(a+i,r2inv);
                    __mmask16 mask = _mm512_cmp_epi32_mask(u,_mm512_set1_epi32(p),5);
                    _mm512_storeu_epi32(a+i,_mm512_mask_sub_epi32(u,mask,u,_mm512_set1_epi32(p)));
                }
            }*/
        }
    }
    void finalize(uint* g){
#ifndef LOCAL
#pragma omp parallel num_threads(8) proc_bind(close)
#endif
        {
#ifndef LOCAL
#pragma omp for
#endif
            for(int th=0;th<8;th++){
                for(int i=th<<(t-4);i<(th+1)<<(t-4);i+=16){
                    __m512i u = mont(g+i,r2inv);
                    __mmask16 mask = _mm512_cmp_epi32_mask(u,_mm512_set1_epi32(p),5);
                    _mm512_storeu_epi32(g+i,_mm512_mask_sub_epi32(u,mask,u,_mm512_set1_epi32(p)));
                }
            }
        }
    }
};

using io_struct::input_t;
using io_struct::output_t;

constexpr int MAX_N = 524288;
constexpr int num_thread = 8;

/*void solve_thread(input_t &input, output_t &output, int id, int *ret){
    int cnt[256] = {};
    int lef[256] = {};
    int N_=1; while(N_<input.N)N_*=2;
    const int B = 8192;
    int p[B] = {};
    for(int L = id*N_/(num_thread);L<(id+1)*N_/(num_thread);L+=B){
        for(int i=0;i<input.K;i++){
            cnt[i] = 0;
        }
        for(int i=L;i<L+B&&i<input.N;i++){
            cnt[input.b[i]] ++;
        }
        lef[0]=0;
        for(int i=0;i+1<input.K;i++){
            lef[i+1] = lef[i] + cnt[i];
        }
        for(int i=0;i<input.K;i++){
            cnt[i] = 0;
        }
        for(int i=L;i<L+B&&i<input.N;i++){
            int b = input.b[i];
            p[lef[b]+cnt[b]] = i;
            cnt[b] ++;
        }
        
        for(int i=0;i<input.N;i++){
            int a = input.a[i];
            int* j = p+lef[a];
            int* last = p+(lef[a]+cnt[a]);
            while(j!=last){
                ret[input.N+*j-i] ++;
                ++j;
            }
        }
    }
    for(int j=0;j<input.N;j++){
        ret[j] += ret[j+input.N];
    }
}*/

void solve_thread(input_t &input, output_t &output, int id, uint16_t *ret){
    int N_=1; while(N_<input.N)N_*=2;
    alignas(64) uint16_t cnt[256] = {};
    uint16_t lef[256] = {};
    //int N_=1; while(N_<input.N)N_*=2;
    constexpr int B = 8192;
    //constexpr int B = 16384;
    uint16_t p[B];
    //const int B = solve_thread_B;
    //int *cnt = solve_thread_cnt[id];
    //int *lef = solve_thread_lef[id];
    //int *p = solve_thread_p[id];
    /*if(id<num_thread-1){
        for(int L = id*N_/(num_thread);L<(id+1)*N_/(num_thread);L+=B){
            for(int i=0;i<input.K;++i){
                cnt[i] = 0;
            }
            for(int i=L;i<L+B;++i){
                ++cnt[input.b[i]];
            }
            lef[0]=0;
            for(int i=0;i<input.K-1;++i){
                lef[i+1] = lef[i] + cnt[i];
            }
            for(int i=0;i<input.K;++i){
                cnt[i] = 0;
            }
            for(int i=L;i<L+B;++i){
                int b = input.b[i];
                p[lef[b]+cnt[b]++] = i-L;
            }
            
            for(int i=0;i<input.N;++i)
            {
                int a = input.a[i];
                for(int j=lef[a];j<lef[a]+cnt[a];++j){
                    ++ret[input.N-i+p[j]+L];
                }
            }
        }
    }
    else {*/
        for(int L = id*N_/(num_thread);L<(id+1)*N_/(num_thread);L+=B){
            for(int i=0;i<input.K;++i){
                cnt[i] = 0;
            }
            for(int i=L;i<L+B&&i<input.N;++i){
                ++cnt[input.b[i]];
            }
            lef[0]=0;
            for(int i=0;i<input.K-1;++i){
                lef[i+1] = lef[i] + cnt[i];
            }
            for(int i=0;i<input.K;++i){
                cnt[i] = 0;
            }
            for(int i=L;i<L+B&&i<input.N;++i){
                int b = input.b[i];
                p[lef[b]+cnt[b]++] = i-L;
            }
            
            for(int i=0;i<input.N;++i)
            {
                int a = input.a[i];
                for(int j=lef[a];j<lef[a]+cnt[a];++j){
                    ++ret[input.N-i+p[j]+L];
                }
            }
        }
    //}
    for(int j=0;j<input.N;j++){
        ret[j] += ret[j+input.N];
    }
}

void solve_thread_16384(input_t &input, output_t &output, int id, uint16_t *ret){
    int N_=1; while(N_<input.N)N_*=2;
    alignas(64) uint16_t cnt[256] = {};
    uint16_t lef[256] = {};
    //int N_=1; while(N_<input.N)N_*=2;
    //constexpr int B = 8192;
    constexpr int B = 16384;
    uint16_t p[B];
    //const int B = solve_thread_B;
    //int *cnt = solve_thread_cnt[id];
    //int *lef = solve_thread_lef[id];
    //int *p = solve_thread_p[id];
    /*if(id<num_thread-1){
        for(int L = id*N_/(num_thread);L<(id+1)*N_/(num_thread);L+=B){
            for(int i=0;i<input.K;++i){
                cnt[i] = 0;
            }
            for(int i=L;i<L+B;++i){
                ++cnt[input.b[i]];
            }
            lef[0]=0;
            for(int i=0;i<input.K-1;++i){
                lef[i+1] = lef[i] + cnt[i];
            }
            for(int i=0;i<input.K;++i){
                cnt[i] = 0;
            }
            for(int i=L;i<L+B;++i){
                int b = input.b[i];
                p[lef[b]+cnt[b]++] = i-L;
            }
            
            for(int i=0;i<input.N;++i)
            {
                int a = input.a[i];
                for(int j=lef[a];j<lef[a]+cnt[a];++j){
                    ++ret[input.N-i+p[j]+L];
                }
            }
        }
    }
    else {*/
        for(int L = id*N_/(num_thread);L<(id+1)*N_/(num_thread);L+=B){
            for(int i=0;i<input.K;++i){
                cnt[i] = 0;
            }
            for(int i=L;i<L+B&&i<input.N;++i){
                ++cnt[input.b[i]];
            }
            lef[0]=0;
            for(int i=0;i<input.K-1;++i){
                lef[i+1] = lef[i] + cnt[i];
            }
            for(int i=0;i<input.K;++i){
                cnt[i] = 0;
            }
            for(int i=L;i<L+B&&i<input.N;++i){
                int b = input.b[i];
                p[lef[b]+cnt[b]++] = i-L;
            }
            
            for(int i=0;i<input.N;++i)
            {
                int a = input.a[i];
                for(int j=lef[a];j<lef[a]+cnt[a];++j){
                    ++ret[input.N-i+p[j]+L];
                }
            }
        }
    //}
    for(int j=0;j<input.N;j++){
        ret[j] += ret[j+input.N];
    }
}

alignas(64) uint16_t ret_thread_16[num_thread][MAX_N+16];
alignas(64) uint ret[MAX_N+16];
alignas(64) uint ret_thread[num_thread][MAX_N+16];

const int FFT_P = 7340033;
CONV<FFT_P,18> conv_18;
CONV<FFT_P,19> conv_19;
CONV<FFT_P,20> conv_20;

void solve(input_t &input, output_t &output)
{
    int t=1; while((1<<t)<input.N)t++;
    
    if(input.K==256){
        #ifndef LOCAL
                #pragma omp parallel num_threads(8) proc_bind(spread)
        #endif
                    {
        #ifndef LOCAL
                #pragma omp for
        #endif
                        for(int i=0;i<num_thread;i++){
                            //solve_thread(input,output,i,ret_thread[i]);
                            if(t==16)solve_thread(input,output,i,ret_thread_16[i]);
                            else solve_thread_16384(input,output,i,ret_thread_16[i]);
                        }
                    }
                    
        /*#ifndef LOCAL
                    #pragma omp parallel num_threads(8) proc_bind(spread)
        #endif
                    {
        #ifndef LOCAL
                        #pragma omp for
        #endif
                        for(int k=0;k<8;k++){
                            for(int i=0;i<num_thread;i++){
                                for(int j=(k<<(t-3));j<((k+1)<<(t-3));j+=16){
                                    _mm512_storeu_epi32(ret+j,_mm512_add_epi32(set32(ret+j),set32(ret_thread[i]+j)));
                                }
                            }
                        }
                    }*/
        
#ifndef LOCAL
            #pragma omp parallel num_threads(8) proc_bind(spread)
#endif
        {
#ifndef LOCAL
#pragma omp for
#endif
            for(int k=0;k<8;k++){
                for(int i=0;i<num_thread;i++){
                    for(int j=(k<<(t-3));j<((k+1)<<(t-3));j++){
                        //ret[j] += ret_thread[i][j];
                        ret[j] += ret_thread_16[i][j];
                    }
                }
            }
        }
        
        std::pair<int,int> best[8];
#ifndef LOCAL
            #pragma omp parallel num_threads(8) proc_bind(spread)
#endif
        {
#ifndef LOCAL
#pragma omp for
#endif
            for(int k=0;k<8;k++){
                best[k] = make_pair(ret[0],0);
                //for(int j=(k<<(t-3));j<((k+1)<<(t-3));j++){
                //    if(best[k].first < ret[j]){
                //        best[k].first = ret[j];
                //        best[k].second = j;
                //    }
                //}
                __m512i v = _mm512_setzero_epi32();
                for(int j=(k<<(t-3));j<((k+1)<<(t-3));j+=8){
                    //auto w = _mm512_setr_epi32(ret[j],j,ret[j+1],j+1,ret[j+2],j+2,ret[j+3],j+3,ret[j+4],j+4,ret[j+5],j+5,ret[j+6],j+6,ret[j+7],j+7);
                    auto w = _mm512_setr_epi32(j,ret[j],j+1,ret[j+1],j+2,ret[j+2],j+3,ret[j+3],j+4,ret[j+4],j+5,ret[j+5],j+6,ret[j+6],j+7,ret[j+7]);
                    v = _mm512_max_epi64(v,w);
                }
                uint temp[16];
                _mm512_storeu_epi32(temp,v);
                for(int j=0;j<8;j++){
                    if(best[k].first < temp[2*j+1]){
                        best[k].first = temp[2*j+1];
                        best[k].second = temp[2*j];
                    }
                }
            }
        }
        output.n_spells = input.N;
        output.n_shifts = 0;
        for(int k=0;k<8;k++){
            if(output.n_spells > input.N - best[k].first){
                output.n_spells = input.N - best[k].first;
                output.n_shifts = best[k].second;
            }
        }
        
        /*std::pair<int,int> best(input.N-ret[0],0);
        for(int i=0;i<input.N;i++){
            if(best.first > input.N-ret[i]){
                best.first = input.N-ret[i];
                best.second = i;
            }
        }
        output.n_spells = best.first;
        output.n_shifts = best.second;*/
        
        return;
    }
    
    input.b.resize(1<<t,input.K);
    
    if(t==19){
        for(int i_th=0;i_th<1;i_th++){
            for(int i=i_th;i<input.K;i+=1){
                const int t_ = 20;
                auto &conv = conv_20;
                {
                    
                    if(1){
    #ifndef LOCAL
    #pragma omp parallel num_threads(8) proc_bind(spread)
    #endif
                        {
    #ifndef LOCAL
    #pragma omp sections
    #endif
                            {
    #ifndef LOCAL
    #pragma omp section
    #endif
                                {
                                    for(int k=0;k<(1<<(t_-2));k++){
                                        conv.a[k]=0;
                                    }
                                }
    #ifndef LOCAL
    #pragma omp section
    #endif
                                {
                                    for(int k=(1<<(t_-2));k<(2<<(t_-2));k++){
                                        conv.a[k]=0;
                                    }
                                }
    #ifndef LOCAL
    #pragma omp section
    #endif
                                {
                                    for(int k=(2<<(t_-2));k<(3<<(t_-2));k++){
                                        conv.a[k]=0;
                                    }
                                }
    #ifndef LOCAL
    #pragma omp section
    #endif
                                {
                                    for(int k=(3<<(t_-2));k<(4<<(t_-2));k++){
                                        conv.a[k]=0;
                                    }
                                }
    #ifndef LOCAL
    #pragma omp section
    #endif
                                {
                                    for(int k=0;k<(1<<(t_-2));k++){
                                        conv.b[k]=0;
                                    }
                                }
    #ifndef LOCAL
    #pragma omp section
    #endif
                                {
                                    for(int k=(1<<(t_-2));k<(2<<(t_-2));k++){
                                        conv.b[k]=0;
                                    }
                                }
    #ifndef LOCAL
    #pragma omp section
    #endif
                                {
                                    for(int k=(2<<(t_-2));k<(3<<(t_-2));k++){
                                        conv.b[k]=0;
                                    }
                                }
    #ifndef LOCAL
    #pragma omp section
    #endif
                                {
                                    for(int k=(3<<(t_-2));k<(4<<(t_-2));k++){
                                        conv.b[k]=0;
                                    }
                                }
                            }
                        }
                    }
                    __m512i rr512[8];
                    for(int k=0;k<8;k++){
                        rr512[k] = conv.v_deep[k];
                    }
#ifndef LOCAL
#pragma omp parallel num_threads(8) proc_bind(spread)
#endif
                    {
#ifndef LOCAL
#pragma omp for
#endif
                        for(int th_=0;th_<8;th_++){
                            uint temp[16];
                            if(th_<4){
                                int th=th_;
                                for(int l=th<<(t_-6);l<(th+1)<<(t_-6);l++){
                                    __m512i sum = _mm512_setzero_epi32();
                                    bool f=false;
                                    for(int k=0;k<8;k++){
                                        int j=(k<<(t_-4))+l;
                                        if(j<input.N&&input.a[input.N-1-j]==i){
                                            sum = _mm512_add_epi32(sum,rr512[k]);
                                            f = true;
                                        }
                                    }
                                    if(f){
                                        _mm512_storeu_epi32(temp,sum);
                                        conv.a[l] = temp[0];
                                        conv.a[(1<<(t_-4))+l] = temp[8];
                                        conv.a[(2<<(t_-4))+l] = temp[4];
                                        conv.a[(3<<(t_-4))+l] = temp[12];
                                        conv.a[(4<<(t_-4))+l] = temp[2];
                                        conv.a[(5<<(t_-4))+l] = temp[10];
                                        conv.a[(6<<(t_-4))+l] = temp[6];
                                        conv.a[(7<<(t_-4))+l] = temp[14];
                                        conv.a[(8<<(t_-4))+l] = temp[1];
                                        conv.a[(9<<(t_-4))+l] = temp[9];
                                        conv.a[(10<<(t_-4))+l] = temp[5];
                                        conv.a[(11<<(t_-4))+l] = temp[13];
                                        conv.a[(12<<(t_-4))+l] = temp[3];
                                        conv.a[(13<<(t_-4))+l] = temp[11];
                                        conv.a[(14<<(t_-4))+l] = temp[7];
                                        conv.a[(15<<(t_-4))+l] = temp[15];
                                    }
                                }
                            }
                            else {
                                int th=th_-4;
                                for(int l=th<<(t_-6);l<(th+1)<<(t_-6);l++){
                                    __m512i sum = _mm512_setzero_epi32();
                                    bool f=false;
                                    for(int k=0;k<8;k++){
                                        int j=(k<<(t_-4))+l;
                                        if(input.b[j]==i){
                                            sum = _mm512_add_epi32(sum,rr512[k]);
                                            f = true;
                                        }
                                    }
                                    if(f){
                                        _mm512_storeu_epi32(temp,sum);
                                        conv.b[l] = temp[0];
                                        conv.b[(1<<(t_-4))+l] = temp[8];
                                        conv.b[(2<<(t_-4))+l] = temp[4];
                                        conv.b[(3<<(t_-4))+l] = temp[12];
                                        conv.b[(4<<(t_-4))+l] = temp[2];
                                        conv.b[(5<<(t_-4))+l] = temp[10];
                                        conv.b[(6<<(t_-4))+l] = temp[6];
                                        conv.b[(7<<(t_-4))+l] = temp[14];
                                        conv.b[(8<<(t_-4))+l] = temp[1];
                                        conv.b[(9<<(t_-4))+l] = temp[9];
                                        conv.b[(10<<(t_-4))+l] = temp[5];
                                        conv.b[(11<<(t_-4))+l] = temp[13];
                                        conv.b[(12<<(t_-4))+l] = temp[3];
                                        conv.b[(13<<(t_-4))+l] = temp[11];
                                        conv.b[(14<<(t_-4))+l] = temp[7];
                                        conv.b[(15<<(t_-4))+l] = temp[15];
                                    }
                                }
                            }
                        }
                    }
/*#ifndef LOCAL
#pragma omp parallel num_threads(8) proc_bind(spread)
#endif
                    {
#ifndef LOCAL
#pragma omp for
#endif
                        for(int th=0;th<8;th++){
                            for(int l=th<<(t_-7);l<(th+1)<<(t_-7);l++){
                                __m512i sum = _mm512_setzero_epi32();
                                bool f=false;
                                for(int k=0;k<8;k++){
                                    int j=(k<<(t_-4))+l;
                                    if(j<input.N&&input.a[input.N-1-j]==i){
                                        sum = _mm512_add_epi32(sum,rr512[k]);
                                        f=true;
                                    }
                                }
                                if(f){
                                    uint temp[16];
                                    _mm512_storeu_epi32(temp,sum);
                                    for(int k=0;k<16;k++){
                                        int j=(k<<(t_-4))+l;
                                        conv.a[j] = temp[BTR_16(k)];
                                    }
                                }
                                else {
                                    for(int k=0;k<16;k++){
                                        conv.a[(k<<(t_-4))+l] = 0;
                                    }
                                }
                            }
                        }
                    }
#ifndef LOCAL
#pragma omp parallel num_threads(8) proc_bind(spread)
#endif
                    {
#ifndef LOCAL
#pragma omp for
#endif
                        for(int th=0;th<8;th++){
                            for(int l=th<<(t_-7);l<(th+1)<<(t_-7);l++){
                                __m512i sum = _mm512_setzero_epi32();
                                bool f=false;
                                for(int k=0;k<8;k++){
                                    int j=(k<<(t_-4))+l;
                                    if(j<input.N&&input.b[j]==i){
                                        sum = _mm512_add_epi32(sum,rr512[k]);
                                        f=true;
                                    }
                                }
                                if(f){
                                    uint temp[16];
                                    _mm512_storeu_epi32(temp,sum);
                                    for(int k=0;k<16;k++){
                                        int j=(k<<(t_-4))+l;
                                        conv.b[j] = temp[BTR_16(k)];
                                    }
                                }
                                else {
                                    for(int k=0;k<16;k++){
                                        conv.b[(k<<(t_-4))+l] = 0;
                                    }
                                }
                            }
                        }
                    }*/
                }
                conv.run();
#ifndef LOCAL
#pragma omp parallel num_threads(8) proc_bind(spread)
#endif
                {
#ifndef LOCAL
#pragma omp for
#endif
                    for(int th=0;th<8;th++){
                        for(int j=th<<(t-3);j<min((th+1)<<(t-3),(int)input.N);j++){
                            ret[j] += conv.a[j] + conv.a[j+input.N];
                        }
                    }
                }
                
                /*if(i+1<input.K){
#ifndef LOCAL
#pragma omp parallel num_threads(8) proc_bind(spread)
#endif
                    {
#ifndef LOCAL
#pragma omp for
#endif
                        for(int th=0;th<4;th++){
                            if(th==0){
                                for(int k=0;k<(1<<(t_-1));k++){
                                    conv.a[k]=0;
                                }
                            }
                            else if(th==1){
                                for(int k=1<<(t_-1);k<(1<<t_);k++){
                                    conv.a[k]=0;
                                }
                            }
                            else if(th==2){
                                for(int k=0;k<(1<<t_);k++){
                                    conv.b[k]=0;
                                }
                            }
                            else {
                                for(int k=1<<(t_-1);k<(1<<t_);k++){
                                    conv.b[k]=0;
                                }
                            }
                        }
                    }
                }*/
            }
        }
    }
    else if(t==18){
        for(int i_th=0;i_th<1;i_th++){
            for(int i=i_th;i<input.K;i+=1){
                const int t_ = 19;
                auto &conv = conv_19;
                {
                    if(1){
    #ifndef LOCAL
    #pragma omp parallel num_threads(8) proc_bind(spread)
    #endif
                        {
    #ifndef LOCAL
    #pragma omp sections
    #endif
                            {
    #ifndef LOCAL
    #pragma omp section
    #endif
                                {
                                    for(int k=0;k<(1<<(t_-2));k++){
                                        conv.a[k]=0;
                                    }
                                }
    #ifndef LOCAL
    #pragma omp section
    #endif
                                {
                                    for(int k=(1<<(t_-2));k<(2<<(t_-2));k++){
                                        conv.a[k]=0;
                                    }
                                }
    #ifndef LOCAL
    #pragma omp section
    #endif
                                {
                                    for(int k=(2<<(t_-2));k<(3<<(t_-2));k++){
                                        conv.a[k]=0;
                                    }
                                }
    #ifndef LOCAL
    #pragma omp section
    #endif
                                {
                                    for(int k=(3<<(t_-2));k<(4<<(t_-2));k++){
                                        conv.a[k]=0;
                                    }
                                }
    #ifndef LOCAL
    #pragma omp section
    #endif
                                {
                                    for(int k=0;k<(1<<(t_-2));k++){
                                        conv.b[k]=0;
                                    }
                                }
    #ifndef LOCAL
    #pragma omp section
    #endif
                                {
                                    for(int k=(1<<(t_-2));k<(2<<(t_-2));k++){
                                        conv.b[k]=0;
                                    }
                                }
    #ifndef LOCAL
    #pragma omp section
    #endif
                                {
                                    for(int k=(2<<(t_-2));k<(3<<(t_-2));k++){
                                        conv.b[k]=0;
                                    }
                                }
    #ifndef LOCAL
    #pragma omp section
    #endif
                                {
                                    for(int k=(3<<(t_-2));k<(4<<(t_-2));k++){
                                        conv.b[k]=0;
                                    }
                                }
                            }
                        }
                    }
                    __m512i rr512[8];
                    for(int k=0;k<8;k++){
                        rr512[k] = conv.v_deep[k];
                    }
#ifndef LOCAL
#pragma omp parallel num_threads(8) proc_bind(spread)
#endif
                    {
#ifndef LOCAL
#pragma omp for
#endif
                        for(int th_=0;th_<8;th_++){
                            uint temp[16];
                            if(th_<4){
                                int th=th_;
                                for(int l=th<<(t_-6);l<(th+1)<<(t_-6);l++){
                                    __m512i sum = _mm512_setzero_epi32();
                                    bool f=false;
                                    for(int k=0;k<8;k++){
                                        int j=(k<<(t_-4))+l;
                                        if(j<input.N&&input.a[input.N-1-j]==i){
                                            sum = _mm512_add_epi32(sum,rr512[k]);
                                            f = true;
                                        }
                                    }
                                    if(f){
                                        _mm512_storeu_epi32(temp,sum);
                                        conv.a[l] = temp[0];
                                        conv.a[(1<<(t_-4))+l] = temp[8];
                                        conv.a[(2<<(t_-4))+l] = temp[4];
                                        conv.a[(3<<(t_-4))+l] = temp[12];
                                        conv.a[(4<<(t_-4))+l] = temp[2];
                                        conv.a[(5<<(t_-4))+l] = temp[10];
                                        conv.a[(6<<(t_-4))+l] = temp[6];
                                        conv.a[(7<<(t_-4))+l] = temp[14];
                                        conv.a[(8<<(t_-4))+l] = temp[1];
                                        conv.a[(9<<(t_-4))+l] = temp[9];
                                        conv.a[(10<<(t_-4))+l] = temp[5];
                                        conv.a[(11<<(t_-4))+l] = temp[13];
                                        conv.a[(12<<(t_-4))+l] = temp[3];
                                        conv.a[(13<<(t_-4))+l] = temp[11];
                                        conv.a[(14<<(t_-4))+l] = temp[7];
                                        conv.a[(15<<(t_-4))+l] = temp[15];
                                    }
                                }
                            }
                            else {
                                int th=th_-4;
                                for(int l=th<<(t_-6);l<(th+1)<<(t_-6);l++){
                                    __m512i sum = _mm512_setzero_epi32();
                                    bool f=false;
                                    for(int k=0;k<8;k++){
                                        int j=(k<<(t_-4))+l;
                                        if(input.b[j]==i){
                                            sum = _mm512_add_epi32(sum,rr512[k]);
                                            f = true;
                                        }
                                    }
                                    if(f){
                                        _mm512_storeu_epi32(temp,sum);
                                        conv.b[l] = temp[0];
                                        conv.b[(1<<(t_-4))+l] = temp[8];
                                        conv.b[(2<<(t_-4))+l] = temp[4];
                                        conv.b[(3<<(t_-4))+l] = temp[12];
                                        conv.b[(4<<(t_-4))+l] = temp[2];
                                        conv.b[(5<<(t_-4))+l] = temp[10];
                                        conv.b[(6<<(t_-4))+l] = temp[6];
                                        conv.b[(7<<(t_-4))+l] = temp[14];
                                        conv.b[(8<<(t_-4))+l] = temp[1];
                                        conv.b[(9<<(t_-4))+l] = temp[9];
                                        conv.b[(10<<(t_-4))+l] = temp[5];
                                        conv.b[(11<<(t_-4))+l] = temp[13];
                                        conv.b[(12<<(t_-4))+l] = temp[3];
                                        conv.b[(13<<(t_-4))+l] = temp[11];
                                        conv.b[(14<<(t_-4))+l] = temp[7];
                                        conv.b[(15<<(t_-4))+l] = temp[15];
                                    }
                                }
                            }
                        }
                    }
/*#ifndef LOCAL
#pragma omp parallel num_threads(8) proc_bind(spread)
#endif
                    {
#ifndef LOCAL
#pragma omp for
#endif
                        for(int th=0;th<8;th++){
                            for(int l=th<<(t_-7);l<(th+1)<<(t_-7);l++){
                                __m512i sum = _mm512_setzero_epi32();
                                bool f=false;
                                for(int k=0;k<8;k++){
                                    int j=(k<<(t_-4))+l;
                                    if(j<input.N&&input.a[input.N-1-j]==i){
                                        sum = _mm512_add_epi32(sum,rr512[k]);
                                        f=true;
                                    }
                                }
                                if(f){
                                    uint temp[16];
                                    _mm512_storeu_epi32(temp,sum);
                                    for(int k=0;k<16;k++){
                                        int j=(k<<(t_-4))+l;
                                        conv.a[j] = temp[BTR_16(k)];
                                    }
                                }
                                else {
                                    for(int k=0;k<16;k++){
                                        conv.a[(k<<(t_-4))+l] = 0;
                                    }
                                }
                            }
                        }
                    }
#ifndef LOCAL
#pragma omp parallel num_threads(8) proc_bind(spread)
#endif
                    {
#ifndef LOCAL
#pragma omp for
#endif
                        for(int th=0;th<8;th++){
                            for(int l=th<<(t_-7);l<(th+1)<<(t_-7);l++){
                                __m512i sum = _mm512_setzero_epi32();
                                bool f=false;
                                for(int k=0;k<8;k++){
                                    int j=(k<<(t_-4))+l;
                                    if(j<input.N&&input.b[j]==i){
                                        sum = _mm512_add_epi32(sum,rr512[k]);
                                        f=true;
                                    }
                                }
                                if(f){
                                    uint temp[16];
                                    _mm512_storeu_epi32(temp,sum);
                                    for(int k=0;k<16;k++){
                                        int j=(k<<(t_-4))+l;
                                        conv.b[j] = temp[BTR_16(k)];
                                    }
                                }
                                else {
                                    for(int k=0;k<16;k++){
                                        conv.b[(k<<(t_-4))+l] = 0;
                                    }
                                }
                            }
                        }
                    }*/
                }
                conv.run();
#ifndef LOCAL
#pragma omp parallel num_threads(8) proc_bind(spread)
#endif
                {
#ifndef LOCAL
#pragma omp for
#endif
                    for(int th=0;th<8;th++){
                        for(int j=th<<(t-3);j<min((th+1)<<(t-3),(int)input.N);j++){
                            ret[j] += conv.a[j] + conv.a[j+input.N];
                        }
                    }
                }
                
                /*if(i+1<input.K){
#ifndef LOCAL
#pragma omp parallel num_threads(8) proc_bind(spread)
#endif
                    {
#ifndef LOCAL
#pragma omp sections
#endif
                        {
#ifndef LOCAL
#pragma omp section
#endif
                            {
                                for(int k=0;k<(1<<(t_-2));k++){
                                    conv.a[k]=0;
                                }
                            }
#ifndef LOCAL
#pragma omp section
#endif
                            {
                                for(int k=(1<<(t_-2));k<(2<<(t_-2));k++){
                                    conv.a[k]=0;
                                }
                            }
#ifndef LOCAL
#pragma omp section
#endif
                            {
                                for(int k=(2<<(t_-2));k<(3<<(t_-2));k++){
                                    conv.a[k]=0;
                                }
                            }
#ifndef LOCAL
#pragma omp section
#endif
                            {
                                for(int k=(3<<(t_-2));k<(4<<(t_-2));k++){
                                    conv.a[k]=0;
                                }
                            }
#ifndef LOCAL
#pragma omp section
#endif
                            {
                                for(int k=0;k<(1<<(t_-2));k++){
                                    conv.b[k]=0;
                                }
                            }
#ifndef LOCAL
#pragma omp section
#endif
                            {
                                for(int k=(1<<(t_-2));k<(2<<(t_-2));k++){
                                    conv.b[k]=0;
                                }
                            }
#ifndef LOCAL
#pragma omp section
#endif
                            {
                                for(int k=(2<<(t_-2));k<(3<<(t_-2));k++){
                                    conv.b[k]=0;
                                }
                            }
#ifndef LOCAL
#pragma omp section
#endif
                            {
                                for(int k=(3<<(t_-2));k<(4<<(t_-2));k++){
                                    conv.b[k]=0;
                                }
                            }
                        }
                    }
                }*/
            }
        }
    }
    else {
        //input.b.resize(1<<t,-1);
        //input.a.resize(1<<t,-1);
        for(int i_th=0;i_th<1;i_th++){
            for(int i=i_th;i<input.K;i+=1){
                const int t_ = 18;
                auto &conv = conv_18;
                {
                    if(1){
    #ifndef LOCAL
    #pragma omp parallel num_threads(8) proc_bind(spread)
    #endif
                        {
    #ifndef LOCAL
    #pragma omp sections
    #endif
                            {
    #ifndef LOCAL
    #pragma omp section
    #endif
                                {
                                    for(int k=0;k<(1<<(t_-2));k++){
                                        conv.a[k]=0;
                                    }
                                }
    #ifndef LOCAL
    #pragma omp section
    #endif
                                {
                                    for(int k=(1<<(t_-2));k<(2<<(t_-2));k++){
                                        conv.a[k]=0;
                                    }
                                }
    #ifndef LOCAL
    #pragma omp section
    #endif
                                {
                                    for(int k=(2<<(t_-2));k<(3<<(t_-2));k++){
                                        conv.a[k]=0;
                                    }
                                }
    #ifndef LOCAL
    #pragma omp section
    #endif
                                {
                                    for(int k=(3<<(t_-2));k<(4<<(t_-2));k++){
                                        conv.a[k]=0;
                                    }
                                }
    #ifndef LOCAL
    #pragma omp section
    #endif
                                {
                                    for(int k=0;k<(1<<(t_-2));k++){
                                        conv.b[k]=0;
                                    }
                                }
    #ifndef LOCAL
    #pragma omp section
    #endif
                                {
                                    for(int k=(1<<(t_-2));k<(2<<(t_-2));k++){
                                        conv.b[k]=0;
                                    }
                                }
    #ifndef LOCAL
    #pragma omp section
    #endif
                                {
                                    for(int k=(2<<(t_-2));k<(3<<(t_-2));k++){
                                        conv.b[k]=0;
                                    }
                                }
    #ifndef LOCAL
    #pragma omp section
    #endif
                                {
                                    for(int k=(3<<(t_-2));k<(4<<(t_-2));k++){
                                        conv.b[k]=0;
                                    }
                                }
                            }
                        }
                    }
                    __m512i rr512[8];
                    for(int k=0;k<8;k++){
                        rr512[k] = conv.v_deep[k];
                    }
#ifndef LOCAL
#pragma omp parallel num_threads(8) proc_bind(spread)
#endif
                    {
#ifndef LOCAL
#pragma omp for
#endif
                        for(int th_=0;th_<8;th_++){
                            uint temp[16];
                            if(th_<4){
                                int th=th_;
                                for(int l=th<<(t_-6);l<(th+1)<<(t_-6);l++){
                                    __m512i sum = _mm512_setzero_epi32();
                                    bool f=false;
                                    for(int k=0;k<8;k++){
                                        int j=(k<<(t_-4))+l;
                                        if(j<input.N&&input.a[input.N-1-j]==i){
                                            sum = _mm512_add_epi32(sum,rr512[k]);
                                            f = true;
                                        }
                                    }
                                    if(f){
                                        _mm512_storeu_epi32(temp,sum);
                                        conv.a[l] = temp[0];
                                        conv.a[(1<<(t_-4))+l] = temp[8];
                                        conv.a[(2<<(t_-4))+l] = temp[4];
                                        conv.a[(3<<(t_-4))+l] = temp[12];
                                        conv.a[(4<<(t_-4))+l] = temp[2];
                                        conv.a[(5<<(t_-4))+l] = temp[10];
                                        conv.a[(6<<(t_-4))+l] = temp[6];
                                        conv.a[(7<<(t_-4))+l] = temp[14];
                                        conv.a[(8<<(t_-4))+l] = temp[1];
                                        conv.a[(9<<(t_-4))+l] = temp[9];
                                        conv.a[(10<<(t_-4))+l] = temp[5];
                                        conv.a[(11<<(t_-4))+l] = temp[13];
                                        conv.a[(12<<(t_-4))+l] = temp[3];
                                        conv.a[(13<<(t_-4))+l] = temp[11];
                                        conv.a[(14<<(t_-4))+l] = temp[7];
                                        conv.a[(15<<(t_-4))+l] = temp[15];
                                    }
                                }
                            }
                            else {
                                int th=th_-4;
                                for(int l=th<<(t_-6);l<(th+1)<<(t_-6);l++){
                                    __m512i sum = _mm512_setzero_epi32();
                                    bool f=false;
                                    for(int k=0;k<8;k++){
                                        int j=(k<<(t_-4))+l;
                                        if(input.b[j]==i){
                                            sum = _mm512_add_epi32(sum,rr512[k]);
                                            f = true;
                                        }
                                    }
                                    if(f){
                                        _mm512_storeu_epi32(temp,sum);
                                        conv.b[l] = temp[0];
                                        conv.b[(1<<(t_-4))+l] = temp[8];
                                        conv.b[(2<<(t_-4))+l] = temp[4];
                                        conv.b[(3<<(t_-4))+l] = temp[12];
                                        conv.b[(4<<(t_-4))+l] = temp[2];
                                        conv.b[(5<<(t_-4))+l] = temp[10];
                                        conv.b[(6<<(t_-4))+l] = temp[6];
                                        conv.b[(7<<(t_-4))+l] = temp[14];
                                        conv.b[(8<<(t_-4))+l] = temp[1];
                                        conv.b[(9<<(t_-4))+l] = temp[9];
                                        conv.b[(10<<(t_-4))+l] = temp[5];
                                        conv.b[(11<<(t_-4))+l] = temp[13];
                                        conv.b[(12<<(t_-4))+l] = temp[3];
                                        conv.b[(13<<(t_-4))+l] = temp[11];
                                        conv.b[(14<<(t_-4))+l] = temp[7];
                                        conv.b[(15<<(t_-4))+l] = temp[15];
                                    }
                                }
                            }
                        }
                    }
                }
                conv.run();
#ifndef LOCAL
#pragma omp parallel num_threads(8) proc_bind(spread)
#endif
                {
#ifndef LOCAL
#pragma omp for
#endif
                    for(int th=0;th<8;th++){
                        for(int j=th<<(t-3);j<min((th+1)<<(t-3),(int)input.N);j++){
                            ret[j] += conv.a[j] + conv.a[j+input.N];
                        }
                    }
                }
                
                /*if(i+1<input.K){
#ifndef LOCAL
#pragma omp parallel num_threads(8) proc_bind(spread)
#endif
                    {
#ifndef LOCAL
#pragma omp sections
#endif
                        {
#ifndef LOCAL
#pragma omp section
#endif
                            {
                                for(int k=0;k<(1<<(t_-2));k++){
                                    conv.a[k]=0;
                                }
                            }
#ifndef LOCAL
#pragma omp section
#endif
                            {
                                for(int k=(1<<(t_-2));k<(2<<(t_-2));k++){
                                    conv.a[k]=0;
                                }
                            }
#ifndef LOCAL
#pragma omp section
#endif
                            {
                                for(int k=(2<<(t_-2));k<(3<<(t_-2));k++){
                                    conv.a[k]=0;
                                }
                            }
#ifndef LOCAL
#pragma omp section
#endif
                            {
                                for(int k=(3<<(t_-2));k<(4<<(t_-2));k++){
                                    conv.a[k]=0;
                                }
                            }
#ifndef LOCAL
#pragma omp section
#endif
                            {
                                for(int k=0;k<(1<<(t_-2));k++){
                                    conv.b[k]=0;
                                }
                            }
#ifndef LOCAL
#pragma omp section
#endif
                            {
                                for(int k=(1<<(t_-2));k<(2<<(t_-2));k++){
                                    conv.b[k]=0;
                                }
                            }
#ifndef LOCAL
#pragma omp section
#endif
                            {
                                for(int k=(2<<(t_-2));k<(3<<(t_-2));k++){
                                    conv.b[k]=0;
                                }
                            }
#ifndef LOCAL
#pragma omp section
#endif
                            {
                                for(int k=(3<<(t_-2));k<(4<<(t_-2));k++){
                                    conv.b[k]=0;
                                }
                            }
                        }
                    }
                }*/
            }
        }
    }
    
/*#ifndef LOCAL
#pragma omp parallel num_threads(8) proc_bind(spread)
#endif
    {
#ifndef LOCAL
#pragma omp for
#endif
        for(int k=0;k<8;k++){
            for(int i=0;i<num_thread;i++)
            {
                for(int j=(k<<(t-3));j<((k+1)<<(t-3));j+=16){
                    _mm512_storeu_epi32(ret+j,_mm512_add_epi32(set32(ret+j),set32(ret_thread[i]+j)));
                }
            }
        }
    }*/
    
    if(t==19)conv_20.finalize(ret);
    else if(t==18)conv_19.finalize(ret);
    else conv_18.finalize(ret);
     
    std::pair<int,int> best[8];
#ifndef LOCAL
        #pragma omp parallel num_threads(8) proc_bind(spread)
#endif
    {
#ifndef LOCAL
#pragma omp for
#endif
        for(int k=0;k<8;k++){
            best[k] = make_pair(ret[0],0);
            /*for(int j=(k<<(t-3));j<min((k+1)<<(t-3),(int)input.N);j++){
                if(best[k].first < ret[j]){
                    best[k].first = ret[j];
                    best[k].second = j;
                }
            }*/
            __m512i v = _mm512_setzero_epi32();
            for(int j=(k<<(t-3));j<((k+1)<<(t-3));j+=8){
                //auto w = _mm512_setr_epi32(ret[j],j,ret[j+1],j+1,ret[j+2],j+2,ret[j+3],j+3,ret[j+4],j+4,ret[j+5],j+5,ret[j+6],j+6,ret[j+7],j+7);
                auto w = _mm512_setr_epi32(j,ret[j],j+1,ret[j+1],j+2,ret[j+2],j+3,ret[j+3],j+4,ret[j+4],j+5,ret[j+5],j+6,ret[j+6],j+7,ret[j+7]);
                v = _mm512_max_epi64(v,w);
            }
            uint temp[16];
            _mm512_storeu_epi32(temp,v);
            for(int j=0;j<8;j++){
                if(best[k].first < temp[2*j+1]){
                    best[k].first = temp[2*j+1];
                    best[k].second = temp[2*j];
                }
            }
        }
    }
    output.n_spells = input.N;
    output.n_shifts = 0;
    for(int k=0;k<8;k++){
        if(output.n_spells > input.N - best[k].first){
            output.n_spells = input.N - best[k].first;
            output.n_shifts = best[k].second;
        }
    }
    output.n_shifts = (output.n_shifts+1)%input.N;
    
    /*std::pair<int,int> best(input.N-ret[0],0);
    for(int i=0;i<input.N;i++){
        if(best.first > input.N-ret[i]){
            best.first = input.N-ret[i];
            best.second = i;
        }
    }
    output.n_spells = best.first;
    output.n_shifts = (best.second+1)%input.N;*/
}
//#pragma optimize("O3", off)
//#pragma optimize("O2", on)

#ifdef LOCAL
//using namespace std;
int main(){
    input_t in;
    output_t out;
    
    in.N = 1<<17;
    in.K = 16;
    in.a.resize(in.N);
    in.b.resize(in.N);
    for(int i=0;i<in.N;i++){
        in.a[i] = rand()%in.K;
    }
    for(int i=0;i<in.N;i++){
        in.b[i] = rand()%in.K;
    }
    
    /*std::cin>>in.N>>in.K;
    in.a.resize(in.N);
    in.b.resize(in.N);
    for(int i=0;i<in.N;i++){
        int x;
        std::cin>>x;
        in.a[i] = x;
    }
    for(int i=0;i<in.N;i++){
        int x;
        std::cin>>x;
        in.b[i] = x;
    }*/
    
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    //cerr<<ts.tv_sec<<"."<<ts.tv_nsec<<endl;
    double start = ts.tv_sec + ts.tv_nsec * 1e-9;
    //std::cerr<<clock()<<std::endl;
    solve(in,out);
    //std::cerr<<clock()<<std::endl;
    clock_gettime(CLOCK_REALTIME, &ts);
    double end = ts.tv_sec + ts.tv_nsec * 1e-9;
    cerr.precision(20);
    cerr<<end-start<<endl;
    //cerr<<ts.tv_sec<<"."<<ts.tv_nsec<<endl;
    
    std::cerr<<out.n_spells<<" "<<out.n_shifts<<std::endl;
}
#endif



