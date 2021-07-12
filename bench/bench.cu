#include <cstdio>
#include <cstring>
#include <cassert>
#include <unistd.h>

#include "fixnum/warp_fixnum.cu"
#include "array/fixnum_array.h"
#include "functions/quorem_preinv.cu"
#include "functions/modexp.cu"
#include "functions/multi_modexp.cu"
#include "modnum/modnum_monty_redc.cu"
#include "modnum/modnum_monty_cios.cu"
#include "functions/paillier_decrypt.cu"
#include "functions/paillier_encrypt.cu"

using namespace std;
using namespace cuFIXNUM;

/*
k-bit * k-bit = 2k-bit
res = a * b
res - low k bit
*/
template <typename fixnum>
struct mul_lo
{
    __device__ void operator()(fixnum &res, fixnum a, fixnum b)
    {
        fixnum s;
        fixnum::mul_lo(s, a, b);
        res = s;
    }
};

/*
k-bit * k-bit = 2k-bit
res = a * b
res_hi - high k bit
res_lo - low k bit
*/
template <typename fixnum>
struct mul_wide
{
    __device__ void operator()(fixnum &res_hi, fixnum &res_lo, fixnum a, fixnum b)
    {
        fixnum rr, ss;
        fixnum::mul_wide(ss, rr, a, b);
        res_hi = ss;
        res_lo = rr;
    }
};

template <typename fixnum>
struct sqr_wide
{
    __device__ void operator()(fixnum &r, fixnum a)
    {
        fixnum rr, ss;
        fixnum::sqr_wide(ss, rr, a);
        r = ss;
    }
};

/*
modexp<modnum_tp>::modexp(fixnum mod, fixnum exp)
res = x^exp % mod
*/
template <typename modnum>
struct my_modexp
{
    typedef typename modnum::fixnum fixnum;

    __device__ void operator()(fixnum &res, fixnum x, fixnum exp, fixnum mod)
    {
        modexp<modnum> me(mod, exp);
        fixnum zz;
        me(zz, x);
        res = zz;
    };
};

/*
same to modexp   
*/
template <typename modnum>
struct my_multi_modexp
{
    typedef typename modnum::fixnum fixnum;

    __device__ void operator()(fixnum &z, fixnum x, fixnum e, fixnum mod)
    {
        multi_modexp<modnum> mme(mod);
        fixnum zz;
        mme(zz, x, e);
        z = zz;
    };
};

template <typename modnum>
struct my_mult_mod
{
    typedef typename modnum::fixnum fixnum;

    __device__ void operator()(fixnum &res, fixnum x, fixnum y, fixnum mod)
    {
        quorem_preinv<fixnum> pmod(mod);
        fixnum hi, lo;
        fixnum::mul_wide(hi, lo, x, y);

        fixnum zz;
        pmod(zz, hi, lo);
        res = zz;
    };
};

uint8_t *long2bytes(long *input, int length, int fn_bytes)
{
    uint8_t *bytes = new uint8_t[length * fn_bytes];
    memset(bytes, 0x00, length * fn_bytes);
    for (int i = 0; i < length; i++)
    {
        uint8_t *ptr = (uint8_t *)(&input[i]);
        bytes[(i + 1) * fn_bytes - 1] = ptr[0];
        bytes[(i + 1) * fn_bytes - 2] = ptr[1];
        bytes[(i + 1) * fn_bytes - 3] = ptr[2];
        bytes[(i + 1) * fn_bytes - 4] = ptr[3];
    }

    return bytes;
}

long *bytes2long(uint8_t *bytes, int length, int fn_bytes = 256)
{
    long *res = new long[length / fn_bytes];
    for (int i = 0; i < length / fn_bytes; i++)
    {
        uint8_t *ptr = (uint8_t *)&res[i];
        ptr[0] = bytes[(i + 1) * fn_bytes - 1];
        ptr[1] = bytes[(i + 1) * fn_bytes - 2];
        ptr[2] = bytes[(i + 1) * fn_bytes - 3];
        ptr[3] = bytes[(i + 1) * fn_bytes - 4];
    }

    return res;
}

void printBytes(uint8_t *bytes, int length, int fn_bytes = 256)
{
    for (int i = 0; i < length; i++)
    {
        printf("%x ", bytes[i]);
        if ((i + 1) % fn_bytes == 0)
            printf("\n");
    }
}

template <int fn_bytes, typename word_fixnum, template <typename> class Func>
void bench(int nelts)
{
    typedef warp_fixnum<fn_bytes, word_fixnum> fixnum;
    typedef fixnum_array<fixnum> fixnum_array;

    if (nelts == 0)
    {
        puts(" -*-  nelts == 0; skipping...  -*-");
        return;
    }

    uint8_t *input_0 = new uint8_t[fn_bytes * nelts];
    uint8_t *input_1 = new uint8_t[fn_bytes * nelts];
    uint8_t *input_2 = new uint8_t[fn_bytes * nelts];
    uint8_t *data = new uint8_t[fn_bytes * nelts];
    memset(input_0, 0x00, fn_bytes * nelts);
    memset(input_1, 0x00, fn_bytes * nelts);
    memset(input_2, 0x00, fn_bytes * nelts);
    memset(data, 0x00, fn_bytes * nelts);
    for (int i = 0; i < fn_bytes * nelts; i++)
    {
        input_0[i] = 0x11;
        input_1[i] = 0x12;
        input_2[i] = 0x13;
    }

    /*
    for (int i = 0; i < nelts * fn_bytes; i++)
    {
        input_0[i] = 0x01;
        input_1[i] = 0x01;
        input_2[i] = 0x02;
    }
    */

    fixnum_array *res_hi, *res_lo, *in_0, *in_1, *in_2;
    in_0 = fixnum_array::create(input_0, fn_bytes * nelts, fn_bytes);
    in_1 = fixnum_array::create(input_1, fn_bytes * nelts, fn_bytes);
    in_2 = fixnum_array::create(input_2, fn_bytes * nelts, fn_bytes);
    res_hi = fixnum_array::create(nelts);
    res_lo = fixnum_array::create(nelts);

    // warm up
    //fixnum_array::template map<Func>(res_lo, in_0, in_0, in_1);

    clock_t c = clock();
    fixnum_array::template map<Func>(res_lo, in_0, in_1, in_2);
    c = clock() - c;

    //printf("res lo\n");
    res_lo->retrieve_all(data, nelts * fn_bytes, &nelts);
    //printBytes(data, nelts * fn_bytes, fn_bytes);
    //printf("\n");

    double secinv = (double)CLOCKS_PER_SEC / c;
    double total_MiB = fixnum::BYTES * (double)nelts / (1 << 20);
    printf(" %4d   %3d    %6.1f   %7.3f  %12.1f\n",
           fixnum::BITS, fixnum::digit::BITS, total_MiB,
           1 / secinv, nelts * 1e-3 * secinv);

    delete in_0;
    delete in_1;
    delete in_2;
    delete res_hi;
    delete res_lo;
    delete[] input_0;
    delete[] input_1;
    delete[] input_2;
    delete[] data;
}

template <template <typename> class Func>
void bench_func(const char *fn_name, int nelts)
{
    printf("Function: %s, #elts: %de3\n", fn_name, (int)(nelts * 1e-3));
    printf("fixnum digit  total data   time       Kops/s\n");
    printf(" bits  bits     (MiB)    (seconds)\n");
    bench<4, u32_fixnum, Func>(nelts);
    bench<8, u32_fixnum, Func>(nelts);
    bench<16, u32_fixnum, Func>(nelts);
    bench<32, u32_fixnum, Func>(nelts);
    bench<64, u32_fixnum, Func>(nelts);
    bench<128, u32_fixnum, Func>(nelts);
    puts("");

    bench<8, u64_fixnum, Func>(nelts);
    bench<16, u64_fixnum, Func>(nelts);
    bench<32, u64_fixnum, Func>(nelts);
    bench<64, u64_fixnum, Func>(nelts);
    bench<128, u64_fixnum, Func>(nelts);
    bench<256, u64_fixnum, Func>(nelts);
    puts("");
}

template <typename fixnum>
using modexp_redc = my_modexp<modnum_monty_redc<fixnum>>;

template <typename fixnum>
using modexp_cios = my_modexp<modnum_monty_cios<fixnum>>;

template <typename fixnum>
using multi_modexp_redc = my_multi_modexp<modnum_monty_redc<fixnum>>;

template <typename fixnum>
using multi_modexp_cios = my_multi_modexp<modnum_monty_cios<fixnum>>;

template <typename fixnum>
using multi_mod_redc = my_mult_mod<modnum_monty_redc<fixnum>>;

template <typename fixnum>
using multi_mod_cios = my_mult_mod<modnum_monty_cios<fixnum>>;

/*
template< typename fixnum >
struct pencrypt {
    __device__ void operator()(fixnum &z, fixnum p, fixnum q, fixnum r, fixnum m) {
        fixnum n, zz;
        fixnum::mul_lo(n, p, q);
        paillier_encrypt<fixnum> enc(n);
        enc(zz, m, r);
        z = zz;
    };
};
*/

template <typename fixnum>
struct pencrypt
{
    __device__ void operator()(fixnum &z, fixnum n, fixnum m, fixnum r)
    {
        fixnum zz;
        paillier_encrypt<fixnum> enc(n);
        enc(zz, m, r);
        z = zz;
    };
};

template <typename fixnum>
struct paillier_encrypt_manner
{
    __device__ void operator()(fixnum &encrypted, fixnum n, fixnum n_2, fixnum g, fixnum m, fixnum r)
    {
        fixnum g_m, r_n, res;
        multi_modexp<modnum_monty_redc<fixnum>> mme(n_2);
        quorem_preinv<fixnum> mulmod(n_2);
        mme(g_m, g, m);
        mme(r_n, r, n);
        
        fixnum hi, lo;
        fixnum::mul_wide(hi, lo, g_m, r_n);
        mulmod(res, hi, lo);
        encrypted = res;
    };
};

/*
template< typename fixnum >
struct pdecrypt {
    __device__ void operator()(fixnum &z, fixnum ct, fixnum p, fixnum q, fixnum r, fixnum m) {
        
        if (fixnum::cmp(p, q) == 0
              || fixnum::cmp(r, p) == 0
              || fixnum::cmp(r, q) == 0) {
            printf("equal \n");
            z = fixnum::zero();
            return;
        }
        
        paillier_decrypt<fixnum> dec(p, q);
        fixnum n, zz;
        dec(zz, fixnum::zero(), ct);
        //dec(z, fixnum::zero(), ct);
        //fixnum::mul_lo(n, p, q);
        //quorem_preinv<fixnum> qr(n);
        //qr(m, fixnum::zero(), m);

        // z = (z != m)
        //z = fixnum::digit( !! fixnum::cmp(zz, m));
        z = zz;
    };
};
*/

template <typename fixnum>
struct pdecrypt
{
    __device__ void operator()(fixnum &z, fixnum ct, fixnum p, fixnum q)
    {
        paillier_decrypt<fixnum> dec(p, q);
        fixnum zz;
        dec(zz, fixnum::zero(), ct);
        z = zz;
    };
};

template <typename fixnum>
struct paillier_decrypt_manner
{
    __device__ void operator()(fixnum &decrypted, fixnum ctx, fixnum lamda, fixnum n, fixnum n_2, fixnum g_lamda_inv)
    {
        fixnum ctx_lamda, res;
        multi_modexp<modnum_monty_redc<fixnum>> mme_n2(n_2);
        quorem_preinv<fixnum> mulmod(n);
        
        //q = L(c^lamda mod n^2)
        mme_n2(ctx_lamda, ctx, lamda);
        fixnum::sub(ctx_lamda ,ctx_lamda ,fixnum::one());
        fixnum q, r;
        mulmod(q, r, fixnum::zero(), ctx_lamda);
        
        //q*g_lamda_inv
        fixnum hi, lo;
        fixnum::mul_wide(hi, lo, q, g_lamda_inv);
        mulmod(res, hi, lo);
        decrypted = res;
    };
};

void host_function(int num)
{

    // fixnum represents 256-byte numbers, using a 64-bit "basic fixnum".
    typedef warp_fixnum<256, u64_fixnum> fixnum;
    typedef fixnum_array<fixnum> fixnum_array;

    int nelts = num, byteslen = 256;
    fixnum_array *ct, *pt, *p, *q, *n, *g, *n2, *lamda, *g_lamda_inv;

    uint8_t *private_key_p = new uint8_t[nelts * byteslen];
    uint8_t *private_key_q = new uint8_t[nelts * byteslen];
    uint8_t *private_key_lamda = new uint8_t[nelts * byteslen];
    uint8_t *private_key_g_lamda_inv = new uint8_t[nelts * byteslen];
    uint8_t *public_key_n = new uint8_t[nelts * byteslen];
    uint8_t *public_key_g = new uint8_t[nelts * byteslen];
    uint8_t *public_key_n2 = new uint8_t[nelts * byteslen];
    uint8_t *data = new uint8_t[nelts * byteslen];
    memset(private_key_p, 0x00, nelts * byteslen);
    memset(private_key_q, 0x00, nelts * byteslen);
    memset(private_key_lamda, 0x00, nelts * byteslen);
    memset(private_key_g_lamda_inv, 0x00, nelts * byteslen);
    memset(public_key_n, 0x00, nelts * byteslen);
    memset(public_key_g, 0x00, nelts * byteslen);
    memset(public_key_n2, 0x00, nelts * byteslen);
    memset(data, 0x00, nelts * byteslen);

    for (int i = 0; i < nelts; i++)
    {
        private_key_p[i * byteslen] = 0x05;
        private_key_q[i * byteslen] = 0x07;
        private_key_lamda[i * byteslen] = 0x0c;
        private_key_g_lamda_inv[i * byteslen] = 0x03;

        public_key_n[i * byteslen] = 0x23;
        public_key_g[i * byteslen] = 0x24;
        public_key_n2[i * byteslen] = 0xc9;
        public_key_n2[i * byteslen + 1] = 0x04;
    }

    for (int i = 0; i < nelts * byteslen; i++)
    {
        private_key_p[i] = 0x05;
        private_key_q[i] = 0x07;
        private_key_lamda[i] = 0x0c;
        private_key_g_lamda_inv[i] = 0x03;

        public_key_n[i] = 0x23;
        public_key_g[i] = 0x24;
        public_key_n2[i] = 0xc9;
    }

    p = fixnum_array::create(private_key_p, nelts * byteslen, byteslen);
    q = fixnum_array::create(private_key_q, nelts * byteslen, byteslen);
    n = fixnum_array::create(public_key_n, nelts * byteslen, byteslen);
    g = fixnum_array::create(public_key_g, nelts * byteslen, byteslen);
    n2 = fixnum_array::create(public_key_n2, nelts * byteslen, byteslen);
    lamda = fixnum_array::create(private_key_lamda, nelts * byteslen, byteslen);
    g_lamda_inv = fixnum_array::create(private_key_g_lamda_inv, nelts * byteslen, byteslen);

    ct = fixnum_array::create(nelts);
    pt = fixnum_array::create(nelts);
    printf("p->length() = %d \n", p->length());
    printf("q->length() = %d \n", q->length());
    printf("n->length() = %d \n", n->length());
    printf("g->length() = %d \n", g->length());
    printf("n2->length() = %d \n", n2->length());
    printf("lamda->length() = %d \n", lamda->length());
    printf("g_lamda_inv->length() = %d \n", g_lamda_inv->length());
    printf("ct->length() = %d \n", ct->length());
    printf("pt->length() = %d \n", pt->length());

    for (int i = 0; i < nelts; i++)
    {
        data[i * byteslen] = 0x03;
    }

    for (int i = 0; i < nelts * byteslen; i++)
    {
        data[i] = 0x11;
    }


    fixnum_array *m = fixnum_array::create(data, nelts * byteslen, byteslen);
    fixnum_array *r = fixnum_array::create(data, nelts * byteslen, byteslen);
    printf("m->length() = %d \n", m->length());
    printf("r->length() = %d \n", r->length());

    //warm up
    //fixnum_array::template map<paillier_encrypt_manner>(ct, n, n2, g, m, r);

    clock_t t = clock();
    //fixnum_array::template map<paillier_encrypt_manner>(ct, n, n2, g, m, r);
    t = clock() - t;
    double secinv = (double)CLOCKS_PER_SEC / t;
    printf("manner encrypt time cost:%f \n", 1 / secinv);

    t = clock();
    fixnum_array::template map<pencrypt>(ct, n, m, r);
    t = clock() - t;
    secinv = (double)CLOCKS_PER_SEC / t;
    printf("encrypt time cost:%f \n", 1 / secinv);

    //warm up
    //fixnum_array::template map<paillier_decrypt_manner>(pt, ct, lamda, n, n2, g_lamda_inv);

    t = clock();
    fixnum_array::template map<paillier_decrypt_manner>(pt, ct, lamda, n, n2, g_lamda_inv);
    t = clock() - t;
    secinv = (double)CLOCKS_PER_SEC / t;
    printf("manner decrypt time cost:%f \n", 1 / secinv);

    uint8_t *res = new uint8_t[nelts * fixnum::BYTES];
    memset(res, 0x00, nelts * fixnum::BYTES);
    pt->retrieve_all(res, nelts * fixnum::BYTES, &nelts);
    printBytes(res, 1, fixnum::BYTES);

    t = clock();
    //fixnum_array::template map<pdecrypt>(pt, ct, p, q);
    t = clock() - t;
    secinv = (double)CLOCKS_PER_SEC / t;
    printf("decrypt time cost:%f \n", 1 / secinv);

    /*
    int len = 1000;
    printf("p \n");
    memset(data, 0x00, nelts * fixnum::BYTES);
    p->retrieve_all(data, nelts * fixnum::BYTES, &len);
    printBytes(data, nelts * fixnum::BYTES);
    printf("q \n");
    memset(data, 0x00, nelts * fixnum::BYTES);
    q->retrieve_all(data, nelts * fixnum::BYTES, &len);
    printBytes(data, nelts * fixnum::BYTES);
    printf("m \n");
    memset(data, 0x00, nelts * fixnum::BYTES);
    m->retrieve_all(data, nelts * fixnum::BYTES, &len);
    printBytes(data, nelts * fixnum::BYTES);
    printf("r \n");
    memset(data, 0x00, nelts * fixnum::BYTES);
    r->retrieve_all(data, nelts * fixnum::BYTES, &len);
    printBytes(data, nelts * fixnum::BYTES);

    printf("ct \n");
    memset(data, 0x00, nelts * fixnum::BYTES);
    ct->retrieve_all(data, nelts * fixnum::BYTES, &len);
    printBytes(data, nelts * fixnum::BYTES);

    printf("pt \n");
    memset(data, 0x00, nelts * fixnum::BYTES);
    pt->retrieve_all(data, nelts * fixnum::BYTES, &len);
    printBytes(data, nelts * fixnum::BYTES);
    */

    delete[] private_key_p;
    delete[] private_key_q;
    delete[] public_key_n;
    delete[] data;
    delete ct;
    delete pt;
    delete p;
    delete q;
}

int main(int argc, char *argv[])
{
    long m = 1;
    if (argc > 1)
        m = atol(argv[1]);
    m = std::max(m, 1000L);
    
    /*
    bench_func<mul_lo>("mul_lo", m);
    puts("");
    bench_func<mul_wide>("mul_wide", m);
    puts("");
    bench_func<sqr_wide>("sqr_wide", m);
    puts("");
    bench_func<modexp_redc>("modexp redc", m);
    puts("");
    bench_func<modexp_cios>("modexp cios", m);
    puts("");
    bench_func<multi_modexp_redc>("multi modexp redc", m);
    puts("");
    */
    
    bench_func<modexp_cios>("multi modexp cios", m);
    puts("");
    bench_func<modexp_redc>("multi modexp redc", m);
    puts("");
    
    bench_func<multi_mod_redc>("mult mod redc", m);
    puts("");
    bench_func<multi_mod_cios>("mult mod cios", m);
    puts("");

    return 0;
}
