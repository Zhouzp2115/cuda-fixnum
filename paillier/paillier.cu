#include "paillier.h"

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

typedef warp_fixnum<256, u64_fixnum> fixnum;
typedef fixnum_array<fixnum> fixnum_array_;

template <typename fixnum>
struct paillier_encrypt_func
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
struct paillier_decrypt_func
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

void paillierSetGPUDevice(int id)
{
    cudaSetDevice(id);
}

PublicKey::PublicKey(char *key_file)
{
    FILE *file = fopen(key_file, "rb");
    fread(n, sizeof(uint8_t), 256, file);
    fread(n2, sizeof(uint8_t), 256, file);
    fclose(file);
    
    //init array_n with 10000
    uint8_t *repeated = repeat(n, 10000);
    array_n = (void *)fixnum_array_::create(repeated, 10000 * 256, 256);
    delete[] repeated;

    repeated = repeat(n2, 10000);
    array_n2 = (void *)fixnum_array_::create(repeated, 10000 * 256, 256);
    delete[] repeated;

}

PublicKey::~PublicKey()
{
    fixnum_array_ *ptr = (fixnum_array_ *)array_n;
    delete ptr;
}

uint8_t *PublicKey::repeat(uint8_t *input, int nelts)
{
    uint8_t *repeated = new uint8_t[nelts * 256];
    for (int i = 0; i < nelts; i++)
        memcpy(repeated + i * 256, input, 256);

    return repeated;
}

void *PublicKey::getkeys_array_n(int nelts)
{
    fixnum_array_ *array_n_ptr = (fixnum_array_ *)array_n;

    if (array_n_ptr->length() < nelts)
    {
        delete array_n_ptr;
        uint8_t *repeated = repeat(n, nelts);
        array_n = (void *)fixnum_array_::create(repeated, nelts * 256, 256);

        delete[] repeated;
    }

    return array_n;
}

void *PublicKey::getkeys_array_n2(int nelts)
{
    fixnum_array_ *array_n2_ptr = (fixnum_array_ *)array_n2;

    if (array_n2_ptr->length() < nelts)
    {
        delete array_n2_ptr;
        uint8_t *repeated = repeat(n2, nelts);
        array_n2 = (void *)fixnum_array_::create(repeated, nelts * 256, 256);

        delete[] repeated;
    }

    return array_n2;
}

uint8_t *PublicKey::encrypt(uint8_t *m, uint8_t *r, int nelts, int element_byte_len)
{
    fixnum_array_ *array_m = fixnum_array_::create(m, nelts * element_byte_len, element_byte_len);
    fixnum_array_ *array_r = fixnum_array_::create(r, nelts * element_byte_len, element_byte_len);
    fixnum_array_ *array_n_ptr = (fixnum_array_ *)getkeys_array_n(nelts);

    fixnum_array_ *ctx = fixnum_array_::create(nelts);
    fixnum_array_::template map<paillier_encrypt_func>(ctx, array_n_ptr, array_m, array_r);
    
    uint8_t *ctx_ptr = new uint8_t[nelts * 256];
    int size = nelts;
    ctx->retrieve_all(ctx_ptr, nelts * 256, &size);
    
    delete array_m;
    delete array_r;
 
    return ctx_ptr;
}

PrivateKey::PrivateKey(char *key_file)
{
    FILE *file = fopen(key_file, "rb");
    fread(n, sizeof(uint8_t), 256, file);
    fread(n2, sizeof(uint8_t), 256, file);
    fread(p, sizeof(uint8_t), 256, file);
    fread(q, sizeof(uint8_t), 256, file);
    fread(lamda, sizeof(uint8_t), 256, file);
    fread(lg_inv, sizeof(uint8_t), 256, file);
    fclose(file);
    
    //init  with 10000 element
    uint8_t *repeated = repeat(n, 10000);
    array_n = (void *)fixnum_array_::create(repeated, 10000 * 256, 256);
    delete[] repeated;

    repeated = repeat(n2, 10000);
    array_n2 = (void *)fixnum_array_::create(repeated, 10000 * 256, 256);
    delete[] repeated;

    repeated = repeat(p, 10000);
    array_p = (void *)fixnum_array_::create(repeated, 10000 * 256, 256);
    delete[] repeated;

    repeated = repeat(q, 10000);
    array_q = (void *)fixnum_array_::create(repeated, 10000 * 256, 256);
    delete[] repeated;

    repeated = repeat(lamda, 10000);
    array_lamda = (void *)fixnum_array_::create(repeated, 10000 * 256, 256);
    delete[] repeated;

    repeated = repeat(lg_inv, 10000);
    array_lg_inv = (void *)fixnum_array_::create(repeated, 10000 * 256, 256);
    delete[] repeated;
}

PrivateKey::~PrivateKey()
{
    fixnum_array_ *ptr = (fixnum_array_ *)array_n;
    delete ptr;
    ptr = (fixnum_array_ *)array_n2;
    delete ptr;
    ptr = (fixnum_array_ *)array_p;
    delete ptr;
    ptr = (fixnum_array_ *)array_q;
    delete ptr;
    ptr = (fixnum_array_ *)array_lamda;
    delete ptr;
    ptr = (fixnum_array_ *)array_lg_inv;
    delete ptr;
}

uint8_t *PrivateKey::repeat(uint8_t *input, int nelts)
{
    uint8_t *repeated = new uint8_t[nelts * 256];
    for (int i = 0; i < nelts; i++)
        memcpy(repeated + i * 256, input, 256);

    return repeated;
}

void *PrivateKey::getkeys_array_n(int nelts)
{
    fixnum_array_ *array_n_ptr = (fixnum_array_ *)array_n;

    if (array_n_ptr->length() < nelts)
    {
        delete array_n_ptr;
        uint8_t *repeated = repeat(n, nelts);
        array_n = (void *)fixnum_array_::create(repeated, nelts * 256, 256);

        delete[] repeated;
    }

    return array_n;
}

void *PrivateKey::getkeys_array_n2(int nelts)
{
    fixnum_array_ *array_n2_ptr = (fixnum_array_ *)array_n2;

    if (array_n2_ptr->length() < nelts)
    {
        delete array_n2_ptr;
        uint8_t *repeated = repeat(n2, nelts);
        array_n2 = (void *)fixnum_array_::create(repeated, nelts * 256, 256);

        delete[] repeated;
    }

    return array_n2;
}

void *PrivateKey::getkeys_array_lamda(int nelts)
{
    fixnum_array_ *array_lamda_ptr = (fixnum_array_ *)array_lamda;

    if (array_lamda_ptr->length() < nelts)
    {
        delete array_lamda_ptr;
        uint8_t *repeated = repeat(lamda, nelts);
        array_lamda = (void *)fixnum_array_::create(repeated, nelts * 256, 256);

        delete[] repeated;
    }

    return array_lamda;
}

void *PrivateKey::getkeys_array_lg_nv(int nelts)
{
    fixnum_array_ *array_lg_nv_ptr = (fixnum_array_ *)array_lg_inv;

    if (array_lg_nv_ptr->length() < nelts)
    {
        delete array_lg_nv_ptr;
        uint8_t *repeated = repeat(lg_inv, nelts);
        array_lg_inv = (void *)fixnum_array_::create(repeated, nelts * 256, 256);

        delete[] repeated;
    }

    return array_lg_inv;
}

uint8_t *PrivateKey::decrypt(uint8_t *ctx, int nelts)
{
    fixnum_array_ *array_ctx = fixnum_array_::create(ctx, nelts * 256, 256);
    fixnum_array_ *array_ptx = fixnum_array_::create(nelts);
    fixnum_array_ *array_lamda_ptr = (fixnum_array_ *)getkeys_array_lamda(nelts);
    fixnum_array_ *array_n_ptr = (fixnum_array_ *)getkeys_array_n(nelts);
    fixnum_array_ *array_n2_ptr = (fixnum_array_ *)getkeys_array_n2(nelts);
    fixnum_array_ *array_lg_inv_ptr = (fixnum_array_ *)getkeys_array_lg_nv(nelts);

    fixnum_array_::template map<paillier_decrypt_func>(array_ptx, array_ctx, array_lamda_ptr, array_n_ptr,
                                                       array_n2_ptr, array_lg_inv_ptr);

    uint8_t *ptx_ptr = new uint8_t[nelts * 256];
    int size = nelts;
    array_ptx->retrieve_all(ptx_ptr, nelts * 256, &size);

    delete array_ctx;
    delete array_ptx;

    return ptx_ptr;
}

//efficent
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
template <typename fixnum>
using modexp_cios = my_modexp<modnum_monty_cios<fixnum>>;


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
template <typename fixnum>
using multi_modexp_cios = my_multi_modexp<modnum_monty_cios<fixnum>>;

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
template <typename fixnum>
using multi_mod = my_mult_mod<modnum_monty_redc<fixnum>>;

/*
uint8_t * dot_he(uint8_t *ctx, uint8_t *ptx, uint8_t *r, uint8_t *r_mask, int batch_size, int input_size, PublicKey *pk)
{
    int bytelen = 256;
    uint8_t *encrypted_r = pk->encrypt(r, r_mask, batch_size, 8);

    fixnum_array_ *array_ctx = fixnum_array_::create(ctx, batch_size * input_size * bytelen, bytelen);
    fixnum_array_ *array_ptx = fixnum_array_::create(ptx, batch_size * input_size * 8, 8);
    fixnum_array_ *array_muled = fixnum_array_::create(batch_size * input_size);
    fixnum_array_ *array_n2 = (fixnum_array_ *)pk->getkeys_array_n2(batch_size * input_size);
    
    //mul
    fixnum_array_::template map<modexp_cios>(array_muled, array_ctx, array_ptx, array_n2);

    uint8_t *muled_ptr = new uint8_t[batch_size * input_size * bytelen];
    int size = batch_size * input_size * bytelen;
    array_muled->retrieve_all(muled_ptr, batch_size * input_size * bytelen, &size);
    
    //add
    fixnum_array_ *array_res = fixnum_array_::create(encrypted_r, batch_size * bytelen, bytelen);
    for (int i = 0; i < input_size; i++)
    {
        fixnum_array_ *array_to_add = fixnum_array_::create(muled_ptr + i * batch_size * bytelen, batch_size * bytelen, bytelen);
        fixnum_array_ *array_added = fixnum_array_::create(batch_size);
        fixnum_array_::template map<multi_mod>(array_added, array_res, array_to_add, array_n2);

        delete array_res;
        delete array_to_add;
        array_res = array_added;
    }

    uint8_t *res_ptr = new uint8_t[batch_size * bytelen];
    size = batch_size;
    array_res->retrieve_all(res_ptr, batch_size * bytelen, &size);

    return res_ptr;
}
*/

uint8_t *sum_func(uint8_t *left, uint8_t *right, fixnum_array_ *array_n2, int nelts)
{
    fixnum_array_ *array_left = fixnum_array_::create(left , nelts * 256, 256);
    fixnum_array_ *array_right = fixnum_array_::create(right , nelts * 256, 256);
    fixnum_array_ *array_res = fixnum_array_::create(nelts);

    fixnum_array_::template map<multi_mod>(array_res, array_left, array_right, array_n2);

    uint8_t *res_ptr = new uint8_t[nelts * 256];
    int size = nelts;
    array_res->retrieve_all(res_ptr, nelts * 256, &size);
    
    delete array_left;
    delete array_right;
    delete array_res;
    return res_ptr;
}

uint8_t *sum(uint8_t *data, fixnum_array_ *array_n2, int nelts, int batch_len)
{
    if (nelts <= 1)
        return data;
    
    if (nelts % 2 == 0)
    {
        nelts = nelts / 2;
        uint8_t *res = sum_func(data, data + nelts * batch_len * 256, array_n2, nelts * batch_len);
        delete[] data;
        return sum(res, array_n2, nelts, batch_len);
    }
    else
    {
        uint8_t *remain = data + (nelts - 1) * batch_len * 256;
        remain = sum_func(remain, data, array_n2, batch_len);
        memcpy(data, remain, batch_len * 256);
        delete[] remain;
        
        nelts = nelts / 2;
        uint8_t *res = sum_func(data, data + nelts * batch_len * 256, array_n2, nelts * batch_len);
        delete[] data;
        return sum(res, array_n2, nelts, batch_len);
    }
    
}

uint8_t * dot_he(uint8_t *ctx, uint8_t *ptx, uint8_t *r, uint8_t *r_mask, int batch_size, int input_size, PublicKey *pk)
{
    int bytelen = 256;
    uint8_t *encrypted_r = pk->encrypt(r, r_mask, batch_size, 8);

    fixnum_array_ *array_ctx = fixnum_array_::create(ctx, batch_size * input_size * bytelen, bytelen);
    fixnum_array_ *array_ptx = fixnum_array_::create(ptx, batch_size * input_size * 8, 8);
    fixnum_array_ *array_muled = fixnum_array_::create(batch_size * input_size);
    fixnum_array_ *array_n2 = (fixnum_array_ *)pk->getkeys_array_n2(batch_size * input_size);
    
    //mul
    fixnum_array_::template map<modexp_cios>(array_muled, array_ctx, array_ptx, array_n2);

    uint8_t *muled_ptr = new uint8_t[batch_size * input_size * bytelen];
    int size = batch_size * input_size * bytelen;
    array_muled->retrieve_all(muled_ptr, batch_size * input_size * bytelen, &size);
    
    //add
    uint8_t *sumed = sum(muled_ptr, array_n2, input_size, batch_size);
    fixnum_array_ *array_r = fixnum_array_::create(encrypted_r, batch_size * bytelen, bytelen);
    fixnum_array_ *array_sumed = fixnum_array_::create(sumed, batch_size * bytelen, bytelen);
    fixnum_array_ *array_res = fixnum_array_::create(batch_size);
    fixnum_array_::template map<multi_mod>(array_res, array_r, array_sumed, array_n2);
    uint8_t *res_ptr = new uint8_t[batch_size * 256];
    size = batch_size;
    array_res->retrieve_all(res_ptr, batch_size * 256, &size);

    delete array_ctx;
    delete array_ptx;
    delete array_muled;
    delete array_r;
    delete array_sumed;
    delete array_res;
    delete encrypted_r;

    return res_ptr;
}