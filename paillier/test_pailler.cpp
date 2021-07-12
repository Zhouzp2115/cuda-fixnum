#include "paillier.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "emp-tool/emp-tool.h"

using namespace std;
using namespace emp;

uint8_t *long2bytes(long *input, int length, int fn_bytes)
{
    uint8_t *bytes = new uint8_t[length * fn_bytes];
    memset(bytes, 0x00, length * fn_bytes);
    for (int i = 0; i < length; i++)
    {
        long input_i = input[i];
        for (int j = 0; j < sizeof(long); j++)
        {
            bytes[i * fn_bytes + j] = input_i & 0xff;
            input_i = input_i >> 8;
        }
    }

    return bytes;
}

long *bytes2long(uint8_t *bytes, int length, int fn_bytes = 256)
{
    int nelts = length / fn_bytes;
    long *res = new long[nelts];
    memset(res, 0x00, nelts * sizeof(long));
    for (int i = 0; i < nelts; i++)
    {
        for (int j = 0; j < sizeof(long); j++)
        {
            uint64_t byte_long = (uint64_t)bytes[j];
            byte_long = byte_long << (j * 8);
            res[i] = res[i] | byte_long;
        }

        bytes += fn_bytes;
    }

    return res;
}

void printlong(long *data, int length)
{
    for (int i = 0; i < length; i++)
        printf("%d ", data[i]);
    printf("\n");
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

void test_init()
{
    PublicKey *pk = new PublicKey("./publickey.byte");
    PrivateKey *sk = new PrivateKey("./privatekey.byte");

    uint8_t m[8] = {0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x14, 0x16};
    uint8_t *ctx = pk->encrypt(m, m, 1, 8);
    uint8_t *ptx = sk->decrypt(ctx, 1);

    for (int i = 0; i < 10; i++)
        printf("%x ", ptx[i]);
    printf("\n");

    delete[] ctx;
    delete[] ptx;
    delete pk;
    delete sk;
}


void test_func(int device)
{
    paillierSetGPUDevice(device);

    int batch_size = 128, input_size = 1000;
    PublicKey *pk = new PublicKey("./publickey.byte");

    uint8_t *x_uint8 = new uint8_t[batch_size * input_size * 8];
    uint8_t *r_uint8 = new uint8_t[batch_size * input_size * 8];
    uint8_t *w_uint8 = new uint8_t[batch_size * input_size * 8];
    for (int i = 0; i < batch_size * input_size * 8; i++)
    {
        x_uint8[i] = rand() % 256;
        r_uint8[i] = rand() % 256;
        w_uint8[i] = rand() % 256;
    }

    uint8_t *encrypted_w = pk->encrypt(w_uint8, w_uint8, batch_size * input_size, sizeof(long));
    delete[] w_uint8;

    cout << "data init ok" << endl;

    for (int i = 0; i < 3; i++)
    {
        auto satrt = clock_start();
        dot_he(encrypted_w, x_uint8, r_uint8, r_uint8, input_size, batch_size, pk);
        cout << "dot_he time cost " << time_from(satrt) / 1000000 << endl;

        satrt = clock_start();
        dot_he(encrypted_w, x_uint8, r_uint8, r_uint8, input_size, batch_size, pk);
        cout << "exp + sum time cost " << time_from(satrt) / 1000000 << endl;
    }
}

int main(int argc, char *argv[])
{
    //test_func(atoi(argv[1]));

    paillierSetGPUDevice(0);
    
    PublicKey *pk = new PublicKey("./publickey.byte");
    PrivateKey *sk = new PrivateKey("./privatekey.byte");

    long w[4] = {112, 142, 115, 369};
    uint8_t *w_uint8 = long2bytes(w, 4, sizeof(long));

    clock_t t = clock();
    uint8_t *encrypted_w = pk->encrypt(w_uint8, w_uint8, 4, sizeof(long));
    delete[] w_uint8;

    t = clock() - t;
    printf("encrypt time cost:%f \n", t * 1.0 / CLOCKS_PER_SEC);

    //test dot_product
    int batch_size = 2, inputsize = 4;
    long x[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    long r[2] = {5, 6};
    uint8_t *x_uint8 = long2bytes(x, batch_size * inputsize, sizeof(long));
    uint8_t *r_uint8 = long2bytes(r, batch_size, sizeof(long));

    t = clock();
    //repeat ctx
    //n->n*m every element repeat m times
    uint8_t *encrypted_w_repeated = new uint8_t[batch_size * inputsize * 256];
    for (int i = 0; i < inputsize;i++)
    {
        for (int j = 0; j < batch_size; j++)
            memcpy(encrypted_w_repeated + (i * batch_size + j) * 256, encrypted_w + i * 256, 256);
    }
    delete[] encrypted_w;
    
    printf("encrypted w  repeated\n");
    uint8_t *decrypted_w = sk->decrypt(encrypted_w_repeated, batch_size * inputsize);
    printlong(bytes2long(decrypted_w, batch_size * inputsize * 256, 256), batch_size * inputsize);

    uint8_t *dot_res = dot_he(encrypted_w_repeated, x_uint8, r_uint8, r_uint8, batch_size, inputsize, pk);
    t = clock() - t;
    printf("dot_product time cost:%f \n", t * 1.0 / CLOCKS_PER_SEC);

    t = clock();
    uint8_t *ptx = sk->decrypt(dot_res, batch_size);
    t = clock() - t;
    printf("decrypt time cost:%f \n", t * 1.0 / CLOCKS_PER_SEC);
    
    printf("dot res \n");
    printlong(bytes2long(ptx, batch_size * 256, 256), 2);

    return 0;
}