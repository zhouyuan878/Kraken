
#include "kraken_headers.hpp"
#include "krakendb.hpp"
#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>

using namespace std;
using namespace kraken;


typedef struct {
    string id;
    string seq;
    string quals;
} DNASequence;


__device__ __host__
uint64_t getRightMostDigits(uint64_t inval, int num_digit) {
    uint64_t temp = inval << (64 - num_digit);
    return temp;
}


__device__ __host__
uint64_t getLeftMostDigits(uint64_t inval, int num_digit) {
    uint64_t temp = inval >> (64 - num_digit);
    return temp;
}


__device__ __host__
uint64_t getKmerDigit(uint64_t* inputDNA, int kmer_index) {
    if (kmer_index % 32 == 0) {
        return inputDNA[kmer_index / 32];
    }
    else {
        int num_leftDigits = 2 * (32 - kmer_index % 32);
        int num_rightDigits = 64 - num_leftDigits;
        uint64_t firstVal = inputDNA[kmer_index / 32];
        uint64_t secondVal = inputDNA[kmer_index / 32 + 1];
        uint64_t leftDigits = getRightMostDigits(firstVal, num_leftDigits);
        uint64_t rightDigits = getLeftMostDigits(secondVal, num_rightDigits);
        uint64_t result = leftDigits | rightDigits;
        return result;
    }
}




void die(char *message) {
    printf("%s\n", message);
    exit(1);
}

vector<uint64_t> read_Sequence(string input) {
    int kmer_num = input.length() / 32;
    int kmer_index = 0;
    vector<uint64_t> result;
    
    while (kmer_index < kmer_num) {
        uint64_t kmer = 0;
        for (int i = 32 * kmer_index; i < 32 * (kmer_index + 1); i++) {
            char current = input.at(i);
            kmer <<= 2;
            switch (current) {
                case 'A': case 'a':
                    break;
                case 'C': case 'c':
                    kmer |= 1;
                    break;
                case 'G': case 'g':
                    kmer |= 2;
                    break;
                case 'T': case 't':
                    kmer |= 3;
                    break;
                default:
                    break;
            }
        }
        result.push_back(kmer);
        kmer_index++;
    }
    return result;
}




__global__ void minimizer_kernel(uint64_t *seq_GPU, uint64_t *binkey_GPU) {
    
    int block_id = blockIdx.x + gridDim.x * blockIdx.y;
    int thread_id = blockDim.x * block_id + threadIdx.x;
    
    KrakenDB temp;
    
    if (threadIdx.x < 32) {
        binkey_GPU[thread_id] = temp.bin_key(getKmerDigit(seq_GPU, threadIdx.x));
    }
}


int main(int argc, char **argv) {
    int seq_size = 4 * sizeof(uint64_t);
    int binkey_size = 4 * sizeof(uint64_t);
    
    uint64_t *binkey_CPU;
    uint64_t seq_CPU[2];
    seq_CPU[0] = 10;
    seq_CPU[1] = 200011;
    uint64_t *binkey_GPU;
    uint64_t *seq_GPU;
    
    
    if (cudaMalloc((void **) &binkey_GPU, seq_size) != cudaSuccess) die("Error allocating GPU memory");
    if (cudaMalloc((void **) &seq_GPU, binkey_size) != cudaSuccess) die("Error allocating GPU memory");
    
    
    cudaMemcpy(binkey_GPU, binkey_CPU, binkey_size, cudaMemcpyHostToDevice);
    cudaMemcpy(seq_GPU, seq_CPU, seq_size, cudaMemcpyHostToDevice);
    
    minimizer_kernel <<< 1 , 32 >>> (seq_GPU, binkey_GPU);
    
    cudaMemcpy(binkey_CPU, binkey_GPU, binkey_size, cudaMemcpyDeviceToHost);
    cout<< binkey_CPU[0] << endl;
    
    return binkey_CPU[0];
    cudaFree(binkey_GPU);
    cudaFree(seq_GPU);
    
    
    
}




