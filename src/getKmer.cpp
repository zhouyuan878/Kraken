
#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>


using namespace std;

typedef struct {
    std::string id;
    std::string seq;
    std::string quals;
} DNASequence;


uint64_t getKmer(DNASequence DNA_Input, int kmer_index) {
    if (kmer_index + 31 > DNA_Input.seq.length()) {
        exit(0);
    }
    
    uint64_t kmer = 0;
    string input = DNA_Input.seq;
    
    for (int i = kmer_index; i < kmer_index + 32; i++) {
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
    return kmer;
}


int main( int argc, char* argv[] )
{
    DNASequence test;
    test.seq = "aaaaaaaccccccccccccgggggggttttttttttttt";
    uint64_t temp = getKmer(test,2);
    cout<< temp << endl;
}


