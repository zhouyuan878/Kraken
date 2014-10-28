

#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
#include <stdint.h>

using namespace std;

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


int main( int argc, char* argv[] )
{
    vector<uint64_t> test;
    test = read_Sequence("aaaaaaaccccccccccccgggggggttttttaaaaaaaccccccccccccgggggggtttttt");
    cout<< test[0] << endl;
    cout<< test[1] << endl;
}


