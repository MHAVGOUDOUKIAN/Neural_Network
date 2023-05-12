#ifndef RANDOM_H_INCLUDED
#define RANDOM_H_INCLUDED

#include <math.h>

inline float randomf(float Min, float Max) // Renvoie un nombre entier dans l'intervalle [Min;Max]
{
    float offset=0;
    if(abs(Min) > 0) { offset = Min; }
    float res = float(rand())/float(RAND_MAX) * (Max+1 + Min*(-1)) ;
    return res + offset;
}

inline int randomi(int Min, int Max) // Renvoie un nombre entier dans l'intervalle [Min;Max]
{
    int offset=0;
    if(abs(Min) > 0) { offset = Min; }
    int res = float(rand())/float(RAND_MAX) * (Max+1 + Min*(-1));
    return res + offset;
}

#endif // RANDOM_H_INCLUDED
