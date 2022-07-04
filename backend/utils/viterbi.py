# -*- coding: utf-8 -*-
# @Time  : 2022/5/27 16:31
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : viterbi.py
import numpy as np


def viterbi(transProbs, emissionProbs, States, startProbs, observation):
    '''
    Input :
        - transProbs    :   matrice di probabilità delle transizioni
        - emissionProbs :   matrice di probabilità delle emissioni
        - States        :   insieme degli hidden states
        - startProbs    :   probabilità di iniziare in quel preciso stato
        - observation   :   le osservazioni effettuate
    Matrici/DataFrame :
        - v             :   conserva le probabilità del cammino più probabile
        - p             :   conserva gli stati del cammino più probabile

    Output :
        - viterbiPath   :   la sequenza di stati che massimizza la probabilità di avere le osservazioni date in input
    '''

    # I valori NaN (Not a Number) vengono sostituiti da uno 0
    transProbs = transProbs.fillna(0)
    emissionProbs = emissionProbs.fillna(0)

    nObservations = len(observation)
    nStates = len(States)

    # Creazione dei DataFrame, v viene riempito di NaN, p da -1
    v = pd.DataFrame(np.nan, columns=np.arange(nObservations), index=States)
    p = pd.DataFrame(-1, columns=np.arange(nObservations), index=States)

    # Prima passo della sequenza
    v[0] = np.log(startProbs[States] * emissionProbs[observation[0]]).T
    p[0] = -1

    # Riempio le matrici
    for k in range(1, nObservations):
        for state in States:
            temp = v[k - 1] + np.log(transProbs[state])
            maxi = np.argmax(temp)
            v[k][state] = np.log(emissionProbs[observation[k]][state]) + temp[maxi]
            p[k][state] = maxi
        if np.isinf(v[k]).sum() == nStates:
            # caso di transizione non riconosciuta nel training set
            # ricomputo senza considerare la transizione
            print("impossible to find an optimal transition for the observation: " +
                  str(observation[k - 1]) + "-->" + str(observation[k]) + "\n")
            temp = v[k - 1]
            maxi = np.argmax(temp)
            for state in States:
                v[k][state] = np.log(emissionProbs[observation[k]][state]) + temp[maxi]
                p[k][state] = maxi

    # Vado all'indietro per capire qual è il path più probabile
    viterbiPath = np.full(nObservations, "nan", dtype=np.dtype("U25"))
    statek = np.argmax(v[nObservations - 1])
    viterbiPath[nObservations - 1] = States[statek]
    for k in range(nObservations - 1, 0, -1):
        statek = p[k][States[statek]]
        viterbiPath[k - 1] = States[statek]
    return viterbiPath