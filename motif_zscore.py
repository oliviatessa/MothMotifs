# %%
import os 
import pickle 
import numpy as np
import math

import pandas as pd 

# %% [markdown]
# # Load the pruned networks 

# %%
f = 'pruned_bias'
timestamp = '2022_11_05__01_04_10'

#Load the networks
sparseNetsFile = 'sparseNetworks.pkl'
sparseNetworks = pickle.load(open(os.path.join(f, timestamp, sparseNetsFile), 'rb'))

#Load the losses
losses = pickle.load(open(os.path.join(f, timestamp, 'preprocessedNets', 'pruneLosses.pkl'),'rb'))

losses = np.array(losses)
losses = np.transpose(losses)

lossesDF = pd.DataFrame(losses, columns=['0%', '15%', '25%', '35%', '45%', '55%', '65%', '75%', '85%', '90%', '91%', 
                                        '92%', '93%', '94%', '95%', '96%', '97%', '98%' ])

# %% [markdown]
# # Find the masks

# %%
masks = []
for i in range(len(sparseNetworks)):
    net = sparseNetworks[i]
    sparsity = net[0]
    net = net[1]
    maskNet = []
    for i in np.arange(0,len(net),2):
        m = np.abs(net[i]) * np.reciprocal(np.abs(net[i]), where = np.abs(net[i])!=0)
        m = np.round(m)
        mask = m.astype(int)
        
        #Add bias term
        biasRow = np.abs(net[i+1]) * np.reciprocal(np.abs(net[i+1]), where = np.abs(net[i+1])!=0)
        mask = np.vstack([mask, biasRow])
        
        maskNet.append(mask)
        
    
    masks.append((sparsity, maskNet))

# %% [markdown]
# # Find the motif z-score

# %% [markdown]
# ## Motif counting functions

# %% [markdown]
# ### First-order motifs

# %%
def fom(m):
        '''
        Calculates the number of first-order motifs in the network (equivalent to the number of edges).
        
        Input(s): the mask of the pruned network, as a list of matrices
        Returns: FOM (total number of first-order motifs), FOMList (Number of weights and number of bias connections
                in each layer)
        '''
        FOM = 0
        FOMList = [[0,0],[0,0],[0,0],[0,0],[0,0]] #[[Num weights, num biases]] in each layer

        for i in range(len(m)): 
                #Count number of connections between weights
                w_connections = np.count_nonzero(m[i][0:-1])
                FOMList[i][0] = w_connections
                #Count number of connections from bias
                b_connections = np.count_nonzero(m[i][-1])
                FOMList[i][1] = b_connections

                connections = w_connections + b_connections
                FOM += connections
        return FOM, FOMList

# %% [markdown]
# ### Second-order motifs

# %% [markdown]
# #### Diverging

# %%
def sodm(m):
    '''
    Calculates the number of second-order diverging motifs in the network.
    Also calculates the remaining number of nodes in the network. 
        
    Input(s): the mask of the pruned network, as a list of matrices
    Returns: SODM (total number of second-order diverging motifs in the network), numFC (Number of remaining nodes 
        with downstream output)
    '''
    
    SODM = 0
    numFC = [0,0,0,0,0,0] #Number of remaining nodes with downstream output

    for i in range(len(m)): 
        nodes = 0
        #Calculate second-order diverging motifs
        for row in m[i]:
            n = np.count_nonzero(row)
            if n >= 2:
                SODM += math.factorial(n)/(math.factorial(2)*math.factorial(n-2))
                    
            #also calculate number of remaining nodes
            if n > 0: 
                nodes += 1
                    
        numFC[i] = nodes
            
            #Count number of nodes in final layer 
        if i == 4: 
            nodes = 0 
            for row in m[i].T:
                n = np.count_nonzero(row)
                if n > 0 :
                    nodes += 1
            numFC[i+1] = nodes
    
    return SODM, numFC

# %% [markdown]
# #### Converging

# %%
def socm(m):
    '''
    Calculates the number of second-order converging motifs in the network.
        
    Input(s): the mask of the pruned network, as a list of matrices
    Returns: SOCM (total number of second-order converging motifs in the network)
    '''

    SOCM = 0

    for i in range(len(m)): 
        #Calculate second-order converging motifs
        for column in m[i].T:
            n = np.count_nonzero(column)
            if n >= 2:
                SOCM += math.factorial(n)/(math.factorial(2)*math.factorial(n-2))

    return SOCM

# %% [markdown]
# #### Chain

# %%
def sochain(m):
    '''
    Calculates the number of second-order chain motifs in the network.
        
    Input(s): the mask of the pruned network, as a list of matrices
    Returns: SOChain (total number of second-order chain motifs in the network)
    ''' 
    
    SOChain = 0 
    for i in range(len(m)): 
        #Calculate second-order chain motifs
        if i != 4: 
            #Exclude the bias by excluding the last row
            SOChain += np.count_nonzero(np.matmul(m[i][0:-1],m[i+1][0:-1]))
            
            #Add in the motifs from the bias terms 
            SOChain += np.count_nonzero(np.matmul(m[i][-1],m[i+1][0:-1]))
        else: 
            pass

    return SOChain

# %% [markdown]
# ### Third-order motifs

# %% [markdown]
# #### Diverging

# %%
def todm(m):
    '''
    Calculates the number of third-order diverging motifs in the network.
        
    Input(s): the mask of the pruned network, as a list of matrices
    Returns: TODM (total number of third-order diverging motifs in the network)
    '''

    TODM = 0 
    for i in range(len(m)): 
        #Calculate third-order diverging motifs
        for row in m[i]:
            n = np.count_nonzero(row)
            if n >= 3:
                TODM += math.factorial(n)/(math.factorial(3)*math.factorial(n-3))

    return TODM

# %% [markdown]
# #### Converging

# %%
def tocm(m):
    '''
    Calculates the number of third-order converging motifs in the network.
        
    Input(s): the mask of the pruned network, as a list of matrices
    Returns: TOCM (total number of third-order converging motifs in the network)
    '''

    TOCM = 0 
    for i in range(len(m)): 
        #Calculate third-order converging motifs 
        for column in m[i].T:
            n = np.count_nonzero(column)
            if n >= 3:
                TOCM += math.factorial(n)/(math.factorial(2)*math.factorial(n-2))

    return TOCM

# %% [markdown]
# #### Chain

# %%
def tochain(m):
    '''
    Calculates the number of third-order chain motifs in the network.
        
    Input(s): the mask of the pruned network, as a list of matrices
    Returns: TOChain (total number of third-order chain motifs in the network)
    '''

    TOChain = 0
    for i in range(len(m)): 
        #Calculate third-order chain motifs 
        if i in (0,1,2): 
            #Count non-zero elements of (Layer 1 * Layer 2 * Layer 3)
            #Exclude the bias by excluding the last row
            m1 = np.matmul(m[i][0:-1],m[i+1][0:-1])
            TOChain += np.count_nonzero(np.matmul(m1,m[i+2][0:-1]))
                
            #Add in the motifs from the bias terms 
            mbias = np.matmul(m[i][-1],m[i+1][0:-1])
            TOChain += np.count_nonzero(np.matmul(mbias,m[i+2][0:-1]))
        else: 
            pass

    return TOChain

# %% [markdown]
# ### Random networks

# %% [markdown]
# #### Build random network

# %%
def buildRandomNet(numFC, FOMList):
    '''
    Builds randomly connected network with the same number of weights and bias connections as the real network. 

    Input(s): numFC (Number of remaining nodes with downstream output), FOMList (Number of weights and number of 
        bias connections in each layer)
    Returns: the mask of the random network, as a list of matrices
    '''
    #random weight matrix = np.array(([1]*num connections between weights)+
    #                       [0]*(num possible connections - num connections between weights))
    #           numFC[0]-1 because we need to discount bias 
    r1 = np.array([1] * (FOMList[0][0]) + [0] * (((numFC[0]-1)*(numFC[1]-1))-(FOMList[0][0])))
    #random bias matrix = np.array(([1]*num connections between bias and next nodes)+
    #                       [0]*(num possible connections - num connections between bias and next nodes))

    #There is always a live bias, so number of possible connections between bias and next
    #   layer would be numFC[i+1]-1 (to remove bias).
    r1b = np.array([1] * FOMList[0][1] + [0] * ((numFC[1]-1)-FOMList[0][1]))

    r2 = np.array([1] * (FOMList[1][0]) + [0] * (((numFC[1]-1)*(numFC[2]-1))-(FOMList[1][0])))
    r3 = np.array([1] * (FOMList[2][0]) + [0] * (((numFC[2]-1)*(numFC[3]-1))-(FOMList[2][0])))
    r4 = np.array([1] * (FOMList[3][0]) + [0] * (((numFC[3]-1)*(numFC[4]-1))-(FOMList[3][0])))
    r5 = np.array([1] * (FOMList[4][0]) + [0] * (((numFC[4]-1)*(numFC[5]))-(FOMList[4][0]))) #no bias in last layer

    r2b = np.array([1] * FOMList[1][1] + [0] * ((numFC[2]-1)-FOMList[1][1]))
    r3b = np.array([1] * FOMList[2][1] + [0] * ((numFC[3]-1)-FOMList[2][1]))
    r4b = np.array([1] * FOMList[3][1] + [0] * ((numFC[4]-1)-FOMList[3][1]))
    r5b = np.array([1] * FOMList[4][1] + [0] * ((numFC[5])-FOMList[4][1]))
        
    np.random.shuffle(r1)
    np.random.shuffle(r2)
    np.random.shuffle(r3)
    np.random.shuffle(r4)
    np.random.shuffle(r5)

    np.random.shuffle(r1b)
    np.random.shuffle(r2b)
    np.random.shuffle(r3b)
    np.random.shuffle(r4b)
    np.random.shuffle(r5b)
            
    randomNet = [np.vstack([np.reshape(r1, (((numFC[0]-1),(numFC[1]-1)))),r1b]),
                np.vstack([np.reshape(r2, (((numFC[1]-1),(numFC[2]-1)))),r2b]),
                np.vstack([np.reshape(r3, (((numFC[2]-1),(numFC[3]-1)))),r3b]),
                np.vstack([np.reshape(r4, (((numFC[3]-1),(numFC[4]-1)))),r4b]),
                np.vstack([np.reshape(r5, (((numFC[4]-1),(numFC[5])))),r5b])]

    return randomNet

# %% [markdown]
# #### Find motifs for random networks 

# %%
def randomNetMotifs(randomNet):
    '''
    Finds all of the motifs for the random network.

    Input(s): the mask of the random network, as a list of matrices
    Returns: rFOM (random first-order motifs), rFOMList (remaining connections in each layer), rSODM (random second-oder 
        diverging motifs), rSOCM (random second-order converging motifs), rSOChain (random second-order chain motifs), 
        rTODM (random third-order diverging motifs), rTOCM (random third-order converging motifs), rTOChain (random 
        third-order chain motifs)
    '''

    rFOM, rFOMList = fom(randomNet)
    rSODM, rnumFC = sodm(randomNet)
    rSOCM = socm(randomNet)
    rSOChain = sochain(randomNet)
    rTODM = todm(randomNet)
    rTOCM = tocm(randomNet)
    rTOChain = tochain(randomNet)
    
    return rFOM, rFOMList, rSODM, rSOCM, rSOChain, rTODM, rTOCM, rTOChain

# %% [markdown]
# #### Average random motifs

# %%
def buildRandomMotifsDF(numFC, FOMList, numRand=1000):
    '''
    Builds dataframe of numRand number of random network motif counts. 
    To calculate the z-score, we need to compare the real network to many randomly generated networks. This function 
        adds all of that information to a dataframe. 

    Input(s): numFC (Number of remaining nodes with downstream output), FOMList (Number of weights and number of 
        bias connections in each layer), numRand=1000 (number of random networks we want to generate, default 1000)
    Returns: random network motif dataframe
    '''
    randomNetDF = pd.DataFrame(columns=['rSODM',
                                        'rSOCM',
                                        'rSOChain',
                                        'rTODM',
                                        'rTOCM', 
                                        'rTOChain'])

    for r in range(numRand):
        randomNet = buildRandomNet(numFC, FOMList)
        rFOM, rFOMList, rSODM, rSOCM, rSOChain, rTODM, rTOCM, rTOChain = randomNetMotifs(randomNet)

        rMotifs = [float(rSODM), float(rSOCM), float(rSOChain), float(rTODM), float(rTOCM), float(rTOChain)]
        randomNetDF.loc[len(randomNetDF.index)] = rMotifs

    return randomNetDF

# %% [markdown]
# ## Z-score dataframe

# %%
masksTest = masks[0:5]

# %%
zscoreDF = pd.DataFrame(columns=['Sparsity Index', 'Masks',
                                '1-O motifs (real)', 
                                'S-O diverging motifs (real)', 'S-O converging motifs (real)', 
                                'S-O chain motifs (real)',  'T-O chain motifs (real)',
                                'T-O diverging motifs (real)', 'T-O converging motifs (real)',
                                
                                'Avg - S-O diverging motifs (random)', 'Avg - S-O converging motifs (random)', 
                                'Avg - S-O chain motifs (random)',  'Avg - T-O chain motifs (random)',
                                'Avg - T-O diverging motifs (random)', 'Avg - T-O converging motifs (random)',
                                 
                                'SD - S-O diverging motifs (random)', 'SD - S-O converging motifs (random)', 
                                'SD - S-O chain motifs (random)',  'SD - T-O chain motifs (random)',
                                'SD - T-O diverging motifs (random)', 'SD - T-O converging motifs (random)',
                                
                                'Z - S-O diverging motifs', 'Z - S-O converging motifs', 
                                'Z - S-O chain motifs',  'Z - T-O chain motifs',
                                'Z - T-O diverging motifs', 'Z - T-O converging motifs',
                                'Number of nodes in each layer with downstream output', 
                                'Number of connections in each layer'])

# %%
for (sparsity, m) in masksTest: 
    FOM, FOMList = fom(m)
    SODM, numFC = sodm(m)
    SOCM = socm(m)
    SOChain = sochain(m)
    TODM = todm(m)
    TOCM = tocm(m)
    TOChain = tochain(m)

    randomNetDF = buildRandomMotifsDF(numFC, FOMList, numRand=1000)

    AvgrSODM = randomNetDF['rSODM'].mean()
    AvgrSOCM = randomNetDF['rSOCM'].mean()
    AvgrSOChain = randomNetDF['rSOChain'].mean()
    AvgrTODM = randomNetDF['rTODM'].mean()
    AvgrTOCM = randomNetDF['rTOCM'].mean()
    AvgrTOChain = randomNetDF['rTOChain'].mean()

    SDrSODM = randomNetDF['rSODM'].std()
    SDrSOCM = randomNetDF['rSOCM'].std()
    SDrSOChain = randomNetDF['rSOChain'].std()
    SDrTODM = randomNetDF['rTODM'].std()
    SDrTOCM = randomNetDF['rTOCM'].std()
    SDrTOChain = randomNetDF['rTOChain'].std()

    ZSODM = (SODM - AvgrSODM)/SDrSODM
    ZSOCM = (SOCM - AvgrSOCM)/SDrSOCM
    ZSOChain = (SOChain - AvgrSOChain)/SDrSOChain
    ZTODM = (TODM - AvgrTODM)/SDrTODM
    ZTOCM = (TOCM - AvgrTOCM)/SDrTOCM
    ZTOChain = (TOChain - AvgrTOChain)/SDrTOChain

    zscoreData = [float(sparsity), m, 
                    float(FOM), 
                    float(SODM), float(SOCM), float(SOChain),
                    float(TODM), float(TOCM), float(TOChain),

                    float(AvgrSODM), float(AvgrSOCM), float(AvgrSOChain),
                    float(AvgrTODM), float(AvgrTOCM), float(AvgrTOChain),

                    float(SDrSODM), float(SDrSOCM), float(SDrSOChain),
                    float(SDrTODM), float(SDrTOCM), float(SDrTOChain),

                    float(ZSODM), float(ZSOCM), float(ZSOChain),
                    float(ZTODM), float(ZTOCM), float(ZTOChain),

                    numFC, FOMList]

    zscoreDF.loc[len(zscoreDF.index)] = zscoreData

zscoreDF.to_csv(os.path.join(f, timestamp, 'zscoreDF.csv'))
# %%
print(zscoreDF.head())


