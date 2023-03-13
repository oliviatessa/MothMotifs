# %%
import os 
import pickle 
import numpy as np
import math
import scipy

import pandas as pd 

# %% [markdown]
# # Load the pruned networks 

# %%
modeltimestamp = '2022_11_05__01_04_10'

dataOutput = '/gscratch/dynamicsai/otthomas/MothPruning/mothMachineLearning_dataAndFigs/DataOutput/Experiments/pruned_bias/pruned_bias/'
modelSubdir = os.path.join(dataOutput, modeltimestamp)

zscoreOutput = '/gscratch/dynamicsai/otthomas/MothMotifs/MothMotifs/DataOutput/zscoreTables/'
zscoreSubdir = os.path.join(zscoreOutput, modeltimestamp)
if not os.path.exists(zscoreSubdir):
    os.mkdir(zscoreSubdir)

preProcessSubdir = os.path.join(dataOutput, modeltimestamp, 'preprocessedMasks')
if not os.path.exists(preProcessSubdir):
    os.mkdir(preProcessSubdir)
    
#Load the networks
sparseNetsFile = 'sparseNetworks.pkl'
sparseNetworks = pickle.load(open(os.path.join( modelSubdir, sparseNetsFile), 'rb'))

masksFile = 'masks_minmax_Adam5.pkl'
masks = pickle.load(open(os.path.join(modelSubdir, masksFile), 'rb'))

bmasksFile = 'bmasks_minmax_Adam5.pkl'
bmasks = pickle.load(open(os.path.join(modelSubdir, bmasksFile), 'rb'))

#Load the losses
losses = pickle.load(open(os.path.join(modelSubdir, 'preprocessedNets', 'pruneLosses.pkl'),'rb'))

losses = np.array(losses)
losses = np.transpose(losses)

lossesDF = pd.DataFrame(losses, columns=['0%', '15%', '25%', '35%', '45%', '55%', '65%', '75%', '85%', '90%', '91%', 
                                        '92%', '93%', '94%', '95%', '96%', '97%', '98%' ])

# %% [markdown]
# # Find the masks

# %%
findSparseMasks = True

if findSparseMasks == True:
    #Only find all the sparse masks if they are not already saved
    sparseMasks = []
    for i in range(len(sparseNetworks)):
        sparsity = sparseNetworks[i][0]
        #Extract the masks using the sparsity index 
        # masks is organized by [sparsity index][layer number][network number]

            
        m = [masks[sparsity][j][i] for j in range(5)]
        bm = [bmasks[sparsity][j][i] for j in range(5)]

        #Combine mask and bias mask by adding bias mask as last row of mask 
        mask = [np.append(m[j], np.array(bm[j]).reshape([1, len(bm[j])]), axis=0) for j in range(5)]

        sparseMasks.append((sparsity, mask))
    
    pickle.dump(sparseMasks, open(os.path.join(preProcessSubdir, 'sparseOptimalMasks.pkl'), 'wb'))

else:
    sparseMasks = pickle.load(open(os.path.join(preProcessSubdir, 'sparseOptimalMasks.pkl'), 'rb'))



# %%
'''
Organization of masks for evaluating motifs throughout pruning.
'''
findallPrunedMasks = True

if findallPrunedMasks == True:

    #Only find all the sparse masks if they are not already saved
    allPrunedNets = []
    for i in range(len(sparseNetworks)):
        net = []
        for s in range(18): #18?
            m = [masks[s][j][i] for j in range(5)] #combines all layers that belong to a network at a certain sparsity
            bm = [bmasks[s][j][i] for j in range(5)]

            mask = [np.append(m[j], np.array(bm[j]).reshape([1, len(bm[j])]), axis=0) for j in range(5)]
            net.append((s, mask))

        allPrunedNets.append(net)
    
    pickle.dump(allPrunedNets, open(os.path.join(preProcessSubdir, 'allPrunedMasks.pkl'), 'wb'))

else:
    allPrunedNets = pickle.load(open(os.path.join(preProcessSubdir, 'allPrunedMasks.pkl'), 'rb'))


# %% [markdown]
# # Remove ghost nodes 

# %% [markdown]
# Ghost nodes: nodes with no upstream input

# %%
def rmGhostNodes(masks, rm=True, allnets=False): 
    if rm == True:
        sparseMasks_wo_ghosts = []

        #We need slightly different code for removing ghost nodes from all pruned networks
        if allnets == True:
            print('here')
            for k in range(len(masks)): #iterate over the networks
                net = []
                for s in range(len(masks[k])): #iterate over the sparsities 
                    count = 0

                    #Iterate over the masking layers in each network 
                    for l in range(len(masks[k][s][1])):
                        m = masks[k][s][1][l]
                        print(m.shape)
                        for i in range(len(m)):
                            #Iterate over the columns of the mask 
                            for j in range(len(m[i].T)):
                                column = m[i].T[j]
                                #Check to see if there are any connections between this node and the nodes in the previous layer. 
                                #If there are no connections, that means there are no upstream connections and this is a ghost node. 
                                n = np.count_nonzero(column)
                                if n == 0:
                                    #print('Found a ghost node: %s node in layer %s.' % (j, i))
                                    count += 1
                                    #There is no input into this node 
                                    #so make all downstream connections 0

                                    #i+1 gets us to the next mask 
                                    #where the jth row is the ghost node 
                                    m[i+1][j] = m[i+1][j] * 0
                        net.append((masks[k][s][0],m))

                    sparseMasks_wo_ghosts.append(net)
            
        else:
            for k in range(len(masks)):
                count = 0
                m = masks[k][1]
                #Iterate over the masking layers in each network 
                for i in range(len(m)):
                    #Iterate over the columns of the mask 
                    for j in range(len(m[i].T)):
                        column = m[i].T[j]
                        #Check to see if there are any connections between this node and the nodes in the previous layer. 
                        #If there are no connections, that means there are no upstream connections and this is a ghost node. 
                        n = np.count_nonzero(column)
                        if n == 0:
                            #print('Found a ghost node: %s node in layer %s.' % (j, i))
                            count += 1
                            #There is no input into this node 
                            #so make all downstream connections 0

                            #i+1 gets us to the next mask 
                            #where the jth row is the ghost node 
                            m[i+1][j] = m[i+1][j] * 0

                sparseMasks_wo_ghosts.append((masks[k][0],m))

            pickle.dump(sparseMasks_wo_ghosts, open(os.path.join(preProcessSubdir, 'masks_wo_ghost_nodes.pkl'), 'wb'))

    else:
        sparseMasks_wo_ghosts = pickle.load(open(os.path.join(preProcessSubdir, 'masks_wo_ghost_nodes.pkl'), 'rb'))

    return sparseMasks_wo_ghosts

# %% [markdown]
# # Remove dead nodes

# %% [markdown]
# Dead nodes: nodes with no downstream output

# %%
def rmDeadNodes(masks, rm=True, allNets=False): 
    if rm == True:
        sparseMasks_wo_dead = []

        #We need slightly different code for removing ghost nodes from all pruned networks
        if allNets == True:
            for k in range(len(masks)): #iterate over the networks
                net = []
                for s in range(len(masks[k])): #iterate over the sparsities 
                    count = 0
                    m = masks[k][s][1]
                    #Reverse iterate over the masking layers in each network, excluding the input and output layers
                    for i in reversed(range(1, len(m))):
                        #Iterate over the rows of the mask, skipping the bias (last row)
                        for j in range(len(m[i])-1):
                            row = m[i][j]
                            #Check to see if there are any connections between this node and the nodes in the next layer. 
                            #If there are no connections, that means there are no downstream connections and this is a dead node. 
                            n = np.count_nonzero(row)
                            if n == 0:
                                #print('Found a dead node: %s node in layer %s.' % (j, i))
                                count += 1
                                #There is no output from this node 
                                #so make all upstream connections 0

                                #i-1 gets us to the previous mask 
                                #where the jth column is the ghost node 
                                m[i-1].T[j] = m[i-1].T[j] * 0

                    net.append((masks[k][s][0],m))
                    
                sparseMasks_wo_dead.append(net)
        else:
            for k in range(len(masks)):
                count = 0
                m = masks[k][1]
                #Reverse iterate over the masking layers in each network, excluding the input and output layers
                for i in reversed(range(1, len(m))):
                    #Iterate over the rows of the mask, skipping the bias (last row)
                    for j in range(len(m[i])-1):
                        row = m[i][j]
                        #Check to see if there are any connections between this node and the nodes in the next layer. 
                        #If there are no connections, that means there are no downstream connections and this is a dead node. 
                        n = np.count_nonzero(row)
                        if n == 0:
                            #print('Found a dead node: %s node in layer %s.' % (j, i))
                            count += 1
                            #There is no output from this node 
                            #so make all upstream connections 0

                            #i-1 gets us to the previous mask 
                            #where the jth column is the ghost node 
                            m[i-1].T[j] = m[i-1].T[j] * 0

                sparseMasks_wo_dead.append((masks[k][0],m))

        pickle.dump(sparseMasks_wo_dead, open(os.path.join(preProcessSubdir, 'masks_wo_dead_nodes.pkl'), 'wb'))

    else:
        sparseMasks_wo_dead = pickle.load(open(os.path.join(preProcessSubdir, 'masks_wo_dead_nodes.pkl'), 'rb'))

    return sparseMasks_wo_dead

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
    numFCUS = [0,0,0,0,0,0]

    for i in range(len(m)):
        nodes = 0
        #Calculate second-order converging motifs
        for column in m[i].T:
            n = np.count_nonzero(column)
            if n >= 2:
                SOCM += math.factorial(n)/(math.factorial(2)*math.factorial(n-2))

            
            #also calculate number of remaining nodes with upstream input, skip input 
            if n > 0: 
                nodes += 1
                    
        numFCUS[i+1] = nodes

    return SOCM, numFCUS

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
# #### Bi-fan

# %%
def bifan(m):
    '''
    Calculates the number of bi-fan motifs in the network.
        
    Input(s): the mask of the pruned network, as a list of matrices
    Returns: BIFAN (total number of bi-fan motifs in the network)
    '''
    
    BIFAN = 0

    for i in range(len(m)): 
        #For each row, calculate the dot product of the row with the rest of the rows in the mask 
        for j in range(len(m[i])-1):
            row = m[i][j]
            mat = m[i][j+1:]
            count = np.dot(mat, row) #Each element in count represents the number of bifans row j shares with all subsequent rows 

            #Calculate the number of bifans
            for n in count: 
                n = int(n)
                if n >= 2: 
                    BIFAN += math.factorial(n)/(math.factorial(2)*math.factorial(n-2))
    
    return BIFAN

# %% [markdown]
# #### Bi-parallel

# %%
def bipar(m):
    '''
    Calculates the number of bi-parallel motifs in the network.
        
    Input(s): the mask of the pruned network, as a list of matrices
    Returns: BIPAR (total number of bi-parallel motifs in the network)
    '''
    BIPAR = 0 
    for i in range(len(m)): 
        if i != 4: 
            #Find the product between two layers.
            #Exclude the bias by excluding the last row
            prod = np.matmul(m[i],m[i+1][0:-1])
            
            #Take the factorial of the whole product matrix. 
            #Factorial will have NaN values so np.sum may cause an error 
            fact_mat = scipy.special.factorial(prod)
            
            fact_2 = math.factorial(2)

            fact_mat_2 = scipy.special.factorial(prod-2)

            denom = fact_2 * fact_mat_2

            comb_mat = np.divide(fact_mat, denom)

            if math.isnan(np.sum(np.ma.masked_invalid(comb_mat))):
                pass
            else:
                #Number of bi-parallel motifs is the sum of the resultant matrix
                BIPAR += np.sum(np.ma.masked_invalid(comb_mat))
            
            
        else: 
            pass
    
    return BIPAR

# %% [markdown]
# ### Random networks

# %% [markdown]
# #### Build random network

# %%
def randomPruning(FOMList, numFC): 
    '''
    Randomly prunes a fully-connected random network to the same number of weights and nodes (in each layer respectively) as the real network. 
    This part of constructing the random network occurs before the ghost and dead nodal pruning of the real network.
    '''

    #Remove the bias from numFC
    numFC = [numFC[i]-1 if FOMList[i][1] != 0 else numFC[i] for i in range(len(numFC)-1)] #range(len(numFC)-1) because there is never a bias in the output layer
    numFC.append(7) #add in final layer

    #Build fully-connected network of zeros with only the live nodes found after post-pruning
    # the ineffuctual nodes. 
    r1 = np.zeros((numFC[0]*numFC[1]))
    r2 = np.zeros((numFC[1]*numFC[2]))
    r3 = np.zeros((numFC[2]*numFC[3]))
    r4 = np.zeros((numFC[3]*numFC[4]))
    r5 = np.zeros((numFC[4]*numFC[5]))

    #Nlw corresponds to the number of remaining weights or biases in the real network. 
    Nlw1 = FOMList[0][0]
    Nlw2 = FOMList[1][0]
    Nlw3 = FOMList[2][0]
    Nlw4 = FOMList[3][0]
    Nlw5 = FOMList[4][0]

    #Nln corresponds to the total number of live nodes after post-pruning associated with
    # one layer (inc. input and output nodes). 
    Nln1 = numFC[0]+numFC[1]
    Nln2 = numFC[1]+numFC[2]
    Nln3 = numFC[2]+numFC[3]
    Nln4 = numFC[3]+numFC[4]
    Nln5 = numFC[4]+numFC[5]

    #Nr corresponds to the difference between the total remaining weights and total remaining
    # nodes. 
    Nr1 = Nlw1 - Nln1
    Nr2 = Nlw2 - Nln2
    Nr3 = Nlw3 - Nln3
    Nr4 = Nlw4 - Nln4
    Nr5 = Nlw5 - Nln5

    NList = [(Nlw1, Nln1, Nr1),(Nlw2, Nln2, Nr2),(Nlw3, Nln3, Nr3),
             (Nlw4, Nln4, Nr4),(Nlw5, Nln5, Nr5)]
    
    rList = [r1, r2, r3, r4, r5]

    #print('Nlw1: %s, Nln1: %s, Nr1: %s' %(Nlw1, Nln1, Nr1))
    #print('Nlw2: %s, Nln2: %s, Nr2: %s' %(Nlw2, Nln2, Nr2))
    #print('Nlw3: %s, Nln3: %s, Nr3: %s' %(Nlw3, Nln3, Nr3))
    #print('Nlw4: %s, Nln4: %s, Nr4: %s' %(Nlw4, Nln4, Nr4))
    #print('Nlw5: %s, Nln5: %s, Nr5: %s' %(Nlw5, Nln5, Nr5))

    #Set the first Nr values of the list to one if Nr is greater than zero and then randomly shuffle. 
    for i in range(len(rList)):
        if NList[i][2] > 0:
            rList[i][0:NList[i][2]] = 1
            np.random.shuffle(rList[i])
        else:
            pass


    #Reshape the matrices
    for i in range(len(rList)):
        rList[i] = np.reshape(rList[i], (numFC[i],numFC[i+1]))


    #The following is to assure we have at least one live weight connected to each node in
    # the layer (inputs and outputs). 

    for r in range(len(rList)):
        if NList[r][2] > 0: 
            extraNodeCount = 0
            #For each row place a random 1
            for i in range(len(rList[r])):
                #print('r: %s and i: %s' %(r,i))
                row = np.array(rList[r][i])
                zeroElements = np.nonzero(row==0)[0] #Finds all indices with a zero

                #There is a small chance that there are no zero elements even after random shuffling. 
                #In this situation, we need to keep track of the extra node, and add it in later
                if len(zeroElements) != 0: #if we've found a nonzero element
                    idx = np.random.choice(zeroElements) #Picks a random index
                    rList[r][i][idx] = 1 #Sets the value at that index to one
                else: 
                    extraNodeCount += 1

            #For each column place a random 1
            for j in range(len(rList[r].T)):
                #print('r: %s and j: %s' %(r,j))
                col = np.array(rList[r].T[j])
                zeroElements = np.nonzero(col==0)[0] #Finds all indices with a zero
                if len(zeroElements) != 0:
                    idx = np.random.choice(zeroElements) #Picks a random index
                    rList[r].T[j][idx] = 1 #Sets the value at that index to one
                else: 
                    extraNodeCount += 1

            if extraNodeCount != 0: 
                #If we have some extra nodes, add them in randomly. 
                for n in range(extraNodeCount):
                    z = np.nonzero(rList[r]==0)
                    idx = np.random.choice(np.arange(len(z[0])))
                    x = z[0][idx]
                    y = z[1][idx]
                    rList[r][x][y] = 1

        else:
            if r == 3:
                if NList[r][2] < 0:
                    if abs(NList[r][2]) <= (NList[r][0]-numFC[r]):
                        flatr = rList[r].flatten()
                        flatr[0:abs(NList[r][2])] = 1
                        np.random.shuffle(flatr)
                        rList[r] = np.reshape(flatr, (numFC[r],numFC[r+1]))
                    else:
                        flatr = rList[r].flatten()
                        flatr[0:(NList[r][0]-numFC[r])] = 1
                        np.random.shuffle(flatr)
                        rList[r] = np.reshape(flatr, (numFC[r],numFC[r+1]))


                #For each row place a random 1
                for i in range(len(rList[r])):
                    #print('r: %s and i: %s' %(r,i))
                    row = np.array(rList[r][i])
                    zeroElements = np.nonzero(row==0)[0] #Finds all indices with a zero
                    idx = np.random.choice(zeroElements) #Picks a random index
                    rList[r][i][idx] = 1 #Sets the value at that index to one

                if abs(NList[r][2])+numFC[r] < FOMList[r][0]:
                    diff = FOMList[r][0] - (abs(NList[r][2])+numFC[r])
                    for d in range(diff): #place a random one in the matrix 
                        z = np.nonzero(rList[r]==0)
                        idx = np.random.choice(np.arange(len(z[0])))
                        x = z[0][idx]
                        y = z[1][idx]
                        rList[r][x][y] = 1

            if r == 4: 
                #If the number of nodes in the penultimate layer is less than the nodes in the output layer
                if numFC[r] < numFC[r+1]:
                    w = FOMList[r][0]
                    numExtraCols = numFC[r+1]-numFC[r]
                    m = np.eye(numFC[r])
                    #If the number of remaining weights is greater that the number of nodes in the output layer
                    if w > numFC[r+1]:
                        numExtraW = w - numFC[r+1]
                        for k in range(numExtraCols):
                            col = np.zeros((numFC[r],1))
                            z = np.nonzero(col==0)
                            idx = np.random.choice(z[0])
                            col[idx] = 1
                            m = np.append(m, col, axis=1)
                        for t in range(numExtraW):
                            z = np.nonzero(m==0)
                            idx = np.random.choice(np.arange(len(z[0])))
                            x = z[0][idx]
                            y = z[1][idx]
                            m[x][y] = 1
                    else:
                        for k in range(numExtraCols):
                            col = np.zeros((numFC[r],1))
                            z = np.nonzero(col==0)
                            idx = np.random.choice(z[0])
                            col[idx] = 1
                            m = np.append(m, col, axis=1)
                else: #if numFC[r] >= numFC[r+1]
                    w = FOMList[r][0]
                    numExtraRows = numFC[r]-numFC[r+1]
                    m = np.eye(numFC[r+1])
                    #If the number of remaining weights is greater than the number of nodes in the penultimate layer
                    if w > numFC[r]:
                        numExtraW = w - (numFC[r+1]+numExtraRows)
                        for k in range(numExtraRows):
                            row = np.zeros((1,numFC[r+1]))
                            z = np.nonzero(row==0)
                            idx = np.random.choice(z[1])
                            row[0][idx] = 1
                            m = np.append(m, row, axis=0)
                        for t in range(numExtraW):
                            z = np.nonzero(m==0)
                            idx = np.random.choice(np.arange(len(z[0])))
                            x = z[0][idx]
                            y = z[1][idx]
                            m[x][y] = 1
                    else:
                        for k in range(numExtraRows):
                            row = np.zeros((1,numFC[r+1]))
                            z = np.nonzero(row==0)
                            idx = np.random.choice(z[1])
                            row[0][idx] = 1
                            m = np.append(m, row, axis=0)
                            
                np.random.shuffle(m)
                rList[r]=m

    '''Create the random bias vectors'''

    #Build zeros vector the length of the bias term
    r1b = np.zeros((numFC[1]))
    r2b = np.zeros((numFC[2]))
    r3b = np.zeros((numFC[3]))
    r4b = np.zeros((numFC[4]))
    r5b = np.zeros((numFC[5]))

    #Set the number of live bias terms to 1
    r1b[0:FOMList[0][1]] = 1 
    r2b[0:FOMList[1][1]] = 1
    r3b[0:FOMList[2][1]] = 1 
    r4b[0:FOMList[3][1]] = 1 
    r5b[0:FOMList[4][1]] = 1 

    #Randomly shuffle the bias matrices
    np.random.shuffle(r1b)
    np.random.shuffle(r2b)
    np.random.shuffle(r3b)
    np.random.shuffle(r4b)
    np.random.shuffle(r5b)

    rbList = [r1b, r2b, r3b, r4b, r5b]
    
    randomNet = []
    for i in range(len(rList)):
        randomNet.append(np.vstack([rList[i],rbList[i]]))


    return randomNet


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
    rSOCM, rnumFCUS = socm(randomNet)
    rSOChain = sochain(randomNet)
    rTODM = todm(randomNet)
    rTOCM = tocm(randomNet)
    rTOChain = tochain(randomNet)
    rBIFAN = bifan(randomNet)
    rBIPAR = bipar(randomNet)
    
    return rFOM, rFOMList, rSODM, rSOCM, rSOChain, rTODM, rTOCM, rTOChain, rBIFAN, rBIPAR, rnumFC, rnumFCUS 

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
                                        'rTOChain',
                                        'rBIFAN',
                                        'rBIPAR',
                                        'rnumFC',
                                        'rnumFCUS'])
    #

    for r in range(numRand):
        randomNet = randomPruning(FOMList, numFC)
        rFOM, rFOMList, rSODM, rSOCM, rSOChain, rTODM, rTOCM, rTOChain, rBIFAN, rBIPAR, rnumFC, rnumFCUS = randomNetMotifs(randomNet) #

        rMotifs = [float(rSODM), float(rSOCM), float(rSOChain), float(rTODM), float(rTOCM), float(rTOChain), float(rBIFAN), float(rBIPAR), rnumFC, rnumFCUS] #
        randomNetDF.loc[len(randomNetDF.index)] = rMotifs

    return randomNetDF

# %% [markdown]
# ## Z-score dataframe

# %% [markdown]
# ### Remove ghost and dead nodes from the networks 

# %% [markdown]
# sparseMasks = sparseMasks[0:5]

# %% [markdown]
# sparseMasks_wo_G = rmGhostNodes(sparseMasks, rm=True)
# sparseMasks_wo_G_D = rmDeadNodes(sparseMasks_wo_G, rm=True)

# %%
zscoreDF = pd.DataFrame(columns=['Sparsity Index', 'Masks',
                                '1-O motifs (real)', 
                                'S-O diverging motifs (real)', 'S-O converging motifs (real)', 
                                'S-O chain motifs (real)',  'T-O chain motifs (real)',
                                'T-O diverging motifs (real)', 'T-O converging motifs (real)',
                                'T-O Bi-Fan motifs (real)', 'T-O Bi-Parallel motifs (real)',
                                
                                'Avg - S-O diverging motifs (random)', 'Avg - S-O converging motifs (random)', 
                                'Avg - S-O chain motifs (random)',  'Avg - T-O chain motifs (random)',
                                'Avg - T-O diverging motifs (random)', 'Avg - T-O converging motifs (random)',
                                'Avg - T-O Bi-Fan motifs', 'Avg - T-O Bi-Parallel motifs',

                                 
                                'SD - S-O diverging motifs (random)', 'SD - S-O converging motifs (random)', 
                                'SD - S-O chain motifs (random)',  'SD - T-O chain motifs (random)',
                                'SD - T-O diverging motifs (random)', 'SD - T-O converging motifs (random)',
                                'SD - T-O Bi-Fan motifs', 'SD - T-O Bi-Parallel motifs',
                                
                                'Z - S-O diverging motifs', 'Z - S-O converging motifs', 
                                'Z - S-O chain motifs',  'Z - T-O chain motifs',
                                'Z - T-O diverging motifs', 'Z - T-O converging motifs',
                                'Z - T-O Bi-Fan motifs', 'Z - T-O Bi-Parallel motifs',

                                'Number of nodes in each layer with downstream output',
                                'Number of nodes in each layer with upstream input', 
                                'Number of connections in each layer'])

#
#
#
#

# %% [markdown]
# for (sparsity, m) in sparseMasks_wo_G_D: 
#     FOM, FOMList = fom(m)
#     SODM, numFC = sodm(m)
#     SOCM, numFCUS = socm(m)
#     SOChain = sochain(m)
#     TODM = todm(m)
#     TOCM = tocm(m)
#     TOChain = tochain(m)
#     BIFAN = bifan(m)
#     BIPAR = bipar(m)
# 
#     randomNetDF = buildRandomMotifsDF(numFC, FOMList, numRand=10)
# 
#     AvgrSODM = randomNetDF['rSODM'].mean()
#     AvgrSOCM = randomNetDF['rSOCM'].mean()
#     AvgrSOChain = randomNetDF['rSOChain'].mean()
#     AvgrTODM = randomNetDF['rTODM'].mean()
#     AvgrTOCM = randomNetDF['rTOCM'].mean()
#     AvgrTOChain = randomNetDF['rTOChain'].mean()
#     AvgrBIFAN = randomNetDF['rBIFAN'].mean()
#     AvgrBIPAR = randomNetDF['rBIPAR'].mean()
# 
#     SDrSODM = randomNetDF['rSODM'].std()
#     SDrSOCM = randomNetDF['rSOCM'].std()
#     SDrSOChain = randomNetDF['rSOChain'].std()
#     SDrTODM = randomNetDF['rTODM'].std()
#     SDrTOCM = randomNetDF['rTOCM'].std()
#     SDrTOChain = randomNetDF['rTOChain'].std()
#     SDrBIFAN = randomNetDF['rBIFAN'].std()
#     SDrBIPAR = randomNetDF['rBIPAR'].std()
# 
#     ZSODM = (SODM - AvgrSODM)/SDrSODM
#     ZSOCM = (SOCM - AvgrSOCM)/SDrSOCM
#     ZSOChain = (SOChain - AvgrSOChain)/SDrSOChain
#     ZTODM = (TODM - AvgrTODM)/SDrTODM
#     ZTOCM = (TOCM - AvgrTOCM)/SDrTOCM
#     ZTOChain = (TOChain - AvgrTOChain)/SDrTOChain
#     ZBIFAN = (BIFAN - AvgrBIFAN)/SDrBIFAN
#     ZBIPAR = (BIPAR - AvgrBIPAR)/SDrBIPAR
# 
#     zscoreData = [float(sparsity), m, 
#                     float(FOM), 
#                     float(SODM), float(SOCM), float(SOChain),
#                     float(TODM), float(TOCM), float(TOChain),
#                     float(BIFAN), float(BIPAR),
# 
#                     float(AvgrSODM), float(AvgrSOCM), float(AvgrSOChain),
#                     float(AvgrTODM), float(AvgrTOCM), float(AvgrTOChain),
#                     float(AvgrBIFAN), float(AvgrBIPAR),
# 
#                     float(SDrSODM), float(SDrSOCM), float(SDrSOChain),
#                     float(SDrTODM), float(SDrTOCM), float(SDrTOChain),
#                     float(SDrBIFAN), float(SDrBIPAR),
# 
#                     float(ZSODM), float(ZSOCM), float(ZSOChain),
#                     float(ZTODM), float(ZTOCM), float(ZTOChain),
#                     float(ZBIFAN), float(ZBIPAR),
# 
#                     numFC, numFCUS, FOMList]
#     
#     #
#     #
#     #
#     #
# 
#     zscoreDF.loc[len(zscoreDF.index)] = zscoreData
# 
# zscoreDF.to_csv(os.path.join(zscoreSubdir, 'zscoreDF.csv'))
# 

# %%
allPrunedNets = allPrunedNets[0:1]
print("done")

allsparseMasks_wo_G = rmGhostNodes(allPrunedNets, rm=True, allnets=True)

#allsparseMasks_wo_G_D = rmDeadNodes(allsparseMasks_wo_G, rm=True, allNets=True)


# %% [markdown]
# count=0
# for net in allsparseMasks_wo_G_D:
#     for s in range(len(net)):
#         sparsity = net[s][0]
#         m = net[s][1]
# 
#         FOM, FOMList = fom(m)
#         SODM, numFC = sodm(m)
#         SOCM, numFCUS = socm(m)
# 
#         randNet = randomPruning(FOMList, numFC)
# 
#         rFOM, rFOMList = fom(randNet)
#         rSODM, rnumFC = sodm(randNet)
#         rSOCM, rnumFCUS = socm(randNet)
# 
#         if FOMList != rFOMList:
#             print('Network %s' %(count))
#             print('Real numFC down: %s' %(numFC))
#             print('Real FOMList: %s' %(FOMList))
#             print('Random FOMList: %s' %(rFOMList))
# 
#         if numFC != rnumFC:
#             print('Network %s' %(count))
#             print('Real numFC down: %s' %(numFC))
#             print('Random numFC down: %s' %(rnumFC))
# 
#         if numFCUS != rnumFCUS:
#             print('Network %s' %(count))
#             print('Real numFC up: %s' %(numFCUS))
#             print('Random numFC up: %s' %(rnumFCUS))
# 
#         count+=1
# 
#     print('done checking')
#     
# '''
# randNet_wo_G = rmGhostNodes(randNetX)
# randNet_wo_G_D = rmDeadNodes(randNet_wo_G)
# 
# for (sparsity, m) in sparseMasks_wo_G_D:
#     FOM, numW_and_B = fom(m)
#     SODM, numFC = sodm(m)
#     SOCM, numFCUS = socm(m)
#     
#     randNet = randNet_wo_G_D[1]
# 
#     rFOM, rFOMList = fom(randNet)
#     rSODM, rnumFC = sodm(randNet)
#     rSOCM, rnumFCUS = socm(randNet)
# 
#     print(numW_and_B)
#     print(rFOMList)
#     print(numFC)
#     print(rnumFC)
#     print(numFCUS)
#     print(rnumFCUS)
# '''


