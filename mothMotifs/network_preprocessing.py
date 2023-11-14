class NetworkPreprocess:
    def __init__(self):
        ...

    def networkOrg():
        '''
        Organization of masks for evaluating motifs throughout pruning.
        '''
        findallPrunedMasks = True

        if findallPrunedMasks == True:

            #Only find all the sparse masks if they are not already saved
            allPrunedNets = []
            for i in range(len(sparseNetworks)):
                sparsity = sparseNetworks[i][0] #Things get wonky after most sparse network is found
                net = []
                for s in range(sparsity+1): #18?
                    m = [masks[s][j][i] for j in range(5)] #combines all layers that belong to a network at a certain sparsity
                    bm = [bmasks[s][j][i] for j in range(5)]

                    mask = [np.append(m[j], np.array(bm[j]).reshape([1, len(bm[j])]), axis=0) for j in range(5)]
                    net.append((s, mask))

                allPrunedNets.append(net)
            
            pickle.dump(allPrunedNets, open(os.path.join(preProcessSubdir, 'allPrunedMasks.pkl'), 'wb'))

        else:
            allPrunedNets = pickle.load(open(os.path.join(preProcessSubdir, 'allPrunedMasks.pkl'), 'rb'))


    def rmGhostNodes(masks, rm=True, allnets=False): 
        if rm == True:
            sparseMasks_wo_ghosts = []

            #We need slightly different code for removing ghost nodes from all pruned networks
            if allnets == True:
                messNets = []
                for k in range(len(masks)): #iterate over the networks
                    net = []
                    for s in range(len(masks[k])): #iterate over the sparsities 
                        count = 0

                        #Iterate over the masking layers in each network 
                        #for l in range(len(masks[k][s][1])):
                        m = masks[k][s][1]
                        #print(m.shape)
                        #print(len(m))
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
                                    if i == 4:
                                        messNets.append(k)

                                    #i+1 gets us to the next mask 
                                    #where the jth row is the ghost node
                                    else: 
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