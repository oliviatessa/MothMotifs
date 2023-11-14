class RandomNets:
    def __init__(self, FOMList, numFC):
        self.FOMList = FOMList
        self.numFC = numFC
    
    def randomPruning(self.FOMList, self.numFC): 
        '''
        Randomly prunes a fully-connected random network to the same number of weights and nodes (in each layer respectively) as the real network. 
        This part of constructing the random network occurs before the ghost and dead nodal pruning of the real network.
        '''

        #Remove the bias from numFC
        self.numFC = [self.numFC[i]-1 if self.FOMList[i][1] != 0 else self.numFC[i] for i in range(len(self.numFC)-1)] #range(len(numFC)-1) because there is never a bias in the output layer
        self.numFC.append(7) #add in final layer

        #Build fully-connected network of zeros with only the live nodes found after post-pruning
        # the ineffuctual nodes. 
        r1 = np.zeros((self.numFC[0]*self.numFC[1]))
        r2 = np.zeros((self.numFC[1]*self.numFC[2]))
        r3 = np.zeros((self.numFC[2]*self.numFC[3]))
        r4 = np.zeros((self.numFC[3]*self.numFC[4]))
        r5 = np.zeros((self.numFC[4]*self.numFC[5]))

        #Nlw corresponds to the number of remaining weights or biases in the real network. 
        Nlw1 = self.FOMList[0][0]
        Nlw2 = self.FOMList[1][0]
        Nlw3 = self.FOMList[2][0]
        Nlw4 = self.FOMList[3][0]
        Nlw5 = self.FOMList[4][0]

        #Nln corresponds to the total number of live nodes after post-pruning associated with
        # one layer (inc. input and output nodes). 
        Nln1 = self.numFC[0]+self.numFC[1]
        Nln2 = self.numFC[1]+self.numFC[2]
        Nln3 = self.numFC[2]+self.numFC[3]
        Nln4 = self.numFC[3]+self.numFC[4]
        Nln5 = self.numFC[4]+self.numFC[5]

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
            rList[i] = np.reshape(rList[i], (self.numFC[i],self.numFC[i+1]))


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
                        if abs(NList[r][2]) <= (NList[r][0]-self.numFC[r]):
                            flatr = rList[r].flatten()
                            flatr[0:abs(NList[r][2])] = 1
                            np.random.shuffle(flatr)
                            rList[r] = np.reshape(flatr, (self.numFC[r],self.numFC[r+1]))
                        else:
                            flatr = rList[r].flatten()
                            flatr[0:(NList[r][0]-self.numFC[r])] = 1
                            np.random.shuffle(flatr)
                            rList[r] = np.reshape(flatr, (self.numFC[r],self.numFC[r+1]))


                    #For each row place a random 1
                    for i in range(len(rList[r])):
                        #print('r: %s and i: %s' %(r,i))
                        row = np.array(rList[r][i])
                        zeroElements = np.nonzero(row==0)[0] #Finds all indices with a zero
                        idx = np.random.choice(zeroElements) #Picks a random index
                        rList[r][i][idx] = 1 #Sets the value at that index to one

                    if abs(NList[r][2])+self.numFC[r] < self.FOMList[r][0]:
                        diff = self.FOMList[r][0] - (abs(NList[r][2])+self.numFC[r])
                        for d in range(diff): #place a random one in the matrix 
                            z = np.nonzero(rList[r]==0)
                            idx = np.random.choice(np.arange(len(z[0])))
                            x = z[0][idx]
                            y = z[1][idx]
                            rList[r][x][y] = 1

                if r == 4: 
                    #If the number of nodes in the penultimate layer is less than the nodes in the output layer
                    if self.numFC[r] < self.numFC[r+1]:
                        w = self.FOMList[r][0]
                        numExtraCols = self.numFC[r+1]-self.numFC[r]
                        m = np.eye(self.numFC[r])
                        #If the number of remaining weights is greater that the number of nodes in the output layer
                        if w > self.numFC[r+1]:
                            numExtraW = w - self.numFC[r+1]
                            for k in range(numExtraCols):
                                col = np.zeros((self.numFC[r],1))
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
                                col = np.zeros((self.numFC[r],1))
                                z = np.nonzero(col==0)
                                idx = np.random.choice(z[0])
                                col[idx] = 1
                                m = np.append(m, col, axis=1)
                    else: #if numFC[r] >= numFC[r+1]
                        w = self.FOMList[r][0]
                        numExtraRows = self.numFC[r]-self.numFC[r+1]
                        m = np.eye(self.numFC[r+1])
                        #If the number of remaining weights is greater than the number of nodes in the penultimate layer
                        if w > self.numFC[r]:
                            numExtraW = w - (self.numFC[r+1]+numExtraRows)
                            for k in range(numExtraRows):
                                row = np.zeros((1,self.numFC[r+1]))
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
                                row = np.zeros((1,self.numFC[r+1]))
                                z = np.nonzero(row==0)
                                idx = np.random.choice(z[1])
                                row[0][idx] = 1
                                m = np.append(m, row, axis=0)
                                
                    np.random.shuffle(m)
                    rList[r]=m

        '''Create the random bias vectors'''

        #Build zeros vector the length of the bias term
        r1b = np.zeros((self.numFC[1]))
        r2b = np.zeros((self.numFC[2]))
        r3b = np.zeros((self.numFC[3]))
        r4b = np.zeros((self.numFC[4]))
        r5b = np.zeros((self.numFC[5]))

        #Set the number of live bias terms to 1
        r1b[0:self.FOMList[0][1]] = 1 
        r2b[0:self.FOMList[1][1]] = 1
        r3b[0:self.FOMList[2][1]] = 1 
        r4b[0:self.FOMList[3][1]] = 1 
        r5b[0:self.FOMList[4][1]] = 1 

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