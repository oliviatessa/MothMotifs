class Subgraphs:
    def __init__(self, m):
        self.m = m

    def fom(self):
        '''
        Calculates the number of first-order motifs in the network (equivalent to the number of edges).
        
        Input(s): the mask of the pruned network, as a list of matrices
        Returns: FOM (total number of first-order motifs), FOMList (Number of weights and number of bias connections
                in each layer)
        '''
        FOM = 0
        FOMList = [[0,0],[0,0],[0,0],[0,0],[0,0]] #[[Num weights, num biases]] in each layer

        for i in range(len(self.m)): 
                #Count number of connections between weights
                w_connections = np.count_nonzero(self.m[i][0:-1])
                FOMList[i][0] = w_connections
                #Count number of connections from bias
                b_connections = np.count_nonzero(self.m[i][-1])
                FOMList[i][1] = b_connections

                connections = w_connections + b_connections
                FOM += connections
        return FOM, FOMList

    def sodm(self):
        '''
        Calculates the number of second-order diverging motifs in the network.
        Also calculates the remaining number of nodes in the network. 
            
        Input(s): the mask of the pruned network, as a list of matrices
        Returns: SODM (total number of second-order diverging motifs in the network), numFC (Number of remaining nodes 
            with downstream output)
        '''
        
        SODM = 0
        numFC = [0,0,0,0,0,0] #Number of remaining nodes with downstream output

        for i in range(len(self.m)): 
            nodes = 0
            #Calculate second-order diverging motifs
            for row in self.m[i]:
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
                for row in self.m[i].T:
                    n = np.count_nonzero(row)
                    if n > 0 :
                        nodes += 1
                numFC[i+1] = nodes
        
        return SODM, numFC

    def socm(self):
        '''
        Calculates the number of second-order converging motifs in the network.
            
        Input(s): the mask of the pruned network, as a list of matrices
        Returns: SOCM (total number of second-order converging motifs in the network)
        '''

        SOCM = 0
        numFCUS = [0,0,0,0,0,0]

        for i in range(len(self.m)):
            nodes = 0
            #Calculate second-order converging motifs
            for column in self.m[i].T:
                n = np.count_nonzero(column)
                if n >= 2:
                    SOCM += math.factorial(n)/(math.factorial(2)*math.factorial(n-2))

                
                #also calculate number of remaining nodes with upstream input, skip input 
                if n > 0: 
                    nodes += 1
                        
            numFCUS[i+1] = nodes

        return SOCM, numFCUS

    def sochain(self):
        '''
        Calculates the number of second-order chain motifs in the network.
            
        Input(s): the mask of the pruned network, as a list of matrices
        Returns: SOChain (total number of second-order chain motifs in the network)
        ''' 
        
        SOChain = 0 
        for i in range(len(self.m)): 
            #Calculate second-order chain motifs
            if i != 4: 
                #Exclude the bias by excluding the last row
                SOChain += np.count_nonzero(np.matmul(self.m[i][0:-1],self.m[i+1][0:-1]))
                
                #Add in the motifs from the bias terms 
                SOChain += np.count_nonzero(np.matmul(self.m[i][-1],self.m[i+1][0:-1]))
            else: 
                pass

        return SOChain

    def todm(self):
        '''
        Calculates the number of third-order diverging motifs in the network.
            
        Input(s): the mask of the pruned network, as a list of matrices
        Returns: TODM (total number of third-order diverging motifs in the network)
        '''

        TODM = 0 
        for i in range(len(self.m)): 
            #Calculate third-order diverging motifs
            for row in self.m[i]:
                n = np.count_nonzero(row)
                if n >= 3:
                    TODM += math.factorial(n)/(math.factorial(3)*math.factorial(n-3))

        return TODM

    def tocm(self):
        '''
        Calculates the number of third-order converging motifs in the network.
            
        Input(s): the mask of the pruned network, as a list of matrices
        Returns: TOCM (total number of third-order converging motifs in the network)
        '''

        TOCM = 0 
        for i in range(len(self.m)): 
            #Calculate third-order converging motifs 
            for column in self.m[i].T:
                n = np.count_nonzero(column)
                if n >= 3:
                    TOCM += math.factorial(n)/(math.factorial(2)*math.factorial(n-2))

        return TOCM

    def tochain(self):
        '''
        Calculates the number of third-order chain motifs in the network.
            
        Input(s): the mask of the pruned network, as a list of matrices
        Returns: TOChain (total number of third-order chain motifs in the network)
        '''

        TOChain = 0
        for i in range(len(self.m)): 
            #Calculate third-order chain motifs 
            if i in (0,1,2): 
                #Count non-zero elements of (Layer 1 * Layer 2 * Layer 3)
                #Exclude the bias by excluding the last row
                m1 = np.matmul(self.m[i][0:-1],self.m[i+1][0:-1])
                TOChain += np.count_nonzero(np.matmul(m1,self.m[i+2][0:-1]))
                    
                #Add in the motifs from the bias terms 
                mbias = np.matmul(self.m[i][-1],self.m[i+1][0:-1])
                TOChain += np.count_nonzero(np.matmul(mbias,self.m[i+2][0:-1]))
            else: 
                pass

        return TOChain

    def bifan(self):
        '''
        Calculates the number of bi-fan motifs in the network.
            
        Input(s): the mask of the pruned network, as a list of matrices
        Returns: BIFAN (total number of bi-fan motifs in the network)
        '''
        
        BIFAN = 0

        for i in range(len(self.m)): 
            #For each row, calculate the dot product of the row with the rest of the rows in the mask 
            for j in range(len(self.m[i])-1):
                row = self.m[i][j]
                mat = self.m[i][j+1:]
                count = np.dot(mat, row) #Each element in count represents the number of bifans row j shares with all subsequent rows 

                #Calculate the number of bifans
                for n in count: 
                    n = int(n)
                    if n >= 2: 
                        BIFAN += math.factorial(n)/(math.factorial(2)*math.factorial(n-2))
        
        return BIFAN

    def bipar(self):
        '''
        Calculates the number of bi-parallel motifs in the network.
            
        Input(s): the mask of the pruned network, as a list of matrices
        Returns: BIPAR (total number of bi-parallel motifs in the network)
        '''
        BIPAR = 0 
        for i in range(len(self.m)): 
            if i != 4: 
                #Find the product between two layers.
                #Exclude the bias by excluding the last row
                prod = np.matmul(self.m[i],self.m[i+1][0:-1])
                
                comb_mat = scipy.special.comb(prod, 2)
                #Take the factorial of the whole product matrix. 
                #Factorial will have NaN values so np.sum may cause an error 
                #fact_mat = scipy.special.factorial(prod)
                
                #fact_2 = math.factorial(2)

                #fact_mat_2 = scipy.special.factorial(prod-2)

                #denom = fact_2 * fact_mat_2

                #comb_mat = np.divide(fact_mat, denom)

                '''
                if math.isnan(np.sum(np.ma.masked_invalid(comb_mat))):
                    pass
                else:
                    #Number of bi-parallel motifs is the sum of the resultant matrix
                    BIPAR += np.sum(np.ma.masked_invalid(comb_mat))
                '''
                
                BIPAR += np.sum(comb_mat)
                
            else: 
                pass
        
        return BIPAR

    def count_all_subgraphs(self):
        '''
        '''
        FOM, FOMList = self.fom(self.m)
        SODM, numFC = self.sodm(self.m)
        SOCM, numFCUS = self.socm(self.m)
        SOChain = self.sochain(self.m)
        TODM = self.todm(self.m)
        TOCM = self.tocm(self.m)
        TOChain = self.tochain(self.m)
        BIFAN = self.bifan(self.m)
        BIPAR = self.bipar(self.m)

        subgraphs = {"FOM":FOM,
                    "FOMList":FOMList,
                    "SODM":SODM,
                    "numFC":numFC,
                    "SOCM":SOCM,
                    "numFCUS":numFCUS,
                    "SOChain":SOChain,
                    "TODM":TODM,
                    "TOCM":TOCM,
                    "TOChain":TOChain,
                    "BIFAN":BIFAN,
                    "BIPAR":BIPAR
                    }

        return subgraphs