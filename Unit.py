#Class to store information related to units (subjects) in network

class unit():    
    def __init__(self, C_in, U_in, adj):
        self.C = C_in #Baseline demographics/characteristics
        self.U = U_in #Unobserved/latent/hidden confounders
        self.A = 0 #Treatment
        self.M = 0 #Mediating variable -- conceptually associated with network
        self.Y = 0 #Outcome
        self.adj = adj #List of nodes adjacent to This node