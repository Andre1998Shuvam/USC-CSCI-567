import numpy as np

class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_{t-1} = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: A dictionary mapping each observation symbol to its index 
        - state_dict: A dictionary mapping each state to its index
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array where alpha[i, t-1] = P(Z_t = s_i, X_{1:t}=x_{1:t})
                 (note that this is alpha[i, t-1] instead of alpha[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        alpha = np.zeros([S, L])
        ######################################################
        # TODO: compute and return the forward messages alpha
        ######################################################
        alpha[:, 0] = self.pi * self.B[:, O[0]]
        prod_sum = np.sum(self.A * alpha[:, 0], axis=0)
        for i in range(1, L):
            for s in range(S):
                alpha[s][i] = self.B[s][O[i]] * \
                    np.sum([self.A[s_1][s] * alpha[s_1][i-1]
                            for s_1 in range(S)])

        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array where beta[i, t-1] = P(X_{t+1:T}=x_{t+1:T} | Z_t = s_i)
                    (note that this is beta[i, t-1] instead of beta[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        beta = np.zeros([S, L])
        #######################################################
        # TODO: compute and return the backward messages beta
        #######################################################
        beta[:, L-1] = 1
        for i in range(L-2, -1, -1):
            for s in range(S):
                beta[s][i] = np.sum([self.B[s_1][O[i+1]] * self.A[s][s_1] * beta[s_1][i+1]
                                     for s_1 in range(S)])

        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(X_{1:T}=x_{1:T})
        """

        #####################################################
        # TODO: compute and return prob = P(X_{1:T}=x_{1:T})
        #   using the forward/backward messages
        #####################################################
        return np.sum(self.forward(Osequence)[:, 0] * self.backward(Osequence)[:, 0], axis=0)

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - gamma: (num_state*L) A numpy array where gamma[i, t-1] = P(Z_t = s_i | X_{1:T}=x_{1:T})
                           (note that this is gamma[i, t-1] instead of gamma[i, t])
        """
        ######################################################################
        # TODO: compute and return gamma using the forward/backward messages
        ######################################################################
        gamma = np.zeros((len(self.pi), len(Osequence)))
        gamma = self.forward(Osequence) * self.backward(Osequence) / self.sequence_prob(Osequence)

        return gamma

    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array where prob[i, j, t-1] = 
                    P(Z_t = s_i, Z_{t+1} = s_j | X_{1:T}=x_{1:T})
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        #####################################################################
        # TODO: compute and return prob using the forward/backward messages
        #####################################################################
        O = self.find_item(Osequence)
        for i in range(S):
            for j in range(S):
                for l in range(L-1):
                    prob[i][j][l] = self.forward(Osequence)[
                        i][l] * self.A[i][j] * self.B[j][O[l+1]] * self.backward(Osequence)[j][l+1]

                    prob[i][j][l] = prob[i][j][l] / \
                        self.sequence_prob(Osequence)

        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden states (return actual states instead of their indices;
                    you might find the given function self.find_key useful)
        """
        path = []
        ################################################################################
        # TODO: implement the Viterbi algorithm and return the most likely state path
        ################################################################################
        O = self.find_item(Osequence)
        L = len(Osequence)
        delta = np.zeros((len(self.pi), len(Osequence)))
        delta_1 = np.zeros((len(self.pi), len(Osequence)), dtype=int)
        delta[:, 0] = self.pi * self.B[:, O[0]]

        for t in range(1, len(Osequence)):
            for s in range(len(self.pi)):
                delta[s][t] = self.B[s][O[t]] * \
                    np.max([self.A[s_1][s] * delta[s_1][t-1]
                            for s_1 in range(len(self.pi))])
                delta_1[s][t] = np.argmax([self.A[s_1][s] * delta[s_1][t-1]
                                           for s_1 in range(len(self.pi))])

        path.append(np.argmax([delta[s][L-1] for s in range(len(self.pi))]))
        for t in range(L - 1, 0, -1):
            path.append(delta_1[path[-1]][t])

        path = [self.find_key(self.state_dict, z) for z in path][::-1]
        return path

    # DO NOT MODIFY CODE BELOW

    def find_key(self, obs_dict, idx):
        for item in obs_dict:
            if obs_dict[item] == idx:
                return item

    def find_item(self, Osequence):
        O = []
        for item in Osequence:
            O.append(self.obs_dict[item])
        return O
