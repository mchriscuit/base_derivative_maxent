import numpy as np
import re
from Levenshtein import editops
from tqdm import tqdm

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
                    BASE-DERIVATIVE MAXENT DEFINITION
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


class BDMaxEnt:
    def __init__(
        self,
        inputs: list,
        candidates: list,
        winner_idxs: list,
        lexemes: list,
        constraint_names: list,
        constraint_types: list,
        constraint_definitions: list,
        weights,
    ):

        """*=*=*= CANDIDATE INFORMATION  *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
        n_forms       : int, number of tableaux
        inputs        : list of str, inputs for each tableau
        candidates    : list of lists, candidates for each tableau
        winner_idxs   : list of int, index of winning candidate for
                        each tableau
        lexemes       : list of tuples, sequence of lexemes for each input
        outputs       : list of str, the winning candidate for each tableau
        """
        self.n_forms = len(inputs)
        self.inputs = inputs
        self.candidates = candidates
        self.winner_idxs = winner_idxs
        self.lexemes = lexemes
        self.outputs = [candidates[i][winner_idxs[i]] for i in range(self.n_forms)]

        """ *=*=*= BASE INFORMATION  *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
            lex2idx   : dict, maps a sequence of lexemes to a tableau
            lex2win   : dict, maps a sequence of lexemes to the index of
                        the winning candidate for its tableau
        """
        self.lex2idx = {m: i for i, m in enumerate(lexemes)}
        self.lex2win = {lexemes[i]: winner_idxs[i] for i in range(self.n_forms)}

        """ *=*=*= CONSTRAINT INFORMATION  *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
            constraint_names        : list of str, names of each constraint
            constraint_types        : list of str, the type of each constraint
            constraint_definitions  : list of tuples, regex of each constraint
            weights                 : numpy array, n x 1 vector of weights for
                                      each constraint
        """
        self.constraint_names = constraint_names
        self.constraint_types = constraint_types
        self.constraint_definitions = constraint_definitions
        self.weights = weights

    def parse(self, input: str, candidate: str):
        """Goal:
          Align and process the input and candidate strings
        Takes in:
          input     : str, input string
          candidate : str, candidate string
        Returns:
          parsed input and candidate strings
        """
        i = list(input)
        c = list(candidate)

        ## Calculate the alignment between the input and candidate
        operations = editops(input, candidate)
        for operation in operations:
            edit, input_idx, candidate_idx = operation
            if edit == "delete":
                c.insert(input_idx, "_")
            elif edit == "insert":
                i.insert(candidate_idx, "_")

        i = "".join(i)
        c = "".join(c)
        return i, c

    def compute_harmonies(self, violations):
        """Takes in:
          violations : `m x n` matrix of constraint violations
        Returns:
          `m x 1` vector of harmonies
        """
        return np.dot(violations, self.weights)

    def compute_neg_exp_harmonies(self, harmonies):
        """Takes in:
          violations : `m x 1` vector of harmonies
        Returns:
          `m x 1` vector of exponentiated negative harmonies
        """
        return np.exp(-harmonies)

    def compute_marginalization(self, neg_exp_harmonies):
        """Takes in:
          violations : `m x 1` vector of exponentiated negative harmonies
        Returns:
          sum over all exponentiated negative harmonies
        """
        return np.sum(neg_exp_harmonies, axis=0)

    def compute_probabilities(self, violations):
        """Takes in:
          violations : `m x n` matrix of constraint violations
        Returns:
          `m x 1` vector of probabilities
        """
        harmonies = self.compute_harmonies(violations)
        neg_exp_harmonies = self.compute_neg_exp_harmonies(harmonies)
        marginalization = self.compute_marginalization(neg_exp_harmonies)

        return neg_exp_harmonies / marginalization

    def compute_violation(
        self, input: str, candidate: str, lexemes: tuple, bases=None, base_lexemes=None
    ):
        """Goal:
          Takes in an input-candidate pair and returns an `m x 1`
          vector of violations
        Takes in:
          input        : str, input string
          candidates   : str, candidate string
          lexemes      : tuple, sequence of lexemes for the input
          bases        : tuple, candidates, violations, and probabilities
          base_lexemes : tuple, sequence of lexemes for the bases
        Returns:
          `m` vector of violations
        """
        n_inputs = len(self.inputs)

        ## Parse the input-candidate pair
        parsed_input, parsed_candidate = self.parse(input, candidate)

        ## Are there eligible bases?
        if bases and base_lexemes:

            ## Retrieve the bases and computed probabilities
            base_forms, base_probs = bases
            n_base_lexemes = len(base_lexemes)

            ## Slice off the candidate to include only the part that
            ## matches the base
            remainder = candidate.split("-")[:n_base_lexemes]
            remainder = "-".join(remainder)

            ## Parse the base_remainder pair for each possible base
            parsed_base_pairs = []
            for base_form in base_forms:
                base_pair = self.parse(base_form, remainder)
                parsed_base_pairs.append(base_pair)

        ## Initialize violation profiles
        n_constraints = len(self.constraint_names)
        violation = np.zeros(n_constraints)

        ## Loop through each constraint
        for i, constraint_type in enumerate(self.constraint_types):
            # M: Markedness, assign a violation for each appearance
            #    of some regex in a candidate
            # S: Substitution, assign a violation if for any pair of
            #    I-O characters, some change happens
            # D: Deletion, assign a violation, if '_'
            #    is found on the output candidate
            # I: Insertion, assign a violation, if '_'
            #    is found on the input candidate
            # BD: Base Deletion, assign a violation if '_' is found in the
            #    output candidate
            # BI: Base Insertion, assign a violation if '_' is found in the
            #    base candidate
            # BS: Base Substitution, assign a violation for any pair of
            #    B-D characters, some change happens

            ## I-O and M constraints
            constraint_definition = self.constraint_definitions[i]
            if constraint_type == "M":
                pattern = constraint_definition
                violation[i] = sum(1 for _ in re.finditer(pattern, parsed_candidate))

            elif constraint_type == "D":
                pattern = "_"
                violation[i] = sum(1 for _ in re.finditer(pattern, parsed_candidate))

            elif constraint_type == "I":
                pattern = "_"
                violation[i] = sum(1 for _ in re.finditer(pattern, parsed_candidate))

            elif constraint_type == "S":
                count = 0
                p1, p2 = constraint_definition
                io_paired = zip(parsed_input, parsed_candidate)
                for j, (s1, s2) in enumerate(io_paired):
                    if re.search(p1, s1) and re.search(p2, s2):
                        count += 1
                    elif re.search(p2, s1) and re.search(p1, s2):
                        count += 1
                violation[i] = count

            ## Base-Derivative constraints
            elif constraint_type == "BD" and bases and base_lexemes:

                ## Check if there is a positional requirement:
                condition = ""
                if len(constraint_definition) == 3:
                    condition = constraint_definition[0]

                ## For each base, count and weight by their probability
                for j, parsed_base_pair in enumerate(parsed_base_pairs):
                    parsed_base, parsed_remainder = parsed_base_pair
                    base_prob = base_probs[j]
                    if not re.search(condition, parsed_base):
                        continue
                    count = 0
                    for parsed_remainder_segment in parsed_remainder:
                        if re.search("_", parsed_remainder_segment):
                            count += 1
                    violation[i] += count * base_prob

            elif constraint_type == "BI" and bases and base_lexemes:

                ## Check if there is a positional requirement:
                condition = ""
                if len(constraint_definition) == 3:
                    condition = constraint_definition[0]

                ## For each base, count and weight by their probability
                for j, parsed_base_pair in enumerate(parsed_base_pairs):
                    parsed_base, parsed_remainder = parsed_base_pair
                    base_prob = base_probs[j]
                    if not re.search(condition, parsed_base):
                        continue
                    count = 0
                    for parsed_base_segment in parsed_base:
                        if re.search("_", parsed_base_segment):
                            count += 1
                    violation[i] += count * base_prob

            elif constraint_type == "BS" and bases and base_lexemes:

                ## Check if there is a positional requirement:
                condition = ""
                if len(constraint_definition) == 3:
                    condition = constraint_definition[0]
                    p1, p2 = constraint_definition[1:]
                else:
                    p1, p2 = constraint_definition

                ## For each base, count and weight by their probability
                for j, parsed_base_pair in enumerate(parsed_base_pairs):
                    parsed_base, parsed_remainder = parsed_base_pair
                    base_prob = base_probs[j]
                    if not re.search(condition, parsed_base):
                        continue
                    count = 0
                    bd_paired = zip(parsed_base, parsed_remainder)
                    for k, (s1, s2) in enumerate(bd_paired):
                        if re.search(p1, s1) and re.search(p2, s2):
                            count += 1
                        elif re.search(p2, s2) and re.search(p1, s1):
                            count += 1
                    violation[i] += count * base_prob

        return violation

    def compute_output(self, tableau_idx):
        """Goal: Calculates the violations and probability of
              each candidate for a given tableau
        Takes in:
          tableau_idx     : int, index for the tableau to calculate
                            probabilities over
          show_violations : boolean, for debugging, prints out the
                            probabilities calculated over each candidate
        Returns:
          candidates    : list, candidates
          violations    : numpy array, `m x n` matrix of violations
          probabilities : numpy array, `m x 1` vector of probabilities
        """
        input = self.inputs[tableau_idx]
        candidates = self.candidates[tableau_idx]
        lexemes = self.lexemes[tableau_idx]

        ## Check if the form has a base
        # if removing the right-most morpheme does not have an output, then
        # we assume that the form has no base
        has_base = False
        potential_base = lexemes[:-1]
        if potential_base in self.lex2idx.keys():
            base_tableau_idx = self.lex2idx[potential_base]

            ## Generate the winner for the base form
            cs, vs, ps = self.compute_output(base_tableau_idx)
            bases = (cs, ps)
            base_lexemes = self.lexemes[base_tableau_idx]
            has_base = True

        ## Compute violations for each candidate
        violations = []
        for candidate in candidates:
            if has_base:
                violation = self.compute_violation(
                    input, candidate, lexemes, bases, base_lexemes
                )
            else:
                violation = self.compute_violation(input, candidate, lexemes)
            violations.append(violation)

        ## Turn the violations into a matrix
        violations = np.array(violations)

        ## Compute the probability over all possible candidates
        probabilities = self.compute_probabilities(violations)

        return candidates, violations, probabilities

    def learn(self, eta=0.05, iters=10000):
        """Goal: Runs the stochastic gradient descent algorithm
        given the data
        Takes in:
          eta            : float, plasticity measure
          iters          : int, number of iterations
        Returns:
          weights : numpy array, `n x 1` vector of constraint weights
          history : list, history of constraint weights over iteration
        """

        history = [self.weights.squeeze()]

        ## One iteration consists of a single pass through all the data
        for i in tqdm(range(iters)):

            ## Get the order of the candidates that we will be running through
            order = np.random.permutation(range(self.n_forms))

            ## For each tableau
            for tableau_idx in order:

                ## Get the input
                input = self.inputs[tableau_idx]

                ## Get the candidates
                candidates = self.candidates[tableau_idx]

                ## Get the lexeme(s)
                lexemes = self.lexemes[tableau_idx]

                ## Get the winner index
                w_idx = self.winner_idxs[tableau_idx]

                ## Calculate the violations of the winning candidate
                w_candidate = candidates[w_idx]

                ## Determine what the base of the lexeme sequences is
                ## If removing the outer-most lexeme does not result in an
                ## appropriate output, then we assume that the form has no base
                potential_base = lexemes[:-1]
                if potential_base in self.lex2idx.keys():
                    base_tableau_idx = self.lex2idx[potential_base]

                    ## Get the observed base for that lexeme combination
                    w_base = ([self.outputs[base_tableau_idx]], [1])
                    base_lexemes = self.lexemes[base_tableau_idx]
                    w_vs = self.compute_violation(
                        input, w_candidate, lexemes, w_base, base_lexemes
                    )
                else:
                    w_vs = self.compute_violation(input, w_candidate, lexemes)

                ## Calculate the probabilities of the candidates
                cs, vs, ps = self.compute_output(tableau_idx)

                ## Compute the gradient
                dL_dw = w_vs - (vs * ps).sum(0, keepdims=True)

                # update the weight
                self.weights = self.weights - eta * dL_dw.T

                # change any negative numbers to 0
                self.weights.T[self.weights.T < 0] = 0

                # store the weights
                history.append(self.weights.squeeze())

        return self.weights, history
