import numpy as np
import argparse

## User-defined modules
import bd.bd as bd
import utils.utils as utils

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
                        RUNNING THE MODEL
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """
if __name__ == "__main__":
    """*=*=*=*=*=*= READ FROM COMMAND LINE *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*="""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data", "-d", type=str, nargs="?", help="Name of data file.")
    parser.add_argument(
        "--constraints", "-c", type=str, nargs="?", help="Name of constraints file."
    )
    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        nargs="?",
        default=10000,
        help="Number of iterations to run model.",
    )
    args = parser.parse_args()

    """ *=*=*=*=*=*= IMPORT DATA *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
    """
    ## (1) CONSTRAINTS
    constraint_information = utils.read_constraints(args.constraints)
    names, types, definitions = constraint_information
    n_constraints = len(names)

    ## (2) INPUTS AND CANDIDATES
    data = utils.read_data(args.data)
    data = list(zip(*data))
    inputs, winner_idxs, lexemes, candidates = data

    ## (3) WEIGHTS
    weights = np.zeros(n_constraints)
    weights = np.expand_dims(weights, axis=1)
    weights = np.random.uniform(0, 10, weights.shape)

    ## Initialize MaxEnt object
    model = bd.BDMaxEnt(
        inputs, candidates, winner_idxs, lexemes, names, types, definitions, weights
    )

    ## Learn
    w, h = model.learn(iters=args.iterations)

    """ *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
                        PRINT OUT RESULTS
    =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """

    ## Print out constraint weights
    for cn, cw in zip(names, w.squeeze()):
        print("{:<25} : {:<10}".format(cn, np.round(cw, 3)))

    ## Print out candidate violations and probabilities
    for i, input in enumerate(inputs):

        ## Get input and constraint information
        x = [f"/{input}/"] + names + [""]
        x = np.array(x)

        ## Get candidate information
        cs, vs, ps = model.compute_output(i)
        cs = np.array(cs)[:, np.newaxis]
        o = np.hstack((cs, vs, ps))

        ## Stack the information
        tableau = np.vstack((x, o))
        print(tableau)
