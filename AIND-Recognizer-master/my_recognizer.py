import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    # TODO implement the recognizer
    for X, lengths in test_set.get_all_Xlengths().values():
        log_L = {}
        best_guess = None
        highest_score = float("-inf")
        for word, model in models.items():
            try:
                word_score = model.score(X, lengths)
                log_L[word] = word_score

                if word_score > highest_score:
                    highest_score = word_score
                    best_guess = word
            except:
                log_L[word] = float("-inf")

        guesses.append(best_guess)
        probabilities.append(log_L)

    return probabilities, guesses
    #raise NotImplementedError
