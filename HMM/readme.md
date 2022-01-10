  HMM 

Hidden Markov Models (50 points)
================================

General instructions
--------------------

*   In this task you will implement various inference algorithms for **HMM** and also apply them to sentence tagging. We provide the bootstrap code and you are expected to complete the functions.
*   Do not import libraries other than those already imported in the original code.
*   Please follow the type annotations. You have to make the function’s return values match the required type.
*   Only modifications in files {`hmm.py`, `tagger.py`} in the "work" directory will be accepted and graded. All other modifications will be ignored. You can work directly on Vocareum, or download all files from "work", code in your own workspace, and then upload the changes (recommended).
*   Click the Submit button when you are ready to submit your code for auto-grading. Your final grade is determined by your **last** submission.

Q1 Implement the inference algorithms
-------------------------------------

In `hmm.py`, you will find a class called HMM whose attributes specify the model parameters of a Hidden Markov Model (including its initial state probability, transition probability, and emission probability). You need to implement the following six functions

*   `forward`: compute the forward messages
*   `backward`: compute the backward messages
*   `sequence_prob`: compute the probability of observing a particular sequence
*   `posterior_prob`: compute the probability of the state at a particular time step given the observation sequence
*   `likelihood_prob`: compute the probability of state transition at a particular time step given the observation sequence
*   `viterbi`: compute the most likely hidden state path using the Viterbi algorithm.

We have discussed how to compute all these via dynamic programming in the lecture. Here, the only thing you need to pay extra attention to is that the indexing system is slightly different between the python code and the formulas we discussed (the former starts from 0 and the latter starts from 1). Read the comments in the code carefully to get a better sense of this discrepancy.

Q2 Application to speech tagging
--------------------------------

Part-of-Speech (POS) is a category of words (or, more generally, of lexical items) which have similar grammatical properties. (Example: noun, verb, adjective, adverb, pronoun, preposition, conjunction, interjection, and sometimes numeral, article, or determiner.) Part-of-Speech Tagging (POST) is the process of marking up a word in a text (corpus) as corresponding to a particular part of speech, based on both its definition and its context.

Here you will use HMM to perform POST, where the tags are states and the words are observations. We collect our dataset and tags with the Dataset class. Dataset class includes tags, train\_data and test\_data. Both datasets include a list of sentences, and each sentence is an object of the Line class. You only need to implement the `model_training` function and the `speech_tagging` function in `tagger.py`.

*   `model_training`: in this function, you need to build an instance of the HMM class by setting its five parameters `(pi, A, B, obs_dict, state_dict)`. The way you estimate the parameter `pi, A, B` is simply by counting the corresponding frequency from the given training set, as we discussed in the class. Read the comments in the code for more instructions.
*   `speech_tagging`: given the HMM built from model\_training, now your task is to run the Viterbi algorithm to find the most likely tagging of a given sentence. One particular detail you need to take care of is when you meet a new word which was unseen in the training dataset. In this case, you need to update the dictionary `obs_dict` accordingly, and also expand the emission matrix by assuming that the probability of seeing this new word under any state is 1e-6. Again, read the comments in the code for more instructions.

Q3 Testing
----------

Once you finish these two parts, run `hmm_test_script.py`. We will first run all your inference algorithms on a toy HMM model specified in `hmm_model.json`, and then also your tagging code on the dataset stored in `pos_sentences.txt` and `pos_tags.txt`. In both cases, the script tells you what your outputs vs. the correct outputs are.

Grading guideline
-----------------

1 Inference algorithms (30 points)

1.  `forward` function - 5 = 5x1 points
2.  `backward` function - 5 = 5x1 points
3.  `sequence_prob` function - 2.5 = 5x0.5 points
4.  `posterior_prob` function - 5 = 5x1 points
5.  `likelihood_prob` function - 5 = 5x1 points
6.  `viterbi` function - 7.5 = 5\*1.5 points

There are 5 sets of grading data used to initialize the HMM class and test your functions. To receive full credits, your output of functions 1-5 should be within an error of 1e-6, and your output of the viterbi function should be identical with ours.

2 Application to Part-of-Speech Tagging (20 points)

1.  `model_training` - 10 = 10x(your\_correct\_pred\_cnt/our\_correct\_pred\_cnt)
2.  `speech_tagging` - 10 = 10x(your\_correct\_pred\_cnt/our\_correct\_pred\_cnt)

We will use the dataset given to you for grading this part (with a different random seed). We will train your model and our model on same train\_data. `model_training` function and `speech_tagging` function will be tested separately.

In order to check your model\_training function, we will use 50 sentences from `train_data` to do Part-of-Speech Tagging (your model + our tagging function vs. our model + our tagging function). To receive full credits, your prediction accuracy should be identical or better than ours.

In order to check your `speech_tagging` function, we will use 50 sentences from `test_data` to do Part-of-Speech Tagging (your model + your tagging function vs. our model + our tagging function). Again, to receive full credits, your prediction accuracy should be identical or better than ours.