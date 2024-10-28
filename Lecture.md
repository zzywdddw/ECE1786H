## LEC 1: Word Embedding Properties & Meaning Extraction  
#### 2024.9.10 Tuesday
Word 1 - apple - Vapple = {a<sub>0</sub>, a<sub>1</sub>,a<sub>2</sub>,...,a<sub>99</sub>,}  
Word 2 - banana - Vbanana = {b<sub>0</sub>, b<sub>1</sub>,b<sub>2</sub>,...,b<sub>99</sub>,}

Let's say apple and banana are similar - both fruit - we expect their vectors to be 'close' in a numerical way.    

One way to measure this closeness is with Euclidean Distance; e.g. the distance between Vapple and Vbanana is:  
![image](https://github.com/user-attachments/assets/732e6a5c-4ff1-497c-a7eb-60468895ab9e)  
this is called the ' Statistical Approcah' to NLP   

Cosine similarity:  
![image](https://github.com/user-attachments/assets/a29f188c-7aea-48e4-8aa7-1e2b44db0b3f)  
This method considers the direction of the vector, normalizes out magnitude  

Also, beside the surprising relationships:  
V<sub>queen</sub> - V<sub>king</sub> = V<sub>woman</sub> - V<sub>man</sub>  
V<sub>big</sub> - V<sub>biggest</sub> = V<sub>small</sub> - V<sub>smallest</sub>  

This can be used to answer a question like: "big is to biggest as small is to ....?"  
By computing: V<sub>answer</sub> = V<sub>biggest</sub> - V<sub>big</sub> + V<sub>small</sub>    




## LEC 2: How Word Embeddings are Created
#### 2024.9.17 Tuesday  
1. The mathematician went to the store  
2. The engineer went to the store
3. The mathematician solved the problem
4. The engineer solved the problem
   
Sentence 1 & 2 imply a similarity between engineer and mathematician. This is an example of what is called the <strong>Distributional Hypothesis</strong>: "Words appearing in similar contexts are related"  
<br/><br/>
How the input and output are represented?  
-> 1-hot encoding  
![image](https://github.com/user-attachments/assets/13ccc01d-f352-4586-aa58-2781b3e96b85)  
<br/><br/>
let's say the vocabulary size |V| = 10  
let's assume the embedding size dimension = 4  
that means we want 10 embeddings of size 4 each  
these are typically stored in a matrix, like  
![image](https://github.com/user-attachments/assets/ae0f31f3-b156-49f6-a51a-78d0bbbd832d)  
![image](https://github.com/user-attachments/assets/5d212fb7-1716-476e-bda5-9f631b9db057)  
Word A is the target, i.e.: <strong>e</strong><sup>i</sup><sub>j</sub>  
Word B is the context, but also the correct 'label' for the supervised training, represented as 1-hot encoding.  
An interesting nuance/improvement that isn't obvious: observe that the w<sup>i</sup><sub>j</sub> - the weights in the above neural network model are also word embeddings that are being trained!  
Why is that? There is one associated with each output, which is again one word in the vocabulary.  

The model is using these to 'match' (convolution-wise) to the input embedding. [This takes a bunch of thinking to understand]  
Then just replace the wij with the eij




## LEC 3: Classification of Language using word embeddings
#### 2024.9.24 Tuesday  
For A2, we need to train two types of networks to detect if a sentence is either Objective or Subjective.  
So we want to classify sentences as objective or subjective:  
1. each of these sentence is input as regular(ASSCII) text;
2. the first processing step is called 'tokenization' which often (but not always) break words into smaller units.
3. Each of those units will need an embedding - in the case that there is a word that doesnt break down into known units, it will be assigned the 'unknown' token and associated embedding
We will use GloVe embeddings with dimension = 100 this time.
So the input to the models is a sentence of words that converted int oa sequence of embeddings.
![image](https://github.com/user-attachments/assets/5ac374f9-7d3a-49ef-a5cc-2bb3cc0cd0a3)
Notice that input sentences would be of different lengths; but some neural nets are set up for fixed-length inputs:
1. In that case, must “pad” inputs with zeroes (i.e. embeddings that are zero) to make a batch of all equal size length inputs.
2. CNN

Convolutional Neural Net (CNN)  
Let’s first spend a few minutes reviewing CNNs  
Recall that pictures are made up of pixels, each pixel is possibly three numbers, the amount of Red, Green and Blue in the pixel. Say a picture is 1000x1000 pixels:
![image](https://github.com/user-attachments/assets/2d347533-3bc0-434a-beef-4d81a311546f)  

We would like to look for  
1. Single words that indicate subjective/objective
2. Pairs of words, triplets, 4 words...?
    - In general to look for K words
    - In CNN-terminology, we'd say that we want to train kernels of size K X 100
    - In a similar way that CNN kernels scan across the field of a picture (in computer cision), these a kernel would 'scan' across a sentence
  
In the context of classifying the language sentence, a kernel, which might be of size 2x100, would just scan across the sentence's word embeddings once.  
It would be trained to look for a 2-word pattern of meaning that would contribute to learning the labels: subjective or objective  

Assignment 2 suggests having n1 kernels of size k1 x100 and n2 of size k2x100. Each one randomly initialized, as all weights/kernels are before training  

If you have an input sentence of N words a kernel size of k1=2x100? (Where 100 is the embedding dimension?); assume the stride=1. (what is stride?) What is the size of the output?  
   - Get N-1 values out; if you have n1 such kernels, then you get n1 x (N-1)
   - In general for kernel of size k, you get N-k+1 values
![image](https://github.com/user-attachments/assets/9c66d021-9177-49cd-8627-b6b4c8577ca1)


Yoon & Kim (paper referenced in A2) suggest choosing the maximum across all N-k+1 values  
   - i.e. maxpool
   - Then feed all of those maximum values, from all the kernels into a multi-layer perceptron (MLP) - also called linear, fully connected layer(s).

Does this make sense?
   Should look for patterns of 1-6 words that give sense of whether sentence is objective or subjective  


<strong>Important note </strong>: for Method 2, I also asked you to enable the training of the embeddings themselves. That is, the gradient descent is set to propagate back into the embeddings, so that they learn to do strong>this task</strong> specifically well  
   - This is the case for the next network - Transformers
   - Looking ahead: it will be important to understand that the embeddings themselves are essentially network parameters, and so it makes sense to train them all together
        - the only difference is that you use different parameters depending on what the actual input is
    
<strong>Aside</strong>: Recurrent Neural Networks (RNNs) had traditionally been used for Natural Language Processing
   - An RNN typically takes 1 embedding in at a time, and has a cycle in which the output feeds back into the network, along with the next input:
    ![image](https://github.com/user-attachments/assets/a915f1c9-c2bd-4762-a250-c25f64b737c1)
   - RNNs (LSTMs, GRUs) were always rather problematic in that convergence of training was difficult to achieve and unreliable.
   - Also, to me, the fact that all information was “pinched” through the size of the embedding meant that lots of information was lost
   - Transformers are closer to CNN’s in that sense, and they have essentially replaced RNNs for NLP





## LEC 4: Introduction to Language Models &Transformers
#### 2024.10.1 Tuesday  
Transformers are the state-of-the-art method for:  
1. Classification of language - not MLP, or CNN as in A2, and not recurrent neural networks (RNNs) as found in the pre-2018 literature & discussed
2. Generation of language: Generation is qualitatively different than classification (in the same way reading and writing are different, perhaps), but neural networks that can do one can be re-purposed to do the other.
<br>
Given a prior sequence of words, a Language Model determines the probability that each word in the vocabulary is a good next word.
e.g.  Partial sentence: “I believe clean running water is important for ...”
      - High probability words: health; success; everyone;
      - Low probability words: lights; computers; desks

One definition of good:  
1. Grammatically correct (when appended to the prior words)
2. Makes sense (when appended)
i.e. that, with that next word, the sentence or partial sentence is likely to found in the use of the language
<br>
A little more specifically, the task of a language model is to do this:

Given: One or more words in a sequence
Compute: probability of every word in the vocabulary being the next word according to 1 & 2 above (but possibly much more 'goodness')
<br>
Assume that the size of the vocabulary |v| = M

(So we have W0,W1,..,WM-1 as input words)  

Now given a sequence of n words X0,X1,...,Xn-1  

![image](https://github.com/user-attachments/assets/47dad16e-618d-41f1-87b8-127cbf5d403e)  

P(W0 is Xn) means the probability that word W0 is the next word (denoted as Xn) given the previous sequence

<br>  

Given that we can do this, we can use these probs to compute the ‘likliehood’ of an entire sequence of words. (Again, the likliehood that the sequence is grammatical/makes sense; or that this sequence would be found in the use of the language)  
- Do by computing P(X0) P(X1) ... P(Xn-1) for an n-word sequence.
  
- We can judge a language model by computing this probability on a fixed sequence of words that are known to be "good." Must always use the same sequence of words to compare different models
   
- The Perplexity of a model is a function of the above product, but it is both inverted (so that lower numbers are “better”) and normalized by taking the nth root

<br>  

So, the language model that we want is predicator that looks like:  

![image](https://github.com/user-attachments/assets/51a7754a-6908-4546-85ce-3438a4aad4e4) 

Here is a one view of the picture of a Transformer:  

![image](https://github.com/user-attachments/assets/69ac3267-0020-479a-97d1-7242a5268b81)  

Three important comments/insights:  

1. One reason this is called a Transformer is that, for each Ti block, the number of inputs = number of outputs. So, the information coming in is ‘transformed’ but not increased or decreased in size. There are d x n numbers coming in and d x n going out, where d is the embedding size.
   
      - This allows an number (K) of Ti blocks to be easily stacked
        
      - Using that, one way that the big transformers are made big is by adding more blocks this way, i.e. just making K bigge
        
2. The number of embeddings coming in - n - is called the <strong>context size</strong>, and the input itself is called the <strong>context</strong>; it is very important
      - Context is very important in all human communication!
      - Original Vashwani Transformer, n= 512
           - GPT-2 n = 1024 (1K)
           - GPT-3 n = 2048 (2K)
           - GPT-4 n= 8K -> 32K; and now much higher still
      - While bigger seems to always be better, the attention block inside the transformer is n**2 in the computational complexity; we’ll cover that next, but some of the newer models are finding ways around it
4.  Observe the “Language head” which, to repeat, looks just like the output in Assignment 1, Section 3
   A. This is used when training the network to be a language model; key note: when we want to use this network to do classification, we chop off this language head, and put a classifier "head" - an MLP just to do the classification on it, and train it as a classifier, with the parameters of the pre-trained transformer blocks left intact.
5. The big models are most useful when pre-trained on lots of words - e.g. GPT-2 was trained on billions of words, GPT-4 trillions
6. The sequence of words input - X0, X1, X2 ... does not contain any information about the order of the words (despite it looking like it does)
   - Does order of words matter to make this prediction?
   - e.g. fox the brown quick vs. the quick brown fox
   - Certainly that ordering must matter! (RNNs don’t have this issue)

Note: he word embeddings themselves are learned as part of the same training process, as discussed above.



## LEC 5: The Core Mechanisms of Transformers
#### 2024.10.8 Tuesday  
Recall: When training a Transformer from scratch, we train it to be a language model: given a sequence of n words, predict the probability that eacg word in the vocabulary is the next (n+1)st word  
![image](https://github.com/user-attachments/assets/42322de2-4501-44c7-b300-a67815f1f206)  
n = number of tokens in the input sequence.  
d = the size (dimensionality) of the vector representation (embedding) for each token.  
Even n tokens always go in, may use fewer than n, as in above example  
Important: in a single inference, the final MLP just takes in d inputs ( not n x d)  
Which d inputs? The d inputs corresponding to the last input token.  
<br>  
<br>  
Here is the structure of one such Transformer Block:  
![image](https://github.com/user-attachments/assets/b35a6066-c3ba-4d42-88b9-b395aaa00734)
  
Multi-Head Self Attention:  
- The input word embeddings are transformed from their initial, very general meanings(across all uses/contexts of the words) to something more specific to the context - i.e. the other words in the sequence
- e.g. the embedding for "bank" would become different in these contexts:
     1. she sat on the river bank
     2. He emptied his bank account
     3. They hsould not bank on the result
From * in above picture consider how to compute the outputs Yi from the input Xi（ignoring skip connections for now)

e.g.  X0    X1     X2   X3   X4  
      He    emptied his bank account  
Self attention asks the question: how similar is each word to all the preceding words and itself?  
e.g. how similar is X3 (bank) to X0(He)?  
how similar is X3 (bank) to X1(emptied)?  
how similar is X3 (bank) to X2(his)?  
how similar is X3 (bank) to X3(bank)?  

How have we computed a single number that says how similar/realted two words are?  
=> use the dot product of the word embedding - bigger means more similar  
Define: score(Xi, Xj) = Xi *(dot) Xj  
Then, we need to normalize across these scores when use it to compute combination:  
So define:![image](https://github.com/user-attachments/assets/ed1507b8-403a-43ba-a806-b66264c288d6)  
The score alpha gives the relative importance of Xj to Xi and we use it to compute a new embedding, Yi that combines different proportions of the Xj, like so:  
![image](https://github.com/user-attachments/assets/2f1305db-fba4-416d-af17-5d6c30196239)  
So, we are adding a fraction of the meaning of those other words into the original embedding; the fraction depends on how similar the words are.  
This is how "bank" gets more "river" into it. The literature refers to these as 'contextual embeddings'  
Compute the Yi from i=0 up to n-1(if all occupied with embeddings)  
<br>  

Notice that the Xi get used in three ways:  
1. as the focus Xi in score(Xi, Xj) - we will refer to this as the <strong>"query"</strong>
2. as the 'search' Xj in score(Xi, Xj) - call this the <strong>"key"</strong>
3. To compute the Yi in ** above - we call this the <strong>"value"</strong>
<br>
In all thress cases we will transform the input Xi by multipying it times ( three didfferent) matrices consisting.
The matrices will be a size that leaves the size of the ouput the same as the Xi input, hence just transformed.  

![image](https://github.com/user-attachments/assets/d4cb7d0f-b37e-4a04-914f-1a28eee0af49)

#### Therefore, the overall computation becomes:  
![image](https://github.com/user-attachments/assets/0ffcab7a-fbcf-4f73-b49d-799ad5bb5a5d)  
<br>  
<br>  
Now, return to the specific Transformer block above:  
![image](https://github.com/user-attachments/assets/fc24cb40-525f-4cf4-9fc3-23366dd4994c)  
  
The other parts of the above tramsformer block are more common  
1. Skip connections(red lines)- are an insurance policy against failed optimization - essentially 'skips' the block if nothing useful happening, but keeps the information passing through the block
2. Layer Normalization, Dropout and weight decay also used.

Very important: The computation in between the dashed lines are all independent! Yi is a function of some or all of the Xi, but can all be done in parrallel! This speed-up was crucial to the ability to train against huge amounts of training data.  
Also, the feed-forward MLPs are isolated - i.e. there are n separated MLPs, not one big one, and their parameters are all the same.  
Think of the transform block as a set independently computed "rows", where there is one "row" per input token/embedding. Each row has the same trained parameters in, it, including the layer norm.


## LEC 6: Language Generation using Transformers
#### 2024.10.15 Tuesday  
Recall: A language model is trained to predict the next word that comes after angiven input sequence of words.  
So, if you can dothat, then you can predict a whole sequence of output words, one at a time, by taking each predicated word, append it to the input sequence of words and then predicting/generating the next word after that and so on.  
=> This is called "auto-regressive" generation  
<br>  
Here is the what I'd call the "Auto-regressive loop:" (maybe obvious, but is very important):  
e.g. If the input started as: "The clean river flowed"  
Call model to infer and generate next words: "into"  
<br>  
Then, the next input to the model is "The clean river flowed into"  
Next word might be "the"  
<br>  
Next input would be "The clean river flowed into the"  
Generate next word, and so on. This is how chatGPT delivers what you ask for.  
<br>  
Each word is pretty expensive, in that it is a full inference run just to get one word, of a very large model.  
![image](https://github.com/user-attachments/assets/05191384-3870-428b-b8aa-8b71ef4865b6)  
So, for a given input sequence, which single word is selected as the output?  
The process of selecting the word, based on the output probabilities, is called decoding.  
<br>  
Method 1: Greedy Decoding. Just select the highest probability word.  
- Greedy does not work well in general - it picks obvious words, but these often lead to boring, uninteresting sequences of words; also repetitive
- Greedy may choose the most likely next word, but does not result in the most likely dequence of generated words
- Get stuck in a highly local optimum in the space of all possible generated sequences
<br>
We can express this issue mathematically as follows: Given an input sequence of embeddings/words/tokens X0 ... Xn-1 we want the generated sequence of output words Y0 ... Yg-1 of g words to be the most likely sequence

i.e. we want P(Y0) x P(Y1) x ... x P(Yg-1) to be maximized  

But, we dont know P(Y1) when selecting Y0 (or Yj, j>i when selecting Yi)  

This is a hard problem because there are M the power of g possible sequences of g output words, which is a big exponential. Here M is the size of the vocabulary If M = 50000, g=20, 50000**20 is huge. Worse, each one of those is a full forward inference of the model!  

![image](https://github.com/user-attachments/assets/784a5b7d-c271-4762-a775-c902129d1e0d)

<br>
Method 2: Beam Search is a heuristic that prunes the full search tree a lot.  

General Description:  
- Walk down the tree, keeping the K-most probable sequences  
- At each level of the tree, consider top V possible next words for each of the K sequences. (Means you need K separate inferences instead of min 1)
- Compute the full sequence probability of those possible KxV sequences
      -> keep the K highest
- repeat until have generated the number of desired tokens(or hit stop token)
<br>
Method 3: Sampling (most commonly used)

Given: The set of output probabilities P(W0), P(W1), .., P(Wm-1)  

Select: The next word through a random prcess, in which the probability of selecting word Wi is P(wi)  

![image](https://github.com/user-attachments/assets/d1023084-f517-47e9-abf2-ef33548fa686)  

This random process has a nice side-effect: if you don’t like the output sequence that you get, you can just try again and get a new one.  

Also, this process actually reflects the fact that there are many ways to answer a given question, or to create language.  

However, this randomness is also part of the source of the ‘hallucinations’ that you’ve probably heard of from chatGPT/LLMs. Some bad luck on the first word could just send the answer in the wrong direction!!!!  

<br>  

There are several variations to know about:  

Top-k sampling: rather than select from all M tokens in the vocabulary, only select from the top K most probable words  

Top-p sampling: only select from the top words that all together have the sum of probabilities = p (or closest). 0 <=p <= 1 (most common used)  

   - if set p=1 that mean use all M tokens

   - often p = 0.8

There is one more important adjustment to this process that is important: The probabilities from the network output are adjusted to control whether the generated sequences are more or less creative/diverse.
It is done with a parameter,t, called the temperature.  

A high T gives more diverse words, done by adjusting probs before sampling.  
   - T = 1 is normal - the probabilities are unchanged
   - T > 1 makes less probable words more likely
   - T < 1 makes more probable words more likely
   - T = 0 makes decoding greedy
  
![image](https://github.com/user-attachments/assets/6a5976e5-a076-43f3-abfa-3db58c0e7fc7)
 
<br>  

Other notes:  
- Often a conbination of top-p and temperature are the commonly used generation parameters  
- There is also a repetition penalty - e.g. divide li by 1,3 if the tiken corresponding to li has already been used in this generation







