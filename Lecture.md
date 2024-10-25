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
