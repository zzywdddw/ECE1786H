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
![Uploading image.png…]()

