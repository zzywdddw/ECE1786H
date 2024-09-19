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




