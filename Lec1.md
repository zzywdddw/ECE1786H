### Natural Language Processing  
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
