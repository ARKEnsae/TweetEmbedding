# 1. 100 k tweets

## 1. Variation de la window, learning_rate fixé (=0.01)
dim = 20  
K = 5  
sample = 0.001

*learning_rate = 0.01*


1. *window = 2*  

  **Corr = -0.143**
  
  Évolution sur 10 epochs : -0.0989, -0.1538, -0.2747, -0.2308, -0.2308, -0.1703, -0.2308, -0.2473, -0.1978, -0.1429

2. window = 3  

  **Corr = 0.126**
  
  Évolution sur 10 epochs : '0.159', '0.258', '0.225', '0.104', '0.104', '0.082', '0.082', '0.082', '0.082', '0.126'
  
3. window = 4  

  **Corr = -0.275**
  
  Évolution sur 10 epochs : '-0.440', '-0.374', '-0.352', '-0.264', '-0.291', '-0.297', '-0.280', '-0.275', '-0.286', '-0.275'
  
## 2. Variation du learning_rate, window fixé (=3)


