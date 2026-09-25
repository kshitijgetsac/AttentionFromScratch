import numpy as np
import math

'''
should accept the entire word that we are trying to tokenize and return the np.array for that particular word
'''
class ConvertToEmbeddings:
    def __init__(self,positionalEmbeddings=None):
        self.positionalEmbeddings = positionalEmbeddings
    def embedText(self,text:str)->np.array:
        n = len(text)
        arr = []
        for char in text:
        #possibilites are a-z, A-Z and 0-9, not taking any special characters here keeping v1 simple 
        #also omitting spaces as of now because do not know how to handle it yet
            if char.isdigit():
                currentId = ord(char) - ord('0')
            if char == " ":
                continue
            currentId = ord(char) - ord('a')
            arr.append(currentId)
        for idx,embedding in enumerate(arr):
            arr[idx] = idx + embedding
        #add more ways to add positional embeddings here, tbd in the future
        embeddingArray = []
        for num in arr:
            currArray  = [0] * 26
            currArray[num%26] = 1
            embeddingArray.append(currArray)
        return np.array(embeddingArray)

'''
for now keeping the attention block simple with q,k and v matrices equal to dimension of text
will project down the dimensions as well
'''
class MLP:
    def __init__(self,rows,cols,bias=None):
        self.rows = rows
        self.cols = cols
        self.weightMatrix = np.random.rand(rows,cols)
        if bias:
            self.biasMatrix = np.random.rand(1,cols)
        else:
            self.biasMatrix = None
        self.computedScaledAttention = None
    def forward(self,computedDotAttentionMat,OriginalMatafterLayerNorm1):
        self.computedScaledAttention = computedAttentionMat @ self.weightMatrix
        if self.biasMatrix:
            self.computedScaledAttention += self.biasMatrix
        #question is how to add OriginalMatrix here since we have increased the size of our Matrix during feed forward step
        return self.computedScaledAttention
    
        
    
class ComputeAttention:
    def __init__(self,model_dim,key_dim=None):
        # The projection dimensions depend on the embedding width, not the
        # number of tokens in the current input.
        self.model_dim = model_dim
        self.key_dim = key_dim or model_dim
        self.Q = np.random.rand(model_dim,self.key_dim)
        self.K = np.random.rand(model_dim,self.key_dim)
        self.V = np.random.rand(model_dim,self.key_dim)
    def calculateExponential(self,attentionScores):
        attentionScoreExp = []
        rowSum = []
        for idx,row in enumerate(attentionScores,0):
            sum_xp = 0
            for val in row:
                sum_xp += np.exp(val)
            rowSum.append(sum_xp)
        
        for i,row in enumerate(attentionScores,0):
            currRow = []
            for val in row:
                if val == -np.inf:
                    currRow.append(0.0)
                else:
                    currRow.append(np.exp(val)/rowSum[i])
            currRow = np.array(currRow,dtype=np.float32)
            attentionScoreExp.append(currRow)
        return np.array(attentionScoreExp,dtype=np.float32)

    def computeAttention(self,wordEmbeddings):
        computedQuery = wordEmbeddings @ self.Q
        computedKey = wordEmbeddings @ self.K
        computedValue = wordEmbeddings @ self.V
        attentionScores = (computedQuery @ computedKey.T)
        attentionScores = attentionScores / math.sqrt(self.key_dim)
        scores = np.tril(np.ones(attentionScores.shape),k=0)
        attentionScores = np.where(scores, attentionScores,-np.inf)
        AttentionWeights = self.calculateExponential(attentionScores)
        context = AttentionWeights @ computedValue
        return context



if __name__ == "__main__":
    text = "the cat sat on the mat"
    EmbeddingCls = ConvertToEmbeddings()
    embeddedText = EmbeddingCls.embedText(text)
    rows,cols = embeddedText.shape[0],embeddedText.shape[1]
    AttentionCls = ComputeAttention(model_dim=cols)
    ret = AttentionCls.computeAttention(embeddedText)
    print(ret)


    #next step is attention




        
