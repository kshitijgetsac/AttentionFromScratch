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
        self.cols = cols
        self.weightMatrix = np.random.rand(cols,4*cols)
        if bias:
            self.biasMatrix = np.random.rand(rows,4*cols)
        else:
            self.biasMatrix = None
        self.weighDownMat = np.random.rand(4*cols,cols)
        # self.computedScaledAttention = None
    def layer_norm(self,x, gamma, beta, eps=1e-5):
      mean = x.mean(axis=-1, keepdims=True)
      var = x.var(axis=-1, keepdims=True)  # population variance
      return (x - mean) / np.sqrt(var + eps)
    def gelu(self, x):
        return 0.5 * x * (
            1.0 + np.tanh(
                np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
            )
        )
    def forward(self,context):
        context = self.layer_norm(context,0,0)
        print(context.shape,self.weightMatrix.shape,self.biasMatrix.shape)
        self.computedScaledAttention = context @ self.weightMatrix
        print(self.computedScaledAttention.shape)
        if self.biasMatrix is not None:
            self.computedScaledAttention += self.biasMatrix
        self.computedScaledAttention = self.gelu(
            self.computedScaledAttention
        )
        #question is how to add OriginalMatrix here since we have increased the size of our Matrix during feed forward step
        #next is to bring it down to vocab size
        out = self.computedScaledAttention @ self.weighDownMat
        return out
    
        
    
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
    LinearMlp = MLP(ret.shape[0],ret.shape[1],bias=True)
    output = LinearMlp.forward(ret)
    # Token ID 0 means "a", 1 means "b", etc.
    vocabulary = "abcdefghijklmnopqrstuvwxyz"
    vocab_size = len(vocabulary)
    model_dim = output.shape[-1]
    rng = np.random.default_rng(42)
    W_vocab = rng.normal(
        loc=0.0,
        scale=0.02,
        size=(model_dim, vocab_size),
    )
    logits = output @ W_vocab
    next_character_scores = logits[-1]
    next_character_id = int(np.argmax(next_character_scores))
    next_character = vocabulary[next_character_id]

    print("Next character:", next_character)
    print("Updated text:", text + next_character)







        
