import numpy as np

'''
should accept the entire word that we are trying to tokenize and return the np.array for that particular word
'''
class ConvertToEmbeddings:
    def __init__(self):
        return None
    def embedText(self,text:str)->np.array:
        n = len(text)
        arr = []
        for char in text:
        #possibilites are a-z, A-Z and 0-9, not taking any special characters here keeping v1 simple
            if isDigit(char):
                currentId = ord(char) - ord('0')
            currentId = ord(char) - ord('a')
            arr.append(currentId)
         for idx,embedding in enumerate(arr):
        arr[idx] = idx + embedding
        return np.array(arr)




if __name__ == "__main__":
    text = "the cat sat on the mat"
    EmbeddingCls = ConvertToEmbeddings()
    embeddedText = EmbeddingCls.embedText(text)
    



        

