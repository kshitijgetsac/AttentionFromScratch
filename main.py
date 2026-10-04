"""Small NumPy decoder blocks with explicit backward passes.

Inputs are (sequence_length, model_dim) arrays. Each forward call caches
one sequence for its subsequent backward call.
"""
import numpy as np


def matmul(left, right):
    """2-D product without the macOS Accelerate matmul warning issue."""
    return np.einsum('ij,jk->ik', left, right, optimize=False)


class ConvertToEmbeddings:
    """Legacy one-hot demo encoder; training uses its own fixed vocabulary."""
    def __init__(self, positionalEmbeddings=None):
        self.positionalEmbeddings = positionalEmbeddings

    def embedText(self, text):
        ids = [ord(char) - ord('a') for char in text if 'a' <= char <= 'z']
        return np.eye(26)[ids]


class LayerNorm:
    def __init__(self, model_dim, eps=1e-5):
        self.gamma = np.ones(model_dim)
        self.beta = np.zeros(model_dim)
        self.eps = eps
        self.params = {'gamma': self.gamma, 'beta': self.beta}
        self.grads = {name: np.zeros_like(value) for name, value in self.params.items()}

    def forward(self, x):
        centered = x - x.mean(axis=-1, keepdims=True)
        self.inv_std = 1.0 / np.sqrt((centered**2).mean(axis=-1, keepdims=True) + self.eps)
        self.normalized = centered * self.inv_std
        return self.normalized * self.gamma + self.beta

    def backward(self, grad_output):
        self.grads['gamma'] += (grad_output * self.normalized).sum(axis=0)
        self.grads['beta'] += grad_output.sum(axis=0)
        grad = grad_output * self.gamma
        return self.inv_std * (
            grad - grad.mean(axis=-1, keepdims=True)
            - self.normalized * (grad * self.normalized).mean(axis=-1, keepdims=True)
        )


class MLP:
    def __init__(self, rows=None, cols=None, bias=True, rng=None):
        # rows remains accepted; parameters are shared across positions.
        if cols is None or cols <= 0:
            raise ValueError('cols must be a positive embedding width')
        rng = rng if rng is not None else np.random.default_rng()
        self.cols = cols
        self.weightMatrix = rng.normal(0.0, 0.02, (cols, 4 * cols))
        self.weighDownMat = rng.normal(0.0, 0.02, (4 * cols, cols))
        self.biasMatrix = np.zeros(4 * cols) if bias else None
        self.downBias = np.zeros(cols) if bias else None
        self.params = {'up_weight': self.weightMatrix, 'down_weight': self.weighDownMat}
        if bias:
            self.params.update(up_bias=self.biasMatrix, down_bias=self.downBias)
        self.grads = {name: np.zeros_like(value) for name, value in self.params.items()}

    @staticmethod
    def gelu(x):
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    @staticmethod
    def gelu_derivative(x):
        scale = np.sqrt(2.0 / np.pi)
        tanh = np.tanh(scale * (x + 0.044715 * x**3))
        return (0.5 * (1.0 + tanh)
                + 0.5 * x * (1.0 - tanh**2) * scale * (1.0 + 3.0 * 0.044715 * x**2))

    def forward(self, context):
        self.input = context
        self.hidden = matmul(context, self.weightMatrix)
        if self.biasMatrix is not None:
            self.hidden = self.hidden + self.biasMatrix
        self.computedScaledAttention = self.gelu(self.hidden)
        out = matmul(self.computedScaledAttention, self.weighDownMat)
        if self.downBias is not None:
            out = out + self.downBias
        return out

    def backward(self, grad_output):
        self.grads['down_weight'] += matmul(self.computedScaledAttention.T, grad_output)
        grad_hidden = matmul(grad_output, self.weighDownMat.T) * self.gelu_derivative(self.hidden)
        self.grads['up_weight'] += matmul(self.input.T, grad_hidden)
        if self.biasMatrix is not None:
            self.grads['down_bias'] += grad_output.sum(axis=0)
            self.grads['up_bias'] += grad_hidden.sum(axis=0)
        return matmul(grad_hidden, self.weightMatrix.T)


class ComputeAttention:
    """One causal attention head, including an output projection."""
    def __init__(self, model_dim, key_dim=None, rng=None):
        self.model_dim = model_dim
        self.key_dim = model_dim if key_dim is None else key_dim
        if model_dim <= 0 or self.key_dim <= 0:
            raise ValueError('attention dimensions must be positive')
        rng = rng if rng is not None else np.random.default_rng()
        self.Q = rng.normal(0.0, 0.02, (model_dim, self.key_dim))
        self.K = rng.normal(0.0, 0.02, (model_dim, self.key_dim))
        self.V = rng.normal(0.0, 0.02, (model_dim, self.key_dim))
        self.W_out = rng.normal(0.0, 0.02, (self.key_dim, model_dim))
        self.params = {'Q': self.Q, 'K': self.K, 'V': self.V, 'out_weight': self.W_out}
        self.grads = {name: np.zeros_like(value) for name, value in self.params.items()}

    def calculateExponential(self, attentionScores):
        shifted = attentionScores - attentionScores.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=-1, keepdims=True)

    def computeAttention(self, wordEmbeddings):
        self.input = wordEmbeddings
        self.query = matmul(wordEmbeddings, self.Q)
        self.key = matmul(wordEmbeddings, self.K)
        self.value = matmul(wordEmbeddings, self.V)
        scores = matmul(self.query, self.key.T) / np.sqrt(self.key_dim)
        self.mask = np.tril(np.ones(scores.shape, dtype=bool))
        self.weights = self.calculateExponential(np.where(self.mask, scores, -np.inf))
        self.context = matmul(self.weights, self.value)
        return matmul(self.context, self.W_out)

    def forward(self, x):
        return self.computeAttention(x)

    def backward(self, grad_output):
        self.grads['out_weight'] += matmul(self.context.T, grad_output)
        grad_context = matmul(grad_output, self.W_out.T)
        grad_weights = matmul(grad_context, self.value.T)
        grad_value = matmul(self.weights.T, grad_context)
        grad_scores = self.weights * (
            grad_weights - (grad_weights * self.weights).sum(axis=-1, keepdims=True)
        )
        grad_scores = np.where(self.mask, grad_scores, 0.0) / np.sqrt(self.key_dim)
        grad_query = matmul(grad_scores, self.key)
        grad_key = matmul(grad_scores.T, self.query)
        self.grads['Q'] += matmul(self.input.T, grad_query)
        self.grads['K'] += matmul(self.input.T, grad_key)
        self.grads['V'] += matmul(self.input.T, grad_value)
        return (matmul(grad_query, self.Q.T) + matmul(grad_key, self.K.T)
                + matmul(grad_value, self.V.T))


if __name__ == '__main__':
    text = 'the cat sat on the mat'
    rng = np.random.default_rng(42)
    embeddedText = ConvertToEmbeddings().embedText(text)
    print(embeddedText)
    rows, cols = embeddedText.shape
    print(embeddedText.shape)
    attention = ComputeAttention(cols, rng=rng)
    mlp = MLP(rows, cols, bias=True, rng=rng)
    x = embeddedText + attention.forward(LayerNorm(cols).forward(embeddedText))
    x = x + mlp.forward(LayerNorm(cols).forward(x))
    output = LayerNorm(cols).forward(x)
    vocabulary = 'abcdefghijklmnopqrstuvwxyz'
    W_vocab = rng.normal(0.0, 0.02, (cols, len(vocabulary)))
    logits = matmul(output, W_vocab)
    next_character = vocabulary[int(np.argmax(logits[-1]))]
    print('Output shape:', output.shape)
    print('Next character (untrained):', next_character)
    print('Updated text:', text + next_character)
