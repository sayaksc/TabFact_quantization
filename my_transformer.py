import numpy as np
import torch
import torch.nn as nn
import math




def quantize(x,m_max=8, e_max=8):

    x = x.clone()

    m, e = torch.frexp(x)

    m = torch.round(m * (2**(m_max-1))) / (2**(m_max-1))


    quantized_x = m * (2.0 ** e)

    quantized_x[e < -(e_max-1)] = 0.0
    quantized_x[e > (e_max-1)] = (1-2**(-m_max+1)) * 2**(e_max-1) * torch.sign(quantized_x[e > (e_max-1)])

    return quantized_x





class MultiLayerTransformer(nn.Module):
    def __init__(self, vocab_size_in, vocab_size_out=None, d_model=512, n_heads=8, d_ff=2048, num_layers=1,  dropout=0.1, dtype=torch.float32, device='cpu'):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.dtype = dtype
        self.device = device
        self.var = 0
        self.vocab_size_in = vocab_size_in
        
        if self.d_ff is None:
            self.d_ff = 4 * d_model

        if vocab_size_out is None:
            vocab_size_out = vocab_size_in
        self.vocab_size_out = vocab_size_out

        self.embedding = nn.Embedding(vocab_size_in, d_model, dtype=dtype)

        self.LM = nn.Linear(d_model, vocab_size_out, dtype=dtype)

        self.layers = nn.ModuleList([TransformerLayer(d_model, n_heads, d_ff, dropout, dtype) for _ in range(num_layers)])


        self.to(device)




    
    def pos_encoding(self, X, B, n):

        pe = torch.zeros(n, self.d_model, dtype=self.dtype, device=self.device)
        
        position = torch.arange(n, dtype=self.dtype, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=self.dtype, device=self.device) * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)

        if self.d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        return pe.unsqueeze(0).expand(B, -1, -1)

    
    def forward(self, X, quantization=False, m_max=None, e_max=None):
        # X is already on the correct device from generate_EQ
        # Just ensure it's long type for embedding lookup
        if quantization:
            if X.dtype != torch.long:
                X = X.long()

            B, n = X.shape   # B, N

            self.embedding = nn.Embedding(self.vocab_size_in, self.d_model, dtype=self.dtype).to(self.device).from_pretrained(quantize(self.embedding.weight, m_max, e_max), freeze=False)
            X = self.embedding(X) 
            X = quantize(X, m_max, e_max)
            X = quantize(X + quantize(self.pos_encoding(X, B, n), m_max, e_max), m_max, e_max)


            for layer in self.layers:
                X = layer(X, quantization=quantization, m_max=m_max, e_max=e_max)
                X = quantize(X, m_max, e_max)


            self.LM.weight.copy_(quantize(self.LM.weight, m_max, e_max))
            
            self.LM.bias.copy_(quantize(self.LM.bias, m_max, e_max))

            X = quantize(self.LM(X), m_max, e_max)
            self.var = self.embedding.weight
            return X

        
        if X.dtype != torch.long:
            X = X.long()

        B, n = X.shape   # B, N
        X = self.embedding(X) 
        X = X + self.pos_encoding(X, B, n)

        for layer in self.layers:
            X = layer(X, quantization=False)
        
        X = self.LM(X)
        self.var = self.embedding.weight
        return X
    

    def convert(self, device):
        self.embedding = self.embedding.to(device=device)
        self.LM = self.LM.to(device=device)
        self.device = device

        for layer in self.layers:
            layer.convert(device)

    def update_dtype_device(self, dtype=None, device=None):
        """Update the dtype and device attributes after model conversion"""
        if dtype is not None:
            self.dtype = dtype
            for layer in self.layers:
                layer.dtype = dtype
        if device is not None:
            self.device = device
            for layer in self.layers:
                layer.device = device



class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, dtype=torch.float32, bias=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_h = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        self.dtype = dtype

        self.layer_norm1 = nn.LayerNorm(self.d_model, dtype=self.dtype)
        self.layer_norm2 = nn.LayerNorm(self.d_model, dtype=self.dtype)


        self.WQ = nn.Linear(d_model, d_model, dtype=self.dtype, bias=bias)
        self.WK = nn.Linear(d_model, d_model, dtype=self.dtype, bias=bias)
        self.WV = nn.Linear(d_model, d_model, dtype=self.dtype, bias=bias)

        self.Wh = nn.Linear(self.d_model, self.d_model, dtype=self.dtype, bias=bias)

        self.MLP1 = nn.Linear(self.d_model, d_ff, dtype=self.dtype)
        self.MLP2 = nn.Linear(d_ff, self.d_model, dtype=self.dtype)

        self.var = 0

        # self.MLP = nn.Sequential(
        #     nn.Linear(self.d_model, d_ff, dtype=self.dtype),
        #     nn.ReLU(),
        #     self.dropout,
        #     nn.Linear(d_ff, self.d_model, dtype=self.dtype)
        # )


    def self_attention(self, Q, K, V, mask=None, quantization=False, m_max=None, e_max=None):

        if quantization:
            QK = Q @ K.transpose(-2, -1) / math.sqrt(self.d_h)
            QK = quantize(QK, m_max, e_max)


            if mask is not None:
                if self.dtype == torch.float16:
                    QK = QK.masked_fill(mask==0, -1e4)
                else:
                    QK = QK.masked_fill(mask==0, -1e9)
            A = torch.softmax(QK, dtype=self.dtype, dim=-1)
            A = quantize(A, m_max, e_max)

            if False in torch.isfinite(A):
                print("WARNING: Softmax output contains non-finite values")
                print("="*100)

            A = self.dropout(A)
            self.var = QK
            return quantize(A @ V, m_max, e_max)

        QK = Q @ K.transpose(-2, -1) / math.sqrt(self.d_h)


        if mask is not None:
            if self.dtype == torch.float16:
                QK = QK.masked_fill(mask==0, -1e4)
            else:
                QK = QK.masked_fill(mask==0, -1e9)
        A = torch.softmax(QK, dtype=self.dtype, dim=-1)

        if False in torch.isfinite(A):
            print("WARNING: Softmax output contains non-finite values")
            print("="*100)

        A = self.dropout(A)
        self.var = QK
        return A @ V


    def MLP(self, X, quantization=False, m_max=None, e_max=None):
        if quantization:
            out = self.MLP1(X)
            out = quantize(out, m_max, e_max)
            out = nn.ReLU()(out)
            out = self.dropout(out)
            out = self.MLP2(out)
            out = quantize(out, m_max, e_max)
            return out
        out = self.MLP1(X)
        out = nn.ReLU()(out)
        out = self.dropout(out)
        out = self.MLP2(out)
        return out


    def forward(self, X, mask=None, quantization=False, m_max=None, e_max=None):
        B, N, _ = X.shape


        if quantization:
            self.WQ.weight.copy_(quantize(self.WQ.weight, m_max, e_max))
            if self.WQ.bias is not None:
                self.WQ.bias.copy_(quantize(self.WQ.bias, m_max, e_max))
            self.WK.weight.copy_(quantize(self.WK.weight, m_max, e_max))
            if self.WK.bias is not None:
                self.WK.bias.copy_(quantize(self.WK.bias, m_max, e_max))
            self.WV.weight.copy_(quantize(self.WV.weight, m_max, e_max))
            if self.WV.bias is not None:
                self.WV.bias.copy_(quantize(self.WV.bias, m_max, e_max))
            self.Wh.weight.copy_(quantize(self.Wh.weight, m_max, e_max))
            if self.Wh.bias is not None:
                self.Wh.bias.copy_(quantize(self.Wh.bias, m_max, e_max))
            self.MLP1.weight.copy_(quantize(self.MLP1.weight, m_max, e_max))
            if self.MLP1.bias is not None:
                self.MLP1.bias.copy_(quantize(self.MLP1.bias, m_max, e_max))
            self.MLP2.weight.copy_(quantize(self.MLP2.weight, m_max, e_max))
            if self.MLP2.bias is not None:
                self.MLP2.bias.copy_(quantize(self.MLP2.bias, m_max, e_max))

            Q = self.WQ(X) # B, N, d
            K = self.WK(X) # B, N, d
            V = self.WV(X) # B, N, d

            Q = quantize(Q, m_max, e_max)
            K = quantize(K, m_max, e_max)
            V = quantize(V, m_max, e_max)

            Q = Q.view(B, N, self.n_heads, self.d_h).permute(0, 2, 1, 3) # B, H, N, d
            K = K.view(B, N, self.n_heads, self.d_h).permute(0, 2, 1, 3)
            V = V.view(B, N, self.n_heads, self.d_h).permute(0, 2, 1, 3)



            SA = self.self_attention(Q, K, V, mask, quantization, m_max, e_max) #B, H, N, d


            SA = SA.permute(0, 2, 1, 3).reshape((B, N, self.d_model))

            if self.Wh is not None:
                SA = quantize(self.Wh(SA), m_max, e_max) # (B, n, d_model)

            SA = self.dropout(SA)
            SA = self.layer_norm1(X + SA)
            SA = quantize(SA, m_max, e_max)

            MLP_output = self.MLP(SA, quantization=True, m_max=m_max, e_max=e_max)

            return quantize(self.layer_norm2(MLP_output + SA), m_max=m_max, e_max=e_max)

        Q = self.WQ(X) # B, N, d
        K = self.WK(X) # B, N, d
        V = self.WV(X) # B, N, d

        Q = Q.view(B, N, self.n_heads, self.d_h).permute(0, 2, 1, 3) # B, H, N, d
        K = K.view(B, N, self.n_heads, self.d_h).permute(0, 2, 1, 3)
        V = V.view(B, N, self.n_heads, self.d_h).permute(0, 2, 1, 3)



        SA = self.self_attention(Q, K, V, mask) #B, H, N, d

        SA = SA.permute(0, 2, 1, 3).reshape((B, N, self.d_model))

        if self.Wh is not None:
            SA = self.Wh(SA) # (B, n, d_model)
            
        SA = self.dropout(SA)
        SA = self.layer_norm1(X + SA)

        MLP_output = self.MLP(SA)

        return self.layer_norm2(MLP_output + SA)



    

    def convert(self, device):
        self.layer_norm1 = self.layer_norm1.to(device=device)
        self.layer_norm2 = self.layer_norm2.to(device=device)

        self.WQ = self.WQ.to(device=device)
        self.WK = self.WK.to(device=device)
        self.WV = self.WV.to(device=device)
        self.Wh = self.Wh.to(device=device)

        self.MLP1 = self.MLP1.to(device=device)
        self.MLP2 = self.MLP2.to(device=device)

        self.device = device
