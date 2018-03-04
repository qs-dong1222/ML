import torch as t
from torch.autograd import Variable
import torch.utils.data as utdata
import numpy as np




# hyper params
EPOCH = 500




xs = np.load("./embedding_text/npy-format-data/article_feature.npy")
ys = np.load("./embedding_text/npy-format-data/article_label.npy")
n_word2embed = len(ys)





class Word2Vector(t.nn.Module):
    def __init__(self, n_word2embed, n_embed_dim, n_context_words):
        super(Word2Vector, self).__init__()
        self.m = t.nn.Embedding(n_word2embed, n_embed_dim)
        self.dense = t.nn.Sequential(
            t.nn.Linear(n_context_words*n_embed_dim, 256),
            t.nn.ReLU(),
            t.nn.Linear(256, n_word2embed),
            t.nn.Dropout(p=0.1)
        )

    def forward(self, input):
        embeded = self.m(input)
        embeded = embeded.view(1, -1)
        output = self.dense(embeded)
        return output, self.m






# word2vec = Word2Vector(n_word2embed=n_word2embed, n_embed_dim=100, n_context_words=len(xs[0])).cuda()
#
#
#
# # loss function
# loss_func = t.nn.CrossEntropyLoss().cuda()
#
# # optimizer
# opt = t.optim.Adam(word2vec.parameters(), lr=0.001, weight_decay=1e-5)
#
#
#
#
# word2vec.train()
#
# for epoch in range(EPOCH):
#     for wordContext_idx, label_idx in zip(xs, ys):
#         wordContext_idx = Variable(t.LongTensor(wordContext_idx)).cuda()
#         label_idx = Variable(t.LongTensor([int(label_idx)])).cuda()
#         pred = word2vec(wordContext_idx)[0]
#         loss = loss_func(pred, label_idx)
#         opt.zero_grad()
#         loss.backward()
#         opt.step()
#
#     print(loss)
#
#
#
# t.save(word2vec, "./model-word2vector.pkl")
#
# word2vec.eval()





word2vec = t.load("./model-word2vector.pkl")



fp = open("./embedding_text/word2idx_dict.txt", 'r')
word_idx_dict = eval(fp.read())
fp.close()
fp = open("./embedding_text/idx2word_dict.txt", 'r')
idx_word_dict = eval(fp.read())
fp.close()

# ws = ['there', 'comes', 'new', 'people']
# ws = ['you', 'have', 'away', 'for']
# ws = ['person', 'you', 'to', 'see']
ws = ['day', 'after', 'testified', 'in']

test = Variable(t.LongTensor([word_idx_dict[w] for w in ws])).cuda()

probs = word2vec(test)[0]
wordIdx = t.max(probs, dim=1)[1].data.cpu()[0]
print("word : ", idx_word_dict[wordIdx])

