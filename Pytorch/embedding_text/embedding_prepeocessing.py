import numpy as np

# hyper params
n_word2embed = 4
text_path = "./article.txt"
text_file_name = "article"



fp = open(text_path, 'r')
text = fp.read().replace(",", " , ").replace(".", " . ")
fp.close()
text = text.split()

pick_len = len(text) - n_word2embed

train_words = []
label_words = []
for mid in range(n_word2embed//2, n_word2embed//2+pick_len-1):
    train_words.append([text[mid-1], text[mid-2], text[mid+1], text[mid+2]])
    label_words.append(text[mid])



word_set = set(text)
word_idx_dict = { word:idx for idx, word in enumerate(word_set) }
idx_word_dict = { idx:word for (word, idx) in word_idx_dict.items()}
print(word_idx_dict)
print(idx_word_dict)


train_xs = []
train_ys = []

for word_group, label in zip(train_words, label_words):
    train_xs.append([word_idx_dict[w] for w in word_group])
    train_ys.append(word_idx_dict[label])


train_xs = np.array(train_xs)
train_ys = np.array(train_ys)


np.save("./npy-format-data/"+text_file_name+"_feature.npy", train_xs)
np.save("./npy-format-data/"+text_file_name+"_label.npy", train_ys)

fp = open('./word2idx_dict.txt', 'w')
fp.write(str(word_idx_dict))
fp.close()

fp = open('./idx2word_dict.txt', 'w')
fp.write(str(idx_word_dict))
fp.close()
