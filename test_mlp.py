from src.mlp import MLP
from src.dataset import ProteinDataset
from random import shuffle
from torchsummary import summary

# total_len = 0
# min_len = 100000
# max_len = 0
# len_list = []
# for seq in dataset.protein_seqs:
#     len_list.append(len(seq['seq']))
#     total_len += len(seq['seq'])
#     min_len = min(min_len, len(seq['seq']))
#     max_len = max(max_len, len(seq['seq']))

# print(total_len/len(dataset.protein_seqs))
# print(min_len)
# print(max_len)

# plt.hist(len_list, bins=10)
epochs, batch_size, learning_rate, hidden, split_len = 100, 32, 1e-4, 32, 20
save_file = (
    f"./model_dict/mlp_{batch_size}_{learning_rate}_{epochs}_{hidden}_{split_len}.pth"
)


protein_seqs = ProteinDataset.load_data("assignment1_data")
print(len(protein_seqs))

split = int(0.8 * len(protein_seqs))
shuffle(protein_seqs)

test_data = ProteinDataset(protein_seqs[:split], train=True, split_len=split_len)
valid_data = ProteinDataset(protein_seqs[split:], train=False)
print(test_data.seqs.shape)
print(test_data.ssps.shape)

mlp = MLP(
    epochs=epochs,
    batch_size=batch_size,
    learning_rate=learning_rate,
    hidden=hidden,
    split_len=split_len,
)
summary(mlp, (test_data.seqs.shape[0], test_data.seqs.shape[1]), device="cpu")
# exit()

mlp.load_data(test_data, valid_data)
mlp.train_model(verbose=True)
mlp.save_res(save_file)
q3s = mlp.valid(verbose=True)


from sklearn.metrics import mean_squared_error

print(f"Valid data len: {len(q3s)}")
print(f"Average q3 accuracy: {100 * sum(q3s) / len(q3s)}%")
mse = mean_squared_error(q3s, [sum(q3s) / len(q3s)] * len(q3s))
print(f"Mean squared error: {100 * mse}%")

# plt.plot(q3s)

import pandas as pd
import matplotlib.pyplot as plt


# Read the CSV file
df = pd.read_csv(f"{save_file}_loss.csv")
df.plot(x="epoch", y="loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.show()
