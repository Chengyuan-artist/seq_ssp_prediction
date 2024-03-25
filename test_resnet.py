from random import shuffle
from torch.utils.data import DataLoader
import torch
from torch import nn
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torchsummary import summary

from src.dataset import ProteinDataset
from src.resnet1d import ResNet1D


writer = SummaryWriter("./log")

split_len = 20
batch_size = 32
protein_seqs = ProteinDataset.load_data("assignment1_data")

split = int(0.8 * len(protein_seqs))
shuffle(protein_seqs)

test_data = ProteinDataset(
    protein_seqs[:split], train=True, split_len=split_len, transform=True
)
valid_data = ProteinDataset(protein_seqs[split:], train=False)

train_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)

print(test_data.seqs.shape)
print(test_data.ssps.shape)

device_str = "cuda"
device = torch.device(device_str if torch.cuda.is_available() else "cpu")
kernel_size = 16
stride = 2
n_block = 48
downsample_gap = 6
increasefilter_gap = 12
model = ResNet1D(
    in_channels=20,
    base_filters=64,  # 64 for ResNet1D, 352 for ResNeXt1D
    kernel_size=kernel_size,
    stride=stride,
    groups=32,
    n_block=n_block,
    n_classes=3 * split_len,
    downsample_gap=downsample_gap,
    increasefilter_gap=increasefilter_gap,
    use_bn=False,
    use_do=True,
)
model.to(device)

summary(model, (test_data.seqs.shape[1], test_data.seqs.shape[2]), device=device_str)
# exit()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=10
)
loss_func = torch.nn.CrossEntropyLoss()


is_debug = False
n_epoch = 30
step = 0
for epoch in tqdm(range(n_epoch), desc="epoch", leave=False):

    # train
    model.train()
    prog_iter = tqdm(train_loader, desc="Training", leave=False)
    current_loss = 0.0
    cnt = 0
    for batch_idx, batch in enumerate(prog_iter):

        input_x, input_y = tuple(t.to(device) for t in batch)
        pred = model(input_x)
        loss = loss_func(pred, input_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
        current_loss += loss.item()
        cnt += 1

        writer.add_scalar("Loss/step", loss.item(), step)

        if is_debug:
            break
        
    writer.add_scalar("Loss/epoch", current_loss / cnt , epoch)
    scheduler.step(epoch)
    
save_file = f"./model_dict/resnet1d_{n_block}_{split_len}.pth"
print("==> Saving...")
state = {
    "model": model.state_dict()
}
torch.save(state, save_file)
del state
print("Saved")


def cal_q3(ssp_true, ssp_predicted):
    q3 = sum(
        [1 for i in range(len(ssp_true)) if ssp_true[i] == ssp_predicted[i]]
    ) / len(ssp_true)
    return q3


# valid
verbose = False
model.eval()
prog_iter_test = tqdm(valid_loader, desc="Validing", leave=False)
with torch.no_grad():
    q3s = []
    for batch_idx, batch in enumerate(prog_iter_test):
        inputs, targets = tuple(t for t in batch)

        for seq, ssp in zip(inputs, targets):
            split_seq = ProteinDataset.split_seq(seq, split_len)
            seq_tensors = [ProteinDataset.seq_to_onehot(seq, reqular_len=split_len).T for seq in split_seq]
            input_tensor = torch.stack(seq_tensors)
            input_tensor = input_tensor.to(device_str)
            # print(input_tensor.shape)
            output_tensor = model(input_tensor)
            # print(output_tensor.shape)

            one_hot = ProteinDataset.prob_to_onehot(output_tensor.view(-1, 3))
            # print(one_hot.shape)
            ssp_predicted = ProteinDataset.onehot_to_sequence(one_hot, is_ssp=True)
            if verbose:
                print(f"True_len: {len(ssp)}")
                print(f"Pred_len: {len(ssp_predicted)}")
                print(f"True: {ssp}")
                print(f"Pred: {ssp_predicted}")

            q3 = cal_q3(ssp, ssp_predicted)
            q3s.append(q3)
            if verbose:
                print(f"Q3: {q3}")


from sklearn.metrics import mean_squared_error

print(f"Valid data len: {len(q3s)}")
print(f"Average q3 accuracy: {100 * sum(q3s) / len(q3s)}%")
mse = mean_squared_error(q3s, [sum(q3s) / len(q3s)] * len(q3s))
print(f"Mean squared error: {100 * mse}%")
