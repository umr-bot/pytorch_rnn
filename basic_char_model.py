import torch
from torch import nn 
from model import MyRNN
from utils import CharacterTable, transform_3, batch, datagen_simple, get_device
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Baseline models training script")
parser.add_argument("--hidden_size", default=128, help="hidden layer size")
parser.add_argument("--dropout", default=0.2, help="dropout regularization rate")
parser.add_argument("--num_epochs", default=10 , help="hidden layer1 size")
parser.add_argument("--data_dir",help="path to unigrams")
parser.add_argument("--train_batch_size", default=128, help="train batch size")
parser.add_argument("--val_batch_size", default=256, help="val batch size")
parser.add_argument("--foldset_num", default=1, help="foldset number to use")
parser.add_argument("--checkpoints_dir", default="eng_checkpoints", help="directory in which to save checkpoints")
parser.add_argument("--history_fn", default="history.txt", help="file name to save model history in")
parser.add_argument("--model", help="model type to train with")
parser.add_argument("--model_num", default=0, help="Model to start training from")

args = parser.parse_args()
args.num_epochs, args.train_batch_size, args.val_batch_size, args.dropout = int(args.num_epochs), int(args.train_batch_size), int(args.val_batch_size), float(args.dropout)

with open(args.data_dir+"/foldset"+str(args.foldset_num)+"/train") as f:
    train_tups = [line.strip('\n').split(',') for line in f]
train_dec_tokens, train_tokens = zip(*train_tups)

with open(args.data_dir+"/foldset"+str(args.foldset_num)+"/val") as f:
    val_tups = [line.strip('\n').split(',') for line in f]
val_dec_tokens, val_tokens = zip(*val_tups)
def make_windows(text):
    # Padding
    text = list(text).copy() # convert tuple to list in case input not list
    # Creating lists that will hold our input and target sequences
    input_seq = []
    target_seq = []

    for i in range(len(text)):
        # Remove last character for input sequence
        input_seq.append(text[i][:-1])
        # Remove first character for target sequence
        target_seq.append(text[i][1:])
    return input_seq, target_seq
train_tokens, train_dec_tokens = make_windows(train_dec_tokens)
val_tokens, val_dec_tokens = make_windows(val_dec_tokens)

input_chars = set(' '.join(train_tokens) + '*' + '\t') # * and \t are EOS and SOS respectively
target_chars = set(' '.join(train_dec_tokens) + '*' + '\t')
total_chars = input_chars.union(target_chars)
nb_total_chars = len(total_chars)
total_ctable = input_ctable = target_ctable = CharacterTable(total_chars)
maxlen = max([len(token) for token in train_tokens]) + 2

train_steps = len(train_tokens) // args.num_epochs
val_steps = len(val_tokens) // args.num_epochs

from model import MyRNN
model = MyRNN(input_size=total_ctable.size, output_size=total_ctable.size, hidden_dim=10, n_layers=1)
n_epochs = 100; lr=0.01
# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

device = get_device()

start=int(args.model_num)
for epoch in range(start,args.num_epochs):
    print(f"Epoch {str(epoch+1)}")
    st_ind,et_ind = int(len(train_tokens)*(epoch/args.num_epochs)), int(len(train_tokens)*((epoch+1)/args.num_epochs) )
    sv_ind,ev_ind = int(len(val_tokens)*(epoch/args.num_epochs)), int(len(val_tokens)*((epoch+1)/args.num_epochs))

    train_x_padded=transform_3(train_tokens[st_ind:et_ind], maxlen=maxlen)
    train_y_padded=transform_3(train_dec_tokens[st_ind:et_ind], maxlen=maxlen)

    val_x_padded=transform_3(val_tokens[sv_ind:ev_ind], maxlen=maxlen)
    val_y_padded=transform_3(val_dec_tokens[sv_ind:ev_ind], maxlen=maxlen)

    train_X_iter = batch(train_x_padded,maxlen=maxlen,ctable=total_ctable,batch_size=args.train_batch_size,reverse=False)
    train_y_iter = batch(train_y_padded,maxlen=maxlen,ctable=total_ctable,batch_size=args.train_batch_size,reverse=False)
    val_X_iter = batch(val_x_padded,maxlen=maxlen,ctable=total_ctable,batch_size=args.train_batch_size,reverse=False)
    val_y_iter = batch(val_y_padded,maxlen=maxlen,ctable=total_ctable,batch_size=args.train_batch_size,reverse=False)

    train_loader = datagen_simple(train_X_iter, train_y_iter)
    val_loader = datagen_simple(val_X_iter, val_y_iter)

#    optimizer.zero_grad() # Clears existing gradients from previous epoch
#    for step in tqdm(range(100)):
#        x, x_tar = next(train_loader)
#        input_seq=torch.Tensor(x)
#        input_seq.to(device)
#        output, hidden = model(x)
#        loss = criterion(output, x_tar)
#        loss.backward() # Does backpropagation and calculates gradients
#        optimizer.step() # Updates the weights accordingly
#
#    #if epoch%10 == 0:
#    print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
#    print("Loss: {:.4f}".format(loss.item()))


