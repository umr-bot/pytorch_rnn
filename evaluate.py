import torch
from torch import nn
import numpy as np
from utils import get_device

def  predict(model, ctable, character):
    character = ctable.encode(character,nb_rows=1)
    character = torch.from_numpy(character[np.newaxis,...]) # make batch of size 1
    device = get_device()
    character.to(device)

    out, hidden = model(character)
    prob = nn.functional.softmax(out[-1], dim=0).data
    text_out = ctable.decode(prob.detach().numpy())[1]
    return text_out, hidden
def sample(model, ctable, out_len, start='hey'):
    #out_len=5; start = "tho"
    model.eval() # eval mode
    start = start.lower()
    # First off, run through the starting characters
    chars = [ch for ch in start]
    size = out_len - len(chars)
    # Now pass in the previous characters and get a new one
    for ii in range(size):
        char, h = predict(model, ctable, chars)
        chars.append(char)

    return ''.join(chars)

model = torch.load("v1.pb")
#sample(model, ctable=total_ctable, out_len=5, start="tho")

