import re
import os
import unidecode
import numpy as np
import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Layer
#from model_attention import AttentionLayer
from tensorflow.keras import optimizers, metrics, backend as K
#from model import truncated_acc, truncated_loss, recall, precision, f1_score
from tqdm import tqdm
import json

import torch
np.random.seed(1234)

SOS = '\t' # start of sequence.
EOS = '*' # end of sequence.
CHARS = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
BAM_CHARS = list("\-fsé\.lɲhaiyjuŋtɔ*opèxewbçknmvqcrɛzgd")
REMOVE_CHARS = '[#$%"\+@<=>!&,-.?:;()*\[\]^_`{|}~/\d\t\n\r\x0b\x0c]'

def get_device():
    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")
    return device

class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one-hot integer representation
    + Decode the one-hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    """
    def __init__(self, chars):
        """Initialize character table.
        # Arguments
          chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char2index = dict((c, i) for i, c in enumerate(self.chars))
        self.index2char = dict((i, c) for i, c in enumerate(self.chars))
        self.size = len(self.chars)
    
    def encode(self, C, nb_rows):
        """One-hot encode given string C.
        # Arguments
          C: string, to be encoded.
          nb_rows: Number of rows in the returned one-hot encoding. This is
          used to keep the # of rows for each data the same via padding.
        """
        x = np.zeros((nb_rows, len(self.chars)), dtype=np.float32)
        for i, c in enumerate(C):
            try: x[i, self.char2index[c]] = 1.0
            except:
                #print(f"Char not working {C}")
                #x[i, 0] = 1.0
                continue
        return x

    def decode(self, x, calc_argmax=True):
        """Decode the given vector or 2D array to their character output.
        # Arguments
          x: A vector or 2D array of probabilities or one-hot encodings,
          or a vector of character indices (used with `calc_argmax=False`).
          calc_argmax: Whether to find the character index with maximum
          probability, defaults to `True`.
        """
        if calc_argmax:
            indices = x.argmax(axis=-1)
        else:
            indices = x
        chars = ''.join(self.index2char[ind] for ind in indices)
        return indices, chars

    def sample_multinomial(self, preds, temperature=1.0):
        """Sample index and character output from `preds`,
        an array of softmax probabilities with shape (1, 1, nb_chars).
        """
        # Reshaped to 1D array of shape (nb_chars,).
        preds = np.reshape(preds, len(self.chars)).astype(np.float64)
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probs = np.random.multinomial(1, preds, 1)
        index = np.argmax(probs)
        char  = self.index2char[index]
        return index, char

def get_type_lists(list_1,list_2):
    """Get word type lists out of two parralel word token lists"""
    tups = list(zip(list_1,list_2)) # pair tokens in list_1 and list_2 according to indices
    tups_set = list(set(tups)) # filter only unique token tuples
        
    t_set1,t_set2 = [list(t) for t in zip(*tups_set)] # unpack lists in tup_set
    return t_set1,t_set2

def read_text(data_path, list_of_books):
    text = ''
    for book in list_of_books:
        file_path = os.path.join(data_path, book)
        #strings = unidecode.unidecode(open(file_path).read())
        #text += strings + ' '
        with open(file_path,'rb') as f:
            data = [x.decode('utf8') for x in f]
        text += ' '.join(d for d in data)
    return text


def tokenize(text):
    tokens = [re.sub(REMOVE_CHARS, '', token)
              for token in re.split("[-\n ]", text)]
    tokens = [re.sub("'", " ", token) for token in tokens]
    return tokens

    
def add_speling_erors(token, error_rate):
    """Simulate some artificial spelling mistakes."""
    assert(0.0 <= error_rate < 1.0)
    if len(token) < 3:
        return token
    rand = np.random.rand()
    # Here are 4 different ways spelling mistakes can occur,
    # each of which has equal chance.
    prob = error_rate / 4.0
    if rand < prob:
        # Replace a character with a random character.
        random_char_index = np.random.randint(len(token))
        token = token[:random_char_index] + np.random.choice(CHARS) \
                + token[random_char_index + 1:]
    elif prob < rand < prob * 2:
        # Delete a character.
        random_char_index = np.random.randint(len(token))
        token = token[:random_char_index] + token[random_char_index + 1:]
    elif prob * 2 < rand < prob * 3:
        # Add a random character.
        random_char_index = np.random.randint(len(token))
        token = token[:random_char_index] + np.random.choice(CHARS) \
                + token[random_char_index:]
    elif prob * 3 < rand < prob * 4:
        # Transpose 2 characters.
        random_char_index = np.random.randint(len(token) - 1)
        token = token[:random_char_index]  + token[random_char_index + 1] \
                + token[random_char_index] + token[random_char_index + 2:]
    else:
        # No spelling errors.
        pass
    return token

def gen_errorful_tokens(fn_in,fn_out,fp="data/", error_rate=0.8, write_flag="True", return_flag="False"):
    text = read_text(data_path=fp,list_of_books=[fn_in])
    vocab = tokenize(text)
    vocab = list(filter(None,set(vocab)))
    # 'maxlen' is the length of the longest word in the vocabulary
    # plus two SOS and EOS characters.
    #maxlen = max([len(token) for token in vocab]) + 2
    error_tokens = dict()
    for token in vocab:
        error_tokens[token] = add_speling_erors(token=token,error_rate=error_rate)
    if write_flag:
        with open(fn_out,"w") as f:
           f.write(json.dumps(error_tokens, indent=4, sort_keys=True))
    if return_flag: return error_tokens

def gen_errorful_files(fn_in,fn_out, fp_out="error_folder/",start=300, num_files=400):
    for i in tqdm(range(start, num_files)):
        fn_out_ = fp_out + fn_out + '_' + str(i) + ".txt"
        gen_errorful_tokens(fn_in,fn_out_)

def transform(tokens, maxlen, error_rate=0.3, shuffle=True):
    """Transform tokens into model inputs and targets.
    All inputs and targets are padded to maxlen with EOS character.
    """
    if shuffle:
        print('Shuffling data.')
        np.random.shuffle(tokens)
    encoder_tokens = []
    decoder_tokens = []
    target_tokens = []
    for token in tokens:
        encoder = add_speling_erors(token, error_rate=error_rate)
        encoder += EOS * (maxlen - len(encoder)) # Padded to maxlen.
        encoder_tokens.append(encoder)
    
        decoder = SOS + token
        decoder += EOS * (maxlen - len(decoder))
        decoder_tokens.append(decoder)
    
        target = decoder[1:]
        target += EOS * (maxlen - len(target))
        target_tokens.append(target)
        assert(len(encoder) == len(decoder) == len(target))

    return encoder_tokens, decoder_tokens, target_tokens

def transform_3(tokens, maxlen):
    """Transform tokens into model inputs and targets.
    All inputs and targets are padded to maxlen with EOS character.
    """
    encoder_tokens = []
    for token in tokens:
        encoder = token
        encoder += EOS * (maxlen - len(encoder)) # Padded to maxlen.
        encoder_tokens.append(encoder)
    
    return encoder_tokens

def transform2(tokens, maxlen, shuffle=False, dec_tokens=[], chrs=[], reverse=False):
    """Transform tokens into model inputs and targets.
    All inputs and targets are padded to maxlen with EOS character.
    """
    encoder_tokens = []
    decoder_tokens = []
    target_tokens = []
    copy_tokens,copy_dec_tokens=[],[]
    if chrs != []:
        for i in range(len(tokens)):
            tok,dec_tok = tokens[i], dec_tokens[i]
            if set(tok).issubset(chrs) and set(dec_tok).issubset(chrs):
                copy_tokens.append(tok)
                copy_dec_tokens.append(dec_tok)
        tokens, dec_tokens = copy_tokens,copy_dec_tokens

    assert(len(tokens)==len(dec_tokens))
    for i in range(len(tokens)):
        token,dec_token = tokens[i], dec_tokens[i]
        if len(token) > 0: # only deal with tokens longer than length 3
            #encoder = add_speling_erors(token, error_rate=error_rate)
            encoder = token
            encoder += EOS * (maxlen - len(encoder)) # Padded to maxlen.
            if reverse: encoder = encoder[::-1]
            #encoder_tokens.append(encoder)
        
            decoder = SOS + dec_token
            decoder += EOS * (maxlen - len(decoder))
            #decoder_tokens.append(decoder)
        
            target = decoder[1:]
            target += EOS * (maxlen - len(target))
            #target_tokens.append(target)
            if (len(encoder) == len(decoder) == len(target)):
                encoder_tokens.append(encoder)
                decoder_tokens.append(decoder)
                target_tokens.append(target)
            else: continue

    return encoder_tokens, decoder_tokens, target_tokens


def my_transform(tokens, maxlen, error_tokens, shuffle=False,reverse=False):
    """Transform tokens into model inputs and targets.
    All inputs and targets are padded to maxlen with EOS character.
    """
    if shuffle:
        print('Shuffling data.')
        np.random.shuffle(tokens)
    encoder_tokens = []
    decoder_tokens = []
    target_tokens = []
    for token in tokens:
        #encoder = add_speling_erors(token, error_rate=error_rate)
        encoder = error_tokens[token] # error_tokens is a dict mapping correct
                                      # tokens to possible errorful tokens
        encoder += EOS * (maxlen - len(encoder)) # Padded to maxlen.
        if reverse: encoder = encoder[::-1]
        encoder_tokens.append(encoder)

        decoder = SOS + token
        decoder += EOS * (maxlen - len(decoder))
        decoder_tokens.append(decoder)
    
        target = decoder[1:]
        target += EOS * (maxlen - len(target))
        target_tokens.append(target)
        assert(len(encoder) == len(decoder) == len(target))

    return encoder_tokens, decoder_tokens, target_tokens

def batch(tokens, maxlen, ctable, batch_size=128, reverse=False):
    """Split data into chunks of `batch_size` examples."""
    def generate(tokens, reverse):
        while(True): # This flag yields an infinite generator.
            for token in tokens:
                if reverse:
                    token = token[::-1]
                yield token
    
    token_iterator = generate(tokens, reverse)
    data_batch = np.zeros((batch_size, maxlen, ctable.size),
                          dtype=np.float32)
    while(True):
        for i in range(batch_size):
            token = next(token_iterator)
            data_batch[i] = ctable.encode(token, maxlen)
            #except: print(token)
        yield torch.from_numpy(data_batch)

def batch_triplet(token_triplets, maxlen, ctable, batch_size=128):
    def generate(token_triplets):
        while(True): # This flag yields an infinite generator.
            for token_triplet in token_triplets: yield token_triplet
    token_iterator = generate(token_triplets)
    data_batch = np.zeros((batch_size, 3, maxlen, ctable.size), dtype=np.float32)
    while(True):
        for i in range(batch_size):
            token = next(token_iterator)
            data_batch[i] = (ctable.encode(token[0], maxlen), ctable.encode(token[1], maxlen),ctable.encode(token[2], maxlen))
            #except: print(token)
        yield data_batch

def batch_bigram(token_pairs, maxlen, ctable, batch_size=128, reverse=False):
    """Split data into chunks of `batch_size` examples."""
    def generate(token_pairs, reverse):
        while(True): # This flag yields an infinite generator.
            for token_pair in token_pairs:
                #if reverse: token = token[::-1]
                yield token_pair
    
    token_iterator = generate(token_pairs, reverse)
    data_batch = np.zeros((batch_size, maxlen, ctable.size),
                          dtype=np.float32)
    while(True):
        for i in range(batch_size):
            token = next(token_iterator)
            data_batch[i] = ctable.encode(token, maxlen)
            #except: print(token)
        yield data_batch

def datagen_simple(input_iter, target_iter):
    """Utility function to load data into required model format."""
    while(True):
        input_ = next(input_iter)
        target = next(target_iter)
        yield (input_, target)

def datagen(encoder_iter, decoder_iter, target_iter):
    """Utility function to load data into required model format."""
    inputs = zip(encoder_iter, decoder_iter)
    while(True):
        encoder_input, decoder_input = next(inputs)
        target = next(target_iter)
        yield ([encoder_input, decoder_input], target)

def datagen_bahdanau(encoder_iter, decoder_iter):
    """Utility function to load data into required model format."""
    inputs = zip(encoder_iter, decoder_iter)
    while(True):
        encoder_input, decoder_input = next(inputs)
        #target = next(target_iter)
        yield (encoder_input, decoder_input)

def datagen_triplet(encoder_iter, decoder_iter, target_iter):
    """Utility function to load data into required model format."""
    inputs = zip(encoder_iter, decoder_iter)
    while(True):
        x0, decoder_input = next(inputs)
        x1, blank = next(inputs)
        x2, blank = next(inputs)
        target = next(target_iter)
        yield ([x0,x1,x2, decoder_input], target)

#def decode_sequences(inputs, targets, input_ctable, target_ctable,
#                     maxlen, reverse, encoder_model, decoder_model,
#                     nb_examples, sample_mode='argmax', random=True):
#    input_tokens = []
#    target_tokens = []
#    
#    if random:
#        indices = np.random.randint(0, len(inputs), nb_examples)
#    else:
#        indices = range(nb_examples)
#        
#    for index in indices:
#        input_tokens.append(inputs[index])
#        target_tokens.append(targets[index])
#    input_sequences = batch(input_tokens, maxlen, input_ctable,
#                            nb_examples, reverse)
#    input_sequences = next(input_sequences)
#    
#    # Procedure for inference mode (sampling):
#    # 1) Encode input and retrieve initial decoder state.
#    # 2) Run one step of decoder with this initial state
#    #    and a start-of-sequence character as target.
#    #    Output will be the next target character.
#    # 3) Repeat with the current target character and current states.
#
#    # Encode the input as state vectors.    
#    states_value = encoder_model.predict(input_sequences)
#    
#    # Create batch of empty target sequences of length 1 character.
#    target_sequences = np.zeros((nb_examples, 1, target_ctable.size))
#    # Populate the first element of target sequence
#    # with the start-of-sequence character.
#    target_sequences[:, 0, target_ctable.char2index[SOS]] = 1.0
#
#    # Sampling loop for a batch of sequences.
#    # Exit condition: either hit max character limit
#    # or encounter end-of-sequence character.
#    decoded_tokens = [''] * nb_examples
#    for _ in range(maxlen):
#        # `char_probs` has shape
#        # (nb_examples, 1, nb_target_chars)
#        char_probs, h, c = decoder_model.predict(
#            [target_sequences] + states_value)
#
#        # Reset the target sequences.
#        target_sequences = np.zeros((nb_examples, 1, target_ctable.size))
#
#        # Sample next character using argmax or multinomial mode.
#        sampled_chars = []
#        for i in range(nb_examples):
#            if sample_mode == 'argmax':
#                next_index, next_char = target_ctable.decode(
#                    char_probs[i], calc_argmax=True)
#            elif sample_mode == 'multinomial':
#                next_index, next_char = target_ctable.sample_multinomial(
#                    char_probs[i], temperature=0.5)
#            else:
#                raise Exception(
#                    "`sample_mode` accepts `argmax` or `multinomial`.")
#            decoded_tokens[i] += next_char
#            sampled_chars.append(next_char) 
#            # Update target sequence with index of next character.
#            target_sequences[i, 0, next_index] = 1.0
#
#        stop_char = set(sampled_chars)
#        if len(stop_char) == 1 and stop_char.pop() == EOS:
#            break
#            
#        # Update states.
#        states_value = [h, c]
#    
#    # Sampling finished.
#    input_tokens   = [re.sub('[%s]' % EOS, '', token)
#                      for token in input_tokens]
#    target_tokens  = [re.sub('[%s]' % EOS, '', token)
#                      for token in target_tokens]
#    decoded_tokens = [re.sub('[%s]' % EOS, '', token)
#                      for token in decoded_tokens]
#    return input_tokens, target_tokens, decoded_tokens

#def decode_sequences_attention(inputs, targets, input_ctable, target_ctable,
#                     maxlen, reverse, encoder_model, decoder_model,
#                     nb_examples, sample_mode='argmax', Random=True):
#    input_tokens, target_tokens = [],[]
#    for index in range(nb_examples):
#        input_tokens.append(inputs[index])
#        target_tokens.append(targets[index])
#
#    input_sequences = batch(input_tokens, maxlen, input_ctable, nb_examples, reverse)
#    input_sequences = next(input_sequences)
#    # Encode the input as state vectors.
#    encoder_output_inference,states_value = encoder_model.predict(input_sequences)
#    # Create batch of empty target sequences of length 1 character.
#    target_sequences = np.zeros((nb_examples, 1, target_ctable.size))
#    target_sequences[:, 0, target_ctable.char2index['\t']] = 1.0
#    #context_vector, attention_weights=attention(states_value[1],target_sequences)#att(enc_out,dec_out)
#
#    # Sampling loop for a batch of sequences.
#    # Exit condition: either hit max character limit
#    # or encounter end-of-sequence character.
#    decoded_tokens = [''] * nb_examples
#    for _ in range(maxlen):
#        # `char_probs` has shape
#        # (nb_examples, 1, nb_target_chars)
#        char_probs, h, c = decoder_model.predict([target_sequences,states_value,encoder_output_inference])
#
#        # Reset the target sequences.
#        target_sequences = np.zeros((nb_examples, 1, target_ctable.size))
#
#        # Sample next character using argmax or multinomial mode.
#        sampled_chars = []
#        for i in range(nb_examples):
#            if sample_mode == 'argmax':
#                next_index, next_char = target_ctable.decode(
#                        char_probs[i], calc_argmax=True)
#            elif sample_mode == 'multinomial':
#                next_index, next_char = target_ctable.sample_multinomial(
#                        char_probs[i], temperature=0.5)
#            else:
#                raise Exception(
#                        "`sample_mode` accepts `argmax` or `multinomial`.")
#            decoded_tokens[i] += next_char
#            sampled_chars.append(next_char) 
#            # Update target sequence with index of next character.
#            target_sequences[i, 0, next_index] = 1.0
#
#        stop_char = set(sampled_chars)
#        if len(stop_char) == 1 and stop_char.pop() == EOS: break
#
#        # Update states.
#        states_value = [h, c]
#    # Sampling finished.
#    input_tokens   = [re.sub('[%s]' % EOS, '', token)
#            for token in input_tokens]
#    target_tokens  = [re.sub('[%s]' % EOS, '', token)
#            for token in target_tokens]
#    decoded_tokens = [re.sub('[%s]' % EOS, '', token)
#            for token in decoded_tokens]
#    return input_tokens, target_tokens, decoded_tokens

#def restore_model_attention(path_to_full_model, hidden_size, lstm_2_flag=True):
#    # Compile the model.
#    custom_objects = {"Attention_layer":AttentionLayer, "recall": recall, "f1_score":f1_score, precision: "precision"}
#    model = load_model(path_to_full_model,custom_objects=custom_objects)
#    print("Printing out main model layer details: layer_name,layer_shape,layer_is_trainable")
#    for i, layer in enumerate(model.layers):
#        print(i, layer.name, layer.output_shape, layer.trainable)
#
#    encoder_inputs = model.input[0] # encoder_data
#    encoder_lstm1 = model.get_layer('encoder_lstm_1')
#    if lstm_2_flag==True: encoder_lstm2 = model.get_layer('encoder_lstm_2')
#
#    encoder_outputs_1 = encoder_lstm1(encoder_inputs)
#    if lstm_2_flag==True: encoder_outputs, state_h, state_c = encoder_lstm2(encoder_outputs_1)
#    else: encoder_outputs, state_h, state_c = encoder_outputs_1
#
#    encoder_output_states = [state_h, state_c]
#    encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs,encoder_output_states])
#
#    decoder_inputs = model.input[1] # decoder_data
#    decoder_state_input_h = Input(shape=(hidden_size,))
#    decoder_state_input_c = Input(shape=(hidden_size,))
#    # only used to connect encoder_states into decoder model
#    encoder_output_inputs = Input(shape=(hidden_size,))
#    decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
#    decoder_lstm = model.get_layer('decoder_lstm')
#    decoder_outputs, decoder_state_output_h, decoder_state_output_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
#    #decoder_states = [decoder_state_input_h, decoder_state_input_c]
#    #if 1==1:
#    #    #decoder_attention = AttentionLayer() #Luong
#    #    decoder_attention = model.get_layer("attention_layer")
#    #    decoder_outputs = decoder_attention([encoder_outputs[0],decoder_outputs])
#    attention_matmul = model.get_layer("tf.linalg.matmul")
#    luong_score = attention_matmul(decoder_outputs, encoder_output_inputs, transpose_b=True)
#    attention_softmax = model.get_layer("activation")
#    alignment = attention_softmax(luong_score)
#    attention_matmul_1 = model.get_layer("tf.linalg.matmul_1")
#    #attention_expand_dims = model.get_layer("tf.exapand_dims")
#    context = attention_matmul_1(alignment, K.expand_dims(encoder_output_inputs,axis=0))
#    attention_concatenate = model.get_layer("concatenate")
#    decoder_combined_context = attention_concatenate([context, decoder_outputs])
#
#    decoder_softmax = model.get_layer('decoder_softmax')
#    decoder_outputs = decoder_softmax(decoder_combined_context)
#
#    decoder_model = Model([decoder_inputs,decoder_state_inputs,encoder_output_inputs], [decoder_outputs,decoder_state_output_h, decoder_state_output_c])
#
#    return model, encoder_model, decoder_model

#def restore_model(path_to_full_model, hidden_size, lstm_2_flag=True, attention=True):
#    """Restore model to construct the encoder and decoder."""
#    custom_objects={}
#    if attention: custom_objects = {"Attention_layer":AttentionLayer,'truncated_acc': truncated_acc, 'truncated_loss': truncated_loss, 'recall': recall, "precision": precision, "f1_score": f1_score}
#    else: custom_objects = {'truncated_acc': truncated_acc, 'truncated_loss': truncated_loss, 'recall': recall, "precision": precision, "f1_score": f1_score}
#    model = load_model(path_to_full_model, custom_objects=custom_objects)
#    
#    encoder_inputs = model.input[0] # encoder_data
#    encoder_lstm1 = model.get_layer('encoder_lstm_1')
#    if lstm_2_flag==True: encoder_lstm2 = model.get_layer('encoder_lstm_2')
#    
#    encoder_outputs = encoder_lstm1(encoder_inputs)
#    if lstm_2_flag==True: _, state_h, state_c = encoder_lstm2(encoder_outputs)
#    else: _, state_h, state_c = encoder_outputs
#
#    encoder_states = [state_h, state_c]
#    encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)
#
#    decoder_inputs = model.input[1] # decoder_data
#    decoder_state_input_h = Input(shape=(hidden_size,))
#    decoder_state_input_c = Input(shape=(hidden_size,))
#    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
#    decoder_lstm = model.get_layer('decoder_lstm')
#    decoder_outputs, state_h, state_c = decoder_lstm(
#        decoder_inputs, initial_state=decoder_states_inputs)
#    decoder_states = [state_h, state_c]
#    #if attention_flag:
#        #decoder_attention = AttentionLayer() #Luong
#        #decoder_attention = model.get_layer()
#        #decoder_outputs = decoder_attention([encoder_outputs,decoder_outputs])
#    decoder_softmax = model.get_layer('decoder_softmax')
#    decoder_outputs = decoder_softmax(decoder_outputs)
#    decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs,
#                          outputs=[decoder_outputs] + decoder_states)
#    return model, encoder_model, decoder_model

#class AttentionLayer(Layer):
#
#    def __init__(self, **kwargs):
#        super(AttentionLayer, self).__init__(**kwargs)
#
#    def get_config(self):
#        config = super().get_config()
#        #config["mask"] = self.mask
#        return config
#
#    def compute_mask(self, inputs, mask=None):
#        #self.mask = mask
#        if mask == None:
#            return None
#        return mask[1]
#    def compute_output_shape(self, input_shape):
#        return (input_shape[1][0],input_shape[1][1],input_shape[1][2]*2)
#
#    def call(self, inputs, mask=None):
#        encoder_outputs, decoder_outputs = inputs
#
#        """
#        Task 3 attention
#
#        Start
#        """
#        luong_score = tf.matmul(decoder_outputs, encoder_outputs, transpose_b=True)
#        alignment = tf.nn.softmax(luong_score, axis=2)
#        context = tf.matmul(K.expand_dims(alignment,axis=2), K.expand_dims(encoder_outputs,axis=1))
#        encoder_vector = K.squeeze(context,axis=2)
#
#        """
#        End Task 3
#        """
#        # [batch,max_dec,2*emb]
#        new_decoder_outputs = K.concatenate([decoder_outputs, encoder_vector])
#        
#        return new_decoder_outputs
#
