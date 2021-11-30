import tensorflow as tf
import tensorflow_addons as tfa

import pickle


with open("train.pickle", "rb") as f:
    (
        input_tensor_train,
        target_tensor_train,
        input_tensor_val,
        target_tensor_val,
        inp_lang_tokenizer,
        targ_lang_tokenizer,
        vocab_inp_size,
        vocab_tar_size,
        max_length_input,
        max_length_output,
        vocab_inp_size,
        vocab_tar_size,
    ) = pickle.load(f)

BUFFER_SIZE = 32000
BATCH_SIZE = 64
# Limiting the training examples for faster training
embedding_dim = 256
units = 1024


def call(
    input_tensor_train,
    target_tensor_train,
    input_tensor_val,
    target_tensor_val,
    inp_lang_tokenizer,
    targ_lang_tokenizer,
    BATCH_SIZE,
):

    # input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (input_tensor_train, target_tensor_train)
    )
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(
        BATCH_SIZE, drop_remainder=True
    )

    val_dataset = tf.data.Dataset.from_tensor_slices(
        (input_tensor_val, target_tensor_val)
    )
    val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

    return train_dataset, val_dataset, inp_lang_tokenizer, targ_lang_tokenizer


train_dataset, val_dataset, inp_lang_tokenizer, targ_lang_tokenizer = call(
    input_tensor_train,
    target_tensor_train,
    input_tensor_val,
    target_tensor_val,
    inp_lang_tokenizer,
    targ_lang_tokenizer,
    BATCH_SIZE,
)

example_input_batch, example_target_batch = next(iter(train_dataset))
example_input_batch.shape, example_target_batch.shape

example_input_batch, example_target_batch = next(iter(train_dataset))
example_input_batch.shape, example_target_batch.shape

# print("max_length_spanish, max_length_english, vocab_size_spanish, vocab_size_english")
# max_length_input, max_length_output, vocab_inp_size, vocab_tar_size


# Load weights and Define Models
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        # LSTM layer in Encoder
        self.lstm_layer = tf.keras.layers.LSTM(
            self.enc_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )

    def call(self, x):
        hidden = x[1]
        embeddin_op = self.embedding(x[0])
        output, h, c = self.lstm_layer(embeddin_op, initial_state=hidden)  #
        return output, h, c

    def initialize_hidden_state(self):
        return [
            tf.zeros((self.batch_sz, self.enc_units)),
            tf.zeros((self.batch_sz, self.enc_units)),
        ]


encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)


# sample input
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_h, sample_c = encoder([example_input_batch, sample_hidden])  #

encoder.load_weights("./out/logs/10_Loss_0.0947_encoder.h5")


class Decoder(tf.keras.Model):
    def __init__(
        self, vocab_size, embedding_dim, dec_units, batch_sz, attention_type="luong"
    ):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.attention_type = attention_type

        # Embedding Layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        # Final Dense layer on which softmax will be applied
        self.fc = tf.keras.layers.Dense(vocab_size)

        # Define the fundamental cell for decoder recurrent structure
        self.decoder_rnn_cell = tf.keras.layers.LSTMCell(self.dec_units)

        # Sampler
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()

        # Create attention mechanism with memory = None
        self.attention_mechanism = self.build_attention_mechanism(
            self.dec_units,
            None,
            self.batch_sz * [max_length_input],
            self.attention_type,
        )

        # Wrap attention mechanism with the fundamental rnn cell of decoder
        self.rnn_cell = self.build_rnn_cell(batch_sz)

        # Define the decoder with respect to fundamental rnn cell
        self.decoder = tfa.seq2seq.BasicDecoder(
            self.rnn_cell, sampler=self.sampler, output_layer=self.fc
        )

    def build_rnn_cell(self, batch_sz):
        rnn_cell = tfa.seq2seq.AttentionWrapper(
            self.decoder_rnn_cell,
            self.attention_mechanism,
            attention_layer_size=self.dec_units,
        )
        return rnn_cell

    def build_attention_mechanism(
        self, dec_units, memory, memory_sequence_length, attention_type="luong"
    ):
        # ------------- #
        # typ: Which sort of attention (Bahdanau, Luong)
        # dec_units: final dimension of attention outputs
        # memory: encoder hidden states of shape (batch_size, max_length_input, enc_units)
        # memory_sequence_length: 1d array of shape (batch_size) with every element set to max_length_input (for masking purpose)

        if attention_type == "bahdanau":
            return tfa.seq2seq.BahdanauAttention(
                units=dec_units,
                memory=memory,
                memory_sequence_length=memory_sequence_length,
            )
        else:
            return tfa.seq2seq.LuongAttention(
                units=dec_units,
                memory=memory,
                memory_sequence_length=memory_sequence_length,
            )

    def build_initial_state(self, batch_sz, encoder_state, Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(
            batch_size=batch_sz, dtype=Dtype
        )
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
        return decoder_initial_state

    def call(self, x):
        initial_state = x[1]
        embedding_op = self.embedding(x[0])
        outputs, _, _ = self.decoder(
            embedding_op,
            initial_state=initial_state,
            sequence_length=self.batch_sz * [max_length_output - 1],
        )
        return outputs


# Test decoder stack

decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE, "luong")
sample_x = tf.random.uniform((BATCH_SIZE, max_length_output))
decoder.attention_mechanism.setup_memory(sample_output)
initial_state = decoder.build_initial_state(
    BATCH_SIZE, [sample_h, sample_c], tf.float32
)
sample_decoder_outputs = decoder([sample_x, initial_state])

decoder.load_weights("./out/logs/10_Loss_0.0947_decoder.h5")

# Evaluation


def evaluate_sentence(sentence):
    sentence = sentence  # dataset_creator.preprocess_sentence(sentence)

    inputs = [inp_lang_tokenizer.word_index[i] for i in sentence.split(" ")]  #
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
        [inputs], maxlen=max_length_input, padding="post"
    )
    inputs = tf.convert_to_tensor(inputs)
    inference_batch_size = inputs.shape[0]
    result = ""

    enc_start_state = [
        tf.zeros((inference_batch_size, units)),
        tf.zeros((inference_batch_size, units)),
    ]
    enc_out, enc_h, enc_c = encoder([inputs, enc_start_state])

    dec_h = enc_h
    dec_c = enc_c

    start_tokens = tf.fill(
        [inference_batch_size], targ_lang_tokenizer.word_index["cls"]
    )
    end_token = targ_lang_tokenizer.word_index["sep"]

    greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

    # Instantiate BasicDecoder object
    decoder_instance = tfa.seq2seq.BasicDecoder(
        cell=decoder.rnn_cell, sampler=greedy_sampler, output_layer=decoder.fc
    )
    # Setup Memory in decoder stack
    decoder.attention_mechanism.setup_memory(enc_out)

    # set decoder_initial_state
    decoder_initial_state = decoder.build_initial_state(
        inference_batch_size, [enc_h, enc_c], tf.float32
    )

    ### Since the BasicDecoder wraps around Decoder's rnn cell only, you have to ensure that the inputs to BasicDecoder
    ### decoding step is output of embedding layer. tfa.seq2seq.GreedyEmbeddingSampler() takes care of this.
    ### You only need to get the weights of embedding layer, which can be done by decoder.embedding.variables[0] and pass this callabble to BasicDecoder's call() function

    decoder_embedding_matrix = decoder.embedding.variables[0]

    outputs, _, _ = decoder_instance(
        decoder_embedding_matrix,
        start_tokens=start_tokens,
        end_token=end_token,
        initial_state=decoder_initial_state,
    )
    return outputs.sample_id.numpy()


def translate(sentence):
    result = evaluate_sentence(sentence)
    result = targ_lang_tokenizer.sequences_to_texts(result)
    result = (
        str(result)
        .replace(" sep", "")
        .replace("]", "")
        .replace("[", "")
        .replace("'", "")
    )
    return result


def give_suggestion(input):
    print(input)
    try:
        return translate(input)
    except Exception:
        return
