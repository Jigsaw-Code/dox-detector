import tensorflow as tf
import tensorflow_text as tf_text
import tensorflow as tf
import transformers


MAX_SEQ_LEN = 512  # max number is 512
NUM_LABELS = 2
do_lower_case = True

model_path = '/models/cth/pt/'
vocab_file_path = model_path + 'vocab.txt'
    

class BertTokenizer(tf.Module):
    def __init__(self, vocab_file_path, sequence_length=MAX_SEQ_LEN, lower_case=True):
        self.CLS_ID = tf.constant(101, dtype=tf.int64)
        self.SEP_ID = tf.constant(102, dtype=tf.int64)
        self.PAD_ID = tf.constant(0, dtype=tf.int64)

        self.sequence_length = tf.constant(sequence_length)

        self.bert_tokenizer = tf_text.BertTokenizer(
            vocab_lookup_table=vocab_file_path,
            token_out_type=tf.int64,
            lower_case=lower_case,
        )

    @tf.function
    def __call__(self, inputs) -> tf.Tensor:
        """
        Perform the BERT preprocessing from text -> input token ids
        """
        tokens = self.bert_tokenizer.tokenize(inputs)
        # Flatten the ragged tensors
        tokens = tokens.merge_dims(1, 2)

        # Add start and end token ids to the id sequence
        start_tokens = tf.fill([tf.shape(inputs)[0], 1], self.CLS_ID)
        end_tokens = tf.fill([tf.shape(inputs)[0], 1], self.SEP_ID)
        tokens = tf.concat([start_tokens, tokens, end_tokens], axis=1)

        # # Truncate to sequence length
        tokens = tokens[:, : self.sequence_length]

        # # Convert ragged tensor to tensor and pad with PAD_ID
        tokens = tokens.to_tensor(default_value=self.PAD_ID)

        # # Pad to sequence length
        pad = self.sequence_length - tf.shape(tokens)[1]
        tokens = tf.pad(tokens, [[0, 0], [0, pad]], constant_values=self.PAD_ID)

        input_ids = tf.reshape(tokens, [-1, self.sequence_length])
        input_mask = tf.cast(input_ids > 0, tf.int64)
        input_mask = tf.reshape(input_mask, [-1, self.sequence_length])

        zeros_dims = tf.stack(tf.shape(input_mask))
        input_type_ids = tf.fill(zeros_dims, 0)
        input_type_ids = tf.cast(input_type_ids, tf.int64)
        return {
            'input_ids': input_ids,
            'attention_mask': input_mask
        }


# NOTE: the specific model here will need to be overwritten because AutoModel doesn't work
class CustomModel(transformers.TFDistilBertForSequenceClassification):
# class CustomModel(transformers.TFBertForSequenceClassification):

    @tf.function(input_signature=[tf.TensorSpec(None, tf.string)])
    def serving(self, inputs):
        tokenized_text = self.tokenizer([inputs])
        output = self.call(tokenized_text).logits
        output = tf.nn.softmax(
            output, axis=None, name=None
        )
        return output

config = transformers.AutoConfig.from_pretrained(
    model_path,
    num_labels=NUM_LABELS,
    from_pt=True
)

model = CustomModel.from_pretrained(
    model_path,
    config=config,
    from_pt=True
)

model.tokenizer = BertTokenizer(vocab_file_path=vocab_file_path)

model.summary()

output_path = '/'.join(
    model_path.split('/')[:-2]
) + '/tf'

print(
    f"Saving model to: {output_path}"
)

model.save_pretrained(
    output_path,
    saved_model=True
)
