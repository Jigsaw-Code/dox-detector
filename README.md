# NYU & Jigsaw Models
Models from joint research project on doxing and calls to harassment.

Paper: [A large-scale characterization of online incitements to harassment across platforms](https://dl.acm.org/doi/abs/10.1145/3487552.3487852)

## Details
The following is some high-level details about each model. 
More information can be found in the current manuscript.
Additional metadata can be found in the `config.json` for each model.

Path structures:

`models/MODEL_NAME/pt/`
* `config.json`: hyperparameters used during training, architecture information, etc.
* `pytorch_model.bin`: trained model weights
* `special_tokens_map.json`: tokenizer special token values
* `tokenizer_config.json`: trained tokenizer arguments
* `training_args.bin`: training arguments saved as bin file
* `vocab.txt`: tokenizer vocabulary file (using the default `distilbert` vocab file.)

`models/MODEL_NAME/tf/`
* `config.json`: tensorflow config file
* `tf_model.h5`: tensorflow HDF5 model
* `saved_model/1/saved_model.pb`: saved model that can be loaded in keras or tensorflow serving
* `saved_model/1/variables`: model variables directory

The conversion between pytorch and tensorflow was doing using the `/bin/convert_pt_tf.py` script.

## Running Models
The models are both in pytorch and tensorflow format. They were created using the Huggingface Transformers library.
The `tpu.ipynb` file in the `notebooks` directory shows an example of how to classify a large amount of data using a TPU and a pre-trained model.

### pytorch
However the general process for loading a pytorch model is as follows:

```python
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
model_path = 'path/to/model/file/'

config = AutoConfig.from_pretrained(
    model_path,
    num_labels=2
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    config=config
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    use_fast=True
)

padding = 'max_length' # pad the input text to the maximum length
max_length = 512 # set length based on model type

example_text = 'example input text'

args = (
    (example_text,)
)
# Tokenize the texts
result = tokenizer(*args, padding=padding, max_length=max_length, truncation=True)

output = model(result)
print(output) # can run argmax on this to get predicted class, or use outputs for probability
```

### Tensorflow Serving
The tensorflow version of the model can also be deployed via tensorflow serving so that input text can be sent to it via an API layer. You can start the tensorflow serving version of the model with the following:
```bash
./bin/tensorflow_serving.sh /models/{MODEL_NAME}/tf/saved_model/ {MODEL_NAME}
```
Note, this assumes you have docker installed.

You can then make test predictions against the model using the `bin/get_predictions.py` file. You need to configure the file manually:
* `sentence`: example input text you are classifying.
* `model_name`: name of the model deployed above.


## Resources
* https://huggingface.co/transformers/
* https://www.tensorflow.org/tfx/guide/serving