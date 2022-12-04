# T5_encoder_decoder_prompt_tuning_for_text_generation

The code base for the model implementations for the paper ""T5 encoder decoder soft prompt tuning for text generation.


## Required Packages

All the codes were run in an Anacoda Python 3.10 environment. After cloning the repository follow the below instructions to install all the packages required.

```sh
# Install the requirements form the requirements.txt file
$ python -m pip install -U pip
$ pip install -r requirements.txt
```

## Downloading Trained Models and Data

To perform text generation and classification using the trained models produced, download the "trained_models.zip" archive file from the following link () and extract the content into the "trained_models" folder of the repo.

The data used from training the above moels can be accessed from the following link (). To experiment with the models further, download the "data.zip" archive file from the aforementioned link and extract the content into the "data" folder of the repo.


## Running the text generation demo

Once the pre-trained models are downloaded as mentioned above, the "Text_Generation_Demo.ipynb" Jupyter notebook can be executed to generate poitive and negative texts using the T5 model with encode-decoder soft prompts.
