# Logo classifier V1.0

### Function

Recognize logo from a picture 

### Support

Only 1 picture at a time right now.

Only local command line and local file address input is supported.

Only following brands are included:
'BMW', 'Ford', 'Honda', 'Toyota', 'VW'

### Requirements

Packsages:

tb-nightly==2.4.0a20201015
tensorflow==2.2.0
Pillow==8.0.0
numpy 
pandas 
matplotlib

Files:

images, logo_model1, class_names.txt, logoclassifierV1.0.py


### Usage

Try `python logoclassifierV1.0.py "filename"`

For example, `python logoclassifierV1.0.py "./images/BMW/images (29).jpeg"`

