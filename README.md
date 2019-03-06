# COMP6208 Advanced Machine Learning
## Group Project - NIMET

### test.py
Contains Keras model for basis CNN image classifier.
Currently set to use PlaidML backend for training on AMD GPU.
To use with Tensorflow, comment out lines

```python
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
```

Optionally also change all

```python
from keras...
```
to 
```python
from tensorflow.keras...
```
