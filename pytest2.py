import logoclassifierV1 as lc1
from logoclassifier_modeltrainer as mt

import pytest
import numbers

#Make sure only 2 sets of data: train and validation
def test_data_transform1():
    assert mt.train_ds == 2

#make sure class names are all counted
def train_class_name_verify():
    assert mt.train_ds.class_names == ['BMW', 'Ford', 'Honda', 'Toyota', 'VW']

def val_class_name_verify():
    assert mt.val_ds.class_names == ['BMW', 'Ford', 'Honda', 'Toyota', 'VW']

#ensure the accuracy number is within the range
def acc_range():
    assert mt.max(acc) < 1
    assert mt.min(acc) > 0

def val_acc_range():
    assert mt.max(val_acc) < 1
    assert mt.min(val_acc) > 0

#ensure the loss number is within the range
def loss_range():
    assert mt.loss > 0
    assert mt.val_loss > 0

