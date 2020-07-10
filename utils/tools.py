"""
Created on Tue Jun 23 20:15:11 2020

@author: sarroutim2
"""

"""Contains a set of helper functions.
"""


class Dict2Obj(dict):
    """Converts dicts to objects.
    """

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def merge(self, other, overwrite=True):
        for name in other:
            if overwrite or name not in self:
                self[name] = other[name]
