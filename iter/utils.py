from collections import OrderedDict
from typing import TypeVar, Callable, overload

_T = TypeVar("_T")

def _named_tuple_getitem(self, item):
    return getattr(self, item)

def _named_tuple_setitem(self, item, value):
    setattr(self, item, value)

@overload
def namedtuple(cls: type[_T], /) -> type[_T]: ...

@overload
def namedtuple(cls: None = None, /) -> Callable[[type[_T]], type[_T]]: ...

def namedtuple(cls=None):
    def wrap(cls):
        setattr(cls, "__getitem__", _named_tuple_getitem)
        setattr(cls, "__setitem__", _named_tuple_setitem)
        return cls

    if cls is None:
        return wrap

    return wrap(cls)

class NamedTuple2:
    _annotations: OrderedDict[int, str]

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)
        inverse_cls_order = list()
        cls = self.__class__
        while cls != NamedTuple2:
            inverse_cls_order.append(cls)
            cls = cls.__bases__[0]
        inverse_cls_order = list(reversed(inverse_cls_order))
        self._annotations = OrderedDict()

        already_seen = set()
        for cls in inverse_cls_order:
            for field, tp in cls.__annotations__.items():
                if field in already_seen:
                    continue
                already_seen.add(field)
                self._annotations[len(self._annotations)] = field
        return self

    def __getitem__(self, item):
        if isinstance(item, int) and item >= len(self):
            raise IndexError
        return getattr(self, self._annotations[item])

    def __len__(self):
        return len(self._annotations)
