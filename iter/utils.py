from collections import OrderedDict


class NamedTuple2:
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
        if item >= len(self):
            raise IndexError
        return getattr(self, self._annotations[item])

    def __len__(self):
        return len(self._annotations)
