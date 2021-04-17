# Class-based dictionary, with attributes corresponding to keys.
class cdict(dict):

    __slots__ = ()

    @classmethod
    def from_object(cls, obj):
        return cls((a, getattr(obj, a, None)) for a in dir(obj))

    __init__ = lambda self, *args, **kwargs: super().__init__(*args, **kwargs)
    __repr__ = lambda self: self.__class__.__name__ + ("((" + ",".join("(" + ",".join(repr(i) for i in item) + ")" for item in super().items()) + ("," if len(self) == 1 else "") + "))") if self else "()"
    __str__ = lambda self: super().__repr__()
    __iter__ = lambda self: iter(tuple(super().__iter__()))
    __call__ = lambda self, k: self.__getitem__(k)

    def __getattr__(self, k):
        try:
            return self.__getattribute__(k)
        except AttributeError:
            pass
        if not k.startswith("__") or not k.endswith("__"):
            try:
                return self.__getitem__(k)
            except KeyError as ex:
                raise AttributeError(*ex.args)
        raise AttributeError(k)

    def __setattr__(self, k, v):
        if k.startswith("__") and k.endswith("__"):
            return object.__setattr__(self, k, v)
        return self.__setitem__(k, v)

    def __dir__(self):
        data = set(object.__dir__(self))
        data.update(self)
        return data

    @property
    def __dict__(self):
        return self

    ___repr__ = lambda self: super().__repr__()
    to_dict = lambda self: dict(**self)
    to_list = lambda self: list(super().values())


# A dict with key-value pairs fed from more dict-like objects.
class fdict(cdict):

    __slots__ = ("_feed",)

    def get_feed(self):
        feed = object.__getattribute__(self, "_feed")
        if callable(feed):
            return feed()
        return feed

    def _keys(self):
        found = set()
        for k in super().keys():
            found.add(k)
            yield k
        for f in self.get_feed():
            for k in f:
                if k not in found:
                    found.add(k)
                    yield k

    def keys(self):
        try:
            self.get_feed()
        except AttributeError:
            return super().keys()
        return self._keys()

    __iter__ = lambda self: iter(super().keys())

    def _values(self):
        found = set()
        for k, v in super().items():
            found.add(k)
            yield v
        for f in self.get_feed():
            for k, v in f.items():
                if k not in found:
                    found.add(k)
                    yield v

    def values(self):
        try:
            self.get_feed()
        except AttributeError:
            return super().values()
        return self._values()

    def _items(self):
        found = set()
        for k, v in super().items():
            found.add(k)
            yield k, v
        for f in self.get_feed():
            for k, v in f.items():
                if k not in found:
                    found.add(k)
                    yield k, v

    def items(self):
        try:
            self.get_feed()
        except AttributeError:
            return super().items()
        return self._items()

    def _len_(self):
        size = len(self)
        try:
            self.get_feed()
        except AttributeError:
            return size
        for f in self.get_feed():
            try:
                size += f._len_()
            except AttributeError:
                size += len(f)
        return size

    def __getitem__(self, k):
        try:
            return super().__getitem__(k)
        except KeyError:
            pass
        try:
            self.get_feed()
        except AttributeError:
            raise KeyError(k)
        for f in self.get_feed():
            try:
                return f.__getitem__(k)
            except KeyError:
                pass
        raise KeyError(k)

    def __setattr__(self, k, v):
        if k == "_feed" or k.startswith("__") and k.endswith("__"):
            return object.__setattr__(self, k, v)
        return self.__setitem__(k, v)

    def __dir__(self):
        data = set(object.__dir__(self))
        data.update(self)
        try:
            self.get_feed()
        except AttributeError:
            return data
        for f in self.get_feed():
            data.update(f)
        return data

    def get(self, k, default=None):
        try:
            return self[k]
        except KeyError:
            return default
