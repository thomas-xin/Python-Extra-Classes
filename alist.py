import numpy, itertools, collections, concurrent.futures
np = numpy
from itertools import repeat
from collections import deque


class alist(collections.abc.MutableSequence, collections.abc.Callable):

    maxoff = (1 << 24) - 1
    minsize = 9
    __slots__ = ("hash", "block", "offs", "size", "data", "frozenset", "queries", "_index")

    # For thread-safety: Waits until the list is not busy performing an operation.
    def waiting(self):
        func = self
        def call(self, *args, force=False, **kwargs):
            if not force and type(self.block) is concurrent.futures.Future:
                self.block.result(timeout=12)
            return func(self, *args, **kwargs)
        return call

    # For thread-safety: Blocks the list until the operation is complete.
    def blocking(self):
        func = self
        def call(self, *args, force=False, **kwargs):
            if not force and type(self.block) is concurrent.futures.Future:
                self.block.result(timeout=12)
            self.block = concurrent.futures.Future()
            self.hash = None
            self.frozenset = None
            try:
                del self.queries
            except AttributeError:
                pass
            try:
                output = func(self, *args, **kwargs)
            except:
                try:
                    self.block.set_result(None)
                except concurrent.futures.InvalidStateError:
                    pass
                raise
            try:
                self.block.set_result(None)
            except concurrent.futures.InvalidStateError:
                pass
            return output
        return call

    # Init takes arguments and casts to a deque if possible, else generates as a single value. Allocates space equal to 3 times the length of the input iterable.
    def __init__(self, *args, fromarray=True, **void):
        fut = getattr(self, "block", None)
        self.block = concurrent.futures.Future()
        self.hash = None
        self.frozenset = None
        if fut:
            try:
                del self.queries
            except AttributeError:
                pass
            try:
                del self._index
            except AttributeError:
                pass
        if not args:
            self.offs = 0
            self.size = 0
            self.data = None
            try:
                self.block.set_result(None)
            except concurrent.futures.InvalidStateError:
                pass
            if fut:
                try:
                    fut.set_result(None)
                except concurrent.futures.InvalidStateError:
                    pass
            return
        elif len(args) == 1:
            iterable = args[0]
        else:
            iterable = args
        if issubclass(type(iterable), self.__class__) and iterable:
            self.offs = iterable.offs
            self.size = iterable.size
            if fromarray:
                self.data = iterable.data
            else:
                self.data = iterable.data.copy()
        elif fromarray and isinstance(iterable, np.ndarray):
            self.offs = 0
            self.size = len(iterable)
            self.data = iterable
        else:
            if not issubclass(type(iterable), collections.abc.Sequence) or issubclass(type(iterable), collections.abc.Mapping) or type(iterable) in (str, bytes):
                try:
                    iterable = deque(iterable)
                except TypeError:
                    iterable = [iterable]
            self.size = len(iterable)
            if fromarray:
                size = self.size
                self.offs = 0
            else:
                size = max(self.minsize, self.size * 3)
                self.offs = size // 3
            self.data = np.empty(size, dtype=object)
            self.view[:] = iterable
        if not fut or fut.done():
            try:
                self.block.set_result(None)
            except concurrent.futures.InvalidStateError:
                pass
            if fut:
                try:
                    fut.set_result(None)
                except concurrent.futures.InvalidStateError:
                    pass

    def __getstate__(self):
        if self.size <= 0:
            self.clear()
            self.data = None
            self.offs = 0
        return self.data, self.offs, self.size

    def __setstate__(self, s):
        if type(s) is tuple:
            if len(s) == 2:
                if s[0] is None:
                    for k, v in s[1].items():
                        setattr(self, k, v)
                    self.block = None
                    return
            elif len(s) == 3:
                self.data, self.offs, self.size = s
                self.hash = None
                self.frozenset = None
                try:
                    del self.queries
                except AttributeError:
                    pass
                self.block = None
                return
        raise TypeError("Unpickling failed:", s)

    def __getattr__(self, k):
        try:
            return self.__getattribute__(k)
        except AttributeError:
            pass
        return getattr(self.__getattribute__("view"), k)

    def __dir__(self):
        data = set(object.__dir__(self))
        data.update(dir(self.view))
        return data

    # Returns a numpy array representing the items currently "in" the list.
    @property
    def view(self):
        data = self.__getattribute__("data")
        if data is None:
            return []
        offs, size = [self.__getattribute__(i) for i in ("offs", "size")]
        return data[offs:offs + size]

    @waiting
    def __call__(self, arg=1, *void1, **void2):
        if arg == 1:
            return self.copy()
        return self * arg

    # Returns the hash value of the data in the list.
    def __hash__(self):
        if self.hash is None:
            self.hash = hash(self.view.tobytes())
        return self.hash

    def to_frozenset(self):
        if self.frozenset is None:
            self.frozenset = frozenset(self)
        return self.frozenset

    # Basic functions
    __str__ = lambda self: "[" + ", ".join(repr(i) for i in iter(self)) + "]"
    __repr__ = lambda self: f"{self.__class__.__name__}({tuple(self) if self.__bool__() else ''})"
    __bool__ = lambda self: self.size > 0

    # Arithmetic functions

    @blocking
    def __iadd__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.add(arr, iterable, out=arr)
        return self

    @blocking
    def __isub__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.subtract(arr, iterable, out=arr)
        return self

    @blocking
    def __imul__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.multiply(arr, iterable, out=arr)
        return self

    @blocking
    def __imatmul__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        temp = np.matmul(arr, iterable)
        self.size = len(temp)
        arr[:self.size] = temp
        return self

    @blocking
    def __itruediv__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.true_divide(arr, iterable, out=arr)
        return self

    @blocking
    def __ifloordiv__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.floor_divide(arr, iterable, out=arr)
        return self

    @blocking
    def __imod__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.mod(arr, iterable, out=arr)
        return self

    @blocking
    def __ipow__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.power(arr, iterable, out=arr)
        return self

    @blocking
    def __ilshift__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        try:
            np.left_shift(arr, iterable, out=arr)
        except (TypeError, ValueError):
            np.multiply(arr, np.power(2, iterable), out=arr)
        return self

    @blocking
    def __irshift__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        try:
            np.right_shift(arr, iterable, out=arr)
        except (TypeError, ValueError):
            np.divide(arr, np.power(2, iterable), out=arr)
        return self

    @blocking
    def __iand__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.logical_and(arr, iterable, out=arr)
        return self

    @blocking
    def __ixor__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.logical_xor(arr, iterable, out=arr)
        return self

    @blocking
    def __ior__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.logical_or(arr, iterable, out=arr)
        return self

    @waiting
    def __neg__(self):
        return self.__class__(-self.view)

    @waiting
    def __pos__(self):
        return self

    @waiting
    def __abs__(self):
        d = self.data
        return self.__class__(np.abs(self.view))

    @waiting
    def __invert__(self):
        return self.__class__(np.invert(self.view))

    @waiting
    def __add__(self, other):
        temp = self.copy()
        temp += other
        return temp

    @waiting
    def __sub__(self, other):
        temp = self.copy()
        temp -= other
        return temp

    @waiting
    def __mul__(self, other):
        temp = self.copy()
        temp *= other
        return temp

    @waiting
    def __matmul__(self, other):
        temp1 = self.view
        temp2 = self.to_iterable(other)
        result = temp1 @ temp2
        return self.__class__(result)

    @waiting
    def __truediv__(self, other):
        temp = self.copy()
        temp /= other
        return temp

    @waiting
    def __floordiv__(self, other):
        temp = self.copy()
        temp //= other
        return temp

    @waiting
    def __mod__(self, other):
        temp = self.copy()
        temp %= other
        return temp

    @waiting
    def __pow__(self, other):
        temp = self.copy()
        temp **= other
        return temp

    @waiting
    def __lshift__(self, other):
        temp = self.copy()
        temp <<= other
        return temp

    @waiting
    def __rshift__(self, other):
        temp = self.copy()
        temp >>= other
        return temp

    @waiting
    def __and__(self, other):
        temp = self.copy()
        temp &= other
        return temp

    @waiting
    def __xor__(self, other):
        temp = self.copy()
        temp ^= other
        return temp

    @waiting
    def __or__(self, other):
        temp = self.copy()
        temp |= other
        return temp

    @waiting
    def __round__(self, prec=0):
        temp = np.round(self.view, prec)
        return self.__class__(temp)

    @waiting
    def __trunc__(self):
        temp = np.trunc(self.view)
        return self.__class__(temp)

    @waiting
    def __floor__(self):
        temp = np.floor(self.view)
        return self.__class__(temp)

    @waiting
    def __ceil__(self):
        temp = np.ceil(self.view)
        return self.__class__(temp)

    __index__ = lambda self: self.view
    __radd__ = __add__
    __rsub__ = lambda self, other: -self + other
    __rmul__ = __mul__
    __rmatmul__ = __matmul__

    @waiting
    def __rtruediv__(self, other):
        temp = self.__class__(self.data)
        iterable = self.to_iterable(other)
        arr = temp.view
        np.true_divide(iterable, arr, out=arr)
        return temp

    @waiting
    def __rfloordiv__(self, other):
        temp = self.__class__(self.data)
        iterable = self.to_iterable(other)
        arr = temp.view
        np.floor_divide(iterable, arr, out=arr)
        return temp

    @waiting
    def __rmod__(self, other):
        temp = self.__class__(self.data)
        iterable = self.to_iterable(other)
        arr = temp.view
        np.mod(iterable, arr, out=arr)
        return temp

    @waiting
    def __rpow__(self, other):
        temp = self.__class__(self.data)
        iterable = self.to_iterable(other)
        arr = temp.view
        np.power(iterable, arr, out=arr)
        return temp

    @waiting
    def __rlshift__(self, other):
        temp = self.__class__(self.data)
        iterable = self.to_iterable(other)
        arr = temp.view
        try:
            np.left_shift(iterable, arr, out=arr)
        except (TypeError, ValueError):
            np.multiply(iterable, np.power(2, arr), out=arr)
        return temp

    @waiting
    def __rrshift__(self, other):
        temp = self.__class__(self.data)
        iterable = self.to_iterable(other)
        arr = temp.view
        try:
            np.right_shift(iterable, arr, out=arr)
        except (TypeError, ValueError):
            np.divide(iterable, np.power(2, arr), out=arr)
        return temp

    __rand__ = __and__
    __rxor__ = __xor__
    __ror__ = __or__

    # Comparison operations

    @waiting
    def __lt__(self, other):
        it = self.to_iterable(other)
        return self.view < other

    @waiting
    def __le__(self, other):
        it = self.to_iterable(other)
        return self.view <= other

    @waiting
    def __eq__(self, other):
        try:
            it = self.to_iterable(other)
            return self.view == other
        except (TypeError, IndexError):
            return

    @waiting
    def __ne__(self, other):
        try:
            it = self.to_iterable(other)
            return self.view != other
        except (TypeError, IndexError):
            return

    @waiting
    def __gt__(self, other):
        it = self.to_iterable(other)
        return self.view > other

    @waiting
    def __ge__(self, other):
        it = self.to_iterable(other)
        return self.view >= other

    # Takes ints, floats, slices and iterables for indexing
    @waiting
    def __getitem__(self, *args):
        if len(args) == 1:
            key = args[0]
            if type(key) in (float, complex):
                return get(self.view, key, 1)
            if type(key) is int:
                try:
                    key = key % self.size
                except ZeroDivisionError:
                    raise IndexError("Array List index out of range.")
                return self.view.__getitem__(key)
            if type(key) is slice:
                if key.step in (None, 1):
                    start = key.start
                    if start is None:
                        start = 0
                    stop = key.stop
                    if stop is None:
                        stop = self.size
                    if start >= self.size or stop <= start and (stop >= 0 or stop + self.size <= start):
                        return self.__class__()
                    temp = self.__class__(self, fromarray=True)
                    if start < 0:
                        if start < -self.size:
                            start = 0
                        else:
                            start %= self.size
                    if stop < 0:
                        stop %= self.size
                    elif stop > self.size:
                        stop = self.size
                    temp.offs += start
                    temp.size = stop - start
                    if not temp.size:
                        return self.__class__()
                    return temp
            return self.__class__(self.view.__getitem__(key), fromarray=True)
        return self.__class__(self.view.__getitem__(*args), fromarray=True)

    # Takes ints, slices and iterables for indexing
    @blocking
    def __setitem__(self, *args):
        if len(args) == 2:
            key = args[0]
            if type(key) is int:
                try:
                    key = key % self.size
                except ZeroDivisionError:
                    raise IndexError("Array List index out of range.")
            return self.view.__setitem__(key, args[1])
        return self.view.__setitem__(*args)

    # Takes ints and slices for indexing
    @blocking
    def __delitem__(self, key):
        if type(key) is slice:
            s = key.indices(self.size)
            return self.pops(xrange(*s))
        try:
            len(key)
        except TypeError:
            return self.pop(key, force=True)
        return self.pops(key)

    # Basic sequence functions
    __len__ = lambda self: self.size
    __length_hint__ = __len__
    __iter__ = lambda self: iter(self.view)
    __reversed__ = lambda self: iter(np.flip(self.view))

    def next(self):
        try:
            self._index = (self._index + 1) % self.size
        except AttributeError:
            self._index = 0
        return self[self._index]

    @waiting
    def __bytes__(self):
        return bytes(round(i) & 255 for i in self.view)

    def __contains__(self, item):
        try:
            if self.queries >= 8:
                return item in self.to_frozenset()
            if self.frozenset is not None:
                return item in self.frozenset
            self.queries += 1
        except AttributeError:
            self.queries = 1
        return item in self.view

    __copy__ = lambda self: self.copy()

    # Creates an iterable from an iterator, making sure the shape matches.
    def to_iterable(self, other, force=False):
        if not issubclass(type(other), collections.abc.Sequence) or issubclass(type(other), collections.abc.Mapping):
            try:
                other = list(other)
            except TypeError:
                other = [other]
        if len(other) not in (1, self.size) and not force:
            raise IndexError(f"Unable to perform operation on objects with size {self.size} and {len(other)}.")
        return other

    @blocking
    def clear(self):
        self.size = 0
        if self.data is not None:
            self.offs = len(self.data) // 3
        else:
            self.offs = 0
        return self

    @waiting
    def copy(self):
        return self.__class__(self.view)

    @waiting
    def sort(self, *args, **kwargs):
        return self.__class__(sorted(self.view, *args, **kwargs))

    @waiting
    def shuffle(self, *args, **kwargs):
        return self.__class__(shuffle(self.view, *args, **kwargs))

    @waiting
    def reverse(self):
        return self.__class__(np.flip(self.view))

    # Rotates the list a certain amount of steps, using np.roll for large rotate operations.
    @blocking
    def rotate(self, steps):
        s = self.size
        if not s:
            return self
        steps %= s
        if steps > s >> 1:
            steps -= s
        if abs(steps) < self.minsize:
            while steps > 0:
                self.appendleft(self.popright(force=True), force=True)
                steps -= 1
            while steps < 0:
                self.appendright(self.popleft(force=True), force=True)
                steps += 1
            return self
        self.offs = (len(self.data) - self.size) // 3
        self.view[:] = np.roll(self.view, steps)
        return self

    @blocking
    def rotateleft(self, steps):
        return self.rotate(-steps, force=True)

    rotateright = rotate

    # Re-initializes the list if the positional offsets are too large or if the list is empty.
    @blocking
    def isempty(self):
        if self.size:
            if abs(len(self.data) // 3 - self.offs) > self.maxoff:
                self.reconstitute(force=True)
            return False
        if len(self.data) > 4096:
            self.data = None
            self.offs = 0
        elif self.data is not None:
            self.offs = len(self.data) // 3
        return True

    # For compatibility with dict.get
    @waiting
    def get(self, key, default=None):
        try:
            return self[key]
        except LookupError:
            return default

    @blocking
    def popleft(self):
        temp = self.data[self.offs]
        self.offs += 1
        self.size -= 1
        self.isempty(force=True)
        return temp

    @blocking
    def popright(self):
        temp = self.data[self.offs + self.size - 1]
        self.size -= 1
        self.isempty(force=True)
        return temp

    # Removes an item from the list. O(n) time complexity.
    @blocking
    def pop(self, index=None, *args):
        try:
            if index is None:
                return self.popright(force=True)
            if index >= len(self.data):
                return self.popright(force=True)
            elif index == 0:
                return self.popleft(force=True)
            index %= self.size
            temp = self.data[index + self.offs]
            if index > self.size >> 1:
                self.view[index:-1] = self.data[self.offs + index + 1:self.offs + self.size]
            else:
                self.view[1:index + 1] = self.data[self.offs:self.offs + index]
                self.offs += 1
            self.size -= 1
            return temp
        except LookupError:
            if not args:
                raise
            return args[0]

    # Inserts an item into the list. O(n) time complexity.
    @blocking
    def insert(self, index, value):
        if self.data is None:
            self.__init__((value,))
            return self
        if index >= self.size:
            return self.append(value, force=True)
        elif index == 0:
            return self.appendleft(value, force=True)
        index %= self.size
        if index > self.size >> 1:
            if self.size + self.offs + 1 >= len(self.data):
                self.reconstitute(force=True)
            self.size += 1
            self.view[index + 1:] = self.view[index:-1]
        else:
            if self.offs < 1:
                self.reconstitute(force=True)
            self.size += 1
            self.offs -= 1
            self.view[:index] = self.view[1:index + 1]
        self.view[index] = value
        return self

    # Insertion sort using a binary search to find target position. O(n) time complexity.
    @blocking
    def insort(self, value, key=None, sorted=True):
        if self.data is None:
            self.__init__((value,))
            return self
        if not sorted:
            self.__init__(sorted(self, key=key))
        if key is None:
            return self.insert(np.searchsorted(self.view, value), value, force=True)
        v = key(value)
        x = self.size
        index = (x >> 1) + self.offs
        gap = 3 + x >> 2
        seen = {}
        d = self.data
        while index not in seen and index >= self.offs and index < self.offs + self.size:
            check = key(d[index])
            if check < v:
                seen[index] = True
                index += gap
            else:
                seen[index] = False
                index -= gap
            gap = 1 + gap >> 1
        index -= self.offs - seen.get(index, 0)
        if index <= 0:
            return self.appendleft(value, force=True)
        return self.insert(index, value, force=True)

    # Removes all instances of a certain value from the list.
    @blocking
    def remove(self, value, key=None, sorted=False):
        pops = self.search(value, key, sorted, force=True)
        if pops:
            self.pops(pops, force=True)
        return self

    discard = remove

    # Removes all duplicate values from the list.
    @blocking
    def removedups(self, sort=True):
        if self.data is not None:
            if sort:
                try:
                    temp = np.unique(self.view)
                except:
                    temp = sorted(set(self.view))
            elif sort is None:
                temp = tuple(set(self.view))
            else:
                temp = deque()
                found = set()
                for x in self.view:
                    y = x
                    if isinstance(y, collections.abc.MutableSequence):
                        y = tuple(y)
                    if y not in temp:
                        found.add(y)
                        temp.append(x)
            self.size = len(temp)
            self.offs = (len(self.data) - self.size) // 3
            self.view[:] = temp
        return self

    uniq = unique = removedups

    # Returns first matching value in list.
    @waiting
    def index(self, value, key=None, sorted=False):
        return self.search(value, key, sorted, force=True)[0]

    # Returns last matching value in list.
    @waiting
    def rindex(self, value, key=None, sorted=False):
        return self.search(value, key, sorted, force=True)[-1]

    # Returns indices representing positions for all instances of the target found in list, using binary search when applicable.
    @waiting
    def search(self, value, key=None, sorted=False):
        if key is None:
            if sorted and self.size > self.minsize:
                i = np.searchsorted(self.view, value)
                if self.view[i] != value:
                    raise IndexError(f"{value} not found.")
                pops = self.__class__()
                pops.append(i)
                for x in range(i + self.offs - 1, -1, -1):
                    if self.data[x] == value:
                        pops.appendleft(x - self.offs)
                    else:
                        break
                for x in range(i + self.offs + 1, self.offs + self.size):
                    if self.data[x] == value:
                        pops.append(x - self.offs)
                    else:
                        break
                return pops
            else:
                return self.__class__(np.arange(self.size, dtype=np.uint32)[self.view == value])
        if sorted:
            v = value
            d = self.data
            pops = self.__class__()
            x = len(d)
            index = (x >> 1) + self.offs
            gap = x >> 2
            seen = {}
            while index not in seen and index >= self.offs and index < self.offs + self.size:
                check = key(d[index])
                if check < v:
                    seen[index] = True
                    index += gap
                elif check == v:
                    break
                else:
                    seen[index] = False
                    index -= gap
                gap = 1 + gap >> 1
            i = index + seen.get(index, 0)
            while i in d and key(d[i]) == v:
                pops.append(i - self.offs)
                i += 1
            i = index + seen.get(index, 0) - 1
            while i in d and key(d[i]) == v:
                pops.append(i - self.offs)
                i -= 1
        else:
            pops = self.__class__(i for i, x in enumerate(self.view) if key(x) == value)
        if not pops:
            raise IndexError(f"{value} not found.")
        return pops

    find = findall = search

    # Counts the amount of instances of the target within the list.
    @waiting
    def count(self, value, key=None):
        if key is None:
            return sum(self.view == value)
        return sum(1 for i in self if key(i) == value)

    concat = lambda self, value: self.__class__(np.concatenate([self.view, value]))

    # Appends item at the start of the list, reallocating when necessary.
    @blocking
    def appendleft(self, value):
        if self.data is None:
            self.__init__((value,))
            return self
        if self.offs <= 0:
            self.reconstitute(force=True)
        self.offs -= 1
        self.size += 1
        self.data[self.offs] = value
        return self

    # Appends item at the end of the list, reallocating when necessary.
    @blocking
    def append(self, value):
        if self.data is None:
            self.__init__((value,))
            return self
        if self.offs + self.size >= len(self.data):
            self.reconstitute(force=True)
        self.data[self.offs + self.size] = value
        self.size += 1
        return self

    add = lambda self, value: object.__getattribute__(self, ("append", "appendleft")[random.randint(0, 1)])(value)
    appendright = append

    # Appends iterable at the start of the list, reallocating when necessary.
    @blocking
    def extendleft(self, value):
        if self.data is None or not self.size:
            self.__init__(reversed(value))
            return self
        value = self.to_iterable(reversed(value), force=True)
        if self.offs >= len(value):
            self.data[self.offs - len(value):self.offs] = value
            self.offs -= len(value)
            self.size += len(value)
            return self
        self.__init__(np.concatenate([value, self.view]))
        return self

    # Appends iterable at the end of the list, reallocating when necessary.
    @blocking
    def extend(self, value):
        if self.data is None or not self.size:
            self.__init__(value)
            return self
        value = self.to_iterable(value, force=True)
        if len(self.data) - self.offs - self.size >= len(value):
            self.data[self.offs + self.size:self.offs + self.size + len(value)] = value
            self.size += len(value)
            return self
        self.__init__(np.concatenate([self.view, value]))
        return self

    extendright = extend

    # Similar to str.join
    @waiting
    def join(self, iterable):
        iterable = self.to_iterable(iterable)
        temp = deque()
        for i, v in enumerate(iterable):
            try:
                temp.extend(v)
            except TypeError:
                temp.append(v)
            if i != len(iterable) - 1:
                temp.extend(self.view)
        return self.__class__(temp)

    # Similar to str.replace().
    @blocking
    def replace(self, original, new):
        view = self.view
        for i, v in enumerate(view):
            if v == original:
                view[i] = new
        return self

    # Fills list with value(s).
    @blocking
    def fill(self, value):
        self.offs = (len(self.data) - self.size) // 3
        self.view[:] = value
        return self

    # For compatibility with dict() attributes.
    keys = lambda self: range(len(self))
    values = lambda self: iter(self)
    items = lambda self: enumerate(self)

    # For compatibility with set() attributes.
    @waiting
    def isdisjoint(self, other):
        if type(other) not in (set, frozenset):
            other = frozenset(other)
        return self.to_frozenset().isdisjoint(other)

    @waiting
    def issubset(self, other):
        if type(other) not in (set, frozenset):
            other = frozenset(other)
        return self.to_frozenset().issubset(other)

    @waiting
    def issuperset(self, other):
        if type(other) not in (set, frozenset):
            other = frozenset(other)
        return self.to_frozenset().issuperset(other)

    @waiting
    def union(self, *others):
        args = deque()
        for other in others:
            if type(other) not in (set, frozenset):
                other = frozenset(other)
            args.append(other)
        return self.to_frozenset().union(*args)

    @waiting
    def intersection(self, *others):
        args = deque()
        for other in others:
            if type(other) not in (set, frozenset):
                other = frozenset(other)
            args.append(other)
        return self.to_frozenset().intersection(*args)

    @waiting
    def difference(self, *others):
        args = deque()
        for other in others:
            if type(other) not in (set, frozenset):
                other = frozenset(other)
            args.append(other)
        return self.to_frozenset().difference(*args)

    @waiting
    def symmetric_difference(self, other):
        if type(other) not in (set, frozenset):
            other = frozenset(other)
        return self.to_frozenset().symmetric_difference(other)

    @blocking
    def update(self, *others, uniq=True):
        for other in others:
            if issubclass(other, collections.abc.Mapping):
                other = other.values()
            self.extend(other, force=True)
        if uniq:
            self.uniq(False, force=True)
        return self

    @blocking
    def intersection_update(self, *others, uniq=True):
        pops = set()
        for other in others:
            if issubclass(other, collections.abc.Mapping):
                other = other.values()
            if type(other) not in (set, frozenset):
                other = frozenset(other)
            for i, v in enumerate(self):
                if v not in other:
                    pops.add(i)
        self.pops(pops)
        if uniq:
            self.uniq(False, force=True)
        return self

    @blocking
    def difference_update(self, *others, uniq=True):
        pops = set()
        for other in others:
            if issubclass(other, collections.abc.Mapping):
                other = other.values()
            if type(other) not in (set, frozenset):
                other = frozenset(other)
            for i, v in enumerate(self):
                if v in other:
                    pops.add(i)
        self.pops(pops)
        if uniq:
            self.uniq(False, force=True)
        return self

    @blocking
    def symmetric_difference_update(self, other):
        data = set(self)
        if issubclass(other, collections.abc.Mapping):
            other = other.values()
        if type(other) not in (set, frozenset):
            other = frozenset(other)
        data.symmetric_difference_update(other)
        self.__init__(data)
        return self

    # Clips all values in list to input boundaries.
    @blocking
    def clip(self, a, b=None):
        if b is None:
            b = -a
        if a > b:
            a, b = b, a
        arr = self.view
        np.clip(arr, a, b, out=arr)
        return self

    # Casting values to various types.

    @waiting
    def real(self):
        return self.__class__(np.real(self.view))

    @waiting
    def imag(self):
        return self.__class__(np.imag(self.view))

    @waiting
    def float(self):
        return self.__class__(float(i.real) for i in self.view)

    @waiting
    def complex(self):
        return self.__class__(complex(i) for i in self.view)

    @waiting
    def mpf(self):
        return self.__class__(mpf(i.real) for i in self.view)

    @waiting
    def sum(self):
        return np.sum(self.view)

    @waiting
    def mean(self):
        return np.mean(self.view)

    @waiting
    def product(self):
        return np.prod(self.view)

    prod = product

    # Reallocates list.
    @blocking
    def reconstitute(self, data=None):
        self.__init__(data if data is not None else self.view, fromarray=False)
        return self

    # Removes items according to an array of indices.
    @blocking
    def delitems(self, iterable):
        iterable = self.to_iterable(iterable, force=True)
        if len(iterable) < 1:
            return self
        if len(iterable) == 1:
            return self.pop(iterable[0], force=True)
        temp = np.delete(self.view, np.asarray(iterable, dtype=np.int32))
        self.size = len(temp)
        if self.data is not None:
            self.offs = (len(self.data) - self.size) // 3
            self.view[:] = temp
        else:
            self.reconstitute(temp, force=True)
        return self

    pops = delitems

hlist = alist
arange = lambda *args, **kwargs: alist(range(*args, **kwargs))
afull = lambda size, n=0: alist(repeat(n, size))
azero = lambda size: alist(repeat(0, size))
