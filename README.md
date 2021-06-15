# Included:

## alist: a numpy-based list-like data type
- Mostly stable multithreaded support
- Pickle library support
- Allows `a[b]`, `a[b:c]`, `a[[b, c]]` etc indexing as numpy arrays do
- Treats numerical operations such as `+`, `-`, `*`, `/`, `**`, `&`, `%` etc as numpy operations, with appropriately in-place modifications when applicable
- Is a proper subclass of `collections.abc.MutableSequence`, meaning `len`, `bool`, `iter` and the like are all supported.
- Directly supports floating point indexing, in which case will linearly interpolate between adjacent elements when possible.
- Automatically creates a frozenset copy of itself when applicable to improve efficiency of `x in y` and related operations

### Methods
- `.next()` => `object` O(1): Returns the next item in the list, starting from 0. The state of this is kept individually per list, and cycles back to the beginning after reaching the end.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.next()`<br>`5`<br>`>>> A.next()`<br>`4`<br>`>>> A.next()`<br>`3`<br>`>>> A.next()`<br>`5`
- `.clear()` => `alist` O(1): Empties the list, removing all of its contents, then returning itself.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.clear()`<br>`[]`
- `.copy(deep=False)` => `alist` O(n): Creates a copy of the list (with the `deep` argument indicating to copy recursively), returning itself.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> B = A.copy()`<br>`>>> B`<br>`[5, 4, 3]`
- `.sort(*args, **kwargs)` => `alist` O(n*log(n)): Sorts the list according to the python builtin [`sorted`](https://docs.python.org/3/library/functions.html#sorted) function, returning itself.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.sort()`<br>`[3, 4, 5]`
- `.shuffle(*args, **kwargs)` => `alist` O(n): Shuffles the list according to the python builtin [`random.shuffle`](https://docs.python.org/3/library/random.html#random.shuffle) function, returning itself.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.shuffle()`<br>`[4, 5, 3]`
- `.reverse()` => `alist` O(1): Reverses the list using numpy's [`numpy.flip`](https://numpy.org/doc/stable/reference/generated/numpy.flip.html) function, returning itself.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.reverse()`<br>`[3, 4, 5]`
- `.rotate(steps)` `.rotateright` => `alist` O(n): Rotates the list to the right (or left if `steps` is negative), by repeatedly popping from one end of the list and appending to the other, switching to [`numpy.roll`](https://numpy.org/doc/stable/reference/generated/numpy.roll.html) if 9 or more steps would be required.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.rotate(1)`<br>`[3, 5, 4]`
- `.rotateleft(steps)` => `alist` O(n): The reverse of `.rotate`, rotating the list in the opposite direction.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.rotateleft(1)`<br>`[4, 3, 5]`
- `.isempty()` => `bool` O(n): Determines if the list is empty. O(1) under normal conditions, but will automatically re-allocate the list if there is too much wasted space in it due to other operations.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.isempty()`<br>`False`
- `.get(key, default=None)` => `object` O(1): Fetches an entry in the list using `key` as index, returning `default` if not found. Works similarly to [`dict.get`](https://docs.python.org/3/library/stdtypes.html#dict.get).
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.get(4, 100)`<br>`100`
- `.popleft()` => `object` O(1): Pops the leftmost (index 0) entry in the list and returns it.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.popleft()`<br>`5`<br>`>>> A`<br>`[4, 3]`
- `.popright()` => `object` O(1): Pops the rightmost (index -1) entry in the list and returns it.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.popright()`<br>`3`<br>`>>> A`<br>`[5, 4]`
- `.pop(index=None)` => `object` O(n): Pops an entry at the specified index, returning it, and also shifts the smaller half of the list to close the gap. Most computationally expensive to pop the middle element.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.pop(1)`<br>`4`<br>`>>> A`<br>`[5, 3]`
- `.insert(index, value)` => `alist` O(n): Inserts the specified value before the designated index, similarly to [`list.insert`](https://docs.python.org/3/tutorial/datastructures.html#list.insert), shifting the smaller half of the list to create the space, and reallocating the list to 3x its original size if no buffer space is available at the closest end. Returns the resulting list.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.insert(1, 100)`<br>`[5, 100, 4, 3]`
- `.insort(value, key=None, sort=True)` => `alist` O(n*log(n)): Similar to `.insert`, but performs insertion sort to choose the appropriate position to insert the value. `key` is used in the same way as it is in [`sorted`](https://docs.python.org/3/library/functions.html#sorted), and the `sort` parameter determines whether the entire list is to be sorted prior to insertion; set to `False` for already sorted lists to reduce computation time to O(n). Returns the resulting list.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.insort(4.5)`<br>`[3, 4, 4.5, 5]`
- `.remove(value, count=None, key=None, sort=False, last=False)` `.discard` => `alist` O(n): Removes up to `count` instances of the target from the list, `key` being an optional identifier function called on the list's values to match to `value`, `sort` indicating whether or not the list is already sorted (set to `True` for already sorted lists to reduce computation time slightly), and `last` indicating whether to remove starting from the end of the list instead of the beginning, only relevant if `count` is set. Returns the resulting list.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.remove(5)`<br>`[4, 3]`
- `.removedups(sort=True)` `uniq` `unique` => `alist` O(n*log(n)): Removes all duplicate items in the list according to the `==` operation, with an optional `sort` argument indicating whether to sort the list simultaneously. Can be close to O(n) in most circumstances.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.append(4)`<br>`[5, 4, 3, 4]`<br>`>>> A.uniq(sort=False)`<br>`[5, 4, 3]`
- `.index(value, key=None, sort=False)` => `int` O(n): Searches for the index of the first entry in the list that matches the specified value similarly to [`list.index`](https://docs.python.org/3/tutorial/datastructures.html#list.index), with an optional function `key` to be used on the entries in the list. Pass `sort=True` to use binary search to improve computational efficiency to O(log(n)) for lists that are already sorted.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.index(4)`<br>`1`
- `.rindex(value, key=None, sort=False)` => `int` O(n): The [`str.rindex`](https://docs.python.org/3/library/stdtypes.html#str.rindex) counterpart of `.index`, returns the first index from the right side rather than the left.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.insort(4)`<br>`[3, 4, 4, 5]`<br>`>>> A.rindex(4, sort=True)`<br>`2`
