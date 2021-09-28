# Included:

## alist: a numpy-based list-like data type

### Features
- Mostly stable multithreaded support
- Pickle library support
- Allows `a[b]`, `a[b:c]`, `a[[b, c]]` etc indexing as numpy arrays do
- Treats numerical operations such as `+`, `-`, `*`, `/`, `**`, `&`, `%` etc as numpy operations, with appropriately in-place modifications when applicable
- Is a proper subclass of `collections.abc.MutableSequence`, meaning `len`, `bool`, `iter` and the like are all supported.
- Directly supports floating point indexing, in which case will linearly interpolate between adjacent elements when possible.
- Automatically allocates empty buffers on either side of the list, in order to greatly improve efficiency of appending/removing from it.
- Automatically creates a frozenset copy of itself when applicable to improve efficiency of consequent `x in y` and related operations
```py
>>> A = alist((5, 4, 3))
>>> A
[5, 4, 3]
>>> A[1]
4
>>> A[-1]
3
>>> A[:2]
[5, 4]
>>> A[[0, 2]]
[5, 3]
>>> A[[False, True, True]]
[4, 3]
>>> A[1.25]
3.75
>>> A + [1, 2, 3]
[6, 6, 6]
>>> A * 2
[10, 8, 6]
>>> A.concat([2, 1])
[5, 4, 3, 2, 1]
>>> A
[5, 4, 3]
>>> A[0] = 100
>>> A
[100, 4, 3]
>>> A[:2] = -1
>>> A
[-1, -1, 3]
>>> A[[1, 2]] = (64, 4096)
>>> A
[-1, 64, 4096]
>>> A[[True, False, True]] = [49, 81]
>>> A
[49, 64, 81]
>>> A[0.75] = 0
>>> A
[36.75, 16, 81]
```
- Note: `__eq__` and `__ne__` (`==` and `!=`) will act like the python `list` counterparts, not the numpy versions:
```py
>>> A == [36.75, 16, 81]
True
>>> A != "a"
True
```
- To use the numpy variants that return an array of booleans, see `.eq` below.
### Methods
Note: methods highlighted in **bold** can potentially mutate (modify) the contents of the list.
- *@property* `.view` => `numpy.ndarray` O(1): Returns the numpy array representing the contents of the list. `dtype` `object`.
- *@property* `.data` => `numpy.ndarray` O(1): Returns the numpy array representing the entire list buffer, including the data outside the current list.
- *@property* `.offs` => `int` O(1): Returns the index relative to the list buffer where the current list data begins.
- *@property* `.size` => `int` O(1): Returns the length of the valid part of the list.
- `.next()` => `object` O(1): Returns the next item in the list, starting from 0. The state of this is kept individually per list, and cycles back to the beginning after reaching the end.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.next()`<br>`5`<br>`>>> A.next()`<br>`4`<br>`>>> A.next()`<br>`3`<br>`>>> A.next()`<br>`5`
- **`.clear()`** => `alist` O(1): Empties the list, removing all of its contents, then returning itself.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.clear()`<br>`[]`
- `.copy(deep=False)` => `alist` O(n): Creates a copy of the list (with the `deep` argument indicating to copy recursively), returning it.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> B = A.copy()`<br>`>>> B`<br>`[5, 4, 3]`
- **`.sort(*args, **kwargs)`** => `alist` O(n*log(n)): Sorts the list according to the python builtin [`sorted`](https://docs.python.org/3/library/functions.html#sorted) function, returning itself.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.sort()`<br>`[3, 4, 5]`
- **`.shuffle(*args, **kwargs)`** => `alist` O(n): Shuffles the list according to the python builtin [`random.shuffle`](https://docs.python.org/3/library/random.html#random.shuffle) function, returning itself.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.shuffle()`<br>`[4, 5, 3]`
- **`.reverse()`** => `alist` O(1): Reverses the list using numpy's [`numpy.flip`](https://numpy.org/doc/stable/reference/generated/numpy.flip.html) function, returning itself.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.reverse()`<br>`[3, 4, 5]`
- **`.rotate(steps)`** **`.rotateright`** => `alist` O(n): Rotates the list to the right (or left if `steps` is negative), by repeatedly popping from one end of the list and appending to the other, switching to [`numpy.roll`](https://numpy.org/doc/stable/reference/generated/numpy.roll.html) if 9 or more steps would be required.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.rotate(1)`<br>`[3, 5, 4]`
- **`.rotateleft(steps)`** => `alist` O(n): The reverse of `.rotate`, rotating the list in the opposite direction.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.rotateleft(1)`<br>`[4, 3, 5]`
- `.get(key, default=None)` => `object` O(1): Fetches an entry in the list using `key` as index, returning `default` if not found. Works similarly to [`dict.get`](https://docs.python.org/3/library/stdtypes.html#dict.get).
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.get(4, 100)`<br>`100`
- **`.popleft()`** => `object` O(1): Pops the leftmost (index 0) entry in the list and returns it.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.popleft()`<br>`5`<br>`>>> A`<br>`[4, 3]`
- **`.popright()`** => `object` O(1): Pops the rightmost (index -1) entry in the list and returns it.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.popright()`<br>`3`<br>`>>> A`<br>`[5, 4]`
- **`.pop(index=None)`** => `object` O(n): Pops an entry at the specified index, returning it, and also shifts the smaller half of the list to close the gap. Falls back to `.popleft`/`popright` when removing from either end of the list; most computationally expensive to pop the middle element.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.pop(1)`<br>`4`<br>`>>> A`<br>`[5, 3]`
- **`.insert(index, value)`** => `alist` O(n): Inserts the specified value before the designated index, similarly to [`list.insert`](https://docs.python.org/3/tutorial/datastructures.html#list.insert), shifting the smaller half of the list to create the space, and reallocating the list to 3x its original size if no buffer space is available at the closest end. Returns the resulting list.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.insert(1, 100)`<br>`[5, 100, 4, 3]`
- **`.insort(value, key=None, sort=True)`** => `alist` O(n): Similar to `.insert`, but performs insertion sort to choose the appropriate position to insert the value. `key` is used in the same way as it is in [`sorted`](https://docs.python.org/3/library/functions.html#sorted), and the `sort` parameter indicates whether the entire list is already sorted prior to insertion; set to `False` to sort the entire list during insertion; this increases the time complexity to O(n*log(n)). Returns the resulting list.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.insort(4.5, sort=False)`<br>`[3, 4, 4.5, 5]`
- **`.remove(value, count=None, key=None, sort=False, last=False)`** **`.discard`** => `alist` O(n): Removes up to `count` instances of the target from the list, `key` being an optional identifier function called on the list's values to match to `value`, `sort` indicating whether or not the list is already sorted (set to `True` for already sorted lists to reduce computation time slightly), and `last` indicating whether to remove starting from the end of the list instead of the beginning, only relevant if `count` is set. Returns the resulting list.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.remove(5)`<br>`[4, 3]`
- **`.removedups(sort=True)`** **`uniq`** **`unique`** => `alist` O(n*log(n)): Removes all duplicate items in the list according to the `==` operation, with an optional `sort` argument indicating whether to sort the list simultaneously
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.append(4)`<br>`[5, 4, 3, 4]`<br>`>>> A.uniq(sort=False)`<br>`[5, 4, 3]`
- `.index(value, key=None, sort=False)` => `int` O(n): Searches for the index of the first entry in the list that matches the specified value similarly to [`list.index`](https://docs.python.org/3/tutorial/datastructures.html#list.index), with an optional function `key` to be used on the entries in the list. Pass `sort=True` to use binary search to improve computational efficiency to O(log(n)) for lists that are already sorted.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.index(4)`<br>`1`
- `.rindex(value, key=None, sort=False)` => `int` O(n): The [`str.rindex`](https://docs.python.org/3/library/stdtypes.html#str.rindex) counterpart of `.index`, returns the first index from the right side rather than the left.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.insort(4, sort=False)`<br>`[3, 4, 4, 5]`<br>`>>> A.rindex(4, sort=True)`<br>`2`
- `.search(value, key=None, sort=False)` `find` `findall` => `alist` O(n): Similar to `.index` and `.rindex`, but returns a list of indices of all the matching elements in the list.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.insert(3, 5)`<br>`[5, 4, 3, 5]`<br>`>>> A.appendleft("5")`<br>`['5', 5, 4, 3, 5]`<br>`>>> A.search(5, key=lambda x: int(x))`<br>`[0, 1, 4]`
- `.count(value, key=None)` => `int` O(n): Counts the amount of instances of `value` in the list, with optional identifier function `key`.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.count(1, key=lambda x: x >= 4)`<br>`2`
- `.concat(value)` => `alist` O(n+k): Returns a copy of the list with the target list concatenated to the end. Does not modify either lists.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.concat([2, 1])`<br>`[5, 4, 3, 2, 1]`
- **`.appendleft(value)`** => `alist` O(1): Appends the specified value into the front of the list. Uses the buffer to lower time complexity to O(1), re-allocating the list for O(n) when necessary.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.appendleft(6)`<br>`[6, 5, 4, 3]`
- **`.append(value)`** **`appendright`** => `alist` O(1): Appends the specified value to the end of the list. Uses the buffer to lower time complexity to O(1), re-allocating the list for O(n) when necessary.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.append({2})`<br>`[5, 4, 3, {2}]`
- **`.add(value)`** => `alist` O(1): Adds the specified value to the list, in a random direction, similar to [`set.add`](https://docs.python.org/3/library/stdtypes.html#frozenset.add). Can potentially save on buffer re-allocations compared to `appendleft` and `append`.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.add({(9, 8): [7, 6]})`<br>`[{(9, 8): [7, 6]}, 5, 4, 3]`
- **`.extendleft(value)`** => `alist` O(n+k): Extends the list to the left using elements from iterable `value`, similarly to [`collections.deque.extendleft`](https://docs.python.org/3/library/collections.html#collections.deque.extendleft).
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.extendleft(range(6, 10))`<br>`[9, 8, 7, 6, 5, 4, 3]`
- **`.extend(value)`** **`extendright`** => `alist` O(n+k): Extends the list to the right using elements from iterable `value`.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.extend(("2", b"1"))`<br>`[5, 4, 3, '2', b'1']`
- `.join(value)` => `alist` O(nk): Concatenates a copy of the list to every item in the specified iterable `value`, similar to [`str.join`](https://docs.python.org/3/library/stdtypes.html#str.join).
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.join({"a", "b", "c"})`<br>`['a', 5, 4, 3, 'b', 5, 4, 3, 'c']`
- **`.replace(original, new)`** => `alist` O(n): Replaces elements in the list matching `original` with `new`, similar to [`str.replace`](https://docs.python.org/3/library/stdtypes.html#str.replace).
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.replace(3, "3")`<br>`[5, 4, '3']`
- **`.strip(*values)`** => `alist` O(n): Removes all trailing elements matching any item in `values`, similar to [`str.strip`](https://docs.python.org/3/library/stdtypes.html#str.strip).
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.strip(4, 3)`<br>`[5]`
- **`.fill(value)`** => `alist` O(n): Fills the list with the specified value or iterable.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.fill(-1)`<br>`[-1, -1, -1]`<br>`>>> A.fill([1, 2, 3, 4])`<br>`[1, 2, 3, 4]`
- `.keys()` => `range` O(1): Returns a range object representing the list's indices, similar to [`dict.keys`](https://docs.python.org/3/library/stdtypes.html#dict.keys)
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.keys()`<br>`range(0, 3)`
- `.values()` => `iter` O(1): Returns an iterator, similar to [`dict.values`](https://docs.python.org/3/library/stdtypes.html#dict.values). Functionally identical to calling [`iter`](https://docs.python.org/3/library/functions.html#iter) on the list.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.values()`<br>`<iterator object at 0x0000000000000000>`
- `.items()` => `enumerate` O(1): Returns an iterator, similar to [`dict.items`](https://docs.python.org/3/library/stdtypes.html#dict.values). Functionally identical to calling [`enumerate`](https://docs.python.org/3/library/functions.html#iter) on the list.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.items()`<br>`<enumerate object at 0x0000000000000000>`
- `.eq(other)` => `numpy.ndarray` O(n): Returns a bool numpy array representing the typical `==` operation for numpy arrays.
- `.ne(other)` => `numpy.ndarray` O(n): Returns a bool numpy array representing the typical `!=` operation for numpy arrays.
- `.isdisjoint(other)` => `bool` O(n): Performs [`set.isdisjoint`](https://docs.python.org/3/library/stdtypes.html#frozenset.isdisjoint) on the list. Caches the resulting copied set of the list for future set operations, until the list is modified.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.isdisjoint({6, 7, 8})`<br>`True`
- `.issubset(other)` => `bool` O(n): Performs [`set.issubset`](https://docs.python.org/3/library/stdtypes.html#frozenset.issubset) on the list. Caches the resulting copied set of the list for future set operations, until the list is modified.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.issubset((1, 4, 3, 8, 5))`<br>`True`
- `.issuperset(other)` => `bool` O(n): Performs [`set.issuperset`](https://docs.python.org/3/library/stdtypes.html#frozenset.issuperset) on the list. Caches the resulting copied set of the list for future set operations, until the list is modified.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.issuperset([3, 5])`<br>`True`
- `.union(*others)` => `frozenset` O(n+k): Performs [`frozenset.union`](https://docs.python.org/3/library/stdtypes.html#frozenset.union) on the list. Caches the resulting copied set of the list for future set operations, until the list is modified.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.union({1, 2, 3}, [7, 8, 9], (6, 10))`<br>`frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9, 10})`
- `.intersection(*others)` => `frozenset` O(n+k): Performs [`frozenset.intersection`](https://docs.python.org/3/library/stdtypes.html#frozenset.intersection) on the list. Caches the resulting copied set of the list for future set operations, until the list is modified.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.intersection([3, 4, 7, 8], {5: None, 4: True, 3: False, "a": "b"})`<br>`frozenset({3, 4})`
- `.difference(*others)` => `frozenset` O(n+k): Performs [`frozenset.difference`](https://docs.python.org/3/library/stdtypes.html#frozenset.difference) on the list. Caches the resulting copied set of the list for future set operations, until the list is modified.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.difference((1, 2, 3), (5, 6, 7))`<br>`frozenset({4})`
- `.symmetric_difference(other)` => `frozenset` O(n+k): Performs [`frozenset.symmetric_difference`](https://docs.python.org/3/library/stdtypes.html#frozenset.symmetric_difference) on the list. Caches the resulting copied set of the list for future set operations, until the list is modified.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.symmetric_difference([1, 2, 3, 4, 5, 6, 7])`<br>`frozenset({1, 2, 6, 7})`
- **`.update(*others, uniq=True)`** => `alist` O(n+k): Performs [`frozenset.update`](https://docs.python.org/3/library/stdtypes.html#frozenset.update) on the list. Note: This operation will mutate (modify) the contents of the original list.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.update([1, 2, 3], {5, 6, 7})`<br>`[1, 2, 3, 4, 5, 6, 7]`
- **`.intersection_update`** **`difference_update`** **`symmetric_difference_update`** Implementations of the corresponding set operations; all of these modify the original list.
- **`.clip(a, b=None)`** => `alist` O(n): Performs [`numpy.clip`](https://numpy.org/doc/stable/reference/generated/numpy.clip.html) on the list in-place, with the range defaulting to `[-a, a]` if `b` is not supplied.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.clip(1, 4)`<br>`[4, 4, 3]`
- `.real()` => `alist` O(n): Returns a copy of the list with all values cast to real numbers, as in [`numpy.real`](https://numpy.org/doc/stable/reference/generated/numpy.real.html).
- `.imag` `.float` `.complex` Implementations of the other casting functions on the array.
- `.sum()` => `float` O(n): Returns the sum of the items in the list.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.sum()`<br>`12`
- `.mean()` => `float` O(n): Returns the mean of the items in the list.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.mean()`<br>`4`
- `.product()` `prod` => `float` O(n): Returns the product of the items in the list.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.product()`<br>`60`
- **`.delitems(iterable, keep=False)`** **`pops`** => `alist` O(n): Performs [`numpy.delete`](https://numpy.org/doc/stable/reference/generated/numpy.delete.html), removing all the items in the list according to the indices supplied in `iterable`, and replaces the contents of the list with the result. Returns the resulting list if `keep` is `False`, otherwise a new list containing the removed elements. Falls back to `.pop` when only removing 1 element, in order to optimise where possible.
  - Example:<br>`>>> A = alist((5, 4, 3))`<br>`>>> A.pops([0, 2])`<br>`[4]`
