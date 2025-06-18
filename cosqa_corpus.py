
def get_last_weekday_in_month(year, month, weekday):
        """Get the last weekday in a given month. e.g:

        >>> # the last monday in Jan 2013
        >>> Calendar.get_last_weekday_in_month(2013, 1, MON)
        datetime.date(2013, 1, 28)
        """
        day = date(year, month, monthrange(year, month)[1])
        while True:
            if day.weekday() == weekday:
                break
            day = day - timedelta(days=1)
        return day

def __rmatmul__(self, other):
        """
        Matrix multiplication using binary `@` operator in Python>=3.5.
        """
        return self.T.dot(np.transpose(other)).T

def file_exists(fname):
    """Check if a file exists and is non-empty.
    """
    try:
        return fname and os.path.exists(fname) and os.path.getsize(fname) > 0
    except OSError:
        return False

def strtobytes(input, encoding):
    """Take a str and transform it into a byte array."""
    py_version = sys.version_info[0]
    if py_version >= 3:
        return _strtobytes_py3(input, encoding)
    return _strtobytes_py2(input, encoding)

def has_synset(word: str) -> list:
    """" Returns a list of synsets of a word after lemmatization. """

    return wn.synsets(lemmatize(word, neverstem=True))

def read_set_from_file(filename: str) -> Set[str]:
    """
    Extract a de-duped collection (set) of text from a file.
    Expected file format is one item per line.
    """
    collection = set()
    with open(filename, 'r') as file_:
        for line in file_:
            collection.add(line.rstrip())
    return collection

def tsv_escape(x: Any) -> str:
    """
    Escape data for tab-separated value (TSV) format.
    """
    if x is None:
        return ""
    x = str(x)
    return x.replace("\t", "\\t").replace("\n", "\\n")

def _kbhit_unix() -> bool:
    """
    Under UNIX: is a keystroke available?
    """
    dr, dw, de = select.select([sys.stdin], [], [], 0)
    return dr != []

def remove_empty_text(utterances: List[Utterance]) -> List[Utterance]:
    """Remove empty utterances from a list of utterances
    Args:
        utterances: The list of utterance we are processing
    """
    return [utter for utter in utterances if utter.text.strip() != ""]

def get_system_drives():
    """
    Get the available drive names on the system. Always returns a list.
    """
    drives = []
    if os.name == 'nt':
        import ctypes
        bitmask = ctypes.windll.kernel32.GetLogicalDrives()
        letter = ord('A')
        while bitmask > 0:
            if bitmask & 1:
                name = chr(letter) + ':' + os.sep
                if os.path.isdir(name):
                    drives.append(name)
            bitmask >>= 1
            letter += 1
    else:
        current_drive = get_drive(os.getcwd())
        if current_drive:
            drive = current_drive
        else:
            drive = os.sep
        drives.append(drive)

    return drives

def area (self):
    """area() -> number

    Returns the area of this Polygon.
    """
    area = 0.0
    
    for segment in self.segments():
      area += ((segment.p.x * segment.q.y) - (segment.q.x * segment.p.y))/2

    return area

def is_rate_limited(response):
        """
        Checks if the response has been rate limited by CARTO APIs

        :param response: The response rate limited by CARTO APIs
        :type response: requests.models.Response class

        :return: Boolean
        """
        if (response.status_code == codes.too_many_requests and 'Retry-After' in response.headers and
                int(response.headers['Retry-After']) >= 0):
            return True

        return False

def convert_bytes_to_ints(in_bytes, num):
    """Convert a byte array into an integer array. The number of bytes forming an integer
    is defined by num

    :param in_bytes: the input bytes
    :param num: the number of bytes per int
    :return the integer array"""
    dt = numpy.dtype('>i' + str(num))
    return numpy.frombuffer(in_bytes, dt)

def last_location_of_minimum(x):
    """
    Returns the last location of the minimal value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    return 1.0 - np.argmin(x[::-1]) / len(x) if len(x) > 0 else np.NaN

def iprotate(l, steps=1):
    r"""Like rotate, but modifies `l` in-place.

    >>> l = [1,2,3]
    >>> iprotate(l) is l
    True
    >>> l
    [2, 3, 1]
    >>> iprotate(iprotate(l, 2), -3)
    [1, 2, 3]

    """
    if len(l):
        steps %= len(l)
        if steps:
            firstPart = l[:steps]
            del l[:steps]
            l.extend(firstPart)
    return l

def bytes_hack(buf):
    """
    Hacky workaround for old installs of the library on systems without python-future that were
    keeping the 2to3 update from working after auto-update.
    """
    ub = None
    if sys.version_info > (3,):
        ub = buf
    else:
        ub = bytes(buf)

    return ub

def is_integer(value: Any) -> bool:
    """Return true if a value is an integer number."""
    return (isinstance(value, int) and not isinstance(value, bool)) or (
        isinstance(value, float) and isfinite(value) and int(value) == value
    )

def split(text: str) -> List[str]:
    """Split a text into a list of tokens.

    :param text: the text to split
    :return: tokens
    """
    return [word for word in SEPARATOR.split(text) if word.strip(' \t')]

def remove_empty_text(utterances: List[Utterance]) -> List[Utterance]:
    """Remove empty utterances from a list of utterances
    Args:
        utterances: The list of utterance we are processing
    """
    return [utter for utter in utterances if utter.text.strip() != ""]

def to_int64(a):
    """Return view of the recarray with all int32 cast to int64."""
    # build new dtype and replace i4 --> i8
    def promote_i4(typestr):
        if typestr[1:] == 'i4':
            typestr = typestr[0]+'i8'
        return typestr

    dtype = [(name, promote_i4(typestr)) for name,typestr in a.dtype.descr]
    return a.astype(dtype)

def _run_sync(self, method: Callable, *args, **kwargs) -> Any:
        """
        Utility method to run commands synchronously for testing.
        """
        if self.loop.is_running():
            raise RuntimeError("Event loop is already running.")

        if not self.is_connected:
            self.loop.run_until_complete(self.connect())

        task = asyncio.Task(method(*args, **kwargs), loop=self.loop)
        result = self.loop.run_until_complete(task)

        self.loop.run_until_complete(self.quit())

        return result

def _duplicates(list_):
    """Return dict mapping item -> indices."""
    item_indices = {}
    for i, item in enumerate(list_):
        try:
            item_indices[item].append(i)
        except KeyError:  # First time seen
            item_indices[item] = [i]
    return item_indices

def valid_date(x: str) -> bool:
    """
    Retrun ``True`` if ``x`` is a valid YYYYMMDD date;
    otherwise return ``False``.
    """
    try:
        if x != dt.datetime.strptime(x, DATE_FORMAT).strftime(DATE_FORMAT):
            raise ValueError
        return True
    except ValueError:
        return False

def browse_dialog_dir():
    """
    Open up a GUI browse dialog window and let to user pick a target directory.
    :return str: Target directory path
    """
    _go_to_package()
    logger_directory.info("enter browse_dialog")
    _path_bytes = subprocess.check_output(['python', 'gui_dir_browse.py'], shell=False)
    _path = _fix_path_bytes(_path_bytes, file=False)
    if len(_path) >= 1:
        _path = _path[0]
    else:
        _path = ""
    logger_directory.info("chosen path: {}".format(_path))
    logger_directory.info("exit browse_dialog")
    return _path

def get_timezone() -> Tuple[datetime.tzinfo, str]:
    """Discover the current time zone and it's standard string representation (for source{d})."""
    dt = get_datetime_now().astimezone()
    tzstr = dt.strftime("%z")
    tzstr = tzstr[:-2] + ":" + tzstr[-2:]
    return dt.tzinfo, tzstr

def read32(bytestream):
  """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def interact(self, container: Container) -> None:
        """
        Connects to the PTY (pseudo-TTY) for a given container.
        Blocks until the user exits the PTY.
        """
        cmd = "/bin/bash -c 'source /.environment && /bin/bash'"
        cmd = "docker exec -it {} {}".format(container.id, cmd)
        subprocess.call(cmd, shell=True)

def lower_camel_case_from_underscores(string):
    """generate a lower-cased camelCase string from an underscore_string.
    For example: my_variable_name -> myVariableName"""
    components = string.split('_')
    string = components[0]
    for component in components[1:]:
        string += component[0].upper() + component[1:]
    return string

def psutil_phymem_usage():
    """
    Return physical memory usage (float)
    Requires the cross-platform psutil (>=v0.3) library
    (https://github.com/giampaolo/psutil)
    """
    import psutil
    # This is needed to avoid a deprecation warning error with
    # newer psutil versions
    try:
        percent = psutil.virtual_memory().percent
    except:
        percent = psutil.phymem_usage().percent
    return percent

def butlast(iterable):
    """Yield all items from ``iterable`` except the last one.

    >>> list(butlast(['spam', 'eggs', 'ham']))
    ['spam', 'eggs']

    >>> list(butlast(['spam']))
    []

    >>> list(butlast([]))
    []
    """
    iterable = iter(iterable)
    try:
        first = next(iterable)
    except StopIteration:
        return
    for second in iterable:
        yield first
        first = second

def is_natural(x):
    """A non-negative integer."""
    try:
        is_integer = int(x) == x
    except (TypeError, ValueError):
        return False
    return is_integer and x >= 0

def file_exists(self) -> bool:
        """ Check if the settings file exists or not """
        cfg_path = self.file_path
        assert cfg_path

        return path.isfile(cfg_path)

def _short_repr(obj):
  """Helper function returns a truncated repr() of an object."""
  stringified = pprint.saferepr(obj)
  if len(stringified) > 200:
    return '%s... (%d bytes)' % (stringified[:200], len(stringified))
  return stringified

def release_lock():
    """Release lock on compilation directory."""
    get_lock.n_lock -= 1
    assert get_lock.n_lock >= 0
    # Only really release lock once all lock requests have ended.
    if get_lock.lock_is_enabled and get_lock.n_lock == 0:
        get_lock.start_time = None
        get_lock.unlocker.unlock()

def remove_once(gset, elem):
    """Remove the element from a set, lists or dict.
    
        >>> L = ["Lucy"]; S = set(["Sky"]); D = { "Diamonds": True };
        >>> remove_once(L, "Lucy"); remove_once(S, "Sky"); remove_once(D, "Diamonds");
        >>> print L, S, D
        [] set([]) {}

    Returns the element if it was removed. Raises one of the exceptions in 
    :obj:`RemoveError` otherwise.
    """
    remove = getattr(gset, 'remove', None)
    if remove is not None: remove(elem)
    else: del gset[elem]
    return elem

def prevPlot(self):
        """Moves the displayed plot to the previous one"""
        if self.stacker.currentIndex() > 0:
            self.stacker.setCurrentIndex(self.stacker.currentIndex()-1)

def get_last_weekday_in_month(year, month, weekday):
        """Get the last weekday in a given month. e.g:

        >>> # the last monday in Jan 2013
        >>> Calendar.get_last_weekday_in_month(2013, 1, MON)
        datetime.date(2013, 1, 28)
        """
        day = date(year, month, monthrange(year, month)[1])
        while True:
            if day.weekday() == weekday:
                break
            day = day - timedelta(days=1)
        return day

def find_duplicates(l: list) -> set:
    """
    Return the duplicates in a list.

    The function relies on
    https://stackoverflow.com/questions/9835762/find-and-list-duplicates-in-a-list .
    Parameters
    ----------
    l : list
        Name

    Returns
    -------
    set
        Duplicated values

    >>> find_duplicates([1,2,3])
    set()
    >>> find_duplicates([1,2,1])
    {1}
    """
    return set([x for x in l if l.count(x) > 1])

def change_bgcolor_enable(self, state):
        """
        This is implementet so column min/max is only active when bgcolor is
        """
        self.dataModel.bgcolor(state)
        self.bgcolor_global.setEnabled(not self.is_series and state > 0)

def sorted_chain(*ranges: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Chain & sort ranges."""
    return sorted(itertools.chain(*ranges))

def csv_to_numpy(string_like, dtype=None):  # type: (str) -> np.array
    """Convert a CSV object to a numpy array.

    Args:
        string_like (str): CSV string.
        dtype (dtype, optional):  Data type of the resulting array. If None, the dtypes will be determined by the
                                        contents of each column, individually. This argument can only be used to
                                        'upcast' the array.  For downcasting, use the .astype(t) method.
    Returns:
        (np.array): numpy array
    """
    stream = StringIO(string_like)
    return np.genfromtxt(stream, dtype=dtype, delimiter=',')

def remove_once(gset, elem):
    """Remove the element from a set, lists or dict.
    
        >>> L = ["Lucy"]; S = set(["Sky"]); D = { "Diamonds": True };
        >>> remove_once(L, "Lucy"); remove_once(S, "Sky"); remove_once(D, "Diamonds");
        >>> print L, S, D
        [] set([]) {}

    Returns the element if it was removed. Raises one of the exceptions in 
    :obj:`RemoveError` otherwise.
    """
    remove = getattr(gset, 'remove', None)
    if remove is not None: remove(elem)
    else: del gset[elem]
    return elem

def _my_hash(arg_list):
    # type: (List[Any]) -> int
    """Simple helper hash function"""
    res = 0
    for arg in arg_list:
        res = res * 31 + hash(arg)
    return res

def most_significant_bit(lst: np.ndarray) -> int:
    """
    A helper function that finds the position of the most significant bit in a 1darray of 1s and 0s,
    i.e. the first position where a 1 appears, reading left to right.

    :param lst: a 1d array of 0s and 1s with at least one 1
    :return: the first position in lst that a 1 appears
    """
    return np.argwhere(np.asarray(lst) == 1)[0][0]

def _skip_section(self):
        """Skip a section"""
        self._last = self._f.readline()
        while len(self._last) > 0 and len(self._last[0].strip()) == 0:
            self._last = self._f.readline()

def get_column_names(engine: Engine, tablename: str) -> List[str]:
    """
    Get all the database column names for the specified table.
    """
    return [info.name for info in gen_columns_info(engine, tablename)]

def remove_empty_text(utterances: List[Utterance]) -> List[Utterance]:
    """Remove empty utterances from a list of utterances
    Args:
        utterances: The list of utterance we are processing
    """
    return [utter for utter in utterances if utter.text.strip() != ""]

def is_integer(value: Any) -> bool:
    """Return true if a value is an integer number."""
    return (isinstance(value, int) and not isinstance(value, bool)) or (
        isinstance(value, float) and isfinite(value) and int(value) == value
    )

def _check_whitespace(string):
    """
    Make sure thre is no whitespace in the given string. Will raise a
    ValueError if whitespace is detected
    """
    if string.count(' ') + string.count('\t') + string.count('\n') > 0:
        raise ValueError(INSTRUCTION_HAS_WHITESPACE)

def remove_nans_1D(*args) -> tuple:
    """Remove nans in a set of 1D arrays.

    Removes indicies in all arrays if any array is nan at that index.
    All input arrays must have the same size.

    Parameters
    ----------
    args : 1D arrays

    Returns
    -------
    tuple
        Tuple of 1D arrays in same order as given, with nan indicies removed.
    """
    vals = np.isnan(args[0])
    for a in args:
        vals |= np.isnan(a)
    return tuple(np.array(a)[~vals] for a in args)

def rate_limited(max_per_hour: int, *args: Any) -> Callable[..., Any]:
    """Rate limit a function."""
    return util.rate_limited(max_per_hour, *args)

def clean_map(obj: Mapping[Any, Any]) -> Mapping[Any, Any]:
    """
    Return a new copied dictionary without the keys with ``None`` values from
    the given Mapping object.
    """
    return {k: v for k, v in obj.items() if v is not None}

def get_pylint_options(config_dir='.'):
    # type: (str) -> List[str]
    """Checks for local config overrides for `pylint`
    and add them in the correct `pylint` `options` format.

    :param config_dir:
    :return: List [str]
    """
    if PYLINT_CONFIG_NAME in os.listdir(config_dir):
        pylint_config_path = PYLINT_CONFIG_NAME
    else:
        pylint_config_path = DEFAULT_PYLINT_CONFIG_PATH

    return ['--rcfile={}'.format(pylint_config_path)]

def prin(*args, **kwargs):
    r"""Like ``print``, but a function. I.e. prints out all arguments as
    ``print`` would do. Specify output stream like this::

      print('ERROR', `out="sys.stderr"``).

    """
    print >> kwargs.get('out',None), " ".join([str(arg) for arg in args])

def iter_lines(file_like: Iterable[str]) -> Generator[str, None, None]:
    """ Helper for iterating only nonempty lines without line breaks"""
    for line in file_like:
        line = line.rstrip('\r\n')
        if line:
            yield line

def url_host(url: str) -> str:
    """
    Parses hostname from URL.
    :param url: URL
    :return: hostname
    """
    from urllib.parse import urlparse
    res = urlparse(url)
    return res.netloc.split(':')[0] if res.netloc else ''

def validate_django_compatible_with_python():
    """
    Verify Django 1.11 is present if Python 2.7 is active

    Installation of pinax-cli requires the correct version of Django for
    the active Python version. If the developer subsequently changes
    the Python version the installed Django may no longer be compatible.
    """
    python_version = sys.version[:5]
    django_version = django.get_version()
    if sys.version_info == (2, 7) and django_version >= "2":
        click.BadArgumentUsage("Please install Django v1.11 for Python {}, or switch to Python >= v3.4".format(python_version))

def memory_usage():
    """return memory usage of python process in MB

    from
    http://fa.bianp.net/blog/2013/different-ways-to-get-memory-consumption-or-lessons-learned-from-memory_profiler/
    psutil is quicker

    >>> isinstance(memory_usage(),float)
    True

    """
    try:
        import psutil
        import os
    except ImportError:
        return _memory_usage_ps()

    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem

def _check_whitespace(string):
    """
    Make sure thre is no whitespace in the given string. Will raise a
    ValueError if whitespace is detected
    """
    if string.count(' ') + string.count('\t') + string.count('\n') > 0:
        raise ValueError(INSTRUCTION_HAS_WHITESPACE)

def recClearTag(element):
    """Applies maspy.xml.clearTag() to the tag attribute of the "element" and
    recursively to all child elements.

    :param element: an :instance:`xml.etree.Element`
    """
    children = element.getchildren()
    if len(children) > 0:
        for child in children:
            recClearTag(child)
    element.tag = clearTag(element.tag)

def position(self) -> Position:
        """The current position of the cursor."""
        return Position(self._index, self._lineno, self._col_offset)

def grep(pattern, filename):
    """Very simple grep that returns the first matching line in a file.
    String matching only, does not do REs as currently implemented.
    """
    try:
        # for line in file
        # if line matches pattern:
        #    return line
        return next((L for L in open(filename) if L.find(pattern) >= 0))
    except StopIteration:
        return ''

def find_column(token):
    """ Compute column:
            input is the input text string
            token is a token instance
    """
    i = token.lexpos
    input = token.lexer.lexdata

    while i > 0:
        if input[i - 1] == '\n':
            break
        i -= 1

    column = token.lexpos - i + 1

    return column

def file_uptodate(fname, cmp_fname):
    """Check if a file exists, is non-empty and is more recent than cmp_fname.
    """
    try:
        return (file_exists(fname) and file_exists(cmp_fname) and
                getmtime(fname) >= getmtime(cmp_fname))
    except OSError:
        return False

def numeric_part(s):
    """Returns the leading numeric part of a string.

    >>> numeric_part("20-alpha")
    20
    >>> numeric_part("foo")
    >>> numeric_part("16b")
    16
    """

    m = re_numeric_part.match(s)
    if m:
        return int(m.group(1))
    return None

def numchannels(samples:np.ndarray) -> int:
    """
    return the number of channels present in samples

    samples: a numpy array as returned by sndread

    for multichannel audio, samples is always interleaved,
    meaning that samples[n] returns always a frame, which
    is either a single scalar for mono audio, or an array
    for multichannel audio.
    """
    if len(samples.shape) == 1:
        return 1
    else:
        return samples.shape[1]

def exponential_backoff(attempt: int, cap: int=1200) -> timedelta:
    """Calculate a delay to retry using an exponential backoff algorithm.

    It is an exponential backoff with random jitter to prevent failures
    from being retried at the same time. It is a good fit for most
    applications.

    :arg attempt: the number of attempts made
    :arg cap: maximum delay, defaults to 20 minutes
    """
    base = 3
    temp = min(base * 2 ** attempt, cap)
    return timedelta(seconds=temp / 2 + random.randint(0, temp / 2))

def strtobytes(input, encoding):
    """Take a str and transform it into a byte array."""
    py_version = sys.version_info[0]
    if py_version >= 3:
        return _strtobytes_py3(input, encoding)
    return _strtobytes_py2(input, encoding)

def is_relative_url(url):
    """ simple method to determine if a url is relative or absolute """
    if url.startswith("#"):
        return None
    if url.find("://") > 0 or url.startswith("//"):
        # either 'http(s)://...' or '//cdn...' and therefore absolute
        return False
    return True

async def parallel_results(future_map: Sequence[Tuple]) -> Dict:
    """
    Run parallel execution of futures and return mapping of their results to the provided keys.
    Just a neat shortcut around ``asyncio.gather()``

    :param future_map: Keys to futures mapping, e.g.: ( ('nav', get_nav()), ('content, get_content()) )
    :return: Dict with futures results mapped to keys {'nav': {1:2}, 'content': 'xyz'}
    """
    ctx_methods = OrderedDict(future_map)
    fs = list(ctx_methods.values())
    results = await asyncio.gather(*fs)
    results = {
        key: results[idx] for idx, key in enumerate(ctx_methods.keys())
    }
    return results

def chars(string: any) -> str:
        """Return all (and only) the chars in the given string."""
        return ''.join([c if c.isalpha() else '' for c in str(string)])

def inverted_dict(d):
    """Return a dict with swapped keys and values

    >>> inverted_dict({0: ('a', 'b'), 1: 'cd'}) == {'cd': 1, ('a', 'b'): 0}
    True
    """
    return dict((force_hashable(v), k) for (k, v) in viewitems(dict(d)))

def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.

    Checks whether all objects in arrays have the same shape or length.

    Parameters
    ----------
    arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """

    uniques = np.unique([_num_samples(X) for X in arrays if X is not None])
    if len(uniques) > 1:
        raise ValueError("Found arrays with inconsistent numbers of samples: %s"
                         % str(uniques))

def is_integer(value: Any) -> bool:
    """Return true if a value is an integer number."""
    return (isinstance(value, int) and not isinstance(value, bool)) or (
        isinstance(value, float) and isfinite(value) and int(value) == value
    )

def bfx(value, msb, lsb):
    """! @brief Extract a value from a bitfield."""
    mask = bitmask((msb, lsb))
    return (value & mask) >> lsb

def zfill(x, width):
    """zfill(x, width) -> string

    Pad a numeric string x with zeros on the left, to fill a field
    of the specified width.  The string x is never truncated.

    """
    if not isinstance(x, basestring):
        x = repr(x)
    return x.zfill(width)

def maybe_infer_dtype_type(element):
    """Try to infer an object's dtype, for use in arithmetic ops

    Uses `element.dtype` if that's available.
    Objects implementing the iterator protocol are cast to a NumPy array,
    and from there the array's type is used.

    Parameters
    ----------
    element : object
        Possibly has a `.dtype` attribute, and possibly the iterator
        protocol.

    Returns
    -------
    tipo : type

    Examples
    --------
    >>> from collections import namedtuple
    >>> Foo = namedtuple("Foo", "dtype")
    >>> maybe_infer_dtype_type(Foo(np.dtype("i8")))
    numpy.int64
    """
    tipo = None
    if hasattr(element, 'dtype'):
        tipo = element.dtype
    elif is_list_like(element):
        element = np.asarray(element)
        tipo = element.dtype
    return tipo

def _check_whitespace(string):
    """
    Make sure thre is no whitespace in the given string. Will raise a
    ValueError if whitespace is detected
    """
    if string.count(' ') + string.count('\t') + string.count('\n') > 0:
        raise ValueError(INSTRUCTION_HAS_WHITESPACE)

def iprotate(l, steps=1):
    r"""Like rotate, but modifies `l` in-place.

    >>> l = [1,2,3]
    >>> iprotate(l) is l
    True
    >>> l
    [2, 3, 1]
    >>> iprotate(iprotate(l, 2), -3)
    [1, 2, 3]

    """
    if len(l):
        steps %= len(l)
        if steps:
            firstPart = l[:steps]
            del l[:steps]
            l.extend(firstPart)
    return l

def segment_str(text: str, phoneme_inventory: Set[str] = PHONEMES) -> str:
    """
    Takes as input a string in Kunwinjku and segments it into phoneme-like
    units based on the standard orthographic rules specified at
    http://bininjgunwok.org.au/
    """

    text = text.lower()
    text = segment_into_tokens(text, phoneme_inventory)
    return text

def has_key(cls, *args):
        """
        Check whether flyweight object with specified key has already been created.

        Returns:
            bool: True if already created, False if not
        """
        key = args if len(args) > 1 else args[0]
        return key in cls._instances

def last_modified(self) -> Optional[datetime.datetime]:
        """The value of Last-Modified HTTP header, or None.

        This header is represented as a `datetime` object.
        """
        httpdate = self._headers.get(hdrs.LAST_MODIFIED)
        if httpdate is not None:
            timetuple = parsedate(httpdate)
            if timetuple is not None:
                return datetime.datetime(*timetuple[:6],
                                         tzinfo=datetime.timezone.utc)
        return None

def tofile(self, fileobj):
		"""
		write a cache object to the fileobj as a lal cache file
		"""
		for entry in self:
			print >>fileobj, str(entry)
		fileobj.close()

def _brief_print_list(lst, limit=7):
    """Print at most `limit` elements of list."""
    lst = list(lst)
    if len(lst) > limit:
        return _brief_print_list(lst[:limit//2], limit) + ', ..., ' + \
            _brief_print_list(lst[-limit//2:], limit)
    return ', '.join(["'%s'"%str(i) for i in lst])

def _tree_line(self, no_type: bool = False) -> str:
        """Return the receiver's contribution to tree diagram."""
        return self._tree_line_prefix() + " " + self.iname()

def uconcatenate(arrs, axis=0):
    """Concatenate a sequence of arrays.

    This wrapper around numpy.concatenate preserves units. All input arrays
    must have the same units.  See the documentation of numpy.concatenate for
    full details.

    Examples
    --------
    >>> from unyt import cm
    >>> A = [1, 2, 3]*cm
    >>> B = [2, 3, 4]*cm
    >>> uconcatenate((A, B))
    unyt_array([1, 2, 3, 2, 3, 4], 'cm')

    """
    v = np.concatenate(arrs, axis=axis)
    v = _validate_numpy_wrapper_units(v, arrs)
    return v

def _parse_tuple_string(argument):
        """ Return a tuple from parsing 'a,b,c,d' -> (a,b,c,d) """
        if isinstance(argument, str):
            return tuple(int(p.strip()) for p in argument.split(','))
        return argument

def get_from_gnucash26_date(date_str: str) -> date:
    """ Creates a datetime from GnuCash 2.6 date string """
    date_format = "%Y%m%d"
    result = datetime.strptime(date_str, date_format).date()
    return result

def camel_to_snake_case(string):
    """Converts 'string' presented in camel case to snake case.

    e.g.: CamelCase => snake_case
    """
    s = _1.sub(r'\1_\2', string)
    return _2.sub(r'\1_\2', s).lower()

def count(args):
    """ count occurences in a list of lists
    >>> count([['a','b'],['a']])
    defaultdict(int, {'a' : 2, 'b' : 1})
    """
    counts = defaultdict(int)
    for arg in args:
        for item in arg:
            counts[item] = counts[item] + 1
    return counts

def encode_list(key, list_):
    # type: (str, Iterable) -> Dict[str, str]
    """
    Converts a list into a space-separated string and puts it in a dictionary

    :param key: Dictionary key to store the list
    :param list_: A list of objects
    :return: A dictionary key->string or an empty dictionary
    """
    if not list_:
        return {}
    return {key: " ".join(str(i) for i in list_)}

def natural_sort(list_to_sort: Iterable[str]) -> List[str]:
    """
    Sorts a list of strings case insensitively as well as numerically.

    For example: ['a1', 'A2', 'a3', 'A11', 'a22']

    To sort a list in place, don't call this method, which makes a copy. Instead, do this:

    my_list.sort(key=natural_keys)

    :param list_to_sort: the list being sorted
    :return: the list sorted naturally
    """
    return sorted(list_to_sort, key=natural_keys)

def last_location_of_minimum(x):
    """
    Returns the last location of the minimal value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    return 1.0 - np.argmin(x[::-1]) / len(x) if len(x) > 0 else np.NaN

def has_synset(word: str) -> list:
    """" Returns a list of synsets of a word after lemmatization. """

    return wn.synsets(lemmatize(word, neverstem=True))

def are_token_parallel(sequences: Sequence[Sized]) -> bool:
    """
    Returns True if all sequences in the list have the same length.
    """
    if not sequences or len(sequences) == 1:
        return True
    return all(len(s) == len(sequences[0]) for s in sequences)

def release_lock():
    """Release lock on compilation directory."""
    get_lock.n_lock -= 1
    assert get_lock.n_lock >= 0
    # Only really release lock once all lock requests have ended.
    if get_lock.lock_is_enabled and get_lock.n_lock == 0:
        get_lock.start_time = None
        get_lock.unlocker.unlock()

def _(f, x):
    """
    filter for dict, note `f` should have signature: `f::key->value->bool`
    """
    return {k: v for k, v in x.items() if f(k, v)}

def list_to_str(list, separator=','):
    """
    >>> list = [0, 0, 7]
    >>> list_to_str(list)
    '0,0,7'
    """
    list = [str(x) for x in list]
    return separator.join(list)

def margin(text):
    r"""Add a margin to both ends of each line in the string.

    Example:
        >>> margin('line1\nline2')
        '  line1  \n  line2  '
    """
    lines = str(text).split('\n')
    return '\n'.join('  {}  '.format(l) for l in lines)

def get_period_last_3_months() -> str:
    """ Returns the last week as a period string """
    today = Datum()
    today.today()

    # start_date = today - timedelta(weeks=13)
    start_date = today.clone()
    start_date.subtract_months(3)

    period = get_period(start_date.date, today.date)
    return period

def split(text: str) -> List[str]:
    """Split a text into a list of tokens.

    :param text: the text to split
    :return: tokens
    """
    return [word for word in SEPARATOR.split(text) if word.strip(' \t')]

def inverted_dict(d):
    """Return a dict with swapped keys and values

    >>> inverted_dict({0: ('a', 'b'), 1: 'cd'}) == {'cd': 1, ('a', 'b'): 0}
    True
    """
    return dict((force_hashable(v), k) for (k, v) in viewitems(dict(d)))

def iterate_items(dictish):
    """ Return a consistent (key, value) iterable on dict-like objects,
    including lists of tuple pairs.

    Example:

        >>> list(iterate_items({'a': 1}))
        [('a', 1)]
        >>> list(iterate_items([('a', 1), ('b', 2)]))
        [('a', 1), ('b', 2)]
    """
    if hasattr(dictish, 'iteritems'):
        return dictish.iteritems()
    if hasattr(dictish, 'items'):
        return dictish.items()
    return dictish

def closest_values(L):
    """Closest values

    :param L: list of values
    :returns: two values from L with minimal distance
    :modifies: the order of L
    :complexity: O(n log n), for n=len(L)
    """
    assert len(L) >= 2
    L.sort()
    valmin, argmin = min((L[i] - L[i - 1], i) for i in range(1, len(L)))
    return L[argmin - 1], L[argmin]

def is_client(self):
        """Return True if Glances is running in client mode."""
        return (self.args.client or self.args.browser) and not self.args.server

def toStringArray(name, a, width = 0):
    """
    Returns an array (any sequence of floats, really) as a string.
    """
    string = name + ": "
    cnt = 0
    for i in a:
        string += "%4.2f  " % i 
        if width > 0 and (cnt + 1) % width == 0:
            string += '\n'
        cnt += 1
    return string

def _isint(string):
    """
    >>> _isint("123")
    True
    >>> _isint("123.45")
    False
    """
    return type(string) is int or \
           (isinstance(string, _binary_type) or isinstance(string, _text_type)) and \
           _isconvertible(int, string)

def psutil_phymem_usage():
    """
    Return physical memory usage (float)
    Requires the cross-platform psutil (>=v0.3) library
    (https://github.com/giampaolo/psutil)
    """
    import psutil
    # This is needed to avoid a deprecation warning error with
    # newer psutil versions
    try:
        percent = psutil.virtual_memory().percent
    except:
        percent = psutil.phymem_usage().percent
    return percent

def text_coords(string, position):
    r"""
    Transform a simple index into a human-readable position in a string.

    This function accepts a string and an index, and will return a triple of
    `(lineno, columnno, line)` representing the position through the text. It's
    useful for displaying a string index in a human-readable way::

        >>> s = "abcdef\nghijkl\nmnopqr\nstuvwx\nyz"
        >>> text_coords(s, 0)
        (0, 0, 'abcdef')
        >>> text_coords(s, 4)
        (0, 4, 'abcdef')
        >>> text_coords(s, 6)
        (0, 6, 'abcdef')
        >>> text_coords(s, 7)
        (1, 0, 'ghijkl')
        >>> text_coords(s, 11)
        (1, 4, 'ghijkl')
        >>> text_coords(s, 15)
        (2, 1, 'mnopqr')
    """
    line_start = string.rfind('\n', 0, position) + 1
    line_end = string.find('\n', position)
    lineno = string.count('\n', 0, position)
    columnno = position - line_start
    line = string[line_start:line_end]
    return (lineno, columnno, line)

def highlight(text: str, color_code: int, bold: bool=False) -> str:
    """Wraps the given string with terminal color codes.

    Args:
        text: The content to highlight.
        color_code: The color to highlight with, e.g. 'shelltools.RED'.
        bold: Whether to bold the content in addition to coloring.

    Returns:
        The highlighted string.
    """
    return '{}\033[{}m{}\033[0m'.format(
        '\033[1m' if bold else '',
        color_code,
        text,)

def assign_parent(node: astroid.node_classes.NodeNG) -> astroid.node_classes.NodeNG:
    """return the higher parent which is not an AssignName, Tuple or List node
    """
    while node and isinstance(node, (astroid.AssignName, astroid.Tuple, astroid.List)):
        node = node.parent
    return node

def stretch(iterable, n=2):
    r"""Repeat each item in `iterable` `n` times.

    Example:

    >>> list(stretch(range(3), 2))
    [0, 0, 1, 1, 2, 2]
    """
    times = range(n)
    for item in iterable:
        for i in times: yield item

def excel_datetime(timestamp, epoch=None):
    """Return datetime object from timestamp in Excel serial format.

    Convert LSM time stamps.

    >>> excel_datetime(40237.029999999795)
    datetime.datetime(2010, 2, 28, 0, 43, 11, 999982)

    """
    if epoch is None:
        epoch = datetime.datetime.fromordinal(693594)
    return epoch + datetime.timedelta(timestamp)

def callable_validator(v: Any) -> AnyCallable:
    """
    Perform a simple check if the value is callable.

    Note: complete matching of argument type hints and return types is not performed
    """
    if callable(v):
        return v

    raise errors.CallableError(value=v)

def pack_bits( longbits ):
    """Crunch a 64-bit int (8 bool bytes) into a bitfield."""
    byte = longbits & (0x0101010101010101)
    byte = (byte | (byte>>7)) & (0x0003000300030003)
    byte = (byte | (byte>>14)) & (0x0000000f0000000f)
    byte = (byte | (byte>>28)) & (0x00000000000000ff)
    return byte

def enum_mark_last(iterable, start=0):
    """
    Returns a generator over iterable that tells whether the current item is the last one.
    Usage:
        >>> iterable = range(10)
        >>> for index, is_last, item in enum_mark_last(iterable):
        >>>     print(index, item, end='\n' if is_last else ', ')
    """
    it = iter(iterable)
    count = start
    try:
        last = next(it)
    except StopIteration:
        return
    for val in it:
        yield count, False, last
        last = val
        count += 1
    yield count, True, last

def pack_bits( longbits ):
    """Crunch a 64-bit int (8 bool bytes) into a bitfield."""
    byte = longbits & (0x0101010101010101)
    byte = (byte | (byte>>7)) & (0x0003000300030003)
    byte = (byte | (byte>>14)) & (0x0000000f0000000f)
    byte = (byte | (byte>>28)) & (0x00000000000000ff)
    return byte

def to_iso_string(self) -> str:
        """ Returns full ISO string for the given date """
        assert isinstance(self.value, datetime)
        return datetime.isoformat(self.value)

def _brief_print_list(lst, limit=7):
    """Print at most `limit` elements of list."""
    lst = list(lst)
    if len(lst) > limit:
        return _brief_print_list(lst[:limit//2], limit) + ', ..., ' + \
            _brief_print_list(lst[-limit//2:], limit)
    return ', '.join(["'%s'"%str(i) for i in lst])

def get_period_last_3_months() -> str:
    """ Returns the last week as a period string """
    today = Datum()
    today.today()

    # start_date = today - timedelta(weeks=13)
    start_date = today.clone()
    start_date.subtract_months(3)

    period = get_period(start_date.date, today.date)
    return period

def truncate(value: Decimal, n_digits: int) -> Decimal:
    """Truncates a value to a number of decimals places"""
    return Decimal(math.trunc(value * (10 ** n_digits))) / (10 ** n_digits)

def timeit(func, *args, **kwargs):
    """
    Time execution of function. Returns (res, seconds).

    >>> res, timing = timeit(time.sleep, 1)
    """
    start_time = time.time()
    res = func(*args, **kwargs)
    timing = time.time() - start_time
    return res, timing

def is_sqlatype_string(coltype: Union[TypeEngine, VisitableType]) -> bool:
    """
    Is the SQLAlchemy column type a string type?
    """
    coltype = _coltype_to_typeengine(coltype)
    return isinstance(coltype, sqltypes.String)

def clean_all_buckets(self):
        """
        Removes all buckets from all hashes and their content.
        """
        bucket_keys = self.redis_object.keys(pattern='nearpy_*')
        if len(bucket_keys) > 0:
            self.redis_object.delete(*bucket_keys)

def snake_to_camel(s: str) -> str:
    """Convert string from snake case to camel case."""

    fragments = s.split('_')

    return fragments[0] + ''.join(x.title() for x in fragments[1:])

def list_to_str(list, separator=','):
    """
    >>> list = [0, 0, 7]
    >>> list_to_str(list)
    '0,0,7'
    """
    list = [str(x) for x in list]
    return separator.join(list)

def output_dir(self, *args) -> str:
        """ Directory where to store output """
        return os.path.join(self.project_dir, 'output', *args)

def writable_stream(handle):
    """Test whether a stream can be written to.
    """
    if isinstance(handle, io.IOBase) and sys.version_info >= (3, 5):
        return handle.writable()
    try:
        handle.write(b'')
    except (io.UnsupportedOperation, IOError):
        return False
    else:
        return True

def strtobytes(input, encoding):
    """Take a str and transform it into a byte array."""
    py_version = sys.version_info[0]
    if py_version >= 3:
        return _strtobytes_py3(input, encoding)
    return _strtobytes_py2(input, encoding)

def contains(self, token: str) -> bool:
        """Return if the token is in the list or not."""
        self._validate_token(token)
        return token in self

def detect_model_num(string):
    """Takes a string related to a model name and extract its model number.

    For example:
        '000000-bootstrap.index' => 0
    """
    match = re.match(MODEL_NUM_REGEX, string)
    if match:
        return int(match.group())
    return None

def get_column_names(engine: Engine, tablename: str) -> List[str]:
    """
    Get all the database column names for the specified table.
    """
    return [info.name for info in gen_columns_info(engine, tablename)]

def uppercase_chars(string: any) -> str:
        """Return all (and only) the uppercase chars in the given string."""
        return ''.join([c if c.isupper() else '' for c in str(string)])

def is_line_in_file(filename: str, line: str) -> bool:
    """
    Detects whether a line is present within a file.

    Args:
        filename: file to check
        line: line to search for (as an exact match)
    """
    assert "\n" not in line
    with open(filename, "r") as file:
        for fileline in file:
            if fileline == line:
                return True
        return False

def shape(self) -> Tuple[int, ...]:
        """Shape of histogram's data.

        Returns
        -------
        One-element tuple with the number of bins along each axis.
        """
        return tuple(bins.bin_count for bins in self._binnings)

def is_sqlatype_integer(coltype: Union[TypeEngine, VisitableType]) -> bool:
    """
    Is the SQLAlchemy column type an integer type?
    """
    coltype = _coltype_to_typeengine(coltype)
    return isinstance(coltype, sqltypes.Integer)

def _(f, x):
    """
    filter for dict, note `f` should have signature: `f::key->value->bool`
    """
    return {k: v for k, v in x.items() if f(k, v)}

def most_frequent(lst):
    """
    Returns the item that appears most frequently in the given list.
    """
    lst = lst[:]
    highest_freq = 0
    most_freq = None

    for val in unique(lst):
        if lst.count(val) > highest_freq:
            most_freq = val
            highest_freq = lst.count(val)
            
    return most_freq

def simple_eq(one: Instance, two: Instance, attrs: List[str]) -> bool:
    """
    Test if two objects are equal, based on a comparison of the specified
    attributes ``attrs``.
    """
    return all(getattr(one, a) == getattr(two, a) for a in attrs)

def split(text: str) -> List[str]:
    """Split a text into a list of tokens.

    :param text: the text to split
    :return: tokens
    """
    return [word for word in SEPARATOR.split(text) if word.strip(' \t')]

def to_int64(a):
    """Return view of the recarray with all int32 cast to int64."""
    # build new dtype and replace i4 --> i8
    def promote_i4(typestr):
        if typestr[1:] == 'i4':
            typestr = typestr[0]+'i8'
        return typestr

    dtype = [(name, promote_i4(typestr)) for name,typestr in a.dtype.descr]
    return a.astype(dtype)

def imt2tup(string):
    """
    >>> imt2tup('PGA')
    ('PGA',)
    >>> imt2tup('SA(1.0)')
    ('SA', 1.0)
    >>> imt2tup('SA(1)')
    ('SA', 1.0)
    """
    s = string.strip()
    if not s.endswith(')'):
        # no parenthesis, PGA is considered the same as PGA()
        return (s,)
    name, rest = s.split('(', 1)
    return (name,) + tuple(float(x) for x in ast.literal_eval(rest[:-1] + ','))

def read(self, count=0):
        """ Read """
        return self.f.read(count) if count > 0 else self.f.read()

def list_to_str(lst):
    """
    Turn a list into a comma- and/or and-separated string.

    Parameters
    ----------
    lst : :obj:`list`
        A list of strings to join into a single string.

    Returns
    -------
    str_ : :obj:`str`
        A string with commas and/or ands separating th elements from ``lst``.

    """
    if len(lst) == 1:
        str_ = lst[0]
    elif len(lst) == 2:
        str_ = ' and '.join(lst)
    elif len(lst) > 2:
        str_ = ', '.join(lst[:-1])
        str_ += ', and {0}'.format(lst[-1])
    else:
        raise ValueError('List of length 0 provided.')
    return str_

def _store_helper(model: Action, session: Optional[Session] = None) -> None:
    """Help store an action."""
    if session is None:
        session = _make_session()

    session.add(model)
    session.commit()
    session.close()

def _cursorLeft(self):
        """ Handles "cursor left" events """
        if self.cursorPos > 0:
            self.cursorPos -= 1
            sys.stdout.write(console.CURSOR_LEFT)
            sys.stdout.flush()

def has_key(cls, *args):
        """
        Check whether flyweight object with specified key has already been created.

        Returns:
            bool: True if already created, False if not
        """
        key = args if len(args) > 1 else args[0]
        return key in cls._instances

def strtobytes(input, encoding):
    """Take a str and transform it into a byte array."""
    py_version = sys.version_info[0]
    if py_version >= 3:
        return _strtobytes_py3(input, encoding)
    return _strtobytes_py2(input, encoding)

def get_last_weekday_in_month(year, month, weekday):
        """Get the last weekday in a given month. e.g:

        >>> # the last monday in Jan 2013
        >>> Calendar.get_last_weekday_in_month(2013, 1, MON)
        datetime.date(2013, 1, 28)
        """
        day = date(year, month, monthrange(year, month)[1])
        while True:
            if day.weekday() == weekday:
                break
            day = day - timedelta(days=1)
        return day

def camel_to_snake_case(string):
    """Converts 'string' presented in camel case to snake case.

    e.g.: CamelCase => snake_case
    """
    s = _1.sub(r'\1_\2', string)
    return _2.sub(r'\1_\2', s).lower()

def get_keys_of_max_n(dict_obj, n):
    """Returns the keys that maps to the top n max values in the given dict.

    Example:
    --------
    >>> dict_obj = {'a':2, 'b':1, 'c':5}
    >>> get_keys_of_max_n(dict_obj, 2)
    ['a', 'c']
    """
    return sorted([
        item[0]
        for item in sorted(
            dict_obj.items(), key=lambda item: item[1], reverse=True
        )[:n]
    ])

def parse_reading(val: str) -> Optional[float]:
    """ Convert reading value to float (if possible) """
    try:
        return float(val)
    except ValueError:
        logging.warning('Reading of "%s" is not a number', val)
        return None

def flatten_list(x: List[Any]) -> List[Any]:
    """
    Converts a list of lists into a flat list.
    
    Args:
        x: list of lists 

    Returns:
        flat list
        
    As per
    http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python

    """  # noqa
    return [item for sublist in x for item in sublist]

def datetime_from_isoformat(value: str):
    """Return a datetime object from an isoformat string.

    Args:
        value (str): Datetime string in isoformat.

    """
    if sys.version_info >= (3, 7):
        return datetime.fromisoformat(value)

    return datetime.strptime(value, '%Y-%m-%dT%H:%M:%S.%f')

def get_domain(url):
    """
    Get domain part of an url.

    For example: https://www.python.org/doc/ -> https://www.python.org
    """
    parse_result = urlparse(url)
    domain = "{schema}://{netloc}".format(
        schema=parse_result.scheme, netloc=parse_result.netloc)
    return domain

def argmax(self, rows: List[Row], column: ComparableColumn) -> List[Row]:
        """
        Takes a list of rows and a column name and returns a list containing a single row (dict from
        columns to cells) that has the maximum numerical value in the given column. We return a list
        instead of a single dict to be consistent with the return type of ``select`` and
        ``all_rows``.
        """
        if not rows:
            return []
        value_row_pairs = [(row.values[column.name], row) for row in rows]
        if not value_row_pairs:
            return []
        # Returns a list containing the row with the max cell value.
        return [sorted(value_row_pairs, key=lambda x: x[0], reverse=True)[0][1]]

def tanimoto_set_similarity(x: Iterable[X], y: Iterable[X]) -> float:
    """Calculate the tanimoto set similarity."""
    a, b = set(x), set(y)
    union = a | b

    if not union:
        return 0.0

    return len(a & b) / len(union)

def reverse_mapping(mapping):
	"""
	For every key, value pair, return the mapping for the
	equivalent value, key pair

	>>> reverse_mapping({'a': 'b'}) == {'b': 'a'}
	True
	"""
	keys, values = zip(*mapping.items())
	return dict(zip(values, keys))

def exponential_backoff(attempt: int, cap: int=1200) -> timedelta:
    """Calculate a delay to retry using an exponential backoff algorithm.

    It is an exponential backoff with random jitter to prevent failures
    from being retried at the same time. It is a good fit for most
    applications.

    :arg attempt: the number of attempts made
    :arg cap: maximum delay, defaults to 20 minutes
    """
    base = 3
    temp = min(base * 2 ** attempt, cap)
    return timedelta(seconds=temp / 2 + random.randint(0, temp / 2))

def get_table_names_from_metadata(metadata: MetaData) -> List[str]:
    """
    Returns all database table names found in an SQLAlchemy :class:`MetaData`
    object.
    """
    return [table.name for table in metadata.tables.values()]

def clean_map(obj: Mapping[Any, Any]) -> Mapping[Any, Any]:
    """
    Return a new copied dictionary without the keys with ``None`` values from
    the given Mapping object.
    """
    return {k: v for k, v in obj.items() if v is not None}

def dtypes(self):
        """Returns all column names and their data types as a list.

        >>> df.dtypes
        [('age', 'int'), ('name', 'string')]
        """
        return [(str(f.name), f.dataType.simpleString()) for f in self.schema.fields]

def is_integer(value: Any) -> bool:
    """Return true if a value is an integer number."""
    return (isinstance(value, int) and not isinstance(value, bool)) or (
        isinstance(value, float) and isfinite(value) and int(value) == value
    )

def flatten_multidict(multidict):
    """Return flattened dictionary from ``MultiDict``."""
    return dict([(key, value if len(value) > 1 else value[0])
                 for (key, value) in multidict.iterlists()])

def dict_to_enum_fn(d: Dict[str, Any], enum_class: Type[Enum]) -> Enum:
    """
    Converts an ``dict`` to a ``Enum``.
    """
    return enum_class[d['name']]

def almost_hermitian(gate: Gate) -> bool:
    """Return true if gate tensor is (almost) Hermitian"""
    return np.allclose(asarray(gate.asoperator()),
                       asarray(gate.H.asoperator()))

def recClearTag(element):
    """Applies maspy.xml.clearTag() to the tag attribute of the "element" and
    recursively to all child elements.

    :param element: an :instance:`xml.etree.Element`
    """
    children = element.getchildren()
    if len(children) > 0:
        for child in children:
            recClearTag(child)
    element.tag = clearTag(element.tag)

def sample_normal(mean, var, rng):
    """Sample from independent normal distributions

    Each element is an independent normal distribution.

    Parameters
    ----------
    mean : numpy.ndarray
      Means of the normal distribution. Shape --> (batch_num, sample_dim)
    var : numpy.ndarray
      Variance of the normal distribution. Shape --> (batch_num, sample_dim)
    rng : numpy.random.RandomState

    Returns
    -------
    ret : numpy.ndarray
       The sampling result. Shape --> (batch_num, sample_dim)
    """
    ret = numpy.sqrt(var) * rng.randn(*mean.shape) + mean
    return ret

def _my_hash(arg_list):
    # type: (List[Any]) -> int
    """Simple helper hash function"""
    res = 0
    for arg in arg_list:
        res = res * 31 + hash(arg)
    return res

def read_set_from_file(filename: str) -> Set[str]:
    """
    Extract a de-duped collection (set) of text from a file.
    Expected file format is one item per line.
    """
    collection = set()
    with open(filename, 'r') as file_:
        for line in file_:
            collection.add(line.rstrip())
    return collection

def timeit(func, *args, **kwargs):
    """
    Time execution of function. Returns (res, seconds).

    >>> res, timing = timeit(time.sleep, 1)
    """
    start_time = time.time()
    res = func(*args, **kwargs)
    timing = time.time() - start_time
    return res, timing

def to_int64(a):
    """Return view of the recarray with all int32 cast to int64."""
    # build new dtype and replace i4 --> i8
    def promote_i4(typestr):
        if typestr[1:] == 'i4':
            typestr = typestr[0]+'i8'
        return typestr

    dtype = [(name, promote_i4(typestr)) for name,typestr in a.dtype.descr]
    return a.astype(dtype)

def inject_nulls(data: Mapping, field_names) -> dict:
    """Insert None as value for missing fields."""

    record = dict()

    for field in field_names:
        record[field] = data.get(field, None)

    return record

def clean_column_names(df: DataFrame) -> DataFrame:
    """
    Strip the whitespace from all column names in the given DataFrame
    and return the result.
    """
    f = df.copy()
    f.columns = [col.strip() for col in f.columns]
    return f

def bytes_hack(buf):
    """
    Hacky workaround for old installs of the library on systems without python-future that were
    keeping the 2to3 update from working after auto-update.
    """
    ub = None
    if sys.version_info > (3,):
        ub = buf
    else:
        ub = bytes(buf)

    return ub

def chars(string: any) -> str:
        """Return all (and only) the chars in the given string."""
        return ''.join([c if c.isalpha() else '' for c in str(string)])

def iter_lines(file_like: Iterable[str]) -> Generator[str, None, None]:
    """ Helper for iterating only nonempty lines without line breaks"""
    for line in file_like:
        line = line.rstrip('\r\n')
        if line:
            yield line

def valid_file(path: str) -> bool:
    """
    Verifies that a string path actually exists and is a file

    :param path: The path to verify
    :return: **True** if path exist and is a file
    """
    path = Path(path).expanduser()
    log.debug("checking if %s is a valid file", path)
    return path.exists() and path.is_file()

def cli_run():
    """docstring for argparse"""
    parser = argparse.ArgumentParser(description='Stupidly simple code answers from StackOverflow')
    parser.add_argument('query', help="What's the problem ?", type=str, nargs='+')
    parser.add_argument('-t','--tags', help='semicolon separated tags -> python;lambda')
    args = parser.parse_args()
    main(args)

def pack_bits( longbits ):
    """Crunch a 64-bit int (8 bool bytes) into a bitfield."""
    byte = longbits & (0x0101010101010101)
    byte = (byte | (byte>>7)) & (0x0003000300030003)
    byte = (byte | (byte>>14)) & (0x0000000f0000000f)
    byte = (byte | (byte>>28)) & (0x00000000000000ff)
    return byte

def grep(pattern, filename):
    """Very simple grep that returns the first matching line in a file.
    String matching only, does not do REs as currently implemented.
    """
    try:
        # for line in file
        # if line matches pattern:
        #    return line
        return next((L for L in open(filename) if L.find(pattern) >= 0))
    except StopIteration:
        return ''

def flatten_list(l: List[list]) -> list:
    """ takes a list of lists, l and returns a flat list
    """
    return [v for inner_l in l for v in inner_l]

def get_deprecation_reason(
    node: Union[EnumValueDefinitionNode, FieldDefinitionNode]
) -> Optional[str]:
    """Given a field or enum value node, get deprecation reason as string."""
    from ..execution import get_directive_values

    deprecated = get_directive_values(GraphQLDeprecatedDirective, node)
    return deprecated["reason"] if deprecated else None

def dtypes(self):
        """Returns all column names and their data types as a list.

        >>> df.dtypes
        [('age', 'int'), ('name', 'string')]
        """
        return [(str(f.name), f.dataType.simpleString()) for f in self.schema.fields]

def _prm_get_longest_stringsize(string_list):
        """ Returns the longest string size for a string entry across data."""
        maxlength = 1

        for stringar in string_list:
            if isinstance(stringar, np.ndarray):
                if stringar.ndim > 0:
                    for string in stringar.ravel():
                        maxlength = max(len(string), maxlength)
                else:
                    maxlength = max(len(stringar.tolist()), maxlength)
            else:
                maxlength = max(len(stringar), maxlength)

        # Make the string Col longer than needed in order to allow later on slightly larger strings
        return int(maxlength * 1.5)

def first_location_of_maximum(x):
    """
    Returns the first location of the maximum value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.argmax(x) / len(x) if len(x) > 0 else np.NaN

def astensor(array: TensorLike) -> BKTensor:
    """Covert numpy array to tensorflow tensor"""
    tensor = tf.convert_to_tensor(value=array, dtype=CTYPE)
    return tensor

def cmd_dot(conf: Config):
    """Print out a neat targets dependency tree based on requested targets.

    Use graphviz to render the dot file, e.g.:

    > ybt dot :foo :bar | dot -Tpng -o graph.png
    """
    build_context = BuildContext(conf)
    populate_targets_graph(build_context, conf)
    if conf.output_dot_file is None:
        write_dot(build_context, conf, sys.stdout)
    else:
        with open(conf.output_dot_file, 'w') as out_file:
            write_dot(build_context, conf, out_file)

def has_value(cls, value: int) -> bool:
        """True if specified value exists in int enum; otherwise, False."""
        return any(value == item.value for item in cls)

def _rindex(mylist: Sequence[T], x: T) -> int:
    """Index of the last occurrence of x in the sequence."""
    return len(mylist) - mylist[::-1].index(x) - 1

def _skip_section(self):
        """Skip a section"""
        self._last = self._f.readline()
        while len(self._last) > 0 and len(self._last[0].strip()) == 0:
            self._last = self._f.readline()

def post(self, endpoint: str, **kwargs) -> dict:
        """HTTP POST operation to API endpoint."""

        return self._request('POST', endpoint, **kwargs)

def auto_up(self, count=1, go_to_start_of_line_if_history_changes=False):
        """
        If we're not on the first line (of a multiline input) go a line up,
        otherwise go back in history. (If nothing is selected.)
        """
        if self.complete_state:
            self.complete_previous(count=count)
        elif self.document.cursor_position_row > 0:
            self.cursor_up(count=count)
        elif not self.selection_state:
            self.history_backward(count=count)

            # Go to the start of the line?
            if go_to_start_of_line_if_history_changes:
                self.cursor_position += self.document.get_start_of_line_position()

def to_0d_array(value: Any) -> np.ndarray:
    """Given a value, wrap it in a 0-D numpy.ndarray.
    """
    if np.isscalar(value) or (isinstance(value, np.ndarray) and
                              value.ndim == 0):
        return np.array(value)
    else:
        return to_0d_object_array(value)

def _my_hash(arg_list):
    # type: (List[Any]) -> int
    """Simple helper hash function"""
    res = 0
    for arg in arg_list:
        res = res * 31 + hash(arg)
    return res

def astensor(array: TensorLike) -> BKTensor:
    """Covert numpy array to tensorflow tensor"""
    tensor = tf.convert_to_tensor(value=array, dtype=CTYPE)
    return tensor

def is_not_null(df: DataFrame, col_name: str) -> bool:
    """
    Return ``True`` if the given DataFrame has a column of the given
    name (string), and there exists at least one non-NaN value in that
    column; return ``False`` otherwise.
    """
    if (
        isinstance(df, pd.DataFrame)
        and col_name in df.columns
        and df[col_name].notnull().any()
    ):
        return True
    else:
        return False

def cli_run():
    """docstring for argparse"""
    parser = argparse.ArgumentParser(description='Stupidly simple code answers from StackOverflow')
    parser.add_argument('query', help="What's the problem ?", type=str, nargs='+')
    parser.add_argument('-t','--tags', help='semicolon separated tags -> python;lambda')
    args = parser.parse_args()
    main(args)

def dotproduct(X, Y):
    """Return the sum of the element-wise product of vectors x and y.
    >>> dotproduct([1, 2, 3], [1000, 100, 10])
    1230
    """
    return sum([x * y for x, y in zip(X, Y)])

def _parse_tuple_string(argument):
        """ Return a tuple from parsing 'a,b,c,d' -> (a,b,c,d) """
        if isinstance(argument, str):
            return tuple(int(p.strip()) for p in argument.split(','))
        return argument

def calculate_dimensions(image, long_side, short_side):
    """Returns the thumbnail dimensions depending on the images format."""
    if image.width >= image.height:
        return '{0}x{1}'.format(long_side, short_side)
    return '{0}x{1}'.format(short_side, long_side)

def fmt_camel(name):
    """
    Converts name to lower camel case. Words are identified by capitalization,
    dashes, and underscores.
    """
    words = split_words(name)
    assert len(words) > 0
    first = words.pop(0).lower()
    return first + ''.join([word.capitalize() for word in words])

def from_file(filename, mime=False):
    """"
    Accepts a filename and returns the detected filetype.  Return
    value is the mimetype if mime=True, otherwise a human readable
    name.

    >>> magic.from_file("testdata/test.pdf", mime=True)
    'application/pdf'
    """
    m = _get_magic_type(mime)
    return m.from_file(filename)

def list_to_str(list, separator=','):
    """
    >>> list = [0, 0, 7]
    >>> list_to_str(list)
    '0,0,7'
    """
    list = [str(x) for x in list]
    return separator.join(list)

def clean_column_names(df: DataFrame) -> DataFrame:
    """
    Strip the whitespace from all column names in the given DataFrame
    and return the result.
    """
    f = df.copy()
    f.columns = [col.strip() for col in f.columns]
    return f

def _relative_frequency(self, word):
		"""Computes the log relative frequency for a word form"""

		count = self.type_counts.get(word, 0)
		return math.log(count/len(self.type_counts)) if count > 0 else 0

def list_to_str(lst):
    """
    Turn a list into a comma- and/or and-separated string.

    Parameters
    ----------
    lst : :obj:`list`
        A list of strings to join into a single string.

    Returns
    -------
    str_ : :obj:`str`
        A string with commas and/or ands separating th elements from ``lst``.

    """
    if len(lst) == 1:
        str_ = lst[0]
    elif len(lst) == 2:
        str_ = ' and '.join(lst)
    elif len(lst) > 2:
        str_ = ', '.join(lst[:-1])
        str_ += ', and {0}'.format(lst[-1])
    else:
        raise ValueError('List of length 0 provided.')
    return str_

def getIndex(predicateFn: Callable[[T], bool], items: List[T]) -> int:
    """
    Finds the index of an item in list, which satisfies predicate
    :param predicateFn: predicate function to run on items of list
    :param items: list of tuples
    :return: first index for which predicate function returns True
    """
    try:
        return next(i for i, v in enumerate(items) if predicateFn(v))
    except StopIteration:
        return -1

def isarray(array, test, dim=2):
    """Returns True if test is True for all array elements.
    Otherwise, returns False.
    """
    if dim > 1:
        return all(isarray(array[i], test, dim - 1)
                   for i in range(len(array)))
    return all(test(i) for i in array)

def is_quoted(arg: str) -> bool:
    """
    Checks if a string is quoted
    :param arg: the string being checked for quotes
    :return: True if a string is quoted
    """
    return len(arg) > 1 and arg[0] == arg[-1] and arg[0] in constants.QUOTES

def get_language():
    """
    Wrapper around Django's `get_language` utility.
    For Django >= 1.8, `get_language` returns None in case no translation is activate.
    Here we patch this behavior e.g. for back-end functionality requiring access to translated fields
    """
    from parler import appsettings
    language = dj_get_language()
    if language is None and appsettings.PARLER_DEFAULT_ACTIVATE:
        return appsettings.PARLER_DEFAULT_LANGUAGE_CODE
    else:
        return language

def chars(string: any) -> str:
        """Return all (and only) the chars in the given string."""
        return ''.join([c if c.isalpha() else '' for c in str(string)])

def infer_format(filename:str) -> str:
    """Return extension identifying format of given filename"""
    _, ext = os.path.splitext(filename)
    return ext

def rate_limited(max_per_hour: int, *args: Any) -> Callable[..., Any]:
    """Rate limit a function."""
    return util.rate_limited(max_per_hour, *args)

def _find_conda():
    """Find the conda executable robustly across conda versions.

    Returns
    -------
    conda : str
        Path to the conda executable.

    Raises
    ------
    IOError
        If the executable cannot be found in either the CONDA_EXE environment
        variable or in the PATH.

    Notes
    -----
    In POSIX platforms in conda >= 4.4, conda can be set up as a bash function
    rather than an executable. (This is to enable the syntax
    ``conda activate env-name``.) In this case, the environment variable
    ``CONDA_EXE`` contains the path to the conda executable. In other cases,
    we use standard search for the appropriate name in the PATH.

    See https://github.com/airspeed-velocity/asv/issues/645 for more details.
    """
    if 'CONDA_EXE' in os.environ:
        conda = os.environ['CONDA_EXE']
    else:
        conda = util.which('conda')
    return conda

def nTimes(n, f, *args, **kwargs):
    r"""Call `f` `n` times with `args` and `kwargs`.
    Useful e.g. for simplistic timing.

    Examples:

    >>> nTimes(3, sys.stdout.write, 'hallo\n')
    hallo
    hallo
    hallo

    """
    for i in xrange(n): f(*args, **kwargs)

def elmo_loss2ppl(losses: List[np.ndarray]) -> float:
    """ Calculates perplexity by loss

    Args:
        losses: list of numpy arrays of model losses

    Returns:
        perplexity : float
    """
    avg_loss = np.mean(losses)
    return float(np.exp(avg_loss))

def url_concat(url, args):
    """Concatenate url and argument dictionary regardless of whether
    url has existing query parameters.

    >>> url_concat("http://example.com/foo?a=b", dict(c="d"))
    'http://example.com/foo?a=b&c=d'
    """
    if not args: return url
    if url[-1] not in ('?', '&'):
        url += '&' if ('?' in url) else '?'
    return url + urllib.urlencode(args)

def get_versions(reporev=True):
    """Get version information for components used by Spyder"""
    import sys
    import platform

    import qtpy
    import qtpy.QtCore

    revision = None
    if reporev:
        from spyder.utils import vcs
        revision, branch = vcs.get_git_revision(os.path.dirname(__dir__))

    if not sys.platform == 'darwin':  # To avoid a crash with our Mac app
        system = platform.system()
    else:
        system = 'Darwin'

    return {
        'spyder': __version__,
        'python': platform.python_version(),  # "2.7.3"
        'bitness': 64 if sys.maxsize > 2**32 else 32,
        'qt': qtpy.QtCore.__version__,
        'qt_api': qtpy.API_NAME,      # PyQt5
        'qt_api_ver': qtpy.PYQT_VERSION,
        'system': system,   # Linux, Windows, ...
        'release': platform.release(),  # XP, 10.6, 2.2.0, etc.
        'revision': revision,  # '9fdf926eccce'
    }

def indexes_equal(a: Index, b: Index) -> bool:
    """
    Are two indexes equal? Checks by comparing ``str()`` versions of them.
    (AM UNSURE IF THIS IS ENOUGH.)
    """
    return str(a) == str(b)

def find_first_in_list(txt: str, str_list: [str]) -> int:  # type: ignore
    """
    Returns the index of the earliest occurence of an item from a list in a string

    Ex: find_first_in_list('foobar', ['bar', 'fin']) -> 3
    """
    start = len(txt) + 1
    for item in str_list:
        if start > txt.find(item) > -1:
            start = txt.find(item)
    return start if len(txt) + 1 > start > -1 else -1

def csv_to_numpy(string_like, dtype=None):  # type: (str) -> np.array
    """Convert a CSV object to a numpy array.

    Args:
        string_like (str): CSV string.
        dtype (dtype, optional):  Data type of the resulting array. If None, the dtypes will be determined by the
                                        contents of each column, individually. This argument can only be used to
                                        'upcast' the array.  For downcasting, use the .astype(t) method.
    Returns:
        (np.array): numpy array
    """
    stream = StringIO(string_like)
    return np.genfromtxt(stream, dtype=dtype, delimiter=',')

def count(args):
    """ count occurences in a list of lists
    >>> count([['a','b'],['a']])
    defaultdict(int, {'a' : 2, 'b' : 1})
    """
    counts = defaultdict(int)
    for arg in args:
        for item in arg:
            counts[item] = counts[item] + 1
    return counts

def _isint(string):
    """
    >>> _isint("123")
    True
    >>> _isint("123.45")
    False
    """
    return type(string) is int or \
           (isinstance(string, _binary_type) or isinstance(string, _text_type)) and \
           _isconvertible(int, string)

def shape(self) -> Tuple[int, ...]:
        """Shape of histogram's data.

        Returns
        -------
        One-element tuple with the number of bins along each axis.
        """
        return tuple(bins.bin_count for bins in self._binnings)

def get_keys_of_max_n(dict_obj, n):
    """Returns the keys that maps to the top n max values in the given dict.

    Example:
    --------
    >>> dict_obj = {'a':2, 'b':1, 'c':5}
    >>> get_keys_of_max_n(dict_obj, 2)
    ['a', 'c']
    """
    return sorted([
        item[0]
        for item in sorted(
            dict_obj.items(), key=lambda item: item[1], reverse=True
        )[:n]
    ])

def pset(iterable=(), pre_size=8):
    """
    Creates a persistent set from iterable. Optionally takes a sizing parameter equivalent to that
    used for :py:func:`pmap`.

    >>> s1 = pset([1, 2, 3, 2])
    >>> s1
    pset([1, 2, 3])
    """
    if not iterable:
        return _EMPTY_PSET

    return PSet._from_iterable(iterable, pre_size=pre_size)

def hsv2rgb_spectrum(hsv):
    """Generates RGB values from HSV values in line with a typical light
    spectrum."""
    h, s, v = hsv
    return hsv2rgb_raw(((h * 192) >> 8, s, v))

def post(self, endpoint: str, **kwargs) -> dict:
        """HTTP POST operation to API endpoint."""

        return self._request('POST', endpoint, **kwargs)

def clean(ctx, text):
    """
    Removes all non-printable characters from a text string
    """
    text = conversions.to_string(text, ctx)
    return ''.join([c for c in text if ord(c) >= 32])

def __as_list(value: List[JsonObjTypes]) -> List[JsonTypes]:
        """ Return a json array as a list

        :param value: array
        :return: array with JsonObj instances removed
        """
        return [e._as_dict if isinstance(e, JsonObj) else e for e in value]

def __add_method(m: lmap.Map, key: T, method: Method) -> lmap.Map:
        """Swap the methods atom to include method with key."""
        return m.assoc(key, method)

def isfile_notempty(inputfile: str) -> bool:
        """Check if the input filename with path is a file and is not empty."""
        try:
            return isfile(inputfile) and getsize(inputfile) > 0
        except TypeError:
            raise TypeError('inputfile is not a valid type')

def pack_bits( longbits ):
    """Crunch a 64-bit int (8 bool bytes) into a bitfield."""
    byte = longbits & (0x0101010101010101)
    byte = (byte | (byte>>7)) & (0x0003000300030003)
    byte = (byte | (byte>>14)) & (0x0000000f0000000f)
    byte = (byte | (byte>>28)) & (0x00000000000000ff)
    return byte

def __iter__(self):
        """Define a generator function and return it"""
        def generator():
            for i, obj in enumerate(self._sequence):
                if i >= self._limit:
                    break
                yield obj
            raise StopIteration
        return generator

def clean_int(x) -> int:
    """
    Returns its parameter as an integer, or raises
    ``django.forms.ValidationError``.
    """
    try:
        return int(x)
    except ValueError:
        raise forms.ValidationError(
            "Cannot convert to integer: {}".format(repr(x)))

def iter_lines(file_like: Iterable[str]) -> Generator[str, None, None]:
    """ Helper for iterating only nonempty lines without line breaks"""
    for line in file_like:
        line = line.rstrip('\r\n')
        if line:
            yield line

def check64bit(current_system="python"):
    """checks if you are on a 64 bit platform"""
    if current_system == "python":
        return sys.maxsize > 2147483647
    elif current_system == "os":
        import platform
        pm = platform.machine()
        if pm != ".." and pm.endswith('64'):  # recent Python (not Iron)
            return True
        else:
            if 'PROCESSOR_ARCHITEW6432' in os.environ:
                return True  # 32 bit program running on 64 bit Windows
            try:
                # 64 bit Windows 64 bit program
                return os.environ['PROCESSOR_ARCHITECTURE'].endswith('64')
            except IndexError:
                pass  # not Windows
            try:
                # this often works in Linux
                return '64' in platform.architecture()[0]
            except Exception:
                # is an older version of Python, assume also an older os@
                # (best we can guess)
                return False

def strtobytes(input, encoding):
    """Take a str and transform it into a byte array."""
    py_version = sys.version_info[0]
    if py_version >= 3:
        return _strtobytes_py3(input, encoding)
    return _strtobytes_py2(input, encoding)

def _skip_section(self):
        """Skip a section"""
        self._last = self._f.readline()
        while len(self._last) > 0 and len(self._last[0].strip()) == 0:
            self._last = self._f.readline()

def strings_to_integers(strings: Iterable[str]) -> Iterable[int]:
    """
    Convert a list of strings to a list of integers.

    :param strings: a list of string
    :return: a list of converted integers

    .. doctest::

        >>> strings_to_integers(['1', '1.0', '-0.2'])
        [1, 1, 0]
    """
    return strings_to_(strings, lambda x: int(float(x)))

def is_finite(value: Any) -> bool:
    """Return true if a value is a finite number."""
    return isinstance(value, int) or (isinstance(value, float) and isfinite(value))

def format_repr(obj, attributes) -> str:
    """Format an object's repr method with specific attributes."""

    attribute_repr = ', '.join(('{}={}'.format(attr, repr(getattr(obj, attr)))
                                for attr in attributes))
    return "{0}({1})".format(obj.__class__.__qualname__, attribute_repr)

def _read_words(filename):
  """Reads words from a file."""
  with tf.gfile.GFile(filename, "r") as f:
    if sys.version_info[0] >= 3:
      return f.read().replace("\n", " %s " % EOS).split()
    else:
      return f.read().decode("utf-8").replace("\n", " %s " % EOS).split()

def _my_hash(arg_list):
    # type: (List[Any]) -> int
    """Simple helper hash function"""
    res = 0
    for arg in arg_list:
        res = res * 31 + hash(arg)
    return res

def u16le_list_to_byte_list(data):
    """! @brief Convert a halfword array into a byte array"""
    byteData = []
    for h in data:
        byteData.extend([h & 0xff, (h >> 8) & 0xff])
    return byteData

def _kbhit_unix() -> bool:
    """
    Under UNIX: is a keystroke available?
    """
    dr, dw, de = select.select([sys.stdin], [], [], 0)
    return dr != []

def listify(a):
    """
    Convert a scalar ``a`` to a list and all iterables to list as well.

    Examples
    --------
    >>> listify(0)
    [0]

    >>> listify([1,2,3])
    [1, 2, 3]

    >>> listify('a')
    ['a']

    >>> listify(np.array([1,2,3]))
    [1, 2, 3]

    >>> listify('string')
    ['string']
    """
    if a is None:
        return []
    elif not isinstance(a, (tuple, list, np.ndarray)):
        return [a]
    return list(a)

def sort_by_modified(files_or_folders: list) -> list:
    """
    Sort files or folders by modified time

    Args:
        files_or_folders: list of files or folders

    Returns:
        list
    """
    return sorted(files_or_folders, key=os.path.getmtime, reverse=True)

def position(self) -> Position:
        """The current position of the cursor."""
        return Position(self._index, self._lineno, self._col_offset)

def lint(fmt='colorized'):
    """Run verbose PyLint on source. Optionally specify fmt=html for HTML output."""
    if fmt == 'html':
        outfile = 'pylint_report.html'
        local('pylint -f %s davies > %s || true' % (fmt, outfile))
        local('open %s' % outfile)
    else:
        local('pylint -f %s davies || true' % fmt)

def cli_run():
    """docstring for argparse"""
    parser = argparse.ArgumentParser(description='Stupidly simple code answers from StackOverflow')
    parser.add_argument('query', help="What's the problem ?", type=str, nargs='+')
    parser.add_argument('-t','--tags', help='semicolon separated tags -> python;lambda')
    args = parser.parse_args()
    main(args)

def csv_to_numpy(string_like, dtype=None):  # type: (str) -> np.array
    """Convert a CSV object to a numpy array.

    Args:
        string_like (str): CSV string.
        dtype (dtype, optional):  Data type of the resulting array. If None, the dtypes will be determined by the
                                        contents of each column, individually. This argument can only be used to
                                        'upcast' the array.  For downcasting, use the .astype(t) method.
    Returns:
        (np.array): numpy array
    """
    stream = StringIO(string_like)
    return np.genfromtxt(stream, dtype=dtype, delimiter=',')

def clean_column_names(df: DataFrame) -> DataFrame:
    """
    Strip the whitespace from all column names in the given DataFrame
    and return the result.
    """
    f = df.copy()
    f.columns = [col.strip() for col in f.columns]
    return f

def valid_file(path: str) -> bool:
    """
    Verifies that a string path actually exists and is a file

    :param path: The path to verify
    :return: **True** if path exist and is a file
    """
    path = Path(path).expanduser()
    log.debug("checking if %s is a valid file", path)
    return path.exists() and path.is_file()

def _gauss(mean: int, sigma: int) -> int:
        """
        Creates a variation from a base value

        Args:
            mean: base value
            sigma: gaussian sigma

        Returns: random value

        """
        return int(random.gauss(mean, sigma))

def split(text: str) -> List[str]:
    """Split a text into a list of tokens.

    :param text: the text to split
    :return: tokens
    """
    return [word for word in SEPARATOR.split(text) if word.strip(' \t')]

def most_frequent(lst):
    """
    Returns the item that appears most frequently in the given list.
    """
    lst = lst[:]
    highest_freq = 0
    most_freq = None

    for val in unique(lst):
        if lst.count(val) > highest_freq:
            most_freq = val
            highest_freq = lst.count(val)
            
    return most_freq

def rmglob(pattern: str) -> None:
    """
    Deletes all files whose filename matches the glob ``pattern`` (via
    :func:`glob.glob`).
    """
    for f in glob.glob(pattern):
        os.remove(f)

def try_cast_int(s):
    """(str) -> int
    All the digits in a given string are concatenated and converted into a single number.
    """
    try:
        temp = re.findall('\d', str(s))
        temp = ''.join(temp)
        return int(temp)
    except:
        return s

def _parse_tuple_string(argument):
        """ Return a tuple from parsing 'a,b,c,d' -> (a,b,c,d) """
        if isinstance(argument, str):
            return tuple(int(p.strip()) for p in argument.split(','))
        return argument

def get_column_names(engine: Engine, tablename: str) -> List[str]:
    """
    Get all the database column names for the specified table.
    """
    return [info.name for info in gen_columns_info(engine, tablename)]

def get_last_weekday_in_month(year, month, weekday):
        """Get the last weekday in a given month. e.g:

        >>> # the last monday in Jan 2013
        >>> Calendar.get_last_weekday_in_month(2013, 1, MON)
        datetime.date(2013, 1, 28)
        """
        day = date(year, month, monthrange(year, month)[1])
        while True:
            if day.weekday() == weekday:
                break
            day = day - timedelta(days=1)
        return day

def _gauss(mean: int, sigma: int) -> int:
        """
        Creates a variation from a base value

        Args:
            mean: base value
            sigma: gaussian sigma

        Returns: random value

        """
        return int(random.gauss(mean, sigma))

def fetchallfirstvalues(self, sql: str, *args) -> List[Any]:
        """Executes SQL; returns list of first values of each row."""
        rows = self.fetchall(sql, *args)
        return [row[0] for row in rows]

def fcast(value: float) -> TensorLike:
    """Cast to float tensor"""
    newvalue = tf.cast(value, FTYPE)
    if DEVICE == 'gpu':
        newvalue = newvalue.gpu()  # Why is this needed?  # pragma: no cover
    return newvalue

def PrintIndented(self, file, ident, code):
        """Takes an array, add indentation to each entry and prints it."""
        for entry in code:
            print >>file, '%s%s' % (ident, entry)

def truncate(value: Decimal, n_digits: int) -> Decimal:
    """Truncates a value to a number of decimals places"""
    return Decimal(math.trunc(value * (10 ** n_digits))) / (10 ** n_digits)

def replace_keys(record: Mapping, key_map: Mapping) -> dict:
    """New record with renamed keys including keys only found in key_map."""

    return {key_map[k]: v for k, v in record.items() if k in key_map}

def sort_by_modified(files_or_folders: list) -> list:
    """
    Sort files or folders by modified time

    Args:
        files_or_folders: list of files or folders

    Returns:
        list
    """
    return sorted(files_or_folders, key=os.path.getmtime, reverse=True)

def debugTreePrint(node,pfx="->"):
  """Purely a debugging aid: Ascii-art picture of a tree descended from node"""
  print pfx,node.item
  for c in node.children:
    debugTreePrint(c,"  "+pfx)

def argmax(self, rows: List[Row], column: ComparableColumn) -> List[Row]:
        """
        Takes a list of rows and a column name and returns a list containing a single row (dict from
        columns to cells) that has the maximum numerical value in the given column. We return a list
        instead of a single dict to be consistent with the return type of ``select`` and
        ``all_rows``.
        """
        if not rows:
            return []
        value_row_pairs = [(row.values[column.name], row) for row in rows]
        if not value_row_pairs:
            return []
        # Returns a list containing the row with the max cell value.
        return [sorted(value_row_pairs, key=lambda x: x[0], reverse=True)[0][1]]

def lower_camel_case_from_underscores(string):
    """generate a lower-cased camelCase string from an underscore_string.
    For example: my_variable_name -> myVariableName"""
    components = string.split('_')
    string = components[0]
    for component in components[1:]:
        string += component[0].upper() + component[1:]
    return string

def strtobytes(input, encoding):
    """Take a str and transform it into a byte array."""
    py_version = sys.version_info[0]
    if py_version >= 3:
        return _strtobytes_py3(input, encoding)
    return _strtobytes_py2(input, encoding)

def memory_full():
    """Check if the memory is too full for further caching."""
    current_process = psutil.Process(os.getpid())
    return (current_process.memory_percent() >
            config.MAXIMUM_CACHE_MEMORY_PERCENTAGE)

def valid_substitution(strlen, index):
    """
    skip performing substitutions that are outside the bounds of the string
    """
    values = index[0]
    return all([strlen > i for i in values])

def isarray(array, test, dim=2):
    """Returns True if test is True for all array elements.
    Otherwise, returns False.
    """
    if dim > 1:
        return all(isarray(array[i], test, dim - 1)
                   for i in range(len(array)))
    return all(test(i) for i in array)

def flatten_multidict(multidict):
    """Return flattened dictionary from ``MultiDict``."""
    return dict([(key, value if len(value) > 1 else value[0])
                 for (key, value) in multidict.iterlists()])

def chars(string: any) -> str:
        """Return all (and only) the chars in the given string."""
        return ''.join([c if c.isalpha() else '' for c in str(string)])

def _check_whitespace(string):
    """
    Make sure thre is no whitespace in the given string. Will raise a
    ValueError if whitespace is detected
    """
    if string.count(' ') + string.count('\t') + string.count('\n') > 0:
        raise ValueError(INSTRUCTION_HAS_WHITESPACE)

def _gauss(mean: int, sigma: int) -> int:
        """
        Creates a variation from a base value

        Args:
            mean: base value
            sigma: gaussian sigma

        Returns: random value

        """
        return int(random.gauss(mean, sigma))

def _run_sync(self, method: Callable, *args, **kwargs) -> Any:
        """
        Utility method to run commands synchronously for testing.
        """
        if self.loop.is_running():
            raise RuntimeError("Event loop is already running.")

        if not self.is_connected:
            self.loop.run_until_complete(self.connect())

        task = asyncio.Task(method(*args, **kwargs), loop=self.loop)
        result = self.loop.run_until_complete(task)

        self.loop.run_until_complete(self.quit())

        return result

def replace_in_list(stringlist: Iterable[str],
                    replacedict: Dict[str, str]) -> List[str]:
    """
    Returns a list produced by applying :func:`multiple_replace` to every
    string in ``stringlist``.

    Args:
        stringlist: list of source strings
        replacedict: dictionary mapping "original" to "replacement" strings

    Returns:
        list of final strings

    """
    newlist = []
    for fromstring in stringlist:
        newlist.append(multiple_replace(fromstring, replacedict))
    return newlist

def string(value) -> str:
        """ string dict/object/value to JSON """
        return system_json.dumps(Json(value).safe_object(), ensure_ascii=False)

def fast_median(a):
    """Fast median operation for masked array using 50th-percentile
    """
    a = checkma(a)
    #return scoreatpercentile(a.compressed(), 50)
    if a.count() > 0:
        out = np.percentile(a.compressed(), 50)
    else:
        out = np.ma.masked
    return out

def dict_of_sets_add(dictionary, key, value):
    # type: (DictUpperBound, Any, Any) -> None
    """Add value to a set in a dictionary by key

    Args:
        dictionary (DictUpperBound): Dictionary to which to add values
        key (Any): Key within dictionary
        value (Any): Value to add to set in dictionary

    Returns:
        None

    """
    set_objs = dictionary.get(key, set())
    set_objs.add(value)
    dictionary[key] = set_objs

def arcball_map_to_sphere(point, center, radius):
    """Return unit sphere coordinates from window coordinates."""
    v0 = (point[0] - center[0]) / radius
    v1 = (center[1] - point[1]) / radius
    n = v0*v0 + v1*v1
    if n > 1.0:
        # position outside of sphere
        n = math.sqrt(n)
        return numpy.array([v0/n, v1/n, 0.0])
    else:
        return numpy.array([v0, v1, math.sqrt(1.0 - n)])

def is_rate_limited(response):
        """
        Checks if the response has been rate limited by CARTO APIs

        :param response: The response rate limited by CARTO APIs
        :type response: requests.models.Response class

        :return: Boolean
        """
        if (response.status_code == codes.too_many_requests and 'Retry-After' in response.headers and
                int(response.headers['Retry-After']) >= 0):
            return True

        return False

def _create_empty_array(self, frames, always_2d, dtype):
        """Create an empty array with appropriate shape."""
        import numpy as np
        if always_2d or self.channels > 1:
            shape = frames, self.channels
        else:
            shape = frames,
        return np.empty(shape, dtype, order='C')

def is_integer(value: Any) -> bool:
    """Return true if a value is an integer number."""
    return (isinstance(value, int) and not isinstance(value, bool)) or (
        isinstance(value, float) and isfinite(value) and int(value) == value
    )

def rank(tensor: BKTensor) -> int:
    """Return the number of dimensions of a tensor"""
    if isinstance(tensor, np.ndarray):
        return len(tensor.shape)

    return len(tensor[0].size())

def timeit(func, *args, **kwargs):
    """
    Time execution of function. Returns (res, seconds).

    >>> res, timing = timeit(time.sleep, 1)
    """
    start_time = time.time()
    res = func(*args, **kwargs)
    timing = time.time() - start_time
    return res, timing

def pack_bits( longbits ):
    """Crunch a 64-bit int (8 bool bytes) into a bitfield."""
    byte = longbits & (0x0101010101010101)
    byte = (byte | (byte>>7)) & (0x0003000300030003)
    byte = (byte | (byte>>14)) & (0x0000000f0000000f)
    byte = (byte | (byte>>28)) & (0x00000000000000ff)
    return byte

def assert_equal(first, second, msg_fmt="{msg}"):
    """Fail unless first equals second, as determined by the '==' operator.

    >>> assert_equal(5, 5.0)
    >>> assert_equal("Hello World!", "Goodbye!")
    Traceback (most recent call last):
        ...
    AssertionError: 'Hello World!' != 'Goodbye!'

    The following msg_fmt arguments are supported:
    * msg - the default error message
    * first - the first argument
    * second - the second argument
    """

    if isinstance(first, dict) and isinstance(second, dict):
        assert_dict_equal(first, second, msg_fmt)
    elif not first == second:
        msg = "{!r} != {!r}".format(first, second)
        fail(msg_fmt.format(msg=msg, first=first, second=second))

def _parse_tuple_string(argument):
        """ Return a tuple from parsing 'a,b,c,d' -> (a,b,c,d) """
        if isinstance(argument, str):
            return tuple(int(p.strip()) for p in argument.split(','))
        return argument

def _rindex(mylist: Sequence[T], x: T) -> int:
    """Index of the last occurrence of x in the sequence."""
    return len(mylist) - mylist[::-1].index(x) - 1

def iter_lines(file_like: Iterable[str]) -> Generator[str, None, None]:
    """ Helper for iterating only nonempty lines without line breaks"""
    for line in file_like:
        line = line.rstrip('\r\n')
        if line:
            yield line

def do_quit(self, _: argparse.Namespace) -> bool:
        """Exit this application"""
        self._should_quit = True
        return self._STOP_AND_EXIT

def set_cell_value(cell, value):
    """
    Convenience method for setting the value of an openpyxl cell

    This is necessary since the value property changed from internal_value
    to value between version 1.* and 2.*.
    """
    if OPENPYXL_MAJOR_VERSION > 1:
        cell.value = value
    else:
        cell.internal_value = value

def dotproduct(X, Y):
    """Return the sum of the element-wise product of vectors x and y.
    >>> dotproduct([1, 2, 3], [1000, 100, 10])
    1230
    """
    return sum([x * y for x, y in zip(X, Y)])

def argmax(self, rows: List[Row], column: ComparableColumn) -> List[Row]:
        """
        Takes a list of rows and a column name and returns a list containing a single row (dict from
        columns to cells) that has the maximum numerical value in the given column. We return a list
        instead of a single dict to be consistent with the return type of ``select`` and
        ``all_rows``.
        """
        if not rows:
            return []
        value_row_pairs = [(row.values[column.name], row) for row in rows]
        if not value_row_pairs:
            return []
        # Returns a list containing the row with the max cell value.
        return [sorted(value_row_pairs, key=lambda x: x[0], reverse=True)[0][1]]

def _mid(pt1, pt2):
    """
    (Point, Point) -> Point
    Return the point that lies in between the two input points.
    """
    (x0, y0), (x1, y1) = pt1, pt2
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)

def maybe_infer_dtype_type(element):
    """Try to infer an object's dtype, for use in arithmetic ops

    Uses `element.dtype` if that's available.
    Objects implementing the iterator protocol are cast to a NumPy array,
    and from there the array's type is used.

    Parameters
    ----------
    element : object
        Possibly has a `.dtype` attribute, and possibly the iterator
        protocol.

    Returns
    -------
    tipo : type

    Examples
    --------
    >>> from collections import namedtuple
    >>> Foo = namedtuple("Foo", "dtype")
    >>> maybe_infer_dtype_type(Foo(np.dtype("i8")))
    numpy.int64
    """
    tipo = None
    if hasattr(element, 'dtype'):
        tipo = element.dtype
    elif is_list_like(element):
        element = np.asarray(element)
        tipo = element.dtype
    return tipo

def _parse_date(string: str) -> datetime.date:
    """Parse an ISO format date (YYYY-mm-dd).

    >>> _parse_date('1990-01-02')
    datetime.date(1990, 1, 2)
    """
    return datetime.datetime.strptime(string, '%Y-%m-%d').date()

def hash_file(fileobj):
    """
    :param fileobj: a file object
    :return: a hash of the file content
    """
    hasher = hashlib.md5()
    buf = fileobj.read(65536)
    while len(buf) > 0:
        hasher.update(buf)
        buf = fileobj.read(65536)
    return hasher.hexdigest()

def array2string(arr: numpy.ndarray) -> str:
        """Format numpy array as a string."""
        shape = str(arr.shape)[1:-1]
        if shape.endswith(","):
            shape = shape[:-1]
        return numpy.array2string(arr, threshold=11) + "%s[%s]" % (arr.dtype, shape)

def remove_empty_text(utterances: List[Utterance]) -> List[Utterance]:
    """Remove empty utterances from a list of utterances
    Args:
        utterances: The list of utterance we are processing
    """
    return [utter for utter in utterances if utter.text.strip() != ""]

def setup_cache(app: Flask, cache_config) -> Optional[Cache]:
    """Setup the flask-cache on a flask app"""
    if cache_config and cache_config.get('CACHE_TYPE') != 'null':
        return Cache(app, config=cache_config)

    return None

def read32(bytestream):
  """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def flatten_list(x: List[Any]) -> List[Any]:
    """
    Converts a list of lists into a flat list.
    
    Args:
        x: list of lists 

    Returns:
        flat list
        
    As per
    http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python

    """  # noqa
    return [item for sublist in x for item in sublist]

def camel_to_snake(s: str) -> str:
    """Convert string from camel case to snake case."""

    return CAMEL_CASE_RE.sub(r'_\1', s).strip().lower()

def mmap(func, iterable):
    """Wrapper to make map() behave the same on Py2 and Py3."""

    if sys.version_info[0] > 2:
        return [i for i in map(func, iterable)]
    else:
        return map(func, iterable)

def find_first_in_list(txt: str, str_list: [str]) -> int:  # type: ignore
    """
    Returns the index of the earliest occurence of an item from a list in a string

    Ex: find_first_in_list('foobar', ['bar', 'fin']) -> 3
    """
    start = len(txt) + 1
    for item in str_list:
        if start > txt.find(item) > -1:
            start = txt.find(item)
    return start if len(txt) + 1 > start > -1 else -1

def stretch(iterable, n=2):
    r"""Repeat each item in `iterable` `n` times.

    Example:

    >>> list(stretch(range(3), 2))
    [0, 0, 1, 1, 2, 2]
    """
    times = range(n)
    for item in iterable:
        for i in times: yield item

def string(value) -> str:
        """ string dict/object/value to JSON """
        return system_json.dumps(Json(value).safe_object(), ensure_ascii=False)

def butlast(iterable):
    """Yield all items from ``iterable`` except the last one.

    >>> list(butlast(['spam', 'eggs', 'ham']))
    ['spam', 'eggs']

    >>> list(butlast(['spam']))
    []

    >>> list(butlast([]))
    []
    """
    iterable = iter(iterable)
    try:
        first = next(iterable)
    except StopIteration:
        return
    for second in iterable:
        yield first
        first = second

def inverted_dict(d):
    """Return a dict with swapped keys and values

    >>> inverted_dict({0: ('a', 'b'), 1: 'cd'}) == {'cd': 1, ('a', 'b'): 0}
    True
    """
    return dict((force_hashable(v), k) for (k, v) in viewitems(dict(d)))

def in_transaction(self):
        """Check if this database is in a transactional context."""
        if not hasattr(self.local, 'tx'):
            return False
        return len(self.local.tx) > 0

def score_small_straight_yatzy(dice: List[int]) -> int:
    """
    Small straight scoring according to yatzy rules
    """
    dice_set = set(dice)
    if _are_two_sets_equal({1, 2, 3, 4, 5}, dice_set):
        return sum(dice)
    return 0

def _str_to_list(value, separator):
    """Convert a string to a list with sanitization."""
    value_list = [item.strip() for item in value.split(separator)]
    value_list_sanitized = builtins.list(filter(None, value_list))
    if len(value_list_sanitized) > 0:
        return value_list_sanitized
    else:
        raise ValueError('Invalid list variable.')

def most_frequent(lst):
    """
    Returns the item that appears most frequently in the given list.
    """
    lst = lst[:]
    highest_freq = 0
    most_freq = None

    for val in unique(lst):
        if lst.count(val) > highest_freq:
            most_freq = val
            highest_freq = lst.count(val)
            
    return most_freq

def remove_empty_text(utterances: List[Utterance]) -> List[Utterance]:
    """Remove empty utterances from a list of utterances
    Args:
        utterances: The list of utterance we are processing
    """
    return [utter for utter in utterances if utter.text.strip() != ""]

def read32(bytestream):
  """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def dtypes(self):
        """Returns all column names and their data types as a list.

        >>> df.dtypes
        [('age', 'int'), ('name', 'string')]
        """
        return [(str(f.name), f.dataType.simpleString()) for f in self.schema.fields]

def clean(ctx, text):
    """
    Removes all non-printable characters from a text string
    """
    text = conversions.to_string(text, ctx)
    return ''.join([c for c in text if ord(c) >= 32])

def process_literal_param(self, value: Optional[List[int]],
                              dialect: Dialect) -> str:
        """Convert things on the way from Python to the database."""
        retval = self._intlist_to_dbstr(value)
        return retval

def label_from_bin(buf):
    """
    Converts binary representation label to integer.

    :param buf: Binary representation of label.
    :return: MPLS Label and BoS bit.
    """

    mpls_label = type_desc.Int3.to_user(six.binary_type(buf))
    return mpls_label >> 4, mpls_label & 1

def has_table(self, name):
        """Return ``True`` if the table *name* exists in the database."""
        return len(self.sql("SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                            parameters=(name,), asrecarray=False, cache=False)) > 0

def is_integer(value: Any) -> bool:
    """Return true if a value is an integer number."""
    return (isinstance(value, int) and not isinstance(value, bool)) or (
        isinstance(value, float) and isfinite(value) and int(value) == value
    )

def flatten_list(l: List[list]) -> list:
    """ takes a list of lists, l and returns a flat list
    """
    return [v for inner_l in l for v in inner_l]

def __remove_trailing_zeros(self, collection):
        """Removes trailing zeroes from indexable collection of numbers"""
        index = len(collection) - 1
        while index >= 0 and collection[index] == 0:
            index -= 1

        return collection[:index + 1]

def rl_get_point() -> int:  # pragma: no cover
    """
    Returns the offset of the current cursor position in rl_line_buffer
    """
    if rl_type == RlType.GNU:
        return ctypes.c_int.in_dll(readline_lib, "rl_point").value

    elif rl_type == RlType.PYREADLINE:
        return readline.rl.mode.l_buffer.point

    else:
        return 0

def top(self, topn=10):
        """
        Get a list of the top ``topn`` features in this :class:`.Feature`\.

        Examples
        --------

        .. code-block:: python

        >>> myFeature = Feature([('the', 2), ('pine', 1), ('trapezoid', 5)])
        >>> myFeature.top(1)
        [('trapezoid', 5)]

        Parameters
        ----------
        topn : int

        Returns
        -------
        list
        """
        return [self[i] for i in argsort(list(zip(*self))[1])[::-1][:topn]]

def camel_to_snake(s: str) -> str:
    """Convert string from camel case to snake case."""

    return CAMEL_CASE_RE.sub(r'_\1', s).strip().lower()

def try_cast_int(s):
    """(str) -> int
    All the digits in a given string are concatenated and converted into a single number.
    """
    try:
        temp = re.findall('\d', str(s))
        temp = ''.join(temp)
        return int(temp)
    except:
        return s

def has_key(cls, *args):
        """
        Check whether flyweight object with specified key has already been created.

        Returns:
            bool: True if already created, False if not
        """
        key = args if len(args) > 1 else args[0]
        return key in cls._instances

def isarray(array, test, dim=2):
    """Returns True if test is True for all array elements.
    Otherwise, returns False.
    """
    if dim > 1:
        return all(isarray(array[i], test, dim - 1)
                   for i in range(len(array)))
    return all(test(i) for i in array)

def remove_blank_lines(string):
    """ Removes all blank lines in @string

        -> #str without blank lines
    """
    return "\n".join(line
                     for line in string.split("\n")
                     if len(line.strip()))

def dict_of_sets_add(dictionary, key, value):
    # type: (DictUpperBound, Any, Any) -> None
    """Add value to a set in a dictionary by key

    Args:
        dictionary (DictUpperBound): Dictionary to which to add values
        key (Any): Key within dictionary
        value (Any): Value to add to set in dictionary

    Returns:
        None

    """
    set_objs = dictionary.get(key, set())
    set_objs.add(value)
    dictionary[key] = set_objs

def s3_get(url: str, temp_file: IO) -> None:
    """Pull a file directly from S3."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)

def read_text_from_file(path: str) -> str:
    """ Reads text file contents """
    with open(path) as text_file:
        content = text_file.read()

    return content

def _check_whitespace(string):
    """
    Make sure thre is no whitespace in the given string. Will raise a
    ValueError if whitespace is detected
    """
    if string.count(' ') + string.count('\t') + string.count('\n') > 0:
        raise ValueError(INSTRUCTION_HAS_WHITESPACE)

def get_timezone() -> Tuple[datetime.tzinfo, str]:
    """Discover the current time zone and it's standard string representation (for source{d})."""
    dt = get_datetime_now().astimezone()
    tzstr = dt.strftime("%z")
    tzstr = tzstr[:-2] + ":" + tzstr[-2:]
    return dt.tzinfo, tzstr

def clean(ctx, text):
    """
    Removes all non-printable characters from a text string
    """
    text = conversions.to_string(text, ctx)
    return ''.join([c for c in text if ord(c) >= 32])

def is_unicode(string):
    """Validates that the object itself is some kinda string"""
    str_type = str(type(string))

    if str_type.find('str') > 0 or str_type.find('unicode') > 0:
        return True

    return False

def rate_limited(max_per_hour: int, *args: Any) -> Callable[..., Any]:
    """Rate limit a function."""
    return util.rate_limited(max_per_hour, *args)

def get_last_weekday_in_month(year, month, weekday):
        """Get the last weekday in a given month. e.g:

        >>> # the last monday in Jan 2013
        >>> Calendar.get_last_weekday_in_month(2013, 1, MON)
        datetime.date(2013, 1, 28)
        """
        day = date(year, month, monthrange(year, month)[1])
        while True:
            if day.weekday() == weekday:
                break
            day = day - timedelta(days=1)
        return day

def is_unitary(matrix: np.ndarray) -> bool:
    """
    A helper function that checks if a matrix is unitary.

    :param matrix: a matrix to test unitarity of
    :return: true if and only if matrix is unitary
    """
    rows, cols = matrix.shape
    if rows != cols:
        return False
    return np.allclose(np.eye(rows), matrix.dot(matrix.T.conj()))

def SGT(self, a, b):
        """Signed greater-than comparison"""
        # http://gavwood.com/paper.pdf
        s0, s1 = to_signed(a), to_signed(b)
        return Operators.ITEBV(256, s0 > s1, 1, 0)

def snake_to_camel(s: str) -> str:
    """Convert string from snake case to camel case."""

    fragments = s.split('_')

    return fragments[0] + ''.join(x.title() for x in fragments[1:])

def find_first_in_list(txt: str, str_list: [str]) -> int:  # type: ignore
    """
    Returns the index of the earliest occurence of an item from a list in a string

    Ex: find_first_in_list('foobar', ['bar', 'fin']) -> 3
    """
    start = len(txt) + 1
    for item in str_list:
        if start > txt.find(item) > -1:
            start = txt.find(item)
    return start if len(txt) + 1 > start > -1 else -1

def replaceStrs(s, *args):
    r"""Replace all ``(frm, to)`` tuples in `args` in string `s`.

    >>> replaceStrs("nothing is better than warm beer",
    ...             ('nothing','warm beer'), ('warm beer','nothing'))
    'warm beer is better than nothing'

    """
    if args == (): return s
    mapping = dict((frm, to) for frm, to in args)
    return re.sub("|".join(map(re.escape, mapping.keys())),
                  lambda match:mapping[match.group(0)], s)

def chars(string: any) -> str:
        """Return all (and only) the chars in the given string."""
        return ''.join([c if c.isalpha() else '' for c in str(string)])

def file_exists(fname):
    """Check if a file exists and is non-empty.
    """
    try:
        return fname and os.path.exists(fname) and os.path.getsize(fname) > 0
    except OSError:
        return False

def cli_run():
    """docstring for argparse"""
    parser = argparse.ArgumentParser(description='Stupidly simple code answers from StackOverflow')
    parser.add_argument('query', help="What's the problem ?", type=str, nargs='+')
    parser.add_argument('-t','--tags', help='semicolon separated tags -> python;lambda')
    args = parser.parse_args()
    main(args)

def _reshuffle(mat, shape):
    """Reshuffle the indicies of a bipartite matrix A[ij,kl] -> A[lj,ki]."""
    return np.reshape(
        np.transpose(np.reshape(mat, shape), (3, 1, 2, 0)),
        (shape[3] * shape[1], shape[0] * shape[2]))

def remove_falsy_values(counter: Mapping[Any, int]) -> Mapping[Any, int]:
    """Remove all values that are zero."""
    return {
        label: count
        for label, count in counter.items()
        if count
    }

async def executemany(self, sql: str, parameters: Iterable[Iterable[Any]]) -> None:
        """Execute the given multiquery."""
        await self._execute(self._cursor.executemany, sql, parameters)

def normcdf(x, log=False):
    """Normal cumulative density function."""
    y = np.atleast_1d(x).copy()
    flib.normcdf(y)
    if log:
        if (y>0).all():
            return np.log(y)
        return -np.inf
    return y

def pack_bits( longbits ):
    """Crunch a 64-bit int (8 bool bytes) into a bitfield."""
    byte = longbits & (0x0101010101010101)
    byte = (byte | (byte>>7)) & (0x0003000300030003)
    byte = (byte | (byte>>14)) & (0x0000000f0000000f)
    byte = (byte | (byte>>28)) & (0x00000000000000ff)
    return byte

def _parse_date(string: str) -> datetime.date:
    """Parse an ISO format date (YYYY-mm-dd).

    >>> _parse_date('1990-01-02')
    datetime.date(1990, 1, 2)
    """
    return datetime.datetime.strptime(string, '%Y-%m-%d').date()

def is_relative_url(url):
    """ simple method to determine if a url is relative or absolute """
    if url.startswith("#"):
        return None
    if url.find("://") > 0 or url.startswith("//"):
        # either 'http(s)://...' or '//cdn...' and therefore absolute
        return False
    return True

def normalize(numbers):
    """Multiply each number by a constant such that the sum is 1.0
    >>> normalize([1,2,1])
    [0.25, 0.5, 0.25]
    """
    total = float(sum(numbers))
    return [n / total for n in numbers]

def zlib_compress(data):
    """
    Compress things in a py2/3 safe fashion
    >>> json_str = '{"test": 1}'
    >>> blob = zlib_compress(json_str)
    """
    if PY3K:
        if isinstance(data, str):
            return zlib.compress(bytes(data, 'utf-8'))
        return zlib.compress(data)
    return zlib.compress(data)

def _run_sync(self, method: Callable, *args, **kwargs) -> Any:
        """
        Utility method to run commands synchronously for testing.
        """
        if self.loop.is_running():
            raise RuntimeError("Event loop is already running.")

        if not self.is_connected:
            self.loop.run_until_complete(self.connect())

        task = asyncio.Task(method(*args, **kwargs), loop=self.loop)
        result = self.loop.run_until_complete(task)

        self.loop.run_until_complete(self.quit())

        return result

def get_cursor(self):
        """Return the virtual cursor position.

        The cursor can be moved with the :any:`move` method.

        Returns:
            Tuple[int, int]: The (x, y) coordinate of where :any:`print_str`
                will continue from.

        .. seealso:: :any:move`
        """
        x, y = self._cursor
        width, height = self.parent.get_size()
        while x >= width:
            x -= width
            y += 1
        if y >= height and self.scrollMode == 'scroll':
            y = height - 1
        return x, y

def file_lines(bblfile:str) -> iter:
    """Yield lines found in given file"""
    with open(bblfile) as fd:
        yield from (line.rstrip() for line in fd if line.rstrip())

def iter_lines(file_like: Iterable[str]) -> Generator[str, None, None]:
    """ Helper for iterating only nonempty lines without line breaks"""
    for line in file_like:
        line = line.rstrip('\r\n')
        if line:
            yield line

def _tree_line(self, no_type: bool = False) -> str:
        """Return the receiver's contribution to tree diagram."""
        return self._tree_line_prefix() + " " + self.iname()

def issubset(self, other):
        """
        Report whether another set contains this set.

        Example:
            >>> OrderedSet([1, 2, 3]).issubset({1, 2})
            False
            >>> OrderedSet([1, 2, 3]).issubset({1, 2, 3, 4})
            True
            >>> OrderedSet([1, 2, 3]).issubset({1, 4, 3, 5})
            False
        """
        if len(self) > len(other):  # Fast check for obvious cases
            return False
        return all(item in other for item in self)

def get_domain(url):
    """
    Get domain part of an url.

    For example: https://www.python.org/doc/ -> https://www.python.org
    """
    parse_result = urlparse(url)
    domain = "{schema}://{netloc}".format(
        schema=parse_result.scheme, netloc=parse_result.netloc)
    return domain

def rmglob(pattern: str) -> None:
    """
    Deletes all files whose filename matches the glob ``pattern`` (via
    :func:`glob.glob`).
    """
    for f in glob.glob(pattern):
        os.remove(f)

def is_natural(x):
    """A non-negative integer."""
    try:
        is_integer = int(x) == x
    except (TypeError, ValueError):
        return False
    return is_integer and x >= 0

def fcast(value: float) -> TensorLike:
    """Cast to float tensor"""
    newvalue = tf.cast(value, FTYPE)
    if DEVICE == 'gpu':
        newvalue = newvalue.gpu()  # Why is this needed?  # pragma: no cover
    return newvalue

def proper_round(n):
    """
    rounds float to closest int
    :rtype: int
    :param n: float
    """
    return int(n) + (n / abs(n)) * int(abs(n - int(n)) >= 0.5) if n != 0 else 0

def url_host(url: str) -> str:
    """
    Parses hostname from URL.
    :param url: URL
    :return: hostname
    """
    from urllib.parse import urlparse
    res = urlparse(url)
    return res.netloc.split(':')[0] if res.netloc else ''

def count(args):
    """ count occurences in a list of lists
    >>> count([['a','b'],['a']])
    defaultdict(int, {'a' : 2, 'b' : 1})
    """
    counts = defaultdict(int)
    for arg in args:
        for item in arg:
            counts[item] = counts[item] + 1
    return counts

def stretch(iterable, n=2):
    r"""Repeat each item in `iterable` `n` times.

    Example:

    >>> list(stretch(range(3), 2))
    [0, 0, 1, 1, 2, 2]
    """
    times = range(n)
    for item in iterable:
        for i in times: yield item

def _gauss(mean: int, sigma: int) -> int:
        """
        Creates a variation from a base value

        Args:
            mean: base value
            sigma: gaussian sigma

        Returns: random value

        """
        return int(random.gauss(mean, sigma))

def get_keys_of_max_n(dict_obj, n):
    """Returns the keys that maps to the top n max values in the given dict.

    Example:
    --------
    >>> dict_obj = {'a':2, 'b':1, 'c':5}
    >>> get_keys_of_max_n(dict_obj, 2)
    ['a', 'c']
    """
    return sorted([
        item[0]
        for item in sorted(
            dict_obj.items(), key=lambda item: item[1], reverse=True
        )[:n]
    ])

def flatten_multidict(multidict):
    """Return flattened dictionary from ``MultiDict``."""
    return dict([(key, value if len(value) > 1 else value[0])
                 for (key, value) in multidict.iterlists()])

def release_lock():
    """Release lock on compilation directory."""
    get_lock.n_lock -= 1
    assert get_lock.n_lock >= 0
    # Only really release lock once all lock requests have ended.
    if get_lock.lock_is_enabled and get_lock.n_lock == 0:
        get_lock.start_time = None
        get_lock.unlocker.unlock()

def s3_get(url: str, temp_file: IO) -> None:
    """Pull a file directly from S3."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)

def _find_conda():
    """Find the conda executable robustly across conda versions.

    Returns
    -------
    conda : str
        Path to the conda executable.

    Raises
    ------
    IOError
        If the executable cannot be found in either the CONDA_EXE environment
        variable or in the PATH.

    Notes
    -----
    In POSIX platforms in conda >= 4.4, conda can be set up as a bash function
    rather than an executable. (This is to enable the syntax
    ``conda activate env-name``.) In this case, the environment variable
    ``CONDA_EXE`` contains the path to the conda executable. In other cases,
    we use standard search for the appropriate name in the PATH.

    See https://github.com/airspeed-velocity/asv/issues/645 for more details.
    """
    if 'CONDA_EXE' in os.environ:
        conda = os.environ['CONDA_EXE']
    else:
        conda = util.which('conda')
    return conda

def strings_to_integers(strings: Iterable[str]) -> Iterable[int]:
    """
    Convert a list of strings to a list of integers.

    :param strings: a list of string
    :return: a list of converted integers

    .. doctest::

        >>> strings_to_integers(['1', '1.0', '-0.2'])
        [1, 1, 0]
    """
    return strings_to_(strings, lambda x: int(float(x)))

def truncate_string(value, max_width=None):
    """Truncate string values."""
    if isinstance(value, text_type) and max_width is not None and len(value) > max_width:
        return value[:max_width]
    return value

def check_key(self, key: str) -> bool:
        """
        Checks if key exists in datastore. True if yes, False if no.

        :param: SHA512 hash key

        :return: whether or key not exists in datastore
        """
        keys = self.get_keys()
        return key in keys

def split_unit(value):
    """ Split a number from its unit
        1px -> (q, 'px')
    Args:
        value (str): input
    returns:
        tuple
    """
    r = re.search('^(\-?[\d\.]+)(.*)$', str(value))
    return r.groups() if r else ('', '')

def dict_of_sets_add(dictionary, key, value):
    # type: (DictUpperBound, Any, Any) -> None
    """Add value to a set in a dictionary by key

    Args:
        dictionary (DictUpperBound): Dictionary to which to add values
        key (Any): Key within dictionary
        value (Any): Value to add to set in dictionary

    Returns:
        None

    """
    set_objs = dictionary.get(key, set())
    set_objs.add(value)
    dictionary[key] = set_objs

def get_versions(reporev=True):
    """Get version information for components used by Spyder"""
    import sys
    import platform

    import qtpy
    import qtpy.QtCore

    revision = None
    if reporev:
        from spyder.utils import vcs
        revision, branch = vcs.get_git_revision(os.path.dirname(__dir__))

    if not sys.platform == 'darwin':  # To avoid a crash with our Mac app
        system = platform.system()
    else:
        system = 'Darwin'

    return {
        'spyder': __version__,
        'python': platform.python_version(),  # "2.7.3"
        'bitness': 64 if sys.maxsize > 2**32 else 32,
        'qt': qtpy.QtCore.__version__,
        'qt_api': qtpy.API_NAME,      # PyQt5
        'qt_api_ver': qtpy.PYQT_VERSION,
        'system': system,   # Linux, Windows, ...
        'release': platform.release(),  # XP, 10.6, 2.2.0, etc.
        'revision': revision,  # '9fdf926eccce'
    }

def _find_conda():
    """Find the conda executable robustly across conda versions.

    Returns
    -------
    conda : str
        Path to the conda executable.

    Raises
    ------
    IOError
        If the executable cannot be found in either the CONDA_EXE environment
        variable or in the PATH.

    Notes
    -----
    In POSIX platforms in conda >= 4.4, conda can be set up as a bash function
    rather than an executable. (This is to enable the syntax
    ``conda activate env-name``.) In this case, the environment variable
    ``CONDA_EXE`` contains the path to the conda executable. In other cases,
    we use standard search for the appropriate name in the PATH.

    See https://github.com/airspeed-velocity/asv/issues/645 for more details.
    """
    if 'CONDA_EXE' in os.environ:
        conda = os.environ['CONDA_EXE']
    else:
        conda = util.which('conda')
    return conda

def remove_falsy_values(counter: Mapping[Any, int]) -> Mapping[Any, int]:
    """Remove all values that are zero."""
    return {
        label: count
        for label, count in counter.items()
        if count
    }

def rms(x):
    """"Root Mean Square"

    Arguments:
        x (seq of float): A sequence of numerical values

    Returns:
        The square root of the average of the squares of the values

        math.sqrt(sum(x_i**2 for x_i in x) / len(x))

        or

        return (np.array(x) ** 2).mean() ** 0.5

    >>> rms([0, 2, 4, 4])
    3.0
    """
    try:
        return (np.array(x) ** 2).mean() ** 0.5
    except:
        x = np.array(dropna(x))
        invN = 1.0 / len(x)
        return (sum(invN * (x_i ** 2) for x_i in x)) ** .5

def templategetter(tmpl):
    """
    This is a dirty little template function generator that turns single-brace
    Mustache-style template strings into functions that interpolate dict keys:

    >>> get_name = templategetter("{first} {last}")
    >>> get_name({'first': 'Shawn', 'last': 'Allen'})
    'Shawn Allen'
    """
    tmpl = tmpl.replace('{', '%(')
    tmpl = tmpl.replace('}', ')s')
    return lambda data: tmpl % data

def flatten_list(x: List[Any]) -> List[Any]:
    """
    Converts a list of lists into a flat list.
    
    Args:
        x: list of lists 

    Returns:
        flat list
        
    As per
    http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python

    """  # noqa
    return [item for sublist in x for item in sublist]

def to_clipboard(self, excel=True, sep=None, **kwargs):
        r"""
        Copy object to the system clipboard.

        Write a text representation of object to the system clipboard.
        This can be pasted into Excel, for example.

        Parameters
        ----------
        excel : bool, default True
            - True, use the provided separator, writing in a csv format for
              allowing easy pasting into excel.
            - False, write a string representation of the object to the
              clipboard.

        sep : str, default ``'\t'``
            Field delimiter.
        **kwargs
            These parameters will be passed to DataFrame.to_csv.

        See Also
        --------
        DataFrame.to_csv : Write a DataFrame to a comma-separated values
            (csv) file.
        read_clipboard : Read text from clipboard and pass to read_table.

        Notes
        -----
        Requirements for your platform.

          - Linux : `xclip`, or `xsel` (with `gtk` or `PyQt4` modules)
          - Windows : none
          - OS X : none

        Examples
        --------
        Copy the contents of a DataFrame to the clipboard.

        >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['A', 'B', 'C'])
        >>> df.to_clipboard(sep=',')
        ... # Wrote the following to the system clipboard:
        ... # ,A,B,C
        ... # 0,1,2,3
        ... # 1,4,5,6

        We can omit the the index by passing the keyword `index` and setting
        it to false.

        >>> df.to_clipboard(sep=',', index=False)
        ... # Wrote the following to the system clipboard:
        ... # A,B,C
        ... # 1,2,3
        ... # 4,5,6
        """
        from pandas.io import clipboards
        clipboards.to_clipboard(self, excel=excel, sep=sep, **kwargs)

def pack_bits( longbits ):
    """Crunch a 64-bit int (8 bool bytes) into a bitfield."""
    byte = longbits & (0x0101010101010101)
    byte = (byte | (byte>>7)) & (0x0003000300030003)
    byte = (byte | (byte>>14)) & (0x0000000f0000000f)
    byte = (byte | (byte>>28)) & (0x00000000000000ff)
    return byte

def memory_full():
    """Check if the memory is too full for further caching."""
    current_process = psutil.Process(os.getpid())
    return (current_process.memory_percent() >
            config.MAXIMUM_CACHE_MEMORY_PERCENTAGE)

def has_value(cls, value: int) -> bool:
        """True if specified value exists in int enum; otherwise, False."""
        return any(value == item.value for item in cls)

def get_datatype(self, table: str, column: str) -> str:
        """Returns database SQL datatype for a column: e.g. VARCHAR."""
        return self.flavour.get_datatype(self, table, column).upper()

def dict_to_enum_fn(d: Dict[str, Any], enum_class: Type[Enum]) -> Enum:
    """
    Converts an ``dict`` to a ``Enum``.
    """
    return enum_class[d['name']]

def __replace_all(repls: dict, input: str) -> str:
    """ Replaces from a string **input** all the occurrences of some
    symbols according to mapping **repls**.

    :param dict repls: where #key is the old character and
    #value is the one to substitute with;
    :param str input: original string where to apply the
    replacements;
    :return: *(str)* the string with the desired characters replaced
    """
    return re.sub('|'.join(re.escape(key) for key in repls.keys()),
                  lambda k: repls[k.group(0)], input)

def str_to_time(time_str: str) -> datetime.datetime:
    """
    Convert human readable string to datetime.datetime.
    """
    pieces: Any = [int(piece) for piece in time_str.split('-')]
    return datetime.datetime(*pieces)

def SwitchToThisWindow(handle: int) -> None:
    """
    SwitchToThisWindow from Win32.
    handle: int, the handle of a native window.
    """
    ctypes.windll.user32.SwitchToThisWindow(ctypes.c_void_p(handle), 1)

def fcast(value: float) -> TensorLike:
    """Cast to float tensor"""
    newvalue = tf.cast(value, FTYPE)
    if DEVICE == 'gpu':
        newvalue = newvalue.gpu()  # Why is this needed?  # pragma: no cover
    return newvalue

def get_versions(reporev=True):
    """Get version information for components used by Spyder"""
    import sys
    import platform

    import qtpy
    import qtpy.QtCore

    revision = None
    if reporev:
        from spyder.utils import vcs
        revision, branch = vcs.get_git_revision(os.path.dirname(__dir__))

    if not sys.platform == 'darwin':  # To avoid a crash with our Mac app
        system = platform.system()
    else:
        system = 'Darwin'

    return {
        'spyder': __version__,
        'python': platform.python_version(),  # "2.7.3"
        'bitness': 64 if sys.maxsize > 2**32 else 32,
        'qt': qtpy.QtCore.__version__,
        'qt_api': qtpy.API_NAME,      # PyQt5
        'qt_api_ver': qtpy.PYQT_VERSION,
        'system': system,   # Linux, Windows, ...
        'release': platform.release(),  # XP, 10.6, 2.2.0, etc.
        'revision': revision,  # '9fdf926eccce'
    }

def astensor(array: TensorLike) -> BKTensor:
    """Covert numpy array to tensorflow tensor"""
    tensor = tf.convert_to_tensor(value=array, dtype=CTYPE)
    return tensor

def pretty_dict(d):
    """Return dictionary d's repr but with the items sorted.
    >>> pretty_dict({'m': 'M', 'a': 'A', 'r': 'R', 'k': 'K'})
    "{'a': 'A', 'k': 'K', 'm': 'M', 'r': 'R'}"
    >>> pretty_dict({z: C, y: B, x: A})
    '{x: A, y: B, z: C}'
    """
    return '{%s}' % ', '.join('%r: %r' % (k, v)
                              for k, v in sorted(d.items(), key=repr))

def array_to_npy(array_like):  # type: (np.array or Iterable or int or float) -> object
    """Convert an array like object to the NPY format.

    To understand better what an array like object is see:
    https://docs.scipy.org/doc/numpy/user/basics.creation.html#converting-python-array-like-objects-to-numpy-arrays

    Args:
        array_like (np.array or Iterable or int or float): array like object to be converted to NPY.

    Returns:
        (obj): NPY array.
    """
    buffer = BytesIO()
    np.save(buffer, array_like)
    return buffer.getvalue()

def read_byte_data(self, addr, cmd):
        """read_byte_data(addr, cmd) -> result

        Perform SMBus Read Byte Data transaction.
        """
        self._set_addr(addr)
        res = SMBUS.i2c_smbus_read_byte_data(self._fd, ffi.cast("__u8", cmd))
        if res == -1:
            raise IOError(ffi.errno)
        return res

def uppercase_chars(string: any) -> str:
        """Return all (and only) the uppercase chars in the given string."""
        return ''.join([c if c.isupper() else '' for c in str(string)])

def get_last_day_of_month(t: datetime) -> int:
    """
    Returns day number of the last day of the month
    :param t: datetime
    :return: int
    """
    tn = t + timedelta(days=32)
    tn = datetime(year=tn.year, month=tn.month, day=1)
    tt = tn - timedelta(hours=1)
    return tt.day

def valid_date(x: str) -> bool:
    """
    Retrun ``True`` if ``x`` is a valid YYYYMMDD date;
    otherwise return ``False``.
    """
    try:
        if x != dt.datetime.strptime(x, DATE_FORMAT).strftime(DATE_FORMAT):
            raise ValueError
        return True
    except ValueError:
        return False

def _parse_property(self, node):
        # type: (ElementTree.Element) -> Tuple[str, Any]
        """
        Parses a property node

        :param node: The property node
        :return: A (name, value) tuple
        :raise KeyError: Attribute missing
        """
        # Get information
        name = node.attrib[ATTR_NAME]
        vtype = node.attrib.get(ATTR_VALUE_TYPE, TYPE_STRING)

        # Look for a value as a single child node
        try:
            value_node = next(iter(node))
            value = self._parse_value_node(vtype, value_node)
        except StopIteration:
            # Value is an attribute
            value = self._convert_value(vtype, node.attrib[ATTR_VALUE])

        return name, value

def validate_django_compatible_with_python():
    """
    Verify Django 1.11 is present if Python 2.7 is active

    Installation of pinax-cli requires the correct version of Django for
    the active Python version. If the developer subsequently changes
    the Python version the installed Django may no longer be compatible.
    """
    python_version = sys.version[:5]
    django_version = django.get_version()
    if sys.version_info == (2, 7) and django_version >= "2":
        click.BadArgumentUsage("Please install Django v1.11 for Python {}, or switch to Python >= v3.4".format(python_version))

def is_intersection(g, n):
    """
    Determine if a node is an intersection

    graph: 1 -->-- 2 -->-- 3

    >>> is_intersection(g, 2)
    False

    graph:
     1 -- 2 -- 3
          |
          4

    >>> is_intersection(g, 2)
    True

    Parameters
    ----------
    g : networkx DiGraph
    n : node id

    Returns
    -------
    bool

    """
    return len(set(g.predecessors(n) + g.successors(n))) > 2

def interpolate(f1: float, f2: float, factor: float) -> float:
    """ Linearly interpolate between two float values. """
    return f1 + (f2 - f1) * factor

def split(text: str) -> List[str]:
    """Split a text into a list of tokens.

    :param text: the text to split
    :return: tokens
    """
    return [word for word in SEPARATOR.split(text) if word.strip(' \t')]

def strings_to_integers(strings: Iterable[str]) -> Iterable[int]:
    """
    Convert a list of strings to a list of integers.

    :param strings: a list of string
    :return: a list of converted integers

    .. doctest::

        >>> strings_to_integers(['1', '1.0', '-0.2'])
        [1, 1, 0]
    """
    return strings_to_(strings, lambda x: int(float(x)))

def clean(ctx, text):
    """
    Removes all non-printable characters from a text string
    """
    text = conversions.to_string(text, ctx)
    return ''.join([c for c in text if ord(c) >= 32])

def reduce(function, initval=None):
	"""
	Curried version of the built-in reduce.
	
	>>> reduce(lambda x,y: x+y)( [1, 2, 3, 4, 5] )
	15
	"""
	if initval is None:
		return lambda s: __builtin__.reduce(function, s)
	else:
		return lambda s: __builtin__.reduce(function, s, initval)

def get_domain(url):
    """
    Get domain part of an url.

    For example: https://www.python.org/doc/ -> https://www.python.org
    """
    parse_result = urlparse(url)
    domain = "{schema}://{netloc}".format(
        schema=parse_result.scheme, netloc=parse_result.netloc)
    return domain

def tsv_escape(x: Any) -> str:
    """
    Escape data for tab-separated value (TSV) format.
    """
    if x is None:
        return ""
    x = str(x)
    return x.replace("\t", "\\t").replace("\n", "\\n")

def memory_full():
    """Check if the memory is too full for further caching."""
    current_process = psutil.Process(os.getpid())
    return (current_process.memory_percent() >
            config.MAXIMUM_CACHE_MEMORY_PERCENTAGE)

def rate_limited(max_per_hour: int, *args: Any) -> Callable[..., Any]:
    """Rate limit a function."""
    return util.rate_limited(max_per_hour, *args)

def s3_get(url: str, temp_file: IO) -> None:
    """Pull a file directly from S3."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)

def read_set_from_file(filename: str) -> Set[str]:
    """
    Extract a de-duped collection (set) of text from a file.
    Expected file format is one item per line.
    """
    collection = set()
    with open(filename, 'r') as file_:
        for line in file_:
            collection.add(line.rstrip())
    return collection

def recClearTag(element):
    """Applies maspy.xml.clearTag() to the tag attribute of the "element" and
    recursively to all child elements.

    :param element: an :instance:`xml.etree.Element`
    """
    children = element.getchildren()
    if len(children) > 0:
        for child in children:
            recClearTag(child)
    element.tag = clearTag(element.tag)

def _str_to_list(value, separator):
    """Convert a string to a list with sanitization."""
    value_list = [item.strip() for item in value.split(separator)]
    value_list_sanitized = builtins.list(filter(None, value_list))
    if len(value_list_sanitized) > 0:
        return value_list_sanitized
    else:
        raise ValueError('Invalid list variable.')

def most_frequent(lst):
    """
    Returns the item that appears most frequently in the given list.
    """
    lst = lst[:]
    highest_freq = 0
    most_freq = None

    for val in unique(lst):
        if lst.count(val) > highest_freq:
            most_freq = val
            highest_freq = lst.count(val)
            
    return most_freq

def mmap(func, iterable):
    """Wrapper to make map() behave the same on Py2 and Py3."""

    if sys.version_info[0] > 2:
        return [i for i in map(func, iterable)]
    else:
        return map(func, iterable)

def strings_to_integers(strings: Iterable[str]) -> Iterable[int]:
    """
    Convert a list of strings to a list of integers.

    :param strings: a list of string
    :return: a list of converted integers

    .. doctest::

        >>> strings_to_integers(['1', '1.0', '-0.2'])
        [1, 1, 0]
    """
    return strings_to_(strings, lambda x: int(float(x)))

def try_cast_int(s):
    """(str) -> int
    All the digits in a given string are concatenated and converted into a single number.
    """
    try:
        temp = re.findall('\d', str(s))
        temp = ''.join(temp)
        return int(temp)
    except:
        return s

def _validate_authority_uri_abs_path(host, path):
        """Ensure that path in URL with authority starts with a leading slash.

        Raise ValueError if not.
        """
        if len(host) > 0 and len(path) > 0 and not path.startswith("/"):
            raise ValueError(
                "Path in a URL with authority " "should start with a slash ('/') if set"
            )

def warn_if_nans_exist(X):
    """Warn if nans exist in a numpy array."""
    null_count = count_rows_with_nans(X)
    total = len(X)
    percent = 100 * null_count / total

    if null_count > 0:
        warning_message = \
            'Warning! Found {} rows of {} ({:0.2f}%) with nan values. Only ' \
            'complete rows will be plotted.'.format(null_count, total, percent)
        warnings.warn(warning_message, DataWarning)

def _get_parsing_plan_for_multifile_children(self, obj_on_fs: PersistedObject, desired_type: Type[Any],
                                                 logger: Logger) -> Dict[str, Any]:
        """
        Implementation of AnyParser API
        """
        raise Exception('This should never happen, since this parser relies on underlying parsers')

def inverted_dict_of_lists(d):
    """Return a dict where the keys are all the values listed in the values of the original dict

    >>> inverted_dict_of_lists({0: ['a', 'b'], 1: 'cd'}) == {'a': 0, 'b': 0, 'cd': 1}
    True
    """
    new_dict = {}
    for (old_key, old_value_list) in viewitems(dict(d)):
        for new_key in listify(old_value_list):
            new_dict[new_key] = old_key
    return new_dict

def dotproduct(X, Y):
    """Return the sum of the element-wise product of vectors x and y.
    >>> dotproduct([1, 2, 3], [1000, 100, 10])
    1230
    """
    return sum([x * y for x, y in zip(X, Y)])

def copy_without_prompts(self):
        """Copy text to clipboard without prompts"""
        text = self.get_selected_text()
        lines = text.split(os.linesep)
        for index, line in enumerate(lines):
            if line.startswith('>>> ') or line.startswith('... '):
                lines[index] = line[4:]
        text = os.linesep.join(lines)
        QApplication.clipboard().setText(text)

def valid_file(path: str) -> bool:
    """
    Verifies that a string path actually exists and is a file

    :param path: The path to verify
    :return: **True** if path exist and is a file
    """
    path = Path(path).expanduser()
    log.debug("checking if %s is a valid file", path)
    return path.exists() and path.is_file()

def dotproduct(X, Y):
    """Return the sum of the element-wise product of vectors x and y.
    >>> dotproduct([1, 2, 3], [1000, 100, 10])
    1230
    """
    return sum([x * y for x, y in zip(X, Y)])

def maybe_infer_dtype_type(element):
    """Try to infer an object's dtype, for use in arithmetic ops

    Uses `element.dtype` if that's available.
    Objects implementing the iterator protocol are cast to a NumPy array,
    and from there the array's type is used.

    Parameters
    ----------
    element : object
        Possibly has a `.dtype` attribute, and possibly the iterator
        protocol.

    Returns
    -------
    tipo : type

    Examples
    --------
    >>> from collections import namedtuple
    >>> Foo = namedtuple("Foo", "dtype")
    >>> maybe_infer_dtype_type(Foo(np.dtype("i8")))
    numpy.int64
    """
    tipo = None
    if hasattr(element, 'dtype'):
        tipo = element.dtype
    elif is_list_like(element):
        element = np.asarray(element)
        tipo = element.dtype
    return tipo

def has_synset(word: str) -> list:
    """" Returns a list of synsets of a word after lemmatization. """

    return wn.synsets(lemmatize(word, neverstem=True))

def str_to_time(time_str: str) -> datetime.datetime:
    """
    Convert human readable string to datetime.datetime.
    """
    pieces: Any = [int(piece) for piece in time_str.split('-')]
    return datetime.datetime(*pieces)

def hex_to_int(value):
    """
    Convert hex string like "\x0A\xE3" to 2787.
    """
    if version_info.major >= 3:
        return int.from_bytes(value, "big")
    return int(value.encode("hex"), 16)

def count(args):
    """ count occurences in a list of lists
    >>> count([['a','b'],['a']])
    defaultdict(int, {'a' : 2, 'b' : 1})
    """
    counts = defaultdict(int)
    for arg in args:
        for item in arg:
            counts[item] = counts[item] + 1
    return counts

def uppercase_chars(string: any) -> str:
        """Return all (and only) the uppercase chars in the given string."""
        return ''.join([c if c.isupper() else '' for c in str(string)])

def get_from_gnucash26_date(date_str: str) -> date:
    """ Creates a datetime from GnuCash 2.6 date string """
    date_format = "%Y%m%d"
    result = datetime.strptime(date_str, date_format).date()
    return result

def partition_items(count, bin_size):
	"""
	Given the total number of items, determine the number of items that
	can be added to each bin with a limit on the bin size.

	So if you want to partition 11 items into groups of 3, you'll want
	three of three and one of two.

	>>> partition_items(11, 3)
	[3, 3, 3, 2]

	But if you only have ten items, you'll have two groups of three and
	two of two.

	>>> partition_items(10, 3)
	[3, 3, 2, 2]
	"""
	num_bins = int(math.ceil(count / float(bin_size)))
	bins = [0] * num_bins
	for i in range(count):
		bins[i % num_bins] += 1
	return bins

def _read_section(self):
        """Read and return an entire section"""
        lines = [self._last[self._last.find(":")+1:]]
        self._last = self._f.readline()
        while len(self._last) > 0 and len(self._last[0].strip()) == 0:
            lines.append(self._last)
            self._last = self._f.readline()
        return lines

def argmax(self, rows: List[Row], column: ComparableColumn) -> List[Row]:
        """
        Takes a list of rows and a column name and returns a list containing a single row (dict from
        columns to cells) that has the maximum numerical value in the given column. We return a list
        instead of a single dict to be consistent with the return type of ``select`` and
        ``all_rows``.
        """
        if not rows:
            return []
        value_row_pairs = [(row.values[column.name], row) for row in rows]
        if not value_row_pairs:
            return []
        # Returns a list containing the row with the max cell value.
        return [sorted(value_row_pairs, key=lambda x: x[0], reverse=True)[0][1]]

def most_significant_bit(lst: np.ndarray) -> int:
    """
    A helper function that finds the position of the most significant bit in a 1darray of 1s and 0s,
    i.e. the first position where a 1 appears, reading left to right.

    :param lst: a 1d array of 0s and 1s with at least one 1
    :return: the first position in lst that a 1 appears
    """
    return np.argwhere(np.asarray(lst) == 1)[0][0]

def valid_date(x: str) -> bool:
    """
    Retrun ``True`` if ``x`` is a valid YYYYMMDD date;
    otherwise return ``False``.
    """
    try:
        if x != dt.datetime.strptime(x, DATE_FORMAT).strftime(DATE_FORMAT):
            raise ValueError
        return True
    except ValueError:
        return False

def most_frequent(lst):
    """
    Returns the item that appears most frequently in the given list.
    """
    lst = lst[:]
    highest_freq = 0
    most_freq = None

    for val in unique(lst):
        if lst.count(val) > highest_freq:
            most_freq = val
            highest_freq = lst.count(val)
            
    return most_freq

def _get_ipv6_from_binary(self, bin_addr):
        """Converts binary address to Ipv6 format."""

        hi = bin_addr >> 64
        lo = bin_addr & 0xFFFFFFFF
        return socket.inet_ntop(socket.AF_INET6, struct.pack("!QQ", hi, lo))

def iprotate(l, steps=1):
    r"""Like rotate, but modifies `l` in-place.

    >>> l = [1,2,3]
    >>> iprotate(l) is l
    True
    >>> l
    [2, 3, 1]
    >>> iprotate(iprotate(l, 2), -3)
    [1, 2, 3]

    """
    if len(l):
        steps %= len(l)
        if steps:
            firstPart = l[:steps]
            del l[:steps]
            l.extend(firstPart)
    return l

def has_value(cls, value: int) -> bool:
        """True if specified value exists in int enum; otherwise, False."""
        return any(value == item.value for item in cls)

def connect_to_database_odbc_access(self,
                                        dsn: str,
                                        autocommit: bool = True) -> None:
        """Connects to an Access database via ODBC, with the DSN
        prespecified."""
        self.connect(engine=ENGINE_ACCESS, interface=INTERFACE_ODBC,
                     dsn=dsn, autocommit=autocommit)

def encode_list(key, list_):
    # type: (str, Iterable) -> Dict[str, str]
    """
    Converts a list into a space-separated string and puts it in a dictionary

    :param key: Dictionary key to store the list
    :param list_: A list of objects
    :return: A dictionary key->string or an empty dictionary
    """
    if not list_:
        return {}
    return {key: " ".join(str(i) for i in list_)}

def valid_file(path: str) -> bool:
    """
    Verifies that a string path actually exists and is a file

    :param path: The path to verify
    :return: **True** if path exist and is a file
    """
    path = Path(path).expanduser()
    log.debug("checking if %s is a valid file", path)
    return path.exists() and path.is_file()

def valid_date(x: str) -> bool:
    """
    Retrun ``True`` if ``x`` is a valid YYYYMMDD date;
    otherwise return ``False``.
    """
    try:
        if x != dt.datetime.strptime(x, DATE_FORMAT).strftime(DATE_FORMAT):
            raise ValueError
        return True
    except ValueError:
        return False

def has_enumerated_namespace_name(self, namespace: str, name: str) -> bool:
        """Check that the namespace is defined by an enumeration and that the name is a member."""
        return self.has_enumerated_namespace(namespace) and name in self.namespace_to_terms[namespace]

def list_to_str(lst):
    """
    Turn a list into a comma- and/or and-separated string.

    Parameters
    ----------
    lst : :obj:`list`
        A list of strings to join into a single string.

    Returns
    -------
    str_ : :obj:`str`
        A string with commas and/or ands separating th elements from ``lst``.

    """
    if len(lst) == 1:
        str_ = lst[0]
    elif len(lst) == 2:
        str_ = ' and '.join(lst)
    elif len(lst) > 2:
        str_ = ', '.join(lst[:-1])
        str_ += ', and {0}'.format(lst[-1])
    else:
        raise ValueError('List of length 0 provided.')
    return str_

def rate_limited(max_per_hour: int, *args: Any) -> Callable[..., Any]:
    """Rate limit a function."""
    return util.rate_limited(max_per_hour, *args)

def _check_whitespace(string):
    """
    Make sure thre is no whitespace in the given string. Will raise a
    ValueError if whitespace is detected
    """
    if string.count(' ') + string.count('\t') + string.count('\n') > 0:
        raise ValueError(INSTRUCTION_HAS_WHITESPACE)

def fcast(value: float) -> TensorLike:
    """Cast to float tensor"""
    newvalue = tf.cast(value, FTYPE)
    if DEVICE == 'gpu':
        newvalue = newvalue.gpu()  # Why is this needed?  # pragma: no cover
    return newvalue

def Exit(msg, code=1):
    """Exit execution with return code and message
    :param msg: Message displayed prior to exit
    :param code: code returned upon exiting
    """
    print >> sys.stderr, msg
    sys.exit(code)

def dtypes(self):
        """Returns all column names and their data types as a list.

        >>> df.dtypes
        [('age', 'int'), ('name', 'string')]
        """
        return [(str(f.name), f.dataType.simpleString()) for f in self.schema.fields]

def _rindex(mylist: Sequence[T], x: T) -> int:
    """Index of the last occurrence of x in the sequence."""
    return len(mylist) - mylist[::-1].index(x) - 1

def rmglob(pattern: str) -> None:
    """
    Deletes all files whose filename matches the glob ``pattern`` (via
    :func:`glob.glob`).
    """
    for f in glob.glob(pattern):
        os.remove(f)

def timeit(func, *args, **kwargs):
    """
    Time execution of function. Returns (res, seconds).

    >>> res, timing = timeit(time.sleep, 1)
    """
    start_time = time.time()
    res = func(*args, **kwargs)
    timing = time.time() - start_time
    return res, timing

def are_token_parallel(sequences: Sequence[Sized]) -> bool:
    """
    Returns True if all sequences in the list have the same length.
    """
    if not sequences or len(sequences) == 1:
        return True
    return all(len(s) == len(sequences[0]) for s in sequences)

def full(self):
        """Return ``True`` if the queue is full, ``False``
        otherwise (not reliable!).

        Only applicable if :attr:`maxsize` is set.

        """
        return self.maxsize and len(self.list) >= self.maxsize or False

def is_iterable(etype) -> bool:
    """ Determine whether etype is a List or other iterable """
    return type(etype) is GenericMeta and issubclass(etype.__extra__, Iterable)

def extend(a: dict, b: dict) -> dict:
    """Merge two dicts and return a new dict. Much like subclassing works."""
    res = a.copy()
    res.update(b)
    return res

def _skip_section(self):
        """Skip a section"""
        self._last = self._f.readline()
        while len(self._last) > 0 and len(self._last[0].strip()) == 0:
            self._last = self._f.readline()

def timeit(func, *args, **kwargs):
    """
    Time execution of function. Returns (res, seconds).

    >>> res, timing = timeit(time.sleep, 1)
    """
    start_time = time.time()
    res = func(*args, **kwargs)
    timing = time.time() - start_time
    return res, timing

def de_duplicate(items):
    """Remove any duplicate item, preserving order

    >>> de_duplicate([1, 2, 1, 2])
    [1, 2]
    """
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return result

def lower_camel_case_from_underscores(string):
    """generate a lower-cased camelCase string from an underscore_string.
    For example: my_variable_name -> myVariableName"""
    components = string.split('_')
    string = components[0]
    for component in components[1:]:
        string += component[0].upper() + component[1:]
    return string

def butlast(iterable):
    """Yield all items from ``iterable`` except the last one.

    >>> list(butlast(['spam', 'eggs', 'ham']))
    ['spam', 'eggs']

    >>> list(butlast(['spam']))
    []

    >>> list(butlast([]))
    []
    """
    iterable = iter(iterable)
    try:
        first = next(iterable)
    except StopIteration:
        return
    for second in iterable:
        yield first
        first = second

def timeit(func, *args, **kwargs):
    """
    Time execution of function. Returns (res, seconds).

    >>> res, timing = timeit(time.sleep, 1)
    """
    start_time = time.time()
    res = func(*args, **kwargs)
    timing = time.time() - start_time
    return res, timing

def getIndex(predicateFn: Callable[[T], bool], items: List[T]) -> int:
    """
    Finds the index of an item in list, which satisfies predicate
    :param predicateFn: predicate function to run on items of list
    :param items: list of tuples
    :return: first index for which predicate function returns True
    """
    try:
        return next(i for i, v in enumerate(items) if predicateFn(v))
    except StopIteration:
        return -1

async def fetchall(self) -> Iterable[sqlite3.Row]:
        """Fetch all remaining rows."""
        return await self._execute(self._cursor.fetchall)

def is_natural(x):
    """A non-negative integer."""
    try:
        is_integer = int(x) == x
    except (TypeError, ValueError):
        return False
    return is_integer and x >= 0

def _isint(string):
    """
    >>> _isint("123")
    True
    >>> _isint("123.45")
    False
    """
    return type(string) is int or \
           (isinstance(string, _binary_type) or isinstance(string, _text_type)) and \
           _isconvertible(int, string)

def file_exists(fname):
    """Check if a file exists and is non-empty.
    """
    try:
        return fname and os.path.exists(fname) and os.path.getsize(fname) > 0
    except OSError:
        return False

def _rindex(mylist: Sequence[T], x: T) -> int:
    """Index of the last occurrence of x in the sequence."""
    return len(mylist) - mylist[::-1].index(x) - 1

def dtypes(self):
        """Returns all column names and their data types as a list.

        >>> df.dtypes
        [('age', 'int'), ('name', 'string')]
        """
        return [(str(f.name), f.dataType.simpleString()) for f in self.schema.fields]

def recClearTag(element):
    """Applies maspy.xml.clearTag() to the tag attribute of the "element" and
    recursively to all child elements.

    :param element: an :instance:`xml.etree.Element`
    """
    children = element.getchildren()
    if len(children) > 0:
        for child in children:
            recClearTag(child)
    element.tag = clearTag(element.tag)

def is_unicode(string):
    """Validates that the object itself is some kinda string"""
    str_type = str(type(string))

    if str_type.find('str') > 0 or str_type.find('unicode') > 0:
        return True

    return False

def s3_get(url: str, temp_file: IO) -> None:
    """Pull a file directly from S3."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)

def timeit(func, *args, **kwargs):
    """
    Time execution of function. Returns (res, seconds).

    >>> res, timing = timeit(time.sleep, 1)
    """
    start_time = time.time()
    res = func(*args, **kwargs)
    timing = time.time() - start_time
    return res, timing

def get_line_number(line_map, offset):
    """Find a line number, given a line map and a character offset."""
    for lineno, line_offset in enumerate(line_map, start=1):
        if line_offset > offset:
            return lineno
    return -1

def _mid(pt1, pt2):
    """
    (Point, Point) -> Point
    Return the point that lies in between the two input points.
    """
    (x0, y0), (x1, y1) = pt1, pt2
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)

def snake_to_camel(value):
    """
    Converts a snake_case_string to a camelCaseString.

    >>> snake_to_camel("foo_bar_baz")
    'fooBarBaz'
    """
    camel = "".join(word.title() for word in value.split("_"))
    return value[:1].lower() + camel[1:]

def ResetConsoleColor() -> bool:
    """
    Reset to the default text color on console window.
    Return bool, True if succeed otherwise False.
    """
    if sys.stdout:
        sys.stdout.flush()
    bool(ctypes.windll.kernel32.SetConsoleTextAttribute(_ConsoleOutputHandle, _DefaultConsoleColor))

def most_significant_bit(lst: np.ndarray) -> int:
    """
    A helper function that finds the position of the most significant bit in a 1darray of 1s and 0s,
    i.e. the first position where a 1 appears, reading left to right.

    :param lst: a 1d array of 0s and 1s with at least one 1
    :return: the first position in lst that a 1 appears
    """
    return np.argwhere(np.asarray(lst) == 1)[0][0]

def Exit(msg, code=1):
    """Exit execution with return code and message
    :param msg: Message displayed prior to exit
    :param code: code returned upon exiting
    """
    print >> sys.stderr, msg
    sys.exit(code)

def has_synset(word: str) -> list:
    """" Returns a list of synsets of a word after lemmatization. """

    return wn.synsets(lemmatize(word, neverstem=True))

def _gauss(mean: int, sigma: int) -> int:
        """
        Creates a variation from a base value

        Args:
            mean: base value
            sigma: gaussian sigma

        Returns: random value

        """
        return int(random.gauss(mean, sigma))

def getCollectDServer(queue, cfg):
    """Get the appropriate collectd server (multi processed or not)"""
    server = CollectDServerMP if cfg.collectd_workers > 1 else CollectDServer
    return server(queue, cfg)

def last_location_of_minimum(x):
    """
    Returns the last location of the minimal value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    return 1.0 - np.argmin(x[::-1]) / len(x) if len(x) > 0 else np.NaN

def s3_get(url: str, temp_file: IO) -> None:
    """Pull a file directly from S3."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)

def flatten_list(l: List[list]) -> list:
    """ takes a list of lists, l and returns a flat list
    """
    return [v for inner_l in l for v in inner_l]

def has_key(cls, *args):
        """
        Check whether flyweight object with specified key has already been created.

        Returns:
            bool: True if already created, False if not
        """
        key = args if len(args) > 1 else args[0]
        return key in cls._instances

def most_significant_bit(lst: np.ndarray) -> int:
    """
    A helper function that finds the position of the most significant bit in a 1darray of 1s and 0s,
    i.e. the first position where a 1 appears, reading left to right.

    :param lst: a 1d array of 0s and 1s with at least one 1
    :return: the first position in lst that a 1 appears
    """
    return np.argwhere(np.asarray(lst) == 1)[0][0]

def val_mb(valstr: Union[int, str]) -> str:
    """
    Converts a value in bytes (in string format) to megabytes.
    """
    try:
        return "{:.3f}".format(int(valstr) / (1024 * 1024))
    except (TypeError, ValueError):
        return '?'

def rate_limited(max_per_hour: int, *args: Any) -> Callable[..., Any]:
    """Rate limit a function."""
    return util.rate_limited(max_per_hour, *args)

def moving_average(iterable, n):
    """
    From Python collections module documentation

    moving_average([40, 30, 50, 46, 39, 44]) --> 40.0 42.0 45.0 43.0
    """
    it = iter(iterable)
    d = collections.deque(itertools.islice(it, n - 1))
    d.appendleft(0)
    s = sum(d)
    for elem in it:
        s += elem - d.popleft()
        d.append(elem)
        yield s / float(n)

def encode_list(key, list_):
    # type: (str, Iterable) -> Dict[str, str]
    """
    Converts a list into a space-separated string and puts it in a dictionary

    :param key: Dictionary key to store the list
    :param list_: A list of objects
    :return: A dictionary key->string or an empty dictionary
    """
    if not list_:
        return {}
    return {key: " ".join(str(i) for i in list_)}

def _isint(string):
    """
    >>> _isint("123")
    True
    >>> _isint("123.45")
    False
    """
    return type(string) is int or \
           (isinstance(string, _binary_type) or isinstance(string, _text_type)) and \
           _isconvertible(int, string)

def _close(self):
        """
        Release the USB interface again.
        """
        self._usb_handle.releaseInterface()
        try:
            # If we're using PyUSB >= 1.0 we can re-attach the kernel driver here.
            self._usb_handle.dev.attach_kernel_driver(0)
        except:
            pass
        self._usb_int = None
        self._usb_handle = None
        return True

def dag_longest_path(graph, source, target):
    """
    Finds the longest path in a dag between two nodes
    """
    if source == target:
        return [source]
    allpaths = nx.all_simple_paths(graph, source, target)
    longest_path = []
    for l in allpaths:
        if len(l) > len(longest_path):
            longest_path = l
    return longest_path

def most_frequent(lst):
    """
    Returns the item that appears most frequently in the given list.
    """
    lst = lst[:]
    highest_freq = 0
    most_freq = None

    for val in unique(lst):
        if lst.count(val) > highest_freq:
            most_freq = val
            highest_freq = lst.count(val)
            
    return most_freq

def normalize(numbers):
    """Multiply each number by a constant such that the sum is 1.0
    >>> normalize([1,2,1])
    [0.25, 0.5, 0.25]
    """
    total = float(sum(numbers))
    return [n / total for n in numbers]

def _my_hash(arg_list):
    # type: (List[Any]) -> int
    """Simple helper hash function"""
    res = 0
    for arg in arg_list:
        res = res * 31 + hash(arg)
    return res

def truncate(value: Decimal, n_digits: int) -> Decimal:
    """Truncates a value to a number of decimals places"""
    return Decimal(math.trunc(value * (10 ** n_digits))) / (10 ** n_digits)

def debugTreePrint(node,pfx="->"):
  """Purely a debugging aid: Ascii-art picture of a tree descended from node"""
  print pfx,node.item
  for c in node.children:
    debugTreePrint(c,"  "+pfx)

def dict_of_sets_add(dictionary, key, value):
    # type: (DictUpperBound, Any, Any) -> None
    """Add value to a set in a dictionary by key

    Args:
        dictionary (DictUpperBound): Dictionary to which to add values
        key (Any): Key within dictionary
        value (Any): Value to add to set in dictionary

    Returns:
        None

    """
    set_objs = dictionary.get(key, set())
    set_objs.add(value)
    dictionary[key] = set_objs

def camel_to_snake(s: str) -> str:
    """Convert string from camel case to snake case."""

    return CAMEL_CASE_RE.sub(r'_\1', s).strip().lower()

def strings_to_integers(strings: Iterable[str]) -> Iterable[int]:
    """
    Convert a list of strings to a list of integers.

    :param strings: a list of string
    :return: a list of converted integers

    .. doctest::

        >>> strings_to_integers(['1', '1.0', '-0.2'])
        [1, 1, 0]
    """
    return strings_to_(strings, lambda x: int(float(x)))

def most_frequent(lst):
    """
    Returns the item that appears most frequently in the given list.
    """
    lst = lst[:]
    highest_freq = 0
    most_freq = None

    for val in unique(lst):
        if lst.count(val) > highest_freq:
            most_freq = val
            highest_freq = lst.count(val)
            
    return most_freq

def _my_hash(arg_list):
    # type: (List[Any]) -> int
    """Simple helper hash function"""
    res = 0
    for arg in arg_list:
        res = res * 31 + hash(arg)
    return res

def fmt_camel(name):
    """
    Converts name to lower camel case. Words are identified by capitalization,
    dashes, and underscores.
    """
    words = split_words(name)
    assert len(words) > 0
    first = words.pop(0).lower()
    return first + ''.join([word.capitalize() for word in words])

def is_integer(value: Any) -> bool:
    """Return true if a value is an integer number."""
    return (isinstance(value, int) and not isinstance(value, bool)) or (
        isinstance(value, float) and isfinite(value) and int(value) == value
    )

def _gauss(mean: int, sigma: int) -> int:
        """
        Creates a variation from a base value

        Args:
            mean: base value
            sigma: gaussian sigma

        Returns: random value

        """
        return int(random.gauss(mean, sigma))

def zfill(x, width):
    """zfill(x, width) -> string

    Pad a numeric string x with zeros on the left, to fill a field
    of the specified width.  The string x is never truncated.

    """
    if not isinstance(x, basestring):
        x = repr(x)
    return x.zfill(width)

def method_caller(method_name, *args, **kwargs):
	"""
	Return a function that will call a named method on the
	target object with optional positional and keyword
	arguments.

	>>> lower = method_caller('lower')
	>>> lower('MyString')
	'mystring'
	"""
	def call_method(target):
		func = getattr(target, method_name)
		return func(*args, **kwargs)
	return call_method

def obj_in_list_always(target_list, obj):
    """
    >>> l = [1,1,1]
    >>> obj_in_list_always(l, 1)
    True
    >>> l.append(2)
    >>> obj_in_list_always(l, 1)
    False
    """
    for item in set(target_list):
        if item is not obj:
            return False
    return True

def count(args):
    """ count occurences in a list of lists
    >>> count([['a','b'],['a']])
    defaultdict(int, {'a' : 2, 'b' : 1})
    """
    counts = defaultdict(int)
    for arg in args:
        for item in arg:
            counts[item] = counts[item] + 1
    return counts

def read32(bytestream):
  """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def _(f, x):
    """
    filter for dict, note `f` should have signature: `f::key->value->bool`
    """
    return {k: v for k, v in x.items() if f(k, v)}

def rank(tensor: BKTensor) -> int:
    """Return the number of dimensions of a tensor"""
    if isinstance(tensor, np.ndarray):
        return len(tensor.shape)

    return len(tensor[0].size())

def cmd_dot(conf: Config):
    """Print out a neat targets dependency tree based on requested targets.

    Use graphviz to render the dot file, e.g.:

    > ybt dot :foo :bar | dot -Tpng -o graph.png
    """
    build_context = BuildContext(conf)
    populate_targets_graph(build_context, conf)
    if conf.output_dot_file is None:
        write_dot(build_context, conf, sys.stdout)
    else:
        with open(conf.output_dot_file, 'w') as out_file:
            write_dot(build_context, conf, out_file)

def set_cell_value(cell, value):
    """
    Convenience method for setting the value of an openpyxl cell

    This is necessary since the value property changed from internal_value
    to value between version 1.* and 2.*.
    """
    if OPENPYXL_MAJOR_VERSION > 1:
        cell.value = value
    else:
        cell.internal_value = value

def valid_file(path: str) -> bool:
    """
    Verifies that a string path actually exists and is a file

    :param path: The path to verify
    :return: **True** if path exist and is a file
    """
    path = Path(path).expanduser()
    log.debug("checking if %s is a valid file", path)
    return path.exists() and path.is_file()

def mmap(func, iterable):
    """Wrapper to make map() behave the same on Py2 and Py3."""

    if sys.version_info[0] > 2:
        return [i for i in map(func, iterable)]
    else:
        return map(func, iterable)

def tanimoto_set_similarity(x: Iterable[X], y: Iterable[X]) -> float:
    """Calculate the tanimoto set similarity."""
    a, b = set(x), set(y)
    union = a | b

    if not union:
        return 0.0

    return len(a & b) / len(union)

def flatten_list(l: List[list]) -> list:
    """ takes a list of lists, l and returns a flat list
    """
    return [v for inner_l in l for v in inner_l]

def argmax(self, rows: List[Row], column: ComparableColumn) -> List[Row]:
        """
        Takes a list of rows and a column name and returns a list containing a single row (dict from
        columns to cells) that has the maximum numerical value in the given column. We return a list
        instead of a single dict to be consistent with the return type of ``select`` and
        ``all_rows``.
        """
        if not rows:
            return []
        value_row_pairs = [(row.values[column.name], row) for row in rows]
        if not value_row_pairs:
            return []
        # Returns a list containing the row with the max cell value.
        return [sorted(value_row_pairs, key=lambda x: x[0], reverse=True)[0][1]]

def cli_run():
    """docstring for argparse"""
    parser = argparse.ArgumentParser(description='Stupidly simple code answers from StackOverflow')
    parser.add_argument('query', help="What's the problem ?", type=str, nargs='+')
    parser.add_argument('-t','--tags', help='semicolon separated tags -> python;lambda')
    args = parser.parse_args()
    main(args)

def __remove_method(m: lmap.Map, key: T) -> lmap.Map:
        """Swap the methods atom to remove method with key."""
        return m.dissoc(key)

def _gauss(mean: int, sigma: int) -> int:
        """
        Creates a variation from a base value

        Args:
            mean: base value
            sigma: gaussian sigma

        Returns: random value

        """
        return int(random.gauss(mean, sigma))

def returned(n):
	"""Generate a random walk and return True if the walker has returned to
	the origin after taking `n` steps.
	"""
	## `takei` yield lazily so we can short-circuit and avoid computing the rest of the walk
	for pos in randwalk() >> drop(1) >> takei(xrange(n-1)):
		if pos == Origin:
			return True
	return False

def looks_like_url(url):
    """ Simplified check to see if the text appears to be a URL.

    Similar to `urlparse` but much more basic.

    Returns:
      True if the url str appears to be valid.
      False otherwise.

    >>> url = looks_like_url("totalgood.org")
    >>> bool(url)
    True
    """
    if not isinstance(url, basestring):
        return False
    if not isinstance(url, basestring) or len(url) >= 1024 or not cre_url.match(url):
        return False
    return True

def clean_column_names(df: DataFrame) -> DataFrame:
    """
    Strip the whitespace from all column names in the given DataFrame
    and return the result.
    """
    f = df.copy()
    f.columns = [col.strip() for col in f.columns]
    return f

def get_domain(url):
    """
    Get domain part of an url.

    For example: https://www.python.org/doc/ -> https://www.python.org
    """
    parse_result = urlparse(url)
    domain = "{schema}://{netloc}".format(
        schema=parse_result.scheme, netloc=parse_result.netloc)
    return domain

def preconnect(self, size=-1):
        """(pre)Connects some or all redis clients inside the pool.

        Args:
            size (int): number of redis clients to build and to connect
                (-1 means all clients if pool max_size > -1)

        Raises:
            ClientError: when size == -1 and pool max_size == -1
        """
        if size == -1 and self.max_size == -1:
            raise ClientError("size=-1 not allowed with pool max_size=-1")
        limit = min(size, self.max_size) if size != -1 else self.max_size
        clients = yield [self.get_connected_client() for _ in range(0, limit)]
        for client in clients:
            self.release_client(client)

def min(self):
        """
        :returns the minimum of the column
        """
        res = self._qexec("min(%s)" % self._name)
        if len(res) > 0:
            self._min = res[0][0]
        return self._min

def left_zero_pad(s, blocksize):
    """
    Left padding with zero bytes to a given block size

    :param s:
    :param blocksize:
    :return:
    """
    if blocksize > 0 and len(s) % blocksize:
        s = (blocksize - len(s) % blocksize) * b('\000') + s
    return s

def dotproduct(X, Y):
    """Return the sum of the element-wise product of vectors x and y.
    >>> dotproduct([1, 2, 3], [1000, 100, 10])
    1230
    """
    return sum([x * y for x, y in zip(X, Y)])

def s3_get(url: str, temp_file: IO) -> None:
    """Pull a file directly from S3."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)

def year(date):
    """ Returns the year.

    :param date:
        The string date with this format %m/%d/%Y
    :type date:
        String

    :returns:
        int

    :example:
        >>> year('05/1/2015')
        2015
    """
    try:
        fmt = '%m/%d/%Y'
        return datetime.strptime(date, fmt).timetuple().tm_year
    except ValueError:
        return 0

def try_cast_int(s):
    """(str) -> int
    All the digits in a given string are concatenated and converted into a single number.
    """
    try:
        temp = re.findall('\d', str(s))
        temp = ''.join(temp)
        return int(temp)
    except:
        return s

def str_upper(x):
    """Converts all strings in a column to uppercase.

    :returns: an expression containing the converted strings.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.


    >>> df.text.str.upper()
    Expression = str_upper(text)
    Length: 5 dtype: str (expression)
    ---------------------------------
    0    SOMETHING
    1  VERY PRETTY
    2    IS COMING
    3          OUR
    4         WAY.

    """
    sl = _to_string_sequence(x).upper()
    return column.ColumnStringArrow(sl.bytes, sl.indices, sl.length, sl.offset, string_sequence=sl)

def rank(tensor: BKTensor) -> int:
    """Return the number of dimensions of a tensor"""
    if isinstance(tensor, np.ndarray):
        return len(tensor.shape)

    return len(tensor[0].size())

def argmax(self, rows: List[Row], column: ComparableColumn) -> List[Row]:
        """
        Takes a list of rows and a column name and returns a list containing a single row (dict from
        columns to cells) that has the maximum numerical value in the given column. We return a list
        instead of a single dict to be consistent with the return type of ``select`` and
        ``all_rows``.
        """
        if not rows:
            return []
        value_row_pairs = [(row.values[column.name], row) for row in rows]
        if not value_row_pairs:
            return []
        # Returns a list containing the row with the max cell value.
        return [sorted(value_row_pairs, key=lambda x: x[0], reverse=True)[0][1]]

def uuid(self, version: int = None) -> str:
        """Generate random UUID.

        :param version: UUID version.
        :return: UUID
        """
        bits = self.random.getrandbits(128)
        return str(uuid.UUID(int=bits, version=version))

def csv_to_numpy(string_like, dtype=None):  # type: (str) -> np.array
    """Convert a CSV object to a numpy array.

    Args:
        string_like (str): CSV string.
        dtype (dtype, optional):  Data type of the resulting array. If None, the dtypes will be determined by the
                                        contents of each column, individually. This argument can only be used to
                                        'upcast' the array.  For downcasting, use the .astype(t) method.
    Returns:
        (np.array): numpy array
    """
    stream = StringIO(string_like)
    return np.genfromtxt(stream, dtype=dtype, delimiter=',')

def setup_cache(app: Flask, cache_config) -> Optional[Cache]:
    """Setup the flask-cache on a flask app"""
    if cache_config and cache_config.get('CACHE_TYPE') != 'null':
        return Cache(app, config=cache_config)

    return None

def flatten_list(x: List[Any]) -> List[Any]:
    """
    Converts a list of lists into a flat list.
    
    Args:
        x: list of lists 

    Returns:
        flat list
        
    As per
    http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python

    """  # noqa
    return [item for sublist in x for item in sublist]

def fprint(expr, print_ascii=False):
    r"""This function chooses whether to use ascii characters to represent
    a symbolic expression in the notebook or to use sympy's pprint.

    >>> from sympy import cos
    >>> omega=Symbol("omega")
    >>> fprint(cos(omega),print_ascii=True)
    cos(omega)


    """
    if print_ascii:
        pprint(expr, use_unicode=False, num_columns=120)
    else:
        return expr

async def executemany(self, sql: str, parameters: Iterable[Iterable[Any]]) -> None:
        """Execute the given multiquery."""
        await self._execute(self._cursor.executemany, sql, parameters)

async def fetchall(self) -> Iterable[sqlite3.Row]:
        """Fetch all remaining rows."""
        return await self._execute(self._cursor.fetchall)

def proper_round(n):
    """
    rounds float to closest int
    :rtype: int
    :param n: float
    """
    return int(n) + (n / abs(n)) * int(abs(n - int(n)) >= 0.5) if n != 0 else 0

def rate_limited(max_per_hour: int, *args: Any) -> Callable[..., Any]:
    """Rate limit a function."""
    return util.rate_limited(max_per_hour, *args)

def is_none(string_, default='raise'):
    """
    Check if a string is equivalent to None.

    Parameters
    ----------
    string_ : str
    default : {'raise', False}
        Default behaviour if none of the "None" strings is detected.

    Returns
    -------
    is_none : bool

    Examples
    --------
    >>> is_none('2', default=False)
    False
    >>> is_none('undefined', default=False)
    True
    """
    none = ['none', 'undefined', 'unknown', 'null', '']
    if string_.lower() in none:
        return True
    elif not default:
        return False
    else:
        raise ValueError('The value \'{}\' cannot be mapped to none.'
                         .format(string_))

def samefile(a: str, b: str) -> bool:
    """Check if two pathes represent the same file."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normpath(a) == os.path.normpath(b)

async def login(
        username: str, password: str, brand: str,
        websession: ClientSession = None) -> API:
    """Log in to the API."""
    api = API(brand, websession)
    await api.authenticate(username, password)
    return api

def should_rollover(self, record: LogRecord) -> bool:
        """
        Determine if rollover should occur.

        record is not used, as we are just comparing times, but it is needed so
        the method signatures are the same
        """
        t = int(time.time())
        if t >= self.rollover_at:
            return True
        return False

def do_quit(self, _: argparse.Namespace) -> bool:
        """Exit this application"""
        self._should_quit = True
        return self._STOP_AND_EXIT

def fast_median(a):
    """Fast median operation for masked array using 50th-percentile
    """
    a = checkma(a)
    #return scoreatpercentile(a.compressed(), 50)
    if a.count() > 0:
        out = np.percentile(a.compressed(), 50)
    else:
        out = np.ma.masked
    return out

def strings_to_integers(strings: Iterable[str]) -> Iterable[int]:
    """
    Convert a list of strings to a list of integers.

    :param strings: a list of string
    :return: a list of converted integers

    .. doctest::

        >>> strings_to_integers(['1', '1.0', '-0.2'])
        [1, 1, 0]
    """
    return strings_to_(strings, lambda x: int(float(x)))

def is_relative_url(url):
    """ simple method to determine if a url is relative or absolute """
    if url.startswith("#"):
        return None
    if url.find("://") > 0 or url.startswith("//"):
        # either 'http(s)://...' or '//cdn...' and therefore absolute
        return False
    return True

def remove_blank_lines(string):
    """ Removes all blank lines in @string

        -> #str without blank lines
    """
    return "\n".join(line
                     for line in string.split("\n")
                     if len(line.strip()))

def _request(self, method: str, endpoint: str, params: dict = None, data: dict = None, headers: dict = None) -> dict:
        """HTTP request method of interface implementation."""

def after_epoch(self, **_) -> None:
        """Save/override the latest model after every epoch."""
        SaveEvery.save_model(model=self._model, name_suffix=self._OUTPUT_NAME, on_failure=self._on_save_failure)

def last(self):
        """Last time step available.

        Example:
            >>> sdat = StagyyData('path/to/run')
            >>> assert(sdat.steps.last is sdat.steps[-1])
        """
        if self._last is UNDETERMINED:
            # not necessarily the last one...
            self._last = self.sdat.tseries.index[-1]
        return self[self._last]

def prevPlot(self):
        """Moves the displayed plot to the previous one"""
        if self.stacker.currentIndex() > 0:
            self.stacker.setCurrentIndex(self.stacker.currentIndex()-1)

def decodebytes(input):
    """Decode base64 string to byte array."""
    py_version = sys.version_info[0]
    if py_version >= 3:
        return _decodebytes_py3(input)
    return _decodebytes_py2(input)

def check64bit(current_system="python"):
    """checks if you are on a 64 bit platform"""
    if current_system == "python":
        return sys.maxsize > 2147483647
    elif current_system == "os":
        import platform
        pm = platform.machine()
        if pm != ".." and pm.endswith('64'):  # recent Python (not Iron)
            return True
        else:
            if 'PROCESSOR_ARCHITEW6432' in os.environ:
                return True  # 32 bit program running on 64 bit Windows
            try:
                # 64 bit Windows 64 bit program
                return os.environ['PROCESSOR_ARCHITECTURE'].endswith('64')
            except IndexError:
                pass  # not Windows
            try:
                # this often works in Linux
                return '64' in platform.architecture()[0]
            except Exception:
                # is an older version of Python, assume also an older os@
                # (best we can guess)
                return False

def is_prime(n):
    """
    Check if n is a prime number
    """
    if n % 2 == 0 and n > 2:
        return False
    return all(n % i for i in range(3, int(math.sqrt(n)) + 1, 2))

def recClearTag(element):
    """Applies maspy.xml.clearTag() to the tag attribute of the "element" and
    recursively to all child elements.

    :param element: an :instance:`xml.etree.Element`
    """
    children = element.getchildren()
    if len(children) > 0:
        for child in children:
            recClearTag(child)
    element.tag = clearTag(element.tag)

def array2string(arr: numpy.ndarray) -> str:
        """Format numpy array as a string."""
        shape = str(arr.shape)[1:-1]
        if shape.endswith(","):
            shape = shape[:-1]
        return numpy.array2string(arr, threshold=11) + "%s[%s]" % (arr.dtype, shape)

def sortBy(self, keyfunc, ascending=True, numPartitions=None):
        """
        Sorts this RDD by the given keyfunc

        >>> tmp = [('a', 1), ('b', 2), ('1', 3), ('d', 4), ('2', 5)]
        >>> sc.parallelize(tmp).sortBy(lambda x: x[0]).collect()
        [('1', 3), ('2', 5), ('a', 1), ('b', 2), ('d', 4)]
        >>> sc.parallelize(tmp).sortBy(lambda x: x[1]).collect()
        [('a', 1), ('b', 2), ('1', 3), ('d', 4), ('2', 5)]
        """
        return self.keyBy(keyfunc).sortByKey(ascending, numPartitions).values()

def top(self, topn=10):
        """
        Get a list of the top ``topn`` features in this :class:`.Feature`\.

        Examples
        --------

        .. code-block:: python

        >>> myFeature = Feature([('the', 2), ('pine', 1), ('trapezoid', 5)])
        >>> myFeature.top(1)
        [('trapezoid', 5)]

        Parameters
        ----------
        topn : int

        Returns
        -------
        list
        """
        return [self[i] for i in argsort(list(zip(*self))[1])[::-1][:topn]]

def multiple_replace(string, replacements):
    # type: (str, Dict[str,str]) -> str
    """Simultaneously replace multiple strigns in a string

    Args:
        string (str): Input string
        replacements (Dict[str,str]): Replacements dictionary

    Returns:
        str: String with replacements

    """
    pattern = re.compile("|".join([re.escape(k) for k in sorted(replacements, key=len, reverse=True)]), flags=re.DOTALL)
    return pattern.sub(lambda x: replacements[x.group(0)], string)

def fast_median(a):
    """Fast median operation for masked array using 50th-percentile
    """
    a = checkma(a)
    #return scoreatpercentile(a.compressed(), 50)
    if a.count() > 0:
        out = np.percentile(a.compressed(), 50)
    else:
        out = np.ma.masked
    return out

def uniqued(iterable):
    """Return unique list of ``iterable`` items preserving order.

    >>> uniqued('spameggs')
    ['s', 'p', 'a', 'm', 'e', 'g']
    """
    seen = set()
    return [item for item in iterable if item not in seen and not seen.add(item)]

def _check_samples_nodups(fnames):
    """Ensure a set of input VCFs do not have duplicate samples.
    """
    counts = defaultdict(int)
    for f in fnames:
        for s in get_samples(f):
            counts[s] += 1
    duplicates = [s for s, c in counts.items() if c > 1]
    if duplicates:
        raise ValueError("Duplicate samples found in inputs %s: %s" % (duplicates, fnames))

def get_window_dim():
    """ gets the dimensions depending on python version and os"""
    version = sys.version_info

    if version >= (3, 3):
        return _size_36()
    if platform.system() == 'Windows':
        return _size_windows()
    return _size_27()

def __remove_trailing_zeros(self, collection):
        """Removes trailing zeroes from indexable collection of numbers"""
        index = len(collection) - 1
        while index >= 0 and collection[index] == 0:
            index -= 1

        return collection[:index + 1]

def decode_base64(data: str) -> bytes:
    """Decode base64, padding being optional.

    :param data: Base64 data as an ASCII byte string
    :returns: The decoded byte string.
    """
    missing_padding = len(data) % 4
    if missing_padding != 0:
        data += "=" * (4 - missing_padding)
    return base64.decodebytes(data.encode("utf-8"))

def get_last_day_of_month(t: datetime) -> int:
    """
    Returns day number of the last day of the month
    :param t: datetime
    :return: int
    """
    tn = t + timedelta(days=32)
    tn = datetime(year=tn.year, month=tn.month, day=1)
    tt = tn - timedelta(hours=1)
    return tt.day

def obj_in_list_always(target_list, obj):
    """
    >>> l = [1,1,1]
    >>> obj_in_list_always(l, 1)
    True
    >>> l.append(2)
    >>> obj_in_list_always(l, 1)
    False
    """
    for item in set(target_list):
        if item is not obj:
            return False
    return True

def is_sqlatype_string(coltype: Union[TypeEngine, VisitableType]) -> bool:
    """
    Is the SQLAlchemy column type a string type?
    """
    coltype = _coltype_to_typeengine(coltype)
    return isinstance(coltype, sqltypes.String)

def has_changed (filename):
    """Check if filename has changed since the last check. If this
    is the first check, assume the file is changed."""
    key = os.path.abspath(filename)
    mtime = get_mtime(key)
    if key not in _mtime_cache:
        _mtime_cache[key] = mtime
        return True
    return mtime > _mtime_cache[key]

def copy_without_prompts(self):
        """Copy text to clipboard without prompts"""
        text = self.get_selected_text()
        lines = text.split(os.linesep)
        for index, line in enumerate(lines):
            if line.startswith('>>> ') or line.startswith('... '):
                lines[index] = line[4:]
        text = os.linesep.join(lines)
        QApplication.clipboard().setText(text)

def is_sqlatype_numeric(coltype: Union[TypeEngine, VisitableType]) -> bool:
    """
    Is the SQLAlchemy column type one that inherits from :class:`Numeric`,
    such as :class:`Float`, :class:`Decimal`?
    """
    coltype = _coltype_to_typeengine(coltype)
    return isinstance(coltype, sqltypes.Numeric)

def warn_if_nans_exist(X):
    """Warn if nans exist in a numpy array."""
    null_count = count_rows_with_nans(X)
    total = len(X)
    percent = 100 * null_count / total

    if null_count > 0:
        warning_message = \
            'Warning! Found {} rows of {} ({:0.2f}%) with nan values. Only ' \
            'complete rows will be plotted.'.format(null_count, total, percent)
        warnings.warn(warning_message, DataWarning)

def check_lengths(*arrays):
    """
    tool to ensure input and output data have the same number of samples

    Parameters
    ----------
    *arrays : iterable of arrays to be checked

    Returns
    -------
    None
    """
    lengths = [len(array) for array in arrays]
    if len(np.unique(lengths)) > 1:
        raise ValueError('Inconsistent data lengths: {}'.format(lengths))

def last_location_of_minimum(x):
    """
    Returns the last location of the minimal value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    return 1.0 - np.argmin(x[::-1]) / len(x) if len(x) > 0 else np.NaN

def csv_to_numpy(string_like, dtype=None):  # type: (str) -> np.array
    """Convert a CSV object to a numpy array.

    Args:
        string_like (str): CSV string.
        dtype (dtype, optional):  Data type of the resulting array. If None, the dtypes will be determined by the
                                        contents of each column, individually. This argument can only be used to
                                        'upcast' the array.  For downcasting, use the .astype(t) method.
    Returns:
        (np.array): numpy array
    """
    stream = StringIO(string_like)
    return np.genfromtxt(stream, dtype=dtype, delimiter=',')

def dag_longest_path(graph, source, target):
    """
    Finds the longest path in a dag between two nodes
    """
    if source == target:
        return [source]
    allpaths = nx.all_simple_paths(graph, source, target)
    longest_path = []
    for l in allpaths:
        if len(l) > len(longest_path):
            longest_path = l
    return longest_path

def _duplicates(list_):
    """Return dict mapping item -> indices."""
    item_indices = {}
    for i, item in enumerate(list_):
        try:
            item_indices[item].append(i)
        except KeyError:  # First time seen
            item_indices[item] = [i]
    return item_indices

def stretch(iterable, n=2):
    r"""Repeat each item in `iterable` `n` times.

    Example:

    >>> list(stretch(range(3), 2))
    [0, 0, 1, 1, 2, 2]
    """
    times = range(n)
    for item in iterable:
        for i in times: yield item

def recClearTag(element):
    """Applies maspy.xml.clearTag() to the tag attribute of the "element" and
    recursively to all child elements.

    :param element: an :instance:`xml.etree.Element`
    """
    children = element.getchildren()
    if len(children) > 0:
        for child in children:
            recClearTag(child)
    element.tag = clearTag(element.tag)

def flatten_list(l: List[list]) -> list:
    """ takes a list of lists, l and returns a flat list
    """
    return [v for inner_l in l for v in inner_l]

def is_any_type_set(sett: Set[Type]) -> bool:
    """
    Helper method to check if a set of types is the {AnyObject} singleton

    :param sett:
    :return:
    """
    return len(sett) == 1 and is_any_type(min(sett))

def long_substr(data):
    """Return the longest common substring in a list of strings.
    
    Credit: http://stackoverflow.com/questions/2892931/longest-common-substring-from-more-than-two-strings-python
    """
    substr = ''
    if len(data) > 1 and len(data[0]) > 0:
        for i in range(len(data[0])):
            for j in range(len(data[0])-i+1):
                if j > len(substr) and all(data[0][i:i+j] in x for x in data):
                    substr = data[0][i:i+j]
    elif len(data) == 1:
        substr = data[0]
    return substr

def isfinite(data: mx.nd.NDArray) -> mx.nd.NDArray:
    """Performs an element-wise check to determine if the NDArray contains an infinite element or not.
       TODO: remove this funciton after upgrade to MXNet 1.4.* in favor of mx.ndarray.contrib.isfinite()
    """
    is_data_not_nan = data == data
    is_data_not_infinite = data.abs() != np.inf
    return mx.nd.logical_and(is_data_not_infinite, is_data_not_nan)

def strtobytes(input, encoding):
    """Take a str and transform it into a byte array."""
    py_version = sys.version_info[0]
    if py_version >= 3:
        return _strtobytes_py3(input, encoding)
    return _strtobytes_py2(input, encoding)

def set_range(self, min_val, max_val):
        """Set the range of the colormap to [*min_val*, *max_val*]
        """
        if min_val > max_val:
            max_val, min_val = min_val, max_val
        self.values = (((self.values * 1.0 - self.values.min()) /
                        (self.values.max() - self.values.min()))
                       * (max_val - min_val) + min_val)

def closest_values(L):
    """Closest values

    :param L: list of values
    :returns: two values from L with minimal distance
    :modifies: the order of L
    :complexity: O(n log n), for n=len(L)
    """
    assert len(L) >= 2
    L.sort()
    valmin, argmin = min((L[i] - L[i - 1], i) for i in range(1, len(L)))
    return L[argmin - 1], L[argmin]

def execute_sql(self, query):
        """
        Executes a given query string on an open postgres database.

        """
        c = self.con.cursor()
        c.execute(query)
        result = []
        if c.rowcount > 0:
            try:
                result = c.fetchall()
            except psycopg2.ProgrammingError:
                pass
        return result

def _parse_tuple_string(argument):
        """ Return a tuple from parsing 'a,b,c,d' -> (a,b,c,d) """
        if isinstance(argument, str):
            return tuple(int(p.strip()) for p in argument.split(','))
        return argument

def is_any_type_set(sett: Set[Type]) -> bool:
    """
    Helper method to check if a set of types is the {AnyObject} singleton

    :param sett:
    :return:
    """
    return len(sett) == 1 and is_any_type(min(sett))

def flatten_list(x: List[Any]) -> List[Any]:
    """
    Converts a list of lists into a flat list.
    
    Args:
        x: list of lists 

    Returns:
        flat list
        
    As per
    http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python

    """  # noqa
    return [item for sublist in x for item in sublist]

def valid_date(x: str) -> bool:
    """
    Retrun ``True`` if ``x`` is a valid YYYYMMDD date;
    otherwise return ``False``.
    """
    try:
        if x != dt.datetime.strptime(x, DATE_FORMAT).strftime(DATE_FORMAT):
            raise ValueError
        return True
    except ValueError:
        return False

def position(self) -> Position:
        """The current position of the cursor."""
        return Position(self._index, self._lineno, self._col_offset)

def _check_whitespace(string):
    """
    Make sure thre is no whitespace in the given string. Will raise a
    ValueError if whitespace is detected
    """
    if string.count(' ') + string.count('\t') + string.count('\n') > 0:
        raise ValueError(INSTRUCTION_HAS_WHITESPACE)

def str_upper(x):
    """Converts all strings in a column to uppercase.

    :returns: an expression containing the converted strings.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.


    >>> df.text.str.upper()
    Expression = str_upper(text)
    Length: 5 dtype: str (expression)
    ---------------------------------
    0    SOMETHING
    1  VERY PRETTY
    2    IS COMING
    3          OUR
    4         WAY.

    """
    sl = _to_string_sequence(x).upper()
    return column.ColumnStringArrow(sl.bytes, sl.indices, sl.length, sl.offset, string_sequence=sl)

def _isint(string):
    """
    >>> _isint("123")
    True
    >>> _isint("123.45")
    False
    """
    return type(string) is int or \
           (isinstance(string, _binary_type) or isinstance(string, _text_type)) and \
           _isconvertible(int, string)

def non_increasing(values):
    """True if values are not increasing."""
    return all(x >= y for x, y in zip(values, values[1:]))

def last_location_of_minimum(x):
    """
    Returns the last location of the minimal value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    return 1.0 - np.argmin(x[::-1]) / len(x) if len(x) > 0 else np.NaN

def read(self, start_position: int, size: int) -> memoryview:
        """
        Return a view into the memory
        """
        return memoryview(self._bytes)[start_position:start_position + size]

def _prm_get_longest_stringsize(string_list):
        """ Returns the longest string size for a string entry across data."""
        maxlength = 1

        for stringar in string_list:
            if isinstance(stringar, np.ndarray):
                if stringar.ndim > 0:
                    for string in stringar.ravel():
                        maxlength = max(len(string), maxlength)
                else:
                    maxlength = max(len(stringar.tolist()), maxlength)
            else:
                maxlength = max(len(stringar), maxlength)

        # Make the string Col longer than needed in order to allow later on slightly larger strings
        return int(maxlength * 1.5)

async def stdout(self) -> AsyncGenerator[str, None]:
        """Asynchronous generator for lines from subprocess stdout."""
        await self.wait_running()
        async for line in self._subprocess.stdout:  # type: ignore
            yield line

def same_network(atree, btree) -> bool:
    """True if given trees share the same structure of powernodes,
    independently of (power)node names,
    and same edge topology between (power)nodes.

    """
    return same_hierarchy(atree, btree) and same_topology(atree, btree)

def is_strict_numeric(n: Node) -> bool:
    """ numeric denotes typed literals with datatypes xsd:integer, xsd:decimal, xsd:float, and xsd:double. """
    return is_typed_literal(n) and cast(Literal, n).datatype in [XSD.integer, XSD.decimal, XSD.float, XSD.double]

def dtypes(self):
        """Returns all column names and their data types as a list.

        >>> df.dtypes
        [('age', 'int'), ('name', 'string')]
        """
        return [(str(f.name), f.dataType.simpleString()) for f in self.schema.fields]

def bytes_hack(buf):
    """
    Hacky workaround for old installs of the library on systems without python-future that were
    keeping the 2to3 update from working after auto-update.
    """
    ub = None
    if sys.version_info > (3,):
        ub = buf
    else:
        ub = bytes(buf)

    return ub

def camel_to_snake(s: str) -> str:
    """Convert string from camel case to snake case."""

    return CAMEL_CASE_RE.sub(r'_\1', s).strip().lower()

def genfirstvalues(cursor: Cursor, arraysize: int = 1000) \
        -> Generator[Any, None, None]:
    """
    Generate the first value in each row.

    Args:
        cursor: the cursor
        arraysize: split fetches into chunks of this many records

    Yields:
        the first value of each row
    """
    return (row[0] for row in genrows(cursor, arraysize))

def get_language():
    """
    Wrapper around Django's `get_language` utility.
    For Django >= 1.8, `get_language` returns None in case no translation is activate.
    Here we patch this behavior e.g. for back-end functionality requiring access to translated fields
    """
    from parler import appsettings
    language = dj_get_language()
    if language is None and appsettings.PARLER_DEFAULT_ACTIVATE:
        return appsettings.PARLER_DEFAULT_LANGUAGE_CODE
    else:
        return language

def valid_file(path: str) -> bool:
    """
    Verifies that a string path actually exists and is a file

    :param path: The path to verify
    :return: **True** if path exist and is a file
    """
    path = Path(path).expanduser()
    log.debug("checking if %s is a valid file", path)
    return path.exists() and path.is_file()

def flatten_multidict(multidict):
    """Return flattened dictionary from ``MultiDict``."""
    return dict([(key, value if len(value) > 1 else value[0])
                 for (key, value) in multidict.iterlists()])

def clean(ctx, text):
    """
    Removes all non-printable characters from a text string
    """
    text = conversions.to_string(text, ctx)
    return ''.join([c for c in text if ord(c) >= 32])

def most_frequent(lst):
    """
    Returns the item that appears most frequently in the given list.
    """
    lst = lst[:]
    highest_freq = 0
    most_freq = None

    for val in unique(lst):
        if lst.count(val) > highest_freq:
            most_freq = val
            highest_freq = lst.count(val)
            
    return most_freq

def b64_decode(data: bytes) -> bytes:
    """
    :param data: Base 64 encoded data to decode.
    :type data: bytes
    :return: Base 64 decoded data.
    :rtype: bytes
    """
    missing_padding = len(data) % 4
    if missing_padding != 0:
        data += b'=' * (4 - missing_padding)
    return urlsafe_b64decode(data)

def flatten_list(x: List[Any]) -> List[Any]:
    """
    Converts a list of lists into a flat list.
    
    Args:
        x: list of lists 

    Returns:
        flat list
        
    As per
    http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python

    """  # noqa
    return [item for sublist in x for item in sublist]

def str_to_time(time_str: str) -> datetime.datetime:
    """
    Convert human readable string to datetime.datetime.
    """
    pieces: Any = [int(piece) for piece in time_str.split('-')]
    return datetime.datetime(*pieces)

def block_diag(*blocks: np.ndarray) -> np.ndarray:
    """Concatenates blocks into a block diagonal matrix.

    Args:
        *blocks: Square matrices to place along the diagonal of the result.

    Returns:
        A block diagonal matrix with the given blocks along its diagonal.

    Raises:
        ValueError: A block isn't square.
    """
    for b in blocks:
        if b.shape[0] != b.shape[1]:
            raise ValueError('Blocks must be square.')

    if not blocks:
        return np.zeros((0, 0), dtype=np.complex128)

    n = sum(b.shape[0] for b in blocks)
    dtype = functools.reduce(_merge_dtypes, (b.dtype for b in blocks))

    result = np.zeros(shape=(n, n), dtype=dtype)
    i = 0
    for b in blocks:
        j = i + b.shape[0]
        result[i:j, i:j] = b
        i = j

    return result

async def fetchall(self) -> Iterable[sqlite3.Row]:
        """Fetch all remaining rows."""
        return await self._execute(self._cursor.fetchall)

def astensor(array: TensorLike) -> BKTensor:
    """Covert numpy array to tensorflow tensor"""
    tensor = tf.convert_to_tensor(value=array, dtype=CTYPE)
    return tensor

def split(text: str) -> List[str]:
    """Split a text into a list of tokens.

    :param text: the text to split
    :return: tokens
    """
    return [word for word in SEPARATOR.split(text) if word.strip(' \t')]

def __replace_all(repls: dict, input: str) -> str:
    """ Replaces from a string **input** all the occurrences of some
    symbols according to mapping **repls**.

    :param dict repls: where #key is the old character and
    #value is the one to substitute with;
    :param str input: original string where to apply the
    replacements;
    :return: *(str)* the string with the desired characters replaced
    """
    return re.sub('|'.join(re.escape(key) for key in repls.keys()),
                  lambda k: repls[k.group(0)], input)

def iterate_items(dictish):
    """ Return a consistent (key, value) iterable on dict-like objects,
    including lists of tuple pairs.

    Example:

        >>> list(iterate_items({'a': 1}))
        [('a', 1)]
        >>> list(iterate_items([('a', 1), ('b', 2)]))
        [('a', 1), ('b', 2)]
    """
    if hasattr(dictish, 'iteritems'):
        return dictish.iteritems()
    if hasattr(dictish, 'items'):
        return dictish.items()
    return dictish

def bytes_hack(buf):
    """
    Hacky workaround for old installs of the library on systems without python-future that were
    keeping the 2to3 update from working after auto-update.
    """
    ub = None
    if sys.version_info > (3,):
        ub = buf
    else:
        ub = bytes(buf)

    return ub

def remove_once(gset, elem):
    """Remove the element from a set, lists or dict.
    
        >>> L = ["Lucy"]; S = set(["Sky"]); D = { "Diamonds": True };
        >>> remove_once(L, "Lucy"); remove_once(S, "Sky"); remove_once(D, "Diamonds");
        >>> print L, S, D
        [] set([]) {}

    Returns the element if it was removed. Raises one of the exceptions in 
    :obj:`RemoveError` otherwise.
    """
    remove = getattr(gset, 'remove', None)
    if remove is not None: remove(elem)
    else: del gset[elem]
    return elem

def is_prime(n):
    """
    Check if n is a prime number
    """
    if n % 2 == 0 and n > 2:
        return False
    return all(n % i for i in range(3, int(math.sqrt(n)) + 1, 2))

def get_now_sql_datetime():
    """
    *A datetime stamp in MySQL format: ``YYYY-MM-DDTHH:MM:SS``*

    **Return:**
        - ``now`` -- current time and date in MySQL format

    **Usage:**
        .. code-block:: python 

            from fundamentals import times
            now = times.get_now_sql_datetime()
            print now

            # OUT: 2016-03-18T11:08:23 
    """
    ## > IMPORTS ##
    from datetime import datetime, date, time
    now = datetime.now()
    now = now.strftime("%Y-%m-%dT%H:%M:%S")

    return now

def chars(string: any) -> str:
        """Return all (and only) the chars in the given string."""
        return ''.join([c if c.isalpha() else '' for c in str(string)])

def strings_to_integers(strings: Iterable[str]) -> Iterable[int]:
    """
    Convert a list of strings to a list of integers.

    :param strings: a list of string
    :return: a list of converted integers

    .. doctest::

        >>> strings_to_integers(['1', '1.0', '-0.2'])
        [1, 1, 0]
    """
    return strings_to_(strings, lambda x: int(float(x)))

def _isint(string):
    """
    >>> _isint("123")
    True
    >>> _isint("123.45")
    False
    """
    return type(string) is int or \
           (isinstance(string, _binary_type) or isinstance(string, _text_type)) and \
           _isconvertible(int, string)

def lint(fmt='colorized'):
    """Run verbose PyLint on source. Optionally specify fmt=html for HTML output."""
    if fmt == 'html':
        outfile = 'pylint_report.html'
        local('pylint -f %s davies > %s || true' % (fmt, outfile))
        local('open %s' % outfile)
    else:
        local('pylint -f %s davies || true' % fmt)

def count(args):
    """ count occurences in a list of lists
    >>> count([['a','b'],['a']])
    defaultdict(int, {'a' : 2, 'b' : 1})
    """
    counts = defaultdict(int)
    for arg in args:
        for item in arg:
            counts[item] = counts[item] + 1
    return counts

def get_column_names(engine: Engine, tablename: str) -> List[str]:
    """
    Get all the database column names for the specified table.
    """
    return [info.name for info in gen_columns_info(engine, tablename)]

def argmax(self, rows: List[Row], column: ComparableColumn) -> List[Row]:
        """
        Takes a list of rows and a column name and returns a list containing a single row (dict from
        columns to cells) that has the maximum numerical value in the given column. We return a list
        instead of a single dict to be consistent with the return type of ``select`` and
        ``all_rows``.
        """
        if not rows:
            return []
        value_row_pairs = [(row.values[column.name], row) for row in rows]
        if not value_row_pairs:
            return []
        # Returns a list containing the row with the max cell value.
        return [sorted(value_row_pairs, key=lambda x: x[0], reverse=True)[0][1]]

def sample_normal(mean, var, rng):
    """Sample from independent normal distributions

    Each element is an independent normal distribution.

    Parameters
    ----------
    mean : numpy.ndarray
      Means of the normal distribution. Shape --> (batch_num, sample_dim)
    var : numpy.ndarray
      Variance of the normal distribution. Shape --> (batch_num, sample_dim)
    rng : numpy.random.RandomState

    Returns
    -------
    ret : numpy.ndarray
       The sampling result. Shape --> (batch_num, sample_dim)
    """
    ret = numpy.sqrt(var) * rng.randn(*mean.shape) + mean
    return ret

def mmap(func, iterable):
    """Wrapper to make map() behave the same on Py2 and Py3."""

    if sys.version_info[0] > 2:
        return [i for i in map(func, iterable)]
    else:
        return map(func, iterable)

def count(self, elem):
        """
        Return the number of elements equal to elem present in the queue

        >>> pdeque([1, 2, 1]).count(1)
        2
        """
        return self._left_list.count(elem) + self._right_list.count(elem)

def dictlist_wipe_key(dict_list: Iterable[Dict], key: str) -> None:
    """
    Process an iterable of dictionaries. For each dictionary ``d``, delete
    ``d[key]`` if it exists.
    """
    for d in dict_list:
        d.pop(key, None)

def remove_blank_lines(string):
    """ Removes all blank lines in @string

        -> #str without blank lines
    """
    return "\n".join(line
                     for line in string.split("\n")
                     if len(line.strip()))

def find_first_in_list(txt: str, str_list: [str]) -> int:  # type: ignore
    """
    Returns the index of the earliest occurence of an item from a list in a string

    Ex: find_first_in_list('foobar', ['bar', 'fin']) -> 3
    """
    start = len(txt) + 1
    for item in str_list:
        if start > txt.find(item) > -1:
            start = txt.find(item)
    return start if len(txt) + 1 > start > -1 else -1

def dictlist_wipe_key(dict_list: Iterable[Dict], key: str) -> None:
    """
    Process an iterable of dictionaries. For each dictionary ``d``, delete
    ``d[key]`` if it exists.
    """
    for d in dict_list:
        d.pop(key, None)

def debugTreePrint(node,pfx="->"):
  """Purely a debugging aid: Ascii-art picture of a tree descended from node"""
  print pfx,node.item
  for c in node.children:
    debugTreePrint(c,"  "+pfx)

def _sum_cycles_from_tokens(self, tokens: List[str]) -> int:
        """Sum the total number of cycles over a list of tokens."""
        return sum((int(self._nonnumber_pattern.sub('', t)) for t in tokens))

def percentile(sorted_list, percent, key=lambda x: x):
    """Find the percentile of a sorted list of values.

    Arguments
    ---------
    sorted_list : list
        A sorted (ascending) list of values.
    percent : float
        A float value from 0.0 to 1.0.
    key : function, optional
        An optional function to compute a value from each element of N.

    Returns
    -------
    float
        The desired percentile of the value list.

    Examples
    --------
    >>> sorted_list = [4,6,8,9,11]
    >>> percentile(sorted_list, 0.4)
    7.0
    >>> percentile(sorted_list, 0.44)
    8.0
    >>> percentile(sorted_list, 0.6)
    8.5
    >>> percentile(sorted_list, 0.99)
    11.0
    >>> percentile(sorted_list, 1)
    11.0
    >>> percentile(sorted_list, 0)
    4.0
    """
    if not sorted_list:
        return None
    if percent == 1:
        return float(sorted_list[-1])
    if percent == 0:
        return float(sorted_list[0])
    n = len(sorted_list)
    i = percent * n
    if ceil(i) == i:
        i = int(i)
        return (sorted_list[i-1] + sorted_list[i]) / 2
    return float(sorted_list[ceil(i)-1])

def get_from_gnucash26_date(date_str: str) -> date:
    """ Creates a datetime from GnuCash 2.6 date string """
    date_format = "%Y%m%d"
    result = datetime.strptime(date_str, date_format).date()
    return result

def last_location_of_minimum(x):
    """
    Returns the last location of the minimal value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    return 1.0 - np.argmin(x[::-1]) / len(x) if len(x) > 0 else np.NaN

def dtypes(self):
        """Returns all column names and their data types as a list.

        >>> df.dtypes
        [('age', 'int'), ('name', 'string')]
        """
        return [(str(f.name), f.dataType.simpleString()) for f in self.schema.fields]

def fib(n):
    """Fibonacci example function

    Args:
      n (int): integer

    Returns:
      int: n-th Fibonacci number
    """
    assert n > 0
    a, b = 1, 1
    for i in range(n - 1):
        a, b = b, a + b
    return a

def most_frequent(lst):
    """
    Returns the item that appears most frequently in the given list.
    """
    lst = lst[:]
    highest_freq = 0
    most_freq = None

    for val in unique(lst):
        if lst.count(val) > highest_freq:
            most_freq = val
            highest_freq = lst.count(val)
            
    return most_freq

def valid_file(path: str) -> bool:
    """
    Verifies that a string path actually exists and is a file

    :param path: The path to verify
    :return: **True** if path exist and is a file
    """
    path = Path(path).expanduser()
    log.debug("checking if %s is a valid file", path)
    return path.exists() and path.is_file()

def to_bytes(data: Any) -> bytearray:
    """
    Convert anything to a ``bytearray``.
    
    See
    
    - http://stackoverflow.com/questions/7585435/best-way-to-convert-string-to-bytes-in-python-3
    - http://stackoverflow.com/questions/10459067/how-to-convert-my-bytearrayb-x9e-x18k-x9a-to-something-like-this-x9e-x1
    """  # noqa
    if isinstance(data, int):
        return bytearray([data])
    return bytearray(data, encoding='latin-1')

def s3_get(url: str, temp_file: IO) -> None:
    """Pull a file directly from S3."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)

def try_instance_init(self, instance, late_start=False):
        """Try to "initialize" the given module instance.

        :param instance: instance to init
        :type instance: object
        :param late_start: If late_start, don't look for last_init_try
        :type late_start: bool
        :return: True on successful init. False if instance init method raised any Exception.
        :rtype: bool
        """
        try:
            instance.init_try += 1
            # Maybe it's a retry
            if not late_start and instance.init_try > 1:
                # Do not try until too frequently, or it's too loopy
                if instance.last_init_try > time.time() - MODULE_INIT_PERIOD:
                    logger.info("Too early to retry initialization, retry period is %d seconds",
                                MODULE_INIT_PERIOD)
                    # logger.info("%s / %s", instance.last_init_try, time.time())
                    return False
            instance.last_init_try = time.time()

            logger.info("Trying to initialize module: %s", instance.name)

            # If it's an external module, create/update Queues()
            if instance.is_external:
                instance.create_queues(self.daemon.sync_manager)

            # The module instance init function says if initialization is ok
            if not instance.init():
                logger.warning("Module %s initialisation failed.", instance.name)
                return False
            logger.info("Module %s is initialized.", instance.name)
        except Exception as exp:  # pylint: disable=broad-except
            # pragma: no cover, simple protection
            msg = "The module instance %s raised an exception " \
                  "on initialization: %s, I remove it!" % (instance.name, str(exp))
            self.configuration_errors.append(msg)
            logger.error(msg)
            logger.exception(exp)
            return False

        return True

def add_colons(s):
    """Add colons after every second digit.

    This function is used in functions to prettify serials.

    >>> add_colons('teststring')
    'te:st:st:ri:ng'
    """
    return ':'.join([s[i:i + 2] for i in range(0, len(s), 2)])

def has_jongsung(letter):
    """Check whether this letter contains Jongsung"""
    if len(letter) != 1:
        raise Exception('The target string must be one letter.')
    if not is_hangul(letter):
        raise NotHangulException('The target string must be Hangul')

    code = lt.hangul_index(letter)
    return code % NUM_JONG > 0

def _check_whitespace(string):
    """
    Make sure thre is no whitespace in the given string. Will raise a
    ValueError if whitespace is detected
    """
    if string.count(' ') + string.count('\t') + string.count('\n') > 0:
        raise ValueError(INSTRUCTION_HAS_WHITESPACE)

def s3_get(url: str, temp_file: IO) -> None:
    """Pull a file directly from S3."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)

def try_cast_int(s):
    """(str) -> int
    All the digits in a given string are concatenated and converted into a single number.
    """
    try:
        temp = re.findall('\d', str(s))
        temp = ''.join(temp)
        return int(temp)
    except:
        return s

def memory_full():
    """Check if the memory is too full for further caching."""
    current_process = psutil.Process(os.getpid())
    return (current_process.memory_percent() >
            config.MAXIMUM_CACHE_MEMORY_PERCENTAGE)

def uppercase_chars(string: any) -> str:
        """Return all (and only) the uppercase chars in the given string."""
        return ''.join([c if c.isupper() else '' for c in str(string)])

def zfill(x, width):
    """zfill(x, width) -> string

    Pad a numeric string x with zeros on the left, to fill a field
    of the specified width.  The string x is never truncated.

    """
    if not isinstance(x, basestring):
        x = repr(x)
    return x.zfill(width)

def _skip_section(self):
        """Skip a section"""
        self._last = self._f.readline()
        while len(self._last) > 0 and len(self._last[0].strip()) == 0:
            self._last = self._f.readline()

def normalize(numbers):
    """Multiply each number by a constant such that the sum is 1.0
    >>> normalize([1,2,1])
    [0.25, 0.5, 0.25]
    """
    total = float(sum(numbers))
    return [n / total for n in numbers]

def iso_string_to_python_datetime(
        isostring: str) -> Optional[datetime.datetime]:
    """
    Takes an ISO-8601 string and returns a ``datetime``.
    """
    if not isostring:
        return None  # if you parse() an empty string, you get today's date
    return dateutil.parser.parse(isostring)

def proper_round(n):
    """
    rounds float to closest int
    :rtype: int
    :param n: float
    """
    return int(n) + (n / abs(n)) * int(abs(n - int(n)) >= 0.5) if n != 0 else 0

def get_case_insensitive_dict_key(d: Dict, k: str) -> Optional[str]:
    """
    Within the dictionary ``d``, find a key that matches (in case-insensitive
    fashion) the key ``k``, and return it (or ``None`` if there isn't one).
    """
    for key in d.keys():
        if k.lower() == key.lower():
            return key
    return None

def find_index(segmentation, stroke_id):
    """
    >>> find_index([[0, 1, 2], [3, 4], [5, 6, 7]], 0)
    0
    >>> find_index([[0, 1, 2], [3, 4], [5, 6, 7]], 1)
    0
    >>> find_index([[0, 1, 2], [3, 4], [5, 6, 7]], 5)
    2
    >>> find_index([[0, 1, 2], [3, 4], [5, 6, 7]], 6)
    2
    """
    for i, symbol in enumerate(segmentation):
        for sid in symbol:
            if sid == stroke_id:
                return i
    return -1

def from_file(file_path) -> dict:
        """ Load JSON file """
        with io.open(file_path, 'r', encoding='utf-8') as json_stream:
            return Json.parse(json_stream, True)

def inverted_dict(d):
    """Return a dict with swapped keys and values

    >>> inverted_dict({0: ('a', 'b'), 1: 'cd'}) == {'cd': 1, ('a', 'b'): 0}
    True
    """
    return dict((force_hashable(v), k) for (k, v) in viewitems(dict(d)))

def most_significant_bit(lst: np.ndarray) -> int:
    """
    A helper function that finds the position of the most significant bit in a 1darray of 1s and 0s,
    i.e. the first position where a 1 appears, reading left to right.

    :param lst: a 1d array of 0s and 1s with at least one 1
    :return: the first position in lst that a 1 appears
    """
    return np.argwhere(np.asarray(lst) == 1)[0][0]

def pack_bits( longbits ):
    """Crunch a 64-bit int (8 bool bytes) into a bitfield."""
    byte = longbits & (0x0101010101010101)
    byte = (byte | (byte>>7)) & (0x0003000300030003)
    byte = (byte | (byte>>14)) & (0x0000000f0000000f)
    byte = (byte | (byte>>28)) & (0x00000000000000ff)
    return byte

def check_oneof(**kwargs):
    """Raise ValueError if more than one keyword argument is not none.

    Args:
        kwargs (dict): The keyword arguments sent to the function.

    Returns: None

    Raises:
        ValueError: If more than one entry in kwargs is not none.
    """
    # Sanity check: If no keyword arguments were sent, this is fine.
    if not kwargs:
        return None

    not_nones = [val for val in kwargs.values() if val is not None]
    if len(not_nones) > 1:
        raise ValueError('Only one of {fields} should be set.'.format(
            fields=', '.join(sorted(kwargs.keys())),
        ))

def grep(pattern, filename):
    """Very simple grep that returns the first matching line in a file.
    String matching only, does not do REs as currently implemented.
    """
    try:
        # for line in file
        # if line matches pattern:
        #    return line
        return next((L for L in open(filename) if L.find(pattern) >= 0))
    except StopIteration:
        return ''

def execute(cur, *args):
    """Utility function to print sqlite queries before executing.

    Use instead of cur.execute().  First argument is cursor.

    cur.execute(stmt)
    becomes
    util.execute(cur, stmt)
    """
    stmt = args[0]
    if len(args) > 1:
        stmt = stmt.replace('%', '%%').replace('?', '%r')
        print(stmt % (args[1]))
    return cur.execute(*args)

def butlast(iterable):
    """Yield all items from ``iterable`` except the last one.

    >>> list(butlast(['spam', 'eggs', 'ham']))
    ['spam', 'eggs']

    >>> list(butlast(['spam']))
    []

    >>> list(butlast([]))
    []
    """
    iterable = iter(iterable)
    try:
        first = next(iterable)
    except StopIteration:
        return
    for second in iterable:
        yield first
        first = second

def prin(*args, **kwargs):
    r"""Like ``print``, but a function. I.e. prints out all arguments as
    ``print`` would do. Specify output stream like this::

      print('ERROR', `out="sys.stderr"``).

    """
    print >> kwargs.get('out',None), " ".join([str(arg) for arg in args])

def camelize(key):
    """Convert a python_style_variable_name to lowerCamelCase.

    Examples
    --------
    >>> camelize('variable_name')
    'variableName'
    >>> camelize('variableName')
    'variableName'
    """
    return ''.join(x.capitalize() if i > 0 else x
                   for i, x in enumerate(key.split('_')))

def mmap(func, iterable):
    """Wrapper to make map() behave the same on Py2 and Py3."""

    if sys.version_info[0] > 2:
        return [i for i in map(func, iterable)]
    else:
        return map(func, iterable)

def __init__(self, enum_obj: Any) -> None:
        """Initialize attributes for informative output.

        :param enum_obj: Enum object.
        """
        if enum_obj:
            self.name = enum_obj
            self.items = ', '.join([str(i) for i in enum_obj])
        else:
            self.items = ''

def fast_median(a):
    """Fast median operation for masked array using 50th-percentile
    """
    a = checkma(a)
    #return scoreatpercentile(a.compressed(), 50)
    if a.count() > 0:
        out = np.percentile(a.compressed(), 50)
    else:
        out = np.ma.masked
    return out

def _groups_of_size(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks."""
    # _groups_of_size('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

def get_versions(reporev=True):
    """Get version information for components used by Spyder"""
    import sys
    import platform

    import qtpy
    import qtpy.QtCore

    revision = None
    if reporev:
        from spyder.utils import vcs
        revision, branch = vcs.get_git_revision(os.path.dirname(__dir__))

    if not sys.platform == 'darwin':  # To avoid a crash with our Mac app
        system = platform.system()
    else:
        system = 'Darwin'

    return {
        'spyder': __version__,
        'python': platform.python_version(),  # "2.7.3"
        'bitness': 64 if sys.maxsize > 2**32 else 32,
        'qt': qtpy.QtCore.__version__,
        'qt_api': qtpy.API_NAME,      # PyQt5
        'qt_api_ver': qtpy.PYQT_VERSION,
        'system': system,   # Linux, Windows, ...
        'release': platform.release(),  # XP, 10.6, 2.2.0, etc.
        'revision': revision,  # '9fdf926eccce'
    }

def long_substring(str_a, str_b):
    """
    Looks for a longest common string between any two given strings passed
    :param str_a: str
    :param str_b: str

    Big Thanks to Pulkit Kathuria(@kevincobain2000) for the function
    The function is derived from jProcessing toolkit suite
    """
    data = [str_a, str_b]
    substr = ''
    if len(data) > 1 and len(data[0]) > 0:
        for i in range(len(data[0])):
            for j in range(len(data[0])-i+1):
                if j > len(substr) and all(data[0][i:i+j] in x for x in data):
                    substr = data[0][i:i+j]
    return substr.strip()

def is_finite(value: Any) -> bool:
    """Return true if a value is a finite number."""
    return isinstance(value, int) or (isinstance(value, float) and isfinite(value))

def squash(self, a, b):
        """
        Returns a generator that squashes two iterables into one.

        ```
        ['this', 'that'], [[' and', ' or']] => ['this and', 'this or', 'that and', 'that or']
        ```
        """

        return ((''.join(x) if isinstance(x, tuple) else x) for x in itertools.product(a, b))

def SetCursorPos(x: int, y: int) -> bool:
    """
    SetCursorPos from Win32.
    Set mouse cursor to point x, y.
    x: int.
    y: int.
    Return bool, True if succeed otherwise False.
    """
    return bool(ctypes.windll.user32.SetCursorPos(x, y))

def uppercase_chars(string: any) -> str:
        """Return all (and only) the uppercase chars in the given string."""
        return ''.join([c if c.isupper() else '' for c in str(string)])

def normalize(numbers):
    """Multiply each number by a constant such that the sum is 1.0
    >>> normalize([1,2,1])
    [0.25, 0.5, 0.25]
    """
    total = float(sum(numbers))
    return [n / total for n in numbers]

def clean_map(obj: Mapping[Any, Any]) -> Mapping[Any, Any]:
    """
    Return a new copied dictionary without the keys with ``None`` values from
    the given Mapping object.
    """
    return {k: v for k, v in obj.items() if v is not None}

def is_unicode(string):
    """Validates that the object itself is some kinda string"""
    str_type = str(type(string))

    if str_type.find('str') > 0 or str_type.find('unicode') > 0:
        return True

    return False

def pack_bits( longbits ):
    """Crunch a 64-bit int (8 bool bytes) into a bitfield."""
    byte = longbits & (0x0101010101010101)
    byte = (byte | (byte>>7)) & (0x0003000300030003)
    byte = (byte | (byte>>14)) & (0x0000000f0000000f)
    byte = (byte | (byte>>28)) & (0x00000000000000ff)
    return byte

def wipe_table(self, table: str) -> int:
        """Delete all records from a table. Use caution!"""
        sql = "DELETE FROM " + self.delimit(table)
        return self.db_exec(sql)

def year(date):
    """ Returns the year.

    :param date:
        The string date with this format %m/%d/%Y
    :type date:
        String

    :returns:
        int

    :example:
        >>> year('05/1/2015')
        2015
    """
    try:
        fmt = '%m/%d/%Y'
        return datetime.strptime(date, fmt).timetuple().tm_year
    except ValueError:
        return 0

def _interface_exists(self, interface):
        """Check whether interface exists."""
        ios_cfg = self._get_running_config()
        parse = HTParser(ios_cfg)
        itfcs_raw = parse.find_lines("^interface " + interface)
        return len(itfcs_raw) > 0

def templategetter(tmpl):
    """
    This is a dirty little template function generator that turns single-brace
    Mustache-style template strings into functions that interpolate dict keys:

    >>> get_name = templategetter("{first} {last}")
    >>> get_name({'first': 'Shawn', 'last': 'Allen'})
    'Shawn Allen'
    """
    tmpl = tmpl.replace('{', '%(')
    tmpl = tmpl.replace('}', ')s')
    return lambda data: tmpl % data

def timeit(func, *args, **kwargs):
    """
    Time execution of function. Returns (res, seconds).

    >>> res, timing = timeit(time.sleep, 1)
    """
    start_time = time.time()
    res = func(*args, **kwargs)
    timing = time.time() - start_time
    return res, timing

def iprotate(l, steps=1):
    r"""Like rotate, but modifies `l` in-place.

    >>> l = [1,2,3]
    >>> iprotate(l) is l
    True
    >>> l
    [2, 3, 1]
    >>> iprotate(iprotate(l, 2), -3)
    [1, 2, 3]

    """
    if len(l):
        steps %= len(l)
        if steps:
            firstPart = l[:steps]
            del l[:steps]
            l.extend(firstPart)
    return l

def is_unicode(string):
    """Validates that the object itself is some kinda string"""
    str_type = str(type(string))

    if str_type.find('str') > 0 or str_type.find('unicode') > 0:
        return True

    return False

def post(self, endpoint: str, **kwargs) -> dict:
        """HTTP POST operation to API endpoint."""

        return self._request('POST', endpoint, **kwargs)

def get_window_dim():
    """ gets the dimensions depending on python version and os"""
    version = sys.version_info

    if version >= (3, 3):
        return _size_36()
    if platform.system() == 'Windows':
        return _size_windows()
    return _size_27()

def last_location_of_minimum(x):
    """
    Returns the last location of the minimal value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    return 1.0 - np.argmin(x[::-1]) / len(x) if len(x) > 0 else np.NaN

def gen_lower(x: Iterable[str]) -> Generator[str, None, None]:
    """
    Args:
        x: iterable of strings

    Yields:
        each string in lower case
    """
    for string in x:
        yield string.lower()

def calculate_fft(data, tbin):
    """
    Function to calculate the Fourier transform of data.
    
    
    Parameters
    ----------
    data : numpy.ndarray
        1D or 2D array containing time series.
    tbin : float
        Bin size of time series (in ms).
    
    
    Returns
    -------
    freqs : numpy.ndarray
        Frequency axis of signal in Fourier space.         
    fft : numpy.ndarray
        Signal in Fourier space.
        
    """
    if len(np.shape(data)) > 1:
        n = len(data[0])
        return np.fft.fftfreq(n, tbin * 1e-3), np.fft.fft(data, axis=1)
    else:
        n = len(data)
        return np.fft.fftfreq(n, tbin * 1e-3), np.fft.fft(data)

def get_keys_of_max_n(dict_obj, n):
    """Returns the keys that maps to the top n max values in the given dict.

    Example:
    --------
    >>> dict_obj = {'a':2, 'b':1, 'c':5}
    >>> get_keys_of_max_n(dict_obj, 2)
    ['a', 'c']
    """
    return sorted([
        item[0]
        for item in sorted(
            dict_obj.items(), key=lambda item: item[1], reverse=True
        )[:n]
    ])

def assign_parent(node: astroid.node_classes.NodeNG) -> astroid.node_classes.NodeNG:
    """return the higher parent which is not an AssignName, Tuple or List node
    """
    while node and isinstance(node, (astroid.AssignName, astroid.Tuple, astroid.List)):
        node = node.parent
    return node

def get_commits_modified_file(self, filepath: str) -> List[str]:
        """
        Given a filepath, returns all the commits that modified this file
        (following renames).

        :param str filepath: path to the file
        :return: the list of commits' hash
        """
        path = str(Path(filepath))

        commits = []
        try:
            commits = self.git.log("--follow", "--format=%H", path).split('\n')
        except GitCommandError:
            logger.debug("Could not find information of file %s", path)

        return commits

async def executemany(self, sql: str, parameters: Iterable[Iterable[Any]]) -> None:
        """Execute the given multiquery."""
        await self._execute(self._cursor.executemany, sql, parameters)

def __remove_trailing_zeros(self, collection):
        """Removes trailing zeroes from indexable collection of numbers"""
        index = len(collection) - 1
        while index >= 0 and collection[index] == 0:
            index -= 1

        return collection[:index + 1]

def get_day_name(self) -> str:
        """ Returns the day name """
        weekday = self.value.isoweekday() - 1
        return calendar.day_name[weekday]

def fib(n):
    """Fibonacci example function

    Args:
      n (int): integer

    Returns:
      int: n-th Fibonacci number
    """
    assert n > 0
    a, b = 1, 1
    for i in range(n - 1):
        a, b = b, a + b
    return a

def get_language():
    """
    Wrapper around Django's `get_language` utility.
    For Django >= 1.8, `get_language` returns None in case no translation is activate.
    Here we patch this behavior e.g. for back-end functionality requiring access to translated fields
    """
    from parler import appsettings
    language = dj_get_language()
    if language is None and appsettings.PARLER_DEFAULT_ACTIVATE:
        return appsettings.PARLER_DEFAULT_LANGUAGE_CODE
    else:
        return language

def _kbhit_unix() -> bool:
    """
    Under UNIX: is a keystroke available?
    """
    dr, dw, de = select.select([sys.stdin], [], [], 0)
    return dr != []

def __gt__(self, other):
        """Test for greater than."""
        if isinstance(other, Address):
            return str(self) > str(other)
        raise TypeError

def similarity(word1: str, word2: str) -> float:
    """
    Get cosine similarity between two words.
    If a word is not in the vocabulary, KeyError will be raised.

    :param string word1: first word
    :param string word2: second word
    :return: the cosine similarity between the two word vectors
    """
    return _MODEL.similarity(word1, word2)

def snake_case(a_string):
    """Returns a snake cased version of a string.

    :param a_string: any :class:`str` object.

    Usage:
        >>> snake_case('FooBar')
        "foo_bar"
    """

    partial = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', a_string)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', partial).lower()

def get_default_bucket_key(buckets: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Returns the default bucket from a list of buckets, i.e. the largest bucket.

    :param buckets: List of buckets.
    :return: The largest bucket in the list.
    """
    return max(buckets)

def __remove_method(m: lmap.Map, key: T) -> lmap.Map:
        """Swap the methods atom to remove method with key."""
        return m.dissoc(key)

def rate_limited(max_per_hour: int, *args: Any) -> Callable[..., Any]:
    """Rate limit a function."""
    return util.rate_limited(max_per_hour, *args)

def __gt__(self, other):
        """Test for greater than."""
        if isinstance(other, Address):
            return str(self) > str(other)
        raise TypeError

def file_or_stdin() -> Callable:
    """
    Returns a file descriptor from stdin or opening a file from a given path.
    """

    def parse(path):
        if path is None or path == "-":
            return sys.stdin
        else:
            return data_io.smart_open(path)

    return parse

def inverted_dict(d):
    """Return a dict with swapped keys and values

    >>> inverted_dict({0: ('a', 'b'), 1: 'cd'}) == {'cd': 1, ('a', 'b'): 0}
    True
    """
    return dict((force_hashable(v), k) for (k, v) in viewitems(dict(d)))

def _duplicates(list_):
    """Return dict mapping item -> indices."""
    item_indices = {}
    for i, item in enumerate(list_):
        try:
            item_indices[item].append(i)
        except KeyError:  # First time seen
            item_indices[item] = [i]
    return item_indices

def fetchvalue(self, sql: str, *args) -> Optional[Any]:
        """Executes SQL; returns the first value of the first row, or None."""
        row = self.fetchone(sql, *args)
        if row is None:
            return None
        return row[0]

def get_column_names(engine: Engine, tablename: str) -> List[str]:
    """
    Get all the database column names for the specified table.
    """
    return [info.name for info in gen_columns_info(engine, tablename)]

def most_significant_bit(lst: np.ndarray) -> int:
    """
    A helper function that finds the position of the most significant bit in a 1darray of 1s and 0s,
    i.e. the first position where a 1 appears, reading left to right.

    :param lst: a 1d array of 0s and 1s with at least one 1
    :return: the first position in lst that a 1 appears
    """
    return np.argwhere(np.asarray(lst) == 1)[0][0]

def dictlist_wipe_key(dict_list: Iterable[Dict], key: str) -> None:
    """
    Process an iterable of dictionaries. For each dictionary ``d``, delete
    ``d[key]`` if it exists.
    """
    for d in dict_list:
        d.pop(key, None)

def string(value) -> str:
        """ string dict/object/value to JSON """
        return system_json.dumps(Json(value).safe_object(), ensure_ascii=False)

def _rindex(mylist: Sequence[T], x: T) -> int:
    """Index of the last occurrence of x in the sequence."""
    return len(mylist) - mylist[::-1].index(x) - 1

def split(text: str) -> List[str]:
    """Split a text into a list of tokens.

    :param text: the text to split
    :return: tokens
    """
    return [word for word in SEPARATOR.split(text) if word.strip(' \t')]

def Exit(msg, code=1):
    """Exit execution with return code and message
    :param msg: Message displayed prior to exit
    :param code: code returned upon exiting
    """
    print >> sys.stderr, msg
    sys.exit(code)

def cli_run():
    """docstring for argparse"""
    parser = argparse.ArgumentParser(description='Stupidly simple code answers from StackOverflow')
    parser.add_argument('query', help="What's the problem ?", type=str, nargs='+')
    parser.add_argument('-t','--tags', help='semicolon separated tags -> python;lambda')
    args = parser.parse_args()
    main(args)

def argmax(self, rows: List[Row], column: ComparableColumn) -> List[Row]:
        """
        Takes a list of rows and a column name and returns a list containing a single row (dict from
        columns to cells) that has the maximum numerical value in the given column. We return a list
        instead of a single dict to be consistent with the return type of ``select`` and
        ``all_rows``.
        """
        if not rows:
            return []
        value_row_pairs = [(row.values[column.name], row) for row in rows]
        if not value_row_pairs:
            return []
        # Returns a list containing the row with the max cell value.
        return [sorted(value_row_pairs, key=lambda x: x[0], reverse=True)[0][1]]

def running_containers(name_filter: str) -> List[str]:
    """
    :raises docker.exceptions.APIError
    """
    return [container.short_id for container in
            docker_client.containers.list(filters={"name": name_filter})]

def remove_empty_text(utterances: List[Utterance]) -> List[Utterance]:
    """Remove empty utterances from a list of utterances
    Args:
        utterances: The list of utterance we are processing
    """
    return [utter for utter in utterances if utter.text.strip() != ""]

def _isint(string):
    """
    >>> _isint("123")
    True
    >>> _isint("123.45")
    False
    """
    return type(string) is int or \
           (isinstance(string, _binary_type) or isinstance(string, _text_type)) and \
           _isconvertible(int, string)

def bfx(value, msb, lsb):
    """! @brief Extract a value from a bitfield."""
    mask = bitmask((msb, lsb))
    return (value & mask) >> lsb

def indexes_equal(a: Index, b: Index) -> bool:
    """
    Are two indexes equal? Checks by comparing ``str()`` versions of them.
    (AM UNSURE IF THIS IS ENOUGH.)
    """
    return str(a) == str(b)

def __as_list(value: List[JsonObjTypes]) -> List[JsonTypes]:
        """ Return a json array as a list

        :param value: array
        :return: array with JsonObj instances removed
        """
        return [e._as_dict if isinstance(e, JsonObj) else e for e in value]

def cookies(self) -> Dict[str, str]:
        """The parsed cookies attached to this request."""
        cookies = SimpleCookie()
        cookies.load(self.headers.get('Cookie', ''))
        return {key: cookie.value for key, cookie in cookies.items()}

def count(args):
    """ count occurences in a list of lists
    >>> count([['a','b'],['a']])
    defaultdict(int, {'a' : 2, 'b' : 1})
    """
    counts = defaultdict(int)
    for arg in args:
        for item in arg:
            counts[item] = counts[item] + 1
    return counts

def to_bytes(data: Any) -> bytearray:
    """
    Convert anything to a ``bytearray``.
    
    See
    
    - http://stackoverflow.com/questions/7585435/best-way-to-convert-string-to-bytes-in-python-3
    - http://stackoverflow.com/questions/10459067/how-to-convert-my-bytearrayb-x9e-x18k-x9a-to-something-like-this-x9e-x1
    """  # noqa
    if isinstance(data, int):
        return bytearray([data])
    return bytearray(data, encoding='latin-1')

def zfill(x, width):
    """zfill(x, width) -> string

    Pad a numeric string x with zeros on the left, to fill a field
    of the specified width.  The string x is never truncated.

    """
    if not isinstance(x, basestring):
        x = repr(x)
    return x.zfill(width)

def position(self) -> Position:
        """The current position of the cursor."""
        return Position(self._index, self._lineno, self._col_offset)

def de_duplicate(items):
    """Remove any duplicate item, preserving order

    >>> de_duplicate([1, 2, 1, 2])
    [1, 2]
    """
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return result

def get_keys_of_max_n(dict_obj, n):
    """Returns the keys that maps to the top n max values in the given dict.

    Example:
    --------
    >>> dict_obj = {'a':2, 'b':1, 'c':5}
    >>> get_keys_of_max_n(dict_obj, 2)
    ['a', 'c']
    """
    return sorted([
        item[0]
        for item in sorted(
            dict_obj.items(), key=lambda item: item[1], reverse=True
        )[:n]
    ])

def get_last_day_of_month(t: datetime) -> int:
    """
    Returns day number of the last day of the month
    :param t: datetime
    :return: int
    """
    tn = t + timedelta(days=32)
    tn = datetime(year=tn.year, month=tn.month, day=1)
    tt = tn - timedelta(hours=1)
    return tt.day

def get_default_bucket_key(buckets: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Returns the default bucket from a list of buckets, i.e. the largest bucket.

    :param buckets: List of buckets.
    :return: The largest bucket in the list.
    """
    return max(buckets)

def docker_environment(env):
    """
    Transform dictionary of environment variables into Docker -e parameters.

    >>> result = docker_environment({'param1': 'val1', 'param2': 'val2'})
    >>> result in ['-e "param1=val1" -e "param2=val2"', '-e "param2=val2" -e "param1=val1"']
    True
    """
    return ' '.join(
        ["-e \"%s=%s\"" % (key, value.replace("$", "\\$").replace("\"", "\\\"").replace("`", "\\`"))
         for key, value in env.items()])

def call_api(self, resource_path, method,
                 path_params=None, query_params=None, header_params=None,
                 body=None, post_params=None, files=None,
                 response_type=None, auth_settings=None, asynchronous=None,
                 _return_http_data_only=None, collection_formats=None, _preload_content=True,
                 _request_timeout=None):
        """
        Makes the HTTP request (synchronous) and return the deserialized data.
        To make an async request, set the asynchronous parameter.

        :param resource_path: Path to method endpoint.
        :param method: Method to call.
        :param path_params: Path parameters in the url.
        :param query_params: Query parameters in the url.
        :param header_params: Header parameters to be
            placed in the request header.
        :param body: Request body.
        :param post_params dict: Request post form parameters,
            for `application/x-www-form-urlencoded`, `multipart/form-data`.
        :param auth_settings list: Auth Settings names for the request.
        :param response: Response data type.
        :param files dict: key -> filename, value -> filepath,
            for `multipart/form-data`.
        :param asynchronous bool: execute request asynchronously
        :param _return_http_data_only: response data without head status code and headers
        :param collection_formats: dict of collection formats for path, query,
            header, and post parameters.
        :param _preload_content: if False, the urllib3.HTTPResponse object will be returned without
                                 reading/decoding response data. Default is True.
        :param _request_timeout: timeout setting for this request. If one number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of (connection, read) timeouts.
        :return:
            If asynchronous parameter is True,
            the request will be called asynchronously.
            The method will return the request thread.
            If parameter asynchronous is False or missing,
            then the method will return the response directly.
        """
        if not asynchronous:
            return self.__call_api(resource_path, method,
                                   path_params, query_params, header_params,
                                   body, post_params, files,
                                   response_type, auth_settings,
                                   _return_http_data_only, collection_formats, _preload_content, _request_timeout)
        else:
            thread = self.pool.apply_async(self.__call_api, (resource_path, method,
                                           path_params, query_params,
                                           header_params, body,
                                           post_params, files,
                                           response_type, auth_settings,
                                           _return_http_data_only,
                                           collection_formats, _preload_content, _request_timeout))
        return thread

def snake_to_camel(value):
    """
    Converts a snake_case_string to a camelCaseString.

    >>> snake_to_camel("foo_bar_baz")
    'fooBarBaz'
    """
    camel = "".join(word.title() for word in value.split("_"))
    return value[:1].lower() + camel[1:]

def camelize(key):
    """Convert a python_style_variable_name to lowerCamelCase.

    Examples
    --------
    >>> camelize('variable_name')
    'variableName'
    >>> camelize('variableName')
    'variableName'
    """
    return ''.join(x.capitalize() if i > 0 else x
                   for i, x in enumerate(key.split('_')))

def is_natural(x):
    """A non-negative integer."""
    try:
        is_integer = int(x) == x
    except (TypeError, ValueError):
        return False
    return is_integer and x >= 0

def parsehttpdate(string_):
    """
    Parses an HTTP date into a datetime object.

        >>> parsehttpdate('Thu, 01 Jan 1970 01:01:01 GMT')
        datetime.datetime(1970, 1, 1, 1, 1, 1)
    """
    try:
        t = time.strptime(string_, "%a, %d %b %Y %H:%M:%S %Z")
    except ValueError:
        return None
    return datetime.datetime(*t[:6])

def count(self, elem):
        """
        Return the number of elements equal to elem present in the queue

        >>> pdeque([1, 2, 1]).count(1)
        2
        """
        return self._left_list.count(elem) + self._right_list.count(elem)

def check_key(self, key: str) -> bool:
        """
        Checks if key exists in datastore. True if yes, False if no.

        :param: SHA512 hash key

        :return: whether or key not exists in datastore
        """
        keys = self.get_keys()
        return key in keys

def find_duplicates(l: list) -> set:
    """
    Return the duplicates in a list.

    The function relies on
    https://stackoverflow.com/questions/9835762/find-and-list-duplicates-in-a-list .
    Parameters
    ----------
    l : list
        Name

    Returns
    -------
    set
        Duplicated values

    >>> find_duplicates([1,2,3])
    set()
    >>> find_duplicates([1,2,1])
    {1}
    """
    return set([x for x in l if l.count(x) > 1])

def is_not_null(df: DataFrame, col_name: str) -> bool:
    """
    Return ``True`` if the given DataFrame has a column of the given
    name (string), and there exists at least one non-NaN value in that
    column; return ``False`` otherwise.
    """
    if (
        isinstance(df, pd.DataFrame)
        and col_name in df.columns
        and df[col_name].notnull().any()
    ):
        return True
    else:
        return False

def pack_bits( longbits ):
    """Crunch a 64-bit int (8 bool bytes) into a bitfield."""
    byte = longbits & (0x0101010101010101)
    byte = (byte | (byte>>7)) & (0x0003000300030003)
    byte = (byte | (byte>>14)) & (0x0000000f0000000f)
    byte = (byte | (byte>>28)) & (0x00000000000000ff)
    return byte

def year(date):
    """ Returns the year.

    :param date:
        The string date with this format %m/%d/%Y
    :type date:
        String

    :returns:
        int

    :example:
        >>> year('05/1/2015')
        2015
    """
    try:
        fmt = '%m/%d/%Y'
        return datetime.strptime(date, fmt).timetuple().tm_year
    except ValueError:
        return 0

def argmax(self, rows: List[Row], column: ComparableColumn) -> List[Row]:
        """
        Takes a list of rows and a column name and returns a list containing a single row (dict from
        columns to cells) that has the maximum numerical value in the given column. We return a list
        instead of a single dict to be consistent with the return type of ``select`` and
        ``all_rows``.
        """
        if not rows:
            return []
        value_row_pairs = [(row.values[column.name], row) for row in rows]
        if not value_row_pairs:
            return []
        # Returns a list containing the row with the max cell value.
        return [sorted(value_row_pairs, key=lambda x: x[0], reverse=True)[0][1]]

def _kbhit_unix() -> bool:
    """
    Under UNIX: is a keystroke available?
    """
    dr, dw, de = select.select([sys.stdin], [], [], 0)
    return dr != []

def get_default_bucket_key(buckets: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Returns the default bucket from a list of buckets, i.e. the largest bucket.

    :param buckets: List of buckets.
    :return: The largest bucket in the list.
    """
    return max(buckets)

def translate_dict(cls, val):
        """Translate dicts to scala Maps"""
        escaped = ', '.join(
            ["{} -> {}".format(cls.translate_str(k), cls.translate(v)) for k, v in val.items()]
        )
        return 'Map({})'.format(escaped)

def iprotate(l, steps=1):
    r"""Like rotate, but modifies `l` in-place.

    >>> l = [1,2,3]
    >>> iprotate(l) is l
    True
    >>> l
    [2, 3, 1]
    >>> iprotate(iprotate(l, 2), -3)
    [1, 2, 3]

    """
    if len(l):
        steps %= len(l)
        if steps:
            firstPart = l[:steps]
            del l[:steps]
            l.extend(firstPart)
    return l

def read_set_from_file(filename: str) -> Set[str]:
    """
    Extract a de-duped collection (set) of text from a file.
    Expected file format is one item per line.
    """
    collection = set()
    with open(filename, 'r') as file_:
        for line in file_:
            collection.add(line.rstrip())
    return collection

def year(date):
    """ Returns the year.

    :param date:
        The string date with this format %m/%d/%Y
    :type date:
        String

    :returns:
        int

    :example:
        >>> year('05/1/2015')
        2015
    """
    try:
        fmt = '%m/%d/%Y'
        return datetime.strptime(date, fmt).timetuple().tm_year
    except ValueError:
        return 0

def _my_hash(arg_list):
    # type: (List[Any]) -> int
    """Simple helper hash function"""
    res = 0
    for arg in arg_list:
        res = res * 31 + hash(arg)
    return res

def datetime_iso_format(date):
    """
    Return an ISO-8601 representation of a datetime object.
    """
    return "{0:0>4}-{1:0>2}-{2:0>2}T{3:0>2}:{4:0>2}:{5:0>2}Z".format(
        date.year, date.month, date.day, date.hour,
        date.minute, date.second)

def check_key(self, key: str) -> bool:
        """
        Checks if key exists in datastore. True if yes, False if no.

        :param: SHA512 hash key

        :return: whether or key not exists in datastore
        """
        keys = self.get_keys()
        return key in keys

def get_bin_edges_from_axis(axis) -> np.ndarray:
    """ Get bin edges from a ROOT hist axis.

    Note:
        Doesn't include over- or underflow bins!

    Args:
        axis (ROOT.TAxis): Axis from which the bin edges should be extracted.
    Returns:
        Array containing the bin edges.
    """
    # Don't include over- or underflow bins
    bins = range(1, axis.GetNbins() + 1)
    # Bin edges
    bin_edges = np.empty(len(bins) + 1)
    bin_edges[:-1] = [axis.GetBinLowEdge(i) for i in bins]
    bin_edges[-1] = axis.GetBinUpEdge(axis.GetNbins())

    return bin_edges

def top(self, topn=10):
        """
        Get a list of the top ``topn`` features in this :class:`.Feature`\.

        Examples
        --------

        .. code-block:: python

        >>> myFeature = Feature([('the', 2), ('pine', 1), ('trapezoid', 5)])
        >>> myFeature.top(1)
        [('trapezoid', 5)]

        Parameters
        ----------
        topn : int

        Returns
        -------
        list
        """
        return [self[i] for i in argsort(list(zip(*self))[1])[::-1][:topn]]

def cmd_dot(conf: Config):
    """Print out a neat targets dependency tree based on requested targets.

    Use graphviz to render the dot file, e.g.:

    > ybt dot :foo :bar | dot -Tpng -o graph.png
    """
    build_context = BuildContext(conf)
    populate_targets_graph(build_context, conf)
    if conf.output_dot_file is None:
        write_dot(build_context, conf, sys.stdout)
    else:
        with open(conf.output_dot_file, 'w') as out_file:
            write_dot(build_context, conf, out_file)

def astensor(array: TensorLike) -> BKTensor:
    """Covert numpy array to tensorflow tensor"""
    tensor = tf.convert_to_tensor(value=array, dtype=CTYPE)
    return tensor

def grep(pattern, filename):
    """Very simple grep that returns the first matching line in a file.
    String matching only, does not do REs as currently implemented.
    """
    try:
        # for line in file
        # if line matches pattern:
        #    return line
        return next((L for L in open(filename) if L.find(pattern) >= 0))
    except StopIteration:
        return ''

def lowercase_chars(string: any) -> str:
        """Return all (and only) the lowercase chars in the given string."""
        return ''.join([c if c.islower() else '' for c in str(string)])

def list_to_str(lst):
    """
    Turn a list into a comma- and/or and-separated string.

    Parameters
    ----------
    lst : :obj:`list`
        A list of strings to join into a single string.

    Returns
    -------
    str_ : :obj:`str`
        A string with commas and/or ands separating th elements from ``lst``.

    """
    if len(lst) == 1:
        str_ = lst[0]
    elif len(lst) == 2:
        str_ = ' and '.join(lst)
    elif len(lst) > 2:
        str_ = ', '.join(lst[:-1])
        str_ += ', and {0}'.format(lst[-1])
    else:
        raise ValueError('List of length 0 provided.')
    return str_

def get_language(query: str) -> str:
    """Tries to work out the highlight.js language of a given file name or
    shebang. Returns an empty string if none match.
    """
    query = query.lower()
    for language in LANGUAGES:
        if query.endswith(language):
            return language
    return ''

def get_datatype(self, table: str, column: str) -> str:
        """Returns database SQL datatype for a column: e.g. VARCHAR."""
        return self.flavour.get_datatype(self, table, column).upper()

def lint(fmt='colorized'):
    """Run verbose PyLint on source. Optionally specify fmt=html for HTML output."""
    if fmt == 'html':
        outfile = 'pylint_report.html'
        local('pylint -f %s davies > %s || true' % (fmt, outfile))
        local('open %s' % outfile)
    else:
        local('pylint -f %s davies || true' % fmt)

def chars(string: any) -> str:
        """Return all (and only) the chars in the given string."""
        return ''.join([c if c.isalpha() else '' for c in str(string)])

def is_finite(value: Any) -> bool:
    """Return true if a value is a finite number."""
    return isinstance(value, int) or (isinstance(value, float) and isfinite(value))

def _strip_top_comments(lines: Sequence[str], line_separator: str) -> str:
        """Strips # comments that exist at the top of the given lines"""
        lines = copy.copy(lines)
        while lines and lines[0].startswith("#"):
            lines = lines[1:]
        return line_separator.join(lines)

def find_first_in_list(txt: str, str_list: [str]) -> int:  # type: ignore
    """
    Returns the index of the earliest occurence of an item from a list in a string

    Ex: find_first_in_list('foobar', ['bar', 'fin']) -> 3
    """
    start = len(txt) + 1
    for item in str_list:
        if start > txt.find(item) > -1:
            start = txt.find(item)
    return start if len(txt) + 1 > start > -1 else -1

def scope_logger(cls):
    """
    Class decorator for adding a class local logger

    Example:
    >>> @scope_logger
    >>> class Test:
    >>>     def __init__(self):
    >>>         self.log.info("class instantiated")
    >>> t = Test()
    
    """
    cls.log = logging.getLogger('{0}.{1}'.format(cls.__module__, cls.__name__))
    return cls

def is_quoted(arg: str) -> bool:
    """
    Checks if a string is quoted
    :param arg: the string being checked for quotes
    :return: True if a string is quoted
    """
    return len(arg) > 1 and arg[0] == arg[-1] and arg[0] in constants.QUOTES

def dotproduct(X, Y):
    """Return the sum of the element-wise product of vectors x and y.
    >>> dotproduct([1, 2, 3], [1000, 100, 10])
    1230
    """
    return sum([x * y for x, y in zip(X, Y)])

def isfile_notempty(inputfile: str) -> bool:
        """Check if the input filename with path is a file and is not empty."""
        try:
            return isfile(inputfile) and getsize(inputfile) > 0
        except TypeError:
            raise TypeError('inputfile is not a valid type')

def read_flat(self):
        """
        Read a PNG file and decode it into flat row flat pixel format.

        Returns (*width*, *height*, *pixels*, *metadata*).

        May use excessive memory.

        `pixels` are returned in flat row flat pixel format.

        See also the :meth:`read` method which returns pixels in the
        more stream-friendly boxed row flat pixel format.
        """
        x, y, pixel, meta = self.read()
        arraycode = 'BH'[meta['bitdepth'] > 8]
        pixel = array(arraycode, itertools.chain(*pixel))
        return x, y, pixel, meta

def SwitchToThisWindow(handle: int) -> None:
    """
    SwitchToThisWindow from Win32.
    handle: int, the handle of a native window.
    """
    ctypes.windll.user32.SwitchToThisWindow(ctypes.c_void_p(handle), 1)

def hex_color_to_tuple(hex):
    """ convent hex color to tuple
    "#ffffff"   ->  (255, 255, 255)
    "#ffff00ff" ->  (255, 255, 0, 255)
    """
    hex = hex[1:]
    length = len(hex) // 2
    return tuple(int(hex[i*2:i*2+2], 16) for i in range(length))

def mmap(func, iterable):
    """Wrapper to make map() behave the same on Py2 and Py3."""

    if sys.version_info[0] > 2:
        return [i for i in map(func, iterable)]
    else:
        return map(func, iterable)

def has_value(cls, value: int) -> bool:
        """True if specified value exists in int enum; otherwise, False."""
        return any(value == item.value for item in cls)

def __remove_trailing_zeros(self, collection):
        """Removes trailing zeroes from indexable collection of numbers"""
        index = len(collection) - 1
        while index >= 0 and collection[index] == 0:
            index -= 1

        return collection[:index + 1]

def Exit(msg, code=1):
    """Exit execution with return code and message
    :param msg: Message displayed prior to exit
    :param code: code returned upon exiting
    """
    print >> sys.stderr, msg
    sys.exit(code)

def check64bit(current_system="python"):
    """checks if you are on a 64 bit platform"""
    if current_system == "python":
        return sys.maxsize > 2147483647
    elif current_system == "os":
        import platform
        pm = platform.machine()
        if pm != ".." and pm.endswith('64'):  # recent Python (not Iron)
            return True
        else:
            if 'PROCESSOR_ARCHITEW6432' in os.environ:
                return True  # 32 bit program running on 64 bit Windows
            try:
                # 64 bit Windows 64 bit program
                return os.environ['PROCESSOR_ARCHITECTURE'].endswith('64')
            except IndexError:
                pass  # not Windows
            try:
                # this often works in Linux
                return '64' in platform.architecture()[0]
            except Exception:
                # is an older version of Python, assume also an older os@
                # (best we can guess)
                return False

def clean_column_names(df: DataFrame) -> DataFrame:
    """
    Strip the whitespace from all column names in the given DataFrame
    and return the result.
    """
    f = df.copy()
    f.columns = [col.strip() for col in f.columns]
    return f

def __rmatmul__(self, other):
        """
        Matrix multiplication using binary `@` operator in Python>=3.5.
        """
        return self.T.dot(np.transpose(other)).T

def decodebytes(input):
    """Decode base64 string to byte array."""
    py_version = sys.version_info[0]
    if py_version >= 3:
        return _decodebytes_py3(input)
    return _decodebytes_py2(input)

def string(value) -> str:
        """ string dict/object/value to JSON """
        return system_json.dumps(Json(value).safe_object(), ensure_ascii=False)

def parse_reading(val: str) -> Optional[float]:
    """ Convert reading value to float (if possible) """
    try:
        return float(val)
    except ValueError:
        logging.warning('Reading of "%s" is not a number', val)
        return None

def from_buffer(buffer, mime=False):
    """
    Accepts a binary string and returns the detected filetype.  Return
    value is the mimetype if mime=True, otherwise a human readable
    name.

    >>> magic.from_buffer(open("testdata/test.pdf").read(1024))
    'PDF document, version 1.2'
    """
    m = _get_magic_type(mime)
    return m.from_buffer(buffer)

def mmap(func, iterable):
    """Wrapper to make map() behave the same on Py2 and Py3."""

    if sys.version_info[0] > 2:
        return [i for i in map(func, iterable)]
    else:
        return map(func, iterable)

def check_key(self, key: str) -> bool:
        """
        Checks if key exists in datastore. True if yes, False if no.

        :param: SHA512 hash key

        :return: whether or key not exists in datastore
        """
        keys = self.get_keys()
        return key in keys

def capitalize(string):
    """Capitalize a sentence.

    Parameters
    ----------
    string : `str`
        String to capitalize.

    Returns
    -------
    `str`
        Capitalized string.

    Examples
    --------
    >>> capitalize('worD WORD WoRd')
    'Word word word'
    """
    if not string:
        return string
    if len(string) == 1:
        return string.upper()
    return string[0].upper() + string[1:].lower()

def iterate_items(dictish):
    """ Return a consistent (key, value) iterable on dict-like objects,
    including lists of tuple pairs.

    Example:

        >>> list(iterate_items({'a': 1}))
        [('a', 1)]
        >>> list(iterate_items([('a', 1), ('b', 2)]))
        [('a', 1), ('b', 2)]
    """
    if hasattr(dictish, 'iteritems'):
        return dictish.iteritems()
    if hasattr(dictish, 'items'):
        return dictish.items()
    return dictish

def _run_sync(self, method: Callable, *args, **kwargs) -> Any:
        """
        Utility method to run commands synchronously for testing.
        """
        if self.loop.is_running():
            raise RuntimeError("Event loop is already running.")

        if not self.is_connected:
            self.loop.run_until_complete(self.connect())

        task = asyncio.Task(method(*args, **kwargs), loop=self.loop)
        result = self.loop.run_until_complete(task)

        self.loop.run_until_complete(self.quit())

        return result

def str_to_time(time_str: str) -> datetime.datetime:
    """
    Convert human readable string to datetime.datetime.
    """
    pieces: Any = [int(piece) for piece in time_str.split('-')]
    return datetime.datetime(*pieces)

def maybe_infer_dtype_type(element):
    """Try to infer an object's dtype, for use in arithmetic ops

    Uses `element.dtype` if that's available.
    Objects implementing the iterator protocol are cast to a NumPy array,
    and from there the array's type is used.

    Parameters
    ----------
    element : object
        Possibly has a `.dtype` attribute, and possibly the iterator
        protocol.

    Returns
    -------
    tipo : type

    Examples
    --------
    >>> from collections import namedtuple
    >>> Foo = namedtuple("Foo", "dtype")
    >>> maybe_infer_dtype_type(Foo(np.dtype("i8")))
    numpy.int64
    """
    tipo = None
    if hasattr(element, 'dtype'):
        tipo = element.dtype
    elif is_list_like(element):
        element = np.asarray(element)
        tipo = element.dtype
    return tipo

def get_keys_of_max_n(dict_obj, n):
    """Returns the keys that maps to the top n max values in the given dict.

    Example:
    --------
    >>> dict_obj = {'a':2, 'b':1, 'c':5}
    >>> get_keys_of_max_n(dict_obj, 2)
    ['a', 'c']
    """
    return sorted([
        item[0]
        for item in sorted(
            dict_obj.items(), key=lambda item: item[1], reverse=True
        )[:n]
    ])

def get_now_sql_datetime():
    """
    *A datetime stamp in MySQL format: ``YYYY-MM-DDTHH:MM:SS``*

    **Return:**
        - ``now`` -- current time and date in MySQL format

    **Usage:**
        .. code-block:: python 

            from fundamentals import times
            now = times.get_now_sql_datetime()
            print now

            # OUT: 2016-03-18T11:08:23 
    """
    ## > IMPORTS ##
    from datetime import datetime, date, time
    now = datetime.now()
    now = now.strftime("%Y-%m-%dT%H:%M:%S")

    return now

def export_to_dot(self, filename: str = 'output') -> None:
        """ Export the graph to the dot file "filename.dot". """
        with open(filename + '.dot', 'w') as output:
            output.write(self.as_dot())

def read_flat(self):
        """
        Read a PNG file and decode it into flat row flat pixel format.

        Returns (*width*, *height*, *pixels*, *metadata*).

        May use excessive memory.

        `pixels` are returned in flat row flat pixel format.

        See also the :meth:`read` method which returns pixels in the
        more stream-friendly boxed row flat pixel format.
        """
        x, y, pixel, meta = self.read()
        arraycode = 'BH'[meta['bitdepth'] > 8]
        pixel = array(arraycode, itertools.chain(*pixel))
        return x, y, pixel, meta

def to_iso_string(self) -> str:
        """ Returns full ISO string for the given date """
        assert isinstance(self.value, datetime)
        return datetime.isoformat(self.value)

def from_iso_time(timestring, use_dateutil=True):
    """Parse an ISO8601-formatted datetime string and return a datetime.time
    object.
    """
    if not _iso8601_time_re.match(timestring):
        raise ValueError('Not a valid ISO8601-formatted time string')
    if dateutil_available and use_dateutil:
        return parser.parse(timestring).time()
    else:
        if len(timestring) > 8:  # has microseconds
            fmt = '%H:%M:%S.%f'
        else:
            fmt = '%H:%M:%S'
        return datetime.datetime.strptime(timestring, fmt).time()

def get_timezone() -> Tuple[datetime.tzinfo, str]:
    """Discover the current time zone and it's standard string representation (for source{d})."""
    dt = get_datetime_now().astimezone()
    tzstr = dt.strftime("%z")
    tzstr = tzstr[:-2] + ":" + tzstr[-2:]
    return dt.tzinfo, tzstr

def decodebytes(input):
    """Decode base64 string to byte array."""
    py_version = sys.version_info[0]
    if py_version >= 3:
        return _decodebytes_py3(input)
    return _decodebytes_py2(input)

def fix_title_capitalization(title):
    """Try to capitalize properly a title string."""
    if re.search("[A-Z]", title) and re.search("[a-z]", title):
        return title
    word_list = re.split(' +', title)
    final = [word_list[0].capitalize()]
    for word in word_list[1:]:
        if word.upper() in COMMON_ACRONYMS:
            final.append(word.upper())
        elif len(word) > 3:
            final.append(word.capitalize())
        else:
            final.append(word.lower())
    return " ".join(final)

def argsort_k_smallest(x, k):
    """ Return no more than ``k`` indices of smallest values. """
    if k == 0:
        return np.array([], dtype=np.intp)
    if k is None or k >= len(x):
        return np.argsort(x)
    indices = np.argpartition(x, k)[:k]
    values = x[indices]
    return indices[np.argsort(values)]

def area (self):
    """area() -> number

    Returns the area of this Polygon.
    """
    area = 0.0
    
    for segment in self.segments():
      area += ((segment.p.x * segment.q.y) - (segment.q.x * segment.p.y))/2

    return area

def PrintIndented(self, file, ident, code):
        """Takes an array, add indentation to each entry and prints it."""
        for entry in code:
            print >>file, '%s%s' % (ident, entry)

def getElementsBy(self, cond: Callable[[Element], bool]) -> NodeList:
        """Get elements in this document which matches condition."""
        return getElementsBy(self, cond)

def debugTreePrint(node,pfx="->"):
  """Purely a debugging aid: Ascii-art picture of a tree descended from node"""
  print pfx,node.item
  for c in node.children:
    debugTreePrint(c,"  "+pfx)

def get_from_gnucash26_date(date_str: str) -> date:
    """ Creates a datetime from GnuCash 2.6 date string """
    date_format = "%Y%m%d"
    result = datetime.strptime(date_str, date_format).date()
    return result

def remove_blank_lines(string):
    """ Removes all blank lines in @string

        -> #str without blank lines
    """
    return "\n".join(line
                     for line in string.split("\n")
                     if len(line.strip()))

def uuid2buid(value):
    """
    Convert a UUID object to a 22-char BUID string

    >>> u = uuid.UUID('33203dd2-f2ef-422f-aeb0-058d6f5f7089')
    >>> uuid2buid(u)
    'MyA90vLvQi-usAWNb19wiQ'
    """
    if six.PY3:  # pragma: no cover
        return urlsafe_b64encode(value.bytes).decode('utf-8').rstrip('=')
    else:
        return six.text_type(urlsafe_b64encode(value.bytes).rstrip('='))

def strtobytes(input, encoding):
    """Take a str and transform it into a byte array."""
    py_version = sys.version_info[0]
    if py_version >= 3:
        return _strtobytes_py3(input, encoding)
    return _strtobytes_py2(input, encoding)

def gen_lower(x: Iterable[str]) -> Generator[str, None, None]:
    """
    Args:
        x: iterable of strings

    Yields:
        each string in lower case
    """
    for string in x:
        yield string.lower()

def __remove_trailing_zeros(self, collection):
        """Removes trailing zeroes from indexable collection of numbers"""
        index = len(collection) - 1
        while index >= 0 and collection[index] == 0:
            index -= 1

        return collection[:index + 1]

def iter_lines(file_like: Iterable[str]) -> Generator[str, None, None]:
    """ Helper for iterating only nonempty lines without line breaks"""
    for line in file_like:
        line = line.rstrip('\r\n')
        if line:
            yield line

def dag_longest_path(graph, source, target):
    """
    Finds the longest path in a dag between two nodes
    """
    if source == target:
        return [source]
    allpaths = nx.all_simple_paths(graph, source, target)
    longest_path = []
    for l in allpaths:
        if len(l) > len(longest_path):
            longest_path = l
    return longest_path

def is_sqlatype_string(coltype: Union[TypeEngine, VisitableType]) -> bool:
    """
    Is the SQLAlchemy column type a string type?
    """
    coltype = _coltype_to_typeengine(coltype)
    return isinstance(coltype, sqltypes.String)

def replace(s, old, new, maxreplace=-1):
    """replace (str, old, new[, maxreplace]) -> string

    Return a copy of string str with all occurrences of substring
    old replaced by new. If the optional argument maxreplace is
    given, only the first maxreplace occurrences are replaced.

    """
    return s.replace(old, new, maxreplace)

def most_frequent(lst):
    """
    Returns the item that appears most frequently in the given list.
    """
    lst = lst[:]
    highest_freq = 0
    most_freq = None

    for val in unique(lst):
        if lst.count(val) > highest_freq:
            most_freq = val
            highest_freq = lst.count(val)
            
    return most_freq

def _prm_get_longest_stringsize(string_list):
        """ Returns the longest string size for a string entry across data."""
        maxlength = 1

        for stringar in string_list:
            if isinstance(stringar, np.ndarray):
                if stringar.ndim > 0:
                    for string in stringar.ravel():
                        maxlength = max(len(string), maxlength)
                else:
                    maxlength = max(len(stringar.tolist()), maxlength)
            else:
                maxlength = max(len(stringar), maxlength)

        # Make the string Col longer than needed in order to allow later on slightly larger strings
        return int(maxlength * 1.5)

def rmglob(pattern: str) -> None:
    """
    Deletes all files whose filename matches the glob ``pattern`` (via
    :func:`glob.glob`).
    """
    for f in glob.glob(pattern):
        os.remove(f)

def list_depth(list_, func=max, _depth=0):
    """
    Returns the deepest level of nesting within a list of lists

    Args:
       list_  : a nested listlike object
       func   : depth aggregation strategy (defaults to max)
       _depth : internal var

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list_ = [[[[[1]]], [3]], [[1], [3]], [[1], [3]]]
        >>> result = (list_depth(list_, _depth=0))
        >>> print(result)

    """
    depth_list = [list_depth(item, func=func, _depth=_depth + 1)
                  for item in  list_ if util_type.is_listlike(item)]
    if len(depth_list) > 0:
        return func(depth_list)
    else:
        return _depth

def rank(tensor: BKTensor) -> int:
    """Return the number of dimensions of a tensor"""
    if isinstance(tensor, np.ndarray):
        return len(tensor.shape)

    return len(tensor[0].size())

def get_longest_line_length(text):
    """Get the length longest line in a paragraph"""
    lines = text.split("\n")
    length = 0

    for i in range(len(lines)):
        if len(lines[i]) > length:
            length = len(lines[i])

    return length

def flatten_list(x: List[Any]) -> List[Any]:
    """
    Converts a list of lists into a flat list.
    
    Args:
        x: list of lists 

    Returns:
        flat list
        
    As per
    http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python

    """  # noqa
    return [item for sublist in x for item in sublist]

def _rindex(mylist: Sequence[T], x: T) -> int:
    """Index of the last occurrence of x in the sequence."""
    return len(mylist) - mylist[::-1].index(x) - 1

def constant(times: np.ndarray, amp: complex) -> np.ndarray:
    """Continuous constant pulse.

    Args:
        times: Times to output pulse for.
        amp: Complex pulse amplitude.
    """
    return np.full(len(times), amp, dtype=np.complex_)

def is_finite(value: Any) -> bool:
    """Return true if a value is a finite number."""
    return isinstance(value, int) or (isinstance(value, float) and isfinite(value))

def post(self, endpoint: str, **kwargs) -> dict:
        """HTTP POST operation to API endpoint."""

        return self._request('POST', endpoint, **kwargs)

def sort_by_modified(files_or_folders: list) -> list:
    """
    Sort files or folders by modified time

    Args:
        files_or_folders: list of files or folders

    Returns:
        list
    """
    return sorted(files_or_folders, key=os.path.getmtime, reverse=True)

def cli_run():
    """docstring for argparse"""
    parser = argparse.ArgumentParser(description='Stupidly simple code answers from StackOverflow')
    parser.add_argument('query', help="What's the problem ?", type=str, nargs='+')
    parser.add_argument('-t','--tags', help='semicolon separated tags -> python;lambda')
    args = parser.parse_args()
    main(args)

def flatten_list(x: List[Any]) -> List[Any]:
    """
    Converts a list of lists into a flat list.
    
    Args:
        x: list of lists 

    Returns:
        flat list
        
    As per
    http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python

    """  # noqa
    return [item for sublist in x for item in sublist]

def indices_to_labels(self, indices: Sequence[int]) -> List[str]:
        """ Converts a sequence of indices into their corresponding labels."""

        return [(self.INDEX_TO_LABEL[index]) for index in indices]

def strip_codes(s: Any) -> str:
    """ Strip all color codes from a string.
        Returns empty string for "falsey" inputs.
    """
    return codepat.sub('', str(s) if (s or (s == 0)) else '')

def _(f, x):
    """
    filter for dict, note `f` should have signature: `f::key->value->bool`
    """
    return {k: v for k, v in x.items() if f(k, v)}

def to_jupyter(graph: BELGraph, chart: Optional[str] = None) -> Javascript:
    """Render the graph as JavaScript in a Jupyter Notebook."""
    with open(os.path.join(HERE, 'render_with_javascript.js'), 'rt') as f:
        js_template = Template(f.read())

    return Javascript(js_template.render(**_get_context(graph, chart=chart)))

def valid_file(path: str) -> bool:
    """
    Verifies that a string path actually exists and is a file

    :param path: The path to verify
    :return: **True** if path exist and is a file
    """
    path = Path(path).expanduser()
    log.debug("checking if %s is a valid file", path)
    return path.exists() and path.is_file()

def normalize(numbers):
    """Multiply each number by a constant such that the sum is 1.0
    >>> normalize([1,2,1])
    [0.25, 0.5, 0.25]
    """
    total = float(sum(numbers))
    return [n / total for n in numbers]

def is_line_in_file(filename: str, line: str) -> bool:
    """
    Detects whether a line is present within a file.

    Args:
        filename: file to check
        line: line to search for (as an exact match)
    """
    assert "\n" not in line
    with open(filename, "r") as file:
        for fileline in file:
            if fileline == line:
                return True
        return False

def negate_mask(mask):
    """Returns the negated mask.

    If elements of input mask have 0 and non-zero values, then the returned matrix will have all elements 0 (1) where
    the original one has non-zero (0).

    :param mask: Input mask
    :type mask: np.array
    :return: array of same shape and dtype=int8 as input array
    :rtype: np.array
    """
    res = np.ones(mask.shape, dtype=np.int8)
    res[mask > 0] = 0

    return res

def _run_sync(self, method: Callable, *args, **kwargs) -> Any:
        """
        Utility method to run commands synchronously for testing.
        """
        if self.loop.is_running():
            raise RuntimeError("Event loop is already running.")

        if not self.is_connected:
            self.loop.run_until_complete(self.connect())

        task = asyncio.Task(method(*args, **kwargs), loop=self.loop)
        result = self.loop.run_until_complete(task)

        self.loop.run_until_complete(self.quit())

        return result

def pack_bits( longbits ):
    """Crunch a 64-bit int (8 bool bytes) into a bitfield."""
    byte = longbits & (0x0101010101010101)
    byte = (byte | (byte>>7)) & (0x0003000300030003)
    byte = (byte | (byte>>14)) & (0x0000000f0000000f)
    byte = (byte | (byte>>28)) & (0x00000000000000ff)
    return byte

def local_machine_uuid():
    """Return local machine unique identifier.

    >>> uuid = local_machine_uuid()

    """

    result = subprocess.check_output(
        'hal-get-property --udi '
        '/org/freedesktop/Hal/devices/computer '
        '--key system.hardware.uuid'.split()
        ).strip()

    return uuid.UUID(hex=result)

def get_period_last_3_months() -> str:
    """ Returns the last week as a period string """
    today = Datum()
    today.today()

    # start_date = today - timedelta(weeks=13)
    start_date = today.clone()
    start_date.subtract_months(3)

    period = get_period(start_date.date, today.date)
    return period

def dtypes(self):
        """Returns all column names and their data types as a list.

        >>> df.dtypes
        [('age', 'int'), ('name', 'string')]
        """
        return [(str(f.name), f.dataType.simpleString()) for f in self.schema.fields]

def remove_empty_text(utterances: List[Utterance]) -> List[Utterance]:
    """Remove empty utterances from a list of utterances
    Args:
        utterances: The list of utterance we are processing
    """
    return [utter for utter in utterances if utter.text.strip() != ""]

def read_set_from_file(filename: str) -> Set[str]:
    """
    Extract a de-duped collection (set) of text from a file.
    Expected file format is one item per line.
    """
    collection = set()
    with open(filename, 'r') as file_:
        for line in file_:
            collection.add(line.rstrip())
    return collection

def top(self, topn=10):
        """
        Get a list of the top ``topn`` features in this :class:`.Feature`\.

        Examples
        --------

        .. code-block:: python

        >>> myFeature = Feature([('the', 2), ('pine', 1), ('trapezoid', 5)])
        >>> myFeature.top(1)
        [('trapezoid', 5)]

        Parameters
        ----------
        topn : int

        Returns
        -------
        list
        """
        return [self[i] for i in argsort(list(zip(*self))[1])[::-1][:topn]]

def get_timezone() -> Tuple[datetime.tzinfo, str]:
    """Discover the current time zone and it's standard string representation (for source{d})."""
    dt = get_datetime_now().astimezone()
    tzstr = dt.strftime("%z")
    tzstr = tzstr[:-2] + ":" + tzstr[-2:]
    return dt.tzinfo, tzstr

def indexes_equal(a: Index, b: Index) -> bool:
    """
    Are two indexes equal? Checks by comparing ``str()`` versions of them.
    (AM UNSURE IF THIS IS ENOUGH.)
    """
    return str(a) == str(b)

def cli_run():
    """docstring for argparse"""
    parser = argparse.ArgumentParser(description='Stupidly simple code answers from StackOverflow')
    parser.add_argument('query', help="What's the problem ?", type=str, nargs='+')
    parser.add_argument('-t','--tags', help='semicolon separated tags -> python;lambda')
    args = parser.parse_args()
    main(args)

def execute(cur, *args):
    """Utility function to print sqlite queries before executing.

    Use instead of cur.execute().  First argument is cursor.

    cur.execute(stmt)
    becomes
    util.execute(cur, stmt)
    """
    stmt = args[0]
    if len(args) > 1:
        stmt = stmt.replace('%', '%%').replace('?', '%r')
        print(stmt % (args[1]))
    return cur.execute(*args)

def normalize(numbers):
    """Multiply each number by a constant such that the sum is 1.0
    >>> normalize([1,2,1])
    [0.25, 0.5, 0.25]
    """
    total = float(sum(numbers))
    return [n / total for n in numbers]

def is_line_in_file(filename: str, line: str) -> bool:
    """
    Detects whether a line is present within a file.

    Args:
        filename: file to check
        line: line to search for (as an exact match)
    """
    assert "\n" not in line
    with open(filename, "r") as file:
        for fileline in file:
            if fileline == line:
                return True
        return False

def mmap(func, iterable):
    """Wrapper to make map() behave the same on Py2 and Py3."""

    if sys.version_info[0] > 2:
        return [i for i in map(func, iterable)]
    else:
        return map(func, iterable)

def last_location_of_minimum(x):
    """
    Returns the last location of the minimal value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    return 1.0 - np.argmin(x[::-1]) / len(x) if len(x) > 0 else np.NaN

def csv_to_numpy(string_like, dtype=None):  # type: (str) -> np.array
    """Convert a CSV object to a numpy array.

    Args:
        string_like (str): CSV string.
        dtype (dtype, optional):  Data type of the resulting array. If None, the dtypes will be determined by the
                                        contents of each column, individually. This argument can only be used to
                                        'upcast' the array.  For downcasting, use the .astype(t) method.
    Returns:
        (np.array): numpy array
    """
    stream = StringIO(string_like)
    return np.genfromtxt(stream, dtype=dtype, delimiter=',')

def normcdf(x, log=False):
    """Normal cumulative density function."""
    y = np.atleast_1d(x).copy()
    flib.normcdf(y)
    if log:
        if (y>0).all():
            return np.log(y)
        return -np.inf
    return y

def find_first_in_list(txt: str, str_list: [str]) -> int:  # type: ignore
    """
    Returns the index of the earliest occurence of an item from a list in a string

    Ex: find_first_in_list('foobar', ['bar', 'fin']) -> 3
    """
    start = len(txt) + 1
    for item in str_list:
        if start > txt.find(item) > -1:
            start = txt.find(item)
    return start if len(txt) + 1 > start > -1 else -1

def mmap(func, iterable):
    """Wrapper to make map() behave the same on Py2 and Py3."""

    if sys.version_info[0] > 2:
        return [i for i in map(func, iterable)]
    else:
        return map(func, iterable)

def is_orthogonal(
        matrix: np.ndarray,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8) -> bool:
    """Determines if a matrix is approximately orthogonal.

    A matrix is orthogonal if it's square and real and its transpose is its
    inverse.

    Args:
        matrix: The matrix to check.
        rtol: The per-matrix-entry relative tolerance on equality.
        atol: The per-matrix-entry absolute tolerance on equality.

    Returns:
        Whether the matrix is orthogonal within the given tolerance.
    """
    return (matrix.shape[0] == matrix.shape[1] and
            np.all(np.imag(matrix) == 0) and
            np.allclose(matrix.dot(matrix.T), np.eye(matrix.shape[0]),
                        rtol=rtol,
                        atol=atol))

def butlast(iterable):
    """Yield all items from ``iterable`` except the last one.

    >>> list(butlast(['spam', 'eggs', 'ham']))
    ['spam', 'eggs']

    >>> list(butlast(['spam']))
    []

    >>> list(butlast([]))
    []
    """
    iterable = iter(iterable)
    try:
        first = next(iterable)
    except StopIteration:
        return
    for second in iterable:
        yield first
        first = second

def year(date):
    """ Returns the year.

    :param date:
        The string date with this format %m/%d/%Y
    :type date:
        String

    :returns:
        int

    :example:
        >>> year('05/1/2015')
        2015
    """
    try:
        fmt = '%m/%d/%Y'
        return datetime.strptime(date, fmt).timetuple().tm_year
    except ValueError:
        return 0

def _brief_print_list(lst, limit=7):
    """Print at most `limit` elements of list."""
    lst = list(lst)
    if len(lst) > limit:
        return _brief_print_list(lst[:limit//2], limit) + ', ..., ' + \
            _brief_print_list(lst[-limit//2:], limit)
    return ', '.join(["'%s'"%str(i) for i in lst])

def is_not_null(df: DataFrame, col_name: str) -> bool:
    """
    Return ``True`` if the given DataFrame has a column of the given
    name (string), and there exists at least one non-NaN value in that
    column; return ``False`` otherwise.
    """
    if (
        isinstance(df, pd.DataFrame)
        and col_name in df.columns
        and df[col_name].notnull().any()
    ):
        return True
    else:
        return False

def chars(string: any) -> str:
        """Return all (and only) the chars in the given string."""
        return ''.join([c if c.isalpha() else '' for c in str(string)])

def get_value(self) -> Decimal:
        """ Returns the current value of stocks """
        quantity = self.get_quantity()
        price = self.get_last_available_price()
        if not price:
            # raise ValueError("no price found for", self.full_symbol)
            return Decimal(0)

        value = quantity * price.value
        return value

def setup_cache(app: Flask, cache_config) -> Optional[Cache]:
    """Setup the flask-cache on a flask app"""
    if cache_config and cache_config.get('CACHE_TYPE') != 'null':
        return Cache(app, config=cache_config)

    return None

def find_first_in_list(txt: str, str_list: [str]) -> int:  # type: ignore
    """
    Returns the index of the earliest occurence of an item from a list in a string

    Ex: find_first_in_list('foobar', ['bar', 'fin']) -> 3
    """
    start = len(txt) + 1
    for item in str_list:
        if start > txt.find(item) > -1:
            start = txt.find(item)
    return start if len(txt) + 1 > start > -1 else -1

def get_keys_of_max_n(dict_obj, n):
    """Returns the keys that maps to the top n max values in the given dict.

    Example:
    --------
    >>> dict_obj = {'a':2, 'b':1, 'c':5}
    >>> get_keys_of_max_n(dict_obj, 2)
    ['a', 'c']
    """
    return sorted([
        item[0]
        for item in sorted(
            dict_obj.items(), key=lambda item: item[1], reverse=True
        )[:n]
    ])

def year(date):
    """ Returns the year.

    :param date:
        The string date with this format %m/%d/%Y
    :type date:
        String

    :returns:
        int

    :example:
        >>> year('05/1/2015')
        2015
    """
    try:
        fmt = '%m/%d/%Y'
        return datetime.strptime(date, fmt).timetuple().tm_year
    except ValueError:
        return 0

def _mid(pt1, pt2):
    """
    (Point, Point) -> Point
    Return the point that lies in between the two input points.
    """
    (x0, y0), (x1, y1) = pt1, pt2
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)

def valid_file(path: str) -> bool:
    """
    Verifies that a string path actually exists and is a file

    :param path: The path to verify
    :return: **True** if path exist and is a file
    """
    path = Path(path).expanduser()
    log.debug("checking if %s is a valid file", path)
    return path.exists() and path.is_file()

def is_inside_lambda(node: astroid.node_classes.NodeNG) -> bool:
    """Return true if given node is inside lambda"""
    parent = node.parent
    while parent is not None:
        if isinstance(parent, astroid.Lambda):
            return True
        parent = parent.parent
    return False

def stdout_encode(u, default='utf-8'):
    """ Encodes a given string with the proper standard out encoding
        If sys.stdout.encoding isn't specified, it this defaults to @default

        @default: default encoding

        -> #str with standard out encoding
    """
    # from http://stackoverflow.com/questions/3627793/best-output-type-and-
    #   encoding-practices-for-repr-functions
    encoding = sys.stdout.encoding or default
    return u.encode(encoding, "replace").decode(encoding, "replace")

def read_text_from_file(path: str) -> str:
    """ Reads text file contents """
    with open(path) as text_file:
        content = text_file.read()

    return content

def get_domain(url):
    """
    Get domain part of an url.

    For example: https://www.python.org/doc/ -> https://www.python.org
    """
    parse_result = urlparse(url)
    domain = "{schema}://{netloc}".format(
        schema=parse_result.scheme, netloc=parse_result.netloc)
    return domain

def _prm_get_longest_stringsize(string_list):
        """ Returns the longest string size for a string entry across data."""
        maxlength = 1

        for stringar in string_list:
            if isinstance(stringar, np.ndarray):
                if stringar.ndim > 0:
                    for string in stringar.ravel():
                        maxlength = max(len(string), maxlength)
                else:
                    maxlength = max(len(stringar.tolist()), maxlength)
            else:
                maxlength = max(len(stringar), maxlength)

        # Make the string Col longer than needed in order to allow later on slightly larger strings
        return int(maxlength * 1.5)

def multiple_replace(string, replacements):
    # type: (str, Dict[str,str]) -> str
    """Simultaneously replace multiple strigns in a string

    Args:
        string (str): Input string
        replacements (Dict[str,str]): Replacements dictionary

    Returns:
        str: String with replacements

    """
    pattern = re.compile("|".join([re.escape(k) for k in sorted(replacements, key=len, reverse=True)]), flags=re.DOTALL)
    return pattern.sub(lambda x: replacements[x.group(0)], string)

def uppercase_chars(string: any) -> str:
        """Return all (and only) the uppercase chars in the given string."""
        return ''.join([c if c.isupper() else '' for c in str(string)])

def remove_empty_text(utterances: List[Utterance]) -> List[Utterance]:
    """Remove empty utterances from a list of utterances
    Args:
        utterances: The list of utterance we are processing
    """
    return [utter for utter in utterances if utter.text.strip() != ""]

def astensor(array: TensorLike) -> BKTensor:
    """Covert numpy array to tensorflow tensor"""
    tensor = tf.convert_to_tensor(value=array, dtype=CTYPE)
    return tensor

def is_finite(value: Any) -> bool:
    """Return true if a value is a finite number."""
    return isinstance(value, int) or (isinstance(value, float) and isfinite(value))

def remove_parenthesis_around_tz(cls, timestr):
        """get rid of parenthesis around timezone: (GMT) => GMT

        :return: the new string if parenthesis were found, `None` otherwise
        """
        parenthesis = cls.TIMEZONE_PARENTHESIS.match(timestr)
        if parenthesis is not None:
            return parenthesis.group(1)

def tanimoto_set_similarity(x: Iterable[X], y: Iterable[X]) -> float:
    """Calculate the tanimoto set similarity."""
    a, b = set(x), set(y)
    union = a | b

    if not union:
        return 0.0

    return len(a & b) / len(union)

def get_timezone() -> Tuple[datetime.tzinfo, str]:
    """Discover the current time zone and it's standard string representation (for source{d})."""
    dt = get_datetime_now().astimezone()
    tzstr = dt.strftime("%z")
    tzstr = tzstr[:-2] + ":" + tzstr[-2:]
    return dt.tzinfo, tzstr

def str_to_time(time_str: str) -> datetime.datetime:
    """
    Convert human readable string to datetime.datetime.
    """
    pieces: Any = [int(piece) for piece in time_str.split('-')]
    return datetime.datetime(*pieces)

def isarray(array, test, dim=2):
    """Returns True if test is True for all array elements.
    Otherwise, returns False.
    """
    if dim > 1:
        return all(isarray(array[i], test, dim - 1)
                   for i in range(len(array)))
    return all(test(i) for i in array)

def _str_to_list(value, separator):
    """Convert a string to a list with sanitization."""
    value_list = [item.strip() for item in value.split(separator)]
    value_list_sanitized = builtins.list(filter(None, value_list))
    if len(value_list_sanitized) > 0:
        return value_list_sanitized
    else:
        raise ValueError('Invalid list variable.')

def __as_list(value: List[JsonObjTypes]) -> List[JsonTypes]:
        """ Return a json array as a list

        :param value: array
        :return: array with JsonObj instances removed
        """
        return [e._as_dict if isinstance(e, JsonObj) else e for e in value]

def _check_whitespace(string):
    """
    Make sure thre is no whitespace in the given string. Will raise a
    ValueError if whitespace is detected
    """
    if string.count(' ') + string.count('\t') + string.count('\n') > 0:
        raise ValueError(INSTRUCTION_HAS_WHITESPACE)

def simple_eq(one: Instance, two: Instance, attrs: List[str]) -> bool:
    """
    Test if two objects are equal, based on a comparison of the specified
    attributes ``attrs``.
    """
    return all(getattr(one, a) == getattr(two, a) for a in attrs)

def year(date):
    """ Returns the year.

    :param date:
        The string date with this format %m/%d/%Y
    :type date:
        String

    :returns:
        int

    :example:
        >>> year('05/1/2015')
        2015
    """
    try:
        fmt = '%m/%d/%Y'
        return datetime.strptime(date, fmt).timetuple().tm_year
    except ValueError:
        return 0

def debugTreePrint(node,pfx="->"):
  """Purely a debugging aid: Ascii-art picture of a tree descended from node"""
  print pfx,node.item
  for c in node.children:
    debugTreePrint(c,"  "+pfx)

def butlast(iterable):
    """Yield all items from ``iterable`` except the last one.

    >>> list(butlast(['spam', 'eggs', 'ham']))
    ['spam', 'eggs']

    >>> list(butlast(['spam']))
    []

    >>> list(butlast([]))
    []
    """
    iterable = iter(iterable)
    try:
        first = next(iterable)
    except StopIteration:
        return
    for second in iterable:
        yield first
        first = second

def __repr__(self) -> str:
        """Return the string representation of self."""
        return '{0}({1})'.format(type(self).__name__, repr(self.string))

def _reshuffle(mat, shape):
    """Reshuffle the indicies of a bipartite matrix A[ij,kl] -> A[lj,ki]."""
    return np.reshape(
        np.transpose(np.reshape(mat, shape), (3, 1, 2, 0)),
        (shape[3] * shape[1], shape[0] * shape[2]))

def array2string(arr: numpy.ndarray) -> str:
        """Format numpy array as a string."""
        shape = str(arr.shape)[1:-1]
        if shape.endswith(","):
            shape = shape[:-1]
        return numpy.array2string(arr, threshold=11) + "%s[%s]" % (arr.dtype, shape)

def _mid(pt1, pt2):
    """
    (Point, Point) -> Point
    Return the point that lies in between the two input points.
    """
    (x0, y0), (x1, y1) = pt1, pt2
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)

def is_quoted(arg: str) -> bool:
    """
    Checks if a string is quoted
    :param arg: the string being checked for quotes
    :return: True if a string is quoted
    """
    return len(arg) > 1 and arg[0] == arg[-1] and arg[0] in constants.QUOTES

def memory_full():
    """Check if the memory is too full for further caching."""
    current_process = psutil.Process(os.getpid())
    return (current_process.memory_percent() >
            config.MAXIMUM_CACHE_MEMORY_PERCENTAGE)

def quoted_or_list(items: List[str]) -> Optional[str]:
    """Given [A, B, C] return "'A', 'B', or 'C'".

    Note: We use single quotes here, since these are also used by repr().
    """
    return or_list([f"'{item}'" for item in items])

def _my_hash(arg_list):
    # type: (List[Any]) -> int
    """Simple helper hash function"""
    res = 0
    for arg in arg_list:
        res = res * 31 + hash(arg)
    return res

def first_digits(s, default=0):
    """Return the fist (left-hand) digits in a string as a single integer, ignoring sign (+/-).
    >>> first_digits('+123.456')
    123
    """
    s = re.split(r'[^0-9]+', str(s).strip().lstrip('+-' + charlist.whitespace))
    if len(s) and len(s[0]):
        return int(s[0])
    return default

def isarray(array, test, dim=2):
    """Returns True if test is True for all array elements.
    Otherwise, returns False.
    """
    if dim > 1:
        return all(isarray(array[i], test, dim - 1)
                   for i in range(len(array)))
    return all(test(i) for i in array)

def pack_bits( longbits ):
    """Crunch a 64-bit int (8 bool bytes) into a bitfield."""
    byte = longbits & (0x0101010101010101)
    byte = (byte | (byte>>7)) & (0x0003000300030003)
    byte = (byte | (byte>>14)) & (0x0000000f0000000f)
    byte = (byte | (byte>>28)) & (0x00000000000000ff)
    return byte

def run_time() -> timedelta:
    """

    :return:
    """

    delta = start_time if start_time else datetime.utcnow()
    return datetime.utcnow() - delta

def uppercase_chars(string: any) -> str:
        """Return all (and only) the uppercase chars in the given string."""
        return ''.join([c if c.isupper() else '' for c in str(string)])

def butlast(iterable):
    """Yield all items from ``iterable`` except the last one.

    >>> list(butlast(['spam', 'eggs', 'ham']))
    ['spam', 'eggs']

    >>> list(butlast(['spam']))
    []

    >>> list(butlast([]))
    []
    """
    iterable = iter(iterable)
    try:
        first = next(iterable)
    except StopIteration:
        return
    for second in iterable:
        yield first
        first = second

def returned(n):
	"""Generate a random walk and return True if the walker has returned to
	the origin after taking `n` steps.
	"""
	## `takei` yield lazily so we can short-circuit and avoid computing the rest of the walk
	for pos in randwalk() >> drop(1) >> takei(xrange(n-1)):
		if pos == Origin:
			return True
	return False

def get_column_names(engine: Engine, tablename: str) -> List[str]:
    """
    Get all the database column names for the specified table.
    """
    return [info.name for info in gen_columns_info(engine, tablename)]

def to_np(*args):
    """ convert GPU arras to numpy and return them"""
    if len(args) > 1:
        return (cp.asnumpy(x) for x in args)
    else:
        return cp.asnumpy(args[0])

def count(self, elem):
        """
        Return the number of elements equal to elem present in the queue

        >>> pdeque([1, 2, 1]).count(1)
        2
        """
        return self._left_list.count(elem) + self._right_list.count(elem)

def _RetryRequest(self, timeout=None, **request_args):
    """Retry the request a few times before we determine it failed.

    Sometimes the frontend becomes loaded and issues a 500 error to throttle the
    clients. We wait Client.error_poll_min seconds between each attempt to back
    off the frontend. Note that this does not affect any timing algorithm in the
    client itself which is controlled by the Timer() class.

    Args:
      timeout: Timeout for retry.
      **request_args: Args to the requests.request call.

    Returns:
      a tuple of duration, urllib.request.urlopen response.
    """
    while True:
      try:
        now = time.time()
        if not timeout:
          timeout = config.CONFIG["Client.http_timeout"]

        result = requests.request(**request_args)
        # By default requests doesn't raise on HTTP error codes.
        result.raise_for_status()

        # Requests does not always raise an exception when an incorrect response
        # is received. This fixes that behaviour.
        if not result.ok:
          raise requests.RequestException(response=result)

        return time.time() - now, result

      # Catch any exceptions that dont have a code (e.g. socket.error).
      except IOError as e:
        self.consecutive_connection_errors += 1
        # Request failed. If we connected successfully before we attempt a few
        # connections before we determine that it really failed. This might
        # happen if the front end is loaded and returns a few throttling 500
        # messages.
        if self.active_base_url is not None:
          # Propagate 406 immediately without retrying, as 406 is a valid
          # response that indicates a need for enrollment.
          response = getattr(e, "response", None)
          if getattr(response, "status_code", None) == 406:
            raise

          if self.consecutive_connection_errors >= self.retry_error_limit:
            # We tried several times but this really did not work, just fail it.
            logging.info(
                "Too many connection errors to %s, retrying another URL",
                self.active_base_url)
            self.active_base_url = None
            raise e

          # Back off hard to allow the front end to recover.
          logging.debug(
              "Unable to connect to frontend. Backing off %s seconds.",
              self.error_poll_min)
          self.Wait(self.error_poll_min)

        # We never previously connected, maybe the URL/proxy is wrong? Just fail
        # right away to allow callers to try a different URL.
        else:
          raise e

def iprotate(l, steps=1):
    r"""Like rotate, but modifies `l` in-place.

    >>> l = [1,2,3]
    >>> iprotate(l) is l
    True
    >>> l
    [2, 3, 1]
    >>> iprotate(iprotate(l, 2), -3)
    [1, 2, 3]

    """
    if len(l):
        steps %= len(l)
        if steps:
            firstPart = l[:steps]
            del l[:steps]
            l.extend(firstPart)
    return l

def enum_mark_last(iterable, start=0):
    """
    Returns a generator over iterable that tells whether the current item is the last one.
    Usage:
        >>> iterable = range(10)
        >>> for index, is_last, item in enum_mark_last(iterable):
        >>>     print(index, item, end='\n' if is_last else ', ')
    """
    it = iter(iterable)
    count = start
    try:
        last = next(it)
    except StopIteration:
        return
    for val in it:
        yield count, False, last
        last = val
        count += 1
    yield count, True, last

def rate_limited(max_per_hour: int, *args: Any) -> Callable[..., Any]:
    """Rate limit a function."""
    return util.rate_limited(max_per_hour, *args)

def memory_usage():
    """return memory usage of python process in MB

    from
    http://fa.bianp.net/blog/2013/different-ways-to-get-memory-consumption-or-lessons-learned-from-memory_profiler/
    psutil is quicker

    >>> isinstance(memory_usage(),float)
    True

    """
    try:
        import psutil
        import os
    except ImportError:
        return _memory_usage_ps()

    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem

def dotproduct(X, Y):
    """Return the sum of the element-wise product of vectors x and y.
    >>> dotproduct([1, 2, 3], [1000, 100, 10])
    1230
    """
    return sum([x * y for x, y in zip(X, Y)])

def rmglob(pattern: str) -> None:
    """
    Deletes all files whose filename matches the glob ``pattern`` (via
    :func:`glob.glob`).
    """
    for f in glob.glob(pattern):
        os.remove(f)

def duration_expired(start_time, duration_seconds):
    """
    Return True if ``duration_seconds`` have expired since ``start_time``
    """

    if duration_seconds is not None:
        delta_seconds = datetime_delta_to_seconds(dt.datetime.now() - start_time)

        if delta_seconds >= duration_seconds:
            return True

    return False

def mmap(func, iterable):
    """Wrapper to make map() behave the same on Py2 and Py3."""

    if sys.version_info[0] > 2:
        return [i for i in map(func, iterable)]
    else:
        return map(func, iterable)

def get_valid_filename(s):
    """
    Shamelessly taken from Django.
    https://github.com/django/django/blob/master/django/utils/text.py

    Return the given string converted to a string that can be used for a clean
    filename. Remove leading and trailing spaces; convert other spaces to
    underscores; and remove anything that is not an alphanumeric, dash,
    underscore, or dot.
    >>> get_valid_filename("john's portrait in 2004.jpg")
    'johns_portrait_in_2004.jpg'
    """
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)

async def enter_captcha(self, url: str, sid: str) -> str:
        """
        Override this method for processing captcha.

        :param url: link to captcha image
        :param sid: captcha id. I do not know why pass here but may be useful
        :return captcha value
        """
        raise VkCaptchaNeeded(url, sid)

def get_from_gnucash26_date(date_str: str) -> date:
    """ Creates a datetime from GnuCash 2.6 date string """
    date_format = "%Y%m%d"
    result = datetime.strptime(date_str, date_format).date()
    return result

def genfirstvalues(cursor: Cursor, arraysize: int = 1000) \
        -> Generator[Any, None, None]:
    """
    Generate the first value in each row.

    Args:
        cursor: the cursor
        arraysize: split fetches into chunks of this many records

    Yields:
        the first value of each row
    """
    return (row[0] for row in genrows(cursor, arraysize))

def list_to_str(lst):
    """
    Turn a list into a comma- and/or and-separated string.

    Parameters
    ----------
    lst : :obj:`list`
        A list of strings to join into a single string.

    Returns
    -------
    str_ : :obj:`str`
        A string with commas and/or ands separating th elements from ``lst``.

    """
    if len(lst) == 1:
        str_ = lst[0]
    elif len(lst) == 2:
        str_ = ' and '.join(lst)
    elif len(lst) > 2:
        str_ = ', '.join(lst[:-1])
        str_ += ', and {0}'.format(lst[-1])
    else:
        raise ValueError('List of length 0 provided.')
    return str_

def clean_column_names(df: DataFrame) -> DataFrame:
    """
    Strip the whitespace from all column names in the given DataFrame
    and return the result.
    """
    f = df.copy()
    f.columns = [col.strip() for col in f.columns]
    return f

def wait_for_shutdown_signal(
        self,
        please_stop=False,  # ASSIGN SIGNAL TO STOP EARLY
        allow_exit=False,  # ALLOW "exit" COMMAND ON CONSOLE TO ALSO STOP THE APP
        wait_forever=True  # IGNORE CHILD THREADS, NEVER EXIT.  False => IF NO CHILD THREADS LEFT, THEN EXIT
    ):
        """
        FOR USE BY PROCESSES THAT NEVER DIE UNLESS EXTERNAL SHUTDOWN IS REQUESTED

        CALLING THREAD WILL SLEEP UNTIL keyboard interrupt, OR please_stop, OR "exit"

        :param please_stop:
        :param allow_exit:
        :param wait_forever:: Assume all needed threads have been launched. When done
        :return:
        """
        self_thread = Thread.current()
        if self_thread != MAIN_THREAD or self_thread != self:
            Log.error("Only the main thread can sleep forever (waiting for KeyboardInterrupt)")

        if isinstance(please_stop, Signal):
            # MUTUAL SIGNALING MAKES THESE TWO EFFECTIVELY THE SAME SIGNAL
            self.please_stop.on_go(please_stop.go)
            please_stop.on_go(self.please_stop.go)
        else:
            please_stop = self.please_stop

        if not wait_forever:
            # TRIGGER SIGNAL WHEN ALL CHILDREN THEADS ARE DONE
            with self_thread.child_lock:
                pending = copy(self_thread.children)
            children_done = AndSignals(please_stop, len(pending))
            children_done.signal.on_go(self.please_stop.go)
            for p in pending:
                p.stopped.on_go(children_done.done)

        try:
            if allow_exit:
                _wait_for_exit(please_stop)
            else:
                _wait_for_interrupt(please_stop)
        except KeyboardInterrupt as _:
            Log.alert("SIGINT Detected!  Stopping...")
        except SystemExit as _:
            Log.alert("SIGTERM Detected!  Stopping...")
        finally:
            self.stop()

def __replace_all(repls: dict, input: str) -> str:
    """ Replaces from a string **input** all the occurrences of some
    symbols according to mapping **repls**.

    :param dict repls: where #key is the old character and
    #value is the one to substitute with;
    :param str input: original string where to apply the
    replacements;
    :return: *(str)* the string with the desired characters replaced
    """
    return re.sub('|'.join(re.escape(key) for key in repls.keys()),
                  lambda k: repls[k.group(0)], input)

def decode(self, bytes, raw=False):
        """decode(bytearray, raw=False) -> value

        Decodes the given bytearray according to this PrimitiveType
        definition.

        NOTE: The parameter ``raw`` is present to adhere to the
        ``decode()`` inteface, but has no effect for PrimitiveType
        definitions.
        """
        return struct.unpack(self.format, buffer(bytes))[0]

def _rindex(mylist: Sequence[T], x: T) -> int:
    """Index of the last occurrence of x in the sequence."""
    return len(mylist) - mylist[::-1].index(x) - 1

def non_increasing(values):
    """True if values are not increasing."""
    return all(x >= y for x, y in zip(values, values[1:]))

def string(value) -> str:
        """ string dict/object/value to JSON """
        return system_json.dumps(Json(value).safe_object(), ensure_ascii=False)

def __remove_trailing_zeros(self, collection):
        """Removes trailing zeroes from indexable collection of numbers"""
        index = len(collection) - 1
        while index >= 0 and collection[index] == 0:
            index -= 1

        return collection[:index + 1]

def get_column_names(engine: Engine, tablename: str) -> List[str]:
    """
    Get all the database column names for the specified table.
    """
    return [info.name for info in gen_columns_info(engine, tablename)]

def strids2ids(tokens: Iterable[str]) -> List[int]:
    """
    Returns sequence of integer ids given a sequence of string ids.

    :param tokens: List of integer tokens.
    :return: List of word ids.
    """
    return list(map(int, tokens))

def get_keys_of_max_n(dict_obj, n):
    """Returns the keys that maps to the top n max values in the given dict.

    Example:
    --------
    >>> dict_obj = {'a':2, 'b':1, 'c':5}
    >>> get_keys_of_max_n(dict_obj, 2)
    ['a', 'c']
    """
    return sorted([
        item[0]
        for item in sorted(
            dict_obj.items(), key=lambda item: item[1], reverse=True
        )[:n]
    ])

def get_timezone() -> Tuple[datetime.tzinfo, str]:
    """Discover the current time zone and it's standard string representation (for source{d})."""
    dt = get_datetime_now().astimezone()
    tzstr = dt.strftime("%z")
    tzstr = tzstr[:-2] + ":" + tzstr[-2:]
    return dt.tzinfo, tzstr

def __replace_all(repls: dict, input: str) -> str:
    """ Replaces from a string **input** all the occurrences of some
    symbols according to mapping **repls**.

    :param dict repls: where #key is the old character and
    #value is the one to substitute with;
    :param str input: original string where to apply the
    replacements;
    :return: *(str)* the string with the desired characters replaced
    """
    return re.sub('|'.join(re.escape(key) for key in repls.keys()),
                  lambda k: repls[k.group(0)], input)

def is_line_in_file(filename: str, line: str) -> bool:
    """
    Detects whether a line is present within a file.

    Args:
        filename: file to check
        line: line to search for (as an exact match)
    """
    assert "\n" not in line
    with open(filename, "r") as file:
        for fileline in file:
            if fileline == line:
                return True
        return False

def strings_to_integers(strings: Iterable[str]) -> Iterable[int]:
    """
    Convert a list of strings to a list of integers.

    :param strings: a list of string
    :return: a list of converted integers

    .. doctest::

        >>> strings_to_integers(['1', '1.0', '-0.2'])
        [1, 1, 0]
    """
    return strings_to_(strings, lambda x: int(float(x)))

def most_significant_bit(lst: np.ndarray) -> int:
    """
    A helper function that finds the position of the most significant bit in a 1darray of 1s and 0s,
    i.e. the first position where a 1 appears, reading left to right.

    :param lst: a 1d array of 0s and 1s with at least one 1
    :return: the first position in lst that a 1 appears
    """
    return np.argwhere(np.asarray(lst) == 1)[0][0]

def to_dict(self):
        """

        Returns: the tree item as a dictionary

        """
        if self.childCount() > 0:
            value = {}
            for index in range(self.childCount()):
                value.update(self.child(index).to_dict())
        else:
            value = self.value

        return {self.name: value}

def find_first_in_list(txt: str, str_list: [str]) -> int:  # type: ignore
    """
    Returns the index of the earliest occurence of an item from a list in a string

    Ex: find_first_in_list('foobar', ['bar', 'fin']) -> 3
    """
    start = len(txt) + 1
    for item in str_list:
        if start > txt.find(item) > -1:
            start = txt.find(item)
    return start if len(txt) + 1 > start > -1 else -1

def same_network(atree, btree) -> bool:
    """True if given trees share the same structure of powernodes,
    independently of (power)node names,
    and same edge topology between (power)nodes.

    """
    return same_hierarchy(atree, btree) and same_topology(atree, btree)

def iterate_items(dictish):
    """ Return a consistent (key, value) iterable on dict-like objects,
    including lists of tuple pairs.

    Example:

        >>> list(iterate_items({'a': 1}))
        [('a', 1)]
        >>> list(iterate_items([('a', 1), ('b', 2)]))
        [('a', 1), ('b', 2)]
    """
    if hasattr(dictish, 'iteritems'):
        return dictish.iteritems()
    if hasattr(dictish, 'items'):
        return dictish.items()
    return dictish

def most_frequent(lst):
    """
    Returns the item that appears most frequently in the given list.
    """
    lst = lst[:]
    highest_freq = 0
    most_freq = None

    for val in unique(lst):
        if lst.count(val) > highest_freq:
            most_freq = val
            highest_freq = lst.count(val)
            
    return most_freq

def _create_empty_array(self, frames, always_2d, dtype):
        """Create an empty array with appropriate shape."""
        import numpy as np
        if always_2d or self.channels > 1:
            shape = frames, self.channels
        else:
            shape = frames,
        return np.empty(shape, dtype, order='C')

def python_utc_datetime_to_sqlite_strftime_string(
        value: datetime.datetime) -> str:
    """
    Converts a Python datetime to a string literal compatible with SQLite,
    including the millisecond field.
    """
    millisec_str = str(round(value.microsecond / 1000)).zfill(3)
    return value.strftime("%Y-%m-%d %H:%M:%S") + "." + millisec_str

def debugTreePrint(node,pfx="->"):
  """Purely a debugging aid: Ascii-art picture of a tree descended from node"""
  print pfx,node.item
  for c in node.children:
    debugTreePrint(c,"  "+pfx)

def strtobytes(input, encoding):
    """Take a str and transform it into a byte array."""
    py_version = sys.version_info[0]
    if py_version >= 3:
        return _strtobytes_py3(input, encoding)
    return _strtobytes_py2(input, encoding)

def drop_post(self):
        """Remove .postXXXX postfix from version"""
        post_index = self.version.find('.post')
        if post_index >= 0:
            self.version = self.version[:post_index]

def safe_pow(base, exp):
    """safe version of pow"""
    if exp > MAX_EXPONENT:
        raise RuntimeError("Invalid exponent, max exponent is {}".format(MAX_EXPONENT))
    return base ** exp

def __remove_trailing_zeros(self, collection):
        """Removes trailing zeroes from indexable collection of numbers"""
        index = len(collection) - 1
        while index >= 0 and collection[index] == 0:
            index -= 1

        return collection[:index + 1]

def callable_validator(v: Any) -> AnyCallable:
    """
    Perform a simple check if the value is callable.

    Note: complete matching of argument type hints and return types is not performed
    """
    if callable(v):
        return v

    raise errors.CallableError(value=v)

def astensor(array: TensorLike) -> BKTensor:
    """Covert numpy array to tensorflow tensor"""
    tensor = tf.convert_to_tensor(value=array, dtype=CTYPE)
    return tensor

def is_finite(value: Any) -> bool:
    """Return true if a value is a finite number."""
    return isinstance(value, int) or (isinstance(value, float) and isfinite(value))

def dotproduct(X, Y):
    """Return the sum of the element-wise product of vectors x and y.
    >>> dotproduct([1, 2, 3], [1000, 100, 10])
    1230
    """
    return sum([x * y for x, y in zip(X, Y)])

def valid_date(x: str) -> bool:
    """
    Retrun ``True`` if ``x`` is a valid YYYYMMDD date;
    otherwise return ``False``.
    """
    try:
        if x != dt.datetime.strptime(x, DATE_FORMAT).strftime(DATE_FORMAT):
            raise ValueError
        return True
    except ValueError:
        return False

def mkdir(self, target_folder):
        """
        Create a folder on S3.

        Examples
        --------
            >>> s3utils.mkdir("path/to/my_folder")
            Making directory: path/to/my_folder
        """
        self.printv("Making directory: %s" % target_folder)
        self.k.key = re.sub(r"^/|/$", "", target_folder) + "/"
        self.k.set_contents_from_string('')
        self.k.close()

def fast_median(a):
    """Fast median operation for masked array using 50th-percentile
    """
    a = checkma(a)
    #return scoreatpercentile(a.compressed(), 50)
    if a.count() > 0:
        out = np.percentile(a.compressed(), 50)
    else:
        out = np.ma.masked
    return out

def check_key(self, key: str) -> bool:
        """
        Checks if key exists in datastore. True if yes, False if no.

        :param: SHA512 hash key

        :return: whether or key not exists in datastore
        """
        keys = self.get_keys()
        return key in keys

def remove_prefix(text, prefix):
	"""
	Remove the prefix from the text if it exists.

	>>> remove_prefix('underwhelming performance', 'underwhelming ')
	'performance'

	>>> remove_prefix('something special', 'sample')
	'something special'
	"""
	null, prefix, rest = text.rpartition(prefix)
	return rest

def local_machine_uuid():
    """Return local machine unique identifier.

    >>> uuid = local_machine_uuid()

    """

    result = subprocess.check_output(
        'hal-get-property --udi '
        '/org/freedesktop/Hal/devices/computer '
        '--key system.hardware.uuid'.split()
        ).strip()

    return uuid.UUID(hex=result)

def valid_file(path: str) -> bool:
    """
    Verifies that a string path actually exists and is a file

    :param path: The path to verify
    :return: **True** if path exist and is a file
    """
    path = Path(path).expanduser()
    log.debug("checking if %s is a valid file", path)
    return path.exists() and path.is_file()

def process_literal_param(self, value: Optional[List[int]],
                              dialect: Dialect) -> str:
        """Convert things on the way from Python to the database."""
        retval = self._intlist_to_dbstr(value)
        return retval

def find_first(pattern: str, path: str) -> str:
    """
    Finds first file in ``path`` whose filename matches ``pattern`` (via
    :func:`fnmatch.fnmatch`), or raises :exc:`IndexError`.
    """
    try:
        return find(pattern, path)[0]
    except IndexError:
        log.critical('''Couldn't find "{}" in "{}"''', pattern, path)
        raise

def is_any_type_set(sett: Set[Type]) -> bool:
    """
    Helper method to check if a set of types is the {AnyObject} singleton

    :param sett:
    :return:
    """
    return len(sett) == 1 and is_any_type(min(sett))

def _short_repr(obj):
  """Helper function returns a truncated repr() of an object."""
  stringified = pprint.saferepr(obj)
  if len(stringified) > 200:
    return '%s... (%d bytes)' % (stringified[:200], len(stringified))
  return stringified

def rmglob(pattern: str) -> None:
    """
    Deletes all files whose filename matches the glob ``pattern`` (via
    :func:`glob.glob`).
    """
    for f in glob.glob(pattern):
        os.remove(f)

def convert_bytes_to_ints(in_bytes, num):
    """Convert a byte array into an integer array. The number of bytes forming an integer
    is defined by num

    :param in_bytes: the input bytes
    :param num: the number of bytes per int
    :return the integer array"""
    dt = numpy.dtype('>i' + str(num))
    return numpy.frombuffer(in_bytes, dt)

def _create_empty_array(self, frames, always_2d, dtype):
        """Create an empty array with appropriate shape."""
        import numpy as np
        if always_2d or self.channels > 1:
            shape = frames, self.channels
        else:
            shape = frames,
        return np.empty(shape, dtype, order='C')

def union(cls, *sets):
        """
        >>> from utool.util_set import *  # NOQA
        """
        import utool as ut
        lists_ = ut.flatten([list(s) for s in sets])
        return cls(lists_)

def _cleanup(path: str) -> None:
    """Cleanup temporary directory."""
    if os.path.isdir(path):
        shutil.rmtree(path)

def is_not_null(df: DataFrame, col_name: str) -> bool:
    """
    Return ``True`` if the given DataFrame has a column of the given
    name (string), and there exists at least one non-NaN value in that
    column; return ``False`` otherwise.
    """
    if (
        isinstance(df, pd.DataFrame)
        and col_name in df.columns
        and df[col_name].notnull().any()
    ):
        return True
    else:
        return False

def __rmatmul__(self, other):
        """
        Matrix multiplication using binary `@` operator in Python>=3.5.
        """
        return self.T.dot(np.transpose(other)).T

def running_containers(name_filter: str) -> List[str]:
    """
    :raises docker.exceptions.APIError
    """
    return [container.short_id for container in
            docker_client.containers.list(filters={"name": name_filter})]

def file_lines(bblfile:str) -> iter:
    """Yield lines found in given file"""
    with open(bblfile) as fd:
        yield from (line.rstrip() for line in fd if line.rstrip())

def uppercase_chars(string: any) -> str:
        """Return all (and only) the uppercase chars in the given string."""
        return ''.join([c if c.isupper() else '' for c in str(string)])

def zfill(x, width):
    """zfill(x, width) -> string

    Pad a numeric string x with zeros on the left, to fill a field
    of the specified width.  The string x is never truncated.

    """
    if not isinstance(x, basestring):
        x = repr(x)
    return x.zfill(width)

def cli_run():
    """docstring for argparse"""
    parser = argparse.ArgumentParser(description='Stupidly simple code answers from StackOverflow')
    parser.add_argument('query', help="What's the problem ?", type=str, nargs='+')
    parser.add_argument('-t','--tags', help='semicolon separated tags -> python;lambda')
    args = parser.parse_args()
    main(args)

def snake_to_camel(s: str) -> str:
    """Convert string from snake case to camel case."""

    fragments = s.split('_')

    return fragments[0] + ''.join(x.title() for x in fragments[1:])

def valid_substitution(strlen, index):
    """
    skip performing substitutions that are outside the bounds of the string
    """
    values = index[0]
    return all([strlen > i for i in values])

def recall_score(y_true, y_pred, average='micro', suffix=False):
    """Compute the recall.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The best value is 1 and the worst value is 0.

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.

    Example:
        >>> from seqeval.metrics import recall_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> recall_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_true = len(true_entities)

    score = nb_correct / nb_true if nb_true > 0 else 0

    return score

def default_parser() -> argparse.ArgumentParser:
    """Create a parser for CLI arguments and options."""
    parser = argparse.ArgumentParser(
        prog=CONSOLE_SCRIPT,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    build_parser(parser)
    return parser

def pack_bits( longbits ):
    """Crunch a 64-bit int (8 bool bytes) into a bitfield."""
    byte = longbits & (0x0101010101010101)
    byte = (byte | (byte>>7)) & (0x0003000300030003)
    byte = (byte | (byte>>14)) & (0x0000000f0000000f)
    byte = (byte | (byte>>28)) & (0x00000000000000ff)
    return byte

def try_cast_int(s):
    """(str) -> int
    All the digits in a given string are concatenated and converted into a single number.
    """
    try:
        temp = re.findall('\d', str(s))
        temp = ''.join(temp)
        return int(temp)
    except:
        return s

def bfx(value, msb, lsb):
    """! @brief Extract a value from a bitfield."""
    mask = bitmask((msb, lsb))
    return (value & mask) >> lsb

def callable_validator(v: Any) -> AnyCallable:
    """
    Perform a simple check if the value is callable.

    Note: complete matching of argument type hints and return types is not performed
    """
    if callable(v):
        return v

    raise errors.CallableError(value=v)

def _find_conda():
    """Find the conda executable robustly across conda versions.

    Returns
    -------
    conda : str
        Path to the conda executable.

    Raises
    ------
    IOError
        If the executable cannot be found in either the CONDA_EXE environment
        variable or in the PATH.

    Notes
    -----
    In POSIX platforms in conda >= 4.4, conda can be set up as a bash function
    rather than an executable. (This is to enable the syntax
    ``conda activate env-name``.) In this case, the environment variable
    ``CONDA_EXE`` contains the path to the conda executable. In other cases,
    we use standard search for the appropriate name in the PATH.

    See https://github.com/airspeed-velocity/asv/issues/645 for more details.
    """
    if 'CONDA_EXE' in os.environ:
        conda = os.environ['CONDA_EXE']
    else:
        conda = util.which('conda')
    return conda

def inverted_dict(d):
    """Return a dict with swapped keys and values

    >>> inverted_dict({0: ('a', 'b'), 1: 'cd'}) == {'cd': 1, ('a', 'b'): 0}
    True
    """
    return dict((force_hashable(v), k) for (k, v) in viewitems(dict(d)))

def is_sqlatype_integer(coltype: Union[TypeEngine, VisitableType]) -> bool:
    """
    Is the SQLAlchemy column type an integer type?
    """
    coltype = _coltype_to_typeengine(coltype)
    return isinstance(coltype, sqltypes.Integer)

def flatten_list(l: List[list]) -> list:
    """ takes a list of lists, l and returns a flat list
    """
    return [v for inner_l in l for v in inner_l]

def truncate(value: Decimal, n_digits: int) -> Decimal:
    """Truncates a value to a number of decimals places"""
    return Decimal(math.trunc(value * (10 ** n_digits))) / (10 ** n_digits)

def Exit(msg, code=1):
    """Exit execution with return code and message
    :param msg: Message displayed prior to exit
    :param code: code returned upon exiting
    """
    print >> sys.stderr, msg
    sys.exit(code)

def cpu_count() -> int:
    """Returns the number of processors on this machine."""
    if multiprocessing is None:
        return 1
    try:
        return multiprocessing.cpu_count()
    except NotImplementedError:
        pass
    try:
        return os.sysconf("SC_NPROCESSORS_CONF")
    except (AttributeError, ValueError):
        pass
    gen_log.error("Could not detect number of processors; assuming 1")
    return 1

def flatten_list(x: List[Any]) -> List[Any]:
    """
    Converts a list of lists into a flat list.
    
    Args:
        x: list of lists 

    Returns:
        flat list
        
    As per
    http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python

    """  # noqa
    return [item for sublist in x for item in sublist]

def _gauss(mean: int, sigma: int) -> int:
        """
        Creates a variation from a base value

        Args:
            mean: base value
            sigma: gaussian sigma

        Returns: random value

        """
        return int(random.gauss(mean, sigma))

def add_mark_at(string, index, mark):
    """
    Add mark to the index-th character of the given string. Return the new string after applying change.
    Notice: index > 0
    """
    if index == -1:
        return string
    # Python can handle the case which index is out of range of given string
    return string[:index] + add_mark_char(string[index], mark) + string[index+1:]

def stretch(iterable, n=2):
    r"""Repeat each item in `iterable` `n` times.

    Example:

    >>> list(stretch(range(3), 2))
    [0, 0, 1, 1, 2, 2]
    """
    times = range(n)
    for item in iterable:
        for i in times: yield item

def is_up_to_date(outfile, basedatetime):
        # type: (AnyStr, datetime) -> bool
        """Return true if outfile exists and is no older than base datetime."""
        if os.path.exists(outfile):
            if os.path.getmtime(outfile) >= basedatetime:
                return True
        return False

def _check_whitespace(string):
    """
    Make sure thre is no whitespace in the given string. Will raise a
    ValueError if whitespace is detected
    """
    if string.count(' ') + string.count('\t') + string.count('\n') > 0:
        raise ValueError(INSTRUCTION_HAS_WHITESPACE)

def first_location_of_maximum(x):
    """
    Returns the first location of the maximum value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.argmax(x) / len(x) if len(x) > 0 else np.NaN

def _kbhit_unix() -> bool:
    """
    Under UNIX: is a keystroke available?
    """
    dr, dw, de = select.select([sys.stdin], [], [], 0)
    return dr != []

def get_last_weekday_in_month(year, month, weekday):
        """Get the last weekday in a given month. e.g:

        >>> # the last monday in Jan 2013
        >>> Calendar.get_last_weekday_in_month(2013, 1, MON)
        datetime.date(2013, 1, 28)
        """
        day = date(year, month, monthrange(year, month)[1])
        while True:
            if day.weekday() == weekday:
                break
            day = day - timedelta(days=1)
        return day

def camel_to_snake_case(string):
    """Converts 'string' presented in camel case to snake case.

    e.g.: CamelCase => snake_case
    """
    s = _1.sub(r'\1_\2', string)
    return _2.sub(r'\1_\2', s).lower()

def get_timezone() -> Tuple[datetime.tzinfo, str]:
    """Discover the current time zone and it's standard string representation (for source{d})."""
    dt = get_datetime_now().astimezone()
    tzstr = dt.strftime("%z")
    tzstr = tzstr[:-2] + ":" + tzstr[-2:]
    return dt.tzinfo, tzstr

def bulk_load_docs(es, docs):
    """Bulk load docs

    Args:
        es: elasticsearch handle
        docs: Iterator of doc objects - includes index_name
    """

    chunk_size = 200

    try:
        results = elasticsearch.helpers.bulk(es, docs, chunk_size=chunk_size)
        log.debug(f"Elasticsearch documents loaded: {results[0]}")

        # elasticsearch.helpers.parallel_bulk(es, terms, chunk_size=chunk_size, thread_count=4)
        if len(results[1]) > 0:
            log.error("Bulk load errors {}".format(results))
    except elasticsearch.ElasticsearchException as e:
        log.error("Indexing error: {}\n".format(e))

def inject_nulls(data: Mapping, field_names) -> dict:
    """Insert None as value for missing fields."""

    record = dict()

    for field in field_names:
        record[field] = data.get(field, None)

    return record

def _check_whitespace(string):
    """
    Make sure thre is no whitespace in the given string. Will raise a
    ValueError if whitespace is detected
    """
    if string.count(' ') + string.count('\t') + string.count('\n') > 0:
        raise ValueError(INSTRUCTION_HAS_WHITESPACE)

def string(value) -> str:
        """ string dict/object/value to JSON """
        return system_json.dumps(Json(value).safe_object(), ensure_ascii=False)

def do_quit(self, _: argparse.Namespace) -> bool:
        """Exit this application"""
        self._should_quit = True
        return self._STOP_AND_EXIT

def get_column_names(engine: Engine, tablename: str) -> List[str]:
    """
    Get all the database column names for the specified table.
    """
    return [info.name for info in gen_columns_info(engine, tablename)]

def SGT(self, a, b):
        """Signed greater-than comparison"""
        # http://gavwood.com/paper.pdf
        s0, s1 = to_signed(a), to_signed(b)
        return Operators.ITEBV(256, s0 > s1, 1, 0)

def get_keys_of_max_n(dict_obj, n):
    """Returns the keys that maps to the top n max values in the given dict.

    Example:
    --------
    >>> dict_obj = {'a':2, 'b':1, 'c':5}
    >>> get_keys_of_max_n(dict_obj, 2)
    ['a', 'c']
    """
    return sorted([
        item[0]
        for item in sorted(
            dict_obj.items(), key=lambda item: item[1], reverse=True
        )[:n]
    ])

def rank(tensor: BKTensor) -> int:
    """Return the number of dimensions of a tensor"""
    if isinstance(tensor, np.ndarray):
        return len(tensor.shape)

    return len(tensor[0].size())

def is_prime(n):
    """
    Check if n is a prime number
    """
    if n % 2 == 0 and n > 2:
        return False
    return all(n % i for i in range(3, int(math.sqrt(n)) + 1, 2))

def has_key(cls, *args):
        """
        Check whether flyweight object with specified key has already been created.

        Returns:
            bool: True if already created, False if not
        """
        key = args if len(args) > 1 else args[0]
        return key in cls._instances

def issubset(self, other):
        """
        Report whether another set contains this set.

        Example:
            >>> OrderedSet([1, 2, 3]).issubset({1, 2})
            False
            >>> OrderedSet([1, 2, 3]).issubset({1, 2, 3, 4})
            True
            >>> OrderedSet([1, 2, 3]).issubset({1, 4, 3, 5})
            False
        """
        if len(self) > len(other):  # Fast check for obvious cases
            return False
        return all(item in other for item in self)

def after_third_friday(day=None):
    """ check if day is after month's 3rd friday """
    day = day if day is not None else datetime.datetime.now()
    now = day.replace(day=1, hour=16, minute=0, second=0, microsecond=0)
    now += relativedelta.relativedelta(weeks=2, weekday=relativedelta.FR)
    return day > now

def _request(self, method: str, endpoint: str, params: dict = None, data: dict = None, headers: dict = None) -> dict:
        """HTTP request method of interface implementation."""

def _write_json(obj, path):  # type: (object, str) -> None
    """Writes a serializeable object as a JSON file"""
    with open(path, 'w') as f:
        json.dump(obj, f)

def flatten_list(x: List[Any]) -> List[Any]:
    """
    Converts a list of lists into a flat list.
    
    Args:
        x: list of lists 

    Returns:
        flat list
        
    As per
    http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python

    """  # noqa
    return [item for sublist in x for item in sublist]

def get_last_weekday_in_month(year, month, weekday):
        """Get the last weekday in a given month. e.g:

        >>> # the last monday in Jan 2013
        >>> Calendar.get_last_weekday_in_month(2013, 1, MON)
        datetime.date(2013, 1, 28)
        """
        day = date(year, month, monthrange(year, month)[1])
        while True:
            if day.weekday() == weekday:
                break
            day = day - timedelta(days=1)
        return day

def argmax(iterable, key=None, both=False):
    """
    >>> argmax([4,2,-5])
    0
    >>> argmax([4,2,-5], key=abs)
    2
    >>> argmax([4,2,-5], key=abs, both=True)
    (2, 5)
    """
    if key is not None:
        it = imap(key, iterable)
    else:
        it = iter(iterable)
    score, argmax = reduce(max, izip(it, count()))
    if both:
        return argmax, score
    return argmax

def get_system_flags() -> FrozenSet[Flag]:
    """Return the set of implemented system flags."""
    return frozenset({Seen, Recent, Deleted, Flagged, Answered, Draft})

def list_to_str(list, separator=','):
    """
    >>> list = [0, 0, 7]
    >>> list_to_str(list)
    '0,0,7'
    """
    list = [str(x) for x in list]
    return separator.join(list)

def to_bytes(data: Any) -> bytearray:
    """
    Convert anything to a ``bytearray``.
    
    See
    
    - http://stackoverflow.com/questions/7585435/best-way-to-convert-string-to-bytes-in-python-3
    - http://stackoverflow.com/questions/10459067/how-to-convert-my-bytearrayb-x9e-x18k-x9a-to-something-like-this-x9e-x1
    """  # noqa
    if isinstance(data, int):
        return bytearray([data])
    return bytearray(data, encoding='latin-1')

def getIndex(predicateFn: Callable[[T], bool], items: List[T]) -> int:
    """
    Finds the index of an item in list, which satisfies predicate
    :param predicateFn: predicate function to run on items of list
    :param items: list of tuples
    :return: first index for which predicate function returns True
    """
    try:
        return next(i for i, v in enumerate(items) if predicateFn(v))
    except StopIteration:
        return -1

def mmap(func, iterable):
    """Wrapper to make map() behave the same on Py2 and Py3."""

    if sys.version_info[0] > 2:
        return [i for i in map(func, iterable)]
    else:
        return map(func, iterable)

def Exit(msg, code=1):
    """Exit execution with return code and message
    :param msg: Message displayed prior to exit
    :param code: code returned upon exiting
    """
    print >> sys.stderr, msg
    sys.exit(code)

def hex_to_int(value):
    """
    Convert hex string like "\x0A\xE3" to 2787.
    """
    if version_info.major >= 3:
        return int.from_bytes(value, "big")
    return int(value.encode("hex"), 16)

def timestamp_with_tzinfo(dt):
    """
    Serialize a date/time value into an ISO8601 text representation
    adjusted (if needed) to UTC timezone.

    For instance:
    >>> serialize_date(datetime(2012, 4, 10, 22, 38, 20, 604391))
    '2012-04-10T22:38:20.604391Z'
    """
    utc = tzutc()

    if dt.tzinfo:
        dt = dt.astimezone(utc).replace(tzinfo=None)
    return dt.isoformat() + 'Z'

def is_unicode(string):
    """Validates that the object itself is some kinda string"""
    str_type = str(type(string))

    if str_type.find('str') > 0 or str_type.find('unicode') > 0:
        return True

    return False

async def fetchall(self) -> Iterable[sqlite3.Row]:
        """Fetch all remaining rows."""
        return await self._execute(self._cursor.fetchall)

def _duplicates(list_):
    """Return dict mapping item -> indices."""
    item_indices = {}
    for i, item in enumerate(list_):
        try:
            item_indices[item].append(i)
        except KeyError:  # First time seen
            item_indices[item] = [i]
    return item_indices

def list_to_str(list, separator=','):
    """
    >>> list = [0, 0, 7]
    >>> list_to_str(list)
    '0,0,7'
    """
    list = [str(x) for x in list]
    return separator.join(list)

def dictlist_wipe_key(dict_list: Iterable[Dict], key: str) -> None:
    """
    Process an iterable of dictionaries. For each dictionary ``d``, delete
    ``d[key]`` if it exists.
    """
    for d in dict_list:
        d.pop(key, None)

def replace_variables(self, source: str, variables: dict) -> str:
        """Replace {{variable-name}} with stored value."""
        try:
            replaced = re.sub(
                "{{(.*?)}}", lambda m: variables.get(m.group(1), ""), source
            )
        except TypeError:
            replaced = source
        return replaced

def is_empty_shape(sh: ShExJ.Shape) -> bool:
        """ Determine whether sh has any value """
        return sh.closed is None and sh.expression is None and sh.extra is None and \
            sh.semActs is None

def stretch(iterable, n=2):
    r"""Repeat each item in `iterable` `n` times.

    Example:

    >>> list(stretch(range(3), 2))
    [0, 0, 1, 1, 2, 2]
    """
    times = range(n)
    for item in iterable:
        for i in times: yield item

def is_unicode(string):
    """Validates that the object itself is some kinda string"""
    str_type = str(type(string))

    if str_type.find('str') > 0 or str_type.find('unicode') > 0:
        return True

    return False

def has_key(cls, *args):
        """
        Check whether flyweight object with specified key has already been created.

        Returns:
            bool: True if already created, False if not
        """
        key = args if len(args) > 1 else args[0]
        return key in cls._instances

def duplicates(coll):
    """Return the duplicated items in the given collection

    :param coll: a collection
    :returns: a list of the duplicated items in the collection

    >>> duplicates([1, 1, 2, 3, 3, 4, 1, 1])
    [1, 3]

    """
    return list(set(x for x in coll if coll.count(x) > 1))

def proper_round(n):
    """
    rounds float to closest int
    :rtype: int
    :param n: float
    """
    return int(n) + (n / abs(n)) * int(abs(n - int(n)) >= 0.5) if n != 0 else 0

def _dfs_cycle_detect(graph, node, path, visited_nodes):
    """
    search graph for cycle using DFS continuing from node
    path contains the list of visited nodes currently on the stack
    visited_nodes is the set of already visited nodes
    :param graph:
    :param node:
    :param path:
    :param visited_nodes:
    :return:
    """
    visited_nodes.add(node)
    for target in graph[node]:
        if target in path:
            # cycle found => return current path
            return path + [target]
        else:
            return _dfs_cycle_detect(graph, target, path + [target], visited_nodes)
    return None

def read_set_from_file(filename: str) -> Set[str]:
    """
    Extract a de-duped collection (set) of text from a file.
    Expected file format is one item per line.
    """
    collection = set()
    with open(filename, 'r') as file_:
        for line in file_:
            collection.add(line.rstrip())
    return collection

def __replace_all(repls: dict, input: str) -> str:
    """ Replaces from a string **input** all the occurrences of some
    symbols according to mapping **repls**.

    :param dict repls: where #key is the old character and
    #value is the one to substitute with;
    :param str input: original string where to apply the
    replacements;
    :return: *(str)* the string with the desired characters replaced
    """
    return re.sub('|'.join(re.escape(key) for key in repls.keys()),
                  lambda k: repls[k.group(0)], input)

def read_set_from_file(filename: str) -> Set[str]:
    """
    Extract a de-duped collection (set) of text from a file.
    Expected file format is one item per line.
    """
    collection = set()
    with open(filename, 'r') as file_:
        for line in file_:
            collection.add(line.rstrip())
    return collection

def file_exists(fname):
    """Check if a file exists and is non-empty.
    """
    try:
        return fname and os.path.exists(fname) and os.path.getsize(fname) > 0
    except OSError:
        return False

def do_quit(self, _: argparse.Namespace) -> bool:
        """Exit this application"""
        self._should_quit = True
        return self._STOP_AND_EXIT

def collect_static() -> bool:
    """
    Runs Django ``collectstatic`` command in silent mode.

    :return: always ``True``
    """
    from django.core.management import execute_from_command_line
    # from django.conf import settings
    # if not os.listdir(settings.STATIC_ROOT):
    wf('Collecting static files... ', False)
    execute_from_command_line(['./manage.py', 'collectstatic', '-c', '--noinput', '-v0'])
    wf('[+]\n')
    return True

def check_key(self, key: str) -> bool:
        """
        Checks if key exists in datastore. True if yes, False if no.

        :param: SHA512 hash key

        :return: whether or key not exists in datastore
        """
        keys = self.get_keys()
        return key in keys

def safe_pow(base, exp):
    """safe version of pow"""
    if exp > MAX_EXPONENT:
        raise RuntimeError("Invalid exponent, max exponent is {}".format(MAX_EXPONENT))
    return base ** exp

def lsr_pairwise_dense(comp_mat, alpha=0.0, initial_params=None):
    """Compute the LSR estimate of model parameters given dense data.

    This function implements the Luce Spectral Ranking inference algorithm
    [MG15]_ for dense pairwise-comparison data.

    The data is described by a pairwise-comparison matrix ``comp_mat`` such
    that ``comp_mat[i,j]`` contains the number of times that item ``i`` wins
    against item ``j``.

    In comparison to :func:`~choix.lsr_pairwise`, this function is particularly
    efficient for dense pairwise-comparison datasets (i.e., containing many
    comparisons for a large fraction of item pairs).

    The argument ``initial_params`` can be used to iteratively refine an
    existing parameter estimate (see the implementation of
    :func:`~choix.ilsr_pairwise` for an idea on how this works). If it is set
    to `None` (the default), the all-ones vector is used.

    The transition rates of the LSR Markov chain are initialized with
    ``alpha``. When ``alpha > 0``, this corresponds to a form of regularization
    (see :ref:`regularization` for details).

    Parameters
    ----------
    comp_mat : np.array
        2D square matrix describing the pairwise-comparison outcomes.
    alpha : float, optional
        Regularization parameter.
    initial_params : array_like, optional
        Parameters used to build the transition rates of the LSR Markov chain.

    Returns
    -------
    params : np.array
        An estimate of model parameters.
    """
    n_items = comp_mat.shape[0]
    ws, chain = _init_lsr(n_items, alpha, initial_params)
    denom = np.tile(ws, (n_items, 1))
    chain += comp_mat.T / (denom + denom.T)
    chain -= np.diag(chain.sum(axis=1))
    return log_transform(statdist(chain))

def product(*args, **kwargs):
    """ Yields all permutations with replacement:
        list(product("cat", repeat=2)) => 
        [("c", "c"), 
         ("c", "a"), 
         ("c", "t"), 
         ("a", "c"), 
         ("a", "a"), 
         ("a", "t"), 
         ("t", "c"), 
         ("t", "a"), 
         ("t", "t")]
    """
    p = [[]]
    for iterable in map(tuple, args) * kwargs.get("repeat", 1):
        p = [x + [y] for x in p for y in iterable]
    for p in p:
        yield tuple(p)

def _skip_section(self):
        """Skip a section"""
        self._last = self._f.readline()
        while len(self._last) > 0 and len(self._last[0].strip()) == 0:
            self._last = self._f.readline()

def long_substr(data):
    """Return the longest common substring in a list of strings.
    
    Credit: http://stackoverflow.com/questions/2892931/longest-common-substring-from-more-than-two-strings-python
    """
    substr = ''
    if len(data) > 1 and len(data[0]) > 0:
        for i in range(len(data[0])):
            for j in range(len(data[0])-i+1):
                if j > len(substr) and all(data[0][i:i+j] in x for x in data):
                    substr = data[0][i:i+j]
    elif len(data) == 1:
        substr = data[0]
    return substr

def _parse_date(string: str) -> datetime.date:
    """Parse an ISO format date (YYYY-mm-dd).

    >>> _parse_date('1990-01-02')
    datetime.date(1990, 1, 2)
    """
    return datetime.datetime.strptime(string, '%Y-%m-%d').date()

def read_flat(self):
        """
        Read a PNG file and decode it into flat row flat pixel format.

        Returns (*width*, *height*, *pixels*, *metadata*).

        May use excessive memory.

        `pixels` are returned in flat row flat pixel format.

        See also the :meth:`read` method which returns pixels in the
        more stream-friendly boxed row flat pixel format.
        """
        x, y, pixel, meta = self.read()
        arraycode = 'BH'[meta['bitdepth'] > 8]
        pixel = array(arraycode, itertools.chain(*pixel))
        return x, y, pixel, meta

def iter_lines(file_like: Iterable[str]) -> Generator[str, None, None]:
    """ Helper for iterating only nonempty lines without line breaks"""
    for line in file_like:
        line = line.rstrip('\r\n')
        if line:
            yield line

def is_sqlatype_string(coltype: Union[TypeEngine, VisitableType]) -> bool:
    """
    Is the SQLAlchemy column type a string type?
    """
    coltype = _coltype_to_typeengine(coltype)
    return isinstance(coltype, sqltypes.String)

def import_by_path(path: str) -> Callable:
    """Import a class or function given it's absolute path.

    Parameters
    ----------
    path:
      Path to object to import
    """

    module_path, _, class_name = path.rpartition('.')
    return getattr(import_module(module_path), class_name)

def find_first_in_list(txt: str, str_list: [str]) -> int:  # type: ignore
    """
    Returns the index of the earliest occurence of an item from a list in a string

    Ex: find_first_in_list('foobar', ['bar', 'fin']) -> 3
    """
    start = len(txt) + 1
    for item in str_list:
        if start > txt.find(item) > -1:
            start = txt.find(item)
    return start if len(txt) + 1 > start > -1 else -1

def clean_column_names(df: DataFrame) -> DataFrame:
    """
    Strip the whitespace from all column names in the given DataFrame
    and return the result.
    """
    f = df.copy()
    f.columns = [col.strip() for col in f.columns]
    return f

def sample_normal(mean, var, rng):
    """Sample from independent normal distributions

    Each element is an independent normal distribution.

    Parameters
    ----------
    mean : numpy.ndarray
      Means of the normal distribution. Shape --> (batch_num, sample_dim)
    var : numpy.ndarray
      Variance of the normal distribution. Shape --> (batch_num, sample_dim)
    rng : numpy.random.RandomState

    Returns
    -------
    ret : numpy.ndarray
       The sampling result. Shape --> (batch_num, sample_dim)
    """
    ret = numpy.sqrt(var) * rng.randn(*mean.shape) + mean
    return ret

async def cursor(self) -> Cursor:
        """Create an aiosqlite cursor wrapping a sqlite3 cursor object."""
        return Cursor(self, await self._execute(self._conn.cursor))

def is_line_in_file(filename: str, line: str) -> bool:
    """
    Detects whether a line is present within a file.

    Args:
        filename: file to check
        line: line to search for (as an exact match)
    """
    assert "\n" not in line
    with open(filename, "r") as file:
        for fileline in file:
            if fileline == line:
                return True
        return False

def decodebytes(input):
    """Decode base64 string to byte array."""
    py_version = sys.version_info[0]
    if py_version >= 3:
        return _decodebytes_py3(input)
    return _decodebytes_py2(input)

def most_significant_bit(lst: np.ndarray) -> int:
    """
    A helper function that finds the position of the most significant bit in a 1darray of 1s and 0s,
    i.e. the first position where a 1 appears, reading left to right.

    :param lst: a 1d array of 0s and 1s with at least one 1
    :return: the first position in lst that a 1 appears
    """
    return np.argwhere(np.asarray(lst) == 1)[0][0]

def strings_to_integers(strings: Iterable[str]) -> Iterable[int]:
    """
    Convert a list of strings to a list of integers.

    :param strings: a list of string
    :return: a list of converted integers

    .. doctest::

        >>> strings_to_integers(['1', '1.0', '-0.2'])
        [1, 1, 0]
    """
    return strings_to_(strings, lambda x: int(float(x)))

def ensure_list(iterable: Iterable[A]) -> List[A]:
    """
    An Iterable may be a list or a generator.
    This ensures we get a list without making an unnecessary copy.
    """
    if isinstance(iterable, list):
        return iterable
    else:
        return list(iterable)

def csv_to_numpy(string_like, dtype=None):  # type: (str) -> np.array
    """Convert a CSV object to a numpy array.

    Args:
        string_like (str): CSV string.
        dtype (dtype, optional):  Data type of the resulting array. If None, the dtypes will be determined by the
                                        contents of each column, individually. This argument can only be used to
                                        'upcast' the array.  For downcasting, use the .astype(t) method.
    Returns:
        (np.array): numpy array
    """
    stream = StringIO(string_like)
    return np.genfromtxt(stream, dtype=dtype, delimiter=',')

def lint(fmt='colorized'):
    """Run verbose PyLint on source. Optionally specify fmt=html for HTML output."""
    if fmt == 'html':
        outfile = 'pylint_report.html'
        local('pylint -f %s davies > %s || true' % (fmt, outfile))
        local('open %s' % outfile)
    else:
        local('pylint -f %s davies || true' % fmt)

def _protected_log(x1):
    """Closure of log for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)

def bfx(value, msb, lsb):
    """! @brief Extract a value from a bitfield."""
    mask = bitmask((msb, lsb))
    return (value & mask) >> lsb

def mmap(func, iterable):
    """Wrapper to make map() behave the same on Py2 and Py3."""

    if sys.version_info[0] > 2:
        return [i for i in map(func, iterable)]
    else:
        return map(func, iterable)

def url_concat(url, args):
    """Concatenate url and argument dictionary regardless of whether
    url has existing query parameters.

    >>> url_concat("http://example.com/foo?a=b", dict(c="d"))
    'http://example.com/foo?a=b&c=d'
    """
    if not args: return url
    if url[-1] not in ('?', '&'):
        url += '&' if ('?' in url) else '?'
    return url + urllib.urlencode(args)

def valid_file(path: str) -> bool:
    """
    Verifies that a string path actually exists and is a file

    :param path: The path to verify
    :return: **True** if path exist and is a file
    """
    path = Path(path).expanduser()
    log.debug("checking if %s is a valid file", path)
    return path.exists() and path.is_file()

def is_not_null(df: DataFrame, col_name: str) -> bool:
    """
    Return ``True`` if the given DataFrame has a column of the given
    name (string), and there exists at least one non-NaN value in that
    column; return ``False`` otherwise.
    """
    if (
        isinstance(df, pd.DataFrame)
        and col_name in df.columns
        and df[col_name].notnull().any()
    ):
        return True
    else:
        return False

def _rindex(mylist: Sequence[T], x: T) -> int:
    """Index of the last occurrence of x in the sequence."""
    return len(mylist) - mylist[::-1].index(x) - 1

def s3_get(url: str, temp_file: IO) -> None:
    """Pull a file directly from S3."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)

def find_first(pattern: str, path: str) -> str:
    """
    Finds first file in ``path`` whose filename matches ``pattern`` (via
    :func:`fnmatch.fnmatch`), or raises :exc:`IndexError`.
    """
    try:
        return find(pattern, path)[0]
    except IndexError:
        log.critical('''Couldn't find "{}" in "{}"''', pattern, path)
        raise

def truncate(value: Decimal, n_digits: int) -> Decimal:
    """Truncates a value to a number of decimals places"""
    return Decimal(math.trunc(value * (10 ** n_digits))) / (10 ** n_digits)

def load_preprocess_images(image_paths: List[str], image_size: tuple) -> List[np.ndarray]:
    """
    Load and pre-process the images specified with absolute paths.

    :param image_paths: List of images specified with paths.
    :param image_size: Tuple to resize the image to (Channels, Height, Width)
    :return: A list of loaded images (numpy arrays).
    """
    image_size = image_size[1:]  # we do not need the number of channels
    images = []
    for image_path in image_paths:
        images.append(load_preprocess_image(image_path, image_size))
    return images

def list_to_str(list, separator=','):
    """
    >>> list = [0, 0, 7]
    >>> list_to_str(list)
    '0,0,7'
    """
    list = [str(x) for x in list]
    return separator.join(list)

def bytes_hack(buf):
    """
    Hacky workaround for old installs of the library on systems without python-future that were
    keeping the 2to3 update from working after auto-update.
    """
    ub = None
    if sys.version_info > (3,):
        ub = buf
    else:
        ub = bytes(buf)

    return ub

def is_line_in_file(filename: str, line: str) -> bool:
    """
    Detects whether a line is present within a file.

    Args:
        filename: file to check
        line: line to search for (as an exact match)
    """
    assert "\n" not in line
    with open(filename, "r") as file:
        for fileline in file:
            if fileline == line:
                return True
        return False

def samefile(a: str, b: str) -> bool:
    """Check if two pathes represent the same file."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normpath(a) == os.path.normpath(b)

def _parse_tuple_string(argument):
        """ Return a tuple from parsing 'a,b,c,d' -> (a,b,c,d) """
        if isinstance(argument, str):
            return tuple(int(p.strip()) for p in argument.split(','))
        return argument

def _rindex(mylist: Sequence[T], x: T) -> int:
    """Index of the last occurrence of x in the sequence."""
    return len(mylist) - mylist[::-1].index(x) - 1

def are_token_parallel(sequences: Sequence[Sized]) -> bool:
    """
    Returns True if all sequences in the list have the same length.
    """
    if not sequences or len(sequences) == 1:
        return True
    return all(len(s) == len(sequences[0]) for s in sequences)

def local_machine_uuid():
    """Return local machine unique identifier.

    >>> uuid = local_machine_uuid()

    """

    result = subprocess.check_output(
        'hal-get-property --udi '
        '/org/freedesktop/Hal/devices/computer '
        '--key system.hardware.uuid'.split()
        ).strip()

    return uuid.UUID(hex=result)

def url_host(url: str) -> str:
    """
    Parses hostname from URL.
    :param url: URL
    :return: hostname
    """
    from urllib.parse import urlparse
    res = urlparse(url)
    return res.netloc.split(':')[0] if res.netloc else ''

def _check_update_(self):
        """Check if the current version of the library is outdated."""
        try:
            data = requests.get("https://pypi.python.org/pypi/jira/json", timeout=2.001).json()

            released_version = data['info']['version']
            if parse_version(released_version) > parse_version(__version__):
                warnings.warn(
                    "You are running an outdated version of JIRA Python %s. Current version is %s. Do not file any bugs against older versions." % (
                        __version__, released_version))
        except requests.RequestException:
            pass
        except Exception as e:
            logging.warning(e)

def last_location_of_minimum(x):
    """
    Returns the last location of the minimal value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    return 1.0 - np.argmin(x[::-1]) / len(x) if len(x) > 0 else np.NaN

def __gt__(self, other):
        """Test for greater than."""
        if isinstance(other, Address):
            return str(self) > str(other)
        raise TypeError

def last(self):
        """Last time step available.

        Example:
            >>> sdat = StagyyData('path/to/run')
            >>> assert(sdat.steps.last is sdat.steps[-1])
        """
        if self._last is UNDETERMINED:
            # not necessarily the last one...
            self._last = self.sdat.tseries.index[-1]
        return self[self._last]

def cli_run():
    """docstring for argparse"""
    parser = argparse.ArgumentParser(description='Stupidly simple code answers from StackOverflow')
    parser.add_argument('query', help="What's the problem ?", type=str, nargs='+')
    parser.add_argument('-t','--tags', help='semicolon separated tags -> python;lambda')
    args = parser.parse_args()
    main(args)

def _get_or_default(mylist, i, default=None):
    """return list item number, or default if don't exist"""
    if i >= len(mylist):
        return default
    else :
        return mylist[i]

def cli_run():
    """docstring for argparse"""
    parser = argparse.ArgumentParser(description='Stupidly simple code answers from StackOverflow')
    parser.add_argument('query', help="What's the problem ?", type=str, nargs='+')
    parser.add_argument('-t','--tags', help='semicolon separated tags -> python;lambda')
    args = parser.parse_args()
    main(args)

def right_replace(string, old, new, count=1):
    """
    Right replaces ``count`` occurrences of ``old`` with ``new`` in ``string``.
    For example::

        right_replace('one_two_two', 'two', 'three') -> 'one_two_three'
    """
    if not string:
        return string
    return new.join(string.rsplit(old, count))

def gcd_float(numbers, tol=1e-8):
    """
    Returns the greatest common divisor for a sequence of numbers.
    Uses a numerical tolerance, so can be used on floats

    Args:
        numbers: Sequence of numbers.
        tol: Numerical tolerance

    Returns:
        (int) Greatest common divisor of numbers.
    """

    def pair_gcd_tol(a, b):
        """Calculate the Greatest Common Divisor of a and b.

        Unless b==0, the result will have the same sign as b (so that when
        b is divided by it, the result comes out positive).
        """
        while b > tol:
            a, b = b, a % b
        return a

    n = numbers[0]
    for i in numbers:
        n = pair_gcd_tol(n, i)
    return n

def de_duplicate(items):
    """Remove any duplicate item, preserving order

    >>> de_duplicate([1, 2, 1, 2])
    [1, 2]
    """
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return result

def __replace_all(repls: dict, input: str) -> str:
    """ Replaces from a string **input** all the occurrences of some
    symbols according to mapping **repls**.

    :param dict repls: where #key is the old character and
    #value is the one to substitute with;
    :param str input: original string where to apply the
    replacements;
    :return: *(str)* the string with the desired characters replaced
    """
    return re.sub('|'.join(re.escape(key) for key in repls.keys()),
                  lambda k: repls[k.group(0)], input)

def version():
    """Wrapper for opj_version library routine."""
    OPENJPEG.opj_version.restype = ctypes.c_char_p
    library_version = OPENJPEG.opj_version()
    if sys.hexversion >= 0x03000000:
        return library_version.decode('utf-8')
    else:
        return library_version

def uppercase_chars(string: any) -> str:
        """Return all (and only) the uppercase chars in the given string."""
        return ''.join([c if c.isupper() else '' for c in str(string)])

def get_now_sql_datetime():
    """
    *A datetime stamp in MySQL format: ``YYYY-MM-DDTHH:MM:SS``*

    **Return:**
        - ``now`` -- current time and date in MySQL format

    **Usage:**
        .. code-block:: python 

            from fundamentals import times
            now = times.get_now_sql_datetime()
            print now

            # OUT: 2016-03-18T11:08:23 
    """
    ## > IMPORTS ##
    from datetime import datetime, date, time
    now = datetime.now()
    now = now.strftime("%Y-%m-%dT%H:%M:%S")

    return now

def assert_equal(first, second, msg_fmt="{msg}"):
    """Fail unless first equals second, as determined by the '==' operator.

    >>> assert_equal(5, 5.0)
    >>> assert_equal("Hello World!", "Goodbye!")
    Traceback (most recent call last):
        ...
    AssertionError: 'Hello World!' != 'Goodbye!'

    The following msg_fmt arguments are supported:
    * msg - the default error message
    * first - the first argument
    * second - the second argument
    """

    if isinstance(first, dict) and isinstance(second, dict):
        assert_dict_equal(first, second, msg_fmt)
    elif not first == second:
        msg = "{!r} != {!r}".format(first, second)
        fail(msg_fmt.format(msg=msg, first=first, second=second))

def right_replace(string, old, new, count=1):
    """
    Right replaces ``count`` occurrences of ``old`` with ``new`` in ``string``.
    For example::

        right_replace('one_two_two', 'two', 'three') -> 'one_two_three'
    """
    if not string:
        return string
    return new.join(string.rsplit(old, count))

def is_builtin_object(node: astroid.node_classes.NodeNG) -> bool:
    """Returns True if the given node is an object from the __builtin__ module."""
    return node and node.root().name == BUILTINS_NAME

def iprotate(l, steps=1):
    r"""Like rotate, but modifies `l` in-place.

    >>> l = [1,2,3]
    >>> iprotate(l) is l
    True
    >>> l
    [2, 3, 1]
    >>> iprotate(iprotate(l, 2), -3)
    [1, 2, 3]

    """
    if len(l):
        steps %= len(l)
        if steps:
            firstPart = l[:steps]
            del l[:steps]
            l.extend(firstPart)
    return l

def bfx(value, msb, lsb):
    """! @brief Extract a value from a bitfield."""
    mask = bitmask((msb, lsb))
    return (value & mask) >> lsb

def get_domain(url):
    """
    Get domain part of an url.

    For example: https://www.python.org/doc/ -> https://www.python.org
    """
    parse_result = urlparse(url)
    domain = "{schema}://{netloc}".format(
        schema=parse_result.scheme, netloc=parse_result.netloc)
    return domain

def capitalize(string):
    """Capitalize a sentence.

    Parameters
    ----------
    string : `str`
        String to capitalize.

    Returns
    -------
    `str`
        Capitalized string.

    Examples
    --------
    >>> capitalize('worD WORD WoRd')
    'Word word word'
    """
    if not string:
        return string
    if len(string) == 1:
        return string.upper()
    return string[0].upper() + string[1:].lower()

def min(self):
        """
        :returns the minimum of the column
        """
        res = self._qexec("min(%s)" % self._name)
        if len(res) > 0:
            self._min = res[0][0]
        return self._min

def get_last_weekday_in_month(year, month, weekday):
        """Get the last weekday in a given month. e.g:

        >>> # the last monday in Jan 2013
        >>> Calendar.get_last_weekday_in_month(2013, 1, MON)
        datetime.date(2013, 1, 28)
        """
        day = date(year, month, monthrange(year, month)[1])
        while True:
            if day.weekday() == weekday:
                break
            day = day - timedelta(days=1)
        return day

def checksum(path):
    """Calculcate checksum for a file."""
    hasher = hashlib.sha1()
    with open(path, 'rb') as stream:
        buf = stream.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = stream.read(BLOCKSIZE)
    return hasher.hexdigest()

def fib(n):
    """Fibonacci example function

    Args:
      n (int): integer

    Returns:
      int: n-th Fibonacci number
    """
    assert n > 0
    a, b = 1, 1
    for i in range(n - 1):
        a, b = b, a + b
    return a

def get_days_in_month(year: int, month: int) -> int:
    """ Returns number of days in the given month.
    1-based numbers as arguments. i.e. November = 11 """
    month_range = calendar.monthrange(year, month)
    return month_range[1]

def has_key(cls, *args):
        """
        Check whether flyweight object with specified key has already been created.

        Returns:
            bool: True if already created, False if not
        """
        key = args if len(args) > 1 else args[0]
        return key in cls._instances

def get_last_weekday_in_month(year, month, weekday):
        """Get the last weekday in a given month. e.g:

        >>> # the last monday in Jan 2013
        >>> Calendar.get_last_weekday_in_month(2013, 1, MON)
        datetime.date(2013, 1, 28)
        """
        day = date(year, month, monthrange(year, month)[1])
        while True:
            if day.weekday() == weekday:
                break
            day = day - timedelta(days=1)
        return day

def string(value) -> str:
        """ string dict/object/value to JSON """
        return system_json.dumps(Json(value).safe_object(), ensure_ascii=False)

def pairwise(iterable):
    """From itertools cookbook. [a, b, c, ...] -> (a, b), (b, c), ..."""
    first, second = tee(iterable)
    next(second, None)
    return zip(first, second)

def to_graphviz(graph):
    """

    :param graph:
    :return:
    """
    ret = ['digraph g {']
    vertices = []

    node_ids = dict([(name, 'node' + idx) for (idx, name) in enumerate(list(graph))])

    for node in list(graph):
        ret.append('  "%s" [label="%s"];' % (node_ids[node], node))
        for target in graph[node]:
            vertices.append('  "%s" -> "%s";' % (node_ids[node], node_ids[target]))

    ret += vertices
    ret.append('}')
    return '\n'.join(ret)

def file_or_stdin() -> Callable:
    """
    Returns a file descriptor from stdin or opening a file from a given path.
    """

    def parse(path):
        if path is None or path == "-":
            return sys.stdin
        else:
            return data_io.smart_open(path)

    return parse

def get_file_extension(filename):
    """ Return the extension if the filename has it. None if not.

    :param filename: The filename.
    :return: Extension or None.
    """
    filename_x = filename.split('.')
    if len(filename_x) > 1:
        if filename_x[-1].strip() is not '':
            return filename_x[-1]
    return None

def truncate(value: Decimal, n_digits: int) -> Decimal:
    """Truncates a value to a number of decimals places"""
    return Decimal(math.trunc(value * (10 ** n_digits))) / (10 ** n_digits)

def list_to_str(lst):
    """
    Turn a list into a comma- and/or and-separated string.

    Parameters
    ----------
    lst : :obj:`list`
        A list of strings to join into a single string.

    Returns
    -------
    str_ : :obj:`str`
        A string with commas and/or ands separating th elements from ``lst``.

    """
    if len(lst) == 1:
        str_ = lst[0]
    elif len(lst) == 2:
        str_ = ' and '.join(lst)
    elif len(lst) > 2:
        str_ = ', '.join(lst[:-1])
        str_ += ', and {0}'.format(lst[-1])
    else:
        raise ValueError('List of length 0 provided.')
    return str_

def normalize_column_names(df):
    r""" Clean up whitespace in column names. See better version at `pugnlp.clean_columns`

    >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=['Hello World', 'not here'])
    >>> normalize_column_names(df)
    ['hello_world', 'not_here']
    """
    columns = df.columns if hasattr(df, 'columns') else df
    columns = [c.lower().replace(' ', '_') for c in columns]
    return columns

def _width_is_big_enough(image, width):
    """Check that the image width is superior to `width`"""
    if width > image.size[0]:
        raise ImageSizeError(image.size[0], width)

def bulk_load_docs(es, docs):
    """Bulk load docs

    Args:
        es: elasticsearch handle
        docs: Iterator of doc objects - includes index_name
    """

    chunk_size = 200

    try:
        results = elasticsearch.helpers.bulk(es, docs, chunk_size=chunk_size)
        log.debug(f"Elasticsearch documents loaded: {results[0]}")

        # elasticsearch.helpers.parallel_bulk(es, terms, chunk_size=chunk_size, thread_count=4)
        if len(results[1]) > 0:
            log.error("Bulk load errors {}".format(results))
    except elasticsearch.ElasticsearchException as e:
        log.error("Indexing error: {}\n".format(e))

def read_text_from_file(path: str) -> str:
    """ Reads text file contents """
    with open(path) as text_file:
        content = text_file.read()

    return content

def detect_model_num(string):
    """Takes a string related to a model name and extract its model number.

    For example:
        '000000-bootstrap.index' => 0
    """
    match = re.match(MODEL_NUM_REGEX, string)
    if match:
        return int(match.group())
    return None

def normcdf(x, log=False):
    """Normal cumulative density function."""
    y = np.atleast_1d(x).copy()
    flib.normcdf(y)
    if log:
        if (y>0).all():
            return np.log(y)
        return -np.inf
    return y

def find_first(pattern: str, path: str) -> str:
    """
    Finds first file in ``path`` whose filename matches ``pattern`` (via
    :func:`fnmatch.fnmatch`), or raises :exc:`IndexError`.
    """
    try:
        return find(pattern, path)[0]
    except IndexError:
        log.critical('''Couldn't find "{}" in "{}"''', pattern, path)
        raise

def _sum_cycles_from_tokens(self, tokens: List[str]) -> int:
        """Sum the total number of cycles over a list of tokens."""
        return sum((int(self._nonnumber_pattern.sub('', t)) for t in tokens))

def cmd_dot(conf: Config):
    """Print out a neat targets dependency tree based on requested targets.

    Use graphviz to render the dot file, e.g.:

    > ybt dot :foo :bar | dot -Tpng -o graph.png
    """
    build_context = BuildContext(conf)
    populate_targets_graph(build_context, conf)
    if conf.output_dot_file is None:
        write_dot(build_context, conf, sys.stdout)
    else:
        with open(conf.output_dot_file, 'w') as out_file:
            write_dot(build_context, conf, out_file)

def _mid(pt1, pt2):
    """
    (Point, Point) -> Point
    Return the point that lies in between the two input points.
    """
    (x0, y0), (x1, y1) = pt1, pt2
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)

def _prm_get_longest_stringsize(string_list):
        """ Returns the longest string size for a string entry across data."""
        maxlength = 1

        for stringar in string_list:
            if isinstance(stringar, np.ndarray):
                if stringar.ndim > 0:
                    for string in stringar.ravel():
                        maxlength = max(len(string), maxlength)
                else:
                    maxlength = max(len(stringar.tolist()), maxlength)
            else:
                maxlength = max(len(stringar), maxlength)

        # Make the string Col longer than needed in order to allow later on slightly larger strings
        return int(maxlength * 1.5)

def str_to_time(time_str: str) -> datetime.datetime:
    """
    Convert human readable string to datetime.datetime.
    """
    pieces: Any = [int(piece) for piece in time_str.split('-')]
    return datetime.datetime(*pieces)

def de_duplicate(items):
    """Remove any duplicate item, preserving order

    >>> de_duplicate([1, 2, 1, 2])
    [1, 2]
    """
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return result

def long_substr(data):
    """Return the longest common substring in a list of strings.
    
    Credit: http://stackoverflow.com/questions/2892931/longest-common-substring-from-more-than-two-strings-python
    """
    substr = ''
    if len(data) > 1 and len(data[0]) > 0:
        for i in range(len(data[0])):
            for j in range(len(data[0])-i+1):
                if j > len(substr) and all(data[0][i:i+j] in x for x in data):
                    substr = data[0][i:i+j]
    elif len(data) == 1:
        substr = data[0]
    return substr

def get_default_bucket_key(buckets: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Returns the default bucket from a list of buckets, i.e. the largest bucket.

    :param buckets: List of buckets.
    :return: The largest bucket in the list.
    """
    return max(buckets)

def __next__(self):
        """
        :return: int
        """
        self.current += 1
        if self.current > self.total:
            raise StopIteration
        else:
            return self.iterable[self.current - 1]

def recClearTag(element):
    """Applies maspy.xml.clearTag() to the tag attribute of the "element" and
    recursively to all child elements.

    :param element: an :instance:`xml.etree.Element`
    """
    children = element.getchildren()
    if len(children) > 0:
        for child in children:
            recClearTag(child)
    element.tag = clearTag(element.tag)

def dictlist_wipe_key(dict_list: Iterable[Dict], key: str) -> None:
    """
    Process an iterable of dictionaries. For each dictionary ``d``, delete
    ``d[key]`` if it exists.
    """
    for d in dict_list:
        d.pop(key, None)

def __remove_trailing_zeros(self, collection):
        """Removes trailing zeroes from indexable collection of numbers"""
        index = len(collection) - 1
        while index >= 0 and collection[index] == 0:
            index -= 1

        return collection[:index + 1]

def is_sqlatype_integer(coltype: Union[TypeEngine, VisitableType]) -> bool:
    """
    Is the SQLAlchemy column type an integer type?
    """
    coltype = _coltype_to_typeengine(coltype)
    return isinstance(coltype, sqltypes.Integer)

def get_from_gnucash26_date(date_str: str) -> date:
    """ Creates a datetime from GnuCash 2.6 date string """
    date_format = "%Y%m%d"
    result = datetime.strptime(date_str, date_format).date()
    return result

def dict_to_enum_fn(d: Dict[str, Any], enum_class: Type[Enum]) -> Enum:
    """
    Converts an ``dict`` to a ``Enum``.
    """
    return enum_class[d['name']]

def is_sqlatype_integer(coltype: Union[TypeEngine, VisitableType]) -> bool:
    """
    Is the SQLAlchemy column type an integer type?
    """
    coltype = _coltype_to_typeengine(coltype)
    return isinstance(coltype, sqltypes.Integer)

def clean_map(obj: Mapping[Any, Any]) -> Mapping[Any, Any]:
    """
    Return a new copied dictionary without the keys with ``None`` values from
    the given Mapping object.
    """
    return {k: v for k, v in obj.items() if v is not None}

def valid_date(x: str) -> bool:
    """
    Retrun ``True`` if ``x`` is a valid YYYYMMDD date;
    otherwise return ``False``.
    """
    try:
        if x != dt.datetime.strptime(x, DATE_FORMAT).strftime(DATE_FORMAT):
            raise ValueError
        return True
    except ValueError:
        return False

def branches():
    # type: () -> List[str]
    """ Return a list of branches in the current repo.

    Returns:
        list[str]: A list of branches in the current repo.
    """
    out = shell.run(
        'git branch',
        capture=True,
        never_pretend=True
    ).stdout.strip()
    return [x.strip('* \t\n') for x in out.splitlines()]

def cache_page(page_cache, page_hash, cache_size):
    """Add a page to the page cache."""
    page_cache.append(page_hash)
    if len(page_cache) > cache_size:
        page_cache.pop(0)

def butlast(iterable):
    """Yield all items from ``iterable`` except the last one.

    >>> list(butlast(['spam', 'eggs', 'ham']))
    ['spam', 'eggs']

    >>> list(butlast(['spam']))
    []

    >>> list(butlast([]))
    []
    """
    iterable = iter(iterable)
    try:
        first = next(iterable)
    except StopIteration:
        return
    for second in iterable:
        yield first
        first = second

def _log_response(response):
    """Log out information about a ``Request`` object.

    After calling ``requests.request`` or one of its convenience methods, the
    object returned can be passed to this method. If done, information about
    the object returned is logged.

    :return: Nothing is returned.

    """
    message = u'Received HTTP {0} response: {1}'.format(
        response.status_code,
        response.text
    )
    if response.status_code >= 400:  # pragma: no cover
        logger.warning(message)
    else:
        logger.debug(message)

def empty_wav(wav_path: Union[Path, str]) -> bool:
    """Check if a wav contains data"""
    with wave.open(str(wav_path), 'rb') as wav_f:
        return wav_f.getnframes() == 0

def _in_qtconsole() -> bool:
    """
    A small utility function which determines if we're running in QTConsole's context.
    """
    try:
        from IPython import get_ipython
        try:
            from ipykernel.zmqshell import ZMQInteractiveShell
            shell_object = ZMQInteractiveShell
        except ImportError:
            from IPython.kernel.zmq import zmqshell
            shell_object = zmqshell.ZMQInteractiveShell
        return isinstance(get_ipython(), shell_object)
    except Exception:
        return False

def get_last_weekday_in_month(year, month, weekday):
        """Get the last weekday in a given month. e.g:

        >>> # the last monday in Jan 2013
        >>> Calendar.get_last_weekday_in_month(2013, 1, MON)
        datetime.date(2013, 1, 28)
        """
        day = date(year, month, monthrange(year, month)[1])
        while True:
            if day.weekday() == weekday:
                break
            day = day - timedelta(days=1)
        return day

def read(self, start_position: int, size: int) -> memoryview:
        """
        Return a view into the memory
        """
        return memoryview(self._bytes)[start_position:start_position + size]

def auto_up(self, count=1, go_to_start_of_line_if_history_changes=False):
        """
        If we're not on the first line (of a multiline input) go a line up,
        otherwise go back in history. (If nothing is selected.)
        """
        if self.complete_state:
            self.complete_previous(count=count)
        elif self.document.cursor_position_row > 0:
            self.cursor_up(count=count)
        elif not self.selection_state:
            self.history_backward(count=count)

            # Go to the start of the line?
            if go_to_start_of_line_if_history_changes:
                self.cursor_position += self.document.get_start_of_line_position()

def set_cell_value(cell, value):
    """
    Convenience method for setting the value of an openpyxl cell

    This is necessary since the value property changed from internal_value
    to value between version 1.* and 2.*.
    """
    if OPENPYXL_MAJOR_VERSION > 1:
        cell.value = value
    else:
        cell.internal_value = value

def writable_stream(handle):
    """Test whether a stream can be written to.
    """
    if isinstance(handle, io.IOBase) and sys.version_info >= (3, 5):
        return handle.writable()
    try:
        handle.write(b'')
    except (io.UnsupportedOperation, IOError):
        return False
    else:
        return True

def sorted(self):
        """Utility function for sort_file_tabs_alphabetically()."""
        for i in range(0, self.tabs.tabBar().count() - 1):
            if (self.tabs.tabBar().tabText(i) >
                    self.tabs.tabBar().tabText(i + 1)):
                return False
        return True

def ask_bool(question: str, default: bool = True) -> bool:
    """Asks a question yes no style"""
    default_q = "Y/n" if default else "y/N"
    answer = input("{0} [{1}]: ".format(question, default_q))
    lower = answer.lower()
    if not lower:
        return default
    return lower == "y"

def ensure_list(iterable: Iterable[A]) -> List[A]:
    """
    An Iterable may be a list or a generator.
    This ensures we get a list without making an unnecessary copy.
    """
    if isinstance(iterable, list):
        return iterable
    else:
        return list(iterable)

def isfile_notempty(inputfile: str) -> bool:
        """Check if the input filename with path is a file and is not empty."""
        try:
            return isfile(inputfile) and getsize(inputfile) > 0
        except TypeError:
            raise TypeError('inputfile is not a valid type')

def _gauss(mean: int, sigma: int) -> int:
        """
        Creates a variation from a base value

        Args:
            mean: base value
            sigma: gaussian sigma

        Returns: random value

        """
        return int(random.gauss(mean, sigma))

def stretch(iterable, n=2):
    r"""Repeat each item in `iterable` `n` times.

    Example:

    >>> list(stretch(range(3), 2))
    [0, 0, 1, 1, 2, 2]
    """
    times = range(n)
    for item in iterable:
        for i in times: yield item

def has_value(cls, value: int) -> bool:
        """True if specified value exists in int enum; otherwise, False."""
        return any(value == item.value for item in cls)

def issubset(self, other):
        """
        Report whether another set contains this set.

        Example:
            >>> OrderedSet([1, 2, 3]).issubset({1, 2})
            False
            >>> OrderedSet([1, 2, 3]).issubset({1, 2, 3, 4})
            True
            >>> OrderedSet([1, 2, 3]).issubset({1, 4, 3, 5})
            False
        """
        if len(self) > len(other):  # Fast check for obvious cases
            return False
        return all(item in other for item in self)

def parse_dim(features, check=True):
    """Return the features dimension, raise if error

    Raise IOError if features have not all the same positive
    dimension.  Return dim (int), the features dimension.

    """
    # try:
    dim = features[0].shape[1]
    # except IndexError:
    #     dim = 1

    if check and not dim > 0:
        raise IOError('features dimension must be strictly positive')
    if check and not all([d == dim for d in [x.shape[1] for x in features]]):
        raise IOError('all files must have the same feature dimension')
    return dim

def _kbhit_unix() -> bool:
    """
    Under UNIX: is a keystroke available?
    """
    dr, dw, de = select.select([sys.stdin], [], [], 0)
    return dr != []

def camel_to_snake_case(string):
    """Converts 'string' presented in camel case to snake case.

    e.g.: CamelCase => snake_case
    """
    s = _1.sub(r'\1_\2', string)
    return _2.sub(r'\1_\2', s).lower()

def write_text(filename: str, text: str) -> None:
    """
    Writes text to a file.
    """
    with open(filename, 'w') as f:  # type: TextIO
        print(text, file=f)

def fmt_camel(name):
    """
    Converts name to lower camel case. Words are identified by capitalization,
    dashes, and underscores.
    """
    words = split_words(name)
    assert len(words) > 0
    first = words.pop(0).lower()
    return first + ''.join([word.capitalize() for word in words])

def get_default_bucket_key(buckets: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Returns the default bucket from a list of buckets, i.e. the largest bucket.

    :param buckets: List of buckets.
    :return: The largest bucket in the list.
    """
    return max(buckets)

def _brief_print_list(lst, limit=7):
    """Print at most `limit` elements of list."""
    lst = list(lst)
    if len(lst) > limit:
        return _brief_print_list(lst[:limit//2], limit) + ', ..., ' + \
            _brief_print_list(lst[-limit//2:], limit)
    return ', '.join(["'%s'"%str(i) for i in lst])

def _isint(string):
    """
    >>> _isint("123")
    True
    >>> _isint("123.45")
    False
    """
    return type(string) is int or \
           (isinstance(string, _binary_type) or isinstance(string, _text_type)) and \
           _isconvertible(int, string)

def position(self) -> Position:
        """The current position of the cursor."""
        return Position(self._index, self._lineno, self._col_offset)

def get_edge_relations(graph: BELGraph) -> Mapping[Tuple[BaseEntity, BaseEntity], Set[str]]:
    """Build a dictionary of {node pair: set of edge types}."""
    return group_dict_set(
        ((u, v), d[RELATION])
        for u, v, d in graph.edges(data=True)
    )

def flatten_list(x: List[Any]) -> List[Any]:
    """
    Converts a list of lists into a flat list.
    
    Args:
        x: list of lists 

    Returns:
        flat list
        
    As per
    http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python

    """  # noqa
    return [item for sublist in x for item in sublist]

def arcball_map_to_sphere(point, center, radius):
    """Return unit sphere coordinates from window coordinates."""
    v0 = (point[0] - center[0]) / radius
    v1 = (center[1] - point[1]) / radius
    n = v0*v0 + v1*v1
    if n > 1.0:
        # position outside of sphere
        n = math.sqrt(n)
        return numpy.array([v0/n, v1/n, 0.0])
    else:
        return numpy.array([v0, v1, math.sqrt(1.0 - n)])

def file_exists(self) -> bool:
        """ Check if the settings file exists or not """
        cfg_path = self.file_path
        assert cfg_path

        return path.isfile(cfg_path)

def local_machine_uuid():
    """Return local machine unique identifier.

    >>> uuid = local_machine_uuid()

    """

    result = subprocess.check_output(
        'hal-get-property --udi '
        '/org/freedesktop/Hal/devices/computer '
        '--key system.hardware.uuid'.split()
        ).strip()

    return uuid.UUID(hex=result)

def is_prime(n):
    """
    Check if n is a prime number
    """
    if n % 2 == 0 and n > 2:
        return False
    return all(n % i for i in range(3, int(math.sqrt(n)) + 1, 2))

def __setitem__(self, *args, **kwargs):
        """ Cut if needed. """
        super(History, self).__setitem__(*args, **kwargs)
        if len(self) > self.size:
            self.popitem(False)

def truncate(value: Decimal, n_digits: int) -> Decimal:
    """Truncates a value to a number of decimals places"""
    return Decimal(math.trunc(value * (10 ** n_digits))) / (10 ** n_digits)

def split(text: str) -> List[str]:
    """Split a text into a list of tokens.

    :param text: the text to split
    :return: tokens
    """
    return [word for word in SEPARATOR.split(text) if word.strip(' \t')]

def isarray(array, test, dim=2):
    """Returns True if test is True for all array elements.
    Otherwise, returns False.
    """
    if dim > 1:
        return all(isarray(array[i], test, dim - 1)
                   for i in range(len(array)))
    return all(test(i) for i in array)

def debugTreePrint(node,pfx="->"):
  """Purely a debugging aid: Ascii-art picture of a tree descended from node"""
  print pfx,node.item
  for c in node.children:
    debugTreePrint(c,"  "+pfx)

def list_adb_devices_by_usb_id():
    """List the usb id of all android devices connected to the computer that
    are detected by adb.

    Returns:
        A list of strings that are android device usb ids. Empty if there's
        none.
    """
    out = adb.AdbProxy().devices(['-l'])
    clean_lines = new_str(out, 'utf-8').strip().split('\n')
    results = []
    for line in clean_lines:
        tokens = line.strip().split()
        if len(tokens) > 2 and tokens[1] == 'device':
            results.append(tokens[2])
    return results

def snake_case(a_string):
    """Returns a snake cased version of a string.

    :param a_string: any :class:`str` object.

    Usage:
        >>> snake_case('FooBar')
        "foo_bar"
    """

    partial = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', a_string)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', partial).lower()

def has_key(cls, *args):
        """
        Check whether flyweight object with specified key has already been created.

        Returns:
            bool: True if already created, False if not
        """
        key = args if len(args) > 1 else args[0]
        return key in cls._instances

def decode(self, bytes, raw=False):
        """decode(bytearray, raw=False) -> value

        Decodes the given bytearray according to this PrimitiveType
        definition.

        NOTE: The parameter ``raw`` is present to adhere to the
        ``decode()`` inteface, but has no effect for PrimitiveType
        definitions.
        """
        return struct.unpack(self.format, buffer(bytes))[0]

def returns(self) -> T.Optional[DocstringReturns]:
        """Return return information indicated in docstring."""
        try:
            return next(
                DocstringReturns.from_meta(meta)
                for meta in self.meta
                if meta.args[0] in {"return", "returns", "yield", "yields"}
            )
        except StopIteration:
            return None

def dict_of_sets_add(dictionary, key, value):
    # type: (DictUpperBound, Any, Any) -> None
    """Add value to a set in a dictionary by key

    Args:
        dictionary (DictUpperBound): Dictionary to which to add values
        key (Any): Key within dictionary
        value (Any): Value to add to set in dictionary

    Returns:
        None

    """
    set_objs = dictionary.get(key, set())
    set_objs.add(value)
    dictionary[key] = set_objs

def getIndex(predicateFn: Callable[[T], bool], items: List[T]) -> int:
    """
    Finds the index of an item in list, which satisfies predicate
    :param predicateFn: predicate function to run on items of list
    :param items: list of tuples
    :return: first index for which predicate function returns True
    """
    try:
        return next(i for i, v in enumerate(items) if predicateFn(v))
    except StopIteration:
        return -1

def running_containers(name_filter: str) -> List[str]:
    """
    :raises docker.exceptions.APIError
    """
    return [container.short_id for container in
            docker_client.containers.list(filters={"name": name_filter})]

def getIndex(predicateFn: Callable[[T], bool], items: List[T]) -> int:
    """
    Finds the index of an item in list, which satisfies predicate
    :param predicateFn: predicate function to run on items of list
    :param items: list of tuples
    :return: first index for which predicate function returns True
    """
    try:
        return next(i for i, v in enumerate(items) if predicateFn(v))
    except StopIteration:
        return -1

def rank(tensor: BKTensor) -> int:
    """Return the number of dimensions of a tensor"""
    if isinstance(tensor, np.ndarray):
        return len(tensor.shape)

    return len(tensor[0].size())

def get_keys_of_max_n(dict_obj, n):
    """Returns the keys that maps to the top n max values in the given dict.

    Example:
    --------
    >>> dict_obj = {'a':2, 'b':1, 'c':5}
    >>> get_keys_of_max_n(dict_obj, 2)
    ['a', 'c']
    """
    return sorted([
        item[0]
        for item in sorted(
            dict_obj.items(), key=lambda item: item[1], reverse=True
        )[:n]
    ])

def is_finite(value: Any) -> bool:
    """Return true if a value is a finite number."""
    return isinstance(value, int) or (isinstance(value, float) and isfinite(value))

def butlast(iterable):
    """Yield all items from ``iterable`` except the last one.

    >>> list(butlast(['spam', 'eggs', 'ham']))
    ['spam', 'eggs']

    >>> list(butlast(['spam']))
    []

    >>> list(butlast([]))
    []
    """
    iterable = iter(iterable)
    try:
        first = next(iterable)
    except StopIteration:
        return
    for second in iterable:
        yield first
        first = second

def constant(times: np.ndarray, amp: complex) -> np.ndarray:
    """Continuous constant pulse.

    Args:
        times: Times to output pulse for.
        amp: Complex pulse amplitude.
    """
    return np.full(len(times), amp, dtype=np.complex_)

def normcdf(x, log=False):
    """Normal cumulative density function."""
    y = np.atleast_1d(x).copy()
    flib.normcdf(y)
    if log:
        if (y>0).all():
            return np.log(y)
        return -np.inf
    return y

def replaceStrs(s, *args):
    r"""Replace all ``(frm, to)`` tuples in `args` in string `s`.

    >>> replaceStrs("nothing is better than warm beer",
    ...             ('nothing','warm beer'), ('warm beer','nothing'))
    'warm beer is better than nothing'

    """
    if args == (): return s
    mapping = dict((frm, to) for frm, to in args)
    return re.sub("|".join(map(re.escape, mapping.keys())),
                  lambda match:mapping[match.group(0)], s)

def is_unicode(string):
    """Validates that the object itself is some kinda string"""
    str_type = str(type(string))

    if str_type.find('str') > 0 or str_type.find('unicode') > 0:
        return True

    return False

def bfx(value, msb, lsb):
    """! @brief Extract a value from a bitfield."""
    mask = bitmask((msb, lsb))
    return (value & mask) >> lsb

def _(f, x):
    """
    filter for dict, note `f` should have signature: `f::key->value->bool`
    """
    return {k: v for k, v in x.items() if f(k, v)}

def is_unicode(string):
    """Validates that the object itself is some kinda string"""
    str_type = str(type(string))

    if str_type.find('str') > 0 or str_type.find('unicode') > 0:
        return True

    return False

def _protected_log(x1):
    """Closure of log for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)

def file_exists(fname):
    """Check if a file exists and is non-empty.
    """
    try:
        return fname and os.path.exists(fname) and os.path.getsize(fname) > 0
    except OSError:
        return False

def check_max_filesize(chosen_file, max_size):
    """
    Checks file sizes for host
    """
    if os.path.getsize(chosen_file) > max_size:
        return False
    else:
        return True

def pretty_describe(object, nestedness=0, indent=2):
    """Maintain dict ordering - but make string version prettier"""
    if not isinstance(object, dict):
        return str(object)
    sep = f'\n{" " * nestedness * indent}'
    out = sep.join((f'{k}: {pretty_describe(v, nestedness + 1)}' for k, v in object.items()))
    if nestedness > 0 and out:
        return f'{sep}{out}'
    return out

def indent(text: str, num: int = 2) -> str:
    """Indent a piece of text."""
    lines = text.splitlines()
    return "\n".join(indent_iterable(lines, num=num))

def is_any_type_set(sett: Set[Type]) -> bool:
    """
    Helper method to check if a set of types is the {AnyObject} singleton

    :param sett:
    :return:
    """
    return len(sett) == 1 and is_any_type(min(sett))

def extend(a: dict, b: dict) -> dict:
    """Merge two dicts and return a new dict. Much like subclassing works."""
    res = a.copy()
    res.update(b)
    return res

def str_to_time(time_str: str) -> datetime.datetime:
    """
    Convert human readable string to datetime.datetime.
    """
    pieces: Any = [int(piece) for piece in time_str.split('-')]
    return datetime.datetime(*pieces)

def _read_section(self):
        """Read and return an entire section"""
        lines = [self._last[self._last.find(":")+1:]]
        self._last = self._f.readline()
        while len(self._last) > 0 and len(self._last[0].strip()) == 0:
            lines.append(self._last)
            self._last = self._f.readline()
        return lines

def timeit(func, *args, **kwargs):
    """
    Time execution of function. Returns (res, seconds).

    >>> res, timing = timeit(time.sleep, 1)
    """
    start_time = time.time()
    res = func(*args, **kwargs)
    timing = time.time() - start_time
    return res, timing

def isarray(array, test, dim=2):
    """Returns True if test is True for all array elements.
    Otherwise, returns False.
    """
    if dim > 1:
        return all(isarray(array[i], test, dim - 1)
                   for i in range(len(array)))
    return all(test(i) for i in array)

def dict_to_enum_fn(d: Dict[str, Any], enum_class: Type[Enum]) -> Enum:
    """
    Converts an ``dict`` to a ``Enum``.
    """
    return enum_class[d['name']]

def decode(self, bytes, raw=False):
        """decode(bytearray, raw=False) -> value

        Decodes the given bytearray according to this PrimitiveType
        definition.

        NOTE: The parameter ``raw`` is present to adhere to the
        ``decode()`` inteface, but has no effect for PrimitiveType
        definitions.
        """
        return struct.unpack(self.format, buffer(bytes))[0]

def valid_file(path: str) -> bool:
    """
    Verifies that a string path actually exists and is a file

    :param path: The path to verify
    :return: **True** if path exist and is a file
    """
    path = Path(path).expanduser()
    log.debug("checking if %s is a valid file", path)
    return path.exists() and path.is_file()

def _my_hash(arg_list):
    # type: (List[Any]) -> int
    """Simple helper hash function"""
    res = 0
    for arg in arg_list:
        res = res * 31 + hash(arg)
    return res

def decode_base64(data: str) -> bytes:
    """Decode base64, padding being optional.

    :param data: Base64 data as an ASCII byte string
    :returns: The decoded byte string.
    """
    missing_padding = len(data) % 4
    if missing_padding != 0:
        data += "=" * (4 - missing_padding)
    return base64.decodebytes(data.encode("utf-8"))

def lowercase_chars(string: any) -> str:
        """Return all (and only) the lowercase chars in the given string."""
        return ''.join([c if c.islower() else '' for c in str(string)])

def file_exists(fname):
    """Check if a file exists and is non-empty.
    """
    try:
        return fname and os.path.exists(fname) and os.path.getsize(fname) > 0
    except OSError:
        return False

def get_view_selection(self):
        """Get actual tree selection object and all respective models of selected rows"""
        if not self.MODEL_STORAGE_ID:
            return None, None

        # avoid selection requests on empty tree views -> case warnings in gtk3
        if len(self.store) == 0:
            paths = []
        else:
            model, paths = self._tree_selection.get_selected_rows()

        # get all related models for selection from respective tree store field
        selected_model_list = []
        for path in paths:
            model = self.store[path][self.MODEL_STORAGE_ID]
            selected_model_list.append(model)
        return self._tree_selection, selected_model_list

def public(self) -> 'PrettyDir':
        """Returns public attributes of the inspected object."""
        return PrettyDir(
            self.obj, [pattr for pattr in self.pattrs if not pattr.name.startswith('_')]
        )

async def executemany(self, sql: str, parameters: Iterable[Iterable[Any]]) -> None:
        """Execute the given multiquery."""
        await self._execute(self._cursor.executemany, sql, parameters)

def iter_lines(file_like: Iterable[str]) -> Generator[str, None, None]:
    """ Helper for iterating only nonempty lines without line breaks"""
    for line in file_like:
        line = line.rstrip('\r\n')
        if line:
            yield line

def output_dir(self, *args) -> str:
        """ Directory where to store output """
        return os.path.join(self.project_dir, 'output', *args)

def astensor(array: TensorLike) -> BKTensor:
    """Covert numpy array to tensorflow tensor"""
    tensor = tf.convert_to_tensor(value=array, dtype=CTYPE)
    return tensor

def get_pylint_options(config_dir='.'):
    # type: (str) -> List[str]
    """Checks for local config overrides for `pylint`
    and add them in the correct `pylint` `options` format.

    :param config_dir:
    :return: List [str]
    """
    if PYLINT_CONFIG_NAME in os.listdir(config_dir):
        pylint_config_path = PYLINT_CONFIG_NAME
    else:
        pylint_config_path = DEFAULT_PYLINT_CONFIG_PATH

    return ['--rcfile={}'.format(pylint_config_path)]

def string(value) -> str:
        """ string dict/object/value to JSON """
        return system_json.dumps(Json(value).safe_object(), ensure_ascii=False)

def _check_limit(self):
        """Intenal method: check if current cache size exceeds maximum cache
           size and pop the oldest item in this case"""

        # First compress
        self._compress()

        # Then check the max size
        if len(self._store) >= self._max_size:
            self._store.popitem(last=False)

def get_default_bucket_key(buckets: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Returns the default bucket from a list of buckets, i.e. the largest bucket.

    :param buckets: List of buckets.
    :return: The largest bucket in the list.
    """
    return max(buckets)

def codes_get_size(handle, key):
    # type: (cffi.FFI.CData, str) -> int
    """
    Get the number of coded value from a key.
    If several keys of the same name are present, the total sum is returned.

    :param bytes key: the keyword to get the size of

    :rtype: int
    """
    size = ffi.new('size_t *')
    _codes_get_size(handle, key.encode(ENC), size)
    return size[0]

def try_cast_int(s):
    """(str) -> int
    All the digits in a given string are concatenated and converted into a single number.
    """
    try:
        temp = re.findall('\d', str(s))
        temp = ''.join(temp)
        return int(temp)
    except:
        return s

def remove_namespaces(root):
    """Call this on an lxml.etree document to remove all namespaces"""
    for elem in root.getiterator():
        if not hasattr(elem.tag, 'find'):
            continue

        i = elem.tag.find('}')
        if i >= 0:
            elem.tag = elem.tag[i + 1:]

    objectify.deannotate(root, cleanup_namespaces=True)

def remove_blank_lines(string):
    """ Removes all blank lines in @string

        -> #str without blank lines
    """
    return "\n".join(line
                     for line in string.split("\n")
                     if len(line.strip()))

def _brief_print_list(lst, limit=7):
    """Print at most `limit` elements of list."""
    lst = list(lst)
    if len(lst) > limit:
        return _brief_print_list(lst[:limit//2], limit) + ', ..., ' + \
            _brief_print_list(lst[-limit//2:], limit)
    return ', '.join(["'%s'"%str(i) for i in lst])

def update(self, iterable):
        """
        Return a new PSet with elements in iterable added

        >>> s1 = s(1, 2)
        >>> s1.update([3, 4, 4])
        pset([1, 2, 3, 4])
        """
        e = self.evolver()
        for element in iterable:
            e.add(element)

        return e.persistent()

def is_line_in_file(filename: str, line: str) -> bool:
    """
    Detects whether a line is present within a file.

    Args:
        filename: file to check
        line: line to search for (as an exact match)
    """
    assert "\n" not in line
    with open(filename, "r") as file:
        for fileline in file:
            if fileline == line:
                return True
        return False

def convert_bytes_to_ints(in_bytes, num):
    """Convert a byte array into an integer array. The number of bytes forming an integer
    is defined by num

    :param in_bytes: the input bytes
    :param num: the number of bytes per int
    :return the integer array"""
    dt = numpy.dtype('>i' + str(num))
    return numpy.frombuffer(in_bytes, dt)

def zfill(x, width):
    """zfill(x, width) -> string

    Pad a numeric string x with zeros on the left, to fill a field
    of the specified width.  The string x is never truncated.

    """
    if not isinstance(x, basestring):
        x = repr(x)
    return x.zfill(width)

def list_to_str(lst):
    """
    Turn a list into a comma- and/or and-separated string.

    Parameters
    ----------
    lst : :obj:`list`
        A list of strings to join into a single string.

    Returns
    -------
    str_ : :obj:`str`
        A string with commas and/or ands separating th elements from ``lst``.

    """
    if len(lst) == 1:
        str_ = lst[0]
    elif len(lst) == 2:
        str_ = ' and '.join(lst)
    elif len(lst) > 2:
        str_ = ', '.join(lst[:-1])
        str_ += ', and {0}'.format(lst[-1])
    else:
        raise ValueError('List of length 0 provided.')
    return str_

def csv_to_numpy(string_like, dtype=None):  # type: (str) -> np.array
    """Convert a CSV object to a numpy array.

    Args:
        string_like (str): CSV string.
        dtype (dtype, optional):  Data type of the resulting array. If None, the dtypes will be determined by the
                                        contents of each column, individually. This argument can only be used to
                                        'upcast' the array.  For downcasting, use the .astype(t) method.
    Returns:
        (np.array): numpy array
    """
    stream = StringIO(string_like)
    return np.genfromtxt(stream, dtype=dtype, delimiter=',')

def genfirstvalues(cursor: Cursor, arraysize: int = 1000) \
        -> Generator[Any, None, None]:
    """
    Generate the first value in each row.

    Args:
        cursor: the cursor
        arraysize: split fetches into chunks of this many records

    Yields:
        the first value of each row
    """
    return (row[0] for row in genrows(cursor, arraysize))

def first_digits(s, default=0):
    """Return the fist (left-hand) digits in a string as a single integer, ignoring sign (+/-).
    >>> first_digits('+123.456')
    123
    """
    s = re.split(r'[^0-9]+', str(s).strip().lstrip('+-' + charlist.whitespace))
    if len(s) and len(s[0]):
        return int(s[0])
    return default

def uppercase_chars(string: any) -> str:
        """Return all (and only) the uppercase chars in the given string."""
        return ''.join([c if c.isupper() else '' for c in str(string)])

def _exit(self, status_code):
        """Properly kill Python process including zombie threads."""
        # If there are active threads still running infinite loops, sys.exit
        # won't kill them but os._exit will. os._exit skips calling cleanup
        # handlers, flushing stdio buffers, etc.
        exit_func = os._exit if threading.active_count() > 1 else sys.exit
        exit_func(status_code)

def is_unicode(string):
    """Validates that the object itself is some kinda string"""
    str_type = str(type(string))

    if str_type.find('str') > 0 or str_type.find('unicode') > 0:
        return True

    return False

def dict_to_enum_fn(d: Dict[str, Any], enum_class: Type[Enum]) -> Enum:
    """
    Converts an ``dict`` to a ``Enum``.
    """
    return enum_class[d['name']]

def _relative_frequency(self, word):
		"""Computes the log relative frequency for a word form"""

		count = self.type_counts.get(word, 0)
		return math.log(count/len(self.type_counts)) if count > 0 else 0

def replace_variables(self, source: str, variables: dict) -> str:
        """Replace {{variable-name}} with stored value."""
        try:
            replaced = re.sub(
                "{{(.*?)}}", lambda m: variables.get(m.group(1), ""), source
            )
        except TypeError:
            replaced = source
        return replaced

def _duplicates(list_):
    """Return dict mapping item -> indices."""
    item_indices = {}
    for i, item in enumerate(list_):
        try:
            item_indices[item].append(i)
        except KeyError:  # First time seen
            item_indices[item] = [i]
    return item_indices

def update_kwargs(kwargs, **keyvalues):
    """Update dict with keys and values if keys do not already exist.

    >>> kwargs = {'one': 1, }
    >>> update_kwargs(kwargs, one=None, two=2)
    >>> kwargs == {'one': 1, 'two': 2}
    True

    """
    for key, value in keyvalues.items():
        if key not in kwargs:
            kwargs[key] = value

async def executemany(self, sql: str, parameters: Iterable[Iterable[Any]]) -> None:
        """Execute the given multiquery."""
        await self._execute(self._cursor.executemany, sql, parameters)

def _skip_section(self):
        """Skip a section"""
        self._last = self._f.readline()
        while len(self._last) > 0 and len(self._last[0].strip()) == 0:
            self._last = self._f.readline()

def samefile(a: str, b: str) -> bool:
    """Check if two pathes represent the same file."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normpath(a) == os.path.normpath(b)

def file_exists(fname):
    """Check if a file exists and is non-empty.
    """
    try:
        return fname and os.path.exists(fname) and os.path.getsize(fname) > 0
    except OSError:
        return False

async def parallel_results(future_map: Sequence[Tuple]) -> Dict:
    """
    Run parallel execution of futures and return mapping of their results to the provided keys.
    Just a neat shortcut around ``asyncio.gather()``

    :param future_map: Keys to futures mapping, e.g.: ( ('nav', get_nav()), ('content, get_content()) )
    :return: Dict with futures results mapped to keys {'nav': {1:2}, 'content': 'xyz'}
    """
    ctx_methods = OrderedDict(future_map)
    fs = list(ctx_methods.values())
    results = await asyncio.gather(*fs)
    results = {
        key: results[idx] for idx, key in enumerate(ctx_methods.keys())
    }
    return results

def split(text: str) -> List[str]:
    """Split a text into a list of tokens.

    :param text: the text to split
    :return: tokens
    """
    return [word for word in SEPARATOR.split(text) if word.strip(' \t')]

def strtobytes(input, encoding):
    """Take a str and transform it into a byte array."""
    py_version = sys.version_info[0]
    if py_version >= 3:
        return _strtobytes_py3(input, encoding)
    return _strtobytes_py2(input, encoding)

def has_synset(word: str) -> list:
    """" Returns a list of synsets of a word after lemmatization. """

    return wn.synsets(lemmatize(word, neverstem=True))

def read_text_from_file(path: str) -> str:
    """ Reads text file contents """
    with open(path) as text_file:
        content = text_file.read()

    return content

def timeit(func, *args, **kwargs):
    """
    Time execution of function. Returns (res, seconds).

    >>> res, timing = timeit(time.sleep, 1)
    """
    start_time = time.time()
    res = func(*args, **kwargs)
    timing = time.time() - start_time
    return res, timing

def head(self) -> Any:
        """Retrive first element in List."""

        lambda_list = self._get_value()
        return lambda_list(lambda head, _: head)

def get_from_gnucash26_date(date_str: str) -> date:
    """ Creates a datetime from GnuCash 2.6 date string """
    date_format = "%Y%m%d"
    result = datetime.strptime(date_str, date_format).date()
    return result

def has_value(cls, value: int) -> bool:
        """True if specified value exists in int enum; otherwise, False."""
        return any(value == item.value for item in cls)

def _str_to_list(value, separator):
    """Convert a string to a list with sanitization."""
    value_list = [item.strip() for item in value.split(separator)]
    value_list_sanitized = builtins.list(filter(None, value_list))
    if len(value_list_sanitized) > 0:
        return value_list_sanitized
    else:
        raise ValueError('Invalid list variable.')

def valid_date(x: str) -> bool:
    """
    Retrun ``True`` if ``x`` is a valid YYYYMMDD date;
    otherwise return ``False``.
    """
    try:
        if x != dt.datetime.strptime(x, DATE_FORMAT).strftime(DATE_FORMAT):
            raise ValueError
        return True
    except ValueError:
        return False

def truncate(value: Decimal, n_digits: int) -> Decimal:
    """Truncates a value to a number of decimals places"""
    return Decimal(math.trunc(value * (10 ** n_digits))) / (10 ** n_digits)

def stop(self) -> None:
        """Stops the analysis as soon as possible."""
        if self._stop and not self._posted_kork:
            self._stop()
            self._stop = None

def auto_up(self, count=1, go_to_start_of_line_if_history_changes=False):
        """
        If we're not on the first line (of a multiline input) go a line up,
        otherwise go back in history. (If nothing is selected.)
        """
        if self.complete_state:
            self.complete_previous(count=count)
        elif self.document.cursor_position_row > 0:
            self.cursor_up(count=count)
        elif not self.selection_state:
            self.history_backward(count=count)

            # Go to the start of the line?
            if go_to_start_of_line_if_history_changes:
                self.cursor_position += self.document.get_start_of_line_position()

def _reshuffle(mat, shape):
    """Reshuffle the indicies of a bipartite matrix A[ij,kl] -> A[lj,ki]."""
    return np.reshape(
        np.transpose(np.reshape(mat, shape), (3, 1, 2, 0)),
        (shape[3] * shape[1], shape[0] * shape[2]))

def integer_partition(size: int, nparts: int) -> Iterator[List[List[int]]]:
    """ Partition a list of integers into a list of partitions """
    for part in algorithm_u(range(size), nparts):
        yield part

def Exit(msg, code=1):
    """Exit execution with return code and message
    :param msg: Message displayed prior to exit
    :param code: code returned upon exiting
    """
    print >> sys.stderr, msg
    sys.exit(code)

def filter_float(n: Node, query: str) -> float:
    """
    Filter and ensure that the returned value is of type int.
    """
    return _scalariter2item(n, query, float)

def lcm(num1, num2):
    """
    Find the lowest common multiple of 2 numbers

    :type num1: number
    :param num1: The first number to find the lcm for

    :type num2: number
    :param num2: The second number to find the lcm for
    """

    if num1 > num2:
        bigger = num1
    else:
        bigger = num2
    while True:
        if bigger % num1 == 0 and bigger % num2 == 0:
            return bigger
        bigger += 1

def _rindex(mylist: Sequence[T], x: T) -> int:
    """Index of the last occurrence of x in the sequence."""
    return len(mylist) - mylist[::-1].index(x) - 1

def is_up_to_date(outfile, basedatetime):
        # type: (AnyStr, datetime) -> bool
        """Return true if outfile exists and is no older than base datetime."""
        if os.path.exists(outfile):
            if os.path.getmtime(outfile) >= basedatetime:
                return True
        return False

def from_file(file_path) -> dict:
        """ Load JSON file """
        with io.open(file_path, 'r', encoding='utf-8') as json_stream:
            return Json.parse(json_stream, True)

def run_time() -> timedelta:
    """

    :return:
    """

    delta = start_time if start_time else datetime.utcnow()
    return datetime.utcnow() - delta

def get_from_gnucash26_date(date_str: str) -> date:
    """ Creates a datetime from GnuCash 2.6 date string """
    date_format = "%Y%m%d"
    result = datetime.strptime(date_str, date_format).date()
    return result

def negate_mask(mask):
    """Returns the negated mask.

    If elements of input mask have 0 and non-zero values, then the returned matrix will have all elements 0 (1) where
    the original one has non-zero (0).

    :param mask: Input mask
    :type mask: np.array
    :return: array of same shape and dtype=int8 as input array
    :rtype: np.array
    """
    res = np.ones(mask.shape, dtype=np.int8)
    res[mask > 0] = 0

    return res

def _(f, x):
    """
    filter for dict, note `f` should have signature: `f::key->value->bool`
    """
    return {k: v for k, v in x.items() if f(k, v)}

def _cnx_is_empty(in_file):
    """Check if cnr or cns files are empty (only have a header)
    """
    with open(in_file) as in_handle:
        for i, line in enumerate(in_handle):
            if i > 0:
                return False
    return True

def get_domain(url):
    """
    Get domain part of an url.

    For example: https://www.python.org/doc/ -> https://www.python.org
    """
    parse_result = urlparse(url)
    domain = "{schema}://{netloc}".format(
        schema=parse_result.scheme, netloc=parse_result.netloc)
    return domain

def maybe_infer_dtype_type(element):
    """Try to infer an object's dtype, for use in arithmetic ops

    Uses `element.dtype` if that's available.
    Objects implementing the iterator protocol are cast to a NumPy array,
    and from there the array's type is used.

    Parameters
    ----------
    element : object
        Possibly has a `.dtype` attribute, and possibly the iterator
        protocol.

    Returns
    -------
    tipo : type

    Examples
    --------
    >>> from collections import namedtuple
    >>> Foo = namedtuple("Foo", "dtype")
    >>> maybe_infer_dtype_type(Foo(np.dtype("i8")))
    numpy.int64
    """
    tipo = None
    if hasattr(element, 'dtype'):
        tipo = element.dtype
    elif is_list_like(element):
        element = np.asarray(element)
        tipo = element.dtype
    return tipo

def memory_usage():
    """return memory usage of python process in MB

    from
    http://fa.bianp.net/blog/2013/different-ways-to-get-memory-consumption-or-lessons-learned-from-memory_profiler/
    psutil is quicker

    >>> isinstance(memory_usage(),float)
    True

    """
    try:
        import psutil
        import os
    except ImportError:
        return _memory_usage_ps()

    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem

def snake_to_camel(s: str) -> str:
    """Convert string from snake case to camel case."""

    fragments = s.split('_')

    return fragments[0] + ''.join(x.title() for x in fragments[1:])

def was_into_check(self) -> bool:
        """
        Checks if the king of the other side is attacked. Such a position is not
        valid and could only be reached by an illegal move.
        """
        king = self.king(not self.turn)
        return king is not None and self.is_attacked_by(self.turn, king)

def is_not_null(df: DataFrame, col_name: str) -> bool:
    """
    Return ``True`` if the given DataFrame has a column of the given
    name (string), and there exists at least one non-NaN value in that
    column; return ``False`` otherwise.
    """
    if (
        isinstance(df, pd.DataFrame)
        and col_name in df.columns
        and df[col_name].notnull().any()
    ):
        return True
    else:
        return False

def url_host(url: str) -> str:
    """
    Parses hostname from URL.
    :param url: URL
    :return: hostname
    """
    from urllib.parse import urlparse
    res = urlparse(url)
    return res.netloc.split(':')[0] if res.netloc else ''

def load_preprocess_images(image_paths: List[str], image_size: tuple) -> List[np.ndarray]:
    """
    Load and pre-process the images specified with absolute paths.

    :param image_paths: List of images specified with paths.
    :param image_size: Tuple to resize the image to (Channels, Height, Width)
    :return: A list of loaded images (numpy arrays).
    """
    image_size = image_size[1:]  # we do not need the number of channels
    images = []
    for image_path in image_paths:
        images.append(load_preprocess_image(image_path, image_size))
    return images

def is_relative_url(url):
    """ simple method to determine if a url is relative or absolute """
    if url.startswith("#"):
        return None
    if url.find("://") > 0 or url.startswith("//"):
        # either 'http(s)://...' or '//cdn...' and therefore absolute
        return False
    return True

def is_natural(x):
    """A non-negative integer."""
    try:
        is_integer = int(x) == x
    except (TypeError, ValueError):
        return False
    return is_integer and x >= 0

def find_unit_clause(clauses, model):
    """Find a forced assignment if possible from a clause with only 1
    variable not bound in the model.
    >>> find_unit_clause([A|B|C, B|~C, ~A|~B], {A:True})
    (B, False)
    """
    for clause in clauses:
        P, value = unit_clause_assign(clause, model)
        if P: return P, value
    return None, None

def valid_date(x: str) -> bool:
    """
    Retrun ``True`` if ``x`` is a valid YYYYMMDD date;
    otherwise return ``False``.
    """
    try:
        if x != dt.datetime.strptime(x, DATE_FORMAT).strftime(DATE_FORMAT):
            raise ValueError
        return True
    except ValueError:
        return False

def is_unicode(string):
    """Validates that the object itself is some kinda string"""
    str_type = str(type(string))

    if str_type.find('str') > 0 or str_type.find('unicode') > 0:
        return True

    return False

def dotproduct(X, Y):
    """Return the sum of the element-wise product of vectors x and y.
    >>> dotproduct([1, 2, 3], [1000, 100, 10])
    1230
    """
    return sum([x * y for x, y in zip(X, Y)])

def chars(string: any) -> str:
        """Return all (and only) the chars in the given string."""
        return ''.join([c if c.isalpha() else '' for c in str(string)])

def listify(a):
    """
    Convert a scalar ``a`` to a list and all iterables to list as well.

    Examples
    --------
    >>> listify(0)
    [0]

    >>> listify([1,2,3])
    [1, 2, 3]

    >>> listify('a')
    ['a']

    >>> listify(np.array([1,2,3]))
    [1, 2, 3]

    >>> listify('string')
    ['string']
    """
    if a is None:
        return []
    elif not isinstance(a, (tuple, list, np.ndarray)):
        return [a]
    return list(a)

def year(date):
    """ Returns the year.

    :param date:
        The string date with this format %m/%d/%Y
    :type date:
        String

    :returns:
        int

    :example:
        >>> year('05/1/2015')
        2015
    """
    try:
        fmt = '%m/%d/%Y'
        return datetime.strptime(date, fmt).timetuple().tm_year
    except ValueError:
        return 0

def clean_column_names(df: DataFrame) -> DataFrame:
    """
    Strip the whitespace from all column names in the given DataFrame
    and return the result.
    """
    f = df.copy()
    f.columns = [col.strip() for col in f.columns]
    return f

def __remove_trailing_zeros(self, collection):
        """Removes trailing zeroes from indexable collection of numbers"""
        index = len(collection) - 1
        while index >= 0 and collection[index] == 0:
            index -= 1

        return collection[:index + 1]

def find_first_in_list(txt: str, str_list: [str]) -> int:  # type: ignore
    """
    Returns the index of the earliest occurence of an item from a list in a string

    Ex: find_first_in_list('foobar', ['bar', 'fin']) -> 3
    """
    start = len(txt) + 1
    for item in str_list:
        if start > txt.find(item) > -1:
            start = txt.find(item)
    return start if len(txt) + 1 > start > -1 else -1

def get_now_utc_notz_datetime() -> datetime.datetime:
    """
    Get the UTC time now, but with no timezone information,
    in :class:`datetime.datetime` format.
    """
    now = datetime.datetime.utcnow()
    return now.replace(tzinfo=None)

def __replace_all(repls: dict, input: str) -> str:
    """ Replaces from a string **input** all the occurrences of some
    symbols according to mapping **repls**.

    :param dict repls: where #key is the old character and
    #value is the one to substitute with;
    :param str input: original string where to apply the
    replacements;
    :return: *(str)* the string with the desired characters replaced
    """
    return re.sub('|'.join(re.escape(key) for key in repls.keys()),
                  lambda k: repls[k.group(0)], input)

def make_dep_graph(depender):
	"""Returns a digraph string fragment based on the passed-in module
	"""
	shutit_global.shutit_global_object.yield_to_draw()
	digraph = ''
	for dependee_id in depender.depends_on:
		digraph = (digraph + '"' + depender.module_id + '"->"' + dependee_id + '";\n')
	return digraph

def isarray(array, test, dim=2):
    """Returns True if test is True for all array elements.
    Otherwise, returns False.
    """
    if dim > 1:
        return all(isarray(array[i], test, dim - 1)
                   for i in range(len(array)))
    return all(test(i) for i in array)

def clean(ctx, text):
    """
    Removes all non-printable characters from a text string
    """
    text = conversions.to_string(text, ctx)
    return ''.join([c for c in text if ord(c) >= 32])

def get_from_gnucash26_date(date_str: str) -> date:
    """ Creates a datetime from GnuCash 2.6 date string """
    date_format = "%Y%m%d"
    result = datetime.strptime(date_str, date_format).date()
    return result

def read_text_from_file(path: str) -> str:
    """ Reads text file contents """
    with open(path) as text_file:
        content = text_file.read()

    return content

def _find_conda():
    """Find the conda executable robustly across conda versions.

    Returns
    -------
    conda : str
        Path to the conda executable.

    Raises
    ------
    IOError
        If the executable cannot be found in either the CONDA_EXE environment
        variable or in the PATH.

    Notes
    -----
    In POSIX platforms in conda >= 4.4, conda can be set up as a bash function
    rather than an executable. (This is to enable the syntax
    ``conda activate env-name``.) In this case, the environment variable
    ``CONDA_EXE`` contains the path to the conda executable. In other cases,
    we use standard search for the appropriate name in the PATH.

    See https://github.com/airspeed-velocity/asv/issues/645 for more details.
    """
    if 'CONDA_EXE' in os.environ:
        conda = os.environ['CONDA_EXE']
    else:
        conda = util.which('conda')
    return conda

def connect_to_database_odbc_access(self,
                                        dsn: str,
                                        autocommit: bool = True) -> None:
        """Connects to an Access database via ODBC, with the DSN
        prespecified."""
        self.connect(engine=ENGINE_ACCESS, interface=INTERFACE_ODBC,
                     dsn=dsn, autocommit=autocommit)

def _find_conda():
    """Find the conda executable robustly across conda versions.

    Returns
    -------
    conda : str
        Path to the conda executable.

    Raises
    ------
    IOError
        If the executable cannot be found in either the CONDA_EXE environment
        variable or in the PATH.

    Notes
    -----
    In POSIX platforms in conda >= 4.4, conda can be set up as a bash function
    rather than an executable. (This is to enable the syntax
    ``conda activate env-name``.) In this case, the environment variable
    ``CONDA_EXE`` contains the path to the conda executable. In other cases,
    we use standard search for the appropriate name in the PATH.

    See https://github.com/airspeed-velocity/asv/issues/645 for more details.
    """
    if 'CONDA_EXE' in os.environ:
        conda = os.environ['CONDA_EXE']
    else:
        conda = util.which('conda')
    return conda

def clean(ctx, text):
    """
    Removes all non-printable characters from a text string
    """
    text = conversions.to_string(text, ctx)
    return ''.join([c for c in text if ord(c) >= 32])

def _parse_tuple_string(argument):
        """ Return a tuple from parsing 'a,b,c,d' -> (a,b,c,d) """
        if isinstance(argument, str):
            return tuple(int(p.strip()) for p in argument.split(','))
        return argument

def remove_blank_lines(string):
    """ Removes all blank lines in @string

        -> #str without blank lines
    """
    return "\n".join(line
                     for line in string.split("\n")
                     if len(line.strip()))

def returned(n):
	"""Generate a random walk and return True if the walker has returned to
	the origin after taking `n` steps.
	"""
	## `takei` yield lazily so we can short-circuit and avoid computing the rest of the walk
	for pos in randwalk() >> drop(1) >> takei(xrange(n-1)):
		if pos == Origin:
			return True
	return False

def get_now_utc_notz_datetime() -> datetime.datetime:
    """
    Get the UTC time now, but with no timezone information,
    in :class:`datetime.datetime` format.
    """
    now = datetime.datetime.utcnow()
    return now.replace(tzinfo=None)

def replace_in_list(stringlist: Iterable[str],
                    replacedict: Dict[str, str]) -> List[str]:
    """
    Returns a list produced by applying :func:`multiple_replace` to every
    string in ``stringlist``.

    Args:
        stringlist: list of source strings
        replacedict: dictionary mapping "original" to "replacement" strings

    Returns:
        list of final strings

    """
    newlist = []
    for fromstring in stringlist:
        newlist.append(multiple_replace(fromstring, replacedict))
    return newlist

def __replace_all(repls: dict, input: str) -> str:
    """ Replaces from a string **input** all the occurrences of some
    symbols according to mapping **repls**.

    :param dict repls: where #key is the old character and
    #value is the one to substitute with;
    :param str input: original string where to apply the
    replacements;
    :return: *(str)* the string with the desired characters replaced
    """
    return re.sub('|'.join(re.escape(key) for key in repls.keys()),
                  lambda k: repls[k.group(0)], input)

def astensor(array: TensorLike) -> BKTensor:
    """Covert numpy array to tensorflow tensor"""
    tensor = tf.convert_to_tensor(value=array, dtype=CTYPE)
    return tensor

def _rindex(mylist: Sequence[T], x: T) -> int:
    """Index of the last occurrence of x in the sequence."""
    return len(mylist) - mylist[::-1].index(x) - 1

def set_range(self, min_val, max_val):
        """Set the range of the colormap to [*min_val*, *max_val*]
        """
        if min_val > max_val:
            max_val, min_val = min_val, max_val
        self.values = (((self.values * 1.0 - self.values.min()) /
                        (self.values.max() - self.values.min()))
                       * (max_val - min_val) + min_val)

def read_set_from_file(filename: str) -> Set[str]:
    """
    Extract a de-duped collection (set) of text from a file.
    Expected file format is one item per line.
    """
    collection = set()
    with open(filename, 'r') as file_:
        for line in file_:
            collection.add(line.rstrip())
    return collection

def Exit(msg, code=1):
    """Exit execution with return code and message
    :param msg: Message displayed prior to exit
    :param code: code returned upon exiting
    """
    print >> sys.stderr, msg
    sys.exit(code)

def isarray(array, test, dim=2):
    """Returns True if test is True for all array elements.
    Otherwise, returns False.
    """
    if dim > 1:
        return all(isarray(array[i], test, dim - 1)
                   for i in range(len(array)))
    return all(test(i) for i in array)

def remove_blank_lines(string):
    """ Removes all blank lines in @string

        -> #str without blank lines
    """
    return "\n".join(line
                     for line in string.split("\n")
                     if len(line.strip()))

def genfirstvalues(cursor: Cursor, arraysize: int = 1000) \
        -> Generator[Any, None, None]:
    """
    Generate the first value in each row.

    Args:
        cursor: the cursor
        arraysize: split fetches into chunks of this many records

    Yields:
        the first value of each row
    """
    return (row[0] for row in genrows(cursor, arraysize))

def _parse_tuple_string(argument):
        """ Return a tuple from parsing 'a,b,c,d' -> (a,b,c,d) """
        if isinstance(argument, str):
            return tuple(int(p.strip()) for p in argument.split(','))
        return argument

def get_keys_of_max_n(dict_obj, n):
    """Returns the keys that maps to the top n max values in the given dict.

    Example:
    --------
    >>> dict_obj = {'a':2, 'b':1, 'c':5}
    >>> get_keys_of_max_n(dict_obj, 2)
    ['a', 'c']
    """
    return sorted([
        item[0]
        for item in sorted(
            dict_obj.items(), key=lambda item: item[1], reverse=True
        )[:n]
    ])

def debugTreePrint(node,pfx="->"):
  """Purely a debugging aid: Ascii-art picture of a tree descended from node"""
  print pfx,node.item
  for c in node.children:
    debugTreePrint(c,"  "+pfx)

def _create_empty_array(self, frames, always_2d, dtype):
        """Create an empty array with appropriate shape."""
        import numpy as np
        if always_2d or self.channels > 1:
            shape = frames, self.channels
        else:
            shape = frames,
        return np.empty(shape, dtype, order='C')

def fast_median(a):
    """Fast median operation for masked array using 50th-percentile
    """
    a = checkma(a)
    #return scoreatpercentile(a.compressed(), 50)
    if a.count() > 0:
        out = np.percentile(a.compressed(), 50)
    else:
        out = np.ma.masked
    return out

def check_key(self, key: str) -> bool:
        """
        Checks if key exists in datastore. True if yes, False if no.

        :param: SHA512 hash key

        :return: whether or key not exists in datastore
        """
        keys = self.get_keys()
        return key in keys

def dict_of_sets_add(dictionary, key, value):
    # type: (DictUpperBound, Any, Any) -> None
    """Add value to a set in a dictionary by key

    Args:
        dictionary (DictUpperBound): Dictionary to which to add values
        key (Any): Key within dictionary
        value (Any): Value to add to set in dictionary

    Returns:
        None

    """
    set_objs = dictionary.get(key, set())
    set_objs.add(value)
    dictionary[key] = set_objs

def to_bytes(data: Any) -> bytearray:
    """
    Convert anything to a ``bytearray``.
    
    See
    
    - http://stackoverflow.com/questions/7585435/best-way-to-convert-string-to-bytes-in-python-3
    - http://stackoverflow.com/questions/10459067/how-to-convert-my-bytearrayb-x9e-x18k-x9a-to-something-like-this-x9e-x1
    """  # noqa
    if isinstance(data, int):
        return bytearray([data])
    return bytearray(data, encoding='latin-1')

def read_text_from_file(path: str) -> str:
    """ Reads text file contents """
    with open(path) as text_file:
        content = text_file.read()

    return content

def from_uuid(value: uuid.UUID) -> ulid.ULID:
    """
    Create a new :class:`~ulid.ulid.ULID` instance from the given :class:`~uuid.UUID` value.

    :param value: UUIDv4 value
    :type value: :class:`~uuid.UUID`
    :return: ULID from UUID value
    :rtype: :class:`~ulid.ulid.ULID`
    """
    return ulid.ULID(value.bytes)

def to_bytes(data: Any) -> bytearray:
    """
    Convert anything to a ``bytearray``.
    
    See
    
    - http://stackoverflow.com/questions/7585435/best-way-to-convert-string-to-bytes-in-python-3
    - http://stackoverflow.com/questions/10459067/how-to-convert-my-bytearrayb-x9e-x18k-x9a-to-something-like-this-x9e-x1
    """  # noqa
    if isinstance(data, int):
        return bytearray([data])
    return bytearray(data, encoding='latin-1')

def is_relative_url(url):
    """ simple method to determine if a url is relative or absolute """
    if url.startswith("#"):
        return None
    if url.find("://") > 0 or url.startswith("//"):
        # either 'http(s)://...' or '//cdn...' and therefore absolute
        return False
    return True

def argmax(self, rows: List[Row], column: ComparableColumn) -> List[Row]:
        """
        Takes a list of rows and a column name and returns a list containing a single row (dict from
        columns to cells) that has the maximum numerical value in the given column. We return a list
        instead of a single dict to be consistent with the return type of ``select`` and
        ``all_rows``.
        """
        if not rows:
            return []
        value_row_pairs = [(row.values[column.name], row) for row in rows]
        if not value_row_pairs:
            return []
        # Returns a list containing the row with the max cell value.
        return [sorted(value_row_pairs, key=lambda x: x[0], reverse=True)[0][1]]

def returned(n):
	"""Generate a random walk and return True if the walker has returned to
	the origin after taking `n` steps.
	"""
	## `takei` yield lazily so we can short-circuit and avoid computing the rest of the walk
	for pos in randwalk() >> drop(1) >> takei(xrange(n-1)):
		if pos == Origin:
			return True
	return False

def dictlist_wipe_key(dict_list: Iterable[Dict], key: str) -> None:
    """
    Process an iterable of dictionaries. For each dictionary ``d``, delete
    ``d[key]`` if it exists.
    """
    for d in dict_list:
        d.pop(key, None)

def make_indices_to_labels(labels: Set[str]) -> Dict[int, str]:
    """ Creates a mapping from indices to labels. """

    return {index: label for index, label in
            enumerate(["pad"] + sorted(list(labels)))}

def get_terminal_width():
    """ -> #int width of the terminal window """
    # http://www.brandonrubin.me/2014/03/18/python-snippet-get-terminal-width/
    command = ['tput', 'cols']
    try:
        width = int(subprocess.check_output(command))
    except OSError as e:
        print(
            "Invalid Command '{0}': exit status ({1})".format(
                command[0], e.errno))
    except subprocess.CalledProcessError as e:
        print(
            "'{0}' returned non-zero exit status: ({1})".format(
                command, e.returncode))
    else:
        return width

def _cnx_is_empty(in_file):
    """Check if cnr or cns files are empty (only have a header)
    """
    with open(in_file) as in_handle:
        for i, line in enumerate(in_handle):
            if i > 0:
                return False
    return True

def flush(self):
        """
        Ensure all logging output has been flushed
        """
        if len(self._buffer) > 0:
            self.logger.log(self.level, self._buffer)
            self._buffer = str()

def _brief_print_list(lst, limit=7):
    """Print at most `limit` elements of list."""
    lst = list(lst)
    if len(lst) > limit:
        return _brief_print_list(lst[:limit//2], limit) + ', ..., ' + \
            _brief_print_list(lst[-limit//2:], limit)
    return ', '.join(["'%s'"%str(i) for i in lst])

def assert_or_raise(stmt: bool, exception: Exception,
                    *exception_args, **exception_kwargs) -> None:
  """
  If the statement is false, raise the given exception.
  """
  if not stmt:
    raise exception(*exception_args, **exception_kwargs)

def strip_codes(s: Any) -> str:
    """ Strip all color codes from a string.
        Returns empty string for "falsey" inputs.
    """
    return codepat.sub('', str(s) if (s or (s == 0)) else '')

def is_line_in_file(filename: str, line: str) -> bool:
    """
    Detects whether a line is present within a file.

    Args:
        filename: file to check
        line: line to search for (as an exact match)
    """
    assert "\n" not in line
    with open(filename, "r") as file:
        for fileline in file:
            if fileline == line:
                return True
        return False

def assign_parent(node: astroid.node_classes.NodeNG) -> astroid.node_classes.NodeNG:
    """return the higher parent which is not an AssignName, Tuple or List node
    """
    while node and isinstance(node, (astroid.AssignName, astroid.Tuple, astroid.List)):
        node = node.parent
    return node

def columns_equal(a: Column, b: Column) -> bool:
    """
    Are two SQLAlchemy columns are equal? Checks based on:

    - column ``name``
    - column ``type`` (see :func:`column_types_equal`)
    - ``nullable``
    """
    return (
        a.name == b.name and
        column_types_equal(a.type, b.type) and
        a.nullable == b.nullable
    )

def PrintIndented(self, file, ident, code):
        """Takes an array, add indentation to each entry and prints it."""
        for entry in code:
            print >>file, '%s%s' % (ident, entry)

def argmax(self, rows: List[Row], column: ComparableColumn) -> List[Row]:
        """
        Takes a list of rows and a column name and returns a list containing a single row (dict from
        columns to cells) that has the maximum numerical value in the given column. We return a list
        instead of a single dict to be consistent with the return type of ``select`` and
        ``all_rows``.
        """
        if not rows:
            return []
        value_row_pairs = [(row.values[column.name], row) for row in rows]
        if not value_row_pairs:
            return []
        # Returns a list containing the row with the max cell value.
        return [sorted(value_row_pairs, key=lambda x: x[0], reverse=True)[0][1]]

def sample_normal(mean, var, rng):
    """Sample from independent normal distributions

    Each element is an independent normal distribution.

    Parameters
    ----------
    mean : numpy.ndarray
      Means of the normal distribution. Shape --> (batch_num, sample_dim)
    var : numpy.ndarray
      Variance of the normal distribution. Shape --> (batch_num, sample_dim)
    rng : numpy.random.RandomState

    Returns
    -------
    ret : numpy.ndarray
       The sampling result. Shape --> (batch_num, sample_dim)
    """
    ret = numpy.sqrt(var) * rng.randn(*mean.shape) + mean
    return ret

def _duplicates(list_):
    """Return dict mapping item -> indices."""
    item_indices = {}
    for i, item in enumerate(list_):
        try:
            item_indices[item].append(i)
        except KeyError:  # First time seen
            item_indices[item] = [i]
    return item_indices

def hex_to_int(value):
    """
    Convert hex string like "\x0A\xE3" to 2787.
    """
    if version_info.major >= 3:
        return int.from_bytes(value, "big")
    return int(value.encode("hex"), 16)

def is_quoted(arg: str) -> bool:
    """
    Checks if a string is quoted
    :param arg: the string being checked for quotes
    :return: True if a string is quoted
    """
    return len(arg) > 1 and arg[0] == arg[-1] and arg[0] in constants.QUOTES

def get_system_flags() -> FrozenSet[Flag]:
    """Return the set of implemented system flags."""
    return frozenset({Seen, Recent, Deleted, Flagged, Answered, Draft})

def dag_longest_path(graph, source, target):
    """
    Finds the longest path in a dag between two nodes
    """
    if source == target:
        return [source]
    allpaths = nx.all_simple_paths(graph, source, target)
    longest_path = []
    for l in allpaths:
        if len(l) > len(longest_path):
            longest_path = l
    return longest_path

def get_property_as_float(self, name: str) -> float:
        """Return the value of a float property.

        :return: The property value (float).

        Raises exception if property with name doesn't exist.

        .. versionadded:: 1.0

        Scriptable: Yes
        """
        return float(self.__instrument.get_property(name))

def rollapply(data, window, fn):
    """
    Apply a function fn over a rolling window of size window.

    Args:
        * data (Series or DataFrame): Series or DataFrame
        * window (int): Window size
        * fn (function): Function to apply over the rolling window.
            For a series, the return value is expected to be a single
            number. For a DataFrame, it shuold return a new row.

    Returns:
        * Object of same dimensions as data
    """
    res = data.copy()
    res[:] = np.nan
    n = len(data)

    if window > n:
        return res

    for i in range(window - 1, n):
        res.iloc[i] = fn(data.iloc[i - window + 1:i + 1])

    return res

def is_unitary(matrix: np.ndarray) -> bool:
    """
    A helper function that checks if a matrix is unitary.

    :param matrix: a matrix to test unitarity of
    :return: true if and only if matrix is unitary
    """
    rows, cols = matrix.shape
    if rows != cols:
        return False
    return np.allclose(np.eye(rows), matrix.dot(matrix.T.conj()))

def flatten_list(x: List[Any]) -> List[Any]:
    """
    Converts a list of lists into a flat list.
    
    Args:
        x: list of lists 

    Returns:
        flat list
        
    As per
    http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python

    """  # noqa
    return [item for sublist in x for item in sublist]

def get_table_names_from_metadata(metadata: MetaData) -> List[str]:
    """
    Returns all database table names found in an SQLAlchemy :class:`MetaData`
    object.
    """
    return [table.name for table in metadata.tables.values()]

def _mid(pt1, pt2):
    """
    (Point, Point) -> Point
    Return the point that lies in between the two input points.
    """
    (x0, y0), (x1, y1) = pt1, pt2
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)

def indexes_equal(a: Index, b: Index) -> bool:
    """
    Are two indexes equal? Checks by comparing ``str()`` versions of them.
    (AM UNSURE IF THIS IS ENOUGH.)
    """
    return str(a) == str(b)

def long_substr(data):
    """Return the longest common substring in a list of strings.
    
    Credit: http://stackoverflow.com/questions/2892931/longest-common-substring-from-more-than-two-strings-python
    """
    substr = ''
    if len(data) > 1 and len(data[0]) > 0:
        for i in range(len(data[0])):
            for j in range(len(data[0])-i+1):
                if j > len(substr) and all(data[0][i:i+j] in x for x in data):
                    substr = data[0][i:i+j]
    elif len(data) == 1:
        substr = data[0]
    return substr

def truncate(value: Decimal, n_digits: int) -> Decimal:
    """Truncates a value to a number of decimals places"""
    return Decimal(math.trunc(value * (10 ** n_digits))) / (10 ** n_digits)

def binary(length):
    """
        returns a a random string that represent a binary representation

    :param length: number of bits
    """
    num = randint(1, 999999)
    mask = '0' * length
    return (mask + ''.join([str(num >> i & 1) for i in range(7, -1, -1)]))[-length:]

def is_finite(value: Any) -> bool:
    """Return true if a value is a finite number."""
    return isinstance(value, int) or (isinstance(value, float) and isfinite(value))

def clear(self) -> None:
        """Resets all headers and content for this response."""
        self._headers = httputil.HTTPHeaders(
            {
                "Server": "TornadoServer/%s" % tornado.version,
                "Content-Type": "text/html; charset=UTF-8",
                "Date": httputil.format_timestamp(time.time()),
            }
        )
        self.set_default_headers()
        self._write_buffer = []  # type: List[bytes]
        self._status_code = 200
        self._reason = httputil.responses[200]

def is_closing(self) -> bool:
        """Return ``True`` if this connection is closing.

        The connection is considered closing if either side has
        initiated its closing handshake or if the stream has been
        shut down uncleanly.
        """
        return self.stream.closed() or self.client_terminated or self.server_terminated

def copy_session(session: requests.Session) -> requests.Session:
    """Duplicates a requests.Session."""
    new = requests.Session()
    new.cookies = requests.utils.cookiejar_from_dict(requests.utils.dict_from_cookiejar(session.cookies))
    new.headers = session.headers.copy()
    return new

def listify(a):
    """
    Convert a scalar ``a`` to a list and all iterables to list as well.

    Examples
    --------
    >>> listify(0)
    [0]

    >>> listify([1,2,3])
    [1, 2, 3]

    >>> listify('a')
    ['a']

    >>> listify(np.array([1,2,3]))
    [1, 2, 3]

    >>> listify('string')
    ['string']
    """
    if a is None:
        return []
    elif not isinstance(a, (tuple, list, np.ndarray)):
        return [a]
    return list(a)

def _str_to_list(value, separator):
    """Convert a string to a list with sanitization."""
    value_list = [item.strip() for item in value.split(separator)]
    value_list_sanitized = builtins.list(filter(None, value_list))
    if len(value_list_sanitized) > 0:
        return value_list_sanitized
    else:
        raise ValueError('Invalid list variable.')

def _run_sync(self, method: Callable, *args, **kwargs) -> Any:
        """
        Utility method to run commands synchronously for testing.
        """
        if self.loop.is_running():
            raise RuntimeError("Event loop is already running.")

        if not self.is_connected:
            self.loop.run_until_complete(self.connect())

        task = asyncio.Task(method(*args, **kwargs), loop=self.loop)
        result = self.loop.run_until_complete(task)

        self.loop.run_until_complete(self.quit())

        return result

def is_line_in_file(filename: str, line: str) -> bool:
    """
    Detects whether a line is present within a file.

    Args:
        filename: file to check
        line: line to search for (as an exact match)
    """
    assert "\n" not in line
    with open(filename, "r") as file:
        for fileline in file:
            if fileline == line:
                return True
        return False

def __remove_trailing_zeros(self, collection):
        """Removes trailing zeroes from indexable collection of numbers"""
        index = len(collection) - 1
        while index >= 0 and collection[index] == 0:
            index -= 1

        return collection[:index + 1]

def memory_full():
    """Check if the memory is too full for further caching."""
    current_process = psutil.Process(os.getpid())
    return (current_process.memory_percent() >
            config.MAXIMUM_CACHE_MEMORY_PERCENTAGE)

def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.

    Checks whether all objects in arrays have the same shape or length.

    Parameters
    ----------
    arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """

    uniques = np.unique([_num_samples(X) for X in arrays if X is not None])
    if len(uniques) > 1:
        raise ValueError("Found arrays with inconsistent numbers of samples: %s"
                         % str(uniques))

def strtobytes(input, encoding):
    """Take a str and transform it into a byte array."""
    py_version = sys.version_info[0]
    if py_version >= 3:
        return _strtobytes_py3(input, encoding)
    return _strtobytes_py2(input, encoding)

def _cleanup(path: str) -> None:
    """Cleanup temporary directory."""
    if os.path.isdir(path):
        shutil.rmtree(path)

def indexes_equal(a: Index, b: Index) -> bool:
    """
    Are two indexes equal? Checks by comparing ``str()`` versions of them.
    (AM UNSURE IF THIS IS ENOUGH.)
    """
    return str(a) == str(b)

def __getattr__(self, item: str) -> Callable:
        """Get a callable that sends the actual API request internally."""
        return functools.partial(self.call_action, item)

def calculate_single_tanimoto_set_distances(target: Iterable[X], dict_of_sets: Mapping[Y, Set[X]]) -> Mapping[Y, float]:
    """Return a dictionary of distances keyed by the keys in the given dict.

    Distances are calculated based on pairwise tanimoto similarity of the sets contained

    :param set target: A set
    :param dict_of_sets: A dict of {x: set of y}
    :type dict_of_sets: dict
    :return: A similarity dicationary based on the set overlap (tanimoto) score between the target set and the sets in
            dos
    :rtype: dict
    """
    target_set = set(target)

    return {
        k: tanimoto_set_similarity(target_set, s)
        for k, s in dict_of_sets.items()
    }

def top(self, topn=10):
        """
        Get a list of the top ``topn`` features in this :class:`.Feature`\.

        Examples
        --------

        .. code-block:: python

        >>> myFeature = Feature([('the', 2), ('pine', 1), ('trapezoid', 5)])
        >>> myFeature.top(1)
        [('trapezoid', 5)]

        Parameters
        ----------
        topn : int

        Returns
        -------
        list
        """
        return [self[i] for i in argsort(list(zip(*self))[1])[::-1][:topn]]

def chars(string: any) -> str:
        """Return all (and only) the chars in the given string."""
        return ''.join([c if c.isalpha() else '' for c in str(string)])

def quaternion_imag(quaternion):
    """Return imaginary part of quaternion.

    >>> quaternion_imag([3, 0, 1, 2])
    array([ 0.,  1.,  2.])

    """
    return numpy.array(quaternion[1:4], dtype=numpy.float64, copy=True)

def get_case_insensitive_dict_key(d: Dict, k: str) -> Optional[str]:
    """
    Within the dictionary ``d``, find a key that matches (in case-insensitive
    fashion) the key ``k``, and return it (or ``None`` if there isn't one).
    """
    for key in d.keys():
        if k.lower() == key.lower():
            return key
    return None

def inverted_dict(d):
    """Return a dict with swapped keys and values

    >>> inverted_dict({0: ('a', 'b'), 1: 'cd'}) == {'cd': 1, ('a', 'b'): 0}
    True
    """
    return dict((force_hashable(v), k) for (k, v) in viewitems(dict(d)))

def to_bytes(data: Any) -> bytearray:
    """
    Convert anything to a ``bytearray``.
    
    See
    
    - http://stackoverflow.com/questions/7585435/best-way-to-convert-string-to-bytes-in-python-3
    - http://stackoverflow.com/questions/10459067/how-to-convert-my-bytearrayb-x9e-x18k-x9a-to-something-like-this-x9e-x1
    """  # noqa
    if isinstance(data, int):
        return bytearray([data])
    return bytearray(data, encoding='latin-1')

def remove_blank_lines(string):
    """ Removes all blank lines in @string

        -> #str without blank lines
    """
    return "\n".join(line
                     for line in string.split("\n")
                     if len(line.strip()))

def snake_case(a_string):
    """Returns a snake cased version of a string.

    :param a_string: any :class:`str` object.

    Usage:
        >>> snake_case('FooBar')
        "foo_bar"
    """

    partial = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', a_string)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', partial).lower()

async def executemany(self, sql: str, parameters: Iterable[Iterable[Any]]) -> None:
        """Execute the given multiquery."""
        await self._execute(self._cursor.executemany, sql, parameters)

def csv_to_numpy(string_like, dtype=None):  # type: (str) -> np.array
    """Convert a CSV object to a numpy array.

    Args:
        string_like (str): CSV string.
        dtype (dtype, optional):  Data type of the resulting array. If None, the dtypes will be determined by the
                                        contents of each column, individually. This argument can only be used to
                                        'upcast' the array.  For downcasting, use the .astype(t) method.
    Returns:
        (np.array): numpy array
    """
    stream = StringIO(string_like)
    return np.genfromtxt(stream, dtype=dtype, delimiter=',')

def find_first_in_list(txt: str, str_list: [str]) -> int:  # type: ignore
    """
    Returns the index of the earliest occurence of an item from a list in a string

    Ex: find_first_in_list('foobar', ['bar', 'fin']) -> 3
    """
    start = len(txt) + 1
    for item in str_list:
        if start > txt.find(item) > -1:
            start = txt.find(item)
    return start if len(txt) + 1 > start > -1 else -1

def flatten_multidict(multidict):
    """Return flattened dictionary from ``MultiDict``."""
    return dict([(key, value if len(value) > 1 else value[0])
                 for (key, value) in multidict.iterlists()])

def warn_if_nans_exist(X):
    """Warn if nans exist in a numpy array."""
    null_count = count_rows_with_nans(X)
    total = len(X)
    percent = 100 * null_count / total

    if null_count > 0:
        warning_message = \
            'Warning! Found {} rows of {} ({:0.2f}%) with nan values. Only ' \
            'complete rows will be plotted.'.format(null_count, total, percent)
        warnings.warn(warning_message, DataWarning)

def to_int64(a):
    """Return view of the recarray with all int32 cast to int64."""
    # build new dtype and replace i4 --> i8
    def promote_i4(typestr):
        if typestr[1:] == 'i4':
            typestr = typestr[0]+'i8'
        return typestr

    dtype = [(name, promote_i4(typestr)) for name,typestr in a.dtype.descr]
    return a.astype(dtype)

def __next__(self):
        """
        :return: int
        """
        self.current += 1
        if self.current > self.total:
            raise StopIteration
        else:
            return self.iterable[self.current - 1]

def _cnx_is_empty(in_file):
    """Check if cnr or cns files are empty (only have a header)
    """
    with open(in_file) as in_handle:
        for i, line in enumerate(in_handle):
            if i > 0:
                return False
    return True

def join_states(*states: State) -> State:
    """Join two state vectors into a larger qubit state"""
    vectors = [ket.vec for ket in states]
    vec = reduce(outer_product, vectors)
    return State(vec.tensor, vec.qubits)

def blk_coverage_1d(blk, size):
    """Return the part of a 1d array covered by a block.

    :param blk: size of the 1d block
    :param size: size of the 1d a image
    :return: a tuple of size covered and remaining size

    Example:

        >>> blk_coverage_1d(7, 100)
        (98, 2)

    """
    rem = size % blk
    maxpix = size - rem
    return maxpix, rem

def cpu_count() -> int:
    """Returns the number of processors on this machine."""
    if multiprocessing is None:
        return 1
    try:
        return multiprocessing.cpu_count()
    except NotImplementedError:
        pass
    try:
        return os.sysconf("SC_NPROCESSORS_CONF")
    except (AttributeError, ValueError):
        pass
    gen_log.error("Could not detect number of processors; assuming 1")
    return 1

def debugTreePrint(node,pfx="->"):
  """Purely a debugging aid: Ascii-art picture of a tree descended from node"""
  print pfx,node.item
  for c in node.children:
    debugTreePrint(c,"  "+pfx)

def rmglob(pattern: str) -> None:
    """
    Deletes all files whose filename matches the glob ``pattern`` (via
    :func:`glob.glob`).
    """
    for f in glob.glob(pattern):
        os.remove(f)

def from_file(file_path) -> dict:
        """ Load JSON file """
        with io.open(file_path, 'r', encoding='utf-8') as json_stream:
            return Json.parse(json_stream, True)

def most_frequent(lst):
    """
    Returns the item that appears most frequently in the given list.
    """
    lst = lst[:]
    highest_freq = 0
    most_freq = None

    for val in unique(lst):
        if lst.count(val) > highest_freq:
            most_freq = val
            highest_freq = lst.count(val)
            
    return most_freq

def array2string(arr: numpy.ndarray) -> str:
        """Format numpy array as a string."""
        shape = str(arr.shape)[1:-1]
        if shape.endswith(","):
            shape = shape[:-1]
        return numpy.array2string(arr, threshold=11) + "%s[%s]" % (arr.dtype, shape)

def dotproduct(X, Y):
    """Return the sum of the element-wise product of vectors x and y.
    >>> dotproduct([1, 2, 3], [1000, 100, 10])
    1230
    """
    return sum([x * y for x, y in zip(X, Y)])

def tanimoto_set_similarity(x: Iterable[X], y: Iterable[X]) -> float:
    """Calculate the tanimoto set similarity."""
    a, b = set(x), set(y)
    union = a | b

    if not union:
        return 0.0

    return len(a & b) / len(union)

def most_significant_bit(lst: np.ndarray) -> int:
    """
    A helper function that finds the position of the most significant bit in a 1darray of 1s and 0s,
    i.e. the first position where a 1 appears, reading left to right.

    :param lst: a 1d array of 0s and 1s with at least one 1
    :return: the first position in lst that a 1 appears
    """
    return np.argwhere(np.asarray(lst) == 1)[0][0]

def string(value) -> str:
        """ string dict/object/value to JSON """
        return system_json.dumps(Json(value).safe_object(), ensure_ascii=False)

def is_up_to_date(outfile, basedatetime):
        # type: (AnyStr, datetime) -> bool
        """Return true if outfile exists and is no older than base datetime."""
        if os.path.exists(outfile):
            if os.path.getmtime(outfile) >= basedatetime:
                return True
        return False

def get_from_gnucash26_date(date_str: str) -> date:
    """ Creates a datetime from GnuCash 2.6 date string """
    date_format = "%Y%m%d"
    result = datetime.strptime(date_str, date_format).date()
    return result

def _parse_date(string: str) -> datetime.date:
    """Parse an ISO format date (YYYY-mm-dd).

    >>> _parse_date('1990-01-02')
    datetime.date(1990, 1, 2)
    """
    return datetime.datetime.strptime(string, '%Y-%m-%d').date()

def s3_get(url: str, temp_file: IO) -> None:
    """Pull a file directly from S3."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)

def is_rate_limited(response):
        """
        Checks if the response has been rate limited by CARTO APIs

        :param response: The response rate limited by CARTO APIs
        :type response: requests.models.Response class

        :return: Boolean
        """
        if (response.status_code == codes.too_many_requests and 'Retry-After' in response.headers and
                int(response.headers['Retry-After']) >= 0):
            return True

        return False

def __remove_trailing_zeros(self, collection):
        """Removes trailing zeroes from indexable collection of numbers"""
        index = len(collection) - 1
        while index >= 0 and collection[index] == 0:
            index -= 1

        return collection[:index + 1]

def has_changed (filename):
    """Check if filename has changed since the last check. If this
    is the first check, assume the file is changed."""
    key = os.path.abspath(filename)
    mtime = get_mtime(key)
    if key not in _mtime_cache:
        _mtime_cache[key] = mtime
        return True
    return mtime > _mtime_cache[key]

def shape(self) -> Tuple[int, ...]:
        """Shape of histogram's data.

        Returns
        -------
        One-element tuple with the number of bins along each axis.
        """
        return tuple(bins.bin_count for bins in self._binnings)

def pack_bits( longbits ):
    """Crunch a 64-bit int (8 bool bytes) into a bitfield."""
    byte = longbits & (0x0101010101010101)
    byte = (byte | (byte>>7)) & (0x0003000300030003)
    byte = (byte | (byte>>14)) & (0x0000000f0000000f)
    byte = (byte | (byte>>28)) & (0x00000000000000ff)
    return byte

def header_status(header):
    """Parse HTTP status line, return status (int) and reason."""
    status_line = header[:header.find('\r')]
    # 'HTTP/1.1 200 OK' -> (200, 'OK')
    fields = status_line.split(None, 2)
    return int(fields[1]), fields[2]

def snake_to_camel(s: str) -> str:
    """Convert string from snake case to camel case."""

    fragments = s.split('_')

    return fragments[0] + ''.join(x.title() for x in fragments[1:])

def has_key(cls, *args):
        """
        Check whether flyweight object with specified key has already been created.

        Returns:
            bool: True if already created, False if not
        """
        key = args if len(args) > 1 else args[0]
        return key in cls._instances

def array_to_npy(array_like):  # type: (np.array or Iterable or int or float) -> object
    """Convert an array like object to the NPY format.

    To understand better what an array like object is see:
    https://docs.scipy.org/doc/numpy/user/basics.creation.html#converting-python-array-like-objects-to-numpy-arrays

    Args:
        array_like (np.array or Iterable or int or float): array like object to be converted to NPY.

    Returns:
        (obj): NPY array.
    """
    buffer = BytesIO()
    np.save(buffer, array_like)
    return buffer.getvalue()

def butlast(iterable):
    """Yield all items from ``iterable`` except the last one.

    >>> list(butlast(['spam', 'eggs', 'ham']))
    ['spam', 'eggs']

    >>> list(butlast(['spam']))
    []

    >>> list(butlast([]))
    []
    """
    iterable = iter(iterable)
    try:
        first = next(iterable)
    except StopIteration:
        return
    for second in iterable:
        yield first
        first = second

def rate_limited(max_per_hour: int, *args: Any) -> Callable[..., Any]:
    """Rate limit a function."""
    return util.rate_limited(max_per_hour, *args)

def indent(text: str, num: int = 2) -> str:
    """Indent a piece of text."""
    lines = text.splitlines()
    return "\n".join(indent_iterable(lines, num=num))

def percent_of(percent, whole):
    """Calculates the value of a percent of a number
    ie: 5% of 20 is what --> 1
    
    Args:
        percent (float): The percent of a number
        whole (float): The whole of the number
        
    Returns:
        float: The value of a percent
        
    Example:
    >>> percent_of(25, 100)
    25.0
    >>> percent_of(5, 20)
    1.0
    
    """
    percent = float(percent)
    whole = float(whole)
    return (percent * whole) / 100

def guess_mimetype(filename):
    """Guesses the mimetype of a file based on the given ``filename``.

    .. code-block:: python

        >>> guess_mimetype('example.txt')
        'text/plain'
        >>> guess_mimetype('/foo/bar/example')
        'application/octet-stream'

    Parameters
    ----------
    filename : str
        The file name or path for which the mimetype is to be guessed
    """
    fn = os.path.basename(filename)
    return mimetypes.guess_type(fn)[0] or 'application/octet-stream'

def enum_mark_last(iterable, start=0):
    """
    Returns a generator over iterable that tells whether the current item is the last one.
    Usage:
        >>> iterable = range(10)
        >>> for index, is_last, item in enum_mark_last(iterable):
        >>>     print(index, item, end='\n' if is_last else ', ')
    """
    it = iter(iterable)
    count = start
    try:
        last = next(it)
    except StopIteration:
        return
    for val in it:
        yield count, False, last
        last = val
        count += 1
    yield count, True, last

def hsv2rgb_spectrum(hsv):
    """Generates RGB values from HSV values in line with a typical light
    spectrum."""
    h, s, v = hsv
    return hsv2rgb_raw(((h * 192) >> 8, s, v))

def __replace_all(repls: dict, input: str) -> str:
    """ Replaces from a string **input** all the occurrences of some
    symbols according to mapping **repls**.

    :param dict repls: where #key is the old character and
    #value is the one to substitute with;
    :param str input: original string where to apply the
    replacements;
    :return: *(str)* the string with the desired characters replaced
    """
    return re.sub('|'.join(re.escape(key) for key in repls.keys()),
                  lambda k: repls[k.group(0)], input)

def inverted_dict(d):
    """Return a dict with swapped keys and values

    >>> inverted_dict({0: ('a', 'b'), 1: 'cd'}) == {'cd': 1, ('a', 'b'): 0}
    True
    """
    return dict((force_hashable(v), k) for (k, v) in viewitems(dict(d)))

def get_tokens(line: str) -> Iterator[str]:
    """
    Yields tokens from input string.

    :param line: Input string.
    :return: Iterator over tokens.
    """
    for token in line.rstrip().split():
        if len(token) > 0:
            yield token

def getElementByWdomId(id: str) -> Optional[WebEventTarget]:
    """Get element with ``wdom_id``."""
    if not id:
        return None
    elif id == 'document':
        return get_document()
    elif id == 'window':
        return get_document().defaultView
    elm = WdomElement._elements_with_wdom_id.get(id)
    return elm

def returned(n):
	"""Generate a random walk and return True if the walker has returned to
	the origin after taking `n` steps.
	"""
	## `takei` yield lazily so we can short-circuit and avoid computing the rest of the walk
	for pos in randwalk() >> drop(1) >> takei(xrange(n-1)):
		if pos == Origin:
			return True
	return False

def _(f, x):
    """
    filter for dict, note `f` should have signature: `f::key->value->bool`
    """
    return {k: v for k, v in x.items() if f(k, v)}

def fast_median(a):
    """Fast median operation for masked array using 50th-percentile
    """
    a = checkma(a)
    #return scoreatpercentile(a.compressed(), 50)
    if a.count() > 0:
        out = np.percentile(a.compressed(), 50)
    else:
        out = np.ma.masked
    return out

def viewport_to_screen_space(framebuffer_size: vec2, point: vec4) -> vec2:
    """Transform point in viewport space to screen space."""
    return (framebuffer_size * point.xy) / point.w

def indexes_equal(a: Index, b: Index) -> bool:
    """
    Are two indexes equal? Checks by comparing ``str()`` versions of them.
    (AM UNSURE IF THIS IS ENOUGH.)
    """
    return str(a) == str(b)

def _cnx_is_empty(in_file):
    """Check if cnr or cns files are empty (only have a header)
    """
    with open(in_file) as in_handle:
        for i, line in enumerate(in_handle):
            if i > 0:
                return False
    return True

def is_running(process_id: int) -> bool:
    """
    Uses the Unix ``ps`` program to see if a process is running.
    """
    pstr = str(process_id)
    encoding = sys.getdefaultencoding()
    s = subprocess.Popen(["ps", "-p", pstr], stdout=subprocess.PIPE)
    for line in s.stdout:
        strline = line.decode(encoding)
        if pstr in strline:
            return True
    return False

def datetime_is_iso(date_str):
    """Attempts to parse a date formatted in ISO 8601 format"""
    try:
        if len(date_str) > 10:
            dt = isodate.parse_datetime(date_str)
        else:
            dt = isodate.parse_date(date_str)
        return True, []
    except:  # Any error qualifies as not ISO format
        return False, ['Datetime provided is not in a valid ISO 8601 format']

def index_exists(self, table: str, indexname: str) -> bool:
        """Does an index exist? (Specific to MySQL.)"""
        # MySQL:
        sql = ("SELECT COUNT(*) FROM information_schema.statistics"
               " WHERE table_name=? AND index_name=?")
        row = self.fetchone(sql, table, indexname)
        return True if row[0] >= 1 else False

def moving_average(arr: np.ndarray, n: int = 3) -> np.ndarray:
    """ Calculate the moving overage over an array.

    Algorithm from: https://stackoverflow.com/a/14314054

    Args:
        arr (np.ndarray): Array over which to calculate the moving average.
        n (int): Number of elements over which to calculate the moving average. Default: 3
    Returns:
        np.ndarray: Moving average calculated over n.
    """
    ret = np.cumsum(arr, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def argmax(self, rows: List[Row], column: ComparableColumn) -> List[Row]:
        """
        Takes a list of rows and a column name and returns a list containing a single row (dict from
        columns to cells) that has the maximum numerical value in the given column. We return a list
        instead of a single dict to be consistent with the return type of ``select`` and
        ``all_rows``.
        """
        if not rows:
            return []
        value_row_pairs = [(row.values[column.name], row) for row in rows]
        if not value_row_pairs:
            return []
        # Returns a list containing the row with the max cell value.
        return [sorted(value_row_pairs, key=lambda x: x[0], reverse=True)[0][1]]

def string(value) -> str:
        """ string dict/object/value to JSON """
        return system_json.dumps(Json(value).safe_object(), ensure_ascii=False)

def dotproduct(X, Y):
    """Return the sum of the element-wise product of vectors x and y.
    >>> dotproduct([1, 2, 3], [1000, 100, 10])
    1230
    """
    return sum([x * y for x, y in zip(X, Y)])

def __remove_trailing_zeros(self, collection):
        """Removes trailing zeroes from indexable collection of numbers"""
        index = len(collection) - 1
        while index >= 0 and collection[index] == 0:
            index -= 1

        return collection[:index + 1]

def genfirstvalues(cursor: Cursor, arraysize: int = 1000) \
        -> Generator[Any, None, None]:
    """
    Generate the first value in each row.

    Args:
        cursor: the cursor
        arraysize: split fetches into chunks of this many records

    Yields:
        the first value of each row
    """
    return (row[0] for row in genrows(cursor, arraysize))

def from_file(file_path) -> dict:
        """ Load JSON file """
        with io.open(file_path, 'r', encoding='utf-8') as json_stream:
            return Json.parse(json_stream, True)

def get_keys_of_max_n(dict_obj, n):
    """Returns the keys that maps to the top n max values in the given dict.

    Example:
    --------
    >>> dict_obj = {'a':2, 'b':1, 'c':5}
    >>> get_keys_of_max_n(dict_obj, 2)
    ['a', 'c']
    """
    return sorted([
        item[0]
        for item in sorted(
            dict_obj.items(), key=lambda item: item[1], reverse=True
        )[:n]
    ])

def right_replace(string, old, new, count=1):
    """
    Right replaces ``count`` occurrences of ``old`` with ``new`` in ``string``.
    For example::

        right_replace('one_two_two', 'two', 'three') -> 'one_two_three'
    """
    if not string:
        return string
    return new.join(string.rsplit(old, count))

def count(args):
    """ count occurences in a list of lists
    >>> count([['a','b'],['a']])
    defaultdict(int, {'a' : 2, 'b' : 1})
    """
    counts = defaultdict(int)
    for arg in args:
        for item in arg:
            counts[item] = counts[item] + 1
    return counts

def proper_round(n):
    """
    rounds float to closest int
    :rtype: int
    :param n: float
    """
    return int(n) + (n / abs(n)) * int(abs(n - int(n)) >= 0.5) if n != 0 else 0

def are_token_parallel(sequences: Sequence[Sized]) -> bool:
    """
    Returns True if all sequences in the list have the same length.
    """
    if not sequences or len(sequences) == 1:
        return True
    return all(len(s) == len(sequences[0]) for s in sequences)

def exclude_from(l, containing = [], equal_to = []):
    """Exclude elements in list l containing any elements from list ex.
    Example:
        >>> l = ['bob', 'r', 'rob\r', '\r\nrobert']
        >>> containing = ['\n', '\r']
        >>> equal_to = ['r']
        >>> exclude_from(l, containing, equal_to)
        ['bob']
    """
      
    cont = lambda li: any(c in li for c in containing)
    eq = lambda li: any(e == li for e in equal_to)
    return [li for li in l if not (cont(li) or eq(li))]

def valid_date(x: str) -> bool:
    """
    Retrun ``True`` if ``x`` is a valid YYYYMMDD date;
    otherwise return ``False``.
    """
    try:
        if x != dt.datetime.strptime(x, DATE_FORMAT).strftime(DATE_FORMAT):
            raise ValueError
        return True
    except ValueError:
        return False

def flatten_list(l: List[list]) -> list:
    """ takes a list of lists, l and returns a flat list
    """
    return [v for inner_l in l for v in inner_l]

def clean_all_buckets(self):
        """
        Removes all buckets from all hashes and their content.
        """
        bucket_keys = self.redis_object.keys(pattern='nearpy_*')
        if len(bucket_keys) > 0:
            self.redis_object.delete(*bucket_keys)

async def cursor(self) -> Cursor:
        """Create an aiosqlite cursor wrapping a sqlite3 cursor object."""
        return Cursor(self, await self._execute(self._conn.cursor))

def rl_get_point() -> int:  # pragma: no cover
    """
    Returns the offset of the current cursor position in rl_line_buffer
    """
    if rl_type == RlType.GNU:
        return ctypes.c_int.in_dll(readline_lib, "rl_point").value

    elif rl_type == RlType.PYREADLINE:
        return readline.rl.mode.l_buffer.point

    else:
        return 0

def to_dict(cls):
        """Make dictionary version of enumerated class.

        Dictionary created this way can be used with def_num.

        Returns:
          A dict (name) -> number
        """
        return dict((item.name, item.number) for item in iter(cls))

def is_valid(cls, arg):
        """Return True if arg is valid value for the class.  If the string
        value is wrong for the enumeration, the encoding will fail.
        """
        return (isinstance(arg, (int, long)) and (arg >= 0)) or \
            isinstance(arg, basestring)

def array2string(arr: numpy.ndarray) -> str:
        """Format numpy array as a string."""
        shape = str(arr.shape)[1:-1]
        if shape.endswith(","):
            shape = shape[:-1]
        return numpy.array2string(arr, threshold=11) + "%s[%s]" % (arr.dtype, shape)

def _str_to_list(value, separator):
    """Convert a string to a list with sanitization."""
    value_list = [item.strip() for item in value.split(separator)]
    value_list_sanitized = builtins.list(filter(None, value_list))
    if len(value_list_sanitized) > 0:
        return value_list_sanitized
    else:
        raise ValueError('Invalid list variable.')

def cmd_dot(conf: Config):
    """Print out a neat targets dependency tree based on requested targets.

    Use graphviz to render the dot file, e.g.:

    > ybt dot :foo :bar | dot -Tpng -o graph.png
    """
    build_context = BuildContext(conf)
    populate_targets_graph(build_context, conf)
    if conf.output_dot_file is None:
        write_dot(build_context, conf, sys.stdout)
    else:
        with open(conf.output_dot_file, 'w') as out_file:
            write_dot(build_context, conf, out_file)

def from_file(file_path) -> dict:
        """ Load JSON file """
        with io.open(file_path, 'r', encoding='utf-8') as json_stream:
            return Json.parse(json_stream, True)

def read_text_from_file(path: str) -> str:
    """ Reads text file contents """
    with open(path) as text_file:
        content = text_file.read()

    return content

def replaceStrs(s, *args):
    r"""Replace all ``(frm, to)`` tuples in `args` in string `s`.

    >>> replaceStrs("nothing is better than warm beer",
    ...             ('nothing','warm beer'), ('warm beer','nothing'))
    'warm beer is better than nothing'

    """
    if args == (): return s
    mapping = dict((frm, to) for frm, to in args)
    return re.sub("|".join(map(re.escape, mapping.keys())),
                  lambda match:mapping[match.group(0)], s)

def is_integer(value: Any) -> bool:
    """Return true if a value is an integer number."""
    return (isinstance(value, int) and not isinstance(value, bool)) or (
        isinstance(value, float) and isfinite(value) and int(value) == value
    )

def __replace_all(repls: dict, input: str) -> str:
    """ Replaces from a string **input** all the occurrences of some
    symbols according to mapping **repls**.

    :param dict repls: where #key is the old character and
    #value is the one to substitute with;
    :param str input: original string where to apply the
    replacements;
    :return: *(str)* the string with the desired characters replaced
    """
    return re.sub('|'.join(re.escape(key) for key in repls.keys()),
                  lambda k: repls[k.group(0)], input)

def get_column_names(engine: Engine, tablename: str) -> List[str]:
    """
    Get all the database column names for the specified table.
    """
    return [info.name for info in gen_columns_info(engine, tablename)]

def __replace_all(repls: dict, input: str) -> str:
    """ Replaces from a string **input** all the occurrences of some
    symbols according to mapping **repls**.

    :param dict repls: where #key is the old character and
    #value is the one to substitute with;
    :param str input: original string where to apply the
    replacements;
    :return: *(str)* the string with the desired characters replaced
    """
    return re.sub('|'.join(re.escape(key) for key in repls.keys()),
                  lambda k: repls[k.group(0)], input)

def split(text: str) -> List[str]:
    """Split a text into a list of tokens.

    :param text: the text to split
    :return: tokens
    """
    return [word for word in SEPARATOR.split(text) if word.strip(' \t')]

def remove_empty_text(utterances: List[Utterance]) -> List[Utterance]:
    """Remove empty utterances from a list of utterances
    Args:
        utterances: The list of utterance we are processing
    """
    return [utter for utter in utterances if utter.text.strip() != ""]

def csv_to_numpy(string_like, dtype=None):  # type: (str) -> np.array
    """Convert a CSV object to a numpy array.

    Args:
        string_like (str): CSV string.
        dtype (dtype, optional):  Data type of the resulting array. If None, the dtypes will be determined by the
                                        contents of each column, individually. This argument can only be used to
                                        'upcast' the array.  For downcasting, use the .astype(t) method.
    Returns:
        (np.array): numpy array
    """
    stream = StringIO(string_like)
    return np.genfromtxt(stream, dtype=dtype, delimiter=',')

def remove_links(text):
    """
    Helper function to remove the links from the input text

    Args:
        text (str): A string

    Returns:
        str: the same text, but with any substring that matches the regex
        for a link removed and replaced with a space

    Example:
        >>> from tweet_parser.getter_methods.tweet_text import remove_links
        >>> text = "lorem ipsum dolor https://twitter.com/RobotPrincessFi"
        >>> remove_links(text)
        'lorem ipsum dolor  '
    """
    tco_link_regex = re.compile("https?://t.co/[A-z0-9].*")
    generic_link_regex = re.compile("(https?://)?(\w*[.]\w+)+([/?=&]+\w+)*")
    remove_tco = re.sub(tco_link_regex, " ", text)
    remove_generic = re.sub(generic_link_regex, " ", remove_tco)
    return remove_generic

def is_relative_url(url):
    """ simple method to determine if a url is relative or absolute """
    if url.startswith("#"):
        return None
    if url.find("://") > 0 or url.startswith("//"):
        # either 'http(s)://...' or '//cdn...' and therefore absolute
        return False
    return True

def array2string(arr: numpy.ndarray) -> str:
        """Format numpy array as a string."""
        shape = str(arr.shape)[1:-1]
        if shape.endswith(","):
            shape = shape[:-1]
        return numpy.array2string(arr, threshold=11) + "%s[%s]" % (arr.dtype, shape)

def mmap(func, iterable):
    """Wrapper to make map() behave the same on Py2 and Py3."""

    if sys.version_info[0] > 2:
        return [i for i in map(func, iterable)]
    else:
        return map(func, iterable)

def looks_like_url(url):
    """ Simplified check to see if the text appears to be a URL.

    Similar to `urlparse` but much more basic.

    Returns:
      True if the url str appears to be valid.
      False otherwise.

    >>> url = looks_like_url("totalgood.org")
    >>> bool(url)
    True
    """
    if not isinstance(url, basestring):
        return False
    if not isinstance(url, basestring) or len(url) >= 1024 or not cre_url.match(url):
        return False
    return True

def issubset(self, other):
        """
        Report whether another set contains this set.

        Example:
            >>> OrderedSet([1, 2, 3]).issubset({1, 2})
            False
            >>> OrderedSet([1, 2, 3]).issubset({1, 2, 3, 4})
            True
            >>> OrderedSet([1, 2, 3]).issubset({1, 4, 3, 5})
            False
        """
        if len(self) > len(other):  # Fast check for obvious cases
            return False
        return all(item in other for item in self)

def clean_map(obj: Mapping[Any, Any]) -> Mapping[Any, Any]:
    """
    Return a new copied dictionary without the keys with ``None`` values from
    the given Mapping object.
    """
    return {k: v for k, v in obj.items() if v is not None}

def process_literal_param(self, value: Optional[List[int]],
                              dialect: Dialect) -> str:
        """Convert things on the way from Python to the database."""
        retval = self._intlist_to_dbstr(value)
        return retval

def remove_once(gset, elem):
    """Remove the element from a set, lists or dict.
    
        >>> L = ["Lucy"]; S = set(["Sky"]); D = { "Diamonds": True };
        >>> remove_once(L, "Lucy"); remove_once(S, "Sky"); remove_once(D, "Diamonds");
        >>> print L, S, D
        [] set([]) {}

    Returns the element if it was removed. Raises one of the exceptions in 
    :obj:`RemoveError` otherwise.
    """
    remove = getattr(gset, 'remove', None)
    if remove is not None: remove(elem)
    else: del gset[elem]
    return elem

def astensor(array: TensorLike) -> BKTensor:
    """Covert numpy array to tensorflow tensor"""
    tensor = tf.convert_to_tensor(value=array, dtype=CTYPE)
    return tensor

def validate_django_compatible_with_python():
    """
    Verify Django 1.11 is present if Python 2.7 is active

    Installation of pinax-cli requires the correct version of Django for
    the active Python version. If the developer subsequently changes
    the Python version the installed Django may no longer be compatible.
    """
    python_version = sys.version[:5]
    django_version = django.get_version()
    if sys.version_info == (2, 7) and django_version >= "2":
        click.BadArgumentUsage("Please install Django v1.11 for Python {}, or switch to Python >= v3.4".format(python_version))

def _parse_tuple_string(argument):
        """ Return a tuple from parsing 'a,b,c,d' -> (a,b,c,d) """
        if isinstance(argument, str):
            return tuple(int(p.strip()) for p in argument.split(','))
        return argument

def SwitchToThisWindow(handle: int) -> None:
    """
    SwitchToThisWindow from Win32.
    handle: int, the handle of a native window.
    """
    ctypes.windll.user32.SwitchToThisWindow(ctypes.c_void_p(handle), 1)

def find_first_in_list(txt: str, str_list: [str]) -> int:  # type: ignore
    """
    Returns the index of the earliest occurence of an item from a list in a string

    Ex: find_first_in_list('foobar', ['bar', 'fin']) -> 3
    """
    start = len(txt) + 1
    for item in str_list:
        if start > txt.find(item) > -1:
            start = txt.find(item)
    return start if len(txt) + 1 > start > -1 else -1

def is_unicode(string):
    """Validates that the object itself is some kinda string"""
    str_type = str(type(string))

    if str_type.find('str') > 0 or str_type.find('unicode') > 0:
        return True

    return False

def read_set_from_file(filename: str) -> Set[str]:
    """
    Extract a de-duped collection (set) of text from a file.
    Expected file format is one item per line.
    """
    collection = set()
    with open(filename, 'r') as file_:
        for line in file_:
            collection.add(line.rstrip())
    return collection

def dict_of_sets_add(dictionary, key, value):
    # type: (DictUpperBound, Any, Any) -> None
    """Add value to a set in a dictionary by key

    Args:
        dictionary (DictUpperBound): Dictionary to which to add values
        key (Any): Key within dictionary
        value (Any): Value to add to set in dictionary

    Returns:
        None

    """
    set_objs = dictionary.get(key, set())
    set_objs.add(value)
    dictionary[key] = set_objs

def attrname_to_colname_dict(cls) -> Dict[str, str]:
    """
    Asks an SQLAlchemy class how its attribute names correspond to database
    column names.

    Args:
        cls: SQLAlchemy ORM class

    Returns:
        a dictionary mapping attribute names to database column names
    """
    attr_col = {}  # type: Dict[str, str]
    for attrname, column in gen_columns(cls):
        attr_col[attrname] = column.name
    return attr_col

def get_column_names(engine: Engine, tablename: str) -> List[str]:
    """
    Get all the database column names for the specified table.
    """
    return [info.name for info in gen_columns_info(engine, tablename)]

def Exit(msg, code=1):
    """Exit execution with return code and message
    :param msg: Message displayed prior to exit
    :param code: code returned upon exiting
    """
    print >> sys.stderr, msg
    sys.exit(code)

def dag_longest_path(graph, source, target):
    """
    Finds the longest path in a dag between two nodes
    """
    if source == target:
        return [source]
    allpaths = nx.all_simple_paths(graph, source, target)
    longest_path = []
    for l in allpaths:
        if len(l) > len(longest_path):
            longest_path = l
    return longest_path

def hsv2rgb_spectrum(hsv):
    """Generates RGB values from HSV values in line with a typical light
    spectrum."""
    h, s, v = hsv
    return hsv2rgb_raw(((h * 192) >> 8, s, v))

def __as_list(value: List[JsonObjTypes]) -> List[JsonTypes]:
        """ Return a json array as a list

        :param value: array
        :return: array with JsonObj instances removed
        """
        return [e._as_dict if isinstance(e, JsonObj) else e for e in value]

def strip_codes(s: Any) -> str:
    """ Strip all color codes from a string.
        Returns empty string for "falsey" inputs.
    """
    return codepat.sub('', str(s) if (s or (s == 0)) else '')

def get_last_day_of_month(t: datetime) -> int:
    """
    Returns day number of the last day of the month
    :param t: datetime
    :return: int
    """
    tn = t + timedelta(days=32)
    tn = datetime(year=tn.year, month=tn.month, day=1)
    tt = tn - timedelta(hours=1)
    return tt.day

def get_period_last_3_months() -> str:
    """ Returns the last week as a period string """
    today = Datum()
    today.today()

    # start_date = today - timedelta(weeks=13)
    start_date = today.clone()
    start_date.subtract_months(3)

    period = get_period(start_date.date, today.date)
    return period

def clean_column_names(df: DataFrame) -> DataFrame:
    """
    Strip the whitespace from all column names in the given DataFrame
    and return the result.
    """
    f = df.copy()
    f.columns = [col.strip() for col in f.columns]
    return f

def trade_day(dt, cal='US'):
    """
    Latest trading day w.r.t given dt

    Args:
        dt: date of reference
        cal: trading calendar

    Returns:
        pd.Timestamp: last trading day

    Examples:
        >>> trade_day('2018-12-25').strftime('%Y-%m-%d')
        '2018-12-24'
    """
    from xone import calendar

    dt = pd.Timestamp(dt).date()
    return calendar.trading_dates(start=dt - pd.Timedelta('10D'), end=dt, calendar=cal)[-1]

def get_now_utc_notz_datetime() -> datetime.datetime:
    """
    Get the UTC time now, but with no timezone information,
    in :class:`datetime.datetime` format.
    """
    now = datetime.datetime.utcnow()
    return now.replace(tzinfo=None)

def file_exists(fname):
    """Check if a file exists and is non-empty.
    """
    try:
        return fname and os.path.exists(fname) and os.path.getsize(fname) > 0
    except OSError:
        return False

def numpy_to_yaml(representer: Representer, data: np.ndarray) -> Sequence[Any]:
    """ Write a numpy array to YAML.

    It registers the array under the tag ``!numpy_array``.

    Use with:

    .. code-block:: python

        >>> yaml = ruamel.yaml.YAML()
        >>> yaml.representer.add_representer(np.ndarray, yaml.numpy_to_yaml)

    Note:
        We cannot use ``yaml.register_class`` because it won't register the proper type.
        (It would register the type of the class, rather than of `numpy.ndarray`). Instead,
        we use the above approach to register this method explicitly with the representer.
    """
    return representer.represent_sequence(
        "!numpy_array",
        data.tolist()
    )

def read_set_from_file(filename: str) -> Set[str]:
    """
    Extract a de-duped collection (set) of text from a file.
    Expected file format is one item per line.
    """
    collection = set()
    with open(filename, 'r') as file_:
        for line in file_:
            collection.add(line.rstrip())
    return collection

def looks_like_url(url):
    """ Simplified check to see if the text appears to be a URL.

    Similar to `urlparse` but much more basic.

    Returns:
      True if the url str appears to be valid.
      False otherwise.

    >>> url = looks_like_url("totalgood.org")
    >>> bool(url)
    True
    """
    if not isinstance(url, basestring):
        return False
    if not isinstance(url, basestring) or len(url) >= 1024 or not cre_url.match(url):
        return False
    return True

def clean_all_buckets(self):
        """
        Removes all buckets from all hashes and their content.
        """
        bucket_keys = self.redis_object.keys(pattern='nearpy_*')
        if len(bucket_keys) > 0:
            self.redis_object.delete(*bucket_keys)

def normalize(numbers):
    """Multiply each number by a constant such that the sum is 1.0
    >>> normalize([1,2,1])
    [0.25, 0.5, 0.25]
    """
    total = float(sum(numbers))
    return [n / total for n in numbers]

def _skip_section(self):
        """Skip a section"""
        self._last = self._f.readline()
        while len(self._last) > 0 and len(self._last[0].strip()) == 0:
            self._last = self._f.readline()

def get_now_sql_datetime():
    """
    *A datetime stamp in MySQL format: ``YYYY-MM-DDTHH:MM:SS``*

    **Return:**
        - ``now`` -- current time and date in MySQL format

    **Usage:**
        .. code-block:: python 

            from fundamentals import times
            now = times.get_now_sql_datetime()
            print now

            # OUT: 2016-03-18T11:08:23 
    """
    ## > IMPORTS ##
    from datetime import datetime, date, time
    now = datetime.now()
    now = now.strftime("%Y-%m-%dT%H:%M:%S")

    return now

def is_end_of_month(self) -> bool:
        """ Checks if the date is at the end of the month """
        end_of_month = Datum()
        # get_end_of_month(value)
        end_of_month.end_of_month()
        return self.value == end_of_month.value

def uuid2buid(value):
    """
    Convert a UUID object to a 22-char BUID string

    >>> u = uuid.UUID('33203dd2-f2ef-422f-aeb0-058d6f5f7089')
    >>> uuid2buid(u)
    'MyA90vLvQi-usAWNb19wiQ'
    """
    if six.PY3:  # pragma: no cover
        return urlsafe_b64encode(value.bytes).decode('utf-8').rstrip('=')
    else:
        return six.text_type(urlsafe_b64encode(value.bytes).rstrip('='))

def is_not_null(df: DataFrame, col_name: str) -> bool:
    """
    Return ``True`` if the given DataFrame has a column of the given
    name (string), and there exists at least one non-NaN value in that
    column; return ``False`` otherwise.
    """
    if (
        isinstance(df, pd.DataFrame)
        and col_name in df.columns
        and df[col_name].notnull().any()
    ):
        return True
    else:
        return False

def suppress_stdout():
    """
    Context manager that suppresses stdout.

    Examples:
        >>> with suppress_stdout():
        ...     print('Test print')

        >>> print('test')
        test

    """
    save_stdout = sys.stdout
    sys.stdout = DevNull()
    yield
    sys.stdout = save_stdout

def encode_list(key, list_):
    # type: (str, Iterable) -> Dict[str, str]
    """
    Converts a list into a space-separated string and puts it in a dictionary

    :param key: Dictionary key to store the list
    :param list_: A list of objects
    :return: A dictionary key->string or an empty dictionary
    """
    if not list_:
        return {}
    return {key: " ".join(str(i) for i in list_)}

def gen_lower(x: Iterable[str]) -> Generator[str, None, None]:
    """
    Args:
        x: iterable of strings

    Yields:
        each string in lower case
    """
    for string in x:
        yield string.lower()

def inverted_dict(d):
    """Return a dict with swapped keys and values

    >>> inverted_dict({0: ('a', 'b'), 1: 'cd'}) == {'cd': 1, ('a', 'b'): 0}
    True
    """
    return dict((force_hashable(v), k) for (k, v) in viewitems(dict(d)))

def remove_links(text):
    """
    Helper function to remove the links from the input text

    Args:
        text (str): A string

    Returns:
        str: the same text, but with any substring that matches the regex
        for a link removed and replaced with a space

    Example:
        >>> from tweet_parser.getter_methods.tweet_text import remove_links
        >>> text = "lorem ipsum dolor https://twitter.com/RobotPrincessFi"
        >>> remove_links(text)
        'lorem ipsum dolor  '
    """
    tco_link_regex = re.compile("https?://t.co/[A-z0-9].*")
    generic_link_regex = re.compile("(https?://)?(\w*[.]\w+)+([/?=&]+\w+)*")
    remove_tco = re.sub(tco_link_regex, " ", text)
    remove_generic = re.sub(generic_link_regex, " ", remove_tco)
    return remove_generic

def decode_value(stream):
    """Decode the contents of a value from a serialized stream.

    :param stream: Source data stream
    :type stream: io.BytesIO
    :returns: Decoded value
    :rtype: bytes
    """
    length = decode_length(stream)
    (value,) = unpack_value(">{:d}s".format(length), stream)
    return value

def fcast(value: float) -> TensorLike:
    """Cast to float tensor"""
    newvalue = tf.cast(value, FTYPE)
    if DEVICE == 'gpu':
        newvalue = newvalue.gpu()  # Why is this needed?  # pragma: no cover
    return newvalue

def snake_to_camel(s: str) -> str:
    """Convert string from snake case to camel case."""

    fragments = s.split('_')

    return fragments[0] + ''.join(x.title() for x in fragments[1:])

def check_valid(number, input_base=10):
    """
    Checks if there is an invalid digit in the input number.

    Args:
        number: An number in the following form:
            (int, int, int, ... , '.' , int, int, int)
            (iterable container) containing positive integers of the input base
        input_base(int): The base of the input number.

    Returns:
        bool, True if all digits valid, else False.

    Examples:
        >>> check_valid((1,9,6,'.',5,1,6), 12)
        True
        >>> check_valid((8,1,15,9), 15)
        False
    """
    for n in number:
        if n in (".", "[", "]"):
            continue
        elif n >= input_base:
            if n == 1 and input_base == 1:
                continue
            else:
                return False
    return True

def ResetConsoleColor() -> bool:
    """
    Reset to the default text color on console window.
    Return bool, True if succeed otherwise False.
    """
    if sys.stdout:
        sys.stdout.flush()
    bool(ctypes.windll.kernel32.SetConsoleTextAttribute(_ConsoleOutputHandle, _DefaultConsoleColor))

def get_timezone() -> Tuple[datetime.tzinfo, str]:
    """Discover the current time zone and it's standard string representation (for source{d})."""
    dt = get_datetime_now().astimezone()
    tzstr = dt.strftime("%z")
    tzstr = tzstr[:-2] + ":" + tzstr[-2:]
    return dt.tzinfo, tzstr

def upsert_multi(db, collection, object, match_params=None):
        """
        Wrapper for pymongo.insert_many() and update_many()
        :param db: db connection
        :param collection: collection to update
        :param object: the modifications to apply
        :param match_params: a query that matches the documents to update
        :return: ids of inserted/updated document
        """
        if isinstance(object, list) and len(object) > 0:
            return str(db[collection].insert_many(object).inserted_ids)
        elif isinstance(object, dict):
            return str(db[collection].update_many(match_params, {"$set": object}, upsert=False).upserted_id)

def get_language():
    """
    Wrapper around Django's `get_language` utility.
    For Django >= 1.8, `get_language` returns None in case no translation is activate.
    Here we patch this behavior e.g. for back-end functionality requiring access to translated fields
    """
    from parler import appsettings
    language = dj_get_language()
    if language is None and appsettings.PARLER_DEFAULT_ACTIVATE:
        return appsettings.PARLER_DEFAULT_LANGUAGE_CODE
    else:
        return language

def url_concat(url, args):
    """Concatenate url and argument dictionary regardless of whether
    url has existing query parameters.

    >>> url_concat("http://example.com/foo?a=b", dict(c="d"))
    'http://example.com/foo?a=b&c=d'
    """
    if not args: return url
    if url[-1] not in ('?', '&'):
        url += '&' if ('?' in url) else '?'
    return url + urllib.urlencode(args)

def get_system_flags() -> FrozenSet[Flag]:
    """Return the set of implemented system flags."""
    return frozenset({Seen, Recent, Deleted, Flagged, Answered, Draft})

def chars(string: any) -> str:
        """Return all (and only) the chars in the given string."""
        return ''.join([c if c.isalpha() else '' for c in str(string)])

def count(args):
    """ count occurences in a list of lists
    >>> count([['a','b'],['a']])
    defaultdict(int, {'a' : 2, 'b' : 1})
    """
    counts = defaultdict(int)
    for arg in args:
        for item in arg:
            counts[item] = counts[item] + 1
    return counts

def chars(string: any) -> str:
        """Return all (and only) the chars in the given string."""
        return ''.join([c if c.isalpha() else '' for c in str(string)])

def returned(n):
	"""Generate a random walk and return True if the walker has returned to
	the origin after taking `n` steps.
	"""
	## `takei` yield lazily so we can short-circuit and avoid computing the rest of the walk
	for pos in randwalk() >> drop(1) >> takei(xrange(n-1)):
		if pos == Origin:
			return True
	return False

def memory_full():
    """Check if the memory is too full for further caching."""
    current_process = psutil.Process(os.getpid())
    return (current_process.memory_percent() >
            config.MAXIMUM_CACHE_MEMORY_PERCENTAGE)

def dictlist_wipe_key(dict_list: Iterable[Dict], key: str) -> None:
    """
    Process an iterable of dictionaries. For each dictionary ``d``, delete
    ``d[key]`` if it exists.
    """
    for d in dict_list:
        d.pop(key, None)

def flatten_list(l: List[list]) -> list:
    """ takes a list of lists, l and returns a flat list
    """
    return [v for inner_l in l for v in inner_l]

def process_literal_param(self, value: Optional[List[int]],
                              dialect: Dialect) -> str:
        """Convert things on the way from Python to the database."""
        retval = self._intlist_to_dbstr(value)
        return retval

def __rmatmul__(self, other):
        """
        Matrix multiplication using binary `@` operator in Python>=3.5.
        """
        return self.T.dot(np.transpose(other)).T

def non_increasing(values):
    """True if values are not increasing."""
    return all(x >= y for x, y in zip(values, values[1:]))

def _duplicates(list_):
    """Return dict mapping item -> indices."""
    item_indices = {}
    for i, item in enumerate(list_):
        try:
            item_indices[item].append(i)
        except KeyError:  # First time seen
            item_indices[item] = [i]
    return item_indices

def __replace_all(repls: dict, input: str) -> str:
    """ Replaces from a string **input** all the occurrences of some
    symbols according to mapping **repls**.

    :param dict repls: where #key is the old character and
    #value is the one to substitute with;
    :param str input: original string where to apply the
    replacements;
    :return: *(str)* the string with the desired characters replaced
    """
    return re.sub('|'.join(re.escape(key) for key in repls.keys()),
                  lambda k: repls[k.group(0)], input)

def rate_limited(max_per_hour: int, *args: Any) -> Callable[..., Any]:
    """Rate limit a function."""
    return util.rate_limited(max_per_hour, *args)

def codes_get_size(handle, key):
    # type: (cffi.FFI.CData, str) -> int
    """
    Get the number of coded value from a key.
    If several keys of the same name are present, the total sum is returned.

    :param bytes key: the keyword to get the size of

    :rtype: int
    """
    size = ffi.new('size_t *')
    _codes_get_size(handle, key.encode(ENC), size)
    return size[0]

def setup_cache(app: Flask, cache_config) -> Optional[Cache]:
    """Setup the flask-cache on a flask app"""
    if cache_config and cache_config.get('CACHE_TYPE') != 'null':
        return Cache(app, config=cache_config)

    return None

def truncate(value: Decimal, n_digits: int) -> Decimal:
    """Truncates a value to a number of decimals places"""
    return Decimal(math.trunc(value * (10 ** n_digits))) / (10 ** n_digits)

def call_fset(self, obj, value) -> None:
        """Store the given custom value and call the setter function."""
        vars(obj)[self.name] = self.fset(obj, value)

def count(args):
    """ count occurences in a list of lists
    >>> count([['a','b'],['a']])
    defaultdict(int, {'a' : 2, 'b' : 1})
    """
    counts = defaultdict(int)
    for arg in args:
        for item in arg:
            counts[item] = counts[item] + 1
    return counts

def src2ast(src: str) -> Expression:
    """Return ast.Expression created from source code given in `src`."""
    try:
        return ast.parse(src, mode='eval')
    except SyntaxError:
        raise ValueError("Not a valid expression.") from None

def indexes_equal(a: Index, b: Index) -> bool:
    """
    Are two indexes equal? Checks by comparing ``str()`` versions of them.
    (AM UNSURE IF THIS IS ENOUGH.)
    """
    return str(a) == str(b)

def listify(a):
    """
    Convert a scalar ``a`` to a list and all iterables to list as well.

    Examples
    --------
    >>> listify(0)
    [0]

    >>> listify([1,2,3])
    [1, 2, 3]

    >>> listify('a')
    ['a']

    >>> listify(np.array([1,2,3]))
    [1, 2, 3]

    >>> listify('string')
    ['string']
    """
    if a is None:
        return []
    elif not isinstance(a, (tuple, list, np.ndarray)):
        return [a]
    return list(a)

def inverted_dict(d):
    """Return a dict with swapped keys and values

    >>> inverted_dict({0: ('a', 'b'), 1: 'cd'}) == {'cd': 1, ('a', 'b'): 0}
    True
    """
    return dict((force_hashable(v), k) for (k, v) in viewitems(dict(d)))

def string(value) -> str:
        """ string dict/object/value to JSON """
        return system_json.dumps(Json(value).safe_object(), ensure_ascii=False)

def recClearTag(element):
    """Applies maspy.xml.clearTag() to the tag attribute of the "element" and
    recursively to all child elements.

    :param element: an :instance:`xml.etree.Element`
    """
    children = element.getchildren()
    if len(children) > 0:
        for child in children:
            recClearTag(child)
    element.tag = clearTag(element.tag)

def get_table_names_from_metadata(metadata: MetaData) -> List[str]:
    """
    Returns all database table names found in an SQLAlchemy :class:`MetaData`
    object.
    """
    return [table.name for table in metadata.tables.values()]

def clean_column_names(df: DataFrame) -> DataFrame:
    """
    Strip the whitespace from all column names in the given DataFrame
    and return the result.
    """
    f = df.copy()
    f.columns = [col.strip() for col in f.columns]
    return f

def signed_distance(mesh, points):
    """
    Find the signed distance from a mesh to a list of points.

    * Points OUTSIDE the mesh will have NEGATIVE distance
    * Points within tol.merge of the surface will have POSITIVE distance
    * Points INSIDE the mesh will have POSITIVE distance

    Parameters
    -----------
    mesh   : Trimesh object
    points : (n,3) float, list of points in space

    Returns
    ----------
    signed_distance : (n,3) float, signed distance from point to mesh
    """
    # make sure we have a numpy array
    points = np.asanyarray(points, dtype=np.float64)

    # find the closest point on the mesh to the queried points
    closest, distance, triangle_id = closest_point(mesh, points)

    # we only care about nonzero distances
    nonzero = distance > tol.merge

    if not nonzero.any():
        return distance

    inside = mesh.ray.contains_points(points[nonzero])
    sign = (inside.astype(int) * 2) - 1

    # apply sign to previously computed distance
    distance[nonzero] *= sign

    return distance

def butlast(iterable):
    """Yield all items from ``iterable`` except the last one.

    >>> list(butlast(['spam', 'eggs', 'ham']))
    ['spam', 'eggs']

    >>> list(butlast(['spam']))
    []

    >>> list(butlast([]))
    []
    """
    iterable = iter(iterable)
    try:
        first = next(iterable)
    except StopIteration:
        return
    for second in iterable:
        yield first
        first = second

def _kbhit_unix() -> bool:
    """
    Under UNIX: is a keystroke available?
    """
    dr, dw, de = select.select([sys.stdin], [], [], 0)
    return dr != []

def replace_in_list(stringlist: Iterable[str],
                    replacedict: Dict[str, str]) -> List[str]:
    """
    Returns a list produced by applying :func:`multiple_replace` to every
    string in ``stringlist``.

    Args:
        stringlist: list of source strings
        replacedict: dictionary mapping "original" to "replacement" strings

    Returns:
        list of final strings

    """
    newlist = []
    for fromstring in stringlist:
        newlist.append(multiple_replace(fromstring, replacedict))
    return newlist

def get_creation_date(
            self,
            bucket: str,
            key: str,
    ) -> datetime.datetime:
        """
        Retrieves the creation date for a given key in a given bucket.
        :param bucket: the bucket the object resides in.
        :param key: the key of the object for which the creation date is being retrieved.
        :return: the creation date
        """
        blob_obj = self._get_blob_obj(bucket, key)
        return blob_obj.time_created

def __replace_all(repls: dict, input: str) -> str:
    """ Replaces from a string **input** all the occurrences of some
    symbols according to mapping **repls**.

    :param dict repls: where #key is the old character and
    #value is the one to substitute with;
    :param str input: original string where to apply the
    replacements;
    :return: *(str)* the string with the desired characters replaced
    """
    return re.sub('|'.join(re.escape(key) for key in repls.keys()),
                  lambda k: repls[k.group(0)], input)

def csv_to_numpy(string_like, dtype=None):  # type: (str) -> np.array
    """Convert a CSV object to a numpy array.

    Args:
        string_like (str): CSV string.
        dtype (dtype, optional):  Data type of the resulting array. If None, the dtypes will be determined by the
                                        contents of each column, individually. This argument can only be used to
                                        'upcast' the array.  For downcasting, use the .astype(t) method.
    Returns:
        (np.array): numpy array
    """
    stream = StringIO(string_like)
    return np.genfromtxt(stream, dtype=dtype, delimiter=',')

def file_exists(fname):
    """Check if a file exists and is non-empty.
    """
    try:
        return fname and os.path.exists(fname) and os.path.getsize(fname) > 0
    except OSError:
        return False

def get_last_day_of_month(t: datetime) -> int:
    """
    Returns day number of the last day of the month
    :param t: datetime
    :return: int
    """
    tn = t + timedelta(days=32)
    tn = datetime(year=tn.year, month=tn.month, day=1)
    tt = tn - timedelta(hours=1)
    return tt.day

def safe_pow(base, exp):
    """safe version of pow"""
    if exp > MAX_EXPONENT:
        raise RuntimeError("Invalid exponent, max exponent is {}".format(MAX_EXPONENT))
    return base ** exp

def is_finite(value: Any) -> bool:
    """Return true if a value is a finite number."""
    return isinstance(value, int) or (isinstance(value, float) and isfinite(value))

def strtobytes(input, encoding):
    """Take a str and transform it into a byte array."""
    py_version = sys.version_info[0]
    if py_version >= 3:
        return _strtobytes_py3(input, encoding)
    return _strtobytes_py2(input, encoding)

def is_empty_shape(sh: ShExJ.Shape) -> bool:
        """ Determine whether sh has any value """
        return sh.closed is None and sh.expression is None and sh.extra is None and \
            sh.semActs is None

def chars(string: any) -> str:
        """Return all (and only) the chars in the given string."""
        return ''.join([c if c.isalpha() else '' for c in str(string)])

def get_day_name(self) -> str:
        """ Returns the day name """
        weekday = self.value.isoweekday() - 1
        return calendar.day_name[weekday]

def _my_hash(arg_list):
    # type: (List[Any]) -> int
    """Simple helper hash function"""
    res = 0
    for arg in arg_list:
        res = res * 31 + hash(arg)
    return res

def proper_round(n):
    """
    rounds float to closest int
    :rtype: int
    :param n: float
    """
    return int(n) + (n / abs(n)) * int(abs(n - int(n)) >= 0.5) if n != 0 else 0

def year(date):
    """ Returns the year.

    :param date:
        The string date with this format %m/%d/%Y
    :type date:
        String

    :returns:
        int

    :example:
        >>> year('05/1/2015')
        2015
    """
    try:
        fmt = '%m/%d/%Y'
        return datetime.strptime(date, fmt).timetuple().tm_year
    except ValueError:
        return 0

def astensor(array: TensorLike) -> BKTensor:
    """Covert numpy array to tensorflow tensor"""
    tensor = tf.convert_to_tensor(value=array, dtype=CTYPE)
    return tensor

def MoveWindow(handle: int, x: int, y: int, width: int, height: int, repaint: int = 1) -> bool:
    """
    MoveWindow from Win32.
    handle: int, the handle of a native window.
    x: int.
    y: int.
    width: int.
    height: int.
    repaint: int, use 1 or 0.
    Return bool, True if succeed otherwise False.
    """
    return bool(ctypes.windll.user32.MoveWindow(ctypes.c_void_p(handle), x, y, width, height, repaint))

def is_prime(n):
    """
    Check if n is a prime number
    """
    if n % 2 == 0 and n > 2:
        return False
    return all(n % i for i in range(3, int(math.sqrt(n)) + 1, 2))

def try_cast_int(s):
    """(str) -> int
    All the digits in a given string are concatenated and converted into a single number.
    """
    try:
        temp = re.findall('\d', str(s))
        temp = ''.join(temp)
        return int(temp)
    except:
        return s

def iter_lines(file_like: Iterable[str]) -> Generator[str, None, None]:
    """ Helper for iterating only nonempty lines without line breaks"""
    for line in file_like:
        line = line.rstrip('\r\n')
        if line:
            yield line

def to_0d_array(value: Any) -> np.ndarray:
    """Given a value, wrap it in a 0-D numpy.ndarray.
    """
    if np.isscalar(value) or (isinstance(value, np.ndarray) and
                              value.ndim == 0):
        return np.array(value)
    else:
        return to_0d_object_array(value)

def valid_date(x: str) -> bool:
    """
    Retrun ``True`` if ``x`` is a valid YYYYMMDD date;
    otherwise return ``False``.
    """
    try:
        if x != dt.datetime.strptime(x, DATE_FORMAT).strftime(DATE_FORMAT):
            raise ValueError
        return True
    except ValueError:
        return False

def array2string(arr: numpy.ndarray) -> str:
        """Format numpy array as a string."""
        shape = str(arr.shape)[1:-1]
        if shape.endswith(","):
            shape = shape[:-1]
        return numpy.array2string(arr, threshold=11) + "%s[%s]" % (arr.dtype, shape)

def get_last_day_of_month(t: datetime) -> int:
    """
    Returns day number of the last day of the month
    :param t: datetime
    :return: int
    """
    tn = t + timedelta(days=32)
    tn = datetime(year=tn.year, month=tn.month, day=1)
    tt = tn - timedelta(hours=1)
    return tt.day

def clip_to_seconds(m: Union[int, pd.Series]) -> Union[int, pd.Series]:
        """Clips UTC datetime in nanoseconds to seconds."""
        return m // pd.Timedelta(1, unit='s').value

def dict_to_enum_fn(d: Dict[str, Any], enum_class: Type[Enum]) -> Enum:
    """
    Converts an ``dict`` to a ``Enum``.
    """
    return enum_class[d['name']]

def setlocale(name):
    """
    Context manager with threading lock for set locale on enter, and set it
    back to original state on exit.

    ::

        >>> with setlocale("C"):
        ...     ...
    """
    with LOCALE_LOCK:
        old_locale = locale.setlocale(locale.LC_ALL)
        try:
            yield locale.setlocale(locale.LC_ALL, name)
        finally:
            locale.setlocale(locale.LC_ALL, old_locale)

def to_javascript_(self, table_name: str="data") -> str:
        """Convert the main dataframe to javascript code

        :param table_name: javascript variable name, defaults to "data"
        :param table_name: str, optional
        :return: a javascript constant with the data
        :rtype: str

        :example: ``ds.to_javastript_("myconst")``
        """
        try:
            renderer = pytablewriter.JavaScriptTableWriter
            data = self._build_export(renderer, table_name)
            return data
        except Exception as e:
            self.err(e, "Can not convert data to javascript code")

def file_exists(fname):
    """Check if a file exists and is non-empty.
    """
    try:
        return fname and os.path.exists(fname) and os.path.getsize(fname) > 0
    except OSError:
        return False

def write_text(filename: str, text: str) -> None:
    """
    Writes text to a file.
    """
    with open(filename, 'w') as f:  # type: TextIO
        print(text, file=f)

def isfile_notempty(inputfile: str) -> bool:
        """Check if the input filename with path is a file and is not empty."""
        try:
            return isfile(inputfile) and getsize(inputfile) > 0
        except TypeError:
            raise TypeError('inputfile is not a valid type')

def availability_pdf() -> bool:
    """
    Is a PDF-to-text tool available?
    """
    pdftotext = tools['pdftotext']
    if pdftotext:
        return True
    elif pdfminer:
        log.warning("PDF conversion: pdftotext missing; "
                    "using pdfminer (less efficient)")
        return True
    else:
        return False

def _duplicates(list_):
    """Return dict mapping item -> indices."""
    item_indices = {}
    for i, item in enumerate(list_):
        try:
            item_indices[item].append(i)
        except KeyError:  # First time seen
            item_indices[item] = [i]
    return item_indices

def fetchallfirstvalues(self, sql: str, *args) -> List[Any]:
        """Executes SQL; returns list of first values of each row."""
        rows = self.fetchall(sql, *args)
        return [row[0] for row in rows]

def remove_blank_lines(string):
    """ Removes all blank lines in @string

        -> #str without blank lines
    """
    return "\n".join(line
                     for line in string.split("\n")
                     if len(line.strip()))

def is_line_in_file(filename: str, line: str) -> bool:
    """
    Detects whether a line is present within a file.

    Args:
        filename: file to check
        line: line to search for (as an exact match)
    """
    assert "\n" not in line
    with open(filename, "r") as file:
        for fileline in file:
            if fileline == line:
                return True
        return False

def name_is_valid(name):
    """Return True if the dataset name is valid.

    The name can only be 80 characters long.
    Valid characters: Alpha numeric characters [0-9a-zA-Z]
    Valid special characters: - _ .
    """
    # The name can only be 80 characters long.
    if len(name) > MAX_NAME_LENGTH:
        return False
    return bool(NAME_VALID_CHARS_REGEX.match(name))

def get_column_names(engine: Engine, tablename: str) -> List[str]:
    """
    Get all the database column names for the specified table.
    """
    return [info.name for info in gen_columns_info(engine, tablename)]

def safe_pow(base, exp):
    """safe version of pow"""
    if exp > MAX_EXPONENT:
        raise RuntimeError("Invalid exponent, max exponent is {}".format(MAX_EXPONENT))
    return base ** exp

def text_coords(string, position):
    r"""
    Transform a simple index into a human-readable position in a string.

    This function accepts a string and an index, and will return a triple of
    `(lineno, columnno, line)` representing the position through the text. It's
    useful for displaying a string index in a human-readable way::

        >>> s = "abcdef\nghijkl\nmnopqr\nstuvwx\nyz"
        >>> text_coords(s, 0)
        (0, 0, 'abcdef')
        >>> text_coords(s, 4)
        (0, 4, 'abcdef')
        >>> text_coords(s, 6)
        (0, 6, 'abcdef')
        >>> text_coords(s, 7)
        (1, 0, 'ghijkl')
        >>> text_coords(s, 11)
        (1, 4, 'ghijkl')
        >>> text_coords(s, 15)
        (2, 1, 'mnopqr')
    """
    line_start = string.rfind('\n', 0, position) + 1
    line_end = string.find('\n', position)
    lineno = string.count('\n', 0, position)
    columnno = position - line_start
    line = string[line_start:line_end]
    return (lineno, columnno, line)

def csv_to_numpy(string_like, dtype=None):  # type: (str) -> np.array
    """Convert a CSV object to a numpy array.

    Args:
        string_like (str): CSV string.
        dtype (dtype, optional):  Data type of the resulting array. If None, the dtypes will be determined by the
                                        contents of each column, individually. This argument can only be used to
                                        'upcast' the array.  For downcasting, use the .astype(t) method.
    Returns:
        (np.array): numpy array
    """
    stream = StringIO(string_like)
    return np.genfromtxt(stream, dtype=dtype, delimiter=',')

def has_key(cls, *args):
        """
        Check whether flyweight object with specified key has already been created.

        Returns:
            bool: True if already created, False if not
        """
        key = args if len(args) > 1 else args[0]
        return key in cls._instances

def _isint(string):
    """
    >>> _isint("123")
    True
    >>> _isint("123.45")
    False
    """
    return type(string) is int or \
           (isinstance(string, _binary_type) or isinstance(string, _text_type)) and \
           _isconvertible(int, string)

def build(ctx):
    """Build documentation as HTML.

    The build HTML site is located in the ``doc/_build/html`` directory
    of the package.
    """
    return_code = run_sphinx(ctx.obj['root_dir'])
    if return_code > 0:
        sys.exit(return_code)

def pack_bits( longbits ):
    """Crunch a 64-bit int (8 bool bytes) into a bitfield."""
    byte = longbits & (0x0101010101010101)
    byte = (byte | (byte>>7)) & (0x0003000300030003)
    byte = (byte | (byte>>14)) & (0x0000000f0000000f)
    byte = (byte | (byte>>28)) & (0x00000000000000ff)
    return byte

def rate_limited(max_per_hour: int, *args: Any) -> Callable[..., Any]:
    """Rate limit a function."""
    return util.rate_limited(max_per_hour, *args)

def memory_full():
    """Check if the memory is too full for further caching."""
    current_process = psutil.Process(os.getpid())
    return (current_process.memory_percent() >
            config.MAXIMUM_CACHE_MEMORY_PERCENTAGE)

def _run_sync(self, method: Callable, *args, **kwargs) -> Any:
        """
        Utility method to run commands synchronously for testing.
        """
        if self.loop.is_running():
            raise RuntimeError("Event loop is already running.")

        if not self.is_connected:
            self.loop.run_until_complete(self.connect())

        task = asyncio.Task(method(*args, **kwargs), loop=self.loop)
        result = self.loop.run_until_complete(task)

        self.loop.run_until_complete(self.quit())

        return result

def find_duplicates(l: list) -> set:
    """
    Return the duplicates in a list.

    The function relies on
    https://stackoverflow.com/questions/9835762/find-and-list-duplicates-in-a-list .
    Parameters
    ----------
    l : list
        Name

    Returns
    -------
    set
        Duplicated values

    >>> find_duplicates([1,2,3])
    set()
    >>> find_duplicates([1,2,1])
    {1}
    """
    return set([x for x in l if l.count(x) > 1])

def _parse_tuple_string(argument):
        """ Return a tuple from parsing 'a,b,c,d' -> (a,b,c,d) """
        if isinstance(argument, str):
            return tuple(int(p.strip()) for p in argument.split(','))
        return argument

def __getattr__(self, item: str) -> Callable:
        """Get a callable that sends the actual API request internally."""
        return functools.partial(self.call_action, item)

def fetchallfirstvalues(self, sql: str, *args) -> List[Any]:
        """Executes SQL; returns list of first values of each row."""
        rows = self.fetchall(sql, *args)
        return [row[0] for row in rows]

def memory_full():
    """Check if the memory is too full for further caching."""
    current_process = psutil.Process(os.getpid())
    return (current_process.memory_percent() >
            config.MAXIMUM_CACHE_MEMORY_PERCENTAGE)

def get_from_gnucash26_date(date_str: str) -> date:
    """ Creates a datetime from GnuCash 2.6 date string """
    date_format = "%Y%m%d"
    result = datetime.strptime(date_str, date_format).date()
    return result

def remove_empty_text(utterances: List[Utterance]) -> List[Utterance]:
    """Remove empty utterances from a list of utterances
    Args:
        utterances: The list of utterance we are processing
    """
    return [utter for utter in utterances if utter.text.strip() != ""]

def last_location_of_minimum(x):
    """
    Returns the last location of the minimal value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    return 1.0 - np.argmin(x[::-1]) / len(x) if len(x) > 0 else np.NaN

def maybe_infer_dtype_type(element):
    """Try to infer an object's dtype, for use in arithmetic ops

    Uses `element.dtype` if that's available.
    Objects implementing the iterator protocol are cast to a NumPy array,
    and from there the array's type is used.

    Parameters
    ----------
    element : object
        Possibly has a `.dtype` attribute, and possibly the iterator
        protocol.

    Returns
    -------
    tipo : type

    Examples
    --------
    >>> from collections import namedtuple
    >>> Foo = namedtuple("Foo", "dtype")
    >>> maybe_infer_dtype_type(Foo(np.dtype("i8")))
    numpy.int64
    """
    tipo = None
    if hasattr(element, 'dtype'):
        tipo = element.dtype
    elif is_list_like(element):
        element = np.asarray(element)
        tipo = element.dtype
    return tipo

def memory_full():
    """Check if the memory is too full for further caching."""
    current_process = psutil.Process(os.getpid())
    return (current_process.memory_percent() >
            config.MAXIMUM_CACHE_MEMORY_PERCENTAGE)

def timeit(func, *args, **kwargs):
    """
    Time execution of function. Returns (res, seconds).

    >>> res, timing = timeit(time.sleep, 1)
    """
    start_time = time.time()
    res = func(*args, **kwargs)
    timing = time.time() - start_time
    return res, timing

def method_caller(method_name, *args, **kwargs):
	"""
	Return a function that will call a named method on the
	target object with optional positional and keyword
	arguments.

	>>> lower = method_caller('lower')
	>>> lower('MyString')
	'mystring'
	"""
	def call_method(target):
		func = getattr(target, method_name)
		return func(*args, **kwargs)
	return call_method

def psutil_phymem_usage():
    """
    Return physical memory usage (float)
    Requires the cross-platform psutil (>=v0.3) library
    (https://github.com/giampaolo/psutil)
    """
    import psutil
    # This is needed to avoid a deprecation warning error with
    # newer psutil versions
    try:
        percent = psutil.virtual_memory().percent
    except:
        percent = psutil.phymem_usage().percent
    return percent

def zfill(x, width):
    """zfill(x, width) -> string

    Pad a numeric string x with zeros on the left, to fill a field
    of the specified width.  The string x is never truncated.

    """
    if not isinstance(x, basestring):
        x = repr(x)
    return x.zfill(width)

def get_period_last_3_months() -> str:
    """ Returns the last week as a period string """
    today = Datum()
    today.today()

    # start_date = today - timedelta(weeks=13)
    start_date = today.clone()
    start_date.subtract_months(3)

    period = get_period(start_date.date, today.date)
    return period

def block_diag(*blocks: np.ndarray) -> np.ndarray:
    """Concatenates blocks into a block diagonal matrix.

    Args:
        *blocks: Square matrices to place along the diagonal of the result.

    Returns:
        A block diagonal matrix with the given blocks along its diagonal.

    Raises:
        ValueError: A block isn't square.
    """
    for b in blocks:
        if b.shape[0] != b.shape[1]:
            raise ValueError('Blocks must be square.')

    if not blocks:
        return np.zeros((0, 0), dtype=np.complex128)

    n = sum(b.shape[0] for b in blocks)
    dtype = functools.reduce(_merge_dtypes, (b.dtype for b in blocks))

    result = np.zeros(shape=(n, n), dtype=dtype)
    i = 0
    for b in blocks:
        j = i + b.shape[0]
        result[i:j, i:j] = b
        i = j

    return result

def get_column_names(engine: Engine, tablename: str) -> List[str]:
    """
    Get all the database column names for the specified table.
    """
    return [info.name for info in gen_columns_info(engine, tablename)]

def maybe_infer_dtype_type(element):
    """Try to infer an object's dtype, for use in arithmetic ops

    Uses `element.dtype` if that's available.
    Objects implementing the iterator protocol are cast to a NumPy array,
    and from there the array's type is used.

    Parameters
    ----------
    element : object
        Possibly has a `.dtype` attribute, and possibly the iterator
        protocol.

    Returns
    -------
    tipo : type

    Examples
    --------
    >>> from collections import namedtuple
    >>> Foo = namedtuple("Foo", "dtype")
    >>> maybe_infer_dtype_type(Foo(np.dtype("i8")))
    numpy.int64
    """
    tipo = None
    if hasattr(element, 'dtype'):
        tipo = element.dtype
    elif is_list_like(element):
        element = np.asarray(element)
        tipo = element.dtype
    return tipo

def from_file(file_path) -> dict:
        """ Load JSON file """
        with io.open(file_path, 'r', encoding='utf-8') as json_stream:
            return Json.parse(json_stream, True)

def is_finite(value: Any) -> bool:
    """Return true if a value is a finite number."""
    return isinstance(value, int) or (isinstance(value, float) and isfinite(value))

def clean_map(obj: Mapping[Any, Any]) -> Mapping[Any, Any]:
    """
    Return a new copied dictionary without the keys with ``None`` values from
    the given Mapping object.
    """
    return {k: v for k, v in obj.items() if v is not None}

def average_arrays(arrays: List[mx.nd.NDArray]) -> mx.nd.NDArray:
    """
    Take a list of arrays of the same shape and take the element wise average.

    :param arrays: A list of NDArrays with the same shape that will be averaged.
    :return: The average of the NDArrays in the same context as arrays[0].
    """
    if not arrays:
        raise ValueError("arrays is empty.")
    if len(arrays) == 1:
        return arrays[0]
    check_condition(all(arrays[0].shape == a.shape for a in arrays), "nd array shapes do not match")
    return mx.nd.add_n(*arrays) / len(arrays)

def read_text_from_file(path: str) -> str:
    """ Reads text file contents """
    with open(path) as text_file:
        content = text_file.read()

    return content

def is_not_null(df: DataFrame, col_name: str) -> bool:
    """
    Return ``True`` if the given DataFrame has a column of the given
    name (string), and there exists at least one non-NaN value in that
    column; return ``False`` otherwise.
    """
    if (
        isinstance(df, pd.DataFrame)
        and col_name in df.columns
        and df[col_name].notnull().any()
    ):
        return True
    else:
        return False

def get_domain(url):
    """
    Get domain part of an url.

    For example: https://www.python.org/doc/ -> https://www.python.org
    """
    parse_result = urlparse(url)
    domain = "{schema}://{netloc}".format(
        schema=parse_result.scheme, netloc=parse_result.netloc)
    return domain

def write_text(filename: str, text: str) -> None:
    """
    Writes text to a file.
    """
    with open(filename, 'w') as f:  # type: TextIO
        print(text, file=f)

def isarray(array, test, dim=2):
    """Returns True if test is True for all array elements.
    Otherwise, returns False.
    """
    if dim > 1:
        return all(isarray(array[i], test, dim - 1)
                   for i in range(len(array)))
    return all(test(i) for i in array)

def after_third_friday(day=None):
    """ check if day is after month's 3rd friday """
    day = day if day is not None else datetime.datetime.now()
    now = day.replace(day=1, hour=16, minute=0, second=0, microsecond=0)
    now += relativedelta.relativedelta(weeks=2, weekday=relativedelta.FR)
    return day > now

def stretch(iterable, n=2):
    r"""Repeat each item in `iterable` `n` times.

    Example:

    >>> list(stretch(range(3), 2))
    [0, 0, 1, 1, 2, 2]
    """
    times = range(n)
    for item in iterable:
        for i in times: yield item

def debugTreePrint(node,pfx="->"):
  """Purely a debugging aid: Ascii-art picture of a tree descended from node"""
  print pfx,node.item
  for c in node.children:
    debugTreePrint(c,"  "+pfx)

def memory_full():
    """Check if the memory is too full for further caching."""
    current_process = psutil.Process(os.getpid())
    return (current_process.memory_percent() >
            config.MAXIMUM_CACHE_MEMORY_PERCENTAGE)

def get_account_id_by_fullname(self, fullname: str) -> str:
        """ Locates the account by fullname """
        account = self.get_by_fullname(fullname)
        return account.guid

def get_timezone() -> Tuple[datetime.tzinfo, str]:
    """Discover the current time zone and it's standard string representation (for source{d})."""
    dt = get_datetime_now().astimezone()
    tzstr = dt.strftime("%z")
    tzstr = tzstr[:-2] + ":" + tzstr[-2:]
    return dt.tzinfo, tzstr

def dtypes(self):
        """Returns all column names and their data types as a list.

        >>> df.dtypes
        [('age', 'int'), ('name', 'string')]
        """
        return [(str(f.name), f.dataType.simpleString()) for f in self.schema.fields]

def _reshuffle(mat, shape):
    """Reshuffle the indicies of a bipartite matrix A[ij,kl] -> A[lj,ki]."""
    return np.reshape(
        np.transpose(np.reshape(mat, shape), (3, 1, 2, 0)),
        (shape[3] * shape[1], shape[0] * shape[2]))

def get_column_names(engine: Engine, tablename: str) -> List[str]:
    """
    Get all the database column names for the specified table.
    """
    return [info.name for info in gen_columns_info(engine, tablename)]

def read_text_from_file(path: str) -> str:
    """ Reads text file contents """
    with open(path) as text_file:
        content = text_file.read()

    return content

def butlast(iterable):
    """Yield all items from ``iterable`` except the last one.

    >>> list(butlast(['spam', 'eggs', 'ham']))
    ['spam', 'eggs']

    >>> list(butlast(['spam']))
    []

    >>> list(butlast([]))
    []
    """
    iterable = iter(iterable)
    try:
        first = next(iterable)
    except StopIteration:
        return
    for second in iterable:
        yield first
        first = second

def read_text_from_file(path: str) -> str:
    """ Reads text file contents """
    with open(path) as text_file:
        content = text_file.read()

    return content

def isFull(self):
        """
        Returns True if the response buffer is full, and False otherwise.
        The buffer is full if either (1) the number of items in the value
        list is >= pageSize or (2) the total length of the serialised
        elements in the page is >= maxBufferSize.

        If page_size or max_response_length were not set in the request
        then they're not checked.
        """
        return (
            (self._pageSize > 0 and self._numElements >= self._pageSize) or
            (self._bufferSize >= self._maxBufferSize)
        )

def after_third_friday(day=None):
    """ check if day is after month's 3rd friday """
    day = day if day is not None else datetime.datetime.now()
    now = day.replace(day=1, hour=16, minute=0, second=0, microsecond=0)
    now += relativedelta.relativedelta(weeks=2, weekday=relativedelta.FR)
    return day > now

def callable_validator(v: Any) -> AnyCallable:
    """
    Perform a simple check if the value is callable.

    Note: complete matching of argument type hints and return types is not performed
    """
    if callable(v):
        return v

    raise errors.CallableError(value=v)

def is_line_in_file(filename: str, line: str) -> bool:
    """
    Detects whether a line is present within a file.

    Args:
        filename: file to check
        line: line to search for (as an exact match)
    """
    assert "\n" not in line
    with open(filename, "r") as file:
        for fileline in file:
            if fileline == line:
                return True
        return False

def snake_to_camel(s: str) -> str:
    """Convert string from snake case to camel case."""

    fragments = s.split('_')

    return fragments[0] + ''.join(x.title() for x in fragments[1:])

def last_location_of_minimum(x):
    """
    Returns the last location of the minimal value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    return 1.0 - np.argmin(x[::-1]) / len(x) if len(x) > 0 else np.NaN

def camel_to_snake_case(string):
    """Converts 'string' presented in camel case to snake case.

    e.g.: CamelCase => snake_case
    """
    s = _1.sub(r'\1_\2', string)
    return _2.sub(r'\1_\2', s).lower()

def _isint(string):
    """
    >>> _isint("123")
    True
    >>> _isint("123.45")
    False
    """
    return type(string) is int or \
           (isinstance(string, _binary_type) or isinstance(string, _text_type)) and \
           _isconvertible(int, string)

def lower_camel_case_from_underscores(string):
    """generate a lower-cased camelCase string from an underscore_string.
    For example: my_variable_name -> myVariableName"""
    components = string.split('_')
    string = components[0]
    for component in components[1:]:
        string += component[0].upper() + component[1:]
    return string

def is_sqlatype_integer(coltype: Union[TypeEngine, VisitableType]) -> bool:
    """
    Is the SQLAlchemy column type an integer type?
    """
    coltype = _coltype_to_typeengine(coltype)
    return isinstance(coltype, sqltypes.Integer)

def _brief_print_list(lst, limit=7):
    """Print at most `limit` elements of list."""
    lst = list(lst)
    if len(lst) > limit:
        return _brief_print_list(lst[:limit//2], limit) + ', ..., ' + \
            _brief_print_list(lst[-limit//2:], limit)
    return ', '.join(["'%s'"%str(i) for i in lst])

def has_table(self, name):
        """Return ``True`` if the table *name* exists in the database."""
        return len(self.sql("SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                            parameters=(name,), asrecarray=False, cache=False)) > 0

def proper_round(n):
    """
    rounds float to closest int
    :rtype: int
    :param n: float
    """
    return int(n) + (n / abs(n)) * int(abs(n - int(n)) >= 0.5) if n != 0 else 0

def count(args):
    """ count occurences in a list of lists
    >>> count([['a','b'],['a']])
    defaultdict(int, {'a' : 2, 'b' : 1})
    """
    counts = defaultdict(int)
    for arg in args:
        for item in arg:
            counts[item] = counts[item] + 1
    return counts

def long_substr(data):
    """Return the longest common substring in a list of strings.
    
    Credit: http://stackoverflow.com/questions/2892931/longest-common-substring-from-more-than-two-strings-python
    """
    substr = ''
    if len(data) > 1 and len(data[0]) > 0:
        for i in range(len(data[0])):
            for j in range(len(data[0])-i+1):
                if j > len(substr) and all(data[0][i:i+j] in x for x in data):
                    substr = data[0][i:i+j]
    elif len(data) == 1:
        substr = data[0]
    return substr

def suppress_stdout():
    """
    Context manager that suppresses stdout.

    Examples:
        >>> with suppress_stdout():
        ...     print('Test print')

        >>> print('test')
        test

    """
    save_stdout = sys.stdout
    sys.stdout = DevNull()
    yield
    sys.stdout = save_stdout

def command(self, cmd, *args):
        """
        Sends a command and an (optional) sequence of arguments through to the
        delegated serial interface. Note that the arguments are passed through
        as data.
        """
        self._serial_interface.command(cmd)
        if len(args) > 0:
            self._serial_interface.data(list(args))

def timeit (func, log, limit):
    """Print execution time of the function. For quick'n'dirty profiling."""

    def newfunc (*args, **kwargs):
        """Execute function and print execution time."""
        t = time.time()
        res = func(*args, **kwargs)
        duration = time.time() - t
        if duration > limit:
            print(func.__name__, "took %0.2f seconds" % duration, file=log)
            print(args, file=log)
            print(kwargs, file=log)
        return res
    return update_func_meta(newfunc, func)

def cli_run():
    """docstring for argparse"""
    parser = argparse.ArgumentParser(description='Stupidly simple code answers from StackOverflow')
    parser.add_argument('query', help="What's the problem ?", type=str, nargs='+')
    parser.add_argument('-t','--tags', help='semicolon separated tags -> python;lambda')
    args = parser.parse_args()
    main(args)

def pack_bits( longbits ):
    """Crunch a 64-bit int (8 bool bytes) into a bitfield."""
    byte = longbits & (0x0101010101010101)
    byte = (byte | (byte>>7)) & (0x0003000300030003)
    byte = (byte | (byte>>14)) & (0x0000000f0000000f)
    byte = (byte | (byte>>28)) & (0x00000000000000ff)
    return byte

def date_to_datetime(d):
    """
    >>> date_to_datetime(date(2000, 1, 2))
    datetime.datetime(2000, 1, 2, 0, 0)
    >>> date_to_datetime(datetime(2000, 1, 2, 3, 4, 5))
    datetime.datetime(2000, 1, 2, 3, 4, 5)
    """
    if not isinstance(d, datetime):
        d = datetime.combine(d, datetime.min.time())
    return d

def s3_get(url: str, temp_file: IO) -> None:
    """Pull a file directly from S3."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)

def str_to_time(time_str: str) -> datetime.datetime:
    """
    Convert human readable string to datetime.datetime.
    """
    pieces: Any = [int(piece) for piece in time_str.split('-')]
    return datetime.datetime(*pieces)

def find_duplicates(l: list) -> set:
    """
    Return the duplicates in a list.

    The function relies on
    https://stackoverflow.com/questions/9835762/find-and-list-duplicates-in-a-list .
    Parameters
    ----------
    l : list
        Name

    Returns
    -------
    set
        Duplicated values

    >>> find_duplicates([1,2,3])
    set()
    >>> find_duplicates([1,2,1])
    {1}
    """
    return set([x for x in l if l.count(x) > 1])

def encode_list(key, list_):
    # type: (str, Iterable) -> Dict[str, str]
    """
    Converts a list into a space-separated string and puts it in a dictionary

    :param key: Dictionary key to store the list
    :param list_: A list of objects
    :return: A dictionary key->string or an empty dictionary
    """
    if not list_:
        return {}
    return {key: " ".join(str(i) for i in list_)}

def _rindex(mylist: Sequence[T], x: T) -> int:
    """Index of the last occurrence of x in the sequence."""
    return len(mylist) - mylist[::-1].index(x) - 1

def quaternion_imag(quaternion):
    """Return imaginary part of quaternion.

    >>> quaternion_imag([3, 0, 1, 2])
    array([0., 1., 2.])

    """
    return np.array(quaternion[1:4], dtype=np.float64, copy=True)

def dtypes(self):
        """Returns all column names and their data types as a list.

        >>> df.dtypes
        [('age', 'int'), ('name', 'string')]
        """
        return [(str(f.name), f.dataType.simpleString()) for f in self.schema.fields]

def strtobytes(input, encoding):
    """Take a str and transform it into a byte array."""
    py_version = sys.version_info[0]
    if py_version >= 3:
        return _strtobytes_py3(input, encoding)
    return _strtobytes_py2(input, encoding)

def s3_get(url: str, temp_file: IO) -> None:
    """Pull a file directly from S3."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)

def wipe_table(self, table: str) -> int:
        """Delete all records from a table. Use caution!"""
        sql = "DELETE FROM " + self.delimit(table)
        return self.db_exec(sql)

def clean_map(obj: Mapping[Any, Any]) -> Mapping[Any, Any]:
    """
    Return a new copied dictionary without the keys with ``None`` values from
    the given Mapping object.
    """
    return {k: v for k, v in obj.items() if v is not None}

def zip_with_index(rdd):
    """
    Alternate version of Spark's zipWithIndex that eagerly returns count.
    """
    starts = [0]
    if rdd.getNumPartitions() > 1:
        nums = rdd.mapPartitions(lambda it: [sum(1 for _ in it)]).collect()
        count = sum(nums)
        for i in range(len(nums) - 1):
            starts.append(starts[-1] + nums[i])
    else:
        count = rdd.count()

    def func(k, it):
        for i, v in enumerate(it, starts[k]):
            yield v, i

    return count, rdd.mapPartitionsWithIndex(func)

def is_prime(n):
    """
    Check if n is a prime number
    """
    if n % 2 == 0 and n > 2:
        return False
    return all(n % i for i in range(3, int(math.sqrt(n)) + 1, 2))

def is_unicode(string):
    """Validates that the object itself is some kinda string"""
    str_type = str(type(string))

    if str_type.find('str') > 0 or str_type.find('unicode') > 0:
        return True

    return False

def from_file(file_path) -> dict:
        """ Load JSON file """
        with io.open(file_path, 'r', encoding='utf-8') as json_stream:
            return Json.parse(json_stream, True)

def isfinite(data: mx.nd.NDArray) -> mx.nd.NDArray:
    """Performs an element-wise check to determine if the NDArray contains an infinite element or not.
       TODO: remove this funciton after upgrade to MXNet 1.4.* in favor of mx.ndarray.contrib.isfinite()
    """
    is_data_not_nan = data == data
    is_data_not_infinite = data.abs() != np.inf
    return mx.nd.logical_and(is_data_not_infinite, is_data_not_nan)

def str_to_time(time_str: str) -> datetime.datetime:
    """
    Convert human readable string to datetime.datetime.
    """
    pieces: Any = [int(piece) for piece in time_str.split('-')]
    return datetime.datetime(*pieces)

def stretch(iterable, n=2):
    r"""Repeat each item in `iterable` `n` times.

    Example:

    >>> list(stretch(range(3), 2))
    [0, 0, 1, 1, 2, 2]
    """
    times = range(n)
    for item in iterable:
        for i in times: yield item

def position(self) -> Position:
        """The current position of the cursor."""
        return Position(self._index, self._lineno, self._col_offset)

def _str_to_list(value, separator):
    """Convert a string to a list with sanitization."""
    value_list = [item.strip() for item in value.split(separator)]
    value_list_sanitized = builtins.list(filter(None, value_list))
    if len(value_list_sanitized) > 0:
        return value_list_sanitized
    else:
        raise ValueError('Invalid list variable.')

def get_default_bucket_key(buckets: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Returns the default bucket from a list of buckets, i.e. the largest bucket.

    :param buckets: List of buckets.
    :return: The largest bucket in the list.
    """
    return max(buckets)

def remove_blank_lines(string):
    """ Removes all blank lines in @string

        -> #str without blank lines
    """
    return "\n".join(line
                     for line in string.split("\n")
                     if len(line.strip()))

def console_get_background_flag(con: tcod.console.Console) -> int:
    """Return this consoles current blend mode.

    Args:
        con (Console): Any Console instance.

    .. deprecated:: 8.5
        Check :any:`Console.default_bg_blend` instead.
    """
    return int(lib.TCOD_console_get_background_flag(_console(con)))

def butlast(iterable):
    """Yield all items from ``iterable`` except the last one.

    >>> list(butlast(['spam', 'eggs', 'ham']))
    ['spam', 'eggs']

    >>> list(butlast(['spam']))
    []

    >>> list(butlast([]))
    []
    """
    iterable = iter(iterable)
    try:
        first = next(iterable)
    except StopIteration:
        return
    for second in iterable:
        yield first
        first = second

def iprotate(l, steps=1):
    r"""Like rotate, but modifies `l` in-place.

    >>> l = [1,2,3]
    >>> iprotate(l) is l
    True
    >>> l
    [2, 3, 1]
    >>> iprotate(iprotate(l, 2), -3)
    [1, 2, 3]

    """
    if len(l):
        steps %= len(l)
        if steps:
            firstPart = l[:steps]
            del l[:steps]
            l.extend(firstPart)
    return l

def get_valid_filename(s):
    """
    Shamelessly taken from Django.
    https://github.com/django/django/blob/master/django/utils/text.py

    Return the given string converted to a string that can be used for a clean
    filename. Remove leading and trailing spaces; convert other spaces to
    underscores; and remove anything that is not an alphanumeric, dash,
    underscore, or dot.
    >>> get_valid_filename("john's portrait in 2004.jpg")
    'johns_portrait_in_2004.jpg'
    """
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)

def load_yaml(file):
    """If pyyaml > 5.1 use full_load to avoid warning"""
    if hasattr(yaml, "full_load"):
        return yaml.full_load(file)
    else:
        return yaml.load(file)

def warn_if_nans_exist(X):
    """Warn if nans exist in a numpy array."""
    null_count = count_rows_with_nans(X)
    total = len(X)
    percent = 100 * null_count / total

    if null_count > 0:
        warning_message = \
            'Warning! Found {} rows of {} ({:0.2f}%) with nan values. Only ' \
            'complete rows will be plotted.'.format(null_count, total, percent)
        warnings.warn(warning_message, DataWarning)

def dictlist_wipe_key(dict_list: Iterable[Dict], key: str) -> None:
    """
    Process an iterable of dictionaries. For each dictionary ``d``, delete
    ``d[key]`` if it exists.
    """
    for d in dict_list:
        d.pop(key, None)

def memory_full():
    """Check if the memory is too full for further caching."""
    current_process = psutil.Process(os.getpid())
    return (current_process.memory_percent() >
            config.MAXIMUM_CACHE_MEMORY_PERCENTAGE)

def is_prime(n):
    """
    Check if n is a prime number
    """
    if n % 2 == 0 and n > 2:
        return False
    return all(n % i for i in range(3, int(math.sqrt(n)) + 1, 2))

def from_file(file_path) -> dict:
        """ Load JSON file """
        with io.open(file_path, 'r', encoding='utf-8') as json_stream:
            return Json.parse(json_stream, True)

def uppercase_chars(string: any) -> str:
        """Return all (and only) the uppercase chars in the given string."""
        return ''.join([c if c.isupper() else '' for c in str(string)])

def is_unicode(string):
    """Validates that the object itself is some kinda string"""
    str_type = str(type(string))

    if str_type.find('str') > 0 or str_type.find('unicode') > 0:
        return True

    return False

def iter_fields(self, schema: Schema) -> Iterable[Tuple[str, Field]]:
        """
        Iterate through marshmallow schema fields.

        Generates: name, field pairs

        """
        for name in sorted(schema.fields.keys()):
            field = schema.fields[name]
            yield field.dump_to or name, field

def rank(tensor: BKTensor) -> int:
    """Return the number of dimensions of a tensor"""
    if isinstance(tensor, np.ndarray):
        return len(tensor.shape)

    return len(tensor[0].size())

def str_to_time(time_str: str) -> datetime.datetime:
    """
    Convert human readable string to datetime.datetime.
    """
    pieces: Any = [int(piece) for piece in time_str.split('-')]
    return datetime.datetime(*pieces)

def numeric_part(s):
    """Returns the leading numeric part of a string.

    >>> numeric_part("20-alpha")
    20
    >>> numeric_part("foo")
    >>> numeric_part("16b")
    16
    """

    m = re_numeric_part.match(s)
    if m:
        return int(m.group(1))
    return None

def random_name_gen(size=6):
    """Generate a random python attribute name."""

    return ''.join(
        [random.choice(string.ascii_uppercase)] +
        [random.choice(string.ascii_uppercase + string.digits) for i in range(size - 1)]
    ) if size > 0 else ''

def astensor(array: TensorLike) -> BKTensor:
    """Covert numpy array to tensorflow tensor"""
    tensor = tf.convert_to_tensor(value=array, dtype=CTYPE)
    return tensor

def count(args):
    """ count occurences in a list of lists
    >>> count([['a','b'],['a']])
    defaultdict(int, {'a' : 2, 'b' : 1})
    """
    counts = defaultdict(int)
    for arg in args:
        for item in arg:
            counts[item] = counts[item] + 1
    return counts

def looks_like_url(url):
    """ Simplified check to see if the text appears to be a URL.

    Similar to `urlparse` but much more basic.

    Returns:
      True if the url str appears to be valid.
      False otherwise.

    >>> url = looks_like_url("totalgood.org")
    >>> bool(url)
    True
    """
    if not isinstance(url, basestring):
        return False
    if not isinstance(url, basestring) or len(url) >= 1024 or not cre_url.match(url):
        return False
    return True

def check64bit(current_system="python"):
    """checks if you are on a 64 bit platform"""
    if current_system == "python":
        return sys.maxsize > 2147483647
    elif current_system == "os":
        import platform
        pm = platform.machine()
        if pm != ".." and pm.endswith('64'):  # recent Python (not Iron)
            return True
        else:
            if 'PROCESSOR_ARCHITEW6432' in os.environ:
                return True  # 32 bit program running on 64 bit Windows
            try:
                # 64 bit Windows 64 bit program
                return os.environ['PROCESSOR_ARCHITECTURE'].endswith('64')
            except IndexError:
                pass  # not Windows
            try:
                # this often works in Linux
                return '64' in platform.architecture()[0]
            except Exception:
                # is an older version of Python, assume also an older os@
                # (best we can guess)
                return False

def quoted_or_list(items: List[str]) -> Optional[str]:
    """Given [A, B, C] return "'A', 'B', or 'C'".

    Note: We use single quotes here, since these are also used by repr().
    """
    return or_list([f"'{item}'" for item in items])

def get_timezone() -> Tuple[datetime.tzinfo, str]:
    """Discover the current time zone and it's standard string representation (for source{d})."""
    dt = get_datetime_now().astimezone()
    tzstr = dt.strftime("%z")
    tzstr = tzstr[:-2] + ":" + tzstr[-2:]
    return dt.tzinfo, tzstr

def is_sqlatype_integer(coltype: Union[TypeEngine, VisitableType]) -> bool:
    """
    Is the SQLAlchemy column type an integer type?
    """
    coltype = _coltype_to_typeengine(coltype)
    return isinstance(coltype, sqltypes.Integer)

def _skip_section(self):
        """Skip a section"""
        self._last = self._f.readline()
        while len(self._last) > 0 and len(self._last[0].strip()) == 0:
            self._last = self._f.readline()

def from_file(file_path) -> dict:
        """ Load JSON file """
        with io.open(file_path, 'r', encoding='utf-8') as json_stream:
            return Json.parse(json_stream, True)

def is_not_null(df: DataFrame, col_name: str) -> bool:
    """
    Return ``True`` if the given DataFrame has a column of the given
    name (string), and there exists at least one non-NaN value in that
    column; return ``False`` otherwise.
    """
    if (
        isinstance(df, pd.DataFrame)
        and col_name in df.columns
        and df[col_name].notnull().any()
    ):
        return True
    else:
        return False

def urljoin(*args):
    """
    Joins given arguments into a url, removing duplicate slashes
    Thanks http://stackoverflow.com/a/11326230/1267398

    >>> urljoin('/lol', '///lol', '/lol//')
    '/lol/lol/lol'
    """
    value = "/".join(map(lambda x: str(x).strip('/'), args))
    return "/{}".format(value)

def clean_int(x) -> int:
    """
    Returns its parameter as an integer, or raises
    ``django.forms.ValidationError``.
    """
    try:
        return int(x)
    except ValueError:
        raise forms.ValidationError(
            "Cannot convert to integer: {}".format(repr(x)))

def remove_once(gset, elem):
    """Remove the element from a set, lists or dict.
    
        >>> L = ["Lucy"]; S = set(["Sky"]); D = { "Diamonds": True };
        >>> remove_once(L, "Lucy"); remove_once(S, "Sky"); remove_once(D, "Diamonds");
        >>> print L, S, D
        [] set([]) {}

    Returns the element if it was removed. Raises one of the exceptions in 
    :obj:`RemoveError` otherwise.
    """
    remove = getattr(gset, 'remove', None)
    if remove is not None: remove(elem)
    else: del gset[elem]
    return elem

def _str_to_list(value, separator):
    """Convert a string to a list with sanitization."""
    value_list = [item.strip() for item in value.split(separator)]
    value_list_sanitized = builtins.list(filter(None, value_list))
    if len(value_list_sanitized) > 0:
        return value_list_sanitized
    else:
        raise ValueError('Invalid list variable.')

def get_last_weekday_in_month(year, month, weekday):
        """Get the last weekday in a given month. e.g:

        >>> # the last monday in Jan 2013
        >>> Calendar.get_last_weekday_in_month(2013, 1, MON)
        datetime.date(2013, 1, 28)
        """
        day = date(year, month, monthrange(year, month)[1])
        while True:
            if day.weekday() == weekday:
                break
            day = day - timedelta(days=1)
        return day

def stdev(self):
        """ -> #float :func:numpy.std of the timing intervals """
        return round(np.std(self.array), self.precision)\
            if len(self.array) else None

def _(f, x):
    """
    filter for dict, note `f` should have signature: `f::key->value->bool`
    """
    return {k: v for k, v in x.items() if f(k, v)}

def read_text_from_file(path: str) -> str:
    """ Reads text file contents """
    with open(path) as text_file:
        content = text_file.read()

    return content

def genfirstvalues(cursor: Cursor, arraysize: int = 1000) \
        -> Generator[Any, None, None]:
    """
    Generate the first value in each row.

    Args:
        cursor: the cursor
        arraysize: split fetches into chunks of this many records

    Yields:
        the first value of each row
    """
    return (row[0] for row in genrows(cursor, arraysize))

def space_list(line: str) -> List[int]:
    """
    Given a string, return a list of index positions where a blank space occurs.

    :param line:
    :return:

    >>> space_list("    abc ")
    [0, 1, 2, 3, 7]
    """
    spaces = []
    for idx, car in enumerate(list(line)):
        if car == " ":
            spaces.append(idx)
    return spaces

def is_not_null(df: DataFrame, col_name: str) -> bool:
    """
    Return ``True`` if the given DataFrame has a column of the given
    name (string), and there exists at least one non-NaN value in that
    column; return ``False`` otherwise.
    """
    if (
        isinstance(df, pd.DataFrame)
        and col_name in df.columns
        and df[col_name].notnull().any()
    ):
        return True
    else:
        return False

def rank(tensor: BKTensor) -> int:
    """Return the number of dimensions of a tensor"""
    if isinstance(tensor, np.ndarray):
        return len(tensor.shape)

    return len(tensor[0].size())

def get_property(self, name):
        # type: (str) -> object
        """
        Retrieves a framework or system property. As framework properties don't
        change while it's running, this method don't need to be protected.

        :param name: The property name
        """
        with self.__properties_lock:
            return self.__properties.get(name, os.getenv(name))

def genfirstvalues(cursor: Cursor, arraysize: int = 1000) \
        -> Generator[Any, None, None]:
    """
    Generate the first value in each row.

    Args:
        cursor: the cursor
        arraysize: split fetches into chunks of this many records

    Yields:
        the first value of each row
    """
    return (row[0] for row in genrows(cursor, arraysize))

def is_natural(x):
    """A non-negative integer."""
    try:
        is_integer = int(x) == x
    except (TypeError, ValueError):
        return False
    return is_integer and x >= 0

def get_period_last_3_months() -> str:
    """ Returns the last week as a period string """
    today = Datum()
    today.today()

    # start_date = today - timedelta(weeks=13)
    start_date = today.clone()
    start_date.subtract_months(3)

    period = get_period(start_date.date, today.date)
    return period

def argmax(self, rows: List[Row], column: ComparableColumn) -> List[Row]:
        """
        Takes a list of rows and a column name and returns a list containing a single row (dict from
        columns to cells) that has the maximum numerical value in the given column. We return a list
        instead of a single dict to be consistent with the return type of ``select`` and
        ``all_rows``.
        """
        if not rows:
            return []
        value_row_pairs = [(row.values[column.name], row) for row in rows]
        if not value_row_pairs:
            return []
        # Returns a list containing the row with the max cell value.
        return [sorted(value_row_pairs, key=lambda x: x[0], reverse=True)[0][1]]

def most_significant_bit(lst: np.ndarray) -> int:
    """
    A helper function that finds the position of the most significant bit in a 1darray of 1s and 0s,
    i.e. the first position where a 1 appears, reading left to right.

    :param lst: a 1d array of 0s and 1s with at least one 1
    :return: the first position in lst that a 1 appears
    """
    return np.argwhere(np.asarray(lst) == 1)[0][0]

def isarray(array, test, dim=2):
    """Returns True if test is True for all array elements.
    Otherwise, returns False.
    """
    if dim > 1:
        return all(isarray(array[i], test, dim - 1)
                   for i in range(len(array)))
    return all(test(i) for i in array)

def remove_blank_lines(string):
    """ Removes all blank lines in @string

        -> #str without blank lines
    """
    return "\n".join(line
                     for line in string.split("\n")
                     if len(line.strip()))

def is_builtin_object(node: astroid.node_classes.NodeNG) -> bool:
    """Returns True if the given node is an object from the __builtin__ module."""
    return node and node.root().name == BUILTINS_NAME

def _mid(pt1, pt2):
    """
    (Point, Point) -> Point
    Return the point that lies in between the two input points.
    """
    (x0, y0), (x1, y1) = pt1, pt2
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)

def __del__(self):
        """Frees all resources.
        """
        if hasattr(self, '_Api'):
            self._Api.close()

        self._Logger.info('object destroyed')

def assert_equal(first, second, msg_fmt="{msg}"):
    """Fail unless first equals second, as determined by the '==' operator.

    >>> assert_equal(5, 5.0)
    >>> assert_equal("Hello World!", "Goodbye!")
    Traceback (most recent call last):
        ...
    AssertionError: 'Hello World!' != 'Goodbye!'

    The following msg_fmt arguments are supported:
    * msg - the default error message
    * first - the first argument
    * second - the second argument
    """

    if isinstance(first, dict) and isinstance(second, dict):
        assert_dict_equal(first, second, msg_fmt)
    elif not first == second:
        msg = "{!r} != {!r}".format(first, second)
        fail(msg_fmt.format(msg=msg, first=first, second=second))

def get_language(query: str) -> str:
    """Tries to work out the highlight.js language of a given file name or
    shebang. Returns an empty string if none match.
    """
    query = query.lower()
    for language in LANGUAGES:
        if query.endswith(language):
            return language
    return ''

def prin(*args, **kwargs):
    r"""Like ``print``, but a function. I.e. prints out all arguments as
    ``print`` would do. Specify output stream like this::

      print('ERROR', `out="sys.stderr"``).

    """
    print >> kwargs.get('out',None), " ".join([str(arg) for arg in args])

def post(self, endpoint: str, **kwargs) -> dict:
        """HTTP POST operation to API endpoint."""

        return self._request('POST', endpoint, **kwargs)

def inverted_dict(d):
    """Return a dict with swapped keys and values

    >>> inverted_dict({0: ('a', 'b'), 1: 'cd'}) == {'cd': 1, ('a', 'b'): 0}
    True
    """
    return dict((force_hashable(v), k) for (k, v) in viewitems(dict(d)))

def to_bool(value):
    # type: (Any) -> bool
    """
    Convert a value into a bool but handle "truthy" strings eg, yes, true, ok, y
    """
    if isinstance(value, _compat.string_types):
        return value.upper() in ('Y', 'YES', 'T', 'TRUE', '1', 'OK')
    return bool(value)

def _parse_tuple_string(argument):
        """ Return a tuple from parsing 'a,b,c,d' -> (a,b,c,d) """
        if isinstance(argument, str):
            return tuple(int(p.strip()) for p in argument.split(','))
        return argument

def year(date):
    """ Returns the year.

    :param date:
        The string date with this format %m/%d/%Y
    :type date:
        String

    :returns:
        int

    :example:
        >>> year('05/1/2015')
        2015
    """
    try:
        fmt = '%m/%d/%Y'
        return datetime.strptime(date, fmt).timetuple().tm_year
    except ValueError:
        return 0

def has_synset(word: str) -> list:
    """" Returns a list of synsets of a word after lemmatization. """

    return wn.synsets(lemmatize(word, neverstem=True))

def version():
    """Wrapper for opj_version library routine."""
    OPENJPEG.opj_version.restype = ctypes.c_char_p
    library_version = OPENJPEG.opj_version()
    if sys.hexversion >= 0x03000000:
        return library_version.decode('utf-8')
    else:
        return library_version

def clean_column_names(df: DataFrame) -> DataFrame:
    """
    Strip the whitespace from all column names in the given DataFrame
    and return the result.
    """
    f = df.copy()
    f.columns = [col.strip() for col in f.columns]
    return f

def full(self):
        """Return ``True`` if the queue is full, ``False``
        otherwise (not reliable!).

        Only applicable if :attr:`maxsize` is set.

        """
        return self.maxsize and len(self.list) >= self.maxsize or False

def split(text: str) -> List[str]:
    """Split a text into a list of tokens.

    :param text: the text to split
    :return: tokens
    """
    return [word for word in SEPARATOR.split(text) if word.strip(' \t')]

def tail(filename, number_of_bytes):
    """Returns the last number_of_bytes of filename"""
    with open(filename, "rb") as f:
        if os.stat(filename).st_size > number_of_bytes:
            f.seek(-number_of_bytes, 2)
        return f.read()

def find_editor() -> str:
    """Find a reasonable editor to use by default for the system that the cmd2 application is running on."""
    editor = os.environ.get('EDITOR')
    if not editor:
        if sys.platform[:3] == 'win':
            editor = 'notepad'
        else:
            # Favor command-line editors first so we don't leave the terminal to edit
            for editor in ['vim', 'vi', 'emacs', 'nano', 'pico', 'gedit', 'kate', 'subl', 'geany', 'atom']:
                if which(editor):
                    break
    return editor

def position(self) -> Position:
        """The current position of the cursor."""
        return Position(self._index, self._lineno, self._col_offset)

def assert_equal(first, second, msg_fmt="{msg}"):
    """Fail unless first equals second, as determined by the '==' operator.

    >>> assert_equal(5, 5.0)
    >>> assert_equal("Hello World!", "Goodbye!")
    Traceback (most recent call last):
        ...
    AssertionError: 'Hello World!' != 'Goodbye!'

    The following msg_fmt arguments are supported:
    * msg - the default error message
    * first - the first argument
    * second - the second argument
    """

    if isinstance(first, dict) and isinstance(second, dict):
        assert_dict_equal(first, second, msg_fmt)
    elif not first == second:
        msg = "{!r} != {!r}".format(first, second)
        fail(msg_fmt.format(msg=msg, first=first, second=second))

def is_not_null(df: DataFrame, col_name: str) -> bool:
    """
    Return ``True`` if the given DataFrame has a column of the given
    name (string), and there exists at least one non-NaN value in that
    column; return ``False`` otherwise.
    """
    if (
        isinstance(df, pd.DataFrame)
        and col_name in df.columns
        and df[col_name].notnull().any()
    ):
        return True
    else:
        return False

def get_pixel(framebuf, x, y):
        """Get the color of a given pixel"""
        index = (y >> 3) * framebuf.stride + x
        offset = y & 0x07
        return (framebuf.buf[index] >> offset) & 0x01

def replaceStrs(s, *args):
    r"""Replace all ``(frm, to)`` tuples in `args` in string `s`.

    >>> replaceStrs("nothing is better than warm beer",
    ...             ('nothing','warm beer'), ('warm beer','nothing'))
    'warm beer is better than nothing'

    """
    if args == (): return s
    mapping = dict((frm, to) for frm, to in args)
    return re.sub("|".join(map(re.escape, mapping.keys())),
                  lambda match:mapping[match.group(0)], s)

def is_sqlatype_text_over_one_char(
        coltype: Union[TypeEngine, VisitableType]) -> bool:
    """
    Is the SQLAlchemy column type a string type that's more than one character
    long?
    """
    coltype = _coltype_to_typeengine(coltype)
    return is_sqlatype_text_of_length_at_least(coltype, 2)

def uconcatenate(arrs, axis=0):
    """Concatenate a sequence of arrays.

    This wrapper around numpy.concatenate preserves units. All input arrays
    must have the same units.  See the documentation of numpy.concatenate for
    full details.

    Examples
    --------
    >>> from unyt import cm
    >>> A = [1, 2, 3]*cm
    >>> B = [2, 3, 4]*cm
    >>> uconcatenate((A, B))
    unyt_array([1, 2, 3, 2, 3, 4], 'cm')

    """
    v = np.concatenate(arrs, axis=axis)
    v = _validate_numpy_wrapper_units(v, arrs)
    return v

def try_cast_int(s):
    """(str) -> int
    All the digits in a given string are concatenated and converted into a single number.
    """
    try:
        temp = re.findall('\d', str(s))
        temp = ''.join(temp)
        return int(temp)
    except:
        return s

def has_value(cls, value: int) -> bool:
        """True if specified value exists in int enum; otherwise, False."""
        return any(value == item.value for item in cls)

def s3_get(url: str, temp_file: IO) -> None:
    """Pull a file directly from S3."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)

def cli_run():
    """docstring for argparse"""
    parser = argparse.ArgumentParser(description='Stupidly simple code answers from StackOverflow')
    parser.add_argument('query', help="What's the problem ?", type=str, nargs='+')
    parser.add_argument('-t','--tags', help='semicolon separated tags -> python;lambda')
    args = parser.parse_args()
    main(args)

def __remove_trailing_zeros(self, collection):
        """Removes trailing zeroes from indexable collection of numbers"""
        index = len(collection) - 1
        while index >= 0 and collection[index] == 0:
            index -= 1

        return collection[:index + 1]

def get_view_selection(self):
        """Get actual tree selection object and all respective models of selected rows"""
        if not self.MODEL_STORAGE_ID:
            return None, None

        # avoid selection requests on empty tree views -> case warnings in gtk3
        if len(self.store) == 0:
            paths = []
        else:
            model, paths = self._tree_selection.get_selected_rows()

        # get all related models for selection from respective tree store field
        selected_model_list = []
        for path in paths:
            model = self.store[path][self.MODEL_STORAGE_ID]
            selected_model_list.append(model)
        return self._tree_selection, selected_model_list

def rate_limited(max_per_hour: int, *args: Any) -> Callable[..., Any]:
    """Rate limit a function."""
    return util.rate_limited(max_per_hour, *args)

def _gauss(mean: int, sigma: int) -> int:
        """
        Creates a variation from a base value

        Args:
            mean: base value
            sigma: gaussian sigma

        Returns: random value

        """
        return int(random.gauss(mean, sigma))

def cpu_count() -> int:
    """Returns the number of processors on this machine."""
    if multiprocessing is None:
        return 1
    try:
        return multiprocessing.cpu_count()
    except NotImplementedError:
        pass
    try:
        return os.sysconf("SC_NPROCESSORS_CONF")
    except (AttributeError, ValueError):
        pass
    gen_log.error("Could not detect number of processors; assuming 1")
    return 1

def zfill(x, width):
    """zfill(x, width) -> string

    Pad a numeric string x with zeros on the left, to fill a field
    of the specified width.  The string x is never truncated.

    """
    if not isinstance(x, basestring):
        x = repr(x)
    return x.zfill(width)

def is_not_null(df: DataFrame, col_name: str) -> bool:
    """
    Return ``True`` if the given DataFrame has a column of the given
    name (string), and there exists at least one non-NaN value in that
    column; return ``False`` otherwise.
    """
    if (
        isinstance(df, pd.DataFrame)
        and col_name in df.columns
        and df[col_name].notnull().any()
    ):
        return True
    else:
        return False

def __replace_all(repls: dict, input: str) -> str:
    """ Replaces from a string **input** all the occurrences of some
    symbols according to mapping **repls**.

    :param dict repls: where #key is the old character and
    #value is the one to substitute with;
    :param str input: original string where to apply the
    replacements;
    :return: *(str)* the string with the desired characters replaced
    """
    return re.sub('|'.join(re.escape(key) for key in repls.keys()),
                  lambda k: repls[k.group(0)], input)

def _protected_log(x1):
    """Closure of log for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)

def mmap(func, iterable):
    """Wrapper to make map() behave the same on Py2 and Py3."""

    if sys.version_info[0] > 2:
        return [i for i in map(func, iterable)]
    else:
        return map(func, iterable)

def url_host(url: str) -> str:
    """
    Parses hostname from URL.
    :param url: URL
    :return: hostname
    """
    from urllib.parse import urlparse
    res = urlparse(url)
    return res.netloc.split(':')[0] if res.netloc else ''

def __remove_trailing_zeros(self, collection):
        """Removes trailing zeroes from indexable collection of numbers"""
        index = len(collection) - 1
        while index >= 0 and collection[index] == 0:
            index -= 1

        return collection[:index + 1]

def most_significant_bit(lst: np.ndarray) -> int:
    """
    A helper function that finds the position of the most significant bit in a 1darray of 1s and 0s,
    i.e. the first position where a 1 appears, reading left to right.

    :param lst: a 1d array of 0s and 1s with at least one 1
    :return: the first position in lst that a 1 appears
    """
    return np.argwhere(np.asarray(lst) == 1)[0][0]

def expired(self):
        """Boolean property if this action has expired
        """
        if self.timeout is None:
            return False

        return monotonic() - self.start_time > self.timeout

def issubset(self, other):
        """
        Report whether another set contains this set.

        Example:
            >>> OrderedSet([1, 2, 3]).issubset({1, 2})
            False
            >>> OrderedSet([1, 2, 3]).issubset({1, 2, 3, 4})
            True
            >>> OrderedSet([1, 2, 3]).issubset({1, 4, 3, 5})
            False
        """
        if len(self) > len(other):  # Fast check for obvious cases
            return False
        return all(item in other for item in self)

def iprotate(l, steps=1):
    r"""Like rotate, but modifies `l` in-place.

    >>> l = [1,2,3]
    >>> iprotate(l) is l
    True
    >>> l
    [2, 3, 1]
    >>> iprotate(iprotate(l, 2), -3)
    [1, 2, 3]

    """
    if len(l):
        steps %= len(l)
        if steps:
            firstPart = l[:steps]
            del l[:steps]
            l.extend(firstPart)
    return l

def _extension(modpath: str) -> setuptools.Extension:
    """Make setuptools.Extension."""
    return setuptools.Extension(modpath, [modpath.replace(".", "/") + ".py"])

def url_host(url: str) -> str:
    """
    Parses hostname from URL.
    :param url: URL
    :return: hostname
    """
    from urllib.parse import urlparse
    res = urlparse(url)
    return res.netloc.split(':')[0] if res.netloc else ''

def is_unicode(string):
    """Validates that the object itself is some kinda string"""
    str_type = str(type(string))

    if str_type.find('str') > 0 or str_type.find('unicode') > 0:
        return True

    return False

def get_last_day_of_month(t: datetime) -> int:
    """
    Returns day number of the last day of the month
    :param t: datetime
    :return: int
    """
    tn = t + timedelta(days=32)
    tn = datetime(year=tn.year, month=tn.month, day=1)
    tt = tn - timedelta(hours=1)
    return tt.day

def position(self) -> Position:
        """The current position of the cursor."""
        return Position(self._index, self._lineno, self._col_offset)

def timeit(func, *args, **kwargs):
    """
    Time execution of function. Returns (res, seconds).

    >>> res, timing = timeit(time.sleep, 1)
    """
    start_time = time.time()
    res = func(*args, **kwargs)
    timing = time.time() - start_time
    return res, timing

def _str_to_list(value, separator):
    """Convert a string to a list with sanitization."""
    value_list = [item.strip() for item in value.split(separator)]
    value_list_sanitized = builtins.list(filter(None, value_list))
    if len(value_list_sanitized) > 0:
        return value_list_sanitized
    else:
        raise ValueError('Invalid list variable.')

def last(self):
        """Last time step available.

        Example:
            >>> sdat = StagyyData('path/to/run')
            >>> assert(sdat.steps.last is sdat.steps[-1])
        """
        if self._last is UNDETERMINED:
            # not necessarily the last one...
            self._last = self.sdat.tseries.index[-1]
        return self[self._last]

def getIndex(predicateFn: Callable[[T], bool], items: List[T]) -> int:
    """
    Finds the index of an item in list, which satisfies predicate
    :param predicateFn: predicate function to run on items of list
    :param items: list of tuples
    :return: first index for which predicate function returns True
    """
    try:
        return next(i for i, v in enumerate(items) if predicateFn(v))
    except StopIteration:
        return -1

def fib(n):
    """Fibonacci example function

    Args:
      n (int): integer

    Returns:
      int: n-th Fibonacci number
    """
    assert n > 0
    a, b = 1, 1
    for i in range(n - 1):
        a, b = b, a + b
    return a

def suppress_stdout():
    """
    Context manager that suppresses stdout.

    Examples:
        >>> with suppress_stdout():
        ...     print('Test print')

        >>> print('test')
        test

    """
    save_stdout = sys.stdout
    sys.stdout = DevNull()
    yield
    sys.stdout = save_stdout

def dtypes(self):
        """Returns all column names and their data types as a list.

        >>> df.dtypes
        [('age', 'int'), ('name', 'string')]
        """
        return [(str(f.name), f.dataType.simpleString()) for f in self.schema.fields]

def rate_limited(max_per_hour: int, *args: Any) -> Callable[..., Any]:
    """Rate limit a function."""
    return util.rate_limited(max_per_hour, *args)

def s3_get(url: str, temp_file: IO) -> None:
    """Pull a file directly from S3."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)

def iprotate(l, steps=1):
    r"""Like rotate, but modifies `l` in-place.

    >>> l = [1,2,3]
    >>> iprotate(l) is l
    True
    >>> l
    [2, 3, 1]
    >>> iprotate(iprotate(l, 2), -3)
    [1, 2, 3]

    """
    if len(l):
        steps %= len(l)
        if steps:
            firstPart = l[:steps]
            del l[:steps]
            l.extend(firstPart)
    return l

def has_key(cls, *args):
        """
        Check whether flyweight object with specified key has already been created.

        Returns:
            bool: True if already created, False if not
        """
        key = args if len(args) > 1 else args[0]
        return key in cls._instances

def array2string(arr: numpy.ndarray) -> str:
        """Format numpy array as a string."""
        shape = str(arr.shape)[1:-1]
        if shape.endswith(","):
            shape = shape[:-1]
        return numpy.array2string(arr, threshold=11) + "%s[%s]" % (arr.dtype, shape)

def rmglob(pattern: str) -> None:
    """
    Deletes all files whose filename matches the glob ``pattern`` (via
    :func:`glob.glob`).
    """
    for f in glob.glob(pattern):
        os.remove(f)

def get_property_as_float(self, name: str) -> float:
        """Return the value of a float property.

        :return: The property value (float).

        Raises exception if property with name doesn't exist.

        .. versionadded:: 1.0

        Scriptable: Yes
        """
        return float(self.__instrument.get_property(name))

def zfill(x, width):
    """zfill(x, width) -> string

    Pad a numeric string x with zeros on the left, to fill a field
    of the specified width.  The string x is never truncated.

    """
    if not isinstance(x, basestring):
        x = repr(x)
    return x.zfill(width)

def get_column_names(engine: Engine, tablename: str) -> List[str]:
    """
    Get all the database column names for the specified table.
    """
    return [info.name for info in gen_columns_info(engine, tablename)]

def _exit(self, status_code):
        """Properly kill Python process including zombie threads."""
        # If there are active threads still running infinite loops, sys.exit
        # won't kill them but os._exit will. os._exit skips calling cleanup
        # handlers, flushing stdio buffers, etc.
        exit_func = os._exit if threading.active_count() > 1 else sys.exit
        exit_func(status_code)

def _duplicates(list_):
    """Return dict mapping item -> indices."""
    item_indices = {}
    for i, item in enumerate(list_):
        try:
            item_indices[item].append(i)
        except KeyError:  # First time seen
            item_indices[item] = [i]
    return item_indices

def get_column_names(engine: Engine, tablename: str) -> List[str]:
    """
    Get all the database column names for the specified table.
    """
    return [info.name for info in gen_columns_info(engine, tablename)]

def remove_once(gset, elem):
    """Remove the element from a set, lists or dict.
    
        >>> L = ["Lucy"]; S = set(["Sky"]); D = { "Diamonds": True };
        >>> remove_once(L, "Lucy"); remove_once(S, "Sky"); remove_once(D, "Diamonds");
        >>> print L, S, D
        [] set([]) {}

    Returns the element if it was removed. Raises one of the exceptions in 
    :obj:`RemoveError` otherwise.
    """
    remove = getattr(gset, 'remove', None)
    if remove is not None: remove(elem)
    else: del gset[elem]
    return elem

def remove_blank_lines(string):
    """ Removes all blank lines in @string

        -> #str without blank lines
    """
    return "\n".join(line
                     for line in string.split("\n")
                     if len(line.strip()))

def rl_get_point() -> int:  # pragma: no cover
    """
    Returns the offset of the current cursor position in rl_line_buffer
    """
    if rl_type == RlType.GNU:
        return ctypes.c_int.in_dll(readline_lib, "rl_point").value

    elif rl_type == RlType.PYREADLINE:
        return readline.rl.mode.l_buffer.point

    else:
        return 0

def get_pylint_options(config_dir='.'):
    # type: (str) -> List[str]
    """Checks for local config overrides for `pylint`
    and add them in the correct `pylint` `options` format.

    :param config_dir:
    :return: List [str]
    """
    if PYLINT_CONFIG_NAME in os.listdir(config_dir):
        pylint_config_path = PYLINT_CONFIG_NAME
    else:
        pylint_config_path = DEFAULT_PYLINT_CONFIG_PATH

    return ['--rcfile={}'.format(pylint_config_path)]

def _skip(self, cnt):
        """Read and discard data"""
        while cnt > 0:
            if cnt > 8192:
                buf = self.read(8192)
            else:
                buf = self.read(cnt)
            if not buf:
                break
            cnt -= len(buf)

def get_domain(url):
    """
    Get domain part of an url.

    For example: https://www.python.org/doc/ -> https://www.python.org
    """
    parse_result = urlparse(url)
    domain = "{schema}://{netloc}".format(
        schema=parse_result.scheme, netloc=parse_result.netloc)
    return domain

def _cnx_is_empty(in_file):
    """Check if cnr or cns files are empty (only have a header)
    """
    with open(in_file) as in_handle:
        for i, line in enumerate(in_handle):
            if i > 0:
                return False
    return True

def argmax(iterable, key=None, both=False):
    """
    >>> argmax([4,2,-5])
    0
    >>> argmax([4,2,-5], key=abs)
    2
    >>> argmax([4,2,-5], key=abs, both=True)
    (2, 5)
    """
    if key is not None:
        it = imap(key, iterable)
    else:
        it = iter(iterable)
    score, argmax = reduce(max, izip(it, count()))
    if both:
        return argmax, score
    return argmax

def bfx(value, msb, lsb):
    """! @brief Extract a value from a bitfield."""
    mask = bitmask((msb, lsb))
    return (value & mask) >> lsb

def argmax(self, rows: List[Row], column: ComparableColumn) -> List[Row]:
        """
        Takes a list of rows and a column name and returns a list containing a single row (dict from
        columns to cells) that has the maximum numerical value in the given column. We return a list
        instead of a single dict to be consistent with the return type of ``select`` and
        ``all_rows``.
        """
        if not rows:
            return []
        value_row_pairs = [(row.values[column.name], row) for row in rows]
        if not value_row_pairs:
            return []
        # Returns a list containing the row with the max cell value.
        return [sorted(value_row_pairs, key=lambda x: x[0], reverse=True)[0][1]]

def is_relative_url(url):
    """ simple method to determine if a url is relative or absolute """
    if url.startswith("#"):
        return None
    if url.find("://") > 0 or url.startswith("//"):
        # either 'http(s)://...' or '//cdn...' and therefore absolute
        return False
    return True

def array2string(arr: numpy.ndarray) -> str:
        """Format numpy array as a string."""
        shape = str(arr.shape)[1:-1]
        if shape.endswith(","):
            shape = shape[:-1]
        return numpy.array2string(arr, threshold=11) + "%s[%s]" % (arr.dtype, shape)

def _get_or_default(mylist, i, default=None):
    """return list item number, or default if don't exist"""
    if i >= len(mylist):
        return default
    else :
        return mylist[i]

def hex_to_int(value):
    """
    Convert hex string like "\x0A\xE3" to 2787.
    """
    if version_info.major >= 3:
        return int.from_bytes(value, "big")
    return int(value.encode("hex"), 16)

def remove_links(text):
    """
    Helper function to remove the links from the input text

    Args:
        text (str): A string

    Returns:
        str: the same text, but with any substring that matches the regex
        for a link removed and replaced with a space

    Example:
        >>> from tweet_parser.getter_methods.tweet_text import remove_links
        >>> text = "lorem ipsum dolor https://twitter.com/RobotPrincessFi"
        >>> remove_links(text)
        'lorem ipsum dolor  '
    """
    tco_link_regex = re.compile("https?://t.co/[A-z0-9].*")
    generic_link_regex = re.compile("(https?://)?(\w*[.]\w+)+([/?=&]+\w+)*")
    remove_tco = re.sub(tco_link_regex, " ", text)
    remove_generic = re.sub(generic_link_regex, " ", remove_tco)
    return remove_generic

def _(f, x):
    """
    filter for dict, note `f` should have signature: `f::key->value->bool`
    """
    return {k: v for k, v in x.items() if f(k, v)}

def validate(request: Union[Dict, List], schema: dict) -> Union[Dict, List]:
    """
    Wraps jsonschema.validate, returning the same object passed in.

    Args:
        request: The deserialized-from-json request.
        schema: The jsonschema schema to validate against.

    Raises:
        jsonschema.ValidationError
    """
    jsonschema_validate(request, schema)
    return request

def load_yaml(file):
    """If pyyaml > 5.1 use full_load to avoid warning"""
    if hasattr(yaml, "full_load"):
        return yaml.full_load(file)
    else:
        return yaml.load(file)

def astensor(array: TensorLike) -> BKTensor:
    """Covert numpy array to tensorflow tensor"""
    tensor = tf.convert_to_tensor(value=array, dtype=CTYPE)
    return tensor

def hsv2rgb_spectrum(hsv):
    """Generates RGB values from HSV values in line with a typical light
    spectrum."""
    h, s, v = hsv
    return hsv2rgb_raw(((h * 192) >> 8, s, v))

def format_exp_floats(decimals):
    """
    sometimes the exp. column can be too large
    """
    threshold = 10 ** 5
    return (
        lambda n: "{:.{prec}e}".format(n, prec=decimals) if n > threshold else "{:4.{prec}f}".format(n, prec=decimals)
    )

def login(self, user: str, passwd: str) -> None:
        """Log in to instagram with given username and password and internally store session object.

        :raises InvalidArgumentException: If the provided username does not exist.
        :raises BadCredentialsException: If the provided password is wrong.
        :raises ConnectionException: If connection to Instagram failed.
        :raises TwoFactorAuthRequiredException: First step of 2FA login done, now call :meth:`Instaloader.two_factor_login`."""
        self.context.login(user, passwd)

def is_not_null(df: DataFrame, col_name: str) -> bool:
    """
    Return ``True`` if the given DataFrame has a column of the given
    name (string), and there exists at least one non-NaN value in that
    column; return ``False`` otherwise.
    """
    if (
        isinstance(df, pd.DataFrame)
        and col_name in df.columns
        and df[col_name].notnull().any()
    ):
        return True
    else:
        return False

def quaternion_imag(quaternion):
    """Return imaginary part of quaternion.

    >>> quaternion_imag([3, 0, 1, 2])
    array([0., 1., 2.])

    """
    return np.array(quaternion[1:4], dtype=np.float64, copy=True)

def _skip_section(self):
        """Skip a section"""
        self._last = self._f.readline()
        while len(self._last) > 0 and len(self._last[0].strip()) == 0:
            self._last = self._f.readline()

def is_natural(x):
    """A non-negative integer."""
    try:
        is_integer = int(x) == x
    except (TypeError, ValueError):
        return False
    return is_integer and x >= 0

def stretch(iterable, n=2):
    r"""Repeat each item in `iterable` `n` times.

    Example:

    >>> list(stretch(range(3), 2))
    [0, 0, 1, 1, 2, 2]
    """
    times = range(n)
    for item in iterable:
        for i in times: yield item

def remove_leading_zeros(num: str) -> str:
    """
    Strips zeros while handling -, M, and empty strings
    """
    if not num:
        return num
    if num.startswith('M'):
        ret = 'M' + num[1:].lstrip('0')
    elif num.startswith('-'):
        ret = '-' + num[1:].lstrip('0')
    else:
        ret = num.lstrip('0')
    return '0' if ret in ('', 'M', '-') else ret

def _create_empty_array(self, frames, always_2d, dtype):
        """Create an empty array with appropriate shape."""
        import numpy as np
        if always_2d or self.channels > 1:
            shape = frames, self.channels
        else:
            shape = frames,
        return np.empty(shape, dtype, order='C')

def increment_frame(self):
        """Increment a frame of the animation."""
        self.current_frame += 1

        if self.current_frame >= self.end_frame:
            # Wrap back to the beginning of the animation.
            self.current_frame = 0

def is_integer(value: Any) -> bool:
    """Return true if a value is an integer number."""
    return (isinstance(value, int) and not isinstance(value, bool)) or (
        isinstance(value, float) and isfinite(value) and int(value) == value
    )

def samefile(a: str, b: str) -> bool:
    """Check if two pathes represent the same file."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normpath(a) == os.path.normpath(b)

def __replace_all(repls: dict, input: str) -> str:
    """ Replaces from a string **input** all the occurrences of some
    symbols according to mapping **repls**.

    :param dict repls: where #key is the old character and
    #value is the one to substitute with;
    :param str input: original string where to apply the
    replacements;
    :return: *(str)* the string with the desired characters replaced
    """
    return re.sub('|'.join(re.escape(key) for key in repls.keys()),
                  lambda k: repls[k.group(0)], input)

def dotproduct(X, Y):
    """Return the sum of the element-wise product of vectors x and y.
    >>> dotproduct([1, 2, 3], [1000, 100, 10])
    1230
    """
    return sum([x * y for x, y in zip(X, Y)])

async def fetchall(self) -> Iterable[sqlite3.Row]:
        """Fetch all remaining rows."""
        return await self._execute(self._cursor.fetchall)

def auto_up(self, count=1, go_to_start_of_line_if_history_changes=False):
        """
        If we're not on the first line (of a multiline input) go a line up,
        otherwise go back in history. (If nothing is selected.)
        """
        if self.complete_state:
            self.complete_previous(count=count)
        elif self.document.cursor_position_row > 0:
            self.cursor_up(count=count)
        elif not self.selection_state:
            self.history_backward(count=count)

            # Go to the start of the line?
            if go_to_start_of_line_if_history_changes:
                self.cursor_position += self.document.get_start_of_line_position()

def rate_limited(max_per_hour: int, *args: Any) -> Callable[..., Any]:
    """Rate limit a function."""
    return util.rate_limited(max_per_hour, *args)

def hex_to_int(value):
    """
    Convert hex string like "\x0A\xE3" to 2787.
    """
    if version_info.major >= 3:
        return int.from_bytes(value, "big")
    return int(value.encode("hex"), 16)

def is_integer(value: Any) -> bool:
    """Return true if a value is an integer number."""
    return (isinstance(value, int) and not isinstance(value, bool)) or (
        isinstance(value, float) and isfinite(value) and int(value) == value
    )

def de_duplicate(items):
    """Remove any duplicate item, preserving order

    >>> de_duplicate([1, 2, 1, 2])
    [1, 2]
    """
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return result

def uppercase_chars(string: any) -> str:
        """Return all (and only) the uppercase chars in the given string."""
        return ''.join([c if c.isupper() else '' for c in str(string)])

def cli_run():
    """docstring for argparse"""
    parser = argparse.ArgumentParser(description='Stupidly simple code answers from StackOverflow')
    parser.add_argument('query', help="What's the problem ?", type=str, nargs='+')
    parser.add_argument('-t','--tags', help='semicolon separated tags -> python;lambda')
    args = parser.parse_args()
    main(args)

def _duplicates(list_):
    """Return dict mapping item -> indices."""
    item_indices = {}
    for i, item in enumerate(list_):
        try:
            item_indices[item].append(i)
        except KeyError:  # First time seen
            item_indices[item] = [i]
    return item_indices

def is_line_in_file(filename: str, line: str) -> bool:
    """
    Detects whether a line is present within a file.

    Args:
        filename: file to check
        line: line to search for (as an exact match)
    """
    assert "\n" not in line
    with open(filename, "r") as file:
        for fileline in file:
            if fileline == line:
                return True
        return False

def closest_values(L):
    """Closest values

    :param L: list of values
    :returns: two values from L with minimal distance
    :modifies: the order of L
    :complexity: O(n log n), for n=len(L)
    """
    assert len(L) >= 2
    L.sort()
    valmin, argmin = min((L[i] - L[i - 1], i) for i in range(1, len(L)))
    return L[argmin - 1], L[argmin]

def singularize(word):
    """
    Return the singular form of a word, the reverse of :func:`pluralize`.

    Examples::

        >>> singularize("posts")
        "post"
        >>> singularize("octopi")
        "octopus"
        >>> singularize("sheep")
        "sheep"
        >>> singularize("word")
        "word"
        >>> singularize("CamelOctopi")
        "CamelOctopus"

    """
    for inflection in UNCOUNTABLES:
        if re.search(r'(?i)\b(%s)\Z' % inflection, word):
            return word

    for rule, replacement in SINGULARS:
        if re.search(rule, word):
            return re.sub(rule, replacement, word)
    return word

def factors(n):
    """
    Computes all the integer factors of the number `n`

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_alg import *  # NOQA
        >>> import utool as ut
        >>> result = sorted(ut.factors(10))
        >>> print(result)
        [1, 2, 5, 10]

    References:
        http://stackoverflow.com/questions/6800193/finding-all-the-factors
    """
    return set(reduce(list.__add__,
                      ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))

def get_environment_info() -> dict:
    """
    Information about Cauldron and its Python interpreter.

    :return:
        A dictionary containing information about the Cauldron and its
        Python environment. This information is useful when providing feedback
        and bug reports.
    """
    data = _environ.systems.get_system_data()
    data['cauldron'] = _environ.package_settings.copy()
    return data

def _rindex(mylist: Sequence[T], x: T) -> int:
    """Index of the last occurrence of x in the sequence."""
    return len(mylist) - mylist[::-1].index(x) - 1

def dotproduct(X, Y):
    """Return the sum of the element-wise product of vectors x and y.
    >>> dotproduct([1, 2, 3], [1000, 100, 10])
    1230
    """
    return sum([x * y for x, y in zip(X, Y)])

def remove_empty_text(utterances: List[Utterance]) -> List[Utterance]:
    """Remove empty utterances from a list of utterances
    Args:
        utterances: The list of utterance we are processing
    """
    return [utter for utter in utterances if utter.text.strip() != ""]

def detect_model_num(string):
    """Takes a string related to a model name and extract its model number.

    For example:
        '000000-bootstrap.index' => 0
    """
    match = re.match(MODEL_NUM_REGEX, string)
    if match:
        return int(match.group())
    return None

def is_sqlatype_integer(coltype: Union[TypeEngine, VisitableType]) -> bool:
    """
    Is the SQLAlchemy column type an integer type?
    """
    coltype = _coltype_to_typeengine(coltype)
    return isinstance(coltype, sqltypes.Integer)

def uniqued(iterable):
    """Return unique list of ``iterable`` items preserving order.

    >>> uniqued('spameggs')
    ['s', 'p', 'a', 'm', 'e', 'g']
    """
    seen = set()
    return [item for item in iterable if item not in seen and not seen.add(item)]

def auto_up(self, count=1, go_to_start_of_line_if_history_changes=False):
        """
        If we're not on the first line (of a multiline input) go a line up,
        otherwise go back in history. (If nothing is selected.)
        """
        if self.complete_state:
            self.complete_previous(count=count)
        elif self.document.cursor_position_row > 0:
            self.cursor_up(count=count)
        elif not self.selection_state:
            self.history_backward(count=count)

            # Go to the start of the line?
            if go_to_start_of_line_if_history_changes:
                self.cursor_position += self.document.get_start_of_line_position()

def pruning(self, X, y, cost_mat):
        """ Function that prune the decision tree.

        Parameters
        ----------

        X : array-like of shape = [n_samples, n_features]
            The input samples.

        y_true : array indicator matrix
            Ground truth (correct) labels.

        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        """
        self.tree_.tree_pruned = copy.deepcopy(self.tree_.tree)
        if self.tree_.n_nodes > 0:
            self._pruning(X, y, cost_mat)
            nodes_pruned = self._nodes(self.tree_.tree_pruned)
            self.tree_.n_nodes_pruned = len(nodes_pruned)

def get_from_gnucash26_date(date_str: str) -> date:
    """ Creates a datetime from GnuCash 2.6 date string """
    date_format = "%Y%m%d"
    result = datetime.strptime(date_str, date_format).date()
    return result

def branches():
    # type: () -> List[str]
    """ Return a list of branches in the current repo.

    Returns:
        list[str]: A list of branches in the current repo.
    """
    out = shell.run(
        'git branch',
        capture=True,
        never_pretend=True
    ).stdout.strip()
    return [x.strip('* \t\n') for x in out.splitlines()]

def content_type(self) -> ContentType:
        """Return receiver's content type."""
        return self._ctype if self._ctype else self.parent.content_type()

def butlast(iterable):
    """Yield all items from ``iterable`` except the last one.

    >>> list(butlast(['spam', 'eggs', 'ham']))
    ['spam', 'eggs']

    >>> list(butlast(['spam']))
    []

    >>> list(butlast([]))
    []
    """
    iterable = iter(iterable)
    try:
        first = next(iterable)
    except StopIteration:
        return
    for second in iterable:
        yield first
        first = second

def get_keys_of_max_n(dict_obj, n):
    """Returns the keys that maps to the top n max values in the given dict.

    Example:
    --------
    >>> dict_obj = {'a':2, 'b':1, 'c':5}
    >>> get_keys_of_max_n(dict_obj, 2)
    ['a', 'c']
    """
    return sorted([
        item[0]
        for item in sorted(
            dict_obj.items(), key=lambda item: item[1], reverse=True
        )[:n]
    ])

def _parse_tuple_string(argument):
        """ Return a tuple from parsing 'a,b,c,d' -> (a,b,c,d) """
        if isinstance(argument, str):
            return tuple(int(p.strip()) for p in argument.split(','))
        return argument

def clean_column_names(df: DataFrame) -> DataFrame:
    """
    Strip the whitespace from all column names in the given DataFrame
    and return the result.
    """
    f = df.copy()
    f.columns = [col.strip() for col in f.columns]
    return f

def _rindex(mylist: Sequence[T], x: T) -> int:
    """Index of the last occurrence of x in the sequence."""
    return len(mylist) - mylist[::-1].index(x) - 1

def is_relative_url(url):
    """ simple method to determine if a url is relative or absolute """
    if url.startswith("#"):
        return None
    if url.find("://") > 0 or url.startswith("//"):
        # either 'http(s)://...' or '//cdn...' and therefore absolute
        return False
    return True

def fetchallfirstvalues(self, sql: str, *args) -> List[Any]:
        """Executes SQL; returns list of first values of each row."""
        rows = self.fetchall(sql, *args)
        return [row[0] for row in rows]

def url_concat(url, args):
    """Concatenate url and argument dictionary regardless of whether
    url has existing query parameters.

    >>> url_concat("http://example.com/foo?a=b", dict(c="d"))
    'http://example.com/foo?a=b&c=d'
    """
    if not args: return url
    if url[-1] not in ('?', '&'):
        url += '&' if ('?' in url) else '?'
    return url + urllib.urlencode(args)

def proper_round(n):
    """
    rounds float to closest int
    :rtype: int
    :param n: float
    """
    return int(n) + (n / abs(n)) * int(abs(n - int(n)) >= 0.5) if n != 0 else 0

def uppercase_chars(string: any) -> str:
        """Return all (and only) the uppercase chars in the given string."""
        return ''.join([c if c.isupper() else '' for c in str(string)])

def _prm_get_longest_stringsize(string_list):
        """ Returns the longest string size for a string entry across data."""
        maxlength = 1

        for stringar in string_list:
            if isinstance(stringar, np.ndarray):
                if stringar.ndim > 0:
                    for string in stringar.ravel():
                        maxlength = max(len(string), maxlength)
                else:
                    maxlength = max(len(stringar.tolist()), maxlength)
            else:
                maxlength = max(len(stringar), maxlength)

        # Make the string Col longer than needed in order to allow later on slightly larger strings
        return int(maxlength * 1.5)

def flatten_list(x: List[Any]) -> List[Any]:
    """
    Converts a list of lists into a flat list.
    
    Args:
        x: list of lists 

    Returns:
        flat list
        
    As per
    http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python

    """  # noqa
    return [item for sublist in x for item in sublist]

def count(args):
    """ count occurences in a list of lists
    >>> count([['a','b'],['a']])
    defaultdict(int, {'a' : 2, 'b' : 1})
    """
    counts = defaultdict(int)
    for arg in args:
        for item in arg:
            counts[item] = counts[item] + 1
    return counts

def do_quit(self, _: argparse.Namespace) -> bool:
        """Exit this application"""
        self._should_quit = True
        return self._STOP_AND_EXIT

def replace_in_list(stringlist: Iterable[str],
                    replacedict: Dict[str, str]) -> List[str]:
    """
    Returns a list produced by applying :func:`multiple_replace` to every
    string in ``stringlist``.

    Args:
        stringlist: list of source strings
        replacedict: dictionary mapping "original" to "replacement" strings

    Returns:
        list of final strings

    """
    newlist = []
    for fromstring in stringlist:
        newlist.append(multiple_replace(fromstring, replacedict))
    return newlist

def is_not_null(df: DataFrame, col_name: str) -> bool:
    """
    Return ``True`` if the given DataFrame has a column of the given
    name (string), and there exists at least one non-NaN value in that
    column; return ``False`` otherwise.
    """
    if (
        isinstance(df, pd.DataFrame)
        and col_name in df.columns
        and df[col_name].notnull().any()
    ):
        return True
    else:
        return False

def _hash_the_file(hasher, filename):
    """Helper function for creating hash functions.

    See implementation of :func:`dtoolcore.filehasher.shasum`
    for more usage details.
    """
    BUF_SIZE = 65536
    with open(filename, 'rb') as f:
        buf = f.read(BUF_SIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(BUF_SIZE)
    return hasher

def _my_hash(arg_list):
    # type: (List[Any]) -> int
    """Simple helper hash function"""
    res = 0
    for arg in arg_list:
        res = res * 31 + hash(arg)
    return res

def fix_missing(df, col, name, na_dict):
    """ Fill missing data in a column of df with the median, and add a {name}_na column
    which specifies if the data was missing.
    Parameters:
    -----------
    df: The data frame that will be changed.
    col: The column of data to fix by filling in missing data.
    name: The name of the new filled column in df.
    na_dict: A dictionary of values to create na's of and the value to insert. If
        name is not a key of na_dict the median will fill any missing data. Also
        if name is not a key of na_dict and there is no missing data in col, then
        no {name}_na column is not created.
    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2
    >>> fix_missing(df, df['col1'], 'col1', {})
    >>> df
       col1 col2 col1_na
    0     1    5   False
    1     2    2    True
    2     3    2   False
    >>> df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2
    >>> fix_missing(df, df['col2'], 'col2', {})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2
    >>> df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2
    >>> fix_missing(df, df['col1'], 'col1', {'col1' : 500})
    >>> df
       col1 col2 col1_na
    0     1    5   False
    1   500    2    True
    2     3    2   False
    """
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict):
            df[name+'_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict

def indexes_equal(a: Index, b: Index) -> bool:
    """
    Are two indexes equal? Checks by comparing ``str()`` versions of them.
    (AM UNSURE IF THIS IS ENOUGH.)
    """
    return str(a) == str(b)

def is_natural(x):
    """A non-negative integer."""
    try:
        is_integer = int(x) == x
    except (TypeError, ValueError):
        return False
    return is_integer and x >= 0

def listify(a):
    """
    Convert a scalar ``a`` to a list and all iterables to list as well.

    Examples
    --------
    >>> listify(0)
    [0]

    >>> listify([1,2,3])
    [1, 2, 3]

    >>> listify('a')
    ['a']

    >>> listify(np.array([1,2,3]))
    [1, 2, 3]

    >>> listify('string')
    ['string']
    """
    if a is None:
        return []
    elif not isinstance(a, (tuple, list, np.ndarray)):
        return [a]
    return list(a)

def same(*values):
    """
    Check if all values in a sequence are equal.

    Returns True on empty sequences.

    Examples
    --------
    >>> same(1, 1, 1, 1)
    True
    >>> same(1, 2, 1)
    False
    >>> same()
    True
    """
    if not values:
        return True
    first, rest = values[0], values[1:]
    return all(value == first for value in rest)

def is_string_dtype(arr_or_dtype):
    """
    Check whether the provided array or dtype is of the string dtype.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of the string dtype.

    Examples
    --------
    >>> is_string_dtype(str)
    True
    >>> is_string_dtype(object)
    True
    >>> is_string_dtype(int)
    False
    >>>
    >>> is_string_dtype(np.array(['a', 'b']))
    True
    >>> is_string_dtype(pd.Series([1, 2]))
    False
    """

    # TODO: gh-15585: consider making the checks stricter.
    def condition(dtype):
        return dtype.kind in ('O', 'S', 'U') and not is_period_dtype(dtype)
    return _is_dtype(arr_or_dtype, condition)

def is_prime(n):
    """
    Check if n is a prime number
    """
    if n % 2 == 0 and n > 2:
        return False
    return all(n % i for i in range(3, int(math.sqrt(n)) + 1, 2))

def trunc(obj, max, left=0):
    """
    Convert `obj` to string, eliminate newlines and truncate the string to `max`
    characters. If there are more characters in the string add ``...`` to the
    string. With `left=True`, the string can be truncated at the beginning.

    @note: Does not catch exceptions when converting `obj` to string with `str`.

    >>> trunc('This is a long text.', 8)
    This ...
    >>> trunc('This is a long text.', 8, left)
    ...text.
    """
    s = str(obj)
    s = s.replace('\n', '|')
    if len(s) > max:
        if left:
            return '...'+s[len(s)-max+3:]
        else:
            return s[:(max-3)]+'...'
    else:
        return s

def decodebytes(input):
    """Decode base64 string to byte array."""
    py_version = sys.version_info[0]
    if py_version >= 3:
        return _decodebytes_py3(input)
    return _decodebytes_py2(input)

def snake_to_camel(s: str) -> str:
    """Convert string from snake case to camel case."""

    fragments = s.split('_')

    return fragments[0] + ''.join(x.title() for x in fragments[1:])

def uconcatenate(arrs, axis=0):
    """Concatenate a sequence of arrays.

    This wrapper around numpy.concatenate preserves units. All input arrays
    must have the same units.  See the documentation of numpy.concatenate for
    full details.

    Examples
    --------
    >>> from unyt import cm
    >>> A = [1, 2, 3]*cm
    >>> B = [2, 3, 4]*cm
    >>> uconcatenate((A, B))
    unyt_array([1, 2, 3, 2, 3, 4], 'cm')

    """
    v = np.concatenate(arrs, axis=axis)
    v = _validate_numpy_wrapper_units(v, arrs)
    return v

def mmap(func, iterable):
    """Wrapper to make map() behave the same on Py2 and Py3."""

    if sys.version_info[0] > 2:
        return [i for i in map(func, iterable)]
    else:
        return map(func, iterable)

def getIndex(predicateFn: Callable[[T], bool], items: List[T]) -> int:
    """
    Finds the index of an item in list, which satisfies predicate
    :param predicateFn: predicate function to run on items of list
    :param items: list of tuples
    :return: first index for which predicate function returns True
    """
    try:
        return next(i for i, v in enumerate(items) if predicateFn(v))
    except StopIteration:
        return -1

def count(self, elem):
        """
        Return the number of elements equal to elem present in the queue

        >>> pdeque([1, 2, 1]).count(1)
        2
        """
        return self._left_list.count(elem) + self._right_list.count(elem)

def _close(self):
        """
        Release the USB interface again.
        """
        self._usb_handle.releaseInterface()
        try:
            # If we're using PyUSB >= 1.0 we can re-attach the kernel driver here.
            self._usb_handle.dev.attach_kernel_driver(0)
        except:
            pass
        self._usb_int = None
        self._usb_handle = None
        return True

def long_substr(data):
    """Return the longest common substring in a list of strings.
    
    Credit: http://stackoverflow.com/questions/2892931/longest-common-substring-from-more-than-two-strings-python
    """
    substr = ''
    if len(data) > 1 and len(data[0]) > 0:
        for i in range(len(data[0])):
            for j in range(len(data[0])-i+1):
                if j > len(substr) and all(data[0][i:i+j] in x for x in data):
                    substr = data[0][i:i+j]
    elif len(data) == 1:
        substr = data[0]
    return substr

def mmap(func, iterable):
    """Wrapper to make map() behave the same on Py2 and Py3."""

    if sys.version_info[0] > 2:
        return [i for i in map(func, iterable)]
    else:
        return map(func, iterable)

def get_versions(reporev=True):
    """Get version information for components used by Spyder"""
    import sys
    import platform

    import qtpy
    import qtpy.QtCore

    revision = None
    if reporev:
        from spyder.utils import vcs
        revision, branch = vcs.get_git_revision(os.path.dirname(__dir__))

    if not sys.platform == 'darwin':  # To avoid a crash with our Mac app
        system = platform.system()
    else:
        system = 'Darwin'

    return {
        'spyder': __version__,
        'python': platform.python_version(),  # "2.7.3"
        'bitness': 64 if sys.maxsize > 2**32 else 32,
        'qt': qtpy.QtCore.__version__,
        'qt_api': qtpy.API_NAME,      # PyQt5
        'qt_api_ver': qtpy.PYQT_VERSION,
        'system': system,   # Linux, Windows, ...
        'release': platform.release(),  # XP, 10.6, 2.2.0, etc.
        'revision': revision,  # '9fdf926eccce'
    }

def read32(bytestream):
  """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def is_relative_url(url):
    """ simple method to determine if a url is relative or absolute """
    if url.startswith("#"):
        return None
    if url.find("://") > 0 or url.startswith("//"):
        # either 'http(s)://...' or '//cdn...' and therefore absolute
        return False
    return True

def to_np(*args):
    """ convert GPU arras to numpy and return them"""
    if len(args) > 1:
        return (cp.asnumpy(x) for x in args)
    else:
        return cp.asnumpy(args[0])

def remove_blank_spaces(syllables: List[str]) -> List[str]:
    """
    Given a list of letters, remove any blank spaces or empty strings.

    :param syllables:
    :return:

    >>> remove_blank_spaces(['', 'a', ' ', 'b', ' ', 'c', ''])
    ['a', 'b', 'c']
    """
    cleaned = []
    for syl in syllables:
        if syl == " " or syl == '':
            pass
        else:
            cleaned.append(syl)
    return cleaned

def dict_to_ddb(item):
    # type: (Dict[str, Any]) -> Dict[str, Any]
    # TODO: narrow these types down
    """Converts a native Python dictionary to a raw DynamoDB item.

    :param dict item: Native item
    :returns: DynamoDB item
    :rtype: dict
    """
    serializer = TypeSerializer()
    return {key: serializer.serialize(value) for key, value in item.items()}

def _isint(string):
    """
    >>> _isint("123")
    True
    >>> _isint("123.45")
    False
    """
    return type(string) is int or \
           (isinstance(string, _binary_type) or isinstance(string, _text_type)) and \
           _isconvertible(int, string)

def non_increasing(values):
    """True if values are not increasing."""
    return all(x >= y for x, y in zip(values, values[1:]))

def input(prompt=""):
	"""input([prompt]) -> value

Equivalent to eval(raw_input(prompt))."""
	
	string = stdin_decode(raw_input(prompt))
	
	caller_frame = sys._getframe(1)
	globals = caller_frame.f_globals
	locals = caller_frame.f_locals
	
	return eval(string, globals, locals)

def suppress_stdout():
    """
    Context manager that suppresses stdout.

    Examples:
        >>> with suppress_stdout():
        ...     print('Test print')

        >>> print('test')
        test

    """
    save_stdout = sys.stdout
    sys.stdout = DevNull()
    yield
    sys.stdout = save_stdout

def convert_to_int(x: Any, default: int = None) -> int:
    """
    Transforms its input into an integer, or returns ``default``.
    """
    try:
        return int(x)
    except (TypeError, ValueError):
        return default

def filter_bool(n: Node, query: str) -> bool:
    """
    Filter and ensure that the returned value is of type bool.
    """
    return _scalariter2item(n, query, bool)

def ask_bool(question: str, default: bool = True) -> bool:
    """Asks a question yes no style"""
    default_q = "Y/n" if default else "y/N"
    answer = input("{0} [{1}]: ".format(question, default_q))
    lower = answer.lower()
    if not lower:
        return default
    return lower == "y"

def get_last_day_of_month(t: datetime) -> int:
    """
    Returns day number of the last day of the month
    :param t: datetime
    :return: int
    """
    tn = t + timedelta(days=32)
    tn = datetime(year=tn.year, month=tn.month, day=1)
    tt = tn - timedelta(hours=1)
    return tt.day

def indexes_equal(a: Index, b: Index) -> bool:
    """
    Are two indexes equal? Checks by comparing ``str()`` versions of them.
    (AM UNSURE IF THIS IS ENOUGH.)
    """
    return str(a) == str(b)

def to_bool(value: Any) -> bool:
    """Convert string or other Python object to boolean.

    **Rationalle**

    Passing flags is one of the most common cases of using environment vars and
    as values are strings we need to have an easy way to convert them to
    boolean Python value.

    Without this function int or float string values can be converted as false
    positives, e.g. ``bool('0') => True``, but using this function ensure that
    digit flag be properly converted to boolean value.

    :param value: String or other value.
    """
    return bool(strtobool(value) if isinstance(value, str) else value)

def remove_prefix(text, prefix):
	"""
	Remove the prefix from the text if it exists.

	>>> remove_prefix('underwhelming performance', 'underwhelming ')
	'performance'

	>>> remove_prefix('something special', 'sample')
	'something special'
	"""
	null, prefix, rest = text.rpartition(prefix)
	return rest

def split(text: str) -> List[str]:
    """Split a text into a list of tokens.

    :param text: the text to split
    :return: tokens
    """
    return [word for word in SEPARATOR.split(text) if word.strip(' \t')]

def post(self, endpoint: str, **kwargs) -> dict:
        """HTTP POST operation to API endpoint."""

        return self._request('POST', endpoint, **kwargs)

def cpu_count() -> int:
    """Returns the number of processors on this machine."""
    if multiprocessing is None:
        return 1
    try:
        return multiprocessing.cpu_count()
    except NotImplementedError:
        pass
    try:
        return os.sysconf("SC_NPROCESSORS_CONF")
    except (AttributeError, ValueError):
        pass
    gen_log.error("Could not detect number of processors; assuming 1")
    return 1

def _request(self, method: str, endpoint: str, params: dict = None, data: dict = None, headers: dict = None) -> dict:
        """HTTP request method of interface implementation."""

def rank(tensor: BKTensor) -> int:
    """Return the number of dimensions of a tensor"""
    if isinstance(tensor, np.ndarray):
        return len(tensor.shape)

    return len(tensor[0].size())

def from_buffer(buffer, mime=False):
    """
    Accepts a binary string and returns the detected filetype.  Return
    value is the mimetype if mime=True, otherwise a human readable
    name.

    >>> magic.from_buffer(open("testdata/test.pdf").read(1024))
    'PDF document, version 1.2'
    """
    m = _get_magic_type(mime)
    return m.from_buffer(buffer)

def url_concat(url, args):
    """Concatenate url and argument dictionary regardless of whether
    url has existing query parameters.

    >>> url_concat("http://example.com/foo?a=b", dict(c="d"))
    'http://example.com/foo?a=b&c=d'
    """
    if not args: return url
    if url[-1] not in ('?', '&'):
        url += '&' if ('?' in url) else '?'
    return url + urllib.urlencode(args)

def getVectorFromType(self, dtype) -> Union[bool, None, Tuple[int, int]]:
        """
        :see: doc of method on parent class
        """
        if dtype == BIT:
            return False
        elif isinstance(dtype, Bits):
            return [evalParam(dtype.width) - 1, hInt(0)]

def get_from_gnucash26_date(date_str: str) -> date:
    """ Creates a datetime from GnuCash 2.6 date string """
    date_format = "%Y%m%d"
    result = datetime.strptime(date_str, date_format).date()
    return result

def strtobytes(input, encoding):
    """Take a str and transform it into a byte array."""
    py_version = sys.version_info[0]
    if py_version >= 3:
        return _strtobytes_py3(input, encoding)
    return _strtobytes_py2(input, encoding)

def _(f, x):
    """
    filter for dict, note `f` should have signature: `f::key->value->bool`
    """
    return {k: v for k, v in x.items() if f(k, v)}

def get_day_name(self) -> str:
        """ Returns the day name """
        weekday = self.value.isoweekday() - 1
        return calendar.day_name[weekday]

def convert_camel_case_string(name: str) -> str:
    """Convert camel case string to snake case"""
    string = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", string).lower()

def _cnx_is_empty(in_file):
    """Check if cnr or cns files are empty (only have a header)
    """
    with open(in_file) as in_handle:
        for i, line in enumerate(in_handle):
            if i > 0:
                return False
    return True

def do_quit(self, _: argparse.Namespace) -> bool:
        """Exit this application"""
        self._should_quit = True
        return self._STOP_AND_EXIT

def distinct_permutations(iterable):
    """Yield successive distinct permutations of the elements in *iterable*.

        >>> sorted(distinct_permutations([1, 0, 1]))
        [(0, 1, 1), (1, 0, 1), (1, 1, 0)]

    Equivalent to ``set(permutations(iterable))``, except duplicates are not
    generated and thrown away. For larger input sequences this is much more
    efficient.

    Duplicate permutations arise when there are duplicated elements in the
    input iterable. The number of items returned is
    `n! / (x_1! * x_2! * ... * x_n!)`, where `n` is the total number of
    items input, and each `x_i` is the count of a distinct item in the input
    sequence.

    """
    def make_new_permutations(permutations, e):
        """Internal helper function.
        The output permutations are built up by adding element *e* to the
        current *permutations* at every possible position.
        The key idea is to keep repeated elements (reverse) ordered:
        if e1 == e2 and e1 is before e2 in the iterable, then all permutations
        with e1 before e2 are ignored.

        """
        for permutation in permutations:
            for j in range(len(permutation)):
                yield permutation[:j] + [e] + permutation[j:]
                if permutation[j] == e:
                    break
            else:
                yield permutation + [e]

    permutations = [[]]
    for e in iterable:
        permutations = make_new_permutations(permutations, e)

    return (tuple(t) for t in permutations)

def text_alignment(x, y):
    """
    Align text labels based on the x- and y-axis coordinate values.

    This function is used for computing the appropriate alignment of the text
    label.

    For example, if the text is on the "right" side of the plot, we want it to
    be left-aligned. If the text is on the "top" side of the plot, we want it
    to be bottom-aligned.

    :param x, y: (`int` or `float`) x- and y-axis coordinate respectively.
    :returns: A 2-tuple of strings, the horizontal and vertical alignments
        respectively.
    """
    if x == 0:
        ha = "center"
    elif x > 0:
        ha = "left"
    else:
        ha = "right"
    if y == 0:
        va = "center"
    elif y > 0:
        va = "bottom"
    else:
        va = "top"

    return ha, va

def _validate_image_rank(self, img_array):
        """
        Images must be either 2D or 3D.
        """
        if img_array.ndim == 1 or img_array.ndim > 3:
            msg = "{0}D imagery is not allowed.".format(img_array.ndim)
            raise IOError(msg)

def _request(self, method: str, endpoint: str, params: dict = None, data: dict = None, headers: dict = None) -> dict:
        """HTTP request method of interface implementation."""

def _cnx_is_empty(in_file):
    """Check if cnr or cns files are empty (only have a header)
    """
    with open(in_file) as in_handle:
        for i, line in enumerate(in_handle):
            if i > 0:
                return False
    return True

def _kbhit_unix() -> bool:
    """
    Under UNIX: is a keystroke available?
    """
    dr, dw, de = select.select([sys.stdin], [], [], 0)
    return dr != []

def lint(fmt='colorized'):
    """Run verbose PyLint on source. Optionally specify fmt=html for HTML output."""
    if fmt == 'html':
        outfile = 'pylint_report.html'
        local('pylint -f %s davies > %s || true' % (fmt, outfile))
        local('open %s' % outfile)
    else:
        local('pylint -f %s davies || true' % fmt)

def _hash_the_file(hasher, filename):
    """Helper function for creating hash functions.

    See implementation of :func:`dtoolcore.filehasher.shasum`
    for more usage details.
    """
    BUF_SIZE = 65536
    with open(filename, 'rb') as f:
        buf = f.read(BUF_SIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(BUF_SIZE)
    return hasher

def debugTreePrint(node,pfx="->"):
  """Purely a debugging aid: Ascii-art picture of a tree descended from node"""
  print pfx,node.item
  for c in node.children:
    debugTreePrint(c,"  "+pfx)

def hsv2rgb_spectrum(hsv):
    """Generates RGB values from HSV values in line with a typical light
    spectrum."""
    h, s, v = hsv
    return hsv2rgb_raw(((h * 192) >> 8, s, v))

def read_text_from_file(path: str) -> str:
    """ Reads text file contents """
    with open(path) as text_file:
        content = text_file.read()

    return content

def snake_to_camel(s: str) -> str:
    """Convert string from snake case to camel case."""

    fragments = s.split('_')

    return fragments[0] + ''.join(x.title() for x in fragments[1:])

def pack_bits( longbits ):
    """Crunch a 64-bit int (8 bool bytes) into a bitfield."""
    byte = longbits & (0x0101010101010101)
    byte = (byte | (byte>>7)) & (0x0003000300030003)
    byte = (byte | (byte>>14)) & (0x0000000f0000000f)
    byte = (byte | (byte>>28)) & (0x00000000000000ff)
    return byte

def camel_to_snake(s: str) -> str:
    """Convert string from camel case to snake case."""

    return CAMEL_CASE_RE.sub(r'_\1', s).strip().lower()

def get_window_dim():
    """ gets the dimensions depending on python version and os"""
    version = sys.version_info

    if version >= (3, 3):
        return _size_36()
    if platform.system() == 'Windows':
        return _size_windows()
    return _size_27()

def _prm_get_longest_stringsize(string_list):
        """ Returns the longest string size for a string entry across data."""
        maxlength = 1

        for stringar in string_list:
            if isinstance(stringar, np.ndarray):
                if stringar.ndim > 0:
                    for string in stringar.ravel():
                        maxlength = max(len(string), maxlength)
                else:
                    maxlength = max(len(stringar.tolist()), maxlength)
            else:
                maxlength = max(len(stringar), maxlength)

        # Make the string Col longer than needed in order to allow later on slightly larger strings
        return int(maxlength * 1.5)

def try_cast_int(s):
    """(str) -> int
    All the digits in a given string are concatenated and converted into a single number.
    """
    try:
        temp = re.findall('\d', str(s))
        temp = ''.join(temp)
        return int(temp)
    except:
        return s

def count(args):
    """ count occurences in a list of lists
    >>> count([['a','b'],['a']])
    defaultdict(int, {'a' : 2, 'b' : 1})
    """
    counts = defaultdict(int)
    for arg in args:
        for item in arg:
            counts[item] = counts[item] + 1
    return counts

def astensor(array: TensorLike) -> BKTensor:
    """Covert numpy array to tensorflow tensor"""
    tensor = tf.convert_to_tensor(value=array, dtype=CTYPE)
    return tensor

def has_value(cls, value: int) -> bool:
        """True if specified value exists in int enum; otherwise, False."""
        return any(value == item.value for item in cls)

def fast_median(a):
    """Fast median operation for masked array using 50th-percentile
    """
    a = checkma(a)
    #return scoreatpercentile(a.compressed(), 50)
    if a.count() > 0:
        out = np.percentile(a.compressed(), 50)
    else:
        out = np.ma.masked
    return out

def timeit (func, log, limit):
    """Print execution time of the function. For quick'n'dirty profiling."""

    def newfunc (*args, **kwargs):
        """Execute function and print execution time."""
        t = time.time()
        res = func(*args, **kwargs)
        duration = time.time() - t
        if duration > limit:
            print(func.__name__, "took %0.2f seconds" % duration, file=log)
            print(args, file=log)
            print(kwargs, file=log)
        return res
    return update_func_meta(newfunc, func)

def find_duplicates(l: list) -> set:
    """
    Return the duplicates in a list.

    The function relies on
    https://stackoverflow.com/questions/9835762/find-and-list-duplicates-in-a-list .
    Parameters
    ----------
    l : list
        Name

    Returns
    -------
    set
        Duplicated values

    >>> find_duplicates([1,2,3])
    set()
    >>> find_duplicates([1,2,1])
    {1}
    """
    return set([x for x in l if l.count(x) > 1])

def getIndex(predicateFn: Callable[[T], bool], items: List[T]) -> int:
    """
    Finds the index of an item in list, which satisfies predicate
    :param predicateFn: predicate function to run on items of list
    :param items: list of tuples
    :return: first index for which predicate function returns True
    """
    try:
        return next(i for i, v in enumerate(items) if predicateFn(v))
    except StopIteration:
        return -1

def execute(cur, *args):
    """Utility function to print sqlite queries before executing.

    Use instead of cur.execute().  First argument is cursor.

    cur.execute(stmt)
    becomes
    util.execute(cur, stmt)
    """
    stmt = args[0]
    if len(args) > 1:
        stmt = stmt.replace('%', '%%').replace('?', '%r')
        print(stmt % (args[1]))
    return cur.execute(*args)

def file_exists(fname):
    """Check if a file exists and is non-empty.
    """
    try:
        return fname and os.path.exists(fname) and os.path.getsize(fname) > 0
    except OSError:
        return False

def get_versions(reporev=True):
    """Get version information for components used by Spyder"""
    import sys
    import platform

    import qtpy
    import qtpy.QtCore

    revision = None
    if reporev:
        from spyder.utils import vcs
        revision, branch = vcs.get_git_revision(os.path.dirname(__dir__))

    if not sys.platform == 'darwin':  # To avoid a crash with our Mac app
        system = platform.system()
    else:
        system = 'Darwin'

    return {
        'spyder': __version__,
        'python': platform.python_version(),  # "2.7.3"
        'bitness': 64 if sys.maxsize > 2**32 else 32,
        'qt': qtpy.QtCore.__version__,
        'qt_api': qtpy.API_NAME,      # PyQt5
        'qt_api_ver': qtpy.PYQT_VERSION,
        'system': system,   # Linux, Windows, ...
        'release': platform.release(),  # XP, 10.6, 2.2.0, etc.
        'revision': revision,  # '9fdf926eccce'
    }

def chars(string: any) -> str:
        """Return all (and only) the chars in the given string."""
        return ''.join([c if c.isalpha() else '' for c in str(string)])

def _isbool(string):
    """
    >>> _isbool(True)
    True
    >>> _isbool("False")
    True
    >>> _isbool(1)
    False
    """
    return isinstance(string, _bool_type) or\
        (isinstance(string, (_binary_type, _text_type))
         and
         string in ("True", "False"))

def bfx(value, msb, lsb):
    """! @brief Extract a value from a bitfield."""
    mask = bitmask((msb, lsb))
    return (value & mask) >> lsb

def callable_validator(v: Any) -> AnyCallable:
    """
    Perform a simple check if the value is callable.

    Note: complete matching of argument type hints and return types is not performed
    """
    if callable(v):
        return v

    raise errors.CallableError(value=v)

def setup_cache(app: Flask, cache_config) -> Optional[Cache]:
    """Setup the flask-cache on a flask app"""
    if cache_config and cache_config.get('CACHE_TYPE') != 'null':
        return Cache(app, config=cache_config)

    return None

def file_exists(fname):
    """Check if a file exists and is non-empty.
    """
    try:
        return fname and os.path.exists(fname) and os.path.getsize(fname) > 0
    except OSError:
        return False

def __gt__(self, other):
    """Greater than ordering."""
    if not isinstance(other, Key):
      return NotImplemented
    return self.__tuple() > other.__tuple()

def strictly_positive_int_or_none(val):
    """Parse `val` into either `None` or a strictly positive integer."""
    val = positive_int_or_none(val)
    if val is None or val > 0:
        return val
    raise ValueError('"{}" must be strictly positive'.format(val))

def has_key(cls, *args):
        """
        Check whether flyweight object with specified key has already been created.

        Returns:
            bool: True if already created, False if not
        """
        key = args if len(args) > 1 else args[0]
        return key in cls._instances

def numpy_to_yaml(representer: Representer, data: np.ndarray) -> Sequence[Any]:
    """ Write a numpy array to YAML.

    It registers the array under the tag ``!numpy_array``.

    Use with:

    .. code-block:: python

        >>> yaml = ruamel.yaml.YAML()
        >>> yaml.representer.add_representer(np.ndarray, yaml.numpy_to_yaml)

    Note:
        We cannot use ``yaml.register_class`` because it won't register the proper type.
        (It would register the type of the class, rather than of `numpy.ndarray`). Instead,
        we use the above approach to register this method explicitly with the representer.
    """
    return representer.represent_sequence(
        "!numpy_array",
        data.tolist()
    )

def sections(self) -> list:
        """List of sections."""
        self.config.read(self.filepath)
        return self.config.sections()

def obj_in_list_always(target_list, obj):
    """
    >>> l = [1,1,1]
    >>> obj_in_list_always(l, 1)
    True
    >>> l.append(2)
    >>> obj_in_list_always(l, 1)
    False
    """
    for item in set(target_list):
        if item is not obj:
            return False
    return True

def integer_partition(size: int, nparts: int) -> Iterator[List[List[int]]]:
    """ Partition a list of integers into a list of partitions """
    for part in algorithm_u(range(size), nparts):
        yield part

def mmap(func, iterable):
    """Wrapper to make map() behave the same on Py2 and Py3."""

    if sys.version_info[0] > 2:
        return [i for i in map(func, iterable)]
    else:
        return map(func, iterable)

def assert_raises(ex_type, func, *args, **kwargs):
    r"""
    Checks that a function raises an error when given specific arguments.

    Args:
        ex_type (Exception): exception type
        func (callable): live python function

    CommandLine:
        python -m utool.util_assert assert_raises --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_assert import *  # NOQA
        >>> import utool as ut
        >>> ex_type = AssertionError
        >>> func = len
        >>> # Check that this raises an error when something else does not
        >>> assert_raises(ex_type, assert_raises, ex_type, func, [])
        >>> # Check this does not raise an error when something else does
        >>> assert_raises(ValueError, [].index, 0)
    """
    try:
        func(*args, **kwargs)
    except Exception as ex:
        assert isinstance(ex, ex_type), (
            'Raised %r but type should have been %r' % (ex, ex_type))
        return True
    else:
        raise AssertionError('No error was raised')

def uuid2buid(value):
    """
    Convert a UUID object to a 22-char BUID string

    >>> u = uuid.UUID('33203dd2-f2ef-422f-aeb0-058d6f5f7089')
    >>> uuid2buid(u)
    'MyA90vLvQi-usAWNb19wiQ'
    """
    if six.PY3:  # pragma: no cover
        return urlsafe_b64encode(value.bytes).decode('utf-8').rstrip('=')
    else:
        return six.text_type(urlsafe_b64encode(value.bytes).rstrip('='))

def indexes_equal(a: Index, b: Index) -> bool:
    """
    Are two indexes equal? Checks by comparing ``str()`` versions of them.
    (AM UNSURE IF THIS IS ENOUGH.)
    """
    return str(a) == str(b)

def get_window_dim():
    """ gets the dimensions depending on python version and os"""
    version = sys.version_info

    if version >= (3, 3):
        return _size_36()
    if platform.system() == 'Windows':
        return _size_windows()
    return _size_27()

def _run_sync(self, method: Callable, *args, **kwargs) -> Any:
        """
        Utility method to run commands synchronously for testing.
        """
        if self.loop.is_running():
            raise RuntimeError("Event loop is already running.")

        if not self.is_connected:
            self.loop.run_until_complete(self.connect())

        task = asyncio.Task(method(*args, **kwargs), loop=self.loop)
        result = self.loop.run_until_complete(task)

        self.loop.run_until_complete(self.quit())

        return result

def truncate_string(value, max_width=None):
    """Truncate string values."""
    if isinstance(value, text_type) and max_width is not None and len(value) > max_width:
        return value[:max_width]
    return value

def has_synset(word: str) -> list:
    """" Returns a list of synsets of a word after lemmatization. """

    return wn.synsets(lemmatize(word, neverstem=True))

def dag_longest_path(graph, source, target):
    """
    Finds the longest path in a dag between two nodes
    """
    if source == target:
        return [source]
    allpaths = nx.all_simple_paths(graph, source, target)
    longest_path = []
    for l in allpaths:
        if len(l) > len(longest_path):
            longest_path = l
    return longest_path

def _numbers_units(N):
    """
    >>> _numbers_units(45)
    '123456789012345678901234567890123456789012345'
    """
    lst = range(1, N + 1)
    return "".join(list(map(lambda i: str(i % 10), lst)))

def _duplicates(list_):
    """Return dict mapping item -> indices."""
    item_indices = {}
    for i, item in enumerate(list_):
        try:
            item_indices[item].append(i)
        except KeyError:  # First time seen
            item_indices[item] = [i]
    return item_indices

def put(self, endpoint: str, **kwargs) -> dict:
        """HTTP PUT operation to API endpoint."""

        return self._request('PUT', endpoint, **kwargs)

def inverted_dict(d):
    """Return a dict with swapped keys and values

    >>> inverted_dict({0: ('a', 'b'), 1: 'cd'}) == {'cd': 1, ('a', 'b'): 0}
    True
    """
    return dict((force_hashable(v), k) for (k, v) in viewitems(dict(d)))

def clean_column_names(df: DataFrame) -> DataFrame:
    """
    Strip the whitespace from all column names in the given DataFrame
    and return the result.
    """
    f = df.copy()
    f.columns = [col.strip() for col in f.columns]
    return f

def to_int64(a):
    """Return view of the recarray with all int32 cast to int64."""
    # build new dtype and replace i4 --> i8
    def promote_i4(typestr):
        if typestr[1:] == 'i4':
            typestr = typestr[0]+'i8'
        return typestr

    dtype = [(name, promote_i4(typestr)) for name,typestr in a.dtype.descr]
    return a.astype(dtype)

def iter_lines(file_like: Iterable[str]) -> Generator[str, None, None]:
    """ Helper for iterating only nonempty lines without line breaks"""
    for line in file_like:
        line = line.rstrip('\r\n')
        if line:
            yield line

def _duplicates(list_):
    """Return dict mapping item -> indices."""
    item_indices = {}
    for i, item in enumerate(list_):
        try:
            item_indices[item].append(i)
        except KeyError:  # First time seen
            item_indices[item] = [i]
    return item_indices

def insert_ordered(value, array):
    """
    This will insert the value into the array, keeping it sorted, and returning the
    index where it was inserted
    """

    index = 0

    # search for the last array item that value is larger than
    for n in range(0,len(array)):
        if value >= array[n]: index = n+1

    array.insert(index, value)
    return index

def _parse_tuple_string(argument):
        """ Return a tuple from parsing 'a,b,c,d' -> (a,b,c,d) """
        if isinstance(argument, str):
            return tuple(int(p.strip()) for p in argument.split(','))
        return argument

def inverted_dict(d):
    """Return a dict with swapped keys and values

    >>> inverted_dict({0: ('a', 'b'), 1: 'cd'}) == {'cd': 1, ('a', 'b'): 0}
    True
    """
    return dict((force_hashable(v), k) for (k, v) in viewitems(dict(d)))

def check_python_version():
    """Check if the currently running Python version is new enough."""
    # Required due to multiple with statements on one line
    req_version = (2, 7)
    cur_version = sys.version_info
    if cur_version >= req_version:
        print("Python version... %sOK%s (found %s, requires %s)" %
              (Bcolors.OKGREEN, Bcolors.ENDC, str(platform.python_version()),
               str(req_version[0]) + "." + str(req_version[1])))
    else:
        print("Python version... %sFAIL%s (found %s, requires %s)" %
              (Bcolors.FAIL, Bcolors.ENDC, str(cur_version),
               str(req_version)))

def normalize(numbers):
    """Multiply each number by a constant such that the sum is 1.0
    >>> normalize([1,2,1])
    [0.25, 0.5, 0.25]
    """
    total = float(sum(numbers))
    return [n / total for n in numbers]

def valid_file(path: str) -> bool:
    """
    Verifies that a string path actually exists and is a file

    :param path: The path to verify
    :return: **True** if path exist and is a file
    """
    path = Path(path).expanduser()
    log.debug("checking if %s is a valid file", path)
    return path.exists() and path.is_file()

def read_set_from_file(filename: str) -> Set[str]:
    """
    Extract a de-duped collection (set) of text from a file.
    Expected file format is one item per line.
    """
    collection = set()
    with open(filename, 'r') as file_:
        for line in file_:
            collection.add(line.rstrip())
    return collection

def _(f, x):
    """
    filter for dict, note `f` should have signature: `f::key->value->bool`
    """
    return {k: v for k, v in x.items() if f(k, v)}

def quaternion_imag(quaternion):
    """Return imaginary part of quaternion.

    >>> quaternion_imag([3, 0, 1, 2])
    array([ 0.,  1.,  2.])

    """
    return numpy.array(quaternion[1:4], dtype=numpy.float64, copy=True)

def strings_to_integers(strings: Iterable[str]) -> Iterable[int]:
    """
    Convert a list of strings to a list of integers.

    :param strings: a list of string
    :return: a list of converted integers

    .. doctest::

        >>> strings_to_integers(['1', '1.0', '-0.2'])
        [1, 1, 0]
    """
    return strings_to_(strings, lambda x: int(float(x)))

def is_sqlatype_string(coltype: Union[TypeEngine, VisitableType]) -> bool:
    """
    Is the SQLAlchemy column type a string type?
    """
    coltype = _coltype_to_typeengine(coltype)
    return isinstance(coltype, sqltypes.String)

def url_concat(url, args):
    """Concatenate url and argument dictionary regardless of whether
    url has existing query parameters.

    >>> url_concat("http://example.com/foo?a=b", dict(c="d"))
    'http://example.com/foo?a=b&c=d'
    """
    if not args: return url
    if url[-1] not in ('?', '&'):
        url += '&' if ('?' in url) else '?'
    return url + urllib.urlencode(args)

def _duplicates(list_):
    """Return dict mapping item -> indices."""
    item_indices = {}
    for i, item in enumerate(list_):
        try:
            item_indices[item].append(i)
        except KeyError:  # First time seen
            item_indices[item] = [i]
    return item_indices

def percentile(sorted_list, percent, key=lambda x: x):
    """Find the percentile of a sorted list of values.

    Arguments
    ---------
    sorted_list : list
        A sorted (ascending) list of values.
    percent : float
        A float value from 0.0 to 1.0.
    key : function, optional
        An optional function to compute a value from each element of N.

    Returns
    -------
    float
        The desired percentile of the value list.

    Examples
    --------
    >>> sorted_list = [4,6,8,9,11]
    >>> percentile(sorted_list, 0.4)
    7.0
    >>> percentile(sorted_list, 0.44)
    8.0
    >>> percentile(sorted_list, 0.6)
    8.5
    >>> percentile(sorted_list, 0.99)
    11.0
    >>> percentile(sorted_list, 1)
    11.0
    >>> percentile(sorted_list, 0)
    4.0
    """
    if not sorted_list:
        return None
    if percent == 1:
        return float(sorted_list[-1])
    if percent == 0:
        return float(sorted_list[0])
    n = len(sorted_list)
    i = percent * n
    if ceil(i) == i:
        i = int(i)
        return (sorted_list[i-1] + sorted_list[i]) / 2
    return float(sorted_list[ceil(i)-1])

def top(self, topn=10):
        """
        Get a list of the top ``topn`` features in this :class:`.Feature`\.

        Examples
        --------

        .. code-block:: python

        >>> myFeature = Feature([('the', 2), ('pine', 1), ('trapezoid', 5)])
        >>> myFeature.top(1)
        [('trapezoid', 5)]

        Parameters
        ----------
        topn : int

        Returns
        -------
        list
        """
        return [self[i] for i in argsort(list(zip(*self))[1])[::-1][:topn]]

def find_first_in_list(txt: str, str_list: [str]) -> int:  # type: ignore
    """
    Returns the index of the earliest occurence of an item from a list in a string

    Ex: find_first_in_list('foobar', ['bar', 'fin']) -> 3
    """
    start = len(txt) + 1
    for item in str_list:
        if start > txt.find(item) > -1:
            start = txt.find(item)
    return start if len(txt) + 1 > start > -1 else -1

def _duplicates(list_):
    """Return dict mapping item -> indices."""
    item_indices = {}
    for i, item in enumerate(list_):
        try:
            item_indices[item].append(i)
        except KeyError:  # First time seen
            item_indices[item] = [i]
    return item_indices

def is_unicode(string):
    """Validates that the object itself is some kinda string"""
    str_type = str(type(string))

    if str_type.find('str') > 0 or str_type.find('unicode') > 0:
        return True

    return False

def post(self, endpoint: str, **kwargs) -> dict:
        """HTTP POST operation to API endpoint."""

        return self._request('POST', endpoint, **kwargs)

def cmd_dot(conf: Config):
    """Print out a neat targets dependency tree based on requested targets.

    Use graphviz to render the dot file, e.g.:

    > ybt dot :foo :bar | dot -Tpng -o graph.png
    """
    build_context = BuildContext(conf)
    populate_targets_graph(build_context, conf)
    if conf.output_dot_file is None:
        write_dot(build_context, conf, sys.stdout)
    else:
        with open(conf.output_dot_file, 'w') as out_file:
            write_dot(build_context, conf, out_file)

def get_caller_module():
    """
    Returns the name of the caller's module as a string.

    >>> get_caller_module()
    '__main__'
    """
    stack = inspect.stack()
    assert len(stack) > 1
    caller = stack[2][0]
    return caller.f_globals['__name__']

def flatten_list(l: List[list]) -> list:
    """ takes a list of lists, l and returns a flat list
    """
    return [v for inner_l in l for v in inner_l]

def _rindex(mylist: Sequence[T], x: T) -> int:
    """Index of the last occurrence of x in the sequence."""
    return len(mylist) - mylist[::-1].index(x) - 1

def shape(self) -> Tuple[int, ...]:
        """Shape of histogram's data.

        Returns
        -------
        One-element tuple with the number of bins along each axis.
        """
        return tuple(bins.bin_count for bins in self._binnings)

def stretch(iterable, n=2):
    r"""Repeat each item in `iterable` `n` times.

    Example:

    >>> list(stretch(range(3), 2))
    [0, 0, 1, 1, 2, 2]
    """
    times = range(n)
    for item in iterable:
        for i in times: yield item

def negate_mask(mask):
    """Returns the negated mask.

    If elements of input mask have 0 and non-zero values, then the returned matrix will have all elements 0 (1) where
    the original one has non-zero (0).

    :param mask: Input mask
    :type mask: np.array
    :return: array of same shape and dtype=int8 as input array
    :rtype: np.array
    """
    res = np.ones(mask.shape, dtype=np.int8)
    res[mask > 0] = 0

    return res

def to_iso_string(self) -> str:
        """ Returns full ISO string for the given date """
        assert isinstance(self.value, datetime)
        return datetime.isoformat(self.value)

def _run_sync(self, method: Callable, *args, **kwargs) -> Any:
        """
        Utility method to run commands synchronously for testing.
        """
        if self.loop.is_running():
            raise RuntimeError("Event loop is already running.")

        if not self.is_connected:
            self.loop.run_until_complete(self.connect())

        task = asyncio.Task(method(*args, **kwargs), loop=self.loop)
        result = self.loop.run_until_complete(task)

        self.loop.run_until_complete(self.quit())

        return result

def _mid(pt1, pt2):
    """
    (Point, Point) -> Point
    Return the point that lies in between the two input points.
    """
    (x0, y0), (x1, y1) = pt1, pt2
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)

def top(self, topn=10):
        """
        Get a list of the top ``topn`` features in this :class:`.Feature`\.

        Examples
        --------

        .. code-block:: python

        >>> myFeature = Feature([('the', 2), ('pine', 1), ('trapezoid', 5)])
        >>> myFeature.top(1)
        [('trapezoid', 5)]

        Parameters
        ----------
        topn : int

        Returns
        -------
        list
        """
        return [self[i] for i in argsort(list(zip(*self))[1])[::-1][:topn]]

def memory_full():
    """Check if the memory is too full for further caching."""
    current_process = psutil.Process(os.getpid())
    return (current_process.memory_percent() >
            config.MAXIMUM_CACHE_MEMORY_PERCENTAGE)

def closest_values(L):
    """Closest values

    :param L: list of values
    :returns: two values from L with minimal distance
    :modifies: the order of L
    :complexity: O(n log n), for n=len(L)
    """
    assert len(L) >= 2
    L.sort()
    valmin, argmin = min((L[i] - L[i - 1], i) for i in range(1, len(L)))
    return L[argmin - 1], L[argmin]

def is_running(process_id: int) -> bool:
    """
    Uses the Unix ``ps`` program to see if a process is running.
    """
    pstr = str(process_id)
    encoding = sys.getdefaultencoding()
    s = subprocess.Popen(["ps", "-p", pstr], stdout=subprocess.PIPE)
    for line in s.stdout:
        strline = line.decode(encoding)
        if pstr in strline:
            return True
    return False

def uconcatenate(arrs, axis=0):
    """Concatenate a sequence of arrays.

    This wrapper around numpy.concatenate preserves units. All input arrays
    must have the same units.  See the documentation of numpy.concatenate for
    full details.

    Examples
    --------
    >>> from unyt import cm
    >>> A = [1, 2, 3]*cm
    >>> B = [2, 3, 4]*cm
    >>> uconcatenate((A, B))
    unyt_array([1, 2, 3, 2, 3, 4], 'cm')

    """
    v = np.concatenate(arrs, axis=axis)
    v = _validate_numpy_wrapper_units(v, arrs)
    return v

def Exit(msg, code=1):
    """Exit execution with return code and message
    :param msg: Message displayed prior to exit
    :param code: code returned upon exiting
    """
    print >> sys.stderr, msg
    sys.exit(code)

def find_duplicates(l: list) -> set:
    """
    Return the duplicates in a list.

    The function relies on
    https://stackoverflow.com/questions/9835762/find-and-list-duplicates-in-a-list .
    Parameters
    ----------
    l : list
        Name

    Returns
    -------
    set
        Duplicated values

    >>> find_duplicates([1,2,3])
    set()
    >>> find_duplicates([1,2,1])
    {1}
    """
    return set([x for x in l if l.count(x) > 1])

def read32(bytestream):
  """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def url_host(url: str) -> str:
    """
    Parses hostname from URL.
    :param url: URL
    :return: hostname
    """
    from urllib.parse import urlparse
    res = urlparse(url)
    return res.netloc.split(':')[0] if res.netloc else ''

def snake_to_camel(s: str) -> str:
    """Convert string from snake case to camel case."""

    fragments = s.split('_')

    return fragments[0] + ''.join(x.title() for x in fragments[1:])

def csv_to_numpy(string_like, dtype=None):  # type: (str) -> np.array
    """Convert a CSV object to a numpy array.

    Args:
        string_like (str): CSV string.
        dtype (dtype, optional):  Data type of the resulting array. If None, the dtypes will be determined by the
                                        contents of each column, individually. This argument can only be used to
                                        'upcast' the array.  For downcasting, use the .astype(t) method.
    Returns:
        (np.array): numpy array
    """
    stream = StringIO(string_like)
    return np.genfromtxt(stream, dtype=dtype, delimiter=',')

def shape(self) -> Tuple[int, ...]:
        """Shape of histogram's data.

        Returns
        -------
        One-element tuple with the number of bins along each axis.
        """
        return tuple(bins.bin_count for bins in self._binnings)

def _cnx_is_empty(in_file):
    """Check if cnr or cns files are empty (only have a header)
    """
    with open(in_file) as in_handle:
        for i, line in enumerate(in_handle):
            if i > 0:
                return False
    return True

def list_depth(list_, func=max, _depth=0):
    """
    Returns the deepest level of nesting within a list of lists

    Args:
       list_  : a nested listlike object
       func   : depth aggregation strategy (defaults to max)
       _depth : internal var

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list_ = [[[[[1]]], [3]], [[1], [3]], [[1], [3]]]
        >>> result = (list_depth(list_, _depth=0))
        >>> print(result)

    """
    depth_list = [list_depth(item, func=func, _depth=_depth + 1)
                  for item in  list_ if util_type.is_listlike(item)]
    if len(depth_list) > 0:
        return func(depth_list)
    else:
        return _depth

def _isint(string):
    """
    >>> _isint("123")
    True
    >>> _isint("123.45")
    False
    """
    return type(string) is int or \
           (isinstance(string, _binary_type) or isinstance(string, _text_type)) and \
           _isconvertible(int, string)

def is_any_type_set(sett: Set[Type]) -> bool:
    """
    Helper method to check if a set of types is the {AnyObject} singleton

    :param sett:
    :return:
    """
    return len(sett) == 1 and is_any_type(min(sett))

def csv_to_numpy(string_like, dtype=None):  # type: (str) -> np.array
    """Convert a CSV object to a numpy array.

    Args:
        string_like (str): CSV string.
        dtype (dtype, optional):  Data type of the resulting array. If None, the dtypes will be determined by the
                                        contents of each column, individually. This argument can only be used to
                                        'upcast' the array.  For downcasting, use the .astype(t) method.
    Returns:
        (np.array): numpy array
    """
    stream = StringIO(string_like)
    return np.genfromtxt(stream, dtype=dtype, delimiter=',')

def stop(self) -> None:
        """Stops the analysis as soon as possible."""
        if self._stop and not self._posted_kork:
            self._stop()
            self._stop = None

def proper_round(n):
    """
    rounds float to closest int
    :rtype: int
    :param n: float
    """
    return int(n) + (n / abs(n)) * int(abs(n - int(n)) >= 0.5) if n != 0 else 0

def _my_hash(arg_list):
    # type: (List[Any]) -> int
    """Simple helper hash function"""
    res = 0
    for arg in arg_list:
        res = res * 31 + hash(arg)
    return res

def _check_whitespace(string):
    """
    Make sure thre is no whitespace in the given string. Will raise a
    ValueError if whitespace is detected
    """
    if string.count(' ') + string.count('\t') + string.count('\n') > 0:
        raise ValueError(INSTRUCTION_HAS_WHITESPACE)

def proper_round(n):
    """
    rounds float to closest int
    :rtype: int
    :param n: float
    """
    return int(n) + (n / abs(n)) * int(abs(n - int(n)) >= 0.5) if n != 0 else 0

def dtypes(self):
        """Returns all column names and their data types as a list.

        >>> df.dtypes
        [('age', 'int'), ('name', 'string')]
        """
        return [(str(f.name), f.dataType.simpleString()) for f in self.schema.fields]

async def executemany(self, sql: str, parameters: Iterable[Iterable[Any]]) -> None:
        """Execute the given multiquery."""
        await self._execute(self._cursor.executemany, sql, parameters)

def pluralize(word):
    """
    Return the plural form of a word.

    Examples::

        >>> pluralize("post")
        "posts"
        >>> pluralize("octopus")
        "octopi"
        >>> pluralize("sheep")
        "sheep"
        >>> pluralize("CamelOctopus")
        "CamelOctopi"

    """
    if not word or word.lower() in UNCOUNTABLES:
        return word
    else:
        for rule, replacement in PLURALS:
            if re.search(rule, word):
                return re.sub(rule, replacement, word)
        return word

def method_caller(method_name, *args, **kwargs):
	"""
	Return a function that will call a named method on the
	target object with optional positional and keyword
	arguments.

	>>> lower = method_caller('lower')
	>>> lower('MyString')
	'mystring'
	"""
	def call_method(target):
		func = getattr(target, method_name)
		return func(*args, **kwargs)
	return call_method

async def parallel_results(future_map: Sequence[Tuple]) -> Dict:
    """
    Run parallel execution of futures and return mapping of their results to the provided keys.
    Just a neat shortcut around ``asyncio.gather()``

    :param future_map: Keys to futures mapping, e.g.: ( ('nav', get_nav()), ('content, get_content()) )
    :return: Dict with futures results mapped to keys {'nav': {1:2}, 'content': 'xyz'}
    """
    ctx_methods = OrderedDict(future_map)
    fs = list(ctx_methods.values())
    results = await asyncio.gather(*fs)
    results = {
        key: results[idx] for idx, key in enumerate(ctx_methods.keys())
    }
    return results

def positive_int(val):
    """Parse `val` into a positive integer."""
    if isinstance(val, float):
        raise ValueError('"{}" must not be a float'.format(val))
    val = int(val)
    if val >= 0:
        return val
    raise ValueError('"{}" must be positive'.format(val))

def chars(string: any) -> str:
        """Return all (and only) the chars in the given string."""
        return ''.join([c if c.isalpha() else '' for c in str(string)])

def branches():
    # type: () -> List[str]
    """ Return a list of branches in the current repo.

    Returns:
        list[str]: A list of branches in the current repo.
    """
    out = shell.run(
        'git branch',
        capture=True,
        never_pretend=True
    ).stdout.strip()
    return [x.strip('* \t\n') for x in out.splitlines()]

def execute(cur, *args):
    """Utility function to print sqlite queries before executing.

    Use instead of cur.execute().  First argument is cursor.

    cur.execute(stmt)
    becomes
    util.execute(cur, stmt)
    """
    stmt = args[0]
    if len(args) > 1:
        stmt = stmt.replace('%', '%%').replace('?', '%r')
        print(stmt % (args[1]))
    return cur.execute(*args)

def signed_area(coords):
    """Return the signed area enclosed by a ring using the linear time
    algorithm. A value >= 0 indicates a counter-clockwise oriented ring.
    """
    xs, ys = map(list, zip(*coords))
    xs.append(xs[1])
    ys.append(ys[1])
    return sum(xs[i]*(ys[i+1]-ys[i-1]) for i in range(1, len(coords)))/2.0

def classify_fit(fqdn, result, *argl, **argd):
    """Analyzes the result of a classification algorithm's fitting. See also
    :func:`fit` for explanation of arguments.
    """
    if len(argl) > 2:
        #Usually fit is called with fit(machine, Xtrain, ytrain).
        yP = argl[2]
    out = _generic_fit(fqdn, result, classify_predict, yP, *argl, **argd)
    return out

def thai_to_eng(text: str) -> str:
    """
    Correct text in one language that is incorrectly-typed with a keyboard layout in another language. (type Thai with English keyboard)

    :param str text: Incorrect input (type English with Thai keyboard)
    :return: English text
    """
    return "".join(
        [TH_EN_KEYB_PAIRS[ch] if (ch in TH_EN_KEYB_PAIRS) else ch for ch in text]
    )

def file_or_stdin() -> Callable:
    """
    Returns a file descriptor from stdin or opening a file from a given path.
    """

    def parse(path):
        if path is None or path == "-":
            return sys.stdin
        else:
            return data_io.smart_open(path)

    return parse

def _run_sync(self, method: Callable, *args, **kwargs) -> Any:
        """
        Utility method to run commands synchronously for testing.
        """
        if self.loop.is_running():
            raise RuntimeError("Event loop is already running.")

        if not self.is_connected:
            self.loop.run_until_complete(self.connect())

        task = asyncio.Task(method(*args, **kwargs), loop=self.loop)
        result = self.loop.run_until_complete(task)

        self.loop.run_until_complete(self.quit())

        return result

def file_exists(fname):
    """Check if a file exists and is non-empty.
    """
    try:
        return fname and os.path.exists(fname) and os.path.getsize(fname) > 0
    except OSError:
        return False

def is_relative_url(url):
    """ simple method to determine if a url is relative or absolute """
    if url.startswith("#"):
        return None
    if url.find("://") > 0 or url.startswith("//"):
        # either 'http(s)://...' or '//cdn...' and therefore absolute
        return False
    return True

def most_frequent(lst):
    """
    Returns the item that appears most frequently in the given list.
    """
    lst = lst[:]
    highest_freq = 0
    most_freq = None

    for val in unique(lst):
        if lst.count(val) > highest_freq:
            most_freq = val
            highest_freq = lst.count(val)
            
    return most_freq

def arcball_map_to_sphere(point, center, radius):
    """Return unit sphere coordinates from window coordinates."""
    v0 = (point[0] - center[0]) / radius
    v1 = (center[1] - point[1]) / radius
    n = v0*v0 + v1*v1
    if n > 1.0:
        # position outside of sphere
        n = math.sqrt(n)
        return numpy.array([v0/n, v1/n, 0.0])
    else:
        return numpy.array([v0, v1, math.sqrt(1.0 - n)])

def _my_hash(arg_list):
    # type: (List[Any]) -> int
    """Simple helper hash function"""
    res = 0
    for arg in arg_list:
        res = res * 31 + hash(arg)
    return res

def __gt__(self, other):
        """Test for greater than."""
        if isinstance(other, Address):
            return str(self) > str(other)
        raise TypeError

def flatten_list(x: List[Any]) -> List[Any]:
    """
    Converts a list of lists into a flat list.
    
    Args:
        x: list of lists 

    Returns:
        flat list
        
    As per
    http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python

    """  # noqa
    return [item for sublist in x for item in sublist]

def mmap(func, iterable):
    """Wrapper to make map() behave the same on Py2 and Py3."""

    if sys.version_info[0] > 2:
        return [i for i in map(func, iterable)]
    else:
        return map(func, iterable)

def iter_lines(file_like: Iterable[str]) -> Generator[str, None, None]:
    """ Helper for iterating only nonempty lines without line breaks"""
    for line in file_like:
        line = line.rstrip('\r\n')
        if line:
            yield line

def rank(tensor: BKTensor) -> int:
    """Return the number of dimensions of a tensor"""
    if isinstance(tensor, np.ndarray):
        return len(tensor.shape)

    return len(tensor[0].size())

def uppercase_chars(string: any) -> str:
        """Return all (and only) the uppercase chars in the given string."""
        return ''.join([c if c.isupper() else '' for c in str(string)])

def to_bytes(data: Any) -> bytearray:
    """
    Convert anything to a ``bytearray``.
    
    See
    
    - http://stackoverflow.com/questions/7585435/best-way-to-convert-string-to-bytes-in-python-3
    - http://stackoverflow.com/questions/10459067/how-to-convert-my-bytearrayb-x9e-x18k-x9a-to-something-like-this-x9e-x1
    """  # noqa
    if isinstance(data, int):
        return bytearray([data])
    return bytearray(data, encoding='latin-1')

def truncate(value: Decimal, n_digits: int) -> Decimal:
    """Truncates a value to a number of decimals places"""
    return Decimal(math.trunc(value * (10 ** n_digits))) / (10 ** n_digits)

def dictlist_wipe_key(dict_list: Iterable[Dict], key: str) -> None:
    """
    Process an iterable of dictionaries. For each dictionary ``d``, delete
    ``d[key]`` if it exists.
    """
    for d in dict_list:
        d.pop(key, None)

def impose_legend_limit(limit=30, axes="gca", **kwargs):
    """
    This will erase all but, say, 30 of the legend entries and remake the legend.
    You'll probably have to move it back into your favorite position at this point.
    """
    if axes=="gca": axes = _pylab.gca()

    # make these axes current
    _pylab.axes(axes)

    # loop over all the lines_pylab.
    for n in range(0,len(axes.lines)):
        if n >  limit-1 and not n==len(axes.lines)-1: axes.lines[n].set_label("_nolegend_")
        if n == limit-1 and not n==len(axes.lines)-1: axes.lines[n].set_label("...")

    _pylab.legend(**kwargs)

def zfill(x, width):
    """zfill(x, width) -> string

    Pad a numeric string x with zeros on the left, to fill a field
    of the specified width.  The string x is never truncated.

    """
    if not isinstance(x, basestring):
        x = repr(x)
    return x.zfill(width)

def cli_run():
    """docstring for argparse"""
    parser = argparse.ArgumentParser(description='Stupidly simple code answers from StackOverflow')
    parser.add_argument('query', help="What's the problem ?", type=str, nargs='+')
    parser.add_argument('-t','--tags', help='semicolon separated tags -> python;lambda')
    args = parser.parse_args()
    main(args)

def header_status(header):
    """Parse HTTP status line, return status (int) and reason."""
    status_line = header[:header.find('\r')]
    # 'HTTP/1.1 200 OK' -> (200, 'OK')
    fields = status_line.split(None, 2)
    return int(fields[1]), fields[2]

def mmap(func, iterable):
    """Wrapper to make map() behave the same on Py2 and Py3."""

    if sys.version_info[0] > 2:
        return [i for i in map(func, iterable)]
    else:
        return map(func, iterable)

def lower_camel_case_from_underscores(string):
    """generate a lower-cased camelCase string from an underscore_string.
    For example: my_variable_name -> myVariableName"""
    components = string.split('_')
    string = components[0]
    for component in components[1:]:
        string += component[0].upper() + component[1:]
    return string

def get_keys_of_max_n(dict_obj, n):
    """Returns the keys that maps to the top n max values in the given dict.

    Example:
    --------
    >>> dict_obj = {'a':2, 'b':1, 'c':5}
    >>> get_keys_of_max_n(dict_obj, 2)
    ['a', 'c']
    """
    return sorted([
        item[0]
        for item in sorted(
            dict_obj.items(), key=lambda item: item[1], reverse=True
        )[:n]
    ])

def strtobytes(input, encoding):
    """Take a str and transform it into a byte array."""
    py_version = sys.version_info[0]
    if py_version >= 3:
        return _strtobytes_py3(input, encoding)
    return _strtobytes_py2(input, encoding)

def months_ago(date, nb_months=1):
    """
    Return the given `date` with `nb_months` substracted from it.
    """
    nb_years = nb_months // 12
    nb_months = nb_months % 12

    month_diff = date.month - nb_months

    if month_diff > 0:
        new_month = month_diff
    else:
        new_month = 12 + month_diff
        nb_years += 1

    return date.replace(day=1, month=new_month, year=date.year - nb_years)

def cmd_dot(conf: Config):
    """Print out a neat targets dependency tree based on requested targets.

    Use graphviz to render the dot file, e.g.:

    > ybt dot :foo :bar | dot -Tpng -o graph.png
    """
    build_context = BuildContext(conf)
    populate_targets_graph(build_context, conf)
    if conf.output_dot_file is None:
        write_dot(build_context, conf, sys.stdout)
    else:
        with open(conf.output_dot_file, 'w') as out_file:
            write_dot(build_context, conf, out_file)

def normalize(numbers):
    """Multiply each number by a constant such that the sum is 1.0
    >>> normalize([1,2,1])
    [0.25, 0.5, 0.25]
    """
    total = float(sum(numbers))
    return [n / total for n in numbers]

def iter_lines(file_like: Iterable[str]) -> Generator[str, None, None]:
    """ Helper for iterating only nonempty lines without line breaks"""
    for line in file_like:
        line = line.rstrip('\r\n')
        if line:
            yield line

def guess_mimetype(filename):
    """Guesses the mimetype of a file based on the given ``filename``.

    .. code-block:: python

        >>> guess_mimetype('example.txt')
        'text/plain'
        >>> guess_mimetype('/foo/bar/example')
        'application/octet-stream'

    Parameters
    ----------
    filename : str
        The file name or path for which the mimetype is to be guessed
    """
    fn = os.path.basename(filename)
    return mimetypes.guess_type(fn)[0] or 'application/octet-stream'

def is_integer(value: Any) -> bool:
    """Return true if a value is an integer number."""
    return (isinstance(value, int) and not isinstance(value, bool)) or (
        isinstance(value, float) and isfinite(value) and int(value) == value
    )

def top(self, topn=10):
        """
        Get a list of the top ``topn`` features in this :class:`.Feature`\.

        Examples
        --------

        .. code-block:: python

        >>> myFeature = Feature([('the', 2), ('pine', 1), ('trapezoid', 5)])
        >>> myFeature.top(1)
        [('trapezoid', 5)]

        Parameters
        ----------
        topn : int

        Returns
        -------
        list
        """
        return [self[i] for i in argsort(list(zip(*self))[1])[::-1][:topn]]

def has_key(cls, *args):
        """
        Check whether flyweight object with specified key has already been created.

        Returns:
            bool: True if already created, False if not
        """
        key = args if len(args) > 1 else args[0]
        return key in cls._instances

def iprotate(l, steps=1):
    r"""Like rotate, but modifies `l` in-place.

    >>> l = [1,2,3]
    >>> iprotate(l) is l
    True
    >>> l
    [2, 3, 1]
    >>> iprotate(iprotate(l, 2), -3)
    [1, 2, 3]

    """
    if len(l):
        steps %= len(l)
        if steps:
            firstPart = l[:steps]
            del l[:steps]
            l.extend(firstPart)
    return l

def dictlist_wipe_key(dict_list: Iterable[Dict], key: str) -> None:
    """
    Process an iterable of dictionaries. For each dictionary ``d``, delete
    ``d[key]`` if it exists.
    """
    for d in dict_list:
        d.pop(key, None)

def exists(self):
        """
        Determine if any rows exist for the current query.

        :return: Whether the rows exist or not
        :rtype: bool
        """
        limit = self.limit_

        result = self.limit(1).count() > 0

        self.limit(limit)

        return result

def dtypes(self):
        """Returns all column names and their data types as a list.

        >>> df.dtypes
        [('age', 'int'), ('name', 'string')]
        """
        return [(str(f.name), f.dataType.simpleString()) for f in self.schema.fields]

def chars(string: any) -> str:
        """Return all (and only) the chars in the given string."""
        return ''.join([c if c.isalpha() else '' for c in str(string)])

def is_line_in_file(filename: str, line: str) -> bool:
    """
    Detects whether a line is present within a file.

    Args:
        filename: file to check
        line: line to search for (as an exact match)
    """
    assert "\n" not in line
    with open(filename, "r") as file:
        for fileline in file:
            if fileline == line:
                return True
        return False

def iprotate(l, steps=1):
    r"""Like rotate, but modifies `l` in-place.

    >>> l = [1,2,3]
    >>> iprotate(l) is l
    True
    >>> l
    [2, 3, 1]
    >>> iprotate(iprotate(l, 2), -3)
    [1, 2, 3]

    """
    if len(l):
        steps %= len(l)
        if steps:
            firstPart = l[:steps]
            del l[:steps]
            l.extend(firstPart)
    return l

def normalize_pattern(pattern):
    """Converts backslashes in path patterns to forward slashes.

    Doesn't normalize regular expressions - they may contain escapes.
    """
    if not (pattern.startswith('RE:') or pattern.startswith('!RE:')):
        pattern = _slashes.sub('/', pattern)
    if len(pattern) > 1:
        pattern = pattern.rstrip('/')
    return pattern

def uniqued(iterable):
    """Return unique list of ``iterable`` items preserving order.

    >>> uniqued('spameggs')
    ['s', 'p', 'a', 'm', 'e', 'g']
    """
    seen = set()
    return [item for item in iterable if item not in seen and not seen.add(item)]

def from_file(file_path) -> dict:
        """ Load JSON file """
        with io.open(file_path, 'r', encoding='utf-8') as json_stream:
            return Json.parse(json_stream, True)

def clean_column_names(df: DataFrame) -> DataFrame:
    """
    Strip the whitespace from all column names in the given DataFrame
    and return the result.
    """
    f = df.copy()
    f.columns = [col.strip() for col in f.columns]
    return f

def _kbhit_unix() -> bool:
    """
    Under UNIX: is a keystroke available?
    """
    dr, dw, de = select.select([sys.stdin], [], [], 0)
    return dr != []

def proper_round(n):
    """
    rounds float to closest int
    :rtype: int
    :param n: float
    """
    return int(n) + (n / abs(n)) * int(abs(n - int(n)) >= 0.5) if n != 0 else 0

def string(value) -> str:
        """ string dict/object/value to JSON """
        return system_json.dumps(Json(value).safe_object(), ensure_ascii=False)

def str_to_time(time_str: str) -> datetime.datetime:
    """
    Convert human readable string to datetime.datetime.
    """
    pieces: Any = [int(piece) for piece in time_str.split('-')]
    return datetime.datetime(*pieces)

def _collection_literal_to_py_ast(
    ctx: GeneratorContext, form: Iterable[LispForm]
) -> Iterable[GeneratedPyAST]:
    """Turn a quoted collection literal of Lisp forms into Python AST nodes.

    This function can only handle constant values. It does not call back into
    the generic AST generators, so only constant values will be generated down
    this path."""
    yield from map(partial(_const_val_to_py_ast, ctx), form)

def flatten_list(l: List[list]) -> list:
    """ takes a list of lists, l and returns a flat list
    """
    return [v for inner_l in l for v in inner_l]

def dictlist_wipe_key(dict_list: Iterable[Dict], key: str) -> None:
    """
    Process an iterable of dictionaries. For each dictionary ``d``, delete
    ``d[key]`` if it exists.
    """
    for d in dict_list:
        d.pop(key, None)

def valid_file(path: str) -> bool:
    """
    Verifies that a string path actually exists and is a file

    :param path: The path to verify
    :return: **True** if path exist and is a file
    """
    path = Path(path).expanduser()
    log.debug("checking if %s is a valid file", path)
    return path.exists() and path.is_file()

def segment_intersection(start0, end0, start1, end1):
    r"""Determine the intersection of two line segments.

    Assumes each line is parametric

    .. math::

       \begin{alignat*}{2}
        L_0(s) &= S_0 (1 - s) + E_0 s &&= S_0 + s \Delta_0 \\
        L_1(t) &= S_1 (1 - t) + E_1 t &&= S_1 + t \Delta_1.
       \end{alignat*}

    To solve :math:`S_0 + s \Delta_0 = S_1 + t \Delta_1`, we use the
    cross product:

    .. math::

       \left(S_0 + s \Delta_0\right) \times \Delta_1 =
           \left(S_1 + t \Delta_1\right) \times \Delta_1 \Longrightarrow
       s \left(\Delta_0 \times \Delta_1\right) =
           \left(S_1 - S_0\right) \times \Delta_1.

    Similarly

    .. math::

       \Delta_0 \times \left(S_0 + s \Delta_0\right) =
           \Delta_0 \times \left(S_1 + t \Delta_1\right) \Longrightarrow
       \left(S_1 - S_0\right) \times \Delta_0 =
           \Delta_0 \times \left(S_0 - S_1\right) =
           t \left(\Delta_0 \times \Delta_1\right).

    .. note::

       Since our points are in :math:`\mathbf{R}^2`, the "traditional"
       cross product in :math:`\mathbf{R}^3` will always point in the
       :math:`z` direction, so in the above we mean the :math:`z`
       component of the cross product, rather than the entire vector.

    For example, the diagonal lines

    .. math::

       \begin{align*}
        L_0(s) &= \left[\begin{array}{c} 0 \\ 0 \end{array}\right] (1 - s) +
                  \left[\begin{array}{c} 2 \\ 2 \end{array}\right] s \\
        L_1(t) &= \left[\begin{array}{c} -1 \\ 2 \end{array}\right] (1 - t) +
                  \left[\begin{array}{c} 1 \\ 0 \end{array}\right] t
       \end{align*}

    intersect at :math:`L_0\left(\frac{1}{4}\right) =
    L_1\left(\frac{3}{4}\right) =
    \frac{1}{2} \left[\begin{array}{c} 1 \\ 1 \end{array}\right]`.

    .. image:: ../images/segment_intersection1.png
       :align: center

    .. testsetup:: segment-intersection1, segment-intersection2

       import numpy as np
       from bezier._geometric_intersection import segment_intersection

    .. doctest:: segment-intersection1
       :options: +NORMALIZE_WHITESPACE

       >>> start0 = np.asfortranarray([0.0, 0.0])
       >>> end0 = np.asfortranarray([2.0, 2.0])
       >>> start1 = np.asfortranarray([-1.0, 2.0])
       >>> end1 = np.asfortranarray([1.0, 0.0])
       >>> s, t, _ = segment_intersection(start0, end0, start1, end1)
       >>> s
       0.25
       >>> t
       0.75

    .. testcleanup:: segment-intersection1

       import make_images
       make_images.segment_intersection1(start0, end0, start1, end1, s)

    Taking the parallel (but different) lines

    .. math::

       \begin{align*}
        L_0(s) &= \left[\begin{array}{c} 1 \\ 0 \end{array}\right] (1 - s) +
                  \left[\begin{array}{c} 0 \\ 1 \end{array}\right] s \\
        L_1(t) &= \left[\begin{array}{c} -1 \\ 3 \end{array}\right] (1 - t) +
                  \left[\begin{array}{c} 3 \\ -1 \end{array}\right] t
       \end{align*}

    we should be able to determine that the lines don't intersect, but
    this function is not meant for that check:

    .. image:: ../images/segment_intersection2.png
       :align: center

    .. doctest:: segment-intersection2
       :options: +NORMALIZE_WHITESPACE

       >>> start0 = np.asfortranarray([1.0, 0.0])
       >>> end0 = np.asfortranarray([0.0, 1.0])
       >>> start1 = np.asfortranarray([-1.0, 3.0])
       >>> end1 = np.asfortranarray([3.0, -1.0])
       >>> _, _, success = segment_intersection(start0, end0, start1, end1)
       >>> success
       False

    .. testcleanup:: segment-intersection2

       import make_images
       make_images.segment_intersection2(start0, end0, start1, end1)

    Instead, we use :func:`parallel_lines_parameters`:

    .. testsetup:: segment-intersection2-continued

       import numpy as np
       from bezier._geometric_intersection import parallel_lines_parameters

       start0 = np.asfortranarray([1.0, 0.0])
       end0 = np.asfortranarray([0.0, 1.0])
       start1 = np.asfortranarray([-1.0, 3.0])
       end1 = np.asfortranarray([3.0, -1.0])

    .. doctest:: segment-intersection2-continued

       >>> disjoint, _ = parallel_lines_parameters(start0, end0, start1, end1)
       >>> disjoint
       True

    .. note::

       There is also a Fortran implementation of this function, which
       will be used if it can be built.

    Args:
        start0 (numpy.ndarray): A 1D NumPy ``2``-array that is the start
            vector :math:`S_0` of the parametric line :math:`L_0(s)`.
        end0 (numpy.ndarray): A 1D NumPy ``2``-array that is the end
            vector :math:`E_0` of the parametric line :math:`L_0(s)`.
        start1 (numpy.ndarray): A 1D NumPy ``2``-array that is the start
            vector :math:`S_1` of the parametric line :math:`L_1(s)`.
        end1 (numpy.ndarray): A 1D NumPy ``2``-array that is the end
            vector :math:`E_1` of the parametric line :math:`L_1(s)`.

    Returns:
        Tuple[float, float, bool]: Pair of :math:`s_{\ast}` and
        :math:`t_{\ast}` such that the lines intersect:
        :math:`L_0\left(s_{\ast}\right) = L_1\left(t_{\ast}\right)` and then
        a boolean indicating if an intersection was found (i.e. if the lines
        aren't parallel).
    """
    delta0 = end0 - start0
    delta1 = end1 - start1
    cross_d0_d1 = _helpers.cross_product(delta0, delta1)
    if cross_d0_d1 == 0.0:
        return None, None, False

    else:
        start_delta = start1 - start0
        s = _helpers.cross_product(start_delta, delta1) / cross_d0_d1
        t = _helpers.cross_product(start_delta, delta0) / cross_d0_d1
        return s, t, True

def isarray(array, test, dim=2):
    """Returns True if test is True for all array elements.
    Otherwise, returns False.
    """
    if dim > 1:
        return all(isarray(array[i], test, dim - 1)
                   for i in range(len(array)))
    return all(test(i) for i in array)

def get_from_gnucash26_date(date_str: str) -> date:
    """ Creates a datetime from GnuCash 2.6 date string """
    date_format = "%Y%m%d"
    result = datetime.strptime(date_str, date_format).date()
    return result

def kernel(self, spread=1):
        """ This will return whatever kind of kernel we want to use.
            Must have signature (ndarray size NxM, ndarray size 1xM) -> ndarray size Nx1
        """
        # TODO: use self.kernel_type to choose function

        def gaussian(data, pixel):
            return mvn.pdf(data, mean=pixel, cov=spread)

        return gaussian

def is_sqlatype_string(coltype: Union[TypeEngine, VisitableType]) -> bool:
    """
    Is the SQLAlchemy column type a string type?
    """
    coltype = _coltype_to_typeengine(coltype)
    return isinstance(coltype, sqltypes.String)

def decode_value(stream):
    """Decode the contents of a value from a serialized stream.

    :param stream: Source data stream
    :type stream: io.BytesIO
    :returns: Decoded value
    :rtype: bytes
    """
    length = decode_length(stream)
    (value,) = unpack_value(">{:d}s".format(length), stream)
    return value

def is_intersection(g, n):
    """
    Determine if a node is an intersection

    graph: 1 -->-- 2 -->-- 3

    >>> is_intersection(g, 2)
    False

    graph:
     1 -- 2 -- 3
          |
          4

    >>> is_intersection(g, 2)
    True

    Parameters
    ----------
    g : networkx DiGraph
    n : node id

    Returns
    -------
    bool

    """
    return len(set(g.predecessors(n) + g.successors(n))) > 2

def warn_if_nans_exist(X):
    """Warn if nans exist in a numpy array."""
    null_count = count_rows_with_nans(X)
    total = len(X)
    percent = 100 * null_count / total

    if null_count > 0:
        warning_message = \
            'Warning! Found {} rows of {} ({:0.2f}%) with nan values. Only ' \
            'complete rows will be plotted.'.format(null_count, total, percent)
        warnings.warn(warning_message, DataWarning)

def dotproduct(X, Y):
    """Return the sum of the element-wise product of vectors x and y.
    >>> dotproduct([1, 2, 3], [1000, 100, 10])
    1230
    """
    return sum([x * y for x, y in zip(X, Y)])

def has_synset(word: str) -> list:
    """" Returns a list of synsets of a word after lemmatization. """

    return wn.synsets(lemmatize(word, neverstem=True))

def lint(fmt='colorized'):
    """Run verbose PyLint on source. Optionally specify fmt=html for HTML output."""
    if fmt == 'html':
        outfile = 'pylint_report.html'
        local('pylint -f %s davies > %s || true' % (fmt, outfile))
        local('open %s' % outfile)
    else:
        local('pylint -f %s davies || true' % fmt)

def getPiLambert(n):
    """Returns a list containing first n digits of Pi
    """
    mypi = piGenLambert()
    result = []
    if n > 0:
        result += [next(mypi) for i in range(n)]
    mypi.close()
    return result

def after_third_friday(day=None):
    """ check if day is after month's 3rd friday """
    day = day if day is not None else datetime.datetime.now()
    now = day.replace(day=1, hour=16, minute=0, second=0, microsecond=0)
    now += relativedelta.relativedelta(weeks=2, weekday=relativedelta.FR)
    return day > now

def _get_latest_version():
    """Gets latest Dusty binary version using the GitHub api"""
    url = 'https://api.github.com/repos/{}/releases/latest'.format(constants.DUSTY_GITHUB_PATH)
    conn = urllib.urlopen(url)
    if conn.getcode() >= 300:
        raise RuntimeError('GitHub api returned code {}; can\'t determine latest version.  Aborting'.format(conn.getcode()))
    json_data = conn.read()
    return json.loads(json_data)['tag_name']

def dag_longest_path(graph, source, target):
    """
    Finds the longest path in a dag between two nodes
    """
    if source == target:
        return [source]
    allpaths = nx.all_simple_paths(graph, source, target)
    longest_path = []
    for l in allpaths:
        if len(l) > len(longest_path):
            longest_path = l
    return longest_path

def inverted_dict_of_lists(d):
    """Return a dict where the keys are all the values listed in the values of the original dict

    >>> inverted_dict_of_lists({0: ['a', 'b'], 1: 'cd'}) == {'a': 0, 'b': 0, 'cd': 1}
    True
    """
    new_dict = {}
    for (old_key, old_value_list) in viewitems(dict(d)):
        for new_key in listify(old_value_list):
            new_dict[new_key] = old_key
    return new_dict

def __remove_trailing_zeros(self, collection):
        """Removes trailing zeroes from indexable collection of numbers"""
        index = len(collection) - 1
        while index >= 0 and collection[index] == 0:
            index -= 1

        return collection[:index + 1]

def check_key(self, key: str) -> bool:
        """
        Checks if key exists in datastore. True if yes, False if no.

        :param: SHA512 hash key

        :return: whether or key not exists in datastore
        """
        keys = self.get_keys()
        return key in keys

def dictlist_wipe_key(dict_list: Iterable[Dict], key: str) -> None:
    """
    Process an iterable of dictionaries. For each dictionary ``d``, delete
    ``d[key]`` if it exists.
    """
    for d in dict_list:
        d.pop(key, None)

def _kbhit_unix() -> bool:
    """
    Under UNIX: is a keystroke available?
    """
    dr, dw, de = select.select([sys.stdin], [], [], 0)
    return dr != []

def simple_eq(one: Instance, two: Instance, attrs: List[str]) -> bool:
    """
    Test if two objects are equal, based on a comparison of the specified
    attributes ``attrs``.
    """
    return all(getattr(one, a) == getattr(two, a) for a in attrs)

def decode_value(stream):
    """Decode the contents of a value from a serialized stream.

    :param stream: Source data stream
    :type stream: io.BytesIO
    :returns: Decoded value
    :rtype: bytes
    """
    length = decode_length(stream)
    (value,) = unpack_value(">{:d}s".format(length), stream)
    return value

def count(args):
    """ count occurences in a list of lists
    >>> count([['a','b'],['a']])
    defaultdict(int, {'a' : 2, 'b' : 1})
    """
    counts = defaultdict(int)
    for arg in args:
        for item in arg:
            counts[item] = counts[item] + 1
    return counts

def _(f, x):
    """
    filter for dict, note `f` should have signature: `f::key->value->bool`
    """
    return {k: v for k, v in x.items() if f(k, v)}

def iter_lines(file_like: Iterable[str]) -> Generator[str, None, None]:
    """ Helper for iterating only nonempty lines without line breaks"""
    for line in file_like:
        line = line.rstrip('\r\n')
        if line:
            yield line

def file_exists(fname):
    """Check if a file exists and is non-empty.
    """
    try:
        return fname and os.path.exists(fname) and os.path.getsize(fname) > 0
    except OSError:
        return False

def argmax(self, rows: List[Row], column: ComparableColumn) -> List[Row]:
        """
        Takes a list of rows and a column name and returns a list containing a single row (dict from
        columns to cells) that has the maximum numerical value in the given column. We return a list
        instead of a single dict to be consistent with the return type of ``select`` and
        ``all_rows``.
        """
        if not rows:
            return []
        value_row_pairs = [(row.values[column.name], row) for row in rows]
        if not value_row_pairs:
            return []
        # Returns a list containing the row with the max cell value.
        return [sorted(value_row_pairs, key=lambda x: x[0], reverse=True)[0][1]]

def genfirstvalues(cursor: Cursor, arraysize: int = 1000) \
        -> Generator[Any, None, None]:
    """
    Generate the first value in each row.

    Args:
        cursor: the cursor
        arraysize: split fetches into chunks of this many records

    Yields:
        the first value of each row
    """
    return (row[0] for row in genrows(cursor, arraysize))

def to_iso_string(self) -> str:
        """ Returns full ISO string for the given date """
        assert isinstance(self.value, datetime)
        return datetime.isoformat(self.value)

def proper_round(n):
    """
    rounds float to closest int
    :rtype: int
    :param n: float
    """
    return int(n) + (n / abs(n)) * int(abs(n - int(n)) >= 0.5) if n != 0 else 0

def is_finite(value: Any) -> bool:
    """Return true if a value is a finite number."""
    return isinstance(value, int) or (isinstance(value, float) and isfinite(value))

def _isint(string):
    """
    >>> _isint("123")
    True
    >>> _isint("123.45")
    False
    """
    return type(string) is int or \
           (isinstance(string, _binary_type) or isinstance(string, _text_type)) and \
           _isconvertible(int, string)

def clean_map(obj: Mapping[Any, Any]) -> Mapping[Any, Any]:
    """
    Return a new copied dictionary without the keys with ``None`` values from
    the given Mapping object.
    """
    return {k: v for k, v in obj.items() if v is not None}

def shift(self, m: Union[float, pd.Series]) -> Union[int, pd.Series]:
        """Shifts floats so that the first 10 decimal digits are significant."""
        out = m % 1 * self.TEN_DIGIT_MODULUS // 1
        if isinstance(out, pd.Series):
            return out.astype(int)
        return int(out)

def is_empty_shape(sh: ShExJ.Shape) -> bool:
        """ Determine whether sh has any value """
        return sh.closed is None and sh.expression is None and sh.extra is None and \
            sh.semActs is None

def valid_file(path: str) -> bool:
    """
    Verifies that a string path actually exists and is a file

    :param path: The path to verify
    :return: **True** if path exist and is a file
    """
    path = Path(path).expanduser()
    log.debug("checking if %s is a valid file", path)
    return path.exists() and path.is_file()

def get_window_dim():
    """ gets the dimensions depending on python version and os"""
    version = sys.version_info

    if version >= (3, 3):
        return _size_36()
    if platform.system() == 'Windows':
        return _size_windows()
    return _size_27()

def is_line_in_file(filename: str, line: str) -> bool:
    """
    Detects whether a line is present within a file.

    Args:
        filename: file to check
        line: line to search for (as an exact match)
    """
    assert "\n" not in line
    with open(filename, "r") as file:
        for fileline in file:
            if fileline == line:
                return True
        return False

def sort_by_modified(files_or_folders: list) -> list:
    """
    Sort files or folders by modified time

    Args:
        files_or_folders: list of files or folders

    Returns:
        list
    """
    return sorted(files_or_folders, key=os.path.getmtime, reverse=True)

def execute(cur, *args):
    """Utility function to print sqlite queries before executing.

    Use instead of cur.execute().  First argument is cursor.

    cur.execute(stmt)
    becomes
    util.execute(cur, stmt)
    """
    stmt = args[0]
    if len(args) > 1:
        stmt = stmt.replace('%', '%%').replace('?', '%r')
        print(stmt % (args[1]))
    return cur.execute(*args)

def kdot(x, y, K=2):
    """Algorithm 5.10. Dot product algorithm in K-fold working precision,
    K >= 3.
    """
    xx = x.reshape(-1, x.shape[-1])
    yy = y.reshape(y.shape[0], -1)

    xx = numpy.ascontiguousarray(xx)
    yy = numpy.ascontiguousarray(yy)

    r = _accupy.kdot_helper(xx, yy).reshape((-1,) + x.shape[:-1] + y.shape[1:])
    return ksum(r, K - 1)

def uniqued(iterable):
    """Return unique list of ``iterable`` items preserving order.

    >>> uniqued('spameggs')
    ['s', 'p', 'a', 'm', 'e', 'g']
    """
    seen = set()
    return [item for item in iterable if item not in seen and not seen.add(item)]

def string(value) -> str:
        """ string dict/object/value to JSON """
        return system_json.dumps(Json(value).safe_object(), ensure_ascii=False)

def __replace_all(repls: dict, input: str) -> str:
    """ Replaces from a string **input** all the occurrences of some
    symbols according to mapping **repls**.

    :param dict repls: where #key is the old character and
    #value is the one to substitute with;
    :param str input: original string where to apply the
    replacements;
    :return: *(str)* the string with the desired characters replaced
    """
    return re.sub('|'.join(re.escape(key) for key in repls.keys()),
                  lambda k: repls[k.group(0)], input)

def get_keys_of_max_n(dict_obj, n):
    """Returns the keys that maps to the top n max values in the given dict.

    Example:
    --------
    >>> dict_obj = {'a':2, 'b':1, 'c':5}
    >>> get_keys_of_max_n(dict_obj, 2)
    ['a', 'c']
    """
    return sorted([
        item[0]
        for item in sorted(
            dict_obj.items(), key=lambda item: item[1], reverse=True
        )[:n]
    ])

def hsv2rgb_spectrum(hsv):
    """Generates RGB values from HSV values in line with a typical light
    spectrum."""
    h, s, v = hsv
    return hsv2rgb_raw(((h * 192) >> 8, s, v))

def s3_get(url: str, temp_file: IO) -> None:
    """Pull a file directly from S3."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)

def lowercase_chars(string: any) -> str:
        """Return all (and only) the lowercase chars in the given string."""
        return ''.join([c if c.islower() else '' for c in str(string)])

def from_file(file_path) -> dict:
        """ Load JSON file """
        with io.open(file_path, 'r', encoding='utf-8') as json_stream:
            return Json.parse(json_stream, True)

def pad(a, desiredlength):
    """
    Pad an n-dimensional numpy array with zeros along the zero-th dimension
    so that it is the desired length.  Return it unchanged if it is greater
    than or equal to the desired length
    """

    if len(a) >= desiredlength:
        return a

    islist = isinstance(a, list)
    a = np.array(a)
    diff = desiredlength - len(a)
    shape = list(a.shape)
    shape[0] = diff

    padded = np.concatenate([a, np.zeros(shape, dtype=a.dtype)])
    return padded.tolist() if islist else padded

def clean_int(x) -> int:
    """
    Returns its parameter as an integer, or raises
    ``django.forms.ValidationError``.
    """
    try:
        return int(x)
    except ValueError:
        raise forms.ValidationError(
            "Cannot convert to integer: {}".format(repr(x)))

def get_language():
    """
    Wrapper around Django's `get_language` utility.
    For Django >= 1.8, `get_language` returns None in case no translation is activate.
    Here we patch this behavior e.g. for back-end functionality requiring access to translated fields
    """
    from parler import appsettings
    language = dj_get_language()
    if language is None and appsettings.PARLER_DEFAULT_ACTIVATE:
        return appsettings.PARLER_DEFAULT_LANGUAGE_CODE
    else:
        return language

def _darwin_current_arch(self):
        """Add Mac OS X support."""
        if sys.platform == "darwin":
            if sys.maxsize > 2 ** 32: # 64bits.
                return platform.mac_ver()[2] # Both Darwin and Python are 64bits.
            else: # Python 32 bits
                return platform.processor()

def get_from_gnucash26_date(date_str: str) -> date:
    """ Creates a datetime from GnuCash 2.6 date string """
    date_format = "%Y%m%d"
    result = datetime.strptime(date_str, date_format).date()
    return result

def flatten_multidict(multidict):
    """Return flattened dictionary from ``MultiDict``."""
    return dict([(key, value if len(value) > 1 else value[0])
                 for (key, value) in multidict.iterlists()])

def usetz_now():
    """Determine current time depending on USE_TZ setting.

    Affects Django 1.4 and above only. if `USE_TZ = True`, then returns
    current time according to timezone, else returns current UTC time.

    """
    USE_TZ = getattr(settings, 'USE_TZ', False)
    if USE_TZ and DJANGO_VERSION >= '1.4':
        return now()
    else:
        return datetime.utcnow()

def is_finite(value: Any) -> bool:
    """Return true if a value is a finite number."""
    return isinstance(value, int) or (isinstance(value, float) and isfinite(value))

def zfill(x, width):
    """zfill(x, width) -> string

    Pad a numeric string x with zeros on the left, to fill a field
    of the specified width.  The string x is never truncated.

    """
    if not isinstance(x, basestring):
        x = repr(x)
    return x.zfill(width)

def _parse_date(string: str) -> datetime.date:
    """Parse an ISO format date (YYYY-mm-dd).

    >>> _parse_date('1990-01-02')
    datetime.date(1990, 1, 2)
    """
    return datetime.datetime.strptime(string, '%Y-%m-%d').date()

def read_set_from_file(filename: str) -> Set[str]:
    """
    Extract a de-duped collection (set) of text from a file.
    Expected file format is one item per line.
    """
    collection = set()
    with open(filename, 'r') as file_:
        for line in file_:
            collection.add(line.rstrip())
    return collection

def flatten_list(x: List[Any]) -> List[Any]:
    """
    Converts a list of lists into a flat list.
    
    Args:
        x: list of lists 

    Returns:
        flat list
        
    As per
    http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python

    """  # noqa
    return [item for sublist in x for item in sublist]

def dictlist_convert_to_float(dict_list: Iterable[Dict], key: str) -> None:
    """
    Process an iterable of dictionaries. For each dictionary ``d``, convert
    (in place) ``d[key]`` to a float. If that fails, convert it to ``None``.
    """
    for d in dict_list:
        try:
            d[key] = float(d[key])
        except ValueError:
            d[key] = None

def infer_format(filename:str) -> str:
    """Return extension identifying format of given filename"""
    _, ext = os.path.splitext(filename)
    return ext

def most_frequent(lst):
    """
    Returns the item that appears most frequently in the given list.
    """
    lst = lst[:]
    highest_freq = 0
    most_freq = None

    for val in unique(lst):
        if lst.count(val) > highest_freq:
            most_freq = val
            highest_freq = lst.count(val)
            
    return most_freq

def lowercase_chars(string: any) -> str:
        """Return all (and only) the lowercase chars in the given string."""
        return ''.join([c if c.islower() else '' for c in str(string)])

def normalize_column_names(df):
    r""" Clean up whitespace in column names. See better version at `pugnlp.clean_columns`

    >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=['Hello World', 'not here'])
    >>> normalize_column_names(df)
    ['hello_world', 'not_here']
    """
    columns = df.columns if hasattr(df, 'columns') else df
    columns = [c.lower().replace(' ', '_') for c in columns]
    return columns

def split(text: str) -> List[str]:
    """Split a text into a list of tokens.

    :param text: the text to split
    :return: tokens
    """
    return [word for word in SEPARATOR.split(text) if word.strip(' \t')]

def is_sqlatype_integer(coltype: Union[TypeEngine, VisitableType]) -> bool:
    """
    Is the SQLAlchemy column type an integer type?
    """
    coltype = _coltype_to_typeengine(coltype)
    return isinstance(coltype, sqltypes.Integer)

def command(self, cmd, *args):
        """
        Sends a command and an (optional) sequence of arguments through to the
        delegated serial interface. Note that the arguments are passed through
        as data.
        """
        self._serial_interface.command(cmd)
        if len(args) > 0:
            self._serial_interface.data(list(args))

def url_host(url: str) -> str:
    """
    Parses hostname from URL.
    :param url: URL
    :return: hostname
    """
    from urllib.parse import urlparse
    res = urlparse(url)
    return res.netloc.split(':')[0] if res.netloc else ''

def list_to_str(list, separator=','):
    """
    >>> list = [0, 0, 7]
    >>> list_to_str(list)
    '0,0,7'
    """
    list = [str(x) for x in list]
    return separator.join(list)

def dtypes(self):
        """Returns all column names and their data types as a list.

        >>> df.dtypes
        [('age', 'int'), ('name', 'string')]
        """
        return [(str(f.name), f.dataType.simpleString()) for f in self.schema.fields]

def iter_lines(file_like: Iterable[str]) -> Generator[str, None, None]:
    """ Helper for iterating only nonempty lines without line breaks"""
    for line in file_like:
        line = line.rstrip('\r\n')
        if line:
            yield line

def dict_to_enum_fn(d: Dict[str, Any], enum_class: Type[Enum]) -> Enum:
    """
    Converts an ``dict`` to a ``Enum``.
    """
    return enum_class[d['name']]

def is_natural(x):
    """A non-negative integer."""
    try:
        is_integer = int(x) == x
    except (TypeError, ValueError):
        return False
    return is_integer and x >= 0

def dtypes(self):
        """Returns all column names and their data types as a list.

        >>> df.dtypes
        [('age', 'int'), ('name', 'string')]
        """
        return [(str(f.name), f.dataType.simpleString()) for f in self.schema.fields]

def bulk_load_docs(es, docs):
    """Bulk load docs

    Args:
        es: elasticsearch handle
        docs: Iterator of doc objects - includes index_name
    """

    chunk_size = 200

    try:
        results = elasticsearch.helpers.bulk(es, docs, chunk_size=chunk_size)
        log.debug(f"Elasticsearch documents loaded: {results[0]}")

        # elasticsearch.helpers.parallel_bulk(es, terms, chunk_size=chunk_size, thread_count=4)
        if len(results[1]) > 0:
            log.error("Bulk load errors {}".format(results))
    except elasticsearch.ElasticsearchException as e:
        log.error("Indexing error: {}\n".format(e))

def _parse_date(string: str) -> datetime.date:
    """Parse an ISO format date (YYYY-mm-dd).

    >>> _parse_date('1990-01-02')
    datetime.date(1990, 1, 2)
    """
    return datetime.datetime.strptime(string, '%Y-%m-%d').date()

def get_codes(s: Union[str, 'ChainedBase']) -> List[str]:
    """ Grab all escape codes from a string.
        Returns a list of all escape codes.
    """
    return codegrabpat.findall(str(s))

def _parse_tuple_string(argument):
        """ Return a tuple from parsing 'a,b,c,d' -> (a,b,c,d) """
        if isinstance(argument, str):
            return tuple(int(p.strip()) for p in argument.split(','))
        return argument

def hex_color_to_tuple(hex):
    """ convent hex color to tuple
    "#ffffff"   ->  (255, 255, 255)
    "#ffff00ff" ->  (255, 255, 0, 255)
    """
    hex = hex[1:]
    length = len(hex) // 2
    return tuple(int(hex[i*2:i*2+2], 16) for i in range(length))

def has_key(cls, *args):
        """
        Check whether flyweight object with specified key has already been created.

        Returns:
            bool: True if already created, False if not
        """
        key = args if len(args) > 1 else args[0]
        return key in cls._instances

def is_valid(cls, arg):
        """Return True if arg is valid value for the class.  If the string
        value is wrong for the enumeration, the encoding will fail.
        """
        return (isinstance(arg, (int, long)) and (arg >= 0)) or \
            isinstance(arg, basestring)

def setdefault(self, name: str, default: Any=None) -> Any:
        """Set an attribute with a default value."""
        return self.__dict__.setdefault(name, default)

def post(self, endpoint: str, **kwargs) -> dict:
        """HTTP POST operation to API endpoint."""

        return self._request('POST', endpoint, **kwargs)

def decode(string, base):
    """
    Given a string (string) and a numeric base (base),
    decode the string into an integer.

    Returns the integer
    """

    base = int(base)
    code_string = get_code_string(base)
    result = 0
    if base == 16:
        string = string.lower()
    while len(string) > 0:
        result *= base
        result += code_string.find(string[0])
        string = string[1:]
    return result

def _duplicates(list_):
    """Return dict mapping item -> indices."""
    item_indices = {}
    for i, item in enumerate(list_):
        try:
            item_indices[item].append(i)
        except KeyError:  # First time seen
            item_indices[item] = [i]
    return item_indices

def enrich_complexes(graph: BELGraph) -> None:
    """Add all of the members of the complex abundances to the graph."""
    nodes = list(get_nodes_by_function(graph, COMPLEX))
    for u in nodes:
        for v in u.members:
            graph.add_has_component(u, v)

def get_datatype(self, table: str, column: str) -> str:
        """Returns database SQL datatype for a column: e.g. VARCHAR."""
        return self.flavour.get_datatype(self, table, column).upper()

def iter_lines(file_like: Iterable[str]) -> Generator[str, None, None]:
    """ Helper for iterating only nonempty lines without line breaks"""
    for line in file_like:
        line = line.rstrip('\r\n')
        if line:
            yield line

def cmd_dot(conf: Config):
    """Print out a neat targets dependency tree based on requested targets.

    Use graphviz to render the dot file, e.g.:

    > ybt dot :foo :bar | dot -Tpng -o graph.png
    """
    build_context = BuildContext(conf)
    populate_targets_graph(build_context, conf)
    if conf.output_dot_file is None:
        write_dot(build_context, conf, sys.stdout)
    else:
        with open(conf.output_dot_file, 'w') as out_file:
            write_dot(build_context, conf, out_file)

def debugTreePrint(node,pfx="->"):
  """Purely a debugging aid: Ascii-art picture of a tree descended from node"""
  print pfx,node.item
  for c in node.children:
    debugTreePrint(c,"  "+pfx)

def is_orthogonal(
        matrix: np.ndarray,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8) -> bool:
    """Determines if a matrix is approximately orthogonal.

    A matrix is orthogonal if it's square and real and its transpose is its
    inverse.

    Args:
        matrix: The matrix to check.
        rtol: The per-matrix-entry relative tolerance on equality.
        atol: The per-matrix-entry absolute tolerance on equality.

    Returns:
        Whether the matrix is orthogonal within the given tolerance.
    """
    return (matrix.shape[0] == matrix.shape[1] and
            np.all(np.imag(matrix) == 0) and
            np.allclose(matrix.dot(matrix.T), np.eye(matrix.shape[0]),
                        rtol=rtol,
                        atol=atol))

def layer_with(self, sample: np.ndarray, value: int) -> np.ndarray:
        """Create an identical 2d array where the second row is filled with value"""
        b = np.full((2, len(sample)), value, dtype=float)
        b[0] = sample
        return b

def inverted_dict(d):
    """Return a dict with swapped keys and values

    >>> inverted_dict({0: ('a', 'b'), 1: 'cd'}) == {'cd': 1, ('a', 'b'): 0}
    True
    """
    return dict((force_hashable(v), k) for (k, v) in viewitems(dict(d)))

def valid_file(path: str) -> bool:
    """
    Verifies that a string path actually exists and is a file

    :param path: The path to verify
    :return: **True** if path exist and is a file
    """
    path = Path(path).expanduser()
    log.debug("checking if %s is a valid file", path)
    return path.exists() and path.is_file()

def input_validate_str(string, name, max_len=None, exact_len=None):
    """ Input validation for strings. """
    if type(string) is not str:
        raise pyhsm.exception.YHSM_WrongInputType(name, str, type(string))
    if max_len != None and len(string) > max_len:
        raise pyhsm.exception.YHSM_InputTooLong(name, max_len, len(string))
    if exact_len != None and len(string) != exact_len:
        raise pyhsm.exception.YHSM_WrongInputSize(name, exact_len, len(string))
    return string

def issuperset(self, items):
        """Return whether this collection contains all items.

        >>> Unique(['spam', 'eggs']).issuperset(['spam', 'spam', 'spam'])
        True
        """
        return all(_compat.map(self._seen.__contains__, items))

def is_line_in_file(filename: str, line: str) -> bool:
    """
    Detects whether a line is present within a file.

    Args:
        filename: file to check
        line: line to search for (as an exact match)
    """
    assert "\n" not in line
    with open(filename, "r") as file:
        for fileline in file:
            if fileline == line:
                return True
        return False

def inverted_dict(d):
    """Return a dict with swapped keys and values

    >>> inverted_dict({0: ('a', 'b'), 1: 'cd'}) == {'cd': 1, ('a', 'b'): 0}
    True
    """
    return dict((force_hashable(v), k) for (k, v) in viewitems(dict(d)))

def strings_to_integers(strings: Iterable[str]) -> Iterable[int]:
    """
    Convert a list of strings to a list of integers.

    :param strings: a list of string
    :return: a list of converted integers

    .. doctest::

        >>> strings_to_integers(['1', '1.0', '-0.2'])
        [1, 1, 0]
    """
    return strings_to_(strings, lambda x: int(float(x)))

def _check_whitespace(string):
    """
    Make sure thre is no whitespace in the given string. Will raise a
    ValueError if whitespace is detected
    """
    if string.count(' ') + string.count('\t') + string.count('\n') > 0:
        raise ValueError(INSTRUCTION_HAS_WHITESPACE)

def try_cast_int(s):
    """(str) -> int
    All the digits in a given string are concatenated and converted into a single number.
    """
    try:
        temp = re.findall('\d', str(s))
        temp = ''.join(temp)
        return int(temp)
    except:
        return s

def has_changed (filename):
    """Check if filename has changed since the last check. If this
    is the first check, assume the file is changed."""
    key = os.path.abspath(filename)
    mtime = get_mtime(key)
    if key not in _mtime_cache:
        _mtime_cache[key] = mtime
        return True
    return mtime > _mtime_cache[key]

def connect_to_database_odbc_access(self,
                                        dsn: str,
                                        autocommit: bool = True) -> None:
        """Connects to an Access database via ODBC, with the DSN
        prespecified."""
        self.connect(engine=ENGINE_ACCESS, interface=INTERFACE_ODBC,
                     dsn=dsn, autocommit=autocommit)

def clean_column_names(df: DataFrame) -> DataFrame:
    """
    Strip the whitespace from all column names in the given DataFrame
    and return the result.
    """
    f = df.copy()
    f.columns = [col.strip() for col in f.columns]
    return f

def _get_tuple(self, fields):
    """
    :param fields: a list which contains either 0,1,or 2 values
    :return: a tuple with default values of '';
    """
    v1 = ''
    v2 = ''
    if len(fields) > 0:
      v1 = fields[0]
    if len(fields) > 1:
      v2 = fields[1]
    return v1, v2

def butlast(iterable):
    """Yield all items from ``iterable`` except the last one.

    >>> list(butlast(['spam', 'eggs', 'ham']))
    ['spam', 'eggs']

    >>> list(butlast(['spam']))
    []

    >>> list(butlast([]))
    []
    """
    iterable = iter(iterable)
    try:
        first = next(iterable)
    except StopIteration:
        return
    for second in iterable:
        yield first
        first = second

def autoreload(self, parameter_s=''):
        r"""%autoreload => Reload modules automatically

        %autoreload
        Reload all modules (except those excluded by %aimport) automatically
        now.

        %autoreload 0
        Disable automatic reloading.

        %autoreload 1
        Reload all modules imported with %aimport every time before executing
        the Python code typed.

        %autoreload 2
        Reload all modules (except those excluded by %aimport) every time
        before executing the Python code typed.

        Reloading Python modules in a reliable way is in general
        difficult, and unexpected things may occur. %autoreload tries to
        work around common pitfalls by replacing function code objects and
        parts of classes previously in the module with new versions. This
        makes the following things to work:

        - Functions and classes imported via 'from xxx import foo' are upgraded
          to new versions when 'xxx' is reloaded.

        - Methods and properties of classes are upgraded on reload, so that
          calling 'c.foo()' on an object 'c' created before the reload causes
          the new code for 'foo' to be executed.

        Some of the known remaining caveats are:

        - Replacing code objects does not always succeed: changing a @property
          in a class to an ordinary method or a method to a member variable
          can cause problems (but in old objects only).

        - Functions that are removed (eg. via monkey-patching) from a module
          before it is reloaded are not upgraded.

        - C extension modules cannot be reloaded, and so cannot be
          autoreloaded.

        """
        if parameter_s == '':
            self._reloader.check(True)
        elif parameter_s == '0':
            self._reloader.enabled = False
        elif parameter_s == '1':
            self._reloader.check_all = False
            self._reloader.enabled = True
        elif parameter_s == '2':
            self._reloader.check_all = True
            self._reloader.enabled = True

def _centroids(n_clusters: int, points: List[List[float]]) -> List[List[float]]:
    """ Return n_clusters centroids of points
    """

    k_means = KMeans(n_clusters=n_clusters)
    k_means.fit(points)

    closest, _ = pairwise_distances_argmin_min(k_means.cluster_centers_, points)

    return list(map(list, np.array(points)[closest.tolist()]))

def product(*args, **kwargs):
    """ Yields all permutations with replacement:
        list(product("cat", repeat=2)) => 
        [("c", "c"), 
         ("c", "a"), 
         ("c", "t"), 
         ("a", "c"), 
         ("a", "a"), 
         ("a", "t"), 
         ("t", "c"), 
         ("t", "a"), 
         ("t", "t")]
    """
    p = [[]]
    for iterable in map(tuple, args) * kwargs.get("repeat", 1):
        p = [x + [y] for x in p for y in iterable]
    for p in p:
        yield tuple(p)

def inner(tensor0: BKTensor, tensor1: BKTensor) -> BKTensor:
    """Return the inner product between two tensors"""
    # Note: Relying on fact that vdot flattens arrays
    return np.vdot(tensor0, tensor1)

def _str_to_list(value, separator):
    """Convert a string to a list with sanitization."""
    value_list = [item.strip() for item in value.split(separator)]
    value_list_sanitized = builtins.list(filter(None, value_list))
    if len(value_list_sanitized) > 0:
        return value_list_sanitized
    else:
        raise ValueError('Invalid list variable.')

def _sum_cycles_from_tokens(self, tokens: List[str]) -> int:
        """Sum the total number of cycles over a list of tokens."""
        return sum((int(self._nonnumber_pattern.sub('', t)) for t in tokens))

def strings_to_integers(strings: Iterable[str]) -> Iterable[int]:
    """
    Convert a list of strings to a list of integers.

    :param strings: a list of string
    :return: a list of converted integers

    .. doctest::

        >>> strings_to_integers(['1', '1.0', '-0.2'])
        [1, 1, 0]
    """
    return strings_to_(strings, lambda x: int(float(x)))

def snake_to_camel(s: str) -> str:
    """Convert string from snake case to camel case."""

    fragments = s.split('_')

    return fragments[0] + ''.join(x.title() for x in fragments[1:])

def strings_to_integers(strings: Iterable[str]) -> Iterable[int]:
    """
    Convert a list of strings to a list of integers.

    :param strings: a list of string
    :return: a list of converted integers

    .. doctest::

        >>> strings_to_integers(['1', '1.0', '-0.2'])
        [1, 1, 0]
    """
    return strings_to_(strings, lambda x: int(float(x)))

def _cnx_is_empty(in_file):
    """Check if cnr or cns files are empty (only have a header)
    """
    with open(in_file) as in_handle:
        for i, line in enumerate(in_handle):
            if i > 0:
                return False
    return True

def list_to_str(list, separator=','):
    """
    >>> list = [0, 0, 7]
    >>> list_to_str(list)
    '0,0,7'
    """
    list = [str(x) for x in list]
    return separator.join(list)

def read32(bytestream):
  """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def isarray(array, test, dim=2):
    """Returns True if test is True for all array elements.
    Otherwise, returns False.
    """
    if dim > 1:
        return all(isarray(array[i], test, dim - 1)
                   for i in range(len(array)))
    return all(test(i) for i in array)

def _check_whitespace(string):
    """
    Make sure thre is no whitespace in the given string. Will raise a
    ValueError if whitespace is detected
    """
    if string.count(' ') + string.count('\t') + string.count('\n') > 0:
        raise ValueError(INSTRUCTION_HAS_WHITESPACE)

def chars(string: any) -> str:
        """Return all (and only) the chars in the given string."""
        return ''.join([c if c.isalpha() else '' for c in str(string)])

def rms(x):
    """"Root Mean Square"

    Arguments:
        x (seq of float): A sequence of numerical values

    Returns:
        The square root of the average of the squares of the values

        math.sqrt(sum(x_i**2 for x_i in x) / len(x))

        or

        return (np.array(x) ** 2).mean() ** 0.5

    >>> rms([0, 2, 4, 4])
    3.0
    """
    try:
        return (np.array(x) ** 2).mean() ** 0.5
    except:
        x = np.array(dropna(x))
        invN = 1.0 / len(x)
        return (sum(invN * (x_i ** 2) for x_i in x)) ** .5

def memory_full():
    """Check if the memory is too full for further caching."""
    current_process = psutil.Process(os.getpid())
    return (current_process.memory_percent() >
            config.MAXIMUM_CACHE_MEMORY_PERCENTAGE)

def get_unique_links(self):
        """ Get all unique links in the html of the page source.
            Page links include those obtained from:
            "a"->"href", "img"->"src", "link"->"href", and "script"->"src". """
        page_url = self.get_current_url()
        soup = self.get_beautiful_soup(self.get_page_source())
        links = page_utils._get_unique_links(page_url, soup)
        return links

def is_prime(n):
    """
    Check if n is a prime number
    """
    if n % 2 == 0 and n > 2:
        return False
    return all(n % i for i in range(3, int(math.sqrt(n)) + 1, 2))

def top(self, topn=10):
        """
        Get a list of the top ``topn`` features in this :class:`.Feature`\.

        Examples
        --------

        .. code-block:: python

        >>> myFeature = Feature([('the', 2), ('pine', 1), ('trapezoid', 5)])
        >>> myFeature.top(1)
        [('trapezoid', 5)]

        Parameters
        ----------
        topn : int

        Returns
        -------
        list
        """
        return [self[i] for i in argsort(list(zip(*self))[1])[::-1][:topn]]

def post(self, endpoint: str, **kwargs) -> dict:
        """HTTP POST operation to API endpoint."""

        return self._request('POST', endpoint, **kwargs)

def datetime_from_isoformat(value: str):
    """Return a datetime object from an isoformat string.

    Args:
        value (str): Datetime string in isoformat.

    """
    if sys.version_info >= (3, 7):
        return datetime.fromisoformat(value)

    return datetime.strptime(value, '%Y-%m-%dT%H:%M:%S.%f')

def samefile(a: str, b: str) -> bool:
    """Check if two pathes represent the same file."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normpath(a) == os.path.normpath(b)

def mouse_event(dwFlags: int, dx: int, dy: int, dwData: int, dwExtraInfo: int) -> None:
    """mouse_event from Win32."""
    ctypes.windll.user32.mouse_event(dwFlags, dx, dy, dwData, dwExtraInfo)

def setup_cache(app: Flask, cache_config) -> Optional[Cache]:
    """Setup the flask-cache on a flask app"""
    if cache_config and cache_config.get('CACHE_TYPE') != 'null':
        return Cache(app, config=cache_config)

    return None

def _find_conda():
    """Find the conda executable robustly across conda versions.

    Returns
    -------
    conda : str
        Path to the conda executable.

    Raises
    ------
    IOError
        If the executable cannot be found in either the CONDA_EXE environment
        variable or in the PATH.

    Notes
    -----
    In POSIX platforms in conda >= 4.4, conda can be set up as a bash function
    rather than an executable. (This is to enable the syntax
    ``conda activate env-name``.) In this case, the environment variable
    ``CONDA_EXE`` contains the path to the conda executable. In other cases,
    we use standard search for the appropriate name in the PATH.

    See https://github.com/airspeed-velocity/asv/issues/645 for more details.
    """
    if 'CONDA_EXE' in os.environ:
        conda = os.environ['CONDA_EXE']
    else:
        conda = util.which('conda')
    return conda

def multi_split(s, split):
    # type: (S, Iterable[S]) -> List[S]
    """Splits on multiple given separators."""
    for r in split:
        s = s.replace(r, "|")
    return [i for i in s.split("|") if len(i) > 0]

def read_text_from_file(path: str) -> str:
    """ Reads text file contents """
    with open(path) as text_file:
        content = text_file.read()

    return content

def string(value) -> str:
        """ string dict/object/value to JSON """
        return system_json.dumps(Json(value).safe_object(), ensure_ascii=False)

def bfx(value, msb, lsb):
    """! @brief Extract a value from a bitfield."""
    mask = bitmask((msb, lsb))
    return (value & mask) >> lsb

def clean_int(x) -> int:
    """
    Returns its parameter as an integer, or raises
    ``django.forms.ValidationError``.
    """
    try:
        return int(x)
    except ValueError:
        raise forms.ValidationError(
            "Cannot convert to integer: {}".format(repr(x)))

def read_text_from_file(path: str) -> str:
    """ Reads text file contents """
    with open(path) as text_file:
        content = text_file.read()

    return content

def list_depth(list_, func=max, _depth=0):
    """
    Returns the deepest level of nesting within a list of lists

    Args:
       list_  : a nested listlike object
       func   : depth aggregation strategy (defaults to max)
       _depth : internal var

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_list import *  # NOQA
        >>> list_ = [[[[[1]]], [3]], [[1], [3]], [[1], [3]]]
        >>> result = (list_depth(list_, _depth=0))
        >>> print(result)

    """
    depth_list = [list_depth(item, func=func, _depth=_depth + 1)
                  for item in  list_ if util_type.is_listlike(item)]
    if len(depth_list) > 0:
        return func(depth_list)
    else:
        return _depth

def is_inside_lambda(node: astroid.node_classes.NodeNG) -> bool:
    """Return true if given node is inside lambda"""
    parent = node.parent
    while parent is not None:
        if isinstance(parent, astroid.Lambda):
            return True
        parent = parent.parent
    return False

def get_unique_links(self):
        """ Get all unique links in the html of the page source.
            Page links include those obtained from:
            "a"->"href", "img"->"src", "link"->"href", and "script"->"src". """
        page_url = self.get_current_url()
        soup = self.get_beautiful_soup(self.get_page_source())
        links = page_utils._get_unique_links(page_url, soup)
        return links

def argmax(self, rows: List[Row], column: ComparableColumn) -> List[Row]:
        """
        Takes a list of rows and a column name and returns a list containing a single row (dict from
        columns to cells) that has the maximum numerical value in the given column. We return a list
        instead of a single dict to be consistent with the return type of ``select`` and
        ``all_rows``.
        """
        if not rows:
            return []
        value_row_pairs = [(row.values[column.name], row) for row in rows]
        if not value_row_pairs:
            return []
        # Returns a list containing the row with the max cell value.
        return [sorted(value_row_pairs, key=lambda x: x[0], reverse=True)[0][1]]

def identify_request(request: RequestType) -> bool:
    """
    Try to identify whether this is an ActivityPub request.
    """
    # noinspection PyBroadException
    try:
        data = json.loads(decode_if_bytes(request.body))
        if "@context" in data:
            return True
    except Exception:
        pass
    return False

def dtypes(self):
        """Returns all column names and their data types as a list.

        >>> df.dtypes
        [('age', 'int'), ('name', 'string')]
        """
        return [(str(f.name), f.dataType.simpleString()) for f in self.schema.fields]

def is_sqlatype_integer(coltype: Union[TypeEngine, VisitableType]) -> bool:
    """
    Is the SQLAlchemy column type an integer type?
    """
    coltype = _coltype_to_typeengine(coltype)
    return isinstance(coltype, sqltypes.Integer)

def url_host(url: str) -> str:
    """
    Parses hostname from URL.
    :param url: URL
    :return: hostname
    """
    from urllib.parse import urlparse
    res = urlparse(url)
    return res.netloc.split(':')[0] if res.netloc else ''

def has_key(cls, *args):
        """
        Check whether flyweight object with specified key has already been created.

        Returns:
            bool: True if already created, False if not
        """
        key = args if len(args) > 1 else args[0]
        return key in cls._instances

def enum_mark_last(iterable, start=0):
    """
    Returns a generator over iterable that tells whether the current item is the last one.
    Usage:
        >>> iterable = range(10)
        >>> for index, is_last, item in enum_mark_last(iterable):
        >>>     print(index, item, end='\n' if is_last else ', ')
    """
    it = iter(iterable)
    count = start
    try:
        last = next(it)
    except StopIteration:
        return
    for val in it:
        yield count, False, last
        last = val
        count += 1
    yield count, True, last

def _duplicates(list_):
    """Return dict mapping item -> indices."""
    item_indices = {}
    for i, item in enumerate(list_):
        try:
            item_indices[item].append(i)
        except KeyError:  # First time seen
            item_indices[item] = [i]
    return item_indices

def get_all_args(fn) -> list:
    """
    Returns a list of all arguments for the function fn.

    >>> def foo(x, y, z=100): return x + y + z
    >>> get_all_args(foo)
    ['x', 'y', 'z']
    """
    sig = inspect.signature(fn)
    return list(sig.parameters)

def _kbhit_unix() -> bool:
    """
    Under UNIX: is a keystroke available?
    """
    dr, dw, de = select.select([sys.stdin], [], [], 0)
    return dr != []

def get_timezone() -> Tuple[datetime.tzinfo, str]:
    """Discover the current time zone and it's standard string representation (for source{d})."""
    dt = get_datetime_now().astimezone()
    tzstr = dt.strftime("%z")
    tzstr = tzstr[:-2] + ":" + tzstr[-2:]
    return dt.tzinfo, tzstr

def copy_session(session: requests.Session) -> requests.Session:
    """Duplicates a requests.Session."""
    new = requests.Session()
    new.cookies = requests.utils.cookiejar_from_dict(requests.utils.dict_from_cookiejar(session.cookies))
    new.headers = session.headers.copy()
    return new

def _check_whitespace(string):
    """
    Make sure thre is no whitespace in the given string. Will raise a
    ValueError if whitespace is detected
    """
    if string.count(' ') + string.count('\t') + string.count('\n') > 0:
        raise ValueError(INSTRUCTION_HAS_WHITESPACE)

def debugTreePrint(node,pfx="->"):
  """Purely a debugging aid: Ascii-art picture of a tree descended from node"""
  print pfx,node.item
  for c in node.children:
    debugTreePrint(c,"  "+pfx)

def flatten_list(l: List[list]) -> list:
    """ takes a list of lists, l and returns a flat list
    """
    return [v for inner_l in l for v in inner_l]

def enumerate_chunks (phrase, spacy_nlp):
    """
    iterate through the noun phrases
    """
    if (len(phrase) > 1):
        found = False
        text = " ".join([rl.text for rl in phrase])
        doc = spacy_nlp(text.strip(), parse=True)

        for np in doc.noun_chunks:
            if np.text != text:
                found = True
                yield np.text, find_chunk(phrase, np.text.split(" "))

        if not found and all([rl.pos[0] != "v" for rl in phrase]):
            yield text, phrase

def _my_hash(arg_list):
    # type: (List[Any]) -> int
    """Simple helper hash function"""
    res = 0
    for arg in arg_list:
        res = res * 31 + hash(arg)
    return res

def convert_column(self, values):
        """Normalize values."""
        assert all(values >= 0), 'Cannot normalize a column with negatives'
        total = sum(values)
        if total > 0:
            return values / total
        else:
            return values

def __next__(self):
        """
        :return: int
        """
        self.current += 1
        if self.current > self.total:
            raise StopIteration
        else:
            return self.iterable[self.current - 1]

def uniqued(iterable):
    """Return unique list of items preserving order.

    >>> uniqued([3, 2, 1, 3, 2, 1, 0])
    [3, 2, 1, 0]
    """
    seen = set()
    add = seen.add
    return [i for i in iterable if i not in seen and not add(i)]

def run_web(self, flask, host='127.0.0.1', port=5000, **options):
        # type: (Zsl, str, int, **Any)->None
        """Alias for Flask.run"""
        return flask.run(
            host=flask.config.get('FLASK_HOST', host),
            port=flask.config.get('FLASK_PORT', port),
            debug=flask.config.get('DEBUG', False),
            **options
        )

def wipe_table(self, table: str) -> int:
        """Delete all records from a table. Use caution!"""
        sql = "DELETE FROM " + self.delimit(table)
        return self.db_exec(sql)

def write_text(filename: str, text: str) -> None:
    """
    Writes text to a file.
    """
    with open(filename, 'w') as f:  # type: TextIO
        print(text, file=f)

def get_period_last_3_months() -> str:
    """ Returns the last week as a period string """
    today = Datum()
    today.today()

    # start_date = today - timedelta(weeks=13)
    start_date = today.clone()
    start_date.subtract_months(3)

    period = get_period(start_date.date, today.date)
    return period

def fast_median(a):
    """Fast median operation for masked array using 50th-percentile
    """
    a = checkma(a)
    #return scoreatpercentile(a.compressed(), 50)
    if a.count() > 0:
        out = np.percentile(a.compressed(), 50)
    else:
        out = np.ma.masked
    return out

def _is_video(filepath) -> bool:
    """Check filename extension to see if it's a video file."""
    if os.path.exists(filepath):  # Could be broken symlink
        extension = os.path.splitext(filepath)[1]
        return extension in ('.mkv', '.mp4', '.avi')
    else:
        return False

def snake_to_camel(s: str) -> str:
    """Convert string from snake case to camel case."""

    fragments = s.split('_')

    return fragments[0] + ''.join(x.title() for x in fragments[1:])

def check_key(self, key: str) -> bool:
        """
        Checks if key exists in datastore. True if yes, False if no.

        :param: SHA512 hash key

        :return: whether or key not exists in datastore
        """
        keys = self.get_keys()
        return key in keys

def shape(self) -> Tuple[int, ...]:
        """Shape of histogram's data.

        Returns
        -------
        One-element tuple with the number of bins along each axis.
        """
        return tuple(bins.bin_count for bins in self._binnings)

def str_upper(x):
    """Converts all strings in a column to uppercase.

    :returns: an expression containing the converted strings.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.


    >>> df.text.str.upper()
    Expression = str_upper(text)
    Length: 5 dtype: str (expression)
    ---------------------------------
    0    SOMETHING
    1  VERY PRETTY
    2    IS COMING
    3          OUR
    4         WAY.

    """
    sl = _to_string_sequence(x).upper()
    return column.ColumnStringArrow(sl.bytes, sl.indices, sl.length, sl.offset, string_sequence=sl)

def combine_pdf_as_bytes(pdfs: List[BytesIO]) -> bytes:
    """Combine PDFs and return a byte-string with the result.

    Arguments
    ---------
    pdfs
        A list of BytesIO representations of PDFs

    """
    writer = PdfWriter()
    for pdf in pdfs:
        writer.addpages(PdfReader(pdf).pages)
    bio = BytesIO()
    writer.write(bio)
    bio.seek(0)
    output = bio.read()
    bio.close()
    return output

def cpu_count() -> int:
    """Returns the number of processors on this machine."""
    if multiprocessing is None:
        return 1
    try:
        return multiprocessing.cpu_count()
    except NotImplementedError:
        pass
    try:
        return os.sysconf("SC_NPROCESSORS_CONF")
    except (AttributeError, ValueError):
        pass
    gen_log.error("Could not detect number of processors; assuming 1")
    return 1

def array2string(arr: numpy.ndarray) -> str:
        """Format numpy array as a string."""
        shape = str(arr.shape)[1:-1]
        if shape.endswith(","):
            shape = shape[:-1]
        return numpy.array2string(arr, threshold=11) + "%s[%s]" % (arr.dtype, shape)

def to_np(*args):
    """ convert GPU arras to numpy and return them"""
    if len(args) > 1:
        return (cp.asnumpy(x) for x in args)
    else:
        return cp.asnumpy(args[0])

def is_natural(x):
    """A non-negative integer."""
    try:
        is_integer = int(x) == x
    except (TypeError, ValueError):
        return False
    return is_integer and x >= 0

def psutil_phymem_usage():
    """
    Return physical memory usage (float)
    Requires the cross-platform psutil (>=v0.3) library
    (https://github.com/giampaolo/psutil)
    """
    import psutil
    # This is needed to avoid a deprecation warning error with
    # newer psutil versions
    try:
        percent = psutil.virtual_memory().percent
    except:
        percent = psutil.phymem_usage().percent
    return percent

def integer_partition(size: int, nparts: int) -> Iterator[List[List[int]]]:
    """ Partition a list of integers into a list of partitions """
    for part in algorithm_u(range(size), nparts):
        yield part

def snake_to_camel(s: str) -> str:
    """Convert string from snake case to camel case."""

    fragments = s.split('_')

    return fragments[0] + ''.join(x.title() for x in fragments[1:])

def _isint(string):
    """
    >>> _isint("123")
    True
    >>> _isint("123.45")
    False
    """
    return type(string) is int or \
           (isinstance(string, _binary_type) or isinstance(string, _text_type)) and \
           _isconvertible(int, string)

def valid_date(x: str) -> bool:
    """
    Retrun ``True`` if ``x`` is a valid YYYYMMDD date;
    otherwise return ``False``.
    """
    try:
        if x != dt.datetime.strptime(x, DATE_FORMAT).strftime(DATE_FORMAT):
            raise ValueError
        return True
    except ValueError:
        return False

def session_expired(self):
        """
        Returns True if login_time not set or seconds since
        login time is greater than 200 mins.
        """
        if not self._login_time or (datetime.datetime.now()-self._login_time).total_seconds() > 12000:
            return True

def top(self, topn=10):
        """
        Get a list of the top ``topn`` features in this :class:`.Feature`\.

        Examples
        --------

        .. code-block:: python

        >>> myFeature = Feature([('the', 2), ('pine', 1), ('trapezoid', 5)])
        >>> myFeature.top(1)
        [('trapezoid', 5)]

        Parameters
        ----------
        topn : int

        Returns
        -------
        list
        """
        return [self[i] for i in argsort(list(zip(*self))[1])[::-1][:topn]]

def random_name_gen(size=6):
    """Generate a random python attribute name."""

    return ''.join(
        [random.choice(string.ascii_uppercase)] +
        [random.choice(string.ascii_uppercase + string.digits) for i in range(size - 1)]
    ) if size > 0 else ''

def _parse_tuple_string(argument):
        """ Return a tuple from parsing 'a,b,c,d' -> (a,b,c,d) """
        if isinstance(argument, str):
            return tuple(int(p.strip()) for p in argument.split(','))
        return argument

def repl_complete(text: str, state: int) -> Optional[str]:
    """Completer function for Python's readline/libedit implementation."""
    # Can't complete Keywords, Numerals
    if __NOT_COMPLETEABLE.match(text):
        return None
    elif text.startswith(":"):
        completions = kw.complete(text)
    else:
        ns = get_current_ns()
        completions = ns.complete(text)

    return list(completions)[state] if completions is not None else None

def clean(ctx, text):
    """
    Removes all non-printable characters from a text string
    """
    text = conversions.to_string(text, ctx)
    return ''.join([c for c in text if ord(c) >= 32])

def _brief_print_list(lst, limit=7):
    """Print at most `limit` elements of list."""
    lst = list(lst)
    if len(lst) > limit:
        return _brief_print_list(lst[:limit//2], limit) + ', ..., ' + \
            _brief_print_list(lst[-limit//2:], limit)
    return ', '.join(["'%s'"%str(i) for i in lst])

def normalize(numbers):
    """Multiply each number by a constant such that the sum is 1.0
    >>> normalize([1,2,1])
    [0.25, 0.5, 0.25]
    """
    total = float(sum(numbers))
    return [n / total for n in numbers]

def _cnx_is_empty(in_file):
    """Check if cnr or cns files are empty (only have a header)
    """
    with open(in_file) as in_handle:
        for i, line in enumerate(in_handle):
            if i > 0:
                return False
    return True

def get_keys_of_max_n(dict_obj, n):
    """Returns the keys that maps to the top n max values in the given dict.

    Example:
    --------
    >>> dict_obj = {'a':2, 'b':1, 'c':5}
    >>> get_keys_of_max_n(dict_obj, 2)
    ['a', 'c']
    """
    return sorted([
        item[0]
        for item in sorted(
            dict_obj.items(), key=lambda item: item[1], reverse=True
        )[:n]
    ])

def get_last_day_of_month(t: datetime) -> int:
    """
    Returns day number of the last day of the month
    :param t: datetime
    :return: int
    """
    tn = t + timedelta(days=32)
    tn = datetime(year=tn.year, month=tn.month, day=1)
    tt = tn - timedelta(hours=1)
    return tt.day

def try_cast_int(s):
    """(str) -> int
    All the digits in a given string are concatenated and converted into a single number.
    """
    try:
        temp = re.findall('\d', str(s))
        temp = ''.join(temp)
        return int(temp)
    except:
        return s

def normalize(numbers):
    """Multiply each number by a constant such that the sum is 1.0
    >>> normalize([1,2,1])
    [0.25, 0.5, 0.25]
    """
    total = float(sum(numbers))
    return [n / total for n in numbers]

def clean(ctx, text):
    """
    Removes all non-printable characters from a text string
    """
    text = conversions.to_string(text, ctx)
    return ''.join([c for c in text if ord(c) >= 32])

def iter_lines(file_like: Iterable[str]) -> Generator[str, None, None]:
    """ Helper for iterating only nonempty lines without line breaks"""
    for line in file_like:
        line = line.rstrip('\r\n')
        if line:
            yield line

def clean(ctx, text):
    """
    Removes all non-printable characters from a text string
    """
    text = conversions.to_string(text, ctx)
    return ''.join([c for c in text if ord(c) >= 32])

def ranges_to_set(lst):
    """
    Convert a list of ranges to a set of numbers::

    >>> ranges = [(1,3), (5,6)]
    >>> sorted(list(ranges_to_set(ranges)))
    [1, 2, 3, 5, 6]

    """
    return set(itertools.chain(*(range(x[0], x[1]+1) for x in lst)))

def list_to_str(list, separator=','):
    """
    >>> list = [0, 0, 7]
    >>> list_to_str(list)
    '0,0,7'
    """
    list = [str(x) for x in list]
    return separator.join(list)

def count(args):
    """ count occurences in a list of lists
    >>> count([['a','b'],['a']])
    defaultdict(int, {'a' : 2, 'b' : 1})
    """
    counts = defaultdict(int)
    for arg in args:
        for item in arg:
            counts[item] = counts[item] + 1
    return counts

def make_dep_graph(depender):
	"""Returns a digraph string fragment based on the passed-in module
	"""
	shutit_global.shutit_global_object.yield_to_draw()
	digraph = ''
	for dependee_id in depender.depends_on:
		digraph = (digraph + '"' + depender.module_id + '"->"' + dependee_id + '";\n')
	return digraph

def quaternion_imag(quaternion):
    """Return imaginary part of quaternion.

    >>> quaternion_imag([3, 0, 1, 2])
    array([ 0.,  1.,  2.])

    """
    return numpy.array(quaternion[1:4], dtype=numpy.float64, copy=True)

def multiple_replace(string, replacements):
    # type: (str, Dict[str,str]) -> str
    """Simultaneously replace multiple strigns in a string

    Args:
        string (str): Input string
        replacements (Dict[str,str]): Replacements dictionary

    Returns:
        str: String with replacements

    """
    pattern = re.compile("|".join([re.escape(k) for k in sorted(replacements, key=len, reverse=True)]), flags=re.DOTALL)
    return pattern.sub(lambda x: replacements[x.group(0)], string)

def _cnx_is_empty(in_file):
    """Check if cnr or cns files are empty (only have a header)
    """
    with open(in_file) as in_handle:
        for i, line in enumerate(in_handle):
            if i > 0:
                return False
    return True

def _read_section(self):
        """Read and return an entire section"""
        lines = [self._last[self._last.find(":")+1:]]
        self._last = self._f.readline()
        while len(self._last) > 0 and len(self._last[0].strip()) == 0:
            lines.append(self._last)
            self._last = self._f.readline()
        return lines

def get_tokens(line: str) -> Iterator[str]:
    """
    Yields tokens from input string.

    :param line: Input string.
    :return: Iterator over tokens.
    """
    for token in line.rstrip().split():
        if len(token) > 0:
            yield token

def list_to_str(lst):
    """
    Turn a list into a comma- and/or and-separated string.

    Parameters
    ----------
    lst : :obj:`list`
        A list of strings to join into a single string.

    Returns
    -------
    str_ : :obj:`str`
        A string with commas and/or ands separating th elements from ``lst``.

    """
    if len(lst) == 1:
        str_ = lst[0]
    elif len(lst) == 2:
        str_ = ' and '.join(lst)
    elif len(lst) > 2:
        str_ = ', '.join(lst[:-1])
        str_ += ', and {0}'.format(lst[-1])
    else:
        raise ValueError('List of length 0 provided.')
    return str_

def returned(n):
	"""Generate a random walk and return True if the walker has returned to
	the origin after taking `n` steps.
	"""
	## `takei` yield lazily so we can short-circuit and avoid computing the rest of the walk
	for pos in randwalk() >> drop(1) >> takei(xrange(n-1)):
		if pos == Origin:
			return True
	return False

def _prm_get_longest_stringsize(string_list):
        """ Returns the longest string size for a string entry across data."""
        maxlength = 1

        for stringar in string_list:
            if isinstance(stringar, np.ndarray):
                if stringar.ndim > 0:
                    for string in stringar.ravel():
                        maxlength = max(len(string), maxlength)
                else:
                    maxlength = max(len(stringar.tolist()), maxlength)
            else:
                maxlength = max(len(stringar), maxlength)

        # Make the string Col longer than needed in order to allow later on slightly larger strings
        return int(maxlength * 1.5)

def most_significant_bit(lst: np.ndarray) -> int:
    """
    A helper function that finds the position of the most significant bit in a 1darray of 1s and 0s,
    i.e. the first position where a 1 appears, reading left to right.

    :param lst: a 1d array of 0s and 1s with at least one 1
    :return: the first position in lst that a 1 appears
    """
    return np.argwhere(np.asarray(lst) == 1)[0][0]

def fib(n):
    """Fibonacci example function

    Args:
      n (int): integer

    Returns:
      int: n-th Fibonacci number
    """
    assert n > 0
    a, b = 1, 1
    for i in range(n - 1):
        a, b = b, a + b
    return a

def enum_mark_last(iterable, start=0):
    """
    Returns a generator over iterable that tells whether the current item is the last one.
    Usage:
        >>> iterable = range(10)
        >>> for index, is_last, item in enum_mark_last(iterable):
        >>>     print(index, item, end='\n' if is_last else ', ')
    """
    it = iter(iterable)
    count = start
    try:
        last = next(it)
    except StopIteration:
        return
    for val in it:
        yield count, False, last
        last = val
        count += 1
    yield count, True, last

def most_frequent(lst):
    """
    Returns the item that appears most frequently in the given list.
    """
    lst = lst[:]
    highest_freq = 0
    most_freq = None

    for val in unique(lst):
        if lst.count(val) > highest_freq:
            most_freq = val
            highest_freq = lst.count(val)
            
    return most_freq

def command(self, cmd, *args):
        """
        Sends a command and an (optional) sequence of arguments through to the
        delegated serial interface. Note that the arguments are passed through
        as data.
        """
        self._serial_interface.command(cmd)
        if len(args) > 0:
            self._serial_interface.data(list(args))

def clean_map(obj: Mapping[Any, Any]) -> Mapping[Any, Any]:
    """
    Return a new copied dictionary without the keys with ``None`` values from
    the given Mapping object.
    """
    return {k: v for k, v in obj.items() if v is not None}

def is_sqlatype_integer(coltype: Union[TypeEngine, VisitableType]) -> bool:
    """
    Is the SQLAlchemy column type an integer type?
    """
    coltype = _coltype_to_typeengine(coltype)
    return isinstance(coltype, sqltypes.Integer)

def non_increasing(values):
    """True if values are not increasing."""
    return all(x >= y for x, y in zip(values, values[1:]))

def get_window_dim():
    """ gets the dimensions depending on python version and os"""
    version = sys.version_info

    if version >= (3, 3):
        return _size_36()
    if platform.system() == 'Windows':
        return _size_windows()
    return _size_27()

def sort_by_modified(files_or_folders: list) -> list:
    """
    Sort files or folders by modified time

    Args:
        files_or_folders: list of files or folders

    Returns:
        list
    """
    return sorted(files_or_folders, key=os.path.getmtime, reverse=True)

def get_period_last_3_months() -> str:
    """ Returns the last week as a period string """
    today = Datum()
    today.today()

    # start_date = today - timedelta(weeks=13)
    start_date = today.clone()
    start_date.subtract_months(3)

    period = get_period(start_date.date, today.date)
    return period

def supports_py3(project_name):
    """Check with PyPI if a project supports Python 3."""
    log = logging.getLogger("ciu")
    log.info("Checking {} ...".format(project_name))
    request = requests.get("https://pypi.org/pypi/{}/json".format(project_name))
    if request.status_code >= 400:
        log = logging.getLogger("ciu")
        log.warning("problem fetching {}, assuming ported ({})".format(
                        project_name, request.status_code))
        return True
    response = request.json()
    return any(c.startswith("Programming Language :: Python :: 3")
               for c in response["info"]["classifiers"])

def to_0d_array(value: Any) -> np.ndarray:
    """Given a value, wrap it in a 0-D numpy.ndarray.
    """
    if np.isscalar(value) or (isinstance(value, np.ndarray) and
                              value.ndim == 0):
        return np.array(value)
    else:
        return to_0d_object_array(value)

def issubset(self, other):
        """
        Report whether another set contains this set.

        Example:
            >>> OrderedSet([1, 2, 3]).issubset({1, 2})
            False
            >>> OrderedSet([1, 2, 3]).issubset({1, 2, 3, 4})
            True
            >>> OrderedSet([1, 2, 3]).issubset({1, 4, 3, 5})
            False
        """
        if len(self) > len(other):  # Fast check for obvious cases
            return False
        return all(item in other for item in self)

def repl_complete(text: str, state: int) -> Optional[str]:
    """Completer function for Python's readline/libedit implementation."""
    # Can't complete Keywords, Numerals
    if __NOT_COMPLETEABLE.match(text):
        return None
    elif text.startswith(":"):
        completions = kw.complete(text)
    else:
        ns = get_current_ns()
        completions = ns.complete(text)

    return list(completions)[state] if completions is not None else None

def fprint(expr, print_ascii=False):
    r"""This function chooses whether to use ascii characters to represent
    a symbolic expression in the notebook or to use sympy's pprint.

    >>> from sympy import cos
    >>> omega=Symbol("omega")
    >>> fprint(cos(omega),print_ascii=True)
    cos(omega)


    """
    if print_ascii:
        pprint(expr, use_unicode=False, num_columns=120)
    else:
        return expr

def _check_whitespace(string):
    """
    Make sure thre is no whitespace in the given string. Will raise a
    ValueError if whitespace is detected
    """
    if string.count(' ') + string.count('\t') + string.count('\n') > 0:
        raise ValueError(INSTRUCTION_HAS_WHITESPACE)

def get_property_as_float(self, name: str) -> float:
        """Return the value of a float property.

        :return: The property value (float).

        Raises exception if property with name doesn't exist.

        .. versionadded:: 1.0

        Scriptable: Yes
        """
        return float(self.__instrument.get_property(name))

def get_triangles(graph: DiGraph) -> SetOfNodeTriples:
    """Get a set of triples representing the 3-cycles from a directional graph.

    Each 3-cycle is returned once, with nodes in sorted order.
    """
    return {
        tuple(sorted([a, b, c], key=str))
        for a, b in graph.edges()
        for c in graph.successors(b)
        if graph.has_edge(c, a)
    }

def valid_file(path: str) -> bool:
    """
    Verifies that a string path actually exists and is a file

    :param path: The path to verify
    :return: **True** if path exist and is a file
    """
    path = Path(path).expanduser()
    log.debug("checking if %s is a valid file", path)
    return path.exists() and path.is_file()

def rmglob(pattern: str) -> None:
    """
    Deletes all files whose filename matches the glob ``pattern`` (via
    :func:`glob.glob`).
    """
    for f in glob.glob(pattern):
        os.remove(f)

def is_not_null(df: DataFrame, col_name: str) -> bool:
    """
    Return ``True`` if the given DataFrame has a column of the given
    name (string), and there exists at least one non-NaN value in that
    column; return ``False`` otherwise.
    """
    if (
        isinstance(df, pd.DataFrame)
        and col_name in df.columns
        and df[col_name].notnull().any()
    ):
        return True
    else:
        return False

def is_line_in_file(filename: str, line: str) -> bool:
    """
    Detects whether a line is present within a file.

    Args:
        filename: file to check
        line: line to search for (as an exact match)
    """
    assert "\n" not in line
    with open(filename, "r") as file:
        for fileline in file:
            if fileline == line:
                return True
        return False

def after_third_friday(day=None):
    """ check if day is after month's 3rd friday """
    day = day if day is not None else datetime.datetime.now()
    now = day.replace(day=1, hour=16, minute=0, second=0, microsecond=0)
    now += relativedelta.relativedelta(weeks=2, weekday=relativedelta.FR)
    return day > now

def timeit (func, log, limit):
    """Print execution time of the function. For quick'n'dirty profiling."""

    def newfunc (*args, **kwargs):
        """Execute function and print execution time."""
        t = time.time()
        res = func(*args, **kwargs)
        duration = time.time() - t
        if duration > limit:
            print(func.__name__, "took %0.2f seconds" % duration, file=log)
            print(args, file=log)
            print(kwargs, file=log)
        return res
    return update_func_meta(newfunc, func)

def bfx(value, msb, lsb):
    """! @brief Extract a value from a bitfield."""
    mask = bitmask((msb, lsb))
    return (value & mask) >> lsb

def find_first_in_list(txt: str, str_list: [str]) -> int:  # type: ignore
    """
    Returns the index of the earliest occurence of an item from a list in a string

    Ex: find_first_in_list('foobar', ['bar', 'fin']) -> 3
    """
    start = len(txt) + 1
    for item in str_list:
        if start > txt.find(item) > -1:
            start = txt.find(item)
    return start if len(txt) + 1 > start > -1 else -1

def assert_in(first, second, msg_fmt="{msg}"):
    """Fail if first is not in collection second.

    >>> assert_in("foo", [4, "foo", {}])
    >>> assert_in("bar", [4, "foo", {}])
    Traceback (most recent call last):
        ...
    AssertionError: 'bar' not in [4, 'foo', {}]

    The following msg_fmt arguments are supported:
    * msg - the default error message
    * first - the element looked for
    * second - the container looked in
    """

    if first not in second:
        msg = "{!r} not in {!r}".format(first, second)
        fail(msg_fmt.format(msg=msg, first=first, second=second))

def get_case_insensitive_dict_key(d: Dict, k: str) -> Optional[str]:
    """
    Within the dictionary ``d``, find a key that matches (in case-insensitive
    fashion) the key ``k``, and return it (or ``None`` if there isn't one).
    """
    for key in d.keys():
        if k.lower() == key.lower():
            return key
    return None

def _check_whitespace(string):
    """
    Make sure thre is no whitespace in the given string. Will raise a
    ValueError if whitespace is detected
    """
    if string.count(' ') + string.count('\t') + string.count('\n') > 0:
        raise ValueError(INSTRUCTION_HAS_WHITESPACE)

def maybe_infer_dtype_type(element):
    """Try to infer an object's dtype, for use in arithmetic ops

    Uses `element.dtype` if that's available.
    Objects implementing the iterator protocol are cast to a NumPy array,
    and from there the array's type is used.

    Parameters
    ----------
    element : object
        Possibly has a `.dtype` attribute, and possibly the iterator
        protocol.

    Returns
    -------
    tipo : type

    Examples
    --------
    >>> from collections import namedtuple
    >>> Foo = namedtuple("Foo", "dtype")
    >>> maybe_infer_dtype_type(Foo(np.dtype("i8")))
    numpy.int64
    """
    tipo = None
    if hasattr(element, 'dtype'):
        tipo = element.dtype
    elif is_list_like(element):
        element = np.asarray(element)
        tipo = element.dtype
    return tipo

def do_quit(self, _: argparse.Namespace) -> bool:
        """Exit this application"""
        self._should_quit = True
        return self._STOP_AND_EXIT

def flatten_list(l: List[list]) -> list:
    """ takes a list of lists, l and returns a flat list
    """
    return [v for inner_l in l for v in inner_l]

def dictfetchall(cursor: Cursor) -> List[Dict[str, Any]]:
    """
    Return all rows from a cursor as a list of :class:`OrderedDict` objects.

    Args:
        cursor: the cursor

    Returns:
        a list (one item per row) of :class:`OrderedDict` objects whose key are
        column names and whose values are the row values
    """
    columns = get_fieldnames_from_cursor(cursor)
    return [
        OrderedDict(zip(columns, row))
        for row in cursor.fetchall()
    ]

def count(args):
    """ count occurences in a list of lists
    >>> count([['a','b'],['a']])
    defaultdict(int, {'a' : 2, 'b' : 1})
    """
    counts = defaultdict(int)
    for arg in args:
        for item in arg:
            counts[item] = counts[item] + 1
    return counts

def get_domain(url):
    """
    Get domain part of an url.

    For example: https://www.python.org/doc/ -> https://www.python.org
    """
    parse_result = urlparse(url)
    domain = "{schema}://{netloc}".format(
        schema=parse_result.scheme, netloc=parse_result.netloc)
    return domain

def _request(self, method: str, endpoint: str, params: dict = None, data: dict = None, headers: dict = None) -> dict:
        """HTTP request method of interface implementation."""

def is_natural(x):
    """A non-negative integer."""
    try:
        is_integer = int(x) == x
    except (TypeError, ValueError):
        return False
    return is_integer and x >= 0

def _skip_section(self):
        """Skip a section"""
        self._last = self._f.readline()
        while len(self._last) > 0 and len(self._last[0].strip()) == 0:
            self._last = self._f.readline()

def read_text_from_file(path: str) -> str:
    """ Reads text file contents """
    with open(path) as text_file:
        content = text_file.read()

    return content

def encode_list(key, list_):
    # type: (str, Iterable) -> Dict[str, str]
    """
    Converts a list into a space-separated string and puts it in a dictionary

    :param key: Dictionary key to store the list
    :param list_: A list of objects
    :return: A dictionary key->string or an empty dictionary
    """
    if not list_:
        return {}
    return {key: " ".join(str(i) for i in list_)}

def backspace(self):
        """
        Moves the cursor one place to the left, erasing the character at the
        current position. Cannot move beyond column zero, nor onto the
        previous line.
        """
        if self._cx + self._cw >= 0:
            self.erase()
            self._cx -= self._cw

        self.flush()

def get_valid_filename(s):
    """
    Shamelessly taken from Django.
    https://github.com/django/django/blob/master/django/utils/text.py

    Return the given string converted to a string that can be used for a clean
    filename. Remove leading and trailing spaces; convert other spaces to
    underscores; and remove anything that is not an alphanumeric, dash,
    underscore, or dot.
    >>> get_valid_filename("john's portrait in 2004.jpg")
    'johns_portrait_in_2004.jpg'
    """
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)

def timeit(func, *args, **kwargs):
    """
    Time execution of function. Returns (res, seconds).

    >>> res, timing = timeit(time.sleep, 1)
    """
    start_time = time.time()
    res = func(*args, **kwargs)
    timing = time.time() - start_time
    return res, timing

def strings_to_integers(strings: Iterable[str]) -> Iterable[int]:
    """
    Convert a list of strings to a list of integers.

    :param strings: a list of string
    :return: a list of converted integers

    .. doctest::

        >>> strings_to_integers(['1', '1.0', '-0.2'])
        [1, 1, 0]
    """
    return strings_to_(strings, lambda x: int(float(x)))

def _parse_tuple_string(argument):
        """ Return a tuple from parsing 'a,b,c,d' -> (a,b,c,d) """
        if isinstance(argument, str):
            return tuple(int(p.strip()) for p in argument.split(','))
        return argument

def snake_to_camel(s: str) -> str:
    """Convert string from snake case to camel case."""

    fragments = s.split('_')

    return fragments[0] + ''.join(x.title() for x in fragments[1:])

def array2string(arr: numpy.ndarray) -> str:
        """Format numpy array as a string."""
        shape = str(arr.shape)[1:-1]
        if shape.endswith(","):
            shape = shape[:-1]
        return numpy.array2string(arr, threshold=11) + "%s[%s]" % (arr.dtype, shape)

def lowercase_chars(string: any) -> str:
        """Return all (and only) the lowercase chars in the given string."""
        return ''.join([c if c.islower() else '' for c in str(string)])

def from_file(file_path) -> dict:
        """ Load JSON file """
        with io.open(file_path, 'r', encoding='utf-8') as json_stream:
            return Json.parse(json_stream, True)

def is_not_null(df: DataFrame, col_name: str) -> bool:
    """
    Return ``True`` if the given DataFrame has a column of the given
    name (string), and there exists at least one non-NaN value in that
    column; return ``False`` otherwise.
    """
    if (
        isinstance(df, pd.DataFrame)
        and col_name in df.columns
        and df[col_name].notnull().any()
    ):
        return True
    else:
        return False

def _rindex(mylist: Sequence[T], x: T) -> int:
    """Index of the last occurrence of x in the sequence."""
    return len(mylist) - mylist[::-1].index(x) - 1

def bfx(value, msb, lsb):
    """! @brief Extract a value from a bitfield."""
    mask = bitmask((msb, lsb))
    return (value & mask) >> lsb

def try_cast_int(s):
    """(str) -> int
    All the digits in a given string are concatenated and converted into a single number.
    """
    try:
        temp = re.findall('\d', str(s))
        temp = ''.join(temp)
        return int(temp)
    except:
        return s

def fetchallfirstvalues(self, sql: str, *args) -> List[Any]:
        """Executes SQL; returns list of first values of each row."""
        rows = self.fetchall(sql, *args)
        return [row[0] for row in rows]

def snake_to_camel(s: str) -> str:
    """Convert string from snake case to camel case."""

    fragments = s.split('_')

    return fragments[0] + ''.join(x.title() for x in fragments[1:])

def to_json(self) -> Mapping:
        """Return the properties of this :class:`Sample` as JSON serializable.

        """
        return {str(x): str(y) for x, y in self.items()}

def uppercase_chars(string: any) -> str:
        """Return all (and only) the uppercase chars in the given string."""
        return ''.join([c if c.isupper() else '' for c in str(string)])

def fib(n):
    """Fibonacci example function

    Args:
      n (int): integer

    Returns:
      int: n-th Fibonacci number
    """
    assert n > 0
    a, b = 1, 1
    for i in range(n - 1):
        a, b = b, a + b
    return a

def read(self, start_position: int, size: int) -> memoryview:
        """
        Return a view into the memory
        """
        return memoryview(self._bytes)[start_position:start_position + size]

def s3_get(url: str, temp_file: IO) -> None:
    """Pull a file directly from S3."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)

def remove_leading(needle, haystack):
    """Remove leading needle string (if exists).

    >>> remove_leading('Test', 'TestThisAndThat')
    'ThisAndThat'
    >>> remove_leading('Test', 'ArbitraryName')
    'ArbitraryName'
    """
    if haystack[:len(needle)] == needle:
        return haystack[len(needle):]
    return haystack

def csv_to_numpy(string_like, dtype=None):  # type: (str) -> np.array
    """Convert a CSV object to a numpy array.

    Args:
        string_like (str): CSV string.
        dtype (dtype, optional):  Data type of the resulting array. If None, the dtypes will be determined by the
                                        contents of each column, individually. This argument can only be used to
                                        'upcast' the array.  For downcasting, use the .astype(t) method.
    Returns:
        (np.array): numpy array
    """
    stream = StringIO(string_like)
    return np.genfromtxt(stream, dtype=dtype, delimiter=',')

def inner(tensor0: BKTensor, tensor1: BKTensor) -> BKTensor:
    """Return the inner product between two tensors"""
    # Note: Relying on fact that vdot flattens arrays
    return np.vdot(tensor0, tensor1)

def memory_full():
    """Check if the memory is too full for further caching."""
    current_process = psutil.Process(os.getpid())
    return (current_process.memory_percent() >
            config.MAXIMUM_CACHE_MEMORY_PERCENTAGE)

def _kbhit_unix() -> bool:
    """
    Under UNIX: is a keystroke available?
    """
    dr, dw, de = select.select([sys.stdin], [], [], 0)
    return dr != []

def _rindex(mylist: Sequence[T], x: T) -> int:
    """Index of the last occurrence of x in the sequence."""
    return len(mylist) - mylist[::-1].index(x) - 1

def debugTreePrint(node,pfx="->"):
  """Purely a debugging aid: Ascii-art picture of a tree descended from node"""
  print pfx,node.item
  for c in node.children:
    debugTreePrint(c,"  "+pfx)

def get_from_gnucash26_date(date_str: str) -> date:
    """ Creates a datetime from GnuCash 2.6 date string """
    date_format = "%Y%m%d"
    result = datetime.strptime(date_str, date_format).date()
    return result

def _check_whitespace(string):
    """
    Make sure thre is no whitespace in the given string. Will raise a
    ValueError if whitespace is detected
    """
    if string.count(' ') + string.count('\t') + string.count('\n') > 0:
        raise ValueError(INSTRUCTION_HAS_WHITESPACE)

def long_substr(data):
    """Return the longest common substring in a list of strings.
    
    Credit: http://stackoverflow.com/questions/2892931/longest-common-substring-from-more-than-two-strings-python
    """
    substr = ''
    if len(data) > 1 and len(data[0]) > 0:
        for i in range(len(data[0])):
            for j in range(len(data[0])-i+1):
                if j > len(substr) and all(data[0][i:i+j] in x for x in data):
                    substr = data[0][i:i+j]
    elif len(data) == 1:
        substr = data[0]
    return substr

async def async_run(self) -> None:
        """
        Asynchronously run the worker, does not close connections. Useful when testing.
        """
        self.main_task = self.loop.create_task(self.main())
        await self.main_task

async def async_run(self) -> None:
        """
        Asynchronously run the worker, does not close connections. Useful when testing.
        """
        self.main_task = self.loop.create_task(self.main())
        await self.main_task

def _duplicates(list_):
    """Return dict mapping item -> indices."""
    item_indices = {}
    for i, item in enumerate(list_):
        try:
            item_indices[item].append(i)
        except KeyError:  # First time seen
            item_indices[item] = [i]
    return item_indices

def get_column_names(engine: Engine, tablename: str) -> List[str]:
    """
    Get all the database column names for the specified table.
    """
    return [info.name for info in gen_columns_info(engine, tablename)]

def long_substr(data):
    """Return the longest common substring in a list of strings.
    
    Credit: http://stackoverflow.com/questions/2892931/longest-common-substring-from-more-than-two-strings-python
    """
    substr = ''
    if len(data) > 1 and len(data[0]) > 0:
        for i in range(len(data[0])):
            for j in range(len(data[0])-i+1):
                if j > len(substr) and all(data[0][i:i+j] in x for x in data):
                    substr = data[0][i:i+j]
    elif len(data) == 1:
        substr = data[0]
    return substr

def _kbhit_unix() -> bool:
    """
    Under UNIX: is a keystroke available?
    """
    dr, dw, de = select.select([sys.stdin], [], [], 0)
    return dr != []

def get_default_bucket_key(buckets: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Returns the default bucket from a list of buckets, i.e. the largest bucket.

    :param buckets: List of buckets.
    :return: The largest bucket in the list.
    """
    return max(buckets)

def bfx(value, msb, lsb):
    """! @brief Extract a value from a bitfield."""
    mask = bitmask((msb, lsb))
    return (value & mask) >> lsb

def remove_empty_text(utterances: List[Utterance]) -> List[Utterance]:
    """Remove empty utterances from a list of utterances
    Args:
        utterances: The list of utterance we are processing
    """
    return [utter for utter in utterances if utter.text.strip() != ""]

def dotproduct(X, Y):
    """Return the sum of the element-wise product of vectors x and y.
    >>> dotproduct([1, 2, 3], [1000, 100, 10])
    1230
    """
    return sum([x * y for x, y in zip(X, Y)])

def __remove_trailing_zeros(self, collection):
        """Removes trailing zeroes from indexable collection of numbers"""
        index = len(collection) - 1
        while index >= 0 and collection[index] == 0:
            index -= 1

        return collection[:index + 1]

def indexes_equal(a: Index, b: Index) -> bool:
    """
    Are two indexes equal? Checks by comparing ``str()`` versions of them.
    (AM UNSURE IF THIS IS ENOUGH.)
    """
    return str(a) == str(b)

def read_set_from_file(filename: str) -> Set[str]:
    """
    Extract a de-duped collection (set) of text from a file.
    Expected file format is one item per line.
    """
    collection = set()
    with open(filename, 'r') as file_:
        for line in file_:
            collection.add(line.rstrip())
    return collection

def most_frequent(lst):
    """
    Returns the item that appears most frequently in the given list.
    """
    lst = lst[:]
    highest_freq = 0
    most_freq = None

    for val in unique(lst):
        if lst.count(val) > highest_freq:
            most_freq = val
            highest_freq = lst.count(val)
            
    return most_freq

def availability_pdf() -> bool:
    """
    Is a PDF-to-text tool available?
    """
    pdftotext = tools['pdftotext']
    if pdftotext:
        return True
    elif pdfminer:
        log.warning("PDF conversion: pdftotext missing; "
                    "using pdfminer (less efficient)")
        return True
    else:
        return False

def snake_to_camel(s: str) -> str:
    """Convert string from snake case to camel case."""

    fragments = s.split('_')

    return fragments[0] + ''.join(x.title() for x in fragments[1:])

def truncate(value: Decimal, n_digits: int) -> Decimal:
    """Truncates a value to a number of decimals places"""
    return Decimal(math.trunc(value * (10 ** n_digits))) / (10 ** n_digits)

def samefile(a: str, b: str) -> bool:
    """Check if two pathes represent the same file."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normpath(a) == os.path.normpath(b)

def file_exists(fname):
    """Check if a file exists and is non-empty.
    """
    try:
        return fname and os.path.exists(fname) and os.path.getsize(fname) > 0
    except OSError:
        return False

def CheckDisjointCalendars(self):
    """Check whether any old service periods intersect with any new ones.

    This is a rather coarse check based on
    transitfeed.SevicePeriod.GetDateRange.

    Returns:
      True if the calendars are disjoint or False if not.
    """
    # TODO: Do an exact check here.

    a_service_periods = self.feed_merger.a_schedule.GetServicePeriodList()
    b_service_periods = self.feed_merger.b_schedule.GetServicePeriodList()

    for a_service_period in a_service_periods:
      a_start, a_end = a_service_period.GetDateRange()
      for b_service_period in b_service_periods:
        b_start, b_end = b_service_period.GetDateRange()
        overlap_start = max(a_start, b_start)
        overlap_end = min(a_end, b_end)
        if overlap_end >= overlap_start:
          return False
    return True

def top(self, topn=10):
        """
        Get a list of the top ``topn`` features in this :class:`.Feature`\.

        Examples
        --------

        .. code-block:: python

        >>> myFeature = Feature([('the', 2), ('pine', 1), ('trapezoid', 5)])
        >>> myFeature.top(1)
        [('trapezoid', 5)]

        Parameters
        ----------
        topn : int

        Returns
        -------
        list
        """
        return [self[i] for i in argsort(list(zip(*self))[1])[::-1][:topn]]

def __gt__(self, other):
        """Test for greater than."""
        if isinstance(other, Address):
            return str(self) > str(other)
        raise TypeError

def inject_nulls(data: Mapping, field_names) -> dict:
    """Insert None as value for missing fields."""

    record = dict()

    for field in field_names:
        record[field] = data.get(field, None)

    return record

def connect_to_database_odbc_access(self,
                                        dsn: str,
                                        autocommit: bool = True) -> None:
        """Connects to an Access database via ODBC, with the DSN
        prespecified."""
        self.connect(engine=ENGINE_ACCESS, interface=INTERFACE_ODBC,
                     dsn=dsn, autocommit=autocommit)

def grep(pattern, filename):
    """Very simple grep that returns the first matching line in a file.
    String matching only, does not do REs as currently implemented.
    """
    try:
        # for line in file
        # if line matches pattern:
        #    return line
        return next((L for L in open(filename) if L.find(pattern) >= 0))
    except StopIteration:
        return ''

def recClearTag(element):
    """Applies maspy.xml.clearTag() to the tag attribute of the "element" and
    recursively to all child elements.

    :param element: an :instance:`xml.etree.Element`
    """
    children = element.getchildren()
    if len(children) > 0:
        for child in children:
            recClearTag(child)
    element.tag = clearTag(element.tag)

def PrintIndented(self, file, ident, code):
        """Takes an array, add indentation to each entry and prints it."""
        for entry in code:
            print >>file, '%s%s' % (ident, entry)

def pack_bits( longbits ):
    """Crunch a 64-bit int (8 bool bytes) into a bitfield."""
    byte = longbits & (0x0101010101010101)
    byte = (byte | (byte>>7)) & (0x0003000300030003)
    byte = (byte | (byte>>14)) & (0x0000000f0000000f)
    byte = (byte | (byte>>28)) & (0x00000000000000ff)
    return byte

def full(self):
        """Return ``True`` if the queue is full, ``False``
        otherwise (not reliable!).

        Only applicable if :attr:`maxsize` is set.

        """
        return self.maxsize and len(self.list) >= self.maxsize or False

def kernel(self, spread=1):
        """ This will return whatever kind of kernel we want to use.
            Must have signature (ndarray size NxM, ndarray size 1xM) -> ndarray size Nx1
        """
        # TODO: use self.kernel_type to choose function

        def gaussian(data, pixel):
            return mvn.pdf(data, mean=pixel, cov=spread)

        return gaussian

def _reshuffle(mat, shape):
    """Reshuffle the indicies of a bipartite matrix A[ij,kl] -> A[lj,ki]."""
    return np.reshape(
        np.transpose(np.reshape(mat, shape), (3, 1, 2, 0)),
        (shape[3] * shape[1], shape[0] * shape[2]))

def _my_hash(arg_list):
    # type: (List[Any]) -> int
    """Simple helper hash function"""
    res = 0
    for arg in arg_list:
        res = res * 31 + hash(arg)
    return res

def clean_column_names(df: DataFrame) -> DataFrame:
    """
    Strip the whitespace from all column names in the given DataFrame
    and return the result.
    """
    f = df.copy()
    f.columns = [col.strip() for col in f.columns]
    return f

def dictlist_wipe_key(dict_list: Iterable[Dict], key: str) -> None:
    """
    Process an iterable of dictionaries. For each dictionary ``d``, delete
    ``d[key]`` if it exists.
    """
    for d in dict_list:
        d.pop(key, None)

def remove_once(gset, elem):
    """Remove the element from a set, lists or dict.
    
        >>> L = ["Lucy"]; S = set(["Sky"]); D = { "Diamonds": True };
        >>> remove_once(L, "Lucy"); remove_once(S, "Sky"); remove_once(D, "Diamonds");
        >>> print L, S, D
        [] set([]) {}

    Returns the element if it was removed. Raises one of the exceptions in 
    :obj:`RemoveError` otherwise.
    """
    remove = getattr(gset, 'remove', None)
    if remove is not None: remove(elem)
    else: del gset[elem]
    return elem

def is_unicode(string):
    """Validates that the object itself is some kinda string"""
    str_type = str(type(string))

    if str_type.find('str') > 0 or str_type.find('unicode') > 0:
        return True

    return False

def ResetConsoleColor() -> bool:
    """
    Reset to the default text color on console window.
    Return bool, True if succeed otherwise False.
    """
    if sys.stdout:
        sys.stdout.flush()
    bool(ctypes.windll.kernel32.SetConsoleTextAttribute(_ConsoleOutputHandle, _DefaultConsoleColor))

def pairwise(iterable):
    """From itertools cookbook. [a, b, c, ...] -> (a, b), (b, c), ..."""
    first, second = tee(iterable)
    next(second, None)
    return zip(first, second)

def is_not_null(df: DataFrame, col_name: str) -> bool:
    """
    Return ``True`` if the given DataFrame has a column of the given
    name (string), and there exists at least one non-NaN value in that
    column; return ``False`` otherwise.
    """
    if (
        isinstance(df, pd.DataFrame)
        and col_name in df.columns
        and df[col_name].notnull().any()
    ):
        return True
    else:
        return False

def memory_usage():
    """return memory usage of python process in MB

    from
    http://fa.bianp.net/blog/2013/different-ways-to-get-memory-consumption-or-lessons-learned-from-memory_profiler/
    psutil is quicker

    >>> isinstance(memory_usage(),float)
    True

    """
    try:
        import psutil
        import os
    except ImportError:
        return _memory_usage_ps()

    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem

def de_duplicate(items):
    """Remove any duplicate item, preserving order

    >>> de_duplicate([1, 2, 1, 2])
    [1, 2]
    """
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return result

def rmglob(pattern: str) -> None:
    """
    Deletes all files whose filename matches the glob ``pattern`` (via
    :func:`glob.glob`).
    """
    for f in glob.glob(pattern):
        os.remove(f)

def inverted_dict(d):
    """Return a dict with swapped keys and values

    >>> inverted_dict({0: ('a', 'b'), 1: 'cd'}) == {'cd': 1, ('a', 'b'): 0}
    True
    """
    return dict((force_hashable(v), k) for (k, v) in viewitems(dict(d)))

def almost_hermitian(gate: Gate) -> bool:
    """Return true if gate tensor is (almost) Hermitian"""
    return np.allclose(asarray(gate.asoperator()),
                       asarray(gate.H.asoperator()))

def strings_to_integers(strings: Iterable[str]) -> Iterable[int]:
    """
    Convert a list of strings to a list of integers.

    :param strings: a list of string
    :return: a list of converted integers

    .. doctest::

        >>> strings_to_integers(['1', '1.0', '-0.2'])
        [1, 1, 0]
    """
    return strings_to_(strings, lambda x: int(float(x)))

def is_any_type_set(sett: Set[Type]) -> bool:
    """
    Helper method to check if a set of types is the {AnyObject} singleton

    :param sett:
    :return:
    """
    return len(sett) == 1 and is_any_type(min(sett))

def encode_list(key, list_):
    # type: (str, Iterable) -> Dict[str, str]
    """
    Converts a list into a space-separated string and puts it in a dictionary

    :param key: Dictionary key to store the list
    :param list_: A list of objects
    :return: A dictionary key->string or an empty dictionary
    """
    if not list_:
        return {}
    return {key: " ".join(str(i) for i in list_)}

def issubset(self, other):
        """
        Report whether another set contains this set.

        Example:
            >>> OrderedSet([1, 2, 3]).issubset({1, 2})
            False
            >>> OrderedSet([1, 2, 3]).issubset({1, 2, 3, 4})
            True
            >>> OrderedSet([1, 2, 3]).issubset({1, 4, 3, 5})
            False
        """
        if len(self) > len(other):  # Fast check for obvious cases
            return False
        return all(item in other for item in self)

def flatten_list(l: List[list]) -> list:
    """ takes a list of lists, l and returns a flat list
    """
    return [v for inner_l in l for v in inner_l]

def cli_run():
    """docstring for argparse"""
    parser = argparse.ArgumentParser(description='Stupidly simple code answers from StackOverflow')
    parser.add_argument('query', help="What's the problem ?", type=str, nargs='+')
    parser.add_argument('-t','--tags', help='semicolon separated tags -> python;lambda')
    args = parser.parse_args()
    main(args)

def to_bytes(data: Any) -> bytearray:
    """
    Convert anything to a ``bytearray``.
    
    See
    
    - http://stackoverflow.com/questions/7585435/best-way-to-convert-string-to-bytes-in-python-3
    - http://stackoverflow.com/questions/10459067/how-to-convert-my-bytearrayb-x9e-x18k-x9a-to-something-like-this-x9e-x1
    """  # noqa
    if isinstance(data, int):
        return bytearray([data])
    return bytearray(data, encoding='latin-1')

def getIndex(predicateFn: Callable[[T], bool], items: List[T]) -> int:
    """
    Finds the index of an item in list, which satisfies predicate
    :param predicateFn: predicate function to run on items of list
    :param items: list of tuples
    :return: first index for which predicate function returns True
    """
    try:
        return next(i for i, v in enumerate(items) if predicateFn(v))
    except StopIteration:
        return -1

def full(self):
        """Return ``True`` if the queue is full, ``False``
        otherwise (not reliable!).

        Only applicable if :attr:`maxsize` is set.

        """
        return self.maxsize and len(self.list) >= self.maxsize or False

def toHdlConversion(self, top, topName: str, saveTo: str) -> List[str]:
        """
        :param top: object which is represenation of design
        :param topName: name which should be used for ipcore
        :param saveTo: path of directory where generated files should be stored

        :return: list of file namens in correct compile order
        """
        raise NotImplementedError(
            "Implement this function for your type of your top module")

def find_first_in_list(txt: str, str_list: [str]) -> int:  # type: ignore
    """
    Returns the index of the earliest occurence of an item from a list in a string

    Ex: find_first_in_list('foobar', ['bar', 'fin']) -> 3
    """
    start = len(txt) + 1
    for item in str_list:
        if start > txt.find(item) > -1:
            start = txt.find(item)
    return start if len(txt) + 1 > start > -1 else -1

def default_parser() -> argparse.ArgumentParser:
    """Create a parser for CLI arguments and options."""
    parser = argparse.ArgumentParser(
        prog=CONSOLE_SCRIPT,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    build_parser(parser)
    return parser

def titleize(text):
    """Capitalizes all the words and replaces some characters in the string 
    to create a nicer looking title.
    """
    if len(text) == 0: # if empty string, return it
        return text
    else:
        text = text.lower() # lower all char
        # delete redundant empty space 
        chunks = [chunk[0].upper() + chunk[1:] for chunk in text.split(" ") if len(chunk) >= 1]
        return " ".join(chunks)

def _my_hash(arg_list):
    # type: (List[Any]) -> int
    """Simple helper hash function"""
    res = 0
    for arg in arg_list:
        res = res * 31 + hash(arg)
    return res

def _isint(string):
    """
    >>> _isint("123")
    True
    >>> _isint("123.45")
    False
    """
    return type(string) is int or \
           (isinstance(string, _binary_type) or isinstance(string, _text_type)) and \
           _isconvertible(int, string)

def is_line_in_file(filename: str, line: str) -> bool:
    """
    Detects whether a line is present within a file.

    Args:
        filename: file to check
        line: line to search for (as an exact match)
    """
    assert "\n" not in line
    with open(filename, "r") as file:
        for fileline in file:
            if fileline == line:
                return True
        return False

def set_int(bytearray_, byte_index, _int):
    """
    Set value in bytearray to int
    """
    # make sure were dealing with an int
    _int = int(_int)
    _bytes = struct.unpack('2B', struct.pack('>h', _int))
    bytearray_[byte_index:byte_index + 2] = _bytes
    return bytearray_

def _run_sync(self, method: Callable, *args, **kwargs) -> Any:
        """
        Utility method to run commands synchronously for testing.
        """
        if self.loop.is_running():
            raise RuntimeError("Event loop is already running.")

        if not self.is_connected:
            self.loop.run_until_complete(self.connect())

        task = asyncio.Task(method(*args, **kwargs), loop=self.loop)
        result = self.loop.run_until_complete(task)

        self.loop.run_until_complete(self.quit())

        return result

def multi_split(s, split):
    # type: (S, Iterable[S]) -> List[S]
    """Splits on multiple given separators."""
    for r in split:
        s = s.replace(r, "|")
    return [i for i in s.split("|") if len(i) > 0]

def get_period_last_3_months() -> str:
    """ Returns the last week as a period string """
    today = Datum()
    today.today()

    # start_date = today - timedelta(weeks=13)
    start_date = today.clone()
    start_date.subtract_months(3)

    period = get_period(start_date.date, today.date)
    return period

def get_column_names(engine: Engine, tablename: str) -> List[str]:
    """
    Get all the database column names for the specified table.
    """
    return [info.name for info in gen_columns_info(engine, tablename)]

def dtypes(self):
        """Returns all column names and their data types as a list.

        >>> df.dtypes
        [('age', 'int'), ('name', 'string')]
        """
        return [(str(f.name), f.dataType.simpleString()) for f in self.schema.fields]

def _duplicates(list_):
    """Return dict mapping item -> indices."""
    item_indices = {}
    for i, item in enumerate(list_):
        try:
            item_indices[item].append(i)
        except KeyError:  # First time seen
            item_indices[item] = [i]
    return item_indices

def infer_format(filename:str) -> str:
    """Return extension identifying format of given filename"""
    _, ext = os.path.splitext(filename)
    return ext

def read_text_from_file(path: str) -> str:
    """ Reads text file contents """
    with open(path) as text_file:
        content = text_file.read()

    return content

def remove_blank_lines(string):
    """ Removes all blank lines in @string

        -> #str without blank lines
    """
    return "\n".join(line
                     for line in string.split("\n")
                     if len(line.strip()))

def _(f, x):
    """
    filter for dict, note `f` should have signature: `f::key->value->bool`
    """
    return {k: v for k, v in x.items() if f(k, v)}

def _my_hash(arg_list):
    # type: (List[Any]) -> int
    """Simple helper hash function"""
    res = 0
    for arg in arg_list:
        res = res * 31 + hash(arg)
    return res

def _(f, x):
    """
    filter for dict, note `f` should have signature: `f::key->value->bool`
    """
    return {k: v for k, v in x.items() if f(k, v)}

async def cursor(self) -> Cursor:
        """Create an aiosqlite cursor wrapping a sqlite3 cursor object."""
        return Cursor(self, await self._execute(self._conn.cursor))

def is_integer(value: Any) -> bool:
    """Return true if a value is an integer number."""
    return (isinstance(value, int) and not isinstance(value, bool)) or (
        isinstance(value, float) and isfinite(value) and int(value) == value
    )

def infer_format(filename:str) -> str:
    """Return extension identifying format of given filename"""
    _, ext = os.path.splitext(filename)
    return ext

def decodebytes(input):
    """Decode base64 string to byte array."""
    py_version = sys.version_info[0]
    if py_version >= 3:
        return _decodebytes_py3(input)
    return _decodebytes_py2(input)

def fetchvalue(self, sql: str, *args) -> Optional[Any]:
        """Executes SQL; returns the first value of the first row, or None."""
        row = self.fetchone(sql, *args)
        if row is None:
            return None
        return row[0]

def string(value) -> str:
        """ string dict/object/value to JSON """
        return system_json.dumps(Json(value).safe_object(), ensure_ascii=False)

def is_sqlatype_integer(coltype: Union[TypeEngine, VisitableType]) -> bool:
    """
    Is the SQLAlchemy column type an integer type?
    """
    coltype = _coltype_to_typeengine(coltype)
    return isinstance(coltype, sqltypes.Integer)

def debugTreePrint(node,pfx="->"):
  """Purely a debugging aid: Ascii-art picture of a tree descended from node"""
  print pfx,node.item
  for c in node.children:
    debugTreePrint(c,"  "+pfx)

def warn_if_nans_exist(X):
    """Warn if nans exist in a numpy array."""
    null_count = count_rows_with_nans(X)
    total = len(X)
    percent = 100 * null_count / total

    if null_count > 0:
        warning_message = \
            'Warning! Found {} rows of {} ({:0.2f}%) with nan values. Only ' \
            'complete rows will be plotted.'.format(null_count, total, percent)
        warnings.warn(warning_message, DataWarning)

def recClearTag(element):
    """Applies maspy.xml.clearTag() to the tag attribute of the "element" and
    recursively to all child elements.

    :param element: an :instance:`xml.etree.Element`
    """
    children = element.getchildren()
    if len(children) > 0:
        for child in children:
            recClearTag(child)
    element.tag = clearTag(element.tag)

def dfromdm(dm):
    """Returns distance given distance modulus.
    """
    if np.size(dm)>1:
        dm = np.atleast_1d(dm)
    return 10**(1+dm/5)

def get_window_dim():
    """ gets the dimensions depending on python version and os"""
    version = sys.version_info

    if version >= (3, 3):
        return _size_36()
    if platform.system() == 'Windows':
        return _size_windows()
    return _size_27()

def get_last_weekday_in_month(year, month, weekday):
        """Get the last weekday in a given month. e.g:

        >>> # the last monday in Jan 2013
        >>> Calendar.get_last_weekday_in_month(2013, 1, MON)
        datetime.date(2013, 1, 28)
        """
        day = date(year, month, monthrange(year, month)[1])
        while True:
            if day.weekday() == weekday:
                break
            day = day - timedelta(days=1)
        return day

def get_window_dim():
    """ gets the dimensions depending on python version and os"""
    version = sys.version_info

    if version >= (3, 3):
        return _size_36()
    if platform.system() == 'Windows':
        return _size_windows()
    return _size_27()

def str_to_time(time_str: str) -> datetime.datetime:
    """
    Convert human readable string to datetime.datetime.
    """
    pieces: Any = [int(piece) for piece in time_str.split('-')]
    return datetime.datetime(*pieces)

def from_file(file_path) -> dict:
        """ Load JSON file """
        with io.open(file_path, 'r', encoding='utf-8') as json_stream:
            return Json.parse(json_stream, True)

def hsv2rgb_spectrum(hsv):
    """Generates RGB values from HSV values in line with a typical light
    spectrum."""
    h, s, v = hsv
    return hsv2rgb_raw(((h * 192) >> 8, s, v))

def dotproduct(X, Y):
    """Return the sum of the element-wise product of vectors x and y.
    >>> dotproduct([1, 2, 3], [1000, 100, 10])
    1230
    """
    return sum([x * y for x, y in zip(X, Y)])

def long_substr(data):
    """Return the longest common substring in a list of strings.
    
    Credit: http://stackoverflow.com/questions/2892931/longest-common-substring-from-more-than-two-strings-python
    """
    substr = ''
    if len(data) > 1 and len(data[0]) > 0:
        for i in range(len(data[0])):
            for j in range(len(data[0])-i+1):
                if j > len(substr) and all(data[0][i:i+j] in x for x in data):
                    substr = data[0][i:i+j]
    elif len(data) == 1:
        substr = data[0]
    return substr

def last_location_of_minimum(x):
    """
    Returns the last location of the minimal value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    return 1.0 - np.argmin(x[::-1]) / len(x) if len(x) > 0 else np.NaN

def argmax(self, rows: List[Row], column: ComparableColumn) -> List[Row]:
        """
        Takes a list of rows and a column name and returns a list containing a single row (dict from
        columns to cells) that has the maximum numerical value in the given column. We return a list
        instead of a single dict to be consistent with the return type of ``select`` and
        ``all_rows``.
        """
        if not rows:
            return []
        value_row_pairs = [(row.values[column.name], row) for row in rows]
        if not value_row_pairs:
            return []
        # Returns a list containing the row with the max cell value.
        return [sorted(value_row_pairs, key=lambda x: x[0], reverse=True)[0][1]]

def year(date):
    """ Returns the year.

    :param date:
        The string date with this format %m/%d/%Y
    :type date:
        String

    :returns:
        int

    :example:
        >>> year('05/1/2015')
        2015
    """
    try:
        fmt = '%m/%d/%Y'
        return datetime.strptime(date, fmt).timetuple().tm_year
    except ValueError:
        return 0

def add_colons(s):
    """Add colons after every second digit.

    This function is used in functions to prettify serials.

    >>> add_colons('teststring')
    'te:st:st:ri:ng'
    """
    return ':'.join([s[i:i + 2] for i in range(0, len(s), 2)])

def lint(fmt='colorized'):
    """Run verbose PyLint on source. Optionally specify fmt=html for HTML output."""
    if fmt == 'html':
        outfile = 'pylint_report.html'
        local('pylint -f %s davies > %s || true' % (fmt, outfile))
        local('open %s' % outfile)
    else:
        local('pylint -f %s davies || true' % fmt)

def snake_to_camel(s: str) -> str:
    """Convert string from snake case to camel case."""

    fragments = s.split('_')

    return fragments[0] + ''.join(x.title() for x in fragments[1:])

def strtobytes(input, encoding):
    """Take a str and transform it into a byte array."""
    py_version = sys.version_info[0]
    if py_version >= 3:
        return _strtobytes_py3(input, encoding)
    return _strtobytes_py2(input, encoding)

def spanning_tree_count(graph: nx.Graph) -> int:
    """Return the number of unique spanning trees of a graph, using
    Kirchhoff's matrix tree theorem.
    """
    laplacian = nx.laplacian_matrix(graph).toarray()
    comatrix = laplacian[:-1, :-1]
    det = np.linalg.det(comatrix)
    count = int(round(det))
    return count

def dotproduct(X, Y):
    """Return the sum of the element-wise product of vectors x and y.
    >>> dotproduct([1, 2, 3], [1000, 100, 10])
    1230
    """
    return sum([x * y for x, y in zip(X, Y)])

def maybe_infer_dtype_type(element):
    """Try to infer an object's dtype, for use in arithmetic ops

    Uses `element.dtype` if that's available.
    Objects implementing the iterator protocol are cast to a NumPy array,
    and from there the array's type is used.

    Parameters
    ----------
    element : object
        Possibly has a `.dtype` attribute, and possibly the iterator
        protocol.

    Returns
    -------
    tipo : type

    Examples
    --------
    >>> from collections import namedtuple
    >>> Foo = namedtuple("Foo", "dtype")
    >>> maybe_infer_dtype_type(Foo(np.dtype("i8")))
    numpy.int64
    """
    tipo = None
    if hasattr(element, 'dtype'):
        tipo = element.dtype
    elif is_list_like(element):
        element = np.asarray(element)
        tipo = element.dtype
    return tipo

def split(text: str) -> List[str]:
    """Split a text into a list of tokens.

    :param text: the text to split
    :return: tokens
    """
    return [word for word in SEPARATOR.split(text) if word.strip(' \t')]

def read32(bytestream):
  """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def dict_of_sets_add(dictionary, key, value):
    # type: (DictUpperBound, Any, Any) -> None
    """Add value to a set in a dictionary by key

    Args:
        dictionary (DictUpperBound): Dictionary to which to add values
        key (Any): Key within dictionary
        value (Any): Value to add to set in dictionary

    Returns:
        None

    """
    set_objs = dictionary.get(key, set())
    set_objs.add(value)
    dictionary[key] = set_objs

def butlast(iterable):
    """Yield all items from ``iterable`` except the last one.

    >>> list(butlast(['spam', 'eggs', 'ham']))
    ['spam', 'eggs']

    >>> list(butlast(['spam']))
    []

    >>> list(butlast([]))
    []
    """
    iterable = iter(iterable)
    try:
        first = next(iterable)
    except StopIteration:
        return
    for second in iterable:
        yield first
        first = second

def astensor(array: TensorLike) -> BKTensor:
    """Covert numpy array to tensorflow tensor"""
    tensor = tf.convert_to_tensor(value=array, dtype=CTYPE)
    return tensor

def try_cast_int(s):
    """(str) -> int
    All the digits in a given string are concatenated and converted into a single number.
    """
    try:
        temp = re.findall('\d', str(s))
        temp = ''.join(temp)
        return int(temp)
    except:
        return s

def most_significant_bit(lst: np.ndarray) -> int:
    """
    A helper function that finds the position of the most significant bit in a 1darray of 1s and 0s,
    i.e. the first position where a 1 appears, reading left to right.

    :param lst: a 1d array of 0s and 1s with at least one 1
    :return: the first position in lst that a 1 appears
    """
    return np.argwhere(np.asarray(lst) == 1)[0][0]

def define_struct(defn):
    """
    Register a struct definition globally

    >>> define_struct('struct abcd {int x; int y;}')
    """
    struct = parse_type(defn)
    ALL_TYPES[struct.name] = struct
    return struct

def lint(fmt='colorized'):
    """Run verbose PyLint on source. Optionally specify fmt=html for HTML output."""
    if fmt == 'html':
        outfile = 'pylint_report.html'
        local('pylint -f %s davies > %s || true' % (fmt, outfile))
        local('open %s' % outfile)
    else:
        local('pylint -f %s davies || true' % fmt)

def get_current_item(self):
        """Returns (first) selected item or None"""
        l = self.selectedIndexes()
        if len(l) > 0:
            return self.model().get_item(l[0])

def median(data):
    """
    Return the median of numeric data, unsing the "mean of middle two" method.
    If ``data`` is empty, ``0`` is returned.

    Examples
    --------

    >>> median([1, 3, 5])
    3.0

    When the number of data points is even, the median is interpolated:
    >>> median([1, 3, 5, 7])
    4.0
    """

    if len(data) == 0:
        return None

    data = sorted(data)
    return float((data[len(data) // 2] + data[(len(data) - 1) // 2]) / 2.)

def last_location_of_minimum(x):
    """
    Returns the last location of the minimal value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    return 1.0 - np.argmin(x[::-1]) / len(x) if len(x) > 0 else np.NaN

def rank(tensor: BKTensor) -> int:
    """Return the number of dimensions of a tensor"""
    if isinstance(tensor, np.ndarray):
        return len(tensor.shape)

    return len(tensor[0].size())

def lower_camel_case_from_underscores(string):
    """generate a lower-cased camelCase string from an underscore_string.
    For example: my_variable_name -> myVariableName"""
    components = string.split('_')
    string = components[0]
    for component in components[1:]:
        string += component[0].upper() + component[1:]
    return string

def get_last_weekday_in_month(year, month, weekday):
        """Get the last weekday in a given month. e.g:

        >>> # the last monday in Jan 2013
        >>> Calendar.get_last_weekday_in_month(2013, 1, MON)
        datetime.date(2013, 1, 28)
        """
        day = date(year, month, monthrange(year, month)[1])
        while True:
            if day.weekday() == weekday:
                break
            day = day - timedelta(days=1)
        return day

def issubset(self, other):
        """
        Report whether another set contains this set.

        Example:
            >>> OrderedSet([1, 2, 3]).issubset({1, 2})
            False
            >>> OrderedSet([1, 2, 3]).issubset({1, 2, 3, 4})
            True
            >>> OrderedSet([1, 2, 3]).issubset({1, 4, 3, 5})
            False
        """
        if len(self) > len(other):  # Fast check for obvious cases
            return False
        return all(item in other for item in self)

def isarray(array, test, dim=2):
    """Returns True if test is True for all array elements.
    Otherwise, returns False.
    """
    if dim > 1:
        return all(isarray(array[i], test, dim - 1)
                   for i in range(len(array)))
    return all(test(i) for i in array)

def argmax(self, rows: List[Row], column: ComparableColumn) -> List[Row]:
        """
        Takes a list of rows and a column name and returns a list containing a single row (dict from
        columns to cells) that has the maximum numerical value in the given column. We return a list
        instead of a single dict to be consistent with the return type of ``select`` and
        ``all_rows``.
        """
        if not rows:
            return []
        value_row_pairs = [(row.values[column.name], row) for row in rows]
        if not value_row_pairs:
            return []
        # Returns a list containing the row with the max cell value.
        return [sorted(value_row_pairs, key=lambda x: x[0], reverse=True)[0][1]]

def debugTreePrint(node,pfx="->"):
  """Purely a debugging aid: Ascii-art picture of a tree descended from node"""
  print pfx,node.item
  for c in node.children:
    debugTreePrint(c,"  "+pfx)

def get_from_gnucash26_date(date_str: str) -> date:
    """ Creates a datetime from GnuCash 2.6 date string """
    date_format = "%Y%m%d"
    result = datetime.strptime(date_str, date_format).date()
    return result

def get_datatype(self, table: str, column: str) -> str:
        """Returns database SQL datatype for a column: e.g. VARCHAR."""
        return self.flavour.get_datatype(self, table, column).upper()

def strings_to_integers(strings: Iterable[str]) -> Iterable[int]:
    """
    Convert a list of strings to a list of integers.

    :param strings: a list of string
    :return: a list of converted integers

    .. doctest::

        >>> strings_to_integers(['1', '1.0', '-0.2'])
        [1, 1, 0]
    """
    return strings_to_(strings, lambda x: int(float(x)))

def flatten_multidict(multidict):
    """Return flattened dictionary from ``MultiDict``."""
    return dict([(key, value if len(value) > 1 else value[0])
                 for (key, value) in multidict.iterlists()])

def get_input_nodes(G: nx.DiGraph) -> List[str]:
    """ Get all input nodes from a network. """
    return [n for n, d in G.in_degree() if d == 0]

def __replace_all(repls: dict, input: str) -> str:
    """ Replaces from a string **input** all the occurrences of some
    symbols according to mapping **repls**.

    :param dict repls: where #key is the old character and
    #value is the one to substitute with;
    :param str input: original string where to apply the
    replacements;
    :return: *(str)* the string with the desired characters replaced
    """
    return re.sub('|'.join(re.escape(key) for key in repls.keys()),
                  lambda k: repls[k.group(0)], input)

def get_timezone() -> Tuple[datetime.tzinfo, str]:
    """Discover the current time zone and it's standard string representation (for source{d})."""
    dt = get_datetime_now().astimezone()
    tzstr = dt.strftime("%z")
    tzstr = tzstr[:-2] + ":" + tzstr[-2:]
    return dt.tzinfo, tzstr

def attr_names(cls) -> List[str]:
        """
        Returns annotated attribute names
        :return: List[str]
        """
        return [k for k, v in cls.attr_types().items()]

def grep(pattern, filename):
    """Very simple grep that returns the first matching line in a file.
    String matching only, does not do REs as currently implemented.
    """
    try:
        # for line in file
        # if line matches pattern:
        #    return line
        return next((L for L in open(filename) if L.find(pattern) >= 0))
    except StopIteration:
        return ''

def proper_round(n):
    """
    rounds float to closest int
    :rtype: int
    :param n: float
    """
    return int(n) + (n / abs(n)) * int(abs(n - int(n)) >= 0.5) if n != 0 else 0

def __rmatmul__(self, other):
        """
        Matrix multiplication using binary `@` operator in Python>=3.5.
        """
        return self.T.dot(np.transpose(other)).T

def memory_full():
    """Check if the memory is too full for further caching."""
    current_process = psutil.Process(os.getpid())
    return (current_process.memory_percent() >
            config.MAXIMUM_CACHE_MEMORY_PERCENTAGE)

def split(text: str) -> List[str]:
    """Split a text into a list of tokens.

    :param text: the text to split
    :return: tokens
    """
    return [word for word in SEPARATOR.split(text) if word.strip(' \t')]

def most_significant_bit(lst: np.ndarray) -> int:
    """
    A helper function that finds the position of the most significant bit in a 1darray of 1s and 0s,
    i.e. the first position where a 1 appears, reading left to right.

    :param lst: a 1d array of 0s and 1s with at least one 1
    :return: the first position in lst that a 1 appears
    """
    return np.argwhere(np.asarray(lst) == 1)[0][0]

def rank(tensor: BKTensor) -> int:
    """Return the number of dimensions of a tensor"""
    if isinstance(tensor, np.ndarray):
        return len(tensor.shape)

    return len(tensor[0].size())

def is_blankspace(self, char):
        """
        Test if a character is a blankspace.

        Parameters
        ----------
        char : str
            The character to test.

        Returns
        -------
        ret : bool
            True if character is a blankspace, False otherwise.

        """
        if len(char) > 1:
            raise TypeError("Expected a char.")
        if char in self.blankspaces:
            return True
        return False

def remove_empty_text(utterances: List[Utterance]) -> List[Utterance]:
    """Remove empty utterances from a list of utterances
    Args:
        utterances: The list of utterance we are processing
    """
    return [utter for utter in utterances if utter.text.strip() != ""]

def stretch(iterable, n=2):
    r"""Repeat each item in `iterable` `n` times.

    Example:

    >>> list(stretch(range(3), 2))
    [0, 0, 1, 1, 2, 2]
    """
    times = range(n)
    for item in iterable:
        for i in times: yield item

def consistent_shuffle(*lists):
    """
    Shuffle lists consistently.

    Parameters
    ----------
    *lists
        Variable length number of lists

    Returns
    -------
    shuffled_lists : tuple of lists
        All of the lists are shuffled consistently

    Examples
    --------
    >>> import mpu, random; random.seed(8)
    >>> mpu.consistent_shuffle([1,2,3], ['a', 'b', 'c'], ['A', 'B', 'C'])
    ([3, 2, 1], ['c', 'b', 'a'], ['C', 'B', 'A'])
    """
    perm = list(range(len(lists[0])))
    random.shuffle(perm)
    lists = tuple([sublist[index] for index in perm]
                  for sublist in lists)
    return lists

def dictlist_wipe_key(dict_list: Iterable[Dict], key: str) -> None:
    """
    Process an iterable of dictionaries. For each dictionary ``d``, delete
    ``d[key]`` if it exists.
    """
    for d in dict_list:
        d.pop(key, None)

def execute(cur, *args):
    """Utility function to print sqlite queries before executing.

    Use instead of cur.execute().  First argument is cursor.

    cur.execute(stmt)
    becomes
    util.execute(cur, stmt)
    """
    stmt = args[0]
    if len(args) > 1:
        stmt = stmt.replace('%', '%%').replace('?', '%r')
        print(stmt % (args[1]))
    return cur.execute(*args)

def file_exists(fname):
    """Check if a file exists and is non-empty.
    """
    try:
        return fname and os.path.exists(fname) and os.path.getsize(fname) > 0
    except OSError:
        return False

def uconcatenate(arrs, axis=0):
    """Concatenate a sequence of arrays.

    This wrapper around numpy.concatenate preserves units. All input arrays
    must have the same units.  See the documentation of numpy.concatenate for
    full details.

    Examples
    --------
    >>> from unyt import cm
    >>> A = [1, 2, 3]*cm
    >>> B = [2, 3, 4]*cm
    >>> uconcatenate((A, B))
    unyt_array([1, 2, 3, 2, 3, 4], 'cm')

    """
    v = np.concatenate(arrs, axis=axis)
    v = _validate_numpy_wrapper_units(v, arrs)
    return v

def full(self):
        """Return ``True`` if the queue is full, ``False``
        otherwise (not reliable!).

        Only applicable if :attr:`maxsize` is set.

        """
        return self.maxsize and len(self.list) >= self.maxsize or False

def is_quoted(arg: str) -> bool:
    """
    Checks if a string is quoted
    :param arg: the string being checked for quotes
    :return: True if a string is quoted
    """
    return len(arg) > 1 and arg[0] == arg[-1] and arg[0] in constants.QUOTES

def get_timezone() -> Tuple[datetime.tzinfo, str]:
    """Discover the current time zone and it's standard string representation (for source{d})."""
    dt = get_datetime_now().astimezone()
    tzstr = dt.strftime("%z")
    tzstr = tzstr[:-2] + ":" + tzstr[-2:]
    return dt.tzinfo, tzstr

def __replace_all(repls: dict, input: str) -> str:
    """ Replaces from a string **input** all the occurrences of some
    symbols according to mapping **repls**.

    :param dict repls: where #key is the old character and
    #value is the one to substitute with;
    :param str input: original string where to apply the
    replacements;
    :return: *(str)* the string with the desired characters replaced
    """
    return re.sub('|'.join(re.escape(key) for key in repls.keys()),
                  lambda k: repls[k.group(0)], input)

def to_bytes(data: Any) -> bytearray:
    """
    Convert anything to a ``bytearray``.
    
    See
    
    - http://stackoverflow.com/questions/7585435/best-way-to-convert-string-to-bytes-in-python-3
    - http://stackoverflow.com/questions/10459067/how-to-convert-my-bytearrayb-x9e-x18k-x9a-to-something-like-this-x9e-x1
    """  # noqa
    if isinstance(data, int):
        return bytearray([data])
    return bytearray(data, encoding='latin-1')

def argmax(self, rows: List[Row], column: ComparableColumn) -> List[Row]:
        """
        Takes a list of rows and a column name and returns a list containing a single row (dict from
        columns to cells) that has the maximum numerical value in the given column. We return a list
        instead of a single dict to be consistent with the return type of ``select`` and
        ``all_rows``.
        """
        if not rows:
            return []
        value_row_pairs = [(row.values[column.name], row) for row in rows]
        if not value_row_pairs:
            return []
        # Returns a list containing the row with the max cell value.
        return [sorted(value_row_pairs, key=lambda x: x[0], reverse=True)[0][1]]

def get_column_names(engine: Engine, tablename: str) -> List[str]:
    """
    Get all the database column names for the specified table.
    """
    return [info.name for info in gen_columns_info(engine, tablename)]

def cpu_count() -> int:
    """Returns the number of processors on this machine."""
    if multiprocessing is None:
        return 1
    try:
        return multiprocessing.cpu_count()
    except NotImplementedError:
        pass
    try:
        return os.sysconf("SC_NPROCESSORS_CONF")
    except (AttributeError, ValueError):
        pass
    gen_log.error("Could not detect number of processors; assuming 1")
    return 1

async def async_run(self) -> None:
        """
        Asynchronously run the worker, does not close connections. Useful when testing.
        """
        self.main_task = self.loop.create_task(self.main())
        await self.main_task

def flatten_list(l: List[list]) -> list:
    """ takes a list of lists, l and returns a flat list
    """
    return [v for inner_l in l for v in inner_l]

def _rindex(mylist: Sequence[T], x: T) -> int:
    """Index of the last occurrence of x in the sequence."""
    return len(mylist) - mylist[::-1].index(x) - 1

def flatten_list(x: List[Any]) -> List[Any]:
    """
    Converts a list of lists into a flat list.
    
    Args:
        x: list of lists 

    Returns:
        flat list
        
    As per
    http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python

    """  # noqa
    return [item for sublist in x for item in sublist]

def head(self) -> Any:
        """Retrive first element in List."""

        lambda_list = self._get_value()
        return lambda_list(lambda head, _: head)

def dictlist_convert_to_float(dict_list: Iterable[Dict], key: str) -> None:
    """
    Process an iterable of dictionaries. For each dictionary ``d``, convert
    (in place) ``d[key]`` to a float. If that fails, convert it to ``None``.
    """
    for d in dict_list:
        try:
            d[key] = float(d[key])
        except ValueError:
            d[key] = None

def iter_lines(file_like: Iterable[str]) -> Generator[str, None, None]:
    """ Helper for iterating only nonempty lines without line breaks"""
    for line in file_like:
        line = line.rstrip('\r\n')
        if line:
            yield line

def compatible_staticpath(path):
    """
    Try to return a path to static the static files compatible all
    the way back to Django 1.2. If anyone has a cleaner or better
    way to do this let me know!
    """

    if VERSION >= (1, 10):
        # Since Django 1.10, forms.Media automatically invoke static
        # lazily on the path if it is relative.
        return path
    try:
        # >= 1.4
        from django.templatetags.static import static
        return static(path)
    except ImportError:
        pass
    try:
        # >= 1.3
        return '%s/%s' % (settings.STATIC_URL.rstrip('/'), path)
    except AttributeError:
        pass
    try:
        return '%s/%s' % (settings.PAGEDOWN_URL.rstrip('/'), path)
    except AttributeError:
        pass
    return '%s/%s' % (settings.MEDIA_URL.rstrip('/'), path)

def ranges_to_set(lst):
    """
    Convert a list of ranges to a set of numbers::

    >>> ranges = [(1,3), (5,6)]
    >>> sorted(list(ranges_to_set(ranges)))
    [1, 2, 3, 5, 6]

    """
    return set(itertools.chain(*(range(x[0], x[1]+1) for x in lst)))

def mkdir(self, target_folder):
        """
        Create a folder on S3.

        Examples
        --------
            >>> s3utils.mkdir("path/to/my_folder")
            Making directory: path/to/my_folder
        """
        self.printv("Making directory: %s" % target_folder)
        self.k.key = re.sub(r"^/|/$", "", target_folder) + "/"
        self.k.set_contents_from_string('')
        self.k.close()

def fetchallfirstvalues(self, sql: str, *args) -> List[Any]:
        """Executes SQL; returns list of first values of each row."""
        rows = self.fetchall(sql, *args)
        return [row[0] for row in rows]

def grep(pattern, filename):
    """Very simple grep that returns the first matching line in a file.
    String matching only, does not do REs as currently implemented.
    """
    try:
        # for line in file
        # if line matches pattern:
        #    return line
        return next((L for L in open(filename) if L.find(pattern) >= 0))
    except StopIteration:
        return ''

def _sum_cycles_from_tokens(self, tokens: List[str]) -> int:
        """Sum the total number of cycles over a list of tokens."""
        return sum((int(self._nonnumber_pattern.sub('', t)) for t in tokens))

def __rmatmul__(self, other):
        """
        Matrix multiplication using binary `@` operator in Python>=3.5.
        """
        return self.T.dot(np.transpose(other)).T

def uniqued(iterable):
    """Return unique list of items preserving order.

    >>> uniqued([3, 2, 1, 3, 2, 1, 0])
    [3, 2, 1, 0]
    """
    seen = set()
    add = seen.add
    return [i for i in iterable if i not in seen and not add(i)]

def extend(a: dict, b: dict) -> dict:
    """Merge two dicts and return a new dict. Much like subclassing works."""
    res = a.copy()
    res.update(b)
    return res

def list_to_str(list, separator=','):
    """
    >>> list = [0, 0, 7]
    >>> list_to_str(list)
    '0,0,7'
    """
    list = [str(x) for x in list]
    return separator.join(list)

def do_quit(self, _: argparse.Namespace) -> bool:
        """Exit this application"""
        self._should_quit = True
        return self._STOP_AND_EXIT

def from_file(file_path) -> dict:
        """ Load JSON file """
        with io.open(file_path, 'r', encoding='utf-8') as json_stream:
            return Json.parse(json_stream, True)

def getIndex(predicateFn: Callable[[T], bool], items: List[T]) -> int:
    """
    Finds the index of an item in list, which satisfies predicate
    :param predicateFn: predicate function to run on items of list
    :param items: list of tuples
    :return: first index for which predicate function returns True
    """
    try:
        return next(i for i, v in enumerate(items) if predicateFn(v))
    except StopIteration:
        return -1

def flatten_list(l: List[list]) -> list:
    """ takes a list of lists, l and returns a flat list
    """
    return [v for inner_l in l for v in inner_l]

def read_text_from_file(path: str) -> str:
    """ Reads text file contents """
    with open(path) as text_file:
        content = text_file.read()

    return content

def csv_to_numpy(string_like, dtype=None):  # type: (str) -> np.array
    """Convert a CSV object to a numpy array.

    Args:
        string_like (str): CSV string.
        dtype (dtype, optional):  Data type of the resulting array. If None, the dtypes will be determined by the
                                        contents of each column, individually. This argument can only be used to
                                        'upcast' the array.  For downcasting, use the .astype(t) method.
    Returns:
        (np.array): numpy array
    """
    stream = StringIO(string_like)
    return np.genfromtxt(stream, dtype=dtype, delimiter=',')

def proper_round(n):
    """
    rounds float to closest int
    :rtype: int
    :param n: float
    """
    return int(n) + (n / abs(n)) * int(abs(n - int(n)) >= 0.5) if n != 0 else 0

def isfile_notempty(inputfile: str) -> bool:
        """Check if the input filename with path is a file and is not empty."""
        try:
            return isfile(inputfile) and getsize(inputfile) > 0
        except TypeError:
            raise TypeError('inputfile is not a valid type')

def fcast(value: float) -> TensorLike:
    """Cast to float tensor"""
    newvalue = tf.cast(value, FTYPE)
    if DEVICE == 'gpu':
        newvalue = newvalue.gpu()  # Why is this needed?  # pragma: no cover
    return newvalue

def genfirstvalues(cursor: Cursor, arraysize: int = 1000) \
        -> Generator[Any, None, None]:
    """
    Generate the first value in each row.

    Args:
        cursor: the cursor
        arraysize: split fetches into chunks of this many records

    Yields:
        the first value of each row
    """
    return (row[0] for row in genrows(cursor, arraysize))

def _cnx_is_empty(in_file):
    """Check if cnr or cns files are empty (only have a header)
    """
    with open(in_file) as in_handle:
        for i, line in enumerate(in_handle):
            if i > 0:
                return False
    return True

def uppercase_chars(string: any) -> str:
        """Return all (and only) the uppercase chars in the given string."""
        return ''.join([c if c.isupper() else '' for c in str(string)])

def excel_datetime(timestamp, epoch=None):
    """Return datetime object from timestamp in Excel serial format.

    Convert LSM time stamps.

    >>> excel_datetime(40237.029999999795)
    datetime.datetime(2010, 2, 28, 0, 43, 11, 999982)

    """
    if epoch is None:
        epoch = datetime.datetime.fromordinal(693594)
    return epoch + datetime.timedelta(timestamp)

def last_location_of_minimum(x):
    """
    Returns the last location of the minimal value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    return 1.0 - np.argmin(x[::-1]) / len(x) if len(x) > 0 else np.NaN

def simple_moving_average(x, n=10):
    """
    Calculate simple moving average

    Parameters
    ----------
    x : ndarray
        A numpy array
    n : integer
        The number of sample points used to make average

    Returns
    -------
    ndarray
        A 1 x n numpy array instance
    """
    if x.ndim > 1 and len(x[0]) > 1:
        x = np.average(x, axis=1)
    a = np.ones(n) / float(n)
    return np.convolve(x, a, 'valid')

def de_duplicate(items):
    """Remove any duplicate item, preserving order

    >>> de_duplicate([1, 2, 1, 2])
    [1, 2]
    """
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return result

def recClearTag(element):
    """Applies maspy.xml.clearTag() to the tag attribute of the "element" and
    recursively to all child elements.

    :param element: an :instance:`xml.etree.Element`
    """
    children = element.getchildren()
    if len(children) > 0:
        for child in children:
            recClearTag(child)
    element.tag = clearTag(element.tag)

def argmax(iterable, key=None, both=False):
    """
    >>> argmax([4,2,-5])
    0
    >>> argmax([4,2,-5], key=abs)
    2
    >>> argmax([4,2,-5], key=abs, both=True)
    (2, 5)
    """
    if key is not None:
        it = imap(key, iterable)
    else:
        it = iter(iterable)
    score, argmax = reduce(max, izip(it, count()))
    if both:
        return argmax, score
    return argmax

def _protected_log(x1):
    """Closure of log for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)

def _kbhit_unix() -> bool:
    """
    Under UNIX: is a keystroke available?
    """
    dr, dw, de = select.select([sys.stdin], [], [], 0)
    return dr != []

def mostLikely(self, pred):
    """ Helper function to return a scalar value representing the most
        likely outcome given a probability distribution
    """
    if len(pred) == 1:
      return pred.keys()[0]

    mostLikelyOutcome = None
    maxProbability = 0

    for prediction, probability in pred.items():
      if probability > maxProbability:
        mostLikelyOutcome = prediction
        maxProbability = probability

    return mostLikelyOutcome

def last_location_of_minimum(x):
    """
    Returns the last location of the minimal value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    return 1.0 - np.argmin(x[::-1]) / len(x) if len(x) > 0 else np.NaN

def from_file(file_path) -> dict:
        """ Load JSON file """
        with io.open(file_path, 'r', encoding='utf-8') as json_stream:
            return Json.parse(json_stream, True)

def moving_average(arr: np.ndarray, n: int = 3) -> np.ndarray:
    """ Calculate the moving overage over an array.

    Algorithm from: https://stackoverflow.com/a/14314054

    Args:
        arr (np.ndarray): Array over which to calculate the moving average.
        n (int): Number of elements over which to calculate the moving average. Default: 3
    Returns:
        np.ndarray: Moving average calculated over n.
    """
    ret = np.cumsum(arr, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def truncate(value: Decimal, n_digits: int) -> Decimal:
    """Truncates a value to a number of decimals places"""
    return Decimal(math.trunc(value * (10 ** n_digits))) / (10 ** n_digits)

def rmglob(pattern: str) -> None:
    """
    Deletes all files whose filename matches the glob ``pattern`` (via
    :func:`glob.glob`).
    """
    for f in glob.glob(pattern):
        os.remove(f)

def singularize(word):
    """
    Return the singular form of a word, the reverse of :func:`pluralize`.

    Examples::

        >>> singularize("posts")
        "post"
        >>> singularize("octopi")
        "octopus"
        >>> singularize("sheep")
        "sheep"
        >>> singularize("word")
        "word"
        >>> singularize("CamelOctopi")
        "CamelOctopus"

    """
    for inflection in UNCOUNTABLES:
        if re.search(r'(?i)\b(%s)\Z' % inflection, word):
            return word

    for rule, replacement in SINGULARS:
        if re.search(rule, word):
            return re.sub(rule, replacement, word)
    return word

def astensor(array: TensorLike) -> BKTensor:
    """Covert numpy array to tensorflow tensor"""
    tensor = tf.convert_to_tensor(value=array, dtype=CTYPE)
    return tensor

def hsv2rgb_spectrum(hsv):
    """Generates RGB values from HSV values in line with a typical light
    spectrum."""
    h, s, v = hsv
    return hsv2rgb_raw(((h * 192) >> 8, s, v))

def strictly_positive_int_or_none(val):
    """Parse `val` into either `None` or a strictly positive integer."""
    val = positive_int_or_none(val)
    if val is None or val > 0:
        return val
    raise ValueError('"{}" must be strictly positive'.format(val))

def remove_blank_lines(string):
    """ Removes all blank lines in @string

        -> #str without blank lines
    """
    return "\n".join(line
                     for line in string.split("\n")
                     if len(line.strip()))

def clean(ctx, text):
    """
    Removes all non-printable characters from a text string
    """
    text = conversions.to_string(text, ctx)
    return ''.join([c for c in text if ord(c) >= 32])

def extend(a: dict, b: dict) -> dict:
    """Merge two dicts and return a new dict. Much like subclassing works."""
    res = a.copy()
    res.update(b)
    return res

def last_location_of_minimum(x):
    """
    Returns the last location of the minimal value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    return 1.0 - np.argmin(x[::-1]) / len(x) if len(x) > 0 else np.NaN

def snake_to_camel(s: str) -> str:
    """Convert string from snake case to camel case."""

    fragments = s.split('_')

    return fragments[0] + ''.join(x.title() for x in fragments[1:])

def dictlist_wipe_key(dict_list: Iterable[Dict], key: str) -> None:
    """
    Process an iterable of dictionaries. For each dictionary ``d``, delete
    ``d[key]`` if it exists.
    """
    for d in dict_list:
        d.pop(key, None)

def is_running(process_id: int) -> bool:
    """
    Uses the Unix ``ps`` program to see if a process is running.
    """
    pstr = str(process_id)
    encoding = sys.getdefaultencoding()
    s = subprocess.Popen(["ps", "-p", pstr], stdout=subprocess.PIPE)
    for line in s.stdout:
        strline = line.decode(encoding)
        if pstr in strline:
            return True
    return False

def _protected_log(x1):
    """Closure of log for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)

def _my_hash(arg_list):
    # type: (List[Any]) -> int
    """Simple helper hash function"""
    res = 0
    for arg in arg_list:
        res = res * 31 + hash(arg)
    return res

def is_sqlatype_string(coltype: Union[TypeEngine, VisitableType]) -> bool:
    """
    Is the SQLAlchemy column type a string type?
    """
    coltype = _coltype_to_typeengine(coltype)
    return isinstance(coltype, sqltypes.String)

def has_obstory_metadata(self, status_id):
        """
        Check for the presence of the given metadata item

        :param string status_id:
            The metadata item ID
        :return:
            True if we have a metadata item with this ID, False otherwise
        """
        self.con.execute('SELECT 1 FROM archive_metadata WHERE publicId=%s;', (status_id,))
        return len(self.con.fetchall()) > 0

def has_key(cls, *args):
        """
        Check whether flyweight object with specified key has already been created.

        Returns:
            bool: True if already created, False if not
        """
        key = args if len(args) > 1 else args[0]
        return key in cls._instances

def top(self, topn=10):
        """
        Get a list of the top ``topn`` features in this :class:`.Feature`\.

        Examples
        --------

        .. code-block:: python

        >>> myFeature = Feature([('the', 2), ('pine', 1), ('trapezoid', 5)])
        >>> myFeature.top(1)
        [('trapezoid', 5)]

        Parameters
        ----------
        topn : int

        Returns
        -------
        list
        """
        return [self[i] for i in argsort(list(zip(*self))[1])[::-1][:topn]]

def require(executable: str, explanation: str = "") -> None:
    """
    Ensures that the external tool is available.
    Asserts upon failure.
    """
    assert shutil.which(executable), "Need {!r} on the PATH.{}".format(
        executable, "\n" + explanation if explanation else "")

def rank(tensor: BKTensor) -> int:
    """Return the number of dimensions of a tensor"""
    if isinstance(tensor, np.ndarray):
        return len(tensor.shape)

    return len(tensor[0].size())

def make_indices_to_labels(labels: Set[str]) -> Dict[int, str]:
    """ Creates a mapping from indices to labels. """

    return {index: label for index, label in
            enumerate(["pad"] + sorted(list(labels)))}

def issubset(self, other):
        """
        Report whether another set contains this set.

        Example:
            >>> OrderedSet([1, 2, 3]).issubset({1, 2})
            False
            >>> OrderedSet([1, 2, 3]).issubset({1, 2, 3, 4})
            True
            >>> OrderedSet([1, 2, 3]).issubset({1, 4, 3, 5})
            False
        """
        if len(self) > len(other):  # Fast check for obvious cases
            return False
        return all(item in other for item in self)

def without(seq1, seq2):
    r"""Return a list with all elements in `seq2` removed from `seq1`, order
    preserved.

    Examples:

    >>> without([1,2,3,1,2], [1])
    [2, 3, 2]
    """
    if isSet(seq2): d2 = seq2
    else: d2 = set(seq2)
    return [elt for elt in seq1 if elt not in d2]

def read32(bytestream):
  """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def _check_stream_timeout(started, timeout):
    """Check if the timeout has been reached and raise a `StopIteration` if so.
    """
    if timeout:
        elapsed = datetime.datetime.utcnow() - started
        if elapsed.seconds > timeout:
            raise StopIteration

def tanimoto_set_similarity(x: Iterable[X], y: Iterable[X]) -> float:
    """Calculate the tanimoto set similarity."""
    a, b = set(x), set(y)
    union = a | b

    if not union:
        return 0.0

    return len(a & b) / len(union)

def _cnx_is_empty(in_file):
    """Check if cnr or cns files are empty (only have a header)
    """
    with open(in_file) as in_handle:
        for i, line in enumerate(in_handle):
            if i > 0:
                return False
    return True

def find_first_in_list(txt: str, str_list: [str]) -> int:  # type: ignore
    """
    Returns the index of the earliest occurence of an item from a list in a string

    Ex: find_first_in_list('foobar', ['bar', 'fin']) -> 3
    """
    start = len(txt) + 1
    for item in str_list:
        if start > txt.find(item) > -1:
            start = txt.find(item)
    return start if len(txt) + 1 > start > -1 else -1

def do_quit(self, _: argparse.Namespace) -> bool:
        """Exit this application"""
        self._should_quit = True
        return self._STOP_AND_EXIT

def remove_blank_lines(string):
    """ Removes all blank lines in @string

        -> #str without blank lines
    """
    return "\n".join(line
                     for line in string.split("\n")
                     if len(line.strip()))

def rmglob(pattern: str) -> None:
    """
    Deletes all files whose filename matches the glob ``pattern`` (via
    :func:`glob.glob`).
    """
    for f in glob.glob(pattern):
        os.remove(f)

def remove_empty_text(utterances: List[Utterance]) -> List[Utterance]:
    """Remove empty utterances from a list of utterances
    Args:
        utterances: The list of utterance we are processing
    """
    return [utter for utter in utterances if utter.text.strip() != ""]

def position(self) -> Position:
        """The current position of the cursor."""
        return Position(self._index, self._lineno, self._col_offset)

def remove_empty_text(utterances: List[Utterance]) -> List[Utterance]:
    """Remove empty utterances from a list of utterances
    Args:
        utterances: The list of utterance we are processing
    """
    return [utter for utter in utterances if utter.text.strip() != ""]

def _duplicates(list_):
    """Return dict mapping item -> indices."""
    item_indices = {}
    for i, item in enumerate(list_):
        try:
            item_indices[item].append(i)
        except KeyError:  # First time seen
            item_indices[item] = [i]
    return item_indices

def trim_decimals(s, precision=-3):
        """
        Convert from scientific notation using precision
        """
        encoded = s.encode('ascii', 'ignore')
        str_val = ""
        if six.PY3:
            str_val = str(encoded, encoding='ascii', errors='ignore')[:precision]
        else:
            # If precision is 0, this must be handled seperately
            if precision == 0:
                str_val = str(encoded)
            else:
                str_val = str(encoded)[:precision]
        if len(str_val) > 0:
            return float(str_val)
        else:
            return 0

def arcball_map_to_sphere(point, center, radius):
    """Return unit sphere coordinates from window coordinates."""
    v0 = (point[0] - center[0]) / radius
    v1 = (center[1] - point[1]) / radius
    n = v0*v0 + v1*v1
    if n > 1.0:
        # position outside of sphere
        n = math.sqrt(n)
        return numpy.array([v0/n, v1/n, 0.0])
    else:
        return numpy.array([v0, v1, math.sqrt(1.0 - n)])

def __remove_trailing_zeros(self, collection):
        """Removes trailing zeroes from indexable collection of numbers"""
        index = len(collection) - 1
        while index >= 0 and collection[index] == 0:
            index -= 1

        return collection[:index + 1]

def flatten_multidict(multidict):
    """Return flattened dictionary from ``MultiDict``."""
    return dict([(key, value if len(value) > 1 else value[0])
                 for (key, value) in multidict.iterlists()])

def is_natural(x):
    """A non-negative integer."""
    try:
        is_integer = int(x) == x
    except (TypeError, ValueError):
        return False
    return is_integer and x >= 0

def split(text: str) -> List[str]:
    """Split a text into a list of tokens.

    :param text: the text to split
    :return: tokens
    """
    return [word for word in SEPARATOR.split(text) if word.strip(' \t')]

def normcdf(x, log=False):
    """Normal cumulative density function."""
    y = np.atleast_1d(x).copy()
    flib.normcdf(y)
    if log:
        if (y>0).all():
            return np.log(y)
        return -np.inf
    return y

def split(text: str) -> List[str]:
    """Split a text into a list of tokens.

    :param text: the text to split
    :return: tokens
    """
    return [word for word in SEPARATOR.split(text) if word.strip(' \t')]

def to_int64(a):
    """Return view of the recarray with all int32 cast to int64."""
    # build new dtype and replace i4 --> i8
    def promote_i4(typestr):
        if typestr[1:] == 'i4':
            typestr = typestr[0]+'i8'
        return typestr

    dtype = [(name, promote_i4(typestr)) for name,typestr in a.dtype.descr]
    return a.astype(dtype)

def read_set_from_file(filename: str) -> Set[str]:
    """
    Extract a de-duped collection (set) of text from a file.
    Expected file format is one item per line.
    """
    collection = set()
    with open(filename, 'r') as file_:
        for line in file_:
            collection.add(line.rstrip())
    return collection

def is_natural(x):
    """A non-negative integer."""
    try:
        is_integer = int(x) == x
    except (TypeError, ValueError):
        return False
    return is_integer and x >= 0

def str_to_time(time_str: str) -> datetime.datetime:
    """
    Convert human readable string to datetime.datetime.
    """
    pieces: Any = [int(piece) for piece in time_str.split('-')]
    return datetime.datetime(*pieces)

def test_string(self, string: str) -> bool:
        """If `string` comes next, return ``True`` and advance offset.

        Args:
            string: string to test
        """
        if self.input.startswith(string, self.offset):
            self.offset += len(string)
            return True
        return False

def Exit(msg, code=1):
    """Exit execution with return code and message
    :param msg: Message displayed prior to exit
    :param code: code returned upon exiting
    """
    print >> sys.stderr, msg
    sys.exit(code)

def truncate(value: Decimal, n_digits: int) -> Decimal:
    """Truncates a value to a number of decimals places"""
    return Decimal(math.trunc(value * (10 ** n_digits))) / (10 ** n_digits)

def capture_stdout():
    """Intercept standard output in a with-context
    :return: cStringIO instance

    >>> with capture_stdout() as stdout:
            ...
        print stdout.getvalue()
    """
    stdout = sys.stdout
    sys.stdout = six.moves.cStringIO()
    try:
        yield sys.stdout
    finally:
        sys.stdout = stdout

def list_to_str(lst):
    """
    Turn a list into a comma- and/or and-separated string.

    Parameters
    ----------
    lst : :obj:`list`
        A list of strings to join into a single string.

    Returns
    -------
    str_ : :obj:`str`
        A string with commas and/or ands separating th elements from ``lst``.

    """
    if len(lst) == 1:
        str_ = lst[0]
    elif len(lst) == 2:
        str_ = ' and '.join(lst)
    elif len(lst) > 2:
        str_ = ', '.join(lst[:-1])
        str_ += ', and {0}'.format(lst[-1])
    else:
        raise ValueError('List of length 0 provided.')
    return str_

def s3_get(url: str, temp_file: IO) -> None:
    """Pull a file directly from S3."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)

def memory_read(self, start_position: int, size: int) -> memoryview:
        """
        Read and return a view of ``size`` bytes from memory starting at ``start_position``.
        """
        return self._memory.read(start_position, size)

def spanning_tree_count(graph: nx.Graph) -> int:
    """Return the number of unique spanning trees of a graph, using
    Kirchhoff's matrix tree theorem.
    """
    laplacian = nx.laplacian_matrix(graph).toarray()
    comatrix = laplacian[:-1, :-1]
    det = np.linalg.det(comatrix)
    count = int(round(det))
    return count

def get_valid_filename(s):
    """
    Returns the given string converted to a string that can be used for a clean
    filename. Specifically, leading and trailing spaces are removed; other
    spaces are converted to underscores; and anything that is not a unicode
    alphanumeric, dash, underscore, or dot, is removed.
    >>> get_valid_filename("john's portrait in 2004.jpg")
    'johns_portrait_in_2004.jpg'
    """
    s = s.strip().replace(" ", "_")
    return re.sub(r"(?u)[^-\w.]", "", s)

def is_natural(x):
    """A non-negative integer."""
    try:
        is_integer = int(x) == x
    except (TypeError, ValueError):
        return False
    return is_integer and x >= 0

def find_duplicates(l: list) -> set:
    """
    Return the duplicates in a list.

    The function relies on
    https://stackoverflow.com/questions/9835762/find-and-list-duplicates-in-a-list .
    Parameters
    ----------
    l : list
        Name

    Returns
    -------
    set
        Duplicated values

    >>> find_duplicates([1,2,3])
    set()
    >>> find_duplicates([1,2,1])
    {1}
    """
    return set([x for x in l if l.count(x) > 1])

def from_file(file_path) -> dict:
        """ Load JSON file """
        with io.open(file_path, 'r', encoding='utf-8') as json_stream:
            return Json.parse(json_stream, True)

def most_significant_bit(lst: np.ndarray) -> int:
    """
    A helper function that finds the position of the most significant bit in a 1darray of 1s and 0s,
    i.e. the first position where a 1 appears, reading left to right.

    :param lst: a 1d array of 0s and 1s with at least one 1
    :return: the first position in lst that a 1 appears
    """
    return np.argwhere(np.asarray(lst) == 1)[0][0]

def most_frequent(lst):
    """
    Returns the item that appears most frequently in the given list.
    """
    lst = lst[:]
    highest_freq = 0
    most_freq = None

    for val in unique(lst):
        if lst.count(val) > highest_freq:
            most_freq = val
            highest_freq = lst.count(val)
            
    return most_freq

def _is_numeric(self, values):
        """Check to be sure values are numbers before doing numerical operations."""
        if len(values) > 0:
            assert isinstance(values[0], (float, int)), \
                "values must be numbers to perform math operations. Got {}".format(
                    type(values[0]))
        return True

def _check_whitespace(string):
    """
    Make sure thre is no whitespace in the given string. Will raise a
    ValueError if whitespace is detected
    """
    if string.count(' ') + string.count('\t') + string.count('\n') > 0:
        raise ValueError(INSTRUCTION_HAS_WHITESPACE)

def _isint(string):
    """
    >>> _isint("123")
    True
    >>> _isint("123.45")
    False
    """
    return type(string) is int or \
           (isinstance(string, _binary_type) or isinstance(string, _text_type)) and \
           _isconvertible(int, string)

def get_margin(length):
    """Add enough tabs to align in two columns"""
    if length > 23:
        margin_left = "\t"
        chars = 1
    elif length > 15:
        margin_left = "\t\t"
        chars = 2
    elif length > 7:
        margin_left = "\t\t\t"
        chars = 3
    else:
        margin_left = "\t\t\t\t"
        chars = 4
    return margin_left

def _cnx_is_empty(in_file):
    """Check if cnr or cns files are empty (only have a header)
    """
    with open(in_file) as in_handle:
        for i, line in enumerate(in_handle):
            if i > 0:
                return False
    return True

def is_integer(value: Any) -> bool:
    """Return true if a value is an integer number."""
    return (isinstance(value, int) and not isinstance(value, bool)) or (
        isinstance(value, float) and isfinite(value) and int(value) == value
    )

def indexes_equal(a: Index, b: Index) -> bool:
    """
    Are two indexes equal? Checks by comparing ``str()`` versions of them.
    (AM UNSURE IF THIS IS ENOUGH.)
    """
    return str(a) == str(b)

def is_relative_url(url):
    """ simple method to determine if a url is relative or absolute """
    if url.startswith("#"):
        return None
    if url.find("://") > 0 or url.startswith("//"):
        # either 'http(s)://...' or '//cdn...' and therefore absolute
        return False
    return True

def get_keys_of_max_n(dict_obj, n):
    """Returns the keys that maps to the top n max values in the given dict.

    Example:
    --------
    >>> dict_obj = {'a':2, 'b':1, 'c':5}
    >>> get_keys_of_max_n(dict_obj, 2)
    ['a', 'c']
    """
    return sorted([
        item[0]
        for item in sorted(
            dict_obj.items(), key=lambda item: item[1], reverse=True
        )[:n]
    ])

def nTimes(n, f, *args, **kwargs):
    r"""Call `f` `n` times with `args` and `kwargs`.
    Useful e.g. for simplistic timing.

    Examples:

    >>> nTimes(3, sys.stdout.write, 'hallo\n')
    hallo
    hallo
    hallo

    """
    for i in xrange(n): f(*args, **kwargs)

def encode_list(key, list_):
    # type: (str, Iterable) -> Dict[str, str]
    """
    Converts a list into a space-separated string and puts it in a dictionary

    :param key: Dictionary key to store the list
    :param list_: A list of objects
    :return: A dictionary key->string or an empty dictionary
    """
    if not list_:
        return {}
    return {key: " ".join(str(i) for i in list_)}

def _width_is_big_enough(image, width):
    """Check that the image width is superior to `width`"""
    if width > image.size[0]:
        raise ImageSizeError(image.size[0], width)

def assert_equal(first, second, msg_fmt="{msg}"):
    """Fail unless first equals second, as determined by the '==' operator.

    >>> assert_equal(5, 5.0)
    >>> assert_equal("Hello World!", "Goodbye!")
    Traceback (most recent call last):
        ...
    AssertionError: 'Hello World!' != 'Goodbye!'

    The following msg_fmt arguments are supported:
    * msg - the default error message
    * first - the first argument
    * second - the second argument
    """

    if isinstance(first, dict) and isinstance(second, dict):
        assert_dict_equal(first, second, msg_fmt)
    elif not first == second:
        msg = "{!r} != {!r}".format(first, second)
        fail(msg_fmt.format(msg=msg, first=first, second=second))

def add_colons(s):
    """Add colons after every second digit.

    This function is used in functions to prettify serials.

    >>> add_colons('teststring')
    'te:st:st:ri:ng'
    """
    return ':'.join([s[i:i + 2] for i in range(0, len(s), 2)])

def label_from_bin(buf):
    """
    Converts binary representation label to integer.

    :param buf: Binary representation of label.
    :return: MPLS Label and BoS bit.
    """

    mpls_label = type_desc.Int3.to_user(six.binary_type(buf))
    return mpls_label >> 4, mpls_label & 1

def zfill(x, width):
    """zfill(x, width) -> string

    Pad a numeric string x with zeros on the left, to fill a field
    of the specified width.  The string x is never truncated.

    """
    if not isinstance(x, basestring):
        x = repr(x)
    return x.zfill(width)

def valid_file(path: str) -> bool:
    """
    Verifies that a string path actually exists and is a file

    :param path: The path to verify
    :return: **True** if path exist and is a file
    """
    path = Path(path).expanduser()
    log.debug("checking if %s is a valid file", path)
    return path.exists() and path.is_file()

def _mid(pt1, pt2):
    """
    (Point, Point) -> Point
    Return the point that lies in between the two input points.
    """
    (x0, y0), (x1, y1) = pt1, pt2
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)

def copen(filepath, flag='r', encoding=None):

    """
    FIXME: How to test this ?

    >>> c = copen(__file__)
    >>> c is not None
    True
    """
    if encoding is None:
        encoding = locale.getdefaultlocale()[1]

    return codecs.open(filepath, flag, encoding)

def _(f, x):
    """
    filter for dict, note `f` should have signature: `f::key->value->bool`
    """
    return {k: v for k, v in x.items() if f(k, v)}

def memory_full():
    """Check if the memory is too full for further caching."""
    current_process = psutil.Process(os.getpid())
    return (current_process.memory_percent() >
            config.MAXIMUM_CACHE_MEMORY_PERCENTAGE)

def dag_longest_path(graph, source, target):
    """
    Finds the longest path in a dag between two nodes
    """
    if source == target:
        return [source]
    allpaths = nx.all_simple_paths(graph, source, target)
    longest_path = []
    for l in allpaths:
        if len(l) > len(longest_path):
            longest_path = l
    return longest_path

def uppercase_chars(string: any) -> str:
        """Return all (and only) the uppercase chars in the given string."""
        return ''.join([c if c.isupper() else '' for c in str(string)])

def convert_bytes_to_ints(in_bytes, num):
    """Convert a byte array into an integer array. The number of bytes forming an integer
    is defined by num

    :param in_bytes: the input bytes
    :param num: the number of bytes per int
    :return the integer array"""
    dt = numpy.dtype('>i' + str(num))
    return numpy.frombuffer(in_bytes, dt)

def astensor(array: TensorLike) -> BKTensor:
    """Covert numpy array to tensorflow tensor"""
    tensor = tf.convert_to_tensor(value=array, dtype=CTYPE)
    return tensor

def camel_to_snake(s: str) -> str:
    """Convert string from camel case to snake case."""

    return CAMEL_CASE_RE.sub(r'_\1', s).strip().lower()

def isfile_notempty(inputfile: str) -> bool:
        """Check if the input filename with path is a file and is not empty."""
        try:
            return isfile(inputfile) and getsize(inputfile) > 0
        except TypeError:
            raise TypeError('inputfile is not a valid type')

def _(f, x):
    """
    filter for dict, note `f` should have signature: `f::key->value->bool`
    """
    return {k: v for k, v in x.items() if f(k, v)}

def post(self, endpoint: str, **kwargs) -> dict:
        """HTTP POST operation to API endpoint."""

        return self._request('POST', endpoint, **kwargs)

def create_opengl_object(gl_gen_function, n=1):
    """Returns int pointing to an OpenGL texture"""
    handle = gl.GLuint(1)
    gl_gen_function(n, byref(handle))  # Create n Empty Objects
    if n > 1:
        return [handle.value + el for el in range(n)]  # Return list of handle values
    else:
        return handle.value

def __as_list(value: List[JsonObjTypes]) -> List[JsonTypes]:
        """ Return a json array as a list

        :param value: array
        :return: array with JsonObj instances removed
        """
        return [e._as_dict if isinstance(e, JsonObj) else e for e in value]

def text_to_bool(value: str) -> bool:
    """
    Tries to convert a text value to a bool. If unsuccessful returns if value is None or not

    :param value: Value to check
    """
    try:
        return bool(strtobool(value))
    except (ValueError, AttributeError):
        return value is not None

def _run_sync(self, method: Callable, *args, **kwargs) -> Any:
        """
        Utility method to run commands synchronously for testing.
        """
        if self.loop.is_running():
            raise RuntimeError("Event loop is already running.")

        if not self.is_connected:
            self.loop.run_until_complete(self.connect())

        task = asyncio.Task(method(*args, **kwargs), loop=self.loop)
        result = self.loop.run_until_complete(task)

        self.loop.run_until_complete(self.quit())

        return result

def is_any_type_set(sett: Set[Type]) -> bool:
    """
    Helper method to check if a set of types is the {AnyObject} singleton

    :param sett:
    :return:
    """
    return len(sett) == 1 and is_any_type(min(sett))

def _check_env_var(envvar: str) -> bool:
    """Check Environment Variable to verify that it is set and not empty.

    :param envvar: Environment Variable to Check.

    :returns: True if Environment Variable is set and not empty.

    :raises: KeyError if Environment Variable is not set or is empty.

    .. versionadded:: 0.0.12
    """
    if os.getenv(envvar) is None:
        raise KeyError(
            "Required ENVVAR: {0} is not set".format(envvar))
    if not os.getenv(envvar):  # test if env var is empty
        raise KeyError(
            "Required ENVVAR: {0} is empty".format(envvar))
    return True

def warn_if_nans_exist(X):
    """Warn if nans exist in a numpy array."""
    null_count = count_rows_with_nans(X)
    total = len(X)
    percent = 100 * null_count / total

    if null_count > 0:
        warning_message = \
            'Warning! Found {} rows of {} ({:0.2f}%) with nan values. Only ' \
            'complete rows will be plotted.'.format(null_count, total, percent)
        warnings.warn(warning_message, DataWarning)

async def executemany(self, sql: str, parameters: Iterable[Iterable[Any]]) -> None:
        """Execute the given multiquery."""
        await self._execute(self._cursor.executemany, sql, parameters)

def is_iterable_but_not_string(obj):
    """
    Determine whether or not obj is iterable but not a string (eg, a list, set, tuple etc).
    """
    return hasattr(obj, '__iter__') and not isinstance(obj, str) and not isinstance(obj, bytes)

def make_file_readable (filename):
    """Make file user readable if it is not a link."""
    if not os.path.islink(filename):
        util.set_mode(filename, stat.S_IRUSR)

def apply(filter):
    """Manufacture decorator that filters return value with given function.

    ``filter``:
      Callable that takes a single parameter.
    """
    def decorator(callable):
        return lambda *args, **kwargs: filter(callable(*args, **kwargs))
    return decorator

def ensure_newline(self):
        """
        use before any custom printing when using the progress iter to ensure
        your print statement starts on a new line instead of at the end of a
        progress line
        """
        DECTCEM_SHOW = '\033[?25h'  # show cursor
        AT_END = DECTCEM_SHOW + '\n'
        if not self._cursor_at_newline:
            self.write(AT_END)
            self._cursor_at_newline = True

def variance(arr):
  """variance of the values, must have 2 or more entries.

  :param arr: list of numbers
  :type arr: number[] a number array
  :return: variance
  :rtype: float

  """
  avg = average(arr)
  return sum([(float(x)-avg)**2 for x in arr])/float(len(arr)-1)

def is_bool_matrix(l):
    r"""Checks if l is a 2D numpy array of bools

    """
    if isinstance(l, np.ndarray):
        if l.ndim == 2 and (l.dtype == bool):
            return True
    return False

def is_integer_array(val):
    """
    Checks whether a variable is a numpy integer array.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        True if the variable is a numpy integer array. Otherwise False.

    """
    return is_np_array(val) and issubclass(val.dtype.type, np.integer)

def rrmdir(directory):
    """
    Recursivly delete a directory

    :param directory: directory to remove
    """
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(directory)

def values(self):
        """Gets the user enter max and min values of where the 
        raster points should appear on the y-axis

        :returns: (float, float) -- (min, max) y-values to bound the raster plot by
        """
        lower = float(self.lowerSpnbx.value())
        upper = float(self.upperSpnbx.value())
        return (lower, upper)

def list_formatter(handler, item, value):
    """Format list."""
    return u', '.join(str(v) for v in value)

def distinct(xs):
    """Get the list of distinct values with preserving order."""
    # don't use collections.OrderedDict because we do support Python 2.6
    seen = set()
    return [x for x in xs if x not in seen and not seen.add(x)]

def graph_from_dot_file(path):
    """Load graph as defined by a DOT file.
    
    The file is assumed to be in DOT format. It will
    be loaded, parsed and a Dot class will be returned, 
    representing the graph.
    """
    
    fd = file(path, 'rb')
    data = fd.read()
    fd.close()
    
    return graph_from_dot_data(data)

def butlast(iterable):
    """Yield all items from ``iterable`` except the last one.

    >>> list(butlast(['spam', 'eggs', 'ham']))
    ['spam', 'eggs']

    >>> list(butlast(['spam']))
    []

    >>> list(butlast([]))
    []
    """
    iterable = iter(iterable)
    try:
        first = next(iterable)
    except StopIteration:
        return
    for second in iterable:
        yield first
        first = second

def _read_stream_for_size(stream, buf_size=65536):
    """Reads a stream discarding the data read and returns its size."""
    size = 0
    while True:
        buf = stream.read(buf_size)
        size += len(buf)
        if not buf:
            break
    return size

def s3_get(url: str, temp_file: IO) -> None:
    """Pull a file directly from S3."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)

def Proxy(f):
  """A helper to create a proxy method in a class."""

  def Wrapped(self, *args):
    return getattr(self, f)(*args)

  return Wrapped

def is_float(value):
    """must be a float"""
    return isinstance(value, float) or isinstance(value, int) or isinstance(value, np.float64), float(value)

def cross_join(df1, df2):
    """
    Return a dataframe that is a cross between dataframes
    df1 and df2

    ref: https://github.com/pydata/pandas/issues/5401
    """
    if len(df1) == 0:
        return df2

    if len(df2) == 0:
        return df1

    # Add as lists so that the new index keeps the items in
    # the order that they are added together
    all_columns = pd.Index(list(df1.columns) + list(df2.columns))
    df1['key'] = 1
    df2['key'] = 1
    return pd.merge(df1, df2, on='key').loc[:, all_columns]

def cli_run():
    """docstring for argparse"""
    parser = argparse.ArgumentParser(description='Stupidly simple code answers from StackOverflow')
    parser.add_argument('query', help="What's the problem ?", type=str, nargs='+')
    parser.add_argument('-t','--tags', help='semicolon separated tags -> python;lambda')
    args = parser.parse_args()
    main(args)

def setdefault(obj, field, default):
    """Set an object's field to default if it doesn't have a value"""
    setattr(obj, field, getattr(obj, field, default))

def unique(seq):
    """Return the unique elements of a collection even if those elements are
       unhashable and unsortable, like dicts and sets"""
    cleaned = []
    for each in seq:
        if each not in cleaned:
            cleaned.append(each)
    return cleaned

def clean_int(x) -> int:
    """
    Returns its parameter as an integer, or raises
    ``django.forms.ValidationError``.
    """
    try:
        return int(x)
    except ValueError:
        raise forms.ValidationError(
            "Cannot convert to integer: {}".format(repr(x)))

def contains_empty(features):
    """Check features data are not empty

    :param features: The features data to check.
    :type features: list of numpy arrays.

    :return: True if one of the array is empty, False else.

    """
    if not features:
        return True
    for feature in features:
        if feature.shape[0] == 0:
            return True
    return False

def _go_to_line(editor, line):
    """
    Move cursor to this line in the current buffer.
    """
    b = editor.application.current_buffer
    b.cursor_position = b.document.translate_row_col_to_index(max(0, int(line) - 1), 0)

def test_value(self, value):
        """Test if value is an instance of int."""
        if not isinstance(value, int):
            raise ValueError('expected int value: ' + str(type(value)))

def truncate(value: Decimal, n_digits: int) -> Decimal:
    """Truncates a value to a number of decimals places"""
    return Decimal(math.trunc(value * (10 ** n_digits))) / (10 ** n_digits)

def add_exec_permission_to(target_file):
    """Add executable permissions to the file

    :param target_file: the target file whose permission to be changed
    """
    mode = os.stat(target_file).st_mode
    os.chmod(target_file, mode | stat.S_IXUSR)

def _manhattan_distance(vec_a, vec_b):
    """Return manhattan distance between two lists of numbers."""
    if len(vec_a) != len(vec_b):
        raise ValueError('len(vec_a) must equal len(vec_b)')
    return sum(map(lambda a, b: abs(a - b), vec_a, vec_b))

def dag_longest_path(graph, source, target):
    """
    Finds the longest path in a dag between two nodes
    """
    if source == target:
        return [source]
    allpaths = nx.all_simple_paths(graph, source, target)
    longest_path = []
    for l in allpaths:
        if len(l) > len(longest_path):
            longest_path = l
    return longest_path

def getpackagepath():
    """
     *Get the root path for this python package - used in unit testing code*
    """
    moduleDirectory = os.path.dirname(__file__)
    packagePath = os.path.dirname(__file__) + "/../"

    return packagePath

def to_bin(data, width):
    """
    Convert an unsigned integer to a numpy binary array with the first
    element the MSB and the last element the LSB.
    """
    data_str = bin(data & (2**width-1))[2:].zfill(width)
    return [int(x) for x in tuple(data_str)]

def __remove_trailing_zeros(self, collection):
        """Removes trailing zeroes from indexable collection of numbers"""
        index = len(collection) - 1
        while index >= 0 and collection[index] == 0:
            index -= 1

        return collection[:index + 1]

def fix(h, i):
    """Rearrange the heap after the item at position i got updated."""
    down(h, i, h.size())
    up(h, i)

def crop_box(im, box=False, **kwargs):
    """Uses box coordinates to crop an image without resizing it first."""
    if box:
        im = im.crop(box)
    return im

def _linear_interpolation(x, X, Y):
    """Given two data points [X,Y], linearly interpolate those at x.
    """
    return (Y[1] * (x - X[0]) + Y[0] * (X[1] - x)) / (X[1] - X[0])

def is_numeric_dtype(dtype):
    """Return ``True`` if ``dtype`` is a numeric type."""
    dtype = np.dtype(dtype)
    return np.issubsctype(getattr(dtype, 'base', None), np.number)

def delimited(items, character='|'):
    """Returns a character delimited version of the provided list as a Python string"""
    return '|'.join(items) if type(items) in (list, tuple, set) else items

def json_iter (path):
    """
    iterator for JSON-per-line in a file pattern
    """
    with open(path, 'r') as f:
        for line in f.readlines():
            yield json.loads(line)

def is_valid(number):
    """determines whether the card number is valid."""
    n = str(number)
    if not n.isdigit():
        return False
    return int(n[-1]) == get_check_digit(n[:-1])

def isbinary(*args):
    """Checks if value can be part of binary/bitwise operations."""
    return all(map(lambda c: isnumber(c) or isbool(c), args))

def rm_empty_indices(*args):
    """
    Remove unwanted list indices. First argument is the list
    of indices to remove. Other elements are the lists
    to trim.
    """
    rm_inds = args[0]

    if not rm_inds:
        return args[1:]

    keep_inds = [i for i in range(len(args[1])) if i not in rm_inds]

    return [[a[i] for i in keep_inds] for a in args[1:]]

def clear_instance(cls):
        """unset _instance for this class and singleton parents.
        """
        if not cls.initialized():
            return
        for subclass in cls._walk_mro():
            if isinstance(subclass._instance, cls):
                # only clear instances that are instances
                # of the calling class
                subclass._instance = None

def stderr(a):
    """
    Calculate the standard error of a.
    """
    return np.nanstd(a) / np.sqrt(sum(np.isfinite(a)))

def __init__(self, capacity=10):
        """
        Initialize python List with capacity of 10 or user given input.
        Python List type is a dynamic array, so we have to restrict its
        dynamic nature to make it work like a static array.
        """
        super().__init__()
        self._array = [None] * capacity
        self._front = 0
        self._rear = 0

def login(self, user: str, passwd: str) -> None:
        """Log in to instagram with given username and password and internally store session object.

        :raises InvalidArgumentException: If the provided username does not exist.
        :raises BadCredentialsException: If the provided password is wrong.
        :raises ConnectionException: If connection to Instagram failed.
        :raises TwoFactorAuthRequiredException: First step of 2FA login done, now call :meth:`Instaloader.two_factor_login`."""
        self.context.login(user, passwd)

def get_shape(img):
    """Return the shape of img.

    Paramerers
    -----------
    img:

    Returns
    -------
    shape: tuple
    """
    if hasattr(img, 'shape'):
        shape = img.shape
    else:
        shape = img.get_data().shape
    return shape

def rotateImage(image, angle):
    """
        rotates a 2d array to a multiple of 90 deg.
        0 = default
        1 = 90 deg. cw
        2 = 180 deg.
        3 = 90 deg. ccw
    """
    image = [list(row) for row in image]

    for n in range(angle % 4):
        image = list(zip(*image[::-1]))

    return image

def staticdir():
    """Return the location of the static data directory."""
    root = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(root, "static")

def add_chart(self, chart, row, col):
        """
        Adds a chart to the worksheet at (row, col).

        :param xltable.Chart Chart: chart to add to the workbook.
        :param int row: Row to add the chart at.
        """
        self.__charts.append((chart, (row, col)))

def from_series(cls, series):
        """Convert a pandas.Series into an xarray.DataArray.

        If the series's index is a MultiIndex, it will be expanded into a
        tensor product of one-dimensional coordinates (filling in missing
        values with NaN). Thus this operation should be the inverse of the
        `to_series` method.
        """
        # TODO: add a 'name' parameter
        name = series.name
        df = pd.DataFrame({name: series})
        ds = Dataset.from_dataframe(df)
        return ds[name]

def genfirstvalues(cursor: Cursor, arraysize: int = 1000) \
        -> Generator[Any, None, None]:
    """
    Generate the first value in each row.

    Args:
        cursor: the cursor
        arraysize: split fetches into chunks of this many records

    Yields:
        the first value of each row
    """
    return (row[0] for row in genrows(cursor, arraysize))

def contains(self, token: str) -> bool:
        """Return if the token is in the list or not."""
        self._validate_token(token)
        return token in self

def sem(inlist):
    """
Returns the estimated standard error of the mean (sx-bar) of the
values in the passed list.  sem = stdev / sqrt(n)

Usage:   lsem(inlist)
"""
    sd = stdev(inlist)
    n = len(inlist)
    return sd / math.sqrt(n)

def get(key, default=None):
    """ return the key from the request
    """
    data = get_form() or get_query_string()
    return data.get(key, default)

def use_kwargs(self, *args, **kwargs) -> typing.Callable:
        """Decorator that injects parsed arguments into a view function or method.

        Receives the same arguments as `webargs.core.Parser.use_kwargs`.

        """
        return super().use_kwargs(*args, **kwargs)

def _check_whitespace(string):
    """
    Make sure thre is no whitespace in the given string. Will raise a
    ValueError if whitespace is detected
    """
    if string.count(' ') + string.count('\t') + string.count('\n') > 0:
        raise ValueError(INSTRUCTION_HAS_WHITESPACE)

def eof(fd):
    """Determine if end-of-file is reached for file fd."""
    b = fd.read(1)
    end = len(b) == 0
    if not end:
        curpos = fd.tell()
        fd.seek(curpos - 1)
    return end

def clear_globals_reload_modules(self):
        """Clears globals and reloads modules"""

        self.code_array.clear_globals()
        self.code_array.reload_modules()

        # Clear result cache
        self.code_array.result_cache.clear()

def raw_connection_from(engine_or_conn):
    """Extract a raw_connection and determine if it should be automatically closed.

    Only connections opened by this package will be closed automatically.
    """
    if hasattr(engine_or_conn, 'cursor'):
        return engine_or_conn, False
    if hasattr(engine_or_conn, 'connection'):
        return engine_or_conn.connection, False
    return engine_or_conn.raw_connection(), True

def selectnone(table, field, complement=False):
    """Select rows where the given field is `None`."""

    return select(table, field, lambda v: v is None, complement=complement)

def axes_off(ax):
    """Get rid of all axis ticks, lines, etc.
    """
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)

def is_defined(self, objtxt, force_import=False):
        """Return True if object is defined"""
        return self.interpreter.is_defined(objtxt, force_import)

def filter_query_string(query):
    """
        Return a version of the query string with the _e, _k and _s values
        removed.
    """
    return '&'.join([q for q in query.split('&')
        if not (q.startswith('_k=') or q.startswith('_e=') or q.startswith('_s'))])

def get_timezone() -> Tuple[datetime.tzinfo, str]:
    """Discover the current time zone and it's standard string representation (for source{d})."""
    dt = get_datetime_now().astimezone()
    tzstr = dt.strftime("%z")
    tzstr = tzstr[:-2] + ":" + tzstr[-2:]
    return dt.tzinfo, tzstr

def normalise_string(string):
    """ Strips trailing whitespace from string, lowercases it and replaces
        spaces with underscores
    """
    string = (string.strip()).lower()
    return re.sub(r'\W+', '_', string)

def transpose(table):
    """
    transpose matrix
    """
    t = []
    for i in range(0, len(table[0])):
        t.append([row[i] for row in table])
    return t

def is_same_shape(self, other_im, check_channels=False):
        """ Checks if two images have the same height and width (and optionally channels).

        Parameters
        ----------
        other_im : :obj:`Image`
            image to compare
        check_channels : bool
            whether or not to check equality of the channels

        Returns
        -------
        bool
            True if the images are the same shape, False otherwise
        """
        if self.height == other_im.height and self.width == other_im.width:
            if check_channels and self.channels != other_im.channels:
                return False
            return True
        return False

def is_identifier(string):
    """Check if string could be a valid python identifier

    :param string: string to be tested
    :returns: True if string can be a python identifier, False otherwise
    :rtype: bool
    """
    matched = PYTHON_IDENTIFIER_RE.match(string)
    return bool(matched) and not keyword.iskeyword(string)

def remove_hop_by_hop_headers(headers):
    """Remove all HTTP/1.1 "Hop-by-Hop" headers from a list or
    :class:`Headers` object.  This operation works in-place.

    .. versionadded:: 0.5

    :param headers: a list or :class:`Headers` object.
    """
    headers[:] = [
        (key, value) for key, value in headers if not is_hop_by_hop_header(key)
    ]

def coerce(self, value):
        """Convert from whatever is given to a list of scalars for the lookup_field."""
        if isinstance(value, dict):
            value = [value]
        if not isiterable_notstring(value):
            value = [value]
        return [coerce_single_instance(self.lookup_field, v) for v in value]

def rlognormal(mu, tau, size=None):
    """
    Return random lognormal variates.
    """

    return np.random.lognormal(mu, np.sqrt(1. / tau), size)

def on_train_end(self, logs):
        """ Print training time at end of training """
        duration = timeit.default_timer() - self.train_start
        print('done, took {:.3f} seconds'.format(duration))

def guess_title(basename):
    """ Attempt to guess the title from the filename """

    base, _ = os.path.splitext(basename)
    return re.sub(r'[ _-]+', r' ', base).title()

def log_no_newline(self, msg):
      """ print the message to the predefined log file without newline """
      self.print2file(self.logfile, False, False, msg)

def read_text_from_file(path: str) -> str:
    """ Reads text file contents """
    with open(path) as text_file:
        content = text_file.read()

    return content

def remove_duplicates(lst):
    """
    Emulate what a Python ``set()`` does, but keeping the element's order.
    """
    dset = set()
    return [l for l in lst if l not in dset and not dset.add(l)]

def computeDelaunayTriangulation(points):
    """ Takes a list of point objects (which must have x and y fields).
        Returns a list of 3-tuples: the indices of the points that form a
        Delaunay triangle.
    """
    siteList = SiteList(points)
    context  = Context()
    context.triangulate = True
    voronoi(siteList,context)
    return context.triangles

def is_string_dtype(arr_or_dtype):
    """
    Check whether the provided array or dtype is of the string dtype.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of the string dtype.

    Examples
    --------
    >>> is_string_dtype(str)
    True
    >>> is_string_dtype(object)
    True
    >>> is_string_dtype(int)
    False
    >>>
    >>> is_string_dtype(np.array(['a', 'b']))
    True
    >>> is_string_dtype(pd.Series([1, 2]))
    False
    """

    # TODO: gh-15585: consider making the checks stricter.
    def condition(dtype):
        return dtype.kind in ('O', 'S', 'U') and not is_period_dtype(dtype)
    return _is_dtype(arr_or_dtype, condition)

def to_str(obj):
    """Attempts to convert given object to a string object
    """
    if not isinstance(obj, str) and PY3 and isinstance(obj, bytes):
        obj = obj.decode('utf-8')
    return obj if isinstance(obj, string_types) else str(obj)

def dedupe_list(seq):
    """
    Utility function to remove duplicates from a list
    :param seq: The sequence (list) to deduplicate
    :return: A list with original duplicates removed
    """
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]

def smooth_array(array, amount=1):
    """

    Returns the nearest-neighbor (+/- amount) smoothed array.
    This does not modify the array or slice off the funny end points.

    """
    if amount==0: return array

    # we have to store the old values in a temp array to keep the
    # smoothing from affecting the smoothing
    new_array = _n.array(array)

    for n in range(len(array)):
        new_array[n] = smooth(array, n, amount)

    return new_array

def get_case_insensitive_dict_key(d: Dict, k: str) -> Optional[str]:
    """
    Within the dictionary ``d``, find a key that matches (in case-insensitive
    fashion) the key ``k``, and return it (or ``None`` if there isn't one).
    """
    for key in d.keys():
        if k.lower() == key.lower():
            return key
    return None

def sort_func(self, key):
        """Sorting logic for `Quantity` objects."""
        if key == self._KEYS.VALUE:
            return 'aaa'
        if key == self._KEYS.SOURCE:
            return 'zzz'
        return key

def _index_ordering(redshift_list):
        """

        :param redshift_list: list of redshifts
        :return: indexes in acending order to be evaluated (from z=0 to z=z_source)
        """
        redshift_list = np.array(redshift_list)
        sort_index = np.argsort(redshift_list)
        return sort_index

def cprint(string, fg=None, bg=None, end='\n', target=sys.stdout):
    """Print a colored string to the target handle.

    fg and bg specify foreground- and background colors, respectively. The
    remaining keyword arguments are the same as for Python's built-in print
    function. Colors are returned to their defaults before the function
    returns.

    """
    _color_manager.set_color(fg, bg)
    target.write(string + end)
    target.flush()  # Needed for Python 3.x
    _color_manager.set_defaults()

def indexes_equal(a: Index, b: Index) -> bool:
    """
    Are two indexes equal? Checks by comparing ``str()`` versions of them.
    (AM UNSURE IF THIS IS ENOUGH.)
    """
    return str(a) == str(b)

def is_readable(filename):
    """Check if file is a regular file and is readable."""
    return os.path.isfile(filename) and os.access(filename, os.R_OK)

def ln_norm(x, mu, sigma=1.0):
    """ Natural log of scipy norm function truncated at zero """
    return np.log(stats.norm(loc=mu, scale=sigma).pdf(x))

def ensure_dir(f):
    """ Ensure a a file exists and if not make the relevant path """
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

def redirect_output(fileobj):
    """Redirect standard out to file."""
    old = sys.stdout
    sys.stdout = fileobj
    try:
        yield fileobj
    finally:
        sys.stdout = old

def close_database_session(session):
    """Close connection with the database"""

    try:
        session.close()
    except OperationalError as e:
        raise DatabaseError(error=e.orig.args[1], code=e.orig.args[0])

def parse_cookies(self, req, name, field):
        """Pull the value from the cookiejar."""
        return core.get_value(req.COOKIES, name, field)

def unique(seq):
    """Return the unique elements of a collection even if those elements are
       unhashable and unsortable, like dicts and sets"""
    cleaned = []
    for each in seq:
        if each not in cleaned:
            cleaned.append(each)
    return cleaned

def save(self, fname):
        """ Saves the dictionary in json format
        :param fname: file to save to
        """
        with open(fname, 'wb') as f:
            json.dump(self, f)

def redirect_stdout(new_stdout):
    """Redirect the stdout

    Args:
        new_stdout (io.StringIO): New stdout to use instead
    """
    old_stdout, sys.stdout = sys.stdout, new_stdout
    try:
        yield None
    finally:
        sys.stdout = old_stdout

def command_py2to3(args):
    """
    Apply '2to3' tool (Python2 to Python3 conversion tool) to Python sources.
    """
    from lib2to3.main import main
    sys.exit(main("lib2to3.fixes", args=args.sources))

def _skip_section(self):
        """Skip a section"""
        self._last = self._f.readline()
        while len(self._last) > 0 and len(self._last[0].strip()) == 0:
            self._last = self._f.readline()

def get_git_branch(git_path='git'):
    """Returns the name of the current git branch
    """
    branch_match = call((git_path, 'rev-parse', '--symbolic-full-name', 'HEAD'))
    if branch_match == "HEAD":
        return None
    else:
        return os.path.basename(branch_match)

def check_auth(email, password):
    """Check if a username/password combination is valid.
    """
    try:
        user = User.get(User.email == email)
    except User.DoesNotExist:
        return False
    return password == user.password

def _read_date_from_string(str1):
    """
    Reads the date from a string in the format YYYY/MM/DD and returns
    :class: datetime.date
    """
    full_date = [int(x) for x in str1.split('/')]
    return datetime.date(full_date[0], full_date[1], full_date[2])

def find_le(a, x):
    """Find rightmost value less than or equal to x."""
    i = bs.bisect_right(a, x)
    if i: return i - 1
    raise ValueError

def write_line(self, line, count=1):
        """writes the line and count newlines after the line"""
        self.write(line)
        self.write_newlines(count)

def is_int(value):
    """Return `True` if ``value`` is an integer."""
    if isinstance(value, bool):
        return False
    try:
        int(value)
        return True
    except (ValueError, TypeError):
        return False

def ma(self):
        """Represent data as a masked array.

        The array is returned with column-first indexing, i.e. for a data file with
        columns X Y1 Y2 Y3 ... the array a will be a[0] = X, a[1] = Y1, ... .

        inf and nan are filtered via :func:`numpy.isfinite`.
        """
        a = self.array
        return numpy.ma.MaskedArray(a, mask=numpy.logical_not(numpy.isfinite(a)))

def isetdiff_flags(list1, list2):
    """
    move to util_iter
    """
    set2 = set(list2)
    return (item not in set2 for item in list1)

def text(value, encoding="utf-8", errors="strict"):
    """Convert a value to str on Python 3 and unicode on Python 2."""
    if isinstance(value, text_type):
        return value
    elif isinstance(value, bytes):
        return text_type(value, encoding, errors)
    else:
        return text_type(value)

def filter_dict(d, keys):
    """
    Creates a new dict from an existing dict that only has the given keys
    """
    return {k: v for k, v in d.items() if k in keys}

def _dt_to_epoch(dt):
        """Convert datetime to epoch seconds."""
        try:
            epoch = dt.timestamp()
        except AttributeError:  # py2
            epoch = (dt - datetime(1970, 1, 1)).total_seconds()
        return epoch

def sanitize_word(s):
    """Remove non-alphanumerical characters from metric word.
    And trim excessive underscores.
    """
    s = re.sub('[^\w-]+', '_', s)
    s = re.sub('__+', '_', s)
    return s.strip('_')

def web(host, port):
    """Start web application"""
    from .webserver.web import get_app
    get_app().run(host=host, port=port)

def valid_date(x: str) -> bool:
    """
    Retrun ``True`` if ``x`` is a valid YYYYMMDD date;
    otherwise return ``False``.
    """
    try:
        if x != dt.datetime.strptime(x, DATE_FORMAT).strftime(DATE_FORMAT):
            raise ValueError
        return True
    except ValueError:
        return False

def datetime_delta_to_ms(delta):
    """
    Given a datetime.timedelta object, return the delta in milliseconds
    """
    delta_ms = delta.days * 24 * 60 * 60 * 1000
    delta_ms += delta.seconds * 1000
    delta_ms += delta.microseconds / 1000
    delta_ms = int(delta_ms)
    return delta_ms

def reset(self):
		"""
		Resets the iterator to the start.

		Any remaining values in the current iteration are discarded.
		"""
		self.__iterator, self.__saved = itertools.tee(self.__saved)

def median_high(data):
    """Return the high median of data.

    When the number of data points is odd, the middle value is returned.
    When it is even, the larger of the two middle values is returned.

    """
    data = sorted(data)
    n = len(data)
    if n == 0:
        raise StatisticsError("no median for empty data")
    return data[n // 2]

def pop(self, index=-1):
		"""Remove and return the item at index."""
		value = self._list.pop(index)
		del self._dict[value]
		return value

def is_same_file (filename1, filename2):
    """Check if filename1 and filename2 point to the same file object.
    There can be false negatives, ie. the result is False, but it is
    the same file anyway. Reason is that network filesystems can create
    different paths to the same physical file.
    """
    if filename1 == filename2:
        return True
    if os.name == 'posix':
        return os.path.samefile(filename1, filename2)
    return is_same_filename(filename1, filename2)

def writeCSV(data, headers, csvFile):
  """Write data with column headers to a CSV."""
  with open(csvFile, "wb") as f:
    writer = csv.writer(f, delimiter=",")
    writer.writerow(headers)
    writer.writerows(data)

def DeleteLog() -> None:
        """Delete log file."""
        if os.path.exists(Logger.FileName):
            os.remove(Logger.FileName)

def truncate(value: Decimal, n_digits: int) -> Decimal:
    """Truncates a value to a number of decimals places"""
    return Decimal(math.trunc(value * (10 ** n_digits))) / (10 ** n_digits)

def join(mapping, bind, values):
    """ Merge all the strings. Put space between them. """
    return [' '.join([six.text_type(v) for v in values if v is not None])]

def clean(some_string, uppercase=False):
    """
    helper to clean up an input string
    """
    if uppercase:
        return some_string.strip().upper()
    else:
        return some_string.strip().lower()

def _count_leading_whitespace(text):
  """Returns the number of characters at the beginning of text that are whitespace."""
  idx = 0
  for idx, char in enumerate(text):
    if not char.isspace():
      return idx
  return idx + 1

def cos_sin_deg(deg):
    """Return the cosine and sin for the given angle
    in degrees, with special-case handling of multiples
    of 90 for perfect right angles
    """
    deg = deg % 360.0
    if deg == 90.0:
        return 0.0, 1.0
    elif deg == 180.0:
        return -1.0, 0
    elif deg == 270.0:
        return 0, -1.0
    rad = math.radians(deg)
    return math.cos(rad), math.sin(rad)

def __add__(self, other):
        """Handle the `+` operator."""
        return self._handle_type(other)(self.value + other.value)

def create_conda_env(sandbox_dir, env_name, dependencies, options=()):
    """
    Create a conda environment inside the current sandbox for the given list of dependencies and options.

    Parameters
    ----------
    sandbox_dir : str
    env_name : str
    dependencies : list
        List of conda specs
    options
        List of additional options to pass to conda.  Things like ["-c", "conda-forge"]

    Returns
    -------
    (env_dir, env_name)
    """

    env_dir = os.path.join(sandbox_dir, env_name)
    cmdline = ["conda", "create", "--yes", "--copy", "--quiet", "-p", env_dir] + list(options) + dependencies

    log.info("Creating conda environment: ")
    log.info("  command line: %s", cmdline)
    subprocess.check_call(cmdline, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    log.debug("Environment created")

    return env_dir, env_name

def is_same_shape(self, other_im, check_channels=False):
        """ Checks if two images have the same height and width (and optionally channels).

        Parameters
        ----------
        other_im : :obj:`Image`
            image to compare
        check_channels : bool
            whether or not to check equality of the channels

        Returns
        -------
        bool
            True if the images are the same shape, False otherwise
        """
        if self.height == other_im.height and self.width == other_im.width:
            if check_channels and self.channels != other_im.channels:
                return False
            return True
        return False

def _get_pretty_string(obj):
    """Return a prettier version of obj

    Parameters
    ----------
    obj : object
        Object to pretty print

    Returns
    -------
    s : str
        Pretty print object repr
    """
    sio = StringIO()
    pprint.pprint(obj, stream=sio)
    return sio.getvalue()
