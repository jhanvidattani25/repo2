
def get_iter_string_reader(stdin):
    """ return an iterator that returns a chunk of a string every time it is
    called.  notice that even though bufsize_type might be line buffered, we're
    not doing any line buffering here.  that's because our StreamBufferer
    handles all buffering.  we just need to return a reasonable-sized chunk. """
    bufsize = 1024
    iter_str = (stdin[i:i + bufsize] for i in range(0, len(stdin), bufsize))
    return get_iter_chunk_reader(iter_str)

def retry_on_signal(function):
    """Retries function until it doesn't raise an EINTR error"""
    while True:
        try:
            return function()
        except EnvironmentError, e:
            if e.errno != errno.EINTR:
                raise

def head(filename, n=10):
    """ prints the top `n` lines of a file """
    with freader(filename) as fr:
        for _ in range(n):
            print(fr.readline().strip())

def isInteractive():
    """
    A basic check of if the program is running in interactive mode
    """
    if sys.stdout.isatty() and os.name != 'nt':
        #Hopefully everything but ms supports '\r'
        try:
            import threading
        except ImportError:
            return False
        else:
            return True
    else:
        return False

def _read_date_from_string(str1):
    """
    Reads the date from a string in the format YYYY/MM/DD and returns
    :class: datetime.date
    """
    full_date = [int(x) for x in str1.split('/')]
    return datetime.date(full_date[0], full_date[1], full_date[2])

def process_kill(pid, sig=None):
    """Send signal to process.
    """
    sig = sig or signal.SIGTERM
    os.kill(pid, sig)

def ReadManyFromPath(filepath):
  """Reads a Python object stored in a specified YAML file.

  Args:
    filepath: A filepath to the YAML file.

  Returns:
    A Python data structure corresponding to the YAML in the given file.
  """
  with io.open(filepath, mode="r", encoding="utf-8") as filedesc:
    return ReadManyFromFile(filedesc)

def mouse_get_pos():
    """

    :return:
    """
    p = POINT()
    AUTO_IT.AU3_MouseGetPos(ctypes.byref(p))
    return p.x, p.y

def memory_usage(method):
  """Log memory usage before and after a method."""
  def wrapper(*args, **kwargs):
    logging.info('Memory before method %s is %s.',
                 method.__name__, runtime.memory_usage().current())
    result = method(*args, **kwargs)
    logging.info('Memory after method %s is %s',
                 method.__name__, runtime.memory_usage().current())
    return result
  return wrapper

def getpackagepath():
    """
     *Get the root path for this python package - used in unit testing code*
    """
    moduleDirectory = os.path.dirname(__file__)
    packagePath = os.path.dirname(__file__) + "/../"

    return packagePath

def tuplize(nested):
  """Recursively converts iterables into tuples.

  Args:
    nested: A nested structure of items and iterables.

  Returns:
    A nested structure of items and tuples.
  """
  if isinstance(nested, str):
    return nested
  try:
    return tuple(map(tuplize, nested))
  except TypeError:
    return nested

def _get_column_types(self, data):
        """Get a list of the data types for each column in *data*."""
        columns = list(zip_longest(*data))
        return [self._get_column_type(column) for column in columns]

def bytesize(arr):
    """
    Returns the memory byte size of a Numpy array as an integer.
    """
    byte_size = np.prod(arr.shape) * np.dtype(arr.dtype).itemsize
    return byte_size

def get(self, key):  
        """ get a set of keys from redis """
        res = self.connection.get(key)
        print(res)
        return res

def _crop_list_to_size(l, size):
    """Make a list a certain size"""
    for x in range(size - len(l)):
        l.append(False)
    for x in range(len(l) - size):
        l.pop()
    return l

def connect(self):
        """
        Connects to publisher
        """
        self.client = redis.Redis(
            host=self.host, port=self.port, password=self.password)

def read_raw(data_path):
    """
    Parameters
    ----------
    data_path : str
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data

def lpush(self, key, *args):
        """Emulate lpush."""
        redis_list = self._get_list(key, 'LPUSH', create=True)

        # Creates the list at this key if it doesn't exist, and appends args to its beginning
        args_reversed = [self._encode(arg) for arg in args]
        args_reversed.reverse()
        updated_list = args_reversed + redis_list
        self.redis[self._encode(key)] = updated_list

        # Return the length of the list after the push operation
        return len(updated_list)

def _read_group_h5(filename, groupname):
    """Return group content.

    Args:
        filename (:class:`pathlib.Path`): path of hdf5 file.
        groupname (str): name of group to read.
    Returns:
        :class:`numpy.array`: content of group.
    """
    with h5py.File(filename, 'r') as h5f:
        data = h5f[groupname][()]
    return data

def match_files(files, pattern: Pattern):
    """Yields file name if matches a regular expression pattern."""

    for name in files:
        if re.match(pattern, name):
            yield name

def _load_ngram(name):
    """Dynamically import the python module with the ngram defined as a dictionary.
    Since bigger ngrams are large files its wasteful to always statically import them if they're not used.
    """
    module = importlib.import_module('lantern.analysis.english_ngrams.{}'.format(name))
    return getattr(module, name)

def CleanseComments(line):
  """Removes //-comments and single-line C-style /* */ comments.

  Args:
    line: A line of C++ source.

  Returns:
    The line with single-line comments removed.
  """
  commentpos = line.find('//')
  if commentpos != -1 and not IsCppString(line[:commentpos]):
    line = line[:commentpos].rstrip()
  # get rid of /* ... */
  return _RE_PATTERN_CLEANSE_LINE_C_COMMENTS.sub('', line)

def load(self, path):
        """Load the pickled model weights."""
        with io.open(path, 'rb') as fin:
            self.weights = pickle.load(fin)

def find_number(regex, s):
    """Find a number using a given regular expression.
    If the string cannot be found, returns None.
    The regex should contain one matching group, 
    as only the result of the first group is returned.
    The group should only contain numeric characters ([0-9]+).
    
    s - The string to search.
    regex - A string containing the regular expression.
    
    Returns an integer or None.
    """
    result = find_string(regex, s)
    if result is None:
        return None
    return int(result)

def getScriptLocation():
	"""Helper function to get the location of a Python file."""
	location = os.path.abspath("./")
	if __file__.rfind("/") != -1:
		location = __file__[:__file__.rfind("/")]
	return location

def is_valid_regex(string):
    """
    Checks whether the re module can compile the given regular expression.

    Parameters
    ----------
    string: str

    Returns
    -------
    boolean
    """
    try:
        re.compile(string)
        is_valid = True
    except re.error:
        is_valid = False
    return is_valid

def split_strings_in_list_retain_spaces(orig_list):
    """
    Function to split every line in a list, and retain spaces for a rejoin
    :param orig_list: Original list
    :return:
        A List with split lines

    """
    temp_list = list()
    for line in orig_list:
        line_split = __re.split(r'(\s+)', line)
        temp_list.append(line_split)

    return temp_list

def upcaseTokens(s,l,t):
    """Helper parse action to convert tokens to upper case."""
    return [ tt.upper() for tt in map(_ustr,t) ]

def detach_index(self, name):
        """

        :param name:
        :return:
        """
        assert type(name) == str

        if name in self._indexes:
            del self._indexes[name]

def price_rounding(price, decimals=2):
    """Takes a decimal price and rounds to a number of decimal places"""
    try:
        exponent = D('.' + decimals * '0')
    except InvalidOperation:
        # Currencies with no decimal places, ex. JPY, HUF
        exponent = D()
    return price.quantize(exponent, rounding=ROUND_UP)

def strip_spaces(value, sep=None, join=True):
    """Cleans trailing whitespaces and replaces also multiple whitespaces with a single space."""
    value = value.strip()
    value = [v.strip() for v in value.split(sep)]
    join_sep = sep or ' '
    return join_sep.join(value) if join else value

def gaussian_kernel(gstd):
    """Generate odd sized truncated Gaussian

    The generated filter kernel has a cutoff at $3\sigma$
    and is normalized to sum to 1

    Parameters
    -------------
    gstd : float
            Standard deviation of filter

    Returns
    -------------
    g : ndarray
            Array with kernel coefficients
    """
    Nc = np.ceil(gstd*3)*2+1
    x = np.linspace(-(Nc-1)/2,(Nc-1)/2,Nc,endpoint=True)
    g = np.exp(-.5*((x/gstd)**2))
    g = g/np.sum(g)

    return g

def plot_dot_graph(graph, filename=None):
    """
    Plots a graph in graphviz dot notation.

    :param graph: the dot notation graph
    :type graph: str
    :param filename: the (optional) file to save the generated plot to. The extension determines the file format.
    :type filename: str
    """
    if not plot.pygraphviz_available:
        logger.error("Pygraphviz is not installed, cannot generate graph plot!")
        return
    if not plot.PIL_available:
        logger.error("PIL is not installed, cannot display graph plot!")
        return

    agraph = AGraph(graph)
    agraph.layout(prog='dot')
    if filename is None:
        filename = tempfile.mktemp(suffix=".png")
    agraph.draw(filename)
    image = Image.open(filename)
    image.show()

def unique(list):
    """ Returns a copy of the list without duplicates.
    """
    unique = []; [unique.append(x) for x in list if x not in unique]
    return unique

def is_a_sequence(var, allow_none=False):
    """ Returns True if var is a list or a tuple (but not a string!)
    """
    return isinstance(var, (list, tuple)) or (var is None and allow_none)

def strip_html(string, keep_tag_content=False):
    """
    Remove html code contained into the given string.

    :param string: String to manipulate.
    :type string: str
    :param keep_tag_content: True to preserve tag content, False to remove tag and its content too (default).
    :type keep_tag_content: bool
    :return: String with html removed.
    :rtype: str
    """
    r = HTML_TAG_ONLY_RE if keep_tag_content else HTML_RE
    return r.sub('', string)

def get_plugin_icon(self):
        """Return widget icon"""
        path = osp.join(self.PLUGIN_PATH, self.IMG_PATH)
        return ima.icon('pylint', icon_path=path)

def remove_last_entry(self):
        """Remove the last NoteContainer in the Bar."""
        self.current_beat -= 1.0 / self.bar[-1][1]
        self.bar = self.bar[:-1]
        return self.current_beat

def generate_header(headerfields, oldheader, group_by_field):
    """Returns a header as a list, ready to write to TSV file"""
    fieldtypes = ['peptidefdr', 'peptidepep', 'nopsms', 'proteindata',
                  'precursorquant', 'isoquant']
    return generate_general_header(headerfields, fieldtypes,
                                   peptabledata.HEADER_PEPTIDE, oldheader,
                                   group_by_field)

def remove_last_entry(self):
        """Remove the last NoteContainer in the Bar."""
        self.current_beat -= 1.0 / self.bar[-1][1]
        self.bar = self.bar[:-1]
        return self.current_beat

def str2int(num, radix=10, alphabet=BASE85):
    """helper function for quick base conversions from strings to integers"""
    return NumConv(radix, alphabet).str2int(num)

def slugify(string):
    """
    Removes non-alpha characters, and converts spaces to hyphens. Useful for making file names.


    Source: http://stackoverflow.com/questions/5574042/string-slugification-in-python
    """
    string = re.sub('[^\w .-]', '', string)
    string = string.replace(" ", "-")
    return string

def is_int(value):
    """Return `True` if ``value`` is an integer."""
    if isinstance(value, bool):
        return False
    try:
        int(value)
        return True
    except (ValueError, TypeError):
        return False

def ms_to_datetime(ms):
    """
    Converts a millisecond accuracy timestamp to a datetime
    """
    dt = datetime.datetime.utcfromtimestamp(ms / 1000)
    return dt.replace(microsecond=(ms % 1000) * 1000).replace(tzinfo=pytz.utc)

def _npiter(arr):
    """Wrapper for iterating numpy array"""
    for a in np.nditer(arr, flags=["refs_ok"]):
        c = a.item()
        if c is not None:
            yield c

def strip_accents(string):
    """
    Strip all the accents from the string
    """
    return u''.join(
        (character for character in unicodedata.normalize('NFD', string)
         if unicodedata.category(character) != 'Mn'))

def to_capitalized_camel_case(snake_case_string):
    """
    Convert a string from snake case to camel case with the first letter capitalized. For example, "some_var"
    would become "SomeVar".

    :param snake_case_string: Snake-cased string to convert to camel case.
    :returns: Camel-cased version of snake_case_string.
    """
    parts = snake_case_string.split('_')
    return ''.join([i.title() for i in parts])

def drop_bad_characters(text):
    """Takes a text and drops all non-printable and non-ascii characters and
    also any whitespace characters that aren't space.

    :arg str text: the text to fix

    :returns: text with all bad characters dropped

    """
    # Strip all non-ascii and non-printable characters
    text = ''.join([c for c in text if c in ALLOWED_CHARS])
    return text

def to_linspace(self):
        """
        convert from arange to linspace
        """
        num = int((self.stop-self.start)/(self.step))
        return Linspace(self.start, self.stop-self.step, num)

def flatten(lis):
    """Given a list, possibly nested to any level, return it flattened."""
    new_lis = []
    for item in lis:
        if isinstance(item, collections.Sequence) and not isinstance(item, basestring):
            new_lis.extend(flatten(item))
        else:
            new_lis.append(item)
    return new_lis

def trim_trailing_silence(self):
        """Trim the trailing silence of the pianoroll."""
        length = self.get_active_length()
        self.pianoroll = self.pianoroll[:length]

def negate(self):
        """Reverse the range"""
        self.from_value, self.to_value = self.to_value, self.from_value
        self.include_lower, self.include_upper = self.include_upper, self.include_lower

def remove_bad(string):
    """
    remove problem characters from string
    """
    remove = [':', ',', '(', ')', ' ', '|', ';', '\'']
    for c in remove:
        string = string.replace(c, '_')
    return string

def clean(some_string, uppercase=False):
    """
    helper to clean up an input string
    """
    if uppercase:
        return some_string.strip().upper()
    else:
        return some_string.strip().lower()

def urlize_twitter(text):
    """
    Replace #hashtag and @username references in a tweet with HTML text.
    """
    html = TwitterText(text).autolink.auto_link()
    return mark_safe(html.replace(
        'twitter.com/search?q=', 'twitter.com/search/realtime/'))

def inpaint(self):
        """ Replace masked-out elements in an array using an iterative image inpainting algorithm. """

        import inpaint
        filled = inpaint.replace_nans(np.ma.filled(self.raster_data, np.NAN).astype(np.float32), 3, 0.01, 2)
        self.raster_data = np.ma.masked_invalid(filled)

def cleanup(self):
        """Forcefully delete objects from memory

        In an ideal world, this shouldn't be necessary. Garbage
        collection guarantees that anything without reference
        is automatically removed.

        However, because this application is designed to be run
        multiple times from the same interpreter process, extra
        case must be taken to ensure there are no memory leaks.

        Explicitly deleting objects shines a light on where objects
        may still be referenced in the form of an error. No errors
        means this was uneccesary, but that's ok.

        """

        for instance in self.context:
            del(instance)

        for plugin in self.plugins:
            del(plugin)

def match(string, patterns):
    """Given a string return true if it matches the supplied list of
    patterns.

    Parameters
    ----------
    string : str
        The string to be matched.
    patterns : None or [pattern, ...]
        The series of regular expressions to attempt to match.
    """
    if patterns is None:
        return True
    else:
        return any(re.match(pattern, string)
                   for pattern in patterns)

def _remove_empty_items(d, required):
  """Return a new dict with any empty items removed.

  Note that this is not a deep check. If d contains a dictionary which
  itself contains empty items, those are never checked.

  This method exists to make to_serializable() functions cleaner.
  We could revisit this some day, but for now, the serialized objects are
  stripped of empty values to keep the output YAML more compact.

  Args:
    d: a dictionary
    required: list of required keys (for example, TaskDescriptors always emit
      the "task-id", even if None)

  Returns:
    A dictionary with empty items removed.
  """

  new_dict = {}
  for k, v in d.items():
    if k in required:
      new_dict[k] = v
    elif isinstance(v, int) or v:
      # "if v" would suppress emitting int(0)
      new_dict[k] = v

  return new_dict

def file_matches(filename, patterns):
    """Does this filename match any of the patterns?"""
    return any(fnmatch.fnmatch(filename, pat) for pat in patterns)

def replace_all(text, dic):
    """Takes a string and dictionary. replaces all occurrences of i with j"""

    for i, j in dic.iteritems():
        text = text.replace(i, j)
    return text

def _replace_nan(a, val):
    """
    replace nan in a by val, and returns the replaced array and the nan
    position
    """
    mask = isnull(a)
    return where_method(val, mask, a), mask

def align_file_position(f, size):
    """ Align the position in the file to the next block of specified size """
    align = (size - 1) - (f.tell() % size)
    f.seek(align, 1)

def set_property(self, key, value):
        """
        Update only one property in the dict
        """
        self.properties[key] = value
        self.sync_properties()

def __next__(self, reward, ask_id, lbl):
        """For Python3 compatibility of generator."""
        return self.next(reward, ask_id, lbl)

def handle_whitespace(text):
    r"""Handles whitespace cleanup.

    Tabs are "smartly" retabbed (see sub_retab). Lines that contain
    only whitespace are truncated to a single newline.
    """
    text = re_retab.sub(sub_retab, text)
    text = re_whitespace.sub('', text).strip()
    return text

def normalize_array(lst):
    """Normalizes list

    :param lst: Array of floats
    :return: Normalized (in [0, 1]) input array
    """
    np_arr = np.array(lst)
    x_normalized = np_arr / np_arr.max(axis=0)
    return list(x_normalized)

def list_replace(subject_list, replacement, string):
    """
    To replace a list of items by a single replacement
    :param subject_list: list
    :param replacement: string
    :param string: string
    :return: string
    """
    for s in subject_list:
        string = string.replace(s, replacement)
    return string

def normalize(data):
    """Normalize the data to be in the [0, 1] range.

    :param data:
    :return: normalized data
    """
    out_data = data.copy()

    for i, sample in enumerate(out_data):
        out_data[i] /= sum(out_data[i])

    return out_data

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

def normalize_array(lst):
    """Normalizes list

    :param lst: Array of floats
    :return: Normalized (in [0, 1]) input array
    """
    np_arr = np.array(lst)
    x_normalized = np_arr / np_arr.max(axis=0)
    return list(x_normalized)

def restart_program():
    """
    DOES NOT WORK WELL WITH MOPIDY
    Hack from
    https://www.daniweb.com/software-development/python/code/260268/restart-your-python-program
    to support updating the settings, since mopidy is not able to do that yet
    Restarts the current program
    Note: this function does not return. Any cleanup action (like
    saving data) must be done before calling this function
    """

    python = sys.executable
    os.execl(python, python, * sys.argv)

def first_unique_char(s):
    """
    :type s: str
    :rtype: int
    """
    if (len(s) == 1):
        return 0
    ban = []
    for i in range(len(s)):
        if all(s[i] != s[k] for k in range(i + 1, len(s))) == True and s[i] not in ban:
            return i
        else:
            ban.append(s[i])
    return -1

def http_request_json(*args, **kwargs):
    """

    See: http_request
    """
    ret, status = http_request(*args, **kwargs)
    return json.loads(ret), status

def one_hot_encoding(input_tensor, num_labels):
    """ One-hot encode labels from input """
    xview = input_tensor.view(-1, 1).to(torch.long)

    onehot = torch.zeros(xview.size(0), num_labels, device=input_tensor.device, dtype=torch.float)
    onehot.scatter_(1, xview, 1)
    return onehot.view(list(input_tensor.shape) + [-1])

def save_session_to_file(self, sessionfile):
        """Not meant to be used directly, use :meth:`Instaloader.save_session_to_file`."""
        pickle.dump(requests.utils.dict_from_cookiejar(self._session.cookies), sessionfile)

def flat(l):
    """
Returns the flattened version of a '2D' list.  List-correlate to the a.flat()
method of NumPy arrays.

Usage:    flat(l)
"""
    newl = []
    for i in range(len(l)):
        for j in range(len(l[i])):
            newl.append(l[i][j])
    return newl

def do_serial(self, p):
		"""Set the serial port, e.g.: /dev/tty.usbserial-A4001ib8"""
		try:
			self.serial.port = p
			self.serial.open()
			print 'Opening serial port: %s' % p
		except Exception, e:
			print 'Unable to open serial port: %s' % p

def do_restart(self, line):
        """
        Attempt to restart the bot.
        """
        self.bot._frame = 0
        self.bot._namespace.clear()
        self.bot._namespace.update(self.bot._initial_namespace)

def ReadTif(tifFile):
        """Reads a tif file to a 2D NumPy array"""
        img = Image.open(tifFile)
        img = np.array(img)
        return img

def _fill(self):
    """Advance the iterator without returning the old head."""
    try:
      self._head = self._iterable.next()
    except StopIteration:
      self._head = None

def do_serial(self, p):
		"""Set the serial port, e.g.: /dev/tty.usbserial-A4001ib8"""
		try:
			self.serial.port = p
			self.serial.open()
			print 'Opening serial port: %s' % p
		except Exception, e:
			print 'Unable to open serial port: %s' % p

def get_previous(self):
        """Get the billing cycle prior to this one. May return None"""
        return BillingCycle.objects.filter(date_range__lt=self.date_range).order_by('date_range').last()

def _repr(obj):
    """Show the received object as precise as possible."""
    vals = ", ".join("{}={!r}".format(
        name, getattr(obj, name)) for name in obj._attribs)
    if vals:
        t = "{}(name={}, {})".format(obj.__class__.__name__, obj.name, vals)
    else:
        t = "{}(name={})".format(obj.__class__.__name__, obj.name)
    return t

def urlencoded(body, charset='ascii', **kwargs):
    """Converts query strings into native Python objects"""
    return parse_query_string(text(body, charset=charset), False)

def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return F.pad(img, self.padding, self.fill, self.padding_mode)

def INIT_LIST_EXPR(self, cursor):
        """Returns a list of literal values."""
        values = [self.parse_cursor(child)
                  for child in list(cursor.get_children())]
        return values

def parse(filename):
    """Parse ASDL from the given file and return a Module node describing it."""
    with open(filename) as f:
        parser = ASDLParser()
        return parser.parse(f.read())

def new_figure_manager_given_figure(num, figure):
    """
    Create a new figure manager instance for the given figure.
    """
    fig = figure
    frame = FigureFrameWx(num, fig)
    figmgr = frame.get_figure_manager()
    if matplotlib.is_interactive():
        figmgr.frame.Show()

    return figmgr

def seconds(num):
    """
    Pause for this many seconds
    """
    now = pytime.time()
    end = now + num
    until(end)

def software_fibonacci(n):
    """ a normal old python function to return the Nth fibonacci number. """
    a, b = 0, 1
    for i in range(n):
        a, b = b, a + b
    return a

def rand_elem(seq, n=None):
    """returns a random element from seq n times. If n is None, it continues indefinitly"""
    return map(random.choice, repeat(seq, n) if n is not None else repeat(seq))

def finish_plot():
    """Helper for plotting."""
    plt.legend()
    plt.grid(color='0.7')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def rgba_bytes_tuple(self, x):
        """Provides the color corresponding to value `x` in the
        form of a tuple (R,G,B,A) with int values between 0 and 255.
        """
        return tuple(int(u*255.9999) for u in self.rgba_floats_tuple(x))

def rotateImage(img, angle):
    """

    querries scipy.ndimage.rotate routine
    :param img: image to be rotated
    :param angle: angle to be rotated (radian)
    :return: rotated image
    """
    imgR = scipy.ndimage.rotate(img, angle, reshape=False)
    return imgR

def print_out(self, *lst):
      """ Print list of strings to the predefined stdout. """
      self.print2file(self.stdout, True, True, *lst)

def round_to_n(x, n):
    """
    Round to sig figs
    """
    return round(x, -int(np.floor(np.log10(x))) + (n - 1))

def handle_exception(error):
        """Simple method for handling exceptions raised by `PyBankID`.

        :param flask_pybankid.FlaskPyBankIDError error: The exception to handle.
        :return: The exception represented as a dictionary.
        :rtype: dict

        """
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

def fail_print(error):
    """Print an error in red text.
    Parameters
        error (HTTPError)
            Error object to print.
    """
    print(COLORS.fail, error.message, COLORS.end)
    print(COLORS.fail, error.errors, COLORS.end)

def round_figures(x, n):
    """Returns x rounded to n significant figures."""
    return round(x, int(n - math.ceil(math.log10(abs(x)))))

def _prt_line_detail(self, prt, line, lnum=""):
        """Print each field and its value."""
        data = zip(self.flds, line.split('\t'))
        txt = ["{:2}) {:13} {}".format(i, hdr, val) for i, (hdr, val) in enumerate(data)]
        prt.write("{LNUM}\n{TXT}\n".format(LNUM=lnum, TXT='\n'.join(txt)))

def round_to_float(number, precision):
    """Round a float to a precision"""
    rounded = Decimal(str(floor((number + precision / 2) // precision))
                      ) * Decimal(str(precision))
    return float(rounded)

def show_partitioning(rdd, show=True):
    """Seems to be significantly more expensive on cluster than locally"""
    if show:
        partitionCount = rdd.getNumPartitions()
        try:
            valueCount = rdd.countApprox(1000, confidence=0.50)
        except:
            valueCount = -1
        try:
            name = rdd.name() or None
        except:
            pass
        name = name or "anonymous"
        logging.info("For RDD %s, there are %d partitions with on average %s values" % 
                     (name, partitionCount, int(valueCount/float(partitionCount))))

def managepy(cmd, extra=None):
    """Run manage.py using this component's specific Django settings"""

    extra = extra.split() if extra else []
    run_django_cli(['invoke', cmd] + extra)

def parse(self, s):
        """
        Parses a date string formatted like ``YYYY-MM-DD``.
        """
        return datetime.datetime.strptime(s, self.date_format).date()

def start():
    """Starts the web server."""
    global app
    bottle.run(app, host=conf.WebHost, port=conf.WebPort,
               debug=conf.WebAutoReload, reloader=conf.WebAutoReload,
               quiet=conf.WebQuiet)

def _cumprod(l):
  """Cumulative product of a list.

  Args:
    l: a list of integers
  Returns:
    a list with one more element (starting with 1)
  """
  ret = [1]
  for item in l:
    ret.append(ret[-1] * item)
  return ret

def server(port):
    """Start the Django dev server."""
    args = ['python', 'manage.py', 'runserver']
    if port:
        args.append(port)
    run.main(args)

def camel_case(self, snake_case):
        """ Convert snake case to camel case """
        components = snake_case.split('_')
        return components[0] + "".join(x.title() for x in components[1:])

def fixpath(path):
    """Uniformly format a path."""
    return os.path.normpath(os.path.realpath(os.path.expanduser(path)))

def input(self, prompt, default=None, show_default=True):
        """Provide a command prompt."""
        return click.prompt(prompt, default=default, show_default=show_default)

def replacing_symlink(source, link_name):
    """Create symlink that overwrites any existing target.
    """
    with make_tmp_name(link_name) as tmp_link_name:
        os.symlink(source, tmp_link_name)
        replace_file_or_dir(link_name, tmp_link_name)

def bitsToString(arr):
  """Returns a string representing a numpy array of 0's and 1's"""
  s = array('c','.'*len(arr))
  for i in xrange(len(arr)):
    if arr[i] == 1:
      s[i]='*'
  return s

def save(variable, filename):
    """Save variable on given path using Pickle
    
    Args:
        variable: what to save
        path (str): path of the output
    """
    fileObj = open(filename, 'wb')
    pickle.dump(variable, fileObj)
    fileObj.close()

def home_lib(home):
    """Return the lib dir under the 'home' installation scheme"""
    if hasattr(sys, 'pypy_version_info'):
        lib = 'site-packages'
    else:
        lib = os.path.join('lib', 'python')
    return os.path.join(home, lib)

def draw(graph, fname):
    """Draw a graph and save it into a file"""
    ag = networkx.nx_agraph.to_agraph(graph)
    ag.draw(fname, prog='dot')

def _quit(self, *args):
        """ quit crash """
        self.logger.warn('Bye!')
        sys.exit(self.exit())

def _std(self,x):
        """
        Compute standard deviation with ddof degrees of freedom
        """
        return np.nanstd(x.values,ddof=self._ddof)

def _re_raise_as(NewExc, *args, **kw):
    """Raise a new exception using the preserved traceback of the last one."""
    etype, val, tb = sys.exc_info()
    raise NewExc(*args, **kw), None, tb

def aug_sysargv(cmdstr):
    """ DEBUG FUNC modify argv to look like you ran a command """
    import shlex
    argv = shlex.split(cmdstr)
    sys.argv.extend(argv)

def isInteractive():
    """
    A basic check of if the program is running in interactive mode
    """
    if sys.stdout.isatty() and os.name != 'nt':
        #Hopefully everything but ms supports '\r'
        try:
            import threading
        except ImportError:
            return False
        else:
            return True
    else:
        return False

def rand_elem(seq, n=None):
    """returns a random element from seq n times. If n is None, it continues indefinitly"""
    return map(random.choice, repeat(seq, n) if n is not None else repeat(seq))

def cpp_prog_builder(build_context, target):
    """Build a C++ binary executable"""
    yprint(build_context.conf, 'Build CppProg', target)
    workspace_dir = build_context.get_workspace('CppProg', target.name)
    build_cpp(build_context, target, target.compiler_config, workspace_dir)

def read_dict_from_file(file_path):
    """
    Read a dictionary of strings from a file
    """
    with open(file_path) as file:
        lines = file.read().splitlines()

    obj = {}
    for line in lines:
        key, value = line.split(':', maxsplit=1)
        obj[key] = eval(value)

    return obj

def CleanseComments(line):
  """Removes //-comments and single-line C-style /* */ comments.

  Args:
    line: A line of C++ source.

  Returns:
    The line with single-line comments removed.
  """
  commentpos = line.find('//')
  if commentpos != -1 and not IsCppString(line[:commentpos]):
    line = line[:commentpos].rstrip()
  # get rid of /* ... */
  return _RE_PATTERN_CLEANSE_LINE_C_COMMENTS.sub('', line)

def read_dict_from_file(file_path):
    """
    Read a dictionary of strings from a file
    """
    with open(file_path) as file:
        lines = file.read().splitlines()

    obj = {}
    for line in lines:
        key, value = line.split(':', maxsplit=1)
        obj[key] = eval(value)

    return obj

def get_login_credentials(args):
  """
    Gets the login credentials from the user, if not specified while invoking
    the script.
    @param args: arguments provided to the script.
    """
  if not args.username:
    args.username = raw_input("Enter Username: ")
  if not args.password:
    args.password = getpass.getpass("Enter Password: ")

def read_dict_from_file(file_path):
    """
    Read a dictionary of strings from a file
    """
    with open(file_path) as file:
        lines = file.read().splitlines()

    obj = {}
    for line in lines:
        key, value = line.split(':', maxsplit=1)
        obj[key] = eval(value)

    return obj

def timespan(start_time):
    """Return time in milliseconds from start_time"""

    timespan = datetime.datetime.now() - start_time
    timespan_ms = timespan.total_seconds() * 1000
    return timespan_ms

def get_code(module):
    """
    Compile and return a Module's code object.
    """
    fp = open(module.path)
    try:
        return compile(fp.read(), str(module.name), 'exec')
    finally:
        fp.close()

def selectnotnone(table, field, complement=False):
    """Select rows where the given field is not `None`."""

    return select(table, field, lambda v: v is not None,
                  complement=complement)

def selectnotnone(table, field, complement=False):
    """Select rows where the given field is not `None`."""

    return select(table, field, lambda v: v is not None,
                  complement=complement)

def read_key(self, key, bucket_name=None):
        """
        Reads a key from S3

        :param key: S3 key that will point to the file
        :type key: str
        :param bucket_name: Name of the bucket in which the file is stored
        :type bucket_name: str
        """

        obj = self.get_key(key, bucket_name)
        return obj.get()['Body'].read().decode('utf-8')

def selectnotin(table, field, value, complement=False):
    """Select rows where the given field is not a member of the given value."""

    return select(table, field, lambda v: v not in value,
                  complement=complement)

def wget(url):
    """
    Download the page into a string
    """
    import urllib.parse
    request = urllib.request.urlopen(url)
    filestring = request.read()
    return filestring

def get_last(self, table=None):
        """Just the last entry."""
        if table is None: table = self.main_table
        query = 'SELECT * FROM "%s" ORDER BY ROWID DESC LIMIT 1;' % table
        return self.own_cursor.execute(query).fetchone()

def get_active_window(self):
        """
        The current active :class:`.Window`.
        """
        app = get_app()

        try:
            return self._active_window_for_cli[app]
        except KeyError:
            self._active_window_for_cli[app] = self._last_active_window or self.windows[0]
            return self.windows[0]

def date(start, end):
    """Get a random date between two dates"""

    stime = date_to_timestamp(start)
    etime = date_to_timestamp(end)

    ptime = stime + random.random() * (etime - stime)

    return datetime.date.fromtimestamp(ptime)

def load_yaml(filepath):
    """Convenience function for loading yaml-encoded data from disk."""
    with open(filepath) as f:
        txt = f.read()
    return yaml.load(txt)

def get_randomized_guid_sample(self, item_count):
        """ Fetch a subset of randomzied GUIDs from the whitelist """
        dataset = self.get_whitelist()
        random.shuffle(dataset)
        return dataset[:item_count]

def main():
    """Ideally we shouldn't lose the first second of events"""
    time.sleep(1)
    with Input() as input_generator:
        for e in input_generator:
            print(repr(e))

def selectin(table, field, value, complement=False):
    """Select rows where the given field is a member of the given value."""

    return select(table, field, lambda v: v in value,
                  complement=complement)

def redirect_output(fileobj):
    """Redirect standard out to file."""
    old = sys.stdout
    sys.stdout = fileobj
    try:
        yield fileobj
    finally:
        sys.stdout = old

def process_kill(pid, sig=None):
    """Send signal to process.
    """
    sig = sig or signal.SIGTERM
    os.kill(pid, sig)

def append_pdf(input_pdf: bytes, output_writer: PdfFileWriter):
    """
    Appends a PDF to a pyPDF writer. Legacy interface.
    """
    append_memory_pdf_to_writer(input_pdf=input_pdf,
                                writer=output_writer)

def _send(self, data):
        """Send data to statsd."""
        if not self._sock:
            self.connect()
        self._do_send(data)

def type(self):
        """Returns type of the data for the given FeatureType."""
        if self is FeatureType.TIMESTAMP:
            return list
        if self is FeatureType.BBOX:
            return BBox
        return dict

def send(self, data):
        """
        Send data to the child process through.
        """
        self.stdin.write(data)
        self.stdin.flush()

def strip_html(string, keep_tag_content=False):
    """
    Remove html code contained into the given string.

    :param string: String to manipulate.
    :type string: str
    :param keep_tag_content: True to preserve tag content, False to remove tag and its content too (default).
    :type keep_tag_content: bool
    :return: String with html removed.
    :rtype: str
    """
    r = HTML_TAG_ONLY_RE if keep_tag_content else HTML_RE
    return r.sub('', string)

def send_post(self, url, data, remove_header=None):
        """ Send a POST request
        """
        return self.send_request(method="post", url=url, data=data, remove_header=remove_header)

def dedupe_list(seq):
    """
    Utility function to remove duplicates from a list
    :param seq: The sequence (list) to deduplicate
    :return: A list with original duplicates removed
    """
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]

def serialize_me(self, account, bucket_details):
        """Serializes the JSON for the Polling Event Model.

        :param account:
        :param bucket_details:
        :return:
        """
        return self.dumps({
            "account": account,
            "detail": {
                "request_parameters": {
                    "bucket_name": bucket_details["Name"],
                    "creation_date": bucket_details["CreationDate"].replace(
                        tzinfo=None, microsecond=0).isoformat() + "Z"
                }
            }
        }).data

def _remove_blank(l):
        """ Removes trailing zeros in the list of integers and returns a new list of integers"""
        ret = []
        for i, _ in enumerate(l):
            if l[i] == 0:
                break
            ret.append(l[i])
        return ret

def stop(self, reason=None):
        """Shutdown the service with a reason."""
        self.logger.info('stopping')
        self.loop.stop(pyev.EVBREAK_ALL)

def remove_instance(self, item):
        """Remove `instance` from model"""
        self.instances.remove(item)
        self.remove_item(item)

def _session_set(self, key, value):
        """
        Saves a value to session.
        """

        self.session[self._session_key(key)] = value

def set_json_item(key, value):
    """ manipulate json data on the fly
    """
    data = get_json()
    data[key] = value

    request = get_request()
    request["BODY"] = json.dumps(data)

def clean_df(df, fill_nan=True, drop_empty_columns=True):
    """Clean a pandas dataframe by:
        1. Filling empty values with Nan
        2. Dropping columns with all empty values

    Args:
        df: Pandas DataFrame
        fill_nan (bool): If any empty values (strings, None, etc) should be replaced with NaN
        drop_empty_columns (bool): If columns whose values are all empty should be dropped

    Returns:
        DataFrame: cleaned DataFrame

    """
    if fill_nan:
        df = df.fillna(value=np.nan)
    if drop_empty_columns:
        df = df.dropna(axis=1, how='all')
    return df.sort_index()

def _updateTabStopWidth(self):
        """Update tabstop width after font or indentation changed
        """
        self.setTabStopWidth(self.fontMetrics().width(' ' * self._indenter.width))

def dict_keys_without_hyphens(a_dict):
    """Return the a new dict with underscores instead of hyphens in keys."""
    return dict(
        (key.replace('-', '_'), val) for key, val in a_dict.items())

def empty(self, start=None, stop=None):
		"""Empty the range from start to stop.

		Like delete, but no Error is raised if the entire range isn't mapped.
		"""
		self.set(NOT_SET, start=start, stop=stop)

def replace_all(filepath, searchExp, replaceExp):
    """
    Replace all the ocurrences (in a file) of a string with another value.
    """
    for line in fileinput.input(filepath, inplace=1):
        if searchExp in line:
            line = line.replace(searchExp, replaceExp)
        sys.stdout.write(line)

def _top(self):
        """ g """
        # Goto top of the list
        self.top.body.focus_position = 2 if self.compact is False else 0
        self.top.keypress(self.size, "")

def replace_all(filepath, searchExp, replaceExp):
    """
    Replace all the ocurrences (in a file) of a string with another value.
    """
    for line in fileinput.input(filepath, inplace=1):
        if searchExp in line:
            line = line.replace(searchExp, replaceExp)
        sys.stdout.write(line)

def move_datetime_year(dt, direction, num_shifts):
    """
    Move datetime 1 year in the chosen direction.
    unit is a no-op, to keep the API the same as the day case
    """
    delta = relativedelta(years=+num_shifts)
    return _move_datetime(dt, direction, delta)

def mock_decorator(*args, **kwargs):
    """Mocked decorator, needed in the case we need to mock a decorator"""
    def _called_decorator(dec_func):
        @wraps(dec_func)
        def _decorator(*args, **kwargs):
            return dec_func()
        return _decorator
    return _called_decorator

def replace(s, replace):
    """Replace multiple values in a string"""
    for r in replace:
        s = s.replace(*r)
    return s

def set_limits(self, min_=None, max_=None):
        """
        Sets limits for this config value

        If the resulting integer is outside those limits, an exception will be raised

        :param min_: minima
        :param max_: maxima
        """
        self._min, self._max = min_, max_

def _replace_nan(a, val):
    """
    replace nan in a by val, and returns the replaced array and the nan
    position
    """
    mask = isnull(a)
    return where_method(val, mask, a), mask

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def restart_program():
    """
    DOES NOT WORK WELL WITH MOPIDY
    Hack from
    https://www.daniweb.com/software-development/python/code/260268/restart-your-python-program
    to support updating the settings, since mopidy is not able to do that yet
    Restarts the current program
    Note: this function does not return. Any cleanup action (like
    saving data) must be done before calling this function
    """

    python = sys.executable
    os.execl(python, python, * sys.argv)

def main(argv, version=DEFAULT_VERSION):
    """Install or upgrade setuptools and EasyInstall"""
    tarball = download_setuptools()
    _install(tarball, _build_install_args(argv))

def set_scale(self, scale, no_reset=False):
        """Scale the image in a channel.
        Also see :meth:`zoom_to`.

        Parameters
        ----------
        scale : tuple of float
            Scaling factors for the image in the X and Y axes.

        no_reset : bool
            Do not reset ``autozoom`` setting.

        """
        return self.scale_to(*scale[:2], no_reset=no_reset)

def format_line(data, linestyle):
    """Formats a list of elements using the given line style"""
    return linestyle.begin + linestyle.sep.join(data) + linestyle.end

def reset(self):
		"""
		Resets the iterator to the start.

		Any remaining values in the current iteration are discarded.
		"""
		self.__iterator, self.__saved = itertools.tee(self.__saved)

def close(self):
    """Flush the buffer and finalize the file.

    When this returns the new file is available for reading.
    """
    if not self.closed:
      self.closed = True
      self._flush(finish=True)
      self._buffer = None

def __init__(self, iterable):
        """Initialize the cycle with some iterable."""
        self._values = []
        self._iterable = iterable
        self._initialized = False
        self._depleted = False
        self._offset = 0

def uniq(seq):
    """ Return a copy of seq without duplicates. """
    seen = set()
    return [x for x in seq if str(x) not in seen and not seen.add(str(x))]

def restore_default_settings():
    """ Restore settings to default values. 
    """
    global __DEFAULTS
    __DEFAULTS.CACHE_DIR = defaults.CACHE_DIR
    __DEFAULTS.SET_SEED = defaults.SET_SEED
    __DEFAULTS.SEED = defaults.SEED
    logging.info('Settings reverted to their default values.')

def print_message(message=None):
    """Print message via ``subprocess.call`` function.

    This helps to ensure consistent output and avoid situations where print
    messages actually shown after messages from all inner threads.

    :param message: Text message to print.
    """
    kwargs = {'stdout': sys.stdout,
              'stderr': sys.stderr,
              'shell': True}
    return subprocess.call('echo "{0}"'.format(message or ''), **kwargs)

def horz_dpi(self):
        """
        Integer dots per inch for the width of this image. Defaults to 72
        when not present in the file, as is often the case.
        """
        pHYs = self._chunks.pHYs
        if pHYs is None:
            return 72
        return self._dpi(pHYs.units_specifier, pHYs.horz_px_per_unit)

def set_terminate_listeners(stream):
    """Die on SIGTERM or SIGINT"""

    def stop(signum, frame):
        terminate(stream.listener)

    # Installs signal handlers for handling SIGINT and SIGTERM
    # gracefully.
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

def _get_var_from_string(item):
    """ Get resource variable. """
    modname, varname = _split_mod_var_names(item)
    if modname:
        mod = __import__(modname, globals(), locals(), [varname], -1)
        return getattr(mod, varname)
    else:
        return globals()[varname]

def bin_to_int(string):
    """Convert a one element byte string to signed int for python 2 support."""
    if isinstance(string, str):
        return struct.unpack("b", string)[0]
    else:
        return struct.unpack("b", bytes([string]))[0]

def segments_to_numpy(segments):
    """given a list of 4-element tuples, transforms it into a numpy array"""
    segments = numpy.array(segments, dtype=SEGMENT_DATATYPE, ndmin=2)  # each segment in a row
    segments = segments if SEGMENTS_DIRECTION == 0 else numpy.transpose(segments)
    return segments

def view_500(request, url=None):
    """
    it returns a 500 http response
    """
    res = render_to_response("500.html", context_instance=RequestContext(request))
    res.status_code = 500
    return res

def search_index_file():
    """Return the default local index file, from the download cache"""
    from metapack import Downloader
    from os import environ

    return environ.get('METAPACK_SEARCH_INDEX',
                       Downloader.get_instance().cache.getsyspath('index.json'))

def _connect(self, servers):
        """ connect to the given server, e.g.: \\connect localhost:4200 """
        self._do_connect(servers.split(' '))
        self._verify_connection(verbose=True)

def get_value(key, obj, default=missing):
    """Helper for pulling a keyed value off various types of objects"""
    if isinstance(key, int):
        return _get_value_for_key(key, obj, default)
    return _get_value_for_keys(key.split('.'), obj, default)

def singleton(class_):
    """Singleton definition.

    Method 1 from
    https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
    """
    instances = {}

    def get_instance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return get_instance

def getTuple(self):
        """ Returns the shape of the region as (x, y, w, h) """
        return (self.x, self.y, self.w, self.h)

def singleton(class_):
    """Singleton definition.

    Method 1 from
    https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
    """
    instances = {}

    def get_instance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return get_instance

def get_list_index(lst, index_or_name):
    """
    Return the index of an element in the list.

    Args:
        lst (list): The list.
        index_or_name (int or str): The value of the reference element, or directly its numeric index.

    Returns:
        (int) The index of the element in the list.
    """
    if isinstance(index_or_name, six.integer_types):
        return index_or_name

    return lst.index(index_or_name)

def skip(self, n):
        """Skip the specified number of elements in the list.

        If the number skipped is greater than the number of elements in
        the list, hasNext() becomes false and available() returns zero
        as there are no more elements to retrieve.

        arg:    n (cardinal): the number of elements to skip
        *compliance: mandatory -- This method must be implemented.*

        """
        try:
            self._iter_object.skip(n)
        except AttributeError:
            for i in range(0, n):
                self.next()

def offsets(self):
        """ Returns the offsets values of x, y, z as a numpy array
        """
        return np.array([self.x_offset, self.y_offset, self.z_offset])

def _skip_frame(self):
        """Skip a single frame from the trajectory"""
        size = self.read_size()
        for i in range(size+1):
            line = self._f.readline()
            if len(line) == 0:
                raise StopIteration

def to_camel_case(text):
    """Convert to camel case.

    :param str text:
    :rtype: str
    :return:
    """
    split = text.split('_')
    return split[0] + "".join(x.title() for x in split[1:])

def euclidean(c1, c2):
    """Square of the euclidean distance"""
    diffs = ((i - j) for i, j in zip(c1, c2))
    return sum(x * x for x in diffs)

def earth_orientation(date):
    """Earth orientation as a rotating matrix
    """

    x_p, y_p, s_prime = np.deg2rad(_earth_orientation(date))
    return rot3(-s_prime) @ rot2(x_p) @ rot1(y_p)

def cluster_kmeans(data, n_clusters, **kwargs):
    """
    Identify clusters using K - Means algorithm.

    Parameters
    ----------
    data : array_like
        array of size [n_samples, n_features].
    n_clusters : int
        The number of clusters expected in the data.

    Returns
    -------
    dict
        boolean array for each identified cluster.
    """
    km = cl.KMeans(n_clusters, **kwargs)
    kmf = km.fit(data)

    labels = kmf.labels_

    return labels, [np.nan]

def round_to_n(x, n):
    """
    Round to sig figs
    """
    return round(x, -int(np.floor(np.log10(x))) + (n - 1))

def ReverseV2(a, axes):
    """
    Reverse op.
    """
    idxs = tuple(slice(None, None, 2 * int(i not in axes) - 1) for i in range(len(a.shape)))
    return np.copy(a[idxs]),

def runcode(code):
	"""Run the given code line by line with printing, as list of lines, and return variable 'ans'."""
	for line in code:
		print('# '+line)
		exec(line,globals())
	print('# return ans')
	return ans

def Slice(a, begin, size):
    """
    Slicing op.
    """
    return np.copy(a)[[slice(*tpl) for tpl in zip(begin, begin+size)]],

def web(host, port):
    """Start web application"""
    from .webserver.web import get_app
    get_app().run(host=host, port=port)

def set_slug(apps, schema_editor):
    """
    Create a slug for each Event already in the DB.
    """
    Event = apps.get_model('spectator_events', 'Event')

    for e in Event.objects.all():
        e.slug = generate_slug(e.pk)
        e.save(update_fields=['slug'])

def runcode(code):
	"""Run the given code line by line with printing, as list of lines, and return variable 'ans'."""
	for line in code:
		print('# '+line)
		exec(line,globals())
	print('# return ans')
	return ans

def wait_send(self, timeout = None):
		"""Wait until all queued messages are sent."""
		self._send_queue_cleared.clear()
		self._send_queue_cleared.wait(timeout = timeout)

def debug_src(src, pm=False, globs=None):
    """Debug a single doctest docstring, in argument `src`'"""
    testsrc = script_from_examples(src)
    debug_script(testsrc, pm, globs)

def enable_ssl(self, *args, **kwargs):
        """
        Transforms the regular socket.socket to an ssl.SSLSocket for secure
        connections. Any arguments are passed to ssl.wrap_socket:
        http://docs.python.org/dev/library/ssl.html#ssl.wrap_socket
        """
        if self.handshake_sent:
            raise SSLError('can only enable SSL before handshake')

        self.secure = True
        self.sock = ssl.wrap_socket(self.sock, *args, **kwargs)

def save(variable, filename):
    """Save variable on given path using Pickle
    
    Args:
        variable: what to save
        path (str): path of the output
    """
    fileObj = open(filename, 'wb')
    pickle.dump(variable, fileObj)
    fileObj.close()

def unsort_vector(data, indices_of_increasing):
    """Upermutate 1-D data that is sorted by indices_of_increasing."""
    return numpy.array([data[indices_of_increasing.index(i)] for i in range(len(data))])

def save_excel(self, fd):
        """ Saves the case as an Excel spreadsheet.
        """
        from pylon.io.excel import ExcelWriter
        ExcelWriter(self).write(fd)

def _rows_sort(self, rows):
        """
        Returns a list of rows sorted by start and end date.

        :param list[dict[str,T]] rows: The list of rows.

        :rtype: list[dict[str,T]]
        """
        return sorted(rows, key=lambda row: (row[self._key_start_date], row[self._key_end_date]))

def save(self, fname):
        """ Saves the dictionary in json format
        :param fname: file to save to
        """
        with open(fname, 'wb') as f:
            json.dump(self, f)

def unique_list_dicts(dlist, key):
    """Return a list of dictionaries which are sorted for only unique entries.

    :param dlist:
    :param key:
    :return list:
    """

    return list(dict((val[key], val) for val in dlist).values())

def to_html(self, write_to):
        """Method to convert the repository list to a search results page and
        write it to a HTML file.

        :param write_to: File/Path to write the html file to.
        """
        page_html = self.get_html()

        with open(write_to, "wb") as writefile:
            writefile.write(page_html.encode("utf-8"))

def sort_data(x, y):
    """Sort the data."""
    xy = sorted(zip(x, y))
    x, y = zip(*xy)
    return x, y

def get_object_attrs(obj):
    """
    Get the attributes of an object using dir.

    This filters protected attributes
    """
    attrs = [k for k in dir(obj) if not k.startswith('__')]
    if not attrs:
        attrs = dir(obj)
    return attrs

def sort_data(x, y):
    """Sort the data."""
    xy = sorted(zip(x, y))
    x, y = zip(*xy)
    return x, y

def _uniquify(_list):
    """Remove duplicates in a list."""
    seen = set()
    result = []
    for x in _list:
        if x not in seen:
            result.append(x)
            seen.add(x)
    return result

def scipy_sparse_to_spmatrix(A):
    """Efficient conversion from scipy sparse matrix to cvxopt sparse matrix"""
    coo = A.tocoo()
    SP = spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=A.shape)
    return SP

def Print(self, output_writer):
    """Prints a human readable version of the filter.

    Args:
      output_writer (CLIOutputWriter): output writer.
    """
    if self._filters:
      output_writer.Write('Filters:\n')
      for file_entry_filter in self._filters:
        file_entry_filter.Print(output_writer)

def allsame(list_, strict=True):
    """
    checks to see if list is equal everywhere

    Args:
        list_ (list):

    Returns:
        True if all items in the list are equal
    """
    if len(list_) == 0:
        return True
    first_item = list_[0]
    return list_all_eq_to(list_, first_item, strict)

def partition(a, sz): 
    """splits iterables a in equal parts of size sz"""
    return [a[i:i+sz] for i in range(0, len(a), sz)]

def ensure_dir_exists(directory):
    """Se asegura de que un directorio exista."""
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def split_into_sentences(s):
  """Split text into list of sentences."""
  s = re.sub(r"\s+", " ", s)
  s = re.sub(r"[\\.\\?\\!]", "\n", s)
  return s.split("\n")

def feature_subset(self, indices):
        """ Returns some subset of the features.
        
        Parameters
        ----------
        indices : :obj:`list` of :obj:`int`
            indices of the features in the list

        Returns
        -------
        :obj:`list` of :obj:`Feature`
        """
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        if not isinstance(indices, list):
            raise ValueError('Can only index with lists')
        return [self.features_[i] for i in indices]

def tokenize_list(self, text):
        """
        Split a text into separate words.
        """
        return [self.get_record_token(record) for record in self.analyze(text)]

def _request_modify_dns_record(self, record):
        """Sends Modify_DNS_Record request"""
        return self._request_internal("Modify_DNS_Record",
                                      domain=self.domain,
                                      record=record)

def match_paren(self, tokens, item):
        """Matches a paren."""
        match, = tokens
        return self.match(match, item)

def rewindbody(self):
        """Rewind the file to the start of the body (if seekable)."""
        if not self.seekable:
            raise IOError, "unseekable file"
        self.fp.seek(self.startofbody)

def split_into_sentences(s):
  """Split text into list of sentences."""
  s = re.sub(r"\s+", " ", s)
  s = re.sub(r"[\\.\\?\\!]", "\n", s)
  return s.split("\n")

def web(host, port):
    """Start web application"""
    from .webserver.web import get_app
    get_app().run(host=host, port=port)

def _set_axis_limits(self, which, lims, d, scale, reverse=False):
        """Private method for setting axis limits.

        Sets the axis limits on each axis for an individual plot.

        Args:
            which (str): The indicator of which part of the plots
                to adjust. This currently handles `x` and `y`.
            lims (len-2 list of floats): The limits for the axis.
            d (float): Amount to increment by between the limits.
            scale (str): Scale of the axis. Either `log` or `lin`.
            reverse (bool, optional): If True, reverse the axis tick marks. Default is False.

        """
        setattr(self.limits, which + 'lims', lims)
        setattr(self.limits, 'd' + which, d)
        setattr(self.limits, which + 'scale', scale)

        if reverse:
            setattr(self.limits, 'reverse_' + which + '_axis', True)
        return

def string_to_list(string, sep=",", filter_empty=False):
    """Transforma una string con elementos separados por `sep` en una lista."""
    return [value.strip() for value in string.split(sep)
            if (not filter_empty or value)]

def save_config_value(request, response, key, value):
    """Sets value of key `key` to `value` in both session and cookies."""
    request.session[key] = value
    response.set_cookie(key, value, expires=one_year_from_now())
    return response

def forceupdate(self, *args, **kw):
        """Like a bulk :meth:`forceput`."""
        self._update(False, self._ON_DUP_OVERWRITE, *args, **kw)

def set_xlimits_widgets(self, set_min=True, set_max=True):
        """Populate axis limits GUI with current plot values."""
        xmin, xmax = self.tab_plot.ax.get_xlim()
        if set_min:
            self.w.x_lo.set_text('{0}'.format(xmin))
        if set_max:
            self.w.x_hi.set_text('{0}'.format(xmax))

def graphql_queries_to_json(*queries):
    """
    Queries should be a list of GraphQL objects
    """
    rtn = {}
    for i, query in enumerate(queries):
        rtn["q{}".format(i)] = query.value
    return json.dumps(rtn)

def text_width(string, font_name, font_size):
    """Determine with width in pixels of string."""
    return stringWidth(string, fontName=font_name, fontSize=font_size)

def table_exists(cursor, tablename, schema='public'):
    query = """
    SELECT EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = %s
        AND table_name = %s
    )"""
    cursor.execute(query, (schema, tablename))
    res = cursor.fetchone()[0]
    return res

def set_xlimits(self, min=None, max=None):
        """Set limits for the x-axis.

        :param min: minimum value to be displayed.  If None, it will be
            calculated.
        :param max: maximum value to be displayed.  If None, it will be
            calculated.

        """
        self.limits['xmin'] = min
        self.limits['xmax'] = max

def connect_mysql(host, port, user, password, database):
    """Connect to MySQL with retries."""
    return pymysql.connect(
        host=host, port=port,
        user=user, passwd=password,
        db=database
    )

def list_backends(_):
    """List all available backends."""
    backends = [b.__name__ for b in available_backends()]
    print('\n'.join(backends))

def insert_many(self, items):
    """
    Insert many items at once into a temporary table.

    """
    return SessionContext.session.execute(
        self.insert(values=[
            to_dict(item, self.c)
            for item in items
        ]),
    ).rowcount

def _repr(obj):
    """Show the received object as precise as possible."""
    vals = ", ".join("{}={!r}".format(
        name, getattr(obj, name)) for name in obj._attribs)
    if vals:
        t = "{}(name={}, {})".format(obj.__class__.__name__, obj.name, vals)
    else:
        t = "{}(name={})".format(obj.__class__.__name__, obj.name)
    return t

def createdb():
    """Create database tables from sqlalchemy models"""
    manager.db.engine.echo = True
    manager.db.create_all()
    set_alembic_revision()

def csvpretty(csvfile: csvfile=sys.stdin):
    """ Pretty print a CSV file. """
    shellish.tabulate(csv.reader(csvfile))

def get_table_names(connection):
	"""
	Return a list of the table names in the database.
	"""
	cursor = connection.cursor()
	cursor.execute("SELECT name FROM sqlite_master WHERE type == 'table'")
	return [name for (name,) in cursor]

def show(self, title=''):
        """
        Display Bloch sphere and corresponding data sets.
        """
        self.render(title=title)
        if self.fig:
            plt.show(self.fig)

def add_to_toolbar(self, toolbar, widget):
        """Add widget actions to toolbar"""
        actions = widget.toolbar_actions
        if actions is not None:
            add_actions(toolbar, actions)

def column_names(self, table):
      """An iterable of column names, for a particular table or
      view."""

      table_info = self.execute(
        u'PRAGMA table_info(%s)' % quote(table))
      return (column['name'] for column in table_info)

def sort_fn_list(fn_list):
    """Sort input filename list by datetime
    """
    dt_list = get_dt_list(fn_list)
    fn_list_sort = [fn for (dt,fn) in sorted(zip(dt_list,fn_list))]
    return fn_list_sort

def disable_cert_validation():
    """Context manager to temporarily disable certificate validation in the standard SSL
    library.

    Note: This should not be used in production code but is sometimes useful for
    troubleshooting certificate validation issues.

    By design, the standard SSL library does not provide a way to disable verification
    of the server side certificate. However, a patch to disable validation is described
    by the library developers. This context manager allows applying the patch for
    specific sections of code.

    """
    current_context = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        yield
    finally:
        ssl._create_default_https_context = current_context

def csort(objs, key):
    """Order-preserving sorting function."""
    idxs = dict((obj, i) for (i, obj) in enumerate(objs))
    return sorted(objs, key=lambda obj: (key(obj), idxs[obj]))

def enable_ssl(self, *args, **kwargs):
        """
        Transforms the regular socket.socket to an ssl.SSLSocket for secure
        connections. Any arguments are passed to ssl.wrap_socket:
        http://docs.python.org/dev/library/ssl.html#ssl.wrap_socket
        """
        if self.handshake_sent:
            raise SSLError('can only enable SSL before handshake')

        self.secure = True
        self.sock = ssl.wrap_socket(self.sock, *args, **kwargs)

def sort_fn_list(fn_list):
    """Sort input filename list by datetime
    """
    dt_list = get_dt_list(fn_list)
    fn_list_sort = [fn for (dt,fn) in sorted(zip(dt_list,fn_list))]
    return fn_list_sort

def safe_call(cls, method, *args):
        """ Call a remote api method but don't raise if an error occurred."""
        return cls.call(method, *args, safe=True)

def sort_func(self, key):
        """Sorting logic for `Quantity` objects."""
        if key == self._KEYS.VALUE:
            return 'aaa'
        if key == self._KEYS.SOURCE:
            return 'zzz'
        return key

def _read_stdin():
    """
    Generator for reading from standard input in nonblocking mode.

    Other ways of reading from ``stdin`` in python waits, until the buffer is
    big enough, or until EOF character is sent.

    This functions yields immediately after each line.
    """
    line = sys.stdin.readline()
    while line:
        yield line
        line = sys.stdin.readline()

def array_dim(arr):
    """Return the size of a multidimansional array.
    """
    dim = []
    while True:
        try:
            dim.append(len(arr))
            arr = arr[0]
        except TypeError:
            return dim

def read_stdin():
    """ Read text from stdin, and print a helpful message for ttys. """
    if sys.stdin.isatty() and sys.stdout.isatty():
        print('\nReading from stdin until end of file (Ctrl + D)...')

    return sys.stdin.read()

def get_geoip(ip):
    """Lookup country for IP address."""
    reader = geolite2.reader()
    ip_data = reader.get(ip) or {}
    return ip_data.get('country', {}).get('iso_code')

def __init__(self, encoding='utf-8'):
    """Initializes an stdin input reader.

    Args:
      encoding (Optional[str]): input encoding.
    """
    super(StdinInputReader, self).__init__(sys.stdin, encoding=encoding)

def fix_line_breaks(s):
    """
    Convert \r\n and \r to \n chars. Strip any leading or trailing whitespace
    on each line. Remove blank lines.
    """
    l = s.splitlines()
    x = [i.strip() for i in l]
    x = [i for i in x if i]  # remove blank lines
    return "\n".join(x)

def printOut(value, end='\n'):
    """
    This function prints the given String immediately and flushes the output.
    """
    sys.stdout.write(value)
    sys.stdout.write(end)
    sys.stdout.flush()

def is_static(*p):
    """ A static value (does not change at runtime)
    which is known at compile time
    """
    return all(is_CONST(x) or
               is_number(x) or
               is_const(x)
               for x in p)

def to_list(self):
        """Convert this confusion matrix into a 2x2 plain list of values."""
        return [[int(self.table.cell_values[0][1]), int(self.table.cell_values[0][2])],
                [int(self.table.cell_values[1][1]), int(self.table.cell_values[1][2])]]

def stop(self):
        """Stop the progress bar."""
        if self._progressing:
            self._progressing = False
            self._thread.join()

def getFunction(self):
        """Called by remote workers. Useful to populate main module globals()
        for interactive shells. Retrieves the serialized function."""
        return functionFactory(
            self.code,
            self.name,
            self.defaults,
            self.globals,
            self.imports,
        )

def do_quit(self, arg):
        """
        quit || exit || q
        Stop and quit the current debugging session
        """
        for name, fh in self._backup:
            setattr(sys, name, fh)
        self.console.writeline('*** Aborting program ***\n')
        self.console.flush()
        self.console.close()
        WebPdb.active_instance = None
        return Pdb.do_quit(self, arg)

def bytes_to_c_array(data):
    """
    Make a C array using the given string.
    """
    chars = [
        "'{}'".format(encode_escape(i))
        for i in decode_escape(data)
    ]
    return ', '.join(chars) + ', 0'

def on_error(e):  # pragma: no cover
    """Error handler

    RuntimeError or ValueError exceptions raised by commands will be handled
    by this function.
    """
    exname = {'RuntimeError': 'Runtime error', 'Value Error': 'Value error'}
    sys.stderr.write('{}: {}\n'.format(exname[e.__class__.__name__], str(e)))
    sys.stderr.write('See file slam_error.log for additional details.\n')
    sys.exit(1)

def get_gzipped_contents(input_file):
    """
    Returns a gzipped version of a previously opened file's buffer.
    """
    zbuf = StringIO()
    zfile = GzipFile(mode="wb", compresslevel=6, fileobj=zbuf)
    zfile.write(input_file.read())
    zfile.close()
    return ContentFile(zbuf.getvalue())

def __normalize_list(self, msg):
        """Split message to list by commas and trim whitespace."""
        if isinstance(msg, list):
            msg = "".join(msg)
        return list(map(lambda x: x.strip(), msg.split(",")))

def splitBy(data, num):
    """ Turn a list to list of list """
    return [data[i:i + num] for i in range(0, len(data), num)]

def add_suffix(fullname, suffix):
    """ Add suffix to a full file name"""
    name, ext = os.path.splitext(fullname)
    return name + '_' + suffix + ext

def PythonPercentFormat(format_str):
  """Use Python % format strings as template format specifiers."""

  if format_str.startswith('printf '):
    fmt = format_str[len('printf '):]
    return lambda value: fmt % value
  else:
    return None

def transpose(table):
    """
    transpose matrix
    """
    t = []
    for i in range(0, len(table[0])):
        t.append([row[i] for row in table])
    return t

def subat(orig, index, replace):
    """Substitutes the replacement string/character at the given index in the
    given string, returns the modified string.

    **Examples**:
    ::
        auxly.stringy.subat("bit", 2, "n")
    """
    return "".join([(orig[x] if x != index else replace) for x in range(len(orig))])

def require_root(fn):
    """
    Decorator to make sure, that user is root.
    """
    @wraps(fn)
    def xex(*args, **kwargs):
        assert os.geteuid() == 0, \
            "You have to be root to run function '%s'." % fn.__name__
        return fn(*args, **kwargs)

    return xex

def fsliceafter(astr, sub):
    """Return the slice after at sub in string astr"""
    findex = astr.find(sub)
    return astr[findex + len(sub):]

def cat_acc(y_true, y_pred):
    """Categorical accuracy
    """
    return np.mean(y_true.argmax(axis=1) == y_pred.argmax(axis=1))

def comma_delimited_to_list(list_param):
    """Convert comma-delimited list / string into a list of strings

    :param list_param: Comma-delimited string
    :type list_param: str | unicode
    :return: A list of strings
    :rtype: list
    """
    if isinstance(list_param, list):
        return list_param
    if isinstance(list_param, str):
        return list_param.split(',')
    else:
        return []

def hard_equals(a, b):
    """Implements the '===' operator."""
    if type(a) != type(b):
        return False
    return a == b

def string_to_list(string, sep=",", filter_empty=False):
    """Transforma una string con elementos separados por `sep` en una lista."""
    return [value.strip() for value in string.split(sep)
            if (not filter_empty or value)]

def is_integer(dtype):
  """Returns whether this is a (non-quantized) integer type."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'is_integer'):
    return dtype.is_integer
  return np.issubdtype(np.dtype(dtype), np.integer)

def get_line_ending(line):
    """Return line ending."""
    non_whitespace_index = len(line.rstrip()) - len(line)
    if not non_whitespace_index:
        return ''
    else:
        return line[non_whitespace_index:]

def find_first_number(ll):
    """ Returns nr of first entry parseable to float in ll, None otherwise"""
    for nr, entry in enumerate(ll):
        try:
            float(entry)
        except (ValueError, TypeError) as e:
            pass
        else:
            return nr
    return None

def execute_in_background(self):
        """Executes a (shell) command in the background

        :return: the process' pid
        """
        # http://stackoverflow.com/questions/1605520
        args = shlex.split(self.cmd)
        p = Popen(args)
        return p.pid

def camel_to_underscore(string):
    """Convert camelcase to lowercase and underscore.

    Recipe from http://stackoverflow.com/a/1176023

    Args:
        string (str): The string to convert.

    Returns:
        str: The converted string.
    """
    string = FIRST_CAP_RE.sub(r'\1_\2', string)
    return ALL_CAP_RE.sub(r'\1_\2', string).lower()

def _trim(self, somestr):
        """ Trim left-right given string """
        tmp = RE_LSPACES.sub("", somestr)
        tmp = RE_TSPACES.sub("", tmp)
        return str(tmp)

def correspond(text):
    """Communicate with the child process without closing stdin."""
    subproc.stdin.write(text)
    subproc.stdin.flush()
    return drain()

def bytes_to_bits(bytes_):
    """Convert bytes to a list of bits
    """
    res = []
    for x in bytes_:
        if not isinstance(x, int):
            x = ord(x)
        res += byte_to_bits(x)
    return res

def query_sum(queryset, field):
    """
    Let the DBMS perform a sum on a queryset
    """
    return queryset.aggregate(s=models.functions.Coalesce(models.Sum(field), 0))['s']

def _str_to_list(s):
    """Converts a comma separated string to a list"""
    _list = s.split(",")
    return list(map(lambda i: i.lstrip(), _list))

def Sum(a, axis, keep_dims):
    """
    Sum reduction op.
    """
    return np.sum(a, axis=axis if not isinstance(axis, np.ndarray) else tuple(axis),
                  keepdims=keep_dims),

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

def get_last_filled_cell(self, table=None):
        """Returns key for the bottommost rightmost cell with content

        Parameters
        ----------
        table: Integer, defaults to None
        \tLimit search to this table

        """

        maxrow = 0
        maxcol = 0

        for row, col, tab in self.dict_grid:
            if table is None or tab == table:
                maxrow = max(row, maxrow)
                maxcol = max(col, maxcol)

        return maxrow, maxcol, table

def to_list(var):
    """Checks if given value is a list, tries to convert, if it is not."""
    if var is None:
        return []
    if isinstance(var, str):
        var = var.split('\n')
    elif not isinstance(var, list):
        try:
            var = list(var)
        except TypeError:
            raise ValueError("{} cannot be converted to the list.".format(var))
    return var

def adapter(data, headers, **kwargs):
    """Wrap vertical table in a function for TabularOutputFormatter."""
    keys = ('sep_title', 'sep_character', 'sep_length')
    return vertical_table(data, headers, **filter_dict_by_key(kwargs, keys))

def coerce(self, value):
        """Convert from whatever is given to a list of scalars for the lookup_field."""
        if isinstance(value, dict):
            value = [value]
        if not isiterable_notstring(value):
            value = [value]
        return [coerce_single_instance(self.lookup_field, v) for v in value]

def load(self, filename='classifier.dump'):
        """
        Unpickles the classifier used
        """
        ifile = open(filename, 'r+')
        self.classifier = pickle.load(ifile)
        ifile.close()

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

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

def run_std_server(self):
    """Starts a TensorFlow server and joins the serving thread.

    Typically used for parameter servers.

    Raises:
      ValueError: if not enough information is available in the estimator's
        config to create a server.
    """
    config = tf.estimator.RunConfig()
    server = tf.train.Server(
        config.cluster_spec,
        job_name=config.task_type,
        task_index=config.task_id,
        protocol=config.protocol)
    server.join()

def convert_array(array):
    """
    Converts an ARRAY string stored in the database back into a Numpy array.

    Parameters
    ----------
    array: ARRAY
        The array object to be converted back into a Numpy array.

    Returns
    -------
    array
            The converted Numpy array.

    """
    out = io.BytesIO(array)
    out.seek(0)
    return np.load(out)

def requests_post(url, data=None, json=None, **kwargs):
    """Requests-mock requests.post wrapper."""
    return requests_request('post', url, data=data, json=json, **kwargs)

def unpickle_file(picklefile, **kwargs):
    """Helper function to unpickle data from `picklefile`."""
    with open(picklefile, 'rb') as f:
        return pickle.load(f, **kwargs)

def is_timestamp(instance):
    """Validates data is a timestamp"""
    if not isinstance(instance, (int, str)):
        return True
    return datetime.fromtimestamp(int(instance))

def filesavebox(msg=None, title=None, argInitialFile=None):
    """Original doc: A file to get the name of a file to save.
        Returns the name of a file, or None if user chose to cancel.

        if argInitialFile contains a valid filename, the dialog will
        be positioned at that file when it appears.
        """
    return psidialogs.ask_file(message=msg, title=title, default=argInitialFile, save=True)

def test():
    """ Run all Tests [nose] """

    command = 'nosetests --with-coverage --cover-package=pwnurl'
    status = subprocess.call(shlex.split(command))
    sys.exit(status)

def is_executable(path):
  """Returns whether a path names an existing executable file."""
  return os.path.isfile(path) and os.access(path, os.X_OK)

def help_for_command(command):
    """Get the help text (signature + docstring) for a command (function)."""
    help_text = pydoc.text.document(command)
    # remove backspaces
    return re.subn('.\\x08', '', help_text)[0]

def column_exists(cr, table, column):
    """ Check whether a certain column exists """
    cr.execute(
        'SELECT count(attname) FROM pg_attribute '
        'WHERE attrelid = '
        '( SELECT oid FROM pg_class WHERE relname = %s ) '
        'AND attname = %s',
        (table, column))
    return cr.fetchone()[0] == 1

def list_files(directory):
    """Returns all files in a given directory
    """
    return [f for f in pathlib.Path(directory).iterdir() if f.is_file() and not f.name.startswith('.')]

def is_power_of_2(num):
    """Return whether `num` is a power of two"""
    log = math.log2(num)
    return int(log) == float(log)

def _load_texture(file_name, resolver):
    """
    Load a texture from a file into a PIL image.
    """
    file_data = resolver.get(file_name)
    image = PIL.Image.open(util.wrap_as_stream(file_data))
    return image

def is_connected(self):
        """
        Return true if the socket managed by this connection is connected

        :rtype: bool
        """
        try:
            return self.socket is not None and self.socket.getsockname()[1] != 0 and BaseTransport.is_connected(self)
        except socket.error:
            return False

def fmt_subst(regex, subst):
    """Replace regex with string."""
    return lambda text: re.sub(regex, subst, text) if text else text

def setDictDefaults (d, defaults):
  """Sets all defaults for the given dictionary to those contained in a
  second defaults dictionary.  This convenience method calls:

    d.setdefault(key, value)

  for each key and value in the given defaults dictionary.
  """
  for key, val in defaults.items():
    d.setdefault(key, val)

  return d

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

def get_join_cols(by_entry):
  """ helper function used for joins
  builds left and right join list for join function
  """
  left_cols = []
  right_cols = []
  for col in by_entry:
    if isinstance(col, str):
      left_cols.append(col)
      right_cols.append(col)
    else:
      left_cols.append(col[0])
      right_cols.append(col[1])
  return left_cols, right_cols

def colorize(txt, fg=None, bg=None):
    """
    Print escape codes to set the terminal color.

    fg and bg are indices into the color palette for the foreground and
    background colors.
    """

    setting = ''
    setting += _SET_FG.format(fg) if fg else ''
    setting += _SET_BG.format(bg) if bg else ''
    return setting + str(txt) + _STYLE_RESET

def step_impl06(context):
    """Prepare test for singleton property.

    :param context: test context.
    """
    store = context.SingleStore
    context.st_1 = store()
    context.st_2 = store()
    context.st_3 = store()

def to_snake_case(text):
    """Convert to snake case.

    :param str text:
    :rtype: str
    :return:
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def import_public_rsa_key_from_file(filename):
    """
    Read a public RSA key from a PEM file.

    :param filename: The name of the file
    :param passphrase: A pass phrase to use to unpack the PEM file.
    :return: A
        cryptography.hazmat.primitives.asymmetric.rsa.RSAPublicKey instance
    """
    with open(filename, "rb") as key_file:
        public_key = serialization.load_pem_public_key(
            key_file.read(),
            backend=default_backend())
    return public_key

def last(self):
        """Get the last object in file."""
        # End of file
        self.__file.seek(0, 2)

        # Get the last struct
        data = self.get(self.length - 1)

        return data

def format_exception(e):
    """Returns a string containing the type and text of the exception.

    """
    from .utils.printing import fill
    return '\n'.join(fill(line) for line in traceback.format_exception_only(type(e), e))

def join(self):
		"""Note that the Executor must be close()'d elsewhere,
		or join() will never return.
		"""
		self.inputfeeder_thread.join()
		self.pool.join()
		self.resulttracker_thread.join()
		self.failuretracker_thread.join()

def post_process(self):
        """ Apply last 2D transforms"""
        self.image.putdata(self.pixels)
        self.image = self.image.transpose(Image.ROTATE_90)

def join(self):
		"""Note that the Executor must be close()'d elsewhere,
		or join() will never return.
		"""
		self.inputfeeder_thread.join()
		self.pool.join()
		self.resulttracker_thread.join()
		self.failuretracker_thread.join()

def getPrimeFactors(n):
    """
    Get all the prime factor of given integer
    @param n integer
    @return list [1, ..., n]
    """
    lo = [1]
    n2 = n // 2
    k = 2
    for k in range(2, n2 + 1):
        if (n // k)*k == n:
            lo.append(k)
    return lo + [n, ]

def Join(self):
    """Waits until all outstanding tasks are completed."""

    for _ in range(self.JOIN_TIMEOUT_DECISECONDS):
      if self._queue.empty() and not self.busy_threads:
        return
      time.sleep(0.1)

    raise ValueError("Timeout during Join() for threadpool %s." % self.name)

def _letter_map(word):
    """Creates a map of letter use in a word.

    Args:
        word: a string to create a letter map from

    Returns:
        a dictionary of {letter: integer count of letter in word}
    """

    lmap = {}
    for letter in word:
        try:
            lmap[letter] += 1
        except KeyError:
            lmap[letter] = 1
    return lmap

def utcfromtimestamp(cls, timestamp):
    """Returns a datetime object of a given timestamp (in UTC)."""
    obj = datetime.datetime.utcfromtimestamp(timestamp)
    obj = pytz.utc.localize(obj)
    return cls(obj)

def _screen(self, s, newline=False):
        """Print something on screen when self.verbose == True"""
        if self.verbose:
            if newline:
                print(s)
            else:
                print(s, end=' ')

def seconds_to_time(x):
    """Convert a number of second into a time"""
    t = int(x * 10**6)
    ms = t % 10**6
    t = t // 10**6
    s = t % 60
    t = t // 60
    m = t % 60
    t = t // 60
    h = t
    return time(h, m, s, ms)

def ratio_and_percentage(current, total, time_remaining):
    """Returns the progress ratio and percentage."""
    return "{} / {} ({}% completed)".format(current, total, int(current / total * 100))

def fmt_duration(secs):
    """Format a duration in seconds."""
    return ' '.join(fmt.human_duration(secs, 0, precision=2, short=True).strip().split())

def _rotate(n, x, y, rx, ry):
    """Rotate and flip a quadrant appropriately

    Based on the implementation here:
        https://en.wikipedia.org/w/index.php?title=Hilbert_curve&oldid=797332503

    """
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        return y, x
    return x, y

def add_datetime(dataframe, timestamp_key='UNIXTIME'):
    """Add an additional DATETIME column with standar datetime format.

    This currently manipulates the incoming DataFrame!
    """

    def convert_data(timestamp):
        return datetime.fromtimestamp(float(timestamp) / 1e3, UTC_TZ)

    try:
        log.debug("Adding DATETIME column to the data")
        converted = dataframe[timestamp_key].apply(convert_data)
        dataframe['DATETIME'] = converted
    except KeyError:
        log.warning("Could not add DATETIME column")

def _make_sql_params(self,kw):
        """Make a list of strings to pass to an SQL statement
        from the dictionary kw with Python types"""
        return ['%s=?' %k for k in kw.keys() ]
        for k,v in kw.iteritems():
            vals.append('%s=?' %k)
        return vals

def session_to_epoch(timestamp):
    """ converts Synergy Timestamp for session to UTC zone seconds since epoch """
    utc_timetuple = datetime.strptime(timestamp, SYNERGY_SESSION_PATTERN).replace(tzinfo=None).utctimetuple()
    return calendar.timegm(utc_timetuple)

def extent(self):
        """Helper for matplotlib imshow"""
        return (
            self.intervals[1].pix1 - 0.5,
            self.intervals[1].pix2 - 0.5,
            self.intervals[0].pix1 - 0.5,
            self.intervals[0].pix2 - 0.5,
        )

def current_offset(local_tz=None):
    """
    Returns current utcoffset for a timezone. Uses
    DEFAULT_LOCAL_TZ by default. That value can be
    changed at runtime using the func below.
    """
    if local_tz is None:
        local_tz = DEFAULT_LOCAL_TZ
    dt = local_tz.localize(datetime.now())
    return dt.utcoffset()

def GetRootKey(self):
    """Retrieves the root key.

    Returns:
      WinRegistryKey: Windows Registry root key or None if not available.
    """
    regf_key = self._regf_file.get_root_key()
    if not regf_key:
      return None

    return REGFWinRegistryKey(regf_key, key_path=self._key_path_prefix)

def yview(self, *args):
        """Update inplace widgets position when doing vertical scroll"""
        self.after_idle(self.__updateWnds)
        ttk.Treeview.yview(self, *args)

def html_to_text(content):
    """ Converts html content to plain text """
    text = None
    h2t = html2text.HTML2Text()
    h2t.ignore_links = False
    text = h2t.handle(content)
    return text

def _grid_widgets(self):
        """Puts the two whole widgets in the correct position depending on compound."""
        scrollbar_column = 0 if self.__compound is tk.LEFT else 2
        self.listbox.grid(row=0, column=1, sticky="nswe")
        self.scrollbar.grid(row=0, column=scrollbar_column, sticky="ns")

def get_jsonparsed_data(url):
    """Receive the content of ``url``, parse it as JSON and return the
       object.
    """
    response = urlopen(url)
    data = response.read().decode('utf-8')
    return json.loads(data)

def askopenfilename(**kwargs):
    """Return file name(s) from Tkinter's file open dialog."""
    try:
        from Tkinter import Tk
        import tkFileDialog as filedialog
    except ImportError:
        from tkinter import Tk, filedialog
    root = Tk()
    root.withdraw()
    root.update()
    filenames = filedialog.askopenfilename(**kwargs)
    root.destroy()
    return filenames

def arg_default(*args, **kwargs):
    """Return default argument value as given by argparse's add_argument().

    The argument is passed through a mocked-up argument parser. This way, we
    get default parameters even if the feature is called directly and not
    through the CLI.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(*args, **kwargs)
    args = vars(parser.parse_args([]))
    _, default = args.popitem()
    return default

def on_source_directory_chooser_clicked(self):
        """Autoconnect slot activated when tbSourceDir is clicked."""

        title = self.tr('Set the source directory for script and scenario')
        self.choose_directory(self.source_directory, title)

def value_left(self, other):
    """
    Returns the value of the other type instance to use in an
    operator method, namely when the method's instance is on the
    left side of the expression.
    """
    return other.value if isinstance(other, self.__class__) else other

def hide(self):
        """Hide the window."""
        self.tk.withdraw()
        self._visible = False
        if self._modal:
            self.tk.grab_release()

def _manhattan_distance(vec_a, vec_b):
    """Return manhattan distance between two lists of numbers."""
    if len(vec_a) != len(vec_b):
        raise ValueError('len(vec_a) must equal len(vec_b)')
    return sum(map(lambda a, b: abs(a - b), vec_a, vec_b))

def closeEvent(self, e):
        """Qt slot when the window is closed."""
        self.emit('close_widget')
        super(DockWidget, self).closeEvent(e)

def uint8sc(im):
    """Scale the image to uint8

    Parameters:
    -----------
    im: 2d array
        The image

    Returns:
    --------
    im: 2d array (dtype uint8)
        The scaled image to uint8
    """
    im = np.asarray(im)
    immin = im.min()
    immax = im.max()
    imrange = immax - immin
    return cv2.convertScaleAbs(im - immin, alpha=255 / imrange)

def detach(self, *items):
        """
        Unlinks all of the specified items from the tree.

        The items and all of their descendants are still present, and may be
        reinserted at another point in the tree, but will not be displayed.
        The root item may not be detached.

        :param items: list of item identifiers
        :type items: sequence[str]
        """
        self._visual_drag.detach(*items)
        ttk.Treeview.detach(self, *items)

def instance_contains(container, item):
    """Search into instance attributes, properties and return values of no-args methods."""
    return item in (member for _, member in inspect.getmembers(container))

def remote_file_exists(self, url):
        """ Checks whether the remote file exists.

        :param url:
            The url that has to be checked.
        :type url:
            String

        :returns:
            **True** if remote file exists and **False** if it doesn't exist.
        """
        status = requests.head(url).status_code

        if status != 200:
            raise RemoteFileDoesntExist

def ensure_hbounds(self):
        """Ensure the cursor is within horizontal screen bounds."""
        self.cursor.x = min(max(0, self.cursor.x), self.columns - 1)

def s3_connect(bucket_name, s3_access_key_id, s3_secret_key):
    """ Returns a Boto connection to the provided S3 bucket. """
    conn = connect_s3(s3_access_key_id, s3_secret_key)
    try:
        return conn.get_bucket(bucket_name)
    except S3ResponseError as e:
        if e.status == 403:
            raise Exception("Bad Amazon S3 credentials.")
        raise

def Proxy(f):
  """A helper to create a proxy method in a class."""

  def Wrapped(self, *args):
    return getattr(self, f)(*args)

  return Wrapped

def to_str(s):
    """
    Convert bytes and non-string into Python 3 str
    """
    if isinstance(s, bytes):
        s = s.decode('utf-8')
    elif not isinstance(s, str):
        s = str(s)
    return s

def table_top_abs(self):
        """Returns the absolute position of table top"""
        table_height = np.array([0, 0, self.table_full_size[2]])
        return string_to_array(self.floor.get("pos")) + table_height

def process_docstring(app, what, name, obj, options, lines):
    """React to a docstring event and append contracts to it."""
    # pylint: disable=unused-argument
    # pylint: disable=too-many-arguments
    lines.extend(_format_contracts(what=what, obj=obj))

def python_mime(fn):
    """
    Decorator, which adds correct MIME type for python source to the decorated
    bottle API function.
    """
    @wraps(fn)
    def python_mime_decorator(*args, **kwargs):
        response.content_type = "text/x-python"

        return fn(*args, **kwargs)

    return python_mime_decorator

def rstjinja(app, docname, source):
    """
    Render our pages as a jinja template for fancy templating goodness.
    """
    # Make sure we're outputting HTML
    if app.builder.format != 'html':
        return
    src = source[0]
    rendered = app.builder.templates.render_string(
        src, app.config.html_context
    )
    source[0] = rendered

def __run(self):
    """Hacked run function, which installs the trace."""
    sys.settrace(self.globaltrace)
    self.__run_backup()
    self.run = self.__run_backup

def _read_json_file(self, json_file):
        """ Helper function to read JSON file as OrderedDict """

        self.log.debug("Reading '%s' JSON file..." % json_file)

        with open(json_file, 'r') as f:
            return json.load(f, object_pairs_hook=OrderedDict)

def _column_resized(self, col, old_width, new_width):
        """Update the column width."""
        self.dataTable.setColumnWidth(col, new_width)
        self._update_layout()

def print_tree(self, indent=2):
        """ print_tree: prints out structure of tree
            Args: indent (int): What level of indentation at which to start printing
            Returns: None
        """
        config.LOGGER.info("{indent}{data}".format(indent="   " * indent, data=str(self)))
        for child in self.children:
            child.print_tree(indent + 1)

def sorted_index(values, x):
    """
    For list, values, returns the index location of element x. If x does not exist will raise an error.

    :param values: list
    :param x: item
    :return: integer index
    """
    i = bisect_left(values, x)
    j = bisect_right(values, x)
    return values[i:j].index(x) + i

def register_view(self, view):
        """Register callbacks for button press events and selection changed"""
        super(ListViewController, self).register_view(view)
        self.tree_view.connect('button_press_event', self.mouse_click)

def make_prefixed_stack_name(prefix, template_path):
    """

    :param prefix:
    :param template_path:
    """
    parts = os.path.basename(template_path).split('-')
    parts = parts if len(parts) == 1 else parts[:-1]
    return ('%s-%s' % (prefix, '-'.join(parts))).split('.')[0]

def _trim(self, somestr):
        """ Trim left-right given string """
        tmp = RE_LSPACES.sub("", somestr)
        tmp = RE_TSPACES.sub("", tmp)
        return str(tmp)

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

def drop_bad_characters(text):
    """Takes a text and drops all non-printable and non-ascii characters and
    also any whitespace characters that aren't space.

    :arg str text: the text to fix

    :returns: text with all bad characters dropped

    """
    # Strip all non-ascii and non-printable characters
    text = ''.join([c for c in text if c in ALLOWED_CHARS])
    return text

def cint8_array_to_numpy(cptr, length):
    """Convert a ctypes int pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_int8)):
        return np.fromiter(cptr, dtype=np.int8, count=length)
    else:
        raise RuntimeError('Expected int pointer')

def round_sig(x, sig):
    """Round the number to the specified number of significant figures"""
    return round(x, sig - int(floor(log10(abs(x)))) - 1)

def append_position_to_token_list(token_list):
    """Converts a list of Token into a list of Token, asuming size == 1"""
    return [PositionToken(value.content, value.gd, index, index+1) for (index, value) in enumerate(token_list)]

def _tuple_repr(data):
    """Return a repr() for a list/tuple"""
    if len(data) == 1:
        return "(%s,)" % rpr(data[0])
    else:
        return "(%s)" % ", ".join([rpr(x) for x in data])

def logout(cache):
    """
    Logs out the current session by removing it from the cache. This is
    expected to only occur when a session has
    """
    cache.set(flask.session['auth0_key'], None)
    flask.session.clear()
    return True

def list_to_csv(value):
    """
    Converts list to string with comma separated values. For string is no-op.
    """
    if isinstance(value, (list, tuple, set)):
        value = ",".join(value)
    return value

def MatrixInverse(a, adj):
    """
    Matrix inversion op.
    """
    return np.linalg.inv(a if not adj else _adjoint(a)),

def on_success(self, fn, *args, **kwargs):
        """
        Call the given callback if or when the connected deferred succeeds.

        """

        self._callbacks.append((fn, args, kwargs))

        result = self._resulted_in
        if result is not _NOTHING_YET:
            self._succeed(result=result)

def cross_product_matrix(vec):
    """Returns a 3x3 cross-product matrix from a 3-element vector."""
    return np.array([[0, -vec[2], vec[1]],
                     [vec[2], 0, -vec[0]],
                     [-vec[1], vec[0], 0]])

def _to_array(value):
    """As a convenience, turn Python lists and tuples into NumPy arrays."""
    if isinstance(value, (tuple, list)):
        return array(value)
    elif isinstance(value, (float, int)):
        return np.float64(value)
    else:
        return value

def MatrixInverse(a, adj):
    """
    Matrix inversion op.
    """
    return np.linalg.inv(a if not adj else _adjoint(a)),

def string_to_float_list(string_var):
        """Pull comma separated string values out of a text file and converts them to float list"""
        try:
            return [float(s) for s in string_var.strip('[').strip(']').split(', ')]
        except:
            return [float(s) for s in string_var.strip('[').strip(']').split(',')]

def _get_local_ip():
        """
        Get the local ip of this device

        :return: Ip of this computer
        :rtype: str
        """
        return set([x[4][0] for x in socket.getaddrinfo(
            socket.gethostname(),
            80,
            socket.AF_INET
        )]).pop()

def _interval_to_bound_points(array):
    """
    Helper function which returns an array
    with the Intervals' boundaries.
    """

    array_boundaries = np.array([x.left for x in array])
    array_boundaries = np.concatenate(
        (array_boundaries, np.array([array[-1].right])))

    return array_boundaries

def log(self, level, msg=None, *args, **kwargs):
        """Writes log out at any arbitray level."""

        return self._log(level, msg, args, kwargs)

def _unordered_iterator(self):
        """
        Return the value of each QuerySet, but also add the '#' property to each
        return item.
        """
        for i, qs in zip(self._queryset_idxs, self._querysets):
            for item in qs:
                setattr(item, '#', i)
                yield item

def boolean(value):
    """
    Configuration-friendly boolean type converter.

    Supports both boolean-valued and string-valued inputs (e.g. from env vars).

    """
    if isinstance(value, bool):
        return value

    if value == "":
        return False

    return strtobool(value)

def group_by(iterable, key_func):
    """Wrap itertools.groupby to make life easier."""
    groups = (
        list(sub) for key, sub in groupby(iterable, key_func)
    )
    return zip(groups, groups)

def isfunc(x):
    """
    Returns `True` if the given value is a function or method object.

    Arguments:
        x (mixed): value to check.

    Returns:
        bool
    """
    return any([
        inspect.isfunction(x) and not asyncio.iscoroutinefunction(x),
        inspect.ismethod(x) and not asyncio.iscoroutinefunction(x)
    ])

def jaccard(c_1, c_2):
    """
    Calculates the Jaccard similarity between two sets of nodes. Called by mroc.

    Inputs:  - c_1: Community (set of nodes) 1.
             - c_2: Community (set of nodes) 2.

    Outputs: - jaccard_similarity: The Jaccard similarity of these two communities.
    """
    nom = np.intersect1d(c_1, c_2).size
    denom = np.union1d(c_1, c_2).size
    return nom/denom

def reload_localzone():
    """Reload the cached localzone. You need to call this if the timezone has changed."""
    global _cache_tz
    _cache_tz = pytz.timezone(get_localzone_name())
    utils.assert_tz_offset(_cache_tz)
    return _cache_tz

def levenshtein_distance_metric(a, b):
    """ 1 - farthest apart (same number of words, all diff). 0 - same"""
    return (levenshtein_distance(a, b) / (2.0 * max(len(a), len(b), 1)))

def __init__(self):
        """Initialize the state of the object"""
        self.state = self.STATE_INITIALIZING
        self.state_start = time.time()

def json_dumps(self, obj):
        """Serializer for consistency"""
        return json.dumps(obj, sort_keys=True, indent=4, separators=(',', ': '))

def uint32_to_uint8(cls, img):
        """
        Cast uint32 RGB image to 4 uint8 channels.
        """
        return np.flipud(img.view(dtype=np.uint8).reshape(img.shape + (4,)))

def json_template(data, template_name, template_context):
    """Old style, use JSONTemplateResponse instead of this.
    """
    html = render_to_string(template_name, template_context)
    data = data or {}
    data['html'] = html
    return HttpResponse(json_encode(data), content_type='application/json')

def endline_semicolon_check(self, original, loc, tokens):
        """Check for semicolons at the end of lines."""
        return self.check_strict("semicolon at end of line", original, loc, tokens)

def timestamping_validate(data, schema):
    """
    Custom validation function which inserts a timestamp for when the
    validation occurred
    """
    jsonschema.validate(data, schema)
    data['timestamp'] = str(time.time())

def get_uniques(l):
    """ Returns a list with no repeated elements.
    """
    result = []

    for i in l:
        if i not in result:
            result.append(i)

    return result

def _spawn_kafka_consumer_thread(self):
        """Spawns a kafka continuous consumer thread"""
        self.logger.debug("Spawn kafka consumer thread""")
        self._consumer_thread = Thread(target=self._consumer_loop)
        self._consumer_thread.setDaemon(True)
        self._consumer_thread.start()

def assert_error(text, check, n=1):
    """Assert that text has n errors of type check."""
    assert_error.description = "No {} error for '{}'".format(check, text)
    assert(check in [error[0] for error in lint(text)])

def send(self, topic, *args, **kwargs):
        """
        Appends the prefix to the topic before sendingf
        """
        prefix_topic = self.heroku_kafka.prefix_topic(topic)
        return super(HerokuKafkaProducer, self).send(prefix_topic, *args, **kwargs)

def assert_error(text, check, n=1):
    """Assert that text has n errors of type check."""
    assert_error.description = "No {} error for '{}'".format(check, text)
    assert(check in [error[0] for error in lint(text)])

def remove_rows_matching(df, column, match):
    """
    Return a ``DataFrame`` with rows where `column` values match `match` are removed.

    The selected `column` series of values from the supplied Pandas ``DataFrame`` is compared
    to `match`, and those rows that match are removed from the DataFrame.

    :param df: Pandas ``DataFrame``
    :param column: Column indexer
    :param match: ``str`` match target
    :return: Pandas ``DataFrame`` filtered
    """
    df = df.copy()
    mask = df[column].values != match
    return df.iloc[mask, :]

def _assert_is_type(name, value, value_type):
    """Assert that a value must be a given type."""
    if not isinstance(value, value_type):
        if type(value_type) is tuple:
            types = ', '.join(t.__name__ for t in value_type)
            raise ValueError('{0} must be one of ({1})'.format(name, types))
        else:
            raise ValueError('{0} must be {1}'
                             .format(name, value_type.__name__))

def get_known_read_position(fp, buffered=True):
    """ 
    Return a position in a file which is known to be read & handled.
    It assumes a buffered file and streaming processing. 
    """
    buffer_size = io.DEFAULT_BUFFER_SIZE if buffered else 0
    return max(fp.tell() - buffer_size, 0)

def make_symmetric(dict):
    """Makes the given dictionary symmetric. Values are assumed to be unique."""
    for key, value in list(dict.items()):
        dict[value] = key
    return dict

def kubectl(*args, input=None, **flags):
    """Simple wrapper to kubectl."""
    # Build command line call.
    line = ['kubectl'] + list(args)
    line = line + get_flag_args(**flags)
    if input is not None:
        line = line + ['-f', '-']
    # Run subprocess
    output = subprocess.run(
        line,
        input=input,
        capture_output=True,
        text=True
    )
    return output

def _get_context(argspec, kwargs):
    """Prepare a context for the serialization.

    :param argspec: The argspec of the serialization function.
    :param kwargs: Dict with context
    :return: Keywords arguments that function can accept.
    """
    if argspec.keywords is not None:
        return kwargs
    return dict((arg, kwargs[arg]) for arg in argspec.args if arg in kwargs)

def straight_line_show(title, length=100, linestyle="=", pad=0):
        """Print a formatted straight line.
        """
        print(StrTemplate.straight_line(
            title=title, length=length, linestyle=linestyle, pad=pad))

def __init__(self,operand,operator,**args):
        """
        Accepts a NumberGenerator operand, an operator, and
        optional arguments to be provided to the operator when calling
        it on the operand.
        """
        # Note that it's currently not possible to set
        # parameters in the superclass when creating an instance,
        # because **args is used by this class itself.
        super(UnaryOperator,self).__init__()

        self.operand=operand
        self.operator=operator
        self.args=args

def extract_all(zipfile, dest_folder):
    """
    reads the zip file, determines compression
    and unzips recursively until source files 
    are extracted 
    """
    z = ZipFile(zipfile)
    print(z)
    z.extract(dest_folder)

def retry_until_not_none_or_limit_reached(method, limit, sleep_s=1,
                                          catch_exceptions=()):
  """Executes a method until the retry limit is hit or not None is returned."""
  return retry_until_valid_or_limit_reached(
      method, limit, lambda x: x is not None, sleep_s, catch_exceptions)

def recursively_update(d, d2):
  """dict.update but which merges child dicts (dict2 takes precedence where there's conflict)."""
  for k, v in d2.items():
    if k in d:
      if isinstance(v, dict):
        recursively_update(d[k], v)
        continue
    d[k] = v

def _linear_interpolation(x, X, Y):
    """Given two data points [X,Y], linearly interpolate those at x.
    """
    return (Y[1] * (x - X[0]) + Y[0] * (X[1] - x)) / (X[1] - X[0])

def increment(self, amount=1):
        """
        Increments the main progress bar by amount.
        """
        self._primaryProgressBar.setValue(self.value() + amount)
        QApplication.instance().processEvents()

def get_tail(self):
        """Gets tail

        :return: Tail of linked list
        """
        node = self.head
        last_node = self.head

        while node is not None:
            last_node = node
            node = node.next_node

        return last_node

def safe_url(url):
    """Remove password from printed connection URLs."""
    parsed = urlparse(url)
    if parsed.password is not None:
        pwd = ':%s@' % parsed.password
        url = url.replace(pwd, ':*****@')
    return url

def put_pidfile( pidfile_path, pid ):
    """
    Put a PID into a pidfile
    """
    with open( pidfile_path, "w" ) as f:
        f.write("%s" % pid)
        os.fsync(f.fileno())

    return

def get(url):
    """Recieving the JSON file from uulm"""
    response = urllib.request.urlopen(url)
    data = response.read()
    data = data.decode("utf-8")
    data = json.loads(data)
    return data

def get_services():
        """
        Retrieve a list of all system services.

        @see: L{get_active_services},
            L{start_service}, L{stop_service},
            L{pause_service}, L{resume_service}

        @rtype:  list( L{win32.ServiceStatusProcessEntry} )
        @return: List of service status descriptors.
        """
        with win32.OpenSCManager(
            dwDesiredAccess = win32.SC_MANAGER_ENUMERATE_SERVICE
            ) as hSCManager:
                try:
                    return win32.EnumServicesStatusEx(hSCManager)
                except AttributeError:
                    return win32.EnumServicesStatus(hSCManager)

def addClassKey(self, klass, key, obj):
        """
        Adds an object to the collection, based on klass and key.

        @param klass: The class of the object.
        @param key: The datastore key of the object.
        @param obj: The loaded instance from the datastore.
        """
        d = self._getClass(klass)

        d[key] = obj

def has_multiline_items(maybe_list: Optional[Sequence[str]]):
    """Check whether one of the items in the list has multiple lines."""
    return maybe_list and any(is_multiline(item) for item in maybe_list)

def _split(value):
    """Split input/output value into two values."""
    if isinstance(value, str):
        # iterable, but not meant for splitting
        return value, value
    try:
        invalue, outvalue = value
    except TypeError:
        invalue = outvalue = value
    except ValueError:
        raise ValueError("Only single values and pairs are allowed")
    return invalue, outvalue

def from_file(filename):
    """
    load an nparray object from a json filename

    @parameter str filename: path to the file
    """
    f = open(filename, 'r')
    j = json.load(f)
    f.close()

    return from_dict(j)

def __add__(self, other):
        """Handle the `+` operator."""
        return self._handle_type(other)(self.value + other.value)

def Load(file):
    """ Loads a model from specified file """
    with open(file, 'rb') as file:
        model = dill.load(file)
        return model

def get_soup(page=''):
    """
    Returns a bs4 object of the page requested
    """
    content = requests.get('%s/%s' % (BASE_URL, page)).text
    return BeautifulSoup(content)

def log_all(self, file):
        """Log all data received from RFLink to file."""
        global rflink_log
        if file == None:
            rflink_log = None
        else:
            log.debug('logging to: %s', file)
            rflink_log = open(file, 'a')

def generate_uuid():
    """Generate a UUID."""
    r_uuid = base64.urlsafe_b64encode(uuid.uuid4().bytes)
    return r_uuid.decode().replace('=', '')

def keys_to_snake_case(camel_case_dict):
    """
    Make a copy of a dictionary with all keys converted to snake case. This is just calls to_snake_case on
    each of the keys in the dictionary and returns a new dictionary.

    :param camel_case_dict: Dictionary with the keys to convert.
    :type camel_case_dict: Dictionary.

    :return: Dictionary with the keys converted to snake case.
    """
    return dict((to_snake_case(key), value) for (key, value) in camel_case_dict.items())

def generate_uuid():
    """Generate a UUID."""
    r_uuid = base64.urlsafe_b64encode(uuid.uuid4().bytes)
    return r_uuid.decode().replace('=', '')

def c2u(name):
    """Convert camelCase (used in PHP) to Python-standard snake_case.

    Src:
    https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case

    Parameters
    ----------
    name: A function or variable name in camelCase

    Returns
    -------
    str: The name in snake_case

    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s1 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    return s1

def is_valid_email(email):
    """
    Check if email is valid
    """
    pattern = re.compile(r'[\w\.-]+@[\w\.-]+[.]\w+')
    return bool(pattern.match(email))

def _interval_to_bound_points(array):
    """
    Helper function which returns an array
    with the Intervals' boundaries.
    """

    array_boundaries = np.array([x.left for x in array])
    array_boundaries = np.concatenate(
        (array_boundaries, np.array([array[-1].right])))

    return array_boundaries

def validate_email(email):
    """
    Validates an email address
    Source: Himanshu Shankar (https://github.com/iamhssingh)
    Parameters
    ----------
    email: str

    Returns
    -------
    bool
    """
    from django.core.validators import validate_email
    from django.core.exceptions import ValidationError
    try:
        validate_email(email)
        return True
    except ValidationError:
        return False

def force_iterable(f):
    """Will make any functions return an iterable objects by wrapping its result in a list."""
    def wrapper(*args, **kwargs):
        r = f(*args, **kwargs)
        if hasattr(r, '__iter__'):
            return r
        else:
            return [r]
    return wrapper

def parse(self, s):
        """
        Parses a date string formatted like ``YYYY-MM-DD``.
        """
        return datetime.datetime.strptime(s, self.date_format).date()

def clear_caches():
    """Jinja2 keeps internal caches for environments and lexers.  These are
    used so that Jinja2 doesn't have to recreate environments and lexers all
    the time.  Normally you don't have to care about that but if you are
    measuring memory consumption you may want to clean the caches.
    """
    from jinja2.environment import _spontaneous_environments
    from jinja2.lexer import _lexer_cache
    _spontaneous_environments.clear()
    _lexer_cache.clear()

def __init__(self, response):
        """Initialize a ResponseException instance.

        :param response: A requests.response instance.

        """
        self.response = response
        super(ResponseException, self).__init__(
            "received {} HTTP response".format(response.status_code)
        )

def venv():
    """Install venv + deps."""
    try:
        import virtualenv  # NOQA
    except ImportError:
        sh("%s -m pip install virtualenv" % PYTHON)
    if not os.path.isdir("venv"):
        sh("%s -m virtualenv venv" % PYTHON)
    sh("venv\\Scripts\\pip install -r %s" % (REQUIREMENTS_TXT))

def stringify_dict_contents(dct):
    """Turn dict keys and values into native strings."""
    return {
        str_if_nested_or_str(k): str_if_nested_or_str(v)
        for k, v in dct.items()
    }

def check_alert(self, text):
    """
    Assert an alert is showing with the given text.
    """

    try:
        alert = Alert(world.browser)
        if alert.text != text:
            raise AssertionError(
                "Alert text expected to be {!r}, got {!r}.".format(
                    text, alert.text))
    except WebDriverException:
        # PhantomJS is kinda poor
        pass

def pdf(x, mu, std):
    """Probability density function (normal distribution)"""
    return (1.0 / (std * sqrt(2 * pi))) * np.exp(-(x - mu) ** 2 / (2 * std ** 2))

def adapter(data, headers, **kwargs):
    """Wrap vertical table in a function for TabularOutputFormatter."""
    keys = ('sep_title', 'sep_character', 'sep_length')
    return vertical_table(data, headers, **filter_dict_by_key(kwargs, keys))

def out(self, output, newline=True):
        """Outputs a string to the console (stdout)."""
        click.echo(output, nl=newline)

def check_by_selector(self, selector):
    """Check the checkbox matching the CSS selector."""
    elem = find_element_by_jquery(world.browser, selector)
    if not elem.is_selected():
        elem.click()

def timestamp_to_datetime(timestamp):
    """Convert an ARF timestamp to a datetime.datetime object (naive local time)"""
    from datetime import datetime, timedelta
    obj = datetime.fromtimestamp(timestamp[0])
    return obj + timedelta(microseconds=int(timestamp[1]))

def check_by_selector(self, selector):
    """Check the checkbox matching the CSS selector."""
    elem = find_element_by_jquery(world.browser, selector)
    if not elem.is_selected():
        elem.click()

def map(cls, iterable, func, *a, **kw):
    """
    Iterable-first replacement of Python's built-in `map()` function.
    """

    return cls(func(x, *a, **kw) for x in iterable)

def _close_websocket(self):
        """Closes the websocket connection."""
        close_method = getattr(self._websocket, "close", None)
        if callable(close_method):
            asyncio.ensure_future(close_method(), loop=self._event_loop)
        self._websocket = None
        self._dispatch_event(event="close")

def _group_dict_set(iterator):
    """Make a dict that accumulates the values for each key in an iterator of doubles.

    :param iter[tuple[A,B]] iterator: An iterator
    :rtype: dict[A,set[B]]
    """
    d = defaultdict(set)
    for key, value in iterator:
        d[key].add(value)
    return dict(d)

def _increase_file_handle_limit():
    """Raise the open file handles permitted by the Dusty daemon process
    and its child processes. The number we choose here needs to be within
    the OS X default kernel hard limit, which is 10240."""
    logging.info('Increasing file handle limit to {}'.format(constants.FILE_HANDLE_LIMIT))
    resource.setrlimit(resource.RLIMIT_NOFILE,
                       (constants.FILE_HANDLE_LIMIT, resource.RLIM_INFINITY))

def asMaskedArray(self):
        """ Creates converts to a masked array
        """
        return ma.masked_array(data=self.data, mask=self.mask, fill_value=self.fill_value)

def acquire_nix(lock_file):  # pragma: no cover
    """Acquire a lock file on linux or osx."""
    fd = os.open(lock_file, OPEN_MODE)

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (IOError, OSError):
        os.close(fd)
    else:
        return fd

def array(self):
        """
        return the underlying numpy array
        """
        return np.arange(self.start, self.stop, self.step)

def __Logout(si):
   """
   Disconnect (logout) service instance
   @param si: Service instance (returned from Connect)
   """
   try:
      if si:
         content = si.RetrieveContent()
         content.sessionManager.Logout()
   except Exception as e:
      pass

def is_square_matrix(mat):
    """Test if an array is a square matrix."""
    mat = np.array(mat)
    if mat.ndim != 2:
        return False
    shape = mat.shape
    return shape[0] == shape[1]

def save_dict_to_file(filename, dictionary):
  """Saves dictionary as CSV file."""
  with open(filename, 'w') as f:
    writer = csv.writer(f)
    for k, v in iteritems(dictionary):
      writer.writerow([str(k), str(v)])

def argmax(l,f=None):
    """http://stackoverflow.com/questions/5098580/implementing-argmax-in-python"""
    if f:
        l = [f(i) for i in l]
    return max(enumerate(l), key=lambda x:x[1])[0]

def _write_color_colorama (fp, text, color):
    """Colorize text with given color."""
    foreground, background, style = get_win_color(color)
    colorama.set_console(foreground=foreground, background=background,
      style=style)
    fp.write(text)
    colorama.reset_console()

def SegmentMax(a, ids):
    """
    Segmented max op.
    """
    func = lambda idxs: np.amax(a[idxs], axis=0)
    return seg_map(func, a, ids),

def write_fits(data, header, file_name):
    """
    Combine data and a fits header to write a fits file.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be written.

    header : astropy.io.fits.hduheader
        The header for the fits file.

    file_name : string
        The file to write

    Returns
    -------
    None
    """
    hdu = fits.PrimaryHDU(data)
    hdu.header = header
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(file_name, overwrite=True)
    logging.info("Wrote {0}".format(file_name))
    return

def md5_string(s):
    """
    Shortcut to create md5 hash
    :param s:
    :return:
    """
    m = hashlib.md5()
    m.update(s)
    return str(m.hexdigest())

def write_string(value, buff, byteorder='big'):
    """Write a string to a file-like object."""
    data = value.encode('utf-8')
    write_numeric(USHORT, len(data), buff, byteorder)
    buff.write(data)

def get_file_md5sum(path):
    """Calculate the MD5 hash for a file."""
    with open(path, 'rb') as fh:
        h = str(hashlib.md5(fh.read()).hexdigest())
    return h

def set_icon(self, bmp):
        """Sets main window icon to given wx.Bitmap"""

        _icon = wx.EmptyIcon()
        _icon.CopyFromBitmap(bmp)
        self.SetIcon(_icon)

def start_task(self, task):
        """Begin logging of a task

        Stores the time this task was started in order to return
        time lapsed when `complete_task` is called.

        Parameters
        ----------
        task : str
            Name of the task to be started
        """
        self.info("Calculating {}...".format(task))
        self.tasks[task] = self.timer()

def __call__(self, xy):
        """Project x and y"""
        x, y = xy
        return (self.x(x), self.y(y))

def merge(self, obj):
        """This function merge another object's values with this instance

        :param obj: An object to be merged with into this layer
        :type obj: object
        """
        for attribute in dir(obj):
            if '__' in attribute:
                continue
            setattr(self, attribute, getattr(obj, attribute))

def elXpath(self, xpath, dom=None):
        """check if element is present by css"""
        if dom is None:
            dom = self.browser
        return expect(dom.is_element_present_by_xpath, args=[xpath])

def dict_merge(set1, set2):
    """Joins two dictionaries."""
    return dict(list(set1.items()) + list(set2.items()))

def serialize_yaml_tofile(filename, resource):
    """
    Serializes a K8S resource to YAML-formatted file.
    """
    stream = file(filename, "w")
    yaml.dump(resource, stream, default_flow_style=False)

def fn_min(self, a, axis=None):
        """
        Return the minimum of an array, ignoring any NaNs.

        :param a: The array.
        :return: The minimum value of the array.
        """

        return numpy.nanmin(self._to_ndarray(a), axis=axis)

def guess_title(basename):
    """ Attempt to guess the title from the filename """

    base, _ = os.path.splitext(basename)
    return re.sub(r'[ _-]+', r' ', base).title()

def _obj_cursor_to_dictionary(self, cursor):
        """Handle conversion of pymongo cursor into a JSON object formatted for UI consumption

        :param dict cursor: a mongo document that should be converted to primitive types for the client code
        :returns: a primitive dictionary
        :rtype: dict
        """
        if not cursor:
            return cursor

        cursor = json.loads(json.dumps(cursor, cls=BSONEncoder))

        if cursor.get("_id"):
            cursor["id"] = cursor.get("_id")
            del cursor["_id"]

        return cursor

def dict_to_html_attrs(dict_):
    """
    Banana banana
    """
    res = ' '.join('%s="%s"' % (k, v) for k, v in dict_.items())
    return res

def mostCommonItem(lst):
    """Choose the most common item from the list, or the first item if all
    items are unique."""
    # This elegant solution from: http://stackoverflow.com/a/1518632/1760218
    lst = [l for l in lst if l]
    if lst:
        return max(set(lst), key=lst.count)
    else:
        return None

def extract_module_locals(depth=0):
    """Returns (module, locals) of the funciton `depth` frames away from the caller"""
    f = sys._getframe(depth + 1)
    global_ns = f.f_globals
    module = sys.modules[global_ns['__name__']]
    return (module, f.f_locals)

def _most_common(iterable):
    """Returns the most common element in `iterable`."""
    data = Counter(iterable)
    return max(data, key=data.__getitem__)

def astype(array, y):
  """A functional form of the `astype` method.

  Args:
    array: The array or number to cast.
    y: An array or number, as the input, whose type should be that of array.

  Returns:
    An array or number with the same dtype as `y`.
  """
  if isinstance(y, autograd.core.Node):
    return array.astype(numpy.array(y.value).dtype)
  return array.astype(numpy.array(y).dtype)

def dict_merge(set1, set2):
    """Joins two dictionaries."""
    return dict(list(set1.items()) + list(set2.items()))

def save(self):
        """save the current session
        override, if session was saved earlier"""
        if self.path:
            self._saveState(self.path)
        else:
            self.saveAs()

def user_return(self, frame, return_value):
        """This function is called when a return trap is set here."""
        pdb.Pdb.user_return(self, frame, return_value)

def move_up(lines=1, file=sys.stdout):
    """ Move the cursor up a number of lines.

        Esc[ValueA:
        Moves the cursor up by the specified number of lines without changing
        columns. If the cursor is already on the top line, ANSI.SYS ignores
        this sequence.
    """
    move.up(lines).write(file=file)

def string_to_identity(identity_str):
    """Parse string into Identity dictionary."""
    m = _identity_regexp.match(identity_str)
    result = m.groupdict()
    log.debug('parsed identity: %s', result)
    return {k: v for k, v in result.items() if v}

def move_up(lines=1, file=sys.stdout):
    """ Move the cursor up a number of lines.

        Esc[ValueA:
        Moves the cursor up by the specified number of lines without changing
        columns. If the cursor is already on the top line, ANSI.SYS ignores
        this sequence.
    """
    move.up(lines).write(file=file)

def get_url_args(url):
    """ Returns a dictionary from a URL params """
    url_data = urllib.parse.urlparse(url)
    arg_dict = urllib.parse.parse_qs(url_data.query)
    return arg_dict

def _parse_array(self, tensor_proto):
        """Grab data in TensorProto and convert to numpy array."""
        try:
            from onnx.numpy_helper import to_array
        except ImportError as e:
            raise ImportError("Unable to import onnx which is required {}".format(e))
        np_array = to_array(tensor_proto).reshape(tuple(tensor_proto.dims))
        return mx.nd.array(np_array)

def list2dict(list_of_options):
    """Transforms a list of 2 element tuples to a dictionary"""
    d = {}
    for key, value in list_of_options:
        d[key] = value
    return d

def unit_ball_L2(shape):
  """A tensorflow variable tranfomed to be constrained in a L2 unit ball.

  EXPERIMENTAL: Do not use for adverserial examples if you need to be confident
  they are strong attacks. We are not yet confident in this code.
  """
  x = tf.Variable(tf.zeros(shape))
  return constrain_L2(x)

def to_dicts(recarray):
    """convert record array to a dictionaries"""
    for rec in recarray:
        yield dict(zip(recarray.dtype.names, rec.tolist()))

def to_distribution_values(self, values):
        """
        Returns numpy array of natural logarithms of ``values``.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # avoid RuntimeWarning: divide by zero encountered in log
            return numpy.log(values)

def dt_to_ts(value):
    """ If value is a datetime, convert to timestamp """
    if not isinstance(value, datetime):
        return value
    return calendar.timegm(value.utctimetuple()) + value.microsecond / 1000000.0

def log_loss(preds, labels):
    """Logarithmic loss with non-necessarily-binary labels."""
    log_likelihood = np.sum(labels * np.log(preds)) / len(preds)
    return -log_likelihood

def filehash(path):
    """Make an MD5 hash of a file, ignoring any differences in line
    ending characters."""
    with open(path, "rU") as f:
        return md5(py3compat.str_to_bytes(f.read())).hexdigest()

def full_like(array, value, dtype=None):
    """ Create a shared memory array with the same shape and type as a given array, filled with `value`.
    """
    shared = empty_like(array, dtype)
    shared[:] = value
    return shared

def caller_locals():
    """Get the local variables in the caller's frame."""
    import inspect
    frame = inspect.currentframe()
    try:
        return frame.f_back.f_back.f_locals
    finally:
        del frame

def last_modified_date(filename):
    """Last modified timestamp as a UTC datetime"""
    mtime = os.path.getmtime(filename)
    dt = datetime.datetime.utcfromtimestamp(mtime)
    return dt.replace(tzinfo=pytz.utc)

def mag(z):
    """Get the magnitude of a vector."""
    if isinstance(z[0], np.ndarray):
        return np.array(list(map(np.linalg.norm, z)))
    else:
        return np.linalg.norm(z)

def get_obj(ref):
    """Get object from string reference."""
    oid = int(ref)
    return server.id2ref.get(oid) or server.id2obj[oid]

def normalize_path(path):
    """
    Convert a path to its canonical, case-normalized, absolute version.

    """
    return os.path.normcase(os.path.realpath(os.path.expanduser(path)))

def enable_gtk3(self, app=None):
        """Enable event loop integration with Gtk3 (gir bindings).

        Parameters
        ----------
        app : ignored
           Ignored, it's only a placeholder to keep the call signature of all
           gui activation methods consistent, which simplifies the logic of
           supporting magics.

        Notes
        -----
        This methods sets the PyOS_InputHook for Gtk3, which allows
        the Gtk3 to integrate with terminal based applications like
        IPython.
        """
        from pydev_ipython.inputhookgtk3 import create_inputhook_gtk3
        self.set_inputhook(create_inputhook_gtk3(self._stdin_file))
        self._current_gui = GUI_GTK

def v_normalize(v):
    """
    Normalizes the given vector.
    
    The vector given may have any number of dimensions.
    """
    vmag = v_magnitude(v)
    return [ v[i]/vmag  for i in range(len(v)) ]

def get_table_names(connection):
	"""
	Return a list of the table names in the database.
	"""
	cursor = connection.cursor()
	cursor.execute("SELECT name FROM sqlite_master WHERE type == 'table'")
	return [name for (name,) in cursor]

def _to_array(value):
    """As a convenience, turn Python lists and tuples into NumPy arrays."""
    if isinstance(value, (tuple, list)):
        return array(value)
    elif isinstance(value, (float, int)):
        return np.float64(value)
    else:
        return value

def getvariable(name):
    """Get the value of a local variable somewhere in the call stack."""
    import inspect
    fr = inspect.currentframe()
    try:
        while fr:
            fr = fr.f_back
            vars = fr.f_locals
            if name in vars:
                return vars[name]
    except:
        pass
    return None

def _histplot_bins(column, bins=100):
    """Helper to get bins for histplot."""
    col_min = np.min(column)
    col_max = np.max(column)
    return range(col_min, col_max + 2, max((col_max - col_min) // bins, 1))

def _is_target_a_directory(link, rel_target):
	"""
	If creating a symlink from link to a target, determine if target
	is a directory (relative to dirname(link)).
	"""
	target = os.path.join(os.path.dirname(link), rel_target)
	return os.path.isdir(target)

def merge(self, other):
        """
        Merge this range object with another (ranges need not overlap or abut).

        :returns: a new Range object representing the interval containing both
                  ranges.
        """
        newstart = min(self._start, other.start)
        newend = max(self._end, other.end)
        return Range(newstart, newend)

def algo_exp(x, m, t, b):
    """mono-exponential curve."""
    return m*np.exp(-t*x)+b

def to_str(obj):
    """Attempts to convert given object to a string object
    """
    if not isinstance(obj, str) and PY3 and isinstance(obj, bytes):
        obj = obj.decode('utf-8')
    return obj if isinstance(obj, string_types) else str(obj)

def _array2cstr(arr):
    """ Serializes a numpy array to a compressed base64 string """
    out = StringIO()
    np.save(out, arr)
    return b64encode(out.getvalue())

def one_hot(x, size, dtype=np.float32):
  """Make a n+1 dim one-hot array from n dim int-categorical array."""
  return np.array(x[..., np.newaxis] == np.arange(size), dtype)

def loadb(b):
    """Deserialize ``b`` (instance of ``bytes``) to a Python object."""
    assert isinstance(b, (bytes, bytearray))
    return std_json.loads(b.decode('utf-8'))

def read_numpy(fh, byteorder, dtype, count, offsetsize):
    """Read tag data from file and return as numpy array."""
    dtype = 'b' if dtype[-1] == 's' else byteorder+dtype[-1]
    return fh.read_array(dtype, count)

def fetch(self):
        """
        Fetch & return a new `Domain` object representing the domain's current
        state

        :rtype: Domain
        :raises DOAPIError: if the API endpoint replies with an error (e.g., if
            the domain no longer exists)
        """
        api = self.doapi_manager
        return api._domain(api.request(self.url)["domain"])

def run (self):
        """Handle keyboard interrupt and other errors."""
        try:
            self.run_checked()
        except KeyboardInterrupt:
            thread.interrupt_main()
        except Exception:
            self.internal_error()

def autoconvert(string):
    """Try to convert variables into datatypes."""
    for fn in (boolify, int, float):
        try:
            return fn(string)
        except ValueError:
            pass
    return string

def _stdin_ready_posix():
    """Return True if there's something to read on stdin (posix version)."""
    infds, outfds, erfds = select.select([sys.stdin],[],[],0)
    return bool(infds)

def list_i2str(ilist):
    """
    Convert an integer list into a string list.
    """
    slist = []
    for el in ilist:
        slist.append(str(el))
    return slist

def askopenfilename(**kwargs):
    """Return file name(s) from Tkinter's file open dialog."""
    try:
        from Tkinter import Tk
        import tkFileDialog as filedialog
    except ImportError:
        from tkinter import Tk, filedialog
    root = Tk()
    root.withdraw()
    root.update()
    filenames = filedialog.askopenfilename(**kwargs)
    root.destroy()
    return filenames

def open_file(file, mode):
	"""Open a file.

	:arg file: file-like or path-like object.
	:arg str mode: ``mode`` argument for :func:`open`.
	"""
	if hasattr(file, "read"):
		return file
	if hasattr(file, "open"):
		return file.open(mode)
	return open(file, mode)

def timedelta_seconds(timedelta):
    """Returns the total timedelta duration in seconds."""
    return (timedelta.total_seconds() if hasattr(timedelta, "total_seconds")
            else timedelta.days * 24 * 3600 + timedelta.seconds +
                 timedelta.microseconds / 1000000.)

def load_image(fname):
    """ read an image from file - PIL doesnt close nicely """
    with open(fname, "rb") as f:
        i = Image.open(fname)
        #i.load()
        return i

def re_raise(self):
        """ Raise this exception with the original traceback """
        if self.exc_info is not None:
            six.reraise(type(self), self, self.exc_info[2])
        else:
            raise self

def load(filename):
    """
    Load the state from the given file, moving to the file's directory during
    load (temporarily, moving back after loaded)

    Parameters
    ----------
    filename : string
        name of the file to open, should be a .pkl file
    """
    path, name = os.path.split(filename)
    path = path or '.'

    with util.indir(path):
        return pickle.load(open(name, 'rb'))

def remote_file_exists(self, url):
        """ Checks whether the remote file exists.

        :param url:
            The url that has to be checked.
        :type url:
            String

        :returns:
            **True** if remote file exists and **False** if it doesn't exist.
        """
        status = requests.head(url).status_code

        if status != 200:
            raise RemoteFileDoesntExist

def get_order(self, codes):
        """Return evidence codes in order shown in code2name."""
        return sorted(codes, key=lambda e: [self.ev2idx.get(e)])

def insort_no_dup(lst, item):
    """
    If item is not in lst, add item to list at its sorted position
    """
    import bisect
    ix = bisect.bisect_left(lst, item)
    if lst[ix] != item: 
        lst[ix:ix] = [item]

def merge(left, right, how='inner', key=None, left_key=None, right_key=None,
          left_as='left', right_as='right'):
    """ Performs a join using the union join function. """
    return join(left, right, how, key, left_key, right_key,
                join_fn=make_union_join(left_as, right_as))

def _fetch_all_as_dict(self, cursor):
        """
        Iterates over the result set and converts each row to a dictionary

        :return: A list of dictionaries where each row is a dictionary
        :rtype: list of dict
        """
        desc = cursor.description
        return [
            dict(zip([col[0] for col in desc], row))
            for row in cursor.fetchall()
        ]

def tab(self, output):
        """Output data in excel-compatible tab-delimited format"""
        import csv
        csvwriter = csv.writer(self.outfile, dialect=csv.excel_tab)
        csvwriter.writerows(output)

def test():
    """Test for ReverseDNS class"""
    dns = ReverseDNS()

    print(dns.lookup('192.168.0.1'))
    print(dns.lookup('8.8.8.8'))

    # Test cache
    print(dns.lookup('8.8.8.8'))

def add_widgets(self, *widgets_or_spacings):
        """Add widgets/spacing to dialog vertical layout"""
        layout = self.layout()
        for widget_or_spacing in widgets_or_spacings:
            if isinstance(widget_or_spacing, int):
                layout.addSpacing(widget_or_spacing)
            else:
                layout.addWidget(widget_or_spacing)

def _sort_r(sorted, processed, key, deps, dependency_tree):
    """Recursive topological sort implementation."""
    if key in processed:
        return
    processed.add(key)
    for dep_key in deps:
        dep_deps = dependency_tree.get(dep_key)
        if dep_deps is None:
            log.debug('"%s" not found, skipped', Repr(dep_key))
            continue
        _sort_r(sorted, processed, dep_key, dep_deps, dependency_tree)
    sorted.append((key, deps))

def save_list(key, *values):
    """Convert the given list of parameters to a JSON object.

    JSON object is of the form:
    { key: [values[0], values[1], ... ] },
    where values represent the given list of parameters.

    """
    return json.dumps({key: [_get_json(value) for value in values]})

def readwav(filename):
    """Read a WAV file and returns the data and sample rate

    ::

        from spectrum.io import readwav
        readwav()

    """
    from scipy.io.wavfile import read as readwav
    samplerate, signal = readwav(filename)
    return signal, samplerate

def iget_list_column_slice(list_, start=None, stop=None, stride=None):
    """ iterator version of get_list_column """
    if isinstance(start, slice):
        slice_ = start
    else:
        slice_ = slice(start, stop, stride)
    return (row[slice_] for row in list_)

def rnormal(mu, tau, size=None):
    """
    Random normal variates.
    """
    return np.random.normal(mu, 1. / np.sqrt(tau), size)

def dimensions(self):
        """Get width and height of a PDF"""
        size = self.pdf.getPage(0).mediaBox
        return {'w': float(size[2]), 'h': float(size[3])}

def algo_exp(x, m, t, b):
    """mono-exponential curve."""
    return m*np.exp(-t*x)+b

def double_sha256(data):
    """A standard compound hash."""
    return bytes_as_revhex(hashlib.sha256(hashlib.sha256(data).digest()).digest())

def load_data(filename):
    """
    :rtype : numpy matrix
    """
    data = pandas.read_csv(filename, header=None, delimiter='\t', skiprows=9)
    return data.as_matrix()

def unpickle(pickle_file):
    """Unpickle a python object from the given path."""
    pickle = None
    with open(pickle_file, "rb") as pickle_f:
        pickle = dill.load(pickle_f)
    if not pickle:
        LOG.error("Could not load python object from file")
    return pickle

def file_to_str(fname):
    """
    Read a file into a string
    PRE: fname is a small file (to avoid hogging memory and its discontents)
    """
    data = None
    # rU = read with Universal line terminator
    with open(fname, 'rU') as fd:
        data = fd.read()
    return data

def sinwave(n=4,inc=.25):
	"""
	Returns a DataFrame with the required format for 
	a surface (sine wave) plot

	Parameters:
	-----------
		n : int
			Ranges for X and Y axis (-n,n)
		n_y : int
			Size of increment along the axis
	"""	
	x=np.arange(-n,n,inc)
	y=np.arange(-n,n,inc)
	X,Y=np.meshgrid(x,y)
	R = np.sqrt(X**2 + Y**2)
	Z = np.sin(R)/(.5*R)
	return pd.DataFrame(Z,index=x,columns=y)

def get_property(self, filename):
        """Opens the file and reads the value"""

        with open(self.filepath(filename)) as f:
            return f.read().strip()

def plot(self):
        """Plot the empirical histogram versus best-fit distribution's PDF."""
        plt.plot(self.bin_edges, self.hist, self.bin_edges, self.best_pdf)

def _readuntil(f, end=_TYPE_END):
	"""Helper function to read bytes until a certain end byte is hit"""
	buf = bytearray()
	byte = f.read(1)
	while byte != end:
		if byte == b'':
			raise ValueError('File ended unexpectedly. Expected end byte {}.'.format(end))
		buf += byte
		byte = f.read(1)
	return buf

def log_magnitude_spectrum(frames):
    """Compute the log of the magnitude spectrum of frames"""
    return N.log(N.abs(N.fft.rfft(frames)).clip(1e-5, N.inf))

def get_serial_number_string(self):
        """ Get the Serial Number String from the HID device.

        :return:    The Serial Number String
        :rtype:     unicode

        """
        self._check_device_status()
        str_p = ffi.new("wchar_t[]", 255)
        rv = hidapi.hid_get_serial_number_string(self._device, str_p, 255)
        if rv == -1:
            raise IOError("Failed to read serial number string from HID "
                          "device: {0}".format(self._get_last_error_string()))
        return ffi.string(str_p)

def confusion_matrix(self):
        """Confusion matrix plot
        """
        return plot.confusion_matrix(self.y_true, self.y_pred,
                                     self.target_names, ax=_gen_ax())

def readTuple(self, line, n=3):
        """ Reads a tuple of numbers. e.g. vertices, normals or teture coords.
        """
        numbers = [num for num in line.split(' ') if num]
        return [float(num) for num in numbers[1:n + 1]]

def pop(self, index=-1):
		"""Remove and return the item at index."""
		value = self._list.pop(index)
		del self._dict[value]
		return value

def _get_config_or_default(self, key, default, as_type=lambda x: x):
        """Return a main config value, or default if it does not exist."""

        if self.main_config.has_option(self.main_section, key):
            return as_type(self.main_config.get(self.main_section, key))
        return default

def caller_locals():
    """Get the local variables in the caller's frame."""
    import inspect
    frame = inspect.currentframe()
    try:
        return frame.f_back.f_back.f_locals
    finally:
        del frame

def get_mnist(data_type="train", location="/tmp/mnist"):
    """
    Get mnist dataset with features and label as ndarray.
    Data would be downloaded automatically if it doesn't present at the specific location.

    :param data_type: "train" for training data and "test" for testing data.
    :param location: Location to store mnist dataset.
    :return: (features: ndarray, label: ndarray)
    """
    X, Y = mnist.read_data_sets(location, data_type)
    return X, Y + 1

def append(self, item):
        """ append item and print it to stdout """
        print(item)
        super(MyList, self).append(item)

def loads(s, model=None, parser=None):
    """Deserialize s (a str) to a Python object."""
    with StringIO(s) as f:
        return load(f, model=model, parser=parser)

def printdict(adict):
    """printdict"""
    dlist = list(adict.keys())
    dlist.sort()
    for i in range(0, len(dlist)):
        print(dlist[i], adict[dlist[i]])

def readwav(filename):
    """Read a WAV file and returns the data and sample rate

    ::

        from spectrum.io import readwav
        readwav()

    """
    from scipy.io.wavfile import read as readwav
    samplerate, signal = readwav(filename)
    return signal, samplerate

def print_matrix(X, decimals=1):
    """Pretty printing for numpy matrix X"""
    for row in np.round(X, decimals=decimals):
        print(row)

def loadmat(filename):
    """This function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = sploadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def pprint(self, seconds):
        """
        Pretty Prints seconds as Hours:Minutes:Seconds.MilliSeconds

        :param seconds:  The time in seconds.
        """
        return ("%d:%02d:%02d.%03d", reduce(lambda ll, b: divmod(ll[0], b) + ll[1:], [(seconds * 1000,), 1000, 60, 60]))

def readTuple(self, line, n=3):
        """ Reads a tuple of numbers. e.g. vertices, normals or teture coords.
        """
        numbers = [num for num in line.split(' ') if num]
        return [float(num) for num in numbers[1:n + 1]]

def print(*a):
    """ print just one that returns what you give it instead of None """
    try:
        _print(*a)
        return a[0] if len(a) == 1 else a
    except:
        _print(*a)

def ReadManyFromPath(filepath):
  """Reads a Python object stored in a specified YAML file.

  Args:
    filepath: A filepath to the YAML file.

  Returns:
    A Python data structure corresponding to the YAML in the given file.
  """
  with io.open(filepath, mode="r", encoding="utf-8") as filedesc:
    return ReadManyFromFile(filedesc)

def __PrintEnumDocstringLines(self, enum_type):
        description = enum_type.description or '%s enum type.' % enum_type.name
        for line in textwrap.wrap('r"""%s' % description,
                                  self.__printer.CalculateWidth()):
            self.__printer(line)
        PrintIndentedDescriptions(self.__printer, enum_type.values, 'Values')
        self.__printer('"""')

def map_keys_deep(f, dct):
    """
    Implementation of map that recurses. This tests the same keys at every level of dict and in lists
    :param f: 2-ary function expecting a key and value and returns a modified key
    :param dct: Dict for deep processing
    :return: Modified dct with matching props mapped
    """
    return _map_deep(lambda k, v: [f(k, v), v], dct)

def index():
    """ Display productpage with normal user and test user buttons"""
    global productpage

    table = json2html.convert(json = json.dumps(productpage),
                              table_attributes="class=\"table table-condensed table-bordered table-hover\"")

    return render_template('index.html', serviceTable=table)

def logout():
    """ Log out the active user
    """
    flogin.logout_user()
    next = flask.request.args.get('next')
    return flask.redirect(next or flask.url_for("user"))

def head_and_tail_print(self, n=5):
        """Display the first and last n elements of a DataFrame."""
        from IPython import display
        display.display(display.HTML(self._head_and_tail_table(n)))

def get_instance(key, expire=None):
    """Return an instance of RedisSet."""
    global _instances
    try:
        instance = _instances[key]
    except KeyError:
        instance = RedisSet(
            key,
            _redis,
            expire=expire
        )
        _instances[key] = instance

    return instance

def update_dict(obj, dict, attributes):
    """Update dict with fields from obj.attributes.

    :param obj: the object updated into dict
    :param dict: the result dictionary
    :param attributes: a list of attributes belonging to obj
    """
    for attribute in attributes:
        if hasattr(obj, attribute) and getattr(obj, attribute) is not None:
            dict[attribute] = getattr(obj, attribute)

def rpop(self, key):
        """Emulate lpop."""
        redis_list = self._get_list(key, 'RPOP')

        if self._encode(key) not in self.redis:
            return None

        try:
            value = redis_list.pop()
            if len(redis_list) == 0:
                self.delete(key)
            return value
        except (IndexError):
            # Redis returns nil if popping from an empty list
            return None

def property_as_list(self, property_name):
        """ property() but encapsulates it in a list, if it's a
        single-element property.
        """
        try:
            res = self._a_tags[property_name]
        except KeyError:
            return []

        if type(res) == list:
            return res
        else:
            return [res]

def toJson(protoObject, indent=None):
    """
    Serialises a protobuf object as json
    """
    # Using the internal method because this way we can reformat the JSON
    js = json_format.MessageToDict(protoObject, False)
    return json.dumps(js, indent=indent)

def calculate_top_margin(self):
		"""
		Calculate the margin in pixels above the plot area, setting
		border_top.
		"""
		self.border_top = 5
		if self.show_graph_title:
			self.border_top += self.title_font_size
		self.border_top += 5
		if self.show_graph_subtitle:
			self.border_top += self.subtitle_font_size

def reindex_axis(self, labels, axis=0, **kwargs):
        """
        Conform Series to new index with optional filling logic.

        .. deprecated:: 0.21.0
            Use ``Series.reindex`` instead.
        """
        # for compatibility with higher dims
        if axis != 0:
            raise ValueError("cannot reindex series on non-zero axis!")
        msg = ("'.reindex_axis' is deprecated and will be removed in a future "
               "version. Use '.reindex' instead.")
        warnings.warn(msg, FutureWarning, stacklevel=2)

        return self.reindex(index=labels, **kwargs)

def listen_for_updates(self):
        """Attach a callback on the group pubsub"""
        self.toredis.subscribe(self.group_pubsub, callback=self.callback)

def remove_file_from_s3(awsclient, bucket, key):
    """Remove a file from an AWS S3 bucket.

    :param awsclient:
    :param bucket:
    :param key:
    :return:
    """
    client_s3 = awsclient.get_client('s3')
    response = client_s3.delete_object(Bucket=bucket, Key=key)

def _heappush_max(heap, item):
    """ why is this not in heapq """
    heap.append(item)
    heapq._siftdown_max(heap, 0, len(heap) - 1)

def clean_py_files(path):
    """
    Removes all .py files.

    :param path: the path
    :return: None
    """

    for dirname, subdirlist, filelist in os.walk(path):

        for f in filelist:
            if f.endswith('py'):
                os.remove(os.path.join(dirname, f))

def from_pydatetime(cls, pydatetime):
        """
        Creates sql datetime2 object from Python datetime object
        ignoring timezone
        @param pydatetime: Python datetime object
        @return: sql datetime2 object
        """
        return cls(date=Date.from_pydate(pydatetime.date),
                   time=Time.from_pytime(pydatetime.time))

def unique(seq):
    """Return the unique elements of a collection even if those elements are
       unhashable and unsortable, like dicts and sets"""
    cleaned = []
    for each in seq:
        if each not in cleaned:
            cleaned.append(each)
    return cleaned

def locate(command, on):
    """Locate the command's man page."""
    location = find_page_location(command, on)
    click.echo(location)

def strip_accents(text):
    """
    Strip agents from a string.
    """

    normalized_str = unicodedata.normalize('NFD', text)

    return ''.join([
        c for c in normalized_str if unicodedata.category(c) != 'Mn'])

def value_left(self, other):
    """
    Returns the value of the other type instance to use in an
    operator method, namely when the method's instance is on the
    left side of the expression.
    """
    return other.value if isinstance(other, self.__class__) else other

def dedup_list(l):
    """Given a list (l) will removing duplicates from the list,
       preserving the original order of the list. Assumes that
       the list entrie are hashable."""
    dedup = set()
    return [ x for x in l if not (x in dedup or dedup.add(x))]

def imapchain(*a, **kwa):
    """ Like map but also chains the results. """

    imap_results = map( *a, **kwa )
    return itertools.chain( *imap_results )

def remove_element(self, e):
        """Remove element `e` from model
        """
        
        if e.label is not None: self.elementdict.pop(e.label)
        self.elementlist.remove(e)

def block_view(arr, block=(3, 3)):
    """Provide a 2D block view to 2D array.

    No error checking made. Therefore meaningful (as implemented) only for
    blocks strictly compatible with the shape of A.

    """

    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape = (arr.shape[0] // block[0], arr.shape[1] // block[1]) + block
    strides = (block[0] * arr.strides[0], block[1] * arr.strides[1]) + arr.strides
    return ast(arr, shape=shape, strides=strides)

def rfft2d_freqs(h, w):
    """Computes 2D spectrum frequencies."""

    fy = np.fft.fftfreq(h)[:, None]
    # when we have an odd input dimension we need to keep one additional
    # frequency and later cut off 1 pixel
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[: w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w)[: w // 2 + 1]
    return np.sqrt(fx * fx + fy * fy)

def strip_html(string, keep_tag_content=False):
    """
    Remove html code contained into the given string.

    :param string: String to manipulate.
    :type string: str
    :param keep_tag_content: True to preserve tag content, False to remove tag and its content too (default).
    :type keep_tag_content: bool
    :return: String with html removed.
    :rtype: str
    """
    r = HTML_TAG_ONLY_RE if keep_tag_content else HTML_RE
    return r.sub('', string)

def get(s, delimiter='', format="diacritical"):
    """Return pinyin of string, the string must be unicode
    """
    return delimiter.join(_pinyin_generator(u(s), format=format))

def pop(h):
    """Pop the heap value from the heap."""
    n = h.size() - 1
    h.swap(0, n)
    down(h, 0, n)
    return h.pop()

def _match_literal(self, a, b=None):
        """Match two names."""

        return a.lower() == b if not self.case_sensitive else a == b

def drop_trailing_zeros(num):
    """
    Drops the trailing zeros in a float that is printed.
    """
    txt = '%f' %(num)
    txt = txt.rstrip('0')
    if txt.endswith('.'):
        txt = txt[:-1]
    return txt

def ub_to_str(string):
    """
    converts py2 unicode / py3 bytestring into str
    Args:
        string (unicode, byte_string): string to be converted
        
    Returns:
        (str)
    """
    if not isinstance(string, str):
        if six.PY2:
            return str(string)
        else:
            return string.decode()
    return string

def clean_df(df, fill_nan=True, drop_empty_columns=True):
    """Clean a pandas dataframe by:
        1. Filling empty values with Nan
        2. Dropping columns with all empty values

    Args:
        df: Pandas DataFrame
        fill_nan (bool): If any empty values (strings, None, etc) should be replaced with NaN
        drop_empty_columns (bool): If columns whose values are all empty should be dropped

    Returns:
        DataFrame: cleaned DataFrame

    """
    if fill_nan:
        df = df.fillna(value=np.nan)
    if drop_empty_columns:
        df = df.dropna(axis=1, how='all')
    return df.sort_index()

def is_defined(self, objtxt, force_import=False):
        """Return True if object is defined"""
        return self.interpreter.is_defined(objtxt, force_import)

def remove_from_lib(self, name):
        """ Remove an object from the bin folder. """
        self.__remove_path(os.path.join(self.root_dir, "lib", name))

def last_modified_date(filename):
    """Last modified timestamp as a UTC datetime"""
    mtime = os.path.getmtime(filename)
    dt = datetime.datetime.utcfromtimestamp(mtime)
    return dt.replace(tzinfo=pytz.utc)

def strip_accents(s):
    """
    Strip accents to prepare for slugification.
    """
    nfkd = unicodedata.normalize('NFKD', unicode(s))
    return u''.join(ch for ch in nfkd if not unicodedata.combining(ch))

def dict_pop_or(d, key, default=None):
    """ Try popping a key from a dict.
        Instead of raising KeyError, just return the default value.
    """
    val = default
    with suppress(KeyError):
        val = d.pop(key)
    return val

def clean(self, text):
        """Remove all unwanted characters from text."""
        return ''.join([c for c in text if c in self.alphabet])

def filter_(stream_spec, filter_name, *args, **kwargs):
    """Alternate name for ``filter``, so as to not collide with the
    built-in python ``filter`` operator.
    """
    return filter(stream_spec, filter_name, *args, **kwargs)

def strip_spaces(x):
    """
    Strips spaces
    :param x:
    :return:
    """
    x = x.replace(b' ', b'')
    x = x.replace(b'\t', b'')
    return x

def glr_path_static():
    """Returns path to packaged static files"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '_static'))

def preprocess_french(trans, fr_nlp, remove_brackets_content=True):
    """ Takes a list of sentences in french and preprocesses them."""

    if remove_brackets_content:
        trans = pangloss.remove_content_in_brackets(trans, "[]")
    # Not sure why I have to split and rejoin, but that fixes a Spacy token
    # error.
    trans = fr_nlp(" ".join(trans.split()[:]))
    #trans = fr_nlp(trans)
    trans = " ".join([token.lower_ for token in trans if not token.is_punct])

    return trans

def ungzip_data(input_data):
    """Return a string of data after gzip decoding

    :param the input gziped data
    :return  the gzip decoded data"""
    buf = StringIO(input_data)
    f = gzip.GzipFile(fileobj=buf)
    return f

def _breakRemNewlines(tag):
	"""non-recursively break spaces and remove newlines in the tag"""
	for i,c in enumerate(tag.contents):
		if type(c) != bs4.element.NavigableString:
			continue
		c.replace_with(re.sub(r' {2,}', ' ', c).replace('\n',''))

def _visual_width(line):
    """Get the the number of columns required to display a string"""

    return len(re.sub(colorama.ansitowin32.AnsiToWin32.ANSI_CSI_RE, "", line))

def rm_keys_from_dict(d, keys):
    """
    Given a dictionary and a key list, remove any data in the dictionary with the given keys.

    :param dict d: Metadata
    :param list keys: Keys to be removed
    :return dict d: Metadata
    """
    # Loop for each key given
    for key in keys:
        # Is the key in the dictionary?
        if key in d:
            try:
                d.pop(key, None)
            except KeyError:
                # Not concerned with an error. Keep going.
                pass
    return d

def remove_series(self, series):
        """Removes a :py:class:`.Series` from the chart.

        :param Series series: The :py:class:`.Series` to remove.
        :raises ValueError: if you try to remove the last\
        :py:class:`.Series`."""

        if len(self.all_series()) == 1:
            raise ValueError("Cannot remove last series from %s" % str(self))
        self._all_series.remove(series)
        series._chart = None

def __init__(self, filename, mode, encoding=None):
        """Use the specified filename for streamed logging."""
        FileHandler.__init__(self, filename, mode, encoding)
        self.mode = mode
        self.encoding = encoding

def wget(url):
    """
    Download the page into a string
    """
    import urllib.parse
    request = urllib.request.urlopen(url)
    filestring = request.read()
    return filestring

def normalize_value(text):
    """
    This removes newlines and multiple spaces from a string.
    """
    result = text.replace('\n', ' ')
    result = re.subn('[ ]{2,}', ' ', result)[0]
    return result

def issuperset(self, other):
        """Report whether this RangeSet contains another set."""
        self._binary_sanity_check(other)
        return set.issuperset(self, other)

def sanitize_word(s):
    """Remove non-alphanumerical characters from metric word.
    And trim excessive underscores.
    """
    s = re.sub('[^\w-]+', '_', s)
    s = re.sub('__+', '_', s)
    return s.strip('_')

def to_bytes(value):
    """ str to bytes (py3k) """
    vtype = type(value)

    if vtype == bytes or vtype == type(None):
        return value

    try:
        return vtype.encode(value)
    except UnicodeEncodeError:
        pass
    return value

def slugify(s, delimiter='-'):
    """
    Normalize `s` into ASCII and replace non-word characters with `delimiter`.
    """
    s = unicodedata.normalize('NFKD', to_unicode(s)).encode('ascii', 'ignore').decode('ascii')
    return RE_SLUG.sub(delimiter, s).strip(delimiter).lower()

def _create_statusicon(self):
        """Return a new Gtk.StatusIcon."""
        statusicon = Gtk.StatusIcon()
        statusicon.set_from_gicon(self._icons.get_gicon('media'))
        statusicon.set_tooltip_text(_("udiskie"))
        return statusicon

def remove_list_duplicates(lista, unique=False):
    """
    Remove duplicated elements in a list.
    Args:
        lista: List with elements to clean duplicates.
    """
    result = []
    allready = []

    for elem in lista:
        if elem not in result:
            result.append(elem)
        else:
            allready.append(elem)

    if unique:
        for elem in allready:
            result = list(filter((elem).__ne__, result))

    return result

def __grid_widgets(self):
        """Places all the child widgets in the appropriate positions."""
        scrollbar_column = 0 if self.__compound is tk.LEFT else 2
        self._canvas.grid(row=0, column=1, sticky="nswe")
        self._scrollbar.grid(row=0, column=scrollbar_column, sticky="ns")

def readCommaList(fileList):
    """ Return a list of the files with the commas removed. """
    names=fileList.split(',')
    fileList=[]
    for item in names:
        fileList.append(item)
    return fileList

def datetime_local_to_utc(local):
    """
    Simple function to convert naive :std:`datetime.datetime` object containing
    local time to a naive :std:`datetime.datetime` object with UTC time.
    """
    timestamp = time.mktime(local.timetuple())
    return datetime.datetime.utcfromtimestamp(timestamp)

def _update_index_on_df(df, index_names):
    """Helper function to restore index information after collection. Doesn't
    use self so we can serialize this."""
    if index_names:
        df = df.set_index(index_names)
        # Remove names from unnamed indexes
        index_names = _denormalize_index_names(index_names)
        df.index.names = index_names
    return df

def replace_all(filepath, searchExp, replaceExp):
    """
    Replace all the ocurrences (in a file) of a string with another value.
    """
    for line in fileinput.input(filepath, inplace=1):
        if searchExp in line:
            line = line.replace(searchExp, replaceExp)
        sys.stdout.write(line)

def sort_matrix(a,n=0):
    """
    This will rearrange the array a[n] from lowest to highest, and
    rearrange the rest of a[i]'s in the same way. It is dumb and slow.

    Returns a numpy array.
    """
    a = _n.array(a)
    return a[:,a[n,:].argsort()]

def write_property(fh, key, value):
  """
    Write a single property to the file in Java properties format.

    :param fh: a writable file-like object
    :param key: the key to write
    :param value: the value to write
  """
  if key is COMMENT:
    write_comment(fh, value)
    return

  _require_string(key, 'keys')
  _require_string(value, 'values')

  fh.write(_escape_key(key))
  fh.write(b'=')
  fh.write(_escape_value(value))
  fh.write(b'\n')

def substitute(dict_, source):
    """ Perform re.sub with the patterns in the given dict
    Args:
      dict_: {pattern: repl}
      source: str
    """
    d_esc = (re.escape(k) for k in dict_.keys())
    pattern = re.compile('|'.join(d_esc))
    return pattern.sub(lambda x: dict_[x.group()], source)

def get_column(self, X, column):
        """Return a column of the given matrix.

        Args:
            X: `numpy.ndarray` or `pandas.DataFrame`.
            column: `int` or `str`.

        Returns:
            np.ndarray: Selected column.
        """
        if isinstance(X, pd.DataFrame):
            return X[column].values

        return X[:, column]

def subn_filter(s, find, replace, count=0):
    """A non-optimal implementation of a regex filter"""
    return re.gsub(find, replace, count, s)

def populate_obj(obj, attrs):
    """Populates an object's attributes using the provided dict
    """
    for k, v in attrs.iteritems():
        setattr(obj, k, v)

def _trim(image):
    """Trim a PIL image and remove white space."""
    background = PIL.Image.new(image.mode, image.size, image.getpixel((0, 0)))
    diff = PIL.ImageChops.difference(image, background)
    diff = PIL.ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        image = image.crop(bbox)
    return image

def mock_add_spec(self, spec, spec_set=False):
        """Add a spec to a mock. `spec` can either be an object or a
        list of strings. Only attributes on the `spec` can be fetched as
        attributes from the mock.

        If `spec_set` is True then only attributes on the spec can be set."""
        self._mock_add_spec(spec, spec_set)
        self._mock_set_magics()

def add_params_to_url(url, params):
    """Adds params to url

    :param url: Url
    :param params: Params to add
    :return: original url with new params
    """
    url_parts = list(urlparse.urlparse(url))  # get url parts
    query = dict(urlparse.parse_qsl(url_parts[4]))  # get url query
    query.update(params)  # add new params
    url_parts[4] = urlencode(query)
    return urlparse.urlunparse(url_parts)

def replaceNewlines(string, newlineChar):
	"""There's probably a way to do this with string functions but I was lazy.
		Replace all instances of \r or \n in a string with something else."""
	if newlineChar in string:
		segments = string.split(newlineChar)
		string = ""
		for segment in segments:
			string += segment
	return string

def uniform_noise(points):
    """Init a uniform noise variable."""
    return np.random.rand(1) * np.random.uniform(points, 1) \
        + random.sample([2, -2], 1)

def replaceNewlines(string, newlineChar):
	"""There's probably a way to do this with string functions but I was lazy.
		Replace all instances of \r or \n in a string with something else."""
	if newlineChar in string:
		segments = string.split(newlineChar)
		string = ""
		for segment in segments:
			string += segment
	return string

def alter_change_column(self, table, column, field):
        """Support change columns."""
        return self._update_column(table, column, lambda a, b: b)

def create_symlink(source, link_name):
    """
    Creates symbolic link for either operating system.

    http://stackoverflow.com/questions/6260149/os-symlink-support-in-windows
    """
    os_symlink = getattr(os, "symlink", None)
    if isinstance(os_symlink, collections.Callable):
        os_symlink(source, link_name)
    else:
        import ctypes
        csl = ctypes.windll.kernel32.CreateSymbolicLinkW
        csl.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
        csl.restype = ctypes.c_ubyte
        flags = 1 if os.path.isdir(source) else 0
        if csl(link_name, source, flags) == 0:
            raise ctypes.WinError()

def __repr__(self):
    """Returns a stringified representation of this object."""
    return str({'name': self._name, 'watts': self._watts,
                'type': self._output_type, 'id': self._integration_id})

def make_aware(value, timezone):
    """
    Makes a naive datetime.datetime in a given time zone aware.
    """
    if hasattr(timezone, 'localize') and value not in (datetime.datetime.min, datetime.datetime.max):
        # available for pytz time zones
        return timezone.localize(value, is_dst=None)
    else:
        # may be wrong around DST changes
        return value.replace(tzinfo=timezone)

def _is_root():
    """Checks if the user is rooted."""
    import os
    import ctypes
    try:
        return os.geteuid() == 0
    except AttributeError:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    return False

def process_docstring(app, what, name, obj, options, lines):
    """React to a docstring event and append contracts to it."""
    # pylint: disable=unused-argument
    # pylint: disable=too-many-arguments
    lines.extend(_format_contracts(what=what, obj=obj))

def reset_password(app, appbuilder, username, password):
    """
        Resets a user's password
    """
    _appbuilder = import_application(app, appbuilder)
    user = _appbuilder.sm.find_user(username=username)
    if not user:
        click.echo("User {0} not found.".format(username))
    else:
        _appbuilder.sm.reset_password(user.id, password)
        click.echo(click.style("User {0} reseted.".format(username), fg="green"))

def check(text):
    """Check the text."""
    err = "malapropisms.misc"
    msg = u"'{}' is a malapropism."

    illogics = [
        "the infinitesimal universe",
        "a serial experience",
        "attack my voracity",
    ]

    return existence_check(text, illogics, err, msg, offset=1)

def format_doc_text(text):
    """
    A very thin wrapper around textwrap.fill to consistently wrap documentation text
    for display in a command line environment. The text is wrapped to 99 characters with an
    indentation depth of 4 spaces. Each line is wrapped independently in order to preserve
    manually added line breaks.

    :param text: The text to format, it is cleaned by inspect.cleandoc.
    :return: The formatted doc text.
    """

    return '\n'.join(
        textwrap.fill(line, width=99, initial_indent='    ', subsequent_indent='    ')
        for line in inspect.cleandoc(text).splitlines())

def extract_all(zipfile, dest_folder):
    """
    reads the zip file, determines compression
    and unzips recursively until source files 
    are extracted 
    """
    z = ZipFile(zipfile)
    print(z)
    z.extract(dest_folder)

def get_decimal_quantum(precision):
    """Return minimal quantum of a number, as defined by precision."""
    assert isinstance(precision, (int, decimal.Decimal))
    return decimal.Decimal(10) ** (-precision)

def angle(x0, y0, x1, y1):
    """ Returns the angle between two points.
    """
    return degrees(atan2(y1-y0, x1-x0))

def getvariable(name):
    """Get the value of a local variable somewhere in the call stack."""
    import inspect
    fr = inspect.currentframe()
    try:
        while fr:
            fr = fr.f_back
            vars = fr.f_locals
            if name in vars:
                return vars[name]
    except:
        pass
    return None

def apply(filter):
    """Manufacture decorator that filters return value with given function.

    ``filter``:
      Callable that takes a single parameter.
    """
    def decorator(callable):
        return lambda *args, **kwargs: filter(callable(*args, **kwargs))
    return decorator

def create_response(self, request, content, content_type):
        """Returns a response object for the request. Can be overridden to return different responses."""

        return HttpResponse(content=content, content_type=content_type)

def latlng(arg):
    """Converts a lat/lon pair to a comma-separated string.

    For example:

    sydney = {
        "lat" : -33.8674869,
        "lng" : 151.2069902
    }

    convert.latlng(sydney)
    # '-33.8674869,151.2069902'

    For convenience, also accepts lat/lon pair as a string, in
    which case it's returned unchanged.

    :param arg: The lat/lon pair.
    :type arg: string or dict or list or tuple
    """
    if is_string(arg):
        return arg

    normalized = normalize_lat_lng(arg)
    return "%s,%s" % (format_float(normalized[0]), format_float(normalized[1]))

def mostCommonItem(lst):
    """Choose the most common item from the list, or the first item if all
    items are unique."""
    # This elegant solution from: http://stackoverflow.com/a/1518632/1760218
    lst = [l for l in lst if l]
    if lst:
        return max(set(lst), key=lst.count)
    else:
        return None

def extend(self, iterable):
        """Extend the list by appending all the items in the given list."""
        return super(Collection, self).extend(
            self._ensure_iterable_is_valid(iterable))

def path_to_list(pathstr):
    """Conver a path string to a list of path elements."""
    return [elem for elem in pathstr.split(os.path.pathsep) if elem]

def allclose(a, b):
    """
    Test that a and b are close and match in shape.

    Parameters
    ----------
    a : ndarray
        First array to check

    b : ndarray
        First array to check
    """
    from numpy import allclose
    return (a.shape == b.shape) and allclose(a, b)

def counter(items):
    """
    Simplest required implementation of collections.Counter. Required as 2.6
    does not have Counter in collections.
    """
    results = {}
    for item in items:
        results[item] = results.get(item, 0) + 1
    return results

def set_subparsers_args(self, *args, **kwargs):
        """
        Sets args and kwargs that are passed when creating a subparsers group
        in an argparse.ArgumentParser i.e. when calling
        argparser.ArgumentParser.add_subparsers
        """
        self.subparsers_args = args
        self.subparsers_kwargs = kwargs

def __reversed__(self):
        """
        Return a reversed iterable over the items in the dictionary. Items are
        iterated over in their reverse sort order.

        Iterating views while adding or deleting entries in the dictionary may
        raise a RuntimeError or fail to iterate over all entries.
        """
        _dict = self._dict
        return iter((key, _dict[key]) for key in reversed(self._list))

def add_option(self, *args, **kwargs):
        """Add optparse or argparse option depending on CmdHelper initialization."""
        if self.parseTool == 'argparse':
            if args and args[0] == '':   # no short option
                args = args[1:]
            return self.parser.add_argument(*args, **kwargs)
        else:
            return self.parser.add_option(*args, **kwargs)

def delete(filething):
    """ delete(filething)

    Arguments:
        filething (filething)
    Raises:
        mutagen.MutagenError

    Remove tags from a file.
    """

    t = MP4(filething)
    filething.fileobj.seek(0)
    t.delete(filething)

def __init__(self, name, flag, **kwargs):
    """
    Argument class constructor, should be used inside a class that inherits the BaseAction class.

    :param name(str): the optional argument name to be used with two slahes (--cmd)
    :param flag(str): a short flag for the argument (-c)
    :param \*\*kwargs: all keywords arguments supported for argparse actions.
    """
    self.name = name
    self.flag = flag
    self.options = kwargs

def get_system_root_directory():
    """
    Get system root directory (application installed root directory)

    Returns
    -------
    string
        A full path

    """
    root = os.path.dirname(__file__)
    root = os.path.dirname(root)
    root = os.path.abspath(root)
    return root

def intround(value):
    """Given a float returns a rounded int. Should give the same result on
    both Py2/3
    """

    return int(decimal.Decimal.from_float(
        value).to_integral_value(decimal.ROUND_HALF_EVEN))

def be_array_from_bytes(fmt, data):
    """
    Reads an array from bytestring with big-endian data.
    """
    arr = array.array(str(fmt), data)
    return fix_byteorder(arr)

def c_array(ctype, values):
    """Convert a python string to c array."""
    if isinstance(values, np.ndarray) and values.dtype.itemsize == ctypes.sizeof(ctype):
        return (ctype * len(values)).from_buffer_copy(values)
    return (ctype * len(values))(*values)

def round_to_float(number, precision):
    """Round a float to a precision"""
    rounded = Decimal(str(floor((number + precision / 2) // precision))
                      ) * Decimal(str(precision))
    return float(rounded)

def access_ok(self, access):
        """ Check if there is enough permissions for access """
        for c in access:
            if c not in self.perms:
                return False
        return True

def rstjinja(app, docname, source):
    """
    Render our pages as a jinja template for fancy templating goodness.
    """
    # Make sure we're outputting HTML
    if app.builder.format != 'html':
        return
    src = source[0]
    rendered = app.builder.templates.render_string(
        src, app.config.html_context
    )
    source[0] = rendered

def install_postgres(user=None, dbname=None, password=None):
    """Install Postgres on remote"""
    execute(pydiploy.django.install_postgres_server,
            user=user, dbname=dbname, password=password)

async def wait_and_quit(loop):
	"""Wait until all task are executed."""
	from pylp.lib.tasks import running
	if running:
		await asyncio.wait(map(lambda runner: runner.future, running))

async def list(source):
    """Generate a single list from an asynchronous sequence."""
    result = []
    async with streamcontext(source) as streamer:
        async for item in streamer:
            result.append(item)
    yield result

def StringIO(*args, **kwargs):
    """StringIO constructor shim for the async wrapper."""
    raw = sync_io.StringIO(*args, **kwargs)
    return AsyncStringIOWrapper(raw)

def save_pdf(path):
  """
  Saves a pdf of the current matplotlib figure.

  :param path: str, filepath to save to
  """

  pp = PdfPages(path)
  pp.savefig(pyplot.gcf())
  pp.close()

def creation_time(self):
    """dfdatetime.Filetime: creation time or None if not set."""
    timestamp = self._fsntfs_attribute.get_creation_time_as_integer()
    return dfdatetime_filetime.Filetime(timestamp=timestamp)

def download_file(save_path, file_url):
    """ Download file from http url link """

    r = requests.get(file_url)  # create HTTP response object

    with open(save_path, 'wb') as f:
        f.write(r.content)

    return save_path

def nmse(a, b):
    """Returns the normalized mean square error of a and b
    """
    return np.square(a - b).mean() / (a.mean() * b.mean())

def save_pdf(path):
  """
  Saves a pdf of the current matplotlib figure.

  :param path: str, filepath to save to
  """

  pp = PdfPages(path)
  pp.savefig(pyplot.gcf())
  pp.close()

def safe_rmtree(directory):
  """Delete a directory if it's present. If it's not present, no-op."""
  if os.path.exists(directory):
    shutil.rmtree(directory, True)

def extract_vars_above(*names):
    """Extract a set of variables by name from another frame.

    Similar to extractVars(), but with a specified depth of 1, so that names
    are exctracted exactly from above the caller.

    This is simply a convenience function so that the very common case (for us)
    of skipping exactly 1 frame doesn't have to construct a special dict for
    keyword passing."""

    callerNS = sys._getframe(2).f_locals
    return dict((k,callerNS[k]) for k in names)

def __init__(self, name, contained_key):
        """Instantiate an anonymous file-based Bucket around a single key.
        """
        self.name = name
        self.contained_key = contained_key

def is_nullable_list(val, vtype):
    """Return True if list contains either values of type `vtype` or None."""
    return (isinstance(val, list) and
            any(isinstance(v, vtype) for v in val) and
            all((isinstance(v, vtype) or v is None) for v in val))

def compute_boxplot(self, series):
        """
        Compute boxplot for given pandas Series.
        """
        from matplotlib.cbook import boxplot_stats
        series = series[series.notnull()]
        if len(series.values) == 0:
            return {}
        elif not is_numeric_dtype(series):
            return self.non_numeric_stats(series)
        stats = boxplot_stats(list(series.values))[0]
        stats['count'] = len(series.values)
        stats['fliers'] = "|".join(map(str, stats['fliers']))
        return stats

def Any(a, axis, keep_dims):
    """
    Any reduction op.
    """
    return np.any(a, axis=axis if not isinstance(axis, np.ndarray) else tuple(axis),
                  keepdims=keep_dims),

def compute_boxplot(self, series):
        """
        Compute boxplot for given pandas Series.
        """
        from matplotlib.cbook import boxplot_stats
        series = series[series.notnull()]
        if len(series.values) == 0:
            return {}
        elif not is_numeric_dtype(series):
            return self.non_numeric_stats(series)
        stats = boxplot_stats(list(series.values))[0]
        stats['count'] = len(series.values)
        stats['fliers'] = "|".join(map(str, stats['fliers']))
        return stats

def selectin(table, field, value, complement=False):
    """Select rows where the given field is a member of the given value."""

    return select(table, field, lambda v: v in value,
                  complement=complement)

def compute_boxplot(self, series):
        """
        Compute boxplot for given pandas Series.
        """
        from matplotlib.cbook import boxplot_stats
        series = series[series.notnull()]
        if len(series.values) == 0:
            return {}
        elif not is_numeric_dtype(series):
            return self.non_numeric_stats(series)
        stats = boxplot_stats(list(series.values))[0]
        stats['count'] = len(series.values)
        stats['fliers'] = "|".join(map(str, stats['fliers']))
        return stats

def uncheck(self, locator=None, allow_label_click=None, **kwargs):
        """
        Find a check box and uncheck it. The check box can be found via name, id, or label text. ::

            page.uncheck("German")

        Args:
            locator (str, optional): Which check box to uncheck.
            allow_label_click (bool, optional): Attempt to click the label to toggle state if
                element is non-visible. Defaults to :data:`capybara.automatic_label_click`.
            **kwargs: Arbitrary keyword arguments for :class:`SelectorQuery`.
        """

        self._check_with_label(
            "checkbox", False, locator=locator, allow_label_click=allow_label_click, **kwargs)

def upload_as_json(name, mylist):
    """
    Upload the IPList as json payload. 

    :param str name: name of IPList
    :param list: list of IPList entries
    :return: None
    """
    location = list(IPList.objects.filter(name))
    if location:
        iplist = location[0]
        return iplist.upload(json=mylist, as_type='json')

def comma_delimited_to_list(list_param):
    """Convert comma-delimited list / string into a list of strings

    :param list_param: Comma-delimited string
    :type list_param: str | unicode
    :return: A list of strings
    :rtype: list
    """
    if isinstance(list_param, list):
        return list_param
    if isinstance(list_param, str):
        return list_param.split(',')
    else:
        return []

def __getattr__(self, *args, **kwargs):
        """
        Magic method dispatcher
        """

        return xmlrpc.client._Method(self.__request, *args, **kwargs)

def getbyteslice(self, start, end):
        """Direct access to byte data."""
        c = self._rawarray[start:end]
        return c

def get_serializable_data_for_fields(model):
    """
    Return a serialised version of the model's fields which exist as local database
    columns (i.e. excluding m2m and incoming foreign key relations)
    """
    pk_field = model._meta.pk
    # If model is a child via multitable inheritance, use parent's pk
    while pk_field.remote_field and pk_field.remote_field.parent_link:
        pk_field = pk_field.remote_field.model._meta.pk

    obj = {'pk': get_field_value(pk_field, model)}

    for field in model._meta.fields:
        if field.serialize:
            obj[field.name] = get_field_value(field, model)

    return obj

def __init__(self, ba=None):
        """Constructor."""
        self.bytearray = ba or (bytearray(b'\0') * self.SIZEOF)

def calc_base64(s):
    """Return base64 encoded binarystring."""
    s = compat.to_bytes(s)
    s = compat.base64_encodebytes(s).strip()  # return bytestring
    return compat.to_native(s)

def plot_target(target, ax):
    """Ajoute la target au plot"""
    ax.scatter(target[0], target[1], target[2], c="red", s=80)

def prsint(string):
    """
    Parse a string as an integer, encapsulating error handling.

    http://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/prsint_c.html

    :param string: String representing an integer.
    :type string: str
    :return: Integer value obtained by parsing string.
    :rtype: int
    """
    string = stypes.stringToCharP(string)
    intval = ctypes.c_int()
    libspice.prsint_c(string, ctypes.byref(intval))
    return intval.value

def set_default(self, key, value):
        """Set the default value for this key.
        Default only used when no value is provided by the user via
        arg, config or env.
        """
        k = self._real_key(key.lower())
        self._defaults[k] = value

def struct2dict(struct):
    """convert a ctypes structure to a dictionary"""
    return {x: getattr(struct, x) for x in dict(struct._fields_).keys()}

def set_logxticks_for_all(self, row_column_list=None, logticks=None):
        """Manually specify the x-axis log tick values.

        :param row_column_list: a list containing (row, column) tuples to
            specify the subplots, or None to indicate *all* subplots.
        :type row_column_list: list or None
        :param logticks: logarithm of the locations for the ticks along the
            axis.

        For example, if you specify [1, 2, 3], ticks will be placed at 10,
        100 and 1000.

        """
        if row_column_list is None:
            self.ticks['x'] = ['1e%d' % u for u in logticks]
        else:
            for row, column in row_column_list:
                self.set_logxticks(row, column, logticks)

def testable_memoized_property(func=None, key_factory=per_instance, **kwargs):
  """A variant of `memoized_property` that allows for setting of properties (for tests, etc)."""
  getter = memoized_method(func=func, key_factory=key_factory, **kwargs)

  def setter(self, val):
    with getter.put(self) as putter:
      putter(val)

  return property(fget=getter,
                  fset=setter,
                  fdel=lambda self: getter.forget(self))

def is_cached(self, url):
        """Checks if specified URL is cached."""
        try:
            return True if url in self.cache else False
        except TypeError:
            return False

def set_xlimits_widgets(self, set_min=True, set_max=True):
        """Populate axis limits GUI with current plot values."""
        xmin, xmax = self.tab_plot.ax.get_xlim()
        if set_min:
            self.w.x_lo.set_text('{0}'.format(xmin))
        if set_max:
            self.w.x_hi.set_text('{0}'.format(xmax))

def ttl(self):
        """how long you should cache results for cacheable queries"""
        ret = 3600
        cn = self.get_process()
        if "ttl" in cn:
            ret = cn["ttl"]
        return ret

def set_xlimits(self, min=None, max=None):
        """Set limits for the x-axis.

        :param min: minimum value to be displayed.  If None, it will be
            calculated.
        :param max: maximum value to be displayed.  If None, it will be
            calculated.

        """
        self.limits['xmin'] = min
        self.limits['xmax'] = max

def manhattan_distance_numpy(object1, object2):
    """!
    @brief Calculate Manhattan distance between two objects using numpy.

    @param[in] object1 (array_like): The first array_like object.
    @param[in] object2 (array_like): The second array_like object.

    @return (double) Manhattan distance between two objects.

    """
    return numpy.sum(numpy.absolute(object1 - object2), axis=1).T

def add_xlabel(self, text=None):
        """
        Add a label to the x-axis.
        """
        x = self.fit.meta['independent']
        if not text:
            text = '$' + x['tex_symbol'] + r'$ $(\si{' + x['siunitx'] +  r'})$'
        self.plt.set_xlabel(text)

def get_model(name):
    """
    Convert a model's verbose name to the model class. This allows us to
    use the models verbose name in steps.
    """

    model = MODELS.get(name.lower(), None)

    assert model, "Could not locate model by name '%s'" % name

    return model

def setDictDefaults (d, defaults):
  """Sets all defaults for the given dictionary to those contained in a
  second defaults dictionary.  This convenience method calls:

    d.setdefault(key, value)

  for each key and value in the given defaults dictionary.
  """
  for key, val in defaults.items():
    d.setdefault(key, val)

  return d

def test():  # pragma: no cover
    """Execute the unit tests on an installed copy of unyt.

    Note that this function requires pytest to run. If pytest is not
    installed this function will raise ImportError.
    """
    import pytest
    import os

    pytest.main([os.path.dirname(os.path.abspath(__file__))])

def stft(func=None, **kwparams):
  """
  Short Time Fourier Transform for complex data.

  Same to the default STFT strategy, but with new defaults. This is the same
  to:

  .. code-block:: python

    stft.base(transform=numpy.fft.fft, inverse_transform=numpy.fft.ifft)

  See ``stft.base`` docs for more.
  """
  from numpy.fft import fft, ifft
  return stft.base(transform=fft, inverse_transform=ifft)(func, **kwparams)

def _fill(self):
    """Advance the iterator without returning the old head."""
    try:
      self._head = self._iterable.next()
    except StopIteration:
      self._head = None

def prettyprint(d):
        """Print dicttree in Json-like format. keys are sorted
        """
        print(json.dumps(d, sort_keys=True, 
                         indent=4, separators=("," , ": ")))

def sdmethod(meth):
    """
    This is a hack to monkey patch sdproperty to work as expected with instance methods.
    """
    sd = singledispatch(meth)

    def wrapper(obj, *args, **kwargs):
        return sd.dispatch(args[0].__class__)(obj, *args, **kwargs)

    wrapper.register = sd.register
    wrapper.dispatch = sd.dispatch
    wrapper.registry = sd.registry
    wrapper._clear_cache = sd._clear_cache
    functools.update_wrapper(wrapper, meth)
    return wrapper

def _match_literal(self, a, b=None):
        """Match two names."""

        return a.lower() == b if not self.case_sensitive else a == b

def to_capitalized_camel_case(snake_case_string):
    """
    Convert a string from snake case to camel case with the first letter capitalized. For example, "some_var"
    would become "SomeVar".

    :param snake_case_string: Snake-cased string to convert to camel case.
    :returns: Camel-cased version of snake_case_string.
    """
    parts = snake_case_string.split('_')
    return ''.join([i.title() for i in parts])

def query(self, base, filterstr, attrlist=None):
		""" wrapper to search_s """
		return self.conn.search_s(base, ldap.SCOPE_SUBTREE, filterstr, attrlist)

def angle_to_cartesian(lon, lat):
    """Convert spherical coordinates to cartesian unit vectors."""
    theta = np.array(np.pi / 2. - lat)
    return np.vstack((np.sin(theta) * np.cos(lon),
                      np.sin(theta) * np.sin(lon),
                      np.cos(theta))).T

def shutdown(self):
        """close socket, immediately."""
        if self.sock:
            self.sock.close()
            self.sock = None
            self.connected = False

def bin_to_int(string):
    """Convert a one element byte string to signed int for python 2 support."""
    if isinstance(string, str):
        return struct.unpack("b", string)[0]
    else:
        return struct.unpack("b", bytes([string]))[0]

def advance_one_line(self):
    """Advances to next line."""

    current_line = self._current_token.line_number
    while current_line == self._current_token.line_number:
      self._current_token = ConfigParser.Token(*next(self._token_generator))

def FromString(self, string):
    """Parse a bool from a string."""
    if string.lower() in ("false", "no", "n"):
      return False

    if string.lower() in ("true", "yes", "y"):
      return True

    raise TypeValueError("%s is not recognized as a boolean value." % string)

def solve(A, x):
    """Solves a linear equation system with a matrix of shape (n, n) and an
    array of shape (n, ...). The output has the same shape as the second
    argument.
    """
    # https://stackoverflow.com/a/48387507/353337
    x = numpy.asarray(x)
    return numpy.linalg.solve(A, x.reshape(x.shape[0], -1)).reshape(x.shape)

def sort_data(x, y):
    """Sort the data."""
    xy = sorted(zip(x, y))
    x, y = zip(*xy)
    return x, y

def str_to_boolean(input_str):
    """ a conversion function for boolean
    """
    if not isinstance(input_str, six.string_types):
        raise ValueError(input_str)
    input_str = str_quote_stripper(input_str)
    return input_str.lower() in ("true", "t", "1", "y", "yes")

def _index_ordering(redshift_list):
        """

        :param redshift_list: list of redshifts
        :return: indexes in acending order to be evaluated (from z=0 to z=z_source)
        """
        redshift_list = np.array(redshift_list)
        sort_index = np.argsort(redshift_list)
        return sort_index

def schedule_task(self):
        """
        Schedules this publish action as a Celery task.
        """
        from .tasks import publish_task

        publish_task.apply_async(kwargs={'pk': self.pk}, eta=self.scheduled_time)

def get_order(self, codes):
        """Return evidence codes in order shown in code2name."""
        return sorted(codes, key=lambda e: [self.ev2idx.get(e)])

def schedule_task(self):
        """
        Schedules this publish action as a Celery task.
        """
        from .tasks import publish_task

        publish_task.apply_async(kwargs={'pk': self.pk}, eta=self.scheduled_time)

def sort_data(x, y):
    """Sort the data."""
    xy = sorted(zip(x, y))
    x, y = zip(*xy)
    return x, y

def force_stop(self):
        """
        Forcibly terminates all Celery processes.
        """
        r = self.local_renderer
        with self.settings(warn_only=True):
            r.sudo('pkill -9 -f celery')
        r.sudo('rm -f /tmp/celery*.pid')

def has_add_permission(self, request):
        """ Can add this object """
        return request.user.is_authenticated and request.user.is_active and request.user.is_staff

def pointer(self):
        """Get a ctypes void pointer to the memory mapped region.

        :type: ctypes.c_void_p
        """
        return ctypes.cast(ctypes.pointer(ctypes.c_uint8.from_buffer(self.mapping, 0)), ctypes.c_void_p)

def do_forceescape(value):
    """Enforce HTML escaping.  This will probably double escape variables."""
    if hasattr(value, '__html__'):
        value = value.__html__()
    return escape(unicode(value))

def fieldstorage(self):
        """ `cgi.FieldStorage` from `wsgi.input`.
        """
        if self._fieldstorage is None:
            if self._body is not None:
                raise ReadBodyTwiceError()

            self._fieldstorage = cgi.FieldStorage(
                environ=self._environ,
                fp=self._environ['wsgi.input']
            )

        return self._fieldstorage

def _histplot_op(ax, data, **kwargs):
    """Add a histogram for the data to the axes."""
    bins = get_bins(data)
    ax.hist(data, bins=bins, align="left", density=True, **kwargs)
    return ax

def stringc(text, color):
    """
    Return a string with terminal colors.
    """
    if has_colors:
        text = str(text)

        return "\033["+codeCodes[color]+"m"+text+"\033[0m"
    else:
        return text

def napoleon_to_sphinx(docstring, **config_params):
    """
    Convert napoleon docstring to plain sphinx string.

    Args:
        docstring (str): Docstring in napoleon format.
        **config_params (dict): Whatever napoleon doc configuration you want.

    Returns:
        str: Sphinx string.
    """
    if "napoleon_use_param" not in config_params:
        config_params["napoleon_use_param"] = False

    if "napoleon_use_rtype" not in config_params:
        config_params["napoleon_use_rtype"] = False

    config = Config(**config_params)

    return str(GoogleDocstring(docstring, config))

def _possibly_convert_objects(values):
    """Convert arrays of datetime.datetime and datetime.timedelta objects into
    datetime64 and timedelta64, according to the pandas convention.
    """
    return np.asarray(pd.Series(values.ravel())).reshape(values.shape)

def _split(string, splitters):
    """Splits a string into parts at multiple characters"""
    part = ''
    for character in string:
        if character in splitters:
            yield part
            part = ''
        else:
            part += character
    yield part

def _to_corrected_pandas_type(dt):
    """
    When converting Spark SQL records to Pandas DataFrame, the inferred data type may be wrong.
    This method gets the corrected data type for Pandas if that type may be inferred uncorrectly.
    """
    import numpy as np
    if type(dt) == ByteType:
        return np.int8
    elif type(dt) == ShortType:
        return np.int16
    elif type(dt) == IntegerType:
        return np.int32
    elif type(dt) == FloatType:
        return np.float32
    else:
        return None

def split_string(text, chars_per_string):
    """
    Splits one string into multiple strings, with a maximum amount of `chars_per_string` characters per string.
    This is very useful for splitting one giant message into multiples.

    :param text: The text to split
    :param chars_per_string: The number of characters per line the text is split into.
    :return: The splitted text as a list of strings.
    """
    return [text[i:i + chars_per_string] for i in range(0, len(text), chars_per_string)]

def set_time(filename, mod_time):
	"""
	Set the modified time of a file
	"""
	log.debug('Setting modified time to %s', mod_time)
	mtime = calendar.timegm(mod_time.utctimetuple())
	# utctimetuple discards microseconds, so restore it (for consistency)
	mtime += mod_time.microsecond / 1000000
	atime = os.stat(filename).st_atime
	os.utime(filename, (atime, mtime))

def split_into_words(s):
  """Split a sentence into list of words."""
  s = re.sub(r"\W+", " ", s)
  s = re.sub(r"[_0-9]+", " ", s)
  return s.split()

def gauss_pdf(x, mu, sigma):
    """Normalized Gaussian"""
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - mu) ** 2 / 2. / sigma ** 2)

def bulk_query(self, query, *multiparams):
        """Bulk insert or update."""

        with self.get_connection() as conn:
            conn.bulk_query(query, *multiparams)

def add_exec_permission_to(target_file):
    """Add executable permissions to the file

    :param target_file: the target file whose permission to be changed
    """
    mode = os.stat(target_file).st_mode
    os.chmod(target_file, mode | stat.S_IXUSR)

def createdb():
    """Create database tables from sqlalchemy models"""
    manager.db.engine.echo = True
    manager.db.create_all()
    set_alembic_revision()

def col_rename(df,col_name,new_col_name):
    """ Changes a column name in a DataFrame
    Parameters:
    df - DataFrame
        DataFrame to operate on
    col_name - string
        Name of column to change
    new_col_name - string
        New name of column
    """
    col_list = list(df.columns)
    for index,value in enumerate(col_list):
        if value == col_name:
            col_list[index] = new_col_name
            break
    df.columns = col_list

def get_db_version(session):
    """
    :param session: actually it is a sqlalchemy session
    :return: version number
    """
    value = session.query(ProgramInformation.value).filter(ProgramInformation.name == "db_version").scalar()
    return int(value)

def _fill(self):
    """Advance the iterator without returning the old head."""
    try:
      self._head = self._iterable.next()
    except StopIteration:
      self._head = None

def get_checkerboard_matrix(kernel_width):

    """
    example matrix for width = 2

    -1  -1    1   1
    -1  -1    1   1
     1   1   -1  -1
     1   1   -1  -1

    :param kernel_width:
    :return:
    """

    return np.vstack((
        np.hstack((
            -1 * np.ones((kernel_width, kernel_width)), np.ones((kernel_width, kernel_width))
        )),
        np.hstack((
            np.ones((kernel_width, kernel_width)), -1 * np.ones((kernel_width, kernel_width))
        ))
    ))

def cric__lasso():
    """ Lasso Regression
    """
    model = sklearn.linear_model.LogisticRegression(penalty="l1", C=0.002)

    # we want to explain the raw probability outputs of the trees
    model.predict = lambda X: model.predict_proba(X)[:,1]
    
    return model

def norm_slash(name):
    """Normalize path slashes."""

    if isinstance(name, str):
        return name.replace('/', "\\") if not is_case_sensitive() else name
    else:
        return name.replace(b'/', b"\\") if not is_case_sensitive() else name

def hex_escape(bin_str):
  """
  Hex encode a binary string
  """
  printable = string.ascii_letters + string.digits + string.punctuation + ' '
  return ''.join(ch if ch in printable else r'0x{0:02x}'.format(ord(ch)) for ch in bin_str)

def from_file_url(url):
    """ Convert from file:// url to file path
    """
    if url.startswith('file://'):
        url = url[len('file://'):].replace('/', os.path.sep)

    return url

def static_method(cls, f):
        """Decorator which dynamically binds static methods to the model for later use."""
        setattr(cls, f.__name__, staticmethod(f))
        return f

def _uniquify(_list):
    """Remove duplicates in a list."""
    seen = set()
    result = []
    for x in _list:
        if x not in seen:
            result.append(x)
            seen.add(x)
    return result

def _StopStatusUpdateThread(self):
    """Stops the status update thread."""
    self._status_update_active = False
    if self._status_update_thread.isAlive():
      self._status_update_thread.join()
    self._status_update_thread = None

def _check_key(self, key):
        """
        Ensures well-formedness of a key.
        """
        if not len(key) == 2:
            raise TypeError('invalid key: %r' % key)
        elif key[1] not in TYPES:
            raise TypeError('invalid datatype: %s' % key[1])

def _StopStatusUpdateThread(self):
    """Stops the status update thread."""
    self._status_update_active = False
    if self._status_update_thread.isAlive():
      self._status_update_thread.join()
    self._status_update_thread = None

def kill(self):
        """Kill the browser.

        This is useful when the browser is stuck.
        """
        if self.process:
            self.process.kill()
            self.process.wait()

def memory_used(self):
        """To know the allocated memory at function termination.

        ..versionadded:: 4.1

        This property might return None if the function is still running.

        This function should help to show memory leaks or ram greedy code.
        """
        if self._end_memory:
            memory_used = self._end_memory - self._start_memory
            return memory_used
        else:
            return None

def stop_button_click_handler(self):
        """Method to handle what to do when the stop button is pressed"""
        self.stop_button.setDisabled(True)
        # Interrupt computations or stop debugging
        if not self.shellwidget._reading:
            self.interrupt_kernel()
        else:
            self.shellwidget.write_to_stdin('exit')

def has_field(mc, field_name):
    """
    detect if a model has a given field has

    :param field_name:
    :param mc:
    :return:
    """
    try:
        mc._meta.get_field(field_name)
    except FieldDoesNotExist:
        return False
    return True

def _write_json(file, contents):
    """Write a dict to a JSON file."""
    with open(file, 'w') as f:
        return json.dump(contents, f, indent=2, sort_keys=True)

def is_date(thing):
    """Checks if the given thing represents a date

    :param thing: The object to check if it is a date
    :type thing: arbitrary object
    :returns: True if we have a date object
    :rtype: bool
    """
    # known date types
    date_types = (datetime.datetime,
                  datetime.date,
                  DateTime)
    return isinstance(thing, date_types)

def __str__(self):
        """Executes self.function to convert LazyString instance to a real
        str."""
        if not hasattr(self, '_str'):
            self._str=self.function(*self.args, **self.kwargs)
        return self._str

def is_datetime_like(dtype):
    """Check if a dtype is a subclass of the numpy datetime types
    """
    return (np.issubdtype(dtype, np.datetime64) or
            np.issubdtype(dtype, np.timedelta64))

def boolean(value):
    """
    Configuration-friendly boolean type converter.

    Supports both boolean-valued and string-valued inputs (e.g. from env vars).

    """
    if isinstance(value, bool):
        return value

    if value == "":
        return False

    return strtobool(value)

def all_strings(arr):
        """
        Ensures that the argument is a list that either is empty or contains only strings
        :param arr: list
        :return:
        """
        if not isinstance([], list):
            raise TypeError("non-list value found where list is expected")
        return all(isinstance(x, str) for x in arr)

def underscore(text):
    """Converts text that may be camelcased into an underscored format"""
    return UNDERSCORE[1].sub(r'\1_\2', UNDERSCORE[0].sub(r'\1_\2', text)).lower()

def is_numeric_dtype(dtype):
    """Return ``True`` if ``dtype`` is a numeric type."""
    dtype = np.dtype(dtype)
    return np.issubsctype(getattr(dtype, 'base', None), np.number)

def _string_hash(s):
    """String hash (djb2) with consistency between py2/py3 and persistency between runs (unlike `hash`)."""
    h = 5381
    for c in s:
        h = h * 33 + ord(c)
    return h

def _check_valid(key, val, valid):
    """Helper to check valid options"""
    if val not in valid:
        raise ValueError('%s must be one of %s, not "%s"'
                         % (key, valid, val))

def camel_to_(s):
    """
    Convert CamelCase to camel_case
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def has_field(mc, field_name):
    """
    detect if a model has a given field has

    :param field_name:
    :param mc:
    :return:
    """
    try:
        mc._meta.get_field(field_name)
    except FieldDoesNotExist:
        return False
    return True

def blueprint_name_to_url(name):
        """ remove the last . in the string it it ends with a .
            for the url structure must follow the flask routing format
            it should be /model/method instead of /model/method/
        """
        if name[-1:] == ".":
            name = name[:-1]
        name = str(name).replace(".", "/")
        return name

def pid_exists(pid):
    """ Determines if a system process identifer exists in process table.
        """
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    else:
        return True

def call_and_exit(self, cmd, shell=True):
        """Run the *cmd* and exit with the proper exit code."""
        sys.exit(subprocess.call(cmd, shell=shell))

def contains_geometric_info(var):
    """ Check whether the passed variable is a tuple with two floats or integers """
    return isinstance(var, tuple) and len(var) == 2 and all(isinstance(val, (int, float)) for val in var)

def weekly(date=datetime.date.today()):
    """
    Weeks start are fixes at Monday for now.
    """
    return date - datetime.timedelta(days=date.weekday())

def is_bytes(string):
    """Check if a string is a bytes instance

    :param Union[str, bytes] string: A string that may be string or bytes like
    :return: Whether the provided string is a bytes type or not
    :rtype: bool
    """
    if six.PY3 and isinstance(string, (bytes, memoryview, bytearray)):  # noqa
        return True
    elif six.PY2 and isinstance(string, (buffer, bytearray)):  # noqa
        return True
    return False

def Sum(a, axis, keep_dims):
    """
    Sum reduction op.
    """
    return np.sum(a, axis=axis if not isinstance(axis, np.ndarray) else tuple(axis),
                  keepdims=keep_dims),

def closeEvent(self, e):
        """Qt slot when the window is closed."""
        if self._closed:
            return
        res = self.emit('close')
        # Discard the close event if False is returned by one of the callback
        # functions.
        if False in res:  # pragma: no cover
            e.ignore()
            return
        super(GUI, self).closeEvent(e)
        self._closed = True

def print_err(*args, end='\n'):
    """Similar to print, but prints to stderr.
    """
    print(*args, end=end, file=sys.stderr)
    sys.stderr.flush()

def _isstring(dtype):
    """Given a numpy dtype, determines whether it is a string. Returns True
    if the dtype is string or unicode.
    """
    return dtype.type == numpy.unicode_ or dtype.type == numpy.string_

def byteswap(data, word_size=4):
    """ Swap the byte-ordering in a packet with N=4 bytes per word
    """
    return reduce(lambda x,y: x+''.join(reversed(y)), chunks(data, word_size), '')

def is_executable(path):
  """Returns whether a path names an existing executable file."""
  return os.path.isfile(path) and os.access(path, os.X_OK)

def instance_contains(container, item):
    """Search into instance attributes, properties and return values of no-args methods."""
    return item in (member for _, member in inspect.getmembers(container))

def get_dimension_array(array):
    """
    Get dimension of an array getting the number of rows and the max num of
    columns.
    """
    if all(isinstance(el, list) for el in array):
        result = [len(array), len(max([x for x in array], key=len,))]

    # elif array and isinstance(array, list):
    else:
        result = [len(array), 1]

    return result

def is_valid_file(parser, arg):
    """Check if arg is a valid file that already exists on the file system."""
    arg = os.path.abspath(arg)
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg

def ci(a, which=95, axis=None):
    """Return a percentile range from an array of values."""
    p = 50 - which / 2, 50 + which / 2
    return percentiles(a, p, axis)

def is_readable(fp, size=1):
    """
    Check if the file-like object is readable.

    :param fp: file-like object
    :param size: byte size
    :return: bool
    """
    read_size = len(fp.read(size))
    fp.seek(-read_size, 1)
    return read_size == size

def determine_interactive(self):
		"""Determine whether we're in an interactive shell.
		Sets interactivity off if appropriate.
		cf http://stackoverflow.com/questions/24861351/how-to-detect-if-python-script-is-being-run-as-a-background-process
		"""
		try:
			if not sys.stdout.isatty() or os.getpgrp() != os.tcgetpgrp(sys.stdout.fileno()):
				self.interactive = 0
				return False
		except Exception:
			self.interactive = 0
			return False
		if self.interactive == 0:
			return False
		return True

def _first_and_last_element(arr):
    """Returns first and last element of numpy array or sparse matrix."""
    if isinstance(arr, np.ndarray) or hasattr(arr, 'data'):
        # numpy array or sparse matrix with .data attribute
        data = arr.data if sparse.issparse(arr) else arr
        return data.flat[0], data.flat[-1]
    else:
        # Sparse matrices without .data attribute. Only dok_matrix at
        # the time of writing, in this case indexing is fast
        return arr[0, 0], arr[-1, -1]

def __contains__(self, key):
        """
        Invoked when determining whether a specific key is in the dictionary
        using `key in d`.

        The key is looked up case-insensitively.
        """
        k = self._real_key(key)
        return k in self._data

def has_field(mc, field_name):
    """
    detect if a model has a given field has

    :param field_name:
    :param mc:
    :return:
    """
    try:
        mc._meta.get_field(field_name)
    except FieldDoesNotExist:
        return False
    return True

def is_valid_regex(string):
    """
    Checks whether the re module can compile the given regular expression.

    Parameters
    ----------
    string: str

    Returns
    -------
    boolean
    """
    try:
        re.compile(string)
        is_valid = True
    except re.error:
        is_valid = False
    return is_valid

def pid_exists(pid):
    """ Determines if a system process identifer exists in process table.
        """
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    else:
        return True

def is_listish(obj):
    """Check if something quacks like a list."""
    if isinstance(obj, (list, tuple, set)):
        return True
    return is_sequence(obj)

def _using_stdout(self):
        """
        Return whether the handler is using sys.stdout.
        """
        if WINDOWS and colorama:
            # Then self.stream is an AnsiToWin32 object.
            return self.stream.wrapped is sys.stdout

        return self.stream is sys.stdout

def unicode_is_ascii(u_string):
    """Determine if unicode string only contains ASCII characters.

    :param str u_string: unicode string to check. Must be unicode
        and not Python 2 `str`.
    :rtype: bool
    """
    assert isinstance(u_string, str)
    try:
        u_string.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False

def is_hex_string(string):
    """Check if the string is only composed of hex characters."""
    pattern = re.compile(r'[A-Fa-f0-9]+')
    if isinstance(string, six.binary_type):
        string = str(string)
    return pattern.match(string) is not None

def is_lazy_iterable(obj):
    """
    Returns whether *obj* is iterable lazily, such as generators, range objects, etc.
    """
    return isinstance(obj,
        (types.GeneratorType, collections.MappingView, six.moves.range, enumerate))

def user_in_all_groups(user, groups):
    """Returns True if the given user is in all given groups"""
    return user_is_superuser(user) or all(user_in_group(user, group) for group in groups)

def unicode_is_ascii(u_string):
    """Determine if unicode string only contains ASCII characters.

    :param str u_string: unicode string to check. Must be unicode
        and not Python 2 `str`.
    :rtype: bool
    """
    assert isinstance(u_string, str)
    try:
        u_string.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False

def wrap(s, width=80):
    """
    Formats the text input with newlines given the user specified width for
    each line.

    Parameters
    ----------

    s : str
    width : int

    Returns
    -------

    text : str

    Notes
    -----

    .. versionadded:: 1.1

    """
    return '\n'.join(textwrap.wrap(str(s), width=width))

def has_virtualenv(self):
        """
        Returns true if the virtualenv tool is installed.
        """
        with self.settings(warn_only=True):
            ret = self.run_or_local('which virtualenv').strip()
            return bool(ret)

def normalize_text(text, line_len=80, indent=""):
    """Wrap the text on the given line length."""
    return "\n".join(
        textwrap.wrap(
            text, width=line_len, initial_indent=indent, subsequent_indent=indent
        )
    )

def get_substring_idxs(substr, string):
    """
    Return a list of indexes of substr. If substr not found, list is
    empty.

    Arguments:
        substr (str): Substring to match.
        string (str): String to match in.

    Returns:
        list of int: Start indices of substr.
    """
    return [match.start() for match in re.finditer(substr, string)]

def mul(a, b):
  """
  A wrapper around tf multiplication that does more automatic casting of
  the input.
  """
  def multiply(a, b):
    """Multiplication"""
    return a * b
  return op_with_scalar_cast(a, b, multiply)

def is_edge_consistent(graph, u, v):
    """Check if all edges between two nodes have the same relation.

    :param pybel.BELGraph graph: A BEL Graph
    :param tuple u: The source BEL node
    :param tuple v: The target BEL node
    :return: If all edges from the source to target node have the same relation
    :rtype: bool
    """
    if not graph.has_edge(u, v):
        raise ValueError('{} does not contain an edge ({}, {})'.format(graph, u, v))

    return 0 == len(set(d[RELATION] for d in graph.edge[u][v].values()))

def percent_d(data, period):
    """
    %D.

    Formula:
    %D = SMA(%K, 3)
    """
    p_k = percent_k(data, period)
    percent_d = sma(p_k, 3)
    return percent_d

def service_available(service_name):
    """Determine whether a system service is available"""
    try:
        subprocess.check_output(
            ['service', service_name, 'status'],
            stderr=subprocess.STDOUT).decode('UTF-8')
    except subprocess.CalledProcessError as e:
        return b'unrecognized service' not in e.output
    else:
        return True

def _dt_to_epoch(dt):
        """Convert datetime to epoch seconds."""
        try:
            epoch = dt.timestamp()
        except AttributeError:  # py2
            epoch = (dt - datetime(1970, 1, 1)).total_seconds()
        return epoch

def is_hex_string(string):
    """Check if the string is only composed of hex characters."""
    pattern = re.compile(r'[A-Fa-f0-9]+')
    if isinstance(string, six.binary_type):
        string = str(string)
    return pattern.match(string) is not None

def now(timezone=None):
    """
    Return a naive datetime object for the given ``timezone``. A ``timezone``
    is any pytz- like or datetime.tzinfo-like timezone object. If no timezone
    is given, then UTC is assumed.

    This method is best used with pytz installed::

        pip install pytz
    """
    d = datetime.datetime.utcnow()
    if not timezone:
        return d

    return to_timezone(d, timezone).replace(tzinfo=None)

def _is_path(s):
    """Return whether an object is a path."""
    if isinstance(s, string_types):
        try:
            return op.exists(s)
        except (OSError, ValueError):
            return False
    else:
        return False

def popup(self, title, callfn, initialdir=None):
        """Let user select a directory."""
        super(DirectorySelection, self).popup(title, callfn, initialdir)

def _match_literal(self, a, b=None):
        """Match two names."""

        return a.lower() == b if not self.case_sensitive else a == b

def closing_plugin(self, cancelable=False):
        """Perform actions before parent main window is closed"""
        self.dialog_manager.close_all()
        self.shell.exit_interpreter()
        return True

def is_connected(self):
        """
        Return true if the socket managed by this connection is connected

        :rtype: bool
        """
        try:
            return self.socket is not None and self.socket.getsockname()[1] != 0 and BaseTransport.is_connected(self)
        except socket.error:
            return False

def is_timestamp(obj):
    """
    Yaml either have automatically converted it to a datetime object
    or it is a string that will be validated later.
    """
    return isinstance(obj, datetime.datetime) or is_string(obj) or is_int(obj) or is_float(obj)

def count_words(file):
  """ Counts the word frequences in a list of sentences.

  Note:
    This is a helper function for parallel execution of `Vocabulary.from_text`
    method.
  """
  c = Counter()
  with open(file, 'r') as f:
    for l in f:
      words = l.strip().split()
      c.update(words)
  return c

def isreal(obj):
    """
    Test if the argument is a real number (float or integer).

    :param obj: Object
    :type  obj: any

    :rtype: boolean
    """
    return (
        (obj is not None)
        and (not isinstance(obj, bool))
        and isinstance(obj, (int, float))
    )

def validate_email(email):
    """
    Validates an email address
    Source: Himanshu Shankar (https://github.com/iamhssingh)
    Parameters
    ----------
    email: str

    Returns
    -------
    bool
    """
    from django.core.validators import validate_email
    from django.core.exceptions import ValidationError
    try:
        validate_email(email)
        return True
    except ValidationError:
        return False

def irecarray_to_py(a):
    """Slow conversion of a recarray into a list of records with python types.

    Get the field names from :attr:`a.dtype.names`.

    :Returns: iterator so that one can handle big input arrays
    """
    pytypes = [pyify(typestr) for name,typestr in a.dtype.descr]
    def convert_record(r):
        return tuple([converter(value) for converter, value in zip(pytypes,r)])
    return (convert_record(r) for r in a)

def is_image(filename):
    """Determine if given filename is an image."""
    # note: isfile() also accepts symlinks
    return os.path.isfile(filename) and filename.lower().endswith(ImageExts)

def _encode_gif(images, fps):
  """Encodes numpy images into gif string.

  Args:
    images: A 4-D `uint8` `np.array` (or a list of 3-D images) of shape
      `[time, height, width, channels]` where `channels` is 1 or 3.
    fps: frames per second of the animation

  Returns:
    The encoded gif string.

  Raises:
    IOError: If the ffmpeg command returns an error.
  """
  writer = WholeVideoWriter(fps)
  writer.write_multi(images)
  return writer.finish()

def current_zipfile():
    """A function to vend the current zipfile, if any"""
    if zipfile.is_zipfile(sys.argv[0]):
        fd = open(sys.argv[0], "rb")
        return zipfile.ZipFile(fd)

def num_leaves(tree):
    """Determine the number of leaves in a tree"""
    if tree.is_leaf:
        return 1
    else:
        return num_leaves(tree.left_child) + num_leaves(tree.right_child)

def _pip_exists(self):
        """Returns True if pip exists inside the virtual environment. Can be
        used as a naive way to verify that the environment is installed."""
        return os.path.isfile(os.path.join(self.path, 'bin', 'pip'))

def _crop_list_to_size(l, size):
    """Make a list a certain size"""
    for x in range(size - len(l)):
        l.append(False)
    for x in range(len(l) - size):
        l.pop()
    return l

def duplicated_rows(df, col_name):
    """ Return a DataFrame with the duplicated values of the column `col_name`
    in `df`."""
    _check_cols(df, [col_name])

    dups = df[pd.notnull(df[col_name]) & df.duplicated(subset=[col_name])]
    return dups

def delimited(items, character='|'):
    """Returns a character delimited version of the provided list as a Python string"""
    return '|'.join(items) if type(items) in (list, tuple, set) else items

def is_image(filename):
    """Determine if given filename is an image."""
    # note: isfile() also accepts symlinks
    return os.path.isfile(filename) and filename.lower().endswith(ImageExts)

def list2dict(lst):
    """Takes a list of (key,value) pairs and turns it into a dict."""

    dic = {}
    for k,v in lst: dic[k] = v
    return dic

def time_string(seconds):
    """Returns time in seconds as a string formatted HHHH:MM:SS."""
    s = int(round(seconds))  # round to nearest second
    h, s = divmod(s, 3600)  # get hours and remainder
    m, s = divmod(s, 60)  # split remainder into minutes and seconds
    return "%2i:%02i:%02i" % (h, m, s)

def close_database_session(session):
    """Close connection with the database"""

    try:
        session.close()
    except OperationalError as e:
        raise DatabaseError(error=e.orig.args[1], code=e.orig.args[0])

def load_files(files):
    """Load and execute a python file."""

    for py_file in files:
        LOG.debug("exec %s", py_file)
        execfile(py_file, globals(), locals())

def close(self):
        """Flush the file and close it.

        A closed file cannot be written any more. Calling
        :meth:`close` more than once is allowed.
        """
        if not self._closed:
            self.__flush()
            object.__setattr__(self, "_closed", True)

def isnumber(*args):
    """Checks if value is an integer, long integer or float.

    NOTE: Treats booleans as numbers, where True=1 and False=0.
    """
    return all(map(lambda c: isinstance(c, int) or isinstance(c, float), args))

def socket_close(self):
        """Close our socket."""
        if self.sock != NC.INVALID_SOCKET:
            self.sock.close()
        self.sock = NC.INVALID_SOCKET

def _get_local_ip():
        """
        Get the local ip of this device

        :return: Ip of this computer
        :rtype: str
        """
        return set([x[4][0] for x in socket.getaddrinfo(
            socket.gethostname(),
            80,
            socket.AF_INET
        )]).pop()

def lin_interp(x, rangeX, rangeY):
    """
    Interpolate linearly variable x in rangeX onto rangeY.
    """
    s = (x - rangeX[0]) / mag(rangeX[1] - rangeX[0])
    y = rangeY[0] * (1 - s) + rangeY[1] * s
    return y

def checkbox_uncheck(self, force_check=False):
        """
        Wrapper to uncheck a checkbox
        """
        if self.get_attribute('checked'):
            self.click(force_click=force_check)

def median(data):
    """Calculate the median of a list."""
    data.sort()
    num_values = len(data)
    half = num_values // 2
    if num_values % 2:
        return data[half]
    return 0.5 * (data[half-1] + data[half])

def isTestCaseDisabled(test_case_class, method_name):
    """
    I check to see if a method on a TestCase has been disabled via nose's
    convention for disabling a TestCase.  This makes it so that users can
    mix nose's parameterized tests with green as a runner.
    """
    test_method = getattr(test_case_class, method_name)
    return getattr(test_method, "__test__", 'not nose') is False

def set_cursor_position(self, position):
        """Set cursor position"""
        position = self.get_position(position)
        cursor = self.textCursor()
        cursor.setPosition(position)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

def test(ctx, all=False, verbose=False):
    """Run the tests."""
    cmd = 'tox' if all else 'py.test'
    if verbose:
        cmd += ' -v'
    return ctx.run(cmd, pty=True).return_code

def get_selected_values(self, selection):
        """Return a list of values for the given selection."""
        return [v for b, v in self._choices if b & selection]

def __call__(self, actual_value, expect):
        """Main entry point for assertions (called by the wrapper).
        expect is a function the wrapper class uses to assert a given match.
        """
        self._expect = expect
        if self.expected_value is NO_ARG:
            return self.asserts(actual_value)
        return self.asserts(actual_value, self.expected_value)

def save_excel(self, fd):
        """ Saves the case as an Excel spreadsheet.
        """
        from pylon.io.excel import ExcelWriter
        ExcelWriter(self).write(fd)

def set_global(node: Node, key: str, value: Any):
    """Adds passed value to node's globals"""
    node.node_globals[key] = value

def is_valid_file(parser, arg):
    """Check if arg is a valid file that already exists on the file system."""
    arg = os.path.abspath(arg)
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg

def update(self, params):
        """Update the dev_info data from a dictionary.

        Only updates if it already exists in the device.
        """
        dev_info = self.json_state.get('deviceInfo')
        dev_info.update({k: params[k] for k in params if dev_info.get(k)})

def _begins_with_one_of(sentence, parts_of_speech):
    """Return True if the sentence or fragment begins with one of the parts of
    speech in the list, else False"""
    doc = nlp(sentence)
    if doc[0].tag_ in parts_of_speech:
        return True
    return False

def rest_put_stream(self, url, stream, headers=None, session=None, verify=True, cert=None):
        """
        Perform a chunked PUT request to url with requests.session
        This is specifically to upload files.
        """
        res = session.put(url, headers=headers, data=stream, verify=verify, cert=cert)
        return res.text, res.status_code

def EvalBinomialPmf(k, n, p):
    """Evaluates the binomial pmf.

    Returns the probabily of k successes in n trials with probability p.
    """
    return scipy.stats.binom.pmf(k, n, p)

def get_url_nofollow(url):
	""" 
	function to get return code of a url

	Credits: http://blog.jasonantman.com/2013/06/python-script-to-check-a-list-of-urls-for-return-code-and-final-return-code-if-redirected/
	"""
	try:
		response = urlopen(url)
		code = response.getcode()
		return code
	except HTTPError as e:
		return e.code
	except:
		return 0

def energy_string_to_float( string ):
    """
    Convert a string of a calculation energy, e.g. '-1.2345 eV' to a float.

    Args:
        string (str): The string to convert.
  
    Return
        (float) 
    """
    energy_re = re.compile( "(-?\d+\.\d+)" )
    return float( energy_re.match( string ).group(0) )

def compare(a, b):
    """
     Compare items in 2 arrays. Returns sum(abs(a(i)-b(i)))
    """
    s=0
    for i in range(len(a)):
        s=s+abs(a[i]-b[i])
    return s

def hflip(img):
    """Horizontally flip the given PIL Image.

    Args:
        img (PIL Image): Image to be flipped.

    Returns:
        PIL Image:  Horizontall flipped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.transpose(Image.FLIP_LEFT_RIGHT)

def ex(self, cmd):
        """Execute a normal python statement in user namespace."""
        with self.builtin_trap:
            exec cmd in self.user_global_ns, self.user_ns

def get_creation_datetime(filepath):
    """
    Get the date that a file was created.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    creation_datetime : datetime.datetime or None
    """
    if platform.system() == 'Windows':
        return datetime.fromtimestamp(os.path.getctime(filepath))
    else:
        stat = os.stat(filepath)
        try:
            return datetime.fromtimestamp(stat.st_birthtime)
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return None

def _fill_array_from_list(the_list, the_array):
        """Fill an `array` from a `list`"""
        for i, val in enumerate(the_list):
            the_array[i] = val
        return the_array

def main(argv=None):
  """Run a Tensorflow model on the Iris dataset."""
  args = parse_arguments(sys.argv if argv is None else argv)

  tf.logging.set_verbosity(tf.logging.INFO)
  learn_runner.run(
      experiment_fn=get_experiment_fn(args),
      output_dir=args.job_dir)

def _sort_lambda(sortedby='cpu_percent',
                 sortedby_secondary='memory_percent'):
    """Return a sort lambda function for the sortedbykey"""
    ret = None
    if sortedby == 'io_counters':
        ret = _sort_io_counters
    elif sortedby == 'cpu_times':
        ret = _sort_cpu_times
    return ret

def sample_colormap(cmap_name, n_samples):
    """
    Sample a colormap from matplotlib
    """
    colors = []
    colormap = cm.cmap_d[cmap_name]
    for i in np.linspace(0, 1, n_samples):
        colors.append(colormap(i))

    return colors

def calculate_size(name, count):
    """ Calculates the request payload size"""
    data_size = 0
    data_size += calculate_size_str(name)
    data_size += INT_SIZE_IN_BYTES
    return data_size

def set_color(self, fg=None, bg=None, intensify=False, target=sys.stdout):
        """Set foreground- and background colors and intensity."""
        raise NotImplementedError

def polyline(self, arr):
        """Draw a set of lines"""
        for i in range(0, len(arr) - 1):
            self.line(arr[i][0], arr[i][1], arr[i + 1][0], arr[i + 1][1])

def colorize(string, color, *args, **kwargs):
    """
    Implements string formatting along with color specified in colorama.Fore
    """
    string = string.format(*args, **kwargs)
    return color + string + colorama.Fore.RESET

def multiply(traj):
    """Sophisticated simulation of multiplication"""
    z=traj.x*traj.y
    traj.f_add_result('z',z=z, comment='I am the product of two reals!')

def isetdiff_flags(list1, list2):
    """
    move to util_iter
    """
    set2 = set(list2)
    return (item not in set2 for item in list1)

def column_names(self, table):
      """An iterable of column names, for a particular table or
      view."""

      table_info = self.execute(
        u'PRAGMA table_info(%s)' % quote(table))
      return (column['name'] for column in table_info)

def matchfieldnames(field_a, field_b):
    """Check match between two strings, ignoring case and spaces/underscores.
    
    Parameters
    ----------
    a : str
    b : str
    
    Returns
    -------
    bool
    
    """
    normalised_a = field_a.replace(' ', '_').lower()
    normalised_b = field_b.replace(' ', '_').lower()
    
    return normalised_a == normalised_b

def filter_regex(names, regex):
    """
    Return a tuple of strings that match the regular expression pattern.
    """
    return tuple(name for name in names
                 if regex.search(name) is not None)

def assert_looks_like(first, second, msg=None):
    """ Compare two strings if all contiguous whitespace is coalesced. """
    first = _re.sub("\s+", " ", first.strip())
    second = _re.sub("\s+", " ", second.strip())
    if first != second:
        raise AssertionError(msg or "%r does not look like %r" % (first, second))

def constraint_range_dict(self,*args,**kwargs):
        """ 
            Creates a list of dictionaries which each give a constraint for a certain
            section of the dimension.

            bins arguments overwrites resolution
        """
        bins = self.bins(*args,**kwargs)
        return [{self.name+'__gte': a,self.name+'__lt': b} for a,b in zip(bins[:-1],bins[1:])]
        space = self.space(*args,**kwargs)
        resolution = space[1] - space[0]
        return [{self.name+'__gte': s,self.name+'__lt': s+resolution} for s in space]

def is_equal_strings_ignore_case(first, second):
    """The function compares strings ignoring case"""
    if first and second:
        return first.upper() == second.upper()
    else:
        return not (first or second)

def Date(value):
    """Custom type for managing dates in the command-line."""
    from datetime import datetime
    try:
        return datetime(*reversed([int(val) for val in value.split('/')]))
    except Exception as err:
        raise argparse.ArgumentTypeError("invalid date '%s'" % value)

def compare_dict(da, db):
    """
    Compare differencs from two dicts
    """
    sa = set(da.items())
    sb = set(db.items())
    
    diff = sa & sb
    return dict(sa - diff), dict(sb - diff)

def assert_list(self, putative_list, expected_type=string_types, key_arg=None):
    """
    :API: public
    """
    return assert_list(putative_list, expected_type, key_arg=key_arg,
                       raise_type=lambda msg: TargetDefinitionException(self, msg))

def compare(string1, string2):
    """Compare two strings while protecting against timing attacks

    :param str string1: the first string
    :param str string2: the second string

    :returns: True if the strings are equal, False if not
    :rtype: :obj:`bool`
    """
    if len(string1) != len(string2):
        return False
    result = True
    for c1, c2 in izip(string1, string2):
        result &= c1 == c2
    return result

def print_display_png(o):
    """
    A function to display sympy expression using display style LaTeX in PNG.
    """
    s = latex(o, mode='plain')
    s = s.strip('$')
    # As matplotlib does not support display style, dvipng backend is
    # used here.
    png = latex_to_png('$$%s$$' % s, backend='dvipng')
    return png

def __similarity(s1, s2, ngrams_fn, n=3):
    """
        The fraction of n-grams matching between two sequences

        Args:
            s1: a string
            s2: another string
            n: an int for the n in n-gram

        Returns:
            float: the fraction of n-grams matching
    """
    ngrams1, ngrams2 = set(ngrams_fn(s1, n=n)), set(ngrams_fn(s2, n=n))
    matches = ngrams1.intersection(ngrams2)
    return 2 * len(matches) / (len(ngrams1) + len(ngrams2))

def __complex__(self):
        """Called to implement the built-in function complex()."""
        if self._t != 99 or self.key != ['re', 'im']:
            return complex(float(self))
        return complex(float(self.re), float(self.im))

def extract_pdfminer(self, filename, **kwargs):
        """Extract text from pdfs using pdfminer."""
        stdout, _ = self.run(['pdf2txt.py', filename])
        return stdout

def _pdf_at_peak(self):
    """Pdf evaluated at the peak."""
    return (self.peak - self.low) / (self.high - self.low)

def is_valid(number):
    """determines whether the card number is valid."""
    n = str(number)
    if not n.isdigit():
        return False
    return int(n[-1]) == get_check_digit(n[:-1])

def items(self, section_name):
        """:return: list((option, value), ...) pairs of all items in the given section"""
        return [(k, v) for k, v in super(GitConfigParser, self).items(section_name) if k != '__name__']

def validate_email(email):
    """
    Validates an email address
    Source: Himanshu Shankar (https://github.com/iamhssingh)
    Parameters
    ----------
    email: str

    Returns
    -------
    bool
    """
    from django.core.validators import validate_email
    from django.core.exceptions import ValidationError
    try:
        validate_email(email)
        return True
    except ValidationError:
        return False

def is_valid_file(parser,arg):
	"""verify the validity of the given file. Never trust the End-User"""
	if not os.path.exists(arg):
       		parser.error("File %s not found"%arg)
	else:
	       	return arg

def __connect():
    """
    Connect to a redis instance.
    """
    global redis_instance
    if use_tcp_socket:
        redis_instance = redis.StrictRedis(host=hostname, port=port)
    else:
        redis_instance = redis.StrictRedis(unix_socket_path=unix_socket)

def is_valid_file(parser, arg):
    """Check if arg is a valid file that already exists on the file system."""
    arg = os.path.abspath(arg)
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg

def _read_stdin():
    """
    Generator for reading from standard input in nonblocking mode.

    Other ways of reading from ``stdin`` in python waits, until the buffer is
    big enough, or until EOF character is sent.

    This functions yields immediately after each line.
    """
    line = sys.stdin.readline()
    while line:
        yield line
        line = sys.stdin.readline()

def apply(f, obj, *args, **kwargs):
    """Apply a function in parallel to each element of the input"""
    return vectorize(f)(obj, *args, **kwargs)

def read(self):
        """Iterate over all JSON input (Generator)"""

        for line in self.io.read():
            with self.parse_line(line) as j:
                yield j

def get_code(module):
    """
    Compile and return a Module's code object.
    """
    fp = open(module.path)
    try:
        return compile(fp.read(), str(module.name), 'exec')
    finally:
        fp.close()

def stringify_dict_contents(dct):
    """Turn dict keys and values into native strings."""
    return {
        str_if_nested_or_str(k): str_if_nested_or_str(v)
        for k, v in dct.items()
    }

def prettyprint(d):
        """Print dicttree in Json-like format. keys are sorted
        """
        print(json.dumps(d, sort_keys=True, 
                         indent=4, separators=("," , ": ")))

def _parse_canonical_int64(doc):
    """Decode a JSON int64 to bson.int64.Int64."""
    l_str = doc['$numberLong']
    if len(doc) != 1:
        raise TypeError('Bad $numberLong, extra field(s): %s' % (doc,))
    return Int64(l_str)

def is_int(value):
    """Return `True` if ``value`` is an integer."""
    if isinstance(value, bool):
        return False
    try:
        int(value)
        return True
    except (ValueError, TypeError):
        return False

def zoom_cv(x,z):
    """ Zoom the center of image x by a factor of z+1 while retaining the original image size and proportion. """
    if z==0: return x
    r,c,*_ = x.shape
    M = cv2.getRotationMatrix2D((c/2,r/2),0,z+1.)
    return cv2.warpAffine(x,M,(c,r))

def find_commons(lists):
    """Finds common values

    :param lists: List of lists
    :return: List of values that are in common between inner lists
    """
    others = lists[1:]
    return [
        val
        for val in lists[0]
        if is_in_all(val, others)
    ]

def _get_compiled_ext():
    """Official way to get the extension of compiled files (.pyc or .pyo)"""
    for ext, mode, typ in imp.get_suffixes():
        if typ == imp.PY_COMPILED:
            return ext

def not0(a):
    """Return u if u!= 0, return 1 if u == 0"""
    return matrix(list(map(lambda x: 1 if x == 0 else x, a)), a.size)

def return_letters_from_string(text):
    """Get letters from string only."""
    out = ""
    for letter in text:
        if letter.isalpha():
            out += letter
    return out

def count_rows_with_nans(X):
    """Count the number of rows in 2D arrays that contain any nan values."""
    if X.ndim == 2:
        return np.where(np.isnan(X).sum(axis=1) != 0, 1, 0).sum()

def link(self, mu, dist):
        """
        glm link function
        this is useful for going from mu to the linear prediction

        Parameters
        ----------
        mu : array-like of legth n
        dist : Distribution instance

        Returns
        -------
        lp : np.array of length n
        """
        return np.log(mu) - np.log(dist.levels - mu)

def lines(self):
        """

        :return:
        """
        if self._lines is None:
            self._lines = self.obj.content.splitlines()
        return self._lines

def coverage():
    """check code coverage quickly with the default Python"""
    run("coverage run --source {PROJECT_NAME} -m py.test".format(PROJECT_NAME=PROJECT_NAME))
    run("coverage report -m")
    run("coverage html")

    webbrowser.open('file://' + os.path.realpath("htmlcov/index.html"), new=2)

def fail(message=None, exit_status=None):
    """Prints the specified message and exits the program with the specified
    exit status.

    """
    print('Error:', message, file=sys.stderr)
    sys.exit(exit_status or 1)

def md_to_text(content):
    """ Converts markdown content to text """
    text = None
    html = markdown.markdown(content)
    if html:
        text = html_to_text(content)
    return text

def __unixify(self, s):
        """ stupid windows. converts the backslash to forwardslash for consistency """
        return os.path.normpath(s).replace(os.sep, "/")

def format_result(input):
        """From: http://stackoverflow.com/questions/13062300/convert-a-dict-to-sorted-dict-in-python
        """
        items = list(iteritems(input))
        return OrderedDict(sorted(items, key=lambda x: x[0]))

def kill(self):
        """Kill the browser.

        This is useful when the browser is stuck.
        """
        if self.process:
            self.process.kill()
            self.process.wait()

def exit(exit_code=0):
  r"""A function to support exiting from exit hooks.

  Could also be used to exit from the calling scripts in a thread safe manner.
  """
  core.processExitHooks()

  if state.isExitHooked and not hasattr(sys, 'exitfunc'): # The function is called from the exit hook
    sys.stderr.flush()
    sys.stdout.flush()
    os._exit(exit_code) #pylint: disable=W0212

  sys.exit(exit_code)

def vectorsToMatrix(aa, bb):
    """
    Performs the vector multiplication of the elements of two vectors, constructing the 3x3 matrix.
    :param aa: One vector of size 3
    :param bb: Another vector of size 3
    :return: A 3x3 matrix M composed of the products of the elements of aa and bb :
     M_ij = aa_i * bb_j
    """
    MM = np.zeros([3, 3], np.float)
    for ii in range(3):
        for jj in range(3):
            MM[ii, jj] = aa[ii] * bb[jj]
    return MM

def int2str(num, radix=10, alphabet=BASE85):
    """helper function for quick base conversions from integers to strings"""
    return NumConv(radix, alphabet).int2str(num)

def create_rot2d(angle):
    """Create 2D rotation matrix"""
    ca = math.cos(angle)
    sa = math.sin(angle)
    return np.array([[ca, -sa], [sa, ca]])

def packagenameify(s):
  """
  Makes a package name
  """
  return ''.join(w if w in ACRONYMS else w.title() for w in s.split('.')[-1:])

def from_json_str(cls, json_str):
    """Convert json string representation into class instance.

    Args:
      json_str: json representation as string.

    Returns:
      New instance of the class with data loaded from json string.
    """
    return cls.from_json(json.loads(json_str, cls=JsonDecoder))

def scipy_sparse_to_spmatrix(A):
    """Efficient conversion from scipy sparse matrix to cvxopt sparse matrix"""
    coo = A.tocoo()
    SP = spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=A.shape)
    return SP

def build_service_class(metadata):
    """Generate a service class for the service contained in the specified metadata class."""
    i = importlib.import_module(metadata)
    service = i.service
    env = get_jinja_env()
    service_template = env.get_template('service.py.jinja2')
    with open(api_path(service.name.lower()), 'w') as t:
        t.write(service_template.render(service_md=service))

def get_next_scheduled_time(cron_string):
    """Calculate the next scheduled time by creating a crontab object
    with a cron string"""
    itr = croniter.croniter(cron_string, datetime.utcnow())
    return itr.get_next(datetime)

def _trim(image):
    """Trim a PIL image and remove white space."""
    background = PIL.Image.new(image.mode, image.size, image.getpixel((0, 0)))
    diff = PIL.ImageChops.difference(image, background)
    diff = PIL.ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        image = image.crop(bbox)
    return image

def on_close(self, evt):
    """
    Pop-up menu and wx.EVT_CLOSE closing event
    """
    self.stop() # DoseWatcher
    if evt.EventObject is not self: # Avoid deadlocks
      self.Close() # wx.Frame
    evt.Skip()

def _mean_absolute_error(y, y_pred, w):
    """Calculate the mean absolute error."""
    return np.average(np.abs(y_pred - y), weights=w)

def xeval(source, optimize=True):
    """Compiles to native Python bytecode and runs program, returning the
    topmost value on the stack.

    Args:
        optimize: Whether to optimize the code after parsing it.

    Returns:
        None: If the stack is empty
        obj: If the stack contains a single value
        [obj, obj, ...]: If the stack contains many values
    """
    native = xcompile(source, optimize=optimize)
    return native()

def yank(event):
    """
    Paste before cursor.
    """
    event.current_buffer.paste_clipboard_data(
        event.cli.clipboard.get_data(), count=event.arg, paste_mode=PasteMode.EMACS)

def Parse(text):
  """Parses a YAML source into a Python object.

  Args:
    text: A YAML source to parse.

  Returns:
    A Python data structure corresponding to the YAML source.
  """
  precondition.AssertType(text, Text)

  if compatibility.PY2:
    text = text.encode("utf-8")

  return yaml.safe_load(text)

def _monitor_callback_wrapper(callback):
    """A wrapper for the user-defined handle."""
    def callback_handle(name, array, _):
        """ ctypes function """
        callback(name, array)
    return callback_handle

def extract_zip(zip_path, target_folder):
    """
    Extract the content of the zip-file at `zip_path` into `target_folder`.
    """
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target_folder)

def cint32_array_to_numpy(cptr, length):
    """Convert a ctypes int pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_int32)):
        return np.fromiter(cptr, dtype=np.int32, count=length)
    else:
        raise RuntimeError('Expected int pointer')

def _run_asyncio(loop, zmq_context):
    """
    Run asyncio (should be called in a thread) and close the loop and the zmq context when the thread ends
    :param loop:
    :param zmq_context:
    :return:
    """
    try:
        asyncio.set_event_loop(loop)
        loop.run_forever()
    except:
        pass
    finally:
        loop.close()
        zmq_context.destroy(1000)

def pointer(self):
        """Get a ctypes void pointer to the memory mapped region.

        :type: ctypes.c_void_p
        """
        return ctypes.cast(ctypes.pointer(ctypes.c_uint8.from_buffer(self.mapping, 0)), ctypes.c_void_p)

def cfloat32_array_to_numpy(cptr, length):
    """Convert a ctypes float pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_float)):
        return np.fromiter(cptr, dtype=np.float32, count=length)
    else:
        raise RuntimeError('Expected float pointer')

def POINTER(obj):
    """
    Create ctypes pointer to object.

    Notes
    -----
    This function converts None to a real NULL pointer because of bug
    in how ctypes handles None on 64-bit platforms.

    """

    p = ctypes.POINTER(obj)
    if not isinstance(p.from_param, classmethod):
        def from_param(cls, x):
            if x is None:
                return cls()
            else:
                return x
        p.from_param = classmethod(from_param)

    return p

def is_defined(self, objtxt, force_import=False):
        """Return True if object is defined"""
        return self.interpreter.is_defined(objtxt, force_import)

def validate(self, obj):
        """ Raises django.core.exceptions.ValidationError if any validation error exists """

        if not isinstance(obj, self.model_class):
            raise ValidationError('Invalid object(%s) for service %s' % (type(obj), type(self)))
        LOG.debug(u'Object %s state: %s', self.model_class, obj.__dict__)
        obj.full_clean()

def contains_geometric_info(var):
    """ Check whether the passed variable is a tuple with two floats or integers """
    return isinstance(var, tuple) and len(var) == 2 and all(isinstance(val, (int, float)) for val in var)

def _idx_col2rowm(d):
    """Generate indexes to change from col-major to row-major ordering"""
    if 0 == len(d):
        return 1
    if 1 == len(d):
        return np.arange(d[0])
    # order='F' indicates column-major ordering
    idx = np.array(np.arange(np.prod(d))).reshape(d, order='F').T
    return idx.flatten(order='F')

def is_date_type(cls):
    """Return True if the class is a date type."""
    if not isinstance(cls, type):
        return False
    return issubclass(cls, date) and not issubclass(cls, datetime)

def notin(arg, values):
    """
    Like isin, but checks whether this expression's value(s) are not
    contained in the passed values. See isin docs for full usage.
    """
    op = ops.NotContains(arg, values)
    return op.to_expr()

def __contains__ (self, key):
        """Check lowercase key item."""
        assert isinstance(key, basestring)
        return dict.__contains__(self, key.lower())

def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and 1-hot labels."""
  return 100.0 - (
      100.0 *
      np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
      predictions.shape[0])

def get_time(filename):
	"""
	Get the modified time for a file as a datetime instance
	"""
	ts = os.stat(filename).st_mtime
	return datetime.datetime.utcfromtimestamp(ts)

def createdb():
    """Create database tables from sqlalchemy models"""
    manager.db.engine.echo = True
    manager.db.create_all()
    set_alembic_revision()

def is_password_valid(password):
    """
    Check if a password is valid
    """
    pattern = re.compile(r"^.{4,75}$")
    return bool(pattern.match(password))

def cmd_reindex():
    """Uses CREATE INDEX CONCURRENTLY to create a duplicate index, then tries to swap the new index for the original.

    The index swap is done using a short lock timeout to prevent it from interfering with running queries. Retries until
    the rename succeeds.
    """
    db = connect(args.database)
    for idx in args.indexes:
        pg_reindex(db, idx)

def is_port_open(port, host="127.0.0.1"):
    """
    Check if a port is open
    :param port:
    :param host:
    :return bool:
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((host, int(port)))
        s.shutdown(2)
        return True
    except Exception as e:
        return False

def extend(self, iterable):
        """Extend the list by appending all the items in the given list."""
        return super(Collection, self).extend(
            self._ensure_iterable_is_valid(iterable))

def is_power_of_2(num):
    """Return whether `num` is a power of two"""
    log = math.log2(num)
    return int(log) == float(log)

def ci(a, which=95, axis=None):
    """Return a percentile range from an array of values."""
    p = 50 - which / 2, 50 + which / 2
    return percentiles(a, p, axis)

def cmp_contents(filename1, filename2):
    """ Returns True if contents of the files are the same

    Parameters
    ----------
    filename1 : str
        filename of first file to compare
    filename2 : str
        filename of second file to compare

    Returns
    -------
    tf : bool
        True if binary contents of `filename1` is same as binary contents of
        `filename2`, False otherwise.
    """
    with open_readable(filename1, 'rb') as fobj:
        contents1 = fobj.read()
    with open_readable(filename2, 'rb') as fobj:
        contents2 = fobj.read()
    return contents1 == contents2

def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

def is_client(self):
        """Return True if Glances is running in client mode."""
        return (self.args.client or self.args.browser) and not self.args.server

def _linear_interpolation(x, X, Y):
    """Given two data points [X,Y], linearly interpolate those at x.
    """
    return (Y[1] * (x - X[0]) + Y[0] * (X[1] - x)) / (X[1] - X[0])

def _is_utf_8(txt):
    """
    Check a string is utf-8 encoded

    :param bytes txt: utf-8 string
    :return: Whether the string\
    is utf-8 encoded or not
    :rtype: bool
    """
    assert isinstance(txt, six.binary_type)

    try:
        _ = six.text_type(txt, 'utf-8')
    except (TypeError, UnicodeEncodeError):
        return False
    else:
        return True

def loganalytics_data_plane_client(cli_ctx, _):
    """Initialize Log Analytics data client for use with CLI."""
    from .vendored_sdks.loganalytics import LogAnalyticsDataClient
    from azure.cli.core._profile import Profile
    profile = Profile(cli_ctx=cli_ctx)
    cred, _, _ = profile.get_login_credentials(
        resource="https://api.loganalytics.io")
    return LogAnalyticsDataClient(cred)

def validate(self, val):
        """
        Validates that the val is in the list of values for this Enum.

        Returns two element tuple: (bool, string)

        - `bool` - True if valid, False if not
        - `string` - Description of validation error, or None if valid

        :Parameters:
          val
            Value to validate.  Should be a string.
        """
        if val in self.values:
            return True, None
        else:
            return False, "'%s' is not in enum: %s" % (val, str(self.values))

def token_accuracy(labels, outputs):
  """Compute tokenwise (elementwise) accuracy.

  Args:
    labels: ground-truth labels, shape=(batch, seq_length)
    outputs: predicted tokens, shape=(batch, seq_length)
  Returns:
    Two ops, one for getting the current average accuracy and another for
    updating the running average estimate.
  """
  weights = tf.to_float(tf.not_equal(labels, 0))
  return tf.metrics.accuracy(labels, outputs, weights=weights)

def isin(self, column, compare_list):
        """
        Returns a boolean list where each elements is whether that element in the column is in the compare_list.

        :param column: single column name, does not work for multiple columns
        :param compare_list: list of items to compare to
        :return: list of booleans
        """
        return [x in compare_list for x in self._data[self._columns.index(column)]]

def to_dotfile(self):
        """ Writes a DOT graphviz file of the domain structure, and returns the filename"""
        domain = self.get_domain()
        filename = "%s.dot" % (self.__class__.__name__)
        nx.write_dot(domain, filename)
        return filename

def check(self, var):
        """Check whether the provided value is a valid enum constant."""
        if not isinstance(var, _str_type): return False
        return _enum_mangle(var) in self._consts

def add_bg(img, padding, color=COL_WHITE):
    """
    Adds a padding to the given image as background of specified color

    :param img: Input image.
    :param padding: constant padding around the image.
    :param color: background color that needs to filled for the newly padded region.
    :return: New image with background.
    """
    img = gray3(img)
    h, w, d = img.shape
    new_img = np.ones((h + 2*padding, w + 2*padding, d)) * color[:d]
    new_img = new_img.astype(np.uint8)
    set_img_box(new_img, (padding, padding, w, h), img)
    return new_img

def is_element_present(driver, selector, by=By.CSS_SELECTOR):
    """
    Returns whether the specified element selector is present on the page.
    @Params
    driver - the webdriver object (required)
    selector - the locator that is used (required)
    by - the method to search for the locator (Default: By.CSS_SELECTOR)
    @Returns
    Boolean (is element present)
    """
    try:
        driver.find_element(by=by, value=selector)
        return True
    except Exception:
        return False

def seconds(num):
    """
    Pause for this many seconds
    """
    now = pytime.time()
    end = now + num
    until(end)

def instance_contains(container, item):
    """Search into instance attributes, properties and return values of no-args methods."""
    return item in (member for _, member in inspect.getmembers(container))

def deep_update(d, u):
  """Deeply updates a dictionary. List values are concatenated.

  Args:
    d (dict): First dictionary which will be updated
    u (dict): Second dictionary use to extend the first one

  Returns:
    dict: The merge dictionary

  """

  for k, v in u.items():
    if isinstance(v, Mapping):
      d[k] = deep_update(d.get(k, {}), v)
    elif isinstance(v, list):
      existing_elements = d.get(k, [])
      d[k] = existing_elements + [ele for ele in v if ele not in existing_elements]
    else:
      d[k] = v

  return d

def url_syntax_check(url):  # pragma: no cover
    """
    Check the syntax of the given URL.

    :param url: The URL to check the syntax for.
    :type url: str

    :return: The syntax validity.
    :rtype: bool

    .. warning::
        If an empty or a non-string :code:`url` is given, we return :code:`None`.
    """

    if url and isinstance(url, str):
        # The given URL is not empty nor None.
        # and
        # * The given URL is a string.

        # We silently load the configuration.
        load_config(True)

        return Check(url).is_url_valid()

    # We return None, there is nothing to check.
    return None

def strip_comment_marker(text):
    """ Strip # markers at the front of a block of comment text.
    """
    lines = []
    for line in text.splitlines():
        lines.append(line.lstrip('#'))
    text = textwrap.dedent('\n'.join(lines))
    return text

def _clear(self):
        """Resets all assigned data for the current message."""
        self._finished = False
        self._measurement = None
        self._message = None
        self._message_body = None

def add_suffix(fullname, suffix):
    """ Add suffix to a full file name"""
    name, ext = os.path.splitext(fullname)
    return name + '_' + suffix + ext

def trim(self):
        """Clear not used counters"""
        for key, value in list(iteritems(self.counters)):
            if value.empty():
                del self.counters[key]

def adjacency(tree):
    """
    Construct the adjacency matrix of the tree
    :param tree:
    :return:
    """
    dd = ids(tree)
    N = len(dd)
    A = np.zeros((N, N))

    def _adj(node):
        if np.isscalar(node):
            return
        elif isinstance(node, tuple) and len(node) == 2:
            A[dd[node], dd[node[0]]] = 1
            A[dd[node[0]], dd[node]] = 1
            _adj(node[0])

            A[dd[node], dd[node[1]]] = 1
            A[dd[node[1]], dd[node]] = 1
            _adj(node[1])

    _adj(tree)
    return A

def paste(cmd=paste_cmd, stdout=PIPE):
    """Returns system clipboard contents.
    """
    return Popen(cmd, stdout=stdout).communicate()[0].decode('utf-8')

def adjacency(tree):
    """
    Construct the adjacency matrix of the tree
    :param tree:
    :return:
    """
    dd = ids(tree)
    N = len(dd)
    A = np.zeros((N, N))

    def _adj(node):
        if np.isscalar(node):
            return
        elif isinstance(node, tuple) and len(node) == 2:
            A[dd[node], dd[node[0]]] = 1
            A[dd[node[0]], dd[node]] = 1
            _adj(node[0])

            A[dd[node], dd[node[1]]] = 1
            A[dd[node[1]], dd[node]] = 1
            _adj(node[1])

    _adj(tree)
    return A

def clone(src, **kwargs):
    """Clones object with optionally overridden fields"""
    obj = object.__new__(type(src))
    obj.__dict__.update(src.__dict__)
    obj.__dict__.update(kwargs)
    return obj

def managepy(cmd, extra=None):
    """Run manage.py using this component's specific Django settings"""

    extra = extra.split() if extra else []
    run_django_cli(['invoke', cmd] + extra)

def activate():
    """
    Usage:
      containment activate
    """
    # This is derived from the clone
    cli = CommandLineInterface()
    cli.ensure_config()
    cli.write_dockerfile()
    cli.build()
    cli.run()

def ansi(color, text):
    """Wrap text in an ansi escape sequence"""
    code = COLOR_CODES[color]
    return '\033[1;{0}m{1}{2}'.format(code, text, RESET_TERM)

def file_read(filename):
    """Read a file and close it.  Returns the file source."""
    fobj = open(filename,'r');
    source = fobj.read();
    fobj.close()
    return source

def mimetype(self):
        """MIME type of the asset."""
        return (self.environment.mimetypes.get(self.format_extension) or
                self.compiler_mimetype or 'application/octet-stream')

def socket_close(self):
        """Close our socket."""
        if self.sock != NC.INVALID_SOCKET:
            self.sock.close()
        self.sock = NC.INVALID_SOCKET

def ts_func(f):
    """
    This wraps a function that would normally only accept an array
    and allows it to operate on a DataFrame. Useful for applying
    numpy functions to DataFrames.
    """
    def wrap_func(df, *args):
        # TODO: should vectorize to apply over all columns?
        return Chromatogram(f(df.values, *args), df.index, df.columns)
    return wrap_func

def mag(z):
    """Get the magnitude of a vector."""
    if isinstance(z[0], np.ndarray):
        return np.array(list(map(np.linalg.norm, z)))
    else:
        return np.linalg.norm(z)

def apply(f, obj, *args, **kwargs):
    """Apply a function in parallel to each element of the input"""
    return vectorize(f)(obj, *args, **kwargs)

def generate_hash(self, length=30):
        """ Generate random string of given length """
        import random, string
        chars = string.ascii_letters + string.digits
        ran = random.SystemRandom().choice
        hash = ''.join(ran(chars) for i in range(length))
        return hash

def apply(f, obj, *args, **kwargs):
    """Apply a function in parallel to each element of the input"""
    return vectorize(f)(obj, *args, **kwargs)

def angle_between_vectors(x, y):
    """ Compute the angle between vector x and y """
    dp = dot_product(x, y)
    if dp == 0:
        return 0
    xm = magnitude(x)
    ym = magnitude(y)
    return math.acos(dp / (xm*ym)) * (180. / math.pi)

def transform(self, df):
        """
        Transforms a DataFrame in place. Computes all outputs of the DataFrame.

        Args:
            df (pandas.DataFrame): DataFrame to transform.
        """
        for name, function in self.outputs:
            df[name] = function(df)

def has_multiline_items(maybe_list: Optional[Sequence[str]]):
    """Check whether one of the items in the list has multiple lines."""
    return maybe_list and any(is_multiline(item) for item in maybe_list)

def apply_conditional_styles(self, cbfct):
        """
        Ability to provide dynamic styling of the cell based on its value.
        :param cbfct: function(cell_value) should return a dict of format commands to apply to that cell
        :return: self
        """
        for ridx in range(self.nrows):
            for cidx in range(self.ncols):
                fmts = cbfct(self.actual_values.iloc[ridx, cidx])
                fmts and self.iloc[ridx, cidx].apply_styles(fmts)
        return self

def do_exit(self, arg):
        """Exit the shell session."""

        if self.current:
            self.current.close()
        self.resource_manager.close()
        del self.resource_manager
        return True

def SegmentMin(a, ids):
    """
    Segmented min op.
    """
    func = lambda idxs: np.amin(a[idxs], axis=0)
    return seg_map(func, a, ids),

def random_id(length):
    """Generates a random ID of given length"""

    def char():
        """Generate single random char"""

        return random.choice(string.ascii_letters + string.digits)

    return "".join(char() for _ in range(length))

def maxId(self):
        """int: current max id of objects"""
        if len(self.model.db) == 0:
            return 0

        return max(map(lambda obj: obj["id"], self.model.db))

def _join(verb):
    """
    Join helper
    """
    data = pd.merge(verb.x, verb.y, **verb.kwargs)

    # Preserve x groups
    if isinstance(verb.x, GroupedDataFrame):
        data.plydata_groups = list(verb.x.plydata_groups)
    return data

def locked_delete(self):
        """Delete credentials from the SQLAlchemy datastore."""
        filters = {self.key_name: self.key_value}
        self.session.query(self.model_class).filter_by(**filters).delete()

def rpc_fix_code(self, source, directory):
        """Formats Python code to conform to the PEP 8 style guide.

        """
        source = get_source(source)
        return fix_code(source, directory)

def get_average_length_of_string(strings):
    """Computes average length of words

    :param strings: list of words
    :return: Average length of word on list
    """
    if not strings:
        return 0

    return sum(len(word) for word in strings) / len(strings)

def listified_tokenizer(source):
    """Tokenizes *source* and returns the tokens as a list of lists."""
    io_obj = io.StringIO(source)
    return [list(a) for a in tokenize.generate_tokens(io_obj.readline)]

def mean(inlist):
    """
Returns the arithematic mean of the values in the passed list.
Assumes a '1D' list, but will function on the 1st dim of an array(!).

Usage:   lmean(inlist)
"""
    sum = 0
    for item in inlist:
        sum = sum + item
    return sum / float(len(inlist))

def b(s):
	""" Encodes Unicode strings to byte strings, if necessary. """

	return s if isinstance(s, bytes) else s.encode(locale.getpreferredencoding())

def create_alias(self):
        """Create lambda alias with env name and points it to $LATEST."""
        LOG.info('Creating alias %s', self.env)

        try:
            self.lambda_client.create_alias(
                FunctionName=self.app_name,
                Name=self.env,
                FunctionVersion='$LATEST',
                Description='Alias for {}'.format(self.env))
        except boto3.exceptions.botocore.exceptions.ClientError as error:
            LOG.debug('Create alias error: %s', error)
            LOG.info("Alias creation failed. Retrying...")
            raise

def alter_change_column(self, table, column, field):
        """Support change columns."""
        return self._update_column(table, column, lambda a, b: b)

def create_aws_lambda(ctx, bucket, region_name, aws_access_key_id, aws_secret_access_key):
    """Creates an AWS Chalice project for deployment to AWS Lambda."""
    from canari.commands.create_aws_lambda import create_aws_lambda
    create_aws_lambda(ctx.project, bucket, region_name, aws_access_key_id, aws_secret_access_key)

def rgba_bytes_tuple(self, x):
        """Provides the color corresponding to value `x` in the
        form of a tuple (R,G,B,A) with int values between 0 and 255.
        """
        return tuple(int(u*255.9999) for u in self.rgba_floats_tuple(x))

def list_rds(region, filter_by_kwargs):
    """List all RDS thingys."""
    conn = boto.rds.connect_to_region(region)
    instances = conn.get_all_dbinstances()
    return lookup(instances, filter_by=filter_by_kwargs)

def cmyk(c, m, y, k):
    """
    Create a spectra.Color object in the CMYK color space.

    :param float c: c coordinate.
    :param float m: m coordinate.
    :param float y: y coordinate.
    :param float k: k coordinate.

    :rtype: Color
    :returns: A spectra.Color object in the CMYK color space.
    """
    return Color("cmyk", c, m, y, k)

def extract_alzip (archive, compression, cmd, verbosity, interactive, outdir):
    """Extract a ALZIP archive."""
    return [cmd, '-d', outdir, archive]

def html(header_rows):
    """
    Convert a list of tuples describing a table into a HTML string
    """
    name = 'table%d' % next(tablecounter)
    return HtmlTable([map(str, row) for row in header_rows], name).render()

def bytes_base64(x):
    """Turn bytes into base64"""
    if six.PY2:
        return base64.encodestring(x).replace('\n', '')
    return base64.encodebytes(bytes_encode(x)).replace(b'\n', b'')

def _merge_meta(model1, model2):
    """Simple merge of samplesets."""
    w1 = _get_meta(model1)
    w2 = _get_meta(model2)
    return metadata.merge(w1, w2, metadata_conflicts='silent')

def _updateItemComboBoxIndex(self, item, column, num):
        """Callback for comboboxes: notifies us that a combobox for the given item and column has changed"""
        item._combobox_current_index[column] = num
        item._combobox_current_value[column] = item._combobox_option_list[column][num][0]

def multi_pop(d, *args):
    """ pops multiple keys off a dict like object """
    retval = {}
    for key in args:
        if key in d:
            retval[key] = d.pop(key)
    return retval

def onchange(self, value):
        """Called when a new DropDownItem gets selected.
        """
        log.debug('combo box. selected %s' % value)
        self.select_by_value(value)
        return (value, )

def hidden_cursor(self):
        """Return a context manager that hides the cursor while inside it and
        makes it visible on leaving."""
        self.stream.write(self.hide_cursor)
        try:
            yield
        finally:
            self.stream.write(self.normal_cursor)

def clean_whitespace(string, compact=False):
    """Return string with compressed whitespace."""
    for a, b in (('\r\n', '\n'), ('\r', '\n'), ('\n\n', '\n'),
                 ('\t', ' '), ('  ', ' ')):
        string = string.replace(a, b)
    if compact:
        for a, b in (('\n', ' '), ('[ ', '['),
                     ('  ', ' '), ('  ', ' '), ('  ', ' ')):
            string = string.replace(a, b)
    return string.strip()

def cpp_checker(code, working_directory):
    """Return checker."""
    return gcc_checker(code, '.cpp',
                       [os.getenv('CXX', 'g++'), '-std=c++0x'] + INCLUDE_FLAGS,
                       working_directory=working_directory)

def clean_whitespace(string, compact=False):
    """Return string with compressed whitespace."""
    for a, b in (('\r\n', '\n'), ('\r', '\n'), ('\n\n', '\n'),
                 ('\t', ' '), ('  ', ' ')):
        string = string.replace(a, b)
    if compact:
        for a, b in (('\n', ' '), ('[ ', '['),
                     ('  ', ' '), ('  ', ' '), ('  ', ' ')):
            string = string.replace(a, b)
    return string.strip()

def ver_to_tuple(value):
    """
    Convert version like string to a tuple of integers.
    """
    return tuple(int(_f) for _f in re.split(r'\D+', value) if _f)

def vector_distance(a, b):
    """The Euclidean distance between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

def getBuffer(x):
    """
    Copy @x into a (modifiable) ctypes byte array
    """
    b = bytes(x)
    return (c_ubyte * len(b)).from_buffer_copy(bytes(x))

def compose_all(tups):
  """Compose all given tuples together."""
  from . import ast  # I weep for humanity
  return functools.reduce(lambda x, y: x.compose(y), map(ast.make_tuple, tups), ast.make_tuple({}))

def _from_bytes(bytes, byteorder="big", signed=False):
    """This is the same functionality as ``int.from_bytes`` in python 3"""
    return int.from_bytes(bytes, byteorder=byteorder, signed=signed)

def config_parser_to_dict(config_parser):
    """
    Convert a ConfigParser to a dictionary.
    """
    response = {}

    for section in config_parser.sections():
        for option in config_parser.options(section):
            response.setdefault(section, {})[option] = config_parser.get(section, option)

    return response

def be_array_from_bytes(fmt, data):
    """
    Reads an array from bytestring with big-endian data.
    """
    arr = array.array(str(fmt), data)
    return fix_byteorder(arr)

def list_string_to_dict(string):
    """Inputs ``['a', 'b', 'c']``, returns ``{'a': 0, 'b': 1, 'c': 2}``."""
    dictionary = {}
    for idx, c in enumerate(string):
        dictionary.update({c: idx})
    return dictionary

def dot(self, w):
        """Return the dotproduct between self and another vector."""

        return sum([x * y for x, y in zip(self, w)])

def fromiterable(cls, itr):
        """Initialize from iterable"""
        x, y, z = itr
        return cls(x, y, z)

def dot(self, w):
        """Return the dotproduct between self and another vector."""

        return sum([x * y for x, y in zip(self, w)])

def time2seconds(t):
    """Returns seconds since 0h00."""
    return t.hour * 3600 + t.minute * 60 + t.second + float(t.microsecond) / 1e6

def ln_norm(x, mu, sigma=1.0):
    """ Natural log of scipy norm function truncated at zero """
    return np.log(stats.norm(loc=mu, scale=sigma).pdf(x))

def string_to_float_list(string_var):
        """Pull comma separated string values out of a text file and converts them to float list"""
        try:
            return [float(s) for s in string_var.strip('[').strip(']').split(', ')]
        except:
            return [float(s) for s in string_var.strip('[').strip(']').split(',')]

def md5_string(s):
    """
    Shortcut to create md5 hash
    :param s:
    :return:
    """
    m = hashlib.md5()
    m.update(s)
    return str(m.hexdigest())

def convolve_gaussian_2d(image, gaussian_kernel_1d):
    """Convolve 2d gaussian."""
    result = scipy.ndimage.filters.correlate1d(
        image, gaussian_kernel_1d, axis=0)
    result = scipy.ndimage.filters.correlate1d(
        result, gaussian_kernel_1d, axis=1)
    return result

def n_choose_k(n, k):
    """ get the number of quartets as n-choose-k. This is used
    in equal splits to decide whether a split should be exhaustively sampled
    or randomly sampled. Edges near tips can be exhaustive while highly nested
    edges probably have too many quartets
    """
    return int(reduce(MUL, (Fraction(n-i, i+1) for i in range(k)), 1))

def __copy__(self):
        """A magic method to implement shallow copy behavior."""
        return self.__class__.load(self.dump(), context=self.context)

def stddev(values, meanval=None):  #from AI: A Modern Appproach
    """The standard deviation of a set of values.
    Pass in the mean if you already know it."""
    if meanval == None: meanval = mean(values)
    return math.sqrt( sum([(x - meanval)**2 for x in values]) / (len(values)-1) )

def normalize(self, dt, is_dst=False):
        """Correct the timezone information on the given datetime"""
        if dt.tzinfo is self:
            return dt
        if dt.tzinfo is None:
            raise ValueError('Naive time - no tzinfo set')
        return dt.astimezone(self)

def covariance(self,pt0,pt1):
        """ get the covarince between two points implied by Vario2d

        Parameters
        ----------
        pt0 : (iterable of len 2)
            first point x and y
        pt1 : (iterable of len 2)
            second point x and y

        Returns
        -------
        cov : float
            covariance between pt0 and pt1

        """

        x = np.array([pt0[0],pt1[0]])
        y = np.array([pt0[1],pt1[1]])
        names = ["n1","n2"]
        return self.covariance_matrix(x,y,names=names).x[0,1]

def inverse(self):
        """
        Returns inverse of transformation.
        """
        invr = np.linalg.inv(self.affine_matrix)
        return SymmOp(invr)

def covariance(self,pt0,pt1):
        """ get the covarince between two points implied by Vario2d

        Parameters
        ----------
        pt0 : (iterable of len 2)
            first point x and y
        pt1 : (iterable of len 2)
            second point x and y

        Returns
        -------
        cov : float
            covariance between pt0 and pt1

        """

        x = np.array([pt0[0],pt1[0]])
        y = np.array([pt0[1],pt1[1]])
        names = ["n1","n2"]
        return self.covariance_matrix(x,y,names=names).x[0,1]

def fail(message=None, exit_status=None):
    """Prints the specified message and exits the program with the specified
    exit status.

    """
    print('Error:', message, file=sys.stderr)
    sys.exit(exit_status or 1)

def build_parser():
    """Build the script's argument parser."""

    parser = argparse.ArgumentParser(description="The IOTile task supervisor")
    parser.add_argument('-c', '--config', help="config json with options")
    parser.add_argument('-v', '--verbose', action="count", default=0, help="Increase logging verbosity")

    return parser

def extract_zip(zip_path, target_folder):
    """
    Extract the content of the zip-file at `zip_path` into `target_folder`.
    """
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target_folder)

def get_dt_list(fn_list):
    """Get list of datetime objects, extracted from a filename
    """
    dt_list = np.array([fn_getdatetime(fn) for fn in fn_list])
    return dt_list

def ExecuteRaw(self, position, command):
    """Send a command string to gdb."""
    self.EnsureGdbPosition(position[0], None, None)
    return gdb.execute(command, to_string=True)

def subscribe_to_quorum_channel(self):
        """In case the experiment enforces a quorum, listen for notifications
        before creating Partipant objects.
        """
        from dallinger.experiment_server.sockets import chat_backend

        self.log("Bot subscribing to quorum channel.")
        chat_backend.subscribe(self, "quorum")

def print_latex(o):
    """A function to generate the latex representation of sympy
    expressions."""
    if can_print_latex(o):
        s = latex(o, mode='plain')
        s = s.replace('\\dag','\\dagger')
        s = s.strip('$')
        return '$$%s$$' % s
    # Fallback to the string printer
    return None

def _string_hash(s):
    """String hash (djb2) with consistency between py2/py3 and persistency between runs (unlike `hash`)."""
    h = 5381
    for c in s:
        h = h * 33 + ord(c)
    return h

def add_to_js(self, name, var):
        """Add an object to Javascript."""
        frame = self.page().mainFrame()
        frame.addToJavaScriptWindowObject(name, var)

def force_iterable(f):
    """Will make any functions return an iterable objects by wrapping its result in a list."""
    def wrapper(*args, **kwargs):
        r = f(*args, **kwargs)
        if hasattr(r, '__iter__'):
            return r
        else:
            return [r]
    return wrapper

def pause(self):
        """Pause the music"""
        mixer.music.pause()
        self.pause_time = self.get_time()
        self.paused = True

def trans_from_matrix(matrix):
    """ Convert a vtk matrix to a numpy.ndarray """
    t = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            t[i, j] = matrix.GetElement(i, j)
    return t

def get_mod_time(self, path):
        """
        Returns a datetime object representing the last time the file was modified

        :param path: remote file path
        :type path: string
        """
        conn = self.get_conn()
        ftp_mdtm = conn.sendcmd('MDTM ' + path)
        time_val = ftp_mdtm[4:]
        # time_val optionally has microseconds
        try:
            return datetime.datetime.strptime(time_val, "%Y%m%d%H%M%S.%f")
        except ValueError:
            return datetime.datetime.strptime(time_val, '%Y%m%d%H%M%S')

def to_dataframe(products):
        """Return the products from a query response as a Pandas DataFrame
        with the values in their appropriate Python types.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("to_dataframe requires the optional dependency Pandas.")

        return pd.DataFrame.from_dict(products, orient='index')

def restart_program():
    """
    DOES NOT WORK WELL WITH MOPIDY
    Hack from
    https://www.daniweb.com/software-development/python/code/260268/restart-your-python-program
    to support updating the settings, since mopidy is not able to do that yet
    Restarts the current program
    Note: this function does not return. Any cleanup action (like
    saving data) must be done before calling this function
    """

    python = sys.executable
    os.execl(python, python, * sys.argv)

def set_as_object(self, value):
        """
        Sets a new value to map element

        :param value: a new element or map value.
        """
        self.clear()
        map = MapConverter.to_map(value)
        self.append(map)

def construct_from_string(cls, string):
        """
        Construction from a string, raise a TypeError if not
        possible
        """
        if string == cls.name:
            return cls()
        raise TypeError("Cannot construct a '{}' from "
                        "'{}'".format(cls, string))

def _update_font_style(self, font_style):
        """Updates font style widget

        Parameters
        ----------

        font_style: Integer
        \tButton down iif font_style == wx.FONTSTYLE_ITALIC

        """

        toggle_state = font_style & wx.FONTSTYLE_ITALIC == wx.FONTSTYLE_ITALIC

        self.ToggleTool(wx.FONTFLAG_ITALIC, toggle_state)

def string_to_list(string, sep=",", filter_empty=False):
    """Transforma una string con elementos separados por `sep` en una lista."""
    return [value.strip() for value in string.split(sep)
            if (not filter_empty or value)]

def _histplot_op(ax, data, **kwargs):
    """Add a histogram for the data to the axes."""
    bins = get_bins(data)
    ax.hist(data, bins=bins, align="left", density=True, **kwargs)
    return ax

def _single_page_pdf(page):
    """Construct a single page PDF from the provided page in memory"""
    pdf = Pdf.new()
    pdf.pages.append(page)
    bio = BytesIO()
    pdf.save(bio)
    bio.seek(0)
    return bio.read()

def norm_vec(vector):
    """Normalize the length of a vector to one"""
    assert len(vector) == 3
    v = np.array(vector)
    return v/np.sqrt(np.sum(v**2))

def from_json_str(cls, json_str):
    """Convert json string representation into class instance.

    Args:
      json_str: json representation as string.

    Returns:
      New instance of the class with data loaded from json string.
    """
    return cls.from_json(json.loads(json_str, cls=JsonDecoder))

def start():
    """Starts the web server."""
    global app
    bottle.run(app, host=conf.WebHost, port=conf.WebPort,
               debug=conf.WebAutoReload, reloader=conf.WebAutoReload,
               quiet=conf.WebQuiet)

def from_bytes(cls, b):
		"""Create :class:`PNG` from raw bytes.
		
		:arg bytes b: The raw bytes of the PNG file.
		:rtype: :class:`PNG`
		"""
		im = cls()
		im.chunks = list(parse_chunks(b))
		im.init()
		return im

def start():
    """Starts the web server."""
    global app
    bottle.run(app, host=conf.WebHost, port=conf.WebPort,
               debug=conf.WebAutoReload, reloader=conf.WebAutoReload,
               quiet=conf.WebQuiet)

def construct_from_string(cls, string):
        """
        Construction from a string, raise a TypeError if not
        possible
        """
        if string == cls.name:
            return cls()
        raise TypeError("Cannot construct a '{}' from "
                        "'{}'".format(cls, string))

def safe_setattr(obj, name, value):
    """Attempt to setattr but catch AttributeErrors."""
    try:
        setattr(obj, name, value)
        return True
    except AttributeError:
        return False

def add_matplotlib_cmap(cm, name=None):
    """Add a matplotlib colormap."""
    global cmaps
    cmap = matplotlib_to_ginga_cmap(cm, name=name)
    cmaps[cmap.name] = cmap

def camel_to_(s):
    """
    Convert CamelCase to camel_case
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def _tab(content):
    """
    Helper funcation that converts text-based get response
    to tab separated values for additional manipulation.
    """
    response = _data_frame(content).to_csv(index=False,sep='\t')
    return response

def decamelise(text):
    """Convert CamelCase to lower_and_underscore."""
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()

def cint32_array_to_numpy(cptr, length):
    """Convert a ctypes int pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_int32)):
        return np.fromiter(cptr, dtype=np.int32, count=length)
    else:
        raise RuntimeError('Expected int pointer')

def cartesian_product(arrays, flat=True, copy=False):
    """
    Efficient cartesian product of a list of 1D arrays returning the
    expanded array views for each dimensions. By default arrays are
    flattened, which may be controlled with the flat flag. The array
    views can be turned into regular arrays with the copy flag.
    """
    arrays = np.broadcast_arrays(*np.ix_(*arrays))
    if flat:
        return tuple(arr.flatten() if copy else arr.flat for arr in arrays)
    return tuple(arr.copy() if copy else arr for arr in arrays)

def cfloat64_array_to_numpy(cptr, length):
    """Convert a ctypes double pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_double)):
        return np.fromiter(cptr, dtype=np.float64, count=length)
    else:
        raise RuntimeError('Expected double pointer')

def to_snake_case(name):
    """ Given a name in camelCase return in snake_case """
    s1 = FIRST_CAP_REGEX.sub(r'\1_\2', name)
    return ALL_CAP_REGEX.sub(r'\1_\2', s1).lower()

def cleanup_lib(self):
        """ unload the previously loaded shared library """
        if not self.using_openmp:
            #this if statement is necessary because shared libraries that use
            #OpenMP will core dump when unloaded, this is a well-known issue with OpenMP
            logging.debug('unloading shared library')
            _ctypes.dlclose(self.lib._handle)

def _convert_to_array(array_like, dtype):
        """
        Convert Matrix attributes which are array-like or buffer to array.
        """
        if isinstance(array_like, bytes):
            return np.frombuffer(array_like, dtype=dtype)
        return np.asarray(array_like, dtype=dtype)

def pointer(self):
        """Get a ctypes void pointer to the memory mapped region.

        :type: ctypes.c_void_p
        """
        return ctypes.cast(ctypes.pointer(ctypes.c_uint8.from_buffer(self.mapping, 0)), ctypes.c_void_p)

def shape_list(l,shape,dtype):
    """ Shape a list of lists into the appropriate shape and data type """
    return np.array(l, dtype=dtype).reshape(shape)

def pointer(self):
        """Get a ctypes void pointer to the memory mapped region.

        :type: ctypes.c_void_p
        """
        return ctypes.cast(ctypes.pointer(ctypes.c_uint8.from_buffer(self.mapping, 0)), ctypes.c_void_p)

def expect_comment_end(self):
        """Expect a comment end and return the match object.
        """
        match = self._expect_match('#}', COMMENT_END_PATTERN)
        self.advance(match.end())

def round_corner(radius, fill):
    """Draw a round corner"""
    corner = Image.new('L', (radius, radius), 0)  # (0, 0, 0, 0))
    draw = ImageDraw.Draw(corner)
    draw.pieslice((0, 0, radius * 2, radius * 2), 180, 270, fill=fill)
    return corner

def struct2dict(struct):
    """convert a ctypes structure to a dictionary"""
    return {x: getattr(struct, x) for x in dict(struct._fields_).keys()}

def SetValue(self, row, col, value):
        """
        Set value in the pandas DataFrame
        """
        self.dataframe.iloc[row, col] = value

def to_identifier(s):
  """
  Convert snake_case to camel_case.
  """
  if s.startswith('GPS'):
      s = 'Gps' + s[3:]
  return ''.join([i.capitalize() for i in s.split('_')]) if '_' in s else s

def convert_time_string(date_str):
    """ Change a date string from the format 2018-08-15T23:55:17 into a datetime object """
    dt, _, _ = date_str.partition(".")
    dt = datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S")
    return dt

def strip_columns(tab):
    """Strip whitespace from string columns."""
    for colname in tab.colnames:
        if tab[colname].dtype.kind in ['S', 'U']:
            tab[colname] = np.core.defchararray.strip(tab[colname])

def AmericanDateToEpoch(self, date_str):
    """Take a US format date and return epoch."""
    try:
      epoch = time.strptime(date_str, "%m/%d/%Y")
      return int(calendar.timegm(epoch)) * 1000000
    except ValueError:
      return 0

def load_jsonf(fpath, encoding):
    """
    :param unicode fpath:
    :param unicode encoding:
    :rtype: dict | list
    """
    with codecs.open(fpath, encoding=encoding) as f:
        return json.load(f)

def iso_to_datetime(date):
    """ Convert ISO 8601 time format to datetime format

    This function converts a date in ISO format, e.g. ``2017-09-14`` to a `datetime` instance, e.g.
    ``datetime.datetime(2017,9,14,0,0)``

    :param date: date in ISO 8601 format
    :type date: str
    :return: datetime instance
    :rtype: datetime
    """
    chunks = list(map(int, date.split('T')[0].split('-')))
    return datetime.datetime(chunks[0], chunks[1], chunks[2])

def __repr__(self):
    """Returns a stringified representation of this object."""
    return str({'name': self._name, 'watts': self._watts,
                'type': self._output_type, 'id': self._integration_id})

def weekly(date=datetime.date.today()):
    """
    Weeks start are fixes at Monday for now.
    """
    return date - datetime.timedelta(days=date.weekday())

def classnameify(s):
  """
  Makes a classname
  """
  return ''.join(w if w in ACRONYMS else w.title() for w in s.split('_'))

def now(self):
		"""
		Return a :py:class:`datetime.datetime` instance representing the current time.

		:rtype: :py:class:`datetime.datetime`
		"""
		if self.use_utc:
			return datetime.datetime.utcnow()
		else:
			return datetime.datetime.now()

def _idx_col2rowm(d):
    """Generate indexes to change from col-major to row-major ordering"""
    if 0 == len(d):
        return 1
    if 1 == len(d):
        return np.arange(d[0])
    # order='F' indicates column-major ordering
    idx = np.array(np.arange(np.prod(d))).reshape(d, order='F').T
    return idx.flatten(order='F')

def this_week():
        """ Return start and end date of the current week. """
        since = TODAY + delta(weekday=MONDAY(-1))
        until = since + delta(weeks=1)
        return Date(since), Date(until)

def setHSV(self, pixel, hsv):
        """Set single pixel to HSV tuple"""
        color = conversions.hsv2rgb(hsv)
        self._set_base(pixel, color)

def parse_timestamp(timestamp):
    """Parse ISO8601 timestamps given by github API."""
    dt = dateutil.parser.parse(timestamp)
    return dt.astimezone(dateutil.tz.tzutc())

def tick(self):
        """Add one tick to progress bar"""
        self.current += 1
        if self.current == self.factor:
            sys.stdout.write('+')
            sys.stdout.flush()
            self.current = 0

def datetime_to_timezone(date, tz="UTC"):
    """ convert naive datetime to timezone-aware datetime """
    if not date.tzinfo:
        date = date.replace(tzinfo=timezone(get_timezone()))
    return date.astimezone(timezone(tz))

def input(self, prompt, default=None, show_default=True):
        """Provide a command prompt."""
        return click.prompt(prompt, default=default, show_default=show_default)

def get_naive(dt):
  """Gets a naive datetime from a datetime.

  datetime_tz objects can't just have tzinfo replaced with None, you need to
  call asdatetime.

  Args:
    dt: datetime object.

  Returns:
    datetime object without any timezone information.
  """
  if not dt.tzinfo:
    return dt
  if hasattr(dt, "asdatetime"):
    return dt.asdatetime()
  return dt.replace(tzinfo=None)

def activate_subplot(numPlot):
    """Make subplot *numPlot* active on the canvas.

    Use this if a simple ``subplot(numRows, numCols, numPlot)``
    overwrites the subplot instead of activating it.
    """
    # see http://www.mail-archive.com/matplotlib-users@lists.sourceforge.net/msg07156.html
    from pylab import gcf, axes
    numPlot -= 1  # index is 0-based, plots are 1-based
    return axes(gcf().get_axes()[numPlot])

def parse_timestamp(timestamp):
    """Parse ISO8601 timestamps given by github API."""
    dt = dateutil.parser.parse(timestamp)
    return dt.astimezone(dateutil.tz.tzutc())

def inc_date(date_obj, num, date_fmt):
    """Increment the date by a certain number and return date object.
    as the specific string format.
    """
    return (date_obj + timedelta(days=num)).strftime(date_fmt)

def last_month():
        """ Return start and end date of this month. """
        since = TODAY + delta(day=1, months=-1)
        until = since + delta(months=1)
        return Date(since), Date(until)

def select_down(self):
        """move cursor down"""
        r, c = self._index
        self._select_index(r+1, c)

def QA_util_datetime_to_strdate(dt):
    """
    :param dt:  pythone datetime.datetime
    :return:  1999-02-01 string type
    """
    strdate = "%04d-%02d-%02d" % (dt.year, dt.month, dt.day)
    return strdate

def round_sig(x, sig):
    """Round the number to the specified number of significant figures"""
    return round(x, sig - int(floor(log10(abs(x)))) - 1)

def robust_int(v):
    """Parse an int robustly, ignoring commas and other cruft. """

    if isinstance(v, int):
        return v

    if isinstance(v, float):
        return int(v)

    v = str(v).replace(',', '')

    if not v:
        return None

    return int(v)

def fromtimestamp(cls, timestamp):
    """Returns a datetime object of a given timestamp (in local tz)."""
    d = cls.utcfromtimestamp(timestamp)
    return d.astimezone(localtz())

def to_str(s):
    """
    Convert bytes and non-string into Python 3 str
    """
    if isinstance(s, bytes):
        s = s.decode('utf-8')
    elif not isinstance(s, str):
        s = str(s)
    return s

def today(year=None):
    """this day, last year"""
    return datetime.date(int(year), _date.month, _date.day) if year else _date

def log_y_cb(self, w, val):
        """Toggle linear/log scale for Y-axis."""
        self.tab_plot.logy = val
        self.plot_two_columns()

def energy_string_to_float( string ):
    """
    Convert a string of a calculation energy, e.g. '-1.2345 eV' to a float.

    Args:
        string (str): The string to convert.
  
    Return
        (float) 
    """
    energy_re = re.compile( "(-?\d+\.\d+)" )
    return float( energy_re.match( string ).group(0) )

def append_pdf(input_pdf: bytes, output_writer: PdfFileWriter):
    """
    Appends a PDF to a pyPDF writer. Legacy interface.
    """
    append_memory_pdf_to_writer(input_pdf=input_pdf,
                                writer=output_writer)

def add_exec_permission_to(target_file):
    """Add executable permissions to the file

    :param target_file: the target file whose permission to be changed
    """
    mode = os.stat(target_file).st_mode
    os.chmod(target_file, mode | stat.S_IXUSR)

def _iterable_to_varargs_method(func):
    """decorator to convert a method taking a iterable to a *args one"""
    def wrapped(self, *args, **kwargs):
        return func(self, args, **kwargs)
    return wrapped

def auto():
	"""set colouring on if STDOUT is a terminal device, off otherwise"""
	try:
		Style.enabled = False
		Style.enabled = sys.stdout.isatty()
	except (AttributeError, TypeError):
		pass

def set_default(self, key, value):
        """Set the default value for this key.
        Default only used when no value is provided by the user via
        arg, config or env.
        """
        k = self._real_key(key.lower())
        self._defaults[k] = value

def is_int_type(val):
    """Return True if `val` is of integer type."""
    try:               # Python 2
        return isinstance(val, (int, long))
    except NameError:  # Python 3
        return isinstance(val, int)

def empty(self, start=None, stop=None):
		"""Empty the range from start to stop.

		Like delete, but no Error is raised if the entire range isn't mapped.
		"""
		self.set(NOT_SET, start=start, stop=stop)

def matrix_at_check(self, original, loc, tokens):
        """Check for Python 3.5 matrix multiplication."""
        return self.check_py("35", "matrix multiplication", original, loc, tokens)

def show_xticklabels(self, row, column):
        """Show the x-axis tick labels for a subplot.

        :param row,column: specify the subplot.

        """
        subplot = self.get_subplot_at(row, column)
        subplot.show_xticklabels()

def is_timestamp(obj):
    """
    Yaml either have automatically converted it to a datetime object
    or it is a string that will be validated later.
    """
    return isinstance(obj, datetime.datetime) or is_string(obj) or is_int(obj) or is_float(obj)

def clear_global(self):
        """Clear only any cached global data.

        """
        vname = self.varname
        logger.debug(f'global clearning {vname}')
        if vname in globals():
            logger.debug('removing global instance var: {}'.format(vname))
            del globals()[vname]

def is_date(thing):
    """Checks if the given thing represents a date

    :param thing: The object to check if it is a date
    :type thing: arbitrary object
    :returns: True if we have a date object
    :rtype: bool
    """
    # known date types
    date_types = (datetime.datetime,
                  datetime.date,
                  DateTime)
    return isinstance(thing, date_types)

def seconds(num):
    """
    Pause for this many seconds
    """
    now = pytime.time()
    end = now + num
    until(end)

def _get_mtime():
    """
    Get the modified time of the RPM Database.

    Returns:
        Unix ticks
    """
    return os.path.exists(RPM_PATH) and int(os.path.getmtime(RPM_PATH)) or 0

def del_object_from_parent(self):
        """ Delete object from parent object. """
        if self.parent:
            self.parent.objects.pop(self.ref)

def close(*args, **kwargs):
    r"""Close last created figure, alias to ``plt.close()``."""
    _, plt, _ = _import_plt()
    plt.close(*args, **kwargs)

def up(self):
        """Go up in stack and return True if top frame"""
        if self.frame:
            self.frame = self.frame.f_back
            return self.frame is None

def safe_rmtree(directory):
  """Delete a directory if it's present. If it's not present, no-op."""
  if os.path.exists(directory):
    shutil.rmtree(directory, True)

def is_square_matrix(mat):
    """Test if an array is a square matrix."""
    mat = np.array(mat)
    if mat.ndim != 2:
        return False
    shape = mat.shape
    return shape[0] == shape[1]

def invalidate_cache(cpu, address, size):
        """ remove decoded instruction from instruction cache """
        cache = cpu.instruction_cache
        for offset in range(size):
            if address + offset in cache:
                del cache[address + offset]

def _is_one_arg_pos_call(call):
    """Is this a call with exactly 1 argument,
    where that argument is positional?
    """
    return isinstance(call, astroid.Call) and len(call.args) == 1 and not call.keywords

def __exit__(self, *args):
        """Redirect stdout back to the original stdout."""
        sys.stdout = self._orig
        self._devnull.close()

def is_int_type(val):
    """Return True if `val` is of integer type."""
    try:               # Python 2
        return isinstance(val, (int, long))
    except NameError:  # Python 3
        return isinstance(val, int)

def _sanitize(text):
    """Return sanitized Eidos text field for human readability."""
    d = {'-LRB-': '(', '-RRB-': ')'}
    return re.sub('|'.join(d.keys()), lambda m: d[m.group(0)], text)

def _IsDirectory(parent, item):
  """Helper that returns if parent/item is a directory."""
  return tf.io.gfile.isdir(os.path.join(parent, item))

def from_json(cls, json_str):
        """Deserialize the object from a JSON string."""
        d = json.loads(json_str)
        return cls.from_dict(d)

def __contains__ (self, key):
        """Check lowercase key item."""
        assert isinstance(key, basestring)
        return dict.__contains__(self, key.lower())

def dispose_orm():
    """ Properly close pooled database connections """
    log.debug("Disposing DB connection pool (PID %s)", os.getpid())
    global engine
    global Session

    if Session:
        Session.remove()
        Session = None
    if engine:
        engine.dispose()
        engine = None

def is_iter_non_string(obj):
    """test if object is a list or tuple"""
    if isinstance(obj, list) or isinstance(obj, tuple):
        return True
    return False

def lock_file(f, block=False):
    """
    If block=False (the default), die hard and fast if another process has
    already grabbed the lock for this file.

    If block=True, wait for the lock to be released, then continue.
    """
    try:
        flags = fcntl.LOCK_EX
        if not block:
            flags |= fcntl.LOCK_NB
        fcntl.flock(f.fileno(), flags)
    except IOError as e:
        if e.errno in (errno.EACCES, errno.EAGAIN):
            raise SystemExit("ERROR: %s is locked by another process." %
                             f.name)
        raise

def is_valid_row(cls, row):
        """Indicates whether or not the given row contains valid data."""
        for k in row.keys():
            if row[k] is None:
                return False
        return True

def stdin_readable():
    """Determine whether stdin has any data to read."""
    if not WINDOWS:
        try:
            return bool(select([sys.stdin], [], [], 0)[0])
        except Exception:
            logger.log_exc()
    try:
        return not sys.stdin.isatty()
    except Exception:
        logger.log_exc()
    return False

def is_valid_url(url):
    """Checks if a given string is an url"""
    pieces = urlparse(url)
    return all([pieces.scheme, pieces.netloc])

def empty_tree(input_list):
    """Recursively iterate through values in nested lists."""
    for item in input_list:
        if not isinstance(item, list) or not empty_tree(item):
            return False
    return True

def _valid_table_name(name):
    """Verify if a given table name is valid for `rows`

    Rules:
    - Should start with a letter or '_'
    - Letters can be capitalized or not
    - Accepts letters, numbers and _
    """

    if name[0] not in "_" + string.ascii_letters or not set(name).issubset(
        "_" + string.ascii_letters + string.digits
    ):
        return False

    else:
        return True

def _is_proper_sequence(seq):
    """Returns is seq is sequence and not string."""
    return (isinstance(seq, collections.abc.Sequence) and
            not isinstance(seq, str))

def fft_bandpassfilter(data, fs, lowcut, highcut):
    """
    http://www.swharden.com/blog/2009-01-21-signal-filtering-with-python/#comment-16801
    """
    fft = np.fft.fft(data)
    # n = len(data)
    # timestep = 1.0 / fs
    # freq = np.fft.fftfreq(n, d=timestep)
    bp = fft.copy()

    # Zero out fft coefficients
    # bp[10:-10] = 0

    # Normalise
    # bp *= real(fft.dot(fft))/real(bp.dot(bp))

    bp *= fft.dot(fft) / bp.dot(bp)

    # must multipy by 2 to get the correct amplitude
    ibp = 12 * np.fft.ifft(bp)
    return ibp

def has_common(self, other):
        """Return set of common words between two word sets."""
        if not isinstance(other, WordSet):
            raise ValueError('Can compare only WordSets')
        return self.term_set & other.term_set

def test_value(self, value):
        """Test if value is an instance of float."""
        if not isinstance(value, float):
            raise ValueError('expected float value: ' + str(type(value)))

def dictapply(d, fn):
    """
    apply a function to all non-dict values in a dictionary
    """
    for k, v in d.items():
        if isinstance(v, dict):
            v = dictapply(v, fn)
        else:
            d[k] = fn(v)
    return d

def count_(self):
        """
        Returns the number of rows of the main dataframe
        """
        try:
            num = len(self.df.index)
        except Exception as e:
            self.err(e, "Can not count data")
            return
        return num

def _remove_dict_keys_with_value(dict_, val):
  """Removes `dict` keys which have have `self` as value."""
  return {k: v for k, v in dict_.items() if v is not val}

def is_client(self):
        """Return True if Glances is running in client mode."""
        return (self.args.client or self.args.browser) and not self.args.server

def pretty_dict_str(d, indent=2):
    """shows JSON indented representation of d"""
    b = StringIO()
    write_pretty_dict_str(b, d, indent=indent)
    return b.getvalue()

def get_file_size(filename):
    """
    Get the file size of a given file

    :param filename: string: pathname of a file
    :return: human readable filesize
    """
    if os.path.isfile(filename):
        return convert_size(os.path.getsize(filename))
    return None

def multi_pop(d, *args):
    """ pops multiple keys off a dict like object """
    retval = {}
    for key in args:
        if key in d:
            retval[key] = d.pop(key)
    return retval

def safe_delete(filename):
  """Delete a file safely. If it's not present, no-op."""
  try:
    os.unlink(filename)
  except OSError as e:
    if e.errno != errno.ENOENT:
      raise

def purge_dict(idict):
    """Remove null items from a dictionary """
    odict = {}
    for key, val in idict.items():
        if is_null(val):
            continue
        odict[key] = val
    return odict

def timeit(self, metric, func, *args, **kwargs):
        """Time execution of callable and emit metric then return result."""
        return metrics.timeit(metric, func, *args, **kwargs)

def purge_dict(idict):
    """Remove null items from a dictionary """
    odict = {}
    for key, val in idict.items():
        if is_null(val):
            continue
        odict[key] = val
    return odict

def _call_retry(self, force_retry):
        """Call request and retry up to max_attempts times (or none if self.max_attempts=1)"""
        last_exception = None
        for i in range(self.max_attempts):
            try:
                log.info("Calling %s %s" % (self.method, self.url))
                response = self.requests_method(
                    self.url,
                    data=self.data,
                    params=self.params,
                    headers=self.headers,
                    timeout=(self.connect_timeout, self.read_timeout),
                    verify=self.verify_ssl,
                )

                if response is None:
                    log.warn("Got response None")
                    if self._method_is_safe_to_retry():
                        delay = 0.5 + i * 0.5
                        log.info("Waiting %s sec and Retrying since call is a %s" % (delay, self.method))
                        time.sleep(delay)
                        continue
                    else:
                        raise PyMacaronCoreException("Call %s %s returned empty response" % (self.method, self.url))

                return response

            except Exception as e:

                last_exception = e

                retry = force_retry

                if isinstance(e, ReadTimeout):
                    # Log enough to help debugging...
                    log.warn("Got a ReadTimeout calling %s %s" % (self.method, self.url))
                    log.warn("Exception was: %s" % str(e))
                    resp = e.response
                    if not resp:
                        log.info("Requests error has no response.")
                        # TODO: retry=True? Is it really safe?
                    else:
                        b = resp.content
                        log.info("Requests has a response with content: " + pprint.pformat(b))
                    if self._method_is_safe_to_retry():
                        # It is safe to retry
                        log.info("Retrying since call is a %s" % self.method)
                        retry = True

                elif isinstance(e, ConnectTimeout):
                    log.warn("Got a ConnectTimeout calling %s %s" % (self.method, self.url))
                    log.warn("Exception was: %s" % str(e))
                    # ConnectTimeouts are safe to retry whatever the call...
                    retry = True

                if retry:
                    continue
                else:
                    raise e

        # max_attempts has been reached: propagate the last received Exception
        if not last_exception:
            last_exception = Exception("Reached max-attempts (%s). Giving up calling %s %s" % (self.max_attempts, self.method, self.url))
        raise last_exception

def _file_exists(path, filename):
  """Checks if the filename exists under the path."""
  return os.path.isfile(os.path.join(path, filename))

def multidict_to_dict(d):
    """
    Turns a werkzeug.MultiDict or django.MultiValueDict into a dict with
    list values
    :param d: a MultiDict or MultiValueDict instance
    :return: a dict instance
    """
    return dict((k, v[0] if len(v) == 1 else v) for k, v in iterlists(d))

def _check_valid_key(self, key):
        """Checks if a key is valid and raises a ValueError if its not.

        When in need of checking a key for validity, always use this
        method if possible.

        :param key: The key to be checked
        """
        if not isinstance(key, key_type):
            raise ValueError('%r is not a valid key type' % key)
        if not VALID_KEY_RE.match(key):
            raise ValueError('%r contains illegal characters' % key)

def nonull_dict(self):
        """Like dict, but does not hold any null values.

        :return:

        """
        return {k: v for k, v in six.iteritems(self.dict) if v and k != '_codes'}

def _ws_on_close(self, ws: websocket.WebSocketApp):
        """Callback for closing the websocket connection

        Args:
            ws: websocket connection (now closed)
        """
        self.connected = False
        self.logger.error('Websocket closed')
        self._reconnect_websocket()

def to_bipartite_matrix(A):
    """Returns the adjacency matrix of a bipartite graph whose biadjacency
    matrix is `A`.

    `A` must be a NumPy array.

    If `A` has **m** rows and **n** columns, then the returned matrix has **m +
    n** rows and columns.

    """
    m, n = A.shape
    return four_blocks(zeros(m, m), A, A.T, zeros(n, n))

def is_identifier(string):
    """Check if string could be a valid python identifier

    :param string: string to be tested
    :returns: True if string can be a python identifier, False otherwise
    :rtype: bool
    """
    matched = PYTHON_IDENTIFIER_RE.match(string)
    return bool(matched) and not keyword.iskeyword(string)

def convert_string(string):
    """Convert string to int, float or bool.
    """
    if is_int(string):
        return int(string)
    elif is_float(string):
        return float(string)
    elif convert_bool(string)[0]:
        return convert_bool(string)[1]
    elif string == 'None':
        return None
    else:
        return string

def is_non_empty_string(input_string):
    """
    Validate if non empty string

    :param input_string: Input is a *str*.
    :return: True if input is string and non empty.
       Raise :exc:`Exception` otherwise.
    """
    try:
        if not input_string.strip():
            raise ValueError()
    except AttributeError as error:
        raise TypeError(error)

    return True

def get_serialize_format(self, mimetype):
		""" Get the serialization format for the given mimetype """
		format = self.formats.get(mimetype, None)
		if format is None:
			format = formats.get(mimetype, None)
		return format

def equal(obj1, obj2):
    """Calculate equality between two (Comparable) objects."""
    Comparable.log(obj1, obj2, '==')
    equality = obj1.equality(obj2)
    Comparable.log(obj1, obj2, '==', result=equality)
    return equality

def fast_distinct(self):
        """
        Because standard distinct used on the all fields are very slow and works only with PostgreSQL database
        this method provides alternative to the standard distinct method.
        :return: qs with unique objects
        """
        return self.model.objects.filter(pk__in=self.values_list('pk', flat=True))

def contained_in(filename, directory):
    """Test if a file is located within the given directory."""
    filename = os.path.normcase(os.path.abspath(filename))
    directory = os.path.normcase(os.path.abspath(directory))
    return os.path.commonprefix([filename, directory]) == directory

def parse_value(self, value):
        """Cast value to `bool`."""
        parsed = super(BoolField, self).parse_value(value)
        return bool(parsed) if parsed is not None else None

def run(self, value):
        """ Determines if value value is empty.
        Keyword arguments:
        value str -- the value of the associated field to compare
        """
        if self.pass_ and not value.strip():
            return True

        if not value:
            return False
        return True

def tree_render(request, upy_context, vars_dictionary):
    """
    It renders template defined in upy_context's page passed in arguments
    """
    page = upy_context['PAGE']
    return render_to_response(page.template.file_name, vars_dictionary, context_instance=RequestContext(request))

def is_punctuation(text):
    """Check if given string is a punctuation"""
    return not (text.lower() in config.AVRO_VOWELS or
                text.lower() in config.AVRO_CONSONANTS)

def static_urls_js():
    """
    Add global variables to JavaScript about the location and latest version of
    transpiled files.
    Usage::
        {% static_urls_js %}
    """
    if apps.is_installed('django.contrib.staticfiles'):
        from django.contrib.staticfiles.storage import staticfiles_storage
        static_base_url = staticfiles_storage.base_url
    else:
        static_base_url = PrefixNode.handle_simple("STATIC_URL")
    transpile_base_url = urljoin(static_base_url, 'js/transpile/')
    return {
        'static_base_url': static_base_url,
        'transpile_base_url': transpile_base_url,
        'version': LAST_RUN['version']
    }

def set_executable(filename):
    """Set the exectuable bit on the given filename"""
    st = os.stat(filename)
    os.chmod(filename, st.st_mode | stat.S_IEXEC)

def see_doc(obj_with_doc):
    """Copy docstring from existing object to the decorated callable."""
    def decorator(fn):
        fn.__doc__ = obj_with_doc.__doc__
        return fn
    return decorator

def check_clang_apply_replacements_binary(args):
  """Checks if invoking supplied clang-apply-replacements binary works."""
  try:
    subprocess.check_call([args.clang_apply_replacements_binary, '--version'])
  except:
    print('Unable to run clang-apply-replacements. Is clang-apply-replacements '
          'binary correctly specified?', file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

def close_all_but_this(self):
        """Close all files but the current one"""
        self.close_all_right()
        for i in range(0, self.get_stack_count()-1  ):
            self.close_file(0)

def smartSum(x,key,value):
    """ create a new page in x if key is not a page of x
        otherwise add value to x[key] """
    if key not in list(x.keys()):
        x[key] = value
    else:   x[key]+=value

def close( self ):
        """
        Close the db and release memory
        """
        if self.db is not None:
            self.db.commit()
            self.db.close()
            self.db = None

        return

def next (self):    # File-like object.

        """This is to support iterators over a file-like object.
        """

        result = self.readline()
        if result == self._empty_buffer:
            raise StopIteration
        return result

def close(self):
        """Closes the serial port."""
        if self.pyb and self.pyb.serial:
            self.pyb.serial.close()
        self.pyb = None

def _dotify(cls, data):
    """Add dots."""
    return ''.join(char if char in cls.PRINTABLE_DATA else '.' for char in data)

def get_closest_index(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.
    If two numbers are equally close, return the smallest number.

    Parameters
    ----------
    myList : array
        The list in which to find the closest value to myNumber
    myNumber : float
        The number to find the closest to in MyList

    Returns
    -------
    closest_values_index : int
        The index in the array of the number closest to myNumber in myList
    """
    closest_values_index = _np.where(self.time == take_closest(myList, myNumber))[0][0]
    return closest_values_index

def dot_v2(vec1, vec2):
    """Return the dot product of two vectors"""

    return vec1.x * vec2.x + vec1.y * vec2.y

def clear():
    """Clears the console."""
    if sys.platform.startswith("win"):
        call("cls", shell=True)
    else:
        call("clear", shell=True)

def downsample_with_striding(array, factor):
    """Downsample x by factor using striding.

    @return: The downsampled array, of the same type as x.
    """
    return array[tuple(np.s_[::f] for f in factor)]

def add(self, entity):
		"""
		Adds the supplied dict as a new entity
		"""
		result = self._http_req('connections', method='POST', payload=entity)
		status = result['status']
		if not status==201:
			raise ServiceRegistryError(status,"Couldn't add entity")

		self.debug(0x01,result)
		return result['decoded']

def polyline(self, arr):
        """Draw a set of lines"""
        for i in range(0, len(arr) - 1):
            self.line(arr[i][0], arr[i][1], arr[i + 1][0], arr[i + 1][1])

def accel_next(self, *args):
        """Callback to go to the next tab. Called by the accel key.
        """
        if self.get_notebook().get_current_page() + 1 == self.get_notebook().get_n_pages():
            self.get_notebook().set_current_page(0)
        else:
            self.get_notebook().next_page()
        return True

def safe_dump(data, stream=None, **kwds):
    """implementation of safe dumper using Ordered Dict Yaml Dumper"""
    return yaml.dump(data, stream=stream, Dumper=ODYD, **kwds)

def flatten_list(l):
    """ Nested lists to single-level list, does not split strings"""
    return list(chain.from_iterable(repeat(x,1) if isinstance(x,str) else x for x in l))

def display_pil_image(im):
   """Displayhook function for PIL Images, rendered as PNG."""
   from IPython.core import display
   b = BytesIO()
   im.save(b, format='png')
   data = b.getvalue()

   ip_img = display.Image(data=data, format='png', embed=True)
   return ip_img._repr_png_()

def process_docstring(app, what, name, obj, options, lines):
    """React to a docstring event and append contracts to it."""
    # pylint: disable=unused-argument
    # pylint: disable=too-many-arguments
    lines.extend(_format_contracts(what=what, obj=obj))

def load_member(fqn):
    """Loads and returns a class for a given fully qualified name."""
    modulename, member_name = split_fqn(fqn)
    module = __import__(modulename, globals(), locals(), member_name)
    return getattr(module, member_name)

def camel_to_underscore(string):
    """Convert camelcase to lowercase and underscore.

    Recipe from http://stackoverflow.com/a/1176023

    Args:
        string (str): The string to convert.

    Returns:
        str: The converted string.
    """
    string = FIRST_CAP_RE.sub(r'\1_\2', string)
    return ALL_CAP_RE.sub(r'\1_\2', string).lower()

def encode_list(dynamizer, value):
    """ Encode a list for the DynamoDB format """
    encoded_list = []
    dict(map(dynamizer.raw_encode, value))
    for v in value:
        encoded_type, encoded_value = dynamizer.raw_encode(v)
        encoded_list.append({
            encoded_type: encoded_value,
        })
    return 'L', encoded_list

def basic_word_sim(word1, word2):
    """
    Simple measure of similarity: Number of letters in common / max length
    """
    return sum([1 for c in word1 if c in word2]) / max(len(word1), len(word2))

def datetime_to_ms(dt):
    """
    Converts a datetime to a millisecond accuracy timestamp
    """
    seconds = calendar.timegm(dt.utctimetuple())
    return seconds * 1000 + int(dt.microsecond / 1000)

def intersect(d1, d2):
    """Intersect dictionaries d1 and d2 by key *and* value."""
    return dict((k, d1[k]) for k in d1 if k in d2 and d1[k] == d2[k])

def add_index_alias(es, index_name, alias_name):
    """Add index alias to index_name"""

    es.indices.put_alias(index=index_name, name=terms_alias)

def compare(string1, string2):
    """Compare two strings while protecting against timing attacks

    :param str string1: the first string
    :param str string2: the second string

    :returns: True if the strings are equal, False if not
    :rtype: :obj:`bool`
    """
    if len(string1) != len(string2):
        return False
    result = True
    for c1, c2 in izip(string1, string2):
        result &= c1 == c2
    return result

def all_documents(index=INDEX_NAME):
    """
    Get all documents from the given index.

    Returns full Elasticsearch objects so you can get metadata too.
    """
    query = {
        'query': {
            'match_all': {}
        }
    }
    for result in raw_query(query, index=index):
        yield result

def is_int(string):
    """
    Checks if a string is an integer. If the string value is an integer
    return True, otherwise return False. 
    
    Args:
        string: a string to test.

    Returns: 
        boolean
    """
    try:
        a = float(string)
        b = int(a)
    except ValueError:
        return False
    else:
        return a == b

def all_documents(index=INDEX_NAME):
    """
    Get all documents from the given index.

    Returns full Elasticsearch objects so you can get metadata too.
    """
    query = {
        'query': {
            'match_all': {}
        }
    }
    for result in raw_query(query, index=index):
        yield result

def compute_ssim(image1, image2, gaussian_kernel_sigma=1.5,
                 gaussian_kernel_width=11):
    """Computes SSIM.

    Args:
      im1: First PIL Image object to compare.
      im2: Second PIL Image object to compare.

    Returns:
      SSIM float value.
    """
    gaussian_kernel_1d = get_gaussian_kernel(
        gaussian_kernel_width, gaussian_kernel_sigma)
    return SSIM(image1, gaussian_kernel_1d).ssim_value(image2)

def get_average_color(colors):
    """Calculate the average color from the list of colors, where each color
    is a 3-tuple of (r, g, b) values.
    """
    c = reduce(color_reducer, colors)
    total = len(colors)
    return tuple(v / total for v in c)

def set_strict(self, value):
        """
        Set the strict mode active/disable

        :param value:
        :type value: bool
        """
        assert isinstance(value, bool)
        self.__settings.set_strict(value)

def distance_matrix(trains1, trains2, cos, tau):
    """
    Return the *bipartite* (rectangular) distance matrix between the observations in the first and the second list.

    Convenience function; equivalent to ``dissimilarity_matrix(trains1, trains2, cos, tau, "distance")``. Refer to :func:`pymuvr.dissimilarity_matrix` for full documentation.
    """
    return dissimilarity_matrix(trains1, trains2, cos, tau, "distance")

def get_enum_from_name(self, enum_name):
        """
            Return an enum from a name
        Args:
            enum_name (str): name of the enum
        Returns:
            Enum
        """
        return next((e for e in self.enums if e.name == enum_name), None)

def _cal_dist2center(X, center):
    """ Calculate the SSE to the cluster center
    """
    dmemb2cen = scipy.spatial.distance.cdist(X, center.reshape(1,X.shape[1]), metric='seuclidean')
    return(np.sum(dmemb2cen))

def get_enum_from_name(self, enum_name):
        """
            Return an enum from a name
        Args:
            enum_name (str): name of the enum
        Returns:
            Enum
        """
        return next((e for e in self.enums if e.name == enum_name), None)

def accuracy(conf_matrix):
  """
  Given a confusion matrix, returns the accuracy.
  Accuracy Definition: http://research.ics.aalto.fi/events/eyechallenge2005/evaluation.shtml
  """
  total, correct = 0.0, 0.0
  for true_response, guess_dict in conf_matrix.items():
    for guess, count in guess_dict.items():
      if true_response == guess:
        correct += count
      total += count
  return correct/total

def EnumValueName(self, enum, value):
    """Returns the string name of an enum value.

    This is just a small helper method to simplify a common operation.

    Args:
      enum: string name of the Enum.
      value: int, value of the enum.

    Returns:
      string name of the enum value.

    Raises:
      KeyError if either the Enum doesn't exist or the value is not a valid
        value for the enum.
    """
    return self.enum_types_by_name[enum].values_by_number[value].name

def accuracy(conf_matrix):
  """
  Given a confusion matrix, returns the accuracy.
  Accuracy Definition: http://research.ics.aalto.fi/events/eyechallenge2005/evaluation.shtml
  """
  total, correct = 0.0, 0.0
  for true_response, guess_dict in conf_matrix.items():
    for guess, count in guess_dict.items():
      if true_response == guess:
        correct += count
      total += count
  return correct/total

def write_enum(fo, datum, schema):
    """An enum is encoded by a int, representing the zero-based position of
    the symbol in the schema."""
    index = schema['symbols'].index(datum)
    write_int(fo, index)

def confusion_matrix(self):
        """Confusion matrix plot
        """
        return plot.confusion_matrix(self.y_true, self.y_pred,
                                     self.target_names, ax=_gen_ax())

def launched():
    """Test whether the current python environment is the correct lore env.

    :return:  :any:`True` if the environment is launched
    :rtype: bool
    """
    if not PREFIX:
        return False

    return os.path.realpath(sys.prefix) == os.path.realpath(PREFIX)

def native_conn(self):
        """Native connection object."""
        if self.__native is None:
            self.__native = self._get_connection()

        return self.__native

def raise_os_error(_errno, path=None):
    """
    Helper for raising the correct exception under Python 3 while still
    being able to raise the same common exception class in Python 2.7.
    """

    msg = "%s: '%s'" % (strerror(_errno), path) if path else strerror(_errno)
    raise OSError(_errno, msg)

def _make_cmd_list(cmd_list):
    """
    Helper function to easily create the proper json formated string from a list of strs
    :param cmd_list: list of strings
    :return: str json formatted
    """
    cmd = ''
    for i in cmd_list:
        cmd = cmd + '"' + i + '",'
    cmd = cmd[:-1]
    return cmd

def quote(s, unsafe='/'):
    """Pass in a dictionary that has unsafe characters as the keys, and the percent
    encoded value as the value."""
    res = s.replace('%', '%25')
    for c in unsafe:
        res = res.replace(c, '%' + (hex(ord(c)).upper())[2:])
    return res

def eintr_retry(exc_type, f, *args, **kwargs):
    """Calls a function.  If an error of the given exception type with
    interrupted system call (EINTR) occurs calls the function again.
    """
    while True:
        try:
            return f(*args, **kwargs)
        except exc_type as exc:
            if exc.errno != EINTR:
                raise
        else:
            break

def asynchronous(function, event):
    """
    Runs the function asynchronously taking care of exceptions.
    """
    thread = Thread(target=synchronous, args=(function, event))
    thread.daemon = True
    thread.start()

def get_number(s, cast=int):
    """
    Try to get a number out of a string, and cast it.
    """
    import string
    d = "".join(x for x in str(s) if x in string.digits)
    return cast(d)

def _expand(self, str, local_vars={}):
        """Expand $vars in a string."""
        return ninja_syntax.expand(str, self.vars, local_vars)

def str2bytes(x):
  """Convert input argument to bytes"""
  if type(x) is bytes:
    return x
  elif type(x) is str:
    return bytes([ ord(i) for i in x ])
  else:
    return str2bytes(str(x))

def end_block(self):
        """Ends an indentation block, leaving an empty line afterwards"""
        self.current_indent -= 1

        # If we did not add a new line automatically yet, now it's the time!
        if not self.auto_added_line:
            self.writeln()
            self.auto_added_line = True

def copy(self):
		"""Return a shallow copy."""
		return self.__class__(self.operations.copy(), self.collection, self.document)

def cli(yamlfile, directory, out, classname, format):
    """ Generate graphviz representations of the biolink model """
    DotGenerator(yamlfile, format).serialize(classname=classname, dirname=directory, filename=out)

def tanimoto_coefficient(a, b):
    """Measured similarity between two points in a multi-dimensional space.

    Returns:
        1.0 if the two points completely overlap,
        0.0 if the two points are infinitely far apart.
    """
    return sum(map(lambda (x,y): float(x)*float(y), zip(a,b))) / sum([
          -sum(map(lambda (x,y): float(x)*float(y), zip(a,b))),
           sum(map(lambda x: float(x)**2, a)),
           sum(map(lambda x: float(x)**2, b))])

def update(self, other_dict):
        """update() extends rather than replaces existing key lists."""
        for key, value in iter_multi_items(other_dict):
            MultiDict.add(self, key, value)

def empty_line_count_at_the_end(self):
        """
        Return number of empty lines at the end of the document.
        """
        count = 0
        for line in self.lines[::-1]:
            if not line or line.isspace():
                count += 1
            else:
                break

        return count

def update(self, other_dict):
        """update() extends rather than replaces existing key lists."""
        for key, value in iter_multi_items(other_dict):
            MultiDict.add(self, key, value)

def num_leaves(tree):
    """Determine the number of leaves in a tree"""
    if tree.is_leaf:
        return 1
    else:
        return num_leaves(tree.left_child) + num_leaves(tree.right_child)

def resources(self):
        """Retrieve contents of each page of PDF"""
        return [self.pdf.getPage(i) for i in range(self.pdf.getNumPages())]

def count_list(the_list):
    """
    Generates a count of the number of times each unique item appears in a list
    """
    count = the_list.count
    result = [(item, count(item)) for item in set(the_list)]
    result.sort()
    return result

def _extract_node_text(node):
    """Extract text from a given lxml node."""

    texts = map(
        six.text_type.strip, map(six.text_type, map(unescape, node.xpath(".//text()")))
    )
    return " ".join(text for text in texts if text)

def best(self):
        """
        Returns the element with the highest probability.
        """
        b = (-1e999999, None)
        for k, c in iteritems(self.counts):
            b = max(b, (c, k))
        return b[1]

def jaccard(c_1, c_2):
    """
    Calculates the Jaccard similarity between two sets of nodes. Called by mroc.

    Inputs:  - c_1: Community (set of nodes) 1.
             - c_2: Community (set of nodes) 2.

    Outputs: - jaccard_similarity: The Jaccard similarity of these two communities.
    """
    nom = np.intersect1d(c_1, c_2).size
    denom = np.union1d(c_1, c_2).size
    return nom/denom

def array_dim(arr):
    """Return the size of a multidimansional array.
    """
    dim = []
    while True:
        try:
            dim.append(len(arr))
            arr = arr[0]
        except TypeError:
            return dim

def stft(func=None, **kwparams):
  """
  Short Time Fourier Transform for real data keeping the full FFT block.

  Same to the default STFT strategy, but with new defaults. This is the same
  to:

  .. code-block:: python

    stft.base(transform=numpy.fft.fft,
              inverse_transform=lambda *args: numpy.fft.ifft(*args).real)

  See ``stft.base`` docs for more.
  """
  from numpy.fft import fft, ifft
  ifft_r = lambda *args: ifft(*args).real
  return stft.base(transform=fft, inverse_transform=ifft_r)(func, **kwparams)

def covariance(self,pt0,pt1):
        """ get the covarince between two points implied by Vario2d

        Parameters
        ----------
        pt0 : (iterable of len 2)
            first point x and y
        pt1 : (iterable of len 2)
            second point x and y

        Returns
        -------
        cov : float
            covariance between pt0 and pt1

        """

        x = np.array([pt0[0],pt1[0]])
        y = np.array([pt0[1],pt1[1]])
        names = ["n1","n2"]
        return self.covariance_matrix(x,y,names=names).x[0,1]

def Output(self):
    """Output all sections of the page."""
    self.Open()
    self.Header()
    self.Body()
    self.Footer()

def coerce(self, value):
        """Convert from whatever is given to a list of scalars for the lookup_field."""
        if isinstance(value, dict):
            value = [value]
        if not isiterable_notstring(value):
            value = [value]
        return [coerce_single_instance(self.lookup_field, v) for v in value]

def chmod_add_excute(filename):
        """
        Adds execute permission to file.
        :param filename:
        :return:
        """
        st = os.stat(filename)
        os.chmod(filename, st.st_mode | stat.S_IEXEC)

def unique(_list):
    """
    Makes the list have unique items only and maintains the order

    list(set()) won't provide that

    :type _list list
    :rtype: list
    """
    ret = []

    for item in _list:
        if item not in ret:
            ret.append(item)

    return ret

def file_writelines_flush_sync(path, lines):
    """
    Fill file at @path with @lines then flush all buffers
    (Python and system buffers)
    """
    fp = open(path, 'w')
    try:
        fp.writelines(lines)
        flush_sync_file_object(fp)
    finally:
        fp.close()

def __copy__(self):
        """A magic method to implement shallow copy behavior."""
        return self.__class__.load(self.dump(), context=self.context)

def get_lines(handle, line):
    """
    Get zero-indexed line from an open file-like.
    """
    for i, l in enumerate(handle):
        if i == line:
            return l

def mkdir(dir, enter):
    """Create directory with template for topic of the current environment

    """

    if not os.path.exists(dir):
        os.makedirs(dir)

def get_time(filename):
	"""
	Get the modified time for a file as a datetime instance
	"""
	ts = os.stat(filename).st_mtime
	return datetime.datetime.utcfromtimestamp(ts)

def write_file(filename, content):
    """Create the file with the given content"""
    print 'Generating {0}'.format(filename)
    with open(filename, 'wb') as out_f:
        out_f.write(content)

def make_file_read_only(file_path):
    """
    Removes the write permissions for the given file for owner, groups and others.

    :param file_path: The file whose privileges are revoked.
    :raise FileNotFoundError: If the given file does not exist.
    """
    old_permissions = os.stat(file_path).st_mode
    os.chmod(file_path, old_permissions & ~WRITE_PERMISSIONS)

def beta_pdf(x, a, b):
  """Beta distirbution probability density function."""
  bc = 1 / beta(a, b)
  fc = x ** (a - 1)
  sc = (1 - x) ** (b - 1)
  return bc * fc * sc

def file_read(filename):
    """Read a file and close it.  Returns the file source."""
    fobj = open(filename,'r');
    source = fobj.read();
    fobj.close()
    return source

def list_of_lists_to_dict(l):
    """ Convert list of key,value lists to dict

    [['id', 1], ['id', 2], ['id', 3], ['foo': 4]]
    {'id': [1, 2, 3], 'foo': [4]}
    """
    d = {}
    for key, val in l:
        d.setdefault(key, []).append(val)
    return d

def parse_comments_for_file(filename):
    """
    Return a list of all parsed comments in a file.  Mostly for testing &
    interactive use.
    """
    return [parse_comment(strip_stars(comment), next_line)
            for comment, next_line in get_doc_comments(read_file(filename))]

def rnormal(mu, tau, size=None):
    """
    Random normal variates.
    """
    return np.random.normal(mu, 1. / np.sqrt(tau), size)

def _fill_array_from_list(the_list, the_array):
        """Fill an `array` from a `list`"""
        for i, val in enumerate(the_list):
            the_array[i] = val
        return the_array

def force_iterable(f):
    """Will make any functions return an iterable objects by wrapping its result in a list."""
    def wrapper(*args, **kwargs):
        r = f(*args, **kwargs)
        if hasattr(r, '__iter__'):
            return r
        else:
            return [r]
    return wrapper

def fillna(series_or_arr, missing_value=0.0):
    """Fill missing values in pandas objects and numpy arrays.

    Arguments
    ---------
    series_or_arr : pandas.Series, numpy.ndarray
        The numpy array or pandas series for which the missing values
        need to be replaced.
    missing_value : float, int, str
        The value to replace the missing value with. Default 0.0.

    Returns
    -------
    pandas.Series, numpy.ndarray
        The numpy array or pandas series with the missing values
        filled.
    """

    if pandas.notnull(missing_value):
        if isinstance(series_or_arr, (numpy.ndarray)):
            series_or_arr[numpy.isnan(series_or_arr)] = missing_value
        else:
            series_or_arr.fillna(missing_value, inplace=True)

    return series_or_arr

def force_iterable(f):
    """Will make any functions return an iterable objects by wrapping its result in a list."""
    def wrapper(*args, **kwargs):
        r = f(*args, **kwargs)
        if hasattr(r, '__iter__'):
            return r
        else:
            return [r]
    return wrapper

def filter_dict(d, keys):
    """
    Creates a new dict from an existing dict that only has the given keys
    """
    return {k: v for k, v in d.items() if k in keys}

def a2s(a):
    """
     convert 3,3 a matrix to 6 element "s" list  (see Tauxe 1998)
    """
    s = np.zeros((6,), 'f')  # make the a matrix
    for i in range(3):
        s[i] = a[i][i]
    s[3] = a[0][1]
    s[4] = a[1][2]
    s[5] = a[0][2]
    return s

def str2int(string_with_int):
    """ Collect digits from a string """
    return int("".join([char for char in string_with_int if char in string.digits]) or 0)

def sp_rand(m,n,a):
    """
    Generates an mxn sparse 'd' matrix with round(a*m*n) nonzeros.
    """
    if m == 0 or n == 0: return spmatrix([], [], [], (m,n))
    nnz = min(max(0, int(round(a*m*n))), m*n)
    nz = matrix(random.sample(range(m*n), nnz), tc='i')
    return spmatrix(normal(nnz,1), nz%m, matrix([int(ii) for ii in nz/m]), (m,n))

def filter_dict_by_key(d, keys):
    """Filter the dict *d* to remove keys not in *keys*."""
    return {k: v for k, v in d.items() if k in keys}

def iter_finds(regex_obj, s):
    """Generate all matches found within a string for a regex and yield each match as a string"""
    if isinstance(regex_obj, str):
        for m in re.finditer(regex_obj, s):
            yield m.group()
    else:
        for m in regex_obj.finditer(s):
            yield m.group()

def flattened_nested_key_indices(nested_dict):
    """
    Combine the outer and inner keys of nested dictionaries into a single
    ordering.
    """
    outer_keys, inner_keys = collect_nested_keys(nested_dict)
    combined_keys = list(sorted(set(outer_keys + inner_keys)))
    return {k: i for (i, k) in enumerate(combined_keys)}

def split_every(n, iterable):
    """Returns a generator that spits an iteratable into n-sized chunks. The last chunk may have
    less than n elements.

    See http://stackoverflow.com/a/22919323/503377."""
    items = iter(iterable)
    return itertools.takewhile(bool, (list(itertools.islice(items, n)) for _ in itertools.count()))

def highpass(cutoff):
  """
  This strategy uses an exponential approximation for cut-off frequency
  calculation, found by matching the one-pole Laplace lowpass filter
  and mirroring the resulting filter to get a highpass.
  """
  R = thub(exp(cutoff - pi), 2)
  return (1 - R) / (1 + R * z ** -1)

def add_column(filename,column,formula,force=False):
    """ Add a column to a FITS file.

    ADW: Could this be replaced by a ftool?
    """
    columns = parse_formula(formula)
    logger.info("Running file: %s"%filename)
    logger.debug("  Reading columns: %s"%columns)
    data = fitsio.read(filename,columns=columns)

    logger.debug('  Evaluating formula: %s'%formula)
    col = eval(formula)

    col = np.asarray(col,dtype=[(column,col.dtype)])
    insert_columns(filename,col,force=force)
    return True

def parse_comments_for_file(filename):
    """
    Return a list of all parsed comments in a file.  Mostly for testing &
    interactive use.
    """
    return [parse_comment(strip_stars(comment), next_line)
            for comment, next_line in get_doc_comments(read_file(filename))]

def default_static_path():
    """
        Return the path to the javascript bundle
    """
    fdir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(fdir, '../assets/'))

def Timestamp(year, month, day, hour, minute, second):
    """Constructs an object holding a datetime/timestamp value."""
    return datetime.datetime(year, month, day, hour, minute, second)

def initialize_api(flask_app):
    """Initialize an API."""
    if not flask_restplus:
        return

    api = flask_restplus.Api(version="1.0", title="My Example API")
    api.add_resource(HelloWorld, "/hello")

    blueprint = flask.Blueprint("api", __name__, url_prefix="/api")
    api.init_app(blueprint)
    flask_app.register_blueprint(blueprint)

def from_json(cls, json_doc):
        """Parse a JSON string and build an entity."""
        try:
            d = json.load(json_doc)
        except AttributeError:  # catch the read() error
            d = json.loads(json_doc)

        return cls.from_dict(d)

def logout(cache):
    """
    Logs out the current session by removing it from the cache. This is
    expected to only occur when a session has
    """
    cache.set(flask.session['auth0_key'], None)
    flask.session.clear()
    return True

async def restart(request):
    """
    Returns OK, then waits approximately 1 second and restarts container
    """
    def wait_and_restart():
        log.info('Restarting server')
        sleep(1)
        os.system('kill 1')
    Thread(target=wait_and_restart).start()
    return web.json_response({"message": "restarting"})

def lambda_failure_response(*args):
        """
        Helper function to create a Lambda Failure Response

        :return: A Flask Response
        """
        response_data = jsonify(ServiceErrorResponses._LAMBDA_FAILURE)
        return make_response(response_data, ServiceErrorResponses.HTTP_STATUS_CODE_502)

def pointer(self):
        """Get a ctypes void pointer to the memory mapped region.

        :type: ctypes.c_void_p
        """
        return ctypes.cast(ctypes.pointer(ctypes.c_uint8.from_buffer(self.mapping, 0)), ctypes.c_void_p)

def rstjinja(app, docname, source):
    """
    Render our pages as a jinja template for fancy templating goodness.
    """
    # Make sure we're outputting HTML
    if app.builder.format != 'html':
        return
    src = source[0]
    rendered = app.builder.templates.render_string(
        src, app.config.html_context
    )
    source[0] = rendered

def cumsum(inlist):
    """
Returns a list consisting of the cumulative sum of the items in the
passed list.

Usage:   lcumsum(inlist)
"""
    newlist = copy.deepcopy(inlist)
    for i in range(1, len(newlist)):
        newlist[i] = newlist[i] + newlist[i - 1]
    return newlist

def handleFlaskPostRequest(flaskRequest, endpoint):
    """
    Handles the specified flask request for one of the POST URLS
    Invokes the specified endpoint to generate a response.
    """
    if flaskRequest.method == "POST":
        return handleHttpPost(flaskRequest, endpoint)
    elif flaskRequest.method == "OPTIONS":
        return handleHttpOptions()
    else:
        raise exceptions.MethodNotAllowedException()

def screen_cv2(self):
        """cv2 Image of current window screen"""
        pil_image = self.screen.convert('RGB')
        cv2_image = np.array(pil_image)
        pil_image.close()
        # Convert RGB to BGR 
        cv2_image = cv2_image[:, :, ::-1]
        return cv2_image

def init_app(self, app):
        """Initialize Flask application."""
        app.config.from_pyfile('{0}.cfg'.format(app.name), silent=True)

def multidict_to_dict(d):
    """
    Turns a werkzeug.MultiDict or django.MultiValueDict into a dict with
    list values
    :param d: a MultiDict or MultiValueDict instance
    :return: a dict instance
    """
    return dict((k, v[0] if len(v) == 1 else v) for k, v in iterlists(d))

def _join(verb):
    """
    Join helper
    """
    data = pd.merge(verb.x, verb.y, **verb.kwargs)

    # Preserve x groups
    if isinstance(verb.x, GroupedDataFrame):
        data.plydata_groups = list(verb.x.plydata_groups)
    return data

def pretty_print_post(req):
    """Helper to print a "prepared" query. Useful to debug a POST query.

    However pay attention at the formatting used in
    this function because it is programmed to be pretty
    printed and may differ from the actual request.
    """
    print(('{}\n{}\n{}\n\n{}'.format(
        '-----------START-----------',
        req.method + ' ' + req.url,
        '\n'.join('{}: {}'.format(k, v) for k, v in list(req.headers.items())),
        req.body,
    )))

def get_unique_indices(df, axis=1):
    """

    :param df:
    :param axis:
    :return:
    """
    return dict(zip(df.columns.names, dif.columns.levels))

def make_coord_dict(coord):
    """helper function to make a dict from a coordinate for logging"""
    return dict(
        z=int_if_exact(coord.zoom),
        x=int_if_exact(coord.column),
        y=int_if_exact(coord.row),
    )

def do_restart(self, line):
        """Request that the Outstation perform a cold restart. Command syntax is: restart"""
        self.application.master.Restart(opendnp3.RestartType.COLD, restart_callback)

def convert_time_string(date_str):
    """ Change a date string from the format 2018-08-15T23:55:17 into a datetime object """
    dt, _, _ = date_str.partition(".")
    dt = datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S")
    return dt

def mimetype(self):
        """MIME type of the asset."""
        return (self.environment.mimetypes.get(self.format_extension) or
                self.compiler_mimetype or 'application/octet-stream')

def parse(self, s):
        """
        Parses a date string formatted like ``YYYY-MM-DD``.
        """
        return datetime.datetime.strptime(s, self.date_format).date()

def get_handler(self, *args, **options):
        """
        Returns the default WSGI handler for the runner.
        """
        handler = get_internal_wsgi_application()
        from django.contrib.staticfiles.handlers import StaticFilesHandler
        return StaticFilesHandler(handler)

def AmericanDateToEpoch(self, date_str):
    """Take a US format date and return epoch."""
    try:
      epoch = time.strptime(date_str, "%m/%d/%Y")
      return int(calendar.timegm(epoch)) * 1000000
    except ValueError:
      return 0

def get_handler(self, *args, **options):
        """
        Returns the default WSGI handler for the runner.
        """
        handler = get_internal_wsgi_application()
        from django.contrib.staticfiles.handlers import StaticFilesHandler
        return StaticFilesHandler(handler)

def date_to_datetime(x):
    """Convert a date into a datetime"""
    if not isinstance(x, datetime) and isinstance(x, date):
        return datetime.combine(x, time())
    return x

def staticdir():
    """Return the location of the static data directory."""
    root = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(root, "static")

def ms_to_datetime(ms):
    """
    Converts a millisecond accuracy timestamp to a datetime
    """
    dt = datetime.datetime.utcfromtimestamp(ms / 1000)
    return dt.replace(microsecond=(ms % 1000) * 1000).replace(tzinfo=pytz.utc)

def flatten_array(grid):
    """
    Takes a multi-dimensional array and returns a 1 dimensional array with the
    same contents.
    """
    grid = [grid[i][j] for i in range(len(grid)) for j in range(len(grid[i]))]
    while type(grid[0]) is list:
        grid = flatten_array(grid)
    return grid

def this_quarter():
        """ Return start and end date of this quarter. """
        since = TODAY + delta(day=1)
        while since.month % 3 != 0:
            since -= delta(months=1)
        until = since + delta(months=3)
        return Date(since), Date(until)

def setLib(self, lib):
        """ Copy the lib items into our font. """
        for name, item in lib.items():
            self.font.lib[name] = item

def C_dict2array(C):
    """Convert an OrderedDict containing C values to a 1D array."""
    return np.hstack([np.asarray(C[k]).ravel() for k in C_keys])

def _enter_plotting(self, fontsize=9):
        """assumes that a figure is open """
        # interactive_status = matplotlib.is_interactive()
        self.original_fontsize = pyplot.rcParams['font.size']
        pyplot.rcParams['font.size'] = fontsize
        pyplot.hold(False)  # opens a figure window, if non exists
        pyplot.ioff()

def cric__decision_tree():
    """ Decision Tree
    """
    model = sklearn.tree.DecisionTreeClassifier(random_state=0, max_depth=4)

    # we want to explain the raw probability outputs of the trees
    model.predict = lambda X: model.predict_proba(X)[:,1]
    
    return model

def count_levels(value):
    """
        Count how many levels are in a dict:
        scalar, list etc = 0
        {} = 0
        {'a':1} = 1
        {'a' : {'b' : 1}} = 2
        etc...
    """
    if not isinstance(value, dict) or len(value) == 0:
        return 0
    elif len(value) == 0:
        return 0 #An emptu dict has 0
    else:
        nextval = list(value.values())[0]
        return 1 + count_levels(nextval)

def list_of(cls):
    """
    Returns a function that checks that each element in a
    list is of a specific type.
    """
    return lambda l: isinstance(l, list) and all(isinstance(x, cls) for x in l)

def __del__(self):
        """Frees all resources.
        """
        if hasattr(self, '_Api'):
            self._Api.close()

        self._Logger.info('object destroyed')

def pprint(obj, verbose=False, max_width=79, newline='\n'):
    """
    Like `pretty` but print to stdout.
    """
    printer = RepresentationPrinter(sys.stdout, verbose, max_width, newline)
    printer.pretty(obj)
    printer.flush()
    sys.stdout.write(newline)
    sys.stdout.flush()

def lighting(im, b, c):
    """ Adjust image balance and contrast """
    if b==0 and c==1: return im
    mu = np.average(im)
    return np.clip((im-mu)*c+mu+b,0.,1.).astype(np.float32)

def fixed(ctx, number, decimals=2, no_commas=False):
    """
    Formats the given number in decimal format using a period and commas
    """
    value = _round(ctx, number, decimals)
    format_str = '{:f}' if no_commas else '{:,f}'
    return format_str.format(value)

def b(s):
	""" Encodes Unicode strings to byte strings, if necessary. """

	return s if isinstance(s, bytes) else s.encode(locale.getpreferredencoding())

def format_float(value): # not used
    """Modified form of the 'g' format specifier.
    """
    string = "{:g}".format(value).replace("e+", "e")
    string = re.sub("e(-?)0*(\d+)", r"e\1\2", string)
    return string

def strToBool(val):
    """
    Helper function to turn a string representation of "true" into
    boolean True.
    """
    if isinstance(val, str):
        val = val.lower()

    return val in ['true', 'on', 'yes', True]

def pprint(self, seconds):
        """
        Pretty Prints seconds as Hours:Minutes:Seconds.MilliSeconds

        :param seconds:  The time in seconds.
        """
        return ("%d:%02d:%02d.%03d", reduce(lambda ll, b: divmod(ll[0], b) + ll[1:], [(seconds * 1000,), 1000, 60, 60]))

def delete(self, name):
        """
        Deletes the named entry in the cache.
        :param name: the name.
        :return: true if it is deleted.
        """
        if name in self._cache:
            del self._cache[name]
            self.writeCache()
            # TODO clean files
            return True
        return False

def fmt_sz(intval):
    """ Format a byte sized value.
    """
    try:
        return fmt.human_size(intval)
    except (ValueError, TypeError):
        return "N/A".rjust(len(fmt.human_size(0)))

def remove_this_tlink(self,tlink_id):
        """
        Removes the tlink for the given tlink identifier
        @type tlink_id: string
        @param tlink_id: the tlink identifier to be removed
        """
        for tlink in self.get_tlinks():
            if tlink.get_id() == tlink_id:
                self.node.remove(tlink.get_node())
                break

def rotation_from_quaternion(q_wxyz):
        """Convert quaternion array to rotation matrix.

        Parameters
        ----------
        q_wxyz : :obj:`numpy.ndarray` of float
            A quaternion in wxyz order.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            A 3x3 rotation matrix made from the quaternion.
        """
        q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        R = transformations.quaternion_matrix(q_xyzw)[:3,:3]
        return R

def rm(venv_name):
    """ Removes the venv by name """
    inenv = InenvManager()
    venv = inenv.get_venv(venv_name)
    click.confirm("Delete dir {}".format(venv.path))
    shutil.rmtree(venv.path)

def convert_tstamp(response):
	"""
	Convert a Stripe API timestamp response (unix epoch) to a native datetime.

	:rtype: datetime
	"""
	if response is None:
		# Allow passing None to convert_tstamp()
		return response

	# Overrides the set timezone to UTC - I think...
	tz = timezone.utc if settings.USE_TZ else None

	return datetime.datetime.fromtimestamp(response, tz)

def trim(self):
        """Clear not used counters"""
        for key, value in list(iteritems(self.counters)):
            if value.empty():
                del self.counters[key]

def connect():
    """Connect to FTP server, login and return an ftplib.FTP instance."""
    ftp_class = ftplib.FTP if not SSL else ftplib.FTP_TLS
    ftp = ftp_class(timeout=TIMEOUT)
    ftp.connect(HOST, PORT)
    ftp.login(USER, PASSWORD)
    if SSL:
        ftp.prot_p()  # secure data connection
    return ftp

def safe_delete(filename):
  """Delete a file safely. If it's not present, no-op."""
  try:
    os.unlink(filename)
  except OSError as e:
    if e.errno != errno.ENOENT:
      raise

def remove_from_lib(self, name):
        """ Remove an object from the bin folder. """
        self.__remove_path(os.path.join(self.root_dir, "lib", name))

def _manhattan_distance(vec_a, vec_b):
    """Return manhattan distance between two lists of numbers."""
    if len(vec_a) != len(vec_b):
        raise ValueError('len(vec_a) must equal len(vec_b)')
    return sum(map(lambda a, b: abs(a - b), vec_a, vec_b))

def remove_file_from_s3(awsclient, bucket, key):
    """Remove a file from an AWS S3 bucket.

    :param awsclient:
    :param bucket:
    :param key:
    :return:
    """
    client_s3 = awsclient.get_client('s3')
    response = client_s3.delete_object(Bucket=bucket, Key=key)

def first_unique_char(s):
    """
    :type s: str
    :rtype: int
    """
    if (len(s) == 1):
        return 0
    ban = []
    for i in range(len(s)):
        if all(s[i] != s[k] for k in range(i + 1, len(s))) == True and s[i] not in ban:
            return i
        else:
            ban.append(s[i])
    return -1

def safe_delete(filename):
  """Delete a file safely. If it's not present, no-op."""
  try:
    os.unlink(filename)
  except OSError as e:
    if e.errno != errno.ENOENT:
      raise

def removeFromRegistery(obj) :
	"""Removes an object/rabalist from registery. This is useful if you want to allow the garbage collector to free the memory
	taken by the objects you've already loaded. Be careful might cause some discrepenties in your scripts. For objects,
	cascades to free the registeries of related rabalists also"""
	
	if isRabaObject(obj) :
		_unregisterRabaObjectInstance(obj)
	elif isRabaList(obj) :
		_unregisterRabaListInstance(obj)

def ex(self, cmd):
        """Execute a normal python statement in user namespace."""
        with self.builtin_trap:
            exec cmd in self.user_global_ns, self.user_ns

def normalize_value(text):
    """
    This removes newlines and multiple spaces from a string.
    """
    result = text.replace('\n', ' ')
    result = re.subn('[ ]{2,}', ' ', result)[0]
    return result

def user_exists(username):
    """Check if a user exists"""
    try:
        pwd.getpwnam(username)
        user_exists = True
    except KeyError:
        user_exists = False
    return user_exists

def delete_object_from_file(file_name, save_key, file_location):
    """
    Function to delete objects from a shelve
    Args:
        file_name: Shelve storage file name
        save_key: The name of the key the item is stored in
        file_location: The location of the file, derive from the os module

    Returns:

    """
    file = __os.path.join(file_location, file_name)
    shelve_store = __shelve.open(file)
    del shelve_store[save_key]
    shelve_store.close()

def EvalGaussianPdf(x, mu, sigma):
    """Computes the unnormalized PDF of the normal distribution.

    x: value
    mu: mean
    sigma: standard deviation
    
    returns: float probability density
    """
    return scipy.stats.norm.pdf(x, mu, sigma)

def from_json_list(cls, api_client, data):
        """Convert a list of JSON values to a list of models
        """
        return [cls.from_json(api_client, item) for item in data]

def get_incomplete_path(filename):
  """Returns a temporary filename based on filename."""
  random_suffix = "".join(
      random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
  return filename + ".incomplete" + random_suffix

def is_int_type(val):
    """Return True if `val` is of integer type."""
    try:               # Python 2
        return isinstance(val, (int, long))
    except NameError:  # Python 3
        return isinstance(val, int)

def list_get(l, idx, default=None):
    """
    Get from a list with an optional default value.
    """
    try:
        if l[idx]:
            return l[idx]
        else:
            return default
    except IndexError:
        return default

def estimate_complexity(self, x,y,z,n):
        """ 
        calculates a rough guess of runtime based on product of parameters 
        """
        num_calculations = x * y * z * n
        run_time = num_calculations / 100000  # a 2014 PC does about 100k calcs in a second (guess based on prior logs)
        return self.show_time_as_short_string(run_time)

def _get_str_columns(sf):
    """
    Returns a list of names of columns that are string type.
    """
    return [name for name in sf.column_names() if sf[name].dtype == str]

def isSquare(matrix):
    """Check that ``matrix`` is square.

    Returns
    =======
    is_square : bool
        ``True`` if ``matrix`` is square, ``False`` otherwise.

    """
    try:
        try:
            dim1, dim2 = matrix.shape
        except AttributeError:
            dim1, dim2 = _np.array(matrix).shape
    except ValueError:
        return False
    if dim1 == dim2:
        return True
    return False

def search_for_tweets_about(user_id, params):
    """ Search twitter API """
    url = "https://api.twitter.com/1.1/search/tweets.json"
    response = make_twitter_request(url, user_id, params)
    return process_tweets(response.json()["statuses"])

def is_hex_string(string):
    """Check if the string is only composed of hex characters."""
    pattern = re.compile(r'[A-Fa-f0-9]+')
    if isinstance(string, six.binary_type):
        string = str(string)
    return pattern.match(string) is not None

def getHeaders(self):
        """
         Get the headers of this DataFrame.

         Returns:
            The headers of this DataFrame.
        """
        headers = self._impl.getHeaders()
        return tuple(
            headers.getIndex(i) for i in range(self._impl.getNumCols())
        )

def C_dict2array(C):
    """Convert an OrderedDict containing C values to a 1D array."""
    return np.hstack([np.asarray(C[k]).ravel() for k in C_keys])

def caller_locals():
    """Get the local variables in the caller's frame."""
    import inspect
    frame = inspect.currentframe()
    try:
        return frame.f_back.f_back.f_locals
    finally:
        del frame

def multi_pop(d, *args):
    """ pops multiple keys off a dict like object """
    retval = {}
    for key in args:
        if key in d:
            retval[key] = d.pop(key)
    return retval

def __dir__(self):
        u"""Returns a list of children and available helper methods."""
        return sorted(self.keys() | {m for m in dir(self.__class__) if m.startswith('to_')})

def get_flat_size(self):
        """Returns the total length of all of the flattened variables.

        Returns:
            The length of all flattened variables concatenated.
        """
        return sum(
            np.prod(v.get_shape().as_list()) for v in self.variables.values())

def get_tablenames(cur):
    """ Conveinience: """
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tablename_list_ = cur.fetchall()
    tablename_list = [str(tablename[0]) for tablename in tablename_list_ ]
    return tablename_list

def me(self):
        """Similar to :attr:`.Guild.me` except it may return the :class:`.ClientUser` in private message contexts."""
        return self.guild.me if self.guild is not None else self.bot.user

def current_memory_usage():
    """
    Returns this programs current memory usage in bytes
    """
    import psutil
    proc = psutil.Process(os.getpid())
    #meminfo = proc.get_memory_info()
    meminfo = proc.memory_info()
    rss = meminfo[0]  # Resident Set Size / Mem Usage
    vms = meminfo[1]  # Virtual Memory Size / VM Size  # NOQA
    return rss

def get_chat_member(self, user_id):
        """
        Get information about a member of a chat.

        :param int user_id: Unique identifier of the target user
        """
        return self.bot.api_call(
            "getChatMember", chat_id=str(self.id), user_id=str(user_id)
        )

def debug_src(src, pm=False, globs=None):
    """Debug a single doctest docstring, in argument `src`'"""
    testsrc = script_from_examples(src)
    debug_script(testsrc, pm, globs)

def get_encoding(binary):
    """Return the encoding type."""

    try:
        from chardet import detect
    except ImportError:
        LOGGER.error("Please install the 'chardet' module")
        sys.exit(1)

    encoding = detect(binary).get('encoding')

    return 'iso-8859-1' if encoding == 'CP949' else encoding

def get_last_or_frame_exception():
    """Intended to be used going into post mortem routines.  If
    sys.last_traceback is set, we will return that and assume that
    this is what post-mortem will want. If sys.last_traceback has not
    been set, then perhaps we *about* to raise an error and are
    fielding an exception. So assume that sys.exc_info()[2]
    is where we want to look."""

    try:
        if inspect.istraceback(sys.last_traceback):
            # We do have a traceback so prefer that.
            return sys.last_type, sys.last_value, sys.last_traceback
    except AttributeError:
        pass
    return sys.exc_info()

def paste(cmd=paste_cmd, stdout=PIPE):
    """Returns system clipboard contents.
    """
    return Popen(cmd, stdout=stdout).communicate()[0].decode('utf-8')

def confirm_credential_display(force=False):
    if force:
        return True

    msg = """
    [WARNING] Your credential is about to be displayed on screen.
    If this is really what you want, type 'y' and press enter."""

    result = click.confirm(text=msg)
    return result

def dict_hash(dct):
    """Return a hash of the contents of a dictionary"""
    dct_s = json.dumps(dct, sort_keys=True)

    try:
        m = md5(dct_s)
    except TypeError:
        m = md5(dct_s.encode())

    return m.hexdigest()

def get_distance_matrix(x):
    """Get distance matrix given a matrix. Used in testing."""
    square = nd.sum(x ** 2.0, axis=1, keepdims=True)
    distance_square = square + square.transpose() - (2.0 * nd.dot(x, x.transpose()))
    return nd.sqrt(distance_square)

def dict_hash(dct):
    """Return a hash of the contents of a dictionary"""
    dct_s = json.dumps(dct, sort_keys=True)

    try:
        m = md5(dct_s)
    except TypeError:
        m = md5(dct_s.encode())

    return m.hexdigest()

def safe_call(cls, method, *args):
        """ Call a remote api method but don't raise if an error occurred."""
        return cls.call(method, *args, safe=True)

def direct2dDistance(self, point):
        """consider the distance between two mapPoints, ignoring all terrain, pathing issues"""
        if not isinstance(point, MapPoint): return 0.0
        return  ((self.x-point.x)**2 + (self.y-point.y)**2)**(0.5) # simple distance formula

def clone(src, **kwargs):
    """Clones object with optionally overridden fields"""
    obj = object.__new__(type(src))
    obj.__dict__.update(src.__dict__)
    obj.__dict__.update(kwargs)
    return obj

def dict_pop_or(d, key, default=None):
    """ Try popping a key from a dict.
        Instead of raising KeyError, just return the default value.
    """
    val = default
    with suppress(KeyError):
        val = d.pop(key)
    return val

def load_from_file(module_path):
    """
    Load a python module from its absolute filesystem path

    Borrowed from django-cms
    """
    from imp import load_module, PY_SOURCE

    imported = None
    if module_path:
        with open(module_path, 'r') as openfile:
            imported = load_module('mod', openfile, module_path, ('imported', 'r', PY_SOURCE))
    return imported

def close( self ):
        """
        Close the db and release memory
        """
        if self.db is not None:
            self.db.commit()
            self.db.close()
            self.db = None

        return

def get_parent_folder_name(file_path):
    """Finds parent folder of file

    :param file_path: path
    :return: Name of folder container
    """
    return os.path.split(os.path.split(os.path.abspath(file_path))[0])[-1]

def _is_start(event, node, tagName):  # pylint: disable=invalid-name
    """Return true if (event, node) is a start event for tagname."""

    return event == pulldom.START_ELEMENT and node.tagName == tagName

def find_geom(geom, geoms):
    """
    Returns the index of a geometry in a list of geometries avoiding
    expensive equality checks of `in` operator.
    """
    for i, g in enumerate(geoms):
        if g is geom:
            return i

def _index2n(self, index):
        """

        :param index: index convention
        :return: n
        """
        n_float = np.sqrt(index + 1) - 1
        n_int = int(n_float)
        if n_int == n_float:
            n = n_int
        else:
            n = n_int + 1
        return n

def bisect_index(a, x):
    """ Find the leftmost index of an element in a list using binary search.

    Parameters
    ----------
    a: list
        A sorted list.
    x: arbitrary
        The element.

    Returns
    -------
    int
        The index.

    """
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError

def peak_memory_usage():
    """Return peak memory usage in MB"""
    if sys.platform.startswith('win'):
        p = psutil.Process()
        return p.memory_info().peak_wset / 1024 / 1024

    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    factor_mb = 1 / 1024
    if sys.platform == 'darwin':
        factor_mb = 1 / (1024 * 1024)
    return mem * factor_mb

def get_substring_idxs(substr, string):
    """
    Return a list of indexes of substr. If substr not found, list is
    empty.

    Arguments:
        substr (str): Substring to match.
        string (str): String to match in.

    Returns:
        list of int: Start indices of substr.
    """
    return [match.start() for match in re.finditer(substr, string)]

def to_basestring(value):
    """Converts a string argument to a subclass of basestring.

    In python2, byte and unicode strings are mostly interchangeable,
    so functions that deal with a user-supplied argument in combination
    with ascii string constants can use either and should return the type
    the user supplied.  In python3, the two types are not interchangeable,
    so this method is needed to convert byte strings to unicode.
    """
    if isinstance(value, _BASESTRING_TYPES):
        return value
    assert isinstance(value, bytes)
    return value.decode("utf-8")

def string_to_int( s ):
  """Convert a string of bytes into an integer, as per X9.62."""
  result = 0
  for c in s:
    if not isinstance(c, int): c = ord( c )
    result = 256 * result + c
  return result

def Output(self):
    """Output all sections of the page."""
    self.Open()
    self.Header()
    self.Body()
    self.Footer()

def get_by(self, name):
    """get element by name"""
    return next((item for item in self if item.name == name), None)

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

def _lookup_enum_in_ns(namespace, value):
    """Return the attribute of namespace corresponding to value."""
    for attribute in dir(namespace):
        if getattr(namespace, attribute) == value:
            return attribute

def dot_v2(vec1, vec2):
    """Return the dot product of two vectors"""

    return vec1.x * vec2.x + vec1.y * vec2.y

def apply_kwargs(func, **kwargs):
    """Call *func* with kwargs, but only those kwargs that it accepts.
    """
    new_kwargs = {}
    params = signature(func).parameters
    for param_name in params.keys():
        if param_name in kwargs:
            new_kwargs[param_name] = kwargs[param_name]
    return func(**new_kwargs)

def _intermediary_to_dot(tables, relationships):
    """ Returns the dot source representing the database in a string. """
    t = '\n'.join(t.to_dot() for t in tables)
    r = '\n'.join(r.to_dot() for r in relationships)
    return '{}\n{}\n{}\n}}'.format(GRAPH_BEGINNING, t, r)

def prevmonday(num):
    """
    Return unix SECOND timestamp of "num" mondays ago
    """
    today = get_today()
    lastmonday = today - timedelta(days=today.weekday(), weeks=num)
    return lastmonday

def draw(self, mode="triangles"):
        """ Draw collection """

        gl.glDepthMask(0)
        Collection.draw(self, mode)
        gl.glDepthMask(1)

def tail(self, n=10):
        """
        Get an SArray that contains the last n elements in the SArray.

        Parameters
        ----------
        n : int
            The number of elements to fetch

        Returns
        -------
        out : SArray
            A new SArray which contains the last n rows of the current SArray.
        """
        with cython_context():
            return SArray(_proxy=self.__proxy__.tail(n))

def purge_duplicates(list_in):
    """Remove duplicates from list while preserving order.

    Parameters
    ----------
    list_in: Iterable

    Returns
    -------
    list
        List of first occurences in order
    """
    _list = []
    for item in list_in:
        if item not in _list:
            _list.append(item)
    return _list

def get_table_names(connection):
	"""
	Return a list of the table names in the database.
	"""
	cursor = connection.cursor()
	cursor.execute("SELECT name FROM sqlite_master WHERE type == 'table'")
	return [name for (name,) in cursor]

def _histplot_bins(column, bins=100):
    """Helper to get bins for histplot."""
    col_min = np.min(column)
    col_max = np.max(column)
    return range(col_min, col_max + 2, max((col_max - col_min) // bins, 1))

def get_language(self):
        """
        Get the language parameter from the current request.
        """
        return get_language_parameter(self.request, self.query_language_key, default=self.get_default_language(object=object))

def are_equal_xml(a_xml, b_xml):
    """Normalize and compare XML documents for equality. The document may or may not be
    a DataONE type.

    Args:
      a_xml: str
      b_xml: str
        XML documents to compare for equality.

    Returns:
      bool: ``True`` if the XML documents are semantically equivalent.

    """
    a_dom = xml.dom.minidom.parseString(a_xml)
    b_dom = xml.dom.minidom.parseString(b_xml)
    return are_equal_elements(a_dom.documentElement, b_dom.documentElement)

def fetch(table, cols="*", where=(), group="", order=(), limit=(), **kwargs):
    """Convenience wrapper for database SELECT and fetch all."""
    return select(table, cols, where, group, order, limit, **kwargs).fetchall()

def get_max(qs, field):
    """
    get max for queryset.

    qs: queryset
    field: The field name to max.
    """
    max_field = '%s__max' % field
    num = qs.aggregate(Max(field))[max_field]
    return num if num else 0

def deleteAll(self):
        """
        Deletes whole Solr index. Use with care.
        """
        for core in self.endpoints:
            self._send_solr_command(self.endpoints[core], "{\"delete\": { \"query\" : \"*:*\"}}")

def get_free_memory_win():
    """Return current free memory on the machine for windows.

    Warning : this script is really not robust
    Return in MB unit
    """
    stat = MEMORYSTATUSEX()
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return int(stat.ullAvailPhys / 1024 / 1024)

def matrixTimesVector(MM, aa):
    """

    :param MM: A matrix of size 3x3
    :param aa: A vector of size 3
    :return: A vector of size 3 which is the product of the matrix by the vector
    """
    bb = np.zeros(3, np.float)
    for ii in range(3):
        bb[ii] = np.sum(MM[ii, :] * aa)
    return bb

def remove_bad(string):
    """
    remove problem characters from string
    """
    remove = [':', ',', '(', ')', ' ', '|', ';', '\'']
    for c in remove:
        string = string.replace(c, '_')
    return string

def get_last_modified_timestamp(self):
        """
        Looks at the files in a git root directory and grabs the last modified timestamp
        """
        cmd = "find . -print0 | xargs -0 stat -f '%T@ %p' | sort -n | tail -1 | cut -f2- -d' '"
        ps = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        output = ps.communicate()[0]
        print output

def validate_email(email):
    """
    Validates an email address
    Source: Himanshu Shankar (https://github.com/iamhssingh)
    Parameters
    ----------
    email: str

    Returns
    -------
    bool
    """
    from django.core.validators import validate_email
    from django.core.exceptions import ValidationError
    try:
        validate_email(email)
        return True
    except ValidationError:
        return False

def get_model(name):
    """
    Convert a model's verbose name to the model class. This allows us to
    use the models verbose name in steps.
    """

    model = MODELS.get(name.lower(), None)

    assert model, "Could not locate model by name '%s'" % name

    return model

def read_string_from_file(path, encoding="utf8"):
  """
  Read entire contents of file into a string.
  """
  with codecs.open(path, "rb", encoding=encoding) as f:
    value = f.read()
  return value

def current_timestamp():
    """Returns current time as ISO8601 formatted string in the Zulu TZ"""
    now = datetime.utcnow()
    timestamp = now.isoformat()[0:19] + 'Z'

    debug("generated timestamp: {now}".format(now=timestamp))

    return timestamp

def session_to_epoch(timestamp):
    """ converts Synergy Timestamp for session to UTC zone seconds since epoch """
    utc_timetuple = datetime.strptime(timestamp, SYNERGY_SESSION_PATTERN).replace(tzinfo=None).utctimetuple()
    return calendar.timegm(utc_timetuple)

def dimensions(self):
        """Get width and height of a PDF"""
        size = self.pdf.getPage(0).mediaBox
        return {'w': float(size[2]), 'h': float(size[3])}

def debug_src(src, pm=False, globs=None):
    """Debug a single doctest docstring, in argument `src`'"""
    testsrc = script_from_examples(src)
    debug_script(testsrc, pm, globs)

def hex_to_rgb(h):
    """ Returns 0 to 1 rgb from a hex list or tuple """
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255. for i in (0, 2 ,4))

def _escape(s):
    """ Helper method that escapes parameters to a SQL query. """
    e = s
    e = e.replace('\\', '\\\\')
    e = e.replace('\n', '\\n')
    e = e.replace('\r', '\\r')
    e = e.replace("'", "\\'")
    e = e.replace('"', '\\"')
    return e

def remove_ext(fname):
    """Removes the extension from a filename
    """
    bn = os.path.basename(fname)
    return os.path.splitext(bn)[0]

def _cdf(self, xloc, dist, base, cache):
        """Cumulative distribution function."""
        return evaluation.evaluate_forward(dist, base**xloc, cache=cache)

def getdefaultencoding():
    """Return IPython's guess for the default encoding for bytes as text.

    Asks for stdin.encoding first, to match the calling Terminal, but that
    is often None for subprocesses.  Fall back on locale.getpreferredencoding()
    which should be a sensible platform default (that respects LANG environment),
    and finally to sys.getdefaultencoding() which is the most conservative option,
    and usually ASCII.
    """
    enc = get_stream_enc(sys.stdin)
    if not enc or enc=='ascii':
        try:
            # There are reports of getpreferredencoding raising errors
            # in some cases, which may well be fixed, but let's be conservative here.
            enc = locale.getpreferredencoding()
        except Exception:
            pass
    return enc or sys.getdefaultencoding()

def equal(obj1, obj2):
    """Calculate equality between two (Comparable) objects."""
    Comparable.log(obj1, obj2, '==')
    equality = obj1.equality(obj2)
    Comparable.log(obj1, obj2, '==', result=equality)
    return equality

def get_size(objects):
    """Compute the total size of all elements in objects."""
    res = 0
    for o in objects:
        try:
            res += _getsizeof(o)
        except AttributeError:
            print("IGNORING: type=%s; o=%s" % (str(type(o)), str(o)))
    return res

def run(self):
        """Run the event loop."""
        self.signal_init()
        self.listen_init()
        self.logger.info('starting')
        self.loop.start()

def monthly(date=datetime.date.today()):
    """
    Take a date object and return the first day of the month.
    """
    return datetime.date(date.year, date.month, 1)

def previous_quarter(d):
    """
    Retrieve the previous quarter for dt
    """
    from django_toolkit.datetime_util import quarter as datetime_quarter
    return quarter( (datetime_quarter(datetime(d.year, d.month, d.day))[0] + timedelta(days=-1)).date() )

def findfirst(f, coll):
    """Return first occurrence matching f, otherwise None"""
    result = list(dropwhile(f, coll))
    return result[0] if result else None

def isTestCaseDisabled(test_case_class, method_name):
    """
    I check to see if a method on a TestCase has been disabled via nose's
    convention for disabling a TestCase.  This makes it so that users can
    mix nose's parameterized tests with green as a runner.
    """
    test_method = getattr(test_case_class, method_name)
    return getattr(test_method, "__test__", 'not nose') is False

def _get_item_position(self, idx):
        """Return a tuple of (start, end) indices of an item from its index."""
        start = 0 if idx == 0 else self._index[idx - 1] + 1
        end = self._index[idx]
        return start, end

def to_dataframe(products):
        """Return the products from a query response as a Pandas DataFrame
        with the values in their appropriate Python types.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("to_dataframe requires the optional dependency Pandas.")

        return pd.DataFrame.from_dict(products, orient='index')

def index(self, elem):
        """Find the index of elem in the reversed iterator."""
        return _coconut.len(self._iter) - self._iter.index(elem) - 1

def convert_time_string(date_str):
    """ Change a date string from the format 2018-08-15T23:55:17 into a datetime object """
    dt, _, _ = date_str.partition(".")
    dt = datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S")
    return dt

def get_key_by_value(dictionary, search_value):
    """
    searchs a value in a dicionary and returns the key of the first occurrence

    :param dictionary: dictionary to search in
    :param search_value: value to search for
    """
    for key, value in dictionary.iteritems():
        if value == search_value:
            return ugettext(key)

def AmericanDateToEpoch(self, date_str):
    """Take a US format date and return epoch."""
    try:
      epoch = time.strptime(date_str, "%m/%d/%Y")
      return int(calendar.timegm(epoch)) * 1000000
    except ValueError:
      return 0

def _dt_to_epoch(dt):
        """Convert datetime to epoch seconds."""
        try:
            epoch = dt.timestamp()
        except AttributeError:  # py2
            epoch = (dt - datetime(1970, 1, 1)).total_seconds()
        return epoch

def string_to_genomic_range(rstring):
  """ Convert a string to a genomic range

  :param rstring: string representing a genomic range chr1:801-900
  :type rstring:
  :returns: object representing the string
  :rtype: GenomicRange
  """
  m = re.match('([^:]+):(\d+)-(\d+)',rstring)
  if not m: 
    sys.stderr.write("ERROR: problem with range string "+rstring+"\n")
  return GenomicRange(m.group(1),int(m.group(2)),int(m.group(3)))

def convert_2_utc(self, datetime_, timezone):
        """convert to datetime to UTC offset."""

        datetime_ = self.tz_mapper[timezone].localize(datetime_)
        return datetime_.astimezone(pytz.UTC)

def listified_tokenizer(source):
    """Tokenizes *source* and returns the tokens as a list of lists."""
    io_obj = io.StringIO(source)
    return [list(a) for a in tokenize.generate_tokens(io_obj.readline)]

def intty(cls):
        """ Check if we are in a tty. """
        # XXX: temporary hack until we can detect if we are in a pipe or not
        return True

        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            return True

        return False

def code_from_ipynb(nb, markdown=False):
    """
    Get the code for a given notebook

    nb is passed in as a dictionary that's a parsed ipynb file
    """
    code = PREAMBLE
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            # transform the input to executable Python
            code += ''.join(cell['source'])
        if cell['cell_type'] == 'markdown':
            code += '\n# ' + '# '.join(cell['source'])
        # We want a blank newline after each cell's output.
        # And the last line of source doesn't have a newline usually.
        code += '\n\n'
    return code

def generate_unique_host_id():
    """Generate a unique ID, that is somewhat guaranteed to be unique among all
    instances running at the same time."""
    host = ".".join(reversed(socket.gethostname().split(".")))
    pid = os.getpid()
    return "%s.%d" % (host, pid)

def computeFactorial(n):
    """
    computes factorial of n
    """
    sleep_walk(10)
    ret = 1
    for i in range(n):
        ret = ret * (i + 1)
    return ret

def get_url_nofollow(url):
	""" 
	function to get return code of a url

	Credits: http://blog.jasonantman.com/2013/06/python-script-to-check-a-list-of-urls-for-return-code-and-final-return-code-if-redirected/
	"""
	try:
		response = urlopen(url)
		code = response.getcode()
		return code
	except HTTPError as e:
		return e.code
	except:
		return 0

def get_mnist(data_type="train", location="/tmp/mnist"):
    """
    Get mnist dataset with features and label as ndarray.
    Data would be downloaded automatically if it doesn't present at the specific location.

    :param data_type: "train" for training data and "test" for testing data.
    :param location: Location to store mnist dataset.
    :return: (features: ndarray, label: ndarray)
    """
    X, Y = mnist.read_data_sets(location, data_type)
    return X, Y + 1

def get_value(key, obj, default=missing):
    """Helper for pulling a keyed value off various types of objects"""
    if isinstance(key, int):
        return _get_value_for_key(key, obj, default)
    return _get_value_for_keys(key.split('.'), obj, default)

def c_str(string):
    """"Convert a python string to C string."""
    if not isinstance(string, str):
        string = string.decode('ascii')
    return ctypes.c_char_p(string.encode('utf-8'))

def get_inputs_from_cm(index, cm):
    """Return indices of inputs to the node with the given index."""
    return tuple(i for i in range(cm.shape[0]) if cm[i][index])

def software_fibonacci(n):
    """ a normal old python function to return the Nth fibonacci number. """
    a, b = 0, 1
    for i in range(n):
        a, b = b, a + b
    return a

def _read_stream_for_size(stream, buf_size=65536):
    """Reads a stream discarding the data read and returns its size."""
    size = 0
    while True:
        buf = stream.read(buf_size)
        size += len(buf)
        if not buf:
            break
    return size

def open_file(file, mode):
	"""Open a file.

	:arg file: file-like or path-like object.
	:arg str mode: ``mode`` argument for :func:`open`.
	"""
	if hasattr(file, "read"):
		return file
	if hasattr(file, "open"):
		return file.open(mode)
	return open(file, mode)

def save_dot(self, fd):
        """ Saves a representation of the case in the Graphviz DOT language.
        """
        from pylon.io import DotWriter
        DotWriter(self).write(fd)

def read_utf8(fh, byteorder, dtype, count, offsetsize):
    """Read tag data from file and return as unicode string."""
    return fh.read(count).decode('utf-8')

def _column_resized(self, col, old_width, new_width):
        """Update the column width."""
        self.dataTable.setColumnWidth(col, new_width)
        self._update_layout()

def clean_dataframe(df):
    """Fill NaNs with the previous value, the next value or if all are NaN then 1.0"""
    df = df.fillna(method='ffill')
    df = df.fillna(0.0)
    return df

def gcall(func, *args, **kwargs):
    """
    Calls a function, with the given arguments inside Gtk's main loop.
    Example::
        gcall(lbl.set_text, "foo")

    If this call would be made in a thread there could be problems, using
    it inside Gtk's main loop makes it thread safe.
    """
    def idle():
        with gdk.lock:
            return bool(func(*args, **kwargs))
    return gobject.idle_add(idle)

def camel_case_from_underscores(string):
    """generate a CamelCase string from an underscore_string."""
    components = string.split('_')
    string = ''
    for component in components:
        string += component[0].upper() + component[1:]
    return string

def np_hash(a):
    """Return a hash of a NumPy array."""
    if a is None:
        return hash(None)
    # Ensure that hashes are equal whatever the ordering in memory (C or
    # Fortran)
    a = np.ascontiguousarray(a)
    # Compute the digest and return a decimal int
    return int(hashlib.sha1(a.view(a.dtype)).hexdigest(), 16)

def tuple_search(t, i, v):
    """
    Search tuple array by index and value
    :param t: tuple array
    :param i: index of the value in each tuple
    :param v: value
    :return: the first tuple in the array with the specific index / value
    """
    for e in t:
        if e[i] == v:
            return e
    return None

def top(n, width=WIDTH, style=STYLE):
    """Prints the top row of a table"""
    return hrule(n, width, linestyle=STYLES[style].top)

def binSearch(arr, val):
  """ 
  Function for running binary search on a sorted list.

  :param arr: (list) a sorted list of integers to search
  :param val: (int)  a integer to search for in the sorted array
  :returns: (int) the index of the element if it is found and -1 otherwise.
  """
  i = bisect_left(arr, val)
  if i != len(arr) and arr[i] == val:
    return i
  return -1

def equal(list1, list2):
    """ takes flags returns indexes of True values """
    return [item1 == item2 for item1, item2 in broadcast_zip(list1, list2)]

def update_index(index):
    """Re-index every document in a named index."""
    logger.info("Updating search index: '%s'", index)
    client = get_client()
    responses = []
    for model in get_index_models(index):
        logger.info("Updating search index model: '%s'", model.search_doc_type)
        objects = model.objects.get_search_queryset(index).iterator()
        actions = bulk_actions(objects, index=index, action="index")
        response = helpers.bulk(client, actions, chunk_size=get_setting("chunk_size"))
        responses.append(response)
    return responses

def is_in(self, search_list, pair):
        """
        If pair is in search_list, return the index. Otherwise return -1
        """
        index = -1
        for nr, i in enumerate(search_list):
            if(np.all(i == pair)):
                return nr
        return index

def hide(self):
        """Hide the window."""
        self.tk.withdraw()
        self._visible = False
        if self._modal:
            self.tk.grab_release()

def local_minima(img, min_distance = 4):
    r"""
    Returns all local minima from an image.
    
    Parameters
    ----------
    img : array_like
        The image.
    min_distance : integer
        The minimal distance between the minimas in voxels. If it is less, only the lower minima is returned.
    
    Returns
    -------
    indices : sequence
        List of all minima indices.
    values : sequence
        List of all minima values.
    """
    # @TODO: Write a unittest for this.
    fits = numpy.asarray(img)
    minfits = minimum_filter(fits, size=min_distance) # default mode is reflect
    minima_mask = fits == minfits
    good_indices = numpy.transpose(minima_mask.nonzero())
    good_fits = fits[minima_mask]
    order = good_fits.argsort()
    return good_indices[order], good_fits[order]

def oplot(self, x, y, **kw):
        """generic plotting method, overplotting any existing plot """
        self.panel.oplot(x, y, **kw)

def _factor_generator(n):
    """
    From a given natural integer, returns the prime factors and their multiplicity
    :param n: Natural integer
    :return:
    """
    p = prime_factors(n)
    factors = {}
    for p1 in p:
        try:
            factors[p1] += 1
        except KeyError:
            factors[p1] = 1
    return factors

def retrieve_by_id(self, id_):
        """Return a JSSObject for the element with ID id_"""
        items_with_id = [item for item in self if item.id == int(id_)]
        if len(items_with_id) == 1:
            return items_with_id[0].retrieve()

def newest_file(file_iterable):
  """
  Returns the name of the newest file given an iterable of file names.

  """
  return max(file_iterable, key=lambda fname: os.path.getmtime(fname))

def readline(self):
        """Get the next line including the newline or '' on EOF."""
        self.lineno += 1
        if self._buffer:
            return self._buffer.pop()
        else:
            return self.input.readline()

def union_overlapping(intervals):
    """Union any overlapping intervals in the given set."""
    disjoint_intervals = []

    for interval in intervals:
        if disjoint_intervals and disjoint_intervals[-1].overlaps(interval):
            disjoint_intervals[-1] = disjoint_intervals[-1].union(interval)
        else:
            disjoint_intervals.append(interval)

    return disjoint_intervals

def init():
    """
    Execute init tasks for all components (virtualenv, pip).
    """
    print(yellow("# Setting up environment...\n", True))
    virtualenv.init()
    virtualenv.update_requirements()
    print(green("\n# DONE.", True))
    print(green("Type ") + green("activate", True) + green(" to enable your virtual environment."))

def binSearch(arr, val):
  """ 
  Function for running binary search on a sorted list.

  :param arr: (list) a sorted list of integers to search
  :param val: (int)  a integer to search for in the sorted array
  :returns: (int) the index of the element if it is found and -1 otherwise.
  """
  i = bisect_left(arr, val)
  if i != len(arr) and arr[i] == val:
    return i
  return -1

def ratelimit_remaining(self):
        """Number of requests before GitHub imposes a ratelimit.

        :returns: int
        """
        json = self._json(self._get(self._github_url + '/rate_limit'), 200)
        core = json.get('resources', {}).get('core', {})
        self._remaining = core.get('remaining', 0)
        return self._remaining

def _sim_fill(r1, r2, imsize):
    """
        calculate the fill similarity over the image
    """
    bbsize = (
        (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]))
        * (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
    )
    return 1.0 - (bbsize - r1["size"] - r2["size"]) / imsize

def is_sparse_vector(x):
    """ x is a 2D sparse matrix with it's first shape equal to 1.
    """
    return sp.issparse(x) and len(x.shape) == 2 and x.shape[0] == 1

def _manhattan_distance(vec_a, vec_b):
    """Return manhattan distance between two lists of numbers."""
    if len(vec_a) != len(vec_b):
        raise ValueError('len(vec_a) must equal len(vec_b)')
    return sum(map(lambda a, b: abs(a - b), vec_a, vec_b))

def fit_gaussian(x, y, yerr, p0):
    """ Fit a Gaussian to the data """
    try:
        popt, pcov = curve_fit(gaussian, x, y, sigma=yerr, p0=p0, absolute_sigma=True)
    except RuntimeError:
        return [0],[0]
    return popt, pcov

def bytes_to_str(s, encoding='utf-8'):
    """Returns a str if a bytes object is given."""
    if six.PY3 and isinstance(s, bytes):
        return s.decode(encoding)
    return s

def apply_fit(xy,coeffs):
    """ Apply the coefficients from a linear fit to
        an array of x,y positions.

        The coeffs come from the 'coeffs' member of the
        'fit_arrays()' output.
    """
    x_new = coeffs[0][2] + coeffs[0][0]*xy[:,0] + coeffs[0][1]*xy[:,1]
    y_new = coeffs[1][2] + coeffs[1][0]*xy[:,0] + coeffs[1][1]*xy[:,1]

    return x_new,y_new

def go_to_parent_directory(self):
        """Go to parent directory"""
        self.chdir(osp.abspath(osp.join(getcwd_or_home(), os.pardir)))

def old_pad(s):
    """
    Pads an input string to a given block size.
    :param s: string
    :returns: The padded string.
    """
    if len(s) % OLD_BLOCK_SIZE == 0:
        return s

    return Padding.appendPadding(s, blocksize=OLD_BLOCK_SIZE)

def split_into_words(s):
  """Split a sentence into list of words."""
  s = re.sub(r"\W+", " ", s)
  s = re.sub(r"[_0-9]+", " ", s)
  return s.split()

def rewrap(s, width=COLS):
    """ Join all lines from input string and wrap it at specified width """
    s = ' '.join([l.strip() for l in s.strip().split('\n')])
    return '\n'.join(textwrap.wrap(s, width))

def lower_ext(abspath):
    """Convert file extension to lowercase.
    """
    fname, ext = os.path.splitext(abspath)
    return fname + ext.lower()

def flatten(nested):
    """ Return a flatten version of the nested argument """
    flat_return = list()

    def __inner_flat(nested,flat):
        for i in nested:
            __inner_flat(i, flat) if isinstance(i, list) else flat.append(i)
        return flat

    __inner_flat(nested,flat_return)

    return flat_return

def is_valid_url(url):
    """Checks if a given string is an url"""
    pieces = urlparse(url)
    return all([pieces.scheme, pieces.netloc])

def enable_writes(self):
        """Restores the state of the batched queue for writing."""
        self.write_buffer = []
        self.flush_lock = threading.RLock()
        self.flush_thread = FlushThread(self.max_batch_time,
                                        self._flush_writes)

def find_lt(a, x):
    """Find rightmost value less than x"""
    i = bisect.bisect_left(a, x)
    if i:
        return a[i-1]
    raise ValueError

def normalize_value(text):
    """
    This removes newlines and multiple spaces from a string.
    """
    result = text.replace('\n', ' ')
    result = re.subn('[ ]{2,}', ' ', result)[0]
    return result

def has_overlaps(self):
        """
        :returns: True if one or more range in the list overlaps with another
        :rtype: bool
        """
        sorted_list = sorted(self)
        for i in range(0, len(sorted_list) - 1):
            if sorted_list[i].overlaps(sorted_list[i + 1]):
                return True
        return False

def __call__(self, _):
        """Update the progressbar."""
        if self.iter % self.step == 0:
            self.pbar.update(self.step)

        self.iter += 1

def has_attribute(module_name, attribute_name):
    """Is this attribute present?"""
    init_file = '%s/__init__.py' % module_name
    return any(
        [attribute_name in init_line for init_line in open(init_file).readlines()]
    )

def combine(self, a, b):
        """A generator that combines two iterables."""

        for l in (a, b):
            for x in l:
                yield x

def is_serializable(obj):
    """Return `True` if the given object conforms to the Serializable protocol.

    :rtype: bool
    """
    if inspect.isclass(obj):
      return Serializable.is_serializable_type(obj)
    return isinstance(obj, Serializable) or hasattr(obj, '_asdict')

def _delete_local(self, filename):
        """Deletes the specified file from the local filesystem."""

        if os.path.exists(filename):
            os.remove(filename)

def clear():
    """Clears the console."""
    if sys.platform.startswith("win"):
        call("cls", shell=True)
    else:
        call("clear", shell=True)

def login(self, username, password=None, token=None):
        """Login user for protected API calls."""
        self.session.basic_auth(username, password)

def bit_clone( bits ):
    """
    Clone a bitset
    """
    new = BitSet( bits.size )
    new.ior( bits )
    return new

def get_year_start(day=None):
    """Returns January 1 of the given year."""
    day = add_timezone(day or datetime.date.today())
    return day.replace(month=1).replace(day=1)

def feature_union_concat(Xs, nsamples, weights):
    """Apply weights and concatenate outputs from a FeatureUnion"""
    if any(x is FIT_FAILURE for x in Xs):
        return FIT_FAILURE
    Xs = [X if w is None else X * w for X, w in zip(Xs, weights) if X is not None]
    if not Xs:
        return np.zeros((nsamples, 0))
    if any(sparse.issparse(f) for f in Xs):
        return sparse.hstack(Xs).tocsr()
    return np.hstack(Xs)

def fixed(ctx, number, decimals=2, no_commas=False):
    """
    Formats the given number in decimal format using a period and commas
    """
    value = _round(ctx, number, decimals)
    format_str = '{:f}' if no_commas else '{:,f}'
    return format_str.format(value)

def from_pb(cls, pb):
        """Instantiate the object from a protocol buffer.

        Args:
            pb (protobuf)

        Save a reference to the protocol buffer on the object.
        """
        obj = cls._from_pb(pb)
        obj._pb = pb
        return obj

def adapter(data, headers, **kwargs):
    """Wrap vertical table in a function for TabularOutputFormatter."""
    keys = ('sep_title', 'sep_character', 'sep_length')
    return vertical_table(data, headers, **filter_dict_by_key(kwargs, keys))

def _get_wow64():
    """
    Determines if the current process is running in Windows-On-Windows 64 bits.

    @rtype:  bool
    @return: C{True} of the current process is a 32 bit program running in a
        64 bit version of Windows, C{False} if it's either a 32 bit program
        in a 32 bit Windows or a 64 bit program in a 64 bit Windows.
    """
    # Try to determine if the debugger itself is running on WOW64.
    # On error assume False.
    if bits == 64:
        wow64 = False
    else:
        try:
            wow64 = IsWow64Process( GetCurrentProcess() )
        except Exception:
            wow64 = False
    return wow64

def get_lons_from_cartesian(x__, y__):
    """Get longitudes from cartesian coordinates.
    """
    return rad2deg(arccos(x__ / sqrt(x__ ** 2 + y__ ** 2))) * sign(y__)

def raise_os_error(_errno, path=None):
    """
    Helper for raising the correct exception under Python 3 while still
    being able to raise the same common exception class in Python 2.7.
    """

    msg = "%s: '%s'" % (strerror(_errno), path) if path else strerror(_errno)
    raise OSError(_errno, msg)

def stft(func=None, **kwparams):
  """
  Short Time Fourier Transform for real data keeping the full FFT block.

  Same to the default STFT strategy, but with new defaults. This is the same
  to:

  .. code-block:: python

    stft.base(transform=numpy.fft.fft,
              inverse_transform=lambda *args: numpy.fft.ifft(*args).real)

  See ``stft.base`` docs for more.
  """
  from numpy.fft import fft, ifft
  ifft_r = lambda *args: ifft(*args).real
  return stft.base(transform=fft, inverse_transform=ifft_r)(func, **kwparams)

def strip_spaces(s):
    """ Strip excess spaces from a string """
    return u" ".join([c for c in s.split(u' ') if c])

def set_timeout(scope, timeout):
    """
    Defines the time after which Exscript fails if it does not receive a
    prompt from the remote host.

    :type  timeout: int
    :param timeout: The timeout in seconds.
    """
    conn = scope.get('__connection__')
    conn.set_timeout(int(timeout[0]))
    return True

def _stdin_(p):
    """Takes input from user. Works for Python 2 and 3."""
    _v = sys.version[0]
    return input(p) if _v is '3' else raw_input(p)

def _get_printable_columns(columns, row):
    """Return only the part of the row which should be printed.
    """
    if not columns:
        return row

    # Extract the column values, in the order specified.
    return tuple(row[c] for c in columns)

def _stdin_(p):
    """Takes input from user. Works for Python 2 and 3."""
    _v = sys.version[0]
    return input(p) if _v is '3' else raw_input(p)

def _send_file(self, filename):
        """
        Sends a file via FTP.
        """
        # pylint: disable=E1101
        ftp = ftplib.FTP(host=self.host)
        ftp.login(user=self.user, passwd=self.password)
        ftp.set_pasv(True)
        ftp.storbinary("STOR %s" % os.path.basename(filename),
            file(filename, 'rb'))

def many_until1(these, term):
    """Like many_until but must consume at least one of these.
    """
    first = [these()]
    these_results, term_result = many_until(these, term)
    return (first + these_results, term_result)

def write_file(filename, content):
    """Create the file with the given content"""
    print 'Generating {0}'.format(filename)
    with open(filename, 'wb') as out_f:
        out_f.write(content)

def smooth_gaussian(image, sigma=1):
    """Returns Gaussian smoothed image.

    :param image: numpy array or :class:`jicimagelib.image.Image`
    :param sigma: standard deviation
    :returns: :class:`jicimagelib.image.Image`
    """
    return scipy.ndimage.filters.gaussian_filter(image, sigma=sigma, mode="nearest")

def uniform_noise(points):
    """Init a uniform noise variable."""
    return np.random.rand(1) * np.random.uniform(points, 1) \
        + random.sample([2, -2], 1)

def _uniqueid(n=30):
    """Return a unique string with length n.

    :parameter int N: number of character in the uniqueid
    :return: the uniqueid
    :rtype: str
    """
    return ''.join(random.SystemRandom().choice(
                   string.ascii_uppercase + string.ascii_lowercase)
                   for _ in range(n))

def get_var(self, name):
        """ Returns the variable set with the given name.
        """
        for var in self.vars:
            if var.name == name:
                return var
        else:
            raise ValueError

def daterange(start_date, end_date):
    """
    Yield one date per day from starting date to ending date.

    Args:
        start_date (date): starting date.
        end_date (date): ending date.

    Yields:
        date: a date for each day within the range.
    """
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def get_property(self, filename):
        """Opens the file and reads the value"""

        with open(self.filepath(filename)) as f:
            return f.read().strip()

def generate(env):
    """Add Builders and construction variables for SGI MIPS C++ to an Environment."""

    cplusplus.generate(env)

    env['CXX']         = 'CC'
    env['CXXFLAGS']    = SCons.Util.CLVar('-LANG:std')
    env['SHCXX']       = '$CXX'
    env['SHOBJSUFFIX'] = '.o'
    env['STATIC_AND_SHARED_OBJECTS_ARE_THE_SAME'] = 1

def convert_2_utc(self, datetime_, timezone):
        """convert to datetime to UTC offset."""

        datetime_ = self.tz_mapper[timezone].localize(datetime_)
        return datetime_.astimezone(pytz.UTC)

def money(min=0, max=10):
    """Return a str of decimal with two digits after a decimal mark."""
    value = random.choice(range(min * 100, max * 100))
    return "%1.2f" % (float(value) / 100)

def get_uniques(l):
    """ Returns a list with no repeated elements.
    """
    result = []

    for i in l:
        if i not in result:
            result.append(i)

    return result

def uniqueID(size=6, chars=string.ascii_uppercase + string.digits):
    """A quick and dirty way to get a unique string"""
    return ''.join(random.choice(chars) for x in xrange(size))

def asMaskedArray(self):
        """ Creates converts to a masked array
        """
        return ma.masked_array(data=self.data, mask=self.mask, fill_value=self.fill_value)

def sha1(s):
    """ Returns a sha1 of the given string
    """
    h = hashlib.new('sha1')
    h.update(s)
    return h.hexdigest()

def _weighted_selection(l, n):
    """
        Selects  n random elements from a list of (weight, item) tuples.
        Based on code snippet by Nick Johnson
    """
    cuml = []
    items = []
    total_weight = 0.0
    for weight, item in l:
        total_weight += weight
        cuml.append(total_weight)
        items.append(item)

    return [items[bisect.bisect(cuml, random.random()*total_weight)] for _ in range(n)]

def flattened_nested_key_indices(nested_dict):
    """
    Combine the outer and inner keys of nested dictionaries into a single
    ordering.
    """
    outer_keys, inner_keys = collect_nested_keys(nested_dict)
    combined_keys = list(sorted(set(outer_keys + inner_keys)))
    return {k: i for (i, k) in enumerate(combined_keys)}

def move_to_start(self, column_label):
        """Move a column to the first in order."""
        self._columns.move_to_end(column_label, last=False)
        return self

def list_apis(awsclient):
    """List APIs in account."""
    client_api = awsclient.get_client('apigateway')

    apis = client_api.get_rest_apis()['items']

    for api in apis:
        print(json2table(api))

def normalize_array(lst):
    """Normalizes list

    :param lst: Array of floats
    :return: Normalized (in [0, 1]) input array
    """
    np_arr = np.array(lst)
    x_normalized = np_arr / np_arr.max(axis=0)
    return list(x_normalized)

def files_changed():
    """
    Return the list of file changed in the current branch compared to `master`
    """
    with chdir(get_root()):
        result = run_command('git diff --name-only master...', capture='out')
    changed_files = result.stdout.splitlines()

    # Remove empty lines
    return [f for f in changed_files if f]

def prettyprint(d):
        """Print dicttree in Json-like format. keys are sorted
        """
        print(json.dumps(d, sort_keys=True, 
                         indent=4, separators=("," , ": ")))

def from_string(cls, s):
        """Return a `Status` instance from its string representation."""
        for num, text in cls._STATUS2STR.items():
            if text == s:
                return cls(num)
        else:
            raise ValueError("Wrong string %s" % s)

def _get_name(column_like):
    """
    Get the name from a column-like SQLAlchemy expression.

    Works for Columns and Cast expressions.
    """
    if isinstance(column_like, Column):
        return column_like.name
    elif isinstance(column_like, Cast):
        return column_like.clause.name

def get_table_names(connection):
	"""
	Return a list of the table names in the database.
	"""
	cursor = connection.cursor()
	cursor.execute("SELECT name FROM sqlite_master WHERE type == 'table'")
	return [name for (name,) in cursor]

def _reload(self, force=False):
        """Reloads the configuration from the file and environment variables. Useful if using
        `os.environ` instead of this class' `set_env` method, or if the underlying configuration
        file is changed externally.
        """
        self._config_map = dict()
        self._registered_env_keys = set()
        self.__reload_sources(force)
        self.__load_environment_keys()
        self.verify()
        self._clear_memoization()

def commits_with_message(message):
    """All commits with that message (in current branch)"""
    output = log("--grep '%s'" % message, oneline=True, quiet=True)
    lines = output.splitlines()
    return [l.split(' ', 1)[0] for l in lines]

def replaceNewlines(string, newlineChar):
	"""There's probably a way to do this with string functions but I was lazy.
		Replace all instances of \r or \n in a string with something else."""
	if newlineChar in string:
		segments = string.split(newlineChar)
		string = ""
		for segment in segments:
			string += segment
	return string

def get_grid_spatial_dimensions(self, variable):
        """Returns (width, height) for the given variable"""

        data = self.open_dataset(self.service).variables[variable.variable]
        dimensions = list(data.dimensions)
        return data.shape[dimensions.index(variable.x_dimension)], data.shape[dimensions.index(variable.y_dimension)]

def test():
    """Run the unit tests."""
    import unittest
    tests = unittest.TestLoader().discover('tests')
    unittest.TextTestRunner(verbosity=2).run(tests)

def distinct(xs):
    """Get the list of distinct values with preserving order."""
    # don't use collections.OrderedDict because we do support Python 2.6
    seen = set()
    return [x for x in xs if x not in seen and not seen.add(x)]

def autozoom(self, n=None):
        """
        Auto-scales the axes to fit all the data in plot index n. If n == None,
        auto-scale everyone.
        """
        if n==None:
            for p in self.plot_widgets: p.autoRange()
        else:        self.plot_widgets[n].autoRange()

        return self

def horz_dpi(self):
        """
        Integer dots per inch for the width of this image. Defaults to 72
        when not present in the file, as is often the case.
        """
        pHYs = self._chunks.pHYs
        if pHYs is None:
            return 72
        return self._dpi(pHYs.units_specifier, pHYs.horz_px_per_unit)

def _selectItem(self, index):
        """Select item in the list
        """
        self._selectedIndex = index
        self.setCurrentIndex(self.model().createIndex(index, 0))

def get_abi3_suffix():
    """Return the file extension for an abi3-compliant Extension()"""
    for suffix, _, _ in (s for s in imp.get_suffixes() if s[2] == imp.C_EXTENSION):
        if '.abi3' in suffix:  # Unix
            return suffix
        elif suffix == '.pyd':  # Windows
            return suffix

def print_env_info(key, out=sys.stderr):
    """If given environment key is defined, print it out."""
    value = os.getenv(key)
    if value is not None:
        print(key, "=", repr(value), file=out)

def get_file_size(fileobj):
    """
    Returns the size of a file-like object.
    """
    currpos = fileobj.tell()
    fileobj.seek(0, 2)
    total_size = fileobj.tell()
    fileobj.seek(currpos)
    return total_size

def Sum(a, axis, keep_dims):
    """
    Sum reduction op.
    """
    return np.sum(a, axis=axis if not isinstance(axis, np.ndarray) else tuple(axis),
                  keepdims=keep_dims),

def get_winfunc(libname, funcname, restype=None, argtypes=(), _libcache={}):
    """Retrieve a function from a library/DLL, and set the data types."""
    if libname not in _libcache:
        _libcache[libname] = windll.LoadLibrary(libname)
    func = getattr(_libcache[libname], funcname)
    func.argtypes = argtypes
    func.restype = restype
    return func

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

def _get_gid(name):
    """Returns a gid, given a group name."""
    if getgrnam is None or name is None:
        return None
    try:
        result = getgrnam(name)
    except KeyError:
        result = None
    if result is not None:
        return result[2]
    return None

def terminate(self):
    """Override of PantsService.terminate() that cleans up when the Pailgun server is terminated."""
    # Tear down the Pailgun TCPServer.
    if self.pailgun:
      self.pailgun.server_close()

    super(PailgunService, self).terminate()

def retrieve_by_id(self, id_):
        """Return a JSSObject for the element with ID id_"""
        items_with_id = [item for item in self if item.id == int(id_)]
        if len(items_with_id) == 1:
            return items_with_id[0].retrieve()

def is_same_dict(d1, d2):
    """Test two dictionary is equal on values. (ignore order)
    """
    for k, v in d1.items():
        if isinstance(v, dict):
            is_same_dict(v, d2[k])
        else:
            assert d1[k] == d2[k]

    for k, v in d2.items():
        if isinstance(v, dict):
            is_same_dict(v, d1[k])
        else:
            assert d1[k] == d2[k]

def series_index(self, series):
        """
        Return the integer index of *series* in this sequence.
        """
        for idx, s in enumerate(self):
            if series is s:
                return idx
        raise ValueError('series not in chart data object')

def object_as_dict(obj):
    """Turn an SQLAlchemy model into a dict of field names and values.

    Based on https://stackoverflow.com/a/37350445/1579058
    """
    return {c.key: getattr(obj, c.key)
            for c in inspect(obj).mapper.column_attrs}

def _find_first_of(line, substrings):
    """Find earliest occurrence of one of substrings in line.

    Returns pair of index and found substring, or (-1, None)
    if no occurrences of any of substrings were found in line.
    """
    starts = ((line.find(i), i) for i in substrings)
    found = [(i, sub) for i, sub in starts if i != -1]
    if found:
        return min(found)
    else:
        return -1, None

def retrieve_by_id(self, id_):
        """Return a JSSObject for the element with ID id_"""
        items_with_id = [item for item in self if item.id == int(id_)]
        if len(items_with_id) == 1:
            return items_with_id[0].retrieve()

def get_substring_idxs(substr, string):
    """
    Return a list of indexes of substr. If substr not found, list is
    empty.

    Arguments:
        substr (str): Substring to match.
        string (str): String to match in.

    Returns:
        list of int: Start indices of substr.
    """
    return [match.start() for match in re.finditer(substr, string)]

def _get_url(url):
    """Retrieve requested URL"""
    try:
        data = HTTP_SESSION.get(url, stream=True)
        data.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise FetcherException(exc)

    return data

def get_last_id(self, cur, table='reaction'):
        """
        Get the id of the last written row in table

        Parameters
        ----------
        cur: database connection().cursor() object
        table: str
            'reaction', 'publication', 'publication_system', 'reaction_system'

        Returns: id
        """
        cur.execute("SELECT seq FROM sqlite_sequence WHERE name='{0}'"
                    .format(table))
        result = cur.fetchone()
        if result is not None:
            id = result[0]
        else:
            id = 0
        return id

def unapostrophe(text):
    """Strip apostrophe and 's' from the end of a string."""
    text = re.sub(r'[%s]s?$' % ''.join(APOSTROPHES), '', text)
    return text

def tail(self, n=10):
        """
        Get an SArray that contains the last n elements in the SArray.

        Parameters
        ----------
        n : int
            The number of elements to fetch

        Returns
        -------
        out : SArray
            A new SArray which contains the last n rows of the current SArray.
        """
        with cython_context():
            return SArray(_proxy=self.__proxy__.tail(n))

def open_as_pillow(filename):
    """ This way can delete file immediately """
    with __sys_open(filename, 'rb') as f:
        data = BytesIO(f.read())
        return Image.open(data)

def get_methods(*objs):
    """ Return the names of all callable attributes of an object"""
    return set(
        attr
        for obj in objs
        for attr in dir(obj)
        if not attr.startswith('_') and callable(getattr(obj, attr))
    )

async def i2c_write_request(self, command):
        """
        This method performs an I2C write at a given I2C address,
        :param command: {"method": "i2c_write_request", "params": [I2C_DEVICE_ADDRESS, [DATA_TO_WRITE]]}
        :returns:No return message.
        """
        device_address = int(command[0])
        params = command[1]
        params = [int(i) for i in params]
        await self.core.i2c_write_request(device_address, params)

def object_as_dict(obj):
    """Turn an SQLAlchemy model into a dict of field names and values.

    Based on https://stackoverflow.com/a/37350445/1579058
    """
    return {c.key: getattr(obj, c.key)
            for c in inspect(obj).mapper.column_attrs}

def has_field(mc, field_name):
    """
    detect if a model has a given field has

    :param field_name:
    :param mc:
    :return:
    """
    try:
        mc._meta.get_field(field_name)
    except FieldDoesNotExist:
        return False
    return True

def call_out(command):
  """
  Run the given command (with shell=False) and return a tuple of
  (int returncode, str output). Strip the output of enclosing whitespace.
  """
  # start external command process
  p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

  # get outputs
  out, _ = p.communicate()

  return p.returncode, out.strip()

def _check_conversion(key, valid_dict):
    """Check for existence of key in dict, return value or raise error"""
    if key not in valid_dict and key not in valid_dict.values():
        # Only show users the nice string values
        keys = [v for v in valid_dict.keys() if isinstance(v, string_types)]
        raise ValueError('value must be one of %s, not %s' % (keys, key))
    return valid_dict[key] if key in valid_dict else key

def GetPythonLibraryDirectoryPath():
  """Retrieves the Python library directory path."""
  path = sysconfig.get_python_lib(True)
  _, _, path = path.rpartition(sysconfig.PREFIX)

  if path.startswith(os.sep):
    path = path[1:]

  return path

def ismatch(text, pattern):
    """Test whether text contains string or matches regex."""

    if hasattr(pattern, 'search'):
        return pattern.search(text) is not None
    else:
        return pattern in text if Config.options.case_sensitive \
            else pattern.lower() in text.lower()

def get_pid_list():
    """Returns a list of PIDs currently running on the system."""
    pids = [int(x) for x in os.listdir('/proc') if x.isdigit()]
    return pids

def file_found(filename,force):
    """Check if a file exists"""
    if os.path.exists(filename) and not force:
        logger.info("Found %s; skipping..."%filename)
        return True
    else:
        return False

def static_get_type_attr(t, name):
    """
    Get a type attribute statically, circumventing the descriptor protocol.
    """
    for type_ in t.mro():
        try:
            return vars(type_)[name]
        except KeyError:
            pass
    raise AttributeError(name)

def starts_with_prefix_in_list(text, prefixes):
    """
    Return True if the given string starts with one of the prefixes in the given list, otherwise
    return False.

    Arguments:
        text (str): Text to check for prefixes.
        prefixes (list): List of prefixes to check for.

    Returns:
        bool: True if the given text starts with any of the given prefixes, otherwise False.
    """
    for prefix in prefixes:
        if text.startswith(prefix):
            return True
    return False

def GetPythonLibraryDirectoryPath():
  """Retrieves the Python library directory path."""
  path = sysconfig.get_python_lib(True)
  _, _, path = path.rpartition(sysconfig.PREFIX)

  if path.startswith(os.sep):
    path = path[1:]

  return path

def from_string(cls, string):
        """
        Simply logs a warning if the desired enum value is not found.

        :param string:
        :return:
        """

        # find enum value
        for attr in dir(cls):
            value = getattr(cls, attr)
            if value == string:
                return value

        # if not found, log warning and return the value passed in
        logger.warning("{} is not a valid enum value for {}.".format(string, cls.__name__))
        return string

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

def _to_numeric(val):
    """
    Helper function for conversion of various data types into numeric representation.
    """
    if isinstance(val, (int, float, datetime.datetime, datetime.timedelta)):
        return val
    return float(val)

def readwav(filename):
    """Read a WAV file and returns the data and sample rate

    ::

        from spectrum.io import readwav
        readwav()

    """
    from scipy.io.wavfile import read as readwav
    samplerate, signal = readwav(filename)
    return signal, samplerate

def get_buffer(self, data_np, header, format, output=None):
        """Get image as a buffer in (format).
        Format should be 'jpeg', 'png', etc.
        """
        if not have_pil:
            raise Exception("Install PIL to use this method")
        image = PILimage.fromarray(data_np)
        buf = output
        if buf is None:
            buf = BytesIO()
        image.save(buf, format)
        return buf

def getScreenDims(self):
        """returns a tuple that contains (screen_width,screen_height)
        """
        width = ale_lib.getScreenWidth(self.obj)
        height = ale_lib.getScreenHeight(self.obj)
        return (width,height)

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

def unique_list_dicts(dlist, key):
    """Return a list of dictionaries which are sorted for only unique entries.

    :param dlist:
    :param key:
    :return list:
    """

    return list(dict((val[key], val) for val in dlist).values())

def region_from_segment(image, segment):
    """given a segment (rectangle) and an image, returns it's corresponding subimage"""
    x, y, w, h = segment
    return image[y:y + h, x:x + w]

def readwav(filename):
    """Read a WAV file and returns the data and sample rate

    ::

        from spectrum.io import readwav
        readwav()

    """
    from scipy.io.wavfile import read as readwav
    samplerate, signal = readwav(filename)
    return signal, samplerate

def setup_path():
    """Sets up the python include paths to include src"""
    import os.path; import sys

    if sys.argv[0]:
        top_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        sys.path = [os.path.join(top_dir, "src")] + sys.path
        pass
    return

def url_read_text(url, verbose=True):
    r"""
    Directly reads text data from url
    """
    data = url_read(url, verbose)
    text = data.decode('utf8')
    return text

def setup_path():
    """Sets up the python include paths to include src"""
    import os.path; import sys

    if sys.argv[0]:
        top_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        sys.path = [os.path.join(top_dir, "src")] + sys.path
        pass
    return

def mean(inlist):
    """
Returns the arithematic mean of the values in the passed list.
Assumes a '1D' list, but will function on the 1st dim of an array(!).

Usage:   lmean(inlist)
"""
    sum = 0
    for item in inlist:
        sum = sum + item
    return sum / float(len(inlist))

def setup_path():
    """Sets up the python include paths to include src"""
    import os.path; import sys

    if sys.argv[0]:
        top_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        sys.path = [os.path.join(top_dir, "src")] + sys.path
        pass
    return

def get_month_start(day=None):
    """Returns the first day of the given month."""
    day = add_timezone(day or datetime.date.today())
    return day.replace(day=1)

def end_block(self):
        """Ends an indentation block, leaving an empty line afterwards"""
        self.current_indent -= 1

        # If we did not add a new line automatically yet, now it's the time!
        if not self.auto_added_line:
            self.writeln()
            self.auto_added_line = True

def find_nearest_index(arr, value):
    """For a given value, the function finds the nearest value
    in the array and returns its index."""
    arr = np.array(arr)
    index = (abs(arr-value)).argmin()
    return index

def end_block(self):
        """Ends an indentation block, leaving an empty line afterwards"""
        self.current_indent -= 1

        # If we did not add a new line automatically yet, now it's the time!
        if not self.auto_added_line:
            self.writeln()
            self.auto_added_line = True

def argsort_indices(a, axis=-1):
    """Like argsort, but returns an index suitable for sorting the
    the original array even if that array is multidimensional
    """
    a = np.asarray(a)
    ind = list(np.ix_(*[np.arange(d) for d in a.shape]))
    ind[axis] = a.argsort(axis)
    return tuple(ind)

def _remove_from_index(index, obj):
    """Removes object ``obj`` from the ``index``."""
    try:
        index.value_map[indexed_value(index, obj)].remove(obj.id)
    except KeyError:
        pass

def hstrlen(self, name, key):
        """
        Return the number of bytes stored in the value of ``key``
        within hash ``name``
        """
        with self.pipe as pipe:
            return pipe.hstrlen(self.redis_key(name), key)

def get_index_nested(x, i):
    """
    Description:
        Returns the first index of the array (vector) x containing the value i.
    Parameters:
        x: one-dimensional array
        i: search value
    """
    for ind in range(len(x)):
        if i == x[ind]:
            return ind
    return -1

def _get_column_by_db_name(cls, name):
        """
        Returns the column, mapped by db_field name
        """
        return cls._columns.get(cls._db_map.get(name, name))

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

def get_previous_month(self):
        """Returns date range for the previous full month."""
        end = utils.get_month_start() - relativedelta(days=1)
        end = utils.to_datetime(end)
        start = utils.get_month_start(end)
        return start, end

def listified_tokenizer(source):
    """Tokenizes *source* and returns the tokens as a list of lists."""
    io_obj = io.StringIO(source)
    return [list(a) for a in tokenize.generate_tokens(io_obj.readline)]

def ex(self, cmd):
        """Execute a normal python statement in user namespace."""
        with self.builtin_trap:
            exec cmd in self.user_global_ns, self.user_ns

def unique_element(ll):
    """ returns unique elements from a list preserving the original order """
    seen = {}
    result = []
    for item in ll:
        if item in seen:
            continue
        seen[item] = 1
        result.append(item)
    return result

def prepend_line(filepath, line):
    """Rewrite a file adding a line to its beginning.
    """
    with open(filepath) as f:
        lines = f.readlines()

    lines.insert(0, line)

    with open(filepath, 'w') as f:
        f.writelines(lines)

def _get_device_id(self, bus):
        """
        Find the device id
        """
        _dbus = bus.get(SERVICE_BUS, PATH)
        devices = _dbus.devices()

        if self.device is None and self.device_id is None and len(devices) == 1:
            return devices[0]

        for id in devices:
            self._dev = bus.get(SERVICE_BUS, DEVICE_PATH + "/%s" % id)
            if self.device == self._dev.name:
                return id

        return None

def add_parent(self, parent):
        """
        Adds self as child of parent, then adds parent.
        """
        parent.add_child(self)
        self.parent = parent
        return parent

def print(*a):
    """ print just one that returns what you give it instead of None """
    try:
        _print(*a)
        return a[0] if len(a) == 1 else a
    except:
        _print(*a)

def add_noise(Y, sigma):
    """Adds noise to Y"""
    return Y + np.random.normal(0, sigma, Y.shape)

def plot_target(target, ax):
    """Ajoute la target au plot"""
    ax.scatter(target[0], target[1], target[2], c="red", s=80)

def props(cls):
    """
    Class method that returns all defined arguments within the class.
    
    Returns:
      A dictionary containing all action defined arguments (if any).
    """
    return {k:v for (k, v) in inspect.getmembers(cls) if type(v) is Argument}

def are_in_interval(s, l, r, border = 'included'):
        """
        Checks whether all number in the sequence s lie inside the interval formed by
        l and r.
        """
        return numpy.all([IntensityRangeStandardization.is_in_interval(x, l, r, border) for x in s])

def get_public_members(obj):
    """
    Retrieves a list of member-like objects (members or properties) that are
    publically exposed.

    :param obj: The object to probe.
    :return:    A list of strings.
    """
    return {attr: getattr(obj, attr) for attr in dir(obj)
            if not attr.startswith("_")
            and not hasattr(getattr(obj, attr), '__call__')}

def format_time(time):
    """ Formats the given time into HH:MM:SS """
    h, r = divmod(time / 1000, 3600)
    m, s = divmod(r, 60)

    return "%02d:%02d:%02d" % (h, m, s)

def extract_vars_above(*names):
    """Extract a set of variables by name from another frame.

    Similar to extractVars(), but with a specified depth of 1, so that names
    are exctracted exactly from above the caller.

    This is simply a convenience function so that the very common case (for us)
    of skipping exactly 1 frame doesn't have to construct a special dict for
    keyword passing."""

    callerNS = sys._getframe(2).f_locals
    return dict((k,callerNS[k]) for k in names)

def b(s):
	""" Encodes Unicode strings to byte strings, if necessary. """

	return s if isinstance(s, bytes) else s.encode(locale.getpreferredencoding())

def from_dict(cls, d):
        """Create an instance from a dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.ENTRIES})

def go_to_parent_directory(self):
        """Go to parent directory"""
        self.chdir(osp.abspath(osp.join(getcwd_or_home(), os.pardir)))

def spline_interpolate_by_datetime(datetime_axis, y_axis, datetime_new_axis):
    """A datetime-version that takes datetime object list as x_axis
    """
    numeric_datetime_axis = [
        totimestamp(a_datetime) for a_datetime in datetime_axis
    ]

    numeric_datetime_new_axis = [
        totimestamp(a_datetime) for a_datetime in datetime_new_axis
    ]

    return spline_interpolate(
        numeric_datetime_axis, y_axis, numeric_datetime_new_axis)

def gettext(self, string, domain=None, **variables):
        """Translate a string with the current locale."""
        t = self.get_translations(domain)
        return t.ugettext(string) % variables

def MatrixInverse(a, adj):
    """
    Matrix inversion op.
    """
    return np.linalg.inv(a if not adj else _adjoint(a)),

def _get_local_ip():
        """
        Get the local ip of this device

        :return: Ip of this computer
        :rtype: str
        """
        return set([x[4][0] for x in socket.getaddrinfo(
            socket.gethostname(),
            80,
            socket.AF_INET
        )]).pop()

def __call__(self, args):
        """Execute the user function."""
        window, ij = args
        return self.user_func(srcs, window, ij, global_args), window

def l2_norm(params):
    """Computes l2 norm of params by flattening them into a vector."""
    flattened, _ = flatten(params)
    return np.dot(flattened, flattened)

def _not_none(items):
    """Whether the item is a placeholder or contains a placeholder."""
    if not isinstance(items, (tuple, list)):
        items = (items,)
    return all(item is not _none for item in items)

def connect(self, A, B, distance=1):
        """Add a link from A and B of given distance, and also add the inverse
        link if the graph is undirected."""
        self.connect1(A, B, distance)
        if not self.directed: self.connect1(B, A, distance)

def url_read_text(url, verbose=True):
    r"""
    Directly reads text data from url
    """
    data = url_read(url, verbose)
    text = data.decode('utf8')
    return text

def draw_graph(G: nx.DiGraph, filename: str):
    """ Draw a networkx graph with Pygraphviz. """
    A = to_agraph(G)
    A.graph_attr["rankdir"] = "LR"
    A.draw(filename, prog="dot")

def check_int(integer):
    """
    Check if number is integer or not.

    :param integer: Number as str
    :return: Boolean
    """
    if not isinstance(integer, str):
        return False
    if integer[0] in ('-', '+'):
        return integer[1:].isdigit()
    return integer.isdigit()

def search_script_directory(self, path):
        """
        Recursively loop through a directory to find all python
        script files. When one is found, it is analyzed for import statements
        :param path: string
        :return: generator
        """
        for subdir, dirs, files in os.walk(path):
            for file_name in files:
                if file_name.endswith(".py"):
                    self.search_script_file(subdir, file_name)

def on_windows ():
    """ Returns true if running on windows, whether in cygwin or not.
    """
    if bjam.variable("NT"):
        return True

    elif bjam.variable("UNIX"):

        uname = bjam.variable("JAMUNAME")
        if uname and uname[0].startswith("CYGWIN"):
            return True

    return False

def quote(self, s):
        """Return a shell-escaped version of the string s."""

        if six.PY2:
            from pipes import quote
        else:
            from shlex import quote

        return quote(s)

def parse_timestamp(timestamp):
    """Parse ISO8601 timestamps given by github API."""
    dt = dateutil.parser.parse(timestamp)
    return dt.astimezone(dateutil.tz.tzutc())

def end_table_header(self):
        r"""End the table header which will appear on every page."""

        if self.header:
            msg = "Table already has a header"
            raise TableError(msg)

        self.header = True

        self.append(Command(r'endhead'))

def __iter__(self):
        """
        Iterate through tree, leaves first

        following http://stackoverflow.com/questions/6914803/python-iterator-through-tree-with-list-of-children
        """
        for node in chain(*imap(iter, self.children)):
            yield node
        yield self

def is_symbol(string):
    """
    Return true if the string is a mathematical symbol.
    """
    return (
        is_int(string) or is_float(string) or
        is_constant(string) or is_unary(string) or
        is_binary(string) or
        (string == '(') or (string == ')')
    )

def tab(self, output):
        """Output data in excel-compatible tab-delimited format"""
        import csv
        csvwriter = csv.writer(self.outfile, dialect=csv.excel_tab)
        csvwriter.writerows(output)

def _skip_frame(self):
        """Skip the next time frame"""
        for line in self._f:
            if line == 'ITEM: ATOMS\n':
                break
        for i in range(self.num_atoms):
            next(self._f)

def _change_height(self, ax, new_value):
        """Make bars in horizontal bar chart thinner"""
        for patch in ax.patches:
            current_height = patch.get_height()
            diff = current_height - new_value

            # we change the bar height
            patch.set_height(new_value)

            # we recenter the bar
            patch.set_y(patch.get_y() + diff * .5)

def json_iter (path):
    """
    iterator for JSON-per-line in a file pattern
    """
    with open(path, 'r') as f:
        for line in f.readlines():
            yield json.loads(line)

def _clean_up_name(self, name):
        """
        Cleans up the name according to the rules specified in this exact
        function. Uses self.naughty, a list of naughty characters.
        """
        for n in self.naughty: name = name.replace(n, '_')
        return name

def group_by(iterable, key_func):
    """Wrap itertools.groupby to make life easier."""
    groups = (
        list(sub) for key, sub in groupby(iterable, key_func)
    )
    return zip(groups, groups)

def round_to_n(x, n):
    """
    Round to sig figs
    """
    return round(x, -int(np.floor(np.log10(x))) + (n - 1))

def next (self):    # File-like object.

        """This is to support iterators over a file-like object.
        """

        result = self.readline()
        if result == self._empty_buffer:
            raise StopIteration
        return result

def intround(value):
    """Given a float returns a rounded int. Should give the same result on
    both Py2/3
    """

    return int(decimal.Decimal.from_float(
        value).to_integral_value(decimal.ROUND_HALF_EVEN))

def reset(self):
		"""
		Resets the iterator to the start.

		Any remaining values in the current iteration are discarded.
		"""
		self.__iterator, self.__saved = itertools.tee(self.__saved)

def web(host, port):
    """Start web application"""
    from .webserver.web import get_app
    get_app().run(host=host, port=port)

def assert_single_element(iterable):
  """Get the single element of `iterable`, or raise an error.

  :raise: :class:`StopIteration` if there is no element.
  :raise: :class:`ValueError` if there is more than one element.
  """
  it = iter(iterable)
  first_item = next(it)

  try:
    next(it)
  except StopIteration:
    return first_item

  raise ValueError("iterable {!r} has more than one element.".format(iterable))

def strToBool(val):
    """
    Helper function to turn a string representation of "true" into
    boolean True.
    """
    if isinstance(val, str):
        val = val.lower()

    return val in ['true', 'on', 'yes', True]

def _fill(self):
    """Advance the iterator without returning the old head."""
    try:
      self._head = self._iterable.next()
    except StopIteration:
      self._head = None

def __get_xml_text(root):
    """ Return the text for the given root node (xml.dom.minidom). """
    txt = ""
    for e in root.childNodes:
        if (e.nodeType == e.TEXT_NODE):
            txt += e.data
    return txt

def iterparse(source, events=('end',), remove_comments=True, **kw):
    """Thin wrapper around ElementTree.iterparse"""
    return ElementTree.iterparse(source, events, SourceLineParser(), **kw)

def get_encoding(binary):
    """Return the encoding type."""

    try:
        from chardet import detect
    except ImportError:
        LOGGER.error("Please install the 'chardet' module")
        sys.exit(1)

    encoding = detect(binary).get('encoding')

    return 'iso-8859-1' if encoding == 'CP949' else encoding

def iter_with_last(iterable):
    """
    :return: generator of tuples (isLastFlag, item)
    """
    # Ensure it's an iterator and get the first field
    iterable = iter(iterable)
    prev = next(iterable)
    for item in iterable:
        # Lag by one item so I know I'm not at the end
        yield False, prev
        prev = item
    # Last item
    yield True, prev

def conv_dict(self):
        """dictionary of conversion"""
        return dict(integer=self.integer, real=self.real, no_type=self.no_type)

def _assert_is_type(name, value, value_type):
    """Assert that a value must be a given type."""
    if not isinstance(value, value_type):
        if type(value_type) is tuple:
            types = ', '.join(t.__name__ for t in value_type)
            raise ValueError('{0} must be one of ({1})'.format(name, types))
        else:
            raise ValueError('{0} must be {1}'
                             .format(name, value_type.__name__))

def help_for_command(command):
    """Get the help text (signature + docstring) for a command (function)."""
    help_text = pydoc.text.document(command)
    # remove backspaces
    return re.subn('.\\x08', '', help_text)[0]

def render_template(env, filename, values=None):
    """
    Render a jinja template
    """
    if not values:
        values = {}
    tmpl = env.get_template(filename)
    return tmpl.render(values)

def get_absolute_path(*args):
    """Transform relative pathnames into absolute pathnames."""
    directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(directory, *args)

def render_template(template_name, **context):
    """Render a template into a response."""
    tmpl = jinja_env.get_template(template_name)
    context["url_for"] = url_for
    return Response(tmpl.render(context), mimetype="text/html")

def execfile(fname, variables):
    """ This is builtin in python2, but we have to roll our own on py3. """
    with open(fname) as f:
        code = compile(f.read(), fname, 'exec')
        exec(code, variables)

async def join(self, ctx, *, channel: discord.VoiceChannel):
        """Joins a voice channel"""

        if ctx.voice_client is not None:
            return await ctx.voice_client.move_to(channel)

        await channel.connect()

def gaussian_variogram_model(m, d):
    """Gaussian model, m is [psill, range, nugget]"""
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return psill * (1. - np.exp(-d**2./(range_*4./7.)**2.)) + nugget

def _grammatical_join_filter(l, arg=None):
    """
    :param l: List of strings to join
    :param arg: A pipe-separated list of final_join (" and ") and
    initial_join (", ") strings. For example
    :return: A string that grammatically concatenates the items in the list.
    """
    if not arg:
        arg = " and |, "
    try:
        final_join, initial_joins = arg.split("|")
    except ValueError:
        final_join = arg
        initial_joins = ", "
    return grammatical_join(l, initial_joins, final_join)

def fast_exit(code):
    """Exit without garbage collection, this speeds up exit by about 10ms for
    things like bash completion.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)

def flatten_dict_join_keys(dct, join_symbol=" "):
    """ Flatten dict with defined key join symbol.

    :param dct: dict to flatten
    :param join_symbol: default value is " "
    :return:
    """
    return dict( flatten_dict(dct, join=lambda a,b:a+join_symbol+b) )

def get_unique_indices(df, axis=1):
    """

    :param df:
    :param axis:
    :return:
    """
    return dict(zip(df.columns.names, dif.columns.levels))

def flatten_dict_join_keys(dct, join_symbol=" "):
    """ Flatten dict with defined key join symbol.

    :param dct: dict to flatten
    :param join_symbol: default value is " "
    :return:
    """
    return dict( flatten_dict(dct, join=lambda a,b:a+join_symbol+b) )

def lcumsum (inlist):
    """
Returns a list consisting of the cumulative sum of the items in the
passed list.

Usage:   lcumsum(inlist)
"""
    newlist = copy.deepcopy(inlist)
    for i in range(1,len(newlist)):
        newlist[i] = newlist[i] + newlist[i-1]
    return newlist

def get_join_cols(by_entry):
  """ helper function used for joins
  builds left and right join list for join function
  """
  left_cols = []
  right_cols = []
  for col in by_entry:
    if isinstance(col, str):
      left_cols.append(col)
      right_cols.append(col)
    else:
      left_cols.append(col[0])
      right_cols.append(col[1])
  return left_cols, right_cols

def _comment(string):
    """return string as a comment"""
    lines = [line.strip() for line in string.splitlines()]
    return "# " + ("%s# " % linesep).join(lines)

def _default(self, obj):
        """ return a serialized version of obj or raise a TypeError

        :param obj:
        :return: Serialized version of obj
        """
        return obj.__dict__ if isinstance(obj, JsonObj) else json.JSONDecoder().decode(obj)

def unique(iterable):
    """ Returns a list copy in which each item occurs only once (in-order).
    """
    seen = set()
    return [x for x in iterable if x not in seen and not seen.add(x)]

def json_dumps(self, obj):
        """Serializer for consistency"""
        return json.dumps(obj, sort_keys=True, indent=4, separators=(',', ': '))

def join(mapping, bind, values):
    """ Merge all the strings. Put space between them. """
    return [' '.join([six.text_type(v) for v in values if v is not None])]

def json_dumps(self, obj):
        """Serializer for consistency"""
        return json.dumps(obj, sort_keys=True, indent=4, separators=(',', ': '))

def kill(self):
        """Kill the browser.

        This is useful when the browser is stuck.
        """
        if self.process:
            self.process.kill()
            self.process.wait()

def to_json(obj):
    """Return a json string representing the python object obj."""
    i = StringIO.StringIO()
    w = Writer(i, encoding='UTF-8')
    w.write_value(obj)
    return i.getvalue()

def retrieve_by_id(self, id_):
        """Return a JSSObject for the element with ID id_"""
        items_with_id = [item for item in self if item.id == int(id_)]
        if len(items_with_id) == 1:
            return items_with_id[0].retrieve()

def parse_json(filename):
    """ Parse a JSON file
        First remove comments and then use the json module package
        Comments look like :
            // ...
        or
            /*
            ...
            */
    """
    # Regular expression for comments
    comment_re = re.compile(
        '(^)?[^\S\n]*/(?:\*(.*?)\*/[^\S\n]*|/[^\n]*)($)?',
        re.DOTALL | re.MULTILINE
    )

    with open(filename) as f:
        content = ''.join(f.readlines())

        ## Looking for comments
        match = comment_re.search(content)
        while match:
            # single line comment
            content = content[:match.start()] + content[match.end():]
            match = comment_re.search(content)

        # Return json file
        return json.loads(content)

def A(*a):
    """convert iterable object into numpy array"""
    return np.array(a[0]) if len(a)==1 else [np.array(o) for o in a]

def loadb(b):
    """Deserialize ``b`` (instance of ``bytes``) to a Python object."""
    assert isinstance(b, (bytes, bytearray))
    return std_json.loads(b.decode('utf-8'))

def next (self):    # File-like object.

        """This is to support iterators over a file-like object.
        """

        result = self.readline()
        if result == self._empty_buffer:
            raise StopIteration
        return result

def _time_to_json(value):
    """Coerce 'value' to an JSON-compatible representation."""
    if isinstance(value, datetime.time):
        value = value.isoformat()
    return value

def setup():
    """Setup pins"""
    print("Simple drive")
    board.set_pin_mode(L_CTRL_1, Constants.OUTPUT)
    board.set_pin_mode(L_CTRL_2, Constants.OUTPUT)
    board.set_pin_mode(PWM_L, Constants.PWM)
    board.set_pin_mode(R_CTRL_1, Constants.OUTPUT)
    board.set_pin_mode(R_CTRL_2, Constants.OUTPUT)
    board.set_pin_mode(PWM_R, Constants.PWM)

def task_property_present_predicate(service, task, prop):
    """ True if the json_element passed is present for the task specified.
    """
    try:
        response = get_service_task(service, task)
    except Exception as e:
        pass

    return (response is not None) and (prop in response)

def set_color(self, fg=None, bg=None, intensify=False, target=sys.stdout):
        """Set foreground- and background colors and intensity."""
        raise NotImplementedError

def validate(payload, schema):
    """Validate `payload` against `schema`, returning an error list.

    jsonschema provides lots of information in it's errors, but it can be a bit
    of work to extract all the information.
    """
    v = jsonschema.Draft4Validator(
        schema, format_checker=jsonschema.FormatChecker())
    error_list = []
    for error in v.iter_errors(payload):
        message = error.message
        location = '/' + '/'.join([str(c) for c in error.absolute_path])
        error_list.append(message + ' at ' + location)
    return error_list

def normal_noise(points):
    """Init a noise variable."""
    return np.random.rand(1) * np.random.randn(points, 1) \
        + random.sample([2, -2], 1)

def go_to_line(self, line):
        """
        Moves the text cursor to given line.

        :param line: Line to go to.
        :type line: int
        :return: Method success.
        :rtype: bool
        """

        cursor = self.textCursor()
        cursor.setPosition(self.document().findBlockByNumber(line - 1).position())
        self.setTextCursor(cursor)
        return True

def _write_color_ansi (fp, text, color):
    """Colorize text with given color."""
    fp.write(esc_ansicolor(color))
    fp.write(text)
    fp.write(AnsiReset)

def drop_trailing_zeros(num):
    """
    Drops the trailing zeros in a float that is printed.
    """
    txt = '%f' %(num)
    txt = txt.rstrip('0')
    if txt.endswith('.'):
        txt = txt[:-1]
    return txt

def weekly(date=datetime.date.today()):
    """
    Weeks start are fixes at Monday for now.
    """
    return date - datetime.timedelta(days=date.weekday())

def kill_process_children(pid):
    """Find and kill child processes of a process.

    :param pid: PID of parent process (process ID)
    :return: Nothing
    """
    if sys.platform == "darwin":
        kill_process_children_osx(pid)
    elif sys.platform == "linux":
        kill_process_children_unix(pid)
    else:
        pass

def setLib(self, lib):
        """ Copy the lib items into our font. """
        for name, item in lib.items():
            self.font.lib[name] = item

def kill_process(process):
    """Kill the process group associated with the given process. (posix)"""
    logger = logging.getLogger('xenon')
    logger.info('Terminating Xenon-GRPC server.')
    os.kill(process.pid, signal.SIGINT)
    process.wait()

def __init__(self, filename, mode, encoding=None):
        """Use the specified filename for streamed logging."""
        FileHandler.__init__(self, filename, mode, encoding)
        self.mode = mode
        self.encoding = encoding

def kill_process(process):
    """Kill the process group associated with the given process. (posix)"""
    logger = logging.getLogger('xenon')
    logger.info('Terminating Xenon-GRPC server.')
    os.kill(process.pid, signal.SIGINT)
    process.wait()

def install_postgres(user=None, dbname=None, password=None):
    """Install Postgres on remote"""
    execute(pydiploy.django.install_postgres_server,
            user=user, dbname=dbname, password=password)

def l2_norm(arr):
    """
    The l2 norm of an array is is defined as: sqrt(||x||), where ||x|| is the
    dot product of the vector.
    """
    arr = np.asarray(arr)
    return np.sqrt(np.dot(arr.ravel().squeeze(), arr.ravel().squeeze()))

def register_plugin(self):
        """Register plugin in Spyder's main window"""
        self.main.restore_scrollbar_position.connect(
                                               self.restore_scrollbar_position)
        self.main.add_dockwidget(self)

def l2_norm(arr):
    """
    The l2 norm of an array is is defined as: sqrt(||x||), where ||x|| is the
    dot product of the vector.
    """
    arr = np.asarray(arr)
    return np.sqrt(np.dot(arr.ravel().squeeze(), arr.ravel().squeeze()))

def append_text(self, txt):
        """ adds a line of text to a file """
        with open(self.fullname, "a") as myfile:
            myfile.write(txt)

def download_file_from_bucket(self, bucket, file_path, key):
        """ Download file from S3 Bucket """
        with open(file_path, 'wb') as data:
            self.__s3.download_fileobj(bucket, key, data)
            return file_path

def _pad(self, text):
        """Pad the text."""
        top_bottom = ("\n" * self._padding) + " "
        right_left = " " * self._padding * self.PAD_WIDTH
        return top_bottom + right_left + text + right_left + top_bottom

def make_lambda(call):
    """Wrap an AST Call node to lambda expression node.
    call: ast.Call node
    """
    empty_args = ast.arguments(args=[], vararg=None, kwarg=None, defaults=[])
    return ast.Lambda(args=empty_args, body=call)

def robust_int(v):
    """Parse an int robustly, ignoring commas and other cruft. """

    if isinstance(v, int):
        return v

    if isinstance(v, float):
        return int(v)

    v = str(v).replace(',', '')

    if not v:
        return None

    return int(v)

def get_tail(self):
        """Gets tail

        :return: Tail of linked list
        """
        node = self.head
        last_node = self.head

        while node is not None:
            last_node = node
            node = node.next_node

        return last_node

def AddAccuracy(model, softmax, label):
    """Adds an accuracy op to the model"""
    accuracy = brew.accuracy(model, [softmax, label], "accuracy")
    return accuracy

def end_index(self):
        """Return the 1-based index of the last item on this page."""
        paginator = self.paginator
        # Special case for the last page because there can be orphans.
        if self.number == paginator.num_pages:
            return paginator.count
        return (self.number - 1) * paginator.per_page + paginator.first_page

def append(self, item):
        """ append item and print it to stdout """
        print(item)
        super(MyList, self).append(item)

def xyz2lonlat(x, y, z):
    """Convert cartesian to lon lat."""
    lon = xu.rad2deg(xu.arctan2(y, x))
    lat = xu.rad2deg(xu.arctan2(z, xu.sqrt(x**2 + y**2)))
    return lon, lat

def filter_contour(imageFile, opFile):
    """ convert an image by applying a contour """
    im = Image.open(imageFile)
    im1 = im.filter(ImageFilter.CONTOUR)
    im1.save(opFile)

def make_coord_dict(coord):
    """helper function to make a dict from a coordinate for logging"""
    return dict(
        z=int_if_exact(coord.zoom),
        x=int_if_exact(coord.column),
        y=int_if_exact(coord.row),
    )

def dictapply(d, fn):
    """
    apply a function to all non-dict values in a dictionary
    """
    for k, v in d.items():
        if isinstance(v, dict):
            v = dictapply(v, fn)
        else:
            d[k] = fn(v)
    return d

def tree_predict(x, root, proba=False, regression=False):
    """Predicts a probabilities/value/label for the sample x.
    """

    if isinstance(root, Leaf):
        if proba:
            return root.probabilities
        elif regression:
            return root.mean
        else:
            return root.most_frequent

    if root.question.match(x):
        return tree_predict(x, root.true_branch, proba=proba, regression=regression)
    else:
        return tree_predict(x, root.false_branch, proba=proba, regression=regression)

def load_files(files):
    """Load and execute a python file."""

    for py_file in files:
        LOG.debug("exec %s", py_file)
        execfile(py_file, globals(), locals())

def minimise_xyz(xyz):
    """Minimise an (x, y, z) coordinate."""
    x, y, z = xyz
    m = max(min(x, y), min(max(x, y), z))
    return (x-m, y-m, z-m)

def download_file(save_path, file_url):
    """ Download file from http url link """

    r = requests.get(file_url)  # create HTTP response object

    with open(save_path, 'wb') as f:
        f.write(r.content)

    return save_path

def stop(pid):
    """Shut down a specific process.

    Args:
      pid: the pid of the process to shutdown.
    """
    if psutil.pid_exists(pid):
      try:
        p = psutil.Process(pid)
        p.kill()
      except Exception:
        pass

def variance(arr):
  """variance of the values, must have 2 or more entries.

  :param arr: list of numbers
  :type arr: number[] a number array
  :return: variance
  :rtype: float

  """
  avg = average(arr)
  return sum([(float(x)-avg)**2 for x in arr])/float(len(arr)-1)

def dedupe_list(seq):
    """
    Utility function to remove duplicates from a list
    :param seq: The sequence (list) to deduplicate
    :return: A list with original duplicates removed
    """
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]

def csort(objs, key):
    """Order-preserving sorting function."""
    idxs = dict((obj, i) for (i, obj) in enumerate(objs))
    return sorted(objs, key=lambda obj: (key(obj), idxs[obj]))

def get_by(self, name):
    """get element by name"""
    return next((item for item in self if item.name == name), None)

def should_be_hidden_as_cause(exc):
    """ Used everywhere to decide if some exception type should be displayed or hidden as the casue of an error """
    # reduced traceback in case of HasWrongType (instance_of checks)
    from valid8.validation_lib.types import HasWrongType, IsWrongType
    return isinstance(exc, (HasWrongType, IsWrongType))

def _calculate_distance(latlon1, latlon2):
    """Calculates the distance between two points on earth.
    """
    lat1, lon1 = latlon1
    lat2, lon2 = latlon2
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    R = 6371  # radius of the earth in kilometers
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon / 2))**2
    c = 2 * np.pi * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) / 180
    return c

def items(cls):
        """
        All values for this enum
        :return: list of tuples

        """
        return [
            cls.PRECIPITATION,
            cls.WIND,
            cls.TEMPERATURE,
            cls.PRESSURE
        ]

def dot_v3(v, w):
    """Return the dotproduct of two vectors."""

    return sum([x * y for x, y in zip(v, w)])

def from_json_list(cls, api_client, data):
        """Convert a list of JSON values to a list of models
        """
        return [cls.from_json(api_client, item) for item in data]

def center_eigenvalue_diff(mat):
    """Compute the eigvals of mat and then find the center eigval difference."""
    N = len(mat)
    evals = np.sort(la.eigvals(mat))
    diff = np.abs(evals[N/2] - evals[N/2-1])
    return diff

def getPrimeFactors(n):
    """
    Get all the prime factor of given integer
    @param n integer
    @return list [1, ..., n]
    """
    lo = [1]
    n2 = n // 2
    k = 2
    for k in range(2, n2 + 1):
        if (n // k)*k == n:
            lo.append(k)
    return lo + [n, ]

def euclidean(c1, c2):
    """Square of the euclidean distance"""
    diffs = ((i - j) for i, j in zip(c1, c2))
    return sum(x * x for x in diffs)

def chunk_list(l, n):
    """Return `n` size lists from a given list `l`"""
    return [l[i:i + n] for i in range(0, len(l), n)]

def mag(z):
    """Get the magnitude of a vector."""
    if isinstance(z[0], np.ndarray):
        return np.array(list(map(np.linalg.norm, z)))
    else:
        return np.linalg.norm(z)

def dump_nparray(self, obj, class_name=numpy_ndarray_class_name):
        """
        ``numpy.ndarray`` dumper.
        """
        return {"$" + class_name: self._json_convert(obj.tolist())}

def add_to_js(self, name, var):
        """Add an object to Javascript."""
        frame = self.page().mainFrame()
        frame.addToJavaScriptWindowObject(name, var)

def list2string (inlist,delimit=' '):
    """
Converts a 1D list to a single long string for file output, using
the string.join function.

Usage:   list2string (inlist,delimit=' ')
Returns: the string created from inlist
"""
    stringlist = [makestr(_) for _ in inlist]
    return string.join(stringlist,delimit)

def _call(callable_obj, arg_names, namespace):
    """Actually calls the callable with the namespace parsed from the command
    line.

    Args:
        callable_obj: a callable object
        arg_names: name of the function arguments
        namespace: the namespace object parsed from the command line
    """
    arguments = {arg_name: getattr(namespace, arg_name)
                 for arg_name in arg_names}
    return callable_obj(**arguments)

def from_json_str(cls, json_str):
    """Convert json string representation into class instance.

    Args:
      json_str: json representation as string.

    Returns:
      New instance of the class with data loaded from json string.
    """
    return cls.from_json(json.loads(json_str, cls=JsonDecoder))

def eval_script(self, expr):
    """ Evaluates a piece of Javascript in the context of the current page and
    returns its value. """
    ret = self.conn.issue_command("Evaluate", expr)
    return json.loads("[%s]" % ret)[0]

def main(args=sys.argv):
    """
    main entry point for the jardiff CLI
    """

    parser = create_optparser(args[0])
    return cli(parser.parse_args(args[1:]))

def fn_min(self, a, axis=None):
        """
        Return the minimum of an array, ignoring any NaNs.

        :param a: The array.
        :return: The minimum value of the array.
        """

        return numpy.nanmin(self._to_ndarray(a), axis=axis)

def load(obj, cls, default_factory):
    """Create or load an object if necessary.

    Parameters
    ----------
    obj : `object` or `dict` or `None`
    cls : `type`
    default_factory : `function`

    Returns
    -------
    `object`
    """
    if obj is None:
        return default_factory()
    if isinstance(obj, dict):
        return cls.load(obj)
    return obj

def datetime_from_str(string):
    """

    Args:
        string: string of the form YYMMDD-HH_MM_SS, e.g 160930-18_43_01

    Returns: a datetime object

    """


    return datetime.datetime(year=2000+int(string[0:2]), month=int(string[2:4]), day=int(string[4:6]), hour=int(string[7:9]), minute=int(string[10:12]),second=int(string[13:15]))

def open_json(file_name):
    """
    returns json contents as string
    """
    with open(file_name, "r") as json_data:
        data = json.load(json_data)
        return data

def mixedcase(path):
    """Removes underscores and capitalizes the neighbouring character"""
    words = path.split('_')
    return words[0] + ''.join(word.title() for word in words[1:])

def load(cls, fname):
        """
        Loads the flow from a JSON file.

        :param fname: the file to load
        :type fname: str
        :return: the flow
        :rtype: Flow
        """
        with open(fname) as f:
            content = f.readlines()
        return Flow.from_json(''.join(content))

def camel_case_from_underscores(string):
    """generate a CamelCase string from an underscore_string."""
    components = string.split('_')
    string = ''
    for component in components:
        string += component[0].upper() + component[1:]
    return string

def load_graph_from_rdf(fname):
    """ reads an RDF file into a graph """
    print("reading RDF from " + fname + "....")
    store = Graph()
    store.parse(fname, format="n3")
    print("Loaded " + str(len(store)) + " tuples")
    return store

def mixedcase(path):
    """Removes underscores and capitalizes the neighbouring character"""
    words = path.split('_')
    return words[0] + ''.join(word.title() for word in words[1:])

def read_raw(data_path):
    """
    Parameters
    ----------
    data_path : str
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data

def mixedcase(path):
    """Removes underscores and capitalizes the neighbouring character"""
    words = path.split('_')
    return words[0] + ''.join(word.title() for word in words[1:])

def bisect_index(a, x):
    """ Find the leftmost index of an element in a list using binary search.

    Parameters
    ----------
    a: list
        A sorted list.
    x: arbitrary
        The element.

    Returns
    -------
    int
        The index.

    """
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError

def to_binary(s, encoding='utf8'):
    """Portable cast function.

    In python 2 the ``str`` function which is used to coerce objects to bytes does not
    accept an encoding argument, whereas python 3's ``bytes`` function requires one.

    :param s: object to be converted to binary_type
    :return: binary_type instance, representing s.
    """
    if PY3:  # pragma: no cover
        return s if isinstance(s, binary_type) else binary_type(s, encoding=encoding)
    return binary_type(s)

def find_console_handler(logger):
    """Return a stream handler, if it exists."""
    for handler in logger.handlers:
        if (isinstance(handler, logging.StreamHandler) and
                handler.stream == sys.stderr):
            return handler

def _datetime_to_date(arg):
    """
    convert datetime/str to date
    :param arg:
    :return:
    """
    _arg = parse(arg)
    if isinstance(_arg, datetime.datetime):
        _arg = _arg.date()
    return _arg

def __enter__(self):
        """Enable the download log filter."""
        self.logger = logging.getLogger('pip.download')
        self.logger.addFilter(self)

def update_dict(obj, dict, attributes):
    """Update dict with fields from obj.attributes.

    :param obj: the object updated into dict
    :param dict: the result dictionary
    :param attributes: a list of attributes belonging to obj
    """
    for attribute in attributes:
        if hasattr(obj, attribute) and getattr(obj, attribute) is not None:
            dict[attribute] = getattr(obj, attribute)

def clog(color):
    """Same to ``log``, but this one centralizes the message first."""
    logger = log(color)
    return lambda msg: logger(centralize(msg).rstrip())

def _normalize_numpy_indices(i):
    """Normalize the index in case it is a numpy integer or boolean
    array."""
    if isinstance(i, np.ndarray):
        if i.dtype == bool:
            i = tuple(j.tolist() for j in i.nonzero())
        elif i.dtype == int:
            i = i.tolist()
    return i

def format(self, record, *args, **kwargs):
        """
        Format a message in the log

        Act like the normal format, but indent anything that is a
        newline within the message.

        """
        return logging.Formatter.format(
            self, record, *args, **kwargs).replace('\n', '\n' + ' ' * 8)

def convert_types(cls, value):
        """
        Takes a value from MSSQL, and converts it to a value that's safe for
        JSON/Google Cloud Storage/BigQuery.
        """
        if isinstance(value, decimal.Decimal):
            return float(value)
        else:
            return value

def _get_loggers():
    """Return list of Logger classes."""
    from .. import loader
    modules = loader.get_package_modules('logger')
    return list(loader.get_plugins(modules, [_Logger]))

def log_no_newline(self, msg):
      """ print the message to the predefined log file without newline """
      self.print2file(self.logfile, False, False, msg)

def format_time(time):
    """ Formats the given time into HH:MM:SS """
    h, r = divmod(time / 1000, 3600)
    m, s = divmod(r, 60)

    return "%02d:%02d:%02d" % (h, m, s)

def debug(self, text):
		""" Ajout d'un message de log de type DEBUG """
		self.logger.debug("{}{}".format(self.message_prefix, text))

def as_tuple(self, value):
        """Utility function which converts lists to tuples."""
        if isinstance(value, list):
            value = tuple(value)
        return value

def is_bool_matrix(l):
    r"""Checks if l is a 2D numpy array of bools

    """
    if isinstance(l, np.ndarray):
        if l.ndim == 2 and (l.dtype == bool):
            return True
    return False

def set_index(self, index):
        """ Sets the pd dataframe index of all dataframes in the system to index
        """
        for df in self.get_DataFrame(data=True, with_population=False):
            df.index = index

def set_executable(filename):
    """Set the exectuable bit on the given filename"""
    st = os.stat(filename)
    os.chmod(filename, st.st_mode | stat.S_IEXEC)

def get_obj(ref):
    """Get object from string reference."""
    oid = int(ref)
    return server.id2ref.get(oid) or server.id2obj[oid]

def forward(self, step):
        """Move the turtle forward.

        :param step: Integer. Distance to move forward.
        """
        x = self.pos_x + math.cos(math.radians(self.rotation)) * step
        y = self.pos_y + math.sin(math.radians(self.rotation)) * step
        prev_brush_state = self.brush_on
        self.brush_on = True
        self.move(x, y)
        self.brush_on = prev_brush_state

def polyline(self, arr):
        """Draw a set of lines"""
        for i in range(0, len(arr) - 1):
            self.line(arr[i][0], arr[i][1], arr[i + 1][0], arr[i + 1][1])

def _to_java_object_rdd(rdd):
    """ Return a JavaRDD of Object by unpickling

    It will convert each Python object into Java object by Pyrolite, whenever the
    RDD is serialized in batch or not.
    """
    rdd = rdd._reserialize(AutoBatchedSerializer(PickleSerializer()))
    return rdd.ctx._jvm.org.apache.spark.mllib.api.python.SerDe.pythonToJava(rdd._jrdd, True)

def mock_add_spec(self, spec, spec_set=False):
        """Add a spec to a mock. `spec` can either be an object or a
        list of strings. Only attributes on the `spec` can be fetched as
        attributes from the mock.

        If `spec_set` is True then only attributes on the spec can be set."""
        self._mock_add_spec(spec, spec_set)
        self._mock_set_magics()

def screen(self, width, height, colorDepth):
        """
        @summary: record resize event of screen (maybe first event)
        @param width: {int} width of screen
        @param height: {int} height of screen
        @param colorDepth: {int} colorDepth
        """
        screenEvent = ScreenEvent()
        screenEvent.width.value = width
        screenEvent.height.value = height
        screenEvent.colorDepth.value = colorDepth
        self.rec(screenEvent)

def get_subject(self, msg):
        """Extracts the subject line from an EmailMessage object."""

        text, encoding = decode_header(msg['subject'])[-1]

        try:
            text = text.decode(encoding)

        # If it's already decoded, ignore error
        except AttributeError:
            pass

        return text

def to_list(var):
    """Checks if given value is a list, tries to convert, if it is not."""
    if var is None:
        return []
    if isinstance(var, str):
        var = var.split('\n')
    elif not isinstance(var, list):
        try:
            var = list(var)
        except TypeError:
            raise ValueError("{} cannot be converted to the list.".format(var))
    return var

def iflatten(L):
    """Iterative flatten."""
    for sublist in L:
        if hasattr(sublist, '__iter__'):
            for item in iflatten(sublist): yield item
        else: yield sublist

def to_str(s):
    """
    Convert bytes and non-string into Python 3 str
    """
    if isinstance(s, bytes):
        s = s.decode('utf-8')
    elif not isinstance(s, str):
        s = str(s)
    return s

def wait_run_in_executor(func, *args, **kwargs):
    """
    Run blocking code in a different thread and wait
    for the result.

    :param func: Run this function in a different thread
    :param args: Parameters of the function
    :param kwargs: Keyword parameters of the function
    :returns: Return the result of the function
    """

    loop = asyncio.get_event_loop()
    future = loop.run_in_executor(None, functools.partial(func, *args, **kwargs))
    yield from asyncio.wait([future])
    return future.result()

def copy_image_on_background(image, color=WHITE):
    """
    Create a new image by copying the image on a *color* background.

    Args:
        image (PIL.Image.Image): Image to copy
        color (tuple): Background color usually WHITE or BLACK

    Returns:
        PIL.Image.Image

    """
    background = Image.new("RGB", image.size, color)
    background.paste(image, mask=image.split()[3])
    return background

def make_symmetric(dict):
    """Makes the given dictionary symmetric. Values are assumed to be unique."""
    for key, value in list(dict.items()):
        dict[value] = key
    return dict

def std_datestr(self, datestr):
        """Reformat a date string to standard format.
        """
        return date.strftime(
                self.str2date(datestr), self.std_dateformat)

def list_of_lists_to_dict(l):
    """ Convert list of key,value lists to dict

    [['id', 1], ['id', 2], ['id', 3], ['foo': 4]]
    {'id': [1, 2, 3], 'foo': [4]}
    """
    d = {}
    for key, val in l:
        d.setdefault(key, []).append(val)
    return d

def file_found(filename,force):
    """Check if a file exists"""
    if os.path.exists(filename) and not force:
        logger.info("Found %s; skipping..."%filename)
        return True
    else:
        return False

def filter_none(list_of_points):
    """
    
    :param list_of_points: 
    :return: list_of_points with None's removed
    """
    remove_elementnone = filter(lambda p: p is not None, list_of_points)
    remove_sublistnone = filter(lambda p: not contains_none(p), remove_elementnone)
    return list(remove_sublistnone)

def _columns_for_table(table_name):
    """
    Return all of the columns registered for a given table.

    Parameters
    ----------
    table_name : str

    Returns
    -------
    columns : dict of column wrappers
        Keys will be column names.

    """
    return {cname: col
            for (tname, cname), col in _COLUMNS.items()
            if tname == table_name}

def get_naive(dt):
  """Gets a naive datetime from a datetime.

  datetime_tz objects can't just have tzinfo replaced with None, you need to
  call asdatetime.

  Args:
    dt: datetime object.

  Returns:
    datetime object without any timezone information.
  """
  if not dt.tzinfo:
    return dt
  if hasattr(dt, "asdatetime"):
    return dt.asdatetime()
  return dt.replace(tzinfo=None)

def unicode_is_ascii(u_string):
    """Determine if unicode string only contains ASCII characters.

    :param str u_string: unicode string to check. Must be unicode
        and not Python 2 `str`.
    :rtype: bool
    """
    assert isinstance(u_string, str)
    try:
        u_string.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False

def makedirs(path):
    """
    Create directories if they do not exist, otherwise do nothing.

    Return path for convenience
    """
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def _validate_pos(df):
    """Validates the returned positional object
    """
    assert isinstance(df, pd.DataFrame)
    assert ["seqname", "position", "strand"] == df.columns.tolist()
    assert df.position.dtype == np.dtype("int64")
    assert df.strand.dtype == np.dtype("O")
    assert df.seqname.dtype == np.dtype("O")
    return df

def transpose(table):
    """
    transpose matrix
    """
    t = []
    for i in range(0, len(table[0])):
        t.append([row[i] for row in table])
    return t

def set_as_object(self, value):
        """
        Sets a new value to map element

        :param value: a new element or map value.
        """
        self.clear()
        map = MapConverter.to_map(value)
        self.append(map)

def _check_color_dim(val):
    """Ensure val is Nx(n_col), usually Nx3"""
    val = np.atleast_2d(val)
    if val.shape[1] not in (3, 4):
        raise RuntimeError('Value must have second dimension of size 3 or 4')
    return val, val.shape[1]

def match_paren(self, tokens, item):
        """Matches a paren."""
        match, = tokens
        return self.match(match, item)

def equal(obj1, obj2):
    """Calculate equality between two (Comparable) objects."""
    Comparable.log(obj1, obj2, '==')
    equality = obj1.equality(obj2)
    Comparable.log(obj1, obj2, '==', result=equality)
    return equality

def file_matches(filename, patterns):
    """Does this filename match any of the patterns?"""
    return any(fnmatch.fnmatch(filename, pat) for pat in patterns)

def is_valid_image_extension(file_path):
    """is_valid_image_extension."""
    valid_extensions = ['.jpeg', '.jpg', '.gif', '.png']
    _, extension = os.path.splitext(file_path)
    return extension.lower() in valid_extensions

def set_title(self, title, **kwargs):
        """Sets the title on the underlying matplotlib AxesSubplot."""
        ax = self.get_axes()
        ax.set_title(title, **kwargs)

def is_builtin_css_function(name):
    """Returns whether the given `name` looks like the name of a builtin CSS
    function.

    Unrecognized functions not in this list produce warnings.
    """
    name = name.replace('_', '-')

    if name in BUILTIN_FUNCTIONS:
        return True

    # Vendor-specific functions (-foo-bar) are always okay
    if name[0] == '-' and '-' in name[1:]:
        return True

    return False

def _linear_seaborn_(self, label=None, style=None, opts=None):
        """
        Returns a Seaborn linear regression plot
        """
        xticks, yticks = self._get_ticks(opts)
        try:
            fig = sns.lmplot(self.x, self.y, data=self.df)
            fig = self._set_with_height(fig, opts)
            return fig
        except Exception as e:
            self.err(e, self.linear_,
                     "Can not draw linear regression chart")

def map_wrap(f):
    """Wrap standard function to easily pass into 'map' processing.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper

def clear_matplotlib_ticks(self, axis="both"):
        """Clears the default matplotlib ticks."""
        ax = self.get_axes()
        plotting.clear_matplotlib_ticks(ax=ax, axis=axis)

def is_iter_non_string(obj):
    """test if object is a list or tuple"""
    if isinstance(obj, list) or isinstance(obj, tuple):
        return True
    return False

def strip_figures(figure):
	"""
	Strips a figure into multiple figures with a trace on each of them

	Parameters:
	-----------
		figure : Figure
			Plotly Figure
	"""
	fig=[]
	for trace in figure['data']:
		fig.append(dict(data=[trace],layout=figure['layout']))
	return fig

def set_value(self, value):
        """Set value of the checkbox.

        Parameters
        ----------
        value : bool
            value for the checkbox

        """
        if value:
            self.setChecked(Qt.Checked)
        else:
            self.setChecked(Qt.Unchecked)

def raise_figure_window(f=0):
    """
    Raises the supplied figure number or figure window.
    """
    if _fun.is_a_number(f): f = _pylab.figure(f)
    f.canvas.manager.window.raise_()

def IPYTHON_MAIN():
    """Decide if the Ipython command line is running code."""
    import pkg_resources

    runner_frame = inspect.getouterframes(inspect.currentframe())[-2]
    return (
        getattr(runner_frame, "function", None)
        == pkg_resources.load_entry_point("ipython", "console_scripts", "ipython").__name__
    )

def extent(self):
        """Helper for matplotlib imshow"""
        return (
            self.intervals[1].pix1 - 0.5,
            self.intervals[1].pix2 - 0.5,
            self.intervals[0].pix1 - 0.5,
            self.intervals[0].pix2 - 0.5,
        )

def exists(self, filepath):
        """Determines if the specified file/folder exists, even if it
        is on a remote server."""
        if self.is_ssh(filepath):
            self._check_ftp()
            remotepath = self._get_remote(filepath)
            try:
                self.ftp.stat(remotepath)
            except IOError as e:
                if e.errno == errno.ENOENT:
                    return False
            else:
                return True
        else:
            return os.path.exists(filepath)

def plot(self):
        """Plot the empirical histogram versus best-fit distribution's PDF."""
        plt.plot(self.bin_edges, self.hist, self.bin_edges, self.best_pdf)

def has_attribute(module_name, attribute_name):
    """Is this attribute present?"""
    init_file = '%s/__init__.py' % module_name
    return any(
        [attribute_name in init_line for init_line in open(init_file).readlines()]
    )

def set_ylimits(self, row, column, min=None, max=None):
        """Set y-axis limits of a subplot.

        :param row,column: specify the subplot.
        :param min: minimal axis value
        :param max: maximum axis value

        """
        subplot = self.get_subplot_at(row, column)
        subplot.set_ylimits(min, max)

def is_same_dict(d1, d2):
    """Test two dictionary is equal on values. (ignore order)
    """
    for k, v in d1.items():
        if isinstance(v, dict):
            is_same_dict(v, d2[k])
        else:
            assert d1[k] == d2[k]

    for k, v in d2.items():
        if isinstance(v, dict):
            is_same_dict(v, d1[k])
        else:
            assert d1[k] == d2[k]

def roots(self):
    """
    Returns a list with all roots. Needs Numpy.
    """
    import numpy as np
    return np.roots(list(self.values())[::-1]).tolist()

def boxes_intersect(box1, box2):
    """Determines if two rectangles, each input as a tuple
        (xmin, xmax, ymin, ymax), intersect."""
    xmin1, xmax1, ymin1, ymax1 = box1
    xmin2, xmax2, ymin2, ymax2 = box2
    if interval_intersection_width(xmin1, xmax1, xmin2, xmax2) and \
            interval_intersection_width(ymin1, ymax1, ymin2, ymax2):
        return True
    else:
        return False

def copy(a):
    """ Copy an array to the shared memory. 

        Notes
        -----
        copy is not always necessary because the private memory is always copy-on-write.

        Use :code:`a = copy(a)` to immediately dereference the old 'a' on private memory
    """
    shared = anonymousmemmap(a.shape, dtype=a.dtype)
    shared[:] = a[:]
    return shared

def _is_iterable(item):
    """ Checks if an item is iterable (list, tuple, generator), but not string """
    return isinstance(item, collections.Iterable) and not isinstance(item, six.string_types)

def machine_info():
    """Retrieve core and memory information for the current machine.
    """
    import psutil
    BYTES_IN_GIG = 1073741824.0
    free_bytes = psutil.virtual_memory().total
    return [{"memory": float("%.1f" % (free_bytes / BYTES_IN_GIG)), "cores": multiprocessing.cpu_count(),
             "name": socket.gethostname()}]

def is_string(val):
    """Determines whether the passed value is a string, safe for 2/3."""
    try:
        basestring
    except NameError:
        return isinstance(val, str)
    return isinstance(val, basestring)

def with_defaults(method, nparams, defaults=None):
  """Call method with nparams positional parameters, all non-specified defaults are passed None.

  :method: the method to call
  :nparams: the number of parameters the function expects
  :defaults: the default values to pass in for the last len(defaults) params
  """
  args = [None] * nparams if not defaults else defaults + max(nparams - len(defaults), 0) * [None]
  return method(*args)

def is_same_nick(self, left, right):
        """ Check if given nicknames are equal in the server's case mapping. """
        return self.normalize(left) == self.normalize(right)

def _file_type(self, field):
        """ Returns file type for given file field.
        
        Args:
            field (str): File field

        Returns:
            string. File type
        """
        type = mimetypes.guess_type(self._files[field])[0]
        return type.encode("utf-8") if isinstance(type, unicode) else str(type)

def is_date(thing):
    """Checks if the given thing represents a date

    :param thing: The object to check if it is a date
    :type thing: arbitrary object
    :returns: True if we have a date object
    :rtype: bool
    """
    # known date types
    date_types = (datetime.datetime,
                  datetime.date,
                  DateTime)
    return isinstance(thing, date_types)

def update_one(self, query, doc):
        """
        Updates one element of the collection

        :param query: dictionary representing the mongo query
        :param doc: dictionary representing the item to be updated
        :return: UpdateResult
        """
        if self.table is None:
            self.build_table()

        if u"$set" in doc:
            doc = doc[u"$set"]

        allcond = self.parse_query(query)

        try:
            result = self.table.update(doc, allcond)
        except:
            # TODO: check table.update result
            # check what pymongo does in that case
            result = None

        return UpdateResult(raw_result=result)

def eof(fd):
    """Determine if end-of-file is reached for file fd."""
    b = fd.read(1)
    end = len(b) == 0
    if not end:
        curpos = fd.tell()
        fd.seek(curpos - 1)
    return end

def last_month():
        """ Return start and end date of this month. """
        since = TODAY + delta(day=1, months=-1)
        until = since + delta(months=1)
        return Date(since), Date(until)

def _is_image_sequenced(image):
    """Determine if the image is a sequenced image."""
    try:
        image.seek(1)
        image.seek(0)
        result = True
    except EOFError:
        result = False

    return result

def erase_lines(n=1):
    """ Erases n lines from the screen and moves the cursor up to follow
    """
    for _ in range(n):
        print(codes.cursor["up"], end="")
        print(codes.cursor["eol"], end="")

def is_readable_dir(path):
  """Returns whether a path names an existing directory we can list and read files from."""
  return os.path.isdir(path) and os.access(path, os.R_OK) and os.access(path, os.X_OK)

def moving_average(array, n=3):
    """
    Calculates the moving average of an array.

    Parameters
    ----------
    array : array
        The array to have the moving average taken of
    n : int
        The number of points of moving average to take
    
    Returns
    -------
    MovingAverageArray : array
        The n-point moving average of the input array
    """
    ret = _np.cumsum(array, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def memory_used(self):
        """To know the allocated memory at function termination.

        ..versionadded:: 4.1

        This property might return None if the function is still running.

        This function should help to show memory leaks or ram greedy code.
        """
        if self._end_memory:
            memory_used = self._end_memory - self._start_memory
            return memory_used
        else:
            return None

def Min(a, axis, keep_dims):
    """
    Min reduction op.
    """
    return np.amin(a, axis=axis if not isinstance(axis, np.ndarray) else tuple(axis),
                   keepdims=keep_dims),

def check_update():
    """
    Check for app updates and print/log them.
    """
    logging.info('Check for app updates.')
    try:
        update = updater.check_for_app_updates()
    except Exception:
        logging.exception('Check for updates failed.')
        return
    if update:
        print("!!! UPDATE AVAILABLE !!!\n"
              "" + static_data.PROJECT_URL + "\n\n")
        logging.info("Update available: " + static_data.PROJECT_URL)
    else:
        logging.info("No update available.")

def load(self):
        """Load proxy list from configured proxy source"""
        self._list = self._source.load()
        self._list_iter = itertools.cycle(self._list)

def string_input(prompt=''):
    """Python 3 input()/Python 2 raw_input()"""
    v = sys.version[0]
    if v == '3':
        return input(prompt)
    else:
        return raw_input(prompt)

def compute_capture(args):
    x, y, w, h, params = args
    """Callable function for the multiprocessing pool."""
    return x, y, mandelbrot_capture(x, y, w, h, params)

def created_today(self):
        """Return True if created today."""
        if self.datetime.date() == datetime.today().date():
            return True
        return False

def kill_mprocess(process):
    """kill process
    Args:
        process - Popen object for process
    """
    if process and proc_alive(process):
        process.terminate()
        process.communicate()
    return not proc_alive(process)

def assert_valid_input(cls, tag):
        """Check if valid input tag or document."""

        # Fail on unexpected types.
        if not cls.is_tag(tag):
            raise TypeError("Expected a BeautifulSoup 'Tag', but instead recieved type {}".format(type(tag)))

def timeout_thread_handler(timeout, stop_event):
    """A background thread to kill the process if it takes too long.

    Args:
        timeout (float): The number of seconds to wait before killing
            the process.
        stop_event (Event): An optional event to cleanly stop the background
            thread if required during testing.
    """

    stop_happened = stop_event.wait(timeout)
    if stop_happened is False:
        print("Killing program due to %f second timeout" % timeout)

    os._exit(2)

def has_common(self, other):
        """Return set of common words between two word sets."""
        if not isinstance(other, WordSet):
            raise ValueError('Can compare only WordSets')
        return self.term_set & other.term_set

def pause(self):
        """Pause the music"""
        mixer.music.pause()
        self.pause_time = self.get_time()
        self.paused = True

def autoconvert(string):
    """Try to convert variables into datatypes."""
    for fn in (boolify, int, float):
        try:
            return fn(string)
        except ValueError:
            pass
    return string

def autoconvert(string):
    """Try to convert variables into datatypes."""
    for fn in (boolify, int, float):
        try:
            return fn(string)
        except ValueError:
            pass
    return string

def type(self):
        """Returns type of the data for the given FeatureType."""
        if self is FeatureType.TIMESTAMP:
            return list
        if self is FeatureType.BBOX:
            return BBox
        return dict

def isstring(value):
    """Report whether the given value is a byte or unicode string."""
    classes = (str, bytes) if pyutils.PY3 else basestring  # noqa: F821
    return isinstance(value, classes)

def chmod_plus_w(path):
  """Equivalent of unix `chmod +w path`"""
  path_mode = os.stat(path).st_mode
  path_mode &= int('777', 8)
  path_mode |= stat.S_IWRITE
  os.chmod(path, path_mode)

def object_as_dict(obj):
    """Turn an SQLAlchemy model into a dict of field names and values.

    Based on https://stackoverflow.com/a/37350445/1579058
    """
    return {c.key: getattr(obj, c.key)
            for c in inspect(obj).mapper.column_attrs}

def random_choice(sequence):
    """ Same as :meth:`random.choice`, but also supports :class:`set` type to be passed as sequence. """
    return random.choice(tuple(sequence) if isinstance(sequence, set) else sequence)

def object_as_dict(obj):
    """Turn an SQLAlchemy model into a dict of field names and values.

    Based on https://stackoverflow.com/a/37350445/1579058
    """
    return {c.key: getattr(obj, c.key)
            for c in inspect(obj).mapper.column_attrs}

def first_unique_char(s):
    """
    :type s: str
    :rtype: int
    """
    if (len(s) == 1):
        return 0
    ban = []
    for i in range(len(s)):
        if all(s[i] != s[k] for k in range(i + 1, len(s))) == True and s[i] not in ban:
            return i
        else:
            ban.append(s[i])
    return -1

def load_object_by_name(object_name):
    """Load an object from a module by name"""
    mod_name, attr = object_name.rsplit('.', 1)
    mod = import_module(mod_name)
    return getattr(mod, attr)

def erase(self):
        """White out the progress bar."""
        with self._at_last_line():
            self.stream.write(self._term.clear_eol)
        self.stream.flush()

def dictify(a_named_tuple):
    """Transform a named tuple into a dictionary"""
    return dict((s, getattr(a_named_tuple, s)) for s in a_named_tuple._fields)

def __clear_buffers(self):
        """Clears the input and output buffers"""
        try:
            self._port.reset_input_buffer()
            self._port.reset_output_buffer()
        except AttributeError:
            #pySerial 2.7
            self._port.flushInput()
            self._port.flushOutput()

def index_nearest(array, value):
    """
    Finds index of nearest value in array.

    Args:
        array: numpy array
        value:

    Returns:
        int

    http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    """
    idx = (np.abs(array-value)).argmin()
    return idx

def raise_figure_window(f=0):
    """
    Raises the supplied figure number or figure window.
    """
    if _fun.is_a_number(f): f = _pylab.figure(f)
    f.canvas.manager.window.raise_()

def from_dict(cls, d):
        """Create an instance from a dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.ENTRIES})

def stop_refresh(self):
        """Stop redrawing the canvas at the previously set timed interval.
        """
        self.logger.debug("stopping timed refresh")
        self.rf_flags['done'] = True
        self.rf_timer.clear()

def make_file_read_only(file_path):
    """
    Removes the write permissions for the given file for owner, groups and others.

    :param file_path: The file whose privileges are revoked.
    :raise FileNotFoundError: If the given file does not exist.
    """
    old_permissions = os.stat(file_path).st_mode
    os.chmod(file_path, old_permissions & ~WRITE_PERMISSIONS)

def norm(x, mu, sigma=1.0):
    """ Scipy norm function """
    return stats.norm(loc=mu, scale=sigma).pdf(x)

def clear():
    """Clears the console."""
    if sys.platform.startswith("win"):
        call("cls", shell=True)
    else:
        call("clear", shell=True)

def accel_next(self, *args):
        """Callback to go to the next tab. Called by the accel key.
        """
        if self.get_notebook().get_current_page() + 1 == self.get_notebook().get_n_pages():
            self.get_notebook().set_current_page(0)
        else:
            self.get_notebook().next_page()
        return True

def normalize(im, invert=False, scale=None, dtype=np.float64):
    """
    Normalize a field to a (min, max) exposure range, default is (0, 255).
    (min, max) exposure values. Invert the image if requested.
    """
    if dtype not in {np.float16, np.float32, np.float64}:
        raise ValueError('dtype must be numpy.float16, float32, or float64.')
    out = im.astype('float').copy()

    scale = scale or (0.0, 255.0)
    l, u = (float(i) for i in scale)
    out = (out - l) / (u - l)
    if invert:
        out = -out + (out.max() + out.min())
    return out.astype(dtype)

def cleanup(self, app):
        """Close all connections."""
        if hasattr(self.database.obj, 'close_all'):
            self.database.close_all()

def _normalize(mat: np.ndarray):
    """rescales a numpy array, so that min is 0 and max is 255"""
    return ((mat - mat.min()) * (255 / mat.max())).astype(np.uint8)

def _close(self):
        """
        Closes the client connection to the database.
        """
        if self.connection:
            with self.wrap_database_errors:
                self.connection.client.close()

def _normalize(obj):
    """
    Normalize dicts and lists

    :param obj:
    :return: normalized object
    """
    if isinstance(obj, list):
        return [_normalize(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items() if v is not None}
    elif hasattr(obj, 'to_python'):
        return obj.to_python()
    return obj

def pascal_row(n):
    """ Returns n-th row of Pascal's triangle
    """
    result = [1]
    x, numerator = 1, n
    for denominator in range(1, n // 2 + 1):
        x *= numerator
        x /= denominator
        result.append(x)
        numerator -= 1
    if n & 1 == 0:
        result.extend(reversed(result[:-1]))
    else:
        result.extend(reversed(result))
    return result

def run(self):
        """
        Runs the unit test framework. Can be overridden to run anything.
        Returns True on passing and False on failure.
        """
        try:
            import nose
            arguments = [sys.argv[0]] + list(self.test_args)
            return nose.run(argv=arguments)
        except ImportError:
            print()
            print("*** Nose library missing. Please install it. ***")
            print()
            raise

def place(self):
        """Place this container's canvas onto the parent container's canvas."""
        self.place_children()
        self.canvas.append(self.parent.canvas,
                           float(self.left), float(self.top))

def test(nose_argsuments):
    """ Run application tests """
    from nose import run

    params = ['__main__', '-c', 'nose.ini']
    params.extend(nose_argsuments)
    run(argv=params)

def merge_dict(data, *args):
    """Merge any number of dictionaries
    """
    results = {}
    for current in (data,) + args:
        results.update(current)
    return results

def wait_until_exit(self):
        """ Wait until thread exit

            Used for testing purpose only
        """

        if self._timeout is None:
            raise Exception("Thread will never exit. Use stop or specify timeout when starting it!")

        self._thread.join()
        self.stop()

def equal(obj1, obj2):
    """Calculate equality between two (Comparable) objects."""
    Comparable.log(obj1, obj2, '==')
    equality = obj1.equality(obj2)
    Comparable.log(obj1, obj2, '==', result=equality)
    return equality

def cfloat32_array_to_numpy(cptr, length):
    """Convert a ctypes float pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_float)):
        return np.fromiter(cptr, dtype=np.float32, count=length)
    else:
        raise RuntimeError('Expected float pointer')

def euclidean(x, y):
    """Standard euclidean distance.

    ..math::
        D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2
    return np.sqrt(result)

def compute_number_edges(function):
    """
    Compute the number of edges of the CFG
    Args:
        function (core.declarations.function.Function)
    Returns:
        int
    """
    n = 0
    for node in function.nodes:
        n += len(node.sons)
    return n

def bytesize(arr):
    """
    Returns the memory byte size of a Numpy array as an integer.
    """
    byte_size = np.prod(arr.shape) * np.dtype(arr.dtype).itemsize
    return byte_size

def count(data, axis=None):
    """Count the number of non-NA in this array along the given axis or axes
    """
    return np.sum(np.logical_not(isnull(data)), axis=axis)

def delimited(items, character='|'):
    """Returns a character delimited version of the provided list as a Python string"""
    return '|'.join(items) if type(items) in (list, tuple, set) else items

def length(self):
        """Array of vector lengths"""
        return np.sqrt(np.sum(self**2, axis=1)).view(np.ndarray)

def covstr(s):
  """ convert string to int or float. """
  try:
    ret = int(s)
  except ValueError:
    ret = float(s)
  return ret

def flatten_array(grid):
    """
    Takes a multi-dimensional array and returns a 1 dimensional array with the
    same contents.
    """
    grid = [grid[i][j] for i in range(len(grid)) for j in range(len(grid[i]))]
    while type(grid[0]) is list:
        grid = flatten_array(grid)
    return grid

def main(ctx, connection):
    """Command line interface for PyBEL."""
    ctx.obj = Manager(connection=connection)
    ctx.obj.bind()

def run(context, port):
    """ Run the Webserver/SocketIO and app
    """
    global ctx
    ctx = context
    app.run(port=port)

def length(self):
        """Array of vector lengths"""
        return np.sqrt(np.sum(self**2, axis=1)).view(np.ndarray)

def solr_to_date(d):
    """ converts YYYY-MM-DDT00:00:00Z to DD-MM-YYYY """
    return "{day}:{m}:{y}".format(y=d[:4], m=d[5:7], day=d[8:10]) if d else d

def find_nearest_index(arr, value):
    """For a given value, the function finds the nearest value
    in the array and returns its index."""
    arr = np.array(arr)
    index = (abs(arr-value)).argmin()
    return index

def mag(z):
    """Get the magnitude of a vector."""
    if isinstance(z[0], np.ndarray):
        return np.array(list(map(np.linalg.norm, z)))
    else:
        return np.linalg.norm(z)

def total_regular_pixels_from_mask(mask):
    """Compute the total number of unmasked regular pixels in a masks."""

    total_regular_pixels = 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                total_regular_pixels += 1

    return total_regular_pixels

def Max(a, axis, keep_dims):
    """
    Max reduction op.
    """
    return np.amax(a, axis=axis if not isinstance(axis, np.ndarray) else tuple(axis),
                   keepdims=keep_dims),

def line_count(fn):
    """ Get line count of file

    Args:
        fn (str): Path to file

    Return:
          Number of lines in file (int)
    """

    with open(fn) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def save_json(object, handle, indent=2):
    """Save object as json on CNS."""
    obj_json = json.dumps(object, indent=indent, cls=NumpyJSONEncoder)
    handle.write(obj_json)

def wordfreq(text, is_filename=False):
    """Return a dictionary of words and word counts in a string."""
    if is_filename:
        with open(text) as f:
            text = f.read()
    freqs = {}
    for word in text.split():
        lword = word.lower()
        freqs[lword] = freqs.get(lword, 0) + 1
    return freqs

def bytes_base64(x):
    """Turn bytes into base64"""
    if six.PY2:
        return base64.encodestring(x).replace('\n', '')
    return base64.encodebytes(bytes_encode(x)).replace(b'\n', b'')

def count_rows(self, table_name):
        """Return the number of entries in a table by counting them."""
        self.table_must_exist(table_name)
        query = "SELECT COUNT (*) FROM `%s`" % table_name.lower()
        self.own_cursor.execute(query)
        return int(self.own_cursor.fetchone()[0])

def get_list_from_file(file_name):
    """read the lines from a file into a list"""
    with open(file_name, mode='r', encoding='utf-8') as f1:
        lst = f1.readlines()
    return lst

def debug(sequence):
    """
    adds information to the sequence for better debugging, currently only
    an index property on each point in the sequence.
    """
    points = []
    for i, p in enumerate(sequence):
        copy = Point(p)
        copy['index'] = i
        points.append(copy)
    return sequence.__class__(points)

def make_temp(text):
        """
        Creates a temprorary file and writes the `text` into it
        """
        import tempfile
        (handle, path) = tempfile.mkstemp(text=True)
        os.close(handle)
        afile = File(path)
        afile.write(text)
        return afile

def shader_string(body, glsl_version='450 core'):
    """
    Call this method from a function that defines a literal shader string as the "body" argument.
    Dresses up a shader string in three ways:
        1) Insert #version at the top
        2) Insert #line number declaration
        3) un-indents
    The line number information can help debug glsl compile errors.
    The version string needs to be the very first characters in the shader,
    which can be distracting, requiring backslashes or other tricks.
    The unindenting allows you to type the shader code at a pleasing indent level
    in your python method, while still creating an unindented GLSL string at the end.
    """
    line_count = len(body.split('\n'))
    line_number = inspect.currentframe().f_back.f_lineno + 1 - line_count
    return """\
#version %s
%s
""" % (glsl_version, shader_substring(body, stack_frame=2))

def sometimesish(fn):
    """
    Has a 50/50 chance of calling a function
    """
    def wrapped(*args, **kwargs):
        if random.randint(1, 2) == 1:
            return fn(*args, **kwargs)

    return wrapped

def help(self, level=0):
        """return the usage string for available options """
        self.cmdline_parser.formatter.output_level = level
        with _patch_optparse():
            return self.cmdline_parser.format_help()

def average_gradient(data, *kwargs):
    """ Compute average gradient norm of an image
    """
    return np.average(np.array(np.gradient(data))**2)

def perl_cmd():
    """Retrieve path to locally installed conda Perl or first in PATH.
    """
    perl = which(os.path.join(get_bcbio_bin(), "perl"))
    if perl:
        return perl
    else:
        return which("perl")

def dumped(text, level, indent=2):
    """Put curly brackets round an indented text"""
    return indented("{\n%s\n}" % indented(text, level + 1, indent) or "None", level, indent) + "\n"

def cli(yamlfile, directory, out, classname, format):
    """ Generate graphviz representations of the biolink model """
    DotGenerator(yamlfile, format).serialize(classname=classname, dirname=directory, filename=out)

def format_op_hdr():
    """
    Build the header
    """
    txt = 'Base Filename'.ljust(36) + ' '
    txt += 'Lines'.rjust(7) + ' '
    txt += 'Words'.rjust(7) + '  '
    txt += 'Unique'.ljust(8) + ''
    return txt

def resize_image_with_crop_or_pad(img, target_height, target_width):
    """
    Crops and/or pads an image to a target width and height.

    Resizes an image to a target width and height by either cropping the image or padding it with zeros.

    NO CENTER CROP. NO CENTER PAD. (Just fill bottom right or crop bottom right)

    :param img: Numpy array representing the image.
    :param target_height: Target height.
    :param target_width: Target width.
    :return: The cropped and padded image.
    """
    h, w = target_height, target_width
    max_h, max_w, c = img.shape

    # crop
    img = crop_center(img, min(max_h, h), min(max_w, w))

    # pad
    padded_img = np.zeros(shape=(h, w, c), dtype=img.dtype)
    padded_img[:img.shape[0], :img.shape[1], :img.shape[2]] = img

    return padded_img

def interface_direct_class(data_class):
    """help to direct to the correct interface interacting with DB by class name only"""
    if data_class in ASSET:
        interface = AssetsInterface()
    elif data_class in PARTY:
        interface = PartiesInterface()
    elif data_class in BOOK:
        interface = BooksInterface()
    elif data_class in CORPORATE_ACTION:
        interface = CorporateActionsInterface()
    elif data_class in MARKET_DATA:
        interface = MarketDataInterface()
    elif data_class in TRANSACTION:
        interface = TransactionsInterface()
    else:
        interface = AssetManagersInterface()
    return interface

def zero_pad(m, n=1):
    """Pad a matrix with zeros, on all sides."""
    return np.pad(m, (n, n), mode='constant', constant_values=[0])

def chunked(l, n):
    """Chunk one big list into few small lists."""
    return [l[i:i + n] for i in range(0, len(l), n)]

def dcounts(self):
        """
        :return: a data frame with names and distinct counts and fractions for all columns in the database
        """
        print("WARNING: Distinct value count for all tables can take a long time...", file=sys.stderr)
        sys.stderr.flush()

        data = []
        for t in self.tables():
            for c in t.columns():
                data.append([t.name(), c.name(), c.dcount(), t.size(), c.dcount() / float(t.size())])
        df = pd.DataFrame(data, columns=["table", "column", "distinct", "size", "fraction"])
        return df

def _remove_empty_items(d, required):
  """Return a new dict with any empty items removed.

  Note that this is not a deep check. If d contains a dictionary which
  itself contains empty items, those are never checked.

  This method exists to make to_serializable() functions cleaner.
  We could revisit this some day, but for now, the serialized objects are
  stripped of empty values to keep the output YAML more compact.

  Args:
    d: a dictionary
    required: list of required keys (for example, TaskDescriptors always emit
      the "task-id", even if None)

  Returns:
    A dictionary with empty items removed.
  """

  new_dict = {}
  for k, v in d.items():
    if k in required:
      new_dict[k] = v
    elif isinstance(v, int) or v:
      # "if v" would suppress emitting int(0)
      new_dict[k] = v

  return new_dict

def ms_panset(self, viewer, event, data_x, data_y,
                  msg=True):
        """An interactive way to set the pan position.  The location
        (data_x, data_y) will be centered in the window.
        """
        if self.canpan and (event.state == 'down'):
            self._panset(viewer, data_x, data_y, msg=msg)
        return True

def disassemble_file(filename, outstream=None):
    """
    disassemble Python byte-code file (.pyc)

    If given a Python source file (".py") file, we'll
    try to find the corresponding compiled object.
    """
    filename = check_object_path(filename)
    (version, timestamp, magic_int, co, is_pypy,
     source_size) = load_module(filename)
    if type(co) == list:
        for con in co:
            disco(version, con, outstream)
    else:
        disco(version, co, outstream, is_pypy=is_pypy)
    co = None

def load_results(result_files, options, run_set_id=None, columns=None,
                 columns_relevant_for_diff=set()):
    """Version of load_result for multiple input files that will be loaded concurrently."""
    return parallel.map(
        load_result,
        result_files,
        itertools.repeat(options),
        itertools.repeat(run_set_id),
        itertools.repeat(columns),
        itertools.repeat(columns_relevant_for_diff))

def split(s):
  """Uses dynamic programming to infer the location of spaces in a string without spaces."""
  l = [_split(x) for x in _SPLIT_RE.split(s)]
  return [item for sublist in l for item in sublist]

def expandvars_dict(settings):
    """Expands all environment variables in a settings dictionary."""
    return dict(
        (key, os.path.expandvars(value))
        for key, value in settings.iteritems()
    )

def Bernstein(n, k):
    """Bernstein polynomial.

    """
    coeff = binom(n, k)

    def _bpoly(x):
        return coeff * x ** k * (1 - x) ** (n - k)

    return _bpoly

def from_pb(cls, pb):
        """Instantiate the object from a protocol buffer.

        Args:
            pb (protobuf)

        Save a reference to the protocol buffer on the object.
        """
        obj = cls._from_pb(pb)
        obj._pb = pb
        return obj

def get_table_width(table):
    """
    Gets the width of the table that would be printed.
    :rtype: ``int``
    """
    columns = transpose_table(prepare_rows(table))
    widths = [max(len(cell) for cell in column) for column in columns]
    return len('+' + '|'.join('-' * (w + 2) for w in widths) + '+')

def parse(self):
        """
        Parse file specified by constructor.
        """
        f = open(self.parse_log_path, "r")
        self.parse2(f)
        f.close()

def main(idle):
    """Any normal python logic which runs a loop. Can take arguments."""
    while True:

        LOG.debug("Sleeping for {0} seconds.".format(idle))
        time.sleep(idle)

def urlencoded(body, charset='ascii', **kwargs):
    """Converts query strings into native Python objects"""
    return parse_query_string(text(body, charset=charset), False)

def _delete_local(self, filename):
        """Deletes the specified file from the local filesystem."""

        if os.path.exists(filename):
            os.remove(filename)

def _parse_single_response(cls, response_data):
        """de-serialize a JSON-RPC Response/error

        :Returns: | [result, id] for Responses
        :Raises:  | RPCFault+derivates for error-packages/faults, RPCParseError, RPCInvalidRPC
        """

        if not isinstance(response_data, dict):
            raise errors.RPCInvalidRequest("No valid RPC-package.")

        if "id" not in response_data:
            raise errors.RPCInvalidRequest("""Invalid Response, "id" missing.""")

        request_id = response_data['id']

        if "jsonrpc" not in response_data:
            raise errors.RPCInvalidRequest("""Invalid Response, "jsonrpc" missing.""", request_id)
        if not isinstance(response_data["jsonrpc"], (str, unicode)):
            raise errors.RPCInvalidRequest("""Invalid Response, "jsonrpc" must be a string.""")
        if response_data["jsonrpc"] != "2.0":
            raise errors.RPCInvalidRequest("""Invalid jsonrpc version.""", request_id)

        error = response_data.get('error', None)
        result = response_data.get('result', None)

        if error and result:
            raise errors.RPCInvalidRequest("""Invalid Response, only "result" OR "error" allowed.""", request_id)

        if error:
            if not isinstance(error, dict):
                raise errors.RPCInvalidRequest("Invalid Response, invalid error-object.", request_id)

            if not ("code" in error and "message" in error):
                raise errors.RPCInvalidRequest("Invalid Response, invalid error-object.", request_id)

            error_data = error.get("data", None)

            if error['code'] in errors.ERROR_CODE_CLASS_MAP:
                raise errors.ERROR_CODE_CLASS_MAP[error['code']](error_data, request_id)
            else:
                error_object = errors.RPCFault(error_data, request_id)
                error_object.error_code = error['code']
                error_object.message = error['message']
                raise error_object

        return result, request_id

def trim(self):
        """Clear not used counters"""
        for key, value in list(iteritems(self.counters)):
            if value.empty():
                del self.counters[key]

def xml_str_to_dict(s):
    """ Transforms an XML string it to python-zimbra dict format

    For format, see:
      https://github.com/Zimbra-Community/python-zimbra/blob/master/README.md

    :param: a string, containing XML
    :returns: a dict, with python-zimbra format
    """
    xml = minidom.parseString(s)
    return pythonzimbra.tools.xmlserializer.dom_to_dict(xml.firstChild)

def delete_all_eggs(self):
        """ delete all the eggs in the directory specified """
        path_to_delete = os.path.join(self.egg_directory, "lib", "python")
        if os.path.exists(path_to_delete):
            shutil.rmtree(path_to_delete)

def __str__(self):
    """Returns a pretty-printed string for this object."""
    return 'Output name: "%s" watts: %d type: "%s" id: %d' % (
        self._name, self._watts, self._output_type, self._integration_id)

def distinct(l):
    """
    Return a list where the duplicates have been removed.

    Args:
        l (list): the list to filter.

    Returns:
        list: the same list without duplicates.
    """
    seen = set()
    seen_add = seen.add
    return (_ for _ in l if not (_ in seen or seen_add(_)))

def safe_call(cls, method, *args):
        """ Call a remote api method but don't raise if an error occurred."""
        return cls.call(method, *args, safe=True)

def get_height_for_line(self, lineno):
        """
        Return the height of the given line.
        (The height that it would take, if this line became visible.)
        """
        if self.wrap_lines:
            return self.ui_content.get_height_for_line(lineno, self.window_width)
        else:
            return 1

def grandparent_path(self):
        """ return grandparent's path string """
        return os.path.basename(os.path.join(self.path, '../..'))

def get_best_encoding(stream):
    """Returns the default stream encoding if not found."""
    rv = getattr(stream, 'encoding', None) or sys.getdefaultencoding()
    if is_ascii_encoding(rv):
        return 'utf-8'
    return rv

def get_size_in_bytes(self, handle):
        """Return the size in bytes."""
        fpath = self._fpath_from_handle(handle)
        return os.stat(fpath).st_size

def watched_extension(extension):
    """Return True if the given extension is one of the watched extensions"""
    for ext in hamlpy.VALID_EXTENSIONS:
        if extension.endswith('.' + ext):
            return True
    return False

def seconds(num):
    """
    Pause for this many seconds
    """
    now = pytime.time()
    end = now + num
    until(end)

def region_from_segment(image, segment):
    """given a segment (rectangle) and an image, returns it's corresponding subimage"""
    x, y, w, h = segment
    return image[y:y + h, x:x + w]

def set_trace():
    """Start a Pdb instance at the calling frame, with stdout routed to sys.__stdout__."""
    # https://github.com/nose-devs/nose/blob/master/nose/tools/nontrivial.py
    pdb.Pdb(stdout=sys.__stdout__).set_trace(sys._getframe().f_back)

def _calculate_similarity(c):
    """Get a similarity matrix of % of shared sequence

    :param c: cluster object

    :return ma: similarity matrix
    """
    ma = {}
    for idc in c:
        set1 = _get_seqs(c[idc])
        [ma.update({(idc, idc2): _common(set1, _get_seqs(c[idc2]), idc, idc2)}) for idc2 in c if idc != idc2 and (idc2, idc) not in ma]
    # logger.debug("_calculate_similarity_ %s" % ma)
    return ma

def load(self, filename='classifier.dump'):
        """
        Unpickles the classifier used
        """
        ifile = open(filename, 'r+')
        self.classifier = pickle.load(ifile)
        ifile.close()

def isin(elems, line):
    """Check if an element from a list is in a string.

    :type elems: list
    :type line: str

    """
    found = False
    for e in elems:
        if e in line.lower():
            found = True
            break
    return found

def unpickle_file(picklefile, **kwargs):
    """Helper function to unpickle data from `picklefile`."""
    with open(picklefile, 'rb') as f:
        return pickle.load(f, **kwargs)

def _string_width(self, s):
        """Get width of a string in the current font"""
        s = str(s)
        w = 0
        for i in s:
            w += self.character_widths[i]
        return w * self.font_size / 1000.0

def to_pydatetime(self):
        """
        Converts datetimeoffset object into Python's datetime.datetime object
        @return: time zone aware datetime.datetime
        """
        dt = datetime.datetime.combine(self._date.to_pydate(), self._time.to_pytime())
        from .tz import FixedOffsetTimezone
        return dt.replace(tzinfo=_utc).astimezone(FixedOffsetTimezone(self._offset))

def bundle_dir():
    """Handle resource management within an executable file."""
    if frozen():
        directory = sys._MEIPASS
    else:
        directory = os.path.dirname(os.path.abspath(stack()[1][1]))
    if os.path.exists(directory):
        return directory

def add_matplotlib_cmap(cm, name=None):
    """Add a matplotlib colormap."""
    global cmaps
    cmap = matplotlib_to_ginga_cmap(cm, name=name)
    cmaps[cmap.name] = cmap

def disable_cert_validation():
    """Context manager to temporarily disable certificate validation in the standard SSL
    library.

    Note: This should not be used in production code but is sometimes useful for
    troubleshooting certificate validation issues.

    By design, the standard SSL library does not provide a way to disable verification
    of the server side certificate. However, a patch to disable validation is described
    by the library developers. This context manager allows applying the patch for
    specific sections of code.

    """
    current_context = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        yield
    finally:
        ssl._create_default_https_context = current_context

def human__decision_tree():
    """ Decision Tree
    """

    # build data
    N = 1000000
    M = 3
    X = np.zeros((N,M))
    X.shape
    y = np.zeros(N)
    X[0, 0] = 1
    y[0] = 8
    X[1, 1] = 1
    y[1] = 8
    X[2, 0:2] = 1
    y[2] = 4

    # fit model
    xor_model = sklearn.tree.DecisionTreeRegressor(max_depth=2)
    xor_model.fit(X, y)

    return xor_model

def print_images(self, *printable_images):
        """
        This method allows printing several images in one shot. This is useful if the client code does not want the
        printer to make pause during printing
        """
        printable_image = reduce(lambda x, y: x.append(y), list(printable_images))
        self.print_image(printable_image)

def plotfft(s, fmax, doplot=False):
    """
    -----
    Brief
    -----
    This functions computes the Fast Fourier Transform of a signal, returning the frequency and magnitude values.

    -----------
    Description
    -----------
    Fast Fourier Transform (FFT) is a method to computationally calculate the Fourier Transform of discrete finite
    signals. This transform converts the time domain signal into a frequency domain signal by abdicating the temporal
    dimension.

    This function computes the FFT of the input signal and returns the frequency and respective amplitude values.

    ----------
    Parameters
    ----------
    s: array-like
      the input signal.
    fmax: int
      the sampling frequency.
    doplot: boolean
      a variable to indicate whether the plot is done or not.

    Returns
    -------
    f: array-like
      the frequency values (xx axis)
    fs: array-like
      the amplitude of the frequency values (yy axis)
    """

    fs = abs(numpy.fft.fft(s))
    f = numpy.linspace(0, fmax / 2, len(s) / 2)
    if doplot:
        plot(list(f[1:int(len(s) / 2)]), list(fs[1:int(len(s) / 2)]))
    return f[1:int(len(s) / 2)].copy(), fs[1:int(len(s) / 2)].copy()

def get_unique_indices(df, axis=1):
    """

    :param df:
    :param axis:
    :return:
    """
    return dict(zip(df.columns.names, dif.columns.levels))

def multi_pop(d, *args):
    """ pops multiple keys off a dict like object """
    retval = {}
    for key in args:
        if key in d:
            retval[key] = d.pop(key)
    return retval

def computeFactorial(n):
    """
    computes factorial of n
    """
    sleep_walk(10)
    ret = 1
    for i in range(n):
        ret = ret * (i + 1)
    return ret

def auth_request(self, url, headers, body):
        """Perform auth request for token."""

        return self.req.post(url, headers, body=body)

def _comment(string):
    """return string as a comment"""
    lines = [line.strip() for line in string.splitlines()]
    return "# " + ("%s# " % linesep).join(lines)

def psql(sql, show=True):
    """
    Runs SQL against the project's database.
    """
    out = postgres('psql -c "%s"' % sql)
    if show:
        print_command(sql)
    return out

def download_url(url, filename, headers):
    """Download a file from `url` to `filename`."""
    ensure_dirs(filename)
    response = requests.get(url, headers=headers, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(16 * 1024):
                f.write(chunk)

def dictfetchall(cursor):
    """Returns all rows from a cursor as a dict (rather than a headerless table)

    From Django Documentation: https://docs.djangoproject.com/en/dev/topics/db/sql/
    """
    desc = cursor.description
    return [dict(zip([col[0] for col in desc], row)) for row in cursor.fetchall()]

def vline(self, x, y, height, color):
        """Draw a vertical line up to a given length."""
        self.rect(x, y, 1, height, color, fill=True)

def exit(exit_code=0):
  r"""A function to support exiting from exit hooks.

  Could also be used to exit from the calling scripts in a thread safe manner.
  """
  core.processExitHooks()

  if state.isExitHooked and not hasattr(sys, 'exitfunc'): # The function is called from the exit hook
    sys.stderr.flush()
    sys.stdout.flush()
    os._exit(exit_code) #pylint: disable=W0212

  sys.exit(exit_code)

def destroy(self):
        """ Destroy the SQLStepQueue tables in the database """
        with self._db_conn() as conn:
            for table_name in self._tables:
                conn.execute('DROP TABLE IF EXISTS %s' % table_name)
        return self

def _do_auto_predict(machine, X, *args):
    """Performs an automatic prediction for the specified machine and returns
    the predicted values.
    """
    if auto_predict and hasattr(machine, "predict"):
        return machine.predict(X)

def has_edit_permission(self, request):
        """ Can edit this object """
        return request.user.is_authenticated and request.user.is_active and request.user.is_staff

def pretty_dict_str(d, indent=2):
    """shows JSON indented representation of d"""
    b = StringIO()
    write_pretty_dict_str(b, d, indent=indent)
    return b.getvalue()

def unpunctuate(s, *, char_blacklist=string.punctuation):
    """ Remove punctuation from string s. """
    # remove punctuation
    s = "".join(c for c in s if c not in char_blacklist)
    # remove consecutive spaces
    return " ".join(filter(None, s.split(" ")))

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

def do_forceescape(value):
    """Enforce HTML escaping.  This will probably double escape variables."""
    if hasattr(value, '__html__'):
        value = value.__html__()
    return escape(text_type(value))

def get_previous(self):
        """Get the billing cycle prior to this one. May return None"""
        return BillingCycle.objects.filter(date_range__lt=self.date_range).order_by('date_range').last()

def go_to_new_line(self):
        """Go to the end of the current line and create a new line"""
        self.stdkey_end(False, False)
        self.insert_text(self.get_line_separator())

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

def go_to_new_line(self):
        """Go to the end of the current line and create a new line"""
        self.stdkey_end(False, False)
        self.insert_text(self.get_line_separator())

def imp_print(self, text, end):
		"""Directly send utf8 bytes to stdout"""
		sys.stdout.write((text + end).encode("utf-8"))

def All(sequence):
  """
  :param sequence: Any sequence whose elements can be evaluated as booleans.
  :returns: true if all elements of the sequence satisfy True and x.
  """
  return bool(reduce(lambda x, y: x and y, sequence, True))

def _get_printable_columns(columns, row):
    """Return only the part of the row which should be printed.
    """
    if not columns:
        return row

    # Extract the column values, in the order specified.
    return tuple(row[c] for c in columns)

def print(*a):
    """ print just one that returns what you give it instead of None """
    try:
        _print(*a)
        return a[0] if len(a) == 1 else a
    except:
        _print(*a)

def _quit(self, *args):
        """ quit crash """
        self.logger.warn('Bye!')
        sys.exit(self.exit())

def timeit(output):
    """
    If output is string, then print the string and also time used
    """
    b = time.time()
    yield
    print output, 'time used: %.3fs' % (time.time()-b)

def _expand(self, str, local_vars={}):
        """Expand $vars in a string."""
        return ninja_syntax.expand(str, self.vars, local_vars)

def format_time(timestamp):
    """Formats timestamp to human readable format"""
    format_string = '%Y_%m_%d_%Hh%Mm%Ss'
    formatted_time = datetime.datetime.fromtimestamp(timestamp).strftime(format_string)
    return formatted_time

def executable_exists(executable):
    """Test if an executable is available on the system."""
    for directory in os.getenv("PATH").split(":"):
        if os.path.exists(os.path.join(directory, executable)):
            return True
    return False

def pout(msg, log=None):
    """Print 'msg' to stdout, and option 'log' at info level."""
    _print(msg, sys.stdout, log_func=log.info if log else None)

def extract_table_names(query):
    """ Extract table names from an SQL query. """
    # a good old fashioned regex. turns out this worked better than actually parsing the code
    tables_blocks = re.findall(r'(?:FROM|JOIN)\s+(\w+(?:\s*,\s*\w+)*)', query, re.IGNORECASE)
    tables = [tbl
              for block in tables_blocks
              for tbl in re.findall(r'\w+', block)]
    return set(tables)

def print_tree(self, indent=2):
        """ print_tree: prints out structure of tree
            Args: indent (int): What level of indentation at which to start printing
            Returns: None
        """
        config.LOGGER.info("{indent}{data}".format(indent="   " * indent, data=str(self)))
        for child in self.children:
            child.print_tree(indent + 1)

def get_key_by_value(dictionary, search_value):
    """
    searchs a value in a dicionary and returns the key of the first occurrence

    :param dictionary: dictionary to search in
    :param search_value: value to search for
    """
    for key, value in dictionary.iteritems():
        if value == search_value:
            return ugettext(key)

def algo_exp(x, m, t, b):
    """mono-exponential curve."""
    return m*np.exp(-t*x)+b

def iflatten(L):
    """Iterative flatten."""
    for sublist in L:
        if hasattr(sublist, '__iter__'):
            for item in iflatten(sublist): yield item
        else: yield sublist

def longest_run_1d(arr):
    """Return the length of the longest consecutive run of identical values.

    Parameters
    ----------
    arr : bool array
      Input array

    Returns
    -------
    int
      Length of longest run.
    """
    v, rl = rle_1d(arr)[:2]
    return np.where(v, rl, 0).max()

def imflip(img, direction='horizontal'):
    """Flip an image horizontally or vertically.

    Args:
        img (ndarray): Image to be flipped.
        direction (str): The flip direction, either "horizontal" or "vertical".

    Returns:
        ndarray: The flipped image.
    """
    assert direction in ['horizontal', 'vertical']
    if direction == 'horizontal':
        return np.flip(img, axis=1)
    else:
        return np.flip(img, axis=0)

def build_docs(directory):
    """Builds sphinx docs from a given directory."""
    os.chdir(directory)
    process = subprocess.Popen(["make", "html"], cwd=directory)
    process.communicate()

def dedupe_list(l):
    """Remove duplicates from a list preserving the order.

    We might be tempted to use the list(set(l)) idiom, but it doesn't preserve
    the order, which hinders testability and does not work for lists with
    unhashable elements.
    """
    result = []

    for el in l:
        if el not in result:
            result.append(el)

    return result

def _chunk_write(chunk, local_file, progress):
    """Write a chunk to file and update the progress bar"""
    local_file.write(chunk)
    progress.update_with_increment_value(len(chunk))

def safe_int_conv(number):
    """Safely convert a single number to integer."""
    try:
        return int(np.array(number).astype(int, casting='safe'))
    except TypeError:
        raise ValueError('cannot safely convert {} to integer'.format(number))

def getpackagepath():
    """
     *Get the root path for this python package - used in unit testing code*
    """
    moduleDirectory = os.path.dirname(__file__)
    packagePath = os.path.dirname(__file__) + "/../"

    return packagePath

def hash_iterable(it):
	"""Perform a O(1) memory hash of an iterable of arbitrary length.

	hash(tuple(it)) creates a temporary tuple containing all values from it
	which could be a problem if it is large.

	See discussion at:
	https://groups.google.com/forum/#!msg/python-ideas/XcuC01a8SYs/e-doB9TbDwAJ
	"""
	hash_value = hash(type(it))
	for value in it:
		hash_value = hash((hash_value, value))
	return hash_value

def _normal_prompt(self):
        """
        Flushes the prompt before requesting the input

        :return: The command line
        """
        sys.stdout.write(self.__get_ps1())
        sys.stdout.flush()
        return safe_input()

def md5_string(s):
    """
    Shortcut to create md5 hash
    :param s:
    :return:
    """
    m = hashlib.md5()
    m.update(s)
    return str(m.hexdigest())

def write_only_property(f):
    """
    @write_only_property decorator. Creates a property (descriptor attribute)
    that accepts assignment, but not getattr (use in an expression).
    """
    docstring = f.__doc__

    return property(fset=f, doc=docstring)

def _rnd_datetime(self, start, end):
        """Internal random datetime generator.
        """
        return self.from_utctimestamp(
            random.randint(
                int(self.to_utctimestamp(start)),
                int(self.to_utctimestamp(end)),
            )
        )

def message_from_string(s, *args, **kws):
    """Parse a string into a Message object model.

    Optional _class and strict are passed to the Parser constructor.
    """
    from future.backports.email.parser import Parser
    return Parser(*args, **kws).parsestr(s)

def _unique_id(self, prefix):
        """
        Generate a unique (within the graph) identifer
        internal to graph generation.
        """
        _id = self._id_gen
        self._id_gen += 1
        return prefix + str(_id)

def getEdges(npArr):
  """get np array of bin edges"""
  edges = np.concatenate(([0], npArr[:,0] + npArr[:,2]))
  return np.array([Decimal(str(i)) for i in edges])

def _get_random_id():
    """ Get a random (i.e., unique) string identifier"""
    symbols = string.ascii_uppercase + string.ascii_lowercase + string.digits
    return ''.join(random.choice(symbols) for _ in range(15))

def add_arrow(self, x1, y1, x2, y2, **kws):
        """add arrow to plot"""
        self.panel.add_arrow(x1, y1, x2, y2, **kws)

def ci(a, which=95, axis=None):
    """Return a percentile range from an array of values."""
    p = 50 - which / 2, 50 + which / 2
    return percentiles(a, p, axis)

def get_callable_documentation(the_callable):
    """Return a string with the callable signature and its docstring.

    :param the_callable: the callable to be analyzed.
    :type the_callable: function/callable.
    :return: the signature.
    """
    return wrap_text_in_a_box(
        title=get_callable_signature_as_string(the_callable),
        body=(getattr(the_callable, '__doc__') or 'No documentation').replace(
            '\n', '\n\n'),
        style='ascii_double')

def string_to_identity(identity_str):
    """Parse string into Identity dictionary."""
    m = _identity_regexp.match(identity_str)
    result = m.groupdict()
    log.debug('parsed identity: %s', result)
    return {k: v for k, v in result.items() if v}

def main(output=None, error=None, verbose=False):
    """ The main (cli) interface for the pylint runner. """
    runner = Runner(args=["--verbose"] if verbose is not False else None)
    runner.run(output, error)

def wget(url):
    """
    Download the page into a string
    """
    import urllib.parse
    request = urllib.request.urlopen(url)
    filestring = request.read()
    return filestring

def update_one(self, query, doc):
        """
        Updates one element of the collection

        :param query: dictionary representing the mongo query
        :param doc: dictionary representing the item to be updated
        :return: UpdateResult
        """
        if self.table is None:
            self.build_table()

        if u"$set" in doc:
            doc = doc[u"$set"]

        allcond = self.parse_query(query)

        try:
            result = self.table.update(doc, allcond)
        except:
            # TODO: check table.update result
            # check what pymongo does in that case
            result = None

        return UpdateResult(raw_result=result)

def compose(*parameter_functions):
  """Composes multiple modification functions in order.

  Args:
    *parameter_functions: The functions to compose.

  Returns:
    A parameter modification function that consists of applying all the provided
    functions.
  """
  def composed_fn(var_name, variable, phase):
    for fn in parameter_functions:
      variable = fn(var_name, variable, phase)
    return variable
  return composed_fn

def from_pydatetime(cls, pydatetime):
        """
        Creates sql datetime2 object from Python datetime object
        ignoring timezone
        @param pydatetime: Python datetime object
        @return: sql datetime2 object
        """
        return cls(date=Date.from_pydate(pydatetime.date),
                   time=Time.from_pytime(pydatetime.time))

def list_files(directory):
    """Returns all files in a given directory
    """
    return [f for f in pathlib.Path(directory).iterdir() if f.is_file() and not f.name.startswith('.')]

def cover(session):
    """Run the final coverage report.
    This outputs the coverage report aggregating coverage from the unit
    test runs (not system test runs), and then erases coverage data.
    """
    session.interpreter = 'python3.6'
    session.install('coverage', 'pytest-cov')
    session.run('coverage', 'report', '--show-missing', '--fail-under=100')
    session.run('coverage', 'erase')

def get_font_list():
    """Returns a sorted list of all system font names"""

    font_map = pangocairo.cairo_font_map_get_default()
    font_list = [f.get_name() for f in font_map.list_families()]
    font_list.sort()

    return font_list

def run_tests(self):
		"""
		Invoke pytest, replacing argv. Return result code.
		"""
		with _save_argv(_sys.argv[:1] + self.addopts):
			result_code = __import__('pytest').main()
			if result_code:
				raise SystemExit(result_code)

def money(min=0, max=10):
    """Return a str of decimal with two digits after a decimal mark."""
    value = random.choice(range(min * 100, max * 100))
    return "%1.2f" % (float(value) / 100)

def contextMenuEvent(self, event):
        """Reimplement Qt method"""
        self.menu.popup(event.globalPos())
        event.accept()

async def sysinfo(dev: Device):
    """Print out system information (version, MAC addrs)."""
    click.echo(await dev.get_system_info())
    click.echo(await dev.get_interface_information())

def remove_parameter(self, name):
		""" Remove the specified parameter from this query

		:param name: name of a parameter to remove
		:return: None
		"""
		if name in self.__query:
			self.__query.pop(name)

def combinations(l):
    """Pure-Python implementation of itertools.combinations(l, 2)."""
    result = []
    for x in xrange(len(l) - 1):
        ls = l[x + 1:]
        for y in ls:
            result.append((l[x], y))
    return result

def bbox(self):
        """BBox"""
        return self.left, self.top, self.right, self.bottom

def PopTask(self):
    """Retrieves and removes the first task from the heap.

    Returns:
      Task: the task or None if the heap is empty.
    """
    try:
      _, task = heapq.heappop(self._heap)

    except IndexError:
      return None
    self._task_identifiers.remove(task.identifier)
    return task

def parse_cookies(self, req, name, field):
        """Pull the value from the cookiejar."""
        return core.get_value(req.COOKIES, name, field)

def random_color(_min=MIN_COLOR, _max=MAX_COLOR):
    """Returns a random color between min and max."""
    return color(random.randint(_min, _max))

def sample_correlations(self):
        """Returns an `ExpMatrix` containing all pairwise sample correlations.

        Returns
        -------
        `ExpMatrix`
            The sample correlation matrix.

        """
        C = np.corrcoef(self.X.T)
        corr_matrix = ExpMatrix(genes=self.samples, samples=self.samples, X=C)
        return corr_matrix

def get_incomplete_path(filename):
  """Returns a temporary filename based on filename."""
  random_suffix = "".join(
      random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
  return filename + ".incomplete" + random_suffix

def shape(self):
        """Compute the shape of the dataset as (rows, cols)."""
        if not self.data:
            return (0, 0)
        return (len(self.data), len(self.dimensions))

def rnormal(mu, tau, size=None):
    """
    Random normal variates.
    """
    return np.random.normal(mu, 1. / np.sqrt(tau), size)

def get_file_name(url):
  """Returns file name of file at given url."""
  return os.path.basename(urllib.parse.urlparse(url).path) or 'unknown_name'

def rlognormal(mu, tau, size=None):
    """
    Return random lognormal variates.
    """

    return np.random.lognormal(mu, np.sqrt(1. / tau), size)

def root_parent(self, category=None):
        """ Returns the topmost parent of the current category. """
        return next(filter(lambda c: c.is_root, self.hierarchy()))

def get_randomized_guid_sample(self, item_count):
        """ Fetch a subset of randomzied GUIDs from the whitelist """
        dataset = self.get_whitelist()
        random.shuffle(dataset)
        return dataset[:item_count]

def get_offset_topic_partition_count(kafka_config):
    """Given a kafka cluster configuration, return the number of partitions
    in the offset topic. It will raise an UnknownTopic exception if the topic
    cannot be found."""
    metadata = get_topic_partition_metadata(kafka_config.broker_list)
    if CONSUMER_OFFSET_TOPIC not in metadata:
        raise UnknownTopic("Consumer offset topic is missing.")
    return len(metadata[CONSUMER_OFFSET_TOPIC])

def add_range(self, sequence, begin, end):
    """Add a read_range primitive"""
    sequence.parser_tree = parsing.Range(self.value(begin).strip("'"),
                                         self.value(end).strip("'"))
    return True

def getScriptLocation():
	"""Helper function to get the location of a Python file."""
	location = os.path.abspath("./")
	if __file__.rfind("/") != -1:
		location = __file__[:__file__.rfind("/")]
	return location

def max_values(args):
    """ Return possible range for max function. """
    return Interval(max(x.low for x in args), max(x.high for x in args))

def get_last_modified_timestamp(self):
        """
        Looks at the files in a git root directory and grabs the last modified timestamp
        """
        cmd = "find . -print0 | xargs -0 stat -f '%T@ %p' | sort -n | tail -1 | cut -f2- -d' '"
        ps = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        output = ps.communicate()[0]
        print output

def camel_to_snake_case(name):
    """Takes a camelCased string and converts to snake_case."""
    pattern = r'[A-Z][a-z]+|[A-Z]+(?![a-z])'
    return '_'.join(map(str.lower, re.findall(pattern, name)))

def runcode(code):
	"""Run the given code line by line with printing, as list of lines, and return variable 'ans'."""
	for line in code:
		print('# '+line)
		exec(line,globals())
	print('# return ans')
	return ans

def read_utf8(fh, byteorder, dtype, count, offsetsize):
    """Read tag data from file and return as unicode string."""
    return fh.read(count).decode('utf-8')

def run_cmd(command, verbose=True, shell='/bin/bash'):
    """internal helper function to run shell commands and get output"""
    process = Popen(command, shell=True, stdout=PIPE, stderr=STDOUT, executable=shell)
    output = process.stdout.read().decode().strip().split('\n')
    if verbose:
        # return full output including empty lines
        return output
    return [line for line in output if line.strip()]

def replace_all(filepath, searchExp, replaceExp):
    """
    Replace all the ocurrences (in a file) of a string with another value.
    """
    for line in fileinput.input(filepath, inplace=1):
        if searchExp in line:
            line = line.replace(searchExp, replaceExp)
        sys.stdout.write(line)

def import_js(path, lib_name, globals):
    """Imports from javascript source file.
      globals is your globals()"""
    with codecs.open(path_as_local(path), "r", "utf-8") as f:
        js = f.read()
    e = EvalJs()
    e.execute(js)
    var = e.context['var']
    globals[lib_name] = var.to_python()

def file_to_str(fname):
    """
    Read a file into a string
    PRE: fname is a small file (to avoid hogging memory and its discontents)
    """
    data = None
    # rU = read with Universal line terminator
    with open(fname, 'rU') as fd:
        data = fd.read()
    return data

def dedup(seq):
    """Remove duplicates from a list while keeping order."""
    seen = set()
    for item in seq:
        if item not in seen:
            seen.add(item)
            yield item

def loads(string):
  """
  Deserializes Java objects and primitive data serialized by ObjectOutputStream
  from a string.
  """
  f = StringIO.StringIO(string)
  marshaller = JavaObjectUnmarshaller(f)
  marshaller.add_transformer(DefaultObjectTransformer())
  return marshaller.readObject()

def _run_cmd_get_output(cmd):
    """Runs a shell command, returns console output.

    Mimics python3's subprocess.getoutput
    """
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    out, err = process.communicate()
    return out or err

def parse_comments_for_file(filename):
    """
    Return a list of all parsed comments in a file.  Mostly for testing &
    interactive use.
    """
    return [parse_comment(strip_stars(comment), next_line)
            for comment, next_line in get_doc_comments(read_file(filename))]

def _calc_dir_size(path):
    """
    Calculate size of all files in `path`.

    Args:
        path (str): Path to the directory.

    Returns:
        int: Size of the directory in bytes.
    """
    dir_size = 0
    for (root, dirs, files) in os.walk(path):
        for fn in files:
            full_fn = os.path.join(root, fn)
            dir_size += os.path.getsize(full_fn)

    return dir_size

def standard_input():
    """Generator that yields lines from standard input."""
    with click.get_text_stream("stdin") as stdin:
        while stdin.readable():
            line = stdin.readline()
            if line:
                yield line.strip().encode("utf-8")

def get_file_string(filepath):
    """Get string from file."""
    with open(os.path.abspath(filepath)) as f:
        return f.read()

def import_public_rsa_key_from_file(filename):
    """
    Read a public RSA key from a PEM file.

    :param filename: The name of the file
    :param passphrase: A pass phrase to use to unpack the PEM file.
    :return: A
        cryptography.hazmat.primitives.asymmetric.rsa.RSAPublicKey instance
    """
    with open(filename, "rb") as key_file:
        public_key = serialization.load_pem_public_key(
            key_file.read(),
            backend=default_backend())
    return public_key

def cat_acc(y_true, y_pred):
    """Categorical accuracy
    """
    return np.mean(y_true.argmax(axis=1) == y_pred.argmax(axis=1))

def camel_to_underscore(string):
    """Convert camelcase to lowercase and underscore.

    Recipe from http://stackoverflow.com/a/1176023

    Args:
        string (str): The string to convert.

    Returns:
        str: The converted string.
    """
    string = FIRST_CAP_RE.sub(r'\1_\2', string)
    return ALL_CAP_RE.sub(r'\1_\2', string).lower()

def center_eigenvalue_diff(mat):
    """Compute the eigvals of mat and then find the center eigval difference."""
    N = len(mat)
    evals = np.sort(la.eigvals(mat))
    diff = np.abs(evals[N/2] - evals[N/2-1])
    return diff

def recursively_update(d, d2):
  """dict.update but which merges child dicts (dict2 takes precedence where there's conflict)."""
  for k, v in d2.items():
    if k in d:
      if isinstance(v, dict):
        recursively_update(d[k], v)
        continue
    d[k] = v

def entropy(string):
    """Compute entropy on the string"""
    p, lns = Counter(string), float(len(string))
    return -sum(count/lns * math.log(count/lns, 2) for count in p.values())

def __connect():
    """
    Connect to a redis instance.
    """
    global redis_instance
    if use_tcp_socket:
        redis_instance = redis.StrictRedis(host=hostname, port=port)
    else:
        redis_instance = redis.StrictRedis(unix_socket_path=unix_socket)

def _regex_span(_regex, _str, case_insensitive=True):
    """Return all matches in an input string.
    :rtype : regex.match.span
    :param _regex: A regular expression pattern.
    :param _str: Text on which to run the pattern.
    """
    if case_insensitive:
        flags = regex.IGNORECASE | regex.FULLCASE | regex.VERSION1
    else:
        flags = regex.VERSION1
    comp = regex.compile(_regex, flags=flags)
    matches = comp.finditer(_str)
    for match in matches:
        yield match

def __contains__(self, key):
        """Return ``True`` if *key* is present, else ``False``."""
        pickled_key = self._pickle_key(key)
        return bool(self.redis.hexists(self.key, pickled_key))

def getTypeStr(_type):
  r"""Gets the string representation of the given type.
  """
  if isinstance(_type, CustomType):
    return str(_type)

  if hasattr(_type, '__name__'):
    return _type.__name__

  return ''

def values(self):
        """ :see::meth:RedisMap.keys """
        for val in self._client.hvals(self.key_prefix):
            yield self._loads(val)

def get_instance(key, expire=None):
    """Return an instance of RedisSet."""
    global _instances
    try:
        instance = _instances[key]
    except KeyError:
        instance = RedisSet(
            key,
            _redis,
            expire=expire
        )
        _instances[key] = instance

    return instance

def generate_unique_host_id():
    """Generate a unique ID, that is somewhat guaranteed to be unique among all
    instances running at the same time."""
    host = ".".join(reversed(socket.gethostname().split(".")))
    pid = os.getpid()
    return "%s.%d" % (host, pid)

def __setitem__(self, field, value):
        """ :see::meth:RedisMap.__setitem__ """
        return self._client.hset(self.key_prefix, field, self._dumps(value))

def get_week_start_end_day():
    """
    Get the week start date and end date
    """
    t = date.today()
    wd = t.weekday()
    return (t - timedelta(wd), t + timedelta(6 - wd))

def __repr__(self):
    """Returns a stringified representation of this object."""
    return str({'name': self._name, 'watts': self._watts,
                'type': self._output_type, 'id': self._integration_id})

def _defaultdict(dct, fallback=_illegal_character):
    """Wraps the given dictionary such that the given fallback function will be called when a nonexistent key is
    accessed.
    """
    out = defaultdict(lambda: fallback)
    for k, v in six.iteritems(dct):
        out[k] = v
    return out

def unapostrophe(text):
    """Strip apostrophe and 's' from the end of a string."""
    text = re.sub(r'[%s]s?$' % ''.join(APOSTROPHES), '', text)
    return text

def _validate_input_data(self, data, request):
        """ Validate input data.

        :param request: the HTTP request
        :param data: the parsed data
        :return: if validation is performed and succeeds the data is converted
                 into whatever format the validation uses (by default Django's
                 Forms) If not, the data is returned unchanged.
        :raises: HttpStatusCodeError if data is not valid
        """

        validator = self._get_input_validator(request)
        if isinstance(data, (list, tuple)):
            return map(validator.validate, data)
        else:
            return validator.validate(data)

def ismatch(text, pattern):
    """Test whether text contains string or matches regex."""

    if hasattr(pattern, 'search'):
        return pattern.search(text) is not None
    else:
        return pattern in text if Config.options.case_sensitive \
            else pattern.lower() in text.lower()

def get_numbers(s):
    """Extracts all integers from a string an return them in a list"""

    result = map(int, re.findall(r'[0-9]+', unicode(s)))
    return result + [1] * (2 - len(result))

def now_time(str=False):
    """Get the current time."""
    if str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return datetime.datetime.now()

def camel_to_snake_case(name):
    """Takes a camelCased string and converts to snake_case."""
    pattern = r'[A-Z][a-z]+|[A-Z]+(?![a-z])'
    return '_'.join(map(str.lower, re.findall(pattern, name)))

def _add_hash(source):
    """Add a leading hash '#' at the beginning of every line in the source."""
    source = '\n'.join('# ' + line.rstrip()
                       for line in source.splitlines())
    return source

def GetValueByName(self, name):
    """Retrieves a value by name.

    Value names are not unique and pyregf provides first match for the value.

    Args:
      name (str): name of the value or an empty string for the default value.

    Returns:
      WinRegistryValue: Windows Registry value if a corresponding value was
          found or None if not.
    """
    pyregf_value = self._pyregf_key.get_value_by_name(name)
    if not pyregf_value:
      return None

    return REGFWinRegistryValue(pyregf_value)

def get_input(input_func, input_str):
    """
    Get input from the user given an input function and an input string
    """
    val = input_func("Please enter your {0}: ".format(input_str))
    while not val or not len(val.strip()):
        val = input_func("You didn't enter a valid {0}, please try again: ".format(input_str))
    return val

def find_number(regex, s):
    """Find a number using a given regular expression.
    If the string cannot be found, returns None.
    The regex should contain one matching group, 
    as only the result of the first group is returned.
    The group should only contain numeric characters ([0-9]+).
    
    s - The string to search.
    regex - A string containing the regular expression.
    
    Returns an integer or None.
    """
    result = find_string(regex, s)
    if result is None:
        return None
    return int(result)

def hide(self):
        """Hides the main window of the terminal and sets the visible
        flag to False.
        """
        if not HidePrevention(self.window).may_hide():
            return
        self.hidden = True
        self.get_widget('window-root').unstick()
        self.window.hide()

def close( self ):
        """
        Close the db and release memory
        """
        if self.db is not None:
            self.db.commit()
            self.db.close()
            self.db = None

        return

def strip_comment_marker(text):
    """ Strip # markers at the front of a block of comment text.
    """
    lines = []
    for line in text.splitlines():
        lines.append(line.lstrip('#'))
    text = textwrap.dedent('\n'.join(lines))
    return text

def filter_dict_by_key(d, keys):
    """Filter the dict *d* to remove keys not in *keys*."""
    return {k: v for k, v in d.items() if k in keys}

def table_nan_locs(table):
    """
    from http://stackoverflow.com/a/14033137/623735
    # gets the indices of the rows with nan values in a dataframe
    pd.isnull(df).any(1).nonzero()[0]
    """
    ans = []
    for rownum, row in enumerate(table):
        try:
            if pd.isnull(row).any():
                colnums = pd.isnull(row).nonzero()[0]
                ans += [(rownum, colnum) for colnum in colnums]
        except AttributeError:  # table is really just a sequence of scalars
            if pd.isnull(row):
                ans += [(rownum, 0)]
    return ans

def strip_sdist_extras(filelist):
    """Strip generated files that are only present in source distributions.

    We also strip files that are ignored for other reasons, like
    command line arguments, setup.cfg rules or MANIFEST.in rules.
    """
    return [name for name in filelist
            if not file_matches(name, IGNORE)
            and not file_matches_regexps(name, IGNORE_REGEXPS)]

def coverage(ctx, opts=""):
    """
    Execute all tests (normal and slow) with coverage enabled.
    """
    return test(ctx, coverage=True, include_slow=True, opts=opts)

def _remove_blank(l):
        """ Removes trailing zeros in the list of integers and returns a new list of integers"""
        ret = []
        for i, _ in enumerate(l):
            if l[i] == 0:
                break
            ret.append(l[i])
        return ret

def init_checks_registry():
    """Register all globally visible functions.

    The first argument name is either 'physical_line' or 'logical_line'.
    """
    mod = inspect.getmodule(register_check)
    for (name, function) in inspect.getmembers(mod, inspect.isfunction):
        register_check(function)

def unpunctuate(s, *, char_blacklist=string.punctuation):
    """ Remove punctuation from string s. """
    # remove punctuation
    s = "".join(c for c in s if c not in char_blacklist)
    # remove consecutive spaces
    return " ".join(filter(None, s.split(" ")))

def interpolate_logscale_single(start, end, coefficient):
    """ Cosine interpolation """
    return np.exp(np.log(start) + (np.log(end) - np.log(start)) * coefficient)

def __normalize_list(self, msg):
        """Split message to list by commas and trim whitespace."""
        if isinstance(msg, list):
            msg = "".join(msg)
        return list(map(lambda x: x.strip(), msg.split(",")))

def _normal_prompt(self):
        """
        Flushes the prompt before requesting the input

        :return: The command line
        """
        sys.stdout.write(self.__get_ps1())
        sys.stdout.flush()
        return safe_input()

def _str_to_list(s):
    """Converts a comma separated string to a list"""
    _list = s.split(",")
    return list(map(lambda i: i.lstrip(), _list))

def open01(x, limit=1.e-6):
    """Constrain numbers to (0,1) interval"""
    try:
        return np.array([min(max(y, limit), 1. - limit) for y in x])
    except TypeError:
        return min(max(x, limit), 1. - limit)

def pop (self, key):
        """Remove key from dict and return value."""
        if key in self._keys:
            self._keys.remove(key)
        super(ListDict, self).pop(key)

def roundClosestValid(val, res, decimals=None):
        """ round to closest resolution """
        if decimals is None and "." in str(res):
            decimals = len(str(res).split('.')[1])

        return round(round(val / res) * res, decimals)

def _remove_dict_keys_with_value(dict_, val):
  """Removes `dict` keys which have have `self` as value."""
  return {k: v for k, v in dict_.items() if v is not val}

def apply_fit(xy,coeffs):
    """ Apply the coefficients from a linear fit to
        an array of x,y positions.

        The coeffs come from the 'coeffs' member of the
        'fit_arrays()' output.
    """
    x_new = coeffs[0][2] + coeffs[0][0]*xy[:,0] + coeffs[0][1]*xy[:,1]
    y_new = coeffs[1][2] + coeffs[1][0]*xy[:,0] + coeffs[1][1]*xy[:,1]

    return x_new,y_new

def clean_py_files(path):
    """
    Removes all .py files.

    :param path: the path
    :return: None
    """

    for dirname, subdirlist, filelist in os.walk(path):

        for f in filelist:
            if f.endswith('py'):
                os.remove(os.path.join(dirname, f))

def get_param_names(cls):
        """Returns a list of plottable CBC parameter variables"""
        return [m[0] for m in inspect.getmembers(cls) \
            if type(m[1]) == property]

def prune(self, n):
        """prune all but the first (=best) n items"""
        if self.minimize:
            self.data = self.data[:n]
        else:
            self.data = self.data[-1 * n:]

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

def strip_html(string, keep_tag_content=False):
    """
    Remove html code contained into the given string.

    :param string: String to manipulate.
    :type string: str
    :param keep_tag_content: True to preserve tag content, False to remove tag and its content too (default).
    :type keep_tag_content: bool
    :return: String with html removed.
    :rtype: str
    """
    r = HTML_TAG_ONLY_RE if keep_tag_content else HTML_RE
    return r.sub('', string)

def pop(h):
    """Pop the heap value from the heap."""
    n = h.size() - 1
    h.swap(0, n)
    down(h, 0, n)
    return h.pop()

def remove_legend(ax=None):
    """Remove legend for axes or gca.

    See http://osdir.com/ml/python.matplotlib.general/2005-07/msg00285.html
    """
    from pylab import gca, draw
    if ax is None:
        ax = gca()
    ax.legend_ = None
    draw()

def print_log(text, *colors):
    """Print a log message to standard error."""
    sys.stderr.write(sprint("{}: {}".format(script_name, text), *colors) + "\n")

def replaceNewlines(string, newlineChar):
	"""There's probably a way to do this with string functions but I was lazy.
		Replace all instances of \r or \n in a string with something else."""
	if newlineChar in string:
		segments = string.split(newlineChar)
		string = ""
		for segment in segments:
			string += segment
	return string

def next(self):
        """Provides hook for Python2 iterator functionality."""
        _LOGGER.debug("reading next")
        if self.closed:
            _LOGGER.debug("stream is closed")
            raise StopIteration()

        line = self.readline()
        if not line:
            _LOGGER.debug("nothing more to read")
            raise StopIteration()

        return line

def replaceNewlines(string, newlineChar):
	"""There's probably a way to do this with string functions but I was lazy.
		Replace all instances of \r or \n in a string with something else."""
	if newlineChar in string:
		segments = string.split(newlineChar)
		string = ""
		for segment in segments:
			string += segment
	return string

def _dotify(cls, data):
    """Add dots."""
    return ''.join(char if char in cls.PRINTABLE_DATA else '.' for char in data)

def clean(self, text):
        """Remove all unwanted characters from text."""
        return ''.join([c for c in text if c in self.alphabet])

def PyplotHistogram():
    """
    =============================================================
    Demo of the histogram (hist) function with multiple data sets
    =============================================================

    Plot histogram with multiple sample sets and demonstrate:

        * Use of legend with multiple sample sets
        * Stacked bars
        * Step curve with no fill
        * Data sets of different sample sizes

    Selecting different bin counts and sizes can significantly affect the
    shape of a histogram. The Astropy docs have a great section on how to
    select these parameters:
    http://docs.astropy.org/en/stable/visualization/histogram.html
    """

    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(0)

    n_bins = 10
    x = np.random.randn(1000, 3)

    fig, axes = plt.subplots(nrows=2, ncols=2)
    ax0, ax1, ax2, ax3 = axes.flatten()

    colors = ['red', 'tan', 'lime']
    ax0.hist(x, n_bins, normed=1, histtype='bar', color=colors, label=colors)
    ax0.legend(prop={'size': 10})
    ax0.set_title('bars with legend')

    ax1.hist(x, n_bins, normed=1, histtype='bar', stacked=True)
    ax1.set_title('stacked bar')

    ax2.hist(x, n_bins, histtype='step', stacked=True, fill=False)
    ax2.set_title('stack step (unfilled)')

    # Make a multiple-histogram of data-sets with different length.
    x_multi = [np.random.randn(n) for n in [10000, 5000, 2000]]
    ax3.hist(x_multi, n_bins, histtype='bar')
    ax3.set_title('different sample sizes')

    fig.tight_layout()
    return fig

def normalize_value(text):
    """
    This removes newlines and multiple spaces from a string.
    """
    result = text.replace('\n', ' ')
    result = re.subn('[ ]{2,}', ' ', result)[0]
    return result

def _heappush_max(heap, item):
    """ why is this not in heapq """
    heap.append(item)
    heapq._siftdown_max(heap, 0, len(heap) - 1)

def _remove_keywords(d):
    """
    copy the dict, filter_keywords

    Parameters
    ----------
    d : dict
    """
    return { k:v for k, v in iteritems(d) if k not in RESERVED }

def this_week():
        """ Return start and end date of the current week. """
        since = TODAY + delta(weekday=MONDAY(-1))
        until = since + delta(weeks=1)
        return Date(since), Date(until)

def clean(s):
  """Removes trailing whitespace on each line."""
  lines = [l.rstrip() for l in s.split('\n')]
  return '\n'.join(lines)

def __grid_widgets(self):
        """Places all the child widgets in the appropriate positions."""
        scrollbar_column = 0 if self.__compound is tk.LEFT else 2
        self._canvas.grid(row=0, column=1, sticky="nswe")
        self._scrollbar.grid(row=0, column=scrollbar_column, sticky="ns")

def unescape(str):
    """Undoes the effects of the escape() function."""
    out = ''
    prev_backslash = False
    for char in str:
        if not prev_backslash and char == '\\':
            prev_backslash = True
            continue
        out += char
        prev_backslash = False
    return out

def retry_call(func, cleanup=lambda: None, retries=0, trap=()):
	"""
	Given a callable func, trap the indicated exceptions
	for up to 'retries' times, invoking cleanup on the
	exception. On the final attempt, allow any exceptions
	to propagate.
	"""
	attempts = count() if retries == float('inf') else range(retries)
	for attempt in attempts:
		try:
			return func()
		except trap:
			cleanup()

	return func()

def replace_all(filepath, searchExp, replaceExp):
    """
    Replace all the ocurrences (in a file) of a string with another value.
    """
    for line in fileinput.input(filepath, inplace=1):
        if searchExp in line:
            line = line.replace(searchExp, replaceExp)
        sys.stdout.write(line)

def str2int(num, radix=10, alphabet=BASE85):
    """helper function for quick base conversions from strings to integers"""
    return NumConv(radix, alphabet).str2int(num)

def _sub_patterns(patterns, text):
    """
    Apply re.sub to bunch of (pattern, repl)
    """
    for pattern, repl in patterns:
        text = re.sub(pattern, repl, text)
    return text

def string_to_identity(identity_str):
    """Parse string into Identity dictionary."""
    m = _identity_regexp.match(identity_str)
    result = m.groupdict()
    log.debug('parsed identity: %s', result)
    return {k: v for k, v in result.items() if v}

def _replace_nan(a, val):
    """
    replace nan in a by val, and returns the replaced array and the nan
    position
    """
    mask = isnull(a)
    return where_method(val, mask, a), mask

def adapter(data, headers, **kwargs):
    """Wrap vertical table in a function for TabularOutputFormatter."""
    keys = ('sep_title', 'sep_character', 'sep_length')
    return vertical_table(data, headers, **filter_dict_by_key(kwargs, keys))

async def sysinfo(dev: Device):
    """Print out system information (version, MAC addrs)."""
    click.echo(await dev.get_system_info())
    click.echo(await dev.get_interface_information())

def _arrayFromBytes(dataBytes, metadata):
    """Generates and returns a numpy array from raw data bytes.

    :param bytes: raw data bytes as generated by ``numpy.ndarray.tobytes()``
    :param metadata: a dictionary containing the data type and optionally the
        shape parameter to reconstruct a ``numpy.array`` from the raw data
        bytes. ``{"dtype": "float64", "shape": (2, 3)}``

    :returns: ``numpy.array``
    """
    array = numpy.fromstring(dataBytes, dtype=numpy.typeDict[metadata['dtype']])
    if 'shape' in metadata:
        array = array.reshape(metadata['shape'])
    return array

def dir_exists(self):
        """
        Makes a ``HEAD`` requests to the URI.

        :returns: ``True`` if status code is 2xx.
        """

        r = requests.request(self.method if self.method else 'HEAD', self.url, **self.storage_args)
        try: r.raise_for_status()
        except Exception: return False

        return True

def csv_matrix_print(classes, table):
    """
    Return matrix as csv data.

    :param classes: classes list
    :type classes:list
    :param table: table
    :type table:dict
    :return:
    """
    result = ""
    classes.sort()
    for i in classes:
        for j in classes:
            result += str(table[i][j]) + ","
        result = result[:-1] + "\n"
    return result[:-1]

def disable_insecure_request_warning():
    """Suppress warning about untrusted SSL certificate."""
    import requests
    from requests.packages.urllib3.exceptions import InsecureRequestWarning
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

def getFunction(self):
        """Called by remote workers. Useful to populate main module globals()
        for interactive shells. Retrieves the serialized function."""
        return functionFactory(
            self.code,
            self.name,
            self.defaults,
            self.globals,
            self.imports,
        )

def make_post_request(self, url, auth, json_payload):
        """This function executes the request with the provided
        json payload and return the json response"""
        response = requests.post(url, auth=auth, json=json_payload)
        return response.json()

def login(self, username, password=None, token=None):
        """Login user for protected API calls."""
        self.session.basic_auth(username, password)

def scale_image(image, new_width):
    """Resizes an image preserving the aspect ratio.
    """
    (original_width, original_height) = image.size
    aspect_ratio = original_height/float(original_width)
    new_height = int(aspect_ratio * new_width)

    # This scales it wider than tall, since characters are biased
    new_image = image.resize((new_width*2, new_height))
    return new_image

def requests_request(method, url, **kwargs):
    """Requests-mock requests.request wrapper."""
    session = local_sessions.session
    response = session.request(method=method, url=url, **kwargs)
    session.close()
    return response

def to_camel(s):
    """
    :param string s: under_scored string to be CamelCased
    :return: CamelCase version of input
    :rtype: str
    """
    # r'(?!^)_([a-zA-Z]) original regex wasn't process first groups
    return re.sub(r'_([a-zA-Z])', lambda m: m.group(1).upper(), '_' + s)

def reset_params(self):
        """Reset all parameters to their default values."""
        self.__params = dict([p, None] for p in self.param_names)
        self.set_params(self.param_defaults)

def make_post_request(self, url, auth, json_payload):
        """This function executes the request with the provided
        json payload and return the json response"""
        response = requests.post(url, auth=auth, json=json_payload)
        return response.json()

def arr_to_vector(arr):
    """Reshape a multidimensional array to a vector.
    """
    dim = array_dim(arr)
    tmp_arr = []
    for n in range(len(dim) - 1):
        for inner in arr:
            for i in inner:
                tmp_arr.append(i)
        arr = tmp_arr
        tmp_arr = []
    return arr

def _digits(minval, maxval):
    """Digits needed to comforatbly display values in [minval, maxval]"""
    if minval == maxval:
        return 3
    else:
        return min(10, max(2, int(1 + abs(np.log10(maxval - minval)))))

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def positive_integer(anon, obj, field, val):
    """
    Returns a random positive integer (for a Django PositiveIntegerField)
    """
    return anon.faker.positive_integer(field=field)

def get_func_name(func):
    """Return a name which includes the module name and function name."""
    func_name = getattr(func, '__name__', func.__class__.__name__)
    module_name = func.__module__

    if module_name is not None:
        module_name = func.__module__
        return '{}.{}'.format(module_name, func_name)

    return func_name

def fix_call(callable, *args, **kw):
    """
    Call ``callable(*args, **kw)`` fixing any type errors that come out.
    """
    try:
        val = callable(*args, **kw)
    except TypeError:
        exc_info = fix_type_error(None, callable, args, kw)
        reraise(*exc_info)
    return val

def distinct(l):
    """
    Return a list where the duplicates have been removed.

    Args:
        l (list): the list to filter.

    Returns:
        list: the same list without duplicates.
    """
    seen = set()
    seen_add = seen.add
    return (_ for _ in l if not (_ in seen or seen_add(_)))

def print_result_from_timeit(stmt='pass', setup='pass', number=1000000):
    """
    Clean function to know how much time took the execution of one statement
    """
    units = ["s", "ms", "us", "ns"]
    duration = timeit(stmt, setup, number=int(number))
    avg_duration = duration / float(number)
    thousands = int(math.floor(math.log(avg_duration, 1000)))

    print("Total time: %fs. Average run: %.3f%s." % (
        duration, avg_duration * (1000 ** -thousands), units[-thousands]))

def _find(string, sub_string, start_index):
    """Return index of sub_string in string.

    Raise TokenError if sub_string is not found.
    """
    result = string.find(sub_string, start_index)
    if result == -1:
        raise TokenError("expected '{0}'".format(sub_string))
    return result

def return_letters_from_string(text):
    """Get letters from string only."""
    out = ""
    for letter in text:
        if letter.isalpha():
            out += letter
    return out

def move_back(self, dt):
        """ If called after an update, the sprite can move back
        """
        self._position = self._old_position
        self.rect.topleft = self._position
        self.feet.midbottom = self.rect.midbottom

def binSearch(arr, val):
  """ 
  Function for running binary search on a sorted list.

  :param arr: (list) a sorted list of integers to search
  :param val: (int)  a integer to search for in the sorted array
  :returns: (int) the index of the element if it is found and -1 otherwise.
  """
  i = bisect_left(arr, val)
  if i != len(arr) and arr[i] == val:
    return i
  return -1

def list_move_to_front(l,value='other'):
    """if the value is in the list, move it to the front and return it."""
    l=list(l)
    if value in l:
        l.remove(value)
        l.insert(0,value)
    return l

def find_nearest_index(arr, value):
    """For a given value, the function finds the nearest value
    in the array and returns its index."""
    arr = np.array(arr)
    index = (abs(arr-value)).argmin()
    return index

def multiply(self, number):
        """Return a Vector as the product of the vector and a real number."""
        return self.from_list([x * number for x in self.to_list()])

def closest(xarr, val):
    """ Return the index of the closest in xarr to value val """
    idx_closest = np.argmin(np.abs(np.array(xarr) - val))
    return idx_closest

def v_normalize(v):
    """
    Normalizes the given vector.
    
    The vector given may have any number of dimensions.
    """
    vmag = v_magnitude(v)
    return [ v[i]/vmag  for i in range(len(v)) ]

def invertDictMapping(d):
    """ Invert mapping of dictionary (i.e. map values to list of keys) """
    inv_map = {}
    for k, v in d.items():
        inv_map[v] = inv_map.get(v, [])
        inv_map[v].append(k)
    return inv_map

def load(filename):
    """
    Load the state from the given file, moving to the file's directory during
    load (temporarily, moving back after loaded)

    Parameters
    ----------
    filename : string
        name of the file to open, should be a .pkl file
    """
    path, name = os.path.split(filename)
    path = path or '.'

    with util.indir(path):
        return pickle.load(open(name, 'rb'))

def rotation_matrix(sigma):
    """

    https://en.wikipedia.org/wiki/Rotation_matrix

    """

    radians = sigma * np.pi / 180.0

    r11 = np.cos(radians)
    r12 = -np.sin(radians)
    r21 = np.sin(radians)
    r22 = np.cos(radians)

    R = np.array([[r11, r12], [r21, r22]])

    return R

def readme(filename, encoding='utf8'):
    """
    Read the contents of a file
    """

    with io.open(filename, encoding=encoding) as source:
        return source.read()

def _saferound(value, decimal_places):
    """
    Rounds a float value off to the desired precision
    """
    try:
        f = float(value)
    except ValueError:
        return ''
    format = '%%.%df' % decimal_places
    return format % f

def bin_open(fname: str):
    """
    Returns a file descriptor for a plain text or gzipped file, binary read mode
    for subprocess interaction.

    :param fname: The filename to open.
    :return: File descriptor in binary read mode.
    """
    if fname.endswith(".gz"):
        return gzip.open(fname, "rb")
    return open(fname, "rb")

def round_array(array_in):
    """
    arr_out = round_array(array_in)

    Rounds an array and recasts it to int. Also works on scalars.
    """
    if isinstance(array_in, ndarray):
        return np.round(array_in).astype(int)
    else:
        return int(np.round(array_in))

def jsonify(symbol):
    """ returns json format for symbol """
    try:
        # all symbols have a toJson method, try it
        return json.dumps(symbol.toJson(), indent='  ')
    except AttributeError:
        pass
    return json.dumps(symbol, indent='  ')

def round_array(array_in):
    """
    arr_out = round_array(array_in)

    Rounds an array and recasts it to int. Also works on scalars.
    """
    if isinstance(array_in, ndarray):
        return np.round(array_in).astype(int)
    else:
        return int(np.round(array_in))

def _visual_width(line):
    """Get the the number of columns required to display a string"""

    return len(re.sub(colorama.ansitowin32.AnsiToWin32.ANSI_CSI_RE, "", line))

def price_rounding(price, decimals=2):
    """Takes a decimal price and rounds to a number of decimal places"""
    try:
        exponent = D('.' + decimals * '0')
    except InvalidOperation:
        # Currencies with no decimal places, ex. JPY, HUF
        exponent = D()
    return price.quantize(exponent, rounding=ROUND_UP)

def copy_image_on_background(image, color=WHITE):
    """
    Create a new image by copying the image on a *color* background.

    Args:
        image (PIL.Image.Image): Image to copy
        color (tuple): Background color usually WHITE or BLACK

    Returns:
        PIL.Image.Image

    """
    background = Image.new("RGB", image.size, color)
    background.paste(image, mask=image.split()[3])
    return background

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def pstd(self, *args, **kwargs):
        """ Console to STDOUT """
        kwargs['file'] = self.out
        self.print(*args, **kwargs)
        sys.stdout.flush()

def remove_file_from_s3(awsclient, bucket, key):
    """Remove a file from an AWS S3 bucket.

    :param awsclient:
    :param bucket:
    :param key:
    :return:
    """
    client_s3 = awsclient.get_client('s3')
    response = client_s3.delete_object(Bucket=bucket, Key=key)

def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return F.pad(img, self.padding, self.fill, self.padding_mode)

def get_as_string(self, s3_path, encoding='utf-8'):
        """
        Get the contents of an object stored in S3 as string.

        :param s3_path: URL for target S3 location
        :param encoding: Encoding to decode bytes to string
        :return: File contents as a string
        """
        content = self.get_as_bytes(s3_path)
        return content.decode(encoding)

def trigger(self, target: str, trigger: str, parameters: Dict[str, Any]={}):
		"""Calls the specified Trigger of another Area with the optionally given parameters.

		Args:
			target: The name of the target Area.
			trigger: The name of the Trigger.
			parameters: The parameters of the function call.
		"""
		pass

def base64ToImage(imgData, out_path, out_file):
        """ converts a base64 string to a file """
        fh = open(os.path.join(out_path, out_file), "wb")
        fh.write(imgData.decode('base64'))
        fh.close()
        del fh
        return os.path.join(out_path, out_file)

def strip_figures(figure):
	"""
	Strips a figure into multiple figures with a trace on each of them

	Parameters:
	-----------
		figure : Figure
			Plotly Figure
	"""
	fig=[]
	for trace in figure['data']:
		fig.append(dict(data=[trace],layout=figure['layout']))
	return fig

def draw(graph, fname):
    """Draw a graph and save it into a file"""
    ag = networkx.nx_agraph.to_agraph(graph)
    ag.draw(fname, prog='dot')

def head(filename, n=10):
    """ prints the top `n` lines of a file """
    with freader(filename) as fr:
        for _ in range(n):
            print(fr.readline().strip())

def _sslobj(sock):
    """Returns the underlying PySLLSocket object with which the C extension
    functions interface.

    """
    pass
    if isinstance(sock._sslobj, _ssl._SSLSocket):
        return sock._sslobj
    else:
        return sock._sslobj._sslobj

def see_doc(obj_with_doc):
    """Copy docstring from existing object to the decorated callable."""
    def decorator(fn):
        fn.__doc__ = obj_with_doc.__doc__
        return fn
    return decorator

def scatter(self, *args, **kwargs):
        """Add a scatter plot."""
        cls = _make_class(ScatterVisual,
                          _default_marker=kwargs.pop('marker', None),
                          )
        return self._add_item(cls, *args, **kwargs)

def printdict(adict):
    """printdict"""
    dlist = list(adict.keys())
    dlist.sort()
    for i in range(0, len(dlist)):
        print(dlist[i], adict[dlist[i]])

def _cho_factor(A, lower=True, check_finite=True):
    """Implementaton of :func:`scipy.linalg.cho_factor` using
    a function supported in cupy."""

    return cp.linalg.cholesky(A), True

def csv_matrix_print(classes, table):
    """
    Return matrix as csv data.

    :param classes: classes list
    :type classes:list
    :param table: table
    :type table:dict
    :return:
    """
    result = ""
    classes.sort()
    for i in classes:
        for j in classes:
            result += str(table[i][j]) + ","
        result = result[:-1] + "\n"
    return result[:-1]

def ex(self, cmd):
        """Execute a normal python statement in user namespace."""
        with self.builtin_trap:
            exec cmd in self.user_global_ns, self.user_ns

def printdict(adict):
    """printdict"""
    dlist = list(adict.keys())
    dlist.sort()
    for i in range(0, len(dlist)):
        print(dlist[i], adict[dlist[i]])

def handle_logging(self):
        """
        To allow devs to log as early as possible, logging will already be
        handled here
        """

        configure_logging(self.get_scrapy_options())

        # Disable duplicates
        self.__scrapy_options["LOG_ENABLED"] = False

        # Now, after log-level is correctly set, lets log them.
        for msg in self.log_output:
            if msg["level"] is "error":
                self.log.error(msg["msg"])
            elif msg["level"] is "info":
                self.log.info(msg["msg"])
            elif msg["level"] is "debug":
                self.log.debug(msg["msg"])

def translate_index_to_position(self, index):
        """
        Given an index for the text, return the corresponding (row, col) tuple.
        (0-based. Returns (0, 0) for index=0.)
        """
        # Find start of this line.
        row, row_index = self._find_line_start_index(index)
        col = index - row_index

        return row, col

def printc(cls, txt, color=colors.red):
        """Print in color."""
        print(cls.color_txt(txt, color))

def time2seconds(t):
    """Returns seconds since 0h00."""
    return t.hour * 3600 + t.minute * 60 + t.second + float(t.microsecond) / 1e6

def format_exception(e):
    """Returns a string containing the type and text of the exception.

    """
    from .utils.printing import fill
    return '\n'.join(fill(line) for line in traceback.format_exception_only(type(e), e))

def json(body, charset='utf-8', **kwargs):
    """Takes JSON formatted data, converting it into native Python objects"""
    return json_converter.loads(text(body, charset=charset))

def on_mouse_motion(self, x, y, dx, dy):
        """
        Pyglet specific mouse motion callback.
        Forwards and traslates the event to the example
        """
        # Screen coordinates relative to the lower-left corner
        # so we have to flip the y axis to make this consistent with
        # other window libraries
        self.example.mouse_position_event(x, self.buffer_height - y)

def _file_exists(path, filename):
  """Checks if the filename exists under the path."""
  return os.path.isfile(os.path.join(path, filename))

def mpl_outside_legend(ax, **kwargs):
    """ Places a legend box outside a matplotlib Axes instance. """
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), **kwargs)

def is_delimiter(line):
    """ True if a line consists only of a single punctuation character."""
    return bool(line) and line[0] in punctuation and line[0]*len(line) == line

def _add_hash(source):
    """Add a leading hash '#' at the beginning of every line in the source."""
    source = '\n'.join('# ' + line.rstrip()
                       for line in source.splitlines())
    return source

def full(self):
        """Return True if the queue is full"""
        if not self.size: return False
        return len(self.pq) == (self.size + self.removed_count)

def get_randomized_guid_sample(self, item_count):
        """ Fetch a subset of randomzied GUIDs from the whitelist """
        dataset = self.get_whitelist()
        random.shuffle(dataset)
        return dataset[:item_count]

def selecttrue(table, field, complement=False):
    """Select rows where the given field evaluates `True`."""

    return select(table, field, lambda v: bool(v), complement=complement)

def downsample(array, k):
    """Choose k random elements of array."""
    length = array.shape[0]
    indices = random.sample(xrange(length), k)
    return array[indices]

def adapt_array(arr):
    """
    Adapts a Numpy array into an ARRAY string to put into the database.

    Parameters
    ----------
    arr: array
        The Numpy array to be adapted into an ARRAY type that can be inserted into a SQL file.

    Returns
    -------
    ARRAY
            The adapted array object

    """
    out = io.BytesIO()
    np.save(out, arr), out.seek(0)
    return buffer(out.read())

def read_numpy(fd, byte_order, dtype, count):
    """Read tag data from file and return as numpy array."""
    return numpy.fromfile(fd, byte_order+dtype[-1], count)

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

def ReadTif(tifFile):
        """Reads a tif file to a 2D NumPy array"""
        img = Image.open(tifFile)
        img = np.array(img)
        return img

def describe_unique_1d(series):
    """Compute summary statistics of a unique (`S_TYPE_UNIQUE`) variable (a Series).

    Parameters
    ----------
    series : Series
        The variable to describe.

    Returns
    -------
    Series
        The description of the variable as a Series with index being stats keys.
    """
    return pd.Series([base.S_TYPE_UNIQUE], index=['type'], name=series.name)

def read_numpy(fd, byte_order, dtype, count):
    """Read tag data from file and return as numpy array."""
    return numpy.fromfile(fd, byte_order+dtype[-1], count)

def register_service(self, service):
        """
            Register service into the system. Called by Services.
        """
        if service not in self.services:
            self.services.append(service)

def get_list_from_file(file_name):
    """read the lines from a file into a list"""
    with open(file_name, mode='r', encoding='utf-8') as f1:
        lst = f1.readlines()
    return lst

def to_snake_case(text):
    """Convert to snake case.

    :param str text:
    :rtype: str
    :return:
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def get_list_from_file(file_name):
    """read the lines from a file into a list"""
    with open(file_name, mode='r', encoding='utf-8') as f1:
        lst = f1.readlines()
    return lst

def delete_duplicates(seq):
    """
    Remove duplicates from an iterable, preserving the order.

    Args:
        seq: Iterable of various type.

    Returns:
        list: List of unique objects.

    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def load_data(filename):
    """
    :rtype : numpy matrix
    """
    data = pandas.read_csv(filename, header=None, delimiter='\t', skiprows=9)
    return data.as_matrix()

def unique(iterable):
    """ Returns a list copy in which each item occurs only once (in-order).
    """
    seen = set()
    return [x for x in iterable if x not in seen and not seen.add(x)]

def find_le(a, x):
    """Find rightmost value less than or equal to x."""
    i = bs.bisect_right(a, x)
    if i: return i - 1
    raise ValueError

def setPixel(self, x, y, color):
        """Set the pixel at (x,y) to the integers in sequence 'color'."""
        return _fitz.Pixmap_setPixel(self, x, y, color)

def lighting(im, b, c):
    """ Adjust image balance and contrast """
    if b==0 and c==1: return im
    mu = np.average(im)
    return np.clip((im-mu)*c+mu+b,0.,1.).astype(np.float32)

def reseed_random(seed):
    """Reseed factory.fuzzy's random generator."""
    r = random.Random(seed)
    random_internal_state = r.getstate()
    set_random_state(random_internal_state)

def round_sig(x, sig):
    """Round the number to the specified number of significant figures"""
    return round(x, sig - int(floor(log10(abs(x)))) - 1)

def __isub__(self, other):
        """Remove all elements of another set from this RangeSet."""
        self._binary_sanity_check(other)
        set.difference_update(self, other)
        return self

def getTuple(self):
        """ Returns the shape of the region as (x, y, w, h) """
        return (self.x, self.y, self.w, self.h)

def set_font_size(self, size):
        """Convenience method for just changing font size."""
        if self.font.font_size == size:
            pass
        else:
            self.font._set_size(size)

def update_screen(self):
        """Refresh the screen. You don't need to override this except to update only small portins of the screen."""
        self.clock.tick(self.FPS)
        pygame.display.update()

def remove_ext(fname):
    """Removes the extension from a filename
    """
    bn = os.path.basename(fname)
    return os.path.splitext(bn)[0]

def strip_accents(string):
    """
    Strip all the accents from the string
    """
    return u''.join(
        (character for character in unicodedata.normalize('NFD', string)
         if unicodedata.category(character) != 'Mn'))

def main(argv, version=DEFAULT_VERSION):
    """Install or upgrade setuptools and EasyInstall"""
    tarball = download_setuptools()
    _install(tarball, _build_install_args(argv))

def strip_accents(string):
    """
    Strip all the accents from the string
    """
    return u''.join(
        (character for character in unicodedata.normalize('NFD', string)
         if unicodedata.category(character) != 'Mn'))

def get_closest_index(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.
    If two numbers are equally close, return the smallest number.

    Parameters
    ----------
    myList : array
        The list in which to find the closest value to myNumber
    myNumber : float
        The number to find the closest to in MyList

    Returns
    -------
    closest_values_index : int
        The index in the array of the number closest to myNumber in myList
    """
    closest_values_index = _np.where(self.time == take_closest(myList, myNumber))[0][0]
    return closest_values_index

def strip_accents(text):
    """
    Strip agents from a string.
    """

    normalized_str = unicodedata.normalize('NFD', text)

    return ''.join([
        c for c in normalized_str if unicodedata.category(c) != 'Mn'])

def distinct(xs):
    """Get the list of distinct values with preserving order."""
    # don't use collections.OrderedDict because we do support Python 2.6
    seen = set()
    return [x for x in xs if x not in seen and not seen.add(x)]

def list_move_to_front(l,value='other'):
    """if the value is in the list, move it to the front and return it."""
    l=list(l)
    if value in l:
        l.remove(value)
        l.insert(0,value)
    return l

def safe_rmtree(directory):
  """Delete a directory if it's present. If it's not present, no-op."""
  if os.path.exists(directory):
    shutil.rmtree(directory, True)

def signal_handler(signal_name, frame):
    """Quit signal handler."""
    sys.stdout.flush()
    print("\nSIGINT in frame signal received. Quitting...")
    sys.stdout.flush()
    sys.exit(0)

def cleanLines(source, lineSep=os.linesep):
    """
    :param source: some iterable source (list, file, etc)
    :param lineSep: string of separators (chars) that must be removed
    :return: list of non empty lines with removed separators
    """
    stripped = (line.strip(lineSep) for line in source)
    return (line for line in stripped if len(line) != 0)

def remove_dups(seq):
    """remove duplicates from a sequence, preserving order"""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def __delitem__(self, resource):
        """Remove resource instance from internal cache"""
        self.__caches[type(resource)].pop(resource.get_cache_internal_key(), None)

def partial_fit(self, X, y=None, classes=None, **fit_params):
        """Fit the module.

        If the module is initialized, it is not re-initialized, which
        means that this method should be used if you want to continue
        training a model (warm start).

        Parameters
        ----------
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:

            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset

          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.

        y : target data, compatible with skorch.dataset.Dataset
          The same data types as for ``X`` are supported. If your X is
          a Dataset that contains the target, ``y`` may be set to
          None.

        classes : array, sahpe (n_classes,)
          Solely for sklearn compatibility, currently unused.

        **fit_params : dict
          Additional parameters passed to the ``forward`` method of
          the module and to the ``self.train_split`` call.

        """
        if not self.initialized_:
            self.initialize()

        self.notify('on_train_begin', X=X, y=y)
        try:
            self.fit_loop(X, y, **fit_params)
        except KeyboardInterrupt:
            pass
        self.notify('on_train_end', X=X, y=y)
        return self

def split_comma_argument(comma_sep_str):
    """Split a comma separated option into a list."""
    terms = []
    for term in comma_sep_str.split(','):
        if term:
            terms.append(term)
    return terms

def getbyteslice(self, start, end):
        """Direct access to byte data."""
        c = self._rawarray[start:end]
        return c

def _remove_empty_items(d, required):
  """Return a new dict with any empty items removed.

  Note that this is not a deep check. If d contains a dictionary which
  itself contains empty items, those are never checked.

  This method exists to make to_serializable() functions cleaner.
  We could revisit this some day, but for now, the serialized objects are
  stripped of empty values to keep the output YAML more compact.

  Args:
    d: a dictionary
    required: list of required keys (for example, TaskDescriptors always emit
      the "task-id", even if None)

  Returns:
    A dictionary with empty items removed.
  """

  new_dict = {}
  for k, v in d.items():
    if k in required:
      new_dict[k] = v
    elif isinstance(v, int) or v:
      # "if v" would suppress emitting int(0)
      new_dict[k] = v

  return new_dict

def getbyteslice(self, start, end):
        """Direct access to byte data."""
        c = self._rawarray[start:end]
        return c

def clean_colnames(df):
    """ Cleans the column names on a DataFrame
    Parameters:
    df - DataFrame
        The DataFrame to clean
    """
    col_list = []
    for index in range(_dutils.cols(df)):
        col_list.append(df.columns[index].strip().lower().replace(' ','_'))
    df.columns = col_list

def partition(a, sz): 
    """splits iterables a in equal parts of size sz"""
    return [a[i:i+sz] for i in range(0, len(a), sz)]

def split(s):
  """Uses dynamic programming to infer the location of spaces in a string without spaces."""
  l = [_split(x) for x in _SPLIT_RE.split(s)]
  return [item for sublist in l for item in sublist]

def chunks(iterable, chunk):
    """Yield successive n-sized chunks from an iterable."""
    for i in range(0, len(iterable), chunk):
        yield iterable[i:i + chunk]

def retry_test(func):
    """Retries the passed function 3 times before failing"""
    success = False
    ex = Exception("Unknown")
    for i in six.moves.range(3):
        try:
            result = func()
        except Exception as e:
            time.sleep(1)
            ex = e
        else:
            success = True
            break
    if not success:
        raise ex
    assert success
    return result

def generate_unique_host_id():
    """Generate a unique ID, that is somewhat guaranteed to be unique among all
    instances running at the same time."""
    host = ".".join(reversed(socket.gethostname().split(".")))
    pid = os.getpid()
    return "%s.%d" % (host, pid)

def dashrepl(value):
    """
    Replace any non-word characters with a dash.
    """
    patt = re.compile(r'\W', re.UNICODE)
    return re.sub(patt, '-', value)

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

def underscore(text):
    """Converts text that may be camelcased into an underscored format"""
    return UNDERSCORE[1].sub(r'\1_\2', UNDERSCORE[0].sub(r'\1_\2', text)).lower()

def algo_exp(x, m, t, b):
    """mono-exponential curve."""
    return m*np.exp(-t*x)+b

def replace_all(filepath, searchExp, replaceExp):
    """
    Replace all the ocurrences (in a file) of a string with another value.
    """
    for line in fileinput.input(filepath, inplace=1):
        if searchExp in line:
            line = line.replace(searchExp, replaceExp)
        sys.stdout.write(line)

def _stdout_raw(self, s):
        """Writes the string to stdout"""
        print(s, end='', file=sys.stdout)
        sys.stdout.flush()

def get_neg_infinity(dtype):
    """Return an appropriate positive infinity for this dtype.

    Parameters
    ----------
    dtype : np.dtype

    Returns
    -------
    fill_value : positive infinity value corresponding to this dtype.
    """
    if issubclass(dtype.type, (np.floating, np.integer)):
        return -np.inf

    if issubclass(dtype.type, np.complexfloating):
        return -np.inf - 1j * np.inf

    return NINF

def csort(objs, key):
    """Order-preserving sorting function."""
    idxs = dict((obj, i) for (i, obj) in enumerate(objs))
    return sorted(objs, key=lambda obj: (key(obj), idxs[obj]))

def _sanitize(text):
    """Return sanitized Eidos text field for human readability."""
    d = {'-LRB-': '(', '-RRB-': ')'}
    return re.sub('|'.join(d.keys()), lambda m: d[m.group(0)], text)

def unique_list_dicts(dlist, key):
    """Return a list of dictionaries which are sorted for only unique entries.

    :param dlist:
    :param key:
    :return list:
    """

    return list(dict((val[key], val) for val in dlist).values())

def _maybe_fill(arr, fill_value=np.nan):
    """
    if we have a compatible fill_value and arr dtype, then fill
    """
    if _isna_compat(arr, fill_value):
        arr.fill(fill_value)
    return arr

def sort_filenames(filenames):
    """
    sort a list of files by filename only, ignoring the directory names
    """
    basenames = [os.path.basename(x) for x in filenames]
    indexes = [i[0] for i in sorted(enumerate(basenames), key=lambda x:x[1])]
    return [filenames[x] for x in indexes]

def internal_reset(self):
        """
        internal state reset.
        used e.g. in unittests
        """
        log.critical("PIA internal_reset()")
        self.empty_key_toggle = True
        self.current_input_char = None
        self.input_repead = 0

def round_to_int(number, precision):
    """Round a number to a precision"""
    precision = int(precision)
    rounded = (int(number) + precision / 2) // precision * precision
    return rounded

def cmd_reindex():
    """Uses CREATE INDEX CONCURRENTLY to create a duplicate index, then tries to swap the new index for the original.

    The index swap is done using a short lock timeout to prevent it from interfering with running queries. Retries until
    the rename succeeds.
    """
    db = connect(args.database)
    for idx in args.indexes:
        pg_reindex(db, idx)

def returns(self):
        """The return type for this method in a JSON-compatible format.

        This handles the special case of ``None`` which allows ``type(None)`` also.

        :rtype: str | None
        """
        return_type = self.signature.return_type
        none_type = type(None)
        if return_type is not None and return_type is not none_type:
            return return_type.__name__

def im2mat(I):
    """Converts and image to matrix (one pixel per line)"""
    return I.reshape((I.shape[0] * I.shape[1], I.shape[2]))

def Fsphere(q, R):
    """Scattering form-factor amplitude of a sphere normalized to F(q=0)=V

    Inputs:
    -------
        ``q``: independent variable
        ``R``: sphere radius

    Formula:
    --------
        ``4*pi/q^3 * (sin(qR) - qR*cos(qR))``
    """
    return 4 * np.pi / q ** 3 * (np.sin(q * R) - q * R * np.cos(q * R))

def im2mat(I):
    """Converts and image to matrix (one pixel per line)"""
    return I.reshape((I.shape[0] * I.shape[1], I.shape[2]))

def process_docstring(app, what, name, obj, options, lines):
    """Process the docstring for a given python object.

    Called when autodoc has read and processed a docstring. `lines` is a list
    of docstring lines that `_process_docstring` modifies in place to change
    what Sphinx outputs.

    The following settings in conf.py control what styles of docstrings will
    be parsed:

    * ``napoleon_google_docstring`` -- parse Google style docstrings
    * ``napoleon_numpy_docstring`` -- parse NumPy style docstrings

    Parameters
    ----------
    app : sphinx.application.Sphinx
        Application object representing the Sphinx process.
    what : str
        A string specifying the type of the object to which the docstring
        belongs. Valid values: "module", "class", "exception", "function",
        "method", "attribute".
    name : str
        The fully qualified name of the object.
    obj : module, class, exception, function, method, or attribute
        The object to which the docstring belongs.
    options : sphinx.ext.autodoc.Options
        The options given to the directive: an object with attributes
        inherited_members, undoc_members, show_inheritance and noindex that
        are True if the flag option of same name was given to the auto
        directive.
    lines : list of str
        The lines of the docstring, see above.

        .. note:: `lines` is modified *in place*

    Notes
    -----
    This function is (to most parts) taken from the :mod:`sphinx.ext.napoleon`
    module, sphinx version 1.3.1, and adapted to the classes defined here"""
    result_lines = lines
    if app.config.napoleon_numpy_docstring:
        docstring = ExtendedNumpyDocstring(
            result_lines, app.config, app, what, name, obj, options)
        result_lines = docstring.lines()
    if app.config.napoleon_google_docstring:
        docstring = ExtendedGoogleDocstring(
            result_lines, app.config, app, what, name, obj, options)
        result_lines = docstring.lines()

    lines[:] = result_lines[:]

def parse_code(url):
    """
    Parse the code parameter from the a URL

    :param str url: URL to parse
    :return: code query parameter
    :rtype: str
    """
    result = urlparse(url)
    query = parse_qs(result.query)
    return query['code']

def partition(a, sz): 
    """splits iterables a in equal parts of size sz"""
    return [a[i:i+sz] for i in range(0, len(a), sz)]

def json_response(data, status=200):
    """Return a JsonResponse. Make sure you have django installed first."""
    from django.http import JsonResponse
    return JsonResponse(data=data, status=status, safe=isinstance(data, dict))

def chunked(l, n):
    """Chunk one big list into few small lists."""
    return [l[i:i + n] for i in range(0, len(l), n)]

def prompt_yes_or_no(message):
    """ prompt_yes_or_no: Prompt user to reply with a y/n response
        Args: None
        Returns: None
    """
    user_input = input("{} [y/n]:".format(message)).lower()
    if user_input.startswith("y"):
        return True
    elif user_input.startswith("n"):
        return False
    else:
        return prompt_yes_or_no(message)

def split_multiline(value):
    """Split a multiline string into a list, excluding blank lines."""
    return [element for element in (line.strip() for line in value.split('\n'))
            if element]

def get_file_string(filepath):
    """Get string from file."""
    with open(os.path.abspath(filepath)) as f:
        return f.read()

def _split_str(s, n):
    """
    split string into list of strings by specified number.
    """
    length = len(s)
    return [s[i:i + n] for i in range(0, length, n)]

def get_sql(query):
    """ Returns the sql query """
    sql = str(query.statement.compile(dialect=sqlite.dialect(),
                                      compile_kwargs={"literal_binds": True}))
    return sql

def rotate_point(xorigin, yorigin, x, y, angle):
    """Rotate the given point by angle
    """
    rotx = (x - xorigin) * np.cos(angle) - (y - yorigin) * np.sin(angle)
    roty = (x - yorigin) * np.sin(angle) + (y - yorigin) * np.cos(angle)
    return rotx, roty

def forceupdate(self, *args, **kw):
        """Like a bulk :meth:`forceput`."""
        self._update(False, self._ON_DUP_OVERWRITE, *args, **kw)

def _shutdown_transport(self):
        """Unwrap a Python 2.6 SSL socket, so we can call shutdown()"""
        if self.sock is not None:
            try:
                unwrap = self.sock.unwrap
            except AttributeError:
                return
            try:
                self.sock = unwrap()
            except ValueError:
                # Failure within SSL might mean unwrap exists but socket is not
                # deemed wrapped
                pass

def rotateImage(img, angle):
    """

    querries scipy.ndimage.rotate routine
    :param img: image to be rotated
    :param angle: angle to be rotated (radian)
    :return: rotated image
    """
    imgR = scipy.ndimage.rotate(img, angle, reshape=False)
    return imgR

def column_stack_2d(data):
    """Perform column-stacking on a list of 2d data blocks."""
    return list(list(itt.chain.from_iterable(_)) for _ in zip(*data))

def round_figures(x, n):
    """Returns x rounded to n significant figures."""
    return round(x, int(n - math.ceil(math.log10(abs(x)))))

def start():
    """Starts the web server."""
    global app
    bottle.run(app, host=conf.WebHost, port=conf.WebPort,
               debug=conf.WebAutoReload, reloader=conf.WebAutoReload,
               quiet=conf.WebQuiet)

def round_sig(x, sig):
    """Round the number to the specified number of significant figures"""
    return round(x, sig - int(floor(log10(abs(x)))) - 1)

def _callable_once(func):
    """ Returns a function that is only callable once; any other call will do nothing """

    def once(*args, **kwargs):
        if not once.called:
            once.called = True
            return func(*args, **kwargs)

    once.called = False
    return once

def command_py2to3(args):
    """
    Apply '2to3' tool (Python2 to Python3 conversion tool) to Python sources.
    """
    from lib2to3.main import main
    sys.exit(main("lib2to3.fixes", args=args.sources))

def getFunction(self):
        """Called by remote workers. Useful to populate main module globals()
        for interactive shells. Retrieves the serialized function."""
        return functionFactory(
            self.code,
            self.name,
            self.defaults,
            self.globals,
            self.imports,
        )

def get_code(module):
    """
    Compile and return a Module's code object.
    """
    fp = open(module.path)
    try:
        return compile(fp.read(), str(module.name), 'exec')
    finally:
        fp.close()

def _session_set(self, key, value):
        """
        Saves a value to session.
        """

        self.session[self._session_key(key)] = value

def test():
    """Run the unit tests."""
    import unittest
    tests = unittest.TestLoader().discover('tests')
    unittest.TextTestRunner(verbosity=2).run(tests)

def __str__(self):
        """Executes self.function to convert LazyString instance to a real
        str."""
        if not hasattr(self, '_str'):
            self._str=self.function(*self.args, **self.kwargs)
        return self._str

def write_json_corpus(documents, fnm):
    """Write a lisst of Text instances as JSON corpus on disk.
    A JSON corpus contains one document per line, encoded in JSON.

    Parameters
    ----------
    documents: iterable of estnltk.text.Text
        The documents of the corpus
    fnm: str
        The path to save the corpus.
    """
    with codecs.open(fnm, 'wb', 'ascii') as f:
        for document in documents:
            f.write(json.dumps(document) + '\n')
    return documents

def quote(self, s):
        """Return a shell-escaped version of the string s."""

        if six.PY2:
            from pipes import quote
        else:
            from shlex import quote

        return quote(s)

def save_keras_definition(keras_model, path):
    """
    Save a Keras model definition to JSON with given path
    """
    model_json = keras_model.to_json()
    with open(path, "w") as json_file:
        json_file.write(model_json)

def strip_accents(text):
    """
    Strip agents from a string.
    """

    normalized_str = unicodedata.normalize('NFD', text)

    return ''.join([
        c for c in normalized_str if unicodedata.category(c) != 'Mn'])

def save(variable, filename):
    """Save variable on given path using Pickle
    
    Args:
        variable: what to save
        path (str): path of the output
    """
    fileObj = open(filename, 'wb')
    pickle.dump(variable, fileObj)
    fileObj.close()

def file_to_png(fp):
	"""Convert an image to PNG format with Pillow.
	
	:arg file-like fp: The image file.
	:rtype: bytes
	"""
	import PIL.Image # pylint: disable=import-error
	with io.BytesIO() as dest:
		PIL.Image.open(fp).save(dest, "PNG", optimize=True)
		return dest.getvalue()

def str_dict(some_dict):
    """Convert dict of ascii str/unicode to dict of str, if necessary"""
    return {str(k): str(v) for k, v in some_dict.items()}

def is_cached(file_name):
	"""
	Check if a given file is available in the cache or not
	"""

	gml_file_path = join(join(expanduser('~'), OCTOGRID_DIRECTORY), file_name)

	return isfile(gml_file_path)

def clean_float(v):
    """Remove commas from a float"""

    if v is None or not str(v).strip():
        return None

    return float(str(v).replace(',', ''))

def test():        
    """Local test."""
    from spyder.utils.qthelpers import qapplication
    app = qapplication()
    dlg = ProjectDialog(None)
    dlg.show()
    sys.exit(app.exec_())

def findLastCharIndexMatching(text, func):
    """ Return index of last character in string for which func(char) evaluates to True. """
    for i in range(len(text) - 1, -1, -1):
      if func(text[i]):
        return i

def md5_string(s):
    """
    Shortcut to create md5 hash
    :param s:
    :return:
    """
    m = hashlib.md5()
    m.update(s)
    return str(m.hexdigest())

def unique_element(ll):
    """ returns unique elements from a list preserving the original order """
    seen = {}
    result = []
    for item in ll:
        if item in seen:
            continue
        seen[item] = 1
        result.append(item)
    return result

def _count_leading_whitespace(text):
  """Returns the number of characters at the beginning of text that are whitespace."""
  idx = 0
  for idx, char in enumerate(text):
    if not char.isspace():
      return idx
  return idx + 1

def close(self):
        """Close child subprocess"""
        if self._subprocess is not None:
            os.killpg(self._subprocess.pid, signal.SIGTERM)
            self._subprocess = None

def _convert_to_float_if_possible(s):
    """
    A small helper function to convert a string to a numeric value
    if appropriate

    :param s: the string to be converted
    :type s: str
    """
    try:
        ret = float(s)
    except (ValueError, TypeError):
        ret = s
    return ret

def add_to_js(self, name, var):
        """Add an object to Javascript."""
        frame = self.page().mainFrame()
        frame.addToJavaScriptWindowObject(name, var)

def lin_interp(x, rangeX, rangeY):
    """
    Interpolate linearly variable x in rangeX onto rangeY.
    """
    s = (x - rangeX[0]) / mag(rangeX[1] - rangeX[0])
    y = rangeY[0] * (1 - s) + rangeY[1] * s
    return y

def process_kill(pid, sig=None):
    """Send signal to process.
    """
    sig = sig or signal.SIGTERM
    os.kill(pid, sig)

def parse_case_snake_to_camel(snake, upper_first=True):
	"""
	Convert a string from snake_case to CamelCase.

	:param str snake: The snake_case string to convert.
	:param bool upper_first: Whether or not to capitalize the first
		character of the string.
	:return: The CamelCase version of string.
	:rtype: str
	"""
	snake = snake.split('_')
	first_part = snake[0]
	if upper_first:
		first_part = first_part.title()
	return first_part + ''.join(word.title() for word in snake[1:])

def destroy(self):
		"""Finish up a session.
		"""
		if self.session_type == 'bash':
			# TODO: does this work/handle already being logged out/logged in deep OK?
			self.logout()
		elif self.session_type == 'vagrant':
			# TODO: does this work/handle already being logged out/logged in deep OK?
			self.logout()

def fail(message=None, exit_status=None):
    """Prints the specified message and exits the program with the specified
    exit status.

    """
    print('Error:', message, file=sys.stderr)
    sys.exit(exit_status or 1)

def text_remove_empty_lines(text):
    """
    Whitespace normalization:

      - Strip empty lines
      - Strip trailing whitespace
    """
    lines = [ line.rstrip()  for line in text.splitlines()  if line.strip() ]
    return "\n".join(lines)

def ylim(self, low, high, index=1):
        """Set yaxis limits.

        Parameters
        ----------
        low : number
        high : number
        index : int, optional

        Returns
        -------
        Chart

        """
        self.layout['yaxis' + str(index)]['range'] = [low, high]
        return self

def _repr_strip(mystring):
    """
    Returns the string without any initial or final quotes.
    """
    r = repr(mystring)
    if r.startswith("'") and r.endswith("'"):
        return r[1:-1]
    else:
        return r

def ylim(self, low, high, index=1):
        """Set yaxis limits.

        Parameters
        ----------
        low : number
        high : number
        index : int, optional

        Returns
        -------
        Chart

        """
        self.layout['yaxis' + str(index)]['range'] = [low, high]
        return self

def barray(iterlines):
    """
    Array of bytes
    """
    lst = [line.encode('utf-8') for line in iterlines]
    arr = numpy.array(lst)
    return arr

def set_gradclip_const(self, min_value, max_value):
        """
        Configure constant clipping settings.


        :param min_value: the minimum value to clip by
        :param max_value: the maxmimum value to clip by
        """
        callBigDlFunc(self.bigdl_type, "setConstantClip", self.value, min_value, max_value)

def show_xticklabels(self, row, column):
        """Show the x-axis tick labels for a subplot.

        :param row,column: specify the subplot.

        """
        subplot = self.get_subplot_at(row, column)
        subplot.show_xticklabels()

def connect(self):
        """Connects to the given host"""
        self.socket = socket.create_connection(self.address, self.timeout)

def kill_all(self, kill_signal, kill_shell=False):
        """Kill all running processes."""
        for key in self.processes.keys():
            self.kill_process(key, kill_signal, kill_shell)

def set_cursor_position(self, position):
        """Set cursor position"""
        position = self.get_position(position)
        cursor = self.textCursor()
        cursor.setPosition(position)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

def _finish(self):
        """
        Closes and waits for subprocess to exit.
        """
        if self._process.returncode is None:
            self._process.stdin.flush()
            self._process.stdin.close()
            self._process.wait()
            self.closed = True

def _set_axis_limits(self, which, lims, d, scale, reverse=False):
        """Private method for setting axis limits.

        Sets the axis limits on each axis for an individual plot.

        Args:
            which (str): The indicator of which part of the plots
                to adjust. This currently handles `x` and `y`.
            lims (len-2 list of floats): The limits for the axis.
            d (float): Amount to increment by between the limits.
            scale (str): Scale of the axis. Either `log` or `lin`.
            reverse (bool, optional): If True, reverse the axis tick marks. Default is False.

        """
        setattr(self.limits, which + 'lims', lims)
        setattr(self.limits, 'd' + which, d)
        setattr(self.limits, which + 'scale', scale)

        if reverse:
            setattr(self.limits, 'reverse_' + which + '_axis', True)
        return

def downsample_with_striding(array, factor):
    """Downsample x by factor using striding.

    @return: The downsampled array, of the same type as x.
    """
    return array[tuple(np.s_[::f] for f in factor)]

def relative_path(path):
    """
    Return the given path relative to this file.
    """
    return os.path.join(os.path.dirname(__file__), path)

def dropna(self, subset=None):
        """Remove missing values according to Baloo's convention.

        Parameters
        ----------
        subset : list of str, optional
            Which columns to check for missing values in.

        Returns
        -------
        DataFrame
            DataFrame with no null values in columns.

        """
        subset = check_and_obtain_subset_columns(subset, self)
        not_nas = [v.notna() for v in self[subset]._iter()]
        and_filter = reduce(lambda x, y: x & y, not_nas)

        return self[and_filter]

def GetPythonLibraryDirectoryPath():
  """Retrieves the Python library directory path."""
  path = sysconfig.get_python_lib(True)
  _, _, path = path.rpartition(sysconfig.PREFIX)

  if path.startswith(os.sep):
    path = path[1:]

  return path

def dump_parent(self, obj):
        """Dump the parent of a PID."""
        if not self._is_parent(obj):
            return self._dump_relative(obj.pid)
        return None

def title(msg):
    """Sets the title of the console window."""
    if sys.platform.startswith("win"):
        ctypes.windll.kernel32.SetConsoleTitleW(tounicode(msg))

def disown(cmd):
    """Call a system command in the background,
       disown it and hide it's output."""
    subprocess.Popen(cmd,
                     stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)

def sort_fn_list(fn_list):
    """Sort input filename list by datetime
    """
    dt_list = get_dt_list(fn_list)
    fn_list_sort = [fn for (dt,fn) in sorted(zip(dt_list,fn_list))]
    return fn_list_sort

def test_SVD(pca):
    """
    Function to test the validity of singular
    value decomposition by reconstructing original
    data.
    """
    _ = pca
    rec = N.dot(_.U,N.dot(_.sigma,_.V))
    assert N.allclose(_.arr,rec)

def sort_data(data, cols):
    """Sort `data` rows and order columns"""
    return data.sort_values(cols)[cols + ['value']].reset_index(drop=True)

def inside_softimage():
    """Returns a boolean indicating if the code is executed inside softimage."""
    try:
        import maya
        return False
    except ImportError:
        pass
    try:
        from win32com.client import Dispatch as disp
        disp('XSI.Application')
        return True
    except:
        return False

def set_float(self, option, value):
        """Set a float option.

            Args:
                option (str): name of option.
                value (float): value of the option.

            Raises:
                TypeError: Value must be a float.
        """
        if not isinstance(value, float):
            raise TypeError("Value must be a float")
        self.options[option] = value

def symbols():
    """Return a list of symbols."""
    symbols = []
    for line in symbols_stream():
        symbols.append(line.decode('utf-8').strip())
    return symbols

def tokenize_list(self, text):
        """
        Split a text into separate words.
        """
        return [self.get_record_token(record) for record in self.analyze(text)]

def fn_abs(self, value):
        """
        Return the absolute value of a number.

        :param value: The number.
        :return: The absolute value of the number.
        """

        if is_ndarray(value):
            return numpy.absolute(value)
        else:
            return abs(value)

def split_multiline(value):
    """Split a multiline string into a list, excluding blank lines."""
    return [element for element in (line.strip() for line in value.split('\n'))
            if element]

def main(argv=None):
  """Run a Tensorflow model on the Iris dataset."""
  args = parse_arguments(sys.argv if argv is None else argv)

  tf.logging.set_verbosity(tf.logging.INFO)
  learn_runner.run(
      experiment_fn=get_experiment_fn(args),
      output_dir=args.job_dir)

def column_stack_2d(data):
    """Perform column-stacking on a list of 2d data blocks."""
    return list(list(itt.chain.from_iterable(_)) for _ in zip(*data))

def is_integer(obj):
    """Is this an integer.

    :param object obj:
    :return:
    """
    if PYTHON3:
        return isinstance(obj, int)
    return isinstance(obj, (int, long))

def server(port):
    """Start the Django dev server."""
    args = ['python', 'manage.py', 'runserver']
    if port:
        args.append(port)
    run.main(args)

def _writable_dir(path):
    """Whether `path` is a directory, to which the user has write access."""
    return os.path.isdir(path) and os.access(path, os.W_OK)

def timeout_thread_handler(timeout, stop_event):
    """A background thread to kill the process if it takes too long.

    Args:
        timeout (float): The number of seconds to wait before killing
            the process.
        stop_event (Event): An optional event to cleanly stop the background
            thread if required during testing.
    """

    stop_happened = stop_event.wait(timeout)
    if stop_happened is False:
        print("Killing program due to %f second timeout" % timeout)

    os._exit(2)

def remove_stopped_threads (self):
        """Remove the stopped threads from the internal thread list."""
        self.threads = [t for t in self.threads if t.is_alive()]

def open_store_variable(self, name, var):
        """Turn CDMRemote variable into something like a numpy.ndarray."""
        data = indexing.LazilyOuterIndexedArray(CDMArrayWrapper(name, self))
        return Variable(var.dimensions, data, {a: getattr(var, a) for a in var.ncattrs()})

def normalize_time(timestamp):
    """Normalize time in arbitrary timezone to UTC naive object."""
    offset = timestamp.utcoffset()
    if offset is None:
        return timestamp
    return timestamp.replace(tzinfo=None) - offset

def on_error(e):  # pragma: no cover
    """Error handler

    RuntimeError or ValueError exceptions raised by commands will be handled
    by this function.
    """
    exname = {'RuntimeError': 'Runtime error', 'Value Error': 'Value error'}
    sys.stderr.write('{}: {}\n'.format(exname[e.__class__.__name__], str(e)))
    sys.stderr.write('See file slam_error.log for additional details.\n')
    sys.exit(1)

def prevmonday(num):
    """
    Return unix SECOND timestamp of "num" mondays ago
    """
    today = get_today()
    lastmonday = today - timedelta(days=today.weekday(), weeks=num)
    return lastmonday

def _listify(collection):
        """This is a workaround where Collections are no longer iterable
        when using JPype."""
        new_list = []
        for index in range(len(collection)):
            new_list.append(collection[index])
        return new_list

def _shutdown_proc(p, timeout):
  """Wait for a proc to shut down, then terminate or kill it after `timeout`."""
  freq = 10  # how often to check per second
  for _ in range(1 + timeout * freq):
    ret = p.poll()
    if ret is not None:
      logging.info("Shutdown gracefully.")
      return ret
    time.sleep(1 / freq)
  logging.warning("Killing the process.")
  p.kill()
  return p.wait()

def strip_accents(string):
    """
    Strip all the accents from the string
    """
    return u''.join(
        (character for character in unicodedata.normalize('NFD', string)
         if unicodedata.category(character) != 'Mn'))

def to_unix(cls, timestamp):
        """ Wrapper over time module to produce Unix epoch time as a float """
        if not isinstance(timestamp, datetime.datetime):
            raise TypeError('Time.milliseconds expects a datetime object')
        base = time.mktime(timestamp.timetuple())
        return base

def _repr_strip(mystring):
    """
    Returns the string without any initial or final quotes.
    """
    r = repr(mystring)
    if r.startswith("'") and r.endswith("'"):
        return r[1:-1]
    else:
        return r

def normalize_time(timestamp):
    """Normalize time in arbitrary timezone to UTC naive object."""
    offset = timestamp.utcoffset()
    if offset is None:
        return timestamp
    return timestamp.replace(tzinfo=None) - offset

def split(s):
  """Uses dynamic programming to infer the location of spaces in a string without spaces."""
  l = [_split(x) for x in _SPLIT_RE.split(s)]
  return [item for sublist in l for item in sublist]

def __run(self):
    """Hacked run function, which installs the trace."""
    sys.settrace(self.globaltrace)
    self.__run_backup()
    self.run = self.__run_backup

def to_snake_case(s):
    """Converts camel-case identifiers to snake-case."""
    return re.sub('([^_A-Z])([A-Z])', lambda m: m.group(1) + '_' + m.group(2).lower(), s)

def keyReleaseEvent(self, event):
        """
        Pyqt specific key release callback function.
        Translates and forwards events to :py:func:`keyboard_event`.
        """
        self.keyboard_event(event.key(), self.keys.ACTION_RELEASE, 0)

def _swap_rows(self, i, j):
        """Swap i and j rows

        As the side effect, determinant flips.

        """

        L = np.eye(3, dtype='intc')
        L[i, i] = 0
        L[j, j] = 0
        L[i, j] = 1
        L[j, i] = 1
        self._L.append(L.copy())
        self._A = np.dot(L, self._A)

def input(self, prompt, default=None, show_default=True):
        """Provide a command prompt."""
        return click.prompt(prompt, default=default, show_default=show_default)

def askopenfilename(**kwargs):
    """Return file name(s) from Tkinter's file open dialog."""
    try:
        from Tkinter import Tk
        import tkFileDialog as filedialog
    except ImportError:
        from tkinter import Tk, filedialog
    root = Tk()
    root.withdraw()
    root.update()
    filenames = filedialog.askopenfilename(**kwargs)
    root.destroy()
    return filenames

def get_dt_list(fn_list):
    """Get list of datetime objects, extracted from a filename
    """
    dt_list = np.array([fn_getdatetime(fn) for fn in fn_list])
    return dt_list

def toggle_word_wrap(self):
        """
        Toggles document word wrap.

        :return: Method success.
        :rtype: bool
        """

        self.setWordWrapMode(not self.wordWrapMode() and QTextOption.WordWrap or QTextOption.NoWrap)
        return True

def unique_inverse(item_list):
    """
    Like np.unique(item_list, return_inverse=True)
    """
    import utool as ut
    unique_items = ut.unique(item_list)
    inverse = list_alignment(unique_items, item_list)
    return unique_items, inverse

def hide(self):
        """Hide the window."""
        self.tk.withdraw()
        self._visible = False
        if self._modal:
            self.tk.grab_release()

def forward(self, step):
        """Move the turtle forward.

        :param step: Integer. Distance to move forward.
        """
        x = self.pos_x + math.cos(math.radians(self.rotation)) * step
        y = self.pos_y + math.sin(math.radians(self.rotation)) * step
        prev_brush_state = self.brush_on
        self.brush_on = True
        self.move(x, y)
        self.brush_on = prev_brush_state

def empty(self):
        """remove all children from the widget"""
        for k in list(self.children.keys()):
            self.remove_child(self.children[k])

def set_stop_handler(self):
        """
        Initializes functions that are invoked when the user or OS wants to kill this process.
        :return:
        """
        signal.signal(signal.SIGTERM, self.graceful_stop)
        signal.signal(signal.SIGABRT, self.graceful_stop)
        signal.signal(signal.SIGINT, self.graceful_stop)

def __init__(self, master=None, compound=tk.RIGHT, autohidescrollbar=True, **kwargs):
        """
        Create a Listbox with a vertical scrollbar.

        :param master: master widget
        :type master: widget
        :param compound: side for the Scrollbar to be on (:obj:`tk.LEFT` or :obj:`tk.RIGHT`)
        :type compound: str
        :param autohidescrollbar: whether to use an :class:`~ttkwidgets.AutoHideScrollbar` or a :class:`ttk.Scrollbar`
        :type autohidescrollbar: bool
        :param kwargs: keyword arguments passed on to the :class:`tk.Listbox` initializer
        """
        ttk.Frame.__init__(self, master)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.listbox = tk.Listbox(self, **kwargs)
        if autohidescrollbar:
            self.scrollbar = AutoHideScrollbar(self, orient=tk.VERTICAL, command=self.listbox.yview)
        else:
            self.scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.listbox.yview)
        self.config_listbox(yscrollcommand=self.scrollbar.set)
        if compound is not tk.LEFT and compound is not tk.RIGHT:
            raise ValueError("Invalid compound value passed: {0}".format(compound))
        self.__compound = compound
        self._grid_widgets()

def select_default(self):
        """
        Resets the combo box to the original "selected" value from the
        constructor (or the first value if no selected value was specified).
        """
        if self._default is None:
            if not self._set_option_by_index(0):
                utils.error_format(self.description + "\n" +
                "Unable to select default option as the Combo is empty")

        else:
            if not self._set_option(self._default):
                utils.error_format( self.description + "\n" +
                "Unable to select default option as it doesnt exist in the Combo")

def _check_elements_equal(lst):
    """
    Returns true if all of the elements in the list are equal.
    """
    assert isinstance(lst, list), "Input value must be a list."
    return not lst or lst.count(lst[0]) == len(lst)

def get_host_power_status(self):
        """Request the power state of the server.

        :returns: Power State of the server, 'ON' or 'OFF'
        :raises: IloError, on an error from iLO.
        """
        sushy_system = self._get_sushy_system(PROLIANT_SYSTEM_ID)
        return GET_POWER_STATE_MAP.get(sushy_system.power_state)

def timed_call(func, *args, log_level='DEBUG', **kwargs):
    """Logs a function's run time

    :param func: The function to run
    :param args: The args to pass to the function
    :param kwargs: The keyword args to pass to the function
    :param log_level: The log level at which to print the run time
    :return: The function's return value
    """
    start = time()
    r = func(*args, **kwargs)
    t = time() - start
    log(log_level, "Call to '{}' took {:0.6f}s".format(func.__name__, t))
    return r

def _is_image_sequenced(image):
    """Determine if the image is a sequenced image."""
    try:
        image.seek(1)
        image.seek(0)
        result = True
    except EOFError:
        result = False

    return result

def sent2features(sentence, template):
    """ extract features in a sentence

    :type sentence: list of token, each token is a list of tag
    """
    return [word2features(sentence, i, template) for i in range(len(sentence))]

def _namematcher(regex):
    """Checks if a target name matches with an input regular expression."""

    matcher = re_compile(regex)

    def match(target):
        target_name = getattr(target, '__name__', '')
        result = matcher.match(target_name)
        return result

    return match

def _unjsonify(x, isattributes=False):
    """Convert JSON string to an ordered defaultdict."""
    if isattributes:
        obj = json.loads(x)
        return dict_class(obj)
    return json.loads(x)

def eglInitialize(display):
    """ Initialize EGL and return EGL version tuple.
    """
    majorVersion = (_c_int*1)()
    minorVersion = (_c_int*1)()
    res = _lib.eglInitialize(display, majorVersion, minorVersion)
    if res == EGL_FALSE:
        raise RuntimeError('Could not initialize')
    return majorVersion[0], minorVersion[0]

def get_last_or_frame_exception():
    """Intended to be used going into post mortem routines.  If
    sys.last_traceback is set, we will return that and assume that
    this is what post-mortem will want. If sys.last_traceback has not
    been set, then perhaps we *about* to raise an error and are
    fielding an exception. So assume that sys.exc_info()[2]
    is where we want to look."""

    try:
        if inspect.istraceback(sys.last_traceback):
            # We do have a traceback so prefer that.
            return sys.last_type, sys.last_value, sys.last_traceback
    except AttributeError:
        pass
    return sys.exc_info()

def mouse_move_event(self, event):
        """
        Forward mouse cursor position events to the example
        """
        self.example.mouse_position_event(event.x(), event.y())

def list2dict(lst):
    """Takes a list of (key,value) pairs and turns it into a dict."""

    dic = {}
    for k,v in lst: dic[k] = v
    return dic

def path_to_list(pathstr):
    """Conver a path string to a list of path elements."""
    return [elem for elem in pathstr.split(os.path.pathsep) if elem]

def string_to_identity(identity_str):
    """Parse string into Identity dictionary."""
    m = _identity_regexp.match(identity_str)
    result = m.groupdict()
    log.debug('parsed identity: %s', result)
    return {k: v for k, v in result.items() if v}

def normalize_enum_constant(s):
    """Return enum constant `s` converted to a canonical snake-case."""
    if s.islower(): return s
    if s.isupper(): return s.lower()
    return "".join(ch if ch.islower() else "_" + ch.lower() for ch in s).strip("_")

def string_to_float_list(string_var):
        """Pull comma separated string values out of a text file and converts them to float list"""
        try:
            return [float(s) for s in string_var.strip('[').strip(']').split(', ')]
        except:
            return [float(s) for s in string_var.strip('[').strip(']').split(',')]

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def to_bytes(s, encoding="utf-8"):
    """Convert a string to bytes."""
    if isinstance(s, six.binary_type):
        return s
    if six.PY3:
        return bytes(s, encoding)
    return s.encode(encoding)

def C_dict2array(C):
    """Convert an OrderedDict containing C values to a 1D array."""
    return np.hstack([np.asarray(C[k]).ravel() for k in C_keys])

def intToBin(i):
    """ Integer to two bytes """
    # divide in two parts (bytes)
    i1 = i % 256
    i2 = int(i / 256)
    # make string (little endian)
    return i.to_bytes(2, byteorder='little')

def size(self):
        """
        Recursively find size of a tree. Slow.
        """

        if self is NULL:
            return 0
        return 1 + self.left.size() + self.right.size()

def should_skip_logging(func):
    """
    Should we skip logging for this handler?

    """
    disabled = strtobool(request.headers.get("x-request-nolog", "false"))
    return disabled or getattr(func, SKIP_LOGGING, False)

def cleanup_nodes(doc):
    """
    Remove text nodes containing only whitespace
    """
    for node in doc.documentElement.childNodes:
        if node.nodeType == Node.TEXT_NODE and node.nodeValue.isspace():
            doc.documentElement.removeChild(node)
    return doc

def stdout_to_results(s):
    """Turns the multi-line output of a benchmark process into
    a sequence of BenchmarkResult instances."""
    results = s.strip().split('\n')
    return [BenchmarkResult(*r.split()) for r in results]

def _remove_blank(l):
        """ Removes trailing zeros in the list of integers and returns a new list of integers"""
        ret = []
        for i, _ in enumerate(l):
            if l[i] == 0:
                break
            ret.append(l[i])
        return ret

def get_neg_infinity(dtype):
    """Return an appropriate positive infinity for this dtype.

    Parameters
    ----------
    dtype : np.dtype

    Returns
    -------
    fill_value : positive infinity value corresponding to this dtype.
    """
    if issubclass(dtype.type, (np.floating, np.integer)):
        return -np.inf

    if issubclass(dtype.type, np.complexfloating):
        return -np.inf - 1j * np.inf

    return NINF

def retry_call(func, cleanup=lambda: None, retries=0, trap=()):
	"""
	Given a callable func, trap the indicated exceptions
	for up to 'retries' times, invoking cleanup on the
	exception. On the final attempt, allow any exceptions
	to propagate.
	"""
	attempts = count() if retries == float('inf') else range(retries)
	for attempt in attempts:
		try:
			return func()
		except trap:
			cleanup()

	return func()

def union(self, other):
        """Return a new set which is the union of I{self} and I{other}.

        @param other: the other set
        @type other: Set object
        @rtype: the same type as I{self}
        """

        obj = self._clone()
        obj.union_update(other)
        return obj

def set_scrollbars_cb(self, w, tf):
        """This callback is invoked when the user checks the 'Use Scrollbars'
        box in the preferences pane."""
        scrollbars = 'on' if tf else 'off'
        self.t_.set(scrollbars=scrollbars)

def install_from_zip(url):
    """Download and unzip from url."""
    fname = 'tmp.zip'
    downlad_file(url, fname)
    unzip_file(fname)
    print("Removing {}".format(fname))
    os.unlink(fname)

def tuple_check(*args, func=None):
    """Check if arguments are tuple type."""
    func = func or inspect.stack()[2][3]
    for var in args:
        if not isinstance(var, (tuple, collections.abc.Sequence)):
            name = type(var).__name__
            raise TupleError(
                f'Function {func} expected tuple, {name} got instead.')

def min_values(args):
    """ Return possible range for min function. """
    return Interval(min(x.low for x in args), min(x.high for x in args))

def md_to_text(content):
    """ Converts markdown content to text """
    text = None
    html = markdown.markdown(content)
    if html:
        text = html_to_text(content)
    return text

def unit_ball_L2(shape):
  """A tensorflow variable tranfomed to be constrained in a L2 unit ball.

  EXPERIMENTAL: Do not use for adverserial examples if you need to be confident
  they are strong attacks. We are not yet confident in this code.
  """
  x = tf.Variable(tf.zeros(shape))
  return constrain_L2(x)

def list_of_dict(self):
        """
        This will convert the data from a list of list to a list of dictionary
        :return: list of dict
        """
        ret = []
        for row in self:
            ret.append(dict([(self._col_names[i], row[i]) for i in
                             range(len(self._col_names))]))
        return ReprListDict(ret, col_names=self._col_names,
                            col_types=self._col_types,
                            width_limit=self._width_limit,
                            digits=self._digits,
                            convert_unicode=self._convert_unicode)

def copy_and_update(dictionary, update):
    """Returns an updated copy of the dictionary without modifying the original"""
    newdict = dictionary.copy()
    newdict.update(update)
    return newdict

def uri_to_iri_parts(path, query, fragment):
    r"""
    Converts a URI parts to corresponding IRI parts in a given charset.

    Examples for URI versus IRI:

    :param path: The path of URI to convert.
    :param query: The query string of URI to convert.
    :param fragment: The fragment of URI to convert.
    """
    path = url_unquote(path, '%/;?')
    query = url_unquote(query, '%;/?:@&=+,$#')
    fragment = url_unquote(fragment, '%;/?:@&=+,$#')
    return path, query, fragment

def _help():
    """ Display both SQLAlchemy and Python help statements """

    statement = '%s%s' % (shelp, phelp % ', '.join(cntx_.keys()))
    print statement.strip()

def twitter_timeline(screen_name, since_id=None):
    """ Return relevant twitter timeline """
    consumer_key = twitter_credential('consumer_key')
    consumer_secret = twitter_credential('consumer_secret')
    access_token = twitter_credential('access_token')
    access_token_secret = twitter_credential('access_secret')
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    return get_all_tweets(screen_name, api, since_id)

def get_input(input_func, input_str):
    """
    Get input from the user given an input function and an input string
    """
    val = input_func("Please enter your {0}: ".format(input_str))
    while not val or not len(val.strip()):
        val = input_func("You didn't enter a valid {0}, please try again: ".format(input_str))
    return val

def check_str(obj):
        """ Returns a string for various input types """
        if isinstance(obj, str):
            return obj
        if isinstance(obj, float):
            return str(int(obj))
        else:
            return str(obj)

def str2int(string_with_int):
    """ Collect digits from a string """
    return int("".join([char for char in string_with_int if char in string.digits]) or 0)

def parse_parameter(value):
    """
    @return: The best approximation of a type of the given value.
    """
    if any((isinstance(value, float), isinstance(value, int), isinstance(value, bool))):
        return value

    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            if value in string_aliases.true_boolean_aliases:
                return True
            elif value in string_aliases.false_boolean_aliases:
                return False
            else:
                return str(value)

def notin(arg, values):
    """
    Like isin, but checks whether this expression's value(s) are not
    contained in the passed values. See isin docs for full usage.
    """
    op = ops.NotContains(arg, values)
    return op.to_expr()

def is_timestamp(instance):
    """Validates data is a timestamp"""
    if not isinstance(instance, (int, str)):
        return True
    return datetime.fromtimestamp(int(instance))

def HttpResponse401(request, template=KEY_AUTH_401_TEMPLATE,
content=KEY_AUTH_401_CONTENT, content_type=KEY_AUTH_401_CONTENT_TYPE):
    """
    HTTP response for not-authorized access (status code 403)
    """
    return AccessFailedResponse(request, template, content, content_type, status=401)

def subscribe(self, handler):
        """Adds a new event handler."""
        assert callable(handler), "Invalid handler %s" % handler
        self.handlers.append(handler)

def get_page_text(self, page):
        """
        Downloads and returns the full text of a particular page
        in the document.
        """
        url = self.get_page_text_url(page)
        return self._get_url(url)

def get_form_bound_field(form, field_name):
    """
    Intends to get the bound field from the form regarding the field name

    :param form: Django Form: django form instance
    :param field_name: str: name of the field in form instance
    :return: Django Form bound field
    """
    field = form.fields[field_name]
    field = field.get_bound_field(form, field_name)
    return field

def set_scrollregion(self, event=None):
        """ Set the scroll region on the canvas"""
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

def union_overlapping(intervals):
    """Union any overlapping intervals in the given set."""
    disjoint_intervals = []

    for interval in intervals:
        if disjoint_intervals and disjoint_intervals[-1].overlaps(interval):
            disjoint_intervals[-1] = disjoint_intervals[-1].union(interval)
        else:
            disjoint_intervals.append(interval)

    return disjoint_intervals

def print_latex(o):
    """A function to generate the latex representation of sympy
    expressions."""
    if can_print_latex(o):
        s = latex(o, mode='plain')
        s = s.replace('\\dag','\\dagger')
        s = s.strip('$')
        return '$$%s$$' % s
    # Fallback to the string printer
    return None

def flatten_union(table):
    """Extract all union queries from `table`.

    Parameters
    ----------
    table : TableExpr

    Returns
    -------
    Iterable[Union[TableExpr, bool]]
    """
    op = table.op()
    if isinstance(op, ops.Union):
        return toolz.concatv(
            flatten_union(op.left), [op.distinct], flatten_union(op.right)
        )
    return [table]

def comment (self, s, **args):
        """Write GML comment."""
        self.writeln(s=u'comment "%s"' % s, **args)

def page_title(step, title):
    """
    Check that the page title matches the given one.
    """

    with AssertContextManager(step):
        assert_equals(world.browser.title, title)

def setdefault(obj, field, default):
    """Set an object's field to default if it doesn't have a value"""
    setattr(obj, field, getattr(obj, field, default))

def test():  # pragma: no cover
    """Execute the unit tests on an installed copy of unyt.

    Note that this function requires pytest to run. If pytest is not
    installed this function will raise ImportError.
    """
    import pytest
    import os

    pytest.main([os.path.dirname(os.path.abspath(__file__))])

def script_repr(val,imports,prefix,settings):
    """
    Variant of repr() designed for generating a runnable script.

    Instances of types that require special handling can use the
    script_repr_reg dictionary. Using the type as a key, add a
    function that returns a suitable representation of instances of
    that type, and adds the required import statement.

    The repr of a parameter can be suppressed by returning None from
    the appropriate hook in script_repr_reg.
    """
    return pprint(val,imports,prefix,settings,unknown_value=None,
                  qualify=True,separator="\n")

def write_str2file(pathname, astr):
    """writes a string to file"""
    fname = pathname
    fhandle = open(fname, 'wb')
    fhandle.write(astr)
    fhandle.close()

def assert_redirect(self, response, expected_url=None):
        """
        assertRedirects from Django TestCase follows the redirects chains,
        this assertion does not - which is more like real unit testing
        """
        self.assertIn(
            response.status_code,
            self.redirect_codes,
            self._get_redirect_assertion_message(response),
        )
        if expected_url:
            location_header = response._headers.get('location', None)
            self.assertEqual(
                location_header,
                ('Location', str(expected_url)),
                'Response should redirect to {0}, but it redirects to {1} instead'.format(
                    expected_url,
                    location_header[1],
                )
            )

def _write_separator(self):
        """
        Inserts a horizontal (commented) line tot the generated code.
        """
        tmp = self._page_width - ((4 * self.__indent_level) + 2)
        self._write_line('# ' + ('-' * tmp))

def __unixify(self, s):
        """ stupid windows. converts the backslash to forwardslash for consistency """
        return os.path.normpath(s).replace(os.sep, "/")

def extract_zip(zip_path, target_folder):
    """
    Extract the content of the zip-file at `zip_path` into `target_folder`.
    """
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target_folder)

def convert_array(array):
    """
    Converts an ARRAY string stored in the database back into a Numpy array.

    Parameters
    ----------
    array: ARRAY
        The array object to be converted back into a Numpy array.

    Returns
    -------
    array
            The converted Numpy array.

    """
    out = io.BytesIO(array)
    out.seek(0)
    return np.load(out)

def save_excel(self, fd):
        """ Saves the case as an Excel spreadsheet.
        """
        from pylon.io.excel import ExcelWriter
        ExcelWriter(self).write(fd)

def _keys_to_camel_case(self, obj):
        """
        Make a copy of a dictionary with all keys converted to camel case. This is just calls to_camel_case on each of the keys in the dictionary and returns a new dictionary.

        :param obj: Dictionary to convert keys to camel case.
        :return: Dictionary with the input values and all keys in camel case
        """
        return dict((to_camel_case(key), value) for (key, value) in obj.items())

def top(n, width=WIDTH, style=STYLE):
    """Prints the top row of a table"""
    return hrule(n, width, linestyle=STYLES[style].top)

def set_global(node: Node, key: str, value: Any):
    """Adds passed value to node's globals"""
    node.node_globals[key] = value

def handle_exception(error):
        """Simple method for handling exceptions raised by `PyBankID`.

        :param flask_pybankid.FlaskPyBankIDError error: The exception to handle.
        :return: The exception represented as a dictionary.
        :rtype: dict

        """
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

def update_context(self, ctx):
        """ updates the query context with this clauses values """
        assert isinstance(ctx, dict)
        ctx[str(self.context_id)] = self.value

def get_encoding(binary):
    """Return the encoding type."""

    try:
        from chardet import detect
    except ImportError:
        LOGGER.error("Please install the 'chardet' module")
        sys.exit(1)

    encoding = detect(binary).get('encoding')

    return 'iso-8859-1' if encoding == 'CP949' else encoding

def get_url_args(url):
    """ Returns a dictionary from a URL params """
    url_data = urllib.parse.urlparse(url)
    arg_dict = urllib.parse.parse_qs(url_data.query)
    return arg_dict

def clear_list_value(self, value):
        """
        Clean the argument value to eliminate None or Falsy values if needed.
        """
        # Don't go any further: this value is empty.
        if not value:
            return self.empty_value
        # Clean empty items if wanted
        if self.clean_empty:
            value = [v for v in value if v]
        return value or self.empty_value

def get_url_file_name(url):
    """Get the file name from an url
    
    Parameters
    ----------
    url : str

    Returns
    -------
    str
        The file name 
    """

    assert isinstance(url, (str, _oldstr))
    return urlparse.urlparse(url).path.split('/')[-1]

def table_exists(cursor, tablename, schema='public'):
    query = """
    SELECT EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = %s
        AND table_name = %s
    )"""
    cursor.execute(query, (schema, tablename))
    res = cursor.fetchone()[0]
    return res

def _fill_array_from_list(the_list, the_array):
        """Fill an `array` from a `list`"""
        for i, val in enumerate(the_list):
            the_array[i] = val
        return the_array

def isString(s):
    """Convenience method that works with all 2.x versions of Python
    to determine whether or not something is stringlike."""
    try:
        return isinstance(s, unicode) or isinstance(s, basestring)
    except NameError:
        return isinstance(s, str)

def list_formatter(handler, item, value):
    """Format list."""
    return u', '.join(str(v) for v in value)

def average_gradient(data, *kwargs):
    """ Compute average gradient norm of an image
    """
    return np.average(np.array(np.gradient(data))**2)

def shape_list(l,shape,dtype):
    """ Shape a list of lists into the appropriate shape and data type """
    return np.array(l, dtype=dtype).reshape(shape)

def dfs_recursive(graph, node, seen):
    """DFS, detect connected component, recursive implementation

    :param graph: directed graph in listlist or listdict format
    :param int node: to start graph exploration
    :param boolean-table seen: will be set true for the connected component
          containing node.
    :complexity: `O(|V|+|E|)`
    """
    seen[node] = True
    for neighbor in graph[node]:
        if not seen[neighbor]:
            dfs_recursive(graph, neighbor, seen)

def Proxy(f):
  """A helper to create a proxy method in a class."""

  def Wrapped(self, *args):
    return getattr(self, f)(*args)

  return Wrapped

def bbox(self):
        """
        The minimal `~photutils.aperture.BoundingBox` for the cutout
        region with respect to the original (large) image.
        """

        return BoundingBox(self.slices[1].start, self.slices[1].stop,
                           self.slices[0].start, self.slices[0].stop)

def prompt(*args, **kwargs):
    """Prompt the user for input and handle any abort exceptions."""
    try:
        return click.prompt(*args, **kwargs)
    except click.Abort:
        return False

def find_one(cls, *args, **kw):
		"""Get a single document from the collection this class is bound to.
		
		Additional arguments are processed according to `_prepare_find` prior to passing to PyMongo, where positional
		parameters are interpreted as query fragments, parametric keyword arguments combined, and other keyword
		arguments passed along with minor transformation.
		
		Automatically calls `to_mongo` with the retrieved data.
		
		https://api.mongodb.com/python/current/api/pymongo/collection.html#pymongo.collection.Collection.find_one
		"""
		
		if len(args) == 1 and not isinstance(args[0], Filter):
			args = (getattr(cls, cls.__pk__) == args[0], )
		
		Doc, collection, query, options = cls._prepare_find(*args, **kw)
		result = Doc.from_mongo(collection.find_one(query, **options))
		
		return result

def generate_user_token(self, user, salt=None):
        """Generates a unique token associated to the user
        """
        return self.token_serializer.dumps(str(user.id), salt=salt)

def exp_fit_fun(x, a, tau, c):
    """Function used to fit the exponential decay."""
    # pylint: disable=invalid-name
    return a * np.exp(-x / tau) + c

def random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4())  # Convert UUID format to a Python string.
    random = random.upper()  # Make all characters uppercase.
    random = random.replace("-", "")  # Remove the UUID '-'.
    return random[0:string_length]

def end_index(self):
        """
        Returns the 1-based index of the last object on this page,
        relative to total objects found (hits).
        """
        return ((self.number - 1) * self.paginator.per_page +
            len(self.object_list))

def bool_str(string):
    """Returns a boolean from a string imput of 'true' or 'false'"""
    if string not in BOOL_STRS:
        raise ValueError('Invalid boolean string: "{}"'.format(string))
    return True if string == 'true' else False

def quote(self, s):
        """Return a shell-escaped version of the string s."""

        if six.PY2:
            from pipes import quote
        else:
            from shlex import quote

        return quote(s)

def is_valid(data):
        """
        Checks if the input data is a Swagger document

        :param dict data: Data to be validated
        :return: True, if data is a Swagger
        """
        return bool(data) and \
            isinstance(data, dict) and \
            bool(data.get("swagger")) and \
            isinstance(data.get('paths'), dict)

def ynticks(self, nticks, index=1):
        """Set the number of ticks."""
        self.layout['yaxis' + str(index)]['nticks'] = nticks
        return self

def __len__(self):
        """ This will equal 124 for the V1 database. """
        length = 0
        for typ, siz, _ in self.format:
            length += siz
        return length

def _set_lastpage(self):
        """Calculate value of class attribute ``last_page``."""
        self.last_page = (len(self._page_data) - 1) // self.screen.page_size

def venv():
    """Install venv + deps."""
    try:
        import virtualenv  # NOQA
    except ImportError:
        sh("%s -m pip install virtualenv" % PYTHON)
    if not os.path.isdir("venv"):
        sh("%s -m virtualenv venv" % PYTHON)
    sh("venv\\Scripts\\pip install -r %s" % (REQUIREMENTS_TXT))

def ServerLoggingStartupInit():
  """Initialize the server logging configuration."""
  global LOGGER
  if local_log:
    logging.debug("Using local LogInit from %s", local_log)
    local_log.LogInit()
    logging.debug("Using local AppLogInit from %s", local_log)
    LOGGER = local_log.AppLogInit()
  else:
    LogInit()
    LOGGER = AppLogInit()

def init():
    """
    Execute init tasks for all components (virtualenv, pip).
    """
    print(yellow("# Setting up environment...\n", True))
    virtualenv.init()
    virtualenv.update_requirements()
    print(green("\n# DONE.", True))
    print(green("Type ") + green("activate", True) + green(" to enable your virtual environment."))

def _py2_and_3_joiner(sep, joinable):
    """
    Allow '\n'.join(...) statements to work in Py2 and Py3.
    :param sep:
    :param joinable:
    :return:
    """
    if ISPY3:
        sep = bytes(sep, DEFAULT_ENCODING)
    joined = sep.join(joinable)
    return joined.decode(DEFAULT_ENCODING) if ISPY3 else joined

def csort(objs, key):
    """Order-preserving sorting function."""
    idxs = dict((obj, i) for (i, obj) in enumerate(objs))
    return sorted(objs, key=lambda obj: (key(obj), idxs[obj]))

def put_text(self, key, text):
        """Put the text into the storage associated with the key."""
        with open(key, "w") as fh:
            fh.write(text)

def destroy_webdriver(driver):
    """
    Destroy a driver
    """

    # This is some very flaky code in selenium. Hence the retries
    # and catch-all exceptions
    try:
        retry_call(driver.close, tries=2)
    except Exception:
        pass
    try:
        driver.quit()
    except Exception:
        pass

def prepend_line(filepath, line):
    """Rewrite a file adding a line to its beginning.
    """
    with open(filepath) as f:
        lines = f.readlines()

    lines.insert(0, line)

    with open(filepath, 'w') as f:
        f.writelines(lines)

def kill(self):
        """Kill the browser.

        This is useful when the browser is stuck.
        """
        if self.process:
            self.process.kill()
            self.process.wait()

def build_code(self, lang, body):
        """Wrap text with markdown specific flavour."""
        self.out.append("```" + lang)
        self.build_markdown(lang, body)
        self.out.append("```")

def check_by_selector(self, selector):
    """Check the checkbox matching the CSS selector."""
    elem = find_element_by_jquery(world.browser, selector)
    if not elem.is_selected():
        elem.click()

def IPYTHON_MAIN():
    """Decide if the Ipython command line is running code."""
    import pkg_resources

    runner_frame = inspect.getouterframes(inspect.currentframe())[-2]
    return (
        getattr(runner_frame, "function", None)
        == pkg_resources.load_entry_point("ipython", "console_scripts", "ipython").__name__
    )

def _checkSize(self):
        """Automatically resizes widget to display at most max_height_items items"""
        if self._item_height is not None:
            sz = min(self._max_height_items, self.count()) * self._item_height + 5
            sz = max(sz, 20)
            self.setMinimumSize(0, sz)
            self.setMaximumSize(1000000, sz)
            self.resize(self.width(), sz)

def fast_distinct(self):
        """
        Because standard distinct used on the all fields are very slow and works only with PostgreSQL database
        this method provides alternative to the standard distinct method.
        :return: qs with unique objects
        """
        return self.model.objects.filter(pk__in=self.values_list('pk', flat=True))

def __grid_widgets(self):
        """Places all the child widgets in the appropriate positions."""
        scrollbar_column = 0 if self.__compound is tk.LEFT else 2
        self._canvas.grid(row=0, column=1, sticky="nswe")
        self._scrollbar.grid(row=0, column=scrollbar_column, sticky="ns")

def test():
    """Run the unit tests."""
    import unittest
    tests = unittest.TestLoader().discover('tests')
    unittest.TextTestRunner(verbosity=2).run(tests)

def retry_on_signal(function):
    """Retries function until it doesn't raise an EINTR error"""
    while True:
        try:
            return function()
        except EnvironmentError, e:
            if e.errno != errno.EINTR:
                raise

def all_equal(arg1,arg2):
    """
    Return a single boolean for arg1==arg2, even for numpy arrays
    using element-wise comparison.

    Uses all(arg1==arg2) for sequences, and arg1==arg2 otherwise.

    If both objects have an '_infinitely_iterable' attribute, they are
    not be zipped together and are compared directly instead.
    """
    if all(hasattr(el, '_infinitely_iterable') for el in [arg1,arg2]):
        return arg1==arg2
    try:
        return all(a1 == a2 for a1, a2 in zip(arg1, arg2))
    except TypeError:
        return arg1==arg2

def __init__(self, encoding='utf-8'):
    """Initializes an stdin input reader.

    Args:
      encoding (Optional[str]): input encoding.
    """
    super(StdinInputReader, self).__init__(sys.stdin, encoding=encoding)

def check_alert(self, text):
    """
    Assert an alert is showing with the given text.
    """

    try:
        alert = Alert(world.browser)
        if alert.text != text:
            raise AssertionError(
                "Alert text expected to be {!r}, got {!r}.".format(
                    text, alert.text))
    except WebDriverException:
        # PhantomJS is kinda poor
        pass

def write_fits(self, fitsfile):
        """Write the ROI model to a FITS file."""

        tab = self.create_table()
        hdu_data = fits.table_to_hdu(tab)
        hdus = [fits.PrimaryHDU(), hdu_data]
        fits_utils.write_hdus(hdus, fitsfile)

def each_img(img_dir):
    """
    Reads and iterates through each image file in the given directory
    """
    for fname in utils.each_img(img_dir):
        fname = os.path.join(img_dir, fname)
        yield cv.imread(fname), fname

def save_dict_to_file(filename, dictionary):
  """Saves dictionary as CSV file."""
  with open(filename, 'w') as f:
    writer = csv.writer(f)
    for k, v in iteritems(dictionary):
      writer.writerow([str(k), str(v)])

def path_distance(points):
    """
    Compute the path distance from given set of points
    """
    vecs = np.diff(points, axis=0)[:, :3]
    d2 = [np.dot(p, p) for p in vecs]
    return np.sum(np.sqrt(d2))

def series_table_row_offset(self, series):
        """
        Return the number of rows preceding the data table for *series* in
        the Excel worksheet.
        """
        title_and_spacer_rows = series.index * 2
        data_point_rows = series.data_point_offset
        return title_and_spacer_rows + data_point_rows

def extract_words(lines):
    """
    Extract from the given iterable of lines the list of words.

    :param lines: an iterable of lines;
    :return: a generator of words of lines.
    """
    for line in lines:
        for word in re.findall(r"\w+", line):
            yield word

def as_list(self):
        """Return all child objects in nested lists of strings."""
        return [self.name, self.value, [x.as_list for x in self.children]]

def __next__(self, reward, ask_id, lbl):
        """For Python3 compatibility of generator."""
        return self.next(reward, ask_id, lbl)

def element_to_string(element, include_declaration=True, encoding=DEFAULT_ENCODING, method='xml'):
    """ :return: the string value of the element or element tree """

    if isinstance(element, ElementTree):
        element = element.getroot()
    elif not isinstance(element, ElementType):
        element = get_element(element)

    if element is None:
        return u''

    element_as_string = tostring(element, encoding, method).decode(encoding=encoding)
    if include_declaration:
        return element_as_string
    else:
        return strip_xml_declaration(element_as_string)

def group_by(iterable, key_func):
    """Wrap itertools.groupby to make life easier."""
    groups = (
        list(sub) for key, sub in groupby(iterable, key_func)
    )
    return zip(groups, groups)

def _get_minidom_tag_value(station, tag_name):
    """get a value from a tag (if it exists)"""
    tag = station.getElementsByTagName(tag_name)[0].firstChild
    if tag:
        return tag.nodeValue

    return None

def jaccard(c_1, c_2):
    """
    Calculates the Jaccard similarity between two sets of nodes. Called by mroc.

    Inputs:  - c_1: Community (set of nodes) 1.
             - c_2: Community (set of nodes) 2.

    Outputs: - jaccard_similarity: The Jaccard similarity of these two communities.
    """
    nom = np.intersect1d(c_1, c_2).size
    denom = np.union1d(c_1, c_2).size
    return nom/denom

def required_attributes(element, *attributes):
    """Check element for required attributes. Raise ``NotValidXmlException`` on error.

    :param element: ElementTree element
    :param attributes: list of attributes names to check
    :raises NotValidXmlException: if some argument is missing
    """
    if not reduce(lambda still_valid, param: still_valid and param in element.attrib, attributes, True):
        raise NotValidXmlException(msg_err_missing_attributes(element.tag, *attributes))

def xml_str_to_dict(s):
    """ Transforms an XML string it to python-zimbra dict format

    For format, see:
      https://github.com/Zimbra-Community/python-zimbra/blob/master/README.md

    :param: a string, containing XML
    :returns: a dict, with python-zimbra format
    """
    xml = minidom.parseString(s)
    return pythonzimbra.tools.xmlserializer.dom_to_dict(xml.firstChild)

def send(self, topic, *args, **kwargs):
        """
        Appends the prefix to the topic before sendingf
        """
        prefix_topic = self.heroku_kafka.prefix_topic(topic)
        return super(HerokuKafkaProducer, self).send(prefix_topic, *args, **kwargs)

def root_parent(self, category=None):
        """ Returns the topmost parent of the current category. """
        return next(filter(lambda c: c.is_root, self.hierarchy()))

def best(self):
        """
        Returns the element with the highest probability.
        """
        b = (-1e999999, None)
        for k, c in iteritems(self.counts):
            b = max(b, (c, k))
        return b[1]

def _Open(self, hostname, port):
    """Opens the RPC communication channel for clients.

    Args:
      hostname (str): hostname or IP address to connect to for requests.
      port (int): port to connect to for requests.

    Returns:
      bool: True if the communication channel was successfully opened.
    """
    try:
      self._xmlrpc_server = SimpleXMLRPCServer.SimpleXMLRPCServer(
          (hostname, port), logRequests=False, allow_none=True)
    except SocketServer.socket.error as exception:
      logger.warning((
          'Unable to bind a RPC server on {0:s}:{1:d} with error: '
          '{2!s}').format(hostname, port, exception))
      return False

    self._xmlrpc_server.register_function(
        self._callback, self._RPC_FUNCTION_NAME)
    return True

def predict(self, X):
        """Predict the class for X.

        The predicted class for each sample in X is returned.

        Parameters
        ----------
        X : List of ndarrays, one for each training example.
            Each training example's shape is (string1_len,
            string2_len, n_features), where string1_len and
            string2_len are the length of the two training strings and
            n_features the number of features.

        Returns
        -------
        y : iterable of shape = [n_samples]
            The predicted classes.

        """
        return [self.classes[prediction.argmax()] for prediction in self.predict_proba(X)]

def xor(a, b):
        """Bitwise xor on equal length bytearrays."""
        return bytearray(i ^ j for i, j in zip(a, b))

def stop(pid):
    """Shut down a specific process.

    Args:
      pid: the pid of the process to shutdown.
    """
    if psutil.pid_exists(pid):
      try:
        p = psutil.Process(pid)
        p.kill()
      except Exception:
        pass

def yaml_to_param(obj, name):
	"""
	Return the top-level element of a document sub-tree containing the
	YAML serialization of a Python object.
	"""
	return from_pyvalue(u"yaml:%s" % name, unicode(yaml.dump(obj)))

def timeout_thread_handler(timeout, stop_event):
    """A background thread to kill the process if it takes too long.

    Args:
        timeout (float): The number of seconds to wait before killing
            the process.
        stop_event (Event): An optional event to cleanly stop the background
            thread if required during testing.
    """

    stop_happened = stop_event.wait(timeout)
    if stop_happened is False:
        print("Killing program due to %f second timeout" % timeout)

    os._exit(2)

def ParseMany(text):
  """Parses many YAML documents into a list of Python objects.

  Args:
    text: A YAML source with multiple documents embedded.

  Returns:
    A list of Python data structures corresponding to the YAML documents.
  """
  precondition.AssertType(text, Text)

  if compatibility.PY2:
    text = text.encode("utf-8")

  return list(yaml.safe_load_all(text))

def timeout_thread_handler(timeout, stop_event):
    """A background thread to kill the process if it takes too long.

    Args:
        timeout (float): The number of seconds to wait before killing
            the process.
        stop_event (Event): An optional event to cleanly stop the background
            thread if required during testing.
    """

    stop_happened = stop_event.wait(timeout)
    if stop_happened is False:
        print("Killing program due to %f second timeout" % timeout)

    os._exit(2)

def safe_dump(data, stream=None, **kwds):
    """implementation of safe dumper using Ordered Dict Yaml Dumper"""
    return yaml.dump(data, stream=stream, Dumper=ODYD, **kwds)

def cli_command_quit(self, msg):
        """\
        kills the child and exits
        """
        if self.state == State.RUNNING and self.sprocess and self.sprocess.proc:
            self.sprocess.proc.kill()
        else:
            sys.exit(0)

def yaml_to_param(obj, name):
	"""
	Return the top-level element of a document sub-tree containing the
	YAML serialization of a Python object.
	"""
	return from_pyvalue(u"yaml:%s" % name, unicode(yaml.dump(obj)))

def kill_test_logger(logger):
    """Cleans up a test logger object by removing all of its handlers.

    Args:
        logger: The logging object to clean up.
    """
    for h in list(logger.handlers):
        logger.removeHandler(h)
        if isinstance(h, logging.FileHandler):
            h.close()

def trap_exceptions(results, handler, exceptions=Exception):
	"""
	Iterate through the results, but if an exception occurs, stop
	processing the results and instead replace
	the results with the output from the exception handler.
	"""
	try:
		for result in results:
			yield result
	except exceptions as exc:
		for result in always_iterable(handler(exc)):
			yield result

def _most_common(iterable):
    """Returns the most common element in `iterable`."""
    data = Counter(iterable)
    return max(data, key=data.__getitem__)

def connected_socket(address, timeout=3):
    """ yields a connected socket """
    sock = socket.create_connection(address, timeout)
    yield sock
    sock.close()

def value_left(self, other):
    """
    Returns the value of the other type instance to use in an
    operator method, namely when the method's instance is on the
    left side of the expression.
    """
    return other.value if isinstance(other, self.__class__) else other

def _return_result(self, done):
        """Called set the returned future's state that of the future
        we yielded, and set the current future for the iterator.
        """
        chain_future(done, self._running_future)

        self.current_future = done
        self.current_index = self._unfinished.pop(done)

def pprint(obj, verbose=False, max_width=79, newline='\n'):
    """
    Like `pretty` but print to stdout.
    """
    printer = RepresentationPrinter(sys.stdout, verbose, max_width, newline)
    printer.pretty(obj)
    printer.flush()
    sys.stdout.write(newline)
    sys.stdout.flush()

def unzip_file_to_dir(path_to_zip, output_directory):
    """
    Extract a ZIP archive to a directory
    """
    z = ZipFile(path_to_zip, 'r')
    z.extractall(output_directory)
    z.close()

def _get_xy_scaling_parameters(self):
        """Get the X/Y coordinate limits for the full resulting image"""
        return self.mx, self.bx, self.my, self.by

def open01(x, limit=1.e-6):
    """Constrain numbers to (0,1) interval"""
    try:
        return np.array([min(max(y, limit), 1. - limit) for y in x])
    except TypeError:
        return min(max(x, limit), 1. - limit)

def init_mq(self):
        """Init connection and consumer with openstack mq."""
        mq = self.init_connection()
        self.init_consumer(mq)
        return mq.connection

def _on_text_changed(self):
        """ Adjust dirty flag depending on editor's content """
        if not self._cleaning:
            ln = TextHelper(self).cursor_position()[0]
            self._modified_lines.add(ln)

def qsize(self):
        """Return the approximate size of the queue (not reliable!)."""
        self.mutex.acquire()
        n = self._qsize()
        self.mutex.release()
        return n

def _linearInterpolationTransformMatrix(matrix1, matrix2, value):
    """ Linear, 'oldstyle' interpolation of the transform matrix."""
    return tuple(_interpolateValue(matrix1[i], matrix2[i], value) for i in range(len(matrix1)))

def clean(some_string, uppercase=False):
    """
    helper to clean up an input string
    """
    if uppercase:
        return some_string.strip().upper()
    else:
        return some_string.strip().lower()

def stack_as_string():
    """
    stack_as_string
    """
    if sys.version_info.major == 3:
        stack = io.StringIO()
    else:
        stack = io.BytesIO()

    traceback.print_stack(file=stack)
    stack.seek(0)
    stack = stack.read()
    return stack

def qsize(self):
        """Return the approximate size of the queue (not reliable!)."""
        self.mutex.acquire()
        n = self._qsize()
        self.mutex.release()
        return n

def get_previous_month(self):
        """Returns date range for the previous full month."""
        end = utils.get_month_start() - relativedelta(days=1)
        end = utils.to_datetime(end)
        start = utils.get_month_start(end)
        return start, end

def shape_list(l,shape,dtype):
    """ Shape a list of lists into the appropriate shape and data type """
    return np.array(l, dtype=dtype).reshape(shape)

def append_position_to_token_list(token_list):
    """Converts a list of Token into a list of Token, asuming size == 1"""
    return [PositionToken(value.content, value.gd, index, index+1) for (index, value) in enumerate(token_list)]

def ms_to_datetime(ms):
    """
    Converts a millisecond accuracy timestamp to a datetime
    """
    dt = datetime.datetime.utcfromtimestamp(ms / 1000)
    return dt.replace(microsecond=(ms % 1000) * 1000).replace(tzinfo=pytz.utc)

def display_list_by_prefix(names_list, starting_spaces=0):
  """Creates a help string for names_list grouped by prefix."""
  cur_prefix, result_lines = None, []
  space = " " * starting_spaces
  for name in sorted(names_list):
    split = name.split("_", 1)
    prefix = split[0]
    if cur_prefix != prefix:
      result_lines.append(space + prefix + ":")
      cur_prefix = prefix
    result_lines.append(space + "  * " + name)
  return "\n".join(result_lines)

def _get_random_id():
    """ Get a random (i.e., unique) string identifier"""
    symbols = string.ascii_uppercase + string.ascii_lowercase + string.digits
    return ''.join(random.choice(symbols) for _ in range(15))

def is_iterable(value):
    """must be an iterable (list, array, tuple)"""
    return isinstance(value, np.ndarray) or isinstance(value, list) or isinstance(value, tuple), value

def cint32_array_to_numpy(cptr, length):
    """Convert a ctypes int pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_int32)):
        return np.fromiter(cptr, dtype=np.int32, count=length)
    else:
        raise RuntimeError('Expected int pointer')

def flatten_array(grid):
    """
    Takes a multi-dimensional array and returns a 1 dimensional array with the
    same contents.
    """
    grid = [grid[i][j] for i in range(len(grid)) for j in range(len(grid[i]))]
    while type(grid[0]) is list:
        grid = flatten_array(grid)
    return grid

def get_md5_for_file(file):
    """Get the md5 hash for a file.

    :param file: the file to get the md5 hash for
    """
    md5 = hashlib.md5()

    while True:
        data = file.read(md5.block_size)

        if not data:
            break

        md5.update(data)

    return md5.hexdigest()

def _get_compiled_ext():
    """Official way to get the extension of compiled files (.pyc or .pyo)"""
    for ext, mode, typ in imp.get_suffixes():
        if typ == imp.PY_COMPILED:
            return ext

def get_iter_string_reader(stdin):
    """ return an iterator that returns a chunk of a string every time it is
    called.  notice that even though bufsize_type might be line buffered, we're
    not doing any line buffering here.  that's because our StreamBufferer
    handles all buffering.  we just need to return a reasonable-sized chunk. """
    bufsize = 1024
    iter_str = (stdin[i:i + bufsize] for i in range(0, len(stdin), bufsize))
    return get_iter_chunk_reader(iter_str)

def _get_all_constants():
    """
    Get list of all uppercase, non-private globals (doesn't start with ``_``).

    Returns:
        list: Uppercase names defined in `globals()` (variables from this \
              module).
    """
    return [
        key for key in globals().keys()
        if all([
            not key.startswith("_"),          # publicly accesible
            key.upper() == key,               # uppercase
            type(globals()[key]) in _ALLOWED  # and with type from _ALLOWED
        ])
    ]

def dedupe_list(l):
    """Remove duplicates from a list preserving the order.

    We might be tempted to use the list(set(l)) idiom, but it doesn't preserve
    the order, which hinders testability and does not work for lists with
    unhashable elements.
    """
    result = []

    for el in l:
        if el not in result:
            result.append(el)

    return result

def recarray(self):
        """Returns data as :class:`numpy.recarray`."""
        return numpy.rec.fromrecords(self.records, names=self.names)

def ip_address_list(ips):
    """ IP address range validation and expansion. """
    # first, try it as a single IP address
    try:
        return ip_address(ips)
    except ValueError:
        pass
    # then, consider it as an ipaddress.IPv[4|6]Network instance and expand it
    return list(ipaddress.ip_network(u(ips)).hosts())

def be_array_from_bytes(fmt, data):
    """
    Reads an array from bytestring with big-endian data.
    """
    arr = array.array(str(fmt), data)
    return fix_byteorder(arr)

def polite_string(a_string):
    """Returns a "proper" string that should work in both Py3/Py2"""
    if is_py3() and hasattr(a_string, 'decode'):
        try:
            return a_string.decode('utf-8')
        except UnicodeDecodeError:
            return a_string

    return a_string

def import_js(path, lib_name, globals):
    """Imports from javascript source file.
      globals is your globals()"""
    with codecs.open(path_as_local(path), "r", "utf-8") as f:
        js = f.read()
    e = EvalJs()
    e.execute(js)
    var = e.context['var']
    globals[lib_name] = var.to_python()

def paste(cmd=paste_cmd, stdout=PIPE):
    """Returns system clipboard contents.
    """
    return Popen(cmd, stdout=stdout).communicate()[0].decode('utf-8')

def Load(file):
    """ Loads a model from specified file """
    with open(file, 'rb') as file:
        model = dill.load(file)
        return model

def py3round(number):
    """Unified rounding in all python versions."""
    if abs(round(number) - number) == 0.5:
        return int(2.0 * round(number / 2.0))

    return int(round(number))

def load_yaml(filepath):
    """Convenience function for loading yaml-encoded data from disk."""
    with open(filepath) as f:
        txt = f.read()
    return yaml.load(txt)

def delete_lines(self):
        """
        Deletes the document lines under cursor.

        :return: Method success.
        :rtype: bool
        """

        cursor = self.textCursor()
        self.__select_text_under_cursor_blocks(cursor)
        cursor.removeSelectedText()
        cursor.deleteChar()
        return True

def __enter__(self):
        """Acquire a lock on the output file, prevents collisions between multiple runs."""
        self.fd = open(self.filename, 'a')
        fcntl.lockf(self.fd, fcntl.LOCK_EX)
        return self.fd

def to_bytes(value):
    """ str to bytes (py3k) """
    vtype = type(value)

    if vtype == bytes or vtype == type(None):
        return value

    try:
        return vtype.encode(value)
    except UnicodeEncodeError:
        pass
    return value

def ln_norm(x, mu, sigma=1.0):
    """ Natural log of scipy norm function truncated at zero """
    return np.log(stats.norm(loc=mu, scale=sigma).pdf(x))

def _run_cmd_get_output(cmd):
    """Runs a shell command, returns console output.

    Mimics python3's subprocess.getoutput
    """
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    out, err = process.communicate()
    return out or err

def cric__lasso():
    """ Lasso Regression
    """
    model = sklearn.linear_model.LogisticRegression(penalty="l1", C=0.002)

    # we want to explain the raw probability outputs of the trees
    model.predict = lambda X: model.predict_proba(X)[:,1]
    
    return model

def unique(transactions):
    """ Remove any duplicate entries. """
    seen = set()
    # TODO: Handle comments
    return [x for x in transactions if not (x in seen or seen.add(x))]

def survival(value=t, lam=lam, f=failure):
    """Exponential survival likelihood, accounting for censoring"""
    return sum(f * log(lam) - lam * value)

def to_identifier(s):
  """
  Convert snake_case to camel_case.
  """
  if s.startswith('GPS'):
      s = 'Gps' + s[3:]
  return ''.join([i.capitalize() for i in s.split('_')]) if '_' in s else s

def get_longest_orf(orfs):
    """Find longest ORF from the given list of ORFs."""
    sorted_orf = sorted(orfs, key=lambda x: len(x['sequence']), reverse=True)[0]
    return sorted_orf

def get_table_columns(dbconn, tablename):
    """
    Return a list of tuples specifying the column name and type
    """
    cur = dbconn.cursor()
    cur.execute("PRAGMA table_info('%s');" % tablename)
    info = cur.fetchall()
    cols = [(i[1], i[2]) for i in info]
    return cols

def strip_spaces(s):
    """ Strip excess spaces from a string """
    return u" ".join([c for c in s.split(u' ') if c])

def is_non_empty_string(input_string):
    """
    Validate if non empty string

    :param input_string: Input is a *str*.
    :return: True if input is string and non empty.
       Raise :exc:`Exception` otherwise.
    """
    try:
        if not input_string.strip():
            raise ValueError()
    except AttributeError as error:
        raise TypeError(error)

    return True

def _unordered_iterator(self):
        """
        Return the value of each QuerySet, but also add the '#' property to each
        return item.
        """
        for i, qs in zip(self._queryset_idxs, self._querysets):
            for item in qs:
                setattr(item, '#', i)
                yield item

def handle_qbytearray(obj, encoding):
    """Qt/Python2/3 compatibility helper."""
    if isinstance(obj, QByteArray):
        obj = obj.data()

    return to_text_string(obj, encoding=encoding)

def searchlast(self,n=10):
        """Return the last n results (or possibly less if not found). Note that the last results are not necessarily the best ones! Depending on the search type."""            
        solutions = deque([], n)
        for solution in self:
            solutions.append(solution)
        return solutions

def osx_clipboard_get():
    """ Get the clipboard's text on OS X.
    """
    p = subprocess.Popen(['pbpaste', '-Prefer', 'ascii'],
        stdout=subprocess.PIPE)
    text, stderr = p.communicate()
    # Text comes in with old Mac \r line endings. Change them to \n.
    text = text.replace('\r', '\n')
    return text

def urlencoded(body, charset='ascii', **kwargs):
    """Converts query strings into native Python objects"""
    return parse_query_string(text(body, charset=charset), False)

def magnitude(X):
    """Magnitude of a complex matrix."""
    r = np.real(X)
    i = np.imag(X)
    return np.sqrt(r * r + i * i);

def pack_triples_numpy(triples):
    """Packs a list of triple indexes into a 2D numpy array."""
    if len(triples) == 0:
        return np.array([], dtype=np.int64)
    return np.stack(list(map(_transform_triple_numpy, triples)), axis=0)

def image_set_aspect(aspect=1.0, axes="gca"):
    """
    sets the aspect ratio of the current zoom level of the imshow image
    """
    if axes is "gca": axes = _pylab.gca()

    e = axes.get_images()[0].get_extent()
    axes.set_aspect(abs((e[1]-e[0])/(e[3]-e[2]))/aspect)

def positive_integer(anon, obj, field, val):
    """
    Returns a random positive integer (for a Django PositiveIntegerField)
    """
    return anon.faker.positive_integer(field=field)

def str_dict(some_dict):
    """Convert dict of ascii str/unicode to dict of str, if necessary"""
    return {str(k): str(v) for k, v in some_dict.items()}

def runiform(lower, upper, size=None):
    """
    Random uniform variates.
    """
    return np.random.uniform(lower, upper, size)

def stringify_dict_contents(dct):
    """Turn dict keys and values into native strings."""
    return {
        str_if_nested_or_str(k): str_if_nested_or_str(v)
        for k, v in dct.items()
    }

def range(self, chromosome, start, stop, exact=False):
        """
        Shortcut to do range filters on genomic datasets.
        """
        return self._clone(
            filters=[GenomicFilter(chromosome, start, stop, exact)])

def get_neg_infinity(dtype):
    """Return an appropriate positive infinity for this dtype.

    Parameters
    ----------
    dtype : np.dtype

    Returns
    -------
    fill_value : positive infinity value corresponding to this dtype.
    """
    if issubclass(dtype.type, (np.floating, np.integer)):
        return -np.inf

    if issubclass(dtype.type, np.complexfloating):
        return -np.inf - 1j * np.inf

    return NINF

def open_json(file_name):
    """
    returns json contents as string
    """
    with open(file_name, "r") as json_data:
        data = json.load(json_data)
        return data

def strToBool(val):
    """
    Helper function to turn a string representation of "true" into
    boolean True.
    """
    if isinstance(val, str):
        val = val.lower()

    return val in ['true', 'on', 'yes', True]

def be_array_from_bytes(fmt, data):
    """
    Reads an array from bytestring with big-endian data.
    """
    arr = array.array(str(fmt), data)
    return fix_byteorder(arr)

def parse_date(s):
    """Fast %Y-%m-%d parsing."""
    try:
        return datetime.date(int(s[:4]), int(s[5:7]), int(s[8:10]))
    except ValueError:  # other accepted format used in one-day data set
        return datetime.datetime.strptime(s, '%d %B %Y').date()

def read_numpy(fd, byte_order, dtype, count):
    """Read tag data from file and return as numpy array."""
    return numpy.fromfile(fd, byte_order+dtype[-1], count)

def _clear(self):
        """
        Helper that clears the composition.
        """
        draw = ImageDraw.Draw(self._background_image)
        draw.rectangle(self._device.bounding_box,
                       fill="black")
        del draw

def import_public_rsa_key_from_file(filename):
    """
    Read a public RSA key from a PEM file.

    :param filename: The name of the file
    :param passphrase: A pass phrase to use to unpack the PEM file.
    :return: A
        cryptography.hazmat.primitives.asymmetric.rsa.RSAPublicKey instance
    """
    with open(filename, "rb") as key_file:
        public_key = serialization.load_pem_public_key(
            key_file.read(),
            backend=default_backend())
    return public_key

def mouse_move_event(self, event):
        """
        Forward mouse cursor position events to the example
        """
        self.example.mouse_position_event(event.x(), event.y())

def get_as_bytes(self, s3_path):
        """
        Get the contents of an object stored in S3 as bytes

        :param s3_path: URL for target S3 location
        :return: File contents as pure bytes
        """
        (bucket, key) = self._path_to_bucket_and_key(s3_path)
        obj = self.s3.Object(bucket, key)
        contents = obj.get()['Body'].read()
        return contents

def load_data(filename):
    """
    :rtype : numpy matrix
    """
    data = pandas.read_csv(filename, header=None, delimiter='\t', skiprows=9)
    return data.as_matrix()

def _readuntil(f, end=_TYPE_END):
	"""Helper function to read bytes until a certain end byte is hit"""
	buf = bytearray()
	byte = f.read(1)
	while byte != end:
		if byte == b'':
			raise ValueError('File ended unexpectedly. Expected end byte {}.'.format(end))
		buf += byte
		byte = f.read(1)
	return buf

def hash_producer(*args, **kwargs):
    """ Returns a random hash for a confirmation secret. """
    return hashlib.md5(six.text_type(uuid.uuid4()).encode('utf-8')).hexdigest()

def load_yaml(filepath):
    """Convenience function for loading yaml-encoded data from disk."""
    with open(filepath) as f:
        txt = f.read()
    return yaml.load(txt)

def to_camel(s):
    """
    :param string s: under_scored string to be CamelCased
    :return: CamelCase version of input
    :rtype: str
    """
    # r'(?!^)_([a-zA-Z]) original regex wasn't process first groups
    return re.sub(r'_([a-zA-Z])', lambda m: m.group(1).upper(), '_' + s)

def read_dict_from_file(file_path):
    """
    Read a dictionary of strings from a file
    """
    with open(file_path) as file:
        lines = file.read().splitlines()

    obj = {}
    for line in lines:
        key, value = line.split(':', maxsplit=1)
        obj[key] = eval(value)

    return obj

def get_translucent_cmap(r, g, b):

    class TranslucentCmap(BaseColormap):
        glsl_map = """
        vec4 translucent_fire(float t) {{
            return vec4({0}, {1}, {2}, t);
        }}
        """.format(r, g, b)

    return TranslucentCmap()

def get_jsonparsed_data(url):
    """Receive the content of ``url``, parse it as JSON and return the
       object.
    """
    response = urlopen(url)
    data = response.read().decode('utf-8')
    return json.loads(data)

def reduce_freqs(freqlist):
    """
    Add up a list of freq counts to get the total counts.
    """
    allfreqs = np.zeros_like(freqlist[0])
    for f in freqlist:
        allfreqs += f
    return allfreqs

def __setitem__(self, field, value):
        """ :see::meth:RedisMap.__setitem__ """
        return self._client.hset(self.key_prefix, field, self._dumps(value))

def get_from_human_key(self, key):
        """Return the key (aka database value) of a human key (aka Python identifier)."""
        if key in self._identifier_map:
            return self._identifier_map[key]
        raise KeyError(key)

def retrieve_import_alias_mapping(names_list):
    """Creates a dictionary mapping aliases to their respective name.
    import_alias_names is used in module_definitions.py and visit_Call"""
    import_alias_names = dict()

    for alias in names_list:
        if alias.asname:
            import_alias_names[alias.asname] = alias.name
    return import_alias_names

def get_unixtime_registered(self):
        """Returns the user's registration date as a UNIX timestamp."""

        doc = self._request(self.ws_prefix + ".getInfo", True)

        return int(doc.getElementsByTagName("registered")[0].getAttribute("unixtime"))

def heappush_max(heap, item):
    """Push item onto heap, maintaining the heap invariant."""
    heap.append(item)
    _siftdown_max(heap, 0, len(heap) - 1)

def split_comma_argument(comma_sep_str):
    """Split a comma separated option into a list."""
    terms = []
    for term in comma_sep_str.split(','):
        if term:
            terms.append(term)
    return terms

def _multiline_width(multiline_s, line_width_fn=len):
    """Visible width of a potentially multiline content."""
    return max(map(line_width_fn, re.split("[\r\n]", multiline_s)))

def input_dir(self):
        """
        :returns: absolute path to where the job.ini is
        """
        return os.path.abspath(os.path.dirname(self.inputs['job_ini']))

def shader_string(body, glsl_version='450 core'):
    """
    Call this method from a function that defines a literal shader string as the "body" argument.
    Dresses up a shader string in three ways:
        1) Insert #version at the top
        2) Insert #line number declaration
        3) un-indents
    The line number information can help debug glsl compile errors.
    The version string needs to be the very first characters in the shader,
    which can be distracting, requiring backslashes or other tricks.
    The unindenting allows you to type the shader code at a pleasing indent level
    in your python method, while still creating an unindented GLSL string at the end.
    """
    line_count = len(body.split('\n'))
    line_number = inspect.currentframe().f_back.f_lineno + 1 - line_count
    return """\
#version %s
%s
""" % (glsl_version, shader_substring(body, stack_frame=2))

def softplus(attrs, inputs, proto_obj):
    """Applies the sofplus activation function element-wise to the input."""
    new_attrs = translation_utils._add_extra_attributes(attrs, {'act_type' : 'softrelu'})
    return 'Activation', new_attrs, inputs

def get_remote_content(filepath):
        """ A handy wrapper to get a remote file content """
        with hide('running'):
            temp = BytesIO()
            get(filepath, temp)
            content = temp.getvalue().decode('utf-8')
        return content.strip()

def __len__(self):
        """ This will equal 124 for the V1 database. """
        length = 0
        for typ, siz, _ in self.format:
            length += siz
        return length

def remove_bad(string):
    """
    remove problem characters from string
    """
    remove = [':', ',', '(', ')', ' ', '|', ';', '\'']
    for c in remove:
        string = string.replace(c, '_')
    return string

def array_bytes(array):
    """ Estimates the memory of the supplied array in bytes """
    return np.product(array.shape)*np.dtype(array.dtype).itemsize

def unique(seq):
    """Return the unique elements of a collection even if those elements are
       unhashable and unsortable, like dicts and sets"""
    cleaned = []
    for each in seq:
        if each not in cleaned:
            cleaned.append(each)
    return cleaned

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

def strip_accents(string):
    """
    Strip all the accents from the string
    """
    return u''.join(
        (character for character in unicodedata.normalize('NFD', string)
         if unicodedata.category(character) != 'Mn'))

def IsBinary(self, filename):
		"""Returns true if the guessed mimetyped isnt't in text group."""
		mimetype = mimetypes.guess_type(filename)[0]
		if not mimetype:
			return False  # e.g. README, "real" binaries usually have an extension
		# special case for text files which don't start with text/
		if mimetype in TEXT_MIMETYPES:
			return False
		return not mimetype.startswith("text/")

def remove_punctuation(text, exceptions=[]):
    """
    Return a string with punctuation removed.

    Parameters:
        text (str): The text to remove punctuation from.
        exceptions (list): List of symbols to keep in the given text.

    Return:
        str: The input text without the punctuation.
    """

    all_but = [
        r'\w',
        r'\s'
    ]

    all_but.extend(exceptions)

    pattern = '[^{}]'.format(''.join(all_but))

    return re.sub(pattern, '', text)

def SegmentMin(a, ids):
    """
    Segmented min op.
    """
    func = lambda idxs: np.amin(a[idxs], axis=0)
    return seg_map(func, a, ids),

def remove_bad(string):
    """
    remove problem characters from string
    """
    remove = [':', ',', '(', ')', ' ', '|', ';', '\'']
    for c in remove:
        string = string.replace(c, '_')
    return string

def copy_and_update(dictionary, update):
    """Returns an updated copy of the dictionary without modifying the original"""
    newdict = dictionary.copy()
    newdict.update(update)
    return newdict

def slugify(string):
    """
    Removes non-alpha characters, and converts spaces to hyphens. Useful for making file names.


    Source: http://stackoverflow.com/questions/5574042/string-slugification-in-python
    """
    string = re.sub('[^\w .-]', '', string)
    string = string.replace(" ", "-")
    return string

def find_one_by_id(self, _id):
        """
        Find a single document by id

        :param str _id: BSON string repreentation of the Id
        :return: a signle object
        :rtype: dict

        """
        document = (yield self.collection.find_one({"_id": ObjectId(_id)}))
        raise Return(self._obj_cursor_to_dictionary(document))

def strip_spaces(value, sep=None, join=True):
    """Cleans trailing whitespaces and replaces also multiple whitespaces with a single space."""
    value = value.strip()
    value = [v.strip() for v in value.split(sep)]
    join_sep = sep or ' '
    return join_sep.join(value) if join else value

def mostCommonItem(lst):
    """Choose the most common item from the list, or the first item if all
    items are unique."""
    # This elegant solution from: http://stackoverflow.com/a/1518632/1760218
    lst = [l for l in lst if l]
    if lst:
        return max(set(lst), key=lst.count)
    else:
        return None

def __normalize_list(self, msg):
        """Split message to list by commas and trim whitespace."""
        if isinstance(msg, list):
            msg = "".join(msg)
        return list(map(lambda x: x.strip(), msg.split(",")))

def mostCommonItem(lst):
    """Choose the most common item from the list, or the first item if all
    items are unique."""
    # This elegant solution from: http://stackoverflow.com/a/1518632/1760218
    lst = [l for l in lst if l]
    if lst:
        return max(set(lst), key=lst.count)
    else:
        return None

def dump_nparray(self, obj, class_name=numpy_ndarray_class_name):
        """
        ``numpy.ndarray`` dumper.
        """
        return {"$" + class_name: self._json_convert(obj.tolist())}

def moving_average(array, n=3):
    """
    Calculates the moving average of an array.

    Parameters
    ----------
    array : array
        The array to have the moving average taken of
    n : int
        The number of points of moving average to take
    
    Returns
    -------
    MovingAverageArray : array
        The n-point moving average of the input array
    """
    ret = _np.cumsum(array, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def remove_dups(seq):
    """remove duplicates from a sequence, preserving order"""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def reduce_multiline(string):
    """
    reduces a multiline string to a single line of text.


    args:
        string: the text to reduce
    """
    string = str(string)
    return " ".join([item.strip()
                     for item in string.split("\n")
                     if item.strip()])

def get_element_with_id(self, id):
        """Return the element with the specified ID."""
        # Should we maintain a hashmap of ids to make this more efficient? Probably overkill.
        # TODO: Elements can contain nested elements (captions, footnotes, table cells, etc.)
        return next((el for el in self.elements if el.id == id), None)

def store_many(self, sql, values):
        """Abstraction over executemany method"""
        cursor = self.get_cursor()
        cursor.executemany(sql, values)
        self.conn.commit()

def remove_series(self, series):
        """Removes a :py:class:`.Series` from the chart.

        :param Series series: The :py:class:`.Series` to remove.
        :raises ValueError: if you try to remove the last\
        :py:class:`.Series`."""

        if len(self.all_series()) == 1:
            raise ValueError("Cannot remove last series from %s" % str(self))
        self._all_series.remove(series)
        series._chart = None

def matrixTimesVector(MM, aa):
    """

    :param MM: A matrix of size 3x3
    :param aa: A vector of size 3
    :return: A vector of size 3 which is the product of the matrix by the vector
    """
    bb = np.zeros(3, np.float)
    for ii in range(3):
        bb[ii] = np.sum(MM[ii, :] * aa)
    return bb

def unique(_list):
    """
    Makes the list have unique items only and maintains the order

    list(set()) won't provide that

    :type _list list
    :rtype: list
    """
    ret = []

    for item in _list:
        if item not in ret:
            ret.append(item)

    return ret

def multiply(self, number):
        """Return a Vector as the product of the vector and a real number."""
        return self.from_list([x * number for x in self.to_list()])

def clean_py_files(path):
    """
    Removes all .py files.

    :param path: the path
    :return: None
    """

    for dirname, subdirlist, filelist in os.walk(path):

        for f in filelist:
            if f.endswith('py'):
                os.remove(os.path.join(dirname, f))

def validate_string_list(lst):
    """Validate that the input is a list of strings.

    Raises ValueError if not."""
    if not isinstance(lst, list):
        raise ValueError('input %r must be a list' % lst)
    for x in lst:
        if not isinstance(x, basestring):
            raise ValueError('element %r in list must be a string' % x)

def strip_accents(text):
    """
    Strip agents from a string.
    """

    normalized_str = unicodedata.normalize('NFD', text)

    return ''.join([
        c for c in normalized_str if unicodedata.category(c) != 'Mn'])

def close( self ):
        """
        Close the db and release memory
        """
        if self.db is not None:
            self.db.commit()
            self.db.close()
            self.db = None

        return

def normalize_value(text):
    """
    This removes newlines and multiple spaces from a string.
    """
    result = text.replace('\n', ' ')
    result = re.subn('[ ]{2,}', ' ', result)[0]
    return result

def get_url_file_name(url):
    """Get the file name from an url
    
    Parameters
    ----------
    url : str

    Returns
    -------
    str
        The file name 
    """

    assert isinstance(url, (str, _oldstr))
    return urlparse.urlparse(url).path.split('/')[-1]

def strip_accents(s):
    """
    Strip accents to prepare for slugification.
    """
    nfkd = unicodedata.normalize('NFKD', unicode(s))
    return u''.join(ch for ch in nfkd if not unicodedata.combining(ch))

def dump_nparray(self, obj, class_name=numpy_ndarray_class_name):
        """
        ``numpy.ndarray`` dumper.
        """
        return {"$" + class_name: self._json_convert(obj.tolist())}

def drop_empty(rows):
    """Transpose the columns into rows, remove all of the rows that are empty after the first cell, then
    transpose back. The result is that columns that have a header but no data in the body are removed, assuming
    the header is the first row. """
    return zip(*[col for col in zip(*rows) if bool(filter(bool, col[1:]))])

def find_nearest_index(arr, value):
    """For a given value, the function finds the nearest value
    in the array and returns its index."""
    arr = np.array(arr)
    index = (abs(arr-value)).argmin()
    return index

def rm_keys_from_dict(d, keys):
    """
    Given a dictionary and a key list, remove any data in the dictionary with the given keys.

    :param dict d: Metadata
    :param list keys: Keys to be removed
    :return dict d: Metadata
    """
    # Loop for each key given
    for key in keys:
        # Is the key in the dictionary?
        if key in d:
            try:
                d.pop(key, None)
            except KeyError:
                # Not concerned with an error. Keep going.
                pass
    return d

def index_nearest(value, array):
    """
    expects a _n.array
    returns the global minimum of (value-array)^2
    """

    a = (array-value)**2
    return index(a.min(), a)

def software_fibonacci(n):
    """ a normal old python function to return the Nth fibonacci number. """
    a, b = 0, 1
    for i in range(n):
        a, b = b, a + b
    return a

def _breakRemNewlines(tag):
	"""non-recursively break spaces and remove newlines in the tag"""
	for i,c in enumerate(tag.contents):
		if type(c) != bs4.element.NavigableString:
			continue
		c.replace_with(re.sub(r' {2,}', ' ', c).replace('\n',''))

def forget_xy(t):
  """Ignore sizes of dimensions (1, 2) of a 4d tensor in shape inference.

  This allows using smaller input sizes, which create an invalid graph at higher
  layers (for example because a spatial dimension becomes smaller than a conv
  filter) when we only use early parts of it.
  """
  shape = (t.shape[0], None, None, t.shape[3])
  return tf.placeholder_with_default(t, shape)

def text_cleanup(data, key, last_type):
    """ I strip extra whitespace off multi-line strings if they are ready to be stripped!"""
    if key in data and last_type == STRING_TYPE:
        data[key] = data[key].strip()
    return data

def get_dimension_array(array):
    """
    Get dimension of an array getting the number of rows and the max num of
    columns.
    """
    if all(isinstance(el, list) for el in array):
        result = [len(array), len(max([x for x in array], key=len,))]

    # elif array and isinstance(array, list):
    else:
        result = [len(array), 1]

    return result

def _remove_blank(l):
        """ Removes trailing zeros in the list of integers and returns a new list of integers"""
        ret = []
        for i, _ in enumerate(l):
            if l[i] == 0:
                break
            ret.append(l[i])
        return ret

async def async_input(prompt):
    """
    Python's ``input()`` is blocking, which means the event loop we set
    above can't be running while we're blocking there. This method will
    let the loop run while we wait for input.
    """
    print(prompt, end='', flush=True)
    return (await loop.run_in_executor(None, sys.stdin.readline)).rstrip()

def sanitize_word(s):
    """Remove non-alphanumerical characters from metric word.
    And trim excessive underscores.
    """
    s = re.sub('[^\w-]+', '_', s)
    s = re.sub('__+', '_', s)
    return s.strip('_')

def cell_normalize(data):
    """
    Returns the data where the expression is normalized so that the total
    count per cell is equal.
    """
    if sparse.issparse(data):
        data = sparse.csc_matrix(data.astype(float))
        # normalize in-place
        sparse_cell_normalize(data.data,
                data.indices,
                data.indptr,
                data.shape[1],
                data.shape[0])
        return data
    data_norm = data.astype(float)
    total_umis = []
    for i in range(data.shape[1]):
        di = data_norm[:,i]
        total_umis.append(di.sum())
        di /= total_umis[i]
    med = np.median(total_umis)
    data_norm *= med
    return data_norm

def _str_to_list(s):
    """Converts a comma separated string to a list"""
    _list = s.split(",")
    return list(map(lambda i: i.lstrip(), _list))

def phase_correct_first(spec, freq, k):
    """
    First order phase correction.

    Parameters
    ----------
    spec : float array
        The spectrum to be corrected.

    freq : float array
        The frequency axis.

    k : float
        The slope of the phase correction as a function of frequency.

    Returns
    -------
    The phase-corrected spectrum.

    Notes
    -----
    [Keeler2005] Keeler, J (2005). Understanding NMR Spectroscopy, 2nd
        edition. Wiley. Page 88

    """
    c_factor = np.exp(-1j * k * freq)
    c_factor = c_factor.reshape((len(spec.shape) -1) * (1,) + c_factor.shape)
    return spec * c_factor

def wordify(text):
    """Generate a list of words given text, removing punctuation.

    Parameters
    ----------
    text : unicode
        A piece of english text.

    Returns
    -------
    words : list
        List of words.
    """
    stopset = set(nltk.corpus.stopwords.words('english'))
    tokens = nltk.WordPunctTokenizer().tokenize(text)
    return [w for w in tokens if w not in stopset]

def series_table_row_offset(self, series):
        """
        Return the number of rows preceding the data table for *series* in
        the Excel worksheet.
        """
        title_and_spacer_rows = series.index * 2
        data_point_rows = series.data_point_offset
        return title_and_spacer_rows + data_point_rows

def myreplace(astr, thefind, thereplace):
    """in string astr replace all occurences of thefind with thereplace"""
    alist = astr.split(thefind)
    new_s = alist.split(thereplace)
    return new_s

def series_table_row_offset(self, series):
        """
        Return the number of rows preceding the data table for *series* in
        the Excel worksheet.
        """
        title_and_spacer_rows = series.index * 2
        data_point_rows = series.data_point_offset
        return title_and_spacer_rows + data_point_rows

def subn_filter(s, find, replace, count=0):
    """A non-optimal implementation of a regex filter"""
    return re.gsub(find, replace, count, s)

def clean_with_zeros(self,x):
        """ set nan and inf rows from x to zero"""
        x[~np.any(np.isnan(x) | np.isinf(x),axis=1)] = 0
        return x

def read_array(path, mmap_mode=None):
    """Read a .npy array."""
    file_ext = op.splitext(path)[1]
    if file_ext == '.npy':
        return np.load(path, mmap_mode=mmap_mode)
    raise NotImplementedError("The file extension `{}` ".format(file_ext) +
                              "is not currently supported.")

def escape_tex(value):
  """
  Make text tex safe
  """
  newval = value
  for pattern, replacement in LATEX_SUBS:
    newval = pattern.sub(replacement, newval)
  return newval

def index_nearest(value, array):
    """
    expects a _n.array
    returns the global minimum of (value-array)^2
    """

    a = (array-value)**2
    return index(a.min(), a)

def _sub_patterns(patterns, text):
    """
    Apply re.sub to bunch of (pattern, repl)
    """
    for pattern, repl in patterns:
        text = re.sub(pattern, repl, text)
    return text

def as_list(self):
        """Return all child objects in nested lists of strings."""
        return [self.name, self.value, [x.as_list for x in self.children]]

def unapostrophe(text):
    """Strip apostrophe and 's' from the end of a string."""
    text = re.sub(r'[%s]s?$' % ''.join(APOSTROPHES), '', text)
    return text

def parse_querystring(self, req, name, field):
        """Pull a querystring value from the request."""
        return core.get_value(req.args, name, field)

def is_serializable(obj):
    """Return `True` if the given object conforms to the Serializable protocol.

    :rtype: bool
    """
    if inspect.isclass(obj):
      return Serializable.is_serializable_type(obj)
    return isinstance(obj, Serializable) or hasattr(obj, '_asdict')

def device_state(device_id):
    """ Get device state via HTTP GET. """
    if device_id not in devices:
        return jsonify(success=False)
    return jsonify(state=devices[device_id].state)

def get_readonly_fields(self, request, obj=None):
        """Set all fields readonly."""
        return list(self.readonly_fields) + [field.name for field in obj._meta.fields]

def resample(grid, wl, flux):
    """ Resample spectrum onto desired grid """
    flux_rs = (interpolate.interp1d(wl, flux))(grid)
    return flux_rs

def toBase64(s):
    """Represent string / bytes s as base64, omitting newlines"""
    if isinstance(s, str):
        s = s.encode("utf-8")
    return binascii.b2a_base64(s)[:-1]

def downsample_with_striding(array, factor):
    """Downsample x by factor using striding.

    @return: The downsampled array, of the same type as x.
    """
    return array[tuple(np.s_[::f] for f in factor)]

def energy_string_to_float( string ):
    """
    Convert a string of a calculation energy, e.g. '-1.2345 eV' to a float.

    Args:
        string (str): The string to convert.
  
    Return
        (float) 
    """
    energy_re = re.compile( "(-?\d+\.\d+)" )
    return float( energy_re.match( string ).group(0) )

def shape_list(l,shape,dtype):
    """ Shape a list of lists into the appropriate shape and data type """
    return np.array(l, dtype=dtype).reshape(shape)

def utime(self, *args, **kwargs):
        """ Set the access and modified times of the file specified by path. """
        os.utime(self.extended_path, *args, **kwargs)

def batchify(data, batch_size):
    """Reshape data into (num_example, batch_size)"""
    nbatch = data.shape[0] // batch_size
    data = data[:nbatch * batch_size]
    data = data.reshape((batch_size, nbatch)).T
    return data

def unpickle_file(picklefile, **kwargs):
    """Helper function to unpickle data from `picklefile`."""
    with open(picklefile, 'rb') as f:
        return pickle.load(f, **kwargs)

def batchify(data, batch_size):
    """Reshape data into (num_example, batch_size)"""
    nbatch = data.shape[0] // batch_size
    data = data[:nbatch * batch_size]
    data = data.reshape((batch_size, nbatch)).T
    return data

def open_file(file, mode):
	"""Open a file.

	:arg file: file-like or path-like object.
	:arg str mode: ``mode`` argument for :func:`open`.
	"""
	if hasattr(file, "read"):
		return file
	if hasattr(file, "open"):
		return file.open(mode)
	return open(file, mode)

def render_none(self, context, result):
		"""Render empty responses."""
		context.response.body = b''
		del context.response.content_length
		return True

def getfirstline(file, default):
    """
    Returns the first line of a file.
    """
    with open(file, 'rb') as fh:
        content = fh.readlines()
        if len(content) == 1:
            return content[0].decode('utf-8').strip('\n')

    return default

def do_restart(self, line):
        """
        Attempt to restart the bot.
        """
        self.bot._frame = 0
        self.bot._namespace.clear()
        self.bot._namespace.update(self.bot._initial_namespace)

def execfile(fname, variables):
    """ This is builtin in python2, but we have to roll our own on py3. """
    with open(fname) as f:
        code = compile(f.read(), fname, 'exec')
        exec(code, variables)

def extract_module_locals(depth=0):
    """Returns (module, locals) of the funciton `depth` frames away from the caller"""
    f = sys._getframe(depth + 1)
    global_ns = f.f_globals
    module = sys.modules[global_ns['__name__']]
    return (module, f.f_locals)

def read(*args):
    """Reads complete file contents."""
    return io.open(os.path.join(HERE, *args), encoding="utf-8").read()

def get_class_method(cls_or_inst, method_name):
    """
    Returns a method from a given class or instance. When the method doest not exist, it returns `None`. Also works with
    properties and cached properties.
    """
    cls = cls_or_inst if isinstance(cls_or_inst, type) else cls_or_inst.__class__
    meth = getattr(cls, method_name, None)
    if isinstance(meth, property):
        meth = meth.fget
    elif isinstance(meth, cached_property):
        meth = meth.func
    return meth

def imdecode(image_path):
    """Return BGR image read by opencv"""
    import os
    assert os.path.exists(image_path), image_path + ' not found'
    im = cv2.imread(image_path)
    return im

def items(self):
    """Return a list of the (name, value) pairs of the enum.

    These are returned in the order they were defined in the .proto file.
    """
    return [(value_descriptor.name, value_descriptor.number)
            for value_descriptor in self._enum_type.values]

def _openpyxl_read_xl(xl_path: str):
    """ Use openpyxl to read an Excel file. """
    try:
        wb = load_workbook(filename=xl_path, read_only=True)
    except:
        raise
    else:
        return wb

def get_column_keys_and_names(table):
    """
    Return a generator of tuples k, c such that k is the name of the python attribute for
    the column and c is the name of the column in the sql table.
    """
    ins = inspect(table)
    return ((k, c.name) for k, c in ins.mapper.c.items())

def naturalsortkey(s):
    """Natural sort order"""
    return [int(part) if part.isdigit() else part
            for part in re.split('([0-9]+)', s)]

def get_X0(X):
    """ Return zero-th element of a one-element data container.
    """
    if pandas_available and isinstance(X, pd.DataFrame):
        assert len(X) == 1
        x = np.array(X.iloc[0])
    else:
        x, = X
    return x

def project(self, other):
        """Return one vector projected on the vector other"""
        n = other.normalized()
        return self.dot(n) * n

def preprocess(string):
    """
    Preprocess string to transform all diacritics and remove other special characters than appropriate
    
    :param string:
    :return:
    """
    string = unicode(string, encoding="utf-8")
    # convert diacritics to simpler forms
    string = regex1.sub(lambda x: accents[x.group()], string)
    # remove all rest of the unwanted characters
    return regex2.sub('', string).encode('utf-8')

def timestamp(format=DATEFMT, timezone='Africa/Johannesburg'):
    """ Return current datetime with timezone applied
        [all timezones] print sorted(pytz.all_timezones) """

    return formatdate(datetime.now(tz=pytz.timezone(timezone)))

def _try_lookup(table, value, default = ""):
    """ try to get a string from the lookup table, return "" instead of key
    error
    """
    try:
        string = table[ value ]
    except KeyError:
        string = default
    return string

def write_pid_file():
    """Write a file with the PID of this server instance.

    Call when setting up a command line testserver.
    """
    pidfile = os.path.basename(sys.argv[0])[:-3] + '.pid'  # strip .py, add .pid
    with open(pidfile, 'w') as fh:
        fh.write("%d\n" % os.getpid())
        fh.close()

def inverse(d):
    """
    reverse the k:v pairs
    """
    output = {}
    for k, v in unwrap(d).items():
        output[v] = output.get(v, [])
        output[v].append(k)
    return output

def region_from_segment(image, segment):
    """given a segment (rectangle) and an image, returns it's corresponding subimage"""
    x, y, w, h = segment
    return image[y:y + h, x:x + w]

def _round_half_hour(record):
    """
    Round a time DOWN to half nearest half-hour.
    """
    k = record.datetime + timedelta(minutes=-(record.datetime.minute % 30))
    return datetime(k.year, k.month, k.day, k.hour, k.minute, 0)

def unpack2D(_x):
    """
        Helper function for splitting 2D data into x and y component to make
        equations simpler
    """
    _x = np.atleast_2d(_x)
    x = _x[:, 0]
    y = _x[:, 1]
    return x, y

def __round_time(self, dt):
    """Round a datetime object to a multiple of a timedelta
    dt : datetime.datetime object, default now.
    """
    round_to = self._resolution.total_seconds()
    seconds  = (dt - dt.min).seconds
    rounding = (seconds + round_to / 2) // round_to * round_to
    return dt + timedelta(0, rounding - seconds, -dt.microsecond)

def resize_image_with_crop_or_pad(img, target_height, target_width):
    """
    Crops and/or pads an image to a target width and height.

    Resizes an image to a target width and height by either cropping the image or padding it with zeros.

    NO CENTER CROP. NO CENTER PAD. (Just fill bottom right or crop bottom right)

    :param img: Numpy array representing the image.
    :param target_height: Target height.
    :param target_width: Target width.
    :return: The cropped and padded image.
    """
    h, w = target_height, target_width
    max_h, max_w, c = img.shape

    # crop
    img = crop_center(img, min(max_h, h), min(max_w, w))

    # pad
    padded_img = np.zeros(shape=(h, w, c), dtype=img.dtype)
    padded_img[:img.shape[0], :img.shape[1], :img.shape[2]] = img

    return padded_img

def py3round(number):
    """Unified rounding in all python versions."""
    if abs(round(number) - number) == 0.5:
        return int(2.0 * round(number / 2.0))

    return int(round(number))

def _parse(self, date_str, format='%Y-%m-%d'):
        """
        helper function for parsing FRED date string into datetime
        """
        rv = pd.to_datetime(date_str, format=format)
        if hasattr(rv, 'to_pydatetime'):
            rv = rv.to_pydatetime()
        return rv

def home():
    """Temporary helper function to link to the API routes"""
    return dict(links=dict(api='{}{}'.format(request.url, PREFIX[1:]))), \
        HTTPStatus.OK

def parse_comments_for_file(filename):
    """
    Return a list of all parsed comments in a file.  Mostly for testing &
    interactive use.
    """
    return [parse_comment(strip_stars(comment), next_line)
            for comment, next_line in get_doc_comments(read_file(filename))]

def is_valid_row(cls, row):
        """Indicates whether or not the given row contains valid data."""
        for k in row.keys():
            if row[k] is None:
                return False
        return True

def web(host, port):
    """Start web application"""
    from .webserver.web import get_app
    get_app().run(host=host, port=port)

def as_float_array(a):
    """View the quaternion array as an array of floats

    This function is fast (of order 1 microsecond) because no data is
    copied; the returned quantity is just a "view" of the original.

    The output view has one more dimension (of size 4) than the input
    array, but is otherwise the same shape.

    """
    return np.asarray(a, dtype=np.quaternion).view((np.double, 4))

def run(self):
        """Run the event loop."""
        self.signal_init()
        self.listen_init()
        self.logger.info('starting')
        self.loop.start()

def dict_jsonp(param):
    """Convert the parameter into a dictionary before calling jsonp, if it's not already one"""
    if not isinstance(param, dict):
        param = dict(param)
    return jsonp(param)

def file_read(filename):
    """Read a file and close it.  Returns the file source."""
    fobj = open(filename,'r');
    source = fobj.read();
    fobj.close()
    return source

def trigger(self, target: str, trigger: str, parameters: Dict[str, Any]={}):
		"""Calls the specified Trigger of another Area with the optionally given parameters.

		Args:
			target: The name of the target Area.
			trigger: The name of the Trigger.
			parameters: The parameters of the function call.
		"""
		pass

def cat_acc(y_true, y_pred):
    """Categorical accuracy
    """
    return np.mean(y_true.argmax(axis=1) == y_pred.argmax(axis=1))

def tanimoto_coefficient(a, b):
    """Measured similarity between two points in a multi-dimensional space.

    Returns:
        1.0 if the two points completely overlap,
        0.0 if the two points are infinitely far apart.
    """
    return sum(map(lambda (x,y): float(x)*float(y), zip(a,b))) / sum([
          -sum(map(lambda (x,y): float(x)*float(y), zip(a,b))),
           sum(map(lambda x: float(x)**2, a)),
           sum(map(lambda x: float(x)**2, b))])

def flatten(l, types=(list, float)):
    """
    Flat nested list of lists into a single list.
    """
    l = [item if isinstance(item, types) else [item] for item in l]
    return [item for sublist in l for item in sublist]

def dft(blk, freqs, normalize=True):
  """
  Complex non-optimized Discrete Fourier Transform

  Finds the DFT for values in a given frequency list, in order, over the data
  block seen as periodic.

  Parameters
  ----------
  blk :
    An iterable with well-defined length. Don't use this function with Stream
    objects!
  freqs :
    List of frequencies to find the DFT, in rad/sample. FFT implementations
    like numpy.fft.ftt finds the coefficients for N frequencies equally
    spaced as ``line(N, 0, 2 * pi, finish=False)`` for N frequencies.
  normalize :
    If True (default), the coefficient sums are divided by ``len(blk)``,
    and the coefficient for the DC level (frequency equals to zero) is the
    mean of the block. If False, that coefficient would be the sum of the
    data in the block.

  Returns
  -------
  A list of DFT values for each frequency, in the same order that they appear
  in the freqs input.

  Note
  ----
  This isn't a FFT implementation, and performs :math:`O(M . N)` float
  pointing operations, with :math:`M` and :math:`N` equals to the length of
  the inputs. This function can find the DFT for any specific frequency, with
  no need for zero padding or finding all frequencies in a linearly spaced
  band grid with N frequency bins at once.

  """
  dft_data = (sum(xn * cexp(-1j * n * f) for n, xn in enumerate(blk))
                                         for f in freqs)
  if normalize:
    lblk = len(blk)
    return [v / lblk for v in dft_data]
  return list(dft_data)

def pickle_save(thing,fname):
    """save something to a pickle file"""
    pickle.dump(thing, open(fname,"wb"),pickle.HIGHEST_PROTOCOL)
    return thing

def replace_list(items, match, replacement):
    """Replaces occurrences of a match string in a given list of strings and returns
    a list of new strings. The match string can be a regex expression.

    Args:
        items (list):       the list of strings to modify.
        match (str):        the search expression.
        replacement (str):  the string to replace with.
    """
    return [replace(item, match, replacement) for item in items]

def save(self, fname):
        """ Saves the dictionary in json format
        :param fname: file to save to
        """
        with open(fname, 'wb') as f:
            json.dump(self, f)

def plot_decision_boundary(model, X, y, step=0.1, figsize=(10, 8), alpha=0.4, size=20):
    """Plots the classification decision boundary of `model` on `X` with labels `y`.
    Using numpy and matplotlib.
    """

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))

    f, ax = plt.subplots(figsize=figsize)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=alpha)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=size, edgecolor='k')
    plt.show()

def plot_and_save(self, **kwargs):
        """Used when the plot method defined does not create a figure nor calls save_plot
        Then the plot method has to use self.fig"""
        self.fig = pyplot.figure()
        self.plot()
        self.axes = pyplot.gca()
        self.save_plot(self.fig, self.axes, **kwargs)
        pyplot.close(self.fig)

def __unixify(self, s):
        """ stupid windows. converts the backslash to forwardslash for consistency """
        return os.path.normpath(s).replace(os.sep, "/")

def to_dotfile(G: nx.DiGraph, filename: str):
    """ Output a networkx graph to a DOT file. """
    A = to_agraph(G)
    A.write(filename)

def _deserialize(cls, key, value, fields):
        """ Marshal incoming data into Python objects."""
        converter = cls._get_converter_for_field(key, None, fields)
        return converter.deserialize(value)

def generate_write_yaml_to_file(file_name):
    """ generate a method to write the configuration in yaml to the method desired """
    def write_yaml(config):
        with open(file_name, 'w+') as fh:
            fh.write(yaml.dump(config))
    return write_yaml

def print_bintree(tree, indent='  '):
    """print a binary tree"""
    for n in sorted(tree.keys()):
        print "%s%s" % (indent * depth(n,tree), n)

def zoom(ax, xy='x', factor=1):
    """Zoom into axis.

    Parameters
    ----------
    """
    limits = ax.get_xlim() if xy == 'x' else ax.get_ylim()
    new_limits = (0.5*(limits[0] + limits[1])
                  + 1./factor * np.array((-0.5, 0.5)) * (limits[1] - limits[0]))
    if xy == 'x':
        ax.set_xlim(new_limits)
    else:
        ax.set_ylim(new_limits)

def raw_print(*args, **kw):
    """Raw print to sys.__stdout__, otherwise identical interface to print()."""

    print(*args, sep=kw.get('sep', ' '), end=kw.get('end', '\n'),
          file=sys.__stdout__)
    sys.__stdout__.flush()

def page_guiref(arg_s=None):
    """Show a basic reference about the GUI Console."""
    from IPython.core import page
    page.page(gui_reference, auto_html=True)

def pprint_for_ordereddict():
    """
    Context manager that causes pprint() to print OrderedDict objects as nicely
    as standard Python dictionary objects.
    """
    od_saved = OrderedDict.__repr__
    try:
        OrderedDict.__repr__ = dict.__repr__
        yield
    finally:
        OrderedDict.__repr__ = od_saved

def ex(self, cmd):
        """Execute a normal python statement in user namespace."""
        with self.builtin_trap:
            exec cmd in self.user_global_ns, self.user_ns

def geturl(self):
        """
        Returns the URL that was the source of this response.
        If the request that generated this response redirected, this method
        will return the final redirect location.
        """
        if self.retries is not None and len(self.retries.history):
            return self.retries.history[-1].redirect_location
        else:
            return self._request_url

def _elapsed(self):
        """ Returns elapsed time at update. """
        self.last_time = time.time()
        return self.last_time - self.start

def PythonPercentFormat(format_str):
  """Use Python % format strings as template format specifiers."""

  if format_str.startswith('printf '):
    fmt = format_str[len('printf '):]
    return lambda value: fmt % value
  else:
    return None

def rowlenselect(table, n, complement=False):
    """Select rows of length `n`."""

    where = lambda row: len(row) == n
    return select(table, where, complement=complement)

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

def _pick_attrs(attrs, keys):
    """ Return attrs with keys in keys list
    """
    return dict((k, v) for k, v in attrs.items() if k in keys)

def _screen(self, s, newline=False):
        """Print something on screen when self.verbose == True"""
        if self.verbose:
            if newline:
                print(s)
            else:
                print(s, end=' ')

def gen_text(env: TextIOBase, package: str, tmpl: str):
    """Create output from Jinja template."""
    if env:
        env_args = json_datetime.load(env)
    else:
        env_args = {}
    jinja_env = template.setup(package)
    echo(jinja_env.get_template(tmpl).render(**env_args))

def dot_v3(v, w):
    """Return the dotproduct of two vectors."""

    return sum([x * y for x, y in zip(v, w)])

def _get_url(url):
    """Retrieve requested URL"""
    try:
        data = HTTP_SESSION.get(url, stream=True)
        data.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise FetcherException(exc)

    return data

def _cumprod(l):
  """Cumulative product of a list.

  Args:
    l: a list of integers
  Returns:
    a list with one more element (starting with 1)
  """
  ret = [1]
  for item in l:
    ret.append(ret[-1] * item)
  return ret

def image_set_aspect(aspect=1.0, axes="gca"):
    """
    sets the aspect ratio of the current zoom level of the imshow image
    """
    if axes is "gca": axes = _pylab.gca()

    e = axes.get_images()[0].get_extent()
    axes.set_aspect(abs((e[1]-e[0])/(e[3]-e[2]))/aspect)

def string_input(prompt=''):
    """Python 3 input()/Python 2 raw_input()"""
    v = sys.version[0]
    if v == '3':
        return input(prompt)
    else:
        return raw_input(prompt)

def b(s):
	""" Encodes Unicode strings to byte strings, if necessary. """

	return s if isinstance(s, bytes) else s.encode(locale.getpreferredencoding())

def MessageToDict(message,
                  including_default_value_fields=False,
                  preserving_proto_field_name=False):
  """Converts protobuf message to a JSON dictionary.

  Args:
    message: The protocol buffers message instance to serialize.
    including_default_value_fields: If True, singular primitive fields,
        repeated fields, and map fields will always be serialized.  If
        False, only serialize non-empty fields.  Singular message fields
        and oneof fields are not affected by this option.
    preserving_proto_field_name: If True, use the original proto field
        names as defined in the .proto file. If False, convert the field
        names to lowerCamelCase.

  Returns:
    A dict representation of the JSON formatted protocol buffer message.
  """
  printer = _Printer(including_default_value_fields,
                     preserving_proto_field_name)
  # pylint: disable=protected-access
  return printer._MessageToJsonObject(message)

def setPixel(self, x, y, color):
        """Set the pixel at (x,y) to the integers in sequence 'color'."""
        return _fitz.Pixmap_setPixel(self, x, y, color)

def _GetFieldByName(message_descriptor, field_name):
  """Returns a field descriptor by field name.

  Args:
    message_descriptor: A Descriptor describing all fields in message.
    field_name: The name of the field to retrieve.
  Returns:
    The field descriptor associated with the field name.
  """
  try:
    return message_descriptor.fields_by_name[field_name]
  except KeyError:
    raise ValueError('Protocol message %s has no "%s" field.' %
                     (message_descriptor.name, field_name))

def _prepare_proxy(self, conn):
        """
        Establish tunnel connection early, because otherwise httplib
        would improperly set Host: header to proxy's IP:port.
        """
        conn.set_tunnel(self._proxy_host, self.port, self.proxy_headers)
        conn.connect()

def toJson(protoObject, indent=None):
    """
    Serialises a protobuf object as json
    """
    # Using the internal method because this way we can reformat the JSON
    js = json_format.MessageToDict(protoObject, False)
    return json.dumps(js, indent=indent)

def clear_matplotlib_ticks(self, axis="both"):
        """Clears the default matplotlib ticks."""
        ax = self.get_axes()
        plotting.clear_matplotlib_ticks(ax=ax, axis=axis)

def newest_file(file_iterable):
  """
  Returns the name of the newest file given an iterable of file names.

  """
  return max(file_iterable, key=lambda fname: os.path.getmtime(fname))

def update(self, **kwargs):
    """Creates or updates a property for the instance for each parameter."""
    for key, value in kwargs.items():
      setattr(self, key, value)

def _add_hash(source):
    """Add a leading hash '#' at the beginning of every line in the source."""
    source = '\n'.join('# ' + line.rstrip()
                       for line in source.splitlines())
    return source

def hline(self, x, y, width, color):
        """Draw a horizontal line up to a given length."""
        self.rect(x, y, width, 1, color, fill=True)

def make_unique_ngrams(s, n):
    """Make a set of unique n-grams from a string."""
    return set(s[i:i + n] for i in range(len(s) - n + 1))

def _repr(obj):
    """Show the received object as precise as possible."""
    vals = ", ".join("{}={!r}".format(
        name, getattr(obj, name)) for name in obj._attribs)
    if vals:
        t = "{}(name={}, {})".format(obj.__class__.__name__, obj.name, vals)
    else:
        t = "{}(name={})".format(obj.__class__.__name__, obj.name)
    return t

def objectproxy_realaddress(obj):
    """
    Obtain a real address as an integer from an objectproxy.
    """
    voidp = QROOT.TPython.ObjectProxy_AsVoidPtr(obj)
    return C.addressof(C.c_char.from_buffer(voidp))

def finish_plot():
    """Helper for plotting."""
    plt.legend()
    plt.grid(color='0.7')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def resources(self):
        """Retrieve contents of each page of PDF"""
        return [self.pdf.getPage(i) for i in range(self.pdf.getNumPages())]

def smallest_signed_angle(source, target):
    """Find the smallest angle going from angle `source` to angle `target`."""
    dth = target - source
    dth = (dth + np.pi) % (2.0 * np.pi) - np.pi
    return dth

def closeEvent(self, event):
        """closeEvent reimplementation"""
        if self.closing(True):
            event.accept()
        else:
            event.ignore()

def distance_to_line(a, b, p):
    """Closest distance between a line segment and a point

    Args:
        a ([float, float]): x and y coordinates. Line start
        b ([float, float]): x and y coordinates. Line end
        p ([float, float]): x and y coordinates. Point to compute the distance
    Returns:
        float
    """
    return distance(closest_point(a, b, p), p)

def dump_json(obj):
    """Dump Python object as JSON string."""
    return simplejson.dumps(obj, ignore_nan=True, default=json_util.default)

def sine_wave(frequency):
  """Emit a sine wave at the given frequency."""
  xs = tf.reshape(tf.range(_samples(), dtype=tf.float32), [1, _samples(), 1])
  ts = xs / FLAGS.sample_rate
  return tf.sin(2 * math.pi * frequency * ts)

def string_input(prompt=''):
    """Python 3 input()/Python 2 raw_input()"""
    v = sys.version[0]
    if v == '3':
        return input(prompt)
    else:
        return raw_input(prompt)

def full_s(self):
        """ Get the full singular value matrix of self

        Returns
        -------
        Matrix : Matrix

        """
        x = np.zeros((self.shape),dtype=np.float32)

        x[:self.s.shape[0],:self.s.shape[0]] = self.s.as_2d
        s = Matrix(x=x, row_names=self.row_names,
                          col_names=self.col_names, isdiagonal=False,
                          autoalign=False)
        return s

def bound_symbols(self):
        """Set of bound SymPy symbols contained within the equation."""
        try:
            lhs_syms = self.lhs.bound_symbols
        except AttributeError:
            lhs_syms = set()
        try:
            rhs_syms = self.rhs.bound_symbols
        except AttributeError:
            rhs_syms = set()
        return lhs_syms | rhs_syms

def get_dimension_array(array):
    """
    Get dimension of an array getting the number of rows and the max num of
    columns.
    """
    if all(isinstance(el, list) for el in array):
        result = [len(array), len(max([x for x in array], key=len,))]

    # elif array and isinstance(array, list):
    else:
        result = [len(array), 1]

    return result

def is_scalar(value):
    """Test if the given value is a scalar.

    This function also works with memory mapped array values, in contrast to the numpy is_scalar method.

    Args:
        value: the value to test for being a scalar value

    Returns:
        boolean: if the given value is a scalar or not
    """
    return np.isscalar(value) or (isinstance(value, np.ndarray) and (len(np.squeeze(value).shape) == 0))

def calculate_size(name, function):
    """ Calculates the request payload size"""
    data_size = 0
    data_size += calculate_size_str(name)
    data_size += calculate_size_data(function)
    return data_size

def _transform_triple_numpy(x):
    """Transform triple index into a 1-D numpy array."""
    return np.array([x.head, x.relation, x.tail], dtype=np.int64)

def iget_list_column_slice(list_, start=None, stop=None, stride=None):
    """ iterator version of get_list_column """
    if isinstance(start, slice):
        slice_ = start
    else:
        slice_ = slice(start, stop, stride)
    return (row[slice_] for row in list_)

def py3round(number):
    """Unified rounding in all python versions."""
    if abs(round(number) - number) == 0.5:
        return int(2.0 * round(number / 2.0))

    return int(round(number))

def MatrixSolve(a, rhs, adj):
    """
    Matrix solve op.
    """
    return np.linalg.solve(a if not adj else _adjoint(a), rhs),

def py3round(number):
    """Unified rounding in all python versions."""
    if abs(round(number) - number) == 0.5:
        return int(2.0 * round(number / 2.0))

    return int(round(number))

def naturalsortkey(s):
    """Natural sort order"""
    return [int(part) if part.isdigit() else part
            for part in re.split('([0-9]+)', s)]

def to_dicts(recarray):
    """convert record array to a dictionaries"""
    for rec in recarray:
        yield dict(zip(recarray.dtype.names, rec.tolist()))

def unique_list(lst):
    """Make a list unique, retaining order of initial appearance."""
    uniq = []
    for item in lst:
        if item not in uniq:
            uniq.append(item)
    return uniq

def create_rot2d(angle):
    """Create 2D rotation matrix"""
    ca = math.cos(angle)
    sa = math.sin(angle)
    return np.array([[ca, -sa], [sa, ca]])

def naturalsortkey(s):
    """Natural sort order"""
    return [int(part) if part.isdigit() else part
            for part in re.split('([0-9]+)', s)]

def naturalsortkey(s):
    """Natural sort order"""
    return [int(part) if part.isdigit() else part
            for part in re.split('([0-9]+)', s)]

def ub_to_str(string):
    """
    converts py2 unicode / py3 bytestring into str
    Args:
        string (unicode, byte_string): string to be converted
        
    Returns:
        (str)
    """
    if not isinstance(string, str):
        if six.PY2:
            return str(string)
        else:
            return string.decode()
    return string

def sort_by_name(self):
        """Sort list elements by name."""
        super(JSSObjectList, self).sort(key=lambda k: k.name)

def _histplot_bins(column, bins=100):
    """Helper to get bins for histplot."""
    col_min = np.min(column)
    col_max = np.max(column)
    return range(col_min, col_max + 2, max((col_max - col_min) // bins, 1))

def socket_close(self):
        """Close our socket."""
        if self.sock != NC.INVALID_SOCKET:
            self.sock.close()
        self.sock = NC.INVALID_SOCKET

def schunk(string, size):
    """Splits string into n sized chunks."""
    return [string[i:i+size] for i in range(0, len(string), size)]

def _dictfetchall(self, cursor):
        """ Return all rows from a cursor as a dict. """
        columns = [col[0] for col in cursor.description]
        return [
            dict(zip(columns, row))
            for row in cursor.fetchall()
        ]

def _split(string, splitters):
    """Splits a string into parts at multiple characters"""
    part = ''
    for character in string:
        if character in splitters:
            yield part
            part = ''
        else:
            part += character
    yield part

def guess_media_type(filepath):
    """Returns the media-type of the file at the given ``filepath``"""
    o = subprocess.check_output(['file', '--mime-type', '-Lb', filepath])
    o = o.strip()
    return o

def _split(string, splitters):
    """Splits a string into parts at multiple characters"""
    part = ''
    for character in string:
        if character in splitters:
            yield part
            part = ''
        else:
            part += character
    yield part

def mouse_get_pos():
    """

    :return:
    """
    p = POINT()
    AUTO_IT.AU3_MouseGetPos(ctypes.byref(p))
    return p.x, p.y

def cleanLines(source, lineSep=os.linesep):
    """
    :param source: some iterable source (list, file, etc)
    :param lineSep: string of separators (chars) that must be removed
    :return: list of non empty lines with removed separators
    """
    stripped = (line.strip(lineSep) for line in source)
    return (line for line in stripped if len(line) != 0)

def read_credentials(fname):
    """
    read a simple text file from a private location to get
    username and password
    """
    with open(fname, 'r') as f:
        username = f.readline().strip('\n')
        password = f.readline().strip('\n')
    return username, password

def _parse_string_to_list_of_pairs(s, seconds_to_int=False):
  r"""Parses a string into a list of pairs.

  In the input string, each pair is separated by a colon, and the delimiters
  between pairs are any of " ,.;".

  e.g. "rows:32,cols:32"

  Args:
    s: str to parse.
    seconds_to_int: Boolean. If True, then the second elements are returned
      as integers;  otherwise they are strings.

  Returns:
    List of tuple pairs.

  Raises:
    ValueError: Badly formatted string.
  """
  ret = []
  for p in [s.split(":") for s in re.sub("[,.;]", " ", s).split()]:
    if len(p) != 2:
      raise ValueError("bad input to _parse_string_to_list_of_pairs %s" % s)
    if seconds_to_int:
      ret.append((p[0], int(p[1])))
    else:
      ret.append(tuple(p))
  return ret

def _get_or_create_stack(name):
  """Returns a thread local stack uniquified by the given name."""
  stack = getattr(_LOCAL_STACKS, name, None)
  if stack is None:
    stack = []
    setattr(_LOCAL_STACKS, name, stack)
  return stack

def chunks(arr, size):
    """Splits a list into chunks

    :param arr: list to split
    :type arr: :class:`list`
    :param size: number of elements in each chunk
    :type size: :class:`int`
    :return: generator object
    :rtype: :class:`generator`
    """
    for i in _range(0, len(arr), size):
        yield arr[i:i+size]

def calling_logger(height=1):
    """ Obtain a logger for the calling module.

    Uses the inspect module to find the name of the calling function and its
    position in the module hierarchy. With the optional height argument, logs
    for caller's caller, and so forth.

    see: http://stackoverflow.com/a/900404/48251
    """
    stack = inspect.stack()
    height = min(len(stack) - 1, height)
    caller = stack[height]
    scope = caller[0].f_globals
    path = scope['__name__']
    if path == '__main__':
        path = scope['__package__'] or os.path.basename(sys.argv[0])
    return logging.getLogger(path)

def callproc(self, name, params, param_types=None):
    """Calls a procedure.

    :param name: the name of the procedure
    :param params: a list or tuple of parameters to pass to the procedure.
    :param param_types: a list or tuple of type names. If given, each param will be cast via
                        sql_writers typecast method. This is useful to disambiguate procedure calls
                        when several parameters are null and therefore cause overload resoluation
                        issues.
    :return: a 2-tuple of (cursor, params)
    """

    if param_types:
      placeholders = [self.sql_writer.typecast(self.sql_writer.to_placeholder(), t)
                      for t in param_types]
    else:
      placeholders = [self.sql_writer.to_placeholder() for p in params]

    # TODO: This may be Postgres specific...
    qs = "select * from {0}({1});".format(name, ", ".join(placeholders))
    return self.execute(qs, params), params

def force_iterable(f):
    """Will make any functions return an iterable objects by wrapping its result in a list."""
    def wrapper(*args, **kwargs):
        r = f(*args, **kwargs)
        if hasattr(r, '__iter__'):
            return r
        else:
            return [r]
    return wrapper

def wipe(self):
        """ Wipe the store
        """
        query = "DELETE FROM {}".format(self.__tablename__)
        connection = sqlite3.connect(self.sqlite_file)
        cursor = connection.cursor()
        cursor.execute(query)
        connection.commit()

async def async_input(prompt):
    """
    Python's ``input()`` is blocking, which means the event loop we set
    above can't be running while we're blocking there. This method will
    let the loop run while we wait for input.
    """
    print(prompt, end='', flush=True)
    return (await loop.run_in_executor(None, sys.stdin.readline)).rstrip()

def sqliteRowsToDicts(sqliteRows):
    """
    Unpacks sqlite rows as returned by fetchall
    into an array of simple dicts.

    :param sqliteRows: array of rows returned from fetchall DB call
    :return:  array of dicts, keyed by the column names.
    """
    return map(lambda r: dict(zip(r.keys(), r)), sqliteRows)

def loadb(b):
    """Deserialize ``b`` (instance of ``bytes``) to a Python object."""
    assert isinstance(b, (bytes, bytearray))
    return std_json.loads(b.decode('utf-8'))

def _heapify_max(x):
    """Transform list into a maxheap, in-place, in O(len(x)) time."""
    n = len(x)
    for i in reversed(range(n//2)):
        _siftup_max(x, i)

def readCommaList(fileList):
    """ Return a list of the files with the commas removed. """
    names=fileList.split(',')
    fileList=[]
    for item in names:
        fileList.append(item)
    return fileList

def dumps(obj):
    """Outputs json with formatting edits + object handling."""
    return json.dumps(obj, indent=4, sort_keys=True, cls=CustomEncoder)

def _repr_strip(mystring):
    """
    Returns the string without any initial or final quotes.
    """
    r = repr(mystring)
    if r.startswith("'") and r.endswith("'"):
        return r[1:-1]
    else:
        return r

def _heapify_max(x):
    """Transform list into a maxheap, in-place, in O(len(x)) time."""
    n = len(x)
    for i in reversed(range(n//2)):
        _siftup_max(x, i)

def surface(self, zdata, **kwargs):
        """Show a 3D surface plot.

        Extra keyword arguments are passed to `SurfacePlot()`.

        Parameters
        ----------
        zdata : array-like
            A 2D array of the surface Z values.

        """
        self._configure_3d()
        surf = scene.SurfacePlot(z=zdata, **kwargs)
        self.view.add(surf)
        self.view.camera.set_range()
        return surf

def Softsign(a):
    """
    Softsign op.
    """
    return np.divide(a, np.add(np.abs(a), 1)),

def _encode_gif(images, fps):
  """Encodes numpy images into gif string.

  Args:
    images: A 4-D `uint8` `np.array` (or a list of 3-D images) of shape
      `[time, height, width, channels]` where `channels` is 1 or 3.
    fps: frames per second of the animation

  Returns:
    The encoded gif string.

  Raises:
    IOError: If the ffmpeg command returns an error.
  """
  writer = WholeVideoWriter(fps)
  writer.write_multi(images)
  return writer.finish()

def timeout_thread_handler(timeout, stop_event):
    """A background thread to kill the process if it takes too long.

    Args:
        timeout (float): The number of seconds to wait before killing
            the process.
        stop_event (Event): An optional event to cleanly stop the background
            thread if required during testing.
    """

    stop_happened = stop_event.wait(timeout)
    if stop_happened is False:
        print("Killing program due to %f second timeout" % timeout)

    os._exit(2)

def main(idle):
    """Any normal python logic which runs a loop. Can take arguments."""
    while True:

        LOG.debug("Sleeping for {0} seconds.".format(idle))
        time.sleep(idle)

def stop_logging():
    """Stop logging to logfile and console."""
    from . import log
    logger = logging.getLogger("gromacs")
    logger.info("GromacsWrapper %s STOPPED logging", get_version())
    log.clear_handlers(logger)

def get_oauth_token():
    """Retrieve a simple OAuth Token for use with the local http client."""
    url = "{0}/token".format(DEFAULT_ORIGIN["Origin"])
    r = s.get(url=url)
    return r.json()["t"]

def stop(self, reason=None):
        """Shutdown the service with a reason."""
        self.logger.info('stopping')
        self.loop.stop(pyev.EVBREAK_ALL)

def update(packages, env=None, user=None):
    """
    Update conda packages in a conda env

    Attributes
    ----------
        packages: list of packages comma delimited
    """
    packages = ' '.join(packages.split(','))
    cmd = _create_conda_cmd('update', args=[packages, '--yes', '-q'], env=env, user=user)
    return _execcmd(cmd, user=user)

def _kw(keywords):
    """Turn list of keywords into dictionary."""
    r = {}
    for k, v in keywords:
        r[k] = v
    return r

def logv(msg, *args, **kwargs):
    """
    Print out a log message, only if verbose mode.
    """
    if settings.VERBOSE:
        log(msg, *args, **kwargs)

def join(mapping, bind, values):
    """ Merge all the strings. Put space between them. """
    return [' '.join([six.text_type(v) for v in values if v is not None])]

def debug(self, text):
		""" Ajout d'un message de log de type DEBUG """
		self.logger.debug("{}{}".format(self.message_prefix, text))

def _encode_bool(name, value, dummy0, dummy1):
    """Encode a python boolean (True/False)."""
    return b"\x08" + name + (value and b"\x01" or b"\x00")

def set_as_object(self, value):
        """
        Sets a new value to map element

        :param value: a new element or map value.
        """
        self.clear()
        map = MapConverter.to_map(value)
        self.append(map)

def text_width(string, font_name, font_size):
    """Determine with width in pixels of string."""
    return stringWidth(string, fontName=font_name, fontSize=font_size)

def activate(self):
        """Store ipython references in the __builtin__ namespace."""

        add_builtin = self.add_builtin
        for name, func in self.auto_builtins.iteritems():
            add_builtin(name, func)

def __repr__(self):
        """Return list-lookalike of representation string of objects"""
        strings = []
        for currItem in self:
            strings.append("%s" % currItem)
        return "(%s)" % (", ".join(strings))

def add_to_enum(self, clsdict):
        """
        Compile XML mappings in addition to base add behavior.
        """
        super(XmlMappedEnumMember, self).add_to_enum(clsdict)
        self.register_xml_mapping(clsdict)

def get_category(self):
        """Get the category of the item.

        :return: the category of the item.
        :returntype: `unicode`"""
        var = self.xmlnode.prop("category")
        if not var:
            var = "?"
        return var.decode("utf-8")

def get_adjacent_matrix(self):
        """Get adjacency matrix.

        Returns:
            :param adj: adjacency matrix
            :type adj: np.ndarray
        """
        edges = self.edges
        num_edges = len(edges) + 1
        adj = np.zeros([num_edges, num_edges])

        for k in range(num_edges - 1):
            adj[edges[k].L, edges[k].R] = 1
            adj[edges[k].R, edges[k].L] = 1

        return adj

def energy_string_to_float( string ):
    """
    Convert a string of a calculation energy, e.g. '-1.2345 eV' to a float.

    Args:
        string (str): The string to convert.
  
    Return
        (float) 
    """
    energy_re = re.compile( "(-?\d+\.\d+)" )
    return float( energy_re.match( string ).group(0) )

def as_dict(self):
        """Package up the public attributes as a dict."""
        attrs = vars(self)
        return {key: attrs[key] for key in attrs if not key.startswith('_')}

def do_striptags(value):
    """Strip SGML/XML tags and replace adjacent whitespace by one space.
    """
    if hasattr(value, '__html__'):
        value = value.__html__()
    return Markup(unicode(value)).striptags()

def dictfetchall(cursor):
    """Returns all rows from a cursor as a dict (rather than a headerless table)

    From Django Documentation: https://docs.djangoproject.com/en/dev/topics/db/sql/
    """
    desc = cursor.description
    return [dict(zip([col[0] for col in desc], row)) for row in cursor.fetchall()]

def text_cleanup(data, key, last_type):
    """ I strip extra whitespace off multi-line strings if they are ready to be stripped!"""
    if key in data and last_type == STRING_TYPE:
        data[key] = data[key].strip()
    return data

def isbinary(*args):
    """Checks if value can be part of binary/bitwise operations."""
    return all(map(lambda c: isnumber(c) or isbool(c), args))

def __run_spark_submit(lane_yaml, dist_dir, spark_home, spark_args, silent):
    """
    Submits the packaged application to spark using a `spark-submit` subprocess

    Parameters
    ----------
    lane_yaml (str): Path to the YAML lane definition file
    dist_dir (str): Path to the directory where the packaged code is located
    spark_args (str): String of any additional spark config args to be passed when submitting
    silent (bool): Flag indicating whether job output should be printed to console
    """
    # spark-submit binary
    cmd = ['spark-submit' if spark_home is None else os.path.join(spark_home, 'bin/spark-submit')]

    # Supplied spark arguments
    if spark_args:
        cmd += spark_args

    # Packaged App & lane
    cmd += ['--py-files', 'libs.zip,_framework.zip,tasks.zip', 'main.py']
    cmd += ['--lane', lane_yaml]

    logging.info('Submitting to Spark')
    logging.debug(str(cmd))

    # Submit
    devnull = open(os.devnull, 'w')
    outp = {'stderr': STDOUT, 'stdout': devnull} if silent else {}
    call(cmd, cwd=dist_dir, env=MY_ENV, **outp)
    devnull.close()

def angle(x, y):
    """Return the angle between vectors a and b in degrees."""
    return arccos(dot(x, y)/(norm(x)*norm(y)))*180./pi

def set_title(self, title, **kwargs):
        """Sets the title on the underlying matplotlib AxesSubplot."""
        ax = self.get_axes()
        ax.set_title(title, **kwargs)

def write_login(collector, image, **kwargs):
    """Login to a docker registry with write permissions"""
    docker_api = collector.configuration["harpoon"].docker_api
    collector.configuration["authentication"].login(docker_api, image, is_pushing=True, global_docker=True)

def correspond(text):
    """Communicate with the child process without closing stdin."""
    subproc.stdin.write(text)
    subproc.stdin.flush()
    return drain()

def dictapply(d, fn):
    """
    apply a function to all non-dict values in a dictionary
    """
    for k, v in d.items():
        if isinstance(v, dict):
            v = dictapply(v, fn)
        else:
            d[k] = fn(v)
    return d

def filter_dict(d, keys):
    """
    Creates a new dict from an existing dict that only has the given keys
    """
    return {k: v for k, v in d.items() if k in keys}

def dictapply(d, fn):
    """
    apply a function to all non-dict values in a dictionary
    """
    for k, v in d.items():
        if isinstance(v, dict):
            v = dictapply(v, fn)
        else:
            d[k] = fn(v)
    return d

def _varargs_to_iterable_method(func):
    """decorator to convert a *args method to one taking a iterable"""
    def wrapped(self, iterable, **kwargs):
        return func(self, *iterable, **kwargs)
    return wrapped

def redirect(cls, request, response):
        """Redirect to the canonical URI for this resource."""
        if cls.meta.legacy_redirect:
            if request.method in ('GET', 'HEAD',):
                # A SAFE request is allowed to redirect using a 301
                response.status = http.client.MOVED_PERMANENTLY

            else:
                # All other requests must use a 307
                response.status = http.client.TEMPORARY_REDIRECT

        else:
            # Modern redirects are allowed. Let's have some fun.
            # Hopefully you're client supports this.
            # The RFC explicitly discourages UserAgent sniffing.
            response.status = http.client.PERMANENT_REDIRECT

        # Terminate the connection.
        response.close()

def _replace_variables(data, variables):
    """Replace the format variables in all items of data."""
    formatter = string.Formatter()
    return [formatter.vformat(item, [], variables) for item in data]

def format_exc(*exc_info):
    """Show exception with traceback."""
    typ, exc, tb = exc_info or sys.exc_info()
    error = traceback.format_exception(typ, exc, tb)
    return "".join(error)

def to_list(self):
        """Convert this confusion matrix into a 2x2 plain list of values."""
        return [[int(self.table.cell_values[0][1]), int(self.table.cell_values[0][2])],
                [int(self.table.cell_values[1][1]), int(self.table.cell_values[1][2])]]

def string_to_identity(identity_str):
    """Parse string into Identity dictionary."""
    m = _identity_regexp.match(identity_str)
    result = m.groupdict()
    log.debug('parsed identity: %s', result)
    return {k: v for k, v in result.items() if v}

def get_tri_area(pts):
    """
    Given a list of coords for 3 points,
    Compute the area of this triangle.

    Args:
        pts: [a, b, c] three points
    """
    a, b, c = pts[0], pts[1], pts[2]
    v1 = np.array(b) - np.array(a)
    v2 = np.array(c) - np.array(a)
    area_tri = abs(sp.linalg.norm(sp.cross(v1, v2)) / 2)
    return area_tri

def datetime_match(data, dts):
    """
    matching of datetimes in time columns for data filtering
    """
    dts = dts if islistable(dts) else [dts]
    if any([not isinstance(i, datetime.datetime) for i in dts]):
        error_msg = (
            "`time` can only be filtered by datetimes"
        )
        raise TypeError(error_msg)
    return data.isin(dts)

def register_action(action):
  """
  Adds an action to the parser cli.

  :param action(BaseAction): a subclass of the BaseAction class
  """
  sub = _subparsers.add_parser(action.meta('cmd'), help=action.meta('help'))
  sub.set_defaults(cmd=action.meta('cmd'))
  for (name, arg) in action.props().items():
    sub.add_argument(arg.name, arg.flag, **arg.options)
    _actions[action.meta('cmd')] = action

def _uniquify(_list):
    """Remove duplicates in a list."""
    seen = set()
    result = []
    for x in _list:
        if x not in seen:
            result.append(x)
            seen.add(x)
    return result

def parse_command_args():
    """Command line parser."""
    parser = argparse.ArgumentParser(description='Register PB devices.')
    parser.add_argument('num_pb', type=int,
                        help='Number of PBs devices to register.')
    return parser.parse_args()

def tuple_check(*args, func=None):
    """Check if arguments are tuple type."""
    func = func or inspect.stack()[2][3]
    for var in args:
        if not isinstance(var, (tuple, collections.abc.Sequence)):
            name = type(var).__name__
            raise TupleError(
                f'Function {func} expected tuple, {name} got instead.')

def kwargs_to_string(kwargs):
    """
    Given a set of kwargs, turns them into a string which can then be passed to a command.
    :param kwargs: kwargs from a function call.
    :return: outstr: A string, which is '' if no kwargs were given, and the kwargs in string format otherwise.
    """
    outstr = ''
    for arg in kwargs:
        outstr += ' -{} {}'.format(arg, kwargs[arg])
    return outstr

def process_wait(process, timeout=0):
    """
    Pauses script execution until a given process exists.
    :param process:
    :param timeout:
    :return:
    """
    ret = AUTO_IT.AU3_ProcessWait(LPCWSTR(process), INT(timeout))
    return ret

def _most_common(iterable):
    """Returns the most common element in `iterable`."""
    data = Counter(iterable)
    return max(data, key=data.__getitem__)

def unit_ball_L2(shape):
  """A tensorflow variable tranfomed to be constrained in a L2 unit ball.

  EXPERIMENTAL: Do not use for adverserial examples if you need to be confident
  they are strong attacks. We are not yet confident in this code.
  """
  x = tf.Variable(tf.zeros(shape))
  return constrain_L2(x)

def expect_all(a, b):
    """\
    Asserts that two iterables contain the same values.
    """
    assert all(_a == _b for _a, _b in zip_longest(a, b))

def transformer_tall_pretrain_lm_tpu_adafactor():
  """Hparams for transformer on LM pretraining (with 64k vocab) on TPU."""
  hparams = transformer_tall_pretrain_lm()
  update_hparams_for_tpu(hparams)
  hparams.max_length = 1024
  # For multi-problem on TPU we need it in absolute examples.
  hparams.batch_size = 8
  hparams.multiproblem_vocab_size = 2**16
  return hparams

def _assert_is_type(name, value, value_type):
    """Assert that a value must be a given type."""
    if not isinstance(value, value_type):
        if type(value_type) is tuple:
            types = ', '.join(t.__name__ for t in value_type)
            raise ValueError('{0} must be one of ({1})'.format(name, types))
        else:
            raise ValueError('{0} must be {1}'
                             .format(name, value_type.__name__))

def is_timestamp(instance):
    """Validates data is a timestamp"""
    if not isinstance(instance, (int, str)):
        return True
    return datetime.fromtimestamp(int(instance))

def from_array(cls, arr):
        """Convert a structured NumPy array into a Table."""
        return cls().with_columns([(f, arr[f]) for f in arr.dtype.names])

def is_lazy_iterable(obj):
    """
    Returns whether *obj* is iterable lazily, such as generators, range objects, etc.
    """
    return isinstance(obj,
        (types.GeneratorType, collections.MappingView, six.moves.range, enumerate))

def connection_lost(self, exc):
        """Called when asyncio.Protocol loses the network connection."""
        if exc is None:
            self.log.warning('eof from receiver?')
        else:
            self.log.warning('Lost connection to receiver: %s', exc)

        self.transport = None

        if self._connection_lost_callback:
            self._loop.call_soon(self._connection_lost_callback)

def center_text(text, width=80):
    """Center all lines of the text.

    It is assumed that all lines width is smaller then B{width}, because the
    line width will not be checked.

    Args:
        text (str): Text to wrap.
        width (int): Maximum number of characters per line.

    Returns:
        str: Centered text.
    """
    centered = []
    for line in text.splitlines():
        centered.append(line.center(width))
    return "\n".join(centered)

def connection_lost(self, exc):
        """Called when asyncio.Protocol loses the network connection."""
        if exc is None:
            self.log.warning('eof from receiver?')
        else:
            self.log.warning('Lost connection to receiver: %s', exc)

        self.transport = None

        if self._connection_lost_callback:
            self._loop.call_soon(self._connection_lost_callback)

def average(iterator):
    """Iterative mean."""
    count = 0
    total = 0
    for num in iterator:
        count += 1
        total += num
    return float(total)/count

def StringIO(*args, **kwargs):
    """StringIO constructor shim for the async wrapper."""
    raw = sync_io.StringIO(*args, **kwargs)
    return AsyncStringIOWrapper(raw)

def convert_timezone(obj, timezone):
    """Convert `obj` to the timezone `timezone`.

    Parameters
    ----------
    obj : datetime.date or datetime.datetime

    Returns
    -------
    type(obj)
    """
    if timezone is None:
        return obj.replace(tzinfo=None)
    return pytz.timezone(timezone).localize(obj)

async def write(self, data):
        """
        :py:func:`asyncio.coroutine`

        :py:meth:`aioftp.StreamIO.write` proxy
        """
        await self.wait("write")
        start = _now()
        await super().write(data)
        self.append("write", data, start)

def yview(self, *args):
        """Update inplace widgets position when doing vertical scroll"""
        self.after_idle(self.__updateWnds)
        ttk.Treeview.yview(self, *args)

def debug_src(src, pm=False, globs=None):
    """Debug a single doctest docstring, in argument `src`'"""
    testsrc = script_from_examples(src)
    debug_script(testsrc, pm, globs)

def _set_scroll_v(self, *args):
        """Scroll both categories Canvas and scrolling container"""
        self._canvas_categories.yview(*args)
        self._canvas_scroll.yview(*args)

def Distance(lat1, lon1, lat2, lon2):
    """Get distance between pairs of lat-lon points"""

    az12, az21, dist = wgs84_geod.inv(lon1, lat1, lon2, lat2)
    return az21, dist

def codebox(msg="", title=" ", text=""):
    """
    Display some text in a monospaced font, with no line wrapping.
    This function is suitable for displaying code and text that is
    formatted using spaces.

    The text parameter should be a string, or a list or tuple of lines to be
    displayed in the textbox.

    :param str msg: the msg to be displayed
    :param str title: the window title
    :param str text: what to display in the textbox
    """
    return tb.textbox(msg, title, text, codebox=1)

def pp_xml(body):
    """Pretty print format some XML so it's readable."""
    pretty = xml.dom.minidom.parseString(body)
    return pretty.toprettyxml(indent="  ")

def isetdiff_flags(list1, list2):
    """
    move to util_iter
    """
    set2 = set(list2)
    return (item not in set2 for item in list1)

def csv_to_dicts(file, header=None):
    """Reads a csv and returns a List of Dicts with keys given by header row."""
    with open(file) as csvfile:
        return [row for row in csv.DictReader(csvfile, fieldnames=header)]

def printComparison(results, class_or_prop):
	"""
	print(out the results of the comparison using a nice table)
	"""

	data = []

	Row = namedtuple('Row',[class_or_prop,'VALIDATED'])

	for k,v in sorted(results.items(), key=lambda x: x[1]):
		data += [Row(k, str(v))]

	pprinttable(data)

def rgb2gray(img):
    """Converts an RGB image to grayscale using matlab's algorithm."""
    T = np.linalg.inv(np.array([
        [1.0,  0.956,  0.621],
        [1.0, -0.272, -0.647],
        [1.0, -1.106,  1.703],
    ]))
    r_c, g_c, b_c = T[0]
    r, g, b = np.rollaxis(as_float_image(img), axis=-1)
    return r_c * r + g_c * g + b_c * b

def xmltreefromfile(filename):
    """Internal function to read an XML file"""
    try:
        return ElementTree.parse(filename, ElementTree.XMLParser(collect_ids=False))
    except TypeError:
        return ElementTree.parse(filename, ElementTree.XMLParser())

def val_to_bin(edges, x):
    """Convert axis coordinate to bin index."""
    ibin = np.digitize(np.array(x, ndmin=1), edges) - 1
    return ibin

def split_long_sentence(sentence, words_per_line):
    """Takes a sentence and adds a newline every "words_per_line" words.

    Parameters
    ----------
    sentence: str
        Sentene to split
    words_per_line: double
        Add a newline every this many words
    """
    words = sentence.split(' ')
    split_sentence = ''
    for i in range(len(words)):
        split_sentence = split_sentence + words[i]
        if (i+1) % words_per_line == 0:
            split_sentence = split_sentence + '\n'
        elif i != len(words) - 1:
            split_sentence = split_sentence + " "
    return split_sentence

def hook_focus_events(self):
        """ Install the hooks for focus events.

        This method may be overridden by subclasses as needed.

        """
        widget = self.widget
        widget.focusInEvent = self.focusInEvent
        widget.focusOutEvent = self.focusOutEvent

def mouse_move_event(self, event):
        """
        Forward mouse cursor position events to the example
        """
        self.example.mouse_position_event(event.x(), event.y())

def to_dicts(recarray):
    """convert record array to a dictionaries"""
    for rec in recarray:
        yield dict(zip(recarray.dtype.names, rec.tolist()))

def needs_update(self, cache_key):
    """Check if the given cached item is invalid.

    :param cache_key: A CacheKey object (as returned by CacheKeyGenerator.key_for().
    :returns: True if the cached version of the item is out of date.
    """
    if not self.cacheable(cache_key):
      # An uncacheable CacheKey is always out of date.
      return True

    return self._read_sha(cache_key) != cache_key.hash

def _trim(self, somestr):
        """ Trim left-right given string """
        tmp = RE_LSPACES.sub("", somestr)
        tmp = RE_TSPACES.sub("", tmp)
        return str(tmp)

def remove_file_from_s3(awsclient, bucket, key):
    """Remove a file from an AWS S3 bucket.

    :param awsclient:
    :param bucket:
    :param key:
    :return:
    """
    client_s3 = awsclient.get_client('s3')
    response = client_s3.delete_object(Bucket=bucket, Key=key)

def update(self, params):
        """Update the dev_info data from a dictionary.

        Only updates if it already exists in the device.
        """
        dev_info = self.json_state.get('deviceInfo')
        dev_info.update({k: params[k] for k in params if dev_info.get(k)})

def be_array_from_bytes(fmt, data):
    """
    Reads an array from bytestring with big-endian data.
    """
    arr = array.array(str(fmt), data)
    return fix_byteorder(arr)

async def packets_from_tshark(self, packet_callback, packet_count=None, close_tshark=True):
        """
        A coroutine which creates a tshark process, runs the given callback on each packet that is received from it and
        closes the process when it is done.

        Do not use interactively. Can be used in order to insert packets into your own eventloop.
        """
        tshark_process = await self._get_tshark_process(packet_count=packet_count)
        try:
            await self._go_through_packets_from_fd(tshark_process.stdout, packet_callback, packet_count=packet_count)
        except StopCapture:
            pass
        finally:
            if close_tshark:
                await self._close_async()

def _from_bytes(bytes, byteorder="big", signed=False):
    """This is the same functionality as ``int.from_bytes`` in python 3"""
    return int.from_bytes(bytes, byteorder=byteorder, signed=signed)

def _to_array(value):
    """As a convenience, turn Python lists and tuples into NumPy arrays."""
    if isinstance(value, (tuple, list)):
        return array(value)
    elif isinstance(value, (float, int)):
        return np.float64(value)
    else:
        return value

def cast_bytes(s, encoding=None):
    """Source: https://github.com/ipython/ipython_genutils"""
    if not isinstance(s, bytes):
        return encode(s, encoding)
    return s

def to_binary(s, encoding='utf8'):
    """Portable cast function.

    In python 2 the ``str`` function which is used to coerce objects to bytes does not
    accept an encoding argument, whereas python 3's ``bytes`` function requires one.

    :param s: object to be converted to binary_type
    :return: binary_type instance, representing s.
    """
    if PY3:  # pragma: no cover
        return s if isinstance(s, binary_type) else binary_type(s, encoding=encoding)
    return binary_type(s)

def bin_to_int(string):
    """Convert a one element byte string to signed int for python 2 support."""
    if isinstance(string, str):
        return struct.unpack("b", string)[0]
    else:
        return struct.unpack("b", bytes([string]))[0]

def s2b(s):
    """
    String to binary.
    """
    ret = []
    for c in s:
        ret.append(bin(ord(c))[2:].zfill(8))
    return "".join(ret)

def _bytes_to_json(value):
    """Coerce 'value' to an JSON-compatible representation."""
    if isinstance(value, bytes):
        value = base64.standard_b64encode(value).decode("ascii")
    return value

def time2seconds(t):
    """Returns seconds since 0h00."""
    return t.hour * 3600 + t.minute * 60 + t.second + float(t.microsecond) / 1e6

def check_clang_apply_replacements_binary(args):
  """Checks if invoking supplied clang-apply-replacements binary works."""
  try:
    subprocess.check_call([args.clang_apply_replacements_binary, '--version'])
  except:
    print('Unable to run clang-apply-replacements. Is clang-apply-replacements '
          'binary correctly specified?', file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

def runiform(lower, upper, size=None):
    """
    Random uniform variates.
    """
    return np.random.uniform(lower, upper, size)

def getCachedDataKey(engineVersionHash, key):
		"""
		Retrieves the cached data value for the specified engine version hash and dictionary key
		"""
		cacheFile = CachedDataManager._cacheFileForHash(engineVersionHash)
		return JsonDataManager(cacheFile).getKey(key)

def uniq(seq):
    """ Return a copy of seq without duplicates. """
    seen = set()
    return [x for x in seq if str(x) not in seen and not seen.add(str(x))]

def Distance(lat1, lon1, lat2, lon2):
    """Get distance between pairs of lat-lon points"""

    az12, az21, dist = wgs84_geod.inv(lon1, lat1, lon2, lat2)
    return az21, dist

def get_labels(labels):
    """Create unique labels."""
    label_u = unique_labels(labels)
    label_u_line = [i + "_line" for i in label_u]
    return label_u, label_u_line

def get_tri_area(pts):
    """
    Given a list of coords for 3 points,
    Compute the area of this triangle.

    Args:
        pts: [a, b, c] three points
    """
    a, b, c = pts[0], pts[1], pts[2]
    v1 = np.array(b) - np.array(a)
    v2 = np.array(c) - np.array(a)
    area_tri = abs(sp.linalg.norm(sp.cross(v1, v2)) / 2)
    return area_tri

def remove_duplicates(seq):
    """
    Return unique elements from list while preserving order.
    From https://stackoverflow.com/a/480227/2589328
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def var(series):
    """
    Returns the variance of values in a series.

    Args:
        series (pandas.Series): column to summarize.
    """
    if np.issubdtype(series.dtype, np.number):
        return series.var()
    else:
        return np.nan

def uniquify_list(L):
    """Same order unique list using only a list compression."""
    return [e for i, e in enumerate(L) if L.index(e) == i]

def sequence_molecular_weight(seq):
    """Returns the molecular weight of the polypeptide sequence.

    Notes
    -----
    Units = Daltons

    Parameters
    ----------
    seq : str
        Sequence of amino acids.
    """
    if 'X' in seq:
        warnings.warn(_nc_warning_str, NoncanonicalWarning)
    return sum(
        [residue_mwt[aa] * n for aa, n in Counter(seq).items()]) + water_mass

def assert_is_not(expected, actual, message=None, extra=None):
    """Raises an AssertionError if expected is actual."""
    assert expected is not actual, _assert_fail_message(
        message, expected, actual, "is", extra
    )

def do(self):
        """
        Set a restore point (copy the object), then call the method.
        :return: obj.do_method(*args)
        """
        self.restore_point = self.obj.copy()
        return self.do_method(self.obj, *self.args)

def __unixify(self, s):
        """ stupid windows. converts the backslash to forwardslash for consistency """
        return os.path.normpath(s).replace(os.sep, "/")

def python(string: str):
        """
            :param string: String can be type, resource or python case
        """
        return underscore(singularize(string) if Naming._pluralize(string) else string)

def update(self):
        """Update all visuals in the attached canvas."""
        if not self.canvas:
            return
        for visual in self.canvas.visuals:
            self.update_program(visual.program)
        self.canvas.update()

def min_or_none(val1, val2):
    """Returns min(val1, val2) returning None only if both values are None"""
    return min(val1, val2, key=lambda x: sys.maxint if x is None else x)

def read_credentials(fname):
    """
    read a simple text file from a private location to get
    username and password
    """
    with open(fname, 'r') as f:
        username = f.readline().strip('\n')
        password = f.readline().strip('\n')
    return username, password

def _parallel_compare_helper(class_obj, pairs, x, x_link=None):
    """Internal function to overcome pickling problem in python2."""
    return class_obj._compute(pairs, x, x_link)

def transformer_tall_pretrain_lm_tpu_adafactor():
  """Hparams for transformer on LM pretraining (with 64k vocab) on TPU."""
  hparams = transformer_tall_pretrain_lm()
  update_hparams_for_tpu(hparams)
  hparams.max_length = 1024
  # For multi-problem on TPU we need it in absolute examples.
  hparams.batch_size = 8
  hparams.multiproblem_vocab_size = 2**16
  return hparams

def out_shape_from_array(arr):
    """Get the output shape from an array."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.shape
    else:
        return (arr.shape[1],)

def isnumber(*args):
    """Checks if value is an integer, long integer or float.

    NOTE: Treats booleans as numbers, where True=1 and False=0.
    """
    return all(map(lambda c: isinstance(c, int) or isinstance(c, float), args))

def clear(self):
        """Clear the displayed image."""
        self._imgobj = None
        try:
            # See if there is an image on the canvas
            self.canvas.delete_object_by_tag(self._canvas_img_tag)
            self.redraw()
        except KeyError:
            pass

def send(socket, data, num_bytes=20):
    """Send data to specified socket.


    :param socket: open socket instance
    :param data: data to send
    :param num_bytes: number of bytes to read

    :return: received data
    """
    pickled_data = pickle.dumps(data, -1)
    length = str(len(pickled_data)).zfill(num_bytes)
    socket.sendall(length.encode())
    socket.sendall(pickled_data)

def decamelise(text):
    """Convert CamelCase to lower_and_underscore."""
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()

def _request_modify_dns_record(self, record):
        """Sends Modify_DNS_Record request"""
        return self._request_internal("Modify_DNS_Record",
                                      domain=self.domain,
                                      record=record)

def is_equal_strings_ignore_case(first, second):
    """The function compares strings ignoring case"""
    if first and second:
        return first.upper() == second.upper()
    else:
        return not (first or second)

def hclust_linearize(U):
    """Sorts the rows of a matrix by hierarchical clustering.

    Parameters:
        U (ndarray) : matrix of data

    Returns:
        prm (ndarray) : permutation of the rows
    """

    from scipy.cluster import hierarchy
    Z = hierarchy.ward(U)
    return hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z, U))

def convert_array(array):
    """
    Converts an ARRAY string stored in the database back into a Numpy array.

    Parameters
    ----------
    array: ARRAY
        The array object to be converted back into a Numpy array.

    Returns
    -------
    array
            The converted Numpy array.

    """
    out = io.BytesIO(array)
    out.seek(0)
    return np.load(out)

def ask_str(question: str, default: str = None):
    """Asks for a simple string"""
    default_q = " [default: {0}]: ".format(
        default) if default is not None else ""
    answer = input("{0} [{1}]: ".format(question, default_q))

    if answer == "":
        return default
    return answer

def col_rename(df,col_name,new_col_name):
    """ Changes a column name in a DataFrame
    Parameters:
    df - DataFrame
        DataFrame to operate on
    col_name - string
        Name of column to change
    new_col_name - string
        New name of column
    """
    col_list = list(df.columns)
    for index,value in enumerate(col_list):
        if value == col_name:
            col_list[index] = new_col_name
            break
    df.columns = col_list

def get_ctype(rtype, cfunc, *args):
    """ Call a C function that takes a pointer as its last argument and
        return the C object that it contains after the function has finished.

    :param rtype:   C data type is filled by the function
    :param cfunc:   C function to call
    :param args:    Arguments to call function with
    :return:        A pointer to the specified data type
    """
    val_p = backend.ffi.new(rtype)
    args = args + (val_p,)
    cfunc(*args)
    return val_p[0]

def convert_str_to_datetime(df, *, column: str, format: str):
    """
    Convert string column into datetime column

    ---

    ### Parameters

    *mandatory :*
    - `column` (*str*): name of the column to format
    - `format` (*str*): current format of the values (see [available formats](
    https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior))
    """
    df[column] = pd.to_datetime(df[column], format=format)
    return df

def _openResources(self):
        """ Uses numpy.load to open the underlying file
        """
        arr = np.load(self._fileName, allow_pickle=ALLOW_PICKLE)
        check_is_an_array(arr)
        self._array = arr

def _to_numeric(val):
    """
    Helper function for conversion of various data types into numeric representation.
    """
    if isinstance(val, (int, float, datetime.datetime, datetime.timedelta)):
        return val
    return float(val)

def do_exit(self, arg):
        """Exit the shell session."""

        if self.current:
            self.current.close()
        self.resource_manager.close()
        del self.resource_manager
        return True

def get_year_start(day=None):
    """Returns January 1 of the given year."""
    day = add_timezone(day or datetime.date.today())
    return day.replace(month=1).replace(day=1)

def print_matrix(X, decimals=1):
    """Pretty printing for numpy matrix X"""
    for row in np.round(X, decimals=decimals):
        print(row)

def closing_plugin(self, cancelable=False):
        """Perform actions before parent main window is closed"""
        self.dialog_manager.close_all()
        self.shell.exit_interpreter()
        return True

def CleanseComments(line):
  """Removes //-comments and single-line C-style /* */ comments.

  Args:
    line: A line of C++ source.

  Returns:
    The line with single-line comments removed.
  """
  commentpos = line.find('//')
  if commentpos != -1 and not IsCppString(line[:commentpos]):
    line = line[:commentpos].rstrip()
  # get rid of /* ... */
  return _RE_PATTERN_CLEANSE_LINE_C_COMMENTS.sub('', line)

def set_cursor(self, x, y):
        """
        Sets the cursor to the desired position.

        :param x: X position
        :param y: Y position
        """
        curses.curs_set(1)
        self.screen.move(y, x)

def _remove_duplicate_files(xs):
    """Remove files specified multiple times in a list.
    """
    seen = set([])
    out = []
    for x in xs:
        if x["path"] not in seen:
            out.append(x)
            seen.add(x["path"])
    return out

def process_result_value(self, value, dialect):
        """convert value from json to a python object"""
        if value is not None:
            value = simplejson.loads(value)
        return value

def multi_replace(instr, search_list=[], repl_list=None):
    """
    Does a string replace with a list of search and replacements

    TODO: rename
    """
    repl_list = [''] * len(search_list) if repl_list is None else repl_list
    for ser, repl in zip(search_list, repl_list):
        instr = instr.replace(ser, repl)
    return instr

def _zeep_to_dict(cls, obj):
        """Convert a zeep object to a dictionary."""
        res = serialize_object(obj)
        res = cls._get_non_empty_dict(res)
        return res

def test(*args):
    """
    Run unit tests.
    """
    subprocess.call(["py.test-2.7"] + list(args))
    subprocess.call(["py.test-3.4"] + list(args))

def log_y_cb(self, w, val):
        """Toggle linear/log scale for Y-axis."""
        self.tab_plot.logy = val
        self.plot_two_columns()

def main(source):
    """
    For a given command line supplied argument, negotiate the content, parse
    the schema and then return any issues to stdout or if no schema issues,
    return success exit code.
    """
    if source is None:
        click.echo(
            "You need to supply a file or url to a schema to a swagger schema, for"
            "the validator to work."
        )
        return 1
    try:
        load(source)
        click.echo("Validation passed")
        return 0
    except ValidationError as e:
        raise click.ClickException(str(e))

def hasattrs(object, *names):
    """
    Takes in an object and a variable length amount of named attributes,
    and checks to see if the object has each property. If any of the
    attributes are missing, this returns false.

    :param object: an object that may or may not contain the listed attributes
    :param names: a variable amount of attribute names to check for
    :return: True if the object contains each named attribute, false otherwise
    """
    for name in names:
        if not hasattr(object, name):
            return False
    return True

def set_float(val):
    """ utility to set a floating value,
    useful for converting from strings """
    out = None
    if not val in (None, ''):
        try:
            out = float(val)
        except ValueError:
            return None
        if numpy.isnan(out):
            out = default
    return out

def is_timestamp(obj):
    """
    Yaml either have automatically converted it to a datetime object
    or it is a string that will be validated later.
    """
    return isinstance(obj, datetime.datetime) or is_string(obj) or is_int(obj) or is_float(obj)

def last_day(year=_year, month=_month):
    """
    get the current month's last day
    :param year:  default to current year
    :param month:  default to current month
    :return: month's last day
    """
    last_day = calendar.monthrange(year, month)[1]
    return datetime.date(year=year, month=month, day=last_day)

def check_for_positional_argument(kwargs, name, default=False):
    """
    @type kwargs: dict
    @type name: str
    @type default: bool, int, str
    @return: bool, int
    """
    if name in kwargs:
        if str(kwargs[name]) == "True":
            return True
        elif str(kwargs[name]) == "False":
            return False
        else:
            return kwargs[name]

    return default

def _validate_simple(email):
        """Does a simple validation of an email by matching it to a regexps

        :param email: The email to check
        :return: The valid Email address

        :raises: ValueError if value is not a valid email
        """
        name, address = parseaddr(email)
        if not re.match('[^@]+@[^@]+\.[^@]+', address):
            raise ValueError('Invalid email :{email}'.format(email=email))
        return address

def __contains__ (self, key):
        """Check lowercase key item."""
        assert isinstance(key, basestring)
        return dict.__contains__(self, key.lower())

def is_image_file_valid(file_path_name):
    """
    Indicate whether the specified image file is valid or not.


    @param file_path_name: absolute path and file name of an image.


    @return: ``True`` if the image file is valid, ``False`` if the file is
        truncated or does not correspond to a supported image.
    """
    # Image.verify is only implemented for PNG images, and it only verifies
    # the CRC checksum in the image.  The only way to check from within
    # Pillow is to load the image in a try/except and check the error.  If
    # as much info as possible is from the image is needed,
    # ``ImageFile.LOAD_TRUNCATED_IMAGES=True`` needs to bet set and it
    # will attempt to parse as much as possible.
    try:
        with Image.open(file_path_name) as image:
            image.load()
    except IOError:
        return False

    return True

def is_file_exists_error(e):
    """
    Returns whether the exception *e* was raised due to an already existing file or directory.
    """
    if six.PY3:
        return isinstance(e, FileExistsError)  # noqa: F821
    else:
        return isinstance(e, OSError) and e.errno == 17

def do_last(environment, seq):
    """Return the last item of a sequence."""
    try:
        return next(iter(reversed(seq)))
    except StopIteration:
        return environment.undefined('No last item, sequence was empty.')

def stdout_display():
    """ Print results straight to stdout """
    if sys.version_info[0] == 2:
        yield SmartBuffer(sys.stdout)
    else:
        yield SmartBuffer(sys.stdout.buffer)

def has_field(mc, field_name):
    """
    detect if a model has a given field has

    :param field_name:
    :param mc:
    :return:
    """
    try:
        mc._meta.get_field(field_name)
    except FieldDoesNotExist:
        return False
    return True

def readwav(filename):
    """Read a WAV file and returns the data and sample rate

    ::

        from spectrum.io import readwav
        readwav()

    """
    from scipy.io.wavfile import read as readwav
    samplerate, signal = readwav(filename)
    return signal, samplerate

def _rm_name_match(s1, s2):
  """
  determine whether two sequence names from a repeatmasker alignment match.

  :return: True if they are the same string, or if one forms a substring of the
           other, else False
  """
  m_len = min(len(s1), len(s2))
  return s1[:m_len] == s2[:m_len]

def inside_softimage():
    """Returns a boolean indicating if the code is executed inside softimage."""
    try:
        import maya
        return False
    except ImportError:
        pass
    try:
        from win32com.client import Dispatch as disp
        disp('XSI.Application')
        return True
    except:
        return False

def _name_exists(self, name):
        """
        Checks if we already have an opened tab with the same name.
        """
        for i in range(self.count()):
            if self.tabText(i) == name:
                return True
        return False

def set_input_value(self, selector, value):
        """Set the value of the input matched by given selector."""
        script = 'document.querySelector("%s").setAttribute("value", "%s")'
        script = script % (selector, value)
        self.evaluate(script)

def can_access(self, user):
        """Return whether or not `user` can access a project.

        The project's is_ready field must be set for a user to access.

        """
        return self.class_.is_admin(user) or \
            self.is_ready and self.class_ in user.classes

def state(self):
        """Return internal state, useful for testing."""
        return {'c': self.c, 's0': self.s0, 's1': self.s1, 's2': self.s2}

def _check_elements_equal(lst):
    """
    Returns true if all of the elements in the list are equal.
    """
    assert isinstance(lst, list), "Input value must be a list."
    return not lst or lst.count(lst[0]) == len(lst)

def get_single_item(d):
    """Get an item from a dict which contains just one item."""
    assert len(d) == 1, 'Single-item dict must have just one item, not %d.' % len(d)
    return next(six.iteritems(d))

def all_strings(arr):
        """
        Ensures that the argument is a list that either is empty or contains only strings
        :param arr: list
        :return:
        """
        if not isinstance([], list):
            raise TypeError("non-list value found where list is expected")
        return all(isinstance(x, str) for x in arr)

def mpl_outside_legend(ax, **kwargs):
    """ Places a legend box outside a matplotlib Axes instance. """
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), **kwargs)

def _IsDirectory(parent, item):
  """Helper that returns if parent/item is a directory."""
  return tf.io.gfile.isdir(os.path.join(parent, item))

def cli_command_quit(self, msg):
        """\
        kills the child and exits
        """
        if self.state == State.RUNNING and self.sprocess and self.sprocess.proc:
            self.sprocess.proc.kill()
        else:
            sys.exit(0)

def contains_geometric_info(var):
    """ Check whether the passed variable is a tuple with two floats or integers """
    return isinstance(var, tuple) and len(var) == 2 and all(isinstance(val, (int, float)) for val in var)

def np_counts(self):
    """Dictionary of noun phrase frequencies in this text.
    """
    counts = defaultdict(int)
    for phrase in self.noun_phrases:
        counts[phrase] += 1
    return counts

def has_edge(self, p_from, p_to):
        """ Returns True when the graph has the given edge. """
        return p_from in self._edges and p_to in self._edges[p_from]

def fft_freqs(n_fft, fs):
    """Return frequencies for DFT

    Parameters
    ----------
    n_fft : int
        Number of points in the FFT.
    fs : float
        The sampling rate.
    """
    return np.arange(0, (n_fft // 2 + 1)) / float(n_fft) * float(fs)

def watched_extension(extension):
    """Return True if the given extension is one of the watched extensions"""
    for ext in hamlpy.VALID_EXTENSIONS:
        if extension.endswith('.' + ext):
            return True
    return False

def rpc_fix_code(self, source, directory):
        """Formats Python code to conform to the PEP 8 style guide.

        """
        source = get_source(source)
        return fix_code(source, directory)

def check_dependency(self, dependency_path):
        """Check if mtime of dependency_path is greater than stored mtime."""
        stored_hash = self._stamp_file_hashes.get(dependency_path)

        # This file was newly added, or we don't have a file
        # with stored hashes yet. Assume out of date.
        if not stored_hash:
            return False

        return stored_hash == _sha1_for_file(dependency_path)

def is_valid(number):
    """determines whether the card number is valid."""
    n = str(number)
    if not n.isdigit():
        return False
    return int(n[-1]) == get_check_digit(n[:-1])

def contained_in(filename, directory):
    """Test if a file is located within the given directory."""
    filename = os.path.normcase(os.path.abspath(filename))
    directory = os.path.normcase(os.path.abspath(directory))
    return os.path.commonprefix([filename, directory]) == directory

def flatten(l):
    """Flatten a nested list."""
    return sum(map(flatten, l), []) \
        if isinstance(l, list) or isinstance(l, tuple) else [l]

def _pip_exists(self):
        """Returns True if pip exists inside the virtual environment. Can be
        used as a naive way to verify that the environment is installed."""
        return os.path.isfile(os.path.join(self.path, 'bin', 'pip'))

def get_table(ports):
    """
    This function returns a pretty table used to display the port results.

    :param ports: list of found ports
    :return: the table to display
    """
    table = PrettyTable(["Name", "Port", "Protocol", "Description"])
    table.align["Name"] = "l"
    table.align["Description"] = "l"
    table.padding_width = 1

    for port in ports:
        table.add_row(port)

    return table

def is_iterable_of_int(l):
    r""" Checks if l is iterable and contains only integral types """
    if not is_iterable(l):
        return False

    return all(is_int(value) for value in l)

def _startswith(expr, pat):
    """
    Return boolean sequence or scalar indicating whether each string in the sequence or scalar
    starts with passed pattern. Equivalent to str.startswith().

    :param expr:
    :param pat: Character sequence
    :return: sequence or scalar
    """

    return _string_op(expr, Startswith, output_type=types.boolean, _pat=pat)

def is_delimiter(line):
    """ True if a line consists only of a single punctuation character."""
    return bool(line) and line[0] in punctuation and line[0]*len(line) == line

def csv_matrix_print(classes, table):
    """
    Return matrix as csv data.

    :param classes: classes list
    :type classes:list
    :param table: table
    :type table:dict
    :return:
    """
    result = ""
    classes.sort()
    for i in classes:
        for j in classes:
            result += str(table[i][j]) + ","
        result = result[:-1] + "\n"
    return result[:-1]

def isin(value, values):
    """ Check that value is in values """
    for i, v in enumerate(value):
        if v not in np.array(values)[:, i]:
            return False
    return True

def install_rpm_py():
    """Install RPM Python binding."""
    python_path = sys.executable
    cmd = '{0} install.py'.format(python_path)
    exit_status = os.system(cmd)
    if exit_status != 0:
        raise Exception('Command failed: {0}'.format(cmd))

def all_strings(arr):
        """
        Ensures that the argument is a list that either is empty or contains only strings
        :param arr: list
        :return:
        """
        if not isinstance([], list):
            raise TypeError("non-list value found where list is expected")
        return all(isinstance(x, str) for x in arr)

def we_are_in_lyon():
    """Check if we are on a Lyon machine"""
    import socket
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        return False
    return ip.startswith("134.158.")

def _write_json(file, contents):
    """Write a dict to a JSON file."""
    with open(file, 'w') as f:
        return json.dump(contents, f, indent=2, sort_keys=True)

def is_empty(self):
    """Returns True if this node has no children, or if all of its children are ParseNode instances
    and are empty.
    """
    return all(isinstance(c, ParseNode) and c.is_empty for c in self.children)

def _serialize_json(obj, fp):
    """ Serialize ``obj`` as a JSON formatted stream to ``fp`` """
    json.dump(obj, fp, indent=4, default=serialize)

def is_archlinux():
    """return True if the current distribution is running on debian like OS."""
    if platform.system().lower() == 'linux':
        if platform.linux_distribution() == ('', '', ''):
            # undefined distribution. Fixed in python 3.
            if os.path.exists('/etc/arch-release'):
                return True
    return False

def is_password_valid(password):
    """
    Check if a password is valid
    """
    pattern = re.compile(r"^.{4,75}$")
    return bool(pattern.match(password))

def are_in_interval(s, l, r, border = 'included'):
        """
        Checks whether all number in the sequence s lie inside the interval formed by
        l and r.
        """
        return numpy.all([IntensityRangeStandardization.is_in_interval(x, l, r, border) for x in s])

def on_close(self, evt):
    """
    Pop-up menu and wx.EVT_CLOSE closing event
    """
    self.stop() # DoseWatcher
    if evt.EventObject is not self: # Avoid deadlocks
      self.Close() # wx.Frame
    evt.Skip()

def intty(cls):
        """ Check if we are in a tty. """
        # XXX: temporary hack until we can detect if we are in a pipe or not
        return True

        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            return True

        return False

def list_view_changed(self, widget, event, data=None):
        """
        Function shows last rows.
        """
        adj = self.scrolled_window.get_vadjustment()
        adj.set_value(adj.get_upper() - adj.get_page_size())

def has_permission(user, permission_name):
    """Check if a user has a given permission."""
    if user and user.is_superuser:
        return True

    return permission_name in available_perm_names(user)

def check_precomputed_distance_matrix(X):
    """Perform check_array(X) after removing infinite values (numpy.inf) from the given distance matrix.
    """
    tmp = X.copy()
    tmp[np.isinf(tmp)] = 1
    check_array(tmp)

def iter_finds(regex_obj, s):
    """Generate all matches found within a string for a regex and yield each match as a string"""
    if isinstance(regex_obj, str):
        for m in re.finditer(regex_obj, s):
            yield m.group()
    else:
        for m in regex_obj.finditer(s):
            yield m.group()

def is_valid_row(cls, row):
        """Indicates whether or not the given row contains valid data."""
        for k in row.keys():
            if row[k] is None:
                return False
        return True

def user_in_all_groups(user, groups):
    """Returns True if the given user is in all given groups"""
    return user_is_superuser(user) or all(user_in_group(user, group) for group in groups)

def is_natural(x):
    """A non-negative integer."""
    try:
        is_integer = int(x) == x
    except (TypeError, ValueError):
        return False
    return is_integer and x >= 0

def string(value) -> str:
        """ string dict/object/value to JSON """
        return system_json.dumps(Json(value).safe_object(), ensure_ascii=False)

def get_period_last_3_months() -> str:
    """ Returns the last week as a period string """
    today = Datum()
    today.today()

    # start_date = today - timedelta(weeks=13)
    start_date = today.clone()
    start_date.subtract_months(3)

    period = get_period(start_date.date, today.date)
    return period

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

def previous_workday(dt):
    """
    returns previous weekday used for observances
    """
    dt -= timedelta(days=1)
    while dt.weekday() > 4:
        # Mon-Fri are 0-4
        dt -= timedelta(days=1)
    return dt

def __gt__(self, other):
        """Test for greater than."""
        if isinstance(other, Address):
            return str(self) > str(other)
        raise TypeError

def batch_split_sentences(self, texts: List[str]) -> List[List[str]]:
        """
        This method lets you take advantage of spacy's batch processing.
        Default implementation is to just iterate over the texts and call ``split_sentences``.
        """
        return [self.split_sentences(text) for text in texts]

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

def get_column_names(engine: Engine, tablename: str) -> List[str]:
    """
    Get all the database column names for the specified table.
    """
    return [info.name for info in gen_columns_info(engine, tablename)]

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

def get_timezone() -> Tuple[datetime.tzinfo, str]:
    """Discover the current time zone and it's standard string representation (for source{d})."""
    dt = get_datetime_now().astimezone()
    tzstr = dt.strftime("%z")
    tzstr = tzstr[:-2] + ":" + tzstr[-2:]
    return dt.tzinfo, tzstr

def inverted_dict(d):
    """Return a dict with swapped keys and values

    >>> inverted_dict({0: ('a', 'b'), 1: 'cd'}) == {'cd': 1, ('a', 'b'): 0}
    True
    """
    return dict((force_hashable(v), k) for (k, v) in viewitems(dict(d)))

def read_text_from_file(path: str) -> str:
    """ Reads text file contents """
    with open(path) as text_file:
        content = text_file.read()

    return content

def full(self):
        """Return ``True`` if the queue is full, ``False``
        otherwise (not reliable!).

        Only applicable if :attr:`maxsize` is set.

        """
        return self.maxsize and len(self.list) >= self.maxsize or False

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

def remove_empty_text(utterances: List[Utterance]) -> List[Utterance]:
    """Remove empty utterances from a list of utterances
    Args:
        utterances: The list of utterance we are processing
    """
    return [utter for utter in utterances if utter.text.strip() != ""]

def get_pij_matrix(t, diag, A, A_inv):
    """
    Calculates the probability matrix of substitutions i->j over time t,
    given the normalised generator diagonalisation.


    :param t: time
    :type t: float
    :return: probability matrix
    :rtype: numpy.ndarray
    """
    return A.dot(np.diag(np.exp(diag * t))).dot(A_inv)

def flush(self):
        """
        Flush all unwritten data to disk.
        """
        if self._cache_modified_count > 0:
            self.storage.write(self.cache)
            self._cache_modified_count = 0

def is_natural(x):
    """A non-negative integer."""
    try:
        is_integer = int(x) == x
    except (TypeError, ValueError):
        return False
    return is_integer and x >= 0

def _sum_cycles_from_tokens(self, tokens: List[str]) -> int:
        """Sum the total number of cycles over a list of tokens."""
        return sum((int(self._nonnumber_pattern.sub('', t)) for t in tokens))

def __next__(self):
        """
        :return: int
        """
        self.current += 1
        if self.current > self.total:
            raise StopIteration
        else:
            return self.iterable[self.current - 1]

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

def from_file(file_path) -> dict:
        """ Load JSON file """
        with io.open(file_path, 'r', encoding='utf-8') as json_stream:
            return Json.parse(json_stream, True)

def tail(filename, number_of_bytes):
    """Returns the last number_of_bytes of filename"""
    with open(filename, "rb") as f:
        if os.stat(filename).st_size > number_of_bytes:
            f.seek(-number_of_bytes, 2)
        return f.read()

async def executemany(self, sql: str, parameters: Iterable[Iterable[Any]]) -> None:
        """Execute the given multiquery."""
        await self._execute(self._cursor.executemany, sql, parameters)

def get_timezone() -> Tuple[datetime.tzinfo, str]:
    """Discover the current time zone and it's standard string representation (for source{d})."""
    dt = get_datetime_now().astimezone()
    tzstr = dt.strftime("%z")
    tzstr = tzstr[:-2] + ":" + tzstr[-2:]
    return dt.tzinfo, tzstr

def proper_round(n):
    """
    rounds float to closest int
    :rtype: int
    :param n: float
    """
    return int(n) + (n / abs(n)) * int(abs(n - int(n)) >= 0.5) if n != 0 else 0

def is_integer(value: Any) -> bool:
    """Return true if a value is an integer number."""
    return (isinstance(value, int) and not isinstance(value, bool)) or (
        isinstance(value, float) and isfinite(value) and int(value) == value
    )

def _rindex(mylist: Sequence[T], x: T) -> int:
    """Index of the last occurrence of x in the sequence."""
    return len(mylist) - mylist[::-1].index(x) - 1

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

def split(text: str) -> List[str]:
    """Split a text into a list of tokens.

    :param text: the text to split
    :return: tokens
    """
    return [word for word in SEPARATOR.split(text) if word.strip(' \t')]

def clean(ctx, text):
    """
    Removes all non-printable characters from a text string
    """
    text = conversions.to_string(text, ctx)
    return ''.join([c for c in text if ord(c) >= 32])

def indices_to_labels(self, indices: Sequence[int]) -> List[str]:
        """ Converts a sequence of indices into their corresponding labels."""

        return [(self.INDEX_TO_LABEL[index]) for index in indices]

def has_key(cls, *args):
        """
        Check whether flyweight object with specified key has already been created.

        Returns:
            bool: True if already created, False if not
        """
        key = args if len(args) > 1 else args[0]
        return key in cls._instances

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

def _skip_section(self):
        """Skip a section"""
        self._last = self._f.readline()
        while len(self._last) > 0 and len(self._last[0].strip()) == 0:
            self._last = self._f.readline()

async def cursor(self) -> Cursor:
        """Create an aiosqlite cursor wrapping a sqlite3 cursor object."""
        return Cursor(self, await self._execute(self._conn.cursor))

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

def mmap(func, iterable):
    """Wrapper to make map() behave the same on Py2 and Py3."""

    if sys.version_info[0] > 2:
        return [i for i in map(func, iterable)]
    else:
        return map(func, iterable)

def extend(a: dict, b: dict) -> dict:
    """Merge two dicts and return a new dict. Much like subclassing works."""
    res = a.copy()
    res.update(b)
    return res

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

def iter_lines(file_like: Iterable[str]) -> Generator[str, None, None]:
    """ Helper for iterating only nonempty lines without line breaks"""
    for line in file_like:
        line = line.rstrip('\r\n')
        if line:
            yield line

def strtobytes(input, encoding):
    """Take a str and transform it into a byte array."""
    py_version = sys.version_info[0]
    if py_version >= 3:
        return _strtobytes_py3(input, encoding)
    return _strtobytes_py2(input, encoding)

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

def indexes_equal(a: Index, b: Index) -> bool:
    """
    Are two indexes equal? Checks by comparing ``str()`` versions of them.
    (AM UNSURE IF THIS IS ENOUGH.)
    """
    return str(a) == str(b)

def read(self, count=0):
        """ Read """
        return self.f.read(count) if count > 0 else self.f.read()

def bfx(value, msb, lsb):
    """! @brief Extract a value from a bitfield."""
    mask = bitmask((msb, lsb))
    return (value & mask) >> lsb

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

def lowercase_chars(string: any) -> str:
        """Return all (and only) the lowercase chars in the given string."""
        return ''.join([c if c.islower() else '' for c in str(string)])

def is_unicode(string):
    """Validates that the object itself is some kinda string"""
    str_type = str(type(string))

    if str_type.find('str') > 0 or str_type.find('unicode') > 0:
        return True

    return False

def assert_valid_name(name: str) -> str:
    """Uphold the spec rules about naming."""
    error = is_valid_name_error(name)
    if error:
        raise error
    return name

def datetime_iso_format(date):
    """
    Return an ISO-8601 representation of a datetime object.
    """
    return "{0:0>4}-{1:0>2}-{2:0>2}T{3:0>2}:{4:0>2}:{5:0>2}Z".format(
        date.year, date.month, date.day, date.hour,
        date.minute, date.second)

def is_unicode(string):
    """Validates that the object itself is some kinda string"""
    str_type = str(type(string))

    if str_type.find('str') > 0 or str_type.find('unicode') > 0:
        return True

    return False

def gen_lower(x: Iterable[str]) -> Generator[str, None, None]:
    """
    Args:
        x: iterable of strings

    Yields:
        each string in lower case
    """
    for string in x:
        yield string.lower()

def last_commit(self) -> Tuple:
        """Returns a tuple (hash, and commit object)"""
        from libs.repos import git

        return git.get_last_commit(repo_path=self.path)

def indexes_equal(a: Index, b: Index) -> bool:
    """
    Are two indexes equal? Checks by comparing ``str()`` versions of them.
    (AM UNSURE IF THIS IS ENOUGH.)
    """
    return str(a) == str(b)

def dictlist_wipe_key(dict_list: Iterable[Dict], key: str) -> None:
    """
    Process an iterable of dictionaries. For each dictionary ``d``, delete
    ``d[key]`` if it exists.
    """
    for d in dict_list:
        d.pop(key, None)

def lower_camel_case_from_underscores(string):
    """generate a lower-cased camelCase string from an underscore_string.
    For example: my_variable_name -> myVariableName"""
    components = string.split('_')
    string = components[0]
    for component in components[1:]:
        string += component[0].upper() + component[1:]
    return string

def has_synset(word: str) -> list:
    """" Returns a list of synsets of a word after lemmatization. """

    return wn.synsets(lemmatize(word, neverstem=True))

def is_sqlatype_string(coltype: Union[TypeEngine, VisitableType]) -> bool:
    """
    Is the SQLAlchemy column type a string type?
    """
    coltype = _coltype_to_typeengine(coltype)
    return isinstance(coltype, sqltypes.String)

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

def dotproduct(X, Y):
    """Return the sum of the element-wise product of vectors x and y.
    >>> dotproduct([1, 2, 3], [1000, 100, 10])
    1230
    """
    return sum([x * y for x, y in zip(X, Y)])

def cli_run():
    """docstring for argparse"""
    parser = argparse.ArgumentParser(description='Stupidly simple code answers from StackOverflow')
    parser.add_argument('query', help="What's the problem ?", type=str, nargs='+')
    parser.add_argument('-t','--tags', help='semicolon separated tags -> python;lambda')
    args = parser.parse_args()
    main(args)

def chars(string: any) -> str:
        """Return all (and only) the chars in the given string."""
        return ''.join([c if c.isalpha() else '' for c in str(string)])

def SetCursorPos(x: int, y: int) -> bool:
    """
    SetCursorPos from Win32.
    Set mouse cursor to point x, y.
    x: int.
    y: int.
    Return bool, True if succeed otherwise False.
    """
    return bool(ctypes.windll.user32.SetCursorPos(x, y))

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

def inverted_dict(d):
    """Return a dict with swapped keys and values

    >>> inverted_dict({0: ('a', 'b'), 1: 'cd'}) == {'cd': 1, ('a', 'b'): 0}
    True
    """
    return dict((force_hashable(v), k) for (k, v) in viewitems(dict(d)))

def is_natural(x):
    """A non-negative integer."""
    try:
        is_integer = int(x) == x
    except (TypeError, ValueError):
        return False
    return is_integer and x >= 0

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

def str_to_time(time_str: str) -> datetime.datetime:
    """
    Convert human readable string to datetime.datetime.
    """
    pieces: Any = [int(piece) for piece in time_str.split('-')]
    return datetime.datetime(*pieces)

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

def has_key(cls, *args):
        """
        Check whether flyweight object with specified key has already been created.

        Returns:
            bool: True if already created, False if not
        """
        key = args if len(args) > 1 else args[0]
        return key in cls._instances

def snake_to_camel(s: str) -> str:
    """Convert string from snake case to camel case."""

    fragments = s.split('_')

    return fragments[0] + ''.join(x.title() for x in fragments[1:])

def _latex_format(obj: Any) -> str:
    """Format an object as a latex string."""
    if isinstance(obj, float):
        try:
            return sympy.latex(symbolize(obj))
        except ValueError:
            return "{0:.4g}".format(obj)

    return str(obj)

def is_empty_shape(sh: ShExJ.Shape) -> bool:
        """ Determine whether sh has any value """
        return sh.closed is None and sh.expression is None and sh.extra is None and \
            sh.semActs is None

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

def indexes_equal(a: Index, b: Index) -> bool:
    """
    Are two indexes equal? Checks by comparing ``str()`` versions of them.
    (AM UNSURE IF THIS IS ENOUGH.)
    """
    return str(a) == str(b)

def extend(a: dict, b: dict) -> dict:
    """Merge two dicts and return a new dict. Much like subclassing works."""
    res = a.copy()
    res.update(b)
    return res

def pmon(month):
	"""
	P the month

	>>> print(pmon('2012-08'))
	August, 2012
	"""
	year, month = month.split('-')
	return '{month_name}, {year}'.format(
		month_name=calendar.month_name[int(month)],
		year=year,
	)

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

def _lower(string):
    """Custom lower string function.
    Examples:
        FooBar -> foo_bar
    """
    if not string:
        return ""

    new_string = [string[0].lower()]
    for char in string[1:]:
        if char.isupper():
            new_string.append("_")
        new_string.append(char.lower())

    return "".join(new_string)

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

def clean_column_names(df: DataFrame) -> DataFrame:
    """
    Strip the whitespace from all column names in the given DataFrame
    and return the result.
    """
    f = df.copy()
    f.columns = [col.strip() for col in f.columns]
    return f

def area (self):
    """area() -> number

    Returns the area of this Polygon.
    """
    area = 0.0
    
    for segment in self.segments():
      area += ((segment.p.x * segment.q.y) - (segment.q.x * segment.p.y))/2

    return area

def get_day_name(self) -> str:
        """ Returns the day name """
        weekday = self.value.isoweekday() - 1
        return calendar.day_name[weekday]

def _duplicates(list_):
    """Return dict mapping item -> indices."""
    item_indices = {}
    for i, item in enumerate(list_):
        try:
            item_indices[item].append(i)
        except KeyError:  # First time seen
            item_indices[item] = [i]
    return item_indices

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

def is_any_type_set(sett: Set[Type]) -> bool:
    """
    Helper method to check if a set of types is the {AnyObject} singleton

    :param sett:
    :return:
    """
    return len(sett) == 1 and is_any_type(min(sett))

def unzoom_all(self, event=None):
        """ zoom out full data range """
        if len(self.conf.zoom_lims) > 0:
            self.conf.zoom_lims = [self.conf.zoom_lims[0]]
        self.unzoom(event)

def write_text(filename: str, text: str) -> None:
    """
    Writes text to a file.
    """
    with open(filename, 'w') as f:  # type: TextIO
        print(text, file=f)

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

def right_replace(string, old, new, count=1):
    """
    Right replaces ``count`` occurrences of ``old`` with ``new`` in ``string``.
    For example::

        right_replace('one_two_two', 'two', 'three') -> 'one_two_three'
    """
    if not string:
        return string
    return new.join(string.rsplit(old, count))

def unzoom_all(self, event=None):
        """ zoom out full data range """
        if len(self.conf.zoom_lims) > 0:
            self.conf.zoom_lims = [self.conf.zoom_lims[0]]
        self.unzoom(event)

def debugTreePrint(node,pfx="->"):
  """Purely a debugging aid: Ascii-art picture of a tree descended from node"""
  print pfx,node.item
  for c in node.children:
    debugTreePrint(c,"  "+pfx)

def exclude(self, *args, **kwargs) -> "QuerySet":
        """
        Same as .filter(), but with appends all args with NOT
        """
        return self._filter_or_exclude(negate=True, *args, **kwargs)

def camel_to_snake(s: str) -> str:
    """Convert string from camel case to snake case."""

    return CAMEL_CASE_RE.sub(r'_\1', s).strip().lower()

def is_sqlatype_integer(coltype: Union[TypeEngine, VisitableType]) -> bool:
    """
    Is the SQLAlchemy column type an integer type?
    """
    coltype = _coltype_to_typeengine(coltype)
    return isinstance(coltype, sqltypes.Integer)

def get_timezone() -> Tuple[datetime.tzinfo, str]:
    """Discover the current time zone and it's standard string representation (for source{d})."""
    dt = get_datetime_now().astimezone()
    tzstr = dt.strftime("%z")
    tzstr = tzstr[:-2] + ":" + tzstr[-2:]
    return dt.tzinfo, tzstr

def zfill(x, width):
    """zfill(x, width) -> string

    Pad a numeric string x with zeros on the left, to fill a field
    of the specified width.  The string x is never truncated.

    """
    if not isinstance(x, basestring):
        x = repr(x)
    return x.zfill(width)

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

def stretch(iterable, n=2):
    r"""Repeat each item in `iterable` `n` times.

    Example:

    >>> list(stretch(range(3), 2))
    [0, 0, 1, 1, 2, 2]
    """
    times = range(n)
    for item in iterable:
        for i in times: yield item

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

def __remove_trailing_zeros(self, collection):
        """Removes trailing zeroes from indexable collection of numbers"""
        index = len(collection) - 1
        while index >= 0 and collection[index] == 0:
            index -= 1

        return collection[:index + 1]

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

def get_prop_value(name, props, default=None):
    # type: (str, Dict[str, Any], Any) -> Any
    """
    Returns the value of a property or the default one

    :param name: Name of a property
    :param props: Dictionary of properties
    :param default: Default value
    :return: The value of the property or the default one
    """
    if not props:
        return default

    try:
        return props[name]
    except KeyError:
        return default

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

def create_pie_chart(self, snapshot, filename=''):
        """
        Create a pie chart that depicts the distribution of the allocated memory
        for a given `snapshot`. The chart is saved to `filename`.
        """
        try:
            from pylab import figure, title, pie, axes, savefig
            from pylab import sum as pylab_sum
        except ImportError:
            return self.nopylab_msg % ("pie_chart")

        # Don't bother illustrating a pie without pieces.
        if not snapshot.tracked_total:
            return ''

        classlist = []
        sizelist = []
        for k, v in list(snapshot.classes.items()):
            if v['pct'] > 3.0:
                classlist.append(k)
                sizelist.append(v['sum'])
        sizelist.insert(0, snapshot.asizeof_total - pylab_sum(sizelist))
        classlist.insert(0, 'Other')
        #sizelist = [x*0.01 for x in sizelist]

        title("Snapshot (%s) Memory Distribution" % (snapshot.desc))
        figure(figsize=(8,8))
        axes([0.1, 0.1, 0.8, 0.8])
        pie(sizelist, labels=classlist)
        savefig(filename, dpi=50)

        return self.chart_tag % (self.relative_path(filename))

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

def _(f, x):
    """
    filter for dict, note `f` should have signature: `f::key->value->bool`
    """
    return {k: v for k, v in x.items() if f(k, v)}

def samefile(a: str, b: str) -> bool:
    """Check if two pathes represent the same file."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normpath(a) == os.path.normpath(b)

def from_file(file_path) -> dict:
        """ Load JSON file """
        with io.open(file_path, 'r', encoding='utf-8') as json_stream:
            return Json.parse(json_stream, True)

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

def do_quit(self, _: argparse.Namespace) -> bool:
        """Exit this application"""
        self._should_quit = True
        return self._STOP_AND_EXIT

def astensor(array: TensorLike) -> BKTensor:
    """Covert numpy array to tensorflow tensor"""
    tensor = tf.convert_to_tensor(value=array, dtype=CTYPE)
    return tensor

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

def array2string(arr: numpy.ndarray) -> str:
        """Format numpy array as a string."""
        shape = str(arr.shape)[1:-1]
        if shape.endswith(","):
            shape = shape[:-1]
        return numpy.array2string(arr, threshold=11) + "%s[%s]" % (arr.dtype, shape)

def cli_run():
    """docstring for argparse"""
    parser = argparse.ArgumentParser(description='Stupidly simple code answers from StackOverflow')
    parser.add_argument('query', help="What's the problem ?", type=str, nargs='+')
    parser.add_argument('-t','--tags', help='semicolon separated tags -> python;lambda')
    args = parser.parse_args()
    main(args)

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

def mouse_event(dwFlags: int, dx: int, dy: int, dwData: int, dwExtraInfo: int) -> None:
    """mouse_event from Win32."""
    ctypes.windll.user32.mouse_event(dwFlags, dx, dy, dwData, dwExtraInfo)

def array2string(arr: numpy.ndarray) -> str:
        """Format numpy array as a string."""
        shape = str(arr.shape)[1:-1]
        if shape.endswith(","):
            shape = shape[:-1]
        return numpy.array2string(arr, threshold=11) + "%s[%s]" % (arr.dtype, shape)

def _izip(*iterables):
    """ Iterate through multiple lists or arrays of equal size """
    # This izip routine is from itertools
    # izip('ABCD', 'xy') --> Ax By

    iterators = map(iter, iterables)
    while iterators:
        yield tuple(map(next, iterators))

def extend(a: dict, b: dict) -> dict:
    """Merge two dicts and return a new dict. Much like subclassing works."""
    res = a.copy()
    res.update(b)
    return res

def s3_get(url: str, temp_file: IO) -> None:
    """Pull a file directly from S3."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)

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

async def executemany(self, sql: str, parameters: Iterable[Iterable[Any]]) -> None:
        """Execute the given multiquery."""
        await self._execute(self._cursor.executemany, sql, parameters)

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

def release_lock():
    """Release lock on compilation directory."""
    get_lock.n_lock -= 1
    assert get_lock.n_lock >= 0
    # Only really release lock once all lock requests have ended.
    if get_lock.lock_is_enabled and get_lock.n_lock == 0:
        get_lock.start_time = None
        get_lock.unlocker.unlock()

def uppercase_chars(string: any) -> str:
        """Return all (and only) the uppercase chars in the given string."""
        return ''.join([c if c.isupper() else '' for c in str(string)])

def read(self, start_position: int, size: int) -> memoryview:
        """
        Return a view into the memory
        """
        return memoryview(self._bytes)[start_position:start_position + size]

def min(self):
        """
        :returns the minimum of the column
        """
        res = self._qexec("min(%s)" % self._name)
        if len(res) > 0:
            self._min = res[0][0]
        return self._min

def most_significant_bit(lst: np.ndarray) -> int:
    """
    A helper function that finds the position of the most significant bit in a 1darray of 1s and 0s,
    i.e. the first position where a 1 appears, reading left to right.

    :param lst: a 1d array of 0s and 1s with at least one 1
    :return: the first position in lst that a 1 appears
    """
    return np.argwhere(np.asarray(lst) == 1)[0][0]

def append_num_column(self, text: str, index: int):
        """ Add value to the output row, width based on index """
        width = self.columns[index]["width"]
        return f"{text:>{width}}"

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

def _kbhit_unix() -> bool:
    """
    Under UNIX: is a keystroke available?
    """
    dr, dw, de = select.select([sys.stdin], [], [], 0)
    return dr != []

def _gaussian_function(self, datalength: int, values: np.ndarray,
                           height: int, index: int) -> np.ndarray:
        """
        i'th Regression Model Gaussian

        :param: len(x)
        :param: x values
        :param: height of gaussian
        :param: position of gaussian

        :return: gaussian bumps over domain
        """
        return height * np.exp(-(1 / (self.spread_number * datalength)) *
                               (values - ((datalength / self.function_number) * index)) ** 2)

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

def SvcStop(self) -> None:
        """
        Called when the service is being shut down.
        """
        # tell the SCM we're shutting down
        # noinspection PyUnresolvedReferences
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        # fire the stop event
        win32event.SetEvent(self.h_stop_event)

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

def _newer(a, b):
    """Inquire whether file a was written since file b."""
    if not os.path.exists(a):
        return False
    if not os.path.exists(b):
        return True
    return os.path.getmtime(a) >= os.path.getmtime(b)

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

def _environment_variables() -> Dict[str, str]:
    """
    Wraps `os.environ` to filter out non-encodable values.
    """
    return {key: value
            for key, value in os.environ.items()
            if _is_encodable(value)}

def _kbhit_unix() -> bool:
    """
    Under UNIX: is a keystroke available?
    """
    dr, dw, de = select.select([sys.stdin], [], [], 0)
    return dr != []

def truncate_string(value, max_width=None):
    """Truncate string values."""
    if isinstance(value, text_type) and max_width is not None and len(value) > max_width:
        return value[:max_width]
    return value

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

def zoom_out(self):
        """Scale the image down by one scale step."""
        if self._scalefactor >= self._sfmin:
            self._scalefactor -= 1
            self.scale_image()
            self._adjust_scrollbar(1/self._scalestep)
            self.sig_zoom_changed.emit(self.get_scaling())

def __init__(self, enum_obj: Any) -> None:
        """Initialize attributes for informative output.

        :param enum_obj: Enum object.
        """
        if enum_obj:
            self.name = enum_obj
            self.items = ', '.join([str(i) for i in enum_obj])
        else:
            self.items = ''

def count(self, elem):
        """
        Return the number of elements equal to elem present in the queue

        >>> pdeque([1, 2, 1]).count(1)
        2
        """
        return self._left_list.count(elem) + self._right_list.count(elem)

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

def get_period_last_3_months() -> str:
    """ Returns the last week as a period string """
    today = Datum()
    today.today()

    # start_date = today - timedelta(weeks=13)
    start_date = today.clone()
    start_date.subtract_months(3)

    period = get_period(start_date.date, today.date)
    return period

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

def resize(im, short, max_size):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param short: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :return: resized image (NDArray) and scale (float)
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(short) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    return im, im_scale

def indexes_equal(a: Index, b: Index) -> bool:
    """
    Are two indexes equal? Checks by comparing ``str()`` versions of them.
    (AM UNSURE IF THIS IS ENOUGH.)
    """
    return str(a) == str(b)

def proper_round(n):
    """
    rounds float to closest int
    :rtype: int
    :param n: float
    """
    return int(n) + (n / abs(n)) * int(abs(n - int(n)) >= 0.5) if n != 0 else 0

def normalize(numbers):
    """Multiply each number by a constant such that the sum is 1.0
    >>> normalize([1,2,1])
    [0.25, 0.5, 0.25]
    """
    total = float(sum(numbers))
    return [n / total for n in numbers]

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

def memory_full():
    """Check if the memory is too full for further caching."""
    current_process = psutil.Process(os.getpid())
    return (current_process.memory_percent() >
            config.MAXIMUM_CACHE_MEMORY_PERCENTAGE)

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

def proper_round(n):
    """
    rounds float to closest int
    :rtype: int
    :param n: float
    """
    return int(n) + (n / abs(n)) * int(abs(n - int(n)) >= 0.5) if n != 0 else 0

def sorted_by(key: Callable[[raw_types.Qid], Any]) -> 'QubitOrder':
        """A basis that orders qubits ascending based on a key function.

        Args:
            key: A function that takes a qubit and returns a key value. The
                basis will be ordered ascending according to these key values.


        Returns:
            A basis that orders qubits ascending based on a key function.
        """
        return QubitOrder(lambda qubits: tuple(sorted(qubits, key=key)))

def is_finite(value: Any) -> bool:
    """Return true if a value is a finite number."""
    return isinstance(value, int) or (isinstance(value, float) and isfinite(value))

def rate_limited(max_per_hour: int, *args: Any) -> Callable[..., Any]:
    """Rate limit a function."""
    return util.rate_limited(max_per_hour, *args)

def is_finite(value: Any) -> bool:
    """Return true if a value is a finite number."""
    return isinstance(value, int) or (isinstance(value, float) and isfinite(value))

def dict_to_enum_fn(d: Dict[str, Any], enum_class: Type[Enum]) -> Enum:
    """
    Converts an ``dict`` to a ``Enum``.
    """
    return enum_class[d['name']]

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

def is_sqlatype_string(coltype: Union[TypeEngine, VisitableType]) -> bool:
    """
    Is the SQLAlchemy column type a string type?
    """
    coltype = _coltype_to_typeengine(coltype)
    return isinstance(coltype, sqltypes.String)

def smooth_image(image, sigma, sigma_in_physical_coordinates=True, FWHM=False, max_kernel_width=32):
    """
    Smooth an image

    ANTsR function: `smoothImage`

    Arguments
    ---------
    image   
        Image to smooth
    
    sigma   
        Smoothing factor. Can be scalar, in which case the same sigma is applied to each dimension, or a vector of length dim(inimage) to specify a unique smoothness for each dimension.
    
    sigma_in_physical_coordinates : boolean  
        If true, the smoothing factor is in millimeters; if false, it is in pixels.
    
    FWHM : boolean    
        If true, sigma is interpreted as the full-width-half-max (FWHM) of the filter, not the sigma of a Gaussian kernel.
    
    max_kernel_width : scalar    
        Maximum kernel width
    
    Returns
    -------
    ANTsImage
    
    Example
    -------
    >>> import ants
    >>> image = ants.image_read( ants.get_ants_data('r16'))
    >>> simage = ants.smooth_image(image, (1.2,1.5))
    """
    if image.components == 1:
        return _smooth_image_helper(image, sigma, sigma_in_physical_coordinates, FWHM, max_kernel_width)
    else:
        imagelist = utils.split_channels(image)
        newimages = []
        for image in imagelist:
            newimage = _smooth_image_helper(image, sigma, sigma_in_physical_coordinates, FWHM, max_kernel_width)
            newimages.append(newimage)
        return utils.merge_channels(newimages)

def has_changed (filename):
    """Check if filename has changed since the last check. If this
    is the first check, assume the file is changed."""
    key = os.path.abspath(filename)
    mtime = get_mtime(key)
    if key not in _mtime_cache:
        _mtime_cache[key] = mtime
        return True
    return mtime > _mtime_cache[key]

def fcast(value: float) -> TensorLike:
    """Cast to float tensor"""
    newvalue = tf.cast(value, FTYPE)
    if DEVICE == 'gpu':
        newvalue = newvalue.gpu()  # Why is this needed?  # pragma: no cover
    return newvalue

def most_significant_bit(lst: np.ndarray) -> int:
    """
    A helper function that finds the position of the most significant bit in a 1darray of 1s and 0s,
    i.e. the first position where a 1 appears, reading left to right.

    :param lst: a 1d array of 0s and 1s with at least one 1
    :return: the first position in lst that a 1 appears
    """
    return np.argwhere(np.asarray(lst) == 1)[0][0]

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

def zfill(x, width):
    """zfill(x, width) -> string

    Pad a numeric string x with zeros on the left, to fill a field
    of the specified width.  The string x is never truncated.

    """
    if not isinstance(x, basestring):
        x = repr(x)
    return x.zfill(width)

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

def __next__(self):
        """
        :return: int
        """
        self.current += 1
        if self.current > self.total:
            raise StopIteration
        else:
            return self.iterable[self.current - 1]

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

def url_host(url: str) -> str:
    """
    Parses hostname from URL.
    :param url: URL
    :return: hostname
    """
    from urllib.parse import urlparse
    res = urlparse(url)
    return res.netloc.split(':')[0] if res.netloc else ''

def is_natural(x):
    """A non-negative integer."""
    try:
        is_integer = int(x) == x
    except (TypeError, ValueError):
        return False
    return is_integer and x >= 0

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

def url_host(url: str) -> str:
    """
    Parses hostname from URL.
    :param url: URL
    :return: hostname
    """
    from urllib.parse import urlparse
    res = urlparse(url)
    return res.netloc.split(':')[0] if res.netloc else ''

def iter_lines(file_like: Iterable[str]) -> Generator[str, None, None]:
    """ Helper for iterating only nonempty lines without line breaks"""
    for line in file_like:
        line = line.rstrip('\r\n')
        if line:
            yield line

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

def multi_split(s, split):
    # type: (S, Iterable[S]) -> List[S]
    """Splits on multiple given separators."""
    for r in split:
        s = s.replace(r, "|")
    return [i for i in s.split("|") if len(i) > 0]

def is_integer(value: Any) -> bool:
    """Return true if a value is an integer number."""
    return (isinstance(value, int) and not isinstance(value, bool)) or (
        isinstance(value, float) and isfinite(value) and int(value) == value
    )

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

def _(f, x):
    """
    filter for dict, note `f` should have signature: `f::key->value->bool`
    """
    return {k: v for k, v in x.items() if f(k, v)}

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

def list_to_str(list, separator=','):
    """
    >>> list = [0, 0, 7]
    >>> list_to_str(list)
    '0,0,7'
    """
    list = [str(x) for x in list]
    return separator.join(list)

def convert_bytes_to_ints(in_bytes, num):
    """Convert a byte array into an integer array. The number of bytes forming an integer
    is defined by num

    :param in_bytes: the input bytes
    :param num: the number of bytes per int
    :return the integer array"""
    dt = numpy.dtype('>i' + str(num))
    return numpy.frombuffer(in_bytes, dt)

def fetchallfirstvalues(self, sql: str, *args) -> List[Any]:
        """Executes SQL; returns list of first values of each row."""
        rows = self.fetchall(sql, *args)
        return [row[0] for row in rows]

def read_las(source, closefd=True):
    """ Entry point for reading las data in pylas

    Reads the whole file into memory.

    >>> las = read_las("pylastests/simple.las")
    >>> las.classification
    array([1, 1, 1, ..., 1, 1, 1], dtype=uint8)

    Parameters
    ----------
    source : str or io.BytesIO
        The source to read data from

    closefd: bool
            if True and the source is a stream, the function will close it
            after it is done reading


    Returns
    -------
    pylas.lasdatas.base.LasBase
        The object you can interact with to get access to the LAS points & VLRs
    """
    with open_las(source, closefd=closefd) as reader:
        return reader.read()

def _my_hash(arg_list):
    # type: (List[Any]) -> int
    """Simple helper hash function"""
    res = 0
    for arg in arg_list:
        res = res * 31 + hash(arg)
    return res

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

def truncate(value: Decimal, n_digits: int) -> Decimal:
    """Truncates a value to a number of decimals places"""
    return Decimal(math.trunc(value * (10 ** n_digits))) / (10 ** n_digits)

def flatten_list(l: List[list]) -> list:
    """ takes a list of lists, l and returns a flat list
    """
    return [v for inner_l in l for v in inner_l]

def clean_column_names(df: DataFrame) -> DataFrame:
    """
    Strip the whitespace from all column names in the given DataFrame
    and return the result.
    """
    f = df.copy()
    f.columns = [col.strip() for col in f.columns]
    return f

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

def _parse_tuple_string(argument):
        """ Return a tuple from parsing 'a,b,c,d' -> (a,b,c,d) """
        if isinstance(argument, str):
            return tuple(int(p.strip()) for p in argument.split(','))
        return argument

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

def remove_empty_text(utterances: List[Utterance]) -> List[Utterance]:
    """Remove empty utterances from a list of utterances
    Args:
        utterances: The list of utterance we are processing
    """
    return [utter for utter in utterances if utter.text.strip() != ""]

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

def string(value) -> str:
        """ string dict/object/value to JSON """
        return system_json.dumps(Json(value).safe_object(), ensure_ascii=False)

def stop(self) -> None:
        """Stops the analysis as soon as possible."""
        if self._stop and not self._posted_kork:
            self._stop()
            self._stop = None

def non_zero_row(arr):
    """
        0.  Empty row returns False.

            >>> arr = array([])
            >>> non_zero_row(arr)

            False

        1.  Row with a zero returns False.

            >>> arr = array([1, 4, 3, 0, 5, -1, -2])
            >>> non_zero_row(arr)

            False
        2.  Row with no zeros returns True.

            >>> arr = array([-1, -0.1, 0.001, 2])
            >>> non_zero_row(arr)

            True

        :param arr: array
        :type arr: numpy array
        :return empty: If row is completely free of zeros
        :rtype empty: bool
    """

    if len(arr) == 0:
        return False

    for item in arr:
        if item == 0:
            return False

    return True

def unpackbools(integers, dtype='L'):
    """Yield booleans unpacking integers of dtype bit-length.

    >>> list(unpackbools([42], 'B'))
    [False, True, False, True, False, True, False, False]
    """
    atoms = ATOMS[dtype]

    for chunk in integers:
        for a in atoms:
            yield not not chunk & a

def is_done(self):
        """True if the last two moves were Pass or if the position is at a move
        greater than the max depth."""
        return self.position.is_game_over() or self.position.n >= FLAGS.max_game_length

def _generate(self):
        """Parses a file or directory of files into a set of ``Document`` objects."""
        doc_count = 0
        for fp in self.all_files:
            for doc in self._get_docs_for_path(fp):
                yield doc
                doc_count += 1
                if doc_count >= self.max_docs:
                    return

def call_spellchecker(cmd, input_text=None, encoding=None):
    """Call spell checker with arguments."""

    process = get_process(cmd)

    # A buffer has been provided
    if input_text is not None:
        for line in input_text.splitlines():
            # Hunspell truncates lines at `0x1fff` (at least on Windows this has been observed)
            # Avoid truncation by chunking the line on white space and inserting a new line to break it.
            offset = 0
            end = len(line)
            while True:
                chunk_end = offset + 0x1fff
                m = None if chunk_end >= end else RE_LAST_SPACE_IN_CHUNK.search(line, offset, chunk_end)
                if m:
                    chunk_end = m.start(1)
                    chunk = line[offset:m.start(1)]
                    offset = m.end(1)
                else:
                    chunk = line[offset:chunk_end]
                    offset = chunk_end
                # Avoid wasted calls to empty strings
                if chunk and not chunk.isspace():
                    process.stdin.write(chunk + b'\n')
                if offset >= end:
                    break

    return get_process_output(process, encoding)

def multivariate_normal_tril(x,
                             dims,
                             layer_fn=tf.compat.v1.layers.dense,
                             loc_fn=lambda x: x,
                             scale_fn=tril_with_diag_softplus_and_shift,
                             name=None):
  """Constructs a trainable `tfd.MultivariateNormalTriL` distribution.

  This function creates a MultivariateNormal (MVN) with lower-triangular scale
  matrix. By default the MVN is parameterized via affine transformation of input
  tensor `x`. Using default args, this function is mathematically equivalent to:

  ```none
  Y = MVN(loc=matmul(W, x) + b,
          scale_tril=f(reshape_tril(matmul(M, x) + c)))

  where,
    W in R^[d, n]
    M in R^[d*(d+1)/2, n]
    b in R^d
    c in R^d
    f(S) = set_diag(S, softplus(matrix_diag_part(S)) + 1e-5)
  ```

  Observe that `f` makes the diagonal of the triangular-lower scale matrix be
  positive and no smaller than `1e-5`.

  #### Examples

  ```python
  # This example fits a multilinear regression loss.
  import tensorflow as tf
  import tensorflow_probability as tfp

  # Create fictitious training data.
  dtype = np.float32
  n = 3000    # number of samples
  x_size = 4  # size of single x
  y_size = 2  # size of single y
  def make_training_data():
    np.random.seed(142)
    x = np.random.randn(n, x_size).astype(dtype)
    w = np.random.randn(x_size, y_size).astype(dtype)
    b = np.random.randn(1, y_size).astype(dtype)
    true_mean = np.tensordot(x, w, axes=[[-1], [0]]) + b
    noise = np.random.randn(n, y_size).astype(dtype)
    y = true_mean + noise
    return y, x
  y, x = make_training_data()

  # Build TF graph for fitting MVNTriL maximum likelihood estimator.
  mvn = tfp.trainable_distributions.multivariate_normal_tril(x, dims=y_size)
  loss = -tf.reduce_mean(mvn.log_prob(y))
  train_op = tf.train.AdamOptimizer(learning_rate=2.**-3).minimize(loss)
  mse = tf.reduce_mean(tf.squared_difference(y, mvn.mean()))
  init_op = tf.global_variables_initializer()

  # Run graph 1000 times.
  num_steps = 1000
  loss_ = np.zeros(num_steps)   # Style: `_` to indicate sess.run result.
  mse_ = np.zeros(num_steps)
  with tf.Session() as sess:
    sess.run(init_op)
    for it in xrange(loss_.size):
      _, loss_[it], mse_[it] = sess.run([train_op, loss, mse])
      if it % 200 == 0 or it == loss_.size - 1:
        print("iteration:{}  loss:{}  mse:{}".format(it, loss_[it], mse_[it]))

  # ==> iteration:0    loss:38.2020797729  mse:4.17175960541
  #     iteration:200  loss:2.90179634094  mse:0.990987896919
  #     iteration:400  loss:2.82727336884  mse:0.990926623344
  #     iteration:600  loss:2.82726788521  mse:0.990926682949
  #     iteration:800  loss:2.82726788521  mse:0.990926682949
  #     iteration:999  loss:2.82726788521  mse:0.990926682949
  ```

  Args:
    x: `Tensor` with floating type. Must have statically defined rank and
      statically known right-most dimension.
    dims: Scalar, `int`, `Tensor` indicated the MVN event size, i.e., the
      created MVN will be distribution over length-`dims` vectors.
    layer_fn: Python `callable` which takes input `x` and `int` scalar `d` and
      returns a transformation of `x` with shape
      `tf.concat([tf.shape(x)[:-1], [d]], axis=0)`.
      Default value: `tf.layers.dense`.
    loc_fn: Python `callable` which transforms the `loc` parameter. Takes a
      (batch of) length-`dims` vectors and returns a `Tensor` of same shape and
      `dtype`.
      Default value: `lambda x: x`.
    scale_fn: Python `callable` which transforms the `scale` parameters. Takes a
      (batch of) length-`dims * (dims + 1) / 2` vectors and returns a
      lower-triangular `Tensor` of same batch shape with rightmost dimensions
      having shape `[dims, dims]`.
      Default value: `tril_with_diag_softplus_and_shift`.
    name: A `name_scope` name for operations created by this function.
      Default value: `None` (i.e., "multivariate_normal_tril").

  Returns:
    mvntril: An instance of `tfd.MultivariateNormalTriL`.
  """
  with tf.compat.v1.name_scope(name, 'multivariate_normal_tril', [x, dims]):
    x = tf.convert_to_tensor(value=x, name='x')
    x = layer_fn(x, dims + dims * (dims + 1) // 2)
    return tfd.MultivariateNormalTriL(
        loc=loc_fn(x[..., :dims]),
        scale_tril=scale_fn(x[..., dims:]))

def binary(length):
    """
        returns a a random string that represent a binary representation

    :param length: number of bits
    """
    num = randint(1, 999999)
    mask = '0' * length
    return (mask + ''.join([str(num >> i & 1) for i in range(7, -1, -1)]))[-length:]

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

def string(value) -> str:
        """ string dict/object/value to JSON """
        return system_json.dumps(Json(value).safe_object(), ensure_ascii=False)

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

def rate_limited(max_per_hour: int, *args: Any) -> Callable[..., Any]:
    """Rate limit a function."""
    return util.rate_limited(max_per_hour, *args)

def strtobytes(input, encoding):
    """Take a str and transform it into a byte array."""
    py_version = sys.version_info[0]
    if py_version >= 3:
        return _strtobytes_py3(input, encoding)
    return _strtobytes_py2(input, encoding)

def _str_to_list(value, separator):
    """Convert a string to a list with sanitization."""
    value_list = [item.strip() for item in value.split(separator)]
    value_list_sanitized = builtins.list(filter(None, value_list))
    if len(value_list_sanitized) > 0:
        return value_list_sanitized
    else:
        raise ValueError('Invalid list variable.')

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

def is_finite(value: Any) -> bool:
    """Return true if a value is a finite number."""
    return isinstance(value, int) or (isinstance(value, float) and isfinite(value))

def flatten_multidict(multidict):
    """Return flattened dictionary from ``MultiDict``."""
    return dict([(key, value if len(value) > 1 else value[0])
                 for (key, value) in multidict.iterlists()])

def remove_blank_lines(string):
    """ Removes all blank lines in @string

        -> #str without blank lines
    """
    return "\n".join(line
                     for line in string.split("\n")
                     if len(line.strip()))

def truncate(value: Decimal, n_digits: int) -> Decimal:
    """Truncates a value to a number of decimals places"""
    return Decimal(math.trunc(value * (10 ** n_digits))) / (10 ** n_digits)

def incr(name, value=1, rate=1, tags=None):
    """Increment a metric by value.

    >>> import statsdecor
    >>> statsdecor.incr('my.metric')
    """
    client().incr(name, value, rate, tags)

def _cnx_is_empty(in_file):
    """Check if cnr or cns files are empty (only have a header)
    """
    with open(in_file) as in_handle:
        for i, line in enumerate(in_handle):
            if i > 0:
                return False
    return True

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

def flatten_list(l: List[list]) -> list:
    """ takes a list of lists, l and returns a flat list
    """
    return [v for inner_l in l for v in inner_l]

async def async_run(self) -> None:
        """
        Asynchronously run the worker, does not close connections. Useful when testing.
        """
        self.main_task = self.loop.create_task(self.main())
        await self.main_task

def file_uptodate(fname, cmp_fname):
    """Check if a file exists, is non-empty and is more recent than cmp_fname.
    """
    try:
        return (file_exists(fname) and file_exists(cmp_fname) and
                getmtime(fname) >= getmtime(cmp_fname))
    except OSError:
        return False

def iter_lines(file_like: Iterable[str]) -> Generator[str, None, None]:
    """ Helper for iterating only nonempty lines without line breaks"""
    for line in file_like:
        line = line.rstrip('\r\n')
        if line:
            yield line
