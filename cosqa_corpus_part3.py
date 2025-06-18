def wrap_count(method):
    """
    Returns number of wraps around given method.
    """
    number = 0
    while hasattr(method, '__aspects_orig'):
        number += 1
        method = method.__aspects_orig
    return number

def localize(dt):
    """Localize a datetime object to local time."""
    if dt.tzinfo is UTC:
        return (dt + LOCAL_UTC_OFFSET).replace(tzinfo=None)
    # No TZ info so not going to assume anything, return as-is.
    return dt

def _put_header(self):
        """ Standard first line in a PDF. """
        self.session._out('%%PDF-%s' % self.pdf_version)
        if self.session.compression:
            self.session.buffer += '%' + chr(235) + chr(236) + chr(237) + chr(238) + "\n"

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

def deinit(self):
        """Deinitialises the PulseIn and releases any hardware and software
        resources for reuse."""
        # Clean up after ourselves
        self._process.terminate()
        procs.remove(self._process)
        self._mq.remove()
        queues.remove(self._mq)

def Softsign(a):
    """
    Softsign op.
    """
    return np.divide(a, np.add(np.abs(a), 1)),

def err(msg):
    """Pretty-print an error."""
    click.echo(click.style(msg, fg="red", bold=True))

def drag_and_drop(self, droppable):
        """
        Performs drag a element to another elmenet.

        Currently works only on Chrome driver.
        """
        self.scroll_to()
        ActionChains(self.parent.driver).drag_and_drop(self._element, droppable._element).perform()

def compute_partition_size(result, processes):
    """
    Attempts to compute the partition size to evenly distribute work across processes. Defaults to
    1 if the length of result cannot be determined.

    :param result: Result to compute on
    :param processes: Number of processes to use
    :return: Best partition size
    """
    try:
        return max(math.ceil(len(result) / processes), 1)
    except TypeError:
        return 1

def mock_add_spec(self, spec, spec_set=False):
        """Add a spec to a mock. `spec` can either be an object or a
        list of strings. Only attributes on the `spec` can be fetched as
        attributes from the mock.

        If `spec_set` is True then only attributes on the spec can be set."""
        self._mock_add_spec(spec, spec_set)
        self._mock_set_magics()

def _render_table(data, fields=None):
  """ Helper to render a list of dictionaries as an HTML display object. """
  return IPython.core.display.HTML(datalab.utils.commands.HtmlBuilder.render_table(data, fields))

def text_cleanup(data, key, last_type):
    """ I strip extra whitespace off multi-line strings if they are ready to be stripped!"""
    if key in data and last_type == STRING_TYPE:
        data[key] = data[key].strip()
    return data

def get_category(self):
        """Get the category of the item.

        :return: the category of the item.
        :returntype: `unicode`"""
        var = self.xmlnode.prop("category")
        if not var:
            var = "?"
        return var.decode("utf-8")

def styles(self, dictobj):
		"""
		Add or update styles
		"""
		for k in dictobj:
			self.chart_style[k] = dictobj[k]

def load_image(fname):
    """ read an image from file - PIL doesnt close nicely """
    with open(fname, "rb") as f:
        i = Image.open(fname)
        #i.load()
        return i

def _get_latest_version():
    """Gets latest Dusty binary version using the GitHub api"""
    url = 'https://api.github.com/repos/{}/releases/latest'.format(constants.DUSTY_GITHUB_PATH)
    conn = urllib.urlopen(url)
    if conn.getcode() >= 300:
        raise RuntimeError('GitHub api returned code {}; can\'t determine latest version.  Aborting'.format(conn.getcode()))
    json_data = conn.read()
    return json.loads(json_data)['tag_name']

def run(args):
    """Process command line arguments and walk inputs."""
    raw_arguments = get_arguments(args[1:])
    process_arguments(raw_arguments)
    walk.run()
    return True

def add_parent(self, parent):
        """
        Adds self as child of parent, then adds parent.
        """
        parent.add_child(self)
        self.parent = parent
        return parent

def max_values(args):
    """ Return possible range for max function. """
    return Interval(max(x.low for x in args), max(x.high for x in args))

def mostCommonItem(lst):
    """Choose the most common item from the list, or the first item if all
    items are unique."""
    # This elegant solution from: http://stackoverflow.com/a/1518632/1760218
    lst = [l for l in lst if l]
    if lst:
        return max(set(lst), key=lst.count)
    else:
        return None

def run_web(self, flask, host='127.0.0.1', port=5000, **options):
        # type: (Zsl, str, int, **Any)->None
        """Alias for Flask.run"""
        return flask.run(
            host=flask.config.get('FLASK_HOST', host),
            port=flask.config.get('FLASK_PORT', port),
            debug=flask.config.get('DEBUG', False),
            **options
        )

def session_expired(self):
        """
        Returns True if login_time not set or seconds since
        login time is greater than 200 mins.
        """
        if not self._login_time or (datetime.datetime.now()-self._login_time).total_seconds() > 12000:
            return True

def validate_string(option, value):
    """Validates that 'value' is an instance of `basestring` for Python 2
    or `str` for Python 3.
    """
    if isinstance(value, string_type):
        return value
    raise TypeError("Wrong type for %s, value must be "
                    "an instance of %s" % (option, string_type.__name__))

def convert_timeval(seconds_since_epoch):
    """Convert time into C style timeval."""
    frac, whole = math.modf(seconds_since_epoch)
    microseconds = math.floor(frac * 1000000)
    seconds = math.floor(whole)
    return seconds, microseconds

def csort(objs, key):
    """Order-preserving sorting function."""
    idxs = dict((obj, i) for (i, obj) in enumerate(objs))
    return sorted(objs, key=lambda obj: (key(obj), idxs[obj]))

def execute(self, env, args):
        """ Starts a new task.

            `env`
                Runtime ``Environment`` instance.
            `args`
                Arguments object from arg parser.
            """

        # start the task
        if env.task.start(args.task_name):
            env.io.success(u'Task Loaded.')

def __init__(self, enum_obj: Any) -> None:
        """Initialize attributes for informative output.

        :param enum_obj: Enum object.
        """
        if enum_obj:
            self.name = enum_obj
            self.items = ', '.join([str(i) for i in enum_obj])
        else:
            self.items = ''

def give_str(self):
        """
            Give string representation of the callable.
        """
        args = self._args[:]
        kwargs = self._kwargs
        return self._give_str(args, kwargs)

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

def store_data(data):
    """Use this function to store data in a JSON file.

    This function is used for loading up a JSON file and appending additional
    data to the JSON file.

    :param data: the data to add to the JSON file.
    :type data: dict
    """
    with open(url_json_path) as json_file:
        try:
            json_file_data = load(json_file)
            json_file_data.update(data)
        except (AttributeError, JSONDecodeError):
            json_file_data = data
    with open(url_json_path, 'w') as json_file:
        dump(json_file_data, json_file, indent=4, sort_keys=True)

def is_same_nick(self, left, right):
        """ Check if given nicknames are equal in the server's case mapping. """
        return self.normalize(left) == self.normalize(right)

def lint(fmt='colorized'):
    """Run verbose PyLint on source. Optionally specify fmt=html for HTML output."""
    if fmt == 'html':
        outfile = 'pylint_report.html'
        local('pylint -f %s davies > %s || true' % (fmt, outfile))
        local('open %s' % outfile)
    else:
        local('pylint -f %s davies || true' % fmt)

def send_dir(self, local_path, remote_path, user='root'):
        """Upload a directory on the remote host.
        """
        self.enable_user(user)
        return self.ssh_pool.send_dir(user, local_path, remote_path)

def keys(self, index=None):
        """Returns a list of keys in the database
        """
        with self._lmdb.begin() as txn:
            return [key.decode() for key, _ in txn.cursor()]

def _check_elements_equal(lst):
    """
    Returns true if all of the elements in the list are equal.
    """
    assert isinstance(lst, list), "Input value must be a list."
    return not lst or lst.count(lst[0]) == len(lst)

def join(mapping, bind, values):
    """ Merge all the strings. Put space between them. """
    return [' '.join([six.text_type(v) for v in values if v is not None])]

def file_length(file_obj):
    """
    Returns the length in bytes of a given file object.
    Necessary because os.fstat only works on real files and not file-like
    objects. This works on more types of streams, primarily StringIO.
    """
    file_obj.seek(0, 2)
    length = file_obj.tell()
    file_obj.seek(0)
    return length

def make_strslice(lineno, s, lower, upper):
    """ Wrapper: returns String Slice node
    """
    return symbols.STRSLICE.make_node(lineno, s, lower, upper)

def getVectorFromType(self, dtype) -> Union[bool, None, Tuple[int, int]]:
        """
        :see: doc of method on parent class
        """
        if dtype == BIT:
            return False
        elif isinstance(dtype, Bits):
            return [evalParam(dtype.width) - 1, hInt(0)]

def cursor_up(self, count=1):
        """ (for multiline edit). Move cursor to the previous line.  """
        original_column = self.preferred_column or self.document.cursor_position_col
        self.cursor_position += self.document.get_cursor_up_position(
            count=count, preferred_column=original_column)

        # Remember the original column for the next up/down movement.
        self.preferred_column = original_column

def to_html(self, write_to):
        """Method to convert the repository list to a search results page and
        write it to a HTML file.

        :param write_to: File/Path to write the html file to.
        """
        page_html = self.get_html()

        with open(write_to, "wb") as writefile:
            writefile.write(page_html.encode("utf-8"))

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

def url(viewname, *args, **kwargs):
    """Helper for Django's ``reverse`` in templates."""
    return reverse(viewname, args=args, kwargs=kwargs)

def project(self, other):
        """Return one vector projected on the vector other"""
        n = other.normalized()
        return self.dot(n) * n

def readline( file, skip_blank=False ):
    """Read a line from provided file, skipping any blank or comment lines"""
    while 1:
        line = file.readline()
        #print "every line: %r" % line
        if not line: return None 
        if line[0] != '#' and not ( skip_blank and line.isspace() ):
            return line

def accuracy(self):
        """Calculates accuracy

        :return: Accuracy
        """
        true_pos = self.matrix[0][0]
        false_pos = self.matrix[1][0]
        false_neg = self.matrix[0][1]
        true_neg = self.matrix[1][1]

        num = 1.0 * (true_pos + true_neg)
        den = true_pos + true_neg + false_pos + false_neg

        return divide(num, den)

def sp_rand(m,n,a):
    """
    Generates an mxn sparse 'd' matrix with round(a*m*n) nonzeros.
    """
    if m == 0 or n == 0: return spmatrix([], [], [], (m,n))
    nnz = min(max(0, int(round(a*m*n))), m*n)
    nz = matrix(random.sample(range(m*n), nnz), tc='i')
    return spmatrix(normal(nnz,1), nz%m, matrix([int(ii) for ii in nz/m]), (m,n))

def _getTypename(self, defn):
        """ Returns the SQL typename required to store the given FieldDefinition """
        return 'REAL' if defn.type.float or 'TIME' in defn.type.name or defn.dntoeu else 'INTEGER'

def close(self):
        """Flush the file and close it.

        A closed file cannot be written any more. Calling
        :meth:`close` more than once is allowed.
        """
        if not self._closed:
            self.__flush()
            object.__setattr__(self, "_closed", True)

def write_dict_to_yaml(dictionary, path, **kwargs):
    """
    Writes a dictionary to a yaml file
    :param dictionary:  the dictionary to be written
    :param path: the absolute path of the target yaml file
    :param kwargs: optional additional parameters for dumper
    """
    with open(path, 'w') as f:
        yaml.dump(dictionary, f, indent=4, **kwargs)

def findMax(arr):
    """
    in comparison to argrelmax() more simple and  reliable peak finder
    """
    out = np.zeros(shape=arr.shape, dtype=bool)
    _calcMax(arr, out)
    return out

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

def top(n, width=WIDTH, style=STYLE):
    """Prints the top row of a table"""
    return hrule(n, width, linestyle=STYLES[style].top)

def safe_dump(data, stream=None, **kwds):
    """implementation of safe dumper using Ordered Dict Yaml Dumper"""
    return yaml.dump(data, stream=stream, Dumper=ODYD, **kwds)

def top_1_tpu(inputs):
  """find max and argmax over the last dimension.

  Works well on TPU

  Args:
    inputs: A tensor with shape [..., depth]

  Returns:
    values: a Tensor with shape [...]
    indices: a Tensor with shape [...]
  """
  inputs_max = tf.reduce_max(inputs, axis=-1, keepdims=True)
  mask = tf.to_int32(tf.equal(inputs_max, inputs))
  index = tf.range(tf.shape(inputs)[-1]) * mask
  return tf.squeeze(inputs_max, -1), tf.reduce_max(index, axis=-1)

def create_bigquery_table(self, database, schema, table_name, callback,
                              sql):
        """Create a bigquery table. The caller must supply a callback
        that takes one argument, a `google.cloud.bigquery.Table`, and mutates
        it.
        """
        conn = self.get_thread_connection()
        client = conn.handle

        view_ref = self.table_ref(database, schema, table_name, conn)
        view = google.cloud.bigquery.Table(view_ref)
        callback(view)

        with self.exception_handler(sql):
            client.create_table(view)

def cli(env):
    """Show current configuration."""

    settings = config.get_settings_from_client(env.client)
    env.fout(config.config_table(settings))

def to_linspace(self):
        """
        convert from full to linspace
        """
        if hasattr(self.shape, '__len__'):
            raise NotImplementedError("can only convert flat Full arrays to linspace")
        return Linspace(self.fill_value, self.fill_value, self.shape)

def disown(cmd):
    """Call a system command in the background,
       disown it and hide it's output."""
    subprocess.Popen(cmd,
                     stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)

def convert(name):
    """Convert CamelCase to underscore

    Parameters
    ----------
    name : str
        Camelcase string

    Returns
    -------
    name : str
        Converted name
    """ 
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def delete_index(self):
        """
        Delete the index, if it exists.
        """
        es = self._init_connection()
        if es.indices.exists(index=self.index):
            es.indices.delete(index=self.index)

def qrandom(n):
  """
  Creates an array of n true random numbers obtained from the quantum random
  number generator at qrng.anu.edu.au

  This function requires the package quantumrandom and an internet connection.

  Args:
    n (int):
      length of the random array

  Return:
    array of ints:
      array of truly random unsigned 16 bit int values
  """
  import quantumrandom
  return np.concatenate([
    quantumrandom.get_data(data_type='uint16', array_length=1024)
    for i in range(int(np.ceil(n/1024.0)))
  ])[:n]

def remove_instance(self, item):
        """Remove `instance` from model"""
        self.instances.remove(item)
        self.remove_item(item)

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

def make_aware(dt):
    """Appends tzinfo and assumes UTC, if datetime object has no tzinfo already."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

def _RemoveIllegalXMLCharacters(self, xml_string):
    """Removes illegal characters for XML.

    If the input is not a string it will be returned unchanged.

    Args:
      xml_string (str): XML with possible illegal characters.

    Returns:
      str: XML where all illegal characters have been removed.
    """
    if not isinstance(xml_string, py2to3.STRING_TYPES):
      return xml_string

    return self._ILLEGAL_XML_RE.sub('\ufffd', xml_string)

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

def iget_list_column_slice(list_, start=None, stop=None, stride=None):
    """ iterator version of get_list_column """
    if isinstance(start, slice):
        slice_ = start
    else:
        slice_ = slice(start, stop, stride)
    return (row[slice_] for row in list_)

def raw_print(*args, **kw):
    """Raw print to sys.__stdout__, otherwise identical interface to print()."""

    print(*args, sep=kw.get('sep', ' '), end=kw.get('end', '\n'),
          file=sys.__stdout__)
    sys.__stdout__.flush()

def add(self, name, desc, func=None, args=None, krgs=None):
        """Add a menu entry."""
        self.entries.append(MenuEntry(name, desc, func, args or [], krgs or {}))

def md_to_text(content):
    """ Converts markdown content to text """
    text = None
    html = markdown.markdown(content)
    if html:
        text = html_to_text(content)
    return text

def cric__decision_tree():
    """ Decision Tree
    """
    model = sklearn.tree.DecisionTreeClassifier(random_state=0, max_depth=4)

    # we want to explain the raw probability outputs of the trees
    model.predict = lambda X: model.predict_proba(X)[:,1]
    
    return model

def cfloat64_array_to_numpy(cptr, length):
    """Convert a ctypes double pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_double)):
        return np.fromiter(cptr, dtype=np.float64, count=length)
    else:
        raise RuntimeError('Expected double pointer')

def split_elements(value):
    """Split a string with comma or space-separated elements into a list."""
    l = [v.strip() for v in value.split(',')]
    if len(l) == 1:
        l = value.split()
    return l

def _spawn_kafka_consumer_thread(self):
        """Spawns a kafka continuous consumer thread"""
        self.logger.debug("Spawn kafka consumer thread""")
        self._consumer_thread = Thread(target=self._consumer_loop)
        self._consumer_thread.setDaemon(True)
        self._consumer_thread.start()

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

def extract_module_locals(depth=0):
    """Returns (module, locals) of the funciton `depth` frames away from the caller"""
    f = sys._getframe(depth + 1)
    global_ns = f.f_globals
    module = sys.modules[global_ns['__name__']]
    return (module, f.f_locals)

def counter_from_str(self, string):
        """Build word frequency list from incoming string."""
        string_list = [chars for chars in string if chars not in self.punctuation]
        string_joined = ''.join(string_list)
        tokens = self.punkt.word_tokenize(string_joined)
        return Counter(tokens)

def line_line_collide(line1, line2):
    """Determine if two line segments meet.

    This is a helper for :func:`convex_hull_collide` in the
    special case that the two convex hulls are actually
    just line segments. (Even in this case, this is only
    problematic if both segments are on a single line.)

    Args:
        line1 (numpy.ndarray): ``2 x 2`` array of start and end nodes.
        line2 (numpy.ndarray): ``2 x 2`` array of start and end nodes.

    Returns:
        bool: Indicating if the line segments collide.
    """
    s, t, success = segment_intersection(
        line1[:, 0], line1[:, 1], line2[:, 0], line2[:, 1]
    )
    if success:
        return _helpers.in_interval(s, 0.0, 1.0) and _helpers.in_interval(
            t, 0.0, 1.0
        )

    else:
        disjoint, _ = parallel_lines_parameters(
            line1[:, 0], line1[:, 1], line2[:, 0], line2[:, 1]
        )
        return not disjoint

def Serializable(o):
    """Make sure an object is JSON-serializable
    Use this to return errors and other info that does not need to be
    deserialized or does not contain important app data. Best for returning
    error info and such"""
    if isinstance(o, (str, dict, int)):
        return o
    else:
        try:
            json.dumps(o)
            return o
        except Exception:
            LOG.debug("Got a non-serilizeable object: %s" % o)
            return o.__repr__()

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

def schedule_task(self):
        """
        Schedules this publish action as a Celery task.
        """
        from .tasks import publish_task

        publish_task.apply_async(kwargs={'pk': self.pk}, eta=self.scheduled_time)

def is_valid_file(parser, arg):
    """Check if arg is a valid file that already exists on the file system."""
    arg = os.path.abspath(arg)
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg

def write_color(string, name, style='normal', when='auto'):
    """ Write the given colored string to standard out. """
    write(color(string, name, style, when))

def dumps(obj, indent=None, default=None, sort_keys=False, **kw):
    """Dump string."""
    return YAMLEncoder(indent=indent, default=default, sort_keys=sort_keys, **kw).encode(obj)

def toArray(self):
        """
        Returns a copy of this SparseVector as a 1-dimensional NumPy array.
        """
        arr = np.zeros((self.size,), dtype=np.float64)
        arr[self.indices] = self.values
        return arr

def destroy(self):
        """ Cleanup the activty lifecycle listener """
        if self.widget:
            self.set_active(False)
        super(AndroidBarcodeView, self).destroy()

def paginate(self, request, offset=0, limit=None):
        """Paginate queryset."""
        return self.collection.offset(offset).limit(limit), self.collection.count()

def render_template(content, context):
    """ renders context aware template """
    rendered = Template(content).render(Context(context))
    return rendered

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

def unpack(self, s):
        """Parse bytes and return a namedtuple."""
        return self._create(super(NamedStruct, self).unpack(s))

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

def get_points(self):
        """Returns a ketama compatible list of (position, nodename) tuples.
        """
        return [(k, self.runtime._ring[k]) for k in self.runtime._keys]

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

def reversed_lines(path):
    """Generate the lines of file in reverse order."""
    with open(path, 'r') as handle:
        part = ''
        for block in reversed_blocks(handle):
            for c in reversed(block):
                if c == '\n' and part:
                    yield part[::-1]
                    part = ''
                part += c
        if part: yield part[::-1]

def print_param_values(self_):
        """Print the values of all this object's Parameters."""
        self = self_.self
        for name,val in self.param.get_param_values():
            print('%s.%s = %s' % (self.name,name,val))

def get_entity_kind(self, model_obj):
        """
        Returns a tuple for a kind name and kind display name of an entity.
        By default, uses the app_label and model of the model object's content
        type as the kind.
        """
        model_obj_ctype = ContentType.objects.get_for_model(self.queryset.model)
        return (u'{0}.{1}'.format(model_obj_ctype.app_label, model_obj_ctype.model), u'{0}'.format(model_obj_ctype))

def is_valid_ipv6(ip_str):
    """
    Check the validity of an IPv6 address
    """
    try:
        socket.inet_pton(socket.AF_INET6, ip_str)
    except socket.error:
        return False
    return True

def raises_regex(self, expected_exception, expected_regexp):
        """
        Ensures preceding predicates (specifically, :meth:`called_with()`) result in *expected_exception* being raised,
        and the string representation of *expected_exception* must match regular expression *expected_regexp*.
        """
        return unittest_case.assertRaisesRegexp(expected_exception, expected_regexp, self._orig_subject,
                                                *self._args, **self._kwargs)

def _display(self, layout):
        """launch layouts display"""
        print(file=self.out)
        TextWriter().format(layout, self.out)

def isolate_element(self, x):
        """Isolates `x` from its equivalence class."""
        members = list(self.members(x))
        self.delete_set(x)
        self.union(*(v for v in members if v != x))

def url_host(url: str) -> str:
    """
    Parses hostname from URL.
    :param url: URL
    :return: hostname
    """
    from urllib.parse import urlparse
    res = urlparse(url)
    return res.netloc.split(':')[0] if res.netloc else ''

def assert_any_call(self, *args, **kwargs):
        """assert the mock has been called with the specified arguments.

        The assert passes if the mock has *ever* been called, unlike
        `assert_called_with` and `assert_called_once_with` that only pass if
        the call is the most recent one."""
        kall = call(*args, **kwargs)
        if kall not in self.call_args_list:
            expected_string = self._format_mock_call_signature(args, kwargs)
            raise AssertionError(
                '%s call not found' % expected_string
            )

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

def _fullname(o):
    """Return the fully-qualified name of a function."""
    return o.__module__ + "." + o.__name__ if o.__module__ else o.__name__

def stn(s, length, encoding, errors):
    """Convert a string to a null-terminated bytes object.
    """
    s = s.encode(encoding, errors)
    return s[:length] + (length - len(s)) * NUL

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

def guess_extension(amimetype, normalize=False):
    """
    Tries to guess extension for a mimetype.

    @param amimetype: name of a mimetype
    @time amimetype: string
    @return: the extension
    @rtype: string
    """
    ext = _mimes.guess_extension(amimetype)
    if ext and normalize:
        # Normalize some common magic mis-interpreation
        ext = {'.asc': '.txt', '.obj': '.bin'}.get(ext, ext)
        from invenio.legacy.bibdocfile.api_normalizer import normalize_format
        return normalize_format(ext)
    return ext

def indentsize(line):
    """Return the indent size, in spaces, at the start of a line of text."""
    expline = string.expandtabs(line)
    return len(expline) - len(string.lstrip(expline))

def get_single_value(d):
    """Get a value from a dict which contains just one item."""
    assert len(d) == 1, 'Single-item dict must have just one item, not %d.' % len(d)
    return next(six.itervalues(d))

def __init__(self, interval, key):
    """Constructor. See class docstring for parameter details."""
    self.interval = interval
    self.key = key

def expandvars_dict(settings):
    """Expands all environment variables in a settings dictionary."""
    return dict(
        (key, os.path.expandvars(value))
        for key, value in settings.iteritems()
    )

def view_extreme_groups(token, dstore):
    """
    Show the source groups contributing the most to the highest IML
    """
    data = dstore['disagg_by_grp'].value
    data.sort(order='extreme_poe')
    return rst_table(data[::-1])

def perl_cmd():
    """Retrieve path to locally installed conda Perl or first in PATH.
    """
    perl = which(os.path.join(get_bcbio_bin(), "perl"))
    if perl:
        return perl
    else:
        return which("perl")

def topk(arg, k, by=None):
    """
    Returns
    -------
    topk : TopK filter expression
    """
    op = ops.TopK(arg, k, by=by)
    return op.to_expr()

def full(self):
        """Return True if the queue is full"""
        if not self.size: return False
        return len(self.pq) == (self.size + self.removed_count)

def expand_path(path):
  """Returns ``path`` as an absolute path with ~user and env var expansion applied.

  :API: public
  """
  return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))

def _attach_files(filepaths, email_):
    """Take a list of filepaths and attach the files to a MIMEMultipart.

    Args:
        filepaths (list(str)): A list of filepaths.
        email_ (email.MIMEMultipart): A MIMEMultipart email_.
    """
    for filepath in filepaths:
        base = os.path.basename(filepath)
        with open(filepath, "rb") as file:
            part = MIMEApplication(file.read(), Name=base)
            part["Content-Disposition"] = 'attachment; filename="%s"' % base
            email_.attach(part)

def set_sig_figs(n=4):
    """Set the number of significant figures used to print Pint, Pandas, and
    NumPy quantities.

    Args:
        n (int): Number of significant figures to display.
    """
    u.default_format = '.' + str(n) + 'g'
    pd.options.display.float_format = ('{:,.' + str(n) + '}').format

def issuperset(self, items):
        """Return whether this collection contains all items.

        >>> Unique(['spam', 'eggs']).issuperset(['spam', 'spam', 'spam'])
        True
        """
        return all(_compat.map(self._seen.__contains__, items))

def command(self, cmd, *args):
        """
        Sends a command and an (optional) sequence of arguments through to the
        delegated serial interface. Note that the arguments are passed through
        as data.
        """
        self._serial_interface.command(cmd)
        if len(args) > 0:
            self._serial_interface.data(list(args))

def get_sparse_matrix_keys(session, key_table):
    """Return a list of keys for the sparse matrix."""
    return session.query(key_table).order_by(key_table.name).all()

def counter(items):
    """
    Simplest required implementation of collections.Counter. Required as 2.6
    does not have Counter in collections.
    """
    results = {}
    for item in items:
        results[item] = results.get(item, 0) + 1
    return results

def estimate_complexity(self, x,y,z,n):
        """ 
        calculates a rough guess of runtime based on product of parameters 
        """
        num_calculations = x * y * z * n
        run_time = num_calculations / 100000  # a 2014 PC does about 100k calcs in a second (guess based on prior logs)
        return self.show_time_as_short_string(run_time)

def C_dict2array(C):
    """Convert an OrderedDict containing C values to a 1D array."""
    return np.hstack([np.asarray(C[k]).ravel() for k in C_keys])

def position(self, x, y, text):
        """
            ANSI Escape sequences
            http://ascii-table.com/ansi-escape-sequences.php
        """
        sys.stdout.write("\x1b7\x1b[%d;%df%s\x1b8" % (x, y, text))
        sys.stdout.flush()

def json_response(data, status=200):
    """Return a JsonResponse. Make sure you have django installed first."""
    from django.http import JsonResponse
    return JsonResponse(data=data, status=status, safe=isinstance(data, dict))

def flatten(l):
    """Flatten a nested list."""
    return sum(map(flatten, l), []) \
        if isinstance(l, list) or isinstance(l, tuple) else [l]

def resize(self, width, height):
        """
        @summary: override resize function
        @param width: {int} width of widget
        @param height: {int} height of widget
        """
        self._buffer = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)
        QtGui.QWidget.resize(self, width, height)

def file_lines(bblfile:str) -> iter:
    """Yield lines found in given file"""
    with open(bblfile) as fd:
        yield from (line.rstrip() for line in fd if line.rstrip())

def region_from_segment(image, segment):
    """given a segment (rectangle) and an image, returns it's corresponding subimage"""
    x, y, w, h = segment
    return image[y:y + h, x:x + w]

def columns(self):
        """Return names of all the addressable columns (including foreign keys) referenced in user supplied model"""
        res = [col['name'] for col in self.column_definitions]
        res.extend([col['name'] for col in self.foreign_key_definitions])
        return res

def put(self, entity):
    """Registers entity to put to datastore.

    Args:
      entity: an entity or model instance to put.
    """
    actual_entity = _normalize_entity(entity)
    if actual_entity is None:
      return self.ndb_put(entity)
    self.puts.append(actual_entity)

def yum_install(self, packages, ignore_error=False):
        """Install some packages on the remote host.

        :param packages: ist of packages to install.
        """
        return self.run('yum install -y --quiet ' + ' '.join(packages), ignore_error=ignore_error, retry=5)

def load_fasta_file(filename):
    """Load a FASTA file and return the sequences as a list of SeqRecords

    Args:
        filename (str): Path to the FASTA file to load

    Returns:
        list: list of all sequences in the FASTA file as Biopython SeqRecord objects

    """

    with open(filename, "r") as handle:
        records = list(SeqIO.parse(handle, "fasta"))
    return records

def list_to_str(list, separator=','):
    """
    >>> list = [0, 0, 7]
    >>> list_to_str(list)
    '0,0,7'
    """
    list = [str(x) for x in list]
    return separator.join(list)

def _MakeExecutable(self, metadata_script):
    """Add executable permissions to a file.

    Args:
      metadata_script: string, the path to the executable file.
    """
    mode = os.stat(metadata_script).st_mode
    os.chmod(metadata_script, mode | stat.S_IEXEC)

def add_to_parser(self, parser):
        """
        Adds the argument to an argparse.ArgumentParser instance

        @param parser An argparse.ArgumentParser instance
        """
        kwargs = self._get_kwargs()
        args = self._get_args()
        parser.add_argument(*args, **kwargs)

def shutdown():
    """Manually shutdown the async API.

    Cancels all related tasks and all the socket transportation.
    """
    global handler, transport, protocol
    if handler is not None:
        handler.close()
        transport.close()
        handler = None
        transport = None
        protocol = None

def _notnull(expr):
    """
    Return a sequence or scalar according to the input indicating if the values are not null.

    :param expr: sequence or scalar
    :return: sequence or scalar
    """

    if isinstance(expr, SequenceExpr):
        return NotNull(_input=expr, _data_type=types.boolean)
    elif isinstance(expr, Scalar):
        return NotNull(_input=expr, _value_type=types.boolean)

def mean_cl_boot(series, n_samples=1000, confidence_interval=0.95,
                 random_state=None):
    """
    Bootstrapped mean with confidence limits
    """
    return bootstrap_statistics(series, np.mean,
                                n_samples=n_samples,
                                confidence_interval=confidence_interval,
                                random_state=random_state)

def remove_accent_string(string):
    """
    Remove all accent from a whole string.
    """
    return utils.join([add_accent_char(c, Accent.NONE) for c in string])

def many_until1(these, term):
    """Like many_until but must consume at least one of these.
    """
    first = [these()]
    these_results, term_result = many_until(these, term)
    return (first + these_results, term_result)

def parse_fixed_width(types, lines):
    """Parse a fixed width line."""
    values = []
    line = []
    for width, parser in types:
        if not line:
            line = lines.pop(0).replace('\n', '')

        values.append(parser(line[:width]))
        line = line[width:]

    return values

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

def paragraph(separator='\n\n', wrap_start='', wrap_end='',
              html=False, sentences_quantity=3):
    """Return a random paragraph."""
    return paragraphs(quantity=1, separator=separator, wrap_start=wrap_start,
                      wrap_end=wrap_end, html=html,
                      sentences_quantity=sentences_quantity)

def threads_init(gtk=True):
    """Enables multithreading support in Xlib and PyGTK.
    See the module docstring for more info.
    
    :Parameters:
      gtk : bool
        May be set to False to skip the PyGTK module.
    """
    # enable X11 multithreading
    x11.XInitThreads()
    if gtk:
        from gtk.gdk import threads_init
        threads_init()

def lint_file(in_file, out_file=None):
    """Helps remove extraneous whitespace from the lines of a file

    :param file in_file: A readable file or file-like
    :param file out_file: A writable file or file-like
    """
    for line in in_file:
        print(line.strip(), file=out_file)

def parse_response(self, resp):
        """
        Parse the xmlrpc response.
        """
        p, u = self.getparser()
        p.feed(resp.content)
        p.close()
        return u.close()

def onRightUp(self, event=None):
        """ right button up: put back to cursor mode"""
        if event is None:
            return
        self.cursor_mode_action('rightup', event=event)
        self.ForwardEvent(event=event.guiEvent)

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

def save_pdf(path):
  """
  Saves a pdf of the current matplotlib figure.

  :param path: str, filepath to save to
  """

  pp = PdfPages(path)
  pp.savefig(pyplot.gcf())
  pp.close()

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

def __deepcopy__(self, memo):
        """Improve deepcopy speed."""
        return type(self)(value=self._value, enum_ref=self.enum_ref)

def _rank(self, ranking, n):
    """ return the first n sentences with highest ranking """
    return nlargest(n, ranking, key=ranking.get)

def isdir(self, path):
        """Return true if the path refers to an existing directory.

        Parameters
        ----------
        path : str
            Path of directory on the remote side to check.
        """
        result = True
        try:
            self.sftp_client.lstat(path)
        except FileNotFoundError:
            result = False

        return result

def fourier_series(x, f, n=0):
    """
    Returns a symbolic fourier series of order `n`.

    :param n: Order of the fourier series.
    :param x: Independent variable
    :param f: Frequency of the fourier series
    """
    # Make the parameter objects for all the terms
    a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
    # Construct the series
    series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
                     for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    return series

def run_cmd(command, verbose=True, shell='/bin/bash'):
    """internal helper function to run shell commands and get output"""
    process = Popen(command, shell=True, stdout=PIPE, stderr=STDOUT, executable=shell)
    output = process.stdout.read().decode().strip().split('\n')
    if verbose:
        # return full output including empty lines
        return output
    return [line for line in output if line.strip()]

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

def xml_str_to_dict(s):
    """ Transforms an XML string it to python-zimbra dict format

    For format, see:
      https://github.com/Zimbra-Community/python-zimbra/blob/master/README.md

    :param: a string, containing XML
    :returns: a dict, with python-zimbra format
    """
    xml = minidom.parseString(s)
    return pythonzimbra.tools.xmlserializer.dom_to_dict(xml.firstChild)

def smartread(path):
    """Read text from file, automatically detect encoding. ``chardet`` required.
    """
    with open(path, "rb") as f:
        content = f.read()
        result = chardet.detect(content)
        return content.decode(result["encoding"])

def istype(obj, check):
    """Like isinstance(obj, check), but strict.

    This won't catch subclasses.
    """
    if isinstance(check, tuple):
        for cls in check:
            if type(obj) is cls:
                return True
        return False
    else:
        return type(obj) is check

def inverse(self):
        """
        Returns inverse of transformation.
        """
        invr = np.linalg.inv(self.affine_matrix)
        return SymmOp(invr)

def join(self, room):
        """Lets a user join a room on a specific Namespace."""
        self.socket.rooms.add(self._get_room_name(room))

def copy_of_xml_element(elem):
    """
    This method returns a shallow copy of a XML-Element.
    This method is for compatibility with Python 2.6 or earlier..
    In Python 2.7 you can use  'copyElem = elem.copy()'  instead.
    """

    copyElem = ElementTree.Element(elem.tag, elem.attrib)
    for child in elem:
        copyElem.append(child)
    return copyElem

def get_model(name):
    """
    Convert a model's verbose name to the model class. This allows us to
    use the models verbose name in steps.
    """

    model = MODELS.get(name.lower(), None)

    assert model, "Could not locate model by name '%s'" % name

    return model

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

def _remove_blank(l):
        """ Removes trailing zeros in the list of integers and returns a new list of integers"""
        ret = []
        for i, _ in enumerate(l):
            if l[i] == 0:
                break
            ret.append(l[i])
        return ret

def _py_ex_argtype(executable):
    """Returns the code to create the argtype to assign to the methods argtypes
    attribute.
    """
    result = []
    for p in executable.ordered_parameters:
        atypes = p.argtypes
        if atypes is not None:
            result.extend(p.argtypes)
        else:
            print(("No argtypes for: {}".format(p.definition())))

    if type(executable).__name__ == "Function":
        result.extend(executable.argtypes)        
            
    return result

def stoplog(self):
        """ Stop logging.
    
        @return: 1 on success and 0 on error
        @rtype: integer
        """
        if self._file_logger:
            self.logger.removeHandler(_file_logger)
            self._file_logger = None
        return 1

def world_to_view(v):
    """world coords to view coords; v an eu.Vector2, returns (float, float)"""
    return v.x * config.scale_x, v.y * config.scale_y

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

def save(self, f):
        """Save pickled model to file."""
        return pickle.dump((self.perceptron.weights, self.tagdict, self.classes, self.clusters), f, protocol=pickle.HIGHEST_PROTOCOL)

def count_nulls(self, field):
        """
        Count the number of null values in a column
        """
        try:
            n = self.df[field].isnull().sum()
        except KeyError:
            self.warning("Can not find column", field)
            return
        except Exception as e:
            self.err(e, "Can not count nulls")
            return
        self.ok("Found", n, "nulls in column", field)

def update_loan_entry(database, entry):
    """Update a record of a loan report in the provided database.

    @param db: The MongoDB database to operate on. The loans collection will be
        used from this database.
    @type db: pymongo.database.Database
    @param entry: The entry to insert into the database, updating the entry with
        the same recordID if one exists.
    @type entry: dict
    """
    entry = clean_entry(entry)
    database.loans.update(
        {'recordID': entry['recordID']},
        {'$set': entry},
        upsert=True
    )

def path(self):
        """
        Return the path always without the \\?\ prefix.
        """
        path = super(WindowsPath2, self).path
        if path.startswith("\\\\?\\"):
            return path[4:]
        return path

def _jit_pairwise_distances(pos1, pos2):
        """Optimized function for calculating the distance between each pair
        of points in positions1 and positions2.

        Does use python mode as fallback, if a scalar and not an array is
        given.
        """
        n1 = pos1.shape[0]
        n2 = pos2.shape[0]
        D = np.empty((n1, n2))

        for i in range(n1):
            for j in range(n2):
                D[i, j] = np.sqrt(((pos1[i] - pos2[j])**2).sum())
        return D

def decamelise(text):
    """Convert CamelCase to lower_and_underscore."""
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()

def send_photo(self, photo: str, caption: str=None, reply: Message=None, on_success: callable=None,
                   reply_markup: botapi.ReplyMarkup=None):
        """
        Send photo to this peer.
        :param photo: File path to photo to send.
        :param caption: Caption for photo
        :param reply: Message object or message_id to reply to.
        :param on_success: Callback to call when call is complete.

        :type reply: int or Message
        """
        self.twx.send_photo(peer=self, photo=photo, caption=caption, reply=reply, reply_markup=reply_markup,
                            on_success=on_success)

def insert_one(self, mongo_collection, doc, mongo_db=None, **kwargs):
        """
        Inserts a single document into a mongo collection
        https://api.mongodb.com/python/current/api/pymongo/collection.html#pymongo.collection.Collection.insert_one
        """
        collection = self.get_collection(mongo_collection, mongo_db=mongo_db)

        return collection.insert_one(doc, **kwargs)

def source_range(start, end, nr_var_dict):
    """
    Given a range of source numbers, as well as a dictionary
    containing the numbers of each source, returns a dictionary
    containing tuples of the start and end index
    for each source variable type.
    """

    return OrderedDict((k, e-s)
        for k, (s, e)
        in source_range_tuple(start, end, nr_var_dict).iteritems())

def main(args):
    """
    invoke wptools and exit safely
    """
    start = time.time()
    output = get(args)
    _safe_exit(start, output)

def has_value(cls, value: int) -> bool:
        """True if specified value exists in int enum; otherwise, False."""
        return any(value == item.value for item in cls)

def glob_by_extensions(directory, extensions):
    """ Returns files matched by all extensions in the extensions list """
    directorycheck(directory)
    files = []
    xt = files.extend
    for ex in extensions:
        xt(glob.glob('{0}/*.{1}'.format(directory, ex)))
    return files

def print_err(*args, end='\n'):
    """Similar to print, but prints to stderr.
    """
    print(*args, end=end, file=sys.stderr)
    sys.stderr.flush()

def get_table_names_from_metadata(metadata: MetaData) -> List[str]:
    """
    Returns all database table names found in an SQLAlchemy :class:`MetaData`
    object.
    """
    return [table.name for table in metadata.tables.values()]

def call_with_context(func, context, *args):
    """
    Check if given function has more arguments than given. Call it with context
    as last argument or without it.
    """
    return make_context_aware(func, len(args))(*args + (context,))

def write_to(f, mode):
    """Flexible writing, where f can be a filename or f object, if filename, closed after writing"""
    if hasattr(f, 'write'):
        yield f
    else:
        f = open(f, mode)
        yield f
        f.close()

def remove_last_line(self):
        """Removes the last line of the document."""
        editor = self._editor
        text_cursor = editor.textCursor()
        text_cursor.movePosition(text_cursor.End, text_cursor.MoveAnchor)
        text_cursor.select(text_cursor.LineUnderCursor)
        text_cursor.removeSelectedText()
        text_cursor.deletePreviousChar()
        editor.setTextCursor(text_cursor)

def detect(filename, include_confidence=False):
    """
    Detect the encoding of a file.

    Returns only the predicted current encoding as a string.

    If `include_confidence` is True, 
    Returns tuple containing: (str encoding, float confidence)
    """
    f = open(filename)
    detection = chardet.detect(f.read())
    f.close()
    encoding = detection.get('encoding')
    confidence = detection.get('confidence')
    if include_confidence:
        return (encoding, confidence)
    return encoding

def peekiter(iterable):
    """Return first row and also iterable with same items as original"""
    it = iter(iterable)
    one = next(it)

    def gen():
        """Generator that returns first and proxy other items from source"""
        yield one
        while True:
            yield next(it)
    return (one, gen())

def process_request(self, request, response):
        """Logs the basic endpoint requested"""
        self.logger.info('Requested: {0} {1} {2}'.format(request.method, request.relative_uri, request.content_type))

def get_current_frames():
    """Return current threads prepared for 
    further processing.
    """
    return dict(
        (thread_id, {'frame': thread2list(frame), 'time': None})
        for thread_id, frame in sys._current_frames().items()
    )

def select_random(engine, table_or_columns, limit=5):
    """
    Randomly select some rows from table.
    """
    s = select(table_or_columns).order_by(func.random()).limit(limit)
    return engine.execute(s).fetchall()

def to_dotfile(G: nx.DiGraph, filename: str):
    """ Output a networkx graph to a DOT file. """
    A = to_agraph(G)
    A.write(filename)

def delete(self, id):
        """
        Deletes an "object" (line, triangle, image, etc) from the drawing.

        :param int id:
            The id of the object.
        """
        if id in self._images.keys():
            del self._images[id]
        self.tk.delete(id)

def update(self):
        """Update all visuals in the attached canvas."""
        if not self.canvas:
            return
        for visual in self.canvas.visuals:
            self.update_program(visual.program)
        self.canvas.update()

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

def is_sqlatype_string(coltype: Union[TypeEngine, VisitableType]) -> bool:
    """
    Is the SQLAlchemy column type a string type?
    """
    coltype = _coltype_to_typeengine(coltype)
    return isinstance(coltype, sqltypes.String)

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

def SetValue(self, row, col, value):
        """
        Set value in the pandas DataFrame
        """
        self.dataframe.iloc[row, col] = value

def has_parent(self, term):
        """Return True if this GO object has a parent GO ID."""
        for parent in self.parents:
            if parent.item_id == term or parent.has_parent(term):
                return True
        return False

def inpaint(self):
        """ Replace masked-out elements in an array using an iterative image inpainting algorithm. """

        import inpaint
        filled = inpaint.replace_nans(np.ma.filled(self.raster_data, np.NAN).astype(np.float32), 3, 0.01, 2)
        self.raster_data = np.ma.masked_invalid(filled)

def parse(source, remove_comments=True, **kw):
    """Thin wrapper around ElementTree.parse"""
    return ElementTree.parse(source, SourceLineParser(), **kw)

def readline(self):
        """Get the next line including the newline or '' on EOF."""
        self.lineno += 1
        if self._buffer:
            return self._buffer.pop()
        else:
            return self.input.readline()

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

def normalize(name):
    """Normalize name for the Statsd convention"""

    # Name should not contain some specials chars (issue #1068)
    ret = name.replace(':', '')
    ret = ret.replace('%', '')
    ret = ret.replace(' ', '_')

    return ret

def _get_indent_length(line):
    """Return the length of the indentation on the given token's line."""
    result = 0
    for char in line:
        if char == " ":
            result += 1
        elif char == "\t":
            result += _TAB_LENGTH
        else:
            break
    return result

def word_to_id(self, word):
        """Returns the integer word id of a word string."""
        if word in self.vocab:
            return self.vocab[word]
        else:
            return self.unk_id

def activate_subplot(numPlot):
    """Make subplot *numPlot* active on the canvas.

    Use this if a simple ``subplot(numRows, numCols, numPlot)``
    overwrites the subplot instead of activating it.
    """
    # see http://www.mail-archive.com/matplotlib-users@lists.sourceforge.net/msg07156.html
    from pylab import gcf, axes
    numPlot -= 1  # index is 0-based, plots are 1-based
    return axes(gcf().get_axes()[numPlot])

def index(self, item):
        """ Not recommended for use on large lists due to time
            complexity, but it works

            -> #int list index of @item
        """
        for i, x in enumerate(self.iter()):
            if x == item:
                return i
        return None

def score_small_straight_yatzy(dice: List[int]) -> int:
    """
    Small straight scoring according to yatzy rules
    """
    dice_set = set(dice)
    if _are_two_sets_equal({1, 2, 3, 4, 5}, dice_set):
        return sum(dice)
    return 0

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

def _series_col_letter(self, series):
        """
        The letter of the Excel worksheet column in which the data for a
        series appears.
        """
        column_number = 1 + series.categories.depth + series.index
        return self._column_reference(column_number)

def parse_command_args():
    """Command line parser."""
    parser = argparse.ArgumentParser(description='Register PB devices.')
    parser.add_argument('num_pb', type=int,
                        help='Number of PBs devices to register.')
    return parser.parse_args()

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

def __dir__(self):
        u"""Returns a list of children and available helper methods."""
        return sorted(self.keys() | {m for m in dir(self.__class__) if m.startswith('to_')})

def values(self):
        """Gets the user enter max and min values of where the 
        raster points should appear on the y-axis

        :returns: (float, float) -- (min, max) y-values to bound the raster plot by
        """
        lower = float(self.lowerSpnbx.value())
        upper = float(self.upperSpnbx.value())
        return (lower, upper)

def remove_trailing_string(content, trailing):
    """
    Strip trailing component `trailing` from `content` if it exists.
    Used when generating names from view classes.
    """
    if content.endswith(trailing) and content != trailing:
        return content[:-len(trailing)]
    return content

def fill_document(doc):
    """Add a section, a subsection and some text to the document.

    :param doc: the document
    :type doc: :class:`pylatex.document.Document` instance
    """
    with doc.create(Section('A section')):
        doc.append('Some regular text and some ')
        doc.append(italic('italic text. '))

        with doc.create(Subsection('A subsection')):
            doc.append('Also some crazy characters: $&#{}')

def command_py2to3(args):
    """
    Apply '2to3' tool (Python2 to Python3 conversion tool) to Python sources.
    """
    from lib2to3.main import main
    sys.exit(main("lib2to3.fixes", args=args.sources))

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

def branches(self):
        """All branches in a list"""
        result = self.git(self.default + ['branch', '-a', '--no-color'])
        return [l.strip(' *\n') for l in result.split('\n') if l.strip(' *\n')]

def end(self):
        """End of the Glances server session."""
        if not self.args.disable_autodiscover:
            self.autodiscover_client.close()
        self.server.end()

def _full_analysis_mp_alias(br_obj, analysis_set, output_directory, unique_name, verbose, quick_plots):
    """
    Alias for instance method that allows the method to be called in a
    multiprocessing pool. Needed as multiprocessing does not otherwise work
    on object instance methods.
    """
    return (br_obj, unique_name, br_obj.full_analysis(analysis_set, output_directory, verbose = verbose, compile_pdf = verbose, quick_plots = quick_plots))

def gaussian_distribution(mean, stdev, num_pts=50):
    """ get an x and y numpy.ndarray that spans the +/- 4
    standard deviation range of a gaussian distribution with
    a given mean and standard deviation. useful for plotting

    Parameters
    ----------
    mean : float
        the mean of the distribution
    stdev : float
        the standard deviation of the distribution
    num_pts : int
        the number of points in the returned ndarrays.
        Default is 50

    Returns
    -------
    x : numpy.ndarray
        the x-values of the distribution
    y : numpy.ndarray
        the y-values of the distribution

    """
    xstart = mean - (4.0 * stdev)
    xend = mean + (4.0 * stdev)
    x = np.linspace(xstart,xend,num_pts)
    y = (1.0/np.sqrt(2.0*np.pi*stdev*stdev)) * np.exp(-1.0 * ((x - mean)**2)/(2.0*stdev*stdev))
    return x,y

def next(self):
        """Get the next value in the page."""
        item = six.next(self._item_iter)
        result = self._item_to_value(self._parent, item)
        # Since we've successfully got the next value from the
        # iterator, we update the number of remaining.
        self._remaining -= 1
        return result

def ratelimit_remaining(self):
        """Number of requests before GitHub imposes a ratelimit.

        :returns: int
        """
        json = self._json(self._get(self._github_url + '/rate_limit'), 200)
        core = json.get('resources', {}).get('core', {})
        self._remaining = core.get('remaining', 0)
        return self._remaining

def reduce_json(data):
    """Reduce a JSON object"""
    return reduce(lambda x, y: int(x) + int(y), data.values())

def separator(self, menu=None):
        """Add a separator"""
        self.gui.get_menu(menu or self.menu).addSeparator()

def test_replace_colon():
    """py.test for replace_colon"""
    data = (("zone:aap", '@', "zone@aap"),# s, r, replaced
    )    
    for s, r, replaced in data:
        result = replace_colon(s, r)
        assert result == replaced

def get_object_as_string(obj):
    """
    Converts any object to JSON-like readable format, ready to be printed for debugging purposes
    :param obj: Any object
    :return: string
    """
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return '\r\n\;'.join([get_object_as_string(item) for item in obj])
    attrs = vars(obj)
    as_string = ', '.join("%s: %s" % item for item in attrs.items())
    return as_string

def POST(self, *args, **kwargs):
        """ POST request """
        return self._handle_api(self.API_POST, args, kwargs)

def matching_line(lines, keyword):
    """ Returns the first matching line in a list of lines.
    @see match()
    """
    for line in lines:
        matching = match(line,keyword)
        if matching != None:
            return matching
    return None

def subscribe(self, handler):
        """Adds a new event handler."""
        assert callable(handler), "Invalid handler %s" % handler
        self.handlers.append(handler)

def get_host_power_status(self):
        """Request the power state of the server.

        :returns: Power State of the server, 'ON' or 'OFF'
        :raises: IloError, on an error from iLO.
        """
        sushy_system = self._get_sushy_system(PROLIANT_SYSTEM_ID)
        return GET_POWER_STATE_MAP.get(sushy_system.power_state)

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

def _quit(self, *args):
        """ quit crash """
        self.logger.warn('Bye!')
        sys.exit(self.exit())

def random_int(maximum_value):
	""" Random generator (PyCrypto getrandbits wrapper). The result is a non-negative value.

	:param maximum_value: maximum integer value
	:return: int
	"""
	if maximum_value == 0:
		return 0
	elif maximum_value == 1:
		return random_bits(1)

	bits = math.floor(math.log2(maximum_value))
	result = random_bits(bits) + random_int(maximum_value - ((2 ** bits) - 1))
	return result

def _open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None,
          closefd=True, opener=None, *, loop=None, executor=None):
    """Open an asyncio file."""
    if loop is None:
        loop = asyncio.get_event_loop()
    cb = partial(sync_open, file, mode=mode, buffering=buffering,
                 encoding=encoding, errors=errors, newline=newline,
                 closefd=closefd, opener=opener)
    f = yield from loop.run_in_executor(executor, cb)

    return wrap(f, loop=loop, executor=executor)

def add_exec_permission_to(target_file):
    """Add executable permissions to the file

    :param target_file: the target file whose permission to be changed
    """
    mode = os.stat(target_file).st_mode
    os.chmod(target_file, mode | stat.S_IXUSR)

def do_restart(self, line):
        """Request that the Outstation perform a cold restart. Command syntax is: restart"""
        self.application.master.Restart(opendnp3.RestartType.COLD, restart_callback)

def fileModifiedTimestamp(fname):
    """return "YYYY-MM-DD" when the file was modified."""
    modifiedTime=os.path.getmtime(fname)
    stamp=time.strftime('%Y-%m-%d', time.localtime(modifiedTime))
    return stamp

def get_readline_tail(self, n=10):
        """Get the last n items in readline history."""
        end = self.shell.readline.get_current_history_length() + 1
        start = max(end-n, 1)
        ghi = self.shell.readline.get_history_item
        return [ghi(x) for x in range(start, end)]

def from_file_url(url):
    """ Convert from file:// url to file path
    """
    if url.startswith('file://'):
        url = url[len('file://'):].replace('/', os.path.sep)

    return url

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

def flatpages_link_list(request):
    """
    Returns a HttpResponse whose content is a Javascript file representing a
    list of links to flatpages.
    """
    from django.contrib.flatpages.models import FlatPage
    link_list = [(page.title, page.url) for page in FlatPage.objects.all()]
    return render_to_link_list(link_list)

def get_case_insensitive_dict_key(d: Dict, k: str) -> Optional[str]:
    """
    Within the dictionary ``d``, find a key that matches (in case-insensitive
    fashion) the key ``k``, and return it (or ``None`` if there isn't one).
    """
    for key in d.keys():
        if k.lower() == key.lower():
            return key
    return None

def normalize(numbers):
    """Multiply each number by a constant such that the sum is 1.0
    >>> normalize([1,2,1])
    [0.25, 0.5, 0.25]
    """
    total = float(sum(numbers))
    return [n / total for n in numbers]

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

def random_str(size=10):
    """
    create random string of selected size

    :param size: int, length of the string
    :return: the string
    """
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(size))

def resetScale(self):
        """Resets the scale on this image. Correctly aligns time scale, undoes manual scaling"""
        self.img.scale(1./self.imgScale[0], 1./self.imgScale[1])
        self.imgScale = (1.,1.)

def _fast_read(self, infile):
        """Function for fast reading from sensor files."""
        infile.seek(0)
        return(int(infile.read().decode().strip()))

def post_tweet(user_id, message, additional_params={}):
    """
    Helper function to post a tweet 
    """
    url = "https://api.twitter.com/1.1/statuses/update.json"    
    params = { "status" : message }
    params.update(additional_params)
    r = make_twitter_request(url, user_id, params, request_type='POST')
    print (r.text)
    return "Successfully posted a tweet {}".format(message)

def make_lambda(call):
    """Wrap an AST Call node to lambda expression node.
    call: ast.Call node
    """
    empty_args = ast.arguments(args=[], vararg=None, kwarg=None, defaults=[])
    return ast.Lambda(args=empty_args, body=call)

def rndstr(size=16):
    """
    Returns a string of random ascii characters or digits

    :param size: The length of the string
    :return: string
    """
    _basech = string.ascii_letters + string.digits
    return "".join([rnd.choice(_basech) for _ in range(size)])

def set(self):
        """Set the color as current OpenGL color
        """
        glColor4f(self.r, self.g, self.b, self.a)

def find_elements_by_id(self, id_):
        """
        Finds multiple elements by id.

        :Args:
         - id\\_ - The id of the elements to be found.

        :Returns:
         - list of WebElement - a list with elements if any was found.  An
           empty list if not

        :Usage:
            ::

                elements = driver.find_elements_by_id('foo')
        """
        return self.find_elements(by=By.ID, value=id_)

def empty(self):
        """remove all children from the widget"""
        for k in list(self.children.keys()):
            self.remove_child(self.children[k])

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

def __sort_up(self):

        """Sort the updatable objects according to ascending order"""
        if self.__do_need_sort_up:
            self.__up_objects.sort(key=cmp_to_key(self.__up_cmp))
            self.__do_need_sort_up = False

def autozoom(self, n=None):
        """
        Auto-scales the axes to fit all the data in plot index n. If n == None,
        auto-scale everyone.
        """
        if n==None:
            for p in self.plot_widgets: p.autoRange()
        else:        self.plot_widgets[n].autoRange()

        return self

def index(self, elem):
        """Find the index of elem in the reversed iterator."""
        return _coconut.len(self._iter) - self._iter.index(elem) - 1

def json_obj_to_cursor(self, json):
        """(Deprecated) Converts a JSON object to a mongo db cursor

        :param str json: A json string
        :returns: dictionary with ObjectId type
        :rtype: dict
        """
        cursor = json_util.loads(json)
        if "id" in json:
            cursor["_id"] = ObjectId(cursor["id"])
            del cursor["id"]

        return cursor

def _squeeze(x, axis):
  """A version of squeeze that works with dynamic axis."""
  x = tf.convert_to_tensor(value=x, name='x')
  if axis is None:
    return tf.squeeze(x, axis=None)
  axis = tf.convert_to_tensor(value=axis, name='axis', dtype=tf.int32)
  axis += tf.zeros([1], dtype=axis.dtype)  # Make axis at least 1d.
  keep_axis, _ = tf.compat.v1.setdiff1d(tf.range(0, tf.rank(x)), axis)
  return tf.reshape(x, tf.gather(tf.shape(input=x), keep_axis))

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

def setup_request_sessions(self):
        """ Sets up a requests.Session object for sharing headers across API requests.
        """
        self.req_session = requests.Session()
        self.req_session.headers.update(self.headers)

def _check_conversion(key, valid_dict):
    """Check for existence of key in dict, return value or raise error"""
    if key not in valid_dict and key not in valid_dict.values():
        # Only show users the nice string values
        keys = [v for v in valid_dict.keys() if isinstance(v, string_types)]
        raise ValueError('value must be one of %s, not %s' % (keys, key))
    return valid_dict[key] if key in valid_dict else key

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

def qualified_name_import(cls):
    """Full name of a class, including the module. Like qualified_class_name, but when you already have a class """

    parts = qualified_name(cls).split('.')

    return "from {} import {}".format('.'.join(parts[:-1]), parts[-1])

def autoscan():
    """autoscan will check all of the serial ports to see if they have
       a matching VID:PID for a MicroPython board.
    """
    for port in serial.tools.list_ports.comports():
        if is_micropython_usb_device(port):
            connect_serial(port[0])

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, LegipyModel):
        return obj.to_json()
    elif isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    raise TypeError("Type {0} not serializable".format(repr(type(obj))))

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

def _check_graphviz_available(output_format):
    """check if we need graphviz for different output format"""
    try:
        subprocess.call(["dot", "-V"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError:
        print(
            "The output format '%s' is currently not available.\n"
            "Please install 'Graphviz' to have other output formats "
            "than 'dot' or 'vcg'." % output_format
        )
        sys.exit(32)

def GetPythonLibraryDirectoryPath():
  """Retrieves the Python library directory path."""
  path = sysconfig.get_python_lib(True)
  _, _, path = path.rpartition(sysconfig.PREFIX)

  if path.startswith(os.sep):
    path = path[1:]

  return path

def partition_all(n, iterable):
    """Partition a list into equally sized pieces, including last smaller parts
    http://stackoverflow.com/questions/5129102/python-equivalent-to-clojures-partition-all
    """
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            break
        yield chunk

def format_time(timestamp):
    """Formats timestamp to human readable format"""
    format_string = '%Y_%m_%d_%Hh%Mm%Ss'
    formatted_time = datetime.datetime.fromtimestamp(timestamp).strftime(format_string)
    return formatted_time

async def enter_captcha(self, url: str, sid: str) -> str:
        """
        Override this method for processing captcha.

        :param url: link to captcha image
        :param sid: captcha id. I do not know why pass here but may be useful
        :return captcha value
        """
        raise VkCaptchaNeeded(url, sid)

def send(r, stream=False):
    """Just sends the request using its send method and returns its response.  """
    r.send(stream=stream)
    return r.response

def bbox(self):
        """
        The minimal `~photutils.aperture.BoundingBox` for the cutout
        region with respect to the original (large) image.
        """

        return BoundingBox(self.slices[1].start, self.slices[1].stop,
                           self.slices[0].start, self.slices[0].stop)

def get(s, delimiter='', format="diacritical"):
    """Return pinyin of string, the string must be unicode
    """
    return delimiter.join(_pinyin_generator(u(s), format=format))

def iso_string_to_python_datetime(
        isostring: str) -> Optional[datetime.datetime]:
    """
    Takes an ISO-8601 string and returns a ``datetime``.
    """
    if not isostring:
        return None  # if you parse() an empty string, you get today's date
    return dateutil.parser.parse(isostring)

def required_attributes(element, *attributes):
    """Check element for required attributes. Raise ``NotValidXmlException`` on error.

    :param element: ElementTree element
    :param attributes: list of attributes names to check
    :raises NotValidXmlException: if some argument is missing
    """
    if not reduce(lambda still_valid, param: still_valid and param in element.attrib, attributes, True):
        raise NotValidXmlException(msg_err_missing_attributes(element.tag, *attributes))

def get_next_weekday(self, including_today=False):
        """Gets next week day

        :param including_today: If today is sunday and requesting next sunday
        :return: Date of next monday, tuesday ..
        """
        weekday = self.date_time.weekday()
        return Weekday.get_next(weekday, including_today=including_today)

def _write_pidfile(pidfile):
    """ Write file with current process ID.
    """
    pid = str(os.getpid())
    handle = open(pidfile, 'w')
    try:
        handle.write("%s\n" % pid)
    finally:
        handle.close()

def SchemaValidate(self, xsd):
        """Use W3C XSD schema to validate the document as it is
          processed. Activation is only possible before the first
          Read(). If @xsd is None, then XML Schema validation is
           deactivated. """
        ret = libxml2mod.xmlTextReaderSchemaValidate(self._o, xsd)
        return ret

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

def storeByteArray(self, context, page, len, data, returnError):
        """please override"""
        returnError.contents.value = self.IllegalStateError
        raise NotImplementedError("You must override this method.")

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

def format_header_cell(val):
    """
    Formats given header column. This involves changing '_Px_' to '(', '_xP_' to ')' and
    all other '_' to spaces.
    """
    return re.sub('_', ' ', re.sub(r'(_Px_)', '(', re.sub(r'(_xP_)', ')', str(val) )))

def process_module(self, module):
        """inspect the source file to find encoding problem"""
        if module.file_encoding:
            encoding = module.file_encoding
        else:
            encoding = "ascii"

        with module.stream() as stream:
            for lineno, line in enumerate(stream):
                self._check_encoding(lineno + 1, line, encoding)

def Distance(lat1, lon1, lat2, lon2):
    """Get distance between pairs of lat-lon points"""

    az12, az21, dist = wgs84_geod.inv(lon1, lat1, lon2, lat2)
    return az21, dist

def dot_v3(v, w):
    """Return the dotproduct of two vectors."""

    return sum([x * y for x, y in zip(v, w)])

def deduplicate(list_object):
    """Rebuild `list_object` removing duplicated and keeping order"""
    new = []
    for item in list_object:
        if item not in new:
            new.append(item)
    return new

def argmax(l,f=None):
    """http://stackoverflow.com/questions/5098580/implementing-argmax-in-python"""
    if f:
        l = [f(i) for i in l]
    return max(enumerate(l), key=lambda x:x[1])[0]

def angle(vec1, vec2):
    """Returns the angle between two vectors"""
    dot_vec = dot(vec1, vec2)
    mag1 = vec1.length()
    mag2 = vec2.length()
    result = dot_vec / (mag1 * mag2)
    return math.acos(result)

def nmse(a, b):
    """Returns the normalized mean square error of a and b
    """
    return np.square(a - b).mean() / (a.mean() * b.mean())

def connect(self):
        """
        Connects to publisher
        """
        self.client = redis.Redis(
            host=self.host, port=self.port, password=self.password)

def hide(self):
        """Hides the main window of the terminal and sets the visible
        flag to False.
        """
        if not HidePrevention(self.window).may_hide():
            return
        self.hidden = True
        self.get_widget('window-root').unstick()
        self.window.hide()

def _rndPointDisposition(dx, dy):
        """Return random disposition point."""
        x = int(random.uniform(-dx, dx))
        y = int(random.uniform(-dy, dy))
        return (x, y)

def exit_and_fail(self, msg=None, out=None):
    """Exits the runtime with a nonzero exit code, indicating failure.

    :param msg: A string message to print to stderr or another custom file desciptor before exiting.
                (Optional)
    :param out: The file descriptor to emit `msg` to. (Optional)
    """
    self.exit(result=PANTS_FAILED_EXIT_CODE, msg=msg, out=out)

def comment (self, s, **args):
        """Write DOT comment."""
        self.write(u"// ")
        self.writeln(s=s, **args)

def page_guiref(arg_s=None):
    """Show a basic reference about the GUI Console."""
    from IPython.core import page
    page.page(gui_reference, auto_html=True)

def __str__(self):
    """Returns a pretty-printed string for this object."""
    return 'Output name: "%s" watts: %d type: "%s" id: %d' % (
        self._name, self._watts, self._output_type, self._integration_id)

def main():
    """
    Commandline interface to average parameters.
    """
    setup_main_logger(console=True, file_logging=False)
    params = argparse.ArgumentParser(description="Averages parameters from multiple models.")
    arguments.add_average_args(params)
    args = params.parse_args()
    average_parameters(args)

def deserialize_ndarray_npy(d):
    """
    Deserializes a JSONified :obj:`numpy.ndarray` that was created using numpy's
    :obj:`save` function.

    Args:
        d (:obj:`dict`): A dictionary representation of an :obj:`ndarray` object, created
            using :obj:`numpy.save`.

    Returns:
        An :obj:`ndarray` object.
    """
    with io.BytesIO() as f:
        f.write(json.loads(d['npy']).encode('latin-1'))
        f.seek(0)
        return np.load(f)

def interpolate_logscale_single(start, end, coefficient):
    """ Cosine interpolation """
    return np.exp(np.log(start) + (np.log(end) - np.log(start)) * coefficient)

def s2b(s):
    """
    String to binary.
    """
    ret = []
    for c in s:
        ret.append(bin(ord(c))[2:].zfill(8))
    return "".join(ret)

def measure_string(self, text, fontname, fontsize, encoding=0):
        """Measure length of a string for a Base14 font."""
        return _fitz.Tools_measure_string(self, text, fontname, fontsize, encoding)

def dedupFasta(reads):
    """
    Remove sequence duplicates (based on sequence) from FASTA.

    @param reads: a C{dark.reads.Reads} instance.
    @return: a generator of C{dark.reads.Read} instances with no duplicates.
    """
    seen = set()
    add = seen.add
    for read in reads:
        hash_ = md5(read.sequence.encode('UTF-8')).digest()
        if hash_ not in seen:
            add(hash_)
            yield read

def get_distance(F, x):
    """Helper function for margin-based loss. Return a distance matrix given a matrix."""
    n = x.shape[0]

    square = F.sum(x ** 2.0, axis=1, keepdims=True)
    distance_square = square + square.transpose() - (2.0 * F.dot(x, x.transpose()))

    # Adding identity to make sqrt work.
    return F.sqrt(distance_square + F.array(np.identity(n)))

def writefile(openedfile, newcontents):
    """Set the contents of a file."""
    openedfile.seek(0)
    openedfile.truncate()
    openedfile.write(newcontents)

def clip_image(image, clip_min, clip_max):
  """ Clip an image, or an image batch, with upper and lower threshold. """
  return np.minimum(np.maximum(clip_min, image), clip_max)

def nb_to_python(nb_path):
    """convert notebook to python script"""
    exporter = python.PythonExporter()
    output, resources = exporter.from_filename(nb_path)
    return output

def all_versions(req):
    """Get all versions of req from PyPI."""
    import requests
    url = "https://pypi.python.org/pypi/" + req + "/json"
    return tuple(requests.get(url).json()["releases"].keys())

def guess_file_type(kind, filepath=None, youtube_id=None, web_url=None, encoding=None):
    """ guess_file_class: determines what file the content is
        Args:
            filepath (str): filepath of file to check
        Returns: string indicating file's class
    """
    if youtube_id:
        return FileTypes.YOUTUBE_VIDEO_FILE
    elif web_url:
        return FileTypes.WEB_VIDEO_FILE
    elif encoding:
        return FileTypes.BASE64_FILE
    else:
        ext = os.path.splitext(filepath)[1][1:].lower()
        if kind in FILE_TYPE_MAPPING and ext in FILE_TYPE_MAPPING[kind]:
            return FILE_TYPE_MAPPING[kind][ext]
    return None

def write(url, content, **args):
    """Put an object into a ftps URL."""
    with FTPSResource(url, **args) as resource:
        resource.write(content)

def str_dict(some_dict):
    """Convert dict of ascii str/unicode to dict of str, if necessary"""
    return {str(k): str(v) for k, v in some_dict.items()}

def with_tz(request):
    """
    Get the time with TZ enabled

    """
    
    dt = datetime.now() 
    t = Template('{% load tz %}{% localtime on %}{% get_current_timezone as TIME_ZONE %}{{ TIME_ZONE }}{% endlocaltime %}') 
    c = RequestContext(request)
    response = t.render(c)
    return HttpResponse(response)

def update_screen(self):
        """Refresh the screen. You don't need to override this except to update only small portins of the screen."""
        self.clock.tick(self.FPS)
        pygame.display.update()

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

def str_is_well_formed(xml_str):
    """
  Args:
    xml_str : str
      DataONE API XML doc.

  Returns:
    bool: **True** if XML doc is well formed.
  """
    try:
        str_to_etree(xml_str)
    except xml.etree.ElementTree.ParseError:
        return False
    else:
        return True

def _is_already_configured(configuration_details):
    """Returns `True` when alias already in shell config."""
    path = Path(configuration_details.path).expanduser()
    with path.open('r') as shell_config:
        return configuration_details.content in shell_config.read()

def export_context(cls, context):
		""" Export the specified context to be capable context transferring

		:param context: context to export
		:return: tuple
		"""
		if context is None:
			return
		result = [(x.context_name(), x.context_value()) for x in context]
		result.reverse()
		return tuple(result)

def normalize_array(lst):
    """Normalizes list

    :param lst: Array of floats
    :return: Normalized (in [0, 1]) input array
    """
    np_arr = np.array(lst)
    x_normalized = np_arr / np_arr.max(axis=0)
    return list(x_normalized)

def hex_to_rgb(h):
    """ Returns 0 to 1 rgb from a hex list or tuple """
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255. for i in (0, 2 ,4))

def get_number(s, cast=int):
    """
    Try to get a number out of a string, and cast it.
    """
    import string
    d = "".join(x for x in str(s) if x in string.digits)
    return cast(d)

def c2f(r, i, ctype_name):
    """
    Convert strings to complex number instance with specified numpy type.
    """

    ftype = c2f_dict[ctype_name]
    return np.typeDict[ctype_name](ftype(r) + 1j * ftype(i))

def validate_stringlist(s):
    """Validate a list of strings

    Parameters
    ----------
    val: iterable of strings

    Returns
    -------
    list
        list of str

    Raises
    ------
    ValueError"""
    if isinstance(s, six.string_types):
        return [six.text_type(v.strip()) for v in s.split(',') if v.strip()]
    else:
        try:
            return list(map(validate_str, s))
        except TypeError as e:
            raise ValueError(e.message)

def delete_cell(self,  key):
        """Deletes key cell"""

        try:
            self.code_array.pop(key)

        except KeyError:
            pass

        self.grid.code_array.result_cache.clear()

def contains_case_insensitive(adict, akey):
    """Check if key is in adict. The search is case insensitive."""
    for key in adict:
        if key.lower() == akey.lower():
            return True
    return False

def html_to_text(content):
    """ Converts html content to plain text """
    text = None
    h2t = html2text.HTML2Text()
    h2t.ignore_links = False
    text = h2t.handle(content)
    return text

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

def _parse_boolean(value, default=False):
    """
    Attempt to cast *value* into a bool, returning *default* if it fails.
    """
    if value is None:
        return default
    try:
        return bool(value)
    except ValueError:
        return default

def multiply(self, number):
        """Return a Vector as the product of the vector and a real number."""
        return self.from_list([x * number for x in self.to_list()])

def graphql_queries_to_json(*queries):
    """
    Queries should be a list of GraphQL objects
    """
    rtn = {}
    for i, query in enumerate(queries):
        rtn["q{}".format(i)] = query.value
    return json.dumps(rtn)

def get_object_or_child_by_type(self, *types):
        """ Get object if child already been read or get child.

        Use this method for fast access to objects in case of static configurations.

        :param types: requested object types.
        :return: all children of the specified types.
        """

        objects = self.get_objects_or_children_by_type(*types)
        return objects[0] if any(objects) else None

def downcaseTokens(s,l,t):
    """Helper parse action to convert tokens to lower case."""
    return [ tt.lower() for tt in map(_ustr,t) ]

def get_prep_value(self, value):
        """Convert JSON object to a string"""
        if self.null and value is None:
            return None
        return json.dumps(value, **self.dump_kwargs)

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

def command(name, mode):
    """ Label a method as a command with name. """
    def decorator(fn):
        commands[name] = fn.__name__
        _Client._addMethod(fn.__name__, name, mode)
        return fn
    return decorator

def _numbers_units(N):
    """
    >>> _numbers_units(45)
    '123456789012345678901234567890123456789012345'
    """
    lst = range(1, N + 1)
    return "".join(list(map(lambda i: str(i % 10), lst)))

def get_tensor_device(self, tensor_name):
    """The device of a tensor.

    Note that only tf tensors have device assignments.

    Args:
      tensor_name: a string, name of a tensor in the graph.

    Returns:
      a string or None, representing the device name.
    """
    tensor = self._name_to_tensor(tensor_name)
    if isinstance(tensor, tf.Tensor):
      return tensor.device
    else:  # mtf.Tensor
      return None

def _to_base_type(self, msg):
    """Convert a Message value to a Model instance (entity)."""
    ent = _message_to_entity(msg, self._modelclass)
    ent.blob_ = self._protocol_impl.encode_message(msg)
    return ent

def lspearmanr(x,y):
    """
Calculates a Spearman rank-order correlation coefficient.  Taken
from Heiman's Basic Statistics for the Behav. Sci (1st), p.192.

Usage:   lspearmanr(x,y)      where x and y are equal-length lists
Returns: Spearman's r, two-tailed p-value
"""
    TINY = 1e-30
    if len(x) != len(y):
        raise ValueError('Input values not paired in spearmanr.  Aborting.')
    n = len(x)
    rankx = rankdata(x)
    ranky = rankdata(y)
    dsq = sumdiffsquared(rankx,ranky)
    rs = 1 - 6*dsq / float(n*(n**2-1))
    t = rs * math.sqrt((n-2) / ((rs+1.0)*(1.0-rs)))
    df = n-2
    probrs = betai(0.5*df,0.5,df/(df+t*t))  # t already a float
# probability values for rs are from part 2 of the spearman function in
# Numerical Recipies, p.510.  They are close to tables, but not exact. (?)
    return rs, probrs

def lint(ctx: click.Context, amend: bool = False, stage: bool = False):
    """
    Runs all linters

    Args:
        ctx: click context
        amend: whether or not to commit results
        stage: whether or not to stage changes
    """
    _lint(ctx, amend, stage)

def is_managed():
    """
    Check if a Django project is being managed with ``manage.py`` or
    ``django-admin`` scripts

    :return: Check result
    :rtype: bool
    """
    for item in sys.argv:
        if re.search(r'manage.py|django-admin|django', item) is not None:
            return True
    return False

def log_to_json(log):
    """Convert a log record into a list of strings"""
    return [log.timestamp.isoformat()[:22],
            log.level, log.process, log.message]

def _merge_maps(m1, m2):
    """merge two Mapping objects, keeping the type of the first mapping"""
    return type(m1)(chain(m1.items(), m2.items()))

def cov_to_correlation(cov):
    """Compute the correlation matrix given the covariance matrix.

    Parameters
    ----------
    cov : `~numpy.ndarray`
        N x N matrix of covariances among N parameters.

    Returns
    -------
    corr : `~numpy.ndarray`
        N x N matrix of correlations among N parameters.
    """
    err = np.sqrt(np.diag(cov))
    errinv = np.ones_like(err) * np.nan
    m = np.isfinite(err) & (err != 0)
    errinv[m] = 1. / err[m]
    corr = np.array(cov)
    return corr * np.outer(errinv, errinv)

def _push_render(self):
        """Render the plot with bokeh.io and push to notebook.
        """
        bokeh.io.push_notebook(handle=self.handle)
        self.last_update = time.time()

def schemaParse(self):
        """parse a schema definition resource and build an internal
           XML Shema struture which can be used to validate instances. """
        ret = libxml2mod.xmlSchemaParse(self._o)
        if ret is None:raise parserError('xmlSchemaParse() failed')
        __tmp = Schema(_obj=ret)
        return __tmp

def partition(a, sz): 
    """splits iterables a in equal parts of size sz"""
    return [a[i:i+sz] for i in range(0, len(a), sz)]

def sortlevel(self, level=None, ascending=True, sort_remaining=None):
        """
        For internal compatibility with with the Index API.

        Sort the Index. This is for compat with MultiIndex

        Parameters
        ----------
        ascending : boolean, default True
            False to sort in descending order

        level, sort_remaining are compat parameters

        Returns
        -------
        Index
        """
        return self.sort_values(return_indexer=True, ascending=ascending)

def manhattan(h1, h2): # # 7 us @array, 31 us @list \w 100 bins
    r"""
    Equal to Minowski distance with :math:`p=1`.
    
    See also
    --------
    minowski
    """
    h1, h2 = __prepare_histogram(h1, h2)
    return scipy.sum(scipy.absolute(h1 - h2))

def _validate_type_scalar(self, value):
        """ Is not a list or a dict """
        if isinstance(
            value, _int_types + (_str_type, float, date, datetime, bool)
        ):
            return True

def dict_self(self):
        """Return the self object attributes not inherited as dict."""
        return {k: v for k, v in self.__dict__.items() if k in FSM_ATTRS}

def run_time() -> timedelta:
    """

    :return:
    """

    delta = start_time if start_time else datetime.utcnow()
    return datetime.utcnow() - delta

def module_name(self):
        """
        The module where this route's view function was defined.
        """
        if not self.view_func:
            return None
        elif self._controller_cls:
            rv = inspect.getmodule(self._controller_cls).__name__
            return rv
        return inspect.getmodule(self.view_func).__name__

def load(cls, fname):
        """ Loads the dictionary from json file
        :param fname: file to load from
        :return: loaded dictionary
        """
        with open(fname) as f:
            return Config(**json.load(f))

def out(self, output, newline=True):
        """Outputs a string to the console (stdout)."""
        click.echo(output, nl=newline)

def build_list_type_validator(item_validator):
    """Return a function which validates that the value is a list of items
    which are validated using item_validator.
    """
    def validate_list_of_type(value):
        return [item_validator(item) for item in validate_list(value)]
    return validate_list_of_type

def __repr__(self) -> str:
        """Return the string representation of self."""
        return '{0}({1})'.format(type(self).__name__, repr(self.string))

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

def _download(url):
    """Downloads an URL and returns a file-like object open for reading,
    compatible with zipping.ZipFile (it has a seek() method).
    """
    fh = StringIO()

    for line in get(url):
        fh.write(line)

    fh.seek(0)
    return fh

def bitsToString(arr):
  """Returns a string representing a numpy array of 0's and 1's"""
  s = array('c','.'*len(arr))
  for i in xrange(len(arr)):
    if arr[i] == 1:
      s[i]='*'
  return s

def fsliceafter(astr, sub):
    """Return the slice after at sub in string astr"""
    findex = astr.find(sub)
    return astr[findex + len(sub):]

def get_edge_relations(graph: BELGraph) -> Mapping[Tuple[BaseEntity, BaseEntity], Set[str]]:
    """Build a dictionary of {node pair: set of edge types}."""
    return group_dict_set(
        ((u, v), d[RELATION])
        for u, v, d in graph.edges(data=True)
    )

def prettifysql(sql):
    """Returns a prettified version of the SQL as a list of lines to help
    in creating a useful diff between two SQL statements."""
    pretty = []
    for line in sql.split('\n'):
        pretty.extend(["%s,\n" % x for x in line.split(',')])
    return pretty

def de_blank(val):
    """Remove blank elements in `val` and return `ret`"""
    ret = list(val)
    if type(val) == list:
        for idx, item in enumerate(val):
            if item.strip() == '':
                ret.remove(item)
            else:
                ret[idx] = item.strip()
    return ret

def cors_header(func):
    """ @cors_header decorator adds CORS headers """

    @wraps(func)
    def wrapper(self, request, *args, **kwargs):
        res = func(self, request, *args, **kwargs)
        request.setHeader('Access-Control-Allow-Origin', '*')
        request.setHeader('Access-Control-Allow-Headers', 'Content-Type, Access-Control-Allow-Headers, Authorization, X-Requested-With')
        return res

    return wrapper

def version():
    """
    View the current version of the CLI.
    """
    import pkg_resources
    version = pkg_resources.require(PROJECT_NAME)[0].version
    floyd_logger.info(version)

def plot_epsilon_residuals(self):
        """Plots the epsilon residuals for the variogram fit."""
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(range(self.epsilon.size), self.epsilon, c='k', marker='*')
        ax.axhline(y=0.0)
        plt.show()

def get_lines(handle, line):
    """
    Get zero-indexed line from an open file-like.
    """
    for i, l in enumerate(handle):
        if i == line:
            return l

def EvalPoissonPmf(k, lam):
    """Computes the Poisson PMF.

    k: number of events
    lam: parameter lambda in events per unit time

    returns: float probability
    """
    # don't use the scipy function (yet).  for lam=0 it returns NaN;
    # should be 0.0
    # return scipy.stats.poisson.pmf(k, lam)

    return lam ** k * math.exp(-lam) / math.factorial(k)

def set_json_item(key, value):
    """ manipulate json data on the fly
    """
    data = get_json()
    data[key] = value

    request = get_request()
    request["BODY"] = json.dumps(data)

def read_string(cls, string):
        """Decodes a given bencoded string or bytestring.

        Returns decoded structure(s).

        :param str string:
        :rtype: list
        """
        if PY3 and not isinstance(string, byte_types):
            string = string.encode()

        return cls.decode(string)

def _str_to_list(value, separator):
    """Convert a string to a list with sanitization."""
    value_list = [item.strip() for item in value.split(separator)]
    value_list_sanitized = builtins.list(filter(None, value_list))
    if len(value_list_sanitized) > 0:
        return value_list_sanitized
    else:
        raise ValueError('Invalid list variable.')

def _convert(self, image, output=None):
        """Private method for converting a single PNG image to a PDF."""
        with Image.open(image) as im:
            width, height = im.size

            co = CanvasObjects()
            co.add(CanvasImg(image, 1.0, w=width, h=height))

            return WatermarkDraw(co, tempdir=self.tempdir, pagesize=(width, height)).write(output)

def dmap(fn, record):
    """map for a directory"""
    values = (fn(v) for k, v in record.items())
    return dict(itertools.izip(record, values))

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

async def scalar(self, query, as_tuple=False):
        """Get single value from ``select()`` query, i.e. for aggregation.

        :return: result is the same as after sync ``query.scalar()`` call
        """
        query = self._swap_database(query)
        return (await scalar(query, as_tuple=as_tuple))

def safe_rmtree(directory):
  """Delete a directory if it's present. If it's not present, no-op."""
  if os.path.exists(directory):
    shutil.rmtree(directory, True)

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

def _cleanup(path: str) -> None:
    """Cleanup temporary directory."""
    if os.path.isdir(path):
        shutil.rmtree(path)

def onkeyup(self, key, keycode, ctrl, shift, alt):
        """Called when user types and releases a key. 
        The widget should be able to receive the focus in order to emit the event.
        Assign a 'tabindex' attribute to make it focusable.
        
        Args:
            key (str): the character value
            keycode (str): the numeric char code
        """
        return (key, keycode, ctrl, shift, alt)

def resample(grid, wl, flux):
    """ Resample spectrum onto desired grid """
    flux_rs = (interpolate.interp1d(wl, flux))(grid)
    return flux_rs

def set_xlimits_widgets(self, set_min=True, set_max=True):
        """Populate axis limits GUI with current plot values."""
        xmin, xmax = self.tab_plot.ax.get_xlim()
        if set_min:
            self.w.x_lo.set_text('{0}'.format(xmin))
        if set_max:
            self.w.x_hi.set_text('{0}'.format(xmax))

def run_task(func):
    """
    Decorator to wrap an async function in an event loop.
    Use for main sync interface methods.
    """

    def _wrapped(*a, **k):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(func(*a, **k))

    return _wrapped

def hash_producer(*args, **kwargs):
    """ Returns a random hash for a confirmation secret. """
    return hashlib.md5(six.text_type(uuid.uuid4()).encode('utf-8')).hexdigest()

def post(self, endpoint: str, **kwargs) -> dict:
        """HTTP POST operation to API endpoint."""

        return self._request('POST', endpoint, **kwargs)

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

def is_iterable_of_int(l):
    r""" Checks if l is iterable and contains only integral types """
    if not is_iterable(l):
        return False

    return all(is_int(value) for value in l)

def inh(table):
    """
    inverse hyperbolic sine transformation
    """
    t = []
    for i in table:
        t.append(np.ndarray.tolist(np.arcsinh(i)))
    return t

def _replace_variables(data, variables):
    """Replace the format variables in all items of data."""
    formatter = string.Formatter()
    return [formatter.vformat(item, [], variables) for item in data]

def add_device_callback(self, callback):
        """Register a callback to be invoked when a new device appears."""
        _LOGGER.debug('Added new callback %s ', callback)
        self._cb_new_device.append(callback)

def accuracy(self):
        """
        Calculates the accuracy of the tree by comparing
        the model predictions to the dataset
        (TP + TN) / (TP + TN + FP + FN) == (T / (T + F))
        """
        sub_observed = np.array([self.observed.metadata[i] for i in self.observed.arr])
        return float((self.model_predictions() == sub_observed).sum()) / self.data_size

def highlight_words(string, keywords, cls_name='highlighted'):
    """ Given an list of words, this function highlights the matched words in the given string. """

    if not keywords:
        return string
    if not string:
        return ''
    include, exclude = get_text_tokenizer(keywords)
    highlighted = highlight_text(include, string, cls_name, words=True)
    return highlighted

def replace_print(fileobj=sys.stderr):
  """Sys.out replacer, by default with stderr.

  Use it like this:
  with replace_print_with(fileobj):
    print "hello"  # writes to the file
  print "done"  # prints to stdout

  Args:
    fileobj: a file object to replace stdout.

  Yields:
    The printer.
  """
  printer = _Printer(fileobj)

  previous_stdout = sys.stdout
  sys.stdout = printer
  try:
    yield printer
  finally:
    sys.stdout = previous_stdout

def format_op_hdr():
    """
    Build the header
    """
    txt = 'Base Filename'.ljust(36) + ' '
    txt += 'Lines'.rjust(7) + ' '
    txt += 'Words'.rjust(7) + '  '
    txt += 'Unique'.ljust(8) + ''
    return txt

def start(self):
        """Create a background thread for httpd and serve 'forever'"""
        self._process = threading.Thread(target=self._background_runner)
        self._process.start()

def screen_to_latlon(self, x, y):
        """
        Return the latitude and longitude corresponding to a screen point
        :param x: screen x
        :param y: screen y
        :return: latitude and longitude at x,y
        """
        xtile = 1. * x / TILE_SIZE + self.xtile
        ytile = 1. * y / TILE_SIZE + self.ytile
        return self.num2deg(xtile, ytile, self.zoom)

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

def _is_expired_response(self, response):
        """
        Check if the response failed because of an expired access token.
        """
        if response.status_code != 401:
            return False
        challenge = response.headers.get('www-authenticate', '')
        return 'error="invalid_token"' in challenge

def state(self):
    """Returns the current LED state by querying the remote controller."""
    ev = self._query_waiters.request(self.__do_query_state)
    ev.wait(1.0)
    return self._state

def merge(database=None, directory=None, verbose=None):
    """Merge migrations into one."""
    router = get_router(directory, database, verbose)
    router.merge()

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

def from_string(cls, s):
        """Return a `Status` instance from its string representation."""
        for num, text in cls._STATUS2STR.items():
            if text == s:
                return cls(num)
        else:
            raise ValueError("Wrong string %s" % s)

def remove_namespaces(root):
    """Call this on an lxml.etree document to remove all namespaces"""
    for elem in root.getiterator():
        if not hasattr(elem.tag, 'find'):
            continue

        i = elem.tag.find('}')
        if i >= 0:
            elem.tag = elem.tag[i + 1:]

    objectify.deannotate(root, cleanup_namespaces=True)

def _get_column_types(self, data):
        """Get a list of the data types for each column in *data*."""
        columns = list(zip_longest(*data))
        return [self._get_column_type(column) for column in columns]

def _size_36():
    """ returns the rows, columns of terminal """
    from shutil import get_terminal_size
    dim = get_terminal_size()
    if isinstance(dim, list):
        return dim[0], dim[1]
    return dim.lines, dim.columns

def dtype(self):
        """Pixel data type."""
        try:
            return self.data.dtype
        except AttributeError:
            return numpy.dtype('%s%d' % (self._sample_type, self._sample_bytes))

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

def _post(self, url, params, uploads=None):
        """ Wrapper method for POST calls. """
        self._call(self.POST, url, params, uploads)

def dt_to_ts(value):
    """ If value is a datetime, convert to timestamp """
    if not isinstance(value, datetime):
        return value
    return calendar.timegm(value.utctimetuple()) + value.microsecond / 1000000.0

def hex_to_int(value):
    """
    Convert hex string like "\x0A\xE3" to 2787.
    """
    if version_info.major >= 3:
        return int.from_bytes(value, "big")
    return int(value.encode("hex"), 16)

def right_replace(string, old, new, count=1):
    """
    Right replaces ``count`` occurrences of ``old`` with ``new`` in ``string``.
    For example::

        right_replace('one_two_two', 'two', 'three') -> 'one_two_three'
    """
    if not string:
        return string
    return new.join(string.rsplit(old, count))

def reprkwargs(kwargs, sep=', ', fmt="{0!s}={1!r}"):
    """Display kwargs."""
    return sep.join(fmt.format(k, v) for k, v in kwargs.iteritems())

def map_wrap(f):
    """Wrap standard function to easily pass into 'map' processing.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper

def smooth_gaussian(image, sigma=1):
    """Returns Gaussian smoothed image.

    :param image: numpy array or :class:`jicimagelib.image.Image`
    :param sigma: standard deviation
    :returns: :class:`jicimagelib.image.Image`
    """
    return scipy.ndimage.filters.gaussian_filter(image, sigma=sigma, mode="nearest")

def skewness(data):
    """
    Returns the skewness of ``data``.
    """

    if len(data) == 0:
        return None

    num = moment(data, 3)
    denom = moment(data, 2) ** 1.5

    return num / denom if denom != 0 else 0.

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

def device_state(device_id):
    """ Get device state via HTTP GET. """
    if device_id not in devices:
        return jsonify(success=False)
    return jsonify(state=devices[device_id].state)

def _darwin_current_arch(self):
        """Add Mac OS X support."""
        if sys.platform == "darwin":
            if sys.maxsize > 2 ** 32: # 64bits.
                return platform.mac_ver()[2] # Both Darwin and Python are 64bits.
            else: # Python 32 bits
                return platform.processor()

def from_json_str(cls, json_str):
    """Convert json string representation into class instance.

    Args:
      json_str: json representation as string.

    Returns:
      New instance of the class with data loaded from json string.
    """
    return cls.from_json(json.loads(json_str, cls=JsonDecoder))

def _flush(self, buffer):
        """
        Flush the write buffers of the stream if applicable.

        Args:
            buffer (memoryview): Buffer content.
        """
        container, obj = self._client_args
        with _handle_client_exception():
            self._client.put_object(container, obj, buffer)

def fmt_duration(secs):
    """Format a duration in seconds."""
    return ' '.join(fmt.human_duration(secs, 0, precision=2, short=True).strip().split())

def stack_push(self, thing):
        """
        Push 'thing' to the stack, writing the thing to memory and adjusting the stack pointer.
        """
        # increment sp
        sp = self.regs.sp + self.arch.stack_change
        self.regs.sp = sp
        return self.memory.store(sp, thing, endness=self.arch.memory_endness)

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

def add(self, value):
        """Add the element *value* to the set."""
        if value not in self._set:
            self._set.add(value)
            self._list.add(value)

def visit_BinOp(self, node):
        """ Return type depend from both operand of the binary operation. """
        args = [self.visit(arg) for arg in (node.left, node.right)]
        return list({frozenset.union(*x) for x in itertools.product(*args)})

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

def get(cls):
    """Subsystems used outside of any task."""
    return {
      SourceRootConfig,
      Reporting,
      Reproducer,
      RunTracker,
      Changed,
      BinaryUtil.Factory,
      Subprocess.Factory
    }

def stdout_to_results(s):
    """Turns the multi-line output of a benchmark process into
    a sequence of BenchmarkResult instances."""
    results = s.strip().split('\n')
    return [BenchmarkResult(*r.split()) for r in results]

def disconnect(self):
        """Gracefully close connection to stomp server."""
        if self._connected:
            self._connected = False
            self._conn.disconnect()

def read_array(path, mmap_mode=None):
    """Read a .npy array."""
    file_ext = op.splitext(path)[1]
    if file_ext == '.npy':
        return np.load(path, mmap_mode=mmap_mode)
    raise NotImplementedError("The file extension `{}` ".format(file_ext) +
                              "is not currently supported.")

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

def get_attr(self, method_name):
        """Get attribute from the target object"""
        return self.attrs.get(method_name) or self.get_callable_attr(method_name)

def _python_rpath(self):
        """The relative path (from environment root) to python."""
        # Windows virtualenv installation installs pip to the [Ss]cripts
        # folder. Here's a simple check to support:
        if sys.platform == 'win32':
            return os.path.join('Scripts', 'python.exe')
        return os.path.join('bin', 'python')

def _index_range(self, version, symbol, from_version=None, **kwargs):
        """
        Tuple describing range to read from the ndarray - closed:open
        """
        from_index = None
        if from_version:
            from_index = from_version['up_to']
        return from_index, None

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

def is_date_type(cls):
    """Return True if the class is a date type."""
    if not isinstance(cls, type):
        return False
    return issubclass(cls, date) and not issubclass(cls, datetime)

def copy(self):
		"""Return a shallow copy."""
		return self.__class__(self.operations.copy(), self.collection, self.document)

def is_empty(self):
        """Returns True if the root node contains no child elements, no text,
        and no attributes other than **type**. Returns False if any are present."""
        non_type_attributes = [attr for attr in self.node.attrib.keys() if attr != 'type']
        return len(self.node) == 0 and len(non_type_attributes) == 0 \
            and not self.node.text and not self.node.tail

def m(name='', **kwargs):
    """
    Print out memory usage at this point in time

    http://docs.python.org/2/library/resource.html
    http://stackoverflow.com/a/15448600/5006
    http://stackoverflow.com/questions/110259/which-python-memory-profiler-is-recommended
    """
    with Reflect.context(**kwargs) as r:
        kwargs["name"] = name
        instance = M_CLASS(r, stream, **kwargs)
        instance()

def standard_deviation(numbers):
    """Return standard deviation."""
    numbers = list(numbers)
    if not numbers:
        return 0
    mean = sum(numbers) / len(numbers)
    return (sum((n - mean) ** 2 for n in numbers) /
            len(numbers)) ** .5

def isnamedtuple(obj):
    """Heuristic check if an object is a namedtuple."""
    return isinstance(obj, tuple) \
           and hasattr(obj, "_fields") \
           and hasattr(obj, "_asdict") \
           and callable(obj._asdict)

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

def _xxrange(self, start, end, step_count):
        """Generate n values between start and end."""
        _step = (end - start) / float(step_count)
        return (start + (i * _step) for i in xrange(int(step_count)))

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

def increment_frame(self):
        """Increment a frame of the animation."""
        self.current_frame += 1

        if self.current_frame >= self.end_frame:
            # Wrap back to the beginning of the animation.
            self.current_frame = 0

def _gauss(mean: int, sigma: int) -> int:
        """
        Creates a variation from a base value

        Args:
            mean: base value
            sigma: gaussian sigma

        Returns: random value

        """
        return int(random.gauss(mean, sigma))

def decode(self, bytes, raw=False):
        """decode(bytearray, raw=False) -> value

        Decodes the given bytearray according to this PrimitiveType
        definition.

        NOTE: The parameter ``raw`` is present to adhere to the
        ``decode()`` inteface, but has no effect for PrimitiveType
        definitions.
        """
        return struct.unpack(self.format, buffer(bytes))[0]

def _raise_error_if_column_exists(dataset, column_name = 'dataset',
                            dataset_variable_name = 'dataset',
                            column_name_error_message_name = 'column_name'):
    """
    Check if a column exists in an SFrame with error message.
    """
    err_msg = 'The SFrame {0} must contain the column {1}.'.format(
                                                dataset_variable_name,
                                             column_name_error_message_name)
    if column_name not in dataset.column_names():
      raise ToolkitError(str(err_msg))

def random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4())  # Convert UUID format to a Python string.
    random = random.upper()  # Make all characters uppercase.
    random = random.replace("-", "")  # Remove the UUID '-'.
    return random[0:string_length]

def dt_to_qdatetime(dt):
    """Convert a python datetime.datetime object to QDateTime

    :param dt: the datetime object
    :type dt: :class:`datetime.datetime`
    :returns: the QDateTime conversion
    :rtype: :class:`QtCore.QDateTime`
    :raises: None
    """
    return QtCore.QDateTime(QtCore.QDate(dt.year, dt.month, dt.day),
                            QtCore.QTime(dt.hour, dt.minute, dt.second))

def var_dump(*obs):
	"""
	  shows structured information of a object, list, tuple etc
	"""
	i = 0
	for x in obs:
		
		str = var_dump_output(x, 0, '  ', '\n', True)
		print (str.strip())
		
		#dump(x, 0, i, '', object)
		i += 1

def _sslobj(sock):
    """Returns the underlying PySLLSocket object with which the C extension
    functions interface.

    """
    pass
    if isinstance(sock._sslobj, _ssl._SSLSocket):
        return sock._sslobj
    else:
        return sock._sslobj._sslobj

def get_date(date):
    """
    Get the date from a value that could be a date object or a string.

    :param date: The date object or string.

    :returns: The date object.
    """
    if type(date) is str:
        return datetime.strptime(date, '%Y-%m-%d').date()
    else:
        return date

def prettyprint(d):
        """Print dicttree in Json-like format. keys are sorted
        """
        print(json.dumps(d, sort_keys=True, 
                         indent=4, separators=("," , ": ")))

def __del__(self):
        """Cleanup any active connections and free all DDEML resources."""
        if self._hConv:
            DDE.Disconnect(self._hConv)
        if self._idInst:
            DDE.Uninitialize(self._idInst)

def set_int(bytearray_, byte_index, _int):
    """
    Set value in bytearray to int
    """
    # make sure were dealing with an int
    _int = int(_int)
    _bytes = struct.unpack('2B', struct.pack('>h', _int))
    bytearray_[byte_index:byte_index + 2] = _bytes
    return bytearray_

def to_snake_case(s):
    """Converts camel-case identifiers to snake-case."""
    return re.sub('([^_A-Z])([A-Z])', lambda m: m.group(1) + '_' + m.group(2).lower(), s)

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

def _async_requests(urls):
    """
    Sends multiple non-blocking requests. Returns
    a list of responses.

    :param urls:
        List of urls
    """
    session = FuturesSession(max_workers=30)
    futures = [
        session.get(url)
        for url in urls
    ]
    return [ future.result() for future in futures ]

def check_git():
    """Check if git command is available."""
    try:
        with open(os.devnull, "wb") as devnull:
            subprocess.check_call(["git", "--version"], stdout=devnull, stderr=devnull)
    except:
        raise RuntimeError("Please make sure git is installed and on your path.")

def date_to_timestamp(date):
    """
        date to unix timestamp in milliseconds
    """
    date_tuple = date.timetuple()
    timestamp = calendar.timegm(date_tuple) * 1000
    return timestamp

def equal(list1, list2):
    """ takes flags returns indexes of True values """
    return [item1 == item2 for item1, item2 in broadcast_zip(list1, list2)]

def list_add_capitalize(l):
    """
    @type l: list
    @return: list
    """
    nl = []

    for i in l:
        nl.append(i)

        if hasattr(i, "capitalize"):
            nl.append(i.capitalize())

    return list(set(nl))

def get_attribute_name_id(attr):
    """
    Return the attribute name identifier
    """
    return attr.value.id if isinstance(attr.value, ast.Name) else None

def update(self, dictionary=None, **kwargs):
        """
        Adds/overwrites all the keys and values from the dictionary.
        """
        if not dictionary == None: kwargs.update(dictionary)
        for k in list(kwargs.keys()): self[k] = kwargs[k]

def shape_list(l,shape,dtype):
    """ Shape a list of lists into the appropriate shape and data type """
    return np.array(l, dtype=dtype).reshape(shape)

def AddAccuracy(model, softmax, label):
    """Adds an accuracy op to the model"""
    accuracy = brew.accuracy(model, [softmax, label], "accuracy")
    return accuracy

def get_just_date(self):
        """Parses just date from date-time

        :return: Just day, month and year (setting hours to 00:00:00)
        """
        return datetime.datetime(
            self.date_time.year,
            self.date_time.month,
            self.date_time.day
        )

def np_hash(a):
    """Return a hash of a NumPy array."""
    if a is None:
        return hash(None)
    # Ensure that hashes are equal whatever the ordering in memory (C or
    # Fortran)
    a = np.ascontiguousarray(a)
    # Compute the digest and return a decimal int
    return int(hashlib.sha1(a.view(a.dtype)).hexdigest(), 16)

def _uniquify(_list):
    """Remove duplicates in a list."""
    seen = set()
    result = []
    for x in _list:
        if x not in seen:
            result.append(x)
            seen.add(x)
    return result

def deprecated(operation=None):
    """
    Mark an operation deprecated.
    """
    def inner(o):
        o.deprecated = True
        return o
    return inner(operation) if operation else inner

def auto():
	"""set colouring on if STDOUT is a terminal device, off otherwise"""
	try:
		Style.enabled = False
		Style.enabled = sys.stdout.isatty()
	except (AttributeError, TypeError):
		pass

def get_trace_id_from_flask():
    """Get trace_id from flask request headers.

    :rtype: str
    :returns: TraceID in HTTP request headers.
    """
    if flask is None or not flask.request:
        return None

    header = flask.request.headers.get(_FLASK_TRACE_HEADER)

    if header is None:
        return None

    trace_id = header.split("/", 1)[0]

    return trace_id

def generate_hash(self, length=30):
        """ Generate random string of given length """
        import random, string
        chars = string.ascii_letters + string.digits
        ran = random.SystemRandom().choice
        hash = ''.join(ran(chars) for i in range(length))
        return hash

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

def fetchallfirstvalues(self, sql: str, *args) -> List[Any]:
        """Executes SQL; returns list of first values of each row."""
        rows = self.fetchall(sql, *args)
        return [row[0] for row in rows]

def copy_session(session: requests.Session) -> requests.Session:
    """Duplicates a requests.Session."""
    new = requests.Session()
    new.cookies = requests.utils.cookiejar_from_dict(requests.utils.dict_from_cookiejar(session.cookies))
    new.headers = session.headers.copy()
    return new

def advance_one_line(self):
    """Advances to next line."""

    current_line = self._current_token.line_number
    while current_line == self._current_token.line_number:
      self._current_token = ConfigParser.Token(*next(self._token_generator))

def submit(self, fn, *args, **kwargs):
        """Submit an operation"""
        corofn = asyncio.coroutine(lambda: fn(*args, **kwargs))
        return run_coroutine_threadsafe(corofn(), self.loop)

def calc_cR(Q2, sigma):
    """Returns the cR statistic for the variogram fit (see [1])."""
    return Q2 * np.exp(np.sum(np.log(sigma**2))/sigma.shape[0])

def BROADCAST_FILTER_NOT(func):
        """
        Composes the passed filters into an and-joined filter.
        """
        return lambda u, command, *args, **kwargs: not func(u, command, *args, **kwargs)

def __init__(self, stream_start):
    """Initializes a gzip member decompressor wrapper.

    Args:
      stream_start (int): offset to the compressed stream within the containing
          file object.
    """
    self._decompressor = zlib_decompressor.DeflateDecompressor()
    self.last_read = stream_start
    self.uncompressed_offset = 0
    self._compressed_data = b''

def getpass(self, prompt, default=None):
        """Provide a password prompt."""
        return click.prompt(prompt, hide_input=True, default=default)

def file_to_png(fp):
	"""Convert an image to PNG format with Pillow.
	
	:arg file-like fp: The image file.
	:rtype: bytes
	"""
	import PIL.Image # pylint: disable=import-error
	with io.BytesIO() as dest:
		PIL.Image.open(fp).save(dest, "PNG", optimize=True)
		return dest.getvalue()

def input(self, prompt, default=None, show_default=True):
        """Provide a command prompt."""
        return click.prompt(prompt, default=default, show_default=show_default)

def filechunk(f, chunksize):
    """Iterator that allow for piecemeal processing of a file."""
    while True:
        chunk = tuple(itertools.islice(f, chunksize))
        if not chunk:
            return
        yield np.loadtxt(iter(chunk), dtype=np.float64)

def Bernstein(n, k):
    """Bernstein polynomial.

    """
    coeff = binom(n, k)

    def _bpoly(x):
        return coeff * x ** k * (1 - x) ** (n - k)

    return _bpoly

def store_many(self, sql, values):
        """Abstraction over executemany method"""
        cursor = self.get_cursor()
        cursor.executemany(sql, values)
        self.conn.commit()

def start(self):
        """Start the receiver.
        """
        if not self._is_running:
            self._do_run = True
            self._thread.start()
        return self

def relative_path(path):
    """
    Return the given path relative to this file.
    """
    return os.path.join(os.path.dirname(__file__), path)

def get_methods(*objs):
    """ Return the names of all callable attributes of an object"""
    return set(
        attr
        for obj in objs
        for attr in dir(obj)
        if not attr.startswith('_') and callable(getattr(obj, attr))
    )

def puts_err(s='', newline=True, stream=STDERR):
    """Prints given string to stderr."""
    puts(s, newline, stream)

def gaussian_noise(x, severity=1):
  """Gaussian noise corruption to images.

  Args:
    x: numpy array, uncorrupted image, assumed to have uint8 pixel in [0,255].
    severity: integer, severity of corruption.

  Returns:
    numpy array, image with uint8 pixels in [0,255]. Added Gaussian noise.
  """
  c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
  x = np.array(x) / 255.
  x_clip = np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255
  return around_and_astype(x_clip)

def mouse_move_event(self, event):
        """
        Forward mouse cursor position events to the example
        """
        self.example.mouse_position_event(event.x(), event.y())

def get_kind(self, value):
        """Return the kind (type) of the attribute"""
        if isinstance(value, float):
            return 'f'
        elif isinstance(value, int):
            return 'i'
        else:
            raise ValueError("Only integer or floating point values can be stored.")

def conv_block(inputs, filters, dilation_rates_and_kernel_sizes, **kwargs):
  """A block of standard 2d convolutions."""
  return conv_block_internal(conv, inputs, filters,
                             dilation_rates_and_kernel_sizes, **kwargs)

def nlargest(self, n=None):
		"""List the n most common elements and their counts.

		List is from the most
		common to the least.  If n is None, the list all element counts.

		Run time should be O(m log m) where m is len(self)
		Args:
			n (int): The number of elements to return
		"""
		if n is None:
			return sorted(self.counts(), key=itemgetter(1), reverse=True)
		else:
			return heapq.nlargest(n, self.counts(), key=itemgetter(1))

def once(func):
    """Runs a thing once and once only."""
    lock = threading.Lock()

    def new_func(*args, **kwargs):
        if new_func.called:
            return
        with lock:
            if new_func.called:
                return
            rv = func(*args, **kwargs)
            new_func.called = True
            return rv

    new_func = update_wrapper(new_func, func)
    new_func.called = False
    return new_func

def trigger(self, target: str, trigger: str, parameters: Dict[str, Any]={}):
		"""Calls the specified Trigger of another Area with the optionally given parameters.

		Args:
			target: The name of the target Area.
			trigger: The name of the Trigger.
			parameters: The parameters of the function call.
		"""
		pass

def run (self):
        """Handle keyboard interrupt and other errors."""
        try:
            self.run_checked()
        except KeyboardInterrupt:
            thread.interrupt_main()
        except Exception:
            self.internal_error()

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

def _single_page_pdf(page):
    """Construct a single page PDF from the provided page in memory"""
    pdf = Pdf.new()
    pdf.pages.append(page)
    bio = BytesIO()
    pdf.save(bio)
    bio.seek(0)
    return bio.read()

def unixtime_to_datetime(ut):
    """Convert a unixtime timestamp to a datetime object.
    The function converts a timestamp in Unix format to a
    datetime object. UTC timezone will also be set.
    :param ut: Unix timestamp to convert
    :returns: a datetime object
    :raises InvalidDateError: when the given timestamp cannot be
        converted into a valid date
    """

    dt = datetime.datetime.utcfromtimestamp(ut)
    dt = dt.replace(tzinfo=tz.tzutc())
    return dt

def full(self):
        """Return ``True`` if the queue is full, ``False``
        otherwise (not reliable!).

        Only applicable if :attr:`maxsize` is set.

        """
        return self.maxsize and len(self.list) >= self.maxsize or False

def elapsed_time_from(start_time):
    """calculate time delta from latched time and current time"""
    time_then = make_time(start_time)
    time_now = datetime.utcnow().replace(microsecond=0)
    if time_then is None:
        return
    delta_t = time_now - time_then
    return delta_t

def _requiredSize(shape, dtype):
	"""
	Determines the number of bytes required to store a NumPy array with
	the specified shape and datatype.
	"""
	return math.floor(np.prod(np.asarray(shape, dtype=np.uint64)) * np.dtype(dtype).itemsize)

def java_version():
    """Call java and return version information.

    :return unicode: Java version string
    """
    result = subprocess.check_output(
        [c.JAVA, '-version'], stderr=subprocess.STDOUT
    )
    first_line = result.splitlines()[0]
    return first_line.decode()

def _remove_keywords(d):
    """
    copy the dict, filter_keywords

    Parameters
    ----------
    d : dict
    """
    return { k:v for k, v in iteritems(d) if k not in RESERVED }

def get_header(request, header_service):
    """Return request's 'X_POLYAXON_...:' header, as a bytestring.

    Hide some test client ickyness where the header can be unicode.
    """
    service = request.META.get('HTTP_{}'.format(header_service), b'')
    if isinstance(service, str):
        # Work around django test client oddness
        service = service.encode(HTTP_HEADER_ENCODING)
    return service

def serialize(self, value):
        """Takes a datetime object and returns a string"""
        if isinstance(value, str):
            return value
        return value.strftime(DATETIME_FORMAT)

def create_node(self, network, participant):
        """Create a node for a participant."""
        return self.models.MCMCPAgent(network=network, participant=participant)

def __get__(self, obj, objtype):
        if not self.is_method:
            self.is_method = True
        """Support instance methods."""
        return functools.partial(self.__call__, obj)

def _gevent_patch():
    """Patch the modules with gevent

    :return: Default is GEVENT. If it not supports gevent then return MULTITHREAD
    :rtype: int
    """
    try:
        assert gevent
        assert grequests
    except NameError:
        logger.warn('gevent not exist, fallback to multiprocess...')
        return MULTITHREAD
    else:
        monkey.patch_all()  # Must patch before get_photos_info
        return GEVENT

def _to_array(value):
    """As a convenience, turn Python lists and tuples into NumPy arrays."""
    if isinstance(value, (tuple, list)):
        return array(value)
    elif isinstance(value, (float, int)):
        return np.float64(value)
    else:
        return value

def multiply(traj):
    """Sophisticated simulation of multiplication"""
    z=traj.x*traj.y
    traj.f_add_result('z',z=z, comment='I am the product of two reals!')

def from_dict(cls, d):
        """Create an instance from a dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.ENTRIES})

def WriteToPath(obj, filepath):
  """Serializes and writes given Python object to the specified YAML file.

  Args:
    obj: A Python object to serialize.
    filepath: A path to the file into which the object is to be written.
  """
  with io.open(filepath, mode="w", encoding="utf-8") as filedesc:
    WriteToFile(obj, filedesc)

def demo(quiet, shell, speed, prompt, commentecho):
    """Run a demo doitlive session."""
    run(
        DEMO,
        shell=shell,
        speed=speed,
        test_mode=TESTING,
        prompt_template=prompt,
        quiet=quiet,
        commentecho=commentecho,
    )

def requests_post(url, data=None, json=None, **kwargs):
    """Requests-mock requests.post wrapper."""
    return requests_request('post', url, data=data, json=json, **kwargs)

def printmp(msg):
    """Print temporarily, until next print overrides it.
    """
    filler = (80 - len(msg)) * ' '
    print(msg + filler, end='\r')
    sys.stdout.flush()

def is_finite(value: Any) -> bool:
    """Return true if a value is a finite number."""
    return isinstance(value, int) or (isinstance(value, float) and isfinite(value))

def frombits(cls, bits):
        """Series from binary string arguments."""
        return cls.frombitsets(map(cls.BitSet.frombits, bits))

def _check_stream_timeout(started, timeout):
    """Check if the timeout has been reached and raise a `StopIteration` if so.
    """
    if timeout:
        elapsed = datetime.datetime.utcnow() - started
        if elapsed.seconds > timeout:
            raise StopIteration

def remove_parameter(self, name):
		""" Remove the specified parameter from this query

		:param name: name of a parameter to remove
		:return: None
		"""
		if name in self.__query:
			self.__query.pop(name)

def make_executable(script_path):
    """Make `script_path` executable.

    :param script_path: The file to change
    """
    status = os.stat(script_path)
    os.chmod(script_path, status.st_mode | stat.S_IEXEC)

def normalize(self, dt, is_dst=False):
        """Correct the timezone information on the given datetime"""
        if dt.tzinfo is self:
            return dt
        if dt.tzinfo is None:
            raise ValueError('Naive time - no tzinfo set')
        return dt.astimezone(self)

def flatten(l, types=(list, float)):
    """
    Flat nested list of lists into a single list.
    """
    l = [item if isinstance(item, types) else [item] for item in l]
    return [item for sublist in l for item in sublist]

def clean(some_string, uppercase=False):
    """
    helper to clean up an input string
    """
    if uppercase:
        return some_string.strip().upper()
    else:
        return some_string.strip().lower()

def chunk_sequence(sequence, chunk_length):
    """Yield successive n-sized chunks from l."""
    for index in range(0, len(sequence), chunk_length):
        yield sequence[index:index + chunk_length]

def convert_to_int(x: Any, default: int = None) -> int:
    """
    Transforms its input into an integer, or returns ``default``.
    """
    try:
        return int(x)
    except (TypeError, ValueError):
        return default

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

async def acquire_async(self):
        """Acquire the :attr:`lock` asynchronously

        """
        r = self.acquire(blocking=False)
        while not r:
            await asyncio.sleep(.01)
            r = self.acquire(blocking=False)

def query(self, base, filterstr, attrlist=None):
		""" wrapper to search_s """
		return self.conn.search_s(base, ldap.SCOPE_SUBTREE, filterstr, attrlist)

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

def exists(self):
        """Check whether the cluster already exists.

        For example:

        .. literalinclude:: snippets.py
            :start-after: [START bigtable_check_cluster_exists]
            :end-before: [END bigtable_check_cluster_exists]

        :rtype: bool
        :returns: True if the table exists, else False.
        """
        client = self._instance._client
        try:
            client.instance_admin_client.get_cluster(name=self.name)
            return True
        # NOTE: There could be other exceptions that are returned to the user.
        except NotFound:
            return False

def copy_and_update(dictionary, update):
    """Returns an updated copy of the dictionary without modifying the original"""
    newdict = dictionary.copy()
    newdict.update(update)
    return newdict

def _on_release(self, event):
        """Stop dragging."""
        if self._drag_cols or self._drag_rows:
            self._visual_drag.place_forget()
            self._dragged_col = None
            self._dragged_row = None

def interact(self, container: Container) -> None:
        """
        Connects to the PTY (pseudo-TTY) for a given container.
        Blocks until the user exits the PTY.
        """
        cmd = "/bin/bash -c 'source /.environment && /bin/bash'"
        cmd = "docker exec -it {} {}".format(container.id, cmd)
        subprocess.call(cmd, shell=True)

def dumps(obj):
    """Outputs json with formatting edits + object handling."""
    return json.dumps(obj, indent=4, sort_keys=True, cls=CustomEncoder)

def ReadTif(tifFile):
        """Reads a tif file to a 2D NumPy array"""
        img = Image.open(tifFile)
        img = np.array(img)
        return img

def read_numpy(fh, byteorder, dtype, count, offsetsize):
    """Read tag data from file and return as numpy array."""
    dtype = 'b' if dtype[-1] == 's' else byteorder+dtype[-1]
    return fh.read_array(dtype, count)

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

def get_pixel(framebuf, x, y):
        """Get the color of a given pixel"""
        index = (y >> 3) * framebuf.stride + x
        offset = y & 0x07
        return (framebuf.buf[index] >> offset) & 0x01

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

def hard_equals(a, b):
    """Implements the '===' operator."""
    if type(a) != type(b):
        return False
    return a == b

def value(self, progress_indicator):
        """ Interpolate linearly between start and end """
        return interpolate.interpolate_linear_single(self.initial_value, self.final_value, progress_indicator)

def load_jsonf(fpath, encoding):
    """
    :param unicode fpath:
    :param unicode encoding:
    :rtype: dict | list
    """
    with codecs.open(fpath, encoding=encoding) as f:
        return json.load(f)

def is_SYMBOL(token, *symbols):
    """ Returns True if ALL of the given argument are AST nodes
    of the given token (e.g. 'BINARY')
    """
    from symbols.symbol_ import Symbol
    assert all(isinstance(x, Symbol) for x in symbols)
    for sym in symbols:
        if sym.token != token:
            return False

    return True

def __call__(self, func, *args, **kwargs):
        """Shorcut for self.run."""
        return self.run(func, *args, **kwargs)

def load_yaml(file):
    """If pyyaml > 5.1 use full_load to avoid warning"""
    if hasattr(yaml, "full_load"):
        return yaml.full_load(file)
    else:
        return yaml.load(file)

def is_stats_query(query):
    """
    check if the query is a normal search or select query
    :param query:
    :return:
    """
    if not query:
        return False

    # remove all " enclosed strings
    nq = re.sub(r'"[^"]*"', '', query)

    # check if there's | .... select
    if re.findall(r'\|.*\bselect\b', nq, re.I|re.DOTALL):
        return True

    return False

def remove_index(self):
        """Remove Elasticsearch index associated to the campaign"""
        self.index_client.close(self.index_name)
        self.index_client.delete(self.index_name)

def hamming_distance(str1, str2):
    """Calculate the Hamming distance between two bit strings

    Args:
        str1 (str): First string.
        str2 (str): Second string.
    Returns:
        int: Distance between strings.
    Raises:
        VisualizationError: Strings not same length
    """
    if len(str1) != len(str2):
        raise VisualizationError('Strings not same length.')
    return sum(s1 != s2 for s1, s2 in zip(str1, str2))

def _values(self):
        """Getter for series values (flattened)"""
        return [
            val for serie in self.series for val in serie.values
            if val is not None
        ]

def _parse(self, date_str, format='%Y-%m-%d'):
        """
        helper function for parsing FRED date string into datetime
        """
        rv = pd.to_datetime(date_str, format=format)
        if hasattr(rv, 'to_pydatetime'):
            rv = rv.to_pydatetime()
        return rv

def palettebar(height, length, colormap):
    """Return the channels of a palettebar.
    """
    cbar = np.tile(np.arange(length) * 1.0 / (length - 1), (height, 1))
    cbar = (cbar * (colormap.values.max() + 1 - colormap.values.min())
            + colormap.values.min())

    return colormap.palettize(cbar)

def pp_xml(body):
    """Pretty print format some XML so it's readable."""
    pretty = xml.dom.minidom.parseString(body)
    return pretty.toprettyxml(indent="  ")

def _split_comma_separated(string):
    """Return a set of strings."""
    return set(text.strip() for text in string.split(',') if text.strip())

def write_wave(path, audio, sample_rate):
    """Writes a .wav file.

    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)

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

def impad_to_multiple(img, divisor, pad_val=0):
    """Pad an image to ensure each edge to be multiple to some number.

    Args:
        img (ndarray): Image to be padded.
        divisor (int): Padded image edges will be multiple to divisor.
        pad_val (number or sequence): Same as :func:`impad`.

    Returns:
        ndarray: The padded image.
    """
    pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
    pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
    return impad(img, (pad_h, pad_w), pad_val)

def toHdlConversion(self, top, topName: str, saveTo: str) -> List[str]:
        """
        :param top: object which is represenation of design
        :param topName: name which should be used for ipcore
        :param saveTo: path of directory where generated files should be stored

        :return: list of file namens in correct compile order
        """
        raise NotImplementedError(
            "Implement this function for your type of your top module")

def replace(s, old, new, maxreplace=-1):
    """replace (str, old, new[, maxreplace]) -> string

    Return a copy of string str with all occurrences of substring
    old replaced by new. If the optional argument maxreplace is
    given, only the first maxreplace occurrences are replaced.

    """
    return s.replace(old, new, maxreplace)

def setConfigKey(key, value):
		"""
		Sets the config data value for the specified dictionary key
		"""
		configFile = ConfigurationManager._configFile()
		return JsonDataManager(configFile).setKey(key, value)

def get_anchor_href(markup):
    """
    Given HTML markup, return a list of hrefs for each anchor tag.
    """
    soup = BeautifulSoup(markup, 'lxml')
    return ['%s' % link.get('href') for link in soup.find_all('a')]

def _num_cpus_darwin():
    """Return the number of active CPUs on a Darwin system."""
    p = subprocess.Popen(['sysctl','-n','hw.ncpu'],stdout=subprocess.PIPE)
    return p.stdout.read()

def heappop_max(heap):
    """Maxheap version of a heappop."""
    lastelt = heap.pop()    # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        _siftup_max(heap, 0)
        return returnitem
    return lastelt

def add_ul(text, ul):
    """Adds an unordered list to the readme"""
    text += "\n"
    for li in ul:
        text += "- " + li + "\n"
    text += "\n"

    return text

def earth_orientation(date):
    """Earth orientation as a rotating matrix
    """

    x_p, y_p, s_prime = np.deg2rad(_earth_orientation(date))
    return rot3(-s_prime) @ rot2(x_p) @ rot1(y_p)

def close(self):
        """Closes the serial port."""
        if self.pyb and self.pyb.serial:
            self.pyb.serial.close()
        self.pyb = None

def tail(filename, number_of_bytes):
    """Returns the last number_of_bytes of filename"""
    with open(filename, "rb") as f:
        if os.stat(filename).st_size > number_of_bytes:
            f.seek(-number_of_bytes, 2)
        return f.read()

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

def timestamp_from_datetime(dt):
    """
    Compute timestamp from a datetime object that could be timezone aware
    or unaware.
    """
    try:
        utc_dt = dt.astimezone(pytz.utc)
    except ValueError:
        utc_dt = dt.replace(tzinfo=pytz.utc)
    return timegm(utc_dt.timetuple())

def _hess_two_param(self, funct, p0, p1, dl=2e-5, rts=False, **kwargs):
        """
        Hessian of `func` wrt two parameters `p0` and `p1`. (see _graddoc)
        """
        vals0 = self.get_values(p0)
        vals1 = self.get_values(p1)

        f00 = funct(**kwargs)

        self.update(p0, vals0+dl)
        f10 = funct(**kwargs)

        self.update(p1, vals1+dl)
        f11 = funct(**kwargs)

        self.update(p0, vals0)
        f01 = funct(**kwargs)

        if rts:
            self.update(p0, vals0)
            self.update(p1, vals1)
        return (f11 - f10 - f01 + f00) / (dl**2)

def _iter_response(self, url, params=None):
        """Return an enumerable that iterates through a multi-page API request"""
        if params is None:
            params = {}
        params['page_number'] = 1

        # Last page lists itself as next page
        while True:
            response = self._request(url, params)

            for item in response['result_data']:
                yield item

            # Last page lists itself as next page
            if response['service_meta']['next_page_number'] == params['page_number']:
                break

            params['page_number'] += 1

def json(body, charset='utf-8', **kwargs):
    """Takes JSON formatted data, converting it into native Python objects"""
    return json_converter.loads(text(body, charset=charset))

def transcript_sort_key(transcript):
    """
    Key function used to sort transcripts. Taking the negative of
    protein sequence length and nucleotide sequence length so that
    the transcripts with longest sequences come first in the list. This couldn't
    be accomplished with `reverse=True` since we're also sorting by
    transcript name (which places TP53-001 before TP53-002).
    """
    return (
        -len(transcript.protein_sequence),
        -len(transcript.sequence),
        transcript.name
    )

def GetLoggingLocation():
  """Search for and return the file and line number from the log collector.

  Returns:
    (pathname, lineno, func_name) The full path, line number, and function name
    for the logpoint location.
  """
  frame = inspect.currentframe()
  this_file = frame.f_code.co_filename
  frame = frame.f_back
  while frame:
    if this_file == frame.f_code.co_filename:
      if 'cdbg_logging_location' in frame.f_locals:
        ret = frame.f_locals['cdbg_logging_location']
        if len(ret) != 3:
          return (None, None, None)
        return ret
    frame = frame.f_back
  return (None, None, None)

def l2_norm(arr):
    """
    The l2 norm of an array is is defined as: sqrt(||x||), where ||x|| is the
    dot product of the vector.
    """
    arr = np.asarray(arr)
    return np.sqrt(np.dot(arr.ravel().squeeze(), arr.ravel().squeeze()))

def from_pairs_to_array_values(pairs):
    """
        Like from pairs but combines duplicate key values into arrays
    :param pairs:
    :return:
    """
    result = {}
    for pair in pairs:
        result[pair[0]] = concat(prop_or([], pair[0], result), [pair[1]])
    return result

def run_func(self, func_path, *func_args, **kwargs):
        """Run a function in Matlab and return the result.

        Parameters
        ----------
        func_path: str
            Name of function to run or a path to an m-file.
        func_args: object, optional
            Function args to send to the function.
        nargout: int, optional
            Desired number of return arguments.
        kwargs:
            Keyword arguments are passed to Matlab in the form [key, val] so
            that matlab.plot(x, y, '--', LineWidth=2) would be translated into
            plot(x, y, '--', 'LineWidth', 2)

        Returns
        -------
        Result dictionary with keys: 'message', 'result', and 'success'
        """
        if not self.started:
            raise ValueError('Session not started, use start()')

        nargout = kwargs.pop('nargout', 1)
        func_args += tuple(item for pair in zip(kwargs.keys(), kwargs.values())
                           for item in pair)
        dname = os.path.dirname(func_path)
        fname = os.path.basename(func_path)
        func_name, ext = os.path.splitext(fname)
        if ext and not ext == '.m':
            raise TypeError('Need to give path to .m file')
        return self._json_response(cmd='eval',
                                   func_name=func_name,
                                   func_args=func_args or '',
                                   dname=dname,
                                   nargout=nargout)

def step_table_made(self):
        """check if the step table exists"""
        try:
            empty = self.step_table.empty
        except AttributeError:
            empty = True
        return not empty

def _curve(x1, y1, x2, y2, hunit = HUNIT, vunit = VUNIT):
    """
    Return a PyX curved path from (x1, y1) to (x2, y2),
    such that the slope at either end is zero.
    """
    ax1, ax2, axm = x1 * hunit, x2 * hunit, (x1 + x2) * hunit / 2
    ay1, ay2 = y1 * vunit, y2 * vunit
    return pyx.path.curve(ax1, ay1, axm, ay1, axm, ay2, ax2, ay2)

def exception_format():
    """
    Convert exception info into a string suitable for display.
    """
    return "".join(traceback.format_exception(
        sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2]
    ))

def get_cached_data(datatable, **kwargs):
    """ Returns the cached object list under the appropriate key, or None if not set. """
    cache_key = '%s%s' % (CACHE_PREFIX, datatable.get_cache_key(**kwargs))
    data = cache.get(cache_key)
    log.debug("Reading data from cache at %r: %r", cache_key, data)
    return data

def __call__(self, args):
        """Execute the user function."""
        window, ij = args
        return self.user_func(srcs, window, ij, global_args), window

def SvcStop(self) -> None:
        """
        Called when the service is being shut down.
        """
        # tell the SCM we're shutting down
        # noinspection PyUnresolvedReferences
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        # fire the stop event
        win32event.SetEvent(self.h_stop_event)

def margin(text):
    r"""Add a margin to both ends of each line in the string.

    Example:
        >>> margin('line1\nline2')
        '  line1  \n  line2  '
    """
    lines = str(text).split('\n')
    return '\n'.join('  {}  '.format(l) for l in lines)

def reset_default_logger():
    """
    Resets the internal default logger to the initial configuration
    """
    global logger
    global _loglevel
    global _logfile
    global _formatter
    _loglevel = logging.DEBUG
    _logfile = None
    _formatter = None
    logger = setup_logger(name=LOGZERO_DEFAULT_LOGGER, logfile=_logfile, level=_loglevel, formatter=_formatter)

def getBitmap(self):
        """ Captures screen area of this region, at least the part that is on the screen

        Returns image as numpy array
        """
        return PlatformManager.getBitmapFromRect(self.x, self.y, self.w, self.h)

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

def _shuffle(data, idx):
    """Shuffle the data."""
    shuffle_data = []

    for idx_k, idx_v in data:
        shuffle_data.append((idx_k, mx.ndarray.array(idx_v.asnumpy()[idx], idx_v.context)))

    return shuffle_data

def set_ylimits(self, row, column, min=None, max=None):
        """Set y-axis limits of a subplot.

        :param row,column: specify the subplot.
        :param min: minimal axis value
        :param max: maximum axis value

        """
        subplot = self.get_subplot_at(row, column)
        subplot.set_ylimits(min, max)

def _my_hash(arg_list):
    # type: (List[Any]) -> int
    """Simple helper hash function"""
    res = 0
    for arg in arg_list:
        res = res * 31 + hash(arg)
    return res

def isarray(array, test, dim=2):
    """Returns True if test is True for all array elements.
    Otherwise, returns False.
    """
    if dim > 1:
        return all(isarray(array[i], test, dim - 1)
                   for i in range(len(array)))
    return all(test(i) for i in array)

def unit_tangent(self, t):
        """returns the unit tangent vector of the segment at t (centered at
        the origin and expressed as a complex number)."""
        dseg = self.derivative(t)
        return dseg/abs(dseg)

def A(*a):
    """convert iterable object into numpy array"""
    return np.array(a[0]) if len(a)==1 else [np.array(o) for o in a]

def str_to_num(str_value):
        """Convert str_value to an int or a float, depending on the
        numeric value represented by str_value.

        """
        str_value = str(str_value)
        try:
            return int(str_value)
        except ValueError:
            return float(str_value)

def dict_keys_without_hyphens(a_dict):
    """Return the a new dict with underscores instead of hyphens in keys."""
    return dict(
        (key.replace('-', '_'), val) for key, val in a_dict.items())

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

def cleanup_lib(self):
        """ unload the previously loaded shared library """
        if not self.using_openmp:
            #this if statement is necessary because shared libraries that use
            #OpenMP will core dump when unloaded, this is a well-known issue with OpenMP
            logging.debug('unloading shared library')
            _ctypes.dlclose(self.lib._handle)

def set_as_object(self, value):
        """
        Sets a new value to map element

        :param value: a new element or map value.
        """
        self.clear()
        map = MapConverter.to_map(value)
        self.append(map)

def use_theme(theme):
    """Make the given theme current.

    There are two included themes: light_theme, dark_theme.
    """
    global current
    current = theme
    import scene
    if scene.current is not None:
        scene.current.stylize()

def get_best_encoding(stream):
    """Returns the default stream encoding if not found."""
    rv = getattr(stream, 'encoding', None) or sys.getdefaultencoding()
    if is_ascii_encoding(rv):
        return 'utf-8'
    return rv

def is_rpm_package_installed(pkg):
    """ checks if a particular rpm package is installed """

    with settings(hide('warnings', 'running', 'stdout', 'stderr'),
                  warn_only=True, capture=True):

        result = sudo("rpm -q %s" % pkg)
        if result.return_code == 0:
            return True
        elif result.return_code == 1:
            return False
        else:   # print error to user
            print(result)
            raise SystemExit()

def slugify_filename(filename):
    """ Slugify filename """
    name, ext = os.path.splitext(filename)
    slugified = get_slugified_name(name)
    return slugified + ext

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

def get_mouse_location(self):
        """
        Get the current mouse location (coordinates and screen number).

        :return: a namedtuple with ``x``, ``y`` and ``screen_num`` fields
        """
        x = ctypes.c_int(0)
        y = ctypes.c_int(0)
        screen_num = ctypes.c_int(0)
        _libxdo.xdo_get_mouse_location(
            self._xdo, ctypes.byref(x), ctypes.byref(y),
            ctypes.byref(screen_num))
        return mouse_location(x.value, y.value, screen_num.value)

def atlasdb_format_query( query, values ):
    """
    Turn a query into a string for printing.
    Useful for debugging.
    """
    return "".join( ["%s %s" % (frag, "'%s'" % val if type(val) in [str, unicode] else val) for (frag, val) in zip(query.split("?"), values + ("",))] )

def __get_float(section, name):
    """Get the forecasted float from json section."""
    try:
        return float(section[name])
    except (ValueError, TypeError, KeyError):
        return float(0)

def complex_check(*args, func=None):
    """Check if arguments are complex numbers."""
    func = func or inspect.stack()[2][3]
    for var in args:
        if not isinstance(var, numbers.Complex):
            name = type(var).__name__
            raise ComplexError(
                f'Function {func} expected complex number, {name} got instead.')

def deleted(self, instance):
        """
        Convenience method for deleting a model (automatically commits the
        delete to the database and returns with an HTTP 204 status code)
        """
        self.session_manager.delete(instance, commit=True)
        return '', HTTPStatus.NO_CONTENT

def get_propety_by_name(pif, name):
    """Get a property by name"""
    warn("This method has been deprecated in favor of get_property_by_name")
    return next((x for x in pif.properties if x.name == name), None)

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

def data_from_techshop_ws(tws_url):
    """Scrapes data from techshop.ws."""

    r = requests.get(tws_url)
    if r.status_code == 200:
        data = BeautifulSoup(r.text, "lxml")
    else:
        data = "There was an error while accessing data on techshop.ws."

    return data

def _config_win32_domain(self, domain):
        """Configure a Domain registry entry."""
        # we call str() on domain to convert it from unicode to ascii
        self.domain = dns.name.from_text(str(domain))

def apply_to_field_if_exists(effect, field_name, fn, default):
    """
    Apply function to specified field of effect if it is not None,
    otherwise return default.
    """
    value = getattr(effect, field_name, None)
    if value is None:
        return default
    else:
        return fn(value)

def home():
    """Temporary helper function to link to the API routes"""
    return dict(links=dict(api='{}{}'.format(request.url, PREFIX[1:]))), \
        HTTPStatus.OK

def send(self, *args, **kwargs):
        """Writes the passed chunk and flushes it to the client."""
        self.write(*args, **kwargs)
        self.flush()

def _increase_file_handle_limit():
    """Raise the open file handles permitted by the Dusty daemon process
    and its child processes. The number we choose here needs to be within
    the OS X default kernel hard limit, which is 10240."""
    logging.info('Increasing file handle limit to {}'.format(constants.FILE_HANDLE_LIMIT))
    resource.setrlimit(resource.RLIMIT_NOFILE,
                       (constants.FILE_HANDLE_LIMIT, resource.RLIM_INFINITY))

def erase_lines(n=1):
    """ Erases n lines from the screen and moves the cursor up to follow
    """
    for _ in range(n):
        print(codes.cursor["up"], end="")
        print(codes.cursor["eol"], end="")

def write_line(self, line, count=1):
        """writes the line and count newlines after the line"""
        self.write(line)
        self.write_newlines(count)

def get_datatype(self, table: str, column: str) -> str:
        """Returns database SQL datatype for a column: e.g. VARCHAR."""
        return self.flavour.get_datatype(self, table, column).upper()

def make_env_key(app_name, key):
    """Creates an environment key-equivalent for the given key"""
    key = key.replace('-', '_').replace(' ', '_')
    return str("_".join((x.upper() for x in (app_name, key))))

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

def enable_writes(self):
        """Restores the state of the batched queue for writing."""
        self.write_buffer = []
        self.flush_lock = threading.RLock()
        self.flush_thread = FlushThread(self.max_batch_time,
                                        self._flush_writes)

async def list(source):
    """Generate a single list from an asynchronous sequence."""
    result = []
    async with streamcontext(source) as streamer:
        async for item in streamer:
            result.append(item)
    yield result

def perform_pca(A):
    """
    Computes eigenvalues and eigenvectors of covariance matrix of A.
    The rows of a correspond to observations, the columns to variables.
    """
    # First subtract the mean
    M = (A-numpy.mean(A.T, axis=1)).T
    # Get eigenvectors and values of covariance matrix
    return numpy.linalg.eig(numpy.cov(M))

def strictly_positive_int_or_none(val):
    """Parse `val` into either `None` or a strictly positive integer."""
    val = positive_int_or_none(val)
    if val is None or val > 0:
        return val
    raise ValueError('"{}" must be strictly positive'.format(val))

def get_span_char_width(span, column_widths):
    """
    Sum the widths of the columns that make up the span, plus the extra.

    Parameters
    ----------
    span : list of lists of int
        list of [row, column] pairs that make up the span
    column_widths : list of int
        The widths of the columns that make up the table

    Returns
    -------
    total_width : int
        The total width of the span
    """

    start_column = span[0][1]
    column_count = get_span_column_count(span)
    total_width = 0

    for i in range(start_column, start_column + column_count):
        total_width += column_widths[i]

    total_width += column_count - 1

    return total_width

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

def getCollectDServer(queue, cfg):
    """Get the appropriate collectd server (multi processed or not)"""
    server = CollectDServerMP if cfg.collectd_workers > 1 else CollectDServer
    return server(queue, cfg)

def angle(x, y):
    """Return the angle between vectors a and b in degrees."""
    return arccos(dot(x, y)/(norm(x)*norm(y)))*180./pi

def normalize_matrix(matrix):
  """Fold all values of the matrix into [0, 1]."""
  abs_matrix = np.abs(matrix.copy())
  return abs_matrix / abs_matrix.max()

def _serialize_json(obj, fp):
    """ Serialize ``obj`` as a JSON formatted stream to ``fp`` """
    json.dump(obj, fp, indent=4, default=serialize)

def valid_file(path: str) -> bool:
    """
    Verifies that a string path actually exists and is a file

    :param path: The path to verify
    :return: **True** if path exist and is a file
    """
    path = Path(path).expanduser()
    log.debug("checking if %s is a valid file", path)
    return path.exists() and path.is_file()

def _config_section(config, section):
    """Read the configuration file and return a section."""
    path = os.path.join(config.get('config_path'), config.get('config_file'))
    conf = _config_ini(path)
    return conf.get(section)

def commit(self, session=None):
        """Merge modified objects into parent transaction.

        Once commited a transaction object is not usable anymore

        :param:session: current sqlalchemy Session
        """
        if self.__cleared:
            return

        if self._parent:
            # nested transaction
            self._commit_parent()
        else:
            self._commit_repository()
        self._clear()

def _uniqueid(n=30):
    """Return a unique string with length n.

    :parameter int N: number of character in the uniqueid
    :return: the uniqueid
    :rtype: str
    """
    return ''.join(random.SystemRandom().choice(
                   string.ascii_uppercase + string.ascii_lowercase)
                   for _ in range(n))

def similarity(self, other):
        """Calculates the cosine similarity between this vector and another
        vector."""
        if self.magnitude == 0 or other.magnitude == 0:
            return 0

        return self.dot(other) / self.magnitude

def ishex(obj):
    """
    Test if the argument is a string representing a valid hexadecimal digit.

    :param obj: Object
    :type  obj: any

    :rtype: boolean
    """
    return isinstance(obj, str) and (len(obj) == 1) and (obj in string.hexdigits)

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

def unpack_from(self, data, offset=0):
        """See :func:`~bitstruct.unpack_from()`.

        """

        return tuple([v[1] for v in self.unpack_from_any(data, offset)])

def scale_v2(vec, amount):
    """Return a new Vec2 with x and y from vec and multiplied by amount."""

    return Vec2(vec.x * amount, vec.y * amount)

def get_latex_table(self, parameters=None, transpose=False, caption=None,
                        label="tab:model_params", hlines=True, blank_fill="--"):  # pragma: no cover
        """ Generates a LaTeX table from parameter summaries.

        Parameters
        ----------
        parameters : list[str], optional
            A list of what parameters to include in the table. By default, includes all parameters
        transpose : bool, optional
            Defaults to False, which gives each column as a parameter, each chain (framework)
            as a row. You can swap it so that you have a parameter each row and a framework
            each column by setting this to True
        caption : str, optional
            If you want to generate a caption for the table through Python, use this.
            Defaults to an empty string
        label : str, optional
            If you want to generate a label for the table through Python, use this.
            Defaults to an empty string
        hlines : bool, optional
            Inserts ``\\hline`` before and after the header, and at the end of table.
        blank_fill : str, optional
            If a framework does not have a particular parameter, will fill that cell of
            the table with this string.

        Returns
        -------
        str
            the LaTeX table.
        """
        if parameters is None:
            parameters = self.parent._all_parameters
        for p in parameters:
            assert isinstance(p, str), \
                "Generating a LaTeX table requires all parameters have labels"
        num_parameters = len(parameters)
        num_chains = len(self.parent.chains)
        fit_values = self.get_summary(squeeze=False)
        if label is None:
            label = ""
        if caption is None:
            caption = ""

        end_text = " \\\\ \n"
        if transpose:
            column_text = "c" * (num_chains + 1)
        else:
            column_text = "c" * (num_parameters + 1)

        center_text = ""
        hline_text = "\\hline\n"
        if hlines:
            center_text += hline_text + "\t\t"
        if transpose:
            center_text += " & ".join(["Parameter"] + [c.name for c in self.parent.chains]) + end_text
            if hlines:
                center_text += "\t\t" + hline_text
            for p in parameters:
                arr = ["\t\t" + p]
                for chain_res in fit_values:
                    if p in chain_res:
                        arr.append(self.get_parameter_text(*chain_res[p], wrap=True))
                    else:
                        arr.append(blank_fill)
                center_text += " & ".join(arr) + end_text
        else:
            center_text += " & ".join(["Model"] + parameters) + end_text
            if hlines:
                center_text += "\t\t" + hline_text
            for name, chain_res in zip([c.name for c in self.parent.chains], fit_values):
                arr = ["\t\t" + name]
                for p in parameters:
                    if p in chain_res:
                        arr.append(self.get_parameter_text(*chain_res[p], wrap=True))
                    else:
                        arr.append(blank_fill)
                center_text += " & ".join(arr) + end_text
        if hlines:
            center_text += "\t\t" + hline_text
        final_text = get_latex_table_frame(caption, label) % (column_text, center_text)

        return final_text

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

async def terminate(self):
        """Terminate a running script."""
        self.proc.terminate()

        await asyncio.wait_for(self.proc.wait(), self.kill_delay)
        if self.proc.returncode is None:
            self.proc.kill()
        await self.proc.wait()

        await super().terminate()

def isfinite(data: mx.nd.NDArray) -> mx.nd.NDArray:
    """Performs an element-wise check to determine if the NDArray contains an infinite element or not.
       TODO: remove this funciton after upgrade to MXNet 1.4.* in favor of mx.ndarray.contrib.isfinite()
    """
    is_data_not_nan = data == data
    is_data_not_infinite = data.abs() != np.inf
    return mx.nd.logical_and(is_data_not_infinite, is_data_not_nan)

def rollback(name, database=None, directory=None, verbose=None):
    """Rollback a migration with given name."""
    router = get_router(directory, database, verbose)
    router.rollback(name)

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

def aloha_to_html(html_source):
    """Converts HTML5 from Aloha to a more structured HTML5"""
    xml = aloha_to_etree(html_source)
    return etree.tostring(xml, pretty_print=True)

def union(self, other):
        """Return a new set which is the union of I{self} and I{other}.

        @param other: the other set
        @type other: Set object
        @rtype: the same type as I{self}
        """

        obj = self._clone()
        obj.union_update(other)
        return obj

def _opt_call_from_base_type(self, value):
    """Call _from_base_type() if necessary.

    If the value is a _BaseValue instance, unwrap it and call all
    _from_base_type() methods.  Otherwise, return the value
    unchanged.
    """
    if isinstance(value, _BaseValue):
      value = self._call_from_base_type(value.b_val)
    return value

def decompress(f):
    """Decompress a Plan 9 image file.  Assumes f is already cued past the
    initial 'compressed\n' string.
    """

    r = meta(f.read(60))
    return r, decomprest(f, r[4])

def print_log(value_color="", value_noncolor=""):
    """set the colors for text."""
    HEADER = '\033[92m'
    ENDC = '\033[0m'
    print(HEADER + value_color + ENDC + str(value_noncolor))

def to_dict(self):
        """
        Serialize representation of the column for local caching.
        """
        return {'schema': self.schema, 'table': self.table, 'name': self.name, 'type': self.type}

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

def synchronized(obj):
  """
  This function has two purposes:

  1. Decorate a function that automatically synchronizes access to the object
     passed as the first argument (usually `self`, for member methods)
  2. Synchronize access to the object, used in a `with`-statement.

  Note that you can use #wait(), #notify() and #notify_all() only on
  synchronized objects.

  # Example
  ```python
  class Box(Synchronizable):
    def __init__(self):
      self.value = None
    @synchronized
    def get(self):
      return self.value
    @synchronized
    def set(self, value):
      self.value = value

  box = Box()
  box.set('foobar')
  with synchronized(box):
    box.value = 'taz\'dingo'
  print(box.get())
  ```

  # Arguments
  obj (Synchronizable, function): The object to synchronize access to, or a
    function to decorate.

  # Returns
  1. The decorated function.
  2. The value of `obj.synchronizable_condition`, which should implement the
     context-manager interface (to be used in a `with`-statement).
  """

  if hasattr(obj, 'synchronizable_condition'):
    return obj.synchronizable_condition
  elif callable(obj):
    @functools.wraps(obj)
    def wrapper(self, *args, **kwargs):
      with self.synchronizable_condition:
        return obj(self, *args, **kwargs)
    return wrapper
  else:
    raise TypeError('expected Synchronizable instance or callable to decorate')

def indent(block, spaces):
    """ indents paragraphs of text for rst formatting """
    new_block = ''
    for line in block.split('\n'):
        new_block += spaces + line + '\n'
    return new_block

def make_kind_check(python_types, numpy_kind):
    """
    Make a function that checks whether a scalar or array is of a given kind
    (e.g. float, int, datetime, timedelta).
    """
    def check(value):
        if hasattr(value, 'dtype'):
            return value.dtype.kind == numpy_kind
        return isinstance(value, python_types)
    return check

def _get_os_environ_dict(keys):
  """Return a dictionary of key/values from os.environ."""
  return {k: os.environ.get(k, _UNDEFINED) for k in keys}

def shallow_reverse(g):
    """
    Make a shallow copy of a directional graph and reverse the edges. This is a workaround to solve the issue that one
    cannot easily make a shallow reversed copy of a graph in NetworkX 2, since networkx.reverse(copy=False) now returns
    a GraphView, and GraphViews are always read-only.

    :param networkx.DiGraph g:  The graph to reverse.
    :return:                    A new networkx.DiGraph that has all nodes and all edges of the original graph, with
                                edges reversed.
    """

    new_g = networkx.DiGraph()

    new_g.add_nodes_from(g.nodes())
    for src, dst, data in g.edges(data=True):
        new_g.add_edge(dst, src, **data)

    return new_g

def get_translucent_cmap(r, g, b):

    class TranslucentCmap(BaseColormap):
        glsl_map = """
        vec4 translucent_fire(float t) {{
            return vec4({0}, {1}, {2}, t);
        }}
        """.format(r, g, b)

    return TranslucentCmap()

def alter_change_column(self, table, column, field):
        """Support change columns."""
        return self._update_column(table, column, lambda a, b: b)

def sort_genomic_ranges(rngs):
  """sort multiple ranges"""
  return sorted(rngs, key=lambda x: (x.chr, x.start, x.end))

def delete(gandi, resource):
    """Delete DNSSEC key.
    """

    result = gandi.dnssec.delete(resource)
    gandi.echo('Delete successful.')

    return result

def focusInEvent(self, event):
        """Reimplement Qt method to send focus change notification"""
        self.focus_changed.emit()
        return super(ShellWidget, self).focusInEvent(event)

def load_data(filename):
    """
    :rtype : numpy matrix
    """
    data = pandas.read_csv(filename, header=None, delimiter='\t', skiprows=9)
    return data.as_matrix()

def load(filename):
    """Load a pickled obj from the filesystem.

    You better know what you expect from the given pickle, because we don't check it.

    Args:
        filename (str): The filename we load the object from.

    Returns:
        The object we were able to unpickle, else None.
    """
    if not os.path.exists(filename):
        LOG.error("load object - File '%s' does not exist.", filename)
        return None

    obj = None
    with open(filename, 'rb') as obj_file:
        obj = dill.load(obj_file)
    return obj

def get_font_list():
    """Returns a sorted list of all system font names"""

    font_map = pangocairo.cairo_font_map_get_default()
    font_list = [f.get_name() for f in font_map.list_families()]
    font_list.sort()

    return font_list

def tfds_dir():
  """Path to tensorflow_datasets directory."""
  return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def mcc(y_true, y_pred, round=True):
    """Matthews correlation coefficient
    """
    y_true, y_pred = _mask_value_nan(y_true, y_pred)
    if round:
        y_true = np.round(y_true)
        y_pred = np.round(y_pred)
    return skm.matthews_corrcoef(y_true, y_pred)

def np_counts(self):
    """Dictionary of noun phrase frequencies in this text.
    """
    counts = defaultdict(int)
    for phrase in self.noun_phrases:
        counts[phrase] += 1
    return counts

def _get_binary_from_ipv4(self, ip_addr):
        """Converts IPv4 address to binary form."""

        return struct.unpack("!L", socket.inet_pton(socket.AF_INET,
                                                    ip_addr))[0]

def _propagate_mean(mean, linop, dist):
  """Propagate a mean through linear Gaussian transformation."""
  return linop.matmul(mean) + dist.mean()[..., tf.newaxis]

def _get_name(column_like):
    """
    Get the name from a column-like SQLAlchemy expression.

    Works for Columns and Cast expressions.
    """
    if isinstance(column_like, Column):
        return column_like.name
    elif isinstance(column_like, Cast):
        return column_like.clause.name

def strip_codes(s: Any) -> str:
    """ Strip all color codes from a string.
        Returns empty string for "falsey" inputs.
    """
    return codepat.sub('', str(s) if (s or (s == 0)) else '')

def is_cached(file_name):
	"""
	Check if a given file is available in the cache or not
	"""

	gml_file_path = join(join(expanduser('~'), OCTOGRID_DIRECTORY), file_name)

	return isfile(gml_file_path)

def log_loss(preds, labels):
    """Logarithmic loss with non-necessarily-binary labels."""
    log_likelihood = np.sum(labels * np.log(preds)) / len(preds)
    return -log_likelihood

def dict_to_enum_fn(d: Dict[str, Any], enum_class: Type[Enum]) -> Enum:
    """
    Converts an ``dict`` to a ``Enum``.
    """
    return enum_class[d['name']]

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

def predict(self, X):
        """
        Apply transforms to the data, and predict with the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        yp : array-like
            Predicted transformed target
        """
        Xt, _, _ = self._transform(X)
        return self._final_estimator.predict(Xt)

def scale_image(image, new_width):
    """Resizes an image preserving the aspect ratio.
    """
    (original_width, original_height) = image.size
    aspect_ratio = original_height/float(original_width)
    new_height = int(aspect_ratio * new_width)

    # This scales it wider than tall, since characters are biased
    new_image = image.resize((new_width*2, new_height))
    return new_image

def dictfetchall(cursor):
    """Returns all rows from a cursor as a dict (rather than a headerless table)

    From Django Documentation: https://docs.djangoproject.com/en/dev/topics/db/sql/
    """
    desc = cursor.description
    return [dict(zip([col[0] for col in desc], row)) for row in cursor.fetchall()]

def flat_list(input_list):
    r"""
    Given a list of nested lists of arbitrary depth, returns a single level or
    'flat' list.

    """
    x = input_list
    if isinstance(x, list):
        return [a for i in x for a in flat_list(i)]
    else:
        return [x]

def pad_cells(table):
    """Pad each cell to the size of the largest cell in its column."""
    col_sizes = [max(map(len, col)) for col in zip(*table)]
    for row in table:
        for cell_num, cell in enumerate(row):
            row[cell_num] = pad_to(cell, col_sizes[cell_num])
    return table

def _grid_widgets(self):
        """Puts the two whole widgets in the correct position depending on compound."""
        scrollbar_column = 0 if self.__compound is tk.LEFT else 2
        self.listbox.grid(row=0, column=1, sticky="nswe")
        self.scrollbar.grid(row=0, column=scrollbar_column, sticky="ns")

def unique(seq):
    """Return the unique elements of a collection even if those elements are
       unhashable and unsortable, like dicts and sets"""
    cleaned = []
    for each in seq:
        if each not in cleaned:
            cleaned.append(each)
    return cleaned

def xor(a, b):
        """Bitwise xor on equal length bytearrays."""
        return bytearray(i ^ j for i, j in zip(a, b))

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

def set_user_password(environment, parameter, password):
    """
    Sets a user's password in the keyring storage
    """
    username = '%s:%s' % (environment, parameter)
    return password_set(username, password)

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

def limitReal(x, max_denominator=1000000):
    """Creates an pysmt Real constant from x.

    Args:
        x (number): A number to be cast to a pysmt constant.
        max_denominator (int, optional): The maximum size of the denominator.
            Default 1000000.

    Returns:
        A Real constant with the given value and the denominator limited.

    """
    f = Fraction(x).limit_denominator(max_denominator)
    return Real((f.numerator, f.denominator))

def clean(ctx, dry_run=False):
    """Cleanup generated document artifacts."""
    basedir = ctx.sphinx.destdir or "build/docs"
    cleanup_dirs([basedir], dry_run=dry_run)

def get_top_priority(self):
        """Pops the element that has the top (smallest) priority.

        :returns: element with the top (smallest) priority.
        :raises: IndexError -- Priority queue is empty.

        """
        if self.is_empty():
            raise IndexError("Priority queue is empty.")
        _, _, element = heapq.heappop(self.pq)
        if element in self.element_finder:
            del self.element_finder[element]
        return element

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

def __eq__(self, anotherset):
        """Tests on itemlist equality"""
        if not isinstance(anotherset, LR0ItemSet):
            raise TypeError
        if len(self.itemlist) != len(anotherset.itemlist):
            return False
        for element in self.itemlist:
            if element not in anotherset.itemlist:
                return False
        return True

def parse_reading(val: str) -> Optional[float]:
    """ Convert reading value to float (if possible) """
    try:
        return float(val)
    except ValueError:
        logging.warning('Reading of "%s" is not a number', val)
        return None

def straight_line_show(title, length=100, linestyle="=", pad=0):
        """Print a formatted straight line.
        """
        print(StrTemplate.straight_line(
            title=title, length=length, linestyle=linestyle, pad=pad))

def is_interactive(self):
        """ Determine if the user requested interactive mode.
        """
        # The Python interpreter sets sys.flags correctly, so use them!
        if sys.flags.interactive:
            return True

        # IPython does not set sys.flags when -i is specified, so first
        # check it if it is already imported.
        if '__IPYTHON__' not in dir(six.moves.builtins):
            return False

        # Then we check the application singleton and determine based on
        # a variable it sets.
        try:
            from IPython.config.application import Application as App
            return App.initialized() and App.instance().interact
        except (ImportError, AttributeError):
            return False

def camelcase_underscore(name):
    """ Convert camelcase names to underscore """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def double_sha256(data):
    """A standard compound hash."""
    return bytes_as_revhex(hashlib.sha256(hashlib.sha256(data).digest()).digest())

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

def get_current_branch():
    """
    Return the current branch
    """
    cmd = ["git", "rev-parse", "--abbrev-ref", "HEAD"]
    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    return output.strip().decode("utf-8")

def setVolume(self, volume):
        """Changes volume"""
        val = float(val)
        cmd = "volume %s" % val
        self._execute(cmd)

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

def open_with_encoding(filename, encoding, mode='r'):
    """Return opened file with a specific encoding."""
    return io.open(filename, mode=mode, encoding=encoding,
                   newline='')

def get_last_modified_timestamp(self):
        """
        Looks at the files in a git root directory and grabs the last modified timestamp
        """
        cmd = "find . -print0 | xargs -0 stat -f '%T@ %p' | sort -n | tail -1 | cut -f2- -d' '"
        ps = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        output = ps.communicate()[0]
        print output

def _to_lower_alpha_only(s):
    """Return a lowercased string with non alphabetic chars removed.

    White spaces are not to be removed."""
    s = re.sub(r'\n', ' ',  s.lower())
    return re.sub(r'[^a-z\s]', '', s)

def __reversed__(self):
        """
        Return a reversed iterable over the items in the dictionary. Items are
        iterated over in their reverse sort order.

        Iterating views while adding or deleting entries in the dictionary may
        raise a RuntimeError or fail to iterate over all entries.
        """
        _dict = self._dict
        return iter((key, _dict[key]) for key in reversed(self._list))

def user_parse(data):
        """Parse information from the provider."""
        _user = data.get('response', {}).get('user', {})
        yield 'id', _user.get('name')
        yield 'username', _user.get('name')
        yield 'link', _user.get('blogs', [{}])[0].get('url')

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

def _validate_image_rank(self, img_array):
        """
        Images must be either 2D or 3D.
        """
        if img_array.ndim == 1 or img_array.ndim > 3:
            msg = "{0}D imagery is not allowed.".format(img_array.ndim)
            raise IOError(msg)

def psql(sql, show=True):
    """
    Runs SQL against the project's database.
    """
    out = postgres('psql -c "%s"' % sql)
    if show:
        print_command(sql)
    return out

def assert_lock(fname):
    """
    If file is locked then terminate program else lock file.
    """

    if not set_lock(fname):
        logger.error('File {} is already locked. Terminating.'.format(fname))
        sys.exit()

def update(self, *args, **kwargs):
        """ A handy update() method which returns self :)

        :rtype: DictProxy
        """
        super(DictProxy, self).update(*args, **kwargs)
        return self

def compose(*funcs):
    """compose a list of functions"""
    return lambda x: reduce(lambda v, f: f(v), reversed(funcs), x)

def seconds_to_hms(input_seconds):
    """Convert seconds to human-readable time."""
    minutes, seconds = divmod(input_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    hours = int(hours)
    minutes = int(minutes)
    seconds = str(int(seconds)).zfill(2)

    return hours, minutes, seconds

def flatten_array(grid):
    """
    Takes a multi-dimensional array and returns a 1 dimensional array with the
    same contents.
    """
    grid = [grid[i][j] for i in range(len(grid)) for j in range(len(grid[i]))]
    while type(grid[0]) is list:
        grid = flatten_array(grid)
    return grid

def increment(self, amount=1):
        """
        Increments the main progress bar by amount.
        """
        self._primaryProgressBar.setValue(self.value() + amount)
        QApplication.instance().processEvents()

def adapter(data, headers, **kwargs):
    """Wrap vertical table in a function for TabularOutputFormatter."""
    keys = ('sep_title', 'sep_character', 'sep_length')
    return vertical_table(data, headers, **filter_dict_by_key(kwargs, keys))

def __normalize_list(self, msg):
        """Split message to list by commas and trim whitespace."""
        if isinstance(msg, list):
            msg = "".join(msg)
        return list(map(lambda x: x.strip(), msg.split(",")))

def str_traceback(error, tb):
    """Returns a string representation of the traceback.
    """
    if not isinstance(tb, types.TracebackType):
        return tb

    return ''.join(traceback.format_exception(error.__class__, error, tb))

def setup_detect_python2():
        """
        Call this before using the refactoring tools to create them on demand
        if needed.
        """
        if None in [RTs._rt_py2_detect, RTs._rtp_py2_detect]:
            RTs._rt_py2_detect = RefactoringTool(py2_detect_fixers)
            RTs._rtp_py2_detect = RefactoringTool(py2_detect_fixers,
                                                  {'print_function': True})

def _get_bokeh_html(self, chart_obj):
        """
        Get the html for a Bokeh chart
        """
        global bokeh_renderer
        try:
            renderer = bokeh_renderer
            p = renderer.get_plot(chart_obj).state
            script, div = components(p)
            return script + "\n" + div

        except Exception as e:
            self.err(e, self._get_bokeh_html,
                     "Can not get html from the Bokeh rendering engine")

def _timestamp_to_json_row(value):
    """Coerce 'value' to an JSON-compatible representation.

    This version returns floating-point seconds value used in row data.
    """
    if isinstance(value, datetime.datetime):
        value = _microseconds_from_datetime(value) * 1e-6
    return value

def floor(self):
    """Round `x` and `y` down to integers."""
    return Point(int(math.floor(self.x)), int(math.floor(self.y)))

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

def invertDictMapping(d):
    """ Invert mapping of dictionary (i.e. map values to list of keys) """
    inv_map = {}
    for k, v in d.items():
        inv_map[v] = inv_map.get(v, [])
        inv_map[v].append(k)
    return inv_map

def add_todo(request):
    cur = request.cursor
    todo = request.json["todo"]
    cur.execute("""INSERT INTO todos (todo) VALUES (?)""", (todo,))
    last_id = cur.lastrowid
    cur.connection.commit()

    return request.Response(json={"id": last_id, "todo": todo})

def from_uuid(value: uuid.UUID) -> ulid.ULID:
    """
    Create a new :class:`~ulid.ulid.ULID` instance from the given :class:`~uuid.UUID` value.

    :param value: UUIDv4 value
    :type value: :class:`~uuid.UUID`
    :return: ULID from UUID value
    :rtype: :class:`~ulid.ulid.ULID`
    """
    return ulid.ULID(value.bytes)

def replace(table, field, a, b, **kwargs):
    """
    Convenience function to replace all occurrences of `a` with `b` under the
    given field. See also :func:`convert`.

    The ``where`` keyword argument can be given with a callable or expression
    which is evaluated on each row and which should return True if the
    conversion should be applied on that row, else False.

    """

    return convert(table, field, {a: b}, **kwargs)

def volume(self):
        """
        The volume of the primitive extrusion.

        Calculated from polygon and height to avoid mesh creation.

        Returns
        ----------
        volume: float, volume of 3D extrusion
        """
        volume = abs(self.primitive.polygon.area *
                     self.primitive.height)
        return volume

def row_to_dict(row):
    """Convert a table row to a dictionary."""
    o = {}
    for colname in row.colnames:

        if isinstance(row[colname], np.string_) and row[colname].dtype.kind in ['S', 'U']:
            o[colname] = str(row[colname])
        else:
            o[colname] = row[colname]

    return o

def install_from_zip(url):
    """Download and unzip from url."""
    fname = 'tmp.zip'
    downlad_file(url, fname)
    unzip_file(fname)
    print("Removing {}".format(fname))
    os.unlink(fname)

def validate_args(**args):
	"""
	function to check if input query is not None 
	and set missing arguments to default value
	"""
	if not args['query']:
		print("\nMissing required query argument.")
		sys.exit()

	for key in DEFAULTS:
		if key not in args:
			args[key] = DEFAULTS[key]

	return args

def locate(command, on):
    """Locate the command's man page."""
    location = find_page_location(command, on)
    click.echo(location)

def get_csrf_token(response):
    """
    Extract the CSRF token out of the "Set-Cookie" header of a response.
    """
    cookie_headers = [
        h.decode('ascii') for h in response.headers.getlist("Set-Cookie")
    ]
    if not cookie_headers:
        return None
    csrf_headers = [
        h for h in cookie_headers if h.startswith("csrftoken=")
    ]
    if not csrf_headers:
        return None
    match = re.match("csrftoken=([^ ;]+);", csrf_headers[-1])
    return match.group(1)

def denorm(self,arr):
        """Reverse the normalization done to a batch of images.

        Arguments:
            arr: of shape/size (N,3,sz,sz)
        """
        if type(arr) is not np.ndarray: arr = to_np(arr)
        if len(arr.shape)==3: arr = arr[None]
        return self.transform.denorm(np.rollaxis(arr,1,4))

def warn_deprecated(message, stacklevel=2):  # pragma: no cover
    """Warn deprecated."""

    warnings.warn(
        message,
        category=DeprecationWarning,
        stacklevel=stacklevel
    )

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

def get_iter_string_reader(stdin):
    """ return an iterator that returns a chunk of a string every time it is
    called.  notice that even though bufsize_type might be line buffered, we're
    not doing any line buffering here.  that's because our StreamBufferer
    handles all buffering.  we just need to return a reasonable-sized chunk. """
    bufsize = 1024
    iter_str = (stdin[i:i + bufsize] for i in range(0, len(stdin), bufsize))
    return get_iter_chunk_reader(iter_str)

def iter_lines(file_like: Iterable[str]) -> Generator[str, None, None]:
    """ Helper for iterating only nonempty lines without line breaks"""
    for line in file_like:
        line = line.rstrip('\r\n')
        if line:
            yield line

def zoom_out(self):
        """Scale the image down by one scale step."""
        if self._scalefactor >= self._sfmin:
            self._scalefactor -= 1
            self.scale_image()
            self._adjust_scrollbar(1/self._scalestep)
            self.sig_zoom_changed.emit(self.get_scaling())

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

def show_approx(self, numfmt='%.3g'):
        """Show the probabilities rounded and sorted by key, for the
        sake of portable doctests."""
        return ', '.join([('%s: ' + numfmt) % (v, p)
                          for (v, p) in sorted(self.prob.items())])

def start(args):
    """Run server with provided command line arguments.
    """
    application = tornado.web.Application([(r"/run", run.get_handler(args)),
                                           (r"/status", run.StatusHandler)])
    application.runmonitor = RunMonitor()
    application.listen(args.port)
    tornado.ioloop.IOLoop.instance().start()

def _loadfilepath(self, filepath, **kwargs):
        """This loads a geojson file into a geojson python
        dictionary using the json module.
        
        Note: to load with a different text encoding use the encoding argument.
        """
        with open(filepath, "r") as f:
            data = json.load(f, **kwargs)
        return data

def read_corpus(file_name):
    """
    Read and return the data from a corpus json file.
    """
    with io.open(file_name, encoding='utf-8') as data_file:
        return yaml.load(data_file)

def __init__(self, node_def, op, message):
        """Creates an `InvalidArgumentError`."""
        super(InvalidArgumentError, self).__init__(
            node_def, op, message, INVALID_ARGUMENT
        )

def _write_config(config, cfg_file):
    """
    Write a config object to the settings.cfg file.

    :param config: A ConfigParser object to write to the settings.cfg file.
    """
    directory = os.path.dirname(cfg_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(cfg_file, "w+") as output_file:
        config.write(output_file)

def connect(url, username, password):
    """
    Return a connected Bitbucket session
    """

    bb_session = stashy.connect(url, username, password)

    logger.info('Connected to: %s as %s', url, username)

    return bb_session

def send_email_message(self, recipient, subject, html_message, text_message, sender_email, sender_name):
        """ Send email message via Flask-Sendmail.

        Args:
            recipient: Email address or tuple of (Name, Email-address).
            subject: Subject line.
            html_message: The message body in HTML.
            text_message: The message body in plain text.
        """

        if not current_app.testing:  # pragma: no cover

            # Prepare email message
            from flask_sendmail import Message
            message = Message(
                subject,
                recipients=[recipient],
                html=html_message,
                body=text_message)

            # Send email message
            self.mail.send(message)

def chmod(scope, filename, mode):
    """
    Changes the permissions of the given file (or list of files)
    to the given mode. You probably want to use an octal representation
    for the integer, e.g. "chmod(myfile, 0644)".

    :type  filename: string
    :param filename: A filename.
    :type  mode: int
    :param mode: The access permissions.
    """
    for file in filename:
        os.chmod(file, mode[0])
    return True

def format_arg(value):
    """
    :param value:
        Some value in a dataset.
    :type value:
        varies
    :return:
        unicode representation of that value
    :rtype:
        `unicode`
    """
    translator = repr if isinstance(value, six.string_types) else six.text_type
    return translator(value)

def get_abi3_suffix():
    """Return the file extension for an abi3-compliant Extension()"""
    for suffix, _, _ in (s for s in imp.get_suffixes() if s[2] == imp.C_EXTENSION):
        if '.abi3' in suffix:  # Unix
            return suffix
        elif suffix == '.pyd':  # Windows
            return suffix

def unique(input_list):
    """
    Return a list of unique items (similar to set functionality).

    Parameters
    ----------
    input_list : list
        A list containg some items that can occur more than once.

    Returns
    -------
    list
        A list with only unique occurances of an item.

    """
    output = []
    for item in input_list:
        if item not in output:
            output.append(item)
    return output

def parse_comments_for_file(filename):
    """
    Return a list of all parsed comments in a file.  Mostly for testing &
    interactive use.
    """
    return [parse_comment(strip_stars(comment), next_line)
            for comment, next_line in get_doc_comments(read_file(filename))]

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

def __len__(self):
        """Return total data length of the list and its headers."""
        return self.chunk_length() + len(self.type) + len(self.header) + 4

def testable_memoized_property(func=None, key_factory=per_instance, **kwargs):
  """A variant of `memoized_property` that allows for setting of properties (for tests, etc)."""
  getter = memoized_method(func=func, key_factory=key_factory, **kwargs)

  def setter(self, val):
    with getter.put(self) as putter:
      putter(val)

  return property(fget=getter,
                  fset=setter,
                  fdel=lambda self: getter.forget(self))

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

def fill_nulls(self, col: str):
        """
        Fill all null values with NaN values in a column.
        Null values are ``None`` or en empty string

        :param col: column name
        :type col: str

        :example: ``ds.fill_nulls("mycol")``
        """
        n = [None, ""]
        try:
            self.df[col] = self.df[col].replace(n, nan)
        except Exception as e:
            self.err(e)

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

def _is_iterable(item):
    """ Checks if an item is iterable (list, tuple, generator), but not string """
    return isinstance(item, collections.Iterable) and not isinstance(item, six.string_types)

def _tuple_repr(data):
    """Return a repr() for a list/tuple"""
    if len(data) == 1:
        return "(%s,)" % rpr(data[0])
    else:
        return "(%s)" % ", ".join([rpr(x) for x in data])

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

def _covariance_matrix(self, type='noise'):
        """
        Constructs the covariance matrix from PCA
        residuals
        """
        if type == 'sampling':
            return self.sigma**2/(self.n-1)
        elif type == 'noise':
            return 4*self.sigma*N.var(self.rotated(), axis=0)

def isTestCaseDisabled(test_case_class, method_name):
    """
    I check to see if a method on a TestCase has been disabled via nose's
    convention for disabling a TestCase.  This makes it so that users can
    mix nose's parameterized tests with green as a runner.
    """
    test_method = getattr(test_case_class, method_name)
    return getattr(test_method, "__test__", 'not nose') is False

def ensure_tuple(obj):
    """Try and make the given argument into a tuple."""
    if obj is None:
        return tuple()
    if isinstance(obj, Iterable) and not isinstance(obj, six.string_types):
        return tuple(obj)
    return obj,

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

def standard_input():
    """Generator that yields lines from standard input."""
    with click.get_text_stream("stdin") as stdin:
        while stdin.readable():
            line = stdin.readline()
            if line:
                yield line.strip().encode("utf-8")

def factorial(n, mod=None):
    """Calculates factorial iteratively.
    If mod is not None, then return (n! % mod)
    Time Complexity - O(n)"""
    if not (isinstance(n, int) and n >= 0):
        raise ValueError("'n' must be a non-negative integer.")
    if mod is not None and not (isinstance(mod, int) and mod > 0):
        raise ValueError("'mod' must be a positive integer")
    result = 1
    if n == 0:
        return 1
    for i in range(2, n+1):
        result *= i
        if mod:
            result %= mod
    return result

def parse_querystring(self, req, name, field):
        """Pull a querystring value from the request."""
        return core.get_value(req.args, name, field)

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

def hmsToDeg(h, m, s):
    """Convert RA hours, minutes, seconds into an angle in degrees."""
    return h * degPerHMSHour + m * degPerHMSMin + s * degPerHMSSec

def parse_value(self, value):
        """Cast value to `bool`."""
        parsed = super(BoolField, self).parse_value(value)
        return bool(parsed) if parsed is not None else None

def print_with_header(header, message, color, indent=0):
    """
    Use one of the functions below for printing, not this one.
    """
    print()
    padding = ' ' * indent
    print(padding + color + BOLD + header + ENDC + color + message + ENDC)

def SwitchToThisWindow(handle: int) -> None:
    """
    SwitchToThisWindow from Win32.
    handle: int, the handle of a native window.
    """
    ctypes.windll.user32.SwitchToThisWindow(ctypes.c_void_p(handle), 1)

def __unixify(self, s):
        """ stupid windows. converts the backslash to forwardslash for consistency """
        return os.path.normpath(s).replace(os.sep, "/")

def is_integer(value: Any) -> bool:
    """Return true if a value is an integer number."""
    return (isinstance(value, int) and not isinstance(value, bool)) or (
        isinstance(value, float) and isfinite(value) and int(value) == value
    )

def _getVirtualScreenRect(self):
        """ The virtual screen is the bounding box containing all monitors.

        Not all regions in the virtual screen are actually visible. The (0,0) coordinate
        is the top left corner of the primary screen rather than the whole bounding box, so
        some regions of the virtual screen may have negative coordinates if another screen
        is positioned in Windows as further to the left or above the primary screen.

        Returns the rect as (x, y, w, h)
        """
        SM_XVIRTUALSCREEN = 76  # Left of virtual screen
        SM_YVIRTUALSCREEN = 77  # Top of virtual screen
        SM_CXVIRTUALSCREEN = 78 # Width of virtual screen
        SM_CYVIRTUALSCREEN = 79 # Height of virtual screen

        return (self._user32.GetSystemMetrics(SM_XVIRTUALSCREEN), \
                self._user32.GetSystemMetrics(SM_YVIRTUALSCREEN), \
                self._user32.GetSystemMetrics(SM_CXVIRTUALSCREEN), \
                self._user32.GetSystemMetrics(SM_CYVIRTUALSCREEN))

def run_test(func, fobj):
    """Run func with argument fobj and measure execution time.
    @param  func:   function for test
    @param  fobj:   data for test
    @return:        execution time
    """
    gc.disable()
    try:
        begin = time.time()
        func(fobj)
        end = time.time()
    finally:
        gc.enable()
    return end - begin

def perform_permissions_check(self, user, obj, perms):
        """ Performs the permission check. """
        return self.request.forum_permission_handler.can_download_files(obj, user)

def OnUpdateFigurePanel(self, event):
        """Redraw event handler for the figure panel"""

        if self.updating:
            return

        self.updating = True
        self.figure_panel.update(self.get_figure(self.code))
        self.updating = False

def pointer(self):
        """Get a ctypes void pointer to the memory mapped region.

        :type: ctypes.c_void_p
        """
        return ctypes.cast(ctypes.pointer(ctypes.c_uint8.from_buffer(self.mapping, 0)), ctypes.c_void_p)

def round_to_n(x, n):
    """
    Round to sig figs
    """
    return round(x, -int(np.floor(np.log10(x))) + (n - 1))

def group_by(iterable, key_func):
    """Wrap itertools.groupby to make life easier."""
    groups = (
        list(sub) for key, sub in groupby(iterable, key_func)
    )
    return zip(groups, groups)

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

def loganalytics_data_plane_client(cli_ctx, _):
    """Initialize Log Analytics data client for use with CLI."""
    from .vendored_sdks.loganalytics import LogAnalyticsDataClient
    from azure.cli.core._profile import Profile
    profile = Profile(cli_ctx=cli_ctx)
    cred, _, _ = profile.get_login_credentials(
        resource="https://api.loganalytics.io")
    return LogAnalyticsDataClient(cred)

def fig2x(figure, format):
    """Returns svg from matplotlib chart"""

    # Save svg to file like object svg_io
    io = StringIO()
    figure.savefig(io, format=format)

    # Rewind the file like object
    io.seek(0)

    data = io.getvalue()
    io.close()

    return data

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

def _is_subsequence_of(self, sub, sup):
        """
        Parameters
        ----------
        sub : str
        sup : str

        Returns
        -------
        bool
        """
        return bool(re.search(".*".join(sub), sup))

def push(self, el):
        """ Put a new element in the queue. """
        count = next(self.counter)
        heapq.heappush(self._queue, (el, count))

def text_remove_empty_lines(text):
    """
    Whitespace normalization:

      - Strip empty lines
      - Strip trailing whitespace
    """
    lines = [ line.rstrip()  for line in text.splitlines()  if line.strip() ]
    return "\n".join(lines)

def _tree_line(self, no_type: bool = False) -> str:
        """Return the receiver's contribution to tree diagram."""
        return self._tree_line_prefix() + " " + self.iname()

def login(self, username, password=None, token=None):
        """Login user for protected API calls."""
        self.session.basic_auth(username, password)

def onchange(self, value):
        """Called when a new DropDownItem gets selected.
        """
        log.debug('combo box. selected %s' % value)
        self.select_by_value(value)
        return (value, )
