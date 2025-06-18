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

def _stdin_ready_posix():
    """Return True if there's something to read on stdin (posix version)."""
    infds, outfds, erfds = select.select([sys.stdin],[],[],0)
    return bool(infds)

def convert_camel_case_to_snake_case(name):
    """Convert CamelCase to snake_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def keys_to_snake_case(camel_case_dict):
    """
    Make a copy of a dictionary with all keys converted to snake case. This is just calls to_snake_case on
    each of the keys in the dictionary and returns a new dictionary.

    :param camel_case_dict: Dictionary with the keys to convert.
    :type camel_case_dict: Dictionary.

    :return: Dictionary with the keys converted to snake case.
    """
    return dict((to_snake_case(key), value) for (key, value) in camel_case_dict.items())

def adjust_bounding_box(bbox):
    """Adjust the bounding box as specified by user.
    Returns the adjusted bounding box.

    - bbox: Bounding box computed from the canvas drawings.
    It must be a four-tuple of numbers.
    """
    for i in range(0, 4):
        if i in bounding_box:
            bbox[i] = bounding_box[i]
        else:
            bbox[i] += delta_bounding_box[i]
    return bbox

def readme(filename, encoding='utf8'):
    """
    Read the contents of a file
    """

    with io.open(filename, encoding=encoding) as source:
        return source.read()

def _parallel_compare_helper(class_obj, pairs, x, x_link=None):
    """Internal function to overcome pickling problem in python2."""
    return class_obj._compute(pairs, x, x_link)

def is_all_field_none(self):
        """
        :rtype: bool
        """

        if self._type_ is not None:
            return False

        if self._value is not None:
            return False

        if self._name is not None:
            return False

        return True

def sortable_title(instance):
    """Uses the default Plone sortable_text index lower-case
    """
    title = plone_sortable_title(instance)
    if safe_callable(title):
        title = title()
    return title.lower()

def put_pidfile( pidfile_path, pid ):
    """
    Put a PID into a pidfile
    """
    with open( pidfile_path, "w" ) as f:
        f.write("%s" % pid)
        os.fsync(f.fileno())

    return

def _namematcher(regex):
    """Checks if a target name matches with an input regular expression."""

    matcher = re_compile(regex)

    def match(target):
        target_name = getattr(target, '__name__', '')
        result = matcher.match(target_name)
        return result

    return match

def select_fields_as_sql(self):
        """
        Returns the selected fields or expressions as a SQL string.
        """
        return comma_join(list(self._fields) + ['%s AS %s' % (v, k) for k, v in self._calculated_fields.items()])

def to_linspace(self):
        """
        convert from arange to linspace
        """
        num = int((self.stop-self.start)/(self.step))
        return Linspace(self.start, self.stop-self.step, num)

def connect(self):
        """Connects to the given host"""
        self.socket = socket.create_connection(self.address, self.timeout)

def force_iterable(f):
    """Will make any functions return an iterable objects by wrapping its result in a list."""
    def wrapper(*args, **kwargs):
        r = f(*args, **kwargs)
        if hasattr(r, '__iter__'):
            return r
        else:
            return [r]
    return wrapper

def check_exists(filename, oappend=False):
    """
    Avoid overwriting some files accidentally.
    """
    if op.exists(filename):
        if oappend:
            return oappend
        logging.error("`{0}` found, overwrite (Y/N)?".format(filename))
        overwrite = (raw_input() == 'Y')
    else:
        overwrite = True

    return overwrite

def reset(self):
		"""
		Resets the iterator to the start.

		Any remaining values in the current iteration are discarded.
		"""
		self.__iterator, self.__saved = itertools.tee(self.__saved)

def _clean_str(self, s):
        """ Returns a lowercase string with punctuation and bad chars removed
        :param s: string to clean
        """
        return s.translate(str.maketrans('', '', punctuation)).replace('\u200b', " ").strip().lower()

def strtobytes(input, encoding):
    """Take a str and transform it into a byte array."""
    py_version = sys.version_info[0]
    if py_version >= 3:
        return _strtobytes_py3(input, encoding)
    return _strtobytes_py2(input, encoding)

def __enter__(self):
        """ Implements the context manager protocol. Specially useful for asserting exceptions
        """
        clone = self.clone()
        self._contexts.append(clone)
        self.reset()
        return self

def _read_preference_for(self, session):
        """Read only access to the read preference of this instance or session.
        """
        # Override this operation's read preference with the transaction's.
        if session:
            return session._txn_read_preference() or self.__read_preference
        return self.__read_preference

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

def _centroids(n_clusters: int, points: List[List[float]]) -> List[List[float]]:
    """ Return n_clusters centroids of points
    """

    k_means = KMeans(n_clusters=n_clusters)
    k_means.fit(points)

    closest, _ = pairwise_distances_argmin_min(k_means.cluster_centers_, points)

    return list(map(list, np.array(points)[closest.tolist()]))

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

def __consistent_isoformat_utc(datetime_val):
        """
        Function that does what isoformat does but it actually does the same
        every time instead of randomly doing different things on some systems
        and also it represents that time as the equivalent UTC time.
        """
        isotime = datetime_val.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%S%z")
        if isotime[-2] != ":":
            isotime = isotime[:-2] + ":" + isotime[-2:]
        return isotime

def uncomment_line(line, prefix):
    """Remove prefix (and space) from line"""
    if not prefix:
        return line
    if line.startswith(prefix + ' '):
        return line[len(prefix) + 1:]
    if line.startswith(prefix):
        return line[len(prefix):]
    return line

def signed_area(coords):
    """Return the signed area enclosed by a ring using the linear time
    algorithm. A value >= 0 indicates a counter-clockwise oriented ring.
    """
    xs, ys = map(list, zip(*coords))
    xs.append(xs[1])
    ys.append(ys[1])
    return sum(xs[i]*(ys[i+1]-ys[i-1]) for i in range(1, len(coords)))/2.0

def update_cache(self, data):
        """Update a cached value."""
        UTILS.update(self._cache, data)
        self._save_cache()

def get_future_days(self):
        """Return only future Day objects."""
        today = timezone.now().date()

        return Day.objects.filter(date__gte=today)

def execute_in_background(self):
        """Executes a (shell) command in the background

        :return: the process' pid
        """
        # http://stackoverflow.com/questions/1605520
        args = shlex.split(self.cmd)
        p = Popen(args)
        return p.pid

def __set__(self, instance, value):
        """ Set a related object for an instance. """

        self.map[id(instance)] = (weakref.ref(instance), value)

def get_method_names(obj):
        """
        Gets names of all methods implemented in specified object.

        :param obj: an object to introspect.

        :return: a list with method names.
        """
        method_names = []
        
        for method_name in dir(obj):

            method = getattr(obj, method_name)

            if MethodReflector._is_method(method, method_name):
                method_names.append(method_name)

        return method_names

def parse_query_string(query):
    """
    parse_query_string:
    very simplistic. won't do the right thing with list values
    """
    result = {}
    qparts = query.split('&')
    for item in qparts:
        key, value = item.split('=')
        key = key.strip()
        value = value.strip()
        result[key] = unquote_plus(value)
    return result

def color_func(func_name):
    """
    Call color function base on name
    """
    if str(func_name).isdigit():
        return term_color(int(func_name))
    return globals()[func_name]

def get_all_files(folder):
    """
    Generator that loops through all absolute paths of the files within folder

    Parameters
    ----------
    folder: str
    Root folder start point for recursive search.

    Yields
    ------
    fpath: str
    Absolute path of one file in the folders
    """
    for path, dirlist, filelist in os.walk(folder):
        for fn in filelist:
            yield op.join(path, fn)

def _format_list(result):
    """Format list responses into a table."""

    if not result:
        return result

    if isinstance(result[0], dict):
        return _format_list_objects(result)

    table = Table(['value'])
    for item in result:
        table.add_row([iter_to_table(item)])
    return table

def read_proto_object(fobj, klass):
    """Read a block of data and parse using the given protobuf object."""
    log.debug('%s chunk', klass.__name__)
    obj = klass()
    obj.ParseFromString(read_block(fobj))
    log.debug('Header: %s', str(obj))
    return obj

def unique_inverse(item_list):
    """
    Like np.unique(item_list, return_inverse=True)
    """
    import utool as ut
    unique_items = ut.unique(item_list)
    inverse = list_alignment(unique_items, item_list)
    return unique_items, inverse

def last(self):
        """Get the last object in file."""
        # End of file
        self.__file.seek(0, 2)

        # Get the last struct
        data = self.get(self.length - 1)

        return data

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

def get_last_commit(git_path=None):
    """
    Get the HEAD commit SHA1 of repository in current dir.
    """
    if git_path is None: git_path = GIT_PATH
    line = get_last_commit_line(git_path)
    revision_id = line.split()[1]
    return revision_id

def apply_argument_parser(argumentsParser, options=None):
    """ Apply the argument parser. """
    if options is not None:
        args = argumentsParser.parse_args(options)
    else:
        args = argumentsParser.parse_args()
    return args

def _join_masks_from_masked_array(data):
    """Union of masks."""
    if not isinstance(data.mask, np.ndarray):
        # workaround to handle mask compressed to single value
        mask = np.empty(data.data.shape, dtype=np.bool)
        mask.fill(data.mask)
        return mask
    mask = data.mask[0].copy()
    for i in range(1, len(data.mask)):
        mask = np.logical_or(mask, data.mask[i])
    return mask[np.newaxis, :, :]

def _unordered_iterator(self):
        """
        Return the value of each QuerySet, but also add the '#' property to each
        return item.
        """
        for i, qs in zip(self._queryset_idxs, self._querysets):
            for item in qs:
                setattr(item, '#', i)
                yield item

def is_instance_or_subclass(val, class_):
    """Return True if ``val`` is either a subclass or instance of ``class_``."""
    try:
        return issubclass(val, class_)
    except TypeError:
        return isinstance(val, class_)

def set_gradclip_const(self, min_value, max_value):
        """
        Configure constant clipping settings.


        :param min_value: the minimum value to clip by
        :param max_value: the maxmimum value to clip by
        """
        callBigDlFunc(self.bigdl_type, "setConstantClip", self.value, min_value, max_value)

def mutating_method(func):
    """Decorator for methods that are allowed to modify immutable objects"""
    def wrapper(self, *__args, **__kwargs):
        old_mutable = self._mutable
        self._mutable = True
        try:
            # Call the wrapped function
            return func(self, *__args, **__kwargs)
        finally:
            self._mutable = old_mutable
    return wrapper

def stddev(values, meanval=None):  #from AI: A Modern Appproach
    """The standard deviation of a set of values.
    Pass in the mean if you already know it."""
    if meanval == None: meanval = mean(values)
    return math.sqrt( sum([(x - meanval)**2 for x in values]) / (len(values)-1) )

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

def get_table_pos(self, tablename):
        """
        :param str tablename: Name of table to get position of.
        :return: Upper left (row, col) coordinate of the named table.
        """
        _table, (row, col) = self.__tables[tablename]
        return (row, col)

def graph_key_from_tag(tag, entity_index):
    """Returns a key from a tag entity

    Args:
        tag (tag) : this is the tag selected to get the key from
        entity_index (int) : this is the index of the tagged entity

    Returns:
        str : String representing the key for the given tagged entity.
    """
    start_token = tag.get('start_token')
    entity = tag.get('entities', [])[entity_index]
    return str(start_token) + '-' + entity.get('key') + '-' + str(entity.get('confidence'))

def format_result(input):
        """From: http://stackoverflow.com/questions/13062300/convert-a-dict-to-sorted-dict-in-python
        """
        items = list(iteritems(input))
        return OrderedDict(sorted(items, key=lambda x: x[0]))

def buttonUp(self, button=mouse.LEFT):
        """ Releases the specified mouse button.

        Use Mouse.LEFT, Mouse.MIDDLE, Mouse.RIGHT
        """
        self._lock.acquire()
        mouse.release(button)
        self._lock.release()

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

def make_key(self, key, version=None):
        """RedisCache will set prefix+version as prefix for each key."""
        return '{}:{}:{}'.format(
            self.prefix,
            version or self.version,
            key,
        )

def _index_ordering(redshift_list):
        """

        :param redshift_list: list of redshifts
        :return: indexes in acending order to be evaluated (from z=0 to z=z_source)
        """
        redshift_list = np.array(redshift_list)
        sort_index = np.argsort(redshift_list)
        return sort_index

def create_tmpfile(self, content):
        """ Utility method to create temp files. These are cleaned at the end of the test """
        # Not using a context manager to avoid unneccessary identation in test code
        tmpfile, tmpfilepath = tempfile.mkstemp()
        self.tmpfiles.append(tmpfilepath)
        with os.fdopen(tmpfile, "w") as f:
            f.write(content)
        return tmpfilepath

def c_array(ctype, values):
    """Convert a python string to c array."""
    if isinstance(values, np.ndarray) and values.dtype.itemsize == ctypes.sizeof(ctype):
        return (ctype * len(values)).from_buffer_copy(values)
    return (ctype * len(values))(*values)

def is_static(self, filename):
        """Check if a file is a static file (which should be copied, rather
        than compiled using Jinja2).

        A file is considered static if it lives in any of the directories
        specified in ``staticpaths``.

        :param filename: the name of the file to check

        """
        if self.staticpaths is None:
            # We're not using static file support
            return False

        for path in self.staticpaths:
            if filename.startswith(path):
                return True
        return False

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

def autobuild_python_test(path):
    """Add pytest unit tests to be built as part of build/test/output."""

    env = Environment(tools=[])
    target = env.Command(['build/test/output/pytest.log'], [path],
                         action=env.Action(run_pytest, "Running python unit tests"))
    env.AlwaysBuild(target)

def connect(*args, **kwargs):
    """Create database connection, use TraceCursor as the cursor_factory."""
    kwargs['cursor_factory'] = TraceCursor
    conn = pg_connect(*args, **kwargs)
    return conn

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

def asynchronous(function, event):
    """
    Runs the function asynchronously taking care of exceptions.
    """
    thread = Thread(target=synchronous, args=(function, event))
    thread.daemon = True
    thread.start()

def get_last_week_range(weekday_start="Sunday"):
    """ Gets the date for the first and the last day of the previous complete week.

    :param weekday_start: Either "Monday" or "Sunday", indicating the first day of the week.
    :returns: A tuple containing two date objects, for the first and the last day of the week
              respectively.
    """
    today = date.today()
    # Get the first day of the past complete week.
    start_of_week = snap_to_beginning_of_week(today, weekday_start) - timedelta(weeks=1)
    end_of_week = start_of_week + timedelta(days=6)
    return (start_of_week, end_of_week)

def compute_gradient(self):
        """Compute the gradient of the current model using the training set
        """
        delta = self.predict(self.X) - self.y
        return delta.dot(self.X) / len(self.X)

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

def set_stop_handler(self):
        """
        Initializes functions that are invoked when the user or OS wants to kill this process.
        :return:
        """
        signal.signal(signal.SIGTERM, self.graceful_stop)
        signal.signal(signal.SIGABRT, self.graceful_stop)
        signal.signal(signal.SIGINT, self.graceful_stop)

def expandpath(path):
    """
    Expand a filesystem path that may or may not contain user/env vars.

    :param str path: path to expand
    :return str: expanded version of input path
    """
    return os.path.expandvars(os.path.expanduser(path)).replace("//", "/")

def get_url(self, cmd, **args):
        """Expand the request URL for a request."""
        return self.http.base_url + self._mkurl(cmd, *args)

def find_geom(geom, geoms):
    """
    Returns the index of a geometry in a list of geometries avoiding
    expensive equality checks of `in` operator.
    """
    for i, g in enumerate(geoms):
        if g is geom:
            return i

def is_iterable(value):
    """must be an iterable (list, array, tuple)"""
    return isinstance(value, np.ndarray) or isinstance(value, list) or isinstance(value, tuple), value

def memory_usage(self, deep=False):
        """
        Memory usage of my values

        Parameters
        ----------
        deep : bool
            Introspect the data deeply, interrogate
            `object` dtypes for system-level memory consumption

        Returns
        -------
        bytes used

        Notes
        -----
        Memory usage does not include memory consumed by elements that
        are not components of the array if deep=False

        See Also
        --------
        numpy.ndarray.nbytes
        """
        return self._codes.nbytes + self.dtype.categories.memory_usage(
            deep=deep)

def get_now_utc_notz_datetime() -> datetime.datetime:
    """
    Get the UTC time now, but with no timezone information,
    in :class:`datetime.datetime` format.
    """
    now = datetime.datetime.utcnow()
    return now.replace(tzinfo=None)

def __getattr__(self, *args, **kwargs):
        """
        Magic method dispatcher
        """

        return xmlrpc.client._Method(self.__request, *args, **kwargs)

def _iterable_to_varargs_method(func):
    """decorator to convert a method taking a iterable to a *args one"""
    def wrapped(self, *args, **kwargs):
        return func(self, args, **kwargs)
    return wrapped

def datetime64_to_datetime(dt):
    """ convert numpy's datetime64 to datetime """
    dt64 = np.datetime64(dt)
    ts = (dt64 - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
    return datetime.datetime.utcfromtimestamp(ts)

def get_dict_for_attrs(obj, attrs):
    """
    Returns dictionary for each attribute from given ``obj``.
    """
    data = {}
    for attr in attrs:
        data[attr] = getattr(obj, attr)
    return data

def fit_transform(self, raw_documents, y=None):
        """Learn vocabulary and idf, return term-document matrix.
        This is equivalent to fit followed by transform, but more efficiently
        implemented.
        Parameters
        ----------
        raw_documents : iterable
            an iterable which yields either str, unicode or file objects
        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            Tf-idf-weighted document-term matrix.
        """
        documents = super(TfidfVectorizer, self).fit_transform(
            raw_documents=raw_documents, y=y)
        count = CountVectorizer(encoding=self.encoding,
                                decode_error=self.decode_error,
                                strip_accents=self.strip_accents,
                                lowercase=self.lowercase,
                                preprocessor=self.preprocessor,
                                tokenizer=self.tokenizer,
                                stop_words=self.stop_words,
                                token_pattern=self.token_pattern,
                                ngram_range=self.ngram_range,
                                analyzer=self.analyzer,
                                max_df=self.max_df,
                                min_df=self.min_df,
                                max_features=self.max_features,
                                vocabulary=self.vocabulary_,
                                binary=self.binary,
                                dtype=self.dtype)
        count.fit_transform(raw_documents=raw_documents, y=y)
        self.period_ = count.period_
        self.df_ = count.df_
        self.n = count.n
        return documents

def _image_field(self):
        """
        Try to automatically detect an image field
        """
        for field in self.model._meta.fields:
            if isinstance(field, ImageField):
                return field.name

def _concatenate_virtual_arrays(arrs, cols=None, scaling=None):
    """Return a virtual concatenate of several NumPy arrays."""
    return None if not len(arrs) else ConcatenatedArrays(arrs, cols,
                                                         scaling=scaling)

def parse_datetime(dt_str):
    """Parse datetime."""
    date_format = "%Y-%m-%dT%H:%M:%S %z"
    dt_str = dt_str.replace("Z", " +0000")
    return datetime.datetime.strptime(dt_str, date_format)

def __get_xml_text(root):
    """ Return the text for the given root node (xml.dom.minidom). """
    txt = ""
    for e in root.childNodes:
        if (e.nodeType == e.TEXT_NODE):
            txt += e.data
    return txt

def public(self) -> 'PrettyDir':
        """Returns public attributes of the inspected object."""
        return PrettyDir(
            self.obj, [pattr for pattr in self.pattrs if not pattr.name.startswith('_')]
        )

def cmd_reindex():
    """Uses CREATE INDEX CONCURRENTLY to create a duplicate index, then tries to swap the new index for the original.

    The index swap is done using a short lock timeout to prevent it from interfering with running queries. Retries until
    the rename succeeds.
    """
    db = connect(args.database)
    for idx in args.indexes:
        pg_reindex(db, idx)

def rmglob(pattern: str) -> None:
    """
    Deletes all files whose filename matches the glob ``pattern`` (via
    :func:`glob.glob`).
    """
    for f in glob.glob(pattern):
        os.remove(f)

def string_to_identity(identity_str):
    """Parse string into Identity dictionary."""
    m = _identity_regexp.match(identity_str)
    result = m.groupdict()
    log.debug('parsed identity: %s', result)
    return {k: v for k, v in result.items() if v}

def linear_variogram_model(m, d):
    """Linear model, m is [slope, nugget]"""
    slope = float(m[0])
    nugget = float(m[1])
    return slope * d + nugget

def manhattan_distance_numpy(object1, object2):
    """!
    @brief Calculate Manhattan distance between two objects using numpy.

    @param[in] object1 (array_like): The first array_like object.
    @param[in] object2 (array_like): The second array_like object.

    @return (double) Manhattan distance between two objects.

    """
    return numpy.sum(numpy.absolute(object1 - object2), axis=1).T

def random_letters(n):
    """
    Generate a random string from a-zA-Z
    :param n: length of the string
    :return: the random string
    """
    return ''.join(random.SystemRandom().choice(string.ascii_letters) for _ in range(n))

def run(self):
        """Run the event loop."""
        self.signal_init()
        self.listen_init()
        self.logger.info('starting')
        self.loop.start()

def is_timestamp(instance):
    """Validates data is a timestamp"""
    if not isinstance(instance, (int, str)):
        return True
    return datetime.fromtimestamp(int(instance))

def selectnone(table, field, complement=False):
    """Select rows where the given field is `None`."""

    return select(table, field, lambda v: v is None, complement=complement)

def _elapsed(self):
        """ Returns elapsed time at update. """
        self.last_time = time.time()
        return self.last_time - self.start

def is_cyclic(graph):
    """
    Return True if the directed graph g has a cycle. The directed graph
    should be represented as a dictionary mapping of edges for each node.
    """
    path = set()

    def visit(vertex):
        path.add(vertex)
        for neighbour in graph.get(vertex, ()):
            if neighbour in path or visit(neighbour):
                return True
        path.remove(vertex)
        return False

    return any(visit(v) for v in graph)

def is_writable_by_others(filename):
    """Check if file or directory is world writable."""
    mode = os.stat(filename)[stat.ST_MODE]
    return mode & stat.S_IWOTH

def print_float(self, value, decimal_digits=2, justify_right=True):
        """Print a numeric value to the display.  If value is negative
        it will be printed with a leading minus sign.  Decimal digits is the
        desired number of digits after the decimal point.
        """
        format_string = '{{0:0.{0}F}}'.format(decimal_digits)
        self.print_number_str(format_string.format(value), justify_right)

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

def get_default_bucket_key(buckets: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Returns the default bucket from a list of buckets, i.e. the largest bucket.

    :param buckets: List of buckets.
    :return: The largest bucket in the list.
    """
    return max(buckets)

def resize_image(self, data, size):
        """ Resizes the given image to fit inside a box of the given size. """
        from machina.core.compat import PILImage as Image
        image = Image.open(BytesIO(data))

        # Resize!
        image.thumbnail(size, Image.ANTIALIAS)

        string = BytesIO()
        image.save(string, format='PNG')
        return string.getvalue()

def __getattribute__(self, attr):
        """Retrieve attr from current active etree implementation"""
        if (attr not in object.__getattribute__(self, '__dict__')
                and attr not in Etree.__dict__):
            return object.__getattribute__(self._etree, attr)
        return object.__getattribute__(self, attr)

def _psutil_kill_pid(pid):
    """
    http://stackoverflow.com/questions/1230669/subprocess-deleting-child-processes-in-windows
    """
    try:
        parent = Process(pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except NoSuchProcess:
        return

def do_forceescape(value):
    """Enforce HTML escaping.  This will probably double escape variables."""
    if hasattr(value, '__html__'):
        value = value.__html__()
    return escape(text_type(value))

def label_from_bin(buf):
    """
    Converts binary representation label to integer.

    :param buf: Binary representation of label.
    :return: MPLS Label and BoS bit.
    """

    mpls_label = type_desc.Int3.to_user(six.binary_type(buf))
    return mpls_label >> 4, mpls_label & 1

def glpk_read_cplex(path):
    """Reads cplex file and returns glpk problem.

    Returns
    -------
    glp_prob
        A glpk problems (same type as returned by glp_create_prob)
    """
    from swiglpk import glp_create_prob, glp_read_lp

    problem = glp_create_prob()
    glp_read_lp(problem, None, path)
    return problem

def normalize(data):
    """
    Function to normalize data to have mean 0 and unity standard deviation
    (also called z-transform)
    
    
    Parameters
    ----------
    data : numpy.ndarray
    
    
    Returns
    -------
    numpy.ndarray
        z-transform of input array
    
    """
    data = data.astype(float)
    data -= data.mean()
    
    return data / data.std()

async def connect(self):
        """
        Connects to the voice channel associated with this Player.
        """
        await self.node.join_voice_channel(self.channel.guild.id, self.channel.id)

def terminate(self):
    """Override of PantsService.terminate() that cleans up when the Pailgun server is terminated."""
    # Tear down the Pailgun TCPServer.
    if self.pailgun:
      self.pailgun.server_close()

    super(PailgunService, self).terminate()

def _stream_docker_logs(self):
        """Stream stdout and stderr from the task container to this
        process's stdout and stderr, respectively.
        """
        thread = threading.Thread(target=self._stderr_stream_worker)
        thread.start()
        for line in self.docker_client.logs(self.container, stdout=True,
                                            stderr=False, stream=True):
            sys.stdout.write(line)
        thread.join()

def _is_valid_url(url):
        """ Helper function to validate that URLs are well formed, i.e that it contains a valid
            protocol and a valid domain. It does not actually check if the URL exists
        """
        try:
            parsed = urlparse(url)
            mandatory_parts = [parsed.scheme, parsed.netloc]
            return all(mandatory_parts)
        except:
            return False

def check(modname):
    """Check if required dependency is installed"""
    for dependency in DEPENDENCIES:
        if dependency.modname == modname:
            return dependency.check()
    else:
        raise RuntimeError("Unkwown dependency %s" % modname)

def __copy__(self):
        """A magic method to implement shallow copy behavior."""
        return self.__class__.load(self.dump(), context=self.context)

def clearImg(self):
        """Clears the current image"""
        self.img.setImage(np.array([[0]]))
        self.img.image = None

def snake_to_camel(value):
    """
    Converts a snake_case_string to a camelCaseString.

    >>> snake_to_camel("foo_bar_baz")
    'fooBarBaz'
    """
    camel = "".join(word.title() for word in value.split("_"))
    return value[:1].lower() + camel[1:]

def yaml_to_param(obj, name):
	"""
	Return the top-level element of a document sub-tree containing the
	YAML serialization of a Python object.
	"""
	return from_pyvalue(u"yaml:%s" % name, unicode(yaml.dump(obj)))

def getRandomBinaryTreeLeafNode(binaryTree):
    """Get random binary tree node.
    """
    if binaryTree.internal == True:
        if random.random() > 0.5:
            return getRandomBinaryTreeLeafNode(binaryTree.left)
        else:
            return getRandomBinaryTreeLeafNode(binaryTree.right)
    else:
        return binaryTree

def get_shape(self):
		"""
		Return a tuple of this array's dimensions.  This is done by
		querying the Dim children.  Note that once it has been
		created, it is also possible to examine an Array object's
		.array attribute directly, and doing that is much faster.
		"""
		return tuple(int(c.pcdata) for c in self.getElementsByTagName(ligolw.Dim.tagName))[::-1]

def venv():
    """Install venv + deps."""
    try:
        import virtualenv  # NOQA
    except ImportError:
        sh("%s -m pip install virtualenv" % PYTHON)
    if not os.path.isdir("venv"):
        sh("%s -m virtualenv venv" % PYTHON)
    sh("venv\\Scripts\\pip install -r %s" % (REQUIREMENTS_TXT))

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

def get_handler(self, *args, **options):
        """
        Returns the default WSGI handler for the runner.
        """
        handler = get_internal_wsgi_application()
        from django.contrib.staticfiles.handlers import StaticFilesHandler
        return StaticFilesHandler(handler)

def time2seconds(t):
    """Returns seconds since 0h00."""
    return t.hour * 3600 + t.minute * 60 + t.second + float(t.microsecond) / 1e6

def __get__(self, obj, objtype):
        """ Support instance methods """
        import functools
        return functools.partial(self.__call__, obj)

def cli(yamlfile, format, output):
    """ Generate an OWL representation of a biolink model """
    print(OwlSchemaGenerator(yamlfile, format).serialize(output=output))

def to_bytes(self):
		"""Convert the entire image to bytes.
		
		:rtype: bytes
		"""
		chunks = [PNG_SIGN]
		chunks.extend(c[1] for c in self.chunks)
		return b"".join(chunks)

def update(kernel=False):
    """
    Upgrade all packages, skip obsoletes if ``obsoletes=0`` in ``yum.conf``.

    Exclude *kernel* upgrades by default.
    """
    manager = MANAGER
    cmds = {'yum -y --color=never': {False: '--exclude=kernel* update', True: 'update'}}
    cmd = cmds[manager][kernel]
    run_as_root("%(manager)s %(cmd)s" % locals())

def load(cls, tree_path):
        """Create a new instance from a file."""
        with open(tree_path) as f:
            tree_dict = json.load(f)

        return cls.from_dict(tree_dict)

def pickle_load(fname):
    """return the contents of a pickle file"""
    assert type(fname) is str and os.path.exists(fname)
    print("loaded",fname)
    return pickle.load(open(fname,"rb"))

def fn_min(self, a, axis=None):
        """
        Return the minimum of an array, ignoring any NaNs.

        :param a: The array.
        :return: The minimum value of the array.
        """

        return numpy.nanmin(self._to_ndarray(a), axis=axis)

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

def get_days_in_month(year: int, month: int) -> int:
    """ Returns number of days in the given month.
    1-based numbers as arguments. i.e. November = 11 """
    month_range = calendar.monthrange(year, month)
    return month_range[1]

def _xls2col_widths(self, worksheet, tab):
        """Updates col_widths in code_array"""

        for col in xrange(worksheet.ncols):
            try:
                xls_width = worksheet.colinfo_map[col].width
                pys_width = self.xls_width2pys_width(xls_width)
                self.code_array.col_widths[col, tab] = pys_width

            except KeyError:
                pass

def sync_s3(self):
        """Walk the media/static directories and syncs files to S3"""
        bucket, key = self.open_s3()
        for directory in self.DIRECTORIES:
            for root, dirs, files in os.walk(directory):
                self.upload_s3((bucket, key, self.AWS_BUCKET_NAME, directory), root, files, dirs)

def has_add_permission(self, request):
        """ Can add this object """
        return request.user.is_authenticated and request.user.is_active and request.user.is_staff

def from_df(data_frame):
        """Parses data and builds an instance of this class

        :param data_frame: pandas DataFrame
        :return: SqlTable
        """
        labels = data_frame.keys().tolist()
        data = data_frame.values.tolist()
        return SqlTable(labels, data, "{:.3f}", "\n")

def test():
    """Interactive test run."""
    try:
        while 1:
            x, digs = input('Enter (x, digs): ')
            print x, fix(x, digs), sci(x, digs)
    except (EOFError, KeyboardInterrupt):
        pass

def revrank_dict(dict, key=lambda t: t[1], as_tuple=False):
    """ Reverse sorts a #dict by a given key, optionally returning it as a
        #tuple. By default, the @dict is sorted by it's value.

        @dict: the #dict you wish to sorts
        @key: the #sorted key to use
        @as_tuple: returns result as a #tuple ((k, v),...)

        -> :class:OrderedDict or #tuple
    """
    sorted_list = sorted(dict.items(), key=key, reverse=True)
    return OrderedDict(sorted_list) if not as_tuple else tuple(sorted_list)

def count(lines):
  """ Counts the word frequences in a list of sentences.

  Note:
    This is a helper function for parallel execution of `Vocabulary.from_text`
    method.
  """
  words = [w for l in lines for w in l.strip().split()]
  return Counter(words)

def logout(self):
        """
            Logout from the remote server.
        """
        self.client.write('exit\r\n')
        self.client.read_all()
        self.client.close()

def save_partial(self, obj):
        """Partial objects do not serialize correctly in python2.x -- this fixes the bugs"""
        self.save_reduce(_genpartial, (obj.func, obj.args, obj.keywords))

def quaternion_to_rotation_matrix(quaternion):
    """Compute the rotation matrix representated by the quaternion"""
    c, x, y, z = quaternion
    return np.array([
        [c*c + x*x - y*y - z*z, 2*x*y - 2*c*z,         2*x*z + 2*c*y        ],
        [2*x*y + 2*c*z,         c*c - x*x + y*y - z*z, 2*y*z - 2*c*x        ],
        [2*x*z - 2*c*y,         2*y*z + 2*c*x,         c*c - x*x - y*y + z*z]
    ], float)

def count(self, X):
        """
        Called from the fit method, this method gets all the
        words from the corpus and their corresponding frequency
        counts.

        Parameters
        ----------

        X : ndarray or masked ndarray
            Pass in the matrix of vectorized documents, can be masked in
            order to sum the word frequencies for only a subset of documents.

        Returns
        -------

        counts : array
            A vector containing the counts of all words in X (columns)

        """
        # Sum on axis 0 (by columns), each column is a word
        # Convert the matrix to an array
        # Squeeze to remove the 1 dimension objects (like ravel)
        return np.squeeze(np.asarray(X.sum(axis=0)))

def attach_to_container(self, container_id):
        """ A socket attached to the stdin/stdout of a container. The object returned contains a get_socket() function to get a socket.socket
        object and  close_socket() to close the connection """
        sock = self._docker.containers.get(container_id).attach_socket(params={
            'stdin': 1,
            'stdout': 1,
            'stderr': 0,
            'stream': 1,
        })
        # fix a problem with docker-py; we must keep a reference of sock at every time
        return FixDockerSocket(sock)

def stop(self):
        """
        Stop this server so that the calling process can exit
        """
        # unsetup_fuse()
        self.fuse_process.teardown()
        for uuid in self.processes:
            self.processes[uuid].terminate()

def flattened_nested_key_indices(nested_dict):
    """
    Combine the outer and inner keys of nested dictionaries into a single
    ordering.
    """
    outer_keys, inner_keys = collect_nested_keys(nested_dict)
    combined_keys = list(sorted(set(outer_keys + inner_keys)))
    return {k: i for (i, k) in enumerate(combined_keys)}

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

def dict_pop_or(d, key, default=None):
    """ Try popping a key from a dict.
        Instead of raising KeyError, just return the default value.
    """
    val = default
    with suppress(KeyError):
        val = d.pop(key)
    return val

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

def unit_ball_L2(shape):
  """A tensorflow variable tranfomed to be constrained in a L2 unit ball.

  EXPERIMENTAL: Do not use for adverserial examples if you need to be confident
  they are strong attacks. We are not yet confident in this code.
  """
  x = tf.Variable(tf.zeros(shape))
  return constrain_L2(x)

def return_value(self, *args, **kwargs):
        """Extracts the real value to be returned from the wrapping callable.

        :return: The value the double should return when called.
        """

        self._called()
        return self._return_value(*args, **kwargs)

def dequeue(self, block=True):
        """Dequeue a record and return item."""
        return self.queue.get(block, self.queue_get_timeout)

def software_fibonacci(n):
    """ a normal old python function to return the Nth fibonacci number. """
    a, b = 0, 1
    for i in range(n):
        a, b = b, a + b
    return a

async def json_or_text(response):
    """Turns response into a properly formatted json or text object"""
    text = await response.text()
    if response.headers['Content-Type'] == 'application/json; charset=utf-8':
        return json.loads(text)
    return text

def fix_datagrepper_message(message):
    """
    See if a message is (probably) a datagrepper message and attempt to mutate
    it to pass signature validation.

    Datagrepper adds the 'source_name' and 'source_version' keys. If messages happen
    to use those keys, they will fail message validation. Additionally, a 'headers'
    dictionary is present on all responses, regardless of whether it was in the
    original message or not. This is deleted if it's null, which won't be correct in
    all cases. Finally, datagrepper turns the 'timestamp' field into a float, but it
    might have been an integer when the message was signed.

    A copy of the dictionary is made and returned if altering the message is necessary.

    I'm so sorry.

    Args:
        message (dict): A message to clean up.

    Returns:
        dict: A copy of the provided message, with the datagrepper-related keys removed
            if they were present.
    """
    if not ('source_name' in message and 'source_version' in message):
        return message

    # Don't mutate the original message
    message = message.copy()

    del message['source_name']
    del message['source_version']
    # datanommer adds the headers field to the message in all cases.
    # This is a huge problem because if the signature was generated with a 'headers'
    # key set and we delete it here, messages will fail validation, but if we don't
    # messages will fail validation if they didn't have a 'headers' key set.
    #
    # There's no way to know whether or not the headers field was part of the signed
    # message or not. Generally, the problem is datanommer is mutating messages.
    if 'headers' in message and not message['headers']:
        del message['headers']
    if 'timestamp' in message:
        message['timestamp'] = int(message['timestamp'])

    return message

def deleteAll(self):
        """
        Deletes whole Solr index. Use with care.
        """
        for core in self.endpoints:
            self._send_solr_command(self.endpoints[core], "{\"delete\": { \"query\" : \"*:*\"}}")

def add_argument(self, dest, nargs=1, obj=None):
        """Adds a positional argument named `dest` to the parser.

        The `obj` can be used to identify the option in the order list
        that is returned from the parser.
        """
        if obj is None:
            obj = dest
        self._args.append(Argument(dest=dest, nargs=nargs, obj=obj))

def days_in_month(year, month):
    """
    returns number of days for the given year and month

    :param int year: calendar year
    :param int month: calendar month
    :return int:
    """

    eom = _days_per_month[month - 1]
    if is_leap_year(year) and month == 2:
        eom += 1

    return eom

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

def is_valid(email):
        """Email address validation method.

        :param email: Email address to be saved.
        :type email: basestring
        :returns: True if email address is correct, False otherwise.
        :rtype: bool
        """
        if isinstance(email, basestring) and EMAIL_RE.match(email):
            return True
        return False

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

def getpackagepath():
    """
     *Get the root path for this python package - used in unit testing code*
    """
    moduleDirectory = os.path.dirname(__file__)
    packagePath = os.path.dirname(__file__) + "/../"

    return packagePath

def reduce_freqs(freqlist):
    """
    Add up a list of freq counts to get the total counts.
    """
    allfreqs = np.zeros_like(freqlist[0])
    for f in freqlist:
        allfreqs += f
    return allfreqs

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

def append_query_parameter(url, parameters, ignore_if_exists=True):
    """ quick and dirty appending of query parameters to a url """
    if ignore_if_exists:
        for key in parameters.keys():
            if key + "=" in url:
                del parameters[key]
    parameters_str = "&".join(k + "=" + v for k, v in parameters.items())
    append_token = "&" if "?" in url else "?"
    return url + append_token + parameters_str

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

def variance(arr):
  """variance of the values, must have 2 or more entries.

  :param arr: list of numbers
  :type arr: number[] a number array
  :return: variance
  :rtype: float

  """
  avg = average(arr)
  return sum([(float(x)-avg)**2 for x in arr])/float(len(arr)-1)

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

def needs_update(self, cache_key):
    """Check if the given cached item is invalid.

    :param cache_key: A CacheKey object (as returned by CacheKeyGenerator.key_for().
    :returns: True if the cached version of the item is out of date.
    """
    if not self.cacheable(cache_key):
      # An uncacheable CacheKey is always out of date.
      return True

    return self._read_sha(cache_key) != cache_key.hash

def OnRootView(self, event):
        """Reset view to the root of the tree"""
        self.adapter, tree, rows = self.RootNode()
        self.squareMap.SetModel(tree, self.adapter)
        self.RecordHistory()
        self.ConfigureViewTypeChoices()

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

def make_2d(ary):
    """Convert any array into a 2d numpy array.

    In case the array is already more than 2 dimensional, will ravel the
    dimensions after the first.
    """
    dim_0, *_ = np.atleast_1d(ary).shape
    return ary.reshape(dim_0, -1, order="F")

def _wrap(text, columns=80):
    """
    Own "dumb" reimplementation of textwrap.wrap().

    This is because calling .wrap() on bigger strings can take a LOT of
    processor power. And I mean like 8 seconds of 3GHz CPU just to wrap 20kB of
    text without spaces.

    Args:
        text (str): Text to wrap.
        columns (int): Wrap after `columns` characters.

    Returns:
        str: Wrapped text.
    """
    out = []
    for cnt, char in enumerate(text):
        out.append(char)

        if (cnt + 1) % columns == 0:
            out.append("\n")

    return "".join(out)

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

def save_hdf(self,filename,path=''):
        """Saves all relevant data to .h5 file; so state can be restored.
        """
        self.dataframe.to_hdf(filename,'{}/df'.format(path))

def __run(self):
    """Hacked run function, which installs the trace."""
    sys.settrace(self.globaltrace)
    self.__run_backup()
    self.run = self.__run_backup

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

def compare(self, dn, attr, value):
        """
        Compare the ``attr`` of the entry ``dn`` with given ``value``.

        This is a convenience wrapper for the ldap library's ``compare``
        function that returns a boolean value instead of 1 or 0.
        """
        return self.connection.compare_s(dn, attr, value) == 1

def home(self):
        """Set cursor to initial position and reset any shifting."""
        self.command(c.LCD_RETURNHOME)
        self._cursor_pos = (0, 0)
        c.msleep(2)

def RoundToSeconds(cls, timestamp):
    """Takes a timestamp value and rounds it to a second precision."""
    leftovers = timestamp % definitions.MICROSECONDS_PER_SECOND
    scrubbed = timestamp - leftovers
    rounded = round(float(leftovers) / definitions.MICROSECONDS_PER_SECOND)

    return int(scrubbed + rounded * definitions.MICROSECONDS_PER_SECOND)

def to_dict(cls):
        """Make dictionary version of enumerated class.

        Dictionary created this way can be used with def_num.

        Returns:
          A dict (name) -> number
        """
        return dict((item.name, item.number) for item in iter(cls))

def extract_vars_above(*names):
    """Extract a set of variables by name from another frame.

    Similar to extractVars(), but with a specified depth of 1, so that names
    are exctracted exactly from above the caller.

    This is simply a convenience function so that the very common case (for us)
    of skipping exactly 1 frame doesn't have to construct a special dict for
    keyword passing."""

    callerNS = sys._getframe(2).f_locals
    return dict((k,callerNS[k]) for k in names)

def requests_request(method, url, **kwargs):
    """Requests-mock requests.request wrapper."""
    session = local_sessions.session
    response = session.request(method=method, url=url, **kwargs)
    session.close()
    return response

def _numpy_char_to_bytes(arr):
    """Like netCDF4.chartostring, but faster and more flexible.
    """
    # based on: http://stackoverflow.com/a/10984878/809705
    arr = np.array(arr, copy=False, order='C')
    dtype = 'S' + str(arr.shape[-1])
    return arr.view(dtype).reshape(arr.shape[:-1])

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

def writer_acquire(self):
        """Acquire the lock to write"""

        self._order_mutex.acquire()
        self._access_mutex.acquire()
        self._order_mutex.release()

def other_ind(self):
        """last row or column of square A"""
        return np.full(self.n_min, self.size - 1, dtype=np.int)

def apply(self, node):
        """ Apply transformation and return if an update happened. """
        new_node = self.run(node)
        return self.update, new_node

def parser():

    """Return a parser for setting one or more configuration paths"""

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_paths', default=[], action='append',
                        help='path to a configuration directory')
    return parser

def column_stack_2d(data):
    """Perform column-stacking on a list of 2d data blocks."""
    return list(list(itt.chain.from_iterable(_)) for _ in zip(*data))

def close_log(log, verbose=True):
    """Close log

    This method closes and active logging.Logger instance.

    Parameters
    ----------
    log : logging.Logger
        Logging instance

    """

    if verbose:
        print('Closing log file:', log.name)

    # Send closing message.
    log.info('The log file has been closed.')

    # Remove all handlers from log.
    [log.removeHandler(handler) for handler in log.handlers]

def _maybe_fill(arr, fill_value=np.nan):
    """
    if we have a compatible fill_value and arr dtype, then fill
    """
    if _isna_compat(arr, fill_value):
        arr.fill(fill_value)
    return arr

def is_numeric(value):
        """Test if a value is numeric.
        """
        return type(value) in [
            int,
            float,
            
            np.int8,
            np.int16,
            np.int32,
            np.int64,

            np.float16,
            np.float32,
            np.float64,
            np.float128
        ]

def DeleteIndex(self, index):
        """
        Remove a spent coin based on its index.

        Args:
            index (int):
        """
        to_remove = None
        for i in self.Items:
            if i.index == index:
                to_remove = i

        if to_remove:
            self.Items.remove(to_remove)

def links(cls, page):
    """return all links on a page, including potentially rel= links."""
    for match in cls.HREF_RE.finditer(page):
      yield cls.href_match_to_url(match)

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

def multipart_parse_json(api_url, data):
    """
    Send a post request and parse the JSON response (potentially containing
    non-ascii characters).
    @param api_url: the url endpoint to post to.
    @param data: a dictionary that will be passed to requests.post
    """
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    response_text = requests.post(api_url, data=data, headers=headers)\
        .text.encode('ascii', errors='replace')

    return json.loads(response_text.decode())

def delete(self, row):
        """Delete a track value"""
        i = self._get_key_index(row)
        del self.keys[i]

def nearest_intersection_idx(a, b):
    """Determine the index of the point just before two lines with common x values.

    Parameters
    ----------
    a : array-like
        1-dimensional array of y-values for line 1
    b : array-like
        1-dimensional array of y-values for line 2

    Returns
    -------
        An array of indexes representing the index of the values
        just before the intersection(s) of the two lines.

    """
    # Difference in the two y-value sets
    difference = a - b

    # Determine the point just before the intersection of the lines
    # Will return multiple points for multiple intersections
    sign_change_idx, = np.nonzero(np.diff(np.sign(difference)))

    return sign_change_idx

def serve_static(request, path, insecure=False, **kwargs):
    """Collect and serve static files.

    This view serves up static files, much like Django's
    :py:func:`~django.views.static.serve` view, with the addition that it
    collects static files first (if enabled). This allows images, fonts, and
    other assets to be served up without first loading a page using the
    ``{% javascript %}`` or ``{% stylesheet %}`` template tags.

    You can use this view by adding the following to any :file:`urls.py`::

        urlpatterns += static('static/', view='pipeline.views.serve_static')
    """
    # Follow the same logic Django uses for determining access to the
    # static-serving view.
    if not django_settings.DEBUG and not insecure:
        raise ImproperlyConfigured("The staticfiles view can only be used in "
                                   "debug mode or if the --insecure "
                                   "option of 'runserver' is used")

    if not settings.PIPELINE_ENABLED and settings.PIPELINE_COLLECTOR_ENABLED:
        # Collect only the requested file, in order to serve the result as
        # fast as possible. This won't interfere with the template tags in any
        # way, as those will still cause Django to collect all media.
        default_collector.collect(request, files=[path])

    return serve(request, path, document_root=django_settings.STATIC_ROOT,
                 **kwargs)

def wait_for_url(url, timeout=DEFAULT_TIMEOUT):
    """
    Return True if connection to the host and port specified in url
    is successful within the timeout.

    @param url: str: connection url for a TCP service
    @param timeout: int: length of time in seconds to try to connect before giving up
    @raise RuntimeError: if no port is given or can't be guessed via the scheme
    @return: bool
    """
    service = ServiceURL(url, timeout)
    return service.wait()

def clean_all(self, args):
        """Delete all build components; the package cache, package builds,
        bootstrap builds and distributions."""
        self.clean_dists(args)
        self.clean_builds(args)
        self.clean_download_cache(args)

def set_title(self, title, **kwargs):
        """Sets the title on the underlying matplotlib AxesSubplot."""
        ax = self.get_axes()
        ax.set_title(title, **kwargs)

def dedup(seq):
    """Remove duplicates from a list while keeping order."""
    seen = set()
    for item in seq:
        if item not in seen:
            seen.add(item)
            yield item

def get_absolute_path(*args):
    """Transform relative pathnames into absolute pathnames."""
    directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(directory, *args)

def __check_success(resp):
        """ Check a JSON server response to see if it was successful

        :type resp: Dictionary (parsed JSON from response)
        :param resp: the response string

        :rtype: String
        :returns: the success message, if it exists

        :raises: APIError if the success message is not present


        """

        if "success" not in resp.keys():
            try:
                raise APIError('200', 'Operation Failed', resp["error"])
            except KeyError:
                raise APIError('200', 'Operation Failed', str(resp))
        return resp["success"]

def hkm_fc(fdata, Nmax, m, s):
    """ Assume fdata has even rows"""

    f = fdata[:, m]
    L1 = f.size
    MM = int(L1 / 2)
    Q = s.size

    ff = np.zeros(Q, dtype=np.complex128)
    for n in xrange(MM, L1):
        ff[n] = f[n - MM]

    for n in xrange(0, MM):
        ff[n] = f[n + MM]

    # For larger problems, this speeds things up pretty good.
    F = np.fft.fft(ff)
    S = np.fft.fft(s)
    out = 4 * np.pi * np.fft.ifft(F * S)

    return out[0:Nmax + 1]

def to_camel(s):
    """
    :param string s: under_scored string to be CamelCased
    :return: CamelCase version of input
    :rtype: str
    """
    # r'(?!^)_([a-zA-Z]) original regex wasn't process first groups
    return re.sub(r'_([a-zA-Z])', lambda m: m.group(1).upper(), '_' + s)

def get_table_metadata(engine, table):
    """ Extract all useful infos from the given table

    Args:
        engine: SQLAlchemy connection engine
        table: table name

    Returns:
        Dictionary of infos
    """
    metadata = MetaData()
    metadata.reflect(bind=engine, only=[table])
    table_metadata = Table(table, metadata, autoload=True)
    return table_metadata

def get_winfunc(libname, funcname, restype=None, argtypes=(), _libcache={}):
    """Retrieve a function from a library/DLL, and set the data types."""
    if libname not in _libcache:
        _libcache[libname] = windll.LoadLibrary(libname)
    func = getattr(_libcache[libname], funcname)
    func.argtypes = argtypes
    func.restype = restype
    return func

def version():
    """Wrapper for opj_version library routine."""
    OPENJPEG.opj_version.restype = ctypes.c_char_p
    library_version = OPENJPEG.opj_version()
    if sys.hexversion >= 0x03000000:
        return library_version.decode('utf-8')
    else:
        return library_version

def quaternion_imag(quaternion):
    """Return imaginary part of quaternion.

    >>> quaternion_imag([3, 0, 1, 2])
    array([ 0.,  1.,  2.])

    """
    return numpy.array(quaternion[1:4], dtype=numpy.float64, copy=True)

def _add_line_segment(self, x, y):
        """Add a |_LineSegment| operation to the drawing sequence."""
        self._drawing_operations.append(_LineSegment.new(self, x, y))

def _cnx_is_empty(in_file):
    """Check if cnr or cns files are empty (only have a header)
    """
    with open(in_file) as in_handle:
        for i, line in enumerate(in_handle):
            if i > 0:
                return False
    return True

def legend_title_header_element(feature, parent):
    """Retrieve legend title header string from definitions."""
    _ = feature, parent  # NOQA
    header = legend_title_header['string_format']
    return header.capitalize()

def convert_bytes_to_ints(in_bytes, num):
    """Convert a byte array into an integer array. The number of bytes forming an integer
    is defined by num

    :param in_bytes: the input bytes
    :param num: the number of bytes per int
    :return the integer array"""
    dt = numpy.dtype('>i' + str(num))
    return numpy.frombuffer(in_bytes, dt)

def pstd(self, *args, **kwargs):
        """ Console to STDOUT """
        kwargs['file'] = self.out
        self.print(*args, **kwargs)
        sys.stdout.flush()

def to_json(obj):
    """Return a json string representing the python object obj."""
    i = StringIO.StringIO()
    w = Writer(i, encoding='UTF-8')
    w.write_value(obj)
    return i.getvalue()

def get_table_names(connection):
	"""
	Return a list of the table names in the database.
	"""
	cursor = connection.cursor()
	cursor.execute("SELECT name FROM sqlite_master WHERE type == 'table'")
	return [name for (name,) in cursor]

def isfile_notempty(inputfile: str) -> bool:
        """Check if the input filename with path is a file and is not empty."""
        try:
            return isfile(inputfile) and getsize(inputfile) > 0
        except TypeError:
            raise TypeError('inputfile is not a valid type')

def HttpResponse403(request, template=KEY_AUTH_403_TEMPLATE,
content=KEY_AUTH_403_CONTENT, content_type=KEY_AUTH_403_CONTENT_TYPE):
    """
    HTTP response for forbidden access (status code 403)
    """
    return AccessFailedResponse(request, template, content, content_type, status=403)

def correspond(text):
    """Communicate with the child process without closing stdin."""
    subproc.stdin.write(text)
    subproc.stdin.flush()
    return drain()

def check_permission_safety(path):
    """Check if the file at the given path is safe to use as a state file.

    This checks that group and others have no permissions on the file and that the current user is
    the owner.
    """
    f_stats = os.stat(path)
    return (f_stats.st_mode & (stat.S_IRWXG | stat.S_IRWXO)) == 0 and f_stats.st_uid == os.getuid()

def build_code(self, lang, body):
        """Wrap text with markdown specific flavour."""
        self.out.append("```" + lang)
        self.build_markdown(lang, body)
        self.out.append("```")

def load_feature(fname, language):
    """ Load and parse a feature file. """

    fname = os.path.abspath(fname)
    feat = parse_file(fname, language)
    return feat

def default_number_converter(number_str):
    """
    Converts the string representation of a json number into its python object equivalent, an
    int, long, float or whatever type suits.
    """
    is_int = (number_str.startswith('-') and number_str[1:].isdigit()) or number_str.isdigit()
    # FIXME: this handles a wider range of numbers than allowed by the json standard,
    # etc.: float('nan') and float('inf'). But is this a problem?
    return int(number_str) if is_int else float(number_str)

def get_time():
    """Get time from a locally running NTP server"""

    time_request = '\x1b' + 47 * '\0'
    now = struct.unpack("!12I", ntp_service.request(time_request, timeout=5.0).data.read())[10]
    return time.ctime(now - EPOCH_START)

def convert_from_missing_indexer_tuple(indexer, axes):
    """
    create a filtered indexer that doesn't have any missing indexers
    """

    def get_indexer(_i, _idx):
        return (axes[_i].get_loc(_idx['key']) if isinstance(_idx, dict) else
                _idx)

    return tuple(get_indexer(_i, _idx) for _i, _idx in enumerate(indexer))

def check_cv(self, y):
        """Resolve which cross validation strategy is used."""
        y_arr = None
        if self.stratified:
            # Try to convert y to numpy for sklearn's check_cv; if conversion
            # doesn't work, still try.
            try:
                y_arr = to_numpy(y)
            except (AttributeError, TypeError):
                y_arr = y

        if self._is_float(self.cv):
            return self._check_cv_float()
        return self._check_cv_non_float(y_arr)

def old_pad(s):
    """
    Pads an input string to a given block size.
    :param s: string
    :returns: The padded string.
    """
    if len(s) % OLD_BLOCK_SIZE == 0:
        return s

    return Padding.appendPadding(s, blocksize=OLD_BLOCK_SIZE)

def _monitor_callback_wrapper(callback):
    """A wrapper for the user-defined handle."""
    def callback_handle(name, array, _):
        """ ctypes function """
        callback(name, array)
    return callback_handle

def add_noise(Y, sigma):
    """Adds noise to Y"""
    return Y + np.random.normal(0, sigma, Y.shape)

def pairwise(iterable):
    """From itertools cookbook. [a, b, c, ...] -> (a, b), (b, c), ..."""
    first, second = tee(iterable)
    next(second, None)
    return zip(first, second)

def es_field_sort(fld_name):
    """ Used with lambda to sort fields """
    parts = fld_name.split(".")
    if "_" not in parts[-1]:
        parts[-1] = "_" + parts[-1]
    return ".".join(parts)

def chmod_plus_w(path):
  """Equivalent of unix `chmod +w path`"""
  path_mode = os.stat(path).st_mode
  path_mode &= int('777', 8)
  path_mode |= stat.S_IWRITE
  os.chmod(path, path_mode)

def timedcall(executable_function, *args):
    """!
    @brief Executes specified method or function with measuring of execution time.
    
    @param[in] executable_function (pointer): Pointer to function or method.
    @param[in] args (*): Arguments of called function or method.
    
    @return (tuple) Execution time and result of execution of function or method (execution_time, result_execution).
    
    """
    
    time_start = time.clock();
    result = executable_function(*args);
    time_end = time.clock();
    
    return (time_end - time_start, result);

def library(func):
    """
    A decorator for providing a unittest with a library and have it called only
    once.
    """
    @wraps(func)
    def wrapped(*args, **kwargs):
        """Transparent wrapper."""
        return func(*args, **kwargs)
    SINGLES.append(wrapped)
    return wrapped

def _float_almost_equal(float1, float2, places=7):
    """Return True if two numbers are equal up to the
    specified number of "places" after the decimal point.
    """

    if round(abs(float2 - float1), places) == 0:
        return True

    return False

def to_monthly(series, method='ffill', how='end'):
    """
    Convenience method that wraps asfreq_actual
    with 'M' param (method='ffill', how='end').
    """
    return series.asfreq_actual('M', method=method, how=how)

def get_seconds_until_next_day(now=None):
    """
    Returns the number of seconds until the next day (utc midnight). This is the long-term rate limit used by Strava.
    :param now: A (utc) timestamp
    :type now: arrow.arrow.Arrow
    :return: the number of seconds until next day, as int
    """
    if now is None:
        now = arrow.utcnow()
    return (now.ceil('day') - now).seconds

def unproject(self, xy):
        """
        Returns the coordinates from position in meters
        """
        (x, y) = xy
        lng = x/EARTH_RADIUS * RAD_TO_DEG
        lat = 2 * atan(exp(y/EARTH_RADIUS)) - pi/2 * RAD_TO_DEG
        return (lng, lat)

def stdev(self):
        """ -> #float :func:numpy.std of the timing intervals """
        return round(np.std(self.array), self.precision)\
            if len(self.array) else None

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

def get_plugin_icon(self):
        """Return widget icon"""
        path = osp.join(self.PLUGIN_PATH, self.IMG_PATH)
        return ima.icon('pylint', icon_path=path)

def get_shared_memory_bytes():
    """Get the size of the shared memory file system.

    Returns:
        The size of the shared memory file system in bytes.
    """
    # Make sure this is only called on Linux.
    assert sys.platform == "linux" or sys.platform == "linux2"

    shm_fd = os.open("/dev/shm", os.O_RDONLY)
    try:
        shm_fs_stats = os.fstatvfs(shm_fd)
        # The value shm_fs_stats.f_bsize is the block size and the
        # value shm_fs_stats.f_bavail is the number of available
        # blocks.
        shm_avail = shm_fs_stats.f_bsize * shm_fs_stats.f_bavail
    finally:
        os.close(shm_fd)

    return shm_avail

def clear(self):
        """Remove all nodes and edges from the graph.

        Unlike the regular networkx implementation, this does *not*
        remove the graph's name. But all the other graph, node, and
        edge attributes go away.

        """
        self.adj.clear()
        self.node.clear()
        self.graph.clear()

def is_enum_type(type_):
    """ Checks if the given type is an enum type.

    :param type_: The type to check
    :return: True if the type is a enum type, otherwise False
    :rtype: bool
    """

    return isinstance(type_, type) and issubclass(type_, tuple(_get_types(Types.ENUM)))

def is_lazy_iterable(obj):
    """
    Returns whether *obj* is iterable lazily, such as generators, range objects, etc.
    """
    return isinstance(obj,
        (types.GeneratorType, collections.MappingView, six.moves.range, enumerate))

def _crop_list_to_size(l, size):
    """Make a list a certain size"""
    for x in range(size - len(l)):
        l.append(False)
    for x in range(len(l) - size):
        l.pop()
    return l

def round_array(array_in):
    """
    arr_out = round_array(array_in)

    Rounds an array and recasts it to int. Also works on scalars.
    """
    if isinstance(array_in, ndarray):
        return np.round(array_in).astype(int)
    else:
        return int(np.round(array_in))

def to_string(s, encoding='utf-8'):
    """
    Accept unicode(py2) or bytes(py3)

    Returns:
        py2 type: str
        py3 type: str
    """
    if six.PY2:
        return s.encode(encoding)
    if isinstance(s, bytes):
        return s.decode(encoding)
    return s

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

def get(self, key):  
        """ get a set of keys from redis """
        res = self.connection.get(key)
        print(res)
        return res

def _raise_if_wrong_file_signature(stream):
    """ Reads the 4 first bytes of the stream to check that is LASF"""
    file_sig = stream.read(len(headers.LAS_FILE_SIGNATURE))
    if file_sig != headers.LAS_FILE_SIGNATURE:
        raise errors.PylasError(
            "File Signature ({}) is not {}".format(file_sig, headers.LAS_FILE_SIGNATURE)
        )

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

def roundClosestValid(val, res, decimals=None):
        """ round to closest resolution """
        if decimals is None and "." in str(res):
            decimals = len(str(res).split('.')[1])

        return round(round(val / res) * res, decimals)

def objectcount(data, key):
    """return the count of objects of key"""
    objkey = key.upper()
    return len(data.dt[objkey])

def tag_to_dict(html):
    """Extract tag's attributes into a `dict`."""

    element = document_fromstring(html).xpath("//html/body/child::*")[0]
    attributes = dict(element.attrib)
    attributes["text"] = element.text_content()
    return attributes

def ci(a, which=95, axis=None):
    """Return a percentile range from an array of values."""
    p = 50 - which / 2, 50 + which / 2
    return percentiles(a, p, axis)

def _format_json(data, theme):
    """Pretty print a dict as a JSON, with colors if pygments is present."""
    output = json.dumps(data, indent=2, sort_keys=True)

    if pygments and sys.stdout.isatty():
        style = get_style_by_name(theme)
        formatter = Terminal256Formatter(style=style)
        return pygments.highlight(output, JsonLexer(), formatter)

    return output

def value(self):
        """Value of property."""
        if self._prop.fget is None:
            raise AttributeError('Unable to read attribute')
        return self._prop.fget(self._obj)

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

def revnet_164_cifar():
  """Tiny hparams suitable for CIFAR/etc."""
  hparams = revnet_cifar_base()
  hparams.bottleneck = True
  hparams.num_channels = [16, 32, 64]
  hparams.num_layers_per_block = [8, 8, 8]
  return hparams

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

def _is_numeric(self, values):
        """Check to be sure values are numbers before doing numerical operations."""
        if len(values) > 0:
            assert isinstance(values[0], (float, int)), \
                "values must be numbers to perform math operations. Got {}".format(
                    type(values[0]))
        return True

def console_get_background_flag(con: tcod.console.Console) -> int:
    """Return this consoles current blend mode.

    Args:
        con (Console): Any Console instance.

    .. deprecated:: 8.5
        Check :any:`Console.default_bg_blend` instead.
    """
    return int(lib.TCOD_console_get_background_flag(_console(con)))

def render_template_string(source, **context):
    """Renders a template from the given template source string
    with the given context.

    :param source: the sourcecode of the template to be
                   rendered
    :param context: the variables that should be available in the
                    context of the template.
    """
    ctx = _app_ctx_stack.top
    ctx.app.update_template_context(context)
    return _render(ctx.app.jinja_env.from_string(source),
                   context, ctx.app)

def on_error(e):  # pragma: no cover
    """Error handler

    RuntimeError or ValueError exceptions raised by commands will be handled
    by this function.
    """
    exname = {'RuntimeError': 'Runtime error', 'Value Error': 'Value error'}
    sys.stderr.write('{}: {}\n'.format(exname[e.__class__.__name__], str(e)))
    sys.stderr.write('See file slam_error.log for additional details.\n')
    sys.exit(1)

def min_depth(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    if root is None:
        return 0
    if root.left is not None or root.right is not None:
        return max(self.minDepth(root.left), self.minDepth(root.right))+1
    return min(self.minDepth(root.left), self.minDepth(root.right)) + 1

def log(x):
    """
    Natural logarithm
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.log(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.log(x)

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

def remove(self, key):
        """remove the value found at key from the queue"""
        item = self.item_finder.pop(key)
        item[-1] = None
        self.removed_count += 1

def load_streams(chunks):
    """
    Given a gzipped stream of data, yield streams of decompressed data.
    """
    chunks = peekable(chunks)
    while chunks:
        if six.PY3:
            dc = zlib.decompressobj(wbits=zlib.MAX_WBITS | 16)
        else:
            dc = zlib.decompressobj(zlib.MAX_WBITS | 16)
        yield load_stream(dc, chunks)
        if dc.unused_data:
            chunks = peekable(itertools.chain((dc.unused_data,), chunks))

def zero_state(self, batch_size):
        """ Initial state of the network """
        return torch.zeros(batch_size, self.state_dim, dtype=torch.float32)

def makedirs(path):
    """
    Create directories if they do not exist, otherwise do nothing.

    Return path for convenience
    """
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def indent(txt, spacing=4):
    """
    Indent given text using custom spacing, default is set to 4.
    """
    return prefix(str(txt), ''.join([' ' for _ in range(spacing)]))

def schemaValidateFile(self, filename, options):
        """Do a schemas validation of the given resource, it will use
           the SAX streamable validation internally. """
        ret = libxml2mod.xmlSchemaValidateFile(self._o, filename, options)
        return ret

def show_xticklabels(self, row, column):
        """Show the x-axis tick labels for a subplot.

        :param row,column: specify the subplot.

        """
        subplot = self.get_subplot_at(row, column)
        subplot.show_xticklabels()

def convert_ajax_data(self, field_data):
        """
        Due to the way Angular organizes it model, when this Form data is sent using Ajax,
        then for this kind of widget, the sent data has to be converted into a format suitable
        for Django's Form validation.
        """
        data = [key for key, val in field_data.items() if val]
        return data

def on_binop(self, node):    # ('left', 'op', 'right')
        """Binary operator."""
        return op2func(node.op)(self.run(node.left),
                                self.run(node.right))

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

def get_files(client, bucket, prefix=''):
    """Lists files/objects on a bucket.
    
    TODO: docstring"""
    bucket = client.get_bucket(bucket)
    files = list(bucket.list_blobs(prefix=prefix))    
    return files

def normalize_field(self, value):
        """
        Method that must transform the value from string
        Ex: if the expected type is int, it should return int(self._attr)

        """
        if self.default is not None:
            if value is None or value == '':
                value = self.default
        return value

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

def to_pascal_case(s):
    """Transform underscore separated string to pascal case

    """
    return re.sub(r'(?!^)_([a-zA-Z])', lambda m: m.group(1).upper(), s.capitalize())

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

def guess_title(basename):
    """ Attempt to guess the title from the filename """

    base, _ = os.path.splitext(basename)
    return re.sub(r'[ _-]+', r' ', base).title()

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

def str2bool(value):
    """Parse Yes/No/Default string
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    if value.lower() in ('d', 'default', ''):
        return None
    raise argparse.ArgumentTypeError('Expected: (Y)es/(T)rue/(N)o/(F)alse/(D)efault')

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

def _newer(a, b):
    """Inquire whether file a was written since file b."""
    if not os.path.exists(a):
        return False
    if not os.path.exists(b):
        return True
    return os.path.getmtime(a) >= os.path.getmtime(b)

def _internal_kv_get(key):
    """Fetch the value of a binary key."""

    worker = ray.worker.get_global_worker()
    if worker.mode == ray.worker.LOCAL_MODE:
        return _local.get(key)

    return worker.redis_client.hget(key, "value")

def focusInEvent(self, event):
        """Reimplement Qt method to send focus change notification"""
        self.focus_changed.emit()
        return super(PageControlWidget, self).focusInEvent(event)

def load_graph_from_rdf(fname):
    """ reads an RDF file into a graph """
    print("reading RDF from " + fname + "....")
    store = Graph()
    store.parse(fname, format="n3")
    print("Loaded " + str(len(store)) + " tuples")
    return store

def projR(gamma, p):
    """return the KL projection on the row constrints """
    return np.multiply(gamma.T, p / np.maximum(np.sum(gamma, axis=1), 1e-10)).T

def dir_modtime(dpath):
    """
    Returns the latest modification time of all files/subdirectories in a
    directory
    """
    return max(os.path.getmtime(d) for d, _, _ in os.walk(dpath))

def hmean_int(a, a_min=5778, a_max=1149851):
    """ Harmonic mean of an array, returns the closest int
    """
    from scipy.stats import hmean
    return int(round(hmean(np.clip(a, a_min, a_max))))

def set_terminate_listeners(stream):
    """Die on SIGTERM or SIGINT"""

    def stop(signum, frame):
        terminate(stream.listener)

    # Installs signal handlers for handling SIGINT and SIGTERM
    # gracefully.
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

def inc_date(date_obj, num, date_fmt):
    """Increment the date by a certain number and return date object.
    as the specific string format.
    """
    return (date_obj + timedelta(days=num)).strftime(date_fmt)

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

def encode_ndarray(obj):
    """Write a numpy array and its shape to base64 buffers"""
    shape = obj.shape
    if len(shape) == 1:
        shape = (1, obj.shape[0])
    if obj.flags.c_contiguous:
        obj = obj.T
    elif not obj.flags.f_contiguous:
        obj = asfortranarray(obj.T)
    else:
        obj = obj.T
    try:
        data = obj.astype(float64).tobytes()
    except AttributeError:
        data = obj.astype(float64).tostring()

    data = base64.b64encode(data).decode('utf-8')
    return data, shape

def to_0d_array(value: Any) -> np.ndarray:
    """Given a value, wrap it in a 0-D numpy.ndarray.
    """
    if np.isscalar(value) or (isinstance(value, np.ndarray) and
                              value.ndim == 0):
        return np.array(value)
    else:
        return to_0d_object_array(value)

def intersect(d1, d2):
    """Intersect dictionaries d1 and d2 by key *and* value."""
    return dict((k, d1[k]) for k in d1 if k in d2 and d1[k] == d2[k])

def _load_autoreload_magic(self):
        """Load %autoreload magic."""
        from IPython.core.getipython import get_ipython
        try:
            get_ipython().run_line_magic('reload_ext', 'autoreload')
            get_ipython().run_line_magic('autoreload', '2')
        except Exception:
            pass

def is_webdriver_ios(webdriver):
        """
        Check if a web driver if mobile.

        Args:
            webdriver (WebDriver): Selenium webdriver.

        """
        browser = webdriver.capabilities['browserName']

        if (browser == u('iPhone') or 
            browser == u('iPad')):
            return True
        else:
            return False

def connect(*args, **kwargs):
    """Creates or returns a singleton :class:`.Connection` object"""
    global __CONNECTION
    if __CONNECTION is None:
        __CONNECTION = Connection(*args, **kwargs)

    return __CONNECTION

def _bind_parameter(self, parameter, value):
        """Assigns a parameter value to matching instructions in-place."""
        for (instr, param_index) in self._parameter_table[parameter]:
            instr.params[param_index] = value

def compute(args):
    x, y, params = args
    """Callable function for the multiprocessing pool."""
    return x, y, mandelbrot(x, y, params)

def transform(self, df):
        """
        Transforms a DataFrame in place. Computes all outputs of the DataFrame.

        Args:
            df (pandas.DataFrame): DataFrame to transform.
        """
        for name, function in self.outputs:
            df[name] = function(df)

def spanning_tree_count(graph: nx.Graph) -> int:
    """Return the number of unique spanning trees of a graph, using
    Kirchhoff's matrix tree theorem.
    """
    laplacian = nx.laplacian_matrix(graph).toarray()
    comatrix = laplacian[:-1, :-1]
    det = np.linalg.det(comatrix)
    count = int(round(det))
    return count

def uniqueID(size=6, chars=string.ascii_uppercase + string.digits):
    """A quick and dirty way to get a unique string"""
    return ''.join(random.choice(chars) for x in xrange(size))

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

def sys_pipes_forever(encoding=_default_encoding):
    """Redirect all C output to sys.stdout/err
    
    This is not a context manager; it turns on C-forwarding permanently.
    """
    global _mighty_wurlitzer
    if _mighty_wurlitzer is None:
        _mighty_wurlitzer = sys_pipes(encoding)
    _mighty_wurlitzer.__enter__()

def __iter__(self):
        """
        Iterate through tree, leaves first

        following http://stackoverflow.com/questions/6914803/python-iterator-through-tree-with-list-of-children
        """
        for node in chain(*imap(iter, self.children)):
            yield node
        yield self

def git_tag(tag):
    """Tags the current version."""
    print('Tagging "{}"'.format(tag))
    msg = '"Released version {}"'.format(tag)
    Popen(['git', 'tag', '-s', '-m', msg, tag]).wait()

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

def move(self, x, y):
        """Move the virtual cursor.

        Args:
            x (int): x-coordinate to place the cursor.
            y (int): y-coordinate to place the cursor.

        .. seealso:: :any:`get_cursor`, :any:`print_str`, :any:`write`
        """
        self._cursor = self._normalizePoint(x, y)

def Gaussian(x, mu, sig):
    """
    Gaussian pdf.
    :param x: free variable.
    :param mu: mean of the distribution.
    :param sig: standard deviation of the distribution.
    :return: sympy.Expr for a Gaussian pdf.
    """
    return sympy.exp(-(x - mu)**2/(2*sig**2))/sympy.sqrt(2*sympy.pi*sig**2)

def urljoin(*urls):
    """
    The default urlparse.urljoin behavior look strange
    Standard urlparse.urljoin('http://a.com/foo', '/bar')
    Expect: http://a.com/foo/bar
    Actually: http://a.com/bar

    This function fix that.
    """
    return reduce(urlparse.urljoin, [u.strip('/')+'/' for u in urls if u.strip('/')], '').rstrip('/')

def get_raw_input(description, default=False):
    """Get user input from the command line via raw_input / input.

    description (unicode): Text to display before prompt.
    default (unicode or False/None): Default value to display with prompt.
    RETURNS (unicode): User input.
    """
    additional = ' (default: %s)' % default if default else ''
    prompt = '    %s%s: ' % (description, additional)
    user_input = input_(prompt)
    return user_input

def _genTex2D(self):
        """Generate an empty texture in OpenGL"""
        for face in range(6):
            gl.glTexImage2D(self.target0 + face, 0, self.internal_fmt, self.width, self.height, 0,
                            self.pixel_fmt, gl.GL_UNSIGNED_BYTE, 0)

def size(self):
        """
        Recursively find size of a tree. Slow.
        """

        if self is NULL:
            return 0
        return 1 + self.left.size() + self.right.size()

def add_input_variable(self, var):
        """Adds the argument variable as one of the input variable"""
        assert(isinstance(var, Variable))
        self.input_variable_list.append(var)

def binary_stdout():
    """
    A sys.stdout that accepts bytes.
    """

    # First there is a Python3 issue.
    try:
        stdout = sys.stdout.buffer
    except AttributeError:
        # Probably Python 2, where bytes are strings.
        stdout = sys.stdout

    # On Windows the C runtime file orientation needs changing.
    if sys.platform == "win32":
        import msvcrt
        import os
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)

    return stdout

def is_executable(path):
  """Returns whether a path names an existing executable file."""
  return os.path.isfile(path) and os.access(path, os.X_OK)

def _send(self, data):
        """Send data to statsd."""
        if not self._sock:
            self.connect()
        self._do_send(data)

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

def print_trace(self):
        """
        Prints stack trace for current exceptions chain.
        """
        traceback.print_exc()
        for tb in self.tracebacks:
            print tb,
        print ''

def apply_fit(xy,coeffs):
    """ Apply the coefficients from a linear fit to
        an array of x,y positions.

        The coeffs come from the 'coeffs' member of the
        'fit_arrays()' output.
    """
    x_new = coeffs[0][2] + coeffs[0][0]*xy[:,0] + coeffs[0][1]*xy[:,1]
    y_new = coeffs[1][2] + coeffs[1][0]*xy[:,0] + coeffs[1][1]*xy[:,1]

    return x_new,y_new

def mul(a, b):
  """
  A wrapper around tf multiplication that does more automatic casting of
  the input.
  """
  def multiply(a, b):
    """Multiplication"""
    return a * b
  return op_with_scalar_cast(a, b, multiply)

def _get(self, pos):
        """loads widget at given position; handling invalid arguments"""
        res = None, None
        if pos is not None:
            try:
                res = self[pos], pos
            except (IndexError, KeyError):
                pass
        return res

def ensure_index(self, key, unique=False):
        """Wrapper for pymongo.Collection.ensure_index
        """
        return self.collection.ensure_index(key, unique=unique)

def check_output(args):
    """Runs command and returns the output as string."""
    log.debug('run: %s', args)
    out = subprocess.check_output(args=args).decode('utf-8')
    log.debug('out: %r', out)
    return out

def EvalBinomialPmf(k, n, p):
    """Evaluates the binomial pmf.

    Returns the probabily of k successes in n trials with probability p.
    """
    return scipy.stats.binom.pmf(k, n, p)

def filter_bool(n: Node, query: str) -> bool:
    """
    Filter and ensure that the returned value is of type bool.
    """
    return _scalariter2item(n, query, bool)

def rest_put_stream(self, url, stream, headers=None, session=None, verify=True, cert=None):
        """
        Perform a chunked PUT request to url with requests.session
        This is specifically to upload files.
        """
        res = session.put(url, headers=headers, data=stream, verify=verify, cert=cert)
        return res.text, res.status_code

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

def _sim_fill(r1, r2, imsize):
    """
        calculate the fill similarity over the image
    """
    bbsize = (
        (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]))
        * (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
    )
    return 1.0 - (bbsize - r1["size"] - r2["size"]) / imsize

def remove_duplicates(lst):
    """
    Emulate what a Python ``set()`` does, but keeping the element's order.
    """
    dset = set()
    return [l for l in lst if l not in dset and not dset.add(l)]

def create_db_schema(cls, cur, schema_name):
        """
        Create Postgres schema script and execute it on cursor
        """
        create_schema_script = "CREATE SCHEMA {0} ;\n".format(schema_name)
        cur.execute(create_schema_script)

def fileopenbox(msg=None, title=None, argInitialFile=None):
    """Original doc: A dialog to get a file name.
        Returns the name of a file, or None if user chose to cancel.

        if argInitialFile contains a valid filename, the dialog will
        be positioned at that file when it appears.
        """
    return psidialogs.ask_file(message=msg, title=title, default=argInitialFile)

def sg_init(sess):
    r""" Initializes session variables.
    
    Args:
      sess: Session to initialize. 
    """
    # initialize variables
    sess.run(tf.group(tf.global_variables_initializer(),
                      tf.local_variables_initializer()))

def pop(self):
        """
        return the last stack element and delete it from the list
        """
        if not self.empty():
            val = self.stack[-1]
            del self.stack[-1]
            return val

def OnPasteAs(self, event):
        """Clipboard paste as event handler"""

        data = self.main_window.clipboard.get_clipboard()
        key = self.main_window.grid.actions.cursor

        with undo.group(_("Paste As...")):
            self.main_window.actions.paste_as(key, data)

        self.main_window.grid.ForceRefresh()

        event.Skip()

def parse(self, data, mimetype):
        """
        Parses a byte array containing a JSON document and returns a Python object.
        :param data: The byte array containing a JSON document.
        :param MimeType mimetype: The mimetype chose to parse the data.
        :return: A Python object.
        """
        encoding = mimetype.params.get('charset') or 'utf-8'

        return json.loads(data.decode(encoding))

def _is_valid_api_url(self, url):
        """Callback for is_valid_api_url."""
        # Check response is a JSON with ok: 1
        data = {}
        try:
            r = requests.get(url, proxies=self.proxy_servers)
            content = to_text_string(r.content, encoding='utf-8')
            data = json.loads(content)
        except Exception as error:
            logger.error(str(error))

        return data.get('ok', 0) == 1

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

def extract(self):
        """
        Creates a copy of this tabarray in the form of a numpy ndarray.

        Useful if you want to do math on array elements, e.g. if you have a 
        subset of the columns that are all numerical, you can construct a 
        numerical matrix and do matrix operations.

        """
        return np.vstack([self[r] for r in self.dtype.names]).T.squeeze()

def _linear_interpolation(x, X, Y):
    """Given two data points [X,Y], linearly interpolate those at x.
    """
    return (Y[1] * (x - X[0]) + Y[0] * (X[1] - x)) / (X[1] - X[0])

def lin_interp(x, rangeX, rangeY):
    """
    Interpolate linearly variable x in rangeX onto rangeY.
    """
    s = (x - rangeX[0]) / mag(rangeX[1] - rangeX[0])
    y = rangeY[0] * (1 - s) + rangeY[1] * s
    return y

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

def find_whole_word(w):
    """
    Scan through string looking for a location where this word produces a match,
    and return a corresponding MatchObject instance.
    Return None if no position in the string matches the pattern;
    note that this is different from finding a zero-length match at some point in the string.
    """
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

def kill_test_logger(logger):
    """Cleans up a test logger object by removing all of its handlers.

    Args:
        logger: The logging object to clean up.
    """
    for h in list(logger.handlers):
        logger.removeHandler(h)
        if isinstance(h, logging.FileHandler):
            h.close()

def get_tablenames(cur):
    """ Conveinience: """
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tablename_list_ = cur.fetchall()
    tablename_list = [str(tablename[0]) for tablename in tablename_list_ ]
    return tablename_list

def debug_src(src, pm=False, globs=None):
    """Debug a single doctest docstring, in argument `src`'"""
    testsrc = script_from_examples(src)
    debug_script(testsrc, pm, globs)

def _on_text_changed(self):
        """ Adjust dirty flag depending on editor's content """
        if not self._cleaning:
            ln = TextHelper(self).cursor_position()[0]
            self._modified_lines.add(ln)

def safe_delete(filename):
  """Delete a file safely. If it's not present, no-op."""
  try:
    os.unlink(filename)
  except OSError as e:
    if e.errno != errno.ENOENT:
      raise

def see_doc(obj_with_doc):
    """Copy docstring from existing object to the decorated callable."""
    def decorator(fn):
        fn.__doc__ = obj_with_doc.__doc__
        return fn
    return decorator

def _unzip_handle(handle):
    """Transparently unzip the file handle"""
    if isinstance(handle, basestring):
        handle = _gzip_open_filename(handle)
    else:
        handle = _gzip_open_handle(handle)
    return handle

def _loop_timeout_cb(self, main_loop):
        """Stops the loop after the time specified in the `loop` call.
        """
        self._anything_done = True
        logger.debug("_loop_timeout_cb() called")
        main_loop.quit()

def _get_info(self, host, port, unix_socket, auth):
        """Return info dict from specified Redis instance

:param str host: redis host
:param int port: redis port
:rtype: dict

        """

        client = self._client(host, port, unix_socket, auth)
        if client is None:
            return None

        info = client.info()
        del client
        return info

def coords_string_parser(self, coords):
        """Pareses the address string into coordinates to match address_to_coords return object"""

        lat, lon = coords.split(',')
        return {"lat": lat.strip(), "lon": lon.strip(), "bounds": {}}

def create_ellipse(width,height,angle):
    """Create parametric ellipse from 200 points."""
    angle = angle / 180.0 * np.pi
    thetas = np.linspace(0,2*np.pi,200)
    a = width / 2.0
    b = height / 2.0

    x = a*np.cos(thetas)*np.cos(angle) - b*np.sin(thetas)*np.sin(angle)
    y = a*np.cos(thetas)*np.sin(angle) + b*np.sin(thetas)*np.cos(angle)
    z = np.zeros(thetas.shape)
    return np.vstack((x,y,z)).T

def replace(self, text):
        """Do j/v replacement"""
        for (pattern, repl) in self.patterns:
            text = re.subn(pattern, repl, text)[0]
        return text

def _replace_token_range(tokens, start, end, replacement):
    """For a range indicated from start to end, replace with replacement."""
    tokens = tokens[:start] + replacement + tokens[end:]
    return tokens

def save_notebook(work_notebook, write_file):
    """Saves the Jupyter work_notebook to write_file"""
    with open(write_file, 'w') as out_nb:
        json.dump(work_notebook, out_nb, indent=2)

def flatten_list(l):
    """ Nested lists to single-level list, does not split strings"""
    return list(chain.from_iterable(repeat(x,1) if isinstance(x,str) else x for x in l))

def symbols():
    """Return a list of symbols."""
    symbols = []
    for line in symbols_stream():
        symbols.append(line.decode('utf-8').strip())
    return symbols

def fromiterable(cls, itr):
        """Initialize from iterable"""
        x, y, z = itr
        return cls(x, y, z)

def random_id(length):
    """Generates a random ID of given length"""

    def char():
        """Generate single random char"""

        return random.choice(string.ascii_letters + string.digits)

    return "".join(char() for _ in range(length))

def to_comment(value):
  """
  Builds a comment.
  """
  if value is None:
    return
  if len(value.split('\n')) == 1:
    return "* " + value
  else:
    return '\n'.join([' * ' + l for l in value.split('\n')[:-1]])

def generate_random_string(chars=7):
    """

    :param chars:
    :return:
    """
    return u"".join(random.sample(string.ascii_letters * 2 + string.digits, chars))

def main(ctx, connection):
    """Command line interface for PyBEL."""
    ctx.obj = Manager(connection=connection)
    ctx.obj.bind()

def short_description(func):
    """
    Given an object with a docstring, return the first line of the docstring
    """

    doc = inspect.getdoc(func)
    if doc is not None:
        doc = inspect.cleandoc(doc)
        lines = doc.splitlines()
        return lines[0]

    return ""

def refresh_swagger(self):
        """
        Manually refresh the swagger document. This can help resolve errors communicate with the API.
        """
        try:
            os.remove(self._get_swagger_filename(self.swagger_url))
        except EnvironmentError as e:
            logger.warn(os.strerror(e.errno))
        else:
            self.__init__()

def view_500(request, url=None):
    """
    it returns a 500 http response
    """
    res = render_to_response("500.html", context_instance=RequestContext(request))
    res.status_code = 500
    return res

def pop():
        """Remove instance from instance list"""
        pid = os.getpid()
        thread = threading.current_thread()
        Wdb._instances.pop((pid, thread))

def cleanup():
    """Cleanup the output directory"""
    if _output_dir and os.path.exists(_output_dir):
        log.msg_warn("Cleaning up output directory at '{output_dir}' ..."
                     .format(output_dir=_output_dir))
        if not _dry_run:
            shutil.rmtree(_output_dir)

def strip_head(sequence, values):
    """Strips elements of `values` from the beginning of `sequence`."""
    values = set(values)
    return list(itertools.dropwhile(lambda x: x in values, sequence))

def __next__(self, reward, ask_id, lbl):
        """For Python3 compatibility of generator."""
        return self.next(reward, ask_id, lbl)

def std_datestr(self, datestr):
        """Reformat a date string to standard format.
        """
        return date.strftime(
                self.str2date(datestr), self.std_dateformat)

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

def setdefault(self, name: str, default: Any=None) -> Any:
        """Set an attribute with a default value."""
        return self.__dict__.setdefault(name, default)

def Square(x, a, b, c):
    """Second order polynomial

    Inputs:
    -------
        ``x``: independent variable
        ``a``: coefficient of the second-order term
        ``b``: coefficient of the first-order term
        ``c``: additive constant

    Formula:
    --------
        ``a*x^2 + b*x + c``
    """
    return a * x ** 2 + b * x + c

def register(linter):
    """Register the reporter classes with the linter."""
    linter.register_reporter(TextReporter)
    linter.register_reporter(ParseableTextReporter)
    linter.register_reporter(VSTextReporter)
    linter.register_reporter(ColorizedTextReporter)

def covstr(s):
  """ convert string to int or float. """
  try:
    ret = int(s)
  except ValueError:
    ret = float(s)
  return ret

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

def dict_to_querystring(dictionary):
    """Converts a dict to a querystring suitable to be appended to a URL."""
    s = u""
    for d in dictionary.keys():
        s = unicode.format(u"{0}{1}={2}&", s, d, dictionary[d])
    return s[:-1]

def get_indentation(line):
    """Return leading whitespace."""
    if line.strip():
        non_whitespace_index = len(line) - len(line.lstrip())
        return line[:non_whitespace_index]
    else:
        return ''

def __setitem__(self, field, value):
        """ :see::meth:RedisMap.__setitem__ """
        return self._client.hset(self.key_prefix, field, self._dumps(value))

def __init__(self, testnet=False, dryrun=False):
        """TODO doc string"""
        self.testnet = testnet
        self.dryrun = dryrun

def FindMethodByName(self, name):
    """Searches for the specified method, and returns its descriptor."""
    for method in self.methods:
      if name == method.name:
        return method
    return None

def _close(self):
        """
        Closes the client connection to the database.
        """
        if self.connection:
            with self.wrap_database_errors:
                self.connection.client.close()

def ms_to_datetime(ms):
    """
    Converts a millisecond accuracy timestamp to a datetime
    """
    dt = datetime.datetime.utcfromtimestamp(ms / 1000)
    return dt.replace(microsecond=(ms % 1000) * 1000).replace(tzinfo=pytz.utc)

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

def parse(filename):
    """Parse ASDL from the given file and return a Module node describing it."""
    with open(filename) as f:
        parser = ASDLParser()
        return parser.parse(f.read())

def method_header(method_name, nogil=False, idx_as_arg=False):
    """Returns the Cython method header for methods without arguments except
    `self`."""
    if not config.FASTCYTHON:
        nogil = False
    header = 'cpdef inline void %s(self' % method_name
    header += ', int idx)' if idx_as_arg else ')'
    header += ' nogil:' if nogil else ':'
    return header

def struct2dict(struct):
    """convert a ctypes structure to a dictionary"""
    return {x: getattr(struct, x) for x in dict(struct._fields_).keys()}

def translate_fourier(image, dx):
    """ Translate an image in fourier-space with plane waves """
    N = image.shape[0]

    f = 2*np.pi*np.fft.fftfreq(N)
    kx,ky,kz = np.meshgrid(*(f,)*3, indexing='ij')
    kv = np.array([kx,ky,kz]).T

    q = np.fft.fftn(image)*np.exp(-1.j*(kv*dx).sum(axis=-1)).T
    return np.real(np.fft.ifftn(q))

async def async_input(prompt):
    """
    Python's ``input()`` is blocking, which means the event loop we set
    above can't be running while we're blocking there. This method will
    let the loop run while we wait for input.
    """
    print(prompt, end='', flush=True)
    return (await loop.run_in_executor(None, sys.stdin.readline)).rstrip()

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

def format_exc(*exc_info):
    """Show exception with traceback."""
    typ, exc, tb = exc_info or sys.exc_info()
    error = traceback.format_exception(typ, exc, tb)
    return "".join(error)

def p(self):
        """
        Helper property containing the percentage this slider is "filled".
        
        This property is read-only.
        """
        return (self.n-self.nmin)/max((self.nmax-self.nmin),1)

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

def chunks(dictionary, chunk_size):
    """
    Yield successive n-sized chunks from dictionary.
    """
    iterable = iter(dictionary)
    for __ in range(0, len(dictionary), chunk_size):
        yield {key: dictionary[key] for key in islice(iterable, chunk_size)}

def strip_spaces(x):
    """
    Strips spaces
    :param x:
    :return:
    """
    x = x.replace(b' ', b'')
    x = x.replace(b'\t', b'')
    return x

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

def get_oauth_token():
    """Retrieve a simple OAuth Token for use with the local http client."""
    url = "{0}/token".format(DEFAULT_ORIGIN["Origin"])
    r = s.get(url=url)
    return r.json()["t"]

def _convert_latitude(self, latitude):
        """Convert from latitude to the y position in overall map."""
        return int((180 - (180 / pi * log(tan(
            pi / 4 + latitude * pi / 360)))) * (2 ** self._zoom) * self._size / 360)

def get_line_number(line_map, offset):
    """Find a line number, given a line map and a character offset."""
    for lineno, line_offset in enumerate(line_map, start=1):
        if line_offset > offset:
            return lineno
    return -1

def _do_auto_predict(machine, X, *args):
    """Performs an automatic prediction for the specified machine and returns
    the predicted values.
    """
    if auto_predict and hasattr(machine, "predict"):
        return machine.predict(X)

def zs(inlist):
    """
Returns a list of z-scores, one for each score in the passed list.

Usage:   lzs(inlist)
"""
    zscores = []
    for item in inlist:
        zscores.append(z(inlist, item))
    return zscores

def _repr_strip(mystring):
    """
    Returns the string without any initial or final quotes.
    """
    r = repr(mystring)
    if r.startswith("'") and r.endswith("'"):
        return r[1:-1]
    else:
        return r

def _protected_log(x1):
    """Closure of log for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)

def normalize(v, axis=None, eps=1e-10):
  """L2 Normalize along specified axes."""
  return v / max(anorm(v, axis=axis, keepdims=True), eps)

def reverse_code_map(self):
        """Return a map from a code ( usually a string ) to the  shorter numeric value"""

        return {c.value: (c.ikey if c.ikey else c.key) for c in self.codes}

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

def bytes_to_bits(bytes_):
    """Convert bytes to a list of bits
    """
    res = []
    for x in bytes_:
        if not isinstance(x, int):
            x = ord(x)
        res += byte_to_bits(x)
    return res

def unique(list):
    """ Returns a copy of the list without duplicates.
    """
    unique = []; [unique.append(x) for x in list if x not in unique]
    return unique

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

def print_table(*args, **kwargs):
    """
    if csv:
        import csv
        t = csv.writer(sys.stdout, delimiter=";")
        t.writerow(header)
    else:
        t = PrettyTable(header)
        t.align = "r"
        t.align["details"] = "l"
    """
    t = format_table(*args, **kwargs)
    click.echo(t)

def axes_off(ax):
    """Get rid of all axis ticks, lines, etc.
    """
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)

def save_model(self, request, obj, form, change):
        """
        Set currently authenticated user as the author of the gallery.
        """
        obj.author = request.user
        obj.save()

def raise_for_not_ok_status(response):
    """
    Raises a `requests.exceptions.HTTPError` if the response has a non-200
    status code.
    """
    if response.code != OK:
        raise HTTPError('Non-200 response code (%s) for url: %s' % (
            response.code, uridecode(response.request.absoluteURI)))

    return response

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

def imp_print(self, text, end):
		"""Directly send utf8 bytes to stdout"""
		sys.stdout.write((text + end).encode("utf-8"))

def get_size(self):
        """see doc in Term class"""
        self.curses.setupterm()
        return self.curses.tigetnum('cols'), self.curses.tigetnum('lines')

def add_text_to_image(fname, txt, opFilename):
    """ convert an image by adding text """
    ft = ImageFont.load("T://user//dev//src//python//_AS_LIB//timR24.pil")
    #wh = ft.getsize(txt)
    print("Adding text ", txt, " to ", fname, " pixels wide to file " , opFilename)
    im = Image.open(fname)
    draw = ImageDraw.Draw(im)
    draw.text((0, 0), txt, fill=(0, 0, 0), font=ft)
    del draw  
    im.save(opFilename)

def _connection_failed(self, error="Error not specified!"):
        """Clean up after connection failure detected."""
        if not self._error:
            LOG.error("Connection failed: %s", str(error))
            self._error = error

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

def title(self):
        """ The title of this window """
        with switch_window(self._browser, self.name):
            return self._browser.title

def get_in_samples(samples, fn):
    """
    for a list of samples, return the value of a global option
    """
    for sample in samples:
        sample = to_single_data(sample)
        if fn(sample, None):
            return fn(sample)
    return None

def is_empty(self):
    """Returns True if this node has no children, or if all of its children are ParseNode instances
    and are empty.
    """
    return all(isinstance(c, ParseNode) and c.is_empty for c in self.children)

def get_unixtime_registered(self):
        """Returns the user's registration date as a UNIX timestamp."""

        doc = self._request(self.ws_prefix + ".getInfo", True)

        return int(doc.getElementsByTagName("registered")[0].getAttribute("unixtime"))

def proper_round(n):
    """
    rounds float to closest int
    :rtype: int
    :param n: float
    """
    return int(n) + (n / abs(n)) * int(abs(n - int(n)) >= 0.5) if n != 0 else 0

def make_key(observer):
        """Construct a unique, hashable, immutable key for an observer."""

        if hasattr(observer, "__self__"):
            inst = observer.__self__
            method_name = observer.__name__
            key = (id(inst), method_name)
        else:
            key = id(observer)
        return key

def remove_unsafe_chars(text):
    """Remove unsafe unicode characters from a piece of text."""
    if isinstance(text, six.string_types):
        text = UNSAFE_RE.sub('', text)
    return text

def clear_worker_output(self):
        """Drops all of the worker output collections
            Args:
                None
            Returns:
                Nothing
        """
        self.data_store.clear_worker_output()

        # Have the plugin manager reload all the plugins
        self.plugin_manager.load_all_plugins()

        # Store information about commands and workbench
        self._store_information()

def __next__(self):
    """Pop the head off the iterator and return it."""
    res = self._head
    self._fill()
    if res is None:
      raise StopIteration()
    return res

def get_git_branch(git_path='git'):
    """Returns the name of the current git branch
    """
    branch_match = call((git_path, 'rev-parse', '--symbolic-full-name', 'HEAD'))
    if branch_match == "HEAD":
        return None
    else:
        return os.path.basename(branch_match)

def _relpath(name):
    """
    Strip absolute components from path.
    Inspired from zipfile.write().
    """
    return os.path.normpath(os.path.splitdrive(name)[1]).lstrip(_allsep)

def generate_id(self):
        """Generate a fresh id"""
        if self.use_repeatable_ids:
            self.repeatable_id_counter += 1
            return 'autobaked-{}'.format(self.repeatable_id_counter)
        else:
            return str(uuid4())

def push(h, x):
    """Push a new value into heap."""
    h.push(x)
    up(h, h.size()-1)

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

def add_range(self, sequence, begin, end):
    """Add a read_range primitive"""
    sequence.parser_tree = parsing.Range(self.value(begin).strip("'"),
                                         self.value(end).strip("'"))
    return True

def local_accuracy(X_train, y_train, X_test, y_test, attr_test, model_generator, metric, trained_model):
    """ The how well do the features plus a constant base rate sum up to the model output.
    """

    X_train, X_test = to_array(X_train, X_test)

    # how many features to mask
    assert X_train.shape[1] == X_test.shape[1]

    # keep nkeep top features and re-train the model for each test explanation
    yp_test = trained_model.predict(X_test)

    return metric(yp_test, strip_list(attr_test).sum(1))

def _split_batches(self, data, batch_size):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size]

def each_img(img_dir):
    """
    Reads and iterates through each image file in the given directory
    """
    for fname in utils.each_img(img_dir):
        fname = os.path.join(img_dir, fname)
        yield cv.imread(fname), fname

def get_remote_content(filepath):
        """ A handy wrapper to get a remote file content """
        with hide('running'):
            temp = BytesIO()
            get(filepath, temp)
            content = temp.getvalue().decode('utf-8')
        return content.strip()

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

def lowPass(self, *args):
        """
        Creates a copy of the signal with the low pass applied, args specifed are passed through to _butter. 
        :return: 
        """
        return Signal(self._butter(self.samples, 'low', *args), fs=self.fs)

def benchmark(store, n=10000):
    """
    Iterates over all of the referreds, and then iterates over all of the
    referrers that refer to each one.

    Fairly item instantiation heavy.
    """
    R = Referrer

    for referred in store.query(Referred):
        for _reference in store.query(R, R.reference == referred):
            pass

def post_process(self):
        """ Apply last 2D transforms"""
        self.image.putdata(self.pixels)
        self.image = self.image.transpose(Image.ROTATE_90)

def __len__(self):
		"""Get a list of the public data attributes."""
		return len([i for i in (set(dir(self)) - self._STANDARD_ATTRS) if i[0] != '_'])

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

def segments_to_numpy(segments):
    """given a list of 4-element tuples, transforms it into a numpy array"""
    segments = numpy.array(segments, dtype=SEGMENT_DATATYPE, ndmin=2)  # each segment in a row
    segments = segments if SEGMENTS_DIRECTION == 0 else numpy.transpose(segments)
    return segments

def run_hive_script(script):
    """
    Runs the contents of the given script in hive and returns stdout.
    """
    if not os.path.isfile(script):
        raise RuntimeError("Hive script: {0} does not exist.".format(script))
    return run_hive(['-f', script])

def nrows_expected(self):
        """ based on our axes, compute the expected nrows """
        return np.prod([i.cvalues.shape[0] for i in self.index_axes])

def pause(self):
        """Pause the music"""
        mixer.music.pause()
        self.pause_time = self.get_time()
        self.paused = True

def write_pid_file():
    """Write a file with the PID of this server instance.

    Call when setting up a command line testserver.
    """
    pidfile = os.path.basename(sys.argv[0])[:-3] + '.pid'  # strip .py, add .pid
    with open(pidfile, 'w') as fh:
        fh.write("%d\n" % os.getpid())
        fh.close()

def _find(string, sub_string, start_index):
    """Return index of sub_string in string.

    Raise TokenError if sub_string is not found.
    """
    result = string.find(sub_string, start_index)
    if result == -1:
        raise TokenError("expected '{0}'".format(sub_string))
    return result

def get_groups(self, username):
        """Get all groups of a user"""
        username = ldap.filter.escape_filter_chars(self._byte_p2(username))
        userdn = self._get_user(username, NO_ATTR)

        searchfilter = self.group_filter_tmpl % {
            'userdn': userdn,
            'username': username
        }

        groups = self._search(searchfilter, NO_ATTR, self.groupdn)
        ret = []
        for entry in groups:
            ret.append(self._uni(entry[0]))
        return ret

def get_encoding(binary):
    """Return the encoding type."""

    try:
        from chardet import detect
    except ImportError:
        LOGGER.error("Please install the 'chardet' module")
        sys.exit(1)

    encoding = detect(binary).get('encoding')

    return 'iso-8859-1' if encoding == 'CP949' else encoding

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

def inspect_cuda():
    """ Return cuda device information and nvcc/cuda setup """
    nvcc_settings = nvcc_compiler_settings()
    sysconfig.get_config_vars()
    nvcc_compiler = ccompiler.new_compiler()
    sysconfig.customize_compiler(nvcc_compiler)
    customize_compiler_for_nvcc(nvcc_compiler, nvcc_settings)

    output = inspect_cuda_version_and_devices(nvcc_compiler, nvcc_settings)

    return json.loads(output), nvcc_settings

def first_digits(s, default=0):
    """Return the fist (left-hand) digits in a string as a single integer, ignoring sign (+/-).
    >>> first_digits('+123.456')
    123
    """
    s = re.split(r'[^0-9]+', str(s).strip().lstrip('+-' + charlist.whitespace))
    if len(s) and len(s[0]):
        return int(s[0])
    return default

def compute_depth(self):
        """
        Recursively computes true depth of the subtree. Should only
        be needed for debugging. Unless something is wrong, the
        depth field should reflect the correct depth of the subtree.
        """
        left_depth = self.left_node.compute_depth() if self.left_node else 0
        right_depth = self.right_node.compute_depth() if self.right_node else 0
        return 1 + max(left_depth, right_depth)

def write_padding(fp, size, divisor=2):
    """
    Writes padding bytes given the currently written size.

    :param fp: file-like object
    :param divisor: divisor of the byte alignment
    :return: written byte size
    """
    remainder = size % divisor
    if remainder:
        return write_bytes(fp, struct.pack('%dx' % (divisor - remainder)))
    return 0

def _svd(cls, matrix, num_concepts=5):
        """
        Perform singular value decomposition for dimensionality reduction of the input matrix.
        """
        u, s, v = svds(matrix, k=num_concepts)
        return u, s, v

def fit(self, X):
        """ Apply KMeans Clustering
              X: dataset with feature vectors
        """
        self.centers_, self.labels_, self.sse_arr_, self.n_iter_ = \
              _kmeans(X, self.n_clusters, self.max_iter, self.n_trials, self.tol)

def mask_and_flatten(self):
        """Return a vector of the masked data.

        Returns
        -------
        np.ndarray, tuple of indices (np.ndarray), tuple of the mask shape
        """
        self._check_for_mask()

        return self.get_data(smoothed=True, masked=True, safe_copy=False)[self.get_mask_indices()],\
               self.get_mask_indices(), self.mask.shape

def step_impl06(context):
    """Prepare test for singleton property.

    :param context: test context.
    """
    store = context.SingleStore
    context.st_1 = store()
    context.st_2 = store()
    context.st_3 = store()

def generate_seed(seed):
    """Generate seed for random number generator"""
    if seed is None:
        random.seed()
        seed = random.randint(0, sys.maxsize)
    random.seed(a=seed)

    return seed

def replace_variable_node(node, annotation):
    """Replace a node annotated by `nni.variable`.
    node: the AST node to replace
    annotation: annotation string
    """
    assert type(node) is ast.Assign, 'nni.variable is not annotating assignment expression'
    assert len(node.targets) == 1, 'Annotated assignment has more than one left-hand value'
    name, expr = parse_nni_variable(annotation)
    assert test_variable_equal(node.targets[0], name), 'Annotated variable has wrong name'
    node.value = expr
    return node

def home_lib(home):
    """Return the lib dir under the 'home' installation scheme"""
    if hasattr(sys, 'pypy_version_info'):
        lib = 'site-packages'
    else:
        lib = os.path.join('lib', 'python')
    return os.path.join(home, lib)

def seconds_to_hms(seconds):
    """
    Converts seconds float to 'hh:mm:ss.ssssss' format.
    """
    hours = int(seconds / 3600.0)
    minutes = int((seconds / 60.0) % 60.0)
    secs = float(seconds % 60.0)
    return "{0:02d}:{1:02d}:{2:02.6f}".format(hours, minutes, secs)

def contextMenuEvent(self, event):
        """Reimplement Qt method"""
        self.menu.popup(event.globalPos())
        event.accept()

def _try_compile(source, name):
    """Attempts to compile the given source, first as an expression and
       then as a statement if the first approach fails.

       Utility function to accept strings in functions that otherwise
       expect code objects
    """
    try:
        c = compile(source, name, 'eval')
    except SyntaxError:
        c = compile(source, name, 'exec')
    return c

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

def get_example_features(example):
  """Returns the non-sequence features from the provided example."""
  return (example.features.feature if isinstance(example, tf.train.Example)
          else example.context.feature)

def _GetProxies(self):
    """Gather a list of proxies to use."""
    # Detect proxies from the OS environment.
    result = client_utils.FindProxies()

    # Also try to connect directly if all proxies fail.
    result.append("")

    # Also try all proxies configured in the config system.
    result.extend(config.CONFIG["Client.proxy_servers"])

    return result

def _visual_width(line):
    """Get the the number of columns required to display a string"""

    return len(re.sub(colorama.ansitowin32.AnsiToWin32.ANSI_CSI_RE, "", line))

def fetch_token(self, **kwargs):
        """Exchange a code (and 'state' token) for a bearer token"""
        return super(AsanaOAuth2Session, self).fetch_token(self.token_url, client_secret=self.client_secret, **kwargs)

def retry_on_signal(function):
    """Retries function until it doesn't raise an EINTR error"""
    while True:
        try:
            return function()
        except EnvironmentError, e:
            if e.errno != errno.EINTR:
                raise

def mock_decorator(*args, **kwargs):
    """Mocked decorator, needed in the case we need to mock a decorator"""
    def _called_decorator(dec_func):
        @wraps(dec_func)
        def _decorator(*args, **kwargs):
            return dec_func()
        return _decorator
    return _called_decorator

def __remove_trailing_zeros(self, collection):
        """Removes trailing zeroes from indexable collection of numbers"""
        index = len(collection) - 1
        while index >= 0 and collection[index] == 0:
            index -= 1

        return collection[:index + 1]

def is_in(self, point_x, point_y):
        """ Test if a point is within this polygonal region """

        point_array = array(((point_x, point_y),))
        vertices = array(self.points)
        winding = self.inside_rule == "winding"
        result = points_in_polygon(point_array, vertices, winding)
        return result[0]

def rq_job(self):
        """The last RQ Job this ran on"""
        if not self.rq_id or not self.rq_origin:
            return
        try:
            return RQJob.fetch(self.rq_id, connection=get_connection(self.rq_origin))
        except NoSuchJobError:
            return

def get_from_human_key(self, key):
        """Return the key (aka database value) of a human key (aka Python identifier)."""
        if key in self._identifier_map:
            return self._identifier_map[key]
        raise KeyError(key)

def linregress(x, y, return_stats=False):
    """linear regression calculation

    Parameters
    ----
    x :         independent variable (series)
    y :         dependent variable (series)
    return_stats : returns statistical values as well if required (bool)
    

    Returns
    ----
    list of parameters (and statistics)
    """
    a1, a0, r_value, p_value, stderr = scipy.stats.linregress(x, y)

    retval = a1, a0
    if return_stats:
        retval += r_value, p_value, stderr

    return retval

def bounding_box(img):
    r"""
    Return the bounding box incorporating all non-zero values in the image.
    
    Parameters
    ----------
    img : array_like
        An array containing non-zero objects.
        
    Returns
    -------
    bbox : a list of slicer objects defining the bounding box
    """
    locations = numpy.argwhere(img)
    mins = locations.min(0)
    maxs = locations.max(0) + 1
    return [slice(x, y) for x, y in zip(mins, maxs)]

def mod(value, arg):
    """Return the modulo value."""
    try:
        return valid_numeric(value) % valid_numeric(arg)
    except (ValueError, TypeError):
        try:
            return value % arg
        except Exception:
            return ''

def _process_and_sort(s, force_ascii, full_process=True):
    """Return a cleaned string with token sorted."""
    # pull tokens
    ts = utils.full_process(s, force_ascii=force_ascii) if full_process else s
    tokens = ts.split()

    # sort tokens and join
    sorted_string = u" ".join(sorted(tokens))
    return sorted_string.strip()

def splitBy(data, num):
    """ Turn a list to list of list """
    return [data[i:i + num] for i in range(0, len(data), num)]

def set_history_file(self, path):
        """Set path to history file. "" produces no file."""
        if path:
            self.history = prompt_toolkit.history.FileHistory(fixpath(path))
        else:
            self.history = prompt_toolkit.history.InMemoryHistory()

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

def columnclean(column):
        """
        Modifies column header format to be importable into a database
        :param column: raw column header
        :return: cleanedcolumn: reformatted column header
        """
        cleanedcolumn = str(column) \
            .replace('%', 'percent') \
            .replace('(', '_') \
            .replace(')', '') \
            .replace('As', 'Adenosines') \
            .replace('Cs', 'Cytosines') \
            .replace('Gs', 'Guanines') \
            .replace('Ts', 'Thymines') \
            .replace('Ns', 'Unknowns') \
            .replace('index', 'adapterIndex')
        return cleanedcolumn

def strip_spaces(value, sep=None, join=True):
    """Cleans trailing whitespaces and replaces also multiple whitespaces with a single space."""
    value = value.strip()
    value = [v.strip() for v in value.split(sep)]
    join_sep = sep or ' '
    return join_sep.join(value) if join else value

def open_json(file_name):
    """
    returns json contents as string
    """
    with open(file_name, "r") as json_data:
        data = json.load(json_data)
        return data

def beautify(string, *args, **kwargs):
	"""
		Convenient interface to the ecstasy package.

		Arguments:
			string (str): The string to beautify with ecstasy.
			args (list): The positional arguments.
			kwargs (dict): The keyword ('always') arguments.
	"""

	parser = Parser(args, kwargs)
	return parser.beautify(string)

def get_frame_locals(stepback=0):
    """Returns locals dictionary from a given frame.

    :param int stepback:

    :rtype: dict

    """
    with Frame(stepback=stepback) as frame:
        locals_dict = frame.f_locals

    return locals_dict

def attr_names(cls) -> List[str]:
        """
        Returns annotated attribute names
        :return: List[str]
        """
        return [k for k, v in cls.attr_types().items()]

def download_file_from_bucket(self, bucket, file_path, key):
        """ Download file from S3 Bucket """
        with open(file_path, 'wb') as data:
            self.__s3.download_fileobj(bucket, key, data)
            return file_path

def _to_json(self):
        """ Gets a dict of this object's properties so that it can be used to send a dump to the client """
        return dict(( (k, v) for k, v in self.__dict__.iteritems() if k != 'server'))

def show():
    """Show (print out) current environment variables."""
    env = get_environment()

    for key, val in sorted(env.env.items(), key=lambda item: item[0]):
        click.secho('%s = %s' % (key, val))

def _str_to_list(s):
    """Converts a comma separated string to a list"""
    _list = s.split(",")
    return list(map(lambda i: i.lstrip(), _list))

def select_down(self):
        """move cursor down"""
        r, c = self._index
        self._select_index(r+1, c)

def previous_friday(dt):
    """
    If holiday falls on Saturday or Sunday, use previous Friday instead.
    """
    if dt.weekday() == 5:
        return dt - timedelta(1)
    elif dt.weekday() == 6:
        return dt - timedelta(2)
    return dt

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

def mad(v):
    """MAD -- Median absolute deviation. More robust than standard deviation.
    """
    return np.median(np.abs(v - np.median(v)))

def cpp_prog_builder(build_context, target):
    """Build a C++ binary executable"""
    yprint(build_context.conf, 'Build CppProg', target)
    workspace_dir = build_context.get_workspace('CppProg', target.name)
    build_cpp(build_context, target, target.compiler_config, workspace_dir)

def in_directory(path):
    """Context manager (with statement) that changes the current directory
    during the context.
    """
    curdir = os.path.abspath(os.curdir)
    os.chdir(path)
    yield
    os.chdir(curdir)

def get_enum_documentation(class_name, module_name, enum_class_object):
    documentation = """.. _{module_name}.{class_name}:

``enum {class_name}``
+++++++{plus}++

**module:** ``{module_name}``""".format(
        module_name=module_name,
        class_name=class_name,
        plus='+' * len(class_name),
    )

    if enum_class_object.__doc__ and enum_class_object.__doc__.strip():
        documentation += '\n\n{}'.format(_clean_literals(inspect.cleandoc(enum_class_object.__doc__)))

    documentation += '\n\nConstant Values:\n'
    for e in enum_class_object:
        documentation += '\n- ``{}`` (``{}``)'.format(e.name, repr(e.value).lstrip('u'))

    return documentation

def generic_div(a, b):
    """Simple function to divide two numbers"""
    logger.debug('Called generic_div({}, {})'.format(a, b))
    return a / b

def order_by(self, *fields):
        """An alternate to ``sort`` which allows you to specify a list
        of fields and use a leading - (minus) to specify DESCENDING."""
        doc = []
        for field in fields:
            if field.startswith('-'):
                doc.append((field.strip('-'), pymongo.DESCENDING))
            else:
                doc.append((field, pymongo.ASCENDING))
        return self.sort(doc)

def set_label ( self, object, label ):
        """ Sets the label for a specified object.
        """
        label_name = self.label
        if label_name[:1] != '=':
            xsetattr( object, label_name, label )

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

def load(self, filename='classifier.dump'):
        """
        Unpickles the classifier used
        """
        ifile = open(filename, 'r+')
        self.classifier = pickle.load(ifile)
        ifile.close()

def import_by_path(path: str) -> Callable:
    """Import a class or function given it's absolute path.

    Parameters
    ----------
    path:
      Path to object to import
    """

    module_path, _, class_name = path.rpartition('.')
    return getattr(import_module(module_path), class_name)

def visit_Str(self, node):
        """ Set the pythonic string type. """
        self.result[node] = self.builder.NamedType(pytype_to_ctype(str))

def camelcase(string):
    """ Convert string into camel case.

    Args:
        string: String to convert.

    Returns:
        string: Camel case string.

    """

    string = re.sub(r"^[\-_\.]", '', str(string))
    if not string:
        return string
    return lowercase(string[0]) + re.sub(r"[\-_\.\s]([a-z])", lambda matched: uppercase(matched.group(1)), string[1:])

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

def to_lisp(o, keywordize_keys: bool = True):
    """Recursively convert Python collections into Lisp collections."""
    if not isinstance(o, (dict, frozenset, list, set, tuple)):
        return o
    else:  # pragma: no cover
        return _to_lisp_backup(o, keywordize_keys=keywordize_keys)

def copy(self):
        """
        Return a copy of the dictionary.

        This is a middle-deep copy; the copy is independent of the original in
        all attributes that have mutable types except for:

        * The values in the dictionary

        Note that the Python functions :func:`py:copy.copy` and
        :func:`py:copy.deepcopy` can be used to create completely shallow or
        completely deep copies of objects of this class.
        """
        result = NocaseDict()
        result._data = self._data.copy()  # pylint: disable=protected-access
        return result

def fix(h, i):
    """Rearrange the heap after the item at position i got updated."""
    down(h, i, h.size())
    up(h, i)

def Gaussian(x, a, x0, sigma, y0):
    """Gaussian peak

    Inputs:
    -------
        ``x``: independent variable
        ``a``: scaling factor (extremal value)
        ``x0``: center
        ``sigma``: half width at half maximum
        ``y0``: additive constant

    Formula:
    --------
        ``a*exp(-(x-x0)^2)/(2*sigma^2)+y0``
    """
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + y0

def object_as_dict(obj):
    """Turn an SQLAlchemy model into a dict of field names and values.

    Based on https://stackoverflow.com/a/37350445/1579058
    """
    return {c.key: getattr(obj, c.key)
            for c in inspect(obj).mapper.column_attrs}

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

def lighting(im, b, c):
    """ Adjust image balance and contrast """
    if b==0 and c==1: return im
    mu = np.average(im)
    return np.clip((im-mu)*c+mu+b,0.,1.).astype(np.float32)

def rpc_fix_code_with_black(self, source, directory):
        """Formats Python code to conform to the PEP 8 style guide.

        """
        source = get_source(source)
        return fix_code_with_black(source, directory)

def str2int(num, radix=10, alphabet=BASE85):
    """helper function for quick base conversions from strings to integers"""
    return NumConv(radix, alphabet).str2int(num)

def series_index(self, series):
        """
        Return the integer index of *series* in this sequence.
        """
        for idx, s in enumerate(self):
            if series is s:
                return idx
        raise ValueError('series not in chart data object')

def read_next_block(infile, block_size=io.DEFAULT_BUFFER_SIZE):
    """Iterates over the file in blocks."""
    chunk = infile.read(block_size)

    while chunk:
        yield chunk

        chunk = infile.read(block_size)

def find_coord_vars(ncds):
    """
    Finds all coordinate variables in a dataset.

    A variable with the same name as a dimension is called a coordinate variable.
    """
    coord_vars = []

    for d in ncds.dimensions:
        if d in ncds.variables and ncds.variables[d].dimensions == (d,):
            coord_vars.append(ncds.variables[d])

    return coord_vars

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

def get_plain_image_as_widget(self):
        """Used for generating thumbnails.  Does not include overlaid
        graphics.
        """
        arr = self.getwin_array(order=self.rgb_order)
        image = self._get_qimage(arr, self.qimg_fmt)
        return image

def sort_data(x, y):
    """Sort the data."""
    xy = sorted(zip(x, y))
    x, y = zip(*xy)
    return x, y

def previous_key(tuple_of_tuples, key):
    """Returns the key which comes before the give key.

    It Processes a tuple of 2-element tuples and returns the key which comes
    before the given key.
    """
    for i, t in enumerate(tuple_of_tuples):
        if t[0] == key:
            try:
                return tuple_of_tuples[i - 1][0]
            except IndexError:
                return None

def session_to_epoch(timestamp):
    """ converts Synergy Timestamp for session to UTC zone seconds since epoch """
    utc_timetuple = datetime.strptime(timestamp, SYNERGY_SESSION_PATTERN).replace(tzinfo=None).utctimetuple()
    return calendar.timegm(utc_timetuple)

def page_align_content_length(length):
    # type: (int) -> int
    """Compute page boundary alignment
    :param int length: content length
    :rtype: int
    :return: aligned byte boundary
    """
    mod = length % _PAGEBLOB_BOUNDARY
    if mod != 0:
        return length + (_PAGEBLOB_BOUNDARY - mod)
    return length

def error(*args):
    """Display error message via stderr or GUI."""
    if sys.stdin.isatty():
        print('ERROR:', *args, file=sys.stderr)
    else:
        notify_error(*args)

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

def is_nullable_list(val, vtype):
    """Return True if list contains either values of type `vtype` or None."""
    return (isinstance(val, list) and
            any(isinstance(v, vtype) for v in val) and
            all((isinstance(v, vtype) or v is None) for v in val))

def yaml(self):
        """
        returns the yaml output of the dict.
        """
        return ordered_dump(OrderedDict(self),
                            Dumper=yaml.SafeDumper,
                            default_flow_style=False)

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

def save_json(object, handle, indent=2):
    """Save object as json on CNS."""
    obj_json = json.dumps(object, indent=indent, cls=NumpyJSONEncoder)
    handle.write(obj_json)

def compare(dicts):
    """Compare by iteration"""

    common_members = {}
    common_keys = reduce(lambda x, y: x & y, map(dict.keys, dicts))
    for k in common_keys:
        common_members[k] = list(
            reduce(lambda x, y: x & y, [set(d[k]) for d in dicts]))

    return common_members

def rmfile(path):
    """Ensure file deleted also on *Windows* where read-only files need special treatment."""
    if osp.isfile(path):
        if is_win:
            os.chmod(path, 0o777)
        os.remove(path)

def _to_numeric(val):
    """
    Helper function for conversion of various data types into numeric representation.
    """
    if isinstance(val, (int, float, datetime.datetime, datetime.timedelta)):
        return val
    return float(val)

def glr_path_static():
    """Returns path to packaged static files"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '_static'))

def index_all(self, index_name):
        """Index all available documents, using streaming_bulk for speed
        Args:

        index_name (string): The index
        """
        oks = 0
        notoks = 0
        for ok, item in streaming_bulk(
            self.es_client,
            self._iter_documents(index_name)
        ):
            if ok:
                oks += 1
            else:
                notoks += 1
        logging.info(
            "Import results: %d ok, %d not ok",
            oks,
            notoks
        )

def index_nearest(value, array):
    """
    expects a _n.array
    returns the global minimum of (value-array)^2
    """

    a = (array-value)**2
    return index(a.min(), a)

def pods(self):
        """ A list of kubernetes pods corresponding to current workers

        See Also
        --------
        KubeCluster.logs
        """
        return self.core_api.list_namespaced_pod(
            self.namespace,
            label_selector=format_labels(self.pod_template.metadata.labels)
        ).items

def _eager_tasklet(tasklet):
  """Decorator to turn tasklet to run eagerly."""

  @utils.wrapping(tasklet)
  def eager_wrapper(*args, **kwds):
    fut = tasklet(*args, **kwds)
    _run_until_rpc()
    return fut

  return eager_wrapper

def setdict(self, D):
        """Set dictionary array."""

        self.D = np.asarray(D, dtype=self.dtype)

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

def OnContextMenu(self, event):
        """Context menu event handler"""

        self.grid.PopupMenu(self.grid.contextmenu)

        event.Skip()

def is_valid_regex(regex):
    """Function for checking a valid regex."""
    if len(regex) == 0:
        return False
    try:
        re.compile(regex)
        return True
    except sre_constants.error:
        return False

def as_list(callable):
    """Convert a scalar validator in a list validator"""
    @wraps(callable)
    def wrapper(value_iter):
        return [callable(value) for value in value_iter]

    return wrapper

def _trim(image):
    """Trim a PIL image and remove white space."""
    background = PIL.Image.new(image.mode, image.size, image.getpixel((0, 0)))
    diff = PIL.ImageChops.difference(image, background)
    diff = PIL.ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        image = image.crop(bbox)
    return image

def safe_setattr(obj, name, value):
    """Attempt to setattr but catch AttributeErrors."""
    try:
        setattr(obj, name, value)
        return True
    except AttributeError:
        return False

def draw_image(self, ax, image):
        """Process a matplotlib image object and call renderer.draw_image"""
        self.renderer.draw_image(imdata=utils.image_to_base64(image),
                                 extent=image.get_extent(),
                                 coordinates="data",
                                 style={"alpha": image.get_alpha(),
                                        "zorder": image.get_zorder()},
                                 mplobj=image)

def required_header(header):
    """Function that verify if the header parameter is a essential header

    :param header:  A string represented a header
    :returns:       A boolean value that represent if the header is required
    """
    if header in IGNORE_HEADERS:
        return False

    if header.startswith('HTTP_') or header == 'CONTENT_TYPE':
        return True

    return False

def match_empty(self, el):
        """Check if element is empty (if requested)."""

        is_empty = True
        for child in self.get_children(el, tags=False):
            if self.is_tag(child):
                is_empty = False
                break
            elif self.is_content_string(child) and RE_NOT_EMPTY.search(child):
                is_empty = False
                break
        return is_empty

def _sha1_for_file(filename):
    """Return sha1 for contents of filename."""
    with open(filename, "rb") as fileobj:
        contents = fileobj.read()
        return hashlib.sha1(contents).hexdigest()

def any_of(value, *args):
    """ At least one of the items in value should match """

    if len(args):
        value = (value,) + args

    return ExpectationAny(value)

def num_leaves(tree):
    """Determine the number of leaves in a tree"""
    if tree.is_leaf:
        return 1
    else:
        return num_leaves(tree.left_child) + num_leaves(tree.right_child)

def extend(a: dict, b: dict) -> dict:
    """Merge two dicts and return a new dict. Much like subclassing works."""
    res = a.copy()
    res.update(b)
    return res

def input_int_default(question="", default=0):
    """A function that works for both, Python 2.x and Python 3.x.
       It asks the user for input and returns it as a string.
    """
    answer = input_string(question)
    if answer == "" or answer == "yes":
        return default
    else:
        return int(answer)

def normalise_key(self, key):
        """Make sure key is a valid python attribute"""
        key = key.replace('-', '_')
        if key.startswith("noy_"):
            key = key[4:]
        return key

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

def synthesize(self, duration):
        """
        Synthesize white noise

        Args:
            duration (numpy.timedelta64): The duration of the synthesized sound
        """
        sr = self.samplerate.samples_per_second
        seconds = duration / Seconds(1)
        samples = np.random.uniform(low=-1., high=1., size=int(sr * seconds))
        return AudioSamples(samples, self.samplerate)

def coords_on_grid(self, x, y):
        """ Snap coordinates on the grid with integer coordinates """

        if isinstance(x, float):
            x = int(self._round(x))
        if isinstance(y, float):
            y = int(self._round(y))
        if not self._y_coord_down:
            y = self._extents - y
        return x, y

def set_slug(apps, schema_editor):
    """
    Create a slug for each Event already in the DB.
    """
    Event = apps.get_model('spectator_events', 'Event')

    for e in Event.objects.all():
        e.slug = generate_slug(e.pk)
        e.save(update_fields=['slug'])

def subat(orig, index, replace):
    """Substitutes the replacement string/character at the given index in the
    given string, returns the modified string.

    **Examples**:
    ::
        auxly.stringy.subat("bit", 2, "n")
    """
    return "".join([(orig[x] if x != index else replace) for x in range(len(orig))])

def alert(text='', title='', button=OK_TEXT, root=None, timeout=None):
    """Displays a simple message box with text and a single OK button. Returns the text of the button clicked on."""
    assert TKINTER_IMPORT_SUCCEEDED, 'Tkinter is required for pymsgbox'
    return _buttonbox(msg=text, title=title, choices=[str(button)], root=root, timeout=timeout)

def staticdir():
    """Return the location of the static data directory."""
    root = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(root, "static")

def values(self):
        """ :see::meth:RedisMap.keys """
        for val in self._client.hvals(self.key_prefix):
            yield self._loads(val)

def new_from_list(cls, items, **kwargs):
        """Populates the ListView with a string list.

        Args:
            items (list): list of strings to fill the widget with.
        """
        obj = cls(**kwargs)
        for item in items:
            obj.append(ListItem(item))
        return obj

def _dictfetchall(self, cursor):
        """ Return all rows from a cursor as a dict. """
        columns = [col[0] for col in cursor.description]
        return [
            dict(zip(columns, row))
            for row in cursor.fetchall()
        ]

def correlation_2D(image):
    """

    :param image: 2d image
    :return: psd1D, psd2D
    """
    # Take the fourier transform of the image.
    F1 = fftpack.fft2(image)

    # Now shift the quadrants around so that low spatial frequencies are in
    # the center of the 2D fourier transformed image.
    F2 = fftpack.fftshift(F1)

    # Calculate a 2D power spectrum
    psd2D = np.abs(F2)

    # Calculate the azimuthally averaged 1D power spectrum
    psd1D = analysis_util.azimuthalAverage(psd2D)
    return psd1D, psd2D

def maxId(self):
        """int: current max id of objects"""
        if len(self.model.db) == 0:
            return 0

        return max(map(lambda obj: obj["id"], self.model.db))

def to_camel_case(snake_case_name):
    """
    Converts snake_cased_names to CamelCaseNames.

    :param snake_case_name: The name you'd like to convert from.
    :type snake_case_name: string

    :returns: A converted string
    :rtype: string
    """
    bits = snake_case_name.split('_')
    return ''.join([bit.capitalize() for bit in bits])

def _init_glyph(self, plot, mapping, properties):
        """
        Returns a Bokeh glyph object.
        """
        properties = mpl_to_bokeh(properties)
        plot_method = self._plot_methods.get('batched' if self.batched else 'single')
        if isinstance(plot_method, tuple):
            # Handle alternative plot method for flipped axes
            plot_method = plot_method[int(self.invert_axes)]
        renderer = getattr(plot, plot_method)(**dict(properties, **mapping))
        return renderer, renderer.glyph

def val_mb(valstr: Union[int, str]) -> str:
    """
    Converts a value in bytes (in string format) to megabytes.
    """
    try:
        return "{:.3f}".format(int(valstr) / (1024 * 1024))
    except (TypeError, ValueError):
        return '?'

def call_on_if_def(obj, attr_name, callable, default, *args, **kwargs):
    """Calls the provided callable on the provided attribute of ``obj`` if it is defined.

    If not, returns default.
    """
    try:
        attr = getattr(obj, attr_name)
    except AttributeError:
        return default
    else:
        return callable(attr, *args, **kwargs)

def check_github(self):
        """
        If the requirement is frozen to a github url, check for new commits.

        API Tokens
        ----------
        For more than 50 github api calls per hour, pipchecker requires
        authentication with the github api by settings the environemnt
        variable ``GITHUB_API_TOKEN`` or setting the command flag
        --github-api-token='mytoken'``.

        To create a github api token for use at the command line::
             curl -u 'rizumu' -d '{"scopes":["repo"], "note":"pipchecker"}' https://api.github.com/authorizations

        For more info on github api tokens:
            https://help.github.com/articles/creating-an-oauth-token-for-command-line-use
            http://developer.github.com/v3/oauth/#oauth-authorizations-api

        Requirement Format
        ------------------
        Pipchecker gets the sha of frozen repo and checks if it is
        found at the head of any branches. If it is not found then
        the requirement is considered to be out of date.

        Therefore, freezing at the commit hash will provide the expected
        results, but if freezing at a branch or tag name, pipchecker will
        not be able to determine with certainty if the repo is out of date.

        Freeze at the commit hash (sha)::
            git+git://github.com/django/django.git@393c268e725f5b229ecb554f3fac02cfc250d2df#egg=Django
            https://github.com/django/django/archive/393c268e725f5b229ecb554f3fac02cfc250d2df.tar.gz#egg=Django
            https://github.com/django/django/archive/393c268e725f5b229ecb554f3fac02cfc250d2df.zip#egg=Django

        Freeze with a branch name::
            git+git://github.com/django/django.git@master#egg=Django
            https://github.com/django/django/archive/master.tar.gz#egg=Django
            https://github.com/django/django/archive/master.zip#egg=Django

        Freeze with a tag::
            git+git://github.com/django/django.git@1.5b2#egg=Django
            https://github.com/django/django/archive/1.5b2.tar.gz#egg=Django
            https://github.com/django/django/archive/1.5b2.zip#egg=Django

        Do not freeze::
            git+git://github.com/django/django.git#egg=Django

        """
        for name, req in list(self.reqs.items()):
            req_url = req["url"]
            if not req_url:
                continue
            req_url = str(req_url)
            if req_url.startswith("git") and "github.com/" not in req_url:
                continue
            if req_url.endswith((".tar.gz", ".tar.bz2", ".zip")):
                continue

            headers = {
                "content-type": "application/json",
            }
            if self.github_api_token:
                headers["Authorization"] = "token {0}".format(self.github_api_token)
            try:
                path_parts = urlparse(req_url).path.split("#", 1)[0].strip("/").rstrip("/").split("/")

                if len(path_parts) == 2:
                    user, repo = path_parts

                elif 'archive' in path_parts:
                    # Supports URL of format:
                    # https://github.com/django/django/archive/master.tar.gz#egg=Django
                    # https://github.com/django/django/archive/master.zip#egg=Django
                    user, repo = path_parts[:2]
                    repo += '@' + path_parts[-1].replace('.tar.gz', '').replace('.zip', '')

                else:
                    self.style.ERROR("\nFailed to parse %r\n" % (req_url, ))
                    continue
            except (ValueError, IndexError) as e:
                self.stdout.write(self.style.ERROR("\nFailed to parse %r: %s\n" % (req_url, e)))
                continue

            try:
                test_auth = requests.get("https://api.github.com/django/", headers=headers).json()
            except HTTPError as e:
                self.stdout.write("\n%s\n" % str(e))
                return

            if "message" in test_auth and test_auth["message"] == "Bad credentials":
                self.stdout.write(self.style.ERROR("\nGithub API: Bad credentials. Aborting!\n"))
                return
            elif "message" in test_auth and test_auth["message"].startswith("API Rate Limit Exceeded"):
                self.stdout.write(self.style.ERROR("\nGithub API: Rate Limit Exceeded. Aborting!\n"))
                return

            frozen_commit_sha = None
            if ".git" in repo:
                repo_name, frozen_commit_full = repo.split(".git")
                if frozen_commit_full.startswith("@"):
                    frozen_commit_sha = frozen_commit_full[1:]
            elif "@" in repo:
                repo_name, frozen_commit_sha = repo.split("@")

            if frozen_commit_sha is None:
                msg = self.style.ERROR("repo is not frozen")

            if frozen_commit_sha:
                branch_url = "https://api.github.com/repos/{0}/{1}/branches".format(user, repo_name)
                branch_data = requests.get(branch_url, headers=headers).json()

                frozen_commit_url = "https://api.github.com/repos/{0}/{1}/commits/{2}".format(
                    user, repo_name, frozen_commit_sha
                )
                frozen_commit_data = requests.get(frozen_commit_url, headers=headers).json()

                if "message" in frozen_commit_data and frozen_commit_data["message"] == "Not Found":
                    msg = self.style.ERROR("{0} not found in {1}. Repo may be private.".format(frozen_commit_sha[:10], name))
                elif frozen_commit_data["sha"] in [branch["commit"]["sha"] for branch in branch_data]:
                    msg = self.style.BOLD("up to date")
                else:
                    msg = self.style.INFO("{0} is not the head of any branch".format(frozen_commit_data["sha"][:10]))

            if "dist" in req:
                pkg_info = "{dist.project_name} {dist.version}".format(dist=req["dist"])
            elif frozen_commit_sha is None:
                pkg_info = name
            else:
                pkg_info = "{0} {1}".format(name, frozen_commit_sha[:10])
            self.stdout.write("{pkg_info:40} {msg}".format(pkg_info=pkg_info, msg=msg))
            del self.reqs[name]

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

def is_list_of_ipachars(obj):
    """
    Return ``True`` if the given object is a list of IPAChar objects.

    :param object obj: the object to test
    :rtype: bool
    """
    if isinstance(obj, list):
        for e in obj:
            if not isinstance(e, IPAChar):
                return False
        return True
    return False

def register_blueprints(app):
    """Register Flask blueprints."""
    app.register_blueprint(public.public_bp)
    app.register_blueprint(genes.genes_bp)
    app.register_blueprint(cases.cases_bp)
    app.register_blueprint(login.login_bp)
    app.register_blueprint(variants.variants_bp)
    app.register_blueprint(panels.panels_bp)
    app.register_blueprint(dashboard.dashboard_bp)
    app.register_blueprint(api.api_bp)
    app.register_blueprint(alignviewers.alignviewers_bp)
    app.register_blueprint(phenotypes.hpo_bp)
    app.register_blueprint(institutes.overview)

def _converter(self, value):
        """Convert raw input value of the field."""
        if not isinstance(value, datetime.date):
            raise TypeError('{0} is not valid date'.format(value))
        return value

def check_max_filesize(chosen_file, max_size):
    """
    Checks file sizes for host
    """
    if os.path.getsize(chosen_file) > max_size:
        return False
    else:
        return True

def gen_text(env: TextIOBase, package: str, tmpl: str):
    """Create output from Jinja template."""
    if env:
        env_args = json_datetime.load(env)
    else:
        env_args = {}
    jinja_env = template.setup(package)
    echo(jinja_env.get_template(tmpl).render(**env_args))

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def decode_mysql_string_literal(text):
    """
    Removes quotes and decodes escape sequences from given MySQL string literal
    returning the result.

    :param text: MySQL string literal, with the quotes still included.
    :type text: str

    :return: Given string literal with quotes removed and escape sequences
             decoded.
    :rtype: str
    """
    assert text.startswith("'")
    assert text.endswith("'")

    # Ditch quotes from the string literal.
    text = text[1:-1]

    return MYSQL_STRING_ESCAPE_SEQUENCE_PATTERN.sub(
        unescape_single_character,
        text,
    )

def get_long_description():
    """ Read the long description. """
    here = path.abspath(path.dirname(__file__))
    with open(path.join(here, 'README.rst')) as readme:
        return readme.read()
    return None

def is_filelike(ob):
    """Check for filelikeness of an object.

    Needed to distinguish it from file names.
    Returns true if it has a read or a write method.
    """
    if hasattr(ob, 'read') and callable(ob.read):
        return True

    if hasattr(ob, 'write') and callable(ob.write):
        return True

    return False

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

def __add__(self,other):
        """
            If the number of columns matches, we can concatenate two LabeldMatrices
            with the + operator.
        """
        assert self.matrix.shape[1] == other.matrix.shape[1]
        return LabeledMatrix(np.concatenate([self.matrix,other.matrix],axis=0),self.labels)

def parse_s3_url(url):
    """
    Parses S3 URL.

    Returns bucket (domain) and file (full path).
    """
    bucket = ''
    path = ''
    if url:
        result = urlparse(url)
        bucket = result.netloc
        path = result.path.strip('/')
    return bucket, path

def prefix_list(self, prefix, values):
        """
        Add a prefix to a list of values.
        """
        return list(map(lambda value: prefix + " " + value, values))

def is_break_tag(self, el):
        """Check if tag is an element we should break on."""

        name = el.name
        return name in self.break_tags or name in self.user_break_tags

def go_to_new_line(self):
        """Go to the end of the current line and create a new line"""
        self.stdkey_end(False, False)
        self.insert_text(self.get_line_separator())

def SegmentMin(a, ids):
    """
    Segmented min op.
    """
    func = lambda idxs: np.amin(a[idxs], axis=0)
    return seg_map(func, a, ids),

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

def is_valid_file(parser,arg):
	"""verify the validity of the given file. Never trust the End-User"""
	if not os.path.exists(arg):
       		parser.error("File %s not found"%arg)
	else:
	       	return arg

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

def get_img_data(f, maxsize = (1200, 850), first = False):
    """Generate image data using PIL
    """
    img = Image.open(f)
    img.thumbnail(maxsize)
    if first:                     # tkinter is inactive the first time
        bio = io.BytesIO()
        img.save(bio, format = "PNG")
        del img
        return bio.getvalue()
    return ImageTk.PhotoImage(img)

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

def eval_script(self, expr):
    """ Evaluates a piece of Javascript in the context of the current page and
    returns its value. """
    ret = self.conn.issue_command("Evaluate", expr)
    return json.loads("[%s]" % ret)[0]

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

def setAutoRangeOn(self, axisNumber):
        """ Sets the auto-range of the axis on.

            :param axisNumber: 0 (X-axis), 1 (Y-axis), 2, (Both X and Y axes).
        """
        setXYAxesAutoRangeOn(self, self.xAxisRangeCti, self.yAxisRangeCti, axisNumber)

def asin(x):
    """
    Inverse sine
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.arcsin(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.arcsin(x)

def date_to_datetime(x):
    """Convert a date into a datetime"""
    if not isinstance(x, datetime) and isinstance(x, date):
        return datetime.combine(x, time())
    return x

def read_full(stream):
    """Read the full contents of the given stream into memory.

    :return:
        A future containing the complete stream contents.
    """
    assert stream, "stream is required"

    chunks = []
    chunk = yield stream.read()

    while chunk:
        chunks.append(chunk)
        chunk = yield stream.read()

    raise tornado.gen.Return(b''.join(chunks))

def locked_delete(self):
        """Delete credentials from the SQLAlchemy datastore."""
        filters = {self.key_name: self.key_value}
        self.session.query(self.model_class).filter_by(**filters).delete()

def get_parent_dir(name):
    """Get the parent directory of a filename."""
    parent_dir = os.path.dirname(os.path.dirname(name))
    if parent_dir:
        return parent_dir
    return os.path.abspath('.')

def parent_widget(self):
        """ Reimplemented to only return GraphicsItems """
        parent = self.parent()
        if parent is not None and isinstance(parent, QtGraphicsItem):
            return parent.widget

def assert_called_once(_mock_self):
        """assert that the mock was called only once.
        """
        self = _mock_self
        if not self.call_count == 1:
            msg = ("Expected '%s' to have been called once. Called %s times." %
                   (self._mock_name or 'mock', self.call_count))
            raise AssertionError(msg)

def camelcase_to_slash(name):
    """ Converts CamelCase to camel/case

    code ripped from http://stackoverflow.com/questions/1175208/does-the-python-standard-library-have-function-to-convert-camelcase-to-camel-cas
    """

    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1/\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1/\2', s1).lower()

def current_timestamp():
    """Returns current time as ISO8601 formatted string in the Zulu TZ"""
    now = datetime.utcnow()
    timestamp = now.isoformat()[0:19] + 'Z'

    debug("generated timestamp: {now}".format(now=timestamp))

    return timestamp

def strip_accents(string):
    """
    Strip all the accents from the string
    """
    return u''.join(
        (character for character in unicodedata.normalize('NFD', string)
         if unicodedata.category(character) != 'Mn'))

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

def add_url_rule(self, route, endpoint, handler):
        """Add a new url route.

        Args:
            See flask.Flask.add_url_route().
        """
        self.app.add_url_rule(route, endpoint, handler)

def stringToDate(fmt="%Y-%m-%d"):
    """returns a function to convert a string to a datetime.date instance
    using the formatting string fmt as in time.strftime"""
    import time
    import datetime
    def conv_func(s):
        return datetime.date(*time.strptime(s,fmt)[:3])
    return conv_func

def _get_type(self, value):
        """Get the data type for *value*."""
        if value is None:
            return type(None)
        elif type(value) in int_types:
            return int
        elif type(value) in float_types:
            return float
        elif isinstance(value, binary_type):
            return binary_type
        else:
            return text_type

def average(arr):
  """average of the values, must have more than 0 entries.

  :param arr: list of numbers
  :type arr: number[] a number array
  :return: average
  :rtype: float

  """
  if len(arr) == 0:
    sys.stderr.write("ERROR: no content in array to take average\n")
    sys.exit()
  if len(arr) == 1:  return arr[0]
  return float(sum(arr))/float(len(arr))

def versions_request(self):
        """List Available REST API Versions"""
        ret = self.handle_api_exceptions('GET', '', api_ver='')
        return [str_dict(x) for x in ret.json()]

def count(data, axis=None):
    """Count the number of non-NA in this array along the given axis or axes
    """
    return np.sum(np.logical_not(isnull(data)), axis=axis)

def create_search_url(self):
        """ Generates (urlencoded) query string from stored key-values tuples

        :returns: A string containing all arguments in a url-encoded format
        """

        url = '?'
        for key, value in self.arguments.items():
            url += '%s=%s&' % (quote_plus(key), quote_plus(value))
        self.url = url[:-1]
        return self.url

def _decode_request(self, encoded_request):
        """Decode an request previously encoded"""
        obj = self.serializer.loads(encoded_request)
        return request_from_dict(obj, self.spider)

def info(docgraph):
    """print node and edge statistics of a document graph"""
    print networkx.info(docgraph), '\n'
    node_statistics(docgraph)
    print
    edge_statistics(docgraph)

def magic(self, alias):
        """Returns the appropriate IPython code magic when
        called with an alias for a language.
        """
        if alias in self.aliases:
            return self.aliases[alias]
        else:
            return "%%{}\n".format(alias)

def do_serial(self, p):
		"""Set the serial port, e.g.: /dev/tty.usbserial-A4001ib8"""
		try:
			self.serial.port = p
			self.serial.open()
			print 'Opening serial port: %s' % p
		except Exception, e:
			print 'Unable to open serial port: %s' % p

def ReverseV2(a, axes):
    """
    Reverse op.
    """
    idxs = tuple(slice(None, None, 2 * int(i not in axes) - 1) for i in range(len(a.shape)))
    return np.copy(a[idxs]),

def save(self):
        """Saves the updated model to the current entity db.
        """
        self.session.add(self)
        self.session.flush()
        return self

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

def __mul__(self, other):
        """Handle the `*` operator."""
        return self._handle_type(other)(self.value * other.value)

def dimensions(self):
        """Get width and height of a PDF"""
        size = self.pdf.getPage(0).mediaBox
        return {'w': float(size[2]), 'h': float(size[3])}

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

def contains(self, element):
        """
        Ensures :attr:`subject` contains *other*.
        """
        self._run(unittest_case.assertIn, (element, self._subject))
        return ChainInspector(self._subject)

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

def http(self, *args, **kwargs):
        """Starts the process of building a new HTTP route linked to this API instance"""
        kwargs['api'] = self.api
        return http(*args, **kwargs)

def as_float_array(a):
    """View the quaternion array as an array of floats

    This function is fast (of order 1 microsecond) because no data is
    copied; the returned quantity is just a "view" of the original.

    The output view has one more dimension (of size 4) than the input
    array, but is otherwise the same shape.

    """
    return np.asarray(a, dtype=np.quaternion).view((np.double, 4))

def get_month_namedays(self, month=None):
        """Return names as a tuple based on given month.
        If no month given, use current one"""
        if month is None:
            month = datetime.now().month
        return self.NAMEDAYS[month-1]

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

def uppercase_chars(string: any) -> str:
        """Return all (and only) the uppercase chars in the given string."""
        return ''.join([c if c.isupper() else '' for c in str(string)])

def most_significant_bit(lst: np.ndarray) -> int:
    """
    A helper function that finds the position of the most significant bit in a 1darray of 1s and 0s,
    i.e. the first position where a 1 appears, reading left to right.

    :param lst: a 1d array of 0s and 1s with at least one 1
    :return: the first position in lst that a 1 appears
    """
    return np.argwhere(np.asarray(lst) == 1)[0][0]

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

def convert_column(self, values):
        """Normalize values."""
        assert all(values >= 0), 'Cannot normalize a column with negatives'
        total = sum(values)
        if total > 0:
            return values / total
        else:
            return values

def logout():
    """ Log out the active user
    """
    flogin.logout_user()
    next = flask.request.args.get('next')
    return flask.redirect(next or flask.url_for("user"))

def assert_visible(self, locator, msg=None):
        """
        Hard assert for whether and element is present and visible in the current window/frame

        :params locator: the locator of the element to search for
        :params msg: (Optional) msg explaining the difference
        """
        e = driver.find_elements_by_locator(locator)
        if len(e) == 0:
            raise AssertionError("Element at %s was not found" % locator)
        assert e.is_displayed()

def _begins_with_one_of(sentence, parts_of_speech):
    """Return True if the sentence or fragment begins with one of the parts of
    speech in the list, else False"""
    doc = nlp(sentence)
    if doc[0].tag_ in parts_of_speech:
        return True
    return False

def addfield(self, pkt, buf, val):
        """add the field with endianness to the buffer"""
        self.set_endianess(pkt)
        return self.fld.addfield(pkt, buf, val)

def INIT_LIST_EXPR(self, cursor):
        """Returns a list of literal values."""
        values = [self.parse_cursor(child)
                  for child in list(cursor.get_children())]
        return values

def get_qualified_name(_object):
    """Return the Fully Qualified Name from an instance or class."""
    module = _object.__module__
    if hasattr(_object, '__name__'):
        _class = _object.__name__

    else:
        _class = _object.__class__.__name__

    return module + '.' + _class

def register_view(self, view):
        """Register callbacks for button press events and selection changed"""
        super(ListViewController, self).register_view(view)
        self.tree_view.connect('button_press_event', self.mouse_click)

def mtf_image_transformer_cifar_mp_4x():
  """Data parallel CIFAR parameters."""
  hparams = mtf_image_transformer_base_cifar()
  hparams.mesh_shape = "model:4;batch:8"
  hparams.layout = "batch:batch;d_ff:model;heads:model"
  hparams.batch_size = 32
  hparams.num_heads = 8
  hparams.d_ff = 8192
  return hparams

def unique_iter(seq):
    """
    See http://www.peterbe.com/plog/uniqifiers-benchmark
    Originally f8 written by Dave Kirby
    """
    seen = set()
    return [x for x in seq if x not in seen and not seen.add(x)]

def sorted_by(key: Callable[[raw_types.Qid], Any]) -> 'QubitOrder':
        """A basis that orders qubits ascending based on a key function.

        Args:
            key: A function that takes a qubit and returns a key value. The
                basis will be ordered ascending according to these key values.


        Returns:
            A basis that orders qubits ascending based on a key function.
        """
        return QubitOrder(lambda qubits: tuple(sorted(qubits, key=key)))

def _escape(self, s):
        """Escape bad characters for regular expressions.

        Similar to `re.escape` but allows '%' to pass through.

        """
        for ch, r_ch in self.ESCAPE_SETS:
            s = s.replace(ch, r_ch)
        return s

def is_archlinux():
    """return True if the current distribution is running on debian like OS."""
    if platform.system().lower() == 'linux':
        if platform.linux_distribution() == ('', '', ''):
            # undefined distribution. Fixed in python 3.
            if os.path.exists('/etc/arch-release'):
                return True
    return False

def pass_from_pipe(cls):
        """Return password from pipe if not on TTY, else False.
        """
        is_pipe = not sys.stdin.isatty()
        return is_pipe and cls.strip_last_newline(sys.stdin.read())

def heappush_max(heap, item):
    """Push item onto heap, maintaining the heap invariant."""
    heap.append(item)
    _siftdown_max(heap, 0, len(heap) - 1)

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

def confirm(question, default=True):
    """Ask a yes/no question interactively.

    :param question: The text of the question to ask.
    :returns: True if the answer was "yes", False otherwise.
    """
    valid = {"": default, "yes": True, "y": True, "no": False, "n": False}
    while 1:
        choice = input(question + (" [Y/n] " if default else " [y/N] ")).lower()
        if choice in valid:
            return valid[choice]
        print("Please respond with 'y' or 'n' ")

def _add_indent(string, indent):
    """Add indent of ``indent`` spaces to ``string.split("\n")[1:]``

    Useful for formatting in strings to already indented blocks
    """
    lines = string.split("\n")
    first, lines = lines[0], lines[1:]
    lines = ["{indent}{s}".format(indent=" " * indent, s=s)
             for s in lines]
    lines = [first] + lines
    return "\n".join(lines)

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

def filter_bolts(table, header):
  """ filter to keep bolts """
  bolts_info = []
  for row in table:
    if row[0] == 'bolt':
      bolts_info.append(row)
  return bolts_info, header

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

def __or__(self, other):
        """Return the union of two RangeSets as a new RangeSet.

        (I.e. all elements that are in either set.)
        """
        if not isinstance(other, set):
            return NotImplemented
        return self.union(other)

def download_sdk(url):
    """Downloads the SDK and returns a file-like object for the zip content."""
    r = requests.get(url)
    r.raise_for_status()
    return StringIO(r.content)

async def write_register(self, address, value, skip_encode=False):
        """Write a modbus register."""
        await self._request('write_registers', address, value, skip_encode=skip_encode)

def has_enumerated_namespace_name(self, namespace: str, name: str) -> bool:
        """Check that the namespace is defined by an enumeration and that the name is a member."""
        return self.has_enumerated_namespace(namespace) and name in self.namespace_to_terms[namespace]

def rollback(self):
		"""
		Rollback MySQL Transaction to database.
		MySQLDB: If the database and tables support transactions, this rolls 
		back (cancels) the current transaction; otherwise a 
		NotSupportedError is raised.
		
		@author: Nick Verbeck
		@since: 5/12/2008
		"""
		try:
			if self.connection is not None:
				self.connection.rollback()
				self._updateCheckTime()
				self.release()
		except Exception, e:
			pass

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

def progressbar(total, pos, msg=""):
    """
    Given a total and a progress position, output a progress bar
    to stderr. It is important to not output anything else while
    using this, as it relies soley on the behavior of carriage
    return (\\r).

    Can also take an optioal message to add after the
    progressbar. It must not contain newlines.

    The progress bar will look something like this:

    [099/500][=========...............................] ETA: 13:36:59

    Of course, the ETA part should be supplied be the calling
    function.
    """
    width = get_terminal_size()[0] - 40
    rel_pos = int(float(pos) / total * width)
    bar = ''.join(["=" * rel_pos, "." * (width - rel_pos)])

    # Determine how many digits in total (base 10)
    digits_total = len(str(total))
    fmt_width = "%0" + str(digits_total) + "d"
    fmt = "\r[" + fmt_width + "/" + fmt_width + "][%s] %s"

    progress_stream.write(fmt % (pos, total, bar, msg))

def health_check(self):
        """Uses head object to make sure the file exists in S3."""
        logger.debug('Health Check on S3 file for: {namespace}'.format(
            namespace=self.namespace
        ))

        try:
            self.client.head_object(Bucket=self.bucket_name, Key=self.data_file)
            return True
        except ClientError as e:
            logger.debug('Error encountered with S3.  Assume unhealthy')

def run(self, *args, **kwargs):
        """ Connect and run bot in event loop. """
        self.eventloop.run_until_complete(self.connect(*args, **kwargs))
        try:
            self.eventloop.run_forever()
        finally:
            self.eventloop.stop()

def auto_update(cls, function):
        """
        This class method could be used as decorator on subclasses, it ensures
        update method is called after function execution.
        """

        def wrapper(self, *args, **kwargs):
            f = function(self, *args, **kwargs)
            self.update()
            return f
        return wrapper

def get_combined_size(tiles):
    """Calculate combined size of tiles."""
    # TODO: Refactor calculating layout to avoid repetition.
    columns, rows = calc_columns_rows(len(tiles))
    tile_size = tiles[0].image.size
    return (tile_size[0] * columns, tile_size[1] * rows)

def to_bytes_or_none(value):
    """Converts C char arrays to bytes and C NULL values to None."""
    if value == ffi.NULL:
        return None
    elif isinstance(value, ffi.CData):
        return ffi.string(value)
    else:
        raise ValueError('Value must be char[] or NULL')

async def power(source, exponent):
    """Raise the elements of an asynchronous sequence to the given power."""
    async with streamcontext(source) as streamer:
        async for item in streamer:
            yield item ** exponent

def data_directory():
    """Return the absolute path to the directory containing the package data."""
    package_directory = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(package_directory, "data")

def ip_address_list(ips):
    """ IP address range validation and expansion. """
    # first, try it as a single IP address
    try:
        return ip_address(ips)
    except ValueError:
        pass
    # then, consider it as an ipaddress.IPv[4|6]Network instance and expand it
    return list(ipaddress.ip_network(u(ips)).hosts())

def tokenize_words(self, text):
        """Tokenize an input string into a list of words (with punctuation removed)."""
        return [
            self.strip_punctuation(word) for word in text.split(' ')
            if self.strip_punctuation(word)
        ]

def remove_item(self, item):
        """
        Remove (and un-index) an object

        :param item: object to remove
        :type item: alignak.objects.item.Item
        :return: None
        """
        self.unindex_item(item)
        self.items.pop(item.uuid, None)

def wipe(self):
        """ Wipe the store
        """
        keys = list(self.keys()).copy()
        for key in keys:
            self.delete(key)

def submit_form_id(step, id):
    """
    Submit the form having given id.
    """
    form = world.browser.find_element_by_xpath(str('id("{id}")'.format(id=id)))
    form.submit()

def pad_hex(value, bit_size):
    """
    Pads a hex string up to the given bit_size
    """
    value = remove_0x_prefix(value)
    return add_0x_prefix(value.zfill(int(bit_size / 4)))

def functions(self):
        """
        A list of functions declared or defined in this module.
        """
        return [v for v in self.globals.values()
                if isinstance(v, values.Function)]

def wipe_table(self, table: str) -> int:
        """Delete all records from a table. Use caution!"""
        sql = "DELETE FROM " + self.delimit(table)
        return self.db_exec(sql)

def _dict_values_sorted_by_key(dictionary):
    # This should be a yield from instead.
    """Internal helper to return the values of a dictionary, sorted by key.
    """
    for _, value in sorted(dictionary.iteritems(), key=operator.itemgetter(0)):
        yield value

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

def reset_namespace(self):
        """Resets the namespace by removing all names defined by the user"""
        self.shellwidget.reset_namespace(warning=self.reset_warning,
                                         message=True)

def validate_multiindex(self, obj):
        """validate that we can store the multi-index; reset and return the
        new object
        """
        levels = [l if l is not None else "level_{0}".format(i)
                  for i, l in enumerate(obj.index.names)]
        try:
            return obj.reset_index(), levels
        except ValueError:
            raise ValueError("duplicate names/columns in the multi-index when "
                             "storing as a table")

def ensure_dir_exists(directory):
    """Se asegura de que un directorio exista."""
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def euler(self):
        """TODO DEPRECATE THIS?"""
        e_xyz = transformations.euler_from_matrix(self.rotation, 'sxyz')
        return np.array([180.0 / np.pi * a for a in e_xyz])

def get_matrix(self):
        """  Use numpy to create a real matrix object from the data

        :return: the matrix representation of the fvm
        """
        return np.array([ self.get_row_list(i) for i in range(self.row_count()) ])

def strip_comment_marker(text):
    """ Strip # markers at the front of a block of comment text.
    """
    lines = []
    for line in text.splitlines():
        lines.append(line.lstrip('#'))
    text = textwrap.dedent('\n'.join(lines))
    return text

def StringIO(*args, **kwargs):
    """StringIO constructor shim for the async wrapper."""
    raw = sync_io.StringIO(*args, **kwargs)
    return AsyncStringIOWrapper(raw)

def cint32_array_to_numpy(cptr, length):
    """Convert a ctypes int pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_int32)):
        return np.fromiter(cptr, dtype=np.int32, count=length)
    else:
        raise RuntimeError('Expected int pointer')

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

def get_tile_location(self, x, y):
        """Get the screen coordinate for the top-left corner of a tile."""
        x1, y1 = self.origin
        x1 += self.BORDER + (self.BORDER + self.cell_width) * x
        y1 += self.BORDER + (self.BORDER + self.cell_height) * y
        return x1, y1

def allZero(buffer):
    """
    Tries to determine if a buffer is empty.
    
    @type buffer: str
    @param buffer: Buffer to test if it is empty.
        
    @rtype: bool
    @return: C{True} if the given buffer is empty, i.e. full of zeros,
        C{False} if it doesn't.
    """
    allZero = True
    for byte in buffer:
        if byte != "\x00":
            allZero = False
            break
    return allZero

def calculate_month(birth_date):
    """
    Calculates and returns a month number basing on PESEL standard.
    """
    year = int(birth_date.strftime('%Y'))
    month = int(birth_date.strftime('%m')) + ((int(year / 100) - 14) % 5) * 20

    return month

def coverage():
    """check code coverage quickly with the default Python"""
    run("coverage run --source {PROJECT_NAME} -m py.test".format(PROJECT_NAME=PROJECT_NAME))
    run("coverage report -m")
    run("coverage html")

    webbrowser.open('file://' + os.path.realpath("htmlcov/index.html"), new=2)

def packagenameify(s):
  """
  Makes a package name
  """
  return ''.join(w if w in ACRONYMS else w.title() for w in s.split('.')[-1:])

def cleanwrap(func):
    """ Wrapper for Zotero._cleanup
    """

    def enc(self, *args, **kwargs):
        """ Send each item to _cleanup() """
        return (func(self, item, **kwargs) for item in args)

    return enc

def convert_time_string(date_str):
    """ Change a date string from the format 2018-08-15T23:55:17 into a datetime object """
    dt, _, _ = date_str.partition(".")
    dt = datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S")
    return dt

def sets_are_rooted_compat(one_set, other):
    """treats the 2 sets are sets of taxon IDs on the same (unstated)
    universe of taxon ids.
    Returns True clades implied by each are compatible and False otherwise
    """
    if one_set.issubset(other) or other.issubset(one_set):
        return True
    return not intersection_not_empty(one_set, other)

def tob(data, enc='utf8'):
    """ Convert anything to bytes """
    return data.encode(enc) if isinstance(data, six.text_type) else bytes(data)

def best(self):
        """
        Returns the element with the highest probability.
        """
        b = (-1e999999, None)
        for k, c in iteritems(self.counts):
            b = max(b, (c, k))
        return b[1]

def constant(times: np.ndarray, amp: complex) -> np.ndarray:
    """Continuous constant pulse.

    Args:
        times: Times to output pulse for.
        amp: Complex pulse amplitude.
    """
    return np.full(len(times), amp, dtype=np.complex_)

def _file_type(self, field):
        """ Returns file type for given file field.
        
        Args:
            field (str): File field

        Returns:
            string. File type
        """
        type = mimetypes.guess_type(self._files[field])[0]
        return type.encode("utf-8") if isinstance(type, unicode) else str(type)

def touch():
    """Create new bucket."""
    from .models import Bucket
    bucket = Bucket.create()
    db.session.commit()
    click.secho(str(bucket), fg='green')

def load_tiff(file):
    """
    Load a geotiff raster keeping ndv values using a masked array

    Usage:
            data = load_tiff(file)
    """
    ndv, xsize, ysize, geot, projection, datatype = get_geo_info(file)
    data = gdalnumeric.LoadFile(file)
    data = np.ma.masked_array(data, mask=data == ndv, fill_value=ndv)
    return data

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

def column_names(self, table):
      """An iterable of column names, for a particular table or
      view."""

      table_info = self.execute(
        u'PRAGMA table_info(%s)' % quote(table))
      return (column['name'] for column in table_info)

def normalize(data):
    """Normalize the data to be in the [0, 1] range.

    :param data:
    :return: normalized data
    """
    out_data = data.copy()

    for i, sample in enumerate(out_data):
        out_data[i] /= sum(out_data[i])

    return out_data

def create_movie(fig, update_figure, filename, title, fps=15, dpi=100):
    """Helps us to create a movie."""
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata     = dict(title=title)
    writer       = FFMpegWriter(fps=fps, metadata=metadata)

    with writer.saving(fig, filename, dpi):
        t = 0
        while True:
            if update_figure(t):
                writer.grab_frame()
                t += 1
            else:
                break

def zfill(x, width):
    """zfill(x, width) -> string

    Pad a numeric string x with zeros on the left, to fill a field
    of the specified width.  The string x is never truncated.

    """
    if not isinstance(x, basestring):
        x = repr(x)
    return x.zfill(width)

def join(self):
		"""Note that the Executor must be close()'d elsewhere,
		or join() will never return.
		"""
		self.inputfeeder_thread.join()
		self.pool.join()
		self.resulttracker_thread.join()
		self.failuretracker_thread.join()

def __hash__(self):
        """Return ``hash(self)``."""
        return hash((type(self), self.domain, self.range, self.partition))

def lock(self, block=True):
		"""
		Lock connection from being used else where
		"""
		self._locked = True
		return self._lock.acquire(block)

def all_collections(db):
	"""
	Yield all non-sytem collections in db.
	"""
	include_pattern = r'(?!system\.)'
	return (
		db[name]
		for name in db.list_collection_names()
		if re.match(include_pattern, name)
	)

def datetime_to_year_quarter(dt):
    """
    Args:
        dt: a datetime
    Returns:
        tuple of the datetime's year and quarter
    """
    year = dt.year
    quarter = int(math.ceil(float(dt.month)/3))
    return (year, quarter)

def _get_current_label(self):
        """Get the label from the last line read"""
        if len(self._last) == 0:
            raise StopIteration
        return self._last[:self._last.find(":")]

def __exit__(self, *exc):
        """Exit the runtime context. This will end the transaction."""
        if exc[0] is None and exc[1] is None and exc[2] is None:
            self.commit()
        else:
            self.rollback()

def generate_yaml_file(filename, contents):
    """Creates a yaml file with the given content."""
    with open(filename, 'w') as file:
        file.write(yaml.dump(contents, default_flow_style=False))

def find_console_handler(logger):
    """Return a stream handler, if it exists."""
    for handler in logger.handlers:
        if (isinstance(handler, logging.StreamHandler) and
                handler.stream == sys.stderr):
            return handler

def __iter__(self):
		"""Iterate through all elements.

		Multiple copies will be returned if they exist.
		"""
		for value, count in self.counts():
			for _ in range(count):
				yield value

def difference(ydata1, ydata2):
    """

    Returns the number you should add to ydata1 to make it line up with ydata2

    """

    y1 = _n.array(ydata1)
    y2 = _n.array(ydata2)

    return(sum(y2-y1)/len(ydata1))

def make_qs(n, m=None):
    """Make sympy symbols q0, q1, ...
    
    Args:
        n(int), m(int, optional):
            If specified both n and m, returns [qn, q(n+1), ..., qm],
            Only n is specified, returns[q0, q1, ..., qn].

    Return:
        tuple(Symbol): Tuple of sympy symbols.
    """
    try:
        import sympy
    except ImportError:
        raise ImportError("This function requires sympy. Please install it.")
    if m is None:
        syms = sympy.symbols(" ".join(f"q{i}" for i in range(n)))
        if isinstance(syms, tuple):
            return syms
        else:
            return (syms,)
    syms = sympy.symbols(" ".join(f"q{i}" for i in range(n, m)))
    if isinstance(syms, tuple):
        return syms
    else:
        return (syms,)

def stop_logging():
    """Stop logging to logfile and console."""
    from . import log
    logger = logging.getLogger("gromacs")
    logger.info("GromacsWrapper %s STOPPED logging", get_version())
    log.clear_handlers(logger)

def _id(self):
        """What this object is equal to."""
        return (self.__class__, self.number_of_needles, self.needle_positions,
                self.left_end_needle)

def layout(self, indent='    '):
		"""This will indent each new tag in the body by given number of spaces."""


		self.__indent(self.head, indent)
		self.__indent(self.meta, indent)
		self.__indent(self.stylesheet, indent)
		self.__indent(self.header, indent)
		self.__indent(self.body, indent, initial=3)
		self.__indent(self.footer, indent)
		self.__indent(self.body_pre_docinfo, indent, initial=3)
		self.__indent(self.docinfo, indent)

def inverted_dict(d):
    """Return a dict with swapped keys and values

    >>> inverted_dict({0: ('a', 'b'), 1: 'cd'}) == {'cd': 1, ('a', 'b'): 0}
    True
    """
    return dict((force_hashable(v), k) for (k, v) in viewitems(dict(d)))

def set_if_empty(self, param, default):
        """ Set the parameter to the default if it doesn't exist """
        if not self.has(param):
            self.set(param, default)

def show_guestbook():
    """Returns all existing guestbook records."""
    cursor = flask.g.db.execute(
        'SELECT name, message FROM entry ORDER BY id DESC;')
    entries = [{'name': row[0], 'message': row[1]} for row in cursor.fetchall()]
    return jinja2.Template(LAYOUT).render(entries=entries)

def is_empty_object(n, last):
    """n may be the inside of block or object"""
    if n.strip():
        return False
    # seems to be but can be empty code
    last = last.strip()
    markers = {
        ')',
        ';',
    }
    if not last or last[-1] in markers:
        return False
    return True

def print(cls, *args, **kwargs):
        """Print synchronized."""
        # pylint: disable=protected-access
        with _shared._PRINT_LOCK:
            print(*args, **kwargs)
            _sys.stdout.flush()

def sanitize_word(s):
    """Remove non-alphanumerical characters from metric word.
    And trim excessive underscores.
    """
    s = re.sub('[^\w-]+', '_', s)
    s = re.sub('__+', '_', s)
    return s.strip('_')

def average_gradient(data, *kwargs):
    """ Compute average gradient norm of an image
    """
    return np.average(np.array(np.gradient(data))**2)

def surface(cls, predstr):
        """Instantiate a Pred from its quoted string representation."""
        lemma, pos, sense, _ = split_pred_string(predstr)
        return cls(Pred.SURFACE, lemma, pos, sense, predstr)

def torecarray(*args, **kwargs):
    """
    Convenient shorthand for ``toarray(*args, **kwargs).view(np.recarray)``.

    """

    import numpy as np
    return toarray(*args, **kwargs).view(np.recarray)

def json_template(data, template_name, template_context):
    """Old style, use JSONTemplateResponse instead of this.
    """
    html = render_to_string(template_name, template_context)
    data = data or {}
    data['html'] = html
    return HttpResponse(json_encode(data), content_type='application/json')

def to_json(df, state_index, color_index, fills):
        """Transforms dataframe to json response"""
        records = {}
        for i, row in df.iterrows():

            records[row[state_index]] = {
                "fillKey": row[color_index]
            }

        return {
            "data": records,
            "fills": fills
        }

def place(self):
        """Place this container's canvas onto the parent container's canvas."""
        self.place_children()
        self.canvas.append(self.parent.canvas,
                           float(self.left), float(self.top))

def info(self, message, *args, **kwargs):
        """More important level : default for print and save
        """
        self._log(logging.INFO, message, *args, **kwargs)

async def disconnect(self):
        """ Disconnect from target. """
        if not self.connected:
            return

        self.writer.close()
        self.reader = None
        self.writer = None

def cell_ends_with_code(lines):
    """Is the last line of the cell a line with code?"""
    if not lines:
        return False
    if not lines[-1].strip():
        return False
    if lines[-1].startswith('#'):
        return False
    return True

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

def dates_in_range(start_date, end_date):
    """Returns all dates between two dates.

    Inclusive of the start date but not the end date.

    Args:
        start_date (datetime.date)
        end_date (datetime.date)

    Returns:
        (list) of datetime.date objects
    """
    return [
        start_date + timedelta(n)
        for n in range(int((end_date - start_date).days))
    ]

def create_task(coro, loop):
    # pragma: no cover
    """Compatibility wrapper for the loop.create_task() call introduced in
    3.4.2."""
    if hasattr(loop, 'create_task'):
        return loop.create_task(coro)
    return asyncio.Task(coro, loop=loop)

def remove_element(self, e):
        """Remove element `e` from model
        """
        
        if e.label is not None: self.elementdict.pop(e.label)
        self.elementlist.remove(e)

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

def get_codeblock(language, text):
    """ Generates rst codeblock for given text and language """
    rst = "\n\n.. code-block:: " + language + "\n\n"
    for line in text.splitlines():
        rst += "\t" + line + "\n"

    rst += "\n"
    return rst

def focus(self):
        """
        Call this to give this Widget the input focus.
        """
        self._has_focus = True
        self._frame.move_to(self._x, self._y, self._h)
        if self._on_focus is not None:
            self._on_focus()

def iparallel_progbar(mapper, iterable, nprocs=None, starmap=False, flatmap=False, shuffle=False,
                      verbose=True, verbose_flatmap=None, max_cache=-1, **kwargs):
    """Performs a parallel mapping of the given iterable, reporting a progress bar as values get returned. Yields
    objects as soon as they're computed, but does not guarantee that they'll be in the correct order.

    :param mapper: The mapping function to apply to elements of the iterable
    :param iterable: The iterable to map
    :param nprocs: The number of processes (defaults to the number of cpu's)
    :param starmap: If true, the iterable is expected to contain tuples and the mapper function gets each element of a
        tuple as an argument
    :param flatmap: If true, flatten out the returned values if the mapper function returns a list of objects
    :param shuffle: If true, randomly sort the elements before processing them. This might help provide more uniform
        runtimes if processing different objects takes different amounts of time.
    :param verbose: Whether or not to print the progress bar
    :param verbose_flatmap: If performing a flatmap, whether or not to report each object as it's returned
    :param max_cache: Maximum number of mapped objects to permit in the queue at once
    :param kwargs: Any other keyword arguments to pass to the progress bar (see ``progbar``)
    :return: A list of the returned objects, in whatever order they're done being computed
    """

    results = _parallel_progbar_launch(mapper, iterable, nprocs, starmap, flatmap, shuffle, verbose,
                                       verbose_flatmap, max_cache, **kwargs)
    return (x for i, x in results)

def is_hex_string(string):
    """Check if the string is only composed of hex characters."""
    pattern = re.compile(r'[A-Fa-f0-9]+')
    if isinstance(string, six.binary_type):
        string = str(string)
    return pattern.match(string) is not None

def _cell(x):
    """translate an array x into a MATLAB cell array"""
    x_no_none = [i if i is not None else "" for i in x]
    return array(x_no_none, dtype=np_object)

def _reset_bind(self):
        """Internal utility function to reset binding."""
        self.binded = False
        self._buckets = {}
        self._curr_module = None
        self._curr_bucket_key = None

def fixpath(path):
    """Uniformly format a path."""
    return os.path.normpath(os.path.realpath(os.path.expanduser(path)))

def decode_base64(data: str) -> bytes:
    """Decode base64, padding being optional.

    :param data: Base64 data as an ASCII byte string
    :returns: The decoded byte string.
    """
    missing_padding = len(data) % 4
    if missing_padding != 0:
        data += "=" * (4 - missing_padding)
    return base64.decodebytes(data.encode("utf-8"))

def set_default(self, section, option,
                    default):
        """If the option did not exist, create a default value."""
        if not self.parser.has_option(section, option):
            self.parser.set(section, option, default)

def _StopStatusUpdateThread(self):
    """Stops the status update thread."""
    self._status_update_active = False
    if self._status_update_thread.isAlive():
      self._status_update_thread.join()
    self._status_update_thread = None

def Date(value):
    """Custom type for managing dates in the command-line."""
    from datetime import datetime
    try:
        return datetime(*reversed([int(val) for val in value.split('/')]))
    except Exception as err:
        raise argparse.ArgumentTypeError("invalid date '%s'" % value)

def make_stream_handler(graph, formatter):
    """
    Create the stream handler. Used for console/debug output.

    """
    return {
        "class": graph.config.logging.stream_handler.class_,
        "formatter": formatter,
        "level": graph.config.logging.level,
        "stream": graph.config.logging.stream_handler.stream,
    }

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

def _linepoint(self, t, x0, y0, x1, y1):
        """ Returns coordinates for point at t on the line.
            Calculates the coordinates of x and y for a point at t on a straight line.
            The t parameter is a number between 0.0 and 1.0,
            x0 and y0 define the starting point of the line,
            x1 and y1 the ending point of the line.
        """
        # Originally from nodebox-gl
        out_x = x0 + t * (x1 - x0)
        out_y = y0 + t * (y1 - y0)
        return (out_x, out_y)

def unpickle_file(picklefile, **kwargs):
    """Helper function to unpickle data from `picklefile`."""
    with open(picklefile, 'rb') as f:
        return pickle.load(f, **kwargs)

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

def interpolate(f1: float, f2: float, factor: float) -> float:
    """ Linearly interpolate between two float values. """
    return f1 + (f2 - f1) * factor

def __next__(self):
        """
        :return: int
        """
        self.current += 1
        if self.current > self.total:
            raise StopIteration
        else:
            return self.iterable[self.current - 1]

def specialRound(number, rounding):
    """A method used to round a number in the way that UsefulUtils rounds."""
    temp = 0
    if rounding == 0:
        temp = number
    else:
        temp =  round(number, rounding)
    if temp % 1 == 0:
        return int(temp)
    else:
        return float(temp)

def _get_config_or_default(self, key, default, as_type=lambda x: x):
        """Return a main config value, or default if it does not exist."""

        if self.main_config.has_option(self.main_section, key):
            return as_type(self.main_config.get(self.main_section, key))
        return default

def assign_to(self, obj):
    """Assign `x` and `y` to an object that has properties `x` and `y`."""
    obj.x = self.x
    obj.y = self.y

def on_key_press(self, symbol, modifiers):
        """
        Pyglet specific key press callback.
        Forwards and translates the events to :py:func:`keyboard_event`
        """
        self.keyboard_event(symbol, self.keys.ACTION_PRESS, modifiers)

def connect_rds(aws_access_key_id=None, aws_secret_access_key=None, **kwargs):
    """
    :type aws_access_key_id: string
    :param aws_access_key_id: Your AWS Access Key ID

    :type aws_secret_access_key: string
    :param aws_secret_access_key: Your AWS Secret Access Key

    :rtype: :class:`boto.rds.RDSConnection`
    :return: A connection to RDS
    """
    from boto.rds import RDSConnection
    return RDSConnection(aws_access_key_id, aws_secret_access_key, **kwargs)

def generate_nonce():
        """ Generate nonce number """
        nonce = ''.join([str(randint(0, 9)) for i in range(8)])
        return HMAC(
            nonce.encode(),
            "secret".encode(),
            sha1
        ).hexdigest()

def get_system_uid():
    """Get a (probably) unique ID to identify a system.
    Used to differentiate votes.
    """
    try:
        if os.name == 'nt':
            return get_nt_system_uid()
        if sys.platform == 'darwin':
            return get_osx_system_uid()
    except Exception:
        return get_mac_uid()
    else:
        return get_mac_uid()

def monkey_restore():
    """restore real versions. Inverse of `monkey_patch`"""
    for k, v in originals.items():
        setattr(time_mod, k, v)
    
    global epoch
    epoch = None