def MultiArgMax(x):
  """
  Get tuple (actually a generator) of indices where the max value of
  array x occurs. Requires that x have a max() method, as x.max()
  (in the case of NumPy) is much faster than max(x).
  For a simpler, faster argmax when there is only a single maximum entry,
  or when knowing only the first index where the maximum occurs,
  call argmax() on a NumPy array.

  :param x: Any sequence that has a max() method.
  :returns: Generator with the indices where the max value occurs.
  """
  m = x.max()
  return (i for i, v in enumerate(x) if v == m)

def transformer_tpu_1b():
  """Hparams for machine translation with ~1.1B parameters."""
  hparams = transformer_tpu()
  hparams.hidden_size = 2048
  hparams.filter_size = 8192
  hparams.num_hidden_layers = 8
  # smaller batch size to avoid OOM
  hparams.batch_size = 1024
  hparams.activation_dtype = "bfloat16"
  hparams.weight_dtype = "bfloat16"
  # maximize number of parameters relative to computation by not sharing.
  hparams.shared_embedding_and_softmax_weights = False
  return hparams

def astensor(array: TensorLike) -> BKTensor:
    """Covert numpy array to tensorflow tensor"""
    tensor = tf.convert_to_tensor(value=array, dtype=CTYPE)
    return tensor

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

def handle_qbytearray(obj, encoding):
    """Qt/Python2/3 compatibility helper."""
    if isinstance(obj, QByteArray):
        obj = obj.data()

    return to_text_string(obj, encoding=encoding)

def find_centroid(region):
    """
    Finds an approximate centroid for a region that is within the region.
    
    Parameters
    ----------
    region : np.ndarray(shape=(m, n), dtype='bool')
        mask of the region.

    Returns
    -------
    i, j : tuple(int, int)
        2d index within the region nearest the center of mass.
    """

    x, y = center_of_mass(region)
    w = np.argwhere(region)
    i, j = w[np.argmin(np.linalg.norm(w - (x, y), axis=1))]
    return i, j

def day_to_month(timeperiod):
    """:param timeperiod: as string in YYYYMMDD00 format
    :return string in YYYYMM0000 format"""
    t = datetime.strptime(timeperiod, SYNERGY_DAILY_PATTERN)
    return t.strftime(SYNERGY_MONTHLY_PATTERN)

def write_file(filename, content):
    """Create the file with the given content"""
    print 'Generating {0}'.format(filename)
    with open(filename, 'wb') as out_f:
        out_f.write(content)

def c2s(self,p=[0,0]):
        """Convert from canvas to screen coordinates"""

        return((p[0]-self.canvasx(self.cx1),p[1]-self.canvasy(self.cy1)))

def clog(color):
    """Same to ``log``, but this one centralizes the message first."""
    logger = log(color)
    return lambda msg: logger(centralize(msg).rstrip())

def setdefaults(dct, defaults):
    """Given a target dct and a dict of {key:default value} pairs,
    calls setdefault for all of those pairs."""
    for key in defaults:
        dct.setdefault(key, defaults[key])

    return dct

def _count_leading_whitespace(text):
  """Returns the number of characters at the beginning of text that are whitespace."""
  idx = 0
  for idx, char in enumerate(text):
    if not char.isspace():
      return idx
  return idx + 1

def register_modele(self, modele: Modele):
        """ Register a modele onto the lemmatizer

        :param modele: Modele to register
        """
        self.lemmatiseur._modeles[modele.gr()] = modele

def index(self, value):
		"""
		Return the smallest index of the row(s) with this column
		equal to value.
		"""
		for i in xrange(len(self.parentNode)):
			if getattr(self.parentNode[i], self.Name) == value:
				return i
		raise ValueError(value)

def parse_list(cls, api, json_list):
        """
            Parse a list of JSON objects into
            a result set of model instances.
        """
        results = []
        for json_obj in json_list:
            if json_obj:
                obj = cls.parse(api, json_obj)
                results.append(obj)

        return results

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

def __convert_none_to_zero(self, ts):
        """
        Convert None values to 0 so the data works with Matplotlib
        :param ts:
        :return: a list with 0s where Nones existed
        """

        if not ts:
            return ts

        ts_clean = [val if val else 0 for val in ts]

        return ts_clean

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

def table_exists(self, table):
        """Returns whether the given table exists.

           :param table:
           :type table: BQTable
        """
        if not self.dataset_exists(table.dataset):
            return False

        try:
            self.client.tables().get(projectId=table.project_id,
                                     datasetId=table.dataset_id,
                                     tableId=table.table_id).execute()
        except http.HttpError as ex:
            if ex.resp.status == 404:
                return False
            raise

        return True

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

def listfolderpath(p):
    """
    generator of list folder in the path.
    folders only
    """
    for entry in scandir.scandir(p):
        if entry.is_dir():
            yield entry.path

def create_run_logfile(folder):
    """Create a 'run.log' within folder. This file contains the time of the
       latest successful run.
    """
    with open(os.path.join(folder, "run.log"), "w") as f:
        datestring = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        f.write("timestamp: '%s'" % datestring)

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

def from_pydatetime(cls, pydatetime):
        """
        Creates sql datetime2 object from Python datetime object
        ignoring timezone
        @param pydatetime: Python datetime object
        @return: sql datetime2 object
        """
        return cls(date=Date.from_pydate(pydatetime.date),
                   time=Time.from_pytime(pydatetime.time))

def call_fset(self, obj, value) -> None:
        """Store the given custom value and call the setter function."""
        vars(obj)[self.name] = self.fset(obj, value)

def remover(file_path):
    """Delete a file or directory path only if it exists."""
    if os.path.isfile(file_path):
        os.remove(file_path)
        return True
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)
        return True
    else:
        return False

def trapz2(f, x=None, y=None, dx=1.0, dy=1.0):
    """Double integrate."""
    return numpy.trapz(numpy.trapz(f, x=y, dx=dy), x=x, dx=dx)

def _latex_format(obj: Any) -> str:
    """Format an object as a latex string."""
    if isinstance(obj, float):
        try:
            return sympy.latex(symbolize(obj))
        except ValueError:
            return "{0:.4g}".format(obj)

    return str(obj)

def is_in(self, search_list, pair):
        """
        If pair is in search_list, return the index. Otherwise return -1
        """
        index = -1
        for nr, i in enumerate(search_list):
            if(np.all(i == pair)):
                return nr
        return index

def xmltreefromfile(filename):
    """Internal function to read an XML file"""
    try:
        return ElementTree.parse(filename, ElementTree.XMLParser(collect_ids=False))
    except TypeError:
        return ElementTree.parse(filename, ElementTree.XMLParser())

def _breakRemNewlines(tag):
	"""non-recursively break spaces and remove newlines in the tag"""
	for i,c in enumerate(tag.contents):
		if type(c) != bs4.element.NavigableString:
			continue
		c.replace_with(re.sub(r' {2,}', ' ', c).replace('\n',''))

def list2string (inlist,delimit=' '):
    """
Converts a 1D list to a single long string for file output, using
the string.join function.

Usage:   list2string (inlist,delimit=' ')
Returns: the string created from inlist
"""
    stringlist = [makestr(_) for _ in inlist]
    return string.join(stringlist,delimit)

def message_from_string(s, *args, **kws):
    """Parse a string into a Message object model.

    Optional _class and strict are passed to the Parser constructor.
    """
    from future.backports.email.parser import Parser
    return Parser(*args, **kws).parsestr(s)

def _construct_from_json(self, rec):
        """ Construct this Dagobah instance from a JSON document. """

        self.delete()

        for required_key in ['dagobah_id', 'created_jobs']:
            setattr(self, required_key, rec[required_key])

        for job_json in rec.get('jobs', []):
            self._add_job_from_spec(job_json)

        self.commit(cascade=True)

def get_deprecation_reason(
    node: Union[EnumValueDefinitionNode, FieldDefinitionNode]
) -> Optional[str]:
    """Given a field or enum value node, get deprecation reason as string."""
    from ..execution import get_directive_values

    deprecated = get_directive_values(GraphQLDeprecatedDirective, node)
    return deprecated["reason"] if deprecated else None

def update(self):
        """Updates image to be displayed with new time frame."""
        if self.single_channel:
            self.im.set_data(self.data[self.ind, :, :])
        else:
            self.im.set_data(self.data[self.ind, :, :, :])
        self.ax.set_ylabel('time frame %s' % self.ind)
        self.im.axes.figure.canvas.draw()

def post_worker_init(worker):
    """Hook into Gunicorn to display message after launching.

    This mimics the behaviour of Django's stock runserver command.
    """
    quit_command = 'CTRL-BREAK' if sys.platform == 'win32' else 'CONTROL-C'
    sys.stdout.write(
        "Django version {djangover}, Gunicorn version {gunicornver}, "
        "using settings {settings!r}\n"
        "Starting development server at {urls}\n"
        "Quit the server with {quit_command}.\n".format(
            djangover=django.get_version(),
            gunicornver=gunicorn.__version__,
            settings=os.environ.get('DJANGO_SETTINGS_MODULE'),
            urls=', '.join('http://{0}/'.format(b) for b in worker.cfg.bind),
            quit_command=quit_command,
        ),
    )

def generate_chunks(string, num_chars):
    """Yield num_chars-character chunks from string."""
    for start in range(0, len(string), num_chars):
        yield string[start:start+num_chars]

def _varargs_to_iterable_method(func):
    """decorator to convert a *args method to one taking a iterable"""
    def wrapped(self, iterable, **kwargs):
        return func(self, *iterable, **kwargs)
    return wrapped

def is_local_url(target):
    """Determine if URL is safe to redirect to."""
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ('http', 'https') and \
        ref_url.netloc == test_url.netloc

def on_welcome(self, connection, event):
        """
        Join the channel once connected to the IRC server.
        """
        connection.join(self.channel, key=settings.IRC_CHANNEL_KEY or "")

def stop(self):
        """Stops playback"""
        if self.isPlaying is True:
            self._execute("stop")
            self._changePlayingState(False)

def get(self):
        """Get the highest priority Processing Block from the queue."""
        with self._mutex:
            entry = self._queue.pop()
            del self._block_map[entry[2]]
            return entry[2]

def get_selection_owner(self, selection):
        """Return the window that owns selection (an atom), or X.NONE if
        there is no owner for the selection. Can raise BadAtom."""
        r = request.GetSelectionOwner(display = self.display,
                                      selection = selection)
        return r.owner

def py3round(number):
    """Unified rounding in all python versions."""
    if abs(round(number) - number) == 0.5:
        return int(2.0 * round(number / 2.0))

    return int(round(number))

def _checkSize(self):
        """Automatically resizes widget to display at most max_height_items items"""
        if self._item_height is not None:
            sz = min(self._max_height_items, self.count()) * self._item_height + 5
            sz = max(sz, 20)
            self.setMinimumSize(0, sz)
            self.setMaximumSize(1000000, sz)
            self.resize(self.width(), sz)

def update_menu(self):
        """Update context menu"""
        self.menu.clear()
        add_actions(self.menu, self.create_context_menu_actions())

def count_string_diff(a,b):
    """Return the number of characters in two strings that don't exactly match"""
    shortest = min(len(a), len(b))
    return sum(a[i] != b[i] for i in range(shortest))

def append_scope(self):
        """Create a new scope in the current frame."""
        self.stack.current.append(Scope(self.stack.current.current))

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

def _extension(modpath: str) -> setuptools.Extension:
    """Make setuptools.Extension."""
    return setuptools.Extension(modpath, [modpath.replace(".", "/") + ".py"])

def sbatch_template(self):
        """:return Jinja sbatch template for the current tag"""
        template = self.sbatch_template_str
        if template.startswith('#!'):
            # script is embedded in YAML
            return jinja_environment.from_string(template)
        return jinja_environment.get_template(template)

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

def max(self):
        """
        Returns the maximum value of the domain.

        :rtype: `float` or `np.inf`
        """
        return int(self._max) if not np.isinf(self._max) else self._max

def _sanitize(text):
    """Return sanitized Eidos text field for human readability."""
    d = {'-LRB-': '(', '-RRB-': ')'}
    return re.sub('|'.join(d.keys()), lambda m: d[m.group(0)], text)

def shape(self):
        """
        Return a tuple of axis dimensions
        """
        return tuple(len(self._get_axis(a)) for a in self._AXIS_ORDERS)

def can_elasticsearch(record):
    """Check if a given record is indexed.

    :param record: A record object.
    :returns: If the record is indexed returns `True`, otherwise `False`.
    """
    search = request._methodview.search_class()
    search = search.get_record(str(record.id))
    return search.count() == 1

def save(self, fname):
        """ Saves the dictionary in json format
        :param fname: file to save to
        """
        with open(fname, 'wb') as f:
            json.dump(self, f)

def test_string(self, string: str) -> bool:
        """If `string` comes next, return ``True`` and advance offset.

        Args:
            string: string to test
        """
        if self.input.startswith(string, self.offset):
            self.offset += len(string)
            return True
        return False

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

def arg_bool(name, default=False):
    """ Fetch a query argument, as a boolean. """
    v = request.args.get(name, '')
    if not len(v):
        return default
    return v in BOOL_TRUISH

def load_config(filename="logging.ini", *args, **kwargs):
    """
    Load logger config from file
    
    Keyword arguments:
    filename -- configuration filename (Default: "logging.ini")
    *args -- options passed to fileConfig
    **kwargs -- options passed to fileConfigg
    
    """
    logging.config.fileConfig(filename, *args, **kwargs)

def nested_update(d, u):
    """Merge two nested dicts.

    Nested dicts are sometimes used for representing various recursive structures. When
    updating such a structure, it may be convenient to present the updated data as a
    corresponding recursive structure. This function will then apply the update.

    Args:
      d: dict
        dict that will be updated in-place. May or may not contain nested dicts.

      u: dict
        dict with contents that will be merged into ``d``. May or may not contain
        nested dicts.

    """
    for k, v in list(u.items()):
        if isinstance(v, collections.Mapping):
            r = nested_update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d

def round_data(filter_data):
    """ round the data"""
    for index, _ in enumerate(filter_data):
        filter_data[index][0] = round(filter_data[index][0] / 100.0) * 100.0
    return filter_data

def right_outer(self):
        """
            Performs Right Outer Join
            :return right_outer: dict
        """
        self.get_collections_data()
        right_outer_join = self.merge_join_docs(
            set(self.collections_data['right'].keys()))
        return right_outer_join

def excel_key(index):
    """create a key for index by converting index into a base-26 number, using A-Z as the characters."""
    X = lambda n: ~n and X((n // 26)-1) + chr(65 + (n % 26)) or ''
    return X(int(index))

def dedup_list(l):
    """Given a list (l) will removing duplicates from the list,
       preserving the original order of the list. Assumes that
       the list entrie are hashable."""
    dedup = set()
    return [ x for x in l if not (x in dedup or dedup.add(x))]

def size(dtype):
  """Returns the number of bytes to represent this `dtype`."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'size'):
    return dtype.size
  return np.dtype(dtype).itemsize

def sort_nicely(l):
    """Sort the given list in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    l.sort(key=alphanum_key)

def this_week():
        """ Return start and end date of the current week. """
        since = TODAY + delta(weekday=MONDAY(-1))
        until = since + delta(weeks=1)
        return Date(since), Date(until)

def is_vector(inp):
    """ Returns true if the input can be interpreted as a 'true' vector

    Note
    ----
    Does only check dimensions, not if type is numeric

    Parameters
    ----------
    inp : numpy.ndarray or something that can be converted into ndarray

    Returns
    -------
    Boolean
        True for vectors: ndim = 1 or ndim = 2 and shape of one axis = 1
        False for all other arrays
    """
    inp = np.asarray(inp)
    nr_dim = np.ndim(inp)
    if nr_dim == 1:
        return True
    elif (nr_dim == 2) and (1 in inp.shape):
        return True
    else:
        return False

def isoformat(dt):
    """Return an ISO-8601 formatted string from the provided datetime object"""
    if not isinstance(dt, datetime.datetime):
        raise TypeError("Must provide datetime.datetime object to isoformat")

    if dt.tzinfo is None:
        raise ValueError("naive datetime objects are not allowed beyond the library boundaries")

    return dt.isoformat().replace("+00:00", "Z")

def __iter__(self):
        """Overloads iter(condition), and also, for bit in condition. The
        values yielded by the iterator are True (1), False (0), or
        None (#)."""
        for bit, mask in zip(self._bits, self._mask):
            yield bit if mask else None

def raw(self):
        """
        Build query and passes to `Elasticsearch`, then returns the raw
        format returned.
        """
        es = self.get_es()

        params = dict(self.query_params)
        mlt_fields = self.mlt_fields or params.pop('mlt_fields', [])

        body = self.s.build_search() if self.s else ''

        hits = es.mlt(
            index=self.index, doc_type=self.doctype, id=self.id,
            mlt_fields=mlt_fields, body=body, **params)

        log.debug(hits)

        return hits

def ex(self, cmd):
        """Execute a normal python statement in user namespace."""
        with self.builtin_trap:
            exec cmd in self.user_global_ns, self.user_ns

def clear(self):
        """Clear the displayed image."""
        self._imgobj = None
        try:
            # See if there is an image on the canvas
            self.canvas.delete_object_by_tag(self._canvas_img_tag)
            self.redraw()
        except KeyError:
            pass

def class_check(vector):
    """
    Check different items in matrix classes.

    :param vector: input vector
    :type vector : list
    :return: bool
    """
    for i in vector:
        if not isinstance(i, type(vector[0])):
            return False
    return True

def string_to_date(value):
    """
    Return a Python date that corresponds to the specified string
    representation.

    @param value: string representation of a date.

    @return: an instance ``datetime.datetime`` represented by the string.
    """
    if isinstance(value, datetime.date):
        return value

    return dateutil.parser.parse(value).date()

def delete_entry(self, key):
        """Delete an object from the redis table"""
        pipe = self.client.pipeline()
        pipe.srem(self.keys_container, key)
        pipe.delete(key)
        pipe.execute()

def parse_cookies_str(cookies):
    """
    parse cookies str to dict
    :param cookies: cookies str
    :type cookies: str
    :return: cookie dict
    :rtype: dict
    """
    cookie_dict = {}
    for record in cookies.split(";"):
        key, value = record.strip().split("=", 1)
        cookie_dict[key] = value
    return cookie_dict

def count_replica(self, partition):
        """Return count of replicas of given partition."""
        return sum(1 for b in partition.replicas if b in self.brokers)

def exists(self):
        """
        Returns true if the job is still running or zero-os still knows about this job ID

        After a job is finished, a job remains on zero-os for max of 5min where you still can read the job result
        after the 5 min is gone, the job result is no more fetchable
        :return: bool
        """
        r = self._client._redis
        flag = '{}:flag'.format(self._queue)
        return bool(r.exists(flag))

def __getitem__(self, index):
    """Get the item at the given index.

    Index is a tuple of (row, col)
    """
    row, col = index
    return self.rows[row][col]

def compare(left, right):
    """
    yields EVENT,ENTRY pairs describing the differences between left
    and right, which are filenames for a pair of zip files
    """

    with open_zip(left) as l:
        with open_zip(right) as r:
            return compare_zips(l, r)

def cell(self, rowName, columnName):
        """
        Returns the value of the cell on the given row and column.
        """
        return self.matrix[self.rowIndices[rowName], self.columnIndices[columnName]]

def segment_str(text: str, phoneme_inventory: Set[str] = PHONEMES) -> str:
    """
    Takes as input a string in Kunwinjku and segments it into phoneme-like
    units based on the standard orthographic rules specified at
    http://bininjgunwok.org.au/
    """

    text = text.lower()
    text = segment_into_tokens(text, phoneme_inventory)
    return text

def flatten_list(l: List[list]) -> list:
    """ takes a list of lists, l and returns a flat list
    """
    return [v for inner_l in l for v in inner_l]

def _pad(self, text):
        """Pad the text."""
        top_bottom = ("\n" * self._padding) + " "
        right_left = " " * self._padding * self.PAD_WIDTH
        return top_bottom + right_left + text + right_left + top_bottom

def add_to_enum(self, clsdict):
        """
        Compile XML mappings in addition to base add behavior.
        """
        super(XmlMappedEnumMember, self).add_to_enum(clsdict)
        self.register_xml_mapping(clsdict)

def safe_url(url):
    """Remove password from printed connection URLs."""
    parsed = urlparse(url)
    if parsed.password is not None:
        pwd = ':%s@' % parsed.password
        url = url.replace(pwd, ':*****@')
    return url

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

def stats(self):
        """
        Return a new raw REST interface to stats resources

        :rtype: :py:class:`ns1.rest.stats.Stats`
        """
        import ns1.rest.stats
        return ns1.rest.stats.Stats(self.config)

def _composed_doc(fs):
    """
    Generate a docstring for the composition of fs.
    """
    if not fs:
        # Argument name for the docstring.
        return 'n'

    return '{f}({g})'.format(f=fs[0].__name__, g=_composed_doc(fs[1:]))

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

def wheel(delta=1):
    """ Sends a wheel event for the provided number of clicks. May be negative to reverse
    direction. """
    location = get_position()
    e = Quartz.CGEventCreateMouseEvent(
        None,
        Quartz.kCGEventScrollWheel,
        location,
        Quartz.kCGMouseButtonLeft)
    e2 = Quartz.CGEventCreateScrollWheelEvent(
        None,
        Quartz.kCGScrollEventUnitLine,
        1,
        delta)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, e)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, e2)

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

def help(self, level=0):
        """return the usage string for available options """
        self.cmdline_parser.formatter.output_level = level
        with _patch_optparse():
            return self.cmdline_parser.format_help()

def as_dict(df, ix=':'):
    """ converts df to dict and adds a datetime field if df is datetime """
    if isinstance(df.index, pd.DatetimeIndex):
        df['datetime'] = df.index
    return df.to_dict(orient='records')[ix]

def get_h5file(file_path, mode='r'):
    """ Return the h5py.File given its file path.

    Parameters
    ----------
    file_path: string
        HDF5 file path

    mode: string
        r   Readonly, file must exist
        r+  Read/write, file must exist
        w   Create file, truncate if exists
        w-  Create file, fail if exists
        a   Read/write if exists, create otherwise (default)

    Returns
    -------
    h5file: h5py.File
    """
    if not op.exists(file_path):
        raise IOError('Could not find file {}.'.format(file_path))

    try:
        h5file = h5py.File(file_path, mode=mode)
    except:
        raise
    else:
        return h5file

def get_mi_vec(slab):
    """
    Convenience function which returns the unit vector aligned
    with the miller index.
    """
    mvec = np.cross(slab.lattice.matrix[0], slab.lattice.matrix[1])
    return mvec / np.linalg.norm(mvec)

def angle(x0, y0, x1, y1):
    """ Returns the angle between two points.
    """
    return degrees(atan2(y1-y0, x1-x0))

def pretty_dict_string(d, indent=0):
    """Pretty output of nested dictionaries.
    """
    s = ''
    for key, value in sorted(d.items()):
        s += '    ' * indent + str(key)
        if isinstance(value, dict):
             s += '\n' + pretty_dict_string(value, indent+1)
        else:
             s += '=' + str(value) + '\n'
    return s

def clear_last_lines(self, n):
        """Clear last N lines of terminal output.
        """
        self.term.stream.write(
            self.term.move_up * n + self.term.clear_eos)
        self.term.stream.flush()

def load_from_file(cls, file_path: str):
        """ Read and reconstruct the data from a JSON file. """
        with open(file_path, "r") as f:
            data = json.load(f)
            item = cls.decode(data=data)
        return item

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

def is_callable(*p):
    """ True if all the args are functions and / or subroutines
    """
    import symbols
    return all(isinstance(x, symbols.FUNCTION) for x in p)

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

def to_dotfile(self):
        """ Writes a DOT graphviz file of the domain structure, and returns the filename"""
        domain = self.get_domain()
        filename = "%s.dot" % (self.__class__.__name__)
        nx.write_dot(domain, filename)
        return filename

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

def _heappush_max(heap, item):
    """ why is this not in heapq """
    heap.append(item)
    heapq._siftdown_max(heap, 0, len(heap) - 1)

def add_swagger(app, json_route, html_route):
    """
    a convenience method for both adding a swagger.json route,
    as well as adding a page showing the html documentation
    """
    app.router.add_route('GET', json_route, create_swagger_json_handler(app))
    add_swagger_api_route(app, html_route, json_route)

def prepare(self):
        """Prepare the handler, ensuring RabbitMQ is connected or start a new
        connection attempt.

        """
        super(RabbitMQRequestHandler, self).prepare()
        if self._rabbitmq_is_closed:
            self._connect_to_rabbitmq()

def Exit(msg, code=1):
    """Exit execution with return code and message
    :param msg: Message displayed prior to exit
    :param code: code returned upon exiting
    """
    print >> sys.stderr, msg
    sys.exit(code)

def qubits(self):
        """Return a list of qubits as (QuantumRegister, index) pairs."""
        return [(v, i) for k, v in self.qregs.items() for i in range(v.size)]

def listen_for_updates(self):
        """Attach a callback on the group pubsub"""
        self.toredis.subscribe(self.group_pubsub, callback=self.callback)

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

def _chunk_write(chunk, local_file, progress):
    """Write a chunk to file and update the progress bar"""
    local_file.write(chunk)
    progress.update_with_increment_value(len(chunk))

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

def get_size_in_bytes(self, handle):
        """Return the size in bytes."""
        fpath = self._fpath_from_handle(handle)
        return os.stat(fpath).st_size

def web(host, port):
    """Start web application"""
    from .webserver.web import get_app
    get_app().run(host=host, port=port)

def main(argv=sys.argv, stream=sys.stderr):
    """Entry point for ``tappy`` command."""
    args = parse_args(argv)
    suite = build_suite(args)
    runner = unittest.TextTestRunner(verbosity=args.verbose, stream=stream)
    result = runner.run(suite)

    return get_status(result)

def split_every(n, iterable):
    """Returns a generator that spits an iteratable into n-sized chunks. The last chunk may have
    less than n elements.

    See http://stackoverflow.com/a/22919323/503377."""
    items = iter(iterable)
    return itertools.takewhile(bool, (list(itertools.islice(items, n)) for _ in itertools.count()))

def commit(self, message=None, amend=False, stage=True):
        """Commit any changes, optionally staging all changes beforehand."""
        return git_commit(self.repo_dir, message=message,
                          amend=amend, stage=stage)

def calc_list_average(l):
    """
    Calculates the average value of a list of numbers
    Returns a float
    """
    total = 0.0
    for value in l:
        total += value
    return total / len(l)

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

def forget_xy(t):
  """Ignore sizes of dimensions (1, 2) of a 4d tensor in shape inference.

  This allows using smaller input sizes, which create an invalid graph at higher
  layers (for example because a spatial dimension becomes smaller than a conv
  filter) when we only use early parts of it.
  """
  shape = (t.shape[0], None, None, t.shape[3])
  return tf.placeholder_with_default(t, shape)

def click_by_selector(self, selector):
    """Click the element matching the CSS selector."""
    # No need for separate button press step with selector style.
    elem = find_element_by_jquery(world.browser, selector)
    elem.click()

def _sub_patterns(patterns, text):
    """
    Apply re.sub to bunch of (pattern, repl)
    """
    for pattern, repl in patterns:
        text = re.sub(pattern, repl, text)
    return text

def add_0x(string):
    """Add 0x to string at start.
    """
    if isinstance(string, bytes):
        string = string.decode('utf-8')
    return '0x' + str(string)

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

def _update_bordercolor(self, bordercolor):
        """Updates background color"""

        border_color = wx.SystemSettings_GetColour(wx.SYS_COLOUR_ACTIVEBORDER)
        border_color.SetRGB(bordercolor)

        self.linecolor_choice.SetColour(border_color)

def clean_axis(axis):
    """Remove ticks, tick labels, and frame from axis"""
    axis.get_xaxis().set_ticks([])
    axis.get_yaxis().set_ticks([])
    for spine in list(axis.spines.values()):
        spine.set_visible(False)

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

def find_one(self, query):
        """Find one wrapper with conversion to dictionary

        :param dict query: A Mongo query
        """
        mongo_response = yield self.collection.find_one(query)
        raise Return(self._obj_cursor_to_dictionary(mongo_response))

def replace_list(items, match, replacement):
    """Replaces occurrences of a match string in a given list of strings and returns
    a list of new strings. The match string can be a regex expression.

    Args:
        items (list):       the list of strings to modify.
        match (str):        the search expression.
        replacement (str):  the string to replace with.
    """
    return [replace(item, match, replacement) for item in items]

def available_gpus():
  """List of GPU device names detected by TensorFlow."""
  local_device_protos = device_lib.list_local_devices()
  return [x.name for x in local_device_protos if x.device_type == 'GPU']

def count_rows(self, table_name):
        """Return the number of entries in a table by counting them."""
        self.table_must_exist(table_name)
        query = "SELECT COUNT (*) FROM `%s`" % table_name.lower()
        self.own_cursor.execute(query)
        return int(self.own_cursor.fetchone()[0])

def getMedian(numericValues):
    """
    Gets the median of a list of values
    Returns a float/int
    """
    theValues = sorted(numericValues)

    if len(theValues) % 2 == 1:
        return theValues[(len(theValues) + 1) / 2 - 1]
    else:
        lower = theValues[len(theValues) / 2 - 1]
        upper = theValues[len(theValues) / 2]

        return (float(lower + upper)) / 2

def stringify_dict_contents(dct):
    """Turn dict keys and values into native strings."""
    return {
        str_if_nested_or_str(k): str_if_nested_or_str(v)
        for k, v in dct.items()
    }

def ComplementEquivalence(*args, **kwargs):
    """Change x != y to not(x == y)."""
    return ast.Complement(
        ast.Equivalence(*args, **kwargs), **kwargs)

def vsh(cmd, *args, **kw):
    """ Execute a command installed into the active virtualenv.
    """
    args = '" "'.join(i.replace('"', r'\"') for i in args)
    easy.sh('"%s" "%s"' % (venv_bin(cmd), args))

def seaborn_bar_(self, label=None, style=None, opts=None):
        """
        Get a Seaborn bar chart
        """
        try:
            fig = sns.barplot(self.x, self.y, palette="BuGn_d")
            return fig
        except Exception as e:
            self.err(e, self.seaborn_bar_,
                     "Can not get Seaborn bar chart object")

def symmetrise(matrix, tri='upper'):
    """
    Will copy the selected (upper or lower) triangle of a square matrix
    to the opposite side, so that the matrix is symmetrical.
    Alters in place.
    """
    if tri == 'upper':
        tri_fn = np.triu_indices
    else:
        tri_fn = np.tril_indices
    size = matrix.shape[0]
    matrix[tri_fn(size)[::-1]] = matrix[tri_fn(size)]
    return matrix

def restore_button_state(self):
        """Helper to restore button state."""
        self.parent.pbnNext.setEnabled(self.next_button_state)
        self.parent.pbnBack.setEnabled(self.back_button_state)

def request_type(self):
        """Retrieve the type of the request, by fetching it from
        `xenon.proto.xenon_pb2`."""
        if self.static and not self.uses_request:
            return getattr(xenon_pb2, 'Empty')

        if not self.uses_request:
            return None

        return getattr(xenon_pb2, self.request_name)

def median(lst):
    """ Calcuates the median value in a @lst """
    #: http://stackoverflow.com/a/24101534
    sortedLst = sorted(lst)
    lstLen = len(lst)
    index = (lstLen - 1) // 2
    if (lstLen % 2):
        return sortedLst[index]
    else:
        return (sortedLst[index] + sortedLst[index + 1])/2.0

def distL1(x1,y1,x2,y2):
    """Compute the L1-norm (Manhattan) distance between two points.

    The distance is rounded to the closest integer, for compatibility
    with the TSPLIB convention.

    The two points are located on coordinates (x1,y1) and (x2,y2),
    sent as parameters"""
    return int(abs(x2-x1) + abs(y2-y1)+.5)

def handle_test(self, command, **options):
        """Send a test error to APM Server"""
        # can't be async for testing
        config = {"async_mode": False}
        for key in ("service_name", "secret_token"):
            if options.get(key):
                config[key] = options[key]
        client = DjangoClient(**config)
        client.error_logger = ColoredLogger(self.stderr)
        client.logger = ColoredLogger(self.stderr)
        self.write(
            "Trying to send a test error to APM Server using these settings:\n\n"
            "SERVICE_NAME:\t%s\n"
            "SECRET_TOKEN:\t%s\n"
            "SERVER:\t\t%s\n\n" % (client.config.service_name, client.config.secret_token, client.config.server_url)
        )

        try:
            raise TestException("Hi there!")
        except TestException:
            client.capture_exception()
            if not client.error_logger.errors:
                self.write(
                    "Success! We tracked the error successfully! You should be"
                    " able to see it in a few seconds at the above URL"
                )
        finally:
            client.close()

def dump_stmt_strings(stmts, fname):
    """Save printed statements in a file.

    Parameters
    ----------
    stmts_in : list[indra.statements.Statement]
        A list of statements to save in a text file.
    fname : Optional[str]
        The name of a text file to save the printed statements into.
    """
    with open(fname, 'wb') as fh:
        for st in stmts:
            fh.write(('%s\n' % st).encode('utf-8'))

def delete_connection():
    """
    Stop and destroy Bloomberg connection
    """
    if _CON_SYM_ in globals():
        con = globals().pop(_CON_SYM_)
        if not getattr(con, '_session').start(): con.stop()

def kill_dashboard(self, check_alive=True):
        """Kill the dashboard.

        Args:
            check_alive (bool): Raise an exception if the process was already
                dead.
        """
        self._kill_process_type(
            ray_constants.PROCESS_TYPE_DASHBOARD, check_alive=check_alive)

def exit(self):
        """Handle interactive exit.

        This method calls the ask_exit callback."""
        if self.confirm_exit:
            if self.ask_yes_no('Do you really want to exit ([y]/n)?','y'):
                self.ask_exit()
        else:
            self.ask_exit()

def _elapsed_time(begin_time, end_time):
    """Assuming format YYYY-MM-DD hh:mm:ss

    Returns the elapsed time in seconds
    """

    bt = _str2datetime(begin_time)
    et = _str2datetime(end_time)

    return float((et - bt).seconds)

def md5_string(s):
    """
    Shortcut to create md5 hash
    :param s:
    :return:
    """
    m = hashlib.md5()
    m.update(s)
    return str(m.hexdigest())

def sigterm(self, signum, frame):
        """
        These actions will be done after SIGTERM.
        """
        self.logger.warning("Caught signal %s. Stopping daemon." % signum)
        sys.exit(0)

def kill(self):
        """Kill the browser.

        This is useful when the browser is stuck.
        """
        if self.process:
            self.process.kill()
            self.process.wait()

def inject_nulls(data: Mapping, field_names) -> dict:
    """Insert None as value for missing fields."""

    record = dict()

    for field in field_names:
        record[field] = data.get(field, None)

    return record

def text_width(string, font_name, font_size):
    """Determine with width in pixels of string."""
    return stringWidth(string, fontName=font_name, fontSize=font_size)

def minimise_xyz(xyz):
    """Minimise an (x, y, z) coordinate."""
    x, y, z = xyz
    m = max(min(x, y), min(max(x, y), z))
    return (x-m, y-m, z-m)

def urlize_twitter(text):
    """
    Replace #hashtag and @username references in a tweet with HTML text.
    """
    html = TwitterText(text).autolink.auto_link()
    return mark_safe(html.replace(
        'twitter.com/search?q=', 'twitter.com/search/realtime/'))

def get_from_headers(request, key):
    """Try to read a value named ``key`` from the headers.
    """
    value = request.headers.get(key)
    return to_native(value)

def pylog(self, *args, **kwargs):
        """Display all available logging information."""
        printerr(self.name, args, kwargs, traceback.format_exc())

def request(self, method, url, body=None, headers={}):
        """Send a complete request to the server."""
        self._send_request(method, url, body, headers)

def loads(cls, s):
        """
        Load an instance of this class from YAML.

        """
        with closing(StringIO(s)) as fileobj:
            return cls.load(fileobj)

def compute_jaccard_index(x_set, y_set):
    """Return the Jaccard similarity coefficient of 2 given sets.

    Args:
        x_set (set): first set.
        y_set (set): second set.

    Returns:
        float: Jaccard similarity coefficient.

    """
    if not x_set or not y_set:
        return 0.0

    intersection_cardinal = len(x_set & y_set)
    union_cardinal = len(x_set | y_set)

    return intersection_cardinal / float(union_cardinal)

def sort_matrix(a,n=0):
    """
    This will rearrange the array a[n] from lowest to highest, and
    rearrange the rest of a[i]'s in the same way. It is dumb and slow.

    Returns a numpy array.
    """
    a = _n.array(a)
    return a[:,a[n,:].argsort()]

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

def go_to_parent_directory(self):
        """Go to parent directory"""
        self.chdir(osp.abspath(osp.join(getcwd_or_home(), os.pardir)))

def haversine(x):
    """Return the haversine of an angle

    haversine(x) = sin(x/2)**2, where x is an angle in radians
    """
    y = .5*x
    y = np.sin(y)
    return y*y

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

def expand_args(cmd_args):
    """split command args to args list
    returns a list of args

    :param cmd_args: command args, can be tuple, list or str
    """
    if isinstance(cmd_args, (tuple, list)):
        args_list = list(cmd_args)
    else:
        args_list = shlex.split(cmd_args)
    return args_list

def vline(self, x, y, height, color):
        """Draw a vertical line up to a given length."""
        self.rect(x, y, 1, height, color, fill=True)

def is_dataframe(obj):
    """
    Returns True if the given object is a Pandas Data Frame.

    Parameters
    ----------
    obj: instance
        The object to test whether or not is a Pandas DataFrame.
    """
    try:
        # This is the best method of type checking
        from pandas import DataFrame
        return isinstance(obj, DataFrame)
    except ImportError:
        # Pandas is not a dependency, so this is scary
        return obj.__class__.__name__ == "DataFrame"

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

def get_dict_to_encoded_url(data):
    """
    Converts a dict to an encoded URL.
    Example: given  data = {'a': 1, 'b': 2}, it returns 'a=1&b=2'
    """
    unicode_data = dict([(k, smart_str(v)) for k, v in data.items()])
    encoded = urllib.urlencode(unicode_data)
    return encoded

def to_bin(data, width):
    """
    Convert an unsigned integer to a numpy binary array with the first
    element the MSB and the last element the LSB.
    """
    data_str = bin(data & (2**width-1))[2:].zfill(width)
    return [int(x) for x in tuple(data_str)]

def lower_ext(abspath):
    """Convert file extension to lowercase.
    """
    fname, ext = os.path.splitext(abspath)
    return fname + ext.lower()

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

def __init__(self, token, editor=None):
        """Create a GistAPI object

        Arguments:
            token: an authentication token
            editor: path to the editor to use when editing a gist

        """
        self.token = token
        self.editor = editor
        self.session = requests.Session()

def _request_modify_dns_record(self, record):
        """Sends Modify_DNS_Record request"""
        return self._request_internal("Modify_DNS_Record",
                                      domain=self.domain,
                                      record=record)

def iflatten(L):
    """Iterative flatten."""
    for sublist in L:
        if hasattr(sublist, '__iter__'):
            for item in iflatten(sublist): yield item
        else: yield sublist

def finish_plot():
    """Helper for plotting."""
    plt.legend()
    plt.grid(color='0.7')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def incr(self, key, incr_by=1):
        """Increment the key by the given amount."""
        return self.database.hincrby(self.key, key, incr_by)

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

def load_object_at_path(path):
    """Load an object from disk at explicit path"""
    with open(path, 'r') as f:
        data = _deserialize(f.read())
        return aadict(data)

def task_property_present_predicate(service, task, prop):
    """ True if the json_element passed is present for the task specified.
    """
    try:
        response = get_service_task(service, task)
    except Exception as e:
        pass

    return (response is not None) and (prop in response)

def items(self, limit=0):
        """Return iterator for items in each page"""
        i = ItemIterator(self.iterator)
        i.limit = limit
        return i

def assert_valid_input(cls, tag):
        """Check if valid input tag or document."""

        # Fail on unexpected types.
        if not cls.is_tag(tag):
            raise TypeError("Expected a BeautifulSoup 'Tag', but instead recieved type {}".format(type(tag)))

def remove_all_handler(self):
        """
        Unlink the file handler association.
        """
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            self._handler_cache.append(handler)

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

def confirm_credential_display(force=False):
    if force:
        return True

    msg = """
    [WARNING] Your credential is about to be displayed on screen.
    If this is really what you want, type 'y' and press enter."""

    result = click.confirm(text=msg)
    return result

def describe_enum_value(enum_value):
    """Build descriptor for Enum instance.

    Args:
      enum_value: Enum value to provide descriptor for.

    Returns:
      Initialized EnumValueDescriptor instance describing the Enum instance.
    """
    enum_value_descriptor = EnumValueDescriptor()
    enum_value_descriptor.name = six.text_type(enum_value.name)
    enum_value_descriptor.number = enum_value.number
    return enum_value_descriptor

def _cosine(a, b):
    """ Return the len(a & b) / len(a) """
    return 1. * len(a & b) / (math.sqrt(len(a)) * math.sqrt(len(b)))

def to_dict(dictish):
    """
    Given something that closely resembles a dictionary, we attempt
    to coerce it into a propery dictionary.
    """
    if hasattr(dictish, 'iterkeys'):
        m = dictish.iterkeys
    elif hasattr(dictish, 'keys'):
        m = dictish.keys
    else:
        raise ValueError(dictish)

    return dict((k, dictish[k]) for k in m())

def tokenize_list(self, text):
        """
        Split a text into separate words.
        """
        return [self.get_record_token(record) for record in self.analyze(text)]

def pack_bits( longbits ):
    """Crunch a 64-bit int (8 bool bytes) into a bitfield."""
    byte = longbits & (0x0101010101010101)
    byte = (byte | (byte>>7)) & (0x0003000300030003)
    byte = (byte | (byte>>14)) & (0x0000000f0000000f)
    byte = (byte | (byte>>28)) & (0x00000000000000ff)
    return byte

def filedata(self):
        """Property providing access to the :class:`.FileDataAPI`"""
        if self._filedata_api is None:
            self._filedata_api = self.get_filedata_api()
        return self._filedata_api

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

def _linearInterpolationTransformMatrix(matrix1, matrix2, value):
    """ Linear, 'oldstyle' interpolation of the transform matrix."""
    return tuple(_interpolateValue(matrix1[i], matrix2[i], value) for i in range(len(matrix1)))

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

def wr_row_mergeall(self, worksheet, txtstr, fmt, row_idx):
        """Merge all columns and place text string in widened cell."""
        hdridxval = len(self.hdrs) - 1
        worksheet.merge_range(row_idx, 0, row_idx, hdridxval, txtstr, fmt)
        return row_idx + 1

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

def parse_func_kwarg_keys(func, with_vals=False):
    """ hacky inference of kwargs keys

    SeeAlso:
        argparse_funckw
        recursive_parse_kwargs
        parse_kwarg_keys
        parse_func_kwarg_keys
        get_func_kwargs

    """
    sourcecode = get_func_sourcecode(func, strip_docstr=True,
                                     strip_comments=True)
    kwkeys = parse_kwarg_keys(sourcecode, with_vals=with_vals)
    #ut.get_func_kwargs  TODO
    return kwkeys

def _depr(fn, usage, stacklevel=3):
    """Internal convenience function for deprecation warnings"""
    warn('{0} is deprecated. Use {1} instead'.format(fn, usage),
         stacklevel=stacklevel, category=DeprecationWarning)

def abfIDfromFname(fname):
    """given a filename, return the ABFs ID string."""
    fname=os.path.abspath(fname)
    basename=os.path.basename(fname)
    return os.path.splitext(basename)[0]

def _clean_workers(self):
        """Delete periodically workers in workers bag."""
        while self._bag_collector:
            self._bag_collector.popleft()
        self._timer_worker_delete.stop()

def bounding_box_from(points, i, i1, thr):
    """Creates bounding box for a line segment

    Args:
        points (:obj:`list` of :obj:`Point`)
        i (int): Line segment start, index in points array
        i1 (int): Line segment end, index in points array
    Returns:
        (float, float, float, float): with bounding box min x, min y, max x and max y
    """
    pi = points[i]
    pi1 = points[i1]

    min_lat = min(pi.lat, pi1.lat)
    min_lon = min(pi.lon, pi1.lon)
    max_lat = max(pi.lat, pi1.lat)
    max_lon = max(pi.lon, pi1.lon)

    return min_lat-thr, min_lon-thr, max_lat+thr, max_lon+thr

def s3_connect(bucket_name, s3_access_key_id, s3_secret_key):
    """ Returns a Boto connection to the provided S3 bucket. """
    conn = connect_s3(s3_access_key_id, s3_secret_key)
    try:
        return conn.get_bucket(bucket_name)
    except S3ResponseError as e:
        if e.status == 403:
            raise Exception("Bad Amazon S3 credentials.")
        raise

def dump_nparray(self, obj, class_name=numpy_ndarray_class_name):
        """
        ``numpy.ndarray`` dumper.
        """
        return {"$" + class_name: self._json_convert(obj.tolist())}

def html(header_rows):
    """
    Convert a list of tuples describing a table into a HTML string
    """
    name = 'table%d' % next(tablecounter)
    return HtmlTable([map(str, row) for row in header_rows], name).render()

def install():
        """
        Installs ScoutApm SQL Instrumentation by monkeypatching the `cursor`
        method of BaseDatabaseWrapper, to return a wrapper that instruments any
        calls going through it.
        """

        @monkeypatch_method(BaseDatabaseWrapper)
        def cursor(original, self, *args, **kwargs):
            result = original(*args, **kwargs)
            return _DetailedTracingCursorWrapper(result, self)

        logger.debug("Monkey patched SQL")

def mkdir(dir, enter):
    """Create directory with template for topic of the current environment

    """

    if not os.path.exists(dir):
        os.makedirs(dir)

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

def pop(self, index=-1):
		"""Remove and return the item at index."""
		value = self._list.pop(index)
		del self._dict[value]
		return value

def handle_whitespace(text):
    r"""Handles whitespace cleanup.

    Tabs are "smartly" retabbed (see sub_retab). Lines that contain
    only whitespace are truncated to a single newline.
    """
    text = re_retab.sub(sub_retab, text)
    text = re_whitespace.sub('', text).strip()
    return text

def AmericanDateToEpoch(self, date_str):
    """Take a US format date and return epoch."""
    try:
      epoch = time.strptime(date_str, "%m/%d/%Y")
      return int(calendar.timegm(epoch)) * 1000000
    except ValueError:
      return 0

def __round_time(self, dt):
    """Round a datetime object to a multiple of a timedelta
    dt : datetime.datetime object, default now.
    """
    round_to = self._resolution.total_seconds()
    seconds  = (dt - dt.min).seconds
    rounding = (seconds + round_to / 2) // round_to * round_to
    return dt + timedelta(0, rounding - seconds, -dt.microsecond)

def random_alphanum(length):
    """
    Return a random string of ASCII letters and digits.

    :param int length: The length of string to return
    :returns: A random string
    :rtype: str
    """
    charset = string.ascii_letters + string.digits
    return random_string(length, charset)

def datatype(dbtype, description, cursor):
    """Google AppEngine Helper to convert a data type into a string."""
    dt = cursor.db.introspection.get_field_type(dbtype, description)
    if type(dt) is tuple:
        return dt[0]
    else:
        return dt

def _encode_bool(name, value, dummy0, dummy1):
    """Encode a python boolean (True/False)."""
    return b"\x08" + name + (value and b"\x01" or b"\x00")

def get_scalar_product(self, other):
        """Returns the scalar product of this vector with the given
        other vector."""
        return self.x*other.x+self.y*other.y

def iterlists(self):
        """Like :meth:`items` but returns an iterator."""
        for key, values in dict.iteritems(self):
            yield key, list(values)

def _init_unique_sets(self):
        """Initialise sets used for uniqueness checking."""

        ks = dict()
        for t in self._unique_checks:
            key = t[0]
            ks[key] = set() # empty set
        return ks

def run(self, forever=True):
        """start the bot"""
        loop = self.create_connection()
        self.add_signal_handlers()
        if forever:
            loop.run_forever()

def FromString(self, string):
    """Parse a bool from a string."""
    if string.lower() in ("false", "no", "n"):
      return False

    if string.lower() in ("true", "yes", "y"):
      return True

    raise TypeValueError("%s is not recognized as a boolean value." % string)

def vectorize(values):
    """
    Takes a value or list of values and returns a single result, joined by ","
    if necessary.
    """
    if isinstance(values, list):
        return ','.join(str(v) for v in values)
    return values

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

def ss(*args, **kwargs):
    """
    exactly like s, but doesn't return variable names or file positions (useful for logging)

    since -- 10-15-2015
    return -- str
    """
    if not args:
        raise ValueError("you didn't pass any arguments to print out")

    with Reflect.context(args, **kwargs) as r:
        instance = V_CLASS(r, stream, **kwargs)
        return instance.value().strip()

def hamming(s, t):
    """
    Calculate the Hamming distance between two strings. From Wikipedia article: Iterative with two matrix rows.

    :param s: string 1
    :type s: str
    :param t: string 2
    :type s: str
    :return: Hamming distance
    :rtype: float
    """
    if len(s) != len(t):
        raise ValueError('Hamming distance needs strings of equal length.')
    return sum(s_ != t_ for s_, t_ in zip(s, t))

def add_object(self, object):
        """Add object to db session. Only for session-centric object-database mappers."""
        if object.id is None:
            object.get_id()
        self.db.engine.save(object)

def default_static_path():
    """
        Return the path to the javascript bundle
    """
    fdir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(fdir, '../assets/'))

def test():
    """Test for ReverseDNS class"""
    dns = ReverseDNS()

    print(dns.lookup('192.168.0.1'))
    print(dns.lookup('8.8.8.8'))

    # Test cache
    print(dns.lookup('8.8.8.8'))

def _is_start(event, node, tagName):  # pylint: disable=invalid-name
    """Return true if (event, node) is a start event for tagname."""

    return event == pulldom.START_ELEMENT and node.tagName == tagName

def on_mouse_motion(self, x, y, dx, dy):
        """
        Pyglet specific mouse motion callback.
        Forwards and traslates the event to the example
        """
        # Screen coordinates relative to the lower-left corner
        # so we have to flip the y axis to make this consistent with
        # other window libraries
        self.example.mouse_position_event(x, self.buffer_height - y)

def get_single_file_info(self, rel_path):
        """ Gets last change time for a single file """

        f_path = self.get_full_file_path(rel_path)
        return get_single_file_info(f_path, rel_path)

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

async def stop(self):
        """Stop the current task process.

        Starts with SIGTERM, gives the process 1 second to terminate, then kills it
        """
        # negate pid so that signals apply to process group
        pgid = -self.process.pid
        try:
            os.kill(pgid, signal.SIGTERM)
            await asyncio.sleep(1)
            os.kill(pgid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            return

def build_columns(self, X, verbose=False):
        """construct the model matrix columns for the term

        Parameters
        ----------
        X : array-like
            Input dataset with n rows

        verbose : bool
            whether to show warnings

        Returns
        -------
        scipy sparse array with n rows
        """
        return sp.sparse.csc_matrix(X[:, self.feature][:, np.newaxis])

def to_camel_case(snake_case_string):
    """
    Convert a string from snake case to camel case. For example, "some_var" would become "someVar".

    :param snake_case_string: Snake-cased string to convert to camel case.
    :returns: Camel-cased version of snake_case_string.
    """
    parts = snake_case_string.lstrip('_').split('_')
    return parts[0] + ''.join([i.title() for i in parts[1:]])

def datetime_to_timestamp(dt):
    """Convert a UTC datetime to a Unix timestamp"""
    delta = dt - datetime.utcfromtimestamp(0)
    return delta.seconds + delta.days * 24 * 3600

def block(seed):
    """ Return block of normal random numbers

    Parameters
    ----------
    seed : {None, int}
        The seed to generate the noise.sd

    Returns
    --------
    noise : numpy.ndarray
        Array of random numbers
    """
    num = SAMPLE_RATE * BLOCK_SIZE
    rng = RandomState(seed % 2**32)
    variance = SAMPLE_RATE / 2
    return rng.normal(size=num, scale=variance**0.5)

def create_h5py_with_large_cache(filename, cache_size_mb):
    """
Allows to open the hdf5 file with specified cache size
    """
    # h5py does not allow to control the cache size from the high level
    # we employ the workaround
    # sources:
    #http://stackoverflow.com/questions/14653259/how-to-set-cache-settings-while-using-h5py-high-level-interface
    #https://groups.google.com/forum/#!msg/h5py/RVx1ZB6LpE4/KH57vq5yw2AJ
    propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    settings = list(propfaid.get_cache())
    settings[2] = 1024 * 1024 * cache_size_mb
    propfaid.set_cache(*settings)
    fid = h5py.h5f.create(filename, flags=h5py.h5f.ACC_EXCL, fapl=propfaid)
    fin = h5py.File(fid)
    return fin

def check(text):
    """Check the text."""
    err = "misc.currency"
    msg = u"Incorrect use of symbols in {}."

    symbols = [
        "\$[\d]* ?(?:dollars|usd|us dollars)"
    ]

    return existence_check(text, symbols, err, msg)

def escape_tex(value):
  """
  Make text tex safe
  """
  newval = value
  for pattern, replacement in LATEX_SUBS:
    newval = pattern.sub(replacement, newval)
  return newval

def import_path(self):
    """The full remote import path as used in import statements in `.go` source files."""
    return os.path.join(self.remote_root, self.pkg) if self.pkg else self.remote_root

def run(self):
        """
        consume message from channel on the consuming thread.
        """
        LOGGER.debug("rabbitmq.Service.run")
        try:
            self.channel.start_consuming()
        except Exception as e:
            LOGGER.warn("rabbitmq.Service.run - Exception raised while consuming")

def filter_set(input, **params):
    """
    Apply WHERE filter to input dataset
    :param input:
    :param params:
    :return: filtered data
    """
    PARAM_WHERE = 'where'

    return Converter.df2list(pd.DataFrame.from_records(input).query(params.get(PARAM_WHERE)))

def _trim(self, somestr):
        """ Trim left-right given string """
        tmp = RE_LSPACES.sub("", somestr)
        tmp = RE_TSPACES.sub("", tmp)
        return str(tmp)

def exists(self):
        """
        Performs an existence check on the remote database.

        :returns: Boolean True if the database exists, False otherwise
        """
        resp = self.r_session.head(self.database_url)
        if resp.status_code not in [200, 404]:
            resp.raise_for_status()

        return resp.status_code == 200

def getTopRight(self):
        """
        Retrieves a tuple with the x,y coordinates of the upper right point of the ellipse. 
        Requires the radius and the coordinates to be numbers
        """
        return (float(self.get_cx()) + float(self.get_rx()), float(self.get_cy()) + float(self.get_ry()))

def filter_none(list_of_points):
    """
    
    :param list_of_points: 
    :return: list_of_points with None's removed
    """
    remove_elementnone = filter(lambda p: p is not None, list_of_points)
    remove_sublistnone = filter(lambda p: not contains_none(p), remove_elementnone)
    return list(remove_sublistnone)

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

def show(self):
        """ Ensure the widget is shown.
        Calling this method will also set the widget visibility to True.
        """
        self.visible = True
        if self.proxy_is_active:
            self.proxy.ensure_visible()

def calculate_delay(original, delay):
    """
        Calculate the delay
    """
    original = datetime.strptime(original, '%H:%M')
    delayed = datetime.strptime(delay, '%H:%M')
    diff = delayed - original
    return diff.total_seconds() // 60

def to_array(self):
        """Convert the table to a structured NumPy array."""
        dt = np.dtype(list(zip(self.labels, (c.dtype for c in self.columns))))
        arr = np.empty_like(self.columns[0], dt)
        for label in self.labels:
            arr[label] = self[label]
        return arr

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

def get_request(self, request):
        """Sets token-based auth headers."""
        request.transport_user = self.username
        request.transport_password = self.api_key
        return request

def make_unique_ngrams(s, n):
    """Make a set of unique n-grams from a string."""
    return set(s[i:i + n] for i in range(len(s) - n + 1))

def merge(left, right, how='inner', key=None, left_key=None, right_key=None,
          left_as='left', right_as='right'):
    """ Performs a join using the union join function. """
    return join(left, right, how, key, left_key, right_key,
                join_fn=make_union_join(left_as, right_as))

def readTuple(self, line, n=3):
        """ Reads a tuple of numbers. e.g. vertices, normals or teture coords.
        """
        numbers = [num for num in line.split(' ') if num]
        return [float(num) for num in numbers[1:n + 1]]

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

def get_inputs_from_cm(index, cm):
    """Return indices of inputs to the node with the given index."""
    return tuple(i for i in range(cm.shape[0]) if cm[i][index])

def _request(self, method: str, endpoint: str, params: dict = None, data: dict = None, headers: dict = None) -> dict:
        """HTTP request method of interface implementation."""

def RemoveMethod(self, function):
        """
        Removes the specified function's MethodWrapper from the
        added_methods list, so we don't re-bind it when making a clone.
        """
        self.added_methods = [dm for dm in self.added_methods if not dm.method is function]

def eglInitialize(display):
    """ Initialize EGL and return EGL version tuple.
    """
    majorVersion = (_c_int*1)()
    minorVersion = (_c_int*1)()
    res = _lib.eglInitialize(display, majorVersion, minorVersion)
    if res == EGL_FALSE:
        raise RuntimeError('Could not initialize')
    return majorVersion[0], minorVersion[0]

def _date_to_json(value):
    """Coerce 'value' to an JSON-compatible representation."""
    if isinstance(value, datetime.date):
        value = value.isoformat()
    return value

def has_next_async(self):
    """Return a Future whose result will say whether a next item is available.

    See the module docstring for the usage pattern.
    """
    if self._fut is None:
      self._fut = self._iter.getq()
    flag = True
    try:
      yield self._fut
    except EOFError:
      flag = False
    raise tasklets.Return(flag)

def magnitude(X):
    """Magnitude of a complex matrix."""
    r = np.real(X)
    i = np.imag(X)
    return np.sqrt(r * r + i * i);

def _heapreplace_max(heap, item):
    """Maxheap version of a heappop followed by a heappush."""
    returnitem = heap[0]    # raises appropriate IndexError if heap is empty
    heap[0] = item
    _siftup_max(heap, 0)
    return returnitem

def on_stop(self):
        """
        stop publisher
        """
        LOGGER.debug("zeromq.Publisher.on_stop")
        self.zmqsocket.close()
        self.zmqcontext.destroy()

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

def included_length(self):
        """Surveyed length, not including "excluded" shots"""
        return sum([shot.length for shot in self.shots if shot.is_included])

def _parse_tuple_string(argument):
        """ Return a tuple from parsing 'a,b,c,d' -> (a,b,c,d) """
        if isinstance(argument, str):
            return tuple(int(p.strip()) for p in argument.split(','))
        return argument

def empty(self, start=None, stop=None):
		"""Empty the range from start to stop.

		Like delete, but no Error is raised if the entire range isn't mapped.
		"""
		self.set(NOT_SET, start=start, stop=stop)

def filter(self, f, operator="and"):
        """
        Add a filter to the query

        Takes a Filter object, or a filterable DSL object.
        """
        if self._filtered:
            self._filter_dsl.filter(f)
        else:
            self._build_filtered_query(f, operator)
        return self

def FromString(s, **kwargs):
    """Like FromFile, but takes a string."""
    
    f = StringIO.StringIO(s)
    return FromFile(f, **kwargs)

def get_highlighted_code(name, code, type='terminal'):
    """
    If pygments are available on the system
    then returned output is colored. Otherwise
    unchanged content is returned.
    """
    import logging
    try:
        import pygments
        pygments
    except ImportError:
        return code
    from pygments import highlight
    from pygments.lexers import guess_lexer_for_filename, ClassNotFound
    from pygments.formatters import TerminalFormatter

    try:
        lexer = guess_lexer_for_filename(name, code)
        formatter = TerminalFormatter()
        content = highlight(code, lexer, formatter)
    except ClassNotFound:
        logging.debug("Couldn't guess Lexer, will not use pygments.")
        content = code
    return content

def map(cls, iterable, func, *a, **kw):
    """
    Iterable-first replacement of Python's built-in `map()` function.
    """

    return cls(func(x, *a, **kw) for x in iterable)

def _query_for_reverse_geocoding(lat, lng):
    """
    Given a lat & lng, what's the string search query.

    If the API changes, change this function. Only for internal use.
    """
    # have to do some stupid f/Decimal/str stuff to (a) ensure we get as much
    # decimal places as the user already specified and (b) to ensure we don't
    # get e-5 stuff
    return "{0:f},{1:f}".format(Decimal(str(lat)), Decimal(str(lng)))

def Output(self):
    """Output all sections of the page."""
    self.Open()
    self.Header()
    self.Body()
    self.Footer()

def previous_workday(dt):
    """
    returns previous weekday used for observances
    """
    dt -= timedelta(days=1)
    while dt.weekday() > 4:
        # Mon-Fri are 0-4
        dt -= timedelta(days=1)
    return dt

def test():        
    """Local test."""
    from spyder.utils.qthelpers import qapplication
    app = qapplication()
    dlg = ProjectDialog(None)
    dlg.show()
    sys.exit(app.exec_())

def read_utf8(fh, byteorder, dtype, count, offsetsize):
    """Read tag data from file and return as unicode string."""
    return fh.read(count).decode('utf-8')

def _make_index(df, cols=META_IDX):
    """Create an index from the columns of a dataframe"""
    return pd.MultiIndex.from_tuples(
        pd.unique(list(zip(*[df[col] for col in cols]))), names=tuple(cols))

async def readline(self):
        """
        This is an asyncio adapted version of pyserial read.
        It provides a non-blocking read and returns a line of data read.

        :return: A line of data
        """
        future = asyncio.Future()
        data_available = False
        while True:
            if not data_available:
                if not self.my_serial.inWaiting():
                    await asyncio.sleep(self.sleep_tune)
                else:
                    data_available = True
                    data = self.my_serial.readline()
                    future.set_result(data)
            else:
                if not future.done():
                    await asyncio.sleep(self.sleep_tune)
                else:
                    return future.result()

def extract_keywords_from_text(self, text):
        """Method to extract keywords from the text provided.

        :param text: Text to extract keywords from, provided as a string.
        """
        sentences = nltk.tokenize.sent_tokenize(text)
        self.extract_keywords_from_sentences(sentences)

def read_folder(directory):
    """read text files in directory and returns them as array

    Args:
        directory: where the text files are

    Returns:
        Array of text
    """
    res = []
    for filename in os.listdir(directory):
        with io.open(os.path.join(directory, filename), encoding="utf-8") as f:
            content = f.read()
            res.append(content)
    return res

def server(port):
    """Start the Django dev server."""
    args = ['python', 'manage.py', 'runserver']
    if port:
        args.append(port)
    run.main(args)

def remove_blank_lines(string):
    """ Removes all blank lines in @string

        -> #str without blank lines
    """
    return "\n".join(line
                     for line in string.split("\n")
                     if len(line.strip()))

def _generate_key_map(entity_list, key, entity_class):
    """ Helper method to generate map from key to entity object for given list of dicts.

    Args:
      entity_list: List consisting of dict.
      key: Key in each dict which will be key in the map.
      entity_class: Class representing the entity.

    Returns:
      Map mapping key to entity object.
    """

    key_map = {}
    for obj in entity_list:
      key_map[obj[key]] = entity_class(**obj)

    return key_map

def _take_ownership(self):
        """Make the Python instance take ownership of the GIBaseInfo. i.e.
        unref if the python instance gets gc'ed.
        """

        if self:
            ptr = cast(self.value, GIBaseInfo)
            _UnrefFinalizer.track(self, ptr)
            self.__owns = True

def nlevels(self):
        """
        Get the number of factor levels for each categorical column.

        :returns: A list of the number of levels per column.
        """
        levels = self.levels()
        return [len(l) for l in levels] if levels else 0

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

def __iter__(self):
        """
        Returns the list of modes.

        :return:
        """
        return iter([v for k, v in sorted(self._modes.items())])

def change_cell(self, x, y, ch, fg, bg):
        """Change cell in position (x;y).
        """
        self.console.draw_char(x, y, ch, fg, bg)

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

def _return_comma_list(self, l):
        """ get a list and return a string with comma separated list values
        Examples ['to', 'ta'] will return 'to,ta'.
        """
        if isinstance(l, (text_type, int)):
            return l

        if not isinstance(l, list):
            raise TypeError(l, ' should be a list of integers, \
not {0}'.format(type(l)))

        str_ids = ','.join(str(i) for i in l)

        return str_ids

def each_img(dir_path):
    """
    Iterates through each image in the given directory. (not recursive)
    :param dir_path: Directory path where images files are present
    :return: Iterator to iterate through image files
    """
    for fname in os.listdir(dir_path):
        if fname.endswith('.jpg') or fname.endswith('.png') or fname.endswith('.bmp'):
            yield fname

def unique_everseen(iterable, filterfalse_=itertools.filterfalse):
    """Unique elements, preserving order."""
    # Itertools recipes:
    # https://docs.python.org/3/library/itertools.html#itertools-recipes
    seen = set()
    seen_add = seen.add
    for element in filterfalse_(seen.__contains__, iterable):
        seen_add(element)
        yield element

def get_py_source(file):
    """
    Retrieves and returns the source code for any Python
    files requested by the UI via the host agent

    @param file [String] The fully qualified path to a file
    """
    try:
        response = None
        pysource = ""

        if regexp_py.search(file) is None:
            response = {"error": "Only Python source files are allowed. (*.py)"}
        else:
            with open(file, 'r') as pyfile:
                pysource = pyfile.read()

            response = {"data": pysource}

    except Exception as e:
        response = {"error": str(e)}
    finally:
        return response

def delete_collection(mongo_uri, database_name, collection_name):
    """
    Delete a mongo document collection using pymongo. Mongo daemon assumed to be running.

    Inputs: - mongo_uri: A MongoDB URI.
            - database_name: The mongo database name as a python string.
            - collection_name: The mongo collection as a python string.
    """
    client = pymongo.MongoClient(mongo_uri)

    db = client[database_name]

    db.drop_collection(collection_name)

def thai_to_eng(text: str) -> str:
    """
    Correct text in one language that is incorrectly-typed with a keyboard layout in another language. (type Thai with English keyboard)

    :param str text: Incorrect input (type English with Thai keyboard)
    :return: English text
    """
    return "".join(
        [TH_EN_KEYB_PAIRS[ch] if (ch in TH_EN_KEYB_PAIRS) else ch for ch in text]
    )

def normalize_value(text):
    """
    This removes newlines and multiple spaces from a string.
    """
    result = text.replace('\n', ' ')
    result = re.subn('[ ]{2,}', ' ', result)[0]
    return result

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

def init_db():
    """
    Drops and re-creates the SQL schema
    """
    db.drop_all()
    db.configure_mappers()
    db.create_all()
    db.session.commit()

def _preprocess(df):
    """
    given a DataFrame where records are stored row-wise, rearrange it
    such that records are stored column-wise.
    """

    df = df.stack()

    df.index.rename(["id", "time"], inplace=True)  # .reset_index()
    df.name = "value"
    df = df.reset_index()

    return df

def point_in_multipolygon(point, multipoly):
    """
    valid whether the point is located in a mulitpolygon (donut polygon is not supported)

    Keyword arguments:
    point      -- point geojson object
    multipoly  -- multipolygon geojson object

    if(point inside multipoly) return true else false
    """
    coords_array = [multipoly['coordinates']] if multipoly[
        'type'] == "MultiPolygon" else multipoly['coordinates']

    for coords in coords_array:
        if _point_in_polygon(point, coords):
            return True

    return False

def get_auth():
    """Get authentication."""
    import getpass
    user = input("User Name: ")  # noqa
    pswd = getpass.getpass('Password: ')
    return Github(user, pswd)

def log_finished(self):
		"""Log that this task is done."""
		delta = time.perf_counter() - self.start_time
		logger.log("Finished '", logger.cyan(self.name),
			"' after ", logger.magenta(time_to_text(delta)))

def table_top_abs(self):
        """Returns the absolute position of table top"""
        table_height = np.array([0, 0, self.table_full_size[2]])
        return string_to_array(self.floor.get("pos")) + table_height

def merge(self, other):
        """ Merge another stats. """
        Stats.merge(self, other)
        self.changes += other.changes

def fcast(value: float) -> TensorLike:
    """Cast to float tensor"""
    newvalue = tf.cast(value, FTYPE)
    if DEVICE == 'gpu':
        newvalue = newvalue.gpu()  # Why is this needed?  # pragma: no cover
    return newvalue

def _import(module, cls):
    """
    A messy way to import library-specific classes.
    TODO: I should really make a factory class or something, but I'm lazy.
    Plus, factories remind me a lot of java...
    """
    global Scanner

    try:
        cls = str(cls)
        mod = __import__(str(module), globals(), locals(), [cls], 1)
        Scanner = getattr(mod, cls)
    except ImportError:
        pass

def solve(A, x):
    """Solves a linear equation system with a matrix of shape (n, n) and an
    array of shape (n, ...). The output has the same shape as the second
    argument.
    """
    # https://stackoverflow.com/a/48387507/353337
    x = numpy.asarray(x)
    return numpy.linalg.solve(A, x.reshape(x.shape[0], -1)).reshape(x.shape)

def save_keras_definition(keras_model, path):
    """
    Save a Keras model definition to JSON with given path
    """
    model_json = keras_model.to_json()
    with open(path, "w") as json_file:
        json_file.write(model_json)

def replace_month_abbr_with_num(date_str, lang=DEFAULT_DATE_LANG):
    """Replace month strings occurrences with month number."""
    num, abbr = get_month_from_date_str(date_str, lang)
    return re.sub(abbr, str(num), date_str, flags=re.IGNORECASE)

def _save_cookies(requests_cookiejar, filename):
    """Save cookies to a file."""
    with open(filename, 'wb') as handle:
        pickle.dump(requests_cookiejar, handle)

def pickle_data(data, picklefile):
    """Helper function to pickle `data` in `picklefile`."""
    with open(picklefile, 'wb') as f:
        pickle.dump(data, f, protocol=2)

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

def All(sequence):
  """
  :param sequence: Any sequence whose elements can be evaluated as booleans.
  :returns: true if all elements of the sequence satisfy True and x.
  """
  return bool(reduce(lambda x, y: x and y, sequence, True))

def sort_func(self, key):
        """Sorting logic for `Quantity` objects."""
        if key == self._KEYS.VALUE:
            return 'aaa'
        if key == self._KEYS.SOURCE:
            return 'zzz'
        return key

def show_intro(self):
        """Show intro to IPython help"""
        from IPython.core.usage import interactive_usage
        self.main.help.show_rich_text(interactive_usage)

def _time_to_json(value):
    """Coerce 'value' to an JSON-compatible representation."""
    if isinstance(value, datetime.time):
        value = value.isoformat()
    return value

def parse_obj(o):
    """
    Parses a given dictionary with the key being the OBD PID and the value its
    returned value by the OBD interface
    :param dict o:
    :return:
    """
    r = {}
    for k, v in o.items():
        if is_unable_to_connect(v):
            r[k] = None

        try:
            r[k] = parse_value(k, v)
        except (ObdPidParserUnknownError, AttributeError, TypeError):
            r[k] = None
    return r

def _euclidean_dist(vector_a, vector_b):
    """
    :param vector_a:    A list of numbers.
    :param vector_b:    A list of numbers.
    :returns:           The euclidean distance between the two vectors.
    """
    dist = 0
    for (x, y) in zip(vector_a, vector_b):
        dist += (x-y)*(x-y)
    return math.sqrt(dist)

def _most_common(iterable):
    """Returns the most common element in `iterable`."""
    data = Counter(iterable)
    return max(data, key=data.__getitem__)

def getFlaskResponse(responseString, httpStatus=200):
    """
    Returns a Flask response object for the specified data and HTTP status.
    """
    return flask.Response(responseString, status=httpStatus, mimetype=MIMETYPE)

def isCommaList(inputFilelist):
    """Return True if the input is a comma separated list of names."""
    if isinstance(inputFilelist, int) or isinstance(inputFilelist, np.int32):
        ilist = str(inputFilelist)
    else:
        ilist = inputFilelist
    if "," in ilist:
        return True
    return False

def ismatch(text, pattern):
    """Test whether text contains string or matches regex."""

    if hasattr(pattern, 'search'):
        return pattern.search(text) is not None
    else:
        return pattern in text if Config.options.case_sensitive \
            else pattern.lower() in text.lower()

def extract_args(argv):
    """
    take sys.argv that is used to call a command-line script and return a correctly split list of arguments
    for example, this input: ["eqarea.py", "-f", "infile", "-F", "outfile", "-A"]
    will return this output: [['f', 'infile'], ['F', 'outfile'], ['A']]
    """
    string = " ".join(argv)
    string = string.split(' -')
    program = string[0]
    arguments = [s.split() for s in string[1:]]
    return arguments

def _update_globals():
    """
    Patch the globals to remove the objects not available on some platforms.

    XXX it'd be better to test assertions about bytecode instead.
    """

    if not sys.platform.startswith('java') and sys.platform != 'cli':
        return
    incompatible = 'extract_constant', 'get_module_constant'
    for name in incompatible:
        del globals()[name]
        __all__.remove(name)

def getPiLambert(n):
    """Returns a list containing first n digits of Pi
    """
    mypi = piGenLambert()
    result = []
    if n > 0:
        result += [next(mypi) for i in range(n)]
    mypi.close()
    return result

def uniform_iterator(sequence):
    """Uniform (key, value) iteration on a `dict`,
    or (idx, value) on a `list`."""

    if isinstance(sequence, abc.Mapping):
        return six.iteritems(sequence)
    else:
        return enumerate(sequence)

def is_array(self, key):
        """Return True if variable is a numpy array"""
        data = self.model.get_data()
        return isinstance(data[key], (ndarray, MaskedArray))

def prnt(self):
        """
        Prints DB data representation of the object.
        """
        print("= = = =\n\n%s object key: \033[32m%s\033[0m" % (self.__class__.__name__, self.key))
        pprnt(self._data or self.clean_value())

def _validate_date_str(str_):
    """Validate str as a date and return string version of date"""

    if not str_:
        return None

    # Convert to datetime so we can validate it's a real date that exists then
    # convert it back to the string.
    try:
        date = datetime.strptime(str_, DATE_FMT)
    except ValueError:
        msg = 'Invalid date format, should be YYYY-MM-DD'
        raise argparse.ArgumentTypeError(msg)

    return date.strftime(DATE_FMT)

def is_valid_varname(varname):
    """ Checks syntax and validity of a variable name """
    if not isinstance(varname, six.string_types):
        return False
    match_obj = re.match(varname_regex, varname)
    valid_syntax = match_obj is not None
    valid_name = not keyword.iskeyword(varname)
    isvalid = valid_syntax and valid_name
    return isvalid

def Join(self):
    """Waits until all outstanding tasks are completed."""

    for _ in range(self.JOIN_TIMEOUT_DECISECONDS):
      if self._queue.empty() and not self.busy_threads:
        return
      time.sleep(0.1)

    raise ValueError("Timeout during Join() for threadpool %s." % self.name)

def gaussian_kernel(sigma, truncate=4.0):
    """Return Gaussian that truncates at the given number of std deviations.

    Adapted from https://github.com/nicjhan/gaussian-filter
    """

    sigma = float(sigma)
    radius = int(truncate * sigma + 0.5)

    x, y = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    sigma = sigma ** 2

    k = 2 * np.exp(-0.5 * (x ** 2 + y ** 2) / sigma)
    k = k / np.sum(k)

    return k

def display_pil_image(im):
   """Displayhook function for PIL Images, rendered as PNG."""
   from IPython.core import display
   b = BytesIO()
   im.save(b, format='png')
   data = b.getvalue()

   ip_img = display.Image(data=data, format='png', embed=True)
   return ip_img._repr_png_()

def filter_lines_from_comments(lines):
    """ Filter the lines from comments and non code lines. """
    for line_nb, raw_line in enumerate(lines):
        clean_line = remove_comments_from_line(raw_line)
        if clean_line == '':
            continue
        yield line_nb, clean_line, raw_line

def has_changed (filename):
    """Check if filename has changed since the last check. If this
    is the first check, assume the file is changed."""
    key = os.path.abspath(filename)
    mtime = get_mtime(key)
    if key not in _mtime_cache:
        _mtime_cache[key] = mtime
        return True
    return mtime > _mtime_cache[key]

def iter_except_top_row_tcs(self):
        """Generate each `a:tc` element in non-first rows of range."""
        for tr in self._tbl.tr_lst[self._top + 1:self._bottom]:
            for tc in tr.tc_lst[self._left:self._right]:
                yield tc

def submit_by_selector(self, selector):
    """Submit the form matching the CSS selector."""
    elem = find_element_by_jquery(world.browser, selector)
    elem.submit()

def _load_texture(file_name, resolver):
    """
    Load a texture from a file into a PIL image.
    """
    file_data = resolver.get(file_name)
    image = PIL.Image.open(util.wrap_as_stream(file_data))
    return image

def strToBool(val):
    """
    Helper function to turn a string representation of "true" into
    boolean True.
    """
    if isinstance(val, str):
        val = val.lower()

    return val in ['true', 'on', 'yes', True]

def set_primary_key(self, table, column):
        """Create a Primary Key constraint on a specific column when the table is already created."""
        self.execute('ALTER TABLE {0} ADD PRIMARY KEY ({1})'.format(wrap(table), column))
        self._printer('\tAdded primary key to {0} on column {1}'.format(wrap(table), column))

def DeleteLog() -> None:
        """Delete log file."""
        if os.path.exists(Logger.FileName):
            os.remove(Logger.FileName)

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

def autocorr_coeff(x, t, tau1, tau2):
    """Calculate the autocorrelation coefficient."""
    return corr_coeff(x, x, t, tau1, tau2)

def Load(file):
    """ Loads a model from specified file """
    with open(file, 'rb') as file:
        model = dill.load(file)
        return model

def save(self, *args, **kwargs):
        """Saves an animation

        A wrapper around :meth:`matplotlib.animation.Animation.save`
        """
        self.timeline.index -= 1  # required for proper starting point for save
        self.animation.save(*args, **kwargs)

def Proxy(f):
  """A helper to create a proxy method in a class."""

  def Wrapped(self, *args):
    return getattr(self, f)(*args)

  return Wrapped

def converged(matrix1, matrix2):
    """
    Check for convergence by determining if 
    matrix1 and matrix2 are approximately equal.
    
    :param matrix1: The matrix to compare with matrix2
    :param matrix2: The matrix to compare with matrix1
    :returns: True if matrix1 and matrix2 approximately equal
    """
    if isspmatrix(matrix1) or isspmatrix(matrix2):
        return sparse_allclose(matrix1, matrix2)

    return np.allclose(matrix1, matrix2)

def process_literal_param(self, value: Optional[List[int]],
                              dialect: Dialect) -> str:
        """Convert things on the way from Python to the database."""
        retval = self._intlist_to_dbstr(value)
        return retval

def teardown(self):
        """
        Stop and remove the container if it exists.
        """
        while self._http_clients:
            self._http_clients.pop().close()
        if self.created:
            self.halt()

def same_network(atree, btree) -> bool:
    """True if given trees share the same structure of powernodes,
    independently of (power)node names,
    and same edge topology between (power)nodes.

    """
    return same_hierarchy(atree, btree) and same_topology(atree, btree)

def test():  # pragma: no cover
    """Execute the unit tests on an installed copy of unyt.

    Note that this function requires pytest to run. If pytest is not
    installed this function will raise ImportError.
    """
    import pytest
    import os

    pytest.main([os.path.dirname(os.path.abspath(__file__))])

def load(self, path):
        """Load the pickled model weights."""
        with io.open(path, 'rb') as fin:
            self.weights = pickle.load(fin)

def default_diff(latest_config, current_config):
    """Determine if two revisions have actually changed."""
    # Pop off the fields we don't care about:
    pop_no_diff_fields(latest_config, current_config)

    diff = DeepDiff(
        latest_config,
        current_config,
        ignore_order=True
    )
    return diff

def _upper(val_list):
    """
    :param val_list: a list of strings
    :return: a list of upper-cased strings
    """
    res = []
    for ele in val_list:
        res.append(ele.upper())
    return res

def disable_wx(self):
        """Disable event loop integration with wxPython.

        This merely sets PyOS_InputHook to NULL.
        """
        if self._apps.has_key(GUI_WX):
            self._apps[GUI_WX]._in_event_loop = False
        self.clear_inputhook()

def attr_cache_clear(self):
        node = extract_node("""def cache_clear(self): pass""")
        return BoundMethod(proxy=node, bound=self._instance.parent.scope())

def update_target(self, name, current, total):
        """Updates progress bar for a specified target."""
        self.refresh(self._bar(name, current, total))

def guess_encoding(text, default=DEFAULT_ENCODING):
    """Guess string encoding.

    Given a piece of text, apply character encoding detection to
    guess the appropriate encoding of the text.
    """
    result = chardet.detect(text)
    return normalize_result(result, default=default)

def write_string(value, buff, byteorder='big'):
    """Write a string to a file-like object."""
    data = value.encode('utf-8')
    write_numeric(USHORT, len(data), buff, byteorder)
    buff.write(data)

def softplus(attrs, inputs, proto_obj):
    """Applies the sofplus activation function element-wise to the input."""
    new_attrs = translation_utils._add_extra_attributes(attrs, {'act_type' : 'softrelu'})
    return 'Activation', new_attrs, inputs

def get_size(objects):
    """Compute the total size of all elements in objects."""
    res = 0
    for o in objects:
        try:
            res += _getsizeof(o)
        except AttributeError:
            print("IGNORING: type=%s; o=%s" % (str(type(o)), str(o)))
    return res

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

def datetime_from_isoformat(value: str):
    """Return a datetime object from an isoformat string.

    Args:
        value (str): Datetime string in isoformat.

    """
    if sys.version_info >= (3, 7):
        return datetime.fromisoformat(value)

    return datetime.strptime(value, '%Y-%m-%dT%H:%M:%S.%f')

def btc_make_p2sh_address( script_hex ):
    """
    Make a P2SH address from a hex script
    """
    h = hashing.bin_hash160(binascii.unhexlify(script_hex))
    addr = bin_hash160_to_address(h, version_byte=multisig_version_byte)
    return addr

def filter_list_by_indices(lst, indices):
    """Return a modified list containing only the indices indicated.

    Args:
        lst: Original list of values
        indices: List of indices to keep from the original list

    Returns:
        list: Filtered list of values

    """
    return [x for i, x in enumerate(lst) if i in indices]

def mcc(y, z):
    """Matthews correlation coefficient
    """
    tp, tn, fp, fn = contingency_table(y, z)
    return (tp * tn - fp * fn) / K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

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

def set_property(self, key, value):
        """
        Update only one property in the dict
        """
        self.properties[key] = value
        self.sync_properties()

def serialisasi(self):
        """Mengembalikan hasil serialisasi objek Makna ini.

        :returns: Dictionary hasil serialisasi
        :rtype: dict
        """

        return {
            "kelas": self.kelas,
            "submakna": self.submakna,
            "info": self.info,
            "contoh": self.contoh
        }

def _normal_prompt(self):
        """
        Flushes the prompt before requesting the input

        :return: The command line
        """
        sys.stdout.write(self.__get_ps1())
        sys.stdout.flush()
        return safe_input()

def token(name):
    """Marker for a token

    :param str name: Name of tokenizer
    """

    def wrap(f):
        tokenizers.append((name, f))
        return f

    return wrap

def copy(self):
        """Return a copy of this list with each element copied to new memory
        """
        out = type(self)()
        for series in self:
            out.append(series.copy())
        return out

def up(self):
        
        """Moves the layer up in the stacking order.
        
        """
        
        i = self.index()
        if i != None:
            del self.canvas.layers[i]
            i = min(len(self.canvas.layers), i+1)
            self.canvas.layers.insert(i, self)

def selecttrue(table, field, complement=False):
    """Select rows where the given field evaluates `True`."""

    return select(table, field, lambda v: bool(v), complement=complement)

def __cmp__(self, other):
        """Comparsion not implemented."""
        # Stops python 2 from allowing comparsion of arbitrary objects
        raise TypeError('unorderable types: {}, {}'
                        ''.format(self.__class__.__name__, type(other)))

def s3(ctx, bucket_name, data_file, region):
    """Use the S3 SWAG backend."""
    if not ctx.data_file:
        ctx.data_file = data_file

    if not ctx.bucket_name:
        ctx.bucket_name = bucket_name

    if not ctx.region:
        ctx.region = region

    ctx.type = 's3'

def get_previous_month(self):
        """Returns date range for the previous full month."""
        end = utils.get_month_start() - relativedelta(days=1)
        end = utils.to_datetime(end)
        start = utils.get_month_start(end)
        return start, end

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

def test3():
    """Test the multiprocess
    """
    import time
    
    p = MVisionProcess()
    p.start()
    time.sleep(5)
    p.stop()

def batch(items, size):
    """Batches a list into a list of lists, with sub-lists sized by a specified
    batch size."""
    return [items[x:x + size] for x in xrange(0, len(items), size)]

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

def has_edge(self, edge):
        """
        Return whether an edge exists.

        @type  edge: tuple
        @param edge: Edge.

        @rtype:  boolean
        @return: Truth-value for edge existence.
        """
        u, v = edge
        return (u, v) in self.edge_properties

def dict_update_newkeys(dict_, dict2):
    """ Like dict.update, but does not overwrite items """
    for key, val in six.iteritems(dict2):
        if key not in dict_:
            dict_[key] = val

def install_plugin(username, repo):
    """Installs a Blended plugin from GitHub"""
    print("Installing plugin from " + username + "/" + repo)

    pip.main(['install', '-U', "git+git://github.com/" +
              username + "/" + repo + ".git"])

def ynticks(self, nticks, index=1):
        """Set the number of ticks."""
        self.layout['yaxis' + str(index)]['nticks'] = nticks
        return self

def keys(self):
        """Return ids of all indexed documents."""
        result = []
        if self.fresh_index is not None:
            result += self.fresh_index.keys()
        if self.opt_index is not None:
            result += self.opt_index.keys()
        return result

def split(text: str) -> List[str]:
    """Split a text into a list of tokens.

    :param text: the text to split
    :return: tokens
    """
    return [word for word in SEPARATOR.split(text) if word.strip(' \t')]

def merge_dict(data, *args):
    """Merge any number of dictionaries
    """
    results = {}
    for current in (data,) + args:
        results.update(current)
    return results

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

def is_symlink(self):
        """
        Whether this path is a symbolic link.
        """
        try:
            return S_ISLNK(self.lstat().st_mode)
        except OSError as e:
            if e.errno != ENOENT:
                raise
            # Path doesn't exist
            return False

def camel_to_snake_case(string):
    """Converts 'string' presented in camel case to snake case.

    e.g.: CamelCase => snake_case
    """
    s = _1.sub(r'\1_\2', string)
    return _2.sub(r'\1_\2', s).lower()

def _on_scale(self, event):
        """
        Callback for the Scale widget, inserts an int value into the Entry.

        :param event: Tkinter event
        """
        self._entry.delete(0, tk.END)
        self._entry.insert(0, str(self._variable.get()))

def create_index(config):
    """Create the root index."""
    filename = pathlib.Path(config.cache_path) / "index.json"
    index = {"version": __version__}
    with open(filename, "w") as out:
        out.write(json.dumps(index, indent=2))

def timeit(method):
    """
    A Python decorator for printing out the execution time for a function.

    Adapted from:
    www.andreas-jung.com/contents/a-python-decorator-for-measuring-the-execution-time-of-methods
    """
    def timed(*args, **kw):
        time_start = time.time()
        result = method(*args, **kw)
        time_end = time.time()
        print('timeit: %r %2.2f sec (%r, %r) ' % (method.__name__, time_end-time_start, str(args)[:20], kw))
        return result

    return timed

def request(method, url, **kwargs):
    """
    Wrapper for the `requests.request()` function.
    It accepts the same arguments as the original, plus an optional `retries`
    that overrides the default retry mechanism.
    """
    retries = kwargs.pop('retries', None)
    with Session(retries=retries) as session:
        return session.request(method=method, url=url, **kwargs)

def writeBoolean(self, n):
        """
        Writes a Boolean to the stream.
        """
        t = TYPE_BOOL_TRUE

        if n is False:
            t = TYPE_BOOL_FALSE

        self.stream.write(t)

def text(value, encoding="utf-8", errors="strict"):
    """Convert a value to str on Python 3 and unicode on Python 2."""
    if isinstance(value, text_type):
        return value
    elif isinstance(value, bytes):
        return text_type(value, encoding, errors)
    else:
        return text_type(value)

def concat(cls, iterables):
    """
    Similar to #itertools.chain.from_iterable().
    """

    def generator():
      for it in iterables:
        for element in it:
          yield element
    return cls(generator())

def add_plot(x, y, xl, yl, fig, ax, LATEX=False, linestyle=None, **kwargs):
    """Add plots to an existing plot"""
    if LATEX:
        xl_data = xl[1]  # NOQA
        yl_data = yl[1]
    else:
        xl_data = xl[0]  # NOQA
        yl_data = yl[0]

    for idx in range(len(y)):
        ax.plot(x, y[idx], label=yl_data[idx], linestyle=linestyle)

    ax.legend(loc='upper right')
    ax.set_ylim(auto=True)

def make_symmetric(dict):
    """Makes the given dictionary symmetric. Values are assumed to be unique."""
    for key, value in list(dict.items()):
        dict[value] = key
    return dict

def __exit__(self, *exc_info):
        """Close connection to NATS when used in a context manager"""

        self._loop.create_task(self._close(Client.CLOSED, True))

def gauss_pdf(x, mu, sigma):
    """Normalized Gaussian"""
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - mu) ** 2 / 2. / sigma ** 2)

def do_file_show(client, args):
    """Output file contents to stdout"""
    for src_uri in args.uris:
        client.download_file(src_uri, sys.stdout.buffer)

    return True

def trivial_partition(set_):
    """Returns a parition of given set into 1-element subsets.

    :return: Trivial partition of given set, i.e. iterable containing disjoint
             1-element sets, each consisting of a single element
             from given set
    """
    ensure_countable(set_)

    result = ((x,) for x in set_)
    return _harmonize_subset_types(set_, result)

def _cpu(self):
        """Record CPU usage."""
        value = int(psutil.cpu_percent())
        set_metric("cpu", value, category=self.category)
        gauge("cpu", value)

def get_mac_dot_app_dir(directory):
    """Returns parent directory of mac .app

    Args:

       directory (str): Current directory

    Returns:

       (str): Parent directory of mac .app
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(directory)))

def extract_alzip (archive, compression, cmd, verbosity, interactive, outdir):
    """Extract a ALZIP archive."""
    return [cmd, '-d', outdir, archive]

def widget(self, f):
        """
        Return an interactive function widget for the given function.

        The widget is only constructed, not displayed nor attached to
        the function.

        Returns
        -------
        An instance of ``self.cls`` (typically :class:`interactive`).

        Parameters
        ----------
        f : function
            The function to which the interactive widgets are tied.
        """
        return self.cls(f, self.opts, **self.kwargs)

def _from_dict(cls, _dict):
        """Initialize a ListCollectionsResponse object from a json dictionary."""
        args = {}
        if 'collections' in _dict:
            args['collections'] = [
                Collection._from_dict(x) for x in (_dict.get('collections'))
            ]
        return cls(**args)

def to_iso_string(self) -> str:
        """ Returns full ISO string for the given date """
        assert isinstance(self.value, datetime)
        return datetime.isoformat(self.value)

def get_bin_indices(self, values):
        """Returns index tuple in histogram of bin which contains value"""
        return tuple([self.get_axis_bin_index(values[ax_i], ax_i)
                      for ax_i in range(self.dimensions)])

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

def close_other_windows(self):
        """
        Closes all not current windows. Useful for tests - after each test you
        can automatically close all windows.
        """
        main_window_handle = self.current_window_handle
        for window_handle in self.window_handles:
            if window_handle == main_window_handle:
                continue
            self.switch_to_window(window_handle)
            self.close()
        self.switch_to_window(main_window_handle)

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

def ensure_dir(f):
    """ Ensure a a file exists and if not make the relevant path """
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

def compress(obj):
    """Outputs json without whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(',', ':'),
                      cls=CustomEncoder)

def pack_triples_numpy(triples):
    """Packs a list of triple indexes into a 2D numpy array."""
    if len(triples) == 0:
        return np.array([], dtype=np.int64)
    return np.stack(list(map(_transform_triple_numpy, triples)), axis=0)

def human_uuid():
    """Returns a good UUID for using as a human readable string."""
    return base64.b32encode(
        hashlib.sha1(uuid.uuid4().bytes).digest()).lower().strip('=')

def house_explosions():
    """
    Data from http://indexed.blogspot.com/2007/12/meltdown-indeed.html
    """
    chart = PieChart2D(int(settings.width * 1.7), settings.height)
    chart.add_data([10, 10, 30, 200])
    chart.set_pie_labels([
        'Budding Chemists',
        'Propane issues',
        'Meth Labs',
        'Attempts to escape morgage',
        ])
    chart.download('pie-house-explosions.png')

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

def classnameify(s):
  """
  Makes a classname
  """
  return ''.join(w if w in ACRONYMS else w.title() for w in s.split('_'))

def is_int(value):
    """Return `True` if ``value`` is an integer."""
    if isinstance(value, bool):
        return False
    try:
        int(value)
        return True
    except (ValueError, TypeError):
        return False

def array(self):
        """
        The underlying array of shape (N, L, I)
        """
        return numpy.array([self[sid].array for sid in sorted(self)])

def _extract_traceback(start):
    """
    SNAGGED FROM traceback.py

    RETURN list OF dicts DESCRIBING THE STACK TRACE
    """
    tb = sys.exc_info()[2]
    for i in range(start):
        tb = tb.tb_next
    return _parse_traceback(tb)

def launch_server():
    """Launches the django server at 127.0.0.1:8000
    """
    print(os.path.dirname(os.path.abspath(__file__)))
    cur_dir = os.getcwd()
    path = os.path.dirname(os.path.abspath(__file__))
    run = True
    os.chdir(path)
    os.system('python manage.py runserver --nostatic')
    os.chdir(cur_dir)

def column_exists(cr, table, column):
    """ Check whether a certain column exists """
    cr.execute(
        'SELECT count(attname) FROM pg_attribute '
        'WHERE attrelid = '
        '( SELECT oid FROM pg_class WHERE relname = %s ) '
        'AND attname = %s',
        (table, column))
    return cr.fetchone()[0] == 1

def student_t(degrees_of_freedom, confidence=0.95):
    """Return Student-t statistic for given DOF and confidence interval."""
    return scipy.stats.t.interval(alpha=confidence,
                                  df=degrees_of_freedom)[-1]

def computeFactorial(n):
    """
    computes factorial of n
    """
    sleep_walk(10)
    ret = 1
    for i in range(n):
        ret = ret * (i + 1)
    return ret

def _get_memoized_value(func, args, kwargs):
    """Used internally by memoize decorator to get/store function results"""
    key = (repr(args), repr(kwargs))

    if not key in func._cache_dict:
        ret = func(*args, **kwargs)
        func._cache_dict[key] = ret

    return func._cache_dict[key]

def _request_limit_reached(exception):
    """ Checks if exception was raised because of too many executed requests. (This is a temporal solution and
    will be changed in later package versions.)

    :param exception: Exception raised during download
    :type exception: Exception
    :return: True if exception is caused because too many requests were executed at once and False otherwise
    :rtype: bool
    """
    return isinstance(exception, requests.HTTPError) and \
        exception.response.status_code == requests.status_codes.codes.TOO_MANY_REQUESTS

def where_is(strings, pattern, n=1, lookup_func=re.match):
    """Return index of the nth match found of pattern in strings

    Parameters
    ----------
    strings: list of str
        List of strings

    pattern: str
        Pattern to be matched

    nth: int
        Number of times the match must happen to return the item index.

    lookup_func: callable
        Function to match each item in strings to the pattern, e.g., re.match or re.search.

    Returns
    -------
    index: int
        Index of the nth item that matches the pattern.
        If there are no n matches will return -1
    """
    count = 0
    for idx, item in enumerate(strings):
        if lookup_func(pattern, item):
            count += 1
            if count == n:
                return idx
    return -1

def indexTupleFromItem(self, treeItem): # TODO: move to BaseTreeItem?
        """ Return (first column model index, last column model index) tuple for a configTreeItem
        """
        if not treeItem:
            return (QtCore.QModelIndex(), QtCore.QModelIndex())

        if not treeItem.parentItem: # TODO: only necessary because of childNumber?
            return (QtCore.QModelIndex(), QtCore.QModelIndex())

        # Is there a bug in Qt in QStandardItemModel::indexFromItem?
        # It passes the parent in createIndex. TODO: investigate

        row =  treeItem.childNumber()
        return (self.createIndex(row, 0, treeItem),
                self.createIndex(row, self.columnCount() - 1, treeItem))

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

def begin_stream_loop(stream, poll_interval):
    """Start and maintain the streaming connection..."""
    while should_continue():
        try:
            stream.start_polling(poll_interval)
        except Exception as e:
            # Infinite restart
            logger.error("Exception while polling. Restarting in 1 second.", exc_info=True)
            time.sleep(1)

def reloader_thread(softexit=False):
    """If ``soft_exit`` is True, we use sys.exit(); otherwise ``os_exit``
    will be used to end the process.
    """
    while RUN_RELOADER:
        if code_changed():
            # force reload
            if softexit:
                sys.exit(3)
            else:
                os._exit(3)
        time.sleep(1)

def dashrepl(value):
    """
    Replace any non-word characters with a dash.
    """
    patt = re.compile(r'\W', re.UNICODE)
    return re.sub(patt, '-', value)

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

def coverage():
    """Run coverage tests."""
    # Note: coverage options are controlled by .coveragerc file
    install()
    test_setup()
    sh("%s -m coverage run %s" % (PYTHON, TEST_SCRIPT))
    sh("%s -m coverage report" % PYTHON)
    sh("%s -m coverage html" % PYTHON)
    sh("%s -m webbrowser -t htmlcov/index.html" % PYTHON)

def _compress_obj(obj, level):
    """Compress object to bytes.
    """
    return zlib.compress(pickle.dumps(obj, protocol=2), level)

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

def get_labels(labels):
    """Create unique labels."""
    label_u = unique_labels(labels)
    label_u_line = [i + "_line" for i in label_u]
    return label_u, label_u_line

def singleton_per_scope(_cls, _scope=None, _renew=False, *args, **kwargs):
    """Instanciate a singleton per scope."""

    result = None

    singletons = SINGLETONS_PER_SCOPES.setdefault(_scope, {})

    if _renew or _cls not in singletons:
        singletons[_cls] = _cls(*args, **kwargs)

    result = singletons[_cls]

    return result

def output_scores(self, name=None):
        """ Returns: N x #class scores, summed to one for each box."""
        return tf.nn.softmax(self.label_logits, name=name)

def can_access(self, user):
        """Return whether or not `user` can access a project.

        The project's is_ready field must be set for a user to access.

        """
        return self.class_.is_admin(user) or \
            self.is_ready and self.class_ in user.classes

def loadmat(filename):
    """This function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = sploadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def sort_by_modified(files_or_folders: list) -> list:
    """
    Sort files or folders by modified time

    Args:
        files_or_folders: list of files or folders

    Returns:
        list
    """
    return sorted(files_or_folders, key=os.path.getmtime, reverse=True)

def get_title(soup):
  """Given a soup, pick out a title"""
  if soup.title:
    return soup.title.string
  if soup.h1:
    return soup.h1.string
  return ''

def print(*a):
    """ print just one that returns what you give it instead of None """
    try:
        _print(*a)
        return a[0] if len(a) == 1 else a
    except:
        _print(*a)

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

def sparse_to_matrix(sparse):
    """
    Take a sparse (n,3) list of integer indexes of filled cells,
    turn it into a dense (m,o,p) matrix.

    Parameters
    -----------
    sparse: (n,3) int, index of filled cells

    Returns
    ------------
    dense: (m,o,p) bool, matrix of filled cells
    """

    sparse = np.asanyarray(sparse, dtype=np.int)
    if not util.is_shape(sparse, (-1, 3)):
        raise ValueError('sparse must be (n,3)!')

    shape = sparse.max(axis=0) + 1
    matrix = np.zeros(np.product(shape), dtype=np.bool)
    multiplier = np.array([np.product(shape[1:]), shape[2], 1])

    index = (sparse * multiplier).sum(axis=1)
    matrix[index] = True

    dense = matrix.reshape(shape)
    return dense

def is_unicode(string):
    """Validates that the object itself is some kinda string"""
    str_type = str(type(string))

    if str_type.find('str') > 0 or str_type.find('unicode') > 0:
        return True

    return False

def save_session(self, sid, session, namespace=None):
        """Store the user session for a client.

        The only difference with the :func:`socketio.Server.save_session`
        method is that when the ``namespace`` argument is not given the
        namespace associated with the class is used.
        """
        return self.server.save_session(
            sid, session, namespace=namespace or self.namespace)

def normalize_text(text, line_len=80, indent=""):
    """Wrap the text on the given line length."""
    return "\n".join(
        textwrap.wrap(
            text, width=line_len, initial_indent=indent, subsequent_indent=indent
        )
    )

def memory_read(self, start_position: int, size: int) -> memoryview:
        """
        Read and return a view of ``size`` bytes from memory starting at ``start_position``.
        """
        return self._memory.read(start_position, size)

def head_and_tail_print(self, n=5):
        """Display the first and last n elements of a DataFrame."""
        from IPython import display
        display.display(display.HTML(self._head_and_tail_table(n)))

def compare_dict(da, db):
    """
    Compare differencs from two dicts
    """
    sa = set(da.items())
    sb = set(db.items())
    
    diff = sa & sb
    return dict(sa - diff), dict(sb - diff)

def _from_dict(cls, _dict):
        """Initialize a KeyValuePair object from a json dictionary."""
        args = {}
        if 'key' in _dict:
            args['key'] = Key._from_dict(_dict.get('key'))
        if 'value' in _dict:
            args['value'] = Value._from_dict(_dict.get('value'))
        return cls(**args)

def save_hdf5(X, y, path):
    """Save data as a HDF5 file.

    Args:
        X (numpy or scipy sparse matrix): Data matrix
        y (numpy array): Target vector.
        path (str): Path to the HDF5 file to save data.
    """

    with h5py.File(path, 'w') as f:
        is_sparse = 1 if sparse.issparse(X) else 0
        f['issparse'] = is_sparse
        f['target'] = y

        if is_sparse:
            if not sparse.isspmatrix_csr(X):
                X = X.tocsr()

            f['shape'] = np.array(X.shape)
            f['data'] = X.data
            f['indices'] = X.indices
            f['indptr'] = X.indptr
        else:
            f['data'] = X

def center_eigenvalue_diff(mat):
    """Compute the eigvals of mat and then find the center eigval difference."""
    N = len(mat)
    evals = np.sort(la.eigvals(mat))
    diff = np.abs(evals[N/2] - evals[N/2-1])
    return diff

def _npiter(arr):
    """Wrapper for iterating numpy array"""
    for a in np.nditer(arr, flags=["refs_ok"]):
        c = a.item()
        if c is not None:
            yield c

def test(ctx, all=False, verbose=False):
    """Run the tests."""
    cmd = 'tox' if all else 'py.test'
    if verbose:
        cmd += ' -v'
    return ctx.run(cmd, pty=True).return_code

def _write_color_ansi (fp, text, color):
    """Colorize text with given color."""
    fp.write(esc_ansicolor(color))
    fp.write(text)
    fp.write(AnsiReset)

def enableEditing(self, enabled):
        """Enable the editing buttons to add/remove rows/columns and to edit the data.

        This method is also a slot.
        In addition, the data of model will be made editable,
        if the `enabled` parameter is true.

        Args:
            enabled (bool): This flag indicates, if the buttons
                shall be activated.

        """
        for button in self.buttons[1:]:
            button.setEnabled(enabled)
            if button.isChecked():
                button.setChecked(False)

        model = self.tableView.model()

        if model is not None:
            model.enableEditing(enabled)

def _re_raise_as(NewExc, *args, **kw):
    """Raise a new exception using the preserved traceback of the last one."""
    etype, val, tb = sys.exc_info()
    raise NewExc(*args, **kw), None, tb

def intToBin(i):
    """ Integer to two bytes """
    # devide in two parts (bytes)
    i1 = i % 256
    i2 = int(i / 256)
    # make string (little endian)
    return chr(i1) + chr(i2)

def p_postfix_expr(self, p):
        """postfix_expr : left_hand_side_expr
                        | left_hand_side_expr PLUSPLUS
                        | left_hand_side_expr MINUSMINUS
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = ast.UnaryOp(op=p[2], value=p[1], postfix=True)

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

def send_post(self, url, data, remove_header=None):
        """ Send a POST request
        """
        return self.send_request(method="post", url=url, data=data, remove_header=remove_header)

def current_zipfile():
    """A function to vend the current zipfile, if any"""
    if zipfile.is_zipfile(sys.argv[0]):
        fd = open(sys.argv[0], "rb")
        return zipfile.ZipFile(fd)

def array_sha256(a):
    """Create a SHA256 hash from a Numpy array."""
    dtype = str(a.dtype).encode()
    shape = numpy.array(a.shape)
    sha = hashlib.sha256()
    sha.update(dtype)
    sha.update(shape)
    sha.update(a.tobytes())
    return sha.hexdigest()

def parse_host_port (host_port):
    """Parse a host:port string into separate components."""
    host, port = urllib.splitport(host_port.strip())
    if port is not None:
        if urlutil.is_numeric_port(port):
            port = int(port)
    return host, port

def start(self, test_connection=True):
        """Starts connection to server if not existent.

        NO-OP if connection is already established.
        Makes ping-pong test as well if desired.

        """
        if self._context is None:
            self._logger.debug('Starting Client')
            self._context = zmq.Context()
            self._poll = zmq.Poller()
            self._start_socket()
            if test_connection:
                self.test_ping()

def yn_prompt(msg, default=True):
    """
    Prompts the user for yes or no.
    """
    ret = custom_prompt(msg, ["y", "n"], "y" if default else "n")
    if ret == "y":
        return True
    return False

def _pooling_output_shape(input_shape, pool_size=(2, 2),
                          strides=None, padding='VALID'):
  """Helper: compute the output shape for the pooling layer."""
  dims = (1,) + pool_size + (1,)  # NHWC
  spatial_strides = strides or (1,) * len(pool_size)
  strides = (1,) + spatial_strides + (1,)
  pads = padtype_to_pads(input_shape, dims, strides, padding)
  operand_padded = onp.add(input_shape, onp.add(*zip(*pads)))
  t = onp.floor_divide(onp.subtract(operand_padded, dims), strides) + 1
  return tuple(t)

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

def trigger_fullscreen_action(self, fullscreen):
        """
        Toggle fullscreen from outside the GUI,
        causes the GUI to updated and run all its actions.
        """
        action = self.action_group.get_action('fullscreen')
        action.set_active(fullscreen)

def clean_strings(iterable):
    """
    Take a list of strings and clear whitespace 
    on each one. If a value in the list is not a 
    string pass it through untouched.

    Args:
        iterable: mixed list

    Returns: 
        mixed list
    """
    retval = []
    for val in iterable:
        try:
            retval.append(val.strip())
        except(AttributeError):
            retval.append(val)
    return retval

def random_name_gen(size=6):
    """Generate a random python attribute name."""

    return ''.join(
        [random.choice(string.ascii_uppercase)] +
        [random.choice(string.ascii_uppercase + string.digits) for i in range(size - 1)]
    ) if size > 0 else ''

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def init_rotating_logger(level, logfile, max_files, max_bytes):
  """Initializes a rotating logger

  It also makes sure that any StreamHandler is removed, so as to avoid stdout/stderr
  constipation issues
  """
  logging.basicConfig()

  root_logger = logging.getLogger()
  log_format = "[%(asctime)s] [%(levelname)s] %(filename)s: %(message)s"

  root_logger.setLevel(level)
  handler = RotatingFileHandler(logfile, maxBytes=max_bytes, backupCount=max_files)
  handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))
  root_logger.addHandler(handler)

  for handler in root_logger.handlers:
    root_logger.debug("Associated handlers - " + str(handler))
    if isinstance(handler, logging.StreamHandler):
      root_logger.debug("Removing StreamHandler: " + str(handler))
      root_logger.handlers.remove(handler)

def datetime_to_ms(dt):
    """
    Converts a datetime to a millisecond accuracy timestamp
    """
    seconds = calendar.timegm(dt.utctimetuple())
    return seconds * 1000 + int(dt.microsecond / 1000)

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

def encode_batch(self, inputBatch):
        """Encodes a whole batch of input arrays, without learning."""
        X      = inputBatch
        encode = self.encode
        Y      = np.array([ encode(x) for x in X])
        return Y

def hide(self):
        """Hide the window."""
        self.tk.withdraw()
        self._visible = False
        if self._modal:
            self.tk.grab_release()

def dt2str(dt, flagSeconds=True):
    """Converts datetime object to str if not yet an str."""
    if isinstance(dt, str):
        return dt
    return dt.strftime(_FMTS if flagSeconds else _FMT)

def calculate_size(name, timeout):
    """ Calculates the request payload size"""
    data_size = 0
    data_size += calculate_size_str(name)
    data_size += LONG_SIZE_IN_BYTES
    return data_size

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

def drop_bad_characters(text):
    """Takes a text and drops all non-printable and non-ascii characters and
    also any whitespace characters that aren't space.

    :arg str text: the text to fix

    :returns: text with all bad characters dropped

    """
    # Strip all non-ascii and non-printable characters
    text = ''.join([c for c in text if c in ALLOWED_CHARS])
    return text

def downgrade():
    """Downgrade database."""
    op.drop_table('transaction')
    if op._proxy.migration_context.dialect.supports_sequences:
        op.execute(DropSequence(Sequence('transaction_id_seq')))

def _records_commit(record_ids):
    """Commit all records."""
    for record_id in record_ids:
        record = Record.get_record(record_id)
        record.commit()

def __call__(self, r):
        """Update the request headers."""
        r.headers['Authorization'] = 'JWT {jwt}'.format(jwt=self.token)
        return r

def into2dBlocks(arr, n0, n1):
    """
    similar to blockshaped
    but splits an array into n0*n1 blocks
    """
    s0, s1 = arr.shape
    b = blockshaped(arr, s0// n0, s1// n1)
    return b.reshape(n0, n1, *b.shape[1:])

def ver_to_tuple(value):
    """
    Convert version like string to a tuple of integers.
    """
    return tuple(int(_f) for _f in re.split(r'\D+', value) if _f)

def bootstrap_indexes(data, n_samples=10000):
    """
Given data points data, where axis 0 is considered to delineate points, return
an generator for sets of bootstrap indexes. This can be used as a list
of bootstrap indexes (with list(bootstrap_indexes(data))) as well.
    """
    for _ in xrange(n_samples):
        yield randint(data.shape[0], size=(data.shape[0],))

def clean_float(v):
    """Remove commas from a float"""

    if v is None or not str(v).strip():
        return None

    return float(str(v).replace(',', ''))

def conf(self):
        """Generate the Sphinx `conf.py` configuration file

        Returns:
            (str): the contents of the `conf.py` file.
        """
        return self.env.get_template('conf.py.j2').render(
            metadata=self.metadata,
            package=self.package)

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

def cycle_focus(self):
        """
        Cycle through all windows.
        """
        windows = self.windows()
        new_index = (windows.index(self.active_window) + 1) % len(windows)
        self.active_window = windows[new_index]

def has_virtualenv(self):
        """
        Returns true if the virtualenv tool is installed.
        """
        with self.settings(warn_only=True):
            ret = self.run_or_local('which virtualenv').strip()
            return bool(ret)

def update(table, values, where=(), **kwargs):
    """Convenience wrapper for database UPDATE."""
    where = dict(where, **kwargs).items()
    sql, args = makeSQL("UPDATE", table, values=values, where=where)
    return execute(sql, args).rowcount

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

def click(self):
        """Click the element

        :returns: page element instance
        """
        try:
            self.wait_until_clickable().web_element.click()
        except StaleElementReferenceException:
            # Retry if element has changed
            self.web_element.click()
        return self

def conv3x3(in_channels, out_channels, stride=1):
    """
    3x3 convolution with padding.
    Original code has had bias turned off, because Batch Norm would remove the bias either way
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

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

def format_time(time):
    """ Formats the given time into HH:MM:SS """
    h, r = divmod(time / 1000, 3600)
    m, s = divmod(r, 60)

    return "%02d:%02d:%02d" % (h, m, s)

def getExperiments(uuid: str):
    """ list active (running or completed) experiments"""
    return jsonify([x.deserialize() for x in Experiment.query.all()])

def ExpireObject(self, key):
    """Expire a specific object from cache."""
    node = self._hash.pop(key, None)
    if node:
      self._age.Unlink(node)
      self.KillObject(node.data)

      return node.data

def do_forceescape(value):
    """Enforce HTML escaping.  This will probably double escape variables."""
    if hasattr(value, '__html__'):
        value = value.__html__()
    return escape(unicode(value))

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

def format_vars(args):
    """Format the given vars in the form: 'flag=value'"""
    variables = []
    for key, value in args.items():
        if value:
            variables += ['{0}={1}'.format(key, value)]
    return variables

def generic_add(a, b):
    """Simple function to add two numbers"""
    logger.debug('Called generic_add({}, {})'.format(a, b))
    return a + b

def one_hot_encoding(input_tensor, num_labels):
    """ One-hot encode labels from input """
    xview = input_tensor.view(-1, 1).to(torch.long)

    onehot = torch.zeros(xview.size(0), num_labels, device=input_tensor.device, dtype=torch.float)
    onehot.scatter_(1, xview, 1)
    return onehot.view(list(input_tensor.shape) + [-1])

def align_file_position(f, size):
    """ Align the position in the file to the next block of specified size """
    align = (size - 1) - (f.tell() % size)
    f.seek(align, 1)

def close(self):
		"""
		Send a terminate request and then disconnect from the serial device.
		"""
		if self._initialized:
			self.stop()
		self.logged_in = False
		return self.serial_h.close()

def unique_list_dicts(dlist, key):
    """Return a list of dictionaries which are sorted for only unique entries.

    :param dlist:
    :param key:
    :return list:
    """

    return list(dict((val[key], val) for val in dlist).values())

def yticks(self):
        """Compute the yticks labels of this grid, used for plotting the y-axis ticks when visualizing a regular"""
        return np.linspace(np.min(self[:, 0]), np.max(self[:, 0]), 4)

def wait_until_exit(self):
        """ Wait until all the threads are finished.

        """
        [t.join() for t in self.threads]

        self.threads = list()

def load(file_object):
  """
  Deserializes Java primitive data and objects serialized by ObjectOutputStream
  from a file-like object.
  """
  marshaller = JavaObjectUnmarshaller(file_object)
  marshaller.add_transformer(DefaultObjectTransformer())
  return marshaller.readObject()

def roll_dice():
    """
    Roll a die.

    :return: None
    """
    sums = 0  # will return the sum of the roll calls.
    while True:
        roll = random.randint(1, 6)
        sums += roll
        if(input("Enter y or n to continue: ").upper()) == 'N':
            print(sums)  # prints the sum of the roll calls
            break

def tab(self, output):
        """Output data in excel-compatible tab-delimited format"""
        import csv
        csvwriter = csv.writer(self.outfile, dialect=csv.excel_tab)
        csvwriter.writerows(output)

def rank(tensor: BKTensor) -> int:
    """Return the number of dimensions of a tensor"""
    if isinstance(tensor, np.ndarray):
        return len(tensor.shape)

    return len(tensor[0].size())

def save_list(key, *values):
    """Convert the given list of parameters to a JSON object.

    JSON object is of the form:
    { key: [values[0], values[1], ... ] },
    where values represent the given list of parameters.

    """
    return json.dumps({key: [_get_json(value) for value in values]})

def excel_datetime(timestamp, epoch=None):
    """Return datetime object from timestamp in Excel serial format.

    Convert LSM time stamps.

    >>> excel_datetime(40237.029999999795)
    datetime.datetime(2010, 2, 28, 0, 43, 11, 999982)

    """
    if epoch is None:
        epoch = datetime.datetime.fromordinal(693594)
    return epoch + datetime.timedelta(timestamp)

def ask_folder(message='Select folder.', default='', title=''):
    """
    A dialog to get a directory name.
    Returns the name of a directory, or None if user chose to cancel.
    If the "default" argument specifies a directory name, and that
    directory exists, then the dialog box will start with that directory.

    :param message: message to be displayed.
    :param title: window title
    :param default: default folder path
    :rtype: None or string
    """
    return backend_api.opendialog("ask_folder", dict(message=message, default=default, title=title))

def convert_types(cls, value):
        """
        Takes a value from MSSQL, and converts it to a value that's safe for
        JSON/Google Cloud Storage/BigQuery.
        """
        if isinstance(value, decimal.Decimal):
            return float(value)
        else:
            return value

def is_array(type_):
    """returns True, if type represents C++ array type, False otherwise"""
    nake_type = remove_alias(type_)
    nake_type = remove_reference(nake_type)
    nake_type = remove_cv(nake_type)
    return isinstance(nake_type, cpptypes.array_t)

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

def pprint(obj, verbose=False, max_width=79, newline='\n'):
    """
    Like `pretty` but print to stdout.
    """
    printer = RepresentationPrinter(sys.stdout, verbose, max_width, newline)
    printer.pretty(obj)
    printer.flush()
    sys.stdout.write(newline)
    sys.stdout.flush()

def initialize_worker(self, process_num=None):
        """
        reinitialize consumer for process in multiprocesing
        """
        self.initialize(self.grid, self.num_of_paths, self.seed)

def get_unicode_str(obj):
    """Makes sure obj is a unicode string."""
    if isinstance(obj, six.text_type):
        return obj
    if isinstance(obj, six.binary_type):
        return obj.decode("utf-8", errors="ignore")
    return six.text_type(obj)

def safe_int_conv(number):
    """Safely convert a single number to integer."""
    try:
        return int(np.array(number).astype(int, casting='safe'))
    except TypeError:
        raise ValueError('cannot safely convert {} to integer'.format(number))

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

def mouseMoveEvent(self, event):
        """ Handle the mouse move event for a drag operation.

        """
        self.declaration.mouse_move_event(event)
        super(QtGraphicsView, self).mouseMoveEvent(event)

def dispatch(self):
    """Wraps the dispatch method to add session support."""
    try:
      webapp2.RequestHandler.dispatch(self)
    finally:
      self.session_store.save_sessions(self.response)

def non_increasing(values):
    """True if values are not increasing."""
    return all(x >= y for x, y in zip(values, values[1:]))

def scaled_fft(fft, scale=1.0):
    """
    Produces a nicer graph, I'm not sure if this is correct
    """
    data = np.zeros(len(fft))
    for i, v in enumerate(fft):
        data[i] = scale * (i * v) / NUM_SAMPLES

    return data

def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and 1-hot labels."""
  return 100.0 - (
      100.0 *
      np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
      predictions.shape[0])

def timedelta2millisecond(td):
    """Get milliseconds from a timedelta."""
    milliseconds = td.days * 24 * 60 * 60 * 1000
    milliseconds += td.seconds * 1000
    milliseconds += td.microseconds / 1000
    return milliseconds

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

def check_many(self, domains):
        """
        Check availability for a number of domains. Returns a dictionary
        mapping the domain names to their statuses as a string
        ("active"/"free").
        """
        return dict((item.domain, item.status) for item in self.check_domain_request(domains))

def update(dct, dct_merge):
    """Recursively merge dicts."""
    for key, value in dct_merge.items():
        if key in dct and isinstance(dct[key], dict):
            dct[key] = update(dct[key], value)
        else:
            dct[key] = value
    return dct

def load_model_from_package(name, **overrides):
    """Load a model from an installed package."""
    cls = importlib.import_module(name)
    return cls.load(**overrides)

def experiment_property(prop):
    """Get a property of the experiment by name."""
    exp = experiment(session)
    p = getattr(exp, prop)
    return success_response(field=prop, data=p, request_type=prop)

def _histplot_bins(column, bins=100):
    """Helper to get bins for histplot."""
    col_min = np.min(column)
    col_max = np.max(column)
    return range(col_min, col_max + 2, max((col_max - col_min) // bins, 1))

def jupytext_cli(args=None):
    """Entry point for the jupytext script"""
    try:
        jupytext(args)
    except (ValueError, TypeError, IOError) as err:
        sys.stderr.write('[jupytext] Error: ' + str(err) + '\n')
        exit(1)

def is_strict_numeric(n: Node) -> bool:
    """ numeric denotes typed literals with datatypes xsd:integer, xsd:decimal, xsd:float, and xsd:double. """
    return is_typed_literal(n) and cast(Literal, n).datatype in [XSD.integer, XSD.decimal, XSD.float, XSD.double]

def split_addresses(email_string_list):
    """
    Converts a string containing comma separated email addresses
    into a list of email addresses.
    """
    return [f for f in [s.strip() for s in email_string_list.split(",")] if f]

def clear_table(dbconn, table_name):
    """
    Delete all rows from a table
    :param dbconn: data base connection
    :param table_name: name of the table
    :return:
    """
    cur = dbconn.cursor()
    cur.execute("DELETE FROM '{name}'".format(name=table_name))
    dbconn.commit()

def resource_property(klass, name, **kwargs):
    """Builds a resource object property."""
    klass.PROPERTIES[name] = kwargs

    def getter(self):
        return getattr(self, '_%s' % name, kwargs.get('default', None))

    if kwargs.get('readonly', False):
        setattr(klass, name, property(getter))
    else:
        def setter(self, value):
            setattr(self, '_%s' % name, value)
        setattr(klass, name, property(getter, setter))

def norm_vec(vector):
    """Normalize the length of a vector to one"""
    assert len(vector) == 3
    v = np.array(vector)
    return v/np.sqrt(np.sum(v**2))

def get_bucket_page(page):
    """
    Returns all the keys in a s3 bucket paginator page.
    """
    key_list = page.get('Contents', [])
    logger.debug("Retrieving page with {} keys".format(
        len(key_list),
    ))
    return dict((k.get('Key'), k) for k in key_list)

def install_handle_input(self):
        """Install the hook."""
        self.pointer = self.get_fptr()

        self.hooked = ctypes.windll.user32.SetWindowsHookExA(
            13,
            self.pointer,
            ctypes.windll.kernel32.GetModuleHandleW(None),
            0
        )
        if not self.hooked:
            return False
        return True

def main(argv=None):
    """Main command line interface."""

    if argv is None:
        argv = sys.argv[1:]

    cli = CommandLineTool()
    return cli.run(argv)

def connect(self, A, B, distance=1):
        """Add a link from A and B of given distance, and also add the inverse
        link if the graph is undirected."""
        self.connect1(A, B, distance)
        if not self.directed: self.connect1(B, A, distance)

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

def render(template, context):
        """Wrapper to the jinja2 render method from a template file

        Parameters
        ----------
        template : str
            Path to template file.
        context : dict
            Dictionary with kwargs context to populate the template
        """

        path, filename = os.path.split(template)

        return jinja2.Environment(
            loader=jinja2.FileSystemLoader(path or './')
        ).get_template(filename).render(context)

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

def _add_params_docstring(params):
    """ Add params to doc string
    """
    p_string = "\nAccepts the following paramters: \n"
    for param in params:
         p_string += "name: %s, required: %s, description: %s \n" % (param['name'], param['required'], param['description'])
    return p_string

def do_stc_disconnectall(self, s):
        """Remove connections to all chassis (test ports) in this session."""
        if self._not_joined():
            return
        try:
            self._stc.disconnectall()
        except resthttp.RestHttpError as e:
            print(e)
            return
        print('OK')

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

def reset(self):
        """Reset the instance

        - reset rows and header
        """

        self._hline_string = None
        self._row_size = None
        self._header = []
        self._rows = []

def recarray(self):
        """Returns data as :class:`numpy.recarray`."""
        return numpy.rec.fromrecords(self.records, names=self.names)

def minify(path):
    """
    Load a javascript file and minify.

    Parameters
    ------------
    path: str, path of resource
    """

    if 'http' in path:
        data = requests.get(path).content.decode(
            'ascii', errors='ignore')
    else:
        with open(path, 'rb') as f:
            # some of these assholes use unicode spaces -_-
            data = f.read().decode('ascii',
                                   errors='ignore')
    # don't re- minify
    if '.min.' in path:
        return data

    try:
        return jsmin.jsmin(data)
    except BaseException:
        return data

def __unroll(self, rolled):
        """Converts parameter matrices into an array."""
        return np.array(np.concatenate([matrix.flatten() for matrix in rolled], axis=1)).reshape(-1)

def getlines(filename, module_globals=None):
    """Get the lines for a file from the cache.
    Update the cache if it doesn't contain an entry for this file already."""

    if filename in cache:
        return cache[filename][2]

    try:
        return updatecache(filename, module_globals)
    except MemoryError:
        clearcache()
        return []

def documentation(self):
        """
        Get the documentation that the server sends for the API.
        """
        newclient = self.__class__(self.session, self.root_url)
        return newclient.get_raw('/')

def is_text(obj, name=None):
    """
    returns True if object is text-like
    """
    try:  # python2
        ans = isinstance(obj, basestring)
    except NameError:  # python3
        ans = isinstance(obj, str)
    if name:
        print("is_text: (%s) %s = %s" % (ans, name, obj.__class__),
              file=sys.stderr)
    return ans

def _decode(self, obj, context):
        """
        Get the python representation of the obj
        """
        return b''.join(map(int2byte, [c + 0x60 for c in bytearray(obj)])).decode("utf8")

def allele_clusters(dists, t=0.025):
    """Flat clusters from distance matrix

    Args:
        dists (numpy.array): pdist distance matrix
        t (float): fcluster (tree cutting) distance threshold

    Returns:
        dict of lists: cluster number to list of indices of distances in cluster
    """
    clusters = fcluster(linkage(dists), 0.025, criterion='distance')
    cluster_idx = defaultdict(list)
    for idx, cl in enumerate(clusters):
        cluster_idx[cl].append(idx)
    return cluster_idx

def get_member(thing_obj, member_string):
    """Get a member from an object by (string) name"""
    mems = {x[0]: x[1] for x in inspect.getmembers(thing_obj)}
    if member_string in mems:
        return mems[member_string]

def __init__(self, iterable):
        """Initialize the cycle with some iterable."""
        self._values = []
        self._iterable = iterable
        self._initialized = False
        self._depleted = False
        self._offset = 0

def refresh(self, document):
		""" Load a new copy of a document from the database.  does not
			replace the old one """
		try:
			old_cache_size = self.cache_size
			self.cache_size = 0
			obj = self.query(type(document)).filter_by(mongo_id=document.mongo_id).one()
		finally:
			self.cache_size = old_cache_size
		self.cache_write(obj)
		return obj

def comments(self):
    """The AST comments."""
    if self._comments is None:
      self._comments = [c for c in self.grammar.children if c.is_type(TokenType.comment)]
    return self._comments

def get_url(self, routename, **kargs):
        """ Return a string that matches a named route """
        return '/' + self.routes.build(routename, **kargs).split(';', 1)[1]

def disable(self):
        """
        Disable the button, if in non-expert mode;
        unset its activity flag come-what-may.
        """
        if not self._expert:
            self.config(state='disable')
        self._active = False

def assert_is_not(expected, actual, message=None, extra=None):
    """Raises an AssertionError if expected is actual."""
    assert expected is not actual, _assert_fail_message(
        message, expected, actual, "is", extra
    )

def convert_args_to_sets(f):
    """
    Converts all args to 'set' type via self.setify function.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        args = (setify(x) for x in args)
        return f(*args, **kwargs)
    return wrapper

def find_frequencies(data, freq=44100, bits=16):
    """Convert audio data into a frequency-amplitude table using fast fourier
    transformation.

    Return a list of tuples (frequency, amplitude).

    Data should only contain one channel of audio.
    """
    # Fast fourier transform
    n = len(data)
    p = _fft(data)
    uniquePts = numpy.ceil((n + 1) / 2.0)

    # Scale by the length (n) and square the value to get the amplitude
    p = [(abs(x) / float(n)) ** 2 * 2 for x in p[0:uniquePts]]
    p[0] = p[0] / 2
    if n % 2 == 0:
        p[-1] = p[-1] / 2

    # Generate the frequencies and zip with the amplitudes
    s = freq / float(n)
    freqArray = numpy.arange(0, uniquePts * s, s)
    return zip(freqArray, p)

def __getattr__(self, name):
        """Return wrapper to named api method."""
        return functools.partial(self._obj.request, self._api_prefix + name)

def get_connection(self, host, port, db):
        """
        Returns a ``StrictRedis`` connection instance.
        """
        return redis.StrictRedis(
            host=host,
            port=port,
            db=db,
            decode_responses=True
        )

def drop_empty(rows):
    """Transpose the columns into rows, remove all of the rows that are empty after the first cell, then
    transpose back. The result is that columns that have a header but no data in the body are removed, assuming
    the header is the first row. """
    return zip(*[col for col in zip(*rows) if bool(filter(bool, col[1:]))])

def head(self, path, query=None, data=None, redirects=True):
        """
        HEAD request wrapper for :func:`request()`
        """
        return self.request('HEAD', path, query, None, redirects)

def from_array(cls, arr):
        """Convert a structured NumPy array into a Table."""
        return cls().with_columns([(f, arr[f]) for f in arr.dtype.names])

def serve_dtool_directory(directory, port):
    """Serve the datasets in a directory over HTTP."""
    os.chdir(directory)
    server_address = ("localhost", port)
    httpd = DtoolHTTPServer(server_address, DtoolHTTPRequestHandler)
    httpd.serve_forever()

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

def byteswap(data, word_size=4):
    """ Swap the byte-ordering in a packet with N=4 bytes per word
    """
    return reduce(lambda x,y: x+''.join(reversed(y)), chunks(data, word_size), '')

def get_line_ending(line):
    """Return line ending."""
    non_whitespace_index = len(line.rstrip()) - len(line)
    if not non_whitespace_index:
        return ''
    else:
        return line[non_whitespace_index:]

def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def copyFile(input, output, replace=None):
    """Copy a file whole from input to output."""

    _found = findFile(output)
    if not _found or (_found and replace):
        shutil.copy2(input, output)

def files_changed():
    """
    Return the list of file changed in the current branch compared to `master`
    """
    with chdir(get_root()):
        result = run_command('git diff --name-only master...', capture='out')
    changed_files = result.stdout.splitlines()

    # Remove empty lines
    return [f for f in changed_files if f]

def _is_readable(self, obj):
        """Check if the argument is a readable file-like object."""
        try:
            read = getattr(obj, 'read')
        except AttributeError:
            return False
        else:
            return is_method(read, max_arity=1)

def setup_cache(app: Flask, cache_config) -> Optional[Cache]:
    """Setup the flask-cache on a flask app"""
    if cache_config and cache_config.get('CACHE_TYPE') != 'null':
        return Cache(app, config=cache_config)

    return None

def terminate(self):
        """Terminate the pool immediately."""
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

def _replace_service_arg(self, name, index, args):
        """ Replace index in list with service """
        args[index] = self.get_instantiated_service(name)

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

def rdist(x, y):
    """Reduced Euclidean distance.

    Parameters
    ----------
    x: array of shape (embedding_dim,)
    y: array of shape (embedding_dim,)

    Returns
    -------
    The squared euclidean distance between x and y
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2

    return result

def check_output(args, env=None, sp=subprocess):
    """Call an external binary and return its stdout."""
    log.debug('calling %s with env %s', args, env)
    output = sp.check_output(args=args, env=env)
    log.debug('output: %r', output)
    return output

def yank(event):
    """
    Paste before cursor.
    """
    event.current_buffer.paste_clipboard_data(
        event.cli.clipboard.get_data(), count=event.arg, paste_mode=PasteMode.EMACS)

def boolean(flag):
    """
    Convert string in boolean
    """
    s = flag.lower()
    if s in ('1', 'yes', 'true'):
        return True
    elif s in ('0', 'no', 'false'):
        return False
    raise ValueError('Unknown flag %r' % s)

def handle_request_parsing_error(err, req, schema, error_status_code, error_headers):
    """webargs error handler that uses Flask-RESTful's abort function to return
    a JSON error response to the client.
    """
    abort(error_status_code, errors=err.messages)

def image_format(value):
    """
    Confirms that the uploaded image is of supported format.

    Args:
        value (File): The file with an `image` property containing the image

    Raises:
        django.forms.ValidationError

    """

    if value.image.format.upper() not in constants.ALLOWED_IMAGE_FORMATS:
        raise ValidationError(MESSAGE_INVALID_IMAGE_FORMAT)

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

def _hide_tick_lines_and_labels(axis):
    """
    Set visible property of ticklines and ticklabels of an axis to False
    """
    for item in axis.get_ticklines() + axis.get_ticklabels():
        item.set_visible(False)

def _write_separator(self):
        """
        Inserts a horizontal (commented) line tot the generated code.
        """
        tmp = self._page_width - ((4 * self.__indent_level) + 2)
        self._write_line('# ' + ('-' * tmp))

def _random_x(self):
        """If the database is empty, generate a random vector."""
        return (tuple(random.random() for _ in range(self.fmodel.dim_x)),)

def clone(src, **kwargs):
    """Clones object with optionally overridden fields"""
    obj = object.__new__(type(src))
    obj.__dict__.update(src.__dict__)
    obj.__dict__.update(kwargs)
    return obj

def _insert_row(self, i, index):
        """
        Insert a new row in the Series.

        :param i: index location to insert
        :param index: index value to insert into the index list
        :return: nothing
        """
        if i == len(self._index):
            self._add_row(index)
        else:
            self._index.insert(i, index)
            self._data.insert(i, None)

def loadb(b):
    """Deserialize ``b`` (instance of ``bytes``) to a Python object."""
    assert isinstance(b, (bytes, bytearray))
    return std_json.loads(b.decode('utf-8'))

def getDimensionForImage(filename, maxsize):
    """Return scaled image size in (width, height) format.
    The scaling preserves the aspect ratio.
    If PIL is not found returns None."""
    try:
        from PIL import Image
    except ImportError:
        return None
    img = Image.open(filename)
    width, height = img.size
    if width > maxsize[0] or height > maxsize[1]:
        img.thumbnail(maxsize)
        out.info("Downscaled display size from %s to %s" % ((width, height), img.size))
    return img.size

def fetch(table, cols="*", where=(), group="", order=(), limit=(), **kwargs):
    """Convenience wrapper for database SELECT and fetch all."""
    return select(table, cols, where, group, order, limit, **kwargs).fetchall()

def tinsel(to_patch, module_name, decorator=mock_decorator):
    """
    Decorator for simple in-place decorator mocking for tests

    Args:
        to_patch: the string path of the function to patch
        module_name: complete string path of the module to reload
        decorator (optional): replacement decorator. By default a pass-through
            will be used.

    Returns:
        A wrapped test function, during the context of execution the specified
        path is patched.

    """
    def fn_decorator(function):
        def wrapper(*args, **kwargs):
            with patch(to_patch, decorator):
                m = importlib.import_module(module_name)
                reload(m)
                function(*args, **kwargs)

            reload(m)
        return wrapper
    return fn_decorator

def s3_get(url: str, temp_file: IO) -> None:
    """Pull a file directly from S3."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)

def inverse_jacobian(self, maps):
        """Returns the Jacobian for transforming mass1 and mass2 to
        mchirp and eta.
        """
        m1 = maps[parameters.mass1]
        m2 = maps[parameters.mass2]
        mchirp = conversions.mchirp_from_mass1_mass2(m1, m2)
        eta = conversions.eta_from_mass1_mass2(m1, m2)
        return -1. * mchirp / eta**(6./5)

def test():
    """ Run all Tests [nose] """

    command = 'nosetests --with-coverage --cover-package=pwnurl'
    status = subprocess.call(shlex.split(command))
    sys.exit(status)

def setdefault(obj, field, default):
    """Set an object's field to default if it doesn't have a value"""
    setattr(obj, field, getattr(obj, field, default))

def to_bool(value):
    # type: (Any) -> bool
    """
    Convert a value into a bool but handle "truthy" strings eg, yes, true, ok, y
    """
    if isinstance(value, _compat.string_types):
        return value.upper() in ('Y', 'YES', 'T', 'TRUE', '1', 'OK')
    return bool(value)

def close_all():
    """Close all open/active plotters"""
    for key, p in _ALL_PLOTTERS.items():
        p.close()
    _ALL_PLOTTERS.clear()
    return True

def _is_image_sequenced(image):
    """Determine if the image is a sequenced image."""
    try:
        image.seek(1)
        image.seek(0)
        result = True
    except EOFError:
        result = False

    return result

def direct2dDistance(self, point):
        """consider the distance between two mapPoints, ignoring all terrain, pathing issues"""
        if not isinstance(point, MapPoint): return 0.0
        return  ((self.x-point.x)**2 + (self.y-point.y)**2)**(0.5) # simple distance formula

def is_git_repo():
    """Check whether the current folder is a Git repo."""
    cmd = "git", "rev-parse", "--git-dir"
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def _swap_curly(string):
    """Swap single and double curly brackets"""
    return (
        string.replace('{{ ', '{{').replace('{{', '\x00').replace('{', '{{')
        .replace('\x00', '{').replace(' }}', '}}').replace('}}', '\x00')
        .replace('}', '}}').replace('\x00', '}')
    )

def add_colons(s):
    """Add colons after every second digit.

    This function is used in functions to prettify serials.

    >>> add_colons('teststring')
    'te:st:st:ri:ng'
    """
    return ':'.join([s[i:i + 2] for i in range(0, len(s), 2)])

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

def _strvar(a, prec='{:G}'):
    r"""Return variable as a string to print, with given precision."""
    return ' '.join([prec.format(i) for i in np.atleast_1d(a)])

def save(variable, filename):
    """Save variable on given path using Pickle
    
    Args:
        variable: what to save
        path (str): path of the output
    """
    fileObj = open(filename, 'wb')
    pickle.dump(variable, fileObj)
    fileObj.close()

def clear(self):
        """Remove all items."""
        self._fwdm.clear()
        self._invm.clear()
        self._sntl.nxt = self._sntl.prv = self._sntl

def stop_refresh(self):
        """Stop redrawing the canvas at the previously set timed interval.
        """
        self.logger.debug("stopping timed refresh")
        self.rf_flags['done'] = True
        self.rf_timer.clear()

def calculate_bounding_box_from_image(im, curr_page):
    """This function uses a PIL routine to get the bounding box of the rendered
    image."""
    xMax, y_max = im.size
    bounding_box = im.getbbox() # note this uses ltrb convention
    if not bounding_box:
        #print("\nWarning: could not calculate a bounding box for this page."
        #      "\nAn empty page is assumed.", file=sys.stderr)
        bounding_box = (xMax/2, y_max/2, xMax/2, y_max/2)

    bounding_box = list(bounding_box) # make temporarily mutable

    # Compensate for reversal of the image y convention versus PDF.
    bounding_box[1] = y_max - bounding_box[1]
    bounding_box[3] = y_max - bounding_box[3]

    full_page_box = curr_page.mediaBox # should have been set already to chosen box

    # Convert pixel units to PDF's bp units.
    convert_x = float(full_page_box.getUpperRight_x()
                     - full_page_box.getLowerLeft_x()) / xMax
    convert_y = float(full_page_box.getUpperRight_y()
                     - full_page_box.getLowerLeft_y()) / y_max

    # Get final box; note conversion to lower-left point, upper-right point format.
    final_box = [
        bounding_box[0] * convert_x,
        bounding_box[3] * convert_y,
        bounding_box[2] * convert_x,
        bounding_box[1] * convert_y]

    return final_box

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

def _removeTags(tags, objects):
    """ Removes tags from objects """
    for t in tags:
        for o in objects:
            o.tags.remove(t)

    return True

def indent(text, amount, ch=' '):
    """Indents a string by the given amount of characters."""
    padding = amount * ch
    return ''.join(padding+line for line in text.splitlines(True))

def _get_printable_columns(columns, row):
    """Return only the part of the row which should be printed.
    """
    if not columns:
        return row

    # Extract the column values, in the order specified.
    return tuple(row[c] for c in columns)

def start():
    """Starts the web server."""
    global app
    bottle.run(app, host=conf.WebHost, port=conf.WebPort,
               debug=conf.WebAutoReload, reloader=conf.WebAutoReload,
               quiet=conf.WebQuiet)

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

def _get_random_id():
    """ Get a random (i.e., unique) string identifier"""
    symbols = string.ascii_uppercase + string.ascii_lowercase + string.digits
    return ''.join(random.choice(symbols) for _ in range(15))

def get_access_datetime(filepath):
    """
    Get the last time filepath was accessed.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    access_datetime : datetime.datetime
    """
    import tzlocal
    tz = tzlocal.get_localzone()
    mtime = datetime.fromtimestamp(os.path.getatime(filepath))
    return mtime.replace(tzinfo=tz)

def _get_background_color(self):
        """Returns background color rgb tuple of right line"""

        color = self.cell_attributes[self.key]["bgcolor"]
        return tuple(c / 255.0 for c in color_pack2rgb(color))

def check_empty_dict(GET_dict):
    """
    Returns True if the GET querstring contains on values, but it can contain
    empty keys.
    This is better than doing not bool(request.GET) as an empty key will return
    True
    """
    empty = True
    for k, v in GET_dict.items():
        # Don't disable on p(age) or 'all' GET param
        if v and k != 'p' and k != 'all':
            empty = False
    return empty

def listunion(ListOfLists):
    """
    Take the union of a list of lists.

    Take a Python list of Python lists::

            [[l11,l12, ...], [l21,l22, ...], ... , [ln1, ln2, ...]]

    and return the aggregated list::

            [l11,l12, ..., l21, l22 , ...]

    For a list of two lists, e.g. `[a, b]`, this is like::

            a.extend(b)

    **Parameters**

            **ListOfLists** :  Python list

                    Python list of Python lists.

    **Returns**

            **u** :  Python list

                    Python list created by taking the union of the
                    lists in `ListOfLists`.

    """
    u = []
    for s in ListOfLists:
        if s != None:
            u.extend(s)
    return u

def get_all_args(fn) -> list:
    """
    Returns a list of all arguments for the function fn.

    >>> def foo(x, y, z=100): return x + y + z
    >>> get_all_args(foo)
    ['x', 'y', 'z']
    """
    sig = inspect.signature(fn)
    return list(sig.parameters)

def __len__(self):
        """ This will equal 124 for the V1 database. """
        length = 0
        for typ, siz, _ in self.format:
            length += siz
        return length

def dict_merge(set1, set2):
    """Joins two dictionaries."""
    return dict(list(set1.items()) + list(set2.items()))

def _DateToEpoch(date):
  """Converts python datetime to epoch microseconds."""
  tz_zero = datetime.datetime.utcfromtimestamp(0)
  diff_sec = int((date - tz_zero).total_seconds())
  return diff_sec * 1000000

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

def is_cached(self, url):
        """Checks if specified URL is cached."""
        try:
            return True if url in self.cache else False
        except TypeError:
            return False

def swap_memory():
    """Swap system memory as a (total, used, free, sin, sout) tuple."""
    mem = _psutil_mswindows.get_virtual_mem()
    total = mem[2]
    free = mem[3]
    used = total - free
    percent = usage_percent(used, total, _round=1)
    return nt_swapmeminfo(total, used, free, percent, 0, 0)

def exclude(self, *args, **kwargs) -> "QuerySet":
        """
        Same as .filter(), but with appends all args with NOT
        """
        return self._filter_or_exclude(negate=True, *args, **kwargs)

def find_ge(a, x):
    """Find leftmost item greater than or equal to x."""
    i = bs.bisect_left(a, x)
    if i != len(a): return i
    raise ValueError

def get_document_frequency(self, term):
        """
        Returns the number of documents the specified term appears in.
        """
        if term not in self._terms:
            raise IndexError(TERM_DOES_NOT_EXIST)
        else:
            return len(self._terms[term])

def _trace_full (frame, event, arg):
    """Trace every executed line."""
    if event == "line":
        _trace_line(frame, event, arg)
    else:
        _trace(frame, event, arg)
    return _trace_full

def percentile(values, k):
    """Find the percentile of a list of values.

    :param list values: The list of values to find the percentile of
    :param int k: The percentile to find
    :rtype: float or int

    """
    if not values:
        return None
    values.sort()
    index = (len(values) * (float(k) / 100)) - 1
    return values[int(math.ceil(index))]

def _ioctl (self, func, args):
        """Call ioctl() with given parameters."""
        import fcntl
        return fcntl.ioctl(self.sockfd.fileno(), func, args)

def create_dir_rec(path: Path):
    """
    Create a folder recursive.

    :param path: path
    :type path: ~pathlib.Path
    """
    if not path.exists():
        Path.mkdir(path, parents=True, exist_ok=True)

def bit_clone( bits ):
    """
    Clone a bitset
    """
    new = BitSet( bits.size )
    new.ior( bits )
    return new

def OnMove(self, event):
        """Main window move event"""

        # Store window position in config
        position = self.main_window.GetScreenPositionTuple()

        config["window_position"] = repr(position)

def get_bottomrect_idx(self, pos):
        """ Determine if cursor is on bottom right corner of a hot spot."""
        for i, r in enumerate(self.link_bottom_rects):
            if r.Contains(pos):
                return i
        return -1

def get_type(self):
        """Get the type of the item.

        :return: the type of the item.
        :returntype: `unicode`"""
        item_type = self.xmlnode.prop("type")
        if not item_type:
            item_type = "?"
        return item_type.decode("utf-8")

def numpy(self):
        """ Grabs image data and converts it to a numpy array """
        # load GDCM's image reading functionality
        image_reader = gdcm.ImageReader()
        image_reader.SetFileName(self.fname)
        if not image_reader.Read():
            raise IOError("Could not read DICOM image")
        pixel_array = self._gdcm_to_numpy(image_reader.GetImage())
        return pixel_array

def argmax(attrs, inputs, proto_obj):
    """Returns indices of the maximum values along an axis"""
    axis = attrs.get('axis', 0)
    keepdims = attrs.get('keepdims', 1)
    argmax_op = symbol.argmax(inputs[0], axis=axis, keepdims=keepdims)
    # onnx argmax operator always expects int64 as output type
    cast_attrs = {'dtype': 'int64'}
    return 'cast', cast_attrs, argmax_op

def __call__(self, xy):
        """Project x and y"""
        x, y = xy
        return (self.x(x), self.y(y))

def file_read(filename):
    """Read a file and close it.  Returns the file source."""
    fobj = open(filename,'r');
    source = fobj.read();
    fobj.close()
    return source

def submitbutton(self, request, tag):
        """
        Render an INPUT element of type SUBMIT which will post this form to the
        server.
        """
        return tags.input(type='submit',
                          name='__submit__',
                          value=self._getDescription())

def header_length(bytearray):
    """Return the length of s when it is encoded with base64."""
    groups_of_3, leftover = divmod(len(bytearray), 3)
    # 4 bytes out for each 3 bytes (or nonzero fraction thereof) in.
    n = groups_of_3 * 4
    if leftover:
        n += 4
    return n

def _get_minidom_tag_value(station, tag_name):
    """get a value from a tag (if it exists)"""
    tag = station.getElementsByTagName(tag_name)[0].firstChild
    if tag:
        return tag.nodeValue

    return None

def generic_add(a, b):
    print
    """Simple function to add two numbers"""
    logger.info('Called generic_add({}, {})'.format(a, b))
    return a + b

def str_to_class(class_name):
    """
    Returns a class based on class name    
    """
    mod_str, cls_str = class_name.rsplit('.', 1)
    mod = __import__(mod_str, globals(), locals(), [''])
    cls = getattr(mod, cls_str)
    return cls

def _map_table_name(self, model_names):
        """
        Pre foregin_keys potrbejeme pre z nazvu tabulky zistit class,
        tak si to namapujme
        """

        for model in model_names:
            if isinstance(model, tuple):
                model = model[0]

            try:
                model_cls = getattr(self.models, model)
                self.table_to_class[class_mapper(model_cls).tables[0].name] = model
            except AttributeError:
                pass

def unpatch(obj, name):
    """
    Undo the effects of patch(func, obj, name)
    """
    setattr(obj, name, getattr(obj, name).original)

def load_file(self, filename):
        """Read in file contents and set the current string."""
        with open(filename, 'r') as sourcefile:
            self.set_string(sourcefile.read())

def sorted_chain(*ranges: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Chain & sort ranges."""
    return sorted(itertools.chain(*ranges))

def _synced(method, self, args, kwargs):
    """Underlying synchronized wrapper."""
    with self._lock:
        return method(*args, **kwargs)

def _update_fontcolor(self, fontcolor):
        """Updates text font color button

        Parameters
        ----------

        fontcolor: Integer
        \tText color in integer RGB format

        """

        textcolor = wx.SystemSettings_GetColour(wx.SYS_COLOUR_WINDOWTEXT)
        textcolor.SetRGB(fontcolor)

        self.textcolor_choice.SetColour(textcolor)

def to_percentage(number, rounding=2):
    """Creates a percentage string representation from the given `number`. The
    number is multiplied by 100 before adding a '%' character.

    Raises `ValueError` if `number` cannot be converted to a number.
    """
    number = float(number) * 100
    number_as_int = int(number)
    rounded = round(number, rounding)

    return '{}%'.format(number_as_int if number_as_int == rounded else rounded)

def PythonPercentFormat(format_str):
  """Use Python % format strings as template format specifiers."""

  if format_str.startswith('printf '):
    fmt = format_str[len('printf '):]
    return lambda value: fmt % value
  else:
    return None

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

def dist_sq(self, other):
    """Distance squared to some other point."""
    dx = self.x - other.x
    dy = self.y - other.y
    return dx**2 + dy**2

def append_pdf(input_pdf: bytes, output_writer: PdfFileWriter):
    """
    Appends a PDF to a pyPDF writer. Legacy interface.
    """
    append_memory_pdf_to_writer(input_pdf=input_pdf,
                                writer=output_writer)

def israw(self, **kwargs):
        """
        Returns True if the PTY should operate in raw mode.

        If the container was not started with tty=True, this will return False.
        """

        if self.raw is None:
            info = self._container_info()
            self.raw = self.stdout.isatty() and info['Config']['Tty']

        return self.raw

def setRect(self, rect):
		"""
		Sets the window bounds from a tuple of (x,y,w,h)
		"""
		self.x, self.y, self.w, self.h = rect

def dictlist_replace(dict_list: Iterable[Dict], key: str, value: Any) -> None:
    """
    Process an iterable of dictionaries. For each dictionary ``d``, change
    (in place) ``d[key]`` to ``value``.
    """
    for d in dict_list:
        d[key] = value

def add_suffix(fullname, suffix):
    """ Add suffix to a full file name"""
    name, ext = os.path.splitext(fullname)
    return name + '_' + suffix + ext

def csvpretty(csvfile: csvfile=sys.stdin):
    """ Pretty print a CSV file. """
    shellish.tabulate(csv.reader(csvfile))

def exists(self, digest):
        """
        Check if a blob exists

        :param digest: Hex digest of the blob
        :return: Boolean indicating existence of the blob
        """
        return self.conn.client.blob_exists(self.container_name, digest)

def trim_trailing_silence(self):
        """Trim the trailing silence of the pianoroll."""
        length = self.get_active_length()
        self.pianoroll = self.pianoroll[:length]

def zero_pixels(self):
        """ Return an array of the zero pixels.

        Returns
        -------
        :obj:`numpy.ndarray`
             Nx2 array of the zero pixels
        """
        zero_px = np.where(np.sum(self.raw_data, axis=2) == 0)
        zero_px = np.c_[zero_px[0], zero_px[1]]
        return zero_px

def vec(self):
        """:obj:`numpy.ndarray` : Vector representation for this camera.
        """
        return np.r_[self.fx, self.fy, self.cx, self.cy, self.skew, self.height, self.width]

def __neg__(self):
        """Unary negation"""
        return self.__class__(self[0], self._curve.p()-self[1], self._curve)

def get_system_cpu_times():
    """Return system CPU times as a namedtuple."""
    user, nice, system, idle = _psutil_osx.get_system_cpu_times()
    return _cputimes_ntuple(user, nice, system, idle)

def weighted_std(values, weights):
    """ Calculate standard deviation weighted by errors """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return np.sqrt(variance)

def u16le_list_to_byte_list(data):
    """! @brief Convert a halfword array into a byte array"""
    byteData = []
    for h in data:
        byteData.extend([h & 0xff, (h >> 8) & 0xff])
    return byteData

def addClassKey(self, klass, key, obj):
        """
        Adds an object to the collection, based on klass and key.

        @param klass: The class of the object.
        @param key: The datastore key of the object.
        @param obj: The loaded instance from the datastore.
        """
        d = self._getClass(klass)

        d[key] = obj

def read(self):
        """https://picamera.readthedocs.io/en/release-1.13/recipes1.html#capturing-to-a-pil-image"""
        stream = BytesIO()
        self.cam.capture(stream, format='png')
        # "Rewind" the stream to the beginning so we can read its content
        stream.seek(0)
        return Image.open(stream)

def _config_parse(self):
        """Replacer oslo_config.cfg.ConfigParser.parse for in-memory cfg."""
        res = super(cfg.ConfigParser, self).parse(Backend._config_string_io)
        return res

def _configure_logger():
    """Configure the logging module."""
    if not app.debug:
        _configure_logger_for_production(logging.getLogger())
    elif not app.testing:
        _configure_logger_for_debugging(logging.getLogger())

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

def __remove_method(m: lmap.Map, key: T) -> lmap.Map:
        """Swap the methods atom to remove method with key."""
        return m.dissoc(key)

def read_img(path):
    """ Reads image specified by path into numpy.ndarray"""
    img = cv2.resize(cv2.imread(path, 0), (80, 30)).astype(np.float32) / 255
    img = np.expand_dims(img.transpose(1, 0), 0)
    return img

def flipwritable(fn, mode=None):
    """
    Flip the writability of a file and return the old mode. Returns None
    if the file is already writable.
    """
    if os.access(fn, os.W_OK):
        return None
    old_mode = os.stat(fn).st_mode
    os.chmod(fn, stat.S_IWRITE | old_mode)
    return old_mode

def leaf_nodes(self):
        """
        Return an interable of nodes with no edges pointing at them. This is
        helpful to find all nodes without dependencies.
        """
        # Now contains all nodes that contain dependencies.
        deps = {item for sublist in self.edges.values() for item in sublist}
        # contains all nodes *without* any dependencies (leaf nodes)
        return self.nodes - deps

def write_json_response(self, response):
    """ write back json response """
    self.write(tornado.escape.json_encode(response))
    self.set_header("Content-Type", "application/json")

def _possibly_convert_objects(values):
    """Convert arrays of datetime.datetime and datetime.timedelta objects into
    datetime64 and timedelta64, according to the pandas convention.
    """
    return np.asarray(pd.Series(values.ravel())).reshape(values.shape)

def to_np(*args):
    """ convert GPU arras to numpy and return them"""
    if len(args) > 1:
        return (cp.asnumpy(x) for x in args)
    else:
        return cp.asnumpy(args[0])

def _skip_newlines(self):
        """Increment over newlines."""
        while self._cur_token['type'] is TT.lbreak and not self._finished:
            self._increment()

def read(self, start_position: int, size: int) -> memoryview:
        """
        Return a view into the memory
        """
        return memoryview(self._bytes)[start_position:start_position + size]

def chmod(f):
    """ change mod to writeable """
    try:
        os.chmod(f, S_IWRITE)  # windows (cover all)
    except Exception as e:
        pass
    try:
        os.chmod(f, 0o777)  # *nix
    except Exception as e:
        pass

def close_all():
    r"""Close all opened windows."""

    # Windows can be closed by releasing all references to them so they can be
    # garbage collected. May not be necessary to call close().
    global _qtg_windows
    for window in _qtg_windows:
        window.close()
    _qtg_windows = []

    global _qtg_widgets
    for widget in _qtg_widgets:
        widget.close()
    _qtg_widgets = []

    global _plt_figures
    for fig in _plt_figures:
        _, plt, _ = _import_plt()
        plt.close(fig)
    _plt_figures = []

def _create_empty_array(self, frames, always_2d, dtype):
        """Create an empty array with appropriate shape."""
        import numpy as np
        if always_2d or self.channels > 1:
            shape = frames, self.channels
        else:
            shape = frames,
        return np.empty(shape, dtype, order='C')

def get_python_dict(scala_map):
    """Return a dict from entries in a scala.collection.immutable.Map"""
    python_dict = {}
    keys = get_python_list(scala_map.keys().toList())
    for key in keys:
        python_dict[key] = scala_map.apply(key)
    return python_dict

def _std(self,x):
        """
        Compute standard deviation with ddof degrees of freedom
        """
        return np.nanstd(x.values,ddof=self._ddof)

def __getattr__(self, name):
        """ For attributes not found in self, redirect
        to the properties dictionary """

        try:
            return self.__dict__[name]
        except KeyError:
            if hasattr(self._properties,name):
                return getattr(self._properties, name)

def _isstring(dtype):
    """Given a numpy dtype, determines whether it is a string. Returns True
    if the dtype is string or unicode.
    """
    return dtype.type == numpy.unicode_ or dtype.type == numpy.string_

def is_closed(self):
        """ Check if session was closed. """
        return (self.state == SESSION_STATE.CLOSED 
                or self.state == SESSION_STATE.CLOSING)

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

def __getattr__(self, item: str) -> Callable:
        """Get a callable that sends the actual API request internally."""
        return functools.partial(self.call_action, item)

def get_boto_session(
        region,
        aws_access_key_id=None,
        aws_secret_access_key=None,
        aws_session_token=None
        ):
    """Get a boto3 session."""
    return boto3.session.Session(
        region_name=region,
        aws_secret_access_key=aws_secret_access_key,
        aws_access_key_id=aws_access_key_id,
        aws_session_token=aws_session_token
    )

def _update_font_style(self, font_style):
        """Updates font style widget

        Parameters
        ----------

        font_style: Integer
        \tButton down iif font_style == wx.FONTSTYLE_ITALIC

        """

        toggle_state = font_style & wx.FONTSTYLE_ITALIC == wx.FONTSTYLE_ITALIC

        self.ToggleTool(wx.FONTFLAG_ITALIC, toggle_state)

def __contains__(self, key):
        """Return ``True`` if *key* is present, else ``False``."""
        pickled_key = self._pickle_key(key)
        return bool(self.redis.hexists(self.key, pickled_key))

def slugify(string):
    """
    Removes non-alpha characters, and converts spaces to hyphens. Useful for making file names.


    Source: http://stackoverflow.com/questions/5574042/string-slugification-in-python
    """
    string = re.sub('[^\w .-]', '', string)
    string = string.replace(" ", "-")
    return string

async def wait_and_quit(loop):
	"""Wait until all task are executed."""
	from pylp.lib.tasks import running
	if running:
		await asyncio.wait(map(lambda runner: runner.future, running))

def detect_model_num(string):
    """Takes a string related to a model name and extract its model number.

    For example:
        '000000-bootstrap.index' => 0
    """
    match = re.match(MODEL_NUM_REGEX, string)
    if match:
        return int(match.group())
    return None

async def sysinfo(dev: Device):
    """Print out system information (version, MAC addrs)."""
    click.echo(await dev.get_system_info())
    click.echo(await dev.get_interface_information())

def get_tree_type(tree):
    """
    returns the type of the (sub)tree: Root, Nucleus or Satellite

    Parameters
    ----------
    tree : nltk.tree.ParentedTree
        a tree representing a rhetorical structure (or a part of it)
    """
    tree_type = tree.label()
    assert tree_type in SUBTREE_TYPES, "tree_type: {}".format(tree_type)
    return tree_type

def url_fix_common_typos (url):
    """Fix common typos in given URL like forgotten colon."""
    if url.startswith("http//"):
        url = "http://" + url[6:]
    elif url.startswith("https//"):
        url = "https://" + url[7:]
    return url

def get_shape_mask(self, shape_obj):
        """
        Return full mask where True marks pixels within the given shape.
        """
        wd, ht = self.get_size()
        yi = np.mgrid[:ht].reshape(-1, 1)
        xi = np.mgrid[:wd].reshape(1, -1)
        pts = np.asarray((xi, yi)).T
        contains = shape_obj.contains_pts(pts)
        return contains

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

def get_starting_chunk(filename, length=1024):
    """
    :param filename: File to open and get the first little chunk of.
    :param length: Number of bytes to read, default 1024.
    :returns: Starting chunk of bytes.
    """
    # Ensure we open the file in binary mode
    with open(filename, 'rb') as f:
        chunk = f.read(length)
        return chunk

def adjust(cols, light):
    """Create palette."""
    raw_colors = [cols[0], *cols, "#FFFFFF",
                  "#000000", *cols, "#FFFFFF"]

    return colors.generic_adjust(raw_colors, light)

def timedelta_seconds(timedelta):
    """Returns the total timedelta duration in seconds."""
    return (timedelta.total_seconds() if hasattr(timedelta, "total_seconds")
            else timedelta.days * 24 * 3600 + timedelta.seconds +
                 timedelta.microseconds / 1000000.)

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

def _nbytes(buf):
    """Return byte-size of a memoryview or buffer."""
    if isinstance(buf, memoryview):
        if PY3:
            # py3 introduces nbytes attribute
            return buf.nbytes
        else:
            # compute nbytes on py2
            size = buf.itemsize
            for dim in buf.shape:
                size *= dim
            return size
    else:
        # not a memoryview, raw bytes/ py2 buffer
        return len(buf)

def edge_index(self):
        """A map to look up the index of a edge"""
        return dict((edge, index) for index, edge in enumerate(self.edges))

def get_ref_dict(self, schema):
        """Method to create a dictionary containing a JSON reference to the
        schema in the spec
        """
        schema_key = make_schema_key(schema)
        ref_schema = build_reference(
            "schema", self.openapi_version.major, self.refs[schema_key]
        )
        if getattr(schema, "many", False):
            return {"type": "array", "items": ref_schema}
        return ref_schema

def subkey(dct, keys):
    """Get an entry from a dict of dicts by the list of keys to 'follow'
    """
    key = keys[0]
    if len(keys) == 1:
        return dct[key]
    return subkey(dct[key], keys[1:])

def case_us2mc(x):
    """ underscore to mixed case notation """
    return re.sub(r'_([a-z])', lambda m: (m.group(1).upper()), x)

def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

def _snake_to_camel_case(value):
    """Convert snake case string to camel case."""
    words = value.split("_")
    return words[0] + "".join(map(str.capitalize, words[1:]))

def _file_exists(path, filename):
  """Checks if the filename exists under the path."""
  return os.path.isfile(os.path.join(path, filename))

def numpy_aware_eq(a, b):
    """Return whether two objects are equal via recursion, using
    :func:`numpy.array_equal` for comparing numpy arays.
    """
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    if ((isinstance(a, Iterable) and isinstance(b, Iterable)) and
            not isinstance(a, str) and not isinstance(b, str)):
        if len(a) != len(b):
            return False
        return all(numpy_aware_eq(x, y) for x, y in zip(a, b))
    return a == b

def remove_from_string(string, values):
    """

    Parameters
    ----------
    string:
    values:

    Returns
    -------
    """
    for v in values:
        string = string.replace(v, '')

    return string

def get_sql(query):
    """ Returns the sql query """
    sql = str(query.statement.compile(dialect=sqlite.dialect(),
                                      compile_kwargs={"literal_binds": True}))
    return sql

def get_mtime(fname):
        """
        Find the time this file was last modified.

        :param fname: File name
        :return: The last time the file was modified.
        """
        try:
            mtime = os.stat(fname).st_mtime_ns
        except OSError:
            # The file might be right in the middle of being written
            # so sleep
            time.sleep(1)
            mtime = os.stat(fname).st_mtime_ns

        return mtime

def replace_variables(self, source: str, variables: dict) -> str:
        """Replace {{variable-name}} with stored value."""
        try:
            replaced = re.sub(
                "{{(.*?)}}", lambda m: variables.get(m.group(1), ""), source
            )
        except TypeError:
            replaced = source
        return replaced

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

def mouse_event(dwFlags: int, dx: int, dy: int, dwData: int, dwExtraInfo: int) -> None:
    """mouse_event from Win32."""
    ctypes.windll.user32.mouse_event(dwFlags, dx, dy, dwData, dwExtraInfo)

def show(self, title=''):
        """
        Display Bloch sphere and corresponding data sets.
        """
        self.render(title=title)
        if self.fig:
            plt.show(self.fig)

def array2string(arr: numpy.ndarray) -> str:
        """Format numpy array as a string."""
        shape = str(arr.shape)[1:-1]
        if shape.endswith(","):
            shape = shape[:-1]
        return numpy.array2string(arr, threshold=11) + "%s[%s]" % (arr.dtype, shape)

def string_list_to_array(l):
    """
    Turns a Python unicode string list into a Java String array.

    :param l: the string list
    :type: list
    :rtype: java string array
    :return: JB_Object
    """
    result = javabridge.get_env().make_object_array(len(l), javabridge.get_env().find_class("java/lang/String"))
    for i in range(len(l)):
        javabridge.get_env().set_object_array_element(result, i, javabridge.get_env().new_string_utf(l[i]))
    return result

def drop_indexes(self):
        """Delete all indexes for the database"""
        LOG.warning("Dropping all indexe")
        for collection_name in INDEXES:
            LOG.warning("Dropping all indexes for collection name %s", collection_name)
            self.db[collection_name].drop_indexes()

def objectproxy_realaddress(obj):
    """
    Obtain a real address as an integer from an objectproxy.
    """
    voidp = QROOT.TPython.ObjectProxy_AsVoidPtr(obj)
    return C.addressof(C.c_char.from_buffer(voidp))

def ask_str(question: str, default: str = None):
    """Asks for a simple string"""
    default_q = " [default: {0}]: ".format(
        default) if default is not None else ""
    answer = input("{0} [{1}]: ".format(question, default_q))

    if answer == "":
        return default
    return answer

def debugTreePrint(node,pfx="->"):
  """Purely a debugging aid: Ascii-art picture of a tree descended from node"""
  print pfx,node.item
  for c in node.children:
    debugTreePrint(c,"  "+pfx)

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

def get_method_from_module(module_path, method_name):
    """ from a valid python module path, get the run method name passed """
    top_module = __import__(module_path)

    module = top_module
    # we tunnel down until we find the module we want
    for submodule_name in module_path.split('.')[1:]:
        module = getattr(module, submodule_name)

    assert hasattr(module, method_name), \
        "unable to find method {0} from module {1}. does the method exist?".format(method_name, module_path)
    return getattr(module, method_name)

def refresh_core(self):
        """Query device for all attributes that exist regardless of power state.

        This will force a refresh for all device queries that are valid to
        request at any time.  It's the only safe suite of queries that we can
        make if we do not know the current state (on or off+standby).

        This does not return any data, it just issues the queries.
        """
        self.log.info('Sending out mass query for all attributes')
        for key in ATTR_CORE:
            self.query(key)

def process_instance(self, instance):
        self.log.debug("e = mc^2")
        self.log.info("About to fail..")
        self.log.warning("Failing.. soooon..")
        self.log.critical("Ok, you're done.")
        assert False, """ValidateFailureMock was destined to fail..

Here's some extended information about what went wrong.

It has quite the long string associated with it, including
a few newlines and a list.

- Item 1
- Item 2

"""

def _update_plot(self, _):
        """Callback to redraw the plot to reflect the new parameter values."""
        # Since all sliders call this same callback without saying who they are
        # I need to update the values for all parameters. This can be
        # circumvented by creating a seperate callback function for each
        # parameter.
        for param in self.model.params:
            param.value = self._sliders[param].val
        for indep_var, dep_var in self._projections:
            self._update_specific_plot(indep_var, dep_var)

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

def set_xticks_for_all(self, row_column_list=None, ticks=None):
        """Manually specify the x-axis tick values.

        :param row_column_list: a list containing (row, column) tuples to
            specify the subplots, or None to indicate *all* subplots.
        :type row_column_list: list or None
        :param ticks: list of tick values.

        """
        if row_column_list is None:
            self.ticks['x'] = ticks
        else:
            for row, column in row_column_list:
                self.set_xticks(row, column, ticks)

def _screen(self, s, newline=False):
        """Print something on screen when self.verbose == True"""
        if self.verbose:
            if newline:
                print(s)
            else:
                print(s, end=' ')

def iter_fields(self, schema: Schema) -> Iterable[Tuple[str, Field]]:
        """
        Iterate through marshmallow schema fields.

        Generates: name, field pairs

        """
        for name in sorted(schema.fields.keys()):
            field = schema.fields[name]
            yield field.dump_to or name, field

def time_func(func, name, *args, **kwargs):
    """ call a func with args and kwargs, print name of func and how
    long it took. """
    tic = time.time()
    out = func(*args, **kwargs)
    toc = time.time()
    print('%s took %0.2f seconds' % (name, toc - tic))
    return out

def unaccentuate(s):
    """ Replace accentuated chars in string by their non accentuated equivalent. """
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

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

def nest(thing):
    """Use tensorflows nest function if available, otherwise just wrap object in an array"""
    tfutil = util.get_module('tensorflow.python.util')
    if tfutil:
        return tfutil.nest.flatten(thing)
    else:
        return [thing]

def unwind(self):
        """ Get a list of all ancestors in descending order of level, including a new instance  of self
        """
        return [ QuadKey(self.key[:l+1]) for l in reversed(range(len(self.key))) ]

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

def log_no_newline(self, msg):
      """ print the message to the predefined log file without newline """
      self.print2file(self.logfile, False, False, msg)

def grandparent_path(self):
        """ return grandparent's path string """
        return os.path.basename(os.path.join(self.path, '../..'))

def list2dict(list_of_options):
    """Transforms a list of 2 element tuples to a dictionary"""
    d = {}
    for key, value in list_of_options:
        d[key] = value
    return d

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

def md5_hash_file(fh):
    """Return the md5 hash of the given file-object"""
    md5 = hashlib.md5()
    while True:
        data = fh.read(8192)
        if not data:
            break
        md5.update(data)
    return md5.hexdigest()

def string_to_list(string, sep=",", filter_empty=False):
    """Transforma una string con elementos separados por `sep` en una lista."""
    return [value.strip() for value in string.split(sep)
            if (not filter_empty or value)]

def sections(self) -> list:
        """List of sections."""
        self.config.read(self.filepath)
        return self.config.sections()

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

def resize_by_area(img, size):
  """image resize function used by quite a few image problems."""
  return tf.to_int64(
      tf.image.resize_images(img, [size, size], tf.image.ResizeMethod.AREA))

def count_list(the_list):
    """
    Generates a count of the number of times each unique item appears in a list
    """
    count = the_list.count
    result = [(item, count(item)) for item in set(the_list)]
    result.sort()
    return result

def _open_text(fname, **kwargs):
    """On Python 3 opens a file in text mode by using fs encoding and
    a proper en/decoding errors handler.
    On Python 2 this is just an alias for open(name, 'rt').
    """
    if PY3:
        kwargs.setdefault('encoding', ENCODING)
        kwargs.setdefault('errors', ENCODING_ERRS)
    return open(fname, "rt", **kwargs)

def add_object(self, obj):
        """Add object to local and app environment storage

        :param obj: Instance of a .NET object
        """
        if obj.top_level_object:
            if isinstance(obj, DotNetNamespace):
                self.namespaces[obj.name] = obj
        self.objects[obj.id] = obj

def callPlaybook(self, playbook, ansibleArgs, wait=True, tags=["all"]):
        """
        Run a playbook.

        :param playbook: An Ansible playbook to run.
        :param ansibleArgs: Arguments to pass to the playbook.
        :param wait: Wait for the play to finish if true.
        :param tags: Control tags for the play.
        """
        playbook = os.path.join(self.playbooks, playbook)  # Path to playbook being executed
        verbosity = "-vvvvv" if logger.isEnabledFor(logging.DEBUG) else "-v"
        command = ["ansible-playbook", verbosity, "--tags", ",".join(tags), "--extra-vars"]
        command.append(" ".join(["=".join(i) for i in ansibleArgs.items()]))  # Arguments being passed to playbook
        command.append(playbook)

        logger.debug("Executing Ansible call `%s`", " ".join(command))
        p = subprocess.Popen(command)
        if wait:
            p.communicate()
            if p.returncode != 0:
                # FIXME: parse error codes
                raise RuntimeError("Ansible reported an error when executing playbook %s" % playbook)

def get_first_lang():
    """Get the first lang of Accept-Language Header.
    """
    request_lang = request.headers.get('Accept-Language').split(',')
    if request_lang:
        lang = locale.normalize(request_lang[0]).split('.')[0]
    else:
        lang = False
    return lang

def start(self):
        """Activate the TypingStream on stdout"""
        self.streams.append(sys.stdout)
        sys.stdout = self.stream

def execute(self, sql, params=None):
        """Just a pointer to engine.execute
        """
        # wrap in a transaction to ensure things are committed
        # https://github.com/smnorris/pgdata/issues/3
        with self.engine.begin() as conn:
            result = conn.execute(sql, params)
        return result

def _finish(self):
        """
        Closes and waits for subprocess to exit.
        """
        if self._process.returncode is None:
            self._process.stdin.flush()
            self._process.stdin.close()
            self._process.wait()
            self.closed = True

def maskIndex(self):
        """ Returns a boolean index with True if the value is masked.

            Always has the same shape as the maksedArray.data, event if the mask is a single boolan.
        """
        if isinstance(self.mask, bool):
            return np.full(self.data.shape, self.mask, dtype=np.bool)
        else:
            return self.mask

def next (self):    # File-like object.

        """This is to support iterators over a file-like object.
        """

        result = self.readline()
        if result == self._empty_buffer:
            raise StopIteration
        return result

def clean_column_names(df: DataFrame) -> DataFrame:
    """
    Strip the whitespace from all column names in the given DataFrame
    and return the result.
    """
    f = df.copy()
    f.columns = [col.strip() for col in f.columns]
    return f

def set_attr(self, name, value):
    """ Sets the value of an attribute. """
    self.exec_script("node.setAttribute(%s, %s)" % (repr(name), repr(value)))

def utime(self, *args, **kwargs):
        """ Set the access and modified times of the file specified by path. """
        os.utime(self.extended_path, *args, **kwargs)

def do_restart(self, line):
        """
        Attempt to restart the bot.
        """
        self.bot._frame = 0
        self.bot._namespace.clear()
        self.bot._namespace.update(self.bot._initial_namespace)

def use_kwargs(self, *args, **kwargs) -> typing.Callable:
        """Decorator that injects parsed arguments into a view function or method.

        Receives the same arguments as `webargs.core.Parser.use_kwargs`.

        """
        return super().use_kwargs(*args, **kwargs)

def _scaleSinglePoint(point, scale=1, convertToInteger=True):
    """
    Scale a single point
    """
    x, y = point
    if convertToInteger:
        return int(round(x * scale)), int(round(y * scale))
    else:
        return (x * scale, y * scale)

def get_property(self, name):
        # type: (str) -> object
        """
        Retrieves a framework or system property. As framework properties don't
        change while it's running, this method don't need to be protected.

        :param name: The property name
        """
        with self.__properties_lock:
            return self.__properties.get(name, os.getenv(name))

def map_parameters(cls, params):
        """Maps parameters to form field names"""

        d = {}
        for k, v in six.iteritems(params):
            d[cls.FIELD_MAP.get(k.lower(), k)] = v
        return d

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