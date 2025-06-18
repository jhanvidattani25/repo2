def writeBoolean(self, n):
        """
        Writes a Boolean to the stream.
        """
        t = TYPE_BOOL_TRUE

        if n is False:
            t = TYPE_BOOL_FALSE

        self.stream.write(t)

def paste(xsel=False):
    """Returns system clipboard contents."""
    selection = "primary" if xsel else "clipboard"
    try:
        return subprocess.Popen(["xclip", "-selection", selection, "-o"], stdout=subprocess.PIPE).communicate()[0].decode("utf-8")
    except OSError as why:
        raise XclipNotFound

def _format_json(data, theme):
    """Pretty print a dict as a JSON, with colors if pygments is present."""
    output = json.dumps(data, indent=2, sort_keys=True)

    if pygments and sys.stdout.isatty():
        style = get_style_by_name(theme)
        formatter = Terminal256Formatter(style=style)
        return pygments.highlight(output, JsonLexer(), formatter)

    return output

def create_path(path):
    """Creates a absolute path in the file system.

    :param path: The path to be created
    """
    import os
    if not os.path.exists(path):
        os.makedirs(path)

def _vector_or_scalar(x, type='row'):
    """Convert an object to either a scalar or a row or column vector."""
    if isinstance(x, (list, tuple)):
        x = np.array(x)
    if isinstance(x, np.ndarray):
        assert x.ndim == 1
        if type == 'column':
            x = x[:, None]
    return x

def experiment_property(prop):
    """Get a property of the experiment by name."""
    exp = experiment(session)
    p = getattr(exp, prop)
    return success_response(field=prop, data=p, request_type=prop)

def data_from_file(file):
    """Return (first channel data, sample frequency, sample width) from a .wav
    file."""
    fp = wave.open(file, 'r')
    data = fp.readframes(fp.getnframes())
    channels = fp.getnchannels()
    freq = fp.getframerate()
    bits = fp.getsampwidth()

    # Unpack bytes -- warning currently only tested with 16 bit wavefiles. 32
    # bit not supported.
    data = struct.unpack(('%sh' % fp.getnframes()) * channels, data)

    # Only use first channel
    channel1 = []
    n = 0
    for d in data:
        if n % channels == 0:
            channel1.append(d)
        n += 1
    fp.close()
    return (channel1, freq, bits)

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

def timespan(start_time):
    """Return time in milliseconds from start_time"""

    timespan = datetime.datetime.now() - start_time
    timespan_ms = timespan.total_seconds() * 1000
    return timespan_ms

def _convert_to_array(array_like, dtype):
        """
        Convert Matrix attributes which are array-like or buffer to array.
        """
        if isinstance(array_like, bytes):
            return np.frombuffer(array_like, dtype=dtype)
        return np.asarray(array_like, dtype=dtype)

def get_uniques(l):
    """ Returns a list with no repeated elements.
    """
    result = []

    for i in l:
        if i not in result:
            result.append(i)

    return result

def interp(x, xp, *args, **kwargs):
    """Wrap interpolate_1d for deprecated interp."""
    return interpolate_1d(x, xp, *args, **kwargs)

def _array2cstr(arr):
    """ Serializes a numpy array to a compressed base64 string """
    out = StringIO()
    np.save(out, arr)
    return b64encode(out.getvalue())

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

def _string_hash(s):
    """String hash (djb2) with consistency between py2/py3 and persistency between runs (unlike `hash`)."""
    h = 5381
    for c in s:
        h = h * 33 + ord(c)
    return h

def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

def _encode_bool(name, value, dummy0, dummy1):
    """Encode a python boolean (True/False)."""
    return b"\x08" + name + (value and b"\x01" or b"\x00")

def transform_to_3d(points,normal,z=0):
    """Project points into 3d from 2d points."""
    d = np.cross(normal, (0, 0, 1))
    M = rotation_matrix(d)
    transformed_points = M.dot(points.T).T + z
    return transformed_points

def _not(condition=None, **kwargs):
    """
    Return the opposite of input condition.

    :param condition: condition to process.

    :result: not condition.
    :rtype: bool
    """

    result = True

    if condition is not None:
        result = not run(condition, **kwargs)

    return result

def HttpResponse403(request, template=KEY_AUTH_403_TEMPLATE,
content=KEY_AUTH_403_CONTENT, content_type=KEY_AUTH_403_CONTENT_TYPE):
    """
    HTTP response for forbidden access (status code 403)
    """
    return AccessFailedResponse(request, template, content, content_type, status=403)

def items(self, section_name):
        """:return: list((option, value), ...) pairs of all items in the given section"""
        return [(k, v) for k, v in super(GitConfigParser, self).items(section_name) if k != '__name__']

def mag(z):
    """Get the magnitude of a vector."""
    if isinstance(z[0], np.ndarray):
        return np.array(list(map(np.linalg.norm, z)))
    else:
        return np.linalg.norm(z)

def config_parser_to_dict(config_parser):
    """
    Convert a ConfigParser to a dictionary.
    """
    response = {}

    for section in config_parser.sections():
        for option in config_parser.options(section):
            response.setdefault(section, {})[option] = config_parser.get(section, option)

    return response

def __add__(self, other):
        """Handle the `+` operator."""
        return self._handle_type(other)(self.value + other.value)

def connect_mysql(host, port, user, password, database):
    """Connect to MySQL with retries."""
    return pymysql.connect(
        host=host, port=port,
        user=user, passwd=password,
        db=database
    )

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

def connect(url, username, password):
    """
    Return a connected Bitbucket session
    """

    bb_session = stashy.connect(url, username, password)

    logger.info('Connected to: %s as %s', url, username)

    return bb_session

def add_blank_row(self, label):
        """
        Add a blank row with only an index value to self.df.
        This is done inplace.
        """
        col_labels = self.df.columns
        blank_item = pd.Series({}, index=col_labels, name=label)
        # use .loc to add in place (append won't do that)
        self.df.loc[blank_item.name] = blank_item
        return self.df

def teardown(self):
        """
        Stop and remove the container if it exists.
        """
        while self._http_clients:
            self._http_clients.pop().close()
        if self.created:
            self.halt()

def dumped(text, level, indent=2):
    """Put curly brackets round an indented text"""
    return indented("{\n%s\n}" % indented(text, level + 1, indent) or "None", level, indent) + "\n"

def context(self):
        """
        Create a context manager that ensures code runs within action's context.

        The action does NOT finish when the context is exited.
        """
        parent = _ACTION_CONTEXT.set(self)
        try:
            yield self
        finally:
            _ACTION_CONTEXT.reset(parent)

def pformat(object, indent=1, width=80, depth=None):
    """Format a Python object into a pretty-printed representation."""
    return PrettyPrinter(indent=indent, width=width, depth=depth).pformat(object)

def replace_sys_args(new_args):
    """Temporarily replace sys.argv with current arguments

    Restores sys.argv upon exit of the context manager.
    """
    # Replace sys.argv arguments
    # for module import
    old_args = sys.argv
    sys.argv = new_args
    try:
        yield
    finally:
        sys.argv = old_args

def serialize(obj):
    """Takes a object and produces a dict-like representation

    :param obj: the object to serialize
    """
    if isinstance(obj, list):
        return [serialize(o) for o in obj]
    return GenericSerializer(ModelProviderImpl()).serialize(obj)

def advance_one_line(self):
    """Advances to next line."""

    current_line = self._current_token.line_number
    while current_line == self._current_token.line_number:
      self._current_token = ConfigParser.Token(*next(self._token_generator))

def generate_swagger_html(swagger_static_root, swagger_json_url):
    """
    given a root directory for the swagger statics, and
    a swagger json path, return back a swagger html designed
    to use those values.
    """
    tmpl = _get_template("swagger.html")
    return tmpl.render(
        swagger_root=swagger_static_root, swagger_json_url=swagger_json_url
    )

def do_next(self, args):
        """Step over the next statement
        """
        self._do_print_from_last_cmd = True
        self._interp.step_over()
        return True

def __add__(self,other):
        """
            If the number of columns matches, we can concatenate two LabeldMatrices
            with the + operator.
        """
        assert self.matrix.shape[1] == other.matrix.shape[1]
        return LabeledMatrix(np.concatenate([self.matrix,other.matrix],axis=0),self.labels)

def get_line_flux(line_wave, wave, flux, **kwargs):
    """Interpolated flux at a given wavelength (calls np.interp)."""
    return np.interp(line_wave, wave, flux, **kwargs)

def send(message, request_context=None, binary=False):
    """Sends a message to websocket.

    :param str message: data to send

    :param request_context:

    :raises IOError: If unable to send a message.
    """
    if binary:
        return uwsgi.websocket_send_binary(message, request_context)

    return uwsgi.websocket_send(message, request_context)

def get_number(s, cast=int):
    """
    Try to get a number out of a string, and cast it.
    """
    import string
    d = "".join(x for x in str(s) if x in string.digits)
    return cast(d)

def get_hline():
    """ gets a horiztonal line """
    return Window(
        width=LayoutDimension.exact(1),
        height=LayoutDimension.exact(1),
        content=FillControl('-', token=Token.Line))

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

def to_snake_case(name):
    """ Given a name in camelCase return in snake_case """
    s1 = FIRST_CAP_REGEX.sub(r'\1_\2', name)
    return ALL_CAP_REGEX.sub(r'\1_\2', s1).lower()

def populate_obj(obj, attrs):
    """Populates an object's attributes using the provided dict
    """
    for k, v in attrs.iteritems():
        setattr(obj, k, v)

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

def copyFile(input, output, replace=None):
    """Copy a file whole from input to output."""

    _found = findFile(output)
    if not _found or (_found and replace):
        shutil.copy2(input, output)

def push(h, x):
    """Push a new value into heap."""
    h.push(x)
    up(h, h.size()-1)

def yank(event):
    """
    Paste before cursor.
    """
    event.current_buffer.paste_clipboard_data(
        event.cli.clipboard.get_data(), count=event.arg, paste_mode=PasteMode.EMACS)

def filter_contour(imageFile, opFile):
    """ convert an image by applying a contour """
    im = Image.open(imageFile)
    im1 = im.filter(ImageFilter.CONTOUR)
    im1.save(opFile)

def count(lines):
  """ Counts the word frequences in a list of sentences.

  Note:
    This is a helper function for parallel execution of `Vocabulary.from_text`
    method.
  """
  words = [w for l in lines for w in l.strip().split()]
  return Counter(words)

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

def count_replica(self, partition):
        """Return count of replicas of given partition."""
        return sum(1 for b in partition.replicas if b in self.brokers)

def visit_Name(self, node):
        """ Get range for parameters for examples or false branching. """
        return self.add(node, self.result[node.id])

def mkdir(dir, enter):
    """Create directory with template for topic of the current environment

    """

    if not os.path.exists(dir):
        os.makedirs(dir)

def qrot(vector, quaternion):
    """Rotate a 3D vector using quaternion algebra.

    Implemented by Vladimir Kulikovskiy.

    Parameters
    ----------
    vector: np.array
    quaternion: np.array

    Returns
    -------
    np.array

    """
    t = 2 * np.cross(quaternion[1:], vector)
    v_rot = vector + quaternion[0] * t + np.cross(quaternion[1:], t)
    return v_rot

def _numpy_char_to_bytes(arr):
    """Like netCDF4.chartostring, but faster and more flexible.
    """
    # based on: http://stackoverflow.com/a/10984878/809705
    arr = np.array(arr, copy=False, order='C')
    dtype = 'S' + str(arr.shape[-1])
    return arr.view(dtype).reshape(arr.shape[:-1])

def _string_hash(s):
    """String hash (djb2) with consistency between py2/py3 and persistency between runs (unlike `hash`)."""
    h = 5381
    for c in s:
        h = h * 33 + ord(c)
    return h

def csv_to_dicts(file, header=None):
    """Reads a csv and returns a List of Dicts with keys given by header row."""
    with open(file) as csvfile:
        return [row for row in csv.DictReader(csvfile, fieldnames=header)]

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

def one_hot(x, size, dtype=np.float32):
  """Make a n+1 dim one-hot array from n dim int-categorical array."""
  return np.array(x[..., np.newaxis] == np.arange(size), dtype)

def round_to_int(number, precision):
    """Round a number to a precision"""
    precision = int(precision)
    rounded = (int(number) + precision / 2) // precision * precision
    return rounded

def create_object(cls, members):
    """Promise an object of class `cls` with content `members`."""
    obj = cls.__new__(cls)
    obj.__dict__ = members
    return obj

def to_unicode_repr( _letter ):
    """ helpful in situations where browser/app may recognize Unicode encoding
        in the \u0b8e type syntax but not actual unicode glyph/code-point"""
    # Python 2-3 compatible
    return u"u'"+ u"".join( [ u"\\u%04x"%ord(l) for l in _letter ] ) + u"'"

def create_path(path):
    """Creates a absolute path in the file system.

    :param path: The path to be created
    """
    import os
    if not os.path.exists(path):
        os.makedirs(path)

def string_input(prompt=''):
    """Python 3 input()/Python 2 raw_input()"""
    v = sys.version[0]
    if v == '3':
        return input(prompt)
    else:
        return raw_input(prompt)

def cfloat64_array_to_numpy(cptr, length):
    """Convert a ctypes double pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_double)):
        return np.fromiter(cptr, dtype=np.float64, count=length)
    else:
        raise RuntimeError('Expected double pointer')

def yn_prompt(msg, default=True):
    """
    Prompts the user for yes or no.
    """
    ret = custom_prompt(msg, ["y", "n"], "y" if default else "n")
    if ret == "y":
        return True
    return False

def _display(self, layout):
        """launch layouts display"""
        print(file=self.out)
        TextWriter().format(layout, self.out)

def assert_list(self, putative_list, expected_type=string_types, key_arg=None):
    """
    :API: public
    """
    return assert_list(putative_list, expected_type, key_arg=key_arg,
                       raise_type=lambda msg: TargetDefinitionException(self, msg))

def _xxrange(self, start, end, step_count):
        """Generate n values between start and end."""
        _step = (end - start) / float(step_count)
        return (start + (i * _step) for i in xrange(int(step_count)))

def assert_exactly_one_true(bool_list):
    """This method asserts that only one value of the provided list is True.

    :param bool_list: List of booleans to check
    :return: True if only one value is True, False otherwise
    """
    assert isinstance(bool_list, list)
    counter = 0
    for item in bool_list:
        if item:
            counter += 1
    return counter == 1

def _get_random_id():
    """ Get a random (i.e., unique) string identifier"""
    symbols = string.ascii_uppercase + string.ascii_lowercase + string.digits
    return ''.join(random.choice(symbols) for _ in range(15))

async def list(source):
    """Generate a single list from an asynchronous sequence."""
    result = []
    async with streamcontext(source) as streamer:
        async for item in streamer:
            result.append(item)
    yield result

def csv_to_dicts(file, header=None):
    """Reads a csv and returns a List of Dicts with keys given by header row."""
    with open(file) as csvfile:
        return [row for row in csv.DictReader(csvfile, fieldnames=header)]

def _attrprint(d, delimiter=', '):
    """Print a dictionary of attributes in the DOT format"""
    return delimiter.join(('"%s"="%s"' % item) for item in sorted(d.items()))

def get_next_scheduled_time(cron_string):
    """Calculate the next scheduled time by creating a crontab object
    with a cron string"""
    itr = croniter.croniter(cron_string, datetime.utcnow())
    return itr.get_next(datetime)

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

def dot_product(self, other):
        """ Return the dot product of the given vectors. """
        return self.x * other.x + self.y * other.y

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

def list_to_csv(value):
    """
    Converts list to string with comma separated values. For string is no-op.
    """
    if isinstance(value, (list, tuple, set)):
        value = ",".join(value)
    return value

def average(iterator):
    """Iterative mean."""
    count = 0
    total = 0
    for num in iterator:
        count += 1
        total += num
    return float(total)/count

def cint32_array_to_numpy(cptr, length):
    """Convert a ctypes int pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_int32)):
        return np.fromiter(cptr, dtype=np.int32, count=length)
    else:
        raise RuntimeError('Expected int pointer')

def _aws_get_instance_by_tag(region, name, tag, raw):
    """Get all instances matching a tag."""
    client = boto3.session.Session().client('ec2', region)
    matching_reservations = client.describe_instances(Filters=[{'Name': tag, 'Values': [name]}]).get('Reservations', [])
    instances = []
    [[instances.append(_aws_instance_from_dict(region, instance, raw))  # pylint: disable=expression-not-assigned
      for instance in reservation.get('Instances')] for reservation in matching_reservations if reservation]
    return instances

def cfloat64_array_to_numpy(cptr, length):
    """Convert a ctypes double pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_double)):
        return np.fromiter(cptr, dtype=np.float64, count=length)
    else:
        raise RuntimeError('Expected double pointer')

def loganalytics_data_plane_client(cli_ctx, _):
    """Initialize Log Analytics data client for use with CLI."""
    from .vendored_sdks.loganalytics import LogAnalyticsDataClient
    from azure.cli.core._profile import Profile
    profile = Profile(cli_ctx=cli_ctx)
    cred, _, _ = profile.get_login_credentials(
        resource="https://api.loganalytics.io")
    return LogAnalyticsDataClient(cred)

def cfloat32_array_to_numpy(cptr, length):
    """Convert a ctypes float pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_float)):
        return np.fromiter(cptr, dtype=np.float32, count=length)
    else:
        raise RuntimeError('Expected float pointer')

def underscore(text):
    """Converts text that may be camelcased into an underscored format"""
    return UNDERSCORE[1].sub(r'\1_\2', UNDERSCORE[0].sub(r'\1_\2', text)).lower()

def cint8_array_to_numpy(cptr, length):
    """Convert a ctypes int pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_int8)):
        return np.fromiter(cptr, dtype=np.int8, count=length)
    else:
        raise RuntimeError('Expected int pointer')

def get_stoplist(language):
    """Returns an built-in stop-list for the language as a set of words."""
    file_path = os.path.join("stoplists", "%s.txt" % language)
    try:
        stopwords = pkgutil.get_data("justext", file_path)
    except IOError:
        raise ValueError(
            "Stoplist for language '%s' is missing. "
            "Please use function 'get_stoplists' for complete list of stoplists "
            "and feel free to contribute by your own stoplist." % language
        )

    return frozenset(w.decode("utf8").lower() for w in stopwords.splitlines())

def add_str(window, line_num, str):
    """ attempt to draw str on screen and ignore errors if they occur """
    try:
        window.addstr(line_num, 0, str)
    except curses.error:
        pass

def relative_path(path):
    """
    Return the given path relative to this file.
    """
    return os.path.join(os.path.dirname(__file__), path)

def dictfetchall(cursor):
    """Returns all rows from a cursor as a dict (rather than a headerless table)

    From Django Documentation: https://docs.djangoproject.com/en/dev/topics/db/sql/
    """
    desc = cursor.description
    return [dict(zip([col[0] for col in desc], row)) for row in cursor.fetchall()]

def xmltreefromfile(filename):
    """Internal function to read an XML file"""
    try:
        return ElementTree.parse(filename, ElementTree.XMLParser(collect_ids=False))
    except TypeError:
        return ElementTree.parse(filename, ElementTree.XMLParser())

def _dictfetchall(self, cursor):
        """ Return all rows from a cursor as a dict. """
        columns = [col[0] for col in cursor.description]
        return [
            dict(zip(columns, row))
            for row in cursor.fetchall()
        ]

def beta_pdf(x, a, b):
  """Beta distirbution probability density function."""
  bc = 1 / beta(a, b)
  fc = x ** (a - 1)
  sc = (1 - x) ** (b - 1)
  return bc * fc * sc

def filter_out(queryset, setting_name):
  """
  Remove unwanted results from queryset
  """
  kwargs = helpers.get_settings().get(setting_name, {}).get('FILTER_OUT', {})
  queryset = queryset.exclude(**kwargs)
  return queryset

def intToBin(i):
    """ Integer to two bytes """
    # divide in two parts (bytes)
    i1 = i % 256
    i2 = int(i / 256)
    # make string (little endian)
    return i.to_bytes(2, byteorder='little')

def listlike(obj):
    """Is an object iterable like a list (and not a string)?"""
    
    return hasattr(obj, "__iter__") \
    and not issubclass(type(obj), str)\
    and not issubclass(type(obj), unicode)

def table_top_abs(self):
        """Returns the absolute position of table top"""
        table_height = np.array([0, 0, self.table_full_size[2]])
        return string_to_array(self.floor.get("pos")) + table_height

def pdf(x, mu, std):
    """Probability density function (normal distribution)"""
    return (1.0 / (std * sqrt(2 * pi))) * np.exp(-(x - mu) ** 2 / (2 * std ** 2))

def bytes_to_c_array(data):
    """
    Make a C array using the given string.
    """
    chars = [
        "'{}'".format(encode_escape(i))
        for i in decode_escape(data)
    ]
    return ', '.join(chars) + ', 0'

def gray2bgr(img):
    """Convert a grayscale image to BGR image.

    Args:
        img (ndarray or str): The input image.

    Returns:
        ndarray: The converted BGR image.
    """
    img = img[..., None] if img.ndim == 2 else img
    out_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return out_img

def mean_date(dt_list):
    """Calcuate mean datetime from datetime list
    """
    dt_list_sort = sorted(dt_list)
    dt_list_sort_rel = [dt - dt_list_sort[0] for dt in dt_list_sort]
    avg_timedelta = sum(dt_list_sort_rel, timedelta())/len(dt_list_sort_rel)
    return dt_list_sort[0] + avg_timedelta

def rotate_img(im, deg, mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_AREA):
    """ Rotates an image by deg degrees

    Arguments:
        deg (float): degree to rotate.
    """
    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c//2,r//2),deg,1)
    return cv2.warpAffine(im,M,(c,r), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS+interpolation)

def similarity(self, other):
        """Calculates the cosine similarity between this vector and another
        vector."""
        if self.magnitude == 0 or other.magnitude == 0:
            return 0

        return self.dot(other) / self.magnitude

def rotate_img(im, deg, mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_AREA):
    """ Rotates an image by deg degrees

    Arguments:
        deg (float): degree to rotate.
    """
    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c//2,r//2),deg,1)
    return cv2.warpAffine(im,M,(c,r), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS+interpolation)

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

def screen_cv2(self):
        """cv2 Image of current window screen"""
        pil_image = self.screen.convert('RGB')
        cv2_image = np.array(pil_image)
        pil_image.close()
        # Convert RGB to BGR 
        cv2_image = cv2_image[:, :, ::-1]
        return cv2_image

def direct2dDistance(self, point):
        """consider the distance between two mapPoints, ignoring all terrain, pathing issues"""
        if not isinstance(point, MapPoint): return 0.0
        return  ((self.x-point.x)**2 + (self.y-point.y)**2)**(0.5) # simple distance formula

def _model_unique(ins):
    """ Get unique constraints info

    :type ins: sqlalchemy.orm.mapper.Mapper
    :rtype: list[tuple[str]]
    """
    unique = []
    for t in ins.tables:
        for c in t.constraints:
            if isinstance(c, UniqueConstraint):
                unique.append(tuple(col.key for col in c.columns))
    return unique

def horz_dpi(self):
        """
        Integer dots per inch for the width of this image. Defaults to 72
        when not present in the file, as is often the case.
        """
        pHYs = self._chunks.pHYs
        if pHYs is None:
            return 72
        return self._dpi(pHYs.units_specifier, pHYs.horz_px_per_unit)

def parse(self, s):
        """
        Parses a date string formatted like ``YYYY-MM-DD``.
        """
        return datetime.datetime.strptime(s, self.date_format).date()

def estimate_complexity(self, x,y,z,n):
        """ 
        calculates a rough guess of runtime based on product of parameters 
        """
        num_calculations = x * y * z * n
        run_time = num_calculations / 100000  # a 2014 PC does about 100k calcs in a second (guess based on prior logs)
        return self.show_time_as_short_string(run_time)

def weekly(date=datetime.date.today()):
    """
    Weeks start are fixes at Monday for now.
    """
    return date - datetime.timedelta(days=date.weekday())

def inh(table):
    """
    inverse hyperbolic sine transformation
    """
    t = []
    for i in table:
        t.append(np.ndarray.tolist(np.arcsinh(i)))
    return t

def daterange(start, end, delta=timedelta(days=1), lower=Interval.CLOSED, upper=Interval.OPEN):
    """Returns a generator which creates the next value in the range on demand"""
    date_interval = Interval(lower=lower, lower_value=start, upper_value=end, upper=upper)
    current = start if start in date_interval else start + delta
    while current in date_interval:
        yield current
        current = current + delta

async def _thread_coro(self, *args):
        """ Coroutine called by MapAsync. It's wrapping the call of
        run_in_executor to run the synchronous function as thread """
        return await self._loop.run_in_executor(
            self._executor, self._function, *args)

def start_of_month(val):
    """
    Return a new datetime.datetime object with values that represent
    a start of a month.
    :param val: Date to ...
    :type val: datetime.datetime | datetime.date
    :rtype: datetime.datetime
    """
    if type(val) == date:
        val = datetime.fromordinal(val.toordinal())
    return start_of_day(val).replace(day=1)

def check_output(args, env=None, sp=subprocess):
    """Call an external binary and return its stdout."""
    log.debug('calling %s with env %s', args, env)
    output = sp.check_output(args=args, env=env)
    log.debug('output: %r', output)
    return output

def datetime_to_ms(dt):
    """
    Converts a datetime to a millisecond accuracy timestamp
    """
    seconds = calendar.timegm(dt.utctimetuple())
    return seconds * 1000 + int(dt.microsecond / 1000)

def retry_on_signal(function):
    """Retries function until it doesn't raise an EINTR error"""
    while True:
        try:
            return function()
        except EnvironmentError, e:
            if e.errno != errno.EINTR:
                raise

def datetime_to_timezone(date, tz="UTC"):
    """ convert naive datetime to timezone-aware datetime """
    if not date.tzinfo:
        date = date.replace(tzinfo=timezone(get_timezone()))
    return date.astimezone(timezone(tz))

def test(*args):
    """
    Run unit tests.
    """
    subprocess.call(["py.test-2.7"] + list(args))
    subprocess.call(["py.test-3.4"] + list(args))

def ToDatetime(self):
    """Converts Timestamp to datetime."""
    return datetime.utcfromtimestamp(
        self.seconds + self.nanos / float(_NANOS_PER_SECOND))

def sortable_title(instance):
    """Uses the default Plone sortable_text index lower-case
    """
    title = plone_sortable_title(instance)
    if safe_callable(title):
        title = title()
    return title.lower()

def localize(dt):
    """Localize a datetime object to local time."""
    if dt.tzinfo is UTC:
        return (dt + LOCAL_UTC_OFFSET).replace(tzinfo=None)
    # No TZ info so not going to assume anything, return as-is.
    return dt

def percent_cb(name, complete, total):
    """ Callback for updating target progress """
    logger.debug(
        "{}: {} transferred out of {}".format(
            name, sizeof_fmt(complete), sizeof_fmt(total)
        )
    )
    progress.update_target(name, complete, total)

def now(self):
		"""
		Return a :py:class:`datetime.datetime` instance representing the current time.

		:rtype: :py:class:`datetime.datetime`
		"""
		if self.use_utc:
			return datetime.datetime.utcnow()
		else:
			return datetime.datetime.now()

def to_pascal_case(s):
    """Transform underscore separated string to pascal case

    """
    return re.sub(r'(?!^)_([a-zA-Z])', lambda m: m.group(1).upper(), s.capitalize())

def now(self):
		"""
		Return a :py:class:`datetime.datetime` instance representing the current time.

		:rtype: :py:class:`datetime.datetime`
		"""
		if self.use_utc:
			return datetime.datetime.utcnow()
		else:
			return datetime.datetime.now()

def _convert_date_to_dict(field_date):
        """
        Convert native python ``datetime.date`` object  to a format supported by the API
        """
        return {DAY: field_date.day, MONTH: field_date.month, YEAR: field_date.year}

def ToDatetime(self):
    """Converts Timestamp to datetime."""
    return datetime.utcfromtimestamp(
        self.seconds + self.nanos / float(_NANOS_PER_SECOND))

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

def parse_timestamp(timestamp):
    """Parse ISO8601 timestamps given by github API."""
    dt = dateutil.parser.parse(timestamp)
    return dt.astimezone(dateutil.tz.tzutc())

def add_to_js(self, name, var):
        """Add an object to Javascript."""
        frame = self.page().mainFrame()
        frame.addToJavaScriptWindowObject(name, var)

def fromtimestamp(cls, timestamp):
    """Returns a datetime object of a given timestamp (in local tz)."""
    d = cls.utcfromtimestamp(timestamp)
    return d.astimezone(localtz())

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

def datetime64_to_datetime(dt):
    """ convert numpy's datetime64 to datetime """
    dt64 = np.datetime64(dt)
    ts = (dt64 - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
    return datetime.datetime.utcfromtimestamp(ts)

def batch_tensor(self, name):
        """ A buffer of a given value in a 'flat' (minibatch-indexed) format """
        if name in self.transition_tensors:
            return tensor_util.merge_first_two_dims(self.transition_tensors[name])
        else:
            return self.rollout_tensors[name]

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

def export(defn):
    """Decorator to explicitly mark functions that are exposed in a lib."""
    globals()[defn.__name__] = defn
    __all__.append(defn.__name__)
    return defn

def parse(source, remove_comments=True, **kw):
    """Thin wrapper around ElementTree.parse"""
    return ElementTree.parse(source, SourceLineParser(), **kw)

def decorator(func):
  r"""Makes the passed decorators to support optional args.
  """
  def wrapper(__decorated__=None, *Args, **KwArgs):
    if __decorated__ is None: # the decorator has some optional arguments.
      return lambda _func: func(_func, *Args, **KwArgs)

    else:
      return func(__decorated__, *Args, **KwArgs)

  return wrap(wrapper, func)

def show_image(self, key):
        """Show image (item is a PIL image)"""
        data = self.model.get_data()
        data[key].show()

def get_default_args(func):
    """
    returns a dictionary of arg_name:default_values for the input function
    """
    args, varargs, keywords, defaults = getargspec_no_self(func)
    return dict(zip(args[-len(defaults):], defaults))

def _interval_to_bound_points(array):
    """
    Helper function which returns an array
    with the Intervals' boundaries.
    """

    array_boundaries = np.array([x.left for x in array])
    array_boundaries = np.concatenate(
        (array_boundaries, np.array([array[-1].right])))

    return array_boundaries

def closing_plugin(self, cancelable=False):
        """Perform actions before parent main window is closed"""
        self.dialog_manager.close_all()
        self.shell.exit_interpreter()
        return True

def test():        
    """Local test."""
    from spyder.utils.qthelpers import qapplication
    app = qapplication()
    dlg = ProjectDialog(None)
    dlg.show()
    sys.exit(app.exec_())

def del_label(self, name):
        """Delete a label by name."""
        labels_tag = self.root[0]
        labels_tag.remove(self._find_label(name))

def mixedcase(path):
    """Removes underscores and capitalizes the neighbouring character"""
    words = path.split('_')
    return words[0] + ''.join(word.title() for word in words[1:])

def delete_all_eggs(self):
        """ delete all the eggs in the directory specified """
        path_to_delete = os.path.join(self.egg_directory, "lib", "python")
        if os.path.exists(path_to_delete):
            shutil.rmtree(path_to_delete)

def get_system_cpu_times():
    """Return system CPU times as a namedtuple."""
    user, nice, system, idle = _psutil_osx.get_system_cpu_times()
    return _cputimes_ntuple(user, nice, system, idle)

def remove(self, document_id, namespace, timestamp):
        """Removes documents from Solr

        The input is a python dictionary that represents a mongo document.
        """
        self.solr.delete(id=u(document_id),
                         commit=(self.auto_commit_interval == 0))

def update_hash_from_str(hsh, str_input):
    """
    Convert a str to object supporting buffer API and update a hash with it.
    """
    byte_input = str(str_input).encode("UTF-8")
    hsh.update(byte_input)

def make_regex(separator):
    """Utility function to create regexp for matching escaped separators
    in strings.

    """
    return re.compile(r'(?:' + re.escape(separator) + r')?((?:[^' +
                      re.escape(separator) + r'\\]|\\.)+)')

def dictify(a_named_tuple):
    """Transform a named tuple into a dictionary"""
    return dict((s, getattr(a_named_tuple, s)) for s in a_named_tuple._fields)

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

def c_str(string):
    """"Convert a python string to C string."""
    if not isinstance(string, str):
        string = string.decode('ascii')
    return ctypes.c_char_p(string.encode('utf-8'))

def endline_semicolon_check(self, original, loc, tokens):
        """Check for semicolons at the end of lines."""
        return self.check_strict("semicolon at end of line", original, loc, tokens)

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

def get(self):
        """Get the highest priority Processing Block from the queue."""
        with self._mutex:
            entry = self._queue.pop()
            del self._block_map[entry[2]]
            return entry[2]

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

def from_json(cls, json_str):
        """Deserialize the object from a JSON string."""
        d = json.loads(json_str)
        return cls.from_dict(d)

def update(kernel=False):
    """
    Upgrade all packages, skip obsoletes if ``obsoletes=0`` in ``yum.conf``.

    Exclude *kernel* upgrades by default.
    """
    manager = MANAGER
    cmds = {'yum -y --color=never': {False: '--exclude=kernel* update', True: 'update'}}
    cmd = cmds[manager][kernel]
    run_as_root("%(manager)s %(cmd)s" % locals())

def guess_encoding(text, default=DEFAULT_ENCODING):
    """Guess string encoding.

    Given a piece of text, apply character encoding detection to
    guess the appropriate encoding of the text.
    """
    result = chardet.detect(text)
    return normalize_result(result, default=default)

def commajoin_as_strings(iterable):
    """ Join the given iterable with ',' """
    return _(u',').join((six.text_type(i) for i in iterable))

def supports_color():
    """
    Returns True if the running system's terminal supports color, and False
    otherwise.
    """
    unsupported_platform = (sys.platform in ('win32', 'Pocket PC'))
    # isatty is not always implemented, #6223.
    is_a_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    if unsupported_platform or not is_a_tty:
        return False
    return True

def seconds_to_hms(seconds):
    """
    Converts seconds float to 'hh:mm:ss.ssssss' format.
    """
    hours = int(seconds / 3600.0)
    minutes = int((seconds / 60.0) % 60.0)
    secs = float(seconds % 60.0)
    return "{0:02d}:{1:02d}:{2:02.6f}".format(hours, minutes, secs)

def __contains__(self, key):
        """
        Invoked when determining whether a specific key is in the dictionary
        using `key in d`.

        The key is looked up case-insensitively.
        """
        k = self._real_key(key)
        return k in self._data

def get_truetype(value):
    """Convert a string to a pythonized parameter."""
    if value in ["true", "True", "y", "Y", "yes"]:
        return True
    if value in ["false", "False", "n", "N", "no"]:
        return False
    if value.isdigit():
        return int(value)
    return str(value)

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

def timed_rotating_file_handler(name, logname, filename, when='h',
                                interval=1, backupCount=0,
                                encoding=None, delay=False, utc=False):
    """
    A Bark logging handler logging output to a named file.  At
    intervals specified by the 'when', the file will be rotated, under
    control of 'backupCount'.

    Similar to logging.handlers.TimedRotatingFileHandler.
    """

    return wrap_log_handler(logging.handlers.TimedRotatingFileHandler(
        filename, when=when, interval=interval, backupCount=backupCount,
        encoding=encoding, delay=delay, utc=utc))

def is_identifier(string):
    """Check if string could be a valid python identifier

    :param string: string to be tested
    :returns: True if string can be a python identifier, False otherwise
    :rtype: bool
    """
    matched = PYTHON_IDENTIFIER_RE.match(string)
    return bool(matched) and not keyword.iskeyword(string)

def uniform_iterator(sequence):
    """Uniform (key, value) iteration on a `dict`,
    or (idx, value) on a `list`."""

    if isinstance(sequence, abc.Mapping):
        return six.iteritems(sequence)
    else:
        return enumerate(sequence)

def _guess_type(val):
        """Guess the input type of the parameter based off the default value, if unknown use text"""
        if isinstance(val, bool):
            return "choice"
        elif isinstance(val, int):
            return "number"
        elif isinstance(val, float):
            return "number"
        elif isinstance(val, str):
            return "text"
        elif hasattr(val, 'read'):
            return "file"
        else:
            return "text"

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

def _platform_is_windows(platform=sys.platform):
        """Is the current OS a Windows?"""
        matched = platform in ('cygwin', 'win32', 'win64')
        if matched:
            error_msg = "Windows isn't supported yet"
            raise OSError(error_msg)
        return matched

def _xls2col_widths(self, worksheet, tab):
        """Updates col_widths in code_array"""

        for col in xrange(worksheet.ncols):
            try:
                xls_width = worksheet.colinfo_map[col].width
                pys_width = self.xls_width2pys_width(xls_width)
                self.code_array.col_widths[col, tab] = pys_width

            except KeyError:
                pass

def keys_to_snake_case(camel_case_dict):
    """
    Make a copy of a dictionary with all keys converted to snake case. This is just calls to_snake_case on
    each of the keys in the dictionary and returns a new dictionary.

    :param camel_case_dict: Dictionary with the keys to convert.
    :type camel_case_dict: Dictionary.

    :return: Dictionary with the keys converted to snake case.
    """
    return dict((to_snake_case(key), value) for (key, value) in camel_case_dict.items())

def _bytes_to_json(value):
    """Coerce 'value' to an JSON-compatible representation."""
    if isinstance(value, bytes):
        value = base64.standard_b64encode(value).decode("ascii")
    return value

def dict_hash(dct):
    """Return a hash of the contents of a dictionary"""
    dct_s = json.dumps(dct, sort_keys=True)

    try:
        m = md5(dct_s)
    except TypeError:
        m = md5(dct_s.encode())

    return m.hexdigest()

def int_to_date(date):
    """
    Convert an int of form yyyymmdd to a python date object.
    """

    year = date // 10**4
    month = date % 10**4 // 10**2
    day = date % 10**2

    return datetime.date(year, month, day)

def filter_dict(d, keys):
    """
    Creates a new dict from an existing dict that only has the given keys
    """
    return {k: v for k, v in d.items() if k in keys}

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

def dict_update_newkeys(dict_, dict2):
    """ Like dict.update, but does not overwrite items """
    for key, val in six.iteritems(dict2):
        if key not in dict_:
            dict_[key] = val

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

def update(self, other_dict):
        """update() extends rather than replaces existing key lists."""
        for key, value in iter_multi_items(other_dict):
            MultiDict.add(self, key, value)

def _internet_on(address):
    """
    Check to see if the internet is on by pinging a set address.
    :param address: the IP or address to hit
    :return: a boolean - true if can be reached, false if not.
    """
    try:
        urllib2.urlopen(address, timeout=1)
        return True
    except urllib2.URLError as err:
        return False

def _defaultdict(dct, fallback=_illegal_character):
    """Wraps the given dictionary such that the given fallback function will be called when a nonexistent key is
    accessed.
    """
    out = defaultdict(lambda: fallback)
    for k, v in six.iteritems(dct):
        out[k] = v
    return out

def is_json_file(filename, show_warnings = False):
    """Check configuration file type is JSON
    Return a boolean indicating wheather the file is JSON format or not
    """
    try:
        config_dict = load_config(filename, file_type = "json")
        is_json = True
    except:
        is_json = False
    return(is_json)

def _remove_dict_keys_with_value(dict_, val):
  """Removes `dict` keys which have have `self` as value."""
  return {k: v for k, v in dict_.items() if v is not val}

def post_commit_hook(argv):
    """Hook: for checking commit message."""
    _, stdout, _ = run("git log -1 --format=%B HEAD")
    message = "\n".join(stdout)
    options = {"allow_empty": True}

    if not _check_message(message, options):
        click.echo(
            "Commit message errors (fix with 'git commit --amend').",
            file=sys.stderr)
        return 1  # it should not fail with exit
    return 0

def setdefaults(dct, defaults):
    """Given a target dct and a dict of {key:default value} pairs,
    calls setdefault for all of those pairs."""
    for key in defaults:
        dct.setdefault(key, defaults[key])

    return dct

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

def dict_to_html_attrs(dict_):
    """
    Banana banana
    """
    res = ' '.join('%s="%s"' % (k, v) for k, v in dict_.items())
    return res

def is_binary(filename):
    """ Returns True if the file is binary

    """
    with open(filename, 'rb') as fp:
        data = fp.read(1024)
        if not data:
            return False
        if b'\0' in data:
            return True
        return False

def dict_to_querystring(dictionary):
    """Converts a dict to a querystring suitable to be appended to a URL."""
    s = u""
    for d in dictionary.keys():
        s = unicode.format(u"{0}{1}={2}&", s, d, dictionary[d])
    return s[:-1]

def _check_elements_equal(lst):
    """
    Returns true if all of the elements in the list are equal.
    """
    assert isinstance(lst, list), "Input value must be a list."
    return not lst or lst.count(lst[0]) == len(lst)

def dict_to_querystring(dictionary):
    """Converts a dict to a querystring suitable to be appended to a URL."""
    s = u""
    for d in dictionary.keys():
        s = unicode.format(u"{0}{1}={2}&", s, d, dictionary[d])
    return s[:-1]

def _check_elements_equal(lst):
    """
    Returns true if all of the elements in the list are equal.
    """
    assert isinstance(lst, list), "Input value must be a list."
    return not lst or lst.count(lst[0]) == len(lst)

def nonull_dict(self):
        """Like dict, but does not hold any null values.

        :return:

        """
        return {k: v for k, v in six.iteritems(self.dict) if v and k != '_codes'}

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

def updateFromKwargs(self, properties, kwargs, collector, **unused):
        """Primary entry point to turn 'kwargs' into 'properties'"""
        properties[self.name] = self.getFromKwargs(kwargs)

def is_callable(*p):
    """ True if all the args are functions and / or subroutines
    """
    import symbols
    return all(isinstance(x, symbols.FUNCTION) for x in p)

async def disconnect(self):
        """ Disconnect from target. """
        if not self.connected:
            return

        self.writer.close()
        self.reader = None
        self.writer = None

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

def test():
    """Run the unit tests."""
    import unittest
    tests = unittest.TestLoader().discover('tests')
    unittest.TextTestRunner(verbosity=2).run(tests)

def is_datetime_like(dtype):
    """Check if a dtype is a subclass of the numpy datetime types
    """
    return (np.issubdtype(dtype, np.datetime64) or
            np.issubdtype(dtype, np.timedelta64))

def serialize_json_string(self, value):
        """
        Tries to load an encoded json string back into an object
        :param json_string:
        :return:
        """

        # Check if the value might be a json string
        if not isinstance(value, six.string_types):
            return value

        # Make sure it starts with a brace
        if not value.startswith('{') or value.startswith('['):
            return value

        # Try to load the string
        try:
            return json.loads(value)
        except:
            return value

def is_defined(self, objtxt, force_import=False):
        """Return True if object is defined"""
        return self.interpreter.is_defined(objtxt, force_import)

def get_hline():
    """ gets a horiztonal line """
    return Window(
        width=LayoutDimension.exact(1),
        height=LayoutDimension.exact(1),
        content=FillControl('-', token=Token.Line))

def group_exists(groupname):
    """Check if a group exists"""
    try:
        grp.getgrnam(groupname)
        group_exists = True
    except KeyError:
        group_exists = False
    return group_exists

def sync(self, recursive=False):
        """
        Syncs the information from this item to the tree and view.
        """
        self.syncTree(recursive=recursive)
        self.syncView(recursive=recursive)

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

def get_distance_between_two_points(self, one, two):
        """Returns the distance between two XYPoints."""
        dx = one.x - two.x
        dy = one.y - two.y
        return math.sqrt(dx * dx + dy * dy)

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

def post_process(self):
        """ Apply last 2D transforms"""
        self.image.putdata(self.pixels)
        self.image = self.image.transpose(Image.ROTATE_90)

def _not_none(items):
    """Whether the item is a placeholder or contains a placeholder."""
    if not isinstance(items, (tuple, list)):
        items = (items,)
    return all(item is not _none for item in items)

def delete_all_from_db():
    """Clear the database.

    Used for testing and debugging.

    """
    # The models.CASCADE property is set on all ForeignKey fields, so tables can
    # be deleted in any order without breaking constraints.
    for model in django.apps.apps.get_models():
        model.objects.all().delete()

def is_complex(dtype):
  """Returns whether this is a complex floating point type."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'is_complex'):
    return dtype.is_complex
  return np.issubdtype(np.dtype(dtype), np.complex)

def delete(build_folder):
    """Delete build directory and all its contents.
    """
    if _meta_.del_build in ["on", "ON"] and os.path.exists(build_folder):
        shutil.rmtree(build_folder)

def _stdin_ready_posix():
    """Return True if there's something to read on stdin (posix version)."""
    infds, outfds, erfds = select.select([sys.stdin],[],[],0)
    return bool(infds)

def json_response(data, status=200):
    """Return a JsonResponse. Make sure you have django installed first."""
    from django.http import JsonResponse
    return JsonResponse(data=data, status=status, safe=isinstance(data, dict))

def _is_path(s):
    """Return whether an object is a path."""
    if isinstance(s, string_types):
        try:
            return op.exists(s)
        except (OSError, ValueError):
            return False
    else:
        return False

def see_doc(obj_with_doc):
    """Copy docstring from existing object to the decorated callable."""
    def decorator(fn):
        fn.__doc__ = obj_with_doc.__doc__
        return fn
    return decorator

def isToneCal(self):
        """Whether the currently selected calibration stimulus type is the calibration curve

        :returns: boolean -- if the current combo box selection is calibration curve
        """
        return self.ui.calTypeCmbbx.currentIndex() == self.ui.calTypeCmbbx.count() -1

def hmsToDeg(h, m, s):
    """Convert RA hours, minutes, seconds into an angle in degrees."""
    return h * degPerHMSHour + m * degPerHMSMin + s * degPerHMSSec

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

def prepare(doc):
    """Sets the caption_found and plot_found variables to False."""
    doc.caption_found = False
    doc.plot_found = False
    doc.listings_counter = 0

def validate(key):
    """Check that the key is a string or bytestring.

    That's the only valid type of key.
    """
    if not isinstance(key, (str, bytes)):
        raise KeyError('Key must be of type str or bytes, found type {}'.format(type(key)))

def _normal_prompt(self):
        """
        Flushes the prompt before requesting the input

        :return: The command line
        """
        sys.stdout.write(self.__get_ps1())
        sys.stdout.flush()
        return safe_input()

def maxDepth(self, currentDepth=0):
        """Compute the depth of the longest branch of the tree"""
        if not any((self.left, self.right)):
            return currentDepth
        result = 0
        for child in (self.left, self.right):
            if child:
                result = max(result, child.maxDepth(currentDepth + 1))
        return result

def from_rectangle(box):
        """ Create a vector randomly within the given rectangle. """
        x = box.left + box.width * random.uniform(0, 1)
        y = box.bottom + box.height * random.uniform(0, 1)
        return Vector(x, y)

def launched():
    """Test whether the current python environment is the correct lore env.

    :return:  :any:`True` if the environment is launched
    :rtype: bool
    """
    if not PREFIX:
        return False

    return os.path.realpath(sys.prefix) == os.path.realpath(PREFIX)

def hline(self, x, y, width, color):
        """Draw a horizontal line up to a given length."""
        self.rect(x, y, width, 1, color, fill=True)

def is_sequence(obj):
    """Check if `obj` is a sequence, but not a string or bytes."""
    return isinstance(obj, Sequence) and not (
        isinstance(obj, str) or BinaryClass.is_valid_type(obj))

def isnamedtuple(obj):
    """Heuristic check if an object is a namedtuple."""
    return isinstance(obj, tuple) \
           and hasattr(obj, "_fields") \
           and hasattr(obj, "_asdict") \
           and callable(obj._asdict)

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

def print_yaml(o):
    """Pretty print an object as YAML."""
    print(yaml.dump(o, default_flow_style=False, indent=4, encoding='utf-8'))

def issuperset(self, other):
        """Report whether this RangeSet contains another set."""
        self._binary_sanity_check(other)
        return set.issuperset(self, other)

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

def check(text):
    """Check the text."""
    err = "misc.currency"
    msg = u"Incorrect use of symbols in {}."

    symbols = [
        "\$[\d]* ?(?:dollars|usd|us dollars)"
    ]

    return existence_check(text, symbols, err, msg)

def listlike(obj):
    """Is an object iterable like a list (and not a string)?"""
    
    return hasattr(obj, "__iter__") \
    and not issubclass(type(obj), str)\
    and not issubclass(type(obj), unicode)

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

def keys(self):
        """Return a list of all keys in the dictionary.

        Returns:
            list of str: [key1,key2,...,keyN]
        """
        all_keys = [k.decode('utf-8') for k,v in self.rdb.hgetall(self.session_hash).items()]
        return all_keys

def _valid_other_type(x, types):
    """
    Do all elements of x have a type from types?
    """
    return all(any(isinstance(el, t) for t in types) for el in np.ravel(x))

def escape_tex(value):
  """
  Make text tex safe
  """
  newval = value
  for pattern, replacement in LATEX_SUBS:
    newval = pattern.sub(replacement, newval)
  return newval

def _pip_exists(self):
        """Returns True if pip exists inside the virtual environment. Can be
        used as a naive way to verify that the environment is installed."""
        return os.path.isfile(os.path.join(self.path, 'bin', 'pip'))

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

def is_datetime_like(dtype):
    """Check if a dtype is a subclass of the numpy datetime types
    """
    return (np.issubdtype(dtype, np.datetime64) or
            np.issubdtype(dtype, np.timedelta64))

def hidden_cursor(self):
        """Return a context manager that hides the cursor while inside it and
        makes it visible on leaving."""
        self.stream.write(self.hide_cursor)
        try:
            yield
        finally:
            self.stream.write(self.normal_cursor)

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

def copy(doc, dest, src):
    """Copy element from sequence, member from mapping.

    :param doc: the document base
    :param dest: the destination
    :type dest: Pointer
    :param src: the source
    :type src: Pointer
    :return: the new object
    """

    return Target(doc).copy(dest, src).document

def is_string(val):
    """Determines whether the passed value is a string, safe for 2/3."""
    try:
        basestring
    except NameError:
        return isinstance(val, str)
    return isinstance(val, basestring)

def read_from_file(file_path, encoding="utf-8"):
    """
    Read helper method

    :type file_path: str|unicode
    :type encoding: str|unicode
    :rtype: str|unicode
    """
    with codecs.open(file_path, "r", encoding) as f:
        return f.read()

def _stdin_ready_posix():
    """Return True if there's something to read on stdin (posix version)."""
    infds, outfds, erfds = select.select([sys.stdin],[],[],0)
    return bool(infds)

def _is_root():
    """Checks if the user is rooted."""
    import os
    import ctypes
    try:
        return os.geteuid() == 0
    except AttributeError:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    return False

def _valid_other_type(x, types):
    """
    Do all elements of x have a type from types?
    """
    return all(any(isinstance(el, t) for t in types) for el in np.ravel(x))

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

def user_in_all_groups(user, groups):
    """Returns True if the given user is in all given groups"""
    return user_is_superuser(user) or all(user_in_group(user, group) for group in groups)

def items(self):
    """Return a list of the (name, value) pairs of the enum.

    These are returned in the order they were defined in the .proto file.
    """
    return [(value_descriptor.name, value_descriptor.number)
            for value_descriptor in self._enum_type.values]

def n_choose_k(n, k):
    """ get the number of quartets as n-choose-k. This is used
    in equal splits to decide whether a split should be exhaustively sampled
    or randomly sampled. Edges near tips can be exhaustive while highly nested
    edges probably have too many quartets
    """
    return int(reduce(MUL, (Fraction(n-i, i+1) for i in range(k)), 1))

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

def revnet_164_cifar():
  """Tiny hparams suitable for CIFAR/etc."""
  hparams = revnet_cifar_base()
  hparams.bottleneck = True
  hparams.num_channels = [16, 32, 64]
  hparams.num_layers_per_block = [8, 8, 8]
  return hparams

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

def mtf_image_transformer_cifar_mp_4x():
  """Data parallel CIFAR parameters."""
  hparams = mtf_image_transformer_base_cifar()
  hparams.mesh_shape = "model:4;batch:8"
  hparams.layout = "batch:batch;d_ff:model;heads:model"
  hparams.batch_size = 32
  hparams.num_heads = 8
  hparams.d_ff = 8192
  return hparams

def image_set_aspect(aspect=1.0, axes="gca"):
    """
    sets the aspect ratio of the current zoom level of the imshow image
    """
    if axes is "gca": axes = _pylab.gca()

    e = axes.get_images()[0].get_extent()
    axes.set_aspect(abs((e[1]-e[0])/(e[3]-e[2]))/aspect)

def Flush(self):
    """Flush all items from cache."""
    while self._age:
      node = self._age.PopLeft()
      self.KillObject(node.data)

    self._hash = dict()

def _propagate_mean(mean, linop, dist):
  """Propagate a mean through linear Gaussian transformation."""
  return linop.matmul(mean) + dist.mean()[..., tf.newaxis]

def invalidate_cache(cpu, address, size):
        """ remove decoded instruction from instruction cache """
        cache = cpu.instruction_cache
        for offset in range(size):
            if address + offset in cache:
                del cache[address + offset]

def convertToBool():
    """ Convert a byte value to boolean (0 or 1) if
    the global flag strictBool is True
    """
    if not OPTIONS.strictBool.value:
        return []

    REQUIRES.add('strictbool.asm')

    result = []
    result.append('pop af')
    result.append('call __NORMALIZE_BOOLEAN')
    result.append('push af')

    return result

def normalize(x, min_value, max_value):
    """Normalize value between min and max values.
    It also clips the values, so that you cannot have values higher or lower
    than 0 - 1."""
    x = (x - min_value) / (max_value - min_value)
    return clip(x, 0, 1)

def prepare_for_reraise(error, exc_info=None):
    """Prepares the exception for re-raising with reraise method.

    This method attaches type and traceback info to the error object
    so that reraise can properly reraise it using this info.

    """
    if not hasattr(error, "_type_"):
        if exc_info is None:
            exc_info = sys.exc_info()
        error._type_ = exc_info[0]
        error._traceback = exc_info[2]
    return error

def close_all_but_this(self):
        """Close all files but the current one"""
        self.close_all_right()
        for i in range(0, self.get_stack_count()-1  ):
            self.close_file(0)

def eval_in_system_namespace(self, exec_str):
        """
            Get Callable for specified string (for GUI-based editing)
        """
        ns = self.cmd_namespace
        try:
            return eval(exec_str, ns)
        except Exception as e:
            self.logger.warning('Could not execute %s, gave error %s', exec_str, e)
            return None

def _close_socket(self):
        """Shutdown and close the Socket.

        :return:
        """
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except (OSError, socket.error):
            pass
        self.socket.close()

def exec_function(ast, globals_map):
    """Execute a python code object in the given environment.

    Args:
      globals_map: Dictionary to use as the globals context.
    Returns:
      locals_map: Dictionary of locals from the environment after execution.
    """
    locals_map = globals_map
    exec ast in globals_map, locals_map
    return locals_map

def cleanup(self, app):
        """Close all connections."""
        if hasattr(self.database.obj, 'close_all'):
            self.database.close_all()

def get_unicode_str(obj):
    """Makes sure obj is a unicode string."""
    if isinstance(obj, six.text_type):
        return obj
    if isinstance(obj, six.binary_type):
        return obj.decode("utf-8", errors="ignore")
    return six.text_type(obj)

def close_all_but_this(self):
        """Close all files but the current one"""
        self.close_all_right()
        for i in range(0, self.get_stack_count()-1  ):
            self.close_file(0)

def exp_fit_fun(x, a, tau, c):
    """Function used to fit the exponential decay."""
    # pylint: disable=invalid-name
    return a * np.exp(-x / tau) + c

def _findNearest(arr, value):
    """ Finds the value in arr that value is closest to
    """
    arr = np.array(arr)
    # find nearest value in array
    idx = (abs(arr-value)).argmin()
    return arr[idx]

def gauss_pdf(x, mu, sigma):
    """Normalized Gaussian"""
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - mu) ** 2 / 2. / sigma ** 2)

def remove_examples_all():
    """remove arduino/examples/all directory.

    :rtype: None

    """
    d = examples_all_dir()
    if d.exists():
        log.debug('remove %s', d)
        d.rmtree()
    else:
        log.debug('nothing to remove: %s', d)

def resources(self):
        """Retrieve contents of each page of PDF"""
        return [self.pdf.getPage(i) for i in range(self.pdf.getNumPages())]

def cli_command_quit(self, msg):
        """\
        kills the child and exits
        """
        if self.state == State.RUNNING and self.sprocess and self.sprocess.proc:
            self.sprocess.proc.kill()
        else:
            sys.exit(0)

def dot(self, w):
        """Return the dotproduct between self and another vector."""

        return sum([x * y for x, y in zip(self, w)])

def printc(cls, txt, color=colors.red):
        """Print in color."""
        print(cls.color_txt(txt, color))

def need_update(a, b):
    """
    Check if file a is newer than file b and decide whether or not to update
    file b. Can generalize to two lists.
    """
    a = listify(a)
    b = listify(b)

    return any((not op.exists(x)) for x in b) or \
           all((os.stat(x).st_size == 0 for x in b)) or \
           any(is_newer_file(x, y) for x in a for y in b)

def lengths( self ):
        """
        The cell lengths.

        Args:
            None

        Returns:
            (np.array(a,b,c)): The cell lengths.
        """
        return( np.array( [ math.sqrt( sum( row**2 ) ) for row in self.matrix ] ) )

def random_str(size=10):
    """
    create random string of selected size

    :param size: int, length of the string
    :return: the string
    """
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(size))

def get_table_columns(dbconn, tablename):
    """
    Return a list of tuples specifying the column name and type
    """
    cur = dbconn.cursor()
    cur.execute("PRAGMA table_info('%s');" % tablename)
    info = cur.fetchall()
    cols = [(i[1], i[2]) for i in info]
    return cols

def remove_duplicates(lst):
    """
    Emulate what a Python ``set()`` does, but keeping the element's order.
    """
    dset = set()
    return [l for l in lst if l not in dset and not dset.add(l)]

def _on_select(self, *args):
        """
        Function bound to event of selection in the Combobox, calls callback if callable
        
        :param args: Tkinter event
        """
        if callable(self.__callback):
            self.__callback(self.selection)

def fft_spectrum(frames, fft_points=512):
    """This function computes the one-dimensional n-point discrete Fourier
    Transform (DFT) of a real-valued array by means of an efficient algorithm
    called the Fast Fourier Transform (FFT). Please refer to
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.rfft.html
    for further details.

    Args:
        frames (array): The frame array in which each row is a frame.
        fft_points (int): The length of FFT. If fft_length is greater than frame_len, the frames will be zero-padded.

    Returns:
            array: The fft spectrum.
            If frames is an num_frames x sample_per_frame matrix, output
            will be num_frames x FFT_LENGTH.
    """
    SPECTRUM_VECTOR = np.fft.rfft(frames, n=fft_points, axis=-1, norm=None)
    return np.absolute(SPECTRUM_VECTOR)

def isetdiff_flags(list1, list2):
    """
    move to util_iter
    """
    set2 = set(list2)
    return (item not in set2 for item in list1)

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

def file_empty(fp):
    """Determine if a file is empty or not."""
    # for python 2 we need to use a homemade peek()
    if six.PY2:
        contents = fp.read()
        fp.seek(0)
        return not bool(contents)

    else:
        return not fp.peek()

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

def get_file_size(filename):
    """
    Get the file size of a given file

    :param filename: string: pathname of a file
    :return: human readable filesize
    """
    if os.path.isfile(filename):
        return convert_size(os.path.getsize(filename))
    return None

def _check_for_int(x):
    """
    This is a compatibility function that takes a C{float} and converts it to an
    C{int} if the values are equal.
    """
    try:
        y = int(x)
    except (OverflowError, ValueError):
        pass
    else:
        # There is no way in AMF0 to distinguish between integers and floats
        if x == x and y == x:
            return y

    return x

def fill_form(form, data):
    """Prefill form with data.

    :param form: The form to fill.
    :param data: The data to insert in the form.
    :returns: A pre-filled form.
    """
    for (key, value) in data.items():
        if hasattr(form, key):
            if isinstance(value, dict):
                fill_form(getattr(form, key), value)
            else:
                getattr(form, key).data = value
    return form

def check_clang_apply_replacements_binary(args):
  """Checks if invoking supplied clang-apply-replacements binary works."""
  try:
    subprocess.check_call([args.clang_apply_replacements_binary, '--version'])
  except:
    print('Unable to run clang-apply-replacements. Is clang-apply-replacements '
          'binary correctly specified?', file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

def _maybe_fill(arr, fill_value=np.nan):
    """
    if we have a compatible fill_value and arr dtype, then fill
    """
    if _isna_compat(arr, fill_value):
        arr.fill(fill_value)
    return arr

def extract_alzip (archive, compression, cmd, verbosity, interactive, outdir):
    """Extract a ALZIP archive."""
    return [cmd, '-d', outdir, archive]

def _maybe_fill(arr, fill_value=np.nan):
    """
    if we have a compatible fill_value and arr dtype, then fill
    """
    if _isna_compat(arr, fill_value):
        arr.fill(fill_value)
    return arr

def get_lons_from_cartesian(x__, y__):
    """Get longitudes from cartesian coordinates.
    """
    return rad2deg(arccos(x__ / sqrt(x__ ** 2 + y__ ** 2))) * sign(y__)

def filter_(stream_spec, filter_name, *args, **kwargs):
    """Alternate name for ``filter``, so as to not collide with the
    built-in python ``filter`` operator.
    """
    return filter(stream_spec, filter_name, *args, **kwargs)

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

def find_lt(a, x):
    """Find rightmost value less than x."""
    i = bs.bisect_left(a, x)
    if i: return i - 1
    raise ValueError

def get_stationary_distribution(self):
        """Compute the stationary distribution of states.
        """
        # The stationary distribution is proportional to the left-eigenvector
        # associated with the largest eigenvalue (i.e., 1) of the transition
        # matrix.
        check_is_fitted(self, "transmat_")
        eigvals, eigvecs = np.linalg.eig(self.transmat_.T)
        eigvec = np.real_if_close(eigvecs[:, np.argmax(eigvals)])
        return eigvec / eigvec.sum()

def apply_fit(xy,coeffs):
    """ Apply the coefficients from a linear fit to
        an array of x,y positions.

        The coeffs come from the 'coeffs' member of the
        'fit_arrays()' output.
    """
    x_new = coeffs[0][2] + coeffs[0][0]*xy[:,0] + coeffs[0][1]*xy[:,1]
    y_new = coeffs[1][2] + coeffs[1][0]*xy[:,0] + coeffs[1][1]*xy[:,1]

    return x_new,y_new

def _tf_squared_euclidean(X, Y):
        """Squared Euclidean distance between the rows of `X` and `Y`.
        """
        return tf.reduce_sum(tf.pow(tf.subtract(X, Y), 2), axis=1)

def exp_fit_fun(x, a, tau, c):
    """Function used to fit the exponential decay."""
    # pylint: disable=invalid-name
    return a * np.exp(-x / tau) + c

def euclidean(x, y):
    """Standard euclidean distance.

    ..math::
        D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2
    return np.sqrt(result)

def create_table_from_fits(fitsfile, hduname, colnames=None):
    """Memory efficient function for loading a table from a FITS
    file."""

    if colnames is None:
        return Table.read(fitsfile, hduname)

    cols = []
    with fits.open(fitsfile, memmap=True) as h:
        for k in colnames:
            data = h[hduname].data.field(k)
            cols += [Column(name=k, data=data)]
    return Table(cols)

def _gcd_array(X):
    """
    Return the largest real value h such that all elements in x are integer
    multiples of h.
    """
    greatest_common_divisor = 0.0
    for x in X:
        greatest_common_divisor = _gcd(greatest_common_divisor, x)

    return greatest_common_divisor

def lint(args):
    """Run lint checks using flake8."""
    application = get_current_application()
    if not args:
        args = [application.name, 'tests']
    args = ['flake8'] + list(args)
    run.main(args, standalone_mode=False)

def torecarray(*args, **kwargs):
    """
    Convenient shorthand for ``toarray(*args, **kwargs).view(np.recarray)``.

    """

    import numpy as np
    return toarray(*args, **kwargs).view(np.recarray)

def _type_bool(label,default=False):
    """Shortcut fot boolean like fields"""
    return label, abstractSearch.nothing, abstractRender.boolen, default

def join_cols(cols):
    """Join list of columns into a string for a SQL query"""
    return ", ".join([i for i in cols]) if isinstance(cols, (list, tuple, set)) else cols

def parse_form(self, req, name, field):
        """Pull a form value from the request."""
        return core.get_value(req.POST, name, field)

def type_converter(text):
    """ I convert strings into integers, floats, and strings! """
    if text.isdigit():
        return int(text), int

    try:
        return float(text), float
    except ValueError:
        return text, STRING_TYPE

def cors_header(func):
    """ @cors_header decorator adds CORS headers """

    @wraps(func)
    def wrapper(self, request, *args, **kwargs):
        res = func(self, request, *args, **kwargs)
        request.setHeader('Access-Control-Allow-Origin', '*')
        request.setHeader('Access-Control-Allow-Headers', 'Content-Type, Access-Control-Allow-Headers, Authorization, X-Requested-With')
        return res

    return wrapper

def pdf(x, mu, std):
    """Probability density function (normal distribution)"""
    return (1.0 / (std * sqrt(2 * pi))) * np.exp(-(x - mu) ** 2 / (2 * std ** 2))

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

def pdf(x, mu, std):
    """Probability density function (normal distribution)"""
    return (1.0 / (std * sqrt(2 * pi))) * np.exp(-(x - mu) ** 2 / (2 * std ** 2))

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

def _spawn_kafka_consumer_thread(self):
        """Spawns a kafka continuous consumer thread"""
        self.logger.debug("Spawn kafka consumer thread""")
        self._consumer_thread = Thread(target=self._consumer_loop)
        self._consumer_thread.setDaemon(True)
        self._consumer_thread.start()

def flatpages_link_list(request):
    """
    Returns a HttpResponse whose content is a Javascript file representing a
    list of links to flatpages.
    """
    from django.contrib.flatpages.models import FlatPage
    link_list = [(page.title, page.url) for page in FlatPage.objects.all()]
    return render_to_link_list(link_list)

def values(self):
        """Gets the user enter max and min values of where the 
        raster points should appear on the y-axis

        :returns: (float, float) -- (min, max) y-values to bound the raster plot by
        """
        lower = float(self.lowerSpnbx.value())
        upper = float(self.upperSpnbx.value())
        return (lower, upper)

def sqlmany(self, stringname, *args):
        """Wrapper for executing many SQL calls on my connection.

        First arg is the name of a query, either a key in the
        precompiled JSON or a method name in
        ``allegedb.alchemy.Alchemist``. Remaining arguments should be
        tuples of argument sequences to be passed to the query.

        """
        if hasattr(self, 'alchemist'):
            return getattr(self.alchemist.many, stringname)(*args)
        s = self.strings[stringname]
        return self.connection.cursor().executemany(s, args)

def convolve_gaussian_2d(image, gaussian_kernel_1d):
    """Convolve 2d gaussian."""
    result = scipy.ndimage.filters.correlate1d(
        image, gaussian_kernel_1d, axis=0)
    result = scipy.ndimage.filters.correlate1d(
        result, gaussian_kernel_1d, axis=1)
    return result

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

def asynchronous(function, event):
    """
    Runs the function asynchronously taking care of exceptions.
    """
    thread = Thread(target=synchronous, args=(function, event))
    thread.daemon = True
    thread.start()

def HttpResponse403(request, template=KEY_AUTH_403_TEMPLATE,
content=KEY_AUTH_403_CONTENT, content_type=KEY_AUTH_403_CONTENT_TYPE):
    """
    HTTP response for forbidden access (status code 403)
    """
    return AccessFailedResponse(request, template, content, content_type, status=403)

def similarity(self, other):
        """Calculates the cosine similarity between this vector and another
        vector."""
        if self.magnitude == 0 or other.magnitude == 0:
            return 0

        return self.dot(other) / self.magnitude

def default_static_path():
    """
        Return the path to the javascript bundle
    """
    fdir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(fdir, '../assets/'))

def count_list(the_list):
    """
    Generates a count of the number of times each unique item appears in a list
    """
    count = the_list.count
    result = [(item, count(item)) for item in set(the_list)]
    result.sort()
    return result

def round_to_float(number, precision):
    """Round a float to a precision"""
    rounded = Decimal(str(floor((number + precision / 2) // precision))
                      ) * Decimal(str(precision))
    return float(rounded)

def _calc_overlap_count(
    markers1: dict,
    markers2: dict,
):
    """Calculate overlap count between the values of two dictionaries

    Note: dict values must be sets
    """
    overlaps=np.zeros((len(markers1), len(markers2)))

    j=0
    for marker_group in markers1:
        tmp = [len(markers2[i].intersection(markers1[marker_group])) for i in markers2.keys()]
        overlaps[j,:] = tmp
        j += 1

    return overlaps

def intround(value):
    """Given a float returns a rounded int. Should give the same result on
    both Py2/3
    """

    return int(decimal.Decimal.from_float(
        value).to_integral_value(decimal.ROUND_HALF_EVEN))

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

def focusInEvent(self, event):
        """Reimplement Qt method to send focus change notification"""
        self.focus_changed.emit()
        return super(PageControlWidget, self).focusInEvent(event)

def mkdir(dir, enter):
    """Create directory with template for topic of the current environment

    """

    if not os.path.exists(dir):
        os.makedirs(dir)

def one_hot(x, size, dtype=np.float32):
  """Make a n+1 dim one-hot array from n dim int-categorical array."""
  return np.array(x[..., np.newaxis] == np.arange(size), dtype)

def iter_finds(regex_obj, s):
    """Generate all matches found within a string for a regex and yield each match as a string"""
    if isinstance(regex_obj, str):
        for m in re.finditer(regex_obj, s):
            yield m.group()
    else:
        for m in regex_obj.finditer(s):
            yield m.group()

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

def concat(cls, iterables):
    """
    Similar to #itertools.chain.from_iterable().
    """

    def generator():
      for it in iterables:
        for element in it:
          yield element
    return cls(generator())

def format_result(input):
        """From: http://stackoverflow.com/questions/13062300/convert-a-dict-to-sorted-dict-in-python
        """
        items = list(iteritems(input))
        return OrderedDict(sorted(items, key=lambda x: x[0]))

def bulk_query(self, query, *multiparams):
        """Bulk insert or update."""

        with self.get_connection() as conn:
            conn.bulk_query(query, *multiparams)

def Trie(S):
    """
    :param S: set of words
    :returns: trie containing all words from S
    :complexity: linear in total word sizes from S
    """
    T = None
    for w in S:
        T = add(T, w)
    return T

def __set__(self, instance, value):
        """ Set a related object for an instance. """

        self.map[id(instance)] = (weakref.ref(instance), value)

def recarray(self):
        """Returns data as :class:`numpy.recarray`."""
        return numpy.rec.fromrecords(self.records, names=self.names)

def go_to_background():
    """ Daemonize the running process. """
    try:
        if os.fork():
            sys.exit()
    except OSError as errmsg:
        LOGGER.error('Fork failed: {0}'.format(errmsg))
        sys.exit('Fork failed')

def generate_unique_host_id():
    """Generate a unique ID, that is somewhat guaranteed to be unique among all
    instances running at the same time."""
    host = ".".join(reversed(socket.gethostname().split(".")))
    pid = os.getpid()
    return "%s.%d" % (host, pid)

def compress(self, data_list):
        """
        Return the cleaned_data of the form, everything should already be valid
        """
        data = {}
        if data_list:
            return dict(
                (f.name, data_list[i]) for i, f in enumerate(self.form))
        return data

def init_db():
    """
    Drops and re-creates the SQL schema
    """
    db.drop_all()
    db.configure_mappers()
    db.create_all()
    db.session.commit()

def safe_format(s, **kwargs):
  """
  :type s str
  """
  return string.Formatter().vformat(s, (), defaultdict(str, **kwargs))

def _init_unique_sets(self):
        """Initialise sets used for uniqueness checking."""

        ks = dict()
        for t in self._unique_checks:
            key = t[0]
            ks[key] = set() # empty set
        return ks

def straight_line_show(title, length=100, linestyle="=", pad=0):
        """Print a formatted straight line.
        """
        print(StrTemplate.straight_line(
            title=title, length=length, linestyle=linestyle, pad=pad))

def make_executable(script_path):
    """Make `script_path` executable.

    :param script_path: The file to change
    """
    status = os.stat(script_path)
    os.chmod(script_path, status.st_mode | stat.S_IEXEC)

def  make_html_code( self, lines ):
        """ convert a code sequence to HTML """
        line = code_header + '\n'
        for l in lines:
            line = line + html_quote( l ) + '\n'

        return line + code_footer

def cross_product_matrix(vec):
    """Returns a 3x3 cross-product matrix from a 3-element vector."""
    return np.array([[0, -vec[2], vec[1]],
                     [vec[2], 0, -vec[0]],
                     [-vec[1], vec[0], 0]])

def index_nearest(value, array):
    """
    expects a _n.array
    returns the global minimum of (value-array)^2
    """

    a = (array-value)**2
    return index(a.min(), a)

def main(args=sys.argv):
    """
    main entry point for the jardiff CLI
    """

    parser = create_optparser(args[0])
    return cli(parser.parse_args(args[1:]))

def free(self):
        """Free the underlying C array"""
        if self._ptr is None:
            return
        Gauged.array_free(self.ptr)
        FloatArray.ALLOCATIONS -= 1
        self._ptr = None

def from_points(cls, list_of_lists):
        """
        Creates a *Polygon* instance out of a list of lists, each sublist being populated with
        `pyowm.utils.geo.Point` instances
        :param list_of_lists: list
        :type: list_of_lists: iterable_of_polygons
        :returns:  a *Polygon* instance

        """
        result = []
        for l in list_of_lists:
            curve = []
            for point in l:
                curve.append((point.lon, point.lat))
            result.append(curve)
        return Polygon(result)

def connect():
    """Connect to FTP server, login and return an ftplib.FTP instance."""
    ftp_class = ftplib.FTP if not SSL else ftplib.FTP_TLS
    ftp = ftp_class(timeout=TIMEOUT)
    ftp.connect(HOST, PORT)
    ftp.login(USER, PASSWORD)
    if SSL:
        ftp.prot_p()  # secure data connection
    return ftp

def tmpfile(prefix, direc):
    """Returns the path to a newly created temporary file."""
    return tempfile.mktemp(prefix=prefix, suffix='.pdb', dir=direc)

def connect(host, port, username, password):
        """Connect and login to an FTP server and return ftplib.FTP object."""
        # Instantiate ftplib client
        session = ftplib.FTP()

        # Connect to host without auth
        session.connect(host, port)

        # Authenticate connection
        session.login(username, password)
        return session

def unique_list(lst):
    """Make a list unique, retaining order of initial appearance."""
    uniq = []
    for item in lst:
        if item not in uniq:
            uniq.append(item)
    return uniq

def connect():
    """Connect to FTP server, login and return an ftplib.FTP instance."""
    ftp_class = ftplib.FTP if not SSL else ftplib.FTP_TLS
    ftp = ftp_class(timeout=TIMEOUT)
    ftp.connect(HOST, PORT)
    ftp.login(USER, PASSWORD)
    if SSL:
        ftp.prot_p()  # secure data connection
    return ftp

def exp_fit_fun(x, a, tau, c):
    """Function used to fit the exponential decay."""
    # pylint: disable=invalid-name
    return a * np.exp(-x / tau) + c

def All(sequence):
  """
  :param sequence: Any sequence whose elements can be evaluated as booleans.
  :returns: true if all elements of the sequence satisfy True and x.
  """
  return bool(reduce(lambda x, y: x and y, sequence, True))

def zero_state(self, batch_size):
        """ Initial state of the network """
        return torch.zeros(batch_size, self.state_dim, dtype=torch.float32)

def _fullname(o):
    """Return the fully-qualified name of a function."""
    return o.__module__ + "." + o.__name__ if o.__module__ else o.__name__

def create_index(config):
    """Create the root index."""
    filename = pathlib.Path(config.cache_path) / "index.json"
    index = {"version": __version__}
    with open(filename, "w") as out:
        out.write(json.dumps(index, indent=2))

def issorted(list_, op=operator.le):
    """
    Determines if a list is sorted

    Args:
        list_ (list):
        op (func): sorted operation (default=operator.le)

    Returns:
        bool : True if the list is sorted
    """
    return all(op(list_[ix], list_[ix + 1]) for ix in range(len(list_) - 1))

def is_valid(number):
    """determines whether the card number is valid."""
    n = str(number)
    if not n.isdigit():
        return False
    return int(n[-1]) == get_check_digit(n[:-1])

def us2mc(string):
    """Transform an underscore_case string to a mixedCase string"""
    return re.sub(r'_([a-z])', lambda m: (m.group(1).upper()), string)

def csv2yaml(in_file, out_file=None):
    """Convert a CSV SampleSheet to YAML run_info format.
    """
    if out_file is None:
        out_file = "%s.yaml" % os.path.splitext(in_file)[0]
    barcode_ids = _generate_barcode_ids(_read_input_csv(in_file))
    lanes = _organize_lanes(_read_input_csv(in_file), barcode_ids)
    with open(out_file, "w") as out_handle:
        out_handle.write(yaml.safe_dump(lanes, default_flow_style=False))
    return out_file

def get_average_length_of_string(strings):
    """Computes average length of words

    :param strings: list of words
    :return: Average length of word on list
    """
    if not strings:
        return 0

    return sum(len(word) for word in strings) / len(strings)

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

def good(txt):
    """Print, emphasized 'good', the given 'txt' message"""

    print("%s# %s%s%s" % (PR_GOOD_CC, get_time_stamp(), txt, PR_NC))
    sys.stdout.flush()

def move_to(self, ypos, xpos):
        """
            move the cursor to the given co-ordinates.  Co-ordinates are 1
            based, as listed in the status area of the terminal.
        """
        # the screen's co-ordinates are 1 based, but the command is 0 based
        xpos -= 1
        ypos -= 1
        self.exec_command("MoveCursor({0}, {1})".format(ypos, xpos).encode("ascii"))

def dict_from_object(obj: object):
    """Convert a object into dictionary with all of its readable attributes."""

    # If object is a dict instance, no need to convert.
    return (obj if isinstance(obj, dict)
            else {attr: getattr(obj, attr)
                  for attr in dir(obj) if not attr.startswith('_')})

def ensure_hbounds(self):
        """Ensure the cursor is within horizontal screen bounds."""
        self.cursor.x = min(max(0, self.cursor.x), self.columns - 1)

def strip_spaces(s):
    """ Strip excess spaces from a string """
    return u" ".join([c for c in s.split(u' ') if c])

def scatter(self, *args, **kwargs):
        """Add a scatter plot."""
        cls = _make_class(ScatterVisual,
                          _default_marker=kwargs.pop('marker', None),
                          )
        return self._add_item(cls, *args, **kwargs)

def download_file_from_bucket(self, bucket, file_path, key):
        """ Download file from S3 Bucket """
        with open(file_path, 'wb') as data:
            self.__s3.download_fileobj(bucket, key, data)
            return file_path

def imdecode(image_path):
    """Return BGR image read by opencv"""
    import os
    assert os.path.exists(image_path), image_path + ' not found'
    im = cv2.imread(image_path)
    return im

def ex(self, cmd):
        """Execute a normal python statement in user namespace."""
        with self.builtin_trap:
            exec cmd in self.user_global_ns, self.user_ns

def split(s):
  """Uses dynamic programming to infer the location of spaces in a string without spaces."""
  l = [_split(x) for x in _SPLIT_RE.split(s)]
  return [item for sublist in l for item in sublist]

def dt2jd(dt):
    """Convert datetime to julian date
    """
    a = (14 - dt.month)//12
    y = dt.year + 4800 - a
    m = dt.month + 12*a - 3
    return dt.day + ((153*m + 2)//5) + 365*y + y//4 - y//100 + y//400 - 32045

def smooth_gaussian(image, sigma=1):
    """Returns Gaussian smoothed image.

    :param image: numpy array or :class:`jicimagelib.image.Image`
    :param sigma: standard deviation
    :returns: :class:`jicimagelib.image.Image`
    """
    return scipy.ndimage.filters.gaussian_filter(image, sigma=sigma, mode="nearest")

def _time_to_json(value):
    """Coerce 'value' to an JSON-compatible representation."""
    if isinstance(value, datetime.time):
        value = value.isoformat()
    return value

def EvalGaussianPdf(x, mu, sigma):
    """Computes the unnormalized PDF of the normal distribution.

    x: value
    mu: mean
    sigma: standard deviation
    
    returns: float probability density
    """
    return scipy.stats.norm.pdf(x, mu, sigma)

def convert_timestamp(timestamp):
    """
    Converts bokehJS timestamp to datetime64.
    """
    datetime = dt.datetime.utcfromtimestamp(timestamp/1000.)
    return np.datetime64(datetime.replace(tzinfo=None))

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

def accuracy(self):
        """
        Calculates the accuracy of the tree by comparing
        the model predictions to the dataset
        (TP + TN) / (TP + TN + FP + FN) == (T / (T + F))
        """
        sub_observed = np.array([self.observed.metadata[i] for i in self.observed.arr])
        return float((self.model_predictions() == sub_observed).sum()) / self.data_size

def cli(yamlfile, format, context):
    """ Generate JSONLD file from biolink schema """
    print(JSONLDGenerator(yamlfile, format).serialize(context=context))

def double_sha256(data):
    """A standard compound hash."""
    return bytes_as_revhex(hashlib.sha256(hashlib.sha256(data).digest()).digest())

def get_cantons(self):
        """
        Return the list of unique cantons, sorted by name.
        """
        return sorted(list(set([
            location.canton for location in self.get_locations().values()
        ])))

def get_default_args(func):
    """
    returns a dictionary of arg_name:default_values for the input function
    """
    args, varargs, keywords, defaults = getargspec_no_self(func)
    return dict(zip(args[-len(defaults):], defaults))

def get_method_name(method):
    """
    Returns given method name.

    :param method: Method to retrieve the name.
    :type method: object
    :return: Method name.
    :rtype: unicode
    """

    name = get_object_name(method)
    if name.startswith("__") and not name.endswith("__"):
        name = "_{0}{1}".format(get_object_name(method.im_class), name)
    return name

def _add_default_arguments(parser):
    """Add the default arguments to the parser.

    :param argparse.ArgumentParser parser: The argument parser

    """
    parser.add_argument('-c', '--config', action='store', dest='config',
                        help='Path to the configuration file')
    parser.add_argument('-f', '--foreground', action='store_true', dest='foreground',
                        help='Run the application interactively')

def get_methods(*objs):
    """ Return the names of all callable attributes of an object"""
    return set(
        attr
        for obj in objs
        for attr in dir(obj)
        if not attr.startswith('_') and callable(getattr(obj, attr))
    )

def add_blank_row(self, label):
        """
        Add a blank row with only an index value to self.df.
        This is done inplace.
        """
        col_labels = self.df.columns
        blank_item = pd.Series({}, index=col_labels, name=label)
        # use .loc to add in place (append won't do that)
        self.df.loc[blank_item.name] = blank_item
        return self.df

def items(self, section_name):
        """:return: list((option, value), ...) pairs of all items in the given section"""
        return [(k, v) for k, v in super(GitConfigParser, self).items(section_name) if k != '__name__']

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

def get_keys_from_class(cc):
    """Return list of the key property names for a class """
    return [prop.name for prop in cc.properties.values() \
            if 'key' in prop.qualifiers]

def rm(venv_name):
    """ Removes the venv by name """
    inenv = InenvManager()
    venv = inenv.get_venv(venv_name)
    click.confirm("Delete dir {}".format(venv.path))
    shutil.rmtree(venv.path)

def columns(self):
        """Return names of all the addressable columns (including foreign keys) referenced in user supplied model"""
        res = [col['name'] for col in self.column_definitions]
        res.extend([col['name'] for col in self.foreign_key_definitions])
        return res

def remove_non_magic_cols(self):
        """
        Remove all non-MagIC columns from all tables.
        """
        for table_name in self.tables:
            table = self.tables[table_name]
            table.remove_non_magic_cols_from_table()

def get_obj(ref):
    """Get object from string reference."""
    oid = int(ref)
    return server.id2ref.get(oid) or server.id2obj[oid]

def _split_comma_separated(string):
    """Return a set of strings."""
    return set(text.strip() for text in string.split(',') if text.strip())

def angle(x0, y0, x1, y1):
    """ Returns the angle between two points.
    """
    return degrees(atan2(y1-y0, x1-x0))

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

def reset():
    """Delete the session and clear temporary directories

    """
    shutil.rmtree(session['img_input_dir'])
    shutil.rmtree(session['img_output_dir'])
    session.clear()
    return jsonify(ok='true')

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

def get_colors(img):
    """
    Returns a list of all the image's colors.
    """
    w, h = img.size
    return [color[:3] for count, color in img.convert('RGB').getcolors(w * h)]

def pop(h):
    """Pop the heap value from the heap."""
    n = h.size() - 1
    h.swap(0, n)
    down(h, 0, n)
    return h.pop()

def memory():
    """Determine memory specifications of the machine.

    Returns
    -------
    mem_info : dictonary
        Holds the current values for the total, free and used memory of the system.
    """

    mem_info = dict()

    for k, v in psutil.virtual_memory()._asdict().items():
           mem_info[k] = int(v)
           
    return mem_info

def check_precomputed_distance_matrix(X):
    """Perform check_array(X) after removing infinite values (numpy.inf) from the given distance matrix.
    """
    tmp = X.copy()
    tmp[np.isinf(tmp)] = 1
    check_array(tmp)

def calculate_month(birth_date):
    """
    Calculates and returns a month number basing on PESEL standard.
    """
    year = int(birth_date.strftime('%Y'))
    month = int(birth_date.strftime('%m')) + ((int(year / 100) - 14) % 5) * 20

    return month

def linedelimited (inlist,delimiter):
    """
Returns a string composed of elements in inlist, with each element
separated by 'delimiter.'  Used by function writedelimited.  Use '\t'
for tab-delimiting.

Usage:   linedelimited (inlist,delimiter)
"""
    outstr = ''
    for item in inlist:
        if type(item) != StringType:
            item = str(item)
        outstr = outstr + item + delimiter
    outstr = outstr[0:-1]
    return outstr

def get_month_start_end_day():
    """
    Get the month start date a nd end date
    """
    t = date.today()
    n = mdays[t.month]
    return (date(t.year, t.month, 1), date(t.year, t.month, n))

def dequeue(self, block=True):
        """Dequeue a record and return item."""
        return self.queue.get(block, self.queue_get_timeout)

def return_value(self, *args, **kwargs):
        """Extracts the real value to be returned from the wrapping callable.

        :return: The value the double should return when called.
        """

        self._called()
        return self._return_value(*args, **kwargs)

def get_best_encoding(stream):
    """Returns the default stream encoding if not found."""
    rv = getattr(stream, 'encoding', None) or sys.getdefaultencoding()
    if is_ascii_encoding(rv):
        return 'utf-8'
    return rv

def relpath(self):
        """ Return this path as a relative path,
        based from the current working directory.
        """
        cwd = self.__class__(os.getcwdu())
        return cwd.relpathto(self)

def we_are_in_lyon():
    """Check if we are on a Lyon machine"""
    import socket
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        return False
    return ip.startswith("134.158.")

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

def is_callable(*p):
    """ True if all the args are functions and / or subroutines
    """
    import symbols
    return all(isinstance(x, symbols.FUNCTION) for x in p)

def skip_connection_distance(a, b):
    """The distance between two skip-connections."""
    if a[2] != b[2]:
        return 1.0
    len_a = abs(a[1] - a[0])
    len_b = abs(b[1] - b[0])
    return (abs(a[0] - b[0]) + abs(len_a - len_b)) / (max(a[0], b[0]) + max(len_a, len_b))

def eqstr(a, b):
    """
    Determine whether two strings are equivalent.

    http://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/eqstr_c.html

    :param a: Arbitrary character string.
    :type a: str
    :param b: Arbitrary character string.
    :type b: str
    :return: True if A and B are equivalent.
    :rtype: bool
    """
    return bool(libspice.eqstr_c(stypes.stringToCharP(a), stypes.stringToCharP(b)))

def get_by(self, name):
    """get element by name"""
    return next((item for item in self if item.name == name), None)

def validate(self, *args, **kwargs): # pylint: disable=arguments-differ
        """
        Validate a parameter dict against a parameter schema from an ocrd-tool.json

        Args:
            obj (dict):
            schema (dict):
        """
        return super(ParameterValidator, self)._validate(*args, **kwargs)

def get_parent_dir(name):
    """Get the parent directory of a filename."""
    parent_dir = os.path.dirname(os.path.dirname(name))
    if parent_dir:
        return parent_dir
    return os.path.abspath('.')

def me(self):
        """Similar to :attr:`.Guild.me` except it may return the :class:`.ClientUser` in private message contexts."""
        return self.guild.me if self.guild is not None else self.bot.user

def get_size_in_bytes(self, handle):
        """Return the size in bytes."""
        fpath = self._fpath_from_handle(handle)
        return os.stat(fpath).st_size

def show_guestbook():
    """Returns all existing guestbook records."""
    cursor = flask.g.db.execute(
        'SELECT name, message FROM entry ORDER BY id DESC;')
    entries = [{'name': row[0], 'message': row[1]} for row in cursor.fetchall()]
    return jinja2.Template(LAYOUT).render(entries=entries)

def get_month_start(day=None):
    """Returns the first day of the given month."""
    day = add_timezone(day or datetime.date.today())
    return day.replace(day=1)

def rank(idx, dim):
    """Calculate the index rank according to Bertran's notation."""
    idxm = multi_index(idx, dim)
    out = 0
    while idxm[-1:] == (0,):
        out += 1
        idxm = idxm[:-1]
    return out

def get_last_commit(git_path=None):
    """
    Get the HEAD commit SHA1 of repository in current dir.
    """
    if git_path is None: git_path = GIT_PATH
    line = get_last_commit_line(git_path)
    revision_id = line.split()[1]
    return revision_id

def csvpretty(csvfile: csvfile=sys.stdin):
    """ Pretty print a CSV file. """
    shellish.tabulate(csv.reader(csvfile))

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

def _split_str(s, n):
    """
    split string into list of strings by specified number.
    """
    length = len(s)
    return [s[i:i + n] for i in range(0, length, n)]

def qsize(self):
        """Return the approximate size of the queue (not reliable!)."""
        self.mutex.acquire()
        n = self._qsize()
        self.mutex.release()
        return n

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

def items(self):
    """Return a list of the (name, value) pairs of the enum.

    These are returned in the order they were defined in the .proto file.
    """
    return [(value_descriptor.name, value_descriptor.number)
            for value_descriptor in self._enum_type.values]

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

def go_to_new_line(self):
        """Go to the end of the current line and create a new line"""
        self.stdkey_end(False, False)
        self.insert_text(self.get_line_separator())

def get_font_list():
    """Returns a sorted list of all system font names"""

    font_map = pangocairo.cairo_font_map_get_default()
    font_list = [f.get_name() for f in font_map.list_families()]
    font_list.sort()

    return font_list

def has_parent(self, term):
        """Return True if this GO object has a parent GO ID."""
        for parent in self.parents:
            if parent.item_id == term or parent.has_parent(term):
                return True
        return False

def unique_list_dicts(dlist, key):
    """Return a list of dictionaries which are sorted for only unique entries.

    :param dlist:
    :param key:
    :return list:
    """

    return list(dict((val[key], val) for val in dlist).values())

def invalidate_cache(cpu, address, size):
        """ remove decoded instruction from instruction cache """
        cache = cpu.instruction_cache
        for offset in range(size):
            if address + offset in cache:
                del cache[address + offset]

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

def keys_to_snake_case(camel_case_dict):
    """
    Make a copy of a dictionary with all keys converted to snake case. This is just calls to_snake_case on
    each of the keys in the dictionary and returns a new dictionary.

    :param camel_case_dict: Dictionary with the keys to convert.
    :type camel_case_dict: Dictionary.

    :return: Dictionary with the keys converted to snake case.
    """
    return dict((to_snake_case(key), value) for (key, value) in camel_case_dict.items())

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

def timer():
    """
    Timer used for calculate time elapsed
    """
    if sys.platform == "win32":
        default_timer = time.clock
    else:
        default_timer = time.time

    return default_timer()

def last_day(year=_year, month=_month):
    """
    get the current month's last day
    :param year:  default to current year
    :param month:  default to current month
    :return: month's last day
    """
    last_day = calendar.monthrange(year, month)[1]
    return datetime.date(year=year, month=month, day=last_day)

def unit_tangent(self, t):
        """returns the unit tangent vector of the segment at t (centered at
        the origin and expressed as a complex number)."""
        dseg = self.derivative(t)
        return dseg/abs(dseg)

def get_obj_cols(df):
    """
    Returns names of 'object' columns in the DataFrame.
    """
    obj_cols = []
    for idx, dt in enumerate(df.dtypes):
        if dt == 'object' or is_category(dt):
            obj_cols.append(df.columns.values[idx])

    return obj_cols

def match_aspect_to_viewport(self):
        """Updates Camera.aspect to match the viewport's aspect ratio."""
        viewport = self.viewport
        self.aspect = float(viewport.width) / viewport.height

def get_property_by_name(pif, name):
    """Get a property by name"""
    return next((x for x in pif.properties if x.name == name), None)

def us2mc(string):
    """Transform an underscore_case string to a mixedCase string"""
    return re.sub(r'_([a-z])', lambda m: (m.group(1).upper()), string)

def _uniquify(_list):
    """Remove duplicates in a list."""
    seen = set()
    result = []
    for x in _list:
        if x not in seen:
            result.append(x)
            seen.add(x)
    return result

def fmt_duration(secs):
    """Format a duration in seconds."""
    return ' '.join(fmt.human_duration(secs, 0, precision=2, short=True).strip().split())

def get_module_path(modname):
    """Return module *modname* base path"""
    return osp.abspath(osp.dirname(sys.modules[modname].__file__))

def np_hash(a):
    """Return a hash of a NumPy array."""
    if a is None:
        return hash(None)
    # Ensure that hashes are equal whatever the ordering in memory (C or
    # Fortran)
    a = np.ascontiguousarray(a)
    # Compute the digest and return a decimal int
    return int(hashlib.sha1(a.view(a.dtype)).hexdigest(), 16)

def get(s, delimiter='', format="diacritical"):
    """Return pinyin of string, the string must be unicode
    """
    return delimiter.join(_pinyin_generator(u(s), format=format))

def center_eigenvalue_diff(mat):
    """Compute the eigvals of mat and then find the center eigval difference."""
    N = len(mat)
    evals = np.sort(la.eigvals(mat))
    diff = np.abs(evals[N/2] - evals[N/2-1])
    return diff

def get_property_by_name(pif, name):
    """Get a property by name"""
    return next((x for x in pif.properties if x.name == name), None)

def center_eigenvalue_diff(mat):
    """Compute the eigvals of mat and then find the center eigval difference."""
    N = len(mat)
    evals = np.sort(la.eigvals(mat))
    diff = np.abs(evals[N/2] - evals[N/2-1])
    return diff

def get_file_size(fileobj):
    """
    Returns the size of a file-like object.
    """
    currpos = fileobj.tell()
    fileobj.seek(0, 2)
    total_size = fileobj.tell()
    fileobj.seek(currpos)
    return total_size

def center_eigenvalue_diff(mat):
    """Compute the eigvals of mat and then find the center eigval difference."""
    N = len(mat)
    evals = np.sort(la.eigvals(mat))
    diff = np.abs(evals[N/2] - evals[N/2-1])
    return diff

def array_bytes(array):
    """ Estimates the memory of the supplied array in bytes """
    return np.product(array.shape)*np.dtype(array.dtype).itemsize

def clear_es():
        """Clear all indexes in the es core"""
        # TODO: should receive a catalog slug.
        ESHypermap.es.indices.delete(ESHypermap.index_name, ignore=[400, 404])
        LOGGER.debug('Elasticsearch: Index cleared')

def get_idx_rect(index_list):
    """Extract the boundaries from a list of indexes"""
    rows, cols = list(zip(*[(i.row(), i.column()) for i in index_list]))
    return ( min(rows), max(rows), min(cols), max(cols) )

def _get_node_parent(self, age, pos):
        """Get the parent node of node, whch is located in tree's node list.

        Returns:
            object: The parent node.
        """
        return self.nodes[age][int(pos / self.comp)]

def __repr__(self):
        """Return list-lookalike of representation string of objects"""
        strings = []
        for currItem in self:
            strings.append("%s" % currItem)
        return "(%s)" % (", ".join(strings))

def dedup_list(l):
    """Given a list (l) will removing duplicates from the list,
       preserving the original order of the list. Assumes that
       the list entrie are hashable."""
    dedup = set()
    return [ x for x in l if not (x in dedup or dedup.add(x))]

def tf2():
  """Provide the root module of a TF-2.0 API for use within TensorBoard.

  Returns:
    The root module of a TF-2.0 API, if available.

  Raises:
    ImportError: if a TF-2.0 API is not available.
  """
  # Import the `tf` compat API from this file and check if it's already TF 2.0.
  if tf.__version__.startswith('2.'):
    return tf
  elif hasattr(tf, 'compat') and hasattr(tf.compat, 'v2'):
    # As a fallback, try `tensorflow.compat.v2` if it's defined.
    return tf.compat.v2
  raise ImportError('cannot import tensorflow 2.0 API')

def split_addresses(email_string_list):
    """
    Converts a string containing comma separated email addresses
    into a list of email addresses.
    """
    return [f for f in [s.strip() for s in email_string_list.split(",")] if f]

def size():
    """Determines the height and width of the console window

        Returns:
            tuple of int: The height in lines, then width in characters
    """
    try:
        assert os != 'nt' and sys.stdout.isatty()
        rows, columns = os.popen('stty size', 'r').read().split()
    except (AssertionError, AttributeError, ValueError):
        # in case of failure, use dimensions of a full screen 13" laptop
        rows, columns = DEFAULT_HEIGHT, DEFAULT_WIDTH

    return int(rows), int(columns)

def _encode_bool(name, value, dummy0, dummy1):
    """Encode a python boolean (True/False)."""
    return b"\x08" + name + (value and b"\x01" or b"\x00")

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

def write_enum(fo, datum, schema):
    """An enum is encoded by a int, representing the zero-based position of
    the symbol in the schema."""
    index = schema['symbols'].index(datum)
    write_int(fo, index)

def get_bottomrect_idx(self, pos):
        """ Determine if cursor is on bottom right corner of a hot spot."""
        for i, r in enumerate(self.link_bottom_rects):
            if r.Contains(pos):
                return i
        return -1

def _dt_to_epoch(dt):
        """Convert datetime to epoch seconds."""
        try:
            epoch = dt.timestamp()
        except AttributeError:  # py2
            epoch = (dt - datetime(1970, 1, 1)).total_seconds()
        return epoch

def table_top_abs(self):
        """Returns the absolute position of table top"""
        table_height = np.array([0, 0, self.table_full_size[2]])
        return string_to_array(self.floor.get("pos")) + table_height

def plot_epsilon_residuals(self):
        """Plots the epsilon residuals for the variogram fit."""
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(range(self.epsilon.size), self.epsilon, c='k', marker='*')
        ax.axhline(y=0.0)
        plt.show()

def get_property_by_name(pif, name):
    """Get a property by name"""
    return next((x for x in pif.properties if x.name == name), None)

def image_set_aspect(aspect=1.0, axes="gca"):
    """
    sets the aspect ratio of the current zoom level of the imshow image
    """
    if axes is "gca": axes = _pylab.gca()

    e = axes.get_images()[0].get_extent()
    axes.set_aspect(abs((e[1]-e[0])/(e[3]-e[2]))/aspect)

def forceupdate(self, *args, **kw):
        """Like a bulk :meth:`forceput`."""
        self._update(False, self._ON_DUP_OVERWRITE, *args, **kw)

def get_nt_system_uid():
    """Get the MachineGuid from
    HKEY_LOCAL_MACHINE\Software\Microsoft\Cryptography\MachineGuid
    """
    try:
        import _winreg as winreg
    except ImportError:
        import winreg
    lm = winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE)
    try:
        key = winreg.OpenKey(lm, r"Software\Microsoft\Cryptography")
        try:
            return winreg.QueryValueEx(key, "MachineGuid")[0]
        finally:
            key.Close()
    finally:
        lm.Close()

def _escape(s):
    """ Helper method that escapes parameters to a SQL query. """
    e = s
    e = e.replace('\\', '\\\\')
    e = e.replace('\n', '\\n')
    e = e.replace('\r', '\\r')
    e = e.replace("'", "\\'")
    e = e.replace('"', '\\"')
    return e

def get_element_with_id(self, id):
        """Return the element with the specified ID."""
        # Should we maintain a hashmap of ids to make this more efficient? Probably overkill.
        # TODO: Elements can contain nested elements (captions, footnotes, table cells, etc.)
        return next((el for el in self.elements if el.id == id), None)

def vector_distance(a, b):
    """The Euclidean distance between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

def url(self):
        """ The url of this window """
        with switch_window(self._browser, self.name):
            return self._browser.url

def euclidean(c1, c2):
    """Square of the euclidean distance"""
    diffs = ((i - j) for i, j in zip(c1, c2))
    return sum(x * x for x in diffs)

def get_free_memory_win():
    """Return current free memory on the machine for windows.

    Warning : this script is really not robust
    Return in MB unit
    """
    stat = MEMORYSTATUSEX()
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return int(stat.ullAvailPhys / 1024 / 1024)

def xpathEvalExpression(self, str):
        """Evaluate the XPath expression in the given context. """
        ret = libxml2mod.xmlXPathEvalExpression(str, self._o)
        if ret is None:raise xpathError('xmlXPathEvalExpression() failed')
        return xpathObjectRet(ret)

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

def is_in(self, point_x, point_y):
        """ Test if a point is within this polygonal region """

        point_array = array(((point_x, point_y),))
        vertices = array(self.points)
        winding = self.inside_rule == "winding"
        result = points_in_polygon(point_array, vertices, winding)
        return result[0]

def extent_count(self):
        """
        Returns the volume group extent count.
        """
        self.open()
        count = lvm_vg_get_extent_count(self.handle)
        self.close()
        return count

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

def title(self):
        """ The title of this window """
        with switch_window(self._browser, self.name):
            return self._browser.title

def visit_BoolOp(self, node):
        """ Return type may come from any boolop operand. """
        return sum((self.visit(value) for value in node.values), [])

def __get_xml_text(root):
    """ Return the text for the given root node (xml.dom.minidom). """
    txt = ""
    for e in root.childNodes:
        if (e.nodeType == e.TEXT_NODE):
            txt += e.data
    return txt

def runcode(code):
	"""Run the given code line by line with printing, as list of lines, and return variable 'ans'."""
	for line in code:
		print('# '+line)
		exec(line,globals())
	print('# return ans')
	return ans

def fetch_event(urls):
    """
    This parallel fetcher uses gevent one uses gevent
    """
    rs = (grequests.get(u) for u in urls)
    return [content.json() for content in grequests.map(rs)]

def get_order(self, codes):
        """Return evidence codes in order shown in code2name."""
        return sorted(codes, key=lambda e: [self.ev2idx.get(e)])

def equal(list1, list2):
    """ takes flags returns indexes of True values """
    return [item1 == item2 for item1, item2 in broadcast_zip(list1, list2)]

def select(self, cmd, *args, **kwargs):
        """ Execute the SQL command and return the data rows as tuples
        """
        self.cursor.execute(cmd, *args, **kwargs)
        return self.cursor.fetchall()

def go_to_parent_directory(self):
        """Go to parent directory"""
        self.chdir(osp.abspath(osp.join(getcwd_or_home(), os.pardir)))

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

def _top(self):
        """ g """
        # Goto top of the list
        self.top.body.focus_position = 2 if self.compact is False else 0
        self.top.keypress(self.size, "")

def exp_fit_fun(x, a, tau, c):
    """Function used to fit the exponential decay."""
    # pylint: disable=invalid-name
    return a * np.exp(-x / tau) + c

def to_gtp(coord):
    """Converts from a Minigo coordinate to a GTP coordinate."""
    if coord is None:
        return 'pass'
    y, x = coord
    return '{}{}'.format(_GTP_COLUMNS[x], go.N - y)

def nb_to_python(nb_path):
    """convert notebook to python script"""
    exporter = python.PythonExporter()
    output, resources = exporter.from_filename(nb_path)
    return output

def searchlast(self,n=10):
        """Return the last n results (or possibly less if not found). Note that the last results are not necessarily the best ones! Depending on the search type."""            
        solutions = deque([], n)
        for solution in self:
            solutions.append(solution)
        return solutions

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

def _text_to_graphiz(self, text):
        """create a graphviz graph from text"""
        dot = Source(text, format='svg')
        return dot.pipe().decode('utf-8')

def get_colors(img):
    """
    Returns a list of all the image's colors.
    """
    w, h = img.size
    return [color[:3] for count, color in img.convert('RGB').getcolors(w * h)]

def _round_half_hour(record):
    """
    Round a time DOWN to half nearest half-hour.
    """
    k = record.datetime + timedelta(minutes=-(record.datetime.minute % 30))
    return datetime(k.year, k.month, k.day, k.hour, k.minute, 0)

def get_X0(X):
    """ Return zero-th element of a one-element data container.
    """
    if pandas_available and isinstance(X, pd.DataFrame):
        assert len(X) == 1
        x = np.array(X.iloc[0])
    else:
        x, = X
    return x

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

def security(self):
        """Print security object information for a pdf document"""
        return {k: v for i in self.pdf.resolvedObjects.items() for k, v in i[1].items()}

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

def dot(a, b):
    """Take arrays `a` and `b` and form the dot product between the last axis
    of `a` and the first of `b`.
    """
    b = numpy.asarray(b)
    return numpy.dot(a, b.reshape(b.shape[0], -1)).reshape(a.shape[:-1] + b.shape[1:])

def guess_encoding(text, default=DEFAULT_ENCODING):
    """Guess string encoding.

    Given a piece of text, apply character encoding detection to
    guess the appropriate encoding of the text.
    """
    result = chardet.detect(text)
    return normalize_result(result, default=default)

def check_precomputed_distance_matrix(X):
    """Perform check_array(X) after removing infinite values (numpy.inf) from the given distance matrix.
    """
    tmp = X.copy()
    tmp[np.isinf(tmp)] = 1
    check_array(tmp)

def __gzip(filename):
		""" Compress a file returning the new filename (.gz)
		"""
		zipname = filename + '.gz'
		file_pointer = open(filename,'rb')
		zip_pointer = gzip.open(zipname,'wb')
		zip_pointer.writelines(file_pointer)
		file_pointer.close()
		zip_pointer.close()
		return zipname

def forceupdate(self, *args, **kw):
        """Like a bulk :meth:`forceput`."""
        self._update(False, self._ON_DUP_OVERWRITE, *args, **kw)

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

def md5_hash_file(fh):
    """Return the md5 hash of the given file-object"""
    md5 = hashlib.md5()
    while True:
        data = fh.read(8192)
        if not data:
            break
        md5.update(data)
    return md5.hexdigest()

def software_fibonacci(n):
    """ a normal old python function to return the Nth fibonacci number. """
    a, b = 0, 1
    for i in range(n):
        a, b = b, a + b
    return a

def h5ToDict(h5, readH5pyDataset=True):
    """ Read a hdf5 file into a dictionary """
    h = h5py.File(h5, "r")
    ret = unwrapArray(h, recursive=True, readH5pyDataset=readH5pyDataset)
    if readH5pyDataset: h.close()
    return ret

def current_zipfile():
    """A function to vend the current zipfile, if any"""
    if zipfile.is_zipfile(sys.argv[0]):
        fd = open(sys.argv[0], "rb")
        return zipfile.ZipFile(fd)

def __unixify(self, s):
        """ stupid windows. converts the backslash to forwardslash for consistency """
        return os.path.normpath(s).replace(os.sep, "/")

def __init__(self, encoding='utf-8'):
    """Initializes an stdin input reader.

    Args:
      encoding (Optional[str]): input encoding.
    """
    super(StdinInputReader, self).__init__(sys.stdin, encoding=encoding)

def _add_hash(source):
    """Add a leading hash '#' at the beginning of every line in the source."""
    source = '\n'.join('# ' + line.rstrip()
                       for line in source.splitlines())
    return source

def apply(f, obj, *args, **kwargs):
    """Apply a function in parallel to each element of the input"""
    return vectorize(f)(obj, *args, **kwargs)

def double_sha256(data):
    """A standard compound hash."""
    return bytes_as_revhex(hashlib.sha256(hashlib.sha256(data).digest()).digest())

def drop_empty(rows):
    """Transpose the columns into rows, remove all of the rows that are empty after the first cell, then
    transpose back. The result is that columns that have a header but no data in the body are removed, assuming
    the header is the first row. """
    return zip(*[col for col in zip(*rows) if bool(filter(bool, col[1:]))])

def heappush_max(heap, item):
    """Push item onto heap, maintaining the heap invariant."""
    heap.append(item)
    _siftdown_max(heap, 0, len(heap) - 1)

def split_addresses(email_string_list):
    """
    Converts a string containing comma separated email addresses
    into a list of email addresses.
    """
    return [f for f in [s.strip() for s in email_string_list.split(",")] if f]

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

def _heapify_max(x):
    """Transform list into a maxheap, in-place, in O(len(x)) time."""
    n = len(x)
    for i in reversed(range(n//2)):
        _siftup_max(x, i)

def uniq(seq):
    """ Return a copy of seq without duplicates. """
    seen = set()
    return [x for x in seq if str(x) not in seen and not seen.add(str(x))]

def pop(h):
    """Pop the heap value from the heap."""
    n = h.size() - 1
    h.swap(0, n)
    down(h, 0, n)
    return h.pop()

def replace_all(filepath, searchExp, replaceExp):
    """
    Replace all the ocurrences (in a file) of a string with another value.
    """
    for line in fileinput.input(filepath, inplace=1):
        if searchExp in line:
            line = line.replace(searchExp, replaceExp)
        sys.stdout.write(line)

def __call__(self, kind: Optional[str] = None, **kwargs):
        """Use the plotter as callable."""
        return plot(self.histogram, kind=kind, **kwargs)

def ci(a, which=95, axis=None):
    """Return a percentile range from an array of values."""
    p = 50 - which / 2, 50 + which / 2
    return percentiles(a, p, axis)

def dtype(self):
        """Pixel data type."""
        try:
            return self.data.dtype
        except AttributeError:
            return numpy.dtype('%s%d' % (self._sample_type, self._sample_bytes))

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

def area(x,y):
    """
    Calculate the area of a polygon given as x(...),y(...)
    Implementation of Shoelace formula
    """
    # http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def _getSuperFunc(self, s, func):
        """Return the the super function."""

        return getattr(super(self.cls(), s), func.__name__)

def val_to_bin(edges, x):
    """Convert axis coordinate to bin index."""
    ibin = np.digitize(np.array(x, ndmin=1), edges) - 1
    return ibin

def val_to_bin(edges, x):
    """Convert axis coordinate to bin index."""
    ibin = np.digitize(np.array(x, ndmin=1), edges) - 1
    return ibin

def compare(dicts):
    """Compare by iteration"""

    common_members = {}
    common_keys = reduce(lambda x, y: x & y, map(dict.keys, dicts))
    for k in common_keys:
        common_members[k] = list(
            reduce(lambda x, y: x & y, [set(d[k]) for d in dicts]))

    return common_members

def convert_timestamp(timestamp):
    """
    Converts bokehJS timestamp to datetime64.
    """
    datetime = dt.datetime.utcfromtimestamp(timestamp/1000.)
    return np.datetime64(datetime.replace(tzinfo=None))

def _get_compiled_ext():
    """Official way to get the extension of compiled files (.pyc or .pyo)"""
    for ext, mode, typ in imp.get_suffixes():
        if typ == imp.PY_COMPILED:
            return ext

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

def inh(table):
    """
    inverse hyperbolic sine transformation
    """
    t = []
    for i in table:
        t.append(np.ndarray.tolist(np.arcsinh(i)))
    return t

def to_camel_case(text):
    """Convert to camel case.

    :param str text:
    :rtype: str
    :return:
    """
    split = text.split('_')
    return split[0] + "".join(x.title() for x in split[1:])

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

def _IsRetryable(error):
  """Returns whether error is likely to be retryable."""
  if not isinstance(error, MySQLdb.OperationalError):
    return False
  if not error.args:
    return False
  code = error.args[0]
  return code in _RETRYABLE_ERRORS

def iter_finds(regex_obj, s):
    """Generate all matches found within a string for a regex and yield each match as a string"""
    if isinstance(regex_obj, str):
        for m in re.finditer(regex_obj, s):
            yield m.group()
    else:
        for m in regex_obj.finditer(s):
            yield m.group()

def pid_exists(pid):
    """ Determines if a system process identifer exists in process table.
        """
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    else:
        return True

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

def _isstring(dtype):
    """Given a numpy dtype, determines whether it is a string. Returns True
    if the dtype is string or unicode.
    """
    return dtype.type == numpy.unicode_ or dtype.type == numpy.string_

def find_lt(a, x):
    """Find rightmost value less than x."""
    i = bs.bisect_left(a, x)
    if i: return i - 1
    raise ValueError

def is_in(self, point_x, point_y):
        """ Test if a point is within this polygonal region """

        point_array = array(((point_x, point_y),))
        vertices = array(self.points)
        winding = self.inside_rule == "winding"
        result = points_in_polygon(point_array, vertices, winding)
        return result[0]

def should_skip_logging(func):
    """
    Should we skip logging for this handler?

    """
    disabled = strtobool(request.headers.get("x-request-nolog", "false"))
    return disabled or getattr(func, SKIP_LOGGING, False)

def calc_cR(Q2, sigma):
    """Returns the cR statistic for the variogram fit (see [1])."""
    return Q2 * np.exp(np.sum(np.log(sigma**2))/sigma.shape[0])

def is_builtin_type(tp):
    """Checks if the given type is a builtin one.
    """
    return hasattr(__builtins__, tp.__name__) and tp is getattr(__builtins__, tp.__name__)

def apply_fit(xy,coeffs):
    """ Apply the coefficients from a linear fit to
        an array of x,y positions.

        The coeffs come from the 'coeffs' member of the
        'fit_arrays()' output.
    """
    x_new = coeffs[0][2] + coeffs[0][0]*xy[:,0] + coeffs[0][1]*xy[:,1]
    y_new = coeffs[1][2] + coeffs[1][0]*xy[:,0] + coeffs[1][1]*xy[:,1]

    return x_new,y_new

def forget_coords(self):
        """Forget all loaded coordinates."""
        self.w.ntotal.set_text('0')
        self.coords_dict.clear()
        self.redo()

def flat_list(lst):
    """This function flatten given nested list.
    Argument:
        nested list
    Returns:
        flat list
    """
    if isinstance(lst, list):
        for item in lst:
            for i in flat_list(item):
                yield i
    else:
        yield lst

def safe_exit(output):
    """exit without breaking pipes."""
    try:
        sys.stdout.write(output)
        sys.stdout.flush()
    except IOError:
        pass

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

def get_order(self):
        """
        Return a list of dicionaries. See `set_order`.
        """
        return [dict(reverse=r[0], key=r[1]) for r in self.get_model()]

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

def mkdir(dir, enter):
    """Create directory with template for topic of the current environment

    """

    if not os.path.exists(dir):
        os.makedirs(dir)

def get_document_frequency(self, term):
        """
        Returns the number of documents the specified term appears in.
        """
        if term not in self._terms:
            raise IndexError(TERM_DOES_NOT_EXIST)
        else:
            return len(self._terms[term])

def destroy(self):
        """ Cleanup the activty lifecycle listener """
        if self.widget:
            self.set_active(False)
        super(AndroidBarcodeView, self).destroy()

def connect():
    """Connect to FTP server, login and return an ftplib.FTP instance."""
    ftp_class = ftplib.FTP if not SSL else ftplib.FTP_TLS
    ftp = ftp_class(timeout=TIMEOUT)
    ftp.connect(HOST, PORT)
    ftp.login(USER, PASSWORD)
    if SSL:
        ftp.prot_p()  # secure data connection
    return ftp

def one_for_all(self, deps):
        """Because there are dependencies that depend on other
        dependencies are created lists into other lists.
        Thus creating this loop create one-dimensional list and
        remove double packages from dependencies.
        """
        requires, dependencies = [], []
        deps.reverse()
        # Inverting the list brings the
        # dependencies in order to be installed.
        requires = Utils().dimensional_list(deps)
        dependencies = Utils().remove_dbs(requires)
        return dependencies

def trigger_fullscreen_action(self, fullscreen):
        """
        Toggle fullscreen from outside the GUI,
        causes the GUI to updated and run all its actions.
        """
        action = self.action_group.get_action('fullscreen')
        action.set_active(fullscreen)

def download_json(local_filename, url, clobber=False):
    """Download the given JSON file, and pretty-print before we output it."""
    with open(local_filename, 'w') as json_file:
        json_file.write(json.dumps(requests.get(url).json(), sort_keys=True, indent=2, separators=(',', ': ')))

def is_password_valid(password):
    """
    Check if a password is valid
    """
    pattern = re.compile(r"^.{4,75}$")
    return bool(pattern.match(password))

def drag_and_drop(self, droppable):
        """
        Performs drag a element to another elmenet.

        Currently works only on Chrome driver.
        """
        self.scroll_to()
        ActionChains(self.parent.driver).drag_and_drop(self._element, droppable._element).perform()

def gauss_pdf(x, mu, sigma):
    """Normalized Gaussian"""
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - mu) ** 2 / 2. / sigma ** 2)

def url_encode(url):
    """
    Convert special characters using %xx escape.

    :param url: str
    :return: str - encoded url
    """
    if isinstance(url, text_type):
        url = url.encode('utf8')
    return quote(url, ':/%?&=')

def gauss_pdf(x, mu, sigma):
    """Normalized Gaussian"""
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - mu) ** 2 / 2. / sigma ** 2)

def go_to_new_line(self):
        """Go to the end of the current line and create a new line"""
        self.stdkey_end(False, False)
        self.insert_text(self.get_line_separator())

def ExecuteRaw(self, position, command):
    """Send a command string to gdb."""
    self.EnsureGdbPosition(position[0], None, None)
    return gdb.execute(command, to_string=True)

def finish():
    """Print warning about interrupt and empty the job queue."""
    out.warn("Interrupted!")
    for t in threads:
        t.stop()
    jobs.clear()
    out.warn("Waiting for download threads to finish.")

def rlognormal(mu, tau, size=None):
    """
    Return random lognormal variates.
    """

    return np.random.lognormal(mu, np.sqrt(1. / tau), size)

def calculate_boundingbox(lng, lat, miles):
    """
    Given a latitude, longitude and a distance in miles, calculate
    the co-ordinates of the bounding box 2*miles on long each side with the
    given co-ordinates at the center.
    """

    latChange = change_in_latitude(miles)
    latSouth = lat - latChange
    latNorth = lat + latChange
    lngChange = change_in_longitude(lat, miles)
    lngWest = lng + lngChange
    lngEast = lng - lngChange
    return (lngWest, latSouth, lngEast, latNorth)

def uniqueID(size=6, chars=string.ascii_uppercase + string.digits):
    """A quick and dirty way to get a unique string"""
    return ''.join(random.choice(chars) for x in xrange(size))

def toBase64(s):
    """Represent string / bytes s as base64, omitting newlines"""
    if isinstance(s, str):
        s = s.encode("utf-8")
    return binascii.b2a_base64(s)[:-1]

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

def intersect(d1, d2):
    """Intersect dictionaries d1 and d2 by key *and* value."""
    return dict((k, d1[k]) for k in d1 if k in d2 and d1[k] == d2[k])

def _rndPointDisposition(dx, dy):
        """Return random disposition point."""
        x = int(random.uniform(-dx, dx))
        y = int(random.uniform(-dy, dy))
        return (x, y)

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

def sine_wave(frequency):
  """Emit a sine wave at the given frequency."""
  xs = tf.reshape(tf.range(_samples(), dtype=tf.float32), [1, _samples(), 1])
  ts = xs / FLAGS.sample_rate
  return tf.sin(2 * math.pi * frequency * ts)

def rank(idx, dim):
    """Calculate the index rank according to Bertran's notation."""
    idxm = multi_index(idx, dim)
    out = 0
    while idxm[-1:] == (0,):
        out += 1
        idxm = idxm[:-1]
    return out

def bitsToString(arr):
  """Returns a string representing a numpy array of 0's and 1's"""
  s = array('c','.'*len(arr))
  for i in xrange(len(arr)):
    if arr[i] == 1:
      s[i]='*'
  return s

def find_nearest_index(arr, value):
    """For a given value, the function finds the nearest value
    in the array and returns its index."""
    arr = np.array(arr)
    index = (abs(arr-value)).argmin()
    return index

def _get_random_id():
    """ Get a random (i.e., unique) string identifier"""
    symbols = string.ascii_uppercase + string.ascii_lowercase + string.digits
    return ''.join(random.choice(symbols) for _ in range(15))

def do_next(self, args):
        """Step over the next statement
        """
        self._do_print_from_last_cmd = True
        self._interp.step_over()
        return True

def uniqueID(size=6, chars=string.ascii_uppercase + string.digits):
    """A quick and dirty way to get a unique string"""
    return ''.join(random.choice(chars) for x in xrange(size))

def make_file_read_only(file_path):
    """
    Removes the write permissions for the given file for owner, groups and others.

    :param file_path: The file whose privileges are revoked.
    :raise FileNotFoundError: If the given file does not exist.
    """
    old_permissions = os.stat(file_path).st_mode
    os.chmod(file_path, old_permissions & ~WRITE_PERMISSIONS)

def angle(x0, y0, x1, y1):
    """ Returns the angle between two points.
    """
    return degrees(atan2(y1-y0, x1-x0))

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

def longest_run(da, dim='time'):
    """Return the length of the longest consecutive run of True values.

        Parameters
        ----------
        arr : N-dimensional array (boolean)
          Input array
        dim : Xarray dimension (default = 'time')
          Dimension along which to calculate consecutive run

        Returns
        -------
        N-dimensional array (int)
          Length of longest run of True values along dimension
        """

    d = rle(da, dim=dim)
    rl_long = d.max(dim=dim)

    return rl_long

def fopenat(base_fd, path):
    """
    Does openat read-only, then does fdopen to get a file object
    """

    return os.fdopen(openat(base_fd, path, os.O_RDONLY), 'rb')

def vars_class(cls):
    """Return a dict of vars for the given class, including all ancestors.

    This differs from the usual behaviour of `vars` which returns attributes
    belonging to the given class and not its ancestors.
    """
    return dict(chain.from_iterable(
        vars(cls).items() for cls in reversed(cls.__mro__)))

def flatten(l, types=(list, float)):
    """
    Flat nested list of lists into a single list.
    """
    l = [item if isinstance(item, types) else [item] for item in l]
    return [item for sublist in l for item in sublist]

def mean_date(dt_list):
    """Calcuate mean datetime from datetime list
    """
    dt_list_sort = sorted(dt_list)
    dt_list_sort_rel = [dt - dt_list_sort[0] for dt in dt_list_sort]
    avg_timedelta = sum(dt_list_sort_rel, timedelta())/len(dt_list_sort_rel)
    return dt_list_sort[0] + avg_timedelta

def _ensure_element(tup, elem):
    """
    Create a tuple containing all elements of tup, plus elem.

    Returns the new tuple and the index of elem in the new tuple.
    """
    try:
        return tup, tup.index(elem)
    except ValueError:
        return tuple(chain(tup, (elem,))), len(tup)

def bitdepth(self):
        """The number of bits per sample in the audio encoding (an int).
        Only available for certain file formats (zero where
        unavailable).
        """
        if hasattr(self.mgfile.info, 'bits_per_sample'):
            return self.mgfile.info.bits_per_sample
        return 0

def to_python(self, value):
        """
        Convert a string from a form into an Enum value.
        """
        if value is None:
            return value
        if isinstance(value, self.enum):
            return value
        return self.enum[value]

def get_obj_cols(df):
    """
    Returns names of 'object' columns in the DataFrame.
    """
    obj_cols = []
    for idx, dt in enumerate(df.dtypes):
        if dt == 'object' or is_category(dt):
            obj_cols.append(df.columns.values[idx])

    return obj_cols

def force_iterable(f):
    """Will make any functions return an iterable objects by wrapping its result in a list."""
    def wrapper(*args, **kwargs):
        r = f(*args, **kwargs)
        if hasattr(r, '__iter__'):
            return r
        else:
            return [r]
    return wrapper

def _read_date_from_string(str1):
    """
    Reads the date from a string in the format YYYY/MM/DD and returns
    :class: datetime.date
    """
    full_date = [int(x) for x in str1.split('/')]
    return datetime.date(full_date[0], full_date[1], full_date[2])

def wr_row_mergeall(self, worksheet, txtstr, fmt, row_idx):
        """Merge all columns and place text string in widened cell."""
        hdridxval = len(self.hdrs) - 1
        worksheet.merge_range(row_idx, 0, row_idx, hdridxval, txtstr, fmt)
        return row_idx + 1

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

def sort_fn_list(fn_list):
    """Sort input filename list by datetime
    """
    dt_list = get_dt_list(fn_list)
    fn_list_sort = [fn for (dt,fn) in sorted(zip(dt_list,fn_list))]
    return fn_list_sort

def fast_distinct(self):
        """
        Because standard distinct used on the all fields are very slow and works only with PostgreSQL database
        this method provides alternative to the standard distinct method.
        :return: qs with unique objects
        """
        return self.model.objects.filter(pk__in=self.values_list('pk', flat=True))

def Proxy(f):
  """A helper to create a proxy method in a class."""

  def Wrapped(self, *args):
    return getattr(self, f)(*args)

  return Wrapped

def metres2latlon(mx, my, origin_shift= 2 * pi * 6378137 / 2.0):
    """Converts XY point from Spherical Mercator EPSG:900913 to lat/lon in
    WGS84 Datum"""
    lon = (mx / origin_shift) * 180.0
    lat = (my / origin_shift) * 180.0

    lat = 180 / pi * (2 * atan( exp( lat * pi / 180.0)) - pi / 2.0)
    return lat, lon

def next (self):    # File-like object.

        """This is to support iterators over a file-like object.
        """

        result = self.readline()
        if result == self._empty_buffer:
            raise StopIteration
        return result

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

def unique(seq):
    """Return the unique elements of a collection even if those elements are
       unhashable and unsortable, like dicts and sets"""
    cleaned = []
    for each in seq:
        if each not in cleaned:
            cleaned.append(each)
    return cleaned

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

def _add_hash(source):
    """Add a leading hash '#' at the beginning of every line in the source."""
    source = '\n'.join('# ' + line.rstrip()
                       for line in source.splitlines())
    return source

def _rank(self, ranking, n):
    """ return the first n sentences with highest ranking """
    return nlargest(n, ranking, key=ranking.get)

def drop_indexes(self):
        """Delete all indexes for the database"""
        LOG.warning("Dropping all indexe")
        for collection_name in INDEXES:
            LOG.warning("Dropping all indexes for collection name %s", collection_name)
            self.db[collection_name].drop_indexes()

def last(self):
        """Get the last object in file."""
        # End of file
        self.__file.seek(0, 2)

        # Get the last struct
        data = self.get(self.length - 1)

        return data

def debug_src(src, pm=False, globs=None):
    """Debug a single doctest docstring, in argument `src`'"""
    testsrc = script_from_examples(src)
    debug_script(testsrc, pm, globs)

def get_last_row(dbconn, tablename, n=1, uuid=None):
    """
    Returns the last `n` rows in the table
    """
    return fetch(dbconn, tablename, n, uuid, end=True)

def save(self, fname: str):
        """
        Saves this training state to fname.
        """
        with open(fname, "wb") as fp:
            pickle.dump(self, fp)

def display_len(text):
    """
    Get the display length of a string. This can differ from the character
    length if the string contains wide characters.
    """
    text = unicodedata.normalize('NFD', text)
    return sum(char_width(char) for char in text)

def isString(s):
    """Convenience method that works with all 2.x versions of Python
    to determine whether or not something is stringlike."""
    try:
        return isinstance(s, unicode) or isinstance(s, basestring)
    except NameError:
        return isinstance(s, str)

def rel_path(filename):
    """
    Function that gets relative path to the filename
    """
    return os.path.join(os.getcwd(), os.path.dirname(__file__), filename)

def test():        
    """Local test."""
    from spyder.utils.qthelpers import qapplication
    app = qapplication()
    dlg = ProjectDialog(None)
    dlg.show()
    sys.exit(app.exec_())

def get_method_name(method):
    """
    Returns given method name.

    :param method: Method to retrieve the name.
    :type method: object
    :return: Method name.
    :rtype: unicode
    """

    name = get_object_name(method)
    if name.startswith("__") and not name.endswith("__"):
        name = "_{0}{1}".format(get_object_name(method.im_class), name)
    return name

def const_rand(size, seed=23980):
    """ Generate a random array with a fixed seed.
    """
    old_seed = np.random.seed()
    np.random.seed(seed)
    out = np.random.rand(size)
    np.random.seed(old_seed)
    return out

def get_action_methods(self):
        """
        return a list of methods on this class for executing actions.
        methods are return as a list of (name, func) tuples
        """
        return [(name, getattr(self, name))
                for name, _ in Action.get_command_types()]

def start():
    """Starts the web server."""
    global app
    bottle.run(app, host=conf.WebHost, port=conf.WebPort,
               debug=conf.WebAutoReload, reloader=conf.WebAutoReload,
               quiet=conf.WebQuiet)

async def sysinfo(dev: Device):
    """Print out system information (version, MAC addrs)."""
    click.echo(await dev.get_system_info())
    click.echo(await dev.get_interface_information())

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

def post_object_async(self, path, **kwds):
    """POST to an object."""
    return self.do_request_async(self.api_url + path, 'POST', **kwds)

def findfirst(f, coll):
    """Return first occurrence matching f, otherwise None"""
    result = list(dropwhile(f, coll))
    return result[0] if result else None

def start(self):
        """Create a background thread for httpd and serve 'forever'"""
        self._process = threading.Thread(target=self._background_runner)
        self._process.start()

def return_letters_from_string(text):
    """Get letters from string only."""
    out = ""
    for letter in text:
        if letter.isalpha():
            out += letter
    return out

def start(self):
        """Create a background thread for httpd and serve 'forever'"""
        self._process = threading.Thread(target=self._background_runner)
        self._process.start()

def parse_querystring(self, req, name, field):
        """Pull a querystring value from the request."""
        return core.get_value(req.args, name, field)

def inject_into_urllib3():
    """
    Monkey-patch urllib3 with SecureTransport-backed SSL-support.
    """
    util.ssl_.SSLContext = SecureTransportContext
    util.HAS_SNI = HAS_SNI
    util.ssl_.HAS_SNI = HAS_SNI
    util.IS_SECURETRANSPORT = True
    util.ssl_.IS_SECURETRANSPORT = True

def strip_spaces(x):
    """
    Strips spaces
    :param x:
    :return:
    """
    x = x.replace(b' ', b'')
    x = x.replace(b'\t', b'')
    return x

def fmt_duration(secs):
    """Format a duration in seconds."""
    return ' '.join(fmt.human_duration(secs, 0, precision=2, short=True).strip().split())

def argsort_indices(a, axis=-1):
    """Like argsort, but returns an index suitable for sorting the
    the original array even if that array is multidimensional
    """
    a = np.asarray(a)
    ind = list(np.ix_(*[np.arange(d) for d in a.shape]))
    ind[axis] = a.argsort(axis)
    return tuple(ind)

def file_or_default(path, default, function = None):
    """ Return a default value if a file does not exist """
    try:
        result = file_get_contents(path)
        if function != None: return function(result)
        return result
    except IOError as e:
        if e.errno == errno.ENOENT: return default
        raise

def this_quarter():
        """ Return start and end date of this quarter. """
        since = TODAY + delta(day=1)
        while since.month % 3 != 0:
            since -= delta(months=1)
        until = since + delta(months=3)
        return Date(since), Date(until)

def strToBool(val):
    """
    Helper function to turn a string representation of "true" into
    boolean True.
    """
    if isinstance(val, str):
        val = val.lower()

    return val in ['true', 'on', 'yes', True]

def get_month_start(day=None):
    """Returns the first day of the given month."""
    day = add_timezone(day or datetime.date.today())
    return day.replace(day=1)

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

def _string_width(self, s):
        """Get width of a string in the current font"""
        s = str(s)
        w = 0
        for i in s:
            w += self.character_widths[i]
        return w * self.font_size / 1000.0

def isString(s):
    """Convenience method that works with all 2.x versions of Python
    to determine whether or not something is stringlike."""
    try:
        return isinstance(s, unicode) or isinstance(s, basestring)
    except NameError:
        return isinstance(s, str)

def find_le(a, x):
    """Find rightmost value less than or equal to x."""
    i = bs.bisect_right(a, x)
    if i: return i - 1
    raise ValueError

def crop_box(im, box=False, **kwargs):
    """Uses box coordinates to crop an image without resizing it first."""
    if box:
        im = im.crop(box)
    return im

def datatype(dbtype, description, cursor):
    """Google AppEngine Helper to convert a data type into a string."""
    dt = cursor.db.introspection.get_field_type(dbtype, description)
    if type(dt) is tuple:
        return dt[0]
    else:
        return dt

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

def end_index(self):
        """Return the 1-based index of the last item on this page."""
        paginator = self.paginator
        # Special case for the last page because there can be orphans.
        if self.number == paginator.num_pages:
            return paginator.count
        return (self.number - 1) * paginator.per_page + paginator.first_page

def filtered_image(self, im):
        """Returns a filtered image after applying the Fourier-space filters"""
        q = np.fft.fftn(im)
        for k,v in self.filters:
            q[k] -= v
        return np.real(np.fft.ifftn(q))

def get_last_row(dbconn, tablename, n=1, uuid=None):
    """
    Returns the last `n` rows in the table
    """
    return fetch(dbconn, tablename, n, uuid, end=True)

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

def uint32_to_uint8(cls, img):
        """
        Cast uint32 RGB image to 4 uint8 channels.
        """
        return np.flipud(img.view(dtype=np.uint8).reshape(img.shape + (4,)))

def get_user_by_id(self, id):
        """Retrieve a User object by ID."""
        return self.db_adapter.get_object(self.UserClass, id=id)

def asynchronous(function, event):
    """
    Runs the function asynchronously taking care of exceptions.
    """
    thread = Thread(target=synchronous, args=(function, event))
    thread.daemon = True
    thread.start()

def reduce_fn(x):
    """
    Aggregation function to get the first non-zero value.
    """
    values = x.values if pd and isinstance(x, pd.Series) else x
    for v in values:
        if not is_nan(v):
            return v
    return np.NaN

def _EnforceProcessMemoryLimit(self, memory_limit):
    """Enforces a process memory limit.

    Args:
      memory_limit (int): maximum number of bytes the process is allowed
          to allocate, where 0 represents no limit and None a default of
          4 GiB.
    """
    # Resource is not supported on Windows.
    if resource:
      if memory_limit is None:
        memory_limit = 4 * 1024 * 1024 * 1024
      elif memory_limit == 0:
        memory_limit = resource.RLIM_INFINITY

      resource.setrlimit(resource.RLIMIT_DATA, (memory_limit, memory_limit))

def check_many(self, domains):
        """
        Check availability for a number of domains. Returns a dictionary
        mapping the domain names to their statuses as a string
        ("active"/"free").
        """
        return dict((item.domain, item.status) for item in self.check_domain_request(domains))

def end_block(self):
        """Ends an indentation block, leaving an empty line afterwards"""
        self.current_indent -= 1

        # If we did not add a new line automatically yet, now it's the time!
        if not self.auto_added_line:
            self.writeln()
            self.auto_added_line = True

def get_weights_from_kmodel(kmodel):
        """
        Convert kmodel's weights to bigdl format.
        We are supposing the order is the same as the execution order.
        :param kmodel: keras model
        :return: list of ndarray
        """
        layers_with_weights = [layer for layer in kmodel.layers if layer.weights]
        bweights = []
        for klayer in layers_with_weights:
            # bws would be [weights, bias] or [weights]
            bws = WeightsConverter.get_bigdl_weights_from_klayer(klayer)
            for w in bws:
                bweights.append(w)
        return bweights

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

def get_nt_system_uid():
    """Get the MachineGuid from
    HKEY_LOCAL_MACHINE\Software\Microsoft\Cryptography\MachineGuid
    """
    try:
        import _winreg as winreg
    except ImportError:
        import winreg
    lm = winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE)
    try:
        key = winreg.OpenKey(lm, r"Software\Microsoft\Cryptography")
        try:
            return winreg.QueryValueEx(key, "MachineGuid")[0]
        finally:
            key.Close()
    finally:
        lm.Close()

def __init__(self, find, subcon):
        """Initialize."""
        Subconstruct.__init__(self, subcon)
        self.find = find

def value(self):
        """Value of property."""
        if self._prop.fget is None:
            raise AttributeError('Unable to read attribute')
        return self._prop.fget(self._obj)

def prepend_line(filepath, line):
    """Rewrite a file adding a line to its beginning.
    """
    with open(filepath) as f:
        lines = f.readlines()

    lines.insert(0, line)

    with open(filepath, 'w') as f:
        f.writelines(lines)

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

def get_func_posargs_name(f):
    """Returns the name of the function f's keyword argument parameter if it exists, otherwise None"""
    sigparams = inspect.signature(f).parameters
    for p in sigparams:
        if sigparams[p].kind == inspect.Parameter.VAR_POSITIONAL:
            return sigparams[p].name
    return None

def check_git():
    """Check if git command is available."""
    try:
        with open(os.devnull, "wb") as devnull:
            subprocess.check_call(["git", "--version"], stdout=devnull, stderr=devnull)
    except:
        raise RuntimeError("Please make sure git is installed and on your path.")

def as_list(self):
        """Return all child objects in nested lists of strings."""
        return [self.name, self.value, [x.as_list for x in self.children]]

def positive_integer(anon, obj, field, val):
    """
    Returns a random positive integer (for a Django PositiveIntegerField)
    """
    return anon.faker.positive_integer(field=field)

def method(func):
    """Wrap a function as a method."""
    attr = abc.abstractmethod(func)
    attr.__imethod__ = True
    return attr

def get_lines(handle, line):
    """
    Get zero-indexed line from an open file-like.
    """
    for i, l in enumerate(handle):
        if i == line:
            return l

def type_converter(text):
    """ I convert strings into integers, floats, and strings! """
    if text.isdigit():
        return int(text), int

    try:
        return float(text), float
    except ValueError:
        return text, STRING_TYPE

def create_path(path):
    """Creates a absolute path in the file system.

    :param path: The path to be created
    """
    import os
    if not os.path.exists(path):
        os.makedirs(path)

def is_int(value):
    """Return `True` if ``value`` is an integer."""
    if isinstance(value, bool):
        return False
    try:
        int(value)
        return True
    except (ValueError, TypeError):
        return False

def h5ToDict(h5, readH5pyDataset=True):
    """ Read a hdf5 file into a dictionary """
    h = h5py.File(h5, "r")
    ret = unwrapArray(h, recursive=True, readH5pyDataset=readH5pyDataset)
    if readH5pyDataset: h.close()
    return ret

def int_to_date(date):
    """
    Convert an int of form yyyymmdd to a python date object.
    """

    year = date // 10**4
    month = date % 10**4 // 10**2
    day = date % 10**2

    return datetime.date(year, month, day)

def norm(x, mu, sigma=1.0):
    """ Scipy norm function """
    return stats.norm(loc=mu, scale=sigma).pdf(x)

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

def add_noise(Y, sigma):
    """Adds noise to Y"""
    return Y + np.random.normal(0, sigma, Y.shape)

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

def _load_data(filepath):
  """Loads the images and latent values into Numpy arrays."""
  with h5py.File(filepath, "r") as h5dataset:
    image_array = np.array(h5dataset["images"])
    # The 'label' data set in the hdf5 file actually contains the float values
    # and not the class labels.
    values_array = np.array(h5dataset["labels"])
  return image_array, values_array

def invertDictMapping(d):
    """ Invert mapping of dictionary (i.e. map values to list of keys) """
    inv_map = {}
    for k, v in d.items():
        inv_map[v] = inv_map.get(v, [])
        inv_map[v].append(k)
    return inv_map

def chunk_list(l, n):
    """Return `n` size lists from a given list `l`"""
    return [l[i:i + n] for i in range(0, len(l), n)]

def is_valid_ip(ip_address):
    """
    Check Validity of an IP address
    """
    valid = True
    try:
        socket.inet_aton(ip_address.strip())
    except:
        valid = False
    return valid

def main(idle):
    """Any normal python logic which runs a loop. Can take arguments."""
    while True:

        LOG.debug("Sleeping for {0} seconds.".format(idle))
        time.sleep(idle)

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

def _init_unique_sets(self):
        """Initialise sets used for uniqueness checking."""

        ks = dict()
        for t in self._unique_checks:
            key = t[0]
            ks[key] = set() # empty set
        return ks

def is_a_sequence(var, allow_none=False):
    """ Returns True if var is a list or a tuple (but not a string!)
    """
    return isinstance(var, (list, tuple)) or (var is None and allow_none)

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

def conv_dict(self):
        """dictionary of conversion"""
        return dict(integer=self.integer, real=self.real, no_type=self.no_type)

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

def get_hline():
    """ gets a horiztonal line """
    return Window(
        width=LayoutDimension.exact(1),
        height=LayoutDimension.exact(1),
        content=FillControl('-', token=Token.Line))

def _writable_dir(path):
    """Whether `path` is a directory, to which the user has write access."""
    return os.path.isdir(path) and os.access(path, os.W_OK)

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

def isdir(path, **kwargs):
    """Check if *path* is a directory"""
    import os.path
    return os.path.isdir(path, **kwargs)

def batch(items, size):
    """Batches a list into a list of lists, with sub-lists sized by a specified
    batch size."""
    return [items[x:x + size] for x in xrange(0, len(items), size)]

def is_float_array(l):
    r"""Checks if l is a numpy array of floats (any dimension

    """
    if isinstance(l, np.ndarray):
        if l.dtype.kind == 'f':
            return True
    return False

def myreplace(astr, thefind, thereplace):
    """in string astr replace all occurences of thefind with thereplace"""
    alist = astr.split(thefind)
    new_s = alist.split(thereplace)
    return new_s

def is_iter_non_string(obj):
    """test if object is a list or tuple"""
    if isinstance(obj, list) or isinstance(obj, tuple):
        return True
    return False

def round_to_x_digits(number, digits):
    """
    Returns 'number' rounded to 'digits' digits.
    """
    return round(number * math.pow(10, digits)) / math.pow(10, digits)

def __next__(self):
    """Pop the head off the iterator and return it."""
    res = self._head
    self._fill()
    if res is None:
      raise StopIteration()
    return res

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

def next (self):    # File-like object.

        """This is to support iterators over a file-like object.
        """

        result = self.readline()
        if result == self._empty_buffer:
            raise StopIteration
        return result

def as_tuple(self, value):
        """Utility function which converts lists to tuples."""
        if isinstance(value, list):
            value = tuple(value)
        return value

def __reversed__(self):
        """
        Return a reversed iterable over the items in the dictionary. Items are
        iterated over in their reverse sort order.

        Iterating views while adding or deleting entries in the dictionary may
        raise a RuntimeError or fail to iterate over all entries.
        """
        _dict = self._dict
        return iter((key, _dict[key]) for key in reversed(self._list))

def register_modele(self, modele: Modele):
        """ Register a modele onto the lemmatizer

        :param modele: Modele to register
        """
        self.lemmatiseur._modeles[modele.gr()] = modele

def split_every(n, iterable):
    """Returns a generator that spits an iteratable into n-sized chunks. The last chunk may have
    less than n elements.

    See http://stackoverflow.com/a/22919323/503377."""
    items = iter(iterable)
    return itertools.takewhile(bool, (list(itertools.islice(items, n)) for _ in itertools.count()))

def flatten(l):
    """Flatten a nested list."""
    return sum(map(flatten, l), []) \
        if isinstance(l, list) or isinstance(l, tuple) else [l]

def directory_files(path):
    """Yield directory file names."""

    for entry in os.scandir(path):
        if not entry.name.startswith('.') and entry.is_file():
            yield entry.name

def read_array(path, mmap_mode=None):
    """Read a .npy array."""
    file_ext = op.splitext(path)[1]
    if file_ext == '.npy':
        return np.load(path, mmap_mode=mmap_mode)
    raise NotImplementedError("The file extension `{}` ".format(file_ext) +
                              "is not currently supported.")

def group_by(iterable, key_func):
    """Wrap itertools.groupby to make life easier."""
    groups = (
        list(sub) for key, sub in groupby(iterable, key_func)
    )
    return zip(groups, groups)

def get_python():
    """Determine the path to the virtualenv python"""
    if sys.platform == 'win32':
        python = path.join(VE_ROOT, 'Scripts', 'python.exe')
    else:
        python = path.join(VE_ROOT, 'bin', 'python')
    return python

def render_template(template_name, **context):
    """Render a template into a response."""
    tmpl = jinja_env.get_template(template_name)
    context["url_for"] = url_for
    return Response(tmpl.render(context), mimetype="text/html")

def selectnone(table, field, complement=False):
    """Select rows where the given field is `None`."""

    return select(table, field, lambda v: v is None, complement=complement)

def _join(verb):
    """
    Join helper
    """
    data = pd.merge(verb.x, verb.y, **verb.kwargs)

    # Preserve x groups
    if isinstance(verb.x, GroupedDataFrame):
        data.plydata_groups = list(verb.x.plydata_groups)
    return data

def stn(s, length, encoding, errors):
    """Convert a string to a null-terminated bytes object.
    """
    s = s.encode(encoding, errors)
    return s[:length] + (length - len(s)) * NUL

def join_images(img_files, out_file):
    """Join the list of images into the out file"""
    images = [PIL.Image.open(f) for f in img_files]
    joined = PIL.Image.new(
        'RGB',
        (sum(i.size[0] for i in images), max(i.size[1] for i in images))
    )
    left = 0
    for img in images:
        joined.paste(im=img, box=(left, 0))
        left = left + img.size[0]
    joined.save(out_file)

def _dict(content):
    """
    Helper funcation that converts text-based get response
    to a python dictionary for additional manipulation.
    """
    if _has_pandas:
        data = _data_frame(content).to_dict(orient='records')
    else:
        response = loads(content)
        key = [x for x in response.keys() if x in c.response_data][0]
        data = response[key]
    return data

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

def IPYTHON_MAIN():
    """Decide if the Ipython command line is running code."""
    import pkg_resources

    runner_frame = inspect.getouterframes(inspect.currentframe())[-2]
    return (
        getattr(runner_frame, "function", None)
        == pkg_resources.load_entry_point("ipython", "console_scripts", "ipython").__name__
    )

def flatten_dict_join_keys(dct, join_symbol=" "):
    """ Flatten dict with defined key join symbol.

    :param dct: dict to flatten
    :param join_symbol: default value is " "
    :return:
    """
    return dict( flatten_dict(dct, join=lambda a,b:a+join_symbol+b) )

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

def _to_diagonally_dominant(mat):
    """Make matrix unweighted diagonally dominant using the Laplacian."""
    mat += np.diag(np.sum(mat != 0, axis=1) + 0.01)
    return mat

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

def traverse_setter(obj, attribute, value):
    """
    Traverses the object and sets the supplied attribute on the
    object. Supports Dimensioned and DimensionedPlot types.
    """
    obj.traverse(lambda x: setattr(x, attribute, value))

def add_to_js(self, name, var):
        """Add an object to Javascript."""
        frame = self.page().mainFrame()
        frame.addToJavaScriptWindowObject(name, var)

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

def dump_json(obj):
    """Dump Python object as JSON string."""
    return simplejson.dumps(obj, ignore_nan=True, default=json_util.default)

def get_property(self, filename):
        """Opens the file and reads the value"""

        with open(self.filepath(filename)) as f:
            return f.read().strip()

def pretty_dict_str(d, indent=2):
    """shows JSON indented representation of d"""
    b = StringIO()
    write_pretty_dict_str(b, d, indent=indent)
    return b.getvalue()

def help_for_command(command):
    """Get the help text (signature + docstring) for a command (function)."""
    help_text = pydoc.text.document(command)
    # remove backspaces
    return re.subn('.\\x08', '', help_text)[0]

def save(self, fname):
        """ Saves the dictionary in json format
        :param fname: file to save to
        """
        with open(fname, 'wb') as f:
            json.dump(self, f)

def prepend_line(filepath, line):
    """Rewrite a file adding a line to its beginning.
    """
    with open(filepath) as f:
        lines = f.readlines()

    lines.insert(0, line)

    with open(filepath, 'w') as f:
        f.writelines(lines)

def validate(raw_schema, target=None, **kwargs):
    """
    Given the python representation of a JSONschema as defined in the swagger
    spec, validate that the schema complies to spec.  If `target` is provided,
    that target will be validated against the provided schema.
    """
    schema = schema_validator(raw_schema, **kwargs)
    if target is not None:
        validate_object(target, schema=schema, **kwargs)

def build_output(self, fout):
        """Squash self.out into string.

        Join every line in self.out with a new line and write the
        result to the output file.
        """
        fout.write('\n'.join([s for s in self.out]))

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, LegipyModel):
        return obj.to_json()
    elif isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    raise TypeError("Type {0} not serializable".format(repr(type(obj))))

def generic_add(a, b):
    """Simple function to add two numbers"""
    logger.debug('Called generic_add({}, {})'.format(a, b))
    return a + b

def _unjsonify(x, isattributes=False):
    """Convert JSON string to an ordered defaultdict."""
    if isattributes:
        obj = json.loads(x)
        return dict_class(obj)
    return json.loads(x)

def get_absolute_path(*args):
    """Transform relative pathnames into absolute pathnames."""
    directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(directory, *args)

def graphql_queries_to_json(*queries):
    """
    Queries should be a list of GraphQL objects
    """
    rtn = {}
    for i, query in enumerate(queries):
        rtn["q{}".format(i)] = query.value
    return json.dumps(rtn)

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

def _clean_dict(target_dict, whitelist=None):
    """ Convenience function that removes a dicts keys that have falsy values
    """
    assert isinstance(target_dict, dict)
    return {
        ustr(k).strip(): ustr(v).strip()
        for k, v in target_dict.items()
        if v not in (None, Ellipsis, [], (), "")
        and (not whitelist or k in whitelist)
    }

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

def calculate_embedding(self, batch_image_bytes):
    """Get the embeddings for a given JPEG image.

    Args:
      batch_image_bytes: As if returned from [ff.read() for ff in file_list].

    Returns:
      The Inception embeddings (bottleneck layer output)
    """
    return self.tf_session.run(
        self.embedding, feed_dict={self.input_jpeg: batch_image_bytes})

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

def copy_no_perm(src, dst):
    """
    Copies a file from *src* to *dst* including meta data except for permission bits.
    """
    shutil.copy(src, dst)
    perm = os.stat(dst).st_mode
    shutil.copystat(src, dst)
    os.chmod(dst, perm)

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

def flatten_dict_join_keys(dct, join_symbol=" "):
    """ Flatten dict with defined key join symbol.

    :param dct: dict to flatten
    :param join_symbol: default value is " "
    :return:
    """
    return dict( flatten_dict(dct, join=lambda a,b:a+join_symbol+b) )

def filename_addstring(filename, text):
    """
    Add `text` to filename, keeping the extension in place
    For example when adding a timestamp to the filename
    """
    fn, ext = os.path.splitext(filename)
    return fn + text + ext

def pop(self):
        """
        return the last stack element and delete it from the list
        """
        if not self.empty():
            val = self.stack[-1]
            del self.stack[-1]
            return val

def interpolate_logscale_single(start, end, coefficient):
    """ Cosine interpolation """
    return np.exp(np.log(start) + (np.log(end) - np.log(start)) * coefficient)

def get_last_modified_timestamp(self):
        """
        Looks at the files in a git root directory and grabs the last modified timestamp
        """
        cmd = "find . -print0 | xargs -0 stat -f '%T@ %p' | sort -n | tail -1 | cut -f2- -d' '"
        ps = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        output = ps.communicate()[0]
        print output

def _stdin_(p):
    """Takes input from user. Works for Python 2 and 3."""
    _v = sys.version[0]
    return input(p) if _v is '3' else raw_input(p)

def get_list_dimensions(_list):
    """
    Takes a nested list and returns the size of each dimension followed
    by the element type in the list
    """
    if isinstance(_list, list) or isinstance(_list, tuple):
        return [len(_list)] + get_list_dimensions(_list[0])
    return []

def sort_filenames(filenames):
    """
    sort a list of files by filename only, ignoring the directory names
    """
    basenames = [os.path.basename(x) for x in filenames]
    indexes = [i[0] for i in sorted(enumerate(basenames), key=lambda x:x[1])]
    return [filenames[x] for x in indexes]

def levenshtein_distance_metric(a, b):
    """ 1 - farthest apart (same number of words, all diff). 0 - same"""
    return (levenshtein_distance(a, b) / (2.0 * max(len(a), len(b), 1)))

def wait_until_exit(self):
        """ Wait until thread exit

            Used for testing purpose only
        """

        if self._timeout is None:
            raise Exception("Thread will never exit. Use stop or specify timeout when starting it!")

        self._thread.join()
        self.stop()

def timed (log=sys.stderr, limit=2.0):
    """Decorator to run a function with timing info."""
    return lambda func: timeit(func, log, limit)

def dict_jsonp(param):
    """Convert the parameter into a dictionary before calling jsonp, if it's not already one"""
    if not isinstance(param, dict):
        param = dict(param)
    return jsonp(param)

def txt_line_iterator(path):
  """Iterate through lines of file."""
  with tf.gfile.Open(path) as f:
    for line in f:
      yield line.strip()

def get_size(objects):
    """Compute the total size of all elements in objects."""
    res = 0
    for o in objects:
        try:
            res += _getsizeof(o)
        except AttributeError:
            print("IGNORING: type=%s; o=%s" % (str(type(o)), str(o)))
    return res

def distinct(xs):
    """Get the list of distinct values with preserving order."""
    # don't use collections.OrderedDict because we do support Python 2.6
    seen = set()
    return [x for x in xs if x not in seen and not seen.add(x)]

def stderr(a):
    """
    Calculate the standard error of a.
    """
    return np.nanstd(a) / np.sqrt(sum(np.isfinite(a)))

def get_table_names(connection):
	"""
	Return a list of the table names in the database.
	"""
	cursor = connection.cursor()
	cursor.execute("SELECT name FROM sqlite_master WHERE type == 'table'")
	return [name for (name,) in cursor]

def time(func, *args, **kwargs):
    """
    Call the supplied function with the supplied arguments,
    and return the total execution time as a float in seconds.

    The precision of the returned value depends on the precision of
    `time.time()` on your platform.

    Arguments:
        func: the function to run.
        *args: positional arguments to pass into the function.
        **kwargs: keyword arguments to pass into the function.
    Returns:
        Execution time of the function as a float in seconds.
    """
    start_time = time_module.time()
    func(*args, **kwargs)
    end_time = time_module.time()
    return end_time - start_time

def equal(list1, list2):
    """ takes flags returns indexes of True values """
    return [item1 == item2 for item1, item2 in broadcast_zip(list1, list2)]

def camel_case_from_underscores(string):
    """generate a CamelCase string from an underscore_string."""
    components = string.split('_')
    string = ''
    for component in components:
        string += component[0].upper() + component[1:]
    return string

def classnameify(s):
  """
  Makes a classname
  """
  return ''.join(w if w in ACRONYMS else w.title() for w in s.split('_'))

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

def force_to_string(unknown):
    """
    converts and unknown type to string for display purposes.
    
    """
    result = ''
    if type(unknown) is str:
        result = unknown
    if type(unknown) is int:
        result = str(unknown)
    if type(unknown) is float:
        result = str(unknown)
    if type(unknown) is dict:
        result = Dict2String(unknown)
    if type(unknown) is list:
        result = List2String(unknown)
    return result

def nan_pixels(self):
        """ Return an array of the NaN pixels.

        Returns
        -------
        :obj:`numpy.ndarray`
             Nx2 array of the NaN pixels
        """
        nan_px = np.where(np.isnan(np.sum(self.raw_data, axis=2)))
        nan_px = np.c_[nan_px[0], nan_px[1]]
        return nan_px

def clean_error(err):
    """
    Take stderr bytes returned from MicroPython and attempt to create a
    non-verbose error message.
    """
    if err:
        decoded = err.decode('utf-8')
        try:
            return decoded.split('\r\n')[-2]
        except Exception:
            return decoded
    return 'There was an error.'

def recarray(self):
        """Returns data as :class:`numpy.recarray`."""
        return numpy.rec.fromrecords(self.records, names=self.names)

def downcaseTokens(s,l,t):
    """Helper parse action to convert tokens to lower case."""
    return [ tt.lower() for tt in map(_ustr,t) ]

def torecarray(*args, **kwargs):
    """
    Convenient shorthand for ``toarray(*args, **kwargs).view(np.recarray)``.

    """

    import numpy as np
    return toarray(*args, **kwargs).view(np.recarray)

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

def get_all_attributes(klass_or_instance):
    """Get all attribute members (attribute, property style method).
    """
    pairs = list()
    for attr, value in inspect.getmembers(
            klass_or_instance, lambda x: not inspect.isroutine(x)):
        if not (attr.startswith("__") or attr.endswith("__")):
            pairs.append((attr, value))
    return pairs

def session_to_epoch(timestamp):
    """ converts Synergy Timestamp for session to UTC zone seconds since epoch """
    utc_timetuple = datetime.strptime(timestamp, SYNERGY_SESSION_PATTERN).replace(tzinfo=None).utctimetuple()
    return calendar.timegm(utc_timetuple)

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

def end_table_header(self):
        r"""End the table header which will appear on every page."""

        if self.header:
            msg = "Table already has a header"
            raise TableError(msg)

        self.header = True

        self.append(Command(r'endhead'))

def main(args=sys.argv):
    """
    main entry point for the jardiff CLI
    """

    parser = create_optparser(args[0])
    return cli(parser.parse_args(args[1:]))

def raise_os_error(_errno, path=None):
    """
    Helper for raising the correct exception under Python 3 while still
    being able to raise the same common exception class in Python 2.7.
    """

    msg = "%s: '%s'" % (strerror(_errno), path) if path else strerror(_errno)
    raise OSError(_errno, msg)

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

def _multiline_width(multiline_s, line_width_fn=len):
    """Visible width of a potentially multiline content."""
    return max(map(line_width_fn, re.split("[\r\n]", multiline_s)))

def exec_function(ast, globals_map):
    """Execute a python code object in the given environment.

    Args:
      globals_map: Dictionary to use as the globals context.
    Returns:
      locals_map: Dictionary of locals from the environment after execution.
    """
    locals_map = globals_map
    exec ast in globals_map, locals_map
    return locals_map

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

def _include_yaml(loader, node):
    """Load another YAML file and embeds it using the !include tag.

    Example:
        device_tracker: !include device_tracker.yaml
    """
    return load_yaml(os.path.join(os.path.dirname(loader.name), node.value))

def comma_converter(float_string):
    """Convert numbers to floats whether the decimal point is '.' or ','"""
    trans_table = maketrans(b',', b'.')
    return float(float_string.translate(trans_table))

def datetime_local_to_utc(local):
    """
    Simple function to convert naive :std:`datetime.datetime` object containing
    local time to a naive :std:`datetime.datetime` object with UTC time.
    """
    timestamp = time.mktime(local.timetuple())
    return datetime.datetime.utcfromtimestamp(timestamp)

def to_identifier(s):
  """
  Convert snake_case to camel_case.
  """
  if s.startswith('GPS'):
      s = 'Gps' + s[3:]
  return ''.join([i.capitalize() for i in s.split('_')]) if '_' in s else s

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

def image_set_aspect(aspect=1.0, axes="gca"):
    """
    sets the aspect ratio of the current zoom level of the imshow image
    """
    if axes is "gca": axes = _pylab.gca()

    e = axes.get_images()[0].get_extent()
    axes.set_aspect(abs((e[1]-e[0])/(e[3]-e[2]))/aspect)

def lock(self, block=True):
		"""
		Lock connection from being used else where
		"""
		self._locked = True
		return self._lock.acquire(block)

def _validate_pos(df):
    """Validates the returned positional object
    """
    assert isinstance(df, pd.DataFrame)
    assert ["seqname", "position", "strand"] == df.columns.tolist()
    assert df.position.dtype == np.dtype("int64")
    assert df.strand.dtype == np.dtype("O")
    assert df.seqname.dtype == np.dtype("O")
    return df

def autoconvert(string):
    """Try to convert variables into datatypes."""
    for fn in (boolify, int, float):
        try:
            return fn(string)
        except ValueError:
            pass
    return string

def to_distribution_values(self, values):
        """
        Returns numpy array of natural logarithms of ``values``.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # avoid RuntimeWarning: divide by zero encountered in log
            return numpy.log(values)

def _valid_other_type(x, types):
    """
    Do all elements of x have a type from types?
    """
    return all(any(isinstance(el, t) for t in types) for el in np.ravel(x))

def find_console_handler(logger):
    """Return a stream handler, if it exists."""
    for handler in logger.handlers:
        if (isinstance(handler, logging.StreamHandler) and
                handler.stream == sys.stderr):
            return handler

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

def clog(color):
    """Same to ``log``, but this one centralizes the message first."""
    logger = log(color)
    return lambda msg: logger(centralize(msg).rstrip())

def is_defined(self, obj, force_import=False):
        """Return True if object is defined in current namespace"""
        from spyder_kernels.utils.dochelpers import isdefined

        ns = self._get_current_namespace(with_magics=True)
        return isdefined(obj, force_import=force_import, namespace=ns)

def format(self, record, *args, **kwargs):
        """
        Format a message in the log

        Act like the normal format, but indent anything that is a
        newline within the message.

        """
        return logging.Formatter.format(
            self, record, *args, **kwargs).replace('\n', '\n' + ' ' * 8)

def is_delimiter(line):
    """ True if a line consists only of a single punctuation character."""
    return bool(line) and line[0] in punctuation and line[0]*len(line) == line

def load_config(filename="logging.ini", *args, **kwargs):
    """
    Load logger config from file
    
    Keyword arguments:
    filename -- configuration filename (Default: "logging.ini")
    *args -- options passed to fileConfig
    **kwargs -- options passed to fileConfigg
    
    """
    logging.config.fileConfig(filename, *args, **kwargs)

def is_parameter(self):
        """Whether this is a function parameter."""
        return (isinstance(self.scope, CodeFunction)
                and self in self.scope.parameters)

def print_log(value_color="", value_noncolor=""):
    """set the colors for text."""
    HEADER = '\033[92m'
    ENDC = '\033[0m'
    print(HEADER + value_color + ENDC + str(value_noncolor))

def is_seq(obj):
    """ Returns True if object is not a string but is iterable """
    if not hasattr(obj, '__iter__'):
        return False
    if isinstance(obj, basestring):
        return False
    return True

def logger(message, level=10):
    """Handle logging."""
    logging.getLogger(__name__).log(level, str(message))

def is_listish(obj):
    """Check if something quacks like a list."""
    if isinstance(obj, (list, tuple, set)):
        return True
    return is_sequence(obj)

def _convert_to_array(array_like, dtype):
        """
        Convert Matrix attributes which are array-like or buffer to array.
        """
        if isinstance(array_like, bytes):
            return np.frombuffer(array_like, dtype=dtype)
        return np.asarray(array_like, dtype=dtype)

def isin(value, values):
    """ Check that value is in values """
    for i, v in enumerate(value):
        if v not in np.array(values)[:, i]:
            return False
    return True

def bitsToString(arr):
  """Returns a string representing a numpy array of 0's and 1's"""
  s = array('c','.'*len(arr))
  for i in xrange(len(arr)):
    if arr[i] == 1:
      s[i]='*'
  return s

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

def _match_literal(self, a, b=None):
        """Match two names."""

        return a.lower() == b if not self.case_sensitive else a == b

def make_symmetric(dict):
    """Makes the given dictionary symmetric. Values are assumed to be unique."""
    for key, value in list(dict.items()):
        dict[value] = key
    return dict

def isnumber(*args):
    """Checks if value is an integer, long integer or float.

    NOTE: Treats booleans as numbers, where True=1 and False=0.
    """
    return all(map(lambda c: isinstance(c, int) or isinstance(c, float), args))

def relpath(path):
    """Path helper, gives you a path relative to this file"""
    return os.path.normpath(
        os.path.join(os.path.abspath(os.path.dirname(__file__)), path)
    )

def cudaDriverGetVersion():
    """
    Get installed CUDA driver version.

    Return the version of the installed CUDA driver as an integer. If
    no driver is detected, 0 is returned.

    Returns
    -------
    version : int
        Driver version.

    """

    version = ctypes.c_int()
    status = _libcudart.cudaDriverGetVersion(ctypes.byref(version))
    cudaCheckStatus(status)
    return version.value

def israw(self, **kwargs):
        """
        Returns True if the PTY should operate in raw mode.

        If the container was not started with tty=True, this will return False.
        """

        if self.raw is None:
            info = self._container_info()
            self.raw = self.stdout.isatty() and info['Config']['Tty']

        return self.raw

def skip_connection_distance(a, b):
    """The distance between two skip-connections."""
    if a[2] != b[2]:
        return 1.0
    len_a = abs(a[1] - a[0])
    len_b = abs(b[1] - b[0])
    return (abs(a[0] - b[0]) + abs(len_a - len_b)) / (max(a[0], b[0]) + max(len_a, len_b))

def html(header_rows):
    """
    Convert a list of tuples describing a table into a HTML string
    """
    name = 'table%d' % next(tablecounter)
    return HtmlTable([map(str, row) for row in header_rows], name).render()

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

def ver_to_tuple(value):
    """
    Convert version like string to a tuple of integers.
    """
    return tuple(int(_f) for _f in re.split(r'\D+', value) if _f)

def is_type(value):
        """Determine if value is an instance or subclass of the class Type."""
        if isinstance(value, type):
            return issubclass(value, Type)
        return isinstance(value, Type)

def _strvar(a, prec='{:G}'):
    r"""Return variable as a string to print, with given precision."""
    return ' '.join([prec.format(i) for i in np.atleast_1d(a)])

def check_filename(filename):
    """
    Returns a boolean stating if the filename is safe to use or not. Note that
    this does not test for "legal" names accepted, but a more restricted set of:
    Letters, numbers, spaces, hyphens, underscores and periods.

    :param filename: name of a file as a string
    :return: boolean if it is a safe file name
    """
    if not isinstance(filename, str):
        raise TypeError("filename must be a string")
    if regex.path.linux.filename.search(filename):
        return True
    return False

def GeneratePassphrase(length=20):
  """Create a 20 char passphrase with easily typeable chars."""
  valid_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
  valid_chars += "0123456789 ,-_&$#"
  return "".join(random.choice(valid_chars) for i in range(length))

def is_float(value):
    """must be a float"""
    return isinstance(value, float) or isinstance(value, int) or isinstance(value, np.float64), float(value)

def _sub_patterns(patterns, text):
    """
    Apply re.sub to bunch of (pattern, repl)
    """
    for pattern, repl in patterns:
        text = re.sub(pattern, repl, text)
    return text

def on_source_directory_chooser_clicked(self):
        """Autoconnect slot activated when tbSourceDir is clicked."""

        title = self.tr('Set the source directory for script and scenario')
        self.choose_directory(self.source_directory, title)

def from_json_list(cls, api_client, data):
        """Convert a list of JSON values to a list of models
        """
        return [cls.from_json(api_client, item) for item in data]

def clean_all(self, args):
        """Delete all build components; the package cache, package builds,
        bootstrap builds and distributions."""
        self.clean_dists(args)
        self.clean_builds(args)
        self.clean_download_cache(args)

def _merge_maps(m1, m2):
    """merge two Mapping objects, keeping the type of the first mapping"""
    return type(m1)(chain(m1.items(), m2.items()))

def strip(notebook):
    """Remove outputs from a notebook."""
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None

def find_whole_word(w):
    """
    Scan through string looking for a location where this word produces a match,
    and return a corresponding MatchObject instance.
    Return None if no position in the string matches the pattern;
    note that this is different from finding a zero-length match at some point in the string.
    """
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

def __delitem__(self, resource):
        """Remove resource instance from internal cache"""
        self.__caches[type(resource)].pop(resource.get_cache_internal_key(), None)

def autozoom(self, n=None):
        """
        Auto-scales the axes to fit all the data in plot index n. If n == None,
        auto-scale everyone.
        """
        if n==None:
            for p in self.plot_widgets: p.autoRange()
        else:        self.plot_widgets[n].autoRange()

        return self

def invalidate_cache(cpu, address, size):
        """ remove decoded instruction from instruction cache """
        cache = cpu.instruction_cache
        for offset in range(size):
            if address + offset in cache:
                del cache[address + offset]

def color_to_hex(color):
    """Convert matplotlib color code to hex color code"""
    if color is None or colorConverter.to_rgba(color)[3] == 0:
        return 'none'
    else:
        rgb = colorConverter.to_rgb(color)
        return '#{0:02X}{1:02X}{2:02X}'.format(*(int(255 * c) for c in rgb))

def erase_lines(n=1):
    """ Erases n lines from the screen and moves the cursor up to follow
    """
    for _ in range(n):
        print(codes.cursor["up"], end="")
        print(codes.cursor["eol"], end="")

def horizontal_line(ax, scale, i, **kwargs):
    """
    Draws the i-th horizontal line parallel to the lower axis.

    Parameters
    ----------
    ax: Matplotlib AxesSubplot
        The subplot to draw on.
    scale: float, 1.0
        Simplex scale size.
    i: float
        The index of the line to draw
    kwargs: Dictionary
        Any kwargs to pass through to Matplotlib.
    """

    p1 = (0, i, scale - i)
    p2 = (scale - i, i, 0)
    line(ax, p1, p2, **kwargs)

def terminate(self):
        """Terminate all workers and threads."""
        for t in self._threads:
            t.quit()
        self._thread = []
        self._workers = []

def raise_figure_window(f=0):
    """
    Raises the supplied figure number or figure window.
    """
    if _fun.is_a_number(f): f = _pylab.figure(f)
    f.canvas.manager.window.raise_()

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

def clear_matplotlib_ticks(self, axis="both"):
        """Clears the default matplotlib ticks."""
        ax = self.get_axes()
        plotting.clear_matplotlib_ticks(ax=ax, axis=axis)

def set_ylimits(self, row, column, min=None, max=None):
        """Set y-axis limits of a subplot.

        :param row,column: specify the subplot.
        :param min: minimal axis value
        :param max: maximum axis value

        """
        subplot = self.get_subplot_at(row, column)
        subplot.set_ylimits(min, max)

def _norm(self, x):
    """Compute the safe norm."""
    return tf.sqrt(tf.reduce_sum(tf.square(x), keepdims=True, axis=-1) + 1e-7)

def show(self, imgs, ax=None):
        """ Visualize the persistence image

        """

        ax = ax or plt.gca()

        if type(imgs) is not list:
            imgs = [imgs]

        for i, img in enumerate(imgs):
            ax.imshow(img, cmap=plt.get_cmap("plasma"))
            ax.axis("off")

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

def downsample(array, k):
    """Choose k random elements of array."""
    length = array.shape[0]
    indices = random.sample(xrange(length), k)
    return array[indices]

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

def _digits(minval, maxval):
    """Digits needed to comforatbly display values in [minval, maxval]"""
    if minval == maxval:
        return 3
    else:
        return min(10, max(2, int(1 + abs(np.log10(maxval - minval)))))

def compare(a, b):
    """
     Compare items in 2 arrays. Returns sum(abs(a(i)-b(i)))
    """
    s=0
    for i in range(len(a)):
        s=s+abs(a[i]-b[i])
    return s

def compare(left, right):
    """
    yields EVENT,ENTRY pairs describing the differences between left
    and right, which are filenames for a pair of zip files
    """

    with open_zip(left) as l:
        with open_zip(right) as r:
            return compare_zips(l, r)

def coverage():
    """Run coverage tests."""
    # Note: coverage options are controlled by .coveragerc file
    install()
    test_setup()
    sh("%s -m coverage run %s" % (PYTHON, TEST_SCRIPT))
    sh("%s -m coverage report" % PYTHON)
    sh("%s -m coverage html" % PYTHON)
    sh("%s -m webbrowser -t htmlcov/index.html" % PYTHON)

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

def cpp_prog_builder(build_context, target):
    """Build a C++ binary executable"""
    yprint(build_context.conf, 'Build CppProg', target)
    workspace_dir = build_context.get_workspace('CppProg', target.name)
    build_cpp(build_context, target, target.compiler_config, workspace_dir)

def dictmerge(x, y):
    """
    merge two dictionaries
    """
    z = x.copy()
    z.update(y)
    return z

def advance_one_line(self):
    """Advances to next line."""

    current_line = self._current_token.line_number
    while current_line == self._current_token.line_number:
      self._current_token = ConfigParser.Token(*next(self._token_generator))

def merge(self, other):
        """ Merge another stats. """
        Stats.merge(self, other)
        self.changes += other.changes

def display_len(text):
    """
    Get the display length of a string. This can differ from the character
    length if the string contains wide characters.
    """
    text = unicodedata.normalize('NFD', text)
    return sum(char_width(char) for char in text)

def Proxy(f):
  """A helper to create a proxy method in a class."""

  def Wrapped(self, *args):
    return getattr(self, f)(*args)

  return Wrapped

def _message_to_string(message, data=None):
    """ Gives a string representation of a PB2 message. """
    if data is None:
        data = _json_from_message(message)

    return "Message {} from {} to {}: {}".format(
        message.namespace, message.source_id, message.destination_id, data)

def fn_min(self, a, axis=None):
        """
        Return the minimum of an array, ignoring any NaNs.

        :param a: The array.
        :return: The minimum value of the array.
        """

        return numpy.nanmin(self._to_ndarray(a), axis=axis)

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

def min_values(args):
    """ Return possible range for min function. """
    return Interval(min(x.low for x in args), min(x.high for x in args))

def post_process(self):
        """ Apply last 2D transforms"""
        self.image.putdata(self.pixels)
        self.image = self.image.transpose(Image.ROTATE_90)

def makedirs(path, mode=0o777, exist_ok=False):
    """A wrapper of os.makedirs()."""
    os.makedirs(path, mode, exist_ok)

def autoconvert(string):
    """Try to convert variables into datatypes."""
    for fn in (boolify, int, float):
        try:
            return fn(string)
        except ValueError:
            pass
    return string

def _from_dict(cls, _dict):
        """Initialize a ListCollectionsResponse object from a json dictionary."""
        args = {}
        if 'collections' in _dict:
            args['collections'] = [
                Collection._from_dict(x) for x in (_dict.get('collections'))
            ]
        return cls(**args)

def most_common(items):
    """
    Wanted functionality from Counters (new in Python 2.7).
    """
    counts = {}
    for i in items:
        counts.setdefault(i, 0)
        counts[i] += 1
    return max(six.iteritems(counts), key=operator.itemgetter(1))

def find_one(cls, *args, **kwargs):
        """Run a find_one on this model's collection.  The arguments to
        ``Model.find_one`` are the same as to ``pymongo.Collection.find_one``."""
        database, collection = cls._collection_key.split('.')
        return current()[database][collection].find_one(*args, **kwargs)

def indentsize(line):
    """Return the indent size, in spaces, at the start of a line of text."""
    expline = string.expandtabs(line)
    return len(expline) - len(string.lstrip(expline))

def mostCommonItem(lst):
    """Choose the most common item from the list, or the first item if all
    items are unique."""
    # This elegant solution from: http://stackoverflow.com/a/1518632/1760218
    lst = [l for l in lst if l]
    if lst:
        return max(set(lst), key=lst.count)
    else:
        return None

def make_env_key(app_name, key):
    """Creates an environment key-equivalent for the given key"""
    key = key.replace('-', '_').replace(' ', '_')
    return str("_".join((x.upper() for x in (app_name, key))))

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

def touch():
    """Create new bucket."""
    from .models import Bucket
    bucket = Bucket.create()
    db.session.commit()
    click.secho(str(bucket), fg='green')

def align_file_position(f, size):
    """ Align the position in the file to the next block of specified size """
    align = (size - 1) - (f.tell() % size)
    f.seek(align, 1)

def format_header_cell(val):
    """
    Formats given header column. This involves changing '_Px_' to '(', '_xP_' to ')' and
    all other '_' to spaces.
    """
    return re.sub('_', ' ', re.sub(r'(_Px_)', '(', re.sub(r'(_xP_)', ')', str(val) )))

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

def copy(self):
        """Return a shallow copy of the sorted dictionary."""
        return self.__class__(self._key, self._load, self._iteritems())

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

def list_string_to_dict(string):
    """Inputs ``['a', 'b', 'c']``, returns ``{'a': 0, 'b': 1, 'c': 2}``."""
    dictionary = {}
    for idx, c in enumerate(string):
        dictionary.update({c: idx})
    return dictionary

def _match_space_at_line(line):
    """Return a re.match object if an empty comment was found on line."""
    regex = re.compile(r"^{0}$".format(_MDL_COMMENT))
    return regex.match(line)

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

def _comment(string):
    """return string as a comment"""
    lines = [line.strip() for line in string.splitlines()]
    return "# " + ("%s# " % linesep).join(lines)

def count_generator(generator, memory_efficient=True):
    """Count number of item in generator.

    memory_efficient=True, 3 times slower, but memory_efficient.
    memory_efficient=False, faster, but cost more memory.
    """
    if memory_efficient:
        counter = 0
        for _ in generator:
            counter += 1
        return counter
    else:
        return len(list(generator))

def replace_sys_args(new_args):
    """Temporarily replace sys.argv with current arguments

    Restores sys.argv upon exit of the context manager.
    """
    # Replace sys.argv arguments
    # for module import
    old_args = sys.argv
    sys.argv = new_args
    try:
        yield
    finally:
        sys.argv = old_args

def chunk_list(l, n):
    """Return `n` size lists from a given list `l`"""
    return [l[i:i + n] for i in range(0, len(l), n)]

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

def generate_matrices(dim = 40):
  """
  Generates the matrices that positive and negative samples are multiplied
  with.  The matrix for positive samples is randomly drawn from a uniform
  distribution, with elements in [-1, 1].  The matrix for negative examples
  is the sum of the positive matrix with a matrix drawn from a normal
  distribution with mean 0 variance 1.
  """
  positive = numpy.random.uniform(-1, 1, (dim, dim))
  negative = positive + numpy.random.normal(0, 1, (dim, dim))
  return positive, negative

def qr(self,text):
        """ Print QR Code for the provided string """
        qr_code = qrcode.QRCode(version=4, box_size=4, border=1)
        qr_code.add_data(text)
        qr_code.make(fit=True)
        qr_img = qr_code.make_image()
        im = qr_img._img.convert("RGB")
        # Convert the RGB image in printable image
        self._convert_image(im)

def to_unicode_repr( _letter ):
    """ helpful in situations where browser/app may recognize Unicode encoding
        in the \u0b8e type syntax but not actual unicode glyph/code-point"""
    # Python 2-3 compatible
    return u"u'"+ u"".join( [ u"\\u%04x"%ord(l) for l in _letter ] ) + u"'"

def compute(args):
    x, y, params = args
    """Callable function for the multiprocessing pool."""
    return x, y, mandelbrot(x, y, params)

def clear_global(self):
        """Clear only any cached global data.

        """
        vname = self.varname
        logger.debug(f'global clearning {vname}')
        if vname in globals():
            logger.debug('removing global instance var: {}'.format(vname))
            del globals()[vname]

def get(self):
        """retrieve a result from the pool

        if nothing is already completed when this method is called, it will
        block until something comes back

        if the pool's function exited via exception, that will come back as
        a result here as well, but will be re-raised in :meth:`get`.

        .. note::
            if there is nothing in the pool's output queue when this method is
            called, it will block until something is ready

        :returns:
            a return value from one of the function's invocations if it exited
            normally

        :raises:
            :class:`PoolClosed` if the pool was closed before a result could be
            produced for thie call

        :raises: any exception that was raised inside the worker function
        """
        if self.closed:
            raise PoolClosed()

        while self._getcount not in self._cache:
            counter, result = self.outq.get()
            self._cache[counter] = result

        result, succeeded = self._cache.pop(self._getcount)
        self._getcount += 1

        if not succeeded:
            klass, exc, tb = result
            raise klass, exc, tb
        return result

def _release(self):
        """Destroy self since closures cannot be called again."""
        del self.funcs
        del self.variables
        del self.variable_values
        del self.satisfied

def compute_capture(args):
    x, y, w, h, params = args
    """Callable function for the multiprocessing pool."""
    return x, y, mandelbrot_capture(x, y, w, h, params)

def __delitem__(self, key):
        """Remove a variable from this dataset.
        """
        del self._variables[key]
        self._coord_names.discard(key)

def get(self):
        """Get the highest priority Processing Block from the queue."""
        with self._mutex:
            entry = self._queue.pop()
            del self._block_map[entry[2]]
            return entry[2]

def remove_columns(self, data, columns):
        """ This method removes columns in data

        :param data: original Pandas dataframe
        :param columns: list of columns to remove
        :type data: pandas.DataFrame
        :type columns: list of strings

        :returns: Pandas dataframe with removed columns
        :rtype: pandas.DataFrame
        """

        for column in columns:
            if column in data.columns:
                data = data.drop(column, axis=1)

        return data

def _synced(method, self, args, kwargs):
    """Underlying synchronized wrapper."""
    with self._lock:
        return method(*args, **kwargs)

def _delete_local(self, filename):
        """Deletes the specified file from the local filesystem."""

        if os.path.exists(filename):
            os.remove(filename)

def get_table_names(connection):
	"""
	Return a list of the table names in the database.
	"""
	cursor = connection.cursor()
	cursor.execute("SELECT name FROM sqlite_master WHERE type == 'table'")
	return [name for (name,) in cursor]

def remove_elements(target, indices):
    """Remove multiple elements from a list and return result.
    This implementation is faster than the alternative below.
    Also note the creation of a new list to avoid altering the
    original. We don't have any current use for the original
    intact list, but may in the future..."""

    copied = list(target)

    for index in reversed(indices):
        del copied[index]
    return copied

def get_window(self): 
        """
        Returns the object's parent window. Returns None if no window found.
        """
        x = self
        while not x._parent == None and \
              not isinstance(x._parent, Window): 
                  x = x._parent
        return x._parent

def remove_bad(string):
    """
    remove problem characters from string
    """
    remove = [':', ',', '(', ')', ' ', '|', ';', '\'']
    for c in remove:
        string = string.replace(c, '_')
    return string

def restore_scrollbar_position(self):
        """Restoring scrollbar position after main window is visible"""
        scrollbar_pos = self.get_option('scrollbar_position', None)
        if scrollbar_pos is not None:
            self.explorer.treewidget.set_scrollbar_position(scrollbar_pos)

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

def index_nearest(value, array):
    """
    expects a _n.array
    returns the global minimum of (value-array)^2
    """

    a = (array-value)**2
    return index(a.min(), a)

def deprecate(func):
  """ A deprecation warning emmiter as a decorator. """
  @wraps(func)
  def wrapper(*args, **kwargs):
    warn("Deprecated, this will be removed in the future", DeprecationWarning)
    return func(*args, **kwargs)
  wrapper.__doc__ = "Deprecated.\n" + (wrapper.__doc__ or "")
  return wrapper

def remove_node(self, node):
        """ Remove a node from this network. """
        if _debug: Network._debug("remove_node %r", node)

        self.nodes.remove(node)
        node.lan = None

def deprecate(func):
  """ A deprecation warning emmiter as a decorator. """
  @wraps(func)
  def wrapper(*args, **kwargs):
    warn("Deprecated, this will be removed in the future", DeprecationWarning)
    return func(*args, **kwargs)
  wrapper.__doc__ = "Deprecated.\n" + (wrapper.__doc__ or "")
  return wrapper

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

def is_admin(self):
        """Is the user a system administrator"""
        return self.role == self.roles.administrator.value and self.state == State.approved

def start_connect(self):
    """Tries to connect to the Heron Server

    ``loop()`` method needs to be called after this.
    """
    Log.debug("In start_connect() of %s" % self._get_classname())
    # TODO: specify buffer size, exception handling
    self.create_socket(socket.AF_INET, socket.SOCK_STREAM)

    # when ready, handle_connect is called
    self._connecting = True
    self.connect(self.endpoint)

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

def isString(s):
    """Convenience method that works with all 2.x versions of Python
    to determine whether or not something is stringlike."""
    try:
        return isinstance(s, unicode) or isinstance(s, basestring)
    except NameError:
        return isinstance(s, str)

def lognorm(x, mu, sigma=1.0):
    """ Log-normal function from scipy """
    return stats.lognorm(sigma, scale=mu).pdf(x)

def _api_type(self, value):
        """
        Returns the API type of the given value based on its python type.

        """
        if isinstance(value, six.string_types):
            return 'string'
        elif isinstance(value, six.integer_types):
            return 'integer'
        elif type(value) is datetime.datetime:
            return 'date'

def denorm(self,arr):
        """Reverse the normalization done to a batch of images.

        Arguments:
            arr: of shape/size (N,3,sz,sz)
        """
        if type(arr) is not np.ndarray: arr = to_np(arr)
        if len(arr.shape)==3: arr = arr[None]
        return self.transform.denorm(np.rollaxis(arr,1,4))

def is_seq(obj):
    """
    Check if an object is a sequence.
    """
    return (not is_str(obj) and not is_dict(obj) and
            (hasattr(obj, "__getitem__") or hasattr(obj, "__iter__")))

def isTestCaseDisabled(test_case_class, method_name):
    """
    I check to see if a method on a TestCase has been disabled via nose's
    convention for disabling a TestCase.  This makes it so that users can
    mix nose's parameterized tests with green as a runner.
    """
    test_method = getattr(test_case_class, method_name)
    return getattr(test_method, "__test__", 'not nose') is False

def _histplot_bins(column, bins=100):
    """Helper to get bins for histplot."""
    col_min = np.min(column)
    col_max = np.max(column)
    return range(col_min, col_max + 2, max((col_max - col_min) // bins, 1))

def prepare(doc):
    """Sets the caption_found and plot_found variables to False."""
    doc.caption_found = False
    doc.plot_found = False
    doc.listings_counter = 0

def path_for_import(name):
    """
    Returns the directory path for the given package or module.
    """
    return os.path.dirname(os.path.abspath(import_module(name).__file__))

def tokenize(string):
    """Match and yield all the tokens of the input string."""
    for match in TOKENS_REGEX.finditer(string):
        yield Token(match.lastgroup, match.group().strip(), match.span())

def A(*a):
    """convert iterable object into numpy array"""
    return np.array(a[0]) if len(a)==1 else [np.array(o) for o in a]

def chunked(l, n):
    """Chunk one big list into few small lists."""
    return [l[i:i + n] for i in range(0, len(l), n)]

def contains_all(self, array):
        """Test if `array` is an array of real numbers."""
        dtype = getattr(array, 'dtype', None)
        if dtype is None:
            dtype = np.result_type(*array)
        return is_real_dtype(dtype)

def quit(self):
        """ Exit the program due to user's choices.
        """
        self.script.LOG.warn("Abort due to user choice!")
        sys.exit(self.QUIT_RC)

def recarray(self):
        """Returns data as :class:`numpy.recarray`."""
        return numpy.rec.fromrecords(self.records, names=self.names)

def print_images(self, *printable_images):
        """
        This method allows printing several images in one shot. This is useful if the client code does not want the
        printer to make pause during printing
        """
        printable_image = reduce(lambda x, y: x.append(y), list(printable_images))
        self.print_image(printable_image)

def ma(self):
        """Represent data as a masked array.

        The array is returned with column-first indexing, i.e. for a data file with
        columns X Y1 Y2 Y3 ... the array a will be a[0] = X, a[1] = Y1, ... .

        inf and nan are filtered via :func:`numpy.isfinite`.
        """
        a = self.array
        return numpy.ma.MaskedArray(a, mask=numpy.logical_not(numpy.isfinite(a)))

def _divide(self, x1, x2, out):
        """Raw pointwise multiplication of two elements."""
        self.tspace._divide(x1.tensor, x2.tensor, out.tensor)

def _to_json(self):
        """ Gets a dict of this object's properties so that it can be used to send a dump to the client """
        return dict(( (k, v) for k, v in self.__dict__.iteritems() if k != 'server'))

def _user_yes_no_query(self, question):
        """ Helper asking if the user want to download the file

        Note:
            Dowloading huge file can take a while

        """
        sys.stdout.write('%s [y/n]\n' % question)
        while True:
            try:
                return strtobool(raw_input().lower())
            except ValueError:
                sys.stdout.write('Please respond with \'y\' or \'n\'.\n')

def json_datetime_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, date)):
        serial = obj.isoformat()
        return serial

    if ObjectId is not None and isinstance(obj, ObjectId):
        # TODO: try to use bson.json_util instead
        return str(obj)

    raise TypeError("Type not serializable")

def downcaseTokens(s,l,t):
    """Helper parse action to convert tokens to lower case."""
    return [ tt.lower() for tt in map(_ustr,t) ]

def to_one_hot(dataY):
    """Convert the vector of labels dataY into one-hot encoding.

    :param dataY: vector of labels
    :return: one-hot encoded labels
    """
    nc = 1 + np.max(dataY)
    onehot = [np.zeros(nc, dtype=np.int8) for _ in dataY]
    for i, j in enumerate(dataY):
        onehot[i][j] = 1
    return onehot

def matrix_at_check(self, original, loc, tokens):
        """Check for Python 3.5 matrix multiplication."""
        return self.check_py("35", "matrix multiplication", original, loc, tokens)

def is_equal_strings_ignore_case(first, second):
    """The function compares strings ignoring case"""
    if first and second:
        return first.upper() == second.upper()
    else:
        return not (first or second)

def _fix_up(self, cls, code_name):
    """Internal helper called to tell the property its name.

    This is called by _fix_up_properties() which is called by
    MetaModel when finishing the construction of a Model subclass.
    The name passed in is the name of the class attribute to which the
    Property is assigned (a.k.a. the code name).  Note that this means
    that each Property instance must be assigned to (at most) one
    class attribute.  E.g. to declare three strings, you must call
    StringProperty() three times, you cannot write

      foo = bar = baz = StringProperty()
    """
    self._code_name = code_name
    if self._name is None:
      self._name = code_name

def dot_v3(v, w):
    """Return the dotproduct of two vectors."""

    return sum([x * y for x, y in zip(v, w)])

def dict_update_newkeys(dict_, dict2):
    """ Like dict.update, but does not overwrite items """
    for key, val in six.iteritems(dict2):
        if key not in dict_:
            dict_[key] = val

def download_json(local_filename, url, clobber=False):
    """Download the given JSON file, and pretty-print before we output it."""
    with open(local_filename, 'w') as json_file:
        json_file.write(json.dumps(requests.get(url).json(), sort_keys=True, indent=2, separators=(',', ': ')))

def read_img(path):
    """ Reads image specified by path into numpy.ndarray"""
    img = cv2.resize(cv2.imread(path, 0), (80, 30)).astype(np.float32) / 255
    img = np.expand_dims(img.transpose(1, 0), 0)
    return img

def download_json(local_filename, url, clobber=False):
    """Download the given JSON file, and pretty-print before we output it."""
    with open(local_filename, 'w') as json_file:
        json_file.write(json.dumps(requests.get(url).json(), sort_keys=True, indent=2, separators=(',', ': ')))

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

def _download_py3(link, path, __hdr__):
    """Download a file from a link in Python 3."""
    try:
        req = urllib.request.Request(link, headers=__hdr__)
        u = urllib.request.urlopen(req)
    except Exception as e:
        raise Exception(' Download failed with the error:\n{}'.format(e))

    with open(path, 'wb') as outf:
        for l in u:
            outf.write(l)
    u.close()

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

def to_dotfile(self):
        """ Writes a DOT graphviz file of the domain structure, and returns the filename"""
        domain = self.get_domain()
        filename = "%s.dot" % (self.__class__.__name__)
        nx.write_dot(domain, filename)
        return filename

def _openpyxl_read_xl(xl_path: str):
    """ Use openpyxl to read an Excel file. """
    try:
        wb = load_workbook(filename=xl_path, read_only=True)
    except:
        raise
    else:
        return wb

def _drop_str_columns(df):
    """

    Parameters
    ----------
    df : DataFrame

    Returns
    -------

    """
    str_columns = filter(lambda pair: pair[1].char == 'S', df._gather_dtypes().items())
    str_column_names = list(map(lambda pair: pair[0], str_columns))

    return df.drop(str_column_names)

def get_order(self, codes):
        """Return evidence codes in order shown in code2name."""
        return sorted(codes, key=lambda e: [self.ev2idx.get(e)])

def upcaseTokens(s,l,t):
    """Helper parse action to convert tokens to upper case."""
    return [ tt.upper() for tt in map(_ustr,t) ]

def C_dict2array(C):
    """Convert an OrderedDict containing C values to a 1D array."""
    return np.hstack([np.asarray(C[k]).ravel() for k in C_keys])

def remove_duplicates(lst):
    """
    Emulate what a Python ``set()`` does, but keeping the element's order.
    """
    dset = set()
    return [l for l in lst if l not in dset and not dset.add(l)]

def normalize_path(path):
    """
    Convert a path to its canonical, case-normalized, absolute version.

    """
    return os.path.normcase(os.path.realpath(os.path.expanduser(path)))

def haversine(x):
    """Return the haversine of an angle

    haversine(x) = sin(x/2)**2, where x is an angle in radians
    """
    y = .5*x
    y = np.sin(y)
    return y*y

def __get_float(section, name):
    """Get the forecasted float from json section."""
    try:
        return float(section[name])
    except (ValueError, TypeError, KeyError):
        return float(0)

def min_values(args):
    """ Return possible range for min function. """
    return Interval(min(x.low for x in args), min(x.high for x in args))

def write_color(string, name, style='normal', when='auto'):
    """ Write the given colored string to standard out. """
    write(color(string, name, style, when))

def close(self):
        """Close port."""
        os.close(self.in_d)
        os.close(self.out_d)

def resources(self):
        """Retrieve contents of each page of PDF"""
        return [self.pdf.getPage(i) for i in range(self.pdf.getNumPages())]

def extract_keywords_from_text(self, text):
        """Method to extract keywords from the text provided.

        :param text: Text to extract keywords from, provided as a string.
        """
        sentences = nltk.tokenize.sent_tokenize(text)
        self.extract_keywords_from_sentences(sentences)

def deserialize_date(string):
    """
    Deserializes string to date.

    :param string: str.
    :type string: str
    :return: date.
    :rtype: date
    """
    try:
        from dateutil.parser import parse
        return parse(string).date()
    except ImportError:
        return string

def get_soup(page=''):
    """
    Returns a bs4 object of the page requested
    """
    content = requests.get('%s/%s' % (BASE_URL, page)).text
    return BeautifulSoup(content)

def parse_date(s):
    """
    Parse a date using dateutil.parser.parse if available,
    falling back to datetime.datetime.strptime if not
    """
    if isinstance(s, (datetime.datetime, datetime.date)):
        return s
    try:
        from dateutil.parser import parse
    except ImportError:
        parse = lambda d: datetime.datetime.strptime(d, "%Y-%m-%d")
    return parse(s)

def clean_dataframe(df):
    """Fill NaNs with the previous value, the next value or if all are NaN then 1.0"""
    df = df.fillna(method='ffill')
    df = df.fillna(0.0)
    return df

def _synced(method, self, args, kwargs):
    """Underlying synchronized wrapper."""
    with self._lock:
        return method(*args, **kwargs)

def fsliceafter(astr, sub):
    """Return the slice after at sub in string astr"""
    findex = astr.find(sub)
    return astr[findex + len(sub):]

def map_wrap(f):
    """Wrap standard function to easily pass into 'map' processing.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper

def flatten(l, types=(list, float)):
    """
    Flat nested list of lists into a single list.
    """
    l = [item if isinstance(item, types) else [item] for item in l]
    return [item for sublist in l for item in sublist]

def list_formatter(handler, item, value):
    """Format list."""
    return u', '.join(str(v) for v in value)

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

def debug(self, text):
		""" Ajout d'un message de log de type DEBUG """
		self.logger.debug("{}{}".format(self.message_prefix, text))

def safe_int_conv(number):
    """Safely convert a single number to integer."""
    try:
        return int(np.array(number).astype(int, casting='safe'))
    except TypeError:
        raise ValueError('cannot safely convert {} to integer'.format(number))

def quote(self, s):
        """Return a shell-escaped version of the string s."""

        if six.PY2:
            from pipes import quote
        else:
            from shlex import quote

        return quote(s)

def safe_int_conv(number):
    """Safely convert a single number to integer."""
    try:
        return int(np.array(number).astype(int, casting='safe'))
    except TypeError:
        raise ValueError('cannot safely convert {} to integer'.format(number))

def create_path(path):
    """Creates a absolute path in the file system.

    :param path: The path to be created
    """
    import os
    if not os.path.exists(path):
        os.makedirs(path)

def translate_fourier(image, dx):
    """ Translate an image in fourier-space with plane waves """
    N = image.shape[0]

    f = 2*np.pi*np.fft.fftfreq(N)
    kx,ky,kz = np.meshgrid(*(f,)*3, indexing='ij')
    kv = np.array([kx,ky,kz]).T

    q = np.fft.fftn(image)*np.exp(-1.j*(kv*dx).sum(axis=-1)).T
    return np.real(np.fft.ifftn(q))

def perform_pca(A):
    """
    Computes eigenvalues and eigenvectors of covariance matrix of A.
    The rows of a correspond to observations, the columns to variables.
    """
    # First subtract the mean
    M = (A-numpy.mean(A.T, axis=1)).T
    # Get eigenvectors and values of covariance matrix
    return numpy.linalg.eig(numpy.cov(M))

def main_func(args=None):
    """Main funcion when executing this module as script

    :param args: commandline arguments
    :type args: list
    :returns: None
    :rtype: None
    :raises: None
    """
    # we have to initialize a gui even if we dont need one right now.
    # as soon as you call maya.standalone.initialize(), a QApplication
    # with type Tty is created. This is the type for conosle apps.
    # Because i have not found a way to replace that, we just init the gui.
    guimain.init_gui()

    main.init()
    launcher = Launcher()
    parsed, unknown = launcher.parse_args(args)
    parsed.func(parsed, unknown)

def debug_on_error(type, value, tb):
    """Code due to Thomas Heller - published in Python Cookbook (O'Reilley)"""
    traceback.print_exc(type, value, tb)
    print()
    pdb.pm()

def from_rectangle(box):
        """ Create a vector randomly within the given rectangle. """
        x = box.left + box.width * random.uniform(0, 1)
        y = box.bottom + box.height * random.uniform(0, 1)
        return Vector(x, y)

def set_trace():
    """Start a Pdb instance at the calling frame, with stdout routed to sys.__stdout__."""
    # https://github.com/nose-devs/nose/blob/master/nose/tools/nontrivial.py
    pdb.Pdb(stdout=sys.__stdout__).set_trace(sys._getframe().f_back)

def _remove_duplicates(objects):
    """Removes duplicate objects.

    http://www.peterbe.com/plog/uniqifiers-benchmark.
    """
    seen, uniq = set(), []
    for obj in objects:
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        uniq.append(obj)
    return uniq

def set_trace():
    """Start a Pdb instance at the calling frame, with stdout routed to sys.__stdout__."""
    # https://github.com/nose-devs/nose/blob/master/nose/tools/nontrivial.py
    pdb.Pdb(stdout=sys.__stdout__).set_trace(sys._getframe().f_back)

def to_camel_case(snake_case_string):
    """
    Convert a string from snake case to camel case. For example, "some_var" would become "someVar".

    :param snake_case_string: Snake-cased string to convert to camel case.
    :returns: Camel-cased version of snake_case_string.
    """
    parts = snake_case_string.lstrip('_').split('_')
    return parts[0] + ''.join([i.title() for i in parts[1:]])

def dimensions(self):
        """Get width and height of a PDF"""
        size = self.pdf.getPage(0).mediaBox
        return {'w': float(size[2]), 'h': float(size[3])}

def do_history(self, line):
        """history Display a list of commands that have been entered."""
        self._split_args(line, 0, 0)
        for idx, item in enumerate(self._history):
            d1_cli.impl.util.print_info("{0: 3d} {1}".format(idx, item))

def quote(s, unsafe='/'):
    """Pass in a dictionary that has unsafe characters as the keys, and the percent
    encoded value as the value."""
    res = s.replace('%', '%25')
    for c in unsafe:
        res = res.replace(c, '%' + (hex(ord(c)).upper())[2:])
    return res

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

def visit_BoolOp(self, node):
        """ Return type may come from any boolop operand. """
        return sum((self.visit(value) for value in node.values), [])

def linearRegressionAnalysis(series):
    """
    Returns factor and offset of linear regression function by least
    squares method.

    """
    n = safeLen(series)
    sumI = sum([i for i, v in enumerate(series) if v is not None])
    sumV = sum([v for i, v in enumerate(series) if v is not None])
    sumII = sum([i * i for i, v in enumerate(series) if v is not None])
    sumIV = sum([i * v for i, v in enumerate(series) if v is not None])
    denominator = float(n * sumII - sumI * sumI)
    if denominator == 0:
        return None
    else:
        factor = (n * sumIV - sumI * sumV) / denominator / series.step
        offset = sumII * sumV - sumIV * sumI
        offset = offset / denominator - factor * series.start
        return factor, offset

def from_bytes(cls, b):
		"""Create :class:`PNG` from raw bytes.
		
		:arg bytes b: The raw bytes of the PNG file.
		:rtype: :class:`PNG`
		"""
		im = cls()
		im.chunks = list(parse_chunks(b))
		im.init()
		return im

def searchlast(self,n=10):
        """Return the last n results (or possibly less if not found). Note that the last results are not necessarily the best ones! Depending on the search type."""            
        solutions = deque([], n)
        for solution in self:
            solutions.append(solution)
        return solutions

def less_strict_bool(x):
    """Idempotent and None-safe version of strict_bool."""
    if x is None:
        return False
    elif x is True or x is False:
        return x
    else:
        return strict_bool(x)

def _get_token(self, oauth_request, token_type='access'):
        """Try to find the token for the provided request token key."""
        token_field = oauth_request.get_parameter('oauth_token')
        token = self.data_store.lookup_token(token_type, token_field)
        if not token:
            raise OAuthError('Invalid %s token: %s' % (token_type, token_field))
        return token

def pause(self):
        """Pause the music"""
        mixer.music.pause()
        self.pause_time = self.get_time()
        self.paused = True

def fn_min(self, a, axis=None):
        """
        Return the minimum of an array, ignoring any NaNs.

        :param a: The array.
        :return: The minimum value of the array.
        """

        return numpy.nanmin(self._to_ndarray(a), axis=axis)

def draw_image(self, ax, image):
        """Process a matplotlib image object and call renderer.draw_image"""
        self.renderer.draw_image(imdata=utils.image_to_base64(image),
                                 extent=image.get_extent(),
                                 coordinates="data",
                                 style={"alpha": image.get_alpha(),
                                        "zorder": image.get_zorder()},
                                 mplobj=image)

def average(iterator):
    """Iterative mean."""
    count = 0
    total = 0
    for num in iterator:
        count += 1
        total += num
    return float(total)/count

def cart2pol(x, y):
    """Cartesian to Polar coordinates conversion."""
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho

def get_column_keys_and_names(table):
    """
    Return a generator of tuples k, c such that k is the name of the python attribute for
    the column and c is the name of the column in the sql table.
    """
    ins = inspect(table)
    return ((k, c.name) for k, c in ins.mapper.c.items())

def asyncStarCmap(asyncCallable, iterable):
    """itertools.starmap for deferred callables using cooperative multitasking
    """
    results = []
    yield coopStar(asyncCallable, results.append, iterable)
    returnValue(results)

def get_obj_cols(df):
    """
    Returns names of 'object' columns in the DataFrame.
    """
    obj_cols = []
    for idx, dt in enumerate(df.dtypes):
        if dt == 'object' or is_category(dt):
            obj_cols.append(df.columns.values[idx])

    return obj_cols

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

def get_obj_cols(df):
    """
    Returns names of 'object' columns in the DataFrame.
    """
    obj_cols = []
    for idx, dt in enumerate(df.dtypes):
        if dt == 'object' or is_category(dt):
            obj_cols.append(df.columns.values[idx])

    return obj_cols

def paint_cube(self, x, y):
        """
        Paints a cube at a certain position a color.

        Parameters
        ----------
        x: int
            Horizontal position of the upper left corner of the cube.
        y: int
            Vertical position of the upper left corner of the cube.

        """
        # get the color
        color = self.next_color()
        # calculate the position
        cube_pos = [x, y, x + self.cube_size, y + self.cube_size]
        # draw the cube
        draw = ImageDraw.Draw(im=self.image)
        draw.rectangle(xy=cube_pos, fill=color)

def onchange(self, value):
        """Called when a new DropDownItem gets selected.
        """
        log.debug('combo box. selected %s' % value)
        self.select_by_value(value)
        return (value, )

def p_postfix_expr(self, p):
        """postfix_expr : left_hand_side_expr
                        | left_hand_side_expr PLUSPLUS
                        | left_hand_side_expr MINUSMINUS
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = ast.UnaryOp(op=p[2], value=p[1], postfix=True)

def phantomjs_retrieve(url, data=None):
    """Retrieve the given URL using PhantomJS.
    PhantomJS will evaluate all scripts and return the HTML after body.onload.
    
    url  - The page URL to retrieve
    data - The form data. TODO: Currently ignored.

    Returns a status code (e.g. 200) and the HTML as a unicode string.
    """
    range_limit()
    print "pGET", url
    process = subprocess.Popen(['phantomjs', PHANTOM_SCRIPT, url], stdout=subprocess.PIPE)
    out = process.communicate()
    process.wait()
    response = out[0].decode('utf-8', 'ignore')
    status = response[:2]
    body = response[3:] # After the 'ok ' part.
    if status == 'ok':
        return 200, body
    else:
        return 404, body

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

def get_list_dimensions(_list):
    """
    Takes a nested list and returns the size of each dimension followed
    by the element type in the list
    """
    if isinstance(_list, list) or isinstance(_list, tuple):
        return [len(_list)] + get_list_dimensions(_list[0])
    return []

def getTypeStr(_type):
  r"""Gets the string representation of the given type.
  """
  if isinstance(_type, CustomType):
    return str(_type)

  if hasattr(_type, '__name__'):
    return _type.__name__

  return ''

def iget_list_column_slice(list_, start=None, stop=None, stride=None):
    """ iterator version of get_list_column """
    if isinstance(start, slice):
        slice_ = start
    else:
        slice_ = slice(start, stop, stride)
    return (row[slice_] for row in list_)

def pformat(o, indent=1, width=80, depth=None):
    """Format a Python o into a pretty-printed representation."""
    return PrettyPrinter(indent=indent, width=width, depth=depth).pformat(o)

def this_quarter():
        """ Return start and end date of this quarter. """
        since = TODAY + delta(day=1)
        while since.month % 3 != 0:
            since -= delta(months=1)
        until = since + delta(months=3)
        return Date(since), Date(until)

def print_trace(self):
        """
        Prints stack trace for current exceptions chain.
        """
        traceback.print_exc()
        for tb in self.tracebacks:
            print tb,
        print ''

def help_for_command(command):
    """Get the help text (signature + docstring) for a command (function)."""
    help_text = pydoc.text.document(command)
    # remove backspaces
    return re.subn('.\\x08', '', help_text)[0]

def safe_exit(output):
    """exit without breaking pipes."""
    try:
        sys.stdout.write(output)
        sys.stdout.flush()
    except IOError:
        pass

def check_git():
    """Check if git command is available."""
    try:
        with open(os.devnull, "wb") as devnull:
            subprocess.check_call(["git", "--version"], stdout=devnull, stderr=devnull)
    except:
        raise RuntimeError("Please make sure git is installed and on your path.")

def py(self, output):
        """Output data as a nicely-formatted python data structure"""
        import pprint
        pprint.pprint(output, stream=self.outfile)

def get_column_keys_and_names(table):
    """
    Return a generator of tuples k, c such that k is the name of the python attribute for
    the column and c is the name of the column in the sql table.
    """
    ins = inspect(table)
    return ((k, c.name) for k, c in ins.mapper.c.items())

def pretty(obj, verbose=False, max_width=79, newline='\n'):
    """
    Pretty print the object's representation.
    """
    stream = StringIO()
    printer = RepresentationPrinter(stream, verbose, max_width, newline)
    printer.pretty(obj)
    printer.flush()
    return stream.getvalue()

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

def prnt(self):
        """
        Prints DB data representation of the object.
        """
        print("= = = =\n\n%s object key: \033[32m%s\033[0m" % (self.__class__.__name__, self.key))
        pprnt(self._data or self.clean_value())

def timestamp_to_microseconds(timestamp):
    """Convert a timestamp string into a microseconds value
    :param timestamp
    :return time in microseconds
    """
    timestamp_str = datetime.datetime.strptime(timestamp, ISO_DATETIME_REGEX)
    epoch_time_secs = calendar.timegm(timestamp_str.timetuple())
    epoch_time_mus = epoch_time_secs * 1e6 + timestamp_str.microsecond
    return epoch_time_mus

def stdout_display():
    """ Print results straight to stdout """
    if sys.version_info[0] == 2:
        yield SmartBuffer(sys.stdout)
    else:
        yield SmartBuffer(sys.stdout.buffer)

def mostCommonItem(lst):
    """Choose the most common item from the list, or the first item if all
    items are unique."""
    # This elegant solution from: http://stackoverflow.com/a/1518632/1760218
    lst = [l for l in lst if l]
    if lst:
        return max(set(lst), key=lst.count)
    else:
        return None

def start(self, timeout=None):
        """
        Startup of the node.
        :param join: optionally wait for the process to end (default : True)
        :return: None
        """

        assert super(PyrosBase, self).start(timeout=timeout)
        # Because we currently use this to setup connection
        return self.name

def filter_regex(names, regex):
    """
    Return a tuple of strings that match the regular expression pattern.
    """
    return tuple(name for name in names
                 if regex.search(name) is not None)

def value(self):
        """Value of property."""
        if self._prop.fget is None:
            raise AttributeError('Unable to read attribute')
        return self._prop.fget(self._obj)

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

def load(self):
        """Load proxy list from configured proxy source"""
        self._list = self._source.load()
        self._list_iter = itertools.cycle(self._list)

def _get_points(self):
        """
        Subclasses may override this method.
        """
        return tuple([self._getitem__points(i)
                     for i in range(self._len__points())])

def format_pylint_disables(error_names, tag=True):
    """
    Format a list of error_names into a 'pylint: disable=' line.
    """
    tag_str = "lint-amnesty, " if tag else ""
    if error_names:
        return u"  # {tag}pylint: disable={disabled}".format(
            disabled=", ".join(sorted(error_names)),
            tag=tag_str,
        )
    else:
        return ""

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

def lowstrip(term):
    """Convert to lowercase and strip spaces"""
    term = re.sub('\s+', ' ', term)
    term = term.lower()
    return term

def chmod(self, mode):
        """
        Change the mode (permissions) of this file.  The permissions are
        unix-style and identical to those used by python's C{os.chmod}
        function.

        @param mode: new permissions
        @type mode: int
        """
        self.sftp._log(DEBUG, 'chmod(%s, %r)' % (hexlify(self.handle), mode))
        attr = SFTPAttributes()
        attr.st_mode = mode
        self.sftp._request(CMD_FSETSTAT, self.handle, attr)

def region_from_segment(image, segment):
    """given a segment (rectangle) and an image, returns it's corresponding subimage"""
    x, y, w, h = segment
    return image[y:y + h, x:x + w]

def test(ctx, all=False, verbose=False):
    """Run the tests."""
    cmd = 'tox' if all else 'py.test'
    if verbose:
        cmd += ' -v'
    return ctx.run(cmd, pty=True).return_code

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

def set_value(self, value):
        """Set value of the checkbox.

        Parameters
        ----------
        value : bool
            value for the checkbox

        """
        if value:
            self.setCheckState(Qt.Checked)
        else:
            self.setCheckState(Qt.Unchecked)

def show_xticklabels(self, row, column):
        """Show the x-axis tick labels for a subplot.

        :param row,column: specify the subplot.

        """
        subplot = self.get_subplot_at(row, column)
        subplot.show_xticklabels()

def resizeEvent(self, event):
        """Reimplement Qt method"""
        if not self.isMaximized() and not self.fullscreen_flag:
            self.window_size = self.size()
        QMainWindow.resizeEvent(self, event)

        # To be used by the tour to be able to resize
        self.sig_resized.emit(event)

def PrintSummaryTable(self):
    """Prints a summary table."""
    print("""

As of {0:s} the repository contains:

| **File paths covered** | **{1:d}** |
| :------------------ | ------: |
| **Registry keys covered** | **{2:d}** |
| **Total artifacts** | **{3:d}** |
""".format(
    time.strftime('%Y-%m-%d'), self.path_count, self.reg_key_count,
    self.total_count))

def state(self):
    """Returns the current LED state by querying the remote controller."""
    ev = self._query_waiters.request(self.__do_query_state)
    ev.wait(1.0)
    return self._state

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

def full(self):
        """Return True if the queue is full"""
        if not self.size: return False
        return len(self.pq) == (self.size + self.removed_count)

def get_tablenames(cur):
    """ Conveinience: """
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tablename_list_ = cur.fetchall()
    tablename_list = [str(tablename[0]) for tablename in tablename_list_ ]
    return tablename_list

def remove_duplicates(lst):
    """
    Emulate what a Python ``set()`` does, but keeping the element's order.
    """
    dset = set()
    return [l for l in lst if l not in dset and not dset.add(l)]

def rel_path(filename):
    """
    Function that gets relative path to the filename
    """
    return os.path.join(os.getcwd(), os.path.dirname(__file__), filename)

def convert_timestamp(timestamp):
    """
    Converts bokehJS timestamp to datetime64.
    """
    datetime = dt.datetime.utcfromtimestamp(timestamp/1000.)
    return np.datetime64(datetime.replace(tzinfo=None))

def get_file_name(url):
  """Returns file name of file at given url."""
  return os.path.basename(urllib.parse.urlparse(url).path) or 'unknown_name'

def _quit(self, *args):
        """ quit crash """
        self.logger.warn('Bye!')
        sys.exit(self.exit())

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

def prepare(self):
        """Prepare the handler, ensuring RabbitMQ is connected or start a new
        connection attempt.

        """
        super(RabbitMQRequestHandler, self).prepare()
        if self._rabbitmq_is_closed:
            self._connect_to_rabbitmq()

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

def rnormal(mu, tau, size=None):
    """
    Random normal variates.
    """
    return np.random.normal(mu, 1. / np.sqrt(tau), size)

def _num_cpus_darwin():
    """Return the number of active CPUs on a Darwin system."""
    p = subprocess.Popen(['sysctl','-n','hw.ncpu'],stdout=subprocess.PIPE)
    return p.stdout.read()

def newest_file(file_iterable):
  """
  Returns the name of the newest file given an iterable of file names.

  """
  return max(file_iterable, key=lambda fname: os.path.getmtime(fname))

def _rndPointDisposition(dx, dy):
        """Return random disposition point."""
        x = int(random.uniform(-dx, dx))
        y = int(random.uniform(-dy, dy))
        return (x, y)

def get_module_path(modname):
    """Return module *modname* base path"""
    return osp.abspath(osp.dirname(sys.modules[modname].__file__))

def quote(self, s):
        """Return a shell-escaped version of the string s."""

        if six.PY2:
            from pipes import quote
        else:
            from shlex import quote

        return quote(s)

def timeit(output):
    """
    If output is string, then print the string and also time used
    """
    b = time.time()
    yield
    print output, 'time used: %.3fs' % (time.time()-b)

def read_string(buff, byteorder='big'):
    """Read a string from a file-like object."""
    length = read_numeric(USHORT, buff, byteorder)
    return buff.read(length).decode('utf-8')

def add_to_toolbar(self, toolbar, widget):
        """Add widget actions to toolbar"""
        actions = widget.toolbar_actions
        if actions is not None:
            add_actions(toolbar, actions)

def load_data(filename):
    """
    :rtype : numpy matrix
    """
    data = pandas.read_csv(filename, header=None, delimiter='\t', skiprows=9)
    return data.as_matrix()

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

def lines(input):
    """Remove comments and empty lines"""
    for raw_line in input:
        line = raw_line.strip()
        if line and not line.startswith('#'):
            yield strip_comments(line)

def get_user_name():
    """Get user name provide by operating system
    """

    if sys.platform == 'win32':
        #user = os.getenv('USERPROFILE')
        user = os.getenv('USERNAME')
    else:
        user = os.getenv('LOGNAME')

    return user

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

def compute_y(self, coefficients, num_x):
        """ Return calculated y-values for the domain of x-values in [1, num_x]. """
        y_vals = []

        for x in range(1, num_x + 1):
            y = sum([c * x ** i for i, c in enumerate(coefficients[::-1])])
            y_vals.append(y)

        return y_vals

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

def _add_pos1(token):
    """
    Adds a 'pos1' element to a frog token.
    """
    result = token.copy()
    result['pos1'] = _POSMAP[token['pos'].split("(")[0]]
    return result

def read_string_from_file(path, encoding="utf8"):
  """
  Read entire contents of file into a string.
  """
  with codecs.open(path, "rb", encoding=encoding) as f:
    value = f.read()
  return value

def relative_path(path):
    """
    Return the given path relative to this file.
    """
    return os.path.join(os.path.dirname(__file__), path)

def getfirstline(file, default):
    """
    Returns the first line of a file.
    """
    with open(file, 'rb') as fh:
        content = fh.readlines()
        if len(content) == 1:
            return content[0].decode('utf-8').strip('\n')

    return default

def page_guiref(arg_s=None):
    """Show a basic reference about the GUI Console."""
    from IPython.core import page
    page.page(gui_reference, auto_html=True)

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

def save_excel(self, fd):
        """ Saves the case as an Excel spreadsheet.
        """
        from pylon.io.excel import ExcelWriter
        ExcelWriter(self).write(fd)

async def async_input(prompt):
    """
    Python's ``input()`` is blocking, which means the event loop we set
    above can't be running while we're blocking there. This method will
    let the loop run while we wait for input.
    """
    print(prompt, end='', flush=True)
    return (await loop.run_in_executor(None, sys.stdin.readline)).rstrip()

def set_font_size(self, size):
        """Convenience method for just changing font size."""
        if self.font.font_size == size:
            pass
        else:
            self.font._set_size(size)

def open_json(file_name):
    """
    returns json contents as string
    """
    with open(file_name, "r") as json_data:
        data = json.load(json_data)
        return data

def get_lines(handle, line):
    """
    Get zero-indexed line from an open file-like.
    """
    for i, l in enumerate(handle):
        if i == line:
            return l

def is_read_only(object):
    """
    Returns if given object is read only ( built-in or extension ).

    :param object: Object.
    :type object: object
    :return: Is object read only.
    :rtype: bool
    """

    try:
        attribute = "_trace__read__"
        setattr(object, attribute, True)
        delattr(object, attribute)
        return False
    except (TypeError, AttributeError):
        return True

def draw_header(self, stream, header):
        """Draw header with underline"""
        stream.writeln('=' * (len(header) + 4))
        stream.writeln('| ' + header + ' |')
        stream.writeln('=' * (len(header) + 4))
        stream.writeln()

def url_read_text(url, verbose=True):
    r"""
    Directly reads text data from url
    """
    data = url_read(url, verbose)
    text = data.decode('utf8')
    return text

def get_xy_grids(ds, stride=1, getval=False):
    """Return 2D arrays of x and y map coordinates for input GDAL Dataset 
    """
    gt = ds.GetGeoTransform()
    #stride = stride_m/gt[1]
    pX = np.arange(0, ds.RasterXSize, stride)
    pY = np.arange(0, ds.RasterYSize, stride)
    psamp = np.meshgrid(pX, pY)
    mX, mY = pixelToMap(psamp[0], psamp[1], gt)
    return mX, mY

def _parse_config(config_file_path):
    """ Parse Config File from yaml file. """
    config_file = open(config_file_path, 'r')
    config = yaml.load(config_file)
    config_file.close()
    return config

def invertDictMapping(d):
    """ Invert mapping of dictionary (i.e. map values to list of keys) """
    inv_map = {}
    for k, v in d.items():
        inv_map[v] = inv_map.get(v, [])
        inv_map[v].append(k)
    return inv_map

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

def json_iter (path):
    """
    iterator for JSON-per-line in a file pattern
    """
    with open(path, 'r') as f:
        for line in f.readlines():
            yield json.loads(line)

def mouse_move_event(self, event):
        """
        Forward mouse cursor position events to the example
        """
        self.example.mouse_position_event(event.x(), event.y())

def get_var_type(self, name):
        """
        Return type string, compatible with numpy.
        """
        name = create_string_buffer(name)
        type_ = create_string_buffer(MAXSTRLEN)
        self.library.get_var_type.argtypes = [c_char_p, c_char_p]
        self.library.get_var_type(name, type_)
        return type_.value

def acquire_node(self, node):
        """
        acquire a single redis node
        """
        try:
            return node.set(self.resource, self.lock_key, nx=True, px=self.ttl)
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
            return False

def java_version():
    """Call java and return version information.

    :return unicode: Java version string
    """
    result = subprocess.check_output(
        [c.JAVA, '-version'], stderr=subprocess.STDOUT
    )
    first_line = result.splitlines()[0]
    return first_line.decode()

def expireat(self, key, when):
        """Emulate expireat"""
        expire_time = datetime.fromtimestamp(when)
        key = self._encode(key)
        if key in self.redis:
            self.timeouts[key] = expire_time
            return True
        return False

def init_checks_registry():
    """Register all globally visible functions.

    The first argument name is either 'physical_line' or 'logical_line'.
    """
    mod = inspect.getmodule(register_check)
    for (name, function) in inspect.getmembers(mod, inspect.isfunction):
        register_check(function)

def find_root(self):
        """ Traverse parent refs to top. """
        cmd = self
        while cmd.parent:
            cmd = cmd.parent
        return cmd

def _load_texture(file_name, resolver):
    """
    Load a texture from a file into a PIL image.
    """
    file_data = resolver.get(file_name)
    image = PIL.Image.open(util.wrap_as_stream(file_data))
    return image

def _tuple_repr(data):
    """Return a repr() for a list/tuple"""
    if len(data) == 1:
        return "(%s,)" % rpr(data[0])
    else:
        return "(%s)" % ", ".join([rpr(x) for x in data])

def Load(file):
    """ Loads a model from specified file """
    with open(file, 'rb') as file:
        model = dill.load(file)
        return model

def iter_finds(regex_obj, s):
    """Generate all matches found within a string for a regex and yield each match as a string"""
    if isinstance(regex_obj, str):
        for m in re.finditer(regex_obj, s):
            yield m.group()
    else:
        for m in regex_obj.finditer(s):
            yield m.group()

def make_bintree(levels):
    """Make a symmetrical binary tree with @levels"""
    G = nx.DiGraph()
    root = '0'
    G.add_node(root)
    add_children(G, root, levels, 2)
    return G

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

def _makes_clone(_func, *args, **kw):
    """
    A decorator that returns a clone of the current object so that
    we can re-use the object for similar requests.
    """
    self = args[0]._clone()
    _func(self, *args[1:], **kw)
    return self

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

def linedelimited (inlist,delimiter):
    """
Returns a string composed of elements in inlist, with each element
separated by 'delimiter.'  Used by function writedelimited.  Use '\t'
for tab-delimiting.

Usage:   linedelimited (inlist,delimiter)
"""
    outstr = ''
    for item in inlist:
        if type(item) != StringType:
            item = str(item)
        outstr = outstr + item + delimiter
    outstr = outstr[0:-1]
    return outstr

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

def alert(text='', title='', button=OK_TEXT, root=None, timeout=None):
    """Displays a simple message box with text and a single OK button. Returns the text of the button clicked on."""
    assert TKINTER_IMPORT_SUCCEEDED, 'Tkinter is required for pymsgbox'
    return _buttonbox(msg=text, title=title, choices=[str(button)], root=root, timeout=timeout)

def cmd_reindex():
    """Uses CREATE INDEX CONCURRENTLY to create a duplicate index, then tries to swap the new index for the original.

    The index swap is done using a short lock timeout to prevent it from interfering with running queries. Retries until
    the rename succeeds.
    """
    db = connect(args.database)
    for idx in args.indexes:
        pg_reindex(db, idx)

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

def update_redirect(self):
        """
            Call it on your own endpoint's to update the back history navigation.
            If you bypass it, the next submit or back will go over it.
        """
        page_history = Stack(session.get("page_history", []))
        page_history.push(request.url)
        session["page_history"] = page_history.to_json()

def get_column_keys_and_names(table):
    """
    Return a generator of tuples k, c such that k is the name of the python attribute for
    the column and c is the name of the column in the sql table.
    """
    ins = inspect(table)
    return ((k, c.name) for k, c in ins.mapper.c.items())

def filter_dict_by_key(d, keys):
    """Filter the dict *d* to remove keys not in *keys*."""
    return {k: v for k, v in d.items() if k in keys}

def sent2features(sentence, template):
    """ extract features in a sentence

    :type sentence: list of token, each token is a list of tag
    """
    return [word2features(sentence, i, template) for i in range(len(sentence))]

def handle_whitespace(text):
    r"""Handles whitespace cleanup.

    Tabs are "smartly" retabbed (see sub_retab). Lines that contain
    only whitespace are truncated to a single newline.
    """
    text = re_retab.sub(sub_retab, text)
    text = re_whitespace.sub('', text).strip()
    return text

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

def _remove_blank(l):
        """ Removes trailing zeros in the list of integers and returns a new list of integers"""
        ret = []
        for i, _ in enumerate(l):
            if l[i] == 0:
                break
            ret.append(l[i])
        return ret

def erase_lines(n=1):
    """ Erases n lines from the screen and moves the cursor up to follow
    """
    for _ in range(n):
        print(codes.cursor["up"], end="")
        print(codes.cursor["eol"], end="")

def auto():
	"""set colouring on if STDOUT is a terminal device, off otherwise"""
	try:
		Style.enabled = False
		Style.enabled = sys.stdout.isatty()
	except (AttributeError, TypeError):
		pass

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

def _remove_dict_keys_with_value(dict_, val):
  """Removes `dict` keys which have have `self` as value."""
  return {k: v for k, v in dict_.items() if v is not val}

def figsize(x=8, y=7., aspect=1.):
    """ manually set the default figure size of plots
    ::Arguments::
        x (float): x-axis size
        y (float): y-axis size
        aspect (float): aspect ratio scalar
    """
    # update rcparams with adjusted figsize params
    mpl.rcParams.update({'figure.figsize': (x*aspect, y)})

def to_identifier(s):
  """
  Convert snake_case to camel_case.
  """
  if s.startswith('GPS'):
      s = 'Gps' + s[3:]
  return ''.join([i.capitalize() for i in s.split('_')]) if '_' in s else s

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

def vowels(self):
        """
        Return a new IPAString, containing only the vowels in the current string.

        :rtype: IPAString
        """
        return IPAString(ipa_chars=[c for c in self.ipa_chars if c.is_vowel])

def prune(self, n):
        """prune all but the first (=best) n items"""
        if self.minimize:
            self.data = self.data[:n]
        else:
            self.data = self.data[-1 * n:]

def submitbutton(self, request, tag):
        """
        Render an INPUT element of type SUBMIT which will post this form to the
        server.
        """
        return tags.input(type='submit',
                          name='__submit__',
                          value=self._getDescription())

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

def flatten_list(l):
    """ Nested lists to single-level list, does not split strings"""
    return list(chain.from_iterable(repeat(x,1) if isinstance(x,str) else x for x in l))

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

def purge_dict(idict):
    """Remove null items from a dictionary """
    odict = {}
    for key, val in idict.items():
        if is_null(val):
            continue
        odict[key] = val
    return odict

def set_color(self, fg=None, bg=None, intensify=False, target=sys.stdout):
        """Set foreground- and background colors and intensity."""
        raise NotImplementedError

def pop(self, index=-1):
		"""Remove and return the item at index."""
		value = self._list.pop(index)
		del self._dict[value]
		return value

def selecttrue(table, field, complement=False):
    """Select rows where the given field evaluates `True`."""

    return select(table, field, lambda v: bool(v), complement=complement)

def unaccentuate(s):
    """ Replace accentuated chars in string by their non accentuated equivalent. """
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def less_strict_bool(x):
    """Idempotent and None-safe version of strict_bool."""
    if x is None:
        return False
    elif x is True or x is False:
        return x
    else:
        return strict_bool(x)

def subn_filter(s, find, replace, count=0):
    """A non-optimal implementation of a regex filter"""
    return re.gsub(find, replace, count, s)

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

def __unixify(self, s):
        """ stupid windows. converts the backslash to forwardslash for consistency """
        return os.path.normpath(s).replace(os.sep, "/")

def batch(items, size):
    """Batches a list into a list of lists, with sub-lists sized by a specified
    batch size."""
    return [items[x:x + size] for x in xrange(0, len(items), size)]

def multi_replace(instr, search_list=[], repl_list=None):
    """
    Does a string replace with a list of search and replacements

    TODO: rename
    """
    repl_list = [''] * len(search_list) if repl_list is None else repl_list
    for ser, repl in zip(search_list, repl_list):
        instr = instr.replace(ser, repl)
    return instr

def configure_relation(graph, ns, mappings):
    """
    Register relation endpoint(s) between two resources.

    """
    convention = RelationConvention(graph)
    convention.configure(ns, mappings)

def _replace_nan(a, val):
    """
    replace nan in a by val, and returns the replaced array and the nan
    position
    """
    mask = isnull(a)
    return where_method(val, mask, a), mask

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

def http_request_json(*args, **kwargs):
    """

    See: http_request
    """
    ret, status = http_request(*args, **kwargs)
    return json.loads(ret), status

def stdoutwriteline(*args):
    """
    @type args: tuple
    @return: None
    """
    s = ""

    for i in args:
        s += str(i) + " "

    s = s.strip()
    sys.stdout.write(str(s) + "\n")
    sys.stdout.flush()

    return s

def copy_user_agent_from_driver(self):
        """ Updates requests' session user-agent with the driver's user agent

        This method will start the browser process if its not already running.
        """
        selenium_user_agent = self.driver.execute_script("return navigator.userAgent;")
        self.headers.update({"user-agent": selenium_user_agent})

def _show(self, message, indent=0, enable_verbose=True):  # pragma: no cover
        """Message printer.
        """
        if enable_verbose:
            print("    " * indent + message)

def save_session_to_file(self, sessionfile):
        """Not meant to be used directly, use :meth:`Instaloader.save_session_to_file`."""
        pickle.dump(requests.utils.dict_from_cookiejar(self._session.cookies), sessionfile)

def intround(value):
    """Given a float returns a rounded int. Should give the same result on
    both Py2/3
    """

    return int(decimal.Decimal.from_float(
        value).to_integral_value(decimal.ROUND_HALF_EVEN))

def is_http_running_on(port):
  """ Check if an http server runs on a given port.

  Args:
    The port to check.
  Returns:
    True if it is used by an http server. False otherwise.
  """
  try:
    conn = httplib.HTTPConnection('127.0.0.1:' + str(port))
    conn.connect()
    conn.close()
    return True
  except Exception:
    return False

def test():
    """Run the unit tests."""
    import unittest
    tests = unittest.TestLoader().discover('tests')
    unittest.TextTestRunner(verbosity=2).run(tests)

def download(url, encoding='utf-8'):
    """Returns the text fetched via http GET from URL, read as `encoding`"""
    import requests
    response = requests.get(url)
    response.encoding = encoding
    return response.text

def sent2features(sentence, template):
    """ extract features in a sentence

    :type sentence: list of token, each token is a list of tag
    """
    return [word2features(sentence, i, template) for i in range(len(sentence))]

def clear(self):
        """ clear plot """
        self.axes.cla()
        self.conf.ntrace = 0
        self.conf.xlabel = ''
        self.conf.ylabel = ''
        self.conf.title  = ''

def __init__(self, pos, cell, motion, cellmotion):
        self.pos = pos
        """(x, y) position of the mouse on the screen.
        type: (int, int)"""
        self.cell = cell
        """(x, y) position of the mouse snapped to a cell on the root console.
        type: (int, int)"""
        self.motion = motion
        """(x, y) motion of the mouse on the screen.
        type: (int, int)"""
        self.cellmotion = cellmotion
        """(x, y) mostion of the mouse moving over cells on the root console.
        type: (int, int)"""

def resize(self, size):
		""" Resize current array. If size is None, then array became nonfixed-length array. If new size is
		less then current size and value, then value will be truncated (lesser significant bits will be
		truncated).

		:param size:
		:return:
		"""
		if size is not None:
			self.__value = int(WBinArray(self.__value)[:size])
		self.__size = size

def print_out(self, *lst):
      """ Print list of strings to the predefined stdout. """
      self.print2file(self.stdout, True, True, *lst)

def raise_for_not_ok_status(response):
    """
    Raises a `requests.exceptions.HTTPError` if the response has a non-200
    status code.
    """
    if response.code != OK:
        raise HTTPError('Non-200 response code (%s) for url: %s' % (
            response.code, uridecode(response.request.absoluteURI)))

    return response

def disable_busy_cursor():
    """Disable the hourglass cursor and listen for layer changes."""
    while QgsApplication.instance().overrideCursor() is not None and \
            QgsApplication.instance().overrideCursor().shape() == \
            QtCore.Qt.WaitCursor:
        QgsApplication.instance().restoreOverrideCursor()

async def json_or_text(response):
    """Turns response into a properly formatted json or text object"""
    text = await response.text()
    if response.headers['Content-Type'] == 'application/json; charset=utf-8':
        return json.loads(text)
    return text

def get_auth():
    """Get authentication."""
    import getpass
    user = input("User Name: ")  # noqa
    pswd = getpass.getpass('Password: ')
    return Github(user, pswd)

def http(self, *args, **kwargs):
        """Starts the process of building a new HTTP route linked to this API instance"""
        kwargs['api'] = self.api
        return http(*args, **kwargs)

def ILIKE(pattern):
    """Unix shell-style wildcards. Case-insensitive"""
    return P(lambda x: fnmatch.fnmatch(x.lower(), pattern.lower()))

def documentation(self):
        """
        Get the documentation that the server sends for the API.
        """
        newclient = self.__class__(self.session, self.root_url)
        return newclient.get_raw('/')

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

def get_key_goids(self, goids):
        """Given GO IDs, return key GO IDs."""
        go2obj = self.go2obj
        return set(go2obj[go].id for go in goids)

def requests_post(url, data=None, json=None, **kwargs):
    """Requests-mock requests.post wrapper."""
    return requests_request('post', url, data=data, json=json, **kwargs)

def get_idx_rect(index_list):
    """Extract the boundaries from a list of indexes"""
    rows, cols = list(zip(*[(i.row(), i.column()) for i in index_list]))
    return ( min(rows), max(rows), min(cols), max(cols) )

def _go_to_line(editor, line):
    """
    Move cursor to this line in the current buffer.
    """
    b = editor.application.current_buffer
    b.cursor_position = b.document.translate_row_col_to_index(max(0, int(line) - 1), 0)

def find_start_point(self):
        """
        Find the first location in our array that is not empty
        """
        for i, row in enumerate(self.data):
            for j, _ in enumerate(row):
                if self.data[i, j] != 0:  # or not np.isfinite(self.data[i,j]):
                    return i, j

def uniquify_list(L):
    """Same order unique list using only a list compression."""
    return [e for i, e in enumerate(L) if L.index(e) == i]

def norm_vec(vector):
    """Normalize the length of a vector to one"""
    assert len(vector) == 3
    v = np.array(vector)
    return v/np.sqrt(np.sum(v**2))

def get_file_string(filepath):
    """Get string from file."""
    with open(os.path.abspath(filepath)) as f:
        return f.read()

def filter_regex(names, regex):
    """
    Return a tuple of strings that match the regular expression pattern.
    """
    return tuple(name for name in names
                 if regex.search(name) is not None)

def _get_sql(filename):
    """Returns the contents of the sql file from the given ``filename``."""
    with open(os.path.join(SQL_DIR, filename), 'r') as f:
        return f.read()

def out_shape_from_array(arr):
    """Get the output shape from an array."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.shape
    else:
        return (arr.shape[1],)

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

def filter_regex(names, regex):
    """
    Return a tuple of strings that match the regular expression pattern.
    """
    return tuple(name for name in names
                 if regex.search(name) is not None)

def pickle_load(fname):
    """return the contents of a pickle file"""
    assert type(fname) is str and os.path.exists(fname)
    print("loaded",fname)
    return pickle.load(open(fname,"rb"))

def get_key_by_value(dictionary, search_value):
    """
    searchs a value in a dicionary and returns the key of the first occurrence

    :param dictionary: dictionary to search in
    :param search_value: value to search for
    """
    for key, value in dictionary.iteritems():
        if value == search_value:
            return ugettext(key)

def plot_dist_normal(s, mu, sigma):
    """
    plot distribution
    """
    import matplotlib.pyplot as plt
    count, bins, ignored = plt.hist(s, 30, normed=True)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) \
            * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), \
            linewidth = 2, color = 'r')
    plt.show()

def _pdf_at_peak(self):
    """Pdf evaluated at the peak."""
    return (self.peak - self.low) / (self.high - self.low)

def _dict_to_proto(py_dict, proto):
        """
        Converts a python dictionary to the proto supplied

        :param py_dict: The dictionary to convert
        :type py_dict: dict
        :param proto: The proto object to merge with dictionary
        :type proto: protobuf
        :return: A parsed python dictionary in provided proto format
        :raises:
            ParseError: On JSON parsing problems.
        """
        dict_json_str = json.dumps(py_dict)
        return json_format.Parse(dict_json_str, proto)

def round_to_n(x, n):
    """
    Round to sig figs
    """
    return round(x, -int(np.floor(np.log10(x))) + (n - 1))

def trigger(self, target: str, trigger: str, parameters: Dict[str, Any]={}):
		"""Calls the specified Trigger of another Area with the optionally given parameters.

		Args:
			target: The name of the target Area.
			trigger: The name of the Trigger.
			parameters: The parameters of the function call.
		"""
		pass

def get_rounded(self, digits):
        """ Return a vector with the elements rounded to the given number of digits. """
        result = self.copy()
        result.round(digits)
        return result

def open_with_encoding(filename, encoding, mode='r'):
    """Return opened file with a specific encoding."""
    return io.open(filename, mode=mode, encoding=encoding,
                   newline='')

def call_and_exit(self, cmd, shell=True):
        """Run the *cmd* and exit with the proper exit code."""
        sys.exit(subprocess.call(cmd, shell=shell))

def merge(left, right, how='inner', key=None, left_key=None, right_key=None,
          left_as='left', right_as='right'):
    """ Performs a join using the union join function. """
    return join(left, right, how, key, left_key, right_key,
                join_fn=make_union_join(left_as, right_as))

async def result_processor(tasks):
    """An async result aggregator that combines all the results
       This gets executed in unsync.loop and unsync.thread"""
    output = {}
    for task in tasks:
        num, res = await task
        output[num] = res
    return output

def add_text(text, x=0.01, y=0.01, axes="gca", draw=True, **kwargs):
    """
    Adds text to the axes at the specified position.

    **kwargs go to the axes.text() function.
    """
    if axes=="gca": axes = _pylab.gca()
    axes.text(x, y, text, transform=axes.transAxes, **kwargs)
    if draw: _pylab.draw()

def safe_quotes(text, escape_single_quotes=False):
    """htmlify string"""
    if isinstance(text, str):
        safe_text = text.replace('"', "&quot;")
        if escape_single_quotes:
            safe_text = safe_text.replace("'", "&#92;'")
        return safe_text.replace('True', 'true')
    return text

def make_post_request(self, url, auth, json_payload):
        """This function executes the request with the provided
        json payload and return the json response"""
        response = requests.post(url, auth=auth, json=json_payload)
        return response.json()

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

def error(*args):
    """Display error message via stderr or GUI."""
    if sys.stdin.isatty():
        print('ERROR:', *args, file=sys.stderr)
    else:
        notify_error(*args)

def close(*args, **kwargs):
    r"""Close last created figure, alias to ``plt.close()``."""
    _, plt, _ = _import_plt()
    plt.close(*args, **kwargs)

def head(filename, n=10):
    """ prints the top `n` lines of a file """
    with freader(filename) as fr:
        for _ in range(n):
            print(fr.readline().strip())

def save_session_to_file(self, sessionfile):
        """Not meant to be used directly, use :meth:`Instaloader.save_session_to_file`."""
        pickle.dump(requests.utils.dict_from_cookiejar(self._session.cookies), sessionfile)

def return_letters_from_string(text):
    """Get letters from string only."""
    out = ""
    for letter in text:
        if letter.isalpha():
            out += letter
    return out

def save_session_to_file(self, sessionfile):
        """Not meant to be used directly, use :meth:`Instaloader.save_session_to_file`."""
        pickle.dump(requests.utils.dict_from_cookiejar(self._session.cookies), sessionfile)

def format_exc(limit=None):
    """Like print_exc() but return a string. Backport for Python 2.3."""
    try:
        etype, value, tb = sys.exc_info()
        return ''.join(traceback.format_exception(etype, value, tb, limit))
    finally:
        etype = value = tb = None

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def _get_name(self, key):
        """ get display name for a key, or mangle for display """
        if key in self.display_names:
            return self.display_names[key]

        return key.capitalize()

def scipy_sparse_to_spmatrix(A):
    """Efficient conversion from scipy sparse matrix to cvxopt sparse matrix"""
    coo = A.tocoo()
    SP = spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=A.shape)
    return SP

def _screen(self, s, newline=False):
        """Print something on screen when self.verbose == True"""
        if self.verbose:
            if newline:
                print(s)
            else:
                print(s, end=' ')

def scipy_sparse_to_spmatrix(A):
    """Efficient conversion from scipy sparse matrix to cvxopt sparse matrix"""
    coo = A.tocoo()
    SP = spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=A.shape)
    return SP

def _stdout_raw(self, s):
        """Writes the string to stdout"""
        print(s, end='', file=sys.stdout)
        sys.stdout.flush()

def write_wav(path, samples, sr=16000):
    """
    Write to given samples to a wav file.
    The samples are expected to be floating point numbers
    in the range of -1.0 to 1.0.

    Args:
        path (str): The path to write the wav to.
        samples (np.array): A float array .
        sr (int): The sampling rate.
    """
    max_value = np.abs(np.iinfo(np.int16).min)
    data = (samples * max_value).astype(np.int16)
    scipy.io.wavfile.write(path, sr, data)

def string_to_identity(identity_str):
    """Parse string into Identity dictionary."""
    m = _identity_regexp.match(identity_str)
    result = m.groupdict()
    log.debug('parsed identity: %s', result)
    return {k: v for k, v in result.items() if v}

def get_soup(page=''):
    """
    Returns a bs4 object of the page requested
    """
    content = requests.get('%s/%s' % (BASE_URL, page)).text
    return BeautifulSoup(content)

def init_checks_registry():
    """Register all globally visible functions.

    The first argument name is either 'physical_line' or 'logical_line'.
    """
    mod = inspect.getmodule(register_check)
    for (name, function) in inspect.getmembers(mod, inspect.isfunction):
        register_check(function)

def mouse_out(self):
        """
        Performs a mouse out the element.

        Currently works only on Chrome driver.
        """
        self.scroll_to()
        ActionChains(self.parent.driver).move_by_offset(0, 0).click().perform()

def push(h, x):
    """Push a new value into heap."""
    h.push(x)
    up(h, h.size()-1)

def _set_scroll_v(self, *args):
        """Scroll both categories Canvas and scrolling container"""
        self._canvas_categories.yview(*args)
        self._canvas_scroll.yview(*args)

def bash(filename):
    """Runs a bash script in the local directory"""
    sys.stdout.flush()
    subprocess.call("bash {}".format(filename), shell=True)

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

def _add_hash(source):
    """Add a leading hash '#' at the beginning of every line in the source."""
    source = '\n'.join('# ' + line.rstrip()
                       for line in source.splitlines())
    return source

def selecttrue(table, field, complement=False):
    """Select rows where the given field evaluates `True`."""

    return select(table, field, lambda v: bool(v), complement=complement)

def execute_in_background(self):
        """Executes a (shell) command in the background

        :return: the process' pid
        """
        # http://stackoverflow.com/questions/1605520
        args = shlex.split(self.cmd)
        p = Popen(args)
        return p.pid

def selecttrue(table, field, complement=False):
    """Select rows where the given field evaluates `True`."""

    return select(table, field, lambda v: bool(v), complement=complement)

def add_to_js(self, name, var):
        """Add an object to Javascript."""
        frame = self.page().mainFrame()
        frame.addToJavaScriptWindowObject(name, var)

def selecttrue(table, field, complement=False):
    """Select rows where the given field evaluates `True`."""

    return select(table, field, lambda v: bool(v), complement=complement)

def ratio_and_percentage(current, total, time_remaining):
    """Returns the progress ratio and percentage."""
    return "{} / {} ({}% completed)".format(current, total, int(current / total * 100))

def ask_dir(self):
		"""
		dialogue box for choosing directory
		"""
		args ['directory'] = askdirectory(**self.dir_opt) 
		self.dir_text.set(args ['directory'])

def numpy(self):
        """ Grabs image data and converts it to a numpy array """
        # load GDCM's image reading functionality
        image_reader = gdcm.ImageReader()
        image_reader.SetFileName(self.fname)
        if not image_reader.Read():
            raise IOError("Could not read DICOM image")
        pixel_array = self._gdcm_to_numpy(image_reader.GetImage())
        return pixel_array

def selectgt(table, field, value, complement=False):
    """Select rows where the given field is greater than the given value."""

    value = Comparable(value)
    return selectop(table, field, value, operator.gt, complement=complement)

def read_utf8(fh, byteorder, dtype, count, offsetsize):
    """Read tag data from file and return as unicode string."""
    return fh.read(count).decode('utf-8')

def selectnotnone(table, field, complement=False):
    """Select rows where the given field is not `None`."""

    return select(table, field, lambda v: v is not None,
                  complement=complement)

def ReadManyFromPath(filepath):
  """Reads a Python object stored in a specified YAML file.

  Args:
    filepath: A filepath to the YAML file.

  Returns:
    A Python data structure corresponding to the YAML in the given file.
  """
  with io.open(filepath, mode="r", encoding="utf-8") as filedesc:
    return ReadManyFromFile(filedesc)

def selectnotnone(table, field, complement=False):
    """Select rows where the given field is not `None`."""

    return select(table, field, lambda v: v is not None,
                  complement=complement)

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

def filter_by_ids(original_list, ids_to_filter):
    """Filter a list of dicts by IDs using an id key on each dict."""
    if not ids_to_filter:
        return original_list

    return [i for i in original_list if i['id'] in ids_to_filter]

def load_yaml(filepath):
    """Convenience function for loading yaml-encoded data from disk."""
    with open(filepath) as f:
        txt = f.read()
    return yaml.load(txt)

def __init__(self):
    """Initializes an attribute container identifier."""
    super(AttributeContainerIdentifier, self).__init__()
    self._identifier = id(self)

def main_func(args=None):
    """Main funcion when executing this module as script

    :param args: commandline arguments
    :type args: list
    :returns: None
    :rtype: None
    :raises: None
    """
    # we have to initialize a gui even if we dont need one right now.
    # as soon as you call maya.standalone.initialize(), a QApplication
    # with type Tty is created. This is the type for conosle apps.
    # Because i have not found a way to replace that, we just init the gui.
    guimain.init_gui()

    main.init()
    launcher = Launcher()
    parsed, unknown = launcher.parse_args(args)
    parsed.func(parsed, unknown)

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

def _trim(image):
    """Trim a PIL image and remove white space."""
    background = PIL.Image.new(image.mode, image.size, image.getpixel((0, 0)))
    diff = PIL.ImageChops.difference(image, background)
    diff = PIL.ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        image = image.crop(bbox)
    return image

def remove_series(self, series):
        """Removes a :py:class:`.Series` from the chart.

        :param Series series: The :py:class:`.Series` to remove.
        :raises ValueError: if you try to remove the last\
        :py:class:`.Series`."""

        if len(self.all_series()) == 1:
            raise ValueError("Cannot remove last series from %s" % str(self))
        self._all_series.remove(series)
        series._chart = None

def del_Unnamed(df):
    """
    Deletes all the unnamed columns

    :param df: pandas dataframe
    """
    cols_del=[c for c in df.columns if 'Unnamed' in c]
    return df.drop(cols_del,axis=1)

def get_header(request, header_service):
    """Return request's 'X_POLYAXON_...:' header, as a bytestring.

    Hide some test client ickyness where the header can be unicode.
    """
    service = request.META.get('HTTP_{}'.format(header_service), b'')
    if isinstance(service, str):
        # Work around django test client oddness
        service = service.encode(HTTP_HEADER_ENCODING)
    return service

def _removeTags(tags, objects):
    """ Removes tags from objects """
    for t in tags:
        for o in objects:
            o.tags.remove(t)

    return True

def save_session_to_file(self, sessionfile):
        """Not meant to be used directly, use :meth:`Instaloader.save_session_to_file`."""
        pickle.dump(requests.utils.dict_from_cookiejar(self._session.cookies), sessionfile)

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

def get_var(name, factory=None):
    """Gets a global variable given its name.

    If factory is not None and the variable is not set, factory
    is a callable that will set the variable.

    If not set, returns None.
    """
    if name not in _VARS and factory is not None:
        _VARS[name] = factory()
    return _VARS.get(name)

def remove_non_magic_cols(self):
        """
        Remove all non-MagIC columns from all tables.
        """
        for table_name in self.tables:
            table = self.tables[table_name]
            table.remove_non_magic_cols_from_table()

def turn(self):
        """Turn the ring for a single position.
        For example, [a, b, c, d] becomes [b, c, d, a]."""
        first = self._data.pop(0)
        self._data.append(first)

def clean_axis(axis):
    """Remove ticks, tick labels, and frame from axis"""
    axis.get_xaxis().set_ticks([])
    axis.get_yaxis().set_ticks([])
    for spine in list(axis.spines.values()):
        spine.set_visible(False)

def set_log_level(logger_name: str, log_level: str, propagate: bool = False):
    """Set the log level of the specified logger."""
    log = logging.getLogger(logger_name)
    log.propagate = propagate
    log.setLevel(log_level)

def split_comma_argument(comma_sep_str):
    """Split a comma separated option into a list."""
    terms = []
    for term in comma_sep_str.split(','):
        if term:
            terms.append(term)
    return terms

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

def set_pivot_keys(self, foreign_key, other_key):
        """
        Set the key names for the pivot model instance
        """
        self.__foreign_key = foreign_key
        self.__other_key = other_key

        return self

def mock_add_spec(self, spec, spec_set=False):
        """Add a spec to a mock. `spec` can either be an object or a
        list of strings. Only attributes on the `spec` can be fetched as
        attributes from the mock.

        If `spec_set` is True then only attributes on the spec can be set."""
        self._mock_add_spec(spec, spec_set)
        self._mock_set_magics()

def fmt_subst(regex, subst):
    """Replace regex with string."""
    return lambda text: re.sub(regex, subst, text) if text else text

def discard(self, element):
        """Remove element from the RangeSet if it is a member.

        If the element is not a member, do nothing.
        """
        try:
            i = int(element)
            set.discard(self, i)
        except ValueError:
            pass

def median(self):
        """Computes the median of a log-normal distribution built with the stats data."""
        mu = self.mean()
        ret_val = math.exp(mu)
        if math.isnan(ret_val):
            ret_val = float("inf")
        return ret_val

def load_file(self, filename):
        """Read in file contents and set the current string."""
        with open(filename, 'r') as sourcefile:
            self.set_string(sourcefile.read())

def normalise_string(string):
    """ Strips trailing whitespace from string, lowercases it and replaces
        spaces with underscores
    """
    string = (string.strip()).lower()
    return re.sub(r'\W+', '_', string)

def to_utc(self, dt):
        """Convert any timestamp to UTC (with tzinfo)."""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=self.utc)
        return dt.astimezone(self.utc)

def dashrepl(value):
    """
    Replace any non-word characters with a dash.
    """
    patt = re.compile(r'\W', re.UNICODE)
    return re.sub(patt, '-', value)

def setdefaults(dct, defaults):
    """Given a target dct and a dict of {key:default value} pairs,
    calls setdefault for all of those pairs."""
    for key in defaults:
        dct.setdefault(key, defaults[key])

    return dct

def im2mat(I):
    """Converts and image to matrix (one pixel per line)"""
    return I.reshape((I.shape[0] * I.shape[1], I.shape[2]))

def check_str(obj):
        """ Returns a string for various input types """
        if isinstance(obj, str):
            return obj
        if isinstance(obj, float):
            return str(int(obj))
        else:
            return str(obj)

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

def issuperset(self, other):
        """Report whether this RangeSet contains another set."""
        self._binary_sanity_check(other)
        return set.issuperset(self, other)

def get_jsonparsed_data(url):
    """Receive the content of ``url``, parse it as JSON and return the
       object.
    """
    response = urlopen(url)
    data = response.read().decode('utf-8')
    return json.loads(data)

def feed_eof(self):
        """Send a potentially "ragged" EOF.

        This method will raise an SSL_ERROR_EOF exception if the EOF is
        unexpected.
        """
        self._incoming.write_eof()
        ssldata, appdata = self.feed_ssldata(b'')
        assert appdata == [] or appdata == [b'']

def maxDepth(self, currentDepth=0):
        """Compute the depth of the longest branch of the tree"""
        if not any((self.left, self.right)):
            return currentDepth
        result = 0
        for child in (self.left, self.right):
            if child:
                result = max(result, child.maxDepth(currentDepth + 1))
        return result

def bounds_to_poly(bounds):
    """
    Constructs a shapely Polygon from the provided bounds tuple.

    Parameters
    ----------
    bounds: tuple
        Tuple representing the (left, bottom, right, top) coordinates

    Returns
    -------
    polygon: shapely.geometry.Polygon
        Shapely Polygon geometry of the bounds
    """
    x0, y0, x1, y1 = bounds
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

def reset(self):
		"""
		Resets the iterator to the start.

		Any remaining values in the current iteration are discarded.
		"""
		self.__iterator, self.__saved = itertools.tee(self.__saved)

def format_exc(*exc_info):
    """Show exception with traceback."""
    typ, exc, tb = exc_info or sys.exc_info()
    error = traceback.format_exception(typ, exc, tb)
    return "".join(error)

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

def _shuffle(data, idx):
    """Shuffle the data."""
    shuffle_data = []

    for idx_k, idx_v in data:
        shuffle_data.append((idx_k, mx.ndarray.array(idx_v.asnumpy()[idx], idx_v.context)))

    return shuffle_data

def round_to_n(x, n):
    """
    Round to sig figs
    """
    return round(x, -int(np.floor(np.log10(x))) + (n - 1))

def _shuffle(data, idx):
    """Shuffle the data."""
    shuffle_data = []

    for idx_k, idx_v in data:
        shuffle_data.append((idx_k, mx.ndarray.array(idx_v.asnumpy()[idx], idx_v.context)))

    return shuffle_data

def round_to_n(x, n):
    """
    Round to sig figs
    """
    return round(x, -int(np.floor(np.log10(x))) + (n - 1))

def lowPass(self, *args):
        """
        Creates a copy of the signal with the low pass applied, args specifed are passed through to _butter. 
        :return: 
        """
        return Signal(self._butter(self.samples, 'low', *args), fs=self.fs)

def begin_stream_loop(stream, poll_interval):
    """Start and maintain the streaming connection..."""
    while should_continue():
        try:
            stream.start_polling(poll_interval)
        except Exception as e:
            # Infinite restart
            logger.error("Exception while polling. Restarting in 1 second.", exc_info=True)
            time.sleep(1)

def sine_wave(frequency):
  """Emit a sine wave at the given frequency."""
  xs = tf.reshape(tf.range(_samples(), dtype=tf.float32), [1, _samples(), 1])
  ts = xs / FLAGS.sample_rate
  return tf.sin(2 * math.pi * frequency * ts)

def lint(args):
    """Run lint checks using flake8."""
    application = get_current_application()
    if not args:
        args = [application.name, 'tests']
    args = ['flake8'] + list(args)
    run.main(args, standalone_mode=False)

def readTuple(self, line, n=3):
        """ Reads a tuple of numbers. e.g. vertices, normals or teture coords.
        """
        numbers = [num for num in line.split(' ') if num]
        return [float(num) for num in numbers[1:n + 1]]

def stub_main():
    """setuptools blah: it still can't run a module as a script entry_point"""
    from google.apputils import run_script_module
    import butcher.main
    run_script_module.RunScriptModule(butcher.main)

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

def save(variable, filename):
    """Save variable on given path using Pickle
    
    Args:
        variable: what to save
        path (str): path of the output
    """
    fileObj = open(filename, 'wb')
    pickle.dump(variable, fileObj)
    fileObj.close()

def _skip_frame(self):
        """Skip the next time frame"""
        for line in self._f:
            if line == 'ITEM: ATOMS\n':
                break
        for i in range(self.num_atoms):
            next(self._f)

def save_excel(self, fd):
        """ Saves the case as an Excel spreadsheet.
        """
        from pylon.io.excel import ExcelWriter
        ExcelWriter(self).write(fd)

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

def getbyteslice(self, start, end):
        """Direct access to byte data."""
        c = self._rawarray[start:end]
        return c

def pickle_save(thing,fname):
    """save something to a pickle file"""
    pickle.dump(thing, open(fname,"wb"),pickle.HIGHEST_PROTOCOL)
    return thing

def is_full_slice(obj, l):
    """
    We have a full length slice.
    """
    return (isinstance(obj, slice) and obj.start == 0 and obj.stop == l and
            obj.step is None)

def resize_image(self, data, size):
        """ Resizes the given image to fit inside a box of the given size. """
        from machina.core.compat import PILImage as Image
        image = Image.open(BytesIO(data))

        # Resize!
        image.thumbnail(size, Image.ANTIALIAS)

        string = BytesIO()
        image.save(string, format='PNG')
        return string.getvalue()

def stop(self, dummy_signum=None, dummy_frame=None):
        """ Shutdown process (this method is also a signal handler) """
        logging.info('Shutting down ...')
        self.socket.close()
        sys.exit(0)

def stop(self):
        """Stops the background synchronization thread"""
        with self.synclock:
            if self.syncthread is not None:
                self.syncthread.cancel()
                self.syncthread = None

def symlink_remove(link):
    """Remove a symlink. Used for model shortcut links.

    link (unicode / Path): The path to the symlink.
    """
    # https://stackoverflow.com/q/26554135/6400719
    if os.path.isdir(path2str(link)) and is_windows:
        # this should only be on Py2.7 and windows
        os.rmdir(path2str(link))
    else:
        os.unlink(path2str(link))

def _get_column_types(self, data):
        """Get a list of the data types for each column in *data*."""
        columns = list(zip_longest(*data))
        return [self._get_column_type(column) for column in columns]

def sort_filenames(filenames):
    """
    sort a list of files by filename only, ignoring the directory names
    """
    basenames = [os.path.basename(x) for x in filenames]
    indexes = [i[0] for i in sorted(enumerate(basenames), key=lambda x:x[1])]
    return [filenames[x] for x in indexes]

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

def _config_win32_domain(self, domain):
        """Configure a Domain registry entry."""
        # we call str() on domain to convert it from unicode to ascii
        self.domain = dns.name.from_text(str(domain))

def _dict_values_sorted_by_key(dictionary):
    # This should be a yield from instead.
    """Internal helper to return the values of a dictionary, sorted by key.
    """
    for _, value in sorted(dictionary.iteritems(), key=operator.itemgetter(0)):
        yield value

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

def direct2dDistance(self, point):
        """consider the distance between two mapPoints, ignoring all terrain, pathing issues"""
        if not isinstance(point, MapPoint): return 0.0
        return  ((self.x-point.x)**2 + (self.y-point.y)**2)**(0.5) # simple distance formula

def chmod_add_excute(filename):
        """
        Adds execute permission to file.
        :param filename:
        :return:
        """
        st = os.stat(filename)
        os.chmod(filename, st.st_mode | stat.S_IEXEC)

def _removeStopwords(text_list):
    """
    Removes stopwords contained in a list of words.

    :param text_string: A list of strings.
    :type text_string: list.

    :returns: The input ``text_list`` with stopwords removed.
    :rtype: list
    """

    output_list = []

    for word in text_list:
        if word.lower() not in _stopwords:
            output_list.append(word)

    return output_list

def GetPythonLibraryDirectoryPath():
  """Retrieves the Python library directory path."""
  path = sysconfig.get_python_lib(True)
  _, _, path = path.rpartition(sysconfig.PREFIX)

  if path.startswith(os.sep):
    path = path[1:]

  return path

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

def _python_rpath(self):
        """The relative path (from environment root) to python."""
        # Windows virtualenv installation installs pip to the [Ss]cripts
        # folder. Here's a simple check to support:
        if sys.platform == 'win32':
            return os.path.join('Scripts', 'python.exe')
        return os.path.join('bin', 'python')

def _not(condition=None, **kwargs):
    """
    Return the opposite of input condition.

    :param condition: condition to process.

    :result: not condition.
    :rtype: bool
    """

    result = True

    if condition is not None:
        result = not run(condition, **kwargs)

    return result

def setValue(self, p_float):
        """Override method to set a value to show it as 0 to 100.

        :param p_float: The float number that want to be set.
        :type p_float: float
        """
        p_float = p_float * 100

        super(PercentageSpinBox, self).setValue(p_float)

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

def datetime_from_timestamp(timestamp, content):
    """
    Helper function to add timezone information to datetime,
    so that datetime is comparable to other datetime objects in recent versions
    that now also have timezone information.
    """
    return set_date_tzinfo(
        datetime.fromtimestamp(timestamp),
        tz_name=content.settings.get('TIMEZONE', None))

def split_on(s, sep=" "):
    """Split s by sep, unless it's inside a quote."""
    pattern = '''((?:[^%s"']|"[^"]*"|'[^']*')+)''' % sep

    return [_strip_speechmarks(t) for t in re.split(pattern, s)[1::2]]

def SetValue(self, row, col, value):
        """
        Set value in the pandas DataFrame
        """
        self.dataframe.iloc[row, col] = value

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

def move_to(x, y):
    """Moves the brush to a particular position.

    Arguments:
        x - a number between -250 and 250.
        y - a number between -180 and 180.
    """
    _make_cnc_request("coord/{0}/{1}".format(x, y))
    state['turtle'].goto(x, y)

def _split_str(s, n):
    """
    split string into list of strings by specified number.
    """
    length = len(s)
    return [s[i:i + n] for i in range(0, length, n)]

def stopwatch_now():
    """Get a timevalue for interval comparisons

    When possible it is a monotonic clock to prevent backwards time issues.
    """
    if six.PY2:
        now = time.time()
    else:
        now = time.monotonic()
    return now

def case_us2mc(x):
    """ underscore to mixed case notation """
    return re.sub(r'_([a-z])', lambda m: (m.group(1).upper()), x)

def file_found(filename,force):
    """Check if a file exists"""
    if os.path.exists(filename) and not force:
        logger.info("Found %s; skipping..."%filename)
        return True
    else:
        return False

def restart(self, reset=False):
        """
        Quit and Restart Spyder application.

        If reset True it allows to reset spyder on restart.
        """
        # Get start path to use in restart script
        spyder_start_directory = get_module_path('spyder')
        restart_script = osp.join(spyder_start_directory, 'app', 'restart.py')

        # Get any initial argument passed when spyder was started
        # Note: Variables defined in bootstrap.py and spyder/app/start.py
        env = os.environ.copy()
        bootstrap_args = env.pop('SPYDER_BOOTSTRAP_ARGS', None)
        spyder_args = env.pop('SPYDER_ARGS')

        # Get current process and python running spyder
        pid = os.getpid()
        python = sys.executable

        # Check if started with bootstrap.py
        if bootstrap_args is not None:
            spyder_args = bootstrap_args
            is_bootstrap = True
        else:
            is_bootstrap = False

        # Pass variables as environment variables (str) to restarter subprocess
        env['SPYDER_ARGS'] = spyder_args
        env['SPYDER_PID'] = str(pid)
        env['SPYDER_IS_BOOTSTRAP'] = str(is_bootstrap)
        env['SPYDER_RESET'] = str(reset)

        if DEV:
            if os.name == 'nt':
                env['PYTHONPATH'] = ';'.join(sys.path)
            else:
                env['PYTHONPATH'] = ':'.join(sys.path)

        # Build the command and popen arguments depending on the OS
        if os.name == 'nt':
            # Hide flashing command prompt
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            shell = False
        else:
            startupinfo = None
            shell = True

        command = '"{0}" "{1}"'
        command = command.format(python, restart_script)

        try:
            if self.closing(True):
                subprocess.Popen(command, shell=shell, env=env,
                                 startupinfo=startupinfo)
                self.console.quit()
        except Exception as error:
            # If there is an error with subprocess, Spyder should not quit and
            # the error can be inspected in the internal console
            print(error)  # spyder: test-skip
            print(command)

def MatrixSolve(a, rhs, adj):
    """
    Matrix solve op.
    """
    return np.linalg.solve(a if not adj else _adjoint(a), rhs),

def _bindingsToDict(self, bindings):
        """
        Given a binding from the sparql query result,
        create a dict of plain text
        """
        myDict = {}
        for key, val in bindings.iteritems():
            myDict[key.toPython().replace('?', '')] = val.toPython()
        return myDict

def MatrixSolve(a, rhs, adj):
    """
    Matrix solve op.
    """
    return np.linalg.solve(a if not adj else _adjoint(a), rhs),

def createdb():
    """Create database tables from sqlalchemy models"""
    manager.db.engine.echo = True
    manager.db.create_all()
    set_alembic_revision()

def algo_exp(x, m, t, b):
    """mono-exponential curve."""
    return m*np.exp(-t*x)+b

def createdb():
    """Create database tables from sqlalchemy models"""
    manager.db.engine.echo = True
    manager.db.create_all()
    set_alembic_revision()

def sort_dict(d, key=None, reverse=False):
    """
    Sorts a dict by value.

    Args:
        d: Input dictionary
        key: Function which takes an tuple (key, object) and returns a value to
            compare and sort by. By default, the function compares the values
            of the dict i.e. key = lambda t : t[1]
        reverse: Allows to reverse sort order.

    Returns:
        OrderedDict object whose keys are ordered according to their value.
    """
    kv_items = [kv for kv in d.items()]

    # Sort kv_items according to key.
    if key is None:
        kv_items.sort(key=lambda t: t[1], reverse=reverse)
    else:
        kv_items.sort(key=key, reverse=reverse)

    # Build ordered dict.
    return collections.OrderedDict(kv_items)

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

def arglexsort(arrays):
    """
    Returns the indices of the lexicographical sorting
    order of the supplied arrays.
    """
    dtypes = ','.join(array.dtype.str for array in arrays)
    recarray = np.empty(len(arrays[0]), dtype=dtypes)
    for i, array in enumerate(arrays):
        recarray['f%s' % i] = array
    return recarray.argsort()

def get_tablenames(cur):
    """ Conveinience: """
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tablename_list_ = cur.fetchall()
    tablename_list = [str(tablename[0]) for tablename in tablename_list_ ]
    return tablename_list

def sort_dict(d, key=None, reverse=False):
    """
    Sorts a dict by value.

    Args:
        d: Input dictionary
        key: Function which takes an tuple (key, object) and returns a value to
            compare and sort by. By default, the function compares the values
            of the dict i.e. key = lambda t : t[1]
        reverse: Allows to reverse sort order.

    Returns:
        OrderedDict object whose keys are ordered according to their value.
    """
    kv_items = [kv for kv in d.items()]

    # Sort kv_items according to key.
    if key is None:
        kv_items.sort(key=lambda t: t[1], reverse=reverse)
    else:
        kv_items.sort(key=key, reverse=reverse)

    # Build ordered dict.
    return collections.OrderedDict(kv_items)

def __init__(self):
    """Initializes the database file object."""
    super(Sqlite3DatabaseFile, self).__init__()
    self._connection = None
    self._cursor = None
    self.filename = None
    self.read_only = None

def arglexsort(arrays):
    """
    Returns the indices of the lexicographical sorting
    order of the supplied arrays.
    """
    dtypes = ','.join(array.dtype.str for array in arrays)
    recarray = np.empty(len(arrays[0]), dtype=dtypes)
    for i, array in enumerate(arrays):
        recarray['f%s' % i] = array
    return recarray.argsort()

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

def sort_nicely(l):
    """Sort the given list in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    l.sort(key=alphanum_key)

def weighted_std(values, weights):
    """ Calculate standard deviation weighted by errors """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return np.sqrt(variance)

def get_order(self, codes):
        """Return evidence codes in order shown in code2name."""
        return sorted(codes, key=lambda e: [self.ev2idx.get(e)])

def _std(self,x):
        """
        Compute standard deviation with ddof degrees of freedom
        """
        return np.nanstd(x.values,ddof=self._ddof)

def asynchronous(function, event):
    """
    Runs the function asynchronously taking care of exceptions.
    """
    thread = Thread(target=synchronous, args=(function, event))
    thread.daemon = True
    thread.start()

def get_function_class(function_name):
    """
    Return the type for the requested function

    :param function_name: the function to return
    :return: the type for that function (i.e., this is a class, not an instance)
    """

    if function_name in _known_functions:

        return _known_functions[function_name]

    else:

        raise UnknownFunction("Function %s is not known. Known functions are: %s" %
                              (function_name, ",".join(_known_functions.keys())))

def safe_call(cls, method, *args):
        """ Call a remote api method but don't raise if an error occurred."""
        return cls.call(method, *args, safe=True)

def text_width(string, font_name, font_size):
    """Determine with width in pixels of string."""
    return stringWidth(string, fontName=font_name, fontSize=font_size)

def is_static(*p):
    """ A static value (does not change at runtime)
    which is known at compile time
    """
    return all(is_CONST(x) or
               is_number(x) or
               is_const(x)
               for x in p)

def incidence(boundary):
    """
    given an Nxm matrix containing boundary info between simplices,
    compute indidence info matrix
    not very reusable; should probably not be in this lib
    """
    return GroupBy(boundary).split(np.arange(boundary.size) // boundary.shape[1])

def _update_staticmethod(self, oldsm, newsm):
        """Update a staticmethod update."""
        # While we can't modify the staticmethod object itself (it has no
        # mutable attributes), we *can* extract the underlying function
        # (by calling __get__(), which returns it) and update it in-place.
        # We don't have the class available to pass to __get__() but any
        # object except None will do.
        self._update(None, None, oldsm.__get__(0), newsm.__get__(0))

def column_stack_2d(data):
    """Perform column-stacking on a list of 2d data blocks."""
    return list(list(itt.chain.from_iterable(_)) for _ in zip(*data))

def disconnect(self):
        """Gracefully close connection to stomp server."""
        if self._connected:
            self._connected = False
            self._conn.disconnect()

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

def _configure_logger():
    """Configure the logging module."""
    if not app.debug:
        _configure_logger_for_production(logging.getLogger())
    elif not app.testing:
        _configure_logger_for_debugging(logging.getLogger())

def _StopStatusUpdateThread(self):
    """Stops the status update thread."""
    self._status_update_active = False
    if self._status_update_thread.isAlive():
      self._status_update_thread.join()
    self._status_update_thread = None

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

def stop_button_click_handler(self):
        """Method to handle what to do when the stop button is pressed"""
        self.stop_button.setDisabled(True)
        # Interrupt computations or stop debugging
        if not self.shellwidget._reading:
            self.interrupt_kernel()
        else:
            self.shellwidget.write_to_stdin('exit')

def md5_string(s):
    """
    Shortcut to create md5 hash
    :param s:
    :return:
    """
    m = hashlib.md5()
    m.update(s)
    return str(m.hexdigest())

def disown(cmd):
    """Call a system command in the background,
       disown it and hide it's output."""
    subprocess.Popen(cmd,
                     stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)

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

def magnitude(X):
    """Magnitude of a complex matrix."""
    r = np.real(X)
    i = np.imag(X)
    return np.sqrt(r * r + i * i);

def loads(s, model=None, parser=None):
    """Deserialize s (a str) to a Python object."""
    with StringIO(s) as f:
        return load(f, model=model, parser=parser)

def dot(self, w):
        """Return the dotproduct between self and another vector."""

        return sum([x * y for x, y in zip(self, w)])

def from_bytes(cls, b):
		"""Create :class:`PNG` from raw bytes.
		
		:arg bytes b: The raw bytes of the PNG file.
		:rtype: :class:`PNG`
		"""
		im = cls()
		im.chunks = list(parse_chunks(b))
		im.init()
		return im

def mean_date(dt_list):
    """Calcuate mean datetime from datetime list
    """
    dt_list_sort = sorted(dt_list)
    dt_list_sort_rel = [dt - dt_list_sort[0] for dt in dt_list_sort]
    avg_timedelta = sum(dt_list_sort_rel, timedelta())/len(dt_list_sort_rel)
    return dt_list_sort[0] + avg_timedelta

def _serialize_json(obj, fp):
    """ Serialize ``obj`` as a JSON formatted stream to ``fp`` """
    json.dump(obj, fp, indent=4, default=serialize)

def is_valid_url(url):
    """Checks if a given string is an url"""
    pieces = urlparse(url)
    return all([pieces.scheme, pieces.netloc])

def fmt_duration(secs):
    """Format a duration in seconds."""
    return ' '.join(fmt.human_duration(secs, 0, precision=2, short=True).strip().split())

def bytesize(arr):
    """
    Returns the memory byte size of a Numpy array as an integer.
    """
    byte_size = np.prod(arr.shape) * np.dtype(arr.dtype).itemsize
    return byte_size

def format_float(value): # not used
    """Modified form of the 'g' format specifier.
    """
    string = "{:g}".format(value).replace("e+", "e")
    string = re.sub("e(-?)0*(\d+)", r"e\1\2", string)
    return string

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

def covstr(s):
  """ convert string to int or float. """
  try:
    ret = int(s)
  except ValueError:
    ret = float(s)
  return ret

def _system_parameters(**kwargs):
    """
    Returns system keyword arguments removing Nones.

    Args:
        kwargs: system keyword arguments.

    Returns:
        dict: system keyword arguments.
    """
    return {key: value for key, value in kwargs.items()
            if (value is not None or value == {})}

def _from_bytes(bytes, byteorder="big", signed=False):
    """This is the same functionality as ``int.from_bytes`` in python 3"""
    return int.from_bytes(bytes, byteorder=byteorder, signed=signed)

def is_string(val):
    """Determines whether the passed value is a string, safe for 2/3."""
    try:
        basestring
    except NameError:
        return isinstance(val, str)
    return isinstance(val, basestring)

def _match_literal(self, a, b=None):
        """Match two names."""

        return a.lower() == b if not self.case_sensitive else a == b

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

def is_hex_string(string):
    """Check if the string is only composed of hex characters."""
    pattern = re.compile(r'[A-Fa-f0-9]+')
    if isinstance(string, six.binary_type):
        string = str(string)
    return pattern.match(string) is not None

def local_accuracy(X_train, y_train, X_test, y_test, attr_test, model_generator, metric, trained_model):
    """ The how well do the features plus a constant base rate sum up to the model output.
    """

    X_train, X_test = to_array(X_train, X_test)

    # how many features to mask
    assert X_train.shape[1] == X_test.shape[1]

    # keep nkeep top features and re-train the model for each test explanation
    yp_test = trained_model.predict(X_test)

    return metric(yp_test, strip_list(attr_test).sum(1))

def is_hex_string(string):
    """Check if the string is only composed of hex characters."""
    pattern = re.compile(r'[A-Fa-f0-9]+')
    if isinstance(string, six.binary_type):
        string = str(string)
    return pattern.match(string) is not None

def doc_to_html(doc, doc_format="ROBOT"):
    """Convert documentation to HTML"""
    from robot.libdocpkg.htmlwriter import DocToHtml
    return DocToHtml(doc_format)(doc)

def replace(scope, strings, source, dest):
    """
    Returns a copy of the given string (or list of strings) in which all
    occurrences of the given source are replaced by the given dest.

    :type  strings: string
    :param strings: A string, or a list of strings.
    :type  source: string
    :param source: What to replace.
    :type  dest: string
    :param dest: What to replace it with.
    :rtype:  string
    :return: The resulting string, or list of strings.
    """
    return [s.replace(source[0], dest[0]) for s in strings]

def parse_datetime(dt_str):
    """Parse datetime."""
    date_format = "%Y-%m-%dT%H:%M:%S %z"
    dt_str = dt_str.replace("Z", " +0000")
    return datetime.datetime.strptime(dt_str, date_format)

def c_str(string):
    """"Convert a python string to C string."""
    if not isinstance(string, str):
        string = string.decode('ascii')
    return ctypes.c_char_p(string.encode('utf-8'))

def pack_triples_numpy(triples):
    """Packs a list of triple indexes into a 2D numpy array."""
    if len(triples) == 0:
        return np.array([], dtype=np.int64)
    return np.stack(list(map(_transform_triple_numpy, triples)), axis=0)

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

def str2bytes(x):
  """Convert input argument to bytes"""
  if type(x) is bytes:
    return x
  elif type(x) is str:
    return bytes([ ord(i) for i in x ])
  else:
    return str2bytes(str(x))

def _string_hash(s):
    """String hash (djb2) with consistency between py2/py3 and persistency between runs (unlike `hash`)."""
    h = 5381
    for c in s:
        h = h * 33 + ord(c)
    return h

def visit_Str(self, node):
        """ Set the pythonic string type. """
        self.result[node] = self.builder.NamedType(pytype_to_ctype(str))

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

def coerce(self, value):
        """Convert from whatever is given to a list of scalars for the lookup_field."""
        if isinstance(value, dict):
            value = [value]
        if not isiterable_notstring(value):
            value = [value]
        return [coerce_single_instance(self.lookup_field, v) for v in value]

def drop_bad_characters(text):
    """Takes a text and drops all non-printable and non-ascii characters and
    also any whitespace characters that aren't space.

    :arg str text: the text to fix

    :returns: text with all bad characters dropped

    """
    # Strip all non-ascii and non-printable characters
    text = ''.join([c for c in text if c in ALLOWED_CHARS])
    return text

def removeFromRegistery(obj) :
	"""Removes an object/rabalist from registery. This is useful if you want to allow the garbage collector to free the memory
	taken by the objects you've already loaded. Be careful might cause some discrepenties in your scripts. For objects,
	cascades to free the registeries of related rabalists also"""
	
	if isRabaObject(obj) :
		_unregisterRabaObjectInstance(obj)
	elif isRabaList(obj) :
		_unregisterRabaListInstance(obj)

def filter_none(list_of_points):
    """
    
    :param list_of_points: 
    :return: list_of_points with None's removed
    """
    remove_elementnone = filter(lambda p: p is not None, list_of_points)
    remove_sublistnone = filter(lambda p: not contains_none(p), remove_elementnone)
    return list(remove_sublistnone)

def _unzip_handle(handle):
    """Transparently unzip the file handle"""
    if isinstance(handle, basestring):
        handle = _gzip_open_filename(handle)
    else:
        handle = _gzip_open_handle(handle)
    return handle

def fsliceafter(astr, sub):
    """Return the slice after at sub in string astr"""
    findex = astr.find(sub)
    return astr[findex + len(sub):]

def _update_texttuple(self, x, y, s, cs, d):
        """Update the text tuple at `x` and `y` with the given `s` and `d`"""
        pos = (x, y, cs)
        for i, (old_x, old_y, old_s, old_cs, old_d) in enumerate(self.value):
            if (old_x, old_y, old_cs) == pos:
                self.value[i] = (old_x, old_y, s, old_cs, d)
                return
        raise ValueError("No text tuple found at {0}!".format(pos))

def authenticate(self, transport, account_name, password=None):
        """
        Authenticates account using soap method.
        """
        Authenticator.authenticate(self, transport, account_name, password)

        if password == None:
            return self.pre_auth(transport, account_name)
        else:
            return self.auth(transport, account_name, password)

def upcaseTokens(s,l,t):
    """Helper parse action to convert tokens to upper case."""
    return [ tt.upper() for tt in map(_ustr,t) ]

def _wait_for_response(self):
		"""
		Wait until the user accepted or rejected the request
		"""
		while not self.server.response_code:
			time.sleep(2)
		time.sleep(5)
		self.server.shutdown()

def copy_and_update(dictionary, update):
    """Returns an updated copy of the dictionary without modifying the original"""
    newdict = dictionary.copy()
    newdict.update(update)
    return newdict

def init_db():
    """
    Drops and re-creates the SQL schema
    """
    db.drop_all()
    db.configure_mappers()
    db.create_all()
    db.session.commit()

def _do_auto_predict(machine, X, *args):
    """Performs an automatic prediction for the specified machine and returns
    the predicted values.
    """
    if auto_predict and hasattr(machine, "predict"):
        return machine.predict(X)

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

def to_pascal_case(s):
    """Transform underscore separated string to pascal case

    """
    return re.sub(r'(?!^)_([a-zA-Z])', lambda m: m.group(1).upper(), s.capitalize())

def row_to_dict(row):
    """Convert a table row to a dictionary."""
    o = {}
    for colname in row.colnames:

        if isinstance(row[colname], np.string_) and row[colname].dtype.kind in ['S', 'U']:
            o[colname] = str(row[colname])
        else:
            o[colname] = row[colname]

    return o

def color_string(color, string):
    """
    Colorizes a given string, if coloring is available.
    """
    if not color_available:
        return string

    return color + string + colorama.Fore.RESET

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

def get_stoplist(language):
    """Returns an built-in stop-list for the language as a set of words."""
    file_path = os.path.join("stoplists", "%s.txt" % language)
    try:
        stopwords = pkgutil.get_data("justext", file_path)
    except IOError:
        raise ValueError(
            "Stoplist for language '%s' is missing. "
            "Please use function 'get_stoplists' for complete list of stoplists "
            "and feel free to contribute by your own stoplist." % language
        )

    return frozenset(w.decode("utf8").lower() for w in stopwords.splitlines())

def create_tmpfile(self, content):
        """ Utility method to create temp files. These are cleaned at the end of the test """
        # Not using a context manager to avoid unneccessary identation in test code
        tmpfile, tmpfilepath = tempfile.mkstemp()
        self.tmpfiles.append(tmpfilepath)
        with os.fdopen(tmpfile, "w") as f:
            f.write(content)
        return tmpfilepath

def aandb(a, b):
    """Return a matrix of logic comparison of A or B"""
    return matrix(np.logical_and(a, b).astype('float'), a.size)

def json_template(data, template_name, template_context):
    """Old style, use JSONTemplateResponse instead of this.
    """
    html = render_to_string(template_name, template_context)
    data = data or {}
    data['html'] = html
    return HttpResponse(json_encode(data), content_type='application/json')

def find(self, *args, **kwargs):
        """Same as :meth:`pymongo.collection.Collection.find`, except
        it returns the right document class.
        """
        return Cursor(self, *args, wrap=self.document_class, **kwargs)

def unfolding(tens, i):
    """Compute the i-th unfolding of a tensor."""
    return reshape(tens.full(), (np.prod(tens.n[0:(i+1)]), -1))

def members(self, uid="*", objects=False):
        """ members() issues an ldap query for all users, and returns a dict
            for each matching entry. This can be quite slow, and takes roughly
            3s to complete. You may optionally restrict the scope by specifying
            a uid, which is roughly equivalent to a search(uid='foo')
        """
        entries = self.search(uid='*')
        if objects:
            return self.memberObjects(entries)
        result = []
        for entry in entries:
            result.append(entry[1])
        return result

def flatten_all_but_last(a):
  """Flatten all dimensions of a except the last."""
  ret = tf.reshape(a, [-1, tf.shape(a)[-1]])
  if not tf.executing_eagerly():
    ret.set_shape([None] + a.get_shape().as_list()[-1:])
  return ret

def list_move_to_front(l,value='other'):
    """if the value is in the list, move it to the front and return it."""
    l=list(l)
    if value in l:
        l.remove(value)
        l.insert(0,value)
    return l

def afx_small():
  """Small transformer model with small batch size for fast step times."""
  hparams = transformer.transformer_tpu()
  hparams.filter_size = 1024
  hparams.num_heads = 4
  hparams.num_hidden_layers = 3
  hparams.batch_size = 512
  return hparams

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

def has_attribute(module_name, attribute_name):
    """Is this attribute present?"""
    init_file = '%s/__init__.py' % module_name
    return any(
        [attribute_name in init_line for init_line in open(init_file).readlines()]
    )

def sort_genomic_ranges(rngs):
  """sort multiple ranges"""
  return sorted(rngs, key=lambda x: (x.chr, x.start, x.end))

def is_square_matrix(mat):
    """Test if an array is a square matrix."""
    mat = np.array(mat)
    if mat.ndim != 2:
        return False
    shape = mat.shape
    return shape[0] == shape[1]

def save_session(self, sid, session, namespace=None):
        """Store the user session for a client.

        The only difference with the :func:`socketio.Server.save_session`
        method is that when the ``namespace`` argument is not given the
        namespace associated with the class is used.
        """
        return self.server.save_session(
            sid, session, namespace=namespace or self.namespace)

def is_connected(self):
        """
        Return true if the socket managed by this connection is connected

        :rtype: bool
        """
        try:
            return self.socket is not None and self.socket.getsockname()[1] != 0 and BaseTransport.is_connected(self)
        except socket.error:
            return False

def show(config):
    """Show revision list"""
    with open(config, 'r'):
        main.show(yaml.load(open(config)))

def txt_line_iterator(path):
  """Iterate through lines of file."""
  with tf.gfile.Open(path) as f:
    for line in f:
      yield line.strip()

def _validate_type_scalar(self, value):
        """ Is not a list or a dict """
        if isinstance(
            value, _int_types + (_str_type, float, date, datetime, bool)
        ):
            return True

def encode_dataset(dataset, vocabulary):
  """Encode from strings to token ids.

  Args:
    dataset: a tf.data.Dataset with string values.
    vocabulary: a mesh_tensorflow.transformer.Vocabulary
  Returns:
    a tf.data.Dataset with integer-vector values ending in EOS=1
  """
  def encode(features):
    return {k: vocabulary.encode_tf(v) for k, v in features.items()}
  return dataset.map(encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)

def generate_write_yaml_to_file(file_name):
    """ generate a method to write the configuration in yaml to the method desired """
    def write_yaml(config):
        with open(file_name, 'w+') as fh:
            fh.write(yaml.dump(config))
    return write_yaml

def Stop(self):
    """Stops the process status RPC server."""
    self._Close()

    if self._rpc_thread.isAlive():
      self._rpc_thread.join()
    self._rpc_thread = None

def is_valid_ipv6(ip_str):
    """
    Check the validity of an IPv6 address
    """
    try:
        socket.inet_pton(socket.AF_INET6, ip_str)
    except socket.error:
        return False
    return True

def flush():
    """Try to flush all stdio buffers, both from python and from C."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except (AttributeError, ValueError, IOError):
        pass  # unsupported
    try:
        libc.fflush(None)
    except (AttributeError, ValueError, IOError):
        pass

def get_list_representation(self):
        """Returns this subset's representation as a list of indices."""
        if self.is_list:
            return self.list_or_slice
        else:
            return self[list(range(self.num_examples))]

def terminate(self):
        """Terminate all workers and threads."""
        for t in self._threads:
            t.quit()
        self._thread = []
        self._workers = []

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

def write_tsv_line_from_list(linelist, outfp):
    """Utility method to convert list to tsv line with carriage return"""
    line = '\t'.join(linelist)
    outfp.write(line)
    outfp.write('\n')

def join(self):
		"""Note that the Executor must be close()'d elsewhere,
		or join() will never return.
		"""
		self.inputfeeder_thread.join()
		self.pool.join()
		self.resulttracker_thread.join()
		self.failuretracker_thread.join()

def _make_sentence(txt):
    """Make a sentence from a piece of text."""
    #Make sure first letter is capitalized
    txt = txt.strip(' ')
    txt = txt[0].upper() + txt[1:] + '.'
    return txt

def asyncStarCmap(asyncCallable, iterable):
    """itertools.starmap for deferred callables using cooperative multitasking
    """
    results = []
    yield coopStar(asyncCallable, results.append, iterable)
    returnValue(results)

def write_wav(path, samples, sr=16000):
    """
    Write to given samples to a wav file.
    The samples are expected to be floating point numbers
    in the range of -1.0 to 1.0.

    Args:
        path (str): The path to write the wav to.
        samples (np.array): A float array .
        sr (int): The sampling rate.
    """
    max_value = np.abs(np.iinfo(np.int16).min)
    data = (samples * max_value).astype(np.int16)
    scipy.io.wavfile.write(path, sr, data)

def make_directory(path):
    """
    Make a directory and any intermediate directories that don't already
    exist. This function handles the case where two threads try to create
    a directory at once.
    """
    if not os.path.exists(path):
        # concurrent writes that try to create the same dir can fail
        try:
            os.makedirs(path)

        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise e

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

def quaternion_to_rotation_matrix(quaternion):
    """Compute the rotation matrix representated by the quaternion"""
    c, x, y, z = quaternion
    return np.array([
        [c*c + x*x - y*y - z*z, 2*x*y - 2*c*z,         2*x*z + 2*c*y        ],
        [2*x*y + 2*c*z,         c*c - x*x + y*y - z*z, 2*y*z - 2*c*x        ],
        [2*x*z - 2*c*y,         2*y*z + 2*c*x,         c*c - x*x - y*y + z*z]
    ], float)

def write_file(filename, content):
    """Create the file with the given content"""
    print 'Generating {0}'.format(filename)
    with open(filename, 'wb') as out_f:
        out_f.write(content)

def ms_to_datetime(ms):
    """
    Converts a millisecond accuracy timestamp to a datetime
    """
    dt = datetime.datetime.utcfromtimestamp(ms / 1000)
    return dt.replace(microsecond=(ms % 1000) * 1000).replace(tzinfo=pytz.utc)

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

def session_to_epoch(timestamp):
    """ converts Synergy Timestamp for session to UTC zone seconds since epoch """
    utc_timetuple = datetime.strptime(timestamp, SYNERGY_SESSION_PATTERN).replace(tzinfo=None).utctimetuple()
    return calendar.timegm(utc_timetuple)

def get_own_ip():
    """Get the host's ip number.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
    except socket.gaierror:
        ip_ = "127.0.0.1"
    else:
        ip_ = sock.getsockname()[0]
    finally:
        sock.close()
    return ip_

def make_aware(dt):
    """Appends tzinfo and assumes UTC, if datetime object has no tzinfo already."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

def populate_obj(obj, attrs):
    """Populates an object's attributes using the provided dict
    """
    for k, v in attrs.iteritems():
        setattr(obj, k, v)

def timestamp_to_datetime(timestamp):
    """Convert an ARF timestamp to a datetime.datetime object (naive local time)"""
    from datetime import datetime, timedelta
    obj = datetime.fromtimestamp(timestamp[0])
    return obj + timedelta(microseconds=int(timestamp[1]))

def str2bytes(x):
  """Convert input argument to bytes"""
  if type(x) is bytes:
    return x
  elif type(x) is str:
    return bytes([ ord(i) for i in x ])
  else:
    return str2bytes(str(x))

def make_aware(dt):
    """Appends tzinfo and assumes UTC, if datetime object has no tzinfo already."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

def html(header_rows):
    """
    Convert a list of tuples describing a table into a HTML string
    """
    name = 'table%d' % next(tablecounter)
    return HtmlTable([map(str, row) for row in header_rows], name).render()

def yview(self, *args):
        """Update inplace widgets position when doing vertical scroll"""
        self.after_idle(self.__updateWnds)
        ttk.Treeview.yview(self, *args)

def is_gzipped_fastq(file_name):
    """
    Determine whether indicated file appears to be a gzipped FASTQ.

    :param str file_name: Name/path of file to check as gzipped FASTQ.
    :return bool: Whether indicated file appears to be in gzipped FASTQ format.
    """
    _, ext = os.path.splitext(file_name)
    return file_name.endswith(".fastq.gz") or file_name.endswith(".fq.gz")

def restore_scrollbar_position(self):
        """Restoring scrollbar position after main window is visible"""
        scrollbar_pos = self.get_option('scrollbar_position', None)
        if scrollbar_pos is not None:
            self.explorer.treewidget.set_scrollbar_position(scrollbar_pos)

def is_listish(obj):
    """Check if something quacks like a list."""
    if isinstance(obj, (list, tuple, set)):
        return True
    return is_sequence(obj)

def translate_fourier(image, dx):
    """ Translate an image in fourier-space with plane waves """
    N = image.shape[0]

    f = 2*np.pi*np.fft.fftfreq(N)
    kx,ky,kz = np.meshgrid(*(f,)*3, indexing='ij')
    kv = np.array([kx,ky,kz]).T

    q = np.fft.fftn(image)*np.exp(-1.j*(kv*dx).sum(axis=-1)).T
    return np.real(np.fft.ifftn(q))

def full(self):
        """Return True if the queue is full"""
        if not self.size: return False
        return len(self.pq) == (self.size + self.removed_count)

def get_pg_connection(host, user, port, password, database, ssl={}):
    """ PostgreSQL connection """

    return psycopg2.connect(host=host,
                            user=user,
                            port=port,
                            password=password,
                            dbname=database,
                            sslmode=ssl.get('sslmode', None),
                            sslcert=ssl.get('sslcert', None),
                            sslkey=ssl.get('sslkey', None),
                            sslrootcert=ssl.get('sslrootcert', None),
                            )

def defvalkey(js, key, default=None, take_none=True):
    """
    Returns js[key] if set, otherwise default. Note js[key] can be None.
    :param js:
    :param key:
    :param default:
    :param take_none:
    :return:
    """
    if js is None:
        return default
    if key not in js:
        return default
    if js[key] is None and not take_none:
        return default
    return js[key]

def get_python_dict(scala_map):
    """Return a dict from entries in a scala.collection.immutable.Map"""
    python_dict = {}
    keys = get_python_list(scala_map.keys().toList())
    for key in keys:
        python_dict[key] = scala_map.apply(key)
    return python_dict

def resize_image_to_fit_width(image, dest_w):
    """
    Resize and image to fit the passed in width, keeping the aspect ratio the same

    :param image: PIL.Image
    :param dest_w: The desired width
    """
    scale_factor = dest_w / image.size[0]
    dest_h = image.size[1] * scale_factor
    
    scaled_image = image.resize((int(dest_w), int(dest_h)), PIL.Image.ANTIALIAS)

    return scaled_image

def list(self):
        """position in 3d space"""
        return [self._pos3d.x, self._pos3d.y, self._pos3d.z]

def yn_prompt(msg, default=True):
    """
    Prompts the user for yes or no.
    """
    ret = custom_prompt(msg, ["y", "n"], "y" if default else "n")
    if ret == "y":
        return True
    return False

def GetAttributeNs(self, localName, namespaceURI):
        """Provides the value of the specified attribute """
        ret = libxml2mod.xmlTextReaderGetAttributeNs(self._o, localName, namespaceURI)
        return ret

def find_geom(geom, geoms):
    """
    Returns the index of a geometry in a list of geometries avoiding
    expensive equality checks of `in` operator.
    """
    for i, g in enumerate(geoms):
        if g is geom:
            return i

def _using_stdout(self):
        """
        Return whether the handler is using sys.stdout.
        """
        if WINDOWS and colorama:
            # Then self.stream is an AnsiToWin32 object.
            return self.stream.wrapped is sys.stdout

        return self.stream is sys.stdout

def merge(left, right, how='inner', key=None, left_key=None, right_key=None,
          left_as='left', right_as='right'):
    """ Performs a join using the union join function. """
    return join(left, right, how, key, left_key, right_key,
                join_fn=make_union_join(left_as, right_as))

def file_found(filename,force):
    """Check if a file exists"""
    if os.path.exists(filename) and not force:
        logger.info("Found %s; skipping..."%filename)
        return True
    else:
        return False

def _linearInterpolationTransformMatrix(matrix1, matrix2, value):
    """ Linear, 'oldstyle' interpolation of the transform matrix."""
    return tuple(_interpolateValue(matrix1[i], matrix2[i], value) for i in range(len(matrix1)))

def is_unix_style(flags):
    """Check if we should use Unix style."""

    return (util.platform() != "windows" or (not bool(flags & REALPATH) and get_case(flags))) and not flags & _FORCEWIN

def invertDictMapping(d):
    """ Invert mapping of dictionary (i.e. map values to list of keys) """
    inv_map = {}
    for k, v in d.items():
        inv_map[v] = inv_map.get(v, [])
        inv_map[v].append(k)
    return inv_map

def start(args):
    """Run server with provided command line arguments.
    """
    application = tornado.web.Application([(r"/run", run.get_handler(args)),
                                           (r"/status", run.StatusHandler)])
    application.runmonitor = RunMonitor()
    application.listen(args.port)
    tornado.ioloop.IOLoop.instance().start()

def generic_div(a, b):
    """Simple function to divide two numbers"""
    logger.debug('Called generic_div({}, {})'.format(a, b))
    return a / b

def scipy_sparse_to_spmatrix(A):
    """Efficient conversion from scipy sparse matrix to cvxopt sparse matrix"""
    coo = A.tocoo()
    SP = spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=A.shape)
    return SP

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

def _frombuffer(ptr, frames, channels, dtype):
    """Create NumPy array from a pointer to some memory."""
    framesize = channels * dtype.itemsize
    data = np.frombuffer(ffi.buffer(ptr, frames * framesize), dtype=dtype)
    data.shape = -1, channels
    return data

def _single_page_pdf(page):
    """Construct a single page PDF from the provided page in memory"""
    pdf = Pdf.new()
    pdf.pages.append(page)
    bio = BytesIO()
    pdf.save(bio)
    bio.seek(0)
    return bio.read()

def from_json_list(cls, api_client, data):
        """Convert a list of JSON values to a list of models
        """
        return [cls.from_json(api_client, item) for item in data]

def query_fetch_one(self, query, values):
        """
        Executes a db query, gets the first value, and closes the connection.
        """
        self.cursor.execute(query, values)
        retval = self.cursor.fetchone()
        self.__close_db()
        return retval

def walk_tree(root):
    """Pre-order depth-first"""
    yield root

    for child in root.children:
        for el in walk_tree(child):
            yield el

def listlike(obj):
    """Is an object iterable like a list (and not a string)?"""
    
    return hasattr(obj, "__iter__") \
    and not issubclass(type(obj), str)\
    and not issubclass(type(obj), unicode)

def strip_spaces(x):
    """
    Strips spaces
    :param x:
    :return:
    """
    x = x.replace(b' ', b'')
    x = x.replace(b'\t', b'')
    return x

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

def retry_on_signal(function):
    """Retries function until it doesn't raise an EINTR error"""
    while True:
        try:
            return function()
        except EnvironmentError, e:
            if e.errno != errno.EINTR:
                raise

def __iter__(self):
		"""Iterate through all elements.

		Multiple copies will be returned if they exist.
		"""
		for value, count in self.counts():
			for _ in range(count):
				yield value

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

def txt_line_iterator(path):
  """Iterate through lines of file."""
  with tf.gfile.Open(path) as f:
    for line in f:
      yield line.strip()

def dt_to_ts(value):
    """ If value is a datetime, convert to timestamp """
    if not isinstance(value, datetime):
        return value
    return calendar.timegm(value.utctimetuple()) + value.microsecond / 1000000.0

def itervalues(d, **kw):
    """Return an iterator over the values of a dictionary."""
    if not PY2:
        return iter(d.values(**kw))
    return d.itervalues(**kw)

def norm_slash(name):
    """Normalize path slashes."""

    if isinstance(name, str):
        return name.replace('/', "\\") if not is_case_sensitive() else name
    else:
        return name.replace(b'/', b"\\") if not is_case_sensitive() else name

def group_by(iterable, key_func):
    """Wrap itertools.groupby to make life easier."""
    groups = (
        list(sub) for key, sub in groupby(iterable, key_func)
    )
    return zip(groups, groups)

def Proxy(f):
  """A helper to create a proxy method in a class."""

  def Wrapped(self, *args):
    return getattr(self, f)(*args)

  return Wrapped

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

def intToBin(i):
    """ Integer to two bytes """
    # divide in two parts (bytes)
    i1 = i % 256
    i2 = int(i / 256)
    # make string (little endian)
    return i.to_bytes(2, byteorder='little')

def get_java_path():
  """Get the path of java executable"""
  java_home = os.environ.get("JAVA_HOME")
  return os.path.join(java_home, BIN_DIR, "java")

def excel_key(index):
    """create a key for index by converting index into a base-26 number, using A-Z as the characters."""
    X = lambda n: ~n and X((n // 26)-1) + chr(65 + (n % 26)) or ''
    return X(int(index))

def test_python_java_rt():
    """ Run Python test cases against Java runtime classes. """
    sub_env = {'PYTHONPATH': _build_dir()}

    log.info('Executing Python unit tests (against Java runtime classes)...')
    return jpyutil._execute_python_scripts(python_java_rt_tests,
                                           env=sub_env)

def split_into_sentences(s):
  """Split text into list of sentences."""
  s = re.sub(r"\s+", " ", s)
  s = re.sub(r"[\\.\\?\\!]", "\n", s)
  return s.split("\n")

def get_attributes(var):
    """
    Given a varaible, return the list of attributes that are available inside
    of a template
    """
    is_valid = partial(is_valid_in_template, var)
    return list(filter(is_valid, dir(var)))

def _sim_fill(r1, r2, imsize):
    """
        calculate the fill similarity over the image
    """
    bbsize = (
        (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]))
        * (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
    )
    return 1.0 - (bbsize - r1["size"] - r2["size"]) / imsize

def delimited(items, character='|'):
    """Returns a character delimited version of the provided list as a Python string"""
    return '|'.join(items) if type(items) in (list, tuple, set) else items

def is_builtin_type(tp):
    """Checks if the given type is a builtin one.
    """
    return hasattr(__builtins__, tp.__name__) and tp is getattr(__builtins__, tp.__name__)

def _timestamp_to_json_row(value):
    """Coerce 'value' to an JSON-compatible representation.

    This version returns floating-point seconds value used in row data.
    """
    if isinstance(value, datetime.datetime):
        value = _microseconds_from_datetime(value) * 1e-6
    return value

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

def parse_json_date(value):
    """
    Parses an ISO8601 formatted datetime from a string value
    """
    if not value:
        return None

    return datetime.datetime.strptime(value, JSON_DATETIME_FORMAT).replace(tzinfo=pytz.UTC)

def _to_numeric(val):
    """
    Helper function for conversion of various data types into numeric representation.
    """
    if isinstance(val, (int, float, datetime.datetime, datetime.timedelta)):
        return val
    return float(val)

def read_json(location):
    """Open and load JSON from file.

    location (Path): Path to JSON file.
    RETURNS (dict): Loaded JSON content.
    """
    location = ensure_path(location)
    with location.open('r', encoding='utf8') as f:
        return ujson.load(f)

def check_str(obj):
        """ Returns a string for various input types """
        if isinstance(obj, str):
            return obj
        if isinstance(obj, float):
            return str(int(obj))
        else:
            return str(obj)

def json_serialize(obj):
    """
    Simple generic JSON serializer for common objects.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()

    if hasattr(obj, 'id'):
        return jsonify(obj.id)

    if hasattr(obj, 'name'):
        return jsonify(obj.name)

    raise TypeError('{0} is not JSON serializable'.format(obj))

def to_pascal_case(s):
    """Transform underscore separated string to pascal case

    """
    return re.sub(r'(?!^)_([a-zA-Z])', lambda m: m.group(1).upper(), s.capitalize())

def load_schema(schema_path):
    """Prepare the api specification for request and response validation.

    :returns: a mapping from :class:`RequestMatcher` to :class:`ValidatorMap`
        for every operation in the api specification.
    :rtype: dict
    """
    with open(schema_path, 'r') as schema_file:
        schema = simplejson.load(schema_file)
    resolver = RefResolver('', '', schema.get('models', {}))
    return build_request_to_validator_map(schema, resolver)

def __init__(self, testnet=False, dryrun=False):
        """TODO doc string"""
        self.testnet = testnet
        self.dryrun = dryrun

def unique(iterable):
    """ Returns a list copy in which each item occurs only once (in-order).
    """
    seen = set()
    return [x for x in iterable if x not in seen and not seen.add(x)]

def test():
    """ Run all Tests [nose] """

    command = 'nosetests --with-coverage --cover-package=pwnurl'
    status = subprocess.call(shlex.split(command))
    sys.exit(status)

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

def assert_is_not(expected, actual, message=None, extra=None):
    """Raises an AssertionError if expected is actual."""
    assert expected is not actual, _assert_fail_message(
        message, expected, actual, "is", extra
    )

def sigterm(self, signum, frame):
        """
        These actions will be done after SIGTERM.
        """
        self.logger.warning("Caught signal %s. Stopping daemon." % signum)
        sys.exit(0)

def guess_url(url):
    """Guess if URL is a http or ftp URL.
    @param url: the URL to check
    @ptype url: unicode
    @return: url with http:// or ftp:// prepended if it's detected as
      a http respective ftp URL.
    @rtype: unicode
    """
    if url.lower().startswith("www."):
        # syntactic sugar
        return "http://%s" % url
    elif url.lower().startswith("ftp."):
        # syntactic sugar
        return "ftp://%s" % url
    return url

def normalize(v, axis=None, eps=1e-10):
  """L2 Normalize along specified axes."""
  return v / max(anorm(v, axis=axis, keepdims=True), eps)

def _unzip_handle(handle):
    """Transparently unzip the file handle"""
    if isinstance(handle, basestring):
        handle = _gzip_open_filename(handle)
    else:
        handle = _gzip_open_handle(handle)
    return handle

def finish_plot():
    """Helper for plotting."""
    plt.legend()
    plt.grid(color='0.7')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def upcaseTokens(s,l,t):
    """Helper parse action to convert tokens to upper case."""
    return [ tt.upper() for tt in map(_ustr,t) ]

def visit_BoolOp(self, node):
        """ Return type may come from any boolop operand. """
        return sum((self.visit(value) for value in node.values), [])

def download(url, encoding='utf-8'):
    """Returns the text fetched via http GET from URL, read as `encoding`"""
    import requests
    response = requests.get(url)
    response.encoding = encoding
    return response.text

def s3(ctx, bucket_name, data_file, region):
    """Use the S3 SWAG backend."""
    if not ctx.data_file:
        ctx.data_file = data_file

    if not ctx.bucket_name:
        ctx.bucket_name = bucket_name

    if not ctx.region:
        ctx.region = region

    ctx.type = 's3'

def url_read_text(url, verbose=True):
    r"""
    Directly reads text data from url
    """
    data = url_read(url, verbose)
    text = data.decode('utf8')
    return text

def afx_small():
  """Small transformer model with small batch size for fast step times."""
  hparams = transformer.transformer_tpu()
  hparams.filter_size = 1024
  hparams.num_heads = 4
  hparams.num_hidden_layers = 3
  hparams.batch_size = 512
  return hparams

def is_safe_url(url, host=None):
    """Return ``True`` if the url is a safe redirection.

    The safe redirection means that it doesn't point to a different host.
    Always returns ``False`` on an empty url.
    """
    if not url:
        return False
    netloc = urlparse.urlparse(url)[1]
    return not netloc or netloc == host

def get_last_row(dbconn, tablename, n=1, uuid=None):
    """
    Returns the last `n` rows in the table
    """
    return fetch(dbconn, tablename, n, uuid, end=True)

def url(self):
        """ The url of this window """
        with switch_window(self._browser, self.name):
            return self._browser.url

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

def get_url_args(url):
    """ Returns a dictionary from a URL params """
    url_data = urllib.parse.urlparse(url)
    arg_dict = urllib.parse.parse_qs(url_data.query)
    return arg_dict

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

def colorbar(height, length, colormap):
    """Return the channels of a colorbar.
    """
    cbar = np.tile(np.arange(length) * 1.0 / (length - 1), (height, 1))
    cbar = (cbar * (colormap.values.max() - colormap.values.min())
            + colormap.values.min())

    return colormap.colorize(cbar)

def set_empty(self, row, column):
        """Keep one of the subplots completely empty.

        :param row,column: specify the subplot.

        """
        subplot = self.get_subplot_at(row, column)
        subplot.set_empty()

def fromDict(cls, _dict):
        """ Builds instance from dictionary of properties. """
        obj = cls()
        obj.__dict__.update(_dict)
        return obj

def _requiredSize(shape, dtype):
	"""
	Determines the number of bytes required to store a NumPy array with
	the specified shape and datatype.
	"""
	return math.floor(np.prod(np.asarray(shape, dtype=np.uint64)) * np.dtype(dtype).itemsize)

def quote(self, s):
        """Return a shell-escaped version of the string s."""

        if six.PY2:
            from pipes import quote
        else:
            from shlex import quote

        return quote(s)

def normalized(vector):
    """
    Get unit vector for a given one.

    :param vector:
        Numpy vector as coordinates in Cartesian space, or an array of such.
    :returns:
        Numpy array of the same shape and structure where all vectors are
        normalized. That is, each coordinate component is divided by its
        vector's length.
    """
    length = numpy.sum(vector * vector, axis=-1)
    length = numpy.sqrt(length.reshape(length.shape + (1, )))
    return vector / length

def static_method(cls, f):
        """Decorator which dynamically binds static methods to the model for later use."""
        setattr(cls, f.__name__, staticmethod(f))
        return f

def align_to_mmap(num, round_up):
    """
    Align the given integer number to the closest page offset, which usually is 4096 bytes.

    :param round_up: if True, the next higher multiple of page size is used, otherwise
        the lower page_size will be used (i.e. if True, 1 becomes 4096, otherwise it becomes 0)
    :return: num rounded to closest page"""
    res = (num // ALLOCATIONGRANULARITY) * ALLOCATIONGRANULARITY
    if round_up and (res != num):
        res += ALLOCATIONGRANULARITY
    # END handle size
    return res

def set(self, f):
        """Call a function after a delay, unless another function is set
        in the meantime."""
        self.stop()
        self._create_timer(f)
        self.start()

def close(self):
		"""
		Send a terminate request and then disconnect from the serial device.
		"""
		if self._initialized:
			self.stop()
		self.logged_in = False
		return self.serial_h.close()

def test():
    """Interactive test run."""
    try:
        while 1:
            x, digs = input('Enter (x, digs): ')
            print x, fix(x, digs), sci(x, digs)
    except (EOFError, KeyboardInterrupt):
        pass

def dict_from_object(obj: object):
    """Convert a object into dictionary with all of its readable attributes."""

    # If object is a dict instance, no need to convert.
    return (obj if isinstance(obj, dict)
            else {attr: getattr(obj, attr)
                  for attr in dir(obj) if not attr.startswith('_')})

def _normal_prompt(self):
        """
        Flushes the prompt before requesting the input

        :return: The command line
        """
        sys.stdout.write(self.__get_ps1())
        sys.stdout.flush()
        return safe_input()

def get_flat_size(self):
        """Returns the total length of all of the flattened variables.

        Returns:
            The length of all flattened variables concatenated.
        """
        return sum(
            np.prod(v.get_shape().as_list()) for v in self.variables.values())

def read_credentials(fname):
    """
    read a simple text file from a private location to get
    username and password
    """
    with open(fname, 'r') as f:
        username = f.readline().strip('\n')
        password = f.readline().strip('\n')
    return username, password

def get_methods(*objs):
    """ Return the names of all callable attributes of an object"""
    return set(
        attr
        for obj in objs
        for attr in dir(obj)
        if not attr.startswith('_') and callable(getattr(obj, attr))
    )

def b2u(string):
    """ bytes to unicode """
    if (isinstance(string, bytes) or
        (PY2 and isinstance(string, str))):
        return string.decode('utf-8')
    return string

def recarray(self):
        """Returns data as :class:`numpy.recarray`."""
        return numpy.rec.fromrecords(self.records, names=self.names)

def validate_int(value):
    """ Integer validator """

    if value and not isinstance(value, int):
        try:
            int(str(value))
        except (TypeError, ValueError):
            raise ValidationError('not a valid number')
    return value

def list2dict(list_of_options):
    """Transforms a list of 2 element tuples to a dictionary"""
    d = {}
    for key, value in list_of_options:
        d[key] = value
    return d

def is_valid_ipv6(ip_str):
    """
    Check the validity of an IPv6 address
    """
    try:
        socket.inet_pton(socket.AF_INET6, ip_str)
    except socket.error:
        return False
    return True

def _values(self):
        """Getter for series values (flattened)"""
        return [
            val for serie in self.series for val in serie.values
            if val is not None
        ]

def check_X_y(X, y):
    """
    tool to ensure input and output data have the same number of samples

    Parameters
    ----------
    X : array-like
    y : array-like

    Returns
    -------
    None
    """
    if len(X) != len(y):
        raise ValueError('Inconsistent input and output data shapes. '\
                         'found X: {} and y: {}'.format(X.shape, y.shape))

def open_json(file_name):
    """
    returns json contents as string
    """
    with open(file_name, "r") as json_data:
        data = json.load(json_data)
        return data

def AsPrimitiveProto(self):
    """Return an old style protocol buffer object."""
    if self.protobuf:
      result = self.protobuf()
      result.ParseFromString(self.SerializeToString())
      return result

def sav_to_pandas_rpy2(input_file):
    """
    SPSS .sav files to Pandas DataFrame through Rpy2

    :param input_file: string

    :return:
    """
    import pandas.rpy.common as com

    w = com.robj.r('foreign::read.spss("%s", to.data.frame=TRUE)' % input_file)
    return com.convert_robj(w)

def is_timestamp(instance):
    """Validates data is a timestamp"""
    if not isinstance(instance, (int, str)):
        return True
    return datetime.fromtimestamp(int(instance))

def ln_norm(x, mu, sigma=1.0):
    """ Natural log of scipy norm function truncated at zero """
    return np.log(stats.norm(loc=mu, scale=sigma).pdf(x))

def iter_except_top_row_tcs(self):
        """Generate each `a:tc` element in non-first rows of range."""
        for tr in self._tbl.tr_lst[self._top + 1:self._bottom]:
            for tc in tr.tc_lst[self._left:self._right]:
                yield tc

def ln_norm(x, mu, sigma=1.0):
    """ Natural log of scipy norm function truncated at zero """
    return np.log(stats.norm(loc=mu, scale=sigma).pdf(x))

def web(host, port):
    """Start web application"""
    from .webserver.web import get_app
    get_app().run(host=host, port=port)

def tokenize(string):
    """Match and yield all the tokens of the input string."""
    for match in TOKENS_REGEX.finditer(string):
        yield Token(match.lastgroup, match.group().strip(), match.span())

def get_type(self):
        """Get the type of the item.

        :return: the type of the item.
        :returntype: `unicode`"""
        item_type = self.xmlnode.prop("type")
        if not item_type:
            item_type = "?"
        return item_type.decode("utf-8")

def _is_osx_107():
    """
    :return:
        A bool if the current machine is running OS X 10.7
    """

    if sys.platform != 'darwin':
        return False
    version = platform.mac_ver()[0]
    return tuple(map(int, version.split('.')))[0:2] == (10, 7)

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

def magnitude(X):
    """Magnitude of a complex matrix."""
    r = np.real(X)
    i = np.imag(X)
    return np.sqrt(r * r + i * i);

def swap_memory():
    """Swap system memory as a (total, used, free, sin, sout) tuple."""
    mem = _psutil_mswindows.get_virtual_mem()
    total = mem[2]
    free = mem[3]
    used = total - free
    percent = usage_percent(used, total, _round=1)
    return nt_swapmeminfo(total, used, free, percent, 0, 0)

def cross_product_matrix(vec):
    """Returns a 3x3 cross-product matrix from a 3-element vector."""
    return np.array([[0, -vec[2], vec[1]],
                     [vec[2], 0, -vec[0]],
                     [-vec[1], vec[0], 0]])

def from_pb(cls, pb):
        """Instantiate the object from a protocol buffer.

        Args:
            pb (protobuf)

        Save a reference to the protocol buffer on the object.
        """
        obj = cls._from_pb(pb)
        obj._pb = pb
        return obj

def make_executable(script_path):
    """Make `script_path` executable.

    :param script_path: The file to change
    """
    status = os.stat(script_path)
    os.chmod(script_path, status.st_mode | stat.S_IEXEC)

def _write_json(file, contents):
    """Write a dict to a JSON file."""
    with open(file, 'w') as f:
        return json.dump(contents, f, indent=2, sort_keys=True)

def dedupe(items):
    """Remove duplicates from a sequence (of hashable items) while maintaining
    order. NOTE: This only works if items in the list are hashable types.

    Taken from the Python Cookbook, 3rd ed. Such a great book!

    """
    seen = set()
    for item in items:
        if item not in seen:
            yield item
            seen.add(item)

def set_property(self, key, value):
        """
        Update only one property in the dict
        """
        self.properties[key] = value
        self.sync_properties()

def as_float_array(a):
    """View the quaternion array as an array of floats

    This function is fast (of order 1 microsecond) because no data is
    copied; the returned quantity is just a "view" of the original.

    The output view has one more dimension (of size 4) than the input
    array, but is otherwise the same shape.

    """
    return np.asarray(a, dtype=np.quaternion).view((np.double, 4))

def _write_color_colorama (fp, text, color):
    """Colorize text with given color."""
    foreground, background, style = get_win_color(color)
    colorama.set_console(foreground=foreground, background=background,
      style=style)
    fp.write(text)
    colorama.reset_console()

def zeros(self, name, **kwargs):
        """Create an array. Keyword arguments as per
        :func:`zarr.creation.zeros`."""
        return self._write_op(self._zeros_nosync, name, **kwargs)

def save_dict_to_file(filename, dictionary):
  """Saves dictionary as CSV file."""
  with open(filename, 'w') as f:
    writer = csv.writer(f)
    for k, v in iteritems(dictionary):
      writer.writerow([str(k), str(v)])

def add_noise(Y, sigma):
    """Adds noise to Y"""
    return Y + np.random.normal(0, sigma, Y.shape)

def average_gradient(data, *kwargs):
    """ Compute average gradient norm of an image
    """
    return np.average(np.array(np.gradient(data))**2)

def stopwatch_now():
    """Get a timevalue for interval comparisons

    When possible it is a monotonic clock to prevent backwards time issues.
    """
    if six.PY2:
        now = time.time()
    else:
        now = time.monotonic()
    return now

def safe_dump(data, stream=None, **kwds):
    """implementation of safe dumper using Ordered Dict Yaml Dumper"""
    return yaml.dump(data, stream=stream, Dumper=ODYD, **kwds)

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

def load_yaml_file(file_path: str):
    """Load a YAML file from path"""
    with codecs.open(file_path, 'r') as f:
        return yaml.safe_load(f)

def safe_unicode(string):
    """If Python 2, replace non-ascii characters and return encoded string."""
    if not PY3:
        uni = string.replace(u'\u2019', "'")
        return uni.encode('utf-8')
        
    return string

def yaml_to_param(obj, name):
	"""
	Return the top-level element of a document sub-tree containing the
	YAML serialization of a Python object.
	"""
	return from_pyvalue(u"yaml:%s" % name, unicode(yaml.dump(obj)))

def url_fix_common_typos (url):
    """Fix common typos in given URL like forgotten colon."""
    if url.startswith("http//"):
        url = "http://" + url[6:]
    elif url.startswith("https//"):
        url = "https://" + url[7:]
    return url

def yaml(self):
        """
        returns the yaml output of the dict.
        """
        return ordered_dump(OrderedDict(self),
                            Dumper=yaml.SafeDumper,
                            default_flow_style=False)

def matshow(*args, **kwargs):
    """
    imshow without interpolation like as matshow
    :param args:
    :param kwargs:
    :return:
    """
    kwargs['interpolation'] = kwargs.pop('interpolation', 'none')
    return plt.imshow(*args, **kwargs)

def extract_all(zipfile, dest_folder):
    """
    reads the zip file, determines compression
    and unzips recursively until source files 
    are extracted 
    """
    z = ZipFile(zipfile)
    print(z)
    z.extract(dest_folder)

def handle_m2m(self, sender, instance, **kwargs):
    """ Handle many to many relationships """
    self.handle_save(instance.__class__, instance)

def extract(self, destination):
        """Extract the archive."""
        with zipfile.ZipFile(self.archive, 'r') as zip_ref:
            zip_ref.extractall(destination)

def get_python_dict(scala_map):
    """Return a dict from entries in a scala.collection.immutable.Map"""
    python_dict = {}
    keys = get_python_list(scala_map.keys().toList())
    for key in keys:
        python_dict[key] = scala_map.apply(key)
    return python_dict

def compress(data, **kwargs):
    """zlib.compress(data, **kwargs)
    
    """ + zopfli.__COMPRESSOR_DOCSTRING__  + """
    Returns:
      String containing a zlib container
    """
    kwargs['gzip_mode'] = 0
    return zopfli.zopfli.compress(data, **kwargs)

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

def init_mq(self):
        """Init connection and consumer with openstack mq."""
        mq = self.init_connection()
        self.init_consumer(mq)
        return mq.connection

def dot_product(self, other):
        """ Return the dot product of the given vectors. """
        return self.x * other.x + self.y * other.y

def is_real_floating_dtype(dtype):
    """Return ``True`` if ``dtype`` is a real floating point type."""
    dtype = np.dtype(dtype)
    return np.issubsctype(getattr(dtype, 'base', None), np.floating)

def max(self):
        """
        Returns the maximum value of the domain.

        :rtype: `float` or `np.inf`
        """
        return int(self._max) if not np.isinf(self._max) else self._max

def bash(filename):
    """Runs a bash script in the local directory"""
    sys.stdout.flush()
    subprocess.call("bash {}".format(filename), shell=True)

def log_loss(preds, labels):
    """Logarithmic loss with non-necessarily-binary labels."""
    log_likelihood = np.sum(labels * np.log(preds)) / len(preds)
    return -log_likelihood

def list_of_lists_to_dict(l):
    """ Convert list of key,value lists to dict

    [['id', 1], ['id', 2], ['id', 3], ['foo': 4]]
    {'id': [1, 2, 3], 'foo': [4]}
    """
    d = {}
    for key, val in l:
        d.setdefault(key, []).append(val)
    return d

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

def list2dict(list_of_options):
    """Transforms a list of 2 element tuples to a dictionary"""
    d = {}
    for key, value in list_of_options:
        d[key] = value
    return d

def simulate(self):
        """Generates a random integer in the available range."""
        min_ = (-sys.maxsize - 1) if self._min is None else self._min
        max_ = sys.maxsize if self._max is None else self._max
        return random.randint(min_, max_)

def in_directory(path):
    """Context manager (with statement) that changes the current directory
    during the context.
    """
    curdir = os.path.abspath(os.curdir)
    os.chdir(path)
    yield
    os.chdir(curdir)

def median(data):
    """Calculate the median of a list."""
    data.sort()
    num_values = len(data)
    half = num_values // 2
    if num_values % 2:
        return data[half]
    return 0.5 * (data[half-1] + data[half])

def filter_(stream_spec, filter_name, *args, **kwargs):
    """Alternate name for ``filter``, so as to not collide with the
    built-in python ``filter`` operator.
    """
    return filter(stream_spec, filter_name, *args, **kwargs)

def append_pdf(input_pdf: bytes, output_writer: PdfFileWriter):
    """
    Appends a PDF to a pyPDF writer. Legacy interface.
    """
    append_memory_pdf_to_writer(input_pdf=input_pdf,
                                writer=output_writer)

def date_to_timestamp(date):
    """
        date to unix timestamp in milliseconds
    """
    date_tuple = date.timetuple()
    timestamp = calendar.timegm(date_tuple) * 1000
    return timestamp

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

def FindMethodByName(self, name):
    """Searches for the specified method, and returns its descriptor."""
    for method in self.methods:
      if name == method.name:
        return method
    return None

def mock_decorator(*args, **kwargs):
    """Mocked decorator, needed in the case we need to mock a decorator"""
    def _called_decorator(dec_func):
        @wraps(dec_func)
        def _decorator(*args, **kwargs):
            return dec_func()
        return _decorator
    return _called_decorator

def flatten_list(l):
    """ Nested lists to single-level list, does not split strings"""
    return list(chain.from_iterable(repeat(x,1) if isinstance(x,str) else x for x in l))

def add_matplotlib_cmap(cm, name=None):
    """Add a matplotlib colormap."""
    global cmaps
    cmap = matplotlib_to_ginga_cmap(cm, name=name)
    cmaps[cmap.name] = cmap

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

def find_one(self, query):
        """Find one wrapper with conversion to dictionary

        :param dict query: A Mongo query
        """
        mongo_response = yield self.collection.find_one(query)
        raise Return(self._obj_cursor_to_dictionary(mongo_response))

def loadb(b):
    """Deserialize ``b`` (instance of ``bytes``) to a Python object."""
    assert isinstance(b, (bytes, bytearray))
    return std_json.loads(b.decode('utf-8'))

def test_replace_colon():
    """py.test for replace_colon"""
    data = (("zone:aap", '@', "zone@aap"),# s, r, replaced
    )    
    for s, r, replaced in data:
        result = replace_colon(s, r)
        assert result == replaced

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

def main(ctx, connection):
    """Command line interface for PyBEL."""
    ctx.obj = Manager(connection=connection)
    ctx.obj.bind()

def getfirstline(file, default):
    """
    Returns the first line of a file.
    """
    with open(file, 'rb') as fh:
        content = fh.readlines()
        if len(content) == 1:
            return content[0].decode('utf-8').strip('\n')

    return default

def maybeparens(lparen, item, rparen):
    """Wrap an item in optional parentheses, only applying them if necessary."""
    return item | lparen.suppress() + item + rparen.suppress()

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

def _check_and_convert_bools(self):
        """Replace boolean variables by the characters 'F'/'T'
        """
        replacements = {
            True: 'T',
            False: 'F',
        }

        for key in self.bools:
            if isinstance(self[key], bool):
                self[key] = replacements[self[key]]

def __exit__(self, *exc):
        """Exit the runtime context. This will end the transaction."""
        if exc[0] is None and exc[1] is None and exc[2] is None:
            self.commit()
        else:
            self.rollback()

def stringify_dict_contents(dct):
    """Turn dict keys and values into native strings."""
    return {
        str_if_nested_or_str(k): str_if_nested_or_str(v)
        for k, v in dct.items()
    }

def up(self):
        
        """Moves the layer up in the stacking order.
        
        """
        
        i = self.index()
        if i != None:
            del self.canvas.layers[i]
            i = min(len(self.canvas.layers), i+1)
            self.canvas.layers.insert(i, self)

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

def set_locale(request):
    """Return locale from GET lang param or automatically."""
    return request.query.get('lang', app.ps.babel.select_locale_by_request(request))

def is_power_of_2(num):
    """Return whether `num` is a power of two"""
    log = math.log2(num)
    return int(log) == float(log)

def _qrcode_to_file(qrcode, out_filepath):
    """ Save a `qrcode` object into `out_filepath`.
    Parameters
    ----------
    qrcode: qrcode object

    out_filepath: str
        Path to the output file.
    """
    try:
        qrcode.save(out_filepath)
    except Exception as exc:
        raise IOError('Error trying to save QR code file {}.'.format(out_filepath)) from exc
    else:
        return qrcode

def prepare(doc):
    """Sets the caption_found and plot_found variables to False."""
    doc.caption_found = False
    doc.plot_found = False
    doc.listings_counter = 0

def copy(obj):
    def copy(self):
        """
        Copy self to a new object.
        """
        from copy import deepcopy

        return deepcopy(self)
    obj.copy = copy
    return obj

def shot_noise(x, severity=1):
  """Shot noise corruption to images.

  Args:
    x: numpy array, uncorrupted image, assumed to have uint8 pixel in [0,255].
    severity: integer, severity of corruption.

  Returns:
    numpy array, image with uint8 pixels in [0,255]. Added shot noise.
  """
  c = [60, 25, 12, 5, 3][severity - 1]
  x = np.array(x) / 255.
  x_clip = np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255
  return around_and_astype(x_clip)

def positive_integer(anon, obj, field, val):
    """
    Returns a random positive integer (for a Django PositiveIntegerField)
    """
    return anon.faker.positive_integer(field=field)

def _normalize(image):
  """Normalize the image to zero mean and unit variance."""
  offset = tf.constant(MEAN_RGB, shape=[1, 1, 3])
  image -= offset

  scale = tf.constant(STDDEV_RGB, shape=[1, 1, 3])
  image /= scale
  return image

def sometimesish(fn):
    """
    Has a 50/50 chance of calling a function
    """
    def wrapped(*args, **kwargs):
        if random.randint(1, 2) == 1:
            return fn(*args, **kwargs)

    return wrapped

def log_normalize(data):
    """Perform log transform log(x + 1).
    
    Parameters
    ----------
    data : array_like
    
    """
    if sp.issparse(data):
        data = data.copy()
        data.data = np.log2(data.data + 1)
        return data

    return np.log2(data.astype(np.float64) + 1)

def runiform(lower, upper, size=None):
    """
    Random uniform variates.
    """
    return np.random.uniform(lower, upper, size)

def decode_arr(data):
    """Extract a numpy array from a base64 buffer"""
    data = data.encode('utf-8')
    return frombuffer(base64.b64decode(data), float64)

def get_idx_rect(index_list):
    """Extract the boundaries from a list of indexes"""
    rows, cols = list(zip(*[(i.row(), i.column()) for i in index_list]))
    return ( min(rows), max(rows), min(cols), max(cols) )

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

def input(self,pin):
        """Read the specified pin and return HIGH/true if the pin is pulled high,
        or LOW/false if pulled low.
        """
        return self.mraa_gpio.Gpio.read(self.mraa_gpio.Gpio(pin))

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

def _iter_keys(key):
    """! Iterate over subkeys of a key
    """
    for i in range(winreg.QueryInfoKey(key)[0]):
        yield winreg.OpenKey(key, winreg.EnumKey(key, i))

def shape(self):
        """Compute the shape of the dataset as (rows, cols)."""
        if not self.data:
            return (0, 0)
        return (len(self.data), len(self.dimensions))

def _parse_config(config_file_path):
    """ Parse Config File from yaml file. """
    config_file = open(config_file_path, 'r')
    config = yaml.load(config_file)
    config_file.close()
    return config

def local_accuracy(X_train, y_train, X_test, y_test, attr_test, model_generator, metric, trained_model):
    """ The how well do the features plus a constant base rate sum up to the model output.
    """

    X_train, X_test = to_array(X_train, X_test)

    # how many features to mask
    assert X_train.shape[1] == X_test.shape[1]

    # keep nkeep top features and re-train the model for each test explanation
    yp_test = trained_model.predict(X_test)

    return metric(yp_test, strip_list(attr_test).sum(1))

def load_data(filename):
    """
    :rtype : numpy matrix
    """
    data = pandas.read_csv(filename, header=None, delimiter='\t', skiprows=9)
    return data.as_matrix()

def one_hot2string(arr, vocab):
    """Convert a one-hot encoded array back to string
    """
    tokens = one_hot2token(arr)
    indexToLetter = _get_index_dict(vocab)

    return [''.join([indexToLetter[x] for x in row]) for row in tokens]

def lambda_from_file(python_file):
    """
    Reads a python file and returns a awslambda.Code object
    :param python_file:
    :return:
    """
    lambda_function = []
    with open(python_file, 'r') as f:
        lambda_function.extend(f.read().splitlines())

    return awslambda.Code(ZipFile=(Join('\n', lambda_function)))

def reverse_code_map(self):
        """Return a map from a code ( usually a string ) to the  shorter numeric value"""

        return {c.value: (c.ikey if c.ikey else c.key) for c in self.codes}

def html_to_text(content):
    """ Converts html content to plain text """
    text = None
    h2t = html2text.HTML2Text()
    h2t.ignore_links = False
    text = h2t.handle(content)
    return text

def _loadfilepath(self, filepath, **kwargs):
        """This loads a geojson file into a geojson python
        dictionary using the json module.
        
        Note: to load with a different text encoding use the encoding argument.
        """
        with open(filepath, "r") as f:
            data = json.load(f, **kwargs)
        return data

def get(url):
    """Recieving the JSON file from uulm"""
    response = urllib.request.urlopen(url)
    data = response.read()
    data = data.decode("utf-8")
    data = json.loads(data)
    return data

def cli(yamlfile, root, format):
    """ Generate CSV/TSV file from biolink model """
    print(CsvGenerator(yamlfile, format).serialize(classes=root))

def read_credentials(fname):
    """
    read a simple text file from a private location to get
    username and password
    """
    with open(fname, 'r') as f:
        username = f.readline().strip('\n')
        password = f.readline().strip('\n')
    return username, password

def dimensions(path):
    """Get width and height of a PDF"""
    pdf = PdfFileReader(path)
    size = pdf.getPage(0).mediaBox
    return {'w': float(size[2]), 'h': float(size[3])}

def list_apis(awsclient):
    """List APIs in account."""
    client_api = awsclient.get_client('apigateway')

    apis = client_api.get_rest_apis()['items']

    for api in apis:
        print(json2table(api))

def execfile(fname, variables):
    """ This is builtin in python2, but we have to roll our own on py3. """
    with open(fname) as f:
        code = compile(f.read(), fname, 'exec')
        exec(code, variables)

def get_code(module):
    """
    Compile and return a Module's code object.
    """
    fp = open(module.path)
    try:
        return compile(fp.read(), str(module.name), 'exec')
    finally:
        fp.close()

def fopenat(base_fd, path):
    """
    Does openat read-only, then does fdopen to get a file object
    """

    return os.fdopen(openat(base_fd, path, os.O_RDONLY), 'rb')

def aux_insertTree(childTree, parentTree):
	"""This a private (You shouldn't have to call it) recursive function that inserts a child tree into a parent tree."""
	if childTree.x1 != None and childTree.x2 != None :
		parentTree.insert(childTree.x1, childTree.x2, childTree.name, childTree.referedObject)

	for c in childTree.children:
		aux_insertTree(c, parentTree)

def do_serial(self, p):
		"""Set the serial port, e.g.: /dev/tty.usbserial-A4001ib8"""
		try:
			self.serial.port = p
			self.serial.open()
			print 'Opening serial port: %s' % p
		except Exception, e:
			print 'Unable to open serial port: %s' % p

def flat_list(lst):
    """This function flatten given nested list.
    Argument:
        nested list
    Returns:
        flat list
    """
    if isinstance(lst, list):
        for item in lst:
            for i in flat_list(item):
                yield i
    else:
        yield lst

def get_feature_order(dataset, features):
    """ Returns a list with the order that features requested appear in
    dataset """
    all_features = dataset.get_feature_names()

    i = [all_features.index(f) for f in features]

    return i

def circ_permutation(items):
    """Calculate the circular permutation for a given list of items."""
    permutations = []
    for i in range(len(items)):
        permutations.append(items[i:] + items[:i])
    return permutations

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

def redirect(view=None, url=None, **kwargs):
    """Redirects to the specified view or url
    """
    if view:
        if url:
            kwargs["url"] = url
        url = flask.url_for(view, **kwargs)
    current_context.exit(flask.redirect(url))

def zero_pad(m, n=1):
    """Pad a matrix with zeros, on all sides."""
    return np.pad(m, (n, n), mode='constant', constant_values=[0])

def get(self, key):  
        """ get a set of keys from redis """
        res = self.connection.get(key)
        print(res)
        return res

def zero_pad(m, n=1):
    """Pad a matrix with zeros, on all sides."""
    return np.pad(m, (n, n), mode='constant', constant_values=[0])

def format_screen(strng):
    """Format a string for screen printing.

    This removes some latex-type format codes."""
    # Paragraph continue
    par_re = re.compile(r'\\$',re.MULTILINE)
    strng = par_re.sub('',strng)
    return strng

def old_pad(s):
    """
    Pads an input string to a given block size.
    :param s: string
    :returns: The padded string.
    """
    if len(s) % OLD_BLOCK_SIZE == 0:
        return s

    return Padding.appendPadding(s, blocksize=OLD_BLOCK_SIZE)

def path(self):
        """Return the project path (aka project root)

        If ``package.__file__`` is ``/foo/foo/__init__.py``, then project.path
        should be ``/foo``.
        """
        return pathlib.Path(self.package.__file__).resolve().parent.parent

def get_max(qs, field):
    """
    get max for queryset.

    qs: queryset
    field: The field name to max.
    """
    max_field = '%s__max' % field
    num = qs.aggregate(Max(field))[max_field]
    return num if num else 0

def case_us2mc(x):
    """ underscore to mixed case notation """
    return re.sub(r'_([a-z])', lambda m: (m.group(1).upper()), x)

def remove(self, path):
        """Remove remote file
        Return:
            bool: true or false"""
        p = self.cmd('shell', 'rm', path)
        stdout, stderr = p.communicate()
        if stdout or stderr:
            return False
        else:
            return True

def endline_semicolon_check(self, original, loc, tokens):
        """Check for semicolons at the end of lines."""
        return self.check_strict("semicolon at end of line", original, loc, tokens)

def parse(self):
        """
        Parse file specified by constructor.
        """
        f = open(self.parse_log_path, "r")
        self.parse2(f)
        f.close()

def get_absolute_path(*args):
    """Transform relative pathnames into absolute pathnames."""
    directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(directory, *args)

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

def unaccentuate(s):
    """ Replace accentuated chars in string by their non accentuated equivalent. """
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def def_linear(fun):
    """Flags that a function is linear wrt all args"""
    defjvp_argnum(fun, lambda argnum, g, ans, args, kwargs:
                  fun(*subval(args, argnum, g), **kwargs))

def add_to_parser(self, parser):
        """
        Adds the argument to an argparse.ArgumentParser instance

        @param parser An argparse.ArgumentParser instance
        """
        kwargs = self._get_kwargs()
        args = self._get_args()
        parser.add_argument(*args, **kwargs)

def random_choice(sequence):
    """ Same as :meth:`random.choice`, but also supports :class:`set` type to be passed as sequence. """
    return random.choice(tuple(sequence) if isinstance(sequence, set) else sequence)

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

def strip_columns(tab):
    """Strip whitespace from string columns."""
    for colname in tab.colnames:
        if tab[colname].dtype.kind in ['S', 'U']:
            tab[colname] = np.core.defchararray.strip(tab[colname])

def export_all(self):
		query = """
			SELECT quote, library, logid
			from quotes
			left outer join quote_log on quotes.quoteid = quote_log.quoteid
			"""
		fields = 'text', 'library', 'log_id'
		return (dict(zip(fields, res)) for res in self.db.execute(query))

def seq_to_str(obj, sep=","):
    """
    Given a sequence convert it to a comma separated string.
    If, however, the argument is a single object, return its string
    representation.
    """
    if isinstance(obj, string_classes):
        return obj
    elif isinstance(obj, (list, tuple)):
        return sep.join([str(x) for x in obj])
    else:
        return str(obj)

def _precision_recall(y_true, y_score, ax=None):
    """
    Plot precision-recall curve.

    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Correct target values (ground truth).
    y_score : array-like, shape = [n_samples]
        Target scores (estimator predictions).
    ax : matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    average_precision = average_precision_score(y_true, y_score)

    if ax is None:
        ax = plt.gca()

    ax.plot(recall, precision, label=('Precision-Recall curve: AUC={0:0.2f}'
                                      .format(average_precision)))
    _set_ax_settings(ax)
    return ax

def dedup_list(l):
    """Given a list (l) will removing duplicates from the list,
       preserving the original order of the list. Assumes that
       the list entrie are hashable."""
    dedup = set()
    return [ x for x in l if not (x in dedup or dedup.add(x))]

def get_lons_from_cartesian(x__, y__):
    """Get longitudes from cartesian coordinates.
    """
    return rad2deg(arccos(x__ / sqrt(x__ ** 2 + y__ ** 2))) * sign(y__)

def remove_dups(seq):
    """remove duplicates from a sequence, preserving order"""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def format_prettytable(table):
    """Converts SoftLayer.CLI.formatting.Table instance to a prettytable."""
    for i, row in enumerate(table.rows):
        for j, item in enumerate(row):
            table.rows[i][j] = format_output(item)

    ptable = table.prettytable()
    ptable.hrules = prettytable.FRAME
    ptable.horizontal_char = '.'
    ptable.vertical_char = ':'
    ptable.junction_char = ':'
    return ptable

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

def timespan(start_time):
    """Return time in milliseconds from start_time"""

    timespan = datetime.datetime.now() - start_time
    timespan_ms = timespan.total_seconds() * 1000
    return timespan_ms

def _remove_dict_keys_with_value(dict_, val):
  """Removes `dict` keys which have have `self` as value."""
  return {k: v for k, v in dict_.items() if v is not val}

def pylog(self, *args, **kwargs):
        """Display all available logging information."""
        printerr(self.name, args, kwargs, traceback.format_exc())

def strip_accents(s):
    """
    Strip accents to prepare for slugification.
    """
    nfkd = unicodedata.normalize('NFKD', unicode(s))
    return u''.join(ch for ch in nfkd if not unicodedata.combining(ch))

def indented_show(text, howmany=1):
        """Print a formatted indented text.
        """
        print(StrTemplate.pad_indent(text=text, howmany=howmany))

def key_to_metric(self, key):
        """Replace all non-letter characters with underscores"""
        return ''.join(l if l in string.letters else '_' for l in key)

def get_ram(self, format_ = "nl"):
		"""
			return a string representations of the ram
		"""
		ram = [self.ram.read(i) for i in range(self.ram.size)]
		return self._format_mem(ram, format_)

def strip_querystring(url):
    """Remove the querystring from the end of a URL."""
    p = six.moves.urllib.parse.urlparse(url)
    return p.scheme + "://" + p.netloc + p.path

def getTypeStr(_type):
  r"""Gets the string representation of the given type.
  """
  if isinstance(_type, CustomType):
    return str(_type)

  if hasattr(_type, '__name__'):
    return _type.__name__

  return ''

def dedupe_list(seq):
    """
    Utility function to remove duplicates from a list
    :param seq: The sequence (list) to deduplicate
    :return: A list with original duplicates removed
    """
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]

async def sysinfo(dev: Device):
    """Print out system information (version, MAC addrs)."""
    click.echo(await dev.get_system_info())
    click.echo(await dev.get_interface_information())

def remove_series(self, series):
        """Removes a :py:class:`.Series` from the chart.

        :param Series series: The :py:class:`.Series` to remove.
        :raises ValueError: if you try to remove the last\
        :py:class:`.Series`."""

        if len(self.all_series()) == 1:
            raise ValueError("Cannot remove last series from %s" % str(self))
        self._all_series.remove(series)
        series._chart = None

def _screen(self, s, newline=False):
        """Print something on screen when self.verbose == True"""
        if self.verbose:
            if newline:
                print(s)
            else:
                print(s, end=' ')

def lowstrip(term):
    """Convert to lowercase and strip spaces"""
    term = re.sub('\s+', ' ', term)
    term = term.lower()
    return term

def _screen(self, s, newline=False):
        """Print something on screen when self.verbose == True"""
        if self.verbose:
            if newline:
                print(s)
            else:
                print(s, end=' ')

def to_str(s):
    """
    Convert bytes and non-string into Python 3 str
    """
    if isinstance(s, bytes):
        s = s.decode('utf-8')
    elif not isinstance(s, str):
        s = str(s)
    return s

def printOut(value, end='\n'):
    """
    This function prints the given String immediately and flushes the output.
    """
    sys.stdout.write(value)
    sys.stdout.write(end)
    sys.stdout.flush()

def normalize_value(text):
    """
    This removes newlines and multiple spaces from a string.
    """
    result = text.replace('\n', ' ')
    result = re.subn('[ ]{2,}', ' ', result)[0]
    return result

def IndexOfNth(s, value, n):
    """Gets the index of Nth occurance of a given character in a string

    :param str s:
        Input string
    :param char value:
        Input char to be searched.
    :param int n:
        Nth occurrence of char to be searched.

    :return:
        Index of the Nth occurrence in the string.
    :rtype: int

    """
    remaining = n
    for i in xrange(0, len(s)):
        if s[i] == value:
            remaining -= 1
            if remaining == 0:
                return i
    return -1

def remove_file_from_s3(awsclient, bucket, key):
    """Remove a file from an AWS S3 bucket.

    :param awsclient:
    :param bucket:
    :param key:
    :return:
    """
    client_s3 = awsclient.get_client('s3')
    response = client_s3.delete_object(Bucket=bucket, Key=key)

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

def dedup_list(l):
    """Given a list (l) will removing duplicates from the list,
       preserving the original order of the list. Assumes that
       the list entrie are hashable."""
    dedup = set()
    return [ x for x in l if not (x in dedup or dedup.add(x))]

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

def _idx_col2rowm(d):
    """Generate indexes to change from col-major to row-major ordering"""
    if 0 == len(d):
        return 1
    if 1 == len(d):
        return np.arange(d[0])
    # order='F' indicates column-major ordering
    idx = np.array(np.arange(np.prod(d))).reshape(d, order='F').T
    return idx.flatten(order='F')

def get_memory_usage():
    """Gets RAM memory usage

    :return: MB of memory used by this process
    """
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss
    return mem / (1024 * 1024)

def _replace_token_range(tokens, start, end, replacement):
    """For a range indicated from start to end, replace with replacement."""
    tokens = tokens[:start] + replacement + tokens[end:]
    return tokens

def print_out(self, *lst):
      """ Print list of strings to the predefined stdout. """
      self.print2file(self.stdout, True, True, *lst)

def _replace_nan(a, val):
    """
    replace nan in a by val, and returns the replaced array and the nan
    position
    """
    mask = isnull(a)
    return where_method(val, mask, a), mask

def parse(self):
        """Parses data in table

        :return: List of list of values in table
        """
        data = []  # add name of section

        for row in self.soup.find_all("tr"):  # cycle through all rows
            parsed = self._parse_row(row)
            if parsed:
                data.append(parsed)

        return data

def stringify_col(df, col_name):
    """
    Take a dataframe and string-i-fy a column of values.
    Turn nan/None into "" and all other values into strings.

    Parameters
    ----------
    df : dataframe
    col_name : string
    """
    df = df.copy()
    df[col_name] = df[col_name].fillna("")
    df[col_name] = df[col_name].astype(str)
    return df

def add_0x(string):
    """Add 0x to string at start.
    """
    if isinstance(string, bytes):
        string = string.decode('utf-8')
    return '0x' + str(string)

def parse_form(self, req, name, field):
        """Pull a form value from the request."""
        return get_value(req.body_arguments, name, field)

def warn_deprecated(message, stacklevel=2):  # pragma: no cover
    """Warn deprecated."""

    warnings.warn(
        message,
        category=DeprecationWarning,
        stacklevel=stacklevel
    )

def resize_by_area(img, size):
  """image resize function used by quite a few image problems."""
  return tf.to_int64(
      tf.image.resize_images(img, [size, size], tf.image.ResizeMethod.AREA))

def _get_points(self):
        """
        Subclasses may override this method.
        """
        return tuple([self._getitem__points(i)
                     for i in range(self._len__points())])

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

def raw_print(*args, **kw):
    """Raw print to sys.__stdout__, otherwise identical interface to print()."""

    print(*args, sep=kw.get('sep', ' '), end=kw.get('end', '\n'),
          file=sys.__stdout__)
    sys.__stdout__.flush()

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

def popup(self, title, callfn, initialdir=None):
        """Let user select a directory."""
        super(DirectorySelection, self).popup(title, callfn, initialdir)

def im2mat(I):
    """Converts and image to matrix (one pixel per line)"""
    return I.reshape((I.shape[0] * I.shape[1], I.shape[2]))

def command_py2to3(args):
    """
    Apply '2to3' tool (Python2 to Python3 conversion tool) to Python sources.
    """
    from lib2to3.main import main
    sys.exit(main("lib2to3.fixes", args=args.sources))

def toBase64(s):
    """Represent string / bytes s as base64, omitting newlines"""
    if isinstance(s, str):
        s = s.encode("utf-8")
    return binascii.b2a_base64(s)[:-1]

def get_soup(page=''):
    """
    Returns a bs4 object of the page requested
    """
    content = requests.get('%s/%s' % (BASE_URL, page)).text
    return BeautifulSoup(content)

def make_file_read_only(file_path):
    """
    Removes the write permissions for the given file for owner, groups and others.

    :param file_path: The file whose privileges are revoked.
    :raise FileNotFoundError: If the given file does not exist.
    """
    old_permissions = os.stat(file_path).st_mode
    os.chmod(file_path, old_permissions & ~WRITE_PERMISSIONS)

def __is_bound_method(method):
    """Return ``True`` if the `method` is a bound method (attached to an class
    instance.

    Args:
        method: A method or function type object.
    """
    if not(hasattr(method, "__func__") and hasattr(method, "__self__")):
        return False

    # Bound methods have a __self__ attribute pointing to the owner instance
    return six.get_method_self(method) is not None

def get_column_keys_and_names(table):
    """
    Return a generator of tuples k, c such that k is the name of the python attribute for
    the column and c is the name of the column in the sql table.
    """
    ins = inspect(table)
    return ((k, c.name) for k, c in ins.mapper.c.items())

def clean_url(url):
        """URL Validation function"""
        if not url.startswith(('http://', 'https://')):
            url = f'http://{url}'

        if not URL_RE.match(url):
            raise BadURLException(f'{url} is not valid')

        return url

def get_max(qs, field):
    """
    get max for queryset.

    qs: queryset
    field: The field name to max.
    """
    max_field = '%s__max' % field
    num = qs.aggregate(Max(field))[max_field]
    return num if num else 0

def stft_magnitude(signal, fft_length,
                   hop_length=None,
                   window_length=None):
  """Calculate the short-time Fourier transform magnitude.

  Args:
    signal: 1D np.array of the input time-domain signal.
    fft_length: Size of the FFT to apply.
    hop_length: Advance (in samples) between each frame passed to FFT.
    window_length: Length of each block of samples to pass to FFT.

  Returns:
    2D np.array where each row contains the magnitudes of the fft_length/2+1
    unique values of the FFT for the corresponding frame of input samples.
  """
  frames = frame(signal, window_length, hop_length)
  # Apply frame window to each frame. We use a periodic Hann (cosine of period
  # window_length) instead of the symmetric Hann of np.hanning (period
  # window_length-1).
  window = periodic_hann(window_length)
  windowed_frames = frames * window
  return np.abs(np.fft.rfft(windowed_frames, int(fft_length)))

def make_unique_ngrams(s, n):
    """Make a set of unique n-grams from a string."""
    return set(s[i:i + n] for i in range(len(s) - n + 1))

def fmt_duration(secs):
    """Format a duration in seconds."""
    return ' '.join(fmt.human_duration(secs, 0, precision=2, short=True).strip().split())

def vowels(self):
        """
        Return a new IPAString, containing only the vowels in the current string.

        :rtype: IPAString
        """
        return IPAString(ipa_chars=[c for c in self.ipa_chars if c.is_vowel])

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

def _is_leap_year(year):
    """Determine if a year is leap year.

    Parameters
    ----------
    year : numeric

    Returns
    -------
    isleap : array of bools
    """
    isleap = ((np.mod(year, 4) == 0) &
              ((np.mod(year, 100) != 0) | (np.mod(year, 400) == 0)))
    return isleap

def first_sunday(self, year, month):
        """Get the first sunday of a month."""
        date = datetime(year, month, 1, 0)
        days_until_sunday = 6 - date.weekday()

        return date + timedelta(days=days_until_sunday)

def series_index(self, series):
        """
        Return the integer index of *series* in this sequence.
        """
        for idx, s in enumerate(self):
            if series is s:
                return idx
        raise ValueError('series not in chart data object')

def _remove_keywords(d):
    """
    copy the dict, filter_keywords

    Parameters
    ----------
    d : dict
    """
    return { k:v for k, v in iteritems(d) if k not in RESERVED }

def right_outer(self):
        """
            Performs Right Outer Join
            :return right_outer: dict
        """
        self.get_collections_data()
        right_outer_join = self.merge_join_docs(
            set(self.collections_data['right'].keys()))
        return right_outer_join

def input_int_default(question="", default=0):
    """A function that works for both, Python 2.x and Python 3.x.
       It asks the user for input and returns it as a string.
    """
    answer = input_string(question)
    if answer == "" or answer == "yes":
        return default
    else:
        return int(answer)

async def result_processor(tasks):
    """An async result aggregator that combines all the results
       This gets executed in unsync.loop and unsync.thread"""
    output = {}
    for task in tasks:
        num, res = await task
        output[num] = res
    return output

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

def getTypeStr(_type):
  r"""Gets the string representation of the given type.
  """
  if isinstance(_type, CustomType):
    return str(_type)

  if hasattr(_type, '__name__'):
    return _type.__name__

  return ''

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

def __matches(s1, s2, ngrams_fn, n=3):
    """
        Returns the n-grams that match between two sequences

        See also: SequenceMatcher.get_matching_blocks

        Args:
            s1: a string
            s2: another string
            n: an int for the n in n-gram

        Returns:
            set:
    """
    ngrams1, ngrams2 = set(ngrams_fn(s1, n=n)), set(ngrams_fn(s2, n=n))
    return ngrams1.intersection(ngrams2)

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def parse_timestamp(timestamp):
    """Parse ISO8601 timestamps given by github API."""
    dt = dateutil.parser.parse(timestamp)
    return dt.astimezone(dateutil.tz.tzutc())

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

def redirect_output(fileobj):
    """Redirect standard out to file."""
    old = sys.stdout
    sys.stdout = fileobj
    try:
        yield fileobj
    finally:
        sys.stdout = old

def round_to_int(number, precision):
    """Round a number to a precision"""
    precision = int(precision)
    rounded = (int(number) + precision / 2) // precision * precision
    return rounded

def intround(value):
    """Given a float returns a rounded int. Should give the same result on
    both Py2/3
    """

    return int(decimal.Decimal.from_float(
        value).to_integral_value(decimal.ROUND_HALF_EVEN))

def paste(xsel=False):
    """Returns system clipboard contents."""
    selection = "primary" if xsel else "clipboard"
    try:
        return subprocess.Popen(["xclip", "-selection", selection, "-o"], stdout=subprocess.PIPE).communicate()[0].decode("utf-8")
    except OSError as why:
        raise XclipNotFound

def round_array(array_in):
    """
    arr_out = round_array(array_in)

    Rounds an array and recasts it to int. Also works on scalars.
    """
    if isinstance(array_in, ndarray):
        return np.round(array_in).astype(int)
    else:
        return int(np.round(array_in))

def normalize(self, string):
        """Normalize the string according to normalization list"""
        return ''.join([self._normalize.get(x, x) for x in nfd(string)])

def lower_ext(abspath):
    """Convert file extension to lowercase.
    """
    fname, ext = os.path.splitext(abspath)
    return fname + ext.lower()

def _linear_interpolation(x, X, Y):
    """Given two data points [X,Y], linearly interpolate those at x.
    """
    return (Y[1] * (x - X[0]) + Y[0] * (X[1] - x)) / (X[1] - X[0])

def __call__(self, func, *args, **kwargs):
        """Shorcut for self.run."""
        return self.run(func, *args, **kwargs)

def abort(err):
    """Abort everything, everywhere."""
    if _debug: abort._debug("abort %r", err)
    global local_controllers

    # tell all the local controllers to abort
    for controller in local_controllers.values():
        controller.abort(err)

def test():  # pragma: no cover
    """Execute the unit tests on an installed copy of unyt.

    Note that this function requires pytest to run. If pytest is not
    installed this function will raise ImportError.
    """
    import pytest
    import os

    pytest.main([os.path.dirname(os.path.abspath(__file__))])

def storeByteArray(self, context, page, len, data, returnError):
        """please override"""
        returnError.contents.value = self.IllegalStateError
        raise NotImplementedError("You must override this method.")

def test(ctx, all=False, verbose=False):
    """Run the tests."""
    cmd = 'tox' if all else 'py.test'
    if verbose:
        cmd += ' -v'
    return ctx.run(cmd, pty=True).return_code

def array(self):
        """
        return the underlying numpy array
        """
        return np.arange(self.start, self.stop, self.step)

def test(nose_argsuments):
    """ Run application tests """
    from nose import run

    params = ['__main__', '-c', 'nose.ini']
    params.extend(nose_argsuments)
    run(argv=params)

def value(self):
        """Value of property."""
        if self._prop.fget is None:
            raise AttributeError('Unable to read attribute')
        return self._prop.fget(self._obj)

def _save_file(self, filename, contents):
        """write the html file contents to disk"""
        with open(filename, 'w') as f:
            f.write(contents)

def requests_post(url, data=None, json=None, **kwargs):
    """Requests-mock requests.post wrapper."""
    return requests_request('post', url, data=data, json=json, **kwargs)

def resetScale(self):
        """Resets the scale on this image. Correctly aligns time scale, undoes manual scaling"""
        self.img.scale(1./self.imgScale[0], 1./self.imgScale[1])
        self.imgScale = (1.,1.)

def prepend_line(filepath, line):
    """Rewrite a file adding a line to its beginning.
    """
    with open(filepath) as f:
        lines = f.readlines()

    lines.insert(0, line)

    with open(filepath, 'w') as f:
        f.writelines(lines)

def compile_filter(bpf_filter, iface=None):
    """Asks Tcpdump to parse the filter, then build the matching
    BPF bytecode using get_bpf_pointer.
    """
    if not TCPDUMP:
        raise Scapy_Exception("tcpdump is not available. Cannot use filter !")
    try:
        process = subprocess.Popen([
            conf.prog.tcpdump,
            "-p",
            "-i", (conf.iface if iface is None else iface),
            "-ddd",
            "-s", str(MTU),
            bpf_filter],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except OSError as ex:
        raise Scapy_Exception("Failed to attach filter: %s" % ex)
    lines, err = process.communicate()
    ret = process.returncode
    if ret:
        raise Scapy_Exception(
            "Failed to attach filter: tcpdump returned: %s" % err
        )
    lines = lines.strip().split(b"\n")
    return get_bpf_pointer(lines)

def plot_target(target, ax):
    """Ajoute la target au plot"""
    ax.scatter(target[0], target[1], target[2], c="red", s=80)

def append(self, item):
        """ append item and print it to stdout """
        print(item)
        super(MyList, self).append(item)

def get_duckduckgo_links(limit, params, headers):
	"""
	function to fetch links equal to limit

	duckduckgo pagination is not static, so there is a limit on
	maximum number of links that can be scraped
	"""
	resp = s.get('https://duckduckgo.com/html', params = params, headers = headers)
	links = scrape_links(resp.content, engine = 'd')
	return links[:limit]

def add_noise(Y, sigma):
    """Adds noise to Y"""
    return Y + np.random.normal(0, sigma, Y.shape)

def main_func(args=None):
    """Main funcion when executing this module as script

    :param args: commandline arguments
    :type args: list
    :returns: None
    :rtype: None
    :raises: None
    """
    # we have to initialize a gui even if we dont need one right now.
    # as soon as you call maya.standalone.initialize(), a QApplication
    # with type Tty is created. This is the type for conosle apps.
    # Because i have not found a way to replace that, we just init the gui.
    guimain.init_gui()

    main.init()
    launcher = Launcher()
    parsed, unknown = launcher.parse_args(args)
    parsed.func(parsed, unknown)

def set_property(self, key, value):
        """
        Update only one property in the dict
        """
        self.properties[key] = value
        self.sync_properties()

def eqstr(a, b):
    """
    Determine whether two strings are equivalent.

    http://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/eqstr_c.html

    :param a: Arbitrary character string.
    :type a: str
    :param b: Arbitrary character string.
    :type b: str
    :return: True if A and B are equivalent.
    :rtype: bool
    """
    return bool(libspice.eqstr_c(stypes.stringToCharP(a), stypes.stringToCharP(b)))

def __dir__(self):
        u"""Returns a list of children and available helper methods."""
        return sorted(self.keys() | {m for m in dir(self.__class__) if m.startswith('to_')})

def ask_dir(self):
		"""
		dialogue box for choosing directory
		"""
		args ['directory'] = askdirectory(**self.dir_opt) 
		self.dir_text.set(args ['directory'])

def add_argument(self, dest, nargs=1, obj=None):
        """Adds a positional argument named `dest` to the parser.

        The `obj` can be used to identify the option in the order list
        that is returned from the parser.
        """
        if obj is None:
            obj = dest
        self._args.append(Argument(dest=dest, nargs=nargs, obj=obj))

def get_last(self, table=None):
        """Just the last entry."""
        if table is None: table = self.main_table
        query = 'SELECT * FROM "%s" ORDER BY ROWID DESC LIMIT 1;' % table
        return self.own_cursor.execute(query).fetchone()

def Min(a, axis, keep_dims):
    """
    Min reduction op.
    """
    return np.amin(a, axis=axis if not isinstance(axis, np.ndarray) else tuple(axis),
                   keepdims=keep_dims),

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

def build_parser():
    """Build argument parsers."""

    parser = argparse.ArgumentParser("Release packages to pypi")
    parser.add_argument('--check', '-c', action="store_true", help="Do a dry run without uploading")
    parser.add_argument('component', help="The component to release as component-version")
    return parser

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

def ma(self):
        """Represent data as a masked array.

        The array is returned with column-first indexing, i.e. for a data file with
        columns X Y1 Y2 Y3 ... the array a will be a[0] = X, a[1] = Y1, ... .

        inf and nan are filtered via :func:`numpy.isfinite`.
        """
        a = self.array
        return numpy.ma.MaskedArray(a, mask=numpy.logical_not(numpy.isfinite(a)))

async def _send_plain_text(self, request: Request, stack: Stack):
        """
        Sends plain text using `_send_text()`.
        """

        await self._send_text(request, stack, None)

def getbyteslice(self, start, end):
        """Direct access to byte data."""
        c = self._rawarray[start:end]
        return c

def strip_querystring(url):
    """Remove the querystring from the end of a URL."""
    p = six.moves.urllib.parse.urlparse(url)
    return p.scheme + "://" + p.netloc + p.path

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

def yield_connections(sock):
    """Run a server on the specified socket."""
    while True:
        log.debug('waiting for connection on %s', sock.getsockname())
        try:
            conn, _ = sock.accept()
        except KeyboardInterrupt:
            return
        conn.settimeout(None)
        log.debug('accepted connection on %s', sock.getsockname())
        yield conn

def to_dicts(recarray):
    """convert record array to a dictionaries"""
    for rec in recarray:
        yield dict(zip(recarray.dtype.names, rec.tolist()))

def set_if_empty(self, param, default):
        """ Set the parameter to the default if it doesn't exist """
        if not self.has(param):
            self.set(param, default)

def is_scalar(value):
    """Test if the given value is a scalar.

    This function also works with memory mapped array values, in contrast to the numpy is_scalar method.

    Args:
        value: the value to test for being a scalar value

    Returns:
        boolean: if the given value is a scalar or not
    """
    return np.isscalar(value) or (isinstance(value, np.ndarray) and (len(np.squeeze(value).shape) == 0))

def set_xlimits_widgets(self, set_min=True, set_max=True):
        """Populate axis limits GUI with current plot values."""
        xmin, xmax = self.tab_plot.ax.get_xlim()
        if set_min:
            self.w.x_lo.set_text('{0}'.format(xmin))
        if set_max:
            self.w.x_hi.set_text('{0}'.format(xmax))

def barray(iterlines):
    """
    Array of bytes
    """
    lst = [line.encode('utf-8') for line in iterlines]
    arr = numpy.array(lst)
    return arr

def add_matplotlib_cmap(cm, name=None):
    """Add a matplotlib colormap."""
    global cmaps
    cmap = matplotlib_to_ginga_cmap(cm, name=name)
    cmaps[cmap.name] = cmap

def _check_elements_equal(lst):
    """
    Returns true if all of the elements in the list are equal.
    """
    assert isinstance(lst, list), "Input value must be a list."
    return not lst or lst.count(lst[0]) == len(lst)

def setPixel(self, x, y, color):
        """Set the pixel at (x,y) to the integers in sequence 'color'."""
        return _fitz.Pixmap_setPixel(self, x, y, color)

def _assert_is_type(name, value, value_type):
    """Assert that a value must be a given type."""
    if not isinstance(value, value_type):
        if type(value_type) is tuple:
            types = ', '.join(t.__name__ for t in value_type)
            raise ValueError('{0} must be one of ({1})'.format(name, types))
        else:
            raise ValueError('{0} must be {1}'
                             .format(name, value_type.__name__))

def setdefaults(dct, defaults):
    """Given a target dct and a dict of {key:default value} pairs,
    calls setdefault for all of those pairs."""
    for key in defaults:
        dct.setdefault(key, defaults[key])

    return dct

async def _thread_coro(self, *args):
        """ Coroutine called by MapAsync. It's wrapping the call of
        run_in_executor to run the synchronous function as thread """
        return await self._loop.run_in_executor(
            self._executor, self._function, *args)

def get_hline():
    """ gets a horiztonal line """
    return Window(
        width=LayoutDimension.exact(1),
        height=LayoutDimension.exact(1),
        content=FillControl('-', token=Token.Line))

def StringIO(*args, **kwargs):
    """StringIO constructor shim for the async wrapper."""
    raw = sync_io.StringIO(*args, **kwargs)
    return AsyncStringIOWrapper(raw)

def _zerosamestates(self, A):
        """
        zeros out states that should be identical

        REQUIRED ARGUMENTS

        A: the matrix whose entries are to be zeroed.

        """

        for pair in self.samestates:
            A[pair[0], pair[1]] = 0
            A[pair[1], pair[0]] = 0

def get_attribute_name_id(attr):
    """
    Return the attribute name identifier
    """
    return attr.value.id if isinstance(attr.value, ast.Name) else None

def __iand__(self, other):
        """Intersect this flag with ``other`` in-place.
        """
        self.known &= other.known
        self.active &= other.active
        return self

def get_method_name(method):
    """
    Returns given method name.

    :param method: Method to retrieve the name.
    :type method: object
    :return: Method name.
    :rtype: unicode
    """

    name = get_object_name(method)
    if name.startswith("__") and not name.endswith("__"):
        name = "_{0}{1}".format(get_object_name(method.im_class), name)
    return name

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

def setup(app):
  """
  Just connects the docstring pre_processor and should_skip functions to be
  applied on all docstrings.

  """
  app.connect('autodoc-process-docstring',
              lambda *args: pre_processor(*args, namer=audiolazy_namer))
  app.connect('autodoc-skip-member', should_skip)

def log_y_cb(self, w, val):
        """Toggle linear/log scale for Y-axis."""
        self.tab_plot.logy = val
        self.plot_two_columns()

def convert_camel_case_to_snake_case(name):
    """Convert CamelCase to snake_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def array_size(x, axis):
  """Calculate the size of `x` along `axis` dimensions only."""
  axis_shape = x.shape if axis is None else tuple(x.shape[a] for a in axis)
  return max(numpy.prod(axis_shape), 1)

def _mean_dict(dict_list):
    """Compute the mean value across a list of dictionaries
    """
    return {k: np.array([d[k] for d in dict_list]).mean()
            for k in dict_list[0].keys()}

def fsliceafter(astr, sub):
    """Return the slice after at sub in string astr"""
    findex = astr.find(sub)
    return astr[findex + len(sub):]

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

def _aggr_mean(inList):
  """ Returns mean of non-None elements of the list
  """
  aggrSum = 0
  nonNone = 0
  for elem in inList:
    if elem != SENTINEL_VALUE_FOR_MISSING_DATA:
      aggrSum += elem
      nonNone += 1
  if nonNone != 0:
    return aggrSum / nonNone
  else:
    return None

def help(self):
        """Prints discovered resources and their associated methods. Nice when
        noodling in the terminal to wrap your head around Magento's insanity.
        """

        print('Resources:')
        print('')
        for name in sorted(self._resources.keys()):
            methods = sorted(self._resources[name]._methods.keys())
            print('{}: {}'.format(bold(name), ', '.join(methods)))

def loganalytics_data_plane_client(cli_ctx, _):
    """Initialize Log Analytics data client for use with CLI."""
    from .vendored_sdks.loganalytics import LogAnalyticsDataClient
    from azure.cli.core._profile import Profile
    profile = Profile(cli_ctx=cli_ctx)
    cred, _, _ = profile.get_login_credentials(
        resource="https://api.loganalytics.io")
    return LogAnalyticsDataClient(cred)

def _repr(obj):
    """Show the received object as precise as possible."""
    vals = ", ".join("{}={!r}".format(
        name, getattr(obj, name)) for name in obj._attribs)
    if vals:
        t = "{}(name={}, {})".format(obj.__class__.__name__, obj.name, vals)
    else:
        t = "{}(name={})".format(obj.__class__.__name__, obj.name)
    return t

def start(self):
        """Create a background thread for httpd and serve 'forever'"""
        self._process = threading.Thread(target=self._background_runner)
        self._process.start()

def out(self, output, newline=True):
        """Outputs a string to the console (stdout)."""
        click.echo(output, nl=newline)

def decode_arr(data):
    """Extract a numpy array from a base64 buffer"""
    data = data.encode('utf-8')
    return frombuffer(base64.b64decode(data), float64)

def csvpretty(csvfile: csvfile=sys.stdin):
    """ Pretty print a CSV file. """
    shellish.tabulate(csv.reader(csvfile))

def encode_batch(self, inputBatch):
        """Encodes a whole batch of input arrays, without learning."""
        X      = inputBatch
        encode = self.encode
        Y      = np.array([ encode(x) for x in X])
        return Y

def main(ctx, connection):
    """Command line interface for PyBEL."""
    ctx.obj = Manager(connection=connection)
    ctx.obj.bind()

def parse(el, typ):
    """
    Parse a ``BeautifulSoup`` element as the given type.
    """
    if not el:
        return typ()
    txt = text(el)
    if not txt:
        return typ()
    return typ(txt)

def sine_wave(frequency):
  """Emit a sine wave at the given frequency."""
  xs = tf.reshape(tf.range(_samples(), dtype=tf.float32), [1, _samples(), 1])
  ts = xs / FLAGS.sample_rate
  return tf.sin(2 * math.pi * frequency * ts)

def as_tree(context):
    """Return info about an object's members as JSON"""

    tree = _build_tree(context, 2, 1)
    if type(tree) == dict:
        tree = [tree] 
    
    return Response(content_type='application/json', body=json.dumps(tree))

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

def QA_util_datetime_to_strdate(dt):
    """
    :param dt:  pythone datetime.datetime
    :return:  1999-02-01 string type
    """
    strdate = "%04d-%02d-%02d" % (dt.year, dt.month, dt.day)
    return strdate

def filter_useless_pass(source):
    """Yield code with useless "pass" lines removed."""
    try:
        marked_lines = frozenset(useless_pass_line_numbers(source))
    except (SyntaxError, tokenize.TokenError):
        marked_lines = frozenset()

    sio = io.StringIO(source)
    for line_number, line in enumerate(sio.readlines(), start=1):
        if line_number not in marked_lines:
            yield line

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

def iget_list_column_slice(list_, start=None, stop=None, stride=None):
    """ iterator version of get_list_column """
    if isinstance(start, slice):
        slice_ = start
    else:
        slice_ = slice(start, stop, stride)
    return (row[slice_] for row in list_)

def get_bin_indices(self, values):
        """Returns index tuple in histogram of bin which contains value"""
        return tuple([self.get_axis_bin_index(values[ax_i], ax_i)
                      for ax_i in range(self.dimensions)])

def arglexsort(arrays):
    """
    Returns the indices of the lexicographical sorting
    order of the supplied arrays.
    """
    dtypes = ','.join(array.dtype.str for array in arrays)
    recarray = np.empty(len(arrays[0]), dtype=dtypes)
    for i, array in enumerate(arrays):
        recarray['f%s' % i] = array
    return recarray.argsort()

def register_view(self, view):
        """Register callbacks for button press events and selection changed"""
        super(ListViewController, self).register_view(view)
        self.tree_view.connect('button_press_event', self.mouse_click)

def indented_show(text, howmany=1):
        """Print a formatted indented text.
        """
        print(StrTemplate.pad_indent(text=text, howmany=howmany))

def Bernstein(n, k):
    """Bernstein polynomial.

    """
    coeff = binom(n, k)

    def _bpoly(x):
        return coeff * x ** k * (1 - x) ** (n - k)

    return _bpoly

def abbreviate_dashed(s):
    """Abbreviates each part of string that is delimited by a '-'."""
    r = []
    for part in s.split('-'):
        r.append(abbreviate(part))
    return '-'.join(r)

def maskIndex(self):
        """ Returns a boolean index with True if the value is masked.

            Always has the same shape as the maksedArray.data, event if the mask is a single boolan.
        """
        if isinstance(self.mask, bool):
            return np.full(self.data.shape, self.mask, dtype=np.bool)
        else:
            return self.mask

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

def split_into_words(s):
  """Split a sentence into list of words."""
  s = re.sub(r"\W+", " ", s)
  s = re.sub(r"[_0-9]+", " ", s)
  return s.split()

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

def matrix_at_check(self, original, loc, tokens):
        """Check for Python 3.5 matrix multiplication."""
        return self.check_py("35", "matrix multiplication", original, loc, tokens)

def render_template(self, source, **kwargs_context):
        r"""Render a template string using sandboxed environment.

        :param source: A string containing the page source.
        :param \*\*kwargs_context: The context associated with the page.
        :returns: The rendered template.
        """
        return self.jinja_env.from_string(source).render(kwargs_context)

def emit_db_sequence_updates(engine):
    """Set database sequence objects to match the source db

       Relevant only when generated from SQLAlchemy connection.
       Needed to avoid subsequent unique key violations after DB build."""

    if engine and engine.name == 'postgresql':
        # not implemented for other RDBMS; necessity unknown
        conn = engine.connect()
        qry = """SELECT 'SELECT last_value FROM ' || n.nspname ||
                         '.' || c.relname || ';' AS qry,
                        n.nspname || '.' || c.relname AS qual_name
                 FROM   pg_namespace n
                 JOIN   pg_class c ON (n.oid = c.relnamespace)
                 WHERE  c.relkind = 'S'"""
        for (qry, qual_name) in list(conn.execute(qry)):
            (lastval, ) = conn.execute(qry).first()
            nextval = int(lastval) + 1
            yield "ALTER SEQUENCE %s RESTART WITH %s;" % (qual_name, nextval)

def method_double_for(self, method_name):
        """Returns the method double for the provided method name, creating one if necessary.

        :param str method_name: The name of the method to retrieve a method double for.
        :return: The mapped ``MethodDouble``.
        :rtype: MethodDouble
        """

        if method_name not in self._method_doubles:
            self._method_doubles[method_name] = MethodDouble(method_name, self._target)

        return self._method_doubles[method_name]

def rgba_bytes_tuple(self, x):
        """Provides the color corresponding to value `x` in the
        form of a tuple (R,G,B,A) with int values between 0 and 255.
        """
        return tuple(int(u*255.9999) for u in self.rgba_floats_tuple(x))

def bulk_query(self, query, *multiparams):
        """Bulk insert or update."""

        with self.get_connection() as conn:
            conn.bulk_query(query, *multiparams)

def init_db():
    """
    Drops and re-creates the SQL schema
    """
    db.drop_all()
    db.configure_mappers()
    db.create_all()
    db.session.commit()

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

def _std(self,x):
        """
        Compute standard deviation with ddof degrees of freedom
        """
        return np.nanstd(x.values,ddof=self._ddof)

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

def print_env_info(key, out=sys.stderr):
    """If given environment key is defined, print it out."""
    value = os.getenv(key)
    if value is not None:
        print(key, "=", repr(value), file=out)

def is_cached(file_name):
	"""
	Check if a given file is available in the cache or not
	"""

	gml_file_path = join(join(expanduser('~'), OCTOGRID_DIRECTORY), file_name)

	return isfile(gml_file_path)

def Stop(self):
    """Stops the process status RPC server."""
    self._Close()

    if self._rpc_thread.isAlive():
      self._rpc_thread.join()
    self._rpc_thread = None

def set_cache_max(self, cache_name, maxsize, **kwargs):
        """
        Sets the maxsize attribute of the named cache
        """
        cache = self._get_cache(cache_name)
        cache.set_maxsize(maxsize, **kwargs)

def _on_release(self, event):
        """Stop dragging."""
        if self._drag_cols or self._drag_rows:
            self._visual_drag.place_forget()
            self._dragged_col = None
            self._dragged_row = None

def update_cache(self, data):
        """Update a cached value."""
        UTILS.update(self._cache, data)
        self._save_cache()

def stop_button_click_handler(self):
        """Method to handle what to do when the stop button is pressed"""
        self.stop_button.setDisabled(True)
        # Interrupt computations or stop debugging
        if not self.shellwidget._reading:
            self.interrupt_kernel()
        else:
            self.shellwidget.write_to_stdin('exit')

def angle(x, y):
    """Return the angle between vectors a and b in degrees."""
    return arccos(dot(x, y)/(norm(x)*norm(y)))*180./pi

def is_cached(file_name):
	"""
	Check if a given file is available in the cache or not
	"""

	gml_file_path = join(join(expanduser('~'), OCTOGRID_DIRECTORY), file_name)

	return isfile(gml_file_path)

def average(iterator):
    """Iterative mean."""
    count = 0
    total = 0
    for num in iterator:
        count += 1
        total += num
    return float(total)/count

def add_object(self, object):
        """Add object to db session. Only for session-centric object-database mappers."""
        if object.id is None:
            object.get_id()
        self.db.engine.save(object)

def _cal_dist2center(X, center):
    """ Calculate the SSE to the cluster center
    """
    dmemb2cen = scipy.spatial.distance.cdist(X, center.reshape(1,X.shape[1]), metric='seuclidean')
    return(np.sum(dmemb2cen))

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

def get_gzipped_contents(input_file):
    """
    Returns a gzipped version of a previously opened file's buffer.
    """
    zbuf = StringIO()
    zfile = GzipFile(mode="wb", compresslevel=6, fileobj=zbuf)
    zfile.write(input_file.read())
    zfile.close()
    return ContentFile(zbuf.getvalue())

def vec_angle(a, b):
    """
    Calculate angle between two vectors
    """
    cosang = np.dot(a, b)
    sinang = fast_norm(np.cross(a, b))
    return np.arctan2(sinang, cosang)

def decamelise(text):
    """Convert CamelCase to lower_and_underscore."""
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()

def elapsed_time_from(start_time):
    """calculate time delta from latched time and current time"""
    time_then = make_time(start_time)
    time_now = datetime.utcnow().replace(microsecond=0)
    if time_then is None:
        return
    delta_t = time_now - time_then
    return delta_t

def _to_lower_alpha_only(s):
    """Return a lowercased string with non alphabetic chars removed.

    White spaces are not to be removed."""
    s = re.sub(r'\n', ' ',  s.lower())
    return re.sub(r'[^a-z\s]', '', s)

def color_func(func_name):
    """
    Call color function base on name
    """
    if str(func_name).isdigit():
        return term_color(int(func_name))
    return globals()[func_name]

def token_list_to_text(tokenlist):
    """
    Concatenate all the text parts again.
    """
    ZeroWidthEscape = Token.ZeroWidthEscape
    return ''.join(item[1] for item in tokenlist if item[0] != ZeroWidthEscape)

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

def next (self):    # File-like object.

        """This is to support iterators over a file-like object.
        """

        result = self.readline()
        if result == self._empty_buffer:
            raise StopIteration
        return result

def get_request(self, request):
        """Sets token-based auth headers."""
        request.transport_user = self.username
        request.transport_password = self.api_key
        return request

def min_or_none(val1, val2):
    """Returns min(val1, val2) returning None only if both values are None"""
    return min(val1, val2, key=lambda x: sys.maxint if x is None else x)

def get_order(self, codes):
        """Return evidence codes in order shown in code2name."""
        return sorted(codes, key=lambda e: [self.ev2idx.get(e)])

def call_and_exit(self, cmd, shell=True):
        """Run the *cmd* and exit with the proper exit code."""
        sys.exit(subprocess.call(cmd, shell=shell))

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

def _set_scroll_v(self, *args):
        """Scroll both categories Canvas and scrolling container"""
        self._canvas_categories.yview(*args)
        self._canvas_scroll.yview(*args)

def tf2():
  """Provide the root module of a TF-2.0 API for use within TensorBoard.

  Returns:
    The root module of a TF-2.0 API, if available.

  Raises:
    ImportError: if a TF-2.0 API is not available.
  """
  # Import the `tf` compat API from this file and check if it's already TF 2.0.
  if tf.__version__.startswith('2.'):
    return tf
  elif hasattr(tf, 'compat') and hasattr(tf.compat, 'v2'):
    # As a fallback, try `tensorflow.compat.v2` if it's defined.
    return tf.compat.v2
  raise ImportError('cannot import tensorflow 2.0 API')

def _run_cmd_get_output(cmd):
    """Runs a shell command, returns console output.

    Mimics python3's subprocess.getoutput
    """
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    out, err = process.communicate()
    return out or err

def tf2():
  """Provide the root module of a TF-2.0 API for use within TensorBoard.

  Returns:
    The root module of a TF-2.0 API, if available.

  Raises:
    ImportError: if a TF-2.0 API is not available.
  """
  # Import the `tf` compat API from this file and check if it's already TF 2.0.
  if tf.__version__.startswith('2.'):
    return tf
  elif hasattr(tf, 'compat') and hasattr(tf.compat, 'v2'):
    # As a fallback, try `tensorflow.compat.v2` if it's defined.
    return tf.compat.v2
  raise ImportError('cannot import tensorflow 2.0 API')

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

def has_attribute(module_name, attribute_name):
    """Is this attribute present?"""
    init_file = '%s/__init__.py' % module_name
    return any(
        [attribute_name in init_line for init_line in open(init_file).readlines()]
    )

def is_bool_matrix(l):
    r"""Checks if l is a 2D numpy array of bools

    """
    if isinstance(l, np.ndarray):
        if l.ndim == 2 and (l.dtype == bool):
            return True
    return False

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

def loadb(b):
    """Deserialize ``b`` (instance of ``bytes``) to a Python object."""
    assert isinstance(b, (bytes, bytearray))
    return std_json.loads(b.decode('utf-8'))

def tokenize_list(self, text):
        """
        Split a text into separate words.
        """
        return [self.get_record_token(record) for record in self.analyze(text)]

def deserialize_date(string):
    """
    Deserializes string to date.

    :param string: str.
    :type string: str
    :return: date.
    :rtype: date
    """
    try:
        from dateutil.parser import parse
        return parse(string).date()
    except ImportError:
        return string

def strip_accents(text):
    """
    Strip agents from a string.
    """

    normalized_str = unicodedata.normalize('NFD', text)

    return ''.join([
        c for c in normalized_str if unicodedata.category(c) != 'Mn'])

def compose(*funcs):
    """compose a list of functions"""
    return lambda x: reduce(lambda v, f: f(v), reversed(funcs), x)

def flatten_all_but_last(a):
  """Flatten all dimensions of a except the last."""
  ret = tf.reshape(a, [-1, tf.shape(a)[-1]])
  if not tf.executing_eagerly():
    ret.set_shape([None] + a.get_shape().as_list()[-1:])
  return ret

def concat(cls, iterables):
    """
    Similar to #itertools.chain.from_iterable().
    """

    def generator():
      for it in iterables:
        for element in it:
          yield element
    return cls(generator())

def max(self):
        """
        Returns the maximum value of the domain.

        :rtype: `float` or `np.inf`
        """
        return int(self._max) if not np.isinf(self._max) else self._max

def OnCellBackgroundColor(self, event):
        """Cell background color event handler"""

        with undo.group(_("Background color")):
            self.grid.actions.set_attr("bgcolor", event.color)

        self.grid.ForceRefresh()

        self.grid.update_attribute_toolbar()

        event.Skip()

def longest_run(da, dim='time'):
    """Return the length of the longest consecutive run of True values.

        Parameters
        ----------
        arr : N-dimensional array (boolean)
          Input array
        dim : Xarray dimension (default = 'time')
          Dimension along which to calculate consecutive run

        Returns
        -------
        N-dimensional array (int)
          Length of longest run of True values along dimension
        """

    d = rle(da, dim=dim)
    rl_long = d.max(dim=dim)

    return rl_long

def build_docs(directory):
    """Builds sphinx docs from a given directory."""
    os.chdir(directory)
    process = subprocess.Popen(["make", "html"], cwd=directory)
    process.communicate()

def ex(self, cmd):
        """Execute a normal python statement in user namespace."""
        with self.builtin_trap:
            exec cmd in self.user_global_ns, self.user_ns

def _multiline_width(multiline_s, line_width_fn=len):
    """Visible width of a potentially multiline content."""
    return max(map(line_width_fn, re.split("[\r\n]", multiline_s)))

def append_pdf(input_pdf: bytes, output_writer: PdfFileWriter):
    """
    Appends a PDF to a pyPDF writer. Legacy interface.
    """
    append_memory_pdf_to_writer(input_pdf=input_pdf,
                                writer=output_writer)

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

def checkbox_uncheck(self, force_check=False):
        """
        Wrapper to uncheck a checkbox
        """
        if self.get_attribute('checked'):
            self.click(force_click=force_check)

def dict_to_numpy_array(d):
    """
    Convert a dict of 1d array to a numpy recarray
    """
    return fromarrays(d.values(), np.dtype([(str(k), v.dtype) for k, v in d.items()]))

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def __set_token_expired(self, value):
        """Internal helper for oauth code"""
        self._token_expired = datetime.datetime.now() + datetime.timedelta(seconds=value)
        return

def check_attribute_exists(instance):
    """ Additional check for the dimension model, to ensure that attributes
    given as the key and label attribute on the dimension exist. """
    attributes = instance.get('attributes', {}).keys()
    if instance.get('key_attribute') not in attributes:
        return False
    label_attr = instance.get('label_attribute')
    if label_attr and label_attr not in attributes:
        return False
    return True

def redirect_output(fileobj):
    """Redirect standard out to file."""
    old = sys.stdout
    sys.stdout = fileobj
    try:
        yield fileobj
    finally:
        sys.stdout = old

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

def dir_path(dir):
    """with dir_path(path) to change into a directory."""
    old_dir = os.getcwd()
    os.chdir(dir)
    yield
    os.chdir(old_dir)

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

def save_form(self, request, form, change):
        """
        Super class ordering is important here - user must get saved first.
        """
        OwnableAdmin.save_form(self, request, form, change)
        return DisplayableAdmin.save_form(self, request, form, change)

def check_str(obj):
        """ Returns a string for various input types """
        if isinstance(obj, str):
            return obj
        if isinstance(obj, float):
            return str(int(obj))
        else:
            return str(obj)

def gaussian_variogram_model(m, d):
    """Gaussian model, m is [psill, range, nugget]"""
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return psill * (1. - np.exp(-d**2./(range_*4./7.)**2.)) + nugget

def title(msg):
    """Sets the title of the console window."""
    if sys.platform.startswith("win"):
        ctypes.windll.kernel32.SetConsoleTitleW(tounicode(msg))

def average_gradient(data, *kwargs):
    """ Compute average gradient norm of an image
    """
    return np.average(np.array(np.gradient(data))**2)

def pprint(self, seconds):
        """
        Pretty Prints seconds as Hours:Minutes:Seconds.MilliSeconds

        :param seconds:  The time in seconds.
        """
        return ("%d:%02d:%02d.%03d", reduce(lambda ll, b: divmod(ll[0], b) + ll[1:], [(seconds * 1000,), 1000, 60, 60]))

def tokenize(string):
    """Match and yield all the tokens of the input string."""
    for match in TOKENS_REGEX.finditer(string):
        yield Token(match.lastgroup, match.group().strip(), match.span())

def _escape(s):
    """ Helper method that escapes parameters to a SQL query. """
    e = s
    e = e.replace('\\', '\\\\')
    e = e.replace('\n', '\\n')
    e = e.replace('\r', '\\r')
    e = e.replace("'", "\\'")
    e = e.replace('"', '\\"')
    return e

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

def dictify(a_named_tuple):
    """Transform a named tuple into a dictionary"""
    return dict((s, getattr(a_named_tuple, s)) for s in a_named_tuple._fields)

def isnumber(*args):
    """Checks if value is an integer, long integer or float.

    NOTE: Treats booleans as numbers, where True=1 and False=0.
    """
    return all(map(lambda c: isinstance(c, int) or isinstance(c, float), args))

def print_tree(self, indent=2):
        """ print_tree: prints out structure of tree
            Args: indent (int): What level of indentation at which to start printing
            Returns: None
        """
        config.LOGGER.info("{indent}{data}".format(indent="   " * indent, data=str(self)))
        for child in self.children:
            child.print_tree(indent + 1)

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

def strip_head(sequence, values):
    """Strips elements of `values` from the beginning of `sequence`."""
    values = set(values)
    return list(itertools.dropwhile(lambda x: x in values, sequence))

def check_create_folder(filename):
    """Check if the folder exisits. If not, create the folder"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

def to_dict(self):
        """Converts the table to a dict."""
        return {"name": self.table_name, "kind": self.table_kind, "data": [r.to_dict() for r in self]}

def has_attribute(module_name, attribute_name):
    """Is this attribute present?"""
    init_file = '%s/__init__.py' % module_name
    return any(
        [attribute_name in init_line for init_line in open(init_file).readlines()]
    )

def _crop_list_to_size(l, size):
    """Make a list a certain size"""
    for x in range(size - len(l)):
        l.append(False)
    for x in range(len(l) - size):
        l.pop()
    return l

def isin_alone(elems, line):
    """Check if an element from a list is the only element of a string.

    :type elems: list
    :type line: str

    """
    found = False
    for e in elems:
        if line.strip().lower() == e.lower():
            found = True
            break
    return found

def unit_key_from_name(name):
  """Return a legal python name for the given name for use as a unit key."""
  result = name

  for old, new in six.iteritems(UNIT_KEY_REPLACEMENTS):
    result = result.replace(old, new)

  # Collapse redundant underscores and convert to uppercase.
  result = re.sub(r'_+', '_', result.upper())

  return result

def is_iter_non_string(obj):
    """test if object is a list or tuple"""
    if isinstance(obj, list) or isinstance(obj, tuple):
        return True
    return False

def string_to_int( s ):
  """Convert a string of bytes into an integer, as per X9.62."""
  result = 0
  for c in s:
    if not isinstance(c, int): c = ord( c )
    result = 256 * result + c
  return result

def valid_uuid(value):
    """ Check if value is a valid UUID. """

    try:
        uuid.UUID(value, version=4)
        return True
    except (TypeError, ValueError, AttributeError):
        return False

def to_dataframe(products):
        """Return the products from a query response as a Pandas DataFrame
        with the values in their appropriate Python types.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("to_dataframe requires the optional dependency Pandas.")

        return pd.DataFrame.from_dict(products, orient='index')

def _get_ipv4_from_binary(self, bin_addr):
        """Converts binary address to Ipv4 format."""

        return socket.inet_ntop(socket.AF_INET, struct.pack("!L", bin_addr))

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

def dt_to_ts(value):
    """ If value is a datetime, convert to timestamp """
    if not isinstance(value, datetime):
        return value
    return calendar.timegm(value.utctimetuple()) + value.microsecond / 1000000.0

def is_valid_folder(parser, arg):
    """Check if arg is a valid file that already exists on the file system."""
    arg = os.path.abspath(arg)
    if not os.path.isdir(arg):
        parser.error("The folder %s does not exist!" % arg)
    else:
        return arg

def abort(self):
        """ ensure the master exit from Barrier """
        self.mutex.release()
        self.turnstile.release()
        self.mutex.release()
        self.turnstile2.release()

def pid_exists(pid):
    """ Determines if a system process identifer exists in process table.
        """
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    else:
        return True

def Gaussian(x, mu, sig):
    """
    Gaussian pdf.
    :param x: free variable.
    :param mu: mean of the distribution.
    :param sig: standard deviation of the distribution.
    :return: sympy.Expr for a Gaussian pdf.
    """
    return sympy.exp(-(x - mu)**2/(2*sig**2))/sympy.sqrt(2*sympy.pi*sig**2)

def All(sequence):
  """
  :param sequence: Any sequence whose elements can be evaluated as booleans.
  :returns: true if all elements of the sequence satisfy True and x.
  """
  return bool(reduce(lambda x, y: x and y, sequence, True))

def angle_v2_rad(vec_a, vec_b):
    """Returns angle between vec_a and vec_b in range [0, PI].  This does not
    distinguish if a is left of or right of b.
    """
    # cos(x) = A * B / |A| * |B|
    return math.acos(vec_a.dot(vec_b) / (vec_a.length() * vec_b.length()))

def _rectangular(n):
    """Checks to see if a 2D list is a valid 2D matrix"""
    for i in n:
        if len(i) != len(n[0]):
            return False
    return True

def txt_line_iterator(path):
  """Iterate through lines of file."""
  with tf.gfile.Open(path) as f:
    for line in f:
      yield line.strip()

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

def is_iterable_of_int(l):
    r""" Checks if l is iterable and contains only integral types """
    if not is_iterable(l):
        return False

    return all(is_int(value) for value in l)

def _to_numeric(val):
    """
    Helper function for conversion of various data types into numeric representation.
    """
    if isinstance(val, (int, float, datetime.datetime, datetime.timedelta)):
        return val
    return float(val)

def is_valid_image_extension(file_path):
    """is_valid_image_extension."""
    valid_extensions = ['.jpeg', '.jpg', '.gif', '.png']
    _, extension = os.path.splitext(file_path)
    return extension.lower() in valid_extensions

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

def gevent_monkey_patch_report(self):
        """
        Report effective gevent monkey patching on the logs.
        """
        try:
            import gevent.socket
            import socket

            if gevent.socket.socket is socket.socket:
                self.log("gevent monkey patching is active")
                return True
            else:
                self.notify_user("gevent monkey patching failed.")
        except ImportError:
            self.notify_user("gevent is not installed, monkey patching failed.")
        return False

def underscore(text):
    """Converts text that may be camelcased into an underscored format"""
    return UNDERSCORE[1].sub(r'\1_\2', UNDERSCORE[0].sub(r'\1_\2', text)).lower()

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

def camel_case_from_underscores(string):
    """generate a CamelCase string from an underscore_string."""
    components = string.split('_')
    string = ''
    for component in components:
        string += component[0].upper() + component[1:]
    return string

def is_int_vector(l):
    r"""Checks if l is a numpy array of integers

    """
    if isinstance(l, np.ndarray):
        if l.ndim == 1 and (l.dtype.kind == 'i' or l.dtype.kind == 'u'):
            return True
    return False

def test_kwargs_are_optional(self):
        """kwarg values always have defaults"""
        with patch("sys.exit") as mock_exit:
            cli = MicroCLITestCase.T("script_name f3".split()).run()
            # kwargs are optional
            mock_exit.assert_called_with(4)

def is_valid_row(cls, row):
        """Indicates whether or not the given row contains valid data."""
        for k in row.keys():
            if row[k] is None:
                return False
        return True

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

def _has_fileno(stream):
    """Returns whether the stream object seems to have a working fileno()

    Tells whether _redirect_stderr is likely to work.

    Parameters
    ----------
    stream : IO stream object

    Returns
    -------
    has_fileno : bool
        True if stream.fileno() exists and doesn't raise OSError or
        UnsupportedOperation
    """
    try:
        stream.fileno()
    except (AttributeError, OSError, IOError, io.UnsupportedOperation):
        return False
    return True

def to_capitalized_camel_case(snake_case_string):
    """
    Convert a string from snake case to camel case with the first letter capitalized. For example, "some_var"
    would become "SomeVar".

    :param snake_case_string: Snake-cased string to convert to camel case.
    :returns: Camel-cased version of snake_case_string.
    """
    parts = snake_case_string.split('_')
    return ''.join([i.title() for i in parts])

def has_multiline_items(maybe_list: Optional[Sequence[str]]):
    """Check whether one of the items in the list has multiple lines."""
    return maybe_list and any(is_multiline(item) for item in maybe_list)

def read(*args):
    """Reads complete file contents."""
    return io.open(os.path.join(HERE, *args), encoding="utf-8").read()

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

def run_func(entry):
    """Runs the function associated with the given MenuEntry."""
    if entry.func:
        if entry.args and entry.krgs:
            return entry.func(*entry.args, **entry.krgs)
        if entry.args:
            return entry.func(*entry.args)
        if entry.krgs:
            return entry.func(**entry.krgs)
        return entry.func()

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

def str_ripper(self, text):
        """Got this code from here:
        http://stackoverflow.com/questions/6116978/python-replace-multiple-strings

        This method takes a set of strings, A, and removes all whole
        elements of set A from string B.

        Input: text string to strip based on instance attribute self.censor
        Output: a stripped (censored) text string
        """
        return self.pattern.sub(lambda m: self.rep[re.escape(m.group(0))], text)

def is_local_url(target):
    """Determine if URL is safe to redirect to."""
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ('http', 'https') and \
        ref_url.netloc == test_url.netloc

def quote(self, s):
        """Return a shell-escaped version of the string s."""

        if six.PY2:
            from pipes import quote
        else:
            from shlex import quote

        return quote(s)

def valid_uuid(value):
    """ Check if value is a valid UUID. """

    try:
        uuid.UUID(value, version=4)
        return True
    except (TypeError, ValueError, AttributeError):
        return False

def getSystemVariable(self, remote, name):
        """Get single system variable from CCU / Homegear"""
        if self._server is not None:
            return self._server.getSystemVariable(remote, name)

def _is_proper_sequence(seq):
    """Returns is seq is sequence and not string."""
    return (isinstance(seq, collections.abc.Sequence) and
            not isinstance(seq, str))

def get_data_table(filename):
  """Returns a DataTable instance built from either the filename, or STDIN if filename is None."""
  with get_file_object(filename, "r") as rf:
    return DataTable(list(csv.reader(rf)))

def _is_path(s):
    """Return whether an object is a path."""
    if isinstance(s, string_types):
        try:
            return op.exists(s)
        except (OSError, ValueError):
            return False
    else:
        return False

def paste(xsel=False):
    """Returns system clipboard contents."""
    selection = "primary" if xsel else "clipboard"
    try:
        return subprocess.Popen(["xclip", "-selection", selection, "-o"], stdout=subprocess.PIPE).communicate()[0].decode("utf-8")
    except OSError as why:
        raise XclipNotFound

def is_image(filename):
    """Determine if given filename is an image."""
    # note: isfile() also accepts symlinks
    return os.path.isfile(filename) and filename.lower().endswith(ImageExts)

def is_iterable_of_int(l):
    r""" Checks if l is iterable and contains only integral types """
    if not is_iterable(l):
        return False

    return all(is_int(value) for value in l)

def list_formatter(handler, item, value):
    """Format list."""
    return u', '.join(str(v) for v in value)

def isnamedtuple(obj):
    """Heuristic check if an object is a namedtuple."""
    return isinstance(obj, tuple) \
           and hasattr(obj, "_fields") \
           and hasattr(obj, "_asdict") \
           and callable(obj._asdict)

def _text_to_graphiz(self, text):
        """create a graphviz graph from text"""
        dot = Source(text, format='svg')
        return dot.pipe().decode('utf-8')

def is_valid_url(url):
    """Checks if a given string is an url"""
    pieces = urlparse(url)
    return all([pieces.scheme, pieces.netloc])

def test_python_java_rt():
    """ Run Python test cases against Java runtime classes. """
    sub_env = {'PYTHONPATH': _build_dir()}

    log.info('Executing Python unit tests (against Java runtime classes)...')
    return jpyutil._execute_python_scripts(python_java_rt_tests,
                                           env=sub_env)

def last_modified_date(filename):
    """Last modified timestamp as a UTC datetime"""
    mtime = os.path.getmtime(filename)
    dt = datetime.datetime.utcfromtimestamp(mtime)
    return dt.replace(tzinfo=pytz.utc)

def Bernstein(n, k):
    """Bernstein polynomial.

    """
    coeff = binom(n, k)

    def _bpoly(x):
        return coeff * x ** k * (1 - x) ** (n - k)

    return _bpoly

def _check_methods(self, methods):
        """ @type methods: tuple """
        for method in methods:
            if method not in self.ALLOWED_METHODS:
                raise Exception('Invalid \'%s\' method' % method)

def timestamp_to_datetime(cls, time_stamp, localized=True):
        """ Converts a UTC timestamp to a datetime.datetime."""
        ret = datetime.datetime.utcfromtimestamp(time_stamp)
        if localized:
            ret = localize(ret, pytz.utc)
        return ret

def is_iterable_but_not_string(obj):
    """
    Determine whether or not obj is iterable but not a string (eg, a list, set, tuple etc).
    """
    return hasattr(obj, '__iter__') and not isinstance(obj, str) and not isinstance(obj, bytes)

def validate(datum, schema, field=None, raise_errors=True):
    """
    Determine if a python datum is an instance of a schema.

    Parameters
    ----------
    datum: Any
        Data being validated
    schema: dict
        Schema
    field: str, optional
        Record field being validated
    raise_errors: bool, optional
        If true, errors are raised for invalid data. If false, a simple
        True (valid) or False (invalid) result is returned


    Example::

        from fastavro.validation import validate
        schema = {...}
        record = {...}
        validate(record, schema)
    """
    record_type = extract_record_type(schema)
    result = None

    validator = VALIDATORS.get(record_type)
    if validator:
        result = validator(datum, schema=schema,
                           parent_ns=field,
                           raise_errors=raise_errors)
    elif record_type in SCHEMA_DEFS:
        result = validate(datum,
                          schema=SCHEMA_DEFS[record_type],
                          field=field,
                          raise_errors=raise_errors)
    else:
        raise UnknownType(record_type)

    if raise_errors and result is False:
        raise ValidationError(ValidationErrorData(datum, schema, field))

    return result

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

def __validate_email(self, email):
        """Checks if a string looks like an email address"""

        e = re.match(self.EMAIL_ADDRESS_REGEX, email, re.UNICODE)
        if e:
            return email
        else:
            error = "Invalid email address: " + str(email)
            msg = self.GRIMOIRELAB_INVALID_FORMAT % {'error': error}
            raise InvalidFormatError(cause=msg)

def is_readable_dir(path):
  """Returns whether a path names an existing directory we can list and read files from."""
  return os.path.isdir(path) and os.access(path, os.R_OK) and os.access(path, os.X_OK)

def email_type(arg):
	"""An argparse type representing an email address."""
	if not is_valid_email_address(arg):
		raise argparse.ArgumentTypeError("{0} is not a valid email address".format(repr(arg)))
	return arg

def pid_exists(pid):
    """ Determines if a system process identifer exists in process table.
        """
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    else:
        return True

def SchemaValidate(self, xsd):
        """Use W3C XSD schema to validate the document as it is
          processed. Activation is only possible before the first
          Read(). If @xsd is None, then XML Schema validation is
           deactivated. """
        ret = libxml2mod.xmlTextReaderSchemaValidate(self._o, xsd)
        return ret

def arg_bool(name, default=False):
    """ Fetch a query argument, as a boolean. """
    v = request.args.get(name, '')
    if not len(v):
        return default
    return v in BOOL_TRUISH

def check_auth(username, pwd):
    """This function is called to check if a username /
    password combination is valid.
    """
    cfg = get_current_config()
    return username == cfg["dashboard_httpauth"].split(
        ":")[0] and pwd == cfg["dashboard_httpauth"].split(":")[1]

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

def json_template(data, template_name, template_context):
    """Old style, use JSONTemplateResponse instead of this.
    """
    html = render_to_string(template_name, template_context)
    data = data or {}
    data['html'] = html
    return HttpResponse(json_encode(data), content_type='application/json')

def issuperset(self, other):
        """Report whether this RangeSet contains another set."""
        self._binary_sanity_check(other)
        return set.issuperset(self, other)

def dumped(text, level, indent=2):
    """Put curly brackets round an indented text"""
    return indented("{\n%s\n}" % indented(text, level + 1, indent) or "None", level, indent) + "\n"

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

def is_builtin_type(tp):
    """Checks if the given type is a builtin one.
    """
    return hasattr(__builtins__, tp.__name__) and tp is getattr(__builtins__, tp.__name__)

def is_valid_variable_name(string_to_check):
    """
    Returns whether the provided name is a valid variable name in Python

    :param string_to_check: the string to be checked
    :return: True or False
    """

    try:

        parse('{} = None'.format(string_to_check))
        return True

    except (SyntaxError, ValueError, TypeError):

        return False

def dumped(text, level, indent=2):
    """Put curly brackets round an indented text"""
    return indented("{\n%s\n}" % indented(text, level + 1, indent) or "None", level, indent) + "\n"

def is_valid_variable_name(string_to_check):
    """
    Returns whether the provided name is a valid variable name in Python

    :param string_to_check: the string to be checked
    :return: True or False
    """

    try:

        parse('{} = None'.format(string_to_check))
        return True

    except (SyntaxError, ValueError, TypeError):

        return False

def _clean_str(self, s):
        """ Returns a lowercase string with punctuation and bad chars removed
        :param s: string to clean
        """
        return s.translate(str.maketrans('', '', punctuation)).replace('\u200b', " ").strip().lower()

def assert_exactly_one_true(bool_list):
    """This method asserts that only one value of the provided list is True.

    :param bool_list: List of booleans to check
    :return: True if only one value is True, False otherwise
    """
    assert isinstance(bool_list, list)
    counter = 0
    for item in bool_list:
        if item:
            counter += 1
    return counter == 1

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

def is_admin(self):
        """Is the user a system administrator"""
        return self.role == self.roles.administrator.value and self.state == State.approved

def get_obj(ref):
    """Get object from string reference."""
    oid = int(ref)
    return server.id2ref.get(oid) or server.id2obj[oid]

def _stdin_(p):
    """Takes input from user. Works for Python 2 and 3."""
    _v = sys.version[0]
    return input(p) if _v is '3' else raw_input(p)

def check(self, var):
        """Check whether the provided value is a valid enum constant."""
        if not isinstance(var, _str_type): return False
        return _enum_mangle(var) in self._consts

def cart2pol(x, y):
    """Cartesian to Polar coordinates conversion."""
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho

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

def __eq__(self, other):
        """Determine if two objects are equal."""
        return isinstance(other, self.__class__) \
            and self._freeze() == other._freeze()

def map_wrap(f):
    """Wrap standard function to easily pass into 'map' processing.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper

def has_multiline_items(maybe_list: Optional[Sequence[str]]):
    """Check whether one of the items in the list has multiple lines."""
    return maybe_list and any(is_multiline(item) for item in maybe_list)

def method(func):
    """Wrap a function as a method."""
    attr = abc.abstractmethod(func)
    attr.__imethod__ = True
    return attr

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

def clean_url(url):
    """
    Remove params, query and fragment parts from URL so that `os.path.basename`
    and `os.path.splitext` can work correctly.

    @param url: URL to clean.
    @type url: str

    @return: Cleaned URL.
    @rtype: str
    """
    parsed = urlparse(url.strip())
    reconstructed = ParseResult(
        parsed.scheme, parsed.netloc, parsed.path,
        params='', query='', fragment='')
    return reconstructed.geturl()

def wrap(string, length, indent):
    """ Wrap a string at a line length """
    newline = "\n" + " " * indent
    return newline.join((string[i : i + length] for i in range(0, len(string), length)))

def close_all():
    """Close all open/active plotters"""
    for key, p in _ALL_PLOTTERS.items():
        p.close()
    _ALL_PLOTTERS.clear()
    return True

def write_float(self, number):
        """ Writes a float to the underlying output file as a 4-byte value. """
        buf = pack(self.byte_order + "f", number)
        self.write(buf)

def _delete_whitespace(self):
        """Delete all whitespace from the end of the line."""
        while isinstance(self._lines[-1], (self._Space, self._LineBreak,
                                           self._Indent)):
            del self._lines[-1]

def angle_between_vectors(x, y):
    """ Compute the angle between vector x and y """
    dp = dot_product(x, y)
    if dp == 0:
        return 0
    xm = magnitude(x)
    ym = magnitude(y)
    return math.acos(dp / (xm*ym)) * (180. / math.pi)

def socket_close(self):
        """Close our socket."""
        if self.sock != NC.INVALID_SOCKET:
            self.sock.close()
        self.sock = NC.INVALID_SOCKET

def head(filename, n=10):
    """ prints the top `n` lines of a file """
    with freader(filename) as fr:
        for _ in range(n):
            print(fr.readline().strip())

def _close_websocket(self):
        """Closes the websocket connection."""
        close_method = getattr(self._websocket, "close", None)
        if callable(close_method):
            asyncio.ensure_future(close_method(), loop=self._event_loop)
        self._websocket = None
        self._dispatch_event(event="close")

def pickle_data(data, picklefile):
    """Helper function to pickle `data` in `picklefile`."""
    with open(picklefile, 'wb') as f:
        pickle.dump(data, f, protocol=2)

def fit_select_best(X, y):
    """
    Selects the best fit of the estimators already implemented by choosing the
    model with the smallest mean square error metric for the trained values.
    """
    models = [fit(X,y) for fit in [fit_linear, fit_quadratic]]
    errors = map(lambda model: mse(y, model.predict(X)), models)

    return min(zip(models, errors), key=itemgetter(1))[0]

def softmax(attrs, inputs, proto_obj):
    """Softmax function."""
    if 'axis' not in attrs:
        attrs = translation_utils._add_extra_attributes(attrs, {'axis': 1})
    return 'softmax', attrs, inputs

def filtered_image(self, im):
        """Returns a filtered image after applying the Fourier-space filters"""
        q = np.fft.fftn(im)
        for k,v in self.filters:
            q[k] -= v
        return np.real(np.fft.ifftn(q))

def disable_wx(self):
        """Disable event loop integration with wxPython.

        This merely sets PyOS_InputHook to NULL.
        """
        if self._apps.has_key(GUI_WX):
            self._apps[GUI_WX]._in_event_loop = False
        self.clear_inputhook()

def underline(self, msg):
        """Underline the input"""
        return click.style(msg, underline=True) if self.colorize else msg

def on_close(self, evt):
    """
    Pop-up menu and wx.EVT_CLOSE closing event
    """
    self.stop() # DoseWatcher
    if evt.EventObject is not self: # Avoid deadlocks
      self.Close() # wx.Frame
    evt.Skip()

def afx_small():
  """Small transformer model with small batch size for fast step times."""
  hparams = transformer.transformer_tpu()
  hparams.filter_size = 1024
  hparams.num_heads = 4
  hparams.num_hidden_layers = 3
  hparams.batch_size = 512
  return hparams

def restore_scrollbar_position(self):
        """Restoring scrollbar position after main window is visible"""
        scrollbar_pos = self.get_option('scrollbar_position', None)
        if scrollbar_pos is not None:
            self.explorer.treewidget.set_scrollbar_position(scrollbar_pos)

def IPYTHON_MAIN():
    """Decide if the Ipython command line is running code."""
    import pkg_resources

    runner_frame = inspect.getouterframes(inspect.currentframe())[-2]
    return (
        getattr(runner_frame, "function", None)
        == pkg_resources.load_entry_point("ipython", "console_scripts", "ipython").__name__
    )

def series_table_row_offset(self, series):
        """
        Return the number of rows preceding the data table for *series* in
        the Excel worksheet.
        """
        title_and_spacer_rows = series.index * 2
        data_point_rows = series.data_point_offset
        return title_and_spacer_rows + data_point_rows

def is_valid_image_extension(file_path):
    """is_valid_image_extension."""
    valid_extensions = ['.jpeg', '.jpg', '.gif', '.png']
    _, extension = os.path.splitext(file_path)
    return extension.lower() in valid_extensions

def getChildElementsByTagName(self, tagName):
    """ Return child elements of type tagName if found, else [] """
    result = []
    for child in self.childNodes:
        if isinstance(child, Element):
            if child.tagName == tagName:
                result.append(child)
    return result

def mcc(y_true, y_pred, round=True):
    """Matthews correlation coefficient
    """
    y_true, y_pred = _mask_value_nan(y_true, y_pred)
    if round:
        y_true = np.round(y_true)
        y_pred = np.round(y_pred)
    return skm.matthews_corrcoef(y_true, y_pred)

def kernelDriverActive(self, interface):
        """
        Tell whether a kernel driver is active on given interface number.
        """
        result = libusb1.libusb_kernel_driver_active(self.__handle, interface)
        if result == 0:
            return False
        elif result == 1:
            return True
        raiseUSBError(result)

def schemaValidateFile(self, filename, options):
        """Do a schemas validation of the given resource, it will use
           the SAX streamable validation internally. """
        ret = libxml2mod.xmlSchemaValidateFile(self._o, filename, options)
        return ret

def clear_globals_reload_modules(self):
        """Clears globals and reloads modules"""

        self.code_array.clear_globals()
        self.code_array.reload_modules()

        # Clear result cache
        self.code_array.result_cache.clear()

def prox_zero(X, step):
    """Proximal operator to project onto zero
    """
    return np.zeros(X.shape, dtype=X.dtype)

def count_rows(self, table_name):
        """Return the number of entries in a table by counting them."""
        self.table_must_exist(table_name)
        query = "SELECT COUNT (*) FROM `%s`" % table_name.lower()
        self.own_cursor.execute(query)
        return int(self.own_cursor.fetchone()[0])

def mkdir(dir, enter):
    """Create directory with template for topic of the current environment

    """

    if not os.path.exists(dir):
        os.makedirs(dir)

def myreplace(astr, thefind, thereplace):
    """in string astr replace all occurences of thefind with thereplace"""
    alist = astr.split(thefind)
    new_s = alist.split(thereplace)
    return new_s

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

def _time_to_json(value):
    """Coerce 'value' to an JSON-compatible representation."""
    if isinstance(value, datetime.time):
        value = value.isoformat()
    return value

def cleanup_lib(self):
        """ unload the previously loaded shared library """
        if not self.using_openmp:
            #this if statement is necessary because shared libraries that use
            #OpenMP will core dump when unloaded, this is a well-known issue with OpenMP
            logging.debug('unloading shared library')
            _ctypes.dlclose(self.lib._handle)

def negate(self):
        """Reverse the range"""
        self.from_value, self.to_value = self.to_value, self.from_value
        self.include_lower, self.include_upper = self.include_upper, self.include_lower

def cfloat32_array_to_numpy(cptr, length):
    """Convert a ctypes float pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_float)):
        return np.fromiter(cptr, dtype=np.float32, count=length)
    else:
        raise RuntimeError('Expected float pointer')

def is_identifier(string):
    """Check if string could be a valid python identifier

    :param string: string to be tested
    :returns: True if string can be a python identifier, False otherwise
    :rtype: bool
    """
    matched = PYTHON_IDENTIFIER_RE.match(string)
    return bool(matched) and not keyword.iskeyword(string)

def cint8_array_to_numpy(cptr, length):
    """Convert a ctypes int pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_int8)):
        return np.fromiter(cptr, dtype=np.int8, count=length)
    else:
        raise RuntimeError('Expected int pointer')

def get_resource_attribute(self, attr):
        """Gets the resource attribute if available

        :param attr: Name of the attribute
        :return: Value of the attribute, if set in the resource. None otherwise
        """
        if attr not in self.resource_attributes:
            raise KeyError("%s is not in resource attributes" % attr)

        return self.resource_attributes[attr]

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

def run(self):
        """Run the event loop."""
        self.signal_init()
        self.listen_init()
        self.logger.info('starting')
        self.loop.start()

def _ram_buffer(self):
        """Setup the RAM buffer from the C++ code."""
        # get the address of the RAM
        address = _LIB.Memory(self._env)
        # create a buffer from the contents of the address location
        buffer_ = ctypes.cast(address, ctypes.POINTER(RAM_VECTOR)).contents
        # create a NumPy array from the buffer
        return np.frombuffer(buffer_, dtype='uint8')

def loadb(b):
    """Deserialize ``b`` (instance of ``bytes``) to a Python object."""
    assert isinstance(b, (bytes, bytearray))
    return std_json.loads(b.decode('utf-8'))

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

def _swap_curly(string):
    """Swap single and double curly brackets"""
    return (
        string.replace('{{ ', '{{').replace('{{', '\x00').replace('{', '{{')
        .replace('\x00', '{').replace(' }}', '}}').replace('}}', '\x00')
        .replace('}', '}}').replace('\x00', '}')
    )

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def now_time(str=False):
    """Get the current time."""
    if str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return datetime.datetime.now()

def _do_auto_predict(machine, X, *args):
    """Performs an automatic prediction for the specified machine and returns
    the predicted values.
    """
    if auto_predict and hasattr(machine, "predict"):
        return machine.predict(X)

def home(self):
        """Set cursor to initial position and reset any shifting."""
        self.command(c.LCD_RETURNHOME)
        self._cursor_pos = (0, 0)
        c.msleep(2)

def _getTypename(self, defn):
        """ Returns the SQL typename required to store the given FieldDefinition """
        return 'REAL' if defn.type.float or 'TIME' in defn.type.name or defn.dntoeu else 'INTEGER'

def INIT_LIST_EXPR(self, cursor):
        """Returns a list of literal values."""
        values = [self.parse_cursor(child)
                  for child in list(cursor.get_children())]
        return values

def torecarray(*args, **kwargs):
    """
    Convenient shorthand for ``toarray(*args, **kwargs).view(np.recarray)``.

    """

    import numpy as np
    return toarray(*args, **kwargs).view(np.recarray)

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, LegipyModel):
        return obj.to_json()
    elif isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    raise TypeError("Type {0} not serializable".format(repr(type(obj))))

def as_float_array(a):
    """View the quaternion array as an array of floats

    This function is fast (of order 1 microsecond) because no data is
    copied; the returned quantity is just a "view" of the original.

    The output view has one more dimension (of size 4) than the input
    array, but is otherwise the same shape.

    """
    return np.asarray(a, dtype=np.quaternion).view((np.double, 4))

def cio_close(cio):
    """Wraps openjpeg library function cio_close.
    """
    OPENJPEG.opj_cio_close.argtypes = [ctypes.POINTER(CioType)]
    OPENJPEG.opj_cio_close(cio)

def remove_trailing_string(content, trailing):
    """
    Strip trailing component `trailing` from `content` if it exists.
    Used when generating names from view classes.
    """
    if content.endswith(trailing) and content != trailing:
        return content[:-len(trailing)]
    return content

def rotate_img(im, deg, mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_AREA):
    """ Rotates an image by deg degrees

    Arguments:
        deg (float): degree to rotate.
    """
    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c//2,r//2),deg,1)
    return cv2.warpAffine(im,M,(c,r), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS+interpolation)

def format_vars(args):
    """Format the given vars in the form: 'flag=value'"""
    variables = []
    for key, value in args.items():
        if value:
            variables += ['{0}={1}'.format(key, value)]
    return variables

def kill_dashboard(self, check_alive=True):
        """Kill the dashboard.

        Args:
            check_alive (bool): Raise an exception if the process was already
                dead.
        """
        self._kill_process_type(
            ray_constants.PROCESS_TYPE_DASHBOARD, check_alive=check_alive)

def cat_acc(y_true, y_pred):
    """Categorical accuracy
    """
    return np.mean(y_true.argmax(axis=1) == y_pred.argmax(axis=1))

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

def softplus(attrs, inputs, proto_obj):
    """Applies the sofplus activation function element-wise to the input."""
    new_attrs = translation_utils._add_extra_attributes(attrs, {'act_type' : 'softrelu'})
    return 'Activation', new_attrs, inputs

def isoformat(dt):
    """Return an ISO-8601 formatted string from the provided datetime object"""
    if not isinstance(dt, datetime.datetime):
        raise TypeError("Must provide datetime.datetime object to isoformat")

    if dt.tzinfo is None:
        raise ValueError("naive datetime objects are not allowed beyond the library boundaries")

    return dt.isoformat().replace("+00:00", "Z")

def print_with_header(header, message, color, indent=0):
    """
    Use one of the functions below for printing, not this one.
    """
    print()
    padding = ' ' * indent
    print(padding + color + BOLD + header + ENDC + color + message + ENDC)

def from_timestamp(microsecond_timestamp):
    """Convert a microsecond timestamp to a UTC datetime instance."""
    # Create datetime without losing precision from floating point (yes, this
    # is actually needed):
    return datetime.datetime.fromtimestamp(
        microsecond_timestamp // 1000000, datetime.timezone.utc
    ).replace(microsecond=(microsecond_timestamp % 1000000))

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

def _DateToEpoch(date):
  """Converts python datetime to epoch microseconds."""
  tz_zero = datetime.datetime.utcfromtimestamp(0)
  diff_sec = int((date - tz_zero).total_seconds())
  return diff_sec * 1000000

def _quit(self, *args):
        """ quit crash """
        self.logger.warn('Bye!')
        sys.exit(self.exit())

def created_today(self):
        """Return True if created today."""
        if self.datetime.date() == datetime.today().date():
            return True
        return False

def monthly(date=datetime.date.today()):
    """
    Take a date object and return the first day of the month.
    """
    return datetime.date(date.year, date.month, 1)

def update_dict(obj, dict, attributes):
    """Update dict with fields from obj.attributes.

    :param obj: the object updated into dict
    :param dict: the result dictionary
    :param attributes: a list of attributes belonging to obj
    """
    for attribute in attributes:
        if hasattr(obj, attribute) and getattr(obj, attribute) is not None:
            dict[attribute] = getattr(obj, attribute)

def _default(self, obj):
        """ return a serialized version of obj or raise a TypeError

        :param obj:
        :return: Serialized version of obj
        """
        return obj.__dict__ if isinstance(obj, JsonObj) else json.JSONDecoder().decode(obj)

def getpackagepath():
    """
     *Get the root path for this python package - used in unit testing code*
    """
    moduleDirectory = os.path.dirname(__file__)
    packagePath = os.path.dirname(__file__) + "/../"

    return packagePath

def print_args(output=sys.stdout):
    """Decorate a function so that print arguments before calling it.

    Args:
      output: writable to print args. (Default: sys.stdout)
    """
    def decorator(func):
        """The decorator function.
        """
        @wraps(func)
        def _(*args, **kwargs):
            """The decorated function.
            """
            output.write(
                "Args: {0}, KwArgs: {1}\n".format(str(args), str(kwargs)))
            return func(*args, **kwargs)
        return _
    return decorator

def __add__(self,other):
        """
            If the number of columns matches, we can concatenate two LabeldMatrices
            with the + operator.
        """
        assert self.matrix.shape[1] == other.matrix.shape[1]
        return LabeledMatrix(np.concatenate([self.matrix,other.matrix],axis=0),self.labels)

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

def is_int(value):
    """Return `True` if ``value`` is an integer."""
    if isinstance(value, bool):
        return False
    try:
        int(value)
        return True
    except (ValueError, TypeError):
        return False

def generator_to_list(fn):
    """This decorator is for flat_list function.
    It converts returned generator to list.
    """
    def wrapper(*args, **kw):
        return list(fn(*args, **kw))
    return wrapper

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

def vec_angle(a, b):
    """
    Calculate angle between two vectors
    """
    cosang = np.dot(a, b)
    sinang = fast_norm(np.cross(a, b))
    return np.arctan2(sinang, cosang)

def parameter_vector(self):
        """An array of all parameters (including frozen parameters)"""
        return np.array([getattr(self, k) for k in self.parameter_names])

def position(self, x, y, text):
        """
            ANSI Escape sequences
            http://ascii-table.com/ansi-escape-sequences.php
        """
        sys.stdout.write("\x1b7\x1b[%d;%df%s\x1b8" % (x, y, text))
        sys.stdout.flush()

def del_label(self, name):
        """Delete a label by name."""
        labels_tag = self.root[0]
        labels_tag.remove(self._find_label(name))

def prepend_line(filepath, line):
    """Rewrite a file adding a line to its beginning.
    """
    with open(filepath) as f:
        lines = f.readlines()

    lines.insert(0, line)

    with open(filepath, 'w') as f:
        f.writelines(lines)

def nonull_dict(self):
        """Like dict, but does not hold any null values.

        :return:

        """
        return {k: v for k, v in six.iteritems(self.dict) if v and k != '_codes'}

def transform(self, df):
        """
        Transforms a DataFrame in place. Computes all outputs of the DataFrame.

        Args:
            df (pandas.DataFrame): DataFrame to transform.
        """
        for name, function in self.outputs:
            df[name] = function(df)

def delete(self, name):
        """Delete object on remote"""
        obj = self._get_object(name)
        if obj:
            return self.driver.delete_object(obj)

def upcaseTokens(s,l,t):
    """Helper parse action to convert tokens to upper case."""
    return [ tt.upper() for tt in map(_ustr,t) ]

def desc(self):
        """Get a short description of the device."""
        return '{0} (ID: {1}) - {2} - {3}'.format(
            self.name, self.device_id, self.type, self.status)

def filter(self, obj, *args, **kwargs):
        """
        Filter the given object through the filter chain.

        :param obj: The object to filter
        :param args: Additional arguments to pass to each filter function.
        :param kwargs: Additional keyword arguments to pass to each filter
                       function.
        :return: The filtered object or :data:`None`

        See the documentation of :class:`Filter` on how filtering operates.

        Returns the object returned by the last function in the filter chain or
        :data:`None` if any function returned :data:`None`.
        """
        for _, _, func in self._filter_order:
            obj = func(obj, *args, **kwargs)
            if obj is None:
                return None
        return obj

def fft(t, y, pow2=False, window=None, rescale=False):
    """
    FFT of y, assuming complex or real-valued inputs. This goes through the 
    numpy fourier transform process, assembling and returning (frequencies, 
    complex fft) given time and signal data y.

    Parameters
    ----------
    t,y
        Time (t) and signal (y) arrays with which to perform the fft. Note the t
        array is assumed to be evenly spaced.
        
    pow2 = False
        Set this to true if you only want to keep the first 2^n data
        points (speeds up the FFT substantially)

    window = None
        Can be set to any of the windowing functions in numpy that require only
        the number of points as the argument, e.g. window='hanning'. 
        
    rescale = False
        If True, the FFT will be rescaled by the square root of the ratio of 
        variances before and after windowing, such that the sum of component 
        amplitudes squared is equal to the actual variance.
    """
    # make sure they're numpy arrays, and make copies to avoid the referencing error
    y = _n.array(y)
    t = _n.array(t)

    # if we're doing the power of 2, do it
    if pow2:
        keep  = 2**int(_n.log2(len(y)))

        # now resize the data
        y.resize(keep)
        t.resize(keep)

    # Window the data
    if not window in [None, False, 0]:
        try:
            # Get the windowing array
            w = eval("_n."+window, dict(_n=_n))(len(y))
            
            # Store the original variance
            v0 = _n.average(abs(y)**2)
            
            # window the time domain data 
            y = y * w
            
            # Rescale by the variance ratio
            if rescale: y = y * _n.sqrt(v0 / _n.average(abs(y)**2))
            
        except:
            print("ERROR: Bad window!")
            return

    # do the actual fft, and normalize
    Y = _n.fft.fftshift( _n.fft.fft(y) / len(t) )
    f = _n.fft.fftshift( _n.fft.fftfreq(len(t), t[1]-t[0]) )
    
    return f, Y

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

def _update_index_on_df(df, index_names):
    """Helper function to restore index information after collection. Doesn't
    use self so we can serialize this."""
    if index_names:
        df = df.set_index(index_names)
        # Remove names from unnamed indexes
        index_names = _denormalize_index_names(index_names)
        df.index.names = index_names
    return df

def is_function(self):
        """return True if callback is a vanilla plain jane function"""
        if self.is_instance() or self.is_class(): return False
        return isinstance(self.callback, (Callable, classmethod))

def guess_encoding(text, default=DEFAULT_ENCODING):
    """Guess string encoding.

    Given a piece of text, apply character encoding detection to
    guess the appropriate encoding of the text.
    """
    result = chardet.detect(text)
    return normalize_result(result, default=default)

def guess_encoding(text, default=DEFAULT_ENCODING):
    """Guess string encoding.

    Given a piece of text, apply character encoding detection to
    guess the appropriate encoding of the text.
    """
    result = chardet.detect(text)
    return normalize_result(result, default=default)

def build_and_start(query, directory):
    """This function will create and then start a new Async task with the
    default callbacks argument defined in the decorator."""

    Async(target=grep, args=[query, directory]).start()

def _string_width(self, s):
        """Get width of a string in the current font"""
        s = str(s)
        w = 0
        for char in s:
            char = ord(char)
            w += self.character_widths[char]
        return w * self.font_size / 1000.0

def _mean_dict(dict_list):
    """Compute the mean value across a list of dictionaries
    """
    return {k: np.array([d[k] for d in dict_list]).mean()
            for k in dict_list[0].keys()}

def _get_column_types(self, data):
        """Get a list of the data types for each column in *data*."""
        columns = list(zip_longest(*data))
        return [self._get_column_type(column) for column in columns]

def lambda_from_file(python_file):
    """
    Reads a python file and returns a awslambda.Code object
    :param python_file:
    :return:
    """
    lambda_function = []
    with open(python_file, 'r') as f:
        lambda_function.extend(f.read().splitlines())

    return awslambda.Code(ZipFile=(Join('\n', lambda_function)))

def __contains__(self, key):
        """
        Invoked when determining whether a specific key is in the dictionary
        using `key in d`.

        The key is looked up case-insensitively.
        """
        k = self._real_key(key)
        return k in self._data

def show_xticklabels(self, row, column):
        """Show the x-axis tick labels for a subplot.

        :param row,column: specify the subplot.

        """
        subplot = self.get_subplot_at(row, column)
        subplot.show_xticklabels()

def is_subdir(a, b):
    """
    Return true if a is a subdirectory of b
    """
    a, b = map(os.path.abspath, [a, b])

    return os.path.commonpath([a, b]) == b

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

def getFileDialogTitle(msg, title):
    """
    Create nicely-formatted string based on arguments msg and title
    :param msg: the msg to be displayed
    :param title: the window title
    :return: None
    """
    if msg and title:
        return "%s - %s" % (title, msg)
    if msg and not title:
        return str(msg)
    if title and not msg:
        return str(title)
    return None

def img_encode(arr, **kwargs):
    """Encode ndarray to base64 string image data
    
    Parameters
    ----------
    arr: ndarray (rows, cols, depth)
    kwargs: passed directly to matplotlib.image.imsave
    """
    sio = BytesIO()
    imsave(sio, arr, **kwargs)
    sio.seek(0)
    img_format = kwargs['format'] if kwargs.get('format') else 'png'
    img_str = base64.b64encode(sio.getvalue()).decode()

    return 'data:image/{};base64,{}'.format(img_format, img_str)

def _check_conversion(key, valid_dict):
    """Check for existence of key in dict, return value or raise error"""
    if key not in valid_dict and key not in valid_dict.values():
        # Only show users the nice string values
        keys = [v for v in valid_dict.keys() if isinstance(v, string_types)]
        raise ValueError('value must be one of %s, not %s' % (keys, key))
    return valid_dict[key] if key in valid_dict else key

def good(txt):
    """Print, emphasized 'good', the given 'txt' message"""

    print("%s# %s%s%s" % (PR_GOOD_CC, get_time_stamp(), txt, PR_NC))
    sys.stdout.flush()

def get_single_item(d):
    """Get an item from a dict which contains just one item."""
    assert len(d) == 1, 'Single-item dict must have just one item, not %d.' % len(d)
    return next(six.iteritems(d))

def is_int(value):
    """Return `True` if ``value`` is an integer."""
    if isinstance(value, bool):
        return False
    try:
        int(value)
        return True
    except (ValueError, TypeError):
        return False

def dictmerge(x, y):
    """
    merge two dictionaries
    """
    z = x.copy()
    z.update(y)
    return z

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

def update(self, other_dict):
        """update() extends rather than replaces existing key lists."""
        for key, value in iter_multi_items(other_dict):
            MultiDict.add(self, key, value)

def swap(self):
        """Return the box (for horizontal graphs)"""
        self.xmin, self.ymin = self.ymin, self.xmin
        self.xmax, self.ymax = self.ymax, self.xmax

def itervalues(d, **kw):
    """Return an iterator over the values of a dictionary."""
    if not PY2:
        return iter(d.values(**kw))
    return d.itervalues(**kw)

def tob(data, enc='utf8'):
    """ Convert anything to bytes """
    return data.encode(enc) if isinstance(data, six.text_type) else bytes(data)

def purge_dict(idict):
    """Remove null items from a dictionary """
    odict = {}
    for key, val in idict.items():
        if is_null(val):
            continue
        odict[key] = val
    return odict

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

def set_history_file(self, path):
        """Set path to history file. "" produces no file."""
        if path:
            self.history = prompt_toolkit.history.FileHistory(fixpath(path))
        else:
            self.history = prompt_toolkit.history.InMemoryHistory()

def get_python_dict(scala_map):
    """Return a dict from entries in a scala.collection.immutable.Map"""
    python_dict = {}
    keys = get_python_list(scala_map.keys().toList())
    for key in keys:
        python_dict[key] = scala_map.apply(key)
    return python_dict

def fast_distinct(self):
        """
        Because standard distinct used on the all fields are very slow and works only with PostgreSQL database
        this method provides alternative to the standard distinct method.
        :return: qs with unique objects
        """
        return self.model.objects.filter(pk__in=self.values_list('pk', flat=True))

def is_cached(file_name):
	"""
	Check if a given file is available in the cache or not
	"""

	gml_file_path = join(join(expanduser('~'), OCTOGRID_DIRECTORY), file_name)

	return isfile(gml_file_path)

def type_converter(text):
    """ I convert strings into integers, floats, and strings! """
    if text.isdigit():
        return int(text), int

    try:
        return float(text), float
    except ValueError:
        return text, STRING_TYPE

def array_bytes(array):
    """ Estimates the memory of the supplied array in bytes """
    return np.product(array.shape)*np.dtype(array.dtype).itemsize

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

def Distance(lat1, lon1, lat2, lon2):
    """Get distance between pairs of lat-lon points"""

    az12, az21, dist = wgs84_geod.inv(lon1, lat1, lon2, lat2)
    return az21, dist

def set_value(self, value):
        """Set value of the checkbox.

        Parameters
        ----------
        value : bool
            value for the checkbox

        """
        if value:
            self.setCheckState(Qt.Checked)
        else:
            self.setCheckState(Qt.Unchecked)

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

def set_locale(request):
    """Return locale from GET lang param or automatically."""
    return request.query.get('lang', app.ps.babel.select_locale_by_request(request))

def elapsed_time_from(start_time):
    """calculate time delta from latched time and current time"""
    time_then = make_time(start_time)
    time_now = datetime.utcnow().replace(microsecond=0)
    if time_then is None:
        return
    delta_t = time_now - time_then
    return delta_t

def getMaxAffiliationInstanceID():
    """

    Returns
    -------
    maximum value of the Primary key from the table "django-tethne_affiliation_instance"
    This is used to calculate the next id for primary key.

    if the table is empty, 0 is returned

    """
    dbconnectionhanlder = DBConnection()
    dbconnectionhanlder.cursor.execute("SELECT max(id) from `django-tethne_affiliation_instance`")
    rows = dbconnectionhanlder.cursor.fetchall()
    dbconnectionhanlder.conn.close()
    if rows[0][0] is None:
        return 0
    else:
        return rows[0][0]

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

def handle_m2m_user(self, sender, instance, **kwargs):
    """ Handle many to many relationships for user field """
    self.handle_save(instance.user.__class__, instance.user)

def center_eigenvalue_diff(mat):
    """Compute the eigvals of mat and then find the center eigval difference."""
    N = len(mat)
    evals = np.sort(la.eigvals(mat))
    diff = np.abs(evals[N/2] - evals[N/2-1])
    return diff

def maxId(self):
        """int: current max id of objects"""
        if len(self.model.db) == 0:
            return 0

        return max(map(lambda obj: obj["id"], self.model.db))

def lsem (inlist):
    """
Returns the estimated standard error of the mean (sx-bar) of the
values in the passed list.  sem = stdev / sqrt(n)

Usage:   lsem(inlist)
"""
    sd = stdev(inlist)
    n = len(inlist)
    return sd/math.sqrt(n)

def json_response(data, status=200):
    """Return a JsonResponse. Make sure you have django installed first."""
    from django.http import JsonResponse
    return JsonResponse(data=data, status=status, safe=isinstance(data, dict))

def import_js(path, lib_name, globals):
    """Imports from javascript source file.
      globals is your globals()"""
    with codecs.open(path_as_local(path), "r", "utf-8") as f:
        js = f.read()
    e = EvalJs()
    e.execute(js)
    var = e.context['var']
    globals[lib_name] = var.to_python()

def json_response(data, status=200):
    """Return a JsonResponse. Make sure you have django installed first."""
    from django.http import JsonResponse
    return JsonResponse(data=data, status=status, safe=isinstance(data, dict))

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

def make_aware(dt):
    """Appends tzinfo and assumes UTC, if datetime object has no tzinfo already."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

def __call__(self, args):
        """Execute the user function."""
        window, ij = args
        return self.user_func(srcs, window, ij, global_args), window

def release(self):
        """
        Releases this resource back to the pool it came from.
        """
        if self.errored:
            self.pool.delete_resource(self)
        else:
            self.pool.release(self)

def _rm_name_match(s1, s2):
  """
  determine whether two sequence names from a repeatmasker alignment match.

  :return: True if they are the same string, or if one forms a substring of the
           other, else False
  """
  m_len = min(len(s1), len(s2))
  return s1[:m_len] == s2[:m_len]

def pull_stream(image):
    """
    Return generator of pull status objects
    """
    return (json.loads(s) for s in _get_docker().pull(image, stream=True))

def _removeTags(tags, objects):
    """ Removes tags from objects """
    for t in tags:
        for o in objects:
            o.tags.remove(t)

    return True

def _composed_doc(fs):
    """
    Generate a docstring for the composition of fs.
    """
    if not fs:
        # Argument name for the docstring.
        return 'n'

    return '{f}({g})'.format(f=fs[0].__name__, g=_composed_doc(fs[1:]))

def date_to_datetime(x):
    """Convert a date into a datetime"""
    if not isinstance(x, datetime) and isinstance(x, date):
        return datetime.combine(x, time())
    return x

def Slice(a, begin, size):
    """
    Slicing op.
    """
    return np.copy(a)[[slice(*tpl) for tpl in zip(begin, begin+size)]],

def unpickle_file(picklefile, **kwargs):
    """Helper function to unpickle data from `picklefile`."""
    with open(picklefile, 'rb') as f:
        return pickle.load(f, **kwargs)

def dot(self, w):
        """Return the dotproduct between self and another vector."""

        return sum([x * y for x, y in zip(self, w)])

def unpatch(obj, name):
    """
    Undo the effects of patch(func, obj, name)
    """
    setattr(obj, name, getattr(obj, name).original)

def draw(graph, fname):
    """Draw a graph and save it into a file"""
    ag = networkx.nx_agraph.to_agraph(graph)
    ag.draw(fname, prog='dot')

def items(iterable):
    """
    Iterates over the items of a sequence. If the sequence supports the
      dictionary protocol (iteritems/items) then we use that. Otherwise
      we use the enumerate built-in function.
    """
    if hasattr(iterable, 'iteritems'):
        return (p for p in iterable.iteritems())
    elif hasattr(iterable, 'items'):
        return (p for p in iterable.items())
    else:
        return (p for p in enumerate(iterable))

def onchange(self, value):
        """Called when a new DropDownItem gets selected.
        """
        log.debug('combo box. selected %s' % value)
        self.select_by_value(value)
        return (value, )

def to_identifier(s):
  """
  Convert snake_case to camel_case.
  """
  if s.startswith('GPS'):
      s = 'Gps' + s[3:]
  return ''.join([i.capitalize() for i in s.split('_')]) if '_' in s else s

def to_json(obj):
    """Return a json string representing the python object obj."""
    i = StringIO.StringIO()
    w = Writer(i, encoding='UTF-8')
    w.write_value(obj)
    return i.getvalue()

def decamelise(text):
    """Convert CamelCase to lower_and_underscore."""
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()

def pretty_dict_str(d, indent=2):
    """shows JSON indented representation of d"""
    b = StringIO()
    write_pretty_dict_str(b, d, indent=indent)
    return b.getvalue()

def camel_case_from_underscores(string):
    """generate a CamelCase string from an underscore_string."""
    components = string.split('_')
    string = ''
    for component in components:
        string += component[0].upper() + component[1:]
    return string

def _from_dict(cls, _dict):
        """Initialize a KeyValuePair object from a json dictionary."""
        args = {}
        if 'key' in _dict:
            args['key'] = Key._from_dict(_dict.get('key'))
        if 'value' in _dict:
            args['value'] = Value._from_dict(_dict.get('value'))
        return cls(**args)

def _cdf(self, xloc, dist, base, cache):
        """Cumulative distribution function."""
        return evaluation.evaluate_forward(dist, base**xloc, cache=cache)

def parallel(processes, threads):
    """
    execute jobs in processes using N threads
    """
    pool = multithread(threads)
    pool.map(run_process, processes)
    pool.close()
    pool.join()

def start_of_month(val):
    """
    Return a new datetime.datetime object with values that represent
    a start of a month.
    :param val: Date to ...
    :type val: datetime.datetime | datetime.date
    :rtype: datetime.datetime
    """
    if type(val) == date:
        val = datetime.fromordinal(val.toordinal())
    return start_of_day(val).replace(day=1)

def _kw(keywords):
    """Turn list of keywords into dictionary."""
    r = {}
    for k, v in keywords:
        r[k] = v
    return r

def to_snake_case(text):
    """Convert to snake case.

    :param str text:
    :rtype: str
    :return:
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

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

def bitsToString(arr):
  """Returns a string representing a numpy array of 0's and 1's"""
  s = array('c','.'*len(arr))
  for i in xrange(len(arr)):
    if arr[i] == 1:
      s[i]='*'
  return s

def add_index_alias(es, index_name, alias_name):
    """Add index alias to index_name"""

    es.indices.put_alias(index=index_name, name=terms_alias)

def _check_and_convert_bools(self):
        """Replace boolean variables by the characters 'F'/'T'
        """
        replacements = {
            True: 'T',
            False: 'F',
        }

        for key in self.bools:
            if isinstance(self[key], bool):
                self[key] = replacements[self[key]]

def index(obj, index=INDEX_NAME, doc_type=DOC_TYPE):
    """
    Index the given document.

    https://elasticsearch-py.readthedocs.io/en/master/api.html#elasticsearch.Elasticsearch.index
    https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-index_.html
    """
    doc = to_dict(obj)

    if doc is None:
        return

    id = doc.pop('id')

    return es_conn.index(index, doc_type, doc, id=id)

def as_tuple(self, value):
        """Utility function which converts lists to tuples."""
        if isinstance(value, list):
            value = tuple(value)
        return value

def format_header_cell(val):
    """
    Formats given header column. This involves changing '_Px_' to '(', '_xP_' to ')' and
    all other '_' to spaces.
    """
    return re.sub('_', ' ', re.sub(r'(_Px_)', '(', re.sub(r'(_xP_)', ')', str(val) )))

def QA_util_datetime_to_strdate(dt):
    """
    :param dt:  pythone datetime.datetime
    :return:  1999-02-01 string type
    """
    strdate = "%04d-%02d-%02d" % (dt.year, dt.month, dt.day)
    return strdate

def iterparse(source, tag, clear=False, events=None):
    """
    iterparse variant that supports 'tag' parameter (like lxml),
    yields elements and clears nodes after parsing.
    """
    for event, elem in ElementTree.iterparse(source, events=events):
        if elem.tag == tag:
            yield elem
        if clear:
            elem.clear()

def command_py2to3(args):
    """
    Apply '2to3' tool (Python2 to Python3 conversion tool) to Python sources.
    """
    from lib2to3.main import main
    sys.exit(main("lib2to3.fixes", args=args.sources))

def is_rfc2822(instance: str):
    """Validates RFC2822 format"""
    if not isinstance(instance, str):
        return True
    return email.utils.parsedate(instance) is not None

def b(s):
	""" Encodes Unicode strings to byte strings, if necessary. """

	return s if isinstance(s, bytes) else s.encode(locale.getpreferredencoding())

def _check_color_dim(val):
    """Ensure val is Nx(n_col), usually Nx3"""
    val = np.atleast_2d(val)
    if val.shape[1] not in (3, 4):
        raise RuntimeError('Value must have second dimension of size 3 or 4')
    return val, val.shape[1]

def aug_sysargv(cmdstr):
    """ DEBUG FUNC modify argv to look like you ran a command """
    import shlex
    argv = shlex.split(cmdstr)
    sys.argv.extend(argv)

def safe_unicode(string):
    """If Python 2, replace non-ascii characters and return encoded string."""
    if not PY3:
        uni = string.replace(u'\u2019', "'")
        return uni.encode('utf-8')
        
    return string

def flatten(l, types=(list, float)):
    """
    Flat nested list of lists into a single list.
    """
    l = [item if isinstance(item, types) else [item] for item in l]
    return [item for sublist in l for item in sublist]

def items(self):
    """Return a list of the (name, value) pairs of the enum.

    These are returned in the order they were defined in the .proto file.
    """
    return [(value_descriptor.name, value_descriptor.number)
            for value_descriptor in self._enum_type.values]

def getpass(self, prompt, default=None):
        """Provide a password prompt."""
        return click.prompt(prompt, hide_input=True, default=default)

def unpack_out(self, name):
        return self.parse("""
            $enum = $enum_class($value.value)
            """, enum_class=self._import_type(), value=name)["enum"]

def _set_widget_background_color(widget, color):
        """
        Changes the base color of a widget (background).
        :param widget: widget to modify
        :param color: the color to apply
        """
        pal = widget.palette()
        pal.setColor(pal.Base, color)
        widget.setPalette(pal)

def get_mi_vec(slab):
    """
    Convenience function which returns the unit vector aligned
    with the miller index.
    """
    mvec = np.cross(slab.lattice.matrix[0], slab.lattice.matrix[1])
    return mvec / np.linalg.norm(mvec)

def c2f(r, i, ctype_name):
    """
    Convert strings to complex number instance with specified numpy type.
    """

    ftype = c2f_dict[ctype_name]
    return np.typeDict[ctype_name](ftype(r) + 1j * ftype(i))

def erase_lines(n=1):
    """ Erases n lines from the screen and moves the cursor up to follow
    """
    for _ in range(n):
        print(codes.cursor["up"], end="")
        print(codes.cursor["eol"], end="")

def contained_in(filename, directory):
    """Test if a file is located within the given directory."""
    filename = os.path.normcase(os.path.abspath(filename))
    directory = os.path.normcase(os.path.abspath(directory))
    return os.path.commonprefix([filename, directory]) == directory

def exec_function(ast, globals_map):
    """Execute a python code object in the given environment.

    Args:
      globals_map: Dictionary to use as the globals context.
    Returns:
      locals_map: Dictionary of locals from the environment after execution.
    """
    locals_map = globals_map
    exec ast in globals_map, locals_map
    return locals_map

def is_collection(obj):
    """Tests if an object is a collection."""

    col = getattr(obj, '__getitem__', False)
    val = False if (not col) else True

    if isinstance(obj, basestring):
        val = False

    return val

def log_exception(exc_info=None, stream=None):
    """Log the 'exc_info' tuple in the server log."""
    exc_info = exc_info or sys.exc_info()
    stream = stream or sys.stderr
    try:
        from traceback import print_exception
        print_exception(exc_info[0], exc_info[1], exc_info[2], None, stream)
        stream.flush()
    finally:
        exc_info = None

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

def __init__(self, filename, formatting_info=False, handle_ambiguous_date=None):
    """Initialize the ExcelWorkbook instance."""
    super().__init__(filename)
    self.workbook = xlrd.open_workbook(self.filename, formatting_info=formatting_info)
    self.handle_ambiguous_date = handle_ambiguous_date

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

def handle_exception(error):
        """Simple method for handling exceptions raised by `PyBankID`.

        :param flask_pybankid.FlaskPyBankIDError error: The exception to handle.
        :return: The exception represented as a dictionary.
        :rtype: dict

        """
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

def contained_in(filename, directory):
    """Test if a file is located within the given directory."""
    filename = os.path.normcase(os.path.abspath(filename))
    directory = os.path.normcase(os.path.abspath(directory))
    return os.path.commonprefix([filename, directory]) == directory

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

def has_multiline_items(maybe_list: Optional[Sequence[str]]):
    """Check whether one of the items in the list has multiple lines."""
    return maybe_list and any(is_multiline(item) for item in maybe_list)

def exec_function(ast, globals_map):
    """Execute a python code object in the given environment.

    Args:
      globals_map: Dictionary to use as the globals context.
    Returns:
      locals_map: Dictionary of locals from the environment after execution.
    """
    locals_map = globals_map
    exec ast in globals_map, locals_map
    return locals_map

def is_connected(self):
        """
        Return true if the socket managed by this connection is connected

        :rtype: bool
        """
        try:
            return self.socket is not None and self.socket.getsockname()[1] != 0 and BaseTransport.is_connected(self)
        except socket.error:
            return False

def call_and_exit(self, cmd, shell=True):
        """Run the *cmd* and exit with the proper exit code."""
        sys.exit(subprocess.call(cmd, shell=shell))

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

def get_tablenames(cur):
    """ Conveinience: """
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tablename_list_ = cur.fetchall()
    tablename_list = [str(tablename[0]) for tablename in tablename_list_ ]
    return tablename_list

def is_list_of_list(item):
    """
    check whether the item is list (tuple)
    and consist of list (tuple) elements
    """
    if (
        type(item) in (list, tuple)
        and len(item)
        and isinstance(item[0], (list, tuple))
    ):
        return True
    return False

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

def is_nullable_list(val, vtype):
    """Return True if list contains either values of type `vtype` or None."""
    return (isinstance(val, list) and
            any(isinstance(v, vtype) for v in val) and
            all((isinstance(v, vtype) or v is None) for v in val))

def _expand(self, str, local_vars={}):
        """Expand $vars in a string."""
        return ninja_syntax.expand(str, self.vars, local_vars)

def task_property_present_predicate(service, task, prop):
    """ True if the json_element passed is present for the task specified.
    """
    try:
        response = get_service_task(service, task)
    except Exception as e:
        pass

    return (response is not None) and (prop in response)

def _merge_args_with_kwargs(args_dict, kwargs_dict):
    """Merge args with kwargs."""
    ret = args_dict.copy()
    ret.update(kwargs_dict)
    return ret

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

def unique_everseen(seq):
    """Solution found here : http://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-python-whilst-preserving-order"""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def is_json_file(filename, show_warnings = False):
    """Check configuration file type is JSON
    Return a boolean indicating wheather the file is JSON format or not
    """
    try:
        config_dict = load_config(filename, file_type = "json")
        is_json = True
    except:
        is_json = False
    return(is_json)

def return_letters_from_string(text):
    """Get letters from string only."""
    out = ""
    for letter in text:
        if letter.isalpha():
            out += letter
    return out

def _writable_dir(path):
    """Whether `path` is a directory, to which the user has write access."""
    return os.path.isdir(path) and os.access(path, os.W_OK)

def get_url_args(url):
    """ Returns a dictionary from a URL params """
    url_data = urllib.parse.urlparse(url)
    arg_dict = urllib.parse.parse_qs(url_data.query)
    return arg_dict

def test_for_image(self, cat, img):
        """Tests if image img in category cat exists"""
        return self.test_for_category(cat) and img in self.items[cat]

def extract_all(zipfile, dest_folder):
    """
    reads the zip file, determines compression
    and unzips recursively until source files 
    are extracted 
    """
    z = ZipFile(zipfile)
    print(z)
    z.extract(dest_folder)

def is_string(obj):
    """Is this a string.

    :param object obj:
    :rtype: bool
    """
    if PYTHON3:
        str_type = (bytes, str)
    else:
        str_type = (bytes, str, unicode)
    return isinstance(obj, str_type)

def parse_float_literal(ast, _variables=None):
    """Parse a float value node in the AST."""
    if isinstance(ast, (FloatValueNode, IntValueNode)):
        return float(ast.value)
    return INVALID

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

def valid_uuid(value):
    """ Check if value is a valid UUID. """

    try:
        uuid.UUID(value, version=4)
        return True
    except (TypeError, ValueError, AttributeError):
        return False

def stft_magnitude(signal, fft_length,
                   hop_length=None,
                   window_length=None):
  """Calculate the short-time Fourier transform magnitude.

  Args:
    signal: 1D np.array of the input time-domain signal.
    fft_length: Size of the FFT to apply.
    hop_length: Advance (in samples) between each frame passed to FFT.
    window_length: Length of each block of samples to pass to FFT.

  Returns:
    2D np.array where each row contains the magnitudes of the fft_length/2+1
    unique values of the FFT for the corresponding frame of input samples.
  """
  frames = frame(signal, window_length, hop_length)
  # Apply frame window to each frame. We use a periodic Hann (cosine of period
  # window_length) instead of the symmetric Hann of np.hanning (period
  # window_length-1).
  window = periodic_hann(window_length)
  windowed_frames = frames * window
  return np.abs(np.fft.rfft(windowed_frames, int(fft_length)))

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

def get_time(filename):
	"""
	Get the modified time for a file as a datetime instance
	"""
	ts = os.stat(filename).st_mtime
	return datetime.datetime.utcfromtimestamp(ts)

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

def _rectangular(n):
    """Checks to see if a 2D list is a valid 2D matrix"""
    for i in n:
        if len(i) != len(n[0]):
            return False
    return True

def rewindbody(self):
        """Rewind the file to the start of the body (if seekable)."""
        if not self.seekable:
            raise IOError, "unseekable file"
        self.fp.seek(self.startofbody)

def _check_2d_shape(X):
    """Check shape of array or sparse matrix.

    Assure that X is always 2D: Unlike numpy we always deal with 2D arrays.
    """
    if X.dtype.names is None and len(X.shape) != 2:
        raise ValueError('X needs to be 2-dimensional, not '
                         '{}-dimensional.'.format(len(X.shape)))

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

def _pip_exists(self):
        """Returns True if pip exists inside the virtual environment. Can be
        used as a naive way to verify that the environment is installed."""
        return os.path.isfile(os.path.join(self.path, 'bin', 'pip'))

def read(filename):
    """Read and return `filename` in root dir of project and return string"""
    return codecs.open(os.path.join(__DIR__, filename), 'r').read()

def instance_contains(container, item):
    """Search into instance attributes, properties and return values of no-args methods."""
    return item in (member for _, member in inspect.getmembers(container))

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

def skewness(data):
    """
    Returns the skewness of ``data``.
    """

    if len(data) == 0:
        return None

    num = moment(data, 3)
    denom = moment(data, 2) ** 1.5

    return num / denom if denom != 0 else 0.

def inpaint(self):
        """ Replace masked-out elements in an array using an iterative image inpainting algorithm. """

        import inpaint
        filled = inpaint.replace_nans(np.ma.filled(self.raster_data, np.NAN).astype(np.float32), 3, 0.01, 2)
        self.raster_data = np.ma.masked_invalid(filled)

def _check_color_dim(val):
    """Ensure val is Nx(n_col), usually Nx3"""
    val = np.atleast_2d(val)
    if val.shape[1] not in (3, 4):
        raise RuntimeError('Value must have second dimension of size 3 or 4')
    return val, val.shape[1]

def filter_none(list_of_points):
    """
    
    :param list_of_points: 
    :return: list_of_points with None's removed
    """
    remove_elementnone = filter(lambda p: p is not None, list_of_points)
    remove_sublistnone = filter(lambda p: not contains_none(p), remove_elementnone)
    return list(remove_sublistnone)

def available_gpus():
  """List of GPU device names detected by TensorFlow."""
  local_device_protos = device_lib.list_local_devices()
  return [x.name for x in local_device_protos if x.device_type == 'GPU']

def cleanLines(source, lineSep=os.linesep):
    """
    :param source: some iterable source (list, file, etc)
    :param lineSep: string of separators (chars) that must be removed
    :return: list of non empty lines with removed separators
    """
    stripped = (line.strip(lineSep) for line in source)
    return (line for line in stripped if len(line) != 0)

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

def camel_to_snake_case(name):
    """Takes a camelCased string and converts to snake_case."""
    pattern = r'[A-Z][a-z]+|[A-Z]+(?![a-z])'
    return '_'.join(map(str.lower, re.findall(pattern, name)))

def invalidate_cache(cpu, address, size):
        """ remove decoded instruction from instruction cache """
        cache = cpu.instruction_cache
        for offset in range(size):
            if address + offset in cache:
                del cache[address + offset]

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

def clear():
    """Clears the console."""
    if sys.platform.startswith("win"):
        call("cls", shell=True)
    else:
        call("clear", shell=True)

def find_lt(a, x):
    """Find rightmost value less than x."""
    i = bs.bisect_left(a, x)
    if i: return i - 1
    raise ValueError

def logout(cache):
    """
    Logs out the current session by removing it from the cache. This is
    expected to only occur when a session has
    """
    cache.set(flask.session['auth0_key'], None)
    flask.session.clear()
    return True

def retrieve_asset(filename):
    """ Retrieves a non-image asset associated with an entry """

    record = model.Image.get(asset_name=filename)
    if not record:
        raise http_error.NotFound("File not found")
    if not record.is_asset:
        raise http_error.Forbidden()

    return flask.send_file(record.file_path, conditional=True)

def Flush(self):
    """Flush all items from cache."""
    while self._age:
      node = self._age.PopLeft()
      self.KillObject(node.data)

    self._hash = dict()

def json_template(data, template_name, template_context):
    """Old style, use JSONTemplateResponse instead of this.
    """
    html = render_to_string(template_name, template_context)
    data = data or {}
    data['html'] = html
    return HttpResponse(json_encode(data), content_type='application/json')

def _closeResources(self):
        """ Closes the root Dataset.
        """
        logger.info("Closing: {}".format(self._fileName))
        self._h5Group.close()
        self._h5Group = None

def arg_bool(name, default=False):
    """ Fetch a query argument, as a boolean. """
    v = request.args.get(name, '')
    if not len(v):
        return default
    return v in BOOL_TRUISH

def closing_plugin(self, cancelable=False):
        """Perform actions before parent main window is closed"""
        self.dialog_manager.close_all()
        self.shell.exit_interpreter()
        return True

def view_500(request, url=None):
    """
    it returns a 500 http response
    """
    res = render_to_response("500.html", context_instance=RequestContext(request))
    res.status_code = 500
    return res

def stop(self, dummy_signum=None, dummy_frame=None):
        """ Shutdown process (this method is also a signal handler) """
        logging.info('Shutting down ...')
        self.socket.close()
        sys.exit(0)

def save_session_to_file(self, sessionfile):
        """Not meant to be used directly, use :meth:`Instaloader.save_session_to_file`."""
        pickle.dump(requests.utils.dict_from_cookiejar(self._session.cookies), sessionfile)

def logout(cache):
    """
    Logs out the current session by removing it from the cache. This is
    expected to only occur when a session has
    """
    cache.set(flask.session['auth0_key'], None)
    flask.session.clear()
    return True

def erase_lines(n=1):
    """ Erases n lines from the screen and moves the cursor up to follow
    """
    for _ in range(n):
        print(codes.cursor["up"], end="")
        print(codes.cursor["eol"], end="")

def staticdir():
    """Return the location of the static data directory."""
    root = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(root, "static")

def trim(self):
        """Clear not used counters"""
        for key, value in list(iteritems(self.counters)):
            if value.empty():
                del self.counters[key]

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

def flatten(nested):
    """ Return a flatten version of the nested argument """
    flat_return = list()

    def __inner_flat(nested,flat):
        for i in nested:
            __inner_flat(i, flat) if isinstance(i, list) else flat.append(i)
        return flat

    __inner_flat(nested,flat_return)

    return flat_return

def merge_pdfs(pdf_filepaths, out_filepath):
    """ Merge all the PDF files in `pdf_filepaths` in a new PDF file `out_filepath`.

    Parameters
    ----------
    pdf_filepaths: list of str
        Paths to PDF files.

    out_filepath: str
        Path to the result PDF file.

    Returns
    -------
    path: str
        The output file path.
    """
    merger = PdfFileMerger()
    for pdf in pdf_filepaths:
        merger.append(PdfFileReader(open(pdf, 'rb')))

    merger.write(out_filepath)

    return out_filepath

def flatten(l, types=(list, float)):
    """
    Flat nested list of lists into a single list.
    """
    l = [item if isinstance(item, types) else [item] for item in l]
    return [item for sublist in l for item in sublist]

def merge_dict(data, *args):
    """Merge any number of dictionaries
    """
    results = {}
    for current in (data,) + args:
        results.update(current)
    return results

def flatten(nested, containers=(list, tuple)):
    """ Flatten a nested list by yielding its scalar items.
    """
    for item in nested:
        if hasattr(item, "next") or isinstance(item, containers):
            for subitem in flatten(item):
                yield subitem
        else:
            yield item

def pprint(self, seconds):
        """
        Pretty Prints seconds as Hours:Minutes:Seconds.MilliSeconds

        :param seconds:  The time in seconds.
        """
        return ("%d:%02d:%02d.%03d", reduce(lambda ll, b: divmod(ll[0], b) + ll[1:], [(seconds * 1000,), 1000, 60, 60]))

def flatten_array(grid):
    """
    Takes a multi-dimensional array and returns a 1 dimensional array with the
    same contents.
    """
    grid = [grid[i][j] for i in range(len(grid)) for j in range(len(grid[i]))]
    while type(grid[0]) is list:
        grid = flatten_array(grid)
    return grid

def version_jar(self):
		"""
		Special case of version() when the executable is a JAR file.
		"""
		cmd = config.get_command('java')
		cmd.append('-jar')
		cmd += self.cmd
		self.version(cmd=cmd, path=self.cmd[0])

def flatten(lis):
    """Given a list, possibly nested to any level, return it flattened."""
    new_lis = []
    for item in lis:
        if isinstance(item, collections.Sequence) and not isinstance(item, basestring):
            new_lis.extend(flatten(item))
        else:
            new_lis.append(item)
    return new_lis

def clear():
    """Clears the console."""
    if sys.platform.startswith("win"):
        call("cls", shell=True)
    else:
        call("clear", shell=True)

def flatten_list(l):
    """ Nested lists to single-level list, does not split strings"""
    return list(chain.from_iterable(repeat(x,1) if isinstance(x,str) else x for x in l))

def parse_comments_for_file(filename):
    """
    Return a list of all parsed comments in a file.  Mostly for testing &
    interactive use.
    """
    return [parse_comment(strip_stars(comment), next_line)
            for comment, next_line in get_doc_comments(read_file(filename))]

def intround(value):
    """Given a float returns a rounded int. Should give the same result on
    both Py2/3
    """

    return int(decimal.Decimal.from_float(
        value).to_integral_value(decimal.ROUND_HALF_EVEN))

def count_string_diff(a,b):
    """Return the number of characters in two strings that don't exactly match"""
    shortest = min(len(a), len(b))
    return sum(a[i] != b[i] for i in range(shortest))

def flush():
    """Try to flush all stdio buffers, both from python and from C."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except (AttributeError, ValueError, IOError):
        pass  # unsupported
    try:
        libc.fflush(None)
    except (AttributeError, ValueError, IOError):
        pass

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

def safe_int_conv(number):
    """Safely convert a single number to integer."""
    try:
        return int(np.array(number).astype(int, casting='safe'))
    except TypeError:
        raise ValueError('cannot safely convert {} to integer'.format(number))

def dedupe(items):
    """Remove duplicates from a sequence (of hashable items) while maintaining
    order. NOTE: This only works if items in the list are hashable types.

    Taken from the Python Cookbook, 3rd ed. Such a great book!

    """
    seen = set()
    for item in items:
        if item not in seen:
            yield item
            seen.add(item)

def go_to_background():
    """ Daemonize the running process. """
    try:
        if os.fork():
            sys.exit()
    except OSError as errmsg:
        LOGGER.error('Fork failed: {0}'.format(errmsg))
        sys.exit('Fork failed')

def generate(env):
    """Add Builders and construction variables for SGI MIPS C++ to an Environment."""

    cplusplus.generate(env)

    env['CXX']         = 'CC'
    env['CXXFLAGS']    = SCons.Util.CLVar('-LANG:std')
    env['SHCXX']       = '$CXX'
    env['SHOBJSUFFIX'] = '.o'
    env['STATIC_AND_SHARED_OBJECTS_ARE_THE_SAME'] = 1

def multis_2_mono(table):
    """
    Converts each multiline string in a table to single line.

    Parameters
    ----------
    table : list of list of str
        A list of rows containing strings

    Returns
    -------
    table : list of lists of str
    """
    for row in range(len(table)):
        for column in range(len(table[row])):
            table[row][column] = table[row][column].replace('\n', ' ')

    return table

def _manhattan_distance(vec_a, vec_b):
    """Return manhattan distance between two lists of numbers."""
    if len(vec_a) != len(vec_b):
        raise ValueError('len(vec_a) must equal len(vec_b)')
    return sum(map(lambda a, b: abs(a - b), vec_a, vec_b))

def safe_format(s, **kwargs):
  """
  :type s str
  """
  return string.Formatter().vformat(s, (), defaultdict(str, **kwargs))

def num_leaves(tree):
    """Determine the number of leaves in a tree"""
    if tree.is_leaf:
        return 1
    else:
        return num_leaves(tree.left_child) + num_leaves(tree.right_child)

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

def _tuple_repr(data):
    """Return a repr() for a list/tuple"""
    if len(data) == 1:
        return "(%s,)" % rpr(data[0])
    else:
        return "(%s)" % ", ".join([rpr(x) for x in data])

def timestamp_to_datetime(timestamp):
    """Convert an ARF timestamp to a datetime.datetime object (naive local time)"""
    from datetime import datetime, timedelta
    obj = datetime.fromtimestamp(timestamp[0])
    return obj + timedelta(microseconds=int(timestamp[1]))

def fromtimestamp(cls, timestamp):
    """Returns a datetime object of a given timestamp (in local tz)."""
    d = cls.utcfromtimestamp(timestamp)
    return d.astimezone(localtz())

def str2int(num, radix=10, alphabet=BASE85):
    """helper function for quick base conversions from strings to integers"""
    return NumConv(radix, alphabet).str2int(num)

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

def _concatenate_virtual_arrays(arrs, cols=None, scaling=None):
    """Return a virtual concatenate of several NumPy arrays."""
    return None if not len(arrs) else ConcatenatedArrays(arrs, cols,
                                                         scaling=scaling)

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

def _concatenate_virtual_arrays(arrs, cols=None, scaling=None):
    """Return a virtual concatenate of several NumPy arrays."""
    return None if not len(arrs) else ConcatenatedArrays(arrs, cols,
                                                         scaling=scaling)

def get_previous(self):
        """Get the billing cycle prior to this one. May return None"""
        return BillingCycle.objects.filter(date_range__lt=self.date_range).order_by('date_range').last()

def _concatenate_virtual_arrays(arrs, cols=None, scaling=None):
    """Return a virtual concatenate of several NumPy arrays."""
    return None if not len(arrs) else ConcatenatedArrays(arrs, cols,
                                                         scaling=scaling)

def set_global(node: Node, key: str, value: Any):
    """Adds passed value to node's globals"""
    node.node_globals[key] = value

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

def unpunctuate(s, *, char_blacklist=string.punctuation):
    """ Remove punctuation from string s. """
    # remove punctuation
    s = "".join(c for c in s if c not in char_blacklist)
    # remove consecutive spaces
    return " ".join(filter(None, s.split(" ")))

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

def get_decimal_quantum(precision):
    """Return minimal quantum of a number, as defined by precision."""
    assert isinstance(precision, (int, decimal.Decimal))
    return decimal.Decimal(10) ** (-precision)

def get_url(self, cmd, **args):
        """Expand the request URL for a request."""
        return self.http.base_url + self._mkurl(cmd, *args)

def euclidean(c1, c2):
    """Square of the euclidean distance"""
    diffs = ((i - j) for i, j in zip(c1, c2))
    return sum(x * x for x in diffs)

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

def list2dict(lst):
    """Takes a list of (key,value) pairs and turns it into a dict."""

    dic = {}
    for k,v in lst: dic[k] = v
    return dic

def double_sha256(data):
    """A standard compound hash."""
    return bytes_as_revhex(hashlib.sha256(hashlib.sha256(data).digest()).digest())

def command_py2to3(args):
    """
    Apply '2to3' tool (Python2 to Python3 conversion tool) to Python sources.
    """
    from lib2to3.main import main
    sys.exit(main("lib2to3.fixes", args=args.sources))

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

def render(template=None, ostr=None, **kwargs):
    """Generate report from a campaign

    :param template: Jinja template to use, ``DEFAULT_TEMPLATE`` is used
    if not specified
    :param ostr: output file or filename. Default is standard output
    """
    jinja_environment.filters['texscape'] = tex_escape
    template = template or DEFAULT_TEMPLATE
    ostr = ostr or sys.stdout
    jinja_template = jinja_environment.get_template(template)
    jinja_template.stream(**kwargs).dump(ostr)

def abs_img(img):
    """ Return an image with the binarised version of the data of `img`."""
    bool_img = np.abs(read_img(img).get_data())
    return bool_img.astype(int)

def _read_date_from_string(str1):
    """
    Reads the date from a string in the format YYYY/MM/DD and returns
    :class: datetime.date
    """
    full_date = [int(x) for x in str1.split('/')]
    return datetime.date(full_date[0], full_date[1], full_date[2])

def convolve_gaussian_2d(image, gaussian_kernel_1d):
    """Convolve 2d gaussian."""
    result = scipy.ndimage.filters.correlate1d(
        image, gaussian_kernel_1d, axis=0)
    result = scipy.ndimage.filters.correlate1d(
        result, gaussian_kernel_1d, axis=1)
    return result

def get_remote_content(filepath):
        """ A handy wrapper to get a remote file content """
        with hide('running'):
            temp = BytesIO()
            get(filepath, temp)
            content = temp.getvalue().decode('utf-8')
        return content.strip()

def Slice(a, begin, size):
    """
    Slicing op.
    """
    return np.copy(a)[[slice(*tpl) for tpl in zip(begin, begin+size)]],

def get_by(self, name):
    """get element by name"""
    return next((item for item in self if item.name == name), None)

def angle(x, y):
    """Return the angle between vectors a and b in degrees."""
    return arccos(dot(x, y)/(norm(x)*norm(y)))*180./pi

def object_as_dict(obj):
    """Turn an SQLAlchemy model into a dict of field names and values.

    Based on https://stackoverflow.com/a/37350445/1579058
    """
    return {c.key: getattr(obj, c.key)
            for c in inspect(obj).mapper.column_attrs}

def similarity(self, other):
        """Calculates the cosine similarity between this vector and another
        vector."""
        if self.magnitude == 0 or other.magnitude == 0:
            return 0

        return self.dot(other) / self.magnitude

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

def size(self):
        """
        Recursively find size of a tree. Slow.
        """

        if self is NULL:
            return 0
        return 1 + self.left.size() + self.right.size()

def uniq(seq):
    """ Return a copy of seq without duplicates. """
    seen = set()
    return [x for x in seq if str(x) not in seen and not seen.add(str(x))]

def size(self):
        """
        Recursively find size of a tree. Slow.
        """

        if self is NULL:
            return 0
        return 1 + self.left.size() + self.right.size()

def data_directory():
    """Return the absolute path to the directory containing the package data."""
    package_directory = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(package_directory, "data")

def num_leaves(tree):
    """Determine the number of leaves in a tree"""
    if tree.is_leaf:
        return 1
    else:
        return num_leaves(tree.left_child) + num_leaves(tree.right_child)

def get_Callable_args_res(clb):
    """Python version independent function to obtain the parameters
    of a typing.Callable object. Returns as tuple: args, result.
    Tested with CPython 2.7, 3.5, 3.6 and Jython 2.7.1.
    """
    try:
        return clb.__args__, clb.__result__
    except AttributeError:
        # Python 3.6
        return clb.__args__[:-1], clb.__args__[-1]

def count_list(the_list):
    """
    Generates a count of the number of times each unique item appears in a list
    """
    count = the_list.count
    result = [(item, count(item)) for item in set(the_list)]
    result.sort()
    return result

def get_max(qs, field):
    """
    get max for queryset.

    qs: queryset
    field: The field name to max.
    """
    max_field = '%s__max' % field
    num = qs.aggregate(Max(field))[max_field]
    return num if num else 0

def count_rows_with_nans(X):
    """Count the number of rows in 2D arrays that contain any nan values."""
    if X.ndim == 2:
        return np.where(np.isnan(X).sum(axis=1) != 0, 1, 0).sum()

def similarity(self, other):
        """Calculates the cosine similarity between this vector and another
        vector."""
        if self.magnitude == 0 or other.magnitude == 0:
            return 0

        return self.dot(other) / self.magnitude

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

def get_language(self):
        """
        Get the language parameter from the current request.
        """
        return get_language_parameter(self.request, self.query_language_key, default=self.get_default_language(object=object))

def entropy(string):
    """Compute entropy on the string"""
    p, lns = Counter(string), float(len(string))
    return -sum(count/lns * math.log(count/lns, 2) for count in p.values())

def _get_str_columns(sf):
    """
    Returns a list of names of columns that are string type.
    """
    return [name for name in sf.column_names() if sf[name].dtype == str]

def _count_leading_whitespace(text):
  """Returns the number of characters at the beginning of text that are whitespace."""
  idx = 0
  for idx, char in enumerate(text):
    if not char.isspace():
      return idx
  return idx + 1

def str2int(string_with_int):
    """ Collect digits from a string """
    return int("".join([char for char in string_with_int if char in string.digits]) or 0)

def str2bytes(x):
  """Convert input argument to bytes"""
  if type(x) is bytes:
    return x
  elif type(x) is str:
    return bytes([ ord(i) for i in x ])
  else:
    return str2bytes(str(x))

def get_image_dimension(self, url):
        """
        Return a tuple that contains (width, height)
        Pass in a url to an image and find out its size without loading the whole file
        If the image wxh could not be found, the tuple will contain `None` values
        """
        w_h = (None, None)
        try:
            if url.startswith('//'):
                url = 'http:' + url
            data = requests.get(url).content
            im = Image.open(BytesIO(data))

            w_h = im.size
        except Exception:
            logger.warning("Error getting image size {}".format(url), exc_info=True)

        return w_h

def ident():
    """
    This routine returns the 3x3 identity matrix.

    http://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/ident_c.html

    :return: The 3x3 identity matrix.
    :rtype: 3x3-Element Array of floats
    """
    matrix = stypes.emptyDoubleMatrix()
    libspice.ident_c(matrix)
    return stypes.cMatrixToNumpy(matrix)

def dir_modtime(dpath):
    """
    Returns the latest modification time of all files/subdirectories in a
    directory
    """
    return max(os.path.getmtime(d) for d, _, _ in os.walk(dpath))

def head_and_tail_print(self, n=5):
        """Display the first and last n elements of a DataFrame."""
        from IPython import display
        display.display(display.HTML(self._head_and_tail_table(n)))

def entropy(string):
    """Compute entropy on the string"""
    p, lns = Counter(string), float(len(string))
    return -sum(count/lns * math.log(count/lns, 2) for count in p.values())

def get_parent_dir(name):
    """Get the parent directory of a filename."""
    parent_dir = os.path.dirname(os.path.dirname(name))
    if parent_dir:
        return parent_dir
    return os.path.abspath('.')

def merge(left, right, how='inner', key=None, left_key=None, right_key=None,
          left_as='left', right_as='right'):
    """ Performs a join using the union join function. """
    return join(left, right, how, key, left_key, right_key,
                join_fn=make_union_join(left_as, right_as))

def findfirst(f, coll):
    """Return first occurrence matching f, otherwise None"""
    result = list(dropwhile(f, coll))
    return result[0] if result else None

def polyline(*points):
    """Converts a list of points to a Path composed of lines connecting those 
    points (i.e. a linear spline or polyline).  See also `polygon()`."""
    return Path(*[Line(points[i], points[i+1])
                  for i in range(len(points) - 1)])

def split_every(n, iterable):
    """Returns a generator that spits an iteratable into n-sized chunks. The last chunk may have
    less than n elements.

    See http://stackoverflow.com/a/22919323/503377."""
    items = iter(iterable)
    return itertools.takewhile(bool, (list(itertools.islice(items, n)) for _ in itertools.count()))

def list2dict(list_of_options):
    """Transforms a list of 2 element tuples to a dictionary"""
    d = {}
    for key, value in list_of_options:
        d[key] = value
    return d

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

def concat(cls, iterables):
    """
    Similar to #itertools.chain.from_iterable().
    """

    def generator():
      for it in iterables:
        for element in it:
          yield element
    return cls(generator())

def get_func_name(func):
    """Return a name which includes the module name and function name."""
    func_name = getattr(func, '__name__', func.__class__.__name__)
    module_name = func.__module__

    if module_name is not None:
        module_name = func.__module__
        return '{}.{}'.format(module_name, func_name)

    return func_name

def symbol_pos_int(*args, **kwargs):
    """Create a sympy.Symbol with positive and integer assumptions."""
    kwargs.update({'positive': True,
                   'integer': True})
    return sympy.Symbol(*args, **kwargs)

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

def one_hot(x, size, dtype=np.float32):
  """Make a n+1 dim one-hot array from n dim int-categorical array."""
  return np.array(x[..., np.newaxis] == np.arange(size), dtype)

def url_to_image(url):
    """
    Fetch an image from url and convert it into a Pillow Image object
    """
    r = requests.get(url)
    image = StringIO(r.content)
    return image

def home():
    """Temporary helper function to link to the API routes"""
    return dict(links=dict(api='{}{}'.format(request.url, PREFIX[1:]))), \
        HTTPStatus.OK

def get(key, default=None):
    """ return the key from the request
    """
    data = get_form() or get_query_string()
    return data.get(key, default)

def debug_src(src, pm=False, globs=None):
    """Debug a single doctest docstring, in argument `src`'"""
    testsrc = script_from_examples(src)
    debug_script(testsrc, pm, globs)

def difference(ydata1, ydata2):
    """

    Returns the number you should add to ydata1 to make it line up with ydata2

    """

    y1 = _n.array(ydata1)
    y2 = _n.array(ydata2)

    return(sum(y2-y1)/len(ydata1))

def to_simple_rdd(sc, features, labels):
    """Convert numpy arrays of features and labels into
    an RDD of pairs.

    :param sc: Spark context
    :param features: numpy array with features
    :param labels: numpy array with labels
    :return: Spark RDD with feature-label pairs
    """
    pairs = [(x, y) for x, y in zip(features, labels)]
    return sc.parallelize(pairs)

def get_available_gpus():
  """
  Returns a list of string names of all available GPUs
  """
  local_device_protos = device_lib.list_local_devices()
  return [x.name for x in local_device_protos if x.device_type == 'GPU']

def create_task(coro, loop):
    # pragma: no cover
    """Compatibility wrapper for the loop.create_task() call introduced in
    3.4.2."""
    if hasattr(loop, 'create_task'):
        return loop.create_task(coro)
    return asyncio.Task(coro, loop=loop)

def conv_block(inputs, filters, dilation_rates_and_kernel_sizes, **kwargs):
  """A block of standard 2d convolutions."""
  return conv_block_internal(conv, inputs, filters,
                             dilation_rates_and_kernel_sizes, **kwargs)

def extract_module_locals(depth=0):
    """Returns (module, locals) of the funciton `depth` frames away from the caller"""
    f = sys._getframe(depth + 1)
    global_ns = f.f_globals
    module = sys.modules[global_ns['__name__']]
    return (module, f.f_locals)

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

def get_file_md5sum(path):
    """Calculate the MD5 hash for a file."""
    with open(path, 'rb') as fh:
        h = str(hashlib.md5(fh.read()).hexdigest())
    return h

def voronoi(data, line_color=None, line_width=2, f_tooltip=None, cmap=None, max_area=1e4, alpha=220):
    """
    Draw the voronoi tesselation of the points

    :param data: data access object
    :param line_color: line color
    :param line_width: line width
    :param f_tooltip: function to generate a tooltip on mouseover
    :param cmap: color map
    :param max_area: scaling constant to determine the color of the voronoi areas
    :param alpha: color alpha
    """
    from geoplotlib.layers import VoronoiLayer
    _global_config.layers.append(VoronoiLayer(data, line_color, line_width, f_tooltip, cmap, max_area, alpha))

def bytesize(arr):
    """
    Returns the memory byte size of a Numpy array as an integer.
    """
    byte_size = np.prod(arr.shape) * np.dtype(arr.dtype).itemsize
    return byte_size

def _elapsed_time(begin_time, end_time):
    """Assuming format YYYY-MM-DD hh:mm:ss

    Returns the elapsed time in seconds
    """

    bt = _str2datetime(begin_time)
    et = _str2datetime(end_time)

    return float((et - bt).seconds)

def from_json_list(cls, api_client, data):
        """Convert a list of JSON values to a list of models
        """
        return [cls.from_json(api_client, item) for item in data]

def get_var(self, name):
        """ Returns the variable set with the given name.
        """
        for var in self.vars:
            if var.name == name:
                return var
        else:
            raise ValueError

def c_array(ctype, values):
    """Convert a python string to c array."""
    if isinstance(values, np.ndarray) and values.dtype.itemsize == ctypes.sizeof(ctype):
        return (ctype * len(values)).from_buffer_copy(values)
    return (ctype * len(values))(*values)

def node__name__(self):
        """Return the name of this node or its class name."""

        return self.node.__name__ \
            if self.node.__name__ is not None else self.node.__class__.__name__

def pointer(self):
        """Get a ctypes void pointer to the memory mapped region.

        :type: ctypes.c_void_p
        """
        return ctypes.cast(ctypes.pointer(ctypes.c_uint8.from_buffer(self.mapping, 0)), ctypes.c_void_p)

def load_object_by_name(object_name):
    """Load an object from a module by name"""
    mod_name, attr = object_name.rsplit('.', 1)
    mod = import_module(mod_name)
    return getattr(mod, attr)

def pointer(self):
        """Get a ctypes void pointer to the memory mapped region.

        :type: ctypes.c_void_p
        """
        return ctypes.cast(ctypes.pointer(ctypes.c_uint8.from_buffer(self.mapping, 0)), ctypes.c_void_p)

def exp_fit_fun(x, a, tau, c):
    """Function used to fit the exponential decay."""
    # pylint: disable=invalid-name
    return a * np.exp(-x / tau) + c

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

def gray2bgr(img):
    """Convert a grayscale image to BGR image.

    Args:
        img (ndarray or str): The input image.

    Returns:
        ndarray: The converted BGR image.
    """
    img = img[..., None] if img.ndim == 2 else img
    out_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return out_img

def matrix_to_gl(matrix):
    """
    Convert a numpy row- major homogenous transformation matrix
    to a flat column- major GLfloat transformation.

    Parameters
    -------------
    matrix : (4,4) float
      Row- major homogenous transform

    Returns
    -------------
    glmatrix : (16,) gl.GLfloat
      Transform in pyglet format
    """
    matrix = np.asanyarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError('matrix must be (4,4)!')

    # switch to column major and flatten to (16,)
    column = matrix.T.flatten()
    # convert to GLfloat
    glmatrix = (gl.GLfloat * 16)(*column)

    return glmatrix

def is_timestamp(instance):
    """Validates data is a timestamp"""
    if not isinstance(instance, (int, str)):
        return True
    return datetime.fromtimestamp(int(instance))

def check_output(args):
    """Runs command and returns the output as string."""
    log.debug('run: %s', args)
    out = subprocess.check_output(args=args).decode('utf-8')
    log.debug('out: %r', out)
    return out

def format_timestamp(timestamp):
        """
        Format the UTC timestamp for Elasticsearch
        eg. 2014-07-09T08:37:18.000Z

        @see https://docs.python.org/2/library/time.html#time.strftime

        :type timestamp int
        :rtype: str
        """
        tz_info = tz.tzutc()
        return datetime.fromtimestamp(timestamp, tz=tz_info).strftime("%Y-%m-%dT%H:%M:%S.000Z")

def get_page_text(self, page):
        """
        Downloads and returns the full text of a particular page
        in the document.
        """
        url = self.get_page_text_url(page)
        return self._get_url(url)

def localize(dt):
    """Localize a datetime object to local time."""
    if dt.tzinfo is UTC:
        return (dt + LOCAL_UTC_OFFSET).replace(tzinfo=None)
    # No TZ info so not going to assume anything, return as-is.
    return dt

def get_parent_dir(name):
    """Get the parent directory of a filename."""
    parent_dir = os.path.dirname(os.path.dirname(name))
    if parent_dir:
        return parent_dir
    return os.path.abspath('.')

def convert_2_utc(self, datetime_, timezone):
        """convert to datetime to UTC offset."""

        datetime_ = self.tz_mapper[timezone].localize(datetime_)
        return datetime_.astimezone(pytz.UTC)

def get_parent_folder_name(file_path):
    """Finds parent folder of file

    :param file_path: path
    :return: Name of folder container
    """
    return os.path.split(os.path.split(os.path.abspath(file_path))[0])[-1]

def _DateToEpoch(date):
  """Converts python datetime to epoch microseconds."""
  tz_zero = datetime.datetime.utcfromtimestamp(0)
  diff_sec = int((date - tz_zero).total_seconds())
  return diff_sec * 1000000

def read(filename):
    """Read and return `filename` in root dir of project and return string"""
    return codecs.open(os.path.join(__DIR__, filename), 'r').read()

def convert_timestamp(timestamp):
    """
    Converts bokehJS timestamp to datetime64.
    """
    datetime = dt.datetime.utcfromtimestamp(timestamp/1000.)
    return np.datetime64(datetime.replace(tzinfo=None))

def family_directory(fonts):
  """Get the path of font project directory."""
  if fonts:
    dirname = os.path.dirname(fonts[0])
    if dirname == '':
      dirname = '.'
    return dirname

def decode_bytes(string):
    """ Decodes a given base64 string into bytes.

    :param str string: The string to decode
    :return: The decoded bytes
    :rtype: bytes
    """

    if is_string_type(type(string)):
        string = bytes(string, "utf-8")
    return base64.decodebytes(string)

def security(self):
        """Print security object information for a pdf document"""
        return {k: v for i in self.pdf.resolvedObjects.items() for k, v in i[1].items()}

def intToBin(i):
    """ Integer to two bytes """
    # devide in two parts (bytes)
    i1 = i % 256
    i2 = int(i / 256)
    # make string (little endian)
    return chr(i1) + chr(i2)

def p(self):
        """
        Helper property containing the percentage this slider is "filled".
        
        This property is read-only.
        """
        return (self.n-self.nmin)/max((self.nmax-self.nmin),1)

def isbinary(*args):
    """Checks if value can be part of binary/bitwise operations."""
    return all(map(lambda c: isnumber(c) or isbool(c), args))

def extra_funcs(*funcs):
  """Decorator which adds extra functions to be downloaded to the pyboard."""
  def extra_funcs_decorator(real_func):
    def wrapper(*args, **kwargs):
      return real_func(*args, **kwargs)
    wrapper.extra_funcs = list(funcs)
    wrapper.source = inspect.getsource(real_func)
    wrapper.name = real_func.__name__
    return wrapper
  return extra_funcs_decorator

def csv_to_dicts(file, header=None):
    """Reads a csv and returns a List of Dicts with keys given by header row."""
    with open(file) as csvfile:
        return [row for row in csv.DictReader(csvfile, fieldnames=header)]

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

def basic_word_sim(word1, word2):
    """
    Simple measure of similarity: Number of letters in common / max length
    """
    return sum([1 for c in word1 if c in word2]) / max(len(word1), len(word2))

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

def remove(parent, idx):
  """Remove a value from a dict."""
  if isinstance(parent, dict):
    del parent[idx]
  elif isinstance(parent, list):
    del parent[int(idx)]
  else:
    raise JSONPathError("Invalid path for operation")

def unique(list):
    """ Returns a copy of the list without duplicates.
    """
    unique = []; [unique.append(x) for x in list if x not in unique]
    return unique

def purge_dict(idict):
    """Remove null items from a dictionary """
    odict = {}
    for key, val in idict.items():
        if is_null(val):
            continue
        odict[key] = val
    return odict

def readwav(filename):
    """Read a WAV file and returns the data and sample rate

    ::

        from spectrum.io import readwav
        readwav()

    """
    from scipy.io.wavfile import read as readwav
    samplerate, signal = readwav(filename)
    return signal, samplerate

def invalidate_cache(cpu, address, size):
        """ remove decoded instruction from instruction cache """
        cache = cpu.instruction_cache
        for offset in range(size):
            if address + offset in cache:
                del cache[address + offset]

def option2tuple(opt):
    """Return a tuple of option, taking possible presence of level into account"""

    if isinstance(opt[0], int):
        tup = opt[1], opt[2:]
    else:
        tup = opt[0], opt[1:]

    return tup

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

def _run_cmd_get_output(cmd):
    """Runs a shell command, returns console output.

    Mimics python3's subprocess.getoutput
    """
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    out, err = process.communicate()
    return out or err

def clean_out_dir(directory):
    """
    Delete all the files and subdirectories in a directory.
    """
    if not isinstance(directory, path):
        directory = path(directory)
    for file_path in directory.files():
        file_path.remove()
    for dir_path in directory.dirs():
        dir_path.rmtree()

def get_width():
    """Get terminal width"""
    # Get terminal size
    ws = struct.pack("HHHH", 0, 0, 0, 0)
    ws = fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, ws)
    lines, columns, x, y = struct.unpack("HHHH", ws)
    width = min(columns * 39 // 40, columns - 2)
    return width

def _normalize_abmn(abmn):
    """return a normalized version of abmn
    """
    abmn_2d = np.atleast_2d(abmn)
    abmn_normalized = np.hstack((
        np.sort(abmn_2d[:, 0:2], axis=1),
        np.sort(abmn_2d[:, 2:4], axis=1),
    ))
    return abmn_normalized

def size(dtype):
  """Returns the number of bytes to represent this `dtype`."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'size'):
    return dtype.size
  return np.dtype(dtype).itemsize

def make_bintree(levels):
    """Make a symmetrical binary tree with @levels"""
    G = nx.DiGraph()
    root = '0'
    G.add_node(root)
    add_children(G, root, levels, 2)
    return G

def index(self, value):
		"""
		Return the smallest index of the row(s) with this column
		equal to value.
		"""
		for i in xrange(len(self.parentNode)):
			if getattr(self.parentNode[i], self.Name) == value:
				return i
		raise ValueError(value)

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

def __getitem__(self, index):
    """Get the item at the given index.

    Index is a tuple of (row, col)
    """
    row, col = index
    return self.rows[row][col]

def update_screen(self):
        """Refresh the screen. You don't need to override this except to update only small portins of the screen."""
        self.clock.tick(self.FPS)
        pygame.display.update()

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

def calc_volume(self, sample: np.ndarray):
        """Find the RMS of the audio"""
        return sqrt(np.mean(np.square(sample)))

def _get_local_ip(self):
        """Try to determine the local IP address of the machine."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            # Use Google Public DNS server to determine own IP
            sock.connect(('8.8.8.8', 80))

            return sock.getsockname()[0]
        except socket.error:
            try:
                return socket.gethostbyname(socket.gethostname())
            except socket.gaierror:
                return '127.0.0.1'
        finally:
            sock.close()

def round_to_float(number, precision):
    """Round a float to a precision"""
    rounded = Decimal(str(floor((number + precision / 2) // precision))
                      ) * Decimal(str(precision))
    return float(rounded)

def getScriptLocation():
	"""Helper function to get the location of a Python file."""
	location = os.path.abspath("./")
	if __file__.rfind("/") != -1:
		location = __file__[:__file__.rfind("/")]
	return location

def series_table_row_offset(self, series):
        """
        Return the number of rows preceding the data table for *series* in
        the Excel worksheet.
        """
        title_and_spacer_rows = series.index * 2
        data_point_rows = series.data_point_offset
        return title_and_spacer_rows + data_point_rows

def class_name(obj):
    """
    Get the name of an object, including the module name if available.
    """

    name = obj.__name__
    module = getattr(obj, '__module__')

    if module:
        name = f'{module}.{name}'
    return name

def initialize_api(flask_app):
    """Initialize an API."""
    if not flask_restplus:
        return

    api = flask_restplus.Api(version="1.0", title="My Example API")
    api.add_resource(HelloWorld, "/hello")

    blueprint = flask.Blueprint("api", __name__, url_prefix="/api")
    api.init_app(blueprint)
    flask_app.register_blueprint(blueprint)

def class_name(obj):
    """
    Get the name of an object, including the module name if available.
    """

    name = obj.__name__
    module = getattr(obj, '__module__')

    if module:
        name = f'{module}.{name}'
    return name

def register_extension_class(ext, base, *args, **kwargs):
    """Instantiate the given extension class and register as a public attribute of the given base.

    README: The expected protocol here is to instantiate the given extension and pass the base
    object as the first positional argument, then unpack args and kwargs as additional arguments to
    the extension's constructor.
    """
    ext_instance = ext.plugin(base, *args, **kwargs)
    setattr(base, ext.name.lstrip('_'), ext_instance)

def dir_modtime(dpath):
    """
    Returns the latest modification time of all files/subdirectories in a
    directory
    """
    return max(os.path.getmtime(d) for d, _, _ in os.walk(dpath))

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

def get_top(self, *args, **kwargs):
        """Return a get_content generator for top submissions.

        Corresponds to the submissions provided by
        ``https://www.reddit.com/top/`` for the session.

        The additional parameters are passed directly into
        :meth:`.get_content`. Note: the `url` parameter cannot be altered.

        """
        return self.get_content(self.config['top'], *args, **kwargs)

def _rectangular(n):
    """Checks to see if a 2D list is a valid 2D matrix"""
    for i in n:
        if len(i) != len(n[0]):
            return False
    return True

def url(self):
        """ The url of this window """
        with switch_window(self._browser, self.name):
            return self._browser.url

def size(self):
        """The size of this parameter, equivalent to self.value.size"""
        return np.multiply.reduce(self.shape, dtype=np.int32)

def read_credentials(fname):
    """
    read a simple text file from a private location to get
    username and password
    """
    with open(fname, 'r') as f:
        username = f.readline().strip('\n')
        password = f.readline().strip('\n')
    return username, password

def length(self):
        """Array of vector lengths"""
        return np.sqrt(np.sum(self**2, axis=1)).view(np.ndarray)

def get_user_by_id(self, id):
        """Retrieve a User object by ID."""
        return self.db_adapter.get_object(self.UserClass, id=id)

async def join(self, ctx, *, channel: discord.VoiceChannel):
        """Joins a voice channel"""

        if ctx.voice_client is not None:
            return await ctx.voice_client.move_to(channel)

        await channel.connect()

def load_object_by_name(object_name):
    """Load an object from a module by name"""
    mod_name, attr = object_name.rsplit('.', 1)
    mod = import_module(mod_name)
    return getattr(mod, attr)

async def join(self, ctx, *, channel: discord.VoiceChannel):
        """Joins a voice channel"""

        if ctx.voice_client is not None:
            return await ctx.voice_client.move_to(channel)

        await channel.connect()

def OnMove(self, event):
        """Main window move event"""

        # Store window position in config
        position = self.main_window.GetScreenPositionTuple()

        config["window_position"] = repr(position)

def _handle_chat_name(self, data):
        """Handle user name changes"""

        self.room.user.nick = data
        self.conn.enqueue_data("user", self.room.user)

def size():
    """Determines the height and width of the console window

        Returns:
            tuple of int: The height in lines, then width in characters
    """
    try:
        assert os != 'nt' and sys.stdout.isatty()
        rows, columns = os.popen('stty size', 'r').read().split()
    except (AssertionError, AttributeError, ValueError):
        # in case of failure, use dimensions of a full screen 13" laptop
        rows, columns = DEFAULT_HEIGHT, DEFAULT_WIDTH

    return int(rows), int(columns)

def table_top_abs(self):
        """Returns the absolute position of table top"""
        table_height = np.array([0, 0, self.table_full_size[2]])
        return string_to_array(self.floor.get("pos")) + table_height

def auth_request(self, url, headers, body):
        """Perform auth request for token."""

        return self.req.post(url, headers, body=body)

def get_plain_image_as_widget(self):
        """Used for generating thumbnails.  Does not include overlaid
        graphics.
        """
        arr = self.getwin_array(order=self.rgb_order)
        image = self._get_qimage(arr, self.qimg_fmt)
        return image

def _get_loggers():
    """Return list of Logger classes."""
    from .. import loader
    modules = loader.get_package_modules('logger')
    return list(loader.get_plugins(modules, [_Logger]))

def dimensions(self):
        """Get width and height of a PDF"""
        size = self.pdf.getPage(0).mediaBox
        return {'w': float(size[2]), 'h': float(size[3])}

def fetch_event(urls):
    """
    This parallel fetcher uses gevent one uses gevent
    """
    rs = (grequests.get(u) for u in urls)
    return [content.json() for content in grequests.map(rs)]

def copy(self):
        """
        Creates a copy of model
        """
        return self.__class__(field_type=self.get_field_type(), data=self.export_data())

def fetch_event(urls):
    """
    This parallel fetcher uses gevent one uses gevent
    """
    rs = (grequests.get(u) for u in urls)
    return [content.json() for content in grequests.map(rs)]

def fast_distinct(self):
        """
        Because standard distinct used on the all fields are very slow and works only with PostgreSQL database
        this method provides alternative to the standard distinct method.
        :return: qs with unique objects
        """
        return self.model.objects.filter(pk__in=self.values_list('pk', flat=True))

def ffmpeg_works():
  """Tries to encode images with ffmpeg to check if it works."""
  images = np.zeros((2, 32, 32, 3), dtype=np.uint8)
  try:
    _encode_gif(images, 2)
    return True
  except (IOError, OSError):
    return False

def managepy(cmd, extra=None):
    """Run manage.py using this component's specific Django settings"""

    extra = extra.split() if extra else []
    run_django_cli(['invoke', cmd] + extra)

def branches(self):
        """All branches in a list"""
        result = self.git(self.default + ['branch', '-a', '--no-color'])
        return [l.strip(' *\n') for l in result.split('\n') if l.strip(' *\n')]

def get_qualified_name(_object):
    """Return the Fully Qualified Name from an instance or class."""
    module = _object.__module__
    if hasattr(_object, '__name__'):
        _class = _object.__name__

    else:
        _class = _object.__class__.__name__

    return module + '.' + _class

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

def __init__(self, form_post_data=None, *args, **kwargs):
        """
        Overriding init so we can set the post vars like a normal form and generate
        the form the same way Django does.
        """
        kwargs.update({'form_post_data': form_post_data})
        super(MongoModelForm, self).__init__(*args, **kwargs)

def security(self):
        """Print security object information for a pdf document"""
        return {k: v for i in self.pdf.resolvedObjects.items() for k, v in i[1].items()}

def object_to_json(obj, indent=2):
    """
        transform object to json
    """
    instance_json = json.dumps(obj, indent=indent, ensure_ascii=False, cls=DjangoJSONEncoder)
    return instance_json

def _text_to_graphiz(self, text):
        """create a graphviz graph from text"""
        dot = Source(text, format='svg')
        return dot.pipe().decode('utf-8')

def full_like(array, value, dtype=None):
    """ Create a shared memory array with the same shape and type as a given array, filled with `value`.
    """
    shared = empty_like(array, dtype)
    shared[:] = value
    return shared

def weighted_std(values, weights):
    """ Calculate standard deviation weighted by errors """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return np.sqrt(variance)

def _check_conversion(key, valid_dict):
    """Check for existence of key in dict, return value or raise error"""
    if key not in valid_dict and key not in valid_dict.values():
        # Only show users the nice string values
        keys = [v for v in valid_dict.keys() if isinstance(v, string_types)]
        raise ValueError('value must be one of %s, not %s' % (keys, key))
    return valid_dict[key] if key in valid_dict else key

def __del__(self):
        """Frees all resources.
        """
        if hasattr(self, '_Api'):
            self._Api.close()

        self._Logger.info('object destroyed')

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

def hex_to_hsv(color):
    """
    Converts from hex to hsv

    Parameters:
    -----------
            color : string
                    Color representation on color

    Example:
            hex_to_hsv('#ff9933')
    """
    color = normalize(color)
    color = color[1:]
    # color=tuple(ord(c)/255.0 for c in color.decode('hex'))
    color = (int(color[0:2], base=16) / 255.0, int(color[2:4],
                                                   base=16) / 255.0, int(color[4:6], base=16) / 255.0)
    return colorsys.rgb_to_hsv(*color)

def polite_string(a_string):
    """Returns a "proper" string that should work in both Py3/Py2"""
    if is_py3() and hasattr(a_string, 'decode'):
        try:
            return a_string.decode('utf-8')
        except UnicodeDecodeError:
            return a_string

    return a_string

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

def action(self):
        """
        This class overrides this method
        """
        self.return_value = self.function(*self.args, **self.kwargs)

def _open_url(url):
    """Open a HTTP connection to the URL and return a file-like object."""
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise IOError("Unable to download {}, HTTP {}".format(url, response.status_code))
    return response

def set_parent_path(self, value):
        """
        Set the parent path and the path from the new parent path.

        :param value: The path to the object's parent
        """

        self._parent_path = value
        self.path = value + r'/' + self.name
        self._update_childrens_parent_path()

def hline(self, x, y, width, color):
        """Draw a horizontal line up to a given length."""
        self.rect(x, y, width, 1, color, fill=True)

def normalize_time(timestamp):
    """Normalize time in arbitrary timezone to UTC naive object."""
    offset = timestamp.utcoffset()
    if offset is None:
        return timestamp
    return timestamp.replace(tzinfo=None) - offset

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

def inverse_transform(self, X):
        """Undo the scaling of X according to feature_range.

        Note that if truncate is true, any truncated points will not
        be restored exactly.

        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            Input data that will be transformed.
        """
        X = check_array(X, copy=self.copy)
        X -= self.min_
        X /= self.scale_
        return X

def do_serial(self, p):
		"""Set the serial port, e.g.: /dev/tty.usbserial-A4001ib8"""
		try:
			self.serial.port = p
			self.serial.open()
			print 'Opening serial port: %s' % p
		except Exception, e:
			print 'Unable to open serial port: %s' % p

def closeEvent(self, e):
        """Qt slot when the window is closed."""
        self.emit('close_widget')
        super(DockWidget, self).closeEvent(e)

def wipe(self):
        """ Wipe the store
        """
        keys = list(self.keys()).copy()
        for key in keys:
            self.delete(key)

def datetime_to_timezone(date, tz="UTC"):
    """ convert naive datetime to timezone-aware datetime """
    if not date.tzinfo:
        date = date.replace(tzinfo=timezone(get_timezone()))
    return date.astimezone(timezone(tz))

def unit_key_from_name(name):
  """Return a legal python name for the given name for use as a unit key."""
  result = name

  for old, new in six.iteritems(UNIT_KEY_REPLACEMENTS):
    result = result.replace(old, new)

  # Collapse redundant underscores and convert to uppercase.
  result = re.sub(r'_+', '_', result.upper())

  return result

def geodetic_to_ecef(latitude, longitude, altitude):
    """Convert WGS84 geodetic coordinates into ECEF
    
    Parameters
    ----------
    latitude : float or array_like
        Geodetic latitude (degrees)
    longitude : float or array_like
        Geodetic longitude (degrees)
    altitude : float or array_like
        Geodetic Height (km) above WGS84 reference ellipsoid.
        
    Returns
    -------
    x, y, z
        numpy arrays of x, y, z locations in km
        
    """


    ellip = np.sqrt(1. - earth_b ** 2 / earth_a ** 2)
    r_n = earth_a / np.sqrt(1. - ellip ** 2 * np.sin(np.deg2rad(latitude)) ** 2)

    # colatitude = 90. - latitude
    x = (r_n + altitude) * np.cos(np.deg2rad(latitude)) * np.cos(np.deg2rad(longitude))
    y = (r_n + altitude) * np.cos(np.deg2rad(latitude)) * np.sin(np.deg2rad(longitude))
    z = (r_n * (1. - ellip ** 2) + altitude) * np.sin(np.deg2rad(latitude))

    return x, y, z

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

def scan(client, query=None, scroll='5m', raise_on_error=True,
         preserve_order=False, size=1000, **kwargs):
    """
    Simple abstraction on top of the
    :meth:`~elasticsearch.Elasticsearch.scroll` api - a simple iterator that
    yields all hits as returned by underlining scroll requests.
    By default scan does not return results in any pre-determined order. To
    have a standard order in the returned documents (either by score or
    explicit sort definition) when scrolling, use ``preserve_order=True``. This
    may be an expensive operation and will negate the performance benefits of
    using ``scan``.
    :arg client: instance of :class:`~elasticsearch.Elasticsearch` to use
    :arg query: body for the :meth:`~elasticsearch.Elasticsearch.search` api
    :arg scroll: Specify how long a consistent view of the index should be
        maintained for scrolled search
    :arg raise_on_error: raises an exception (``ScanError``) if an error is
        encountered (some shards fail to execute). By default we raise.
    :arg preserve_order: don't set the ``search_type`` to ``scan`` - this will
        cause the scroll to paginate with preserving the order. Note that this
        can be an extremely expensive operation and can easily lead to
        unpredictable results, use with caution.
    :arg size: size (per shard) of the batch send at each iteration.
    Any additional keyword arguments will be passed to the initial
    :meth:`~elasticsearch.Elasticsearch.search` call::
        scan(es,
            query={"query": {"match": {"title": "python"}}},
            index="orders-*",
            doc_type="books"
        )
    """
    if not preserve_order:
        kwargs['search_type'] = 'scan'
    # initial search
    resp = client.search(body=query, scroll=scroll, size=size, **kwargs)

    scroll_id = resp.get('_scroll_id')
    if scroll_id is None:
        return

    first_run = True
    while True:
        # if we didn't set search_type to scan initial search contains data
        if preserve_order and first_run:
            first_run = False
        else:
            resp = client.scroll(scroll_id, scroll=scroll)

        for hit in resp['hits']['hits']:
            yield hit

        # check if we have any errrors
        if resp["_shards"]["failed"]:
            logger.warning(
                'Scroll request has failed on %d shards out of %d.',
                resp['_shards']['failed'], resp['_shards']['total']
            )
            if raise_on_error:
                raise ScanError(
                    'Scroll request has failed on %d shards out of %d.' %
                    (resp['_shards']['failed'], resp['_shards']['total'])
                )

        scroll_id = resp.get('_scroll_id')
        # end of scroll
        if scroll_id is None or not resp['hits']['hits']:
            break

def get_file_size(filename):
    """
    Get the file size of a given file

    :param filename: string: pathname of a file
    :return: human readable filesize
    """
    if os.path.isfile(filename):
        return convert_size(os.path.getsize(filename))
    return None

def _to_lower_alpha_only(s):
    """Return a lowercased string with non alphabetic chars removed.

    White spaces are not to be removed."""
    s = re.sub(r'\n', ' ',  s.lower())
    return re.sub(r'[^a-z\s]', '', s)

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

def rnormal(mu, tau, size=None):
    """
    Random normal variates.
    """
    return np.random.normal(mu, 1. / np.sqrt(tau), size)

def is_valid_regex(regex):
    """Function for checking a valid regex."""
    if len(regex) == 0:
        return False
    try:
        re.compile(regex)
        return True
    except sre_constants.error:
        return False

def requests_post(url, data=None, json=None, **kwargs):
    """Requests-mock requests.post wrapper."""
    return requests_request('post', url, data=data, json=json, **kwargs)

def check_length(value, length):
    """
    Checks length of value

    @param value: value to check
    @type value: C{str}

    @param length: length checking for
    @type length: C{int}

    @return: None when check successful

    @raise ValueError: check failed
    """
    _length = len(value)
    if _length != length:
        raise ValueError("length must be %d, not %d" % \
                         (length, _length))

def session_to_epoch(timestamp):
    """ converts Synergy Timestamp for session to UTC zone seconds since epoch """
    utc_timetuple = datetime.strptime(timestamp, SYNERGY_SESSION_PATTERN).replace(tzinfo=None).utctimetuple()
    return calendar.timegm(utc_timetuple)

def is_type(value):
        """Determine if value is an instance or subclass of the class Type."""
        if isinstance(value, type):
            return issubclass(value, Type)
        return isinstance(value, Type)

def session_to_epoch(timestamp):
    """ converts Synergy Timestamp for session to UTC zone seconds since epoch """
    utc_timetuple = datetime.strptime(timestamp, SYNERGY_SESSION_PATTERN).replace(tzinfo=None).utctimetuple()
    return calendar.timegm(utc_timetuple)

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

def is_type(value):
        """Determine if value is an instance or subclass of the class Type."""
        if isinstance(value, type):
            return issubclass(value, Type)
        return isinstance(value, Type)

def hard_equals(a, b):
    """Implements the '===' operator."""
    if type(a) != type(b):
        return False
    return a == b

def chunk_list(l, n):
    """Return `n` size lists from a given list `l`"""
    return [l[i:i + n] for i in range(0, len(l), n)]

def _escape(self, s):
        """Escape bad characters for regular expressions.

        Similar to `re.escape` but allows '%' to pass through.

        """
        for ch, r_ch in self.ESCAPE_SETS:
            s = s.replace(ch, r_ch)
        return s

def copy(string, **kwargs):
    """Copy given string into system clipboard."""
    window = Tk()
    window.withdraw()
    window.clipboard_clear()
    window.clipboard_append(string)
    window.destroy()
    return

def quote(s, unsafe='/'):
    """Pass in a dictionary that has unsafe characters as the keys, and the percent
    encoded value as the value."""
    res = s.replace('%', '%25')
    for c in unsafe:
        res = res.replace(c, '%' + (hex(ord(c)).upper())[2:])
    return res

def __call__(self, xy):
        """Project x and y"""
        x, y = xy
        return (self.x(x), self.y(y))

def fast_exit(code):
    """Exit without garbage collection, this speeds up exit by about 10ms for
    things like bash completion.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)

def __init__(self, ba=None):
        """Constructor."""
        self.bytearray = ba or (bytearray(b'\0') * self.SIZEOF)

def relative_path(path):
    """
    Return the given path relative to this file.
    """
    return os.path.join(os.path.dirname(__file__), path)

def safe_rmtree(directory):
  """Delete a directory if it's present. If it's not present, no-op."""
  if os.path.exists(directory):
    shutil.rmtree(directory, True)

def double_exponential_moving_average(data, period):
    """
    Double Exponential Moving Average.

    Formula:
    DEMA = 2*EMA - EMA(EMA)
    """
    catch_errors.check_for_period_error(data, period)

    dema = (2 * ema(data, period)) - ema(ema(data, period), period)
    return dema

def is_file_url(url):
    """Returns true if the given url is a file url"""
    from .misc import to_text

    if not url:
        return False
    if not isinstance(url, six.string_types):
        try:
            url = getattr(url, "url")
        except AttributeError:
            raise ValueError("Cannot parse url from unknown type: {0!r}".format(url))
    url = to_text(url, encoding="utf-8")
    return urllib_parse.urlparse(url.lower()).scheme == "file"

def save_image(pdf_path, img_path, page_num):
    """

    Creates images for a page of the input pdf document and saves it
    at img_path.

    :param pdf_path: path to pdf to create images for.
    :param img_path: path where to save the images.
    :param page_num: page number to create image from in the pdf file.
    :return:
    """
    pdf_img = Image(filename="{}[{}]".format(pdf_path, page_num))
    with pdf_img.convert("png") as converted:
        # Set white background.
        converted.background_color = Color("white")
        converted.alpha_channel = "remove"
        converted.save(filename=img_path)

def go_to_new_line(self):
        """Go to the end of the current line and create a new line"""
        self.stdkey_end(False, False)
        self.insert_text(self.get_line_separator())

def version_triple(tag):
    """
    returns: a triple of integers from a version tag
    """
    groups = re.match(r'v?(\d+)\.(\d+)\.(\d+)', tag).groups()
    return tuple(int(n) for n in groups)

def __next__(self, reward, ask_id, lbl):
        """For Python3 compatibility of generator."""
        return self.next(reward, ask_id, lbl)

def contains_extractor(document):
    """A basic document feature extractor that returns a dict of words that the
    document contains."""
    tokens = _get_document_tokens(document)
    features = dict((u'contains({0})'.format(w), True) for w in tokens)
    return features

def download_json(local_filename, url, clobber=False):
    """Download the given JSON file, and pretty-print before we output it."""
    with open(local_filename, 'w') as json_file:
        json_file.write(json.dumps(requests.get(url).json(), sort_keys=True, indent=2, separators=(',', ': ')))

def check_precomputed_distance_matrix(X):
    """Perform check_array(X) after removing infinite values (numpy.inf) from the given distance matrix.
    """
    tmp = X.copy()
    tmp[np.isinf(tmp)] = 1
    check_array(tmp)

def clean_tmpdir(path):
    """Invoked atexit, this removes our tmpdir"""
    if os.path.exists(path) and \
       os.path.isdir(path):
        rmtree(path)

def _heappop_max(heap):
    """Maxheap version of a heappop."""
    lastelt = heap.pop()    # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        _siftup_max(heap, 0)
        return returnitem
    return lastelt

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

def iterate(obj):
	"""Loop over an iterable and track progress, including first and last state.
	
	On each iteration yield an Iteration named tuple with the first and last flags, current element index, total
	iterable length (if possible to acquire), and value, in that order.
	
		for iteration in iterate(something):
			iteration.value  # Do something.
	
	You can unpack these safely:
	
		for first, last, index, total, value in iterate(something):
			pass
	
	If you want to unpack the values you are iterating across, you can by wrapping the nested unpacking in parenthesis:
	
		for first, last, index, total, (foo, bar, baz) in iterate(something):
			pass
	
	Even if the length of the iterable can't be reliably determined this function will still capture the "last" state
	of the final loop iteration.  (Basically: this works with generators.)
	
	This process is about 10x slower than simple enumeration on CPython 3.4, so only use it where you actually need to
	track state.  Use `enumerate()` elsewhere.
	"""
	
	global next, Iteration
	next = next
	Iteration = Iteration
	
	total = len(obj) if isinstance(obj, Sized) else None
	iterator = iter(obj)
	first = True
	last = False
	i = 0
	
	try:
		value = next(iterator)
	except StopIteration:
		return
	
	while True:
		try:
			next_value = next(iterator)
		except StopIteration:
			last = True
		
		yield Iteration(first, last, i, total, value)
		if last: return
		
		value = next_value
		i += 1
		first = False

def get_method_name(method):
    """
    Returns given method name.

    :param method: Method to retrieve the name.
    :type method: object
    :return: Method name.
    :rtype: unicode
    """

    name = get_object_name(method)
    if name.startswith("__") and not name.endswith("__"):
        name = "_{0}{1}".format(get_object_name(method.im_class), name)
    return name

def _select_features(example, feature_list=None):
  """Select a subset of features from the example dict."""
  feature_list = feature_list or ["inputs", "targets"]
  return {f: example[f] for f in feature_list}

def datatype(dbtype, description, cursor):
    """Google AppEngine Helper to convert a data type into a string."""
    dt = cursor.db.introspection.get_field_type(dbtype, description)
    if type(dt) is tuple:
        return dt[0]
    else:
        return dt

def get_last(self, table=None):
        """Just the last entry."""
        if table is None: table = self.main_table
        query = 'SELECT * FROM "%s" ORDER BY ROWID DESC LIMIT 1;' % table
        return self.own_cursor.execute(query).fetchone()

def get_list_dimensions(_list):
    """
    Takes a nested list and returns the size of each dimension followed
    by the element type in the list
    """
    if isinstance(_list, list) or isinstance(_list, tuple):
        return [len(_list)] + get_list_dimensions(_list[0])
    return []

def stft_magnitude(signal, fft_length,
                   hop_length=None,
                   window_length=None):
  """Calculate the short-time Fourier transform magnitude.

  Args:
    signal: 1D np.array of the input time-domain signal.
    fft_length: Size of the FFT to apply.
    hop_length: Advance (in samples) between each frame passed to FFT.
    window_length: Length of each block of samples to pass to FFT.

  Returns:
    2D np.array where each row contains the magnitudes of the fft_length/2+1
    unique values of the FFT for the corresponding frame of input samples.
  """
  frames = frame(signal, window_length, hop_length)
  # Apply frame window to each frame. We use a periodic Hann (cosine of period
  # window_length) instead of the symmetric Hann of np.hanning (period
  # window_length-1).
  window = periodic_hann(window_length)
  windowed_frames = frames * window
  return np.abs(np.fft.rfft(windowed_frames, int(fft_length)))

def wget(url):
    """
    Download the page into a string
    """
    import urllib.parse
    request = urllib.request.urlopen(url)
    filestring = request.read()
    return filestring

def _read_text(self, filename):
        """
        Helper that reads the UTF-8 content of the specified file, or
        None if the file doesn't exist. This returns a unicode string.
        """
        with io.open(filename, 'rt', encoding='utf-8') as f:
            return f.read()

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

def unique_list_dicts(dlist, key):
    """Return a list of dictionaries which are sorted for only unique entries.

    :param dlist:
    :param key:
    :return list:
    """

    return list(dict((val[key], val) for val in dlist).values())

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

def display_len(text):
    """
    Get the display length of a string. This can differ from the character
    length if the string contains wide characters.
    """
    text = unicodedata.normalize('NFD', text)
    return sum(char_width(char) for char in text)

def inpaint(self):
        """ Replace masked-out elements in an array using an iterative image inpainting algorithm. """

        import inpaint
        filled = inpaint.replace_nans(np.ma.filled(self.raster_data, np.NAN).astype(np.float32), 3, 0.01, 2)
        self.raster_data = np.ma.masked_invalid(filled)

def get_max(qs, field):
    """
    get max for queryset.

    qs: queryset
    field: The field name to max.
    """
    max_field = '%s__max' % field
    num = qs.aggregate(Max(field))[max_field]
    return num if num else 0

def clean_dataframe(df):
    """Fill NaNs with the previous value, the next value or if all are NaN then 1.0"""
    df = df.fillna(method='ffill')
    df = df.fillna(0.0)
    return df

def count_rows(self, table_name):
        """Return the number of entries in a table by counting them."""
        self.table_must_exist(table_name)
        query = "SELECT COUNT (*) FROM `%s`" % table_name.lower()
        self.own_cursor.execute(query)
        return int(self.own_cursor.fetchone()[0])

def filter_dict(d, keys):
    """
    Creates a new dict from an existing dict that only has the given keys
    """
    return {k: v for k, v in d.items() if k in keys}

def _hide_tick_lines_and_labels(axis):
    """
    Set visible property of ticklines and ticklabels of an axis to False
    """
    for item in axis.get_ticklines() + axis.get_ticklabels():
        item.set_visible(False)

def _drop_str_columns(df):
    """

    Parameters
    ----------
    df : DataFrame

    Returns
    -------

    """
    str_columns = filter(lambda pair: pair[1].char == 'S', df._gather_dtypes().items())
    str_column_names = list(map(lambda pair: pair[0], str_columns))

    return df.drop(str_column_names)

def getpass(self, prompt, default=None):
        """Provide a password prompt."""
        return click.prompt(prompt, hide_input=True, default=default)

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

def vec_angle(a, b):
    """
    Calculate angle between two vectors
    """
    cosang = np.dot(a, b)
    sinang = fast_norm(np.cross(a, b))
    return np.arctan2(sinang, cosang)

def finish():
    """Print warning about interrupt and empty the job queue."""
    out.warn("Interrupted!")
    for t in threads:
        t.stop()
    jobs.clear()
    out.warn("Waiting for download threads to finish.")

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

def search_overlap(self, point_list):
        """
        Returns all intervals that overlap the point_list.
        """
        result = set()
        for j in point_list:
            self.search_point(j, result)
        return result

def guess_encoding(text, default=DEFAULT_ENCODING):
    """Guess string encoding.

    Given a piece of text, apply character encoding detection to
    guess the appropriate encoding of the text.
    """
    result = chardet.detect(text)
    return normalize_result(result, default=default)

def findMax(arr):
    """
    in comparison to argrelmax() more simple and  reliable peak finder
    """
    out = np.zeros(shape=arr.shape, dtype=bool)
    _calcMax(arr, out)
    return out

def get_table_names(connection):
	"""
	Return a list of the table names in the database.
	"""
	cursor = connection.cursor()
	cursor.execute("SELECT name FROM sqlite_master WHERE type == 'table'")
	return [name for (name,) in cursor]

def maxDepth(self, currentDepth=0):
        """Compute the depth of the longest branch of the tree"""
        if not any((self.left, self.right)):
            return currentDepth
        result = 0
        for child in (self.left, self.right):
            if child:
                result = max(result, child.maxDepth(currentDepth + 1))
        return result

def pout(msg, log=None):
    """Print 'msg' to stdout, and option 'log' at info level."""
    _print(msg, sys.stdout, log_func=log.info if log else None)

def argmax(l,f=None):
    """http://stackoverflow.com/questions/5098580/implementing-argmax-in-python"""
    if f:
        l = [f(i) for i in l]
    return max(enumerate(l), key=lambda x:x[1])[0]

def mkdir(dir, enter):
    """Create directory with template for topic of the current environment

    """

    if not os.path.exists(dir):
        os.makedirs(dir)

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

def list_i2str(ilist):
    """
    Convert an integer list into a string list.
    """
    slist = []
    for el in ilist:
        slist.append(str(el))
    return slist

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

def OnMove(self, event):
        """Main window move event"""

        # Store window position in config
        position = self.main_window.GetScreenPositionTuple()

        config["window_position"] = repr(position)

def find(self, name):
        """Return the index of the toc entry with name NAME.

           Return -1 for failure."""
        for i, nm in enumerate(self.data):
            if nm[-1] == name:
                return i
        return -1

def multiply(self, number):
        """Return a Vector as the product of the vector and a real number."""
        return self.from_list([x * number for x in self.to_list()])

def _gcd_array(X):
    """
    Return the largest real value h such that all elements in x are integer
    multiples of h.
    """
    greatest_common_divisor = 0.0
    for x in X:
        greatest_common_divisor = _gcd(greatest_common_divisor, x)

    return greatest_common_divisor

def _normalize(mat: np.ndarray):
    """rescales a numpy array, so that min is 0 and max is 255"""
    return ((mat - mat.min()) * (255 / mat.max())).astype(np.uint8)

def get_image_dimension(self, url):
        """
        Return a tuple that contains (width, height)
        Pass in a url to an image and find out its size without loading the whole file
        If the image wxh could not be found, the tuple will contain `None` values
        """
        w_h = (None, None)
        try:
            if url.startswith('//'):
                url = 'http:' + url
            data = requests.get(url).content
            im = Image.open(BytesIO(data))

            w_h = im.size
        except Exception:
            logger.warning("Error getting image size {}".format(url), exc_info=True)

        return w_h

def read_stdin():
    """ Read text from stdin, and print a helpful message for ttys. """
    if sys.stdin.isatty() and sys.stdout.isatty():
        print('\nReading from stdin until end of file (Ctrl + D)...')

    return sys.stdin.read()

def apply_fit(xy,coeffs):
    """ Apply the coefficients from a linear fit to
        an array of x,y positions.

        The coeffs come from the 'coeffs' member of the
        'fit_arrays()' output.
    """
    x_new = coeffs[0][2] + coeffs[0][0]*xy[:,0] + coeffs[0][1]*xy[:,1]
    y_new = coeffs[1][2] + coeffs[1][0]*xy[:,0] + coeffs[1][1]*xy[:,1]

    return x_new,y_new

def read_key(suppress=False):
    """
    Blocks until a keyboard event happens, then returns that event's name or,
    if missing, its scan code.
    """
    event = read_event(suppress)
    return event.name or event.scan_code

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

def inject_into_urllib3():
    """
    Monkey-patch urllib3 with SecureTransport-backed SSL-support.
    """
    util.ssl_.SSLContext = SecureTransportContext
    util.HAS_SNI = HAS_SNI
    util.ssl_.HAS_SNI = HAS_SNI
    util.IS_SECURETRANSPORT = True
    util.ssl_.IS_SECURETRANSPORT = True

def report_stdout(host, stdout):
    """Take a stdout and print it's lines to output if lines are present.

    :param host: the host where the process is running
    :type host: str
    :param stdout: the std out of that process
    :type stdout: paramiko.channel.Channel
    """
    lines = stdout.readlines()
    if lines:
        print("STDOUT from {host}:".format(host=host))
        for line in lines:
            print(line.rstrip(), file=sys.stdout)

def is_local_url(target):
    """Determine if URL is safe to redirect to."""
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ('http', 'https') and \
        ref_url.netloc == test_url.netloc

def flatten(l, types=(list, float)):
    """
    Flat nested list of lists into a single list.
    """
    l = [item if isinstance(item, types) else [item] for item in l]
    return [item for sublist in l for item in sublist]

def export_all(self):
		query = """
			SELECT quote, library, logid
			from quotes
			left outer join quote_log on quotes.quoteid = quote_log.quoteid
			"""
		fields = 'text', 'library', 'log_id'
		return (dict(zip(fields, res)) for res in self.db.execute(query))

def flatten_list(l):
    """ Nested lists to single-level list, does not split strings"""
    return list(chain.from_iterable(repeat(x,1) if isinstance(x,str) else x for x in l))

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

def flatten(lis):
    """Given a list, possibly nested to any level, return it flattened."""
    new_lis = []
    for item in lis:
        if isinstance(item, collections.Sequence) and not isinstance(item, basestring):
            new_lis.extend(flatten(item))
        else:
            new_lis.append(item)
    return new_lis

def test():
    """Run the unit tests."""
    import unittest
    tests = unittest.TestLoader().discover('tests')
    unittest.TextTestRunner(verbosity=2).run(tests)

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

def generate_seed(seed):
    """Generate seed for random number generator"""
    if seed is None:
        random.seed()
        seed = random.randint(0, sys.maxsize)
    random.seed(a=seed)

    return seed

def __clear_buffers(self):
        """Clears the input and output buffers"""
        try:
            self._port.reset_input_buffer()
            self._port.reset_output_buffer()
        except AttributeError:
            #pySerial 2.7
            self._port.flushInput()
            self._port.flushOutput()

def _change_height(self, ax, new_value):
        """Make bars in horizontal bar chart thinner"""
        for patch in ax.patches:
            current_height = patch.get_height()
            diff = current_height - new_value

            # we change the bar height
            patch.set_height(new_value)

            # we recenter the bar
            patch.set_y(patch.get_y() + diff * .5)

def Flush(self):
    """Flush all items from cache."""
    while self._age:
      node = self._age.PopLeft()
      self.KillObject(node.data)

    self._hash = dict()

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

def flatten_all_but_last(a):
  """Flatten all dimensions of a except the last."""
  ret = tf.reshape(a, [-1, tf.shape(a)[-1]])
  if not tf.executing_eagerly():
    ret.set_shape([None] + a.get_shape().as_list()[-1:])
  return ret

def is_collection(obj):
    """Tests if an object is a collection."""

    col = getattr(obj, '__getitem__', False)
    val = False if (not col) else True

    if isinstance(obj, basestring):
        val = False

    return val

async def terminate(self):
        """Terminate a running script."""
        self.proc.terminate()

        await asyncio.wait_for(self.proc.wait(), self.kill_delay)
        if self.proc.returncode is None:
            self.proc.kill()
        await self.proc.wait()

        await super().terminate()

def is_string(val):
    """Determines whether the passed value is a string, safe for 2/3."""
    try:
        basestring
    except NameError:
        return isinstance(val, str)
    return isinstance(val, basestring)

def go_to_background():
    """ Daemonize the running process. """
    try:
        if os.fork():
            sys.exit()
    except OSError as errmsg:
        LOGGER.error('Fork failed: {0}'.format(errmsg))
        sys.exit('Fork failed')

def _wait_for_response(self):
		"""
		Wait until the user accepted or rejected the request
		"""
		while not self.server.response_code:
			time.sleep(2)
		time.sleep(5)
		self.server.shutdown()

def reportMemory(k, options, field=None, isBytes=False):
    """ Given k kilobytes, report back the correct format as string.
    """
    if options.pretty:
        return prettyMemory(int(k), field=field, isBytes=isBytes)
    else:
        if isBytes:
            k /= 1024.
        if field is not None:
            return "%*dK" % (field - 1, k)  # -1 for the "K"
        else:
            return "%dK" % int(k)

def OnUpdateFigurePanel(self, event):
        """Redraw event handler for the figure panel"""

        if self.updating:
            return

        self.updating = True
        self.figure_panel.update(self.get_figure(self.code))
        self.updating = False

def format_line(data, linestyle):
    """Formats a list of elements using the given line style"""
    return linestyle.begin + linestyle.sep.join(data) + linestyle.end

def format_screen(strng):
    """Format a string for screen printing.

    This removes some latex-type format codes."""
    # Paragraph continue
    par_re = re.compile(r'\\$',re.MULTILINE)
    strng = par_re.sub('',strng)
    return strng

def to_percentage(number, rounding=2):
    """Creates a percentage string representation from the given `number`. The
    number is multiplied by 100 before adding a '%' character.

    Raises `ValueError` if `number` cannot be converted to a number.
    """
    number = float(number) * 100
    number_as_int = int(number)
    rounded = round(number, rounding)

    return '{}%'.format(number_as_int if number_as_int == rounded else rounded)

def web(host, port):
    """Start web application"""
    from .webserver.web import get_app
    get_app().run(host=host, port=port)

def is_cached(file_name):
	"""
	Check if a given file is available in the cache or not
	"""

	gml_file_path = join(join(expanduser('~'), OCTOGRID_DIRECTORY), file_name)

	return isfile(gml_file_path)

def start(self):
        """Create a background thread for httpd and serve 'forever'"""
        self._process = threading.Thread(target=self._background_runner)
        self._process.start()

def irfftn(a, s, axes=None):
    """
    Compute the inverse of the multi-dimensional discrete Fourier transform
    for real input. This function is a wrapper for
    :func:`pyfftw.interfaces.numpy_fft.irfftn`, with an interface similar to
    that of :func:`numpy.fft.irfftn`.

    Parameters
    ----------
    a : array_like
      Input array
    s : sequence of ints
      Shape of the output along each transformed axis (input is cropped or
      zero-padded to match). This parameter is not optional because, unlike
      :func:`ifftn`, the output shape cannot be uniquely determined from
      the input shape.
    axes : sequence of ints, optional (default None)
      Axes over which to compute the inverse DFT.

    Returns
    -------
    af : ndarray
      Inverse DFT of input array
    """

    return pyfftw.interfaces.numpy_fft.irfftn(
        a, s=s, axes=axes, overwrite_input=False,
        planner_effort='FFTW_MEASURE', threads=pyfftw_threads)

def closest(xarr, val):
    """ Return the index of the closest in xarr to value val """
    idx_closest = np.argmin(np.abs(np.array(xarr) - val))
    return idx_closest

def _most_common(iterable):
    """Returns the most common element in `iterable`."""
    data = Counter(iterable)
    return max(data, key=data.__getitem__)

def drop_empty(rows):
    """Transpose the columns into rows, remove all of the rows that are empty after the first cell, then
    transpose back. The result is that columns that have a header but no data in the body are removed, assuming
    the header is the first row. """
    return zip(*[col for col in zip(*rows) if bool(filter(bool, col[1:]))])

def to_snake_case(s):
    """Converts camel-case identifiers to snake-case."""
    return re.sub('([^_A-Z])([A-Z])', lambda m: m.group(1) + '_' + m.group(2).lower(), s)

def dereference_url(url):
    """
    Makes a HEAD request to find the final destination of a URL after
    following any redirects
    """
    res = open_url(url, method='HEAD')
    res.close()
    return res.url

def speedtest(func, *args, **kwargs):
    """ Test the speed of a function. """
    n = 100
    start = time.time()
    for i in range(n): func(*args,**kwargs)
    end = time.time()
    return (end-start)/n

def isin(value, values):
    """ Check that value is in values """
    for i, v in enumerate(value):
        if v not in np.array(values)[:, i]:
            return False
    return True

def ver_to_tuple(value):
    """
    Convert version like string to a tuple of integers.
    """
    return tuple(int(_f) for _f in re.split(r'\D+', value) if _f)

def is_delimiter(line):
    """ True if a line consists only of a single punctuation character."""
    return bool(line) and line[0] in punctuation and line[0]*len(line) == line

def money(min=0, max=10):
    """Return a str of decimal with two digits after a decimal mark."""
    value = random.choice(range(min * 100, max * 100))
    return "%1.2f" % (float(value) / 100)

def clean_float(v):
    """Remove commas from a float"""

    if v is None or not str(v).strip():
        return None

    return float(str(v).replace(',', ''))

def money(min=0, max=10):
    """Return a str of decimal with two digits after a decimal mark."""
    value = random.choice(range(min * 100, max * 100))
    return "%1.2f" % (float(value) / 100)

def remove_examples_all():
    """remove arduino/examples/all directory.

    :rtype: None

    """
    d = examples_all_dir()
    if d.exists():
        log.debug('remove %s', d)
        d.rmtree()
    else:
        log.debug('nothing to remove: %s', d)

def random_letters(n):
    """
    Generate a random string from a-zA-Z
    :param n: length of the string
    :return: the random string
    """
    return ''.join(random.SystemRandom().choice(string.ascii_letters) for _ in range(n))

def _maybe_fill(arr, fill_value=np.nan):
    """
    if we have a compatible fill_value and arr dtype, then fill
    """
    if _isna_compat(arr, fill_value):
        arr.fill(fill_value)
    return arr

def sine_wave(frequency):
  """Emit a sine wave at the given frequency."""
  xs = tf.reshape(tf.range(_samples(), dtype=tf.float32), [1, _samples(), 1])
  ts = xs / FLAGS.sample_rate
  return tf.sin(2 * math.pi * frequency * ts)

def _file_and_exists(val, input_files):
    """Check if an input is a file and exists.

    Checks both locally (staged) and from input files (re-passed but never localized).
    """
    return ((os.path.exists(val) and os.path.isfile(val)) or
            val in input_files)

def read_uint(data, start, length):
    """Extract a uint from a position in a sequence."""
    return int.from_bytes(data[start:start+length], byteorder='big')

def example_write_file_to_disk_if_changed():
    """ Try to remove all comments from a file, and save it if changes were made. """
    my_file = FileAsObj('/tmp/example_file.txt')
    my_file.rm(my_file.egrep('^#'))
    if my_file.changed:
        my_file.save()

def rel_path(filename):
    """
    Function that gets relative path to the filename
    """
    return os.path.join(os.getcwd(), os.path.dirname(__file__), filename)

def is_hex_string(string):
    """Check if the string is only composed of hex characters."""
    pattern = re.compile(r'[A-Fa-f0-9]+')
    if isinstance(string, six.binary_type):
        string = str(string)
    return pattern.match(string) is not None

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

def chkstr(s, v):
    """
    Small routine for checking whether a string is empty
    even a string

    :param s: the string in question
    :param v: variable name
    """
    if type(s) != str:
        raise TypeError("{var} must be str".format(var=v))
    if not s:
        raise ValueError("{var} cannot be empty".format(var=v))

def _get_loggers():
    """Return list of Logger classes."""
    from .. import loader
    modules = loader.get_package_modules('logger')
    return list(loader.get_plugins(modules, [_Logger]))

def crop_box(im, box=False, **kwargs):
    """Uses box coordinates to crop an image without resizing it first."""
    if box:
        im = im.crop(box)
    return im

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

def extent(self):
        """Helper for matplotlib imshow"""
        return (
            self.intervals[1].pix1 - 0.5,
            self.intervals[1].pix2 - 0.5,
            self.intervals[0].pix1 - 0.5,
            self.intervals[0].pix2 - 0.5,
        )

def _get_background_color(self):
        """Returns background color rgb tuple of right line"""

        color = self.cell_attributes[self.key]["bgcolor"]
        return tuple(c / 255.0 for c in color_pack2rgb(color))

def resize(self, size):
        """Return a new Image instance with the given size."""
        return Image(self.pil_image.resize(size, PIL.Image.ANTIALIAS))

def now_time(str=False):
    """Get the current time."""
    if str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return datetime.datetime.now()

def min_max_normalize(img):
    """Centre and normalize a given array.

    Parameters:
    ----------
    img: np.ndarray

    """

    min_img = img.min()
    max_img = img.max()

    return (img - min_img) / (max_img - min_img)

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

def setup_path():
    """Sets up the python include paths to include src"""
    import os.path; import sys

    if sys.argv[0]:
        top_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        sys.path = [os.path.join(top_dir, "src")] + sys.path
        pass
    return

def monthly(date=datetime.date.today()):
    """
    Take a date object and return the first day of the month.
    """
    return datetime.date(date.year, date.month, 1)

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

def findfirst(f, coll):
    """Return first occurrence matching f, otherwise None"""
    result = list(dropwhile(f, coll))
    return result[0] if result else None

def incr(self, key, incr_by=1):
        """Increment the key by the given amount."""
        return self.database.hincrby(self.key, key, incr_by)

def fit_linear(X, y):
    """
    Uses OLS to fit the regression.
    """
    model = linear_model.LinearRegression()
    model.fit(X, y)
    return model

def idx(df, index):
    """Universal indexing for numpy and pandas objects."""
    if isinstance(df, (pd.DataFrame, pd.Series)):
        return df.iloc[index]
    else:
        return df[index, :]

def col_frequencies(col, weights=None, gap_chars='-.'):
    """Frequencies of each residue type (totaling 1.0) in a single column."""
    counts = col_counts(col, weights, gap_chars)
    # Reduce to frequencies
    scale = 1.0 / sum(counts.values())
    return dict((aa, cnt * scale) for aa, cnt in counts.iteritems())

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

def matrix_to_gl(matrix):
    """
    Convert a numpy row- major homogenous transformation matrix
    to a flat column- major GLfloat transformation.

    Parameters
    -------------
    matrix : (4,4) float
      Row- major homogenous transform

    Returns
    -------------
    glmatrix : (16,) gl.GLfloat
      Transform in pyglet format
    """
    matrix = np.asanyarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError('matrix must be (4,4)!')

    # switch to column major and flatten to (16,)
    column = matrix.T.flatten()
    # convert to GLfloat
    glmatrix = (gl.GLfloat * 16)(*column)

    return glmatrix

def _join(verb):
    """
    Join helper
    """
    data = pd.merge(verb.x, verb.y, **verb.kwargs)

    # Preserve x groups
    if isinstance(verb.x, GroupedDataFrame):
        data.plydata_groups = list(verb.x.plydata_groups)
    return data

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

def intty(cls):
        """ Check if we are in a tty. """
        # XXX: temporary hack until we can detect if we are in a pipe or not
        return True

        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            return True

        return False

def wrap_key(self, key):
        """Translate the key into the central cell

           This method is only applicable in case of a periodic system.
        """
        return tuple(np.round(
            self.integer_cell.shortest_vector(key)
        ).astype(int))

async def async_input(prompt):
    """
    Python's ``input()`` is blocking, which means the event loop we set
    above can't be running while we're blocking there. This method will
    let the loop run while we wait for input.
    """
    print(prompt, end='', flush=True)
    return (await loop.run_in_executor(None, sys.stdin.readline)).rstrip()

def index(self, elem):
        """Find the index of elem in the reversed iterator."""
        return _coconut.len(self._iter) - self._iter.index(elem) - 1

def linedelimited (inlist,delimiter):
    """
Returns a string composed of elements in inlist, with each element
separated by 'delimiter.'  Used by function writedelimited.  Use '\t'
for tab-delimiting.

Usage:   linedelimited (inlist,delimiter)
"""
    outstr = ''
    for item in inlist:
        if type(item) != StringType:
            item = str(item)
        outstr = outstr + item + delimiter
    outstr = outstr[0:-1]
    return outstr

def get_inputs_from_cm(index, cm):
    """Return indices of inputs to the node with the given index."""
    return tuple(i for i in range(cm.shape[0]) if cm[i][index])

def main(filename):
    """
    Creates a PDF by embedding the first page from the given image and
    writes some text to it.

    @param[in] filename
        The source filename of the image to embed.
    """

    # Prepare font.
    font_family = 'arial'
    font = Font(font_family, bold=True)
    if not font:
        raise RuntimeError('No font found for %r' % font_family)

    # Initialize PDF document on a stream.
    with Document('output.pdf') as document:

        # Initialize a new page and begin its context.
        with document.Page() as ctx:

            # Open the image to embed.
            with Image(filename) as embed:

                # Set the media box for the page to the same as the
                # image to embed.
                ctx.box = embed.box

                # Embed the image.
                ctx.embed(embed)

            # Write some text.
            ctx.add(Text('Hello World', font, size=14, x=100, y=60))

def _index_ordering(redshift_list):
        """

        :param redshift_list: list of redshifts
        :return: indexes in acending order to be evaluated (from z=0 to z=z_source)
        """
        redshift_list = np.array(redshift_list)
        sort_index = np.argsort(redshift_list)
        return sort_index

def __mul__(self, other):
        """Handle the `*` operator."""
        return self._handle_type(other)(self.value * other.value)

def get_own_ip():
    """Get the host's ip number.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
    except socket.gaierror:
        ip_ = "127.0.0.1"
    else:
        ip_ = sock.getsockname()[0]
    finally:
        sock.close()
    return ip_

def bin_to_int(string):
    """Convert a one element byte string to signed int for python 2 support."""
    if isinstance(string, str):
        return struct.unpack("b", string)[0]
    else:
        return struct.unpack("b", bytes([string]))[0]

def end_index(self):
        """
        Returns the 1-based index of the last object on this page,
        relative to total objects found (hits).
        """
        return ((self.number - 1) * self.paginator.per_page +
            len(self.object_list))

def str2int(num, radix=10, alphabet=BASE85):
    """helper function for quick base conversions from strings to integers"""
    return NumConv(radix, alphabet).str2int(num)

def prevmonday(num):
    """
    Return unix SECOND timestamp of "num" mondays ago
    """
    today = get_today()
    lastmonday = today - timedelta(days=today.weekday(), weeks=num)
    return lastmonday

def intToBin(i):
    """ Integer to two bytes """
    # devide in two parts (bytes)
    i1 = i % 256
    i2 = int(i / 256)
    # make string (little endian)
    return chr(i1) + chr(i2)

def qsize(self):
        """Return the approximate size of the queue (not reliable!)."""
        self.mutex.acquire()
        n = self._qsize()
        self.mutex.release()
        return n

def max(self):
        """
        Returns the maximum value of the domain.

        :rtype: `float` or `np.inf`
        """
        return int(self._max) if not np.isinf(self._max) else self._max

def get_line_ending(line):
    """Return line ending."""
    non_whitespace_index = len(line.rstrip()) - len(line)
    if not non_whitespace_index:
        return ''
    else:
        return line[non_whitespace_index:]

def subkey(dct, keys):
    """Get an entry from a dict of dicts by the list of keys to 'follow'
    """
    key = keys[0]
    if len(keys) == 1:
        return dct[key]
    return subkey(dct[key], keys[1:])

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

def lin_interp(x, rangeX, rangeY):
    """
    Interpolate linearly variable x in rangeX onto rangeY.
    """
    s = (x - rangeX[0]) / mag(rangeX[1] - rangeX[0])
    y = rangeY[0] * (1 - s) + rangeY[1] * s
    return y

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

def get_last_modified_timestamp(self):
        """
        Looks at the files in a git root directory and grabs the last modified timestamp
        """
        cmd = "find . -print0 | xargs -0 stat -f '%T@ %p' | sort -n | tail -1 | cut -f2- -d' '"
        ps = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        output = ps.communicate()[0]
        print output

def intersect_3d(p1, p2):
    """Find the closes point for a given set of lines in 3D.

    Parameters
    ----------
    p1 : (M, N) array_like
        Starting points
    p2 : (M, N) array_like
        End points.

    Returns
    -------
    x : (N,) ndarray
        Least-squares solution - the closest point of the intersections.

    Raises
    ------
    numpy.linalg.LinAlgError
        If computation does not converge.

    """
    v = p2 - p1
    normed_v = unit_vector(v)
    nx = normed_v[:, 0]
    ny = normed_v[:, 1]
    nz = normed_v[:, 2]
    xx = np.sum(nx**2 - 1)
    yy = np.sum(ny**2 - 1)
    zz = np.sum(nz**2 - 1)
    xy = np.sum(nx * ny)
    xz = np.sum(nx * nz)
    yz = np.sum(ny * nz)
    M = np.array([(xx, xy, xz), (xy, yy, yz), (xz, yz, zz)])
    x = np.sum(
        p1[:, 0] * (nx**2 - 1) + p1[:, 1] * (nx * ny) + p1[:, 2] * (nx * nz)
    )
    y = np.sum(
        p1[:, 0] * (nx * ny) + p1[:, 1] * (ny * ny - 1) + p1[:, 2] * (ny * nz)
    )
    z = np.sum(
        p1[:, 0] * (nx * nz) + p1[:, 1] * (ny * nz) + p1[:, 2] * (nz**2 - 1)
    )
    return np.linalg.lstsq(M, np.array((x, y, z)), rcond=None)[0]

def chunk_list(l, n):
    """Return `n` size lists from a given list `l`"""
    return [l[i:i + n] for i in range(0, len(l), n)]

def is_sequence(obj):
    """Check if `obj` is a sequence, but not a string or bytes."""
    return isinstance(obj, Sequence) and not (
        isinstance(obj, str) or BinaryClass.is_valid_type(obj))

def gday_of_year(self):
        """Return the number of days since January 1 of the given year."""
        return (self.date - dt.date(self.date.year, 1, 1)).days

def is_date_type(cls):
    """Return True if the class is a date type."""
    if not isinstance(cls, type):
        return False
    return issubclass(cls, date) and not issubclass(cls, datetime)

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

def runiform(lower, upper, size=None):
    """
    Random uniform variates.
    """
    return np.random.uniform(lower, upper, size)

def _run_cmd_get_output(cmd):
    """Runs a shell command, returns console output.

    Mimics python3's subprocess.getoutput
    """
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    out, err = process.communicate()
    return out or err

def split_every(n, iterable):
    """Returns a generator that spits an iteratable into n-sized chunks. The last chunk may have
    less than n elements.

    See http://stackoverflow.com/a/22919323/503377."""
    items = iter(iterable)
    return itertools.takewhile(bool, (list(itertools.islice(items, n)) for _ in itertools.count()))

def root_parent(self, category=None):
        """ Returns the topmost parent of the current category. """
        return next(filter(lambda c: c.is_root, self.hierarchy()))

def stop_at(iterable, idx):
    """Stops iterating before yielding the specified idx."""
    for i, item in enumerate(iterable):
        if i == idx: return
        yield item

def get(s, delimiter='', format="diacritical"):
    """Return pinyin of string, the string must be unicode
    """
    return delimiter.join(_pinyin_generator(u(s), format=format))

def _dict_values_sorted_by_key(dictionary):
    # This should be a yield from instead.
    """Internal helper to return the values of a dictionary, sorted by key.
    """
    for _, value in sorted(dictionary.iteritems(), key=operator.itemgetter(0)):
        yield value

def previous_quarter(d):
    """
    Retrieve the previous quarter for dt
    """
    from django_toolkit.datetime_util import quarter as datetime_quarter
    return quarter( (datetime_quarter(datetime(d.year, d.month, d.day))[0] + timedelta(days=-1)).date() )

def links(cls, page):
    """return all links on a page, including potentially rel= links."""
    for match in cls.HREF_RE.finditer(page):
      yield cls.href_match_to_url(match)

def get_size(path):
    """ Returns the size in bytes if `path` is a file,
        or the size of all files in `path` if it's a directory.
        Analogous to `du -s`.
    """
    if os.path.isfile(path):
        return os.path.getsize(path)
    return sum(get_size(os.path.join(path, f)) for f in os.listdir(path))

def bytesize(arr):
    """
    Returns the memory byte size of a Numpy array as an integer.
    """
    byte_size = np.prod(arr.shape) * np.dtype(arr.dtype).itemsize
    return byte_size

def _fill(self):
    """Advance the iterator without returning the old head."""
    try:
      self._head = self._iterable.next()
    except StopIteration:
      self._head = None

def memsize(self):
        """ Total array cell + indexes size
        """
        return self.size + 1 + TYPE.size(gl.BOUND_TYPE) * len(self.bounds)

def _fill(self):
    """Advance the iterator without returning the old head."""
    try:
      self._head = self._iterable.next()
    except StopIteration:
      self._head = None

def get_tablenames(cur):
    """ Conveinience: """
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tablename_list_ = cur.fetchall()
    tablename_list = [str(tablename[0]) for tablename in tablename_list_ ]
    return tablename_list

def load(raw_bytes):
        """
        given a bytes object, should return a base python data
        structure that represents the object.
        """
        try:
            if not isinstance(raw_bytes, string_type):
                raw_bytes = raw_bytes.decode()
            return json.loads(raw_bytes)
        except ValueError as e:
            raise SerializationException(str(e))

def _extract_node_text(node):
    """Extract text from a given lxml node."""

    texts = map(
        six.text_type.strip, map(six.text_type, map(unescape, node.xpath(".//text()")))
    )
    return " ".join(text for text in texts if text)

def compose_all(tups):
  """Compose all given tuples together."""
  from . import ast  # I weep for humanity
  return functools.reduce(lambda x, y: x.compose(y), map(ast.make_tuple, tups), ast.make_tuple({}))

def first_sunday(self, year, month):
        """Get the first sunday of a month."""
        date = datetime(year, month, 1, 0)
        days_until_sunday = 6 - date.weekday()

        return date + timedelta(days=days_until_sunday)

def json(body, charset='utf-8', **kwargs):
    """Takes JSON formatted data, converting it into native Python objects"""
    return json_converter.loads(text(body, charset=charset))

def _get_background_color(self):
        """Returns background color rgb tuple of right line"""

        color = self.cell_attributes[self.key]["bgcolor"]
        return tuple(c / 255.0 for c in color_pack2rgb(color))

def parse_json_date(value):
    """
    Parses an ISO8601 formatted datetime from a string value
    """
    if not value:
        return None

    return datetime.datetime.strptime(value, JSON_DATETIME_FORMAT).replace(tzinfo=pytz.UTC)

def _get_compiled_ext():
    """Official way to get the extension of compiled files (.pyc or .pyo)"""
    for ext, mode, typ in imp.get_suffixes():
        if typ == imp.PY_COMPILED:
            return ext

def dump_json(obj):
    """Dump Python object as JSON string."""
    return simplejson.dumps(obj, ignore_nan=True, default=json_util.default)

def get_abi3_suffix():
    """Return the file extension for an abi3-compliant Extension()"""
    for suffix, _, _ in (s for s in imp.get_suffixes() if s[2] == imp.C_EXTENSION):
        if '.abi3' in suffix:  # Unix
            return suffix
        elif suffix == '.pyd':  # Windows
            return suffix

def dump_json(obj):
    """Dump Python object as JSON string."""
    return simplejson.dumps(obj, ignore_nan=True, default=json_util.default)

def getScriptLocation():
	"""Helper function to get the location of a Python file."""
	location = os.path.abspath("./")
	if __file__.rfind("/") != -1:
		location = __file__[:__file__.rfind("/")]
	return location

def json_dumps(self, obj):
        """Serializer for consistency"""
        return json.dumps(obj, sort_keys=True, indent=4, separators=(',', ': '))

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

def to_json(obj):
    """Return a json string representing the python object obj."""
    i = StringIO.StringIO()
    w = Writer(i, encoding='UTF-8')
    w.write_value(obj)
    return i.getvalue()

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

def json_dumps(self, obj):
        """Serializer for consistency"""
        return json.dumps(obj, sort_keys=True, indent=4, separators=(',', ': '))

def typename(obj):
    """Returns the type of obj as a string. More descriptive and specific than
    type(obj), and safe for any object, unlike __class__."""
    if hasattr(obj, '__class__'):
        return getattr(obj, '__class__').__name__
    else:
        return type(obj).__name__

def json_decode(data):
    """
    Decodes the given JSON as primitives
    """
    if isinstance(data, six.binary_type):
        data = data.decode('utf-8')

    return json.loads(data)

def uniq(seq):
    """ Return a copy of seq without duplicates. """
    seen = set()
    return [x for x in seq if str(x) not in seen and not seen.add(str(x))]

def is_collection(obj):
    """Tests if an object is a collection."""

    col = getattr(obj, '__getitem__', False)
    val = False if (not col) else True

    if isinstance(obj, basestring):
        val = False

    return val

def itervalues(d, **kw):
    """Return an iterator over the values of a dictionary."""
    if not PY2:
        return iter(d.values(**kw))
    return d.itervalues(**kw)

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

def getSystemVariable(self, remote, name):
        """Get single system variable from CCU / Homegear"""
        if self._server is not None:
            return self._server.getSystemVariable(remote, name)

def extract_keywords_from_text(self, text):
        """Method to extract keywords from the text provided.

        :param text: Text to extract keywords from, provided as a string.
        """
        sentences = nltk.tokenize.sent_tokenize(text)
        self.extract_keywords_from_sentences(sentences)

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")

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

def __get_xml_text(root):
    """ Return the text for the given root node (xml.dom.minidom). """
    txt = ""
    for e in root.childNodes:
        if (e.nodeType == e.TEXT_NODE):
            txt += e.data
    return txt

def classify_clusters(points, n=10):
    """
    Return an array of K-Means cluster classes for an array of `shapely.geometry.Point` objects.
    """
    arr = [[p.x, p.y] for p in points.values]
    clf = KMeans(n_clusters=n)
    clf.fit(arr)
    classes = clf.predict(arr)
    return classes

def getSystemVariable(self, remote, name):
        """Get single system variable from CCU / Homegear"""
        if self._server is not None:
            return self._server.getSystemVariable(remote, name)

def _sourced_dict(self, source=None, **kwargs):
        """Like ``dict(**kwargs)``, but where the ``source`` key is special.
        """
        if source:
            kwargs['source'] = source
        elif self.source:
            kwargs['source'] = self.source
        return kwargs

def get_element_attribute_or_empty(element, attribute_name):
    """

    Args:
        element (element): The xib's element.
        attribute_name (str): The desired attribute's name.

    Returns:
        The attribute's value, or an empty str if none exists.

    """
    return element.attributes[attribute_name].value if element.hasAttribute(attribute_name) else ""

def make_lambda(call):
    """Wrap an AST Call node to lambda expression node.
    call: ast.Call node
    """
    empty_args = ast.arguments(args=[], vararg=None, kwarg=None, defaults=[])
    return ast.Lambda(args=empty_args, body=call)

def is_changed():
    """ Checks if current project has any noncommited changes. """
    executed, changed_lines = execute_git('status --porcelain', output=False)
    merge_not_finished = mod_path.exists('.git/MERGE_HEAD')
    return changed_lines.strip() or merge_not_finished

def apply_kwargs(func, **kwargs):
    """Call *func* with kwargs, but only those kwargs that it accepts.
    """
    new_kwargs = {}
    params = signature(func).parameters
    for param_name in params.keys():
        if param_name in kwargs:
            new_kwargs[param_name] = kwargs[param_name]
    return func(**new_kwargs)

def color_func(func_name):
    """
    Call color function base on name
    """
    if str(func_name).isdigit():
        return term_color(int(func_name))
    return globals()[func_name]

def last_modified_date(filename):
    """Last modified timestamp as a UTC datetime"""
    mtime = os.path.getmtime(filename)
    dt = datetime.datetime.utcfromtimestamp(mtime)
    return dt.replace(tzinfo=pytz.utc)

def run_tests(self):
		"""
		Invoke pytest, replacing argv. Return result code.
		"""
		with _save_argv(_sys.argv[:1] + self.addopts):
			result_code = __import__('pytest').main()
			if result_code:
				raise SystemExit(result_code)

def copy_user_agent_from_driver(self):
        """ Updates requests' session user-agent with the driver's user agent

        This method will start the browser process if its not already running.
        """
        selenium_user_agent = self.driver.execute_script("return navigator.userAgent;")
        self.headers.update({"user-agent": selenium_user_agent})

def print_matrix(X, decimals=1):
    """Pretty printing for numpy matrix X"""
    for row in np.round(X, decimals=decimals):
        print(row)

def get_table_width(table):
    """
    Gets the width of the table that would be printed.
    :rtype: ``int``
    """
    columns = transpose_table(prepare_rows(table))
    widths = [max(len(cell) for cell in column) for column in columns]
    return len('+' + '|'.join('-' * (w + 2) for w in widths) + '+')

def algo_exp(x, m, t, b):
    """mono-exponential curve."""
    return m*np.exp(-t*x)+b

def open01(x, limit=1.e-6):
    """Constrain numbers to (0,1) interval"""
    try:
        return np.array([min(max(y, limit), 1. - limit) for y in x])
    except TypeError:
        return min(max(x, limit), 1. - limit)

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

def _linear_interpolation(x, X, Y):
    """Given two data points [X,Y], linearly interpolate those at x.
    """
    return (Y[1] * (x - X[0]) + Y[0] * (X[1] - x)) / (X[1] - X[0])

def _text_to_graphiz(self, text):
        """create a graphviz graph from text"""
        dot = Source(text, format='svg')
        return dot.pipe().decode('utf-8')

def disown(cmd):
    """Call a system command in the background,
       disown it and hide it's output."""
    subprocess.Popen(cmd,
                     stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)

def selectgt(table, field, value, complement=False):
    """Select rows where the given field is greater than the given value."""

    value = Comparable(value)
    return selectop(table, field, value, operator.gt, complement=complement)

def index(m, val):
    """
    Return the indices of all the ``val`` in ``m``
    """
    mm = np.array(m)
    idx_tuple = np.where(mm == val)
    idx = idx_tuple[0].tolist()

    return idx

def page_guiref(arg_s=None):
    """Show a basic reference about the GUI Console."""
    from IPython.core import page
    page.page(gui_reference, auto_html=True)

def _unique_rows_numpy(a):
    """return unique rows"""
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def executable_exists(executable):
    """Test if an executable is available on the system."""
    for directory in os.getenv("PATH").split(":"):
        if os.path.exists(os.path.join(directory, executable)):
            return True
    return False

def values(self):
        """return a list of all state values"""
        values = []
        for __, data in self.items():
            values.append(data)
        return values

def btc_make_p2sh_address( script_hex ):
    """
    Make a P2SH address from a hex script
    """
    h = hashing.bin_hash160(binascii.unhexlify(script_hex))
    addr = bin_hash160_to_address(h, version_byte=multisig_version_byte)
    return addr

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

def colorbar(height, length, colormap):
    """Return the channels of a colorbar.
    """
    cbar = np.tile(np.arange(length) * 1.0 / (length - 1), (height, 1))
    cbar = (cbar * (colormap.values.max() - colormap.values.min())
            + colormap.values.min())

    return colormap.colorize(cbar)

def get_dt_list(fn_list):
    """Get list of datetime objects, extracted from a filename
    """
    dt_list = np.array([fn_getdatetime(fn) for fn in fn_list])
    return dt_list

def _init_unique_sets(self):
        """Initialise sets used for uniqueness checking."""

        ks = dict()
        for t in self._unique_checks:
            key = t[0]
            ks[key] = set() # empty set
        return ks

def index(m, val):
    """
    Return the indices of all the ``val`` in ``m``
    """
    mm = np.array(m)
    idx_tuple = np.where(mm == val)
    idx = idx_tuple[0].tolist()

    return idx

def poke_array(self, store, name, elemtype, elements, container, visited, _stack):
        """abstract method"""
        raise NotImplementedError

def get_average_length_of_string(strings):
    """Computes average length of words

    :param strings: list of words
    :return: Average length of word on list
    """
    if not strings:
        return 0

    return sum(len(word) for word in strings) / len(strings)

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

def C_dict2array(C):
    """Convert an OrderedDict containing C values to a 1D array."""
    return np.hstack([np.asarray(C[k]).ravel() for k in C_keys])

def update_screen(self):
        """Refresh the screen. You don't need to override this except to update only small portins of the screen."""
        self.clock.tick(self.FPS)
        pygame.display.update()

def loadmat(filename):
    """This function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = sploadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def add_matplotlib_cmap(cm, name=None):
    """Add a matplotlib colormap."""
    global cmaps
    cmap = matplotlib_to_ginga_cmap(cm, name=name)
    cmaps[cmap.name] = cmap

def loads(s, model=None, parser=None):
    """Deserialize s (a str) to a Python object."""
    with StringIO(s) as f:
        return load(f, model=model, parser=parser)

def _float_almost_equal(float1, float2, places=7):
    """Return True if two numbers are equal up to the
    specified number of "places" after the decimal point.
    """

    if round(abs(float2 - float1), places) == 0:
        return True

    return False

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

def _get_or_create_stack(name):
  """Returns a thread local stack uniquified by the given name."""
  stack = getattr(_LOCAL_STACKS, name, None)
  if stack is None:
    stack = []
    setattr(_LOCAL_STACKS, name, stack)
  return stack

def forget_coords(self):
        """Forget all loaded coordinates."""
        self.w.ntotal.set_text('0')
        self.coords_dict.clear()
        self.redo()

def localize(dt):
    """Localize a datetime object to local time."""
    if dt.tzinfo is UTC:
        return (dt + LOCAL_UTC_OFFSET).replace(tzinfo=None)
    # No TZ info so not going to assume anything, return as-is.
    return dt

def unlock(self):
    """Closes the session to the database."""
    if not hasattr(self, 'session'):
      raise RuntimeError('Error detected! The session that you want to close does not exist any more!')
    logger.debug("Closed database session of '%s'" % self._database)
    self.session.close()
    del self.session

def lognorm(x, mu, sigma=1.0):
    """ Log-normal function from scipy """
    return stats.lognorm(sigma, scale=mu).pdf(x)

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

def log_no_newline(self, msg):
      """ print the message to the predefined log file without newline """
      self.print2file(self.logfile, False, False, msg)

def to_python(self, value):
        """
        Convert a string from a form into an Enum value.
        """
        if value is None:
            return value
        if isinstance(value, self.enum):
            return value
        return self.enum[value]

def pylog(self, *args, **kwargs):
        """Display all available logging information."""
        printerr(self.name, args, kwargs, traceback.format_exc())

def closing_plugin(self, cancelable=False):
        """Perform actions before parent main window is closed"""
        self.dialog_manager.close_all()
        self.shell.exit_interpreter()
        return True

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

def read_credentials(fname):
    """
    read a simple text file from a private location to get
    username and password
    """
    with open(fname, 'r') as f:
        username = f.readline().strip('\n')
        password = f.readline().strip('\n')
    return username, password

def load_config(filename="logging.ini", *args, **kwargs):
    """
    Load logger config from file
    
    Keyword arguments:
    filename -- configuration filename (Default: "logging.ini")
    *args -- options passed to fileConfig
    **kwargs -- options passed to fileConfigg
    
    """
    logging.config.fileConfig(filename, *args, **kwargs)

def get_geoip(ip):
    """Lookup country for IP address."""
    reader = geolite2.reader()
    ip_data = reader.get(ip) or {}
    return ip_data.get('country', {}).get('iso_code')

def setLoggerAll(self, mthd):
        """ Sends all messages to ``logger.[mthd]()`` for handling """
        for key in self._logger_methods:
            self._logger_methods[key] = mthd

def time_range(from_=None, to=None):  # todo datetime conversion
    """
    :param str from_:
    :param str to:

    :return: dict
    """
    args = locals()
    return {
        k.replace('_', ''): v for k, v in args.items()
    }

def _get_loggers():
    """Return list of Logger classes."""
    from .. import loader
    modules = loader.get_package_modules('logger')
    return list(loader.get_plugins(modules, [_Logger]))

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

def timed_rotating_file_handler(name, logname, filename, when='h',
                                interval=1, backupCount=0,
                                encoding=None, delay=False, utc=False):
    """
    A Bark logging handler logging output to a named file.  At
    intervals specified by the 'when', the file will be rotated, under
    control of 'backupCount'.

    Similar to logging.handlers.TimedRotatingFileHandler.
    """

    return wrap_log_handler(logging.handlers.TimedRotatingFileHandler(
        filename, when=when, interval=interval, backupCount=backupCount,
        encoding=encoding, delay=delay, utc=utc))

def path_to_list(pathstr):
    """Conver a path string to a list of path elements."""
    return [elem for elem in pathstr.split(os.path.pathsep) if elem]

def logger(message, level=10):
    """Handle logging."""
    logging.getLogger(__name__).log(level, str(message))

def relative_path(path):
    """
    Return the given path relative to this file.
    """
    return os.path.join(os.path.dirname(__file__), path)

def pylog(self, *args, **kwargs):
        """Display all available logging information."""
        printerr(self.name, args, kwargs, traceback.format_exc())

def fast_exit(code):
    """Exit without garbage collection, this speeds up exit by about 10ms for
    things like bash completion.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)

def write(self, text):
        """Write text. An additional attribute terminator with a value of
           None is added to the logging record to indicate that StreamHandler
           should not add a newline."""
        self.logger.log(self.loglevel, text, extra={'terminator': None})

def web(host, port):
    """Start web application"""
    from .webserver.web import get_app
    get_app().run(host=host, port=port)

def log_no_newline(self, msg):
      """ print the message to the predefined log file without newline """
      self.print2file(self.logfile, False, False, msg)

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

def get_geoip(ip):
    """Lookup country for IP address."""
    reader = geolite2.reader()
    ip_data = reader.get(ip) or {}
    return ip_data.get('country', {}).get('iso_code')

def aug_sysargv(cmdstr):
    """ DEBUG FUNC modify argv to look like you ran a command """
    import shlex
    argv = shlex.split(cmdstr)
    sys.argv.extend(argv)

def compare(a, b):
    """
     Compare items in 2 arrays. Returns sum(abs(a(i)-b(i)))
    """
    s=0
    for i in range(len(a)):
        s=s+abs(a[i]-b[i])
    return s

def torecarray(*args, **kwargs):
    """
    Convenient shorthand for ``toarray(*args, **kwargs).view(np.recarray)``.

    """

    import numpy as np
    return toarray(*args, **kwargs).view(np.recarray)

def camelcase_to_slash(name):
    """ Converts CamelCase to camel/case

    code ripped from http://stackoverflow.com/questions/1175208/does-the-python-standard-library-have-function-to-convert-camelcase-to-camel-cas
    """

    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1/\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1/\2', s1).lower()

def clean_error(err):
    """
    Take stderr bytes returned from MicroPython and attempt to create a
    non-verbose error message.
    """
    if err:
        decoded = err.decode('utf-8')
        try:
            return decoded.split('\r\n')[-2]
        except Exception:
            return decoded
    return 'There was an error.'

def _extract_node_text(node):
    """Extract text from a given lxml node."""

    texts = map(
        six.text_type.strip, map(six.text_type, map(unescape, node.xpath(".//text()")))
    )
    return " ".join(text for text in texts if text)

def main(argv=None):
  """Run a Tensorflow model on the Iris dataset."""
  args = parse_arguments(sys.argv if argv is None else argv)

  tf.logging.set_verbosity(tf.logging.INFO)
  learn_runner.run(
      experiment_fn=get_experiment_fn(args),
      output_dir=args.job_dir)

def _extract_node_text(node):
    """Extract text from a given lxml node."""

    texts = map(
        six.text_type.strip, map(six.text_type, map(unescape, node.xpath(".//text()")))
    )
    return " ".join(text for text in texts if text)

def add_suffix(fullname, suffix):
    """ Add suffix to a full file name"""
    name, ext = os.path.splitext(fullname)
    return name + '_' + suffix + ext

def from_file(filename, mime=False):
    """ Opens file, attempts to identify content based
    off magic number and will return the file extension.
    If mime is True it will return the mime type instead.

    :param filename: path to file
    :param mime: Return mime, not extension
    :return: guessed extension or mime
    """

    head, foot = _file_details(filename)
    return _magic(head, foot, mime, ext_from_filename(filename))

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

def mock_add_spec(self, spec, spec_set=False):
        """Add a spec to a mock. `spec` can either be an object or a
        list of strings. Only attributes on the `spec` can be fetched as
        attributes from the mock.

        If `spec_set` is True then only attributes on the spec can be set."""
        self._mock_add_spec(spec, spec_set)
        self._mock_set_magics()

def markdown_media_css():
    """ Add css requirements to HTML.

    :returns: Editor template context.

    """
    return dict(
        CSS_SET=posixpath.join(
            settings.MARKDOWN_SET_PATH, settings.MARKDOWN_SET_NAME, 'style.css'
        ),
        CSS_SKIN=posixpath.join(
            'django_markdown', 'skins', settings.MARKDOWN_EDITOR_SKIN,
            'style.css'
        )
    )

def less_strict_bool(x):
    """Idempotent and None-safe version of strict_bool."""
    if x is None:
        return False
    elif x is True or x is False:
        return x
    else:
        return strict_bool(x)

def smooth_gaussian(image, sigma=1):
    """Returns Gaussian smoothed image.

    :param image: numpy array or :class:`jicimagelib.image.Image`
    :param sigma: standard deviation
    :returns: :class:`jicimagelib.image.Image`
    """
    return scipy.ndimage.filters.gaussian_filter(image, sigma=sigma, mode="nearest")

def cross_list(*sequences):
  """
  From: http://book.opensourceproject.org.cn/lamp/python/pythoncook2/opensource/0596007973/pythoncook2-chp-19-sect-9.html
  """
  result = [[ ]]
  for seq in sequences:
    result = [sublist+[item] for sublist in result for item in seq]
  return result

def default_static_path():
    """
        Return the path to the javascript bundle
    """
    fdir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(fdir, '../assets/'))

def safe_quotes(text, escape_single_quotes=False):
    """htmlify string"""
    if isinstance(text, str):
        safe_text = text.replace('"', "&quot;")
        if escape_single_quotes:
            safe_text = safe_text.replace("'", "&#92;'")
        return safe_text.replace('True', 'true')
    return text

def clean_dataframe(df):
    """Fill NaNs with the previous value, the next value or if all are NaN then 1.0"""
    df = df.fillna(method='ffill')
    df = df.fillna(0.0)
    return df

def get_filename_safe_string(string):
    """
    Converts a string to a string that is safe for a filename
    Args:
        string (str): A string to make safe for a filename

    Returns:
        str: A string safe for a filename
    """
    invalid_filename_chars = ['\\', '/', ':', '"', '*', '?', '|', '\n',
                              '\r']
    if string is None:
        string = "None"
    for char in invalid_filename_chars:
        string = string.replace(char, "")
    string = string.rstrip(".")

    return string

def _openpyxl_read_xl(xl_path: str):
    """ Use openpyxl to read an Excel file. """
    try:
        wb = load_workbook(filename=xl_path, read_only=True)
    except:
        raise
    else:
        return wb

def date_to_datetime(x):
    """Convert a date into a datetime"""
    if not isinstance(x, datetime) and isinstance(x, date):
        return datetime.combine(x, time())
    return x

def _openpyxl_read_xl(xl_path: str):
    """ Use openpyxl to read an Excel file. """
    try:
        wb = load_workbook(filename=xl_path, read_only=True)
    except:
        raise
    else:
        return wb

async def async_input(prompt):
    """
    Python's ``input()`` is blocking, which means the event loop we set
    above can't be running while we're blocking there. This method will
    let the loop run while we wait for input.
    """
    print(prompt, end='', flush=True)
    return (await loop.run_in_executor(None, sys.stdin.readline)).rstrip()

def aggregate(d, y_size, x_size):
        """Average every 4 elements (2x2) in a 2D array"""
        if d.ndim != 2:
            # we can't guarantee what blocks we are getting and how
            # it should be reshaped to do the averaging.
            raise ValueError("Can't aggregrate (reduce) data arrays with "
                             "more than 2 dimensions.")
        if not (x_size.is_integer() and y_size.is_integer()):
            raise ValueError("Aggregation factors are not integers")
        for agg_size, chunks in zip([y_size, x_size], d.chunks):
            for chunk_size in chunks:
                if chunk_size % agg_size != 0:
                    raise ValueError("Aggregation requires arrays with "
                                     "shapes and chunks divisible by the "
                                     "factor")

        new_chunks = (tuple(int(x / y_size) for x in d.chunks[0]),
                      tuple(int(x / x_size) for x in d.chunks[1]))
        return da.core.map_blocks(_mean, d, y_size, x_size, dtype=d.dtype, chunks=new_chunks)

def make_regex(separator):
    """Utility function to create regexp for matching escaped separators
    in strings.

    """
    return re.compile(r'(?:' + re.escape(separator) + r')?((?:[^' +
                      re.escape(separator) + r'\\]|\\.)+)')

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

def to_dicts(recarray):
    """convert record array to a dictionaries"""
    for rec in recarray:
        yield dict(zip(recarray.dtype.names, rec.tolist()))

def Softsign(a):
    """
    Softsign op.
    """
    return np.divide(a, np.add(np.abs(a), 1)),

def find_whole_word(w):
    """
    Scan through string looking for a location where this word produces a match,
    and return a corresponding MatchObject instance.
    Return None if no position in the string matches the pattern;
    note that this is different from finding a zero-length match at some point in the string.
    """
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

def variance(arr):
  """variance of the values, must have 2 or more entries.

  :param arr: list of numbers
  :type arr: number[] a number array
  :return: variance
  :rtype: float

  """
  avg = average(arr)
  return sum([(float(x)-avg)**2 for x in arr])/float(len(arr)-1)

def plot(self):
        """Plot the empirical histogram versus best-fit distribution's PDF."""
        plt.plot(self.bin_edges, self.hist, self.bin_edges, self.best_pdf)

def calc_volume(self, sample: np.ndarray):
        """Find the RMS of the audio"""
        return sqrt(np.mean(np.square(sample)))

def _histplot_op(ax, data, **kwargs):
    """Add a histogram for the data to the axes."""
    bins = get_bins(data)
    ax.hist(data, bins=bins, align="left", density=True, **kwargs)
    return ax

def variance(arr):
  """variance of the values, must have 2 or more entries.

  :param arr: list of numbers
  :type arr: number[] a number array
  :return: variance
  :rtype: float

  """
  avg = average(arr)
  return sum([(float(x)-avg)**2 for x in arr])/float(len(arr)-1)

def axes_off(ax):
    """Get rid of all axis ticks, lines, etc.
    """
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)

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

def figsize(x=8, y=7., aspect=1.):
    """ manually set the default figure size of plots
    ::Arguments::
        x (float): x-axis size
        y (float): y-axis size
        aspect (float): aspect ratio scalar
    """
    # update rcparams with adjusted figsize params
    mpl.rcParams.update({'figure.figsize': (x*aspect, y)})

def get_last(self, table=None):
        """Just the last entry."""
        if table is None: table = self.main_table
        query = 'SELECT * FROM "%s" ORDER BY ROWID DESC LIMIT 1;' % table
        return self.own_cursor.execute(query).fetchone()

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

def jac(x,a):
    """ Jacobian matrix given Christophe's suggestion of f """
    return (x-a) / np.sqrt(((x-a)**2).sum(1))[:,np.newaxis]

def get_max(qs, field):
    """
    get max for queryset.

    qs: queryset
    field: The field name to max.
    """
    max_field = '%s__max' % field
    num = qs.aggregate(Max(field))[max_field]
    return num if num else 0

def test(*args):
    """
    Run unit tests.
    """
    subprocess.call(["py.test-2.7"] + list(args))
    subprocess.call(["py.test-3.4"] + list(args))

def _heapify_max(x):
    """Transform list into a maxheap, in-place, in O(len(x)) time."""
    n = len(x)
    for i in reversed(range(n//2)):
        _siftup_max(x, i)

def commajoin_as_strings(iterable):
    """ Join the given iterable with ',' """
    return _(u',').join((six.text_type(i) for i in iterable))

def SegmentMax(a, ids):
    """
    Segmented max op.
    """
    func = lambda idxs: np.amax(a[idxs], axis=0)
    return seg_map(func, a, ids),

def as_tuple(self, value):
        """Utility function which converts lists to tuples."""
        if isinstance(value, list):
            value = tuple(value)
        return value

def min_values(args):
    """ Return possible range for min function. """
    return Interval(min(x.low for x in args), min(x.high for x in args))

def md5_hash_file(fh):
    """Return the md5 hash of the given file-object"""
    md5 = hashlib.md5()
    while True:
        data = fh.read(8192)
        if not data:
            break
        md5.update(data)
    return md5.hexdigest()

def change_cell(self, x, y, ch, fg, bg):
        """Change cell in position (x;y).
        """
        self.console.draw_char(x, y, ch, fg, bg)

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

def auto():
	"""set colouring on if STDOUT is a terminal device, off otherwise"""
	try:
		Style.enabled = False
		Style.enabled = sys.stdout.isatty()
	except (AttributeError, TypeError):
		pass

def _rendered_size(text, point_size, font_file):
    """
    Return a (width, height) pair representing the size of *text* in English
    Metric Units (EMU) when rendered at *point_size* in the font defined in
    *font_file*.
    """
    emu_per_inch = 914400
    px_per_inch = 72.0

    font = _Fonts.font(font_file, point_size)
    px_width, px_height = font.getsize(text)

    emu_width = int(px_width / px_per_inch * emu_per_inch)
    emu_height = int(px_height / px_per_inch * emu_per_inch)

    return emu_width, emu_height

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

def submit_form_id(step, id):
    """
    Submit the form having given id.
    """
    form = world.browser.find_element_by_xpath(str('id("{id}")'.format(id=id)))
    form.submit()

def _tofloat(obj):
    """Convert to float if object is a float string."""
    if "inf" in obj.lower().strip():
        return obj
    try:
        return int(obj)
    except ValueError:
        try:
            return float(obj)
        except ValueError:
            return obj

def get_subject(self, msg):
        """Extracts the subject line from an EmailMessage object."""

        text, encoding = decode_header(msg['subject'])[-1]

        try:
            text = text.decode(encoding)

        # If it's already decoded, ignore error
        except AttributeError:
            pass

        return text

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

def downcaseTokens(s,l,t):
    """Helper parse action to convert tokens to lower case."""
    return [ tt.lower() for tt in map(_ustr,t) ]

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

def makedirs(path, mode=0o777, exist_ok=False):
    """A wrapper of os.makedirs()."""
    os.makedirs(path, mode, exist_ok)

def _check_fpos(self, fp_, fpos, offset, block):
        """Check file position matches blocksize"""
        if (fp_.tell() + offset != fpos):
            warnings.warn("Actual "+block+" header size does not match expected")
        return

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

def email_type(arg):
	"""An argparse type representing an email address."""
	if not is_valid_email_address(arg):
		raise argparse.ArgumentTypeError("{0} is not a valid email address".format(repr(arg)))
	return arg

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

def _match_literal(self, a, b=None):
        """Match two names."""

        return a.lower() == b if not self.case_sensitive else a == b

def update(self, *args, **kwargs):
        """ A handy update() method which returns self :)

        :rtype: DictProxy
        """
        super(DictProxy, self).update(*args, **kwargs)
        return self

def all_strings(arr):
        """
        Ensures that the argument is a list that either is empty or contains only strings
        :param arr: list
        :return:
        """
        if not isinstance([], list):
            raise TypeError("non-list value found where list is expected")
        return all(isinstance(x, str) for x in arr)

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

def is_integer(obj):
    """Is this an integer.

    :param object obj:
    :return:
    """
    if PYTHON3:
        return isinstance(obj, int)
    return isinstance(obj, (int, long))

def move_up(lines=1, file=sys.stdout):
    """ Move the cursor up a number of lines.

        Esc[ValueA:
        Moves the cursor up by the specified number of lines without changing
        columns. If the cursor is already on the top line, ANSI.SYS ignores
        this sequence.
    """
    move.up(lines).write(file=file)

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

def list_move_to_front(l,value='other'):
    """if the value is in the list, move it to the front and return it."""
    l=list(l)
    if value in l:
        l.remove(value)
        l.insert(0,value)
    return l

def is_square_matrix(mat):
    """Test if an array is a square matrix."""
    mat = np.array(mat)
    if mat.ndim != 2:
        return False
    shape = mat.shape
    return shape[0] == shape[1]

def align_file_position(f, size):
    """ Align the position in the file to the next block of specified size """
    align = (size - 1) - (f.tell() % size)
    f.seek(align, 1)

def filter_regex(names, regex):
    """
    Return a tuple of strings that match the regular expression pattern.
    """
    return tuple(name for name in names
                 if regex.search(name) is not None)

def gmove(pattern, destination):
    """Move all file found by glob.glob(pattern) to destination directory.

    Args:
        pattern (str): Glob pattern
        destination (str): Path to the destination directory.

    Returns:
        bool: True if the operation is successful, False otherwise.
    """
    for item in glob.glob(pattern):
        if not move(item, destination):
            return False
    return True

def is_sparse_vector(x):
    """ x is a 2D sparse matrix with it's first shape equal to 1.
    """
    return sp.issparse(x) and len(x.shape) == 2 and x.shape[0] == 1

def movingaverage(arr, window):
    """
    Calculates the moving average ("rolling mean") of an array
    of a certain window size.
    """
    m = np.ones(int(window)) / int(window)
    return scipy.ndimage.convolve1d(arr, m, axis=0, mode='reflect')

def _isstring(dtype):
    """Given a numpy dtype, determines whether it is a string. Returns True
    if the dtype is string or unicode.
    """
    return dtype.type == numpy.unicode_ or dtype.type == numpy.string_

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

def me(self):
        """Similar to :attr:`.Guild.me` except it may return the :class:`.ClientUser` in private message contexts."""
        return self.guild.me if self.guild is not None else self.bot.user

def split_multiline(value):
    """Split a multiline string into a list, excluding blank lines."""
    return [element for element in (line.strip() for line in value.split('\n'))
            if element]

def contained_in(filename, directory):
    """Test if a file is located within the given directory."""
    filename = os.path.normcase(os.path.abspath(filename))
    directory = os.path.normcase(os.path.abspath(directory))
    return os.path.commonprefix([filename, directory]) == directory

def graphql_queries_to_json(*queries):
    """
    Queries should be a list of GraphQL objects
    """
    rtn = {}
    for i, query in enumerate(queries):
        rtn["q{}".format(i)] = query.value
    return json.dumps(rtn)

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

def many_until1(these, term):
    """Like many_until but must consume at least one of these.
    """
    first = [these()]
    these_results, term_result = many_until(these, term)
    return (first + these_results, term_result)

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

def ncores_reserved(self):
        """
        Returns the number of cores reserved in this moment.
        A core is reserved if it's still not running but
        we have submitted the task to the queue manager.
        """
        return sum(task.manager.num_cores for task in self if task.status == task.S_SUB)

def is_list_of_list(item):
    """
    check whether the item is list (tuple)
    and consist of list (tuple) elements
    """
    if (
        type(item) in (list, tuple)
        and len(item)
        and isinstance(item[0], (list, tuple))
    ):
        return True
    return False

def ncores_reserved(self):
        """
        Returns the number of cores reserved in this moment.
        A core is reserved if it's still not running but
        we have submitted the task to the queue manager.
        """
        return sum(task.manager.num_cores for task in self if task.status == task.S_SUB)

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

def stop(self):
        """stop server"""
        try:
            self.shutdown()
        except (PyMongoError, ServersError) as exc:
            logger.info("Killing %s with signal, shutdown command failed: %r",
                        self.name, exc)
            return process.kill_mprocess(self.proc)

def build_list_type_validator(item_validator):
    """Return a function which validates that the value is a list of items
    which are validated using item_validator.
    """
    def validate_list_of_type(value):
        return [item_validator(item) for item in validate_list(value)]
    return validate_list_of_type

def compute(args):
    x, y, params = args
    """Callable function for the multiprocessing pool."""
    return x, y, mandelbrot(x, y, params)

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

def parallel(processes, threads):
    """
    execute jobs in processes using N threads
    """
    pool = multithread(threads)
    pool.map(run_process, processes)
    pool.close()
    pool.join()

def isString(s):
    """Convenience method that works with all 2.x versions of Python
    to determine whether or not something is stringlike."""
    try:
        return isinstance(s, unicode) or isinstance(s, basestring)
    except NameError:
        return isinstance(s, str)

def parallel(processes, threads):
    """
    execute jobs in processes using N threads
    """
    pool = multithread(threads)
    pool.map(run_process, processes)
    pool.close()
    pool.join()

def _isstring(dtype):
    """Given a numpy dtype, determines whether it is a string. Returns True
    if the dtype is string or unicode.
    """
    return dtype.type == numpy.unicode_ or dtype.type == numpy.string_

def connect_mysql(host, port, user, password, database):
    """Connect to MySQL with retries."""
    return pymysql.connect(
        host=host, port=port,
        user=user, passwd=password,
        db=database
    )

def isstring(value):
    """Report whether the given value is a byte or unicode string."""
    classes = (str, bytes) if pyutils.PY3 else basestring  # noqa: F821
    return isinstance(value, classes)

async def scalar(self, query, as_tuple=False):
        """Get single value from ``select()`` query, i.e. for aggregation.

        :return: result is the same as after sync ``query.scalar()`` call
        """
        query = self._swap_database(query)
        return (await scalar(query, as_tuple=as_tuple))

def init_db():
    """
    Drops and re-creates the SQL schema
    """
    db.drop_all()
    db.configure_mappers()
    db.create_all()
    db.session.commit()

def executemany(self, sql, *params):
        """Prepare a database query or command and then execute it against
        all parameter sequences  found in the sequence seq_of_params.

        :param sql: the SQL statement to execute with optional ? parameters
        :param params: sequence parameters for the markers in the SQL.
        """
        fut = self._run_operation(self._impl.executemany, sql, *params)
        return fut

def _internet_on(address):
    """
    Check to see if the internet is on by pinging a set address.
    :param address: the IP or address to hit
    :return: a boolean - true if can be reached, false if not.
    """
    try:
        urllib2.urlopen(address, timeout=1)
        return True
    except urllib2.URLError as err:
        return False

def dictify(a_named_tuple):
    """Transform a named tuple into a dictionary"""
    return dict((s, getattr(a_named_tuple, s)) for s in a_named_tuple._fields)

def information(filename):
    """Returns the file exif"""
    check_if_this_file_exist(filename)
    filename = os.path.abspath(filename)
    result = get_json(filename)
    result = result[0]
    return result

def _Enum(docstring, *names):
  """Utility to generate enum classes used by annotations.

  Args:
    docstring: Docstring for the generated enum class.
    *names: Enum names.

  Returns:
    A class that contains enum names as attributes.
  """
  enums = dict(zip(names, range(len(names))))
  reverse = dict((value, key) for key, value in enums.iteritems())
  enums['reverse_mapping'] = reverse
  enums['__doc__'] = docstring
  return type('Enum', (object,), enums)

def _rectangular(n):
    """Checks to see if a 2D list is a valid 2D matrix"""
    for i in n:
        if len(i) != len(n[0]):
            return False
    return True

def setup_path():
    """Sets up the python include paths to include src"""
    import os.path; import sys

    if sys.argv[0]:
        top_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        sys.path = [os.path.join(top_dir, "src")] + sys.path
        pass
    return

def is_exe(fpath):
    """
    Path references an executable file.
    """
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

def _pip_exists(self):
        """Returns True if pip exists inside the virtual environment. Can be
        used as a naive way to verify that the environment is installed."""
        return os.path.isfile(os.path.join(self.path, 'bin', 'pip'))

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

def from_json_str(cls, json_str):
    """Convert json string representation into class instance.

    Args:
      json_str: json representation as string.

    Returns:
      New instance of the class with data loaded from json string.
    """
    return cls.from_json(json.loads(json_str, cls=JsonDecoder))

def is_timestamp(obj):
    """
    Yaml either have automatically converted it to a datetime object
    or it is a string that will be validated later.
    """
    return isinstance(obj, datetime.datetime) or is_string(obj) or is_int(obj) or is_float(obj)

def zeros(self, name, **kwargs):
        """Create an array. Keyword arguments as per
        :func:`zarr.creation.zeros`."""
        return self._write_op(self._zeros_nosync, name, **kwargs)

def launched():
    """Test whether the current python environment is the correct lore env.

    :return:  :any:`True` if the environment is launched
    :rtype: bool
    """
    if not PREFIX:
        return False

    return os.path.realpath(sys.prefix) == os.path.realpath(PREFIX)

def ner_chunk(args):
  """Chunk named entities."""
  chunker = NEChunker(lang=args.lang)
  tag(chunker, args)

def cell_ends_with_code(lines):
    """Is the last line of the cell a line with code?"""
    if not lines:
        return False
    if not lines[-1].strip():
        return False
    if lines[-1].startswith('#'):
        return False
    return True

def get_prep_value(self, value):
        """Convert JSON object to a string"""
        if self.null and value is None:
            return None
        return json.dumps(value, **self.dump_kwargs)

def reset(self):
        """Reset analyzer state
        """
        self.prevframe = None
        self.wasmoving = False
        self.t0 = 0
        self.ismoving = False

def _get_non_empty_list(cls, iter):
        """Return a list of the input, excluding all ``None`` values."""
        res = []
        for value in iter:
            if hasattr(value, 'items'):
                value = cls._get_non_empty_dict(value) or None
            if value is not None:
                res.append(value)
        return res

def Flush(self):
    """Flush all items from cache."""
    while self._age:
      node = self._age.PopLeft()
      self.KillObject(node.data)

    self._hash = dict()

def _normalize(mat: np.ndarray):
    """rescales a numpy array, so that min is 0 and max is 255"""
    return ((mat - mat.min()) * (255 / mat.max())).astype(np.uint8)

def __exit__(self, *args):
        """
        Cleanup any necessary opened files
        """

        if self._output_file_handle:
            self._output_file_handle.close()
            self._output_file_handle = None

def normalize(X):
    """ equivalent to scipy.preprocessing.normalize on sparse matrices
    , but lets avoid another depedency just for a small utility function """
    X = coo_matrix(X)
    X.data = X.data / sqrt(bincount(X.row, X.data ** 2))[X.row]
    return X

def get_selected_values(self, selection):
        """Return a list of values for the given selection."""
        return [v for b, v in self._choices if b & selection]

def isTestCaseDisabled(test_case_class, method_name):
    """
    I check to see if a method on a TestCase has been disabled via nose's
    convention for disabling a TestCase.  This makes it so that users can
    mix nose's parameterized tests with green as a runner.
    """
    test_method = getattr(test_case_class, method_name)
    return getattr(test_method, "__test__", 'not nose') is False

def compare(a, b):
    """
     Compare items in 2 arrays. Returns sum(abs(a(i)-b(i)))
    """
    s=0
    for i in range(len(a)):
        s=s+abs(a[i]-b[i])
    return s

def call_and_exit(self, cmd, shell=True):
        """Run the *cmd* and exit with the proper exit code."""
        sys.exit(subprocess.call(cmd, shell=shell))

def equal(obj1, obj2):
    """Calculate equality between two (Comparable) objects."""
    Comparable.log(obj1, obj2, '==')
    equality = obj1.equality(obj2)
    Comparable.log(obj1, obj2, '==', result=equality)
    return equality

def purge_dict(idict):
    """Remove null items from a dictionary """
    odict = {}
    for key, val in idict.items():
        if is_null(val):
            continue
        odict[key] = val
    return odict

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

def other_ind(self):
        """last row or column of square A"""
        return np.full(self.n_min, self.size - 1, dtype=np.int)

def encode(strs):
    """Encodes a list of strings to a single string.
    :type strs: List[str]
    :rtype: str
    """
    res = ''
    for string in strs.split():
        res += str(len(string)) + ":" + string
    return res

def Max(a, axis, keep_dims):
    """
    Max reduction op.
    """
    return np.amax(a, axis=axis if not isinstance(axis, np.ndarray) else tuple(axis),
                   keepdims=keep_dims),

def __connect():
    """
    Connect to a redis instance.
    """
    global redis_instance
    if use_tcp_socket:
        redis_instance = redis.StrictRedis(host=hostname, port=port)
    else:
        redis_instance = redis.StrictRedis(unix_socket_path=unix_socket)

def from_array(cls, arr):
        """Convert a structured NumPy array into a Table."""
        return cls().with_columns([(f, arr[f]) for f in arr.dtype.names])

def from_dict(cls, d):
        """Create an instance from a dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.ENTRIES})

def recarray(self):
        """Returns data as :class:`numpy.recarray`."""
        return numpy.rec.fromrecords(self.records, names=self.names)

def num_leaves(tree):
    """Determine the number of leaves in a tree"""
    if tree.is_leaf:
        return 1
    else:
        return num_leaves(tree.left_child) + num_leaves(tree.right_child)

def _std(self,x):
        """
        Compute standard deviation with ddof degrees of freedom
        """
        return np.nanstd(x.values,ddof=self._ddof)

def objectcount(data, key):
    """return the count of objects of key"""
    objkey = key.upper()
    return len(data.dt[objkey])

def iterexpand(arry, extra):
    """
    Expand dimensions by iteratively append empty axes.

    Parameters
    ----------
    arry : ndarray
        The original array

    extra : int
        The number of empty axes to append
    """
    for d in range(arry.ndim, arry.ndim+extra):
        arry = expand_dims(arry, axis=d)
    return arry

def aws_to_unix_id(aws_key_id):
    """Converts a AWS Key ID into a UID"""
    uid_bytes = hashlib.sha256(aws_key_id.encode()).digest()[-2:]
    if USING_PYTHON2:
        return 2000 + int(from_bytes(uid_bytes) // 2)
    else:
        return 2000 + (int.from_bytes(uid_bytes, byteorder=sys.byteorder) // 2)

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

def _linear_interpolation(x, X, Y):
    """Given two data points [X,Y], linearly interpolate those at x.
    """
    return (Y[1] * (x - X[0]) + Y[0] * (X[1] - x)) / (X[1] - X[0])

def flattened_nested_key_indices(nested_dict):
    """
    Combine the outer and inner keys of nested dictionaries into a single
    ordering.
    """
    outer_keys, inner_keys = collect_nested_keys(nested_dict)
    combined_keys = list(sorted(set(outer_keys + inner_keys)))
    return {k: i for (i, k) in enumerate(combined_keys)}

def movingaverage(arr, window):
    """
    Calculates the moving average ("rolling mean") of an array
    of a certain window size.
    """
    m = np.ones(int(window)) / int(window)
    return scipy.ndimage.convolve1d(arr, m, axis=0, mode='reflect')

def run(self, *args, **kwargs):
        """ Connect and run bot in event loop. """
        self.eventloop.run_until_complete(self.connect(*args, **kwargs))
        try:
            self.eventloop.run_forever()
        finally:
            self.eventloop.stop()

def one_hot(x, size, dtype=np.float32):
  """Make a n+1 dim one-hot array from n dim int-categorical array."""
  return np.array(x[..., np.newaxis] == np.arange(size), dtype)

def ensure_dir_exists(directory):
    """Se asegura de que un directorio exista."""
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def ReadTif(tifFile):
        """Reads a tif file to a 2D NumPy array"""
        img = Image.open(tifFile)
        img = np.array(img)
        return img

def append_scope(self):
        """Create a new scope in the current frame."""
        self.stack.current.append(Scope(self.stack.current.current))

def _isnan(self):
        """
        Return if each value is NaN.
        """
        if self._can_hold_na:
            return isna(self)
        else:
            # shouldn't reach to this condition by checking hasnans beforehand
            values = np.empty(len(self), dtype=np.bool_)
            values.fill(False)
            return values

def write_color(string, name, style='normal', when='auto'):
    """ Write the given colored string to standard out. """
    write(color(string, name, style, when))

def round_array(array_in):
    """
    arr_out = round_array(array_in)

    Rounds an array and recasts it to int. Also works on scalars.
    """
    if isinstance(array_in, ndarray):
        return np.round(array_in).astype(int)
    else:
        return int(np.round(array_in))

def send_dir(self, local_path, remote_path, user='root'):
        """Upload a directory on the remote host.
        """
        self.enable_user(user)
        return self.ssh_pool.send_dir(user, local_path, remote_path)

def round_array(array_in):
    """
    arr_out = round_array(array_in)

    Rounds an array and recasts it to int. Also works on scalars.
    """
    if isinstance(array_in, ndarray):
        return np.round(array_in).astype(int)
    else:
        return int(np.round(array_in))

def symlink(source, destination):
    """Create a symbolic link"""
    log("Symlinking {} as {}".format(source, destination))
    cmd = [
        'ln',
        '-sf',
        source,
        destination,
    ]
    subprocess.check_call(cmd)

def _array2cstr(arr):
    """ Serializes a numpy array to a compressed base64 string """
    out = StringIO()
    np.save(out, arr)
    return b64encode(out.getvalue())

def set_header(self, key, value):
    """ Sets a HTTP header for future requests. """
    self.conn.issue_command("Header", _normalize_header(key), value)

def rank(self):
        """how high in sorted list each key is. inverse permutation of sorter, such that sorted[rank]==keys"""
        r = np.empty(self.size, np.int)
        r[self.sorter] = np.arange(self.size)
        return r

def batchify(data, batch_size):
    """Reshape data into (num_example, batch_size)"""
    nbatch = data.shape[0] // batch_size
    data = data[:nbatch * batch_size]
    data = data.reshape((batch_size, nbatch)).T
    return data

def Max(a, axis, keep_dims):
    """
    Max reduction op.
    """
    return np.amax(a, axis=axis if not isinstance(axis, np.ndarray) else tuple(axis),
                   keepdims=keep_dims),

def _clip(sid, prefix):
    """Clips a prefix from the beginning of a string if it exists."""
    return sid[len(prefix):] if sid.startswith(prefix) else sid

def load_object_by_name(object_name):
    """Load an object from a module by name"""
    mod_name, attr = object_name.rsplit('.', 1)
    mod = import_module(mod_name)
    return getattr(mod, attr)

def safe_setattr(obj, name, value):
    """Attempt to setattr but catch AttributeErrors."""
    try:
        setattr(obj, name, value)
        return True
    except AttributeError:
        return False

def _time_to_json(value):
    """Coerce 'value' to an JSON-compatible representation."""
    if isinstance(value, datetime.time):
        value = value.isoformat()
    return value

def _Enum(docstring, *names):
  """Utility to generate enum classes used by annotations.

  Args:
    docstring: Docstring for the generated enum class.
    *names: Enum names.

  Returns:
    A class that contains enum names as attributes.
  """
  enums = dict(zip(names, range(len(names))))
  reverse = dict((value, key) for key, value in enums.iteritems())
  enums['reverse_mapping'] = reverse
  enums['__doc__'] = docstring
  return type('Enum', (object,), enums)

def object_as_dict(obj):
    """Turn an SQLAlchemy model into a dict of field names and values.

    Based on https://stackoverflow.com/a/37350445/1579058
    """
    return {c.key: getattr(obj, c.key)
            for c in inspect(obj).mapper.column_attrs}

def return_value(self, *args, **kwargs):
        """Extracts the real value to be returned from the wrapping callable.

        :return: The value the double should return when called.
        """

        self._called()
        return self._return_value(*args, **kwargs)

def fit_linear(X, y):
    """
    Uses OLS to fit the regression.
    """
    model = linear_model.LinearRegression()
    model.fit(X, y)
    return model

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

def reduce_fn(x):
    """
    Aggregation function to get the first non-zero value.
    """
    values = x.values if pd and isinstance(x, pd.Series) else x
    for v in values:
        if not is_nan(v):
            return v
    return np.NaN

def del_label(self, name):
        """Delete a label by name."""
        labels_tag = self.root[0]
        labels_tag.remove(self._find_label(name))

def one_hot(x, size, dtype=np.float32):
  """Make a n+1 dim one-hot array from n dim int-categorical array."""
  return np.array(x[..., np.newaxis] == np.arange(size), dtype)

def safe_delete(filename):
  """Delete a file safely. If it's not present, no-op."""
  try:
    os.unlink(filename)
  except OSError as e:
    if e.errno != errno.ENOENT:
      raise

def read(filename):
    """Read and return `filename` in root dir of project and return string"""
    return codecs.open(os.path.join(__DIR__, filename), 'r').read()

def cli(env, identifier):
    """Delete an image."""

    image_mgr = SoftLayer.ImageManager(env.client)
    image_id = helpers.resolve_id(image_mgr.resolve_ids, identifier, 'image')

    image_mgr.delete_image(image_id)

def read_string_from_file(path, encoding="utf8"):
  """
  Read entire contents of file into a string.
  """
  with codecs.open(path, "rb", encoding=encoding) as f:
    value = f.read()
  return value

def remove_examples_all():
    """remove arduino/examples/all directory.

    :rtype: None

    """
    d = examples_all_dir()
    if d.exists():
        log.debug('remove %s', d)
        d.rmtree()
    else:
        log.debug('nothing to remove: %s', d)

def open_with_encoding(filename, encoding, mode='r'):
    """Return opened file with a specific encoding."""
    return io.open(filename, mode=mode, encoding=encoding,
                   newline='')

def make_bintree(levels):
    """Make a symmetrical binary tree with @levels"""
    G = nx.DiGraph()
    root = '0'
    G.add_node(root)
    add_children(G, root, levels, 2)
    return G

def read_string_from_file(path, encoding="utf8"):
  """
  Read entire contents of file into a string.
  """
  with codecs.open(path, "rb", encoding=encoding) as f:
    value = f.read()
  return value

def str2int(string_with_int):
    """ Collect digits from a string """
    return int("".join([char for char in string_with_int if char in string.digits]) or 0)

def imdecode(image_path):
    """Return BGR image read by opencv"""
    import os
    assert os.path.exists(image_path), image_path + ' not found'
    im = cv2.imread(image_path)
    return im

def _match_space_at_line(line):
    """Return a re.match object if an empty comment was found on line."""
    regex = re.compile(r"^{0}$".format(_MDL_COMMENT))
    return regex.match(line)

def from_bytes(cls, b):
		"""Create :class:`PNG` from raw bytes.
		
		:arg bytes b: The raw bytes of the PNG file.
		:rtype: :class:`PNG`
		"""
		im = cls()
		im.chunks = list(parse_chunks(b))
		im.init()
		return im

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

def set_rate(rate):
    """Defines the ideal rate at which computation is to be performed

    :arg rate: the frequency in Hertz 
    :type rate: int or float

    :raises: TypeError: if argument 'rate' is not int or float
    """
    if not (isinstance(rate, int) or isinstance(rate, float)):
        raise TypeError("argument to set_rate is expected to be int or float")
    global loop_duration
    loop_duration = 1.0/rate

def _platform_pylib_exts():  # nocover
    """
    Returns .so, .pyd, or .dylib depending on linux, win or mac.
    On python3 return the previous with and without abi (e.g.
    .cpython-35m-x86_64-linux-gnu) flags. On python2 returns with
    and without multiarch.
    """
    import sysconfig
    valid_exts = []
    if six.PY2:
        # see also 'SHLIB_EXT'
        base_ext = '.' + sysconfig.get_config_var('SO').split('.')[-1]
    else:
        # return with and without API flags
        # handle PEP 3149 -- ABI version tagged .so files
        base_ext = '.' + sysconfig.get_config_var('EXT_SUFFIX').split('.')[-1]
    for tag in _extension_module_tags():
        valid_exts.append('.' + tag + base_ext)
    valid_exts.append(base_ext)
    return tuple(valid_exts)

def apply_argument_parser(argumentsParser, options=None):
    """ Apply the argument parser. """
    if options is not None:
        args = argumentsParser.parse_args(options)
    else:
        args = argumentsParser.parse_args()
    return args

def _visual_width(line):
    """Get the the number of columns required to display a string"""

    return len(re.sub(colorama.ansitowin32.AnsiToWin32.ANSI_CSI_RE, "", line))

def help(self, level=0):
        """return the usage string for available options """
        self.cmdline_parser.formatter.output_level = level
        with _patch_optparse():
            return self.cmdline_parser.format_help()

def screen_cv2(self):
        """cv2 Image of current window screen"""
        pil_image = self.screen.convert('RGB')
        cv2_image = np.array(pil_image)
        pil_image.close()
        # Convert RGB to BGR 
        cv2_image = cv2_image[:, :, ::-1]
        return cv2_image

def check_create_folder(filename):
    """Check if the folder exisits. If not, create the folder"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

def _normalize(mat: np.ndarray):
    """rescales a numpy array, so that min is 0 and max is 255"""
    return ((mat - mat.min()) * (255 / mat.max())).astype(np.uint8)

def zero_pad(m, n=1):
    """Pad a matrix with zeros, on all sides."""
    return np.pad(m, (n, n), mode='constant', constant_values=[0])

def _quit(self, *args):
        """ quit crash """
        self.logger.warn('Bye!')
        sys.exit(self.exit())

def zero_pad(m, n=1):
    """Pad a matrix with zeros, on all sides."""
    return np.pad(m, (n, n), mode='constant', constant_values=[0])

def algo_exp(x, m, t, b):
    """mono-exponential curve."""
    return m*np.exp(-t*x)+b

def column_exists(cr, table, column):
    """ Check whether a certain column exists """
    cr.execute(
        'SELECT count(attname) FROM pg_attribute '
        'WHERE attrelid = '
        '( SELECT oid FROM pg_class WHERE relname = %s ) '
        'AND attname = %s',
        (table, column))
    return cr.fetchone()[0] == 1

def merge(left, right, how='inner', key=None, left_key=None, right_key=None,
          left_as='left', right_as='right'):
    """ Performs a join using the union join function. """
    return join(left, right, how, key, left_key, right_key,
                join_fn=make_union_join(left_as, right_as))

def query_sum(queryset, field):
    """
    Let the DBMS perform a sum on a queryset
    """
    return queryset.aggregate(s=models.functions.Coalesce(models.Sum(field), 0))['s']

def adapter(data, headers, **kwargs):
    """Wrap vertical table in a function for TabularOutputFormatter."""
    keys = ('sep_title', 'sep_character', 'sep_length')
    return vertical_table(data, headers, **filter_dict_by_key(kwargs, keys))

def user_parse(data):
        """Parse information from the provider."""
        _user = data.get('response', {}).get('user', {})
        yield 'id', _user.get('name')
        yield 'username', _user.get('name')
        yield 'link', _user.get('blogs', [{}])[0].get('url')

def clip_image(image, clip_min, clip_max):
  """ Clip an image, or an image batch, with upper and lower threshold. """
  return np.minimum(np.maximum(clip_min, image), clip_max)

def clean_time(time_string):
    """Return a datetime from the Amazon-provided datetime string"""
    # Get a timezone-aware datetime object from the string
    time = dateutil.parser.parse(time_string)
    if not settings.USE_TZ:
        # If timezone support is not active, convert the time to UTC and
        # remove the timezone field
        time = time.astimezone(timezone.utc).replace(tzinfo=None)
    return time

def updateFromKwargs(self, properties, kwargs, collector, **unused):
        """Primary entry point to turn 'kwargs' into 'properties'"""
        properties[self.name] = self.getFromKwargs(kwargs)

def string_to_float_list(string_var):
        """Pull comma separated string values out of a text file and converts them to float list"""
        try:
            return [float(s) for s in string_var.strip('[').strip(']').split(', ')]
        except:
            return [float(s) for s in string_var.strip('[').strip(']').split(',')]

def wget(url):
    """
    Download the page into a string
    """
    import urllib.parse
    request = urllib.request.urlopen(url)
    filestring = request.read()
    return filestring

def parse_json_date(value):
    """
    Parses an ISO8601 formatted datetime from a string value
    """
    if not value:
        return None

    return datetime.datetime.strptime(value, JSON_DATETIME_FORMAT).replace(tzinfo=pytz.UTC)

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

def get_dict_for_attrs(obj, attrs):
    """
    Returns dictionary for each attribute from given ``obj``.
    """
    data = {}
    for attr in attrs:
        data[attr] = getattr(obj, attr)
    return data

def tree(string, token=[WORD, POS, CHUNK, PNP, REL, ANCHOR, LEMMA]):
    """ Transforms the output of parse() into a Text object.
        The token parameter lists the order of tags in each token in the input string.
    """
    return Text(string, token)

def map_wrap(f):
    """Wrap standard function to easily pass into 'map' processing.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper

def parse(source, remove_comments=True, **kw):
    """Thin wrapper around ElementTree.parse"""
    return ElementTree.parse(source, SourceLineParser(), **kw)

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

def updateFromKwargs(self, properties, kwargs, collector, **unused):
        """Primary entry point to turn 'kwargs' into 'properties'"""
        properties[self.name] = self.getFromKwargs(kwargs)

def clean_dict_keys(d):
    """Convert all keys of the dict 'd' to (ascii-)strings.

    :Raises: UnicodeEncodeError
    """
    new_d = {}
    for (k, v) in d.iteritems():
        new_d[str(k)] = v
    return new_d

def def_linear(fun):
    """Flags that a function is linear wrt all args"""
    defjvp_argnum(fun, lambda argnum, g, ans, args, kwargs:
                  fun(*subval(args, argnum, g), **kwargs))

def test3():
    """Test the multiprocess
    """
    import time
    
    p = MVisionProcess()
    p.start()
    time.sleep(5)
    p.stop()

def reprkwargs(kwargs, sep=', ', fmt="{0!s}={1!r}"):
    """Display kwargs."""
    return sep.join(fmt.format(k, v) for k, v in kwargs.iteritems())

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

def is_file(path):
    """Determine if a Path or string is a file on the file system."""
    try:
        return path.expanduser().absolute().is_file()
    except AttributeError:
        return os.path.isfile(os.path.abspath(os.path.expanduser(str(path))))

def execfile(fname, variables):
    """ This is builtin in python2, but we have to roll our own on py3. """
    with open(fname) as f:
        code = compile(f.read(), fname, 'exec')
        exec(code, variables)

def dir_path(dir):
    """with dir_path(path) to change into a directory."""
    old_dir = os.getcwd()
    os.chdir(dir)
    yield
    os.chdir(old_dir)

def exit_and_fail(self, msg=None, out=None):
    """Exits the runtime with a nonzero exit code, indicating failure.

    :param msg: A string message to print to stderr or another custom file desciptor before exiting.
                (Optional)
    :param out: The file descriptor to emit `msg` to. (Optional)
    """
    self.exit(result=PANTS_FAILED_EXIT_CODE, msg=msg, out=out)

def palettebar(height, length, colormap):
    """Return the channels of a palettebar.
    """
    cbar = np.tile(np.arange(length) * 1.0 / (length - 1), (height, 1))
    cbar = (cbar * (colormap.values.max() + 1 - colormap.values.min())
            + colormap.values.min())

    return colormap.palettize(cbar)

def do_exit(self, arg):
        """Exit the shell session."""

        if self.current:
            self.current.close()
        self.resource_manager.close()
        del self.resource_manager
        return True

def user_return(self, frame, return_value):
        """This function is called when a return trap is set here."""
        pdb.Pdb.user_return(self, frame, return_value)

def unzip_file_to_dir(path_to_zip, output_directory):
    """
    Extract a ZIP archive to a directory
    """
    z = ZipFile(path_to_zip, 'r')
    z.extractall(output_directory)
    z.close()

def _single_page_pdf(page):
    """Construct a single page PDF from the provided page in memory"""
    pdf = Pdf.new()
    pdf.pages.append(page)
    bio = BytesIO()
    pdf.save(bio)
    bio.seek(0)
    return bio.read()

def _basic_field_data(field, obj):
    """Returns ``obj.field`` data as a dict"""
    value = field.value_from_object(obj)
    return {Field.TYPE: FieldType.VAL, Field.VALUE: value}

def translate_fourier(image, dx):
    """ Translate an image in fourier-space with plane waves """
    N = image.shape[0]

    f = 2*np.pi*np.fft.fftfreq(N)
    kx,ky,kz = np.meshgrid(*(f,)*3, indexing='ij')
    kv = np.array([kx,ky,kz]).T

    q = np.fft.fftn(image)*np.exp(-1.j*(kv*dx).sum(axis=-1)).T
    return np.real(np.fft.ifftn(q))

def get_remote_content(filepath):
        """ A handy wrapper to get a remote file content """
        with hide('running'):
            temp = BytesIO()
            get(filepath, temp)
            content = temp.getvalue().decode('utf-8')
        return content.strip()

def random_choice(sequence):
    """ Same as :meth:`random.choice`, but also supports :class:`set` type to be passed as sequence. """
    return random.choice(tuple(sequence) if isinstance(sequence, set) else sequence)

def translate_v3(vec, amount):
    """Return a new Vec3 that is translated version of vec."""

    return Vec3(vec.x+amount, vec.y+amount, vec.z+amount)

def _parallel_compare_helper(class_obj, pairs, x, x_link=None):
    """Internal function to overcome pickling problem in python2."""
    return class_obj._compute(pairs, x, x_link)

def imdecode(image_path):
    """Return BGR image read by opencv"""
    import os
    assert os.path.exists(image_path), image_path + ' not found'
    im = cv2.imread(image_path)
    return im

def screen_cv2(self):
        """cv2 Image of current window screen"""
        pil_image = self.screen.convert('RGB')
        cv2_image = np.array(pil_image)
        pil_image.close()
        # Convert RGB to BGR 
        cv2_image = cv2_image[:, :, ::-1]
        return cv2_image

def get_keys_from_class(cc):
    """Return list of the key property names for a class """
    return [prop.name for prop in cc.properties.values() \
            if 'key' in prop.qualifiers]

def resize(self, size):
        """Return a new Image instance with the given size."""
        return Image(self.pil_image.resize(size, PIL.Image.ANTIALIAS))

def datetime_created(self):
        """Returns file group's create aware *datetime* in UTC format."""
        if self.info().get('datetime_created'):
            return dateutil.parser.parse(self.info()['datetime_created'])

def context(self):
        """ Convenient access to shared context """
        if self._context is not None:
            return self._context
        else:
            logger.warning("Using shared context without a lock")
            return self._executor._shared_context

def counter(items):
    """
    Simplest required implementation of collections.Counter. Required as 2.6
    does not have Counter in collections.
    """
    results = {}
    for item in items:
        results[item] = results.get(item, 0) + 1
    return results

def as_list(self):
        """Return all child objects in nested lists of strings."""
        return [self.name, self.value, [x.as_list for x in self.children]]

def filter_bolts(table, header):
  """ filter to keep bolts """
  bolts_info = []
  for row in table:
    if row[0] == 'bolt':
      bolts_info.append(row)
  return bolts_info, header

def head(filename, n=10):
    """ prints the top `n` lines of a file """
    with freader(filename) as fr:
        for _ in range(n):
            print(fr.readline().strip())

def pylint_raw(options):
    """
    Use check_output to run pylint.
    Because pylint changes the exit code based on the code score,
    we have to wrap it in a try/except block.

    :param options:
    :return:
    """
    command = ['pylint']
    command.extend(options)

    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    outs, __ = proc.communicate()

    return outs.decode()

def print_param_values(self_):
        """Print the values of all this object's Parameters."""
        self = self_.self
        for name,val in self.param.get_param_values():
            print('%s.%s = %s' % (self.name,name,val))

def indented_show(text, howmany=1):
        """Print a formatted indented text.
        """
        print(StrTemplate.pad_indent(text=text, howmany=howmany))

def print_env_info(key, out=sys.stderr):
    """If given environment key is defined, print it out."""
    value = os.getenv(key)
    if value is not None:
        print(key, "=", repr(value), file=out)

def random_letters(n):
    """
    Generate a random string from a-zA-Z
    :param n: length of the string
    :return: the random string
    """
    return ''.join(random.SystemRandom().choice(string.ascii_letters) for _ in range(n))

def info(txt):
    """Print, emphasized 'neutral', the given 'txt' message"""

    print("%s# %s%s%s" % (PR_EMPH_CC, get_time_stamp(), txt, PR_NC))
    sys.stdout.flush()

def GeneratePassphrase(length=20):
  """Create a 20 char passphrase with easily typeable chars."""
  valid_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
  valid_chars += "0123456789 ,-_&$#"
  return "".join(random.choice(valid_chars) for i in range(length))

def pprint(obj, verbose=False, max_width=79, newline='\n'):
    """
    Like `pretty` but print to stdout.
    """
    printer = RepresentationPrinter(sys.stdout, verbose, max_width, newline)
    printer.pretty(obj)
    printer.flush()
    sys.stdout.write(newline)
    sys.stdout.flush()

def batch(items, size):
    """Batches a list into a list of lists, with sub-lists sized by a specified
    batch size."""
    return [items[x:x + size] for x in xrange(0, len(items), size)]

def timeit(output):
    """
    If output is string, then print the string and also time used
    """
    b = time.time()
    yield
    print output, 'time used: %.3fs' % (time.time()-b)

def pytest_runtest_logreport(self, report):
        """Store all test reports for evaluation on finish"""
        rep = report
        res = self.config.hook.pytest_report_teststatus(report=rep)
        cat, letter, word = res
        self.stats.setdefault(cat, []).append(rep)

def prettyprint(d):
        """Print dicttree in Json-like format. keys are sorted
        """
        print(json.dumps(d, sort_keys=True, 
                         indent=4, separators=("," , ": ")))

def generate_write_yaml_to_file(file_name):
    """ generate a method to write the configuration in yaml to the method desired """
    def write_yaml(config):
        with open(file_name, 'w+') as fh:
            fh.write(yaml.dump(config))
    return write_yaml

def printdict(adict):
    """printdict"""
    dlist = list(adict.keys())
    dlist.sort()
    for i in range(0, len(dlist)):
        print(dlist[i], adict[dlist[i]])

def get_last_row(dbconn, tablename, n=1, uuid=None):
    """
    Returns the last `n` rows in the table
    """
    return fetch(dbconn, tablename, n, uuid, end=True)

def show_progress(self):
        """If we are in a progress scope, and no log messages have been
        shown, write out another '.'"""
        if self.in_progress_hanging:
            sys.stdout.write('.')
            sys.stdout.flush()

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

def pprint(o, stream=None, indent=1, width=80, depth=None):
    """Pretty-print a Python o to a stream [default is sys.stdout]."""
    printer = PrettyPrinter(
        stream=stream, indent=indent, width=width, depth=depth)
    printer.pprint(o)

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

def get_table_columns(dbconn, tablename):
    """
    Return a list of tuples specifying the column name and type
    """
    cur = dbconn.cursor()
    cur.execute("PRAGMA table_info('%s');" % tablename)
    info = cur.fetchall()
    cols = [(i[1], i[2]) for i in info]
    return cols

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

def _stdout_raw(self, s):
        """Writes the string to stdout"""
        print(s, end='', file=sys.stdout)
        sys.stdout.flush()

def now(self):
		"""
		Return a :py:class:`datetime.datetime` instance representing the current time.

		:rtype: :py:class:`datetime.datetime`
		"""
		if self.use_utc:
			return datetime.datetime.utcnow()
		else:
			return datetime.datetime.now()

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

def _shape(self, df):
        """
        Calculate table chape considering index levels.
        """

        row, col = df.shape
        return row + df.columns.nlevels, col + df.index.nlevels

def get_encoding(binary):
    """Return the encoding type."""

    try:
        from chardet import detect
    except ImportError:
        LOGGER.error("Please install the 'chardet' module")
        sys.exit(1)

    encoding = detect(binary).get('encoding')

    return 'iso-8859-1' if encoding == 'CP949' else encoding

def pprint(obj, verbose=False, max_width=79, newline='\n'):
    """
    Like `pretty` but print to stdout.
    """
    printer = RepresentationPrinter(sys.stdout, verbose, max_width, newline)
    printer.pretty(obj)
    printer.flush()
    sys.stdout.write(newline)
    sys.stdout.flush()

def _get_os_environ_dict(keys):
  """Return a dictionary of key/values from os.environ."""
  return {k: os.environ.get(k, _UNDEFINED) for k in keys}

def to_json(data):
    """Return data as a JSON string."""
    return json.dumps(data, default=lambda x: x.__dict__, sort_keys=True, indent=4)

def getfield(f):
    """convert values from cgi.Field objects to plain values."""
    if isinstance(f, list):
        return [getfield(x) for x in f]
    else:
        return f.value

def time_string(seconds):
    """Returns time in seconds as a string formatted HHHH:MM:SS."""
    s = int(round(seconds))  # round to nearest second
    h, s = divmod(s, 3600)  # get hours and remainder
    m, s = divmod(s, 60)  # split remainder into minutes and seconds
    return "%2i:%02i:%02i" % (h, m, s)

def fields(self):
        """Returns the list of field names of the model."""
        return (self.attributes.values() + self.lists.values()
                + self.references.values())

def _normal_prompt(self):
        """
        Flushes the prompt before requesting the input

        :return: The command line
        """
        sys.stdout.write(self.__get_ps1())
        sys.stdout.flush()
        return safe_input()

def translate_index_to_position(self, index):
        """
        Given an index for the text, return the corresponding (row, col) tuple.
        (0-based. Returns (0, 0) for index=0.)
        """
        # Find start of this line.
        row, row_index = self._find_line_start_index(index)
        col = index - row_index

        return row, col

def _set_property(self, val, *args):
        """Private method that sets the value currently of the property"""
        val = UserClassAdapter._set_property(self, val, *args)
        if val:
            Adapter._set_property(self, val, *args)
        return val

def get_inputs_from_cm(index, cm):
    """Return indices of inputs to the node with the given index."""
    return tuple(i for i in range(cm.shape[0]) if cm[i][index])

def metadata(self):
        """google.protobuf.Message: the current operation metadata."""
        if not self._operation.HasField("metadata"):
            return None

        return protobuf_helpers.from_any_pb(
            self._metadata_type, self._operation.metadata
        )

def format_result(input):
        """From: http://stackoverflow.com/questions/13062300/convert-a-dict-to-sorted-dict-in-python
        """
        items = list(iteritems(input))
        return OrderedDict(sorted(items, key=lambda x: x[0]))

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

def get_all_items(obj):
    """
    dict.items() but with a separate row for each value in a MultiValueDict
    """
    if hasattr(obj, 'getlist'):
        items = []
        for key in obj:
            for value in obj.getlist(key):
                items.append((key, value))
        return items
    else:
        return obj.items()

def download_json(local_filename, url, clobber=False):
    """Download the given JSON file, and pretty-print before we output it."""
    with open(local_filename, 'w') as json_file:
        json_file.write(json.dumps(requests.get(url).json(), sort_keys=True, indent=2, separators=(',', ': ')))

def get_size(objects):
    """Compute the total size of all elements in objects."""
    res = 0
    for o in objects:
        try:
            res += _getsizeof(o)
        except AttributeError:
            print("IGNORING: type=%s; o=%s" % (str(type(o)), str(o)))
    return res

def get_translucent_cmap(r, g, b):

    class TranslucentCmap(BaseColormap):
        glsl_map = """
        vec4 translucent_fire(float t) {{
            return vec4({0}, {1}, {2}, t);
        }}
        """.format(r, g, b)

    return TranslucentCmap()

def get_qualified_name(_object):
    """Return the Fully Qualified Name from an instance or class."""
    module = _object.__module__
    if hasattr(_object, '__name__'):
        _class = _object.__name__

    else:
        _class = _object.__class__.__name__

    return module + '.' + _class

def execute(self, sql, params=None):
        """Just a pointer to engine.execute
        """
        # wrap in a transaction to ensure things are committed
        # https://github.com/smnorris/pgdata/issues/3
        with self.engine.begin() as conn:
            result = conn.execute(sql, params)
        return result

def get_qualified_name(_object):
    """Return the Fully Qualified Name from an instance or class."""
    module = _object.__module__
    if hasattr(_object, '__name__'):
        _class = _object.__name__

    else:
        _class = _object.__class__.__name__

    return module + '.' + _class

def set_ylimits(self, row, column, min=None, max=None):
        """Set y-axis limits of a subplot.

        :param row,column: specify the subplot.
        :param min: minimal axis value
        :param max: maximum axis value

        """
        subplot = self.get_subplot_at(row, column)
        subplot.set_ylimits(min, max)

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

def run_tests(self):
		"""
		Invoke pytest, replacing argv. Return result code.
		"""
		with _save_argv(_sys.argv[:1] + self.addopts):
			result_code = __import__('pytest').main()
			if result_code:
				raise SystemExit(result_code)

def each_img(dir_path):
    """
    Iterates through each image in the given directory. (not recursive)
    :param dir_path: Directory path where images files are present
    :return: Iterator to iterate through image files
    """
    for fname in os.listdir(dir_path):
        if fname.endswith('.jpg') or fname.endswith('.png') or fname.endswith('.bmp'):
            yield fname

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

def clean_axis(axis):
    """Remove ticks, tick labels, and frame from axis"""
    axis.get_xaxis().set_ticks([])
    axis.get_yaxis().set_ticks([])
    for spine in list(axis.spines.values()):
        spine.set_visible(False)

def get(self):
        """Get the highest priority Processing Block from the queue."""
        with self._mutex:
            entry = self._queue.pop()
            del self._block_map[entry[2]]
            return entry[2]

def seq_to_str(obj, sep=","):
    """
    Given a sequence convert it to a comma separated string.
    If, however, the argument is a single object, return its string
    representation.
    """
    if isinstance(obj, string_classes):
        return obj
    elif isinstance(obj, (list, tuple)):
        return sep.join([str(x) for x in obj])
    else:
        return str(obj)

async def delete(self):
        """
        Delete task (in any state) permanently.

        Returns `True` is task is deleted.
        """
        the_tuple = await self.queue.delete(self.tube, self.task_id)

        self.update_from_tuple(the_tuple)

        return bool(self.state == DONE)

def get_line_ending(line):
    """Return line ending."""
    non_whitespace_index = len(line.rstrip()) - len(line)
    if not non_whitespace_index:
        return ''
    else:
        return line[non_whitespace_index:]

def rndstr(size=16):
    """
    Returns a string of random ascii characters or digits

    :param size: The length of the string
    :return: string
    """
    _basech = string.ascii_letters + string.digits
    return "".join([rnd.choice(_basech) for _ in range(size)])

def column(self):
        """
        Returns a zero-based column number of the beginning of this range.
        """
        line, column = self.source_buffer.decompose_position(self.begin_pos)
        return column

def min_values(args):
    """ Return possible range for min function. """
    return Interval(min(x.low for x in args), min(x.high for x in args))

def get_object_attrs(obj):
    """
    Get the attributes of an object using dir.

    This filters protected attributes
    """
    attrs = [k for k in dir(obj) if not k.startswith('__')]
    if not attrs:
        attrs = dir(obj)
    return attrs

def zrank(self, name, value):
        """
        Returns the rank of the element.

        :param name: str     the name of the redis key
        :param value: the element in the sorted set
        """
        with self.pipe as pipe:
            value = self.valueparse.encode(value)
            return pipe.zrank(self.redis_key(name), value)

def data_directory():
    """Return the absolute path to the directory containing the package data."""
    package_directory = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(package_directory, "data")

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

def view_extreme_groups(token, dstore):
    """
    Show the source groups contributing the most to the highest IML
    """
    data = dstore['disagg_by_grp'].value
    data.sort(order='extreme_poe')
    return rst_table(data[::-1])

def get_file_string(filepath):
    """Get string from file."""
    with open(os.path.abspath(filepath)) as f:
        return f.read()

def str2int(string_with_int):
    """ Collect digits from a string """
    return int("".join([char for char in string_with_int if char in string.digits]) or 0)

def read(fname):
    """Quick way to read a file content."""
    content = None
    with open(os.path.join(here, fname)) as f:
        content = f.read()
    return content

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

def read_string_from_file(path, encoding="utf8"):
  """
  Read entire contents of file into a string.
  """
  with codecs.open(path, "rb", encoding=encoding) as f:
    value = f.read()
  return value

def OnRootView(self, event):
        """Reset view to the root of the tree"""
        self.adapter, tree, rows = self.RootNode()
        self.squareMap.SetModel(tree, self.adapter)
        self.RecordHistory()
        self.ConfigureViewTypeChoices()

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

def size(dtype):
  """Returns the number of bytes to represent this `dtype`."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'size'):
    return dtype.size
  return np.dtype(dtype).itemsize

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

def find_lt(a, x):
    """Find rightmost value less than x"""
    i = bisect.bisect_left(a, x)
    if i:
        return a[i-1]
    raise ValueError

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

def yticks(self):
        """Compute the yticks labels of this grid, used for plotting the y-axis ticks when visualizing a regular"""
        return np.linspace(np.min(self[:, 0]), np.max(self[:, 0]), 4)

def get_jsonparsed_data(url):
    """Receive the content of ``url``, parse it as JSON and return the
       object.
    """
    response = urlopen(url)
    data = response.read().decode('utf-8')
    return json.loads(data)

def _model_unique(ins):
    """ Get unique constraints info

    :type ins: sqlalchemy.orm.mapper.Mapper
    :rtype: list[tuple[str]]
    """
    unique = []
    for t in ins.tables:
        for c in t.constraints:
            if isinstance(c, UniqueConstraint):
                unique.append(tuple(col.key for col in c.columns))
    return unique

def read_string(buff, byteorder='big'):
    """Read a string from a file-like object."""
    length = read_numeric(USHORT, buff, byteorder)
    return buff.read(length).decode('utf-8')

def get_url_args(url):
    """ Returns a dictionary from a URL params """
    url_data = urllib.parse.urlparse(url)
    arg_dict = urllib.parse.parse_qs(url_data.query)
    return arg_dict

def ReadTif(tifFile):
        """Reads a tif file to a 2D NumPy array"""
        img = Image.open(tifFile)
        img = np.array(img)
        return img

def return_type(type_name, formatter=None):
    """Specify that this function returns a typed value.

    Args:
        type_name (str): A type name known to the global typedargs type system
        formatter (str): An optional name of a formatting function specified
            for the type given in type_name.
    """

    def _returns(func):
        annotated(func)
        func.metadata.typed_returnvalue(type_name, formatter)
        return func

    return _returns

def readTuple(self, line, n=3):
        """ Reads a tuple of numbers. e.g. vertices, normals or teture coords.
        """
        numbers = [num for num in line.split(' ') if num]
        return [float(num) for num in numbers[1:n + 1]]

def erase_lines(n=1):
    """ Erases n lines from the screen and moves the cursor up to follow
    """
    for _ in range(n):
        print(codes.cursor["up"], end="")
        print(codes.cursor["eol"], end="")

def from_pb(cls, pb):
        """Instantiate the object from a protocol buffer.

        Args:
            pb (protobuf)

        Save a reference to the protocol buffer on the object.
        """
        obj = cls._from_pb(pb)
        obj._pb = pb
        return obj

def extract_words(lines):
    """
    Extract from the given iterable of lines the list of words.

    :param lines: an iterable of lines;
    :return: a generator of words of lines.
    """
    for line in lines:
        for word in re.findall(r"\w+", line):
            yield word

def ReadTif(tifFile):
        """Reads a tif file to a 2D NumPy array"""
        img = Image.open(tifFile)
        img = np.array(img)
        return img

def _check_for_int(x):
    """
    This is a compatibility function that takes a C{float} and converts it to an
    C{int} if the values are equal.
    """
    try:
        y = int(x)
    except (OverflowError, ValueError):
        pass
    else:
        # There is no way in AMF0 to distinguish between integers and floats
        if x == x and y == x:
            return y

    return x

def r_num(obj):
    """Read list of numbers."""
    if isinstance(obj, (list, tuple)):
        it = iter
    else:
        it = LinesIterator
    dataset = Dataset([Dataset.FLOAT])
    return dataset.load(it(obj))

def process_module(self, module):
        """inspect the source file to find encoding problem"""
        if module.file_encoding:
            encoding = module.file_encoding
        else:
            encoding = "ascii"

        with module.stream() as stream:
            for lineno, line in enumerate(stream):
                self._check_encoding(lineno + 1, line, encoding)

def map_keys_deep(f, dct):
    """
    Implementation of map that recurses. This tests the same keys at every level of dict and in lists
    :param f: 2-ary function expecting a key and value and returns a modified key
    :param dct: Dict for deep processing
    :return: Modified dct with matching props mapped
    """
    return _map_deep(lambda k, v: [f(k, v), v], dct)

def is_builtin_type(tp):
    """Checks if the given type is a builtin one.
    """
    return hasattr(__builtins__, tp.__name__) and tp is getattr(__builtins__, tp.__name__)

def redirect_std():
    """
    Connect stdin/stdout to controlling terminal even if the scripts input and output
    were redirected. This is useful in utilities based on termenu.
    """
    stdin = sys.stdin
    stdout = sys.stdout
    if not sys.stdin.isatty():
        sys.stdin = open_raw("/dev/tty", "r", 0)
    if not sys.stdout.isatty():
        sys.stdout = open_raw("/dev/tty", "w", 0)

    return stdin, stdout

def _help():
    """ Display both SQLAlchemy and Python help statements """

    statement = '%s%s' % (shelp, phelp % ', '.join(cntx_.keys()))
    print statement.strip()

def __setitem__(self, field, value):
        """ :see::meth:RedisMap.__setitem__ """
        return self._client.hset(self.key_prefix, field, self._dumps(value))

def send(message, request_context=None, binary=False):
    """Sends a message to websocket.

    :param str message: data to send

    :param request_context:

    :raises IOError: If unable to send a message.
    """
    if binary:
        return uwsgi.websocket_send_binary(message, request_context)

    return uwsgi.websocket_send(message, request_context)

def __connect():
    """
    Connect to a redis instance.
    """
    global redis_instance
    if use_tcp_socket:
        redis_instance = redis.StrictRedis(host=hostname, port=port)
    else:
        redis_instance = redis.StrictRedis(unix_socket_path=unix_socket)

def mark(self, n=1):
        """Mark the occurrence of a given number of events."""
        self.tick_if_necessary()
        self.count += n
        self.m1_rate.update(n)
        self.m5_rate.update(n)
        self.m15_rate.update(n)

def exit(self):
        """
        Closes the connection
        """
        self.pubsub.unsubscribe()
        self.client.connection_pool.disconnect()

        logger.info("Connection to Redis closed")

def _post(self, url, params, uploads=None):
        """ Wrapper method for POST calls. """
        self._call(self.POST, url, params, uploads)

def prep_regex(patterns):
    """Compile regex patterns."""

    flags = 0 if Config.options.case_sensitive else re.I

    return [re.compile(pattern, flags) for pattern in patterns]

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

def match_paren(self, tokens, item):
        """Matches a paren."""
        match, = tokens
        return self.match(match, item)

def remove_legend(ax=None):
    """Remove legend for axes or gca.

    See http://osdir.com/ml/python.matplotlib.general/2005-07/msg00285.html
    """
    from pylab import gca, draw
    if ax is None:
        ax = gca()
    ax.legend_ = None
    draw()

def unmatched(match):
    """Return unmatched part of re.Match object."""
    start, end = match.span(0)
    return match.string[:start]+match.string[end:]

def seq():
    """
    Counts up sequentially from a number based on the current time

    :rtype int:
    """
    current_frame     = inspect.currentframe().f_back
    trace_string      = ""
    while current_frame.f_back:
      trace_string = trace_string + current_frame.f_back.f_code.co_name
      current_frame = current_frame.f_back
    return counter.get_from_trace(trace_string)

def is_valid_email(email):
    """
    Check if email is valid
    """
    pattern = re.compile(r'[\w\.-]+@[\w\.-]+[.]\w+')
    return bool(pattern.match(email))

def _Members(self, group):
    """Unify members of a group and accounts with the group as primary gid."""
    group.members = set(group.members).union(self.gids.get(group.gid, []))
    return group

def version_triple(tag):
    """
    returns: a triple of integers from a version tag
    """
    groups = re.match(r'v?(\d+)\.(\d+)\.(\d+)', tag).groups()
    return tuple(int(n) for n in groups)

def kill(self):
        """Kill the browser.

        This is useful when the browser is stuck.
        """
        if self.process:
            self.process.kill()
            self.process.wait()

def data_directory():
    """Return the absolute path to the directory containing the package data."""
    package_directory = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(package_directory, "data")

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

def f(x, a, c):
    """ Objective function (sum of squared residuals) """
    v = g(x, a, c)
    return v.dot(v)

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

def clean():
    """clean - remove build artifacts."""
    run('rm -rf build/')
    run('rm -rf dist/')
    run('rm -rf puzzle.egg-info')
    run('find . -name __pycache__ -delete')
    run('find . -name *.pyc -delete')
    run('find . -name *.pyo -delete')
    run('find . -name *~ -delete')

    log.info('cleaned up')

def IPYTHON_MAIN():
    """Decide if the Ipython command line is running code."""
    import pkg_resources

    runner_frame = inspect.getouterframes(inspect.currentframe())[-2]
    return (
        getattr(runner_frame, "function", None)
        == pkg_resources.load_entry_point("ipython", "console_scripts", "ipython").__name__
    )

def filter_dict(d, keys):
    """
    Creates a new dict from an existing dict that only has the given keys
    """
    return {k: v for k, v in d.items() if k in keys}

def check_version():
    """Sanity check version information for corrupt virtualenv symlinks
    """
    if sys.version_info[0:3] == PYTHON_VERSION_INFO[0:3]:
        return

    sys.exit(
        ansi.error() + ' your virtual env points to the wrong python version. '
                       'This is likely because you used a python installer that clobbered '
                       'the system installation, which breaks virtualenv creation. '
                       'To fix, check this symlink, and delete the installation of python '
                       'that it is brokenly pointing to, then delete the virtual env itself '
                       'and rerun lore install: ' + os.linesep + os.linesep + BIN_PYTHON +
        os.linesep
    )

def _remove_none_values(dictionary):
    """ Remove dictionary keys whose value is None """
    return list(map(dictionary.pop,
                    [i for i in dictionary if dictionary[i] is None]))

def set_xlimits_widgets(self, set_min=True, set_max=True):
        """Populate axis limits GUI with current plot values."""
        xmin, xmax = self.tab_plot.ax.get_xlim()
        if set_min:
            self.w.x_lo.set_text('{0}'.format(xmin))
        if set_max:
            self.w.x_hi.set_text('{0}'.format(xmax))

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

def join(self, room):
        """Lets a user join a room on a specific Namespace."""
        self.socket.rooms.add(self._get_room_name(room))

def dedup_list(l):
    """Given a list (l) will removing duplicates from the list,
       preserving the original order of the list. Assumes that
       the list entrie are hashable."""
    dedup = set()
    return [ x for x in l if not (x in dedup or dedup.add(x))]

def get_obj_cols(df):
    """
    Returns names of 'object' columns in the DataFrame.
    """
    obj_cols = []
    for idx, dt in enumerate(df.dtypes):
        if dt == 'object' or is_category(dt):
            obj_cols.append(df.columns.values[idx])

    return obj_cols

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

def on_error(e):  # pragma: no cover
    """Error handler

    RuntimeError or ValueError exceptions raised by commands will be handled
    by this function.
    """
    exname = {'RuntimeError': 'Runtime error', 'Value Error': 'Value error'}
    sys.stderr.write('{}: {}\n'.format(exname[e.__class__.__name__], str(e)))
    sys.stderr.write('See file slam_error.log for additional details.\n')
    sys.exit(1)

def _delete_local(self, filename):
        """Deletes the specified file from the local filesystem."""

        if os.path.exists(filename):
            os.remove(filename)

def __copy__(self):
        """A magic method to implement shallow copy behavior."""
        return self.__class__.load(self.dump(), context=self.context)

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

def _curve(x1, y1, x2, y2, hunit = HUNIT, vunit = VUNIT):
    """
    Return a PyX curved path from (x1, y1) to (x2, y2),
    such that the slope at either end is zero.
    """
    ax1, ax2, axm = x1 * hunit, x2 * hunit, (x1 + x2) * hunit / 2
    ay1, ay2 = y1 * vunit, y2 * vunit
    return pyx.path.curve(ax1, ay1, axm, ay1, axm, ay2, ax2, ay2)

def drop_bad_characters(text):
    """Takes a text and drops all non-printable and non-ascii characters and
    also any whitespace characters that aren't space.

    :arg str text: the text to fix

    :returns: text with all bad characters dropped

    """
    # Strip all non-ascii and non-printable characters
    text = ''.join([c for c in text if c in ALLOWED_CHARS])
    return text

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

def filter_none(list_of_points):
    """
    
    :param list_of_points: 
    :return: list_of_points with None's removed
    """
    remove_elementnone = filter(lambda p: p is not None, list_of_points)
    remove_sublistnone = filter(lambda p: not contains_none(p), remove_elementnone)
    return list(remove_sublistnone)

def unique(iterable):
    """ Returns a list copy in which each item occurs only once (in-order).
    """
    seen = set()
    return [x for x in iterable if x not in seen and not seen.add(x)]

def union_overlapping(intervals):
    """Union any overlapping intervals in the given set."""
    disjoint_intervals = []

    for interval in intervals:
        if disjoint_intervals and disjoint_intervals[-1].overlaps(interval):
            disjoint_intervals[-1] = disjoint_intervals[-1].union(interval)
        else:
            disjoint_intervals.append(interval)

    return disjoint_intervals

def _generate_phrases(self, sentences):
        """Method to generate contender phrases given the sentences of the text
        document.

        :param sentences: List of strings where each string represents a
                          sentence which forms the text.
        :return: Set of string tuples where each tuple is a collection
                 of words forming a contender phrase.
        """
        phrase_list = set()
        # Create contender phrases from sentences.
        for sentence in sentences:
            word_list = [word.lower() for word in wordpunct_tokenize(sentence)]
            phrase_list.update(self._get_phrase_list_from_words(word_list))
        return phrase_list

def unapostrophe(text):
    """Strip apostrophe and 's' from the end of a string."""
    text = re.sub(r'[%s]s?$' % ''.join(APOSTROPHES), '', text)
    return text

def indentsize(line):
    """Return the indent size, in spaces, at the start of a line of text."""
    expline = string.expandtabs(line)
    return len(expline) - len(string.lstrip(expline))

def _do_remove_prefix(name):
    """Strip the possible prefix 'Table: ' from a table name."""
    res = name
    if isinstance(res, str):
        if (res.find('Table: ') == 0):
            res = res.replace('Table: ', '', 1)
    return res

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

def strip_accents(text):
    """
    Strip agents from a string.
    """

    normalized_str = unicodedata.normalize('NFD', text)

    return ''.join([
        c for c in normalized_str if unicodedata.category(c) != 'Mn'])

def fixpath(path):
    """Uniformly format a path."""
    return os.path.normpath(os.path.realpath(os.path.expanduser(path)))

def drop_trailing_zeros_decimal(num):
    """ Drops the trailinz zeros from decimal value.
        Returns a string
    """
    out = str(num)
    return out.rstrip('0').rstrip('.') if '.' in out else out

def symlink(source, destination):
    """Create a symbolic link"""
    log("Symlinking {} as {}".format(source, destination))
    cmd = [
        'ln',
        '-sf',
        source,
        destination,
    ]
    subprocess.check_call(cmd)

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

def zeros(self, name, **kwargs):
        """Create an array. Keyword arguments as per
        :func:`zarr.creation.zeros`."""
        return self._write_op(self._zeros_nosync, name, **kwargs)

def conf(self):
        """Generate the Sphinx `conf.py` configuration file

        Returns:
            (str): the contents of the `conf.py` file.
        """
        return self.env.get_template('conf.py.j2').render(
            metadata=self.metadata,
            package=self.package)

def to_capitalized_camel_case(snake_case_string):
    """
    Convert a string from snake case to camel case with the first letter capitalized. For example, "some_var"
    would become "SomeVar".

    :param snake_case_string: Snake-cased string to convert to camel case.
    :returns: Camel-cased version of snake_case_string.
    """
    parts = snake_case_string.split('_')
    return ''.join([i.title() for i in parts])

def render_to_json(templates, context, request):
    """
    Generate a JSON HttpResponse with rendered template HTML.
    """
    html = render_to_string(
        templates,
        context=context,
        request=request
    )
    _json = json.dumps({
        "html": html
    })
    return HttpResponse(_json)

def to_unicode_repr( _letter ):
    """ helpful in situations where browser/app may recognize Unicode encoding
        in the \u0b8e type syntax but not actual unicode glyph/code-point"""
    # Python 2-3 compatible
    return u"u'"+ u"".join( [ u"\\u%04x"%ord(l) for l in _letter ] ) + u"'"

def multi_replace(instr, search_list=[], repl_list=None):
    """
    Does a string replace with a list of search and replacements

    TODO: rename
    """
    repl_list = [''] * len(search_list) if repl_list is None else repl_list
    for ser, repl in zip(search_list, repl_list):
        instr = instr.replace(ser, repl)
    return instr

def join_cols(cols):
    """Join list of columns into a string for a SQL query"""
    return ", ".join([i for i in cols]) if isinstance(cols, (list, tuple, set)) else cols

def to_snake_case(name):
    """ Given a name in camelCase return in snake_case """
    s1 = FIRST_CAP_REGEX.sub(r'\1_\2', name)
    return ALL_CAP_REGEX.sub(r'\1_\2', s1).lower()

def initialize_api(flask_app):
    """Initialize an API."""
    if not flask_restplus:
        return

    api = flask_restplus.Api(version="1.0", title="My Example API")
    api.add_resource(HelloWorld, "/hello")

    blueprint = flask.Blueprint("api", __name__, url_prefix="/api")
    api.init_app(blueprint)
    flask_app.register_blueprint(blueprint)

def multi_replace(instr, search_list=[], repl_list=None):
    """
    Does a string replace with a list of search and replacements

    TODO: rename
    """
    repl_list = [''] * len(search_list) if repl_list is None else repl_list
    for ser, repl in zip(search_list, repl_list):
        instr = instr.replace(ser, repl)
    return instr

def _comment(string):
    """return string as a comment"""
    lines = [line.strip() for line in string.splitlines()]
    return "# " + ("%s# " % linesep).join(lines)

def _replace_token_range(tokens, start, end, replacement):
    """For a range indicated from start to end, replace with replacement."""
    tokens = tokens[:start] + replacement + tokens[end:]
    return tokens

def make_aware(dt):
    """Appends tzinfo and assumes UTC, if datetime object has no tzinfo already."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

def _sub_patterns(patterns, text):
    """
    Apply re.sub to bunch of (pattern, repl)
    """
    for pattern, repl in patterns:
        text = re.sub(pattern, repl, text)
    return text

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

def requests_request(method, url, **kwargs):
    """Requests-mock requests.request wrapper."""
    session = local_sessions.session
    response = session.request(method=method, url=url, **kwargs)
    session.close()
    return response

def unique(self, name):
    """Make a variable name unique by appending a number if needed."""
    # Make sure the name is valid
    name = self.valid(name)
    # Make sure it's not too long
    name = self.trim(name)
    # Now make sure it's unique
    unique_name = name
    i = 2
    while unique_name in self.names:
      unique_name = name + str(i)
      i += 1
    self.names.add(unique_name)
    return unique_name

def auth_request(self, url, headers, body):
        """Perform auth request for token."""

        return self.req.post(url, headers, body=body)

def check_type_and_size_of_param_list(param_list, expected_length):
    """
    Ensure that param_list is a list with the expected length. Raises a helpful
    ValueError if this is not the case.
    """
    try:
        assert isinstance(param_list, list)
        assert len(param_list) == expected_length
    except AssertionError:
        msg = "param_list must be a list containing {} elements."
        raise ValueError(msg.format(expected_length))

    return None

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

def is_float_array(val):
    """
    Checks whether a variable is a numpy float array.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        True if the variable is a numpy float array. Otherwise False.

    """
    return is_np_array(val) and issubclass(val.dtype.type, np.floating)

def dict_to_querystring(dictionary):
    """Converts a dict to a querystring suitable to be appended to a URL."""
    s = u""
    for d in dictionary.keys():
        s = unicode.format(u"{0}{1}={2}&", s, d, dictionary[d])
    return s[:-1]

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

def _log_disconnect(self):
        """ Decrement connection count """
        if self.logged:
            self.server.stats.connectionClosed()
            self.logged = False

def display_len(text):
    """
    Get the display length of a string. This can differ from the character
    length if the string contains wide characters.
    """
    text = unicodedata.normalize('NFD', text)
    return sum(char_width(char) for char in text)

def download_sdk(url):
    """Downloads the SDK and returns a file-like object for the zip content."""
    r = requests.get(url)
    r.raise_for_status()
    return StringIO(r.content)

def merge_dict(data, *args):
    """Merge any number of dictionaries
    """
    results = {}
    for current in (data,) + args:
        results.update(current)
    return results

def normalise_string(string):
    """ Strips trailing whitespace from string, lowercases it and replaces
        spaces with underscores
    """
    string = (string.strip()).lower()
    return re.sub(r'\W+', '_', string)

def gmove(pattern, destination):
    """Move all file found by glob.glob(pattern) to destination directory.

    Args:
        pattern (str): Glob pattern
        destination (str): Path to the destination directory.

    Returns:
        bool: True if the operation is successful, False otherwise.
    """
    for item in glob.glob(pattern):
        if not move(item, destination):
            return False
    return True

def fetch_event(urls):
    """
    This parallel fetcher uses gevent one uses gevent
    """
    rs = (grequests.get(u) for u in urls)
    return [content.json() for content in grequests.map(rs)]

def ensure_hbounds(self):
        """Ensure the cursor is within horizontal screen bounds."""
        self.cursor.x = min(max(0, self.cursor.x), self.columns - 1)

def url_encode(url):
    """
    Convert special characters using %xx escape.

    :param url: str
    :return: str - encoded url
    """
    if isinstance(url, text_type):
        url = url.encode('utf8')
    return quote(url, ':/%?&=')

def disable_insecure_request_warning():
    """Suppress warning about untrusted SSL certificate."""
    import requests
    from requests.packages.urllib3.exceptions import InsecureRequestWarning
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

def _normalize_abmn(abmn):
    """return a normalized version of abmn
    """
    abmn_2d = np.atleast_2d(abmn)
    abmn_normalized = np.hstack((
        np.sort(abmn_2d[:, 0:2], axis=1),
        np.sort(abmn_2d[:, 2:4], axis=1),
    ))
    return abmn_normalized

def resize_image_to_fit_width(image, dest_w):
    """
    Resize and image to fit the passed in width, keeping the aspect ratio the same

    :param image: PIL.Image
    :param dest_w: The desired width
    """
    scale_factor = dest_w / image.size[0]
    dest_h = image.size[1] * scale_factor
    
    scaled_image = image.resize((int(dest_w), int(dest_h)), PIL.Image.ANTIALIAS)

    return scaled_image

def price_rounding(price, decimals=2):
    """Takes a decimal price and rounds to a number of decimal places"""
    try:
        exponent = D('.' + decimals * '0')
    except InvalidOperation:
        # Currencies with no decimal places, ex. JPY, HUF
        exponent = D()
    return price.quantize(exponent, rounding=ROUND_UP)

def resize_image_to_fit_width(image, dest_w):
    """
    Resize and image to fit the passed in width, keeping the aspect ratio the same

    :param image: PIL.Image
    :param dest_w: The desired width
    """
    scale_factor = dest_w / image.size[0]
    dest_h = image.size[1] * scale_factor
    
    scaled_image = image.resize((int(dest_w), int(dest_h)), PIL.Image.ANTIALIAS)

    return scaled_image

def get_feature_order(dataset, features):
    """ Returns a list with the order that features requested appear in
    dataset """
    all_features = dataset.get_feature_names()

    i = [all_features.index(f) for f in features]

    return i

def process_response(self, response):
        """
        Load a JSON response.

        :param Response response: The HTTP response.
        :return dict: The JSON-loaded content.
        """
        if response.status_code != 200:
            raise TwilioException('Unable to fetch page', response)

        return json.loads(response.text)

def _pad(self, text):
        """Pad the text."""
        top_bottom = ("\n" * self._padding) + " "
        right_left = " " * self._padding * self.PAD_WIDTH
        return top_bottom + right_left + text + right_left + top_bottom

def HttpResponse403(request, template=KEY_AUTH_403_TEMPLATE,
content=KEY_AUTH_403_CONTENT, content_type=KEY_AUTH_403_CONTENT_TYPE):
    """
    HTTP response for forbidden access (status code 403)
    """
    return AccessFailedResponse(request, template, content, content_type, status=403)

def run_tests(self):
		"""
		Invoke pytest, replacing argv. Return result code.
		"""
		with _save_argv(_sys.argv[:1] + self.addopts):
			result_code = __import__('pytest').main()
			if result_code:
				raise SystemExit(result_code)

def _fill(self):
    """Advance the iterator without returning the old head."""
    try:
      self._head = self._iterable.next()
    except StopIteration:
      self._head = None

def encode_dataset(dataset, vocabulary):
  """Encode from strings to token ids.

  Args:
    dataset: a tf.data.Dataset with string values.
    vocabulary: a mesh_tensorflow.transformer.Vocabulary
  Returns:
    a tf.data.Dataset with integer-vector values ending in EOS=1
  """
  def encode(features):
    return {k: vocabulary.encode_tf(v) for k, v in features.items()}
  return dataset.map(encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)

def __get__(self, obj, objtype):
        if not self.is_method:
            self.is_method = True
        """Support instance methods."""
        return functools.partial(self.__call__, obj)

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

def get_own_ip():
    """Get the host's ip number.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
    except socket.gaierror:
        ip_ = "127.0.0.1"
    else:
        ip_ = sock.getsockname()[0]
    finally:
        sock.close()
    return ip_

def execute_only_once():
    """
    Each called in the code to this function is guaranteed to return True the
    first time and False afterwards.

    Returns:
        bool: whether this is the first time this function gets called from this line of code.

    Example:
        .. code-block:: python

            if execute_only_once():
                # do something only once
    """
    f = inspect.currentframe().f_back
    ident = (f.f_code.co_filename, f.f_lineno)
    if ident in _EXECUTE_HISTORY:
        return False
    _EXECUTE_HISTORY.add(ident)
    return True

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

def plot(self):
        """Plot the empirical histogram versus best-fit distribution's PDF."""
        plt.plot(self.bin_edges, self.hist, self.bin_edges, self.best_pdf)

def conv_dict(self):
        """dictionary of conversion"""
        return dict(integer=self.integer, real=self.real, no_type=self.no_type)
