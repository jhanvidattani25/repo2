def print_datetime_object(dt):
    """prints a date-object"""
    print(dt)
    print('ctime  :', dt.ctime())
    print('tuple  :', dt.timetuple())
    print('ordinal:', dt.toordinal())
    print('Year   :', dt.year)
    print('Mon    :', dt.month)
    print('Day    :', dt.day)

def _is_video(filepath) -> bool:
    """Check filename extension to see if it's a video file."""
    if os.path.exists(filepath):  # Could be broken symlink
        extension = os.path.splitext(filepath)[1]
        return extension in ('.mkv', '.mp4', '.avi')
    else:
        return False

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

def is_progressive(image):
    """
    Check to see if an image is progressive.
    """
    if not isinstance(image, Image.Image):
        # Can only check PIL images for progressive encoding.
        return False
    return ('progressive' in image.info) or ('progression' in image.info)

def splitext_no_dot(filename):
    """
    Wrap os.path.splitext to return the name and the extension
    without the '.' (e.g., csv instead of .csv)
    """
    name, ext = os.path.splitext(filename)
    ext = ext.lower()
    return name, ext.strip('.')

def inverseHistogram(hist, bin_range):
    """sample data from given histogram and min, max values within range

    Returns:
        np.array: data that would create the same histogram as given
    """
    data = hist.astype(float) / np.min(hist[np.nonzero(hist)])
    new_data = np.empty(shape=np.sum(data, dtype=int))
    i = 0
    xvals = np.linspace(bin_range[0], bin_range[1], len(data))
    for d, x in zip(data, xvals):
        new_data[i:i + d] = x
        i += int(d)
    return new_data

def _parse_ranges(ranges):
    """ Converts a list of string ranges to a list of [low, high] tuples. """
    for txt in ranges:
        if '-' in txt:
            low, high = txt.split('-')
        else:
            low, high = txt, txt
        yield int(low), int(high)

def getRect(self):
		"""
		Returns the window bounds as a tuple of (x,y,w,h)
		"""
		return (self.x, self.y, self.w, self.h)

def retrieve_import_alias_mapping(names_list):
    """Creates a dictionary mapping aliases to their respective name.
    import_alias_names is used in module_definitions.py and visit_Call"""
    import_alias_names = dict()

    for alias in names_list:
        if alias.asname:
            import_alias_names[alias.asname] = alias.name
    return import_alias_names

def get_list_dimensions(_list):
    """
    Takes a nested list and returns the size of each dimension followed
    by the element type in the list
    """
    if isinstance(_list, list) or isinstance(_list, tuple):
        return [len(_list)] + get_list_dimensions(_list[0])
    return []

def close(self):
    """Send a close message to the external process and join it."""
    try:
      self._conn.send((self._CLOSE, None))
      self._conn.close()
    except IOError:
      # The connection was already closed.
      pass
    self._process.join()

def get_ip_address(ifname):
    """ Hack to get IP address from the interface """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    return socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,  # SIOCGIFADDR
        struct.pack('256s', ifname[:15])
    )[20:24])

def get_file_string(filepath):
    """Get string from file."""
    with open(os.path.abspath(filepath)) as f:
        return f.read()

def WriteManyToPath(objs, filepath):
  """Serializes and writes given Python objects to a multi-document YAML file.

  Args:
    objs: An iterable of Python objects to serialize.
    filepath: A path to the file into which the object is to be written.
  """
  with io.open(filepath, mode="w", encoding="utf-8") as filedesc:
    WriteManyToFile(objs, filedesc)

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

def __init__(self, ba=None):
        """Constructor."""
        self.bytearray = ba or (bytearray(b'\0') * self.SIZEOF)

def angle_v2_rad(vec_a, vec_b):
    """Returns angle between vec_a and vec_b in range [0, PI].  This does not
    distinguish if a is left of or right of b.
    """
    # cos(x) = A * B / |A| * |B|
    return math.acos(vec_a.dot(vec_b) / (vec_a.length() * vec_b.length()))

def is_readable(filename):
    """Check if file is a regular file and is readable."""
    return os.path.isfile(filename) and os.access(filename, os.R_OK)

def unpack_out(self, name):
        return self.parse("""
            $enum = $enum_class($value.value)
            """, enum_class=self._import_type(), value=name)["enum"]

def compressBuffer(buffer):
    """
    Note that this code compresses into a buffer held in memory, rather
    than a disk file. This is done through the use of cStringIO.StringIO().
    """
    # http://jython.xhaus.com/http-compression-in-python-and-jython/
    zbuf = cStringIO.StringIO()
    zfile = gzip.GzipFile(mode='wb', fileobj=zbuf, compresslevel=9)
    zfile.write(buffer)
    zfile.close()
    return zbuf.getvalue()

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

def _prepare_proxy(self, conn):
        """
        Establish tunnel connection early, because otherwise httplib
        would improperly set Host: header to proxy's IP:port.
        """
        conn.set_tunnel(self._proxy_host, self.port, self.proxy_headers)
        conn.connect()

def do_quit(self, _: argparse.Namespace) -> bool:
        """Exit this application"""
        self._should_quit = True
        return self._STOP_AND_EXIT

def rewindbody(self):
        """Rewind the file to the start of the body (if seekable)."""
        if not self.seekable:
            raise IOError, "unseekable file"
        self.fp.seek(self.startofbody)

def to_gtp(coord):
    """Converts from a Minigo coordinate to a GTP coordinate."""
    if coord is None:
        return 'pass'
    y, x = coord
    return '{}{}'.format(_GTP_COLUMNS[x], go.N - y)

def now_time(str=False):
    """Get the current time."""
    if str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return datetime.datetime.now()

def str2int(string_with_int):
    """ Collect digits from a string """
    return int("".join([char for char in string_with_int if char in string.digits]) or 0)

def value_to_python(self, value):
        """
        Converts the input single value into the expected Python data type,
        raising django.core.exceptions.ValidationError if the data can't be
        converted.  Returns the converted value. Subclasses should override
        this.
        """
        if not isinstance(value, bytes):
            raise tldap.exceptions.ValidationError("should be a bytes")
        value = value.decode("utf_8")
        return value

def get_capture_dimensions(capture):
    """Get the dimensions of a capture"""
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return width, height

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

def _exit(self, status_code):
        """Properly kill Python process including zombie threads."""
        # If there are active threads still running infinite loops, sys.exit
        # won't kill them but os._exit will. os._exit skips calling cleanup
        # handlers, flushing stdio buffers, etc.
        exit_func = os._exit if threading.active_count() > 1 else sys.exit
        exit_func(status_code)

def created_today(self):
        """Return True if created today."""
        if self.datetime.date() == datetime.today().date():
            return True
        return False

def delete(self, name):
        """Delete object on remote"""
        obj = self._get_object(name)
        if obj:
            return self.driver.delete_object(obj)

def transformer_tall_pretrain_lm_tpu_adafactor():
  """Hparams for transformer on LM pretraining (with 64k vocab) on TPU."""
  hparams = transformer_tall_pretrain_lm()
  update_hparams_for_tpu(hparams)
  hparams.max_length = 1024
  # For multi-problem on TPU we need it in absolute examples.
  hparams.batch_size = 8
  hparams.multiproblem_vocab_size = 2**16
  return hparams

def scatter(self, *args, **kwargs):
        """Add a scatter plot."""
        cls = _make_class(ScatterVisual,
                          _default_marker=kwargs.pop('marker', None),
                          )
        return self._add_item(cls, *args, **kwargs)

def fit_linear(X, y):
    """
    Uses OLS to fit the regression.
    """
    model = linear_model.LinearRegression()
    model.fit(X, y)
    return model

def ucamel_method(func):
    """
    Decorator to ensure the given snake_case method is also written in
    UpperCamelCase in the given namespace. That was mainly written to
    avoid confusion when using wxPython and its UpperCamelCaseMethods.
    """
    frame_locals = inspect.currentframe().f_back.f_locals
    frame_locals[snake2ucamel(func.__name__)] = func
    return func

def tpr(y, z):
    """True positive rate `tp / (tp + fn)`
    """
    tp, tn, fp, fn = contingency_table(y, z)
    return tp / (tp + fn)

def quit(self):
        """ Exit the program due to user's choices.
        """
        self.script.LOG.warn("Abort due to user choice!")
        sys.exit(self.QUIT_RC)

def tearDown(self):
        """ Clean up environment

        """
        if self.sdkobject and self.sdkobject.id:
            self.sdkobject.delete()
            self.sdkobject.id = None

def get_record_by_name(self, index, name):
        """
        Searches for a single document in the given index on the 'name' field .
        Performs a case-insensitive search by utilizing Elasticsearch's `match_phrase` query.
    
        Args:
            index: `str`. The name of an Elasticsearch index (i.e. biosamples).
            name: `str`. The value of a document's name key to search for.
    
        Returns:
            `dict` containing the document that was indexed into Elasticsearch.
    
        Raises:
            `MultipleHitsException`: More than 1 hit is returned.
        """
        result = self.ES.search(
            index=index,
            body={
                "query": {
                    "match_phrase": {
                        "name": name,
                    }
                }
            }
        )
        hits = result["hits"]["hits"]
        if not hits:
            return {}
        elif len(hits) == 1:
            return hits[0]["_source"]
        else:
            # Mult. records found with same prefix. See if a single record whose name attr matches
            # the match phrase exactly (in a lower-case comparison).  
            for h in hits:
                source = h["_source"]
                record_name = source["name"]
                if record_name.lower().strip() == name.lower().strip():
                    return source
            msg = "match_phrase search found multiple records matching query '{}' for index '{}'.".format(name, index)
            raise MultipleHitsException(msg)

def _comment(string):
    """return string as a comment"""
    lines = [line.strip() for line in string.splitlines()]
    return "# " + ("%s# " % linesep).join(lines)

def template_substitute(text, **kwargs):
    """
    Replace placeholders in text by using the data mapping.
    Other placeholders that is not represented by data is left untouched.

    :param text:   Text to search and replace placeholders.
    :param data:   Data mapping/dict for placeholder key and values.
    :return: Potentially modified text with replaced placeholders.
    """
    for name, value in kwargs.items():
        placeholder_pattern = "{%s}" % name
        if placeholder_pattern in text:
            text = text.replace(placeholder_pattern, value)
    return text

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

def get_server(address=None):
        """Return an SMTP servername guess from outgoing email address."""
        if address:
            domain = address.split("@")[1]
            try:
                return SMTP_SERVERS[domain]
            except KeyError:
                return ("smtp." + domain, 465)
        return (None, None)

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

def maxlevel(lst):
    """Return maximum nesting depth"""
    maxlev = 0
    def f(lst, level):
        nonlocal maxlev
        if isinstance(lst, list):
            level += 1
            maxlev = max(level, maxlev)
            for item in lst:
                f(item, level)
    f(lst, 0)
    return maxlev

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

def stdout_display():
    """ Print results straight to stdout """
    if sys.version_info[0] == 2:
        yield SmartBuffer(sys.stdout)
    else:
        yield SmartBuffer(sys.stdout.buffer)

def get_all_names(self):
        """Return the list of all cached global names"""
        result = set()
        for module in self.names:
            result.update(set(self.names[module]))
        return result

def header_status(header):
    """Parse HTTP status line, return status (int) and reason."""
    status_line = header[:header.find('\r')]
    # 'HTTP/1.1 200 OK' -> (200, 'OK')
    fields = status_line.split(None, 2)
    return int(fields[1]), fields[2]

def get_filetype_icon(fname):
    """Return file type icon"""
    ext = osp.splitext(fname)[1]
    if ext.startswith('.'):
        ext = ext[1:]
    return get_icon( "%s.png" % ext, ima.icon('FileIcon') )

def is_image(filename):
    """Determine if given filename is an image."""
    # note: isfile() also accepts symlinks
    return os.path.isfile(filename) and filename.lower().endswith(ImageExts)

def _get_or_default(mylist, i, default=None):
    """return list item number, or default if don't exist"""
    if i >= len(mylist):
        return default
    else :
        return mylist[i]

def state(self):
        """Return internal state, useful for testing."""
        return {'c': self.c, 's0': self.s0, 's1': self.s1, 's2': self.s2}

def gen_lower(x: Iterable[str]) -> Generator[str, None, None]:
    """
    Args:
        x: iterable of strings

    Yields:
        each string in lower case
    """
    for string in x:
        yield string.lower()

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

def to_bytes(s, encoding="utf-8"):
    """Convert a string to bytes."""
    if isinstance(s, six.binary_type):
        return s
    if six.PY3:
        return bytes(s, encoding)
    return s.encode(encoding)

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

def fopenat(base_fd, path):
    """
    Does openat read-only, then does fdopen to get a file object
    """

    return os.fdopen(openat(base_fd, path, os.O_RDONLY), 'rb')

def convolve_gaussian_2d(image, gaussian_kernel_1d):
    """Convolve 2d gaussian."""
    result = scipy.ndimage.filters.correlate1d(
        image, gaussian_kernel_1d, axis=0)
    result = scipy.ndimage.filters.correlate1d(
        result, gaussian_kernel_1d, axis=1)
    return result

def is_string(val):
    """Determines whether the passed value is a string, safe for 2/3."""
    try:
        basestring
    except NameError:
        return isinstance(val, str)
    return isinstance(val, basestring)

def populate_obj(obj, attrs):
    """Populates an object's attributes using the provided dict
    """
    for k, v in attrs.iteritems():
        setattr(obj, k, v)

def filter_regex(names, regex):
    """
    Return a tuple of strings that match the regular expression pattern.
    """
    return tuple(name for name in names
                 if regex.search(name) is not None)

def uniqued(iterable):
    """Return unique list of items preserving order.

    >>> uniqued([3, 2, 1, 3, 2, 1, 0])
    [3, 2, 1, 0]
    """
    seen = set()
    add = seen.add
    return [i for i in iterable if i not in seen and not add(i)]

def read_raw(data_path):
    """
    Parameters
    ----------
    data_path : str
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data

def add_to_js(self, name, var):
        """Add an object to Javascript."""
        frame = self.page().mainFrame()
        frame.addToJavaScriptWindowObject(name, var)

def base_path(self):
        """Base absolute path of container."""
        return os.path.join(self.container.base_path, self.name)

def subn_filter(s, find, replace, count=0):
    """A non-optimal implementation of a regex filter"""
    return re.gsub(find, replace, count, s)

def _summarize_object_type(model):
    """
        This function returns the summary for a given model
    """
    # the fields for the service's model
    model_fields = {field.name: field for field in list(model.fields())}
    # summarize the model
    return {
        'fields': [{
            'name': key,
            'type': type(convert_peewee_field(value)).__name__
            } for key, value in model_fields.items()
        ]
    }

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

def up(self):
        """Go up in stack and return True if top frame"""
        if self.frame:
            self.frame = self.frame.f_back
            return self.frame is None

def close_error_dlg(self):
        """Close error dialog."""
        if self.error_dlg.dismiss_box.isChecked():
            self.dismiss_error = True
        self.error_dlg.reject()

def cart2pol(x, y):
    """Cartesian to Polar coordinates conversion."""
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho

def set_executable(filename):
    """Set the exectuable bit on the given filename"""
    st = os.stat(filename)
    os.chmod(filename, st.st_mode | stat.S_IEXEC)

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

def isToneCal(self):
        """Whether the currently selected calibration stimulus type is the calibration curve

        :returns: boolean -- if the current combo box selection is calibration curve
        """
        return self.ui.calTypeCmbbx.currentIndex() == self.ui.calTypeCmbbx.count() -1

def normalized_distance(self, image):
        """Calculates the distance of a given image to the
        original image.

        Parameters
        ----------
        image : `numpy.ndarray`
            The image that should be compared to the original image.

        Returns
        -------
        :class:`Distance`
            The distance between the given image and the original image.

        """
        return self.__distance(
            self.__original_image_for_distance,
            image,
            bounds=self.bounds())

def GeneratePassphrase(length=20):
  """Create a 20 char passphrase with easily typeable chars."""
  valid_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
  valid_chars += "0123456789 ,-_&$#"
  return "".join(random.choice(valid_chars) for i in range(length))

def idx(df, index):
    """Universal indexing for numpy and pandas objects."""
    if isinstance(df, (pd.DataFrame, pd.Series)):
        return df.iloc[index]
    else:
        return df[index, :]

def test():
    """Test program for telnetlib.

    Usage: python telnetlib.py [-d] ... [host [port]]

    Default host is localhost; default port is 23.

    """
    debuglevel = 0
    while sys.argv[1:] and sys.argv[1] == '-d':
        debuglevel = debuglevel + 1
        del sys.argv[1]
    host = 'localhost'
    if sys.argv[1:]:
        host = sys.argv[1]
    port = 0
    if sys.argv[2:]:
        portstr = sys.argv[2]
        try:
            port = int(portstr)
        except ValueError:
            port = socket.getservbyname(portstr, 'tcp')
    tn = Telnet()
    tn.set_debuglevel(debuglevel)
    tn.open(host, port)
    tn.interact()
    tn.close()

def draw_graph(G: nx.DiGraph, filename: str):
    """ Draw a networkx graph with Pygraphviz. """
    A = to_agraph(G)
    A.graph_attr["rankdir"] = "LR"
    A.draw(filename, prog="dot")

def add_form_widget_attr(field, attr_name, attr_value, replace=0):
    """
    Adds widget attributes to a bound form field.

    This is helpful if you would like to add a certain class to all your forms
    (i.e. `form-control` to all form fields when you are using Bootstrap)::

        {% load libs_tags %}
        {% for field in form.fields %}
            {% add_form_widget_attr field 'class' 'form-control' as field_ %}
            {{ field_ }}
        {% endfor %}

    The tag will check if the attr already exists and only append your value.
    If you would like to replace existing attrs, set `replace=1`::

        {% add_form_widget_attr field 'class' 'form-control' replace=1 as
          field_ %}


    """
    if not replace:
        attr = field.field.widget.attrs.get(attr_name, '')
        attr += force_text(attr_value)
        field.field.widget.attrs[attr_name] = attr
        return field
    else:
        field.field.widget.attrs[attr_name] = attr_value
        return field

def timestamp_to_microseconds(timestamp):
    """Convert a timestamp string into a microseconds value
    :param timestamp
    :return time in microseconds
    """
    timestamp_str = datetime.datetime.strptime(timestamp, ISO_DATETIME_REGEX)
    epoch_time_secs = calendar.timegm(timestamp_str.timetuple())
    epoch_time_mus = epoch_time_secs * 1e6 + timestamp_str.microsecond
    return epoch_time_mus

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

def normalize_path(filename):
    """Normalize a file/dir name for comparison purposes"""
    return os.path.normcase(os.path.realpath(os.path.normpath(_cygwin_patch(filename))))

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

def pretty_xml(data):
    """Return a pretty formated xml
    """
    parsed_string = minidom.parseString(data.decode('utf-8'))
    return parsed_string.toprettyxml(indent='\t', encoding='utf-8')

def find_last_sublist(list_, sublist):
    """Given a list, find the last occurance of a sublist within it.

    Returns:
        Index where the sublist starts, or None if there is no match.
    """
    for i in reversed(range(len(list_) - len(sublist) + 1)):
        if list_[i] == sublist[0] and list_[i:i + len(sublist)] == sublist:
            return i
    return None

def _set_tab_width(self, tab_width):
        """ Sets the width (in terms of space characters) for tab characters.
        """
        font_metrics = QtGui.QFontMetrics(self.font)
        self._control.setTabStopWidth(tab_width * font_metrics.width(' '))

        self._tab_width = tab_width

def is_valid(number):
    """determines whether the card number is valid."""
    n = str(number)
    if not n.isdigit():
        return False
    return int(n[-1]) == get_check_digit(n[:-1])

def total_regular_pixels_from_mask(mask):
    """Compute the total number of unmasked regular pixels in a masks."""

    total_regular_pixels = 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                total_regular_pixels += 1

    return total_regular_pixels

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

def is_delimiter(line):
    """ True if a line consists only of a single punctuation character."""
    return bool(line) and line[0] in punctuation and line[0]*len(line) == line

def iiscgi(application):
	"""A specialized version of the reference WSGI-CGI server to adapt to Microsoft IIS quirks.
	
	This is not a production quality interface and will behave badly under load.
	"""
	try:
		from wsgiref.handlers import IISCGIHandler
	except ImportError:
		print("Python 3.2 or newer is required.")
	
	if not __debug__:
		warnings.warn("Interactive debugging and other persistence-based processes will not work.")
	
	IISCGIHandler().run(application)

def log(logger, level, message):
    """Logs message to stderr if logging isn't initialized."""

    if logger.parent.name != 'root':
        logger.log(level, message)
    else:
        print(message, file=sys.stderr)

def setup_path():
    """Sets up the python include paths to include src"""
    import os.path; import sys

    if sys.argv[0]:
        top_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        sys.path = [os.path.join(top_dir, "src")] + sys.path
        pass
    return

def _loadf(ins):
    """ Loads a floating point value from a memory address.
    If 2nd arg. start with '*', it is always treated as
    an indirect value.
    """
    output = _float_oper(ins.quad[2])
    output.extend(_fpush())
    return output

def define_struct(defn):
    """
    Register a struct definition globally

    >>> define_struct('struct abcd {int x; int y;}')
    """
    struct = parse_type(defn)
    ALL_TYPES[struct.name] = struct
    return struct

def eof(fd):
    """Determine if end-of-file is reached for file fd."""
    b = fd.read(1)
    end = len(b) == 0
    if not end:
        curpos = fd.tell()
        fd.seek(curpos - 1)
    return end

def endless_permutations(N, random_state=None):
    """
    Generate an endless sequence of random integers from permutations of the
    set [0, ..., N).

    If we call this N times, we will sweep through the entire set without
    replacement, on the (N+1)th call a new permutation will be created, etc.

    Parameters
    ----------
    N: int
        the length of the set
    random_state: int or RandomState, optional
        random seed

    Yields
    ------
    int:
        a random int from the set [0, ..., N)
    """
    generator = check_random_state(random_state)
    while True:
        batch_inds = generator.permutation(N)
        for b in batch_inds:
            yield b

def query_fetch_one(self, query, values):
        """
        Executes a db query, gets the first value, and closes the connection.
        """
        self.cursor.execute(query, values)
        retval = self.cursor.fetchone()
        self.__close_db()
        return retval

def _strip_empty_keys(self, params):
        """Added because the Dropbox OAuth2 flow doesn't
        work when scope is passed in, which is empty.
        """
        keys = [k for k, v in params.items() if v == '']
        for key in keys:
            del params[key]

def focusNext(self, event):
        """Set focus to next item in sequence"""
        try:
            event.widget.tk_focusNext().focus_set()
        except TypeError:
            # see tkinter equivalent code for tk_focusNext to see
            # commented original version
            name = event.widget.tk.call('tk_focusNext', event.widget._w)
            event.widget._nametowidget(str(name)).focus_set()

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

def print_statements(self):
        """Print all INDRA Statements collected by the processors."""
        for i, stmt in enumerate(self.statements):
            print("%s: %s" % (i, stmt))

def is_palindrome(string, strict=True):
    """
    Checks if the string is a palindrome (https://en.wikipedia.org/wiki/Palindrome).

    :param string: String to check.
    :type string: str
    :param strict: True if white spaces matter (default), false otherwise.
    :type strict: bool
    :return: True if the string is a palindrome (like "otto", or "i topi non avevano nipoti" if strict=False),
    False otherwise
    """
    if is_full_string(string):
        if strict:
            return reverse(string) == string
        return is_palindrome(SPACES_RE.sub('', string))
    return False

def is_standalone(self):
        """Return True if Glances is running in standalone mode."""
        return (not self.args.client and
                not self.args.browser and
                not self.args.server and
                not self.args.webserver)

def first_sunday(self, year, month):
        """Get the first sunday of a month."""
        date = datetime(year, month, 1, 0)
        days_until_sunday = 6 - date.weekday()

        return date + timedelta(days=days_until_sunday)

def replace_all(filepath, searchExp, replaceExp):
    """
    Replace all the ocurrences (in a file) of a string with another value.
    """
    for line in fileinput.input(filepath, inplace=1):
        if searchExp in line:
            line = line.replace(searchExp, replaceExp)
        sys.stdout.write(line)

def make_post_request(self, url, auth, json_payload):
        """This function executes the request with the provided
        json payload and return the json response"""
        response = requests.post(url, auth=auth, json=json_payload)
        return response.json()

def IsBinary(self, filename):
		"""Returns true if the guessed mimetyped isnt't in text group."""
		mimetype = mimetypes.guess_type(filename)[0]
		if not mimetype:
			return False  # e.g. README, "real" binaries usually have an extension
		# special case for text files which don't start with text/
		if mimetype in TEXT_MIMETYPES:
			return False
		return not mimetype.startswith("text/")

def unpickle_stats(stats):
    """Unpickle a pstats.Stats object"""
    stats = cPickle.loads(stats)
    stats.stream = True
    return stats

def fmt_camel(name):
    """
    Converts name to lower camel case. Words are identified by capitalization,
    dashes, and underscores.
    """
    words = split_words(name)
    assert len(words) > 0
    first = words.pop(0).lower()
    return first + ''.join([word.capitalize() for word in words])

def create_all(self, check_first: bool = True):
        """Create the empty database (tables).

        :param bool check_first: Defaults to True, don't issue CREATEs for tables already present
         in the target database. Defers to :meth:`sqlalchemy.sql.schema.MetaData.create_all`
        """
        self._metadata.create_all(self.engine, checkfirst=check_first)

def clear_all(self):
        """ clear all files that were to be injected """
        self.injections.clear_all()
        for config_file in CONFIG_FILES:
            self.injections.clear(os.path.join("~", config_file))

def random_choice(sequence):
    """ Same as :meth:`random.choice`, but also supports :class:`set` type to be passed as sequence. """
    return random.choice(tuple(sequence) if isinstance(sequence, set) else sequence)

def _assert_is_type(name, value, value_type):
    """Assert that a value must be a given type."""
    if not isinstance(value, value_type):
        if type(value_type) is tuple:
            types = ', '.join(t.__name__ for t in value_type)
            raise ValueError('{0} must be one of ({1})'.format(name, types))
        else:
            raise ValueError('{0} must be {1}'
                             .format(name, value_type.__name__))

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

def run_command(cmd, *args):
    """
    Runs command on the system with given ``args``.
    """
    command = ' '.join((cmd, args))
    p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p.communicate()
    return p.retcode, stdout, stderr

def short_action_string(self):
        """
        Returns string with actor and verb, allowing target/object
        to be filled in manually.

        Example:
        [actor] [verb] or
        "Joe cool posted a comment"
        """
        output = "{0} ".format(self.actor)
        if self.override_string:
            output += self.override_string
        else:
            output += self.verb
        return output

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

def set_default(self, key, value):
        """Set the default value for this key.
        Default only used when no value is provided by the user via
        arg, config or env.
        """
        k = self._real_key(key.lower())
        self._defaults[k] = value

def db_exists():
    """Test if DATABASES['default'] exists"""
    logger.info("Checking to see if %s already exists", repr(DB["NAME"]))
    try:
        # Hide stderr since it is confusing here
        psql("", stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        return False
    return True

def to_unicode_repr( _letter ):
    """ helpful in situations where browser/app may recognize Unicode encoding
        in the \u0b8e type syntax but not actual unicode glyph/code-point"""
    # Python 2-3 compatible
    return u"u'"+ u"".join( [ u"\\u%04x"%ord(l) for l in _letter ] ) + u"'"

def get_active_ajax_datatable(self):
        """ Returns a single datatable according to the hint GET variable from an AJAX request. """
        data = getattr(self.request, self.request.method)
        datatables_dict = self.get_datatables(only=data['datatable'])
        return list(datatables_dict.values())[0]

def get_active_window():
    """Get the currently focused window
    """
    active_win = None
    default = wnck.screen_get_default()
    while gtk.events_pending():
        gtk.main_iteration(False)
    window_list = default.get_windows()
    if len(window_list) == 0:
        print "No Windows Found"
    for win in window_list:
        if win.is_active():
            active_win = win.get_name()
    return active_win

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

def get_bound(pts):
    """Compute a minimal rectangle that covers all the points."""
    (x0, y0, x1, y1) = (INF, INF, -INF, -INF)
    for (x, y) in pts:
        x0 = min(x0, x)
        y0 = min(y0, y)
        x1 = max(x1, x)
        y1 = max(y1, y)
    return (x0, y0, x1, y1)

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

def elXpath(self, xpath, dom=None):
        """check if element is present by css"""
        if dom is None:
            dom = self.browser
        return expect(dom.is_element_present_by_xpath, args=[xpath])

def drop_trailing_zeros_decimal(num):
    """ Drops the trailinz zeros from decimal value.
        Returns a string
    """
    out = str(num)
    return out.rstrip('0').rstrip('.') if '.' in out else out

def get_short_url(self):
        """ Returns short version of topic url (without page number) """
        return reverse('post_short_url', args=(self.forum.slug, self.slug, self.id))

def closeEvent(self, event):
        """ Called when closing this window.
        """
        logger.debug("closeEvent")
        self.argosApplication.saveSettingsIfNeeded()
        self.finalize()
        self.argosApplication.removeMainWindow(self)
        event.accept()
        logger.debug("closeEvent accepted")

def add_arguments(parser):
    """
    adds arguments for the swap urls command
    """
    parser.add_argument('-o', '--old-environment', help='Old environment name', required=True)
    parser.add_argument('-n', '--new-environment', help='New environment name', required=True)

def _convert_date_to_dict(field_date):
        """
        Convert native python ``datetime.date`` object  to a format supported by the API
        """
        return {DAY: field_date.day, MONTH: field_date.month, YEAR: field_date.year}

def normalize(X):
    """ equivalent to scipy.preprocessing.normalize on sparse matrices
    , but lets avoid another depedency just for a small utility function """
    X = coo_matrix(X)
    X.data = X.data / sqrt(bincount(X.row, X.data ** 2))[X.row]
    return X

def move_back(self, dt):
        """ If called after an update, the sprite can move back
        """
        self._position = self._old_position
        self.rect.topleft = self._position
        self.feet.midbottom = self.rect.midbottom

def is_natural(x):
    """A non-negative integer."""
    try:
        is_integer = int(x) == x
    except (TypeError, ValueError):
        return False
    return is_integer and x >= 0

def decodebytes(input):
    """Decode base64 string to byte array."""
    py_version = sys.version_info[0]
    if py_version >= 3:
        return _decodebytes_py3(input)
    return _decodebytes_py2(input)

def is_unix_style(flags):
    """Check if we should use Unix style."""

    return (util.platform() != "windows" or (not bool(flags & REALPATH) and get_case(flags))) and not flags & _FORCEWIN

def set_index(self, index):
        """ Sets the pd dataframe index of all dataframes in the system to index
        """
        for df in self.get_DataFrame(data=True, with_population=False):
            df.index = index

def predecessors(self, node, graph=None):
        """ Returns a list of all predecessors of the given node """
        if graph is None:
            graph = self.graph
        return [key for key in graph if node in graph[key]]

def has_synset(word: str) -> list:
    """" Returns a list of synsets of a word after lemmatization. """

    return wn.synsets(lemmatize(word, neverstem=True))

def access_token(self):
        """ WeChat access token """
        access_token = self.session.get(self.access_token_key)
        if access_token:
            if not self.expires_at:
                # user provided access_token, just return it
                return access_token

            timestamp = time.time()
            if self.expires_at - timestamp > 60:
                return access_token

        self.fetch_access_token()
        return self.session.get(self.access_token_key)

def format_exception(e):
    """Returns a string containing the type and text of the exception.

    """
    from .utils.printing import fill
    return '\n'.join(fill(line) for line in traceback.format_exception_only(type(e), e))

def shift(self, m: Union[float, pd.Series]) -> Union[int, pd.Series]:
        """Shifts floats so that the first 10 decimal digits are significant."""
        out = m % 1 * self.TEN_DIGIT_MODULUS // 1
        if isinstance(out, pd.Series):
            return out.astype(int)
        return int(out)

def select_up(self):
        """move cursor up"""
        r, c = self._index
        self._select_index(r-1, c)

def _check_task_id(self, context):
        """
        Gets the returned Celery result from the Airflow task
        ID provided to the sensor, and returns True if the
        celery result has been finished execution.

        :param context: Airflow's execution context
        :type context: dict
        :return: True if task has been executed, otherwise False
        :rtype: bool
        """
        ti = context['ti']
        celery_result = ti.xcom_pull(task_ids=self.target_task_id)
        return celery_result.ready()

async def _send_plain_text(self, request: Request, stack: Stack):
        """
        Sends plain text using `_send_text()`.
        """

        await self._send_text(request, stack, None)

def file_exists(self) -> bool:
        """ Check if the settings file exists or not """
        cfg_path = self.file_path
        assert cfg_path

        return path.isfile(cfg_path)

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

def context(self):
        """ Convenient access to shared context """
        if self._context is not None:
            return self._context
        else:
            logger.warning("Using shared context without a lock")
            return self._executor._shared_context

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

def cli(yamlfile, directory, out, classname, format):
    """ Generate graphviz representations of the biolink model """
    DotGenerator(yamlfile, format).serialize(classname=classname, dirname=directory, filename=out)

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

def as_html(self):
        """Generate HTML to display map."""
        if not self._folium_map:
            self.draw()
        return self._inline_map(self._folium_map, self._width, self._height)

def drop_trailing_zeros(num):
    """
    Drops the trailing zeros in a float that is printed.
    """
    txt = '%f' %(num)
    txt = txt.rstrip('0')
    if txt.endswith('.'):
        txt = txt[:-1]
    return txt

def mouse_out(self):
        """
        Performs a mouse out the element.

        Currently works only on Chrome driver.
        """
        self.scroll_to()
        ActionChains(self.parent.driver).move_by_offset(0, 0).click().perform()

def fix_line_breaks(s):
    """
    Convert \r\n and \r to \n chars. Strip any leading or trailing whitespace
    on each line. Remove blank lines.
    """
    l = s.splitlines()
    x = [i.strip() for i in l]
    x = [i for i in x if i]  # remove blank lines
    return "\n".join(x)

def __exit__(self, *args):
        """Redirect stdout back to the original stdout."""
        sys.stdout = self._orig
        self._devnull.close()

def omnihash(obj):
    """ recursively hash unhashable objects """
    if isinstance(obj, set):
        return hash(frozenset(omnihash(e) for e in obj))
    elif isinstance(obj, (tuple, list)):
        return hash(tuple(omnihash(e) for e in obj))
    elif isinstance(obj, dict):
        return hash(frozenset((k, omnihash(v)) for k, v in obj.items()))
    else:
        return hash(obj)

def get_list_representation(self):
        """Returns this subset's representation as a list of indices."""
        if self.is_list:
            return self.list_or_slice
        else:
            return self[list(range(self.num_examples))]

def get_distance_between_two_points(self, one, two):
        """Returns the distance between two XYPoints."""
        dx = one.x - two.x
        dy = one.y - two.y
        return math.sqrt(dx * dx + dy * dy)

def convert_timestamp(timestamp):
    """
    Converts bokehJS timestamp to datetime64.
    """
    datetime = dt.datetime.utcfromtimestamp(timestamp/1000.)
    return np.datetime64(datetime.replace(tzinfo=None))

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

def lazy_property(function):
    """Cache the first return value of a function for all subsequent calls.

    This decorator is usefull for argument-less functions that behave more
    like a global or static property that should be calculated once, but
    lazily (i.e. only if requested).
    """
    cached_val = []
    def _wrapper(*args):
        try:
            return cached_val[0]
        except IndexError:
            ret_val = function(*args)
            cached_val.append(ret_val)
            return ret_val
    return _wrapper

async def repeat(ctx, times: int, content='repeating...'):
    """Repeats a message multiple times."""
    for i in range(times):
        await ctx.send(content)

def correlation(df, rowvar=False):
    """
    Calculate column-wise Pearson correlations using ``numpy.ma.corrcoef``

    Input data is masked to ignore NaNs when calculating correlations. Data is returned as
    a Pandas ``DataFrame`` of column_n x column_n dimensions, with column index copied to
    both axes.

    :param df: Pandas DataFrame
    :return: Pandas DataFrame (n_columns x n_columns) of column-wise correlations
    """

    # Create a correlation matrix for all correlations
    # of the columns (filled with na for all values)
    df = df.copy()
    maskv = np.ma.masked_where(np.isnan(df.values), df.values)
    cdf = np.ma.corrcoef(maskv, rowvar=False)
    cdf = pd.DataFrame(np.array(cdf))
    cdf.columns = df.columns
    cdf.index = df.columns
    cdf = cdf.sort_index(level=0, axis=1)
    cdf = cdf.sort_index(level=0)
    return cdf

def get_path_from_query_string(req):
    """Gets path from query string

    Args:
        req (flask.request): Request object from Flask

    Returns:
        path (str): Value of "path" parameter from query string

    Raises:
        exceptions.UserError: If "path" is not found in query string
    """
    if req.args.get('path') is None:
        raise exceptions.UserError('Path not found in query string')
    return req.args.get('path')

def parse(self, s):
        """
        Parses a date string formatted like ``YYYY-MM-DD``.
        """
        return datetime.datetime.strptime(s, self.date_format).date()

def _is_proper_sequence(seq):
    """Returns is seq is sequence and not string."""
    return (isinstance(seq, collections.abc.Sequence) and
            not isinstance(seq, str))

def GetMountpoints():
  """List all the filesystems mounted on the system."""
  devices = {}

  for filesys in GetFileSystems():
    devices[filesys.f_mntonname] = (filesys.f_mntfromname, filesys.f_fstypename)

  return devices

def update(self, **kwargs):
        """Customize the lazy field"""
        assert not self.called
        self.kw.update(kwargs)
        return self

def reload(self, save_config=True):
        """Reload the device.

        !!!WARNING! there is unsaved configuration!!!
        This command will reboot the system. (y/n)?  [n]
        """
        if save_config:
            self.device.send("copy running-config startup-config")
        self.device("reload", wait_for_string="This command will reboot the system")
        self.device.ctrl.sendline("y")

def properties(self):
        """All compartment properties as a dict."""
        properties = {'id': self._id}
        if self._name is not None:
            properties['name'] = self._name

        return properties

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

def find_start_point(self):
        """
        Find the first location in our array that is not empty
        """
        for i, row in enumerate(self.data):
            for j, _ in enumerate(row):
                if self.data[i, j] != 0:  # or not np.isfinite(self.data[i,j]):
                    return i, j

def is_password_valid(password):
    """
    Check if a password is valid
    """
    pattern = re.compile(r"^.{4,75}$")
    return bool(pattern.match(password))

def _help():
    """ Display both SQLAlchemy and Python help statements """

    statement = '%s%s' % (shelp, phelp % ', '.join(cntx_.keys()))
    print statement.strip()

def is_inside_lambda(node: astroid.node_classes.NodeNG) -> bool:
    """Return true if given node is inside lambda"""
    parent = node.parent
    while parent is not None:
        if isinstance(parent, astroid.Lambda):
            return True
        parent = parent.parent
    return False

def unique_items(seq):
    """Return the unique items from iterable *seq* (in order)."""
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]

def flush(self):
        """Ensure contents are written to file."""
        for name in self.item_names:
            item = self[name]
            item.flush()
        self.file.flush()

def get_page_text(self, page):
        """
        Downloads and returns the full text of a particular page
        in the document.
        """
        url = self.get_page_text_url(page)
        return self._get_url(url)

def items_to_dict(items):
    """
    Converts list of tuples to dictionary with duplicate keys converted to
    lists.

    :param list items:
        List of tuples.

    :returns:
        :class:`dict`

    """

    res = collections.defaultdict(list)

    for k, v in items:
        res[k].append(v)

    return normalize_dict(dict(res))

def sample_colormap(cmap_name, n_samples):
    """
    Sample a colormap from matplotlib
    """
    colors = []
    colormap = cm.cmap_d[cmap_name]
    for i in np.linspace(0, 1, n_samples):
        colors.append(colormap(i))

    return colors

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

def nonlocal_check(self, original, loc, tokens):
        """Check for Python 3 nonlocal statement."""
        return self.check_py("3", "nonlocal statement", original, loc, tokens)

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

def help(self):
        """Prints discovered resources and their associated methods. Nice when
        noodling in the terminal to wrap your head around Magento's insanity.
        """

        print('Resources:')
        print('')
        for name in sorted(self._resources.keys()):
            methods = sorted(self._resources[name]._methods.keys())
            print('{}: {}'.format(bold(name), ', '.join(methods)))

def get_longest_orf(orfs):
    """Find longest ORF from the given list of ORFs."""
    sorted_orf = sorted(orfs, key=lambda x: len(x['sequence']), reverse=True)[0]
    return sorted_orf

def list_string_to_dict(string):
    """Inputs ``['a', 'b', 'c']``, returns ``{'a': 0, 'b': 1, 'c': 2}``."""
    dictionary = {}
    for idx, c in enumerate(string):
        dictionary.update({c: idx})
    return dictionary

def issuperset(self, other):
        """Report whether this RangeSet contains another set."""
        self._binary_sanity_check(other)
        return set.issuperset(self, other)

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

def findLastCharIndexMatching(text, func):
    """ Return index of last character in string for which func(char) evaluates to True. """
    for i in range(len(text) - 1, -1, -1):
      if func(text[i]):
        return i

def jsonify(symbol):
    """ returns json format for symbol """
    try:
        # all symbols have a toJson method, try it
        return json.dumps(symbol.toJson(), indent='  ')
    except AttributeError:
        pass
    return json.dumps(symbol, indent='  ')

def seconds(num):
    """
    Pause for this many seconds
    """
    now = pytime.time()
    end = now + num
    until(end)

def get_axis(array, axis, slice_num):
    """Returns a fixed axis"""

    slice_list = [slice(None)] * array.ndim
    slice_list[axis] = slice_num
    slice_data = array[tuple(slice_list)].T  # transpose for proper orientation

    return slice_data

def vertical_percent(plot, percent=0.1):
    """
    Using the size of the y axis, return a fraction of that size.
    """
    plot_bottom, plot_top = plot.get_ylim()
    return percent * (plot_top - plot_bottom)

def paste(cmd=paste_cmd, stdout=PIPE):
    """Returns system clipboard contents.
    """
    return Popen(cmd, stdout=stdout).communicate()[0].decode('utf-8')

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

def ExecuteRaw(self, position, command):
    """Send a command string to gdb."""
    self.EnsureGdbPosition(position[0], None, None)
    return gdb.execute(command, to_string=True)

def obj_to_string(obj, top=True):
    """
    Turn an arbitrary object into a unicode string. If complex (dict/list/tuple), will be json-encoded.
    """
    obj = prepare_for_json_encoding(obj)
    if type(obj) == six.text_type:
        return obj
    return json.dumps(obj)

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

def snap_to_beginning_of_week(day, weekday_start="Sunday"):
    """ Get the first day of the current week.

    :param day: The input date to snap.
    :param weekday_start: Either "Monday" or "Sunday", indicating the first day of the week.
    :returns: A date representing the first day of the current week.
    """
    delta_days = ((day.weekday() + 1) % 7) if weekday_start is "Sunday" else day.weekday()
    return day - timedelta(days=delta_days)

def load(self, name):
        """Loads and returns foreign library."""
        name = ctypes.util.find_library(name)
        return ctypes.cdll.LoadLibrary(name)

def _executemany(self, cursor, query, parameters):
        """The function is mostly useful for commands that update the database:
           any result set returned by the query is discarded."""
        try:
            self._log(query)
            cursor.executemany(query, parameters)
        except OperationalError as e:  # pragma: no cover
            logging.error('Error connecting to PostgreSQL on %s, e', self.host, e)
            self.close()
            raise

def create_path(path):
    """Creates a absolute path in the file system.

    :param path: The path to be created
    """
    import os
    if not os.path.exists(path):
        os.makedirs(path)

def viewport_to_screen_space(framebuffer_size: vec2, point: vec4) -> vec2:
    """Transform point in viewport space to screen space."""
    return (framebuffer_size * point.xy) / point.w

def _set_module_names_for_sphinx(modules: List, new_name: str):
    """ Trick sphinx into displaying the desired module in these objects' documentation. """
    for obj in modules:
        obj.__module__ = new_name

def commajoin_as_strings(iterable):
    """ Join the given iterable with ',' """
    return _(u',').join((six.text_type(i) for i in iterable))

def enter_room(self, sid, room, namespace=None):
        """Enter a room.

        The only difference with the :func:`socketio.Server.enter_room` method
        is that when the ``namespace`` argument is not given the namespace
        associated with the class is used.
        """
        return self.server.enter_room(sid, room,
                                      namespace=namespace or self.namespace)

def make_exception_message(exc):
    """
    An exception is passed in and this function
    returns the proper string depending on the result
    so it is readable enough.
    """
    if str(exc):
        return '%s: %s\n' % (exc.__class__.__name__, exc)
    else:
        return '%s\n' % (exc.__class__.__name__)

def upsert_single(db, collection, object, match_params=None):
        """
        Wrapper for pymongo.update_one()
        :param db: db connection
        :param collection: collection to update
        :param object: the modifications to apply
        :param match_params: a query that matches the documents to update
        :return: id of updated document
        """
        return str(db[collection].update_one(match_params, {"$set": object}, upsert=True).upserted_id)

def return_letters_from_string(text):
    """Get letters from string only."""
    out = ""
    for letter in text:
        if letter.isalpha():
            out += letter
    return out

def get_page_and_url(session, url):
    """
    Download an HTML page using the requests session and return
    the final URL after following redirects.
    """
    reply = get_reply(session, url)
    return reply.text, reply.url

def rotate_point(xorigin, yorigin, x, y, angle):
    """Rotate the given point by angle
    """
    rotx = (x - xorigin) * np.cos(angle) - (y - yorigin) * np.sin(angle)
    roty = (x - yorigin) * np.sin(angle) + (y - yorigin) * np.cos(angle)
    return rotx, roty

def show_progress(self):
        """If we are in a progress scope, and no log messages have been
        shown, write out another '.'"""
        if self.in_progress_hanging:
            sys.stdout.write('.')
            sys.stdout.flush()

def closeEvent(self, e):
        """Qt slot when the window is closed."""
        self.emit('close_widget')
        super(DockWidget, self).closeEvent(e)

def _extract_node_text(node):
    """Extract text from a given lxml node."""

    texts = map(
        six.text_type.strip, map(six.text_type, map(unescape, node.xpath(".//text()")))
    )
    return " ".join(text for text in texts if text)

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

def move_to(self, ypos, xpos):
        """
            move the cursor to the given co-ordinates.  Co-ordinates are 1
            based, as listed in the status area of the terminal.
        """
        # the screen's co-ordinates are 1 based, but the command is 0 based
        xpos -= 1
        ypos -= 1
        self.exec_command("MoveCursor({0}, {1})".format(ypos, xpos).encode("ascii"))

def head(filename, n=10):
    """ prints the top `n` lines of a file """
    with freader(filename) as fr:
        for _ in range(n):
            print(fr.readline().strip())

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

def mouse_get_pos():
    """

    :return:
    """
    p = POINT()
    AUTO_IT.AU3_MouseGetPos(ctypes.byref(p))
    return p.x, p.y

def _(f, x):
    """
    filter for dict, note `f` should have signature: `f::key->value->bool`
    """
    return {k: v for k, v in x.items() if f(k, v)}

def has_attribute(module_name, attribute_name):
    """Is this attribute present?"""
    init_file = '%s/__init__.py' % module_name
    return any(
        [attribute_name in init_line for init_line in open(init_file).readlines()]
    )

def explained_variance(returns, values):
    """ Calculate how much variance in returns do the values explain """
    exp_var = 1 - torch.var(returns - values) / torch.var(returns)
    return exp_var.item()

def _process_legend(self):
        """
        Disables legends if show_legend is disabled.
        """
        for l in self.handles['plot'].legend:
            l.items[:] = []
            l.border_line_alpha = 0
            l.background_fill_alpha = 0

def get_element_offset(self, ty, position):
        """
        Get byte offset of type's ty element at the given position
        """

        offset = ffi.lib.LLVMPY_OffsetOfElement(self, ty, position)
        if offset == -1:
            raise ValueError("Could not determined offset of {}th "
                    "element of the type '{}'. Is it a struct type?".format(
                    position, str(ty)))
        return offset

def process_bool_arg(arg):
    """ Determine True/False from argument """
    if isinstance(arg, bool):
        return arg
    elif isinstance(arg, basestring):
        if arg.lower() in ["true", "1"]:
            return True
        elif arg.lower() in ["false", "0"]:
            return False

def list2dict(lst):
    """Takes a list of (key,value) pairs and turns it into a dict."""

    dic = {}
    for k,v in lst: dic[k] = v
    return dic

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

def load_parameters(self, source):
        """For JSON, the source it the file path"""
        with open(source) as parameters_source:
            return json.loads(parameters_source.read())

def load_member(fqn):
    """Loads and returns a class for a given fully qualified name."""
    modulename, member_name = split_fqn(fqn)
    module = __import__(modulename, globals(), locals(), member_name)
    return getattr(module, member_name)

def set_strict(self, value):
        """
        Set the strict mode active/disable

        :param value:
        :type value: bool
        """
        assert isinstance(value, bool)
        self.__settings.set_strict(value)

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

async def iso(self, source):
        """Convert to timestamp."""
        from datetime import datetime
        unix_timestamp = int(source)
        return datetime.fromtimestamp(unix_timestamp).isoformat()

def _groups_of_size(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks."""
    # _groups_of_size('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

def show_yticklabels(self, row, column):
        """Show the y-axis tick labels for a subplot.

        :param row,column: specify the subplot.

        """
        subplot = self.get_subplot_at(row, column)
        subplot.show_yticklabels()

def to_int64(a):
    """Return view of the recarray with all int32 cast to int64."""
    # build new dtype and replace i4 --> i8
    def promote_i4(typestr):
        if typestr[1:] == 'i4':
            typestr = typestr[0]+'i8'
        return typestr

    dtype = [(name, promote_i4(typestr)) for name,typestr in a.dtype.descr]
    return a.astype(dtype)

def string(value) -> str:
        """ string dict/object/value to JSON """
        return system_json.dumps(Json(value).safe_object(), ensure_ascii=False)

def _ratelimited_get(self, *args, **kwargs):
        """Perform get request, handling rate limiting."""
        with self._ratelimiter:
            resp = self.session.get(*args, **kwargs)

        # It's possible that Space-Track will return HTTP status 500 with a
        # query rate limit violation. This can happen if a script is cancelled
        # before it has finished sleeping to satisfy the rate limit and it is
        # started again.
        #
        # Let's catch this specific instance and retry once if it happens.
        if resp.status_code == 500:
            # Let's only retry if the error page tells us it's a rate limit
            # violation.
            if 'violated your query rate limit' in resp.text:
                # Mimic the RateLimiter callback behaviour.
                until = time.time() + self._ratelimiter.period
                t = threading.Thread(target=self._ratelimit_callback, args=(until,))
                t.daemon = True
                t.start()
                time.sleep(self._ratelimiter.period)

                # Now retry
                with self._ratelimiter:
                    resp = self.session.get(*args, **kwargs)

        return resp

def Binary(x):
    """Return x as a binary type."""
    if isinstance(x, text_type) and not (JYTHON or IRONPYTHON):
        return x.encode()
    return bytes(x)

def main():
    usage="""
Userspace ioctl example

""" + Fuse.fusage
    server = FiocFS(version="%prog " + fuse.__version__,
                     usage=usage,
                     dash_s_do='setsingle')

    server.parse(errex=1)
    server.main()

def keyPressEvent(self, event):
        """
        Pyqt specific key press callback function.
        Translates and forwards events to :py:func:`keyboard_event`.
        """
        self.keyboard_event(event.key(), self.keys.ACTION_PRESS, 0)

def _print(self, msg, flush=False, end="\n"):
        """Helper function to print connection status messages when in verbose mode."""
        if self._verbose:
            print2(msg, end=end, flush=flush)

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

def _iter_keys(key):
    """! Iterate over subkeys of a key
    """
    for i in range(winreg.QueryInfoKey(key)[0]):
        yield winreg.OpenKey(key, winreg.EnumKey(key, i))

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

def clear_globals_reload_modules(self):
        """Clears globals and reloads modules"""

        self.code_array.clear_globals()
        self.code_array.reload_modules()

        # Clear result cache
        self.code_array.result_cache.clear()

def survival(value=t, lam=lam, f=failure):
    """Exponential survival likelihood, accounting for censoring"""
    return sum(f * log(lam) - lam * value)

def RandomShuffle(a, seed):
    """
    Random uniform op.
    """
    if seed:
        np.random.seed(seed)
    r = a.copy()
    np.random.shuffle(r)
    return r,

def _heapify_max(x):
    """Transform list into a maxheap, in-place, in O(len(x)) time."""
    n = len(x)
    for i in reversed(range(n//2)):
        _siftup_max(x, i)

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

def pagerank_limit_push(s, r, w_i, a_i, push_node, rho):
    """
    Performs a random step without a self-loop.
    """
    # Calculate the A and B quantities to infinity
    A_inf = rho*r[push_node]
    B_inf = (1-rho)*r[push_node]

    # Update approximate Pagerank and residual vectors
    s[push_node] += A_inf
    r[push_node] = 0.0

    # Update residual vector at push node's adjacent nodes
    r[a_i] += B_inf * w_i

def _read_text(self, filename):
        """
        Helper that reads the UTF-8 content of the specified file, or
        None if the file doesn't exist. This returns a unicode string.
        """
        with io.open(filename, 'rt', encoding='utf-8') as f:
            return f.read()

def timeit(self, metric, func, *args, **kwargs):
        """Time execution of callable and emit metric then return result."""
        return metrics.timeit(metric, func, *args, **kwargs)

def _uniform_phi(M):
        """
        Generate M random numbers in [-pi, pi).

        """
        return np.random.uniform(-np.pi, np.pi, M)

def set_clear_color(self, color='black', alpha=None):
        """Set the screen clear color

        This is a wrapper for gl.glClearColor.

        Parameters
        ----------
        color : str | tuple | instance of Color
            Color to use. See vispy.color.Color for options.
        alpha : float | None
            Alpha to use.
        """
        self.glir.command('FUNC', 'glClearColor', *Color(color, alpha).rgba)

def median(data):
    """Calculate the median of a list."""
    data.sort()
    num_values = len(data)
    half = num_values // 2
    if num_values % 2:
        return data[half]
    return 0.5 * (data[half-1] + data[half])

def delete_all_eggs(self):
        """ delete all the eggs in the directory specified """
        path_to_delete = os.path.join(self.egg_directory, "lib", "python")
        if os.path.exists(path_to_delete):
            shutil.rmtree(path_to_delete)

def _get_var_from_string(item):
    """ Get resource variable. """
    modname, varname = _split_mod_var_names(item)
    if modname:
        mod = __import__(modname, globals(), locals(), [varname], -1)
        return getattr(mod, varname)
    else:
        return globals()[varname]

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

def truncate(value: Decimal, n_digits: int) -> Decimal:
    """Truncates a value to a number of decimals places"""
    return Decimal(math.trunc(value * (10 ** n_digits))) / (10 ** n_digits)

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

def ILIKE(pattern):
    """Unix shell-style wildcards. Case-insensitive"""
    return P(lambda x: fnmatch.fnmatch(x.lower(), pattern.lower()))

def max(self):
        """
        The maximum integer value of a value-set. It is only defined when there is exactly one region.

        :return: A integer that represents the maximum integer value of this value-set.
        :rtype:  int
        """

        if len(self.regions) != 1:
            raise ClaripyVSAOperationError("'max()' onlly works on single-region value-sets.")

        return self.get_si(next(iter(self.regions))).max

def validate_int(value):
    """ Integer validator """

    if value and not isinstance(value, int):
        try:
            int(str(value))
        except (TypeError, ValueError):
            raise ValidationError('not a valid number')
    return value

def traverse_setter(obj, attribute, value):
    """
    Traverses the object and sets the supplied attribute on the
    object. Supports Dimensioned and DimensionedPlot types.
    """
    obj.traverse(lambda x: setattr(x, attribute, value))

def wrap(string, length, indent):
    """ Wrap a string at a line length """
    newline = "\n" + " " * indent
    return newline.join((string[i : i + length] for i in range(0, len(string), length)))

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

def wrap(text, width=70, **kwargs):
    """Wrap a single paragraph of text, returning a list of wrapped lines.

    Reformat the single paragraph in 'text' so it fits in lines of no
    more than 'width' columns, and return a list of wrapped lines.  By
    default, tabs in 'text' are expanded with string.expandtabs(), and
    all other whitespace characters (including newline) are converted to
    space.  See TextWrapper class for available keyword args to customize
    wrapping behaviour.
    """
    w = TextWrapper(width=width, **kwargs)
    return w.wrap(text)

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

def fetch_hg_push_log(repo_name, repo_url):
    """
    Run a HgPushlog etl process
    """
    newrelic.agent.add_custom_parameter("repo_name", repo_name)
    process = HgPushlogProcess()
    process.run(repo_url + '/json-pushes/?full=1&version=2', repo_name)

def round_figures(x, n):
    """Returns x rounded to n significant figures."""
    return round(x, int(n - math.ceil(math.log10(abs(x)))))

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

def ask_dir(self):
		"""
		dialogue box for choosing directory
		"""
		args ['directory'] = askdirectory(**self.dir_opt) 
		self.dir_text.set(args ['directory'])

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

def join_cols(cols):
    """Join list of columns into a string for a SQL query"""
    return ", ".join([i for i in cols]) if isinstance(cols, (list, tuple, set)) else cols

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

def set_log_level(logger_name: str, log_level: str, propagate: bool = False):
    """Set the log level of the specified logger."""
    log = logging.getLogger(logger_name)
    log.propagate = propagate
    log.setLevel(log_level)

def stop(self, reason=None):
        """Shutdown the service with a reason."""
        self.logger.info('stopping')
        self.loop.stop(pyev.EVBREAK_ALL)

def min_values(args):
    """ Return possible range for min function. """
    return Interval(min(x.low for x in args), min(x.high for x in args))

def circstd(dts, axis=2):
    """Circular standard deviation"""
    R = np.abs(np.exp(1.0j * dts).mean(axis=axis))
    return np.sqrt(-2.0 * np.log(R))

def __isub__(self, other):
        """Remove all elements of another set from this RangeSet."""
        self._binary_sanity_check(other)
        set.difference_update(self, other)
        return self

def is_integer(dtype):
  """Returns whether this is a (non-quantized) integer type."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'is_integer'):
    return dtype.is_integer
  return np.issubdtype(np.dtype(dtype), np.integer)

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

def printdict(adict):
    """printdict"""
    dlist = list(adict.keys())
    dlist.sort()
    for i in range(0, len(dlist)):
        print(dlist[i], adict[dlist[i]])

def filter_(stream_spec, filter_name, *args, **kwargs):
    """Alternate name for ``filter``, so as to not collide with the
    built-in python ``filter`` operator.
    """
    return filter(stream_spec, filter_name, *args, **kwargs)

def login(self, user: str, passwd: str) -> None:
        """Log in to instagram with given username and password and internally store session object.

        :raises InvalidArgumentException: If the provided username does not exist.
        :raises BadCredentialsException: If the provided password is wrong.
        :raises ConnectionException: If connection to Instagram failed.
        :raises TwoFactorAuthRequiredException: First step of 2FA login done, now call :meth:`Instaloader.two_factor_login`."""
        self.context.login(user, passwd)

def parse_float_literal(ast, _variables=None):
    """Parse a float value node in the AST."""
    if isinstance(ast, (FloatValueNode, IntValueNode)):
        return float(ast.value)
    return INVALID

def get_current_desktop(self):
        """
        Get the current desktop.
        Uses ``_NET_CURRENT_DESKTOP`` of the EWMH spec.
        """
        desktop = ctypes.c_long(0)
        _libxdo.xdo_get_current_desktop(self._xdo, ctypes.byref(desktop))
        return desktop.value

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

def getcoef(self):
        """Get final coefficient map array."""

        global mp_Z_Y1
        return np.swapaxes(mp_Z_Y1, 0, self.xstep.cri.axisK+1)[0]

def isroutine(object):
    """Return true if the object is any kind of function or method."""
    return (isbuiltin(object)
            or isfunction(object)
            or ismethod(object)
            or ismethoddescriptor(object))

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

def merge(self, other):
        """
        Merge this range object with another (ranges need not overlap or abut).

        :returns: a new Range object representing the interval containing both
                  ranges.
        """
        newstart = min(self._start, other.start)
        newend = max(self._end, other.end)
        return Range(newstart, newend)

def unsort_vector(data, indices_of_increasing):
    """Upermutate 1-D data that is sorted by indices_of_increasing."""
    return numpy.array([data[indices_of_increasing.index(i)] for i in range(len(data))])

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

def setDictDefaults (d, defaults):
  """Sets all defaults for the given dictionary to those contained in a
  second defaults dictionary.  This convenience method calls:

    d.setdefault(key, value)

  for each key and value in the given defaults dictionary.
  """
  for key, val in defaults.items():
    d.setdefault(key, val)

  return d

def prompt(*args, **kwargs):
    """Prompt the user for input and handle any abort exceptions."""
    try:
        return click.prompt(*args, **kwargs)
    except click.Abort:
        return False

def get_cell(self, index):
        """
        For a single index and return the value

        :param index: index value
        :return: value
        """
        i = sorted_index(self._index, index) if self._sort else self._index.index(index)
        return self._data[i]

def string_format_func(s):
	"""
	Function used internally to format string data for output to XML.
	Escapes back-slashes and quotes, and wraps the resulting string in
	quotes.
	"""
	return u"\"%s\"" % unicode(s).replace(u"\\", u"\\\\").replace(u"\"", u"\\\"")

def list_backends(_):
    """List all available backends."""
    backends = [b.__name__ for b in available_backends()]
    print('\n'.join(backends))

def json_datetime_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, date)):
        serial = obj.isoformat()
        return serial

    if ObjectId is not None and isinstance(obj, ObjectId):
        # TODO: try to use bson.json_util instead
        return str(obj)

    raise TypeError("Type not serializable")

def retrieve_by_id(self, id_):
        """Return a JSSObject for the element with ID id_"""
        items_with_id = [item for item in self if item.id == int(id_)]
        if len(items_with_id) == 1:
            return items_with_id[0].retrieve()

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

def _cursorLeft(self):
        """ Handles "cursor left" events """
        if self.cursorPos > 0:
            self.cursorPos -= 1
            sys.stdout.write(console.CURSOR_LEFT)
            sys.stdout.flush()

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

def clean_map(obj: Mapping[Any, Any]) -> Mapping[Any, Any]:
    """
    Return a new copied dictionary without the keys with ``None`` values from
    the given Mapping object.
    """
    return {k: v for k, v in obj.items() if v is not None}

def kill_all(self, kill_signal, kill_shell=False):
        """Kill all running processes."""
        for key in self.processes.keys():
            self.kill_process(key, kill_signal, kill_shell)

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

def get_chat_member(self, user_id):
        """
        Get information about a member of a chat.

        :param int user_id: Unique identifier of the target user
        """
        return self.bot.api_call(
            "getChatMember", chat_id=str(self.id), user_id=str(user_id)
        )

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

def updateFromKwargs(self, properties, kwargs, collector, **unused):
        """Primary entry point to turn 'kwargs' into 'properties'"""
        properties[self.name] = self.getFromKwargs(kwargs)

def make_regex(separator):
    """Utility function to create regexp for matching escaped separators
    in strings.

    """
    return re.compile(r'(?:' + re.escape(separator) + r')?((?:[^' +
                      re.escape(separator) + r'\\]|\\.)+)')

def dt2ts(dt):
    """Converts to float representing number of seconds since 1970-01-01 GMT."""
    # Note: no assertion to really keep this fast
    assert isinstance(dt, (datetime.datetime, datetime.date))
    ret = time.mktime(dt.timetuple())
    if isinstance(dt, datetime.datetime):
        ret += 1e-6 * dt.microsecond
    return ret

def security(self):
        """Print security object information for a pdf document"""
        return {k: v for i in self.pdf.resolvedObjects.items() for k, v in i[1].items()}

def numberp(v):
    """Return true iff 'v' is a number."""
    return (not(isinstance(v, bool)) and
            (isinstance(v, int) or isinstance(v, float)))

def fit_gaussian(samples, ddof=0):
    """Calculates the mean and the standard deviation of the given samples.

    Args:
        samples (ndarray): a one or two dimensional array. If one dimensional we calculate the fit using all
            values. If two dimensional, we fit the Gaussian for every set of samples over the first dimension.
        ddof (int): the difference degrees of freedom in the std calculation. See numpy.
    """
    if len(samples.shape) == 1:
        return np.mean(samples), np.std(samples, ddof=ddof)
    return np.mean(samples, axis=1), np.std(samples, axis=1, ddof=ddof)

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

def _rel(self, path):
        """
        Get the relative path for the given path from the current
        file by working around https://bugs.python.org/issue20012.
        """
        return os.path.relpath(
            str(path), self._parent).replace(os.path.sep, '/')

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

def _show_traceback(method):
    """decorator for showing tracebacks in IPython"""
    def m(self, *args, **kwargs):
        try:
            return(method(self, *args, **kwargs))
        except Exception as e:
            ip = get_ipython()
            if ip is None:
                self.log.warning("Exception in widget method %s: %s", method, e, exc_info=True)
            else:
                ip.showtraceback()
    return m

def add_header(self, name, value):
        """ Add an additional response header, not removing duplicates. """
        self._headers.setdefault(_hkey(name), []).append(_hval(value))

def create_aws_lambda(ctx, bucket, region_name, aws_access_key_id, aws_secret_access_key):
    """Creates an AWS Chalice project for deployment to AWS Lambda."""
    from canari.commands.create_aws_lambda import create_aws_lambda
    create_aws_lambda(ctx.project, bucket, region_name, aws_access_key_id, aws_secret_access_key)

def stop(self, timeout=None):
        """Stop the thread."""
        logger.debug("docker plugin - Close thread for container {}".format(self._container.name))
        self._stopper.set()

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

def stepBy(self, steps):
        """steps value up/down by a single step. Single step is defined in singleStep().

        Args:
            steps (int): positiv int steps up, negativ steps down
        """
        self.setValue(self.value() + steps*self.singleStep())

def _mid(string, start, end=None):
    """
    Returns a substring delimited by start and end position.
    """
    if end is None:
        end = len(string)
    return string[start:start + end]

def set(self):
        """Set the internal flag to true.

        All threads waiting for the flag to become true are awakened. Threads
        that call wait() once the flag is true will not block at all.

        """
        with self.__cond:
            self.__flag = True
            self.__cond.notify_all()

async def delete(self):
        """
        Delete task (in any state) permanently.

        Returns `True` is task is deleted.
        """
        the_tuple = await self.queue.delete(self.tube, self.task_id)

        self.update_from_tuple(the_tuple)

        return bool(self.state == DONE)

def convert(self, value, _type):
        """
        Convert instances of textx types and match rules to python types.
        """
        return self.type_convertors.get(_type, lambda x: x)(value)

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

def _compile(pattern, flags):
    """Compile the pattern to regex."""

    return re.compile(WcParse(pattern, flags & FLAG_MASK).parse())

def post_object_async(self, path, **kwds):
    """POST to an object."""
    return self.do_request_async(self.api_url + path, 'POST', **kwds)

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

def upgrade(directory, sql, tag, x_arg, revision):
    """Upgrade to a later version"""
    _upgrade(directory, revision, sql, tag, x_arg)

def classify_fit(fqdn, result, *argl, **argd):
    """Analyzes the result of a classification algorithm's fitting. See also
    :func:`fit` for explanation of arguments.
    """
    if len(argl) > 2:
        #Usually fit is called with fit(machine, Xtrain, ytrain).
        yP = argl[2]
    out = _generic_fit(fqdn, result, classify_predict, yP, *argl, **argd)
    return out

def extract_pdfminer(self, filename, **kwargs):
        """Extract text from pdfs using pdfminer."""
        stdout, _ = self.run(['pdf2txt.py', filename])
        return stdout

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

def is_safe_url(url, host=None):
    """Return ``True`` if the url is a safe redirection.

    The safe redirection means that it doesn't point to a different host.
    Always returns ``False`` on an empty url.
    """
    if not url:
        return False
    netloc = urlparse.urlparse(url)[1]
    return not netloc or netloc == host

def make_indices_to_labels(labels: Set[str]) -> Dict[int, str]:
    """ Creates a mapping from indices to labels. """

    return {index: label for index, label in
            enumerate(["pad"] + sorted(list(labels)))}

def bytes_to_str(s, encoding='utf-8'):
    """Returns a str if a bytes object is given."""
    if six.PY3 and isinstance(s, bytes):
        return s.decode(encoding)
    return s

def is_iterable(obj):
    """
    Are we being asked to look up a list of things, instead of a single thing?
    We check for the `__iter__` attribute so that this can cover types that
    don't have to be known by this module, such as NumPy arrays.

    Strings, however, should be considered as atomic values to look up, not
    iterables. The same goes for tuples, since they are immutable and therefore
    valid entries.

    We don't need to check for the Python 2 `unicode` type, because it doesn't
    have an `__iter__` attribute anyway.
    """
    return (
        hasattr(obj, "__iter__")
        and not isinstance(obj, str)
        and not isinstance(obj, tuple)
    )

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

def set_color(self, fg=None, bg=None, intensify=False, target=sys.stdout):
        """Set foreground- and background colors and intensity."""
        raise NotImplementedError

def get_process_handle(self):
        """
        @rtype:  L{ProcessHandle}
        @return: Process handle received from the system.
            Returns C{None} if the handle is not available.
        """
        # The handle doesn't need to be closed.
        # See http://msdn.microsoft.com/en-us/library/ms681423(VS.85).aspx
        hProcess = self.raw.u.CreateProcessInfo.hProcess
        if hProcess in (0, win32.NULL, win32.INVALID_HANDLE_VALUE):
            hProcess = None
        else:
            hProcess = ProcessHandle(hProcess, False, win32.PROCESS_ALL_ACCESS)
        return hProcess

def set_file_mtime(path, mtime, atime=None):
  """Set access and modification times on a file."""
  if not atime:
    atime = mtime
  f = open(path, 'a')
  try:
    os.utime(path, (atime, mtime))
  finally:
    f.close()

def uuid(self, version: int = None) -> str:
        """Generate random UUID.

        :param version: UUID version.
        :return: UUID
        """
        bits = self.random.getrandbits(128)
        return str(uuid.UUID(int=bits, version=version))

def get_querystring(uri):
    """Get Querystring information from uri.

    :param uri: uri
    :return: querystring info or {}
    """
    parts = urlparse.urlsplit(uri)
    return urlparse.parse_qs(parts.query)

def _get_history_next(self):
        """ callback function for key down """
        if self._has_history:
            ret = self._input_history.return_history(1)
            self.string = ret
            self._curs_pos = len(ret)

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

def resources(self):
        """Retrieve contents of each page of PDF"""
        return [self.pdf.getPage(i) for i in range(self.pdf.getNumPages())]

def logx_linear(x, a, b):
    """logx linear

    Parameters
    ----------
    x: int
    a: float
    b: float

    Returns
    -------
    float
        a * np.log(x) + b
    """
    x = np.log(x)
    return a*x + b

def destroy(self):
		"""Finish up a session.
		"""
		if self.session_type == 'bash':
			# TODO: does this work/handle already being logged out/logged in deep OK?
			self.logout()
		elif self.session_type == 'vagrant':
			# TODO: does this work/handle already being logged out/logged in deep OK?
			self.logout()

def incr(name, value=1, rate=1, tags=None):
    """Increment a metric by value.

    >>> import statsdecor
    >>> statsdecor.incr('my.metric')
    """
    client().incr(name, value, rate, tags)

def get_domain(url):
    """
    Get domain part of an url.

    For example: https://www.python.org/doc/ -> https://www.python.org
    """
    parse_result = urlparse(url)
    domain = "{schema}://{netloc}".format(
        schema=parse_result.scheme, netloc=parse_result.netloc)
    return domain

def send(message, request_context=None, binary=False):
    """Sends a message to websocket.

    :param str message: data to send

    :param request_context:

    :raises IOError: If unable to send a message.
    """
    if binary:
        return uwsgi.websocket_send_binary(message, request_context)

    return uwsgi.websocket_send(message, request_context)

async def set_http_proxy(cls, url: typing.Optional[str]):
        """See `get_http_proxy`."""
        await cls.set_config("http_proxy", "" if url is None else url)

def activate(self):
        """Store ipython references in the __builtin__ namespace."""

        add_builtin = self.add_builtin
        for name, func in self.auto_builtins.iteritems():
            add_builtin(name, func)

def crop_box(im, box=False, **kwargs):
    """Uses box coordinates to crop an image without resizing it first."""
    if box:
        im = im.crop(box)
    return im

def parse_markdown(markdown_content, site_settings):
    """Parse markdown text to html.

    :param markdown_content: Markdown text lists #TODO#
    """
    markdown_extensions = set_markdown_extensions(site_settings)

    html_content = markdown.markdown(
        markdown_content,
        extensions=markdown_extensions,
    )

    return html_content

def abort(err):
    """Abort everything, everywhere."""
    if _debug: abort._debug("abort %r", err)
    global local_controllers

    # tell all the local controllers to abort
    for controller in local_controllers.values():
        controller.abort(err)

def flatten_all_but_last(a):
  """Flatten all dimensions of a except the last."""
  ret = tf.reshape(a, [-1, tf.shape(a)[-1]])
  if not tf.executing_eagerly():
    ret.set_shape([None] + a.get_shape().as_list()[-1:])
  return ret

def unique(iterable):
    """ Returns a list copy in which each item occurs only once (in-order).
    """
    seen = set()
    return [x for x in iterable if x not in seen and not seen.add(x)]

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

def _is_name_used_as_variadic(name, variadics):
    """Check if the given name is used as a variadic argument."""
    return any(
        variadic.value == name or variadic.value.parent_of(name)
        for variadic in variadics
    )

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

def _unique_id(self, prefix):
        """
        Generate a unique (within the graph) identifer
        internal to graph generation.
        """
        _id = self._id_gen
        self._id_gen += 1
        return prefix + str(_id)

def standardize():
    """
    return variant standarize function
    """

    def f(G, bim):
        G_out = standardize_snps(G)
        return G_out, bim

    return f

def is_file_exists_error(e):
    """
    Returns whether the exception *e* was raised due to an already existing file or directory.
    """
    if six.PY3:
        return isinstance(e, FileExistsError)  # noqa: F821
    else:
        return isinstance(e, OSError) and e.errno == 17

def get_indent(text):
    """Get indent of text.

    https://stackoverflow.com/questions/2268532/grab-a-lines-whitespace-
    indention-with-python
    """
    indent = ''

    ret = re.match(r'(\s*)', text)
    if ret:
        indent = ret.group(1)

    return indent

def check_auth(username, pwd):
    """This function is called to check if a username /
    password combination is valid.
    """
    cfg = get_current_config()
    return username == cfg["dashboard_httpauth"].split(
        ":")[0] and pwd == cfg["dashboard_httpauth"].split(":")[1]

def querySQL(self, sql, args=()):
        """For use with SELECT (or SELECT-like PRAGMA) statements.
        """
        if self.debug:
            result = timeinto(self.queryTimes, self._queryandfetch, sql, args)
        else:
            result = self._queryandfetch(sql, args)
        return result

def dump_to_log(self, logger):
        """Send the cmd info and collected stdout to logger."""
        logger.error("Execution ended in %s for cmd %s", self._retcode, self._cmd)
        for line in self._collected_stdout:
            logger.error(STDOUT_LOG_PREFIX + line)

def pretty_dict(d):
    """Return dictionary d's repr but with the items sorted.
    >>> pretty_dict({'m': 'M', 'a': 'A', 'r': 'R', 'k': 'K'})
    "{'a': 'A', 'k': 'K', 'm': 'M', 'r': 'R'}"
    >>> pretty_dict({z: C, y: B, x: A})
    '{x: A, y: B, z: C}'
    """
    return '{%s}' % ', '.join('%r: %r' % (k, v)
                              for k, v in sorted(d.items(), key=repr))

def apply_kwargs(func, **kwargs):
    """Call *func* with kwargs, but only those kwargs that it accepts.
    """
    new_kwargs = {}
    params = signature(func).parameters
    for param_name in params.keys():
        if param_name in kwargs:
            new_kwargs[param_name] = kwargs[param_name]
    return func(**new_kwargs)

def unpunctuate(s, *, char_blacklist=string.punctuation):
    """ Remove punctuation from string s. """
    # remove punctuation
    s = "".join(c for c in s if c not in char_blacklist)
    # remove consecutive spaces
    return " ".join(filter(None, s.split(" ")))

def reconnect(self):
        """Reconnect to rabbitmq server"""
        import pika
        import pika.exceptions

        self.connection = pika.BlockingConnection(pika.URLParameters(self.amqp_url))
        self.channel = self.connection.channel()
        try:
            self.channel.queue_declare(self.name)
        except pika.exceptions.ChannelClosed:
            self.connection = pika.BlockingConnection(pika.URLParameters(self.amqp_url))
            self.channel = self.connection.channel()

def check_dependency(self, dependency_path):
        """Check if mtime of dependency_path is greater than stored mtime."""
        stored_hash = self._stamp_file_hashes.get(dependency_path)

        # This file was newly added, or we don't have a file
        # with stored hashes yet. Assume out of date.
        if not stored_hash:
            return False

        return stored_hash == _sha1_for_file(dependency_path)

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

def find_if_expression_as_statement(node):
    """Finds an "if" expression as a statement"""
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.IfExp)
    )

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

def linebuffered_stdout():
    """ Always line buffer stdout so pipes and redirects are CLI friendly. """
    if sys.stdout.line_buffering:
        return sys.stdout
    orig = sys.stdout
    new = type(orig)(orig.buffer, encoding=orig.encoding, errors=orig.errors,
                     line_buffering=True)
    new.mode = orig.mode
    return new

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

def __PrintEnumDocstringLines(self, enum_type):
        description = enum_type.description or '%s enum type.' % enum_type.name
        for line in textwrap.wrap('r"""%s' % description,
                                  self.__printer.CalculateWidth()):
            self.__printer(line)
        PrintIndentedDescriptions(self.__printer, enum_type.values, 'Values')
        self.__printer('"""')

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

def cross_product_matrix(vec):
    """Returns a 3x3 cross-product matrix from a 3-element vector."""
    return np.array([[0, -vec[2], vec[1]],
                     [vec[2], 0, -vec[0]],
                     [-vec[1], vec[0], 0]])

def changed(self, *value):
        """Checks whether the value has changed since the last call."""
        if self._last_checked_value != value:
            self._last_checked_value = value
            return True
        return False

def filename_addstring(filename, text):
    """
    Add `text` to filename, keeping the extension in place
    For example when adding a timestamp to the filename
    """
    fn, ext = os.path.splitext(filename)
    return fn + text + ext

def forget_canvas(canvas):
    """ Forget about the given canvas. Used by the canvas when closed.
    """
    cc = [c() for c in canvasses if c() is not None]
    while canvas in cc:
        cc.remove(canvas)
    canvasses[:] = [weakref.ref(c) for c in cc]

def tag(self, nerdoc):
        """Tag the given document.
        Parameters
        ----------
        nerdoc: estnltk.estner.Document
            The document to be tagged.

        Returns
        -------
        labels: list of lists of str
            Predicted token Labels for each sentence in the document
        """

        labels = []
        for snt in nerdoc.sentences:
            xseq = [t.feature_list() for t in snt]
            yseq = self.tagger.tag(xseq)
            labels.append(yseq)
        return labels

def _get_ipv6_from_binary(self, bin_addr):
        """Converts binary address to Ipv6 format."""

        hi = bin_addr >> 64
        lo = bin_addr & 0xFFFFFFFF
        return socket.inet_ntop(socket.AF_INET6, struct.pack("!QQ", hi, lo))

def bitdepth(self):
        """The number of bits per sample in the audio encoding (an int).
        Only available for certain file formats (zero where
        unavailable).
        """
        if hasattr(self.mgfile.info, 'bits_per_sample'):
            return self.mgfile.info.bits_per_sample
        return 0

def is_valid_folder(parser, arg):
    """Check if arg is a valid file that already exists on the file system."""
    arg = os.path.abspath(arg)
    if not os.path.isdir(arg):
        parser.error("The folder %s does not exist!" % arg)
    else:
        return arg

def _heappop_max(heap):
    """Maxheap version of a heappop."""
    lastelt = heap.pop()    # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        _siftup_max(heap, 0)
        return returnitem
    return lastelt

def unit_ball_L_inf(shape, precondition=True):
  """A tensorflow variable tranfomed to be constrained in a L_inf unit ball.

  Note that this code also preconditions the gradient to go in the L_inf
  direction of steepest descent.

  EXPERIMENTAL: Do not use for adverserial examples if you need to be confident
  they are strong attacks. We are not yet confident in this code.
  """
  x = tf.Variable(tf.zeros(shape))
  if precondition:
    return constrain_L_inf_precondition(x)
  else:
    return constrain_L_inf(x)

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

def _get_node_parent(self, age, pos):
        """Get the parent node of node, whch is located in tree's node list.

        Returns:
            object: The parent node.
        """
        return self.nodes[age][int(pos / self.comp)]

def get_X0(X):
    """ Return zero-th element of a one-element data container.
    """
    if pandas_available and isinstance(X, pd.DataFrame):
        assert len(X) == 1
        x = np.array(X.iloc[0])
    else:
        x, = X
    return x

def load_library(version):
    """
    Load the correct module according to the version

    :type version: ``str``
    :param version: the version of the library to be loaded (e.g. '2.6')
    :rtype: module object
    """
    check_version(version)
    module_name = SUPPORTED_LIBRARIES[version]
    lib = sys.modules.get(module_name)
    if lib is None:
        lib = importlib.import_module(module_name)
    return lib

def _read_words(filename):
  """Reads words from a file."""
  with tf.gfile.GFile(filename, "r") as f:
    if sys.version_info[0] >= 3:
      return f.read().replace("\n", " %s " % EOS).split()
    else:
      return f.read().decode("utf-8").replace("\n", " %s " % EOS).split()

def generate_dumper(self, mapfile, names):
        """
        Build dumpdata commands
        """
        return self.build_template(mapfile, names, self._dumpdata_template)

def replace_tab_indent(s, replace="    "):
    """
    :param str s: string with tabs
    :param str replace: e.g. 4 spaces
    :rtype: str
    """
    prefix = get_indent_prefix(s)
    return prefix.replace("\t", replace) + s[len(prefix):]

def creation_time(self):
    """dfdatetime.Filetime: creation time or None if not set."""
    timestamp = self._fsntfs_attribute.get_creation_time_as_integer()
    return dfdatetime_filetime.Filetime(timestamp=timestamp)

def parse(self):
        """
        Parse file specified by constructor.
        """
        f = open(self.parse_log_path, "r")
        self.parse2(f)
        f.close()

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

def get_language_parameter(request, query_language_key='language', object=None, default=None):
    """
    Get the language parameter from the current request.
    """
    # This is the same logic as the django-admin uses.
    # The only difference is the origin of the request parameter.
    if not is_multilingual_project():
        # By default, the objects are stored in a single static language.
        # This makes the transition to multilingual easier as well.
        # The default language can operate as fallback language too.
        return default or appsettings.PARLER_LANGUAGES.get_default_language()
    else:
        # In multilingual mode, take the provided language of the request.
        code = request.GET.get(query_language_key)

        if not code:
            # forms: show first tab by default
            code = default or appsettings.PARLER_LANGUAGES.get_first_language()

        return normalize_language_code(code)

def inheritdoc(method):
    """Set __doc__ of *method* to __doc__ of *method* in its parent class.

    Since this is used on :class:`.StringMixIn`, the "parent class" used is
    ``str``. This function can be used as a decorator.
    """
    method.__doc__ = getattr(str, method.__name__).__doc__
    return method

def _next_token(self, skipws=True):
        """Increment _token to the next token and return it."""
        self._token = next(self._tokens).group(0)
        return self._next_token() if skipws and self._token.isspace() else self._token

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

def _expand(self, str, local_vars={}):
        """Expand $vars in a string."""
        return ninja_syntax.expand(str, self.vars, local_vars)

def patch_lines(x):
    """
    Draw lines between groups
    """
    for idx in range(len(x)-1):
        x[idx] = np.vstack([x[idx], x[idx+1][0,:]])
    return x

def get_wordnet_syns(word):
    """
    Utilize wordnet (installed with nltk) to get synonyms for words
    word is the input word
    returns a list of unique synonyms
    """
    synonyms = []
    regex = r"_"
    pat = re.compile(regex)
    synset = nltk.wordnet.wordnet.synsets(word)
    for ss in synset:
        for swords in ss.lemma_names:
            synonyms.append(pat.sub(" ", swords.lower()))
    synonyms = f7(synonyms)
    return synonyms

def dump_json(obj):
    """Dump Python object as JSON string."""
    return simplejson.dumps(obj, ignore_nan=True, default=json_util.default)

def del_Unnamed(df):
    """
    Deletes all the unnamed columns

    :param df: pandas dataframe
    """
    cols_del=[c for c in df.columns if 'Unnamed' in c]
    return df.drop(cols_del,axis=1)

def is_floating(self):
        """Returns whether this is a (non-quantized, real) floating point type."""
        return (
            self.is_numpy_compatible and np.issubdtype(self.as_numpy_dtype, np.floating)
        ) or self.base_dtype == bfloat16

def Print(self, output_writer):
    """Prints a human readable version of the filter.

    Args:
      output_writer (CLIOutputWriter): output writer.
    """
    if self._filters:
      output_writer.Write('Filters:\n')
      for file_entry_filter in self._filters:
        file_entry_filter.Print(output_writer)

def safe_execute_script(driver, script):
    """ When executing a script that contains a jQuery command,
        it's important that the jQuery library has been loaded first.
        This method will load jQuery if it wasn't already loaded. """
    try:
        driver.execute_script(script)
    except Exception:
        # The likely reason this fails is because: "jQuery is not defined"
        activate_jquery(driver)  # It's a good thing we can define it here
        driver.execute_script(script)

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

def prox_zero(X, step):
    """Proximal operator to project onto zero
    """
    return np.zeros(X.shape, dtype=X.dtype)

def with_headers(self, headers):
        """Sets multiple headers on the request and returns the request itself.

        Keyword arguments:
        headers -- a dict-like object which contains the headers to set.
        """
        for key, value in headers.items():
            self.with_header(key, value)
        return self

def hstrlen(self, name, key):
        """
        Return the number of bytes stored in the value of ``key``
        within hash ``name``
        """
        with self.pipe as pipe:
            return pipe.hstrlen(self.redis_key(name), key)

def makedirs(path, mode=0o777, exist_ok=False):
    """A wrapper of os.makedirs()."""
    os.makedirs(path, mode, exist_ok)

def get_user_name():
    """Get user name provide by operating system
    """

    if sys.platform == 'win32':
        #user = os.getenv('USERPROFILE')
        user = os.getenv('USERNAME')
    else:
        user = os.getenv('LOGNAME')

    return user

def get(url):
    """Recieving the JSON file from uulm"""
    response = urllib.request.urlopen(url)
    data = response.read()
    data = data.decode("utf-8")
    data = json.loads(data)
    return data

def query_sum(queryset, field):
    """
    Let the DBMS perform a sum on a queryset
    """
    return queryset.aggregate(s=models.functions.Coalesce(models.Sum(field), 0))['s']

def full_like(array, value, dtype=None):
    """ Create a shared memory array with the same shape and type as a given array, filled with `value`.
    """
    shared = empty_like(array, dtype)
    shared[:] = value
    return shared

def _get_file_sha1(file):
    """Return the SHA1 hash of the given a file-like object as ``file``.
    This will seek the file back to 0 when it's finished.

    """
    bits = file.read()
    file.seek(0)
    h = hashlib.new('sha1', bits).hexdigest()
    return h

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

async def send_message():
    """Example of sending a message."""
    jar = aiohttp.CookieJar(unsafe=True)
    websession = aiohttp.ClientSession(cookie_jar=jar)

    modem = eternalegypt.Modem(hostname=sys.argv[1], websession=websession)
    await modem.login(password=sys.argv[2])

    await modem.sms(phone=sys.argv[3], message=sys.argv[4])

    await modem.logout()
    await websession.close()

def pretty(obj, verbose=False, max_width=79, newline='\n'):
    """
    Pretty print the object's representation.
    """
    stream = StringIO()
    printer = RepresentationPrinter(stream, verbose, max_width, newline)
    printer.pretty(obj)
    printer.flush()
    return stream.getvalue()

def filter_duplicate_key(line, message, line_number, marked_line_numbers,
                         source, previous_line=''):
    """Return '' if first occurrence of the key otherwise return `line`."""
    if marked_line_numbers and line_number == sorted(marked_line_numbers)[0]:
        return ''

    return line

def delegate(self, fn, *args, **kwargs):
        """Return the given operation as an asyncio future."""
        callback = functools.partial(fn, *args, **kwargs)
        coro = self.loop.run_in_executor(self.subexecutor, callback)
        return asyncio.ensure_future(coro)

def _parse_return(cls, result):
        """Extract the result, return value and context from a result object
        """

        return_value = None
        success = result['result']
        context = result['context']

        if 'return_value' in result:
            return_value = result['return_value']

        return success, return_value, context

def running_containers(name_filter: str) -> List[str]:
    """
    :raises docker.exceptions.APIError
    """
    return [container.short_id for container in
            docker_client.containers.list(filters={"name": name_filter})]

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

def _has_fr_route(self):
        """Encapsulating the rules for whether the request was to a Flask endpoint"""
        # 404's, 405's, which might not have a url_rule
        if self._should_use_fr_error_handler():
            return True
        # for all other errors, just check if FR dispatched the route
        if not request.url_rule:
            return False
        return self.owns_endpoint(request.url_rule.endpoint)

def get_timezone() -> Tuple[datetime.tzinfo, str]:
    """Discover the current time zone and it's standard string representation (for source{d})."""
    dt = get_datetime_now().astimezone()
    tzstr = dt.strftime("%z")
    tzstr = tzstr[:-2] + ":" + tzstr[-2:]
    return dt.tzinfo, tzstr

def get_data():
    """
    Currently pretends to talk to an instrument and get back the magnitud
    and phase of the measurement.
    """

    # pretend we're measuring a noisy resonance at zero
    y = 1.0 / (1.0 + 1j*(n_x.get_value()-0.002)*1000) + _n.random.rand()*0.1

    # and that it takes time to do so
    _t.sleep(0.1)

    # return mag phase
    return abs(y), _n.angle(y, True)

def acquire_nix(lock_file):  # pragma: no cover
    """Acquire a lock file on linux or osx."""
    fd = os.open(lock_file, OPEN_MODE)

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (IOError, OSError):
        os.close(fd)
    else:
        return fd

def get_year_start(day=None):
    """Returns January 1 of the given year."""
    day = add_timezone(day or datetime.date.today())
    return day.replace(month=1).replace(day=1)

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

def is_empty(self):
        """Checks for an empty image.
        """
        if(((self.channels == []) and (not self.shape == (0, 0))) or
           ((not self.channels == []) and (self.shape == (0, 0)))):
            raise RuntimeError("Channels-shape mismatch.")
        return self.channels == [] and self.shape == (0, 0)

def get_module_path(modname):
    """Return module *modname* base path"""
    return osp.abspath(osp.dirname(sys.modules[modname].__file__))

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

def _get_image_numpy_dtype(self):
        """
        Get the numpy dtype for the image
        """
        try:
            ftype = self._info['img_equiv_type']
            npy_type = _image_bitpix2npy[ftype]
        except KeyError:
            raise KeyError("unsupported fits data type: %d" % ftype)

        return npy_type

def plfit_lsq(x,y):
    """
    Returns A and B in y=Ax^B
    http://mathworld.wolfram.com/LeastSquaresFittingPowerLaw.html
    """
    n = len(x)
    btop = n * (log(x)*log(y)).sum() - (log(x)).sum()*(log(y)).sum()
    bbottom = n*(log(x)**2).sum() - (log(x).sum())**2
    b = btop / bbottom
    a = ( log(y).sum() - b * log(x).sum() ) / n

    A = exp(a)
    return A,b

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

def globlookup(pattern, root):
    """globlookup finds filesystem objects whose relative path matches the
    given pattern.

    :param pattern: The pattern to wish to match relative filepaths to.
    :param root: The root director to search within.

    """
    for subdir, dirnames, filenames in os.walk(root):
        d = subdir[len(root) + 1:]
        files = (os.path.join(d, f) for f in filenames)
        for f in fnmatch.filter(files, pattern):
            yield f

def logical_or(self, other):
        """logical_or(t) = self(t) or other(t)."""
        return self.operation(other, lambda x, y: int(x or y))

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

def most_common(items):
    """
    Wanted functionality from Counters (new in Python 2.7).
    """
    counts = {}
    for i in items:
        counts.setdefault(i, 0)
        counts[i] += 1
    return max(six.iteritems(counts), key=operator.itemgetter(1))

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

def normalize_vector(v):
    """Take a vector and return the normalized vector
    :param v: a vector v
    :returns : normalized vector v
    """
    norm = np.linalg.norm(v)
    return v/norm if not norm == 0 else v

def last_month():
        """ Return start and end date of this month. """
        since = TODAY + delta(day=1, months=-1)
        until = since + delta(months=1)
        return Date(since), Date(until)

def from_json(cls, json_str):
        """Deserialize the object from a JSON string."""
        d = json.loads(json_str)
        return cls.from_dict(d)

def _check_2d_shape(X):
    """Check shape of array or sparse matrix.

    Assure that X is always 2D: Unlike numpy we always deal with 2D arrays.
    """
    if X.dtype.names is None and len(X.shape) != 2:
        raise ValueError('X needs to be 2-dimensional, not '
                         '{}-dimensional.'.format(len(X.shape)))

def error(self, text):
		""" Ajout d'un message de log de type ERROR """
		self.logger.error("{}{}".format(self.message_prefix, text))

def build(self, **kwargs):
        """Build the lexer."""
        self.lexer = ply.lex.lex(object=self, **kwargs)

def clean_with_zeros(self,x):
        """ set nan and inf rows from x to zero"""
        x[~np.any(np.isnan(x) | np.isinf(x),axis=1)] = 0
        return x

def pages(self):
        """Get pages, reloading the site if needed."""
        rev = self.db.get('site:rev')
        if int(rev) != self.revision:
            self.reload_site()

        return self._pages

async def unignore_all(self, ctx):
        """Unignores all channels in this server from being processed.

        To use this command you must have the Manage Channels permission or have the
        Bot Admin role.
        """
        channels = [c for c in ctx.message.server.channels if c.type is discord.ChannelType.text]
        await ctx.invoke(self.unignore, *channels)

def get_python(self):
        """Only return cursor instance if configured for multiselect"""
        if self.multiselect:
            return super(MultiSelectField, self).get_python()

        return self._get()

def asMaskedArray(self):
        """ Creates converts to a masked array
        """
        return ma.masked_array(data=self.data, mask=self.mask, fill_value=self.fill_value)

def content_type(self) -> ContentType:
        """Return receiver's content type."""
        return self._ctype if self._ctype else self.parent.content_type()

def contains_extractor(document):
    """A basic document feature extractor that returns a dict of words that the
    document contains."""
    tokens = _get_document_tokens(document)
    features = dict((u'contains({0})'.format(w), True) for w in tokens)
    return features

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

def timestamping_validate(data, schema):
    """
    Custom validation function which inserts a timestamp for when the
    validation occurred
    """
    jsonschema.validate(data, schema)
    data['timestamp'] = str(time.time())

def setPixel(self, x, y, color):
        """Set the pixel at (x,y) to the integers in sequence 'color'."""
        return _fitz.Pixmap_setPixel(self, x, y, color)

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

def reload_localzone():
    """Reload the cached localzone. You need to call this if the timezone has changed."""
    global _cache_tz
    _cache_tz = pytz.timezone(get_localzone_name())
    utils.assert_tz_offset(_cache_tz)
    return _cache_tz

def read_stdin():
    """ Read text from stdin, and print a helpful message for ttys. """
    if sys.stdin.isatty() and sys.stdout.isatty():
        print('\nReading from stdin until end of file (Ctrl + D)...')

    return sys.stdin.read()

def assert_iter(**kw):
    """
    Asserts if a given values implements a valid iterable interface.

    Arguments:
        **kw (mixed): value to check if it is an iterable.

    Raises:
        TypeError: if assertion fails.
    """
    for name, value in kw.items():
        if not isiter(value):
            raise TypeError(
                'paco: {} must be an iterable object'.format(name))

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

def sub(name, func,**kwarg):
    """ Add subparser

    """
    sp = subparsers.add_parser(name, **kwarg)
    sp.set_defaults(func=func)
    sp.arg = sp.add_argument
    return sp

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

def __sub__(self, other):
		"""
		Return a Cache containing the entries of self that are not in other.
		"""
		return self.__class__([elem for elem in self if elem not in other])

def is_seq(obj):
    """
    Check if an object is a sequence.
    """
    return (not is_str(obj) and not is_dict(obj) and
            (hasattr(obj, "__getitem__") or hasattr(obj, "__iter__")))

def camel_case(self, snake_case):
        """ Convert snake case to camel case """
        components = snake_case.split('_')
        return components[0] + "".join(x.title() for x in components[1:])

def head(self) -> Any:
        """Retrive first element in List."""

        lambda_list = self._get_value()
        return lambda_list(lambda head, _: head)

def write_float(self, number):
        """ Writes a float to the underlying output file as a 4-byte value. """
        buf = pack(self.byte_order + "f", number)
        self.write(buf)

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

def set_cursor(self, x, y):
        """
        Sets the cursor to the desired position.

        :param x: X position
        :param y: Y position
        """
        curses.curs_set(1)
        self.screen.move(y, x)

def roots(self):
    """
    Returns a list with all roots. Needs Numpy.
    """
    import numpy as np
    return np.roots(list(self.values())[::-1]).tolist()

def toJson(protoObject, indent=None):
    """
    Serialises a protobuf object as json
    """
    # Using the internal method because this way we can reformat the JSON
    js = json_format.MessageToDict(protoObject, False)
    return json.dumps(js, indent=indent)

def _check_surrounded_by_space(self, tokens, i):
        """Check that a binary operator is surrounded by exactly one space."""
        self._check_space(tokens, i, (_MUST, _MUST))

def get_median(temp_list):
    """Return median
    """
    num = len(temp_list)
    temp_list.sort()
    print(temp_list)
    if num % 2 == 0:
        median = (temp_list[int(num/2)] + temp_list[int(num/2) - 1]) / 2
    else:
        median = temp_list[int(num/2)]
    return median

def invalidate_cache(cpu, address, size):
        """ remove decoded instruction from instruction cache """
        cache = cpu.instruction_cache
        for offset in range(size):
            if address + offset in cache:
                del cache[address + offset]

def _bytes_to_json(value):
    """Coerce 'value' to an JSON-compatible representation."""
    if isinstance(value, bytes):
        value = base64.standard_b64encode(value).decode("ascii")
    return value

def StreamWrite(stream, *obj):
    """Writes Python object to Skype application stream."""
    stream.Write(base64.encodestring(pickle.dumps(obj)))

def circles_pycairo(width, height, color):
    """ Implementation of circle border with PyCairo. """

    cairo_color = color / rgb(255, 255, 255)

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)

    # draw a circle in the center
    ctx.new_path()
    ctx.set_source_rgb(cairo_color.red, cairo_color.green, cairo_color.blue)
    ctx.arc(width / 2, height / 2, width / 2, 0, 2 * pi)
    ctx.fill()

    surface.write_to_png('circles.png')

def auth_request(self, url, headers, body):
        """Perform auth request for token."""

        return self.req.post(url, headers, body=body)

def get_all_items(self):
        """
        Returns all items in the combobox dictionary.
        """
        return [self._widget.itemText(k) for k in range(self._widget.count())]

def fast_distinct(self):
        """
        Because standard distinct used on the all fields are very slow and works only with PostgreSQL database
        this method provides alternative to the standard distinct method.
        :return: qs with unique objects
        """
        return self.model.objects.filter(pk__in=self.values_list('pk', flat=True))

def path_to_list(pathstr):
    """Conver a path string to a list of path elements."""
    return [elem for elem in pathstr.split(os.path.pathsep) if elem]

def trans_from_matrix(matrix):
    """ Convert a vtk matrix to a numpy.ndarray """
    t = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            t[i, j] = matrix.GetElement(i, j)
    return t

def _defaultdict(dct, fallback=_illegal_character):
    """Wraps the given dictionary such that the given fallback function will be called when a nonexistent key is
    accessed.
    """
    out = defaultdict(lambda: fallback)
    for k, v in six.iteritems(dct):
        out[k] = v
    return out

def setup(self, pin, mode, pull_up_down=PUD_OFF):
        """Set the input or output mode for a specified pin.  Mode should be
        either OUTPUT or INPUT.
        """
        self.rpi_gpio.setup(pin, self._dir_mapping[mode],
                            pull_up_down=self._pud_mapping[pull_up_down])

def __init__(self):
        """__init__: Performs basic initialisations"""
        # Root parser
        self.parser = argparse.ArgumentParser()
        # Subparsers
        self.subparsers = self.parser.add_subparsers()
        # Parser dictionary, to avoir overwriting existing parsers
        self.parsers = {}

def join(self):
        """Joins the coordinator thread and all worker threads."""
        for thread in self.worker_threads:
            thread.join()
        WorkerThread.join(self)

def load_db(file, db, verbose=True):
    """
    Load :class:`mongomock.database.Database` from a local file.

    :param file: file path.
    :param db: instance of :class:`mongomock.database.Database`.
    :param verbose: bool, toggle on log.
    :return: loaded db.
    """
    db_data = json.load(file, verbose=verbose)
    return _load(db_data, db)

def _write_color_colorama (fp, text, color):
    """Colorize text with given color."""
    foreground, background, style = get_win_color(color)
    colorama.set_console(foreground=foreground, background=background,
      style=style)
    fp.write(text)
    colorama.reset_console()

def cols_str(columns):
    """Concatenate list of columns into a string."""
    cols = ""
    for c in columns:
        cols = cols + wrap(c) + ', '
    return cols[:-2]

def __call__(self, obj, *arg, **kw):
        """
        Call the unbound method.
        
        We essentially build a bound method and call that. This ensures that
        the code for managing observers is invoked in the same was as it would
        be for a bound method.
        """
        bound_method = self._manager.__get__(obj, obj.__class__)
        return bound_method(*arg, **kw)

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

def set_xlimits(self, row, column, min=None, max=None):
        """Set x-axis limits of a subplot.

        :param row,column: specify the subplot.
        :param min: minimal axis value
        :param max: maximum axis value

        """
        subplot = self.get_subplot_at(row, column)
        subplot.set_xlimits(min, max)

def timediff(time):
    """Return the difference in seconds between now and the given time."""
    now = datetime.datetime.utcnow()
    diff = now - time
    diff_sec = diff.total_seconds()
    return diff_sec

def _get_pattern(self, pys_style):
        """Returns xlwt.pattern for pyspread style"""

        # Return None if there is no bgcolor
        if "bgcolor" not in pys_style:
            return

        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN

        bgcolor = wx.Colour()
        bgcolor.SetRGB(pys_style["bgcolor"])
        pattern.pattern_fore_colour = self.color2idx(*bgcolor.Get())

        return pattern

def get_truetype(value):
    """Convert a string to a pythonized parameter."""
    if value in ["true", "True", "y", "Y", "yes"]:
        return True
    if value in ["false", "False", "n", "N", "no"]:
        return False
    if value.isdigit():
        return int(value)
    return str(value)

def identifierify(name):
    """ Clean up name so it works for a Python identifier. """
    name = name.lower()
    name = re.sub('[^a-z0-9]', '_', name)
    return name

def render_template(self, source, **kwargs_context):
        r"""Render a template string using sandboxed environment.

        :param source: A string containing the page source.
        :param \*\*kwargs_context: The context associated with the page.
        :returns: The rendered template.
        """
        return self.jinja_env.from_string(source).render(kwargs_context)

def _sort_lambda(sortedby='cpu_percent',
                 sortedby_secondary='memory_percent'):
    """Return a sort lambda function for the sortedbykey"""
    ret = None
    if sortedby == 'io_counters':
        ret = _sort_io_counters
    elif sortedby == 'cpu_times':
        ret = _sort_cpu_times
    return ret

def inventory(self, source_id, fetch=False, fmt='table'):
        """
        Prints a summary of all objects in the database. Input string or list of strings in **ID** or **unum**
        for specific objects.

        Parameters
        ----------
        source_id: int
            The id from the SOURCES table whose data across all tables is to be printed.
        fetch: bool
            Return the results.
        fmt: str
            Returns the data as a dictionary, array, or astropy.table given 'dict', 'array', or 'table'

        Returns
        -------
        data_tables: dict
            Returns a dictionary of astropy tables with the table name as the keys.

        """
        data_tables = {}

        t = self.query("SELECT * FROM sqlite_master WHERE type='table'", fmt='table')
        all_tables = t['name'].tolist()
        for table in ['sources'] + [t for t in all_tables if
                                    t not in ['sources', 'sqlite_sequence']]:

            try:

                # Get the columns, pull out redundant ones, and query the table for this source's data
                t = self.query("PRAGMA table_info({})".format(table), fmt='table')
                columns = np.array(t['name'])
                types = np.array(t['type'])

                if table == 'sources' or 'source_id' in columns:

                    # If printing, only get simple data types and exclude redundant 'source_id' for nicer printing
                    if not fetch:
                        columns = columns[
                            ((types == 'REAL') | (types == 'INTEGER') | (types == 'TEXT')) & (columns != 'source_id')]

                    # Query the table
                    try:
                        id = 'id' if table.lower() == 'sources' else 'source_id'
                        data = self.query(
                            "SELECT {} FROM {} WHERE {}={}".format(','.join(columns), table, id, source_id),
                            fmt='table')

                        if not data and table.lower() == 'sources':
                            print(
                            'No source with id {}. Try db.search() to search the database for a source_id.'.format(
                                source_id))

                    except:
                        data = None

                    # If there's data for this table, save it
                    if data:
                        if fetch:
                            data_tables[table] = self.query(
                                "SELECT {} FROM {} WHERE {}={}".format(','.join(columns), table, id, source_id), \
                                fetch=True, fmt=fmt)
                        else:
                            data = data[[c.lower() for c in columns]]
                            pprint(data, title=table.upper())

                else:
                    pass

            except:
                print('Could not retrieve data from {} table.'.format(table.upper()))

        if fetch: return data_tables

def process_response(self, response):
        """
        Load a JSON response.

        :param Response response: The HTTP response.
        :return dict: The JSON-loaded content.
        """
        if response.status_code != 200:
            raise TwilioException('Unable to fetch page', response)

        return json.loads(response.text)

def move_datetime_year(dt, direction, num_shifts):
    """
    Move datetime 1 year in the chosen direction.
    unit is a no-op, to keep the API the same as the day case
    """
    delta = relativedelta(years=+num_shifts)
    return _move_datetime(dt, direction, delta)

def _rescale_array(self, array, scale, zero):
        """
        Scale the input array
        """
        if scale != 1.0:
            sval = numpy.array(scale, dtype=array.dtype)
            array *= sval
        if zero != 0.0:
            zval = numpy.array(zero, dtype=array.dtype)
            array += zval

def install_postgres(user=None, dbname=None, password=None):
    """Install Postgres on remote"""
    execute(pydiploy.django.install_postgres_server,
            user=user, dbname=dbname, password=password)

def symbol_pos_int(*args, **kwargs):
    """Create a sympy.Symbol with positive and integer assumptions."""
    kwargs.update({'positive': True,
                   'integer': True})
    return sympy.Symbol(*args, **kwargs)

def lastmod(self, author):
        """Return the last modification of the entry."""
        lastitems = EntryModel.objects.published().order_by('-modification_date').filter(author=author).only('modification_date')
        return lastitems[0].modification_date

def flush_on_close(self, stream):
        """Flush tornado iostream write buffer and prevent further writes.

        Returns a future that resolves when the stream is flushed.

        """
        assert get_thread_ident() == self.ioloop_thread_id
        # Prevent futher writes
        stream.KATCPServer_closing = True
        # Write empty message to get future that resolves when buffer is flushed
        return stream.write('\n')

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

def is_builtin_object(node: astroid.node_classes.NodeNG) -> bool:
    """Returns True if the given node is an object from the __builtin__ module."""
    return node and node.root().name == BUILTINS_NAME

def nothread_quit(self, arg):
        """ quit command when there's just one thread. """

        self.debugger.core.stop()
        self.debugger.core.execution_status = 'Quit command'
        raise Mexcept.DebuggerQuit

def internal_reset(self):
        """
        internal state reset.
        used e.g. in unittests
        """
        log.critical("PIA internal_reset()")
        self.empty_key_toggle = True
        self.current_input_char = None
        self.input_repead = 0

def normalize_column_names(df):
    r""" Clean up whitespace in column names. See better version at `pugnlp.clean_columns`

    >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=['Hello World', 'not here'])
    >>> normalize_column_names(df)
    ['hello_world', 'not_here']
    """
    columns = df.columns if hasattr(df, 'columns') else df
    columns = [c.lower().replace(' ', '_') for c in columns]
    return columns

def clear_matplotlib_ticks(self, axis="both"):
        """Clears the default matplotlib ticks."""
        ax = self.get_axes()
        plotting.clear_matplotlib_ticks(ax=ax, axis=axis)

def backward_char(self, e): # (C-b)
        u"""Move back a character. """
        self.l_buffer.backward_char(self.argument_reset)
        self.finalize()

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

def handle_exception(error):
        """Simple method for handling exceptions raised by `PyBankID`.

        :param flask_pybankid.FlaskPyBankIDError error: The exception to handle.
        :return: The exception represented as a dictionary.
        :rtype: dict

        """
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

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

def set_axis_options(self, row, column, text):
        """Set additionnal options as plain text."""

        subplot = self.get_subplot_at(row, column)
        subplot.set_axis_options(text)

def find_dist_to_centroid(cvects, idx_list, weights=None):
    """ Find the centroid for a set of vectors

    Parameters
    ----------
    cvects : ~numpy.ndarray(3,nsrc) with directional cosine (i.e., x,y,z component) values

    idx_list : [int,...]
      list of the source indices in the cluster

    weights : ~numpy.ndarray(nsrc) with the weights to use.  None for equal weighting

    returns (np.ndarray(nsrc)) distances to the centroid (in degrees)
    """
    centroid = find_centroid(cvects, idx_list, weights)
    dist_vals = np.degrees(np.arccos((centroid * cvects.T[idx_list]).sum(1)))
    return dist_vals, centroid

def _match_literal(self, a, b=None):
        """Match two names."""

        return a.lower() == b if not self.case_sensitive else a == b

def _repr(obj):
    """Show the received object as precise as possible."""
    vals = ", ".join("{}={!r}".format(
        name, getattr(obj, name)) for name in obj._attribs)
    if vals:
        t = "{}(name={}, {})".format(obj.__class__.__name__, obj.name, vals)
    else:
        t = "{}(name={})".format(obj.__class__.__name__, obj.name)
    return t

def kill_mprocess(process):
    """kill process
    Args:
        process - Popen object for process
    """
    if process and proc_alive(process):
        process.terminate()
        process.communicate()
    return not proc_alive(process)

def make_bound(lower, upper, lineno):
    """ Wrapper: Creates an array bound
    """
    return symbols.BOUND.make_node(lower, upper, lineno)

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

def weekly(date=datetime.date.today()):
    """
    Weeks start are fixes at Monday for now.
    """
    return date - datetime.timedelta(days=date.weekday())

def print_tree(self, indent=2):
        """ print_tree: prints out structure of tree
            Args: indent (int): What level of indentation at which to start printing
            Returns: None
        """
        config.LOGGER.info("{indent}{data}".format(indent="   " * indent, data=str(self)))
        for child in self.children:
            child.print_tree(indent + 1)

def tuple_check(*args, func=None):
    """Check if arguments are tuple type."""
    func = func or inspect.stack()[2][3]
    for var in args:
        if not isinstance(var, (tuple, collections.abc.Sequence)):
            name = type(var).__name__
            raise TupleError(
                f'Function {func} expected tuple, {name} got instead.')

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

def bbox(img):
    """Find the bounding box around nonzero elements in the given array

    Copied from https://stackoverflow.com/a/31402351/5703449 .

    Returns:
        rowmin, rowmax, colmin, colmax
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def close_stream(self):
		""" Closes the stream. Performs cleanup. """
		self.keep_listening = False
		self.stream.stop()
		self.stream.close()

def __init__(self, scope, parent):
        """Constructor for try block structures.

        Args:
            scope (CodeEntity): The program scope where this object belongs.
            parent (CodeEntity): This object's parent in the program tree.
        """
        CodeStatement.__init__(self, scope, parent)
        self.body = CodeBlock(scope, self, explicit=True)
        self.catches = []
        self.finally_body = CodeBlock(scope, self, explicit=True)

def _change_height(self, ax, new_value):
        """Make bars in horizontal bar chart thinner"""
        for patch in ax.patches:
            current_height = patch.get_height()
            diff = current_height - new_value

            # we change the bar height
            patch.set_height(new_value)

            # we recenter the bar
            patch.set_y(patch.get_y() + diff * .5)

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

def one_hot2string(arr, vocab):
    """Convert a one-hot encoded array back to string
    """
    tokens = one_hot2token(arr)
    indexToLetter = _get_index_dict(vocab)

    return [''.join([indexToLetter[x] for x in row]) for row in tokens]

def convert_date(date):
    """Convert string to datetime object."""
    date = convert_month(date, shorten=False)
    clean_string = convert_string(date)
    return datetime.strptime(clean_string, DATE_FMT.replace('-',''))

def _kbhit_unix() -> bool:
    """
    Under UNIX: is a keystroke available?
    """
    dr, dw, de = select.select([sys.stdin], [], [], 0)
    return dr != []

def version(self):
        """Spotify version information"""
        url: str = get_url("/service/version.json")
        params = {"service": "remote"}
        r = self._request(url=url, params=params)
        return r.json()

def _request(self, data):
        """Moved out to make testing easier."""
        return requests.post(self.endpoint, data=data.encode("ascii")).content

def file_remove(self, path, filename):
        """Check if filename exists and remove
        """
        if os.path.isfile(path + filename):
            os.remove(path + filename)

def _rnd_datetime(self, start, end):
        """Internal random datetime generator.
        """
        return self.from_utctimestamp(
            random.randint(
                int(self.to_utctimestamp(start)),
                int(self.to_utctimestamp(end)),
            )
        )

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

def delete(build_folder):
    """Delete build directory and all its contents.
    """
    if _meta_.del_build in ["on", "ON"] and os.path.exists(build_folder):
        shutil.rmtree(build_folder)

def _get_or_create_stack(name):
  """Returns a thread local stack uniquified by the given name."""
  stack = getattr(_LOCAL_STACKS, name, None)
  if stack is None:
    stack = []
    setattr(_LOCAL_STACKS, name, stack)
  return stack

def dist(x1, x2, axis=0):
    """Return the distance between two points.

    Set axis=1 if x1 is a vector and x2 a matrix to get a vector of distances.
    """
    return np.linalg.norm(x2 - x1, axis=axis)

def register_service(self, service):
        """
            Register service into the system. Called by Services.
        """
        if service not in self.services:
            self.services.append(service)

def logv(msg, *args, **kwargs):
    """
    Print out a log message, only if verbose mode.
    """
    if settings.VERBOSE:
        log(msg, *args, **kwargs)

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

def most_even(number, group):
    """Divide a number into a list of numbers as even as possible."""
    count, rest = divmod(number, group)
    counts = zip_longest([count] * group, [1] * rest, fillvalue=0)
    chunks = [sum(one) for one in counts]
    logging.debug('chunks: %s', chunks)
    return chunks

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

def add_str(window, line_num, str):
    """ attempt to draw str on screen and ignore errors if they occur """
    try:
        window.addstr(line_num, 0, str)
    except curses.error:
        pass

def convert_camel_case_string(name: str) -> str:
    """Convert camel case string to snake case"""
    string = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", string).lower()

def append_text(self, txt):
        """ adds a line of text to a file """
        with open(self.fullname, "a") as myfile:
            myfile.write(txt)

def write_config(self, outfile):
        """Write the configuration dictionary to an output file."""
        utils.write_yaml(self.config, outfile, default_flow_style=False)

def copy(self):
        """Create an identical (deep) copy of this element."""
        result = self.space.element()
        result.assign(self)
        return result

def _scale_shape(dshape, scale = (1,1,1)):
    """returns the shape after scaling (should be the same as ndimage.zoom"""
    nshape = np.round(np.array(dshape) * np.array(scale))
    return tuple(nshape.astype(np.int))

def update_redirect(self):
        """
            Call it on your own endpoint's to update the back history navigation.
            If you bypass it, the next submit or back will go over it.
        """
        page_history = Stack(session.get("page_history", []))
        page_history.push(request.url)
        session["page_history"] = page_history.to_json()

def connect():
    """Connect to FTP server, login and return an ftplib.FTP instance."""
    ftp_class = ftplib.FTP if not SSL else ftplib.FTP_TLS
    ftp = ftp_class(timeout=TIMEOUT)
    ftp.connect(HOST, PORT)
    ftp.login(USER, PASSWORD)
    if SSL:
        ftp.prot_p()  # secure data connection
    return ftp

def IPYTHON_MAIN():
    """Decide if the Ipython command line is running code."""
    import pkg_resources

    runner_frame = inspect.getouterframes(inspect.currentframe())[-2]
    return (
        getattr(runner_frame, "function", None)
        == pkg_resources.load_entry_point("ipython", "console_scripts", "ipython").__name__
    )

def handle_errors(resp):
    """raise a descriptive exception on a "bad request" response"""
    if resp.status_code == 400:
        raise ApiException(json.loads(resp.content).get('message'))
    return resp

def gaussian_variogram_model(m, d):
    """Gaussian model, m is [psill, range, nugget]"""
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return psill * (1. - np.exp(-d**2./(range_*4./7.)**2.)) + nugget

def _restore_seq_field_pickle(checked_class, item_type, data):
    """Unpickling function for auto-generated PVec/PSet field types."""
    type_ = _seq_field_types[checked_class, item_type]
    return _restore_pickle(type_, data)

def _join(verb):
    """
    Join helper
    """
    data = pd.merge(verb.x, verb.y, **verb.kwargs)

    # Preserve x groups
    if isinstance(verb.x, GroupedDataFrame):
        data.plydata_groups = list(verb.x.plydata_groups)
    return data

def _latest_date(self, query, datetime_field_name):
        """Given a QuerySet and the name of field containing datetimes, return the
        latest (most recent) date.

        Return None if QuerySet is empty.

        """
        return list(
            query.aggregate(django.db.models.Max(datetime_field_name)).values()
        )[0]

def is_punctuation(text):
    """Check if given string is a punctuation"""
    return not (text.lower() in config.AVRO_VOWELS or
                text.lower() in config.AVRO_CONSONANTS)

def closing_plugin(self, cancelable=False):
        """Perform actions before parent main window is closed"""
        self.dialog_manager.close_all()
        self.shell.exit_interpreter()
        return True

def listified_tokenizer(source):
    """Tokenizes *source* and returns the tokens as a list of lists."""
    io_obj = io.StringIO(source)
    return [list(a) for a in tokenize.generate_tokens(io_obj.readline)]

def _shape(self, df):
        """
        Calculate table chape considering index levels.
        """

        row, col = df.shape
        return row + df.columns.nlevels, col + df.index.nlevels

def get_order(self, codes):
        """Return evidence codes in order shown in code2name."""
        return sorted(codes, key=lambda e: [self.ev2idx.get(e)])

def stub_main():
    """setuptools blah: it still can't run a module as a script entry_point"""
    from google.apputils import run_script_module
    import butcher.main
    run_script_module.RunScriptModule(butcher.main)

def set_attrs(self):
        """ set our object attributes """
        self.attrs.encoding = self.encoding
        self.attrs.errors = self.errors

def stop(self):
    """ Stops the playing thread and close """
    with self.lock:
      self.halting = True
      self.go.clear()

def end_of_history(event):
    """
    Move to the end of the input history, i.e., the line currently being entered.
    """
    event.current_buffer.history_forward(count=10**100)
    buff = event.current_buffer
    buff.go_to_history(len(buff._working_lines) - 1)

def needs_check(self):
        """
        Check if enough time has elapsed to perform a check().

        If this time has elapsed, a state change check through
        has_state_changed() should be performed and eventually a sync().

        :rtype: boolean
        """
        if self.lastcheck is None:
            return True
        return time.time() - self.lastcheck >= self.ipchangedetection_sleep

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

def sent2features(sentence, template):
    """ extract features in a sentence

    :type sentence: list of token, each token is a list of tag
    """
    return [word2features(sentence, i, template) for i in range(len(sentence))]

def super_lm_tpu_memtest():
  """Crazy set of hyperparameters to test memory optimizations.

  Quality will be very poor due to lack of attention layers.
  853M parameters
  This seems to run on TPU for languagemodel_lm1b8k_packed as of 2018-01-19.

  Returns:
    An hparams object.
  """
  hparams = super_lm_base()
  hparams.num_model_shards = 1
  hparams.layers = "ffn," * 8
  hparams.hidden_size = 4096
  hparams.filter_size = 12000
  hparams.batch_size = 512
  return hparams

def stop(self):
		""" Stops the video stream and resets the clock. """

		logger.debug("Stopping playback")
		# Stop the clock
		self.clock.stop()
		# Set plauyer status to ready
		self.status = READY

def format(self, record, *args, **kwargs):
        """
        Format a message in the log

        Act like the normal format, but indent anything that is a
        newline within the message.

        """
        return logging.Formatter.format(
            self, record, *args, **kwargs).replace('\n', '\n' + ' ' * 8)

def to_pydatetime(self):
        """
        Converts datetimeoffset object into Python's datetime.datetime object
        @return: time zone aware datetime.datetime
        """
        dt = datetime.datetime.combine(self._date.to_pydate(), self._time.to_pytime())
        from .tz import FixedOffsetTimezone
        return dt.replace(tzinfo=_utc).astimezone(FixedOffsetTimezone(self._offset))

def sqliteRowsToDicts(sqliteRows):
    """
    Unpacks sqlite rows as returned by fetchall
    into an array of simple dicts.

    :param sqliteRows: array of rows returned from fetchall DB call
    :return:  array of dicts, keyed by the column names.
    """
    return map(lambda r: dict(zip(r.keys(), r)), sqliteRows)

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

def show_image(self, key):
        """Show image (item is a PIL image)"""
        data = self.model.get_data()
        data[key].show()

def read_image(filepath):
  """Returns an image tensor."""
  im_bytes = tf.io.read_file(filepath)
  im = tf.image.decode_image(im_bytes, channels=CHANNELS)
  im = tf.image.convert_image_dtype(im, tf.float32)
  return im

def draw(self, mode="triangles"):
        """ Draw collection """

        gl.glDepthMask(0)
        Collection.draw(self, mode)
        gl.glDepthMask(1)

def is_valid_uid(uid):
    """
    :return: True if it is a valid DHIS2 UID, False if not
    """
    pattern = r'^[A-Za-z][A-Za-z0-9]{10}$'
    if not isinstance(uid, string_types):
        return False
    return bool(re.compile(pattern).match(uid))

async def write(self, data):
        """
        :py:func:`asyncio.coroutine`

        :py:meth:`aioftp.StreamIO.write` proxy
        """
        await self.wait("write")
        start = _now()
        await super().write(data)
        self.append("write", data, start)

def _string_width(self, s):
        """Get width of a string in the current font"""
        s = str(s)
        w = 0
        for char in s:
            char = ord(char)
            w += self.character_widths[char]
        return w * self.font_size / 1000.0

def is_integer(obj):
    """Is this an integer.

    :param object obj:
    :return:
    """
    if PYTHON3:
        return isinstance(obj, int)
    return isinstance(obj, (int, long))

def remove_dups(seq):
    """remove duplicates from a sequence, preserving order"""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def load(self):
        """Load proxy list from configured proxy source"""
        self._list = self._source.load()
        self._list_iter = itertools.cycle(self._list)

def empty(self, name, **kwargs):
        """Create an array. Keyword arguments as per
        :func:`zarr.creation.empty`."""
        return self._write_op(self._empty_nosync, name, **kwargs)

def set_interface(interface, name=''):
    """
    don't want to bother with a dsn? Use this method to make an interface available
    """
    global interfaces

    if not interface: raise ValueError('interface is empty')

    # close down the interface before we discard it
    if name in interfaces:
        interfaces[name].close()

    interfaces[name] = interface

def less_strict_bool(x):
    """Idempotent and None-safe version of strict_bool."""
    if x is None:
        return False
    elif x is True or x is False:
        return x
    else:
        return strict_bool(x)

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

def standard_db_name(file_column_name):
    """return a standard name by following rules:
    1. find all regular expression partners ((IDs)|(ID)|([A-Z][a-z]+)|([A-Z]{2,}))
    2. lower very part and join again with _
    This method is only used if values in table[model]['columns'] are str

    :param str file_column_name: name of column in file
    :return: standard name
    :rtype: str
    """
    found = id_re.findall(file_column_name)

    if not found:
        return file_column_name

    return '_'.join(x[0].lower() for x in found)

def getEdges(npArr):
  """get np array of bin edges"""
  edges = np.concatenate(([0], npArr[:,0] + npArr[:,2]))
  return np.array([Decimal(str(i)) for i in edges])

def _run_once(self):
    """Run once, should be called only from loop()"""
    try:
      self.do_wait()
      self._execute_wakeup_tasks()
      self._trigger_timers()
    except Exception as e:
      Log.error("Error occured during _run_once(): " + str(e))
      Log.error(traceback.format_exc())
      self.should_exit = True

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

def __init__(self):
    """Initializes the database file object."""
    super(Sqlite3DatabaseFile, self).__init__()
    self._connection = None
    self._cursor = None
    self.filename = None
    self.read_only = None

def Exponential(x, a, tau, y0):
    """Exponential function

    Inputs:
    -------
        ``x``: independent variable
        ``a``: scaling factor
        ``tau``: time constant
        ``y0``: additive constant

    Formula:
    --------
        ``a*exp(x/tau)+y0``
    """
    return np.exp(x / tau) * a + y0

