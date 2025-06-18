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

def _visual_width(line):
    """Get the the number of columns required to display a string"""

    return len(re.sub(colorama.ansitowin32.AnsiToWin32.ANSI_CSI_RE, "", line))

def string_format_func(s):
	"""
	Function used internally to format string data for output to XML.
	Escapes back-slashes and quotes, and wraps the resulting string in
	quotes.
	"""
	return u"\"%s\"" % unicode(s).replace(u"\\", u"\\\\").replace(u"\"", u"\\\"")

def count(lines):
  """ Counts the word frequences in a list of sentences.

  Note:
    This is a helper function for parallel execution of `Vocabulary.from_text`
    method.
  """
  words = [w for l in lines for w in l.strip().split()]
  return Counter(words)

def connect():
    """Connect to FTP server, login and return an ftplib.FTP instance."""
    ftp_class = ftplib.FTP if not SSL else ftplib.FTP_TLS
    ftp = ftp_class(timeout=TIMEOUT)
    ftp.connect(HOST, PORT)
    ftp.login(USER, PASSWORD)
    if SSL:
        ftp.prot_p()  # secure data connection
    return ftp

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

def _fullname(o):
    """Return the fully-qualified name of a function."""
    return o.__module__ + "." + o.__name__ if o.__module__ else o.__name__

def cross_product_matrix(vec):
    """Returns a 3x3 cross-product matrix from a 3-element vector."""
    return np.array([[0, -vec[2], vec[1]],
                     [vec[2], 0, -vec[0]],
                     [-vec[1], vec[0], 0]])

def method_name(func):
    """Method wrapper that adds the name of the method being called to its arguments list in Pascal case

    """
    @wraps(func)
    def _method_name(*args, **kwargs):
        name = to_pascal_case(func.__name__)
        return func(name=name, *args, **kwargs)
    return _method_name

def Timestamp(year, month, day, hour, minute, second):
    """Constructs an object holding a datetime/timestamp value."""
    return datetime.datetime(year, month, day, hour, minute, second)

def get_input(input_func, input_str):
    """
    Get input from the user given an input function and an input string
    """
    val = input_func("Please enter your {0}: ".format(input_str))
    while not val or not len(val.strip()):
        val = input_func("You didn't enter a valid {0}, please try again: ".format(input_str))
    return val

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

def ceil_nearest(x, dx=1):
    """
    ceil a number to within a given rounding accuracy
    """
    precision = get_sig_digits(dx)
    return round(math.ceil(float(x) / dx) * dx, precision)

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

def good(txt):
    """Print, emphasized 'good', the given 'txt' message"""

    print("%s# %s%s%s" % (PR_GOOD_CC, get_time_stamp(), txt, PR_NC))
    sys.stdout.flush()

def get_item_from_queue(Q, timeout=0.01):
    """ Attempts to retrieve an item from the queue Q. If Q is
        empty, None is returned.

        Blocks for 'timeout' seconds in case the queue is empty,
        so don't use this method for speedy retrieval of multiple
        items (use get_all_from_queue for that).
    """
    try:
        item = Q.get(True, 0.01)
    except Queue.Empty:
        return None

    return item

def fileopenbox(msg=None, title=None, argInitialFile=None):
    """Original doc: A dialog to get a file name.
        Returns the name of a file, or None if user chose to cancel.

        if argInitialFile contains a valid filename, the dialog will
        be positioned at that file when it appears.
        """
    return psidialogs.ask_file(message=msg, title=title, default=argInitialFile)

def create_db(app, appbuilder):
    """
        Create all your database objects (SQLAlchemy specific).
    """
    from flask_appbuilder.models.sqla import Base

    _appbuilder = import_application(app, appbuilder)
    engine = _appbuilder.get_session.get_bind(mapper=None, clause=None)
    Base.metadata.create_all(engine)
    click.echo(click.style("DB objects created", fg="green"))

def text_remove_empty_lines(text):
    """
    Whitespace normalization:

      - Strip empty lines
      - Strip trailing whitespace
    """
    lines = [ line.rstrip()  for line in text.splitlines()  if line.strip() ]
    return "\n".join(lines)

def intersect(self, other):
        """ Return a new :class:`DataFrame` containing rows only in
        both this frame and another frame.

        This is equivalent to `INTERSECT` in SQL.
        """
        return DataFrame(self._jdf.intersect(other._jdf), self.sql_ctx)

def timeit(output):
    """
    If output is string, then print the string and also time used
    """
    b = time.time()
    yield
    print output, 'time used: %.3fs' % (time.time()-b)

def _to_java_object_rdd(rdd):
    """ Return a JavaRDD of Object by unpickling

    It will convert each Python object into Java object by Pyrolite, whenever the
    RDD is serialized in batch or not.
    """
    rdd = rdd._reserialize(AutoBatchedSerializer(PickleSerializer()))
    return rdd.ctx._jvm.org.apache.spark.mllib.api.python.SerDe.pythonToJava(rdd._jrdd, True)

def cleanup_storage(*args):
    """Clean up processes after SIGTERM or SIGINT is received."""
    ShardedClusters().cleanup()
    ReplicaSets().cleanup()
    Servers().cleanup()
    sys.exit(0)

def as_dict(df, ix=':'):
    """ converts df to dict and adds a datetime field if df is datetime """
    if isinstance(df.index, pd.DatetimeIndex):
        df['datetime'] = df.index
    return df.to_dict(orient='records')[ix]

def uniqueID(size=6, chars=string.ascii_uppercase + string.digits):
    """A quick and dirty way to get a unique string"""
    return ''.join(random.choice(chars) for x in xrange(size))

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

def get_function(function_name):
    """
    Given a Python function name, return the function it refers to.
    """
    module, basename = str(function_name).rsplit('.', 1)
    try:
        return getattr(__import__(module, fromlist=[basename]), basename)
    except (ImportError, AttributeError):
        raise FunctionNotFound(function_name)

def add(self, name, desc, func=None, args=None, krgs=None):
        """Add a menu entry."""
        self.entries.append(MenuEntry(name, desc, func, args or [], krgs or {}))

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

def _xxrange(self, start, end, step_count):
        """Generate n values between start and end."""
        _step = (end - start) / float(step_count)
        return (start + (i * _step) for i in xrange(int(step_count)))

def _get_loggers():
    """Return list of Logger classes."""
    from .. import loader
    modules = loader.get_package_modules('logger')
    return list(loader.get_plugins(modules, [_Logger]))

def register_logging_factories(loader):
    """
    Registers default factories for logging standard package.

    :param loader: Loader where you want register default logging factories
    """
    loader.register_factory(logging.Logger, LoggerFactory)
    loader.register_factory(logging.Handler, LoggingHandlerFactory)

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

def get_column_keys_and_names(table):
    """
    Return a generator of tuples k, c such that k is the name of the python attribute for
    the column and c is the name of the column in the sql table.
    """
    ins = inspect(table)
    return ((k, c.name) for k, c in ins.mapper.c.items())

def now_time(str=False):
    """Get the current time."""
    if str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return datetime.datetime.now()

def reraise(error):
    """Re-raises the error that was processed by prepare_for_reraise earlier."""
    if hasattr(error, "_type_"):
        six.reraise(type(error), error, error._traceback)
    raise error

def bbox(self):
        """
        The bounding box ``(ymin, xmin, ymax, xmax)`` of the minimal
        rectangular region containing the source segment.
        """

        # (stop - 1) to return the max pixel location, not the slice index
        return (self._slice[0].start, self._slice[1].start,
                self._slice[0].stop - 1, self._slice[1].stop - 1) * u.pix

def QA_util_datetime_to_strdate(dt):
    """
    :param dt:  pythone datetime.datetime
    :return:  1999-02-01 string type
    """
    strdate = "%04d-%02d-%02d" % (dt.year, dt.month, dt.day)
    return strdate

def vectorize(values):
    """
    Takes a value or list of values and returns a single result, joined by ","
    if necessary.
    """
    if isinstance(values, list):
        return ','.join(str(v) for v in values)
    return values

def get_dt_list(fn_list):
    """Get list of datetime objects, extracted from a filename
    """
    dt_list = np.array([fn_getdatetime(fn) for fn in fn_list])
    return dt_list

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

def _time_to_json(value):
    """Coerce 'value' to an JSON-compatible representation."""
    if isinstance(value, datetime.time):
        value = value.isoformat()
    return value

def get_month_start_end_day():
    """
    Get the month start date a nd end date
    """
    t = date.today()
    n = mdays[t.month]
    return (date(t.year, t.month, 1), date(t.year, t.month, n))

def _matrix3_to_dcm_array(self, m):
        """
        Converts Matrix3 in an array
        :param m: Matrix3
        :returns: 3x3 array
        """
        assert(isinstance(m, Matrix3))
        return np.array([[m.a.x, m.a.y, m.a.z],
                         [m.b.x, m.b.y, m.b.z],
                         [m.c.x, m.c.y, m.c.z]])

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

def get_value(key, obj, default=missing):
    """Helper for pulling a keyed value off various types of objects"""
    if isinstance(key, int):
        return _get_value_for_key(key, obj, default)
    return _get_value_for_keys(key.split('.'), obj, default)

def conv_dict(self):
        """dictionary of conversion"""
        return dict(integer=self.integer, real=self.real, no_type=self.no_type)

def get_by(self, name):
    """get element by name"""
    return next((item for item in self if item.name == name), None)

def join(mapping, bind, values):
    """ Merge all the strings. Put space between them. """
    return [' '.join([six.text_type(v) for v in values if v is not None])]

def security(self):
        """Print security object information for a pdf document"""
        return {k: v for i in self.pdf.resolvedObjects.items() for k, v in i[1].items()}

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

def base_path(self):
        """Base absolute path of container."""
        return os.path.join(self.container.base_path, self.name)

def dict_to_numpy_array(d):
    """
    Convert a dict of 1d array to a numpy recarray
    """
    return fromarrays(d.values(), np.dtype([(str(k), v.dtype) for k, v in d.items()]))

def remove_ext(fname):
    """Removes the extension from a filename
    """
    bn = os.path.basename(fname)
    return os.path.splitext(bn)[0]

def set_global(node: Node, key: str, value: Any):
    """Adds passed value to node's globals"""
    node.node_globals[key] = value

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

def rm(venv_name):
    """ Removes the venv by name """
    inenv = InenvManager()
    venv = inenv.get_venv(venv_name)
    click.confirm("Delete dir {}".format(venv.path))
    shutil.rmtree(venv.path)

def monthly(date=datetime.date.today()):
    """
    Take a date object and return the first day of the month.
    """
    return datetime.date(date.year, date.month, 1)

def delete(filething):
    """Remove tags from a file.

    Args:
        filething (filething)
    Raises:
        mutagen.MutagenError
    """

    f = FLAC(filething)
    filething.fileobj.seek(0)
    f.delete(filething)

def findfirst(f, coll):
    """Return first occurrence matching f, otherwise None"""
    result = list(dropwhile(f, coll))
    return result[0] if result else None

def clear_es():
        """Clear all indexes in the es core"""
        # TODO: should receive a catalog slug.
        ESHypermap.es.indices.delete(ESHypermap.index_name, ignore=[400, 404])
        LOGGER.debug('Elasticsearch: Index cleared')

def safe_rmtree(directory):
  """Delete a directory if it's present. If it's not present, no-op."""
  if os.path.exists(directory):
    shutil.rmtree(directory, True)

def counter(items):
    """
    Simplest required implementation of collections.Counter. Required as 2.6
    does not have Counter in collections.
    """
    results = {}
    for item in items:
        results[item] = results.get(item, 0) + 1
    return results

def percentile_index(a, q):
    """
    Returns the index of the value at the Qth percentile in array a.
    """
    return np.where(
        a==np.percentile(a, q, interpolation='nearest')
    )[0][0]

def pingback_url(self, server_name, target_url):
        """
        Do a pingback call for the target URL.
        """
        try:
            server = ServerProxy(server_name)
            reply = server.pingback.ping(self.entry_url, target_url)
        except (Error, socket.error):
            reply = '%s cannot be pinged.' % target_url
        return reply

def get_java_path():
  """Get the path of java executable"""
  java_home = os.environ.get("JAVA_HOME")
  return os.path.join(java_home, BIN_DIR, "java")

def rotateImage(img, angle):
    """

    querries scipy.ndimage.rotate routine
    :param img: image to be rotated
    :param angle: angle to be rotated (radian)
    :return: rotated image
    """
    imgR = scipy.ndimage.rotate(img, angle, reshape=False)
    return imgR

def get_last_commit_line(git_path=None):
    """
    Get one-line description of HEAD commit for repository in current dir.
    """
    if git_path is None: git_path = GIT_PATH
    output = check_output([git_path, "log", "--pretty=format:'%ad %h %s'",
                           "--date=short", "-n1"])
    return output.strip()[1:-1]

def lighting(im, b, c):
    """ Adjust image balance and contrast """
    if b==0 and c==1: return im
    mu = np.average(im)
    return np.clip((im-mu)*c+mu+b,0.,1.).astype(np.float32)

def last_modified_date(filename):
    """Last modified timestamp as a UTC datetime"""
    mtime = os.path.getmtime(filename)
    dt = datetime.datetime.utcfromtimestamp(mtime)
    return dt.replace(tzinfo=pytz.utc)

def is_date_type(cls):
    """Return True if the class is a date type."""
    if not isinstance(cls, type):
        return False
    return issubclass(cls, date) and not issubclass(cls, datetime)

def prevmonday(num):
    """
    Return unix SECOND timestamp of "num" mondays ago
    """
    today = get_today()
    lastmonday = today - timedelta(days=today.weekday(), weeks=num)
    return lastmonday

def clean(some_string, uppercase=False):
    """
    helper to clean up an input string
    """
    if uppercase:
        return some_string.strip().upper()
    else:
        return some_string.strip().lower()

def display_len(text):
    """
    Get the display length of a string. This can differ from the character
    length if the string contains wide characters.
    """
    text = unicodedata.normalize('NFD', text)
    return sum(char_width(char) for char in text)

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

def previous_quarter(d):
    """
    Retrieve the previous quarter for dt
    """
    from django_toolkit.datetime_util import quarter as datetime_quarter
    return quarter( (datetime_quarter(datetime(d.year, d.month, d.day))[0] + timedelta(days=-1)).date() )

def _histplot_bins(column, bins=100):
    """Helper to get bins for histplot."""
    col_min = np.min(column)
    col_max = np.max(column)
    return range(col_min, col_max + 2, max((col_max - col_min) // bins, 1))

def last_modified_date(filename):
    """Last modified timestamp as a UTC datetime"""
    mtime = os.path.getmtime(filename)
    dt = datetime.datetime.utcfromtimestamp(mtime)
    return dt.replace(tzinfo=pytz.utc)

def multidict_to_dict(d):
    """
    Turns a werkzeug.MultiDict or django.MultiValueDict into a dict with
    list values
    :param d: a MultiDict or MultiValueDict instance
    :return: a dict instance
    """
    return dict((k, v[0] if len(v) == 1 else v) for k, v in iterlists(d))

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

def stdout_display():
    """ Print results straight to stdout """
    if sys.version_info[0] == 2:
        yield SmartBuffer(sys.stdout)
    else:
        yield SmartBuffer(sys.stdout.buffer)

def retrieve_by_id(self, id_):
        """Return a JSSObject for the element with ID id_"""
        items_with_id = [item for item in self if item.id == int(id_)]
        if len(items_with_id) == 1:
            return items_with_id[0].retrieve()

async def delete(self):
        """
        Delete this message

        :return: bool
        """
        return await self.bot.delete_message(self.chat.id, self.message_id)

def border(self):
        """Region formed by taking border elements.

        :returns: :class:`jicimagelib.region.Region`
        """

        border_array = self.bitmap - self.inner.bitmap
        return Region(border_array)

def column_names(self, table):
      """An iterable of column names, for a particular table or
      view."""

      table_info = self.execute(
        u'PRAGMA table_info(%s)' % quote(table))
      return (column['name'] for column in table_info)

def stdout_display():
    """ Print results straight to stdout """
    if sys.version_info[0] == 2:
        yield SmartBuffer(sys.stdout)
    else:
        yield SmartBuffer(sys.stdout.buffer)

def get_line_ending(line):
    """Return line ending."""
    non_whitespace_index = len(line.rstrip()) - len(line)
    if not non_whitespace_index:
        return ''
    else:
        return line[non_whitespace_index:]

def Print(x, data, message, **kwargs):  # pylint: disable=invalid-name
  """Call tf.Print.

  Args:
    x: a Tensor.
    data: a list of Tensor
    message: a string
    **kwargs: keyword arguments to tf.Print
  Returns:
    a Tensor which is identical in value to x
  """
  return PrintOperation(x, data, message, **kwargs).outputs[0]

def getScreenDims(self):
        """returns a tuple that contains (screen_width,screen_height)
        """
        width = ale_lib.getScreenWidth(self.obj)
        height = ale_lib.getScreenHeight(self.obj)
        return (width,height)

def pprint(self, seconds):
        """
        Pretty Prints seconds as Hours:Minutes:Seconds.MilliSeconds

        :param seconds:  The time in seconds.
        """
        return ("%d:%02d:%02d.%03d", reduce(lambda ll, b: divmod(ll[0], b) + ll[1:], [(seconds * 1000,), 1000, 60, 60]))

def get_keys_from_shelve(file_name, file_location):
    """
    Function to retreive all keys in a shelve
    Args:
        file_name: Shelve storage file name
        file_location: The location of the file, derive from the os module

    Returns:
        a list of the keys

    """
    temp_list = list()
    file = __os.path.join(file_location, file_name)
    shelve_store = __shelve.open(file)
    for key in shelve_store:
        temp_list.append(key)
    shelve_store.close()
    return temp_list

def table_top_abs(self):
        """Returns the absolute position of table top"""
        table_height = np.array([0, 0, self.table_full_size[2]])
        return string_to_array(self.floor.get("pos")) + table_height

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

def Output(self):
    """Output all sections of the page."""
    self.Open()
    self.Header()
    self.Body()
    self.Footer()

def qsize(self):
        """Return the approximate size of the queue (not reliable!)."""
        self.mutex.acquire()
        n = self._qsize()
        self.mutex.release()
        return n

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

def max(self):
        """
        The maximum integer value of a value-set. It is only defined when there is exactly one region.

        :return: A integer that represents the maximum integer value of this value-set.
        :rtype:  int
        """

        if len(self.regions) != 1:
            raise ClaripyVSAOperationError("'max()' onlly works on single-region value-sets.")

        return self.get_si(next(iter(self.regions))).max

def group(data, num):
    """ Split data into chunks of num chars each """
    return [data[i:i+num] for i in range(0, len(data), num)]

def check_str(obj):
        """ Returns a string for various input types """
        if isinstance(obj, str):
            return obj
        if isinstance(obj, float):
            return str(int(obj))
        else:
            return str(obj)

def _split_str(s, n):
    """
    split string into list of strings by specified number.
    """
    length = len(s)
    return [s[i:i + n] for i in range(0, length, n)]

def int2str(num, radix=10, alphabet=BASE85):
    """helper function for quick base conversions from integers to strings"""
    return NumConv(radix, alphabet).int2str(num)

def split_into_sentences(s):
  """Split text into list of sentences."""
  s = re.sub(r"\s+", " ", s)
  s = re.sub(r"[\\.\\?\\!]", "\n", s)
  return s.split("\n")

def security(self):
        """Print security object information for a pdf document"""
        return {k: v for i in self.pdf.resolvedObjects.items() for k, v in i[1].items()}

def rollback(name, database=None, directory=None, verbose=None):
    """Rollback a migration with given name."""
    router = get_router(directory, database, verbose)
    router.rollback(name)

def get_page_text(self, page):
        """
        Downloads and returns the full text of a particular page
        in the document.
        """
        url = self.get_page_text_url(page)
        return self._get_url(url)

def get_python():
    """Determine the path to the virtualenv python"""
    if sys.platform == 'win32':
        python = path.join(VE_ROOT, 'Scripts', 'python.exe')
    else:
        python = path.join(VE_ROOT, 'bin', 'python')
    return python

def _get_closest_week(self, metric_date):
        """
        Gets the closest monday to the date provided.
        """
        #find the offset to the closest monday
        days_after_monday = metric_date.isoweekday() - 1

        return metric_date - datetime.timedelta(days=days_after_monday)

def one_hot(x, size, dtype=np.float32):
  """Make a n+1 dim one-hot array from n dim int-categorical array."""
  return np.array(x[..., np.newaxis] == np.arange(size), dtype)

def server(port):
    """Start the Django dev server."""
    args = ['python', 'manage.py', 'runserver']
    if port:
        args.append(port)
    run.main(args)

def get_grid_spatial_dimensions(self, variable):
        """Returns (width, height) for the given variable"""

        data = self.open_dataset(self.service).variables[variable.variable]
        dimensions = list(data.dimensions)
        return data.shape[dimensions.index(variable.x_dimension)], data.shape[dimensions.index(variable.y_dimension)]

def _force_float(v):
    """ Converts given argument to float. On fail logs warning and returns 0.0.

    Args:
        v (any): value to convert to float

    Returns:
        float: converted v or 0.0 if conversion failed.

    """
    try:
        return float(v)
    except Exception as exc:
        return float('nan')
        logger.warning('Failed to convert {} to float with {} error. Using 0 instead.'.format(v, exc))

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

def min_or_none(val1, val2):
    """Returns min(val1, val2) returning None only if both values are None"""
    return min(val1, val2, key=lambda x: sys.maxint if x is None else x)

def end_index(self):
        """
        Returns the 1-based index of the last object on this page,
        relative to total objects found (hits).
        """
        return ((self.number - 1) * self.paginator.per_page +
            len(self.object_list))

def end_of_history(event):
    """
    Move to the end of the input history, i.e., the line currently being entered.
    """
    event.current_buffer.history_forward(count=10**100)
    buff = event.current_buffer
    buff.go_to_history(len(buff._working_lines) - 1)

def objectproxy_realaddress(obj):
    """
    Obtain a real address as an integer from an objectproxy.
    """
    voidp = QROOT.TPython.ObjectProxy_AsVoidPtr(obj)
    return C.addressof(C.c_char.from_buffer(voidp))

def _dotify(cls, data):
    """Add dots."""
    return ''.join(char if char in cls.PRINTABLE_DATA else '.' for char in data)

def size(dtype):
  """Returns the number of bytes to represent this `dtype`."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'size'):
    return dtype.size
  return np.dtype(dtype).itemsize

def dot_v2(vec1, vec2):
    """Return the dot product of two vectors"""

    return vec1.x * vec2.x + vec1.y * vec2.y

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

def dot_v3(v, w):
    """Return the dotproduct of two vectors."""

    return sum([x * y for x, y in zip(v, w)])

def get_page_text(self, page):
        """
        Downloads and returns the full text of a particular page
        in the document.
        """
        url = self.get_page_text_url(page)
        return self._get_url(url)

def drop_all_tables(self):
        """Drop all tables in the database"""
        for table_name in self.table_names():
            self.execute_sql("DROP TABLE %s" % table_name)
        self.connection.commit()

def get_time(filename):
	"""
	Get the modified time for a file as a datetime instance
	"""
	ts = os.stat(filename).st_mtime
	return datetime.datetime.utcfromtimestamp(ts)

def check_str(obj):
        """ Returns a string for various input types """
        if isinstance(obj, str):
            return obj
        if isinstance(obj, float):
            return str(int(obj))
        else:
            return str(obj)

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

def dimensions(self):
        """Get width and height of a PDF"""
        size = self.pdf.getPage(0).mediaBox
        return {'w': float(size[2]), 'h': float(size[3])}

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

def batch_get_item(self, batch_list):
        """
        Return a set of attributes for a multiple items in
        multiple tables using their primary keys.

        :type batch_list: :class:`boto.dynamodb.batch.BatchList`
        :param batch_list: A BatchList object which consists of a
            list of :class:`boto.dynamoddb.batch.Batch` objects.
            Each Batch object contains the information about one
            batch of objects that you wish to retrieve in this
            request.
        """
        request_items = self.dynamize_request_items(batch_list)
        return self.layer1.batch_get_item(request_items,
                                          object_hook=item_object_hook)

def fetch_event(urls):
    """
    This parallel fetcher uses gevent one uses gevent
    """
    rs = (grequests.get(u) for u in urls)
    return [content.json() for content in grequests.map(rs)]

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

def fetch_event(urls):
    """
    This parallel fetcher uses gevent one uses gevent
    """
    rs = (grequests.get(u) for u in urls)
    return [content.json() for content in grequests.map(rs)]

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

def fetch_event(urls):
    """
    This parallel fetcher uses gevent one uses gevent
    """
    rs = (grequests.get(u) for u in urls)
    return [content.json() for content in grequests.map(rs)]

def _bytes_to_json(value):
    """Coerce 'value' to an JSON-compatible representation."""
    if isinstance(value, bytes):
        value = base64.standard_b64encode(value).decode("ascii")
    return value

def printheader(h=None):
    """Print the header for the CSV table."""
    writer = csv.writer(sys.stdout)
    writer.writerow(header_fields(h))

def _dt_to_epoch(dt):
        """Convert datetime to epoch seconds."""
        try:
            epoch = dt.timestamp()
        except AttributeError:  # py2
            epoch = (dt - datetime(1970, 1, 1)).total_seconds()
        return epoch

def go_to_parent_directory(self):
        """Go to parent directory"""
        self.chdir(osp.abspath(osp.join(getcwd_or_home(), os.pardir)))

def page_guiref(arg_s=None):
    """Show a basic reference about the GUI Console."""
    from IPython.core import page
    page.page(gui_reference, auto_html=True)

def average_gradient(data, *kwargs):
    """ Compute average gradient norm of an image
    """
    return np.average(np.array(np.gradient(data))**2)

def reraise(error):
    """Re-raises the error that was processed by prepare_for_reraise earlier."""
    if hasattr(error, "_type_"):
        six.reraise(type(error), error, error._traceback)
    raise error

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

def _escape(s):
    """ Helper method that escapes parameters to a SQL query. """
    e = s
    e = e.replace('\\', '\\\\')
    e = e.replace('\n', '\\n')
    e = e.replace('\r', '\\r')
    e = e.replace("'", "\\'")
    e = e.replace('"', '\\"')
    return e

def normalize_job_id(job_id):
	"""
	Convert a value to a job id.

	:param job_id: Value to convert.
	:type job_id: int, str
	:return: The job id.
	:rtype: :py:class:`uuid.UUID`
	"""
	if not isinstance(job_id, uuid.UUID):
		job_id = uuid.UUID(job_id)
	return job_id

def _escape(s):
    """ Helper method that escapes parameters to a SQL query. """
    e = s
    e = e.replace('\\', '\\\\')
    e = e.replace('\n', '\\n')
    e = e.replace('\r', '\\r')
    e = e.replace("'", "\\'")
    e = e.replace('"', '\\"')
    return e

def ungzip_data(input_data):
    """Return a string of data after gzip decoding

    :param the input gziped data
    :return  the gzip decoded data"""
    buf = StringIO(input_data)
    f = gzip.GzipFile(fileobj=buf)
    return f

def euclidean(x, y):
    """Standard euclidean distance.

    ..math::
        D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2
    return np.sqrt(result)

def runcode(code):
	"""Run the given code line by line with printing, as list of lines, and return variable 'ans'."""
	for line in code:
		print('# '+line)
		exec(line,globals())
	print('# return ans')
	return ans

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

def str2bytes(x):
  """Convert input argument to bytes"""
  if type(x) is bytes:
    return x
  elif type(x) is str:
    return bytes([ ord(i) for i in x ])
  else:
    return str2bytes(str(x))

def main(argv=sys.argv, stream=sys.stderr):
    """Entry point for ``tappy`` command."""
    args = parse_args(argv)
    suite = build_suite(args)
    runner = unittest.TextTestRunner(verbosity=args.verbose, stream=stream)
    result = runner.run(suite)

    return get_status(result)

def activate(self):
        """Store ipython references in the __builtin__ namespace."""

        add_builtin = self.add_builtin
        for name, func in self.auto_builtins.iteritems():
            add_builtin(name, func)

def _loop_timeout_cb(self, main_loop):
        """Stops the loop after the time specified in the `loop` call.
        """
        self._anything_done = True
        logger.debug("_loop_timeout_cb() called")
        main_loop.quit()

def save_config_value(request, response, key, value):
    """Sets value of key `key` to `value` in both session and cookies."""
    request.session[key] = value
    response.set_cookie(key, value, expires=one_year_from_now())
    return response

def call_and_exit(self, cmd, shell=True):
        """Run the *cmd* and exit with the proper exit code."""
        sys.exit(subprocess.call(cmd, shell=shell))

def sets_are_rooted_compat(one_set, other):
    """treats the 2 sets are sets of taxon IDs on the same (unstated)
    universe of taxon ids.
    Returns True clades implied by each are compatible and False otherwise
    """
    if one_set.issubset(other) or other.issubset(one_set):
        return True
    return not intersection_not_empty(one_set, other)

def ip_address_list(ips):
    """ IP address range validation and expansion. """
    # first, try it as a single IP address
    try:
        return ip_address(ips)
    except ValueError:
        pass
    # then, consider it as an ipaddress.IPv[4|6]Network instance and expand it
    return list(ipaddress.ip_network(u(ips)).hosts())

def lint(args):
    """Run lint checks using flake8."""
    application = get_current_application()
    if not args:
        args = [application.name, 'tests']
    args = ['flake8'] + list(args)
    run.main(args, standalone_mode=False)

def skip_connection_distance(a, b):
    """The distance between two skip-connections."""
    if a[2] != b[2]:
        return 1.0
    len_a = abs(a[1] - a[0])
    len_b = abs(b[1] - b[0])
    return (abs(a[0] - b[0]) + abs(len_a - len_b)) / (max(a[0], b[0]) + max(len_a, len_b))

def get_numbers(s):
    """Extracts all integers from a string an return them in a list"""

    result = map(int, re.findall(r'[0-9]+', unicode(s)))
    return result + [1] * (2 - len(result))

def parallel(processes, threads):
    """
    execute jobs in processes using N threads
    """
    pool = multithread(threads)
    pool.map(run_process, processes)
    pool.close()
    pool.join()

def search_for_tweets_about(user_id, params):
    """ Search twitter API """
    url = "https://api.twitter.com/1.1/search/tweets.json"
    response = make_twitter_request(url, user_id, params)
    return process_tweets(response.json()["statuses"])

def fmt_subst(regex, subst):
    """Replace regex with string."""
    return lambda text: re.sub(regex, subst, text) if text else text

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

def chunked_list(_list, _chunk_size=50):
    """
    Break lists into small lists for processing:w
    """
    for i in range(0, len(_list), _chunk_size):
        yield _list[i:i + _chunk_size]

def is_running(self):
        """Returns a bool determining if the process is in a running state or
        not

        :rtype: bool

        """
        return self.state in [self.STATE_IDLE, self.STATE_ACTIVE,
                              self.STATE_SLEEPING]

def is_punctuation(text):
    """Check if given string is a punctuation"""
    return not (text.lower() in config.AVRO_VOWELS or
                text.lower() in config.AVRO_CONSONANTS)

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

def time_func(func, name, *args, **kwargs):
    """ call a func with args and kwargs, print name of func and how
    long it took. """
    tic = time.time()
    out = func(*args, **kwargs)
    toc = time.time()
    print('%s took %0.2f seconds' % (name, toc - tic))
    return out

def struct2dict(struct):
    """convert a ctypes structure to a dictionary"""
    return {x: getattr(struct, x) for x in dict(struct._fields_).keys()}

def commajoin_as_strings(iterable):
    """ Join the given iterable with ',' """
    return _(u',').join((six.text_type(i) for i in iterable))

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

def filter_set(input, **params):
    """
    Apply WHERE filter to input dataset
    :param input:
    :param params:
    :return: filtered data
    """
    PARAM_WHERE = 'where'

    return Converter.df2list(pd.DataFrame.from_records(input).query(params.get(PARAM_WHERE)))

def del_Unnamed(df):
    """
    Deletes all the unnamed columns

    :param df: pandas dataframe
    """
    cols_del=[c for c in df.columns if 'Unnamed' in c]
    return df.drop(cols_del,axis=1)

def filter_list_by_indices(lst, indices):
    """Return a modified list containing only the indices indicated.

    Args:
        lst: Original list of values
        indices: List of indices to keep from the original list

    Returns:
        list: Filtered list of values

    """
    return [x for i, x in enumerate(lst) if i in indices]

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

def filter_list_by_indices(lst, indices):
    """Return a modified list containing only the indices indicated.

    Args:
        lst: Original list of values
        indices: List of indices to keep from the original list

    Returns:
        list: Filtered list of values

    """
    return [x for i, x in enumerate(lst) if i in indices]

def properties(self):
        """All compartment properties as a dict."""
        properties = {'id': self._id}
        if self._name is not None:
            properties['name'] = self._name

        return properties

def unique_list(lst):
    """Make a list unique, retaining order of initial appearance."""
    uniq = []
    for item in lst:
        if item not in uniq:
            uniq.append(item)
    return uniq

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

def locate(command, on):
    """Locate the command's man page."""
    location = find_page_location(command, on)
    click.echo(location)

def get_obj(ref):
    """Get object from string reference."""
    oid = int(ref)
    return server.id2ref.get(oid) or server.id2obj[oid]

def find_geom(geom, geoms):
    """
    Returns the index of a geometry in a list of geometries avoiding
    expensive equality checks of `in` operator.
    """
    for i, g in enumerate(geoms):
        if g is geom:
            return i

def security(self):
        """Print security object information for a pdf document"""
        return {k: v for i in self.pdf.resolvedObjects.items() for k, v in i[1].items()}

def calc_list_average(l):
    """
    Calculates the average value of a list of numbers
    Returns a float
    """
    total = 0.0
    for value in l:
        total += value
    return total / len(l)

def growthfromrange(rangegrowth, startdate, enddate):
    """
    Annual growth given growth from start date to end date.
    """
    _yrs = (pd.Timestamp(enddate) - pd.Timestamp(startdate)).total_seconds() /\
            dt.timedelta(365.25).total_seconds()
    return yrlygrowth(rangegrowth, _yrs)

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

def _time_to_json(value):
    """Coerce 'value' to an JSON-compatible representation."""
    if isinstance(value, datetime.time):
        value = value.isoformat()
    return value

def is_in(self, search_list, pair):
        """
        If pair is in search_list, return the index. Otherwise return -1
        """
        index = -1
        for nr, i in enumerate(search_list):
            if(np.all(i == pair)):
                return nr
        return index

def get_longest_orf(orfs):
    """Find longest ORF from the given list of ORFs."""
    sorted_orf = sorted(orfs, key=lambda x: len(x['sequence']), reverse=True)[0]
    return sorted_orf

def copy(obj):
    def copy(self):
        """
        Copy self to a new object.
        """
        from copy import deepcopy

        return deepcopy(self)
    obj.copy = copy
    return obj

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

def get_input(input_func, input_str):
    """
    Get input from the user given an input function and an input string
    """
    val = input_func("Please enter your {0}: ".format(input_str))
    while not val or not len(val.strip()):
        val = input_func("You didn't enter a valid {0}, please try again: ".format(input_str))
    return val

def get_longest_orf(orfs):
    """Find longest ORF from the given list of ORFs."""
    sorted_orf = sorted(orfs, key=lambda x: len(x['sequence']), reverse=True)[0]
    return sorted_orf

def add_to_toolbar(self, toolbar, widget):
        """Add widget actions to toolbar"""
        actions = widget.toolbar_actions
        if actions is not None:
            add_actions(toolbar, actions)

def distL1(x1,y1,x2,y2):
    """Compute the L1-norm (Manhattan) distance between two points.

    The distance is rounded to the closest integer, for compatibility
    with the TSPLIB convention.

    The two points are located on coordinates (x1,y1) and (x2,y2),
    sent as parameters"""
    return int(abs(x2-x1) + abs(y2-y1)+.5)

def format_exception(e):
    """Returns a string containing the type and text of the exception.

    """
    from .utils.printing import fill
    return '\n'.join(fill(line) for line in traceback.format_exception_only(type(e), e))

def find(self, node, path):
        """Wrapper for lxml`s find."""

        return node.find(path, namespaces=self.namespaces)

def deprecated(operation=None):
    """
    Mark an operation deprecated.
    """
    def inner(o):
        o.deprecated = True
        return o
    return inner(operation) if operation else inner

def _getTypename(self, defn):
        """ Returns the SQL typename required to store the given FieldDefinition """
        return 'REAL' if defn.type.float or 'TIME' in defn.type.name or defn.dntoeu else 'INTEGER'

def es_field_sort(fld_name):
    """ Used with lambda to sort fields """
    parts = fld_name.split(".")
    if "_" not in parts[-1]:
        parts[-1] = "_" + parts[-1]
    return ".".join(parts)

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

def _make_proxy_property(bind_attr, attr_name):
    def proxy_property(self):
        """
        proxy
        """
        bind = getattr(self, bind_attr)
        return getattr(bind, attr_name)
    return property(proxy_property)

def find_ge(a, x):
    """Find leftmost item greater than or equal to x."""
    i = bs.bisect_left(a, x)
    if i != len(a): return i
    raise ValueError

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

def exp_fit_fun(x, a, tau, c):
    """Function used to fit the exponential decay."""
    # pylint: disable=invalid-name
    return a * np.exp(-x / tau) + c

def py(self, output):
        """Output data as a nicely-formatted python data structure"""
        import pprint
        pprint.pprint(output, stream=self.outfile)

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

def slugify(string):
    """
    Removes non-alpha characters, and converts spaces to hyphens. Useful for making file names.


    Source: http://stackoverflow.com/questions/5574042/string-slugification-in-python
    """
    string = re.sub('[^\w .-]', '', string)
    string = string.replace(" ", "-")
    return string

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

def uint32_to_uint8(cls, img):
        """
        Cast uint32 RGB image to 4 uint8 channels.
        """
        return np.flipud(img.view(dtype=np.uint8).reshape(img.shape + (4,)))

def getBuffer(x):
    """
    Copy @x into a (modifiable) ctypes byte array
    """
    b = bytes(x)
    return (c_ubyte * len(b)).from_buffer_copy(bytes(x))

def OnMove(self, event):
        """Main window move event"""

        # Store window position in config
        position = self.main_window.GetScreenPositionTuple()

        config["window_position"] = repr(position)

def _stdout_raw(self, s):
        """Writes the string to stdout"""
        print(s, end='', file=sys.stdout)
        sys.stdout.flush()

def fixed(ctx, number, decimals=2, no_commas=False):
    """
    Formats the given number in decimal format using a period and commas
    """
    value = _round(ctx, number, decimals)
    format_str = '{:f}' if no_commas else '{:,f}'
    return format_str.format(value)

def list_backends(_):
    """List all available backends."""
    backends = [b.__name__ for b in available_backends()]
    print('\n'.join(backends))

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

def dict_from_object(obj: object):
    """Convert a object into dictionary with all of its readable attributes."""

    # If object is a dict instance, no need to convert.
    return (obj if isinstance(obj, dict)
            else {attr: getattr(obj, attr)
                  for attr in dir(obj) if not attr.startswith('_')})

def _convert_latitude(self, latitude):
        """Convert from latitude to the y position in overall map."""
        return int((180 - (180 / pi * log(tan(
            pi / 4 + latitude * pi / 360)))) * (2 ** self._zoom) * self._size / 360)

def get_java_path():
  """Get the path of java executable"""
  java_home = os.environ.get("JAVA_HOME")
  return os.path.join(java_home, BIN_DIR, "java")

def frombits(cls, bits):
        """Series from binary string arguments."""
        return cls.frombitsets(map(cls.BitSet.frombits, bits))

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

def get_kind(self, value):
        """Return the kind (type) of the attribute"""
        if isinstance(value, float):
            return 'f'
        elif isinstance(value, int):
            return 'i'
        else:
            raise ValueError("Only integer or floating point values can be stored.")

def table_top_abs(self):
        """Returns the absolute position of table top"""
        table_height = np.array([0, 0, self.table_full_size[2]])
        return string_to_array(self.floor.get("pos")) + table_height

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

def euclidean(c1, c2):
    """Square of the euclidean distance"""
    diffs = ((i - j) for i, j in zip(c1, c2))
    return sum(x * x for x in diffs)

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

def inh(table):
    """
    inverse hyperbolic sine transformation
    """
    t = []
    for i in table:
        t.append(np.ndarray.tolist(np.arcsinh(i)))
    return t

def path_to_list(pathstr):
    """Conver a path string to a list of path elements."""
    return [elem for elem in pathstr.split(os.path.pathsep) if elem]

def exit_if_missing_graphviz(self):
        """
        Detect the presence of the dot utility to make a png graph.
        """
        (out, err) = utils.capture_shell("which dot")

        if "dot" not in out:
            ui.error(c.MESSAGES["dot_missing"])

def __grid_widgets(self):
        """Places all the child widgets in the appropriate positions."""
        scrollbar_column = 0 if self.__compound is tk.LEFT else 2
        self._canvas.grid(row=0, column=1, sticky="nswe")
        self._scrollbar.grid(row=0, column=scrollbar_column, sticky="ns")

def _uniqueid(n=30):
    """Return a unique string with length n.

    :parameter int N: number of character in the uniqueid
    :return: the uniqueid
    :rtype: str
    """
    return ''.join(random.SystemRandom().choice(
                   string.ascii_uppercase + string.ascii_lowercase)
                   for _ in range(n))

def plot(self):
        """Plot the empirical histogram versus best-fit distribution's PDF."""
        plt.plot(self.bin_edges, self.hist, self.bin_edges, self.best_pdf)

def to_dotfile(self):
        """ Writes a DOT graphviz file of the domain structure, and returns the filename"""
        domain = self.get_domain()
        filename = "%s.dot" % (self.__class__.__name__)
        nx.write_dot(domain, filename)
        return filename

def write_line(self, line, count=1):
        """writes the line and count newlines after the line"""
        self.write(line)
        self.write_newlines(count)

def _gaps_from(intervals):
    """
    From a list of intervals extract
    a list of sorted gaps in the form of [(g,i)]
    where g is the size of the ith gap.
    """
    sliding_window = zip(intervals, intervals[1:])
    gaps = [b[0] - a[1] for a, b in sliding_window]
    return gaps

def _write_json(file, contents):
    """Write a dict to a JSON file."""
    with open(file, 'w') as f:
        return json.dump(contents, f, indent=2, sort_keys=True)

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

def write_str2file(pathname, astr):
    """writes a string to file"""
    fname = pathname
    fhandle = open(fname, 'wb')
    fhandle.write(astr)
    fhandle.close()

def ln_norm(x, mu, sigma=1.0):
    """ Natural log of scipy norm function truncated at zero """
    return np.log(stats.norm(loc=mu, scale=sigma).pdf(x))

def html(header_rows):
    """
    Convert a list of tuples describing a table into a HTML string
    """
    name = 'table%d' % next(tablecounter)
    return HtmlTable([map(str, row) for row in header_rows], name).render()

def _get_random_id():
    """ Get a random (i.e., unique) string identifier"""
    symbols = string.ascii_uppercase + string.ascii_lowercase + string.digits
    return ''.join(random.choice(symbols) for _ in range(15))

def add_header(self, name, value):
        """ Add an additional response header, not removing duplicates. """
        self._headers.setdefault(_hkey(name), []).append(_hval(value))

def runiform(lower, upper, size=None):
    """
    Random uniform variates.
    """
    return np.random.uniform(lower, upper, size)

def get_file_name(url):
  """Returns file name of file at given url."""
  return os.path.basename(urllib.parse.urlparse(url).path) or 'unknown_name'

def normal_noise(points):
    """Init a noise variable."""
    return np.random.rand(1) * np.random.randn(points, 1) \
        + random.sample([2, -2], 1)

def test():
    """Test for ReverseDNS class"""
    dns = ReverseDNS()

    print(dns.lookup('192.168.0.1'))
    print(dns.lookup('8.8.8.8'))

    # Test cache
    print(dns.lookup('8.8.8.8'))

def money(min=0, max=10):
    """Return a str of decimal with two digits after a decimal mark."""
    value = random.choice(range(min * 100, max * 100))
    return "%1.2f" % (float(value) / 100)

def find_le(a, x):
    """Find rightmost value less than or equal to x."""
    i = bs.bisect_right(a, x)
    if i: return i - 1
    raise ValueError

def random_alphanum(length):
    """
    Return a random string of ASCII letters and digits.

    :param int length: The length of string to return
    :returns: A random string
    :rtype: str
    """
    charset = string.ascii_letters + string.digits
    return random_string(length, charset)

def na_if(series, *values):
    """
    If values in a series match a specified value, change them to `np.nan`.

    Args:
        series: Series or vector, often symbolic.
        *values: Value(s) to convert to `np.nan` in the series.
    """

    series = pd.Series(series)
    series[series.isin(values)] = np.nan
    return series

def get_available_gpus():
  """
  Returns a list of string names of all available GPUs
  """
  local_device_protos = device_lib.list_local_devices()
  return [x.name for x in local_device_protos if x.device_type == 'GPU']

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

def circ_permutation(items):
    """Calculate the circular permutation for a given list of items."""
    permutations = []
    for i in range(len(items)):
        permutations.append(items[i:] + items[:i])
    return permutations

def _is_name_used_as_variadic(name, variadics):
    """Check if the given name is used as a variadic argument."""
    return any(
        variadic.value == name or variadic.value.parent_of(name)
        for variadic in variadics
    )

def ancestors(self, node):
        """Returns set of the ancestors of a node as DAGNodes."""
        if isinstance(node, int):
            warnings.warn('Calling ancestors() with a node id is deprecated,'
                          ' use a DAGNode instead',
                          DeprecationWarning, 2)
            node = self._id_to_node[node]

        return nx.ancestors(self._multi_graph, node)

def _trim(image):
    """Trim a PIL image and remove white space."""
    background = PIL.Image.new(image.mode, image.size, image.getpixel((0, 0)))
    diff = PIL.ImageChops.difference(image, background)
    diff = PIL.ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        image = image.crop(bbox)
    return image

def ReadTif(tifFile):
        """Reads a tif file to a 2D NumPy array"""
        img = Image.open(tifFile)
        img = np.array(img)
        return img

def convolve_gaussian_2d(image, gaussian_kernel_1d):
    """Convolve 2d gaussian."""
    result = scipy.ndimage.filters.correlate1d(
        image, gaussian_kernel_1d, axis=0)
    result = scipy.ndimage.filters.correlate1d(
        result, gaussian_kernel_1d, axis=1)
    return result

def get_attr(self, method_name):
        """Get attribute from the target object"""
        return self.attrs.get(method_name) or self.get_callable_attr(method_name)

def get_colors(img):
    """
    Returns a list of all the image's colors.
    """
    w, h = img.size
    return [color[:3] for count, color in img.convert('RGB').getcolors(w * h)]

def osx_clipboard_get():
    """ Get the clipboard's text on OS X.
    """
    p = subprocess.Popen(['pbpaste', '-Prefer', 'ascii'],
        stdout=subprocess.PIPE)
    text, stderr = p.communicate()
    # Text comes in with old Mac \r line endings. Change them to \n.
    text = text.replace('\r', '\n')
    return text

def show(data, negate=False):
    """Show the stretched data.
    """
    from PIL import Image as pil
    data = np.array((data - data.min()) * 255.0 /
                    (data.max() - data.min()), np.uint8)
    if negate:
        data = 255 - data
    img = pil.fromarray(data)
    img.show()

def index(self, elem):
        """Find the index of elem in the reversed iterator."""
        return _coconut.len(self._iter) - self._iter.index(elem) - 1

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

def get_active_ajax_datatable(self):
        """ Returns a single datatable according to the hint GET variable from an AJAX request. """
        data = getattr(self.request, self.request.method)
        datatables_dict = self.get_datatables(only=data['datatable'])
        return list(datatables_dict.values())[0]

def filter_contour(imageFile, opFile):
    """ convert an image by applying a contour """
    im = Image.open(imageFile)
    im1 = im.filter(ImageFilter.CONTOUR)
    im1.save(opFile)

def last_modified_time(path):
    """
    Get the last modified time of path as a Timestamp.
    """
    return pd.Timestamp(os.path.getmtime(path), unit='s', tz='UTC')

def __hash__(self):
        """Return ``hash(self)``."""
        return hash((type(self), self.domain, self.range, self.partition))

def get_parent_folder_name(file_path):
    """Finds parent folder of file

    :param file_path: path
    :return: Name of folder container
    """
    return os.path.split(os.path.split(os.path.abspath(file_path))[0])[-1]

def normalize_vector(v):
    """Take a vector and return the normalized vector
    :param v: a vector v
    :returns : normalized vector v
    """
    norm = np.linalg.norm(v)
    return v/norm if not norm == 0 else v

def get_geoip(ip):
    """Lookup country for IP address."""
    reader = geolite2.reader()
    ip_data = reader.get(ip) or {}
    return ip_data.get('country', {}).get('iso_code')

def other_ind(self):
        """last row or column of square A"""
        return np.full(self.n_min, self.size - 1, dtype=np.int)

def out_shape_from_array(arr):
    """Get the output shape from an array."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.shape
    else:
        return (arr.shape[1],)

def remove_index(self):
        """Remove Elasticsearch index associated to the campaign"""
        self.index_client.close(self.index_name)
        self.index_client.delete(self.index_name)

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

def is_in(self, search_list, pair):
        """
        If pair is in search_list, return the index. Otherwise return -1
        """
        index = -1
        for nr, i in enumerate(search_list):
            if(np.all(i == pair)):
                return nr
        return index

def find_start_point(self):
        """
        Find the first location in our array that is not empty
        """
        for i, row in enumerate(self.data):
            for j, _ in enumerate(row):
                if self.data[i, j] != 0:  # or not np.isfinite(self.data[i,j]):
                    return i, j

def end_block(self):
        """Ends an indentation block, leaving an empty line afterwards"""
        self.current_indent -= 1

        # If we did not add a new line automatically yet, now it's the time!
        if not self.auto_added_line:
            self.writeln()
            self.auto_added_line = True

def get_inputs_from_cm(index, cm):
    """Return indices of inputs to the node with the given index."""
    return tuple(i for i in range(cm.shape[0]) if cm[i][index])

def input(self, prompt, default=None, show_default=True):
        """Provide a command prompt."""
        return click.prompt(prompt, default=default, show_default=show_default)

def last_day(year=_year, month=_month):
    """
    get the current month's last day
    :param year:  default to current year
    :param month:  default to current month
    :return: month's last day
    """
    last_day = calendar.monthrange(year, month)[1]
    return datetime.date(year=year, month=month, day=last_day)

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

def other_ind(self):
        """last row or column of square A"""
        return np.full(self.n_min, self.size - 1, dtype=np.int)

def lin_interp(x, rangeX, rangeY):
    """
    Interpolate linearly variable x in rangeX onto rangeY.
    """
    s = (x - rangeX[0]) / mag(rangeX[1] - rangeX[0])
    y = rangeY[0] * (1 - s) + rangeY[1] * s
    return y

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

def close( self ):
        """
        Close the db and release memory
        """
        if self.db is not None:
            self.db.commit()
            self.db.close()
            self.db = None

        return

def get_last(self, table=None):
        """Just the last entry."""
        if table is None: table = self.main_table
        query = 'SELECT * FROM "%s" ORDER BY ROWID DESC LIMIT 1;' % table
        return self.own_cursor.execute(query).fetchone()

def invertDictMapping(d):
    """ Invert mapping of dictionary (i.e. map values to list of keys) """
    inv_map = {}
    for k, v in d.items():
        inv_map[v] = inv_map.get(v, [])
        inv_map[v].append(k)
    return inv_map

def __len__(self):
        """Return total data length of the list and its headers."""
        return self.chunk_length() + len(self.type) + len(self.header) + 4

def get_iter_string_reader(stdin):
    """ return an iterator that returns a chunk of a string every time it is
    called.  notice that even though bufsize_type might be line buffered, we're
    not doing any line buffering here.  that's because our StreamBufferer
    handles all buffering.  we just need to return a reasonable-sized chunk. """
    bufsize = 1024
    iter_str = (stdin[i:i + bufsize] for i in range(0, len(stdin), bufsize))
    return get_iter_chunk_reader(iter_str)

def difference(ydata1, ydata2):
    """

    Returns the number you should add to ydata1 to make it line up with ydata2

    """

    y1 = _n.array(ydata1)
    y2 = _n.array(ydata2)

    return(sum(y2-y1)/len(ydata1))

def _ioctl (self, func, args):
        """Call ioctl() with given parameters."""
        import fcntl
        return fcntl.ioctl(self.sockfd.fileno(), func, args)

def _index_ordering(redshift_list):
        """

        :param redshift_list: list of redshifts
        :return: indexes in acending order to be evaluated (from z=0 to z=z_source)
        """
        redshift_list = np.array(redshift_list)
        sort_index = np.argsort(redshift_list)
        return sort_index

def _ipv4_text_to_int(self, ip_text):
        """convert ip v4 string to integer."""
        if ip_text is None:
            return None
        assert isinstance(ip_text, str)
        return struct.unpack('!I', addrconv.ipv4.text_to_bin(ip_text))[0]

def _lookup_parent(self, cls):
        """Lookup a transitive parent object that is an instance
            of a given class."""
        codeobj = self.parent
        while codeobj is not None and not isinstance(codeobj, cls):
            codeobj = codeobj.parent
        return codeobj

def A(*a):
    """convert iterable object into numpy array"""
    return np.array(a[0]) if len(a)==1 else [np.array(o) for o in a]

def get_week_start_end_day():
    """
    Get the week start date and end date
    """
    t = date.today()
    wd = t.weekday()
    return (t - timedelta(wd), t + timedelta(6 - wd))

def match_files(files, pattern: Pattern):
    """Yields file name if matches a regular expression pattern."""

    for name in files:
        if re.match(pattern, name):
            yield name

def get_memory_usage():
    """Gets RAM memory usage

    :return: MB of memory used by this process
    """
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss
    return mem / (1024 * 1024)

def extract_words(lines):
    """
    Extract from the given iterable of lines the list of words.

    :param lines: an iterable of lines;
    :return: a generator of words of lines.
    """
    for line in lines:
        for word in re.findall(r"\w+", line):
            yield word

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

def reset(self):
		"""
		Resets the iterator to the start.

		Any remaining values in the current iteration are discarded.
		"""
		self.__iterator, self.__saved = itertools.tee(self.__saved)

def get_week_start_end_day():
    """
    Get the week start date and end date
    """
    t = date.today()
    wd = t.weekday()
    return (t - timedelta(wd), t + timedelta(6 - wd))

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

def pop_row(self, idr=None, tags=False):
        """Pops a row, default the last"""
        idr = idr if idr is not None else len(self.body) - 1
        row = self.body.pop(idr)
        return row if tags else [cell.childs[0] for cell in row]

def convert_time_string(date_str):
    """ Change a date string from the format 2018-08-15T23:55:17 into a datetime object """
    dt, _, _ = date_str.partition(".")
    dt = datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S")
    return dt

def find_start_point(self):
        """
        Find the first location in our array that is not empty
        """
        for i, row in enumerate(self.data):
            for j, _ in enumerate(row):
                if self.data[i, j] != 0:  # or not np.isfinite(self.data[i,j]):
                    return i, j

def _nth(arr, n):
    """
    Return the nth value of array

    If it is missing return NaN
    """
    try:
        return arr.iloc[n]
    except (KeyError, IndexError):
        return np.nan

def record_diff(old, new):
    """Return a JSON-compatible structure capable turn the `new` record back
    into the `old` record. The parameters must be structures compatible with
    json.dumps *or* strings compatible with json.loads. Note that by design,
    `old == record_patch(new, record_diff(old, new))`"""
    old, new = _norm_json_params(old, new)
    return json_delta.diff(new, old, verbose=False)

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

def from_json(cls, json_str):
        """Deserialize the object from a JSON string."""
        d = json.loads(json_str)
        return cls.from_dict(d)

def explained_variance(returns, values):
    """ Calculate how much variance in returns do the values explain """
    exp_var = 1 - torch.var(returns - values) / torch.var(returns)
    return exp_var.item()

def dump_json(obj):
    """Dump Python object as JSON string."""
    return simplejson.dumps(obj, ignore_nan=True, default=json_util.default)

def get_free_memory_win():
    """Return current free memory on the machine for windows.

    Warning : this script is really not robust
    Return in MB unit
    """
    stat = MEMORYSTATUSEX()
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return int(stat.ullAvailPhys / 1024 / 1024)

def read_json(location):
    """Open and load JSON from file.

    location (Path): Path to JSON file.
    RETURNS (dict): Loaded JSON content.
    """
    location = ensure_path(location)
    with location.open('r', encoding='utf8') as f:
        return ujson.load(f)

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

def mpl_outside_legend(ax, **kwargs):
    """ Places a legend box outside a matplotlib Axes instance. """
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), **kwargs)

def json(body, charset='utf-8', **kwargs):
    """Takes JSON formatted data, converting it into native Python objects"""
    return json_converter.loads(text(body, charset=charset))

def focusInEvent(self, event):
        """Reimplement Qt method to send focus change notification"""
        self.focus_changed.emit()
        return super(ControlWidget, self).focusInEvent(event)

def exp_fit_fun(x, a, tau, c):
    """Function used to fit the exponential decay."""
    # pylint: disable=invalid-name
    return a * np.exp(-x / tau) + c

def get_prep_value(self, value):
        """Convert JSON object to a string"""
        if self.null and value is None:
            return None
        return json.dumps(value, **self.dump_kwargs)

def setLib(self, lib):
        """ Copy the lib items into our font. """
        for name, item in lib.items():
            self.font.lib[name] = item

def _unjsonify(x, isattributes=False):
    """Convert JSON string to an ordered defaultdict."""
    if isattributes:
        obj = json.loads(x)
        return dict_class(obj)
    return json.loads(x)

def go_to_parent_directory(self):
        """Go to parent directory"""
        self.chdir(osp.abspath(osp.join(getcwd_or_home(), os.pardir)))

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

def export(defn):
    """Decorator to explicitly mark functions that are exposed in a lib."""
    globals()[defn.__name__] = defn
    __all__.append(defn.__name__)
    return defn

def is_identity():
        """Check to see if this matrix is an identity matrix."""
        for index, row in enumerate(self.dta):
            if row[index] == 1:
                for num, element in enumerate(row):
                    if num != index:
                        if element != 0:
                            return False
            else:
                return False

        return True

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

def do_files_exist(filenames):
  """Whether any of the filenames exist."""
  preexisting = [tf.io.gfile.exists(f) for f in filenames]
  return any(preexisting)

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

def signal_handler(signal_name, frame):
    """Quit signal handler."""
    sys.stdout.flush()
    print("\nSIGINT in frame signal received. Quitting...")
    sys.stdout.flush()
    sys.exit(0)

def _gcd_array(X):
    """
    Return the largest real value h such that all elements in x are integer
    multiples of h.
    """
    greatest_common_divisor = 0.0
    for x in X:
        greatest_common_divisor = _gcd(greatest_common_divisor, x)

    return greatest_common_divisor

def _set_widget_background_color(widget, color):
        """
        Changes the base color of a widget (background).
        :param widget: widget to modify
        :param color: the color to apply
        """
        pal = widget.palette()
        pal.setColor(pal.Base, color)
        widget.setPalette(pal)

def dict_hash(dct):
    """Return a hash of the contents of a dictionary"""
    dct_s = json.dumps(dct, sort_keys=True)

    try:
        m = md5(dct_s)
    except TypeError:
        m = md5(dct_s.encode())

    return m.hexdigest()

def one_hot_encoding(input_tensor, num_labels):
    """ One-hot encode labels from input """
    xview = input_tensor.view(-1, 1).to(torch.long)

    onehot = torch.zeros(xview.size(0), num_labels, device=input_tensor.device, dtype=torch.float)
    onehot.scatter_(1, xview, 1)
    return onehot.view(list(input_tensor.shape) + [-1])

def _add_hash(source):
    """Add a leading hash '#' at the beginning of every line in the source."""
    source = '\n'.join('# ' + line.rstrip()
                       for line in source.splitlines())
    return source

def monthly(date=datetime.date.today()):
    """
    Take a date object and return the first day of the month.
    """
    return datetime.date(date.year, date.month, 1)

def compare(self, dn, attr, value):
        """
        Compare the ``attr`` of the entry ``dn`` with given ``value``.

        This is a convenience wrapper for the ldap library's ``compare``
        function that returns a boolean value instead of 1 or 0.
        """
        return self.connection.compare_s(dn, attr, value) == 1

def _one_exists(input_files):
    """
    at least one file must exist for multiqc to run properly
    """
    for f in input_files:
        if os.path.exists(f):
            return True
    return False

def prox_zero(X, step):
    """Proximal operator to project onto zero
    """
    return np.zeros(X.shape, dtype=X.dtype)

def heappop_max(heap):
    """Maxheap version of a heappop."""
    lastelt = heap.pop()    # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        _siftup_max(heap, 0)
        return returnitem
    return lastelt

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

def _heappush_max(heap, item):
    """ why is this not in heapq """
    heap.append(item)
    heapq._siftdown_max(heap, 0, len(heap) - 1)

def on_binop(self, node):    # ('left', 'op', 'right')
        """Binary operator."""
        return op2func(node.op)(self.run(node.left),
                                self.run(node.right))

def disable_wx(self):
        """Disable event loop integration with wxPython.

        This merely sets PyOS_InputHook to NULL.
        """
        if self._apps.has_key(GUI_WX):
            self._apps[GUI_WX]._in_event_loop = False
        self.clear_inputhook()

def _shape(self, df):
        """
        Calculate table chape considering index levels.
        """

        row, col = df.shape
        return row + df.columns.nlevels, col + df.index.nlevels

def print_item_with_children(ac, classes, level):
    """ Print the given item and all children items """
    print_row(ac.id, ac.name, f"{ac.allocation:,.2f}", level)
    print_children_recursively(classes, ac, level + 1)

def timed (log=sys.stderr, limit=2.0):
    """Decorator to run a function with timing info."""
    return lambda func: timeit(func, log, limit)

def random_str(size=10):
    """
    create random string of selected size

    :param size: int, length of the string
    :return: the string
    """
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(size))

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

def _scaleSinglePoint(point, scale=1, convertToInteger=True):
    """
    Scale a single point
    """
    x, y = point
    if convertToInteger:
        return int(round(x * scale)), int(round(y * scale))
    else:
        return (x * scale, y * scale)

def l2_norm(params):
    """Computes l2 norm of params by flattening them into a vector."""
    flattened, _ = flatten(params)
    return np.dot(flattened, flattened)

def poke_array(self, store, name, elemtype, elements, container, visited, _stack):
        """abstract method"""
        raise NotImplementedError

def is_nullable_list(val, vtype):
    """Return True if list contains either values of type `vtype` or None."""
    return (isinstance(val, list) and
            any(isinstance(v, vtype) for v in val) and
            all((isinstance(v, vtype) or v is None) for v in val))

def read_from_file(file_path, encoding="utf-8"):
    """
    Read helper method

    :type file_path: str|unicode
    :type encoding: str|unicode
    :rtype: str|unicode
    """
    with codecs.open(file_path, "r", encoding) as f:
        return f.read()

def filter_none(list_of_points):
    """
    
    :param list_of_points: 
    :return: list_of_points with None's removed
    """
    remove_elementnone = filter(lambda p: p is not None, list_of_points)
    remove_sublistnone = filter(lambda p: not contains_none(p), remove_elementnone)
    return list(remove_sublistnone)

def _prepare_proxy(self, conn):
        """
        Establish tunnel connection early, because otherwise httplib
        would improperly set Host: header to proxy's IP:port.
        """
        conn.set_tunnel(self._proxy_host, self.port, self.proxy_headers)
        conn.connect()

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

def get_extract_value_function(column_identifier):
    """
    returns a function that extracts the value for a column.
    """
    def extract_value(run_result):
        pos = None
        for i, column in enumerate(run_result.columns):
            if column.title == column_identifier:
                pos = i
                break
        if pos is None:
            sys.exit('CPU time missing for task {0}.'.format(run_result.task_id[0]))
        return Util.to_decimal(run_result.values[pos])
    return extract_value

def shape_list(l,shape,dtype):
    """ Shape a list of lists into the appropriate shape and data type """
    return np.array(l, dtype=dtype).reshape(shape)

def count_rows(self, table_name):
        """Return the number of entries in a table by counting them."""
        self.table_must_exist(table_name)
        query = "SELECT COUNT (*) FROM `%s`" % table_name.lower()
        self.own_cursor.execute(query)
        return int(self.own_cursor.fetchone()[0])

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

def url_encode(url):
    """
    Convert special characters using %xx escape.

    :param url: str
    :return: str - encoded url
    """
    if isinstance(url, text_type):
        url = url.encode('utf8')
    return quote(url, ':/%?&=')

def IPYTHON_MAIN():
    """Decide if the Ipython command line is running code."""
    import pkg_resources

    runner_frame = inspect.getouterframes(inspect.currentframe())[-2]
    return (
        getattr(runner_frame, "function", None)
        == pkg_resources.load_entry_point("ipython", "console_scripts", "ipython").__name__
    )

def deduplicate(list_object):
    """Rebuild `list_object` removing duplicated and keeping order"""
    new = []
    for item in list_object:
        if item not in new:
            new.append(item)
    return new

def to_list(self):
        """Convert this confusion matrix into a 2x2 plain list of values."""
        return [[int(self.table.cell_values[0][1]), int(self.table.cell_values[0][2])],
                [int(self.table.cell_values[1][1]), int(self.table.cell_values[1][2])]]

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

def save_dict_to_file(filename, dictionary):
  """Saves dictionary as CSV file."""
  with open(filename, 'w') as f:
    writer = csv.writer(f)
    for k, v in iteritems(dictionary):
      writer.writerow([str(k), str(v)])

def get_uniques(l):
    """ Returns a list with no repeated elements.
    """
    result = []

    for i in l:
        if i not in result:
            result.append(i)

    return result

def log(x):
    """
    Natural logarithm
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.log(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.log(x)

def hdf5_to_dict(filepath, group='/'):
    """load the content of an hdf5 file to a dict.

    # TODO: how to split domain_type_dev : parameter : value ?
    """
    if not h5py.is_hdf5(filepath):
        raise RuntimeError(filepath, 'is not a valid HDF5 file.')

    with h5py.File(filepath, 'r') as handler:
        dic = walk_hdf5_to_dict(handler[group])
    return dic

def get_point_hash(self, point):
        """
        return geohash for given point with self.precision
        :param point: GeoPoint instance
        :return: string
        """
        return geohash.encode(point.latitude, point.longitude, self.precision)

def Load(file):
    """ Loads a model from specified file """
    with open(file, 'rb') as file:
        model = dill.load(file)
        return model

def __deepcopy__(self, memo):
        """Create a deep copy of the node"""
        # noinspection PyArgumentList
        return self.__class__(
            **{key: deepcopy(getattr(self, key), memo) for key in self.keys}
        )

def from_file(filename):
    """
    load an nparray object from a json filename

    @parameter str filename: path to the file
    """
    f = open(filename, 'r')
    j = json.load(f)
    f.close()

    return from_dict(j)

def pickle_save(thing,fname):
    """save something to a pickle file"""
    pickle.dump(thing, open(fname,"wb"),pickle.HIGHEST_PROTOCOL)
    return thing

def from_json_str(cls, json_str):
    """Convert json string representation into class instance.

    Args:
      json_str: json representation as string.

    Returns:
      New instance of the class with data loaded from json string.
    """
    return cls.from_json(json.loads(json_str, cls=JsonDecoder))

def pause():
	"""Tell iTunes to pause"""

	if not settings.platformCompatible():
		return False

	(output, error) = subprocess.Popen(["osascript", "-e", PAUSE], stdout=subprocess.PIPE).communicate()

def Load(file):
    """ Loads a model from specified file """
    with open(file, 'rb') as file:
        model = dill.load(file)
        return model

def camel_case(self, snake_case):
        """ Convert snake case to camel case """
        components = snake_case.split('_')
        return components[0] + "".join(x.title() for x in components[1:])

def get_frame_locals(stepback=0):
    """Returns locals dictionary from a given frame.

    :param int stepback:

    :rtype: dict

    """
    with Frame(stepback=stepback) as frame:
        locals_dict = frame.f_locals

    return locals_dict

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

def fromtimestamp(cls, timestamp):
    """Returns a datetime object of a given timestamp (in local tz)."""
    d = cls.utcfromtimestamp(timestamp)
    return d.astimezone(localtz())

def downsample(array, k):
    """Choose k random elements of array."""
    length = array.shape[0]
    indices = random.sample(xrange(length), k)
    return array[indices]

def lock(self, block=True):
		"""
		Lock connection from being used else where
		"""
		self._locked = True
		return self._lock.acquire(block)

def case_us2mc(x):
    """ underscore to mixed case notation """
    return re.sub(r'_([a-z])', lambda m: (m.group(1).upper()), x)

def lock(self, block=True):
		"""
		Lock connection from being used else where
		"""
		self._locked = True
		return self._lock.acquire(block)

def length(self):
        """Array of vector lengths"""
        return np.sqrt(np.sum(self**2, axis=1)).view(np.ndarray)

def log_request(self, code='-', size='-'):
        """Selectively log an accepted request."""

        if self.server.logRequests:
            BaseHTTPServer.BaseHTTPRequestHandler.log_request(self, code, size)

def do_file_show(client, args):
    """Output file contents to stdout"""
    for src_uri in args.uris:
        client.download_file(src_uri, sys.stdout.buffer)

    return True

def getLinesFromLogFile(stream):
    """
    Returns all lines written to the passed in stream
    """
    stream.flush()
    stream.seek(0)
    lines = stream.readlines()
    return lines

def torecarray(*args, **kwargs):
    """
    Convenient shorthand for ``toarray(*args, **kwargs).view(np.recarray)``.

    """

    import numpy as np
    return toarray(*args, **kwargs).view(np.recarray)

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

def pylog(self, *args, **kwargs):
        """Display all available logging information."""
        printerr(self.name, args, kwargs, traceback.format_exc())

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

def log_no_newline(self, msg):
      """ print the message to the predefined log file without newline """
      self.print2file(self.logfile, False, False, msg)

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

def _configure_logger():
    """Configure the logging module."""
    if not app.debug:
        _configure_logger_for_production(logging.getLogger())
    elif not app.testing:
        _configure_logger_for_debugging(logging.getLogger())

def input_validate_yubikey_secret(data, name='data'):
    """ Input validation for YHSM_YubiKeySecret or string. """
    if isinstance(data, pyhsm.aead_cmd.YHSM_YubiKeySecret):
        data = data.pack()
    return input_validate_str(data, name)

def extract_log_level_from_environment(k, default):
    """Gets the log level from the environment variable."""
    return LOG_LEVELS.get(os.environ.get(k)) or int(os.environ.get(k, default))

def dumped(text, level, indent=2):
    """Put curly brackets round an indented text"""
    return indented("{\n%s\n}" % indented(text, level + 1, indent) or "None", level, indent) + "\n"

def log_no_newline(self, msg):
      """ print the message to the predefined log file without newline """
      self.print2file(self.logfile, False, False, msg)

def add_arrow(self, x1, y1, x2, y2, **kws):
        """add arrow to plot"""
        self.panel.add_arrow(x1, y1, x2, y2, **kws)

def remove_all_handler(self):
        """
        Unlink the file handler association.
        """
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            self._handler_cache.append(handler)

def auth_request(self, url, headers, body):
        """Perform auth request for token."""

        return self.req.post(url, headers, body=body)

def log_all(self, file):
        """Log all data received from RFLink to file."""
        global rflink_log
        if file == None:
            rflink_log = None
        else:
            log.debug('logging to: %s', file)
            rflink_log = open(file, 'a')

def set_title(self, title, **kwargs):
        """Sets the title on the underlying matplotlib AxesSubplot."""
        ax = self.get_axes()
        ax.set_title(title, **kwargs)

def setLoggerAll(self, mthd):
        """ Sends all messages to ``logger.[mthd]()`` for handling """
        for key in self._logger_methods:
            self._logger_methods[key] = mthd

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

def log_no_newline(self, msg):
      """ print the message to the predefined log file without newline """
      self.print2file(self.logfile, False, False, msg)

def _turn_sigterm_into_systemexit(): # pragma: no cover
    """
    Attempts to turn a SIGTERM exception into a SystemExit exception.
    """
    try:
        import signal
    except ImportError:
        return
    def handle_term(signo, frame):
        raise SystemExit
    signal.signal(signal.SIGTERM, handle_term)

def info(self, message, *args, **kwargs):
        """More important level : default for print and save
        """
        self._log(logging.INFO, message, *args, **kwargs)

def patch_lines(x):
    """
    Draw lines between groups
    """
    for idx in range(len(x)-1):
        x[idx] = np.vstack([x[idx], x[idx+1][0,:]])
    return x

def info(self, message, *args, **kwargs):
        """More important level : default for print and save
        """
        self._log(logging.INFO, message, *args, **kwargs)

def add_queue_handler(queue):
    """Add a queue log handler to the global logger."""
    handler = QueueLogHandler(queue)
    handler.setFormatter(QueueFormatter())
    handler.setLevel(DEBUG)
    GLOBAL_LOGGER.addHandler(handler)

def __init__(self, min_value, max_value, format="%(bar)s: %(percentage) 6.2f%% %(timeinfo)s", width=40, barchar="#", emptychar="-", output=sys.stdout):
		"""		
			:param min_value: minimum value for update(..)
			:param format: format specifier for the output
			:param width: width of the progress bar's (excluding extra text)
			:param barchar: character used to print the bar
			:param output: where to write the output to
		"""
		self.min_value = min_value
		self.max_value = max_value
		self.format = format
		self.width = width
		self.barchar = barchar
		self.emptychar = emptychar
		self.output = output
		
		self.firsttime = True
		self.prevtime = time.time()
		self.starttime = self.prevtime
		self.prevfraction = 0
		self.firsttimedone = False
		self.value = self.min_value

def consts(self):
        """The constants referenced in this code object.
        """
        # We cannot use a set comprehension because consts do not need
        # to be hashable.
        consts = []
        append_const = consts.append
        for instr in self.instrs:
            if isinstance(instr, LOAD_CONST) and instr.arg not in consts:
                append_const(instr.arg)
        return tuple(consts)

def __add__(self, other):
        """Left addition."""
        return chaospy.poly.collection.arithmetics.add(self, other)

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

def get_file_size(filename):
    """
    Get the file size of a given file

    :param filename: string: pathname of a file
    :return: human readable filesize
    """
    if os.path.isfile(filename):
        return convert_size(os.path.getsize(filename))
    return None

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

def sigterm(self, signum, frame):
        """
        These actions will be done after SIGTERM.
        """
        self.logger.warning("Caught signal %s. Stopping daemon." % signum)
        sys.exit(0)

def _stdin_(p):
    """Takes input from user. Works for Python 2 and 3."""
    _v = sys.version[0]
    return input(p) if _v is '3' else raw_input(p)

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

def get_bin_indices(self, values):
        """Returns index tuple in histogram of bin which contains value"""
        return tuple([self.get_axis_bin_index(values[ax_i], ax_i)
                      for ax_i in range(self.dimensions)])

def make_symmetric(dict):
    """Makes the given dictionary symmetric. Values are assumed to be unique."""
    for key, value in list(dict.items()):
        dict[value] = key
    return dict

def good(txt):
    """Print, emphasized 'good', the given 'txt' message"""

    print("%s# %s%s%s" % (PR_GOOD_CC, get_time_stamp(), txt, PR_NC))
    sys.stdout.flush()

def ver_to_tuple(value):
    """
    Convert version like string to a tuple of integers.
    """
    return tuple(int(_f) for _f in re.split(r'\D+', value) if _f)

def clean_with_zeros(self,x):
        """ set nan and inf rows from x to zero"""
        x[~np.any(np.isnan(x) | np.isinf(x),axis=1)] = 0
        return x

def local_accuracy(X_train, y_train, X_test, y_test, attr_test, model_generator, metric, trained_model):
    """ The how well do the features plus a constant base rate sum up to the model output.
    """

    X_train, X_test = to_array(X_train, X_test)

    # how many features to mask
    assert X_train.shape[1] == X_test.shape[1]

    # keep nkeep top features and re-train the model for each test explanation
    yp_test = trained_model.predict(X_test)

    return metric(yp_test, strip_list(attr_test).sum(1))

def url_encode(url):
    """
    Convert special characters using %xx escape.

    :param url: str
    :return: str - encoded url
    """
    if isinstance(url, text_type):
        url = url.encode('utf8')
    return quote(url, ':/%?&=')

def calculate_bbox_area(bbox, rows, cols):
    """Calculate the area of a bounding box in pixels."""
    bbox = denormalize_bbox(bbox, rows, cols)
    x_min, y_min, x_max, y_max = bbox[:4]
    area = (x_max - x_min) * (y_max - y_min)
    return area

def tearDown(self):
        """ Clean up environment

        """
        if self.sdkobject and self.sdkobject.id:
            self.sdkobject.delete()
            self.sdkobject.id = None

def center_eigenvalue_diff(mat):
    """Compute the eigvals of mat and then find the center eigval difference."""
    N = len(mat)
    evals = np.sort(la.eigvals(mat))
    diff = np.abs(evals[N/2] - evals[N/2-1])
    return diff

def handle_m2m(self, sender, instance, **kwargs):
    """ Handle many to many relationships """
    self.handle_save(instance.__class__, instance)

def _get_log_prior_cl_func(self):
        """Get the CL log prior compute function.

        Returns:
            str: the compute function for computing the log prior.
        """
        return SimpleCLFunction.from_string('''
            mot_float_type _computeLogPrior(local const mot_float_type* x, void* data){
                return ''' + self._log_prior_func.get_cl_function_name() + '''(x, data);
            }
        ''', dependencies=[self._log_prior_func])

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

def idx(df, index):
    """Universal indexing for numpy and pandas objects."""
    if isinstance(df, (pd.DataFrame, pd.Series)):
        return df.iloc[index]
    else:
        return df[index, :]

def set_ylimits(self, row, column, min=None, max=None):
        """Set y-axis limits of a subplot.

        :param row,column: specify the subplot.
        :param min: minimal axis value
        :param max: maximum axis value

        """
        subplot = self.get_subplot_at(row, column)
        subplot.set_ylimits(min, max)

def list_apis(awsclient):
    """List APIs in account."""
    client_api = awsclient.get_client('apigateway')

    apis = client_api.get_rest_apis()['items']

    for api in apis:
        print(json2table(api))

def uint32_to_uint8(cls, img):
        """
        Cast uint32 RGB image to 4 uint8 channels.
        """
        return np.flipud(img.view(dtype=np.uint8).reshape(img.shape + (4,)))

def quit(self):
        """ Exit the program due to user's choices.
        """
        self.script.LOG.warn("Abort due to user choice!")
        sys.exit(self.QUIT_RC)

def raise_figure_window(f=0):
    """
    Raises the supplied figure number or figure window.
    """
    if _fun.is_a_number(f): f = _pylab.figure(f)
    f.canvas.manager.window.raise_()

def mixedcase(path):
    """Removes underscores and capitalizes the neighbouring character"""
    words = path.split('_')
    return words[0] + ''.join(word.title() for word in words[1:])

def raise_figure_window(f=0):
    """
    Raises the supplied figure number or figure window.
    """
    if _fun.is_a_number(f): f = _pylab.figure(f)
    f.canvas.manager.window.raise_()

def to_camel(s):
    """
    :param string s: under_scored string to be CamelCased
    :return: CamelCase version of input
    :rtype: str
    """
    # r'(?!^)_([a-zA-Z]) original regex wasn't process first groups
    return re.sub(r'_([a-zA-Z])', lambda m: m.group(1).upper(), '_' + s)

def figsize(x=8, y=7., aspect=1.):
    """ manually set the default figure size of plots
    ::Arguments::
        x (float): x-axis size
        y (float): y-axis size
        aspect (float): aspect ratio scalar
    """
    # update rcparams with adjusted figsize params
    mpl.rcParams.update({'figure.figsize': (x*aspect, y)})

def to_camel_case(text):
    """Convert to camel case.

    :param str text:
    :rtype: str
    :return:
    """
    split = text.split('_')
    return split[0] + "".join(x.title() for x in split[1:])

def _process_legend(self):
        """
        Disables legends if show_legend is disabled.
        """
        for l in self.handles['plot'].legend:
            l.items[:] = []
            l.border_line_alpha = 0
            l.background_fill_alpha = 0

def snake_to_camel(name):
    """Takes a snake_field_name and returns a camelCaseFieldName

    Args:
        name (str): E.g. snake_field_name or SNAKE_FIELD_NAME

    Returns:
        str: camelCase converted name. E.g. capsFieldName
    """
    ret = "".join(x.title() for x in name.split("_"))
    ret = ret[0].lower() + ret[1:]
    return ret

def axes_off(ax):
    """Get rid of all axis ticks, lines, etc.
    """
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)

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

def set_xlimits(self, row, column, min=None, max=None):
        """Set x-axis limits of a subplot.

        :param row,column: specify the subplot.
        :param min: minimal axis value
        :param max: maximum axis value

        """
        subplot = self.get_subplot_at(row, column)
        subplot.set_xlimits(min, max)

def as_tuple(self, value):
        """Utility function which converts lists to tuples."""
        if isinstance(value, list):
            value = tuple(value)
        return value

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

def flatten_list(l):
    """ Nested lists to single-level list, does not split strings"""
    return list(chain.from_iterable(repeat(x,1) if isinstance(x,str) else x for x in l))

def R_rot_3d(th):
    """Return a 3-dimensional rotation matrix.

    Parameters
    ----------
    th: array, shape (n, 3)
        Angles about which to rotate along each axis.

    Returns
    -------
    R: array, shape (n, 3, 3)
    """
    sx, sy, sz = np.sin(th).T
    cx, cy, cz = np.cos(th).T
    R = np.empty((len(th), 3, 3), dtype=np.float)

    R[:, 0, 0] = cy * cz
    R[:, 0, 1] = -cy * sz
    R[:, 0, 2] = sy

    R[:, 1, 0] = sx * sy * cz + cx * sz
    R[:, 1, 1] = -sx * sy * sz + cx * cz
    R[:, 1, 2] = -sx * cy

    R[:, 2, 0] = -cx * sy * cz + sx * sz
    R[:, 2, 1] = cx * sy * sz + sx * cz
    R[:, 2, 2] = cx * cy
    return R

def set_color(self, fg=None, bg=None, intensify=False, target=sys.stdout):
        """Set foreground- and background colors and intensity."""
        raise NotImplementedError

def _heappop_max(heap):
    """Maxheap version of a heappop."""
    lastelt = heap.pop()    # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        _siftup_max(heap, 0)
        return returnitem
    return lastelt

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

def join(self):
		"""Note that the Executor must be close()'d elsewhere,
		or join() will never return.
		"""
		self.inputfeeder_thread.join()
		self.pool.join()
		self.resulttracker_thread.join()
		self.failuretracker_thread.join()

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

def dt_to_ts(value):
    """ If value is a datetime, convert to timestamp """
    if not isinstance(value, datetime):
        return value
    return calendar.timegm(value.utctimetuple()) + value.microsecond / 1000000.0

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

def flatten(l, types=(list, float)):
    """
    Flat nested list of lists into a single list.
    """
    l = [item if isinstance(item, types) else [item] for item in l]
    return [item for sublist in l for item in sublist]

def std_datestr(self, datestr):
        """Reformat a date string to standard format.
        """
        return date.strftime(
                self.str2date(datestr), self.std_dateformat)

def from_string(cls, s):
        """Return a `Status` instance from its string representation."""
        for num, text in cls._STATUS2STR.items():
            if text == s:
                return cls(num)
        else:
            raise ValueError("Wrong string %s" % s)

def _get_minidom_tag_value(station, tag_name):
    """get a value from a tag (if it exists)"""
    tag = station.getElementsByTagName(tag_name)[0].firstChild
    if tag:
        return tag.nodeValue

    return None

def populate_obj(obj, attrs):
    """Populates an object's attributes using the provided dict
    """
    for k, v in attrs.iteritems():
        setattr(obj, k, v)

def update(self, dictionary=None, **kwargs):
        """
        Adds/overwrites all the keys and values from the dictionary.
        """
        if not dictionary == None: kwargs.update(dictionary)
        for k in list(kwargs.keys()): self[k] = kwargs[k]

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

def unit_key_from_name(name):
  """Return a legal python name for the given name for use as a unit key."""
  result = name

  for old, new in six.iteritems(UNIT_KEY_REPLACEMENTS):
    result = result.replace(old, new)

  # Collapse redundant underscores and convert to uppercase.
  result = re.sub(r'_+', '_', result.upper())

  return result

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

def comma_converter(float_string):
    """Convert numbers to floats whether the decimal point is '.' or ','"""
    trans_table = maketrans(b',', b'.')
    return float(float_string.translate(trans_table))

def create_cursor(self, name=None):
        """
        Returns an active connection cursor to the database.
        """
        return Cursor(self.client_connection, self.connection, self.djongo_connection)

def _possibly_convert_objects(values):
    """Convert arrays of datetime.datetime and datetime.timedelta objects into
    datetime64 and timedelta64, according to the pandas convention.
    """
    return np.asarray(pd.Series(values.ravel())).reshape(values.shape)

def onLeftDown(self, event=None):
        """ left button down: report x,y coords, start zooming mode"""
        if event is None:
            return
        self.cursor_mode_action('leftdown', event=event)
        self.ForwardEvent(event=event.guiEvent)

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

def move_up(lines=1, file=sys.stdout):
    """ Move the cursor up a number of lines.

        Esc[ValueA:
        Moves the cursor up by the specified number of lines without changing
        columns. If the cursor is already on the top line, ANSI.SYS ignores
        this sequence.
    """
    move.up(lines).write(file=file)

def is_collection(obj):
    """Tests if an object is a collection."""

    col = getattr(obj, '__getitem__', False)
    val = False if (not col) else True

    if isinstance(obj, basestring):
        val = False

    return val

def plot3d_init(fignum):
    """
    initializes 3D plot
    """
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(fignum)
    ax = fig.add_subplot(111, projection='3d')
    return ax

def apply(self, func, workers=1, job_size=10000):
    """Apply `func` to lines of text in parallel or sequential.

    Args:
      func : a function that takes a list of lines.
    """
    if workers == 1:
      for lines in self.iter_chunks(job_size):
        yield func(lines)
    else:
      with ProcessPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(func, self.iter_chunks(job_size)):
          yield result

def is_valid_url(url):
    """Checks if a given string is an url"""
    pieces = urlparse(url)
    return all([pieces.scheme, pieces.netloc])

def multiprocess_mapping(func, iterable):
    """Multiprocess mapping the given function on the given iterable.

    This only works in Linux and Mac systems since Windows has no forking capability. On Windows we fall back on
    single processing. Also, if we reach memory limits we fall back on single cpu processing.

    Args:
        func (func): the function to apply
        iterable (iterable): the iterable with the elements we want to apply the function on
    """
    if os.name == 'nt':  # In Windows there is no fork.
        return list(map(func, iterable))
    try:
        p = multiprocessing.Pool()
        return_data = list(p.imap(func, iterable))
        p.close()
        p.join()
        return return_data
    except OSError:
        return list(map(func, iterable))

def is_gzipped_fastq(file_name):
    """
    Determine whether indicated file appears to be a gzipped FASTQ.

    :param str file_name: Name/path of file to check as gzipped FASTQ.
    :return bool: Whether indicated file appears to be in gzipped FASTQ format.
    """
    _, ext = os.path.splitext(file_name)
    return file_name.endswith(".fastq.gz") or file_name.endswith(".fq.gz")

def imapchain(*a, **kwa):
    """ Like map but also chains the results. """

    imap_results = map( *a, **kwa )
    return itertools.chain( *imap_results )

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

def compute(args):
    x, y, params = args
    """Callable function for the multiprocessing pool."""
    return x, y, mandelbrot(x, y, params)

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

def get(self, queue_get):
        """
        to get states from multiprocessing.queue
        """
        if isinstance(queue_get, (tuple, list)):
            self.result.extend(queue_get)

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

def cleanup_storage(*args):
    """Clean up processes after SIGTERM or SIGINT is received."""
    ShardedClusters().cleanup()
    ReplicaSets().cleanup()
    Servers().cleanup()
    sys.exit(0)

def isin(value, values):
    """ Check that value is in values """
    for i, v in enumerate(value):
        if v not in np.array(values)[:, i]:
            return False
    return True

def imapchain(*a, **kwa):
    """ Like map but also chains the results. """

    imap_results = map( *a, **kwa )
    return itertools.chain( *imap_results )

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

def norm(x, mu, sigma=1.0):
    """ Scipy norm function """
    return stats.norm(loc=mu, scale=sigma).pdf(x)

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

def get_qapp():
    """Return an instance of QApplication. Creates one if neccessary.

    :returns: a QApplication instance
    :rtype: QApplication
    :raises: None
    """
    global app
    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication([], QtGui.QApplication.GuiClient)
    return app

def store_many(self, sql, values):
        """Abstraction over executemany method"""
        cursor = self.get_cursor()
        cursor.executemany(sql, values)
        self.conn.commit()

def contained_in(filename, directory):
    """Test if a file is located within the given directory."""
    filename = os.path.normcase(os.path.abspath(filename))
    directory = os.path.normcase(os.path.abspath(directory))
    return os.path.commonprefix([filename, directory]) == directory

def unpack(self, s):
        """Parse bytes and return a namedtuple."""
        return self._create(super(NamedStruct, self).unpack(s))

def is_timestamp(instance):
    """Validates data is a timestamp"""
    if not isinstance(instance, (int, str)):
        return True
    return datetime.fromtimestamp(int(instance))

def to_distribution_values(self, values):
        """
        Returns numpy array of natural logarithms of ``values``.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # avoid RuntimeWarning: divide by zero encountered in log
            return numpy.log(values)

def issubset(self, other):
        """Report whether another set contains this RangeSet."""
        self._binary_sanity_check(other)
        return set.issubset(self, other)

def __contains__ (self, key):
        """Check lowercase key item."""
        assert isinstance(key, basestring)
        return dict.__contains__(self, key.lower())

def __neg__(self):
        """Unary negation"""
        return self.__class__(self[0], self._curve.p()-self[1], self._curve)

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

def is_power_of_2(num):
    """Return whether `num` is a power of two"""
    log = math.log2(num)
    return int(log) == float(log)

def to_dotfile(G: nx.DiGraph, filename: str):
    """ Output a networkx graph to a DOT file. """
    A = to_agraph(G)
    A.write(filename)

def is_punctuation(text):
    """Check if given string is a punctuation"""
    return not (text.lower() in config.AVRO_VOWELS or
                text.lower() in config.AVRO_CONSONANTS)

def comment (self, s, **args):
        """Write DOT comment."""
        self.write(u"// ")
        self.writeln(s=s, **args)

def local_accuracy(X_train, y_train, X_test, y_test, attr_test, model_generator, metric, trained_model):
    """ The how well do the features plus a constant base rate sum up to the model output.
    """

    X_train, X_test = to_array(X_train, X_test)

    # how many features to mask
    assert X_train.shape[1] == X_test.shape[1]

    # keep nkeep top features and re-train the model for each test explanation
    yp_test = trained_model.predict(X_test)

    return metric(yp_test, strip_list(attr_test).sum(1))

def process_docstring(app, what, name, obj, options, lines):
    """React to a docstring event and append contracts to it."""
    # pylint: disable=unused-argument
    # pylint: disable=too-many-arguments
    lines.extend(_format_contracts(what=what, obj=obj))

def is_readable_dir(path):
  """Returns whether a path names an existing directory we can list and read files from."""
  return os.path.isdir(path) and os.access(path, os.R_OK) and os.access(path, os.X_OK)

def normal_noise(points):
    """Init a noise variable."""
    return np.random.rand(1) * np.random.randn(points, 1) \
        + random.sample([2, -2], 1)

def are_in_interval(s, l, r, border = 'included'):
        """
        Checks whether all number in the sequence s lie inside the interval formed by
        l and r.
        """
        return numpy.all([IntensityRangeStandardization.is_in_interval(x, l, r, border) for x in s])

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

def is_unix_like(platform=None):
    """Returns whether the given platform is a Unix-like platform with the usual
    Unix filesystem. When the parameter is omitted, it defaults to ``sys.platform``
    """
    platform = platform or sys.platform
    platform = platform.lower()
    return platform.startswith("linux") or platform.startswith("darwin") or \
            platform.startswith("cygwin")

def get_uniques(l):
    """ Returns a list with no repeated elements.
    """
    result = []

    for i in l:
        if i not in result:
            result.append(i)

    return result

def _clear(self):
        """
        Helper that clears the composition.
        """
        draw = ImageDraw.Draw(self._background_image)
        draw.rectangle(self._device.bounding_box,
                       fill="black")
        del draw

def listlike(obj):
    """Is an object iterable like a list (and not a string)?"""
    
    return hasattr(obj, "__iter__") \
    and not issubclass(type(obj), str)\
    and not issubclass(type(obj), unicode)

def forget_coords(self):
        """Forget all loaded coordinates."""
        self.w.ntotal.set_text('0')
        self.coords_dict.clear()
        self.redo()

def norm(x, mu, sigma=1.0):
    """ Scipy norm function """
    return stats.norm(loc=mu, scale=sigma).pdf(x)

def _clear(self):
        """
        Helper that clears the composition.
        """
        draw = ImageDraw.Draw(self._background_image)
        draw.rectangle(self._device.bounding_box,
                       fill="black")
        del draw

def denorm(self,arr):
        """Reverse the normalization done to a batch of images.

        Arguments:
            arr: of shape/size (N,3,sz,sz)
        """
        if type(arr) is not np.ndarray: arr = to_np(arr)
        if len(arr.shape)==3: arr = arr[None]
        return self.transform.denorm(np.rollaxis(arr,1,4))

def _clear(self):
        """
        Helper that clears the composition.
        """
        draw = ImageDraw.Draw(self._background_image)
        draw.rectangle(self._device.bounding_box,
                       fill="black")
        del draw

def normalize(name):
    """Normalize name for the Statsd convention"""

    # Name should not contain some specials chars (issue #1068)
    ret = name.replace(':', '')
    ret = ret.replace('%', '')
    ret = ret.replace(' ', '_')

    return ret

def accel_next(self, *args):
        """Callback to go to the next tab. Called by the accel key.
        """
        if self.get_notebook().get_current_page() + 1 == self.get_notebook().get_n_pages():
            self.get_notebook().set_current_page(0)
        else:
            self.get_notebook().next_page()
        return True

def test(nose_argsuments):
    """ Run application tests """
    from nose import run

    params = ['__main__', '-c', 'nose.ini']
    params.extend(nose_argsuments)
    run(argv=params)

def basic_word_sim(word1, word2):
    """
    Simple measure of similarity: Number of letters in common / max length
    """
    return sum([1 for c in word1 if c in word2]) / max(len(word1), len(word2))

def isetdiff_flags(list1, list2):
    """
    move to util_iter
    """
    set2 = set(list2)
    return (item not in set2 for item in list1)

def _array2cstr(arr):
    """ Serializes a numpy array to a compressed base64 string """
    out = StringIO()
    np.save(out, arr)
    return b64encode(out.getvalue())

def cpp_prog_builder(build_context, target):
    """Build a C++ binary executable"""
    yprint(build_context.conf, 'Build CppProg', target)
    workspace_dir = build_context.get_workspace('CppProg', target.name)
    build_cpp(build_context, target, target.compiler_config, workspace_dir)

def gday_of_year(self):
        """Return the number of days since January 1 of the given year."""
        return (self.date - dt.date(self.date.year, 1, 1)).days

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

def recarray(self):
        """Returns data as :class:`numpy.recarray`."""
        return numpy.rec.fromrecords(self.records, names=self.names)

def int_to_date(date):
    """
    Convert an int of form yyyymmdd to a python date object.
    """

    year = date // 10**4
    month = date % 10**4 // 10**2
    day = date % 10**2

    return datetime.date(year, month, day)

def argsort_indices(a, axis=-1):
    """Like argsort, but returns an index suitable for sorting the
    the original array even if that array is multidimensional
    """
    a = np.asarray(a)
    ind = list(np.ix_(*[np.arange(d) for d in a.shape]))
    ind[axis] = a.argsort(axis)
    return tuple(ind)

def _read_stream_for_size(stream, buf_size=65536):
    """Reads a stream discarding the data read and returns its size."""
    size = 0
    while True:
        buf = stream.read(buf_size)
        size += len(buf)
        if not buf:
            break
    return size

def _numpy_bytes_to_char(arr):
    """Like netCDF4.stringtochar, but faster and more flexible.
    """
    # ensure the array is contiguous
    arr = np.array(arr, copy=False, order='C', dtype=np.string_)
    return arr.reshape(arr.shape + (1,)).view('S1')

def movingaverage(arr, window):
    """
    Calculates the moving average ("rolling mean") of an array
    of a certain window size.
    """
    m = np.ones(int(window)) / int(window)
    return scipy.ndimage.convolve1d(arr, m, axis=0, mode='reflect')

def log_magnitude_spectrum(frames):
    """Compute the log of the magnitude spectrum of frames"""
    return N.log(N.abs(N.fft.rfft(frames)).clip(1e-5, N.inf))

def read_numpy(fd, byte_order, dtype, count):
    """Read tag data from file and return as numpy array."""
    return numpy.fromfile(fd, byte_order+dtype[-1], count)

def list2string (inlist,delimit=' '):
    """
Converts a 1D list to a single long string for file output, using
the string.join function.

Usage:   list2string (inlist,delimit=' ')
Returns: the string created from inlist
"""
    stringlist = [makestr(_) for _ in inlist]
    return string.join(stringlist,delimit)

def _isstring(dtype):
    """Given a numpy dtype, determines whether it is a string. Returns True
    if the dtype is string or unicode.
    """
    return dtype.type == numpy.unicode_ or dtype.type == numpy.string_

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

def Max(a, axis, keep_dims):
    """
    Max reduction op.
    """
    return np.amax(a, axis=axis if not isinstance(axis, np.ndarray) else tuple(axis),
                   keepdims=keep_dims),

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

def read_numpy(fd, byte_order, dtype, count):
    """Read tag data from file and return as numpy array."""
    return numpy.fromfile(fd, byte_order+dtype[-1], count)

def setupLogFile(self):
		"""Set up the logging file for a new session- include date and some whitespace"""
		self.logWrite("\n###############################################")
		self.logWrite("calcpkg.py log from " + str(datetime.datetime.now()))
		self.changeLogging(True)

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

def _to_array(value):
    """As a convenience, turn Python lists and tuples into NumPy arrays."""
    if isinstance(value, (tuple, list)):
        return array(value)
    elif isinstance(value, (float, int)):
        return np.float64(value)
    else:
        return value

def get_oauth_token():
    """Retrieve a simple OAuth Token for use with the local http client."""
    url = "{0}/token".format(DEFAULT_ORIGIN["Origin"])
    r = s.get(url=url)
    return r.json()["t"]

def connected_socket(address, timeout=3):
    """ yields a connected socket """
    sock = socket.create_connection(address, timeout)
    yield sock
    sock.close()

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

def updateFromKwargs(self, properties, kwargs, collector, **unused):
        """Primary entry point to turn 'kwargs' into 'properties'"""
        properties[self.name] = self.getFromKwargs(kwargs)

def listlike(obj):
    """Is an object iterable like a list (and not a string)?"""
    
    return hasattr(obj, "__iter__") \
    and not issubclass(type(obj), str)\
    and not issubclass(type(obj), unicode)

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

def _time_to_json(value):
    """Coerce 'value' to an JSON-compatible representation."""
    if isinstance(value, datetime.time):
        value = value.isoformat()
    return value

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

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, LegipyModel):
        return obj.to_json()
    elif isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    raise TypeError("Type {0} not serializable".format(repr(type(obj))))

def unique(seq):
    """Return the unique elements of a collection even if those elements are
       unhashable and unsortable, like dicts and sets"""
    cleaned = []
    for each in seq:
        if each not in cleaned:
            cleaned.append(each)
    return cleaned

def to_json(obj):
    """Return a json string representing the python object obj."""
    i = StringIO.StringIO()
    w = Writer(i, encoding='UTF-8')
    w.write_value(obj)
    return i.getvalue()

def delete_index(index):
    """Delete index entirely (removes all documents and mapping)."""
    logger.info("Deleting search index: '%s'", index)
    client = get_client()
    return client.indices.delete(index=index)

def time2seconds(t):
    """Returns seconds since 0h00."""
    return t.hour * 3600 + t.minute * 60 + t.second + float(t.microsecond) / 1e6

def clear(self):
        """Remove all nodes and edges from the graph.

        Unlike the regular networkx implementation, this does *not*
        remove the graph's name. But all the other graph, node, and
        edge attributes go away.

        """
        self.adj.clear()
        self.node.clear()
        self.graph.clear()

def fit_linear(X, y):
    """
    Uses OLS to fit the regression.
    """
    model = linear_model.LinearRegression()
    model.fit(X, y)
    return model

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

def open_with_encoding(filename, encoding, mode='r'):
    """Return opened file with a specific encoding."""
    return io.open(filename, mode=mode, encoding=encoding,
                   newline='')

def mouse_get_pos():
    """

    :return:
    """
    p = POINT()
    AUTO_IT.AU3_MouseGetPos(ctypes.byref(p))
    return p.x, p.y

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

def get_encoding(binary):
    """Return the encoding type."""

    try:
        from chardet import detect
    except ImportError:
        LOGGER.error("Please install the 'chardet' module")
        sys.exit(1)

    encoding = detect(binary).get('encoding')

    return 'iso-8859-1' if encoding == 'CP949' else encoding

def Date(value):
    """Custom type for managing dates in the command-line."""
    from datetime import datetime
    try:
        return datetime(*reversed([int(val) for val in value.split('/')]))
    except Exception as err:
        raise argparse.ArgumentTypeError("invalid date '%s'" % value)

def normalize(X):
    """ equivalent to scipy.preprocessing.normalize on sparse matrices
    , but lets avoid another depedency just for a small utility function """
    X = coo_matrix(X)
    X.data = X.data / sqrt(bincount(X.row, X.data ** 2))[X.row]
    return X

def _index_ordering(redshift_list):
        """

        :param redshift_list: list of redshifts
        :return: indexes in acending order to be evaluated (from z=0 to z=z_source)
        """
        redshift_list = np.array(redshift_list)
        sort_index = np.argsort(redshift_list)
        return sort_index

def unproject(self, xy):
        """
        Returns the coordinates from position in meters
        """
        (x, y) = xy
        lng = x/EARTH_RADIUS * RAD_TO_DEG
        lat = 2 * atan(exp(y/EARTH_RADIUS)) - pi/2 * RAD_TO_DEG
        return (lng, lat)

def _pad(self, text):
        """Pad the text."""
        top_bottom = ("\n" * self._padding) + " "
        right_left = " " * self._padding * self.PAD_WIDTH
        return top_bottom + right_left + text + right_left + top_bottom

def normal_noise(points):
    """Init a noise variable."""
    return np.random.rand(1) * np.random.randn(points, 1) \
        + random.sample([2, -2], 1)

def pad_hex(value, bit_size):
    """
    Pads a hex string up to the given bit_size
    """
    value = remove_0x_prefix(value)
    return add_0x_prefix(value.zfill(int(bit_size / 4)))

def text_width(string, font_name, font_size):
    """Determine with width in pixels of string."""
    return stringWidth(string, fontName=font_name, fontSize=font_size)

def clean_time(time_string):
    """Return a datetime from the Amazon-provided datetime string"""
    # Get a timezone-aware datetime object from the string
    time = dateutil.parser.parse(time_string)
    if not settings.USE_TZ:
        # If timezone support is not active, convert the time to UTC and
        # remove the timezone field
        time = time.astimezone(timezone.utc).replace(tzinfo=None)
    return time

def isin(value, values):
    """ Check that value is in values """
    for i, v in enumerate(value):
        if v not in np.array(values)[:, i]:
            return False
    return True

def parse_host_port (host_port):
    """Parse a host:port string into separate components."""
    host, port = urllib.splitport(host_port.strip())
    if port is not None:
        if urlutil.is_numeric_port(port):
            port = int(port)
    return host, port

def get_in_samples(samples, fn):
    """
    for a list of samples, return the value of a global option
    """
    for sample in samples:
        sample = to_single_data(sample)
        if fn(sample, None):
            return fn(sample)
    return None

def clean_markdown(text):
    """
    Parse markdown sintaxt to html.
    """
    result = text

    if isinstance(text, str):
        result = ''.join(
            BeautifulSoup(markdown(text), 'lxml').findAll(text=True))

    return result

def _connection_failed(self, error="Error not specified!"):
        """Clean up after connection failure detected."""
        if not self._error:
            LOG.error("Connection failed: %s", str(error))
            self._error = error

def md_to_text(content):
    """ Converts markdown content to text """
    text = None
    html = markdown.markdown(content)
    if html:
        text = html_to_text(content)
    return text

def local_accuracy(X_train, y_train, X_test, y_test, attr_test, model_generator, metric, trained_model):
    """ The how well do the features plus a constant base rate sum up to the model output.
    """

    X_train, X_test = to_array(X_train, X_test)

    # how many features to mask
    assert X_train.shape[1] == X_test.shape[1]

    # keep nkeep top features and re-train the model for each test explanation
    yp_test = trained_model.predict(X_test)

    return metric(yp_test, strip_list(attr_test).sum(1))

def read_proto_object(fobj, klass):
    """Read a block of data and parse using the given protobuf object."""
    log.debug('%s chunk', klass.__name__)
    obj = klass()
    obj.ParseFromString(read_block(fobj))
    log.debug('Header: %s', str(obj))
    return obj

def _remove_keywords(d):
    """
    copy the dict, filter_keywords

    Parameters
    ----------
    d : dict
    """
    return { k:v for k, v in iteritems(d) if k not in RESERVED }

def _read_date_from_string(str1):
    """
    Reads the date from a string in the format YYYY/MM/DD and returns
    :class: datetime.date
    """
    full_date = [int(x) for x in str1.split('/')]
    return datetime.date(full_date[0], full_date[1], full_date[2])

def bash(filename):
    """Runs a bash script in the local directory"""
    sys.stdout.flush()
    subprocess.call("bash {}".format(filename), shell=True)

def main(args=sys.argv):
    """
    main entry point for the jardiff CLI
    """

    parser = create_optparser(args[0])
    return cli(parser.parse_args(args[1:]))

def run_command(cmd, *args):
    """
    Runs command on the system with given ``args``.
    """
    command = ' '.join((cmd, args))
    p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p.communicate()
    return p.retcode, stdout, stderr

def _render_table(data, fields=None):
  """ Helper to render a list of dictionaries as an HTML display object. """
  return IPython.core.display.HTML(datalab.utils.commands.HtmlBuilder.render_table(data, fields))

def extract_table_names(query):
    """ Extract table names from an SQL query. """
    # a good old fashioned regex. turns out this worked better than actually parsing the code
    tables_blocks = re.findall(r'(?:FROM|JOIN)\s+(\w+(?:\s*,\s*\w+)*)', query, re.IGNORECASE)
    tables = [tbl
              for block in tables_blocks
              for tbl in re.findall(r'\w+', block)]
    return set(tables)

def iget_list_column_slice(list_, start=None, stop=None, stride=None):
    """ iterator version of get_list_column """
    if isinstance(start, slice):
        slice_ = start
    else:
        slice_ = slice(start, stop, stride)
    return (row[slice_] for row in list_)

def extract_table_names(query):
    """ Extract table names from an SQL query. """
    # a good old fashioned regex. turns out this worked better than actually parsing the code
    tables_blocks = re.findall(r'(?:FROM|JOIN)\s+(\w+(?:\s*,\s*\w+)*)', query, re.IGNORECASE)
    tables = [tbl
              for block in tables_blocks
              for tbl in re.findall(r'\w+', block)]
    return set(tables)

def OnPasteAs(self, event):
        """Clipboard paste as event handler"""

        data = self.main_window.clipboard.get_clipboard()
        key = self.main_window.grid.actions.cursor

        with undo.group(_("Paste As...")):
            self.main_window.actions.paste_as(key, data)

        self.main_window.grid.ForceRefresh()

        event.Skip()

def _value_to_color(value, cmap):
    """Convert a value in the range [0,1] to an RGB tuple using a colormap."""
    cm = plt.get_cmap(cmap)
    rgba = cm(value)
    return [int(round(255*v)) for v in rgba[0:3]]

def data_directory():
    """Return the absolute path to the directory containing the package data."""
    package_directory = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(package_directory, "data")

def _rank(self, ranking, n):
    """ return the first n sentences with highest ranking """
    return nlargest(n, ranking, key=ranking.get)

def debug_on_error(type, value, tb):
    """Code due to Thomas Heller - published in Python Cookbook (O'Reilley)"""
    traceback.print_exc(type, value, tb)
    print()
    pdb.pm()

def _put_header(self):
        """ Standard first line in a PDF. """
        self.session._out('%%PDF-%s' % self.pdf_version)
        if self.session.compression:
            self.session.buffer += '%' + chr(235) + chr(236) + chr(237) + chr(238) + "\n"

def get_lines(handle, line):
    """
    Get zero-indexed line from an open file-like.
    """
    for i, l in enumerate(handle):
        if i == line:
            return l

def dimensions(self):
        """Get width and height of a PDF"""
        size = self.pdf.getPage(0).mediaBox
        return {'w': float(size[2]), 'h': float(size[3])}

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

def cor(y_true, y_pred):
    """Compute Pearson correlation coefficient.
    """
    y_true, y_pred = _mask_nan(y_true, y_pred)
    return np.corrcoef(y_true, y_pred)[0, 1]

def make_2d(ary):
    """Convert any array into a 2d numpy array.

    In case the array is already more than 2 dimensional, will ravel the
    dimensions after the first.
    """
    dim_0, *_ = np.atleast_1d(ary).shape
    return ary.reshape(dim_0, -1, order="F")

def topk(arg, k, by=None):
    """
    Returns
    -------
    topk : TopK filter expression
    """
    op = ops.TopK(arg, k, by=by)
    return op.to_expr()

def flatten( iterables ):
    """ Flatten an iterable, except for string elements. """
    for it in iterables:
        if isinstance(it, str):
            yield it
        else:
            for element in it:
                yield element

def return_letters_from_string(text):
    """Get letters from string only."""
    out = ""
    for letter in text:
        if letter.isalpha():
            out += letter
    return out

def set_stop_handler(self):
        """
        Initializes functions that are invoked when the user or OS wants to kill this process.
        :return:
        """
        signal.signal(signal.SIGTERM, self.graceful_stop)
        signal.signal(signal.SIGABRT, self.graceful_stop)
        signal.signal(signal.SIGINT, self.graceful_stop)

def _openResources(self):
        """ Uses numpy.load to open the underlying file
        """
        arr = np.load(self._fileName, allow_pickle=ALLOW_PICKLE)
        check_is_an_array(arr)
        self._array = arr

def fixed(ctx, number, decimals=2, no_commas=False):
    """
    Formats the given number in decimal format using a period and commas
    """
    value = _round(ctx, number, decimals)
    format_str = '{:f}' if no_commas else '{:,f}'
    return format_str.format(value)

def focus(self):
        """
        Call this to give this Widget the input focus.
        """
        self._has_focus = True
        self._frame.move_to(self._x, self._y, self._h)
        if self._on_focus is not None:
            self._on_focus()

def pause(self):
        """Pause the music"""
        mixer.music.pause()
        self.pause_time = self.get_time()
        self.paused = True

def GeneratePassphrase(length=20):
  """Create a 20 char passphrase with easily typeable chars."""
  valid_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
  valid_chars += "0123456789 ,-_&$#"
  return "".join(random.choice(valid_chars) for i in range(length))

def polyline(self, arr):
        """Draw a set of lines"""
        for i in range(0, len(arr) - 1):
            self.line(arr[i][0], arr[i][1], arr[i + 1][0], arr[i + 1][1])

def random_numbers(n):
    """
    Generate a random string from 0-9
    :param n: length of the string
    :return: the random string
    """
    return ''.join(random.SystemRandom().choice(string.digits) for _ in range(n))

def image_set_aspect(aspect=1.0, axes="gca"):
    """
    sets the aspect ratio of the current zoom level of the imshow image
    """
    if axes is "gca": axes = _pylab.gca()

    e = axes.get_images()[0].get_extent()
    axes.set_aspect(abs((e[1]-e[0])/(e[3]-e[2]))/aspect)

def lognorm(x, mu, sigma=1.0):
    """ Log-normal function from scipy """
    return stats.lognorm(sigma, scale=mu).pdf(x)

def axes_off(ax):
    """Get rid of all axis ticks, lines, etc.
    """
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)

def generate_uuid():
    """Generate a UUID."""
    r_uuid = base64.urlsafe_b64encode(uuid.uuid4().bytes)
    return r_uuid.decode().replace('=', '')

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

def make_unique_ngrams(s, n):
    """Make a set of unique n-grams from a string."""
    return set(s[i:i + n] for i in range(len(s) - n + 1))

def axes_off(ax):
    """Get rid of all axis ticks, lines, etc.
    """
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)

def get_parent_folder_name(file_path):
    """Finds parent folder of file

    :param file_path: path
    :return: Name of folder container
    """
    return os.path.split(os.path.split(os.path.abspath(file_path))[0])[-1]

def oplot(self, x, y, **kw):
        """generic plotting method, overplotting any existing plot """
        self.panel.oplot(x, y, **kw)

def get_available_gpus():
  """
  Returns a list of string names of all available GPUs
  """
  local_device_protos = device_lib.list_local_devices()
  return [x.name for x in local_device_protos if x.device_type == 'GPU']

def axes_off(ax):
    """Get rid of all axis ticks, lines, etc.
    """
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)

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

def pprint(self, stream=None, indent=1, width=80, depth=None):
    """
    Pretty print the underlying literal Python object
    """
    pp.pprint(to_literal(self), stream, indent, width, depth)

def get_column_keys_and_names(table):
    """
    Return a generator of tuples k, c such that k is the name of the python attribute for
    the column and c is the name of the column in the sql table.
    """
    ins = inspect(table)
    return ((k, c.name) for k, c in ins.mapper.c.items())

def print_matrix(X, decimals=1):
    """Pretty printing for numpy matrix X"""
    for row in np.round(X, decimals=decimals):
        print(row)

def get_base_dir():
    """
    Return the base directory
    """
    return os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]

def get_single_value(d):
    """Get a value from a dict which contains just one item."""
    assert len(d) == 1, 'Single-item dict must have just one item, not %d.' % len(d)
    return next(six.itervalues(d))

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

def diff(file_, imports):
    """Display the difference between modules in a file and imported modules."""
    modules_not_imported = compare_modules(file_, imports)

    logging.info("The following modules are in {} but do not seem to be imported: "
                 "{}".format(file_, ", ".join(x for x in modules_not_imported)))

def disable_stdout_buffering():
    """This turns off stdout buffering so that outputs are immediately
    materialized and log messages show up before the program exits"""
    stdout_orig = sys.stdout
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
    # NOTE(brandyn): This removes the original stdout
    return stdout_orig

def get_combined_size(tiles):
    """Calculate combined size of tiles."""
    # TODO: Refactor calculating layout to avoid repetition.
    columns, rows = calc_columns_rows(len(tiles))
    tile_size = tiles[0].image.size
    return (tile_size[0] * columns, tile_size[1] * rows)

def cli(env):
    """Show current configuration."""

    settings = config.get_settings_from_client(env.client)
    env.fout(config.config_table(settings))

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

def show():
    """Show (print out) current environment variables."""
    env = get_environment()

    for key, val in sorted(env.env.items(), key=lambda item: item[0]):
        click.secho('%s = %s' % (key, val))

def get_file_name(url):
  """Returns file name of file at given url."""
  return os.path.basename(urllib.parse.urlparse(url).path) or 'unknown_name'

def _prtstr(self, obj, dashes):
        """Print object information using a namedtuple and a format pattern."""
        self.prt.write('{DASHES:{N}}'.format(
            DASHES=self.fmt_dashes.format(DASHES=dashes, ID=obj.item_id),
            N=self.dash_len))
        self.prt.write("{INFO}\n".format(INFO=str(obj)))

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

def debug(ftn, txt):
    """Used for debugging."""
    if debug_p:
        sys.stdout.write("{0}.{1}:{2}\n".format(modname, ftn, txt))
        sys.stdout.flush()

def get_month_start(day=None):
    """Returns the first day of the given month."""
    day = add_timezone(day or datetime.date.today())
    return day.replace(day=1)

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

def init_checks_registry():
    """Register all globally visible functions.

    The first argument name is either 'physical_line' or 'logical_line'.
    """
    mod = inspect.getmodule(register_check)
    for (name, function) in inspect.getmembers(mod, inspect.isfunction):
        register_check(function)

def _prtfmt(self, item_id, dashes):
        """Print object information using a namedtuple and a format pattern."""
        ntprt = self.id2nt[item_id]
        dct = ntprt._asdict()
        self.prt.write('{DASHES:{N}}'.format(
            DASHES=self.fmt_dashes.format(DASHES=dashes, ID=self.nm2prtfmt['ID'].format(**dct)),
            N=self.dash_len))
        self.prt.write("{INFO}\n".format(INFO=self.nm2prtfmt['ITEM'].format(**dct)))

def get_memory_usage():
    """Gets RAM memory usage

    :return: MB of memory used by this process
    """
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss
    return mem / (1024 * 1024)

def print_tree(self, indent=2):
        """ print_tree: prints out structure of tree
            Args: indent (int): What level of indentation at which to start printing
            Returns: None
        """
        config.LOGGER.info("{indent}{data}".format(indent="   " * indent, data=str(self)))
        for child in self.children:
            child.print_tree(indent + 1)

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

def imp_print(self, text, end):
		"""Directly send utf8 bytes to stdout"""
		sys.stdout.write((text + end).encode("utf-8"))

def get_week_start_end_day():
    """
    Get the week start date and end date
    """
    t = date.today()
    wd = t.weekday()
    return (t - timedelta(wd), t + timedelta(6 - wd))

def is_alive(self):
        """
        @rtype:  bool
        @return: C{True} if the process is currently running.
        """
        try:
            self.wait(0)
        except WindowsError:
            e = sys.exc_info()[1]
            return e.winerror == win32.WAIT_TIMEOUT
        return False

def string_input(prompt=''):
    """Python 3 input()/Python 2 raw_input()"""
    v = sys.version[0]
    if v == '3':
        return input(prompt)
    else:
        return raw_input(prompt)

def kill_mprocess(process):
    """kill process
    Args:
        process - Popen object for process
    """
    if process and proc_alive(process):
        process.terminate()
        process.communicate()
    return not proc_alive(process)

def _strip_namespace(self, xml):
        """strips any namespaces from an xml string"""
        p = re.compile(b"xmlns=*[\"\"][^\"\"]*[\"\"]")
        allmatches = p.finditer(xml)
        for match in allmatches:
            xml = xml.replace(match.group(), b"")
        return xml

def list_to_csv(value):
    """
    Converts list to string with comma separated values. For string is no-op.
    """
    if isinstance(value, (list, tuple, set)):
        value = ",".join(value)
    return value

def ave_list_v3(vec_list):
    """Return the average vector of a list of vectors."""

    vec = Vec3(0, 0, 0)
    for v in vec_list:
        vec += v
    num_vecs = float(len(vec_list))
    vec = Vec3(vec.x / num_vecs, vec.y / num_vecs, vec.z / num_vecs)
    return vec

def inheritdoc(method):
    """Set __doc__ of *method* to __doc__ of *method* in its parent class.

    Since this is used on :class:`.StringMixIn`, the "parent class" used is
    ``str``. This function can be used as a decorator.
    """
    method.__doc__ = getattr(str, method.__name__).__doc__
    return method

def url(self):
        """ The url of this window """
        with switch_window(self._browser, self.name):
            return self._browser.url

def input(self, prompt, default=None, show_default=True):
        """Provide a command prompt."""
        return click.prompt(prompt, default=default, show_default=show_default)

def value(self):
        """Value of property."""
        if self._prop.fget is None:
            raise AttributeError('Unable to read attribute')
        return self._prop.fget(self._obj)

def rel_path(filename):
    """
    Function that gets relative path to the filename
    """
    return os.path.join(os.getcwd(), os.path.dirname(__file__), filename)

def _set_property(self, val, *args):
        """Private method that sets the value currently of the property"""
        val = UserClassAdapter._set_property(self, val, *args)
        if val:
            Adapter._set_property(self, val, *args)
        return val

def lastmod(self, author):
        """Return the last modification of the entry."""
        lastitems = EntryModel.objects.published().order_by('-modification_date').filter(author=author).only('modification_date')
        return lastitems[0].modification_date

def value(self):
        """Value of property."""
        if self._prop.fget is None:
            raise AttributeError('Unable to read attribute')
        return self._prop.fget(self._obj)

def calculate_size(name, data_list):
    """ Calculates the request payload size"""
    data_size = 0
    data_size += calculate_size_str(name)
    data_size += INT_SIZE_IN_BYTES
    for data_list_item in data_list:
        data_size += calculate_size_data(data_list_item)
    return data_size

def message_from_string(s, *args, **kws):
    """Parse a string into a Message object model.

    Optional _class and strict are passed to the Parser constructor.
    """
    from future.backports.email.parser import Parser
    return Parser(*args, **kws).parsestr(s)

def _count_leading_whitespace(text):
  """Returns the number of characters at the beginning of text that are whitespace."""
  idx = 0
  for idx, char in enumerate(text):
    if not char.isspace():
      return idx
  return idx + 1

def IsErrorSuppressedByNolint(category, linenum):
  """Returns true if the specified error category is suppressed on this line.

  Consults the global error_suppressions map populated by
  ParseNolintSuppressions/ResetNolintSuppressions.

  Args:
    category: str, the category of the error.
    linenum: int, the current line number.
  Returns:
    bool, True iff the error should be suppressed due to a NOLINT comment.
  """
  return (linenum in _error_suppressions.get(category, set()) or
          linenum in _error_suppressions.get(None, set()))

def _raise_if_wrong_file_signature(stream):
    """ Reads the 4 first bytes of the stream to check that is LASF"""
    file_sig = stream.read(len(headers.LAS_FILE_SIGNATURE))
    if file_sig != headers.LAS_FILE_SIGNATURE:
        raise errors.PylasError(
            "File Signature ({}) is not {}".format(file_sig, headers.LAS_FILE_SIGNATURE)
        )

def unmatched(match):
    """Return unmatched part of re.Match object."""
    start, end = match.span(0)
    return match.string[:start]+match.string[end:]

def correspond(text):
    """Communicate with the child process without closing stdin."""
    subproc.stdin.write(text)
    subproc.stdin.flush()
    return drain()

def loss(loss_value):
  """Calculates aggregated mean loss."""
  total_loss = tf.Variable(0.0, False)
  loss_count = tf.Variable(0, False)
  total_loss_update = tf.assign_add(total_loss, loss_value)
  loss_count_update = tf.assign_add(loss_count, 1)
  loss_op = total_loss / tf.cast(loss_count, tf.float32)
  return [total_loss_update, loss_count_update], loss_op

def create_app():
    """Create a Qt application."""
    global QT_APP
    QT_APP = QApplication.instance()
    if QT_APP is None:  # pragma: no cover
        QT_APP = QApplication(sys.argv)
    return QT_APP

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

def unique_items(seq):
    """Return the unique items from iterable *seq* (in order)."""
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]

def done(self, result):
        """save the geometry before dialog is close to restore it later"""
        self._geometry = self.geometry()
        QtWidgets.QDialog.done(self, result)

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

def resize(self, width, height):
        """
        Pyqt specific resize callback.
        """
        if not self.fbo:
            return

        # pyqt reports sizes in actual buffer size
        self.width = width // self.widget.devicePixelRatio()
        self.height = height // self.widget.devicePixelRatio()
        self.buffer_width = width
        self.buffer_height = height

        super().resize(width, height)

def unique_(self, col):
        """
        Returns unique values in a column
        """
        try:
            df = self.df.drop_duplicates(subset=[col], inplace=False)
            return list(df[col])
        except Exception as e:
            self.err(e, "Can not select unique data")

def urlencoded(body, charset='ascii', **kwargs):
    """Converts query strings into native Python objects"""
    return parse_query_string(text(body, charset=charset), False)

def array_size(x, axis):
  """Calculate the size of `x` along `axis` dimensions only."""
  axis_shape = x.shape if axis is None else tuple(x.shape[a] for a in axis)
  return max(numpy.prod(axis_shape), 1)

def deinit(self):
        """Deinitialises the PulseIn and releases any hardware and software
        resources for reuse."""
        # Clean up after ourselves
        self._process.terminate()
        procs.remove(self._process)
        self._mq.remove()
        queues.remove(self._mq)

def money(min=0, max=10):
    """Return a str of decimal with two digits after a decimal mark."""
    value = random.choice(range(min * 100, max * 100))
    return "%1.2f" % (float(value) / 100)

def wait_until_exit(self):
        """ Wait until all the threads are finished.

        """
        [t.join() for t in self.threads]

        self.threads = list()

def _comment(string):
    """return string as a comment"""
    lines = [line.strip() for line in string.splitlines()]
    return "# " + ("%s# " % linesep).join(lines)

def exec_rabbitmqctl(self, command, args=[], rabbitmqctl_opts=['-q']):
        """
        Execute a ``rabbitmqctl`` command inside a running container.

        :param command: the command to run
        :param args: a list of args for the command
        :param rabbitmqctl_opts:
            a list of extra options to pass to ``rabbitmqctl``
        :returns: a tuple of the command exit code and output
        """
        cmd = ['rabbitmqctl'] + rabbitmqctl_opts + [command] + args
        return self.inner().exec_run(cmd)

def out(self, output, newline=True):
        """Outputs a string to the console (stdout)."""
        click.echo(output, nl=newline)

def gen_random_string(str_len):
    """ generate random string with specified length
    """
    return ''.join(
        random.choice(string.ascii_letters + string.digits) for _ in range(str_len))

def set_global(node: Node, key: str, value: Any):
    """Adds passed value to node's globals"""
    node.node_globals[key] = value

def sometimesish(fn):
    """
    Has a 50/50 chance of calling a function
    """
    def wrapped(*args, **kwargs):
        if random.randint(1, 2) == 1:
            return fn(*args, **kwargs)

    return wrapped

def url(self):
        """ The url of this window """
        with switch_window(self._browser, self.name):
            return self._browser.url

def uniform_noise(points):
    """Init a uniform noise variable."""
    return np.random.rand(1) * np.random.uniform(points, 1) \
        + random.sample([2, -2], 1)

def runiform(lower, upper, size=None):
    """
    Random uniform variates.
    """
    return np.random.uniform(lower, upper, size)

def generate_hash(filepath):
    """Public function that reads a local file and generates a SHA256 hash digest for it"""
    fr = FileReader(filepath)
    data = fr.read_bin()
    return _calculate_sha256(data)

def SampleSum(dists, n):
    """Draws a sample of sums from a list of distributions.

    dists: sequence of Pmf or Cdf objects
    n: sample size

    returns: new Pmf of sums
    """
    pmf = MakePmfFromList(RandomSum(dists) for i in xrange(n))
    return pmf

def _string_hash(s):
    """String hash (djb2) with consistency between py2/py3 and persistency between runs (unlike `hash`)."""
    h = 5381
    for c in s:
        h = h * 33 + ord(c)
    return h

def _interval_to_bound_points(array):
    """
    Helper function which returns an array
    with the Intervals' boundaries.
    """

    array_boundaries = np.array([x.left for x in array])
    array_boundaries = np.concatenate(
        (array_boundaries, np.array([array[-1].right])))

    return array_boundaries

def highlight_words(string, keywords, cls_name='highlighted'):
    """ Given an list of words, this function highlights the matched words in the given string. """

    if not keywords:
        return string
    if not string:
        return ''
    include, exclude = get_text_tokenizer(keywords)
    highlighted = highlight_text(include, string, cls_name, words=True)
    return highlighted

def get_env_default(self, variable, default):
        """
        Fetch environment variables, returning a default if not found
        """
        if variable in os.environ:
            env_var = os.environ[variable]
        else:
            env_var = default
        return env_var

def _normal_prompt(self):
        """
        Flushes the prompt before requesting the input

        :return: The command line
        """
        sys.stdout.write(self.__get_ps1())
        sys.stdout.flush()
        return safe_input()

def read_string(buff, byteorder='big'):
    """Read a string from a file-like object."""
    length = read_numeric(USHORT, buff, byteorder)
    return buff.read(length).decode('utf-8')

def api_home(request, key=None, hproPk=None):
    """Show the home page for the API with all methods"""

    if not check_api_key(request, key, hproPk):
        return HttpResponseForbidden

    return render_to_response('plugIt/api.html', {}, context_instance=RequestContext(request))

def get_as_string(self, s3_path, encoding='utf-8'):
        """
        Get the contents of an object stored in S3 as string.

        :param s3_path: URL for target S3 location
        :param encoding: Encoding to decode bytes to string
        :return: File contents as a string
        """
        content = self.get_as_bytes(s3_path)
        return content.decode(encoding)

def backward_delete_word(self, e): # (Control-Rubout)
        u"""Delete the character behind the cursor. A numeric argument means
        to kill the characters instead of deleting them."""
        self.l_buffer.backward_delete_word(self.argument_reset)
        self.finalize()

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

def drag_and_drop(self, droppable):
        """
        Performs drag a element to another elmenet.

        Currently works only on Chrome driver.
        """
        self.scroll_to()
        ActionChains(self.parent.driver).drag_and_drop(self._element, droppable._element).perform()

def json_iter (path):
    """
    iterator for JSON-per-line in a file pattern
    """
    with open(path, 'r') as f:
        for line in f.readlines():
            yield json.loads(line)

def do_EOF(self, args):
        """Exit on system end of file character"""
        if _debug: ConsoleCmd._debug("do_EOF %r", args)
        return self.do_exit(args)

def _read_json_file(self, json_file):
        """ Helper function to read JSON file as OrderedDict """

        self.log.debug("Reading '%s' JSON file..." % json_file)

        with open(json_file, 'r') as f:
            return json.load(f, object_pairs_hook=OrderedDict)

def format_exception(e):
    """Returns a string containing the type and text of the exception.

    """
    from .utils.printing import fill
    return '\n'.join(fill(line) for line in traceback.format_exception_only(type(e), e))

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

def add_arrow(self, x1, y1, x2, y2, **kws):
        """add arrow to plot"""
        self.panel.add_arrow(x1, y1, x2, y2, **kws)

def readTuple(self, line, n=3):
        """ Reads a tuple of numbers. e.g. vertices, normals or teture coords.
        """
        numbers = [num for num in line.split(' ') if num]
        return [float(num) for num in numbers[1:n + 1]]

def stats(self):
        """ shotcut to pull out useful info for interactive use """
        printDebug("Classes.....: %d" % len(self.all_classes))
        printDebug("Properties..: %d" % len(self.all_properties))

def ReadTif(tifFile):
        """Reads a tif file to a 2D NumPy array"""
        img = Image.open(tifFile)
        img = np.array(img)
        return img

def extract_words(lines):
    """
    Extract from the given iterable of lines the list of words.

    :param lines: an iterable of lines;
    :return: a generator of words of lines.
    """
    for line in lines:
        for word in re.findall(r"\w+", line):
            yield word

def readTuple(self, line, n=3):
        """ Reads a tuple of numbers. e.g. vertices, normals or teture coords.
        """
        numbers = [num for num in line.split(' ') if num]
        return [float(num) for num in numbers[1:n + 1]]

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

def valid_uuid(value):
    """ Check if value is a valid UUID. """

    try:
        uuid.UUID(value, version=4)
        return True
    except (TypeError, ValueError, AttributeError):
        return False

def _read_date_from_string(str1):
    """
    Reads the date from a string in the format YYYY/MM/DD and returns
    :class: datetime.date
    """
    full_date = [int(x) for x in str1.split('/')]
    return datetime.date(full_date[0], full_date[1], full_date[2])

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

def _write_json(file, contents):
    """Write a dict to a JSON file."""
    with open(file, 'w') as f:
        return json.dump(contents, f, indent=2, sort_keys=True)

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

def as_dict(self):
        """Return all child objects in nested dict."""
        dicts = [x.as_dict for x in self.children]
        return {'{0} {1}'.format(self.name, self.value): dicts}

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

def get(self, key):
        """Get a value from the cache.

        Returns None if the key is not in the cache.
        """
        value = redis_conn.get(key)

        if value is not None:
            value = pickle.loads(value)

        return value

def type(self):
        """Returns type of the data for the given FeatureType."""
        if self is FeatureType.TIMESTAMP:
            return list
        if self is FeatureType.BBOX:
            return BBox
        return dict

def __contains__(self, key):
        """Return ``True`` if *key* is present, else ``False``."""
        pickled_key = self._pickle_key(key)
        return bool(self.redis.hexists(self.key, pickled_key))

def get_memory_usage():
    """Gets RAM memory usage

    :return: MB of memory used by this process
    """
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss
    return mem / (1024 * 1024)

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

def decamelise(text):
    """Convert CamelCase to lower_and_underscore."""
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()

def text_width(string, font_name, font_size):
    """Determine with width in pixels of string."""
    return stringWidth(string, fontName=font_name, fontSize=font_size)

def _string_hash(s):
    """String hash (djb2) with consistency between py2/py3 and persistency between runs (unlike `hash`)."""
    h = 5381
    for c in s:
        h = h * 33 + ord(c)
    return h

async def load_unicode(reader):
    """
    Loads UTF8 string
    :param reader:
    :return:
    """
    ivalue = await load_uvarint(reader)
    fvalue = bytearray(ivalue)
    await reader.areadinto(fvalue)
    return str(fvalue, 'utf8')

def make_regex(separator):
    """Utility function to create regexp for matching escaped separators
    in strings.

    """
    return re.compile(r'(?:' + re.escape(separator) + r')?((?:[^' +
                      re.escape(separator) + r'\\]|\\.)+)')

def on_IOError(self, e):
        """ Handle an IOError exception. """

        sys.stderr.write("Error: %s: \"%s\"\n" % (e.strerror, e.filename))

def _namematcher(regex):
    """Checks if a target name matches with an input regular expression."""

    matcher = re_compile(regex)

    def match(target):
        target_name = getattr(target, '__name__', '')
        result = matcher.match(target_name)
        return result

    return match

def error(self, text):
		""" Ajout d'un message de log de type ERROR """
		self.logger.error("{}{}".format(self.message_prefix, text))

def is_valid_email(email):
    """
    Check if email is valid
    """
    pattern = re.compile(r'[\w\.-]+@[\w\.-]+[.]\w+')
    return bool(pattern.match(email))

def int2str(num, radix=10, alphabet=BASE85):
    """helper function for quick base conversions from integers to strings"""
    return NumConv(radix, alphabet).int2str(num)

def select_from_array(cls, array, identifier):
        """Return a region from a numpy array.
        
        :param array: :class:`numpy.ndarray`
        :param identifier: value representing the region to select in the array
        :returns: :class:`jicimagelib.region.Region`
        """

        base_array = np.zeros(array.shape)
        array_coords = np.where(array == identifier)
        base_array[array_coords] = 1

        return cls(base_array)

def to_snake_case(name):
    """ Given a name in camelCase return in snake_case """
    s1 = FIRST_CAP_REGEX.sub(r'\1_\2', name)
    return ALL_CAP_REGEX.sub(r'\1_\2', s1).lower()

def separator(self, menu=None):
        """Add a separator"""
        self.gui.get_menu(menu or self.menu).addSeparator()

def unaccentuate(s):
    """ Replace accentuated chars in string by their non accentuated equivalent. """
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

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

def as_list(callable):
    """Convert a scalar validator in a list validator"""
    @wraps(callable)
    def wrapper(value_iter):
        return [callable(value) for value in value_iter]

    return wrapper

def delete_cell(self,  key):
        """Deletes key cell"""

        try:
            self.code_array.pop(key)

        except KeyError:
            pass

        self.grid.code_array.result_cache.clear()

def _listify(collection):
        """This is a workaround where Collections are no longer iterable
        when using JPype."""
        new_list = []
        for index in range(len(collection)):
            new_list.append(collection[index])
        return new_list

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

def strip_accents(text):
    """
    Strip agents from a string.
    """

    normalized_str = unicodedata.normalize('NFD', text)

    return ''.join([
        c for c in normalized_str if unicodedata.category(c) != 'Mn'])

def log_leave(event, nick, channel):
	"""
	Log a quit or part event.
	"""
	if channel not in pmxbot.config.log_channels:
		return
	ParticipantLogger.store.log(nick, channel, event.type)

def remove_dups(seq):
    """remove duplicates from a sequence, preserving order"""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

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

def not_matching_list(self):
        """
        Return a list of string which don't match the
        given regex.
        """

        pre_result = comp(self.regex)

        return [x for x in self.data if not pre_result.search(str(x))]

def guess_title(basename):
    """ Attempt to guess the title from the filename """

    base, _ = os.path.splitext(basename)
    return re.sub(r'[ _-]+', r' ', base).title()

def filter_list_by_indices(lst, indices):
    """Return a modified list containing only the indices indicated.

    Args:
        lst: Original list of values
        indices: List of indices to keep from the original list

    Returns:
        list: Filtered list of values

    """
    return [x for i, x in enumerate(lst) if i in indices]

def access_to_sympy(self, var_name, access):
        """
        Transform a (multidimensional) variable access to a flattend sympy expression.

        Also works with flat array accesses.
        """
        base_sizes = self.variables[var_name][1]

        expr = sympy.Number(0)

        for dimension, a in enumerate(access):
            base_size = reduce(operator.mul, base_sizes[dimension+1:], sympy.Integer(1))

            expr += base_size*a

        return expr

def md_to_text(content):
    """ Converts markdown content to text """
    text = None
    html = markdown.markdown(content)
    if html:
        text = html_to_text(content)
    return text

def cast_int(x):
    """
    Cast unknown type into integer

    :param any x:
    :return int:
    """
    try:
        x = int(x)
    except ValueError:
        try:
            x = x.strip()
        except AttributeError as e:
            logger_misc.warn("parse_str: AttributeError: String not number or word, {}, {}".format(x, e))
    return x

def do_striptags(value):
    """Strip SGML/XML tags and replace adjacent whitespace by one space.
    """
    if hasattr(value, '__html__'):
        value = value.__html__()
    return Markup(unicode(value)).striptags()

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

def out(self, output, newline=True):
        """Outputs a string to the console (stdout)."""
        click.echo(output, nl=newline)

def unaccentuate(s):
    """ Replace accentuated chars in string by their non accentuated equivalent. """
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def scatterplot_matrix(df, features, downsample_frac=None, figsize=(15, 15)):
    """
    Plot a scatterplot matrix for a list of features, colored by target value.

    Example: `scatterplot_matrix(X, X.columns.tolist(), downsample_frac=0.01)`

    Args:
        df: Pandas dataframe containing the target column (named 'target').
        features: The list of features to include in the correlation plot.
        downsample_frac: Dataframe downsampling rate (0.1 to include 10% of the dataset).
        figsize: The size of the plot.
    """

    if downsample_frac:
        df = df.sample(frac=downsample_frac)

    plt.figure(figsize=figsize)
    sns.pairplot(df[features], hue='target')
    plt.show()

def _loadf(ins):
    """ Loads a floating point value from a memory address.
    If 2nd arg. start with '*', it is always treated as
    an indirect value.
    """
    output = _float_oper(ins.quad[2])
    output.extend(_fpush())
    return output

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

def _histplot_op(ax, data, **kwargs):
    """Add a histogram for the data to the axes."""
    bins = get_bins(data)
    ax.hist(data, bins=bins, align="left", density=True, **kwargs)
    return ax

def _repr_strip(mystring):
    """
    Returns the string without any initial or final quotes.
    """
    r = repr(mystring)
    if r.startswith("'") and r.endswith("'"):
        return r[1:-1]
    else:
        return r

def prox_zero(X, step):
    """Proximal operator to project onto zero
    """
    return np.zeros(X.shape, dtype=X.dtype)

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

def heappop_max(heap):
    """Maxheap version of a heappop."""
    lastelt = heap.pop()    # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        _siftup_max(heap, 0)
        return returnitem
    return lastelt

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

def build_output(self, fout):
        """Squash self.out into string.

        Join every line in self.out with a new line and write the
        result to the output file.
        """
        fout.write('\n'.join([s for s in self.out]))

def normalize_time(timestamp):
    """Normalize time in arbitrary timezone to UTC naive object."""
    offset = timestamp.utcoffset()
    if offset is None:
        return timestamp
    return timestamp.replace(tzinfo=None) - offset

def string_input(prompt=''):
    """Python 3 input()/Python 2 raw_input()"""
    v = sys.version[0]
    if v == '3':
        return input(prompt)
    else:
        return raw_input(prompt)

def _split_comma_separated(string):
    """Return a set of strings."""
    return set(text.strip() for text in string.split(',') if text.strip())

def write_file(filename, content):
    """Create the file with the given content"""
    print 'Generating {0}'.format(filename)
    with open(filename, 'wb') as out_f:
        out_f.write(content)

def replace_all(text, dic):
    """Takes a string and dictionary. replaces all occurrences of i with j"""

    for i, j in dic.iteritems():
        text = text.replace(i, j)
    return text

def set_font_size(self, size):
        """Convenience method for just changing font size."""
        if self.font.font_size == size:
            pass
        else:
            self.font._set_size(size)

def myreplace(astr, thefind, thereplace):
    """in string astr replace all occurences of thefind with thereplace"""
    alist = astr.split(thefind)
    new_s = alist.split(thereplace)
    return new_s

def __call__(self, factory_name, *args, **kwargs):
        """Create object."""
        return self.factories[factory_name](*args, **kwargs)

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

def to_camel(s):
    """
    :param string s: under_scored string to be CamelCased
    :return: CamelCase version of input
    :rtype: str
    """
    # r'(?!^)_([a-zA-Z]) original regex wasn't process first groups
    return re.sub(r'_([a-zA-Z])', lambda m: m.group(1).upper(), '_' + s)

def unescape_all(string):
    """Resolve all html entities to their corresponding unicode character"""
    def escape_single(matchobj):
        return _unicode_for_entity_with_name(matchobj.group(1))
    return entities.sub(escape_single, string)

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

def multi_replace(instr, search_list=[], repl_list=None):
    """
    Does a string replace with a list of search and replacements

    TODO: rename
    """
    repl_list = [''] * len(search_list) if repl_list is None else repl_list
    for ser, repl in zip(search_list, repl_list):
        instr = instr.replace(ser, repl)
    return instr

def replaceNewlines(string, newlineChar):
	"""There's probably a way to do this with string functions but I was lazy.
		Replace all instances of \r or \n in a string with something else."""
	if newlineChar in string:
		segments = string.split(newlineChar)
		string = ""
		for segment in segments:
			string += segment
	return string

def locate(command, on):
    """Locate the command's man page."""
    location = find_page_location(command, on)
    click.echo(location)

def replace_list(items, match, replacement):
    """Replaces occurrences of a match string in a given list of strings and returns
    a list of new strings. The match string can be a regex expression.

    Args:
        items (list):       the list of strings to modify.
        match (str):        the search expression.
        replacement (str):  the string to replace with.
    """
    return [replace(item, match, replacement) for item in items]

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

def myreplace(astr, thefind, thereplace):
    """in string astr replace all occurences of thefind with thereplace"""
    alist = astr.split(thefind)
    new_s = alist.split(thereplace)
    return new_s

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

def _replace_token_range(tokens, start, end, replacement):
    """For a range indicated from start to end, replace with replacement."""
    tokens = tokens[:start] + replacement + tokens[end:]
    return tokens

def copy_and_update(dictionary, update):
    """Returns an updated copy of the dictionary without modifying the original"""
    newdict = dictionary.copy()
    newdict.update(update)
    return newdict

def check_precomputed_distance_matrix(X):
    """Perform check_array(X) after removing infinite values (numpy.inf) from the given distance matrix.
    """
    tmp = X.copy()
    tmp[np.isinf(tmp)] = 1
    check_array(tmp)

def list_move_to_front(l,value='other'):
    """if the value is in the list, move it to the front and return it."""
    l=list(l)
    if value in l:
        l.remove(value)
        l.insert(0,value)
    return l

def raise_for_not_ok_status(response):
    """
    Raises a `requests.exceptions.HTTPError` if the response has a non-200
    status code.
    """
    if response.code != OK:
        raise HTTPError('Non-200 response code (%s) for url: %s' % (
            response.code, uridecode(response.request.absoluteURI)))

    return response

def onRightUp(self, event=None):
        """ right button up: put back to cursor mode"""
        if event is None:
            return
        self.cursor_mode_action('rightup', event=event)
        self.ForwardEvent(event=event.guiEvent)

def HttpResponse401(request, template=KEY_AUTH_401_TEMPLATE,
content=KEY_AUTH_401_CONTENT, content_type=KEY_AUTH_401_CONTENT_TYPE):
    """
    HTTP response for not-authorized access (status code 403)
    """
    return AccessFailedResponse(request, template, content, content_type, status=401)

def normal_noise(points):
    """Init a noise variable."""
    return np.random.rand(1) * np.random.randn(points, 1) \
        + random.sample([2, -2], 1)

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

def min_max_normalize(img):
    """Centre and normalize a given array.

    Parameters:
    ----------
    img: np.ndarray

    """

    min_img = img.min()
    max_img = img.max()

    return (img - min_img) / (max_img - min_img)

def rest_put_stream(self, url, stream, headers=None, session=None, verify=True, cert=None):
        """
        Perform a chunked PUT request to url with requests.session
        This is specifically to upload files.
        """
        res = session.put(url, headers=headers, data=stream, verify=verify, cert=cert)
        return res.text, res.status_code

def _fast_read(self, infile):
        """Function for fast reading from sensor files."""
        infile.seek(0)
        return(int(infile.read().decode().strip()))

def rest_put_stream(self, url, stream, headers=None, session=None, verify=True, cert=None):
        """
        Perform a chunked PUT request to url with requests.session
        This is specifically to upload files.
        """
        res = session.put(url, headers=headers, data=stream, verify=verify, cert=cert)
        return res.text, res.status_code

def from_file(cls, path, encoding, dialect, fields, converters, field_index):
        """Read delimited text from a text file."""

        return cls(open(path, 'r', encoding=encoding), dialect, fields, converters, field_index)

def _reset_bind(self):
        """Internal utility function to reset binding."""
        self.binded = False
        self._buckets = {}
        self._curr_module = None
        self._curr_bucket_key = None

def get_image(self, source):
        """
        Given a file-like object, loads it up into a PIL.Image object
        and returns it.

        :param file source: A file-like object to load the image from.
        :rtype: PIL.Image
        :returns: The loaded image.
        """
        buf = StringIO(source.read())
        return Image.open(buf)

def reset_params(self):
        """Reset all parameters to their default values."""
        self.__params = dict([p, None] for p in self.param_names)
        self.set_params(self.param_defaults)

def covstr(s):
  """ convert string to int or float. """
  try:
    ret = int(s)
  except ValueError:
    ret = float(s)
  return ret

def im2mat(I):
    """Converts and image to matrix (one pixel per line)"""
    return I.reshape((I.shape[0] * I.shape[1], I.shape[2]))

def _parse_return(cls, result):
        """Extract the result, return value and context from a result object
        """

        return_value = None
        success = result['result']
        context = result['context']

        if 'return_value' in result:
            return_value = result['return_value']

        return success, return_value, context

def unbroadcast_numpy_to(array, shape):
  """Reverse the broadcasting operation.

  Args:
    array: An array.
    shape: A shape that could have been broadcasted to the shape of array.

  Returns:
    Array with dimensions summed to match `shape`.
  """
  axis = create_unbroadcast_axis(shape, numpy.shape(array))
  return numpy.reshape(numpy.sum(array, axis=axis), shape)

def seconds(num):
    """
    Pause for this many seconds
    """
    now = pytime.time()
    end = now + num
    until(end)

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

def pause(self):
        """Pause the music"""
        mixer.music.pause()
        self.pause_time = self.get_time()
        self.paused = True

def HttpResponse401(request, template=KEY_AUTH_401_TEMPLATE,
content=KEY_AUTH_401_CONTENT, content_type=KEY_AUTH_401_CONTENT_TYPE):
    """
    HTTP response for not-authorized access (status code 403)
    """
    return AccessFailedResponse(request, template, content, content_type, status=401)

def load(self, filename='classifier.dump'):
        """
        Unpickles the classifier used
        """
        ifile = open(filename, 'r+')
        self.classifier = pickle.load(ifile)
        ifile.close()

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

def confusion_matrix(self):
        """Confusion matrix plot
        """
        return plot.confusion_matrix(self.y_true, self.y_pred,
                                     self.target_names, ax=_gen_ax())

def _get_data(self):
        """
        Extracts the session data from cookie.
        """
        cookie = self.adapter.cookies.get(self.name)
        return self._deserialize(cookie) if cookie else {}

def plot(self):
        """Plot the empirical histogram versus best-fit distribution's PDF."""
        plt.plot(self.bin_edges, self.hist, self.bin_edges, self.best_pdf)

def restore_default_settings():
    """ Restore settings to default values. 
    """
    global __DEFAULTS
    __DEFAULTS.CACHE_DIR = defaults.CACHE_DIR
    __DEFAULTS.SET_SEED = defaults.SET_SEED
    __DEFAULTS.SEED = defaults.SEED
    logging.info('Settings reverted to their default values.')

def to_json(data):
    """Return data as a JSON string."""
    return json.dumps(data, default=lambda x: x.__dict__, sort_keys=True, indent=4)

def get_property(self, filename):
        """Opens the file and reads the value"""

        with open(self.filepath(filename)) as f:
            return f.read().strip()

def __repr__(self):
        """Return list-lookalike of representation string of objects"""
        strings = []
        for currItem in self:
            strings.append("%s" % currItem)
        return "(%s)" % (", ".join(strings))

def grandparent_path(self):
        """ return grandparent's path string """
        return os.path.basename(os.path.join(self.path, '../..'))

def out(self, output, newline=True):
        """Outputs a string to the console (stdout)."""
        click.echo(output, nl=newline)

def _find(string, sub_string, start_index):
    """Return index of sub_string in string.

    Raise TokenError if sub_string is not found.
    """
    result = string.find(sub_string, start_index)
    if result == -1:
        raise TokenError("expected '{0}'".format(sub_string))
    return result

def _attrprint(d, delimiter=', '):
    """Print a dictionary of attributes in the DOT format"""
    return delimiter.join(('"%s"="%s"' % item) for item in sorted(d.items()))

def grandparent_path(self):
        """ return grandparent's path string """
        return os.path.basename(os.path.join(self.path, '../..'))

def series_index(self, series):
        """
        Return the integer index of *series* in this sequence.
        """
        for idx, s in enumerate(self):
            if series is s:
                return idx
        raise ValueError('series not in chart data object')

def value(self):
        """Value of property."""
        if self._prop.fget is None:
            raise AttributeError('Unable to read attribute')
        return self._prop.fget(self._obj)

def _run_parallel_process_with_profiling(self, start_path, stop_path, queue, filename):
        """
        wrapper for usage of profiling
        """
        runctx('Engine._run_parallel_process(self,  start_path, stop_path, queue)', globals(), locals(), filename)

def _run_parallel_process_with_profiling(self, start_path, stop_path, queue, filename):
        """
        wrapper for usage of profiling
        """
        runctx('Engine._run_parallel_process(self,  start_path, stop_path, queue)', globals(), locals(), filename)

def reduce_fn(x):
    """
    Aggregation function to get the first non-zero value.
    """
    values = x.values if pd and isinstance(x, pd.Series) else x
    for v in values:
        if not is_nan(v):
            return v
    return np.NaN

def reraise(error):
    """Re-raises the error that was processed by prepare_for_reraise earlier."""
    if hasattr(error, "_type_"):
        six.reraise(type(error), error, error._traceback)
    raise error

def round_sig(x, sig):
    """Round the number to the specified number of significant figures"""
    return round(x, sig - int(floor(log10(abs(x)))) - 1)

def list(self):
        """position in 3d space"""
        return [self._pos3d.x, self._pos3d.y, self._pos3d.z]

def round_to_x_digits(number, digits):
    """
    Returns 'number' rounded to 'digits' digits.
    """
    return round(number * math.pow(10, digits)) / math.pow(10, digits)

def handle_exception(error):
        """Simple method for handling exceptions raised by `PyBankID`.

        :param flask_pybankid.FlaskPyBankIDError error: The exception to handle.
        :return: The exception represented as a dictionary.
        :rtype: dict

        """
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

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

def _stdin_(p):
    """Takes input from user. Works for Python 2 and 3."""
    _v = sys.version[0]
    return input(p) if _v is '3' else raw_input(p)

def set(self, f):
        """Call a function after a delay, unless another function is set
        in the meantime."""
        self.stop()
        self._create_timer(f)
        self.start()

def __del__(self):
        """Cleanup the session if it was created here"""
        if self._cleanup_session:
            self._session.loop.run_until_complete(self._session.close())

def sleep(self, time):
        """
        Perform an asyncio sleep for the time specified in seconds. T
        his method should be used in place of time.sleep()

        :param time: time in seconds
        :returns: No return value
        """
        try:
            task = asyncio.ensure_future(self.core.sleep(time))
            self.loop.run_until_complete(task)

        except asyncio.CancelledError:
            pass
        except RuntimeError:
            pass

async def _thread_coro(self, *args):
        """ Coroutine called by MapAsync. It's wrapping the call of
        run_in_executor to run the synchronous function as thread """
        return await self._loop.run_in_executor(
            self._executor, self._function, *args)

def get_randomized_guid_sample(self, item_count):
        """ Fetch a subset of randomzied GUIDs from the whitelist """
        dataset = self.get_whitelist()
        random.shuffle(dataset)
        return dataset[:item_count]

def safe_mkdir_for(path, clean=False):
  """Ensure that the parent directory for a file is present.

  If it's not there, create it. If it is, no-op.
  """
  safe_mkdir(os.path.dirname(path), clean=clean)

def downsample(array, k):
    """Choose k random elements of array."""
    length = array.shape[0]
    indices = random.sample(xrange(length), k)
    return array[indices]

def kill_mprocess(process):
    """kill process
    Args:
        process - Popen object for process
    """
    if process and proc_alive(process):
        process.terminate()
        process.communicate()
    return not proc_alive(process)

def read_utf8(fh, byteorder, dtype, count, offsetsize):
    """Read tag data from file and return as unicode string."""
    return fh.read(count).decode('utf-8')

def save_cache(data, filename):
    """Save cookies to a file."""
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle)

def standard_input():
    """Generator that yields lines from standard input."""
    with click.get_text_stream("stdin") as stdin:
        while stdin.readable():
            line = stdin.readline()
            if line:
                yield line.strip().encode("utf-8")

def convert_camel_case_to_snake_case(name):
    """Convert CamelCase to snake_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def load_graph_from_rdf(fname):
    """ reads an RDF file into a graph """
    print("reading RDF from " + fname + "....")
    store = Graph()
    store.parse(fname, format="n3")
    print("Loaded " + str(len(store)) + " tuples")
    return store

def pickle_save(thing,fname):
    """save something to a pickle file"""
    pickle.dump(thing, open(fname,"wb"),pickle.HIGHEST_PROTOCOL)
    return thing

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

def html_to_text(content):
    """ Converts html content to plain text """
    text = None
    h2t = html2text.HTML2Text()
    h2t.ignore_links = False
    text = h2t.handle(content)
    return text

def scipy_sparse_to_spmatrix(A):
    """Efficient conversion from scipy sparse matrix to cvxopt sparse matrix"""
    coo = A.tocoo()
    SP = spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=A.shape)
    return SP

def path(self):
        """
        Return the path always without the \\?\ prefix.
        """
        path = super(WindowsPath2, self).path
        if path.startswith("\\\\?\\"):
            return path[4:]
        return path

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

def normalize_value(text):
    """
    This removes newlines and multiple spaces from a string.
    """
    result = text.replace('\n', ' ')
    result = re.subn('[ ]{2,}', ' ', result)[0]
    return result

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

def clean_axis(axis):
    """Remove ticks, tick labels, and frame from axis"""
    axis.get_xaxis().set_ticks([])
    axis.get_yaxis().set_ticks([])
    for spine in list(axis.spines.values()):
        spine.set_visible(False)

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

def indentsize(line):
    """Return the indent size, in spaces, at the start of a line of text."""
    expline = string.expandtabs(line)
    return len(expline) - len(string.lstrip(expline))

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

def set_scrollregion(self, event=None):
        """ Set the scroll region on the canvas"""
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

def get_line_ending(line):
    """Return line ending."""
    non_whitespace_index = len(line.rstrip()) - len(line)
    if not non_whitespace_index:
        return ''
    else:
        return line[non_whitespace_index:]

def split_every(n, iterable):
    """Returns a generator that spits an iteratable into n-sized chunks. The last chunk may have
    less than n elements.

    See http://stackoverflow.com/a/22919323/503377."""
    items = iter(iterable)
    return itertools.takewhile(bool, (list(itertools.islice(items, n)) for _ in itertools.count()))

def remove_legend(ax=None):
    """Remove legend for axes or gca.

    See http://osdir.com/ml/python.matplotlib.general/2005-07/msg00285.html
    """
    from pylab import gca, draw
    if ax is None:
        ax = gca()
    ax.legend_ = None
    draw()

def date(start, end):
    """Get a random date between two dates"""

    stime = date_to_timestamp(start)
    etime = date_to_timestamp(end)

    ptime = stime + random.random() * (etime - stime)

    return datetime.date.fromtimestamp(ptime)

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

def selecttrue(table, field, complement=False):
    """Select rows where the given field evaluates `True`."""

    return select(table, field, lambda v: bool(v), complement=complement)

def format_screen(strng):
    """Format a string for screen printing.

    This removes some latex-type format codes."""
    # Paragraph continue
    par_re = re.compile(r'\\$',re.MULTILINE)
    strng = par_re.sub('',strng)
    return strng

def strip_querystring(url):
    """Remove the querystring from the end of a URL."""
    p = six.moves.urllib.parse.urlparse(url)
    return p.scheme + "://" + p.netloc + p.path

def selecttrue(table, field, complement=False):
    """Select rows where the given field evaluates `True`."""

    return select(table, field, lambda v: bool(v), complement=complement)

def dedupe_list(seq):
    """
    Utility function to remove duplicates from a list
    :param seq: The sequence (list) to deduplicate
    :return: A list with original duplicates removed
    """
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]

def selecttrue(table, field, complement=False):
    """Select rows where the given field evaluates `True`."""

    return select(table, field, lambda v: bool(v), complement=complement)

def _str_to_list(s):
    """Converts a comma separated string to a list"""
    _list = s.split(",")
    return list(map(lambda i: i.lstrip(), _list))

def _pick_attrs(attrs, keys):
    """ Return attrs with keys in keys list
    """
    return dict((k, v) for k, v in attrs.items() if k in keys)

def fix_dashes(string):
    """Fix bad Unicode special dashes in string."""
    string = string.replace(u'\u05BE', '-')
    string = string.replace(u'\u1806', '-')
    string = string.replace(u'\u2E3A', '-')
    string = string.replace(u'\u2E3B', '-')
    string = unidecode(string)
    return re.sub(r'--+', '-', string)

def _indexes(arr):
    """ Returns the list of all indexes of the given array.

    Currently works for one and two-dimensional arrays

    """
    myarr = np.array(arr)
    if myarr.ndim == 1:
        return list(range(len(myarr)))
    elif myarr.ndim == 2:
        return tuple(itertools.product(list(range(arr.shape[0])),
                                       list(range(arr.shape[1]))))
    else:
        raise NotImplementedError('Only supporting arrays of dimension 1 and 2 as yet.')

def sanitize_word(s):
    """Remove non-alphanumerical characters from metric word.
    And trim excessive underscores.
    """
    s = re.sub('[^\w-]+', '_', s)
    s = re.sub('__+', '_', s)
    return s.strip('_')

def copy(self):
        """Create an identical (deep) copy of this element."""
        result = self.space.element()
        result.assign(self)
        return result

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

async def send(self, data):
        """ Add data to send queue. """
        self.writer.write(data)
        await self.writer.drain()

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

def cli(yamlfile, format, output):
    """ Generate an OWL representation of a biolink model """
    print(OwlSchemaGenerator(yamlfile, format).serialize(output=output))

def get_stripped_file_lines(filename):
    """
    Return lines of a file with whitespace removed
    """
    try:
        lines = open(filename).readlines()
    except FileNotFoundError:
        fatal("Could not open file: {!r}".format(filename))

    return [line.strip() for line in lines]

def serialize(self, value, **kwargs):
        """Serialize every item of the list."""
        return [self.item_type.serialize(val, **kwargs) for val in value]

def draw_image(self, ax, image):
        """Process a matplotlib image object and call renderer.draw_image"""
        self.renderer.draw_image(imdata=utils.image_to_base64(image),
                                 extent=image.get_extent(),
                                 coordinates="data",
                                 style={"alpha": image.get_alpha(),
                                        "zorder": image.get_zorder()},
                                 mplobj=image)

def na_if(series, *values):
    """
    If values in a series match a specified value, change them to `np.nan`.

    Args:
        series: Series or vector, often symbolic.
        *values: Value(s) to convert to `np.nan` in the series.
    """

    series = pd.Series(series)
    series[series.isin(values)] = np.nan
    return series

def get_input(input_func, input_str):
    """
    Get input from the user given an input function and an input string
    """
    val = input_func("Please enter your {0}: ".format(input_str))
    while not val or not len(val.strip()):
        val = input_func("You didn't enter a valid {0}, please try again: ".format(input_str))
    return val

def to_monthly(series, method='ffill', how='end'):
    """
    Convenience method that wraps asfreq_actual
    with 'M' param (method='ffill', how='end').
    """
    return series.asfreq_actual('M', method=method, how=how)

def internal_reset(self):
        """
        internal state reset.
        used e.g. in unittests
        """
        log.critical("PIA internal_reset()")
        self.empty_key_toggle = True
        self.current_input_char = None
        self.input_repead = 0

def dispatch(self):
    """Wraps the dispatch method to add session support."""
    try:
      webapp2.RequestHandler.dispatch(self)
    finally:
      self.session_store.save_sessions(self.response)

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

def _cpu(self):
        """Record CPU usage."""
        value = int(psutil.cpu_percent())
        set_metric("cpu", value, category=self.category)
        gauge("cpu", value)

def plot3d_init(fignum):
    """
    initializes 3D plot
    """
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(fignum)
    ax = fig.add_subplot(111, projection='3d')
    return ax

def roundClosestValid(val, res, decimals=None):
        """ round to closest resolution """
        if decimals is None and "." in str(res):
            decimals = len(str(res).split('.')[1])

        return round(round(val / res) * res, decimals)

def setdefaults(dct, defaults):
    """Given a target dct and a dict of {key:default value} pairs,
    calls setdefault for all of those pairs."""
    for key in defaults:
        dct.setdefault(key, defaults[key])

    return dct

def _transform_triple_numpy(x):
    """Transform triple index into a 1-D numpy array."""
    return np.array([x.head, x.relation, x.tail], dtype=np.int64)

def ensure_hbounds(self):
        """Ensure the cursor is within horizontal screen bounds."""
        self.cursor.x = min(max(0, self.cursor.x), self.columns - 1)

def unpack2D(_x):
    """
        Helper function for splitting 2D data into x and y component to make
        equations simpler
    """
    _x = np.atleast_2d(_x)
    x = _x[:, 0]
    y = _x[:, 1]
    return x, y

def security(self):
        """Print security object information for a pdf document"""
        return {k: v for i in self.pdf.resolvedObjects.items() for k, v in i[1].items()}

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

def get_longest_orf(orfs):
    """Find longest ORF from the given list of ORFs."""
    sorted_orf = sorted(orfs, key=lambda x: len(x['sequence']), reverse=True)[0]
    return sorted_orf

def set_xlimits(self, min=None, max=None):
        """Set limits for the x-axis.

        :param min: minimum value to be displayed.  If None, it will be
            calculated.
        :param max: maximum value to be displayed.  If None, it will be
            calculated.

        """
        self.limits['xmin'] = min
        self.limits['xmax'] = max

def similarity_transformation(rot, mat):
    """ R x M x R^-1 """
    return np.dot(rot, np.dot(mat, np.linalg.inv(rot)))

def sha1(s):
    """ Returns a sha1 of the given string
    """
    h = hashlib.new('sha1')
    h.update(s)
    return h.hexdigest()

def save_pdf(path):
  """
  Saves a pdf of the current matplotlib figure.

  :param path: str, filepath to save to
  """

  pp = PdfPages(path)
  pp.savefig(pyplot.gcf())
  pp.close()

def open(self, flag="c"):
        """Open handle

        set protocol=2 to fix python3

        .. versionadded:: 1.3.1
        """
        return shelve.open(os.path.join(gettempdir(), self.index), flag=flag, protocol=2)

def save_pdf(path):
  """
  Saves a pdf of the current matplotlib figure.

  :param path: str, filepath to save to
  """

  pp = PdfPages(path)
  pp.savefig(pyplot.gcf())
  pp.close()

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

def _save_file(self, filename, contents):
        """write the html file contents to disk"""
        with open(filename, 'w') as f:
            f.write(contents)

def normalise_key(self, key):
        """Make sure key is a valid python attribute"""
        key = key.replace('-', '_')
        if key.startswith("noy_"):
            key = key[4:]
        return key

def is_int_vector(l):
    r"""Checks if l is a numpy array of integers

    """
    if isinstance(l, np.ndarray):
        if l.ndim == 1 and (l.dtype.kind == 'i' or l.dtype.kind == 'u'):
            return True
    return False

def getFieldsColumnLengths(self):
        """
        Gets the maximum length of each column in the field table
        """
        nameLen = 0
        descLen = 0
        for f in self.fields:
            nameLen = max(nameLen, len(f['title']))
            descLen = max(descLen, len(f['description']))
        return (nameLen, descLen)

def resetScale(self):
        """Resets the scale on this image. Correctly aligns time scale, undoes manual scaling"""
        self.img.scale(1./self.imgScale[0], 1./self.imgScale[1])
        self.imgScale = (1.,1.)

def format_exc(limit=None):
    """Like print_exc() but return a string. Backport for Python 2.3."""
    try:
        etype, value, tb = sys.exc_info()
        return ''.join(traceback.format_exception(etype, value, tb, limit))
    finally:
        etype = value = tb = None

def bytesize(arr):
    """
    Returns the memory byte size of a Numpy array as an integer.
    """
    byte_size = np.prod(arr.shape) * np.dtype(arr.dtype).itemsize
    return byte_size

def close(self, wait=False):
        """Close session, shutdown pool."""
        self.session.close()
        self.pool.shutdown(wait=wait)

def lengths( self ):
        """
        The cell lengths.

        Args:
            None

        Returns:
            (np.array(a,b,c)): The cell lengths.
        """
        return( np.array( [ math.sqrt( sum( row**2 ) ) for row in self.matrix ] ) )

def softplus(attrs, inputs, proto_obj):
    """Applies the sofplus activation function element-wise to the input."""
    new_attrs = translation_utils._add_extra_attributes(attrs, {'act_type' : 'softrelu'})
    return 'Activation', new_attrs, inputs

def rand_elem(seq, n=None):
    """returns a random element from seq n times. If n is None, it continues indefinitly"""
    return map(random.choice, repeat(seq, n) if n is not None else repeat(seq))

def basic_word_sim(word1, word2):
    """
    Simple measure of similarity: Number of letters in common / max length
    """
    return sum([1 for c in word1 if c in word2]) / max(len(word1), len(word2))

def other_ind(self):
        """last row or column of square A"""
        return np.full(self.n_min, self.size - 1, dtype=np.int)

def array_bytes(array):
    """ Estimates the memory of the supplied array in bytes """
    return np.product(array.shape)*np.dtype(array.dtype).itemsize

def filter_list_by_indices(lst, indices):
    """Return a modified list containing only the indices indicated.

    Args:
        lst: Original list of values
        indices: List of indices to keep from the original list

    Returns:
        list: Filtered list of values

    """
    return [x for i, x in enumerate(lst) if i in indices]

def impute_data(self,x):
        """Imputes data set containing Nan values"""
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        return imp.fit_transform(x)

def pwm(host, seq, m1, m2, m3, m4):
    """
    Sends control values directly to the engines, overriding control loops.

    Parameters:
    seq -- sequence number
    m1 -- Integer: front left command
    m2 -- Integer: front right command
    m3 -- Integer: back right command
    m4 -- Integer: back left command
    """
    at(host, 'PWM', seq, [m1, m2, m3, m4])

def iget_list_column_slice(list_, start=None, stop=None, stride=None):
    """ iterator version of get_list_column """
    if isinstance(start, slice):
        slice_ = start
    else:
        slice_ = slice(start, stop, stride)
    return (row[slice_] for row in list_)

def unsort_vector(data, indices_of_increasing):
    """Upermutate 1-D data that is sorted by indices_of_increasing."""
    return numpy.array([data[indices_of_increasing.index(i)] for i in range(len(data))])

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

def unsort_vector(data, indices_of_increasing):
    """Upermutate 1-D data that is sorted by indices_of_increasing."""
    return numpy.array([data[indices_of_increasing.index(i)] for i in range(len(data))])

def main():
    """Ideally we shouldn't lose the first second of events"""
    time.sleep(1)
    with Input() as input_generator:
        for e in input_generator:
            print(repr(e))

def SetValue(self, row, col, value):
        """
        Set value in the pandas DataFrame
        """
        self.dataframe.iloc[row, col] = value

def sort_by_name(self):
        """Sort list elements by name."""
        super(JSSObjectList, self).sort(key=lambda k: k.name)

def parse(self, s):
        """
        Parses a date string formatted like ``YYYY-MM-DD``.
        """
        return datetime.datetime.strptime(s, self.date_format).date()

def sort_key(val):
    """Sort key for sorting keys in grevlex order."""
    return numpy.sum((max(val)+1)**numpy.arange(len(val)-1, -1, -1)*val)

def sort_data(x, y):
    """Sort the data."""
    xy = sorted(zip(x, y))
    x, y = zip(*xy)
    return x, y

def display(self):
        """ Get screen width and height """
        w, h = self.session.window_size()
        return Display(w*self.scale, h*self.scale)

def calc_volume(self, sample: np.ndarray):
        """Find the RMS of the audio"""
        return sqrt(np.mean(np.square(sample)))

def enable_proxy(self, host, port):
        """Enable a default web proxy"""

        self.proxy = [host, _number(port)]
        self.proxy_enabled = True

def scipy_sparse_to_spmatrix(A):
    """Efficient conversion from scipy sparse matrix to cvxopt sparse matrix"""
    coo = A.tocoo()
    SP = spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=A.shape)
    return SP

def set_proxy(proxy_url, transport_proxy=None):
    """Create the proxy to PyPI XML-RPC Server"""
    global proxy, PYPI_URL
    PYPI_URL = proxy_url
    proxy = xmlrpc.ServerProxy(
        proxy_url,
        transport=RequestsTransport(proxy_url.startswith('https://')),
        allow_none=True)

def is_sparse_vector(x):
    """ x is a 2D sparse matrix with it's first shape equal to 1.
    """
    return sp.issparse(x) and len(x.shape) == 2 and x.shape[0] == 1

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

def set_xlimits_widgets(self, set_min=True, set_max=True):
        """Populate axis limits GUI with current plot values."""
        xmin, xmax = self.tab_plot.ax.get_xlim()
        if set_min:
            self.w.x_lo.set_text('{0}'.format(xmin))
        if set_max:
            self.w.x_hi.set_text('{0}'.format(xmax))

def sbessely(x, N):
    """Returns a vector of spherical bessel functions yn:

        x:   The argument.
        N:   values of n will run from 0 to N-1.

    """

    out = np.zeros(N, dtype=np.float64)

    out[0] = -np.cos(x) / x
    out[1] = -np.cos(x) / (x ** 2) - np.sin(x) / x

    for n in xrange(2, N):
        out[n] = ((2.0 * n - 1.0) / x) * out[n - 1] - out[n - 2]

    return out

def log_y_cb(self, w, val):
        """Toggle linear/log scale for Y-axis."""
        self.tab_plot.logy = val
        self.plot_two_columns()

def partition(a, sz): 
    """splits iterables a in equal parts of size sz"""
    return [a[i:i+sz] for i in range(0, len(a), sz)]

def finish_plot():
    """Helper for plotting."""
    plt.legend()
    plt.grid(color='0.7')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def split_on(s, sep=" "):
    """Split s by sep, unless it's inside a quote."""
    pattern = '''((?:[^%s"']|"[^"]*"|'[^']*')+)''' % sep

    return [_strip_speechmarks(t) for t in re.split(pattern, s)[1::2]]

def error(*args):
    """Display error message via stderr or GUI."""
    if sys.stdin.isatty():
        print('ERROR:', *args, file=sys.stderr)
    else:
        notify_error(*args)

def show(self, title=''):
        """
        Display Bloch sphere and corresponding data sets.
        """
        self.render(title=title)
        if self.fig:
            plt.show(self.fig)

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

def partition(a, sz): 
    """splits iterables a in equal parts of size sz"""
    return [a[i:i+sz] for i in range(0, len(a), sz)]

def _split_batches(self, data, batch_size):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size]

def getcolslice(self, blc, trc, inc=[], startrow=0, nrow=-1, rowincr=1):
        """Get a slice from a table column holding arrays.
        (see :func:`table.getcolslice`)"""
        return self._table.getcolslice(self._column, blc, trc, inc, startrow, nrow, rowincr)

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

def join_cols(cols):
    """Join list of columns into a string for a SQL query"""
    return ", ".join([i for i in cols]) if isinstance(cols, (list, tuple, set)) else cols

def solve(A, x):
    """Solves a linear equation system with a matrix of shape (n, n) and an
    array of shape (n, ...). The output has the same shape as the second
    argument.
    """
    # https://stackoverflow.com/a/48387507/353337
    x = numpy.asarray(x)
    return numpy.linalg.solve(A, x.reshape(x.shape[0], -1)).reshape(x.shape)

def join_cols(cols):
    """Join list of columns into a string for a SQL query"""
    return ", ".join([i for i in cols]) if isinstance(cols, (list, tuple, set)) else cols

def MatrixSolve(a, rhs, adj):
    """
    Matrix solve op.
    """
    return np.linalg.solve(a if not adj else _adjoint(a), rhs),

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

def MatrixSolve(a, rhs, adj):
    """
    Matrix solve op.
    """
    return np.linalg.solve(a if not adj else _adjoint(a), rhs),

def clear_all(self):
        """Delete all Labels."""
        logger.info("Clearing ALL Labels and LabelKeys.")
        self.session.query(Label).delete(synchronize_session="fetch")
        self.session.query(LabelKey).delete(synchronize_session="fetch")

def primary_keys_full(cls):
        """Get primary key properties for a SQLAlchemy cls.
        Taken from marshmallow_sqlalchemy
        """
        mapper = cls.__mapper__
        return [
            mapper.get_property_by_column(column)
            for column in mapper.primary_key
        ]

def datetime_from_timestamp(timestamp, content):
    """
    Helper function to add timezone information to datetime,
    so that datetime is comparable to other datetime objects in recent versions
    that now also have timezone information.
    """
    return set_date_tzinfo(
        datetime.fromtimestamp(timestamp),
        tz_name=content.settings.get('TIMEZONE', None))

def createdb():
    """Create database tables from sqlalchemy models"""
    manager.db.engine.echo = True
    manager.db.create_all()
    set_alembic_revision()

def has_permission(user, permission_name):
    """Check if a user has a given permission."""
    if user and user.is_superuser:
        return True

    return permission_name in available_perm_names(user)

def from_pydatetime(cls, pydatetime):
        """
        Creates sql datetime2 object from Python datetime object
        ignoring timezone
        @param pydatetime: Python datetime object
        @return: sql datetime2 object
        """
        return cls(date=Date.from_pydate(pydatetime.date),
                   time=Time.from_pytime(pydatetime.time))

def open_with_encoding(filename, encoding, mode='r'):
    """Return opened file with a specific encoding."""
    return io.open(filename, mode=mode, encoding=encoding,
                   newline='')

def sqliteRowsToDicts(sqliteRows):
    """
    Unpacks sqlite rows as returned by fetchall
    into an array of simple dicts.

    :param sqliteRows: array of rows returned from fetchall DB call
    :return:  array of dicts, keyed by the column names.
    """
    return map(lambda r: dict(zip(r.keys(), r)), sqliteRows)

def split_into_words(s):
  """Split a sentence into list of words."""
  s = re.sub(r"\W+", " ", s)
  s = re.sub(r"[_0-9]+", " ", s)
  return s.split()

def export(defn):
    """Decorator to explicitly mark functions that are exposed in a lib."""
    globals()[defn.__name__] = defn
    __all__.append(defn.__name__)
    return defn

def split_into_sentences(s):
  """Split text into list of sentences."""
  s = re.sub(r"\s+", " ", s)
  s = re.sub(r"[\\.\\?\\!]", "\n", s)
  return s.split("\n")

def server(port):
    """Start the Django dev server."""
    args = ['python', 'manage.py', 'runserver']
    if port:
        args.append(port)
    run.main(args)

def split_into_words(s):
  """Split a sentence into list of words."""
  s = re.sub(r"\W+", " ", s)
  s = re.sub(r"[_0-9]+", " ", s)
  return s.split()

def static_method(cls, f):
        """Decorator which dynamically binds static methods to the model for later use."""
        setattr(cls, f.__name__, staticmethod(f))
        return f

def column_stack_2d(data):
    """Perform column-stacking on a list of 2d data blocks."""
    return list(list(itt.chain.from_iterable(_)) for _ in zip(*data))

def is_static(*p):
    """ A static value (does not change at runtime)
    which is known at compile time
    """
    return all(is_CONST(x) or
               is_number(x) or
               is_const(x)
               for x in p)

def quit(self):
        """ Quits the application (called when the last window is closed)
        """
        logger.debug("ArgosApplication.quit called")
        assert len(self.mainWindows) == 0, \
            "Bug: still {} windows present at application quit!".format(len(self.mainWindows))
        self.qApplication.quit()

def circstd(dts, axis=2):
    """Circular standard deviation"""
    R = np.abs(np.exp(1.0j * dts).mean(axis=axis))
    return np.sqrt(-2.0 * np.log(R))

def setdefault(obj, field, default):
    """Set an object's field to default if it doesn't have a value"""
    setattr(obj, field, getattr(obj, field, default))

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

def save(self, *args, **kwargs):
        """Saves an animation

        A wrapper around :meth:`matplotlib.animation.Animation.save`
        """
        self.timeline.index -= 1  # required for proper starting point for save
        self.animation.save(*args, **kwargs)

def read_stdin():
    """ Read text from stdin, and print a helpful message for ttys. """
    if sys.stdin.isatty() and sys.stdout.isatty():
        print('\nReading from stdin until end of file (Ctrl + D)...')

    return sys.stdin.read()

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

def __init__(self, encoding='utf-8'):
    """Initializes an stdin input reader.

    Args:
      encoding (Optional[str]): input encoding.
    """
    super(StdinInputReader, self).__init__(sys.stdin, encoding=encoding)

def generate_split_tsv_lines(fn, header):
    """Returns dicts with header-keys and psm statistic values"""
    for line in generate_tsv_psms_line(fn):
        yield {x: y for (x, y) in zip(header, line.strip().split('\t'))}

def println(msg):
    """
    Convenience function to print messages on a single line in the terminal
    """
    sys.stdout.write(msg)
    sys.stdout.flush()
    sys.stdout.write('\x08' * len(msg))
    sys.stdout.flush()

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

def nothread_quit(self, arg):
        """ quit command when there's just one thread. """

        self.debugger.core.stop()
        self.debugger.core.execution_status = 'Quit command'
        raise Mexcept.DebuggerQuit

def is_complex(dtype):
  """Returns whether this is a complex floating point type."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'is_complex'):
    return dtype.is_complex
  return np.issubdtype(np.dtype(dtype), np.complex)

def exit(self):
        """Stop the simple WSGI server running the appliation."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

def stop(self, reason=None):
        """Shutdown the service with a reason."""
        self.logger.info('stopping')
        self.loop.stop(pyev.EVBREAK_ALL)

def text_remove_empty_lines(text):
    """
    Whitespace normalization:

      - Strip empty lines
      - Strip trailing whitespace
    """
    lines = [ line.rstrip()  for line in text.splitlines()  if line.strip() ]
    return "\n".join(lines)

def json_response(data, status=200):
    """Return a JsonResponse. Make sure you have django installed first."""
    from django.http import JsonResponse
    return JsonResponse(data=data, status=status, safe=isinstance(data, dict))

def list_i2str(ilist):
    """
    Convert an integer list into a string list.
    """
    slist = []
    for el in ilist:
        slist.append(str(el))
    return slist

def __str__(self):
        """Executes self.function to convert LazyString instance to a real
        str."""
        if not hasattr(self, '_str'):
            self._str=self.function(*self.args, **self.kwargs)
        return self._str

def concat(cls, iterables):
    """
    Similar to #itertools.chain.from_iterable().
    """

    def generator():
      for it in iterables:
        for element in it:
          yield element
    return cls(generator())

def next (self):    # File-like object.

        """This is to support iterators over a file-like object.
        """

        result = self.readline()
        if result == self._empty_buffer:
            raise StopIteration
        return result

def pack_triples_numpy(triples):
    """Packs a list of triple indexes into a 2D numpy array."""
    if len(triples) == 0:
        return np.array([], dtype=np.int64)
    return np.stack(list(map(_transform_triple_numpy, triples)), axis=0)

def fmt_duration(secs):
    """Format a duration in seconds."""
    return ' '.join(fmt.human_duration(secs, 0, precision=2, short=True).strip().split())

def runcoro(async_function):
    """
    Runs an asynchronous function without needing to use await - useful for lambda

    Args:
        async_function (Coroutine): The asynchronous function to run
    """

    future = _asyncio.run_coroutine_threadsafe(async_function, client.loop)
    result = future.result()
    return result

def AmericanDateToEpoch(self, date_str):
    """Take a US format date and return epoch."""
    try:
      epoch = time.strptime(date_str, "%m/%d/%Y")
      return int(calendar.timegm(epoch)) * 1000000
    except ValueError:
      return 0

def to_camel_case(text):
    """Convert to camel case.

    :param str text:
    :rtype: str
    :return:
    """
    split = text.split('_')
    return split[0] + "".join(x.title() for x in split[1:])

def entropy(string):
    """Compute entropy on the string"""
    p, lns = Counter(string), float(len(string))
    return -sum(count/lns * math.log(count/lns, 2) for count in p.values())

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

def str_dict(some_dict):
    """Convert dict of ascii str/unicode to dict of str, if necessary"""
    return {str(k): str(v) for k, v in some_dict.items()}

def notin(arg, values):
    """
    Like isin, but checks whether this expression's value(s) are not
    contained in the passed values. See isin docs for full usage.
    """
    op = ops.NotContains(arg, values)
    return op.to_expr()

def exception_format():
    """
    Convert exception info into a string suitable for display.
    """
    return "".join(traceback.format_exception(
        sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2]
    ))

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

def measure_string(self, text, fontname, fontsize, encoding=0):
        """Measure length of a string for a Base14 font."""
        return _fitz.Tools_measure_string(self, text, fontname, fontsize, encoding)

def visit_BoolOp(self, node):
        """ Return type may come from any boolop operand. """
        return sum((self.visit(value) for value in node.values), [])

def __is__(cls, s):
        """Test if string matches this argument's format."""
        return s.startswith(cls.delims()[0]) and s.endswith(cls.delims()[1])

def instance_contains(container, item):
    """Search into instance attributes, properties and return values of no-args methods."""
    return item in (member for _, member in inspect.getmembers(container))

def unpunctuate(s, *, char_blacklist=string.punctuation):
    """ Remove punctuation from string s. """
    # remove punctuation
    s = "".join(c for c in s if c not in char_blacklist)
    # remove consecutive spaces
    return " ".join(filter(None, s.split(" ")))

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

def text_cleanup(data, key, last_type):
    """ I strip extra whitespace off multi-line strings if they are ready to be stripped!"""
    if key in data and last_type == STRING_TYPE:
        data[key] = data[key].strip()
    return data

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

def to_snake_case(name):
    """ Given a name in camelCase return in snake_case """
    s1 = FIRST_CAP_REGEX.sub(r'\1_\2', name)
    return ALL_CAP_REGEX.sub(r'\1_\2', s1).lower()

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

def fsliceafter(astr, sub):
    """Return the slice after at sub in string astr"""
    findex = astr.find(sub)
    return astr[findex + len(sub):]

def do_restart(self, line):
        """Request that the Outstation perform a cold restart. Command syntax is: restart"""
        self.application.master.Restart(opendnp3.RestartType.COLD, restart_callback)

def fsliceafter(astr, sub):
    """Return the slice after at sub in string astr"""
    findex = astr.find(sub)
    return astr[findex + len(sub):]

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

def bytes_to_c_array(data):
    """
    Make a C array using the given string.
    """
    chars = [
        "'{}'".format(encode_escape(i))
        for i in decode_escape(data)
    ]
    return ', '.join(chars) + ', 0'

def is_string(val):
    """Determines whether the passed value is a string, safe for 2/3."""
    try:
        basestring
    except NameError:
        return isinstance(val, str)
    return isinstance(val, basestring)

def _clean_str(self, s):
        """ Returns a lowercase string with punctuation and bad chars removed
        :param s: string to clean
        """
        return s.translate(str.maketrans('', '', punctuation)).replace('\u200b', " ").strip().lower()

def _print(self, msg, flush=False, end="\n"):
        """Helper function to print connection status messages when in verbose mode."""
        if self._verbose:
            print2(msg, end=end, flush=flush)

def _to_lower_alpha_only(s):
    """Return a lowercased string with non alphabetic chars removed.

    White spaces are not to be removed."""
    s = re.sub(r'\n', ' ',  s.lower())
    return re.sub(r'[^a-z\s]', '', s)

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

def drop_trailing_zeros_decimal(num):
    """ Drops the trailinz zeros from decimal value.
        Returns a string
    """
    out = str(num)
    return out.rstrip('0').rstrip('.') if '.' in out else out

def software_fibonacci(n):
    """ a normal old python function to return the Nth fibonacci number. """
    a, b = 0, 1
    for i in range(n):
        a, b = b, a + b
    return a

def info(self, text):
		""" Ajout d'un message de log de type INFO """
		self.logger.info("{}{}".format(self.message_prefix, text))

def show_xticklabels(self, row, column):
        """Show the x-axis tick labels for a subplot.

        :param row,column: specify the subplot.

        """
        subplot = self.get_subplot_at(row, column)
        subplot.show_xticklabels()

def write_document(doc, fnm):
    """Write a Text document to file.

    Parameters
    ----------
    doc: Text
        The document to save.
    fnm: str
        The filename to save the document
    """
    with codecs.open(fnm, 'wb', 'ascii') as f:
        f.write(json.dumps(doc, indent=2))

def bash(filename):
    """Runs a bash script in the local directory"""
    sys.stdout.flush()
    subprocess.call("bash {}".format(filename), shell=True)

def get_func_name(func):
    """Return a name which includes the module name and function name."""
    func_name = getattr(func, '__name__', func.__class__.__name__)
    module_name = func.__module__

    if module_name is not None:
        module_name = func.__module__
        return '{}.{}'.format(module_name, func_name)

    return func_name

def _finish(self):
        """
        Closes and waits for subprocess to exit.
        """
        if self._process.returncode is None:
            self._process.stdin.flush()
            self._process.stdin.close()
            self._process.wait()
            self.closed = True

def get_input(input_func, input_str):
    """
    Get input from the user given an input function and an input string
    """
    val = input_func("Please enter your {0}: ".format(input_str))
    while not val or not len(val.strip()):
        val = input_func("You didn't enter a valid {0}, please try again: ".format(input_str))
    return val

def correspond(text):
    """Communicate with the child process without closing stdin."""
    subproc.stdin.write(text)
    subproc.stdin.flush()
    return drain()

def highpass(cutoff):
  """
  This strategy uses an exponential approximation for cut-off frequency
  calculation, found by matching the one-pole Laplace lowpass filter
  and mirroring the resulting filter to get a highpass.
  """
  R = thub(exp(cutoff - pi), 2)
  return (1 - R) / (1 + R * z ** -1)

def filter_list_by_indices(lst, indices):
    """Return a modified list containing only the indices indicated.

    Args:
        lst: Original list of values
        indices: List of indices to keep from the original list

    Returns:
        list: Filtered list of values

    """
    return [x for i, x in enumerate(lst) if i in indices]

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

def init_db():
    """
    Drops and re-creates the SQL schema
    """
    db.drop_all()
    db.configure_mappers()
    db.create_all()
    db.session.commit()

def any_contains_any(strings, candidates):
    """Whether any of the strings contains any of the candidates."""
    for string in strings:
        for c in candidates:
            if c in string:
                return True

def maybeparens(lparen, item, rparen):
    """Wrap an item in optional parentheses, only applying them if necessary."""
    return item | lparen.suppress() + item + rparen.suppress()

def example_write_file_to_disk_if_changed():
    """ Try to remove all comments from a file, and save it if changes were made. """
    my_file = FileAsObj('/tmp/example_file.txt')
    my_file.rm(my_file.egrep('^#'))
    if my_file.changed:
        my_file.save()

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

def any_contains_any(strings, candidates):
    """Whether any of the strings contains any of the candidates."""
    for string in strings:
        for c in candidates:
            if c in string:
                return True

def getcolslice(self, blc, trc, inc=[], startrow=0, nrow=-1, rowincr=1):
        """Get a slice from a table column holding arrays.
        (see :func:`table.getcolslice`)"""
        return self._table.getcolslice(self._column, blc, trc, inc, startrow, nrow, rowincr)

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

def top(n, width=WIDTH, style=STYLE):
    """Prints the top row of a table"""
    return hrule(n, width, linestyle=STYLES[style].top)

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

def finish():
    """Print warning about interrupt and empty the job queue."""
    out.warn("Interrupted!")
    for t in threads:
        t.stop()
    jobs.clear()
    out.warn("Waiting for download threads to finish.")

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

def is_scalar(value):
    """Test if the given value is a scalar.

    This function also works with memory mapped array values, in contrast to the numpy is_scalar method.

    Args:
        value: the value to test for being a scalar value

    Returns:
        boolean: if the given value is a scalar or not
    """
    return np.isscalar(value) or (isinstance(value, np.ndarray) and (len(np.squeeze(value).shape) == 0))

def issubset(self, other):
        """Report whether another set contains this RangeSet."""
        self._binary_sanity_check(other)
        return set.issubset(self, other)

def _squeeze(x, axis):
  """A version of squeeze that works with dynamic axis."""
  x = tf.convert_to_tensor(value=x, name='x')
  if axis is None:
    return tf.squeeze(x, axis=None)
  axis = tf.convert_to_tensor(value=axis, name='axis', dtype=tf.int32)
  axis += tf.zeros([1], dtype=axis.dtype)  # Make axis at least 1d.
  keep_axis, _ = tf.compat.v1.setdiff1d(tf.range(0, tf.rank(x)), axis)
  return tf.reshape(x, tf.gather(tf.shape(input=x), keep_axis))

def _handle_authentication_error(self):
        """
        Return an authentication error.
        """
        response = make_response('Access Denied')
        response.headers['WWW-Authenticate'] = self.auth.get_authenticate_header()
        response.status_code = 401
        return response

def argmax(attrs, inputs, proto_obj):
    """Returns indices of the maximum values along an axis"""
    axis = attrs.get('axis', 0)
    keepdims = attrs.get('keepdims', 1)
    argmax_op = symbol.argmax(inputs[0], axis=axis, keepdims=keepdims)
    # onnx argmax operator always expects int64 as output type
    cast_attrs = {'dtype': 'int64'}
    return 'cast', cast_attrs, argmax_op

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

def uniqify(cls, seq):
        """Returns a unique list of seq"""
        seen = set()
        seen_add = seen.add
        return [ x for x in seq if x not in seen and not seen_add(x)]

def available_gpus():
  """List of GPU device names detected by TensorFlow."""
  local_device_protos = device_lib.list_local_devices()
  return [x.name for x in local_device_protos if x.device_type == 'GPU']

def _ioctl (self, func, args):
        """Call ioctl() with given parameters."""
        import fcntl
        return fcntl.ioctl(self.sockfd.fileno(), func, args)

def flatten_all_but_last(a):
  """Flatten all dimensions of a except the last."""
  ret = tf.reshape(a, [-1, tf.shape(a)[-1]])
  if not tf.executing_eagerly():
    ret.set_shape([None] + a.get_shape().as_list()[-1:])
  return ret

def vline(self, x, y, height, color):
        """Draw a vertical line up to a given length."""
        self.rect(x, y, 1, height, color, fill=True)

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

def _change_height(self, ax, new_value):
        """Make bars in horizontal bar chart thinner"""
        for patch in ax.patches:
            current_height = patch.get_height()
            diff = current_height - new_value

            # we change the bar height
            patch.set_height(new_value)

            # we recenter the bar
            patch.set_y(patch.get_y() + diff * .5)

def is_closed(self):
        """
        Are all entities connected to other entities.

        Returns
        -----------
        closed : bool
          Every entity is connected at its ends
        """
        closed = all(i == 2 for i in
                     dict(self.vertex_graph.degree()).values())

        return closed

def get_lines(handle, line):
    """
    Get zero-indexed line from an open file-like.
    """
    for i, l in enumerate(handle):
        if i == line:
            return l

def is_lazy_iterable(obj):
    """
    Returns whether *obj* is iterable lazily, such as generators, range objects, etc.
    """
    return isinstance(obj,
        (types.GeneratorType, collections.MappingView, six.moves.range, enumerate))

def series_index(self, series):
        """
        Return the integer index of *series* in this sequence.
        """
        for idx, s in enumerate(self):
            if series is s:
                return idx
        raise ValueError('series not in chart data object')

def contained_in(filename, directory):
    """Test if a file is located within the given directory."""
    filename = os.path.normcase(os.path.abspath(filename))
    directory = os.path.normcase(os.path.abspath(directory))
    return os.path.commonprefix([filename, directory]) == directory

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

def is_function(self):
        """return True if callback is a vanilla plain jane function"""
        if self.is_instance() or self.is_class(): return False
        return isinstance(self.callback, (Callable, classmethod))

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

def get_cell(self, index):
        """
        For a single index and return the value

        :param index: index value
        :return: value
        """
        i = sorted_index(self._index, index) if self._sort else self._index.index(index)
        return self._data[i]

def is_empty(self):
        """Returns True if the root node contains no child elements, no text,
        and no attributes other than **type**. Returns False if any are present."""
        non_type_attributes = [attr for attr in self.node.attrib.keys() if attr != 'type']
        return len(self.node) == 0 and len(non_type_attributes) == 0 \
            and not self.node.text and not self.node.tail

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

def __exit__(self, *exc_info):
        """Close connection to NATS when used in a context manager"""

        self._loop.create_task(self._close(Client.CLOSED, True))

def pprint(self, seconds):
        """
        Pretty Prints seconds as Hours:Minutes:Seconds.MilliSeconds

        :param seconds:  The time in seconds.
        """
        return ("%d:%02d:%02d.%03d", reduce(lambda ll, b: divmod(ll[0], b) + ll[1:], [(seconds * 1000,), 1000, 60, 60]))

def mcc(y, z):
    """Matthews correlation coefficient
    """
    tp, tn, fp, fn = contingency_table(y, z)
    return (tp * tn - fp * fn) / K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

def int2str(num, radix=10, alphabet=BASE85):
    """helper function for quick base conversions from integers to strings"""
    return NumConv(radix, alphabet).int2str(num)

def time2seconds(t):
    """Returns seconds since 0h00."""
    return t.hour * 3600 + t.minute * 60 + t.second + float(t.microsecond) / 1e6

def is_iterable_of_int(l):
    r""" Checks if l is iterable and contains only integral types """
    if not is_iterable(l):
        return False

    return all(is_int(value) for value in l)

def timespan(start_time):
    """Return time in milliseconds from start_time"""

    timespan = datetime.datetime.now() - start_time
    timespan_ms = timespan.total_seconds() * 1000
    return timespan_ms

def Binary(x):
    """Return x as a binary type."""
    if isinstance(x, text_type) and not (JYTHON or IRONPYTHON):
        return x.encode()
    return bytes(x)

def dt_to_ts(value):
    """ If value is a datetime, convert to timestamp """
    if not isinstance(value, datetime):
        return value
    return calendar.timegm(value.utctimetuple()) + value.microsecond / 1000000.0

def value(self, progress_indicator):
        """ Interpolate linearly between start and end """
        return interpolate.interpolate_linear_single(self.initial_value, self.final_value, progress_indicator)

def focusNext(self, event):
        """Set focus to next item in sequence"""
        try:
            event.widget.tk_focusNext().focus_set()
        except TypeError:
            # see tkinter equivalent code for tk_focusNext to see
            # commented original version
            name = event.widget.tk.call('tk_focusNext', event.widget._w)
            event.widget._nametowidget(str(name)).focus_set()

def removeFromRegistery(obj) :
	"""Removes an object/rabalist from registery. This is useful if you want to allow the garbage collector to free the memory
	taken by the objects you've already loaded. Be careful might cause some discrepenties in your scripts. For objects,
	cascades to free the registeries of related rabalists also"""
	
	if isRabaObject(obj) :
		_unregisterRabaObjectInstance(obj)
	elif isRabaList(obj) :
		_unregisterRabaListInstance(obj)

def _set_scroll_v(self, *args):
        """Scroll both categories Canvas and scrolling container"""
        self._canvas_categories.yview(*args)
        self._canvas_scroll.yview(*args)

def removeFromRegistery(obj) :
	"""Removes an object/rabalist from registery. This is useful if you want to allow the garbage collector to free the memory
	taken by the objects you've already loaded. Be careful might cause some discrepenties in your scripts. For objects,
	cascades to free the registeries of related rabalists also"""
	
	if isRabaObject(obj) :
		_unregisterRabaObjectInstance(obj)
	elif isRabaList(obj) :
		_unregisterRabaListInstance(obj)

def get_all_items(self):
        """
        Returns all items in the combobox dictionary.
        """
        return [self._widget.itemText(k) for k in range(self._widget.count())]

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

def get_size(self):
        """see doc in Term class"""
        self.curses.setupterm()
        return self.curses.tigetnum('cols'), self.curses.tigetnum('lines')

def find_if_expression_as_statement(node):
    """Finds an "if" expression as a statement"""
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.IfExp)
    )

def closing_plugin(self, cancelable=False):
        """Perform actions before parent main window is closed"""
        self.dialog_manager.close_all()
        self.shell.exit_interpreter()
        return True

def nonlocal_check(self, original, loc, tokens):
        """Check for Python 3 nonlocal statement."""
        return self.check_py("3", "nonlocal statement", original, loc, tokens)

def __grid_widgets(self):
        """Places all the child widgets in the appropriate positions."""
        scrollbar_column = 0 if self.__compound is tk.LEFT else 2
        self._canvas.grid(row=0, column=1, sticky="nswe")
        self._scrollbar.grid(row=0, column=scrollbar_column, sticky="ns")

def restore_scrollbar_position(self):
        """Restoring scrollbar position after main window is visible"""
        scrollbar_pos = self.get_option('scrollbar_position', None)
        if scrollbar_pos is not None:
            self.explorer.treewidget.set_scrollbar_position(scrollbar_pos)

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

def json_iter (path):
    """
    iterator for JSON-per-line in a file pattern
    """
    with open(path, 'r') as f:
        for line in f.readlines():
            yield json.loads(line)

def add_noise(Y, sigma):
    """Adds noise to Y"""
    return Y + np.random.normal(0, sigma, Y.shape)

def _dict_values_sorted_by_key(dictionary):
    # This should be a yield from instead.
    """Internal helper to return the values of a dictionary, sorted by key.
    """
    for _, value in sorted(dictionary.iteritems(), key=operator.itemgetter(0)):
        yield value

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

def add_bundled_jars():
    """
    Adds the bundled jars to the JVM's classpath.
    """
    # determine lib directory with jars
    rootdir = os.path.split(os.path.dirname(__file__))[0]
    libdir = rootdir + os.sep + "lib"

    # add jars from lib directory
    for l in glob.glob(libdir + os.sep + "*.jar"):
        if l.lower().find("-src.") == -1:
            javabridge.JARS.append(str(l))

def xyz2lonlat(x, y, z):
    """Convert cartesian to lon lat."""
    lon = xu.rad2deg(xu.arctan2(y, x))
    lat = xu.rad2deg(xu.arctan2(z, xu.sqrt(x**2 + y**2)))
    return lon, lat

def get_iter_string_reader(stdin):
    """ return an iterator that returns a chunk of a string every time it is
    called.  notice that even though bufsize_type might be line buffered, we're
    not doing any line buffering here.  that's because our StreamBufferer
    handles all buffering.  we just need to return a reasonable-sized chunk. """
    bufsize = 1024
    iter_str = (stdin[i:i + bufsize] for i in range(0, len(stdin), bufsize))
    return get_iter_chunk_reader(iter_str)

def get_python_dict(scala_map):
    """Return a dict from entries in a scala.collection.immutable.Map"""
    python_dict = {}
    keys = get_python_list(scala_map.keys().toList())
    for key in keys:
        python_dict[key] = scala_map.apply(key)
    return python_dict

def jupytext_cli(args=None):
    """Entry point for the jupytext script"""
    try:
        jupytext(args)
    except (ValueError, TypeError, IOError) as err:
        sys.stderr.write('[jupytext] Error: ' + str(err) + '\n')
        exit(1)

def _get_user_agent(self):
        """Retrieve the request's User-Agent, if available.

        Taken from Flask Login utils.py.
        """
        user_agent = request.headers.get('User-Agent')
        if user_agent:
            user_agent = user_agent.encode('utf-8')
        return user_agent or ''

def load_jsonf(fpath, encoding):
    """
    :param unicode fpath:
    :param unicode encoding:
    :rtype: dict | list
    """
    with codecs.open(fpath, encoding=encoding) as f:
        return json.load(f)

def __run(self):
    """Hacked run function, which installs the trace."""
    sys.settrace(self.globaltrace)
    self.__run_backup()
    self.run = self.__run_backup

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

def pprint(self, ind):
        """pretty prints the tree with indentation"""
        pp = pprint.PrettyPrinter(indent=ind)
        pp.pprint(self.tree)

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

def _trim(self, somestr):
        """ Trim left-right given string """
        tmp = RE_LSPACES.sub("", somestr)
        tmp = RE_TSPACES.sub("", tmp)
        return str(tmp)

def _make_sentence(txt):
    """Make a sentence from a piece of text."""
    #Make sure first letter is capitalized
    txt = txt.strip(' ')
    txt = txt[0].upper() + txt[1:] + '.'
    return txt

def _trim(self, somestr):
        """ Trim left-right given string """
        tmp = RE_LSPACES.sub("", somestr)
        tmp = RE_TSPACES.sub("", tmp)
        return str(tmp)

def unique_list_dicts(dlist, key):
    """Return a list of dictionaries which are sorted for only unique entries.

    :param dlist:
    :param key:
    :return list:
    """

    return list(dict((val[key], val) for val in dlist).values())

def clean(s):
  """Removes trailing whitespace on each line."""
  lines = [l.rstrip() for l in s.split('\n')]
  return '\n'.join(lines)

def keyPressEvent(self, event):
        """
        Pyqt specific key press callback function.
        Translates and forwards events to :py:func:`keyboard_event`.
        """
        self.keyboard_event(event.key(), self.keys.ACTION_PRESS, 0)

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

def _try_join_cancelled_thread(thread):
    """Join a thread, but if the thread doesn't terminate for some time, ignore it
    instead of waiting infinitely."""
    thread.join(10)
    if thread.is_alive():
        logging.warning("Thread %s did not terminate within grace period after cancellation",
                        thread.name)

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

def classify_clusters(points, n=10):
    """
    Return an array of K-Means cluster classes for an array of `shapely.geometry.Point` objects.
    """
    arr = [[p.x, p.y] for p in points.values]
    clf = KMeans(n_clusters=n)
    clf.fit(arr)
    classes = clf.predict(arr)
    return classes

def yview(self, *args):
        """Update inplace widgets position when doing vertical scroll"""
        self.after_idle(self.__updateWnds)
        ttk.Treeview.yview(self, *args)

def fit(self, X):
        """ Apply KMeans Clustering
              X: dataset with feature vectors
        """
        self.centers_, self.labels_, self.sse_arr_, self.n_iter_ = \
              _kmeans(X, self.n_clusters, self.max_iter, self.n_trials, self.tol)

def lambda_tuple_converter(func):
    """
    Converts a Python 2 function as
      lambda (x,y): x + y
    In the Python 3 format:
      lambda x,y : x + y
    """
    if func is not None and func.__code__.co_argcount == 1:
        return lambda *args: func(args[0] if len(args) == 1 else args)
    else:
        return func

def l2_norm(arr):
    """
    The l2 norm of an array is is defined as: sqrt(||x||), where ||x|| is the
    dot product of the vector.
    """
    arr = np.asarray(arr)
    return np.sqrt(np.dot(arr.ravel().squeeze(), arr.ravel().squeeze()))

def lambda_tuple_converter(func):
    """
    Converts a Python 2 function as
      lambda (x,y): x + y
    In the Python 3 format:
      lambda x,y : x + y
    """
    if func is not None and func.__code__.co_argcount == 1:
        return lambda *args: func(args[0] if len(args) == 1 else args)
    else:
        return func

def projR(gamma, p):
    """return the KL projection on the row constrints """
    return np.multiply(gamma.T, p / np.maximum(np.sum(gamma, axis=1), 1e-10)).T

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

def norm_vec(vector):
    """Normalize the length of a vector to one"""
    assert len(vector) == 3
    v = np.array(vector)
    return v/np.sqrt(np.sum(v**2))

def json_to_initkwargs(self, json_data, kwargs):
        """Subclassing hook to specialize how JSON data is converted
        to keyword arguments"""
        if isinstance(json_data, basestring):
            json_data = json.loads(json_data)
        return json_to_initkwargs(self, json_data, kwargs)

def heappush_max(heap, item):
    """Push item onto heap, maintaining the heap invariant."""
    heap.append(item)
    _siftdown_max(heap, 0, len(heap) - 1)

def json_to_initkwargs(self, json_data, kwargs):
        """Subclassing hook to specialize how JSON data is converted
        to keyword arguments"""
        if isinstance(json_data, basestring):
            json_data = json.loads(json_data)
        return json_to_initkwargs(self, json_data, kwargs)

def bytes_to_str(s, encoding='utf-8'):
    """Returns a str if a bytes object is given."""
    if six.PY3 and isinstance(s, bytes):
        return s.decode(encoding)
    return s

def set_ylimits(self, row, column, min=None, max=None):
        """Set y-axis limits of a subplot.

        :param row,column: specify the subplot.
        :param min: minimal axis value
        :param max: maximum axis value

        """
        subplot = self.get_subplot_at(row, column)
        subplot.set_ylimits(min, max)

def dump_nparray(self, obj, class_name=numpy_ndarray_class_name):
        """
        ``numpy.ndarray`` dumper.
        """
        return {"$" + class_name: self._json_convert(obj.tolist())}

def fit_linear(X, y):
    """
    Uses OLS to fit the regression.
    """
    model = linear_model.LinearRegression()
    model.fit(X, y)
    return model

def delayed_close(self):
        """ Delayed close - won't close immediately, but on the next reactor
        loop. """
        self.state = SESSION_STATE.CLOSING
        reactor.callLater(0, self.close)

def get_free_memory_win():
    """Return current free memory on the machine for windows.

    Warning : this script is really not robust
    Return in MB unit
    """
    stat = MEMORYSTATUSEX()
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return int(stat.ullAvailPhys / 1024 / 1024)

def search_for_tweets_about(user_id, params):
    """ Search twitter API """
    url = "https://api.twitter.com/1.1/search/tweets.json"
    response = make_twitter_request(url, user_id, params)
    return process_tweets(response.json()["statuses"])

def pack_triples_numpy(triples):
    """Packs a list of triple indexes into a 2D numpy array."""
    if len(triples) == 0:
        return np.array([], dtype=np.int64)
    return np.stack(list(map(_transform_triple_numpy, triples)), axis=0)

def urlize_twitter(text):
    """
    Replace #hashtag and @username references in a tweet with HTML text.
    """
    html = TwitterText(text).autolink.auto_link()
    return mark_safe(html.replace(
        'twitter.com/search?q=', 'twitter.com/search/realtime/'))

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

def is_int_type(val):
    """Return True if `val` is of integer type."""
    try:               # Python 2
        return isinstance(val, (int, long))
    except NameError:  # Python 3
        return isinstance(val, int)

def dedupe_list(seq):
    """
    Utility function to remove duplicates from a list
    :param seq: The sequence (list) to deduplicate
    :return: A list with original duplicates removed
    """
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]

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

def list2dict(lst):
    """Takes a list of (key,value) pairs and turns it into a dict."""

    dic = {}
    for k,v in lst: dic[k] = v
    return dic

def test(ctx, all=False, verbose=False):
    """Run the tests."""
    cmd = 'tox' if all else 'py.test'
    if verbose:
        cmd += ' -v'
    return ctx.run(cmd, pty=True).return_code

def h5ToDict(h5, readH5pyDataset=True):
    """ Read a hdf5 file into a dictionary """
    h = h5py.File(h5, "r")
    ret = unwrapArray(h, recursive=True, readH5pyDataset=readH5pyDataset)
    if readH5pyDataset: h.close()
    return ret

def cover(session):
    """Run the final coverage report.
    This outputs the coverage report aggregating coverage from the unit
    test runs (not system test runs), and then erases coverage data.
    """
    session.interpreter = 'python3.6'
    session.install('coverage', 'pytest-cov')
    session.run('coverage', 'report', '--show-missing', '--fail-under=100')
    session.run('coverage', 'erase')

def decode_example(self, example):
    """Reconstruct the image from the tf example."""
    img = tf.image.decode_image(
        example, channels=self._shape[-1], dtype=tf.uint8)
    img.set_shape(self._shape)
    return img

def assert_is_not(expected, actual, message=None, extra=None):
    """Raises an AssertionError if expected is actual."""
    assert expected is not actual, _assert_fail_message(
        message, expected, actual, "is", extra
    )

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

def from_file(filename):
    """
    load an nparray object from a json filename

    @parameter str filename: path to the file
    """
    f = open(filename, 'r')
    j = json.load(f)
    f.close()

    return from_dict(j)

def get_time():
    """Get time from a locally running NTP server"""

    time_request = '\x1b' + 47 * '\0'
    now = struct.unpack("!12I", ntp_service.request(time_request, timeout=5.0).data.read())[10]
    return time.ctime(now - EPOCH_START)

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

def logger(message, level=10):
    """Handle logging."""
    logging.getLogger(__name__).log(level, str(message))

def open_usb_handle(self, port_num):
    """open usb port

    Args:
      port_num: port number on the Cambrionix unit

    Return:
      usb handle
    """
    serial = self.get_usb_serial(port_num)
    return local_usb.LibUsbHandle.open(serial_number=serial)

def _histplot_op(ax, data, **kwargs):
    """Add a histogram for the data to the axes."""
    bins = get_bins(data)
    ax.hist(data, bins=bins, align="left", density=True, **kwargs)
    return ax

def list2dict(list_of_options):
    """Transforms a list of 2 element tuples to a dictionary"""
    d = {}
    for key, value in list_of_options:
        d[key] = value
    return d

def survival(value=t, lam=lam, f=failure):
    """Exponential survival likelihood, accounting for censoring"""
    return sum(f * log(lam) - lam * value)

def put_pidfile( pidfile_path, pid ):
    """
    Put a PID into a pidfile
    """
    with open( pidfile_path, "w" ) as f:
        f.write("%s" % pid)
        os.fsync(f.fileno())

    return

def _log_multivariate_normal_density_tied(X, means, covars):
    """Compute Gaussian log-density at X for a tied model."""
    cv = np.tile(covars, (means.shape[0], 1, 1))
    return _log_multivariate_normal_density_full(X, means, cv)

def to_camel(s):
    """
    :param string s: under_scored string to be CamelCased
    :return: CamelCase version of input
    :rtype: str
    """
    # r'(?!^)_([a-zA-Z]) original regex wasn't process first groups
    return re.sub(r'_([a-zA-Z])', lambda m: m.group(1).upper(), '_' + s)

def _eq(self, other):
        """Compare two nodes for equality."""
        return (self.type, self.value) == (other.type, other.value)

def lambda_tuple_converter(func):
    """
    Converts a Python 2 function as
      lambda (x,y): x + y
    In the Python 3 format:
      lambda x,y : x + y
    """
    if func is not None and func.__code__.co_argcount == 1:
        return lambda *args: func(args[0] if len(args) == 1 else args)
    else:
        return func

def _pip_exists(self):
        """Returns True if pip exists inside the virtual environment. Can be
        used as a naive way to verify that the environment is installed."""
        return os.path.isfile(os.path.join(self.path, 'bin', 'pip'))

def copy(obj):
    def copy(self):
        """
        Copy self to a new object.
        """
        from copy import deepcopy

        return deepcopy(self)
    obj.copy = copy
    return obj

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

def _transform_triple_numpy(x):
    """Transform triple index into a 1-D numpy array."""
    return np.array([x.head, x.relation, x.tail], dtype=np.int64)

def _adjust_offset(self, real_wave_mfcc, algo_parameters):
        """
        OFFSET
        """
        self.log(u"Called _adjust_offset")
        self._apply_offset(offset=algo_parameters[0])

def cols_str(columns):
    """Concatenate list of columns into a string."""
    cols = ""
    for c in columns:
        cols = cols + wrap(c) + ', '
    return cols[:-2]

def _ws_on_close(self, ws: websocket.WebSocketApp):
        """Callback for closing the websocket connection

        Args:
            ws: websocket connection (now closed)
        """
        self.connected = False
        self.logger.error('Websocket closed')
        self._reconnect_websocket()

def list2dict(lst):
    """Takes a list of (key,value) pairs and turns it into a dict."""

    dic = {}
    for k,v in lst: dic[k] = v
    return dic

def cell_ends_with_code(lines):
    """Is the last line of the cell a line with code?"""
    if not lines:
        return False
    if not lines[-1].strip():
        return False
    if lines[-1].startswith('#'):
        return False
    return True

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

def get_property_by_name(pif, name):
    """Get a property by name"""
    return next((x for x in pif.properties if x.name == name), None)

def batch(items, size):
    """Batches a list into a list of lists, with sub-lists sized by a specified
    batch size."""
    return [items[x:x + size] for x in xrange(0, len(items), size)]

def set_cursor(self, x, y):
        """
        Sets the cursor to the desired position.

        :param x: X position
        :param y: Y position
        """
        curses.curs_set(1)
        self.screen.move(y, x)

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

def RecurseKeys(self):
    """Recurses the subkeys starting with the key.

    Yields:
      WinRegistryKey: Windows Registry key.
    """
    yield self
    for subkey in self.GetSubkeys():
      for key in subkey.RecurseKeys():
        yield key

def invertDictMapping(d):
    """ Invert mapping of dictionary (i.e. map values to list of keys) """
    inv_map = {}
    for k, v in d.items():
        inv_map[v] = inv_map.get(v, [])
        inv_map[v].append(k)
    return inv_map

def match_paren(self, tokens, item):
        """Matches a paren."""
        match, = tokens
        return self.match(match, item)

def save_dict_to_file(filename, dictionary):
  """Saves dictionary as CSV file."""
  with open(filename, 'w') as f:
    writer = csv.writer(f)
    for k, v in iteritems(dictionary):
      writer.writerow([str(k), str(v)])

def matches(self, s):
    """Whether the pattern matches anywhere in the string s."""
    regex_matches = self.compiled_regex.search(s) is not None
    return not regex_matches if self.inverted else regex_matches

def save_notebook(work_notebook, write_file):
    """Saves the Jupyter work_notebook to write_file"""
    with open(write_file, 'w') as out_nb:
        json.dump(work_notebook, out_nb, indent=2)

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

def write_tsv_line_from_list(linelist, outfp):
    """Utility method to convert list to tsv line with carriage return"""
    line = '\t'.join(linelist)
    outfp.write(line)
    outfp.write('\n')

def cross_product_matrix(vec):
    """Returns a 3x3 cross-product matrix from a 3-element vector."""
    return np.array([[0, -vec[2], vec[1]],
                     [vec[2], 0, -vec[0]],
                     [-vec[1], vec[0], 0]])

def save_hdf(self,filename,path=''):
        """Saves all relevant data to .h5 file; so state can be restored.
        """
        self.dataframe.to_hdf(filename,'{}/df'.format(path))

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

def _write_json(file, contents):
    """Write a dict to a JSON file."""
    with open(file, 'w') as f:
        return json.dump(contents, f, indent=2, sort_keys=True)

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

def md5_string(s):
    """
    Shortcut to create md5 hash
    :param s:
    :return:
    """
    m = hashlib.md5()
    m.update(s)
    return str(m.hexdigest())

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

def __ror__(self, other):
		"""The main machinery of the Pipe, calling the chosen callable with the recorded arguments."""
		
		return self.callable(*(self.args + (other, )), **self.kwargs)

def _get_minidom_tag_value(station, tag_name):
    """get a value from a tag (if it exists)"""
    tag = station.getElementsByTagName(tag_name)[0].firstChild
    if tag:
        return tag.nodeValue

    return None

def _flush(self, buffer):
        """
        Flush the write buffers of the stream if applicable.

        Args:
            buffer (memoryview): Buffer content.
        """
        container, obj = self._client_args
        with _handle_client_exception():
            self._client.put_object(container, obj, buffer)

def xml_str_to_dict(s):
    """ Transforms an XML string it to python-zimbra dict format

    For format, see:
      https://github.com/Zimbra-Community/python-zimbra/blob/master/README.md

    :param: a string, containing XML
    :returns: a dict, with python-zimbra format
    """
    xml = minidom.parseString(s)
    return pythonzimbra.tools.xmlserializer.dom_to_dict(xml.firstChild)

def oplot(self, x, y, **kw):
        """generic plotting method, overplotting any existing plot """
        self.panel.oplot(x, y, **kw)

def series_table_row_offset(self, series):
        """
        Return the number of rows preceding the data table for *series* in
        the Excel worksheet.
        """
        title_and_spacer_rows = series.index * 2
        data_point_rows = series.data_point_offset
        return title_and_spacer_rows + data_point_rows

def transformer_tall_pretrain_lm_tpu_adafactor():
  """Hparams for transformer on LM pretraining (with 64k vocab) on TPU."""
  hparams = transformer_tall_pretrain_lm()
  update_hparams_for_tpu(hparams)
  hparams.max_length = 1024
  # For multi-problem on TPU we need it in absolute examples.
  hparams.batch_size = 8
  hparams.multiproblem_vocab_size = 2**16
  return hparams

def extent(self):
        """Helper for matplotlib imshow"""
        return (
            self.intervals[1].pix1 - 0.5,
            self.intervals[1].pix2 - 0.5,
            self.intervals[0].pix1 - 0.5,
            self.intervals[0].pix2 - 0.5,
        )

def api_test(method='GET', **response_kwargs):
    """ Decorator to ensure API calls are made and return expected data. """

    method = method.lower()

    def api_test_factory(fn):
        @functools.wraps(fn)
        @mock.patch('requests.{}'.format(method))
        def execute_test(method_func, *args, **kwargs):
            method_func.return_value = MockResponse(**response_kwargs)

            expected_url, response = fn(*args, **kwargs)

            method_func.assert_called_once()
            assert_valid_api_call(method_func, expected_url)
            assert isinstance(response, JSONAPIParser)
            assert response.json_data is method_func.return_value.data

        return execute_test

    return api_test_factory

def vertical_percent(plot, percent=0.1):
    """
    Using the size of the y axis, return a fraction of that size.
    """
    plot_bottom, plot_top = plot.get_ylim()
    return percent * (plot_top - plot_bottom)

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

def _ParseYamlFromFile(filedesc):
  """Parses given YAML file."""
  content = filedesc.read()
  return yaml.Parse(content) or collections.OrderedDict()

def get_axis(array, axis, slice_num):
    """Returns a fixed axis"""

    slice_list = [slice(None)] * array.ndim
    slice_list[axis] = slice_num
    slice_data = array[tuple(slice_list)].T  # transpose for proper orientation

    return slice_data

def print_yaml(o):
    """Pretty print an object as YAML."""
    print(yaml.dump(o, default_flow_style=False, indent=4, encoding='utf-8'))

def yaml_to_param(obj, name):
	"""
	Return the top-level element of a document sub-tree containing the
	YAML serialization of a Python object.
	"""
	return from_pyvalue(u"yaml:%s" % name, unicode(yaml.dump(obj)))

def comments(tag, limit=0, flags=0, **kwargs):
    """Get comments only."""

    return [comment for comment in cm.CommentsMatch(tag).get_comments(limit)]

def extract_zip(zip_path, target_folder):
    """
    Extract the content of the zip-file at `zip_path` into `target_folder`.
    """
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target_folder)

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

def extract(self, destination):
        """Extract the archive."""
        with zipfile.ZipFile(self.archive, 'r') as zip_ref:
            zip_ref.extractall(destination)

def map_keys_deep(f, dct):
    """
    Implementation of map that recurses. This tests the same keys at every level of dict and in lists
    :param f: 2-ary function expecting a key and value and returns a modified key
    :param dct: Dict for deep processing
    :return: Modified dct with matching props mapped
    """
    return _map_deep(lambda k, v: [f(k, v), v], dct)

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

def step_next_line(self):
        """Sets cursor as beginning of next line."""
        self._eol.append(self.position)
        self._lineno += 1
        self._col_offset = 0

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

def bulk_query(self, query, *multiparams):
        """Bulk insert or update."""

        with self.get_connection() as conn:
            conn.bulk_query(query, *multiparams)

def mean_date(dt_list):
    """Calcuate mean datetime from datetime list
    """
    dt_list_sort = sorted(dt_list)
    dt_list_sort_rel = [dt - dt_list_sort[0] for dt in dt_list_sort]
    avg_timedelta = sum(dt_list_sort_rel, timedelta())/len(dt_list_sort_rel)
    return dt_list_sort[0] + avg_timedelta

def flatten_list(l):
    """ Nested lists to single-level list, does not split strings"""
    return list(chain.from_iterable(repeat(x,1) if isinstance(x,str) else x for x in l))

def changed(self, *value):
        """Checks whether the value has changed since the last call."""
        if self._last_checked_value != value:
            self._last_checked_value = value
            return True
        return False

def to_dotfile(G: nx.DiGraph, filename: str):
    """ Output a networkx graph to a DOT file. """
    A = to_agraph(G)
    A.write(filename)

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

def draw_header(self, stream, header):
        """Draw header with underline"""
        stream.writeln('=' * (len(header) + 4))
        stream.writeln('| ' + header + ' |')
        stream.writeln('=' * (len(header) + 4))
        stream.writeln()

def urlencoded(body, charset='ascii', **kwargs):
    """Converts query strings into native Python objects"""
    return parse_query_string(text(body, charset=charset), False)

def get_server(address=None):
        """Return an SMTP servername guess from outgoing email address."""
        if address:
            domain = address.split("@")[1]
            try:
                return SMTP_SERVERS[domain]
            except KeyError:
                return ("smtp." + domain, 465)
        return (None, None)

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

def nlevels(self):
        """
        Get the number of factor levels for each categorical column.

        :returns: A list of the number of levels per column.
        """
        levels = self.levels()
        return [len(l) for l in levels] if levels else 0

def to_str(s):
    """
    Convert bytes and non-string into Python 3 str
    """
    if isinstance(s, bytes):
        s = s.decode('utf-8')
    elif not isinstance(s, str):
        s = str(s)
    return s

def unique(list):
    """ Returns a copy of the list without duplicates.
    """
    unique = []; [unique.append(x) for x in list if x not in unique]
    return unique

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

def _normalize_abmn(abmn):
    """return a normalized version of abmn
    """
    abmn_2d = np.atleast_2d(abmn)
    abmn_normalized = np.hstack((
        np.sort(abmn_2d[:, 0:2], axis=1),
        np.sort(abmn_2d[:, 2:4], axis=1),
    ))
    return abmn_normalized

def _get_line_no_from_comments(py_line):
    """Return the line number parsed from the comment or 0."""
    matched = LINECOL_COMMENT_RE.match(py_line)
    if matched:
        return int(matched.group(1))
    else:
        return 0

def denorm(self,arr):
        """Reverse the normalization done to a batch of images.

        Arguments:
            arr: of shape/size (N,3,sz,sz)
        """
        if type(arr) is not np.ndarray: arr = to_np(arr)
        if len(arr.shape)==3: arr = arr[None]
        return self.transform.denorm(np.rollaxis(arr,1,4))

def QA_util_datetime_to_strdate(dt):
    """
    :param dt:  pythone datetime.datetime
    :return:  1999-02-01 string type
    """
    strdate = "%04d-%02d-%02d" % (dt.year, dt.month, dt.day)
    return strdate

def _normalize(mat: np.ndarray):
    """rescales a numpy array, so that min is 0 and max is 255"""
    return ((mat - mat.min()) * (255 / mat.max())).astype(np.uint8)

def datetime_to_ms(dt):
    """
    Converts a datetime to a millisecond accuracy timestamp
    """
    seconds = calendar.timegm(dt.utctimetuple())
    return seconds * 1000 + int(dt.microsecond / 1000)

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

def _histplot_bins(column, bins=100):
    """Helper to get bins for histplot."""
    col_min = np.min(column)
    col_max = np.max(column)
    return range(col_min, col_max + 2, max((col_max - col_min) // bins, 1))

def index(self, value):
		"""
		Return the smallest index of the row(s) with this column
		equal to value.
		"""
		for i in xrange(len(self.parentNode)):
			if getattr(self.parentNode[i], self.Name) == value:
				return i
		raise ValueError(value)

def __len__(self):
        """ This will equal 124 for the V1 database. """
        length = 0
        for typ, siz, _ in self.format:
            length += siz
        return length

def grandparent_path(self):
        """ return grandparent's path string """
        return os.path.basename(os.path.join(self.path, '../..'))

def update_dict(obj, dict, attributes):
    """Update dict with fields from obj.attributes.

    :param obj: the object updated into dict
    :param dict: the result dictionary
    :param attributes: a list of attributes belonging to obj
    """
    for attribute in attributes:
        if hasattr(obj, attribute) and getattr(obj, attribute) is not None:
            dict[attribute] = getattr(obj, attribute)

def check(modname):
    """Check if required dependency is installed"""
    for dependency in DEPENDENCIES:
        if dependency.modname == modname:
            return dependency.check()
    else:
        raise RuntimeError("Unkwown dependency %s" % modname)

def _rgbtomask(self, obj):
        """Convert RGB arrays from mask canvas object back to boolean mask."""
        dat = obj.get_image().get_data()  # RGB arrays
        return dat.sum(axis=2).astype(np.bool)

def toBase64(s):
    """Represent string / bytes s as base64, omitting newlines"""
    if isinstance(s, str):
        s = s.encode("utf-8")
    return binascii.b2a_base64(s)[:-1]

def delete(filething):
    """Remove tags from a file.

    Args:
        filething (filething)
    Raises:
        mutagen.MutagenError
    """

    f = FLAC(filething)
    filething.fileobj.seek(0)
    f.delete(filething)

def get_from_human_key(self, key):
        """Return the key (aka database value) of a human key (aka Python identifier)."""
        if key in self._identifier_map:
            return self._identifier_map[key]
        raise KeyError(key)

def cio_close(cio):
    """Wraps openjpeg library function cio_close.
    """
    OPENJPEG.opj_cio_close.argtypes = [ctypes.POINTER(CioType)]
    OPENJPEG.opj_cio_close(cio)

def input_int_default(question="", default=0):
    """A function that works for both, Python 2.x and Python 3.x.
       It asks the user for input and returns it as a string.
    """
    answer = input_string(question)
    if answer == "" or answer == "yes":
        return default
    else:
        return int(answer)

def do_serial(self, p):
		"""Set the serial port, e.g.: /dev/tty.usbserial-A4001ib8"""
		try:
			self.serial.port = p
			self.serial.open()
			print 'Opening serial port: %s' % p
		except Exception, e:
			print 'Unable to open serial port: %s' % p

def is_timestamp(instance):
    """Validates data is a timestamp"""
    if not isinstance(instance, (int, str)):
        return True
    return datetime.fromtimestamp(int(instance))

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

def inject_into_urllib3():
    """
    Monkey-patch urllib3 with SecureTransport-backed SSL-support.
    """
    util.ssl_.SSLContext = SecureTransportContext
    util.HAS_SNI = HAS_SNI
    util.ssl_.HAS_SNI = HAS_SNI
    util.IS_SECURETRANSPORT = True
    util.ssl_.IS_SECURETRANSPORT = True

def find_path(self, start, end, grid):
        """
        find a path from start to end node on grid using the A* algorithm
        :param start: start node
        :param end: end node
        :param grid: grid that stores all possible steps/tiles as 2D-list
        :return:
        """
        start.g = 0
        start.f = 0
        return super(AStarFinder, self).find_path(start, end, grid)

def strip_querystring(url):
    """Remove the querystring from the end of a URL."""
    p = six.moves.urllib.parse.urlparse(url)
    return p.scheme + "://" + p.netloc + p.path

def _uptime_syllable():
    """Returns uptime in seconds or None, on Syllable."""
    global __boottime
    try:
        __boottime = os.stat('/dev/pty/mst/pty0').st_mtime
        return time.time() - __boottime
    except (NameError, OSError):
        return None

def unpickle(pickle_file):
    """Unpickle a python object from the given path."""
    pickle = None
    with open(pickle_file, "rb") as pickle_f:
        pickle = dill.load(pickle_f)
    if not pickle:
        LOG.error("Could not load python object from file")
    return pickle

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

def parent_widget(self):
        """ Reimplemented to only return GraphicsItems """
        parent = self.parent()
        if parent is not None and isinstance(parent, QtGraphicsItem):
            return parent.widget

def save_keras_definition(keras_model, path):
    """
    Save a Keras model definition to JSON with given path
    """
    model_json = keras_model.to_json()
    with open(path, "w") as json_file:
        json_file.write(model_json)

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

def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return F.pad(img, self.padding, self.fill, self.padding_mode)

def raise_os_error(_errno, path=None):
    """
    Helper for raising the correct exception under Python 3 while still
    being able to raise the same common exception class in Python 2.7.
    """

    msg = "%s: '%s'" % (strerror(_errno), path) if path else strerror(_errno)
    raise OSError(_errno, msg)

def indent(block, spaces):
    """ indents paragraphs of text for rst formatting """
    new_block = ''
    for line in block.split('\n'):
        new_block += spaces + line + '\n'
    return new_block

def money(min=0, max=10):
    """Return a str of decimal with two digits after a decimal mark."""
    value = random.choice(range(min * 100, max * 100))
    return "%1.2f" % (float(value) / 100)

def message_from_string(s, *args, **kws):
    """Parse a string into a Message object model.

    Optional _class and strict are passed to the Parser constructor.
    """
    from future.backports.email.parser import Parser
    return Parser(*args, **kws).parsestr(s)

def rnormal(mu, tau, size=None):
    """
    Random normal variates.
    """
    return np.random.normal(mu, 1. / np.sqrt(tau), size)

def __getattr__(self, name):
        """Return wrapper to named api method."""
        return functools.partial(self._obj.request, self._api_prefix + name)

def _rndPointDisposition(dx, dy):
        """Return random disposition point."""
        x = int(random.uniform(-dx, dx))
        y = int(random.uniform(-dy, dy))
        return (x, y)

def minus(*args):
    """Also, converts either to ints or to floats."""
    if len(args) == 1:
        return -to_numeric(args[0])
    return to_numeric(args[0]) - to_numeric(args[1])

def random_color(_min=MIN_COLOR, _max=MAX_COLOR):
    """Returns a random color between min and max."""
    return color(random.randint(_min, _max))

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

def _remove_nonascii(self, df):
    """Make copy and remove non-ascii characters from it."""

    df_copy = df.copy(deep=True)
    for col in df_copy.columns:
      if (df_copy[col].dtype == np.dtype('O')):
        df_copy[col] = df[col].apply(
          lambda x: re.sub(r'[^\x00-\x7f]', r'', x) if isinstance(x, six.string_types) else x)

    return df_copy

def listified_tokenizer(source):
    """Tokenizes *source* and returns the tokens as a list of lists."""
    io_obj = io.StringIO(source)
    return [list(a) for a in tokenize.generate_tokens(io_obj.readline)]

def __get_float(section, name):
    """Get the forecasted float from json section."""
    try:
        return float(section[name])
    except (ValueError, TypeError, KeyError):
        return float(0)

def read_string(buff, byteorder='big'):
    """Read a string from a file-like object."""
    length = read_numeric(USHORT, buff, byteorder)
    return buff.read(length).decode('utf-8')

def imapchain(*a, **kwa):
    """ Like map but also chains the results. """

    imap_results = map( *a, **kwa )
    return itertools.chain( *imap_results )

def _read_json_file(self, json_file):
        """ Helper function to read JSON file as OrderedDict """

        self.log.debug("Reading '%s' JSON file..." % json_file)

        with open(json_file, 'r') as f:
            return json.load(f, object_pairs_hook=OrderedDict)

def GeneratePassphrase(length=20):
  """Create a 20 char passphrase with easily typeable chars."""
  valid_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
  valid_chars += "0123456789 ,-_&$#"
  return "".join(random.choice(valid_chars) for i in range(length))

def read_utf8(fh, byteorder, dtype, count, offsetsize):
    """Read tag data from file and return as unicode string."""
    return fh.read(count).decode('utf-8')

def survival(value=t, lam=lam, f=failure):
    """Exponential survival likelihood, accounting for censoring"""
    return sum(f * log(lam) - lam * value)

def read_numpy(fh, byteorder, dtype, count, offsetsize):
    """Read tag data from file and return as numpy array."""
    dtype = 'b' if dtype[-1] == 's' else byteorder+dtype[-1]
    return fh.read_array(dtype, count)

def downsample(array, k):
    """Choose k random elements of array."""
    length = array.shape[0]
    indices = random.sample(xrange(length), k)
    return array[indices]

def get(url):
    """Recieving the JSON file from uulm"""
    response = urllib.request.urlopen(url)
    data = response.read()
    data = data.decode("utf-8")
    data = json.loads(data)
    return data

def random_choice(sequence):
    """ Same as :meth:`random.choice`, but also supports :class:`set` type to be passed as sequence. """
    return random.choice(tuple(sequence) if isinstance(sequence, set) else sequence)

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

def pause(self):
        """Pause the music"""
        mixer.music.pause()
        self.pause_time = self.get_time()
        self.paused = True

def connected_socket(address, timeout=3):
    """ yields a connected socket """
    sock = socket.create_connection(address, timeout)
    yield sock
    sock.close()

def makeAnimation(self):
        """Use pymovie to render (visual+audio)+text overlays.
        """
        aclip=mpy.AudioFileClip("sound.wav")
        self.iS=self.iS.set_audio(aclip)
        self.iS.write_videofile("mixedVideo.webm",15,audio=True)
        print("wrote "+"mixedVideo.webm")

def getTopRight(self):
        """
        Retrieves a tuple with the x,y coordinates of the upper right point of the ellipse. 
        Requires the radius and the coordinates to be numbers
        """
        return (float(self.get_cx()) + float(self.get_rx()), float(self.get_cy()) + float(self.get_ry()))

def _histplot_op(ax, data, **kwargs):
    """Add a histogram for the data to the axes."""
    bins = get_bins(data)
    ax.hist(data, bins=bins, align="left", density=True, **kwargs)
    return ax

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

def lognorm(x, mu, sigma=1.0):
    """ Log-normal function from scipy """
    return stats.lognorm(sigma, scale=mu).pdf(x)

def is_password_valid(password):
    """
    Check if a password is valid
    """
    pattern = re.compile(r"^.{4,75}$")
    return bool(pattern.match(password))

def mad(v):
    """MAD -- Median absolute deviation. More robust than standard deviation.
    """
    return np.median(np.abs(v - np.median(v)))

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

def lognorm(x, mu, sigma=1.0):
    """ Log-normal function from scipy """
    return stats.lognorm(sigma, scale=mu).pdf(x)

def unpunctuate(s, *, char_blacklist=string.punctuation):
    """ Remove punctuation from string s. """
    # remove punctuation
    s = "".join(c for c in s if c not in char_blacklist)
    # remove consecutive spaces
    return " ".join(filter(None, s.split(" ")))

def normalise_string(string):
    """ Strips trailing whitespace from string, lowercases it and replaces
        spaces with underscores
    """
    string = (string.strip()).lower()
    return re.sub(r'\W+', '_', string)

def cart2pol(x, y):
    """Cartesian to Polar coordinates conversion."""
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho

def toBase64(s):
    """Represent string / bytes s as base64, omitting newlines"""
    if isinstance(s, str):
        s = s.encode("utf-8")
    return binascii.b2a_base64(s)[:-1]

def multi_pop(d, *args):
    """ pops multiple keys off a dict like object """
    retval = {}
    for key in args:
        if key in d:
            retval[key] = d.pop(key)
    return retval

def _remove_nonascii(self, df):
    """Make copy and remove non-ascii characters from it."""

    df_copy = df.copy(deep=True)
    for col in df_copy.columns:
      if (df_copy[col].dtype == np.dtype('O')):
        df_copy[col] = df[col].apply(
          lambda x: re.sub(r'[^\x00-\x7f]', r'', x) if isinstance(x, six.string_types) else x)

    return df_copy

def _get_printable_columns(columns, row):
    """Return only the part of the row which should be printed.
    """
    if not columns:
        return row

    # Extract the column values, in the order specified.
    return tuple(row[c] for c in columns)

def strip_spaces(value, sep=None, join=True):
    """Cleans trailing whitespaces and replaces also multiple whitespaces with a single space."""
    value = value.strip()
    value = [v.strip() for v in value.split(sep)]
    join_sep = sep or ' '
    return join_sep.join(value) if join else value

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

def head(filename, n=10):
    """ prints the top `n` lines of a file """
    with freader(filename) as fr:
        for _ in range(n):
            print(fr.readline().strip())

def dedup_list(l):
    """Given a list (l) will removing duplicates from the list,
       preserving the original order of the list. Assumes that
       the list entrie are hashable."""
    dedup = set()
    return [ x for x in l if not (x in dedup or dedup.add(x))]

def _attrprint(d, delimiter=', '):
    """Print a dictionary of attributes in the DOT format"""
    return delimiter.join(('"%s"="%s"' % item) for item in sorted(d.items()))

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

def toJson(protoObject, indent=None):
    """
    Serialises a protobuf object as json
    """
    # Using the internal method because this way we can reformat the JSON
    js = json_format.MessageToDict(protoObject, False)
    return json.dumps(js, indent=indent)

def file_remove(self, path, filename):
        """Check if filename exists and remove
        """
        if os.path.isfile(path + filename):
            os.remove(path + filename)

def printc(cls, txt, color=colors.red):
        """Print in color."""
        print(cls.color_txt(txt, color))

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

def raw_print(*args, **kw):
    """Raw print to sys.__stdout__, otherwise identical interface to print()."""

    print(*args, sep=kw.get('sep', ' '), end=kw.get('end', '\n'),
          file=sys.__stdout__)
    sys.__stdout__.flush()

def strip_spaces(x):
    """
    Strips spaces
    :param x:
    :return:
    """
    x = x.replace(b' ', b'')
    x = x.replace(b'\t', b'')
    return x

def _repr(obj):
    """Show the received object as precise as possible."""
    vals = ", ".join("{}={!r}".format(
        name, getattr(obj, name)) for name in obj._attribs)
    if vals:
        t = "{}(name={}, {})".format(obj.__class__.__name__, obj.name, vals)
    else:
        t = "{}(name={})".format(obj.__class__.__name__, obj.name)
    return t

def info(txt):
    """Print, emphasized 'neutral', the given 'txt' message"""

    print("%s# %s%s%s" % (PR_EMPH_CC, get_time_stamp(), txt, PR_NC))
    sys.stdout.flush()

def strip_line(line, sep=os.linesep):
    """
    Removes occurrence of character (sep) from a line of text
    """

    try:
        return line.strip(sep)
    except TypeError:
        return line.decode('utf-8').strip(sep)

def EvalBinomialPmf(k, n, p):
    """Evaluates the binomial pmf.

    Returns the probabily of k successes in n trials with probability p.
    """
    return scipy.stats.binom.pmf(k, n, p)

def lines(input):
    """Remove comments and empty lines"""
    for raw_line in input:
        line = raw_line.strip()
        if line and not line.startswith('#'):
            yield strip_comments(line)

def format_docstring(*args, **kwargs):
    """
    Decorator for clean docstring formatting
    """
    def decorator(func):
        func.__doc__ = getdoc(func).format(*args, **kwargs)
        return func
    return decorator

def unique(transactions):
    """ Remove any duplicate entries. """
    seen = set()
    # TODO: Handle comments
    return [x for x in transactions if not (x in seen or seen.add(x))]

def softplus(attrs, inputs, proto_obj):
    """Applies the sofplus activation function element-wise to the input."""
    new_attrs = translation_utils._add_extra_attributes(attrs, {'act_type' : 'softrelu'})
    return 'Activation', new_attrs, inputs

def strip_accents(string):
    """
    Strip all the accents from the string
    """
    return u''.join(
        (character for character in unicodedata.normalize('NFD', string)
         if unicodedata.category(character) != 'Mn'))

def unapostrophe(text):
    """Strip apostrophe and 's' from the end of a string."""
    text = re.sub(r'[%s]s?$' % ''.join(APOSTROPHES), '', text)
    return text

def screen_cv2(self):
        """cv2 Image of current window screen"""
        pil_image = self.screen.convert('RGB')
        cv2_image = np.array(pil_image)
        pil_image.close()
        # Convert RGB to BGR 
        cv2_image = cv2_image[:, :, ::-1]
        return cv2_image

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

def normalize_value(text):
    """
    This removes newlines and multiple spaces from a string.
    """
    result = text.replace('\n', ' ')
    result = re.subn('[ ]{2,}', ' ', result)[0]
    return result

def add_0x(string):
    """Add 0x to string at start.
    """
    if isinstance(string, bytes):
        string = string.decode('utf-8')
    return '0x' + str(string)

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

def unique(seq):
    """Return the unique elements of a collection even if those elements are
       unhashable and unsortable, like dicts and sets"""
    cleaned = []
    for each in seq:
        if each not in cleaned:
            cleaned.append(each)
    return cleaned

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

def replace_list(items, match, replacement):
    """Replaces occurrences of a match string in a given list of strings and returns
    a list of new strings. The match string can be a regex expression.

    Args:
        items (list):       the list of strings to modify.
        match (str):        the search expression.
        replacement (str):  the string to replace with.
    """
    return [replace(item, match, replacement) for item in items]

def make_aware(dt):
    """Appends tzinfo and assumes UTC, if datetime object has no tzinfo already."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

def fmt_subst(regex, subst):
    """Replace regex with string."""
    return lambda text: re.sub(regex, subst, text) if text else text

def Min(a, axis, keep_dims):
    """
    Min reduction op.
    """
    return np.amin(a, axis=axis if not isinstance(axis, np.ndarray) else tuple(axis),
                   keepdims=keep_dims),

def delistify(x):
    """ A basic slug version of a given parameter list. """
    if isinstance(x, list):
        x = [e.replace("'", "") for e in x]
        return '-'.join(sorted(x))
    return x

def __grid_widgets(self):
        """Places all the child widgets in the appropriate positions."""
        scrollbar_column = 0 if self.__compound is tk.LEFT else 2
        self._canvas.grid(row=0, column=1, sticky="nswe")
        self._scrollbar.grid(row=0, column=scrollbar_column, sticky="ns")

def replace(s, replace):
    """Replace multiple values in a string"""
    for r in replace:
        s = s.replace(*r)
    return s

def expect_all(a, b):
    """\
    Asserts that two iterables contain the same values.
    """
    assert all(_a == _b for _a, _b in zip_longest(a, b))

def _comment(string):
    """return string as a comment"""
    lines = [line.strip() for line in string.splitlines()]
    return "# " + ("%s# " % linesep).join(lines)

def _get_set(self, key, operation, create=False):
        """
        Get (and maybe create) a set by name.
        """
        return self._get_by_type(key, operation, create, b'set', set())

def replace_all(text, dic):
    """Takes a string and dictionary. replaces all occurrences of i with j"""

    for i, j in dic.iteritems():
        text = text.replace(i, j)
    return text

def get_abi3_suffix():
    """Return the file extension for an abi3-compliant Extension()"""
    for suffix, _, _ in (s for s in imp.get_suffixes() if s[2] == imp.C_EXTENSION):
        if '.abi3' in suffix:  # Unix
            return suffix
        elif suffix == '.pyd':  # Windows
            return suffix

def replaceNewlines(string, newlineChar):
	"""There's probably a way to do this with string functions but I was lazy.
		Replace all instances of \r or \n in a string with something else."""
	if newlineChar in string:
		segments = string.split(newlineChar)
		string = ""
		for segment in segments:
			string += segment
	return string

def mouse_get_pos():
    """

    :return:
    """
    p = POINT()
    AUTO_IT.AU3_MouseGetPos(ctypes.byref(p))
    return p.x, p.y

def replace(table, field, a, b, **kwargs):
    """
    Convenience function to replace all occurrences of `a` with `b` under the
    given field. See also :func:`convert`.

    The ``where`` keyword argument can be given with a callable or expression
    which is evaluated on each row and which should return True if the
    conversion should be applied on that row, else False.

    """

    return convert(table, field, {a: b}, **kwargs)

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

def submit_the_only_form(self):
    """
    Look for a form on the page and submit it.

    Asserts if more than one form exists.
    """
    form = ElementSelector(world.browser, str('//form'))
    assert form, "Cannot find a form on the page."
    form.submit()

def get_args(method_or_func):
    """Returns method or function arguments."""
    try:
        # Python 3.0+
        args = list(inspect.signature(method_or_func).parameters.keys())
    except AttributeError:
        # Python 2.7
        args = inspect.getargspec(method_or_func).args
    return args

def raise_for_not_ok_status(response):
    """
    Raises a `requests.exceptions.HTTPError` if the response has a non-200
    status code.
    """
    if response.code != OK:
        raise HTTPError('Non-200 response code (%s) for url: %s' % (
            response.code, uridecode(response.request.absoluteURI)))

    return response

def _repr_strip(mystring):
    """
    Returns the string without any initial or final quotes.
    """
    r = repr(mystring)
    if r.startswith("'") and r.endswith("'"):
        return r[1:-1]
    else:
        return r

def new_iteration(self, prefix):
        """When inside a loop logger, created a new iteration
        """
        # Flush data for the current iteration
        self.flush()

        # Fix prefix
        self.prefix[-1] = prefix
        self.reset_formatter()

def batchify(data, batch_size):
    """Reshape data into (num_example, batch_size)"""
    nbatch = data.shape[0] // batch_size
    data = data[:nbatch * batch_size]
    data = data.reshape((batch_size, nbatch)).T
    return data

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

def cross_product_matrix(vec):
    """Returns a 3x3 cross-product matrix from a 3-element vector."""
    return np.array([[0, -vec[2], vec[1]],
                     [vec[2], 0, -vec[0]],
                     [-vec[1], vec[0], 0]])

def raise_for_not_ok_status(response):
    """
    Raises a `requests.exceptions.HTTPError` if the response has a non-200
    status code.
    """
    if response.code != OK:
        raise HTTPError('Non-200 response code (%s) for url: %s' % (
            response.code, uridecode(response.request.absoluteURI)))

    return response

def s3(ctx, bucket_name, data_file, region):
    """Use the S3 SWAG backend."""
    if not ctx.data_file:
        ctx.data_file = data_file

    if not ctx.bucket_name:
        ctx.bucket_name = bucket_name

    if not ctx.region:
        ctx.region = region

    ctx.type = 's3'

def process_request(self, request, response):
        """Logs the basic endpoint requested"""
        self.logger.info('Requested: {0} {1} {2}'.format(request.method, request.relative_uri, request.content_type))

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

def _npiter(arr):
    """Wrapper for iterating numpy array"""
    for a in np.nditer(arr, flags=["refs_ok"]):
        c = a.item()
        if c is not None:
            yield c

def chmod_add_excute(filename):
        """
        Adds execute permission to file.
        :param filename:
        :return:
        """
        st = os.stat(filename)
        os.chmod(filename, st.st_mode | stat.S_IEXEC)

def max(self):
        """
        Returns the maximum value of the domain.

        :rtype: `float` or `np.inf`
        """
        return int(self._max) if not np.isinf(self._max) else self._max

def add_noise(Y, sigma):
    """Adds noise to Y"""
    return Y + np.random.normal(0, sigma, Y.shape)

def make_unique_ngrams(s, n):
    """Make a set of unique n-grams from a string."""
    return set(s[i:i + n] for i in range(len(s) - n + 1))

def generic_add(a, b):
    """Simple function to add two numbers"""
    logger.debug('Called generic_add({}, {})'.format(a, b))
    return a + b

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

def align_file_position(f, size):
    """ Align the position in the file to the next block of specified size """
    align = (size - 1) - (f.tell() % size)
    f.seek(align, 1)

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

def advance_one_line(self):
    """Advances to next line."""

    current_line = self._current_token.line_number
    while current_line == self._current_token.line_number:
      self._current_token = ConfigParser.Token(*next(self._token_generator))

def rotate_img(im, deg, mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_AREA):
    """ Rotates an image by deg degrees

    Arguments:
        deg (float): degree to rotate.
    """
    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c//2,r//2),deg,1)
    return cv2.warpAffine(im,M,(c,r), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS+interpolation)

def _gcd_array(X):
    """
    Return the largest real value h such that all elements in x are integer
    multiples of h.
    """
    greatest_common_divisor = 0.0
    for x in X:
        greatest_common_divisor = _gcd(greatest_common_divisor, x)

    return greatest_common_divisor

def rotateImage(img, angle):
    """

    querries scipy.ndimage.rotate routine
    :param img: image to be rotated
    :param angle: angle to be rotated (radian)
    :return: rotated image
    """
    imgR = scipy.ndimage.rotate(img, angle, reshape=False)
    return imgR

def chunk_sequence(sequence, chunk_length):
    """Yield successive n-sized chunks from l."""
    for index in range(0, len(sequence), chunk_length):
        yield sequence[index:index + chunk_length]

def __round_time(self, dt):
    """Round a datetime object to a multiple of a timedelta
    dt : datetime.datetime object, default now.
    """
    round_to = self._resolution.total_seconds()
    seconds  = (dt - dt.min).seconds
    rounding = (seconds + round_to / 2) // round_to * round_to
    return dt + timedelta(0, rounding - seconds, -dt.microsecond)

def compose_all(tups):
  """Compose all given tuples together."""
  from . import ast  # I weep for humanity
  return functools.reduce(lambda x, y: x.compose(y), map(ast.make_tuple, tups), ast.make_tuple({}))

def _round_half_hour(record):
    """
    Round a time DOWN to half nearest half-hour.
    """
    k = record.datetime + timedelta(minutes=-(record.datetime.minute % 30))
    return datetime(k.year, k.month, k.day, k.hour, k.minute, 0)

def calc_volume(self, sample: np.ndarray):
        """Find the RMS of the audio"""
        return sqrt(np.mean(np.square(sample)))

def round_sig(x, sig):
    """Round the number to the specified number of significant figures"""
    return round(x, sig - int(floor(log10(abs(x)))) - 1)

def covstr(s):
  """ convert string to int or float. """
  try:
    ret = int(s)
  except ValueError:
    ret = float(s)
  return ret

def round_figures(x, n):
    """Returns x rounded to n significant figures."""
    return round(x, int(n - math.ceil(math.log10(abs(x)))))

def index_nearest(value, array):
    """
    expects a _n.array
    returns the global minimum of (value-array)^2
    """

    a = (array-value)**2
    return index(a.min(), a)

def round_to_n(x, n):
    """
    Round to sig figs
    """
    return round(x, -int(np.floor(np.log10(x))) + (n - 1))

def angle(x, y):
    """Return the angle between vectors a and b in degrees."""
    return arccos(dot(x, y)/(norm(x)*norm(y)))*180./pi

def set(self, f):
        """Call a function after a delay, unless another function is set
        in the meantime."""
        self.stop()
        self._create_timer(f)
        self.start()

def flatten(lis):
    """Given a list, possibly nested to any level, return it flattened."""
    new_lis = []
    for item in lis:
        if isinstance(item, collections.Sequence) and not isinstance(item, basestring):
            new_lis.extend(flatten(item))
        else:
            new_lis.append(item)
    return new_lis

def get_code(module):
    """
    Compile and return a Module's code object.
    """
    fp = open(module.path)
    try:
        return compile(fp.read(), str(module.name), 'exec')
    finally:
        fp.close()

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

def store_many(self, sql, values):
        """Abstraction over executemany method"""
        cursor = self.get_cursor()
        cursor.executemany(sql, values)
        self.conn.commit()

def apply(self, node):
        """ Apply transformation and return if an update happened. """
        new_node = self.run(node)
        return self.update, new_node

def web(host, port):
    """Start web application"""
    from .webserver.web import get_app
    get_app().run(host=host, port=port)

def resize_image(self, data, size):
        """ Resizes the given image to fit inside a box of the given size. """
        from machina.core.compat import PILImage as Image
        image = Image.open(BytesIO(data))

        # Resize!
        image.thumbnail(size, Image.ANTIALIAS)

        string = BytesIO()
        image.save(string, format='PNG')
        return string.getvalue()

def compose(func_list):
    """
    composion of preprocessing functions
    """

    def f(G, bim):
        for func in func_list:
            G, bim = func(G, bim)
        return G, bim

    return f

def multiply(traj):
    """Sophisticated simulation of multiplication"""
    z=traj.x*traj.y
    traj.f_add_result('z',z=z, comment='I am the product of two reals!')

def email_type(arg):
	"""An argparse type representing an email address."""
	if not is_valid_email_address(arg):
		raise argparse.ArgumentTypeError("{0} is not a valid email address".format(repr(arg)))
	return arg

def Dump(obj):
  """Stringifies a Python object into its YAML representation.

  Args:
    obj: A Python object to convert to YAML.

  Returns:
    A YAML representation of the given object.
  """
  text = yaml.safe_dump(obj, default_flow_style=False, allow_unicode=True)

  if compatibility.PY2:
    text = text.decode("utf-8")

  return text

def parser():

    """Return a parser for setting one or more configuration paths"""

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_paths', default=[], action='append',
                        help='path to a configuration directory')
    return parser

def save_session(self, sid, session, namespace=None):
        """Store the user session for a client.

        The only difference with the :func:`socketio.Server.save_session`
        method is that when the ``namespace`` argument is not given the
        namespace associated with the class is used.
        """
        return self.server.save_session(
            sid, session, namespace=namespace or self.namespace)

def set_subparsers_args(self, *args, **kwargs):
        """
        Sets args and kwargs that are passed when creating a subparsers group
        in an argparse.ArgumentParser i.e. when calling
        argparser.ArgumentParser.add_subparsers
        """
        self.subparsers_args = args
        self.subparsers_kwargs = kwargs

def on_pause(self):
        """Sync the database with the current state of the game."""
        self.engine.commit()
        self.strings.save()
        self.funcs.save()
        self.config.write()

def createArgumentParser(description):
    """
    Create an argument parser
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=SortedHelpFormatter)
    return parser

def is_scalar(value):
    """Test if the given value is a scalar.

    This function also works with memory mapped array values, in contrast to the numpy is_scalar method.

    Args:
        value: the value to test for being a scalar value

    Returns:
        boolean: if the given value is a scalar or not
    """
    return np.isscalar(value) or (isinstance(value, np.ndarray) and (len(np.squeeze(value).shape) == 0))

def main(args=sys.argv):
    """
    main entry point for the jardiff CLI
    """

    parser = create_optparser(args[0])
    return cli(parser.parse_args(args[1:]))

def selectnotnone(table, field, complement=False):
    """Select rows where the given field is not `None`."""

    return select(table, field, lambda v: v is not None,
                  complement=complement)

def parse_command_args():
    """Command line parser."""
    parser = argparse.ArgumentParser(description='Register PB devices.')
    parser.add_argument('num_pb', type=int,
                        help='Number of PBs devices to register.')
    return parser.parse_args()

def set_subparsers_args(self, *args, **kwargs):
        """
        Sets args and kwargs that are passed when creating a subparsers group
        in an argparse.ArgumentParser i.e. when calling
        argparser.ArgumentParser.add_subparsers
        """
        self.subparsers_args = args
        self.subparsers_kwargs = kwargs

def select_if(df, fun):
    """Selects columns where fun(ction) is true
    Args:
        fun: a function that will be applied to columns
    """

    def _filter_f(col):
        try:
            return fun(df[col])
        except:
            return False

    cols = list(filter(_filter_f, df.columns))
    return df[cols]

def email_type(arg):
	"""An argparse type representing an email address."""
	if not is_valid_email_address(arg):
		raise argparse.ArgumentTypeError("{0} is not a valid email address".format(repr(arg)))
	return arg

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

def is_valid_folder(parser, arg):
    """Check if arg is a valid file that already exists on the file system."""
    arg = os.path.abspath(arg)
    if not os.path.isdir(arg):
        parser.error("The folder %s does not exist!" % arg)
    else:
        return arg

def auth_request(self, url, headers, body):
        """Perform auth request for token."""

        return self.req.post(url, headers, body=body)

def pairwise_indices(self):
        """ndarray containing tuples of pairwise indices."""
        return np.array([sig.pairwise_indices for sig in self.values]).T

def mouseMoveEvent(self, event):
        """ Handle the mouse move event for a drag operation.

        """
        self.declaration.mouse_move_event(event)
        super(QtGraphicsView, self).mouseMoveEvent(event)

def dump_nparray(self, obj, class_name=numpy_ndarray_class_name):
        """
        ``numpy.ndarray`` dumper.
        """
        return {"$" + class_name: self._json_convert(obj.tolist())}

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

def Diag(a):
    """
    Diag op.
    """
    r = np.zeros(2 * a.shape, dtype=a.dtype)
    for idx, v in np.ndenumerate(a):
        r[2 * idx] = v
    return r,

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

def is_scalar(value):
    """Test if the given value is a scalar.

    This function also works with memory mapped array values, in contrast to the numpy is_scalar method.

    Args:
        value: the value to test for being a scalar value

    Returns:
        boolean: if the given value is a scalar or not
    """
    return np.isscalar(value) or (isinstance(value, np.ndarray) and (len(np.squeeze(value).shape) == 0))

def set_axis_options(self, row, column, text):
        """Set additionnal options as plain text."""

        subplot = self.get_subplot_at(row, column)
        subplot.set_axis_options(text)

def _go_to_line(editor, line):
    """
    Move cursor to this line in the current buffer.
    """
    b = editor.application.current_buffer
    b.cursor_position = b.document.translate_row_col_to_index(max(0, int(line) - 1), 0)

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

def rex_assert(self, rex, byte=False):
        """
        If `rex` expression is not found then raise `DataNotFound` exception.
        """

        self.rex_search(rex, byte=byte)

def _init_unique_sets(self):
        """Initialise sets used for uniqueness checking."""

        ks = dict()
        for t in self._unique_checks:
            key = t[0]
            ks[key] = set() # empty set
        return ks

def contains(self, element):
        """
        Ensures :attr:`subject` contains *other*.
        """
        self._run(unittest_case.assertIn, (element, self._subject))
        return ChainInspector(self._subject)

def setup(self, proxystr='', prompting=True):
        """
        Sets the proxy handler given the option passed on the command
        line.  If an empty string is passed it looks at the HTTP_PROXY
        environment variable.
        """
        self.prompting = prompting
        proxy = self.get_proxy(proxystr)
        if proxy:
            proxy_support = urllib2.ProxyHandler({"http": proxy, "ftp": proxy})
            opener = urllib2.build_opener(proxy_support, urllib2.CacheFTPHandler)
            urllib2.install_opener(opener)

def assert_list(self, putative_list, expected_type=string_types, key_arg=None):
    """
    :API: public
    """
    return assert_list(putative_list, expected_type, key_arg=key_arg,
                       raise_type=lambda msg: TargetDefinitionException(self, msg))

def resize(self, width, height):
        """
        Pyqt specific resize callback.
        """
        if not self.fbo:
            return

        # pyqt reports sizes in actual buffer size
        self.width = width // self.widget.devicePixelRatio()
        self.height = height // self.widget.devicePixelRatio()
        self.buffer_width = width
        self.buffer_height = height

        super().resize(width, height)

def assert_in(obj, seq, message=None, extra=None):
    """Raises an AssertionError if obj is not in seq."""
    assert obj in seq, _assert_fail_message(message, obj, seq, "is not in", extra)

def get_table_width(table):
    """
    Gets the width of the table that would be printed.
    :rtype: ``int``
    """
    columns = transpose_table(prepare_rows(table))
    widths = [max(len(cell) for cell in column) for column in columns]
    return len('+' + '|'.join('-' * (w + 2) for w in widths) + '+')

def assert_is_instance(value, types, message=None, extra=None):
    """Raises an AssertionError if value is not an instance of type(s)."""
    assert isinstance(value, types), _assert_fail_message(
        message, value, types, "is not an instance of", extra
    )

def set_position(self, x, y, width, height):
        """Set window top-left corner position and size"""
        SetWindowPos(self._hwnd, None, x, y, width, height, ctypes.c_uint(0))

def log_y_cb(self, w, val):
        """Toggle linear/log scale for Y-axis."""
        self.tab_plot.logy = val
        self.plot_two_columns()

def delegate(self, fn, *args, **kwargs):
        """Return the given operation as an asyncio future."""
        callback = functools.partial(fn, *args, **kwargs)
        coro = self.loop.run_in_executor(self.subexecutor, callback)
        return asyncio.ensure_future(coro)

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

def run_task(func):
    """
    Decorator to wrap an async function in an event loop.
    Use for main sync interface methods.
    """

    def _wrapped(*a, **k):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(func(*a, **k))

    return _wrapped

def set_mem_per_proc(self, mem_mb):
        """Set the memory per process in megabytes"""
        super().set_mem_per_proc(mem_mb)
        self.qparams["mem_per_cpu"] = self.mem_per_proc

def safe_repr(obj):
    """
    Try to get ``__name__`` first, ``__class__.__name__`` second
    and finally, if we can't get anything acceptable, fallback
    to user a ``repr()`` call.
    """
    name = getattr(obj, '__name__', getattr(obj.__class__, '__name__'))
    if name == 'ndict':
        name = 'dict'
    return name or repr(obj)

async def smap(source, func, *more_sources):
    """Apply a given function to the elements of one or several
    asynchronous sequences.

    Each element is used as a positional argument, using the same order as
    their respective sources. The generation continues until the shortest
    sequence is exhausted. The function is treated synchronously.

    Note: if more than one sequence is provided, they're awaited concurrently
    so that their waiting times don't add up.
    """
    if more_sources:
        source = zip(source, *more_sources)
    async with streamcontext(source) as streamer:
        async for item in streamer:
            yield func(*item) if more_sources else func(item)

def out_shape_from_array(arr):
    """Get the output shape from an array."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.shape
    else:
        return (arr.shape[1],)

async def wait_and_quit(loop):
	"""Wait until all task are executed."""
	from pylp.lib.tasks import running
	if running:
		await asyncio.wait(map(lambda runner: runner.future, running))

def align_file_position(f, size):
    """ Align the position in the file to the next block of specified size """
    align = (size - 1) - (f.tell() % size)
    f.seek(align, 1)

def safe_setattr(obj, name, value):
    """Attempt to setattr but catch AttributeErrors."""
    try:
        setattr(obj, name, value)
        return True
    except AttributeError:
        return False

def torecarray(*args, **kwargs):
    """
    Convenient shorthand for ``toarray(*args, **kwargs).view(np.recarray)``.

    """

    import numpy as np
    return toarray(*args, **kwargs).view(np.recarray)

def load(self):
        """Load proxy list from configured proxy source"""
        self._list = self._source.load()
        self._list_iter = itertools.cycle(self._list)

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

def average_gradient(data, *kwargs):
    """ Compute average gradient norm of an image
    """
    return np.average(np.array(np.gradient(data))**2)

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

def list_rds(region, filter_by_kwargs):
    """List all RDS thingys."""
    conn = boto.rds.connect_to_region(region)
    instances = conn.get_all_dbinstances()
    return lookup(instances, filter_by=filter_by_kwargs)

def __init__(self, enumtype, index, key):
        """ Set up a new instance. """
        self._enumtype = enumtype
        self._index = index
        self._key = key

def show_xticklabels(self, row, column):
        """Show the x-axis tick labels for a subplot.

        :param row,column: specify the subplot.

        """
        subplot = self.get_subplot_at(row, column)
        subplot.show_xticklabels()

def finish_plot():
    """Helper for plotting."""
    plt.legend()
    plt.grid(color='0.7')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def _bytes_to_json(value):
    """Coerce 'value' to an JSON-compatible representation."""
    if isinstance(value, bytes):
        value = base64.standard_b64encode(value).decode("ascii")
    return value

def bin_to_int(string):
    """Convert a one element byte string to signed int for python 2 support."""
    if isinstance(string, str):
        return struct.unpack("b", string)[0]
    else:
        return struct.unpack("b", bytes([string]))[0]

def get_url(self, routename, **kargs):
        """ Return a string that matches a named route """
        return '/' + self.routes.build(routename, **kargs).split(';', 1)[1]

def _skip_frame(self):
        """Skip the next time frame"""
        for line in self._f:
            if line == 'ITEM: ATOMS\n':
                break
        for i in range(self.num_atoms):
            next(self._f)

def mean_cl_boot(series, n_samples=1000, confidence_interval=0.95,
                 random_state=None):
    """
    Bootstrapped mean with confidence limits
    """
    return bootstrap_statistics(series, np.mean,
                                n_samples=n_samples,
                                confidence_interval=confidence_interval,
                                random_state=random_state)

def test3():
    """Test the multiprocess
    """
    import time
    
    p = MVisionProcess()
    p.start()
    time.sleep(5)
    p.stop()

def remove_file_from_s3(awsclient, bucket, key):
    """Remove a file from an AWS S3 bucket.

    :param awsclient:
    :param bucket:
    :param key:
    :return:
    """
    client_s3 = awsclient.get_client('s3')
    response = client_s3.delete_object(Bucket=bucket, Key=key)

def getcolslice(self, blc, trc, inc=[], startrow=0, nrow=-1, rowincr=1):
        """Get a slice from a table column holding arrays.
        (see :func:`table.getcolslice`)"""
        return self._table.getcolslice(self._column, blc, trc, inc, startrow, nrow, rowincr)

def start():
    """Starts the web server."""
    global app
    bottle.run(app, host=conf.WebHost, port=conf.WebPort,
               debug=conf.WebAutoReload, reloader=conf.WebAutoReload,
               quiet=conf.WebQuiet)

def split(s):
  """Uses dynamic programming to infer the location of spaces in a string without spaces."""
  l = [_split(x) for x in _SPLIT_RE.split(s)]
  return [item for sublist in l for item in sublist]

def get_http_method(self, method):
        """Gets the http method that will be called from the requests library"""
        return self.http_methods[method](self.url, **self.http_method_args)

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

def __sort_up(self):

        """Sort the updatable objects according to ascending order"""
        if self.__do_need_sort_up:
            self.__up_objects.sort(key=cmp_to_key(self.__up_cmp))
            self.__do_need_sort_up = False

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

def unsort_vector(data, indices_of_increasing):
    """Upermutate 1-D data that is sorted by indices_of_increasing."""
    return numpy.array([data[indices_of_increasing.index(i)] for i in range(len(data))])

def poke_array(self, store, name, elemtype, elements, container, visited, _stack):
        """abstract method"""
        raise NotImplementedError

def chunked(l, n):
    """Chunk one big list into few small lists."""
    return [l[i:i + n] for i in range(0, len(l), n)]

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

def split_into_words(s):
  """Split a sentence into list of words."""
  s = re.sub(r"\W+", " ", s)
  s = re.sub(r"[_0-9]+", " ", s)
  return s.split()

def fmt_sz(intval):
    """ Format a byte sized value.
    """
    try:
        return fmt.human_size(intval)
    except (ValueError, TypeError):
        return "N/A".rjust(len(fmt.human_size(0)))

def _split_str(s, n):
    """
    split string into list of strings by specified number.
    """
    length = len(s)
    return [s[i:i + n] for i in range(0, length, n)]

def cleanup_lib(self):
        """ unload the previously loaded shared library """
        if not self.using_openmp:
            #this if statement is necessary because shared libraries that use
            #OpenMP will core dump when unloaded, this is a well-known issue with OpenMP
            logging.debug('unloading shared library')
            _ctypes.dlclose(self.lib._handle)

def _split_str(s, n):
    """
    split string into list of strings by specified number.
    """
    length = len(s)
    return [s[i:i + n] for i in range(0, length, n)]

def _get_memoized_value(func, args, kwargs):
    """Used internally by memoize decorator to get/store function results"""
    key = (repr(args), repr(kwargs))

    if not key in func._cache_dict:
        ret = func(*args, **kwargs)
        func._cache_dict[key] = ret

    return func._cache_dict[key]

def locked_delete(self):
        """Delete credentials from the SQLAlchemy datastore."""
        filters = {self.key_name: self.key_value}
        self.session.query(self.model_class).filter_by(**filters).delete()

def triangle_normal(a, b, c):
    """Return a vector orthogonal to the given triangle

       Arguments:
         a, b, c  --  three 3D numpy vectors
    """
    normal = np.cross(a - c, b - c)
    norm = np.linalg.norm(normal)
    return normal/norm

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

def column_stack_2d(data):
    """Perform column-stacking on a list of 2d data blocks."""
    return list(list(itt.chain.from_iterable(_)) for _ in zip(*data))

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

def post_ratelimited(protocol, session, url, headers, data, allow_redirects=False, stream=False):
    """
    There are two error-handling policies implemented here: a fail-fast policy intended for stand-alone scripts which
    fails on all responses except HTTP 200. The other policy is intended for long-running tasks that need to respect
    rate-limiting errors from the server and paper over outages of up to 1 hour.

    Wrap POST requests in a try-catch loop with a lot of error handling logic and some basic rate-limiting. If a request
    fails, and some conditions are met, the loop waits in increasing intervals, up to 1 hour, before trying again. The
    reason for this is that servers often malfunction for short periods of time, either because of ongoing data
    migrations or other maintenance tasks, misconfigurations or heavy load, or because the connecting user has hit a
    throttling policy limit.

    If the loop exited early, consumers of this package that don't implement their own rate-limiting code could quickly
    swamp such a server with new requests. That would only make things worse. Instead, it's better if the request loop
    waits patiently until the server is functioning again.

    If the connecting user has hit a throttling policy, then the server will start to malfunction in many interesting
    ways, but never actually tell the user what is happening. There is no way to distinguish this situation from other
    malfunctions. The only cure is to stop making requests.

    The contract on sessions here is to return the session that ends up being used, or retiring the session if we
    intend to raise an exception. We give up on max_wait timeout, not number of retries.

    An additional resource on handling throttling policies and client back off strategies:
        https://msdn.microsoft.com/en-us/library/office/jj945066(v=exchg.150).aspx#bk_ThrottlingBatch
    """
    thread_id = get_ident()
    wait = 10  # seconds
    retry = 0
    redirects = 0
    # In Python 2, we want this to be a 'str' object so logging doesn't break (all formatting arguments are 'str').
    # We activated 'unicode_literals' at the top of this file, so it would be a 'unicode' object unless we convert
    # to 'str' explicitly. This is a no-op for Python 3.
    log_msg = str('''\
Retry: %(retry)s
Waited: %(wait)s
Timeout: %(timeout)s
Session: %(session_id)s
Thread: %(thread_id)s
Auth type: %(auth)s
URL: %(url)s
HTTP adapter: %(adapter)s
Allow redirects: %(allow_redirects)s
Streaming: %(stream)s
Response time: %(response_time)s
Status code: %(status_code)s
Request headers: %(request_headers)s
Response headers: %(response_headers)s
Request data: %(xml_request)s
Response data: %(xml_response)s
''')
    log_vals = dict(
        retry=retry,
        wait=wait,
        timeout=protocol.TIMEOUT,
        session_id=session.session_id,
        thread_id=thread_id,
        auth=session.auth,
        url=url,
        adapter=session.get_adapter(url),
        allow_redirects=allow_redirects,
        stream=stream,
        response_time=None,
        status_code=None,
        request_headers=headers,
        response_headers=None,
        xml_request=data,
        xml_response=None,
    )
    try:
        while True:
            _back_off_if_needed(protocol.credentials.back_off_until)
            log.debug('Session %s thread %s: retry %s timeout %s POST\'ing to %s after %ss wait', session.session_id,
                      thread_id, retry, protocol.TIMEOUT, url, wait)
            d_start = time_func()
            # Always create a dummy response for logging purposes, in case we fail in the following
            r = DummyResponse(url=url, headers={}, request_headers=headers)
            try:
                r = session.post(url=url, headers=headers, data=data, allow_redirects=False, timeout=protocol.TIMEOUT,
                                 stream=stream)
            except CONNECTION_ERRORS as e:
                log.debug('Session %s thread %s: connection error POST\'ing to %s', session.session_id, thread_id, url)
                r = DummyResponse(url=url, headers={'TimeoutException': e}, request_headers=headers)
            finally:
                log_vals.update(
                    retry=retry,
                    wait=wait,
                    session_id=session.session_id,
                    url=str(r.url),
                    response_time=time_func() - d_start,
                    status_code=r.status_code,
                    request_headers=r.request.headers,
                    response_headers=r.headers,
                    xml_response='[STREAMING]' if stream else r.content,
                )
            log.debug(log_msg, log_vals)
            if _may_retry_on_error(r, protocol, wait):
                log.info("Session %s thread %s: Connection error on URL %s (code %s). Cool down %s secs",
                         session.session_id, thread_id, r.url, r.status_code, wait)
                time.sleep(wait)  # Increase delay for every retry
                retry += 1
                wait *= 2
                session = protocol.renew_session(session)
                continue
            if r.status_code in (301, 302):
                if stream:
                    r.close()
                url, redirects = _redirect_or_fail(r, redirects, allow_redirects)
                continue
            break
    except (RateLimitError, RedirectError) as e:
        log.warning(e.value)
        protocol.retire_session(session)
        raise
    except Exception as e:
        # Let higher layers handle this. Add full context for better debugging.
        log.error(str('%s: %s\n%s'), e.__class__.__name__, str(e), log_msg % log_vals)
        protocol.retire_session(session)
        raise
    if r.status_code == 500 and r.content and is_xml(r.content):
        # Some genius at Microsoft thinks it's OK to send a valid SOAP response as an HTTP 500
        log.debug('Got status code %s but trying to parse content anyway', r.status_code)
    elif r.status_code != 200:
        protocol.retire_session(session)
        try:
            _raise_response_errors(r, protocol, log_msg, log_vals)  # Always raises an exception
        finally:
            if stream:
                r.close()
    log.debug('Session %s thread %s: Useful response from %s', session.session_id, thread_id, url)
    return r, session

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

def get_previous_month(self):
        """Returns date range for the previous full month."""
        end = utils.get_month_start() - relativedelta(days=1)
        end = utils.to_datetime(end)
        start = utils.get_month_start(end)
        return start, end

def _hue(color, **kwargs):
    """ Get hue value of HSL color.
    """
    h = colorsys.rgb_to_hls(*[x / 255.0 for x in color.value[:3]])[0]
    return NumberValue(h * 360.0)

def weighted_std(values, weights):
    """ Calculate standard deviation weighted by errors """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return np.sqrt(variance)

def estimate_complexity(self, x,y,z,n):
        """ 
        calculates a rough guess of runtime based on product of parameters 
        """
        num_calculations = x * y * z * n
        run_time = num_calculations / 100000  # a 2014 PC does about 100k calcs in a second (guess based on prior logs)
        return self.show_time_as_short_string(run_time)

def _update_staticmethod(self, oldsm, newsm):
        """Update a staticmethod update."""
        # While we can't modify the staticmethod object itself (it has no
        # mutable attributes), we *can* extract the underlying function
        # (by calling __get__(), which returns it) and update it in-place.
        # We don't have the class available to pass to __get__() but any
        # object except None will do.
        self._update(None, None, oldsm.__get__(0), newsm.__get__(0))

def submit(self, fn, *args, **kwargs):
        """Submit an operation"""
        corofn = asyncio.coroutine(lambda: fn(*args, **kwargs))
        return run_coroutine_threadsafe(corofn(), self.loop)

def stop(self):
		""" Stops the video stream and resets the clock. """

		logger.debug("Stopping playback")
		# Stop the clock
		self.clock.stop()
		# Set plauyer status to ready
		self.status = READY

async def _thread_coro(self, *args):
        """ Coroutine called by MapAsync. It's wrapping the call of
        run_in_executor to run the synchronous function as thread """
        return await self._loop.run_in_executor(
            self._executor, self._function, *args)

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

def fail(message=None, exit_status=None):
    """Prints the specified message and exits the program with the specified
    exit status.

    """
    print('Error:', message, file=sys.stderr)
    sys.exit(exit_status or 1)

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

def mixedcase(path):
    """Removes underscores and capitalizes the neighbouring character"""
    words = path.split('_')
    return words[0] + ''.join(word.title() for word in words[1:])

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

def add_suffix(fullname, suffix):
    """ Add suffix to a full file name"""
    name, ext = os.path.splitext(fullname)
    return name + '_' + suffix + ext

def _encode_bool(name, value, dummy0, dummy1):
    """Encode a python boolean (True/False)."""
    return b"\x08" + name + (value and b"\x01" or b"\x00")

def angle_to_cartesian(lon, lat):
    """Convert spherical coordinates to cartesian unit vectors."""
    theta = np.array(np.pi / 2. - lat)
    return np.vstack((np.sin(theta) * np.cos(lon),
                      np.sin(theta) * np.sin(lon),
                      np.cos(theta))).T

def one_hot2string(arr, vocab):
    """Convert a one-hot encoded array back to string
    """
    tokens = one_hot2token(arr)
    indexToLetter = _get_index_dict(vocab)

    return [''.join([indexToLetter[x] for x in row]) for row in tokens]

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

def get_title(soup):
  """Given a soup, pick out a title"""
  if soup.title:
    return soup.title.string
  if soup.h1:
    return soup.h1.string
  return ''

def cart2pol(x, y):
    """Cartesian to Polar coordinates conversion."""
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho

def covstr(s):
  """ convert string to int or float. """
  try:
    ret = int(s)
  except ValueError:
    ret = float(s)
  return ret

def get_lons_from_cartesian(x__, y__):
    """Get longitudes from cartesian coordinates.
    """
    return rad2deg(arccos(x__ / sqrt(x__ ** 2 + y__ ** 2))) * sign(y__)

def text_cleanup(data, key, last_type):
    """ I strip extra whitespace off multi-line strings if they are ready to be stripped!"""
    if key in data and last_type == STRING_TYPE:
        data[key] = data[key].strip()
    return data

def load(self, name):
        """Loads and returns foreign library."""
        name = ctypes.util.find_library(name)
        return ctypes.cdll.LoadLibrary(name)

def to_bytes(s, encoding="utf-8"):
    """Convert a string to bytes."""
    if isinstance(s, six.binary_type):
        return s
    if six.PY3:
        return bytes(s, encoding)
    return s.encode(encoding)

def pad_image(arr, max_size=400):
    """Pads an image to a square then resamples to max_size"""
    dim = np.max(arr.shape)
    img = np.zeros((dim, dim, 3), dtype=arr.dtype)
    xl = (dim - arr.shape[0]) // 2
    yl = (dim - arr.shape[1]) // 2
    img[xl:arr.shape[0]+xl, yl:arr.shape[1]+yl, :] = arr
    return resample_image(img, max_size=max_size)

def _parse(self, date_str, format='%Y-%m-%d'):
        """
        helper function for parsing FRED date string into datetime
        """
        rv = pd.to_datetime(date_str, format=format)
        if hasattr(rv, 'to_pydatetime'):
            rv = rv.to_pydatetime()
        return rv

def QA_util_datetime_to_strdate(dt):
    """
    :param dt:  pythone datetime.datetime
    :return:  1999-02-01 string type
    """
    strdate = "%04d-%02d-%02d" % (dt.year, dt.month, dt.day)
    return strdate

def do_striptags(value):
    """Strip SGML/XML tags and replace adjacent whitespace by one space.
    """
    if hasattr(value, '__html__'):
        value = value.__html__()
    return Markup(unicode(value)).striptags()

def inc_date(date_obj, num, date_fmt):
    """Increment the date by a certain number and return date object.
    as the specific string format.
    """
    return (date_obj + timedelta(days=num)).strftime(date_fmt)

def text_remove_empty_lines(text):
    """
    Whitespace normalization:

      - Strip empty lines
      - Strip trailing whitespace
    """
    lines = [ line.rstrip()  for line in text.splitlines()  if line.strip() ]
    return "\n".join(lines)

def convert_camel_case_to_snake_case(name):
    """Convert CamelCase to snake_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def _strptime(self, time_str):
        """Convert an ISO 8601 formatted string in UTC into a
        timezone-aware datetime object."""
        if time_str:
            # Parse UTC string into naive datetime, then add timezone
            dt = datetime.strptime(time_str, __timeformat__)
            return dt.replace(tzinfo=UTC())
        return None

def is_iterable_of_int(l):
    r""" Checks if l is iterable and contains only integral types """
    if not is_iterable(l):
        return False

    return all(is_int(value) for value in l)

def pylog(self, *args, **kwargs):
        """Display all available logging information."""
        printerr(self.name, args, kwargs, traceback.format_exc())

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

def _svd(cls, matrix, num_concepts=5):
        """
        Perform singular value decomposition for dimensionality reduction of the input matrix.
        """
        u, s, v = svds(matrix, k=num_concepts)
        return u, s, v

def writer_acquire(self):
        """Acquire the lock to write"""

        self._order_mutex.acquire()
        self._access_mutex.acquire()
        self._order_mutex.release()

def imp_print(self, text, end):
		"""Directly send utf8 bytes to stdout"""
		sys.stdout.write((text + end).encode("utf-8"))

def set_executable(filename):
    """Set the exectuable bit on the given filename"""
    st = os.stat(filename)
    os.chmod(filename, st.st_mode | stat.S_IEXEC)

def h5ToDict(h5, readH5pyDataset=True):
    """ Read a hdf5 file into a dictionary """
    h = h5py.File(h5, "r")
    ret = unwrapArray(h, recursive=True, readH5pyDataset=readH5pyDataset)
    if readH5pyDataset: h.close()
    return ret

def datetime_to_timezone(date, tz="UTC"):
    """ convert naive datetime to timezone-aware datetime """
    if not date.tzinfo:
        date = date.replace(tzinfo=timezone(get_timezone()))
    return date.astimezone(timezone(tz))

def get_table_width(table):
    """
    Gets the width of the table that would be printed.
    :rtype: ``int``
    """
    columns = transpose_table(prepare_rows(table))
    widths = [max(len(cell) for cell in column) for column in columns]
    return len('+' + '|'.join('-' * (w + 2) for w in widths) + '+')

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

def strToBool(val):
    """
    Helper function to turn a string representation of "true" into
    boolean True.
    """
    if isinstance(val, str):
        val = val.lower()

    return val in ['true', 'on', 'yes', True]

def _validate_pos(df):
    """Validates the returned positional object
    """
    assert isinstance(df, pd.DataFrame)
    assert ["seqname", "position", "strand"] == df.columns.tolist()
    assert df.position.dtype == np.dtype("int64")
    assert df.strand.dtype == np.dtype("O")
    assert df.seqname.dtype == np.dtype("O")
    return df

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

def _check_key(self, key):
        """
        Ensures well-formedness of a key.
        """
        if not len(key) == 2:
            raise TypeError('invalid key: %r' % key)
        elif key[1] not in TYPES:
            raise TypeError('invalid datatype: %s' % key[1])

def get_decimal_quantum(precision):
    """Return minimal quantum of a number, as defined by precision."""
    assert isinstance(precision, (int, decimal.Decimal))
    return decimal.Decimal(10) ** (-precision)

def required_attributes(element, *attributes):
    """Check element for required attributes. Raise ``NotValidXmlException`` on error.

    :param element: ElementTree element
    :param attributes: list of attributes names to check
    :raises NotValidXmlException: if some argument is missing
    """
    if not reduce(lambda still_valid, param: still_valid and param in element.attrib, attributes, True):
        raise NotValidXmlException(msg_err_missing_attributes(element.tag, *attributes))

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

def _valid_other_type(x, types):
    """
    Do all elements of x have a type from types?
    """
    return all(any(isinstance(el, t) for t in types) for el in np.ravel(x))

def afx_small():
  """Small transformer model with small batch size for fast step times."""
  hparams = transformer.transformer_tpu()
  hparams.filter_size = 1024
  hparams.num_heads = 4
  hparams.num_hidden_layers = 3
  hparams.batch_size = 512
  return hparams

def _is_already_configured(configuration_details):
    """Returns `True` when alias already in shell config."""
    path = Path(configuration_details.path).expanduser()
    with path.open('r') as shell_config:
        return configuration_details.content in shell_config.read()

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

def _replace_file(path, content):
  """Writes a file if it doesn't already exist with the same content.

  This is useful because cargo uses timestamps to decide whether to compile things."""
  if os.path.exists(path):
    with open(path, 'r') as f:
      if content == f.read():
        print("Not overwriting {} because it is unchanged".format(path), file=sys.stderr)
        return

  with open(path, 'w') as f:
    f.write(content)

def is_function(self):
        """return True if callback is a vanilla plain jane function"""
        if self.is_instance() or self.is_class(): return False
        return isinstance(self.callback, (Callable, classmethod))

def clean_float(v):
    """Remove commas from a float"""

    if v is None or not str(v).strip():
        return None

    return float(str(v).replace(',', ''))

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

def _is_one_arg_pos_call(call):
    """Is this a call with exactly 1 argument,
    where that argument is positional?
    """
    return isinstance(call, astroid.Call) and len(call.args) == 1 and not call.keywords

def wait_and_join(self, task):
        """ Given a task, waits for it until it finishes
        :param task: Task
        :return:
        """
        while not task.has_started:
            time.sleep(self._polling_time)
        task.thread.join()

def is_changed():
    """ Checks if current project has any noncommited changes. """
    executed, changed_lines = execute_git('status --porcelain', output=False)
    merge_not_finished = mod_path.exists('.git/MERGE_HEAD')
    return changed_lines.strip() or merge_not_finished

def clear_timeline(self):
        """
        Clear the contents of the TimeLine Canvas

        Does not modify the actual markers dictionary and thus after
        redrawing all markers are visible again.
        """
        self._timeline.delete(tk.ALL)
        self._canvas_ticks.delete(tk.ALL)

def ensure_dir_exists(directory):
    """Se asegura de que un directorio exista."""
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

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

def is_nullable_list(val, vtype):
    """Return True if list contains either values of type `vtype` or None."""
    return (isinstance(val, list) and
            any(isinstance(v, vtype) for v in val) and
            all((isinstance(v, vtype) or v is None) for v in val))

def listified_tokenizer(source):
    """Tokenizes *source* and returns the tokens as a list of lists."""
    io_obj = io.StringIO(source)
    return [list(a) for a in tokenize.generate_tokens(io_obj.readline)]

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

def print_ldamodel_topic_words(topic_word_distrib, vocab, n_top=10, row_labels=DEFAULT_TOPIC_NAME_FMT):
    """Print `n_top` values from a LDA model's topic-word distributions."""
    print_ldamodel_distribution(topic_word_distrib, row_labels=row_labels, val_labels=vocab,
                                top_n=n_top)

def drop_trailing_zeros(num):
    """
    Drops the trailing zeros in a float that is printed.
    """
    txt = '%f' %(num)
    txt = txt.rstrip('0')
    if txt.endswith('.'):
        txt = txt[:-1]
    return txt

def _check_elements_equal(lst):
    """
    Returns true if all of the elements in the list are equal.
    """
    assert isinstance(lst, list), "Input value must be a list."
    return not lst or lst.count(lst[0]) == len(lst)

def bitsToString(arr):
  """Returns a string representing a numpy array of 0's and 1's"""
  s = array('c','.'*len(arr))
  for i in xrange(len(arr)):
    if arr[i] == 1:
      s[i]='*'
  return s

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

def html_to_text(content):
    """ Converts html content to plain text """
    text = None
    h2t = html2text.HTML2Text()
    h2t.ignore_links = False
    text = h2t.handle(content)
    return text

def isstring(value):
    """Report whether the given value is a byte or unicode string."""
    classes = (str, bytes) if pyutils.PY3 else basestring  # noqa: F821
    return isinstance(value, classes)

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

def is_collection(obj):
    """Tests if an object is a collection."""

    col = getattr(obj, '__getitem__', False)
    val = False if (not col) else True

    if isinstance(obj, basestring):
        val = False

    return val

def command_py2to3(args):
    """
    Apply '2to3' tool (Python2 to Python3 conversion tool) to Python sources.
    """
    from lib2to3.main import main
    sys.exit(main("lib2to3.fixes", args=args.sources))

def str_check(*args, func=None):
    """Check if arguments are str type."""
    func = func or inspect.stack()[2][3]
    for var in args:
        if not isinstance(var, (str, collections.UserString, collections.abc.Sequence)):
            name = type(var).__name__
            raise StringError(
                f'Function {func} expected str, {name} got instead.')

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

def walk_tree(root):
    """Pre-order depth-first"""
    yield root

    for child in root.children:
        for el in walk_tree(child):
            yield el

def is_int_type(val):
    """Return True if `val` is of integer type."""
    try:               # Python 2
        return isinstance(val, (int, long))
    except NameError:  # Python 3
        return isinstance(val, int)

def __contains__(self, key):
        """
        Invoked when determining whether a specific key is in the dictionary
        using `key in d`.

        The key is looked up case-insensitively.
        """
        k = self._real_key(key)
        return k in self._data

def set_scrollbars_cb(self, w, tf):
        """This callback is invoked when the user checks the 'Use Scrollbars'
        box in the preferences pane."""
        scrollbars = 'on' if tf else 'off'
        self.t_.set(scrollbars=scrollbars)

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

def from_json_list(cls, api_client, data):
        """Convert a list of JSON values to a list of models
        """
        return [cls.from_json(api_client, item) for item in data]

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

def string_to_list(string, sep=",", filter_empty=False):
    """Transforma una string con elementos separados por `sep` en una lista."""
    return [value.strip() for value in string.split(sep)
            if (not filter_empty or value)]

def is_closed(self):
        """ Check if session was closed. """
        return (self.state == SESSION_STATE.CLOSED 
                or self.state == SESSION_STATE.CLOSING)

def xml_str_to_dict(s):
    """ Transforms an XML string it to python-zimbra dict format

    For format, see:
      https://github.com/Zimbra-Community/python-zimbra/blob/master/README.md

    :param: a string, containing XML
    :returns: a dict, with python-zimbra format
    """
    xml = minidom.parseString(s)
    return pythonzimbra.tools.xmlserializer.dom_to_dict(xml.firstChild)

def _lookup_enum_in_ns(namespace, value):
    """Return the attribute of namespace corresponding to value."""
    for attribute in dir(namespace):
        if getattr(namespace, attribute) == value:
            return attribute

def list2dict(lst):
    """Takes a list of (key,value) pairs and turns it into a dict."""

    dic = {}
    for k,v in lst: dic[k] = v
    return dic

def dt_to_ts(value):
    """ If value is a datetime, convert to timestamp """
    if not isinstance(value, datetime):
        return value
    return calendar.timegm(value.utctimetuple()) + value.microsecond / 1000000.0

def bytes_to_str(s, encoding='utf-8'):
    """Returns a str if a bytes object is given."""
    if six.PY3 and isinstance(s, bytes):
        return s.decode(encoding)
    return s

def FromString(self, string):
    """Parse a bool from a string."""
    if string.lower() in ("false", "no", "n"):
      return False

    if string.lower() in ("true", "yes", "y"):
      return True

    raise TypeValueError("%s is not recognized as a boolean value." % string)

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

def is_readable(filename):
    """Check if file is a regular file and is readable."""
    return os.path.isfile(filename) and os.access(filename, os.R_OK)

def symbol_pos_int(*args, **kwargs):
    """Create a sympy.Symbol with positive and integer assumptions."""
    kwargs.update({'positive': True,
                   'integer': True})
    return sympy.Symbol(*args, **kwargs)

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

def camel_case_from_underscores(string):
    """generate a CamelCase string from an underscore_string."""
    components = string.split('_')
    string = ''
    for component in components:
        string += component[0].upper() + component[1:]
    return string

def column_exists(cr, table, column):
    """ Check whether a certain column exists """
    cr.execute(
        'SELECT count(attname) FROM pg_attribute '
        'WHERE attrelid = '
        '( SELECT oid FROM pg_class WHERE relname = %s ) '
        'AND attname = %s',
        (table, column))
    return cr.fetchone()[0] == 1

def unique(iterable):
    """ Returns a list copy in which each item occurs only once (in-order).
    """
    seen = set()
    return [x for x in iterable if x not in seen and not seen.add(x)]

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

def _from_bytes(bytes, byteorder="big", signed=False):
    """This is the same functionality as ``int.from_bytes`` in python 3"""
    return int.from_bytes(bytes, byteorder=byteorder, signed=signed)

def set_locale(request):
    """Return locale from GET lang param or automatically."""
    return request.query.get('lang', app.ps.babel.select_locale_by_request(request))

def isnamedtuple(obj):
    """Heuristic check if an object is a namedtuple."""
    return isinstance(obj, tuple) \
           and hasattr(obj, "_fields") \
           and hasattr(obj, "_asdict") \
           and callable(obj._asdict)

def get_file_name(url):
  """Returns file name of file at given url."""
  return os.path.basename(urllib.parse.urlparse(url).path) or 'unknown_name'

def getConnectionStats(self):
        """Returns dictionary with number of connections for each database.
        
        @return: Dictionary of database connection statistics.
        
        """
        cur = self._conn.cursor()
        cur.execute("""SELECT datname,numbackends FROM pg_stat_database;""")
        rows = cur.fetchall()
        if rows:
            return dict(rows)
        else:
            return {}

def _stdin_ready_posix():
    """Return True if there's something to read on stdin (posix version)."""
    infds, outfds, erfds = select.select([sys.stdin],[],[],0)
    return bool(infds)

def _check_model(obj, models=None):
    """Checks object if it's a peewee model and unique."""
    return isinstance(obj, type) and issubclass(obj, pw.Model) and hasattr(obj, '_meta')

def setVolume(self, volume):
        """Changes volume"""
        val = float(val)
        cmd = "volume %s" % val
        self._execute(cmd)

def get_file_size(filename):
    """
    Get the file size of a given file

    :param filename: string: pathname of a file
    :return: human readable filesize
    """
    if os.path.isfile(filename):
        return convert_size(os.path.getsize(filename))
    return None

def pool_args(function, sequence, kwargs):
    """Return a single iterator of n elements of lists of length 3, given a sequence of len n."""
    return zip(itertools.repeat(function), sequence, itertools.repeat(kwargs))

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

def serve_dtool_directory(directory, port):
    """Serve the datasets in a directory over HTTP."""
    os.chdir(directory)
    server_address = ("localhost", port)
    httpd = DtoolHTTPServer(server_address, DtoolHTTPRequestHandler)
    httpd.serve_forever()

def all_strings(arr):
        """
        Ensures that the argument is a list that either is empty or contains only strings
        :param arr: list
        :return:
        """
        if not isinstance([], list):
            raise TypeError("non-list value found where list is expected")
        return all(isinstance(x, str) for x in arr)

def calculate_month(birth_date):
    """
    Calculates and returns a month number basing on PESEL standard.
    """
    year = int(birth_date.strftime('%Y'))
    month = int(birth_date.strftime('%m')) + ((int(year / 100) - 14) % 5) * 20

    return month

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

def is_file(path):
    """Determine if a Path or string is a file on the file system."""
    try:
        return path.expanduser().absolute().is_file()
    except AttributeError:
        return os.path.isfile(os.path.abspath(os.path.expanduser(str(path))))

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

def debug_src(src, pm=False, globs=None):
    """Debug a single doctest docstring, in argument `src`'"""
    testsrc = script_from_examples(src)
    debug_script(testsrc, pm, globs)

def is_readable_dir(path):
  """Returns whether a path names an existing directory we can list and read files from."""
  return os.path.isdir(path) and os.access(path, os.R_OK) and os.access(path, os.X_OK)

def s3_connect(bucket_name, s3_access_key_id, s3_secret_key):
    """ Returns a Boto connection to the provided S3 bucket. """
    conn = connect_s3(s3_access_key_id, s3_secret_key)
    try:
        return conn.get_bucket(bucket_name)
    except S3ResponseError as e:
        if e.status == 403:
            raise Exception("Bad Amazon S3 credentials.")
        raise

def _validate_pos(df):
    """Validates the returned positional object
    """
    assert isinstance(df, pd.DataFrame)
    assert ["seqname", "position", "strand"] == df.columns.tolist()
    assert df.position.dtype == np.dtype("int64")
    assert df.strand.dtype == np.dtype("O")
    assert df.seqname.dtype == np.dtype("O")
    return df

def GeneratePassphrase(length=20):
  """Create a 20 char passphrase with easily typeable chars."""
  valid_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
  valid_chars += "0123456789 ,-_&$#"
  return "".join(random.choice(valid_chars) for i in range(length))

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

def full(self):
        """Return True if the queue is full"""
        if not self.size: return False
        return len(self.pq) == (self.size + self.removed_count)

def to_gtp(coord):
    """Converts from a Minigo coordinate to a GTP coordinate."""
    if coord is None:
        return 'pass'
    y, x = coord
    return '{}{}'.format(_GTP_COLUMNS[x], go.N - y)

def _check_fields(self, x, y):
		"""
		Check x and y fields parameters and initialize
		"""
		if x is None:
			if self.x is None:
				self.err(
					self._check_fields,
					"X field is not set: please specify a parameter")
				return
			x = self.x
		if y is None:
			if self.y is None:
				self.err(
					self._check_fields,
					"Y field is not set: please specify a parameter")
				return
			y = self.y
		return x, y

def colorize(string, color, *args, **kwargs):
    """
    Implements string formatting along with color specified in colorama.Fore
    """
    string = string.format(*args, **kwargs)
    return color + string + colorama.Fore.RESET

def get(s, delimiter='', format="diacritical"):
    """Return pinyin of string, the string must be unicode
    """
    return delimiter.join(_pinyin_generator(u(s), format=format))

def _trace_full (frame, event, arg):
    """Trace every executed line."""
    if event == "line":
        _trace_line(frame, event, arg)
    else:
        _trace(frame, event, arg)
    return _trace_full

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

def readTuple(self, line, n=3):
        """ Reads a tuple of numbers. e.g. vertices, normals or teture coords.
        """
        numbers = [num for num in line.split(' ') if num]
        return [float(num) for num in numbers[1:n + 1]]

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

def home():
    """Temporary helper function to link to the API routes"""
    return dict(links=dict(api='{}{}'.format(request.url, PREFIX[1:]))), \
        HTTPStatus.OK

def locate(command, on):
    """Locate the command's man page."""
    location = find_page_location(command, on)
    click.echo(location)

def validate(raw_schema, target=None, **kwargs):
    """
    Given the python representation of a JSONschema as defined in the swagger
    spec, validate that the schema complies to spec.  If `target` is provided,
    that target will be validated against the provided schema.
    """
    schema = schema_validator(raw_schema, **kwargs)
    if target is not None:
        validate_object(target, schema=schema, **kwargs)

def locate(command, on):
    """Locate the command's man page."""
    location = find_page_location(command, on)
    click.echo(location)

def translation(language):
    """
    Return a translation object in the default 'django' domain.
    """
    global _translations
    if language not in _translations:
        _translations[language] = Translations(language)
    return _translations[language]

def _clip(sid, prefix):
    """Clips a prefix from the beginning of a string if it exists."""
    return sid[len(prefix):] if sid.startswith(prefix) else sid

def _convert_date_to_dict(field_date):
        """
        Convert native python ``datetime.date`` object  to a format supported by the API
        """
        return {DAY: field_date.day, MONTH: field_date.month, YEAR: field_date.year}

def cleanup(self, app):
        """Close all connections."""
        if hasattr(self.database.obj, 'close_all'):
            self.database.close_all()

def get_common_elements(list1, list2):
    """find the common elements in two lists.  used to support auto align
        might be faster with sets

    Parameters
    ----------
    list1 : list
        a list of objects
    list2 : list
        a list of objects

    Returns
    -------
    list : list
        list of common objects shared by list1 and list2
        
    """
    #result = []
    #for item in list1:
    #    if item in list2:
    #        result.append(item)
    #Return list(set(list1).intersection(set(list2)))
    set2 = set(list2)
    result = [item for item in list1 if item in set2]
    return result

def socket_close(self):
        """Close our socket."""
        if self.sock != NC.INVALID_SOCKET:
            self.sock.close()
        self.sock = NC.INVALID_SOCKET

def mouse_out(self):
        """
        Performs a mouse out the element.

        Currently works only on Chrome driver.
        """
        self.scroll_to()
        ActionChains(self.parent.driver).move_by_offset(0, 0).click().perform()

def cancel(self, event=None):
        """Function called when Cancel-button clicked.

        This method returns focus to parent, and destroys the dialog.
        """

        if self.parent != None:
            self.parent.focus_set()

        self.destroy()

def on_close(self, ws):
        """ Called when websocket connection is closed
        """
        log.debug("Closing WebSocket connection with {}".format(self.url))
        if self.keepalive and self.keepalive.is_alive():
            self.keepalive.do_run = False
            self.keepalive.join()

def _add_hash(source):
    """Add a leading hash '#' at the beginning of every line in the source."""
    source = '\n'.join('# ' + line.rstrip()
                       for line in source.splitlines())
    return source

def _cal_dist2center(X, center):
    """ Calculate the SSE to the cluster center
    """
    dmemb2cen = scipy.spatial.distance.cdist(X, center.reshape(1,X.shape[1]), metric='seuclidean')
    return(np.sum(dmemb2cen))

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

def gaussian_variogram_model(m, d):
    """Gaussian model, m is [psill, range, nugget]"""
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return psill * (1. - np.exp(-d**2./(range_*4./7.)**2.)) + nugget

def md_to_text(content):
    """ Converts markdown content to text """
    text = None
    html = markdown.markdown(content)
    if html:
        text = html_to_text(content)
    return text

def query_collision(collision_object):
        """
        Check to see if the specified object is colliding with any of the objects currently in the Collision Manager
        Returns the first object we are colliding with if there was a collision and None if no collisions was found
        """
        global collidable_objects
        # Note that we use a Brute Force approach for the time being.
        # It performs horribly under heavy loads, but it meets
        # our needs for the time being.
        for obj in collidable_objects:
            # Make sure we don't check ourself against ourself.
            if obj.obj_id is not collision_object.obj_id:
                if collision_object.is_colliding(obj):
                    # A collision has been detected. Return the object that we are colliding with.
                    return obj

        # No collision was noticed. Return None.
        return None

def wrap(string, length, indent):
    """ Wrap a string at a line length """
    newline = "\n" + " " * indent
    return newline.join((string[i : i + length] for i in range(0, len(string), length)))

def _loadfilepath(self, filepath, **kwargs):
        """This loads a geojson file into a geojson python
        dictionary using the json module.
        
        Note: to load with a different text encoding use the encoding argument.
        """
        with open(filepath, "r") as f:
            data = json.load(f, **kwargs)
        return data

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

def has_edit_permission(self, request):
        """ Can edit this object """
        return request.user.is_authenticated and request.user.is_active and request.user.is_staff

def append_pdf(input_pdf: bytes, output_writer: PdfFileWriter):
    """
    Appends a PDF to a pyPDF writer. Legacy interface.
    """
    append_memory_pdf_to_writer(input_pdf=input_pdf,
                                writer=output_writer)

def println(msg):
    """
    Convenience function to print messages on a single line in the terminal
    """
    sys.stdout.write(msg)
    sys.stdout.flush()
    sys.stdout.write('\x08' * len(msg))
    sys.stdout.flush()

def disable_wx(self):
        """Disable event loop integration with wxPython.

        This merely sets PyOS_InputHook to NULL.
        """
        if self._apps.has_key(GUI_WX):
            self._apps[GUI_WX]._in_event_loop = False
        self.clear_inputhook()

def comma_converter(float_string):
    """Convert numbers to floats whether the decimal point is '.' or ','"""
    trans_table = maketrans(b',', b'.')
    return float(float_string.translate(trans_table))

def on_close(self, evt):
    """
    Pop-up menu and wx.EVT_CLOSE closing event
    """
    self.stop() # DoseWatcher
    if evt.EventObject is not self: # Avoid deadlocks
      self.Close() # wx.Frame
    evt.Skip()

def print_env_info(key, out=sys.stderr):
    """If given environment key is defined, print it out."""
    value = os.getenv(key)
    if value is not None:
        print(key, "=", repr(value), file=out)

def set_icon(self, bmp):
        """Sets main window icon to given wx.Bitmap"""

        _icon = wx.EmptyIcon()
        _icon.CopyFromBitmap(bmp)
        self.SetIcon(_icon)

def reprkwargs(kwargs, sep=', ', fmt="{0!s}={1!r}"):
    """Display kwargs."""
    return sep.join(fmt.format(k, v) for k, v in kwargs.iteritems())

def disable_wx(self):
        """Disable event loop integration with wxPython.

        This merely sets PyOS_InputHook to NULL.
        """
        if self._apps.has_key(GUI_WX):
            self._apps[GUI_WX]._in_event_loop = False
        self.clear_inputhook()

def _stop_instance(self):
        """Stop the instance."""
        instance = self._get_instance()
        instance.stop()
        self._wait_on_instance('stopped', self.timeout)

def _update_bordercolor(self, bordercolor):
        """Updates background color"""

        border_color = wx.SystemSettings_GetColour(wx.SYS_COLOUR_ACTIVEBORDER)
        border_color.SetRGB(bordercolor)

        self.linecolor_choice.SetColour(border_color)

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

def get_codeblock(language, text):
    """ Generates rst codeblock for given text and language """
    rst = "\n\n.. code-block:: " + language + "\n\n"
    for line in text.splitlines():
        rst += "\t" + line + "\n"

    rst += "\n"
    return rst

def root_parent(self, category=None):
        """ Returns the topmost parent of the current category. """
        return next(filter(lambda c: c.is_root, self.hierarchy()))

def format(x, format):
    """Uses http://www.cplusplus.com/reference/string/to_string/ for formatting"""
    # don't change the dtype, otherwise for each block the dtype may be different (string length)
    sl = vaex.strings.format(x, format)
    return column.ColumnStringArrow(sl.bytes, sl.indices, sl.length, sl.offset, string_sequence=sl)

def xor(a, b):
        """Bitwise xor on equal length bytearrays."""
        return bytearray(i ^ j for i, j in zip(a, b))

def isetdiff_flags(list1, list2):
    """
    move to util_iter
    """
    set2 = set(list2)
    return (item not in set2 for item in list1)

def _extract_node_text(node):
    """Extract text from a given lxml node."""

    texts = map(
        six.text_type.strip, map(six.text_type, map(unescape, node.xpath(".//text()")))
    )
    return " ".join(text for text in texts if text)

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

def getYamlDocument(filePath):
    """
    Return a yaml file's contents as a dictionary
    """
    with open(filePath) as stream:
        doc = yaml.load(stream)
        return doc

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

def _ParseYamlFromFile(filedesc):
  """Parses given YAML file."""
  content = filedesc.read()
  return yaml.Parse(content) or collections.OrderedDict()

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

def prox_zero(X, step):
    """Proximal operator to project onto zero
    """
    return np.zeros(X.shape, dtype=X.dtype)

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

def register():
    """
    Calls the shots, based on signals
    """
    signals.article_generator_finalized.connect(link_source_files)
    signals.page_generator_finalized.connect(link_source_files)
    signals.page_writer_finalized.connect(write_source_files)

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

def quit(self):
        """
        Closes the browser and shuts down the WebKitGTKDriver executable
        that is started when starting the WebKitGTKDriver
        """
        try:
            RemoteWebDriver.quit(self)
        except http_client.BadStatusLine:
            pass
        finally:
            self.service.stop()

def valid_uuid(value):
    """ Check if value is a valid UUID. """

    try:
        uuid.UUID(value, version=4)
        return True
    except (TypeError, ValueError, AttributeError):
        return False

def email_type(arg):
	"""An argparse type representing an email address."""
	if not is_valid_email_address(arg):
		raise argparse.ArgumentTypeError("{0} is not a valid email address".format(repr(arg)))
	return arg

def is_type(value):
        """Determine if value is an instance or subclass of the class Type."""
        if isinstance(value, type):
            return issubclass(value, Type)
        return isinstance(value, Type)

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

def to_str(s):
    """
    Convert bytes and non-string into Python 3 str
    """
    if isinstance(s, bytes):
        s = s.decode('utf-8')
    elif not isinstance(s, str):
        s = str(s)
    return s

def shannon_entropy(p):
    """Calculates shannon entropy in bits.

    Parameters
    ----------
    p : np.array
        array of probabilities

    Returns
    -------
    shannon entropy in bits
    """
    return -np.sum(np.where(p!=0, p * np.log2(p), 0))

def sort_fn_list(fn_list):
    """Sort input filename list by datetime
    """
    dt_list = get_dt_list(fn_list)
    fn_list_sort = [fn for (dt,fn) in sorted(zip(dt_list,fn_list))]
    return fn_list_sort

def _concatenate_virtual_arrays(arrs, cols=None, scaling=None):
    """Return a virtual concatenate of several NumPy arrays."""
    return None if not len(arrs) else ConcatenatedArrays(arrs, cols,
                                                         scaling=scaling)

def list_of_lists_to_dict(l):
    """ Convert list of key,value lists to dict

    [['id', 1], ['id', 2], ['id', 3], ['foo': 4]]
    {'id': [1, 2, 3], 'foo': [4]}
    """
    d = {}
    for key, val in l:
        d.setdefault(key, []).append(val)
    return d

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

def yum_install(self, packages, ignore_error=False):
        """Install some packages on the remote host.

        :param packages: ist of packages to install.
        """
        return self.run('yum install -y --quiet ' + ' '.join(packages), ignore_error=ignore_error, retry=5)

def config_parser_to_dict(config_parser):
    """
    Convert a ConfigParser to a dictionary.
    """
    response = {}

    for section in config_parser.sections():
        for option in config_parser.options(section):
            response.setdefault(section, {})[option] = config_parser.get(section, option)

    return response

def HttpResponse403(request, template=KEY_AUTH_403_TEMPLATE,
content=KEY_AUTH_403_CONTENT, content_type=KEY_AUTH_403_CONTENT_TYPE):
    """
    HTTP response for forbidden access (status code 403)
    """
    return AccessFailedResponse(request, template, content, content_type, status=403)

def items(self, section_name):
        """:return: list((option, value), ...) pairs of all items in the given section"""
        return [(k, v) for k, v in super(GitConfigParser, self).items(section_name) if k != '__name__']

def version_jar(self):
		"""
		Special case of version() when the executable is a JAR file.
		"""
		cmd = config.get_command('java')
		cmd.append('-jar')
		cmd += self.cmd
		self.version(cmd=cmd, path=self.cmd[0])

def _is_already_configured(configuration_details):
    """Returns `True` when alias already in shell config."""
    path = Path(configuration_details.path).expanduser()
    with path.open('r') as shell_config:
        return configuration_details.content in shell_config.read()

def get_month_start_end_day():
    """
    Get the month start date a nd end date
    """
    t = date.today()
    n = mdays[t.month]
    return (date(t.year, t.month, 1), date(t.year, t.month, n))

def pass_from_pipe(cls):
        """Return password from pipe if not on TTY, else False.
        """
        is_pipe = not sys.stdin.isatty()
        return is_pipe and cls.strip_last_newline(sys.stdin.read())

def count(lines):
  """ Counts the word frequences in a list of sentences.

  Note:
    This is a helper function for parallel execution of `Vocabulary.from_text`
    method.
  """
  words = [w for l in lines for w in l.strip().split()]
  return Counter(words)

def as_list(callable):
    """Convert a scalar validator in a list validator"""
    @wraps(callable)
    def wrapper(value_iter):
        return [callable(value) for value in value_iter]

    return wrapper

def top(n, width=WIDTH, style=STYLE):
    """Prints the top row of a table"""
    return hrule(n, width, linestyle=STYLES[style].top)

def __unixify(self, s):
        """ stupid windows. converts the backslash to forwardslash for consistency """
        return os.path.normpath(s).replace(os.sep, "/")

def get_from_human_key(self, key):
        """Return the key (aka database value) of a human key (aka Python identifier)."""
        if key in self._identifier_map:
            return self._identifier_map[key]
        raise KeyError(key)

def obj_to_string(obj, top=True):
    """
    Turn an arbitrary object into a unicode string. If complex (dict/list/tuple), will be json-encoded.
    """
    obj = prepare_for_json_encoding(obj)
    if type(obj) == six.text_type:
        return obj
    return json.dumps(obj)

def get_oauth_token():
    """Retrieve a simple OAuth Token for use with the local http client."""
    url = "{0}/token".format(DEFAULT_ORIGIN["Origin"])
    r = s.get(url=url)
    return r.json()["t"]

def populate_obj(obj, attrs):
    """Populates an object's attributes using the provided dict
    """
    for k, v in attrs.iteritems():
        setattr(obj, k, v)

def typename(obj):
    """Returns the type of obj as a string. More descriptive and specific than
    type(obj), and safe for any object, unlike __class__."""
    if hasattr(obj, '__class__'):
        return getattr(obj, '__class__').__name__
    else:
        return type(obj).__name__

def unit_ball_L2(shape):
  """A tensorflow variable tranfomed to be constrained in a L2 unit ball.

  EXPERIMENTAL: Do not use for adverserial examples if you need to be confident
  they are strong attacks. We are not yet confident in this code.
  """
  x = tf.Variable(tf.zeros(shape))
  return constrain_L2(x)

def _get_session():
    """Return (and memoize) a database session"""
    session = getattr(g, '_session', None)
    if session is None:
        session = g._session = db.session()
    return session

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

def _MakeExecutable(self, metadata_script):
    """Add executable permissions to a file.

    Args:
      metadata_script: string, the path to the executable file.
    """
    mode = os.stat(metadata_script).st_mode
    os.chmod(metadata_script, mode | stat.S_IEXEC)

def circ_permutation(items):
    """Calculate the circular permutation for a given list of items."""
    permutations = []
    for i in range(len(items)):
        permutations.append(items[i:] + items[:i])
    return permutations

def ms_to_datetime(ms):
    """
    Converts a millisecond accuracy timestamp to a datetime
    """
    dt = datetime.datetime.utcfromtimestamp(ms / 1000)
    return dt.replace(microsecond=(ms % 1000) * 1000).replace(tzinfo=pytz.utc)

def format_result(input):
        """From: http://stackoverflow.com/questions/13062300/convert-a-dict-to-sorted-dict-in-python
        """
        items = list(iteritems(input))
        return OrderedDict(sorted(items, key=lambda x: x[0]))

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

def format_result(input):
        """From: http://stackoverflow.com/questions/13062300/convert-a-dict-to-sorted-dict-in-python
        """
        items = list(iteritems(input))
        return OrderedDict(sorted(items, key=lambda x: x[0]))

def build_code(self, lang, body):
        """Wrap text with markdown specific flavour."""
        self.out.append("```" + lang)
        self.build_markdown(lang, body)
        self.out.append("```")

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

def add_noise(Y, sigma):
    """Adds noise to Y"""
    return Y + np.random.normal(0, sigma, Y.shape)

def check_create_folder(filename):
    """Check if the folder exisits. If not, create the folder"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

def _removeTags(tags, objects):
    """ Removes tags from objects """
    for t in tags:
        for o in objects:
            o.tags.remove(t)

    return True

def from_json_str(cls, json_str):
    """Convert json string representation into class instance.

    Args:
      json_str: json representation as string.

    Returns:
      New instance of the class with data loaded from json string.
    """
    return cls.from_json(json.loads(json_str, cls=JsonDecoder))

def load_model_from_package(name, **overrides):
    """Load a model from an installed package."""
    cls = importlib.import_module(name)
    return cls.load(**overrides)

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

def types(self):
        """
        Return a list of all the variable types that exist in the
        Variables object.
        """
        output = set()
        for var in self.values():
            if var.has_value():
                output.update(var.types())
        return list(output)

def index():
    """ Display productpage with normal user and test user buttons"""
    global productpage

    table = json2html.convert(json = json.dumps(productpage),
                              table_attributes="class=\"table table-condensed table-bordered table-hover\"")

    return render_template('index.html', serviceTable=table)

def send_request(self, *args, **kwargs):
        """Wrapper for session.request
        Handle connection reset error even from pyopenssl
        """
        try:
            return self.session.request(*args, **kwargs)
        except ConnectionError:
            self.session.close()
            return self.session.request(*args, **kwargs)

def crop_box(im, box=False, **kwargs):
    """Uses box coordinates to crop an image without resizing it first."""
    if box:
        im = im.crop(box)
    return im

def transformer_ae_a3():
  """Set of hyperparameters."""
  hparams = transformer_ae_base()
  hparams.batch_size = 4096
  hparams.layer_prepostprocess_dropout = 0.3
  hparams.optimizer = "Adafactor"
  hparams.learning_rate = 0.25
  hparams.learning_rate_warmup_steps = 10000
  return hparams

def csv_to_dicts(file, header=None):
    """Reads a csv and returns a List of Dicts with keys given by header row."""
    with open(file) as csvfile:
        return [row for row in csv.DictReader(csvfile, fieldnames=header)]

def vec_angle(a, b):
    """
    Calculate angle between two vectors
    """
    cosang = np.dot(a, b)
    sinang = fast_norm(np.cross(a, b))
    return np.arctan2(sinang, cosang)

def _ram_buffer(self):
        """Setup the RAM buffer from the C++ code."""
        # get the address of the RAM
        address = _LIB.Memory(self._env)
        # create a buffer from the contents of the address location
        buffer_ = ctypes.cast(address, ctypes.POINTER(RAM_VECTOR)).contents
        # create a NumPy array from the buffer
        return np.frombuffer(buffer_, dtype='uint8')

def create_rot2d(angle):
    """Create 2D rotation matrix"""
    ca = math.cos(angle)
    sa = math.sin(angle)
    return np.array([[ca, -sa], [sa, ca]])

def _monitor_callback_wrapper(callback):
    """A wrapper for the user-defined handle."""
    def callback_handle(name, array, _):
        """ ctypes function """
        callback(name, array)
    return callback_handle

def experiment_property(prop):
    """Get a property of the experiment by name."""
    exp = experiment(session)
    p = getattr(exp, prop)
    return success_response(field=prop, data=p, request_type=prop)

def add_to_js(self, name, var):
        """Add an object to Javascript."""
        frame = self.page().mainFrame()
        frame.addToJavaScriptWindowObject(name, var)

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

def mouse_out(self):
        """
        Performs a mouse out the element.

        Currently works only on Chrome driver.
        """
        self.scroll_to()
        ActionChains(self.parent.driver).move_by_offset(0, 0).click().perform()

def auto():
	"""set colouring on if STDOUT is a terminal device, off otherwise"""
	try:
		Style.enabled = False
		Style.enabled = sys.stdout.isatty()
	except (AttributeError, TypeError):
		pass

def compose(*funcs):
    """compose a list of functions"""
    return lambda x: reduce(lambda v, f: f(v), reversed(funcs), x)

def downcaseTokens(s,l,t):
    """Helper parse action to convert tokens to lower case."""
    return [ tt.lower() for tt in map(_ustr,t) ]

def transform(self, df):
        """
        Transforms a DataFrame in place. Computes all outputs of the DataFrame.

        Args:
            df (pandas.DataFrame): DataFrame to transform.
        """
        for name, function in self.outputs:
            df[name] = function(df)

def screen_cv2(self):
        """cv2 Image of current window screen"""
        pil_image = self.screen.convert('RGB')
        cv2_image = np.array(pil_image)
        pil_image.close()
        # Convert RGB to BGR 
        cv2_image = cv2_image[:, :, ::-1]
        return cv2_image

def translate_fourier(image, dx):
    """ Translate an image in fourier-space with plane waves """
    N = image.shape[0]

    f = 2*np.pi*np.fft.fftfreq(N)
    kx,ky,kz = np.meshgrid(*(f,)*3, indexing='ij')
    kv = np.array([kx,ky,kz]).T

    q = np.fft.fftn(image)*np.exp(-1.j*(kv*dx).sum(axis=-1)).T
    return np.real(np.fft.ifftn(q))

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

def enbw(wnd):
  """ Equivalent Noise Bandwidth in bins (Processing Gain reciprocal). """
  return sum(el ** 2 for el in wnd) / sum(wnd) ** 2 * len(wnd)

def _join(verb):
    """
    Join helper
    """
    data = pd.merge(verb.x, verb.y, **verb.kwargs)

    # Preserve x groups
    if isinstance(verb.x, GroupedDataFrame):
        data.plydata_groups = list(verb.x.plydata_groups)
    return data

def zeros(self, name, **kwargs):
        """Create an array. Keyword arguments as per
        :func:`zarr.creation.zeros`."""
        return self._write_op(self._zeros_nosync, name, **kwargs)

def sort_data(data, cols):
    """Sort `data` rows and order columns"""
    return data.sort_values(cols)[cols + ['value']].reset_index(drop=True)

def zeros(self, name, **kwargs):
        """Create an array. Keyword arguments as per
        :func:`zarr.creation.zeros`."""
        return self._write_op(self._zeros_nosync, name, **kwargs)

def cleanup(self, app):
        """Close all connections."""
        if hasattr(self.database.obj, 'close_all'):
            self.database.close_all()

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

def make_aware(dt):
    """Appends tzinfo and assumes UTC, if datetime object has no tzinfo already."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

def __init__(self, collection, index_type_obj):
        """
            Constructs wrapper for general index creation and deletion

            :param collection Collection
            :param index_type_obj BaseIndex Object of a index sub-class
        """

        self.collection = collection
        self.index_type_obj = index_type_obj

def make_aware(dt):
    """Appends tzinfo and assumes UTC, if datetime object has no tzinfo already."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

def setup(app):
    """Allow this package to be used as Sphinx extension.
    This is also called from the top-level ``__init__.py``.

    :type app: sphinx.application.Sphinx
    """
    from .patches import patch_django_for_autodoc

    # When running, make sure Django doesn't execute querysets
    patch_django_for_autodoc()

    # Generate docstrings for Django model fields
    # Register the docstring processor with sphinx
    app.connect('autodoc-process-docstring', improve_model_docstring)

    # influence skip rules
    app.connect("autodoc-skip-member", autodoc_skip)

def _possibly_convert_objects(values):
    """Convert arrays of datetime.datetime and datetime.timedelta objects into
    datetime64 and timedelta64, according to the pandas convention.
    """
    return np.asarray(pd.Series(values.ravel())).reshape(values.shape)

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

def convertDatetime(t):
    """
    Converts the specified datetime object into its appropriate protocol
    value. This is the number of milliseconds from the epoch.
    """
    epoch = datetime.datetime.utcfromtimestamp(0)
    delta = t - epoch
    millis = delta.total_seconds() * 1000
    return int(millis)

def variance(arr):
  """variance of the values, must have 2 or more entries.

  :param arr: list of numbers
  :type arr: number[] a number array
  :return: variance
  :rtype: float

  """
  avg = average(arr)
  return sum([(float(x)-avg)**2 for x in arr])/float(len(arr)-1)

def timestamp_to_datetime(timestamp):
    """Convert an ARF timestamp to a datetime.datetime object (naive local time)"""
    from datetime import datetime, timedelta
    obj = datetime.fromtimestamp(timestamp[0])
    return obj + timedelta(microseconds=int(timestamp[1]))

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

def datetime_to_ms(dt):
    """
    Converts a datetime to a millisecond accuracy timestamp
    """
    seconds = calendar.timegm(dt.utctimetuple())
    return seconds * 1000 + int(dt.microsecond / 1000)

def toBase64(s):
    """Represent string / bytes s as base64, omitting newlines"""
    if isinstance(s, str):
        s = s.encode("utf-8")
    return binascii.b2a_base64(s)[:-1]

def now(self):
		"""
		Return a :py:class:`datetime.datetime` instance representing the current time.

		:rtype: :py:class:`datetime.datetime`
		"""
		if self.use_utc:
			return datetime.datetime.utcnow()
		else:
			return datetime.datetime.now()

def coords_string_parser(self, coords):
        """Pareses the address string into coordinates to match address_to_coords return object"""

        lat, lon = coords.split(',')
        return {"lat": lat.strip(), "lon": lon.strip(), "bounds": {}}

def ToDatetime(self):
    """Converts Timestamp to datetime."""
    return datetime.utcfromtimestamp(
        self.seconds + self.nanos / float(_NANOS_PER_SECOND))

def _linearInterpolationTransformMatrix(matrix1, matrix2, value):
    """ Linear, 'oldstyle' interpolation of the transform matrix."""
    return tuple(_interpolateValue(matrix1[i], matrix2[i], value) for i in range(len(matrix1)))

def _DateToEpoch(date):
  """Converts python datetime to epoch microseconds."""
  tz_zero = datetime.datetime.utcfromtimestamp(0)
  diff_sec = int((date - tz_zero).total_seconds())
  return diff_sec * 1000000

def each_img(dir_path):
    """
    Iterates through each image in the given directory. (not recursive)
    :param dir_path: Directory path where images files are present
    :return: Iterator to iterate through image files
    """
    for fname in os.listdir(dir_path):
        if fname.endswith('.jpg') or fname.endswith('.png') or fname.endswith('.bmp'):
            yield fname

def datetime_to_timezone(date, tz="UTC"):
    """ convert naive datetime to timezone-aware datetime """
    if not date.tzinfo:
        date = date.replace(tzinfo=timezone(get_timezone()))
    return date.astimezone(timezone(tz))

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

def now_time(str=False):
    """Get the current time."""
    if str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return datetime.datetime.now()

def resize_by_area(img, size):
  """image resize function used by quite a few image problems."""
  return tf.to_int64(
      tf.image.resize_images(img, [size, size], tf.image.ResizeMethod.AREA))

def _iterable_to_varargs_method(func):
    """decorator to convert a method taking a iterable to a *args one"""
    def wrapped(self, *args, **kwargs):
        return func(self, args, **kwargs)
    return wrapped

def decode_bytes(string):
    """ Decodes a given base64 string into bytes.

    :param str string: The string to decode
    :return: The decoded bytes
    :rtype: bytes
    """

    if is_string_type(type(string)):
        string = bytes(string, "utf-8")
    return base64.decodebytes(string)

def recursively_update(d, d2):
  """dict.update but which merges child dicts (dict2 takes precedence where there's conflict)."""
  for k, v in d2.items():
    if k in d:
      if isinstance(v, dict):
        recursively_update(d[k], v)
        continue
    d[k] = v

def val_to_bin(edges, x):
    """Convert axis coordinate to bin index."""
    ibin = np.digitize(np.array(x, ndmin=1), edges) - 1
    return ibin

def __deepcopy__(self, memo):
        """Improve deepcopy speed."""
        return type(self)(value=self._value, enum_ref=self.enum_ref)

def print_bintree(tree, indent='  '):
    """print a binary tree"""
    for n in sorted(tree.keys()):
        print "%s%s" % (indent * depth(n,tree), n)

def def_linear(fun):
    """Flags that a function is linear wrt all args"""
    defjvp_argnum(fun, lambda argnum, g, ans, args, kwargs:
                  fun(*subval(args, argnum, g), **kwargs))

def abs_img(img):
    """ Return an image with the binarised version of the data of `img`."""
    bool_img = np.abs(read_img(img).get_data())
    return bool_img.astype(int)

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

def is_bool_matrix(l):
    r"""Checks if l is a 2D numpy array of bools

    """
    if isinstance(l, np.ndarray):
        if l.ndim == 2 and (l.dtype == bool):
            return True
    return False

def input_int_default(question="", default=0):
    """A function that works for both, Python 2.x and Python 3.x.
       It asks the user for input and returns it as a string.
    """
    answer = input_string(question)
    if answer == "" or answer == "yes":
        return default
    else:
        return int(answer)

def make_file_read_only(file_path):
    """
    Removes the write permissions for the given file for owner, groups and others.

    :param file_path: The file whose privileges are revoked.
    :raise FileNotFoundError: If the given file does not exist.
    """
    old_permissions = os.stat(file_path).st_mode
    os.chmod(file_path, old_permissions & ~WRITE_PERMISSIONS)

def show_xticklabels(self, row, column):
        """Show the x-axis tick labels for a subplot.

        :param row,column: specify the subplot.

        """
        subplot = self.get_subplot_at(row, column)
        subplot.show_xticklabels()

def download_file_from_bucket(self, bucket, file_path, key):
        """ Download file from S3 Bucket """
        with open(file_path, 'wb') as data:
            self.__s3.download_fileobj(bucket, key, data)
            return file_path

def safe_delete(filename):
  """Delete a file safely. If it's not present, no-op."""
  try:
    os.unlink(filename)
  except OSError as e:
    if e.errno != errno.ENOENT:
      raise

def split_into_words(s):
  """Split a sentence into list of words."""
  s = re.sub(r"\W+", " ", s)
  s = re.sub(r"[_0-9]+", " ", s)
  return s.split()

def strip_accents(string):
    """
    Strip all the accents from the string
    """
    return u''.join(
        (character for character in unicodedata.normalize('NFD', string)
         if unicodedata.category(character) != 'Mn'))

def url(self, action, **kwargs):
        """ Construct and return the URL for a specific API service. """
        # TODO : should be static method ?
        return self.URLS['BASE'] % self.URLS[action] % kwargs

def backward_delete_word(self, e): # (Control-Rubout)
        u"""Delete the character behind the cursor. A numeric argument means
        to kill the characters instead of deleting them."""
        self.l_buffer.backward_delete_word(self.argument_reset)
        self.finalize()

def clean_all(self, args):
        """Delete all build components; the package cache, package builds,
        bootstrap builds and distributions."""
        self.clean_dists(args)
        self.clean_builds(args)
        self.clean_download_cache(args)

def __delitem__(self, key):
		"""Remove item with given key from the mapping.

		Runs in O(n), unless removing last item, then in O(1).
		"""
		index, value = self._dict.pop(key)
		key2, value2 = self._list.pop(index)
		assert key == key2
		assert value is value2

		self._fix_indices_after_delete(index)

def restore_button_state(self):
        """Helper to restore button state."""
        self.parent.pbnNext.setEnabled(self.next_button_state)
        self.parent.pbnBack.setEnabled(self.back_button_state)

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

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

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

def _sanitize(text):
    """Return sanitized Eidos text field for human readability."""
    d = {'-LRB-': '(', '-RRB-': ')'}
    return re.sub('|'.join(d.keys()), lambda m: d[m.group(0)], text)

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

def _remove_from_index(index, obj):
    """Removes object ``obj`` from the ``index``."""
    try:
        index.value_map[indexed_value(index, obj)].remove(obj.id)
    except KeyError:
        pass

def _distance(coord1, coord2):
    """
    Return the distance between two points, `coord1` and `coord2`. These
    parameters are assumed to be (x, y) tuples.
    """
    xdist = coord1[0] - coord2[0]
    ydist = coord1[1] - coord2[1]
    return sqrt(xdist*xdist + ydist*ydist)

def _decode(self, obj, context):
        """
        Get the python representation of the obj
        """
        return b''.join(map(int2byte, [c + 0x60 for c in bytearray(obj)])).decode("utf8")

def tpr(y, z):
    """True positive rate `tp / (tp + fn)`
    """
    tp, tn, fp, fn = contingency_table(y, z)
    return tp / (tp + fn)

def is_descriptor_class(desc, include_abstract=False):
    r"""Check calculatable descriptor class or not.

    Returns:
        bool

    """
    return (
        isinstance(desc, type)
        and issubclass(desc, Descriptor)
        and (True if include_abstract else not inspect.isabstract(desc))
    )

def inh(table):
    """
    inverse hyperbolic sine transformation
    """
    t = []
    for i in table:
        t.append(np.ndarray.tolist(np.arcsinh(i)))
    return t

def terminate(self):
        """Terminate all workers and threads."""
        for t in self._threads:
            t.quit()
        self._thread = []
        self._workers = []

def _manhattan_distance(vec_a, vec_b):
    """Return manhattan distance between two lists of numbers."""
    if len(vec_a) != len(vec_b):
        raise ValueError('len(vec_a) must equal len(vec_b)')
    return sum(map(lambda a, b: abs(a - b), vec_a, vec_b))

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

def EvalGaussianPdf(x, mu, sigma):
    """Computes the unnormalized PDF of the normal distribution.

    x: value
    mu: mean
    sigma: standard deviation
    
    returns: float probability density
    """
    return scipy.stats.norm.pdf(x, mu, sigma)

def set_terminate_listeners(stream):
    """Die on SIGTERM or SIGINT"""

    def stop(signum, frame):
        terminate(stream.listener)

    # Installs signal handlers for handling SIGINT and SIGTERM
    # gracefully.
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

def matches(self, s):
    """Whether the pattern matches anywhere in the string s."""
    regex_matches = self.compiled_regex.search(s) is not None
    return not regex_matches if self.inverted else regex_matches

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

def is_unix_like(platform=None):
    """Returns whether the given platform is a Unix-like platform with the usual
    Unix filesystem. When the parameter is omitted, it defaults to ``sys.platform``
    """
    platform = platform or sys.platform
    platform = platform.lower()
    return platform.startswith("linux") or platform.startswith("darwin") or \
            platform.startswith("cygwin")

def calculate_size(name, data_list):
    """ Calculates the request payload size"""
    data_size = 0
    data_size += calculate_size_str(name)
    data_size += INT_SIZE_IN_BYTES
    for data_list_item in data_list:
        data_size += calculate_size_data(data_list_item)
    return data_size

def is_int_vector(l):
    r"""Checks if l is a numpy array of integers

    """
    if isinstance(l, np.ndarray):
        if l.ndim == 1 and (l.dtype.kind == 'i' or l.dtype.kind == 'u'):
            return True
    return False

def trigger(self, target: str, trigger: str, parameters: Dict[str, Any]={}):
		"""Calls the specified Trigger of another Area with the optionally given parameters.

		Args:
			target: The name of the target Area.
			trigger: The name of the Trigger.
			parameters: The parameters of the function call.
		"""
		pass

def get_file_extension_type(filename):
    """
    Return the group associated to the file
    :param filename:
    :return: str
    """
    ext = get_file_extension(filename)
    if ext:
        for name, group in EXTENSIONS.items():
            if ext in group:
                return name
    return "OTHER"

def register(self, target):
        """Registers url_rules on the blueprint
        """
        for rule, options in self.url_rules:
            target.add_url_rule(rule, self.name, self.dispatch_request, **options)

def check_hash_key(query_on, key):
    """Only allows == against query_on.hash_key"""
    return (
        isinstance(key, BaseCondition) and
        (key.operation == "==") and
        (key.column is query_on.hash_key)
    )

def load_files(files):
    """Load and execute a python file."""

    for py_file in files:
        LOG.debug("exec %s", py_file)
        execfile(py_file, globals(), locals())

def _uptime_syllable():
    """Returns uptime in seconds or None, on Syllable."""
    global __boottime
    try:
        __boottime = os.stat('/dev/pty/mst/pty0').st_mtime
        return time.time() - __boottime
    except (NameError, OSError):
        return None

def typescript_compile(source):
    """Compiles the given ``source`` from TypeScript to ES5 using TypescriptServices.js"""
    with open(TS_COMPILER, 'r') as tsservices_js:
        return evaljs(
            (tsservices_js.read(),
             'ts.transpile(dukpy.tscode, {options});'.format(options=TSC_OPTIONS)),
            tscode=source
        )

def __contains__(self, key):
        """
        Invoked when determining whether a specific key is in the dictionary
        using `key in d`.

        The key is looked up case-insensitively.
        """
        k = self._real_key(key)
        return k in self._data

def sort_func(self, key):
        """Sorting logic for `Quantity` objects."""
        if key == self._KEYS.VALUE:
            return 'aaa'
        if key == self._KEYS.SOURCE:
            return 'zzz'
        return key

def __call__(self, func, *args, **kwargs):
        """Shorcut for self.run."""
        return self.run(func, *args, **kwargs)

def check(self, var):
        """Check whether the provided value is a valid enum constant."""
        if not isinstance(var, _str_type): return False
        return _enum_mangle(var) in self._consts

def nonull_dict(self):
        """Like dict, but does not hold any null values.

        :return:

        """
        return {k: v for k, v in six.iteritems(self.dict) if v and k != '_codes'}

def __init__(self):
        """Initialize the state of the object"""
        self.state = self.STATE_INITIALIZING
        self.state_start = time.time()

def clean_dict_keys(d):
    """Convert all keys of the dict 'd' to (ascii-)strings.

    :Raises: UnicodeEncodeError
    """
    new_d = {}
    for (k, v) in d.iteritems():
        new_d[str(k)] = v
    return new_d

def redirect(view=None, url=None, **kwargs):
    """Redirects to the specified view or url
    """
    if view:
        if url:
            kwargs["url"] = url
        url = flask.url_for(view, **kwargs)
    current_context.exit(flask.redirect(url))

def extract_zip(zip_path, target_folder):
    """
    Extract the content of the zip-file at `zip_path` into `target_folder`.
    """
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target_folder)

def keys_to_snake_case(camel_case_dict):
    """
    Make a copy of a dictionary with all keys converted to snake case. This is just calls to_snake_case on
    each of the keys in the dictionary and returns a new dictionary.

    :param camel_case_dict: Dictionary with the keys to convert.
    :type camel_case_dict: Dictionary.

    :return: Dictionary with the keys converted to snake case.
    """
    return dict((to_snake_case(key), value) for (key, value) in camel_case_dict.items())

def is_break_tag(self, el):
        """Check if tag is an element we should break on."""

        name = el.name
        return name in self.break_tags or name in self.user_break_tags

def pretty_dict_str(d, indent=2):
    """shows JSON indented representation of d"""
    b = StringIO()
    write_pretty_dict_str(b, d, indent=indent)
    return b.getvalue()

def ignored_regions(source):
    """Return ignored regions like strings and comments in `source` """
    return [(match.start(), match.end()) for match in _str.finditer(source)]

def _remove_dict_keys_with_value(dict_, val):
  """Removes `dict` keys which have have `self` as value."""
  return {k: v for k, v in dict_.items() if v is not val}

def install_plugin(username, repo):
    """Installs a Blended plugin from GitHub"""
    print("Installing plugin from " + username + "/" + repo)

    pip.main(['install', '-U', "git+git://github.com/" +
              username + "/" + repo + ".git"])

def setdefaults(dct, defaults):
    """Given a target dct and a dict of {key:default value} pairs,
    calls setdefault for all of those pairs."""
    for key in defaults:
        dct.setdefault(key, defaults[key])

    return dct

def to_camel_case(text):
    """Convert to camel case.

    :param str text:
    :rtype: str
    :return:
    """
    split = text.split('_')
    return split[0] + "".join(x.title() for x in split[1:])

def pretty_dict_str(d, indent=2):
    """shows JSON indented representation of d"""
    b = StringIO()
    write_pretty_dict_str(b, d, indent=indent)
    return b.getvalue()

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

def str2bytes(x):
  """Convert input argument to bytes"""
  if type(x) is bytes:
    return x
  elif type(x) is str:
    return bytes([ ord(i) for i in x ])
  else:
    return str2bytes(str(x))

def dict_to_querystring(dictionary):
    """Converts a dict to a querystring suitable to be appended to a URL."""
    s = u""
    for d in dictionary.keys():
        s = unicode.format(u"{0}{1}={2}&", s, d, dictionary[d])
    return s[:-1]

def run_cmd(command, verbose=True, shell='/bin/bash'):
    """internal helper function to run shell commands and get output"""
    process = Popen(command, shell=True, stdout=PIPE, stderr=STDOUT, executable=shell)
    output = process.stdout.read().decode().strip().split('\n')
    if verbose:
        # return full output including empty lines
        return output
    return [line for line in output if line.strip()]

def update(self, other_dict):
        """update() extends rather than replaces existing key lists."""
        for key, value in iter_multi_items(other_dict):
            MultiDict.add(self, key, value)

def set_terminate_listeners(stream):
    """Die on SIGTERM or SIGINT"""

    def stop(signum, frame):
        terminate(stream.listener)

    # Installs signal handlers for handling SIGINT and SIGTERM
    # gracefully.
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

def keys_to_snake_case(camel_case_dict):
    """
    Make a copy of a dictionary with all keys converted to snake case. This is just calls to_snake_case on
    each of the keys in the dictionary and returns a new dictionary.

    :param camel_case_dict: Dictionary with the keys to convert.
    :type camel_case_dict: Dictionary.

    :return: Dictionary with the keys converted to snake case.
    """
    return dict((to_snake_case(key), value) for (key, value) in camel_case_dict.items())

def _cal_dist2center(X, center):
    """ Calculate the SSE to the cluster center
    """
    dmemb2cen = scipy.spatial.distance.cdist(X, center.reshape(1,X.shape[1]), metric='seuclidean')
    return(np.sum(dmemb2cen))

def update(self, params):
        """Update the dev_info data from a dictionary.

        Only updates if it already exists in the device.
        """
        dev_info = self.json_state.get('deviceInfo')
        dev_info.update({k: params[k] for k in params if dev_info.get(k)})

def gauss_box_model(x, amplitude=1.0, mean=0.0, stddev=1.0, hpix=0.5):
    """Integrate a Gaussian profile."""
    z = (x - mean) / stddev
    z2 = z + hpix / stddev
    z1 = z - hpix / stddev
    return amplitude * (norm.cdf(z2) - norm.cdf(z1))

def pretty_dict_str(d, indent=2):
    """shows JSON indented representation of d"""
    b = StringIO()
    write_pretty_dict_str(b, d, indent=indent)
    return b.getvalue()

def min_max_normalize(img):
    """Centre and normalize a given array.

    Parameters:
    ----------
    img: np.ndarray

    """

    min_img = img.min()
    max_img = img.max()

    return (img - min_img) / (max_img - min_img)

def filter_dict_by_key(d, keys):
    """Filter the dict *d* to remove keys not in *keys*."""
    return {k: v for k, v in d.items() if k in keys}

def swap(self):
        """Return the box (for horizontal graphs)"""
        self.xmin, self.ymin = self.ymin, self.xmin
        self.xmax, self.ymax = self.ymax, self.xmax

def dump_nparray(self, obj, class_name=numpy_ndarray_class_name):
        """
        ``numpy.ndarray`` dumper.
        """
        return {"$" + class_name: self._json_convert(obj.tolist())}

def get_number(s, cast=int):
    """
    Try to get a number out of a string, and cast it.
    """
    import string
    d = "".join(x for x in str(s) if x in string.digits)
    return cast(d)

def pretty_dict_str(d, indent=2):
    """shows JSON indented representation of d"""
    b = StringIO()
    write_pretty_dict_str(b, d, indent=indent)
    return b.getvalue()

def plot3d_init(fignum):
    """
    initializes 3D plot
    """
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(fignum)
    ax = fig.add_subplot(111, projection='3d')
    return ax

def safe_dump_all(documents, stream=None, **kwds):
    """
    Serialize a sequence of Python objects into a YAML stream.
    Produce only basic YAML tags.
    If stream is None, return the produced string instead.
    """
    return dump_all(documents, stream, Dumper=SafeDumper, **kwds)

def _update_index_on_df(df, index_names):
    """Helper function to restore index information after collection. Doesn't
    use self so we can serialize this."""
    if index_names:
        df = df.set_index(index_names)
        # Remove names from unnamed indexes
        index_names = _denormalize_index_names(index_names)
        df.index.names = index_names
    return df

async def vc_check(ctx: commands.Context):  # pylint: disable=unused-argument
    """
    Check for whether VC is available in this bot.
    """

    if not discord.voice_client.has_nacl:
        raise commands.CheckFailure("voice cannot be used because PyNaCl is not loaded")

    if not discord.opus.is_loaded():
        raise commands.CheckFailure("voice cannot be used because libopus is not loaded")

    return True

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

def _render_table(data, fields=None):
  """ Helper to render a list of dictionaries as an HTML display object. """
  return IPython.core.display.HTML(datalab.utils.commands.HtmlBuilder.render_table(data, fields))

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

def fast_distinct(self):
        """
        Because standard distinct used on the all fields are very slow and works only with PostgreSQL database
        this method provides alternative to the standard distinct method.
        :return: qs with unique objects
        """
        return self.model.objects.filter(pk__in=self.values_list('pk', flat=True))

def logger(message, level=10):
    """Handle logging."""
    logging.getLogger(__name__).log(level, str(message))

def parse_value(self, value):
        """Cast value to `bool`."""
        parsed = super(BoolField, self).parse_value(value)
        return bool(parsed) if parsed is not None else None

def na_if(series, *values):
    """
    If values in a series match a specified value, change them to `np.nan`.

    Args:
        series: Series or vector, often symbolic.
        *values: Value(s) to convert to `np.nan` in the series.
    """

    series = pd.Series(series)
    series[series.isin(values)] = np.nan
    return series

def set_cursor_position(self, position):
        """Set cursor position"""
        position = self.get_position(position)
        cursor = self.textCursor()
        cursor.setPosition(position)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

def glr_path_static():
    """Returns path to packaged static files"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '_static'))

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

def get_header(request, header_service):
    """Return request's 'X_POLYAXON_...:' header, as a bytestring.

    Hide some test client ickyness where the header can be unicode.
    """
    service = request.META.get('HTTP_{}'.format(header_service), b'')
    if isinstance(service, str):
        # Work around django test client oddness
        service = service.encode(HTTP_HEADER_ENCODING)
    return service

def string_to_float_list(string_var):
        """Pull comma separated string values out of a text file and converts them to float list"""
        try:
            return [float(s) for s in string_var.strip('[').strip(']').split(', ')]
        except:
            return [float(s) for s in string_var.strip('[').strip(']').split(',')]

def __del__(self):
        """Cleanup any active connections and free all DDEML resources."""
        if self._hConv:
            DDE.Disconnect(self._hConv)
        if self._idInst:
            DDE.Uninitialize(self._idInst)

def strToBool(val):
    """
    Helper function to turn a string representation of "true" into
    boolean True.
    """
    if isinstance(val, str):
        val = val.lower()

    return val in ['true', 'on', 'yes', True]

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

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def prepare(doc):
    """Sets the caption_found and plot_found variables to False."""
    doc.caption_found = False
    doc.plot_found = False
    doc.listings_counter = 0

def _renamer(self, tre):
        """ renames newick from numbers to sample names"""
        ## get the tre with numbered tree tip labels
        names = tre.get_leaves()

        ## replace numbered names with snames
        for name in names:
            name.name = self.samples[int(name.name)]

        ## return with only topology and leaf labels
        return tre.write(format=9)

def comment (self, s, **args):
        """Write DOT comment."""
        self.write(u"// ")
        self.writeln(s=s, **args)

def to_str(obj):
    """Attempts to convert given object to a string object
    """
    if not isinstance(obj, str) and PY3 and isinstance(obj, bytes):
        obj = obj.decode('utf-8')
    return obj if isinstance(obj, string_types) else str(obj)

def OnRootView(self, event):
        """Reset view to the root of the tree"""
        self.adapter, tree, rows = self.RootNode()
        self.squareMap.SetModel(tree, self.adapter)
        self.RecordHistory()
        self.ConfigureViewTypeChoices()

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

def screen_cv2(self):
        """cv2 Image of current window screen"""
        pil_image = self.screen.convert('RGB')
        cv2_image = np.array(pil_image)
        pil_image.close()
        # Convert RGB to BGR 
        cv2_image = cv2_image[:, :, ::-1]
        return cv2_image

def to_utc(self, dt):
        """Convert any timestamp to UTC (with tzinfo)."""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=self.utc)
        return dt.astimezone(self.utc)

def hline(self, x, y, width, color):
        """Draw a horizontal line up to a given length."""
        self.rect(x, y, width, 1, color, fill=True)

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

def pylog(self, *args, **kwargs):
        """Display all available logging information."""
        printerr(self.name, args, kwargs, traceback.format_exc())

def _is_already_configured(configuration_details):
    """Returns `True` when alias already in shell config."""
    path = Path(configuration_details.path).expanduser()
    with path.open('r') as shell_config:
        return configuration_details.content in shell_config.read()

def safe_dump(data, stream=None, **kwds):
    """implementation of safe dumper using Ordered Dict Yaml Dumper"""
    return yaml.dump(data, stream=stream, Dumper=ODYD, **kwds)

def duplicated_rows(df, col_name):
    """ Return a DataFrame with the duplicated values of the column `col_name`
    in `df`."""
    _check_cols(df, [col_name])

    dups = df[pd.notnull(df[col_name]) & df.duplicated(subset=[col_name])]
    return dups

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

def eof(fd):
    """Determine if end-of-file is reached for file fd."""
    b = fd.read(1)
    end = len(b) == 0
    if not end:
        curpos = fd.tell()
        fd.seek(curpos - 1)
    return end

def _add_params_docstring(params):
    """ Add params to doc string
    """
    p_string = "\nAccepts the following paramters: \n"
    for param in params:
         p_string += "name: %s, required: %s, description: %s \n" % (param['name'], param['required'], param['description'])
    return p_string

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

def _basic_field_data(field, obj):
    """Returns ``obj.field`` data as a dict"""
    value = field.value_from_object(obj)
    return {Field.TYPE: FieldType.VAL, Field.VALUE: value}

def ensure_us_time_resolution(val):
    """Convert val out of numpy time, for use in to_dict.
    Needed because of numpy bug GH#7619"""
    if np.issubdtype(val.dtype, np.datetime64):
        val = val.astype('datetime64[us]')
    elif np.issubdtype(val.dtype, np.timedelta64):
        val = val.astype('timedelta64[us]')
    return val

def match_paren(self, tokens, item):
        """Matches a paren."""
        match, = tokens
        return self.match(match, item)

def full(self):
        """Return True if the queue is full"""
        if not self.size: return False
        return len(self.pq) == (self.size + self.removed_count)

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

def is_listish(obj):
    """Check if something quacks like a list."""
    if isinstance(obj, (list, tuple, set)):
        return True
    return is_sequence(obj)

def determine_types(self):
        """ Determine ES type names from request data.

        In particular `request.matchdict['collections']` is used to
        determine types names. Its value is comma-separated sequence
        of collection names under which views have been registered.
        """
        from nefertari.elasticsearch import ES
        collections = self.get_collections()
        resources = self.get_resources(collections)
        models = set([res.view.Model for res in resources])
        es_models = [mdl for mdl in models if mdl
                     and getattr(mdl, '_index_enabled', False)]
        types = [ES.src2type(mdl.__name__) for mdl in es_models]
        return types

def determine_types(self):
        """ Determine ES type names from request data.

        In particular `request.matchdict['collections']` is used to
        determine types names. Its value is comma-separated sequence
        of collection names under which views have been registered.
        """
        from nefertari.elasticsearch import ES
        collections = self.get_collections()
        resources = self.get_resources(collections)
        models = set([res.view.Model for res in resources])
        es_models = [mdl for mdl in models if mdl
                     and getattr(mdl, '_index_enabled', False)]
        types = [ES.src2type(mdl.__name__) for mdl in es_models]
        return types

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

def lines(input):
    """Remove comments and empty lines"""
    for raw_line in input:
        line = raw_line.strip()
        if line and not line.startswith('#'):
            yield strip_comments(line)

def __contains__ (self, key):
        """Check lowercase key item."""
        assert isinstance(key, basestring)
        return dict.__contains__(self, key.lower())

def close( self ):
        """
        Close the db and release memory
        """
        if self.db is not None:
            self.db.commit()
            self.db.close()
            self.db = None

        return

def _writable_dir(path):
    """Whether `path` is a directory, to which the user has write access."""
    return os.path.isdir(path) and os.access(path, os.W_OK)

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

def write_enum(fo, datum, schema):
    """An enum is encoded by a int, representing the zero-based position of
    the symbol in the schema."""
    index = schema['symbols'].index(datum)
    write_int(fo, index)

def __is__(cls, s):
        """Test if string matches this argument's format."""
        return s.startswith(cls.delims()[0]) and s.endswith(cls.delims()[1])

def _dump_enum(self, e, top=''):
        """Dump single enum type.
        
        Keyword arguments:
        top -- top namespace
        """
        self._print()
        self._print('enum {} {{'.format(e.name))
        self.defines.append('{}.{}'.format(top,e.name))
        
        self.tabs+=1
        for v in e.value:
            self._print('{} = {};'.format(v.name, v.number))
        self.tabs-=1
        self._print('}')

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

def get_enum_from_name(self, enum_name):
        """
            Return an enum from a name
        Args:
            enum_name (str): name of the enum
        Returns:
            Enum
        """
        return next((e for e in self.enums if e.name == enum_name), None)

def is_defined(self, objtxt, force_import=False):
        """Return True if object is defined"""
        return self.interpreter.is_defined(objtxt, force_import)

def Value(self, name):
    """Returns the value coresponding to the given enum name."""
    if name in self._enum_type.values_by_name:
      return self._enum_type.values_by_name[name].number
    raise ValueError('Enum %s has no value defined for name %s' % (
        self._enum_type.name, name))

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

def unpack_out(self, name):
        return self.parse("""
            $enum = $enum_class($value.value)
            """, enum_class=self._import_type(), value=name)["enum"]

def _validate_pos(df):
    """Validates the returned positional object
    """
    assert isinstance(df, pd.DataFrame)
    assert ["seqname", "position", "strand"] == df.columns.tolist()
    assert df.position.dtype == np.dtype("int64")
    assert df.strand.dtype == np.dtype("O")
    assert df.seqname.dtype == np.dtype("O")
    return df

def sed(match, replacement, path, modifiers=""):
    """
    Perform sed text substitution.
    """
    cmd = "sed -r -i 's/%s/%s/%s' %s" % (match, replacement, modifiers, path)

    process = Subprocess(cmd, shell=True)
    ret, out, err = process.run(timeout=60)
    if ret:
        raise SubprocessError("Sed command failed!")

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

def euler(self):
        """TODO DEPRECATE THIS?"""
        e_xyz = transformations.euler_from_matrix(self.rotation, 'sxyz')
        return np.array([180.0 / np.pi * a for a in e_xyz])

def check(self, var):
        """Check whether the provided value is a valid enum constant."""
        if not isinstance(var, _str_type): return False
        return _enum_mangle(var) in self._consts

def run(self):
        """Run the event loop."""
        self.signal_init()
        self.listen_init()
        self.logger.info('starting')
        self.loop.start()

def is_delimiter(line):
    """ True if a line consists only of a single punctuation character."""
    return bool(line) and line[0] in punctuation and line[0]*len(line) == line

def set_locale(request):
    """Return locale from GET lang param or automatically."""
    return request.query.get('lang', app.ps.babel.select_locale_by_request(request))

def check_auth(username, pwd):
    """This function is called to check if a username /
    password combination is valid.
    """
    cfg = get_current_config()
    return username == cfg["dashboard_httpauth"].split(
        ":")[0] and pwd == cfg["dashboard_httpauth"].split(":")[1]

def current_zipfile():
    """A function to vend the current zipfile, if any"""
    if zipfile.is_zipfile(sys.argv[0]):
        fd = open(sys.argv[0], "rb")
        return zipfile.ZipFile(fd)

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

def safe_exit(output):
    """exit without breaking pipes."""
    try:
        sys.stdout.write(output)
        sys.stdout.flush()
    except IOError:
        pass

def __exit__(self, type, value, traceback):
        """When the `with` statement ends."""

        if not self.asarfile:
            return

        self.asarfile.close()
        self.asarfile = None

def accel_next(self, *args):
        """Callback to go to the next tab. Called by the accel key.
        """
        if self.get_notebook().get_current_page() + 1 == self.get_notebook().get_n_pages():
            self.get_notebook().set_current_page(0)
        else:
            self.get_notebook().next_page()
        return True

def __exit__(self, type, value, traceback):
        """When the `with` statement ends."""

        if not self.asarfile:
            return

        self.asarfile.close()
        self.asarfile = None

def paste(cmd=paste_cmd, stdout=PIPE):
    """Returns system clipboard contents.
    """
    return Popen(cmd, stdout=stdout).communicate()[0].decode('utf-8')

def restore_scrollbar_position(self):
        """Restoring scrollbar position after main window is visible"""
        scrollbar_pos = self.get_option('scrollbar_position', None)
        if scrollbar_pos is not None:
            self.explorer.treewidget.set_scrollbar_position(scrollbar_pos)

def osx_clipboard_get():
    """ Get the clipboard's text on OS X.
    """
    p = subprocess.Popen(['pbpaste', '-Prefer', 'ascii'],
        stdout=subprocess.PIPE)
    text, stderr = p.communicate()
    # Text comes in with old Mac \r line endings. Change them to \n.
    text = text.replace('\r', '\n')
    return text

def format_exception(e):
    """Returns a string containing the type and text of the exception.

    """
    from .utils.printing import fill
    return '\n'.join(fill(line) for line in traceback.format_exception_only(type(e), e))

def close_all_but_this(self):
        """Close all files but the current one"""
        self.close_all_right()
        for i in range(0, self.get_stack_count()-1  ):
            self.close_file(0)

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

def unlock(self):
    """Closes the session to the database."""
    if not hasattr(self, 'session'):
      raise RuntimeError('Error detected! The session that you want to close does not exist any more!')
    logger.debug("Closed database session of '%s'" % self._database)
    self.session.close()
    del self.session

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

def fuzzy_get_tuple(dict_obj, approximate_key, dict_keys=None, key_and_value=False, similarity=0.6, default=None):
    """Find the closest matching key and/or value in a dictionary (must have all string keys!)"""
    return fuzzy_get(dict(('|'.join(str(k2) for k2 in k), v) for (k, v) in viewitems(dict_obj)),
                     '|'.join(str(k) for k in approximate_key), dict_keys=dict_keys,
                     key_and_value=key_and_value, similarity=similarity, default=default)

def index_nearest(value, array):
    """
    expects a _n.array
    returns the global minimum of (value-array)^2
    """

    a = (array-value)**2
    return index(a.min(), a)

def calc_list_average(l):
    """
    Calculates the average value of a list of numbers
    Returns a float
    """
    total = 0.0
    for value in l:
        total += value
    return total / len(l)

def get_typecast_value(self, value, type):
    """ Helper method to determine actual value based on type of feature variable.

    Args:
      value: Value in string form as it was parsed from datafile.
      type: Type denoting the feature flag type.

    Return:
      Value type-casted based on type of feature variable.
    """

    if type == entities.Variable.Type.BOOLEAN:
      return value == 'true'
    elif type == entities.Variable.Type.INTEGER:
      return int(value)
    elif type == entities.Variable.Type.DOUBLE:
      return float(value)
    else:
      return value

def get_scalar_product(self, other):
        """Returns the scalar product of this vector with the given
        other vector."""
        return self.x*other.x+self.y*other.y

def get_long_description():
    """Convert the README file into the long description.
    """
    with open(path.join(root_path, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
    return long_description

def list2string (inlist,delimit=' '):
    """
Converts a 1D list to a single long string for file output, using
the string.join function.

Usage:   list2string (inlist,delimit=' ')
Returns: the string created from inlist
"""
    stringlist = [makestr(_) for _ in inlist]
    return string.join(stringlist,delimit)

def get_best_encoding(stream):
    """Returns the default stream encoding if not found."""
    rv = getattr(stream, 'encoding', None) or sys.getdefaultencoding()
    if is_ascii_encoding(rv):
        return 'utf-8'
    return rv

def _float_almost_equal(float1, float2, places=7):
    """Return True if two numbers are equal up to the
    specified number of "places" after the decimal point.
    """

    if round(abs(float2 - float1), places) == 0:
        return True

    return False

def open_with_encoding(filename, encoding, mode='r'):
    """Return opened file with a specific encoding."""
    return io.open(filename, mode=mode, encoding=encoding,
                   newline='')

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

def main(argv=None):
    """Main command line interface."""

    if argv is None:
        argv = sys.argv[1:]

    cli = CommandLineTool()
    return cli.run(argv)

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

def _maybe_fill(arr, fill_value=np.nan):
    """
    if we have a compatible fill_value and arr dtype, then fill
    """
    if _isna_compat(arr, fill_value):
        arr.fill(fill_value)
    return arr

def intersect(d1, d2):
    """Intersect dictionaries d1 and d2 by key *and* value."""
    return dict((k, d1[k]) for k in d1 if k in d2 and d1[k] == d2[k])

def make_code_from_py(filename):
    """Get source from `filename` and make a code object of it."""
    # Open the source file.
    try:
        source_file = open_source(filename)
    except IOError:
        raise NoSource("No file to run: %r" % filename)

    try:
        source = source_file.read()
    finally:
        source_file.close()

    # We have the source.  `compile` still needs the last line to be clean,
    # so make sure it is, then compile a code object from it.
    if not source or source[-1] != '\n':
        source += '\n'
    code = compile(source, filename, "exec")

    return code

def __init__(self):
    """Initializes a filter object."""
    super(FilterObject, self).__init__()
    self._filter_expression = None
    self._matcher = None

def dist(x1, x2, axis=0):
    """Return the distance between two points.

    Set axis=1 if x1 is a vector and x2 a matrix to get a vector of distances.
    """
    return np.linalg.norm(x2 - x1, axis=axis)

def __init__(self, function):
		"""function: to be called with each stream element as its
		only argument
		"""
		super(filter, self).__init__()
		self.function = function

def euclidean(c1, c2):
    """Square of the euclidean distance"""
    diffs = ((i - j) for i, j in zip(c1, c2))
    return sum(x * x for x in diffs)

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

def tpr(y, z):
    """True positive rate `tp / (tp + fn)`
    """
    tp, tn, fp, fn = contingency_table(y, z)
    return tp / (tp + fn)

def join_cols(cols):
    """Join list of columns into a string for a SQL query"""
    return ", ".join([i for i in cols]) if isinstance(cols, (list, tuple, set)) else cols

def apply_fit(xy,coeffs):
    """ Apply the coefficients from a linear fit to
        an array of x,y positions.

        The coeffs come from the 'coeffs' member of the
        'fit_arrays()' output.
    """
    x_new = coeffs[0][2] + coeffs[0][0]*xy[:,0] + coeffs[0][1]*xy[:,1]
    y_new = coeffs[1][2] + coeffs[1][0]*xy[:,0] + coeffs[1][1]*xy[:,1]

    return x_new,y_new

def vectorize(values):
    """
    Takes a value or list of values and returns a single result, joined by ","
    if necessary.
    """
    if isinstance(values, list):
        return ','.join(str(v) for v in values)
    return values

def figsize(x=8, y=7., aspect=1.):
    """ manually set the default figure size of plots
    ::Arguments::
        x (float): x-axis size
        y (float): y-axis size
        aspect (float): aspect ratio scalar
    """
    # update rcparams with adjusted figsize params
    mpl.rcParams.update({'figure.figsize': (x*aspect, y)})

def convert_time_string(date_str):
    """ Change a date string from the format 2018-08-15T23:55:17 into a datetime object """
    dt, _, _ = date_str.partition(".")
    dt = datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S")
    return dt

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

def mean_cl_boot(series, n_samples=1000, confidence_interval=0.95,
                 random_state=None):
    """
    Bootstrapped mean with confidence limits
    """
    return bootstrap_statistics(series, np.mean,
                                n_samples=n_samples,
                                confidence_interval=confidence_interval,
                                random_state=random_state)

def handle_request_parsing_error(err, req, schema, error_status_code, error_headers):
    """webargs error handler that uses Flask-RESTful's abort function to return
    a JSON error response to the client.
    """
    abort(error_status_code, errors=err.messages)

def list_apis(awsclient):
    """List APIs in account."""
    client_api = awsclient.get_client('apigateway')

    apis = client_api.get_rest_apis()['items']

    for api in apis:
        print(json2table(api))

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

def do_next(self, args):
        """Step over the next statement
        """
        self._do_print_from_last_cmd = True
        self._interp.step_over()
        return True

def checkbox_uncheck(self, force_check=False):
        """
        Wrapper to uncheck a checkbox
        """
        if self.get_attribute('checked'):
            self.click(force_click=force_check)

def export(defn):
    """Decorator to explicitly mark functions that are exposed in a lib."""
    globals()[defn.__name__] = defn
    __all__.append(defn.__name__)
    return defn

def static_url(path, absolute=False):
    """ Shorthand for returning a URL for the requested static file.

    Arguments:

    path -- the path to the file (relative to the static files directory)
    absolute -- whether the link should be absolute or relative
    """

    if os.sep != '/':
        path = '/'.join(path.split(os.sep))

    return flask.url_for('static', filename=path, _external=absolute)

def get_site_name(request):
    """Return the domain:port part of the URL without scheme.
    Eg: facebook.com, 127.0.0.1:8080, etc.
    """
    urlparts = request.urlparts
    return ':'.join([urlparts.hostname, str(urlparts.port)])

def index():
    """ Display productpage with normal user and test user buttons"""
    global productpage

    table = json2html.convert(json = json.dumps(productpage),
                              table_attributes="class=\"table table-condensed table-bordered table-hover\"")

    return render_template('index.html', serviceTable=table)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def jsonify(resource):
    """Return a Flask ``Response`` object containing a
    JSON representation of *resource*.

    :param resource: The resource to act as the basis of the response
    """

    response = flask.jsonify(resource.to_dict())
    response = add_link_headers(response, resource.links())
    return response

def copy(string, **kwargs):
    """Copy given string into system clipboard."""
    window = Tk()
    window.withdraw()
    window.clipboard_clear()
    window.clipboard_append(string)
    window.destroy()
    return

def set_header(self, name, value):
        """ Create a new response header, replacing any previously defined
            headers with the same name. """
        self._headers[_hkey(name)] = [_hval(value)]

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

def end(self):
        """End of the Glances server session."""
        if not self.args.disable_autodiscover:
            self.autodiscover_client.close()
        self.server.end()

def scatterplot_matrix(df, features, downsample_frac=None, figsize=(15, 15)):
    """
    Plot a scatterplot matrix for a list of features, colored by target value.

    Example: `scatterplot_matrix(X, X.columns.tolist(), downsample_frac=0.01)`

    Args:
        df: Pandas dataframe containing the target column (named 'target').
        features: The list of features to include in the correlation plot.
        downsample_frac: Dataframe downsampling rate (0.1 to include 10% of the dataset).
        figsize: The size of the plot.
    """

    if downsample_frac:
        df = df.sample(frac=downsample_frac)

    plt.figure(figsize=figsize)
    sns.pairplot(df[features], hue='target')
    plt.show()

def handle_exception(error):
        """Simple method for handling exceptions raised by `PyBankID`.

        :param flask_pybankid.FlaskPyBankIDError error: The exception to handle.
        :return: The exception represented as a dictionary.
        :rtype: dict

        """
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

def count(data, axis=None):
    """Count the number of non-NA in this array along the given axis or axes
    """
    return np.sum(np.logical_not(isnull(data)), axis=axis)

def dispatch(self):
    """Wraps the dispatch method to add session support."""
    try:
      webapp2.RequestHandler.dispatch(self)
    finally:
      self.session_store.save_sessions(self.response)

def count(data, axis=None):
    """Count the number of non-NA in this array along the given axis or axes
    """
    return np.sum(np.logical_not(isnull(data)), axis=axis)

def staticdir():
    """Return the location of the static data directory."""
    root = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(root, "static")

def cover(session):
    """Run the final coverage report.
    This outputs the coverage report aggregating coverage from the unit
    test runs (not system test runs), and then erases coverage data.
    """
    session.interpreter = 'python3.6'
    session.install('coverage', 'pytest-cov')
    session.run('coverage', 'report', '--show-missing', '--fail-under=100')
    session.run('coverage', 'erase')

def staticdir():
    """Return the location of the static data directory."""
    root = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(root, "static")

def rnormal(mu, tau, size=None):
    """
    Random normal variates.
    """
    return np.random.normal(mu, 1. / np.sqrt(tau), size)

def flatten(nested, containers=(list, tuple)):
    """ Flatten a nested list by yielding its scalar items.
    """
    for item in nested:
        if hasattr(item, "next") or isinstance(item, containers):
            for subitem in flatten(item):
                yield subitem
        else:
            yield item

def is_bool_matrix(l):
    r"""Checks if l is a 2D numpy array of bools

    """
    if isinstance(l, np.ndarray):
        if l.ndim == 2 and (l.dtype == bool):
            return True
    return False

def get_mac_dot_app_dir(directory):
    """Returns parent directory of mac .app

    Args:

       directory (str): Current directory

    Returns:

       (str): Parent directory of mac .app
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(directory)))

def _crop_list_to_size(l, size):
    """Make a list a certain size"""
    for x in range(size - len(l)):
        l.append(False)
    for x in range(len(l) - size):
        l.pop()
    return l

def closing_plugin(self, cancelable=False):
        """Perform actions before parent main window is closed"""
        self.dialog_manager.close_all()
        self.shell.exit_interpreter()
        return True

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

def _closeResources(self):
        """ Closes the root Dataset.
        """
        logger.info("Closing: {}".format(self._fileName))
        self._h5Group.close()
        self._h5Group = None

def software_fibonacci(n):
    """ a normal old python function to return the Nth fibonacci number. """
    a, b = 0, 1
    for i in range(n):
        a, b = b, a + b
    return a

def flush(self):
        """ Force commit changes to the file and stdout """
        if not self.nostdout:
            self.stdout.flush()
        if self.file is not None:
            self.file.flush()

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

def go_to_background():
    """ Daemonize the running process. """
    try:
        if os.fork():
            sys.exit()
    except OSError as errmsg:
        LOGGER.error('Fork failed: {0}'.format(errmsg))
        sys.exit('Fork failed')

def create_app():
    """Create a Qt application."""
    global QT_APP
    QT_APP = QApplication.instance()
    if QT_APP is None:  # pragma: no cover
        QT_APP = QApplication(sys.argv)
    return QT_APP

def go_to_background():
    """ Daemonize the running process. """
    try:
        if os.fork():
            sys.exit()
    except OSError as errmsg:
        LOGGER.error('Fork failed: {0}'.format(errmsg))
        sys.exit('Fork failed')

def pack_triples_numpy(triples):
    """Packs a list of triple indexes into a 2D numpy array."""
    if len(triples) == 0:
        return np.array([], dtype=np.int64)
    return np.stack(list(map(_transform_triple_numpy, triples)), axis=0)

def fixed(ctx, number, decimals=2, no_commas=False):
    """
    Formats the given number in decimal format using a period and commas
    """
    value = _round(ctx, number, decimals)
    format_str = '{:f}' if no_commas else '{:,f}'
    return format_str.format(value)

def _make_index(df, cols=META_IDX):
    """Create an index from the columns of a dataframe"""
    return pd.MultiIndex.from_tuples(
        pd.unique(list(zip(*[df[col] for col in cols]))), names=tuple(cols))

def string_format_func(s):
	"""
	Function used internally to format string data for output to XML.
	Escapes back-slashes and quotes, and wraps the resulting string in
	quotes.
	"""
	return u"\"%s\"" % unicode(s).replace(u"\\", u"\\\\").replace(u"\"", u"\\\"")

def map_parameters(cls, params):
        """Maps parameters to form field names"""

        d = {}
        for k, v in six.iteritems(params):
            d[cls.FIELD_MAP.get(k.lower(), k)] = v
        return d

def position(self, x, y, text):
        """
            ANSI Escape sequences
            http://ascii-table.com/ansi-escape-sequences.php
        """
        sys.stdout.write("\x1b7\x1b[%d;%df%s\x1b8" % (x, y, text))
        sys.stdout.flush()

def RunSphinxAPIDoc(_):
  """Runs sphinx-apidoc to auto-generate documentation."""
  current_directory = os.path.abspath(os.path.dirname(__file__))
  module = os.path.join(current_directory, '..', 'plaso')
  api_directory = os.path.join(current_directory, 'sources', 'api')
  apidoc.main(['-o', api_directory, module, '--force'])

def none(self):
        """
        Returns an empty QuerySet.
        """
        return EmptyQuerySet(model=self.model, using=self._using, connection=self._connection)

def ucamel_method(func):
    """
    Decorator to ensure the given snake_case method is also written in
    UpperCamelCase in the given namespace. That was mainly written to
    avoid confusion when using wxPython and its UpperCamelCaseMethods.
    """
    frame_locals = inspect.currentframe().f_back.f_locals
    frame_locals[snake2ucamel(func.__name__)] = func
    return func

def main(ctx, connection):
    """Command line interface for PyBEL."""
    ctx.obj = Manager(connection=connection)
    ctx.obj.bind()

def ucamel_method(func):
    """
    Decorator to ensure the given snake_case method is also written in
    UpperCamelCase in the given namespace. That was mainly written to
    avoid confusion when using wxPython and its UpperCamelCaseMethods.
    """
    frame_locals = inspect.currentframe().f_back.f_locals
    frame_locals[snake2ucamel(func.__name__)] = func
    return func

def sqliteRowsToDicts(sqliteRows):
    """
    Unpacks sqlite rows as returned by fetchall
    into an array of simple dicts.

    :param sqliteRows: array of rows returned from fetchall DB call
    :return:  array of dicts, keyed by the column names.
    """
    return map(lambda r: dict(zip(r.keys(), r)), sqliteRows)

def myreplace(astr, thefind, thereplace):
    """in string astr replace all occurences of thefind with thereplace"""
    alist = astr.split(thefind)
    new_s = alist.split(thereplace)
    return new_s

def nb_to_python(nb_path):
    """convert notebook to python script"""
    exporter = python.PythonExporter()
    output, resources = exporter.from_filename(nb_path)
    return output

def pack_triples_numpy(triples):
    """Packs a list of triple indexes into a 2D numpy array."""
    if len(triples) == 0:
        return np.array([], dtype=np.int64)
    return np.stack(list(map(_transform_triple_numpy, triples)), axis=0)

def comment (self, s, **args):
        """Write GML comment."""
        self.writeln(s=u'comment "%s"' % s, **args)

def circ_permutation(items):
    """Calculate the circular permutation for a given list of items."""
    permutations = []
    for i in range(len(items)):
        permutations.append(items[i:] + items[:i])
    return permutations

def check_hash_key(query_on, key):
    """Only allows == against query_on.hash_key"""
    return (
        isinstance(key, BaseCondition) and
        (key.operation == "==") and
        (key.column is query_on.hash_key)
    )

def generate_device_id(steamid):
    """Generate Android device id

    :param steamid: Steam ID
    :type steamid: :class:`.SteamID`, :class:`int`
    :return: android device id
    :rtype: str
    """
    h = hexlify(sha1_hash(str(steamid).encode('ascii'))).decode('ascii')
    return "android:%s-%s-%s-%s-%s" % (h[:8], h[8:12], h[12:16], h[16:20], h[20:32])

def __init__(self, token, editor=None):
        """Create a GistAPI object

        Arguments:
            token: an authentication token
            editor: path to the editor to use when editing a gist

        """
        self.token = token
        self.editor = editor
        self.session = requests.Session()

def generate_id(self):
        """Generate a fresh id"""
        if self.use_repeatable_ids:
            self.repeatable_id_counter += 1
            return 'autobaked-{}'.format(self.repeatable_id_counter)
        else:
            return str(uuid4())

def create_db(app, appbuilder):
    """
        Create all your database objects (SQLAlchemy specific).
    """
    from flask_appbuilder.models.sqla import Base

    _appbuilder = import_application(app, appbuilder)
    engine = _appbuilder.get_session.get_bind(mapper=None, clause=None)
    Base.metadata.create_all(engine)
    click.echo(click.style("DB objects created", fg="green"))

def get_incomplete_path(filename):
  """Returns a temporary filename based on filename."""
  random_suffix = "".join(
      random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
  return filename + ".incomplete" + random_suffix

def from_json_list(cls, api_client, data):
        """Convert a list of JSON values to a list of models
        """
        return [cls.from_json(api_client, item) for item in data]

def generate_hash(self, length=30):
        """ Generate random string of given length """
        import random, string
        chars = string.ascii_letters + string.digits
        ran = random.SystemRandom().choice
        hash = ''.join(ran(chars) for i in range(length))
        return hash

def ensure_hbounds(self):
        """Ensure the cursor is within horizontal screen bounds."""
        self.cursor.x = min(max(0, self.cursor.x), self.columns - 1)

def timespan(start_time):
    """Return time in milliseconds from start_time"""

    timespan = datetime.datetime.now() - start_time
    timespan_ms = timespan.total_seconds() * 1000
    return timespan_ms

def lon_lat_bins(bb, coord_bin_width):
    """
    Define bin edges for disaggregation histograms.

    Given bins data as provided by :func:`collect_bin_data`, this function
    finds edges of histograms, taking into account maximum and minimum values
    of magnitude, distance and coordinates as well as requested sizes/numbers
    of bins.
    """
    west, south, east, north = bb
    west = numpy.floor(west / coord_bin_width) * coord_bin_width
    east = numpy.ceil(east / coord_bin_width) * coord_bin_width
    lon_extent = get_longitudinal_extent(west, east)
    lon_bins, _, _ = npoints_between(
        west, 0, 0, east, 0, 0,
        numpy.round(lon_extent / coord_bin_width + 1))
    lat_bins = coord_bin_width * numpy.arange(
        int(numpy.floor(south / coord_bin_width)),
        int(numpy.ceil(north / coord_bin_width) + 1))
    return lon_bins, lat_bins

def get_geoip(ip):
    """Lookup country for IP address."""
    reader = geolite2.reader()
    ip_data = reader.get(ip) or {}
    return ip_data.get('country', {}).get('iso_code')

def int_to_date(date):
    """
    Convert an int of form yyyymmdd to a python date object.
    """

    year = date // 10**4
    month = date % 10**4 // 10**2
    day = date % 10**2

    return datetime.date(year, month, day)

def out_shape_from_array(arr):
    """Get the output shape from an array."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.shape
    else:
        return (arr.shape[1],)

def __round_time(self, dt):
    """Round a datetime object to a multiple of a timedelta
    dt : datetime.datetime object, default now.
    """
    round_to = self._resolution.total_seconds()
    seconds  = (dt - dt.min).seconds
    rounding = (seconds + round_to / 2) // round_to * round_to
    return dt + timedelta(0, rounding - seconds, -dt.microsecond)

def get_font_list():
    """Returns a sorted list of all system font names"""

    font_map = pangocairo.cairo_font_map_get_default()
    font_list = [f.get_name() for f in font_map.list_families()]
    font_list.sort()

    return font_list

def to_utc(self, dt):
        """Convert any timestamp to UTC (with tzinfo)."""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=self.utc)
        return dt.astimezone(self.utc)

def FindMethodByName(self, name):
    """Searches for the specified method, and returns its descriptor."""
    for method in self.methods:
      if name == method.name:
        return method
    return None

def _dt_to_epoch(dt):
        """Convert datetime to epoch seconds."""
        try:
            epoch = dt.timestamp()
        except AttributeError:  # py2
            epoch = (dt - datetime(1970, 1, 1)).total_seconds()
        return epoch

def return_letters_from_string(text):
    """Get letters from string only."""
    out = ""
    for letter in text:
        if letter.isalpha():
            out += letter
    return out

def object_to_json(obj):
    """Convert object that cannot be natively serialized by python to JSON representation."""
    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
        return obj.isoformat()
    return str(obj)

def get_pid_list():
    """Returns a list of PIDs currently running on the system."""
    pids = [int(x) for x in os.listdir('/proc') if x.isdigit()]
    return pids

def method(func):
    """Wrap a function as a method."""
    attr = abc.abstractmethod(func)
    attr.__imethod__ = True
    return attr

def getRect(self):
		"""
		Returns the window bounds as a tuple of (x,y,w,h)
		"""
		return (self.x, self.y, self.w, self.h)

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

def get_body_size(params, boundary):
    """Returns the number of bytes that the multipart/form-data encoding
    of ``params`` will be."""
    size = sum(p.get_size(boundary) for p in MultipartParam.from_params(params))
    return size + len(boundary) + 6

def input_int_default(question="", default=0):
    """A function that works for both, Python 2.x and Python 3.x.
       It asks the user for input and returns it as a string.
    """
    answer = input_string(question)
    if answer == "" or answer == "yes":
        return default
    else:
        return int(answer)

def index(self, elem):
        """Find the index of elem in the reversed iterator."""
        return _coconut.len(self._iter) - self._iter.index(elem) - 1

def get_default_args(func):
    """
    returns a dictionary of arg_name:default_values for the input function
    """
    args, varargs, keywords, defaults = getargspec_no_self(func)
    return dict(zip(args[-len(defaults):], defaults))

def c_array(ctype, values):
    """Convert a python string to c array."""
    if isinstance(values, np.ndarray) and values.dtype.itemsize == ctypes.sizeof(ctype):
        return (ctype * len(values)).from_buffer_copy(values)
    return (ctype * len(values))(*values)

def get_shape(self):
		"""
		Return a tuple of this array's dimensions.  This is done by
		querying the Dim children.  Note that once it has been
		created, it is also possible to examine an Array object's
		.array attribute directly, and doing that is much faster.
		"""
		return tuple(int(c.pcdata) for c in self.getElementsByTagName(ligolw.Dim.tagName))[::-1]

def zeros(self, name, **kwargs):
        """Create an array. Keyword arguments as per
        :func:`zarr.creation.zeros`."""
        return self._write_op(self._zeros_nosync, name, **kwargs)

def dist(x1, x2, axis=0):
    """Return the distance between two points.

    Set axis=1 if x1 is a vector and x2 a matrix to get a vector of distances.
    """
    return np.linalg.norm(x2 - x1, axis=axis)

def delaunay_2d(self, tol=1e-05, alpha=0.0, offset=1.0, bound=False):
        """Apply a delaunay 2D filter along the best fitting plane. This
        extracts the grid's points and perfoms the triangulation on those alone.
        """
        return PolyData(self.points).delaunay_2d(tol=tol, alpha=alpha, offset=offset, bound=bound)

def parse_domain(url):
    """ parse the domain from the url """
    domain_match = lib.DOMAIN_REGEX.match(url)
    if domain_match:
        return domain_match.group()

def close_all():
    """Close all open/active plotters"""
    for key, p in _ALL_PLOTTERS.items():
        p.close()
    _ALL_PLOTTERS.clear()
    return True

def _DateToEpoch(date):
  """Converts python datetime to epoch microseconds."""
  tz_zero = datetime.datetime.utcfromtimestamp(0)
  diff_sec = int((date - tz_zero).total_seconds())
  return diff_sec * 1000000

def forget_canvas(canvas):
    """ Forget about the given canvas. Used by the canvas when closed.
    """
    cc = [c() for c in canvasses if c() is not None]
    while canvas in cc:
        cc.remove(canvas)
    canvasses[:] = [weakref.ref(c) for c in cc]

def AmericanDateToEpoch(self, date_str):
    """Take a US format date and return epoch."""
    try:
      epoch = time.strptime(date_str, "%m/%d/%Y")
      return int(calendar.timegm(epoch)) * 1000000
    except ValueError:
      return 0

def __delitem__(self, key):
        """Remove a variable from this dataset.
        """
        del self._variables[key]
        self._coord_names.discard(key)

def get_file_name(url):
  """Returns file name of file at given url."""
  return os.path.basename(urllib.parse.urlparse(url).path) or 'unknown_name'

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

def walk_tree(root):
    """Pre-order depth-first"""
    yield root

    for child in root.children:
        for el in walk_tree(child):
            yield el

def calc_volume(self, sample: np.ndarray):
        """Find the RMS of the audio"""
        return sqrt(np.mean(np.square(sample)))

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

def load_feature(fname, language):
    """ Load and parse a feature file. """

    fname = os.path.abspath(fname)
    feat = parse_file(fname, language)
    return feat

def get_java_path():
  """Get the path of java executable"""
  java_home = os.environ.get("JAVA_HOME")
  return os.path.join(java_home, BIN_DIR, "java")

def is_collection(obj):
    """Tests if an object is a collection."""

    col = getattr(obj, '__getitem__', False)
    val = False if (not col) else True

    if isinstance(obj, basestring):
        val = False

    return val

def newest_file(file_iterable):
  """
  Returns the name of the newest file given an iterable of file names.

  """
  return max(file_iterable, key=lambda fname: os.path.getmtime(fname))

def cell_ends_with_code(lines):
    """Is the last line of the cell a line with code?"""
    if not lines:
        return False
    if not lines[-1].strip():
        return False
    if lines[-1].startswith('#'):
        return False
    return True

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

def _get_column_types(self, data):
        """Get a list of the data types for each column in *data*."""
        columns = list(zip_longest(*data))
        return [self._get_column_type(column) for column in columns]

def get_table_pos(self, tablename):
        """
        :param str tablename: Name of table to get position of.
        :return: Upper left (row, col) coordinate of the named table.
        """
        _table, (row, col) = self.__tables[tablename]
        return (row, col)

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

def l2_norm(arr):
    """
    The l2 norm of an array is is defined as: sqrt(||x||), where ||x|| is the
    dot product of the vector.
    """
    arr = np.asarray(arr)
    return np.sqrt(np.dot(arr.ravel().squeeze(), arr.ravel().squeeze()))

def calc_volume(self, sample: np.ndarray):
        """Find the RMS of the audio"""
        return sqrt(np.mean(np.square(sample)))

def array_size(x, axis):
  """Calculate the size of `x` along `axis` dimensions only."""
  axis_shape = x.shape if axis is None else tuple(x.shape[a] for a in axis)
  return max(numpy.prod(axis_shape), 1)

def parse_text_to_dict(self, txt):
        """ 
        takes a string and parses via NLP, ready for mapping
        """
        op = {}
        print('TODO - import NLP, split into verbs / nouns')
        op['nouns'] = txt
        op['verbs'] = txt
        
        return op

def get_idx_rect(index_list):
    """Extract the boundaries from a list of indexes"""
    rows, cols = list(zip(*[(i.row(), i.column()) for i in index_list]))
    return ( min(rows), max(rows), min(cols), max(cols) )

def get_list_dimensions(_list):
    """
    Takes a nested list and returns the size of each dimension followed
    by the element type in the list
    """
    if isinstance(_list, list) or isinstance(_list, tuple):
        return [len(_list)] + get_list_dimensions(_list[0])
    return []

def download_file_from_bucket(self, bucket, file_path, key):
        """ Download file from S3 Bucket """
        with open(file_path, 'wb') as data:
            self.__s3.download_fileobj(bucket, key, data)
            return file_path

def log_no_newline(self, msg):
      """ print the message to the predefined log file without newline """
      self.print2file(self.logfile, False, False, msg)

def get_previous_month(self):
        """Returns date range for the previous full month."""
        end = utils.get_month_start() - relativedelta(days=1)
        end = utils.to_datetime(end)
        start = utils.get_month_start(end)
        return start, end

def off(self):
        """Turn off curses"""
        self.win.keypad(0)
        curses.nocbreak()
        curses.echo()
        try:
            curses.curs_set(1)
        except:
            pass
        curses.endwin()

def gday_of_year(self):
        """Return the number of days since January 1 of the given year."""
        return (self.date - dt.date(self.date.year, 1, 1)).days

def send_text(self, text):
        """Send a plain text message to the room."""
        return self.client.api.send_message(self.room_id, text)

def _num_cpus_darwin():
    """Return the number of active CPUs on a Darwin system."""
    p = subprocess.Popen(['sysctl','-n','hw.ncpu'],stdout=subprocess.PIPE)
    return p.stdout.read()

def me(self):
        """Similar to :attr:`.Guild.me` except it may return the :class:`.ClientUser` in private message contexts."""
        return self.guild.me if self.guild is not None else self.bot.user

def get_shape(self):
		"""
		Return a tuple of this array's dimensions.  This is done by
		querying the Dim children.  Note that once it has been
		created, it is also possible to examine an Array object's
		.array attribute directly, and doing that is much faster.
		"""
		return tuple(int(c.pcdata) for c in self.getElementsByTagName(ligolw.Dim.tagName))[::-1]

async def join(self, ctx, *, channel: discord.VoiceChannel):
        """Joins a voice channel"""

        if ctx.voice_client is not None:
            return await ctx.voice_client.move_to(channel)

        await channel.connect()

def _lookup_parent(self, cls):
        """Lookup a transitive parent object that is an instance
            of a given class."""
        codeobj = self.parent
        while codeobj is not None and not isinstance(codeobj, cls):
            codeobj = codeobj.parent
        return codeobj

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

def _run_cmd_get_output(cmd):
    """Runs a shell command, returns console output.

    Mimics python3's subprocess.getoutput
    """
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    out, err = process.communicate()
    return out or err

def _run_cmd_get_output(cmd):
    """Runs a shell command, returns console output.

    Mimics python3's subprocess.getoutput
    """
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    out, err = process.communicate()
    return out or err

def cpp_prog_builder(build_context, target):
    """Build a C++ binary executable"""
    yprint(build_context.conf, 'Build CppProg', target)
    workspace_dir = build_context.get_workspace('CppProg', target.name)
    build_cpp(build_context, target, target.compiler_config, workspace_dir)

def get_parent_folder_name(file_path):
    """Finds parent folder of file

    :param file_path: path
    :return: Name of folder container
    """
    return os.path.split(os.path.split(os.path.abspath(file_path))[0])[-1]

def method_header(method_name, nogil=False, idx_as_arg=False):
    """Returns the Cython method header for methods without arguments except
    `self`."""
    if not config.FASTCYTHON:
        nogil = False
    header = 'cpdef inline void %s(self' % method_name
    header += ', int idx)' if idx_as_arg else ')'
    header += ' nogil:' if nogil else ':'
    return header

def grandparent_path(self):
        """ return grandparent's path string """
        return os.path.basename(os.path.join(self.path, '../..'))

def get_order(self, codes):
        """Return evidence codes in order shown in code2name."""
        return sorted(codes, key=lambda e: [self.ev2idx.get(e)])

def grandparent_path(self):
        """ return grandparent's path string """
        return os.path.basename(os.path.join(self.path, '../..'))

def bulk_query(self, query, *multiparams):
        """Bulk insert or update."""

        with self.get_connection() as conn:
            conn.bulk_query(query, *multiparams)

def read(filename):
    """Read and return `filename` in root dir of project and return string"""
    return codecs.open(os.path.join(__DIR__, filename), 'r').read()

def match_aspect_to_viewport(self):
        """Updates Camera.aspect to match the viewport's aspect ratio."""
        viewport = self.viewport
        self.aspect = float(viewport.width) / viewport.height

def previous_quarter(d):
    """
    Retrieve the previous quarter for dt
    """
    from django_toolkit.datetime_util import quarter as datetime_quarter
    return quarter( (datetime_quarter(datetime(d.year, d.month, d.day))[0] + timedelta(days=-1)).date() )

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

def get_memory_usage():
    """Gets RAM memory usage

    :return: MB of memory used by this process
    """
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss
    return mem / (1024 * 1024)

def dict_to_querystring(dictionary):
    """Converts a dict to a querystring suitable to be appended to a URL."""
    s = u""
    for d in dictionary.keys():
        s = unicode.format(u"{0}{1}={2}&", s, d, dictionary[d])
    return s[:-1]

def extract_module_locals(depth=0):
    """Returns (module, locals) of the funciton `depth` frames away from the caller"""
    f = sys._getframe(depth + 1)
    global_ns = f.f_globals
    module = sys.modules[global_ns['__name__']]
    return (module, f.f_locals)

def set_property(self, key, value):
        """
        Update only one property in the dict
        """
        self.properties[key] = value
        self.sync_properties()

def __absolute__(self, uri):
        """ Get the absolute uri for a file

        :param uri: URI of the resource to be retrieved
        :return: Absolute Path
        """
        return op.abspath(op.join(self.__path__, uri))

def remove_legend(ax=None):
    """Remove legend for axes or gca.

    See http://osdir.com/ml/python.matplotlib.general/2005-07/msg00285.html
    """
    from pylab import gca, draw
    if ax is None:
        ax = gca()
    ax.legend_ = None
    draw()

def read_corpus(file_name):
    """
    Read and return the data from a corpus json file.
    """
    with io.open(file_name, encoding='utf-8') as data_file:
        return yaml.load(data_file)

def _delete_whitespace(self):
        """Delete all whitespace from the end of the line."""
        while isinstance(self._lines[-1], (self._Space, self._LineBreak,
                                           self._Indent)):
            del self._lines[-1]

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

def percentile_index(a, q):
    """
    Returns the index of the value at the Qth percentile in array a.
    """
    return np.where(
        a==np.percentile(a, q, interpolation='nearest')
    )[0][0]

def find_mapping(es_url, index):
    """ Find the mapping given an index """

    mapping = None

    backend = find_perceval_backend(es_url, index)

    if backend:
        mapping = backend.get_elastic_mappings()

    if mapping:
        logging.debug("MAPPING FOUND:\n%s", json.dumps(json.loads(mapping['items']), indent=True))
    return mapping

def get_value(key, obj, default=missing):
    """Helper for pulling a keyed value off various types of objects"""
    if isinstance(key, int):
        return _get_value_for_key(key, obj, default)
    return _get_value_for_keys(key.split('.'), obj, default)

def get_model_index_properties(instance, index):
    """Return the list of properties specified for a model in an index."""
    mapping = get_index_mapping(index)
    doc_type = instance._meta.model_name.lower()
    return list(mapping["mappings"][doc_type]["properties"].keys())

def get_property_by_name(pif, name):
    """Get a property by name"""
    return next((x for x in pif.properties if x.name == name), None)

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

def get_var(self, name):
        """ Returns the variable set with the given name.
        """
        for var in self.vars:
            if var.name == name:
                return var
        else:
            raise ValueError

def disable_stdout_buffering():
    """This turns off stdout buffering so that outputs are immediately
    materialized and log messages show up before the program exits"""
    stdout_orig = sys.stdout
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
    # NOTE(brandyn): This removes the original stdout
    return stdout_orig

def _get_var_from_string(item):
    """ Get resource variable. """
    modname, varname = _split_mod_var_names(item)
    if modname:
        mod = __import__(modname, globals(), locals(), [varname], -1)
        return getattr(mod, varname)
    else:
        return globals()[varname]

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

def timespan(start_time):
    """Return time in milliseconds from start_time"""

    timespan = datetime.datetime.now() - start_time
    timespan_ms = timespan.total_seconds() * 1000
    return timespan_ms

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

def quote(s, unsafe='/'):
    """Pass in a dictionary that has unsafe characters as the keys, and the percent
    encoded value as the value."""
    res = s.replace('%', '%25')
    for c in unsafe:
        res = res.replace(c, '%' + (hex(ord(c)).upper())[2:])
    return res

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

def get_value(self, context):
        """Run python eval on the input string."""
        if self.value:
            return expressions.eval_string(self.value, context)
        else:
            # Empty input raises cryptic EOF syntax err, this more human
            # friendly
            raise ValueError('!py string expression is empty. It must be a '
                             'valid python expression instead.')

def _spawn(self, func, *args, **kwargs):
        """Spawn a handler function.

        Spawns the supplied ``func`` with ``*args`` and ``**kwargs``
        as a gevent greenlet.

        :param func: A callable to call.
        :param args: Arguments to ``func``.
        :param kwargs: Keyword arguments to ``func``.
        """
        gevent.spawn(func, *args, **kwargs)

def get_code_language(self):
        """
        This is largely copied from bokeh.sphinxext.bokeh_plot.run
        """
        js_source = self.get_js_source()
        if self.options.get("include_html", False):
            resources = get_sphinx_resources(include_bokehjs_api=True)
            html_source = BJS_HTML.render(
                css_files=resources.css_files,
                js_files=resources.js_files,
                bjs_script=js_source)
            return [html_source, "html"]
        else:
            return [js_source, "javascript"]

def set(self):
        """Set the color as current OpenGL color
        """
        glColor4f(self.r, self.g, self.b, self.a)

def version_triple(tag):
    """
    returns: a triple of integers from a version tag
    """
    groups = re.match(r'v?(\d+)\.(\d+)\.(\d+)', tag).groups()
    return tuple(int(n) for n in groups)

def get_file_md5sum(path):
    """Calculate the MD5 hash for a file."""
    with open(path, 'rb') as fh:
        h = str(hashlib.md5(fh.read()).hexdigest())
    return h

def strip_accents(text):
    """
    Strip agents from a string.
    """

    normalized_str = unicodedata.normalize('NFD', text)

    return ''.join([
        c for c in normalized_str if unicodedata.category(c) != 'Mn'])

def erase_lines(n=1):
    """ Erases n lines from the screen and moves the cursor up to follow
    """
    for _ in range(n):
        print(codes.cursor["up"], end="")
        print(codes.cursor["eol"], end="")

def _listify(collection):
        """This is a workaround where Collections are no longer iterable
        when using JPype."""
        new_list = []
        for index in range(len(collection)):
            new_list.append(collection[index])
        return new_list

def plot(self):
        """Plot the empirical histogram versus best-fit distribution's PDF."""
        plt.plot(self.bin_edges, self.hist, self.bin_edges, self.best_pdf)

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

def url_to_image(url):
    """
    Fetch an image from url and convert it into a Pillow Image object
    """
    r = requests.get(url)
    image = StringIO(r.content)
    return image

def file_matches(filename, patterns):
    """Does this filename match any of the patterns?"""
    return any(fnmatch.fnmatch(filename, pat) for pat in patterns)

def wget(url):
    """
    Download the page into a string
    """
    import urllib.parse
    request = urllib.request.urlopen(url)
    filestring = request.read()
    return filestring

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

def group_by(iterable, key_func):
    """Wrap itertools.groupby to make life easier."""
    groups = (
        list(sub) for key, sub in groupby(iterable, key_func)
    )
    return zip(groups, groups)

def append_pdf(input_pdf: bytes, output_writer: PdfFileWriter):
    """
    Appends a PDF to a pyPDF writer. Legacy interface.
    """
    append_memory_pdf_to_writer(input_pdf=input_pdf,
                                writer=output_writer)

def title(msg):
    """Sets the title of the console window."""
    if sys.platform.startswith("win"):
        ctypes.windll.kernel32.SetConsoleTitleW(tounicode(msg))

def _maybe_fill(arr, fill_value=np.nan):
    """
    if we have a compatible fill_value and arr dtype, then fill
    """
    if _isna_compat(arr, fill_value):
        arr.fill(fill_value)
    return arr

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

def filter_none(list_of_points):
    """
    
    :param list_of_points: 
    :return: list_of_points with None's removed
    """
    remove_elementnone = filter(lambda p: p is not None, list_of_points)
    remove_sublistnone = filter(lambda p: not contains_none(p), remove_elementnone)
    return list(remove_sublistnone)

def double_sha256(data):
    """A standard compound hash."""
    return bytes_as_revhex(hashlib.sha256(hashlib.sha256(data).digest()).digest())

def get_abi3_suffix():
    """Return the file extension for an abi3-compliant Extension()"""
    for suffix, _, _ in (s for s in imp.get_suffixes() if s[2] == imp.C_EXTENSION):
        if '.abi3' in suffix:  # Unix
            return suffix
        elif suffix == '.pyd':  # Windows
            return suffix

def np_hash(a):
    """Return a hash of a NumPy array."""
    if a is None:
        return hash(None)
    # Ensure that hashes are equal whatever the ordering in memory (C or
    # Fortran)
    a = np.ascontiguousarray(a)
    # Compute the digest and return a decimal int
    return int(hashlib.sha1(a.view(a.dtype)).hexdigest(), 16)

def out_shape_from_array(arr):
    """Get the output shape from an array."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.shape
    else:
        return (arr.shape[1],)

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

def findMax(arr):
    """
    in comparison to argrelmax() more simple and  reliable peak finder
    """
    out = np.zeros(shape=arr.shape, dtype=bool)
    _calcMax(arr, out)
    return out

def osx_clipboard_get():
    """ Get the clipboard's text on OS X.
    """
    p = subprocess.Popen(['pbpaste', '-Prefer', 'ascii'],
        stdout=subprocess.PIPE)
    text, stderr = p.communicate()
    # Text comes in with old Mac \r line endings. Change them to \n.
    text = text.replace('\r', '\n')
    return text

def median(data):
    """Calculate the median of a list."""
    data.sort()
    num_values = len(data)
    half = num_values // 2
    if num_values % 2:
        return data[half]
    return 0.5 * (data[half-1] + data[half])

def _mean_dict(dict_list):
    """Compute the mean value across a list of dictionaries
    """
    return {k: np.array([d[k] for d in dict_list]).mean()
            for k in dict_list[0].keys()}

def uniquify_list(L):
    """Same order unique list using only a list compression."""
    return [e for i, e in enumerate(L) if L.index(e) == i]

def distance_matrix(trains1, trains2, cos, tau):
    """
    Return the *bipartite* (rectangular) distance matrix between the observations in the first and the second list.

    Convenience function; equivalent to ``dissimilarity_matrix(trains1, trains2, cos, tau, "distance")``. Refer to :func:`pymuvr.dissimilarity_matrix` for full documentation.
    """
    return dissimilarity_matrix(trains1, trains2, cos, tau, "distance")

def fit_gaussian(x, y, yerr, p0):
    """ Fit a Gaussian to the data """
    try:
        popt, pcov = curve_fit(gaussian, x, y, sigma=yerr, p0=p0, absolute_sigma=True)
    except RuntimeError:
        return [0],[0]
    return popt, pcov

def save_partial(self, obj):
        """Partial objects do not serialize correctly in python2.x -- this fixes the bugs"""
        self.save_reduce(_genpartial, (obj.func, obj.args, obj.keywords))

def _none_value(self):
        """Get an appropriate "null" value for this field's type. This
        is used internally when setting the field to None.
        """
        if self.out_type == int:
            return 0
        elif self.out_type == float:
            return 0.0
        elif self.out_type == bool:
            return False
        elif self.out_type == six.text_type:
            return u''

def perform_permissions_check(self, user, obj, perms):
        """ Performs the permission check. """
        return self.request.forum_permission_handler.can_download_files(obj, user)

def safe_quotes(text, escape_single_quotes=False):
    """htmlify string"""
    if isinstance(text, str):
        safe_text = text.replace('"', "&quot;")
        if escape_single_quotes:
            safe_text = safe_text.replace("'", "&#92;'")
        return safe_text.replace('True', 'true')
    return text

def has_attribute(module_name, attribute_name):
    """Is this attribute present?"""
    init_file = '%s/__init__.py' % module_name
    return any(
        [attribute_name in init_line for init_line in open(init_file).readlines()]
    )

def serialize(self):
        """Serialize the query to a structure using the query DSL."""
        data = {'doc': self.doc}
        if isinstance(self.query, Query):
            data['query'] = self.query.serialize()
        return data

def we_are_in_lyon():
    """Check if we are on a Lyon machine"""
    import socket
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        return False
    return ip.startswith("134.158.")

def glr_path_static():
    """Returns path to packaged static files"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '_static'))

def check_if_branch_exist(db, root_hash, key_prefix):
    """
    Given a key prefix, return whether this prefix is
    the prefix of an existing key in the trie.
    """
    validate_is_bytes(key_prefix)

    return _check_if_branch_exist(db, root_hash, encode_to_bin(key_prefix))

def _sub_patterns(patterns, text):
    """
    Apply re.sub to bunch of (pattern, repl)
    """
    for pattern, repl in patterns:
        text = re.sub(pattern, repl, text)
    return text

def is_valid_image_extension(file_path):
    """is_valid_image_extension."""
    valid_extensions = ['.jpeg', '.jpg', '.gif', '.png']
    _, extension = os.path.splitext(file_path)
    return extension.lower() in valid_extensions

def min_values(args):
    """ Return possible range for min function. """
    return Interval(min(x.low for x in args), min(x.high for x in args))

def has_value_name(self, name):
        """Check if this `enum` has a particular name among its values.

        :param name: Enumeration value name
        :type name: str
        :rtype: True if there is an enumeration value with the given name
        """
        for val, _ in self._values:
            if val == name:
                return True
        return False

def best(self):
        """
        Returns the element with the highest probability.
        """
        b = (-1e999999, None)
        for k, c in iteritems(self.counts):
            b = max(b, (c, k))
        return b[1]

def is_callable(*p):
    """ True if all the args are functions and / or subroutines
    """
    import symbols
    return all(isinstance(x, symbols.FUNCTION) for x in p)

def ensure_dtype_float(x, default=np.float64):
    r"""Makes sure that x is type of float

    """
    if isinstance(x, np.ndarray):
        if x.dtype.kind == 'f':
            return x
        elif x.dtype.kind == 'i':
            return x.astype(default)
        else:
            raise TypeError('x is of type '+str(x.dtype)+' that cannot be converted to float')
    else:
        raise TypeError('x is not an array')

def get_file_extension_type(filename):
    """
    Return the group associated to the file
    :param filename:
    :return: str
    """
    ext = get_file_extension(filename)
    if ext:
        for name, group in EXTENSIONS.items():
            if ext in group:
                return name
    return "OTHER"

def append_pdf(input_pdf: bytes, output_writer: PdfFileWriter):
    """
    Appends a PDF to a pyPDF writer. Legacy interface.
    """
    append_memory_pdf_to_writer(input_pdf=input_pdf,
                                writer=output_writer)

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

def go_to_background():
    """ Daemonize the running process. """
    try:
        if os.fork():
            sys.exit()
    except OSError as errmsg:
        LOGGER.error('Fork failed: {0}'.format(errmsg))
        sys.exit('Fork failed')

def get_file_md5sum(path):
    """Calculate the MD5 hash for a file."""
    with open(path, 'rb') as fh:
        h = str(hashlib.md5(fh.read()).hexdigest())
    return h

def go_to_background():
    """ Daemonize the running process. """
    try:
        if os.fork():
            sys.exit()
    except OSError as errmsg:
        LOGGER.error('Fork failed: {0}'.format(errmsg))
        sys.exit('Fork failed')

def generate_uuid():
    """Generate a UUID."""
    r_uuid = base64.urlsafe_b64encode(uuid.uuid4().bytes)
    return r_uuid.decode().replace('=', '')

def to_json(data):
    """Return data as a JSON string."""
    return json.dumps(data, default=lambda x: x.__dict__, sort_keys=True, indent=4)

def check_color(cls, raw_image):
        """
        Just check if raw_image is completely white.
        http://stackoverflow.com/questions/14041562/python-pil-detect-if-an-image-is-completely-black-or-white
        """
        # sum(img.convert("L").getextrema()) in (0, 2)
        extrema = raw_image.convert("L").getextrema()
        if extrema == (255, 255): # all white
            raise cls.MonoImageException

def __str__(self):
    """Returns a pretty-printed string for this object."""
    return 'Output name: "%s" watts: %d type: "%s" id: %d' % (
        self._name, self._watts, self._output_type, self._integration_id)

def multi_replace(instr, search_list=[], repl_list=None):
    """
    Does a string replace with a list of search and replacements

    TODO: rename
    """
    repl_list = [''] * len(search_list) if repl_list is None else repl_list
    for ser, repl in zip(search_list, repl_list):
        instr = instr.replace(ser, repl)
    return instr

def safe_format(s, **kwargs):
  """
  :type s str
  """
  return string.Formatter().vformat(s, (), defaultdict(str, **kwargs))

def download_json(local_filename, url, clobber=False):
    """Download the given JSON file, and pretty-print before we output it."""
    with open(local_filename, 'w') as json_file:
        json_file.write(json.dumps(requests.get(url).json(), sort_keys=True, indent=2, separators=(',', ': ')))

def translate_fourier(image, dx):
    """ Translate an image in fourier-space with plane waves """
    N = image.shape[0]

    f = 2*np.pi*np.fft.fftfreq(N)
    kx,ky,kz = np.meshgrid(*(f,)*3, indexing='ij')
    kv = np.array([kx,ky,kz]).T

    q = np.fft.fftn(image)*np.exp(-1.j*(kv*dx).sum(axis=-1)).T
    return np.real(np.fft.ifftn(q))

def _pad(self):
    """Pads the output with an amount of indentation appropriate for the number of open element.

    This method does nothing if the indent value passed to the constructor is falsy.
    """
    if self._indent:
      self.whitespace(self._indent * len(self._open_elements))

def start(self):
        """
        Starts the loop. Calling a running loop is an error.
        """
        assert not self.has_started(), "called start() on an active GeventLoop"
        self._stop_event = Event()
        # note that we don't use safe_greenlets.spawn because we take care of it in _loop by ourselves
        self._greenlet = gevent.spawn(self._loop)

def give_str(self):
        """
            Give string representation of the callable.
        """
        args = self._args[:]
        kwargs = self._kwargs
        return self._give_str(args, kwargs)

def handle_whitespace(text):
    r"""Handles whitespace cleanup.

    Tabs are "smartly" retabbed (see sub_retab). Lines that contain
    only whitespace are truncated to a single newline.
    """
    text = re_retab.sub(sub_retab, text)
    text = re_whitespace.sub('', text).strip()
    return text

def gday_of_year(self):
        """Return the number of days since January 1 of the given year."""
        return (self.date - dt.date(self.date.year, 1, 1)).days

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

def is_password_valid(password):
    """
    Check if a password is valid
    """
    pattern = re.compile(r"^.{4,75}$")
    return bool(pattern.match(password))

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

def user_exists(username):
    """Check if a user exists"""
    try:
        pwd.getpwnam(username)
        user_exists = True
    except KeyError:
        user_exists = False
    return user_exists

def top_level(url, fix_protocol=True):
    """Extract the top level domain from an URL."""
    ext = tld.get_tld(url, fix_protocol=fix_protocol)
    toplevel = '.'.join(urlparse(url).netloc.split('.')[-2:]).split(
        ext)[0] + ext
    return toplevel

def GaussianBlur(X, ksize_width, ksize_height, sigma_x, sigma_y):
    """Apply Gaussian blur to the given data.

    Args:
        X: data to blur
        kernel_size: Gaussian kernel size
        stddev: Gaussian kernel standard deviation (in both X and Y directions)
    """
    return image_transform(
        X,
        cv2.GaussianBlur,
        ksize=(ksize_width, ksize_height),
        sigmaX=sigma_x,
        sigmaY=sigma_y
    )

def get_func_name(func):
    """Return a name which includes the module name and function name."""
    func_name = getattr(func, '__name__', func.__class__.__name__)
    module_name = func.__module__

    if module_name is not None:
        module_name = func.__module__
        return '{}.{}'.format(module_name, func_name)

    return func_name

def gauss_pdf(x, mu, sigma):
    """Normalized Gaussian"""
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - mu) ** 2 / 2. / sigma ** 2)

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

def gen_api_key(username):
    """
    Create a random API key for a user
    :param username:
    :return: Hex encoded SHA512 random string
    """
    salt = str(os.urandom(64)).encode('utf-8')
    return hash_password(username, salt)

def normalize(data):
    """Normalize the data to be in the [0, 1] range.

    :param data:
    :return: normalized data
    """
    out_data = data.copy()

    for i, sample in enumerate(out_data):
        out_data[i] /= sum(out_data[i])

    return out_data

def batch(items, size):
    """Batches a list into a list of lists, with sub-lists sized by a specified
    batch size."""
    return [items[x:x + size] for x in xrange(0, len(items), size)]

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

def generate_hash(self, length=30):
        """ Generate random string of given length """
        import random, string
        chars = string.ascii_letters + string.digits
        ran = random.SystemRandom().choice
        hash = ''.join(ran(chars) for i in range(length))
        return hash

def process_docstring(app, what, name, obj, options, lines):
    """React to a docstring event and append contracts to it."""
    # pylint: disable=unused-argument
    # pylint: disable=too-many-arguments
    lines.extend(_format_contracts(what=what, obj=obj))

def get_average_color(colors):
    """Calculate the average color from the list of colors, where each color
    is a 3-tuple of (r, g, b) values.
    """
    c = reduce(color_reducer, colors)
    total = len(colors)
    return tuple(v / total for v in c)

def chunk_list(l, n):
    """Return `n` size lists from a given list `l`"""
    return [l[i:i + n] for i in range(0, len(l), n)]

def get_numbers(s):
    """Extracts all integers from a string an return them in a list"""

    result = map(int, re.findall(r'[0-9]+', unicode(s)))
    return result + [1] * (2 - len(result))

def get_var(self, name):
        """ Returns the variable set with the given name.
        """
        for var in self.vars:
            if var.name == name:
                return var
        else:
            raise ValueError

def get_numbers(s):
    """Extracts all integers from a string an return them in a list"""

    result = map(int, re.findall(r'[0-9]+', unicode(s)))
    return result + [1] * (2 - len(result))

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

def caller_locals():
    """Get the local variables in the caller's frame."""
    import inspect
    frame = inspect.currentframe()
    try:
        return frame.f_back.f_back.f_locals
    finally:
        del frame

def clear_matplotlib_ticks(self, axis="both"):
        """Clears the default matplotlib ticks."""
        ax = self.get_axes()
        plotting.clear_matplotlib_ticks(ax=ax, axis=axis)

def get_single_item(d):
    """Get an item from a dict which contains just one item."""
    assert len(d) == 1, 'Single-item dict must have just one item, not %d.' % len(d)
    return next(six.iteritems(d))

def type_converter(text):
    """ I convert strings into integers, floats, and strings! """
    if text.isdigit():
        return int(text), int

    try:
        return float(text), float
    except ValueError:
        return text, STRING_TYPE

def find_console_handler(logger):
    """Return a stream handler, if it exists."""
    for handler in logger.handlers:
        if (isinstance(handler, logging.StreamHandler) and
                handler.stream == sys.stderr):
            return handler

def timeit(output):
    """
    If output is string, then print the string and also time used
    """
    b = time.time()
    yield
    print output, 'time used: %.3fs' % (time.time()-b)

def calc_list_average(l):
    """
    Calculates the average value of a list of numbers
    Returns a float
    """
    total = 0.0
    for value in l:
        total += value
    return total / len(l)

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

def get_last_modified_timestamp(self):
        """
        Looks at the files in a git root directory and grabs the last modified timestamp
        """
        cmd = "find . -print0 | xargs -0 stat -f '%T@ %p' | sort -n | tail -1 | cut -f2- -d' '"
        ps = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        output = ps.communicate()[0]
        print output

def get_base_dir():
    """
    Return the base directory
    """
    return os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]

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

def is_cached(file_name):
	"""
	Check if a given file is available in the cache or not
	"""

	gml_file_path = join(join(expanduser('~'), OCTOGRID_DIRECTORY), file_name)

	return isfile(gml_file_path)

def __repr__(self):
    """Returns a stringified representation of this object."""
    return str({'name': self._name, 'watts': self._watts,
                'type': self._output_type, 'id': self._integration_id})

def closest(xarr, val):
    """ Return the index of the closest in xarr to value val """
    idx_closest = np.argmin(np.abs(np.array(xarr) - val))
    return idx_closest

def format_exception(e):
    """Returns a string containing the type and text of the exception.

    """
    from .utils.printing import fill
    return '\n'.join(fill(line) for line in traceback.format_exception_only(type(e), e))

def get_obj_cols(df):
    """
    Returns names of 'object' columns in the DataFrame.
    """
    obj_cols = []
    for idx, dt in enumerate(df.dtypes):
        if dt == 'object' or is_category(dt):
            obj_cols.append(df.columns.values[idx])

    return obj_cols

def inpaint(self):
        """ Replace masked-out elements in an array using an iterative image inpainting algorithm. """

        import inpaint
        filled = inpaint.replace_nans(np.ma.filled(self.raster_data, np.NAN).astype(np.float32), 3, 0.01, 2)
        self.raster_data = np.ma.masked_invalid(filled)

def filter_contour(imageFile, opFile):
    """ convert an image by applying a contour """
    im = Image.open(imageFile)
    im1 = im.filter(ImageFilter.CONTOUR)
    im1.save(opFile)

def get_file_string(filepath):
    """Get string from file."""
    with open(os.path.abspath(filepath)) as f:
        return f.read()

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

def get_jsonparsed_data(url):
    """Receive the content of ``url``, parse it as JSON and return the
       object.
    """
    response = urlopen(url)
    data = response.read().decode('utf-8')
    return json.loads(data)

def get_week_start_end_day():
    """
    Get the week start date and end date
    """
    t = date.today()
    wd = t.weekday()
    return (t - timedelta(wd), t + timedelta(6 - wd))

def printdict(adict):
    """printdict"""
    dlist = list(adict.keys())
    dlist.sort()
    for i in range(0, len(dlist)):
        print(dlist[i], adict[dlist[i]])

def _get_session():
    """Return (and memoize) a database session"""
    session = getattr(g, '_session', None)
    if session is None:
        session = g._session = db.session()
    return session

def redirect_output(fileobj):
    """Redirect standard out to file."""
    old = sys.stdout
    sys.stdout = fileobj
    try:
        yield fileobj
    finally:
        sys.stdout = old

def _nth(arr, n):
    """
    Return the nth value of array

    If it is missing return NaN
    """
    try:
        return arr.iloc[n]
    except (KeyError, IndexError):
        return np.nan

def to_capitalized_camel_case(snake_case_string):
    """
    Convert a string from snake case to camel case with the first letter capitalized. For example, "some_var"
    would become "SomeVar".

    :param snake_case_string: Snake-cased string to convert to camel case.
    :returns: Camel-cased version of snake_case_string.
    """
    parts = snake_case_string.split('_')
    return ''.join([i.title() for i in parts])

def remove_ext(fname):
    """Removes the extension from a filename
    """
    bn = os.path.basename(fname)
    return os.path.splitext(bn)[0]

def previous_quarter(d):
    """
    Retrieve the previous quarter for dt
    """
    from django_toolkit.datetime_util import quarter as datetime_quarter
    return quarter( (datetime_quarter(datetime(d.year, d.month, d.day))[0] + timedelta(days=-1)).date() )

def _file_type(self, field):
        """ Returns file type for given file field.
        
        Args:
            field (str): File field

        Returns:
            string. File type
        """
        type = mimetypes.guess_type(self._files[field])[0]
        return type.encode("utf-8") if isinstance(type, unicode) else str(type)

def save_pdf(path):
  """
  Saves a pdf of the current matplotlib figure.

  :param path: str, filepath to save to
  """

  pp = PdfPages(path)
  pp.savefig(pyplot.gcf())
  pp.close()

def get_file_name(url):
  """Returns file name of file at given url."""
  return os.path.basename(urllib.parse.urlparse(url).path) or 'unknown_name'

def download_file(save_path, file_url):
    """ Download file from http url link """

    r = requests.get(file_url)  # create HTTP response object

    with open(save_path, 'wb') as f:
        f.write(r.content)

    return save_path

def get_X0(X):
    """ Return zero-th element of a one-element data container.
    """
    if pandas_available and isinstance(X, pd.DataFrame):
        assert len(X) == 1
        x = np.array(X.iloc[0])
    else:
        x, = X
    return x

def get_tablenames(cur):
    """ Conveinience: """
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tablename_list_ = cur.fetchall()
    tablename_list = [str(tablename[0]) for tablename in tablename_list_ ]
    return tablename_list

def find_le(a, x):
    """Find rightmost value less than or equal to x."""
    i = bs.bisect_right(a, x)
    if i: return i - 1
    raise ValueError

def enable_proxy(self, host, port):
        """Enable a default web proxy"""

        self.proxy = [host, _number(port)]
        self.proxy_enabled = True

def __getitem__(self, index):
    """Get the item at the given index.

    Index is a tuple of (row, col)
    """
    row, col = index
    return self.rows[row][col]

def onkeyup(self, key, keycode, ctrl, shift, alt):
        """Called when user types and releases a key. 
        The widget should be able to receive the focus in order to emit the event.
        Assign a 'tabindex' attribute to make it focusable.
        
        Args:
            key (str): the character value
            keycode (str): the numeric char code
        """
        return (key, keycode, ctrl, shift, alt)

def on_train_end(self, logs):
        """ Print training time at end of training """
        duration = timeit.default_timer() - self.train_start
        print('done, took {:.3f} seconds'.format(duration))

def extract_keywords_from_text(self, text):
        """Method to extract keywords from the text provided.

        :param text: Text to extract keywords from, provided as a string.
        """
        sentences = nltk.tokenize.sent_tokenize(text)
        self.extract_keywords_from_sentences(sentences)

def pause():
	"""Tell iTunes to pause"""

	if not settings.platformCompatible():
		return False

	(output, error) = subprocess.Popen(["osascript", "-e", PAUSE], stdout=subprocess.PIPE).communicate()

def postprocessor(prediction):
    """Map prediction tensor to labels."""
    prediction = prediction.data.numpy()[0]
    top_predictions = prediction.argsort()[-3:][::-1]
    return [labels[prediction] for prediction in top_predictions]

def insort_no_dup(lst, item):
    """
    If item is not in lst, add item to list at its sorted position
    """
    import bisect
    ix = bisect.bisect_left(lst, item)
    if lst[ix] != item: 
        lst[ix:ix] = [item]

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

def stop(self):
        """Stop the progress bar."""
        if self._progressing:
            self._progressing = False
            self._thread.join()

def size_on_disk(self):
        """
        :return: size of the entire schema in bytes
        """
        return int(self.connection.query(
            """
            SELECT SUM(data_length + index_length)
            FROM information_schema.tables WHERE table_schema='{db}'
            """.format(db=self.database)).fetchone()[0])

def yvals(self):
        """All y values"""
        return [
            val[1] for serie in self.series for val in serie.values
            if val[1] is not None
        ]

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

def list_all(dev: Device):
    """List all available API calls."""
    for name, service in dev.services.items():
        click.echo(click.style("\nService %s" % name, bold=True))
        for method in service.methods:
            click.echo("  %s" % method.name)

def _rectangular(n):
    """Checks to see if a 2D list is a valid 2D matrix"""
    for i in n:
        if len(i) != len(n[0]):
            return False
    return True

def _get_loggers():
    """Return list of Logger classes."""
    from .. import loader
    modules = loader.get_package_modules('logger')
    return list(loader.get_plugins(modules, [_Logger]))

def clean(some_string, uppercase=False):
    """
    helper to clean up an input string
    """
    if uppercase:
        return some_string.strip().upper()
    else:
        return some_string.strip().lower()

def best(self):
        """
        Returns the element with the highest probability.
        """
        b = (-1e999999, None)
        for k, c in iteritems(self.counts):
            b = max(b, (c, k))
        return b[1]

def add_arrow(self, x1, y1, x2, y2, **kws):
        """add arrow to plot"""
        self.panel.add_arrow(x1, y1, x2, y2, **kws)

def timestamp_to_microseconds(timestamp):
    """Convert a timestamp string into a microseconds value
    :param timestamp
    :return time in microseconds
    """
    timestamp_str = datetime.datetime.strptime(timestamp, ISO_DATETIME_REGEX)
    epoch_time_secs = calendar.timegm(timestamp_str.timetuple())
    epoch_time_mus = epoch_time_secs * 1e6 + timestamp_str.microsecond
    return epoch_time_mus

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

def yesno(prompt):
    """Returns True if user answers 'y' """
    prompt += " [y/n]"
    a = ""
    while a not in ["y", "n"]:
        a = input(prompt).lower()

    return a == "y"

def _nth(arr, n):
    """
    Return the nth value of array

    If it is missing return NaN
    """
    try:
        return arr.iloc[n]
    except (KeyError, IndexError):
        return np.nan

def _prepare_proxy(self, conn):
        """
        Establish tunnel connection early, because otherwise httplib
        would improperly set Host: header to proxy's IP:port.
        """
        conn.set_tunnel(self._proxy_host, self.port, self.proxy_headers)
        conn.connect()

def get_parent_folder_name(file_path):
    """Finds parent folder of file

    :param file_path: path
    :return: Name of folder container
    """
    return os.path.split(os.path.split(os.path.abspath(file_path))[0])[-1]

def sanitize_word(s):
    """Remove non-alphanumerical characters from metric word.
    And trim excessive underscores.
    """
    s = re.sub('[^\w-]+', '_', s)
    s = re.sub('__+', '_', s)
    return s.strip('_')

def last_day(year=_year, month=_month):
    """
    get the current month's last day
    :param year:  default to current year
    :param month:  default to current month
    :return: month's last day
    """
    last_day = calendar.monthrange(year, month)[1]
    return datetime.date(year=year, month=month, day=last_day)

def is_hex_string(string):
    """Check if the string is only composed of hex characters."""
    pattern = re.compile(r'[A-Fa-f0-9]+')
    if isinstance(string, six.binary_type):
        string = str(string)
    return pattern.match(string) is not None

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

def _dt_to_epoch(dt):
        """Convert datetime to epoch seconds."""
        try:
            epoch = dt.timestamp()
        except AttributeError:  # py2
            epoch = (dt - datetime(1970, 1, 1)).total_seconds()
        return epoch

def find_last_sublist(list_, sublist):
    """Given a list, find the last occurance of a sublist within it.

    Returns:
        Index where the sublist starts, or None if there is no match.
    """
    for i in reversed(range(len(list_) - len(sublist) + 1)):
        if list_[i] == sublist[0] and list_[i:i + len(sublist)] == sublist:
            return i
    return None

def bytesize(arr):
    """
    Returns the memory byte size of a Numpy array as an integer.
    """
    byte_size = np.prod(arr.shape) * np.dtype(arr.dtype).itemsize
    return byte_size

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

def is_integer(obj):
    """Is this an integer.

    :param object obj:
    :return:
    """
    if PYTHON3:
        return isinstance(obj, int)
    return isinstance(obj, (int, long))

def is_numeric_dtype(dtype):
    """Return ``True`` if ``dtype`` is a numeric type."""
    dtype = np.dtype(dtype)
    return np.issubsctype(getattr(dtype, 'base', None), np.number)

def previous_quarter(d):
    """
    Retrieve the previous quarter for dt
    """
    from django_toolkit.datetime_util import quarter as datetime_quarter
    return quarter( (datetime_quarter(datetime(d.year, d.month, d.day))[0] + timedelta(days=-1)).date() )

def is_date_type(cls):
    """Return True if the class is a date type."""
    if not isinstance(cls, type):
        return False
    return issubclass(cls, date) and not issubclass(cls, datetime)

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

def _to_numeric(val):
    """
    Helper function for conversion of various data types into numeric representation.
    """
    if isinstance(val, (int, float, datetime.datetime, datetime.timedelta)):
        return val
    return float(val)

def get_table_columns(dbconn, tablename):
    """
    Return a list of tuples specifying the column name and type
    """
    cur = dbconn.cursor()
    cur.execute("PRAGMA table_info('%s');" % tablename)
    info = cur.fetchall()
    cols = [(i[1], i[2]) for i in info]
    return cols

def readline( file, skip_blank=False ):
    """Read a line from provided file, skipping any blank or comment lines"""
    while 1:
        line = file.readline()
        #print "every line: %r" % line
        if not line: return None 
        if line[0] != '#' and not ( skip_blank and line.isspace() ):
            return line

def last(self):
        """Get the last object in file."""
        # End of file
        self.__file.seek(0, 2)

        # Get the last struct
        data = self.get(self.length - 1)

        return data

def crop_box(im, box=False, **kwargs):
    """Uses box coordinates to crop an image without resizing it first."""
    if box:
        im = im.crop(box)
    return im

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

def border(self):
        """Region formed by taking border elements.

        :returns: :class:`jicimagelib.region.Region`
        """

        border_array = self.bitmap - self.inner.bitmap
        return Region(border_array)

def _visual_width(line):
    """Get the the number of columns required to display a string"""

    return len(re.sub(colorama.ansitowin32.AnsiToWin32.ANSI_CSI_RE, "", line))

def pointer(self):
        """Get a ctypes void pointer to the memory mapped region.

        :type: ctypes.c_void_p
        """
        return ctypes.cast(ctypes.pointer(ctypes.c_uint8.from_buffer(self.mapping, 0)), ctypes.c_void_p)

def display_pil_image(im):
   """Displayhook function for PIL Images, rendered as PNG."""
   from IPython.core import display
   b = BytesIO()
   im.save(b, format='png')
   data = b.getvalue()

   ip_img = display.Image(data=data, format='png', embed=True)
   return ip_img._repr_png_()

def getWindowPID(self, hwnd):
        """ Gets the process ID that the specified window belongs to """
        pid = ctypes.c_ulong()
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return int(pid.value)

def set_font_size(self, size):
        """Convenience method for just changing font size."""
        if self.font.font_size == size:
            pass
        else:
            self.font._set_size(size)

def predecessors(self, node, graph=None):
        """ Returns a list of all predecessors of the given node """
        if graph is None:
            graph = self.graph
        return [key for key in graph if node in graph[key]]

def indent(self):
        """
        Begins an indented block. Must be used in a 'with' code block.
        All calls to the logger inside of the block will be indented.
        """
        blk = IndentBlock(self, self._indent)
        self._indent += 1
        return blk

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

def closest(xarr, val):
    """ Return the index of the closest in xarr to value val """
    idx_closest = np.argmin(np.abs(np.array(xarr) - val))
    return idx_closest

def hstrlen(self, name, key):
        """
        Return the number of bytes stored in the value of ``key``
        within hash ``name``
        """
        with self.pipe as pipe:
            return pipe.hstrlen(self.redis_key(name), key)

def _index_range(self, version, symbol, from_version=None, **kwargs):
        """
        Tuple describing range to read from the ndarray - closed:open
        """
        from_index = None
        if from_version:
            from_index = from_version['up_to']
        return from_index, None

def upload_file(token, channel_name, file_name):
    """ upload file to a channel """

    slack = Slacker(token)

    slack.files.upload(file_name, channels=channel_name)

def get_lines(handle, line):
    """
    Get zero-indexed line from an open file-like.
    """
    for i, l in enumerate(handle):
        if i == line:
            return l

async def async_input(prompt):
    """
    Python's ``input()`` is blocking, which means the event loop we set
    above can't be running while we're blocking there. This method will
    let the loop run while we wait for input.
    """
    print(prompt, end='', flush=True)
    return (await loop.run_in_executor(None, sys.stdin.readline)).rstrip()

def loads(cls, s):
        """
        Load an instance of this class from YAML.

        """
        with closing(StringIO(s)) as fileobj:
            return cls.load(fileobj)

def heappush_max(heap, item):
    """Push item onto heap, maintaining the heap invariant."""
    heap.append(item)
    _siftdown_max(heap, 0, len(heap) - 1)

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

async def async_input(prompt):
    """
    Python's ``input()`` is blocking, which means the event loop we set
    above can't be running while we're blocking there. This method will
    let the loop run while we wait for input.
    """
    print(prompt, end='', flush=True)
    return (await loop.run_in_executor(None, sys.stdin.readline)).rstrip()

def dict_hash(dct):
    """Return a hash of the contents of a dictionary"""
    dct_s = json.dumps(dct, sort_keys=True)

    try:
        m = md5(dct_s)
    except TypeError:
        m = md5(dct_s.encode())

    return m.hexdigest()

def init_checks_registry():
    """Register all globally visible functions.

    The first argument name is either 'physical_line' or 'logical_line'.
    """
    mod = inspect.getmodule(register_check)
    for (name, function) in inspect.getmembers(mod, inspect.isfunction):
        register_check(function)

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

def _module_name_from_previous_frame(num_frames_back):
    """
    Returns the module name associated with a frame `num_frames_back` in the
    call stack. This function adds 1 to account for itself, so `num_frames_back`
    should be given relative to the caller.
    """
    frm = inspect.stack()[num_frames_back + 1]
    return inspect.getmodule(frm[0]).__name__

def _multilingual(function, *args, **kwargs):
    """ Returns the value from the function with the given name in the given language module.
        By default, language="en".
    """
    return getattr(_module(kwargs.pop("language", "en")), function)(*args, **kwargs)

def infer_dtype_from(val, pandas_dtype=False):
    """
    interpret the dtype from a scalar or array. This is a convenience
    routines to infer dtype from a scalar or an array

    Parameters
    ----------
    pandas_dtype : bool, default False
        whether to infer dtype including pandas extension types.
        If False, scalar/array belongs to pandas extension types is inferred as
        object
    """
    if is_scalar(val):
        return infer_dtype_from_scalar(val, pandas_dtype=pandas_dtype)
    return infer_dtype_from_array(val, pandas_dtype=pandas_dtype)

def hex_escape(bin_str):
  """
  Hex encode a binary string
  """
  printable = string.ascii_letters + string.digits + string.punctuation + ' '
  return ''.join(ch if ch in printable else r'0x{0:02x}'.format(ord(ch)) for ch in bin_str)

def safe_int(val, default=None):
    """
    Returns int() of val if val is not convertable to int use default
    instead

    :param val:
    :param default:
    """

    try:
        val = int(val)
    except (ValueError, TypeError):
        val = default

    return val

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

def py3round(number):
    """Unified rounding in all python versions."""
    if abs(round(number) - number) == 0.5:
        return int(2.0 * round(number / 2.0))

    return int(round(number))

def get_file_size(filename):
    """
    Get the file size of a given file

    :param filename: string: pathname of a file
    :return: human readable filesize
    """
    if os.path.isfile(filename):
        return convert_size(os.path.getsize(filename))
    return None

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

def magnitude(X):
    """Magnitude of a complex matrix."""
    r = np.real(X)
    i = np.imag(X)
    return np.sqrt(r * r + i * i);

def ip_address_list(ips):
    """ IP address range validation and expansion. """
    # first, try it as a single IP address
    try:
        return ip_address(ips)
    except ValueError:
        pass
    # then, consider it as an ipaddress.IPv[4|6]Network instance and expand it
    return list(ipaddress.ip_network(u(ips)).hosts())

def write_file(filename, content):
    """Create the file with the given content"""
    print 'Generating {0}'.format(filename)
    with open(filename, 'wb') as out_f:
        out_f.write(content)

def get_truetype(value):
    """Convert a string to a pythonized parameter."""
    if value in ["true", "True", "y", "Y", "yes"]:
        return True
    if value in ["false", "False", "n", "N", "no"]:
        return False
    if value.isdigit():
        return int(value)
    return str(value)

def get_previous_month(self):
        """Returns date range for the previous full month."""
        end = utils.get_month_start() - relativedelta(days=1)
        end = utils.to_datetime(end)
        start = utils.get_month_start(end)
        return start, end

def _not_none(items):
    """Whether the item is a placeholder or contains a placeholder."""
    if not isinstance(items, (tuple, list)):
        items = (items,)
    return all(item is not _none for item in items)

def isPackage(file_path):
    """
    Determine whether or not a given path is a (sub)package or not.
    """
    return (os.path.isdir(file_path) and
            os.path.isfile(os.path.join(file_path, '__init__.py')))

def poke_array(self, store, name, elemtype, elements, container, visited, _stack):
        """abstract method"""
        raise NotImplementedError

def read_stdin():
    """ Read text from stdin, and print a helpful message for ttys. """
    if sys.stdin.isatty() and sys.stdout.isatty():
        print('\nReading from stdin until end of file (Ctrl + D)...')

    return sys.stdin.read()

def fixed(ctx, number, decimals=2, no_commas=False):
    """
    Formats the given number in decimal format using a period and commas
    """
    value = _round(ctx, number, decimals)
    format_str = '{:f}' if no_commas else '{:,f}'
    return format_str.format(value)

def __sub__(self, other):
		"""
		Return a Cache containing the entries of self that are not in other.
		"""
		return self.__class__([elem for elem in self if elem not in other])

def add_ul(text, ul):
    """Adds an unordered list to the readme"""
    text += "\n"
    for li in ul:
        text += "- " + li + "\n"
    text += "\n"

    return text

def find_all(self, string, callback):
		"""
		Wrapper on iter method, callback gets an iterator result
		"""
		for index, output in self.iter(string):
			callback(index, output)

def iget_list_column_slice(list_, start=None, stop=None, stride=None):
    """ iterator version of get_list_column """
    if isinstance(start, slice):
        slice_ = start
    else:
        slice_ = slice(start, stop, stride)
    return (row[slice_] for row in list_)

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

def __reversed__(self):
        """
        Return a reversed iterable over the items in the dictionary. Items are
        iterated over in their reverse sort order.

        Iterating views while adding or deleting entries in the dictionary may
        raise a RuntimeError or fail to iterate over all entries.
        """
        _dict = self._dict
        return iter((key, _dict[key]) for key in reversed(self._list))

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

def index(self, elem):
        """Find the index of elem in the reversed iterator."""
        return _coconut.len(self._iter) - self._iter.index(elem) - 1

def _openResources(self):
        """ Uses numpy.load to open the underlying file
        """
        arr = np.load(self._fileName, allow_pickle=ALLOW_PICKLE)
        check_is_an_array(arr)
        self._array = arr

def _skip_frame(self):
        """Skip a single frame from the trajectory"""
        size = self.read_size()
        for i in range(size+1):
            line = self._f.readline()
            if len(line) == 0:
                raise StopIteration

def get_hline():
    """ gets a horiztonal line """
    return Window(
        width=LayoutDimension.exact(1),
        height=LayoutDimension.exact(1),
        content=FillControl('-', token=Token.Line))

def txt_line_iterator(path):
  """Iterate through lines of file."""
  with tf.gfile.Open(path) as f:
    for line in f:
      yield line.strip()

def unique(list):
    """ Returns a copy of the list without duplicates.
    """
    unique = []; [unique.append(x) for x in list if x not in unique]
    return unique

def load(file_object):
  """
  Deserializes Java primitive data and objects serialized by ObjectOutputStream
  from a file-like object.
  """
  marshaller = JavaObjectUnmarshaller(file_object)
  marshaller.add_transformer(DefaultObjectTransformer())
  return marshaller.readObject()

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

def enable_proxy(self, host, port):
        """Enable a default web proxy"""

        self.proxy = [host, _number(port)]
        self.proxy_enabled = True

def json_dumps(self, obj):
        """Serializer for consistency"""
        return json.dumps(obj, sort_keys=True, indent=4, separators=(',', ': '))

def java_version():
    """Call java and return version information.

    :return unicode: Java version string
    """
    result = subprocess.check_output(
        [c.JAVA, '-version'], stderr=subprocess.STDOUT
    )
    first_line = result.splitlines()[0]
    return first_line.decode()

def json_get_data(filename):
    """Get data from json file
    """
    with open(filename) as fp:
        json_data = json.load(fp)
        return json_data

    return False

def respond_redirect(self, location='/'):
		"""
		Respond to the client with a 301 message and redirect them with
		a Location header.

		:param str location: The new location to redirect the client to.
		"""
		self.send_response(301)
		self.send_header('Content-Length', 0)
		self.send_header('Location', location)
		self.end_headers()
		return

def _timestamp_to_json_row(value):
    """Coerce 'value' to an JSON-compatible representation.

    This version returns floating-point seconds value used in row data.
    """
    if isinstance(value, datetime.datetime):
        value = _microseconds_from_datetime(value) * 1e-6
    return value

def _screen(self, s, newline=False):
        """Print something on screen when self.verbose == True"""
        if self.verbose:
            if newline:
                print(s)
            else:
                print(s, end=' ')

def chmod_add_excute(filename):
        """
        Adds execute permission to file.
        :param filename:
        :return:
        """
        st = os.stat(filename)
        os.chmod(filename, st.st_mode | stat.S_IEXEC)

def get_last_row(dbconn, tablename, n=1, uuid=None):
    """
    Returns the last `n` rows in the table
    """
    return fetch(dbconn, tablename, n, uuid, end=True)

def save_keras_definition(keras_model, path):
    """
    Save a Keras model definition to JSON with given path
    """
    model_json = keras_model.to_json()
    with open(path, "w") as json_file:
        json_file.write(model_json)

def fields(self):
        """Returns the list of field names of the model."""
        return (self.attributes.values() + self.lists.values()
                + self.references.values())

def param (self, param, kwargs, default_value=False):
        """gets a param from kwargs, or uses a default_value. if found, it's
        removed from kwargs"""
        if param in kwargs:
            value= kwargs[param]
            del kwargs[param]
        else:
            value= default_value
        setattr (self, param, value)

def Proxy(f):
  """A helper to create a proxy method in a class."""

  def Wrapped(self, *args):
    return getattr(self, f)(*args)

  return Wrapped

def l2_norm(arr):
    """
    The l2 norm of an array is is defined as: sqrt(||x||), where ||x|| is the
    dot product of the vector.
    """
    arr = np.asarray(arr)
    return np.sqrt(np.dot(arr.ravel().squeeze(), arr.ravel().squeeze()))

def SegmentMin(a, ids):
    """
    Segmented min op.
    """
    func = lambda idxs: np.amin(a[idxs], axis=0)
    return seg_map(func, a, ids),

def l2_norm(arr):
    """
    The l2 norm of an array is is defined as: sqrt(||x||), where ||x|| is the
    dot product of the vector.
    """
    arr = np.asarray(arr)
    return np.sqrt(np.dot(arr.ravel().squeeze(), arr.ravel().squeeze()))

def prefix_list(self, prefix, values):
        """
        Add a prefix to a list of values.
        """
        return list(map(lambda value: prefix + " " + value, values))

def make_lambda(call):
    """Wrap an AST Call node to lambda expression node.
    call: ast.Call node
    """
    empty_args = ast.arguments(args=[], vararg=None, kwarg=None, defaults=[])
    return ast.Lambda(args=empty_args, body=call)

def add_suffix(fullname, suffix):
    """ Add suffix to a full file name"""
    name, ext = os.path.splitext(fullname)
    return name + '_' + suffix + ext

def _sort_lambda(sortedby='cpu_percent',
                 sortedby_secondary='memory_percent'):
    """Return a sort lambda function for the sortedbykey"""
    ret = None
    if sortedby == 'io_counters':
        ret = _sort_io_counters
    elif sortedby == 'cpu_times':
        ret = _sort_cpu_times
    return ret

def put(self, entity):
    """Registers entity to put to datastore.

    Args:
      entity: an entity or model instance to put.
    """
    actual_entity = _normalize_entity(entity)
    if actual_entity is None:
      return self.ndb_put(entity)
    self.puts.append(actual_entity)

def make_lambda(call):
    """Wrap an AST Call node to lambda expression node.
    call: ast.Call node
    """
    empty_args = ast.arguments(args=[], vararg=None, kwarg=None, defaults=[])
    return ast.Lambda(args=empty_args, body=call)

def _elapsed(self):
        """ Returns elapsed time at update. """
        self.last_time = time.time()
        return self.last_time - self.start

def get_table_width(table):
    """
    Gets the width of the table that would be printed.
    :rtype: ``int``
    """
    columns = transpose_table(prepare_rows(table))
    widths = [max(len(cell) for cell in column) for column in columns]
    return len('+' + '|'.join('-' * (w + 2) for w in widths) + '+')

def __add__(self,other):
        """
            If the number of columns matches, we can concatenate two LabeldMatrices
            with the + operator.
        """
        assert self.matrix.shape[1] == other.matrix.shape[1]
        return LabeledMatrix(np.concatenate([self.matrix,other.matrix],axis=0),self.labels)

def levenshtein_distance_metric(a, b):
    """ 1 - farthest apart (same number of words, all diff). 0 - same"""
    return (levenshtein_distance(a, b) / (2.0 * max(len(a), len(b), 1)))

def indent(txt, spacing=4):
    """
    Indent given text using custom spacing, default is set to 4.
    """
    return prefix(str(txt), ''.join([' ' for _ in range(spacing)]))

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

def _listify(collection):
        """This is a workaround where Collections are no longer iterable
        when using JPype."""
        new_list = []
        for index in range(len(collection)):
            new_list.append(collection[index])
        return new_list

def to_bipartite_matrix(A):
    """Returns the adjacency matrix of a bipartite graph whose biadjacency
    matrix is `A`.

    `A` must be a NumPy array.

    If `A` has **m** rows and **n** columns, then the returned matrix has **m +
    n** rows and columns.

    """
    m, n = A.shape
    return four_blocks(zeros(m, m), A, A.T, zeros(n, n))

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

def _distance(coord1, coord2):
    """
    Return the distance between two points, `coord1` and `coord2`. These
    parameters are assumed to be (x, y) tuples.
    """
    xdist = coord1[0] - coord2[0]
    ydist = coord1[1] - coord2[1]
    return sqrt(xdist*xdist + ydist*ydist)

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

def floor(self):
    """Round `x` and `y` down to integers."""
    return Point(int(math.floor(self.x)), int(math.floor(self.y)))

def get_inputs_from_cm(index, cm):
    """Return indices of inputs to the node with the given index."""
    return tuple(i for i in range(cm.shape[0]) if cm[i][index])

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

def vsh(cmd, *args, **kw):
    """ Execute a command installed into the active virtualenv.
    """
    args = '" "'.join(i.replace('"', r'\"') for i in args)
    easy.sh('"%s" "%s"' % (venv_bin(cmd), args))

def _top(self):
        """ g """
        # Goto top of the list
        self.top.body.focus_position = 2 if self.compact is False else 0
        self.top.keypress(self.size, "")

def _help():
    """ Display both SQLAlchemy and Python help statements """

    statement = '%s%s' % (shelp, phelp % ', '.join(cntx_.keys()))
    print statement.strip()

def index_nearest(value, array):
    """
    expects a _n.array
    returns the global minimum of (value-array)^2
    """

    a = (array-value)**2
    return index(a.min(), a)

def safe_int_conv(number):
    """Safely convert a single number to integer."""
    try:
        return int(np.array(number).astype(int, casting='safe'))
    except TypeError:
        raise ValueError('cannot safely convert {} to integer'.format(number))

def list_of_lists_to_dict(l):
    """ Convert list of key,value lists to dict

    [['id', 1], ['id', 2], ['id', 3], ['foo': 4]]
    {'id': [1, 2, 3], 'foo': [4]}
    """
    d = {}
    for key, val in l:
        d.setdefault(key, []).append(val)
    return d

def get_number(s, cast=int):
    """
    Try to get a number out of a string, and cast it.
    """
    import string
    d = "".join(x for x in str(s) if x in string.digits)
    return cast(d)

def format_result(input):
        """From: http://stackoverflow.com/questions/13062300/convert-a-dict-to-sorted-dict-in-python
        """
        items = list(iteritems(input))
        return OrderedDict(sorted(items, key=lambda x: x[0]))

def update_screen(self):
        """Refresh the screen. You don't need to override this except to update only small portins of the screen."""
        self.clock.tick(self.FPS)
        pygame.display.update()

def list_to_csv(value):
    """
    Converts list to string with comma separated values. For string is no-op.
    """
    if isinstance(value, (list, tuple, set)):
        value = ",".join(value)
    return value

def str_to_num(str_value):
        """Convert str_value to an int or a float, depending on the
        numeric value represented by str_value.

        """
        str_value = str(str_value)
        try:
            return int(str_value)
        except ValueError:
            return float(str_value)

def items(self):
    """Return a list of the (name, value) pairs of the enum.

    These are returned in the order they were defined in the .proto file.
    """
    return [(value_descriptor.name, value_descriptor.number)
            for value_descriptor in self._enum_type.values]

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

def get_tablenames(cur):
    """ Conveinience: """
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tablename_list_ = cur.fetchall()
    tablename_list = [str(tablename[0]) for tablename in tablename_list_ ]
    return tablename_list

def to_snake_case(name):
    """ Given a name in camelCase return in snake_case """
    s1 = FIRST_CAP_REGEX.sub(r'\1_\2', name)
    return ALL_CAP_REGEX.sub(r'\1_\2', s1).lower()

def decode_example(self, example):
    """Reconstruct the image from the tf example."""
    img = tf.image.decode_image(
        example, channels=self._shape[-1], dtype=tf.uint8)
    img.set_shape(self._shape)
    return img

def stringc(text, color):
    """
    Return a string with terminal colors.
    """
    if has_colors:
        text = str(text)

        return "\033["+codeCodes[color]+"m"+text+"\033[0m"
    else:
        return text

def loads(s, model=None, parser=None):
    """Deserialize s (a str) to a Python object."""
    with StringIO(s) as f:
        return load(f, model=model, parser=parser)

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

def info(self, text):
		""" Ajout d'un message de log de type INFO """
		self.logger.info("{}{}".format(self.message_prefix, text))

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

def logger(message, level=10):
    """Handle logging."""
    logging.getLogger(__name__).log(level, str(message))

def lower_ext(abspath):
    """Convert file extension to lowercase.
    """
    fname, ext = os.path.splitext(abspath)
    return fname + ext.lower()

def write(self, text):
        """Write text. An additional attribute terminator with a value of
           None is added to the logging record to indicate that StreamHandler
           should not add a newline."""
        self.logger.log(self.loglevel, text, extra={'terminator': None})

def _maybe_cast_to_float64(da):
    """Cast DataArrays to np.float64 if they are of type np.float32.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray

    Returns
    -------
    DataArray
    """
    if da.dtype == np.float32:
        logging.warning('Datapoints were stored using the np.float32 datatype.'
                        'For accurate reduction operations using bottleneck, '
                        'datapoints are being cast to the np.float64 datatype.'
                        ' For more information see: https://github.com/pydata/'
                        'xarray/issues/1346')
        return da.astype(np.float64)
    else:
        return da

def set_executable(filename):
    """Set the exectuable bit on the given filename"""
    st = os.stat(filename)
    os.chmod(filename, st.st_mode | stat.S_IEXEC)

def dict_self(self):
        """Return the self object attributes not inherited as dict."""
        return {k: v for k, v in self.__dict__.items() if k in FSM_ATTRS}

def writefile(openedfile, newcontents):
    """Set the contents of a file."""
    openedfile.seek(0)
    openedfile.truncate()
    openedfile.write(newcontents)

def to_snake_case(text):
    """Convert to snake case.

    :param str text:
    :rtype: str
    :return:
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def decamelise(text):
    """Convert CamelCase to lower_and_underscore."""
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()

def _extract_node_text(node):
    """Extract text from a given lxml node."""

    texts = map(
        six.text_type.strip, map(six.text_type, map(unescape, node.xpath(".//text()")))
    )
    return " ".join(text for text in texts if text)

def _check_elements_equal(lst):
    """
    Returns true if all of the elements in the list are equal.
    """
    assert isinstance(lst, list), "Input value must be a list."
    return not lst or lst.count(lst[0]) == len(lst)

def paste(cmd=paste_cmd, stdout=PIPE):
    """Returns system clipboard contents.
    """
    return Popen(cmd, stdout=stdout).communicate()[0].decode('utf-8')

def autoconvert(string):
    """Try to convert variables into datatypes."""
    for fn in (boolify, int, float):
        try:
            return fn(string)
        except ValueError:
            pass
    return string

def closeEvent(self, event):
        """ Called when closing this window.
        """
        logger.debug("closeEvent")
        self.argosApplication.saveSettingsIfNeeded()
        self.finalize()
        self.argosApplication.removeMainWindow(self)
        event.accept()
        logger.debug("closeEvent accepted")

def is_identifier(string):
    """Check if string could be a valid python identifier

    :param string: string to be tested
    :returns: True if string can be a python identifier, False otherwise
    :rtype: bool
    """
    matched = PYTHON_IDENTIFIER_RE.match(string)
    return bool(matched) and not keyword.iskeyword(string)

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

def make_symmetric(dict):
    """Makes the given dictionary symmetric. Values are assumed to be unique."""
    for key, value in list(dict.items()):
        dict[value] = key
    return dict

def is_valid_file(parser, arg):
    """Check if arg is a valid file that already exists on the file system."""
    arg = os.path.abspath(arg)
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg

def split_elements(value):
    """Split a string with comma or space-separated elements into a list."""
    l = [v.strip() for v in value.split(',')]
    if len(l) == 1:
        l = value.split()
    return l

def task_property_present_predicate(service, task, prop):
    """ True if the json_element passed is present for the task specified.
    """
    try:
        response = get_service_task(service, task)
    except Exception as e:
        pass

    return (response is not None) and (prop in response)

def list_of_lists_to_dict(l):
    """ Convert list of key,value lists to dict

    [['id', 1], ['id', 2], ['id', 3], ['foo': 4]]
    {'id': [1, 2, 3], 'foo': [4]}
    """
    d = {}
    for key, val in l:
        d.setdefault(key, []).append(val)
    return d

def _replace_file(path, content):
  """Writes a file if it doesn't already exist with the same content.

  This is useful because cargo uses timestamps to decide whether to compile things."""
  if os.path.exists(path):
    with open(path, 'r') as f:
      if content == f.read():
        print("Not overwriting {} because it is unchanged".format(path), file=sys.stderr)
        return

  with open(path, 'w') as f:
    f.write(content)

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

def from_json(s):
    """Given a JSON-encoded message, build an object.

    """
    d = json.loads(s)
    sbp = SBP.from_json_dict(d)
    return sbp

def is_iterable_but_not_string(obj):
    """
    Determine whether or not obj is iterable but not a string (eg, a list, set, tuple etc).
    """
    return hasattr(obj, '__iter__') and not isinstance(obj, str) and not isinstance(obj, bytes)

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

def _is_one_arg_pos_call(call):
    """Is this a call with exactly 1 argument,
    where that argument is positional?
    """
    return isinstance(call, astroid.Call) and len(call.args) == 1 and not call.keywords

def _varargs_to_iterable_method(func):
    """decorator to convert a *args method to one taking a iterable"""
    def wrapped(self, iterable, **kwargs):
        return func(self, *iterable, **kwargs)
    return wrapped

def is_defined(self, objtxt, force_import=False):
        """Return True if object is defined"""
        return self.interpreter.is_defined(objtxt, force_import)

def _iterPoints(self, **kwargs):
        """
        Subclasses may override this method.
        """
        points = self.points
        count = len(points)
        index = 0
        while count:
            yield points[index]
            count -= 1
            index += 1

def safe_quotes(text, escape_single_quotes=False):
    """htmlify string"""
    if isinstance(text, str):
        safe_text = text.replace('"', "&quot;")
        if escape_single_quotes:
            safe_text = safe_text.replace("'", "&#92;'")
        return safe_text.replace('True', 'true')
    return text

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

def IPYTHON_MAIN():
    """Decide if the Ipython command line is running code."""
    import pkg_resources

    runner_frame = inspect.getouterframes(inspect.currentframe())[-2]
    return (
        getattr(runner_frame, "function", None)
        == pkg_resources.load_entry_point("ipython", "console_scripts", "ipython").__name__
    )

def random_filename(path=None):
    """Make a UUID-based file name which is extremely unlikely
    to exist already."""
    filename = uuid4().hex
    if path is not None:
        filename = os.path.join(path, filename)
    return filename

def _begins_with_one_of(sentence, parts_of_speech):
    """Return True if the sentence or fragment begins with one of the parts of
    speech in the list, else False"""
    doc = nlp(sentence)
    if doc[0].tag_ in parts_of_speech:
        return True
    return False

def as_float_array(a):
    """View the quaternion array as an array of floats

    This function is fast (of order 1 microsecond) because no data is
    copied; the returned quantity is just a "view" of the original.

    The output view has one more dimension (of size 4) than the input
    array, but is otherwise the same shape.

    """
    return np.asarray(a, dtype=np.quaternion).view((np.double, 4))

def eof(fd):
    """Determine if end-of-file is reached for file fd."""
    b = fd.read(1)
    end = len(b) == 0
    if not end:
        curpos = fd.tell()
        fd.seek(curpos - 1)
    return end

def is_element_present(self, strategy, locator):
        """Checks whether an element is present.

        :param strategy: Location strategy to use. See :py:class:`~selenium.webdriver.common.by.By` or :py:attr:`~pypom.splinter_driver.ALLOWED_STRATEGIES`.
        :param locator: Location of target element.
        :type strategy: str
        :type locator: str
        :return: ``True`` if element is present, else ``False``.
        :rtype: bool

        """
        return self.driver_adapter.is_element_present(strategy, locator, root=self.root)

def OnDoubleClick(self, event):
        """Double click on a given square in the map"""
        node = HotMapNavigator.findNodeAtPosition(self.hot_map, event.GetPosition())
        if node:
            wx.PostEvent( self, SquareActivationEvent( node=node, point=event.GetPosition(), map=self ) )

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

def asMaskedArray(self):
        """ Creates converts to a masked array
        """
        return ma.masked_array(data=self.data, mask=self.mask, fill_value=self.fill_value)

def _is_readable(self, obj):
        """Check if the argument is a readable file-like object."""
        try:
            read = getattr(obj, 'read')
        except AttributeError:
            return False
        else:
            return is_method(read, max_arity=1)

def _match_literal(self, a, b=None):
        """Match two names."""

        return a.lower() == b if not self.case_sensitive else a == b

def stopwatch_now():
    """Get a timevalue for interval comparisons

    When possible it is a monotonic clock to prevent backwards time issues.
    """
    if six.PY2:
        now = time.time()
    else:
        now = time.monotonic()
    return now

def v_normalize(v):
    """
    Normalizes the given vector.
    
    The vector given may have any number of dimensions.
    """
    vmag = v_magnitude(v)
    return [ v[i]/vmag  for i in range(len(v)) ]

def _validate_pos(df):
    """Validates the returned positional object
    """
    assert isinstance(df, pd.DataFrame)
    assert ["seqname", "position", "strand"] == df.columns.tolist()
    assert df.position.dtype == np.dtype("int64")
    assert df.strand.dtype == np.dtype("O")
    assert df.seqname.dtype == np.dtype("O")
    return df

def set_title(self, title, **kwargs):
        """Sets the title on the underlying matplotlib AxesSubplot."""
        ax = self.get_axes()
        ax.set_title(title, **kwargs)

def autoconvert(string):
    """Try to convert variables into datatypes."""
    for fn in (boolify, int, float):
        try:
            return fn(string)
        except ValueError:
            pass
    return string

def _axes(self):
        """Set the _force_vertical flag when rendering axes"""
        self.view._force_vertical = True
        super(HorizontalGraph, self)._axes()
        self.view._force_vertical = False

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

def clear_matplotlib_ticks(self, axis="both"):
        """Clears the default matplotlib ticks."""
        ax = self.get_axes()
        plotting.clear_matplotlib_ticks(ax=ax, axis=axis)

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

def _change_height(self, ax, new_value):
        """Make bars in horizontal bar chart thinner"""
        for patch in ax.patches:
            current_height = patch.get_height()
            diff = current_height - new_value

            # we change the bar height
            patch.set_height(new_value)

            # we recenter the bar
            patch.set_y(patch.get_y() + diff * .5)

def _IsDirectory(parent, item):
  """Helper that returns if parent/item is a directory."""
  return tf.io.gfile.isdir(os.path.join(parent, item))

def set_axis_options(self, row, column, text):
        """Set additionnal options as plain text."""

        subplot = self.get_subplot_at(row, column)
        subplot.set_axis_options(text)

def downsample(array, k):
    """Choose k random elements of array."""
    length = array.shape[0]
    indices = random.sample(xrange(length), k)
    return array[indices]

def normalize(X):
    """ equivalent to scipy.preprocessing.normalize on sparse matrices
    , but lets avoid another depedency just for a small utility function """
    X = coo_matrix(X)
    X.data = X.data / sqrt(bincount(X.row, X.data ** 2))[X.row]
    return X

def remove(self, entry):
        """Removes an entry"""
        try:
            list = self.cache[entry.key]
            list.remove(entry)
        except:
            pass

def max_values(args):
    """ Return possible range for max function. """
    return Interval(max(x.low for x in args), max(x.high for x in args))

def detach_all(self):
        """
        Detach from all tracked classes and objects.
        Restore the original constructors and cleanse the tracking lists.
        """
        self.detach_all_classes()
        self.objects.clear()
        self.index.clear()
        self._keepalive[:] = []

def get_file_md5sum(path):
    """Calculate the MD5 hash for a file."""
    with open(path, 'rb') as fh:
        h = str(hashlib.md5(fh.read()).hexdigest())
    return h

def _clear_dir(dirName):
    """ Remove a directory and it contents. Ignore any failures.
    """
    # If we got here, clear dir  
    for fname in os.listdir(dirName):
        try:
            os.remove( os.path.join(dirName, fname) )
        except Exception:
            pass
    try:
        os.rmdir(dirName)
    except Exception:
        pass

def pointer(self):
        """Get a ctypes void pointer to the memory mapped region.

        :type: ctypes.c_void_p
        """
        return ctypes.cast(ctypes.pointer(ctypes.c_uint8.from_buffer(self.mapping, 0)), ctypes.c_void_p)

def stoplog(self):
        """ Stop logging.
    
        @return: 1 on success and 0 on error
        @rtype: integer
        """
        if self._file_logger:
            self.logger.removeHandler(_file_logger)
            self._file_logger = None
        return 1

def SegmentMin(a, ids):
    """
    Segmented min op.
    """
    func = lambda idxs: np.amin(a[idxs], axis=0)
    return seg_map(func, a, ids),

def clear():
    """Clears the console."""
    if sys.platform.startswith("win"):
        call("cls", shell=True)
    else:
        call("clear", shell=True)

def _get_minidom_tag_value(station, tag_name):
    """get a value from a tag (if it exists)"""
    tag = station.getElementsByTagName(tag_name)[0].firstChild
    if tag:
        return tag.nodeValue

    return None

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

def __setitem__(self, _ignored, return_value):
        """Item assignment sets the return value and removes any side effect"""
        self.mock.return_value = return_value
        self.mock.side_effect = None

def close(self):
        """Closes the serial port."""
        if self.pyb and self.pyb.serial:
            self.pyb.serial.close()
        self.pyb = None

def finish():
    """Print warning about interrupt and empty the job queue."""
    out.warn("Interrupted!")
    for t in threads:
        t.stop()
    jobs.clear()
    out.warn("Waiting for download threads to finish.")

def requests_post(url, data=None, json=None, **kwargs):
    """Requests-mock requests.post wrapper."""
    return requests_request('post', url, data=data, json=json, **kwargs)

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

def _construct_from_json(self, rec):
        """ Construct this Dagobah instance from a JSON document. """

        self.delete()

        for required_key in ['dagobah_id', 'created_jobs']:
            setattr(self, required_key, rec[required_key])

        for job_json in rec.get('jobs', []):
            self._add_job_from_spec(job_json)

        self.commit(cascade=True)

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

def wrap_count(method):
    """
    Returns number of wraps around given method.
    """
    number = 0
    while hasattr(method, '__aspects_orig'):
        number += 1
        method = method.__aspects_orig
    return number

def _most_common(iterable):
    """Returns the most common element in `iterable`."""
    data = Counter(iterable)
    return max(data, key=data.__getitem__)

def do_next(self, args):
        """Step over the next statement
        """
        self._do_print_from_last_cmd = True
        self._interp.step_over()
        return True

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

def match_paren(self, tokens, item):
        """Matches a paren."""
        match, = tokens
        return self.match(match, item)

def _convert_to_array(array_like, dtype):
        """
        Convert Matrix attributes which are array-like or buffer to array.
        """
        if isinstance(array_like, bytes):
            return np.frombuffer(array_like, dtype=dtype)
        return np.asarray(array_like, dtype=dtype)

def def_linear(fun):
    """Flags that a function is linear wrt all args"""
    defjvp_argnum(fun, lambda argnum, g, ans, args, kwargs:
                  fun(*subval(args, argnum, g), **kwargs))

def _kw(keywords):
    """Turn list of keywords into dictionary."""
    r = {}
    for k, v in keywords:
        r[k] = v
    return r

def has_multiline_items(maybe_list: Optional[Sequence[str]]):
    """Check whether one of the items in the list has multiple lines."""
    return maybe_list and any(is_multiline(item) for item in maybe_list)

def add(self, name, desc, func=None, args=None, krgs=None):
        """Add a menu entry."""
        self.entries.append(MenuEntry(name, desc, func, args or [], krgs or {}))

def combine(self, a, b):
        """A generator that combines two iterables."""

        for l in (a, b):
            for x in l:
                yield x

def to_json(data):
    """Return data as a JSON string."""
    return json.dumps(data, default=lambda x: x.__dict__, sort_keys=True, indent=4)

def many_until1(these, term):
    """Like many_until but must consume at least one of these.
    """
    first = [these()]
    these_results, term_result = many_until(these, term)
    return (first + these_results, term_result)

def _heapify_max(x):
    """Transform list into a maxheap, in-place, in O(len(x)) time."""
    n = len(x)
    for i in reversed(range(n//2)):
        _siftup_max(x, i)

def kill_mprocess(process):
    """kill process
    Args:
        process - Popen object for process
    """
    if process and proc_alive(process):
        process.terminate()
        process.communicate()
    return not proc_alive(process)

def zoom_out(self):
        """Decrease zoom factor and redraw TimeLine"""
        index = self._zoom_factors.index(self._zoom_factor)
        if index == 0:
            # Already zoomed out all the way
            return
        self._zoom_factor = self._zoom_factors[index - 1]
        if self._zoom_factors.index(self._zoom_factor) == 0:
            self._button_zoom_out.config(state=tk.DISABLED)
        self._button_zoom_in.config(state=tk.NORMAL)
        self.draw_timeline()

def initialize_worker(self, process_num=None):
        """
        reinitialize consumer for process in multiprocesing
        """
        self.initialize(self.grid, self.num_of_paths, self.seed)

def _unique_id(self, prefix):
        """
        Generate a unique (within the graph) identifer
        internal to graph generation.
        """
        _id = self._id_gen
        self._id_gen += 1
        return prefix + str(_id)

def compute(args):
    x, y, params = args
    """Callable function for the multiprocessing pool."""
    return x, y, mandelbrot(x, y, params)

def get_distance_matrix(x):
    """Get distance matrix given a matrix. Used in testing."""
    square = nd.sum(x ** 2.0, axis=1, keepdims=True)
    distance_square = square + square.transpose() - (2.0 * nd.dot(x, x.transpose()))
    return nd.sqrt(distance_square)

def compute_capture(args):
    x, y, w, h, params = args
    """Callable function for the multiprocessing pool."""
    return x, y, mandelbrot_capture(x, y, w, h, params)

def _cell(x):
    """translate an array x into a MATLAB cell array"""
    x_no_none = [i if i is not None else "" for i in x]
    return array(x_no_none, dtype=np_object)

def finish():
    """Print warning about interrupt and empty the job queue."""
    out.warn("Interrupted!")
    for t in threads:
        t.stop()
    jobs.clear()
    out.warn("Waiting for download threads to finish.")

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

def connect_mysql(host, port, user, password, database):
    """Connect to MySQL with retries."""
    return pymysql.connect(
        host=host, port=port,
        user=user, passwd=password,
        db=database
    )

def tick(self):
        """Add one tick to progress bar"""
        self.current += 1
        if self.current == self.factor:
            sys.stdout.write('+')
            sys.stdout.flush()
            self.current = 0

def connect_mysql(host, port, user, password, database):
    """Connect to MySQL with retries."""
    return pymysql.connect(
        host=host, port=port,
        user=user, passwd=password,
        db=database
    )

def doc_to_html(doc, doc_format="ROBOT"):
    """Convert documentation to HTML"""
    from robot.libdocpkg.htmlwriter import DocToHtml
    return DocToHtml(doc_format)(doc)

def dictify(a_named_tuple):
    """Transform a named tuple into a dictionary"""
    return dict((s, getattr(a_named_tuple, s)) for s in a_named_tuple._fields)

def setdefault(obj, field, default):
    """Set an object's field to default if it doesn't have a value"""
    setattr(obj, field, getattr(obj, field, default))

def maskIndex(self):
        """ Returns a boolean index with True if the value is masked.

            Always has the same shape as the maksedArray.data, event if the mask is a single boolan.
        """
        if isinstance(self.mask, bool):
            return np.full(self.data.shape, self.mask, dtype=np.bool)
        else:
            return self.mask

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

def last_item(array):
    """Returns the last item of an array in a list or an empty list."""
    if array.size == 0:
        # work around for https://github.com/numpy/numpy/issues/5195
        return []

    indexer = (slice(-1, None),) * array.ndim
    return np.ravel(array[indexer]).tolist()

def ratio_and_percentage(current, total, time_remaining):
    """Returns the progress ratio and percentage."""
    return "{} / {} ({}% completed)".format(current, total, int(current / total * 100))

def _attrprint(d, delimiter=', '):
    """Print a dictionary of attributes in the DOT format"""
    return delimiter.join(('"%s"="%s"' % item) for item in sorted(d.items()))

def set_color(self, fg=None, bg=None, intensify=False, target=sys.stdout):
        """Set foreground- and background colors and intensity."""
        raise NotImplementedError

def _bytes_to_json(value):
    """Coerce 'value' to an JSON-compatible representation."""
    if isinstance(value, bytes):
        value = base64.standard_b64encode(value).decode("ascii")
    return value

def cli(env, identifier):
    """Delete an image."""

    image_mgr = SoftLayer.ImageManager(env.client)
    image_id = helpers.resolve_id(image_mgr.resolve_ids, identifier, 'image')

    image_mgr.delete_image(image_id)

def bytes_to_str(s, encoding='utf-8'):
    """Returns a str if a bytes object is given."""
    if six.PY3 and isinstance(s, bytes):
        return s.decode(encoding)
    return s

def remove_last_line(self):
        """Removes the last line of the document."""
        editor = self._editor
        text_cursor = editor.textCursor()
        text_cursor.movePosition(text_cursor.End, text_cursor.MoveAnchor)
        text_cursor.select(text_cursor.LineUnderCursor)
        text_cursor.removeSelectedText()
        text_cursor.deletePreviousChar()
        editor.setTextCursor(text_cursor)

def norm(x, mu, sigma=1.0):
    """ Scipy norm function """
    return stats.norm(loc=mu, scale=sigma).pdf(x)

def stats(self):
        """ shotcut to pull out useful info for interactive use """
        printDebug("Classes.....: %d" % len(self.all_classes))
        printDebug("Properties..: %d" % len(self.all_properties))

def pdf(x, mu, std):
    """Probability density function (normal distribution)"""
    return (1.0 / (std * sqrt(2 * pi))) * np.exp(-(x - mu) ** 2 / (2 * std ** 2))

def on_mouse_motion(self, x, y, dx, dy):
        """
        Pyglet specific mouse motion callback.
        Forwards and traslates the event to the example
        """
        # Screen coordinates relative to the lower-left corner
        # so we have to flip the y axis to make this consistent with
        # other window libraries
        self.example.mouse_position_event(x, self.buffer_height - y)

def _normalize(mat: np.ndarray):
    """rescales a numpy array, so that min is 0 and max is 255"""
    return ((mat - mat.min()) * (255 / mat.max())).astype(np.uint8)

def _namematcher(regex):
    """Checks if a target name matches with an input regular expression."""

    matcher = re_compile(regex)

    def match(target):
        target_name = getattr(target, '__name__', '')
        result = matcher.match(target_name)
        return result

    return match

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

def read_key(suppress=False):
    """
    Blocks until a keyboard event happens, then returns that event's name or,
    if missing, its scan code.
    """
    event = read_event(suppress)
    return event.name or event.scan_code

def normalize(self, string):
        """Normalize the string according to normalization list"""
        return ''.join([self._normalize.get(x, x) for x in nfd(string)])

def is_string(val):
    """Determines whether the passed value is a string, safe for 2/3."""
    try:
        basestring
    except NameError:
        return isinstance(val, str)
    return isinstance(val, basestring)

def ComplementEquivalence(*args, **kwargs):
    """Change x != y to not(x == y)."""
    return ast.Complement(
        ast.Equivalence(*args, **kwargs), **kwargs)

def test_for_image(self, cat, img):
        """Tests if image img in category cat exists"""
        return self.test_for_category(cat) and img in self.items[cat]

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

def other_ind(self):
        """last row or column of square A"""
        return np.full(self.n_min, self.size - 1, dtype=np.int)

def incidence(boundary):
    """
    given an Nxm matrix containing boundary info between simplices,
    compute indidence info matrix
    not very reusable; should probably not be in this lib
    """
    return GroupBy(boundary).split(np.arange(boundary.size) // boundary.shape[1])

def Max(a, axis, keep_dims):
    """
    Max reduction op.
    """
    return np.amax(a, axis=axis if not isinstance(axis, np.ndarray) else tuple(axis),
                   keepdims=keep_dims),

def ratio_and_percentage(current, total, time_remaining):
    """Returns the progress ratio and percentage."""
    return "{} / {} ({}% completed)".format(current, total, int(current / total * 100))

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

def computeFactorial(n):
    """
    computes factorial of n
    """
    sleep_walk(10)
    ret = 1
    for i in range(n):
        ret = ret * (i + 1)
    return ret

def get_distance_matrix(x):
    """Get distance matrix given a matrix. Used in testing."""
    square = nd.sum(x ** 2.0, axis=1, keepdims=True)
    distance_square = square + square.transpose() - (2.0 * nd.dot(x, x.transpose()))
    return nd.sqrt(distance_square)

def wrap(string, length, indent):
    """ Wrap a string at a line length """
    newline = "\n" + " " * indent
    return newline.join((string[i : i + length] for i in range(0, len(string), length)))

def read_numpy(fh, byteorder, dtype, count, offsetsize):
    """Read tag data from file and return as numpy array."""
    dtype = 'b' if dtype[-1] == 's' else byteorder+dtype[-1]
    return fh.read_array(dtype, count)

def inh(table):
    """
    inverse hyperbolic sine transformation
    """
    t = []
    for i in table:
        t.append(np.ndarray.tolist(np.arcsinh(i)))
    return t

def read_numpy(fh, byteorder, dtype, count, offsetsize):
    """Read tag data from file and return as numpy array."""
    dtype = 'b' if dtype[-1] == 's' else byteorder+dtype[-1]
    return fh.read_array(dtype, count)

def cleanup_storage(*args):
    """Clean up processes after SIGTERM or SIGINT is received."""
    ShardedClusters().cleanup()
    ReplicaSets().cleanup()
    Servers().cleanup()
    sys.exit(0)

def dict_to_numpy_array(d):
    """
    Convert a dict of 1d array to a numpy recarray
    """
    return fromarrays(d.values(), np.dtype([(str(k), v.dtype) for k, v in d.items()]))

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

def from_array(cls, arr):
        """Convert a structured NumPy array into a Table."""
        return cls().with_columns([(f, arr[f]) for f in arr.dtype.names])

def sorted_product_set(array_a, array_b):
  """Compute the product set of array_a and array_b and sort it."""
  return np.sort(
      np.concatenate(
          [array_a[i] * array_b for i in xrange(len(array_a))], axis=0)
  )[::-1]

def _xls2col_widths(self, worksheet, tab):
        """Updates col_widths in code_array"""

        for col in xrange(worksheet.ncols):
            try:
                xls_width = worksheet.colinfo_map[col].width
                pys_width = self.xls_width2pys_width(xls_width)
                self.code_array.col_widths[col, tab] = pys_width

            except KeyError:
                pass

def _tostr(self,obj):
        """ converts a object to list, if object is a list, it creates a
            comma seperated string.
        """
        if not obj:
            return ''
        if isinstance(obj, list):
            return ', '.join(map(self._tostr, obj))
        return str(obj)

def iterlists(self):
        """Like :meth:`items` but returns an iterator."""
        for key, values in dict.iteritems(self):
            yield key, list(values)

def delete_connection():
    """
    Stop and destroy Bloomberg connection
    """
    if _CON_SYM_ in globals():
        con = globals().pop(_CON_SYM_)
        if not getattr(con, '_session').start(): con.stop()

def PopAttributeContainer(self):
    """Pops a serialized attribute container from the list.

    Returns:
      bytes: serialized attribute container data.
    """
    try:
      serialized_data = self._list.pop(0)
      self.data_size -= len(serialized_data)
      return serialized_data

    except IndexError:
      return None

def _ignore_comments(lines_enum):
    """
    Strips comments and filter empty lines.
    """
    for line_number, line in lines_enum:
        line = COMMENT_RE.sub('', line)
        line = line.strip()
        if line:
            yield line_number, line

def listlike(obj):
    """Is an object iterable like a list (and not a string)?"""
    
    return hasattr(obj, "__iter__") \
    and not issubclass(type(obj), str)\
    and not issubclass(type(obj), unicode)

def _quit(self, *args):
        """ quit crash """
        self.logger.warn('Bye!')
        sys.exit(self.exit())

def to_json(obj):
    """Return a json string representing the python object obj."""
    i = StringIO.StringIO()
    w = Writer(i, encoding='UTF-8')
    w.write_value(obj)
    return i.getvalue()

def ip_address_list(ips):
    """ IP address range validation and expansion. """
    # first, try it as a single IP address
    try:
        return ip_address(ips)
    except ValueError:
        pass
    # then, consider it as an ipaddress.IPv[4|6]Network instance and expand it
    return list(ipaddress.ip_network(u(ips)).hosts())

def delete(self):
        """Remove this object."""
        self._client.remove_object(self._instance, self._bucket, self.name)

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

def one_hot(x, size, dtype=np.float32):
  """Make a n+1 dim one-hot array from n dim int-categorical array."""
  return np.array(x[..., np.newaxis] == np.arange(size), dtype)

def _fill_array_from_list(the_list, the_array):
        """Fill an `array` from a `list`"""
        for i, val in enumerate(the_list):
            the_array[i] = val
        return the_array

def add_suffix(fullname, suffix):
    """ Add suffix to a full file name"""
    name, ext = os.path.splitext(fullname)
    return name + '_' + suffix + ext

def filter_dict(d, keys):
    """
    Creates a new dict from an existing dict that only has the given keys
    """
    return {k: v for k, v in d.items() if k in keys}

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

def rotate_img(im, deg, mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_AREA):
    """ Rotates an image by deg degrees

    Arguments:
        deg (float): degree to rotate.
    """
    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c//2,r//2),deg,1)
    return cv2.warpAffine(im,M,(c,r), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS+interpolation)

def rpc_fix_code_with_black(self, source, directory):
        """Formats Python code to conform to the PEP 8 style guide.

        """
        source = get_source(source)
        return fix_code_with_black(source, directory)

def set(self):
        """Set the color as current OpenGL color
        """
        glColor4f(self.r, self.g, self.b, self.a)

def warn_deprecated(message, stacklevel=2):  # pragma: no cover
    """Warn deprecated."""

    warnings.warn(
        message,
        category=DeprecationWarning,
        stacklevel=stacklevel
    )

def _genTex2D(self):
        """Generate an empty texture in OpenGL"""
        for face in range(6):
            gl.glTexImage2D(self.target0 + face, 0, self.internal_fmt, self.width, self.height, 0,
                            self.pixel_fmt, gl.GL_UNSIGNED_BYTE, 0)

def cursor_up(self, count=1):
        """ (for multiline edit). Move cursor to the previous line.  """
        original_column = self.preferred_column or self.document.cursor_position_col
        self.cursor_position += self.document.get_cursor_up_position(
            count=count, preferred_column=original_column)

        # Remember the original column for the next up/down movement.
        self.preferred_column = original_column

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

def get_time(filename):
	"""
	Get the modified time for a file as a datetime instance
	"""
	ts = os.stat(filename).st_mtime
	return datetime.datetime.utcfromtimestamp(ts)

def get_order(self):
        """
        Return a list of dicionaries. See `set_order`.
        """
        return [dict(reverse=r[0], key=r[1]) for r in self.get_model()]

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

def uniqueID(size=6, chars=string.ascii_uppercase + string.digits):
    """A quick and dirty way to get a unique string"""
    return ''.join(random.choice(chars) for x in xrange(size))

def entropy(string):
    """Compute entropy on the string"""
    p, lns = Counter(string), float(len(string))
    return -sum(count/lns * math.log(count/lns, 2) for count in p.values())

def generate_nonce():
        """ Generate nonce number """
        nonce = ''.join([str(randint(0, 9)) for i in range(8)])
        return HMAC(
            nonce.encode(),
            "secret".encode(),
            sha1
        ).hexdigest()

def get_key_by_value(dictionary, search_value):
    """
    searchs a value in a dicionary and returns the key of the first occurrence

    :param dictionary: dictionary to search in
    :param search_value: value to search for
    """
    for key, value in dictionary.iteritems():
        if value == search_value:
            return ugettext(key)

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

def parse_datetime(dt_str):
    """Parse datetime."""
    date_format = "%Y-%m-%dT%H:%M:%S %z"
    dt_str = dt_str.replace("Z", " +0000")
    return datetime.datetime.strptime(dt_str, date_format)

def visit_BoolOp(self, node):
        """ Return type may come from any boolop operand. """
        return sum((self.visit(value) for value in node.values), [])

def _parse_response(self, response):
        """
        Parse http raw respone into python
        dictionary object.
        
        :param str response: http response
        :returns: response dict
        :rtype: dict
        """

        response_dict = {}
        for line in response.splitlines():
            key, value = response.split("=", 1)
            response_dict[key] = value
        return response_dict

def generate_unique_host_id():
    """Generate a unique ID, that is somewhat guaranteed to be unique among all
    instances running at the same time."""
    host = ".".join(reversed(socket.gethostname().split(".")))
    pid = os.getpid()
    return "%s.%d" % (host, pid)

def parse_timestamp(timestamp):
    """Parse ISO8601 timestamps given by github API."""
    dt = dateutil.parser.parse(timestamp)
    return dt.astimezone(dateutil.tz.tzutc())

def _get_os_environ_dict(keys):
  """Return a dictionary of key/values from os.environ."""
  return {k: os.environ.get(k, _UNDEFINED) for k in keys}

def rfc3339_to_datetime(data):
    """convert a rfc3339 date representation into a Python datetime"""
    try:
        ts = time.strptime(data, '%Y-%m-%d')
        return date(*ts[:3])
    except ValueError:
        pass

    try:
        dt, _, tz = data.partition('Z')
        if tz:
            tz = offset(tz)
        else:
            tz = offset('00:00')
        if '.' in dt and dt.rsplit('.', 1)[-1].isdigit():
            ts = time.strptime(dt, '%Y-%m-%dT%H:%M:%S.%f')
        else:
            ts = time.strptime(dt, '%Y-%m-%dT%H:%M:%S')
        return datetime(*ts[:6], tzinfo=tz)
    except ValueError:
        raise ValueError('date-time {!r} is not a valid rfc3339 date representation'.format(data))

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

def _ParseYamlFromFile(filedesc):
  """Parses given YAML file."""
  content = filedesc.read()
  return yaml.Parse(content) or collections.OrderedDict()

def get_first_lang():
    """Get the first lang of Accept-Language Header.
    """
    request_lang = request.headers.get('Accept-Language').split(',')
    if request_lang:
        lang = locale.normalize(request_lang[0]).split('.')[0]
    else:
        lang = False
    return lang

def from_pystr_to_cstr(data):
    """Convert a list of Python str to C pointer

    Parameters
    ----------
    data : list
        list of str
    """

    if not isinstance(data, list):
        raise NotImplementedError
    pointers = (ctypes.c_char_p * len(data))()
    if PY3:
        data = [bytes(d, 'utf-8') for d in data]
    else:
        data = [d.encode('utf-8') if isinstance(d, unicode) else d  # pylint: disable=undefined-variable
                for d in data]
    pointers[:] = data
    return pointers

def get_key_by_value(dictionary, search_value):
    """
    searchs a value in a dicionary and returns the key of the first occurrence

    :param dictionary: dictionary to search in
    :param search_value: value to search for
    """
    for key, value in dictionary.iteritems():
        if value == search_value:
            return ugettext(key)

def osx_clipboard_get():
    """ Get the clipboard's text on OS X.
    """
    p = subprocess.Popen(['pbpaste', '-Prefer', 'ascii'],
        stdout=subprocess.PIPE)
    text, stderr = p.communicate()
    # Text comes in with old Mac \r line endings. Change them to \n.
    text = text.replace('\r', '\n')
    return text

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

def get_parent_dir(name):
    """Get the parent directory of a filename."""
    parent_dir = os.path.dirname(os.path.dirname(name))
    if parent_dir:
        return parent_dir
    return os.path.abspath('.')

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

def set_trace():
    """Start a Pdb instance at the calling frame, with stdout routed to sys.__stdout__."""
    # https://github.com/nose-devs/nose/blob/master/nose/tools/nontrivial.py
    pdb.Pdb(stdout=sys.__stdout__).set_trace(sys._getframe().f_back)

def mag(z):
    """Get the magnitude of a vector."""
    if isinstance(z[0], np.ndarray):
        return np.array(list(map(np.linalg.norm, z)))
    else:
        return np.linalg.norm(z)

def dimensions(path):
    """Get width and height of a PDF"""
    pdf = PdfFileReader(path)
    size = pdf.getPage(0).mediaBox
    return {'w': float(size[2]), 'h': float(size[3])}

def get_from_human_key(self, key):
        """Return the key (aka database value) of a human key (aka Python identifier)."""
        if key in self._identifier_map:
            return self._identifier_map[key]
        raise KeyError(key)

def dimensions(path):
    """Get width and height of a PDF"""
    pdf = PdfFileReader(path)
    size = pdf.getPage(0).mediaBox
    return {'w': float(size[2]), 'h': float(size[3])}

def round_to_float(number, precision):
    """Round a float to a precision"""
    rounded = Decimal(str(floor((number + precision / 2) // precision))
                      ) * Decimal(str(precision))
    return float(rounded)

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

def document(schema):
    """Print a documented teleport version of the schema."""
    teleport_schema = from_val(schema)
    return json.dumps(teleport_schema, sort_keys=True, indent=2)

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

def getScreenDims(self):
        """returns a tuple that contains (screen_width,screen_height)
        """
        width = ale_lib.getScreenWidth(self.obj)
        height = ale_lib.getScreenHeight(self.obj)
        return (width,height)

def on_stop(self):
        """
        stop publisher
        """
        LOGGER.debug("zeromq.Publisher.on_stop")
        self.zmqsocket.close()
        self.zmqcontext.destroy()

def strip_spaces(x):
    """
    Strips spaces
    :param x:
    :return:
    """
    x = x.replace(b' ', b'')
    x = x.replace(b'\t', b'')
    return x

def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return F.pad(img, self.padding, self.fill, self.padding_mode)

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

def pause(self):
        """Pause the music"""
        mixer.music.pause()
        self.pause_time = self.get_time()
        self.paused = True

def out_shape_from_array(arr):
    """Get the output shape from an array."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.shape
    else:
        return (arr.shape[1],)

def add_matplotlib_cmap(cm, name=None):
    """Add a matplotlib colormap."""
    global cmaps
    cmap = matplotlib_to_ginga_cmap(cm, name=name)
    cmaps[cmap.name] = cmap

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

def stackplot(marray, seconds=None, start_time=None, ylabels=None):
    """
    will plot a stack of traces one above the other assuming
    marray.shape = numRows, numSamples
    """
    tarray = np.transpose(marray)
    stackplot_t(tarray, seconds=seconds, start_time=start_time, ylabels=ylabels)
    plt.show()

def get_obj(ref):
    """Get object from string reference."""
    oid = int(ref)
    return server.id2ref.get(oid) or server.id2obj[oid]

def asyncStarCmap(asyncCallable, iterable):
    """itertools.starmap for deferred callables using cooperative multitasking
    """
    results = []
    yield coopStar(asyncCallable, results.append, iterable)
    returnValue(results)

def size_on_disk(self):
        """
        :return: size of the entire schema in bytes
        """
        return int(self.connection.query(
            """
            SELECT SUM(data_length + index_length)
            FROM information_schema.tables WHERE table_schema='{db}'
            """.format(db=self.database)).fetchone()[0])

def _ReturnConnection(self):
		"""
		Returns a connection back to the pool
		
		@author: Nick Verbeck
		@since: 9/7/2008
		"""
		if self.conn is not None:
			if self.connInfo.commitOnEnd is True or self.commitOnEnd is True:
				self.conn.Commit()
					
			Pool().returnConnection(self.conn)
			self.conn = None

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

def git_tag(tag):
    """Tags the current version."""
    print('Tagging "{}"'.format(tag))
    msg = '"Released version {}"'.format(tag)
    Popen(['git', 'tag', '-s', '-m', msg, tag]).wait()

def end_index(self):
        """
        Returns the 1-based index of the last object on this page,
        relative to total objects found (hits).
        """
        return ((self.number - 1) * self.paginator.per_page +
            len(self.object_list))

def make_post_request(self, url, auth, json_payload):
        """This function executes the request with the provided
        json payload and return the json response"""
        response = requests.post(url, auth=auth, json=json_payload)
        return response.json()

def display_len(text):
    """
    Get the display length of a string. This can differ from the character
    length if the string contains wide characters.
    """
    text = unicodedata.normalize('NFD', text)
    return sum(char_width(char) for char in text)

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

def get_from_headers(request, key):
    """Try to read a value named ``key`` from the headers.
    """
    value = request.headers.get(key)
    return to_native(value)

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

def set_color(self, fg=None, bg=None, intensify=False, target=sys.stdout):
        """Set foreground- and background colors and intensity."""
        raise NotImplementedError

def csvpretty(csvfile: csvfile=sys.stdin):
    """ Pretty print a CSV file. """
    shellish.tabulate(csv.reader(csvfile))

def uniquify_list(L):
    """Same order unique list using only a list compression."""
    return [e for i, e in enumerate(L) if L.index(e) == i]

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

def pp_xml(body):
    """Pretty print format some XML so it's readable."""
    pretty = xml.dom.minidom.parseString(body)
    return pretty.toprettyxml(indent="  ")

def get_mi_vec(slab):
    """
    Convenience function which returns the unit vector aligned
    with the miller index.
    """
    mvec = np.cross(slab.lattice.matrix[0], slab.lattice.matrix[1])
    return mvec / np.linalg.norm(mvec)

def print_param_values(self_):
        """Print the values of all this object's Parameters."""
        self = self_.self
        for name,val in self.param.get_param_values():
            print('%s.%s = %s' % (self.name,name,val))

def get_week_start_end_day():
    """
    Get the week start date and end date
    """
    t = date.today()
    wd = t.weekday()
    return (t - timedelta(wd), t + timedelta(6 - wd))

def raw_print(*args, **kw):
    """Raw print to sys.__stdout__, otherwise identical interface to print()."""

    print(*args, sep=kw.get('sep', ' '), end=kw.get('end', '\n'),
          file=sys.__stdout__)
    sys.__stdout__.flush()

def get_screen_resolution(self):
        """Return the screen resolution of the primary screen."""
        widget = QDesktopWidget()
        geometry = widget.availableGeometry(widget.primaryScreen())
        return geometry.width(), geometry.height()

def info(txt):
    """Print, emphasized 'neutral', the given 'txt' message"""

    print("%s# %s%s%s" % (PR_EMPH_CC, get_time_stamp(), txt, PR_NC))
    sys.stdout.flush()

def stringc(text, color):
    """
    Return a string with terminal colors.
    """
    if has_colors:
        text = str(text)

        return "\033["+codeCodes[color]+"m"+text+"\033[0m"
    else:
        return text

def _format_json(data, theme):
    """Pretty print a dict as a JSON, with colors if pygments is present."""
    output = json.dumps(data, indent=2, sort_keys=True)

    if pygments and sys.stdout.isatty():
        style = get_style_by_name(theme)
        formatter = Terminal256Formatter(style=style)
        return pygments.highlight(output, JsonLexer(), formatter)

    return output

def move_page_bottom(self):
        """
        Move the cursor to the last item on the page.
        """
        self.nav.page_index = self.content.range[1]
        self.nav.cursor_index = 0
        self.nav.inverted = True

def timeit(output):
    """
    If output is string, then print the string and also time used
    """
    b = time.time()
    yield
    print output, 'time used: %.3fs' % (time.time()-b)

def empty_tree(input_list):
    """Recursively iterate through values in nested lists."""
    for item in input_list:
        if not isinstance(item, list) or not empty_tree(item):
            return False
    return True

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

def print_float(self, value, decimal_digits=2, justify_right=True):
        """Print a numeric value to the display.  If value is negative
        it will be printed with a leading minus sign.  Decimal digits is the
        desired number of digits after the decimal point.
        """
        format_string = '{{0:0.{0}F}}'.format(decimal_digits)
        self.print_number_str(format_string.format(value), justify_right)

def dict_hash(dct):
    """Return a hash of the contents of a dictionary"""
    dct_s = json.dumps(dct, sort_keys=True)

    try:
        m = md5(dct_s)
    except TypeError:
        m = md5(dct_s.encode())

    return m.hexdigest()

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

def test3():
    """Test the multiprocess
    """
    import time
    
    p = MVisionProcess()
    p.start()
    time.sleep(5)
    p.stop()

def _tuple_repr(data):
    """Return a repr() for a list/tuple"""
    if len(data) == 1:
        return "(%s,)" % rpr(data[0])
    else:
        return "(%s)" % ", ".join([rpr(x) for x in data])

def disown(cmd):
    """Call a system command in the background,
       disown it and hide it's output."""
    subprocess.Popen(cmd,
                     stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)

def pprint(o, stream=None, indent=1, width=80, depth=None):
    """Pretty-print a Python o to a stream [default is sys.stdout]."""
    printer = PrettyPrinter(
        stream=stream, indent=indent, width=width, depth=depth)
    printer.pprint(o)

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

def to_camel_case(snake_case_string):
    """
    Convert a string from snake case to camel case. For example, "some_var" would become "someVar".

    :param snake_case_string: Snake-cased string to convert to camel case.
    :returns: Camel-cased version of snake_case_string.
    """
    parts = snake_case_string.lstrip('_').split('_')
    return parts[0] + ''.join([i.title() for i in parts[1:]])

def version_triple(tag):
    """
    returns: a triple of integers from a version tag
    """
    groups = re.match(r'v?(\d+)\.(\d+)\.(\d+)', tag).groups()
    return tuple(int(n) for n in groups)

def kill_mprocess(process):
    """kill process
    Args:
        process - Popen object for process
    """
    if process and proc_alive(process):
        process.terminate()
        process.communicate()
    return not proc_alive(process)

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

def get_lons_from_cartesian(x__, y__):
    """Get longitudes from cartesian coordinates.
    """
    return rad2deg(arccos(x__ / sqrt(x__ ** 2 + y__ ** 2))) * sign(y__)

def average_price(quantity_1, price_1, quantity_2, price_2):
    """Calculates the average price between two asset states."""
    return (quantity_1 * price_1 + quantity_2 * price_2) / \
            (quantity_1 + quantity_2)

def _from_bytes(bytes, byteorder="big", signed=False):
    """This is the same functionality as ``int.from_bytes`` in python 3"""
    return int.from_bytes(bytes, byteorder=byteorder, signed=signed)

def tick(self):
        """Add one tick to progress bar"""
        self.current += 1
        if self.current == self.factor:
            sys.stdout.write('+')
            sys.stdout.flush()
            self.current = 0

def get_input(input_func, input_str):
    """
    Get input from the user given an input function and an input string
    """
    val = input_func("Please enter your {0}: ".format(input_str))
    while not val or not len(val.strip()):
        val = input_func("You didn't enter a valid {0}, please try again: ".format(input_str))
    return val

def _set_property(self, val, *args):
        """Private method that sets the value currently of the property"""
        val = UserClassAdapter._set_property(self, val, *args)
        if val:
            Adapter._set_property(self, val, *args)
        return val

def screen_cv2(self):
        """cv2 Image of current window screen"""
        pil_image = self.screen.convert('RGB')
        cv2_image = np.array(pil_image)
        pil_image.close()
        # Convert RGB to BGR 
        cv2_image = cv2_image[:, :, ::-1]
        return cv2_image

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

def incr(self, key, incr_by=1):
        """Increment the key by the given amount."""
        return self.database.hincrby(self.key, key, incr_by)

def read_proto_object(fobj, klass):
    """Read a block of data and parse using the given protobuf object."""
    log.debug('%s chunk', klass.__name__)
    obj = klass()
    obj.ParseFromString(read_block(fobj))
    log.debug('Header: %s', str(obj))
    return obj

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

def _prepare_proxy(self, conn):
        """
        Establish tunnel connection early, because otherwise httplib
        would improperly set Host: header to proxy's IP:port.
        """
        conn.set_tunnel(self._proxy_host, self.port, self.proxy_headers)
        conn.connect()

def best(self):
        """
        Returns the element with the highest probability.
        """
        b = (-1e999999, None)
        for k, c in iteritems(self.counts):
            b = max(b, (c, k))
        return b[1]

def map(cls, iterable, func, *a, **kw):
    """
    Iterable-first replacement of Python's built-in `map()` function.
    """

    return cls(func(x, *a, **kw) for x in iterable)

def indent(self, message):
        """
        Sets the indent for standardized output
        :param message: (str)
        :return: (str)
        """
        indent = self.indent_char * self.indent_size
        return indent + message

def loadb(b):
    """Deserialize ``b`` (instance of ``bytes``) to a Python object."""
    assert isinstance(b, (bytes, bytearray))
    return std_json.loads(b.decode('utf-8'))

def assert_called_once(_mock_self):
        """assert that the mock was called only once.
        """
        self = _mock_self
        if not self.call_count == 1:
            msg = ("Expected '%s' to have been called once. Called %s times." %
                   (self._mock_name or 'mock', self.call_count))
            raise AssertionError(msg)

def unique_inverse(item_list):
    """
    Like np.unique(item_list, return_inverse=True)
    """
    import utool as ut
    unique_items = ut.unique(item_list)
    inverse = list_alignment(unique_items, item_list)
    return unique_items, inverse

def generate_random_string(chars=7):
    """

    :param chars:
    :return:
    """
    return u"".join(random.sample(string.ascii_letters * 2 + string.digits, chars))

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

def cli_command_quit(self, msg):
        """\
        kills the child and exits
        """
        if self.state == State.RUNNING and self.sprocess and self.sprocess.proc:
            self.sprocess.proc.kill()
        else:
            sys.exit(0)

def positive_integer(anon, obj, field, val):
    """
    Returns a random positive integer (for a Django PositiveIntegerField)
    """
    return anon.faker.positive_integer(field=field)

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

def positive_integer(anon, obj, field, val):
    """
    Returns a random positive integer (for a Django PositiveIntegerField)
    """
    return anon.faker.positive_integer(field=field)

def bytesize(arr):
    """
    Returns the memory byte size of a Numpy array as an integer.
    """
    byte_size = np.prod(arr.shape) * np.dtype(arr.dtype).itemsize
    return byte_size

def is_integer(obj):
    """Is this an integer.

    :param object obj:
    :return:
    """
    if PYTHON3:
        return isinstance(obj, int)
    return isinstance(obj, (int, long))

def url_read_text(url, verbose=True):
    r"""
    Directly reads text data from url
    """
    data = url_read(url, verbose)
    text = data.decode('utf8')
    return text

def _get_column_types(self, data):
        """Get a list of the data types for each column in *data*."""
        columns = list(zip_longest(*data))
        return [self._get_column_type(column) for column in columns]

def _fast_read(self, infile):
        """Function for fast reading from sensor files."""
        infile.seek(0)
        return(int(infile.read().decode().strip()))

def get_var_type(self, name):
        """
        Return type string, compatible with numpy.
        """
        name = create_string_buffer(name)
        type_ = create_string_buffer(MAXSTRLEN)
        self.library.get_var_type.argtypes = [c_char_p, c_char_p]
        self.library.get_var_type(name, type_)
        return type_.value

def be_array_from_bytes(fmt, data):
    """
    Reads an array from bytestring with big-endian data.
    """
    arr = array.array(str(fmt), data)
    return fix_byteorder(arr)

def paginate(self, request, offset=0, limit=None):
        """Paginate queryset."""
        return self.collection.offset(offset).limit(limit), self.collection.count()

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

def _load_data(filepath):
  """Loads the images and latent values into Numpy arrays."""
  with h5py.File(filepath, "r") as h5dataset:
    image_array = np.array(h5dataset["images"])
    # The 'label' data set in the hdf5 file actually contains the float values
    # and not the class labels.
    values_array = np.array(h5dataset["labels"])
  return image_array, values_array

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

def load_graph_from_rdf(fname):
    """ reads an RDF file into a graph """
    print("reading RDF from " + fname + "....")
    store = Graph()
    store.parse(fname, format="n3")
    print("Loaded " + str(len(store)) + " tuples")
    return store

def read_numpy(fh, byteorder, dtype, count, offsetsize):
    """Read tag data from file and return as numpy array."""
    dtype = 'b' if dtype[-1] == 's' else byteorder+dtype[-1]
    return fh.read_array(dtype, count)

def _fast_read(self, infile):
        """Function for fast reading from sensor files."""
        infile.seek(0)
        return(int(infile.read().decode().strip()))

def map(cls, iterable, func, *a, **kw):
    """
    Iterable-first replacement of Python's built-in `map()` function.
    """

    return cls(func(x, *a, **kw) for x in iterable)

def read(fname):
    """Quick way to read a file content."""
    content = None
    with open(os.path.join(here, fname)) as f:
        content = f.read()
    return content

def _fast_read(self, infile):
        """Function for fast reading from sensor files."""
        infile.seek(0)
        return(int(infile.read().decode().strip()))

def INIT_LIST_EXPR(self, cursor):
        """Returns a list of literal values."""
        values = [self.parse_cursor(child)
                  for child in list(cursor.get_children())]
        return values

def read_stdin():
    """ Read text from stdin, and print a helpful message for ttys. """
    if sys.stdin.isatty() and sys.stdout.isatty():
        print('\nReading from stdin until end of file (Ctrl + D)...')

    return sys.stdin.read()

def _generate_plane(normal, origin):
    """ Returns a vtk.vtkPlane """
    plane = vtk.vtkPlane()
    plane.SetNormal(normal[0], normal[1], normal[2])
    plane.SetOrigin(origin[0], origin[1], origin[2])
    return plane

def parse_comments_for_file(filename):
    """
    Return a list of all parsed comments in a file.  Mostly for testing &
    interactive use.
    """
    return [parse_comment(strip_stars(comment), next_line)
            for comment, next_line in get_doc_comments(read_file(filename))]

def run(self, forever=True):
        """start the bot"""
        loop = self.create_connection()
        self.add_signal_handlers()
        if forever:
            loop.run_forever()

def ReadTif(tifFile):
        """Reads a tif file to a 2D NumPy array"""
        img = Image.open(tifFile)
        img = np.array(img)
        return img

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

def compute_depth(self):
        """
        Recursively computes true depth of the subtree. Should only
        be needed for debugging. Unless something is wrong, the
        depth field should reflect the correct depth of the subtree.
        """
        left_depth = self.left_node.compute_depth() if self.left_node else 0
        right_depth = self.right_node.compute_depth() if self.right_node else 0
        return 1 + max(left_depth, right_depth)

def fail(message=None, exit_status=None):
    """Prints the specified message and exits the program with the specified
    exit status.

    """
    print('Error:', message, file=sys.stderr)
    sys.exit(exit_status or 1)

def redirect_output(fileobj):
    """Redirect standard out to file."""
    old = sys.stdout
    sys.stdout = fileobj
    try:
        yield fileobj
    finally:
        sys.stdout = old

def convert_to_yaml(
        name, value, indentation, indexOfColon, show_multi_line_character):
    """converts a value list into yaml syntax
    :param name: name of object (example: phone)
    :type name: str
    :param value: object contents
    :type value: str, list(str), list(list(str))
    :param indentation: indent all by number of spaces
    :type indentation: int
    :param indexOfColon: use to position : at the name string (-1 for no space)
    :type indexOfColon: int
    :param show_multi_line_character: option to hide "|"
    :type show_multi_line_character: boolean
    :returns: yaml formatted string array of name, value pair
    :rtype: list(str)
    """
    strings = []
    if isinstance(value, list):
        # special case for single item lists:
        if len(value) == 1 \
                and isinstance(value[0], str):
            # value = ["string"] should not be converted to
            # name:
            #   - string
            # but to "name: string" instead
            value = value[0]
        elif len(value) == 1 \
                and isinstance(value[0], list) \
                and len(value[0]) == 1 \
                and isinstance(value[0][0], str):
            # same applies to value = [["string"]]
            value = value[0][0]
    if isinstance(value, str):
        strings.append("%s%s%s: %s" % (
            ' ' * indentation, name, ' ' * (indexOfColon-len(name)),
            indent_multiline_string(value, indentation+4,
                                    show_multi_line_character)))
    elif isinstance(value, list):
        strings.append("%s%s%s: " % (
            ' ' * indentation, name, ' ' * (indexOfColon-len(name))))
        for outer in value:
            # special case for single item sublists
            if isinstance(outer, list) \
                    and len(outer) == 1 \
                    and isinstance(outer[0], str):
                # outer = ["string"] should not be converted to
                # -
                #   - string
                # but to "- string" instead
                outer = outer[0]
            if isinstance(outer, str):
                strings.append("%s- %s" % (
                    ' ' * (indentation+4), indent_multiline_string(
                        outer, indentation+8, show_multi_line_character)))
            elif isinstance(outer, list):
                strings.append("%s- " % (' ' * (indentation+4)))
                for inner in outer:
                    if isinstance(inner, str):
                        strings.append("%s- %s" % (
                            ' ' * (indentation+8), indent_multiline_string(
                                inner, indentation+12,
                                show_multi_line_character)))
    return strings

def __exit__(self, *args):
        """Redirect stdout back to the original stdout."""
        sys.stdout = self._orig
        self._devnull.close()

def main(idle):
    """Any normal python logic which runs a loop. Can take arguments."""
    while True:

        LOG.debug("Sleeping for {0} seconds.".format(idle))
        time.sleep(idle)

def disable_stdout_buffering():
    """This turns off stdout buffering so that outputs are immediately
    materialized and log messages show up before the program exits"""
    stdout_orig = sys.stdout
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
    # NOTE(brandyn): This removes the original stdout
    return stdout_orig

def safe_exit(output):
    """exit without breaking pipes."""
    try:
        sys.stdout.write(output)
        sys.stdout.flush()
    except IOError:
        pass

def _internal_kv_get(key):
    """Fetch the value of a binary key."""

    worker = ray.worker.get_global_worker()
    if worker.mode == ray.worker.LOCAL_MODE:
        return _local.get(key)

    return worker.redis_client.hget(key, "value")

def comment (self, s, **args):
        """Write DOT comment."""
        self.write(u"// ")
        self.writeln(s=s, **args)

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

def to_capitalized_camel_case(snake_case_string):
    """
    Convert a string from snake case to camel case with the first letter capitalized. For example, "some_var"
    would become "SomeVar".

    :param snake_case_string: Snake-cased string to convert to camel case.
    :returns: Camel-cased version of snake_case_string.
    """
    parts = snake_case_string.split('_')
    return ''.join([i.title() for i in parts])

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

def disable(self):
        """
        Disable the button, if in non-expert mode;
        unset its activity flag come-what-may.
        """
        if not self._expert:
            self.config(state='disable')
        self._active = False

def __setitem__(self, field, value):
        """ :see::meth:RedisMap.__setitem__ """
        return self._client.hset(self.key_prefix, field, self._dumps(value))

def normalize_matrix(matrix):
  """Fold all values of the matrix into [0, 1]."""
  abs_matrix = np.abs(matrix.copy())
  return abs_matrix / abs_matrix.max()

def construct_from_string(cls, string):
        """
        Construction from a string, raise a TypeError if not
        possible
        """
        if string == cls.name:
            return cls()
        raise TypeError("Cannot construct a '{}' from "
                        "'{}'".format(cls, string))

def as_float_array(a):
    """View the quaternion array as an array of floats

    This function is fast (of order 1 microsecond) because no data is
    copied; the returned quantity is just a "view" of the original.

    The output view has one more dimension (of size 4) than the input
    array, but is otherwise the same shape.

    """
    return np.asarray(a, dtype=np.quaternion).view((np.double, 4))

def FindMethodByName(self, name):
    """Searches for the specified method, and returns its descriptor."""
    for method in self.methods:
      if name == method.name:
        return method
    return None

def colorbar(height, length, colormap):
    """Return the channels of a colorbar.
    """
    cbar = np.tile(np.arange(length) * 1.0 / (length - 1), (height, 1))
    cbar = (cbar * (colormap.values.max() - colormap.values.min())
            + colormap.values.min())

    return colormap.colorize(cbar)

def parse_domain(url):
    """ parse the domain from the url """
    domain_match = lib.DOMAIN_REGEX.match(url)
    if domain_match:
        return domain_match.group()

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

def subn_filter(s, find, replace, count=0):
    """A non-optimal implementation of a regex filter"""
    return re.gsub(find, replace, count, s)

def add_str(window, line_num, str):
    """ attempt to draw str on screen and ignore errors if they occur """
    try:
        window.addstr(line_num, 0, str)
    except curses.error:
        pass

def _cached_search_compile(pattern, re_verbose, re_version, pattern_type):
    """Cached search compile."""

    return _bregex_parse._SearchParser(pattern, re_verbose, re_version).parse()

def parse_path(path):
    """Parse path string."""
    version, project = path[1:].split('/')
    return dict(version=int(version), project=project)

def _match_space_at_line(line):
    """Return a re.match object if an empty comment was found on line."""
    regex = re.compile(r"^{0}$".format(_MDL_COMMENT))
    return regex.match(line)

def commajoin_as_strings(iterable):
    """ Join the given iterable with ',' """
    return _(u',').join((six.text_type(i) for i in iterable))

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

def colorize(string, color, *args, **kwargs):
    """
    Implements string formatting along with color specified in colorama.Fore
    """
    string = string.format(*args, **kwargs)
    return color + string + colorama.Fore.RESET

def unaccentuate(s):
    """ Replace accentuated chars in string by their non accentuated equivalent. """
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def compute_gradient(self):
        """Compute the gradient of the current model using the training set
        """
        delta = self.predict(self.X) - self.y
        return delta.dot(self.X) / len(self.X)

def strip_accents(string):
    """
    Strip all the accents from the string
    """
    return u''.join(
        (character for character in unicodedata.normalize('NFD', string)
         if unicodedata.category(character) != 'Mn'))

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

def make_file_read_only(file_path):
    """
    Removes the write permissions for the given file for owner, groups and others.

    :param file_path: The file whose privileges are revoked.
    :raise FileNotFoundError: If the given file does not exist.
    """
    old_permissions = os.stat(file_path).st_mode
    os.chmod(file_path, old_permissions & ~WRITE_PERMISSIONS)

def comment (self, s, **args):
        """Write DOT comment."""
        self.write(u"// ")
        self.writeln(s=s, **args)

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

def to_pydatetime(self):
        """
        Converts datetimeoffset object into Python's datetime.datetime object
        @return: time zone aware datetime.datetime
        """
        dt = datetime.datetime.combine(self._date.to_pydate(), self._time.to_pytime())
        from .tz import FixedOffsetTimezone
        return dt.replace(tzinfo=_utc).astimezone(FixedOffsetTimezone(self._offset))

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

def abfIDfromFname(fname):
    """given a filename, return the ABFs ID string."""
    fname=os.path.abspath(fname)
    basename=os.path.basename(fname)
    return os.path.splitext(basename)[0]

def str_dict(some_dict):
    """Convert dict of ascii str/unicode to dict of str, if necessary"""
    return {str(k): str(v) for k, v in some_dict.items()}

def __delitem__(self, resource):
        """Remove resource instance from internal cache"""
        self.__caches[type(resource)].pop(resource.get_cache_internal_key(), None)

def clean(some_string, uppercase=False):
    """
    helper to clean up an input string
    """
    if uppercase:
        return some_string.strip().upper()
    else:
        return some_string.strip().lower()

def cleanwrap(func):
    """ Wrapper for Zotero._cleanup
    """

    def enc(self, *args, **kwargs):
        """ Send each item to _cleanup() """
        return (func(self, item, **kwargs) for item in args)

    return enc

def dict_merge(set1, set2):
    """Joins two dictionaries."""
    return dict(list(set1.items()) + list(set2.items()))

def clean(self, text):
        """Remove all unwanted characters from text."""
        return ''.join([c for c in text if c in self.alphabet])

def dictmerge(x, y):
    """
    merge two dictionaries
    """
    z = x.copy()
    z.update(y)
    return z

def mpl_outside_legend(ax, **kwargs):
    """ Places a legend box outside a matplotlib Axes instance. """
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), **kwargs)

def remove_from_lib(self, name):
        """ Remove an object from the bin folder. """
        self.__remove_path(os.path.join(self.root_dir, "lib", name))

def mpl_outside_legend(ax, **kwargs):
    """ Places a legend box outside a matplotlib Axes instance. """
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), **kwargs)

def slugify(string):
    """
    Removes non-alpha characters, and converts spaces to hyphens. Useful for making file names.


    Source: http://stackoverflow.com/questions/5574042/string-slugification-in-python
    """
    string = re.sub('[^\w .-]', '', string)
    string = string.replace(" ", "-")
    return string

def multiply(self, number):
        """Return a Vector as the product of the vector and a real number."""
        return self.from_list([x * number for x in self.to_list()])

def text_remove_empty_lines(text):
    """
    Whitespace normalization:

      - Strip empty lines
      - Strip trailing whitespace
    """
    lines = [ line.rstrip()  for line in text.splitlines()  if line.strip() ]
    return "\n".join(lines)

def get_property(self, filename):
        """Opens the file and reads the value"""

        with open(self.filepath(filename)) as f:
            return f.read().strip()

def remove_bad(string):
    """
    remove problem characters from string
    """
    remove = [':', ',', '(', ')', ' ', '|', ';', '\'']
    for c in remove:
        string = string.replace(c, '_')
    return string

def get_member(thing_obj, member_string):
    """Get a member from an object by (string) name"""
    mems = {x[0]: x[1] for x in inspect.getmembers(thing_obj)}
    if member_string in mems:
        return mems[member_string]

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

def get_code(module):
    """
    Compile and return a Module's code object.
    """
    fp = open(module.path)
    try:
        return compile(fp.read(), str(module.name), 'exec')
    finally:
        fp.close()

def unsort_vector(data, indices_of_increasing):
    """Upermutate 1-D data that is sorted by indices_of_increasing."""
    return numpy.array([data[indices_of_increasing.index(i)] for i in range(len(data))])

def fopenat(base_fd, path):
    """
    Does openat read-only, then does fdopen to get a file object
    """

    return os.fdopen(openat(base_fd, path, os.O_RDONLY), 'rb')

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

def _openpyxl_read_xl(xl_path: str):
    """ Use openpyxl to read an Excel file. """
    try:
        wb = load_workbook(filename=xl_path, read_only=True)
    except:
        raise
    else:
        return wb

def quote(s, unsafe='/'):
    """Pass in a dictionary that has unsafe characters as the keys, and the percent
    encoded value as the value."""
    res = s.replace('%', '%25')
    for c in unsafe:
        res = res.replace(c, '%' + (hex(ord(c)).upper())[2:])
    return res

def fopenat(base_fd, path):
    """
    Does openat read-only, then does fdopen to get a file object
    """

    return os.fdopen(openat(base_fd, path, os.O_RDONLY), 'rb')

def dashrepl(value):
    """
    Replace any non-word characters with a dash.
    """
    patt = re.compile(r'\W', re.UNICODE)
    return re.sub(patt, '-', value)

def read_proto_object(fobj, klass):
    """Read a block of data and parse using the given protobuf object."""
    log.debug('%s chunk', klass.__name__)
    obj = klass()
    obj.ParseFromString(read_block(fobj))
    log.debug('Header: %s', str(obj))
    return obj

def normalize_value(text):
    """
    This removes newlines and multiple spaces from a string.
    """
    result = text.replace('\n', ' ')
    result = re.subn('[ ]{2,}', ' ', result)[0]
    return result

def __init__(self, find, subcon):
        """Initialize."""
        Subconstruct.__init__(self, subcon)
        self.find = find

def dashrepl(value):
    """
    Replace any non-word characters with a dash.
    """
    patt = re.compile(r'\W', re.UNICODE)
    return re.sub(patt, '-', value)

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

def fmt_subst(regex, subst):
    """Replace regex with string."""
    return lambda text: re.sub(regex, subst, text) if text else text

def _gzip(self, response):
        """Apply gzip compression to a response."""
        bytesio = six.BytesIO()
        with gzip.GzipFile(fileobj=bytesio, mode='w') as gz:
            gz.write(response)
        return bytesio.getvalue()

def replace_tab_indent(s, replace="    "):
    """
    :param str s: string with tabs
    :param str replace: e.g. 4 spaces
    :rtype: str
    """
    prefix = get_indent_prefix(s)
    return prefix.replace("\t", replace) + s[len(prefix):]

def convert_time_string(date_str):
    """ Change a date string from the format 2018-08-15T23:55:17 into a datetime object """
    dt, _, _ = date_str.partition(".")
    dt = datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S")
    return dt

def get(key, default=None):
    """ return the key from the request
    """
    data = get_form() or get_query_string()
    return data.get(key, default)

def stringToDate(fmt="%Y-%m-%d"):
    """returns a function to convert a string to a datetime.date instance
    using the formatting string fmt as in time.strftime"""
    import time
    import datetime
    def conv_func(s):
        return datetime.date(*time.strptime(s,fmt)[:3])
    return conv_func

def _get_url(url):
    """Retrieve requested URL"""
    try:
        data = HTTP_SESSION.get(url, stream=True)
        data.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise FetcherException(exc)

    return data

def eval_script(self, expr):
    """ Evaluates a piece of Javascript in the context of the current page and
    returns its value. """
    ret = self.conn.issue_command("Evaluate", expr)
    return json.loads("[%s]" % ret)[0]

def set_basic_auth(self, username, password):
        """
        Set authenatication.
        """
        from requests.auth import HTTPBasicAuth
        self.auth = HTTPBasicAuth(username, password)
        return self

def set_json_item(key, value):
    """ manipulate json data on the fly
    """
    data = get_json()
    data[key] = value

    request = get_request()
    request["BODY"] = json.dumps(data)

def localeselector():
    """Default locale selector used in abilian applications."""
    # if a user is logged in, use the locale from the user settings
    user = getattr(g, "user", None)
    if user is not None:
        locale = getattr(user, "locale", None)
        if locale:
            return locale

    # Otherwise, try to guess the language from the user accept header the browser
    # transmits.  By default we support en/fr. The best match wins.
    return request.accept_languages.best_match(
        current_app.config["BABEL_ACCEPT_LANGUAGES"]
    )

def xml(cls, res, *args, **kwargs):
        """Parses XML from a response."""
        return parse_xml(res.text, *args, **kwargs)

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

def aug_sysargv(cmdstr):
    """ DEBUG FUNC modify argv to look like you ran a command """
    import shlex
    argv = shlex.split(cmdstr)
    sys.argv.extend(argv)

def save_session_to_file(self, sessionfile):
        """Not meant to be used directly, use :meth:`Instaloader.save_session_to_file`."""
        pickle.dump(requests.utils.dict_from_cookiejar(self._session.cookies), sessionfile)

def load_from_file(cls, filename_prefix):
    """Extracts list of subwords from file."""
    filename = cls._filename(filename_prefix)
    lines, _ = cls._read_lines_from_file(filename)
    # Strip wrapping single quotes
    vocab_list = [line[1:-1] for line in lines]
    return cls(vocab_list=vocab_list)

def _wait_for_response(self):
		"""
		Wait until the user accepted or rejected the request
		"""
		while not self.server.response_code:
			time.sleep(2)
		time.sleep(5)
		self.server.shutdown()

def plot_and_save(self, **kwargs):
        """Used when the plot method defined does not create a figure nor calls save_plot
        Then the plot method has to use self.fig"""
        self.fig = pyplot.figure()
        self.plot()
        self.axes = pyplot.gca()
        self.save_plot(self.fig, self.axes, **kwargs)
        pyplot.close(self.fig)

def set_json_item(key, value):
    """ manipulate json data on the fly
    """
    data = get_json()
    data[key] = value

    request = get_request()
    request["BODY"] = json.dumps(data)

def request(method, url, **kwargs):
    """
    Wrapper for the `requests.request()` function.
    It accepts the same arguments as the original, plus an optional `retries`
    that overrides the default retry mechanism.
    """
    retries = kwargs.pop('retries', None)
    with Session(retries=retries) as session:
        return session.request(method=method, url=url, **kwargs)

def parse_cookies(self, req, name, field):
        """Pull the value from the cookiejar."""
        return core.get_value(req.COOKIES, name, field)

def read_stdin():
    """ Read text from stdin, and print a helpful message for ttys. """
    if sys.stdin.isatty() and sys.stdout.isatty():
        print('\nReading from stdin until end of file (Ctrl + D)...')

    return sys.stdin.read()

def handle_errors(resp):
    """raise a descriptive exception on a "bad request" response"""
    if resp.status_code == 400:
        raise ApiException(json.loads(resp.content).get('message'))
    return resp

def _dump_enum(self, e, top=''):
        """Dump single enum type.
        
        Keyword arguments:
        top -- top namespace
        """
        self._print()
        self._print('enum {} {{'.format(e.name))
        self.defines.append('{}.{}'.format(top,e.name))
        
        self.tabs+=1
        for v in e.value:
            self._print('{} = {};'.format(v.name, v.number))
        self.tabs-=1
        self._print('}')

async def handle(self, record):
        """
        Call the handlers for the specified record.

        This method is used for unpickled records received from a socket, as
        well as those created locally. Logger-level filtering is applied.
        """
        if (not self.disabled) and self.filter(record):
            await self.callHandlers(record)

def __print_table(table):
        """Print a list in tabular format
        Based on https://stackoverflow.com/a/8356620"""

        col_width = [max(len(x) for x in col) for col in zip(*table)]
        print("| " + " | ".join("{:{}}".format(x, col_width[i])
                                for i, x in enumerate(table[0])) + " |")
        print("| " + " | ".join("{:{}}".format('-' * col_width[i], col_width[i])
                                for i, x in enumerate(table[0])) + " |")
        for line in table[1:]:
            print("| " + " | ".join("{:{}}".format(x, col_width[i])
                                    for i, x in enumerate(line)) + " |")

def index(m, val):
    """
    Return the indices of all the ``val`` in ``m``
    """
    mm = np.array(m)
    idx_tuple = np.where(mm == val)
    idx = idx_tuple[0].tolist()

    return idx

def _get_column_types(self, data):
        """Get a list of the data types for each column in *data*."""
        columns = list(zip_longest(*data))
        return [self._get_column_type(column) for column in columns]

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

def typename(obj):
    """Returns the type of obj as a string. More descriptive and specific than
    type(obj), and safe for any object, unlike __class__."""
    if hasattr(obj, '__class__'):
        return getattr(obj, '__class__').__name__
    else:
        return type(obj).__name__

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

def print_item_with_children(ac, classes, level):
    """ Print the given item and all children items """
    print_row(ac.id, ac.name, f"{ac.allocation:,.2f}", level)
    print_children_recursively(classes, ac, level + 1)

def closest(xarr, val):
    """ Return the index of the closest in xarr to value val """
    idx_closest = np.argmin(np.abs(np.array(xarr) - val))
    return idx_closest

def print_failure_message(message):
    """Print a message indicating failure in red color to STDERR.

    :param message: the message to print
    :type message: :class:`str`
    """
    try:
        import colorama

        print(colorama.Fore.RED + message + colorama.Fore.RESET,
              file=sys.stderr)
    except ImportError:
        print(message, file=sys.stderr)

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

def register_view(self, view):
        """Register callbacks for button press events and selection changed"""
        super(ListViewController, self).register_view(view)
        self.tree_view.connect('button_press_event', self.mouse_click)

def rewindbody(self):
        """Rewind the file to the start of the body (if seekable)."""
        if not self.seekable:
            raise IOError, "unseekable file"
        self.fp.seek(self.startofbody)

def _trim(self, somestr):
        """ Trim left-right given string """
        tmp = RE_LSPACES.sub("", somestr)
        tmp = RE_TSPACES.sub("", tmp)
        return str(tmp)

def copy_user_agent_from_driver(self):
        """ Updates requests' session user-agent with the driver's user agent

        This method will start the browser process if its not already running.
        """
        selenium_user_agent = self.driver.execute_script("return navigator.userAgent;")
        self.headers.update({"user-agent": selenium_user_agent})

def movingaverage(arr, window):
    """
    Calculates the moving average ("rolling mean") of an array
    of a certain window size.
    """
    m = np.ones(int(window)) / int(window)
    return scipy.ndimage.convolve1d(arr, m, axis=0, mode='reflect')

def raise_(exception=ABSENT, *args, **kwargs):
    """Raise (or re-raises) an exception.

    :param exception: Exception object to raise, or an exception class.
                      In the latter case, remaining arguments are passed
                      to the exception's constructor.
                      If omitted, the currently handled exception is re-raised.
    """
    if exception is ABSENT:
        raise
    else:
        if inspect.isclass(exception):
            raise exception(*args, **kwargs)
        else:
            if args or kwargs:
                raise TypeError("can't pass arguments along with "
                                "exception object to raise_()")
            raise exception

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

def rand_elem(seq, n=None):
    """returns a random element from seq n times. If n is None, it continues indefinitly"""
    return map(random.choice, repeat(seq, n) if n is not None else repeat(seq))

def round_to_float(number, precision):
    """Round a float to a precision"""
    rounded = Decimal(str(floor((number + precision / 2) // precision))
                      ) * Decimal(str(precision))
    return float(rounded)

def round_to_float(number, precision):
    """Round a float to a precision"""
    rounded = Decimal(str(floor((number + precision / 2) // precision))
                      ) * Decimal(str(precision))
    return float(rounded)

def load_data(filename):
    """
    :rtype : numpy matrix
    """
    data = pandas.read_csv(filename, header=None, delimiter='\t', skiprows=9)
    return data.as_matrix()

def round_figures(x, n):
    """Returns x rounded to n significant figures."""
    return round(x, int(n - math.ceil(math.log10(abs(x)))))

def _fast_read(self, infile):
        """Function for fast reading from sensor files."""
        infile.seek(0)
        return(int(infile.read().decode().strip()))

def call_and_exit(self, cmd, shell=True):
        """Run the *cmd* and exit with the proper exit code."""
        sys.exit(subprocess.call(cmd, shell=shell))

def read_numpy(fd, byte_order, dtype, count):
    """Read tag data from file and return as numpy array."""
    return numpy.fromfile(fd, byte_order+dtype[-1], count)

def get_pull_request(project, num, auth=False):
    """get pull request info  by number
    """
    url = "https://api.github.com/repos/{project}/pulls/{num}".format(project=project, num=num)
    if auth:
        header = make_auth_header()
    else:
        header = None
    response = requests.get(url, headers=header)
    response.raise_for_status()
    return json.loads(response.text, object_hook=Obj)

def unpickle_file(picklefile, **kwargs):
    """Helper function to unpickle data from `picklefile`."""
    with open(picklefile, 'rb') as f:
        return pickle.load(f, **kwargs)

def test():  # pragma: no cover
    """Execute the unit tests on an installed copy of unyt.

    Note that this function requires pytest to run. If pytest is not
    installed this function will raise ImportError.
    """
    import pytest
    import os

    pytest.main([os.path.dirname(os.path.abspath(__file__))])

def html_to_text(content):
    """ Converts html content to plain text """
    text = None
    h2t = html2text.HTML2Text()
    h2t.ignore_links = False
    text = h2t.handle(content)
    return text

def list_blobs(self, prefix=''):
    """Lists names of all blobs by their prefix."""
    return [b.name for b in self.bucket.list_blobs(prefix=prefix)]

def read_numpy(fd, byte_order, dtype, count):
    """Read tag data from file and return as numpy array."""
    return numpy.fromfile(fd, byte_order+dtype[-1], count)

def pickle_save(thing,fname):
    """save something to a pickle file"""
    pickle.dump(thing, open(fname,"wb"),pickle.HIGHEST_PROTOCOL)
    return thing

def get_lines(handle, line):
    """
    Get zero-indexed line from an open file-like.
    """
    for i, l in enumerate(handle):
        if i == line:
            return l

def save(self, *args, **kwargs):
        """Saves an animation

        A wrapper around :meth:`matplotlib.animation.Animation.save`
        """
        self.timeline.index -= 1  # required for proper starting point for save
        self.animation.save(*args, **kwargs)

def lengths_offsets(value):
    """Split the given comma separated value to multiple integer values. """
    values = []
    for item in value.split(','):
        item = int(item)
        values.append(item)
    return values

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

def created_today(self):
        """Return True if created today."""
        if self.datetime.date() == datetime.today().date():
            return True
        return False

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

def unaccentuate(s):
    """ Replace accentuated chars in string by their non accentuated equivalent. """
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

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

def delete_all_eggs(self):
        """ delete all the eggs in the directory specified """
        path_to_delete = os.path.join(self.egg_directory, "lib", "python")
        if os.path.exists(path_to_delete):
            shutil.rmtree(path_to_delete)

def save_session_to_file(self, sessionfile):
        """Not meant to be used directly, use :meth:`Instaloader.save_session_to_file`."""
        pickle.dump(requests.utils.dict_from_cookiejar(self._session.cookies), sessionfile)

def pop():
        """Remove instance from instance list"""
        pid = os.getpid()
        thread = threading.current_thread()
        Wdb._instances.pop((pid, thread))

def save(variable, filename):
    """Save variable on given path using Pickle
    
    Args:
        variable: what to save
        path (str): path of the output
    """
    fileObj = open(filename, 'wb')
    pickle.dump(variable, fileObj)
    fileObj.close()

def unique(list):
    """ Returns a copy of the list without duplicates.
    """
    unique = []; [unique.append(x) for x in list if x not in unique]
    return unique

def save(self, fname):
        """ Saves the dictionary in json format
        :param fname: file to save to
        """
        with open(fname, 'wb') as f:
            json.dump(self, f)

def fast_distinct(self):
        """
        Because standard distinct used on the all fields are very slow and works only with PostgreSQL database
        this method provides alternative to the standard distinct method.
        :return: qs with unique objects
        """
        return self.model.objects.filter(pk__in=self.values_list('pk', flat=True))

def isString(s):
    """Convenience method that works with all 2.x versions of Python
    to determine whether or not something is stringlike."""
    try:
        return isinstance(s, unicode) or isinstance(s, basestring)
    except NameError:
        return isinstance(s, str)

def remove_dups(seq):
    """remove duplicates from a sequence, preserving order"""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def plot_target(target, ax):
    """Ajoute la target au plot"""
    ax.scatter(target[0], target[1], target[2], c="red", s=80)

def kill(self):
        """Kill the browser.

        This is useful when the browser is stuck.
        """
        if self.process:
            self.process.kill()
            self.process.wait()

def _unjsonify(x, isattributes=False):
    """Convert JSON string to an ordered defaultdict."""
    if isattributes:
        obj = json.loads(x)
        return dict_class(obj)
    return json.loads(x)

def delete_s3_bucket(client, resource):
    """Delete an S3 bucket

    This function will try to delete an S3 bucket

    Args:
        client (:obj:`boto3.session.Session.client`): A boto3 client object
        resource (:obj:`Resource`): The resource object to terminate

    Returns:
        `ActionStatus`
    """

    if dbconfig.get('enable_delete_s3_buckets', NS_AUDITOR_REQUIRED_TAGS, False):
        client.delete_bucket(Bucket=resource.id)
    return ActionStatus.SUCCEED, resource.metrics()

def escape_link(url):
    """Remove dangerous URL schemes like javascript: and escape afterwards."""
    lower_url = url.lower().strip('\x00\x1a \n\r\t')
    for scheme in _scheme_blacklist:
        if lower_url.startswith(scheme):
            return ''
    return escape(url, quote=True, smart_amp=False)

def safe_exit(output):
    """exit without breaking pipes."""
    try:
        sys.stdout.write(output)
        sys.stdout.flush()
    except IOError:
        pass

async def send_files_preconf(filepaths, config_path=CONFIG_PATH):
    """Send files using the config.ini settings.

    Args:
        filepaths (list(str)): A list of filepaths.
    """
    config = read_config(config_path)
    subject = "PDF files from pdfebc"
    message = ""
    await send_with_attachments(subject, message, filepaths, config)

def remove_item(self, item):
        """
        Remove (and un-index) an object

        :param item: object to remove
        :type item: alignak.objects.item.Item
        :return: None
        """
        self.unindex_item(item)
        self.items.pop(item.uuid, None)

def _set_scroll_v(self, *args):
        """Scroll both categories Canvas and scrolling container"""
        self._canvas_categories.yview(*args)
        self._canvas_scroll.yview(*args)

def _sanitize(text):
    """Return sanitized Eidos text field for human readability."""
    d = {'-LRB-': '(', '-RRB-': ')'}
    return re.sub('|'.join(d.keys()), lambda m: d[m.group(0)], text)

def time2seconds(t):
    """Returns seconds since 0h00."""
    return t.hour * 3600 + t.minute * 60 + t.second + float(t.microsecond) / 1e6

def strip_querystring(url):
    """Remove the querystring from the end of a URL."""
    p = six.moves.urllib.parse.urlparse(url)
    return p.scheme + "://" + p.netloc + p.path

def dict_from_object(obj: object):
    """Convert a object into dictionary with all of its readable attributes."""

    # If object is a dict instance, no need to convert.
    return (obj if isinstance(obj, dict)
            else {attr: getattr(obj, attr)
                  for attr in dir(obj) if not attr.startswith('_')})

def format_screen(strng):
    """Format a string for screen printing.

    This removes some latex-type format codes."""
    # Paragraph continue
    par_re = re.compile(r'\\$',re.MULTILINE)
    strng = par_re.sub('',strng)
    return strng

def count_rows(self, table_name):
        """Return the number of entries in a table by counting them."""
        self.table_must_exist(table_name)
        query = "SELECT COUNT (*) FROM `%s`" % table_name.lower()
        self.own_cursor.execute(query)
        return int(self.own_cursor.fetchone()[0])

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

def scroll_element_into_view(self):
        """Scroll element into view

        :returns: page element instance
        """
        x = self.web_element.location['x']
        y = self.web_element.location['y']
        self.driver.execute_script('window.scrollTo({0}, {1})'.format(x, y))
        return self

def _replace(self, data, replacements):
        """
        Given a list of 2-tuples (find, repl) this function performs all
        replacements on the input and returns the result.
        """
        for find, repl in replacements:
            data = data.replace(find, repl)
        return data

async def send(self, data):
        """ Add data to send queue. """
        self.writer.write(data)
        await self.writer.drain()

def reset(self):
        """Reset the instance

        - reset rows and header
        """

        self._hline_string = None
        self._row_size = None
        self._header = []
        self._rows = []

def split_elements(value):
    """Split a string with comma or space-separated elements into a list."""
    l = [v.strip() for v in value.split(',')]
    if len(l) == 1:
        l = value.split()
    return l

def reset(self):
		"""
		Resets the iterator to the start.

		Any remaining values in the current iteration are discarded.
		"""
		self.__iterator, self.__saved = itertools.tee(self.__saved)

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

def RemoveMethod(self, function):
        """
        Removes the specified function's MethodWrapper from the
        added_methods list, so we don't re-bind it when making a clone.
        """
        self.added_methods = [dm for dm in self.added_methods if not dm.method is function]

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

def mostCommonItem(lst):
    """Choose the most common item from the list, or the first item if all
    items are unique."""
    # This elegant solution from: http://stackoverflow.com/a/1518632/1760218
    lst = [l for l in lst if l]
    if lst:
        return max(set(lst), key=lst.count)
    else:
        return None

def copy_default_data_file(filename, module=None):
    """Copies file from default data directory to local directory."""
    if module is None:
        module = __get_filetypes_module()
    fullpath = get_default_data_path(filename, module=module)
    shutil.copy(fullpath, ".")

def set_logging_config(log_level, handlers):
    """Set python logging library config.

    Run this ONCE at the start of your process. It formats the python logging
    module's output.
    Defaults logging level to INFO = 20)
    """
    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(name)s:%(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=log_level,
        handlers=handlers)

def move_back(self, dt):
        """ If called after an update, the sprite can move back
        """
        self._position = self._old_position
        self.rect.topleft = self._position
        self.feet.midbottom = self.rect.midbottom

def get_local_image(self, src):
        """\
        returns the bytes of the image file on disk
        """
        return ImageUtils.store_image(self.fetcher, self.article.link_hash, src, self.config)

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

def auto():
	"""set colouring on if STDOUT is a terminal device, off otherwise"""
	try:
		Style.enabled = False
		Style.enabled = sys.stdout.isatty()
	except (AttributeError, TypeError):
		pass

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

def reseed_random(seed):
    """Reseed factory.fuzzy's random generator."""
    r = random.Random(seed)
    random_internal_state = r.getstate()
    set_random_state(random_internal_state)

def price_rounding(price, decimals=2):
    """Takes a decimal price and rounds to a number of decimal places"""
    try:
        exponent = D('.' + decimals * '0')
    except InvalidOperation:
        # Currencies with no decimal places, ex. JPY, HUF
        exponent = D()
    return price.quantize(exponent, rounding=ROUND_UP)

def __run(self):
    """Hacked run function, which installs the trace."""
    sys.settrace(self.globaltrace)
    self.__run_backup()
    self.run = self.__run_backup

def round_data(filter_data):
    """ round the data"""
    for index, _ in enumerate(filter_data):
        filter_data[index][0] = round(filter_data[index][0] / 100.0) * 100.0
    return filter_data

def _sha1_for_file(filename):
    """Return sha1 for contents of filename."""
    with open(filename, "rb") as fileobj:
        contents = fileobj.read()
        return hashlib.sha1(contents).hexdigest()

def image_load_time(self):
        """
        Returns aggregate image load time for all pages.
        """
        load_times = self.get_load_times('image')
        return round(mean(load_times), self.decimal_precision)

def _go_to_line(editor, line):
    """
    Move cursor to this line in the current buffer.
    """
    b = editor.application.current_buffer
    b.cursor_position = b.document.translate_row_col_to_index(max(0, int(line) - 1), 0)

def main(idle):
    """Any normal python logic which runs a loop. Can take arguments."""
    while True:

        LOG.debug("Sleeping for {0} seconds.".format(idle))
        time.sleep(idle)

def get_last_row(dbconn, tablename, n=1, uuid=None):
    """
    Returns the last `n` rows in the table
    """
    return fetch(dbconn, tablename, n, uuid, end=True)

def test():  # pragma: no cover
    """Execute the unit tests on an installed copy of unyt.

    Note that this function requires pytest to run. If pytest is not
    installed this function will raise ImportError.
    """
    import pytest
    import os

    pytest.main([os.path.dirname(os.path.abspath(__file__))])

def pprint(self, seconds):
        """
        Pretty Prints seconds as Hours:Minutes:Seconds.MilliSeconds

        :param seconds:  The time in seconds.
        """
        return ("%d:%02d:%02d.%03d", reduce(lambda ll, b: divmod(ll[0], b) + ll[1:], [(seconds * 1000,), 1000, 60, 60]))

def update_token_tempfile(token):
    """
    Example of function for token update
    """
    with open(tmp, 'w') as f:
        f.write(json.dumps(token, indent=4))

def process_kill(pid, sig=None):
    """Send signal to process.
    """
    sig = sig or signal.SIGTERM
    os.kill(pid, sig)

def download_file(save_path, file_url):
    """ Download file from http url link """

    r = requests.get(file_url)  # create HTTP response object

    with open(save_path, 'wb') as f:
        f.write(r.content)

    return save_path

def string_to_int( s ):
  """Convert a string of bytes into an integer, as per X9.62."""
  result = 0
  for c in s:
    if not isinstance(c, int): c = ord( c )
    result = 256 * result + c
  return result

def write_wave(path, audio, sample_rate):
    """Writes a .wav file.

    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)

def singleton_per_scope(_cls, _scope=None, _renew=False, *args, **kwargs):
    """Instanciate a singleton per scope."""

    result = None

    singletons = SINGLETONS_PER_SCOPES.setdefault(_scope, {})

    if _renew or _cls not in singletons:
        singletons[_cls] = _cls(*args, **kwargs)

    result = singletons[_cls]

    return result

def array_bytes(array):
    """ Estimates the memory of the supplied array in bytes """
    return np.product(array.shape)*np.dtype(array.dtype).itemsize

def functions(self):
        """
        A list of functions declared or defined in this module.
        """
        return [v for v in self.globals.values()
                if isinstance(v, values.Function)]

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

def getcolslice(self, blc, trc, inc=[], startrow=0, nrow=-1, rowincr=1):
        """Get a slice from a table column holding arrays.
        (see :func:`table.getcolslice`)"""
        return self._table.getcolslice(self._column, blc, trc, inc, startrow, nrow, rowincr)

def is_full_slice(obj, l):
    """
    We have a full length slice.
    """
    return (isinstance(obj, slice) and obj.start == 0 and obj.stop == l and
            obj.step is None)

def compare(dicts):
    """Compare by iteration"""

    common_members = {}
    common_keys = reduce(lambda x, y: x & y, map(dict.keys, dicts))
    for k in common_keys:
        common_members[k] = list(
            reduce(lambda x, y: x & y, [set(d[k]) for d in dicts]))

    return common_members

def shutdown(self):
        """close socket, immediately."""
        if self.sock:
            self.sock.close()
            self.sock = None
            self.connected = False

def runcode(code):
	"""Run the given code line by line with printing, as list of lines, and return variable 'ans'."""
	for line in code:
		print('# '+line)
		exec(line,globals())
	print('# return ans')
	return ans

def run(context, port):
    """ Run the Webserver/SocketIO and app
    """
    global ctx
    ctx = context
    app.run(port=port)

def region_from_segment(image, segment):
    """given a segment (rectangle) and an image, returns it's corresponding subimage"""
    x, y, w, h = segment
    return image[y:y + h, x:x + w]

def MatrixSolve(a, rhs, adj):
    """
    Matrix solve op.
    """
    return np.linalg.solve(a if not adj else _adjoint(a), rhs),

def rand_elem(seq, n=None):
    """returns a random element from seq n times. If n is None, it continues indefinitly"""
    return map(random.choice, repeat(seq, n) if n is not None else repeat(seq))

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

def set_default(self_,param_name,value):
        """
        Set the default value of param_name.

        Equivalent to setting param_name on the class.
        """
        cls = self_.cls
        setattr(cls,param_name,value)

def _write_separator(self):
        """
        Inserts a horizontal (commented) line tot the generated code.
        """
        tmp = self._page_width - ((4 * self.__indent_level) + 2)
        self._write_line('# ' + ('-' * tmp))

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

def visit_Str(self, node):
        """ Set the pythonic string type. """
        self.result[node] = self.builder.NamedType(pytype_to_ctype(str))

def as_tuple(self, value):
        """Utility function which converts lists to tuples."""
        if isinstance(value, list):
            value = tuple(value)
        return value

def onchange(self, value):
        """Called when a new DropDownItem gets selected.
        """
        log.debug('combo box. selected %s' % value)
        self.select_by_value(value)
        return (value, )

def write_color(string, name, style='normal', when='auto'):
    """ Write the given colored string to standard out. """
    write(color(string, name, style, when))

def tokenize_list(self, text):
        """
        Split a text into separate words.
        """
        return [self.get_record_token(record) for record in self.analyze(text)]

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

def partition(a, sz): 
    """splits iterables a in equal parts of size sz"""
    return [a[i:i+sz] for i in range(0, len(a), sz)]

def get_var(name, factory=None):
    """Gets a global variable given its name.

    If factory is not None and the variable is not set, factory
    is a callable that will set the variable.

    If not set, returns None.
    """
    if name not in _VARS and factory is not None:
        _VARS[name] = factory()
    return _VARS.get(name)

def tokenize_list(self, text):
        """
        Split a text into separate words.
        """
        return [self.get_record_token(record) for record in self.analyze(text)]

def set_xlimits_widgets(self, set_min=True, set_max=True):
        """Populate axis limits GUI with current plot values."""
        xmin, xmax = self.tab_plot.ax.get_xlim()
        if set_min:
            self.w.x_lo.set_text('{0}'.format(xmin))
        if set_max:
            self.w.x_hi.set_text('{0}'.format(xmax))

def parse_prefix(identifier):
    """
    Parse identifier such as a|c|le|d|li|re|or|AT4G00480.1 and return
    tuple of prefix string (separated at '|') and suffix (AGI identifier)
    """
    pf, id = (), identifier
    if "|" in identifier:
        pf, id = tuple(identifier.split('|')[:-1]), identifier.split('|')[-1]

    return pf, id

def covstr(s):
  """ convert string to int or float. """
  try:
    ret = int(s)
  except ValueError:
    ret = float(s)
  return ret

def split_len(s, length):
    """split string *s* into list of strings no longer than *length*"""
    return [s[i:i+length] for i in range(0, len(s), length)]

def round_to_int(number, precision):
    """Round a number to a precision"""
    precision = int(precision)
    rounded = (int(number) + precision / 2) // precision * precision
    return rounded

def set_ylim(self, xlims, dx, xscale, reverse=False):
        """Set y limits for plot.

        This will set the limits for the y axis
        for the specific plot.

        Args:
            ylims (len-2 list of floats): The limits for the axis.
            dy (float): Amount to increment by between the limits.
            yscale (str): Scale of the axis. Either `log` or `lin`.
            reverse (bool, optional): If True, reverse the axis tick marks. Default is False.

        """
        self._set_axis_limits('y', xlims, dx, xscale, reverse)
        return

def compile(expr, params=None):
    """
    Force compilation of expression for the SQLite target
    """
    from ibis.sql.alchemy import to_sqlalchemy

    return to_sqlalchemy(expr, dialect.make_context(params=params))

def set_left_to_right(self):
        """Set text direction left to right."""
        self.displaymode |= LCD_ENTRYLEFT
        self.write8(LCD_ENTRYMODESET | self.displaymode)

def createdb():
    """Create database tables from sqlalchemy models"""
    manager.db.engine.echo = True
    manager.db.create_all()
    set_alembic_revision()

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

def locked_delete(self):
        """Delete credentials from the SQLAlchemy datastore."""
        filters = {self.key_name: self.key_value}
        self.session.query(self.model_class).filter_by(**filters).delete()

def show(self, title=''):
        """
        Display Bloch sphere and corresponding data sets.
        """
        self.render(title=title)
        if self.fig:
            plt.show(self.fig)

def sqliteRowsToDicts(sqliteRows):
    """
    Unpacks sqlite rows as returned by fetchall
    into an array of simple dicts.

    :param sqliteRows: array of rows returned from fetchall DB call
    :return:  array of dicts, keyed by the column names.
    """
    return map(lambda r: dict(zip(r.keys(), r)), sqliteRows)

def print_error(msg):
    """ Print an error message """
    if IS_POSIX:
        print(u"%s[ERRO] %s%s" % (ANSI_ERROR, msg, ANSI_END))
    else:
        print(u"[ERRO] %s" % (msg))

def Softsign(a):
    """
    Softsign op.
    """
    return np.divide(a, np.add(np.abs(a), 1)),

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

def intty(cls):
        """ Check if we are in a tty. """
        # XXX: temporary hack until we can detect if we are in a pipe or not
        return True

        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            return True

        return False

def get_grid_spatial_dimensions(self, variable):
        """Returns (width, height) for the given variable"""

        data = self.open_dataset(self.service).variables[variable.variable]
        dimensions = list(data.dimensions)
        return data.shape[dimensions.index(variable.x_dimension)], data.shape[dimensions.index(variable.y_dimension)]

def extract_module_locals(depth=0):
    """Returns (module, locals) of the funciton `depth` frames away from the caller"""
    f = sys._getframe(depth + 1)
    global_ns = f.f_globals
    module = sys.modules[global_ns['__name__']]
    return (module, f.f_locals)

def show(self, title=''):
        """
        Display Bloch sphere and corresponding data sets.
        """
        self.render(title=title)
        if self.fig:
            plt.show(self.fig)

def column_stack_2d(data):
    """Perform column-stacking on a list of 2d data blocks."""
    return list(list(itt.chain.from_iterable(_)) for _ in zip(*data))

def info(docgraph):
    """print node and edge statistics of a document graph"""
    print networkx.info(docgraph), '\n'
    node_statistics(docgraph)
    print
    edge_statistics(docgraph)

def _std(self,x):
        """
        Compute standard deviation with ddof degrees of freedom
        """
        return np.nanstd(x.values,ddof=self._ddof)

def getTypeStr(_type):
  r"""Gets the string representation of the given type.
  """
  if isinstance(_type, CustomType):
    return str(_type)

  if hasattr(_type, '__name__'):
    return _type.__name__

  return ''

def circstd(dts, axis=2):
    """Circular standard deviation"""
    R = np.abs(np.exp(1.0j * dts).mean(axis=axis))
    return np.sqrt(-2.0 * np.log(R))

def clear_matplotlib_ticks(self, axis="both"):
        """Clears the default matplotlib ticks."""
        ax = self.get_axes()
        plotting.clear_matplotlib_ticks(ax=ax, axis=axis)

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

def _shuffle(data, idx):
    """Shuffle the data."""
    shuffle_data = []

    for idx_k, idx_v in data:
        shuffle_data.append((idx_k, mx.ndarray.array(idx_v.asnumpy()[idx], idx_v.context)))

    return shuffle_data

def _stdout_raw(self, s):
        """Writes the string to stdout"""
        print(s, end='', file=sys.stdout)
        sys.stdout.flush()

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

def stop_containers(self):
        """ Stops all containers used by this instance of the backend.
        """
        while len(self._containers):
            container = self._containers.pop()
            try:
                container.kill(signal.SIGKILL)
            except docker.errors.APIError:  # probably doesn't exist anymore
                pass

def getcolslice(self, blc, trc, inc=[], startrow=0, nrow=-1, rowincr=1):
        """Get a slice from a table column holding arrays.
        (see :func:`table.getcolslice`)"""
        return self._table.getcolslice(self._column, blc, trc, inc, startrow, nrow, rowincr)

def unit_ball_L2(shape):
  """A tensorflow variable tranfomed to be constrained in a L2 unit ball.

  EXPERIMENTAL: Do not use for adverserial examples if you need to be confident
  they are strong attacks. We are not yet confident in this code.
  """
  x = tf.Variable(tf.zeros(shape))
  return constrain_L2(x)

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

def s3(ctx, bucket_name, data_file, region):
    """Use the S3 SWAG backend."""
    if not ctx.data_file:
        ctx.data_file = data_file

    if not ctx.bucket_name:
        ctx.bucket_name = bucket_name

    if not ctx.region:
        ctx.region = region

    ctx.type = 's3'

def get_obj(ref):
    """Get object from string reference."""
    oid = int(ref)
    return server.id2ref.get(oid) or server.id2obj[oid]

def sort_by_name(self):
        """Sort list elements by name."""
        super(JSSObjectList, self).sort(key=lambda k: k.name)

def _serialize_json(obj, fp):
    """ Serialize ``obj`` as a JSON formatted stream to ``fp`` """
    json.dump(obj, fp, indent=4, default=serialize)

def get_distance(F, x):
    """Helper function for margin-based loss. Return a distance matrix given a matrix."""
    n = x.shape[0]

    square = F.sum(x ** 2.0, axis=1, keepdims=True)
    distance_square = square + square.transpose() - (2.0 * F.dot(x, x.transpose()))

    # Adding identity to make sqrt work.
    return F.sqrt(distance_square + F.array(np.identity(n)))

def str_dict(some_dict):
    """Convert dict of ascii str/unicode to dict of str, if necessary"""
    return {str(k): str(v) for k, v in some_dict.items()}

def chunked(l, n):
    """Chunk one big list into few small lists."""
    return [l[i:i + n] for i in range(0, len(l), n)]

def escapePathForShell(path):
		"""
		Escapes a filesystem path for use as a command-line argument
		"""
		if platform.system() == 'Windows':
			return '"{}"'.format(path.replace('"', '""'))
		else:
			return shellescape.quote(path)

def chunks(iterable, size=1):
    """Splits iterator in chunks."""
    iterator = iter(iterable)

    for element in iterator:
        yield chain([element], islice(iterator, size - 1))

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

def tokenize_list(self, text):
        """
        Split a text into separate words.
        """
        return [self.get_record_token(record) for record in self.analyze(text)]

def FromString(self, string):
    """Parse a bool from a string."""
    if string.lower() in ("false", "no", "n"):
      return False

    if string.lower() in ("true", "yes", "y"):
      return True

    raise TypeValueError("%s is not recognized as a boolean value." % string)

def _escape(s):
    """ Helper method that escapes parameters to a SQL query. """
    e = s
    e = e.replace('\\', '\\\\')
    e = e.replace('\n', '\\n')
    e = e.replace('\r', '\\r')
    e = e.replace("'", "\\'")
    e = e.replace('"', '\\"')
    return e

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

def normalize_array(lst):
    """Normalizes list

    :param lst: Array of floats
    :return: Normalized (in [0, 1]) input array
    """
    np_arr = np.array(lst)
    x_normalized = np_arr / np_arr.max(axis=0)
    return list(x_normalized)

def _clip(sid, prefix):
    """Clips a prefix from the beginning of a string if it exists."""
    return sid[len(prefix):] if sid.startswith(prefix) else sid

def pause(msg="Press Enter to Continue..."):
    """press to continue"""
    print('\n' + Fore.YELLOW + msg + Fore.RESET, end='')
    input()

def schunk(string, size):
    """Splits string into n sized chunks."""
    return [string[i:i+size] for i in range(0, len(string), size)]

def retry_on_signal(function):
    """Retries function until it doesn't raise an EINTR error"""
    while True:
        try:
            return function()
        except EnvironmentError, e:
            if e.errno != errno.EINTR:
                raise

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

def terminate(self):
        """Terminate the pool immediately."""
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

def c_str(string):
    """"Convert a python string to C string."""
    if not isinstance(string, str):
        string = string.decode('ascii')
    return ctypes.c_char_p(string.encode('utf-8'))

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

def delimited(items, character='|'):
    """Returns a character delimited version of the provided list as a Python string"""
    return '|'.join(items) if type(items) in (list, tuple, set) else items

def set_xlimits(self, row, column, min=None, max=None):
        """Set x-axis limits of a subplot.

        :param row,column: specify the subplot.
        :param min: minimal axis value
        :param max: maximum axis value

        """
        subplot = self.get_subplot_at(row, column)
        subplot.set_xlimits(min, max)

def transpose(table):
    """
    transpose matrix
    """
    t = []
    for i in range(0, len(table[0])):
        t.append([row[i] for row in table])
    return t

def execute_in_background(self):
        """Executes a (shell) command in the background

        :return: the process' pid
        """
        # http://stackoverflow.com/questions/1605520
        args = shlex.split(self.cmd)
        p = Popen(args)
        return p.pid

def standard_input():
    """Generator that yields lines from standard input."""
    with click.get_text_stream("stdin") as stdin:
        while stdin.readable():
            line = stdin.readline()
            if line:
                yield line.strip().encode("utf-8")

def correspond(text):
    """Communicate with the child process without closing stdin."""
    subproc.stdin.write(text)
    subproc.stdin.flush()
    return drain()

def file_empty(fp):
    """Determine if a file is empty or not."""
    # for python 2 we need to use a homemade peek()
    if six.PY2:
        contents = fp.read()
        fp.seek(0)
        return not bool(contents)

    else:
        return not fp.peek()

def datetime64_to_datetime(dt):
    """ convert numpy's datetime64 to datetime """
    dt64 = np.datetime64(dt)
    ts = (dt64 - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
    return datetime.datetime.utcfromtimestamp(ts)

def is_admin(self):
        """Is the user a system administrator"""
        return self.role == self.roles.administrator.value and self.state == State.approved

def is_identifier(string):
    """Check if string could be a valid python identifier

    :param string: string to be tested
    :returns: True if string can be a python identifier, False otherwise
    :rtype: bool
    """
    matched = PYTHON_IDENTIFIER_RE.match(string)
    return bool(matched) and not keyword.iskeyword(string)

def is_enum_type(type_):
    """ Checks if the given type is an enum type.

    :param type_: The type to check
    :return: True if the type is a enum type, otherwise False
    :rtype: bool
    """

    return isinstance(type_, type) and issubclass(type_, tuple(_get_types(Types.ENUM)))

def is_valid_email(email):
    """
    Check if email is valid
    """
    pattern = re.compile(r'[\w\.-]+@[\w\.-]+[.]\w+')
    return bool(pattern.match(email))

def get_memory(self, mode):
        """Return a smt bit vector that represents a memory location.
        """
        mem = {
            "pre": self._translator.get_memory_init(),
            "post": self._translator.get_memory_curr(),
        }

        return mem[mode]

def aug_sysargv(cmdstr):
    """ DEBUG FUNC modify argv to look like you ran a command """
    import shlex
    argv = shlex.split(cmdstr)
    sys.argv.extend(argv)

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

def aug_sysargv(cmdstr):
    """ DEBUG FUNC modify argv to look like you ran a command """
    import shlex
    argv = shlex.split(cmdstr)
    sys.argv.extend(argv)

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

def read_stdin():
    """ Read text from stdin, and print a helpful message for ttys. """
    if sys.stdin.isatty() and sys.stdout.isatty():
        print('\nReading from stdin until end of file (Ctrl + D)...')

    return sys.stdin.read()

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

def main(argv=None):
  """Run a Tensorflow model on the Iris dataset."""
  args = parse_arguments(sys.argv if argv is None else argv)

  tf.logging.set_verbosity(tf.logging.INFO)
  learn_runner.run(
      experiment_fn=get_experiment_fn(args),
      output_dir=args.job_dir)

def row_to_dict(row):
    """Convert a table row to a dictionary."""
    o = {}
    for colname in row.colnames:

        if isinstance(row[colname], np.string_) and row[colname].dtype.kind in ['S', 'U']:
            o[colname] = str(row[colname])
        else:
            o[colname] = row[colname]

    return o

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

def table_top_abs(self):
        """Returns the absolute position of table top"""
        table_height = np.array([0, 0, self.table_full_size[2]])
        return string_to_array(self.floor.get("pos")) + table_height

def walk_tree(root):
    """Pre-order depth-first"""
    yield root

    for child in root.children:
        for el in walk_tree(child):
            yield el

def chunk_list(l, n):
    """Return `n` size lists from a given list `l`"""
    return [l[i:i + n] for i in range(0, len(l), n)]

def make_slice_strings(cls, slice_key):
        """
        Converts the given slice key to start and size query parts.
        """
        start = slice_key.start
        size = slice_key.stop - start
        return (str(start), str(size))

def is_callable_tag(tag):
    """ Determine whether :tag: is a valid callable string tag.

    String is assumed to be valid callable if it starts with '{{'
    and ends with '}}'.

    :param tag: String name of tag.
    """
    return (isinstance(tag, six.string_types) and
            tag.strip().startswith('{{') and
            tag.strip().endswith('}}'))

def list_string_to_dict(string):
    """Inputs ``['a', 'b', 'c']``, returns ``{'a': 0, 'b': 1, 'c': 2}``."""
    dictionary = {}
    for idx, c in enumerate(string):
        dictionary.update({c: idx})
    return dictionary

def is_unix_style(flags):
    """Check if we should use Unix style."""

    return (util.platform() != "windows" or (not bool(flags & REALPATH) and get_case(flags))) and not flags & _FORCEWIN

def from_array(cls, arr):
        """Convert a structured NumPy array into a Table."""
        return cls().with_columns([(f, arr[f]) for f in arr.dtype.names])

def afx_small():
  """Small transformer model with small batch size for fast step times."""
  hparams = transformer.transformer_tpu()
  hparams.filter_size = 1024
  hparams.num_heads = 4
  hparams.num_hidden_layers = 3
  hparams.batch_size = 512
  return hparams

def _array2cstr(arr):
    """ Serializes a numpy array to a compressed base64 string """
    out = StringIO()
    np.save(out, arr)
    return b64encode(out.getvalue())

def get_example_features(example):
  """Returns the non-sequence features from the provided example."""
  return (example.features.feature if isinstance(example, tf.train.Example)
          else example.context.feature)

def _to_java_object_rdd(rdd):
    """ Return a JavaRDD of Object by unpickling

    It will convert each Python object into Java object by Pyrolite, whenever the
    RDD is serialized in batch or not.
    """
    rdd = rdd._reserialize(AutoBatchedSerializer(PickleSerializer()))
    return rdd.ctx._jvm.org.apache.spark.mllib.api.python.SerDe.pythonToJava(rdd._jrdd, True)

def _is_path(s):
    """Return whether an object is a path."""
    if isinstance(s, string_types):
        try:
            return op.exists(s)
        except (OSError, ValueError):
            return False
    else:
        return False

def _safe_db(num, den):
    """Properly handle the potential +Inf db SIR instead of raising a
    RuntimeWarning.
    """
    if den == 0:
        return np.inf
    return 10 * np.log10(num / den)

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

def log(x):
    """
    Natural logarithm
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.log(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.log(x)

def isnumber(*args):
    """Checks if value is an integer, long integer or float.

    NOTE: Treats booleans as numbers, where True=1 and False=0.
    """
    return all(map(lambda c: isinstance(c, int) or isinstance(c, float), args))

def update_screen(self):
        """Refresh the screen. You don't need to override this except to update only small portins of the screen."""
        self.clock.tick(self.FPS)
        pygame.display.update()

def mock_add_spec(self, spec, spec_set=False):
        """Add a spec to a mock. `spec` can either be an object or a
        list of strings. Only attributes on the `spec` can be fetched as
        attributes from the mock.

        If `spec_set` is True then only attributes on the spec can be set."""
        self._mock_add_spec(spec, spec_set)
        self._mock_set_magics()

def test(nose_argsuments):
    """ Run application tests """
    from nose import run

    params = ['__main__', '-c', 'nose.ini']
    params.extend(nose_argsuments)
    run(argv=params)

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

def assert_valid_input(cls, tag):
        """Check if valid input tag or document."""

        # Fail on unexpected types.
        if not cls.is_tag(tag):
            raise TypeError("Expected a BeautifulSoup 'Tag', but instead recieved type {}".format(type(tag)))

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

def reseed_random(seed):
    """Reseed factory.fuzzy's random generator."""
    r = random.Random(seed)
    random_internal_state = r.getstate()
    set_random_state(random_internal_state)

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

def stop(self):
    """ Stops the playing thread and close """
    with self.lock:
      self.halting = True
      self.go.clear()

def _save_file(self, filename, contents):
        """write the html file contents to disk"""
        with open(filename, 'w') as f:
            f.write(contents)

def shutdown(self):
        """
        shutdown: to be run by atexit handler. All open connection are closed.
        """
        self.run_clean_thread = False
        self.cleanup(True)
        if self.cleaner_thread.isAlive():
            self.cleaner_thread.join()

def part(z, s):
    r"""Get the real or imaginary part of a complex number."""
    if sage_included:
        if s == 1: return np.real(z)
        elif s == -1: return np.imag(z)
        elif s == 0:
            return z
    else:
        if s == 1: return z.real
        elif s == -1: return z.imag
        elif s == 0: return z

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

def printheader(h=None):
    """Print the header for the CSV table."""
    writer = csv.writer(sys.stdout)
    writer.writerow(header_fields(h))

def extract_zip(zip_path, target_folder):
    """
    Extract the content of the zip-file at `zip_path` into `target_folder`.
    """
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target_folder)

def date_to_timestamp(date):
    """
        date to unix timestamp in milliseconds
    """
    date_tuple = date.timetuple()
    timestamp = calendar.timegm(date_tuple) * 1000
    return timestamp

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

def delete(self, id):
        """
        Deletes an "object" (line, triangle, image, etc) from the drawing.

        :param int id:
            The id of the object.
        """
        if id in self._images.keys():
            del self._images[id]
        self.tk.delete(id)

def _format_title_string(self, title_string):
        """ format mpv's title """
        return self._title_string_format_text_tag(title_string.replace(self.icy_tokkens[0], self.icy_title_prefix))

def SegmentMax(a, ids):
    """
    Segmented max op.
    """
    func = lambda idxs: np.amax(a[idxs], axis=0)
    return seg_map(func, a, ids),

def on_source_directory_chooser_clicked(self):
        """Autoconnect slot activated when tbSourceDir is clicked."""

        title = self.tr('Set the source directory for script and scenario')
        self.choose_directory(self.source_directory, title)

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

def set_scrollregion(self, event=None):
        """ Set the scroll region on the canvas"""
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

def column_exists(cr, table, column):
    """ Check whether a certain column exists """
    cr.execute(
        'SELECT count(attname) FROM pg_attribute '
        'WHERE attrelid = '
        '( SELECT oid FROM pg_class WHERE relname = %s ) '
        'AND attname = %s',
        (table, column))
    return cr.fetchone()[0] == 1

def on_source_directory_chooser_clicked(self):
        """Autoconnect slot activated when tbSourceDir is clicked."""

        title = self.tr('Set the source directory for script and scenario')
        self.choose_directory(self.source_directory, title)

def check_create_folder(filename):
    """Check if the folder exisits. If not, create the folder"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

def restore_scrollbar_position(self):
        """Restoring scrollbar position after main window is visible"""
        scrollbar_pos = self.get_option('scrollbar_position', None)
        if scrollbar_pos is not None:
            self.explorer.treewidget.set_scrollbar_position(scrollbar_pos)

def matching_line(lines, keyword):
    """ Returns the first matching line in a list of lines.
    @see match()
    """
    for line in lines:
        matching = match(line,keyword)
        if matching != None:
            return matching
    return None

def __grid_widgets(self):
        """Places all the child widgets in the appropriate positions."""
        scrollbar_column = 0 if self.__compound is tk.LEFT else 2
        self._canvas.grid(row=0, column=1, sticky="nswe")
        self._scrollbar.grid(row=0, column=scrollbar_column, sticky="ns")

def _if(ctx, logical_test, value_if_true=0, value_if_false=False):
    """
    Returns one value if the condition evaluates to TRUE, and another value if it evaluates to FALSE
    """
    return value_if_true if conversions.to_boolean(logical_test, ctx) else value_if_false

def multidict_to_dict(d):
    """
    Turns a werkzeug.MultiDict or django.MultiValueDict into a dict with
    list values
    :param d: a MultiDict or MultiValueDict instance
    :return: a dict instance
    """
    return dict((k, v[0] if len(v) == 1 else v) for k, v in iterlists(d))

def form_valid(self, form):
        """Security check complete. Log the user in."""
        auth_login(self.request, form.get_user())
        return HttpResponseRedirect(self.get_success_url())

def normalize_array(lst):
    """Normalizes list

    :param lst: Array of floats
    :return: Normalized (in [0, 1]) input array
    """
    np_arr = np.array(lst)
    x_normalized = np_arr / np_arr.max(axis=0)
    return list(x_normalized)

def is_int_vector(l):
    r"""Checks if l is a numpy array of integers

    """
    if isinstance(l, np.ndarray):
        if l.ndim == 1 and (l.dtype.kind == 'i' or l.dtype.kind == 'u'):
            return True
    return False

def convert_timestamp(timestamp):
    """
    Converts bokehJS timestamp to datetime64.
    """
    datetime = dt.datetime.utcfromtimestamp(timestamp/1000.)
    return np.datetime64(datetime.replace(tzinfo=None))

def region_from_segment(image, segment):
    """given a segment (rectangle) and an image, returns it's corresponding subimage"""
    x, y, w, h = segment
    return image[y:y + h, x:x + w]

def __run(self):
    """Hacked run function, which installs the trace."""
    sys.settrace(self.globaltrace)
    self.__run_backup()
    self.run = self.__run_backup

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def format_exc(*exc_info):
    """Show exception with traceback."""
    typ, exc, tb = exc_info or sys.exc_info()
    error = traceback.format_exception(typ, exc, tb)
    return "".join(error)

def image_set_aspect(aspect=1.0, axes="gca"):
    """
    sets the aspect ratio of the current zoom level of the imshow image
    """
    if axes is "gca": axes = _pylab.gca()

    e = axes.get_images()[0].get_extent()
    axes.set_aspect(abs((e[1]-e[0])/(e[3]-e[2]))/aspect)

def str_traceback(error, tb):
    """Returns a string representation of the traceback.
    """
    if not isinstance(tb, types.TracebackType):
        return tb

    return ''.join(traceback.format_exception(error.__class__, error, tb))

def stepBy(self, steps):
        """steps value up/down by a single step. Single step is defined in singleStep().

        Args:
            steps (int): positiv int steps up, negativ steps down
        """
        self.setValue(self.value() + steps*self.singleStep())

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def _match_space_at_line(line):
    """Return a re.match object if an empty comment was found on line."""
    regex = re.compile(r"^{0}$".format(_MDL_COMMENT))
    return regex.match(line)

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

def handle_whitespace(text):
    r"""Handles whitespace cleanup.

    Tabs are "smartly" retabbed (see sub_retab). Lines that contain
    only whitespace are truncated to a single newline.
    """
    text = re_retab.sub(sub_retab, text)
    text = re_whitespace.sub('', text).strip()
    return text

def simple_generate(cls, create, **kwargs):
        """Generate a new instance.

        The instance will be either 'built' or 'created'.

        Args:
            create (bool): whether to 'build' or 'create' the instance.

        Returns:
            object: the generated instance
        """
        strategy = enums.CREATE_STRATEGY if create else enums.BUILD_STRATEGY
        return cls.generate(strategy, **kwargs)

def s3(ctx, bucket_name, data_file, region):
    """Use the S3 SWAG backend."""
    if not ctx.data_file:
        ctx.data_file = data_file

    if not ctx.bucket_name:
        ctx.bucket_name = bucket_name

    if not ctx.region:
        ctx.region = region

    ctx.type = 's3'

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

def match_paren(self, tokens, item):
        """Matches a paren."""
        match, = tokens
        return self.match(match, item)

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

def fval(self, instance):
        """return the raw value that this property is holding internally for instance"""
        try:
            val = instance.__dict__[self.instance_field_name]
        except KeyError as e:
            #raise AttributeError(str(e))
            val = None

        return val

def retry_on_signal(function):
    """Retries function until it doesn't raise an EINTR error"""
    while True:
        try:
            return function()
        except EnvironmentError, e:
            if e.errno != errno.EINTR:
                raise

def  make_html_code( self, lines ):
        """ convert a code sequence to HTML """
        line = code_header + '\n'
        for l in lines:
            line = line + html_quote( l ) + '\n'

        return line + code_footer

def _tuple_repr(data):
    """Return a repr() for a list/tuple"""
    if len(data) == 1:
        return "(%s,)" % rpr(data[0])
    else:
        return "(%s)" % ", ".join([rpr(x) for x in data])

def _histplot_op(ax, data, **kwargs):
    """Add a histogram for the data to the axes."""
    bins = get_bins(data)
    ax.hist(data, bins=bins, align="left", density=True, **kwargs)
    return ax

def ver_to_tuple(value):
    """
    Convert version like string to a tuple of integers.
    """
    return tuple(int(_f) for _f in re.split(r'\D+', value) if _f)

def is_full_slice(obj, l):
    """
    We have a full length slice.
    """
    return (isinstance(obj, slice) and obj.start == 0 and obj.stop == l and
            obj.step is None)

def invertDictMapping(d):
    """ Invert mapping of dictionary (i.e. map values to list of keys) """
    inv_map = {}
    for k, v in d.items():
        inv_map[v] = inv_map.get(v, [])
        inv_map[v].append(k)
    return inv_map

def one_hot2string(arr, vocab):
    """Convert a one-hot encoded array back to string
    """
    tokens = one_hot2token(arr)
    indexToLetter = _get_index_dict(vocab)

    return [''.join([indexToLetter[x] for x in row]) for row in tokens]

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

def stringify_dict_contents(dct):
    """Turn dict keys and values into native strings."""
    return {
        str_if_nested_or_str(k): str_if_nested_or_str(v)
        for k, v in dct.items()
    }

def _to_java_object_rdd(rdd):
    """ Return an JavaRDD of Object by unpickling

    It will convert each Python object into Java object by Pyrolite, whenever the
    RDD is serialized in batch or not.
    """
    rdd = rdd._reserialize(AutoBatchedSerializer(PickleSerializer()))
    return rdd.ctx._jvm.org.apache.spark.ml.python.MLSerDe.pythonToJava(rdd._jrdd, True)

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

def as_list(self):
        """Return all child objects in nested lists of strings."""
        return [self.name, self.value, [x.as_list for x in self.children]]

def fixpath(path):
    """Uniformly format a path."""
    return os.path.normpath(os.path.realpath(os.path.expanduser(path)))

def _npiter(arr):
    """Wrapper for iterating numpy array"""
    for a in np.nditer(arr, flags=["refs_ok"]):
        c = a.item()
        if c is not None:
            yield c

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

def is_type(value):
        """Determine if value is an instance or subclass of the class Type."""
        if isinstance(value, type):
            return issubclass(value, Type)
        return isinstance(value, Type)

def query_sum(queryset, field):
    """
    Let the DBMS perform a sum on a queryset
    """
    return queryset.aggregate(s=models.functions.Coalesce(models.Sum(field), 0))['s']

def to_pascal_case(s):
    """Transform underscore separated string to pascal case

    """
    return re.sub(r'(?!^)_([a-zA-Z])', lambda m: m.group(1).upper(), s.capitalize())

def unique_items(seq):
    """Return the unique items from iterable *seq* (in order)."""
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]

def wireshark(pktlist, *args):
    """Run wireshark on a list of packets"""
    fname = get_temp_file()
    wrpcap(fname, pktlist)
    subprocess.Popen([conf.prog.wireshark, "-r", fname] + list(args))

def step_impl06(context):
    """Prepare test for singleton property.

    :param context: test context.
    """
    store = context.SingleStore
    context.st_1 = store()
    context.st_2 = store()
    context.st_3 = store()

def jac(x,a):
    """ Jacobian matrix given Christophe's suggestion of f """
    return (x-a) / np.sqrt(((x-a)**2).sum(1))[:,np.newaxis]

def autobuild_python_test(path):
    """Add pytest unit tests to be built as part of build/test/output."""

    env = Environment(tools=[])
    target = env.Command(['build/test/output/pytest.log'], [path],
                         action=env.Action(run_pytest, "Running python unit tests"))
    env.AlwaysBuild(target)

def test_python_java_rt():
    """ Run Python test cases against Java runtime classes. """
    sub_env = {'PYTHONPATH': _build_dir()}

    log.info('Executing Python unit tests (against Java runtime classes)...')
    return jpyutil._execute_python_scripts(python_java_rt_tests,
                                           env=sub_env)

def raises_regex(self, expected_exception, expected_regexp):
        """
        Ensures preceding predicates (specifically, :meth:`called_with()`) result in *expected_exception* being raised,
        and the string representation of *expected_exception* must match regular expression *expected_regexp*.
        """
        return unittest_case.assertRaisesRegexp(expected_exception, expected_regexp, self._orig_subject,
                                                *self._args, **self._kwargs)

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

def test_python_java_rt():
    """ Run Python test cases against Java runtime classes. """
    sub_env = {'PYTHONPATH': _build_dir()}

    log.info('Executing Python unit tests (against Java runtime classes)...')
    return jpyutil._execute_python_scripts(python_java_rt_tests,
                                           env=sub_env)

def test():
    """Run the unit tests."""
    import unittest
    tests = unittest.TestLoader().discover('tests')
    unittest.TextTestRunner(verbosity=2).run(tests)

def import_js(path, lib_name, globals):
    """Imports from javascript source file.
      globals is your globals()"""
    with codecs.open(path_as_local(path), "r", "utf-8") as f:
        js = f.read()
    e = EvalJs()
    e.execute(js)
    var = e.context['var']
    globals[lib_name] = var.to_python()

def import_path(self):
    """The full remote import path as used in import statements in `.go` source files."""
    return os.path.join(self.remote_root, self.pkg) if self.pkg else self.remote_root

def sbatch_template(self):
        """:return Jinja sbatch template for the current tag"""
        template = self.sbatch_template_str
        if template.startswith('#!'):
            # script is embedded in YAML
            return jinja_environment.from_string(template)
        return jinja_environment.get_template(template)

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

def update(self, other_dict):
        """update() extends rather than replaces existing key lists."""
        for key, value in iter_multi_items(other_dict):
            MultiDict.add(self, key, value)

def vectorize(values):
    """
    Takes a value or list of values and returns a single result, joined by ","
    if necessary.
    """
    if isinstance(values, list):
        return ','.join(str(v) for v in values)
    return values

def update(self, params):
        """Update the dev_info data from a dictionary.

        Only updates if it already exists in the device.
        """
        dev_info = self.json_state.get('deviceInfo')
        dev_info.update({k: params[k] for k in params if dev_info.get(k)})

def index():
    """ Display productpage with normal user and test user buttons"""
    global productpage

    table = json2html.convert(json = json.dumps(productpage),
                              table_attributes="class=\"table table-condensed table-bordered table-hover\"")

    return render_template('index.html', serviceTable=table)

def set_property(self, key, value):
        """
        Update only one property in the dict
        """
        self.properties[key] = value
        self.sync_properties()

def _spawn_kafka_consumer_thread(self):
        """Spawns a kafka continuous consumer thread"""
        self.logger.debug("Spawn kafka consumer thread""")
        self._consumer_thread = Thread(target=self._consumer_loop)
        self._consumer_thread.setDaemon(True)
        self._consumer_thread.start()

def url(self):
        """ The url of this window """
        with switch_window(self._browser, self.name):
            return self._browser.url

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

def server(self):
        """Returns the size of remote files
        """
        try:
            tar = urllib2.urlopen(self.registry)
            meta = tar.info()
            return int(meta.getheaders("Content-Length")[0])
        except (urllib2.URLError, IndexError):
            return " "

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

def escape(s):
    """Escape a URL including any /."""
    if not isinstance(s, bytes):
        s = s.encode('utf-8')
    return quote(s, safe='~')

def cleanup_storage(*args):
    """Clean up processes after SIGTERM or SIGINT is received."""
    ShardedClusters().cleanup()
    ReplicaSets().cleanup()
    Servers().cleanup()
    sys.exit(0)

def get_url_args(url):
    """ Returns a dictionary from a URL params """
    url_data = urllib.parse.urlparse(url)
    arg_dict = urllib.parse.parse_qs(url_data.query)
    return arg_dict

def kill_all(self, kill_signal, kill_shell=False):
        """Kill all running processes."""
        for key in self.processes.keys():
            self.kill_process(key, kill_signal, kill_shell)

def updateFromKwargs(self, properties, kwargs, collector, **unused):
        """Primary entry point to turn 'kwargs' into 'properties'"""
        properties[self.name] = self.getFromKwargs(kwargs)

def dumps(obj):
    """Outputs json with formatting edits + object handling."""
    return json.dumps(obj, indent=4, sort_keys=True, cls=CustomEncoder)

def notin(arg, values):
    """
    Like isin, but checks whether this expression's value(s) are not
    contained in the passed values. See isin docs for full usage.
    """
    op = ops.NotContains(arg, values)
    return op.to_expr()

def remove_legend(ax=None):
    """Remove legend for axes or gca.

    See http://osdir.com/ml/python.matplotlib.general/2005-07/msg00285.html
    """
    from pylab import gca, draw
    if ax is None:
        ax = gca()
    ax.legend_ = None
    draw()

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

def decode_unicode_string(string):
    """
    Decode string encoded by `unicode_string`
    """
    if string.startswith('[BASE64-DATA]') and string.endswith('[/BASE64-DATA]'):
        return base64.b64decode(string[len('[BASE64-DATA]'):-len('[/BASE64-DATA]')])
    return string

def set_ylimits(self, row, column, min=None, max=None):
        """Set y-axis limits of a subplot.

        :param row,column: specify the subplot.
        :param min: minimal axis value
        :param max: maximum axis value

        """
        subplot = self.get_subplot_at(row, column)
        subplot.set_ylimits(min, max)

def generate_uuid():
    """Generate a UUID."""
    r_uuid = base64.urlsafe_b64encode(uuid.uuid4().bytes)
    return r_uuid.decode().replace('=', '')

def _match_space_at_line(line):
    """Return a re.match object if an empty comment was found on line."""
    regex = re.compile(r"^{0}$".format(_MDL_COMMENT))
    return regex.match(line)

def email_type(arg):
	"""An argparse type representing an email address."""
	if not is_valid_email_address(arg):
		raise argparse.ArgumentTypeError("{0} is not a valid email address".format(repr(arg)))
	return arg

def get_ip_address(ifname):
    """ Hack to get IP address from the interface """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    return socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,  # SIOCGIFADDR
        struct.pack('256s', ifname[:15])
    )[20:24])

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

def _check_list_len(row, length):
        """
        Sanity check for csv parser
        :param row
        :param length
        :return:None
        """
        if len(row) != length:
            raise Exception(
                "row length does not match expected length of " +
                str(length) + "\nrow: " + str(row))

def get_python_dict(scala_map):
    """Return a dict from entries in a scala.collection.immutable.Map"""
    python_dict = {}
    keys = get_python_list(scala_map.keys().toList())
    for key in keys:
        python_dict[key] = scala_map.apply(key)
    return python_dict

def getprop(self, prop_name):
        """Get a property of the device.

        This is a convenience wrapper for "adb shell getprop xxx".

        Args:
            prop_name: A string that is the name of the property to get.

        Returns:
            A string that is the value of the property, or None if the property
            doesn't exist.
        """
        return self.shell(
            ['getprop', prop_name],
            timeout=DEFAULT_GETPROP_TIMEOUT_SEC).decode('utf-8').strip()

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

def add_input_variable(self, var):
        """Adds the argument variable as one of the input variable"""
        assert(isinstance(var, Variable))
        self.input_variable_list.append(var)

def list2dict(lst):
    """Takes a list of (key,value) pairs and turns it into a dict."""

    dic = {}
    for k,v in lst: dic[k] = v
    return dic

def dot_v2(vec1, vec2):
    """Return the dot product of two vectors"""

    return vec1.x * vec2.x + vec1.y * vec2.y

def open_json(file_name):
    """
    returns json contents as string
    """
    with open(file_name, "r") as json_data:
        data = json.load(json_data)
        return data

def dot_v2(vec1, vec2):
    """Return the dot product of two vectors"""

    return vec1.x * vec2.x + vec1.y * vec2.y

def load_jsonf(fpath, encoding):
    """
    :param unicode fpath:
    :param unicode encoding:
    :rtype: dict | list
    """
    with codecs.open(fpath, encoding=encoding) as f:
        return json.load(f)

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

def glog(x,l = 2):
    """
    Generalised logarithm

    :param x: number
    :param p: number added befor logarithm 

    """
    return np.log((x+np.sqrt(x**2+l**2))/2)/np.log(l)

def size(self):
        """Return the viewable size of the Table as @tuple (x,y)"""
        width = max(
            map(lambda x: x.size()[0], self.sections.itervalues()))

        height = sum(
            map(lambda x: x.size()[1], self.sections.itervalues()))

        return width, height

def should_skip_logging(func):
    """
    Should we skip logging for this handler?

    """
    disabled = strtobool(request.headers.get("x-request-nolog", "false"))
    return disabled or getattr(func, SKIP_LOGGING, False)

def world_to_view(v):
    """world coords to view coords; v an eu.Vector2, returns (float, float)"""
    return v.x * config.scale_x, v.y * config.scale_y

def lognorm(x, mu, sigma=1.0):
    """ Log-normal function from scipy """
    return stats.lognorm(sigma, scale=mu).pdf(x)

def rm(venv_name):
    """ Removes the venv by name """
    inenv = InenvManager()
    venv = inenv.get_venv(venv_name)
    click.confirm("Delete dir {}".format(venv.path))
    shutil.rmtree(venv.path)

def setAsApplication(myappid):
    """
    Tells Windows this is an independent application with an unique icon on task bar.

    id is an unique string to identify this application, like: 'mycompany.myproduct.subproduct.version'
    """

    if os.name == 'nt':
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

def _disable_venv(self, env):
        """
        Disable virtualenv and venv in the environment.
        """
        venv = env.pop('VIRTUAL_ENV', None)
        if venv:
            venv_path, sep, env['PATH'] = env['PATH'].partition(os.pathsep)

def _convert_to_array(array_like, dtype):
        """
        Convert Matrix attributes which are array-like or buffer to array.
        """
        if isinstance(array_like, bytes):
            return np.frombuffer(array_like, dtype=dtype)
        return np.asarray(array_like, dtype=dtype)

def unique_everseen(seq):
    """Solution found here : http://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-python-whilst-preserving-order"""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def submit_by_selector(self, selector):
    """Submit the form matching the CSS selector."""
    elem = find_element_by_jquery(world.browser, selector)
    elem.submit()

def list2dict(lst):
    """Takes a list of (key,value) pairs and turns it into a dict."""

    dic = {}
    for k,v in lst: dic[k] = v
    return dic

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

def make_unique_ngrams(s, n):
    """Make a set of unique n-grams from a string."""
    return set(s[i:i + n] for i in range(len(s) - n + 1))

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

def to_dataframe(products):
        """Return the products from a query response as a Pandas DataFrame
        with the values in their appropriate Python types.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("to_dataframe requires the optional dependency Pandas.")

        return pd.DataFrame.from_dict(products, orient='index')

def start(self):
        """Create a background thread for httpd and serve 'forever'"""
        self._process = threading.Thread(target=self._background_runner)
        self._process.start()

def commajoin_as_strings(iterable):
    """ Join the given iterable with ',' """
    return _(u',').join((six.text_type(i) for i in iterable))

def disconnect(self):
        """
        Closes the connection.
        """
        self.logger.debug('Close connection...')

        self.auto_reconnect = False

        if self.websocket is not None:
            self.websocket.close()

def pack_triples_numpy(triples):
    """Packs a list of triple indexes into a 2D numpy array."""
    if len(triples) == 0:
        return np.array([], dtype=np.int64)
    return np.stack(list(map(_transform_triple_numpy, triples)), axis=0)

def settimeout(self, timeout):
        """
        Set the timeout to the websocket.

        timeout: timeout time(second).
        """
        self.sock_opt.timeout = timeout
        if self.sock:
            self.sock.settimeout(timeout)

def select_random(engine, table_or_columns, limit=5):
    """
    Randomly select some rows from table.
    """
    s = select(table_or_columns).order_by(func.random()).limit(limit)
    return engine.execute(s).fetchall()

def _ws_on_close(self, ws: websocket.WebSocketApp):
        """Callback for closing the websocket connection

        Args:
            ws: websocket connection (now closed)
        """
        self.connected = False
        self.logger.error('Websocket closed')
        self._reconnect_websocket()

def classnameify(s):
  """
  Makes a classname
  """
  return ''.join(w if w in ACRONYMS else w.title() for w in s.split('_'))

def _file_exists(path, filename):
  """Checks if the filename exists under the path."""
  return os.path.isfile(os.path.join(path, filename))

def top(n, width=WIDTH, style=STYLE):
    """Prints the top row of a table"""
    return hrule(n, width, linestyle=STYLES[style].top)

def join_field(path):
    """
    RETURN field SEQUENCE AS STRING
    """
    output = ".".join([f.replace(".", "\\.") for f in path if f != None])
    return output if output else "."

def create_widget(self):
        """ Create the toolkit widget for the proxy object.
        """
        d = self.declaration
        button_type = UIButton.UIButtonTypeSystem if d.flat else UIButton.UIButtonTypeRoundedRect
        self.widget = UIButton(buttonWithType=button_type)

def closing_plugin(self, cancelable=False):
        """Perform actions before parent main window is closed"""
        self.dialog_manager.close_all()
        self.shell.exit_interpreter()
        return True

def atom_criteria(*params):
    """An auxiliary function to construct a dictionary of Criteria"""
    result = {}
    for index, param in enumerate(params):
        if param is None:
            continue
        elif isinstance(param, int):
            result[index] = HasAtomNumber(param)
        else:
            result[index] = param
    return result

def normalize_path(filename):
    """Normalize a file/dir name for comparison purposes"""
    return os.path.normcase(os.path.realpath(os.path.normpath(_cygwin_patch(filename))))

def show(config):
    """Show revision list"""
    with open(config, 'r'):
        main.show(yaml.load(open(config)))

def _manhattan_distance(vec_a, vec_b):
    """Return manhattan distance between two lists of numbers."""
    if len(vec_a) != len(vec_b):
        raise ValueError('len(vec_a) must equal len(vec_b)')
    return sum(map(lambda a, b: abs(a - b), vec_a, vec_b))

def x_values_ref(self, series):
        """
        The Excel worksheet reference to the X values for this chart (not
        including the column label).
        """
        top_row = self.series_table_row_offset(series) + 2
        bottom_row = top_row + len(series) - 1
        return "Sheet1!$A$%d:$A$%d" % (top_row, bottom_row)

def asyncStarCmap(asyncCallable, iterable):
    """itertools.starmap for deferred callables using cooperative multitasking
    """
    results = []
    yield coopStar(asyncCallable, results.append, iterable)
    returnValue(results)

def sample_colormap(cmap_name, n_samples):
    """
    Sample a colormap from matplotlib
    """
    colors = []
    colormap = cm.cmap_d[cmap_name]
    for i in np.linspace(0, 1, n_samples):
        colors.append(colormap(i))

    return colors

def write_dict_to_yaml(dictionary, path, **kwargs):
    """
    Writes a dictionary to a yaml file
    :param dictionary:  the dictionary to be written
    :param path: the absolute path of the target yaml file
    :param kwargs: optional additional parameters for dumper
    """
    with open(path, 'w') as f:
        yaml.dump(dictionary, f, indent=4, **kwargs)

def page_guiref(arg_s=None):
    """Show a basic reference about the GUI Console."""
    from IPython.core import page
    page.page(gui_reference, auto_html=True)

def write_fits(self, fitsfile):
        """Write the ROI model to a FITS file."""

        tab = self.create_table()
        hdu_data = fits.table_to_hdu(tab)
        hdus = [fits.PrimaryHDU(), hdu_data]
        fits_utils.write_hdus(hdus, fitsfile)

def md_to_text(content):
    """ Converts markdown content to text """
    text = None
    html = markdown.markdown(content)
    if html:
        text = html_to_text(content)
    return text

def imp_print(self, text, end):
		"""Directly send utf8 bytes to stdout"""
		sys.stdout.write((text + end).encode("utf-8"))

def intersect(d1, d2):
    """Intersect dictionaries d1 and d2 by key *and* value."""
    return dict((k, d1[k]) for k in d1 if k in d2 and d1[k] == d2[k])

def cleanup_nodes(doc):
    """
    Remove text nodes containing only whitespace
    """
    for node in doc.documentElement.childNodes:
        if node.nodeType == Node.TEXT_NODE and node.nodeValue.isspace():
            doc.documentElement.removeChild(node)
    return doc

def intersect(d1, d2):
    """Intersect dictionaries d1 and d2 by key *and* value."""
    return dict((k, d1[k]) for k in d1 if k in d2 and d1[k] == d2[k])

def validate(self, xml_input):
        """
        This method validate the parsing and schema, return a boolean
        """
        parsed_xml = etree.parse(self._handle_xml(xml_input))
        try:
            return self.xmlschema.validate(parsed_xml)
        except AttributeError:
            raise CannotValidate('Set XSD to validate the XML')

def load_data(filename):
    """
    :rtype : numpy matrix
    """
    data = pandas.read_csv(filename, header=None, delimiter='\t', skiprows=9)
    return data.as_matrix()

def print_yaml(o):
    """Pretty print an object as YAML."""
    print(yaml.dump(o, default_flow_style=False, indent=4, encoding='utf-8'))

def yaml_to_param(obj, name):
	"""
	Return the top-level element of a document sub-tree containing the
	YAML serialization of a Python object.
	"""
	return from_pyvalue(u"yaml:%s" % name, unicode(yaml.dump(obj)))

def _digits(minval, maxval):
    """Digits needed to comforatbly display values in [minval, maxval]"""
    if minval == maxval:
        return 3
    else:
        return min(10, max(2, int(1 + abs(np.log10(maxval - minval)))))

def print_yaml(o):
    """Pretty print an object as YAML."""
    print(yaml.dump(o, default_flow_style=False, indent=4, encoding='utf-8'))

def heappop_max(heap):
    """Maxheap version of a heappop."""
    lastelt = heap.pop()    # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        _siftup_max(heap, 0)
        return returnitem
    return lastelt

def softmax(xs):
    """Stable implementation of the softmax function."""
    ys = xs - np.max(xs)
    exps = np.exp(ys)
    return exps / exps.sum(axis=0)

def get_indentation(line):
    """Return leading whitespace."""
    if line.strip():
        non_whitespace_index = len(line) - len(line.lstrip())
        return line[:non_whitespace_index]
    else:
        return ''

def read_bytes(fo, writer_schema=None, reader_schema=None):
    """Bytes are encoded as a long followed by that many bytes of data."""
    size = read_long(fo)
    return fo.read(size)

def extract_zip(zip_path, target_folder):
    """
    Extract the content of the zip-file at `zip_path` into `target_folder`.
    """
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target_folder)

def included_length(self):
        """Surveyed length, not including "excluded" shots"""
        return sum([shot.length for shot in self.shots if shot.is_included])

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

def timeit(output):
    """
    If output is string, then print the string and also time used
    """
    b = time.time()
    yield
    print output, 'time used: %.3fs' % (time.time()-b)

def compress(data, **kwargs):
    """zlib.compress(data, **kwargs)
    
    """ + zopfli.__COMPRESSOR_DOCSTRING__  + """
    Returns:
      String containing a zlib container
    """
    kwargs['gzip_mode'] = 0
    return zopfli.zopfli.compress(data, **kwargs)

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

def input_int_default(question="", default=0):
    """A function that works for both, Python 2.x and Python 3.x.
       It asks the user for input and returns it as a string.
    """
    answer = input_string(question)
    if answer == "" or answer == "yes":
        return default
    else:
        return int(answer)

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

def force_iterable(f):
    """Will make any functions return an iterable objects by wrapping its result in a list."""
    def wrapper(*args, **kwargs):
        r = f(*args, **kwargs)
        if hasattr(r, '__iter__'):
            return r
        else:
            return [r]
    return wrapper

def dict_merge(set1, set2):
    """Joins two dictionaries."""
    return dict(list(set1.items()) + list(set2.items()))

def asynchronous(function, event):
    """
    Runs the function asynchronously taking care of exceptions.
    """
    thread = Thread(target=synchronous, args=(function, event))
    thread.daemon = True
    thread.start()

def dictmerge(x, y):
    """
    merge two dictionaries
    """
    z = x.copy()
    z.update(y)
    return z

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

def bin_to_int(string):
    """Convert a one element byte string to signed int for python 2 support."""
    if isinstance(string, str):
        return struct.unpack("b", string)[0]
    else:
        return struct.unpack("b", bytes([string]))[0]

def newest_file(file_iterable):
  """
  Returns the name of the newest file given an iterable of file names.

  """
  return max(file_iterable, key=lambda fname: os.path.getmtime(fname))

def is_valid_folder(parser, arg):
    """Check if arg is a valid file that already exists on the file system."""
    arg = os.path.abspath(arg)
    if not os.path.isdir(arg):
        parser.error("The folder %s does not exist!" % arg)
    else:
        return arg

def GaussianBlur(X, ksize_width, ksize_height, sigma_x, sigma_y):
    """Apply Gaussian blur to the given data.

    Args:
        X: data to blur
        kernel_size: Gaussian kernel size
        stddev: Gaussian kernel standard deviation (in both X and Y directions)
    """
    return image_transform(
        X,
        cv2.GaussianBlur,
        ksize=(ksize_width, ksize_height),
        sigmaX=sigma_x,
        sigmaY=sigma_y
    )

def md5_string(s):
    """
    Shortcut to create md5 hash
    :param s:
    :return:
    """
    m = hashlib.md5()
    m.update(s)
    return str(m.hexdigest())

def set_slug(apps, schema_editor, class_name):
    """
    Create a slug for each Work already in the DB.
    """
    Cls = apps.get_model('spectator_events', class_name)

    for obj in Cls.objects.all():
        obj.slug = generate_slug(obj.pk)
        obj.save(update_fields=['slug'])

def moving_average(a, n):
    """Moving average over one-dimensional array.

    Parameters
    ----------
    a : np.ndarray
        One-dimensional array.
    n : int
        Number of entries to average over. n=2 means averaging over the currrent
        the previous entry.

    Returns
    -------
    An array view storing the moving average.
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def sigterm(self, signum, frame):
        """
        These actions will be done after SIGTERM.
        """
        self.logger.warning("Caught signal %s. Stopping daemon." % signum)
        sys.exit(0)

def go_to_parent_directory(self):
        """Go to parent directory"""
        self.chdir(osp.abspath(osp.join(getcwd_or_home(), os.pardir)))

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

def many_until1(these, term):
    """Like many_until but must consume at least one of these.
    """
    first = [these()]
    these_results, term_result = many_until(these, term)
    return (first + these_results, term_result)

def is_enum_type(type_):
    """ Checks if the given type is an enum type.

    :param type_: The type to check
    :return: True if the type is a enum type, otherwise False
    :rtype: bool
    """

    return isinstance(type_, type) and issubclass(type_, tuple(_get_types(Types.ENUM)))

def many_until1(these, term):
    """Like many_until but must consume at least one of these.
    """
    first = [these()]
    these_results, term_result = many_until(these, term)
    return (first + these_results, term_result)

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

def setdefault(obj, field, default):
    """Set an object's field to default if it doesn't have a value"""
    setattr(obj, field, getattr(obj, field, default))

def match_paren(self, tokens, item):
        """Matches a paren."""
        match, = tokens
        return self.match(match, item)

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

def word_matches(s1, s2, n=3):
    """
        Word-level n-grams that match between two strings

        Args:
            s1: a string
            s2: another string
            n: an int for the n in n-gram

        Returns:
            set: the n-grams found in both strings
    """
    return __matches(s1, s2, word_ngrams, n=n)

def write(self, text):
        """Write text. An additional attribute terminator with a value of
           None is added to the logging record to indicate that StreamHandler
           should not add a newline."""
        self.logger.log(self.loglevel, text, extra={'terminator': None})

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

def _bytes_to_json(value):
    """Coerce 'value' to an JSON-compatible representation."""
    if isinstance(value, bytes):
        value = base64.standard_b64encode(value).decode("ascii")
    return value

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

def parse_date(s):
    """Fast %Y-%m-%d parsing."""
    try:
        return datetime.date(int(s[:4]), int(s[5:7]), int(s[8:10]))
    except ValueError:  # other accepted format used in one-day data set
        return datetime.datetime.strptime(s, '%d %B %Y').date()

def EvalGaussianPdf(x, mu, sigma):
    """Computes the unnormalized PDF of the normal distribution.

    x: value
    mu: mean
    sigma: standard deviation
    
    returns: float probability density
    """
    return scipy.stats.norm.pdf(x, mu, sigma)

def read_string(buff, byteorder='big'):
    """Read a string from a file-like object."""
    length = read_numeric(USHORT, buff, byteorder)
    return buff.read(length).decode('utf-8')

def _normalize_abmn(abmn):
    """return a normalized version of abmn
    """
    abmn_2d = np.atleast_2d(abmn)
    abmn_normalized = np.hstack((
        np.sort(abmn_2d[:, 0:2], axis=1),
        np.sort(abmn_2d[:, 2:4], axis=1),
    ))
    return abmn_normalized

def disable_stdout_buffering():
    """This turns off stdout buffering so that outputs are immediately
    materialized and log messages show up before the program exits"""
    stdout_orig = sys.stdout
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
    # NOTE(brandyn): This removes the original stdout
    return stdout_orig

def denorm(self,arr):
        """Reverse the normalization done to a batch of images.

        Arguments:
            arr: of shape/size (N,3,sz,sz)
        """
        if type(arr) is not np.ndarray: arr = to_np(arr)
        if len(arr.shape)==3: arr = arr[None]
        return self.transform.denorm(np.rollaxis(arr,1,4))

def pick_unused_port(self):
    """ Pick an unused port. There is a slight chance that this wont work. """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', 0))
    _, port = s.getsockname()
    s.close()
    return port

def is_gzipped_fastq(file_name):
    """
    Determine whether indicated file appears to be a gzipped FASTQ.

    :param str file_name: Name/path of file to check as gzipped FASTQ.
    :return bool: Whether indicated file appears to be in gzipped FASTQ format.
    """
    _, ext = os.path.splitext(file_name)
    return file_name.endswith(".fastq.gz") or file_name.endswith(".fq.gz")

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

def dedup_list(l):
    """Given a list (l) will removing duplicates from the list,
       preserving the original order of the list. Assumes that
       the list entrie are hashable."""
    dedup = set()
    return [ x for x in l if not (x in dedup or dedup.add(x))]

def safe_unicode(string):
    """If Python 2, replace non-ascii characters and return encoded string."""
    if not PY3:
        uni = string.replace(u'\u2019', "'")
        return uni.encode('utf-8')
        
    return string

def is_equal_strings_ignore_case(first, second):
    """The function compares strings ignoring case"""
    if first and second:
        return first.upper() == second.upper()
    else:
        return not (first or second)

def check_str(obj):
        """ Returns a string for various input types """
        if isinstance(obj, str):
            return obj
        if isinstance(obj, float):
            return str(int(obj))
        else:
            return str(obj)

def get_incomplete_path(filename):
  """Returns a temporary filename based on filename."""
  random_suffix = "".join(
      random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
  return filename + ".incomplete" + random_suffix

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

def _to_array(value):
    """As a convenience, turn Python lists and tuples into NumPy arrays."""
    if isinstance(value, (tuple, list)):
        return array(value)
    elif isinstance(value, (float, int)):
        return np.float64(value)
    else:
        return value

def unique_deps(deps):
    """Remove duplicities from deps list of the lists"""
    deps.sort()
    return list(k for k, _ in itertools.groupby(deps))

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

def Max(a, axis, keep_dims):
    """
    Max reduction op.
    """
    return np.amax(a, axis=axis if not isinstance(axis, np.ndarray) else tuple(axis),
                   keepdims=keep_dims),

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

def _config_parse(self):
        """Replacer oslo_config.cfg.ConfigParser.parse for in-memory cfg."""
        res = super(cfg.ConfigParser, self).parse(Backend._config_string_io)
        return res

def normal_noise(points):
    """Init a noise variable."""
    return np.random.rand(1) * np.random.randn(points, 1) \
        + random.sample([2, -2], 1)

def gen_random_string(str_len):
    """ generate random string with specified length
    """
    return ''.join(
        random.choice(string.ascii_letters + string.digits) for _ in range(str_len))

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def gen_random_string(str_len):
    """ generate random string with specified length
    """
    return ''.join(
        random.choice(string.ascii_letters + string.digits) for _ in range(str_len))

def one_hot_encoding(input_tensor, num_labels):
    """ One-hot encode labels from input """
    xview = input_tensor.view(-1, 1).to(torch.long)

    onehot = torch.zeros(xview.size(0), num_labels, device=input_tensor.device, dtype=torch.float)
    onehot.scatter_(1, xview, 1)
    return onehot.view(list(input_tensor.shape) + [-1])

def runiform(lower, upper, size=None):
    """
    Random uniform variates.
    """
    return np.random.uniform(lower, upper, size)

def rpc_fix_code(self, source, directory):
        """Formats Python code to conform to the PEP 8 style guide.

        """
        source = get_source(source)
        return fix_code(source, directory)

def overlap(intv1, intv2):
    """Overlaping of two intervals"""
    return max(0, min(intv1[1], intv2[1]) - max(intv1[0], intv2[0]))

def created_today(self):
        """Return True if created today."""
        if self.datetime.date() == datetime.today().date():
            return True
        return False

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

def log(logger, level, message):
    """Logs message to stderr if logging isn't initialized."""

    if logger.parent.name != 'root':
        logger.log(level, message)
    else:
        print(message, file=sys.stderr)

def readTuple(self, line, n=3):
        """ Reads a tuple of numbers. e.g. vertices, normals or teture coords.
        """
        numbers = [num for num in line.split(' ') if num]
        return [float(num) for num in numbers[1:n + 1]]

def read_string_from_file(path, encoding="utf8"):
  """
  Read entire contents of file into a string.
  """
  with codecs.open(path, "rb", encoding=encoding) as f:
    value = f.read()
  return value

def filechunk(f, chunksize):
    """Iterator that allow for piecemeal processing of a file."""
    while True:
        chunk = tuple(itertools.islice(f, chunksize))
        if not chunk:
            return
        yield np.loadtxt(iter(chunk), dtype=np.float64)

def read_string_from_file(path, encoding="utf8"):
  """
  Read entire contents of file into a string.
  """
  with codecs.open(path, "rb", encoding=encoding) as f:
    value = f.read()
  return value

def load_jsonf(fpath, encoding):
    """
    :param unicode fpath:
    :param unicode encoding:
    :rtype: dict | list
    """
    with codecs.open(fpath, encoding=encoding) as f:
        return json.load(f)

def read_string_from_file(path, encoding="utf8"):
  """
  Read entire contents of file into a string.
  """
  with codecs.open(path, "rb", encoding=encoding) as f:
    value = f.read()
  return value

def get_jsonparsed_data(url):
    """Receive the content of ``url``, parse it as JSON and return the
       object.
    """
    response = urlopen(url)
    data = response.read().decode('utf-8')
    return json.loads(data)

def open_with_encoding(filename, encoding, mode='r'):
    """Return opened file with a specific encoding."""
    return io.open(filename, mode=mode, encoding=encoding,
                   newline='')

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

def Output(self):
    """Output all sections of the page."""
    self.Open()
    self.Header()
    self.Body()
    self.Footer()

def read_utf8(fh, byteorder, dtype, count, offsetsize):
    """Read tag data from file and return as unicode string."""
    return fh.read(count).decode('utf-8')

def url(self):
        """ The url of this window """
        with switch_window(self._browser, self.name):
            return self._browser.url

def read(fname):
    """Quick way to read a file content."""
    content = None
    with open(os.path.join(here, fname)) as f:
        content = f.read()
    return content

def load_jsonf(fpath, encoding):
    """
    :param unicode fpath:
    :param unicode encoding:
    :rtype: dict | list
    """
    with codecs.open(fpath, encoding=encoding) as f:
        return json.load(f)

def h5ToDict(h5, readH5pyDataset=True):
    """ Read a hdf5 file into a dictionary """
    h = h5py.File(h5, "r")
    ret = unwrapArray(h, recursive=True, readH5pyDataset=readH5pyDataset)
    if readH5pyDataset: h.close()
    return ret

def min_max_normalize(img):
    """Centre and normalize a given array.

    Parameters:
    ----------
    img: np.ndarray

    """

    min_img = img.min()
    max_img = img.max()

    return (img - min_img) / (max_img - min_img)

def _load_data(filepath):
  """Loads the images and latent values into Numpy arrays."""
  with h5py.File(filepath, "r") as h5dataset:
    image_array = np.array(h5dataset["images"])
    # The 'label' data set in the hdf5 file actually contains the float values
    # and not the class labels.
    values_array = np.array(h5dataset["labels"])
  return image_array, values_array

def rotate_img(im, deg, mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_AREA):
    """ Rotates an image by deg degrees

    Arguments:
        deg (float): degree to rotate.
    """
    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c//2,r//2),deg,1)
    return cv2.warpAffine(im,M,(c,r), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS+interpolation)

def image_set_aspect(aspect=1.0, axes="gca"):
    """
    sets the aspect ratio of the current zoom level of the imshow image
    """
    if axes is "gca": axes = _pylab.gca()

    e = axes.get_images()[0].get_extent()
    axes.set_aspect(abs((e[1]-e[0])/(e[3]-e[2]))/aspect)

def _loadfilepath(self, filepath, **kwargs):
        """This loads a geojson file into a geojson python
        dictionary using the json module.
        
        Note: to load with a different text encoding use the encoding argument.
        """
        with open(filepath, "r") as f:
            data = json.load(f, **kwargs)
        return data

def parse_querystring(self, req, name, field):
        """Pull a querystring value from the request."""
        return core.get_value(req.args, name, field)

def _or(ctx, *logical):
    """
    Returns TRUE if any argument is TRUE
    """
    for arg in logical:
        if conversions.to_boolean(arg, ctx):
            return True
    return False

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

def extract_all(zipfile, dest_folder):
    """
    reads the zip file, determines compression
    and unzips recursively until source files 
    are extracted 
    """
    z = ZipFile(zipfile)
    print(z)
    z.extract(dest_folder)

def imp_print(self, text, end):
		"""Directly send utf8 bytes to stdout"""
		sys.stdout.write((text + end).encode("utf-8"))

def trigger(self, target: str, trigger: str, parameters: Dict[str, Any]={}):
		"""Calls the specified Trigger of another Area with the optionally given parameters.

		Args:
			target: The name of the target Area.
			trigger: The name of the Trigger.
			parameters: The parameters of the function call.
		"""
		pass

def create_env(env_file):
    """Create environ dictionary from current os.environ and
    variables got from given `env_file`"""

    environ = {}
    with open(env_file, 'r') as f:
        for line in f.readlines():
            line = line.rstrip(os.linesep)
            if '=' not in line:
                continue
            if line.startswith('#'):
                continue
            key, value = line.split('=', 1)
            environ[key] = parse_value(value)
    return environ

def disable_stdout_buffering():
    """This turns off stdout buffering so that outputs are immediately
    materialized and log messages show up before the program exits"""
    stdout_orig = sys.stdout
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
    # NOTE(brandyn): This removes the original stdout
    return stdout_orig

def make_lambda(call):
    """Wrap an AST Call node to lambda expression node.
    call: ast.Call node
    """
    empty_args = ast.arguments(args=[], vararg=None, kwarg=None, defaults=[])
    return ast.Lambda(args=empty_args, body=call)

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

def parse(self, s):
        """
        Parses a date string formatted like ``YYYY-MM-DD``.
        """
        return datetime.datetime.strptime(s, self.date_format).date()

def delete_entry(self, key):
        """Delete an object from the redis table"""
        pipe = self.client.pipeline()
        pipe.srem(self.keys_container, key)
        pipe.delete(key)
        pipe.execute()

def urlencoded(body, charset='ascii', **kwargs):
    """Converts query strings into native Python objects"""
    return parse_query_string(text(body, charset=charset), False)
