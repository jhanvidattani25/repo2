def sem(inlist):
    """
Returns the estimated standard error of the mean (sx-bar) of the
values in the passed list.  sem = stdev / sqrt(n)

Usage:   lsem(inlist)
"""
    sd = stdev(inlist)
    n = len(inlist)
    return sd / math.sqrt(n)

def unwind(self):
        """ Get a list of all ancestors in descending order of level, including a new instance  of self
        """
        return [ QuadKey(self.key[:l+1]) for l in reversed(range(len(self.key))) ]

def plot(self):
        """Plot the empirical histogram versus best-fit distribution's PDF."""
        plt.plot(self.bin_edges, self.hist, self.bin_edges, self.best_pdf)

def closest(xarr, val):
    """ Return the index of the closest in xarr to value val """
    idx_closest = np.argmin(np.abs(np.array(xarr) - val))
    return idx_closest

def polyline(self, arr):
        """Draw a set of lines"""
        for i in range(0, len(arr) - 1):
            self.line(arr[i][0], arr[i][1], arr[i + 1][0], arr[i + 1][1])

def series_index(self, series):
        """
        Return the integer index of *series* in this sequence.
        """
        for idx, s in enumerate(self):
            if series is s:
                return idx
        raise ValueError('series not in chart data object')

def make_post_request(self, url, auth, json_payload):
        """This function executes the request with the provided
        json payload and return the json response"""
        response = requests.post(url, auth=auth, json=json_payload)
        return response.json()

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

def prettyprint(d):
        """Print dicttree in Json-like format. keys are sorted
        """
        print(json.dumps(d, sort_keys=True, 
                         indent=4, separators=("," , ": ")))

def print(*a):
    """ print just one that returns what you give it instead of None """
    try:
        _print(*a)
        return a[0] if len(a) == 1 else a
    except:
        _print(*a)

def clear(self):
        """
        Clear screen and go to 0,0
        """
        # Erase current output first.
        self.erase()

        # Send "Erase Screen" command and go to (0, 0).
        output = self.output

        output.erase_screen()
        output.cursor_goto(0, 0)
        output.flush()

        self.request_absolute_cursor_position()

def _days_in_month(date):
    """The number of days in the month of the given date"""
    if date.month == 12:
        reference = type(date)(date.year + 1, 1, 1)
    else:
        reference = type(date)(date.year, date.month + 1, 1)
    return (reference - timedelta(days=1)).day

def pprint(self, seconds):
        """
        Pretty Prints seconds as Hours:Minutes:Seconds.MilliSeconds

        :param seconds:  The time in seconds.
        """
        return ("%d:%02d:%02d.%03d", reduce(lambda ll, b: divmod(ll[0], b) + ll[1:], [(seconds * 1000,), 1000, 60, 60]))

def index(self, elem):
        """Find the index of elem in the reversed iterator."""
        return _coconut.len(self._iter) - self._iter.index(elem) - 1

def error(*args):
    """Display error message via stderr or GUI."""
    if sys.stdin.isatty():
        print('ERROR:', *args, file=sys.stderr)
    else:
        notify_error(*args)

def roc_auc(y_true, y_score):
    """
    Returns are under the ROC curve
    """
    notnull = ~np.isnan(y_true)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true[notnull], y_score[notnull])
    return sklearn.metrics.auc(fpr, tpr)

def format_exception(e):
    """Returns a string containing the type and text of the exception.

    """
    from .utils.printing import fill
    return '\n'.join(fill(line) for line in traceback.format_exception_only(type(e), e))

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

def printc(cls, txt, color=colors.red):
        """Print in color."""
        print(cls.color_txt(txt, color))

def get_rounded(self, digits):
        """ Return a vector with the elements rounded to the given number of digits. """
        result = self.copy()
        result.round(digits)
        return result

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

def parse_response(self, resp):
        """
        Parse the xmlrpc response.
        """
        p, u = self.getparser()
        p.feed(resp.content)
        p.close()
        return u.close()

def get_obj(ref):
    """Get object from string reference."""
    oid = int(ref)
    return server.id2ref.get(oid) or server.id2obj[oid]

def setup_detect_python2():
        """
        Call this before using the refactoring tools to create them on demand
        if needed.
        """
        if None in [RTs._rt_py2_detect, RTs._rtp_py2_detect]:
            RTs._rt_py2_detect = RefactoringTool(py2_detect_fixers)
            RTs._rtp_py2_detect = RefactoringTool(py2_detect_fixers,
                                                  {'print_function': True})

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

def s3(ctx, bucket_name, data_file, region):
    """Use the S3 SWAG backend."""
    if not ctx.data_file:
        ctx.data_file = data_file

    if not ctx.bucket_name:
        ctx.bucket_name = bucket_name

    if not ctx.region:
        ctx.region = region

    ctx.type = 's3'

def select_fields_as_sql(self):
        """
        Returns the selected fields or expressions as a SQL string.
        """
        return comma_join(list(self._fields) + ['%s AS %s' % (v, k) for k, v in self._calculated_fields.items()])

def kill_mprocess(process):
    """kill process
    Args:
        process - Popen object for process
    """
    if process and proc_alive(process):
        process.terminate()
        process.communicate()
    return not proc_alive(process)

def sent2features(sentence, template):
    """ extract features in a sentence

    :type sentence: list of token, each token is a list of tag
    """
    return [word2features(sentence, i, template) for i in range(len(sentence))]

def pickle_save(thing,fname):
    """save something to a pickle file"""
    pickle.dump(thing, open(fname,"wb"),pickle.HIGHEST_PROTOCOL)
    return thing

def get_randomized_guid_sample(self, item_count):
        """ Fetch a subset of randomzied GUIDs from the whitelist """
        dataset = self.get_whitelist()
        random.shuffle(dataset)
        return dataset[:item_count]

def write_image(filename, image):
    """ Write image data to PNG, JPG file

    :param filename: name of PNG or JPG file to write data to
    :type filename: str
    :param image: image data to write to file
    :type image: numpy array
    """
    data_format = get_data_format(filename)
    if data_format is MimeType.JPG:
        LOGGER.warning('Warning: jpeg is a lossy format therefore saved data will be modified.')
    return Image.fromarray(image).save(filename)

def downsample(array, k):
    """Choose k random elements of array."""
    length = array.shape[0]
    indices = random.sample(xrange(length), k)
    return array[indices]

def plot_and_save(self, **kwargs):
        """Used when the plot method defined does not create a figure nor calls save_plot
        Then the plot method has to use self.fig"""
        self.fig = pyplot.figure()
        self.plot()
        self.axes = pyplot.gca()
        self.save_plot(self.fig, self.axes, **kwargs)
        pyplot.close(self.fig)

def get(url):
    """Recieving the JSON file from uulm"""
    response = urllib.request.urlopen(url)
    data = response.read()
    data = data.decode("utf-8")
    data = json.loads(data)
    return data

def save(self, fname):
        """ Saves the dictionary in json format
        :param fname: file to save to
        """
        with open(fname, 'wb') as f:
            json.dump(self, f)

def iterate_chunks(file, chunk_size):
    """
    Iterate chunks of size chunk_size from a file-like object
    """
    chunk = file.read(chunk_size)
    while chunk:
        yield chunk
        chunk = file.read(chunk_size)

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

def validate_string_list(lst):
    """Validate that the input is a list of strings.

    Raises ValueError if not."""
    if not isinstance(lst, list):
        raise ValueError('input %r must be a list' % lst)
    for x in lst:
        if not isinstance(x, basestring):
            raise ValueError('element %r in list must be a string' % x)

def cmd_reindex():
    """Uses CREATE INDEX CONCURRENTLY to create a duplicate index, then tries to swap the new index for the original.

    The index swap is done using a short lock timeout to prevent it from interfering with running queries. Retries until
    the rename succeeds.
    """
    db = connect(args.database)
    for idx in args.indexes:
        pg_reindex(db, idx)

def is_valid_folder(parser, arg):
    """Check if arg is a valid file that already exists on the file system."""
    arg = os.path.abspath(arg)
    if not os.path.isdir(arg):
        parser.error("The folder %s does not exist!" % arg)
    else:
        return arg

def unapostrophe(text):
    """Strip apostrophe and 's' from the end of a string."""
    text = re.sub(r'[%s]s?$' % ''.join(APOSTROPHES), '', text)
    return text

def QA_util_datetime_to_strdate(dt):
    """
    :param dt:  pythone datetime.datetime
    :return:  1999-02-01 string type
    """
    strdate = "%04d-%02d-%02d" % (dt.year, dt.month, dt.day)
    return strdate

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

def extract(self):
        """
        Creates a copy of this tabarray in the form of a numpy ndarray.

        Useful if you want to do math on array elements, e.g. if you have a 
        subset of the columns that are all numerical, you can construct a 
        numerical matrix and do matrix operations.

        """
        return np.vstack([self[r] for r in self.dtype.names]).T.squeeze()

def delete_entry(self, key):
        """Delete an object from the redis table"""
        pipe = self.client.pipeline()
        pipe.srem(self.keys_container, key)
        pipe.delete(key)
        pipe.execute()

def match_files(files, pattern: Pattern):
    """Yields file name if matches a regular expression pattern."""

    for name in files:
        if re.match(pattern, name):
            yield name

def ma(self):
        """Represent data as a masked array.

        The array is returned with column-first indexing, i.e. for a data file with
        columns X Y1 Y2 Y3 ... the array a will be a[0] = X, a[1] = Y1, ... .

        inf and nan are filtered via :func:`numpy.isfinite`.
        """
        a = self.array
        return numpy.ma.MaskedArray(a, mask=numpy.logical_not(numpy.isfinite(a)))

def _selectItem(self, index):
        """Select item in the list
        """
        self._selectedIndex = index
        self.setCurrentIndex(self.model().createIndex(index, 0))

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

def percentile_index(a, q):
    """
    Returns the index of the value at the Qth percentile in array a.
    """
    return np.where(
        a==np.percentile(a, q, interpolation='nearest')
    )[0][0]

def _clean_str(self, s):
        """ Returns a lowercase string with punctuation and bad chars removed
        :param s: string to clean
        """
        return s.translate(str.maketrans('', '', punctuation)).replace('\u200b', " ").strip().lower()

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

def sanitize_word(s):
    """Remove non-alphanumerical characters from metric word.
    And trim excessive underscores.
    """
    s = re.sub('[^\w-]+', '_', s)
    s = re.sub('__+', '_', s)
    return s.strip('_')

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

def send(r, stream=False):
    """Just sends the request using its send method and returns its response.  """
    r.send(stream=stream)
    return r.response

def handle_data(self, data):
        """
        Djeffify data between tags
        """
        if data.strip():
            data = djeffify_string(data)
        self.djhtml += data

def splitext_no_dot(filename):
    """
    Wrap os.path.splitext to return the name and the extension
    without the '.' (e.g., csv instead of .csv)
    """
    name, ext = os.path.splitext(filename)
    ext = ext.lower()
    return name, ext.strip('.')

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

def autoscan():
    """autoscan will check all of the serial ports to see if they have
       a matching VID:PID for a MicroPython board.
    """
    for port in serial.tools.list_ports.comports():
        if is_micropython_usb_device(port):
            connect_serial(port[0])

def format_float(value): # not used
    """Modified form of the 'g' format specifier.
    """
    string = "{:g}".format(value).replace("e+", "e")
    string = re.sub("e(-?)0*(\d+)", r"e\1\2", string)
    return string

def remove_series(self, series):
        """Removes a :py:class:`.Series` from the chart.

        :param Series series: The :py:class:`.Series` to remove.
        :raises ValueError: if you try to remove the last\
        :py:class:`.Series`."""

        if len(self.all_series()) == 1:
            raise ValueError("Cannot remove last series from %s" % str(self))
        self._all_series.remove(series)
        series._chart = None

def _replace_nan(a, val):
    """
    replace nan in a by val, and returns the replaced array and the nan
    position
    """
    mask = isnull(a)
    return where_method(val, mask, a), mask

def internal_reset(self):
        """
        internal state reset.
        used e.g. in unittests
        """
        log.critical("PIA internal_reset()")
        self.empty_key_toggle = True
        self.current_input_char = None
        self.input_repead = 0

def get_rounded(self, digits):
        """ Return a vector with the elements rounded to the given number of digits. """
        result = self.copy()
        result.round(digits)
        return result

def get_member(thing_obj, member_string):
    """Get a member from an object by (string) name"""
    mems = {x[0]: x[1] for x in inspect.getmembers(thing_obj)}
    if member_string in mems:
        return mems[member_string]

def _replace_service_arg(self, name, index, args):
        """ Replace index in list with service """
        args[index] = self.get_instantiated_service(name)

def isbinary(*args):
    """Checks if value can be part of binary/bitwise operations."""
    return all(map(lambda c: isnumber(c) or isbool(c), args))

def _synced(method, self, args, kwargs):
    """Underlying synchronized wrapper."""
    with self._lock:
        return method(*args, **kwargs)

def _dotify(cls, data):
    """Add dots."""
    return ''.join(char if char in cls.PRINTABLE_DATA else '.' for char in data)

def is_set(self, key):
        """Return True if variable is a set"""
        data = self.model.get_data()
        return isinstance(data[key], set)

def dropna(self):
        """Returns MultiIndex without any rows containing null values according to Baloo's convention.

        Returns
        -------
        MultiIndex
            MultiIndex with no null values.

        """
        not_nas = [v.notna() for v in self.values]
        and_filter = reduce(lambda x, y: x & y, not_nas)

        return self[and_filter]

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

def round_array(array_in):
    """
    arr_out = round_array(array_in)

    Rounds an array and recasts it to int. Also works on scalars.
    """
    if isinstance(array_in, ndarray):
        return np.round(array_in).astype(int)
    else:
        return int(np.round(array_in))

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

def parallel(processes, threads):
    """
    execute jobs in processes using N threads
    """
    pool = multithread(threads)
    pool.map(run_process, processes)
    pool.close()
    pool.join()

def discard(self, element):
        """Remove element from the RangeSet if it is a member.

        If the element is not a member, do nothing.
        """
        try:
            i = int(element)
            set.discard(self, i)
        except ValueError:
            pass

def test_python_java_rt():
    """ Run Python test cases against Java runtime classes. """
    sub_env = {'PYTHONPATH': _build_dir()}

    log.info('Executing Python unit tests (against Java runtime classes)...')
    return jpyutil._execute_python_scripts(python_java_rt_tests,
                                           env=sub_env)

def _config_win32_domain(self, domain):
        """Configure a Domain registry entry."""
        # we call str() on domain to convert it from unicode to ascii
        self.domain = dns.name.from_text(str(domain))

def save_pdf(path):
  """
  Saves a pdf of the current matplotlib figure.

  :param path: str, filepath to save to
  """

  pp = PdfPages(path)
  pp.savefig(pyplot.gcf())
  pp.close()

def generate_id():
    """Generate new UUID"""
    # TODO: Use six.string_type to Py3 compat
    try:
        return unicode(uuid1()).replace(u"-", u"")
    except NameError:
        return str(uuid1()).replace(u"-", u"")

def _save_file(self, filename, contents):
        """write the html file contents to disk"""
        with open(filename, 'w') as f:
            f.write(contents)

def is_set(self, key):
        """Return True if variable is a set"""
        data = self.model.get_data()
        return isinstance(data[key], set)

def get_all_attributes(klass_or_instance):
    """Get all attribute members (attribute, property style method).
    """
    pairs = list()
    for attr, value in inspect.getmembers(
            klass_or_instance, lambda x: not inspect.isroutine(x)):
        if not (attr.startswith("__") or attr.endswith("__")):
            pairs.append((attr, value))
    return pairs

def unique_everseen(seq):
    """Solution found here : http://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-python-whilst-preserving-order"""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def get_obj_cols(df):
    """
    Returns names of 'object' columns in the DataFrame.
    """
    obj_cols = []
    for idx, dt in enumerate(df.dtypes):
        if dt == 'object' or is_category(dt):
            obj_cols.append(df.columns.values[idx])

    return obj_cols

def traverse_setter(obj, attribute, value):
    """
    Traverses the object and sets the supplied attribute on the
    object. Supports Dimensioned and DimensionedPlot types.
    """
    obj.traverse(lambda x: setattr(x, attribute, value))

def is_valid_varname(varname):
    """ Checks syntax and validity of a variable name """
    if not isinstance(varname, six.string_types):
        return False
    match_obj = re.match(varname_regex, varname)
    valid_syntax = match_obj is not None
    valid_name = not keyword.iskeyword(varname)
    isvalid = valid_syntax and valid_name
    return isvalid

def setAutoRangeOn(self, axisNumber):
        """ Sets the auto-range of the axis on.

            :param axisNumber: 0 (X-axis), 1 (Y-axis), 2, (Both X and Y axes).
        """
        setXYAxesAutoRangeOn(self, self.xAxisRangeCti, self.yAxisRangeCti, axisNumber)

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

def wait_on_rate_limit(self, value):
        """Enable or disable automatic rate-limit handling."""
        check_type(value, bool, may_be_none=False)
        self._wait_on_rate_limit = value

def get_last(self, table=None):
        """Just the last entry."""
        if table is None: table = self.main_table
        query = 'SELECT * FROM "%s" ORDER BY ROWID DESC LIMIT 1;' % table
        return self.own_cursor.execute(query).fetchone()

def discard(self, element):
        """Remove element from the RangeSet if it is a member.

        If the element is not a member, do nothing.
        """
        try:
            i = int(element)
            set.discard(self, i)
        except ValueError:
            pass

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

def sys_pipes_forever(encoding=_default_encoding):
    """Redirect all C output to sys.stdout/err
    
    This is not a context manager; it turns on C-forwarding permanently.
    """
    global _mighty_wurlitzer
    if _mighty_wurlitzer is None:
        _mighty_wurlitzer = sys_pipes(encoding)
    _mighty_wurlitzer.__enter__()

def _check_for_errors(etree: ET.ElementTree):
    """Check AniDB response XML tree for errors."""
    if etree.getroot().tag == 'error':
        raise APIError(etree.getroot().text)

def previous_friday(dt):
    """
    If holiday falls on Saturday or Sunday, use previous Friday instead.
    """
    if dt.weekday() == 5:
        return dt - timedelta(1)
    elif dt.weekday() == 6:
        return dt - timedelta(2)
    return dt

def save_model(self, request, obj, form, change):
        """
        Set currently authenticated user as the author of the gallery.
        """
        obj.author = request.user
        obj.save()

def set_position(self, x, y, width, height):
        """Set window top-left corner position and size"""
        SetWindowPos(self._hwnd, None, x, y, width, height, ctypes.c_uint(0))

def set_constraint_bound(self, name, value):
        """Set the upper bound of a constraint."""
        index = self._get_constraint_index(name)
        self.upper_bounds[index] = value
        self._reset_solution()

def log_y_cb(self, w, val):
        """Toggle linear/log scale for Y-axis."""
        self.tab_plot.logy = val
        self.plot_two_columns()

def round_to_int(number, precision):
    """Round a number to a precision"""
    precision = int(precision)
    rounded = (int(number) + precision / 2) // precision * precision
    return rounded

def log_y_cb(self, w, val):
        """Toggle linear/log scale for Y-axis."""
        self.tab_plot.logy = val
        self.plot_two_columns()

def set_axis_options(self, row, column, text):
        """Set additionnal options as plain text."""

        subplot = self.get_subplot_at(row, column)
        subplot.set_axis_options(text)

def set_default(self, key, value):
        """Set the default value for this key.
        Default only used when no value is provided by the user via
        arg, config or env.
        """
        k = self._real_key(key.lower())
        self._defaults[k] = value

def batchify(data, batch_size):
    """Reshape data into (num_example, batch_size)"""
    nbatch = data.shape[0] // batch_size
    data = data[:nbatch * batch_size]
    data = data.reshape((batch_size, nbatch)).T
    return data

def stringc(text, color):
    """
    Return a string with terminal colors.
    """
    if has_colors:
        text = str(text)

        return "\033["+codeCodes[color]+"m"+text+"\033[0m"
    else:
        return text

def extent(self):
        """Helper for matplotlib imshow"""
        return (
            self.intervals[1].pix1 - 0.5,
            self.intervals[1].pix2 - 0.5,
            self.intervals[0].pix1 - 0.5,
            self.intervals[0].pix2 - 0.5,
        )

def _match_space_at_line(line):
    """Return a re.match object if an empty comment was found on line."""
    regex = re.compile(r"^{0}$".format(_MDL_COMMENT))
    return regex.match(line)

def chunks(iterable, chunk):
    """Yield successive n-sized chunks from an iterable."""
    for i in range(0, len(iterable), chunk):
        yield iterable[i:i + chunk]

def _print_memory(self, memory):
        """Print memory.
        """
        for addr, value in memory.items():
            print("    0x%08x : 0x%08x (%d)" % (addr, value, value))

def Slice(a, begin, size):
    """
    Slicing op.
    """
    return np.copy(a)[[slice(*tpl) for tpl in zip(begin, begin+size)]],

def basic_word_sim(word1, word2):
    """
    Simple measure of similarity: Number of letters in common / max length
    """
    return sum([1 for c in word1 if c in word2]) / max(len(word1), len(word2))

def sort_by_name(self):
        """Sort list elements by name."""
        super(JSSObjectList, self).sort(key=lambda k: k.name)

def cached_query(qs, timeout=None):
    """ Auto cached queryset and generate results.
    """
    cache_key = generate_cache_key(qs)
    return get_cached(cache_key, list, args=(qs,), timeout=None)

def sort_filenames(filenames):
    """
    sort a list of files by filename only, ignoring the directory names
    """
    basenames = [os.path.basename(x) for x in filenames]
    indexes = [i[0] for i in sorted(enumerate(basenames), key=lambda x:x[1])]
    return [filenames[x] for x in indexes]

def lin_interp(x, rangeX, rangeY):
    """
    Interpolate linearly variable x in rangeX onto rangeY.
    """
    s = (x - rangeX[0]) / mag(rangeX[1] - rangeX[0])
    y = rangeY[0] * (1 - s) + rangeY[1] * s
    return y

def csort(objs, key):
    """Order-preserving sorting function."""
    idxs = dict((obj, i) for (i, obj) in enumerate(objs))
    return sorted(objs, key=lambda obj: (key(obj), idxs[obj]))

def is_full_slice(obj, l):
    """
    We have a full length slice.
    """
    return (isinstance(obj, slice) and obj.start == 0 and obj.stop == l and
            obj.step is None)

def sort_filenames(filenames):
    """
    sort a list of files by filename only, ignoring the directory names
    """
    basenames = [os.path.basename(x) for x in filenames]
    indexes = [i[0] for i in sorted(enumerate(basenames), key=lambda x:x[1])]
    return [filenames[x] for x in indexes]

def is_full_slice(obj, l):
    """
    We have a full length slice.
    """
    return (isinstance(obj, slice) and obj.start == 0 and obj.stop == l and
            obj.step is None)

def fsliceafter(astr, sub):
    """Return the slice after at sub in string astr"""
    findex = astr.find(sub)
    return astr[findex + len(sub):]

def validate_type(self, type_):
        """Take an str/unicode `type_` and raise a ValueError if it's not 
        a valid type for the object.
        
        A valid type for a field is a value from the types_set attribute of 
        that field's class. 
        
        """
        if type_ is not None and type_ not in self.types_set:
            raise ValueError('Invalid type for %s:%s' % (self.__class__, type_))

def movingaverage(arr, window):
    """
    Calculates the moving average ("rolling mean") of an array
    of a certain window size.
    """
    m = np.ones(int(window)) / int(window)
    return scipy.ndimage.convolve1d(arr, m, axis=0, mode='reflect')

def mpl_outside_legend(ax, **kwargs):
    """ Places a legend box outside a matplotlib Axes instance. """
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), **kwargs)

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

def split(s):
  """Uses dynamic programming to infer the location of spaces in a string without spaces."""
  l = [_split(x) for x in _SPLIT_RE.split(s)]
  return [item for sublist in l for item in sublist]

def pick_unused_port(self):
    """ Pick an unused port. There is a slight chance that this wont work. """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', 0))
    _, port = s.getsockname()
    s.close()
    return port

def split(s):
  """Uses dynamic programming to infer the location of spaces in a string without spaces."""
  l = [_split(x) for x in _SPLIT_RE.split(s)]
  return [item for sublist in l for item in sublist]

def run(context, port):
    """ Run the Webserver/SocketIO and app
    """
    global ctx
    ctx = context
    app.run(port=port)

def _split_str(s, n):
    """
    split string into list of strings by specified number.
    """
    length = len(s)
    return [s[i:i + n] for i in range(0, length, n)]

def enter_room(self, sid, room, namespace=None):
        """Enter a room.

        The only difference with the :func:`socketio.Server.enter_room` method
        is that when the ``namespace`` argument is not given the namespace
        associated with the class is used.
        """
        return self.server.enter_room(sid, room,
                                      namespace=namespace or self.namespace)

def end(self):
        """End of the Glances server session."""
        if not self.args.disable_autodiscover:
            self.autodiscover_client.close()
        self.server.end()

def argsort_indices(a, axis=-1):
    """Like argsort, but returns an index suitable for sorting the
    the original array even if that array is multidimensional
    """
    a = np.asarray(a)
    ind = list(np.ix_(*[np.arange(d) for d in a.shape]))
    ind[axis] = a.argsort(axis)
    return tuple(ind)

def delistify(x):
    """ A basic slug version of a given parameter list. """
    if isinstance(x, list):
        x = [e.replace("'", "") for e in x]
        return '-'.join(sorted(x))
    return x

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

def _dict_values_sorted_by_key(dictionary):
    # This should be a yield from instead.
    """Internal helper to return the values of a dictionary, sorted by key.
    """
    for _, value in sorted(dictionary.iteritems(), key=operator.itemgetter(0)):
        yield value

def query_sum(queryset, field):
    """
    Let the DBMS perform a sum on a queryset
    """
    return queryset.aggregate(s=models.functions.Coalesce(models.Sum(field), 0))['s']

def sort_filenames(filenames):
    """
    sort a list of files by filename only, ignoring the directory names
    """
    basenames = [os.path.basename(x) for x in filenames]
    indexes = [i[0] for i in sorted(enumerate(basenames), key=lambda x:x[1])]
    return [filenames[x] for x in indexes]

def disown(cmd):
    """Call a system command in the background,
       disown it and hide it's output."""
    subprocess.Popen(cmd,
                     stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)

def sort_filenames(filenames):
    """
    sort a list of files by filename only, ignoring the directory names
    """
    basenames = [os.path.basename(x) for x in filenames]
    indexes = [i[0] for i in sorted(enumerate(basenames), key=lambda x:x[1])]
    return [filenames[x] for x in indexes]

def to_capitalized_camel_case(snake_case_string):
    """
    Convert a string from snake case to camel case with the first letter capitalized. For example, "some_var"
    would become "SomeVar".

    :param snake_case_string: Snake-cased string to convert to camel case.
    :returns: Camel-cased version of snake_case_string.
    """
    parts = snake_case_string.split('_')
    return ''.join([i.title() for i in parts])

def sort_by_name(self):
        """Sort list elements by name."""
        super(JSSObjectList, self).sort(key=lambda k: k.name)

def inh(table):
    """
    inverse hyperbolic sine transformation
    """
    t = []
    for i in table:
        t.append(np.ndarray.tolist(np.arcsinh(i)))
    return t

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

def _root_mean_square_error(y, y_pred, w):
    """Calculate the root mean square error."""
    return np.sqrt(np.average(((y_pred - y) ** 2), weights=w))

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

def im2mat(I):
    """Converts and image to matrix (one pixel per line)"""
    return I.reshape((I.shape[0] * I.shape[1], I.shape[2]))

def toArray(self):
        """
        Returns a copy of this SparseVector as a 1-dimensional NumPy array.
        """
        arr = np.zeros((self.size,), dtype=np.float64)
        arr[self.indices] = self.values
        return arr

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

def step_next_line(self):
        """Sets cursor as beginning of next line."""
        self._eol.append(self.position)
        self._lineno += 1
        self._col_offset = 0

def datatype(dbtype, description, cursor):
    """Google AppEngine Helper to convert a data type into a string."""
    dt = cursor.db.introspection.get_field_type(dbtype, description)
    if type(dt) is tuple:
        return dt[0]
    else:
        return dt

def read_sphinx_environment(pth):
    """Read the sphinx environment.pickle file at path `pth`."""

    with open(pth, 'rb') as fo:
        env = pickle.load(fo)
    return env

def _is_image_sequenced(image):
    """Determine if the image is a sequenced image."""
    try:
        image.seek(1)
        image.seek(0)
        result = True
    except EOFError:
        result = False

    return result

def consecutive(data, stepsize=1):
    """Converts array into chunks with consecutive elements of given step size.
    http://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-from-an-array-in-numpy
    """
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)

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

def split_into_sentences(s):
  """Split text into list of sentences."""
  s = re.sub(r"\s+", " ", s)
  s = re.sub(r"[\\.\\?\\!]", "\n", s)
  return s.split("\n")

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

def split_multiline(value):
    """Split a multiline string into a list, excluding blank lines."""
    return [element for element in (line.strip() for line in value.split('\n'))
            if element]

def __contains__ (self, key):
        """Check lowercase key item."""
        assert isinstance(key, basestring)
        return dict.__contains__(self, key.lower())

def cleanLines(source, lineSep=os.linesep):
    """
    :param source: some iterable source (list, file, etc)
    :param lineSep: string of separators (chars) that must be removed
    :return: list of non empty lines with removed separators
    """
    stripped = (line.strip(lineSep) for line in source)
    return (line for line in stripped if len(line) != 0)

def tokenize(string):
    """Match and yield all the tokens of the input string."""
    for match in TOKENS_REGEX.finditer(string):
        yield Token(match.lastgroup, match.group().strip(), match.span())

def reindent(s, numspaces):
    """ reinidents a string (s) by the given number of spaces (numspaces) """
    leading_space = numspaces * ' '
    lines = [leading_space + line.strip()for line in s.splitlines()]
    return '\n'.join(lines)

def wrap_count(method):
    """
    Returns number of wraps around given method.
    """
    number = 0
    while hasattr(method, '__aspects_orig'):
        number += 1
        method = method.__aspects_orig
    return number

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

def list_i2str(ilist):
    """
    Convert an integer list into a string list.
    """
    slist = []
    for el in ilist:
        slist.append(str(el))
    return slist

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

def length(self):
        """Array of vector lengths"""
        return np.sqrt(np.sum(self**2, axis=1)).view(np.ndarray)

def _bindingsToDict(self, bindings):
        """
        Given a binding from the sparql query result,
        create a dict of plain text
        """
        myDict = {}
        for key, val in bindings.iteritems():
            myDict[key.toPython().replace('?', '')] = val.toPython()
        return myDict

def compose_all(tups):
  """Compose all given tuples together."""
  from . import ast  # I weep for humanity
  return functools.reduce(lambda x, y: x.compose(y), map(ast.make_tuple, tups), ast.make_tuple({}))

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

def visit_Str(self, node):
        """ Set the pythonic string type. """
        self.result[node] = self.builder.NamedType(pytype_to_ctype(str))

def get_count(self, query):
        """
        Returns a number of query results. This is faster than .count() on the query
        """
        count_q = query.statement.with_only_columns(
            [func.count()]).order_by(None)
        count = query.session.execute(count_q).scalar()
        return count

def downgrade(directory, sql, tag, x_arg, revision):
    """Revert to a previous version"""
    _downgrade(directory, revision, sql, tag, x_arg)

def unlock(self):
    """Closes the session to the database."""
    if not hasattr(self, 'session'):
      raise RuntimeError('Error detected! The session that you want to close does not exist any more!')
    logger.debug("Closed database session of '%s'" % self._database)
    self.session.close()
    del self.session

def cpp_prog_builder(build_context, target):
    """Build a C++ binary executable"""
    yprint(build_context.conf, 'Build CppProg', target)
    workspace_dir = build_context.get_workspace('CppProg', target.name)
    build_cpp(build_context, target, target.compiler_config, workspace_dir)

def column_exists(cr, table, column):
    """ Check whether a certain column exists """
    cr.execute(
        'SELECT count(attname) FROM pg_attribute '
        'WHERE attrelid = '
        '( SELECT oid FROM pg_class WHERE relname = %s ) '
        'AND attname = %s',
        (table, column))
    return cr.fetchone()[0] == 1

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

def sqliteRowsToDicts(sqliteRows):
    """
    Unpacks sqlite rows as returned by fetchall
    into an array of simple dicts.

    :param sqliteRows: array of rows returned from fetchall DB call
    :return:  array of dicts, keyed by the column names.
    """
    return map(lambda r: dict(zip(r.keys(), r)), sqliteRows)

def dict_hash(dct):
    """Return a hash of the contents of a dictionary"""
    dct_s = json.dumps(dct, sort_keys=True)

    try:
        m = md5(dct_s)
    except TypeError:
        m = md5(dct_s.encode())

    return m.hexdigest()

def get_type_len(self):
        """Retrieve the type and length for a data record."""
        # Check types and set type/len
        self.get_sql()
        return self.type, self.len, self.len_decimal

def replace_all(filepath, searchExp, replaceExp):
    """
    Replace all the ocurrences (in a file) of a string with another value.
    """
    for line in fileinput.input(filepath, inplace=1):
        if searchExp in line:
            line = line.replace(searchExp, replaceExp)
        sys.stdout.write(line)

def get_table_names(connection):
	"""
	Return a list of the table names in the database.
	"""
	cursor = connection.cursor()
	cursor.execute("SELECT name FROM sqlite_master WHERE type == 'table'")
	return [name for (name,) in cursor]

def list_i2str(ilist):
    """
    Convert an integer list into a string list.
    """
    slist = []
    for el in ilist:
        slist.append(str(el))
    return slist

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

def ensure_tuple(obj):
    """Try and make the given argument into a tuple."""
    if obj is None:
        return tuple()
    if isinstance(obj, Iterable) and not isinstance(obj, six.string_types):
        return tuple(obj)
    return obj,

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

def All(sequence):
  """
  :param sequence: Any sequence whose elements can be evaluated as booleans.
  :returns: true if all elements of the sequence satisfy True and x.
  """
  return bool(reduce(lambda x, y: x and y, sequence, True))

def weighted_std(values, weights):
    """ Calculate standard deviation weighted by errors """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return np.sqrt(variance)

def calculate_bounding_box(data):
    """
    Returns a 2 x m array indicating the min and max along each
    dimension.
    """
    mins = data.min(0)
    maxes = data.max(0)
    return mins, maxes

def is_static(*p):
    """ A static value (does not change at runtime)
    which is known at compile time
    """
    return all(is_CONST(x) or
               is_number(x) or
               is_const(x)
               for x in p)

def _create_complete_graph(node_ids):
    """Create a complete graph from the list of node ids.

    Args:
        node_ids: a list of node ids

    Returns:
        An undirected graph (as a networkx.Graph)
    """
    g = nx.Graph()
    g.add_nodes_from(node_ids)
    for (i, j) in combinations(node_ids, 2):
        g.add_edge(i, j)
    return g

def static_method(cls, f):
        """Decorator which dynamically binds static methods to the model for later use."""
        setattr(cls, f.__name__, staticmethod(f))
        return f

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

def glr_path_static():
    """Returns path to packaged static files"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '_static'))

def __connect():
    """
    Connect to a redis instance.
    """
    global redis_instance
    if use_tcp_socket:
        redis_instance = redis.StrictRedis(host=hostname, port=port)
    else:
        redis_instance = redis.StrictRedis(unix_socket_path=unix_socket)

def _update_staticmethod(self, oldsm, newsm):
        """Update a staticmethod update."""
        # While we can't modify the staticmethod object itself (it has no
        # mutable attributes), we *can* extract the underlying function
        # (by calling __get__(), which returns it) and update it in-place.
        # We don't have the class available to pass to __get__() but any
        # object except None will do.
        self._update(None, None, oldsm.__get__(0), newsm.__get__(0))

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

def _update_staticmethod(self, oldsm, newsm):
        """Update a staticmethod update."""
        # While we can't modify the staticmethod object itself (it has no
        # mutable attributes), we *can* extract the underlying function
        # (by calling __get__(), which returns it) and update it in-place.
        # We don't have the class available to pass to __get__() but any
        # object except None will do.
        self._update(None, None, oldsm.__get__(0), newsm.__get__(0))

def apply(f, obj, *args, **kwargs):
    """Apply a function in parallel to each element of the input"""
    return vectorize(f)(obj, *args, **kwargs)

def standard_deviation(numbers):
    """Return standard deviation."""
    numbers = list(numbers)
    if not numbers:
        return 0
    mean = sum(numbers) / len(numbers)
    return (sum((n - mean) ** 2 for n in numbers) /
            len(numbers)) ** .5

def norm_vec(vector):
    """Normalize the length of a vector to one"""
    assert len(vector) == 3
    v = np.array(vector)
    return v/np.sqrt(np.sum(v**2))

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

def length(self):
        """Array of vector lengths"""
        return np.sqrt(np.sum(self**2, axis=1)).view(np.ndarray)

def parse_json_date(value):
    """
    Parses an ISO8601 formatted datetime from a string value
    """
    if not value:
        return None

    return datetime.datetime.strptime(value, JSON_DATETIME_FORMAT).replace(tzinfo=pytz.UTC)

def fetch_header(self):
        """Make a header request to the endpoint."""
        query = self.query().add_query_parameter(req='header')
        return self._parse_messages(self.get_query(query).content)[0]

def dt2str(dt, flagSeconds=True):
    """Converts datetime object to str if not yet an str."""
    if isinstance(dt, str):
        return dt
    return dt.strftime(_FMTS if flagSeconds else _FMT)

def add_device_callback(self, callback):
        """Register a callback to be invoked when a new device appears."""
        _LOGGER.debug('Added new callback %s ', callback)
        self._cb_new_device.append(callback)

def any_contains_any(strings, candidates):
    """Whether any of the strings contains any of the candidates."""
    for string in strings:
        for c in candidates:
            if c in string:
                return True

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

def safe_format(s, **kwargs):
  """
  :type s str
  """
  return string.Formatter().vformat(s, (), defaultdict(str, **kwargs))

def build_output(self, fout):
        """Squash self.out into string.

        Join every line in self.out with a new line and write the
        result to the output file.
        """
        fout.write('\n'.join([s for s in self.out]))

def format_line(data, linestyle):
    """Formats a list of elements using the given line style"""
    return linestyle.begin + linestyle.sep.join(data) + linestyle.end

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

def _strvar(a, prec='{:G}'):
    r"""Return variable as a string to print, with given precision."""
    return ' '.join([prec.format(i) for i in np.atleast_1d(a)])

def entropy(string):
    """Compute entropy on the string"""
    p, lns = Counter(string), float(len(string))
    return -sum(count/lns * math.log(count/lns, 2) for count in p.values())

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

def write_line(self, line, count=1):
        """writes the line and count newlines after the line"""
        self.write(line)
        self.write_newlines(count)

def make_strslice(lineno, s, lower, upper):
    """ Wrapper: returns String Slice node
    """
    return symbols.STRSLICE.make_node(lineno, s, lower, upper)

def filedata(self):
        """Property providing access to the :class:`.FileDataAPI`"""
        if self._filedata_api is None:
            self._filedata_api = self.get_filedata_api()
        return self._filedata_api

def _str_to_list(s):
    """Converts a comma separated string to a list"""
    _list = s.split(",")
    return list(map(lambda i: i.lstrip(), _list))

def backward_char(self, e): # (C-b)
        u"""Move back a character. """
        self.l_buffer.backward_char(self.argument_reset)
        self.finalize()

def top(n, width=WIDTH, style=STYLE):
    """Prints the top row of a table"""
    return hrule(n, width, linestyle=STYLES[style].top)

def _repr_strip(mystring):
    """
    Returns the string without any initial or final quotes.
    """
    r = repr(mystring)
    if r.startswith("'") and r.endswith("'"):
        return r[1:-1]
    else:
        return r

def html_to_text(content):
    """ Converts html content to plain text """
    text = None
    h2t = html2text.HTML2Text()
    h2t.ignore_links = False
    text = h2t.handle(content)
    return text

def text_remove_empty_lines(text):
    """
    Whitespace normalization:

      - Strip empty lines
      - Strip trailing whitespace
    """
    lines = [ line.rstrip()  for line in text.splitlines()  if line.strip() ]
    return "\n".join(lines)

def pretty(obj, verbose=False, max_width=79, newline='\n'):
    """
    Pretty print the object's representation.
    """
    stream = StringIO()
    printer = RepresentationPrinter(stream, verbose, max_width, newline)
    printer.pretty(obj)
    printer.flush()
    return stream.getvalue()

def strip_spaces(value, sep=None, join=True):
    """Cleans trailing whitespaces and replaces also multiple whitespaces with a single space."""
    value = value.strip()
    value = [v.strip() for v in value.split(sep)]
    join_sep = sep or ' '
    return join_sep.join(value) if join else value

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

def lines(input):
    """Remove comments and empty lines"""
    for raw_line in input:
        line = raw_line.strip()
        if line and not line.startswith('#'):
            yield strip_comments(line)

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

def All(sequence):
  """
  :param sequence: Any sequence whose elements can be evaluated as booleans.
  :returns: true if all elements of the sequence satisfy True and x.
  """
  return bool(reduce(lambda x, y: x and y, sequence, True))

def set_title(self, title, **kwargs):
        """Sets the title on the underlying matplotlib AxesSubplot."""
        ax = self.get_axes()
        ax.set_title(title, **kwargs)

def return_letters_from_string(text):
    """Get letters from string only."""
    out = ""
    for letter in text:
        if letter.isalpha():
            out += letter
    return out

def show_yticklabels(self, row, column):
        """Show the y-axis tick labels for a subplot.

        :param row,column: specify the subplot.

        """
        subplot = self.get_subplot_at(row, column)
        subplot.show_yticklabels()

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

def correspond(text):
    """Communicate with the child process without closing stdin."""
    subproc.stdin.write(text)
    subproc.stdin.flush()
    return drain()

def numberp(v):
    """Return true iff 'v' is a number."""
    return (not(isinstance(v, bool)) and
            (isinstance(v, int) or isinstance(v, float)))

def disown(cmd):
    """Call a system command in the background,
       disown it and hide it's output."""
    subprocess.Popen(cmd,
                     stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)

def is_float(value):
    """must be a float"""
    return isinstance(value, float) or isinstance(value, int) or isinstance(value, np.float64), float(value)

def query_sum(queryset, field):
    """
    Let the DBMS perform a sum on a queryset
    """
    return queryset.aggregate(s=models.functions.Coalesce(models.Sum(field), 0))['s']

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

def transformer(data, label):
    """ data preparation """
    data = mx.image.imresize(data, IMAGE_SIZE, IMAGE_SIZE)
    data = mx.nd.transpose(data, (2, 0, 1))
    data = data.astype(np.float32) / 128.0 - 1
    return data, label

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

def resetScale(self):
        """Resets the scale on this image. Correctly aligns time scale, undoes manual scaling"""
        self.img.scale(1./self.imgScale[0], 1./self.imgScale[1])
        self.imgScale = (1.,1.)

def aug_sysargv(cmdstr):
    """ DEBUG FUNC modify argv to look like you ran a command """
    import shlex
    argv = shlex.split(cmdstr)
    sys.argv.extend(argv)

def get_point_hash(self, point):
        """
        return geohash for given point with self.precision
        :param point: GeoPoint instance
        :return: string
        """
        return geohash.encode(point.latitude, point.longitude, self.precision)

def aug_sysargv(cmdstr):
    """ DEBUG FUNC modify argv to look like you ran a command """
    import shlex
    argv = shlex.split(cmdstr)
    sys.argv.extend(argv)

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

def paint(self, tbl):
        """
        Paint the table on terminal
        Currently only print out basic string format
        """
        if not isinstance(tbl, Table):
            logging.error("unable to paint table: invalid object")
            return False

        self.term.stream.write(self.term.clear)

        self.term.stream.write(str(tbl))
        return True

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

def timedelta2millisecond(td):
    """Get milliseconds from a timedelta."""
    milliseconds = td.days * 24 * 60 * 60 * 1000
    milliseconds += td.seconds * 1000
    milliseconds += td.microseconds / 1000
    return milliseconds

def get_truetype(value):
    """Convert a string to a pythonized parameter."""
    if value in ["true", "True", "y", "Y", "yes"]:
        return True
    if value in ["false", "False", "n", "N", "no"]:
        return False
    if value.isdigit():
        return int(value)
    return str(value)

def split(s):
  """Uses dynamic programming to infer the location of spaces in a string without spaces."""
  l = [_split(x) for x in _SPLIT_RE.split(s)]
  return [item for sublist in l for item in sublist]

def __getitem__(self, name):
        """
        A pymongo-like behavior for dynamically obtaining a collection of documents
        """
        if name not in self._collections:
            self._collections[name] = Collection(self.db, name)
        return self._collections[name]

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

def zero_pad(m, n=1):
    """Pad a matrix with zeros, on all sides."""
    return np.pad(m, (n, n), mode='constant', constant_values=[0])

def logger(message, level=10):
    """Handle logging."""
    logging.getLogger(__name__).log(level, str(message))

def do_next(self, args):
        """Step over the next statement
        """
        self._do_print_from_last_cmd = True
        self._interp.step_over()
        return True

def _get_example_length(example):
  """Returns the maximum length between the example inputs and targets."""
  length = tf.maximum(tf.shape(example[0])[0], tf.shape(example[1])[0])
  return length

def get_average_length_of_string(strings):
    """Computes average length of words

    :param strings: list of words
    :return: Average length of word on list
    """
    if not strings:
        return 0

    return sum(len(word) for word in strings) / len(strings)

def flatten4d3d(x):
  """Flatten a 4d-tensor into a 3d-tensor by joining width and height."""
  xshape = shape_list(x)
  result = tf.reshape(x, [xshape[0], xshape[1] * xshape[2], xshape[3]])
  return result

def previous_friday(dt):
    """
    If holiday falls on Saturday or Sunday, use previous Friday instead.
    """
    if dt.weekday() == 5:
        return dt - timedelta(1)
    elif dt.weekday() == 6:
        return dt - timedelta(2)
    return dt

def is_complex(dtype):
  """Returns whether this is a complex floating point type."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'is_complex'):
    return dtype.is_complex
  return np.issubdtype(np.dtype(dtype), np.complex)

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

def _assert_is_type(name, value, value_type):
    """Assert that a value must be a given type."""
    if not isinstance(value, value_type):
        if type(value_type) is tuple:
            types = ', '.join(t.__name__ for t in value_type)
            raise ValueError('{0} must be one of ({1})'.format(name, types))
        else:
            raise ValueError('{0} must be {1}'
                             .format(name, value_type.__name__))

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

def is_lazy_iterable(obj):
    """
    Returns whether *obj* is iterable lazily, such as generators, range objects, etc.
    """
    return isinstance(obj,
        (types.GeneratorType, collections.MappingView, six.moves.range, enumerate))

def is_in(self, search_list, pair):
        """
        If pair is in search_list, return the index. Otherwise return -1
        """
        index = -1
        for nr, i in enumerate(search_list):
            if(np.all(i == pair)):
                return nr
        return index

def is_readable(filename):
    """Check if file is a regular file and is readable."""
    return os.path.isfile(filename) and os.access(filename, os.R_OK)

def find_nearest_index(arr, value):
    """For a given value, the function finds the nearest value
    in the array and returns its index."""
    arr = np.array(arr)
    index = (abs(arr-value)).argmin()
    return index

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

def index(m, val):
    """
    Return the indices of all the ``val`` in ``m``
    """
    mm = np.array(m)
    idx_tuple = np.where(mm == val)
    idx = idx_tuple[0].tolist()

    return idx

def requests_post(url, data=None, json=None, **kwargs):
    """Requests-mock requests.post wrapper."""
    return requests_request('post', url, data=data, json=json, **kwargs)

def merge(left, right, how='inner', key=None, left_key=None, right_key=None,
          left_as='left', right_as='right'):
    """ Performs a join using the union join function. """
    return join(left, right, how, key, left_key, right_key,
                join_fn=make_union_join(left_as, right_as))

def _genTex2D(self):
        """Generate an empty texture in OpenGL"""
        for face in range(6):
            gl.glTexImage2D(self.target0 + face, 0, self.internal_fmt, self.width, self.height, 0,
                            self.pixel_fmt, gl.GL_UNSIGNED_BYTE, 0)

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

def ci(a, which=95, axis=None):
    """Return a percentile range from an array of values."""
    p = 50 - which / 2, 50 + which / 2
    return percentiles(a, p, axis)

def int2str(num, radix=10, alphabet=BASE85):
    """helper function for quick base conversions from integers to strings"""
    return NumConv(radix, alphabet).int2str(num)

def run(self):
        """Run the event loop."""
        self.signal_init()
        self.listen_init()
        self.logger.info('starting')
        self.loop.start()

def int_to_date(date):
    """
    Convert an int of form yyyymmdd to a python date object.
    """

    year = date // 10**4
    month = date % 10**4 // 10**2
    day = date % 10**2

    return datetime.date(year, month, day)

def previous_quarter(d):
    """
    Retrieve the previous quarter for dt
    """
    from django_toolkit.datetime_util import quarter as datetime_quarter
    return quarter( (datetime_quarter(datetime(d.year, d.month, d.day))[0] + timedelta(days=-1)).date() )

def _linearInterpolationTransformMatrix(matrix1, matrix2, value):
    """ Linear, 'oldstyle' interpolation of the transform matrix."""
    return tuple(_interpolateValue(matrix1[i], matrix2[i], value) for i in range(len(matrix1)))

def start(self):
        """Start the receiver.
        """
        if not self._is_running:
            self._do_run = True
            self._thread.start()
        return self

def is_valid_uid(uid):
    """
    :return: True if it is a valid DHIS2 UID, False if not
    """
    pattern = r'^[A-Za-z][A-Za-z0-9]{10}$'
    if not isinstance(uid, string_types):
        return False
    return bool(re.compile(pattern).match(uid))

def _synced(method, self, args, kwargs):
    """Underlying synchronized wrapper."""
    with self._lock:
        return method(*args, **kwargs)

def _check_color_dim(val):
    """Ensure val is Nx(n_col), usually Nx3"""
    val = np.atleast_2d(val)
    if val.shape[1] not in (3, 4):
        raise RuntimeError('Value must have second dimension of size 3 or 4')
    return val, val.shape[1]

def join(self):
		"""Note that the Executor must be close()'d elsewhere,
		or join() will never return.
		"""
		self.inputfeeder_thread.join()
		self.pool.join()
		self.resulttracker_thread.join()
		self.failuretracker_thread.join()

def seconds(num):
    """
    Pause for this many seconds
    """
    now = pytime.time()
    end = now + num
    until(end)

def format_time(time):
    """ Formats the given time into HH:MM:SS """
    h, r = divmod(time / 1000, 3600)
    m, s = divmod(r, 60)

    return "%02d:%02d:%02d" % (h, m, s)

def invertDictMapping(d):
    """ Invert mapping of dictionary (i.e. map values to list of keys) """
    inv_map = {}
    for k, v in d.items():
        inv_map[v] = inv_map.get(v, [])
        inv_map[v].append(k)
    return inv_map

def fileModifiedTimestamp(fname):
    """return "YYYY-MM-DD" when the file was modified."""
    modifiedTime=os.path.getmtime(fname)
    stamp=time.strftime('%Y-%m-%d', time.localtime(modifiedTime))
    return stamp

def start(self):
        """Activate the TypingStream on stdout"""
        self.streams.append(sys.stdout)
        sys.stdout = self.stream

def localize(dt):
    """Localize a datetime object to local time."""
    if dt.tzinfo is UTC:
        return (dt + LOCAL_UTC_OFFSET).replace(tzinfo=None)
    # No TZ info so not going to assume anything, return as-is.
    return dt

def as_tuple(self, value):
        """Utility function which converts lists to tuples."""
        if isinstance(value, list):
            value = tuple(value)
        return value

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

def __hash__(self):
        """Return ``hash(self)``."""
        return hash((type(self), self.domain, self.range, self.partition))

def set_cursor_position(self, position):
        """Set cursor position"""
        position = self.get_position(position)
        cursor = self.textCursor()
        cursor.setPosition(position)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

def downcaseTokens(s,l,t):
    """Helper parse action to convert tokens to lower case."""
    return [ tt.lower() for tt in map(_ustr,t) ]

def _set_scroll_v(self, *args):
        """Scroll both categories Canvas and scrolling container"""
        self._canvas_categories.yview(*args)
        self._canvas_scroll.yview(*args)

def gaussian_variogram_model(m, d):
    """Gaussian model, m is [psill, range, nugget]"""
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return psill * (1. - np.exp(-d**2./(range_*4./7.)**2.)) + nugget

def clear_timeline(self):
        """
        Clear the contents of the TimeLine Canvas

        Does not modify the actual markers dictionary and thus after
        redrawing all markers are visible again.
        """
        self._timeline.delete(tk.ALL)
        self._canvas_ticks.delete(tk.ALL)

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

def yview(self, *args):
        """Update inplace widgets position when doing vertical scroll"""
        self.after_idle(self.__updateWnds)
        ttk.Treeview.yview(self, *args)

async def iso(self, source):
        """Convert to timestamp."""
        from datetime import datetime
        unix_timestamp = int(source)
        return datetime.fromtimestamp(unix_timestamp).isoformat()

def file_empty(fp):
    """Determine if a file is empty or not."""
    # for python 2 we need to use a homemade peek()
    if six.PY2:
        contents = fp.read()
        fp.seek(0)
        return not bool(contents)

    else:
        return not fp.peek()

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

def get_python_dict(scala_map):
    """Return a dict from entries in a scala.collection.immutable.Map"""
    python_dict = {}
    keys = get_python_list(scala_map.keys().toList())
    for key in keys:
        python_dict[key] = scala_map.apply(key)
    return python_dict

def listfolderpath(p):
    """
    generator of list folder in the path.
    folders only
    """
    for entry in scandir.scandir(p):
        if entry.is_dir():
            yield entry.path

def file_empty(fp):
    """Determine if a file is empty or not."""
    # for python 2 we need to use a homemade peek()
    if six.PY2:
        contents = fp.read()
        fp.seek(0)
        return not bool(contents)

    else:
        return not fp.peek()

def render_template(env, filename, values=None):
    """
    Render a jinja template
    """
    if not values:
        values = {}
    tmpl = env.get_template(filename)
    return tmpl.render(values)

def listified_tokenizer(source):
    """Tokenizes *source* and returns the tokens as a list of lists."""
    io_obj = io.StringIO(source)
    return [list(a) for a in tokenize.generate_tokens(io_obj.readline)]

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

def tokenize(string):
    """Match and yield all the tokens of the input string."""
    for match in TOKENS_REGEX.finditer(string):
        yield Token(match.lastgroup, match.group().strip(), match.span())

def save_notebook(work_notebook, write_file):
    """Saves the Jupyter work_notebook to write_file"""
    with open(write_file, 'w') as out_nb:
        json.dump(work_notebook, out_nb, indent=2)

def conv3x3(in_channels, out_channels, stride=1):
    """
    3x3 convolution with padding.
    Original code has had bias turned off, because Batch Norm would remove the bias either way
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def _unjsonify(x, isattributes=False):
    """Convert JSON string to an ordered defaultdict."""
    if isattributes:
        obj = json.loads(x)
        return dict_class(obj)
    return json.loads(x)

def camel_to_(s):
    """
    Convert CamelCase to camel_case
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def toJson(protoObject, indent=None):
    """
    Serialises a protobuf object as json
    """
    # Using the internal method because this way we can reformat the JSON
    js = json_format.MessageToDict(protoObject, False)
    return json.dumps(js, indent=indent)

def normalize_time(timestamp):
    """Normalize time in arbitrary timezone to UTC naive object."""
    offset = timestamp.utcoffset()
    if offset is None:
        return timestamp
    return timestamp.replace(tzinfo=None) - offset

def angle_v2_rad(vec_a, vec_b):
    """Returns angle between vec_a and vec_b in range [0, PI].  This does not
    distinguish if a is left of or right of b.
    """
    # cos(x) = A * B / |A| * |B|
    return math.acos(vec_a.dot(vec_b) / (vec_a.length() * vec_b.length()))

def toListInt(value):
        """
        Convert a value to list of ints, if possible.
        """
        if TypeConverters._can_convert_to_list(value):
            value = TypeConverters.toList(value)
            if all(map(lambda v: TypeConverters._is_integer(v), value)):
                return [int(v) for v in value]
        raise TypeError("Could not convert %s to list of ints" % value)

def do_next(self, args):
        """Step over the next statement
        """
        self._do_print_from_last_cmd = True
        self._interp.step_over()
        return True

def cmd_dns_lookup_reverse(ip_address, verbose):
    """Perform a reverse lookup of a given IP address.

    Example:

    \b
    $ $ habu.dns.lookup.reverse 8.8.8.8
    {
        "hostname": "google-public-dns-a.google.com"
    }
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        print("Looking up %s..." % ip_address, file=sys.stderr)

    answer = lookup_reverse(ip_address)

    if answer:
        print(json.dumps(answer, indent=4))
    else:
        print("[X] %s is not valid IPv4/IPV6 address" % ip_address)

    return True

def stop_process(self):
        """
        Stops the child process.
        """
        self._process.terminate()
        if not self._process.waitForFinished(100):
            self._process.kill()

def yview(self, *args):
        """Update inplace widgets position when doing vertical scroll"""
        self.after_idle(self.__updateWnds)
        ttk.Treeview.yview(self, *args)

def stop(self):
        """
        Stop this server so that the calling process can exit
        """
        # unsetup_fuse()
        self.fuse_process.teardown()
        for uuid in self.processes:
            self.processes[uuid].terminate()

def yaml_to_param(obj, name):
	"""
	Return the top-level element of a document sub-tree containing the
	YAML serialization of a Python object.
	"""
	return from_pyvalue(u"yaml:%s" % name, unicode(yaml.dump(obj)))

def get_methods(*objs):
    """ Return the names of all callable attributes of an object"""
    return set(
        attr
        for obj in objs
        for attr in dir(obj)
        if not attr.startswith('_') and callable(getattr(obj, attr))
    )

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

def l2_norm(arr):
    """
    The l2 norm of an array is is defined as: sqrt(||x||), where ||x|| is the
    dot product of the vector.
    """
    arr = np.asarray(arr)
    return np.sqrt(np.dot(arr.ravel().squeeze(), arr.ravel().squeeze()))

def default_strlen(strlen=None):
    """Sets the default string length for lstring and ilwd:char, if they are
    treated as strings. Default is 50.
    """
    if strlen is not None:
        _default_types_status['default_strlen'] = strlen
        # update the typeDicts as needed
        lstring_as_obj(_default_types_status['lstring_as_obj'])
        ilwd_as_int(_default_types_status['ilwd_as_int'])
    return _default_types_status['default_strlen']

def set_global(node: Node, key: str, value: Any):
    """Adds passed value to node's globals"""
    node.node_globals[key] = value

def generate_id():
    """Generate new UUID"""
    # TODO: Use six.string_type to Py3 compat
    try:
        return unicode(uuid1()).replace(u"-", u"")
    except NameError:
        return str(uuid1()).replace(u"-", u"")

def end_index(self):
        """
        Returns the 1-based index of the last object on this page,
        relative to total objects found (hits).
        """
        return ((self.number - 1) * self.paginator.per_page +
            len(self.object_list))

def to_pascal_case(s):
    """Transform underscore separated string to pascal case

    """
    return re.sub(r'(?!^)_([a-zA-Z])', lambda m: m.group(1).upper(), s.capitalize())

def onLeftDown(self, event=None):
        """ left button down: report x,y coords, start zooming mode"""
        if event is None:
            return
        self.cursor_mode_action('leftdown', event=event)
        self.ForwardEvent(event=event.guiEvent)

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

def norm_vec(vector):
    """Normalize the length of a vector to one"""
    assert len(vector) == 3
    v = np.array(vector)
    return v/np.sqrt(np.sum(v**2))

def test(ctx, all=False, verbose=False):
    """Run the tests."""
    cmd = 'tox' if all else 'py.test'
    if verbose:
        cmd += ' -v'
    return ctx.run(cmd, pty=True).return_code

def txt_line_iterator(path):
  """Iterate through lines of file."""
  with tf.gfile.Open(path) as f:
    for line in f:
      yield line.strip()

def cover(session):
    """Run the final coverage report.
    This outputs the coverage report aggregating coverage from the unit
    test runs (not system test runs), and then erases coverage data.
    """
    session.interpreter = 'python3.6'
    session.install('coverage', 'pytest-cov')
    session.run('coverage', 'report', '--show-missing', '--fail-under=100')
    session.run('coverage', 'erase')

def find_one(line, lookup):
        """
        regexp search with one value to return.

        :param line: Line
        :param lookup: regexp
        :return: Match group or False
        """
        match = re.search(lookup, line)
        if match:
            if match.group(1):
                return match.group(1)
        return False

def teardown_test(self, context):
        """
        Tears down the Django test
        """
        context.test.tearDownClass()
        context.test._post_teardown(run=True)
        del context.test

def glog(x,l = 2):
    """
    Generalised logarithm

    :param x: number
    :param p: number added befor logarithm 

    """
    return np.log((x+np.sqrt(x**2+l**2))/2)/np.log(l)

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

def _ioctl (self, func, args):
        """Call ioctl() with given parameters."""
        import fcntl
        return fcntl.ioctl(self.sockfd.fileno(), func, args)

def dict_update_newkeys(dict_, dict2):
    """ Like dict.update, but does not overwrite items """
    for key, val in six.iteritems(dict2):
        if key not in dict_:
            dict_[key] = val

def get_tablenames(cur):
    """ Conveinience: """
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tablename_list_ = cur.fetchall()
    tablename_list = [str(tablename[0]) for tablename in tablename_list_ ]
    return tablename_list

def inheritdoc(method):
    """Set __doc__ of *method* to __doc__ of *method* in its parent class.

    Since this is used on :class:`.StringMixIn`, the "parent class" used is
    ``str``. This function can be used as a decorator.
    """
    method.__doc__ = getattr(str, method.__name__).__doc__
    return method

def split_elements(value):
    """Split a string with comma or space-separated elements into a list."""
    l = [v.strip() for v in value.split(',')]
    if len(l) == 1:
        l = value.split()
    return l

def _remove_from_index(index, obj):
    """Removes object ``obj`` from the ``index``."""
    try:
        index.value_map[indexed_value(index, obj)].remove(obj.id)
    except KeyError:
        pass

def keys(self, index=None):
        """Returns a list of keys in the database
        """
        with self._lmdb.begin() as txn:
            return [key.decode() for key, _ in txn.cursor()]

def make_bound(lower, upper, lineno):
    """ Wrapper: Creates an array bound
    """
    return symbols.BOUND.make_node(lower, upper, lineno)

def load_image(fname):
    """ read an image from file - PIL doesnt close nicely """
    with open(fname, "rb") as f:
        i = Image.open(fname)
        #i.load()
        return i

def get_dict_to_encoded_url(data):
    """
    Converts a dict to an encoded URL.
    Example: given  data = {'a': 1, 'b': 2}, it returns 'a=1&b=2'
    """
    unicode_data = dict([(k, smart_str(v)) for k, v in data.items()])
    encoded = urllib.urlencode(unicode_data)
    return encoded

def load(self, filename='classifier.dump'):
        """
        Unpickles the classifier used
        """
        ifile = open(filename, 'r+')
        self.classifier = pickle.load(ifile)
        ifile.close()

def get_body_size(params, boundary):
    """Returns the number of bytes that the multipart/form-data encoding
    of ``params`` will be."""
    size = sum(p.get_size(boundary) for p in MultipartParam.from_params(params))
    return size + len(boundary) + 6

def load_graph_from_rdf(fname):
    """ reads an RDF file into a graph """
    print("reading RDF from " + fname + "....")
    store = Graph()
    store.parse(fname, format="n3")
    print("Loaded " + str(len(store)) + " tuples")
    return store

def __init__(self, usb):
    """Constructs a FastbootCommands instance.

    Arguments:
      usb: UsbHandle instance.
    """
    self._usb = usb
    self._protocol = self.protocol_handler(usb)

def get_jsonparsed_data(url):
    """Receive the content of ``url``, parse it as JSON and return the
       object.
    """
    response = urlopen(url)
    data = response.read().decode('utf-8')
    return json.loads(data)

def _if(ctx, logical_test, value_if_true=0, value_if_false=False):
    """
    Returns one value if the condition evaluates to TRUE, and another value if it evaluates to FALSE
    """
    return value_if_true if conversions.to_boolean(logical_test, ctx) else value_if_false

def _find(string, sub_string, start_index):
    """Return index of sub_string in string.

    Raise TokenError if sub_string is not found.
    """
    result = string.find(sub_string, start_index)
    if result == -1:
        raise TokenError("expected '{0}'".format(sub_string))
    return result

def b2u(string):
    """ bytes to unicode """
    if (isinstance(string, bytes) or
        (PY2 and isinstance(string, str))):
        return string.decode('utf-8')
    return string

def glog(x,l = 2):
    """
    Generalised logarithm

    :param x: number
    :param p: number added befor logarithm 

    """
    return np.log((x+np.sqrt(x**2+l**2))/2)/np.log(l)

def human_uuid():
    """Returns a good UUID for using as a human readable string."""
    return base64.b32encode(
        hashlib.sha1(uuid.uuid4().bytes).digest()).lower().strip('=')

def ln_norm(x, mu, sigma=1.0):
    """ Natural log of scipy norm function truncated at zero """
    return np.log(stats.norm(loc=mu, scale=sigma).pdf(x))

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

def normal_log_q(self,z):
        """
        The unnormalized log posterior components for mean-field normal family (the quantity we want to approximate)
        RAO-BLACKWELLIZED!
        """             
        means, scale = self.get_means_and_scales()
        return ss.norm.logpdf(z,loc=means,scale=scale)

def gaussian_variogram_model(m, d):
    """Gaussian model, m is [psill, range, nugget]"""
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return psill * (1. - np.exp(-d**2./(range_*4./7.)**2.)) + nugget

def process_request(self, request, response):
        """Logs the basic endpoint requested"""
        self.logger.info('Requested: {0} {1} {2}'.format(request.method, request.relative_uri, request.content_type))

def val_to_bin(edges, x):
    """Convert axis coordinate to bin index."""
    ibin = np.digitize(np.array(x, ndmin=1), edges) - 1
    return ibin

def log_loss(preds, labels):
    """Logarithmic loss with non-necessarily-binary labels."""
    log_likelihood = np.sum(labels * np.log(preds)) / len(preds)
    return -log_likelihood

def dot_v3(v, w):
    """Return the dotproduct of two vectors."""

    return sum([x * y for x, y in zip(v, w)])

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

def v_normalize(v):
    """
    Normalizes the given vector.
    
    The vector given may have any number of dimensions.
    """
    vmag = v_magnitude(v)
    return [ v[i]/vmag  for i in range(len(v)) ]

def chmod_add_excute(filename):
        """
        Adds execute permission to file.
        :param filename:
        :return:
        """
        st = os.stat(filename)
        os.chmod(filename, st.st_mode | stat.S_IEXEC)

def v_normalize(v):
    """
    Normalizes the given vector.
    
    The vector given may have any number of dimensions.
    """
    vmag = v_magnitude(v)
    return [ v[i]/vmag  for i in range(len(v)) ]

def map(cls, iterable, func, *a, **kw):
    """
    Iterable-first replacement of Python's built-in `map()` function.
    """

    return cls(func(x, *a, **kw) for x in iterable)

def OnTogglePlay(self, event):
        """Toggles the video status between play and hold"""

        if self.player.get_state() == vlc.State.Playing:
            self.player.pause()
        else:
            self.player.play()

        event.Skip()

def clean_with_zeros(self,x):
        """ set nan and inf rows from x to zero"""
        x[~np.any(np.isnan(x) | np.isinf(x),axis=1)] = 0
        return x

def movingaverage(arr, window):
    """
    Calculates the moving average ("rolling mean") of an array
    of a certain window size.
    """
    m = np.ones(int(window)) / int(window)
    return scipy.ndimage.convolve1d(arr, m, axis=0, mode='reflect')

def list_string_to_dict(string):
    """Inputs ``['a', 'b', 'c']``, returns ``{'a': 0, 'b': 1, 'c': 2}``."""
    dictionary = {}
    for idx, c in enumerate(string):
        dictionary.update({c: idx})
    return dictionary

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

def make_regex(separator):
    """Utility function to create regexp for matching escaped separators
    in strings.

    """
    return re.compile(r'(?:' + re.escape(separator) + r')?((?:[^' +
                      re.escape(separator) + r'\\]|\\.)+)')

def _position():
    """Returns the current xy coordinates of the mouse cursor as a two-integer
    tuple by calling the GetCursorPos() win32 function.

    Returns:
      (x, y) tuple of the current xy coordinates of the mouse cursor.
    """

    cursor = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(cursor))
    return (cursor.x, cursor.y)

def comment (self, s, **args):
        """Write DOT comment."""
        self.write(u"// ")
        self.writeln(s=s, **args)

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

def is_cached(file_name):
	"""
	Check if a given file is available in the cache or not
	"""

	gml_file_path = join(join(expanduser('~'), OCTOGRID_DIRECTORY), file_name)

	return isfile(gml_file_path)

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

def close(self):
        """Closes this response."""
        if self._connection:
            self._connection.close()
        self._response.close()

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

def reduce_json(data):
    """Reduce a JSON object"""
    return reduce(lambda x, y: int(x) + int(y), data.values())

def convert_value(bind, value):
    """ Type casting. """
    type_name = get_type(bind)
    try:
        return typecast.cast(type_name, value)
    except typecast.ConverterError:
        return value

def chop(seq, size):
    """Chop a sequence into chunks of the given size."""
    chunk = lambda i: seq[i:i+size]
    return map(chunk,xrange(0,len(seq),size))

def write_str2file(pathname, astr):
    """writes a string to file"""
    fname = pathname
    fhandle = open(fname, 'wb')
    fhandle.write(astr)
    fhandle.close()

def get_key_by_value(dictionary, search_value):
    """
    searchs a value in a dicionary and returns the key of the first occurrence

    :param dictionary: dictionary to search in
    :param search_value: value to search for
    """
    for key, value in dictionary.iteritems():
        if value == search_value:
            return ugettext(key)

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

def MatrixInverse(a, adj):
    """
    Matrix inversion op.
    """
    return np.linalg.inv(a if not adj else _adjoint(a)),

def _write_color_ansi (fp, text, color):
    """Colorize text with given color."""
    fp.write(esc_ansicolor(color))
    fp.write(text)
    fp.write(AnsiReset)

def indentsize(line):
    """Return the indent size, in spaces, at the start of a line of text."""
    expline = string.expandtabs(line)
    return len(expline) - len(string.lstrip(expline))

def write_line(self, line, count=1):
        """writes the line and count newlines after the line"""
        self.write(line)
        self.write_newlines(count)

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

def _write_pidfile(pidfile):
    """ Write file with current process ID.
    """
    pid = str(os.getpid())
    handle = open(pidfile, 'w')
    try:
        handle.write("%s\n" % pid)
    finally:
        handle.close()

def random_color(_min=MIN_COLOR, _max=MAX_COLOR):
    """Returns a random color between min and max."""
    return color(random.randint(_min, _max))

def correspond(text):
    """Communicate with the child process without closing stdin."""
    subproc.stdin.write(text)
    subproc.stdin.flush()
    return drain()

def SegmentMin(a, ids):
    """
    Segmented min op.
    """
    func = lambda idxs: np.amin(a[idxs], axis=0)
    return seg_map(func, a, ids),

def save_dot(self, fd):
        """ Saves a representation of the case in the Graphviz DOT language.
        """
        from pylon.io import DotWriter
        DotWriter(self).write(fd)

def _prepare_proxy(self, conn):
        """
        Establish tunnel connection early, because otherwise httplib
        would improperly set Host: header to proxy's IP:port.
        """
        conn.set_tunnel(self._proxy_host, self.port, self.proxy_headers)
        conn.connect()

def cleanup_nodes(doc):
    """
    Remove text nodes containing only whitespace
    """
    for node in doc.documentElement.childNodes:
        if node.nodeType == Node.TEXT_NODE and node.nodeValue.isspace():
            doc.documentElement.removeChild(node)
    return doc

def mock_decorator(*args, **kwargs):
    """Mocked decorator, needed in the case we need to mock a decorator"""
    def _called_decorator(dec_func):
        @wraps(dec_func)
        def _decorator(*args, **kwargs):
            return dec_func()
        return _decorator
    return _called_decorator

def __get_xml_text(root):
    """ Return the text for the given root node (xml.dom.minidom). """
    txt = ""
    for e in root.childNodes:
        if (e.nodeType == e.TEXT_NODE):
            txt += e.data
    return txt

def cat_acc(y_true, y_pred):
    """Categorical accuracy
    """
    return np.mean(y_true.argmax(axis=1) == y_pred.argmax(axis=1))

def _xxrange(self, start, end, step_count):
        """Generate n values between start and end."""
        _step = (end - start) / float(step_count)
        return (start + (i * _step) for i in xrange(int(step_count)))

def forceupdate(self, *args, **kw):
        """Like a bulk :meth:`forceput`."""
        self._update(False, self._ON_DUP_OVERWRITE, *args, **kw)

def load_yaml(filepath):
    """Convenience function for loading yaml-encoded data from disk."""
    with open(filepath) as f:
        txt = f.read()
    return yaml.load(txt)

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

def chunks(iterable, chunk):
    """Yield successive n-sized chunks from an iterable."""
    for i in range(0, len(iterable), chunk):
        yield iterable[i:i + chunk]

def most_common(items):
    """
    Wanted functionality from Counters (new in Python 2.7).
    """
    counts = {}
    for i in items:
        counts.setdefault(i, 0)
        counts[i] += 1
    return max(six.iteritems(counts), key=operator.itemgetter(1))

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

def most_common(items):
    """
    Wanted functionality from Counters (new in Python 2.7).
    """
    counts = {}
    for i in items:
        counts.setdefault(i, 0)
        counts[i] += 1
    return max(six.iteritems(counts), key=operator.itemgetter(1))

def _is_iterable(item):
    """ Checks if an item is iterable (list, tuple, generator), but not string """
    return isinstance(item, collections.Iterable) and not isinstance(item, six.string_types)

def _position():
    """Returns the current xy coordinates of the mouse cursor as a two-integer
    tuple by calling the GetCursorPos() win32 function.

    Returns:
      (x, y) tuple of the current xy coordinates of the mouse cursor.
    """

    cursor = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(cursor))
    return (cursor.x, cursor.y)

def _parse(self, date_str, format='%Y-%m-%d'):
        """
        helper function for parsing FRED date string into datetime
        """
        rv = pd.to_datetime(date_str, format=format)
        if hasattr(rv, 'to_pydatetime'):
            rv = rv.to_pydatetime()
        return rv

def normalize_text(text, line_len=80, indent=""):
    """Wrap the text on the given line length."""
    return "\n".join(
        textwrap.wrap(
            text, width=line_len, initial_indent=indent, subsequent_indent=indent
        )
    )

def threadid(self):
        """
        Current thread ident. If current thread is main thread then it returns ``None``.

        :type: int or None
        """
        current = self.thread.ident
        main = get_main_thread()
        if main is None:
            return current
        else:
            return current if current != main.ident else None

def format_vars(args):
    """Format the given vars in the form: 'flag=value'"""
    variables = []
    for key, value in args.items():
        if value:
            variables += ['{0}={1}'.format(key, value)]
    return variables

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

def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

def is_iter_non_string(obj):
    """test if object is a list or tuple"""
    if isinstance(obj, list) or isinstance(obj, tuple):
        return True
    return False

def norm(x, mu, sigma=1.0):
    """ Scipy norm function """
    return stats.norm(loc=mu, scale=sigma).pdf(x)

def trans_from_matrix(matrix):
    """ Convert a vtk matrix to a numpy.ndarray """
    t = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            t[i, j] = matrix.GetElement(i, j)
    return t

def All(sequence):
  """
  :param sequence: Any sequence whose elements can be evaluated as booleans.
  :returns: true if all elements of the sequence satisfy True and x.
  """
  return bool(reduce(lambda x, y: x and y, sequence, True))

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

def inspect_cuda():
    """ Return cuda device information and nvcc/cuda setup """
    nvcc_settings = nvcc_compiler_settings()
    sysconfig.get_config_vars()
    nvcc_compiler = ccompiler.new_compiler()
    sysconfig.customize_compiler(nvcc_compiler)
    customize_compiler_for_nvcc(nvcc_compiler, nvcc_settings)

    output = inspect_cuda_version_and_devices(nvcc_compiler, nvcc_settings)

    return json.loads(output), nvcc_settings

def iteritems(data, **kwargs):
    """Iterate over dict items."""
    return iter(data.items(**kwargs)) if IS_PY3 else data.iteritems(**kwargs)

def _array2cstr(arr):
    """ Serializes a numpy array to a compressed base64 string """
    out = StringIO()
    np.save(out, arr)
    return b64encode(out.getvalue())

def default_static_path():
    """
        Return the path to the javascript bundle
    """
    fdir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(fdir, '../assets/'))

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

def random_id(size=8, chars=string.ascii_letters + string.digits):
	"""Generates a random string of given size from the given chars.

	@param size:  The size of the random string.
	@param chars: Constituent pool of characters to draw random characters from.
	@type size:   number
	@type chars:  string
	@rtype:       string
	@return:      The string of random characters.
	"""
	return ''.join(random.choice(chars) for _ in range(size))

def ln_norm(x, mu, sigma=1.0):
    """ Natural log of scipy norm function truncated at zero """
    return np.log(stats.norm(loc=mu, scale=sigma).pdf(x))

def threadid(self):
        """
        Current thread ident. If current thread is main thread then it returns ``None``.

        :type: int or None
        """
        current = self.thread.ident
        main = get_main_thread()
        if main is None:
            return current
        else:
            return current if current != main.ident else None

def ln_norm(x, mu, sigma=1.0):
    """ Natural log of scipy norm function truncated at zero """
    return np.log(stats.norm(loc=mu, scale=sigma).pdf(x))

def get_best_encoding(stream):
    """Returns the default stream encoding if not found."""
    rv = getattr(stream, 'encoding', None) or sys.getdefaultencoding()
    if is_ascii_encoding(rv):
        return 'utf-8'
    return rv

def normalize(X):
    """ equivalent to scipy.preprocessing.normalize on sparse matrices
    , but lets avoid another depedency just for a small utility function """
    X = coo_matrix(X)
    X.data = X.data / sqrt(bincount(X.row, X.data ** 2))[X.row]
    return X

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

def normalize(self):
        """ Normalize data. """

        if self.preprocessed_data.empty:
            data = self.original_data
        else:
            data = self.preprocessed_data

        data = pd.DataFrame(preprocessing.normalize(data), columns=data.columns, index=data.index)
        self.preprocessed_data = data

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

def normalize(df, style = 'mean'):
    """ Returns a normalized version of a DataFrame or Series
    Parameters:
    df - DataFrame or Series
        The data to normalize
    style - function or string, default 'mean'
        The style to use when computing the norms. Takes 'mean' or 'minmax' to
        do mean or min-max normalization respectively. User-defined functions that take
        a pandas Series as input and return a normalized pandas Series are also accepted
    """
    if style == 'mean':
        df_mean,df_std = df.mean(),df.std()
        return (df-df_mean)/df_std
    elif style == 'minmax':
        col_min,col_max = df.min(),df.max()
        return (df-col_min)/(col_max-col_min)
    else:
        return style(df)

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

def normalize(X):
    """ equivalent to scipy.preprocessing.normalize on sparse matrices
    , but lets avoid another depedency just for a small utility function """
    X = coo_matrix(X)
    X.data = X.data / sqrt(bincount(X.row, X.data ** 2))[X.row]
    return X

def column_names(self, table):
      """An iterable of column names, for a particular table or
      view."""

      table_info = self.execute(
        u'PRAGMA table_info(%s)' % quote(table))
      return (column['name'] for column in table_info)

def _normalize_abmn(abmn):
    """return a normalized version of abmn
    """
    abmn_2d = np.atleast_2d(abmn)
    abmn_normalized = np.hstack((
        np.sort(abmn_2d[:, 0:2], axis=1),
        np.sort(abmn_2d[:, 2:4], axis=1),
    ))
    return abmn_normalized

def get_iter_string_reader(stdin):
    """ return an iterator that returns a chunk of a string every time it is
    called.  notice that even though bufsize_type might be line buffered, we're
    not doing any line buffering here.  that's because our StreamBufferer
    handles all buffering.  we just need to return a reasonable-sized chunk. """
    bufsize = 1024
    iter_str = (stdin[i:i + bufsize] for i in range(0, len(stdin), bufsize))
    return get_iter_chunk_reader(iter_str)

def _count_leading_whitespace(text):
  """Returns the number of characters at the beginning of text that are whitespace."""
  idx = 0
  for idx, char in enumerate(text):
    if not char.isspace():
      return idx
  return idx + 1

def __str__(self):
        """Executes self.function to convert LazyString instance to a real
        str."""
        if not hasattr(self, '_str'):
            self._str=self.function(*self.args, **self.kwargs)
        return self._str

def get_size(objects):
    """Compute the total size of all elements in objects."""
    res = 0
    for o in objects:
        try:
            res += _getsizeof(o)
        except AttributeError:
            print("IGNORING: type=%s; o=%s" % (str(type(o)), str(o)))
    return res

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

def count_list(the_list):
    """
    Generates a count of the number of times each unique item appears in a list
    """
    count = the_list.count
    result = [(item, count(item)) for item in set(the_list)]
    result.sort()
    return result

def format_line(data, linestyle):
    """Formats a list of elements using the given line style"""
    return linestyle.begin + linestyle.sep.join(data) + linestyle.end

def get_tweepy_auth(twitter_api_key,
                    twitter_api_secret,
                    twitter_access_token,
                    twitter_access_token_secret):
    """Make a tweepy auth object"""
    auth = tweepy.OAuthHandler(twitter_api_key, twitter_api_secret)
    auth.set_access_token(twitter_access_token, twitter_access_token_secret)
    return auth

def urlencoded(body, charset='ascii', **kwargs):
    """Converts query strings into native Python objects"""
    return parse_query_string(text(body, charset=charset), False)

def aloha_to_html(html_source):
    """Converts HTML5 from Aloha to a more structured HTML5"""
    xml = aloha_to_etree(html_source)
    return etree.tostring(xml, pretty_print=True)

def exit(self):
        """Handle interactive exit.

        This method calls the ask_exit callback."""
        if self.confirm_exit:
            if self.ask_yes_no('Do you really want to exit ([y]/n)?','y'):
                self.ask_exit()
        else:
            self.ask_exit()

def _stdin_(p):
    """Takes input from user. Works for Python 2 and 3."""
    _v = sys.version[0]
    return input(p) if _v is '3' else raw_input(p)

def __del__(self):
        """Cleanup the session if it was created here"""
        if self._cleanup_session:
            self._session.loop.run_until_complete(self._session.close())

def get_just_date(self):
        """Parses just date from date-time

        :return: Just day, month and year (setting hours to 00:00:00)
        """
        return datetime.datetime(
            self.date_time.year,
            self.date_time.month,
            self.date_time.day
        )

def format_screen(strng):
    """Format a string for screen printing.

    This removes some latex-type format codes."""
    # Paragraph continue
    par_re = re.compile(r'\\$',re.MULTILINE)
    strng = par_re.sub('',strng)
    return strng

def open_with_encoding(filename, encoding, mode='r'):
    """Return opened file with a specific encoding."""
    return io.open(filename, mode=mode, encoding=encoding,
                   newline='')

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

def load_image(fname):
    """ read an image from file - PIL doesnt close nicely """
    with open(fname, "rb") as f:
        i = Image.open(fname)
        #i.load()
        return i

def random_choice(sequence):
    """ Same as :meth:`random.choice`, but also supports :class:`set` type to be passed as sequence. """
    return random.choice(tuple(sequence) if isinstance(sequence, set) else sequence)

def do_serial(self, p):
		"""Set the serial port, e.g.: /dev/tty.usbserial-A4001ib8"""
		try:
			self.serial.port = p
			self.serial.open()
			print 'Opening serial port: %s' % p
		except Exception, e:
			print 'Unable to open serial port: %s' % p

def other_ind(self):
        """last row or column of square A"""
        return np.full(self.n_min, self.size - 1, dtype=np.int)

def min_max_normalize(img):
    """Centre and normalize a given array.

    Parameters:
    ----------
    img: np.ndarray

    """

    min_img = img.min()
    max_img = img.max()

    return (img - min_img) / (max_img - min_img)

def __getitem__(self, key):
        """Returns a new PRDD of elements from that key."""
        return self.from_rdd(self._rdd.map(lambda x: x[key]))

def min_max_normalize(img):
    """Centre and normalize a given array.

    Parameters:
    ----------
    img: np.ndarray

    """

    min_img = img.min()
    max_img = img.max()

    return (img - min_img) / (max_img - min_img)

def load_file_to_base64_str(f_path):
    """Loads the content of a file into a base64 string.

    Args:
        f_path: full path to the file including the file name.

    Returns:
        A base64 string representing the content of the file in utf-8 encoding.
    """
    path = abs_path(f_path)
    with io.open(path, 'rb') as f:
        f_bytes = f.read()
        base64_str = base64.b64encode(f_bytes).decode("utf-8")
        return base64_str

def url_to_image(url, flag=cv2.IMREAD_COLOR):
    """ download the image, convert it to a NumPy array, and then read
    it into OpenCV format """
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, flag)
    return image

def visible_area(self):
        """
        Calculated like in the official client.
        Returns (top_left, bottom_right).
        """
        # looks like zeach has a nice big screen
        half_viewport = Vec(1920, 1080) / 2 / self.scale
        top_left = self.world.center - half_viewport
        bottom_right = self.world.center + half_viewport
        return top_left, bottom_right

def rotate_img(im, deg, mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_AREA):
    """ Rotates an image by deg degrees

    Arguments:
        deg (float): degree to rotate.
    """
    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c//2,r//2),deg,1)
    return cv2.warpAffine(im,M,(c,r), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS+interpolation)

def read_numpy(fd, byte_order, dtype, count):
    """Read tag data from file and return as numpy array."""
    return numpy.fromfile(fd, byte_order+dtype[-1], count)

def do_serial(self, p):
		"""Set the serial port, e.g.: /dev/tty.usbserial-A4001ib8"""
		try:
			self.serial.port = p
			self.serial.open()
			print 'Opening serial port: %s' % p
		except Exception, e:
			print 'Unable to open serial port: %s' % p

def read_credentials(fname):
    """
    read a simple text file from a private location to get
    username and password
    """
    with open(fname, 'r') as f:
        username = f.readline().strip('\n')
        password = f.readline().strip('\n')
    return username, password

def fit_gaussian(x, y, yerr, p0):
    """ Fit a Gaussian to the data """
    try:
        popt, pcov = curve_fit(gaussian, x, y, sigma=yerr, p0=p0, absolute_sigma=True)
    except RuntimeError:
        return [0],[0]
    return popt, pcov

def last(self):
        """Get the last object in file."""
        # End of file
        self.__file.seek(0, 2)

        # Get the last struct
        data = self.get(self.length - 1)

        return data

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

def load_from_file(cls, file_path: str):
        """ Read and reconstruct the data from a JSON file. """
        with open(file_path, "r") as f:
            data = json.load(f)
            item = cls.decode(data=data)
        return item

def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return F.pad(img, self.padding, self.fill, self.padding_mode)

def rAsciiLine(ifile):
    """Returns the next non-blank line in an ASCII file."""

    _line = ifile.readline().strip()
    while len(_line) == 0:
        _line = ifile.readline().strip()
    return _line

def filter_query(s):
    """
    Filters given query with the below regex
    and returns lists of quoted and unquoted strings
    """
    matches = re.findall(r'(?:"([^"]*)")|([^"]*)', s)
    result_quoted = [t[0].strip() for t in matches if t[0]]
    result_unquoted = [t[1].strip() for t in matches if t[1]]
    return result_quoted, result_unquoted

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

def list_formatter(handler, item, value):
    """Format list."""
    return u', '.join(str(v) for v in value)

def eof(fd):
    """Determine if end-of-file is reached for file fd."""
    b = fd.read(1)
    end = len(b) == 0
    if not end:
        curpos = fd.tell()
        fd.seek(curpos - 1)
    return end

def Proxy(f):
  """A helper to create a proxy method in a class."""

  def Wrapped(self, *args):
    return getattr(self, f)(*args)

  return Wrapped

def read_large_int(self, bits, signed=True):
        """Reads a n-bits long integer value."""
        return int.from_bytes(
            self.read(bits // 8), byteorder='little', signed=signed)

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

def readwav(filename):
    """Read a WAV file and returns the data and sample rate

    ::

        from spectrum.io import readwav
        readwav()

    """
    from scipy.io.wavfile import read as readwav
    samplerate, signal = readwav(filename)
    return signal, samplerate

def parse_form(self, req, name, field):
        """Pull a form value from the request."""
        return get_value(req.body_arguments, name, field)

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

def is_password_valid(password):
    """
    Check if a password is valid
    """
    pattern = re.compile(r"^.{4,75}$")
    return bool(pattern.match(password))

def standard_input():
    """Generator that yields lines from standard input."""
    with click.get_text_stream("stdin") as stdin:
        while stdin.readable():
            line = stdin.readline()
            if line:
                yield line.strip().encode("utf-8")

def is_password_valid(password):
    """
    Check if a password is valid
    """
    pattern = re.compile(r"^.{4,75}$")
    return bool(pattern.match(password))

def url_read_text(url, verbose=True):
    r"""
    Directly reads text data from url
    """
    data = url_read(url, verbose)
    text = data.decode('utf8')
    return text

def resources(self):
        """Retrieve contents of each page of PDF"""
        return [self.pdf.getPage(i) for i in range(self.pdf.getNumPages())]

def mouse_move_event(self, event):
        """
        Forward mouse cursor position events to the example
        """
        self.example.mouse_position_event(event.x(), event.y())

def cor(y_true, y_pred):
    """Compute Pearson correlation coefficient.
    """
    y_true, y_pred = _mask_nan(y_true, y_pred)
    return np.corrcoef(y_true, y_pred)[0, 1]

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

def log_magnitude_spectrum(frames):
    """Compute the log of the magnitude spectrum of frames"""
    return N.log(N.abs(N.fft.rfft(frames)).clip(1e-5, N.inf))

def get(self, key):  
        """ get a set of keys from redis """
        res = self.connection.get(key)
        print(res)
        return res

def multi_pop(d, *args):
    """ pops multiple keys off a dict like object """
    retval = {}
    for key in args:
        if key in d:
            retval[key] = d.pop(key)
    return retval

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

def multi_pop(d, *args):
    """ pops multiple keys off a dict like object """
    retval = {}
    for key in args:
        if key in d:
            retval[key] = d.pop(key)
    return retval

def from_url(url, db=None, **kwargs):
    """
    Returns an active Redis client generated from the given database URL.

    Will attempt to extract the database id from the path url fragment, if
    none is provided.
    """
    from redis.client import Redis
    return Redis.from_url(url, db, **kwargs)

def __unixify(self, s):
        """ stupid windows. converts the backslash to forwardslash for consistency """
        return os.path.normpath(s).replace(os.sep, "/")

def __setitem__(self, field, value):
        """ :see::meth:RedisMap.__setitem__ """
        return self._client.hset(self.key_prefix, field, self._dumps(value))

def parse_prefix(identifier):
    """
    Parse identifier such as a|c|le|d|li|re|or|AT4G00480.1 and return
    tuple of prefix string (separated at '|') and suffix (AGI identifier)
    """
    pf, id = (), identifier
    if "|" in identifier:
        pf, id = tuple(identifier.split('|')[:-1]), identifier.split('|')[-1]

    return pf, id

def resize_by_area(img, size):
  """image resize function used by quite a few image problems."""
  return tf.to_int64(
      tf.image.resize_images(img, [size, size], tf.image.ResizeMethod.AREA))

def ordered_yaml_dump(data, stream=None, Dumper=None, **kwds):
    """Dumps the stream from an OrderedDict.
    Taken from

    http://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-
    mappings-as-ordereddicts"""
    Dumper = Dumper or yaml.Dumper

    class OrderedDumper(Dumper):
        pass

    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items())
    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)

def lines(input):
    """Remove comments and empty lines"""
    for raw_line in input:
        line = raw_line.strip()
        if line and not line.startswith('#'):
            yield strip_comments(line)

def pretty_dict_str(d, indent=2):
    """shows JSON indented representation of d"""
    b = StringIO()
    write_pretty_dict_str(b, d, indent=indent)
    return b.getvalue()

def to_dict(self):
        """Serialize representation of the table for local caching."""
        return {'schema': self.schema, 'name': self.name, 'columns': [col.to_dict() for col in self._columns],
                'foreign_keys': self.foreign_keys.to_dict(), 'ref_keys': self.ref_keys.to_dict()}

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

def camelcase_underscore(name):
    """ Convert camelcase names to underscore """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def first(series, order_by=None):
    """
    Returns the first value of a series.

    Args:
        series (pandas.Series): column to summarize.

    Kwargs:
        order_by: a pandas.Series or list of series (can be symbolic) to order
            the input series by before summarization.
    """

    if order_by is not None:
        series = order_series_by(series, order_by)
    first_s = series.iloc[0]
    return first_s

def clean(self, text):
        """Remove all unwanted characters from text."""
        return ''.join([c for c in text if c in self.alphabet])

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

def _clean_str(self, s):
        """ Returns a lowercase string with punctuation and bad chars removed
        :param s: string to clean
        """
        return s.translate(str.maketrans('', '', punctuation)).replace('\u200b', " ").strip().lower()

def printc(cls, txt, color=colors.red):
        """Print in color."""
        print(cls.color_txt(txt, color))

def __normalize_list(self, msg):
        """Split message to list by commas and trim whitespace."""
        if isinstance(msg, list):
            msg = "".join(msg)
        return list(map(lambda x: x.strip(), msg.split(",")))

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

def remove_dups(seq):
    """remove duplicates from a sequence, preserving order"""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def csvpretty(csvfile: csvfile=sys.stdin):
    """ Pretty print a CSV file. """
    shellish.tabulate(csv.reader(csvfile))

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

def loadb(b):
    """Deserialize ``b`` (instance of ``bytes``) to a Python object."""
    assert isinstance(b, (bytes, bytearray))
    return std_json.loads(b.decode('utf-8'))

def clean_whitespace(statement):
    """
    Remove any consecutive whitespace characters from the statement text.
    """
    import re

    # Replace linebreaks and tabs with spaces
    statement.text = statement.text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

    # Remove any leeding or trailing whitespace
    statement.text = statement.text.strip()

    # Remove consecutive spaces
    statement.text = re.sub(' +', ' ', statement.text)

    return statement

def _to_base_type(self, msg):
    """Convert a Message value to a Model instance (entity)."""
    ent = _message_to_entity(msg, self._modelclass)
    ent.blob_ = self._protocol_impl.encode_message(msg)
    return ent

def pop(self, index=-1):
		"""Remove and return the item at index."""
		value = self._list.pop(index)
		del self._dict[value]
		return value

def toJson(protoObject, indent=None):
    """
    Serialises a protobuf object as json
    """
    # Using the internal method because this way we can reformat the JSON
    js = json_format.MessageToDict(protoObject, False)
    return json.dumps(js, indent=indent)

def filter_dict(d, keys):
    """
    Creates a new dict from an existing dict that only has the given keys
    """
    return {k: v for k, v in d.items() if k in keys}

def created_today(self):
        """Return True if created today."""
        if self.datetime.date() == datetime.today().date():
            return True
        return False

def _clean_str(self, s):
        """ Returns a lowercase string with punctuation and bad chars removed
        :param s: string to clean
        """
        return s.translate(str.maketrans('', '', punctuation)).replace('\u200b', " ").strip().lower()

def py3round(number):
    """Unified rounding in all python versions."""
    if abs(round(number) - number) == 0.5:
        return int(2.0 * round(number / 2.0))

    return int(round(number))

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

def strip_xml_namespace(root):
    """Strip out namespace data from an ElementTree.

    This function is recursive and will traverse all
    subnodes to the root element

    @param root: the root element

    @return: the same root element, minus namespace
    """
    try:
        root.tag = root.tag.split('}')[1]
    except IndexError:
        pass

    for element in root.getchildren():
        strip_xml_namespace(element)

def filter_(stream_spec, filter_name, *args, **kwargs):
    """Alternate name for ``filter``, so as to not collide with the
    built-in python ``filter`` operator.
    """
    return filter(stream_spec, filter_name, *args, **kwargs)

def clean(s):
  """Removes trailing whitespace on each line."""
  lines = [l.rstrip() for l in s.split('\n')]
  return '\n'.join(lines)

def first_sunday(self, year, month):
        """Get the first sunday of a month."""
        date = datetime(year, month, 1, 0)
        days_until_sunday = 6 - date.weekday()

        return date + timedelta(days=days_until_sunday)

def nonull_dict(self):
        """Like dict, but does not hold any null values.

        :return:

        """
        return {k: v for k, v in six.iteritems(self.dict) if v and k != '_codes'}

def bin_to_int(string):
    """Convert a one element byte string to signed int for python 2 support."""
    if isinstance(string, str):
        return struct.unpack("b", string)[0]
    else:
        return struct.unpack("b", bytes([string]))[0]

def _removeTags(tags, objects):
    """ Removes tags from objects """
    for t in tags:
        for o in objects:
            o.tags.remove(t)

    return True

def time_func(func, name, *args, **kwargs):
    """ call a func with args and kwargs, print name of func and how
    long it took. """
    tic = time.time()
    out = func(*args, **kwargs)
    toc = time.time()
    print('%s took %0.2f seconds' % (name, toc - tic))
    return out

def get_unique_indices(df, axis=1):
    """

    :param df:
    :param axis:
    :return:
    """
    return dict(zip(df.columns.names, dif.columns.levels))

def register_extension_class(ext, base, *args, **kwargs):
    """Instantiate the given extension class and register as a public attribute of the given base.

    README: The expected protocol here is to instantiate the given extension and pass the base
    object as the first positional argument, then unpack args and kwargs as additional arguments to
    the extension's constructor.
    """
    ext_instance = ext.plugin(base, *args, **kwargs)
    setattr(base, ext.name.lstrip('_'), ext_instance)

def lowstrip(term):
    """Convert to lowercase and strip spaces"""
    term = re.sub('\s+', ' ', term)
    term = term.lower()
    return term

def validate_string(option, value):
    """Validates that 'value' is an instance of `basestring` for Python 2
    or `str` for Python 3.
    """
    if isinstance(value, string_type):
        return value
    raise TypeError("Wrong type for %s, value must be "
                    "an instance of %s" % (option, string_type.__name__))

def clear_global(self):
        """Clear only any cached global data.

        """
        vname = self.varname
        logger.debug(f'global clearning {vname}')
        if vname in globals():
            logger.debug('removing global instance var: {}'.format(vname))
            del globals()[vname]

def _stdin_(p):
    """Takes input from user. Works for Python 2 and 3."""
    _v = sys.version[0]
    return input(p) if _v is '3' else raw_input(p)

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

def draw_image(self, ax, image):
        """Process a matplotlib image object and call renderer.draw_image"""
        self.renderer.draw_image(imdata=utils.image_to_base64(image),
                                 extent=image.get_extent(),
                                 coordinates="data",
                                 style={"alpha": image.get_alpha(),
                                        "zorder": image.get_zorder()},
                                 mplobj=image)

def localize(dt):
    """Localize a datetime object to local time."""
    if dt.tzinfo is UTC:
        return (dt + LOCAL_UTC_OFFSET).replace(tzinfo=None)
    # No TZ info so not going to assume anything, return as-is.
    return dt

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

def pop_all(self):
        """
        NON-BLOCKING POP ALL IN QUEUE, IF ANY
        """
        with self.lock:
            output = list(self.queue)
            self.queue.clear()

        return output

def join(mapping, bind, values):
    """ Merge all the strings. Put space between them. """
    return [' '.join([six.text_type(v) for v in values if v is not None])]

def invalidate_cache(cpu, address, size):
        """ remove decoded instruction from instruction cache """
        cache = cpu.instruction_cache
        for offset in range(size):
            if address + offset in cache:
                del cache[address + offset]

def _replace_docstring_header(paragraph):
    """Process NumPy-like function docstrings."""

    # Replace Markdown headers in docstrings with light headers in bold.
    paragraph = re.sub(_docstring_header_pattern,
                       r'*\1*',
                       paragraph,
                       )

    paragraph = re.sub(_docstring_parameters_pattern,
                       r'\n* `\1` (\2)\n',
                       paragraph,
                       )

    return paragraph

def _to_lower_alpha_only(s):
    """Return a lowercased string with non alphabetic chars removed.

    White spaces are not to be removed."""
    s = re.sub(r'\n', ' ',  s.lower())
    return re.sub(r'[^a-z\s]', '', s)

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

async def _send_plain_text(self, request: Request, stack: Stack):
        """
        Sends plain text using `_send_text()`.
        """

        await self._send_text(request, stack, None)

def underscore(text):
    """Converts text that may be camelcased into an underscored format"""
    return UNDERSCORE[1].sub(r'\1_\2', UNDERSCORE[0].sub(r'\1_\2', text)).lower()

def from_traceback(cls, tb):
        """ Construct a Bytecode from the given traceback """
        while tb.tb_next:
            tb = tb.tb_next
        return cls(tb.tb_frame.f_code, current_offset=tb.tb_lasti)

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

def compute(args):
    x, y, params = args
    """Callable function for the multiprocessing pool."""
    return x, y, mandelbrot(x, y, params)

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

def list(self):
        """position in 3d space"""
        return [self._pos3d.x, self._pos3d.y, self._pos3d.z]

def _scale_shape(dshape, scale = (1,1,1)):
    """returns the shape after scaling (should be the same as ndimage.zoom"""
    nshape = np.round(np.array(dshape) * np.array(scale))
    return tuple(nshape.astype(np.int))

def get_value(key, obj, default=missing):
    """Helper for pulling a keyed value off various types of objects"""
    if isinstance(key, int):
        return _get_value_for_key(key, obj, default)
    return _get_value_for_keys(key.split('.'), obj, default)

def index(self, value):
		"""
		Return the smallest index of the row(s) with this column
		equal to value.
		"""
		for i in xrange(len(self.parentNode)):
			if getattr(self.parentNode[i], self.Name) == value:
				return i
		raise ValueError(value)

def weekly(date=datetime.date.today()):
    """
    Weeks start are fixes at Monday for now.
    """
    return date - datetime.timedelta(days=date.weekday())

def get_package_info(package):
    """Gets the PyPI information for a given package."""
    url = 'https://pypi.python.org/pypi/{}/json'.format(package)
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

def setdefault(obj, field, default):
    """Set an object's field to default if it doesn't have a value"""
    setattr(obj, field, getattr(obj, field, default))

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

def _add(self, codeobj):
        """Add a child (variable) to this object."""
        assert isinstance(codeobj, CodeVariable)
        self.variables.append(codeobj)

def keys(self):
        """Return a list of all keys in the dictionary.

        Returns:
            list of str: [key1,key2,...,keyN]
        """
        all_keys = [k.decode('utf-8') for k,v in self.rdb.hgetall(self.session_hash).items()]
        return all_keys

def mock_add_spec(self, spec, spec_set=False):
        """Add a spec to a mock. `spec` can either be an object or a
        list of strings. Only attributes on the `spec` can be fetched as
        attributes from the mock.

        If `spec_set` is True then only attributes on the spec can be set."""
        self._mock_add_spec(spec, spec_set)
        self._mock_set_magics()

def make_key(observer):
        """Construct a unique, hashable, immutable key for an observer."""

        if hasattr(observer, "__self__"):
            inst = observer.__self__
            method_name = observer.__name__
            key = (id(inst), method_name)
        else:
            key = id(observer)
        return key

def update_dict(obj, dict, attributes):
    """Update dict with fields from obj.attributes.

    :param obj: the object updated into dict
    :param dict: the result dictionary
    :param attributes: a list of attributes belonging to obj
    """
    for attribute in attributes:
        if hasattr(obj, attribute) and getattr(obj, attribute) is not None:
            dict[attribute] = getattr(obj, attribute)

def _count_leading_whitespace(text):
  """Returns the number of characters at the beginning of text that are whitespace."""
  idx = 0
  for idx, char in enumerate(text):
    if not char.isspace():
      return idx
  return idx + 1

def get_attr(self, method_name):
        """Get attribute from the target object"""
        return self.attrs.get(method_name) or self.get_callable_attr(method_name)

def selectnotnone(table, field, complement=False):
    """Select rows where the given field is not `None`."""

    return select(table, field, lambda v: v is not None,
                  complement=complement)

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

def get_attribute_name_id(attr):
    """
    Return the attribute name identifier
    """
    return attr.value.id if isinstance(attr.value, ast.Name) else None

def set_property(self, key, value):
        """
        Update only one property in the dict
        """
        self.properties[key] = value
        self.sync_properties()

def _shape(self):
        """Return the tensor shape of the matrix operator"""
        return tuple(reversed(self.output_dims())) + tuple(
            reversed(self.input_dims()))

def set_json_item(key, value):
    """ manipulate json data on the fly
    """
    data = get_json()
    data[key] = value

    request = get_request()
    request["BODY"] = json.dumps(data)

def cfloat64_array_to_numpy(cptr, length):
    """Convert a ctypes double pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_double)):
        return np.fromiter(cptr, dtype=np.float64, count=length)
    else:
        raise RuntimeError('Expected double pointer')

def sub(name, func,**kwarg):
    """ Add subparser

    """
    sp = subparsers.add_parser(name, **kwarg)
    sp.set_defaults(func=func)
    sp.arg = sp.add_argument
    return sp

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

def roc_auc(y_true, y_score):
    """
    Returns are under the ROC curve
    """
    notnull = ~np.isnan(y_true)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true[notnull], y_score[notnull])
    return sklearn.metrics.auc(fpr, tpr)

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

def vec_angle(a, b):
    """
    Calculate angle between two vectors
    """
    cosang = np.dot(a, b)
    sinang = fast_norm(np.cross(a, b))
    return np.arctan2(sinang, cosang)

def similarity_transformation(rot, mat):
    """ R x M x R^-1 """
    return np.dot(rot, np.dot(mat, np.linalg.inv(rot)))

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

def round_figures(x, n):
    """Returns x rounded to n significant figures."""
    return round(x, int(n - math.ceil(math.log10(abs(x)))))

def fetch_token(self, **kwargs):
        """Exchange a code (and 'state' token) for a bearer token"""
        return super(AsanaOAuth2Session, self).fetch_token(self.token_url, client_secret=self.client_secret, **kwargs)

def lint(args):
    """Run lint checks using flake8."""
    application = get_current_application()
    if not args:
        args = [application.name, 'tests']
    args = ['flake8'] + list(args)
    run.main(args, standalone_mode=False)

def contains_extractor(document):
    """A basic document feature extractor that returns a dict of words that the
    document contains."""
    tokens = _get_document_tokens(document)
    features = dict((u'contains({0})'.format(w), True) for w in tokens)
    return features

def _handle_shell(self,cfg_file,*args,**options):
        """Command 'supervisord shell' runs the interactive command shell."""
        args = ("--interactive",) + args
        return supervisorctl.main(("-c",cfg_file) + args)

def interpolate(table, field, fmt, **kwargs):
    """
    Convenience function to interpolate all values in the given `field` using
    the `fmt` string.

    The ``where`` keyword argument can be given with a callable or expression
    which is evaluated on each row and which should return True if the
    conversion should be applied on that row, else False.

    """

    conv = lambda v: fmt % v
    return convert(table, field, conv, **kwargs)

def execute_in_background(self):
        """Executes a (shell) command in the background

        :return: the process' pid
        """
        # http://stackoverflow.com/questions/1605520
        args = shlex.split(self.cmd)
        p = Popen(args)
        return p.pid

def apply(f, obj, *args, **kwargs):
    """Apply a function in parallel to each element of the input"""
    return vectorize(f)(obj, *args, **kwargs)

def bash(filename):
    """Runs a bash script in the local directory"""
    sys.stdout.flush()
    subprocess.call("bash {}".format(filename), shell=True)

def to_lisp(o, keywordize_keys: bool = True):
    """Recursively convert Python collections into Lisp collections."""
    if not isinstance(o, (dict, frozenset, list, set, tuple)):
        return o
    else:  # pragma: no cover
        return _to_lisp_backup(o, keywordize_keys=keywordize_keys)

def movingaverage(arr, window):
    """
    Calculates the moving average ("rolling mean") of an array
    of a certain window size.
    """
    m = np.ones(int(window)) / int(window)
    return scipy.ndimage.convolve1d(arr, m, axis=0, mode='reflect')

def rotateImage(img, angle):
    """

    querries scipy.ndimage.rotate routine
    :param img: image to be rotated
    :param angle: angle to be rotated (radian)
    :return: rotated image
    """
    imgR = scipy.ndimage.rotate(img, angle, reshape=False)
    return imgR

def version_jar(self):
		"""
		Special case of version() when the executable is a JAR file.
		"""
		cmd = config.get_command('java')
		cmd.append('-jar')
		cmd += self.cmd
		self.version(cmd=cmd, path=self.cmd[0])

def make_table_map(table, headers):
    """Create a function to map from rows with the structure of the headers to the structure of the table."""

    header_parts = {}
    for i, h in enumerate(headers):
        header_parts[h] = 'row[{}]'.format(i)

    body_code = 'lambda row: [{}]'.format(','.join(header_parts.get(c.name, 'None') for c in table.columns))
    header_code = 'lambda row: [{}]'.format(
        ','.join(header_parts.get(c.name, "'{}'".format(c.name)) for c in table.columns))

    return eval(header_code), eval(body_code)

def test(*args):
    """
    Run unit tests.
    """
    subprocess.call(["py.test-2.7"] + list(args))
    subprocess.call(["py.test-3.4"] + list(args))

def print_statements(self):
        """Print all INDRA Statements collected by the processors."""
        for i, stmt in enumerate(self.statements):
            print("%s: %s" % (i, stmt))

def test(*args):
    """
    Run unit tests.
    """
    subprocess.call(["py.test-2.7"] + list(args))
    subprocess.call(["py.test-3.4"] + list(args))

def email_type(arg):
	"""An argparse type representing an email address."""
	if not is_valid_email_address(arg):
		raise argparse.ArgumentTypeError("{0} is not a valid email address".format(repr(arg)))
	return arg

def download_file(bucket_name, path, target, sagemaker_session):
    """Download a Single File from S3 into a local path

    Args:
        bucket_name (str): S3 bucket name
        path (str): file path within the bucket
        target (str): destination directory for the downloaded file.
        sagemaker_session (:class:`sagemaker.session.Session`): a sagemaker session to interact with S3.
    """
    path = path.lstrip('/')
    boto_session = sagemaker_session.boto_session

    s3 = boto_session.resource('s3')
    bucket = s3.Bucket(bucket_name)
    bucket.download_file(path, target)

def set_subparsers_args(self, *args, **kwargs):
        """
        Sets args and kwargs that are passed when creating a subparsers group
        in an argparse.ArgumentParser i.e. when calling
        argparser.ArgumentParser.add_subparsers
        """
        self.subparsers_args = args
        self.subparsers_kwargs = kwargs

def save_keras_definition(keras_model, path):
    """
    Save a Keras model definition to JSON with given path
    """
    model_json = keras_model.to_json()
    with open(path, "w") as json_file:
        json_file.write(model_json)

def add_arguments(parser):
    """
    adds arguments for the swap urls command
    """
    parser.add_argument('-o', '--old-environment', help='Old environment name', required=True)
    parser.add_argument('-n', '--new-environment', help='New environment name', required=True)

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

def main(args=sys.argv):
    """
    main entry point for the jardiff CLI
    """

    parser = create_optparser(args[0])
    return cli(parser.parse_args(args[1:]))

def save(self, fname):
        """ Saves the dictionary in json format
        :param fname: file to save to
        """
        with open(fname, 'wb') as f:
            json.dump(self, f)

def set_subparsers_args(self, *args, **kwargs):
        """
        Sets args and kwargs that are passed when creating a subparsers group
        in an argparse.ArgumentParser i.e. when calling
        argparser.ArgumentParser.add_subparsers
        """
        self.subparsers_args = args
        self.subparsers_kwargs = kwargs

def save(self, fname):
        """ Saves the dictionary in json format
        :param fname: file to save to
        """
        with open(fname, 'wb') as f:
            json.dump(self, f)

def ensure_iterable(inst):
    """
    Wraps scalars or string types as a list, or returns the iterable instance.
    """
    if isinstance(inst, string_types):
        return [inst]
    elif not isinstance(inst, collections.Iterable):
        return [inst]
    else:
        return inst

def sav_to_pandas_rpy2(input_file):
    """
    SPSS .sav files to Pandas DataFrame through Rpy2

    :param input_file: string

    :return:
    """
    import pandas.rpy.common as com

    w = com.robj.r('foreign::read.spss("%s", to.data.frame=TRUE)' % input_file)
    return com.convert_robj(w)

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

def write(url, content, **args):
    """Put an object into a ftps URL."""
    with FTPSResource(url, **args) as resource:
        resource.write(content)

def dump_nparray(self, obj, class_name=numpy_ndarray_class_name):
        """
        ``numpy.ndarray`` dumper.
        """
        return {"$" + class_name: self._json_convert(obj.tolist())}

def newest_file(file_iterable):
  """
  Returns the name of the newest file given an iterable of file names.

  """
  return max(file_iterable, key=lambda fname: os.path.getmtime(fname))

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

def unfolding(tens, i):
    """Compute the i-th unfolding of a tensor."""
    return reshape(tens.full(), (np.prod(tens.n[0:(i+1)]), -1))

def expect_all(a, b):
    """\
    Asserts that two iterables contain the same values.
    """
    assert all(_a == _b for _a, _b in zip_longest(a, b))

def cli_command_quit(self, msg):
        """\
        kills the child and exits
        """
        if self.state == State.RUNNING and self.sprocess and self.sprocess.proc:
            self.sprocess.proc.kill()
        else:
            sys.exit(0)

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

def assert_is_not(expected, actual, message=None, extra=None):
    """Raises an AssertionError if expected is actual."""
    assert expected is not actual, _assert_fail_message(
        message, expected, actual, "is", extra
    )

async def repeat(ctx, times: int, content='repeating...'):
    """Repeats a message multiple times."""
    for i in range(times):
        await ctx.send(content)

def expect_all(a, b):
    """\
    Asserts that two iterables contain the same values.
    """
    assert all(_a == _b for _a, _b in zip_longest(a, b))

def _digits(minval, maxval):
    """Digits needed to comforatbly display values in [minval, maxval]"""
    if minval == maxval:
        return 3
    else:
        return min(10, max(2, int(1 + abs(np.log10(maxval - minval)))))

async def wait_and_quit(loop):
	"""Wait until all task are executed."""
	from pylp.lib.tasks import running
	if running:
		await asyncio.wait(map(lambda runner: runner.future, running))

def generate_split_tsv_lines(fn, header):
    """Returns dicts with header-keys and psm statistic values"""
    for line in generate_tsv_psms_line(fn):
        yield {x: y for (x, y) in zip(header, line.strip().split('\t'))}

def asynchronous(function, event):
    """
    Runs the function asynchronously taking care of exceptions.
    """
    thread = Thread(target=synchronous, args=(function, event))
    thread.daemon = True
    thread.start()

def paste(xsel=False):
    """Returns system clipboard contents."""
    selection = "primary" if xsel else "clipboard"
    try:
        return subprocess.Popen(["xclip", "-selection", selection, "-o"], stdout=subprocess.PIPE).communicate()[0].decode("utf-8")
    except OSError as why:
        raise XclipNotFound

def StringIO(*args, **kwargs):
    """StringIO constructor shim for the async wrapper."""
    raw = sync_io.StringIO(*args, **kwargs)
    return AsyncStringIOWrapper(raw)

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

def add(self, value):
        """Add the element *value* to the set."""
        if value not in self._set:
            self._set.add(value)
            self._list.add(value)

def clear_globals_reload_modules(self):
        """Clears globals and reloads modules"""

        self.code_array.clear_globals()
        self.code_array.reload_modules()

        # Clear result cache
        self.code_array.result_cache.clear()

def filter_symlog(y, base=10.0):
    """Symmetrical logarithmic scale.

    Optional arguments:

    *base*:
        The base of the logarithm.
    """
    log_base = np.log(base)
    sign = np.sign(y)
    logs = np.log(np.abs(y) / log_base)
    return sign * logs

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

def __init__(self, interval, key):
    """Constructor. See class docstring for parameter details."""
    self.interval = interval
    self.key = key

def _openpyxl_read_xl(xl_path: str):
    """ Use openpyxl to read an Excel file. """
    try:
        wb = load_workbook(filename=xl_path, read_only=True)
    except:
        raise
    else:
        return wb

def user_return(self, frame, return_value):
        """This function is called when a return trap is set here."""
        pdb.Pdb.user_return(self, frame, return_value)

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

def setwinsize(self, rows, cols):
        """Set the terminal window size of the child tty.
        """
        self._winsize = (rows, cols)
        self.pty.set_size(cols, rows)

def delete(gandi, resource):
    """Delete DNSSEC key.
    """

    result = gandi.dnssec.delete(resource)
    gandi.echo('Delete successful.')

    return result

def _parse_config(config_file_path):
    """ Parse Config File from yaml file. """
    config_file = open(config_file_path, 'r')
    config = yaml.load(config_file)
    config_file.close()
    return config

def create_aws_lambda(ctx, bucket, region_name, aws_access_key_id, aws_secret_access_key):
    """Creates an AWS Chalice project for deployment to AWS Lambda."""
    from canari.commands.create_aws_lambda import create_aws_lambda
    create_aws_lambda(ctx.project, bucket, region_name, aws_access_key_id, aws_secret_access_key)

def confusion_matrix(self):
        """Confusion matrix plot
        """
        return plot.confusion_matrix(self.y_true, self.y_pred,
                                     self.target_names, ax=_gen_ax())

def s2b(s):
    """
    String to binary.
    """
    ret = []
    for c in s:
        ret.append(bin(ord(c))[2:].zfill(8))
    return "".join(ret)

def show_approx(self, numfmt='%.3g'):
        """Show the probabilities rounded and sorted by key, for the
        sake of portable doctests."""
        return ', '.join([('%s: ' + numfmt) % (v, p)
                          for (v, p) in sorted(self.prob.items())])

def _change_height(self, ax, new_value):
        """Make bars in horizontal bar chart thinner"""
        for patch in ax.patches:
            current_height = patch.get_height()
            diff = current_height - new_value

            # we change the bar height
            patch.set_height(new_value)

            # we recenter the bar
            patch.set_y(patch.get_y() + diff * .5)

def _transform_triple_numpy(x):
    """Transform triple index into a 1-D numpy array."""
    return np.array([x.head, x.relation, x.tail], dtype=np.int64)

def text(el, strip=True):
    """
    Return the text of a ``BeautifulSoup`` element
    """
    if not el:
        return ""

    text = el.text
    if strip:
        text = text.strip()
    return text

def underscore(text):
    """Converts text that may be camelcased into an underscored format"""
    return UNDERSCORE[1].sub(r'\1_\2', UNDERSCORE[0].sub(r'\1_\2', text)).lower()

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

def start():
    """Starts the web server."""
    global app
    bottle.run(app, host=conf.WebHost, port=conf.WebPort,
               debug=conf.WebAutoReload, reloader=conf.WebAutoReload,
               quiet=conf.WebQuiet)

def lines(input):
    """Remove comments and empty lines"""
    for raw_line in input:
        line = raw_line.strip()
        if line and not line.startswith('#'):
            yield strip_comments(line)

def get_iter_string_reader(stdin):
    """ return an iterator that returns a chunk of a string every time it is
    called.  notice that even though bufsize_type might be line buffered, we're
    not doing any line buffering here.  that's because our StreamBufferer
    handles all buffering.  we just need to return a reasonable-sized chunk. """
    bufsize = 1024
    iter_str = (stdin[i:i + bufsize] for i in range(0, len(stdin), bufsize))
    return get_iter_chunk_reader(iter_str)

def is_full_slice(obj, l):
    """
    We have a full length slice.
    """
    return (isinstance(obj, slice) and obj.start == 0 and obj.stop == l and
            obj.step is None)

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

def ibatch(iterable, size):
    """Yield a series of batches from iterable, each size elements long."""
    source = iter(iterable)
    while True:
        batch = itertools.islice(source, size)
        yield itertools.chain([next(batch)], batch)

def filtered_image(self, im):
        """Returns a filtered image after applying the Fourier-space filters"""
        q = np.fft.fftn(im)
        for k,v in self.filters:
            q[k] -= v
        return np.real(np.fft.ifftn(q))

def sort_by_name(self):
        """Sort list elements by name."""
        super(JSSObjectList, self).sort(key=lambda k: k.name)

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

def sort_data(data, cols):
    """Sort `data` rows and order columns"""
    return data.sort_values(cols)[cols + ['value']].reset_index(drop=True)

def get_cache(self, decorated_function, *args, **kwargs):
		""" :meth:`WCacheStorage.get_cache` method implementation
		"""
		has_value = self.has(decorated_function, *args, **kwargs)
		cached_value = None
		if has_value is True:
			cached_value = self.get_result(decorated_function, *args, **kwargs)
		return WCacheStorage.CacheEntry(has_value=has_value, cached_value=cached_value)

def unsort_vector(data, indices_of_increasing):
    """Upermutate 1-D data that is sorted by indices_of_increasing."""
    return numpy.array([data[indices_of_increasing.index(i)] for i in range(len(data))])

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

def loadmat(filename):
    """This function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = sploadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

async def _thread_coro(self, *args):
        """ Coroutine called by MapAsync. It's wrapping the call of
        run_in_executor to run the synchronous function as thread """
        return await self._loop.run_in_executor(
            self._executor, self._function, *args)

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

def run_tests(self):
		"""
		Invoke pytest, replacing argv. Return result code.
		"""
		with _save_argv(_sys.argv[:1] + self.addopts):
			result_code = __import__('pytest').main()
			if result_code:
				raise SystemExit(result_code)

def swap_priority(self, key1, key2):
        """
        Fast way to swap the priority level of two items in the pqdict. Raises
        ``KeyError`` if either key does not exist.

        """
        heap = self._heap
        position = self._position
        if key1 not in self or key2 not in self:
            raise KeyError
        pos1, pos2 = position[key1], position[key2]
        heap[pos1].key, heap[pos2].key = key2, key1
        position[key1], position[key2] = pos2, pos1

def recursively_get_files_from_directory(directory):
    """
    Return all filenames under recursively found in a directory
    """
    return [
        os.path.join(root, filename)
        for root, directories, filenames in os.walk(directory)
        for filename in filenames
    ]

def _set_module_names_for_sphinx(modules: List, new_name: str):
    """ Trick sphinx into displaying the desired module in these objects' documentation. """
    for obj in modules:
        obj.__module__ = new_name

def _depr(fn, usage, stacklevel=3):
    """Internal convenience function for deprecation warnings"""
    warn('{0} is deprecated. Use {1} instead'.format(fn, usage),
         stacklevel=stacklevel, category=DeprecationWarning)

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

def camelcase_to_slash(name):
    """ Converts CamelCase to camel/case

    code ripped from http://stackoverflow.com/questions/1175208/does-the-python-standard-library-have-function-to-convert-camelcase-to-camel-cas
    """

    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1/\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1/\2', s1).lower()

def from_pydatetime(cls, pydatetime):
        """
        Creates sql datetime2 object from Python datetime object
        ignoring timezone
        @param pydatetime: Python datetime object
        @return: sql datetime2 object
        """
        return cls(date=Date.from_pydate(pydatetime.date),
                   time=Time.from_pytime(pydatetime.time))

def is_identifier(string):
    """Check if string could be a valid python identifier

    :param string: string to be tested
    :returns: True if string can be a python identifier, False otherwise
    :rtype: bool
    """
    matched = PYTHON_IDENTIFIER_RE.match(string)
    return bool(matched) and not keyword.iskeyword(string)

def from_pydatetime(cls, pydatetime):
        """
        Creates sql datetime2 object from Python datetime object
        ignoring timezone
        @param pydatetime: Python datetime object
        @return: sql datetime2 object
        """
        return cls(date=Date.from_pydate(pydatetime.date),
                   time=Time.from_pytime(pydatetime.time))

def set_scrollregion(self, event=None):
        """ Set the scroll region on the canvas"""
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

def is_alive(self):
        """
        Will test whether the ACS service is up and alive.
        """
        response = self.get_monitoring_heartbeat()
        if response.status_code == 200 and response.content == 'alive':
            return True

        return False

def to_capitalized_camel_case(snake_case_string):
    """
    Convert a string from snake case to camel case with the first letter capitalized. For example, "some_var"
    would become "SomeVar".

    :param snake_case_string: Snake-cased string to convert to camel case.
    :returns: Camel-cased version of snake_case_string.
    """
    parts = snake_case_string.split('_')
    return ''.join([i.title() for i in parts])

def _subclassed(base, *classes):
        """Check if all classes are subclassed from base.
        """
        return all(map(lambda obj: isinstance(obj, base), classes))

def _opt_call_from_base_type(self, value):
    """Call _from_base_type() if necessary.

    If the value is a _BaseValue instance, unwrap it and call all
    _from_base_type() methods.  Otherwise, return the value
    unchanged.
    """
    if isinstance(value, _BaseValue):
      value = self._call_from_base_type(value.b_val)
    return value

def _port_not_in_use():
    """Use the port 0 trick to find a port not in use."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = 0
    s.bind(('', port))
    _, port = s.getsockname()
    return port

def list_to_csv(value):
    """
    Converts list to string with comma separated values. For string is no-op.
    """
    if isinstance(value, (list, tuple, set)):
        value = ",".join(value)
    return value

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

def append_pdf(input_pdf: bytes, output_writer: PdfFileWriter):
    """
    Appends a PDF to a pyPDF writer. Legacy interface.
    """
    append_memory_pdf_to_writer(input_pdf=input_pdf,
                                writer=output_writer)

def load(self, name):
        """Loads and returns foreign library."""
        name = ctypes.util.find_library(name)
        return ctypes.cdll.LoadLibrary(name)

def weighted_std(values, weights):
    """ Calculate standard deviation weighted by errors """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return np.sqrt(variance)

def _maybe_to_categorical(array):
    """
    Coerce to a categorical if a series is given.

    Internal use ONLY.
    """
    if isinstance(array, (ABCSeries, ABCCategoricalIndex)):
        return array._values
    elif isinstance(array, np.ndarray):
        return Categorical(array)
    return array

def circstd(dts, axis=2):
    """Circular standard deviation"""
    R = np.abs(np.exp(1.0j * dts).mean(axis=axis))
    return np.sqrt(-2.0 * np.log(R))

def onchange(self, value):
        """Called when a new DropDownItem gets selected.
        """
        log.debug('combo box. selected %s' % value)
        self.select_by_value(value)
        return (value, )

def save_config_value(request, response, key, value):
    """Sets value of key `key` to `value` in both session and cookies."""
    request.session[key] = value
    response.set_cookie(key, value, expires=one_year_from_now())
    return response

def set_label ( self, object, label ):
        """ Sets the label for a specified object.
        """
        label_name = self.label
        if label_name[:1] != '=':
            xsetattr( object, label_name, label )

def is_timestamp(instance):
    """Validates data is a timestamp"""
    if not isinstance(instance, (int, str)):
        return True
    return datetime.fromtimestamp(int(instance))

def add_exec_permission_to(target_file):
    """Add executable permissions to the file

    :param target_file: the target file whose permission to be changed
    """
    mode = os.stat(target_file).st_mode
    os.chmod(target_file, mode | stat.S_IXUSR)

def next (self):    # File-like object.

        """This is to support iterators over a file-like object.
        """

        result = self.readline()
        if result == self._empty_buffer:
            raise StopIteration
        return result

def _to_numeric(val):
    """
    Helper function for conversion of various data types into numeric representation.
    """
    if isinstance(val, (int, float, datetime.datetime, datetime.timedelta)):
        return val
    return float(val)

def fmt_duration(secs):
    """Format a duration in seconds."""
    return ' '.join(fmt.human_duration(secs, 0, precision=2, short=True).strip().split())

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

def AmericanDateToEpoch(self, date_str):
    """Take a US format date and return epoch."""
    try:
      epoch = time.strptime(date_str, "%m/%d/%Y")
      return int(calendar.timegm(epoch)) * 1000000
    except ValueError:
      return 0

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

def _parse(self, date_str, format='%Y-%m-%d'):
        """
        helper function for parsing FRED date string into datetime
        """
        rv = pd.to_datetime(date_str, format=format)
        if hasattr(rv, 'to_pydatetime'):
            rv = rv.to_pydatetime()
        return rv

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

def _removeTags(tags, objects):
    """ Removes tags from objects """
    for t in tags:
        for o in objects:
            o.tags.remove(t)

    return True

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

def straight_line_show(title, length=100, linestyle="=", pad=0):
        """Print a formatted straight line.
        """
        print(StrTemplate.straight_line(
            title=title, length=length, linestyle=linestyle, pad=pad))

def cudaMalloc(count, ctype=None):
    """
    Allocate device memory.

    Allocate memory on the device associated with the current active
    context.

    Parameters
    ----------
    count : int
        Number of bytes of memory to allocate
    ctype : _ctypes.SimpleType, optional
        ctypes type to cast returned pointer.

    Returns
    -------
    ptr : ctypes pointer
        Pointer to allocated device memory.

    """

    ptr = ctypes.c_void_p()
    status = _libcudart.cudaMalloc(ctypes.byref(ptr), count)
    cudaCheckStatus(status)
    if ctype != None:
        ptr = ctypes.cast(ptr, ctypes.POINTER(ctype))
    return ptr

def check_output(args):
    """Runs command and returns the output as string."""
    log.debug('run: %s', args)
    out = subprocess.check_output(args=args).decode('utf-8')
    log.debug('out: %r', out)
    return out

def check(self, var):
        """Check whether the provided value is a valid enum constant."""
        if not isinstance(var, _str_type): return False
        return _enum_mangle(var) in self._consts

def convert_time_string(date_str):
    """ Change a date string from the format 2018-08-15T23:55:17 into a datetime object """
    dt, _, _ = date_str.partition(".")
    dt = datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S")
    return dt

def check(self, var):
        """Check whether the provided value is a valid enum constant."""
        if not isinstance(var, _str_type): return False
        return _enum_mangle(var) in self._consts

def Sum(a, axis, keep_dims):
    """
    Sum reduction op.
    """
    return np.sum(a, axis=axis if not isinstance(axis, np.ndarray) else tuple(axis),
                  keepdims=keep_dims),

def warn_deprecated(message, stacklevel=2):  # pragma: no cover
    """Warn deprecated."""

    warnings.warn(
        message,
        category=DeprecationWarning,
        stacklevel=stacklevel
    )

def _accumulate(sequence, func):
    """
    Python2 accumulate implementation taken from
    https://docs.python.org/3/library/itertools.html#itertools.accumulate
    """
    iterator = iter(sequence)
    total = next(iterator)
    yield total
    for element in iterator:
        total = func(total, element)
        yield total

def is_git_repo():
    """Check whether the current folder is a Git repo."""
    cmd = "git", "rev-parse", "--git-dir"
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def Sum(a, axis, keep_dims):
    """
    Sum reduction op.
    """
    return np.sum(a, axis=axis if not isinstance(axis, np.ndarray) else tuple(axis),
                  keepdims=keep_dims),

def is_archlinux():
    """return True if the current distribution is running on debian like OS."""
    if platform.system().lower() == 'linux':
        if platform.linux_distribution() == ('', '', ''):
            # undefined distribution. Fixed in python 3.
            if os.path.exists('/etc/arch-release'):
                return True
    return False

def to_snake_case(s):
    """Converts camel-case identifiers to snake-case."""
    return re.sub('([^_A-Z])([A-Z])', lambda m: m.group(1) + '_' + m.group(2).lower(), s)

def group_exists(groupname):
    """Check if a group exists"""
    try:
        grp.getgrnam(groupname)
        group_exists = True
    except KeyError:
        group_exists = False
    return group_exists

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

def lazy_reverse_binmap(f, xs):
    """
    Same as lazy_binmap, except the parameters are flipped for the binary function
    """
    return (f(y, x) for x, y in zip(xs, xs[1:]))

def isin(self, column, compare_list):
        """
        Returns a boolean list where each elements is whether that element in the column is in the compare_list.

        :param column: single column name, does not work for multiple columns
        :param compare_list: list of items to compare to
        :return: list of booleans
        """
        return [x in compare_list for x in self._data[self._columns.index(column)]]

def jsonify(symbol):
    """ returns json format for symbol """
    try:
        # all symbols have a toJson method, try it
        return json.dumps(symbol.toJson(), indent='  ')
    except AttributeError:
        pass
    return json.dumps(symbol, indent='  ')

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

def table_width(self):
        """Return the width of the table including padding and borders."""
        outer_widths = max_dimensions(self.table_data, self.padding_left, self.padding_right)[2]
        outer_border = 2 if self.outer_border else 0
        inner_border = 1 if self.inner_column_border else 0
        return table_width(outer_widths, outer_border, inner_border)

def __contains__ (self, key):
        """Check lowercase key item."""
        assert isinstance(key, basestring)
        return dict.__contains__(self, key.lower())

def save_as_png(self, filename, width=300, height=250, render_time=1):
        """Open saved html file in an virtual browser and save a screen shot to PNG format."""
        self.driver.set_window_size(width, height)
        self.driver.get('file://{path}/{filename}'.format(
            path=os.getcwd(), filename=filename + ".html"))
        time.sleep(render_time)
        self.driver.save_screenshot(filename + ".png")

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

def to_distribution_values(self, values):
        """
        Returns numpy array of natural logarithms of ``values``.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # avoid RuntimeWarning: divide by zero encountered in log
            return numpy.log(values)

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

def to_distribution_values(self, values):
        """
        Returns numpy array of natural logarithms of ``values``.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # avoid RuntimeWarning: divide by zero encountered in log
            return numpy.log(values)

def is_listish(obj):
    """Check if something quacks like a list."""
    if isinstance(obj, (list, tuple, set)):
        return True
    return is_sequence(obj)

def softplus(attrs, inputs, proto_obj):
    """Applies the sofplus activation function element-wise to the input."""
    new_attrs = translation_utils._add_extra_attributes(attrs, {'act_type' : 'softrelu'})
    return 'Activation', new_attrs, inputs

def is_timestamp(instance):
    """Validates data is a timestamp"""
    if not isinstance(instance, (int, str)):
        return True
    return datetime.fromtimestamp(int(instance))

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

def hard_equals(a, b):
    """Implements the '===' operator."""
    if type(a) != type(b):
        return False
    return a == b

def _get_line_no_from_comments(py_line):
    """Return the line number parsed from the comment or 0."""
    matched = LINECOL_COMMENT_RE.match(py_line)
    if matched:
        return int(matched.group(1))
    else:
        return 0

def get_attr(self, method_name):
        """Get attribute from the target object"""
        return self.attrs.get(method_name) or self.get_callable_attr(method_name)

def is_iterable_of_int(l):
    r""" Checks if l is iterable and contains only integral types """
    if not is_iterable(l):
        return False

    return all(is_int(value) for value in l)

def is_lazy_iterable(obj):
    """
    Returns whether *obj* is iterable lazily, such as generators, range objects, etc.
    """
    return isinstance(obj,
        (types.GeneratorType, collections.MappingView, six.moves.range, enumerate))

def should_skip_logging(func):
    """
    Should we skip logging for this handler?

    """
    disabled = strtobool(request.headers.get("x-request-nolog", "false"))
    return disabled or getattr(func, SKIP_LOGGING, False)

def is_iterable_but_not_string(obj):
    """
    Determine whether or not obj is iterable but not a string (eg, a list, set, tuple etc).
    """
    return hasattr(obj, '__iter__') and not isinstance(obj, str) and not isinstance(obj, bytes)

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

def can_route(self, endpoint, method=None, **kwargs):
        """Make sure we can route to the given endpoint or url.

        This checks for `http.get` permission (or other methods) on the ACL of
        route functions, attached via the `ACL` decorator.

        :param endpoint: A URL or endpoint to check for permission to access.
        :param method: The HTTP method to check; defaults to `'GET'`.
        :param **kwargs: The context to pass to predicates.

        """

        view = flask.current_app.view_functions.get(endpoint)
        if not view:
            endpoint, args = flask._request_ctx.top.match(endpoint)
            view = flask.current_app.view_functions.get(endpoint)
        if not view:
            return False

        return self.can('http.' + (method or 'GET').lower(), view, **kwargs)

def pid_exists(pid):
    """ Determines if a system process identifer exists in process table.
        """
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    else:
        return True

def best(self):
        """
        Returns the element with the highest probability.
        """
        b = (-1e999999, None)
        for k, c in iteritems(self.counts):
            b = max(b, (c, k))
        return b[1]

def is_image(filename):
    """Determine if given filename is an image."""
    # note: isfile() also accepts symlinks
    return os.path.isfile(filename) and filename.lower().endswith(ImageExts)

def format_time(time):
    """ Formats the given time into HH:MM:SS """
    h, r = divmod(time / 1000, 3600)
    m, s = divmod(r, 60)

    return "%02d:%02d:%02d" % (h, m, s)

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

def timedelta2millisecond(td):
    """Get milliseconds from a timedelta."""
    milliseconds = td.days * 24 * 60 * 60 * 1000
    milliseconds += td.seconds * 1000
    milliseconds += td.microseconds / 1000
    return milliseconds

def test_value(self, value):
        """Test if value is an instance of float."""
        if not isinstance(value, float):
            raise ValueError('expected float value: ' + str(type(value)))

def dt_to_ts(value):
    """ If value is a datetime, convert to timestamp """
    if not isinstance(value, datetime):
        return value
    return calendar.timegm(value.utctimetuple()) + value.microsecond / 1000000.0

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

def speedtest(func, *args, **kwargs):
    """ Test the speed of a function. """
    n = 100
    start = time.time()
    for i in range(n): func(*args,**kwargs)
    end = time.time()
    return (end-start)/n

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

def _column_resized(self, col, old_width, new_width):
        """Update the column width."""
        self.dataTable.setColumnWidth(col, new_width)
        self._update_layout()

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

def is_builtin_type(tp):
    """Checks if the given type is a builtin one.
    """
    return hasattr(__builtins__, tp.__name__) and tp is getattr(__builtins__, tp.__name__)

def distinct(xs):
    """Get the list of distinct values with preserving order."""
    # don't use collections.OrderedDict because we do support Python 2.6
    seen = set()
    return [x for x in xs if x not in seen and not seen.add(x)]

def is_timestamp(obj):
    """
    Yaml either have automatically converted it to a datetime object
    or it is a string that will be validated later.
    """
    return isinstance(obj, datetime.datetime) or is_string(obj) or is_int(obj) or is_float(obj)

def from_df(data_frame):
        """Parses data and builds an instance of this class

        :param data_frame: pandas DataFrame
        :return: SqlTable
        """
        labels = data_frame.keys().tolist()
        data = data_frame.values.tolist()
        return SqlTable(labels, data, "{:.3f}", "\n")

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

def transpose(table):
    """
    transpose matrix
    """
    t = []
    for i in range(0, len(table[0])):
        t.append([row[i] for row in table])
    return t

def __contains__ (self, key):
        """Check lowercase key item."""
        assert isinstance(key, basestring)
        return dict.__contains__(self, key.lower())

def truncate(self, table):
        """Empty a table by deleting all of its rows."""
        if isinstance(table, (list, set, tuple)):
            for t in table:
                self._truncate(t)
        else:
            self._truncate(table)

def rsa_eq(key1, key2):
    """
    Only works for RSAPublic Keys

    :param key1:
    :param key2:
    :return:
    """
    pn1 = key1.public_numbers()
    pn2 = key2.public_numbers()
    # Check if two RSA keys are in fact the same
    if pn1 == pn2:
        return True
    else:
        return False

def tokenize(string):
    """Match and yield all the tokens of the input string."""
    for match in TOKENS_REGEX.finditer(string):
        yield Token(match.lastgroup, match.group().strip(), match.span())

def raise_for_not_ok_status(response):
    """
    Raises a `requests.exceptions.HTTPError` if the response has a non-200
    status code.
    """
    if response.code != OK:
        raise HTTPError('Non-200 response code (%s) for url: %s' % (
            response.code, uridecode(response.request.absoluteURI)))

    return response

def dispatch(self, request, *args, **kwargs):
        """Dispatch all HTTP methods to the proxy."""
        self.request = DownstreamRequest(request)
        self.args = args
        self.kwargs = kwargs

        self._verify_config()

        self.middleware = MiddlewareSet(self.proxy_middleware)

        return self.proxy()

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

def keys_to_snake_case(camel_case_dict):
    """
    Make a copy of a dictionary with all keys converted to snake case. This is just calls to_snake_case on
    each of the keys in the dictionary and returns a new dictionary.

    :param camel_case_dict: Dictionary with the keys to convert.
    :type camel_case_dict: Dictionary.

    :return: Dictionary with the keys converted to snake case.
    """
    return dict((to_snake_case(key), value) for (key, value) in camel_case_dict.items())

def is_admin(self):
        """Is the user a system administrator"""
        return self.role == self.roles.administrator.value and self.state == State.approved

def abs_img(img):
    """ Return an image with the binarised version of the data of `img`."""
    bool_img = np.abs(read_img(img).get_data())
    return bool_img.astype(int)

def is_timestamp(obj):
    """
    Yaml either have automatically converted it to a datetime object
    or it is a string that will be validated later.
    """
    return isinstance(obj, datetime.datetime) or is_string(obj) or is_int(obj) or is_float(obj)

def str_dict(some_dict):
    """Convert dict of ascii str/unicode to dict of str, if necessary"""
    return {str(k): str(v) for k, v in some_dict.items()}

def type_converter(text):
    """ I convert strings into integers, floats, and strings! """
    if text.isdigit():
        return int(text), int

    try:
        return float(text), float
    except ValueError:
        return text, STRING_TYPE

def to_list(self):
        """Convert this confusion matrix into a 2x2 plain list of values."""
        return [[int(self.table.cell_values[0][1]), int(self.table.cell_values[0][2])],
                [int(self.table.cell_values[1][1]), int(self.table.cell_values[1][2])]]

def is_complex(dtype):
  """Returns whether this is a complex floating point type."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'is_complex'):
    return dtype.is_complex
  return np.issubdtype(np.dtype(dtype), np.complex)

def stop_capture(self):
        """Stop listening for output from the stenotype machine."""
        super(Treal, self).stop_capture()
        if self._machine:
            self._machine.close()
        self._stopped()

def _tableExists(self, tableName):
        cursor=_conn.execute("""
            SELECT * FROM sqlite_master WHERE name ='{0}' and type='table';
        """.format(tableName))
        exists = cursor.fetchone() is not None
        cursor.close()
        return exists

def search_for_tweets_about(user_id, params):
    """ Search twitter API """
    url = "https://api.twitter.com/1.1/search/tweets.json"
    response = make_twitter_request(url, user_id, params)
    return process_tweets(response.json()["statuses"])

def is_image(filename):
    """Determine if given filename is an image."""
    # note: isfile() also accepts symlinks
    return os.path.isfile(filename) and filename.lower().endswith(ImageExts)

def coerce(self, value):
        """Convert from whatever is given to a list of scalars for the lookup_field."""
        if isinstance(value, dict):
            value = [value]
        if not isiterable_notstring(value):
            value = [value]
        return [coerce_single_instance(self.lookup_field, v) for v in value]

def chmod_plus_w(path):
  """Equivalent of unix `chmod +w path`"""
  path_mode = os.stat(path).st_mode
  path_mode &= int('777', 8)
  path_mode |= stat.S_IWRITE
  os.chmod(path, path_mode)

def get_font_list():
    """Returns a sorted list of all system font names"""

    font_map = pangocairo.cairo_font_map_get_default()
    font_list = [f.get_name() for f in font_map.list_families()]
    font_list.sort()

    return font_list

def on_source_directory_chooser_clicked(self):
        """Autoconnect slot activated when tbSourceDir is clicked."""

        title = self.tr('Set the source directory for script and scenario')
        self.choose_directory(self.source_directory, title)

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

def chunks(iterable, chunk):
    """Yield successive n-sized chunks from an iterable."""
    for i in range(0, len(iterable), chunk):
        yield iterable[i:i + chunk]

def invalidate_cache(cpu, address, size):
        """ remove decoded instruction from instruction cache """
        cache = cpu.instruction_cache
        for offset in range(size):
            if address + offset in cache:
                del cache[address + offset]

def make_unique_ngrams(s, n):
    """Make a set of unique n-grams from a string."""
    return set(s[i:i + n] for i in range(len(s) - n + 1))

def finished(self):
        """
        Must be called to print final progress label.
        """
        self.progress_bar.set_state(ProgressBar.STATE_DONE)
        self.progress_bar.show()

def get_unique_indices(df, axis=1):
    """

    :param df:
    :param axis:
    :return:
    """
    return dict(zip(df.columns.names, dif.columns.levels))

def clear_globals_reload_modules(self):
        """Clears globals and reloads modules"""

        self.code_array.clear_globals()
        self.code_array.reload_modules()

        # Clear result cache
        self.code_array.result_cache.clear()

def uniquify_list(L):
    """Same order unique list using only a list compression."""
    return [e for i, e in enumerate(L) if L.index(e) == i]

def stop(self, dummy_signum=None, dummy_frame=None):
        """ Shutdown process (this method is also a signal handler) """
        logging.info('Shutting down ...')
        self.socket.close()
        sys.exit(0)

def compose_all(tups):
  """Compose all given tuples together."""
  from . import ast  # I weep for humanity
  return functools.reduce(lambda x, y: x.compose(y), map(ast.make_tuple, tups), ast.make_tuple({}))

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

def __cmp__(self, other):
        """Comparsion not implemented."""
        # Stops python 2 from allowing comparsion of arbitrary objects
        raise TypeError('unorderable types: {}, {}'
                        ''.format(self.__class__.__name__, type(other)))

def is_valid_image_extension(file_path):
    """is_valid_image_extension."""
    valid_extensions = ['.jpeg', '.jpg', '.gif', '.png']
    _, extension = os.path.splitext(file_path)
    return extension.lower() in valid_extensions

def install_from_zip(url):
    """Download and unzip from url."""
    fname = 'tmp.zip'
    downlad_file(url, fname)
    unzip_file(fname)
    print("Removing {}".format(fname))
    os.unlink(fname)

def Distance(lat1, lon1, lat2, lon2):
    """Get distance between pairs of lat-lon points"""

    az12, az21, dist = wgs84_geod.inv(lon1, lat1, lon2, lat2)
    return az21, dist

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

def get_conn(self):
        """
        Opens a connection to the cloudant service and closes it automatically if used as context manager.

        .. note::
            In the connection form:
            - 'host' equals the 'Account' (optional)
            - 'login' equals the 'Username (or API Key)' (required)
            - 'password' equals the 'Password' (required)

        :return: an authorized cloudant session context manager object.
        :rtype: cloudant
        """
        conn = self.get_connection(self.cloudant_conn_id)

        self._validate_connection(conn)

        cloudant_session = cloudant(user=conn.login, passwd=conn.password, account=conn.host)

        return cloudant_session

def push(self, el):
        """ Put a new element in the queue. """
        count = next(self.counter)
        heapq.heappush(self._queue, (el, count))

def replace_all(filepath, searchExp, replaceExp):
    """
    Replace all the ocurrences (in a file) of a string with another value.
    """
    for line in fileinput.input(filepath, inplace=1):
        if searchExp in line:
            line = line.replace(searchExp, replaceExp)
        sys.stdout.write(line)

def update_target(self, name, current, total):
        """Updates progress bar for a specified target."""
        self.refresh(self._bar(name, current, total))

def is_valid_ipv6(ip_str):
    """
    Check the validity of an IPv6 address
    """
    try:
        socket.inet_pton(socket.AF_INET6, ip_str)
    except socket.error:
        return False
    return True

def get_input(input_func, input_str):
    """
    Get input from the user given an input function and an input string
    """
    val = input_func("Please enter your {0}: ".format(input_str))
    while not val or not len(val.strip()):
        val = input_func("You didn't enter a valid {0}, please try again: ".format(input_str))
    return val

def table_top_abs(self):
        """Returns the absolute position of table top"""
        table_height = np.array([0, 0, self.table_full_size[2]])
        return string_to_array(self.floor.get("pos")) + table_height

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

def _str_to_list(s):
    """Converts a comma separated string to a list"""
    _list = s.split(",")
    return list(map(lambda i: i.lstrip(), _list))

def dist(x1, x2, axis=0):
    """Return the distance between two points.

    Set axis=1 if x1 is a vector and x2 a matrix to get a vector of distances.
    """
    return np.linalg.norm(x2 - x1, axis=axis)

def get_indentation(func):
    """Extracts a function's indentation as a string,
    In contrast to an inspect.indentsize based implementation,
    this function preserves tabs if present.
    """
    src_lines = getsourcelines(func)[0]
    for line in src_lines:
        if not (line.startswith('@') or line.startswith('def') or line.lstrip().startswith('#')):
            return line[:len(line) - len(line.lstrip())]
    return pytypes.default_indent

def counter_from_str(self, string):
        """Build word frequency list from incoming string."""
        string_list = [chars for chars in string if chars not in self.punctuation]
        string_joined = ''.join(string_list)
        tokens = self.punkt.word_tokenize(string_joined)
        return Counter(tokens)

def count_string_diff(a,b):
    """Return the number of characters in two strings that don't exactly match"""
    shortest = min(len(a), len(b))
    return sum(a[i] != b[i] for i in range(shortest))

def _make_index(df, cols=META_IDX):
    """Create an index from the columns of a dataframe"""
    return pd.MultiIndex.from_tuples(
        pd.unique(list(zip(*[df[col] for col in cols]))), names=tuple(cols))

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

def _sort_lambda(sortedby='cpu_percent',
                 sortedby_secondary='memory_percent'):
    """Return a sort lambda function for the sortedbykey"""
    ret = None
    if sortedby == 'io_counters':
        ret = _sort_io_counters
    elif sortedby == 'cpu_times':
        ret = _sort_cpu_times
    return ret

def _accumulate(sequence, func):
    """
    Python2 accumulate implementation taken from
    https://docs.python.org/3/library/itertools.html#itertools.accumulate
    """
    iterator = iter(sequence)
    total = next(iterator)
    yield total
    for element in iterator:
        total = func(total, element)
        yield total

def items(self, limit=0):
        """Return iterator for items in each page"""
        i = ItemIterator(self.iterator)
        i.limit = limit
        return i

def __init__(self, root_section='lago', defaults={}):
        """__init__
        Args:
            root_section (str): root section in the init
            defaults (dict): Default dictonary to load, can be empty.
        """

        self.root_section = root_section
        self._defaults = defaults
        self._config = defaultdict(dict)
        self._config.update(self.load())
        self._parser = None

def conv2d(x_input, w_matrix):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x_input, w_matrix, strides=[1, 1, 1, 1], padding='SAME')

def items(self, section_name):
        """:return: list((option, value), ...) pairs of all items in the given section"""
        return [(k, v) for k, v in super(GitConfigParser, self).items(section_name) if k != '__name__']

def SegmentMin(a, ids):
    """
    Segmented min op.
    """
    func = lambda idxs: np.amin(a[idxs], axis=0)
    return seg_map(func, a, ids),

def __connect():
    """
    Connect to a redis instance.
    """
    global redis_instance
    if use_tcp_socket:
        redis_instance = redis.StrictRedis(host=hostname, port=port)
    else:
        redis_instance = redis.StrictRedis(unix_socket_path=unix_socket)

def percent_d(data, period):
    """
    %D.

    Formula:
    %D = SMA(%K, 3)
    """
    p_k = percent_k(data, period)
    percent_d = sma(p_k, 3)
    return percent_d

def line_segment(X0, X1):
    r"""
    Calculate the voxel coordinates of a straight line between the two given
    end points

    Parameters
    ----------
    X0 and X1 : array_like
        The [x, y] or [x, y, z] coordinates of the start and end points of
        the line.

    Returns
    -------
    coords : list of lists
        A list of lists containing the X, Y, and Z coordinates of all voxels
        that should be drawn between the start and end points to create a solid
        line.
    """
    X0 = sp.around(X0).astype(int)
    X1 = sp.around(X1).astype(int)
    if len(X0) == 3:
        L = sp.amax(sp.absolute([[X1[0]-X0[0]], [X1[1]-X0[1]], [X1[2]-X0[2]]])) + 1
        x = sp.rint(sp.linspace(X0[0], X1[0], L)).astype(int)
        y = sp.rint(sp.linspace(X0[1], X1[1], L)).astype(int)
        z = sp.rint(sp.linspace(X0[2], X1[2], L)).astype(int)
        return [x, y, z]
    else:
        L = sp.amax(sp.absolute([[X1[0]-X0[0]], [X1[1]-X0[1]]])) + 1
        x = sp.rint(sp.linspace(X0[0], X1[0], L)).astype(int)
        y = sp.rint(sp.linspace(X0[1], X1[1], L)).astype(int)
        return [x, y]

def iter_finds(regex_obj, s):
    """Generate all matches found within a string for a regex and yield each match as a string"""
    if isinstance(regex_obj, str):
        for m in re.finditer(regex_obj, s):
            yield m.group()
    else:
        for m in regex_obj.finditer(s):
            yield m.group()

def get_line_flux(line_wave, wave, flux, **kwargs):
    """Interpolated flux at a given wavelength (calls np.interp)."""
    return np.interp(line_wave, wave, flux, **kwargs)

def to_json(data):
    """Return data as a JSON string."""
    return json.dumps(data, default=lambda x: x.__dict__, sort_keys=True, indent=4)

def method(func):
    """Wrap a function as a method."""
    attr = abc.abstractmethod(func)
    attr.__imethod__ = True
    return attr

def version(self):
        """Spotify version information"""
        url: str = get_url("/service/version.json")
        params = {"service": "remote"}
        r = self._request(url=url, params=params)
        return r.json()

def _dt_to_epoch(dt):
        """Convert datetime to epoch seconds."""
        try:
            epoch = dt.timestamp()
        except AttributeError:  # py2
            epoch = (dt - datetime(1970, 1, 1)).total_seconds()
        return epoch

def multiply(traj):
    """Sophisticated simulation of multiplication"""
    z=traj.x*traj.y
    traj.f_add_result('z',z=z, comment='I am the product of two reals!')

def getOffset(self, loc):
        """ Returns the offset between the given point and this point """
        return Location(loc.x - self.x, loc.y - self.y)

def stdout_display():
    """ Print results straight to stdout """
    if sys.version_info[0] == 2:
        yield SmartBuffer(sys.stdout)
    else:
        yield SmartBuffer(sys.stdout.buffer)

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

def unit_ball_L2(shape):
  """A tensorflow variable tranfomed to be constrained in a L2 unit ball.

  EXPERIMENTAL: Do not use for adverserial examples if you need to be confident
  they are strong attacks. We are not yet confident in this code.
  """
  x = tf.Variable(tf.zeros(shape))
  return constrain_L2(x)

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

def rgba_bytes_tuple(self, x):
        """Provides the color corresponding to value `x` in the
        form of a tuple (R,G,B,A) with int values between 0 and 255.
        """
        return tuple(int(u*255.9999) for u in self.rgba_floats_tuple(x))

def open_json(file_name):
    """
    returns json contents as string
    """
    with open(file_name, "r") as json_data:
        data = json.load(json_data)
        return data

def pop(self, index=-1):
		"""Remove and return the item at index."""
		value = self._list.pop(index)
		del self._dict[value]
		return value

def count_rows_with_nans(X):
    """Count the number of rows in 2D arrays that contain any nan values."""
    if X.ndim == 2:
        return np.where(np.isnan(X).sum(axis=1) != 0, 1, 0).sum()

def compute_gradient(self):
        """Compute the gradient of the current model using the training set
        """
        delta = self.predict(self.X) - self.y
        return delta.dot(self.X) / len(self.X)

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

def quit(self):
        """ Quits the application (called when the last window is closed)
        """
        logger.debug("ArgosApplication.quit called")
        assert len(self.mainWindows) == 0, \
            "Bug: still {} windows present at application quit!".format(len(self.mainWindows))
        self.qApplication.quit()

def _count_leading_whitespace(text):
  """Returns the number of characters at the beginning of text that are whitespace."""
  idx = 0
  for idx, char in enumerate(text):
    if not char.isspace():
      return idx
  return idx + 1

def state(self):
        """Return internal state, useful for testing."""
        return {'c': self.c, 's0': self.s0, 's1': self.s1, 's2': self.s2}

def count(lines):
  """ Counts the word frequences in a list of sentences.

  Note:
    This is a helper function for parallel execution of `Vocabulary.from_text`
    method.
  """
  words = [w for l in lines for w in l.strip().split()]
  return Counter(words)

def indent(self):
        """
        Begins an indented block. Must be used in a 'with' code block.
        All calls to the logger inside of the block will be indented.
        """
        blk = IndentBlock(self, self._indent)
        self._indent += 1
        return blk

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

def get_cov(config):
    """Returns the coverage object of pytest-cov."""

    # Check with hasplugin to avoid getplugin exception in older pytest.
    if config.pluginmanager.hasplugin('_cov'):
        plugin = config.pluginmanager.getplugin('_cov')
        if plugin.cov_controller:
            return plugin.cov_controller.cov
    return None

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

def create_path(path):
    """Creates a absolute path in the file system.

    :param path: The path to be created
    """
    import os
    if not os.path.exists(path):
        os.makedirs(path)

def setblocking(fd, blocking):
    """Set the O_NONBLOCK flag for a file descriptor. Availability: Unix."""
    if not fcntl:
        warnings.warn('setblocking() not supported on Windows')
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    if blocking:
        flags |= os.O_NONBLOCK
    else:
        flags &= ~os.O_NONBLOCK
    fcntl.fcntl(fd, fcntl.F_SETFL, flags)

def from_rectangle(box):
        """ Create a vector randomly within the given rectangle. """
        x = box.left + box.width * random.uniform(0, 1)
        y = box.bottom + box.height * random.uniform(0, 1)
        return Vector(x, y)

def paste(xsel=False):
    """Returns system clipboard contents."""
    selection = "primary" if xsel else "clipboard"
    try:
        return subprocess.Popen(["xclip", "-selection", selection, "-o"], stdout=subprocess.PIPE).communicate()[0].decode("utf-8")
    except OSError as why:
        raise XclipNotFound

def process_kill(pid, sig=None):
    """Send signal to process.
    """
    sig = sig or signal.SIGTERM
    os.kill(pid, sig)

def check_git():
    """Check if git command is available."""
    try:
        with open(os.devnull, "wb") as devnull:
            subprocess.check_call(["git", "--version"], stdout=devnull, stderr=devnull)
    except:
        raise RuntimeError("Please make sure git is installed and on your path.")

def fromDict(cls, _dict):
        """ Builds instance from dictionary of properties. """
        obj = cls()
        obj.__dict__.update(_dict)
        return obj

def get_readline_tail(self, n=10):
        """Get the last n items in readline history."""
        end = self.shell.readline.get_current_history_length() + 1
        start = max(end-n, 1)
        ghi = self.shell.readline.get_history_item
        return [ghi(x) for x in range(start, end)]

def surface(cls, predstr):
        """Instantiate a Pred from its quoted string representation."""
        lemma, pos, sense, _ = split_pred_string(predstr)
        return cls(Pred.SURFACE, lemma, pos, sense, predstr)

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

def _ensure_element(tup, elem):
    """
    Create a tuple containing all elements of tup, plus elem.

    Returns the new tuple and the index of elem in the new tuple.
    """
    try:
        return tup, tup.index(elem)
    except ValueError:
        return tuple(chain(tup, (elem,))), len(tup)

def to_unicode_repr( _letter ):
    """ helpful in situations where browser/app may recognize Unicode encoding
        in the \u0b8e type syntax but not actual unicode glyph/code-point"""
    # Python 2-3 compatible
    return u"u'"+ u"".join( [ u"\\u%04x"%ord(l) for l in _letter ] ) + u"'"

def vectorize(values):
    """
    Takes a value or list of values and returns a single result, joined by ","
    if necessary.
    """
    if isinstance(values, list):
        return ','.join(str(v) for v in values)
    return values

def set_icon(self, bmp):
        """Sets main window icon to given wx.Bitmap"""

        _icon = wx.EmptyIcon()
        _icon.CopyFromBitmap(bmp)
        self.SetIcon(_icon)

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

def _update_bordercolor(self, bordercolor):
        """Updates background color"""

        border_color = wx.SystemSettings_GetColour(wx.SYS_COLOUR_ACTIVEBORDER)
        border_color.SetRGB(bordercolor)

        self.linecolor_choice.SetColour(border_color)

def autoconvert(string):
    """Try to convert variables into datatypes."""
    for fn in (boolify, int, float):
        try:
            return fn(string)
        except ValueError:
            pass
    return string

def on_close(self, evt):
    """
    Pop-up menu and wx.EVT_CLOSE closing event
    """
    self.stop() # DoseWatcher
    if evt.EventObject is not self: # Avoid deadlocks
      self.Close() # wx.Frame
    evt.Skip()

def connect(*args, **kwargs):
    """Create database connection, use TraceCursor as the cursor_factory."""
    kwargs['cursor_factory'] = TraceCursor
    conn = pg_connect(*args, **kwargs)
    return conn

def get_screen_resolution(self):
        """Return the screen resolution of the primary screen."""
        widget = QDesktopWidget()
        geometry = widget.availableGeometry(widget.primaryScreen())
        return geometry.width(), geometry.height()

def from_tuple(tup):
    """Convert a tuple into a range with error handling.

    Parameters
    ----------
    tup : tuple (len 2 or 3)
        The tuple to turn into a range.

    Returns
    -------
    range : range
        The range from the tuple.

    Raises
    ------
    ValueError
        Raised when the tuple length is not 2 or 3.
    """
    if len(tup) not in (2, 3):
        raise ValueError(
            'tuple must contain 2 or 3 elements, not: %d (%r' % (
                len(tup),
                tup,
            ),
        )
    return range(*tup)

def _trim(image):
    """Trim a PIL image and remove white space."""
    background = PIL.Image.new(image.mode, image.size, image.getpixel((0, 0)))
    diff = PIL.ImageChops.difference(image, background)
    diff = PIL.ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        image = image.crop(bbox)
    return image

def cint8_array_to_numpy(cptr, length):
    """Convert a ctypes int pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_int8)):
        return np.fromiter(cptr, dtype=np.int8, count=length)
    else:
        raise RuntimeError('Expected int pointer')

def submit(self, fn, *args, **kwargs):
        """Submit an operation"""
        corofn = asyncio.coroutine(lambda: fn(*args, **kwargs))
        return run_coroutine_threadsafe(corofn(), self.loop)

def import_js(path, lib_name, globals):
    """Imports from javascript source file.
      globals is your globals()"""
    with codecs.open(path_as_local(path), "r", "utf-8") as f:
        js = f.read()
    e = EvalJs()
    e.execute(js)
    var = e.context['var']
    globals[lib_name] = var.to_python()

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

def clone_with_copy(src_path, dest_path):
    """Clone a directory try by copying it.

   Args:
        src_path: The directory to be copied.
        dest_path: The location to copy the directory to.
    """
    log.info('Cloning directory tree %s to %s', src_path, dest_path)
    shutil.copytree(src_path, dest_path)

def count_list(the_list):
    """
    Generates a count of the number of times each unique item appears in a list
    """
    count = the_list.count
    result = [(item, count(item)) for item in set(the_list)]
    result.sort()
    return result

def heappop_max(heap):
    """Maxheap version of a heappop."""
    lastelt = heap.pop()    # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        _siftup_max(heap, 0)
        return returnitem
    return lastelt

def coverage():
    """check code coverage quickly with the default Python"""
    run("coverage run --source {PROJECT_NAME} -m py.test".format(PROJECT_NAME=PROJECT_NAME))
    run("coverage report -m")
    run("coverage html")

    webbrowser.open('file://' + os.path.realpath("htmlcov/index.html"), new=2)

def action(self):
        """
        This class overrides this method
        """
        self.return_value = self.function(*self.args, **self.kwargs)

def string_to_int( s ):
  """Convert a string of bytes into an integer, as per X9.62."""
  result = 0
  for c in s:
    if not isinstance(c, int): c = ord( c )
    result = 256 * result + c
  return result

def one_hot2string(arr, vocab):
    """Convert a one-hot encoded array back to string
    """
    tokens = one_hot2token(arr)
    indexToLetter = _get_index_dict(vocab)

    return [''.join([indexToLetter[x] for x in row]) for row in tokens]

def new_from_list(cls, items, **kwargs):
        """Populates the ListView with a string list.

        Args:
            items (list): list of strings to fill the widget with.
        """
        obj = cls(**kwargs)
        for item in items:
            obj.append(ListItem(item))
        return obj

def csvtolist(inputstr):
    """ converts a csv string into a list """
    reader = csv.reader([inputstr], skipinitialspace=True)
    output = []
    for r in reader:
        output += r
    return output

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

def putkeyword(self, keyword, value, makesubrecord=False):
        """Put the value of a column keyword.
        (see :func:`table.putcolkeyword`)"""
        return self._table.putcolkeyword(self._column, keyword, value, makesubrecord)

def format_op_hdr():
    """
    Build the header
    """
    txt = 'Base Filename'.ljust(36) + ' '
    txt += 'Lines'.rjust(7) + ' '
    txt += 'Words'.rjust(7) + '  '
    txt += 'Unique'.ljust(8) + ''
    return txt

def setValue(self, p_float):
        """Override method to set a value to show it as 0 to 100.

        :param p_float: The float number that want to be set.
        :type p_float: float
        """
        p_float = p_float * 100

        super(PercentageSpinBox, self).setValue(p_float)

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

def makedirs(directory):
    """ Resursively create a named directory. """
    parent = os.path.dirname(os.path.abspath(directory))
    if not os.path.exists(parent):
        makedirs(parent)
    os.mkdir(directory)

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

def create_db_schema(cls, cur, schema_name):
        """
        Create Postgres schema script and execute it on cursor
        """
        create_schema_script = "CREATE SCHEMA {0} ;\n".format(schema_name)
        cur.execute(create_schema_script)

def format_time(time):
    """ Formats the given time into HH:MM:SS """
    h, r = divmod(time / 1000, 3600)
    m, s = divmod(r, 60)

    return "%02d:%02d:%02d" % (h, m, s)

def fromDict(cls, _dict):
        """ Builds instance from dictionary of properties. """
        obj = cls()
        obj.__dict__.update(_dict)
        return obj

def create_rot2d(angle):
    """Create 2D rotation matrix"""
    ca = math.cos(angle)
    sa = math.sin(angle)
    return np.array([[ca, -sa], [sa, ca]])

def one_hot(x, size, dtype=np.float32):
  """Make a n+1 dim one-hot array from n dim int-categorical array."""
  return np.array(x[..., np.newaxis] == np.arange(size), dtype)

def list(self):
        """position in 3d space"""
        return [self._pos3d.x, self._pos3d.y, self._pos3d.z]

def sp_rand(m,n,a):
    """
    Generates an mxn sparse 'd' matrix with round(a*m*n) nonzeros.
    """
    if m == 0 or n == 0: return spmatrix([], [], [], (m,n))
    nnz = min(max(0, int(round(a*m*n))), m*n)
    nz = matrix(random.sample(range(m*n), nnz), tc='i')
    return spmatrix(normal(nnz,1), nz%m, matrix([int(ii) for ii in nz/m]), (m,n))

def create_rot2d(angle):
    """Create 2D rotation matrix"""
    ca = math.cos(angle)
    sa = math.sin(angle)
    return np.array([[ca, -sa], [sa, ca]])

def append_table(self, name, **kwargs):
        """Create a new table."""
        self.stack.append(Table(name, **kwargs))

def build_gui(self, container):
        """
        This is responsible for building the viewer's UI.  It should
        place the UI in `container`.  Override this to make a custom
        UI.
        """
        vbox = Widgets.VBox()
        vbox.set_border_width(0)

        w = Viewers.GingaViewerWidget(viewer=self)
        vbox.add_widget(w, stretch=1)

        # need to put this in an hbox with an expanding label or the
        # browser wants to resize the canvas, distorting it
        hbox = Widgets.HBox()
        hbox.add_widget(vbox, stretch=0)
        hbox.add_widget(Widgets.Label(''), stretch=1)

        container.set_widget(hbox)

def timespan(start_time):
    """Return time in milliseconds from start_time"""

    timespan = datetime.datetime.now() - start_time
    timespan_ms = timespan.total_seconds() * 1000
    return timespan_ms

def add_object(self, obj):
        """Add object to local and app environment storage

        :param obj: Instance of a .NET object
        """
        if obj.top_level_object:
            if isinstance(obj, DotNetNamespace):
                self.namespaces[obj.name] = obj
        self.objects[obj.id] = obj

def get_2D_samples_gauss(n, m, sigma, random_state=None):
    """ Deprecated see  make_2D_samples_gauss   """
    return make_2D_samples_gauss(n, m, sigma, random_state=None)

def security(self):
        """Print security object information for a pdf document"""
        return {k: v for i in self.pdf.resolvedObjects.items() for k, v in i[1].items()}

def str_to_class(class_name):
    """
    Returns a class based on class name    
    """
    mod_str, cls_str = class_name.rsplit('.', 1)
    mod = __import__(mod_str, globals(), locals(), [''])
    cls = getattr(mod, cls_str)
    return cls

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

def next(self):
        """Retrieve the next row."""
        # I'm pretty sure this is the completely wrong way to go about this, but
        # oh well, this works.
        if not hasattr(self, '_iter'):
            self._iter = self.readrow_as_dict()
        return self._iter.next()

def move_datetime_year(dt, direction, num_shifts):
    """
    Move datetime 1 year in the chosen direction.
    unit is a no-op, to keep the API the same as the day case
    """
    delta = relativedelta(years=+num_shifts)
    return _move_datetime(dt, direction, delta)

def getBuffer(x):
    """
    Copy @x into a (modifiable) ctypes byte array
    """
    b = bytes(x)
    return (c_ubyte * len(b)).from_buffer_copy(bytes(x))

def write_color(string, name, style='normal', when='auto'):
    """ Write the given colored string to standard out. """
    write(color(string, name, style, when))

def cint8_array_to_numpy(cptr, length):
    """Convert a ctypes int pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_int8)):
        return np.fromiter(cptr, dtype=np.int8, count=length)
    else:
        raise RuntimeError('Expected int pointer')

def dumped(text, level, indent=2):
    """Put curly brackets round an indented text"""
    return indented("{\n%s\n}" % indented(text, level + 1, indent) or "None", level, indent) + "\n"

def cfloat32_array_to_numpy(cptr, length):
    """Convert a ctypes float pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_float)):
        return np.fromiter(cptr, dtype=np.float32, count=length)
    else:
        raise RuntimeError('Expected float pointer')

def version_jar(self):
		"""
		Special case of version() when the executable is a JAR file.
		"""
		cmd = config.get_command('java')
		cmd.append('-jar')
		cmd += self.cmd
		self.version(cmd=cmd, path=self.cmd[0])

def c_str(string):
    """"Convert a python string to C string."""
    if not isinstance(string, str):
        string = string.decode('ascii')
    return ctypes.c_char_p(string.encode('utf-8'))

def write_line(self, line, count=1):
        """writes the line and count newlines after the line"""
        self.write(line)
        self.write_newlines(count)

def cint8_array_to_numpy(cptr, length):
    """Convert a ctypes int pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_int8)):
        return np.fromiter(cptr, dtype=np.int8, count=length)
    else:
        raise RuntimeError('Expected int pointer')

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

def __enter__(self):
        """ Implements the context manager protocol. Specially useful for asserting exceptions
        """
        clone = self.clone()
        self._contexts.append(clone)
        self.reset()
        return self

def custodian_archive(packages=None):
    """Create a lambda code archive for running custodian.

    Lambda archive currently always includes `c7n` and
    `pkg_resources`. Add additional packages in the mode block.

    Example policy that includes additional packages

    .. code-block:: yaml

        policy:
          name: lambda-archive-example
          resource: s3
          mode:
            packages:
              - botocore

    packages: List of additional packages to include in the lambda archive.

    """
    modules = {'c7n', 'pkg_resources'}
    if packages:
        modules = filter(None, modules.union(packages))
    return PythonPackageArchive(*sorted(modules))

def group(data, num):
    """ Split data into chunks of num chars each """
    return [data[i:i+num] for i in range(0, len(data), num)]

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

def imdecode(image_path):
    """Return BGR image read by opencv"""
    import os
    assert os.path.exists(image_path), image_path + ' not found'
    im = cv2.imread(image_path)
    return im

def _xls2col_widths(self, worksheet, tab):
        """Updates col_widths in code_array"""

        for col in xrange(worksheet.ncols):
            try:
                xls_width = worksheet.colinfo_map[col].width
                pys_width = self.xls_width2pys_width(xls_width)
                self.code_array.col_widths[col, tab] = pys_width

            except KeyError:
                pass

def del_Unnamed(df):
    """
    Deletes all the unnamed columns

    :param df: pandas dataframe
    """
    cols_del=[c for c in df.columns if 'Unnamed' in c]
    return df.drop(cols_del,axis=1)

def to_query_parameters(parameters):
    """Converts DB-API parameter values into query parameters.

    :type parameters: Mapping[str, Any] or Sequence[Any]
    :param parameters: A dictionary or sequence of query parameter values.

    :rtype: List[google.cloud.bigquery.query._AbstractQueryParameter]
    :returns: A list of query parameters.
    """
    if parameters is None:
        return []

    if isinstance(parameters, collections_abc.Mapping):
        return to_query_parameters_dict(parameters)

    return to_query_parameters_list(parameters)

def replace_month_abbr_with_num(date_str, lang=DEFAULT_DATE_LANG):
    """Replace month strings occurrences with month number."""
    num, abbr = get_month_from_date_str(date_str, lang)
    return re.sub(abbr, str(num), date_str, flags=re.IGNORECASE)

def chmod_add_excute(filename):
        """
        Adds execute permission to file.
        :param filename:
        :return:
        """
        st = os.stat(filename)
        os.chmod(filename, st.st_mode | stat.S_IEXEC)

def ToDatetime(self):
    """Converts Timestamp to datetime."""
    return datetime.utcfromtimestamp(
        self.seconds + self.nanos / float(_NANOS_PER_SECOND))

def Max(a, axis, keep_dims):
    """
    Max reduction op.
    """
    return np.amax(a, axis=axis if not isinstance(axis, np.ndarray) else tuple(axis),
                   keepdims=keep_dims),

def parse_datetime(dt_str, format):
    """Create a timezone-aware datetime object from a datetime string."""
    t = time.strptime(dt_str, format)
    return datetime(t[0], t[1], t[2], t[3], t[4], t[5], t[6], pytz.UTC)

def vec_angle(a, b):
    """
    Calculate angle between two vectors
    """
    cosang = np.dot(a, b)
    sinang = fast_norm(np.cross(a, b))
    return np.arctan2(sinang, cosang)

def date_to_timestamp(date):
    """
        date to unix timestamp in milliseconds
    """
    date_tuple = date.timetuple()
    timestamp = calendar.timegm(date_tuple) * 1000
    return timestamp

def ansi(color, text):
    """Wrap text in an ansi escape sequence"""
    code = COLOR_CODES[color]
    return '\033[1;{0}m{1}{2}'.format(code, text, RESET_TERM)

def _converter(self, value):
        """Convert raw input value of the field."""
        if not isinstance(value, datetime.date):
            raise TypeError('{0} is not valid date'.format(value))
        return value

def set_as_object(self, value):
        """
        Sets a new value to map element

        :param value: a new element or map value.
        """
        self.clear()
        map = MapConverter.to_map(value)
        self.append(map)

def datetime_to_timestamp(dt):
    """Convert a UTC datetime to a Unix timestamp"""
    delta = dt - datetime.utcfromtimestamp(0)
    return delta.seconds + delta.days * 24 * 3600

def replace_all(text, dic):
    """Takes a string and dictionary. replaces all occurrences of i with j"""

    for i, j in dic.iteritems():
        text = text.replace(i, j)
    return text

def to_pydatetime(self):
        """
        Converts datetimeoffset object into Python's datetime.datetime object
        @return: time zone aware datetime.datetime
        """
        dt = datetime.datetime.combine(self._date.to_pydate(), self._time.to_pytime())
        from .tz import FixedOffsetTimezone
        return dt.replace(tzinfo=_utc).astimezone(FixedOffsetTimezone(self._offset))

def normalize(self, string):
        """Normalize the string according to normalization list"""
        return ''.join([self._normalize.get(x, x) for x in nfd(string)])

def _DateToEpoch(date):
  """Converts python datetime to epoch microseconds."""
  tz_zero = datetime.datetime.utcfromtimestamp(0)
  diff_sec = int((date - tz_zero).total_seconds())
  return diff_sec * 1000000

def __enter__(self):
        """Enable the download log filter."""
        self.logger = logging.getLogger('pip.download')
        self.logger.addFilter(self)

def datetime_local_to_utc(local):
    """
    Simple function to convert naive :std:`datetime.datetime` object containing
    local time to a naive :std:`datetime.datetime` object with UTC time.
    """
    timestamp = time.mktime(local.timetuple())
    return datetime.datetime.utcfromtimestamp(timestamp)

def triangle_area(pt1, pt2, pt3):
    r"""Return the area of a triangle.

    Parameters
    ----------
    pt1: (X,Y) ndarray
        Starting vertex of a triangle
    pt2: (X,Y) ndarray
        Second vertex of a triangle
    pt3: (X,Y) ndarray
        Ending vertex of a triangle

    Returns
    -------
    area: float
        Area of the given triangle.

    """
    a = 0.0

    a += pt1[0] * pt2[1] - pt2[0] * pt1[1]
    a += pt2[0] * pt3[1] - pt3[0] * pt2[1]
    a += pt3[0] * pt1[1] - pt1[0] * pt3[1]

    return abs(a) / 2

def ToDatetime(self):
    """Converts Timestamp to datetime."""
    return datetime.utcfromtimestamp(
        self.seconds + self.nanos / float(_NANOS_PER_SECOND))

def parse_command_args():
    """Command line parser."""
    parser = argparse.ArgumentParser(description='Register PB devices.')
    parser.add_argument('num_pb', type=int,
                        help='Number of PBs devices to register.')
    return parser.parse_args()

def previous_quarter(d):
    """
    Retrieve the previous quarter for dt
    """
    from django_toolkit.datetime_util import quarter as datetime_quarter
    return quarter( (datetime_quarter(datetime(d.year, d.month, d.day))[0] + timedelta(days=-1)).date() )

def __len__(self):
        """ This will equal 124 for the V1 database. """
        length = 0
        for typ, siz, _ in self.format:
            length += siz
        return length

def string_to_int( s ):
  """Convert a string of bytes into an integer, as per X9.62."""
  result = 0
  for c in s:
    if not isinstance(c, int): c = ord( c )
    result = 256 * result + c
  return result

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

def decompress(f):
    """Decompress a Plan 9 image file.  Assumes f is already cued past the
    initial 'compressed\n' string.
    """

    r = meta(f.read(60))
    return r, decomprest(f, r[4])

def is_scalar(value):
    """Test if the given value is a scalar.

    This function also works with memory mapped array values, in contrast to the numpy is_scalar method.

    Args:
        value: the value to test for being a scalar value

    Returns:
        boolean: if the given value is a scalar or not
    """
    return np.isscalar(value) or (isinstance(value, np.ndarray) and (len(np.squeeze(value).shape) == 0))

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

def is_parameter(self):
        """Whether this is a function parameter."""
        return (isinstance(self.scope, CodeFunction)
                and self in self.scope.parameters)

def input_int_default(question="", default=0):
    """A function that works for both, Python 2.x and Python 3.x.
       It asks the user for input and returns it as a string.
    """
    answer = input_string(question)
    if answer == "" or answer == "yes":
        return default
    else:
        return int(answer)

def figsize(x=8, y=7., aspect=1.):
    """ manually set the default figure size of plots
    ::Arguments::
        x (float): x-axis size
        y (float): y-axis size
        aspect (float): aspect ratio scalar
    """
    # update rcparams with adjusted figsize params
    mpl.rcParams.update({'figure.figsize': (x*aspect, y)})

def export(defn):
    """Decorator to explicitly mark functions that are exposed in a lib."""
    globals()[defn.__name__] = defn
    __all__.append(defn.__name__)
    return defn

def assert_list(self, putative_list, expected_type=string_types, key_arg=None):
    """
    :API: public
    """
    return assert_list(putative_list, expected_type, key_arg=key_arg,
                       raise_type=lambda msg: TargetDefinitionException(self, msg))

def delete_cell(self,  key):
        """Deletes key cell"""

        try:
            self.code_array.pop(key)

        except KeyError:
            pass

        self.grid.code_array.result_cache.clear()

def pop():
        """Remove instance from instance list"""
        pid = os.getpid()
        thread = threading.current_thread()
        Wdb._instances.pop((pid, thread))

def runcoro(async_function):
    """
    Runs an asynchronous function without needing to use await - useful for lambda

    Args:
        async_function (Coroutine): The asynchronous function to run
    """

    future = _asyncio.run_coroutine_threadsafe(async_function, client.loop)
    result = future.result()
    return result

def safe_rmtree(directory):
  """Delete a directory if it's present. If it's not present, no-op."""
  if os.path.exists(directory):
    shutil.rmtree(directory, True)

def datetime_from_timestamp(timestamp, content):
    """
    Helper function to add timezone information to datetime,
    so that datetime is comparable to other datetime objects in recent versions
    that now also have timezone information.
    """
    return set_date_tzinfo(
        datetime.fromtimestamp(timestamp),
        tz_name=content.settings.get('TIMEZONE', None))

def remove(self, key):
        """remove the value found at key from the queue"""
        item = self.item_finder.pop(key)
        item[-1] = None
        self.removed_count += 1

def get_average_length_of_string(strings):
    """Computes average length of words

    :param strings: list of words
    :return: Average length of word on list
    """
    if not strings:
        return 0

    return sum(len(word) for word in strings) / len(strings)

def drop_empty(rows):
    """Transpose the columns into rows, remove all of the rows that are empty after the first cell, then
    transpose back. The result is that columns that have a header but no data in the body are removed, assuming
    the header is the first row. """
    return zip(*[col for col in zip(*rows) if bool(filter(bool, col[1:]))])

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

def copy_user_agent_from_driver(self):
        """ Updates requests' session user-agent with the driver's user agent

        This method will start the browser process if its not already running.
        """
        selenium_user_agent = self.driver.execute_script("return navigator.userAgent;")
        self.headers.update({"user-agent": selenium_user_agent})

def show_xticklabels(self, row, column):
        """Show the x-axis tick labels for a subplot.

        :param row,column: specify the subplot.

        """
        subplot = self.get_subplot_at(row, column)
        subplot.show_xticklabels()

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

def set_xlimits_widgets(self, set_min=True, set_max=True):
        """Populate axis limits GUI with current plot values."""
        xmin, xmax = self.tab_plot.ax.get_xlim()
        if set_min:
            self.w.x_lo.set_text('{0}'.format(xmin))
        if set_max:
            self.w.x_hi.set_text('{0}'.format(xmax))

def is_connected(self):
        """
        Return true if the socket managed by this connection is connected

        :rtype: bool
        """
        try:
            return self.socket is not None and self.socket.getsockname()[1] != 0 and BaseTransport.is_connected(self)
        except socket.error:
            return False

def get_lons_from_cartesian(x__, y__):
    """Get longitudes from cartesian coordinates.
    """
    return rad2deg(arccos(x__ / sqrt(x__ ** 2 + y__ ** 2))) * sign(y__)

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

def region_from_segment(image, segment):
    """given a segment (rectangle) and an image, returns it's corresponding subimage"""
    x, y, w, h = segment
    return image[y:y + h, x:x + w]

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

def pid_exists(pid):
    """ Determines if a system process identifer exists in process table.
        """
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    else:
        return True

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

def _guess_extract_method(fname):
  """Guess extraction method, given file name (or path)."""
  for method, extensions in _EXTRACTION_METHOD_TO_EXTS:
    for ext in extensions:
      if fname.endswith(ext):
        return method
  return ExtractMethod.NO_EXTRACT

def _dotify(cls, data):
    """Add dots."""
    return ''.join(char if char in cls.PRINTABLE_DATA else '.' for char in data)

def is_sparse_vector(x):
    """ x is a 2D sparse matrix with it's first shape equal to 1.
    """
    return sp.issparse(x) and len(x.shape) == 2 and x.shape[0] == 1

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

def check(self, var):
        """Check whether the provided value is a valid enum constant."""
        if not isinstance(var, _str_type): return False
        return _enum_mangle(var) in self._consts

def bootstrap_indexes(data, n_samples=10000):
    """
Given data points data, where axis 0 is considered to delineate points, return
an generator for sets of bootstrap indexes. This can be used as a list
of bootstrap indexes (with list(bootstrap_indexes(data))) as well.
    """
    for _ in xrange(n_samples):
        yield randint(data.shape[0], size=(data.shape[0],))

def is_callable(*p):
    """ True if all the args are functions and / or subroutines
    """
    import symbols
    return all(isinstance(x, symbols.FUNCTION) for x in p)

def _aws_get_instance_by_tag(region, name, tag, raw):
    """Get all instances matching a tag."""
    client = boto3.session.Session().client('ec2', region)
    matching_reservations = client.describe_instances(Filters=[{'Name': tag, 'Values': [name]}]).get('Reservations', [])
    instances = []
    [[instances.append(_aws_instance_from_dict(region, instance, raw))  # pylint: disable=expression-not-assigned
      for instance in reservation.get('Instances')] for reservation in matching_reservations if reservation]
    return instances

def is_image(filename):
    """Determine if given filename is an image."""
    # note: isfile() also accepts symlinks
    return os.path.isfile(filename) and filename.lower().endswith(ImageExts)

def default_static_path():
    """
        Return the path to the javascript bundle
    """
    fdir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(fdir, '../assets/'))

def display_len(text):
    """
    Get the display length of a string. This can differ from the character
    length if the string contains wide characters.
    """
    text = unicodedata.normalize('NFD', text)
    return sum(char_width(char) for char in text)

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

def IsBinary(self, filename):
		"""Returns true if the guessed mimetyped isnt't in text group."""
		mimetype = mimetypes.guess_type(filename)[0]
		if not mimetype:
			return False  # e.g. README, "real" binaries usually have an extension
		# special case for text files which don't start with text/
		if mimetype in TEXT_MIMETYPES:
			return False
		return not mimetype.startswith("text/")

def list_move_to_front(l,value='other'):
    """if the value is in the list, move it to the front and return it."""
    l=list(l)
    if value in l:
        l.remove(value)
        l.insert(0,value)
    return l

def make_symmetric(dict):
    """Makes the given dictionary symmetric. Values are assumed to be unique."""
    for key, value in list(dict.items()):
        dict[value] = key
    return dict

def retrieve_by_id(self, id_):
        """Return a JSSObject for the element with ID id_"""
        items_with_id = [item for item in self if item.id == int(id_)]
        if len(items_with_id) == 1:
            return items_with_id[0].retrieve()

def purge_dict(idict):
    """Remove null items from a dictionary """
    odict = {}
    for key, val in idict.items():
        if is_null(val):
            continue
        odict[key] = val
    return odict

def install_rpm_py():
    """Install RPM Python binding."""
    python_path = sys.executable
    cmd = '{0} install.py'.format(python_path)
    exit_status = os.system(cmd)
    if exit_status != 0:
        raise Exception('Command failed: {0}'.format(cmd))

def update(self, other_dict):
        """update() extends rather than replaces existing key lists."""
        for key, value in iter_multi_items(other_dict):
            MultiDict.add(self, key, value)

def is_iter_non_string(obj):
    """test if object is a list or tuple"""
    if isinstance(obj, list) or isinstance(obj, tuple):
        return True
    return False

def C_dict2array(C):
    """Convert an OrderedDict containing C values to a 1D array."""
    return np.hstack([np.asarray(C[k]).ravel() for k in C_keys])

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

def dict_to_querystring(dictionary):
    """Converts a dict to a querystring suitable to be appended to a URL."""
    s = u""
    for d in dictionary.keys():
        s = unicode.format(u"{0}{1}={2}&", s, d, dictionary[d])
    return s[:-1]

def convert_timeval(seconds_since_epoch):
    """Convert time into C style timeval."""
    frac, whole = math.modf(seconds_since_epoch)
    microseconds = math.floor(frac * 1000000)
    seconds = math.floor(whole)
    return seconds, microseconds

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

def growthfromrange(rangegrowth, startdate, enddate):
    """
    Annual growth given growth from start date to end date.
    """
    _yrs = (pd.Timestamp(enddate) - pd.Timestamp(startdate)).total_seconds() /\
            dt.timedelta(365.25).total_seconds()
    return yrlygrowth(rangegrowth, _yrs)

def make_symmetric(dict):
    """Makes the given dictionary symmetric. Values are assumed to be unique."""
    for key, value in list(dict.items()):
        dict[value] = key
    return dict

def double_exponential_moving_average(data, period):
    """
    Double Exponential Moving Average.

    Formula:
    DEMA = 2*EMA - EMA(EMA)
    """
    catch_errors.check_for_period_error(data, period)

    dema = (2 * ema(data, period)) - ema(ema(data, period), period)
    return dema

def multi_pop(d, *args):
    """ pops multiple keys off a dict like object """
    retval = {}
    for key in args:
        if key in d:
            retval[key] = d.pop(key)
    return retval

def inh(table):
    """
    inverse hyperbolic sine transformation
    """
    t = []
    for i in table:
        t.append(np.ndarray.tolist(np.arcsinh(i)))
    return t

def dict_update_newkeys(dict_, dict2):
    """ Like dict.update, but does not overwrite items """
    for key, val in six.iteritems(dict2):
        if key not in dict_:
            dict_[key] = val

def v_normalize(v):
    """
    Normalizes the given vector.
    
    The vector given may have any number of dimensions.
    """
    vmag = v_magnitude(v)
    return [ v[i]/vmag  for i in range(len(v)) ]

def recursively_update(d, d2):
  """dict.update but which merges child dicts (dict2 takes precedence where there's conflict)."""
  for k, v in d2.items():
    if k in d:
      if isinstance(v, dict):
        recursively_update(d[k], v)
        continue
    d[k] = v

def _days_in_month(date):
    """The number of days in the month of the given date"""
    if date.month == 12:
        reference = type(date)(date.year + 1, 1, 1)
    else:
        reference = type(date)(date.year, date.month + 1, 1)
    return (reference - timedelta(days=1)).day

def update(self, other_dict):
        """update() extends rather than replaces existing key lists."""
        for key, value in iter_multi_items(other_dict):
            MultiDict.add(self, key, value)

def count_rows_with_nans(X):
    """Count the number of rows in 2D arrays that contain any nan values."""
    if X.ndim == 2:
        return np.where(np.isnan(X).sum(axis=1) != 0, 1, 0).sum()

def c_array(ctype, values):
    """Convert a python string to c array."""
    if isinstance(values, np.ndarray) and values.dtype.itemsize == ctypes.sizeof(ctype):
        return (ctype * len(values)).from_buffer_copy(values)
    return (ctype * len(values))(*values)

def stderr(a):
    """
    Calculate the standard error of a.
    """
    return np.nanstd(a) / np.sqrt(sum(np.isfinite(a)))

def is_readable_dir(path):
  """Returns whether a path names an existing directory we can list and read files from."""
  return os.path.isdir(path) and os.access(path, os.R_OK) and os.access(path, os.X_OK)

def import_js(path, lib_name, globals):
    """Imports from javascript source file.
      globals is your globals()"""
    with codecs.open(path_as_local(path), "r", "utf-8") as f:
        js = f.read()
    e = EvalJs()
    e.execute(js)
    var = e.context['var']
    globals[lib_name] = var.to_python()

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

def many_until1(these, term):
    """Like many_until but must consume at least one of these.
    """
    first = [these()]
    these_results, term_result = many_until(these, term)
    return (first + these_results, term_result)

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

def reject(self):
        """
        Rejects the snapshot and closes the widget.
        """
        if self.hideWindow():
            self.hideWindow().show()
            
        self.close()
        self.deleteLater()

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

def _trim_zeros_complex(str_complexes, na_rep='NaN'):
    """
    Separates the real and imaginary parts from the complex number, and
    executes the _trim_zeros_float method on each of those.
    """
    def separate_and_trim(str_complex, na_rep):
        num_arr = str_complex.split('+')
        return (_trim_zeros_float([num_arr[0]], na_rep) +
                ['+'] +
                _trim_zeros_float([num_arr[1][:-1]], na_rep) +
                ['j'])

    return [''.join(separate_and_trim(x, na_rep)) for x in str_complexes]

def to_unicode_repr( _letter ):
    """ helpful in situations where browser/app may recognize Unicode encoding
        in the \u0b8e type syntax but not actual unicode glyph/code-point"""
    # Python 2-3 compatible
    return u"u'"+ u"".join( [ u"\\u%04x"%ord(l) for l in _letter ] ) + u"'"

def _text_to_graphiz(self, text):
        """create a graphviz graph from text"""
        dot = Source(text, format='svg')
        return dot.pipe().decode('utf-8')

def to_identifier(s):
  """
  Convert snake_case to camel_case.
  """
  if s.startswith('GPS'):
      s = 'Gps' + s[3:]
  return ''.join([i.capitalize() for i in s.split('_')]) if '_' in s else s

def display(self):
        """ Get screen width and height """
        w, h = self.session.window_size()
        return Display(w*self.scale, h*self.scale)

def _snake_to_camel_case(value):
    """Convert snake case string to camel case."""
    words = value.split("_")
    return words[0] + "".join(map(str.capitalize, words[1:]))

def direct2dDistance(self, point):
        """consider the distance between two mapPoints, ignoring all terrain, pathing issues"""
        if not isinstance(point, MapPoint): return 0.0
        return  ((self.x-point.x)**2 + (self.y-point.y)**2)**(0.5) # simple distance formula

def _snake_to_camel_case(value):
    """Convert snake case string to camel case."""
    words = value.split("_")
    return words[0] + "".join(map(str.capitalize, words[1:]))

def deleted(self, instance):
        """
        Convenience method for deleting a model (automatically commits the
        delete to the database and returns with an HTTP 204 status code)
        """
        self.session_manager.delete(instance, commit=True)
        return '', HTTPStatus.NO_CONTENT

def to_identifier(s):
  """
  Convert snake_case to camel_case.
  """
  if s.startswith('GPS'):
      s = 'Gps' + s[3:]
  return ''.join([i.capitalize() for i in s.split('_')]) if '_' in s else s

def show_image(self, key):
        """Show image (item is a PIL image)"""
        data = self.model.get_data()
        data[key].show()

def check_output(args):
    """Runs command and returns the output as string."""
    log.debug('run: %s', args)
    out = subprocess.check_output(args=args).decode('utf-8')
    log.debug('out: %r', out)
    return out

def page_title(step, title):
    """
    Check that the page title matches the given one.
    """

    with AssertContextManager(step):
        assert_equals(world.browser.title, title)

def __run(self):
    """Hacked run function, which installs the trace."""
    sys.settrace(self.globaltrace)
    self.__run_backup()
    self.run = self.__run_backup

def main():
    """Parse the command line and run :func:`migrate`."""
    parser = get_args_parser()
    args = parser.parse_args()
    config = Config.from_parse_args(args)
    migrate(config)

def is_equal_strings_ignore_case(first, second):
    """The function compares strings ignoring case"""
    if first and second:
        return first.upper() == second.upper()
    else:
        return not (first or second)

def requests_post(url, data=None, json=None, **kwargs):
    """Requests-mock requests.post wrapper."""
    return requests_request('post', url, data=data, json=json, **kwargs)

def __to_localdatetime(val):
    """Convert val into a local datetime for tz Europe/Amsterdam."""
    try:
        dt = datetime.strptime(val, __DATE_FORMAT)
        dt = pytz.timezone(__TIMEZONE).localize(dt)
        return dt
    except (ValueError, TypeError):
        return None

def merge(database=None, directory=None, verbose=None):
    """Merge migrations into one."""
    router = get_router(directory, database, verbose)
    router.merge()

def strToBool(val):
    """
    Helper function to turn a string representation of "true" into
    boolean True.
    """
    if isinstance(val, str):
        val = val.lower()

    return val in ['true', 'on', 'yes', True]

def _add_params_docstring(params):
    """ Add params to doc string
    """
    p_string = "\nAccepts the following paramters: \n"
    for param in params:
         p_string += "name: %s, required: %s, description: %s \n" % (param['name'], param['required'], param['description'])
    return p_string

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

def _add_params_docstring(params):
    """ Add params to doc string
    """
    p_string = "\nAccepts the following paramters: \n"
    for param in params:
         p_string += "name: %s, required: %s, description: %s \n" % (param['name'], param['required'], param['description'])
    return p_string

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

def _attrprint(d, delimiter=', '):
    """Print a dictionary of attributes in the DOT format"""
    return delimiter.join(('"%s"="%s"' % item) for item in sorted(d.items()))

def callJavaFunc(func, *args):
    """ Call Java Function """
    gateway = _get_gateway()
    args = [_py2java(gateway, a) for a in args]
    result = func(*args)
    return _java2py(gateway, result)

def underscore(text):
    """Converts text that may be camelcased into an underscored format"""
    return UNDERSCORE[1].sub(r'\1_\2', UNDERSCORE[0].sub(r'\1_\2', text)).lower()

def to_str(obj):
    """Attempts to convert given object to a string object
    """
    if not isinstance(obj, str) and PY3 and isinstance(obj, bytes):
        obj = obj.decode('utf-8')
    return obj if isinstance(obj, string_types) else str(obj)

def ucamel_method(func):
    """
    Decorator to ensure the given snake_case method is also written in
    UpperCamelCase in the given namespace. That was mainly written to
    avoid confusion when using wxPython and its UpperCamelCaseMethods.
    """
    frame_locals = inspect.currentframe().f_back.f_locals
    frame_locals[snake2ucamel(func.__name__)] = func
    return func

def image_set_aspect(aspect=1.0, axes="gca"):
    """
    sets the aspect ratio of the current zoom level of the imshow image
    """
    if axes is "gca": axes = _pylab.gca()

    e = axes.get_images()[0].get_extent()
    axes.set_aspect(abs((e[1]-e[0])/(e[3]-e[2]))/aspect)

def _converter(self, value):
        """Convert raw input value of the field."""
        if not isinstance(value, datetime.date):
            raise TypeError('{0} is not valid date'.format(value))
        return value

def bbox(self):
        """
        The minimal `~photutils.aperture.BoundingBox` for the cutout
        region with respect to the original (large) image.
        """

        return BoundingBox(self.slices[1].start, self.slices[1].stop,
                           self.slices[0].start, self.slices[0].stop)

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

def _draw_lines_internal(self, coords, colour, bg):
        """Helper to draw lines connecting a set of nodes that are scaled for the Screen."""
        for i, (x, y) in enumerate(coords):
            if i == 0:
                self._screen.move(x, y)
            else:
                self._screen.draw(x, y, colour=colour, bg=bg, thin=True)

def change_dir(directory):
  """
  Wraps a function to run in a given directory.

  """
  def cd_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
      org_path = os.getcwd()
      os.chdir(directory)
      func(*args, **kwargs)
      os.chdir(org_path)
    return wrapper
  return cd_decorator

def remove_non_magic_cols(self):
        """
        Remove all non-MagIC columns from all tables.
        """
        for table_name in self.tables:
            table = self.tables[table_name]
            table.remove_non_magic_cols_from_table()

def title(msg):
    """Sets the title of the console window."""
    if sys.platform.startswith("win"):
        ctypes.windll.kernel32.SetConsoleTitleW(tounicode(msg))

def TextWidget(*args, **kw):
    """Forces a parameter value to be text"""
    kw['value'] = str(kw['value'])
    kw.pop('options', None)
    return TextInput(*args,**kw)

def scale_dtype(arr, dtype):
    """Convert an array from 0..1 to dtype, scaling up linearly
    """
    max_int = np.iinfo(dtype).max
    return (arr * max_int).astype(dtype)

def export(defn):
    """Decorator to explicitly mark functions that are exposed in a lib."""
    globals()[defn.__name__] = defn
    __all__.append(defn.__name__)
    return defn

def shape_list(l,shape,dtype):
    """ Shape a list of lists into the appropriate shape and data type """
    return np.array(l, dtype=dtype).reshape(shape)

def ffmpeg_works():
  """Tries to encode images with ffmpeg to check if it works."""
  images = np.zeros((2, 32, 32, 3), dtype=np.uint8)
  try:
    _encode_gif(images, 2)
    return True
  except (IOError, OSError):
    return False

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

def clear_es():
        """Clear all indexes in the es core"""
        # TODO: should receive a catalog slug.
        ESHypermap.es.indices.delete(ESHypermap.index_name, ignore=[400, 404])
        LOGGER.debug('Elasticsearch: Index cleared')

def copy_user_agent_from_driver(self):
        """ Updates requests' session user-agent with the driver's user agent

        This method will start the browser process if its not already running.
        """
        selenium_user_agent = self.driver.execute_script("return navigator.userAgent;")
        self.headers.update({"user-agent": selenium_user_agent})

def keys(self):
        """Return ids of all indexed documents."""
        result = []
        if self.fresh_index is not None:
            result += self.fresh_index.keys()
        if self.opt_index is not None:
            result += self.opt_index.keys()
        return result

def move_datetime_year(dt, direction, num_shifts):
    """
    Move datetime 1 year in the chosen direction.
    unit is a no-op, to keep the API the same as the day case
    """
    delta = relativedelta(years=+num_shifts)
    return _move_datetime(dt, direction, delta)

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

def get_by(self, name):
    """get element by name"""
    return next((item for item in self if item.name == name), None)

def ylabelsize(self, size, index=1):
        """Set the size of the label.

        Parameters
        ----------
        size : int

        Returns
        -------
        Chart

        """
        self.layout['yaxis' + str(index)]['titlefont']['size'] = size
        return self

def Softsign(a):
    """
    Softsign op.
    """
    return np.divide(a, np.add(np.abs(a), 1)),

def title(msg):
    """Sets the title of the console window."""
    if sys.platform.startswith("win"):
        ctypes.windll.kernel32.SetConsoleTitleW(tounicode(msg))

def isetdiff_flags(list1, list2):
    """
    move to util_iter
    """
    set2 = set(list2)
    return (item not in set2 for item in list1)

def _extract_value(self, value):
        """If the value is true/false/null replace with Python equivalent."""
        return ModelEndpoint._value_map.get(smart_str(value).lower(), value)

def __getattribute__(self, attr):
        """Retrieve attr from current active etree implementation"""
        if (attr not in object.__getattribute__(self, '__dict__')
                and attr not in Etree.__dict__):
            return object.__getattribute__(self._etree, attr)
        return object.__getattribute__(self, attr)

def decode_unicode_string(string):
    """
    Decode string encoded by `unicode_string`
    """
    if string.startswith('[BASE64-DATA]') and string.endswith('[/BASE64-DATA]'):
        return base64.b64decode(string[len('[BASE64-DATA]'):-len('[/BASE64-DATA]')])
    return string

def root_parent(self, category=None):
        """ Returns the topmost parent of the current category. """
        return next(filter(lambda c: c.is_root, self.hierarchy()))

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

def add_element_to_doc(doc, tag, value):
    """Set text value of an etree.Element of tag, appending a new element with given tag if it doesn't exist."""
    element = doc.find(".//%s" % tag)
    if element is None:
        element = etree.SubElement(doc, tag)
    element.text = value

def isin(value, values):
    """ Check that value is in values """
    for i, v in enumerate(value):
        if v not in np.array(values)[:, i]:
            return False
    return True

def read_from_file(file_path, encoding="utf-8"):
    """
    Read helper method

    :type file_path: str|unicode
    :type encoding: str|unicode
    :rtype: str|unicode
    """
    with codecs.open(file_path, "r", encoding) as f:
        return f.read()

def is_sequence(obj):
    """Check if `obj` is a sequence, but not a string or bytes."""
    return isinstance(obj, Sequence) and not (
        isinstance(obj, str) or BinaryClass.is_valid_type(obj))

def _to_lower_alpha_only(s):
    """Return a lowercased string with non alphabetic chars removed.

    White spaces are not to be removed."""
    s = re.sub(r'\n', ' ',  s.lower())
    return re.sub(r'[^a-z\s]', '', s)

def isin(value, values):
    """ Check that value is in values """
    for i, v in enumerate(value):
        if v not in np.array(values)[:, i]:
            return False
    return True

def write_enum(fo, datum, schema):
    """An enum is encoded by a int, representing the zero-based position of
    the symbol in the schema."""
    index = schema['symbols'].index(datum)
    write_int(fo, index)

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

def is_float_array(l):
    r"""Checks if l is a numpy array of floats (any dimension

    """
    if isinstance(l, np.ndarray):
        if l.dtype.kind == 'f':
            return True
    return False

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

def task_property_present_predicate(service, task, prop):
    """ True if the json_element passed is present for the task specified.
    """
    try:
        response = get_service_task(service, task)
    except Exception as e:
        pass

    return (response is not None) and (prop in response)

def fast_exit(code):
    """Exit without garbage collection, this speeds up exit by about 10ms for
    things like bash completion.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)

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

def prepare_query_params(**kwargs):
    """
    Prepares given parameters to be used in querystring.
    """
    return [
        (sub_key, sub_value)
        for key, value in kwargs.items()
        for sub_key, sub_value in expand(value, key)
        if sub_value is not None
    ]

def is_iterable_but_not_string(obj):
    """
    Determine whether or not obj is iterable but not a string (eg, a list, set, tuple etc).
    """
    return hasattr(obj, '__iter__') and not isinstance(obj, str) and not isinstance(obj, bytes)

def expandvars_dict(settings):
    """Expands all environment variables in a settings dictionary."""
    return dict(
        (key, os.path.expandvars(value))
        for key, value in settings.iteritems()
    )

def is_nested_object(obj):
    """
    return a boolean if we have a nested object, e.g. a Series with 1 or
    more Series elements

    This may not be necessarily be performant.

    """

    if isinstance(obj, ABCSeries) and is_object_dtype(obj):

        if any(isinstance(v, ABCSeries) for v in obj.values):
            return True

    return False

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

def _isstring(dtype):
    """Given a numpy dtype, determines whether it is a string. Returns True
    if the dtype is string or unicode.
    """
    return dtype.type == numpy.unicode_ or dtype.type == numpy.string_

def is_function(self):
        """return True if callback is a vanilla plain jane function"""
        if self.is_instance() or self.is_class(): return False
        return isinstance(self.callback, (Callable, classmethod))

def launched():
    """Test whether the current python environment is the correct lore env.

    :return:  :any:`True` if the environment is launched
    :rtype: bool
    """
    if not PREFIX:
        return False

    return os.path.realpath(sys.prefix) == os.path.realpath(PREFIX)

def ffmpeg_version():
    """Returns the available ffmpeg version

    Returns
    ----------
    version : str
        version number as string
    """

    cmd = [
        'ffmpeg',
        '-version'
    ]

    output = sp.check_output(cmd)
    aac_codecs = [
        x for x in
        output.splitlines() if "ffmpeg version " in str(x)
    ][0]
    hay = aac_codecs.decode('ascii')
    match = re.findall(r'ffmpeg version (\d+\.)?(\d+\.)?(\*|\d+)', hay)
    if match:
        return "".join(match[0])
    else:
        return None

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

def is_iterable_of_int(l):
    r""" Checks if l is iterable and contains only integral types """
    if not is_iterable(l):
        return False

    return all(is_int(value) for value in l)

def file_read(filename):
    """Read a file and close it.  Returns the file source."""
    fobj = open(filename,'r');
    source = fobj.read();
    fobj.close()
    return source

def is_archlinux():
    """return True if the current distribution is running on debian like OS."""
    if platform.system().lower() == 'linux':
        if platform.linux_distribution() == ('', '', ''):
            # undefined distribution. Fixed in python 3.
            if os.path.exists('/etc/arch-release'):
                return True
    return False

def _maybe_fill(arr, fill_value=np.nan):
    """
    if we have a compatible fill_value and arr dtype, then fill
    """
    if _isna_compat(arr, fill_value):
        arr.fill(fill_value)
    return arr

def _is_proper_sequence(seq):
    """Returns is seq is sequence and not string."""
    return (isinstance(seq, collections.abc.Sequence) and
            not isinstance(seq, str))

def get_column_definition(self, table, column):
        """Retrieve the column definition statement for a column from a table."""
        # Parse column definitions for match
        for col in self.get_column_definition_all(table):
            if col.strip('`').startswith(column):
                return col.strip(',')

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

def filter_dict_by_key(d, keys):
    """Filter the dict *d* to remove keys not in *keys*."""
    return {k: v for k, v in d.items() if k in keys}

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

def remove_na_arraylike(arr):
    """
    Return array-like containing only true/non-NaN values, possibly empty.
    """
    if is_extension_array_dtype(arr):
        return arr[notna(arr)]
    else:
        return arr[notna(lib.values_from_object(arr))]

def test_value(self, value):
        """Test if value is an instance of float."""
        if not isinstance(value, float):
            raise ValueError('expected float value: ' + str(type(value)))

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

def contains_geometric_info(var):
    """ Check whether the passed variable is a tuple with two floats or integers """
    return isinstance(var, tuple) and len(var) == 2 and all(isinstance(val, (int, float)) for val in var)

def findfirst(f, coll):
    """Return first occurrence matching f, otherwise None"""
    result = list(dropwhile(f, coll))
    return result[0] if result else None

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

def fit_gaussian(x, y, yerr, p0):
    """ Fit a Gaussian to the data """
    try:
        popt, pcov = curve_fit(gaussian, x, y, sigma=yerr, p0=p0, absolute_sigma=True)
    except RuntimeError:
        return [0],[0]
    return popt, pcov

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

def b(s):
	""" Encodes Unicode strings to byte strings, if necessary. """

	return s if isinstance(s, bytes) else s.encode(locale.getpreferredencoding())

def is_lazy_iterable(obj):
    """
    Returns whether *obj* is iterable lazily, such as generators, range objects, etc.
    """
    return isinstance(obj,
        (types.GeneratorType, collections.MappingView, six.moves.range, enumerate))

def initialize_api(flask_app):
    """Initialize an API."""
    if not flask_restplus:
        return

    api = flask_restplus.Api(version="1.0", title="My Example API")
    api.add_resource(HelloWorld, "/hello")

    blueprint = flask.Blueprint("api", __name__, url_prefix="/api")
    api.init_app(blueprint)
    flask_app.register_blueprint(blueprint)

def __contains__ (self, key):
        """Check lowercase key item."""
        assert isinstance(key, basestring)
        return dict.__contains__(self, key.lower())

def init_app(self, app):
        """Initialize Flask application."""
        app.config.from_pyfile('{0}.cfg'.format(app.name), silent=True)

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

def chmod_plus_w(path):
  """Equivalent of unix `chmod +w path`"""
  path_mode = os.stat(path).st_mode
  path_mode &= int('777', 8)
  path_mode |= stat.S_IWRITE
  os.chmod(path, path_mode)

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

def render_template(template_name, **context):
    """Render a template into a response."""
    tmpl = jinja_env.get_template(template_name)
    context["url_for"] = url_for
    return Response(tmpl.render(context), mimetype="text/html")

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

def getFlaskResponse(responseString, httpStatus=200):
    """
    Returns a Flask response object for the specified data and HTTP status.
    """
    return flask.Response(responseString, status=httpStatus, mimetype=MIMETYPE)

def __eq__(self, other):
        """Determine if two objects are equal."""
        return isinstance(other, self.__class__) \
            and self._freeze() == other._freeze()

async def json_or_text(response):
    """Turns response into a properly formatted json or text object"""
    text = await response.text()
    if response.headers['Content-Type'] == 'application/json; charset=utf-8':
        return json.loads(text)
    return text

def random_choice(sequence):
    """ Same as :meth:`random.choice`, but also supports :class:`set` type to be passed as sequence. """
    return random.choice(tuple(sequence) if isinstance(sequence, set) else sequence)

def flatten(l):
    """Flatten a nested list."""
    return sum(map(flatten, l), []) \
        if isinstance(l, list) or isinstance(l, tuple) else [l]

def closest(xarr, val):
    """ Return the index of the closest in xarr to value val """
    idx_closest = np.argmin(np.abs(np.array(xarr) - val))
    return idx_closest

def flatten_array(grid):
    """
    Takes a multi-dimensional array and returns a 1 dimensional array with the
    same contents.
    """
    grid = [grid[i][j] for i in range(len(grid)) for j in range(len(grid[i]))]
    while type(grid[0]) is list:
        grid = flatten_array(grid)
    return grid

def get_duckduckgo_links(limit, params, headers):
	"""
	function to fetch links equal to limit

	duckduckgo pagination is not static, so there is a limit on
	maximum number of links that can be scraped
	"""
	resp = s.get('https://duckduckgo.com/html', params = params, headers = headers)
	links = scrape_links(resp.content, engine = 'd')
	return links[:limit]

def fmt_sz(intval):
    """ Format a byte sized value.
    """
    try:
        return fmt.human_size(intval)
    except (ValueError, TypeError):
        return "N/A".rjust(len(fmt.human_size(0)))

def read_utf8(fh, byteorder, dtype, count, offsetsize):
    """Read tag data from file and return as unicode string."""
    return fh.read(count).decode('utf-8')

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

def read_utf8(fh, byteorder, dtype, count, offsetsize):
    """Read tag data from file and return as unicode string."""
    return fh.read(count).decode('utf-8')

def replace_nones(dict_or_list):
    """Update a dict or list in place to replace
    'none' string values with Python None."""

    def replace_none_in_value(value):
        if isinstance(value, basestring) and value.lower() == "none":
            return None
        return value

    items = dict_or_list.iteritems() if isinstance(dict_or_list, dict) else enumerate(dict_or_list)

    for accessor, value in items:
        if isinstance(value, (dict, list)):
            replace_nones(value)
        else:
            dict_or_list[accessor] = replace_none_in_value(value)

def b(s):
	""" Encodes Unicode strings to byte strings, if necessary. """

	return s if isinstance(s, bytes) else s.encode(locale.getpreferredencoding())

def safe_format(s, **kwargs):
  """
  :type s str
  """
  return string.Formatter().vformat(s, (), defaultdict(str, **kwargs))

def get_column_keys_and_names(table):
    """
    Return a generator of tuples k, c such that k is the name of the python attribute for
    the column and c is the name of the column in the sql table.
    """
    ins = inspect(table)
    return ((k, c.name) for k, c in ins.mapper.c.items())

def _pad(self, text):
        """Pad the text."""
        top_bottom = ("\n" * self._padding) + " "
        right_left = " " * self._padding * self.PAD_WIDTH
        return top_bottom + right_left + text + right_left + top_bottom

def write_color(string, name, style='normal', when='auto'):
    """ Write the given colored string to standard out. """
    write(color(string, name, style, when))

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

def from_array(cls, arr):
        """Convert a structured NumPy array into a Table."""
        return cls().with_columns([(f, arr[f]) for f in arr.dtype.names])

def connect():
    """Connect to FTP server, login and return an ftplib.FTP instance."""
    ftp_class = ftplib.FTP if not SSL else ftplib.FTP_TLS
    ftp = ftp_class(timeout=TIMEOUT)
    ftp.connect(HOST, PORT)
    ftp.login(USER, PASSWORD)
    if SSL:
        ftp.prot_p()  # secure data connection
    return ftp

def _merge_maps(m1, m2):
    """merge two Mapping objects, keeping the type of the first mapping"""
    return type(m1)(chain(m1.items(), m2.items()))

def write(url, content, **args):
    """Put an object into a ftps URL."""
    with FTPSResource(url, **args) as resource:
        resource.write(content)

def cartesian_lists(d):
    """
    turns a dict of lists into a list of dicts that represents
    the cartesian product of the initial lists

    Example
    -------
    cartesian_lists({'a':[0, 2], 'b':[3, 4, 5]}
    returns
    [ {'a':0, 'b':3}, {'a':0, 'b':4}, ... {'a':2, 'b':5} ]

    """
    return [{k: v for k, v in zip(d.keys(), args)}
            for args in itertools.product(*d.values())]

def see_doc(obj_with_doc):
    """Copy docstring from existing object to the decorated callable."""
    def decorator(fn):
        fn.__doc__ = obj_with_doc.__doc__
        return fn
    return decorator

def compare(a, b):
    """
     Compare items in 2 arrays. Returns sum(abs(a(i)-b(i)))
    """
    s=0
    for i in range(len(a)):
        s=s+abs(a[i]-b[i])
    return s

def is_scalar(value):
    """Test if the given value is a scalar.

    This function also works with memory mapped array values, in contrast to the numpy is_scalar method.

    Args:
        value: the value to test for being a scalar value

    Returns:
        boolean: if the given value is a scalar or not
    """
    return np.isscalar(value) or (isinstance(value, np.ndarray) and (len(np.squeeze(value).shape) == 0))

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

def arguments_as_dict(cls, *args, **kwargs):
        """
        Generate the arguments dictionary provided to :py:meth:`generate_name` and :py:meth:`calculate_total_steps`.

        This makes it possible to fetch arguments by name regardless of
        whether they were passed as positional or keyword arguments.  Unnamed
        positional arguments are provided as a tuple under the key ``pos``.
        """
        all_args = (None, ) + args
        return inspect.getcallargs(cls.run, *all_args, **kwargs)

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

def update(self, **kwargs):
    """Creates or updates a property for the instance for each parameter."""
    for key, value in kwargs.items():
      setattr(self, key, value)

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

def get_free_mb(folder):
    """ Return folder/drive free space (in bytes)
    """
    if platform.system() == 'Windows':
        free_bytes = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p(folder), None, None, ctypes.pointer(free_bytes))
        return free_bytes.value/1024/1024
    else:
        st = os.statvfs(folder)
        return st.f_bavail * st.f_frsize/1024/1024

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

def static_method(cls, f):
        """Decorator which dynamically binds static methods to the model for later use."""
        setattr(cls, f.__name__, staticmethod(f))
        return f

def count_string_diff(a,b):
    """Return the number of characters in two strings that don't exactly match"""
    shortest = min(len(a), len(b))
    return sum(a[i] != b[i] for i in range(shortest))

def min_or_none(val1, val2):
    """Returns min(val1, val2) returning None only if both values are None"""
    return min(val1, val2, key=lambda x: sys.maxint if x is None else x)

def execfile(fname, variables):
    """ This is builtin in python2, but we have to roll our own on py3. """
    with open(fname) as f:
        code = compile(f.read(), fname, 'exec')
        exec(code, variables)

def get_size(objects):
    """Compute the total size of all elements in objects."""
    res = 0
    for o in objects:
        try:
            res += _getsizeof(o)
        except AttributeError:
            print("IGNORING: type=%s; o=%s" % (str(type(o)), str(o)))
    return res

def sine_wave(i, frequency=FREQUENCY, framerate=FRAMERATE, amplitude=AMPLITUDE):
    """
    Returns value of a sine wave at a given frequency and framerate
    for a given sample i
    """
    omega = 2.0 * pi * float(frequency)
    sine = sin(omega * (float(i) / float(framerate)))
    return float(amplitude) * sine

def _cdf(self, xloc, dist, cache):
        """Cumulative distribution function."""
        return evaluation.evaluate_forward(dist, numpy.e**xloc, cache=cache)

def clean(self, text):
        """Remove all unwanted characters from text."""
        return ''.join([c for c in text if c in self.alphabet])

def manhattan(h1, h2): # # 7 us @array, 31 us @list \w 100 bins
    r"""
    Equal to Minowski distance with :math:`p=1`.
    
    See also
    --------
    minowski
    """
    h1, h2 = __prepare_histogram(h1, h2)
    return scipy.sum(scipy.absolute(h1 - h2))

def data_from_techshop_ws(tws_url):
    """Scrapes data from techshop.ws."""

    r = requests.get(tws_url)
    if r.status_code == 200:
        data = BeautifulSoup(r.text, "lxml")
    else:
        data = "There was an error while accessing data on techshop.ws."

    return data

def deskew(S):
    """Converts a skew-symmetric cross-product matrix to its corresponding
    vector. Only works for 3x3 matrices.

    Parameters
    ----------
    S : :obj:`numpy.ndarray` of float
        A 3x3 skew-symmetric matrix.

    Returns
    -------
    :obj:`numpy.ndarray` of float
        A 3-entry vector that corresponds to the given cross product matrix.
    """
    x = np.zeros(3)
    x[0] = S[2,1]
    x[1] = S[0,2]
    x[2] = S[1,0]
    return x

def text_remove_empty_lines(text):
    """
    Whitespace normalization:

      - Strip empty lines
      - Strip trailing whitespace
    """
    lines = [ line.rstrip()  for line in text.splitlines()  if line.strip() ]
    return "\n".join(lines)

def average_gradient(data, *kwargs):
    """ Compute average gradient norm of an image
    """
    return np.average(np.array(np.gradient(data))**2)

def unzip_file_to_dir(path_to_zip, output_directory):
    """
    Extract a ZIP archive to a directory
    """
    z = ZipFile(path_to_zip, 'r')
    z.extractall(output_directory)
    z.close()

def get_distance_matrix(x):
    """Get distance matrix given a matrix. Used in testing."""
    square = nd.sum(x ** 2.0, axis=1, keepdims=True)
    distance_square = square + square.transpose() - (2.0 * nd.dot(x, x.transpose()))
    return nd.sqrt(distance_square)

def _mean_absolute_error(y, y_pred, w):
    """Calculate the mean absolute error."""
    return np.average(np.abs(y_pred - y), weights=w)

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

def softmax(xs):
    """Stable implementation of the softmax function."""
    ys = xs - np.max(xs)
    exps = np.exp(ys)
    return exps / exps.sum(axis=0)

def timestamp_filename(basename, ext=None):
    """
    Return a string of the form [basename-TIMESTAMP.ext]
    where TIMESTAMP is of the form YYYYMMDD-HHMMSS-MILSEC
    """
    dt = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    if ext:
        return '%s-%s.%s' % (basename, dt, ext)
    return '%s-%s' % (basename, dt)

def random_id(length):
    """Generates a random ID of given length"""

    def char():
        """Generate single random char"""

        return random.choice(string.ascii_letters + string.digits)

    return "".join(char() for _ in range(length))

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

def zeros(self, name, **kwargs):
        """Create an array. Keyword arguments as per
        :func:`zarr.creation.zeros`."""
        return self._write_op(self._zeros_nosync, name, **kwargs)

def auth_request(self, url, headers, body):
        """Perform auth request for token."""

        return self.req.post(url, headers, body=body)

def get_incomplete_path(filename):
  """Returns a temporary filename based on filename."""
  random_suffix = "".join(
      random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
  return filename + ".incomplete" + random_suffix

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

def generate_random_id(size=6, chars=string.ascii_uppercase + string.digits):
    """Generate random id numbers."""
    return "".join(random.choice(chars) for x in range(size))

def nest(thing):
    """Use tensorflows nest function if available, otherwise just wrap object in an array"""
    tfutil = util.get_module('tensorflow.python.util')
    if tfutil:
        return tfutil.nest.flatten(thing)
    else:
        return [thing]

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

def sine_wave(frequency):
  """Emit a sine wave at the given frequency."""
  xs = tf.reshape(tf.range(_samples(), dtype=tf.float32), [1, _samples(), 1])
  ts = xs / FLAGS.sample_rate
  return tf.sin(2 * math.pi * frequency * ts)

def text_response(self, contents, code=200, headers={}):
        """shortcut to return simple plain/text messages in the response.

        :param contents: a string with the response contents
        :param code: the http status code
        :param headers: a dict with optional headers
        :returns: a :py:class:`flask.Response` with the ``text/plain`` **Content-Type** header.
        """
        return Response(contents, status=code, headers={
            'Content-Type': 'text/plain'
        })

def make_name(estimator):
  """Helper function that returns the name of estimator or the given string
  if a string is given
  """
  if estimator is not None:
    if isinstance(estimator, six.string_types):
      estimator_name = estimator
    else:
      estimator_name = estimator.__class__.__name__
  else:
    estimator_name = None
  return estimator_name

def OnContextMenu(self, event):
        """Context menu event handler"""

        self.grid.PopupMenu(self.grid.contextmenu)

        event.Skip()

def list_files(directory):
    """Returns all files in a given directory
    """
    return [f for f in pathlib.Path(directory).iterdir() if f.is_file() and not f.name.startswith('.')]

def OnMove(self, event):
        """Main window move event"""

        # Store window position in config
        position = self.main_window.GetScreenPositionTuple()

        config["window_position"] = repr(position)

def list_files(directory):
    """Returns all files in a given directory
    """
    return [f for f in pathlib.Path(directory).iterdir() if f.is_file() and not f.name.startswith('.')]

def filter_query_string(query):
    """
        Return a version of the query string with the _e, _k and _s values
        removed.
    """
    return '&'.join([q for q in query.split('&')
        if not (q.startswith('_k=') or q.startswith('_e=') or q.startswith('_s'))])

def unique_items(seq):
    """Return the unique items from iterable *seq* (in order)."""
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]

def adjust(cols, light):
    """Create palette."""
    raw_colors = [cols[0], *cols, "#FFFFFF",
                  "#000000", *cols, "#FFFFFF"]

    return colors.generic_adjust(raw_colors, light)

def median_date(dt_list):
    """Calcuate median datetime from datetime list
    """
    #dt_list_sort = sorted(dt_list)
    idx = len(dt_list)/2
    if len(dt_list) % 2 == 0:
        md = mean_date([dt_list[idx-1], dt_list[idx]])
    else:
        md = dt_list[idx]
    return md

def RadiusGrid(gridSize):
    """
    Return a square grid with values of the distance from the centre 
    of the grid to each gridpoint
    """
    x,y=np.mgrid[0:gridSize,0:gridSize]
    x = x-(gridSize-1.0)/2.0
    y = y-(gridSize-1.0)/2.0
    return np.abs(x+1j*y)

def __absolute__(self, uri):
        """ Get the absolute uri for a file

        :param uri: URI of the resource to be retrieved
        :return: Absolute Path
        """
        return op.abspath(op.join(self.__path__, uri))

def autocorr_coeff(x, t, tau1, tau2):
    """Calculate the autocorrelation coefficient."""
    return corr_coeff(x, x, t, tau1, tau2)

def url(self):
        """ The url of this window """
        with switch_window(self._browser, self.name):
            return self._browser.url

def indentsize(line):
    """Return the indent size, in spaces, at the start of a line of text."""
    expline = string.expandtabs(line)
    return len(expline) - len(string.lstrip(expline))

def as_list(self):
        """Return all child objects in nested lists of strings."""
        return [self.name, self.value, [x.as_list for x in self.children]]

def paren_change(inputstring, opens=opens, closes=closes):
    """Determine the parenthetical change of level (num closes - num opens)."""
    count = 0
    for c in inputstring:
        if c in opens:  # open parens/brackets/braces
            count -= 1
        elif c in closes:  # close parens/brackets/braces
            count += 1
    return count

def get_month_start_end_day():
    """
    Get the month start date a nd end date
    """
    t = date.today()
    n = mdays[t.month]
    return (date(t.year, t.month, 1), date(t.year, t.month, n))

def count(data, axis=None):
    """Count the number of non-NA in this array along the given axis or axes
    """
    return np.sum(np.logical_not(isnull(data)), axis=axis)

def circ_permutation(items):
    """Calculate the circular permutation for a given list of items."""
    permutations = []
    for i in range(len(items)):
        permutations.append(items[i:] + items[:i])
    return permutations

def string_to_list(string, sep=",", filter_empty=False):
    """Transforma una string con elementos separados por `sep` en una lista."""
    return [value.strip() for value in string.split(sep)
            if (not filter_empty or value)]

def data_directory():
    """Return the absolute path to the directory containing the package data."""
    package_directory = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(package_directory, "data")

def go_to_new_line(self):
        """Go to the end of the current line and create a new line"""
        self.stdkey_end(False, False)
        self.insert_text(self.get_line_separator())

def calc_volume(self, sample: np.ndarray):
        """Find the RMS of the audio"""
        return sqrt(np.mean(np.square(sample)))

def make_unique_ngrams(s, n):
    """Make a set of unique n-grams from a string."""
    return set(s[i:i + n] for i in range(len(s) - n + 1))

def read_string(buff, byteorder='big'):
    """Read a string from a file-like object."""
    length = read_numeric(USHORT, buff, byteorder)
    return buff.read(length).decode('utf-8')

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

def extract_module_locals(depth=0):
    """Returns (module, locals) of the funciton `depth` frames away from the caller"""
    f = sys._getframe(depth + 1)
    global_ns = f.f_globals
    module = sys.modules[global_ns['__name__']]
    return (module, f.f_locals)

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

def get_size(self):
        """see doc in Term class"""
        self.curses.setupterm()
        return self.curses.tigetnum('cols'), self.curses.tigetnum('lines')

def ensure_dir_exists(directory):
    """Se asegura de que un directorio exista."""
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

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

def get_system_cpu_times():
    """Return system CPU times as a namedtuple."""
    user, nice, system, idle = _psutil_osx.get_system_cpu_times()
    return _cputimes_ntuple(user, nice, system, idle)

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

def get_current_branch():
    """
    Return the current branch
    """
    cmd = ["git", "rev-parse", "--abbrev-ref", "HEAD"]
    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    return output.strip().decode("utf-8")

def from_dict(cls, d):
        """Create an instance from a dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.ENTRIES})

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

def colorbar(height, length, colormap):
    """Return the channels of a colorbar.
    """
    cbar = np.tile(np.arange(length) * 1.0 / (length - 1), (height, 1))
    cbar = (cbar * (colormap.values.max() - colormap.values.min())
            + colormap.values.min())

    return colormap.colorize(cbar)

def _read_date_from_string(str1):
    """
    Reads the date from a string in the format YYYY/MM/DD and returns
    :class: datetime.date
    """
    full_date = [int(x) for x in str1.split('/')]
    return datetime.date(full_date[0], full_date[1], full_date[2])

def chunk_list(l, n):
    """Return `n` size lists from a given list `l`"""
    return [l[i:i + n] for i in range(0, len(l), n)]

def _get_config_or_default(self, key, default, as_type=lambda x: x):
        """Return a main config value, or default if it does not exist."""

        if self.main_config.has_option(self.main_section, key):
            return as_type(self.main_config.get(self.main_section, key))
        return default

def bytes_to_c_array(data):
    """
    Make a C array using the given string.
    """
    chars = [
        "'{}'".format(encode_escape(i))
        for i in decode_escape(data)
    ]
    return ', '.join(chars) + ', 0'

def disable_stdout_buffering():
    """This turns off stdout buffering so that outputs are immediately
    materialized and log messages show up before the program exits"""
    stdout_orig = sys.stdout
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
    # NOTE(brandyn): This removes the original stdout
    return stdout_orig

def pointer(self):
        """Get a ctypes void pointer to the memory mapped region.

        :type: ctypes.c_void_p
        """
        return ctypes.cast(ctypes.pointer(ctypes.c_uint8.from_buffer(self.mapping, 0)), ctypes.c_void_p)

def get_list_dimensions(_list):
    """
    Takes a nested list and returns the size of each dimension followed
    by the element type in the list
    """
    if isinstance(_list, list) or isinstance(_list, tuple):
        return [len(_list)] + get_list_dimensions(_list[0])
    return []

def load(self, name):
        """Loads and returns foreign library."""
        name = ctypes.util.find_library(name)
        return ctypes.cdll.LoadLibrary(name)

def horz_dpi(self):
        """
        Integer dots per inch for the width of this image. Defaults to 72
        when not present in the file, as is often the case.
        """
        pHYs = self._chunks.pHYs
        if pHYs is None:
            return 72
        return self._dpi(pHYs.units_specifier, pHYs.horz_px_per_unit)

def get_distance_matrix(x):
    """Get distance matrix given a matrix. Used in testing."""
    square = nd.sum(x ** 2.0, axis=1, keepdims=True)
    distance_square = square + square.transpose() - (2.0 * nd.dot(x, x.transpose()))
    return nd.sqrt(distance_square)

def newest_file(file_iterable):
  """
  Returns the name of the newest file given an iterable of file names.

  """
  return max(file_iterable, key=lambda fname: os.path.getmtime(fname))

def plot_target(target, ax):
    """Ajoute la target au plot"""
    ax.scatter(target[0], target[1], target[2], c="red", s=80)

def remove_ext(fname):
    """Removes the extension from a filename
    """
    bn = os.path.basename(fname)
    return os.path.splitext(bn)[0]

def _crop_list_to_size(l, size):
    """Make a list a certain size"""
    for x in range(size - len(l)):
        l.append(False)
    for x in range(len(l) - size):
        l.pop()
    return l

def get_month_start(day=None):
    """Returns the first day of the given month."""
    day = add_timezone(day or datetime.date.today())
    return day.replace(day=1)

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

def qualified_name_import(cls):
    """Full name of a class, including the module. Like qualified_class_name, but when you already have a class """

    parts = qualified_name(cls).split('.')

    return "from {} import {}".format('.'.join(parts[:-1]), parts[-1])

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

def unique(list):
    """ Returns a copy of the list without duplicates.
    """
    unique = []; [unique.append(x) for x in list if x not in unique]
    return unique

def server(self):
        """Returns the size of remote files
        """
        try:
            tar = urllib2.urlopen(self.registry)
            meta = tar.info()
            return int(meta.getheaders("Content-Length")[0])
        except (urllib2.URLError, IndexError):
            return " "

def cric__decision_tree():
    """ Decision Tree
    """
    model = sklearn.tree.DecisionTreeClassifier(random_state=0, max_depth=4)

    # we want to explain the raw probability outputs of the trees
    model.predict = lambda X: model.predict_proba(X)[:,1]
    
    return model

def index(self, elem):
        """Find the index of elem in the reversed iterator."""
        return _coconut.len(self._iter) - self._iter.index(elem) - 1

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

def load_object_by_name(object_name):
    """Load an object from a module by name"""
    mod_name, attr = object_name.rsplit('.', 1)
    mod = import_module(mod_name)
    return getattr(mod, attr)

def __eq__(self, other):
        """Determine if two objects are equal."""
        return isinstance(other, self.__class__) \
            and self._freeze() == other._freeze()

def __getitem__(self, index):
    """Get the item at the given index.

    Index is a tuple of (row, col)
    """
    row, col = index
    return self.rows[row][col]

def _release(self):
        """Destroy self since closures cannot be called again."""
        del self.funcs
        del self.variables
        del self.variable_values
        del self.satisfied

def itervalues(d, **kw):
    """Return an iterator over the values of a dictionary."""
    if not PY2:
        return iter(d.values(**kw))
    return d.itervalues(**kw)

def safe_rmtree(directory):
  """Delete a directory if it's present. If it's not present, no-op."""
  if os.path.exists(directory):
    shutil.rmtree(directory, True)

def del_label(self, name):
        """Delete a label by name."""
        labels_tag = self.root[0]
        labels_tag.remove(self._find_label(name))

def get_properties(cls):
        """Get all properties of the MessageFlags class."""
        property_names = [p for p in dir(cls)
                          if isinstance(getattr(cls, p), property)]
        return property_names

def cleanup(self):
        """Clean up any temporary files."""
        for file in glob.glob(self.basename + '*'):
            os.unlink(file)

def get_coordinates_by_full_name(self, name):
        """Retrieves a person's coordinates by full name"""
        person = self.get_person_by_full_name(name)
        if not person:
            return '', ''
        return person.latitude, person.longitude

def remove(self, key):
        """remove the value found at key from the queue"""
        item = self.item_finder.pop(key)
        item[-1] = None
        self.removed_count += 1

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

def unique(seq):
    """Return the unique elements of a collection even if those elements are
       unhashable and unsortable, like dicts and sets"""
    cleaned = []
    for each in seq:
        if each not in cleaned:
            cleaned.append(each)
    return cleaned

def get_model(name):
    """
    Convert a model's verbose name to the model class. This allows us to
    use the models verbose name in steps.
    """

    model = MODELS.get(name.lower(), None)

    assert model, "Could not locate model by name '%s'" % name

    return model

def check_precomputed_distance_matrix(X):
    """Perform check_array(X) after removing infinite values (numpy.inf) from the given distance matrix.
    """
    tmp = X.copy()
    tmp[np.isinf(tmp)] = 1
    check_array(tmp)

def pause(self):
        """Pause the music"""
        mixer.music.pause()
        self.pause_time = self.get_time()
        self.paused = True

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

def typename(obj):
    """Returns the type of obj as a string. More descriptive and specific than
    type(obj), and safe for any object, unlike __class__."""
    if hasattr(obj, '__class__'):
        return getattr(obj, '__class__').__name__
    else:
        return type(obj).__name__

def datatype(dbtype, description, cursor):
    """Google AppEngine Helper to convert a data type into a string."""
    dt = cursor.db.introspection.get_field_type(dbtype, description)
    if type(dt) is tuple:
        return dt[0]
    else:
        return dt

def get_month_start_date(self):
        """Returns the first day of the current month"""
        now = timezone.now()
        return timezone.datetime(day=1, month=now.month, year=now.year, tzinfo=now.tzinfo)

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

def get_parent_dir(name):
    """Get the parent directory of a filename."""
    parent_dir = os.path.dirname(os.path.dirname(name))
    if parent_dir:
        return parent_dir
    return os.path.abspath('.')

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

def get_property_by_name(pif, name):
    """Get a property by name"""
    return next((x for x in pif.properties if x.name == name), None)

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

def get_propety_by_name(pif, name):
    """Get a property by name"""
    warn("This method has been deprecated in favor of get_property_by_name")
    return next((x for x in pif.properties if x.name == name), None)

def dict_to_html_attrs(dict_):
    """
    Banana banana
    """
    res = ' '.join('%s="%s"' % (k, v) for k, v in dict_.items())
    return res

def get(self, key):  
        """ get a set of keys from redis """
        res = self.connection.get(key)
        print(res)
        return res

def update(self, params):
        """Update the dev_info data from a dictionary.

        Only updates if it already exists in the device.
        """
        dev_info = self.json_state.get('deviceInfo')
        dev_info.update({k: params[k] for k in params if dev_info.get(k)})

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

def compare_dict(da, db):
    """
    Compare differencs from two dicts
    """
    sa = set(da.items())
    sb = set(db.items())
    
    diff = sa & sb
    return dict(sa - diff), dict(sb - diff)

def dir_modtime(dpath):
    """
    Returns the latest modification time of all files/subdirectories in a
    directory
    """
    return max(os.path.getmtime(d) for d, _, _ in os.walk(dpath))

def _increase_file_handle_limit():
    """Raise the open file handles permitted by the Dusty daemon process
    and its child processes. The number we choose here needs to be within
    the OS X default kernel hard limit, which is 10240."""
    logging.info('Increasing file handle limit to {}'.format(constants.FILE_HANDLE_LIMIT))
    resource.setrlimit(resource.RLIMIT_NOFILE,
                       (constants.FILE_HANDLE_LIMIT, resource.RLIM_INFINITY))

def count_rows(self, table_name):
        """Return the number of entries in a table by counting them."""
        self.table_must_exist(table_name)
        query = "SELECT COUNT (*) FROM `%s`" % table_name.lower()
        self.own_cursor.execute(query)
        return int(self.own_cursor.fetchone()[0])

def get_user_by_id(self, id):
        """Retrieve a User object by ID."""
        return self.db_adapter.get_object(self.UserClass, id=id)

def datetime_delta_to_ms(delta):
    """
    Given a datetime.timedelta object, return the delta in milliseconds
    """
    delta_ms = delta.days * 24 * 60 * 60 * 1000
    delta_ms += delta.seconds * 1000
    delta_ms += delta.microseconds / 1000
    delta_ms = int(delta_ms)
    return delta_ms

def url_to_image(url):
    """
    Fetch an image from url and convert it into a Pillow Image object
    """
    r = requests.get(url)
    image = StringIO(r.content)
    return image

def dt2ts(dt):
    """Converts to float representing number of seconds since 1970-01-01 GMT."""
    # Note: no assertion to really keep this fast
    assert isinstance(dt, (datetime.datetime, datetime.date))
    ret = time.mktime(dt.timetuple())
    if isinstance(dt, datetime.datetime):
        ret += 1e-6 * dt.microsecond
    return ret

def Print(self):
        """Prints the values and freqs/probs in ascending order."""
        for val, prob in sorted(self.d.iteritems()):
            print(val, prob)

def ver_to_tuple(value):
    """
    Convert version like string to a tuple of integers.
    """
    return tuple(int(_f) for _f in re.split(r'\D+', value) if _f)

def get_plain_image_as_widget(self):
        """Used for generating thumbnails.  Does not include overlaid
        graphics.
        """
        arr = self.getwin_array(order=self.rgb_order)
        image = self._get_qimage(arr, self.qimg_fmt)
        return image

def generate_unique_host_id():
    """Generate a unique ID, that is somewhat guaranteed to be unique among all
    instances running at the same time."""
    host = ".".join(reversed(socket.gethostname().split(".")))
    pid = os.getpid()
    return "%s.%d" % (host, pid)

def vector_distance(a, b):
    """The Euclidean distance between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

def get_user_by_id(self, id):
        """Retrieve a User object by ID."""
        return self.db_adapter.get_object(self.UserClass, id=id)

def parse_env_var(s):
    """Parse an environment variable string

    Returns a key-value tuple

    Apply the same logic as `docker run -e`:
    "If the operator names an environment variable without specifying a value,
    then the current value of the named variable is propagated into the
    container's environment
    """
    parts = s.split('=', 1)
    if len(parts) == 2:
        k, v = parts
        return (k, v)

    k = parts[0]
    return (k, os.getenv(k, ''))

def get_user_id_from_email(self, email):
        """ Uses the get-all-user-accounts Portals API to retrieve the
        user-id by supplying an email. """
        accts = self.get_all_user_accounts()

        for acct in accts:
            if acct['email'] == email:
                return acct['id']
        return None

def check_create_folder(filename):
    """Check if the folder exisits. If not, create the folder"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

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

def parse_comments_for_file(filename):
    """
    Return a list of all parsed comments in a file.  Mostly for testing &
    interactive use.
    """
    return [parse_comment(strip_stars(comment), next_line)
            for comment, next_line in get_doc_comments(read_file(filename))]

def get_inputs_from_cm(index, cm):
    """Return indices of inputs to the node with the given index."""
    return tuple(i for i in range(cm.shape[0]) if cm[i][index])

def raw_print(*args, **kw):
    """Raw print to sys.__stdout__, otherwise identical interface to print()."""

    print(*args, sep=kw.get('sep', ' '), end=kw.get('end', '\n'),
          file=sys.__stdout__)
    sys.__stdout__.flush()

def process_bool_arg(arg):
    """ Determine True/False from argument """
    if isinstance(arg, bool):
        return arg
    elif isinstance(arg, basestring):
        if arg.lower() in ["true", "1"]:
            return True
        elif arg.lower() in ["false", "0"]:
            return False

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

def getAttributeData(self, name, channel=None):
        """ Returns a attribut """
        return self._getNodeData(name, self._ATTRIBUTENODE, channel)

def dot(self, w):
        """Return the dotproduct between self and another vector."""

        return sum([x * y for x, y in zip(self, w)])

def __getLogger(cls):
    """ Get the logger for this object.

    :returns: (Logger) A Logger object.
    """
    if cls.__logger is None:
      cls.__logger = opf_utils.initLogger(cls)
    return cls.__logger

def depgraph_to_dotsrc(dep_graph, show_cycles, nodot, reverse):
    """Convert the dependency graph (DepGraph class) to dot source code.
    """
    if show_cycles:
        dotsrc = cycles2dot(dep_graph, reverse=reverse)
    elif not nodot:
        dotsrc = dep2dot(dep_graph, reverse=reverse)
    else:
        dotsrc = None
    return dotsrc

def commit(self, message=None, amend=False, stage=True):
        """Commit any changes, optionally staging all changes beforehand."""
        return git_commit(self.repo_dir, message=message,
                          amend=amend, stage=stage)

def plot(self):
        """Plot the empirical histogram versus best-fit distribution's PDF."""
        plt.plot(self.bin_edges, self.hist, self.bin_edges, self.best_pdf)

def chmod_add_excute(filename):
        """
        Adds execute permission to file.
        :param filename:
        :return:
        """
        st = os.stat(filename)
        os.chmod(filename, st.st_mode | stat.S_IEXEC)

def hline(self, x, y, width, color):
        """Draw a horizontal line up to a given length."""
        self.rect(x, y, width, 1, color, fill=True)

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

def del_Unnamed(df):
    """
    Deletes all the unnamed columns

    :param df: pandas dataframe
    """
    cols_del=[c for c in df.columns if 'Unnamed' in c]
    return df.drop(cols_del,axis=1)

def int_to_date(date):
    """
    Convert an int of form yyyymmdd to a python date object.
    """

    year = date // 10**4
    month = date % 10**4 // 10**2
    day = date % 10**2

    return datetime.date(year, month, day)

def poke_array(self, store, name, elemtype, elements, container, visited, _stack):
        """abstract method"""
        raise NotImplementedError

def set_global(node: Node, key: str, value: Any):
    """Adds passed value to node's globals"""
    node.node_globals[key] = value

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

def remove_na_arraylike(arr):
    """
    Return array-like containing only true/non-NaN values, possibly empty.
    """
    if is_extension_array_dtype(arr):
        return arr[notna(arr)]
    else:
        return arr[notna(lib.values_from_object(arr))]

def group_by(iterable, key_func):
    """Wrap itertools.groupby to make life easier."""
    groups = (
        list(sub) for key, sub in groupby(iterable, key_func)
    )
    return zip(groups, groups)

def norm_slash(name):
    """Normalize path slashes."""

    if isinstance(name, str):
        return name.replace('/', "\\") if not is_case_sensitive() else name
    else:
        return name.replace(b'/', b"\\") if not is_case_sensitive() else name

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

def check_git():
    """Check if git command is available."""
    try:
        with open(os.devnull, "wb") as devnull:
            subprocess.check_call(["git", "--version"], stdout=devnull, stderr=devnull)
    except:
        raise RuntimeError("Please make sure git is installed and on your path.")

def remove_duplicates(lst):
    """
    Emulate what a Python ``set()`` does, but keeping the element's order.
    """
    dset = set()
    return [l for l in lst if l not in dset and not dset.add(l)]

def double_sha256(data):
    """A standard compound hash."""
    return bytes_as_revhex(hashlib.sha256(hashlib.sha256(data).digest()).digest())

def _encode_bool(name, value, dummy0, dummy1):
    """Encode a python boolean (True/False)."""
    return b"\x08" + name + (value and b"\x01" or b"\x00")

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

def Value(self, name):
    """Returns the value coresponding to the given enum name."""
    if name in self._enum_type.values_by_name:
      return self._enum_type.values_by_name[name].number
    raise ValueError('Enum %s has no value defined for name %s' % (
        self._enum_type.name, name))

def async_comp_check(self, original, loc, tokens):
        """Check for Python 3.6 async comprehension."""
        return self.check_py("36", "async comprehension", original, loc, tokens)

def _size_36():
    """ returns the rows, columns of terminal """
    from shutil import get_terminal_size
    dim = get_terminal_size()
    if isinstance(dim, list):
        return dim[0], dim[1]
    return dim.lines, dim.columns

def __str__(self):
        """Executes self.function to convert LazyString instance to a real
        str."""
        if not hasattr(self, '_str'):
            self._str=self.function(*self.args, **self.kwargs)
        return self._str

def average_gradient(data, *kwargs):
    """ Compute average gradient norm of an image
    """
    return np.average(np.array(np.gradient(data))**2)

def merge(left, right, how='inner', key=None, left_key=None, right_key=None,
          left_as='left', right_as='right'):
    """ Performs a join using the union join function. """
    return join(left, right, how, key, left_key, right_key,
                join_fn=make_union_join(left_as, right_as))

def _histplot_op(ax, data, **kwargs):
    """Add a histogram for the data to the axes."""
    bins = get_bins(data)
    ax.hist(data, bins=bins, align="left", density=True, **kwargs)
    return ax

def cover(session):
    """Run the final coverage report.
    This outputs the coverage report aggregating coverage from the unit
    test runs (not system test runs), and then erases coverage data.
    """
    session.interpreter = 'python3.6'
    session.install('coverage', 'pytest-cov')
    session.run('coverage', 'report', '--show-missing', '--fail-under=100')
    session.run('coverage', 'erase')

def safe_rmtree(directory):
  """Delete a directory if it's present. If it's not present, no-op."""
  if os.path.exists(directory):
    shutil.rmtree(directory, True)

def fail(message=None, exit_status=None):
    """Prints the specified message and exits the program with the specified
    exit status.

    """
    print('Error:', message, file=sys.stderr)
    sys.exit(exit_status or 1)

def get_file_string(filepath):
    """Get string from file."""
    with open(os.path.abspath(filepath)) as f:
        return f.read()

def _from_list_dict(cls, list_dic):
        """Takes a list of dict like objects and uses `champ_id` field as Id"""
        return cls({_convert_id(dic[cls.CHAMP_ID]): dict(dic) for dic in list_dic})

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

def get_filesize(self, pdf):
        """Compute the filesize of the PDF
        """
        try:
            filesize = float(pdf.get_size())
            return filesize / 1024
        except (POSKeyError, TypeError):
            return 0

def _get_os_environ_dict(keys):
  """Return a dictionary of key/values from os.environ."""
  return {k: os.environ.get(k, _UNDEFINED) for k in keys}

def parse(filename):
    """Parse ASDL from the given file and return a Module node describing it."""
    with open(filename) as f:
        parser = ASDLParser()
        return parser.parse(f.read())

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

def send_file(self, local_path, remote_path, user='root', unix_mode=None):
        """Upload a local file on the remote host.
        """
        self.enable_user(user)
        return self.ssh_pool.send_file(user, local_path, remote_path, unix_mode=unix_mode)

def prepend_line(filepath, line):
    """Rewrite a file adding a line to its beginning.
    """
    with open(filepath) as f:
        lines = f.readlines()

    lines.insert(0, line)

    with open(filepath, 'w') as f:
        f.writelines(lines)

def _fill_array_from_list(the_list, the_array):
        """Fill an `array` from a `list`"""
        for i, val in enumerate(the_list):
            the_array[i] = val
        return the_array

def assign_to(self, obj):
    """Assign `x` and `y` to an object that has properties `x` and `y`."""
    obj.x = self.x
    obj.y = self.y

def BROADCAST_FILTER_NOT(func):
        """
        Composes the passed filters into an and-joined filter.
        """
        return lambda u, command, *args, **kwargs: not func(u, command, *args, **kwargs)

def clean_time(time_string):
    """Return a datetime from the Amazon-provided datetime string"""
    # Get a timezone-aware datetime object from the string
    time = dateutil.parser.parse(time_string)
    if not settings.USE_TZ:
        # If timezone support is not active, convert the time to UTC and
        # remove the timezone field
        time = time.astimezone(timezone.utc).replace(tzinfo=None)
    return time

def filter_list_by_indices(lst, indices):
    """Return a modified list containing only the indices indicated.

    Args:
        lst: Original list of values
        indices: List of indices to keep from the original list

    Returns:
        list: Filtered list of values

    """
    return [x for i, x in enumerate(lst) if i in indices]

def populate_obj(obj, attrs):
    """Populates an object's attributes using the provided dict
    """
    for k, v in attrs.iteritems():
        setattr(obj, k, v)

def filter_none(list_of_points):
    """
    
    :param list_of_points: 
    :return: list_of_points with None's removed
    """
    remove_elementnone = filter(lambda p: p is not None, list_of_points)
    remove_sublistnone = filter(lambda p: not contains_none(p), remove_elementnone)
    return list(remove_sublistnone)

def populate_obj(obj, attrs):
    """Populates an object's attributes using the provided dict
    """
    for k, v in attrs.iteritems():
        setattr(obj, k, v)

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

def underline(self, msg):
        """Underline the input"""
        return click.style(msg, underline=True) if self.colorize else msg

def registered_filters_list(self):
        """
        Return the list of registered filters (as a list of strings).

        The list **only** includes registered filters (**not** the predefined :program:`Jinja2` filters).

        """
        return [filter_name for filter_name in self.__jinja2_environment.filters.keys() if filter_name not in self.__jinja2_predefined_filters ]

def indent(self):
        """
        Begins an indented block. Must be used in a 'with' code block.
        All calls to the logger inside of the block will be indented.
        """
        blk = IndentBlock(self, self._indent)
        self._indent += 1
        return blk

def get_average_color(colors):
    """Calculate the average color from the list of colors, where each color
    is a 3-tuple of (r, g, b) values.
    """
    c = reduce(color_reducer, colors)
    total = len(colors)
    return tuple(v / total for v in c)

def is_valid_file(parser, arg):
    """Check if arg is a valid file that already exists on the file system."""
    arg = os.path.abspath(arg)
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg

def index_nearest(value, array):
    """
    expects a _n.array
    returns the global minimum of (value-array)^2
    """

    a = (array-value)**2
    return index(a.min(), a)

def user_exists(username):
    """Check if a user exists"""
    try:
        pwd.getpwnam(username)
        user_exists = True
    except KeyError:
        user_exists = False
    return user_exists

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

def findMax(arr):
    """
    in comparison to argrelmax() more simple and  reliable peak finder
    """
    out = np.zeros(shape=arr.shape, dtype=bool)
    _calcMax(arr, out)
    return out

def is_file(path):
    """Determine if a Path or string is a file on the file system."""
    try:
        return path.expanduser().absolute().is_file()
    except AttributeError:
        return os.path.isfile(os.path.abspath(os.path.expanduser(str(path))))

def get_member(thing_obj, member_string):
    """Get a member from an object by (string) name"""
    mems = {x[0]: x[1] for x in inspect.getmembers(thing_obj)}
    if member_string in mems:
        return mems[member_string]

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

def find_le(a, x):
    """Find rightmost value less than or equal to x."""
    i = bs.bisect_right(a, x)
    if i: return i - 1
    raise ValueError

def clear(self):
        """Remove all items."""
        self._fwdm.clear()
        self._invm.clear()
        self._sntl.nxt = self._sntl.prv = self._sntl

def count_string_diff(a,b):
    """Return the number of characters in two strings that don't exactly match"""
    shortest = min(len(a), len(b))
    return sum(a[i] != b[i] for i in range(shortest))

def isetdiff_flags(list1, list2):
    """
    move to util_iter
    """
    set2 = set(list2)
    return (item not in set2 for item in list1)

def get_months_apart(d1, d2):
    """
    Get amount of months between dates
    http://stackoverflow.com/a/4040338
    """

    return (d1.year - d2.year)*12 + d1.month - d2.month

def _interval_to_bound_points(array):
    """
    Helper function which returns an array
    with the Intervals' boundaries.
    """

    array_boundaries = np.array([x.left for x in array])
    array_boundaries = np.concatenate(
        (array_boundaries, np.array([array[-1].right])))

    return array_boundaries

def is_builtin_type(tp):
    """Checks if the given type is a builtin one.
    """
    return hasattr(__builtins__, tp.__name__) and tp is getattr(__builtins__, tp.__name__)

def from_bytes(cls, b):
		"""Create :class:`PNG` from raw bytes.
		
		:arg bytes b: The raw bytes of the PNG file.
		:rtype: :class:`PNG`
		"""
		im = cls()
		im.chunks = list(parse_chunks(b))
		im.init()
		return im

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

def glr_path_static():
    """Returns path to packaged static files"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '_static'))

def _increment(self, *args):
        """Move the slider only by increment given by resolution."""
        value = self._var.get()
        if self._resolution:
            value = self._start + int(round((value - self._start) / self._resolution)) * self._resolution
            self._var.set(value)
        self.display_value(value)

def clear(self):
        """Clear the displayed image."""
        self._imgobj = None
        try:
            # See if there is an image on the canvas
            self.canvas.delete_object_by_tag(self._canvas_img_tag)
            self.redraw()
        except KeyError:
            pass

def render_template(env, filename, values=None):
    """
    Render a jinja template
    """
    if not values:
        values = {}
    tmpl = env.get_template(filename)
    return tmpl.render(values)

def click_estimate_slope():
    """
    Takes two clicks and returns the slope.

    Right-click aborts.
    """

    c1 = _pylab.ginput()
    if len(c1)==0:
        return None

    c2 = _pylab.ginput()
    if len(c2)==0:
        return None

    return (c1[0][1]-c2[0][1])/(c1[0][0]-c2[0][0])

def make_2d(ary):
    """Convert any array into a 2d numpy array.

    In case the array is already more than 2 dimensional, will ravel the
    dimensions after the first.
    """
    dim_0, *_ = np.atleast_1d(ary).shape
    return ary.reshape(dim_0, -1, order="F")

def flatten(l):
    """Flatten a nested list."""
    return sum(map(flatten, l), []) \
        if isinstance(l, list) or isinstance(l, tuple) else [l]

def is_executable(path):
  """Returns whether a path names an existing executable file."""
  return os.path.isfile(path) and os.access(path, os.X_OK)

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

def conv_dict(self):
        """dictionary of conversion"""
        return dict(integer=self.integer, real=self.real, no_type=self.no_type)

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

def run (self):
        """Handle keyboard interrupt and other errors."""
        try:
            self.run_checked()
        except KeyboardInterrupt:
            thread.interrupt_main()
        except Exception:
            self.internal_error()

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

def correspond(text):
    """Communicate with the child process without closing stdin."""
    subproc.stdin.write(text)
    subproc.stdin.flush()
    return drain()

def wait_until_exit(self):
        """ Wait until all the threads are finished.

        """
        [t.join() for t in self.threads]

        self.threads = list()

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

def cleanup_storage(*args):
    """Clean up processes after SIGTERM or SIGINT is received."""
    ShardedClusters().cleanup()
    ReplicaSets().cleanup()
    Servers().cleanup()
    sys.exit(0)

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

def exception_format():
    """
    Convert exception info into a string suitable for display.
    """
    return "".join(traceback.format_exception(
        sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2]
    ))

def ttl(self):
        """how long you should cache results for cacheable queries"""
        ret = 3600
        cn = self.get_process()
        if "ttl" in cn:
            ret = cn["ttl"]
        return ret

def version_triple(tag):
    """
    returns: a triple of integers from a version tag
    """
    groups = re.match(r'v?(\d+)\.(\d+)\.(\d+)', tag).groups()
    return tuple(int(n) for n in groups)

def move_back(self, dt):
        """ If called after an update, the sprite can move back
        """
        self._position = self._old_position
        self.rect.topleft = self._position
        self.feet.midbottom = self.rect.midbottom

def count(lines):
  """ Counts the word frequences in a list of sentences.

  Note:
    This is a helper function for parallel execution of `Vocabulary.from_text`
    method.
  """
  words = [w for l in lines for w in l.strip().split()]
  return Counter(words)

def is_empty(self):
        """Checks for an empty image.
        """
        if(((self.channels == []) and (not self.shape == (0, 0))) or
           ((not self.channels == []) and (self.shape == (0, 0)))):
            raise RuntimeError("Channels-shape mismatch.")
        return self.channels == [] and self.shape == (0, 0)

def from_rectangle(box):
        """ Create a vector randomly within the given rectangle. """
        x = box.left + box.width * random.uniform(0, 1)
        y = box.bottom + box.height * random.uniform(0, 1)
        return Vector(x, y)

def update_not_existing_kwargs(to_update, update_from):
    """
    This function updates the keyword aguments from update_from in
    to_update, only if the keys are not set in to_update.

    This is used for updated kwargs from the default dicts.
    """
    if to_update is None:
        to_update = {}
    to_update.update({k:v for k,v in update_from.items() if k not in to_update})
    return to_update

def get_key_by_value(dictionary, search_value):
    """
    searchs a value in a dicionary and returns the key of the first occurrence

    :param dictionary: dictionary to search in
    :param search_value: value to search for
    """
    for key, value in dictionary.iteritems():
        if value == search_value:
            return ugettext(key)

def h5ToDict(h5, readH5pyDataset=True):
    """ Read a hdf5 file into a dictionary """
    h = h5py.File(h5, "r")
    ret = unwrapArray(h, recursive=True, readH5pyDataset=readH5pyDataset)
    if readH5pyDataset: h.close()
    return ret

def insert_slash(string, every=2):
    """insert_slash insert / every 2 char"""
    return os.path.join(string[i:i+every] for i in xrange(0, len(string), every))

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

def gauss_pdf(x, mu, sigma):
    """Normalized Gaussian"""
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - mu) ** 2 / 2. / sigma ** 2)

def __copy__(self):
        """A magic method to implement shallow copy behavior."""
        return self.__class__.load(self.dump(), context=self.context)

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

def combinations(l):
    """Pure-Python implementation of itertools.combinations(l, 2)."""
    result = []
    for x in xrange(len(l) - 1):
        ls = l[x + 1:]
        for y in ls:
            result.append((l[x], y))
    return result

def _get_history_next(self):
        """ callback function for key down """
        if self._has_history:
            ret = self._input_history.return_history(1)
            self.string = ret
            self._curs_pos = len(ret)

def random_id(length):
    """Generates a random ID of given length"""

    def char():
        """Generate single random char"""

        return random.choice(string.ascii_letters + string.digits)

    return "".join(char() for _ in range(length))

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

def rlognormal(mu, tau, size=None):
    """
    Return random lognormal variates.
    """

    return np.random.lognormal(mu, np.sqrt(1. / tau), size)

def set_user_password(environment, parameter, password):
    """
    Sets a user's password in the keyring storage
    """
    username = '%s:%s' % (environment, parameter)
    return password_set(username, password)

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

def get_absolute_path(*args):
    """Transform relative pathnames into absolute pathnames."""
    directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(directory, *args)

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

def get_methods(*objs):
    """ Return the names of all callable attributes of an object"""
    return set(
        attr
        for obj in objs
        for attr in dir(obj)
        if not attr.startswith('_') and callable(getattr(obj, attr))
    )

def get_iter_string_reader(stdin):
    """ return an iterator that returns a chunk of a string every time it is
    called.  notice that even though bufsize_type might be line buffered, we're
    not doing any line buffering here.  that's because our StreamBufferer
    handles all buffering.  we just need to return a reasonable-sized chunk. """
    bufsize = 1024
    iter_str = (stdin[i:i + bufsize] for i in range(0, len(stdin), bufsize))
    return get_iter_chunk_reader(iter_str)

def dict_from_object(obj: object):
    """Convert a object into dictionary with all of its readable attributes."""

    # If object is a dict instance, no need to convert.
    return (obj if isinstance(obj, dict)
            else {attr: getattr(obj, attr)
                  for attr in dir(obj) if not attr.startswith('_')})

def xmltreefromfile(filename):
    """Internal function to read an XML file"""
    try:
        return ElementTree.parse(filename, ElementTree.XMLParser(collect_ids=False))
    except TypeError:
        return ElementTree.parse(filename, ElementTree.XMLParser())

def calc_list_average(l):
    """
    Calculates the average value of a list of numbers
    Returns a float
    """
    total = 0.0
    for value in l:
        total += value
    return total / len(l)

def redirect_output(fileobj):
    """Redirect standard out to file."""
    old = sys.stdout
    sys.stdout = fileobj
    try:
        yield fileobj
    finally:
        sys.stdout = old

def get(s, delimiter='', format="diacritical"):
    """Return pinyin of string, the string must be unicode
    """
    return delimiter.join(_pinyin_generator(u(s), format=format))

def remove_ext(fname):
    """Removes the extension from a filename
    """
    bn = os.path.basename(fname)
    return os.path.splitext(bn)[0]

def estimate_complexity(self, x,y,z,n):
        """ 
        calculates a rough guess of runtime based on product of parameters 
        """
        num_calculations = x * y * z * n
        run_time = num_calculations / 100000  # a 2014 PC does about 100k calcs in a second (guess based on prior logs)
        return self.show_time_as_short_string(run_time)

def do_rewind(self, line):
        """
        rewind
        """
        self.print_response("Rewinding from frame %s to 0" % self.bot._frame)
        self.bot._frame = 0

def get_geoip(ip):
    """Lookup country for IP address."""
    reader = geolite2.reader()
    ip_data = reader.get(ip) or {}
    return ip_data.get('country', {}).get('iso_code')

def _file_exists(path, filename):
  """Checks if the filename exists under the path."""
  return os.path.isfile(os.path.join(path, filename))

def unproject(self, xy):
        """
        Returns the coordinates from position in meters
        """
        (x, y) = xy
        lng = x/EARTH_RADIUS * RAD_TO_DEG
        lat = 2 * atan(exp(y/EARTH_RADIUS)) - pi/2 * RAD_TO_DEG
        return (lng, lat)

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

def _cosine(a, b):
    """ Return the len(a & b) / len(a) """
    return 1. * len(a & b) / (math.sqrt(len(a)) * math.sqrt(len(b)))

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

def get_current_frames():
    """Return current threads prepared for 
    further processing.
    """
    return dict(
        (thread_id, {'frame': thread2list(frame), 'time': None})
        for thread_id, frame in sys._current_frames().items()
    )

def _replace_none(self, aDict):
        """ Replace all None values in a dict with 'none' """
        for k, v in aDict.items():
            if v is None:
                aDict[k] = 'none'

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

def __normalize_list(self, msg):
        """Split message to list by commas and trim whitespace."""
        if isinstance(msg, list):
            msg = "".join(msg)
        return list(map(lambda x: x.strip(), msg.split(",")))

def get_uniques(l):
    """ Returns a list with no repeated elements.
    """
    result = []

    for i in l:
        if i not in result:
            result.append(i)

    return result

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

def distinct(xs):
    """Get the list of distinct values with preserving order."""
    # don't use collections.OrderedDict because we do support Python 2.6
    seen = set()
    return [x for x in xs if x not in seen and not seen.add(x)]

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

def vector_distance(a, b):
    """The Euclidean distance between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

def extract_module_locals(depth=0):
    """Returns (module, locals) of the funciton `depth` frames away from the caller"""
    f = sys._getframe(depth + 1)
    global_ns = f.f_globals
    module = sys.modules[global_ns['__name__']]
    return (module, f.f_locals)

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

def printOut(value, end='\n'):
    """
    This function prints the given String immediately and flushes the output.
    """
    sys.stdout.write(value)
    sys.stdout.write(end)
    sys.stdout.flush()

def _get_item_position(self, idx):
        """Return a tuple of (start, end) indices of an item from its index."""
        start = 0 if idx == 0 else self._index[idx - 1] + 1
        end = self._index[idx]
        return start, end

def warn(self, text):
		""" Ajout d'un message de log de type WARN """
		self.logger.warn("{}{}".format(self.message_prefix, text))

def get_model_index_properties(instance, index):
    """Return the list of properties specified for a model in an index."""
    mapping = get_index_mapping(index)
    doc_type = instance._meta.model_name.lower()
    return list(mapping["mappings"][doc_type]["properties"].keys())

def copy(self):
        """Return a copy of this list with each element copied to new memory
        """
        out = type(self)()
        for series in self:
            out.append(series.copy())
        return out

def open_json(file_name):
    """
    returns json contents as string
    """
    with open(file_name, "r") as json_data:
        data = json.load(json_data)
        return data

def markdown_to_text(body):
    """Converts markdown to text.

    Args:
        body: markdown (or plaintext, or maybe HTML) input

    Returns:
        Plaintext with all tags and frills removed
    """
    # Turn our input into HTML
    md = markdown.markdown(body, extensions=[
        'markdown.extensions.extra'
    ])

    # Safely parse HTML so that we don't have to parse it ourselves
    soup = BeautifulSoup(md, 'html.parser')

    # Return just the text of the parsed HTML
    return soup.get_text()

def last_modified_date(filename):
    """Last modified timestamp as a UTC datetime"""
    mtime = os.path.getmtime(filename)
    dt = datetime.datetime.utcfromtimestamp(mtime)
    return dt.replace(tzinfo=pytz.utc)

def HttpResponse401(request, template=KEY_AUTH_401_TEMPLATE,
content=KEY_AUTH_401_CONTENT, content_type=KEY_AUTH_401_CONTENT_TYPE):
    """
    HTTP response for not-authorized access (status code 403)
    """
    return AccessFailedResponse(request, template, content, content_type, status=401)

def get_idx_rect(index_list):
    """Extract the boundaries from a list of indexes"""
    rows, cols = list(zip(*[(i.row(), i.column()) for i in index_list]))
    return ( min(rows), max(rows), min(cols), max(cols) )

def _getTypename(self, defn):
        """ Returns the SQL typename required to store the given FieldDefinition """
        return 'REAL' if defn.type.float or 'TIME' in defn.type.name or defn.dntoeu else 'INTEGER'

def objectproxy_realaddress(obj):
    """
    Obtain a real address as an integer from an objectproxy.
    """
    voidp = QROOT.TPython.ObjectProxy_AsVoidPtr(obj)
    return C.addressof(C.c_char.from_buffer(voidp))

def server(self):
        """Returns the size of remote files
        """
        try:
            tar = urllib2.urlopen(self.registry)
            meta = tar.info()
            return int(meta.getheaders("Content-Length")[0])
        except (urllib2.URLError, IndexError):
            return " "

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

def task_property_present_predicate(service, task, prop):
    """ True if the json_element passed is present for the task specified.
    """
    try:
        response = get_service_task(service, task)
    except Exception as e:
        pass

    return (response is not None) and (prop in response)

def GetMountpoints():
  """List all the filesystems mounted on the system."""
  devices = {}

  for filesys in GetFileSystems():
    devices[filesys.f_mntonname] = (filesys.f_mntfromname, filesys.f_fstypename)

  return devices

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

def _num_cpus_darwin():
    """Return the number of active CPUs on a Darwin system."""
    p = subprocess.Popen(['sysctl','-n','hw.ncpu'],stdout=subprocess.PIPE)
    return p.stdout.read()

def ensure_dir_exists(directory):
    """Se asegura de que un directorio exista."""
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def _GetValue(self, name):
    """Returns the TextFSMValue object natching the requested name."""
    for value in self.values:
      if value.name == name:
        return value

def issuperset(self, other):
        """Report whether this RangeSet contains another set."""
        self._binary_sanity_check(other)
        return set.issuperset(self, other)

def check_output(args):
    """Runs command and returns the output as string."""
    log.debug('run: %s', args)
    out = subprocess.check_output(args=args).decode('utf-8')
    log.debug('out: %r', out)
    return out

def _stdin_ready_posix():
    """Return True if there's something to read on stdin (posix version)."""
    infds, outfds, erfds = select.select([sys.stdin],[],[],0)
    return bool(infds)

def owner(self):
        """
        Username of document creator
        """
        if self._owner:
            return self._owner
        elif not self.abstract:
            return self.read_meta()._owner

        raise EmptyDocumentException()

def is_iterable(value):
    """must be an iterable (list, array, tuple)"""
    return isinstance(value, np.ndarray) or isinstance(value, list) or isinstance(value, tuple), value

def dict_from_object(obj: object):
    """Convert a object into dictionary with all of its readable attributes."""

    # If object is a dict instance, no need to convert.
    return (obj if isinstance(obj, dict)
            else {attr: getattr(obj, attr)
                  for attr in dir(obj) if not attr.startswith('_')})

def leaf_nodes(self):
        """
        Return an interable of nodes with no edges pointing at them. This is
        helpful to find all nodes without dependencies.
        """
        # Now contains all nodes that contain dependencies.
        deps = {item for sublist in self.edges.values() for item in sublist}
        # contains all nodes *without* any dependencies (leaf nodes)
        return self.nodes - deps

def unique(list):
    """ Returns a copy of the list without duplicates.
    """
    unique = []; [unique.append(x) for x in list if x not in unique]
    return unique

def check_color(cls, raw_image):
        """
        Just check if raw_image is completely white.
        http://stackoverflow.com/questions/14041562/python-pil-detect-if-an-image-is-completely-black-or-white
        """
        # sum(img.convert("L").getextrema()) in (0, 2)
        extrema = raw_image.convert("L").getextrema()
        if extrema == (255, 255): # all white
            raise cls.MonoImageException

def axes_off(ax):
    """Get rid of all axis ticks, lines, etc.
    """
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)

def findMin(arr):
    """
    in comparison to argrelmax() more simple and  reliable peak finder
    """
    out = np.zeros(shape=arr.shape, dtype=bool)
    _calcMin(arr, out)
    return out

def setup_path():
    """Sets up the python include paths to include src"""
    import os.path; import sys

    if sys.argv[0]:
        top_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        sys.path = [os.path.join(top_dir, "src")] + sys.path
        pass
    return

def prevmonday(num):
    """
    Return unix SECOND timestamp of "num" mondays ago
    """
    today = get_today()
    lastmonday = today - timedelta(days=today.weekday(), weeks=num)
    return lastmonday

def merge(left, right, how='inner', key=None, left_key=None, right_key=None,
          left_as='left', right_as='right'):
    """ Performs a join using the union join function. """
    return join(left, right, how, key, left_key, right_key,
                join_fn=make_union_join(left_as, right_as))

def end_index(self):
        """
        Returns the 1-based index of the last object on this page,
        relative to total objects found (hits).
        """
        return ((self.number - 1) * self.paginator.per_page +
            len(self.object_list))

async def async_input(prompt):
    """
    Python's ``input()`` is blocking, which means the event loop we set
    above can't be running while we're blocking there. This method will
    let the loop run while we wait for input.
    """
    print(prompt, end='', flush=True)
    return (await loop.run_in_executor(None, sys.stdin.readline)).rstrip()

def get_property(self, property_name):
        """
        Get a property's value.

        property_name -- the property to get the value of

        Returns the properties value, if found, else None.
        """
        prop = self.find_property(property_name)
        if prop:
            return prop.get_value()

        return None

def copyFile(input, output, replace=None):
    """Copy a file whole from input to output."""

    _found = findFile(output)
    if not _found or (_found and replace):
        shutil.copy2(input, output)

def byte2int(s, index=0):
    """Get the ASCII int value of a character in a string.

    :param s: a string
    :param index: the position of desired character

    :return: ASCII int value
    """
    if six.PY2:
        return ord(s[index])
    return s[index]

def indent(text, amount, ch=' '):
    """Indents a string by the given amount of characters."""
    padding = amount * ch
    return ''.join(padding+line for line in text.splitlines(True))

def weekly(date=datetime.date.today()):
    """
    Weeks start are fixes at Monday for now.
    """
    return date - datetime.timedelta(days=date.weekday())

def cint32_array_to_numpy(cptr, length):
    """Convert a ctypes int pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_int32)):
        return np.fromiter(cptr, dtype=np.int32, count=length)
    else:
        raise RuntimeError('Expected int pointer')

def get_user_name():
    """Get user name provide by operating system
    """

    if sys.platform == 'win32':
        #user = os.getenv('USERPROFILE')
        user = os.getenv('USERNAME')
    else:
        user = os.getenv('LOGNAME')

    return user

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

def _get_minidom_tag_value(station, tag_name):
    """get a value from a tag (if it exists)"""
    tag = station.getElementsByTagName(tag_name)[0].firstChild
    if tag:
        return tag.nodeValue

    return None

def interpolate_slice(slice_rows, slice_cols, interpolator):
    """Interpolate the given slice of the larger array."""
    fine_rows = np.arange(slice_rows.start, slice_rows.stop, slice_rows.step)
    fine_cols = np.arange(slice_cols.start, slice_cols.stop, slice_cols.step)
    return interpolator(fine_cols, fine_rows)

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

def resample(grid, wl, flux):
    """ Resample spectrum onto desired grid """
    flux_rs = (interpolate.interp1d(wl, flux))(grid)
    return flux_rs

def title(self):
        """ The title of this window """
        with switch_window(self._browser, self.name):
            return self._browser.title

def pythonise(id, encoding='ascii'):
    """Return a Python-friendly id"""
    replace = {'-': '_', ':': '_', '/': '_'}
    func = lambda id, pair: id.replace(pair[0], pair[1])
    id = reduce(func, replace.iteritems(), id)
    id = '_%s' % id if id[0] in string.digits else id
    return id.encode(encoding)

def getEventTypeNameFromEnum(self, eType):
        """returns the name of an EVREvent enum value"""

        fn = self.function_table.getEventTypeNameFromEnum
        result = fn(eType)
        return result

def get_geoip(ip):
    """Lookup country for IP address."""
    reader = geolite2.reader()
    ip_data = reader.get(ip) or {}
    return ip_data.get('country', {}).get('iso_code')

def check_git():
    """Check if git command is available."""
    try:
        with open(os.devnull, "wb") as devnull:
            subprocess.check_call(["git", "--version"], stdout=devnull, stderr=devnull)
    except:
        raise RuntimeError("Please make sure git is installed and on your path.")

def is_identifier(string):
    """Check if string could be a valid python identifier

    :param string: string to be tested
    :returns: True if string can be a python identifier, False otherwise
    :rtype: bool
    """
    matched = PYTHON_IDENTIFIER_RE.match(string)
    return bool(matched) and not keyword.iskeyword(string)

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

def to_str(s):
    """
    Convert bytes and non-string into Python 3 str
    """
    if isinstance(s, bytes):
        s = s.decode('utf-8')
    elif not isinstance(s, str):
        s = str(s)
    return s

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

def query_proc_row(procname, args=(), factory=None):
    """
    Execute a stored procedure. Returns the first row of the result set,
    or `None`.
    """
    for row in query_proc(procname, args, factory):
        return row
    return None

def concat(cls, iterables):
    """
    Similar to #itertools.chain.from_iterable().
    """

    def generator():
      for it in iterables:
        for element in it:
          yield element
    return cls(generator())

def lazy_reverse_binmap(f, xs):
    """
    Same as lazy_binmap, except the parameters are flipped for the binary function
    """
    return (f(y, x) for x, y in zip(xs, xs[1:]))

def __next__(self):
    """Pop the head off the iterator and return it."""
    res = self._head
    self._fill()
    if res is None:
      raise StopIteration()
    return res

def _heappush_max(heap, item):
    """ why is this not in heapq """
    heap.append(item)
    heapq._siftdown_max(heap, 0, len(heap) - 1)

def A(*a):
    """convert iterable object into numpy array"""
    return np.array(a[0]) if len(a)==1 else [np.array(o) for o in a]

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

def each_img(dir_path):
    """
    Iterates through each image in the given directory. (not recursive)
    :param dir_path: Directory path where images files are present
    :return: Iterator to iterate through image files
    """
    for fname in os.listdir(dir_path):
        if fname.endswith('.jpg') or fname.endswith('.png') or fname.endswith('.bmp'):
            yield fname

def hex_escape(bin_str):
  """
  Hex encode a binary string
  """
  printable = string.ascii_letters + string.digits + string.punctuation + ' '
  return ''.join(ch if ch in printable else r'0x{0:02x}'.format(ord(ch)) for ch in bin_str)

def get_files(dir_name):
    """Simple directory walker"""
    return [(os.path.join('.', d), [os.path.join(d, f) for f in files]) for d, _, files in os.walk(dir_name)]

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

def _dict_values_sorted_by_key(dictionary):
    # This should be a yield from instead.
    """Internal helper to return the values of a dictionary, sorted by key.
    """
    for _, value in sorted(dictionary.iteritems(), key=operator.itemgetter(0)):
        yield value

def parse_comments_for_file(filename):
    """
    Return a list of all parsed comments in a file.  Mostly for testing &
    interactive use.
    """
    return [parse_comment(strip_stars(comment), next_line)
            for comment, next_line in get_doc_comments(read_file(filename))]

def __iter__(self):
		"""Iterate through all elements.

		Multiple copies will be returned if they exist.
		"""
		for value, count in self.counts():
			for _ in range(count):
				yield value

def isnumber(*args):
    """Checks if value is an integer, long integer or float.

    NOTE: Treats booleans as numbers, where True=1 and False=0.
    """
    return all(map(lambda c: isinstance(c, int) or isinstance(c, float), args))

def _unordered_iterator(self):
        """
        Return the value of each QuerySet, but also add the '#' property to each
        return item.
        """
        for i, qs in zip(self._queryset_idxs, self._querysets):
            for item in qs:
                setattr(item, '#', i)
                yield item

def pretty_xml(data):
    """Return a pretty formated xml
    """
    parsed_string = minidom.parseString(data.decode('utf-8'))
    return parsed_string.toprettyxml(indent='\t', encoding='utf-8')

def get_chunks(source, chunk_len):
    """ Returns an iterator over 'chunk_len' chunks of 'source' """
    return (source[i: i + chunk_len] for i in range(0, len(source), chunk_len))

def pickle_save(thing,fname):
    """save something to a pickle file"""
    pickle.dump(thing, open(fname,"wb"),pickle.HIGHEST_PROTOCOL)
    return thing

def index(self, elem):
        """Find the index of elem in the reversed iterator."""
        return _coconut.len(self._iter) - self._iter.index(elem) - 1

def getpackagepath():
    """
     *Get the root path for this python package - used in unit testing code*
    """
    moduleDirectory = os.path.dirname(__file__)
    packagePath = os.path.dirname(__file__) + "/../"

    return packagePath

def next(self):
        """Retrieve the next row."""
        # I'm pretty sure this is the completely wrong way to go about this, but
        # oh well, this works.
        if not hasattr(self, '_iter'):
            self._iter = self.readrow_as_dict()
        return self._iter.next()

def first_sunday(self, year, month):
        """Get the first sunday of a month."""
        date = datetime(year, month, 1, 0)
        days_until_sunday = 6 - date.weekday()

        return date + timedelta(days=days_until_sunday)

def group_by(iterable, key_func):
    """Wrap itertools.groupby to make life easier."""
    groups = (
        list(sub) for key, sub in groupby(iterable, key_func)
    )
    return zip(groups, groups)

def last(self):
        """Get the last object in file."""
        # End of file
        self.__file.seek(0, 2)

        # Get the last struct
        data = self.get(self.length - 1)

        return data

def html(header_rows):
    """
    Convert a list of tuples describing a table into a HTML string
    """
    name = 'table%d' % next(tablecounter)
    return HtmlTable([map(str, row) for row in header_rows], name).render()

def authenticate(self, transport, account_name, password=None):
        """
        Authenticates account using soap method.
        """
        Authenticator.authenticate(self, transport, account_name, password)

        if password == None:
            return self.pre_auth(transport, account_name)
        else:
            return self.auth(transport, account_name, password)

def render_template(template_name, **context):
    """Render a template into a response."""
    tmpl = jinja_env.get_template(template_name)
    context["url_for"] = url_for
    return Response(tmpl.render(context), mimetype="text/html")

def comma_converter(float_string):
    """Convert numbers to floats whether the decimal point is '.' or ','"""
    trans_table = maketrans(b',', b'.')
    return float(float_string.translate(trans_table))

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

def disable_insecure_request_warning():
    """Suppress warning about untrusted SSL certificate."""
    import requests
    from requests.packages.urllib3.exceptions import InsecureRequestWarning
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

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

def __iter__(self):
		"""Iterate through all elements.

		Multiple copies will be returned if they exist.
		"""
		for value, count in self.counts():
			for _ in range(count):
				yield value

def prepare_path(path):
    """
    Path join helper method
    Join paths if list passed

    :type path: str|unicode|list
    :rtype: str|unicode
    """
    if type(path) == list:
        return os.path.join(*path)
    return path

def finish_plot():
    """Helper for plotting."""
    plt.legend()
    plt.grid(color='0.7')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def is_valid_data(obj):
    """Check if data is JSON serializable.
    """
    if obj:
        try:
            tmp = json.dumps(obj, default=datetime_encoder)
            del tmp
        except (TypeError, UnicodeDecodeError):
            return False
    return True

def check_git():
    """Check if git command is available."""
    try:
        with open(os.devnull, "wb") as devnull:
            subprocess.check_call(["git", "--version"], stdout=devnull, stderr=devnull)
    except:
        raise RuntimeError("Please make sure git is installed and on your path.")

def _deserialize_datetime(self, data):
        """Take any values coming in as a datetime and deserialize them

        """
        for key in data:
            if isinstance(data[key], dict):
                if data[key].get('type') == 'datetime':
                    data[key] = \
                        datetime.datetime.fromtimestamp(data[key]['value'])
        return data

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

def _time_to_json(value):
    """Coerce 'value' to an JSON-compatible representation."""
    if isinstance(value, datetime.time):
        value = value.isoformat()
    return value

def fast_exit(code):
    """Exit without garbage collection, this speeds up exit by about 10ms for
    things like bash completion.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)

def to_json(obj):
    """Return a json string representing the python object obj."""
    i = StringIO.StringIO()
    w = Writer(i, encoding='UTF-8')
    w.write_value(obj)
    return i.getvalue()

def with_headers(self, headers):
        """Sets multiple headers on the request and returns the request itself.

        Keyword arguments:
        headers -- a dict-like object which contains the headers to set.
        """
        for key, value in headers.items():
            self.with_header(key, value)
        return self

def as_tree(context):
    """Return info about an object's members as JSON"""

    tree = _build_tree(context, 2, 1)
    if type(tree) == dict:
        tree = [tree] 
    
    return Response(content_type='application/json', body=json.dumps(tree))

def normalize_matrix(matrix):
  """Fold all values of the matrix into [0, 1]."""
  abs_matrix = np.abs(matrix.copy())
  return abs_matrix / abs_matrix.max()

def process_result_value(self, value, dialect):
        """convert value from json to a python object"""
        if value is not None:
            value = simplejson.loads(value)
        return value

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

def ratio_and_percentage(current, total, time_remaining):
    """Returns the progress ratio and percentage."""
    return "{} / {} ({}% completed)".format(current, total, int(current / total * 100))

def lmx_h1k_f64k():
  """HParams for training languagemodel_lm1b32k_packed.  880M Params."""
  hparams = lmx_base()
  hparams.hidden_size = 1024
  hparams.filter_size = 65536
  hparams.batch_size = 2048
  return hparams

def export(defn):
    """Decorator to explicitly mark functions that are exposed in a lib."""
    globals()[defn.__name__] = defn
    __all__.append(defn.__name__)
    return defn

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

def extend(self, item):
        """Extend the list with another list. Each member of the list must be
        a string."""
        if not isinstance(item, list):
            raise TypeError(
                'You can only extend lists with lists. '
                'You supplied \"%s\"' % type(item))
        for entry in item:
            if not isinstance(entry, str):
                raise TypeError(
                    'Members of this object must be strings. '
                    'You supplied \"%s\"' % type(entry))
            list.append(self, entry)

def process_kill(pid, sig=None):
    """Send signal to process.
    """
    sig = sig or signal.SIGTERM
    os.kill(pid, sig)

def prettifysql(sql):
    """Returns a prettified version of the SQL as a list of lines to help
    in creating a useful diff between two SQL statements."""
    pretty = []
    for line in sql.split('\n'):
        pretty.extend(["%s,\n" % x for x in line.split(',')])
    return pretty

def hard_equals(a, b):
    """Implements the '===' operator."""
    if type(a) != type(b):
        return False
    return a == b

def make_lambda(call):
    """Wrap an AST Call node to lambda expression node.
    call: ast.Call node
    """
    empty_args = ast.arguments(args=[], vararg=None, kwarg=None, defaults=[])
    return ast.Lambda(args=empty_args, body=call)

def print_datetime_object(dt):
    """prints a date-object"""
    print(dt)
    print('ctime  :', dt.ctime())
    print('tuple  :', dt.timetuple())
    print('ordinal:', dt.toordinal())
    print('Year   :', dt.year)
    print('Mon    :', dt.month)
    print('Day    :', dt.day)

def retry_on_signal(function):
    """Retries function until it doesn't raise an EINTR error"""
    while True:
        try:
            return function()
        except EnvironmentError, e:
            if e.errno != errno.EINTR:
                raise

def _clean_str(self, s):
        """ Returns a lowercase string with punctuation and bad chars removed
        :param s: string to clean
        """
        return s.translate(str.maketrans('', '', punctuation)).replace('\u200b', " ").strip().lower()

def _pad(self, text):
        """Pad the text."""
        top_bottom = ("\n" * self._padding) + " "
        right_left = " " * self._padding * self.PAD_WIDTH
        return top_bottom + right_left + text + right_left + top_bottom

def normalize_text(text, line_len=80, indent=""):
    """Wrap the text on the given line length."""
    return "\n".join(
        textwrap.wrap(
            text, width=line_len, initial_indent=indent, subsequent_indent=indent
        )
    )

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

def inh(table):
    """
    inverse hyperbolic sine transformation
    """
    t = []
    for i in table:
        t.append(np.ndarray.tolist(np.arcsinh(i)))
    return t

def _get_loggers():
    """Return list of Logger classes."""
    from .. import loader
    modules = loader.get_package_modules('logger')
    return list(loader.get_plugins(modules, [_Logger]))

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

def get_methods(*objs):
    """ Return the names of all callable attributes of an object"""
    return set(
        attr
        for obj in objs
        for attr in dir(obj)
        if not attr.startswith('_') and callable(getattr(obj, attr))
    )

def read(fname):
    """Quick way to read a file content."""
    content = None
    with open(os.path.join(here, fname)) as f:
        content = f.read()
    return content

def _rectangular(n):
    """Checks to see if a 2D list is a valid 2D matrix"""
    for i in n:
        if len(i) != len(n[0]):
            return False
    return True

def str_time_to_day_seconds(time):
    """
    Converts time strings to integer seconds
    :param time: %H:%M:%S string
    :return: integer seconds
    """
    t = str(time).split(':')
    seconds = int(t[0]) * 3600 + int(t[1]) * 60 + int(t[2])
    return seconds

def remove_dups(seq):
    """remove duplicates from a sequence, preserving order"""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def poke_array(self, store, name, elemtype, elements, container, visited, _stack):
        """abstract method"""
        raise NotImplementedError

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

def cfloat64_array_to_numpy(cptr, length):
    """Convert a ctypes double pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_double)):
        return np.fromiter(cptr, dtype=np.float64, count=length)
    else:
        raise RuntimeError('Expected double pointer')

def A(*a):
    """convert iterable object into numpy array"""
    return np.array(a[0]) if len(a)==1 else [np.array(o) for o in a]

def get_last_row(dbconn, tablename, n=1, uuid=None):
    """
    Returns the last `n` rows in the table
    """
    return fetch(dbconn, tablename, n, uuid, end=True)

def keys(self, index=None):
        """Returns a list of keys in the database
        """
        with self._lmdb.begin() as txn:
            return [key.decode() for key, _ in txn.cursor()]

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

def load(cls, fname):
        """ Loads the dictionary from json file
        :param fname: file to load from
        :return: loaded dictionary
        """
        with open(fname) as f:
            return Config(**json.load(f))

def show_correlation_matrix(self, correlation_matrix):
        """Shows the given correlation matrix as image

        :param correlation_matrix: Correlation matrix of features
        """
        cr_plot.create_correlation_matrix_plot(
            correlation_matrix, self.title, self.headers_to_test
        )
        pyplot.show()

def load(self, name):
        """Loads and returns foreign library."""
        name = ctypes.util.find_library(name)
        return ctypes.cdll.LoadLibrary(name)

def unpatch(obj, name):
    """
    Undo the effects of patch(func, obj, name)
    """
    setattr(obj, name, getattr(obj, name).original)

def load(path):
    """Loads a pushdb maintained in a properties file at the given path."""
    with open(path, 'r') as props:
      properties = Properties.load(props)
      return PushDb(properties)

def get_height_for_line(self, lineno):
        """
        Return the height of the given line.
        (The height that it would take, if this line became visible.)
        """
        if self.wrap_lines:
            return self.ui_content.get_height_for_line(lineno, self.window_width)
        else:
            return 1

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

def generic_add(a, b):
    """Simple function to add two numbers"""
    logger.debug('Called generic_add({}, {})'.format(a, b))
    return a + b

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

def toggle_word_wrap(self):
        """
        Toggles document word wrap.

        :return: Method success.
        :rtype: bool
        """

        self.setWordWrapMode(not self.wordWrapMode() and QTextOption.WordWrap or QTextOption.NoWrap)
        return True

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

def _openResources(self):
        """ Uses numpy.load to open the underlying file
        """
        arr = np.load(self._fileName, allow_pickle=ALLOW_PICKLE)
        check_is_an_array(arr)
        self._array = arr

def transform(self, df):
        """
        Transforms a DataFrame in place. Computes all outputs of the DataFrame.

        Args:
            df (pandas.DataFrame): DataFrame to transform.
        """
        for name, function in self.outputs:
            df[name] = function(df)

def load(self, path):
        """Load the pickled model weights."""
        with io.open(path, 'rb') as fin:
            self.weights = pickle.load(fin)

def _IsDirectory(parent, item):
  """Helper that returns if parent/item is a directory."""
  return tf.io.gfile.isdir(os.path.join(parent, item))

def Load(file):
    """ Loads a model from specified file """
    with open(file, 'rb') as file:
        model = dill.load(file)
        return model

def get_average_color(colors):
    """Calculate the average color from the list of colors, where each color
    is a 3-tuple of (r, g, b) values.
    """
    c = reduce(color_reducer, colors)
    total = len(colors)
    return tuple(v / total for v in c)

def elem_find(self, field, value):
        """
        Return the indices of elements whose field first satisfies the given values

        ``value`` should be unique in self.field.
        This function does not check the uniqueness.

        :param field: name of the supplied field
        :param value: value of field of the elemtn to find
        :return: idx of the elements
        :rtype: list, int, float, str
        """
        if isinstance(value, (int, float, str)):
            value = [value]

        f = list(self.__dict__[field])
        uid = np.vectorize(f.index)(value)
        return self.get_idx(uid)

def camel_case(self, snake_case):
        """ Convert snake case to camel case """
        components = snake_case.split('_')
        return components[0] + "".join(x.title() for x in components[1:])

def lock(self, block=True):
		"""
		Lock connection from being used else where
		"""
		self._locked = True
		return self._lock.acquire(block)

def simple_memoize(callable_object):
    """Simple memoization for functions without keyword arguments.

    This is useful for mapping code objects to module in this context.
    inspect.getmodule() requires a number of system calls, which may slow down
    the tracing considerably. Caching the mapping from code objects (there is
    *one* code object for each function, regardless of how many simultaneous
    activations records there are).

    In this context we can ignore keyword arguments, but a generic memoizer
    ought to take care of that as well.
    """

    cache = dict()

    def wrapper(*rest):
        if rest not in cache:
            cache[rest] = callable_object(*rest)
        return cache[rest]

    return wrapper

def buttonUp(self, button=mouse.LEFT):
        """ Releases the specified mouse button.

        Use Mouse.LEFT, Mouse.MIDDLE, Mouse.RIGHT
        """
        self._lock.acquire()
        mouse.release(button)
        self._lock.release()

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

def lock(self, block=True):
		"""
		Lock connection from being used else where
		"""
		self._locked = True
		return self._lock.acquire(block)

def log_magnitude_spectrum(frames):
    """Compute the log of the magnitude spectrum of frames"""
    return N.log(N.abs(N.fft.rfft(frames)).clip(1e-5, N.inf))

def to_str(obj):
    """Attempts to convert given object to a string object
    """
    if not isinstance(obj, str) and PY3 and isinstance(obj, bytes):
        obj = obj.decode('utf-8')
    return obj if isinstance(obj, string_types) else str(obj)

def log_loss(preds, labels):
    """Logarithmic loss with non-necessarily-binary labels."""
    log_likelihood = np.sum(labels * np.log(preds)) / len(preds)
    return -log_likelihood

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

def load_config(filename="logging.ini", *args, **kwargs):
    """
    Load logger config from file
    
    Keyword arguments:
    filename -- configuration filename (Default: "logging.ini")
    *args -- options passed to fileConfig
    **kwargs -- options passed to fileConfigg
    
    """
    logging.config.fileConfig(filename, *args, **kwargs)

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

def logger(message, level=10):
    """Handle logging."""
    logging.getLogger(__name__).log(level, str(message))

def list_i2str(ilist):
    """
    Convert an integer list into a string list.
    """
    slist = []
    for el in ilist:
        slist.append(str(el))
    return slist

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

def intToBin(i):
    """ Integer to two bytes """
    # devide in two parts (bytes)
    i1 = i % 256
    i2 = int(i / 256)
    # make string (little endian)
    return chr(i1) + chr(i2)

def debug(self, text):
		""" Ajout d'un message de log de type DEBUG """
		self.logger.debug("{}{}".format(self.message_prefix, text))

def unaccentuate(s):
    """ Replace accentuated chars in string by their non accentuated equivalent. """
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def log_no_newline(self, msg):
      """ print the message to the predefined log file without newline """
      self.print2file(self.logfile, False, False, msg)

def QA_util_datetime_to_strdate(dt):
    """
    :param dt:  pythone datetime.datetime
    :return:  1999-02-01 string type
    """
    strdate = "%04d-%02d-%02d" % (dt.year, dt.month, dt.day)
    return strdate

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

def auto():
	"""set colouring on if STDOUT is a terminal device, off otherwise"""
	try:
		Style.enabled = False
		Style.enabled = sys.stdout.isatty()
	except (AttributeError, TypeError):
		pass

def parse_comments_for_file(filename):
    """
    Return a list of all parsed comments in a file.  Mostly for testing &
    interactive use.
    """
    return [parse_comment(strip_stars(comment), next_line)
            for comment, next_line in get_doc_comments(read_file(filename))]

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

def b(s):
	""" Encodes Unicode strings to byte strings, if necessary. """

	return s if isinstance(s, bytes) else s.encode(locale.getpreferredencoding())

def osx_clipboard_get():
    """ Get the clipboard's text on OS X.
    """
    p = subprocess.Popen(['pbpaste', '-Prefer', 'ascii'],
        stdout=subprocess.PIPE)
    text, stderr = p.communicate()
    # Text comes in with old Mac \r line endings. Change them to \n.
    text = text.replace('\r', '\n')
    return text

def pprint(obj, verbose=False, max_width=79, newline='\n'):
    """
    Like `pretty` but print to stdout.
    """
    printer = RepresentationPrinter(sys.stdout, verbose, max_width, newline)
    printer.pretty(obj)
    printer.flush()
    sys.stdout.write(newline)
    sys.stdout.flush()

def mock_add_spec(self, spec, spec_set=False):
        """Add a spec to a mock. `spec` can either be an object or a
        list of strings. Only attributes on the `spec` can be fetched as
        attributes from the mock.

        If `spec_set` is True then only attributes on the spec can be set."""
        self._mock_add_spec(spec, spec_set)
        self._mock_set_magics()

def ansi(color, text):
    """Wrap text in an ansi escape sequence"""
    code = COLOR_CODES[color]
    return '\033[1;{0}m{1}{2}'.format(code, text, RESET_TERM)

def list_string_to_dict(string):
    """Inputs ``['a', 'b', 'c']``, returns ``{'a': 0, 'b': 1, 'c': 2}``."""
    dictionary = {}
    for idx, c in enumerate(string):
        dictionary.update({c: idx})
    return dictionary

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

def fopenat(base_fd, path):
    """
    Does openat read-only, then does fdopen to get a file object
    """

    return os.fdopen(openat(base_fd, path, os.O_RDONLY), 'rb')

def colorize(string, color, *args, **kwargs):
    """
    Implements string formatting along with color specified in colorama.Fore
    """
    string = string.format(*args, **kwargs)
    return color + string + colorama.Fore.RESET

def unpickle_stats(stats):
    """Unpickle a pstats.Stats object"""
    stats = cPickle.loads(stats)
    stats.stream = True
    return stats

def set_executable(filename):
    """Set the exectuable bit on the given filename"""
    st = os.stat(filename)
    os.chmod(filename, st.st_mode | stat.S_IEXEC)

def multi_replace(instr, search_list=[], repl_list=None):
    """
    Does a string replace with a list of search and replacements

    TODO: rename
    """
    repl_list = [''] * len(search_list) if repl_list is None else repl_list
    for ser, repl in zip(search_list, repl_list):
        instr = instr.replace(ser, repl)
    return instr

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

def bitsToString(arr):
  """Returns a string representing a numpy array of 0's and 1's"""
  s = array('c','.'*len(arr))
  for i in xrange(len(arr)):
    if arr[i] == 1:
      s[i]='*'
  return s

def title(msg):
    """Sets the title of the console window."""
    if sys.platform.startswith("win"):
        ctypes.windll.kernel32.SetConsoleTitleW(tounicode(msg))

def create_search_url(self):
        """ Generates (urlencoded) query string from stored key-values tuples

        :returns: A string containing all arguments in a url-encoded format
        """

        url = '?'
        for key, value in self.arguments.items():
            url += '%s=%s&' % (quote_plus(key), quote_plus(value))
        self.url = url[:-1]
        return self.url

def append_user_agent(self, user_agent):
        """Append text to the User-Agent header for the request.

        Use this method to update the User-Agent header by appending the
        given string to the session's User-Agent header separated by a space.

        :param user_agent: A string to append to the User-Agent header
        :type user_agent: str
        """
        old_ua = self.session.headers.get('User-Agent', '')
        ua = old_ua + ' ' + user_agent
        self.session.headers['User-Agent'] = ua.strip()

def makedirs(path):
    """
    Create directories if they do not exist, otherwise do nothing.

    Return path for convenience
    """
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def str2int(string_with_int):
    """ Collect digits from a string """
    return int("".join([char for char in string_with_int if char in string.digits]) or 0)

def force_iterable(f):
    """Will make any functions return an iterable objects by wrapping its result in a list."""
    def wrapper(*args, **kwargs):
        r = f(*args, **kwargs)
        if hasattr(r, '__iter__'):
            return r
        else:
            return [r]
    return wrapper

def _valid_other_type(x, types):
    """
    Do all elements of x have a type from types?
    """
    return all(any(isinstance(el, t) for t in types) for el in np.ravel(x))

def list_of_lists_to_dict(l):
    """ Convert list of key,value lists to dict

    [['id', 1], ['id', 2], ['id', 3], ['foo': 4]]
    {'id': [1, 2, 3], 'foo': [4]}
    """
    d = {}
    for key, val in l:
        d.setdefault(key, []).append(val)
    return d

def _validate_pos(df):
    """Validates the returned positional object
    """
    assert isinstance(df, pd.DataFrame)
    assert ["seqname", "position", "strand"] == df.columns.tolist()
    assert df.position.dtype == np.dtype("int64")
    assert df.strand.dtype == np.dtype("O")
    assert df.seqname.dtype == np.dtype("O")
    return df

def server(port):
    """Start the Django dev server."""
    args = ['python', 'manage.py', 'runserver']
    if port:
        args.append(port)
    run.main(args)

def check(modname):
    """Check if required dependency is installed"""
    for dependency in DEPENDENCIES:
        if dependency.modname == modname:
            return dependency.check()
    else:
        raise RuntimeError("Unkwown dependency %s" % modname)

def server(port):
    """Start the Django dev server."""
    args = ['python', 'manage.py', 'runserver']
    if port:
        args.append(port)
    run.main(args)

def check(modname):
    """Check if required dependency is installed"""
    for dependency in DEPENDENCIES:
        if dependency.modname == modname:
            return dependency.check()
    else:
        raise RuntimeError("Unkwown dependency %s" % modname)

def generate_dumper(self, mapfile, names):
        """
        Build dumpdata commands
        """
        return self.build_template(mapfile, names, self._dumpdata_template)

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

def str2int(string_with_int):
    """ Collect digits from a string """
    return int("".join([char for char in string_with_int if char in string.digits]) or 0)

def colorbar(height, length, colormap):
    """Return the channels of a colorbar.
    """
    cbar = np.tile(np.arange(length) * 1.0 / (length - 1), (height, 1))
    cbar = (cbar * (colormap.values.max() - colormap.values.min())
            + colormap.values.min())

    return colormap.colorize(cbar)

def isin(value, values):
    """ Check that value is in values """
    for i, v in enumerate(value):
        if v not in np.array(values)[:, i]:
            return False
    return True

def finish_plot():
    """Helper for plotting."""
    plt.legend()
    plt.grid(color='0.7')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def init_checks_registry():
    """Register all globally visible functions.

    The first argument name is either 'physical_line' or 'logical_line'.
    """
    mod = inspect.getmodule(register_check)
    for (name, function) in inspect.getmembers(mod, inspect.isfunction):
        register_check(function)

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

def check_git():
    """Check if git command is available."""
    try:
        with open(os.devnull, "wb") as devnull:
            subprocess.check_call(["git", "--version"], stdout=devnull, stderr=devnull)
    except:
        raise RuntimeError("Please make sure git is installed and on your path.")

def plot_and_save(self, **kwargs):
        """Used when the plot method defined does not create a figure nor calls save_plot
        Then the plot method has to use self.fig"""
        self.fig = pyplot.figure()
        self.plot()
        self.axes = pyplot.gca()
        self.save_plot(self.fig, self.axes, **kwargs)
        pyplot.close(self.fig)

def contained_in(filename, directory):
    """Test if a file is located within the given directory."""
    filename = os.path.normcase(os.path.abspath(filename))
    directory = os.path.normcase(os.path.abspath(directory))
    return os.path.commonprefix([filename, directory]) == directory

def set_title(self, title, **kwargs):
        """Sets the title on the underlying matplotlib AxesSubplot."""
        ax = self.get_axes()
        ax.set_title(title, **kwargs)

def _file_exists(path, filename):
  """Checks if the filename exists under the path."""
  return os.path.isfile(os.path.join(path, filename))

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

def is_clicked(self, MouseStateType):
        """
        Did the user depress and release the button to signify a click?
        MouseStateType is the button to query. Values found under StateTypes.py
        """
        return self.previous_mouse_state.query_state(MouseStateType) and (
        not self.current_mouse_state.query_state(MouseStateType))

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

def is_valid_url(url):
    """Checks if a given string is an url"""
    pieces = urlparse(url)
    return all([pieces.scheme, pieces.netloc])

def merge(self, other):
        """ Merge another stats. """
        Stats.merge(self, other)
        self.changes += other.changes

def _check_for_duplicate_sequence_names(self, fasta_file_path):
        """Test if the given fasta file contains sequences with duplicate
        sequence names.

        Parameters
        ----------
        fasta_file_path: string
            path to file that is to be checked

        Returns
        -------
        The name of the first duplicate sequence found, else False.

        """
        found_sequence_names = set()
        for record in SeqIO.parse(fasta_file_path, 'fasta'):
            name = record.name
            if name in found_sequence_names:
                return name
            found_sequence_names.add(name)
        return False

def dict_merge(set1, set2):
    """Joins two dictionaries."""
    return dict(list(set1.items()) + list(set2.items()))

def is_callable(*p):
    """ True if all the args are functions and / or subroutines
    """
    import symbols
    return all(isinstance(x, symbols.FUNCTION) for x in p)

def _get_minidom_tag_value(station, tag_name):
    """get a value from a tag (if it exists)"""
    tag = station.getElementsByTagName(tag_name)[0].firstChild
    if tag:
        return tag.nodeValue

    return None

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

def makedirs(path, mode=0o777, exist_ok=False):
    """A wrapper of os.makedirs()."""
    os.makedirs(path, mode, exist_ok)

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

async def _thread_coro(self, *args):
        """ Coroutine called by MapAsync. It's wrapping the call of
        run_in_executor to run the synchronous function as thread """
        return await self._loop.run_in_executor(
            self._executor, self._function, *args)

def __contains__(self, key):
        """
        Invoked when determining whether a specific key is in the dictionary
        using `key in d`.

        The key is looked up case-insensitively.
        """
        k = self._real_key(key)
        return k in self._data

def alter_change_column(self, table, column, field):
        """Support change columns."""
        return self._update_column(table, column, lambda a, b: b)

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

def ffmpeg_works():
  """Tries to encode images with ffmpeg to check if it works."""
  images = np.zeros((2, 32, 32, 3), dtype=np.uint8)
  try:
    _encode_gif(images, 2)
    return True
  except (IOError, OSError):
    return False

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

def column_exists(cr, table, column):
    """ Check whether a certain column exists """
    cr.execute(
        'SELECT count(attname) FROM pg_attribute '
        'WHERE attrelid = '
        '( SELECT oid FROM pg_class WHERE relname = %s ) '
        'AND attname = %s',
        (table, column))
    return cr.fetchone()[0] == 1

def _most_common(iterable):
    """Returns the most common element in `iterable`."""
    data = Counter(iterable)
    return max(data, key=data.__getitem__)

def check_permission_safety(path):
    """Check if the file at the given path is safe to use as a state file.

    This checks that group and others have no permissions on the file and that the current user is
    the owner.
    """
    f_stats = os.stat(path)
    return (f_stats.st_mode & (stat.S_IRWXG | stat.S_IRWXO)) == 0 and f_stats.st_uid == os.getuid()

def _go_to_line(editor, line):
    """
    Move cursor to this line in the current buffer.
    """
    b = editor.application.current_buffer
    b.cursor_position = b.document.translate_row_col_to_index(max(0, int(line) - 1), 0)

def IPYTHON_MAIN():
    """Decide if the Ipython command line is running code."""
    import pkg_resources

    runner_frame = inspect.getouterframes(inspect.currentframe())[-2]
    return (
        getattr(runner_frame, "function", None)
        == pkg_resources.load_entry_point("ipython", "console_scripts", "ipython").__name__
    )

def move_up(lines=1, file=sys.stdout):
    """ Move the cursor up a number of lines.

        Esc[ValueA:
        Moves the cursor up by the specified number of lines without changing
        columns. If the cursor is already on the top line, ANSI.SYS ignores
        this sequence.
    """
    move.up(lines).write(file=file)

def erase(self):
        """White out the progress bar."""
        with self._at_last_line():
            self.stream.write(self._term.clear_eol)
        self.stream.flush()

def set_cursor(self, x, y):
        """
        Sets the cursor to the desired position.

        :param x: X position
        :param y: Y position
        """
        curses.curs_set(1)
        self.screen.move(y, x)

def __clear_buffers(self):
        """Clears the input and output buffers"""
        try:
            self._port.reset_input_buffer()
            self._port.reset_output_buffer()
        except AttributeError:
            #pySerial 2.7
            self._port.flushInput()
            self._port.flushOutput()

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

def __clear_buffers(self):
        """Clears the input and output buffers"""
        try:
            self._port.reset_input_buffer()
            self._port.reset_output_buffer()
        except AttributeError:
            #pySerial 2.7
            self._port.flushInput()
            self._port.flushOutput()

def list(self):
        """position in 3d space"""
        return [self._pos3d.x, self._pos3d.y, self._pos3d.z]

def clear(self):
        """Clear the displayed image."""
        self._imgobj = None
        try:
            # See if there is an image on the canvas
            self.canvas.delete_object_by_tag(self._canvas_img_tag)
            self.redraw()
        except KeyError:
            pass

def map(cls, iterable, func, *a, **kw):
    """
    Iterable-first replacement of Python's built-in `map()` function.
    """

    return cls(func(x, *a, **kw) for x in iterable)

def cio_close(cio):
    """Wraps openjpeg library function cio_close.
    """
    OPENJPEG.opj_cio_close.argtypes = [ctypes.POINTER(CioType)]
    OPENJPEG.opj_cio_close(cio)

async def result_processor(tasks):
    """An async result aggregator that combines all the results
       This gets executed in unsync.loop and unsync.thread"""
    output = {}
    for task in tasks:
        num, res = await task
        output[num] = res
    return output

def close( self ):
        """
        Close the db and release memory
        """
        if self.db is not None:
            self.db.commit()
            self.db.close()
            self.db = None

        return

def _parallel_compare_helper(class_obj, pairs, x, x_link=None):
    """Internal function to overcome pickling problem in python2."""
    return class_obj._compute(pairs, x, x_link)

def resources(self):
        """Retrieve contents of each page of PDF"""
        return [self.pdf.getPage(i) for i in range(self.pdf.getNumPages())]

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

def __iadd__(self, other_model):
        """Incrementally add the content of another model to this model (+=).

        Copies of all the reactions in the other model are added to this
        model. The objective is the sum of the objective expressions for the
        two models.
        """
        warn('use model.merge instead', DeprecationWarning)
        return self.merge(other_model, objective='sum', inplace=True)

def parallel(processes, threads):
    """
    execute jobs in processes using N threads
    """
    pool = multithread(threads)
    pool.map(run_process, processes)
    pool.close()
    pool.join()

def compare(a, b):
    """
     Compare items in 2 arrays. Returns sum(abs(a(i)-b(i)))
    """
    s=0
    for i in range(len(a)):
        s=s+abs(a[i]-b[i])
    return s

def machine_info():
    """Retrieve core and memory information for the current machine.
    """
    import psutil
    BYTES_IN_GIG = 1073741824.0
    free_bytes = psutil.virtual_memory().total
    return [{"memory": float("%.1f" % (free_bytes / BYTES_IN_GIG)), "cores": multiprocessing.cpu_count(),
             "name": socket.gethostname()}]

def is_value_type_valid_for_exact_conditions(self, value):
    """ Method to validate if the value is valid for exact match type evaluation.

    Args:
      value: Value to validate.

    Returns:
      Boolean: True if value is a string, boolean, or number. Otherwise False.
    """
    # No need to check for bool since bool is a subclass of int
    if isinstance(value, string_types) or isinstance(value, (numbers.Integral, float)):
      return True

    return False

def _parallel_compare_helper(class_obj, pairs, x, x_link=None):
    """Internal function to overcome pickling problem in python2."""
    return class_obj._compute(pairs, x, x_link)

def __eq__(self, other):
        """Determine if two objects are equal."""
        return isinstance(other, self.__class__) \
            and self._freeze() == other._freeze()

def _root(self):
        """Attribute referencing the root node of the tree.

        :returns: the root node of the tree containing this instance.
        :rtype: Node
        """
        _n = self
        while _n.parent:
            _n = _n.parent
        return _n

def pair_strings_sum_formatter(a, b):
  """
  Formats the sum of a and b.

  Note
  ----
  Both inputs are numbers already converted to strings.

  """
  if b[:1] == "-":
    return "{0} - {1}".format(a, b[1:])
  return "{0} + {1}".format(a, b)

def drop_all_tables(self):
        """Drop all tables in the database"""
        for table_name in self.table_names():
            self.execute_sql("DROP TABLE %s" % table_name)
        self.connection.commit()

def retry_on_signal(function):
    """Retries function until it doesn't raise an EINTR error"""
    while True:
        try:
            return function()
        except EnvironmentError, e:
            if e.errno != errno.EINTR:
                raise

def dictify(a_named_tuple):
    """Transform a named tuple into a dictionary"""
    return dict((s, getattr(a_named_tuple, s)) for s in a_named_tuple._fields)

def str_dict(some_dict):
    """Convert dict of ascii str/unicode to dict of str, if necessary"""
    return {str(k): str(v) for k, v in some_dict.items()}

def export(defn):
    """Decorator to explicitly mark functions that are exposed in a lib."""
    globals()[defn.__name__] = defn
    __all__.append(defn.__name__)
    return defn

def paste(xsel=False):
    """Returns system clipboard contents."""
    selection = "primary" if xsel else "clipboard"
    try:
        return subprocess.Popen(["xclip", "-selection", selection, "-o"], stdout=subprocess.PIPE).communicate()[0].decode("utf-8")
    except OSError as why:
        raise XclipNotFound

def Cinv(self):
        """Inverse of the noise covariance."""
        try:
            return np.linalg.inv(self.c)
        except np.linalg.linalg.LinAlgError:
            print('Warning: non-invertible noise covariance matrix c.')
            return np.eye(self.c.shape[0])

def copyFile(input, output, replace=None):
    """Copy a file whole from input to output."""

    _found = findFile(output)
    if not _found or (_found and replace):
        shutil.copy2(input, output)

def _method_scope(input_layer, name):
  """Creates a nested set of name and id scopes and avoids repeats."""
  global _in_method_scope
  # pylint: disable=protected-access

  with input_layer.g.as_default(), \
       scopes.var_and_name_scope(
           None if _in_method_scope else input_layer._scope), \
       scopes.var_and_name_scope((name, None)) as (scope, var_scope):
    was_in_method_scope = _in_method_scope
    yield scope, var_scope
    _in_method_scope = was_in_method_scope

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

def is_serializable(obj):
    """Return `True` if the given object conforms to the Serializable protocol.

    :rtype: bool
    """
    if inspect.isclass(obj):
      return Serializable.is_serializable_type(obj)
    return isinstance(obj, Serializable) or hasattr(obj, '_asdict')

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

def apply_fit(xy,coeffs):
    """ Apply the coefficients from a linear fit to
        an array of x,y positions.

        The coeffs come from the 'coeffs' member of the
        'fit_arrays()' output.
    """
    x_new = coeffs[0][2] + coeffs[0][0]*xy[:,0] + coeffs[0][1]*xy[:,1]
    y_new = coeffs[1][2] + coeffs[1][0]*xy[:,0] + coeffs[1][1]*xy[:,1]

    return x_new,y_new

def num_leaves(tree):
    """Determine the number of leaves in a tree"""
    if tree.is_leaf:
        return 1
    else:
        return num_leaves(tree.left_child) + num_leaves(tree.right_child)

def mag(z):
    """Get the magnitude of a vector."""
    if isinstance(z[0], np.ndarray):
        return np.array(list(map(np.linalg.norm, z)))
    else:
        return np.linalg.norm(z)

def objectcount(data, key):
    """return the count of objects of key"""
    objkey = key.upper()
    return len(data.dt[objkey])

def EvalGaussianPdf(x, mu, sigma):
    """Computes the unnormalized PDF of the normal distribution.

    x: value
    mu: mean
    sigma: standard deviation
    
    returns: float probability density
    """
    return scipy.stats.norm.pdf(x, mu, sigma)

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

def gauss_pdf(x, mu, sigma):
    """Normalized Gaussian"""
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - mu) ** 2 / 2. / sigma ** 2)

def count_list(the_list):
    """
    Generates a count of the number of times each unique item appears in a list
    """
    count = the_list.count
    result = [(item, count(item)) for item in set(the_list)]
    result.sort()
    return result

def _normalize(mat: np.ndarray):
    """rescales a numpy array, so that min is 0 and max is 255"""
    return ((mat - mat.min()) * (255 / mat.max())).astype(np.uint8)

def Pyramid(pos=(0, 0, 0), s=1, height=1, axis=(0, 0, 1), c="dg", alpha=1):
    """
    Build a pyramid of specified base size `s` and `height`, centered at `pos`.
    """
    return Cone(pos, s, height, axis, c, alpha, 4)

def test(nose_argsuments):
    """ Run application tests """
    from nose import run

    params = ['__main__', '-c', 'nose.ini']
    params.extend(nose_argsuments)
    run(argv=params)

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

def log_loss(preds, labels):
    """Logarithmic loss with non-necessarily-binary labels."""
    log_likelihood = np.sum(labels * np.log(preds)) / len(preds)
    return -log_likelihood

def format_result(input):
        """From: http://stackoverflow.com/questions/13062300/convert-a-dict-to-sorted-dict-in-python
        """
        items = list(iteritems(input))
        return OrderedDict(sorted(items, key=lambda x: x[0]))

def gday_of_year(self):
        """Return the number of days since January 1 of the given year."""
        return (self.date - dt.date(self.date.year, 1, 1)).days

def vals2bins(vals,res=100):
    """Maps values to bins
    Args:
    values (list or list of lists) - list of values to map to colors
    res (int) - resolution of the color map (default: 100)
    Returns:
    list of numbers representing bins
    """
    # flatten if list of lists
    if any(isinstance(el, list) for el in vals):
        vals = list(itertools.chain(*vals))
    return list(np.digitize(vals, np.linspace(np.min(vals), np.max(vals)+1, res+1)) - 1)

def gday_of_year(self):
        """Return the number of days since January 1 of the given year."""
        return (self.date - dt.date(self.date.year, 1, 1)).days

def to_dataframe(products):
        """Return the products from a query response as a Pandas DataFrame
        with the values in their appropriate Python types.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("to_dataframe requires the optional dependency Pandas.")

        return pd.DataFrame.from_dict(products, orient='index')

def count_rows_with_nans(X):
    """Count the number of rows in 2D arrays that contain any nan values."""
    if X.ndim == 2:
        return np.where(np.isnan(X).sum(axis=1) != 0, 1, 0).sum()

def from_dict(cls, d):
        """Create an instance from a dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.ENTRIES})

def count(data, axis=None):
    """Count the number of non-NA in this array along the given axis or axes
    """
    return np.sum(np.logical_not(isnull(data)), axis=axis)

def _decode_request(self, encoded_request):
        """Decode an request previously encoded"""
        obj = self.serializer.loads(encoded_request)
        return request_from_dict(obj, self.spider)

def unit_vector(x):
    """Return a unit vector in the same direction as x."""
    y = np.array(x, dtype='float')
    return y/norm(y)

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

def register_type(cls, name):
    """Register `name` as a type to validate as an instance of class `cls`."""
    x = TypeDefinition(name, (cls,), ())
    Validator.types_mapping[name] = x

def pairwise_indices(self):
        """ndarray containing tuples of pairwise indices."""
        return np.array([sig.pairwise_indices for sig in self.values]).T

def printmp(msg):
    """Print temporarily, until next print overrides it.
    """
    filler = (80 - len(msg)) * ' '
    print(msg + filler, end='\r')
    sys.stdout.flush()

def _numpy_bytes_to_char(arr):
    """Like netCDF4.stringtochar, but faster and more flexible.
    """
    # ensure the array is contiguous
    arr = np.array(arr, copy=False, order='C', dtype=np.string_)
    return arr.reshape(arr.shape + (1,)).view('S1')

def delete(self, row):
        """Delete a track value"""
        i = self._get_key_index(row)
        del self.keys[i]

def one_hot(x, size, dtype=np.float32):
  """Make a n+1 dim one-hot array from n dim int-categorical array."""
  return np.array(x[..., np.newaxis] == np.arange(size), dtype)

def cli(env, identifier):
    """Delete an image."""

    image_mgr = SoftLayer.ImageManager(env.client)
    image_id = helpers.resolve_id(image_mgr.resolve_ids, identifier, 'image')

    image_mgr.delete_image(image_id)

def contains_all(self, array):
        """Test if `array` is an array of real numbers."""
        dtype = getattr(array, 'dtype', None)
        if dtype is None:
            dtype = np.result_type(*array)
        return is_real_dtype(dtype)

def _sanitize(text):
    """Return sanitized Eidos text field for human readability."""
    d = {'-LRB-': '(', '-RRB-': ')'}
    return re.sub('|'.join(d.keys()), lambda m: d[m.group(0)], text)

def movingaverage(arr, window):
    """
    Calculates the moving average ("rolling mean") of an array
    of a certain window size.
    """
    m = np.ones(int(window)) / int(window)
    return scipy.ndimage.convolve1d(arr, m, axis=0, mode='reflect')

def deprecate(func):
  """ A deprecation warning emmiter as a decorator. """
  @wraps(func)
  def wrapper(*args, **kwargs):
    warn("Deprecated, this will be removed in the future", DeprecationWarning)
    return func(*args, **kwargs)
  wrapper.__doc__ = "Deprecated.\n" + (wrapper.__doc__ or "")
  return wrapper

def is_int_vector(l):
    r"""Checks if l is a numpy array of integers

    """
    if isinstance(l, np.ndarray):
        if l.ndim == 1 and (l.dtype.kind == 'i' or l.dtype.kind == 'u'):
            return True
    return False

def _isstring(dtype):
    """Given a numpy dtype, determines whether it is a string. Returns True
    if the dtype is string or unicode.
    """
    return dtype.type == numpy.unicode_ or dtype.type == numpy.string_

def datetime64_to_datetime(dt):
    """ convert numpy's datetime64 to datetime """
    dt64 = np.datetime64(dt)
    ts = (dt64 - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
    return datetime.datetime.utcfromtimestamp(ts)

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

def normalize(v, axis=None, eps=1e-10):
  """L2 Normalize along specified axes."""
  return v / max(anorm(v, axis=axis, keepdims=True), eps)

def find_start_point(self):
        """
        Find the first location in our array that is not empty
        """
        for i, row in enumerate(self.data):
            for j, _ in enumerate(row):
                if self.data[i, j] != 0:  # or not np.isfinite(self.data[i,j]):
                    return i, j

def read_mm_header(fd, byte_order, dtype, count):
    """Read MM_HEADER tag from file and return as numpy.rec.array."""
    return numpy.rec.fromfile(fd, MM_HEADER, 1, byteorder=byte_order)[0]

def as_html(self):
        """Generate HTML to display map."""
        if not self._folium_map:
            self.draw()
        return self._inline_map(self._folium_map, self._width, self._height)

def Max(a, axis, keep_dims):
    """
    Max reduction op.
    """
    return np.amax(a, axis=axis if not isinstance(axis, np.ndarray) else tuple(axis),
                   keepdims=keep_dims),

def IsBinary(self, filename):
		"""Returns true if the guessed mimetyped isnt't in text group."""
		mimetype = mimetypes.guess_type(filename)[0]
		if not mimetype:
			return False  # e.g. README, "real" binaries usually have an extension
		# special case for text files which don't start with text/
		if mimetype in TEXT_MIMETYPES:
			return False
		return not mimetype.startswith("text/")

def read_numpy(fd, byte_order, dtype, count):
    """Read tag data from file and return as numpy array."""
    return numpy.fromfile(fd, byte_order+dtype[-1], count)

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

def fn_min(self, a, axis=None):
        """
        Return the minimum of an array, ignoring any NaNs.

        :param a: The array.
        :return: The minimum value of the array.
        """

        return numpy.nanmin(self._to_ndarray(a), axis=axis)

def clean_float(v):
    """Remove commas from a float"""

    if v is None or not str(v).strip():
        return None

    return float(str(v).replace(',', ''))

def _shuffle(data, idx):
    """Shuffle the data."""
    shuffle_data = []

    for idx_k, idx_v in data:
        shuffle_data.append((idx_k, mx.ndarray.array(idx_v.asnumpy()[idx], idx_v.context)))

    return shuffle_data

def to_distribution_values(self, values):
        """
        Returns numpy array of natural logarithms of ``values``.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # avoid RuntimeWarning: divide by zero encountered in log
            return numpy.log(values)

def rank(self):
        """how high in sorted list each key is. inverse permutation of sorter, such that sorted[rank]==keys"""
        r = np.empty(self.size, np.int)
        r[self.sorter] = np.arange(self.size)
        return r

def tokenize(string):
    """Match and yield all the tokens of the input string."""
    for match in TOKENS_REGEX.finditer(string):
        yield Token(match.lastgroup, match.group().strip(), match.span())

def name(self):
        """A unique name for this scraper."""
        return ''.join('_%s' % c if c.isupper() else c for c in self.__class__.__name__).strip('_').lower()

def tokenize(string):
    """Match and yield all the tokens of the input string."""
    for match in TOKENS_REGEX.finditer(string):
        yield Token(match.lastgroup, match.group().strip(), match.span())

def __copy__(self):
        """A magic method to implement shallow copy behavior."""
        return self.__class__.load(self.dump(), context=self.context)

def _to_lower_alpha_only(s):
    """Return a lowercased string with non alphabetic chars removed.

    White spaces are not to be removed."""
    s = re.sub(r'\n', ' ',  s.lower())
    return re.sub(r'[^a-z\s]', '', s)

def is_serializable(obj):
    """Return `True` if the given object conforms to the Serializable protocol.

    :rtype: bool
    """
    if inspect.isclass(obj):
      return Serializable.is_serializable_type(obj)
    return isinstance(obj, Serializable) or hasattr(obj, '_asdict')

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

def _is_date_data(self, data_type):
        """Private method for determining if a data record is of type DATE."""
        dt = DATA_TYPES[data_type]
        if isinstance(self.data, dt['type']):
            self.type = data_type.upper()
            self.len = None
            return True

def go_to_new_line(self):
        """Go to the end of the current line and create a new line"""
        self.stdkey_end(False, False)
        self.insert_text(self.get_line_separator())

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

def seq_include(records, filter_regex):
    """
    Filter any sequences who's seq does not match the filter. Ignore case.
    """
    regex = re.compile(filter_regex)
    for record in records:
        if regex.search(str(record.seq)):
            yield record

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

def parallel(processes, threads):
    """
    execute jobs in processes using N threads
    """
    pool = multithread(threads)
    pool.map(run_process, processes)
    pool.close()
    pool.join()

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

def open_with_encoding(filename, encoding, mode='r'):
    """Return opened file with a specific encoding."""
    return io.open(filename, mode=mode, encoding=encoding,
                   newline='')

def save_dot(self, fd):
        """ Saves a representation of the case in the Graphviz DOT language.
        """
        from pylon.io import DotWriter
        DotWriter(self).write(fd)

def fopenat(base_fd, path):
    """
    Does openat read-only, then does fdopen to get a file object
    """

    return os.fdopen(openat(base_fd, path, os.O_RDONLY), 'rb')

def list_i2str(ilist):
    """
    Convert an integer list into a string list.
    """
    slist = []
    for el in ilist:
        slist.append(str(el))
    return slist

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

def tokenize_words(self, text):
        """Tokenize an input string into a list of words (with punctuation removed)."""
        return [
            self.strip_punctuation(word) for word in text.split(' ')
            if self.strip_punctuation(word)
        ]

def getbyteslice(self, start, end):
        """Direct access to byte data."""
        c = self._rawarray[start:end]
        return c

def fit_gaussian(x, y, yerr, p0):
    """ Fit a Gaussian to the data """
    try:
        popt, pcov = curve_fit(gaussian, x, y, sigma=yerr, p0=p0, absolute_sigma=True)
    except RuntimeError:
        return [0],[0]
    return popt, pcov

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

def flatten(l):
    """Flatten a nested list."""
    return sum(map(flatten, l), []) \
        if isinstance(l, list) or isinstance(l, tuple) else [l]

def zoom_cv(x,z):
    """ Zoom the center of image x by a factor of z+1 while retaining the original image size and proportion. """
    if z==0: return x
    r,c,*_ = x.shape
    M = cv2.getRotationMatrix2D((c/2,r/2),0,z+1.)
    return cv2.warpAffine(x,M,(c,r))

def warp(self, warp_matrix, img, iflag=cv2.INTER_NEAREST):
        """ Function to warp input image given an estimated 2D linear transformation

        :param warp_matrix: Linear 2x3 matrix to use to linearly warp the input images
        :type warp_matrix: ndarray
        :param img: Image to be warped with estimated transformation
        :type img: ndarray
        :param iflag: Interpolation flag, specified interpolation using during resampling of warped image
        :type iflag: cv2.INTER_*
        :return: Warped image using the linear matrix
        """

        height, width = img.shape[:2]
        warped_img = np.zeros_like(img, dtype=img.dtype)

        # Check if image to warp is 2D or 3D. If 3D need to loop over channels
        if (self.interpolation_type == InterpolationType.LINEAR) or img.ndim == 2:
            warped_img = cv2.warpAffine(img.astype(np.float32), warp_matrix, (width, height),
                                        flags=iflag).astype(img.dtype)

        elif img.ndim == 3:
            for idx in range(img.shape[-1]):
                warped_img[..., idx] = cv2.warpAffine(img[..., idx].astype(np.float32), warp_matrix, (width, height),
                                                      flags=iflag).astype(img.dtype)
        else:
            raise ValueError('Image has incorrect number of dimensions: {}'.format(img.ndim))

        return warped_img

def file_read(filename):
    """Read a file and close it.  Returns the file source."""
    fobj = open(filename,'r');
    source = fobj.read();
    fobj.close()
    return source

def set_stop_handler(self):
        """
        Initializes functions that are invoked when the user or OS wants to kill this process.
        :return:
        """
        signal.signal(signal.SIGTERM, self.graceful_stop)
        signal.signal(signal.SIGABRT, self.graceful_stop)
        signal.signal(signal.SIGINT, self.graceful_stop)

def many_until1(these, term):
    """Like many_until but must consume at least one of these.
    """
    first = [these()]
    these_results, term_result = many_until(these, term)
    return (first + these_results, term_result)

def camel_case(self, snake_case):
        """ Convert snake case to camel case """
        components = snake_case.split('_')
        return components[0] + "".join(x.title() for x in components[1:])

def help(self, level=0):
        """return the usage string for available options """
        self.cmdline_parser.formatter.output_level = level
        with _patch_optparse():
            return self.cmdline_parser.format_help()

def unit_key_from_name(name):
  """Return a legal python name for the given name for use as a unit key."""
  result = name

  for old, new in six.iteritems(UNIT_KEY_REPLACEMENTS):
    result = result.replace(old, new)

  # Collapse redundant underscores and convert to uppercase.
  result = re.sub(r'_+', '_', result.upper())

  return result

def column_names(self, table):
      """An iterable of column names, for a particular table or
      view."""

      table_info = self.execute(
        u'PRAGMA table_info(%s)' % quote(table))
      return (column['name'] for column in table_info)

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

def money(min=0, max=10):
    """Return a str of decimal with two digits after a decimal mark."""
    value = random.choice(range(min * 100, max * 100))
    return "%1.2f" % (float(value) / 100)

def truncate(self, table):
        """Empty a table by deleting all of its rows."""
        if isinstance(table, (list, set, tuple)):
            for t in table:
                self._truncate(t)
        else:
            self._truncate(table)

def wget(url):
    """
    Download the page into a string
    """
    import urllib.parse
    request = urllib.request.urlopen(url)
    filestring = request.read()
    return filestring

def out_shape_from_array(arr):
    """Get the output shape from an array."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.shape
    else:
        return (arr.shape[1],)

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

def redirect_output(fileobj):
    """Redirect standard out to file."""
    old = sys.stdout
    sys.stdout = fileobj
    try:
        yield fileobj
    finally:
        sys.stdout = old

def variance(arr):
  """variance of the values, must have 2 or more entries.

  :param arr: list of numbers
  :type arr: number[] a number array
  :return: variance
  :rtype: float

  """
  avg = average(arr)
  return sum([(float(x)-avg)**2 for x in arr])/float(len(arr)-1)

def requests_request(method, url, **kwargs):
    """Requests-mock requests.request wrapper."""
    session = local_sessions.session
    response = session.request(method=method, url=url, **kwargs)
    session.close()
    return response

def column_names(self, table):
      """An iterable of column names, for a particular table or
      view."""

      table_info = self.execute(
        u'PRAGMA table_info(%s)' % quote(table))
      return (column['name'] for column in table_info)

def autopage(self):
        """Iterate through results from all pages.

        :return: all results
        :rtype: generator
        """
        while self.items:
            yield from self.items
            self.items = self.fetch_next()

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

def parse(self):
        """
        Parse file specified by constructor.
        """
        f = open(self.parse_log_path, "r")
        self.parse2(f)
        f.close()

def find(command, on):
    """Find the command usage."""
    output_lines = parse_man_page(command, on)
    click.echo(''.join(output_lines))

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

def distinct(xs):
    """Get the list of distinct values with preserving order."""
    # don't use collections.OrderedDict because we do support Python 2.6
    seen = set()
    return [x for x in xs if x not in seen and not seen.add(x)]

def FromString(self, string):
    """Parse a bool from a string."""
    if string.lower() in ("false", "no", "n"):
      return False

    if string.lower() in ("true", "yes", "y"):
      return True

    raise TypeValueError("%s is not recognized as a boolean value." % string)

def getScriptLocation():
	"""Helper function to get the location of a Python file."""
	location = os.path.abspath("./")
	if __file__.rfind("/") != -1:
		location = __file__[:__file__.rfind("/")]
	return location

def to_datetime(value):
    """Converts a string to a datetime."""
    if value is None:
        return None

    if isinstance(value, six.integer_types):
        return parser.parse(value)
    return parser.isoparse(value)

def rel_path(filename):
    """
    Function that gets relative path to the filename
    """
    return os.path.join(os.getcwd(), os.path.dirname(__file__), filename)

def parse_datetime(dt_str):
    """Parse datetime."""
    date_format = "%Y-%m-%dT%H:%M:%S %z"
    dt_str = dt_str.replace("Z", " +0000")
    return datetime.datetime.strptime(dt_str, date_format)

def word_to_id(self, word):
        """Returns the integer word id of a word string."""
        if word in self.vocab:
            return self.vocab[word]
        else:
            return self.unk_id

def updateFromKwargs(self, properties, kwargs, collector, **unused):
        """Primary entry point to turn 'kwargs' into 'properties'"""
        properties[self.name] = self.getFromKwargs(kwargs)

def split(s):
  """Uses dynamic programming to infer the location of spaces in a string without spaces."""
  l = [_split(x) for x in _SPLIT_RE.split(s)]
  return [item for sublist in l for item in sublist]

def run(args):
    """Process command line arguments and walk inputs."""
    raw_arguments = get_arguments(args[1:])
    process_arguments(raw_arguments)
    walk.run()
    return True

def reload_localzone():
    """Reload the cached localzone. You need to call this if the timezone has changed."""
    global _cache_tz
    _cache_tz = pytz.timezone(get_localzone_name())
    utils.assert_tz_offset(_cache_tz)
    return _cache_tz

def OnPasteAs(self, event):
        """Clipboard paste as event handler"""

        data = self.main_window.clipboard.get_clipboard()
        key = self.main_window.grid.actions.cursor

        with undo.group(_("Paste As...")):
            self.main_window.actions.paste_as(key, data)

        self.main_window.grid.ForceRefresh()

        event.Skip()

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

def get_parent_folder_name(file_path):
    """Finds parent folder of file

    :param file_path: path
    :return: Name of folder container
    """
    return os.path.split(os.path.split(os.path.abspath(file_path))[0])[-1]

def datetime_to_ms(dt):
    """
    Converts a datetime to a millisecond accuracy timestamp
    """
    seconds = calendar.timegm(dt.utctimetuple())
    return seconds * 1000 + int(dt.microsecond / 1000)

def pause(self):
        """Pause the music"""
        mixer.music.pause()
        self.pause_time = self.get_time()
        self.paused = True

def OnMove(self, event):
        """Main window move event"""

        # Store window position in config
        position = self.main_window.GetScreenPositionTuple()

        config["window_position"] = repr(position)

def set_trace():
    """Start a Pdb instance at the calling frame, with stdout routed to sys.__stdout__."""
    # https://github.com/nose-devs/nose/blob/master/nose/tools/nontrivial.py
    pdb.Pdb(stdout=sys.__stdout__).set_trace(sys._getframe().f_back)

def shape(self):
        """Compute the shape of the dataset as (rows, cols)."""
        if not self.data:
            return (0, 0)
        return (len(self.data), len(self.dimensions))

def dimensions(self):
        """Get width and height of a PDF"""
        size = self.pdf.getPage(0).mediaBox
        return {'w': float(size[2]), 'h': float(size[3])}

def _num_cpus_darwin():
    """Return the number of active CPUs on a Darwin system."""
    p = subprocess.Popen(['sysctl','-n','hw.ncpu'],stdout=subprocess.PIPE)
    return p.stdout.read()

def delete_entry(self, key):
        """Delete an object from the redis table"""
        pipe = self.client.pipeline()
        pipe.srem(self.keys_container, key)
        pipe.delete(key)
        pipe.execute()

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

def circ_permutation(items):
    """Calculate the circular permutation for a given list of items."""
    permutations = []
    for i in range(len(items)):
        permutations.append(items[i:] + items[:i])
    return permutations

def call_and_exit(self, cmd, shell=True):
        """Run the *cmd* and exit with the proper exit code."""
        sys.exit(subprocess.call(cmd, shell=shell))

def getoutput_pexpect(self, cmd):
        """Run a command and return its stdout/stderr as a string.

        Parameters
        ----------
        cmd : str
          A command to be executed in the system shell.

        Returns
        -------
        output : str
          A string containing the combination of stdout and stderr from the
        subprocess, in whatever order the subprocess originally wrote to its
        file descriptors (so the order of the information in this string is the
        correct order as would be seen if running the command in a terminal).
        """
        try:
            return pexpect.run(self.sh, args=['-c', cmd]).replace('\r\n', '\n')
        except KeyboardInterrupt:
            print('^C', file=sys.stderr, end='')

def get_stripped_file_lines(filename):
    """
    Return lines of a file with whitespace removed
    """
    try:
        lines = open(filename).readlines()
    except FileNotFoundError:
        fatal("Could not open file: {!r}".format(filename))

    return [line.strip() for line in lines]

def getoutput_pexpect(self, cmd):
        """Run a command and return its stdout/stderr as a string.

        Parameters
        ----------
        cmd : str
          A command to be executed in the system shell.

        Returns
        -------
        output : str
          A string containing the combination of stdout and stderr from the
        subprocess, in whatever order the subprocess originally wrote to its
        file descriptors (so the order of the information in this string is the
        correct order as would be seen if running the command in a terminal).
        """
        try:
            return pexpect.run(self.sh, args=['-c', cmd]).replace('\r\n', '\n')
        except KeyboardInterrupt:
            print('^C', file=sys.stderr, end='')

def _get_str_columns(sf):
    """
    Returns a list of names of columns that are string type.
    """
    return [name for name in sf.column_names() if sf[name].dtype == str]

def unpickle(pickle_file):
    """Unpickle a python object from the given path."""
    pickle = None
    with open(pickle_file, "rb") as pickle_f:
        pickle = dill.load(pickle_f)
    if not pickle:
        LOG.error("Could not load python object from file")
    return pickle

def angle(x0, y0, x1, y1):
    """ Returns the angle between two points.
    """
    return degrees(atan2(y1-y0, x1-x0))

def load(self, filename='classifier.dump'):
        """
        Unpickles the classifier used
        """
        ifile = open(filename, 'r+')
        self.classifier = pickle.load(ifile)
        ifile.close()

def get():
    """ Get local facts about this machine.

    Returns:
        json-compatible dict with all facts of this host
    """
    result = runCommand('facter --json', raise_error_on_fail=True)
    json_facts = result[1]
    facts = json.loads(json_facts)
    return facts

def highlight_region(plt, start_x, end_x):
  """
  Highlight a region on the chart between the specified start and end x-co-ordinates.
  param pyplot plt: matplotlibk pyplot which contains the charts to be highlighted
  param string start_x : epoch time millis
  param string end_x : epoch time millis
  """
  start_x = convert_to_mdate(start_x)
  end_x = convert_to_mdate(end_x)
  plt.axvspan(start_x, end_x, color=CONSTANTS.HIGHLIGHT_COLOR, alpha=CONSTANTS.HIGHLIGHT_ALPHA)

def load_search_freq(fp=SEARCH_FREQ_JSON):
    """
    Load the search_freq from JSON file
    """
    try:
        with open(fp) as f:
            return Counter(json.load(f))
    except FileNotFoundError:
        return Counter()

def barv(d, plt, title=None, rotation='vertical'):
    """A convenience function for plotting a vertical bar plot from a Counter"""
    labels = sorted(d, key=d.get, reverse=True)
    index = range(len(labels))
    plt.xticks(index, labels, rotation=rotation)
    plt.bar(index, [d[v] for v in labels])

    if title is not None:
        plt.title(title)

def datatype(dbtype, description, cursor):
    """Google AppEngine Helper to convert a data type into a string."""
    dt = cursor.db.introspection.get_field_type(dbtype, description)
    if type(dt) is tuple:
        return dt[0]
    else:
        return dt

def plot_target(target, ax):
    """Ajoute la target au plot"""
    ax.scatter(target[0], target[1], target[2], c="red", s=80)

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

def blocking(func, *args, **kwargs):
    """Run a function that uses blocking IO.

    The function is run in the IO thread pool.
    """
    pool = get_io_pool()
    fut = pool.submit(func, *args, **kwargs)
    return fut.result()

def get_longest_orf(orfs):
    """Find longest ORF from the given list of ORFs."""
    sorted_orf = sorted(orfs, key=lambda x: len(x['sequence']), reverse=True)[0]
    return sorted_orf

def disown(cmd):
    """Call a system command in the background,
       disown it and hide it's output."""
    subprocess.Popen(cmd,
                     stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)

def _next_token(self, skipws=True):
        """Increment _token to the next token and return it."""
        self._token = next(self._tokens).group(0)
        return self._next_token() if skipws and self._token.isspace() else self._token

def disown(cmd):
    """Call a system command in the background,
       disown it and hide it's output."""
    subprocess.Popen(cmd,
                     stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)

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

def init_db():
    """
    Drops and re-creates the SQL schema
    """
    db.drop_all()
    db.configure_mappers()
    db.create_all()
    db.session.commit()

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

def Print(self, output_writer):
    """Prints a human readable version of the filter.

    Args:
      output_writer (CLIOutputWriter): output writer.
    """
    if self._filters:
      output_writer.Write('Filters:\n')
      for file_entry_filter in self._filters:
        file_entry_filter.Print(output_writer)

def _get_minidom_tag_value(station, tag_name):
    """get a value from a tag (if it exists)"""
    tag = station.getElementsByTagName(tag_name)[0].firstChild
    if tag:
        return tag.nodeValue

    return None

def print_param_values(self_):
        """Print the values of all this object's Parameters."""
        self = self_.self
        for name,val in self.param.get_param_values():
            print('%s.%s = %s' % (self.name,name,val))

def title(self):
        """ The title of this window """
        with switch_window(self._browser, self.name):
            return self._browser.title

def runcode(code):
	"""Run the given code line by line with printing, as list of lines, and return variable 'ans'."""
	for line in code:
		print('# '+line)
		exec(line,globals())
	print('# return ans')
	return ans

def created_today(self):
        """Return True if created today."""
        if self.datetime.date() == datetime.today().date():
            return True
        return False

def __call__(self, _):
        """Print the current iteration."""
        if self.iter % self.step == 0:
            print(self.fmt.format(self.iter), **self.kwargs)

        self.iter += 1

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

def Print(self):
        """Prints the values and freqs/probs in ascending order."""
        for val, prob in sorted(self.d.iteritems()):
            print(val, prob)

def camel_case_from_underscores(string):
    """generate a CamelCase string from an underscore_string."""
    components = string.split('_')
    string = ''
    for component in components:
        string += component[0].upper() + component[1:]
    return string

def raw_print(*args, **kw):
    """Raw print to sys.__stdout__, otherwise identical interface to print()."""

    print(*args, sep=kw.get('sep', ' '), end=kw.get('end', '\n'),
          file=sys.__stdout__)
    sys.__stdout__.flush()

def text_width(string, font_name, font_size):
    """Determine with width in pixels of string."""
    return stringWidth(string, fontName=font_name, fontSize=font_size)

def indented_show(text, howmany=1):
        """Print a formatted indented text.
        """
        print(StrTemplate.pad_indent(text=text, howmany=howmany))

def assign_to(self, obj):
    """Assign `x` and `y` to an object that has properties `x` and `y`."""
    obj.x = self.x
    obj.y = self.y

def print_bintree(tree, indent='  '):
    """print a binary tree"""
    for n in sorted(tree.keys()):
        print "%s%s" % (indent * depth(n,tree), n)

def plot_target(target, ax):
    """Ajoute la target au plot"""
    ax.scatter(target[0], target[1], target[2], c="red", s=80)

def save_dot(self, fd):
        """ Saves a representation of the case in the Graphviz DOT language.
        """
        from pylon.io import DotWriter
        DotWriter(self).write(fd)

def _comment(string):
    """return string as a comment"""
    lines = [line.strip() for line in string.splitlines()]
    return "# " + ("%s# " % linesep).join(lines)

def _display(self, layout):
        """launch layouts display"""
        print(file=self.out)
        TextWriter().format(layout, self.out)

def parse_cookies(self, req, name, field):
        """Pull the value from the cookiejar."""
        return core.get_value(req.COOKIES, name, field)

def run(args):
    """Process command line arguments and walk inputs."""
    raw_arguments = get_arguments(args[1:])
    process_arguments(raw_arguments)
    walk.run()
    return True

def hard_equals(a, b):
    """Implements the '===' operator."""
    if type(a) != type(b):
        return False
    return a == b

def compute_capture(args):
    x, y, w, h, params = args
    """Callable function for the multiprocessing pool."""
    return x, y, mandelbrot_capture(x, y, w, h, params)

def _comment(string):
    """return string as a comment"""
    lines = [line.strip() for line in string.splitlines()]
    return "# " + ("%s# " % linesep).join(lines)

def get_free_memory_win():
    """Return current free memory on the machine for windows.

    Warning : this script is really not robust
    Return in MB unit
    """
    stat = MEMORYSTATUSEX()
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return int(stat.ullAvailPhys / 1024 / 1024)

def cycle_focus(self):
        """
        Cycle through all windows.
        """
        windows = self.windows()
        new_index = (windows.index(self.active_window) + 1) % len(windows)
        self.active_window = windows[new_index]

def __call__(self, _):
        """Update the progressbar."""
        if self.iter % self.step == 0:
            self.pbar.update(self.step)

        self.iter += 1

def int32_to_negative(int32):
    """Checks if a suspicious number (e.g. ligand position) is in fact a negative number represented as a
    32 bit integer and returns the actual number.
    """
    dct = {}
    if int32 == 4294967295:  # Special case in some structures (note, this is just a workaround)
        return -1
    for i in range(-1000, -1):
        dct[np.uint32(i)] = i
    if int32 in dct:
        return dct[int32]
    else:
        return int32

def value(self):
        """Value of property."""
        if self._prop.fget is None:
            raise AttributeError('Unable to read attribute')
        return self._prop.fget(self._obj)

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

def acknowledge_time(self):
        """
        Processor time when the alarm was acknowledged.

        :type: :class:`~datetime.datetime`
        """
        if (self.is_acknowledged and
                self._proto.acknowledgeInfo.HasField('acknowledgeTime')):
            return parse_isostring(self._proto.acknowledgeInfo.acknowledgeTime)
        return None

def get_order(self):
        """
        Return a list of dicionaries. See `set_order`.
        """
        return [dict(reverse=r[0], key=r[1]) for r in self.get_model()]

def _prepare_proxy(self, conn):
        """
        Establish tunnel connection early, because otherwise httplib
        would improperly set Host: header to proxy's IP:port.
        """
        conn.set_tunnel(self._proxy_host, self.port, self.proxy_headers)
        conn.connect()

def use_theme(theme):
    """Make the given theme current.

    There are two included themes: light_theme, dark_theme.
    """
    global current
    current = theme
    import scene
    if scene.current is not None:
        scene.current.stylize()

def packagenameify(s):
  """
  Makes a package name
  """
  return ''.join(w if w in ACRONYMS else w.title() for w in s.split('.')[-1:])

def getpass(self, prompt, default=None):
        """Provide a password prompt."""
        return click.prompt(prompt, hide_input=True, default=default)

def store_many(self, sql, values):
        """Abstraction over executemany method"""
        cursor = self.get_cursor()
        cursor.executemany(sql, values)
        self.conn.commit()

def disown(cmd):
    """Call a system command in the background,
       disown it and hide it's output."""
    subprocess.Popen(cmd,
                     stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)

def _single_page_pdf(page):
    """Construct a single page PDF from the provided page in memory"""
    pdf = Pdf.new()
    pdf.pages.append(page)
    bio = BytesIO()
    pdf.save(bio)
    bio.seek(0)
    return bio.read()

def hide(self):
        """Hides the main window of the terminal and sets the visible
        flag to False.
        """
        if not HidePrevention(self.window).may_hide():
            return
        self.hidden = True
        self.get_widget('window-root').unstick()
        self.window.hide()

def set_ylimits(self, row, column, min=None, max=None):
        """Set y-axis limits of a subplot.

        :param row,column: specify the subplot.
        :param min: minimal axis value
        :param max: maximum axis value

        """
        subplot = self.get_subplot_at(row, column)
        subplot.set_ylimits(min, max)

def isBlockComment(self, line, column):
        """Check if text at given position is a block comment.

        If language is not known, or text is not parsed yet, ``False`` is returned
        """
        return self._highlighter is not None and \
               self._highlighter.isBlockComment(self.document().findBlockByNumber(line), column)

def handle_qbytearray(obj, encoding):
    """Qt/Python2/3 compatibility helper."""
    if isinstance(obj, QByteArray):
        obj = obj.data()

    return to_text_string(obj, encoding=encoding)

def _get_line_no_from_comments(py_line):
    """Return the line number parsed from the comment or 0."""
    matched = LINECOL_COMMENT_RE.match(py_line)
    if matched:
        return int(matched.group(1))
    else:
        return 0

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

def __enter__(self):
        """ Implements the context manager protocol. Specially useful for asserting exceptions
        """
        clone = self.clone()
        self._contexts.append(clone)
        self.reset()
        return self

def dict_to_querystring(dictionary):
    """Converts a dict to a querystring suitable to be appended to a URL."""
    s = u""
    for d in dictionary.keys():
        s = unicode.format(u"{0}{1}={2}&", s, d, dictionary[d])
    return s[:-1]

def __call__(self, _):
        """Update the progressbar."""
        if self.iter % self.step == 0:
            self.pbar.update(self.step)

        self.iter += 1

def is_nullable_list(val, vtype):
    """Return True if list contains either values of type `vtype` or None."""
    return (isinstance(val, list) and
            any(isinstance(v, vtype) for v in val) and
            all((isinstance(v, vtype) or v is None) for v in val))

def set_font_size(self, size):
        """Convenience method for just changing font size."""
        if self.font.font_size == size:
            pass
        else:
            self.font._set_size(size)

def wait_until_exit(self):
        """ Wait until all the threads are finished.

        """
        [t.join() for t in self.threads]

        self.threads = list()

def intersect(d1, d2):
    """Intersect dictionaries d1 and d2 by key *and* value."""
    return dict((k, d1[k]) for k in d1 if k in d2 and d1[k] == d2[k])

def timespan(start_time):
    """Return time in milliseconds from start_time"""

    timespan = datetime.datetime.now() - start_time
    timespan_ms = timespan.total_seconds() * 1000
    return timespan_ms

def directory_files(path):
    """Yield directory file names."""

    for entry in os.scandir(path):
        if not entry.name.startswith('.') and entry.is_file():
            yield entry.name

def get_range(self, start=None, stop=None):
		"""Return a RangeMap for the range start to stop.

		Returns:
			A RangeMap
		"""
		return self.from_iterable(self.ranges(start, stop))

def safe_exit(output):
    """exit without breaking pipes."""
    try:
        sys.stdout.write(output)
        sys.stdout.flush()
    except IOError:
        pass

def LinSpace(start, stop, num):
    """
    Linspace op.
    """
    return np.linspace(start, stop, num=num, dtype=np.float32),

def ensure_hbounds(self):
        """Ensure the cursor is within horizontal screen bounds."""
        self.cursor.x = min(max(0, self.cursor.x), self.columns - 1)

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

def close(self):
        """Close child subprocess"""
        if self._subprocess is not None:
            os.killpg(self._subprocess.pid, signal.SIGTERM)
            self._subprocess = None

def split_comment(cls, code):
        """ Removes comments (#...) from python code. """
        if '#' not in code: return code
        #: Remove comments only (leave quoted strings as they are)
        subf = lambda m: '' if m.group(0)[0]=='#' else m.group(0)
        return re.sub(cls.re_pytokens, subf, code)

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

def rAsciiLine(ifile):
    """Returns the next non-blank line in an ASCII file."""

    _line = ifile.readline().strip()
    while len(_line) == 0:
        _line = ifile.readline().strip()
    return _line

def import_js(path, lib_name, globals):
    """Imports from javascript source file.
      globals is your globals()"""
    with codecs.open(path_as_local(path), "r", "utf-8") as f:
        js = f.read()
    e = EvalJs()
    e.execute(js)
    var = e.context['var']
    globals[lib_name] = var.to_python()

def read_large_int(self, bits, signed=True):
        """Reads a n-bits long integer value."""
        return int.from_bytes(
            self.read(bits // 8), byteorder='little', signed=signed)

def scale_image(image, new_width):
    """Resizes an image preserving the aspect ratio.
    """
    (original_width, original_height) = image.size
    aspect_ratio = original_height/float(original_width)
    new_height = int(aspect_ratio * new_width)

    # This scales it wider than tall, since characters are biased
    new_image = image.resize((new_width*2, new_height))
    return new_image

def readme():
    """Try converting the README to an RST document. Return it as is on failure."""
    try:
        import pypandoc
        readme_content = pypandoc.convert('README.md', 'rst')
    except(IOError, ImportError):
        print("Warning: no pypandoc module found.")
        try:
            readme_content = open('README.md').read()
        except IOError:
            readme_content = ''
    return readme_content

def _pad(self, text):
        """Pad the text."""
        top_bottom = ("\n" * self._padding) + " "
        right_left = " " * self._padding * self.PAD_WIDTH
        return top_bottom + right_left + text + right_left + top_bottom

def _fast_read(self, infile):
        """Function for fast reading from sensor files."""
        infile.seek(0)
        return(int(infile.read().decode().strip()))

def indented_show(text, howmany=1):
        """Print a formatted indented text.
        """
        print(StrTemplate.pad_indent(text=text, howmany=howmany))

def _read_json_file(self, json_file):
        """ Helper function to read JSON file as OrderedDict """

        self.log.debug("Reading '%s' JSON file..." % json_file)

        with open(json_file, 'r') as f:
            return json.load(f, object_pairs_hook=OrderedDict)

def _basic_field_data(field, obj):
    """Returns ``obj.field`` data as a dict"""
    value = field.value_from_object(obj)
    return {Field.TYPE: FieldType.VAL, Field.VALUE: value}

def get_list_from_file(file_name):
    """read the lines from a file into a list"""
    with open(file_name, mode='r', encoding='utf-8') as f1:
        lst = f1.readlines()
    return lst

def kick(self, channel, nick, comment=""):
        """Send a KICK command."""
        self.send_items('KICK', channel, nick, comment and ':' + comment)

def standard_input():
    """Generator that yields lines from standard input."""
    with click.get_text_stream("stdin") as stdin:
        while stdin.readable():
            line = stdin.readline()
            if line:
                yield line.strip().encode("utf-8")

def chunk_list(l, n):
    """Return `n` size lists from a given list `l`"""
    return [l[i:i + n] for i in range(0, len(l), n)]

def url_read_text(url, verbose=True):
    r"""
    Directly reads text data from url
    """
    data = url_read(url, verbose)
    text = data.decode('utf8')
    return text

def group_by(iterable, key_func):
    """Wrap itertools.groupby to make life easier."""
    groups = (
        list(sub) for key, sub in groupby(iterable, key_func)
    )
    return zip(groups, groups)

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

def rnormal(mu, tau, size=None):
    """
    Random normal variates.
    """
    return np.random.normal(mu, 1. / np.sqrt(tau), size)

def read_data(file, endian, num=1):
    """
    Read a given number of 32-bits unsigned integers from the given file
    with the given endianness.
    """
    res = struct.unpack(endian + 'L' * num, file.read(num * 4))
    if len(res) == 1:
        return res[0]
    return res

def counter_from_str(self, string):
        """Build word frequency list from incoming string."""
        string_list = [chars for chars in string if chars not in self.punctuation]
        string_joined = ''.join(string_list)
        tokens = self.punkt.word_tokenize(string_joined)
        return Counter(tokens)

def get_stripped_file_lines(filename):
    """
    Return lines of a file with whitespace removed
    """
    try:
        lines = open(filename).readlines()
    except FileNotFoundError:
        fatal("Could not open file: {!r}".format(filename))

    return [line.strip() for line in lines]

def safe_exit(output):
    """exit without breaking pipes."""
    try:
        sys.stdout.write(output)
        sys.stdout.flush()
    except IOError:
        pass

def _multiline_width(multiline_s, line_width_fn=len):
    """Visible width of a potentially multiline content."""
    return max(map(line_width_fn, re.split("[\r\n]", multiline_s)))

def do_restart(self, line):
        """Request that the Outstation perform a cold restart. Command syntax is: restart"""
        self.application.master.Restart(opendnp3.RestartType.COLD, restart_callback)

def __setitem__(self, field, value):
        """ :see::meth:RedisMap.__setitem__ """
        return self._client.hset(self.key_prefix, field, self._dumps(value))

def downcaseTokens(s,l,t):
    """Helper parse action to convert tokens to lower case."""
    return [ tt.lower() for tt in map(_ustr,t) ]

def __setitem__(self, field, value):
        """ :see::meth:RedisMap.__setitem__ """
        return self._client.hset(self.key_prefix, field, self._dumps(value))

def algo_exp(x, m, t, b):
    """mono-exponential curve."""
    return m*np.exp(-t*x)+b

def hstrlen(self, name, key):
        """
        Return the number of bytes stored in the value of ``key``
        within hash ``name``
        """
        with self.pipe as pipe:
            return pipe.hstrlen(self.redis_key(name), key)

def get_default_preds():
    """dynamically build autocomplete options based on an external file"""
    g = ontospy.Ontospy(rdfsschema, text=True, verbose=False, hide_base_schemas=False)
    classes = [(x.qname, x.bestDescription()) for x in g.all_classes]
    properties = [(x.qname, x.bestDescription()) for x in g.all_properties]
    commands = [('exit', 'exits the terminal'), ('show', 'show current buffer')]
    return rdfschema + owlschema + classes + properties + commands

def exit(self):
        """
        Closes the connection
        """
        self.pubsub.unsubscribe()
        self.client.connection_pool.disconnect()

        logger.info("Connection to Redis closed")

def changed(self):
        """Returns dict of fields that changed since save (with old values)"""
        return dict(
            (field, self.previous(field))
            for field in self.fields
            if self.has_changed(field)
        )

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

def combinations(l):
    """Pure-Python implementation of itertools.combinations(l, 2)."""
    result = []
    for x in xrange(len(l) - 1):
        ls = l[x + 1:]
        for y in ls:
            result.append((l[x], y))
    return result

def append_pdf(input_pdf: bytes, output_writer: PdfFileWriter):
    """
    Appends a PDF to a pyPDF writer. Legacy interface.
    """
    append_memory_pdf_to_writer(input_pdf=input_pdf,
                                writer=output_writer)

def rank(idx, dim):
    """Calculate the index rank according to Bertran's notation."""
    idxm = multi_index(idx, dim)
    out = 0
    while idxm[-1:] == (0,):
        out += 1
        idxm = idxm[:-1]
    return out

def extract_table_names(query):
    """ Extract table names from an SQL query. """
    # a good old fashioned regex. turns out this worked better than actually parsing the code
    tables_blocks = re.findall(r'(?:FROM|JOIN)\s+(\w+(?:\s*,\s*\w+)*)', query, re.IGNORECASE)
    tables = [tbl
              for block in tables_blocks
              for tbl in re.findall(r'\w+', block)]
    return set(tables)

def _npiter(arr):
    """Wrapper for iterating numpy array"""
    for a in np.nditer(arr, flags=["refs_ok"]):
        c = a.item()
        if c is not None:
            yield c

def __absolute__(self, uri):
        """ Get the absolute uri for a file

        :param uri: URI of the resource to be retrieved
        :return: Absolute Path
        """
        return op.abspath(op.join(self.__path__, uri))

def good(txt):
    """Print, emphasized 'good', the given 'txt' message"""

    print("%s# %s%s%s" % (PR_GOOD_CC, get_time_stamp(), txt, PR_NC))
    sys.stdout.flush()

def pages(self):
        """Get pages, reloading the site if needed."""
        rev = self.db.get('site:rev')
        if int(rev) != self.revision:
            self.reload_site()

        return self._pages

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

def uint32_to_uint8(cls, img):
        """
        Cast uint32 RGB image to 4 uint8 channels.
        """
        return np.flipud(img.view(dtype=np.uint8).reshape(img.shape + (4,)))

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

def strip_accents(s):
    """
    Strip accents to prepare for slugification.
    """
    nfkd = unicodedata.normalize('NFKD', unicode(s))
    return u''.join(ch for ch in nfkd if not unicodedata.combining(ch))

def min_values(args):
    """ Return possible range for min function. """
    return Interval(min(x.low for x in args), min(x.high for x in args))

def unpunctuate(s, *, char_blacklist=string.punctuation):
    """ Remove punctuation from string s. """
    # remove punctuation
    s = "".join(c for c in s if c not in char_blacklist)
    # remove consecutive spaces
    return " ".join(filter(None, s.split(" ")))

def ensure_hbounds(self):
        """Ensure the cursor is within horizontal screen bounds."""
        self.cursor.x = min(max(0, self.cursor.x), self.columns - 1)

def remove_legend(ax=None):
    """Remove legend for axes or gca.

    See http://osdir.com/ml/python.matplotlib.general/2005-07/msg00285.html
    """
    from pylab import gca, draw
    if ax is None:
        ax = gca()
    ax.legend_ = None
    draw()

def timed (log=sys.stderr, limit=2.0):
    """Decorator to run a function with timing info."""
    return lambda func: timeit(func, log, limit)

def strip_spaces(value, sep=None, join=True):
    """Cleans trailing whitespaces and replaces also multiple whitespaces with a single space."""
    value = value.strip()
    value = [v.strip() for v in value.split(sep)]
    join_sep = sep or ' '
    return join_sep.join(value) if join else value

def classnameify(s):
  """
  Makes a classname
  """
  return ''.join(w if w in ACRONYMS else w.title() for w in s.split('_'))

def __normalize_list(self, msg):
        """Split message to list by commas and trim whitespace."""
        if isinstance(msg, list):
            msg = "".join(msg)
        return list(map(lambda x: x.strip(), msg.split(",")))

def multiply(self, number):
        """Return a Vector as the product of the vector and a real number."""
        return self.from_list([x * number for x in self.to_list()])

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

def get_cube(name):
    """ Load the named cube from the current registered ``CubeManager``. """
    manager = get_manager()
    if not manager.has_cube(name):
        raise NotFound('No such cube: %r' % name)
    return manager.get_cube(name)

def remove_dups(seq):
    """remove duplicates from a sequence, preserving order"""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def mark(self, lineno, count=1):
        """Mark a given source line as executed count times.

        Multiple calls to mark for the same lineno add up.
        """
        self.sourcelines[lineno] = self.sourcelines.get(lineno, 0) + count

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

def remove_na_arraylike(arr):
    """
    Return array-like containing only true/non-NaN values, possibly empty.
    """
    if is_extension_array_dtype(arr):
        return arr[notna(arr)]
    else:
        return arr[notna(lib.values_from_object(arr))]

def pop (self, key):
        """Remove key from dict and return value."""
        if key in self._keys:
            self._keys.remove(key)
        super(ListDict, self).pop(key)

def load_image(fname):
    """ read an image from file - PIL doesnt close nicely """
    with open(fname, "rb") as f:
        i = Image.open(fname)
        #i.load()
        return i

def file_read(filename):
    """Read a file and close it.  Returns the file source."""
    fobj = open(filename,'r');
    source = fobj.read();
    fobj.close()
    return source

def remove_file_from_s3(awsclient, bucket, key):
    """Remove a file from an AWS S3 bucket.

    :param awsclient:
    :param bucket:
    :param key:
    :return:
    """
    client_s3 = awsclient.get_client('s3')
    response = client_s3.delete_object(Bucket=bucket, Key=key)

def get_order(self, codes):
        """Return evidence codes in order shown in code2name."""
        return sorted(codes, key=lambda e: [self.ev2idx.get(e)])

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

def to_json(data):
    """Return data as a JSON string."""
    return json.dumps(data, default=lambda x: x.__dict__, sort_keys=True, indent=4)

def focusInEvent(self, event):
        """Reimplement Qt method to send focus change notification"""
        self.focus_changed.emit()
        return super(ControlWidget, self).focusInEvent(event)

def jsonify(symbol):
    """ returns json format for symbol """
    try:
        # all symbols have a toJson method, try it
        return json.dumps(symbol.toJson(), indent='  ')
    except AttributeError:
        pass
    return json.dumps(symbol, indent='  ')

def _remove_dict_keys_with_value(dict_, val):
  """Removes `dict` keys which have have `self` as value."""
  return {k: v for k, v in dict_.items() if v is not val}

def resources(self):
        """Retrieve contents of each page of PDF"""
        return [self.pdf.getPage(i) for i in range(self.pdf.getNumPages())]

def remove_legend(ax=None):
    """Remove legend for axes or gca.

    See http://osdir.com/ml/python.matplotlib.general/2005-07/msg00285.html
    """
    from pylab import gca, draw
    if ax is None:
        ax = gca()
    ax.legend_ = None
    draw()

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

def clean(self, text):
        """Remove all unwanted characters from text."""
        return ''.join([c for c in text if c in self.alphabet])

def clean_float(v):
    """Remove commas from a float"""

    if v is None or not str(v).strip():
        return None

    return float(str(v).replace(',', ''))

def get_line_ending(line):
    """Return line ending."""
    non_whitespace_index = len(line.rstrip()) - len(line)
    if not non_whitespace_index:
        return ''
    else:
        return line[non_whitespace_index:]

def _serialize_json(obj, fp):
    """ Serialize ``obj`` as a JSON formatted stream to ``fp`` """
    json.dump(obj, fp, indent=4, default=serialize)

def _removeTags(tags, objects):
    """ Removes tags from objects """
    for t in tags:
        for o in objects:
            o.tags.remove(t)

    return True

def random_choice(sequence):
    """ Same as :meth:`random.choice`, but also supports :class:`set` type to be passed as sequence. """
    return random.choice(tuple(sequence) if isinstance(sequence, set) else sequence)

def _clip(sid, prefix):
    """Clips a prefix from the beginning of a string if it exists."""
    return sid[len(prefix):] if sid.startswith(prefix) else sid

def pause(self):
        """Pause the music"""
        mixer.music.pause()
        self.pause_time = self.get_time()
        self.paused = True

def format_screen(strng):
    """Format a string for screen printing.

    This removes some latex-type format codes."""
    # Paragraph continue
    par_re = re.compile(r'\\$',re.MULTILINE)
    strng = par_re.sub('',strng)
    return strng

def __pop_top_frame(self):
        """Pops the top frame off the frame stack."""
        popped = self.__stack.pop()
        if self.__stack:
            self.__stack[-1].process_subframe(popped)

def make_aware(dt):
    """Appends tzinfo and assumes UTC, if datetime object has no tzinfo already."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

def cat_acc(y_true, y_pred):
    """Categorical accuracy
    """
    return np.mean(y_true.argmax(axis=1) == y_pred.argmax(axis=1))

def remove(self, key):
        """remove the value found at key from the queue"""
        item = self.item_finder.pop(key)
        item[-1] = None
        self.removed_count += 1

def close_stream(self):
		""" Closes the stream. Performs cleanup. """
		self.keep_listening = False
		self.stream.stop()
		self.stream.close()

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

def print_log(text, *colors):
    """Print a log message to standard error."""
    sys.stderr.write(sprint("{}: {}".format(script_name, text), *colors) + "\n")

def remove_property(self, key=None, value=None):
        """Remove all properties matching both key and value.

        :param str key: Key of the property.
        :param str value: Value of the property.
        """
        for k, v in self.properties[:]:
            if (key is None or key == k) and (value is None or value == v):
                del(self.properties[self.properties.index((k, v))])

def fail_print(error):
    """Print an error in red text.
    Parameters
        error (HTTPError)
            Error object to print.
    """
    print(COLORS.fail, error.message, COLORS.end)
    print(COLORS.fail, error.errors, COLORS.end)

def generic_add(a, b):
    print
    """Simple function to add two numbers"""
    logger.info('Called generic_add({}, {})'.format(a, b))
    return a + b

def subn_filter(s, find, replace, count=0):
    """A non-optimal implementation of a regex filter"""
    return re.gsub(find, replace, count, s)

def csvpretty(csvfile: csvfile=sys.stdin):
    """ Pretty print a CSV file. """
    shellish.tabulate(csv.reader(csvfile))

def dashrepl(value):
    """
    Replace any non-word characters with a dash.
    """
    patt = re.compile(r'\W', re.UNICODE)
    return re.sub(patt, '-', value)

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

def camelcase_underscore(name):
    """ Convert camelcase names to underscore """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def object_type_repr(obj):
    """Returns the name of the object's type.  For some recognized
    singletons the name of the object is returned instead. (For
    example for `None` and `Ellipsis`).
    """
    if obj is None:
        return 'None'
    elif obj is Ellipsis:
        return 'Ellipsis'
    if obj.__class__.__module__ == '__builtin__':
        name = obj.__class__.__name__
    else:
        name = obj.__class__.__module__ + '.' + obj.__class__.__name__
    return '%s object' % name

def parse_querystring(self, req, name, field):
        """Pull a querystring value from the request."""
        return core.get_value(req.args, name, field)

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

def __call__(self, r):
        """Update the request headers."""
        r.headers['Authorization'] = 'JWT {jwt}'.format(jwt=self.token)
        return r

def getTypeStr(_type):
  r"""Gets the string representation of the given type.
  """
  if isinstance(_type, CustomType):
    return str(_type)

  if hasattr(_type, '__name__'):
    return _type.__name__

  return ''

def head(self, path, query=None, data=None, redirects=True):
        """
        HEAD request wrapper for :func:`request()`
        """
        return self.request('HEAD', path, query, None, redirects)

def get_max(qs, field):
    """
    get max for queryset.

    qs: queryset
    field: The field name to max.
    """
    max_field = '%s__max' % field
    num = qs.aggregate(Max(field))[max_field]
    return num if num else 0

def session(self):
        """A context manager for this client's session.

        This function closes the current session when this client goes out of
        scope.
        """
        self._session = requests.session()
        yield
        self._session.close()
        self._session = None

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

def send_post(self, url, data, remove_header=None):
        """ Send a POST request
        """
        return self.send_request(method="post", url=url, data=data, remove_header=remove_header)

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

async def set_http_proxy(cls, url: typing.Optional[str]):
        """See `get_http_proxy`."""
        await cls.set_config("http_proxy", "" if url is None else url)

def print_out(self, *lst):
      """ Print list of strings to the predefined stdout. """
      self.print2file(self.stdout, True, True, *lst)

def parse_cookies(self, req, name, field):
        """Pull the value from the cookiejar."""
        return core.get_value(req.COOKIES, name, field)

def get_db_version(session):
    """
    :param session: actually it is a sqlalchemy session
    :return: version number
    """
    value = session.query(ProgramInformation.value).filter(ProgramInformation.name == "db_version").scalar()
    return int(value)

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

def _rnd_datetime(self, start, end):
        """Internal random datetime generator.
        """
        return self.from_utctimestamp(
            random.randint(
                int(self.to_utctimestamp(start)),
                int(self.to_utctimestamp(end)),
            )
        )

def crop_box(im, box=False, **kwargs):
    """Uses box coordinates to crop an image without resizing it first."""
    if box:
        im = im.crop(box)
    return im

def get_randomized_guid_sample(self, item_count):
        """ Fetch a subset of randomzied GUIDs from the whitelist """
        dataset = self.get_whitelist()
        random.shuffle(dataset)
        return dataset[:item_count]

def __getattr__(self, name):
        """ For attributes not found in self, redirect
        to the properties dictionary """

        try:
            return self.__dict__[name]
        except KeyError:
            if hasattr(self._properties,name):
                return getattr(self._properties, name)

def extract_words(lines):
    """
    Extract from the given iterable of lines the list of words.

    :param lines: an iterable of lines;
    :return: a generator of words of lines.
    """
    for line in lines:
        for word in re.findall(r"\w+", line):
            yield word

def stats(self):
        """
        Return a new raw REST interface to stats resources

        :rtype: :py:class:`ns1.rest.stats.Stats`
        """
        import ns1.rest.stats
        return ns1.rest.stats.Stats(self.config)

def get_iter_string_reader(stdin):
    """ return an iterator that returns a chunk of a string every time it is
    called.  notice that even though bufsize_type might be line buffered, we're
    not doing any line buffering here.  that's because our StreamBufferer
    handles all buffering.  we just need to return a reasonable-sized chunk. """
    bufsize = 1024
    iter_str = (stdin[i:i + bufsize] for i in range(0, len(stdin), bufsize))
    return get_iter_chunk_reader(iter_str)

def restore_default_settings():
    """ Restore settings to default values. 
    """
    global __DEFAULTS
    __DEFAULTS.CACHE_DIR = defaults.CACHE_DIR
    __DEFAULTS.SET_SEED = defaults.SET_SEED
    __DEFAULTS.SEED = defaults.SEED
    logging.info('Settings reverted to their default values.')

def get_jsonparsed_data(url):
    """Receive the content of ``url``, parse it as JSON and return the
       object.
    """
    response = urlopen(url)
    data = response.read().decode('utf-8')
    return json.loads(data)

def set_mem_per_proc(self, mem_mb):
        """Set the memory per process in megabytes"""
        super().set_mem_per_proc(mem_mb)
        self.qparams["mem_per_cpu"] = self.mem_per_proc

def logout():
    """ Log out the active user
    """
    flogin.logout_user()
    next = flask.request.args.get('next')
    return flask.redirect(next or flask.url_for("user"))

def dict_pick(dictionary, allowed_keys):
    """
    Return a dictionary only with keys found in `allowed_keys`
    """
    return {key: value for key, value in viewitems(dictionary) if key in allowed_keys}

def normalize_matrix(matrix):
  """Fold all values of the matrix into [0, 1]."""
  abs_matrix = np.abs(matrix.copy())
  return abs_matrix / abs_matrix.max()

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

def index(m, val):
    """
    Return the indices of all the ``val`` in ``m``
    """
    mm = np.array(m)
    idx_tuple = np.where(mm == val)
    idx = idx_tuple[0].tolist()

    return idx

def cleanup_nodes(doc):
    """
    Remove text nodes containing only whitespace
    """
    for node in doc.documentElement.childNodes:
        if node.nodeType == Node.TEXT_NODE and node.nodeValue.isspace():
            doc.documentElement.removeChild(node)
    return doc

def edge_index(self):
        """A map to look up the index of a edge"""
        return dict((edge, index) for index, edge in enumerate(self.edges))

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

def get_size(objects):
    """Compute the total size of all elements in objects."""
    res = 0
    for o in objects:
        try:
            res += _getsizeof(o)
        except AttributeError:
            print("IGNORING: type=%s; o=%s" % (str(type(o)), str(o)))
    return res

def _delete_keys(dct, keys):
    """Returns a copy of dct without `keys` keys
    """
    c = deepcopy(dct)
    assert isinstance(keys, list)
    for k in keys:
        c.pop(k)
    return c

def is_in(self, search_list, pair):
        """
        If pair is in search_list, return the index. Otherwise return -1
        """
        index = -1
        for nr, i in enumerate(search_list):
            if(np.all(i == pair)):
                return nr
        return index

def remove_instance(self, item):
        """Remove `instance` from model"""
        self.instances.remove(item)
        self.remove_item(item)

def success_response(**data):
    """Return a generic success response."""
    data_out = {}
    data_out["status"] = "success"
    data_out.update(data)
    js = dumps(data_out, default=date_handler)
    return Response(js, status=200, mimetype="application/json")

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

def closest(xarr, val):
    """ Return the index of the closest in xarr to value val """
    idx_closest = np.argmin(np.abs(np.array(xarr) - val))
    return idx_closest

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

def copy_to_temp(object):
    """
    Copy file-like object to temp file and return
    path.
    """
    temp_file = NamedTemporaryFile(delete=False)
    _copy_and_close(object, temp_file)
    return temp_file.name

def sanitize_word(s):
    """Remove non-alphanumerical characters from metric word.
    And trim excessive underscores.
    """
    s = re.sub('[^\w-]+', '_', s)
    s = re.sub('__+', '_', s)
    return s.strip('_')

def home():
    """Temporary helper function to link to the API routes"""
    return dict(links=dict(api='{}{}'.format(request.url, PREFIX[1:]))), \
        HTTPStatus.OK

def _removeTags(tags, objects):
    """ Removes tags from objects """
    for t in tags:
        for o in objects:
            o.tags.remove(t)

    return True

def get_short_url(self):
        """ Returns short version of topic url (without page number) """
        return reverse('post_short_url', args=(self.forum.slug, self.slug, self.id))

def text_cleanup(data, key, last_type):
    """ I strip extra whitespace off multi-line strings if they are ready to be stripped!"""
    if key in data and last_type == STRING_TYPE:
        data[key] = data[key].strip()
    return data

def rotateImage(img, angle):
    """

    querries scipy.ndimage.rotate routine
    :param img: image to be rotated
    :param angle: angle to be rotated (radian)
    :return: rotated image
    """
    imgR = scipy.ndimage.rotate(img, angle, reshape=False)
    return imgR

def myreplace(astr, thefind, thereplace):
    """in string astr replace all occurences of thefind with thereplace"""
    alist = astr.split(thefind)
    new_s = alist.split(thereplace)
    return new_s

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

def replace(s, replace):
    """Replace multiple values in a string"""
    for r in replace:
        s = s.replace(*r)
    return s

def round_figures(x, n):
    """Returns x rounded to n significant figures."""
    return round(x, int(n - math.ceil(math.log10(abs(x)))))

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

def listlike(obj):
    """Is an object iterable like a list (and not a string)?"""
    
    return hasattr(obj, "__iter__") \
    and not issubclass(type(obj), str)\
    and not issubclass(type(obj), unicode)

def get_rounded(self, digits):
        """ Return a vector with the elements rounded to the given number of digits. """
        result = self.copy()
        result.round(digits)
        return result

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

def managepy(cmd, extra=None):
    """Run manage.py using this component's specific Django settings"""

    extra = extra.split() if extra else []
    run_django_cli(['invoke', cmd] + extra)

def retrieve_by_id(self, id_):
        """Return a JSSObject for the element with ID id_"""
        items_with_id = [item for item in self if item.id == int(id_)]
        if len(items_with_id) == 1:
            return items_with_id[0].retrieve()

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

def start(self):
        """Create a background thread for httpd and serve 'forever'"""
        self._process = threading.Thread(target=self._background_runner)
        self._process.start()

def str_from_file(path):
    """
    Return file contents as string.

    """
    with open(path) as f:
        s = f.read().strip()
    return s

def execute_in_background(self):
        """Executes a (shell) command in the background

        :return: the process' pid
        """
        # http://stackoverflow.com/questions/1605520
        args = shlex.split(self.cmd)
        p = Popen(args)
        return p.pid

def unique_items(seq):
    """Return the unique items from iterable *seq* (in order)."""
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]

def stop_server(self):
        """
        Stop receiving connections, wait for all tasks to end, and then 
        terminate the server.
        """
        self.stop = True
        while self.task_count:
            time.sleep(END_RESP)
        self.terminate = True

def sort_nicely(l):
    """Sort the given list in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    l.sort(key=alphanum_key)

def is_writable_by_others(filename):
    """Check if file or directory is world writable."""
    mode = os.stat(filename)[stat.ST_MODE]
    return mode & stat.S_IWOTH

def round_to_n(x, n):
    """
    Round to sig figs
    """
    return round(x, -int(np.floor(np.log10(x))) + (n - 1))

def save_dict_to_file(filename, dictionary):
  """Saves dictionary as CSV file."""
  with open(filename, 'w') as f:
    writer = csv.writer(f)
    for k, v in iteritems(dictionary):
      writer.writerow([str(k), str(v)])

def clean_float(v):
    """Remove commas from a float"""

    if v is None or not str(v).strip():
        return None

    return float(str(v).replace(',', ''))

def write(self):
        """Write content back to file."""
        with open(self.path, 'w') as file_:
            file_.write(self.content)

def get_func_posargs_name(f):
    """Returns the name of the function f's keyword argument parameter if it exists, otherwise None"""
    sigparams = inspect.signature(f).parameters
    for p in sigparams:
        if sigparams[p].kind == inspect.Parameter.VAR_POSITIONAL:
            return sigparams[p].name
    return None

def norm(x, mu, sigma=1.0):
    """ Scipy norm function """
    return stats.norm(loc=mu, scale=sigma).pdf(x)

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

def _get_random_id():
    """ Get a random (i.e., unique) string identifier"""
    symbols = string.ascii_uppercase + string.ascii_lowercase + string.digits
    return ''.join(random.choice(symbols) for _ in range(15))

def get_list_dimensions(_list):
    """
    Takes a nested list and returns the size of each dimension followed
    by the element type in the list
    """
    if isinstance(_list, list) or isinstance(_list, tuple):
        return [len(_list)] + get_list_dimensions(_list[0])
    return []

def fetch_hg_push_log(repo_name, repo_url):
    """
    Run a HgPushlog etl process
    """
    newrelic.agent.add_custom_parameter("repo_name", repo_name)
    process = HgPushlogProcess()
    process.run(repo_url + '/json-pushes/?full=1&version=2', repo_name)

def main():
    """
    Commandline interface to average parameters.
    """
    setup_main_logger(console=True, file_logging=False)
    params = argparse.ArgumentParser(description="Averages parameters from multiple models.")
    arguments.add_average_args(params)
    args = params.parse_args()
    average_parameters(args)

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

def update_cursor_position(self, line, index):
        """Update cursor position."""
        value = 'Line {}, Col {}'.format(line + 1, index + 1)
        self.set_value(value)

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

def populate_obj(obj, attrs):
    """Populates an object's attributes using the provided dict
    """
    for k, v in attrs.iteritems():
        setattr(obj, k, v)

def scroll_element_into_view(self):
        """Scroll element into view

        :returns: page element instance
        """
        x = self.web_element.location['x']
        y = self.web_element.location['y']
        self.driver.execute_script('window.scrollTo({0}, {1})'.format(x, y))
        return self

def set_font_size(self, size):
        """Convenience method for just changing font size."""
        if self.font.font_size == size:
            pass
        else:
            self.font._set_size(size)

def fast_distinct(self):
        """
        Because standard distinct used on the all fields are very slow and works only with PostgreSQL database
        this method provides alternative to the standard distinct method.
        :return: qs with unique objects
        """
        return self.model.objects.filter(pk__in=self.values_list('pk', flat=True))

def set_xlimits_widgets(self, set_min=True, set_max=True):
        """Populate axis limits GUI with current plot values."""
        xmin, xmax = self.tab_plot.ax.get_xlim()
        if set_min:
            self.w.x_lo.set_text('{0}'.format(xmin))
        if set_max:
            self.w.x_hi.set_text('{0}'.format(xmax))

def send(r, stream=False):
    """Just sends the request using its send method and returns its response.  """
    r.send(stream=stream)
    return r.response

def _position():
    """Returns the current xy coordinates of the mouse cursor as a two-integer
    tuple by calling the GetCursorPos() win32 function.

    Returns:
      (x, y) tuple of the current xy coordinates of the mouse cursor.
    """

    cursor = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(cursor))
    return (cursor.x, cursor.y)

def split_into_sentences(s):
  """Split text into list of sentences."""
  s = re.sub(r"\s+", " ", s)
  s = re.sub(r"[\\.\\?\\!]", "\n", s)
  return s.split("\n")

def finished(self):
        """
        Must be called to print final progress label.
        """
        self.progress_bar.set_state(ProgressBar.STATE_DONE)
        self.progress_bar.show()

def _time_to_json(value):
    """Coerce 'value' to an JSON-compatible representation."""
    if isinstance(value, datetime.time):
        value = value.isoformat()
    return value

def _help():
    """ Display both SQLAlchemy and Python help statements """

    statement = '%s%s' % (shelp, phelp % ', '.join(cntx_.keys()))
    print statement.strip()

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

def _shuffle(data, idx):
    """Shuffle the data."""
    shuffle_data = []

    for idx_k, idx_v in data:
        shuffle_data.append((idx_k, mx.ndarray.array(idx_v.asnumpy()[idx], idx_v.context)))

    return shuffle_data

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

def save_session_to_file(self, sessionfile):
        """Not meant to be used directly, use :meth:`Instaloader.save_session_to_file`."""
        pickle.dump(requests.utils.dict_from_cookiejar(self._session.cookies), sessionfile)

def getcolslice(self, blc, trc, inc=[], startrow=0, nrow=-1, rowincr=1):
        """Get a slice from a table column holding arrays.
        (see :func:`table.getcolslice`)"""
        return self._table.getcolslice(self._column, blc, trc, inc, startrow, nrow, rowincr)

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

def grow_slice(slc, size):
    """
    Grow a slice object by 1 in each direction without overreaching the list.

    Parameters
    ----------
    slc: slice
        slice object to grow
    size: int
        list length

    Returns
    -------
    slc: slice
       extended slice 

    """

    return slice(max(0, slc.start-1), min(size, slc.stop+1))

def reset(self):
        """Reset analyzer state
        """
        self.prevframe = None
        self.wasmoving = False
        self.t0 = 0
        self.ismoving = False

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

def aug_sysargv(cmdstr):
    """ DEBUG FUNC modify argv to look like you ran a command """
    import shlex
    argv = shlex.split(cmdstr)
    sys.argv.extend(argv)

def sort_nicely(l):
    """Sort the given list in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    l.sort(key=alphanum_key)

def test(ctx, all=False, verbose=False):
    """Run the tests."""
    cmd = 'tox' if all else 'py.test'
    if verbose:
        cmd += ' -v'
    return ctx.run(cmd, pty=True).return_code

def sort_data(x, y):
    """Sort the data."""
    xy = sorted(zip(x, y))
    x, y = zip(*xy)
    return x, y

def _zerosamestates(self, A):
        """
        zeros out states that should be identical

        REQUIRED ARGUMENTS

        A: the matrix whose entries are to be zeroed.

        """

        for pair in self.samestates:
            A[pair[0], pair[1]] = 0
            A[pair[1], pair[0]] = 0

def sort_data(data, cols):
    """Sort `data` rows and order columns"""
    return data.sort_values(cols)[cols + ['value']].reset_index(drop=True)

def set_global(node: Node, key: str, value: Any):
    """Adds passed value to node's globals"""
    node.node_globals[key] = value

def _histplot_op(ax, data, **kwargs):
    """Add a histogram for the data to the axes."""
    bins = get_bins(data)
    ax.hist(data, bins=bins, align="left", density=True, **kwargs)
    return ax

def __run(self):
    """Hacked run function, which installs the trace."""
    sys.settrace(self.globaltrace)
    self.__run_backup()
    self.run = self.__run_backup

def transpose(table):
    """
    transpose matrix
    """
    t = []
    for i in range(0, len(table[0])):
        t.append([row[i] for row in table])
    return t

def log_y_cb(self, w, val):
        """Toggle linear/log scale for Y-axis."""
        self.tab_plot.logy = val
        self.plot_two_columns()

def _split_arrs(array_2d, slices):
    """
    Equivalent to numpy.split(array_2d, slices),
    but avoids fancy indexing
    """
    if len(array_2d) == 0:
        return np.empty(0, dtype=np.object)

    rtn = np.empty(len(slices) + 1, dtype=np.object)
    start = 0
    for i, s in enumerate(slices):
        rtn[i] = array_2d[start:s]
        start = s
    rtn[-1] = array_2d[start:]
    return rtn

def ynticks(self, nticks, index=1):
        """Set the number of ticks."""
        self.layout['yaxis' + str(index)]['nticks'] = nticks
        return self

def normalize_matrix(matrix):
  """Fold all values of the matrix into [0, 1]."""
  abs_matrix = np.abs(matrix.copy())
  return abs_matrix / abs_matrix.max()

def empty(self, start=None, stop=None):
		"""Empty the range from start to stop.

		Like delete, but no Error is raised if the entire range isn't mapped.
		"""
		self.set(NOT_SET, start=start, stop=stop)

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

def linebuffered_stdout():
    """ Always line buffer stdout so pipes and redirects are CLI friendly. """
    if sys.stdout.line_buffering:
        return sys.stdout
    orig = sys.stdout
    new = type(orig)(orig.buffer, encoding=orig.encoding, errors=orig.errors,
                     line_buffering=True)
    new.mode = orig.mode
    return new

def save_session(self, sid, session, namespace=None):
        """Store the user session for a client.

        The only difference with the :func:`socketio.Server.save_session`
        method is that when the ``namespace`` argument is not given the
        namespace associated with the class is used.
        """
        return self.server.save_session(
            sid, session, namespace=namespace or self.namespace)

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

def strToBool(val):
    """
    Helper function to turn a string representation of "true" into
    boolean True.
    """
    if isinstance(val, str):
        val = val.lower()

    return val in ['true', 'on', 'yes', True]

def unique(iterable):
    """ Returns a list copy in which each item occurs only once (in-order).
    """
    seen = set()
    return [x for x in iterable if x not in seen and not seen.add(x)]

def pair_strings_sum_formatter(a, b):
  """
  Formats the sum of a and b.

  Note
  ----
  Both inputs are numbers already converted to strings.

  """
  if b[:1] == "-":
    return "{0} - {1}".format(a, b[1:])
  return "{0} + {1}".format(a, b)

def setdefaults(dct, defaults):
    """Given a target dct and a dict of {key:default value} pairs,
    calls setdefault for all of those pairs."""
    for key in defaults:
        dct.setdefault(key, defaults[key])

    return dct

def to_json(obj):
    """Return a json string representing the python object obj."""
    i = StringIO.StringIO()
    w = Writer(i, encoding='UTF-8')
    w.write_value(obj)
    return i.getvalue()

def main(argv, version=DEFAULT_VERSION):
    """Install or upgrade setuptools and EasyInstall"""
    tarball = download_setuptools()
    _install(tarball, _build_install_args(argv))

def split_comma_argument(comma_sep_str):
    """Split a comma separated option into a list."""
    terms = []
    for term in comma_sep_str.split(','):
        if term:
            terms.append(term)
    return terms

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

def _sha1_for_file(filename):
    """Return sha1 for contents of filename."""
    with open(filename, "rb") as fileobj:
        contents = fileobj.read()
        return hashlib.sha1(contents).hexdigest()

def convert_camel_case_to_snake_case(name):
    """Convert CamelCase to snake_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def sha1(s):
    """ Returns a sha1 of the given string
    """
    h = hashlib.new('sha1')
    h.update(s)
    return h.hexdigest()

def from_years_range(start_year, end_year):
        """Transform a range of years (two ints) to a DateRange object."""
        start = datetime.date(start_year, 1 , 1)
        end = datetime.date(end_year, 12 , 31)
        return DateRange(start, end)

def has_virtualenv(self):
        """
        Returns true if the virtualenv tool is installed.
        """
        with self.settings(warn_only=True):
            ret = self.run_or_local('which virtualenv').strip()
            return bool(ret)

def glog(x,l = 2):
    """
    Generalised logarithm

    :param x: number
    :param p: number added befor logarithm 

    """
    return np.log((x+np.sqrt(x**2+l**2))/2)/np.log(l)

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

def _repr(obj):
    """Show the received object as precise as possible."""
    vals = ", ".join("{}={!r}".format(
        name, getattr(obj, name)) for name in obj._attribs)
    if vals:
        t = "{}(name={}, {})".format(obj.__class__.__name__, obj.name, vals)
    else:
        t = "{}(name={})".format(obj.__class__.__name__, obj.name)
    return t

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

def display_pil_image(im):
   """Displayhook function for PIL Images, rendered as PNG."""
   from IPython.core import display
   b = BytesIO()
   im.save(b, format='png')
   data = b.getvalue()

   ip_img = display.Image(data=data, format='png', embed=True)
   return ip_img._repr_png_()

def format_exception(e):
    """Returns a string containing the type and text of the exception.

    """
    from .utils.printing import fill
    return '\n'.join(fill(line) for line in traceback.format_exception_only(type(e), e))

def signal_handler(signal_name, frame):
    """Quit signal handler."""
    sys.stdout.flush()
    print("\nSIGINT in frame signal received. Quitting...")
    sys.stdout.flush()
    sys.exit(0)

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

def signal_handler(signal_name, frame):
    """Quit signal handler."""
    sys.stdout.flush()
    print("\nSIGINT in frame signal received. Quitting...")
    sys.stdout.flush()
    sys.exit(0)

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

def unescape_all(string):
    """Resolve all html entities to their corresponding unicode character"""
    def escape_single(matchobj):
        return _unicode_for_entity_with_name(matchobj.group(1))
    return entities.sub(escape_single, string)

def update(self, params):
        """Update the dev_info data from a dictionary.

        Only updates if it already exists in the device.
        """
        dev_info = self.json_state.get('deviceInfo')
        dev_info.update({k: params[k] for k in params if dev_info.get(k)})

def log_request(self, code='-', size='-'):
        """Selectively log an accepted request."""

        if self.server.logRequests:
            BaseHTTPServer.BaseHTTPRequestHandler.log_request(self, code, size)

def update(self, params):
        """Update the dev_info data from a dictionary.

        Only updates if it already exists in the device.
        """
        dev_info = self.json_state.get('deviceInfo')
        dev_info.update({k: params[k] for k in params if dev_info.get(k)})

def calculate_size(name, timeout):
    """ Calculates the request payload size"""
    data_size = 0
    data_size += calculate_size_str(name)
    data_size += LONG_SIZE_IN_BYTES
    return data_size

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

def _skip_frame(self):
        """Skip a single frame from the trajectory"""
        size = self.read_size()
        for i in range(size+1):
            line = self._f.readline()
            if len(line) == 0:
                raise StopIteration

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

def _skip_newlines(self):
        """Increment over newlines."""
        while self._cur_token['type'] is TT.lbreak and not self._finished:
            self._increment()

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

def _skip_frame(self):
        """Skip a single frame from the trajectory"""
        size = self.read_size()
        for i in range(size+1):
            line = self._f.readline()
            if len(line) == 0:
                raise StopIteration

def indent(txt, spacing=4):
    """
    Indent given text using custom spacing, default is set to 4.
    """
    return prefix(str(txt), ''.join([' ' for _ in range(spacing)]))

def roc_auc(y_true, y_score):
    """
    Returns are under the ROC curve
    """
    notnull = ~np.isnan(y_true)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true[notnull], y_score[notnull])
    return sklearn.metrics.auc(fpr, tpr)

def write_color(string, name, style='normal', when='auto'):
    """ Write the given colored string to standard out. """
    write(color(string, name, style, when))

def NeuralNetLearner(dataset, sizes):
   """Layered feed-forward network."""

   activations = map(lambda n: [0.0 for i in range(n)], sizes)
   weights = []

   def predict(example):
      unimplemented()

   return predict

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

def main(idle):
    """Any normal python logic which runs a loop. Can take arguments."""
    while True:

        LOG.debug("Sleeping for {0} seconds.".format(idle))
        time.sleep(idle)

def find_le(a, x):
    """Find rightmost value less than or equal to x."""
    i = bs.bisect_right(a, x)
    if i: return i - 1
    raise ValueError

def seconds(num):
    """
    Pause for this many seconds
    """
    now = pytime.time()
    end = now + num
    until(end)

def html_to_text(content):
    """ Converts html content to plain text """
    text = None
    h2t = html2text.HTML2Text()
    h2t.ignore_links = False
    text = h2t.handle(content)
    return text

def is_full_slice(obj, l):
    """
    We have a full length slice.
    """
    return (isinstance(obj, slice) and obj.start == 0 and obj.stop == l and
            obj.step is None)

def _re_raise_as(NewExc, *args, **kw):
    """Raise a new exception using the preserved traceback of the last one."""
    etype, val, tb = sys.exc_info()
    raise NewExc(*args, **kw), None, tb

def partition(a, sz): 
    """splits iterables a in equal parts of size sz"""
    return [a[i:i+sz] for i in range(0, len(a), sz)]

def alert(text='', title='', button=OK_TEXT, root=None, timeout=None):
    """Displays a simple message box with text and a single OK button. Returns the text of the button clicked on."""
    assert TKINTER_IMPORT_SUCCEEDED, 'Tkinter is required for pymsgbox'
    return _buttonbox(msg=text, title=title, choices=[str(button)], root=root, timeout=timeout)

def output_scores(self, name=None):
        """ Returns: N x #class scores, summed to one for each box."""
        return tf.nn.softmax(self.label_logits, name=name)

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

def sort_by_name(self):
        """Sort list elements by name."""
        super(JSSObjectList, self).sort(key=lambda k: k.name)

def _fill(self):
    """Advance the iterator without returning the old head."""
    try:
      self._head = self._iterable.next()
    except StopIteration:
      self._head = None

def _dict_values_sorted_by_key(dictionary):
    # This should be a yield from instead.
    """Internal helper to return the values of a dictionary, sorted by key.
    """
    for _, value in sorted(dictionary.iteritems(), key=operator.itemgetter(0)):
        yield value

def normalize_text(text, line_len=80, indent=""):
    """Wrap the text on the given line length."""
    return "\n".join(
        textwrap.wrap(
            text, width=line_len, initial_indent=indent, subsequent_indent=indent
        )
    )

def sort_by_name(self):
        """Sort list elements by name."""
        super(JSSObjectList, self).sort(key=lambda k: k.name)

def get_rounded(self, digits):
        """ Return a vector with the elements rounded to the given number of digits. """
        result = self.copy()
        result.round(digits)
        return result

def unique_list_dicts(dlist, key):
    """Return a list of dictionaries which are sorted for only unique entries.

    :param dlist:
    :param key:
    :return list:
    """

    return list(dict((val[key], val) for val in dlist).values())

def comment (self, s, **args):
        """Write GML comment."""
        self.writeln(s=u'comment "%s"' % s, **args)

def get_order(self, codes):
        """Return evidence codes in order shown in code2name."""
        return sorted(codes, key=lambda e: [self.ev2idx.get(e)])

def batch(items, size):
    """Batches a list into a list of lists, with sub-lists sized by a specified
    batch size."""
    return [items[x:x + size] for x in xrange(0, len(items), size)]

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

def split_on(s, sep=" "):
    """Split s by sep, unless it's inside a quote."""
    pattern = '''((?:[^%s"']|"[^"]*"|'[^']*')+)''' % sep

    return [_strip_speechmarks(t) for t in re.split(pattern, s)[1::2]]

def write_tsv_line_from_list(linelist, outfp):
    """Utility method to convert list to tsv line with carriage return"""
    line = '\t'.join(linelist)
    outfp.write(line)
    outfp.write('\n')

def ver_to_tuple(value):
    """
    Convert version like string to a tuple of integers.
    """
    return tuple(int(_f) for _f in re.split(r'\D+', value) if _f)

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

def split_len(s, length):
    """split string *s* into list of strings no longer than *length*"""
    return [s[i:i+length] for i in range(0, len(s), length)]

def upoint2exprpoint(upoint):
    """Convert an untyped point into an Expression point.

    .. seealso::
       For definitions of points and untyped points,
       see the :mod:`pyeda.boolalg.boolfunc` module.
    """
    point = dict()
    for uniqid in upoint[0]:
        point[_LITS[uniqid]] = 0
    for uniqid in upoint[1]:
        point[_LITS[uniqid]] = 1
    return point

def to_snake_case(s):
    """Converts camel-case identifiers to snake-case."""
    return re.sub('([^_A-Z])([A-Z])', lambda m: m.group(1) + '_' + m.group(2).lower(), s)

def logical_or(self, other):
        """logical_or(t) = self(t) or other(t)."""
        return self.operation(other, lambda x, y: int(x or y))

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

def get_random_id(length):
    """Generate a random, alpha-numerical id."""
    alphabet = string.ascii_uppercase + string.ascii_lowercase + string.digits
    return ''.join(random.choice(alphabet) for _ in range(length))

def unescape_all(string):
    """Resolve all html entities to their corresponding unicode character"""
    def escape_single(matchobj):
        return _unicode_for_entity_with_name(matchobj.group(1))
    return entities.sub(escape_single, string)

def save(self):
        """Saves the updated model to the current entity db.
        """
        self.session.add(self)
        self.session.flush()
        return self

def md_to_text(content):
    """ Converts markdown content to text """
    text = None
    html = markdown.markdown(content)
    if html:
        text = html_to_text(content)
    return text

def createdb():
    """Create database tables from sqlalchemy models"""
    manager.db.engine.echo = True
    manager.db.create_all()
    set_alembic_revision()

def main(idle):
    """Any normal python logic which runs a loop. Can take arguments."""
    while True:

        LOG.debug("Sleeping for {0} seconds.".format(idle))
        time.sleep(idle)

def get_tablenames(cur):
    """ Conveinience: """
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tablename_list_ = cur.fetchall()
    tablename_list = [str(tablename[0]) for tablename in tablename_list_ ]
    return tablename_list

def _getTypename(self, defn):
        """ Returns the SQL typename required to store the given FieldDefinition """
        return 'REAL' if defn.type.float or 'TIME' in defn.type.name or defn.dntoeu else 'INTEGER'

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

def visit_Str(self, node):
        """ Set the pythonic string type. """
        self.result[node] = self.builder.NamedType(pytype_to_ctype(str))

def _plot(self):
        """Plot stacked serie lines and stacked secondary lines"""
        for serie in self.series[::-1 if self.stack_from_top else 1]:
            self.line(serie)
        for serie in self.secondary_series[::-1 if self.stack_from_top else 1]:
            self.line(serie, True)

def print(*a):
    """ print just one that returns what you give it instead of None """
    try:
        _print(*a)
        return a[0] if len(a) == 1 else a
    except:
        _print(*a)

def _stdout_raw(self, s):
        """Writes the string to stdout"""
        print(s, end='', file=sys.stdout)
        sys.stdout.flush()

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

def stop(self, reason=None):
        """Shutdown the service with a reason."""
        self.logger.info('stopping')
        self.loop.stop(pyev.EVBREAK_ALL)

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

def load_logged_in_user():
    """If a user id is stored in the session, load the user object from
    the database into ``g.user``."""
    user_id = session.get("user_id")
    g.user = User.query.get(user_id) if user_id is not None else None

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

def u2b(string):
    """ unicode to bytes"""
    if ((PY2 and isinstance(string, unicode)) or
        ((not PY2) and isinstance(string, str))):
        return string.encode('utf-8')
    return string

def _create_complete_graph(node_ids):
    """Create a complete graph from the list of node ids.

    Args:
        node_ids: a list of node ids

    Returns:
        An undirected graph (as a networkx.Graph)
    """
    g = nx.Graph()
    g.add_nodes_from(node_ids)
    for (i, j) in combinations(node_ids, 2):
        g.add_edge(i, j)
    return g

def strip_spaces(x):
    """
    Strips spaces
    :param x:
    :return:
    """
    x = x.replace(b' ', b'')
    x = x.replace(b'\t', b'')
    return x

def pop (self, key):
        """Remove key from dict and return value."""
        if key in self._keys:
            self._keys.remove(key)
        super(ListDict, self).pop(key)

def clean(self, text):
        """Remove all unwanted characters from text."""
        return ''.join([c for c in text if c in self.alphabet])

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

def myreplace(astr, thefind, thereplace):
    """in string astr replace all occurences of thefind with thereplace"""
    alist = astr.split(thefind)
    new_s = alist.split(thereplace)
    return new_s

def scale_image(image, new_width):
    """Resizes an image preserving the aspect ratio.
    """
    (original_width, original_height) = image.size
    aspect_ratio = original_height/float(original_width)
    new_height = int(aspect_ratio * new_width)

    # This scales it wider than tall, since characters are biased
    new_image = image.resize((new_width*2, new_height))
    return new_image

def myreplace(astr, thefind, thereplace):
    """in string astr replace all occurences of thefind with thereplace"""
    alist = astr.split(thefind)
    new_s = alist.split(thereplace)
    return new_s

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

def us2mc(string):
    """Transform an underscore_case string to a mixedCase string"""
    return re.sub(r'_([a-z])', lambda m: (m.group(1).upper()), string)

def list_string_to_dict(string):
    """Inputs ``['a', 'b', 'c']``, returns ``{'a': 0, 'b': 1, 'c': 2}``."""
    dictionary = {}
    for idx, c in enumerate(string):
        dictionary.update({c: idx})
    return dictionary

def _xls2col_widths(self, worksheet, tab):
        """Updates col_widths in code_array"""

        for col in xrange(worksheet.ncols):
            try:
                xls_width = worksheet.colinfo_map[col].width
                pys_width = self.xls_width2pys_width(xls_width)
                self.code_array.col_widths[col, tab] = pys_width

            except KeyError:
                pass

def getEventTypeNameFromEnum(self, eType):
        """returns the name of an EVREvent enum value"""

        fn = self.function_table.getEventTypeNameFromEnum
        result = fn(eType)
        return result

def process_docstring(app, what, name, obj, options, lines):
    """React to a docstring event and append contracts to it."""
    # pylint: disable=unused-argument
    # pylint: disable=too-many-arguments
    lines.extend(_format_contracts(what=what, obj=obj))

def timestamp_to_microseconds(timestamp):
    """Convert a timestamp string into a microseconds value
    :param timestamp
    :return time in microseconds
    """
    timestamp_str = datetime.datetime.strptime(timestamp, ISO_DATETIME_REGEX)
    epoch_time_secs = calendar.timegm(timestamp_str.timetuple())
    epoch_time_mus = epoch_time_secs * 1e6 + timestamp_str.microsecond
    return epoch_time_mus

def is_in(self, search_list, pair):
        """
        If pair is in search_list, return the index. Otherwise return -1
        """
        index = -1
        for nr, i in enumerate(search_list):
            if(np.all(i == pair)):
                return nr
        return index

def FromString(s, **kwargs):
    """Like FromFile, but takes a string."""
    
    f = StringIO.StringIO(s)
    return FromFile(f, **kwargs)

def findLastCharIndexMatching(text, func):
    """ Return index of last character in string for which func(char) evaluates to True. """
    for i in range(len(text) - 1, -1, -1):
      if func(text[i]):
        return i

def FromString(s, **kwargs):
    """Like FromFile, but takes a string."""
    
    f = StringIO.StringIO(s)
    return FromFile(f, **kwargs)

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

def drop_bad_characters(text):
    """Takes a text and drops all non-printable and non-ascii characters and
    also any whitespace characters that aren't space.

    :arg str text: the text to fix

    :returns: text with all bad characters dropped

    """
    # Strip all non-ascii and non-printable characters
    text = ''.join([c for c in text if c in ALLOWED_CHARS])
    return text

def get_lines(handle, line):
    """
    Get zero-indexed line from an open file-like.
    """
    for i, l in enumerate(handle):
        if i == line:
            return l

def RoundToSeconds(cls, timestamp):
    """Takes a timestamp value and rounds it to a second precision."""
    leftovers = timestamp % definitions.MICROSECONDS_PER_SECOND
    scrubbed = timestamp - leftovers
    rounded = round(float(leftovers) / definitions.MICROSECONDS_PER_SECOND)

    return int(scrubbed + rounded * definitions.MICROSECONDS_PER_SECOND)

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

def unpunctuate(s, *, char_blacklist=string.punctuation):
    """ Remove punctuation from string s. """
    # remove punctuation
    s = "".join(c for c in s if c not in char_blacklist)
    # remove consecutive spaces
    return " ".join(filter(None, s.split(" ")))

def _init_unique_sets(self):
        """Initialise sets used for uniqueness checking."""

        ks = dict()
        for t in self._unique_checks:
            key = t[0]
            ks[key] = set() # empty set
        return ks

def _clip(sid, prefix):
    """Clips a prefix from the beginning of a string if it exists."""
    return sid[len(prefix):] if sid.startswith(prefix) else sid

def merge(left, right, how='inner', key=None, left_key=None, right_key=None,
          left_as='left', right_as='right'):
    """ Performs a join using the union join function. """
    return join(left, right, how, key, left_key, right_key,
                join_fn=make_union_join(left_as, right_as))

def _clip(sid, prefix):
    """Clips a prefix from the beginning of a string if it exists."""
    return sid[len(prefix):] if sid.startswith(prefix) else sid

def get_input(input_func, input_str):
    """
    Get input from the user given an input function and an input string
    """
    val = input_func("Please enter your {0}: ".format(input_str))
    while not val or not len(val.strip()):
        val = input_func("You didn't enter a valid {0}, please try again: ".format(input_str))
    return val

def weekly(date=datetime.date.today()):
    """
    Weeks start are fixes at Monday for now.
    """
    return date - datetime.timedelta(days=date.weekday())

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

def asin(x):
    """
    Inverse sine
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.arcsin(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.arcsin(x)

def list_move_to_front(l,value='other'):
    """if the value is in the list, move it to the front and return it."""
    l=list(l)
    if value in l:
        l.remove(value)
        l.insert(0,value)
    return l

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

def runcoro(async_function):
    """
    Runs an asynchronous function without needing to use await - useful for lambda

    Args:
        async_function (Coroutine): The asynchronous function to run
    """

    future = _asyncio.run_coroutine_threadsafe(async_function, client.loop)
    result = future.result()
    return result

def setencoding():
    """Set the string encoding used by the Unicode implementation.  The
    default is 'ascii', but if you're willing to experiment, you can
    change this."""
    encoding = "ascii" # Default value set by _PyUnicode_Init()
    if 0:
        # Enable to support locale aware default string encodings.
        import locale
        loc = locale.getdefaultlocale()
        if loc[1]:
            encoding = loc[1]
    if 0:
        # Enable to switch off string to Unicode coercion and implicit
        # Unicode to string conversion.
        encoding = "undefined"
    if encoding != "ascii":
        # On Non-Unicode builds this will raise an AttributeError...
        sys.setdefaultencoding(encoding)

def lin_interp(x, rangeX, rangeY):
    """
    Interpolate linearly variable x in rangeX onto rangeY.
    """
    s = (x - rangeX[0]) / mag(rangeX[1] - rangeX[0])
    y = rangeY[0] * (1 - s) + rangeY[1] * s
    return y

def pass_from_pipe(cls):
        """Return password from pipe if not on TTY, else False.
        """
        is_pipe = not sys.stdin.isatty()
        return is_pipe and cls.strip_last_newline(sys.stdin.read())

def lin_interp(x, rangeX, rangeY):
    """
    Interpolate linearly variable x in rangeX onto rangeY.
    """
    s = (x - rangeX[0]) / mag(rangeX[1] - rangeX[0])
    y = rangeY[0] * (1 - s) + rangeY[1] * s
    return y

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

def unique_everseen(iterable, filterfalse_=itertools.filterfalse):
    """Unique elements, preserving order."""
    # Itertools recipes:
    # https://docs.python.org/3/library/itertools.html#itertools-recipes
    seen = set()
    seen_add = seen.add
    for element in filterfalse_(seen.__contains__, iterable):
        seen_add(element)
        yield element

def findMax(arr):
    """
    in comparison to argrelmax() more simple and  reliable peak finder
    """
    out = np.zeros(shape=arr.shape, dtype=bool)
    _calcMax(arr, out)
    return out

def raw_print(*args, **kw):
    """Raw print to sys.__stdout__, otherwise identical interface to print()."""

    print(*args, sep=kw.get('sep', ' '), end=kw.get('end', '\n'),
          file=sys.__stdout__)
    sys.__stdout__.flush()

def file_found(filename,force):
    """Check if a file exists"""
    if os.path.exists(filename) and not force:
        logger.info("Found %s; skipping..."%filename)
        return True
    else:
        return False

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

def bash(filename):
    """Runs a bash script in the local directory"""
    sys.stdout.flush()
    subprocess.call("bash {}".format(filename), shell=True)

def _platform_is_windows(platform=sys.platform):
        """Is the current OS a Windows?"""
        matched = platform in ('cygwin', 'win32', 'win64')
        if matched:
            error_msg = "Windows isn't supported yet"
            raise OSError(error_msg)
        return matched

def _most_common(iterable):
    """Returns the most common element in `iterable`."""
    data = Counter(iterable)
    return max(data, key=data.__getitem__)

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

def mask_nonfinite(self):
        """Extend the mask with the image elements where the intensity is NaN."""
        self.mask = np.logical_and(self.mask, (np.isfinite(self.intensity)))

def sg_init(sess):
    r""" Initializes session variables.
    
    Args:
      sess: Session to initialize. 
    """
    # initialize variables
    sess.run(tf.group(tf.global_variables_initializer(),
                      tf.local_variables_initializer()))

async def iso(self, source):
        """Convert to timestamp."""
        from datetime import datetime
        unix_timestamp = int(source)
        return datetime.fromtimestamp(unix_timestamp).isoformat()

def main(args):
    """
    invoke wptools and exit safely
    """
    start = time.time()
    output = get(args)
    _safe_exit(start, output)

def chunks(iterable, n):
    """Yield successive n-sized chunks from iterable object. https://stackoverflow.com/a/312464 """
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]

def is_listish(obj):
    """Check if something quacks like a list."""
    if isinstance(obj, (list, tuple, set)):
        return True
    return is_sequence(obj)

def itervalues(d, **kw):
    """Return an iterator over the values of a dictionary."""
    if not PY2:
        return iter(d.values(**kw))
    return d.itervalues(**kw)

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

def group_by(iterable, key_func):
    """Wrap itertools.groupby to make life easier."""
    groups = (
        list(sub) for key, sub in groupby(iterable, key_func)
    )
    return zip(groups, groups)

def is_square_matrix(mat):
    """Test if an array is a square matrix."""
    mat = np.array(mat)
    if mat.ndim != 2:
        return False
    shape = mat.shape
    return shape[0] == shape[1]

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

def contains_geometric_info(var):
    """ Check whether the passed variable is a tuple with two floats or integers """
    return isinstance(var, tuple) and len(var) == 2 and all(isinstance(val, (int, float)) for val in var)

def test_python_java_rt():
    """ Run Python test cases against Java runtime classes. """
    sub_env = {'PYTHONPATH': _build_dir()}

    log.info('Executing Python unit tests (against Java runtime classes)...')
    return jpyutil._execute_python_scripts(python_java_rt_tests,
                                           env=sub_env)

def _check_format(file_path, content):
    """ check testcase format if valid
    """
    # TODO: replace with JSON schema validation
    if not content:
        # testcase file content is empty
        err_msg = u"Testcase file content is empty: {}".format(file_path)
        logger.log_error(err_msg)
        raise exceptions.FileFormatError(err_msg)

    elif not isinstance(content, (list, dict)):
        # testcase file content does not match testcase format
        err_msg = u"Testcase file content format invalid: {}".format(file_path)
        logger.log_error(err_msg)
        raise exceptions.FileFormatError(err_msg)

def dump_json(obj):
    """Dump Python object as JSON string."""
    return simplejson.dumps(obj, ignore_nan=True, default=json_util.default)

def do_wordwrap(s, width=79, break_long_words=True):
    """
    Return a copy of the string passed to the filter wrapped after
    ``79`` characters.  You can override this default using the first
    parameter.  If you set the second parameter to `false` Jinja will not
    split words apart if they are longer than `width`.
    """
    import textwrap
    return u'\n'.join(textwrap.wrap(s, width=width, expand_tabs=False,
                                   replace_whitespace=False,
                                   break_long_words=break_long_words))

def from_json(value, **kwargs):
        """Coerces JSON string to boolean"""
        if isinstance(value, string_types):
            value = value.upper()
            if value in ('TRUE', 'Y', 'YES', 'ON'):
                return True
            if value in ('FALSE', 'N', 'NO', 'OFF'):
                return False
        if isinstance(value, int):
            return value
        raise ValueError('Could not load boolean from JSON: {}'.format(value))

def is_progressive(image):
    """
    Check to see if an image is progressive.
    """
    if not isinstance(image, Image.Image):
        # Can only check PIL images for progressive encoding.
        return False
    return ('progressive' in image.info) or ('progression' in image.info)

def json_dumps(self, obj):
        """Serializer for consistency"""
        return json.dumps(obj, sort_keys=True, indent=4, separators=(',', ': '))

def set(self):
        """Set the internal flag to true.

        All threads waiting for the flag to become true are awakened. Threads
        that call wait() once the flag is true will not block at all.

        """
        with self.__cond:
            self.__flag = True
            self.__cond.notify_all()

def dumps(obj):
    """Outputs json with formatting edits + object handling."""
    return json.dumps(obj, indent=4, sort_keys=True, cls=CustomEncoder)

def estimate_complexity(self, x,y,z,n):
        """ 
        calculates a rough guess of runtime based on product of parameters 
        """
        num_calculations = x * y * z * n
        run_time = num_calculations / 100000  # a 2014 PC does about 100k calcs in a second (guess based on prior logs)
        return self.show_time_as_short_string(run_time)

def read_json(location):
    """Open and load JSON from file.

    location (Path): Path to JSON file.
    RETURNS (dict): Loaded JSON content.
    """
    location = ensure_path(location)
    with location.open('r', encoding='utf8') as f:
        return ujson.load(f)

def timediff(time):
    """Return the difference in seconds between now and the given time."""
    now = datetime.datetime.utcnow()
    diff = now - time
    diff_sec = diff.total_seconds()
    return diff_sec

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

def hms_to_seconds(time_string):
    """
    Converts string 'hh:mm:ss.ssssss' as a float
    """
    s = time_string.split(':')
    hours = int(s[0])
    minutes = int(s[1])
    secs = float(s[2])
    return hours * 3600 + minutes * 60 + secs

def sort_func(self, key):
        """Sorting logic for `Quantity` objects."""
        if key == self._KEYS.VALUE:
            return 'aaa'
        if key == self._KEYS.SOURCE:
            return 'zzz'
        return key

def fromtimestamp(cls, timestamp):
    """Returns a datetime object of a given timestamp (in local tz)."""
    d = cls.utcfromtimestamp(timestamp)
    return d.astimezone(localtz())

def checkbox_uncheck(self, force_check=False):
        """
        Wrapper to uncheck a checkbox
        """
        if self.get_attribute('checked'):
            self.click(force_click=force_check)

def stop_button_click_handler(self):
        """Method to handle what to do when the stop button is pressed"""
        self.stop_button.setDisabled(True)
        # Interrupt computations or stop debugging
        if not self.shellwidget._reading:
            self.interrupt_kernel()
        else:
            self.shellwidget.write_to_stdin('exit')

def closeEvent(self, e):
        """Qt slot when the window is closed."""
        self.emit('close_widget')
        super(DockWidget, self).closeEvent(e)

def on_key_press(self, symbol, modifiers):
        """
        Pyglet specific key press callback.
        Forwards and translates the events to :py:func:`keyboard_event`
        """
        self.keyboard_event(symbol, self.keys.ACTION_PRESS, modifiers)

def _end_del(self):
        """ Deletes the line content after the cursor  """
        text = self.edit_text[:self.edit_pos]
        self.set_edit_text(text)

def sigterm(self, signum, frame):
        """
        These actions will be done after SIGTERM.
        """
        self.logger.warning("Caught signal %s. Stopping daemon." % signum)
        sys.exit(0)

def _get(self, pos):
        """loads widget at given position; handling invalid arguments"""
        res = None, None
        if pos is not None:
            try:
                res = self[pos], pos
            except (IndexError, KeyError):
                pass
        return res

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

def hide(self):
        """Hide the window."""
        self.tk.withdraw()
        self._visible = False
        if self._modal:
            self.tk.grab_release()

def clear_last_lines(self, n):
        """Clear last N lines of terminal output.
        """
        self.term.stream.write(
            self.term.move_up * n + self.term.clear_eos)
        self.term.stream.flush()

def _on_scale(self, event):
        """
        Callback for the Scale widget, inserts an int value into the Entry.

        :param event: Tkinter event
        """
        self._entry.delete(0, tk.END)
        self._entry.insert(0, str(self._variable.get()))

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

def to_pascal_case(s):
    """Transform underscore separated string to pascal case

    """
    return re.sub(r'(?!^)_([a-zA-Z])', lambda m: m.group(1).upper(), s.capitalize())

def Distance(lat1, lon1, lat2, lon2):
    """Get distance between pairs of lat-lon points"""

    az12, az21, dist = wgs84_geod.inv(lon1, lat1, lon2, lat2)
    return az21, dist

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

def write_to(f, mode):
    """Flexible writing, where f can be a filename or f object, if filename, closed after writing"""
    if hasattr(f, 'write'):
        yield f
    else:
        f = open(f, mode)
        yield f
        f.close()

def _aws_get_instance_by_tag(region, name, tag, raw):
    """Get all instances matching a tag."""
    client = boto3.session.Session().client('ec2', region)
    matching_reservations = client.describe_instances(Filters=[{'Name': tag, 'Values': [name]}]).get('Reservations', [])
    instances = []
    [[instances.append(_aws_instance_from_dict(region, instance, raw))  # pylint: disable=expression-not-assigned
      for instance in reservation.get('Instances')] for reservation in matching_reservations if reservation]
    return instances

def lemmatize(self, text, best_guess=True, return_frequencies=False):
		"""Lemmatize all tokens in a string or a list.  A string is first tokenized using punkt.
		Throw a type error if the input is neither a string nor a list.
		"""
		if isinstance(text, str):
			tokens = wordpunct_tokenize(text)
		elif isinstance(text, list):
			tokens= text
		else:
			raise TypeError("lemmatize only works with strings or lists of string tokens.")

		return [self._lemmatize_token(token, best_guess, return_frequencies) for token in tokens]

def _sort_tensor(tensor):
  """Use `top_k` to sort a `Tensor` along the last dimension."""
  sorted_, _ = tf.nn.top_k(tensor, k=tf.shape(input=tensor)[-1])
  sorted_.set_shape(tensor.shape)
  return sorted_

def norm_vec(vector):
    """Normalize the length of a vector to one"""
    assert len(vector) == 3
    v = np.array(vector)
    return v/np.sqrt(np.sum(v**2))

def start(args):
    """Run server with provided command line arguments.
    """
    application = tornado.web.Application([(r"/run", run.get_handler(args)),
                                           (r"/status", run.StatusHandler)])
    application.runmonitor = RunMonitor()
    application.listen(args.port)
    tornado.ioloop.IOLoop.instance().start()

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

async def json_or_text(response):
    """Turns response into a properly formatted json or text object"""
    text = await response.text()
    if response.headers['Content-Type'] == 'application/json; charset=utf-8':
        return json.loads(text)
    return text

def pprint(obj, verbose=False, max_width=79, newline='\n'):
    """
    Like `pretty` but print to stdout.
    """
    printer = RepresentationPrinter(sys.stdout, verbose, max_width, newline)
    printer.pretty(obj)
    printer.flush()
    sys.stdout.write(newline)
    sys.stdout.flush()

def re_raise(self):
        """ Raise this exception with the original traceback """
        if self.exc_info is not None:
            six.reraise(type(self), self, self.exc_info[2])
        else:
            raise self

def glog(x,l = 2):
    """
    Generalised logarithm

    :param x: number
    :param p: number added befor logarithm 

    """
    return np.log((x+np.sqrt(x**2+l**2))/2)/np.log(l)

def walk_tree(root):
    """Pre-order depth-first"""
    yield root

    for child in root.children:
        for el in walk_tree(child):
            yield el

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

def text_remove_empty_lines(text):
    """
    Whitespace normalization:

      - Strip empty lines
      - Strip trailing whitespace
    """
    lines = [ line.rstrip()  for line in text.splitlines()  if line.strip() ]
    return "\n".join(lines)

def to_list(self):
        """Convert this confusion matrix into a 2x2 plain list of values."""
        return [[int(self.table.cell_values[0][1]), int(self.table.cell_values[0][2])],
                [int(self.table.cell_values[1][1]), int(self.table.cell_values[1][2])]]

def remove_trailing_string(content, trailing):
    """
    Strip trailing component `trailing` from `content` if it exists.
    Used when generating names from view classes.
    """
    if content.endswith(trailing) and content != trailing:
        return content[:-len(trailing)]
    return content

def to_list(self):
        """Convert this confusion matrix into a 2x2 plain list of values."""
        return [[int(self.table.cell_values[0][1]), int(self.table.cell_values[0][2])],
                [int(self.table.cell_values[1][1]), int(self.table.cell_values[1][2])]]

def abs_img(img):
    """ Return an image with the binarised version of the data of `img`."""
    bool_img = np.abs(read_img(img).get_data())
    return bool_img.astype(int)

def get_order(self):
        """
        Return a list of dicionaries. See `set_order`.
        """
        return [dict(reverse=r[0], key=r[1]) for r in self.get_model()]

def _unzip_handle(handle):
    """Transparently unzip the file handle"""
    if isinstance(handle, basestring):
        handle = _gzip_open_filename(handle)
    else:
        handle = _gzip_open_handle(handle)
    return handle

def isbinary(*args):
    """Checks if value can be part of binary/bitwise operations."""
    return all(map(lambda c: isnumber(c) or isbool(c), args))

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

def get_table_names(connection):
	"""
	Return a list of the table names in the database.
	"""
	cursor = connection.cursor()
	cursor.execute("SELECT name FROM sqlite_master WHERE type == 'table'")
	return [name for (name,) in cursor]

def compose_all(tups):
  """Compose all given tuples together."""
  from . import ast  # I weep for humanity
  return functools.reduce(lambda x, y: x.compose(y), map(ast.make_tuple, tups), ast.make_tuple({}))

def resources(self):
        """Retrieve contents of each page of PDF"""
        return [self.pdf.getPage(i) for i in range(self.pdf.getNumPages())]

def camel_to_(s):
    """
    Convert CamelCase to camel_case
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def unique(transactions):
    """ Remove any duplicate entries. """
    seen = set()
    # TODO: Handle comments
    return [x for x in transactions if not (x in seen or seen.add(x))]

def _unjsonify(x, isattributes=False):
    """Convert JSON string to an ordered defaultdict."""
    if isattributes:
        obj = json.loads(x)
        return dict_class(obj)
    return json.loads(x)

def readCommaList(fileList):
    """ Return a list of the files with the commas removed. """
    names=fileList.split(',')
    fileList=[]
    for item in names:
        fileList.append(item)
    return fileList

def _convert_to_array(array_like, dtype):
        """
        Convert Matrix attributes which are array-like or buffer to array.
        """
        if isinstance(array_like, bytes):
            return np.frombuffer(array_like, dtype=dtype)
        return np.asarray(array_like, dtype=dtype)

def _get_triplet_value_list(self, graph, identity, rdf_type):
        """
        Get a list of values from RDF triples when more than one may be present
        """
        values = []
        for elem in graph.objects(identity, rdf_type):
            values.append(elem.toPython())
        return values

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

def load_feature(fname, language):
    """ Load and parse a feature file. """

    fname = os.path.abspath(fname)
    feat = parse_file(fname, language)
    return feat

def __get__(self, obj, objtype):
        if not self.is_method:
            self.is_method = True
        """Support instance methods."""
        return functools.partial(self.__call__, obj)

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

def parsed_args():
    parser = argparse.ArgumentParser(description="""python runtime functions""", epilog="")
    parser.add_argument('command',nargs='*',
        help="Name of the function to run with arguments")
    args = parser.parse_args()
    return (args, parser)

def load_feature(fname, language):
    """ Load and parse a feature file. """

    fname = os.path.abspath(fname)
    feat = parse_file(fname, language)
    return feat

def assert_is_not(expected, actual, message=None, extra=None):
    """Raises an AssertionError if expected is actual."""
    assert expected is not actual, _assert_fail_message(
        message, expected, actual, "is", extra
    )

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

def page_title(step, title):
    """
    Check that the page title matches the given one.
    """

    with AssertContextManager(step):
        assert_equals(world.browser.title, title)

def pylog(self, *args, **kwargs):
        """Display all available logging information."""
        printerr(self.name, args, kwargs, traceback.format_exc())

def _correct_args(func, kwargs):
    """
        Convert a dictionary of arguments including __argv into a list
        for passing to the function.
    """
    args = inspect.getargspec(func)[0]
    return [kwargs[arg] for arg in args] + kwargs['__args']

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

def unpack(self, s):
        """Parse bytes and return a namedtuple."""
        return self._create(super(NamedStruct, self).unpack(s))

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

def sarea_(self, col, x=None, y=None, rsum=None, rmean=None):
		"""
		Get an stacked area chart
		"""
		try:
			charts = self._multiseries(col, x, y, "area", rsum, rmean)
			return hv.Area.stack(charts)
		except Exception as e:
			self.err(e, self.sarea_, "Can not draw stacked area chart")

def get_key_by_value(dictionary, search_value):
    """
    searchs a value in a dicionary and returns the key of the first occurrence

    :param dictionary: dictionary to search in
    :param search_value: value to search for
    """
    for key, value in dictionary.iteritems():
        if value == search_value:
            return ugettext(key)

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

async def sysinfo(dev: Device):
    """Print out system information (version, MAC addrs)."""
    click.echo(await dev.get_system_info())
    click.echo(await dev.get_interface_information())

def glr_path_static():
    """Returns path to packaged static files"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '_static'))

def subscribe_to_quorum_channel(self):
        """In case the experiment enforces a quorum, listen for notifications
        before creating Partipant objects.
        """
        from dallinger.experiment_server.sockets import chat_backend

        self.log("Bot subscribing to quorum channel.")
        chat_backend.subscribe(self, "quorum")

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

def copy(self):
        """Return a copy of this list with each element copied to new memory
        """
        out = type(self)()
        for series in self:
            out.append(series.copy())
        return out

def GetPythonLibraryDirectoryPath():
  """Retrieves the Python library directory path."""
  path = sysconfig.get_python_lib(True)
  _, _, path = path.rpartition(sysconfig.PREFIX)

  if path.startswith(os.sep):
    path = path[1:]

  return path

def C_dict2array(C):
    """Convert an OrderedDict containing C values to a 1D array."""
    return np.hstack([np.asarray(C[k]).ravel() for k in C_keys])

def generate_id():
    """Generate new UUID"""
    # TODO: Use six.string_type to Py3 compat
    try:
        return unicode(uuid1()).replace(u"-", u"")
    except NameError:
        return str(uuid1()).replace(u"-", u"")

def on_welcome(self, connection, event):
        """
        Join the channel once connected to the IRC server.
        """
        connection.join(self.channel, key=settings.IRC_CHANNEL_KEY or "")

def check(self, var):
        """Check whether the provided value is a valid enum constant."""
        if not isinstance(var, _str_type): return False
        return _enum_mangle(var) in self._consts

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

def venv():
    """Install venv + deps."""
    try:
        import virtualenv  # NOQA
    except ImportError:
        sh("%s -m pip install virtualenv" % PYTHON)
    if not os.path.isdir("venv"):
        sh("%s -m virtualenv venv" % PYTHON)
    sh("venv\\Scripts\\pip install -r %s" % (REQUIREMENTS_TXT))

def url_encode(url):
    """
    Convert special characters using %xx escape.

    :param url: str
    :return: str - encoded url
    """
    if isinstance(url, text_type):
        url = url.encode('utf8')
    return quote(url, ':/%?&=')

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

def safe_unicode(string):
    """If Python 2, replace non-ascii characters and return encoded string."""
    if not PY3:
        uni = string.replace(u'\u2019', "'")
        return uni.encode('utf-8')
        
    return string

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

def quadratic_bezier(start, end, c0=(0, 0), c1=(0, 0), steps=50):
    """
    Compute quadratic bezier spline given start and end coordinate and
    two control points.
    """
    steps = np.linspace(0, 1, steps)
    sx, sy = start
    ex, ey = end
    cx0, cy0 = c0
    cx1, cy1 = c1
    xs = ((1-steps)**3*sx + 3*((1-steps)**2)*steps*cx0 +
          3*(1-steps)*steps**2*cx1 + steps**3*ex)
    ys = ((1-steps)**3*sy + 3*((1-steps)**2)*steps*cy0 +
          3*(1-steps)*steps**2*cy1 + steps**3*ey)
    return np.column_stack([xs, ys])

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

def eval_script(self, expr):
    """ Evaluates a piece of Javascript in the context of the current page and
    returns its value. """
    ret = self.conn.issue_command("Evaluate", expr)
    return json.loads("[%s]" % ret)[0]

def classnameify(s):
  """
  Makes a classname
  """
  return ''.join(w if w in ACRONYMS else w.title() for w in s.split('_'))

def sine_wave(frequency):
  """Emit a sine wave at the given frequency."""
  xs = tf.reshape(tf.range(_samples(), dtype=tf.float32), [1, _samples(), 1])
  ts = xs / FLAGS.sample_rate
  return tf.sin(2 * math.pi * frequency * ts)

def readwav(filename):
    """Read a WAV file and returns the data and sample rate

    ::

        from spectrum.io import readwav
        readwav()

    """
    from scipy.io.wavfile import read as readwav
    samplerate, signal = readwav(filename)
    return signal, samplerate

def get_auth():
    """Get authentication."""
    import getpass
    user = input("User Name: ")  # noqa
    pswd = getpass.getpass('Password: ')
    return Github(user, pswd)

def get_page_and_url(session, url):
    """
    Download an HTML page using the requests session and return
    the final URL after following redirects.
    """
    reply = get_reply(session, url)
    return reply.text, reply.url

def manhattan_distance_numpy(object1, object2):
    """!
    @brief Calculate Manhattan distance between two objects using numpy.

    @param[in] object1 (array_like): The first array_like object.
    @param[in] object2 (array_like): The second array_like object.

    @return (double) Manhattan distance between two objects.

    """
    return numpy.sum(numpy.absolute(object1 - object2), axis=1).T

def check_by_selector(self, selector):
    """Check the checkbox matching the CSS selector."""
    elem = find_element_by_jquery(world.browser, selector)
    if not elem.is_selected():
        elem.click()

def autobuild_python_test(path):
    """Add pytest unit tests to be built as part of build/test/output."""

    env = Environment(tools=[])
    target = env.Command(['build/test/output/pytest.log'], [path],
                         action=env.Action(run_pytest, "Running python unit tests"))
    env.AlwaysBuild(target)

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

def intersect(d1, d2):
    """Intersect dictionaries d1 and d2 by key *and* value."""
    return dict((k, d1[k]) for k in d1 if k in d2 and d1[k] == d2[k])

def _get_item_position(self, idx):
        """Return a tuple of (start, end) indices of an item from its index."""
        start = 0 if idx == 0 else self._index[idx - 1] + 1
        end = self._index[idx]
        return start, end

def add_arrow(self, x1, y1, x2, y2, **kws):
        """add arrow to plot"""
        self.panel.add_arrow(x1, y1, x2, y2, **kws)

def map_wrap(f):
    """Wrap standard function to easily pass into 'map' processing.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper

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

def get_average_length_of_string(strings):
    """Computes average length of words

    :param strings: list of words
    :return: Average length of word on list
    """
    if not strings:
        return 0

    return sum(len(word) for word in strings) / len(strings)

def heappush_max(heap, item):
    """Push item onto heap, maintaining the heap invariant."""
    heap.append(item)
    _siftdown_max(heap, 0, len(heap) - 1)

def flush(self):
        """Ensure contents are written to file."""
        for name in self.item_names:
            item = self[name]
            item.flush()
        self.file.flush()

def _multiline_width(multiline_s, line_width_fn=len):
    """Visible width of a potentially multiline content."""
    return max(map(line_width_fn, re.split("[\r\n]", multiline_s)))

def base64ToImage(imgData, out_path, out_file):
        """ converts a base64 string to a file """
        fh = open(os.path.join(out_path, out_file), "wb")
        fh.write(imgData.decode('base64'))
        fh.close()
        del fh
        return os.path.join(out_path, out_file)

def get_decimal_quantum(precision):
    """Return minimal quantum of a number, as defined by precision."""
    assert isinstance(precision, (int, decimal.Decimal))
    return decimal.Decimal(10) ** (-precision)

def _serialize_json(obj, fp):
    """ Serialize ``obj`` as a JSON formatted stream to ``fp`` """
    json.dump(obj, fp, indent=4, default=serialize)

def text_width(string, font_name, font_size):
    """Determine with width in pixels of string."""
    return stringWidth(string, fontName=font_name, fontSize=font_size)

def serialize_yaml_tofile(filename, resource):
    """
    Serializes a K8S resource to YAML-formatted file.
    """
    stream = file(filename, "w")
    yaml.dump(resource, stream, default_flow_style=False)

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

def write_str2file(pathname, astr):
    """writes a string to file"""
    fname = pathname
    fhandle = open(fname, 'wb')
    fhandle.write(astr)
    fhandle.close()

def objectproxy_realaddress(obj):
    """
    Obtain a real address as an integer from an objectproxy.
    """
    voidp = QROOT.TPython.ObjectProxy_AsVoidPtr(obj)
    return C.addressof(C.c_char.from_buffer(voidp))

def write_config(self, outfile):
        """Write the configuration dictionary to an output file."""
        utils.write_yaml(self.config, outfile, default_flow_style=False)

def _rel(self, path):
        """
        Get the relative path for the given path from the current
        file by working around https://bugs.python.org/issue20012.
        """
        return os.path.relpath(
            str(path), self._parent).replace(os.path.sep, '/')

def end_block(self):
        """Ends an indentation block, leaving an empty line afterwards"""
        self.current_indent -= 1

        # If we did not add a new line automatically yet, now it's the time!
        if not self.auto_added_line:
            self.writeln()
            self.auto_added_line = True

def merge(self, other):
        """
        Merge this range object with another (ranges need not overlap or abut).

        :returns: a new Range object representing the interval containing both
                  ranges.
        """
        newstart = min(self._start, other.start)
        newend = max(self._end, other.end)
        return Range(newstart, newend)

def _update_bordercolor(self, bordercolor):
        """Updates background color"""

        border_color = wx.SystemSettings_GetColour(wx.SYS_COLOUR_ACTIVEBORDER)
        border_color.SetRGB(bordercolor)

        self.linecolor_choice.SetColour(border_color)

def merge(self, other):
        """
        Merge this range object with another (ranges need not overlap or abut).

        :returns: a new Range object representing the interval containing both
                  ranges.
        """
        newstart = min(self._start, other.start)
        newend = max(self._end, other.end)
        return Range(newstart, newend)

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

def xml_str_to_dict(s):
    """ Transforms an XML string it to python-zimbra dict format

    For format, see:
      https://github.com/Zimbra-Community/python-zimbra/blob/master/README.md

    :param: a string, containing XML
    :returns: a dict, with python-zimbra format
    """
    xml = minidom.parseString(s)
    return pythonzimbra.tools.xmlserializer.dom_to_dict(xml.firstChild)

def send_notice(self, text):
        """Send a notice (from bot) message to the room."""
        return self.client.api.send_notice(self.room_id, text)

def validate(self, xml_input):
        """
        This method validate the parsing and schema, return a boolean
        """
        parsed_xml = etree.parse(self._handle_xml(xml_input))
        try:
            return self.xmlschema.validate(parsed_xml)
        except AttributeError:
            raise CannotValidate('Set XSD to validate the XML')

def pop(self, index=-1):
		"""Remove and return the item at index."""
		value = self._list.pop(index)
		del self._dict[value]
		return value

def is_empty(self):
        """Returns True if the root node contains no child elements, no text,
        and no attributes other than **type**. Returns False if any are present."""
        non_type_attributes = [attr for attr in self.node.attrib.keys() if attr != 'type']
        return len(self.node) == 0 and len(non_type_attributes) == 0 \
            and not self.node.text and not self.node.tail

def get_lons_from_cartesian(x__, y__):
    """Get longitudes from cartesian coordinates.
    """
    return rad2deg(arccos(x__ / sqrt(x__ ** 2 + y__ ** 2))) * sign(y__)

def _extract_node_text(node):
    """Extract text from a given lxml node."""

    texts = map(
        six.text_type.strip, map(six.text_type, map(unescape, node.xpath(".//text()")))
    )
    return " ".join(text for text in texts if text)

def print_yaml(o):
    """Pretty print an object as YAML."""
    print(yaml.dump(o, default_flow_style=False, indent=4, encoding='utf-8'))

def ensure_index(self, key, unique=False):
        """Wrapper for pymongo.Collection.ensure_index
        """
        return self.collection.ensure_index(key, unique=unique)

def extract_zip(zip_path, target_folder):
    """
    Extract the content of the zip-file at `zip_path` into `target_folder`.
    """
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target_folder)

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

def _most_common(iterable):
    """Returns the most common element in `iterable`."""
    data = Counter(iterable)
    return max(data, key=data.__getitem__)

def pieces(array, chunk_size):
        """Yield successive chunks from array/list/string.
        Final chunk may be truncated if array is not evenly divisible by chunk_size."""
        for i in range(0, len(array), chunk_size): yield array[i:i+chunk_size]

def _add_hash(source):
    """Add a leading hash '#' at the beginning of every line in the source."""
    source = '\n'.join('# ' + line.rstrip()
                       for line in source.splitlines())
    return source

def find_lt(a, x):
    """Find rightmost value less than x."""
    i = bs.bisect_left(a, x)
    if i: return i - 1
    raise ValueError

def static_method(cls, f):
        """Decorator which dynamically binds static methods to the model for later use."""
        setattr(cls, f.__name__, staticmethod(f))
        return f

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

def access(self, accessor, timeout=None):
        """Return a result from an asyncio future."""
        if self.loop.is_running():
            raise RuntimeError("Loop is already running")
        coro = asyncio.wait_for(accessor, timeout, loop=self.loop)
        return self.loop.run_until_complete(coro)

def interpolate_nearest(self, xi, yi, zdata):
        """
        Nearest-neighbour interpolation.
        Calls nearnd to find the index of the closest neighbours to xi,yi

        Parameters
        ----------
         xi : float / array of floats, shape (l,)
            x coordinates on the Cartesian plane
         yi : float / array of floats, shape (l,)
            y coordinates on the Cartesian plane

        Returns
        -------
         zi : float / array of floats, shape (l,)
            nearest-neighbour interpolated value(s) of (xi,yi)
        """
        if zdata.size != self.npoints:
            raise ValueError('zdata should be same size as mesh')

        zdata = self._shuffle_field(zdata)

        ist = np.ones_like(xi, dtype=np.int32)
        ist, dist = _tripack.nearnds(xi, yi, ist, self._x, self._y,
                                     self.lst, self.lptr, self.lend)
        return zdata[ist - 1]

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

def ex(self, cmd):
        """Execute a normal python statement in user namespace."""
        with self.builtin_trap:
            exec cmd in self.user_global_ns, self.user_ns

def inject_into_urllib3():
    """
    Monkey-patch urllib3 with SecureTransport-backed SSL-support.
    """
    util.ssl_.SSLContext = SecureTransportContext
    util.HAS_SNI = HAS_SNI
    util.ssl_.HAS_SNI = HAS_SNI
    util.IS_SECURETRANSPORT = True
    util.ssl_.IS_SECURETRANSPORT = True

def mag(z):
    """Get the magnitude of a vector."""
    if isinstance(z[0], np.ndarray):
        return np.array(list(map(np.linalg.norm, z)))
    else:
        return np.linalg.norm(z)

def isemptyfile(filepath):
    """Determine if the file both exists and isempty

    Args:
        filepath (str, path): file path

    Returns:
        bool
    """
    exists = os.path.exists(safepath(filepath))
    if exists:
        filesize = os.path.getsize(safepath(filepath))
        return filesize == 0
    else:
        return False

def norm(x, mu, sigma=1.0):
    """ Scipy norm function """
    return stats.norm(loc=mu, scale=sigma).pdf(x)

def isstring(value):
    """Report whether the given value is a byte or unicode string."""
    classes = (str, bytes) if pyutils.PY3 else basestring  # noqa: F821
    return isinstance(value, classes)

def get_gzipped_contents(input_file):
    """
    Returns a gzipped version of a previously opened file's buffer.
    """
    zbuf = StringIO()
    zfile = GzipFile(mode="wb", compresslevel=6, fileobj=zbuf)
    zfile.write(input_file.read())
    zfile.close()
    return ContentFile(zbuf.getvalue())

def is_enum_type(type_):
    """ Checks if the given type is an enum type.

    :param type_: The type to check
    :return: True if the type is a enum type, otherwise False
    :rtype: bool
    """

    return isinstance(type_, type) and issubclass(type_, tuple(_get_types(Types.ENUM)))

def EvalGaussianPdf(x, mu, sigma):
    """Computes the unnormalized PDF of the normal distribution.

    x: value
    mu: mean
    sigma: standard deviation
    
    returns: float probability density
    """
    return scipy.stats.norm.pdf(x, mu, sigma)

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

def v_normalize(v):
    """
    Normalizes the given vector.
    
    The vector given may have any number of dimensions.
    """
    vmag = v_magnitude(v)
    return [ v[i]/vmag  for i in range(len(v)) ]

def md5_string(s):
    """
    Shortcut to create md5 hash
    :param s:
    :return:
    """
    m = hashlib.md5()
    m.update(s)
    return str(m.hexdigest())

def _normalize(mat: np.ndarray):
    """rescales a numpy array, so that min is 0 and max is 255"""
    return ((mat - mat.min()) * (255 / mat.max())).astype(np.uint8)

def is_seq(obj):
    """ Returns True if object is not a string but is iterable """
    if not hasattr(obj, '__iter__'):
        return False
    if isinstance(obj, basestring):
        return False
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

def bytes_to_str(s, encoding='utf-8'):
    """Returns a str if a bytes object is given."""
    if six.PY3 and isinstance(s, bytes):
        return s.decode(encoding)
    return s

def get_gzipped_contents(input_file):
    """
    Returns a gzipped version of a previously opened file's buffer.
    """
    zbuf = StringIO()
    zfile = GzipFile(mode="wb", compresslevel=6, fileobj=zbuf)
    zfile.write(input_file.read())
    zfile.close()
    return ContentFile(zbuf.getvalue())

def is_valid_folder(parser, arg):
    """Check if arg is a valid file that already exists on the file system."""
    arg = os.path.abspath(arg)
    if not os.path.isdir(arg):
        parser.error("The folder %s does not exist!" % arg)
    else:
        return arg

def count_string_diff(a,b):
    """Return the number of characters in two strings that don't exactly match"""
    shortest = min(len(a), len(b))
    return sum(a[i] != b[i] for i in range(shortest))

def clean(some_string, uppercase=False):
    """
    helper to clean up an input string
    """
    if uppercase:
        return some_string.strip().upper()
    else:
        return some_string.strip().lower()

def dump_nparray(self, obj, class_name=numpy_ndarray_class_name):
        """
        ``numpy.ndarray`` dumper.
        """
        return {"$" + class_name: self._json_convert(obj.tolist())}

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

def _unique_rows_numpy(a):
    """return unique rows"""
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def scale_image(image, new_width):
    """Resizes an image preserving the aspect ratio.
    """
    (original_width, original_height) = image.size
    aspect_ratio = original_height/float(original_width)
    new_height = int(aspect_ratio * new_width)

    # This scales it wider than tall, since characters are biased
    new_image = image.resize((new_width*2, new_height))
    return new_image

def ensure_us_time_resolution(val):
    """Convert val out of numpy time, for use in to_dict.
    Needed because of numpy bug GH#7619"""
    if np.issubdtype(val.dtype, np.datetime64):
        val = val.astype('datetime64[us]')
    elif np.issubdtype(val.dtype, np.timedelta64):
        val = val.astype('timedelta64[us]')
    return val

def setup_request_sessions(self):
        """ Sets up a requests.Session object for sharing headers across API requests.
        """
        self.req_session = requests.Session()
        self.req_session.headers.update(self.headers)

def lognormcdf(x, mu, tau):
    """Log-normal cumulative density function"""
    x = np.atleast_1d(x)
    return np.array(
        [0.5 * (1 - flib.derf(-(np.sqrt(tau / 2)) * (np.log(y) - mu))) for y in x])

def xml_str_to_dict(s):
    """ Transforms an XML string it to python-zimbra dict format

    For format, see:
      https://github.com/Zimbra-Community/python-zimbra/blob/master/README.md

    :param: a string, containing XML
    :returns: a dict, with python-zimbra format
    """
    xml = minidom.parseString(s)
    return pythonzimbra.tools.xmlserializer.dom_to_dict(xml.firstChild)

def push(h, x):
    """Push a new value into heap."""
    h.push(x)
    up(h, h.size()-1)

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

def run(self):
        """
        consume message from channel on the consuming thread.
        """
        LOGGER.debug("rabbitmq.Service.run")
        try:
            self.channel.start_consuming()
        except Exception as e:
            LOGGER.warn("rabbitmq.Service.run - Exception raised while consuming")

def filter_duplicate_key(line, message, line_number, marked_line_numbers,
                         source, previous_line=''):
    """Return '' if first occurrence of the key otherwise return `line`."""
    if marked_line_numbers and line_number == sorted(marked_line_numbers)[0]:
        return ''

    return line

def reseed_random(seed):
    """Reseed factory.fuzzy's random generator."""
    r = random.Random(seed)
    random_internal_state = r.getstate()
    set_random_state(random_internal_state)

def handle_m2m(self, sender, instance, **kwargs):
    """ Handle many to many relationships """
    self.handle_save(instance.__class__, instance)

def csv_to_dicts(file, header=None):
    """Reads a csv and returns a List of Dicts with keys given by header row."""
    with open(file) as csvfile:
        return [row for row in csv.DictReader(csvfile, fieldnames=header)]

def getYamlDocument(filePath):
    """
    Return a yaml file's contents as a dictionary
    """
    with open(filePath) as stream:
        doc = yaml.load(stream)
        return doc

def resources(self):
        """Retrieve contents of each page of PDF"""
        return [self.pdf.getPage(i) for i in range(self.pdf.getNumPages())]

def load_image(fname):
    """ read an image from file - PIL doesnt close nicely """
    with open(fname, "rb") as f:
        i = Image.open(fname)
        #i.load()
        return i

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

def dot_v3(v, w):
    """Return the dotproduct of two vectors."""

    return sum([x * y for x, y in zip(v, w)])

def standard_input():
    """Generator that yields lines from standard input."""
    with click.get_text_stream("stdin") as stdin:
        while stdin.readable():
            line = stdin.readline()
            if line:
                yield line.strip().encode("utf-8")

def _histplot_op(ax, data, **kwargs):
    """Add a histogram for the data to the axes."""
    bins = get_bins(data)
    ax.hist(data, bins=bins, align="left", density=True, **kwargs)
    return ax

def read(fname):
    """Quick way to read a file content."""
    content = None
    with open(os.path.join(here, fname)) as f:
        content = f.read()
    return content

def generate_hash(filepath):
    """Public function that reads a local file and generates a SHA256 hash digest for it"""
    fr = FileReader(filepath)
    data = fr.read_bin()
    return _calculate_sha256(data)

def versions_request(self):
        """List Available REST API Versions"""
        ret = self.handle_api_exceptions('GET', '', api_ver='')
        return [str_dict(x) for x in ret.json()]

def string_to_identity(identity_str):
    """Parse string into Identity dictionary."""
    m = _identity_regexp.match(identity_str)
    result = m.groupdict()
    log.debug('parsed identity: %s', result)
    return {k: v for k, v in result.items() if v}

def url_read_text(url, verbose=True):
    r"""
    Directly reads text data from url
    """
    data = url_read(url, verbose)
    text = data.decode('utf8')
    return text

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

def _read_json_file(self, json_file):
        """ Helper function to read JSON file as OrderedDict """

        self.log.debug("Reading '%s' JSON file..." % json_file)

        with open(json_file, 'r') as f:
            return json.load(f, object_pairs_hook=OrderedDict)

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

def print(cls, *args, **kwargs):
        """Print synchronized."""
        # pylint: disable=protected-access
        with _shared._PRINT_LOCK:
            print(*args, **kwargs)
            _sys.stdout.flush()

def FromString(self, string):
    """Parse a bool from a string."""
    if string.lower() in ("false", "no", "n"):
      return False

    if string.lower() in ("true", "yes", "y"):
      return True

    raise TypeValueError("%s is not recognized as a boolean value." % string)

def replace_all(text, dic):
    """Takes a string and dictionary. replaces all occurrences of i with j"""

    for i, j in dic.iteritems():
        text = text.replace(i, j)
    return text

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

def recursively_get_files_from_directory(directory):
    """
    Return all filenames under recursively found in a directory
    """
    return [
        os.path.join(root, filename)
        for root, directories, filenames in os.walk(directory)
        for filename in filenames
    ]

def trigger(self, target: str, trigger: str, parameters: Dict[str, Any]={}):
		"""Calls the specified Trigger of another Area with the optionally given parameters.

		Args:
			target: The name of the target Area.
			trigger: The name of the Trigger.
			parameters: The parameters of the function call.
		"""
		pass

def recursively_get_files_from_directory(directory):
    """
    Return all filenames under recursively found in a directory
    """
    return [
        os.path.join(root, filename)
        for root, directories, filenames in os.walk(directory)
        for filename in filenames
    ]

def cfloat64_array_to_numpy(cptr, length):
    """Convert a ctypes double pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_double)):
        return np.fromiter(cptr, dtype=np.float64, count=length)
    else:
        raise RuntimeError('Expected double pointer')

def values(self):
        """ :see::meth:RedisMap.keys """
        for val in self._client.hvals(self.key_prefix):
            yield self._loads(val)

def parse_domain(url):
    """ parse the domain from the url """
    domain_match = lib.DOMAIN_REGEX.match(url)
    if domain_match:
        return domain_match.group()

def dict_to_html_attrs(dict_):
    """
    Banana banana
    """
    res = ' '.join('%s="%s"' % (k, v) for k, v in dict_.items())
    return res

def get_numbers(s):
    """Extracts all integers from a string an return them in a list"""

    result = map(int, re.findall(r'[0-9]+', unicode(s)))
    return result + [1] * (2 - len(result))

def cor(y_true, y_pred):
    """Compute Pearson correlation coefficient.
    """
    y_true, y_pred = _mask_nan(y_true, y_pred)
    return np.corrcoef(y_true, y_pred)[0, 1]

def camel_to_(s):
    """
    Convert CamelCase to camel_case
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def test(ctx, all=False, verbose=False):
    """Run the tests."""
    cmd = 'tox' if all else 'py.test'
    if verbose:
        cmd += ' -v'
    return ctx.run(cmd, pty=True).return_code

def is_valid_email(email):
    """
    Check if email is valid
    """
    pattern = re.compile(r'[\w\.-]+@[\w\.-]+[.]\w+')
    return bool(pattern.match(email))

def _linear_interpolation(x, X, Y):
    """Given two data points [X,Y], linearly interpolate those at x.
    """
    return (Y[1] * (x - X[0]) + Y[0] * (X[1] - x)) / (X[1] - X[0])

def lines(input):
    """Remove comments and empty lines"""
    for raw_line in input:
        line = raw_line.strip()
        if line and not line.startswith('#'):
            yield strip_comments(line)

def confusion_matrix(self):
        """Confusion matrix plot
        """
        return plot.confusion_matrix(self.y_true, self.y_pred,
                                     self.target_names, ax=_gen_ax())

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

def confusion_matrix(self):
        """Confusion matrix plot
        """
        return plot.confusion_matrix(self.y_true, self.y_pred,
                                     self.target_names, ax=_gen_ax())

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

def sine_wave(i, frequency=FREQUENCY, framerate=FRAMERATE, amplitude=AMPLITUDE):
    """
    Returns value of a sine wave at a given frequency and framerate
    for a given sample i
    """
    omega = 2.0 * pi * float(frequency)
    sine = sin(omega * (float(i) / float(framerate)))
    return float(amplitude) * sine

def slugify(string):
    """
    Removes non-alpha characters, and converts spaces to hyphens. Useful for making file names.


    Source: http://stackoverflow.com/questions/5574042/string-slugification-in-python
    """
    string = re.sub('[^\w .-]', '', string)
    string = string.replace(" ", "-")
    return string

def normal_noise(points):
    """Init a noise variable."""
    return np.random.rand(1) * np.random.randn(points, 1) \
        + random.sample([2, -2], 1)

def dedup_list(l):
    """Given a list (l) will removing duplicates from the list,
       preserving the original order of the list. Assumes that
       the list entrie are hashable."""
    dedup = set()
    return [ x for x in l if not (x in dedup or dedup.add(x))]

def add_matplotlib_cmap(cm, name=None):
    """Add a matplotlib colormap."""
    global cmaps
    cmap = matplotlib_to_ginga_cmap(cm, name=name)
    cmaps[cmap.name] = cmap

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

def _request(self, data):
        """Moved out to make testing easier."""
        return requests.post(self.endpoint, data=data.encode("ascii")).content

def remove_bad(string):
    """
    remove problem characters from string
    """
    remove = [':', ',', '(', ')', ' ', '|', ';', '\'']
    for c in remove:
        string = string.replace(c, '_')
    return string

def hard_equals(a, b):
    """Implements the '===' operator."""
    if type(a) != type(b):
        return False
    return a == b

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

def prettyprint(d):
        """Print dicttree in Json-like format. keys are sorted
        """
        print(json.dumps(d, sort_keys=True, 
                         indent=4, separators=("," , ": ")))

def normalize_value(text):
    """
    This removes newlines and multiple spaces from a string.
    """
    result = text.replace('\n', ' ')
    result = re.subn('[ ]{2,}', ' ', result)[0]
    return result

def print_matrix(X, decimals=1):
    """Pretty printing for numpy matrix X"""
    for row in np.round(X, decimals=decimals):
        print(row)

def strip_spaces(s):
    """ Strip excess spaces from a string """
    return u" ".join([c for c in s.split(u' ') if c])

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

def strip_accents(string):
    """
    Strip all the accents from the string
    """
    return u''.join(
        (character for character in unicodedata.normalize('NFD', string)
         if unicodedata.category(character) != 'Mn'))

def pp_xml(body):
    """Pretty print format some XML so it's readable."""
    pretty = xml.dom.minidom.parseString(body)
    return pretty.toprettyxml(indent="  ")

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

def insort_no_dup(lst, item):
    """
    If item is not in lst, add item to list at its sorted position
    """
    import bisect
    ix = bisect.bisect_left(lst, item)
    if lst[ix] != item: 
        lst[ix:ix] = [item]

def remove_duplicates(lst):
    """
    Emulate what a Python ``set()`` does, but keeping the element's order.
    """
    dset = set()
    return [l for l in lst if l not in dset and not dset.add(l)]

def auto():
	"""set colouring on if STDOUT is a terminal device, off otherwise"""
	try:
		Style.enabled = False
		Style.enabled = sys.stdout.isatty()
	except (AttributeError, TypeError):
		pass

def clean(s):
  """Removes trailing whitespace on each line."""
  lines = [l.rstrip() for l in s.split('\n')]
  return '\n'.join(lines)

def toggle_word_wrap(self):
        """
        Toggles document word wrap.

        :return: Method success.
        :rtype: bool
        """

        self.setWordWrapMode(not self.wordWrapMode() and QTextOption.WordWrap or QTextOption.NoWrap)
        return True

def strip_head(sequence, values):
    """Strips elements of `values` from the beginning of `sequence`."""
    values = set(values)
    return list(itertools.dropwhile(lambda x: x in values, sequence))

def print_bintree(tree, indent='  '):
    """print a binary tree"""
    for n in sorted(tree.keys()):
        print "%s%s" % (indent * depth(n,tree), n)

def remove_series(self, series):
        """Removes a :py:class:`.Series` from the chart.

        :param Series series: The :py:class:`.Series` to remove.
        :raises ValueError: if you try to remove the last\
        :py:class:`.Series`."""

        if len(self.all_series()) == 1:
            raise ValueError("Cannot remove last series from %s" % str(self))
        self._all_series.remove(series)
        series._chart = None

def log_finished(self):
		"""Log that this task is done."""
		delta = time.perf_counter() - self.start_time
		logger.log("Finished '", logger.cyan(self.name),
			"' after ", logger.magenta(time_to_text(delta)))

def remove_last_entry(self):
        """Remove the last NoteContainer in the Bar."""
        self.current_beat -= 1.0 / self.bar[-1][1]
        self.bar = self.bar[:-1]
        return self.current_beat

def indented_show(text, howmany=1):
        """Print a formatted indented text.
        """
        print(StrTemplate.pad_indent(text=text, howmany=howmany))

def __delitem__(self, key):
        """Remove a variable from this dataset.
        """
        del self._variables[key]
        self._coord_names.discard(key)

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

def _quit(self, *args):
        """ quit crash """
        self.logger.warn('Bye!')
        sys.exit(self.exit())

def filter_none(list_of_points):
    """
    
    :param list_of_points: 
    :return: list_of_points with None's removed
    """
    remove_elementnone = filter(lambda p: p is not None, list_of_points)
    remove_sublistnone = filter(lambda p: not contains_none(p), remove_elementnone)
    return list(remove_sublistnone)

def _load_data(filepath):
  """Loads the images and latent values into Numpy arrays."""
  with h5py.File(filepath, "r") as h5dataset:
    image_array = np.array(h5dataset["images"])
    # The 'label' data set in the hdf5 file actually contains the float values
    # and not the class labels.
    values_array = np.array(h5dataset["labels"])
  return image_array, values_array

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

def qubits(self):
        """Return a list of qubits as (QuantumRegister, index) pairs."""
        return [(v, i) for k, v in self.qregs.items() for i in range(v.size)]

def _clean_name(self, prefix, obj):
        """Create a C variable name with the given prefix based on the name of obj."""
        return '{}{}_{}'.format(prefix, self._uid(), ''.join(c for c in obj.name if c.isalnum()))

def test():  # pragma: no cover
    """Execute the unit tests on an installed copy of unyt.

    Note that this function requires pytest to run. If pytest is not
    installed this function will raise ImportError.
    """
    import pytest
    import os

    pytest.main([os.path.dirname(os.path.abspath(__file__))])

def close_connection (self):
        """
        Close an opened url connection.
        """
        if self.url_connection is None:
            # no connection is open
            return
        try:
            self.url_connection.close()
        except Exception:
            # ignore close errors
            pass
        self.url_connection = None

def lint_file(in_file, out_file=None):
    """Helps remove extraneous whitespace from the lines of a file

    :param file in_file: A readable file or file-like
    :param file out_file: A writable file or file-like
    """
    for line in in_file:
        print(line.strip(), file=out_file)

def prepare_path(path):
    """
    Path join helper method
    Join paths if list passed

    :type path: str|unicode|list
    :rtype: str|unicode
    """
    if type(path) == list:
        return os.path.join(*path)
    return path

def pop(self, index=-1):
		"""Remove and return the item at index."""
		value = self._list.pop(index)
		del self._dict[value]
		return value

def makedirs(path, mode=0o777, exist_ok=False):
    """A wrapper of os.makedirs()."""
    os.makedirs(path, mode, exist_ok)

def text_cleanup(data, key, last_type):
    """ I strip extra whitespace off multi-line strings if they are ready to be stripped!"""
    if key in data and last_type == STRING_TYPE:
        data[key] = data[key].strip()
    return data

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

def text_cleanup(data, key, last_type):
    """ I strip extra whitespace off multi-line strings if they are ready to be stripped!"""
    if key in data and last_type == STRING_TYPE:
        data[key] = data[key].strip()
    return data

def _stdin_(p):
    """Takes input from user. Works for Python 2 and 3."""
    _v = sys.version[0]
    return input(p) if _v is '3' else raw_input(p)

def is_number(obj):
    """Check if obj is number."""
    return isinstance(obj, (int, float, np.int_, np.float_))

def clean_axis(axis):
    """Remove ticks, tick labels, and frame from axis"""
    axis.get_xaxis().set_ticks([])
    axis.get_yaxis().set_ticks([])
    for spine in list(axis.spines.values()):
        spine.set_visible(False)

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

def cleanup_nodes(doc):
    """
    Remove text nodes containing only whitespace
    """
    for node in doc.documentElement.childNodes:
        if node.nodeType == Node.TEXT_NODE and node.nodeValue.isspace():
            doc.documentElement.removeChild(node)
    return doc

def _clear(self):
        """Resets all assigned data for the current message."""
        self._finished = False
        self._measurement = None
        self._message = None
        self._message_body = None

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

def _updateItemComboBoxIndex(self, item, column, num):
        """Callback for comboboxes: notifies us that a combobox for the given item and column has changed"""
        item._combobox_current_index[column] = num
        item._combobox_current_value[column] = item._combobox_option_list[column][num][0]

def string_to_list(string, sep=",", filter_empty=False):
    """Transforma una string con elementos separados por `sep` en una lista."""
    return [value.strip() for value in string.split(sep)
            if (not filter_empty or value)]

def _convert(tup, dictlist):
    """
    :param tup: a list of tuples
    :param di: a dictionary converted from tup
    :return: dictionary
    """
    di = {}
    for a, b in tup:
        di.setdefault(a, []).append(b)
    for key, val in di.items():
        dictlist.append((key, val))
    return dictlist

def dashrepl(value):
    """
    Replace any non-word characters with a dash.
    """
    patt = re.compile(r'\W', re.UNICODE)
    return re.sub(patt, '-', value)

def _cast_boolean(value):
    """
    Helper to convert config values to boolean as ConfigParser do.
    """
    _BOOLEANS = {'1': True, 'yes': True, 'true': True, 'on': True,
                 '0': False, 'no': False, 'false': False, 'off': False, '': False}
    value = str(value)
    if value.lower() not in _BOOLEANS:
        raise ValueError('Not a boolean: %s' % value)

    return _BOOLEANS[value.lower()]

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

def dashrepl(value):
    """
    Replace any non-word characters with a dash.
    """
    patt = re.compile(r'\W', re.UNICODE)
    return re.sub(patt, '-', value)

def _stdin_(p):
    """Takes input from user. Works for Python 2 and 3."""
    _v = sys.version[0]
    return input(p) if _v is '3' else raw_input(p)

def replace_list(items, match, replacement):
    """Replaces occurrences of a match string in a given list of strings and returns
    a list of new strings. The match string can be a regex expression.

    Args:
        items (list):       the list of strings to modify.
        match (str):        the search expression.
        replacement (str):  the string to replace with.
    """
    return [replace(item, match, replacement) for item in items]

def zeros(self, name, **kwargs):
        """Create an array. Keyword arguments as per
        :func:`zarr.creation.zeros`."""
        return self._write_op(self._zeros_nosync, name, **kwargs)

def replace_month_abbr_with_num(date_str, lang=DEFAULT_DATE_LANG):
    """Replace month strings occurrences with month number."""
    num, abbr = get_month_from_date_str(date_str, lang)
    return re.sub(abbr, str(num), date_str, flags=re.IGNORECASE)

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

def clean_with_zeros(self,x):
        """ set nan and inf rows from x to zero"""
        x[~np.any(np.isnan(x) | np.isinf(x),axis=1)] = 0
        return x

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

def replace_all(text, dic):
    """Takes a string and dictionary. replaces all occurrences of i with j"""

    for i, j in dic.iteritems():
        text = text.replace(i, j)
    return text

def add_0x(string):
    """Add 0x to string at start.
    """
    if isinstance(string, bytes):
        string = string.decode('utf-8')
    return '0x' + str(string)

def safe_unicode(string):
    """If Python 2, replace non-ascii characters and return encoded string."""
    if not PY3:
        uni = string.replace(u'\u2019', "'")
        return uni.encode('utf-8')
        
    return string

def submit(self, fn, *args, **kwargs):
        """Submit an operation"""
        corofn = asyncio.coroutine(lambda: fn(*args, **kwargs))
        return run_coroutine_threadsafe(corofn(), self.loop)

def _replace_nan(a, val):
    """
    replace nan in a by val, and returns the replaced array and the nan
    position
    """
    mask = isnull(a)
    return where_method(val, mask, a), mask

def get_geoip(ip):
    """Lookup country for IP address."""
    reader = geolite2.reader()
    ip_data = reader.get(ip) or {}
    return ip_data.get('country', {}).get('iso_code')

def _get_user_agent(self):
        """Retrieve the request's User-Agent, if available.

        Taken from Flask Login utils.py.
        """
        user_agent = request.headers.get('User-Agent')
        if user_agent:
            user_agent = user_agent.encode('utf-8')
        return user_agent or ''

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

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

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

def _parallel_compare_helper(class_obj, pairs, x, x_link=None):
    """Internal function to overcome pickling problem in python2."""
    return class_obj._compute(pairs, x, x_link)

def add_device_callback(self, callback):
        """Register a callback to be invoked when a new device appears."""
        _LOGGER.debug('Added new callback %s ', callback)
        self._cb_new_device.append(callback)

def _list_available_rest_versions(self):
        """Return a list of the REST API versions supported by the array"""
        url = "https://{0}/api/api_version".format(self._target)

        data = self._request("GET", url, reestablish_session=False)
        return data["version"]

def has_edge(self, p_from, p_to):
        """ Returns True when the graph has the given edge. """
        return p_from in self._edges and p_to in self._edges[p_from]

def type(self):
        """Returns type of the data for the given FeatureType."""
        if self is FeatureType.TIMESTAMP:
            return list
        if self is FeatureType.BBOX:
            return BBox
        return dict

async def send_message():
    """Example of sending a message."""
    jar = aiohttp.CookieJar(unsafe=True)
    websession = aiohttp.ClientSession(cookie_jar=jar)

    modem = eternalegypt.Modem(hostname=sys.argv[1], websession=websession)
    await modem.login(password=sys.argv[2])

    await modem.sms(phone=sys.argv[3], message=sys.argv[4])

    await modem.logout()
    await websession.close()

def get_property(self, filename):
        """Opens the file and reads the value"""

        with open(self.filepath(filename)) as f:
            return f.read().strip()

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

def get_date_field(datetimes, field):
    """Adapted from pandas.tslib.get_date_field"""
    return np.array([getattr(date, field) for date in datetimes])

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

def _get_str_columns(sf):
    """
    Returns a list of names of columns that are string type.
    """
    return [name for name in sf.column_names() if sf[name].dtype == str]

def generate_hash(self, length=30):
        """ Generate random string of given length """
        import random, string
        chars = string.ascii_letters + string.digits
        ran = random.SystemRandom().choice
        hash = ''.join(ran(chars) for i in range(length))
        return hash

def most_common(items):
    """
    Wanted functionality from Counters (new in Python 2.7).
    """
    counts = {}
    for i in items:
        counts.setdefault(i, 0)
        counts[i] += 1
    return max(six.iteritems(counts), key=operator.itemgetter(1))

def get_user_name():
    """Get user name provide by operating system
    """

    if sys.platform == 'win32':
        #user = os.getenv('USERPROFILE')
        user = os.getenv('USERNAME')
    else:
        user = os.getenv('LOGNAME')

    return user

def find_lt(a, x):
    """Find rightmost value less than x"""
    i = bisect.bisect_left(a, x)
    if i:
        return a[i-1]
    raise ValueError

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

def do_next(self, args):
        """Step over the next statement
        """
        self._do_print_from_last_cmd = True
        self._interp.step_over()
        return True

def convert_args_to_sets(f):
    """
    Converts all args to 'set' type via self.setify function.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        args = (setify(x) for x in args)
        return f(*args, **kwargs)
    return wrapper

def bytesize(arr):
    """
    Returns the memory byte size of a Numpy array as an integer.
    """
    byte_size = np.prod(arr.shape) * np.dtype(arr.dtype).itemsize
    return byte_size

def _merge_args_with_kwargs(args_dict, kwargs_dict):
    """Merge args with kwargs."""
    ret = args_dict.copy()
    ret.update(kwargs_dict)
    return ret

def __or__(self, other):
        """Return the union of two RangeSets as a new RangeSet.

        (I.e. all elements that are in either set.)
        """
        if not isinstance(other, set):
            return NotImplemented
        return self.union(other)

def apply(f, obj, *args, **kwargs):
    """Apply a function in parallel to each element of the input"""
    return vectorize(f)(obj, *args, **kwargs)

def getTypeStr(_type):
  r"""Gets the string representation of the given type.
  """
  if isinstance(_type, CustomType):
    return str(_type)

  if hasattr(_type, '__name__'):
    return _type.__name__

  return ''

def map_with_obj(f, dct):
    """
        Implementation of Ramda's mapObjIndexed without the final argument.
        This returns the original key with the mapped value. Use map_key_values to modify the keys too
    :param f: Called with a key and value
    :param dct:
    :return {dict}: Keyed by the original key, valued by the mapped value
    """
    f_dict = {}
    for k, v in dct.items():
        f_dict[k] = f(k, v)
    return f_dict

def rgba_bytes_tuple(self, x):
        """Provides the color corresponding to value `x` in the
        form of a tuple (R,G,B,A) with int values between 0 and 255.
        """
        return tuple(int(u*255.9999) for u in self.rgba_floats_tuple(x))

def transform(self, df):
        """
        Transforms a DataFrame in place. Computes all outputs of the DataFrame.

        Args:
            df (pandas.DataFrame): DataFrame to transform.
        """
        for name, function in self.outputs:
            df[name] = function(df)

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

def _rgbtomask(self, obj):
        """Convert RGB arrays from mask canvas object back to boolean mask."""
        dat = obj.get_image().get_data()  # RGB arrays
        return dat.sum(axis=2).astype(np.bool)

def similarity_transformation(rot, mat):
    """ R x M x R^-1 """
    return np.dot(rot, np.dot(mat, np.linalg.inv(rot)))

def process_bool_arg(arg):
    """ Determine True/False from argument """
    if isinstance(arg, bool):
        return arg
    elif isinstance(arg, basestring):
        if arg.lower() in ["true", "1"]:
            return True
        elif arg.lower() in ["false", "0"]:
            return False

def create_rot2d(angle):
    """Create 2D rotation matrix"""
    ca = math.cos(angle)
    sa = math.sin(angle)
    return np.array([[ca, -sa], [sa, ca]])

def email_type(arg):
	"""An argparse type representing an email address."""
	if not is_valid_email_address(arg):
		raise argparse.ArgumentTypeError("{0} is not a valid email address".format(repr(arg)))
	return arg

def floor(self):
    """Round `x` and `y` down to integers."""
    return Point(int(math.floor(self.x)), int(math.floor(self.y)))

def main(args=sys.argv):
    """
    main entry point for the jardiff CLI
    """

    parser = create_optparser(args[0])
    return cli(parser.parse_args(args[1:]))

def round_figures(x, n):
    """Returns x rounded to n significant figures."""
    return round(x, int(n - math.ceil(math.log10(abs(x)))))

def set_subparsers_args(self, *args, **kwargs):
        """
        Sets args and kwargs that are passed when creating a subparsers group
        in an argparse.ArgumentParser i.e. when calling
        argparser.ArgumentParser.add_subparsers
        """
        self.subparsers_args = args
        self.subparsers_kwargs = kwargs

def round_figures(x, n):
    """Returns x rounded to n significant figures."""
    return round(x, int(n - math.ceil(math.log10(abs(x)))))

def zeros(self, name, **kwargs):
        """Create an array. Keyword arguments as per
        :func:`zarr.creation.zeros`."""
        return self._write_op(self._zeros_nosync, name, **kwargs)

async def wait_and_quit(loop):
	"""Wait until all task are executed."""
	from pylp.lib.tasks import running
	if running:
		await asyncio.wait(map(lambda runner: runner.future, running))

def _openResources(self):
        """ Uses numpy.load to open the underlying file
        """
        arr = np.load(self._fileName, allow_pickle=ALLOW_PICKLE)
        check_is_an_array(arr)
        self._array = arr

def runcode(code):
	"""Run the given code line by line with printing, as list of lines, and return variable 'ans'."""
	for line in code:
		print('# '+line)
		exec(line,globals())
	print('# return ans')
	return ans

def cint32_array_to_numpy(cptr, length):
    """Convert a ctypes int pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_int32)):
        return np.fromiter(cptr, dtype=np.int32, count=length)
    else:
        raise RuntimeError('Expected int pointer')

def lint(ctx: click.Context, amend: bool = False, stage: bool = False):
    """
    Runs all linters

    Args:
        ctx: click context
        amend: whether or not to commit results
        stage: whether or not to stage changes
    """
    _lint(ctx, amend, stage)

def c_array(ctype, values):
    """Convert a python string to c array."""
    if isinstance(values, np.ndarray) and values.dtype.itemsize == ctypes.sizeof(ctype):
        return (ctype * len(values)).from_buffer_copy(values)
    return (ctype * len(values))(*values)

def lint(ctx: click.Context, amend: bool = False, stage: bool = False):
    """
    Runs all linters

    Args:
        ctx: click context
        amend: whether or not to commit results
        stage: whether or not to stage changes
    """
    _lint(ctx, amend, stage)

def _transform_triple_numpy(x):
    """Transform triple index into a 1-D numpy array."""
    return np.array([x.head, x.relation, x.tail], dtype=np.int64)

def _save_cookies(requests_cookiejar, filename):
    """Save cookies to a file."""
    with open(filename, 'wb') as handle:
        pickle.dump(requests_cookiejar, handle)

def from_array(cls, arr):
        """Convert a structured NumPy array into a Table."""
        return cls().with_columns([(f, arr[f]) for f in arr.dtype.names])

def seconds_to_hms(input_seconds):
    """Convert seconds to human-readable time."""
    minutes, seconds = divmod(input_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    hours = int(hours)
    minutes = int(minutes)
    seconds = str(int(seconds)).zfill(2)

    return hours, minutes, seconds

def expect_all(a, b):
    """\
    Asserts that two iterables contain the same values.
    """
    assert all(_a == _b for _a, _b in zip_longest(a, b))

def list(self):
        """position in 3d space"""
        return [self._pos3d.x, self._pos3d.y, self._pos3d.z]

def p_if_statement_2(self, p):
        """if_statement : IF LPAREN expr RPAREN statement ELSE statement"""
        p[0] = ast.If(predicate=p[3], consequent=p[5], alternative=p[7])

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

def update_table_row(self, table, row_idx):
        """Add this instance as a row on a `astropy.table.Table` """
        try:
            table[row_idx]['timestamp'] = self.timestamp
            table[row_idx]['status'] = self.status
        except IndexError:
            print("Index error", len(table), row_idx)

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

async def _thread_coro(self, *args):
        """ Coroutine called by MapAsync. It's wrapping the call of
        run_in_executor to run the synchronous function as thread """
        return await self._loop.run_in_executor(
            self._executor, self._function, *args)

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

def check_no_element_by_selector(self, selector):
    """Assert an element does not exist matching the given selector."""
    elems = find_elements_by_jquery(world.browser, selector)
    if elems:
        raise AssertionError("Expected no matching elements, found {}.".format(
            len(elems)))

async def _thread_coro(self, *args):
        """ Coroutine called by MapAsync. It's wrapping the call of
        run_in_executor to run the synchronous function as thread """
        return await self._loop.run_in_executor(
            self._executor, self._function, *args)

def POST(self, *args, **kwargs):
        """ POST request """
        return self._handle_api(self.API_POST, args, kwargs)

def run(*tasks: Awaitable, loop: asyncio.AbstractEventLoop=asyncio.get_event_loop()):
    """Helper to run tasks in the event loop

    :param tasks: Tasks to run in the event loop.
    :param loop: The event loop.
    """
    futures = [asyncio.ensure_future(task, loop=loop) for task in tasks]
    return loop.run_until_complete(asyncio.gather(*futures))

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

def setobjattr(obj, key, value):
    """Sets an object attribute with the correct data type."""
    try:
        setattr(obj, key, int(value))
    except ValueError:
        try:
            setattr(obj, key, float(value))
        except ValueError:
            # string if not number
            try:
                setattr(obj, key, str(value))
            except UnicodeEncodeError:
                setattr(obj, key, value)

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

def _single_page_pdf(page):
    """Construct a single page PDF from the provided page in memory"""
    pdf = Pdf.new()
    pdf.pages.append(page)
    bio = BytesIO()
    pdf.save(bio)
    bio.seek(0)
    return bio.read()

def check_precomputed_distance_matrix(X):
    """Perform check_array(X) after removing infinite values (numpy.inf) from the given distance matrix.
    """
    tmp = X.copy()
    tmp[np.isinf(tmp)] = 1
    check_array(tmp)

def dim_axis_label(dimensions, separator=', '):
    """
    Returns an axis label for one or more dimensions.
    """
    if not isinstance(dimensions, list): dimensions = [dimensions]
    return separator.join([d.pprint_label for d in dimensions])

def setPixel(self, x, y, color):
        """Set the pixel at (x,y) to the integers in sequence 'color'."""
        return _fitz.Pixmap_setPixel(self, x, y, color)

def get_best_encoding(stream):
    """Returns the default stream encoding if not found."""
    rv = getattr(stream, 'encoding', None) or sys.getdefaultencoding()
    if is_ascii_encoding(rv):
        return 'utf-8'
    return rv

def palettebar(height, length, colormap):
    """Return the channels of a palettebar.
    """
    cbar = np.tile(np.arange(length) * 1.0 / (length - 1), (height, 1))
    cbar = (cbar * (colormap.values.max() + 1 - colormap.values.min())
            + colormap.values.min())

    return colormap.palettize(cbar)

def make_file_read_only(file_path):
    """
    Removes the write permissions for the given file for owner, groups and others.

    :param file_path: The file whose privileges are revoked.
    :raise FileNotFoundError: If the given file does not exist.
    """
    old_permissions = os.stat(file_path).st_mode
    os.chmod(file_path, old_permissions & ~WRITE_PERMISSIONS)

def OnMove(self, event):
        """Main window move event"""

        # Store window position in config
        position = self.main_window.GetScreenPositionTuple()

        config["window_position"] = repr(position)

def inpaint(self):
        """ Replace masked-out elements in an array using an iterative image inpainting algorithm. """

        import inpaint
        filled = inpaint.replace_nans(np.ma.filled(self.raster_data, np.NAN).astype(np.float32), 3, 0.01, 2)
        self.raster_data = np.ma.masked_invalid(filled)

def writefile(openedfile, newcontents):
    """Set the contents of a file."""
    openedfile.seek(0)
    openedfile.truncate()
    openedfile.write(newcontents)

def base64ToImage(imgData, out_path, out_file):
        """ converts a base64 string to a file """
        fh = open(os.path.join(out_path, out_file), "wb")
        fh.write(imgData.decode('base64'))
        fh.close()
        del fh
        return os.path.join(out_path, out_file)

def enable_proxy(self, host, port):
        """Enable a default web proxy"""

        self.proxy = [host, _number(port)]
        self.proxy_enabled = True

def base64ToImage(imgData, out_path, out_file):
        """ converts a base64 string to a file """
        fh = open(os.path.join(out_path, out_file), "wb")
        fh.write(imgData.decode('base64'))
        fh.close()
        del fh
        return os.path.join(out_path, out_file)

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

def toBase64(s):
    """Represent string / bytes s as base64, omitting newlines"""
    if isinstance(s, str):
        s = s.encode("utf-8")
    return binascii.b2a_base64(s)[:-1]

def render_template(env, filename, values=None):
    """
    Render a jinja template
    """
    if not values:
        values = {}
    tmpl = env.get_template(filename)
    return tmpl.render(values)

def _set_scroll_v(self, *args):
        """Scroll both categories Canvas and scrolling container"""
        self._canvas_categories.yview(*args)
        self._canvas_scroll.yview(*args)

def update(self):
        """Updates image to be displayed with new time frame."""
        if self.single_channel:
            self.im.set_data(self.data[self.ind, :, :])
        else:
            self.im.set_data(self.data[self.ind, :, :, :])
        self.ax.set_ylabel('time frame %s' % self.ind)
        self.im.axes.figure.canvas.draw()

def root_parent(self, category=None):
        """ Returns the topmost parent of the current category. """
        return next(filter(lambda c: c.is_root, self.hierarchy()))

def comments(tag, limit=0, flags=0, **kwargs):
    """Get comments only."""

    return [comment for comment in cm.CommentsMatch(tag).get_comments(limit)]

def abs_img(img):
    """ Return an image with the binarised version of the data of `img`."""
    bool_img = np.abs(read_img(img).get_data())
    return bool_img.astype(int)

def safe_setattr(obj, name, value):
    """Attempt to setattr but catch AttributeErrors."""
    try:
        setattr(obj, name, value)
        return True
    except AttributeError:
        return False

def main(ctx, connection):
    """Command line interface for PyBEL."""
    ctx.obj = Manager(connection=connection)
    ctx.obj.bind()

def top(n, width=WIDTH, style=STYLE):
    """Prints the top row of a table"""
    return hrule(n, width, linestyle=STYLES[style].top)

def FromString(self, string):
    """Parse a bool from a string."""
    if string.lower() in ("false", "no", "n"):
      return False

    if string.lower() in ("true", "yes", "y"):
      return True

    raise TypeValueError("%s is not recognized as a boolean value." % string)

def smallest_signed_angle(source, target):
    """Find the smallest angle going from angle `source` to angle `target`."""
    dth = target - source
    dth = (dth + np.pi) % (2.0 * np.pi) - np.pi
    return dth

def is_bool_matrix(l):
    r"""Checks if l is a 2D numpy array of bools

    """
    if isinstance(l, np.ndarray):
        if l.ndim == 2 and (l.dtype == bool):
            return True
    return False

def close_error_dlg(self):
        """Close error dialog."""
        if self.error_dlg.dismiss_box.isChecked():
            self.dismiss_error = True
        self.error_dlg.reject()

def isbinary(*args):
    """Checks if value can be part of binary/bitwise operations."""
    return all(map(lambda c: isnumber(c) or isbool(c), args))

def log_request(self, code='-', size='-'):
        """Selectively log an accepted request."""

        if self.server.logRequests:
            BaseHTTPServer.BaseHTTPRequestHandler.log_request(self, code, size)

def list_autoscaling_group(region, filter_by_kwargs):
    """List all Auto Scaling Groups."""
    conn = boto.ec2.autoscale.connect_to_region(region)
    groups = conn.get_all_groups()
    return lookup(groups, filter_by=filter_by_kwargs)

def glog(x,l = 2):
    """
    Generalised logarithm

    :param x: number
    :param p: number added befor logarithm 

    """
    return np.log((x+np.sqrt(x**2+l**2))/2)/np.log(l)

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

def array_size(x, axis):
  """Calculate the size of `x` along `axis` dimensions only."""
  axis_shape = x.shape if axis is None else tuple(x.shape[a] for a in axis)
  return max(numpy.prod(axis_shape), 1)

def batch(items, size):
    """Batches a list into a list of lists, with sub-lists sized by a specified
    batch size."""
    return [items[x:x + size] for x in xrange(0, len(items), size)]

def __len__(self):
        """Return total data length of the list and its headers."""
        return self.chunk_length() + len(self.type) + len(self.header) + 4

def string_to_identity(identity_str):
    """Parse string into Identity dictionary."""
    m = _identity_regexp.match(identity_str)
    result = m.groupdict()
    log.debug('parsed identity: %s', result)
    return {k: v for k, v in result.items() if v}

def __len__(self):
        """Return total data length of the list and its headers."""
        return self.chunk_length() + len(self.type) + len(self.header) + 4

def _to_bstr(l):
    """Convert to byte string."""

    if isinstance(l, str):
        l = l.encode('ascii', 'backslashreplace')
    elif not isinstance(l, bytes):
        l = str(l).encode('ascii', 'backslashreplace')
    return l

def batch(input_iter, batch_size=32):
  """Batches data from an iterator that returns single items at a time."""
  input_iter = iter(input_iter)
  next_ = list(itertools.islice(input_iter, batch_size))
  while next_:
    yield next_
    next_ = list(itertools.islice(input_iter, batch_size))

def array_bytes(array):
    """ Estimates the memory of the supplied array in bytes """
    return np.product(array.shape)*np.dtype(array.dtype).itemsize

def n_choose_k(n, k):
    """ get the number of quartets as n-choose-k. This is used
    in equal splits to decide whether a split should be exhaustively sampled
    or randomly sampled. Edges near tips can be exhaustive while highly nested
    edges probably have too many quartets
    """
    return int(reduce(MUL, (Fraction(n-i, i+1) for i in range(k)), 1))

def _to_bstr(l):
    """Convert to byte string."""

    if isinstance(l, str):
        l = l.encode('ascii', 'backslashreplace')
    elif not isinstance(l, bytes):
        l = str(l).encode('ascii', 'backslashreplace')
    return l

def ratio_and_percentage(current, total, time_remaining):
    """Returns the progress ratio and percentage."""
    return "{} / {} ({}% completed)".format(current, total, int(current / total * 100))

def bytes_to_c_array(data):
    """
    Make a C array using the given string.
    """
    chars = [
        "'{}'".format(encode_escape(i))
        for i in decode_escape(data)
    ]
    return ', '.join(chars) + ', 0'

def MatrixSolve(a, rhs, adj):
    """
    Matrix solve op.
    """
    return np.linalg.solve(a if not adj else _adjoint(a), rhs),

def cpp_prog_builder(build_context, target):
    """Build a C++ binary executable"""
    yprint(build_context.conf, 'Build CppProg', target)
    workspace_dir = build_context.get_workspace('CppProg', target.name)
    build_cpp(build_context, target, target.compiler_config, workspace_dir)

def MatrixSolve(a, rhs, adj):
    """
    Matrix solve op.
    """
    return np.linalg.solve(a if not adj else _adjoint(a), rhs),

def lighting(im, b, c):
    """ Adjust image balance and contrast """
    if b==0 and c==1: return im
    mu = np.average(im)
    return np.clip((im-mu)*c+mu+b,0.,1.).astype(np.float32)

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

def unique_list(lst):
    """Make a list unique, retaining order of initial appearance."""
    uniq = []
    for item in lst:
        if item not in uniq:
            uniq.append(item)
    return uniq

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

def unique_list(lst):
    """Make a list unique, retaining order of initial appearance."""
    uniq = []
    for item in lst:
        if item not in uniq:
            uniq.append(item)
    return uniq

def getOffset(self, loc):
        """ Returns the offset between the given point and this point """
        return Location(loc.x - self.x, loc.y - self.y)

def sort_data(data, cols):
    """Sort `data` rows and order columns"""
    return data.sort_values(cols)[cols + ['value']].reset_index(drop=True)

def elliot_function( signal, derivative=False ):
    """ A fast approximation of sigmoid """
    s = 1 # steepness
    
    abs_signal = (1 + np.abs(signal * s))
    if derivative:
        return 0.5 * s / abs_signal**2
    else:
        # Return the activation signal
        return 0.5*(signal * s) / abs_signal + 0.5

def csort(objs, key):
    """Order-preserving sorting function."""
    idxs = dict((obj, i) for (i, obj) in enumerate(objs))
    return sorted(objs, key=lambda obj: (key(obj), idxs[obj]))

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

def sort_func(self, key):
        """Sorting logic for `Quantity` objects."""
        if key == self._KEYS.VALUE:
            return 'aaa'
        if key == self._KEYS.SOURCE:
            return 'zzz'
        return key

def logical_or(self, other):
        """logical_or(t) = self(t) or other(t)."""
        return self.operation(other, lambda x, y: int(x or y))

def shall_skip(app, module, private):
    """Check if we want to skip this module.

    :param app: the sphinx app
    :type app: :class:`sphinx.application.Sphinx`
    :param module: the module name
    :type module: :class:`str`
    :param private: True, if privates are allowed
    :type private: :class:`bool`
    """
    logger.debug('Testing if %s should be skipped.', module)
    # skip if it has a "private" name and this is selected
    if module != '__init__.py' and module.startswith('_') and \
       not private:
        logger.debug('Skip %s because its either private or __init__.', module)
        return True
    logger.debug('Do not skip %s', module)
    return False

def set_scrollregion(self, event=None):
        """ Set the scroll region on the canvas"""
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

def generate_split_tsv_lines(fn, header):
    """Returns dicts with header-keys and psm statistic values"""
    for line in generate_tsv_psms_line(fn):
        yield {x: y for (x, y) in zip(header, line.strip().split('\t'))}

def to_pascal_case(s):
    """Transform underscore separated string to pascal case

    """
    return re.sub(r'(?!^)_([a-zA-Z])', lambda m: m.group(1).upper(), s.capitalize())

def string_to_identity(identity_str):
    """Parse string into Identity dictionary."""
    m = _identity_regexp.match(identity_str)
    result = m.groupdict()
    log.debug('parsed identity: %s', result)
    return {k: v for k, v in result.items() if v}

def decamelise(text):
    """Convert CamelCase to lower_and_underscore."""
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()

def sqliteRowsToDicts(sqliteRows):
    """
    Unpacks sqlite rows as returned by fetchall
    into an array of simple dicts.

    :param sqliteRows: array of rows returned from fetchall DB call
    :return:  array of dicts, keyed by the column names.
    """
    return map(lambda r: dict(zip(r.keys(), r)), sqliteRows)

def is_equal_strings_ignore_case(first, second):
    """The function compares strings ignoring case"""
    if first and second:
        return first.upper() == second.upper()
    else:
        return not (first or second)

def stack_push(self, thing):
        """
        Push 'thing' to the stack, writing the thing to memory and adjusting the stack pointer.
        """
        # increment sp
        sp = self.regs.sp + self.arch.stack_change
        self.regs.sp = sp
        return self.memory.store(sp, thing, endness=self.arch.memory_endness)

def dt_to_ts(value):
    """ If value is a datetime, convert to timestamp """
    if not isinstance(value, datetime):
        return value
    return calendar.timegm(value.utctimetuple()) + value.microsecond / 1000000.0

def stackplot(marray, seconds=None, start_time=None, ylabels=None):
    """
    will plot a stack of traces one above the other assuming
    marray.shape = numRows, numSamples
    """
    tarray = np.transpose(marray)
    stackplot_t(tarray, seconds=seconds, start_time=start_time, ylabels=ylabels)
    plt.show()

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

async def _send_plain_text(self, request: Request, stack: Stack):
        """
        Sends plain text using `_send_text()`.
        """

        await self._send_text(request, stack, None)

def to_str(obj):
    """Attempts to convert given object to a string object
    """
    if not isinstance(obj, str) and PY3 and isinstance(obj, bytes):
        obj = obj.decode('utf-8')
    return obj if isinstance(obj, string_types) else str(obj)

def weighted_std(values, weights):
    """ Calculate standard deviation weighted by errors """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return np.sqrt(variance)

def convert_value(bind, value):
    """ Type casting. """
    type_name = get_type(bind)
    try:
        return typecast.cast(type_name, value)
    except typecast.ConverterError:
        return value

def _mean_absolute_error(y, y_pred, w):
    """Calculate the mean absolute error."""
    return np.average(np.abs(y_pred - y), weights=w)

def round_to_int(number, precision):
    """Round a number to a precision"""
    precision = int(precision)
    rounded = (int(number) + precision / 2) // precision * precision
    return rounded

def lsem (inlist):
    """
Returns the estimated standard error of the mean (sx-bar) of the
values in the passed list.  sem = stdev / sqrt(n)

Usage:   lsem(inlist)
"""
    sd = stdev(inlist)
    n = len(inlist)
    return sd/math.sqrt(n)

def FromString(self, string):
    """Parse a bool from a string."""
    if string.lower() in ("false", "no", "n"):
      return False

    if string.lower() in ("true", "yes", "y"):
      return True

    raise TypeValueError("%s is not recognized as a boolean value." % string)

def range(*args, interval=0):
    """Generate a given range of numbers.

    It supports the same arguments as the builtin function.
    An optional interval can be given to space the values out.
    """
    agen = from_iterable.raw(builtins.range(*args))
    return time.spaceout.raw(agen, interval) if interval else agen

def load(self, name):
        """Loads and returns foreign library."""
        name = ctypes.util.find_library(name)
        return ctypes.cdll.LoadLibrary(name)

def set_executable(filename):
    """Set the exectuable bit on the given filename"""
    st = os.stat(filename)
    os.chmod(filename, st.st_mode | stat.S_IEXEC)

async def result_processor(tasks):
    """An async result aggregator that combines all the results
       This gets executed in unsync.loop and unsync.thread"""
    output = {}
    for task in tasks:
        num, res = await task
        output[num] = res
    return output

def _update_staticmethod(self, oldsm, newsm):
        """Update a staticmethod update."""
        # While we can't modify the staticmethod object itself (it has no
        # mutable attributes), we *can* extract the underlying function
        # (by calling __get__(), which returns it) and update it in-place.
        # We don't have the class available to pass to __get__() but any
        # object except None will do.
        self._update(None, None, oldsm.__get__(0), newsm.__get__(0))

def find_task_by_id(self, id, session=None):
        """
        Find task with the given record ID.
        """
        with self._session(session) as session:
            return session.query(TaskRecord).get(id)

def standard_deviation(numbers):
    """Return standard deviation."""
    numbers = list(numbers)
    if not numbers:
        return 0
    mean = sum(numbers) / len(numbers)
    return (sum((n - mean) ** 2 for n in numbers) /
            len(numbers)) ** .5

def getExperiments(uuid: str):
    """ list active (running or completed) experiments"""
    return jsonify([x.deserialize() for x in Experiment.query.all()])

def safe_exit(output):
    """exit without breaking pipes."""
    try:
        sys.stdout.write(output)
        sys.stdout.flush()
    except IOError:
        pass

def convert_camel_case_to_snake_case(name):
    """Convert CamelCase to snake_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def stop(self, timeout=None):
        """Stop the thread."""
        logger.debug("docker plugin - Close thread for container {}".format(self._container.name))
        self._stopper.set()

def inc_date(date_obj, num, date_fmt):
    """Increment the date by a certain number and return date object.
    as the specific string format.
    """
    return (date_obj + timedelta(days=num)).strftime(date_fmt)

def stop_button_click_handler(self):
        """Method to handle what to do when the stop button is pressed"""
        self.stop_button.setDisabled(True)
        # Interrupt computations or stop debugging
        if not self.shellwidget._reading:
            self.interrupt_kernel()
        else:
            self.shellwidget.write_to_stdin('exit')

def excepthook(self, except_type, exception, traceback):
    """Not Used: Custom exception hook to replace sys.excepthook

    This is for CPython's default shell. IPython does not use sys.exepthook.

    https://stackoverflow.com/questions/27674602/hide-traceback-unless-a-debug-flag-is-set
    """
    if except_type is DeepReferenceError:
        print(exception.msg)
    else:
        self.default_excepthook(except_type, exception, traceback)

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

def dict_update_newkeys(dict_, dict2):
    """ Like dict.update, but does not overwrite items """
    for key, val in six.iteritems(dict2):
        if key not in dict_:
            dict_[key] = val

def toList(variable, types=(basestring, int, float, )):
    """Converts a variable of type string, int, float to a list, containing the
    variable as the only element.

    :param variable: any python object
    :type variable: (str, int, float, others)

    :returns: [variable] or variable
    """
    if isinstance(variable, types):
        return [variable]
    else:
        return variable

def set_json_item(key, value):
    """ manipulate json data on the fly
    """
    data = get_json()
    data[key] = value

    request = get_request()
    request["BODY"] = json.dumps(data)

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

def _updateItemComboBoxIndex(self, item, column, num):
        """Callback for comboboxes: notifies us that a combobox for the given item and column has changed"""
        item._combobox_current_index[column] = num
        item._combobox_current_value[column] = item._combobox_option_list[column][num][0]

def xpathEvalExpression(self, str):
        """Evaluate the XPath expression in the given context. """
        ret = libxml2mod.xmlXPathEvalExpression(str, self._o)
        if ret is None:raise xpathError('xmlXPathEvalExpression() failed')
        return xpathObjectRet(ret)

def unit_key_from_name(name):
  """Return a legal python name for the given name for use as a unit key."""
  result = name

  for old, new in six.iteritems(UNIT_KEY_REPLACEMENTS):
    result = result.replace(old, new)

  # Collapse redundant underscores and convert to uppercase.
  result = re.sub(r'_+', '_', result.upper())

  return result

def loads(s, model=None, parser=None):
    """Deserialize s (a str) to a Python object."""
    with StringIO(s) as f:
        return load(f, model=model, parser=parser)

def to_percentage(number, rounding=2):
    """Creates a percentage string representation from the given `number`. The
    number is multiplied by 100 before adding a '%' character.

    Raises `ValueError` if `number` cannot be converted to a number.
    """
    number = float(number) * 100
    number_as_int = int(number)
    rounded = round(number, rounding)

    return '{}%'.format(number_as_int if number_as_int == rounded else rounded)

def _dotify(cls, data):
    """Add dots."""
    return ''.join(char if char in cls.PRINTABLE_DATA else '.' for char in data)

def _to_numeric(val):
    """
    Helper function for conversion of various data types into numeric representation.
    """
    if isinstance(val, (int, float, datetime.datetime, datetime.timedelta)):
        return val
    return float(val)

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

def dump_nparray(self, obj, class_name=numpy_ndarray_class_name):
        """
        ``numpy.ndarray`` dumper.
        """
        return {"$" + class_name: self._json_convert(obj.tolist())}

def get_obj(ref):
    """Get object from string reference."""
    oid = int(ref)
    return server.id2ref.get(oid) or server.id2obj[oid]

def set_global(node: Node, key: str, value: Any):
    """Adds passed value to node's globals"""
    node.node_globals[key] = value

def to_bytes(s, encoding="utf-8"):
    """Convert a string to bytes."""
    if isinstance(s, six.binary_type):
        return s
    if six.PY3:
        return bytes(s, encoding)
    return s.encode(encoding)

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

def split_multiline(value):
    """Split a multiline string into a list, excluding blank lines."""
    return [element for element in (line.strip() for line in value.split('\n'))
            if element]

def _check_elements_equal(lst):
    """
    Returns true if all of the elements in the list are equal.
    """
    assert isinstance(lst, list), "Input value must be a list."
    return not lst or lst.count(lst[0]) == len(lst)

def strip_spaces(value, sep=None, join=True):
    """Cleans trailing whitespaces and replaces also multiple whitespaces with a single space."""
    value = value.strip()
    value = [v.strip() for v in value.split(sep)]
    join_sep = sep or ' '
    return join_sep.join(value) if join else value

def empty_tree(input_list):
    """Recursively iterate through values in nested lists."""
    for item in input_list:
        if not isinstance(item, list) or not empty_tree(item):
            return False
    return True

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

def camelcase_underscore(name):
    """ Convert camelcase names to underscore """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def _uptime_syllable():
    """Returns uptime in seconds or None, on Syllable."""
    global __boottime
    try:
        __boottime = os.stat('/dev/pty/mst/pty0').st_mtime
        return time.time() - __boottime
    except (NameError, OSError):
        return None

def filter_dict(d, keys):
    """
    Creates a new dict from an existing dict that only has the given keys
    """
    return {k: v for k, v in d.items() if k in keys}

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

def subn_filter(s, find, replace, count=0):
    """A non-optimal implementation of a regex filter"""
    return re.gsub(find, replace, count, s)

def raise_for_not_ok_status(response):
    """
    Raises a `requests.exceptions.HTTPError` if the response has a non-200
    status code.
    """
    if response.code != OK:
        raise HTTPError('Non-200 response code (%s) for url: %s' % (
            response.code, uridecode(response.request.absoluteURI)))

    return response

def ss_tot(self):
        """Total sum of squares."""
        return np.sum(np.square(self.y - self.ybar), axis=0)

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

def list2string (inlist,delimit=' '):
    """
Converts a 1D list to a single long string for file output, using
the string.join function.

Usage:   list2string (inlist,delimit=' ')
Returns: the string created from inlist
"""
    stringlist = [makestr(_) for _ in inlist]
    return string.join(stringlist,delimit)

async def wait_and_quit(loop):
	"""Wait until all task are executed."""
	from pylp.lib.tasks import running
	if running:
		await asyncio.wait(map(lambda runner: runner.future, running))

def stdout_display():
    """ Print results straight to stdout """
    if sys.version_info[0] == 2:
        yield SmartBuffer(sys.stdout)
    else:
        yield SmartBuffer(sys.stdout.buffer)

def we_are_in_lyon():
    """Check if we are on a Lyon machine"""
    import socket
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        return False
    return ip.startswith("134.158.")

def get_tile_location(self, x, y):
        """Get the screen coordinate for the top-left corner of a tile."""
        x1, y1 = self.origin
        x1 += self.BORDER + (self.BORDER + self.cell_width) * x
        y1 += self.BORDER + (self.BORDER + self.cell_height) * y
        return x1, y1

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

def tfds_dir():
  """Path to tensorflow_datasets directory."""
  return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

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

def do_exit(self, arg):
        """Exit the shell session."""

        if self.current:
            self.current.close()
        self.resource_manager.close()
        del self.resource_manager
        return True

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

def _assert_is_type(name, value, value_type):
    """Assert that a value must be a given type."""
    if not isinstance(value, value_type):
        if type(value_type) is tuple:
            types = ', '.join(t.__name__ for t in value_type)
            raise ValueError('{0} must be one of ({1})'.format(name, types))
        else:
            raise ValueError('{0} must be {1}'
                             .format(name, value_type.__name__))

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

def launched():
    """Test whether the current python environment is the correct lore env.

    :return:  :any:`True` if the environment is launched
    :rtype: bool
    """
    if not PREFIX:
        return False

    return os.path.realpath(sys.prefix) == os.path.realpath(PREFIX)

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

def this_quarter():
        """ Return start and end date of this quarter. """
        since = TODAY + delta(day=1)
        while since.month % 3 != 0:
            since -= delta(months=1)
        until = since + delta(months=3)
        return Date(since), Date(until)

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

def task_property_present_predicate(service, task, prop):
    """ True if the json_element passed is present for the task specified.
    """
    try:
        response = get_service_task(service, task)
    except Exception as e:
        pass

    return (response is not None) and (prop in response)

def querySQL(self, sql, args=()):
        """For use with SELECT (or SELECT-like PRAGMA) statements.
        """
        if self.debug:
            result = timeinto(self.queryTimes, self._queryandfetch, sql, args)
        else:
            result = self._queryandfetch(sql, args)
        return result

def empty_tree(input_list):
    """Recursively iterate through values in nested lists."""
    for item in input_list:
        if not isinstance(item, list) or not empty_tree(item):
            return False
    return True

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

def unique(list):
    """ Returns a copy of the list without duplicates.
    """
    unique = []; [unique.append(x) for x in list if x not in unique]
    return unique

def is_numeric_dtype(dtype):
    """Return ``True`` if ``dtype`` is a numeric type."""
    dtype = np.dtype(dtype)
    return np.issubsctype(getattr(dtype, 'base', None), np.number)

def ms_to_datetime(ms):
    """
    Converts a millisecond accuracy timestamp to a datetime
    """
    dt = datetime.datetime.utcfromtimestamp(ms / 1000)
    return dt.replace(microsecond=(ms % 1000) * 1000).replace(tzinfo=pytz.utc)

def _stdin_ready_posix():
    """Return True if there's something to read on stdin (posix version)."""
    infds, outfds, erfds = select.select([sys.stdin],[],[],0)
    return bool(infds)

def listified_tokenizer(source):
    """Tokenizes *source* and returns the tokens as a list of lists."""
    io_obj = io.StringIO(source)
    return [list(a) for a in tokenize.generate_tokens(io_obj.readline)]

def _using_stdout(self):
        """
        Return whether the handler is using sys.stdout.
        """
        if WINDOWS and colorama:
            # Then self.stream is an AnsiToWin32 object.
            return self.stream.wrapped is sys.stdout

        return self.stream is sys.stdout

def tokenize_list(self, text):
        """
        Split a text into separate words.
        """
        return [self.get_record_token(record) for record in self.analyze(text)]

def isstring(value):
    """Report whether the given value is a byte or unicode string."""
    classes = (str, bytes) if pyutils.PY3 else basestring  # noqa: F821
    return isinstance(value, classes)

def datetime_to_ms(dt):
    """
    Converts a datetime to a millisecond accuracy timestamp
    """
    seconds = calendar.timegm(dt.utctimetuple())
    return seconds * 1000 + int(dt.microsecond / 1000)

def is_hex_string(string):
    """Check if the string is only composed of hex characters."""
    pattern = re.compile(r'[A-Fa-f0-9]+')
    if isinstance(string, six.binary_type):
        string = str(string)
    return pattern.match(string) is not None

def register_view(self, view):
        """Register callbacks for button press events and selection changed"""
        super(ListViewController, self).register_view(view)
        self.tree_view.connect('button_press_event', self.mouse_click)

def _match_literal(self, a, b=None):
        """Match two names."""

        return a.lower() == b if not self.case_sensitive else a == b

def slugify(string):
    """
    Removes non-alpha characters, and converts spaces to hyphens. Useful for making file names.


    Source: http://stackoverflow.com/questions/5574042/string-slugification-in-python
    """
    string = re.sub('[^\w .-]', '', string)
    string = string.replace(" ", "-")
    return string

def default_number_converter(number_str):
    """
    Converts the string representation of a json number into its python object equivalent, an
    int, long, float or whatever type suits.
    """
    is_int = (number_str.startswith('-') and number_str[1:].isdigit()) or number_str.isdigit()
    # FIXME: this handles a wider range of numbers than allowed by the json standard,
    # etc.: float('nan') and float('inf'). But is this a problem?
    return int(number_str) if is_int else float(number_str)

def _open_text(fname, **kwargs):
    """On Python 3 opens a file in text mode by using fs encoding and
    a proper en/decoding errors handler.
    On Python 2 this is just an alias for open(name, 'rt').
    """
    if PY3:
        kwargs.setdefault('encoding', ENCODING)
        kwargs.setdefault('errors', ENCODING_ERRS)
    return open(fname, "rt", **kwargs)

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

def parse_datetime(dt_str):
    """Parse datetime."""
    date_format = "%Y-%m-%dT%H:%M:%S %z"
    dt_str = dt_str.replace("Z", " +0000")
    return datetime.datetime.strptime(dt_str, date_format)

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

def from_array(cls, arr):
        """Convert a structured NumPy array into a Table."""
        return cls().with_columns([(f, arr[f]) for f in arr.dtype.names])

def is_integer(obj):
    """Is this an integer.

    :param object obj:
    :return:
    """
    if PYTHON3:
        return isinstance(obj, int)
    return isinstance(obj, (int, long))

def _str_to_list(s):
    """Converts a comma separated string to a list"""
    _list = s.split(",")
    return list(map(lambda i: i.lstrip(), _list))

def is_executable(path):
  """Returns whether a path names an existing executable file."""
  return os.path.isfile(path) and os.access(path, os.X_OK)

def csvtolist(inputstr):
    """ converts a csv string into a list """
    reader = csv.reader([inputstr], skipinitialspace=True)
    output = []
    for r in reader:
        output += r
    return output

def is_iterable(value):
    """must be an iterable (list, array, tuple)"""
    return isinstance(value, np.ndarray) or isinstance(value, list) or isinstance(value, tuple), value

def as_list(self):
        """Return all child objects in nested lists of strings."""
        return [self.name, self.value, [x.as_list for x in self.children]]

def get_free_memory_win():
    """Return current free memory on the machine for windows.

    Warning : this script is really not robust
    Return in MB unit
    """
    stat = MEMORYSTATUSEX()
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return int(stat.ullAvailPhys / 1024 / 1024)

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

def _match_literal(self, a, b=None):
        """Match two names."""

        return a.lower() == b if not self.case_sensitive else a == b

def covstr(s):
  """ convert string to int or float. """
  try:
    ret = int(s)
  except ValueError:
    ret = float(s)
  return ret

def is_square_matrix(mat):
    """Test if an array is a square matrix."""
    mat = np.array(mat)
    if mat.ndim != 2:
        return False
    shape = mat.shape
    return shape[0] == shape[1]

def to_str(obj):
    """Attempts to convert given object to a string object
    """
    if not isinstance(obj, str) and PY3 and isinstance(obj, bytes):
        obj = obj.decode('utf-8')
    return obj if isinstance(obj, string_types) else str(obj)

def _valid_other_type(x, types):
    """
    Do all elements of x have a type from types?
    """
    return all(any(isinstance(el, t) for t in types) for el in np.ravel(x))

def bytes_to_str(s, encoding='utf-8'):
    """Returns a str if a bytes object is given."""
    if six.PY3 and isinstance(s, bytes):
        return s.decode(encoding)
    return s

def _is_valid_url(self, url):
        """Callback for is_valid_url."""
        try:
            r = requests.head(url, proxies=self.proxy_servers)
            value = r.status_code in [200]
        except Exception as error:
            logger.error(str(error))
            value = False

        return value

def counter(items):
    """
    Simplest required implementation of collections.Counter. Required as 2.6
    does not have Counter in collections.
    """
    results = {}
    for item in items:
        results[item] = results.get(item, 0) + 1
    return results

def is_numeric_dtype(dtype):
    """Return ``True`` if ``dtype`` is a numeric type."""
    dtype = np.dtype(dtype)
    return np.issubsctype(getattr(dtype, 'base', None), np.number)

def _id(self):
        """What this object is equal to."""
        return (self.__class__, self.number_of_needles, self.needle_positions,
                self.left_end_needle)

def isdir(s):
    """Return true if the pathname refers to an existing directory."""
    try:
        st = os.stat(s)
    except os.error:
        return False
    return stat.S_ISDIR(st.st_mode)

def extract_all(zipfile, dest_folder):
    """
    reads the zip file, determines compression
    and unzips recursively until source files 
    are extracted 
    """
    z = ZipFile(zipfile)
    print(z)
    z.extract(dest_folder)

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

def _push_render(self):
        """Render the plot with bokeh.io and push to notebook.
        """
        bokeh.io.push_notebook(handle=self.handle)
        self.last_update = time.time()

def make_writeable(filename):
    """
    Make sure that the file is writeable.
    Useful if our source is read-only.
    """
    if not os.access(filename, os.W_OK):
        st = os.stat(filename)
        new_permissions = stat.S_IMODE(st.st_mode) | stat.S_IWUSR
        os.chmod(filename, new_permissions)

def add_parent(self, parent):
        """
        Adds self as child of parent, then adds parent.
        """
        parent.add_child(self)
        self.parent = parent
        return parent

def find_object(self, object_type):
        """Finds the closest object of a given type."""
        node = self
        while node is not None:
            if isinstance(node.obj, object_type):
                return node.obj
            node = node.parent

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

def file_md5sum(filename):
    """
    :param filename: The filename of the file to process
    :returns: The MD5 hash of the file
    """
    hash_md5 = hashlib.md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 4), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def upcaseTokens(s,l,t):
    """Helper parse action to convert tokens to upper case."""
    return [ tt.upper() for tt in map(_ustr,t) ]

def axes_off(ax):
    """Get rid of all axis ticks, lines, etc.
    """
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)

def closeEvent(self, e):
        """Qt slot when the window is closed."""
        self.emit('close_widget')
        super(DockWidget, self).closeEvent(e)

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

def add_to_js(self, name, var):
        """Add an object to Javascript."""
        frame = self.page().mainFrame()
        frame.addToJavaScriptWindowObject(name, var)

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

def _sourced_dict(self, source=None, **kwargs):
        """Like ``dict(**kwargs)``, but where the ``source`` key is special.
        """
        if source:
            kwargs['source'] = source
        elif self.source:
            kwargs['source'] = self.source
        return kwargs

def detach_all(self):
        """
        Detach from all tracked classes and objects.
        Restore the original constructors and cleanse the tracking lists.
        """
        self.detach_all_classes()
        self.objects.clear()
        self.index.clear()
        self._keepalive[:] = []

def _transform_triple_numpy(x):
    """Transform triple index into a 1-D numpy array."""
    return np.array([x.head, x.relation, x.tail], dtype=np.int64)

def bit_clone( bits ):
    """
    Clone a bitset
    """
    new = BitSet( bits.size )
    new.ior( bits )
    return new

def test_python_java_rt():
    """ Run Python test cases against Java runtime classes. """
    sub_env = {'PYTHONPATH': _build_dir()}

    log.info('Executing Python unit tests (against Java runtime classes)...')
    return jpyutil._execute_python_scripts(python_java_rt_tests,
                                           env=sub_env)

def _clone(self, *args, **kwargs):
        """
        Ensure attributes are copied to subsequent queries.
        """
        for attr in ("_search_terms", "_search_fields", "_search_ordered"):
            kwargs[attr] = getattr(self, attr)
        return super(SearchableQuerySet, self)._clone(*args, **kwargs)

def quote(self, s):
        """Return a shell-escaped version of the string s."""

        if six.PY2:
            from pipes import quote
        else:
            from shlex import quote

        return quote(s)

def kill(self):
        """Kill the browser.

        This is useful when the browser is stuck.
        """
        if self.process:
            self.process.kill()
            self.process.wait()

def invert(dict_):
    """Return an inverted dictionary, where former values are keys
    and former keys are values.

    .. warning::

        If more than one key maps to any given value in input dictionary,
        it is undefined which one will be chosen for the result.

    :param dict_: Dictionary to swap keys and values in
    :return: Inverted dictionary
    """
    ensure_mapping(dict_)
    return dict_.__class__(izip(itervalues(dict_), iterkeys(dict_)))

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

def _write_color_colorama (fp, text, color):
    """Colorize text with given color."""
    foreground, background, style = get_win_color(color)
    colorama.set_console(foreground=foreground, background=background,
      style=style)
    fp.write(text)
    colorama.reset_console()

def cleanup(self, app):
        """Close all connections."""
        if hasattr(self.database.obj, 'close_all'):
            self.database.close_all()

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

def euclidean(c1, c2):
    """Square of the euclidean distance"""
    diffs = ((i - j) for i, j in zip(c1, c2))
    return sum(x * x for x in diffs)

def _is_valid_url(self, url):
        """Callback for is_valid_url."""
        try:
            r = requests.head(url, proxies=self.proxy_servers)
            value = r.status_code in [200]
        except Exception as error:
            logger.error(str(error))
            value = False

        return value

def execute_until_false(method, interval_s):  # pylint: disable=invalid-name
  """Executes a method forever until the method returns a false value.

  Args:
    method: The callable to execute.
    interval_s: The number of seconds to start the execution after each method
        finishes.
  Returns:
    An Interval object.
  """
  interval = Interval(method, stop_if_false=True)
  interval.start(interval_s)
  return interval

def version_jar(self):
		"""
		Special case of version() when the executable is a JAR file.
		"""
		cmd = config.get_command('java')
		cmd.append('-jar')
		cmd += self.cmd
		self.version(cmd=cmd, path=self.cmd[0])

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _get_url(url):
    """Retrieve requested URL"""
    try:
        data = HTTP_SESSION.get(url, stream=True)
        data.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise FetcherException(exc)

    return data

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

def iget_list_column_slice(list_, start=None, stop=None, stride=None):
    """ iterator version of get_list_column """
    if isinstance(start, slice):
        slice_ = start
    else:
        slice_ = slice(start, stop, stride)
    return (row[slice_] for row in list_)

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

def read_next_block(infile, block_size=io.DEFAULT_BUFFER_SIZE):
    """Iterates over the file in blocks."""
    chunk = infile.read(block_size)

    while chunk:
        yield chunk

        chunk = infile.read(block_size)

def sprint(text, *colors):
    """Format text with color or other effects into ANSI escaped string."""
    return "\33[{}m{content}\33[{}m".format(";".join([str(color) for color in colors]), RESET, content=text) if IS_ANSI_TERMINAL and colors else text

def is_valid_file(parser, arg):
    """Check if arg is a valid file that already exists on the file system."""
    arg = os.path.abspath(arg)
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg

def row_to_dict(row):
    """Convert a table row to a dictionary."""
    o = {}
    for colname in row.colnames:

        if isinstance(row[colname], np.string_) and row[colname].dtype.kind in ['S', 'U']:
            o[colname] = str(row[colname])
        else:
            o[colname] = row[colname]

    return o

def is_valid_url(url):
    """Checks if a given string is an url"""
    pieces = urlparse(url)
    return all([pieces.scheme, pieces.netloc])

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

def _valid_other_type(x, types):
    """
    Do all elements of x have a type from types?
    """
    return all(any(isinstance(el, t) for t in types) for el in np.ravel(x))

def merge_dict(data, *args):
    """Merge any number of dictionaries
    """
    results = {}
    for current in (data,) + args:
        results.update(current)
    return results

def make_symmetric(dict):
    """Makes the given dictionary symmetric. Values are assumed to be unique."""
    for key, value in list(dict.items()):
        dict[value] = key
    return dict

def onchange(self, value):
        """Called when a new DropDownItem gets selected.
        """
        log.debug('combo box. selected %s' % value)
        self.select_by_value(value)
        return (value, )

def explained_variance(returns, values):
    """ Calculate how much variance in returns do the values explain """
    exp_var = 1 - torch.var(returns - values) / torch.var(returns)
    return exp_var.item()

def _updateItemComboBoxIndex(self, item, column, num):
        """Callback for comboboxes: notifies us that a combobox for the given item and column has changed"""
        item._combobox_current_index[column] = num
        item._combobox_current_value[column] = item._combobox_option_list[column][num][0]

def apply(f, obj, *args, **kwargs):
    """Apply a function in parallel to each element of the input"""
    return vectorize(f)(obj, *args, **kwargs)

def count_string_diff(a,b):
    """Return the number of characters in two strings that don't exactly match"""
    shortest = min(len(a), len(b))
    return sum(a[i] != b[i] for i in range(shortest))

def cover(session):
    """Run the final coverage report.
    This outputs the coverage report aggregating coverage from the unit
    test runs (not system test runs), and then erases coverage data.
    """
    session.interpreter = 'python3.6'
    session.install('coverage', 'pytest-cov')
    session.run('coverage', 'report', '--show-missing', '--fail-under=100')
    session.run('coverage', 'erase')

def count_string_diff(a,b):
    """Return the number of characters in two strings that don't exactly match"""
    shortest = min(len(a), len(b))
    return sum(a[i] != b[i] for i in range(shortest))

def nrows_expected(self):
        """ based on our axes, compute the expected nrows """
        return np.prod([i.cvalues.shape[0] for i in self.index_axes])

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

def vector_distance(a, b):
    """The Euclidean distance between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

def set_color(self, fg=None, bg=None, intensify=False, target=sys.stdout):
        """Set foreground- and background colors and intensity."""
        raise NotImplementedError

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

def rAsciiLine(ifile):
    """Returns the next non-blank line in an ASCII file."""

    _line = ifile.readline().strip()
    while len(_line) == 0:
        _line = ifile.readline().strip()
    return _line

def compose_all(tups):
  """Compose all given tuples together."""
  from . import ast  # I weep for humanity
  return functools.reduce(lambda x, y: x.compose(y), map(ast.make_tuple, tups), ast.make_tuple({}))

def max(self):
        """
        Returns the maximum value of the domain.

        :rtype: `float` or `np.inf`
        """
        return int(self._max) if not np.isinf(self._max) else self._max

def recarray(self):
        """Returns data as :class:`numpy.recarray`."""
        return numpy.rec.fromrecords(self.records, names=self.names)

def np_counts(self):
    """Dictionary of noun phrase frequencies in this text.
    """
    counts = defaultdict(int)
    for phrase in self.noun_phrases:
        counts[phrase] += 1
    return counts

def less_strict_bool(x):
    """Idempotent and None-safe version of strict_bool."""
    if x is None:
        return False
    elif x is True or x is False:
        return x
    else:
        return strict_bool(x)

def write_config(self, outfile):
        """Write the configuration dictionary to an output file."""
        utils.write_yaml(self.config, outfile, default_flow_style=False)

def student_t(degrees_of_freedom, confidence=0.95):
    """Return Student-t statistic for given DOF and confidence interval."""
    return scipy.stats.t.interval(alpha=confidence,
                                  df=degrees_of_freedom)[-1]

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

def items(self, section_name):
        """:return: list((option, value), ...) pairs of all items in the given section"""
        return [(k, v) for k, v in super(GitConfigParser, self).items(section_name) if k != '__name__']

def _write_json(file, contents):
    """Write a dict to a JSON file."""
    with open(file, 'w') as f:
        return json.dump(contents, f, indent=2, sort_keys=True)

def connect(*args, **kwargs):
    """Creates or returns a singleton :class:`.Connection` object"""
    global __CONNECTION
    if __CONNECTION is None:
        __CONNECTION = Connection(*args, **kwargs)

    return __CONNECTION

def write_tsv_line_from_list(linelist, outfp):
    """Utility method to convert list to tsv line with carriage return"""
    line = '\t'.join(linelist)
    outfp.write(line)
    outfp.write('\n')

def install_postgres(user=None, dbname=None, password=None):
    """Install Postgres on remote"""
    execute(pydiploy.django.install_postgres_server,
            user=user, dbname=dbname, password=password)

def get_hline():
    """ gets a horiztonal line """
    return Window(
        width=LayoutDimension.exact(1),
        height=LayoutDimension.exact(1),
        content=FillControl('-', token=Token.Line))

def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def set_icon(self, bmp):
        """Sets main window icon to given wx.Bitmap"""

        _icon = wx.EmptyIcon()
        _icon.CopyFromBitmap(bmp)
        self.SetIcon(_icon)

def conv_dict(self):
        """dictionary of conversion"""
        return dict(integer=self.integer, real=self.real, no_type=self.no_type)

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

def get_line_flux(line_wave, wave, flux, **kwargs):
    """Interpolated flux at a given wavelength (calls np.interp)."""
    return np.interp(line_wave, wave, flux, **kwargs)

def _do_layout(self):
        """Sizer hell, returns a sizer that contains all widgets"""

        sizer_csvoptions = wx.FlexGridSizer(5, 4, 5, 5)

        # Adding parameter widgets to sizer_csvoptions
        leftpos = wx.LEFT | wx.ADJUST_MINSIZE
        rightpos = wx.RIGHT | wx.EXPAND

        current_label_margin = 0  # smaller for left column
        other_label_margin = 15

        for label, widget in zip(self.param_labels, self.param_widgets):
            sizer_csvoptions.Add(label, 0, leftpos, current_label_margin)
            sizer_csvoptions.Add(widget, 0, rightpos, current_label_margin)

            current_label_margin, other_label_margin = \
                other_label_margin, current_label_margin

        sizer_csvoptions.AddGrowableCol(1)
        sizer_csvoptions.AddGrowableCol(3)

        self.sizer_csvoptions = sizer_csvoptions

def camelcase_underscore(name):
    """ Convert camelcase names to underscore """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def coords_on_grid(self, x, y):
        """ Snap coordinates on the grid with integer coordinates """

        if isinstance(x, float):
            x = int(self._round(x))
        if isinstance(y, float):
            y = int(self._round(y))
        if not self._y_coord_down:
            y = self._extents - y
        return x, y

def int_to_date(date):
    """
    Convert an int of form yyyymmdd to a python date object.
    """

    year = date // 10**4
    month = date % 10**4 // 10**2
    day = date % 10**2

    return datetime.date(year, month, day)

def set_xlimits(self, row, column, min=None, max=None):
        """Set x-axis limits of a subplot.

        :param row,column: specify the subplot.
        :param min: minimal axis value
        :param max: maximum axis value

        """
        subplot = self.get_subplot_at(row, column)
        subplot.set_xlimits(min, max)

def string_to_float_list(string_var):
        """Pull comma separated string values out of a text file and converts them to float list"""
        try:
            return [float(s) for s in string_var.strip('[').strip(']').split(', ')]
        except:
            return [float(s) for s in string_var.strip('[').strip(']').split(',')]

def multiply(traj):
    """Sophisticated simulation of multiplication"""
    z=traj.x*traj.y
    traj.f_add_result('z',z=z, comment='I am the product of two reals!')

def convert_camel_case_to_snake_case(name):
    """Convert CamelCase to snake_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def schemaParse(self):
        """parse a schema definition resource and build an internal
           XML Shema struture which can be used to validate instances. """
        ret = libxml2mod.xmlSchemaParse(self._o)
        if ret is None:raise parserError('xmlSchemaParse() failed')
        __tmp = Schema(_obj=ret)
        return __tmp

def convolve_fft(array, kernel):
    """
    Convolve an array with a kernel using FFT.
    Implemntation based on the convolve_fft function from astropy.

    https://github.com/astropy/astropy/blob/master/astropy/convolution/convolve.py
    """

    array = np.asarray(array, dtype=np.complex)
    kernel = np.asarray(kernel, dtype=np.complex)

    if array.ndim != kernel.ndim:
        raise ValueError("Image and kernel must have same number of "
                         "dimensions")

    array_shape = array.shape
    kernel_shape = kernel.shape
    new_shape = np.array(array_shape) + np.array(kernel_shape)

    array_slices = []
    kernel_slices = []
    for (new_dimsize, array_dimsize, kernel_dimsize) in zip(
            new_shape, array_shape, kernel_shape):
        center = new_dimsize - (new_dimsize + 1) // 2
        array_slices += [slice(center - array_dimsize // 2,
                         center + (array_dimsize + 1) // 2)]
        kernel_slices += [slice(center - kernel_dimsize // 2,
                          center + (kernel_dimsize + 1) // 2)]

    array_slices = tuple(array_slices)
    kernel_slices = tuple(kernel_slices)

    if not np.all(new_shape == array_shape):
        big_array = np.zeros(new_shape, dtype=np.complex)
        big_array[array_slices] = array
    else:
        big_array = array

    if not np.all(new_shape == kernel_shape):
        big_kernel = np.zeros(new_shape, dtype=np.complex)
        big_kernel[kernel_slices] = kernel
    else:
        big_kernel = kernel

    array_fft = np.fft.fftn(big_array)
    kernel_fft = np.fft.fftn(np.fft.ifftshift(big_kernel))

    rifft = np.fft.ifftn(array_fft * kernel_fft)

    return rifft[array_slices].real

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

def convolve_gaussian_2d(image, gaussian_kernel_1d):
    """Convolve 2d gaussian."""
    result = scipy.ndimage.filters.correlate1d(
        image, gaussian_kernel_1d, axis=0)
    result = scipy.ndimage.filters.correlate1d(
        result, gaussian_kernel_1d, axis=1)
    return result

def convolve_gaussian_2d(image, gaussian_kernel_1d):
    """Convolve 2d gaussian."""
    result = scipy.ndimage.filters.correlate1d(
        image, gaussian_kernel_1d, axis=0)
    result = scipy.ndimage.filters.correlate1d(
        result, gaussian_kernel_1d, axis=1)
    return result

def pad_cells(table):
    """Pad each cell to the size of the largest cell in its column."""
    col_sizes = [max(map(len, col)) for col in zip(*table)]
    for row in table:
        for cell_num, cell in enumerate(row):
            row[cell_num] = pad_to(cell, col_sizes[cell_num])
    return table

def execfile(fname, variables):
    """ This is builtin in python2, but we have to roll our own on py3. """
    with open(fname) as f:
        code = compile(f.read(), fname, 'exec')
        exec(code, variables)

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

def ungzip_data(input_data):
    """Return a string of data after gzip decoding

    :param the input gziped data
    :return  the gzip decoded data"""
    buf = StringIO(input_data)
    f = gzip.GzipFile(fileobj=buf)
    return f

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

def _gcd_array(X):
    """
    Return the largest real value h such that all elements in x are integer
    multiples of h.
    """
    greatest_common_divisor = 0.0
    for x in X:
        greatest_common_divisor = _gcd(greatest_common_divisor, x)

    return greatest_common_divisor

def notin(arg, values):
    """
    Like isin, but checks whether this expression's value(s) are not
    contained in the passed values. See isin docs for full usage.
    """
    op = ops.NotContains(arg, values)
    return op.to_expr()

def np_hash(a):
    """Return a hash of a NumPy array."""
    if a is None:
        return hash(None)
    # Ensure that hashes are equal whatever the ordering in memory (C or
    # Fortran)
    a = np.ascontiguousarray(a)
    # Compute the digest and return a decimal int
    return int(hashlib.sha1(a.view(a.dtype)).hexdigest(), 16)

def string_presenter(self, dumper, data):
    """Presenter to force yaml.dump to use multi-line string style."""
    if '\n' in data:
      return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    else:
      return dumper.represent_scalar('tag:yaml.org,2002:str', data)

def join_cols(cols):
    """Join list of columns into a string for a SQL query"""
    return ", ".join([i for i in cols]) if isinstance(cols, (list, tuple, set)) else cols

def bytes_base64(x):
    """Turn bytes into base64"""
    if six.PY2:
        return base64.encodestring(x).replace('\n', '')
    return base64.encodebytes(bytes_encode(x)).replace(b'\n', b'')

def date_to_datetime(x):
    """Convert a date into a datetime"""
    if not isinstance(x, datetime) and isinstance(x, date):
        return datetime.combine(x, time())
    return x

def uint32_to_uint8(cls, img):
        """
        Cast uint32 RGB image to 4 uint8 channels.
        """
        return np.flipud(img.view(dtype=np.uint8).reshape(img.shape + (4,)))

def tokenize(string):
    """Match and yield all the tokens of the input string."""
    for match in TOKENS_REGEX.finditer(string):
        yield Token(match.lastgroup, match.group().strip(), match.span())

def normalize_enum_constant(s):
    """Return enum constant `s` converted to a canonical snake-case."""
    if s.islower(): return s
    if s.isupper(): return s.lower()
    return "".join(ch if ch.islower() else "_" + ch.lower() for ch in s).strip("_")

def torecarray(*args, **kwargs):
    """
    Convenient shorthand for ``toarray(*args, **kwargs).view(np.recarray)``.

    """

    import numpy as np
    return toarray(*args, **kwargs).view(np.recarray)

def fromDict(cls, _dict):
        """ Builds instance from dictionary of properties. """
        obj = cls()
        obj.__dict__.update(_dict)
        return obj

def array(self):
        """
        The underlying array of shape (N, L, I)
        """
        return numpy.array([self[sid].array for sid in sorted(self)])

def instance_contains(container, item):
    """Search into instance attributes, properties and return values of no-args methods."""
    return item in (member for _, member in inspect.getmembers(container))

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def get_line_flux(line_wave, wave, flux, **kwargs):
    """Interpolated flux at a given wavelength (calls np.interp)."""
    return np.interp(line_wave, wave, flux, **kwargs)

def line_line_intersect(x, y):
    """Compute the intersection point of two lines

    Parameters
    ----------
    x = x4 array: x1, x2, x3, x4
    y = x4 array: y1, y2, y3, y4
    line 1 is defined by p1,p2
    line 2 is defined by p3,p4

    Returns
    -------
    Ix: x-coordinate of intersection
    Iy: y-coordinate of intersection
    """
    A = x[0] * y[1] - y[0] * x[1]
    B = x[2] * y[3] - y[2] * x[4]
    C = (x[0] - x[1]) * (y[2] - y[3]) - (y[0] - y[1]) * (x[2] - x[3])

    Ix = (A * (x[2] - x[3]) - (x[0] - x[1]) * B) / C
    Iy = (A * (y[2] - y[3]) - (y[0] - y[1]) * B) / C
    return Ix, Iy

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

def from_series(series):
        """
        Deseralize a PercentRankTransform the given pandas.Series, as returned
        by `to_series()`.

        Parameters
        ----------
        series : pandas.Series

        Returns
        -------
        PercentRankTransform

        """
        result = PercentRankTransform()
        result.cdf = series.values
        result.bin_edges = series.index.values[1:-1]
        return result

def convolve_gaussian_2d(image, gaussian_kernel_1d):
    """Convolve 2d gaussian."""
    result = scipy.ndimage.filters.correlate1d(
        image, gaussian_kernel_1d, axis=0)
    result = scipy.ndimage.filters.correlate1d(
        result, gaussian_kernel_1d, axis=1)
    return result

def get_cursor(self):
        """Returns current grid cursor cell (row, col, tab)"""

        return self.grid.GetGridCursorRow(), self.grid.GetGridCursorCol(), \
            self.grid.current_table

def copyFile(input, output, replace=None):
    """Copy a file whole from input to output."""

    _found = findFile(output)
    if not _found or (_found and replace):
        shutil.copy2(input, output)

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

def c_array(ctype, values):
    """Convert a python string to c array."""
    if isinstance(values, np.ndarray) and values.dtype.itemsize == ctypes.sizeof(ctype):
        return (ctype * len(values)).from_buffer_copy(values)
    return (ctype * len(values))(*values)

def find_root(self):
        """ Traverse parent refs to top. """
        cmd = self
        while cmd.parent:
            cmd = cmd.parent
        return cmd

def main(argv, version=DEFAULT_VERSION):
    """Install or upgrade setuptools and EasyInstall"""
    tarball = download_setuptools()
    _install(tarball, _build_install_args(argv))

def parse_form(self, req, name, field):
        """Pull a form value from the request."""
        return get_value(req.body_arguments, name, field)

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

def _increment_numeric_suffix(s):
        """Increment (or add) numeric suffix to identifier."""
        if re.match(r".*\d+$", s):
            return re.sub(r"\d+$", lambda n: str(int(n.group(0)) + 1), s)
        return s + "_2"

def count(lines):
  """ Counts the word frequences in a list of sentences.

  Note:
    This is a helper function for parallel execution of `Vocabulary.from_text`
    method.
  """
  words = [w for l in lines for w in l.strip().split()]
  return Counter(words)

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

def convert_timestamp(timestamp):
    """
    Converts bokehJS timestamp to datetime64.
    """
    datetime = dt.datetime.utcfromtimestamp(timestamp/1000.)
    return np.datetime64(datetime.replace(tzinfo=None))

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

def get_ntobj(self):
        """Create namedtuple object with GOEA fields."""
        if self.nts:
            return cx.namedtuple("ntgoea", " ".join(vars(next(iter(self.nts))).keys()))

def array_sha256(a):
    """Create a SHA256 hash from a Numpy array."""
    dtype = str(a.dtype).encode()
    shape = numpy.array(a.shape)
    sha = hashlib.sha256()
    sha.update(dtype)
    sha.update(shape)
    sha.update(a.tobytes())
    return sha.hexdigest()

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

def create_run_logfile(folder):
    """Create a 'run.log' within folder. This file contains the time of the
       latest successful run.
    """
    with open(os.path.join(folder, "run.log"), "w") as f:
        datestring = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        f.write("timestamp: '%s'" % datestring)

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

def construct_from_string(cls, string):
        """
        Construction from a string, raise a TypeError if not
        possible
        """
        if string == cls.name:
            return cls()
        raise TypeError("Cannot construct a '{}' from "
                        "'{}'".format(cls, string))

def weighted_std(values, weights):
    """ Calculate standard deviation weighted by errors """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return np.sqrt(variance)

def be_array_from_bytes(fmt, data):
    """
    Reads an array from bytestring with big-endian data.
    """
    arr = array.array(str(fmt), data)
    return fix_byteorder(arr)

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

def to_dict(self):
        """
        Serialize representation of the column for local caching.
        """
        return {'schema': self.schema, 'table': self.table, 'name': self.name, 'type': self.type}

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

def makedirs(directory):
    """ Resursively create a named directory. """
    parent = os.path.dirname(os.path.abspath(directory))
    if not os.path.exists(parent):
        makedirs(parent)
    os.mkdir(directory)

def indent(s, spaces=4):
    """
    Inserts `spaces` after each string of new lines in `s`
    and before the start of the string.
    """
    new = re.sub('(\n+)', '\\1%s' % (' ' * spaces), s)
    return (' ' * spaces) + new.strip()

def makedirs(directory):
    """ Resursively create a named directory. """
    parent = os.path.dirname(os.path.abspath(directory))
    if not os.path.exists(parent):
        makedirs(parent)
    os.mkdir(directory)

def  make_html_code( self, lines ):
        """ convert a code sequence to HTML """
        line = code_header + '\n'
        for l in lines:
            line = line + html_quote( l ) + '\n'

        return line + code_footer

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

def CreateVertices(self, points):
        """
        Returns a dictionary object with keys that are 2tuples
        represnting a point.
        """
        gr = digraph()

        for z, x, Q in points:
            node = (z, x, Q)
            gr.add_nodes([node])

        return gr

def generate_hash(filepath):
    """Public function that reads a local file and generates a SHA256 hash digest for it"""
    fr = FileReader(filepath)
    data = fr.read_bin()
    return _calculate_sha256(data)

def get_table_width(table):
    """
    Gets the width of the table that would be printed.
    :rtype: ``int``
    """
    columns = transpose_table(prepare_rows(table))
    widths = [max(len(cell) for cell in column) for column in columns]
    return len('+' + '|'.join('-' * (w + 2) for w in widths) + '+')

def unit_key_from_name(name):
  """Return a legal python name for the given name for use as a unit key."""
  result = name

  for old, new in six.iteritems(UNIT_KEY_REPLACEMENTS):
    result = result.replace(old, new)

  # Collapse redundant underscores and convert to uppercase.
  result = re.sub(r'_+', '_', result.upper())

  return result

def normalize_time(timestamp):
    """Normalize time in arbitrary timezone to UTC naive object."""
    offset = timestamp.utcoffset()
    if offset is None:
        return timestamp
    return timestamp.replace(tzinfo=None) - offset

def split_elements(value):
    """Split a string with comma or space-separated elements into a list."""
    l = [v.strip() for v in value.split(',')]
    if len(l) == 1:
        l = value.split()
    return l

def transformer_ae_a3():
  """Set of hyperparameters."""
  hparams = transformer_ae_base()
  hparams.batch_size = 4096
  hparams.layer_prepostprocess_dropout = 0.3
  hparams.optimizer = "Adafactor"
  hparams.learning_rate = 0.25
  hparams.learning_rate_warmup_steps = 10000
  return hparams

def vec_angle(a, b):
    """
    Calculate angle between two vectors
    """
    cosang = np.dot(a, b)
    sinang = fast_norm(np.cross(a, b))
    return np.arctan2(sinang, cosang)

def sp_rand(m,n,a):
    """
    Generates an mxn sparse 'd' matrix with round(a*m*n) nonzeros.
    """
    if m == 0 or n == 0: return spmatrix([], [], [], (m,n))
    nnz = min(max(0, int(round(a*m*n))), m*n)
    nz = matrix(random.sample(range(m*n), nnz), tc='i')
    return spmatrix(normal(nnz,1), nz%m, matrix([int(ii) for ii in nz/m]), (m,n))

def prepend_line(filepath, line):
    """Rewrite a file adding a line to its beginning.
    """
    with open(filepath) as f:
        lines = f.readlines()

    lines.insert(0, line)

    with open(filepath, 'w') as f:
        f.writelines(lines)

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

def create_path(path):
    """Creates a absolute path in the file system.

    :param path: The path to be created
    """
    import os
    if not os.path.exists(path):
        os.makedirs(path)

def interpolate(table, field, fmt, **kwargs):
    """
    Convenience function to interpolate all values in the given `field` using
    the `fmt` string.

    The ``where`` keyword argument can be given with a callable or expression
    which is evaluated on each row and which should return True if the
    conversion should be applied on that row, else False.

    """

    conv = lambda v: fmt % v
    return convert(table, field, conv, **kwargs)

def csvpretty(csvfile: csvfile=sys.stdin):
    """ Pretty print a CSV file. """
    shellish.tabulate(csv.reader(csvfile))

def to_linspace(self):
        """
        convert from full to linspace
        """
        if hasattr(self.shape, '__len__'):
            raise NotImplementedError("can only convert flat Full arrays to linspace")
        return Linspace(self.fill_value, self.fill_value, self.shape)

def cint8_array_to_numpy(cptr, length):
    """Convert a ctypes int pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_int8)):
        return np.fromiter(cptr, dtype=np.int8, count=length)
    else:
        raise RuntimeError('Expected int pointer')

def c_array(ctype, values):
    """Convert a python string to c array."""
    if isinstance(values, np.ndarray) and values.dtype.itemsize == ctypes.sizeof(ctype):
        return (ctype * len(values)).from_buffer_copy(values)
    return (ctype * len(values))(*values)

def hex_escape(bin_str):
  """
  Hex encode a binary string
  """
  printable = string.ascii_letters + string.digits + string.punctuation + ' '
  return ''.join(ch if ch in printable else r'0x{0:02x}'.format(ord(ch)) for ch in bin_str)

def getBuffer(x):
    """
    Copy @x into a (modifiable) ctypes byte array
    """
    b = bytes(x)
    return (c_ubyte * len(b)).from_buffer_copy(bytes(x))

def rex_assert(self, rex, byte=False):
        """
        If `rex` expression is not found then raise `DataNotFound` exception.
        """

        self.rex_search(rex, byte=byte)

def _monitor_callback_wrapper(callback):
    """A wrapper for the user-defined handle."""
    def callback_handle(name, array, _):
        """ ctypes function """
        callback(name, array)
    return callback_handle

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

def getBuffer(x):
    """
    Copy @x into a (modifiable) ctypes byte array
    """
    b = bytes(x)
    return (c_ubyte * len(b)).from_buffer_copy(bytes(x))

def destroy(self):
		"""Finish up a session.
		"""
		if self.session_type == 'bash':
			# TODO: does this work/handle already being logged out/logged in deep OK?
			self.logout()
		elif self.session_type == 'vagrant':
			# TODO: does this work/handle already being logged out/logged in deep OK?
			self.logout()

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

def accel_prev(self, *args):
        """Callback to go to the previous tab. Called by the accel key.
        """
        if self.get_notebook().get_current_page() == 0:
            self.get_notebook().set_current_page(self.get_notebook().get_n_pages() - 1)
        else:
            self.get_notebook().prev_page()
        return True

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

def min_max_normalize(img):
    """Centre and normalize a given array.

    Parameters:
    ----------
    img: np.ndarray

    """

    min_img = img.min()
    max_img = img.max()

    return (img - min_img) / (max_img - min_img)

def cint32_array_to_numpy(cptr, length):
    """Convert a ctypes int pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_int32)):
        return np.fromiter(cptr, dtype=np.int32, count=length)
    else:
        raise RuntimeError('Expected int pointer')

def s3(ctx, bucket_name, data_file, region):
    """Use the S3 SWAG backend."""
    if not ctx.data_file:
        ctx.data_file = data_file

    if not ctx.bucket_name:
        ctx.bucket_name = bucket_name

    if not ctx.region:
        ctx.region = region

    ctx.type = 's3'

def now(self):
		"""
		Return a :py:class:`datetime.datetime` instance representing the current time.

		:rtype: :py:class:`datetime.datetime`
		"""
		if self.use_utc:
			return datetime.datetime.utcnow()
		else:
			return datetime.datetime.now()

def get_files(client, bucket, prefix=''):
    """Lists files/objects on a bucket.
    
    TODO: docstring"""
    bucket = client.get_bucket(bucket)
    files = list(bucket.list_blobs(prefix=prefix))    
    return files

def get_capture_dimensions(capture):
    """Get the dimensions of a capture"""
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return width, height

def exists(self, digest):
        """
        Check if a blob exists

        :param digest: Hex digest of the blob
        :return: Boolean indicating existence of the blob
        """
        return self.conn.client.blob_exists(self.container_name, digest)

def SetValue(self, row, col, value):
        """
        Set value in the pandas DataFrame
        """
        self.dataframe.iloc[row, col] = value

def s2b(s):
    """
    String to binary.
    """
    ret = []
    for c in s:
        ret.append(bin(ord(c))[2:].zfill(8))
    return "".join(ret)

def weekly(date=datetime.date.today()):
    """
    Weeks start are fixes at Monday for now.
    """
    return date - datetime.timedelta(days=date.weekday())

def basic_word_sim(word1, word2):
    """
    Simple measure of similarity: Number of letters in common / max length
    """
    return sum([1 for c in word1 if c in word2]) / max(len(word1), len(word2))

def today(year=None):
    """this day, last year"""
    return datetime.date(int(year), _date.month, _date.day) if year else _date

def get_files(dir_name):
    """Simple directory walker"""
    return [(os.path.join('.', d), [os.path.join(d, f) for f in files]) for d, _, files in os.walk(dir_name)]

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

def xmltreefromfile(filename):
    """Internal function to read an XML file"""
    try:
        return ElementTree.parse(filename, ElementTree.XMLParser(collect_ids=False))
    except TypeError:
        return ElementTree.parse(filename, ElementTree.XMLParser())

def weekly(date=datetime.date.today()):
    """
    Weeks start are fixes at Monday for now.
    """
    return date - datetime.timedelta(days=date.weekday())

def FromString(self, string):
    """Parse a bool from a string."""
    if string.lower() in ("false", "no", "n"):
      return False

    if string.lower() in ("true", "yes", "y"):
      return True

    raise TypeValueError("%s is not recognized as a boolean value." % string)

def parse_json_date(value):
    """
    Parses an ISO8601 formatted datetime from a string value
    """
    if not value:
        return None

    return datetime.datetime.strptime(value, JSON_DATETIME_FORMAT).replace(tzinfo=pytz.UTC)

def get_bucket_page(page):
    """
    Returns all the keys in a s3 bucket paginator page.
    """
    key_list = page.get('Contents', [])
    logger.debug("Retrieving page with {} keys".format(
        len(key_list),
    ))
    return dict((k.get('Key'), k) for k in key_list)

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

def check_dependencies_remote(args):
    """
    Invoke this command on a remote Python.
    """
    cmd = [args.python, '-m', 'depends', args.requirement]
    env = dict(PYTHONPATH=os.path.dirname(__file__))
    return subprocess.check_call(cmd, env=env)

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

def angle_between_vectors(x, y):
    """ Compute the angle between vector x and y """
    dp = dot_product(x, y)
    if dp == 0:
        return 0
    xm = magnitude(x)
    ym = magnitude(y)
    return math.acos(dp / (xm*ym)) * (180. / math.pi)

def ToDatetime(self):
    """Converts Timestamp to datetime."""
    return datetime.utcfromtimestamp(
        self.seconds + self.nanos / float(_NANOS_PER_SECOND))

def ci(a, which=95, axis=None):
    """Return a percentile range from an array of values."""
    p = 50 - which / 2, 50 + which / 2
    return percentiles(a, p, axis)

def dict_to_numpy_array(d):
    """
    Convert a dict of 1d array to a numpy recarray
    """
    return fromarrays(d.values(), np.dtype([(str(k), v.dtype) for k, v in d.items()]))

def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and 1-hot labels."""
  return 100.0 - (
      100.0 *
      np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
      predictions.shape[0])

def drop_trailing_zeros_decimal(num):
    """ Drops the trailinz zeros from decimal value.
        Returns a string
    """
    out = str(num)
    return out.rstrip('0').rstrip('.') if '.' in out else out

def average_price(quantity_1, price_1, quantity_2, price_2):
    """Calculates the average price between two asset states."""
    return (quantity_1 * price_1 + quantity_2 * price_2) / \
            (quantity_1 + quantity_2)

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

def elapsed_time_from(start_time):
    """calculate time delta from latched time and current time"""
    time_then = make_time(start_time)
    time_now = datetime.utcnow().replace(microsecond=0)
    if time_then is None:
        return
    delta_t = time_now - time_then
    return delta_t

def read_string(cls, string):
        """Decodes a given bencoded string or bytestring.

        Returns decoded structure(s).

        :param str string:
        :rtype: list
        """
        if PY3 and not isinstance(string, byte_types):
            string = string.encode()

        return cls.decode(string)

def set(self, f):
        """Call a function after a delay, unless another function is set
        in the meantime."""
        self.stop()
        self._create_timer(f)
        self.start()

def __copy__(self):
        """A magic method to implement shallow copy behavior."""
        return self.__class__.load(self.dump(), context=self.context)

def eval_script(self, expr):
    """ Evaluates a piece of Javascript in the context of the current page and
    returns its value. """
    ret = self.conn.issue_command("Evaluate", expr)
    return json.loads("[%s]" % ret)[0]

def validate_string_list(lst):
    """Validate that the input is a list of strings.

    Raises ValueError if not."""
    if not isinstance(lst, list):
        raise ValueError('input %r must be a list' % lst)
    for x in lst:
        if not isinstance(x, basestring):
            raise ValueError('element %r in list must be a string' % x)

def execfile(fname, variables):
    """ This is builtin in python2, but we have to roll our own on py3. """
    with open(fname) as f:
        code = compile(f.read(), fname, 'exec')
        exec(code, variables)

def camel_to_under(name):
    """
    Converts camel-case string to lowercase string separated by underscores.

    Written by epost (http://stackoverflow.com/questions/1175208).

    :param name: String to be converted
    :return: new String with camel-case converted to lowercase, underscored
    """
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

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

def ndarr2str(arr, encoding='ascii'):
    """ This is used to ensure that the return value of arr.tostring()
    is actually a string.  This will prevent lots of if-checks in calling
    code.  As of numpy v1.6.1 (in Python 3.2.3), the tostring() function
    still returns type 'bytes', not 'str' as it advertises. """
    # be fast, don't check - just assume 'arr' is a numpy array - the tostring
    # call will fail anyway if not
    retval = arr.tostring()
    # would rather check "if isinstance(retval, bytes)", but support 2.5.
    # could rm the if PY3K check, but it makes this faster on 2.x.
    if PY3K and not isinstance(retval, str):
        return retval.decode(encoding)
    else: # is str
        return retval

def setdefault(obj, field, default):
    """Set an object's field to default if it doesn't have a value"""
    setattr(obj, field, getattr(obj, field, default))

def get_decimal_quantum(precision):
    """Return minimal quantum of a number, as defined by precision."""
    assert isinstance(precision, (int, decimal.Decimal))
    return decimal.Decimal(10) ** (-precision)

def get_var(name, factory=None):
    """Gets a global variable given its name.

    If factory is not None and the variable is not set, factory
    is a callable that will set the variable.

    If not set, returns None.
    """
    if name not in _VARS and factory is not None:
        _VARS[name] = factory()
    return _VARS.get(name)

def eval_script(self, expr):
    """ Evaluates a piece of Javascript in the context of the current page and
    returns its value. """
    ret = self.conn.issue_command("Evaluate", expr)
    return json.loads("[%s]" % ret)[0]

def uninstall(cls):
        """Remove the package manager from the system."""
        if os.path.exists(cls.home):
            shutil.rmtree(cls.home)

def unpickle_file(picklefile, **kwargs):
    """Helper function to unpickle data from `picklefile`."""
    with open(picklefile, 'rb') as f:
        return pickle.load(f, **kwargs)

def dedupe_list(seq):
    """
    Utility function to remove duplicates from a list
    :param seq: The sequence (list) to deduplicate
    :return: A list with original duplicates removed
    """
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]

def edge_index(self):
        """A map to look up the index of a edge"""
        return dict((edge, index) for index, edge in enumerate(self.edges))

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

def getTuple(self):
        """ Returns the shape of the region as (x, y, w, h) """
        return (self.x, self.y, self.w, self.h)

def safe_delete(filename):
  """Delete a file safely. If it's not present, no-op."""
  try:
    os.unlink(filename)
  except OSError as e:
    if e.errno != errno.ENOENT:
      raise

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

def pop():
        """Remove instance from instance list"""
        pid = os.getpid()
        thread = threading.current_thread()
        Wdb._instances.pop((pid, thread))

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

def add_exec_permission_to(target_file):
    """Add executable permissions to the file

    :param target_file: the target file whose permission to be changed
    """
    mode = os.stat(target_file).st_mode
    os.chmod(target_file, mode | stat.S_IXUSR)

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

def makedirs(path, mode=0o777, exist_ok=False):
    """A wrapper of os.makedirs()."""
    os.makedirs(path, mode, exist_ok)

def _platform_is_windows(platform=sys.platform):
        """Is the current OS a Windows?"""
        matched = platform in ('cygwin', 'win32', 'win64')
        if matched:
            error_msg = "Windows isn't supported yet"
            raise OSError(error_msg)
        return matched

def _snake_to_camel_case(value):
    """Convert snake case string to camel case."""
    words = value.split("_")
    return words[0] + "".join(map(str.capitalize, words[1:]))

def _platform_is_windows(platform=sys.platform):
        """Is the current OS a Windows?"""
        matched = platform in ('cygwin', 'win32', 'win64')
        if matched:
            error_msg = "Windows isn't supported yet"
            raise OSError(error_msg)
        return matched

def _push_render(self):
        """Render the plot with bokeh.io and push to notebook.
        """
        bokeh.io.push_notebook(handle=self.handle)
        self.last_update = time.time()

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

def FromString(self, string):
    """Parse a bool from a string."""
    if string.lower() in ("false", "no", "n"):
      return False

    if string.lower() in ("true", "yes", "y"):
      return True

    raise TypeValueError("%s is not recognized as a boolean value." % string)

def strToBool(val):
    """
    Helper function to turn a string representation of "true" into
    boolean True.
    """
    if isinstance(val, str):
        val = val.lower()

    return val in ['true', 'on', 'yes', True]

def stringify_dict_contents(dct):
    """Turn dict keys and values into native strings."""
    return {
        str_if_nested_or_str(k): str_if_nested_or_str(v)
        for k, v in dct.items()
    }

def list_to_csv(value):
    """
    Converts list to string with comma separated values. For string is no-op.
    """
    if isinstance(value, (list, tuple, set)):
        value = ",".join(value)
    return value

def _defaultdict(dct, fallback=_illegal_character):
    """Wraps the given dictionary such that the given fallback function will be called when a nonexistent key is
    accessed.
    """
    out = defaultdict(lambda: fallback)
    for k, v in six.iteritems(dct):
        out[k] = v
    return out

def str2bytes(x):
  """Convert input argument to bytes"""
  if type(x) is bytes:
    return x
  elif type(x) is str:
    return bytes([ ord(i) for i in x ])
  else:
    return str2bytes(str(x))

def purge_dict(idict):
    """Remove null items from a dictionary """
    odict = {}
    for key, val in idict.items():
        if is_null(val):
            continue
        odict[key] = val
    return odict

def parse_date(s):
    """Fast %Y-%m-%d parsing."""
    try:
        return datetime.date(int(s[:4]), int(s[5:7]), int(s[8:10]))
    except ValueError:  # other accepted format used in one-day data set
        return datetime.datetime.strptime(s, '%d %B %Y').date()

def get_single_value(d):
    """Get a value from a dict which contains just one item."""
    assert len(d) == 1, 'Single-item dict must have just one item, not %d.' % len(d)
    return next(six.itervalues(d))

def date_to_datetime(x):
    """Convert a date into a datetime"""
    if not isinstance(x, datetime) and isinstance(x, date):
        return datetime.combine(x, time())
    return x

def _remove_keywords(d):
    """
    copy the dict, filter_keywords

    Parameters
    ----------
    d : dict
    """
    return { k:v for k, v in iteritems(d) if k not in RESERVED }

def bitsToString(arr):
  """Returns a string representing a numpy array of 0's and 1's"""
  s = array('c','.'*len(arr))
  for i in xrange(len(arr)):
    if arr[i] == 1:
      s[i]='*'
  return s

def data(self, data):
        """Store a copy of the data."""
        self._data = {det: d.copy() for (det, d) in data.items()}

def lower_ext(abspath):
    """Convert file extension to lowercase.
    """
    fname, ext = os.path.splitext(abspath)
    return fname + ext.lower()

def stop_button_click_handler(self):
        """Method to handle what to do when the stop button is pressed"""
        self.stop_button.setDisabled(True)
        # Interrupt computations or stop debugging
        if not self.shellwidget._reading:
            self.interrupt_kernel()
        else:
            self.shellwidget.write_to_stdin('exit')

def timestamp_to_datetime(timestamp):
    """Convert an ARF timestamp to a datetime.datetime object (naive local time)"""
    from datetime import datetime, timedelta
    obj = datetime.fromtimestamp(timestamp[0])
    return obj + timedelta(microseconds=int(timestamp[1]))

def disable_stdout_buffering():
    """This turns off stdout buffering so that outputs are immediately
    materialized and log messages show up before the program exits"""
    stdout_orig = sys.stdout
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
    # NOTE(brandyn): This removes the original stdout
    return stdout_orig

def scale_image(image, new_width):
    """Resizes an image preserving the aspect ratio.
    """
    (original_width, original_height) = image.size
    aspect_ratio = original_height/float(original_width)
    new_height = int(aspect_ratio * new_width)

    # This scales it wider than tall, since characters are biased
    new_image = image.resize((new_width*2, new_height))
    return new_image

def _renamer(self, tre):
        """ renames newick from numbers to sample names"""
        ## get the tre with numbered tree tip labels
        names = tre.get_leaves()

        ## replace numbered names with snames
        for name in names:
            name.name = self.samples[int(name.name)]

        ## return with only topology and leaf labels
        return tre.write(format=9)

def with_tz(request):
    """
    Get the time with TZ enabled

    """
    
    dt = datetime.now() 
    t = Template('{% load tz %}{% localtime on %}{% get_current_timezone as TIME_ZONE %}{{ TIME_ZONE }}{% endlocaltime %}') 
    c = RequestContext(request)
    response = t.render(c)
    return HttpResponse(response)

def add_exec_permission_to(target_file):
    """Add executable permissions to the file

    :param target_file: the target file whose permission to be changed
    """
    mode = os.stat(target_file).st_mode
    os.chmod(target_file, mode | stat.S_IXUSR)

def get_distance_matrix(x):
    """Get distance matrix given a matrix. Used in testing."""
    square = nd.sum(x ** 2.0, axis=1, keepdims=True)
    distance_square = square + square.transpose() - (2.0 * nd.dot(x, x.transpose()))
    return nd.sqrt(distance_square)

def str_dict(some_dict):
    """Convert dict of ascii str/unicode to dict of str, if necessary"""
    return {str(k): str(v) for k, v in some_dict.items()}

def name2rgb(hue):
    """Originally used to calculate color based on module name.
    """
    r, g, b = colorsys.hsv_to_rgb(hue / 360.0, .8, .7)
    return tuple(int(x * 256) for x in [r, g, b])

def get_static_url():
    """Return a base static url, always ending with a /"""
    path = getattr(settings, 'STATIC_URL', None)
    if not path:
        path = getattr(settings, 'MEDIA_URL', None)
    if not path:
        path = '/'
    return path

def shape_list(l,shape,dtype):
    """ Shape a list of lists into the appropriate shape and data type """
    return np.array(l, dtype=dtype).reshape(shape)

def delete_all_from_db():
    """Clear the database.

    Used for testing and debugging.

    """
    # The models.CASCADE property is set on all ForeignKey fields, so tables can
    # be deleted in any order without breaking constraints.
    for model in django.apps.apps.get_models():
        model.objects.all().delete()

def scipy_sparse_to_spmatrix(A):
    """Efficient conversion from scipy sparse matrix to cvxopt sparse matrix"""
    coo = A.tocoo()
    SP = spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=A.shape)
    return SP

def connect():
    """Connect to FTP server, login and return an ftplib.FTP instance."""
    ftp_class = ftplib.FTP if not SSL else ftplib.FTP_TLS
    ftp = ftp_class(timeout=TIMEOUT)
    ftp.connect(HOST, PORT)
    ftp.login(USER, PASSWORD)
    if SSL:
        ftp.prot_p()  # secure data connection
    return ftp

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

def install_postgres(user=None, dbname=None, password=None):
    """Install Postgres on remote"""
    execute(pydiploy.django.install_postgres_server,
            user=user, dbname=dbname, password=password)

def multiply(self, number):
        """Return a Vector as the product of the vector and a real number."""
        return self.from_list([x * number for x in self.to_list()])

def inheritdoc(method):
    """Set __doc__ of *method* to __doc__ of *method* in its parent class.

    Since this is used on :class:`.StringMixIn`, the "parent class" used is
    ``str``. This function can be used as a decorator.
    """
    method.__doc__ = getattr(str, method.__name__).__doc__
    return method

def decode_arr(data):
    """Extract a numpy array from a base64 buffer"""
    data = data.encode('utf-8')
    return frombuffer(base64.b64decode(data), float64)

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

def round_to_int(number, precision):
    """Round a number to a precision"""
    precision = int(precision)
    rounded = (int(number) + precision / 2) // precision * precision
    return rounded

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

def set_locale(request):
    """Return locale from GET lang param or automatically."""
    return request.query.get('lang', app.ps.babel.select_locale_by_request(request))

def inline_inputs(self):
        """Inline all input latex files references by this document. The
        inlining is accomplished recursively. The document is modified
        in place.
        """
        self.text = texutils.inline(self.text,
                                    os.path.dirname(self._filepath))
        # Remove children
        self._children = {}

def setwinsize(self, rows, cols):
        """Set the terminal window size of the child tty.
        """
        self._winsize = (rows, cols)
        self.pty.set_size(cols, rows)

def downsample_with_striding(array, factor):
    """Downsample x by factor using striding.

    @return: The downsampled array, of the same type as x.
    """
    return array[tuple(np.s_[::f] for f in factor)]

def get_best_encoding(stream):
    """Returns the default stream encoding if not found."""
    rv = getattr(stream, 'encoding', None) or sys.getdefaultencoding()
    if is_ascii_encoding(rv):
        return 'utf-8'
    return rv

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

def _float_almost_equal(float1, float2, places=7):
    """Return True if two numbers are equal up to the
    specified number of "places" after the decimal point.
    """

    if round(abs(float2 - float1), places) == 0:
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

def any_contains_any(strings, candidates):
    """Whether any of the strings contains any of the candidates."""
    for string in strings:
        for c in candidates:
            if c in string:
                return True

def get_code(module):
    """
    Compile and return a Module's code object.
    """
    fp = open(module.path)
    try:
        return compile(fp.read(), str(module.name), 'exec')
    finally:
        fp.close()

def is_complex(dtype):
  """Returns whether this is a complex floating point type."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'is_complex'):
    return dtype.is_complex
  return np.issubdtype(np.dtype(dtype), np.complex)

def command(name, mode):
    """ Label a method as a command with name. """
    def decorator(fn):
        commands[name] = fn.__name__
        _Client._addMethod(fn.__name__, name, mode)
        return fn
    return decorator

def is_complex(dtype):
  """Returns whether this is a complex floating point type."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'is_complex'):
    return dtype.is_complex
  return np.issubdtype(np.dtype(dtype), np.complex)

def closing_plugin(self, cancelable=False):
        """Perform actions before parent main window is closed"""
        self.dialog_manager.close_all()
        self.shell.exit_interpreter()
        return True

def _is_initialized(self, entity):
    """Internal helper to ask if the entity has a value for this Property.

    This returns False if a value is stored but it is None.
    """
    return (not self._required or
            ((self._has_value(entity) or self._default is not None) and
             self._get_value(entity) is not None))

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

def is_punctuation(text):
    """Check if given string is a punctuation"""
    return not (text.lower() in config.AVRO_VOWELS or
                text.lower() in config.AVRO_CONSONANTS)

def clear_es():
        """Clear all indexes in the es core"""
        # TODO: should receive a catalog slug.
        ESHypermap.es.indices.delete(ESHypermap.index_name, ignore=[400, 404])
        LOGGER.debug('Elasticsearch: Index cleared')

def is_int_vector(l):
    r"""Checks if l is a numpy array of integers

    """
    if isinstance(l, np.ndarray):
        if l.ndim == 1 and (l.dtype.kind == 'i' or l.dtype.kind == 'u'):
            return True
    return False

def _not_none(items):
    """Whether the item is a placeholder or contains a placeholder."""
    if not isinstance(items, (tuple, list)):
        items = (items,)
    return all(item is not _none for item in items)

def __add__(self, other):
        """Handle the `+` operator."""
        return self._handle_type(other)(self.value + other.value)

def _is_expired_response(self, response):
        """
        Check if the response failed because of an expired access token.
        """
        if response.status_code != 401:
            return False
        challenge = response.headers.get('www-authenticate', '')
        return 'error="invalid_token"' in challenge

def get_element_with_id(self, id):
        """Return the element with the specified ID."""
        # Should we maintain a hashmap of ids to make this more efficient? Probably overkill.
        # TODO: Elements can contain nested elements (captions, footnotes, table cells, etc.)
        return next((el for el in self.elements if el.id == id), None)

def isbinary(*args):
    """Checks if value can be part of binary/bitwise operations."""
    return all(map(lambda c: isnumber(c) or isbool(c), args))

def add_parent(self, parent):
        """
        Adds self as child of parent, then adds parent.
        """
        parent.add_child(self)
        self.parent = parent
        return parent

def column_exists(cr, table, column):
    """ Check whether a certain column exists """
    cr.execute(
        'SELECT count(attname) FROM pg_attribute '
        'WHERE attrelid = '
        '( SELECT oid FROM pg_class WHERE relname = %s ) '
        'AND attname = %s',
        (table, column))
    return cr.fetchone()[0] == 1

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

def is_enum_type(type_):
    """ Checks if the given type is an enum type.

    :param type_: The type to check
    :return: True if the type is a enum type, otherwise False
    :rtype: bool
    """

    return isinstance(type_, type) and issubclass(type_, tuple(_get_types(Types.ENUM)))

def is_nullable_list(val, vtype):
    """Return True if list contains either values of type `vtype` or None."""
    return (isinstance(val, list) and
            any(isinstance(v, vtype) for v in val) and
            all((isinstance(v, vtype) or v is None) for v in val))

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

def raise_for_not_ok_status(response):
    """
    Raises a `requests.exceptions.HTTPError` if the response has a non-200
    status code.
    """
    if response.code != OK:
        raise HTTPError('Non-200 response code (%s) for url: %s' % (
            response.code, uridecode(response.request.absoluteURI)))

    return response

def write_enum(fo, datum, schema):
    """An enum is encoded by a int, representing the zero-based position of
    the symbol in the schema."""
    index = schema['symbols'].index(datum)
    write_int(fo, index)

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

def _escape(s):
    """ Helper method that escapes parameters to a SQL query. """
    e = s
    e = e.replace('\\', '\\\\')
    e = e.replace('\n', '\\n')
    e = e.replace('\r', '\\r')
    e = e.replace("'", "\\'")
    e = e.replace('"', '\\"')
    return e

def contains_all(self, array):
        """Test if `array` is an array of real numbers."""
        dtype = getattr(array, 'dtype', None)
        if dtype is None:
            dtype = np.result_type(*array)
        return is_real_dtype(dtype)

def get_user_id_from_email(self, email):
        """ Uses the get-all-user-accounts Portals API to retrieve the
        user-id by supplying an email. """
        accts = self.get_all_user_accounts()

        for acct in accts:
            if acct['email'] == email:
                return acct['id']
        return None

def is_function(self):
        """return True if callback is a vanilla plain jane function"""
        if self.is_instance() or self.is_class(): return False
        return isinstance(self.callback, (Callable, classmethod))

def _remove_dict_keys_with_value(dict_, val):
  """Removes `dict` keys which have have `self` as value."""
  return {k: v for k, v in dict_.items() if v is not val}

def is_int_type(val):
    """Return True if `val` is of integer type."""
    try:               # Python 2
        return isinstance(val, (int, long))
    except NameError:  # Python 3
        return isinstance(val, int)

def expandpath(path):
    """
    Expand a filesystem path that may or may not contain user/env vars.

    :param str path: path to expand
    :return str: expanded version of input path
    """
    return os.path.expandvars(os.path.expanduser(path)).replace("//", "/")

def is_square_matrix(mat):
    """Test if an array is a square matrix."""
    mat = np.array(mat)
    if mat.ndim != 2:
        return False
    shape = mat.shape
    return shape[0] == shape[1]

def _expand(self, str, local_vars={}):
        """Expand $vars in a string."""
        return ninja_syntax.expand(str, self.vars, local_vars)

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

def validate_string(option, value):
    """Validates that 'value' is an instance of `basestring` for Python 2
    or `str` for Python 3.
    """
    if isinstance(value, string_type):
        return value
    raise TypeError("Wrong type for %s, value must be "
                    "an instance of %s" % (option, string_type.__name__))

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

def EvalGaussianPdf(x, mu, sigma):
    """Computes the unnormalized PDF of the normal distribution.

    x: value
    mu: mean
    sigma: standard deviation
    
    returns: float probability density
    """
    return scipy.stats.norm.pdf(x, mu, sigma)

def eof(fd):
    """Determine if end-of-file is reached for file fd."""
    b = fd.read(1)
    end = len(b) == 0
    if not end:
        curpos = fd.tell()
        fd.seek(curpos - 1)
    return end

def resources(self):
        """Retrieve contents of each page of PDF"""
        return [self.pdf.getPage(i) for i in range(self.pdf.getNumPages())]

def _num_cpus_darwin():
    """Return the number of active CPUs on a Darwin system."""
    p = subprocess.Popen(['sysctl','-n','hw.ncpu'],stdout=subprocess.PIPE)
    return p.stdout.read()

def parse_domain(url):
    """ parse the domain from the url """
    domain_match = lib.DOMAIN_REGEX.match(url)
    if domain_match:
        return domain_match.group()

def isString(s):
    """Convenience method that works with all 2.x versions of Python
    to determine whether or not something is stringlike."""
    try:
        return isinstance(s, unicode) or isinstance(s, basestring)
    except NameError:
        return isinstance(s, str)

def _relpath(name):
    """
    Strip absolute components from path.
    Inspired from zipfile.write().
    """
    return os.path.normpath(os.path.splitdrive(name)[1]).lstrip(_allsep)

def contains_geometric_info(var):
    """ Check whether the passed variable is a tuple with two floats or integers """
    return isinstance(var, tuple) and len(var) == 2 and all(isinstance(val, (int, float)) for val in var)

def logv(msg, *args, **kwargs):
    """
    Print out a log message, only if verbose mode.
    """
    if settings.VERBOSE:
        log(msg, *args, **kwargs)

def _rectangular(n):
    """Checks to see if a 2D list is a valid 2D matrix"""
    for i in n:
        if len(i) != len(n[0]):
            return False
    return True

def computeFactorial(n):
    """
    computes factorial of n
    """
    sleep_walk(10)
    ret = 1
    for i in range(n):
        ret = ret * (i + 1)
    return ret

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

def data_directory():
    """Return the absolute path to the directory containing the package data."""
    package_directory = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(package_directory, "data")

def is_int_type(val):
    """Return True if `val` is of integer type."""
    try:               # Python 2
        return isinstance(val, (int, long))
    except NameError:  # Python 3
        return isinstance(val, int)

def get_randomized_guid_sample(self, item_count):
        """ Fetch a subset of randomzied GUIDs from the whitelist """
        dataset = self.get_whitelist()
        random.shuffle(dataset)
        return dataset[:item_count]

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

def filter_greys_using_image(image, target):
    """Filter out any values in target not in image

    :param image: image containing values to appear in filtered image
    :param target: the image to filter
    :rtype: 2d  :class:`numpy.ndarray` containing only value in image
        and with the same dimensions as target

    """
    maskbase = numpy.array(range(256), dtype=numpy.uint8)
    mask = numpy.where(numpy.in1d(maskbase, numpy.unique(image)), maskbase, 0)
    return mask[target]

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

def get_remote_content(filepath):
        """ A handy wrapper to get a remote file content """
        with hide('running'):
            temp = BytesIO()
            get(filepath, temp)
            content = temp.getvalue().decode('utf-8')
        return content.strip()

def numberp(v):
    """Return true iff 'v' is a number."""
    return (not(isinstance(v, bool)) and
            (isinstance(v, int) or isinstance(v, float)))

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

def __eq__(self, other):
        """Determine if two objects are equal."""
        return isinstance(other, self.__class__) \
            and self._freeze() == other._freeze()

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

def is_iter_non_string(obj):
    """test if object is a list or tuple"""
    if isinstance(obj, list) or isinstance(obj, tuple):
        return True
    return False

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

def drop_bad_characters(text):
    """Takes a text and drops all non-printable and non-ascii characters and
    also any whitespace characters that aren't space.

    :arg str text: the text to fix

    :returns: text with all bad characters dropped

    """
    # Strip all non-ascii and non-printable characters
    text = ''.join([c for c in text if c in ALLOWED_CHARS])
    return text

def is_writable_by_others(filename):
    """Check if file or directory is world writable."""
    mode = os.stat(filename)[stat.ST_MODE]
    return mode & stat.S_IWOTH

def _clear(self):
        """
        Helper that clears the composition.
        """
        draw = ImageDraw.Draw(self._background_image)
        draw.rectangle(self._device.bounding_box,
                       fill="black")
        del draw

def txt_line_iterator(path):
  """Iterate through lines of file."""
  with tf.gfile.Open(path) as f:
    for line in f:
      yield line.strip()

def Flush(self):
    """Flush all items from cache."""
    while self._age:
      node = self._age.PopLeft()
      self.KillObject(node.data)

    self._hash = dict()

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

def clear_globals_reload_modules(self):
        """Clears globals and reloads modules"""

        self.code_array.clear_globals()
        self.code_array.reload_modules()

        # Clear result cache
        self.code_array.result_cache.clear()

def rewindbody(self):
        """Rewind the file to the start of the body (if seekable)."""
        if not self.seekable:
            raise IOError, "unseekable file"
        self.fp.seek(self.startofbody)

def Flush(self):
    """Flush all items from cache."""
    while self._age:
      node = self._age.PopLeft()
      self.KillObject(node.data)

    self._hash = dict()

def get_file_name(url):
  """Returns file name of file at given url."""
  return os.path.basename(urllib.parse.urlparse(url).path) or 'unknown_name'

def close_all_but_this(self):
        """Close all files but the current one"""
        self.close_all_right()
        for i in range(0, self.get_stack_count()-1  ):
            self.close_file(0)

def _fill_array_from_list(the_list, the_array):
        """Fill an `array` from a `list`"""
        for i, val in enumerate(the_list):
            the_array[i] = val
        return the_array

def _maybe_fill(arr, fill_value=np.nan):
    """
    if we have a compatible fill_value and arr dtype, then fill
    """
    if _isna_compat(arr, fill_value):
        arr.fill(fill_value)
    return arr

def column_stack_2d(data):
    """Perform column-stacking on a list of 2d data blocks."""
    return list(list(itt.chain.from_iterable(_)) for _ in zip(*data))

def apply_filters(df, filters):
        """Basic filtering for a dataframe."""
        idx = pd.Series([True]*df.shape[0])
        for k, v in list(filters.items()):
            if k not in df.columns:
                continue
            idx &= (df[k] == v)

        return df.loc[idx]

def combinations(l):
    """Pure-Python implementation of itertools.combinations(l, 2)."""
    result = []
    for x in xrange(len(l) - 1):
        ls = l[x + 1:]
        for y in ls:
            result.append((l[x], y))
    return result

def BROADCAST_FILTER_NOT(func):
        """
        Composes the passed filters into an and-joined filter.
        """
        return lambda u, command, *args, **kwargs: not func(u, command, *args, **kwargs)

def file_matches(filename, patterns):
    """Does this filename match any of the patterns?"""
    return any(fnmatch.fnmatch(filename, pat) for pat in patterns)

def _sim_fill(r1, r2, imsize):
    """
        calculate the fill similarity over the image
    """
    bbsize = (
        (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]))
        * (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
    )
    return 1.0 - (bbsize - r1["size"] - r2["size"]) / imsize

def apply_fit(xy,coeffs):
    """ Apply the coefficients from a linear fit to
        an array of x,y positions.

        The coeffs come from the 'coeffs' member of the
        'fit_arrays()' output.
    """
    x_new = coeffs[0][2] + coeffs[0][0]*xy[:,0] + coeffs[0][1]*xy[:,1]
    y_new = coeffs[1][2] + coeffs[1][0]*xy[:,0] + coeffs[1][1]*xy[:,1]

    return x_new,y_new

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

def lint(args):
    """Run lint checks using flake8."""
    application = get_current_application()
    if not args:
        args = [application.name, 'tests']
    args = ['flake8'] + list(args)
    run.main(args, standalone_mode=False)

def _sim_fill(r1, r2, imsize):
    """
        calculate the fill similarity over the image
    """
    bbsize = (
        (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]))
        * (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
    )
    return 1.0 - (bbsize - r1["size"] - r2["size"]) / imsize

def docannotate(func):
    """Annotate a function using information from its docstring.

    The annotation actually happens at the time the function is first called
    to improve startup time.  For this function to work, the docstring must be
    formatted correctly.  You should use the typedargs pylint plugin to make
    sure there are no errors in the docstring.
    """

    func = annotated(func)
    func.metadata.load_from_doc = True

    if func.decorated:
        return func

    func.decorated = True
    return decorate(func, _check_and_execute)

def unique(transactions):
    """ Remove any duplicate entries. """
    seen = set()
    # TODO: Handle comments
    return [x for x in transactions if not (x in seen or seen.add(x))]

def initialize_api(flask_app):
    """Initialize an API."""
    if not flask_restplus:
        return

    api = flask_restplus.Api(version="1.0", title="My Example API")
    api.add_resource(HelloWorld, "/hello")

    blueprint = flask.Blueprint("api", __name__, url_prefix="/api")
    api.init_app(blueprint)
    flask_app.register_blueprint(blueprint)

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

def touch_project():
    """
    Touches the project to trigger refreshing its cauldron.json state.
    """
    r = Response()
    project = cd.project.get_internal_project()

    if project:
        project.refresh()
    else:
        r.fail(
            code='NO_PROJECT',
            message='No open project to refresh'
        )

    return r.update(
        sync_time=sync_status.get('time', 0)
    ).flask_serialize()

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

def get_code(module):
    """
    Compile and return a Module's code object.
    """
    fp = open(module.path)
    try:
        return compile(fp.read(), str(module.name), 'exec')
    finally:
        fp.close()

def staticdir():
    """Return the location of the static data directory."""
    root = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(root, "static")

def _concatenate_virtual_arrays(arrs, cols=None, scaling=None):
    """Return a virtual concatenate of several NumPy arrays."""
    return None if not len(arrs) else ConcatenatedArrays(arrs, cols,
                                                         scaling=scaling)

def init_app(self, app):
        """Initialize Flask application."""
        app.config.from_pyfile('{0}.cfg'.format(app.name), silent=True)

def _concatenate_virtual_arrays(arrs, cols=None, scaling=None):
    """Return a virtual concatenate of several NumPy arrays."""
    return None if not len(arrs) else ConcatenatedArrays(arrs, cols,
                                                         scaling=scaling)

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

def generic_add(a, b):
    """Simple function to add two numbers"""
    logger.debug('Called generic_add({}, {})'.format(a, b))
    return a + b

def parse_form(self, req, name, field):
        """Pull a form value from the request."""
        return get_value(req.body_arguments, name, field)

def reseed_random(seed):
    """Reseed factory.fuzzy's random generator."""
    r = random.Random(seed)
    random_internal_state = r.getstate()
    set_random_state(random_internal_state)

def html(header_rows):
    """
    Convert a list of tuples describing a table into a HTML string
    """
    name = 'table%d' % next(tablecounter)
    return HtmlTable([map(str, row) for row in header_rows], name).render()

def enable_proxy(self, host, port):
        """Enable a default web proxy"""

        self.proxy = [host, _number(port)]
        self.proxy_enabled = True

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

def connect(host, port, username, password):
        """Connect and login to an FTP server and return ftplib.FTP object."""
        # Instantiate ftplib client
        session = ftplib.FTP()

        # Connect to host without auth
        session.connect(host, port)

        # Authenticate connection
        session.login(username, password)
        return session

def run(context, port):
    """ Run the Webserver/SocketIO and app
    """
    global ctx
    ctx = context
    app.run(port=port)

def _loadf(ins):
    """ Loads a floating point value from a memory address.
    If 2nd arg. start with '*', it is always treated as
    an indirect value.
    """
    output = _float_oper(ins.quad[2])
    output.extend(_fpush())
    return output

def serialize(self, value):
        """Takes a datetime object and returns a string"""
        if isinstance(value, str):
            return value
        return value.strftime(DATETIME_FORMAT)

def column_stack_2d(data):
    """Perform column-stacking on a list of 2d data blocks."""
    return list(list(itt.chain.from_iterable(_)) for _ in zip(*data))

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

def raise_figure_window(f=0):
    """
    Raises the supplied figure number or figure window.
    """
    if _fun.is_a_number(f): f = _pylab.figure(f)
    f.canvas.manager.window.raise_()

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

def isoformat(dt):
    """Return an ISO-8601 formatted string from the provided datetime object"""
    if not isinstance(dt, datetime.datetime):
        raise TypeError("Must provide datetime.datetime object to isoformat")

    if dt.tzinfo is None:
        raise ValueError("naive datetime objects are not allowed beyond the library boundaries")

    return dt.isoformat().replace("+00:00", "Z")
