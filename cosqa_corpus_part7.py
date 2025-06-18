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

def cli(yamlfile, root, format):
    """ Generate CSV/TSV file from biolink model """
    print(CsvGenerator(yamlfile, format).serialize(classes=root))

def load_logged_in_user():
    """If a user id is stored in the session, load the user object from
    the database into ``g.user``."""
    user_id = session.get("user_id")
    g.user = User.query.get(user_id) if user_id is not None else None

def exists(self):
    """ Checks if the item exists. """
    try:
      return self.metadata is not None
    except datalab.utils.RequestException:
      return False
    except Exception as e:
      raise e

def update_hash(cls, filelike, digest):
    """Update the digest of a single file in a memory-efficient manner."""
    block_size = digest.block_size * 1024
    for chunk in iter(lambda: filelike.read(block_size), b''):
      digest.update(chunk)

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

def is_empty_shape(sh: ShExJ.Shape) -> bool:
        """ Determine whether sh has any value """
        return sh.closed is None and sh.expression is None and sh.extra is None and \
            sh.semActs is None

def get_default_args(func):
    """
    returns a dictionary of arg_name:default_values for the input function
    """
    args, varargs, keywords, defaults = getargspec_no_self(func)
    return dict(zip(args[-len(defaults):], defaults))

def conv_dict(self):
        """dictionary of conversion"""
        return dict(integer=self.integer, real=self.real, no_type=self.no_type)

def _prt_line_detail(self, prt, line, lnum=""):
        """Print each field and its value."""
        data = zip(self.flds, line.split('\t'))
        txt = ["{:2}) {:13} {}".format(i, hdr, val) for i, (hdr, val) in enumerate(data)]
        prt.write("{LNUM}\n{TXT}\n".format(LNUM=lnum, TXT='\n'.join(txt)))

def _selectItem(self, index):
        """Select item in the list
        """
        self._selectedIndex = index
        self.setCurrentIndex(self.model().createIndex(index, 0))

def generate_split_tsv_lines(fn, header):
    """Returns dicts with header-keys and psm statistic values"""
    for line in generate_tsv_psms_line(fn):
        yield {x: y for (x, y) in zip(header, line.strip().split('\t'))}

def __validate_email(self, email):
        """Checks if a string looks like an email address"""

        e = re.match(self.EMAIL_ADDRESS_REGEX, email, re.UNICODE)
        if e:
            return email
        else:
            error = "Invalid email address: " + str(email)
            msg = self.GRIMOIRELAB_INVALID_FORMAT % {'error': error}
            raise InvalidFormatError(cause=msg)

def is_closing(self) -> bool:
        """Return ``True`` if this connection is closing.

        The connection is considered closing if either side has
        initiated its closing handshake or if the stream has been
        shut down uncleanly.
        """
        return self.stream.closed() or self.client_terminated or self.server_terminated

def test(*args):
    """
    Run unit tests.
    """
    subprocess.call(["py.test-2.7"] + list(args))
    subprocess.call(["py.test-3.4"] + list(args))

def set_global(node: Node, key: str, value: Any):
    """Adds passed value to node's globals"""
    node.node_globals[key] = value

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

def is_rfc2822(instance: str):
    """Validates RFC2822 format"""
    if not isinstance(instance, str):
        return True
    return email.utils.parsedate(instance) is not None

def warn(self, text):
		""" Ajout d'un message de log de type WARN """
		self.logger.warn("{}{}".format(self.message_prefix, text))

def get_period_last_3_months() -> str:
    """ Returns the last week as a period string """
    today = Datum()
    today.today()

    # start_date = today - timedelta(weeks=13)
    start_date = today.clone()
    start_date.subtract_months(3)

    period = get_period(start_date.date, today.date)
    return period

def format_exp_floats(decimals):
    """
    sometimes the exp. column can be too large
    """
    threshold = 10 ** 5
    return (
        lambda n: "{:.{prec}e}".format(n, prec=decimals) if n > threshold else "{:4.{prec}f}".format(n, prec=decimals)
    )

def download_json(local_filename, url, clobber=False):
    """Download the given JSON file, and pretty-print before we output it."""
    with open(local_filename, 'w') as json_file:
        json_file.write(json.dumps(requests.get(url).json(), sort_keys=True, indent=2, separators=(',', ': ')))

def image_load_time(self):
        """
        Returns aggregate image load time for all pages.
        """
        load_times = self.get_load_times('image')
        return round(mean(load_times), self.decimal_precision)

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

def split_into_words(s):
  """Split a sentence into list of words."""
  s = re.sub(r"\W+", " ", s)
  s = re.sub(r"[_0-9]+", " ", s)
  return s.split()

def readwav(filename):
    """Read a WAV file and returns the data and sample rate

    ::

        from spectrum.io import readwav
        readwav()

    """
    from scipy.io.wavfile import read as readwav
    samplerate, signal = readwav(filename)
    return signal, samplerate

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

def toarray(self):
        """Returns the data as numpy.array from each partition."""
        rdd = self._rdd.map(lambda x: x.toarray())
        return np.concatenate(rdd.collect())

def end_table_header(self):
        r"""End the table header which will appear on every page."""

        if self.header:
            msg = "Table already has a header"
            raise TableError(msg)

        self.header = True

        self.append(Command(r'endhead'))

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

def get_local_image(self, src):
        """\
        returns the bytes of the image file on disk
        """
        return ImageUtils.store_image(self.fetcher, self.article.link_hash, src, self.config)

def __del__(self):
    """Deletes the database file."""
    if self._delete_file:
      try:
        os.remove(self.name)
      except (OSError, IOError):
        pass

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

def _get_sql(filename):
    """Returns the contents of the sql file from the given ``filename``."""
    with open(os.path.join(SQL_DIR, filename), 'r') as f:
        return f.read()

def assert_list(self, putative_list, expected_type=string_types, key_arg=None):
    """
    :API: public
    """
    return assert_list(putative_list, expected_type, key_arg=key_arg,
                       raise_type=lambda msg: TargetDefinitionException(self, msg))

def access_ok(self, access):
        """ Check if there is enough permissions for access """
        for c in access:
            if c not in self.perms:
                return False
        return True

async def _thread_coro(self, *args):
        """ Coroutine called by MapAsync. It's wrapping the call of
        run_in_executor to run the synchronous function as thread """
        return await self._loop.run_in_executor(
            self._executor, self._function, *args)

def merge(self, obj):
        """This function merge another object's values with this instance

        :param obj: An object to be merged with into this layer
        :type obj: object
        """
        for attribute in dir(obj):
            if '__' in attribute:
                continue
            setattr(self, attribute, getattr(obj, attribute))

def get_uniques(l):
    """ Returns a list with no repeated elements.
    """
    result = []

    for i in l:
        if i not in result:
            result.append(i)

    return result

def downsample_with_striding(array, factor):
    """Downsample x by factor using striding.

    @return: The downsampled array, of the same type as x.
    """
    return array[tuple(np.s_[::f] for f in factor)]

def write_only_property(f):
    """
    @write_only_property decorator. Creates a property (descriptor attribute)
    that accepts assignment, but not getattr (use in an expression).
    """
    docstring = f.__doc__

    return property(fset=f, doc=docstring)

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

def random_color(_min=MIN_COLOR, _max=MAX_COLOR):
    """Returns a random color between min and max."""
    return color(random.randint(_min, _max))

def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

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

def instance_contains(container, item):
    """Search into instance attributes, properties and return values of no-args methods."""
    return item in (member for _, member in inspect.getmembers(container))

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

def RMS_energy(frames):
    """Computes the RMS energy of frames"""
    f = frames.flatten()
    return N.sqrt(N.mean(f * f))

def __repr__(self):
        """Return string representation of object."""
        return str(self.__class__) + '(' + ', '.join([list.__repr__(d) for d in self.data]) + ')'

def dtypes(self):
        """Returns all column names and their data types as a list.

        >>> df.dtypes
        [('age', 'int'), ('name', 'string')]
        """
        return [(str(f.name), f.dataType.simpleString()) for f in self.schema.fields]

def tick(self):
        """Add one tick to progress bar"""
        self.current += 1
        if self.current == self.factor:
            sys.stdout.write('+')
            sys.stdout.flush()
            self.current = 0

def serialize_yaml_tofile(filename, resource):
    """
    Serializes a K8S resource to YAML-formatted file.
    """
    stream = file(filename, "w")
    yaml.dump(resource, stream, default_flow_style=False)

def conv2d(x_input, w_matrix):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x_input, w_matrix, strides=[1, 1, 1, 1], padding='SAME')

def checkbox_uncheck(self, force_check=False):
        """
        Wrapper to uncheck a checkbox
        """
        if self.get_attribute('checked'):
            self.click(force_click=force_check)

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

def confusion_matrix(self):
        """Confusion matrix plot
        """
        return plot.confusion_matrix(self.y_true, self.y_pred,
                                     self.target_names, ax=_gen_ax())

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

def path_for_import(name):
    """
    Returns the directory path for the given package or module.
    """
    return os.path.dirname(os.path.abspath(import_module(name).__file__))

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

def _gaps_from(intervals):
    """
    From a list of intervals extract
    a list of sorted gaps in the form of [(g,i)]
    where g is the size of the ith gap.
    """
    sliding_window = zip(intervals, intervals[1:])
    gaps = [b[0] - a[1] for a, b in sliding_window]
    return gaps

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

def _removeLru(self):
        """
        Remove the least recently used file handle from the cache.
        The pop method removes an element from the right of the deque.
        Returns the name of the file that has been removed.
        """
        (dataFile, handle) = self._cache.pop()
        handle.close()
        return dataFile

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

def _get_item_position(self, idx):
        """Return a tuple of (start, end) indices of an item from its index."""
        start = 0 if idx == 0 else self._index[idx - 1] + 1
        end = self._index[idx]
        return start, end

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

def _get_xy_scaling_parameters(self):
        """Get the X/Y coordinate limits for the full resulting image"""
        return self.mx, self.bx, self.my, self.by

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

def getYamlDocument(filePath):
    """
    Return a yaml file's contents as a dictionary
    """
    with open(filePath) as stream:
        doc = yaml.load(stream)
        return doc

def be_array_from_bytes(fmt, data):
    """
    Reads an array from bytestring with big-endian data.
    """
    arr = array.array(str(fmt), data)
    return fix_byteorder(arr)

def delete_environment(self, environment_name):
        """
        Deletes an environment
        """
        self.ebs.terminate_environment(environment_name=environment_name, terminate_resources=True)

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

def timestamp_to_datetime(cls, time_stamp, localized=True):
        """ Converts a UTC timestamp to a datetime.datetime."""
        ret = datetime.datetime.utcfromtimestamp(time_stamp)
        if localized:
            ret = localize(ret, pytz.utc)
        return ret

def _validate_key(self, key):
        """Returns a boolean indicating if the attribute name is valid or not"""
        return not any([key.startswith(i) for i in self.EXCEPTIONS])

def get_system_flags() -> FrozenSet[Flag]:
    """Return the set of implemented system flags."""
    return frozenset({Seen, Recent, Deleted, Flagged, Answered, Draft})

def get_user_by_id(self, id):
        """Retrieve a User object by ID."""
        return self.db_adapter.get_object(self.UserClass, id=id)

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

def contains_geometric_info(var):
    """ Check whether the passed variable is a tuple with two floats or integers """
    return isinstance(var, tuple) and len(var) == 2 and all(isinstance(val, (int, float)) for val in var)

def object_to_json(obj):
    """Convert object that cannot be natively serialized by python to JSON representation."""
    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
        return obj.isoformat()
    return str(obj)

def force_stop(self):
        """
        Forcibly terminates all Celery processes.
        """
        r = self.local_renderer
        with self.settings(warn_only=True):
            r.sudo('pkill -9 -f celery')
        r.sudo('rm -f /tmp/celery*.pid')

def _update_staticmethod(self, oldsm, newsm):
        """Update a staticmethod update."""
        # While we can't modify the staticmethod object itself (it has no
        # mutable attributes), we *can* extract the underlying function
        # (by calling __get__(), which returns it) and update it in-place.
        # We don't have the class available to pass to __get__() but any
        # object except None will do.
        self._update(None, None, oldsm.__get__(0), newsm.__get__(0))

def duplicated_rows(df, col_name):
    """ Return a DataFrame with the duplicated values of the column `col_name`
    in `df`."""
    _check_cols(df, [col_name])

    dups = df[pd.notnull(df[col_name]) & df.duplicated(subset=[col_name])]
    return dups

def iterparse(source, events=('end',), remove_comments=True, **kw):
    """Thin wrapper around ElementTree.iterparse"""
    return ElementTree.iterparse(source, events, SourceLineParser(), **kw)

def set_limits(self, min_=None, max_=None):
        """
        Sets limits for this config value

        If the resulting integer is outside those limits, an exception will be raised

        :param min_: minima
        :param max_: maxima
        """
        self._min, self._max = min_, max_

def l2_norm(params):
    """Computes l2 norm of params by flattening them into a vector."""
    flattened, _ = flatten(params)
    return np.dot(flattened, flattened)

def isstring(value):
    """Report whether the given value is a byte or unicode string."""
    classes = (str, bytes) if pyutils.PY3 else basestring  # noqa: F821
    return isinstance(value, classes)

def setup():
    """Setup pins"""
    print("Simple drive")
    board.set_pin_mode(L_CTRL_1, Constants.OUTPUT)
    board.set_pin_mode(L_CTRL_2, Constants.OUTPUT)
    board.set_pin_mode(PWM_L, Constants.PWM)
    board.set_pin_mode(R_CTRL_1, Constants.OUTPUT)
    board.set_pin_mode(R_CTRL_2, Constants.OUTPUT)
    board.set_pin_mode(PWM_R, Constants.PWM)

def _normalize(image):
  """Normalize the image to zero mean and unit variance."""
  offset = tf.constant(MEAN_RGB, shape=[1, 1, 3])
  image -= offset

  scale = tf.constant(STDDEV_RGB, shape=[1, 1, 3])
  image /= scale
  return image

def _dotify(cls, data):
    """Add dots."""
    return ''.join(char if char in cls.PRINTABLE_DATA else '.' for char in data)

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

def _findNearest(arr, value):
    """ Finds the value in arr that value is closest to
    """
    arr = np.array(arr)
    # find nearest value in array
    idx = (abs(arr-value)).argmin()
    return arr[idx]

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

def parse_float(float_str):
    """Parse a string of the form 305.48b into a Python float.
       The terminal letter, if present, indicates e.g. billions."""
    factor = __get_factor(float_str)
    if factor != 1:
        float_str = float_str[:-1]

    try:
        return float(float_str.replace(',', '')) * factor
    except ValueError:
        return None

def _add_hash(source):
    """Add a leading hash '#' at the beginning of every line in the source."""
    source = '\n'.join('# ' + line.rstrip()
                       for line in source.splitlines())
    return source

def delete_index(index):
    """Delete index entirely (removes all documents and mapping)."""
    logger.info("Deleting search index: '%s'", index)
    client = get_client()
    return client.indices.delete(index=index)

def _IsDirectory(parent, item):
  """Helper that returns if parent/item is a directory."""
  return tf.io.gfile.isdir(os.path.join(parent, item))

def check_if_numbers_are_consecutive(list_):
    """
    Returns True if numbers in the list are consecutive

    :param list_: list of integers
    :return: Boolean
    """
    return all((True if second - first == 1 else False
                for first, second in zip(list_[:-1], list_[1:])))

def parse_prefix(identifier):
    """
    Parse identifier such as a|c|le|d|li|re|or|AT4G00480.1 and return
    tuple of prefix string (separated at '|') and suffix (AGI identifier)
    """
    pf, id = (), identifier
    if "|" in identifier:
        pf, id = tuple(identifier.split('|')[:-1]), identifier.split('|')[-1]

    return pf, id

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

def set_range(self, min_val, max_val):
        """Set the range of the colormap to [*min_val*, *max_val*]
        """
        if min_val > max_val:
            max_val, min_val = min_val, max_val
        self.values = (((self.values * 1.0 - self.values.min()) /
                        (self.values.max() - self.values.min()))
                       * (max_val - min_val) + min_val)

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

def _spawn(self, func, *args, **kwargs):
        """Spawn a handler function.

        Spawns the supplied ``func`` with ``*args`` and ``**kwargs``
        as a gevent greenlet.

        :param func: A callable to call.
        :param args: Arguments to ``func``.
        :param kwargs: Keyword arguments to ``func``.
        """
        gevent.spawn(func, *args, **kwargs)

def reraise(error):
    """Re-raises the error that was processed by prepare_for_reraise earlier."""
    if hasattr(error, "_type_"):
        six.reraise(type(error), error, error._traceback)
    raise error

def _getSuperFunc(self, s, func):
        """Return the the super function."""

        return getattr(super(self.cls(), s), func.__name__)

def copy(self):
        """Return a shallow copy of the sorted dictionary."""
        return self.__class__(self._key, self._load, self._iteritems())

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

def _writable_dir(path):
    """Whether `path` is a directory, to which the user has write access."""
    return os.path.isdir(path) and os.access(path, os.W_OK)

def mark(self, n=1):
        """Mark the occurrence of a given number of events."""
        self.tick_if_necessary()
        self.count += n
        self.m1_rate.update(n)
        self.m5_rate.update(n)
        self.m15_rate.update(n)

def clear_es():
        """Clear all indexes in the es core"""
        # TODO: should receive a catalog slug.
        ESHypermap.es.indices.delete(ESHypermap.index_name, ignore=[400, 404])
        LOGGER.debug('Elasticsearch: Index cleared')

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

def zero_pad(m, n=1):
    """Pad a matrix with zeros, on all sides."""
    return np.pad(m, (n, n), mode='constant', constant_values=[0])

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

def metres2latlon(mx, my, origin_shift= 2 * pi * 6378137 / 2.0):
    """Converts XY point from Spherical Mercator EPSG:900913 to lat/lon in
    WGS84 Datum"""
    lon = (mx / origin_shift) * 180.0
    lat = (my / origin_shift) * 180.0

    lat = 180 / pi * (2 * atan( exp( lat * pi / 180.0)) - pi / 2.0)
    return lat, lon

def eval(e, amplitude, e_0, alpha, beta):
        """One dimenional log parabola model function"""

        ee = e / e_0
        eeponent = -alpha - beta * np.log(ee)
        return amplitude * ee ** eeponent

def join_states(*states: State) -> State:
    """Join two state vectors into a larger qubit state"""
    vectors = [ket.vec for ket in states]
    vec = reduce(outer_product, vectors)
    return State(vec.tensor, vec.qubits)

def compare(a, b):
    """
     Compare items in 2 arrays. Returns sum(abs(a(i)-b(i)))
    """
    s=0
    for i in range(len(a)):
        s=s+abs(a[i]-b[i])
    return s

def safe_call(cls, method, *args):
        """ Call a remote api method but don't raise if an error occurred."""
        return cls.call(method, *args, safe=True)

def equal(x, y):
    """
    Return True if x == y and False otherwise.

    This function returns False whenever x and/or y is a NaN.

    """
    x = BigFloat._implicit_convert(x)
    y = BigFloat._implicit_convert(y)
    return mpfr.mpfr_equal_p(x, y)

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

def _ParseYamlFromFile(filedesc):
  """Parses given YAML file."""
  content = filedesc.read()
  return yaml.Parse(content) or collections.OrderedDict()

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

def downgrade(directory, sql, tag, x_arg, revision):
    """Revert to a previous version"""
    _downgrade(directory, revision, sql, tag, x_arg)

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

def _mean_absolute_error(y, y_pred, w):
    """Calculate the mean absolute error."""
    return np.average(np.abs(y_pred - y), weights=w)

def angle_between_vectors(x, y):
    """ Compute the angle between vector x and y """
    dp = dot_product(x, y)
    if dp == 0:
        return 0
    xm = magnitude(x)
    ym = magnitude(y)
    return math.acos(dp / (xm*ym)) * (180. / math.pi)

def new_random_state(seed=None, fully_random=False):
    """
    Returns a new random state.

    Parameters
    ----------
    seed : None or int, optional
        Optional seed value to use.
        The same datatypes are allowed as for ``numpy.random.RandomState(seed)``.

    fully_random : bool, optional
        Whether to use numpy's random initialization for the
        RandomState (used if set to True). If False, a seed is sampled from
        the global random state, which is a bit faster and hence the default.

    Returns
    -------
    numpy.random.RandomState
        The new random state.

    """
    if seed is None:
        if not fully_random:
            # sample manually a seed instead of just RandomState(),
            # because the latter one
            # is way slower.
            seed = CURRENT_RANDOM_STATE.randint(SEED_MIN_VALUE, SEED_MAX_VALUE, 1)[0]
    return np.random.RandomState(seed)

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

def get_column_keys_and_names(table):
    """
    Return a generator of tuples k, c such that k is the name of the python attribute for
    the column and c is the name of the column in the sql table.
    """
    ins = inspect(table)
    return ((k, c.name) for k, c in ins.mapper.c.items())

def escape_link(url):
    """Remove dangerous URL schemes like javascript: and escape afterwards."""
    lower_url = url.lower().strip('\x00\x1a \n\r\t')
    for scheme in _scheme_blacklist:
        if lower_url.startswith(scheme):
            return ''
    return escape(url, quote=True, smart_amp=False)

def escapePathForShell(path):
		"""
		Escapes a filesystem path for use as a command-line argument
		"""
		if platform.system() == 'Windows':
			return '"{}"'.format(path.replace('"', '""'))
		else:
			return shellescape.quote(path)

def _zerosamestates(self, A):
        """
        zeros out states that should be identical

        REQUIRED ARGUMENTS

        A: the matrix whose entries are to be zeroed.

        """

        for pair in self.samestates:
            A[pair[0], pair[1]] = 0
            A[pair[1], pair[0]] = 0

def get_parent_folder_name(file_path):
    """Finds parent folder of file

    :param file_path: path
    :return: Name of folder container
    """
    return os.path.split(os.path.split(os.path.abspath(file_path))[0])[-1]

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

def _get_loggers():
    """Return list of Logger classes."""
    from .. import loader
    modules = loader.get_package_modules('logger')
    return list(loader.get_plugins(modules, [_Logger]))

def _adjust_offset(self, real_wave_mfcc, algo_parameters):
        """
        OFFSET
        """
        self.log(u"Called _adjust_offset")
        self._apply_offset(offset=algo_parameters[0])

def load_object_by_name(object_name):
    """Load an object from a module by name"""
    mod_name, attr = object_name.rsplit('.', 1)
    mod = import_module(mod_name)
    return getattr(mod, attr)

def ensure_list(iterable: Iterable[A]) -> List[A]:
    """
    An Iterable may be a list or a generator.
    This ensures we get a list without making an unnecessary copy.
    """
    if isinstance(iterable, list):
        return iterable
    else:
        return list(iterable)

def upload_file(token, channel_name, file_name):
    """ upload file to a channel """

    slack = Slacker(token)

    slack.files.upload(file_name, channels=channel_name)

def incidence(boundary):
    """
    given an Nxm matrix containing boundary info between simplices,
    compute indidence info matrix
    not very reusable; should probably not be in this lib
    """
    return GroupBy(boundary).split(np.arange(boundary.size) // boundary.shape[1])

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

def is_builtin_type(tp):
    """Checks if the given type is a builtin one.
    """
    return hasattr(__builtins__, tp.__name__) and tp is getattr(__builtins__, tp.__name__)

def is_sequence(obj):
    """Check if `obj` is a sequence, but not a string or bytes."""
    return isinstance(obj, Sequence) and not (
        isinstance(obj, str) or BinaryClass.is_valid_type(obj))

def visit_BoolOp(self, node):
        """ Return type may come from any boolop operand. """
        return sum((self.visit(value) for value in node.values), [])

def substitute(dict_, source):
    """ Perform re.sub with the patterns in the given dict
    Args:
      dict_: {pattern: repl}
      source: str
    """
    d_esc = (re.escape(k) for k in dict_.keys())
    pattern = re.compile('|'.join(d_esc))
    return pattern.sub(lambda x: dict_[x.group()], source)

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

def get_language(self):
        """
        Get the language parameter from the current request.
        """
        return get_language_parameter(self.request, self.query_language_key, default=self.get_default_language(object=object))

def generate_random_id(size=6, chars=string.ascii_uppercase + string.digits):
    """Generate random id numbers."""
    return "".join(random.choice(chars) for x in range(size))

def retry_until_not_none_or_limit_reached(method, limit, sleep_s=1,
                                          catch_exceptions=()):
  """Executes a method until the retry limit is hit or not None is returned."""
  return retry_until_valid_or_limit_reached(
      method, limit, lambda x: x is not None, sleep_s, catch_exceptions)

def safe_dump_all(documents, stream=None, **kwds):
    """
    Serialize a sequence of Python objects into a YAML stream.
    Produce only basic YAML tags.
    If stream is None, return the produced string instead.
    """
    return dump_all(documents, stream, Dumper=SafeDumper, **kwds)

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

def return_future(fn):
    """Decorator that turns a synchronous function into one returning a future.

    This should only be applied to non-blocking functions. Will do set_result()
    with the return value, or set_exc_info() if an exception is raised.

    """
    @wraps(fn)
    def decorated(*args, **kwargs):
        return gen.maybe_future(fn(*args, **kwargs))

    return decorated

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

def Slice(a, begin, size):
    """
    Slicing op.
    """
    return np.copy(a)[[slice(*tpl) for tpl in zip(begin, begin+size)]],

def isPackage(file_path):
    """
    Determine whether or not a given path is a (sub)package or not.
    """
    return (os.path.isdir(file_path) and
            os.path.isfile(os.path.join(file_path, '__init__.py')))

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

def decode_arr(data):
    """Extract a numpy array from a base64 buffer"""
    data = data.encode('utf-8')
    return frombuffer(base64.b64decode(data), float64)

def url_encode(url):
    """
    Convert special characters using %xx escape.

    :param url: str
    :return: str - encoded url
    """
    if isinstance(url, text_type):
        url = url.encode('utf8')
    return quote(url, ':/%?&=')

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

def createdb():
    """Create database tables from sqlalchemy models"""
    manager.db.engine.echo = True
    manager.db.create_all()
    set_alembic_revision()

def list_i2str(ilist):
    """
    Convert an integer list into a string list.
    """
    slist = []
    for el in ilist:
        slist.append(str(el))
    return slist

def dictlist_wipe_key(dict_list: Iterable[Dict], key: str) -> None:
    """
    Process an iterable of dictionaries. For each dictionary ``d``, delete
    ``d[key]`` if it exists.
    """
    for d in dict_list:
        d.pop(key, None)

def root_parent(self, category=None):
        """ Returns the topmost parent of the current category. """
        return next(filter(lambda c: c.is_root, self.hierarchy()))

def clear(self):
        """ clear plot """
        self.axes.cla()
        self.conf.ntrace = 0
        self.conf.xlabel = ''
        self.conf.ylabel = ''
        self.conf.title  = ''

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

def get_last(self, table=None):
        """Just the last entry."""
        if table is None: table = self.main_table
        query = 'SELECT * FROM "%s" ORDER BY ROWID DESC LIMIT 1;' % table
        return self.own_cursor.execute(query).fetchone()

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

def toBase64(s):
    """Represent string / bytes s as base64, omitting newlines"""
    if isinstance(s, str):
        s = s.encode("utf-8")
    return binascii.b2a_base64(s)[:-1]

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

def QA_util_datetime_to_strdate(dt):
    """
    :param dt:  pythone datetime.datetime
    :return:  1999-02-01 string type
    """
    strdate = "%04d-%02d-%02d" % (dt.year, dt.month, dt.day)
    return strdate

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

def dict_pick(dictionary, allowed_keys):
    """
    Return a dictionary only with keys found in `allowed_keys`
    """
    return {key: value for key, value in viewitems(dictionary) if key in allowed_keys}

def expireat(self, key, when):
        """Emulate expireat"""
        expire_time = datetime.fromtimestamp(when)
        key = self._encode(key)
        if key in self.redis:
            self.timeouts[key] = expire_time
            return True
        return False

def cfloat32_array_to_numpy(cptr, length):
    """Convert a ctypes float pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_float)):
        return np.fromiter(cptr, dtype=np.float32, count=length)
    else:
        raise RuntimeError('Expected float pointer')

def save(self):
        """save the current session
        override, if session was saved earlier"""
        if self.path:
            self._saveState(self.path)
        else:
            self.saveAs()

def excepthook(self, etype, value, tb):
      """One more defense for GUI apps that call sys.excepthook.

      GUI frameworks like wxPython trap exceptions and call
      sys.excepthook themselves.  I guess this is a feature that
      enables them to keep running after exceptions that would
      otherwise kill their mainloop. This is a bother for IPython
      which excepts to catch all of the program exceptions with a try:
      except: statement.

      Normally, IPython sets sys.excepthook to a CrashHandler instance, so if
      any app directly invokes sys.excepthook, it will look to the user like
      IPython crashed.  In order to work around this, we can disable the
      CrashHandler and replace it with this excepthook instead, which prints a
      regular traceback using our InteractiveTB.  In this fashion, apps which
      call sys.excepthook will generate a regular-looking exception from
      IPython, and the CrashHandler will only be triggered by real IPython
      crashes.

      This hook should be used sparingly, only in places which are not likely
      to be true IPython errors.
      """
      self.showtraceback((etype,value,tb),tb_offset=0)

def col_frequencies(col, weights=None, gap_chars='-.'):
    """Frequencies of each residue type (totaling 1.0) in a single column."""
    counts = col_counts(col, weights, gap_chars)
    # Reduce to frequencies
    scale = 1.0 / sum(counts.values())
    return dict((aa, cnt * scale) for aa, cnt in counts.iteritems())

def _port_not_in_use():
    """Use the port 0 trick to find a port not in use."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = 0
    s.bind(('', port))
    _, port = s.getsockname()
    return port

def cleanup(self):
        """Clean up any temporary files."""
        for file in glob.glob(self.basename + '*'):
            os.unlink(file)

def makeAnimation(self):
        """Use pymovie to render (visual+audio)+text overlays.
        """
        aclip=mpy.AudioFileClip("sound.wav")
        self.iS=self.iS.set_audio(aclip)
        self.iS.write_videofile("mixedVideo.webm",15,audio=True)
        print("wrote "+"mixedVideo.webm")

def remove_ext(fname):
    """Removes the extension from a filename
    """
    bn = os.path.basename(fname)
    return os.path.splitext(bn)[0]

def _interface_exists(self, interface):
        """Check whether interface exists."""
        ios_cfg = self._get_running_config()
        parse = HTParser(ios_cfg)
        itfcs_raw = parse.find_lines("^interface " + interface)
        return len(itfcs_raw) > 0

def pause(msg="Press Enter to Continue..."):
    """press to continue"""
    print('\n' + Fore.YELLOW + msg + Fore.RESET, end='')
    input()

def append_num_column(self, text: str, index: int):
        """ Add value to the output row, width based on index """
        width = self.columns[index]["width"]
        return f"{text:>{width}}"

def find(self, name):
        """Return the index of the toc entry with name NAME.

           Return -1 for failure."""
        for i, nm in enumerate(self.data):
            if nm[-1] == name:
                return i
        return -1

def get_java_path():
  """Get the path of java executable"""
  java_home = os.environ.get("JAVA_HOME")
  return os.path.join(java_home, BIN_DIR, "java")

def update_context(self, ctx):
        """ updates the query context with this clauses values """
        assert isinstance(ctx, dict)
        ctx[str(self.context_id)] = self.value

def add_matplotlib_cmap(cm, name=None):
    """Add a matplotlib colormap."""
    global cmaps
    cmap = matplotlib_to_ginga_cmap(cm, name=name)
    cmaps[cmap.name] = cmap

def __setitem__(self, *args, **kwargs):
        """ Cut if needed. """
        super(History, self).__setitem__(*args, **kwargs)
        if len(self) > self.size:
            self.popitem(False)

def lines(input):
    """Remove comments and empty lines"""
    for raw_line in input:
        line = raw_line.strip()
        if line and not line.startswith('#'):
            yield strip_comments(line)

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

def example_write_file_to_disk_if_changed():
    """ Try to remove all comments from a file, and save it if changes were made. """
    my_file = FileAsObj('/tmp/example_file.txt')
    my_file.rm(my_file.egrep('^#'))
    if my_file.changed:
        my_file.save()

def plot_target(target, ax):
    """Ajoute la target au plot"""
    ax.scatter(target[0], target[1], target[2], c="red", s=80)

def count(self):
        """
        Returns the number of widgets currently displayed (takes child splits
        into account).
        """
        c = self.main_tab_widget.count()
        for child in self.child_splitters:
            c += child.count()
        return c

def wait_until_exit(self):
        """ Wait until thread exit

            Used for testing purpose only
        """

        if self._timeout is None:
            raise Exception("Thread will never exit. Use stop or specify timeout when starting it!")

        self._thread.join()
        self.stop()

def calculate_size(name, max_size):
    """ Calculates the request payload size"""
    data_size = 0
    data_size += calculate_size_str(name)
    data_size += INT_SIZE_IN_BYTES
    return data_size

def sql(self, sql: str, *qmark_params, **named_params):
        """
        :deprecated: use self.statement to execute properly-formatted sql statements
        """
        statement = SingleSqlStatement(sql)
        return self.statement(statement).execute(*qmark_params, **named_params)

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

def _transform_col(self, x, i):
        """Encode one categorical column into labels.

        Args:
            x (pandas.Series): a categorical column to encode
            i (int): column index

        Returns:
            x (pandas.Series): a column with labels.
        """
        return x.fillna(NAN_INT).map(self.label_encoders[i]).fillna(0)

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

def getfirstline(file, default):
    """
    Returns the first line of a file.
    """
    with open(file, 'rb') as fh:
        content = fh.readlines()
        if len(content) == 1:
            return content[0].decode('utf-8').strip('\n')

    return default

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

def get_language(query: str) -> str:
    """Tries to work out the highlight.js language of a given file name or
    shebang. Returns an empty string if none match.
    """
    query = query.lower()
    for language in LANGUAGES:
        if query.endswith(language):
            return language
    return ''

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

def create_object(cls, members):
    """Promise an object of class `cls` with content `members`."""
    obj = cls.__new__(cls)
    obj.__dict__ = members
    return obj

def pprint(self, ind):
        """pretty prints the tree with indentation"""
        pp = pprint.PrettyPrinter(indent=ind)
        pp.pprint(self.tree)

def _message_to_string(message, data=None):
    """ Gives a string representation of a PB2 message. """
    if data is None:
        data = _json_from_message(message)

    return "Message {} from {} to {}: {}".format(
        message.namespace, message.source_id, message.destination_id, data)

def on_press_key(key, callback, suppress=False):
    """
    Invokes `callback` for KEY_DOWN event related to the given key. For details see `hook`.
    """
    return hook_key(key, lambda e: e.event_type == KEY_UP or callback(e), suppress=suppress)

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

def relpath(path):
    """Path helper, gives you a path relative to this file"""
    return os.path.normpath(
        os.path.join(os.path.abspath(os.path.dirname(__file__)), path)
    )

def _open_url(url):
    """Open a HTTP connection to the URL and return a file-like object."""
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise IOError("Unable to download {}, HTTP {}".format(url, response.status_code))
    return response

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

def get_data(self):
        """
        Fetch the data field if it does not exist.
        """
        try:
            return DocumentDataDict(self.__dict__['data'])
        except KeyError:
            self._lazy_load()
            return DocumentDataDict(self.__dict__['data'])

def open(name=None, fileobj=None, closefd=True):
    """
    Use all decompressor possible to make the stream
    """
    return Guesser().open(name=name, fileobj=fileobj, closefd=closefd)

def _strip_top_comments(lines: Sequence[str], line_separator: str) -> str:
        """Strips # comments that exist at the top of the given lines"""
        lines = copy.copy(lines)
        while lines and lines[0].startswith("#"):
            lines = lines[1:]
        return line_separator.join(lines)

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

def loss(loss_value):
  """Calculates aggregated mean loss."""
  total_loss = tf.Variable(0.0, False)
  loss_count = tf.Variable(0, False)
  total_loss_update = tf.assign_add(total_loss, loss_value)
  loss_count_update = tf.assign_add(loss_count, 1)
  loss_op = total_loss / tf.cast(loss_count, tf.float32)
  return [total_loss_update, loss_count_update], loss_op

def _pick_attrs(attrs, keys):
    """ Return attrs with keys in keys list
    """
    return dict((k, v) for k, v in attrs.items() if k in keys)

def is_sparse_vector(x):
    """ x is a 2D sparse matrix with it's first shape equal to 1.
    """
    return sp.issparse(x) and len(x.shape) == 2 and x.shape[0] == 1

def last_item(array):
    """Returns the last item of an array in a list or an empty list."""
    if array.size == 0:
        # work around for https://github.com/numpy/numpy/issues/5195
        return []

    indexer = (slice(-1, None),) * array.ndim
    return np.ravel(array[indexer]).tolist()

def is_int_type(val):
    """Return True if `val` is of integer type."""
    try:               # Python 2
        return isinstance(val, (int, long))
    except NameError:  # Python 3
        return isinstance(val, int)

def _fill_array_from_list(the_list, the_array):
        """Fill an `array` from a `list`"""
        for i, val in enumerate(the_list):
            the_array[i] = val
        return the_array

def __call__(self, *args, **kwargs):
        """ Instanciates a new *Document* from this collection """
        kwargs["mongokat_collection"] = self
        return self.document_class(*args, **kwargs)

def turn(self):
        """Turn the ring for a single position.
        For example, [a, b, c, d] becomes [b, c, d, a]."""
        first = self._data.pop(0)
        self._data.append(first)

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

def _get_points(self):
        """
        Subclasses may override this method.
        """
        return tuple([self._getitem__points(i)
                     for i in range(self._len__points())])

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

def list_to_csv(my_list, csv_file):
    """
    Save a matrix (list of lists) to a file as a CSV

    .. code:: python

        my_list = [["Name", "Location"],
                   ["Chris", "South Pole"],
                   ["Harry", "Depth of Winter"],
                   ["Bob", "Skull"]]

        reusables.list_to_csv(my_list, "example.csv")

    example.csv

    .. code:: csv

        "Name","Location"
        "Chris","South Pole"
        "Harry","Depth of Winter"
        "Bob","Skull"

    :param my_list: list of lists to save to CSV
    :param csv_file: File to save data to
    """
    if PY3:
        csv_handler = open(csv_file, 'w', newline='')
    else:
        csv_handler = open(csv_file, 'wb')

    try:
        writer = csv.writer(csv_handler, delimiter=',', quoting=csv.QUOTE_ALL)
        writer.writerows(my_list)
    finally:
        csv_handler.close()

def pickle_save(thing,fname):
    """save something to a pickle file"""
    pickle.dump(thing, open(fname,"wb"),pickle.HIGHEST_PROTOCOL)
    return thing

def listlike(obj):
    """Is an object iterable like a list (and not a string)?"""
    
    return hasattr(obj, "__iter__") \
    and not issubclass(type(obj), str)\
    and not issubclass(type(obj), unicode)

def timed (log=sys.stderr, limit=2.0):
    """Decorator to run a function with timing info."""
    return lambda func: timeit(func, log, limit)

def gtype(n):
    """
    Return the a string with the data type of a value, for Graph data
    """
    t = type(n).__name__
    return str(t) if t != 'Literal' else 'Literal, {}'.format(n.language)

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

def duplicates(coll):
    """Return the duplicated items in the given collection

    :param coll: a collection
    :returns: a list of the duplicated items in the collection

    >>> duplicates([1, 1, 2, 3, 3, 4, 1, 1])
    [1, 3]

    """
    return list(set(x for x in coll if coll.count(x) > 1))

def get_serialize_format(self, mimetype):
		""" Get the serialization format for the given mimetype """
		format = self.formats.get(mimetype, None)
		if format is None:
			format = formats.get(mimetype, None)
		return format

def process_wait(process, timeout=0):
    """
    Pauses script execution until a given process exists.
    :param process:
    :param timeout:
    :return:
    """
    ret = AUTO_IT.AU3_ProcessWait(LPCWSTR(process), INT(timeout))
    return ret

def values(self):
        """return a list of all state values"""
        values = []
        for __, data in self.items():
            values.append(data)
        return values

def to_json(data):
    """Return data as a JSON string."""
    return json.dumps(data, default=lambda x: x.__dict__, sort_keys=True, indent=4)

def make_slice_strings(cls, slice_key):
        """
        Converts the given slice key to start and size query parts.
        """
        start = slice_key.start
        size = slice_key.stop - start
        return (str(start), str(size))

def plot_and_save(self, **kwargs):
        """Used when the plot method defined does not create a figure nor calls save_plot
        Then the plot method has to use self.fig"""
        self.fig = pyplot.figure()
        self.plot()
        self.axes = pyplot.gca()
        self.save_plot(self.fig, self.axes, **kwargs)
        pyplot.close(self.fig)

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

def ColumnToIndex (col):
        """convert column to index. Eg: ConvertInIndex("AB") = 28"""
        ndx = 0
        for c in col:
                ndx = ndx * 26 + ord(c.upper()) - 64
        return ndx

def now(self):
		"""
		Return a :py:class:`datetime.datetime` instance representing the current time.

		:rtype: :py:class:`datetime.datetime`
		"""
		if self.use_utc:
			return datetime.datetime.utcnow()
		else:
			return datetime.datetime.now()

def __del__(self):
        """Cleanup the session if it was created here"""
        if self._cleanup_session:
            self._session.loop.run_until_complete(self._session.close())

def eval(self, expression, use_compilation_plan=False):
        """evaluates expression in current context and returns its value"""
        code = 'PyJsEvalResult = eval(%s)' % json.dumps(expression)
        self.execute(code, use_compilation_plan=use_compilation_plan)
        return self['PyJsEvalResult']

def _draw_lines_internal(self, coords, colour, bg):
        """Helper to draw lines connecting a set of nodes that are scaled for the Screen."""
        for i, (x, y) in enumerate(coords):
            if i == 0:
                self._screen.move(x, y)
            else:
                self._screen.draw(x, y, colour=colour, bg=bg, thin=True)

def RunSphinxAPIDoc(_):
  """Runs sphinx-apidoc to auto-generate documentation."""
  current_directory = os.path.abspath(os.path.dirname(__file__))
  module = os.path.join(current_directory, '..', 'plaso')
  api_directory = os.path.join(current_directory, 'sources', 'api')
  apidoc.main(['-o', api_directory, module, '--force'])

def split_len(s, length):
    """split string *s* into list of strings no longer than *length*"""
    return [s[i:i+length] for i in range(0, len(s), length)]

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

def today(year=None):
    """this day, last year"""
    return datetime.date(int(year), _date.month, _date.day) if year else _date

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

def primary_keys_full(cls):
        """Get primary key properties for a SQLAlchemy cls.
        Taken from marshmallow_sqlalchemy
        """
        mapper = cls.__mapper__
        return [
            mapper.get_property_by_column(column)
            for column in mapper.primary_key
        ]

def update(self, **kwargs):
        """ Explicitly reload context with DB usage to get access
        to complete DB object.
        """
        self.reload_context(es_based=False, **kwargs)
        return super(ESCollectionView, self).update(**kwargs)

def shape(self):
        """Compute the shape of the dataset as (rows, cols)."""
        if not self.data:
            return (0, 0)
        return (len(self.data), len(self.dimensions))

def _set_request_cache_if_django_cache_hit(key, django_cached_response):
        """
        Sets the value in the request cache if the django cached response was a hit.

        Args:
            key (string)
            django_cached_response (CachedResponse)

        """
        if django_cached_response.is_found:
            DEFAULT_REQUEST_CACHE.set(key, django_cached_response.value)

def track_update(self):
        """Update the lastest updated date in the database."""
        metadata = self.info()
        metadata.updated_at = dt.datetime.now()
        self.commit()

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

def user_in_all_groups(user, groups):
    """Returns True if the given user is in all given groups"""
    return user_is_superuser(user) or all(user_in_group(user, group) for group in groups)

def printheader(h=None):
    """Print the header for the CSV table."""
    writer = csv.writer(sys.stdout)
    writer.writerow(header_fields(h))

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

def parsePoint(line):
    """
    Parse a line of text into an MLlib LabeledPoint object.
    """
    values = [float(s) for s in line.split(' ')]
    if values[0] == -1:   # Convert -1 labels to 0 for MLlib
        values[0] = 0
    return LabeledPoint(values[0], values[1:])

def extent(self):
        """Helper for matplotlib imshow"""
        return (
            self.intervals[1].pix1 - 0.5,
            self.intervals[1].pix2 - 0.5,
            self.intervals[0].pix1 - 0.5,
            self.intervals[0].pix2 - 0.5,
        )

def __setitem__(self, _ignored, return_value):
        """Item assignment sets the return value and removes any side effect"""
        self.mock.return_value = return_value
        self.mock.side_effect = None

def _update_index_on_df(df, index_names):
    """Helper function to restore index information after collection. Doesn't
    use self so we can serialize this."""
    if index_names:
        df = df.set_index(index_names)
        # Remove names from unnamed indexes
        index_names = _denormalize_index_names(index_names)
        df.index.names = index_names
    return df

def title(msg):
    """Sets the title of the console window."""
    if sys.platform.startswith("win"):
        ctypes.windll.kernel32.SetConsoleTitleW(tounicode(msg))

def get_parent_var(name, global_ok=False, default=None, skip_frames=0):
    """
    Directly gets a variable from a parent frame-scope.

    Returns
    --------
    Any
        The content of the variable found by the given name, or None.
    """

    scope = get_parent_scope_from_var(name, global_ok=global_ok, skip_frames=skip_frames + 1)

    if not scope:
        return default

    if name in scope.locals:
        return scope.locals.get(name, default)

    return scope.globals.get(name, default)

def get_offset_topic_partition_count(kafka_config):
    """Given a kafka cluster configuration, return the number of partitions
    in the offset topic. It will raise an UnknownTopic exception if the topic
    cannot be found."""
    metadata = get_topic_partition_metadata(kafka_config.broker_list)
    if CONSUMER_OFFSET_TOPIC not in metadata:
        raise UnknownTopic("Consumer offset topic is missing.")
    return len(metadata[CONSUMER_OFFSET_TOPIC])

def render_template(env, filename, values=None):
    """
    Render a jinja template
    """
    if not values:
        values = {}
    tmpl = env.get_template(filename)
    return tmpl.render(values)

def from_traceback(cls, tb):
        """ Construct a Bytecode from the given traceback """
        while tb.tb_next:
            tb = tb.tb_next
        return cls(tb.tb_frame.f_code, current_offset=tb.tb_lasti)

def string_to_int( s ):
  """Convert a string of bytes into an integer, as per X9.62."""
  result = 0
  for c in s:
    if not isinstance(c, int): c = ord( c )
    result = 256 * result + c
  return result

def restore_default_settings():
    """ Restore settings to default values. 
    """
    global __DEFAULTS
    __DEFAULTS.CACHE_DIR = defaults.CACHE_DIR
    __DEFAULTS.SET_SEED = defaults.SET_SEED
    __DEFAULTS.SEED = defaults.SEED
    logging.info('Settings reverted to their default values.')

def visit_ellipsis(self, node, parent):
        """visit an Ellipsis node by returning a fresh instance of it"""
        return nodes.Ellipsis(
            getattr(node, "lineno", None), getattr(node, "col_offset", None), parent
        )

def _cal_dist2center(X, center):
    """ Calculate the SSE to the cluster center
    """
    dmemb2cen = scipy.spatial.distance.cdist(X, center.reshape(1,X.shape[1]), metric='seuclidean')
    return(np.sum(dmemb2cen))

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

def unmatched(match):
    """Return unmatched part of re.Match object."""
    start, end = match.span(0)
    return match.string[:start]+match.string[end:]

def _validate_authority_uri_abs_path(host, path):
        """Ensure that path in URL with authority starts with a leading slash.

        Raise ValueError if not.
        """
        if len(host) > 0 and len(path) > 0 and not path.startswith("/"):
            raise ValueError(
                "Path in a URL with authority " "should start with a slash ('/') if set"
            )

def _is_osx_107():
    """
    :return:
        A bool if the current machine is running OS X 10.7
    """

    if sys.platform != 'darwin':
        return False
    version = platform.mac_ver()[0]
    return tuple(map(int, version.split('.')))[0:2] == (10, 7)

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

def b(s):
	""" Encodes Unicode strings to byte strings, if necessary. """

	return s if isinstance(s, bytes) else s.encode(locale.getpreferredencoding())

def pprint(self, seconds):
        """
        Pretty Prints seconds as Hours:Minutes:Seconds.MilliSeconds

        :param seconds:  The time in seconds.
        """
        return ("%d:%02d:%02d.%03d", reduce(lambda ll, b: divmod(ll[0], b) + ll[1:], [(seconds * 1000,), 1000, 60, 60]))

def positive_integer(anon, obj, field, val):
    """
    Returns a random positive integer (for a Django PositiveIntegerField)
    """
    return anon.faker.positive_integer(field=field)

def union(cls, *sets):
        """
        >>> from utool.util_set import *  # NOQA
        """
        import utool as ut
        lists_ = ut.flatten([list(s) for s in sets])
        return cls(lists_)

def __add__(self, other):
        """Concatenate two InferenceData objects."""
        return concat(self, other, copy=True, inplace=False)

def lowstrip(term):
    """Convert to lowercase and strip spaces"""
    term = re.sub('\s+', ' ', term)
    term = term.lower()
    return term

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

def get_properties(cls):
        """Get all properties of the MessageFlags class."""
        property_names = [p for p in dir(cls)
                          if isinstance(getattr(cls, p), property)]
        return property_names

def softmax(xs):
    """Stable implementation of the softmax function."""
    ys = xs - np.max(xs)
    exps = np.exp(ys)
    return exps / exps.sum(axis=0)

def set_cursor_position(self, position):
        """Set cursor position"""
        position = self.get_position(position)
        cursor = self.textCursor()
        cursor.setPosition(position)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

def get_chunks(source, chunk_len):
    """ Returns an iterator over 'chunk_len' chunks of 'source' """
    return (source[i: i + chunk_len] for i in range(0, len(source), chunk_len))

def chmod_add_excute(filename):
        """
        Adds execute permission to file.
        :param filename:
        :return:
        """
        st = os.stat(filename)
        os.chmod(filename, st.st_mode | stat.S_IEXEC)

def strip(notebook):
    """Remove outputs from a notebook."""
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None

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

def append_position_to_token_list(token_list):
    """Converts a list of Token into a list of Token, asuming size == 1"""
    return [PositionToken(value.content, value.gd, index, index+1) for (index, value) in enumerate(token_list)]

def shape(self) -> Tuple[int, ...]:
        """Shape of histogram's data.

        Returns
        -------
        One-element tuple with the number of bins along each axis.
        """
        return tuple(bins.bin_count for bins in self._binnings)

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

def is_function(self):
        """return True if callback is a vanilla plain jane function"""
        if self.is_instance() or self.is_class(): return False
        return isinstance(self.callback, (Callable, classmethod))

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

def txt_line_iterator(path):
  """Iterate through lines of file."""
  with tf.gfile.Open(path) as f:
    for line in f:
      yield line.strip()

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

def file_or_default(path, default, function = None):
    """ Return a default value if a file does not exist """
    try:
        result = file_get_contents(path)
        if function != None: return function(result)
        return result
    except IOError as e:
        if e.errno == errno.ENOENT: return default
        raise

def async_update(self, event):
        """New event for light.

        Check that state is part of event.
        Signal that light has updated state.
        """
        self.update_attr(event.get('state', {}))
        super().async_update(event)

def indent(self):
        """
        Begins an indented block. Must be used in a 'with' code block.
        All calls to the logger inside of the block will be indented.
        """
        blk = IndentBlock(self, self._indent)
        self._indent += 1
        return blk

def strip_accents(text):
    """
    Strip agents from a string.
    """

    normalized_str = unicodedata.normalize('NFD', text)

    return ''.join([
        c for c in normalized_str if unicodedata.category(c) != 'Mn'])

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

def polite_string(a_string):
    """Returns a "proper" string that should work in both Py3/Py2"""
    if is_py3() and hasattr(a_string, 'decode'):
        try:
            return a_string.decode('utf-8')
        except UnicodeDecodeError:
            return a_string

    return a_string

def getSystemVariable(self, remote, name):
        """Get single system variable from CCU / Homegear"""
        if self._server is not None:
            return self._server.getSystemVariable(remote, name)

def purge_cache(self, object_type):
        """ Purge the named cache of all values. If no cache exists for object_type, nothing is done """
        if object_type in self.mapping:
            cache = self.mapping[object_type]
            log.debug("Purging [{}] cache of {} values.".format(object_type, len(cache)))
            cache.purge()

def get_lons_from_cartesian(x__, y__):
    """Get longitudes from cartesian coordinates.
    """
    return rad2deg(arccos(x__ / sqrt(x__ ** 2 + y__ ** 2))) * sign(y__)

def _subclassed(base, *classes):
        """Check if all classes are subclassed from base.
        """
        return all(map(lambda obj: isinstance(obj, base), classes))

def bytesize(arr):
    """
    Returns the memory byte size of a Numpy array as an integer.
    """
    byte_size = np.prod(arr.shape) * np.dtype(arr.dtype).itemsize
    return byte_size

def pop (self, key):
        """Remove key from dict and return value."""
        if key in self._keys:
            self._keys.remove(key)
        super(ListDict, self).pop(key)

def selectnotin(table, field, value, complement=False):
    """Select rows where the given field is not a member of the given value."""

    return select(table, field, lambda v: v not in value,
                  complement=complement)

def main():
    """Parse the command line and run :func:`migrate`."""
    parser = get_args_parser()
    args = parser.parse_args()
    config = Config.from_parse_args(args)
    migrate(config)

def transpose(table):
    """
    transpose matrix
    """
    t = []
    for i in range(0, len(table[0])):
        t.append([row[i] for row in table])
    return t

def find_object(self, object_type):
        """Finds the closest object of a given type."""
        node = self
        while node is not None:
            if isinstance(node.obj, object_type):
                return node.obj
            node = node.parent

def readCommaList(fileList):
    """ Return a list of the files with the commas removed. """
    names=fileList.split(',')
    fileList=[]
    for item in names:
        fileList.append(item)
    return fileList

def _match_space_at_line(line):
    """Return a re.match object if an empty comment was found on line."""
    regex = re.compile(r"^{0}$".format(_MDL_COMMENT))
    return regex.match(line)

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

def llen(self, name):
        """
        Returns the length of the list.

        :param name: str     the name of the redis key
        :return: Future()
        """
        with self.pipe as pipe:
            return pipe.llen(self.redis_key(name))

def color_string(color, string):
    """
    Colorizes a given string, if coloring is available.
    """
    if not color_available:
        return string

    return color + string + colorama.Fore.RESET

def plot(self):
        """Plot the empirical histogram versus best-fit distribution's PDF."""
        plt.plot(self.bin_edges, self.hist, self.bin_edges, self.best_pdf)

def load_files(files):
    """Load and execute a python file."""

    for py_file in files:
        LOG.debug("exec %s", py_file)
        execfile(py_file, globals(), locals())

def check_clang_apply_replacements_binary(args):
  """Checks if invoking supplied clang-apply-replacements binary works."""
  try:
    subprocess.check_call([args.clang_apply_replacements_binary, '--version'])
  except:
    print('Unable to run clang-apply-replacements. Is clang-apply-replacements '
          'binary correctly specified?', file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

def download(url, encoding='utf-8'):
    """Returns the text fetched via http GET from URL, read as `encoding`"""
    import requests
    response = requests.get(url)
    response.encoding = encoding
    return response.text

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

def create_rot2d(angle):
    """Create 2D rotation matrix"""
    ca = math.cos(angle)
    sa = math.sin(angle)
    return np.array([[ca, -sa], [sa, ca]])

def send_file(self, local_path, remote_path, user='root', unix_mode=None):
        """Upload a local file on the remote host.
        """
        self.enable_user(user)
        return self.ssh_pool.send_file(user, local_path, remote_path, unix_mode=unix_mode)

def py(self, output):
        """Output data as a nicely-formatted python data structure"""
        import pprint
        pprint.pprint(output, stream=self.outfile)

def trigger_installed(connection: connection, table: str, schema: str='public'):
    """Test whether or not a psycopg2-pgevents trigger is installed for a table.

    Parameters
    ----------
    connection: psycopg2.extensions.connection
        Active connection to a PostGreSQL database.
    table: str
        Table whose trigger-existence will be checked.
    schema: str
        Schema to which the table belongs.

    Returns
    -------
    bool
        True if the trigger is installed, otherwise False.

    """
    installed = False

    log('Checking if {}.{} trigger installed...'.format(schema, table), logger_name=_LOGGER_NAME)

    statement = SELECT_TRIGGER_STATEMENT.format(
        table=table,
        schema=schema
    )

    result = execute(connection, statement)
    if result:
        installed = True

    log('...{}installed'.format('' if installed else 'NOT '), logger_name=_LOGGER_NAME)

    return installed

def _clone(self, *args, **kwargs):
        """
        Ensure attributes are copied to subsequent queries.
        """
        for attr in ("_search_terms", "_search_fields", "_search_ordered"):
            kwargs[attr] = getattr(self, attr)
        return super(SearchableQuerySet, self)._clone(*args, **kwargs)

def is_defined(self, obj, force_import=False):
        """Return True if object is defined in current namespace"""
        from spyder_kernels.utils.dochelpers import isdefined

        ns = self._get_current_namespace(with_magics=True)
        return isdefined(obj, force_import=force_import, namespace=ns)

def read_mm_header(fd, byte_order, dtype, count):
    """Read MM_HEADER tag from file and return as numpy.rec.array."""
    return numpy.rec.fromfile(fd, MM_HEADER, 1, byteorder=byte_order)[0]

def date(start, end):
    """Get a random date between two dates"""

    stime = date_to_timestamp(start)
    etime = date_to_timestamp(end)

    ptime = stime + random.random() * (etime - stime)

    return datetime.date.fromtimestamp(ptime)

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

def _cdf(self, xloc, dist, base, cache):
        """Cumulative distribution function."""
        return evaluation.evaluate_forward(dist, base**xloc, cache=cache)

def get_obj(ref):
    """Get object from string reference."""
    oid = int(ref)
    return server.id2ref.get(oid) or server.id2obj[oid]

def get_months_apart(d1, d2):
    """
    Get amount of months between dates
    http://stackoverflow.com/a/4040338
    """

    return (d1.year - d2.year)*12 + d1.month - d2.month

def stop_server(self):
        """
        Stop receiving connections, wait for all tasks to end, and then 
        terminate the server.
        """
        self.stop = True
        while self.task_count:
            time.sleep(END_RESP)
        self.terminate = True

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

def _read_indexlist(self, name):
        """Read a list of indexes."""
        setattr(self, '_' + name, [self._timeline[int(i)] for i in
                                   self.db.lrange('site:{0}'.format(name), 0,
                                                  -1)])

def __call__(self, factory_name, *args, **kwargs):
        """Create object."""
        return self.factories[factory_name](*args, **kwargs)

def get_from_gnucash26_date(date_str: str) -> date:
    """ Creates a datetime from GnuCash 2.6 date string """
    date_format = "%Y%m%d"
    result = datetime.strptime(date_str, date_format).date()
    return result

def mimetype(self):
        """MIME type of the asset."""
        return (self.environment.mimetypes.get(self.format_extension) or
                self.compiler_mimetype or 'application/octet-stream')

def readable(path):
    """Test whether a path exists and is readable.  Returns None for
    broken symbolic links or a failing stat() and False if
    the file exists but does not have read permission. True is returned
    if the file is readable."""
    try:
        st = os.stat(path)
        return 0 != st.st_mode & READABLE_MASK
    except os.error:
        return None
    return True

async def join(self, ctx, *, channel: discord.VoiceChannel):
        """Joins a voice channel"""

        if ctx.voice_client is not None:
            return await ctx.voice_client.move_to(channel)

        await channel.connect()

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

def almost_hermitian(gate: Gate) -> bool:
    """Return true if gate tensor is (almost) Hermitian"""
    return np.allclose(asarray(gate.asoperator()),
                       asarray(gate.H.asoperator()))

def _callable_once(func):
    """ Returns a function that is only callable once; any other call will do nothing """

    def once(*args, **kwargs):
        if not once.called:
            once.called = True
            return func(*args, **kwargs)

    once.called = False
    return once

def string_input(prompt=''):
    """Python 3 input()/Python 2 raw_input()"""
    v = sys.version[0]
    if v == '3':
        return input(prompt)
    else:
        return raw_input(prompt)

def write_str2file(pathname, astr):
    """writes a string to file"""
    fname = pathname
    fhandle = open(fname, 'wb')
    fhandle.write(astr)
    fhandle.close()

def transform_to_3d(points,normal,z=0):
    """Project points into 3d from 2d points."""
    d = np.cross(normal, (0, 0, 1))
    M = rotation_matrix(d)
    transformed_points = M.dot(points.T).T + z
    return transformed_points

def get():
    """ Get local facts about this machine.

    Returns:
        json-compatible dict with all facts of this host
    """
    result = runCommand('facter --json', raise_error_on_fail=True)
    json_facts = result[1]
    facts = json.loads(json_facts)
    return facts

def camel_to_snake_case(name):
    """Takes a camelCased string and converts to snake_case."""
    pattern = r'[A-Z][a-z]+|[A-Z]+(?![a-z])'
    return '_'.join(map(str.lower, re.findall(pattern, name)))

def sorted(self):
        """Utility function for sort_file_tabs_alphabetically()."""
        for i in range(0, self.tabs.tabBar().count() - 1):
            if (self.tabs.tabBar().tabText(i) >
                    self.tabs.tabBar().tabText(i + 1)):
                return False
        return True

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

def _guess_extract_method(fname):
  """Guess extraction method, given file name (or path)."""
  for method, extensions in _EXTRACTION_METHOD_TO_EXTS:
    for ext in extensions:
      if fname.endswith(ext):
        return method
  return ExtractMethod.NO_EXTRACT

def get_incomplete_path(filename):
  """Returns a temporary filename based on filename."""
  random_suffix = "".join(
      random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
  return filename + ".incomplete" + random_suffix

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

def items(self):
    """Return a list of the (name, value) pairs of the enum.

    These are returned in the order they were defined in the .proto file.
    """
    return [(value_descriptor.name, value_descriptor.number)
            for value_descriptor in self._enum_type.values]

def uncamel(name):
    """Transform CamelCase naming convention into C-ish convention."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def swap(self):
        """Return the box (for horizontal graphs)"""
        self.xmin, self.ymin = self.ymin, self.xmin
        self.xmax, self.ymax = self.ymax, self.xmax

def EvalGaussianPdf(x, mu, sigma):
    """Computes the unnormalized PDF of the normal distribution.

    x: value
    mu: mean
    sigma: standard deviation
    
    returns: float probability density
    """
    return scipy.stats.norm.pdf(x, mu, sigma)

def _get_compiled_ext():
    """Official way to get the extension of compiled files (.pyc or .pyo)"""
    for ext, mode, typ in imp.get_suffixes():
        if typ == imp.PY_COMPILED:
            return ext

def write_text(filename: str, text: str) -> None:
    """
    Writes text to a file.
    """
    with open(filename, 'w') as f:  # type: TextIO
        print(text, file=f)

def __rmatmul__(self, other):
        """
        Matrix multiplication using binary `@` operator in Python>=3.5.
        """
        return self.T.dot(np.transpose(other)).T

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

def timestamp(format=DATEFMT, timezone='Africa/Johannesburg'):
    """ Return current datetime with timezone applied
        [all timezones] print sorted(pytz.all_timezones) """

    return formatdate(datetime.now(tz=pytz.timezone(timezone)))

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

def double_exponential_moving_average(data, period):
    """
    Double Exponential Moving Average.

    Formula:
    DEMA = 2*EMA - EMA(EMA)
    """
    catch_errors.check_for_period_error(data, period)

    dema = (2 * ema(data, period)) - ema(ema(data, period), period)
    return dema

def bin_to_int(string):
    """Convert a one element byte string to signed int for python 2 support."""
    if isinstance(string, str):
        return struct.unpack("b", string)[0]
    else:
        return struct.unpack("b", bytes([string]))[0]

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

def printOut(value, end='\n'):
    """
    This function prints the given String immediately and flushes the output.
    """
    sys.stdout.write(value)
    sys.stdout.write(end)
    sys.stdout.flush()

def _get_example_length(example):
  """Returns the maximum length between the example inputs and targets."""
  length = tf.maximum(tf.shape(example[0])[0], tf.shape(example[1])[0])
  return length

def runcode(code):
	"""Run the given code line by line with printing, as list of lines, and return variable 'ans'."""
	for line in code:
		print('# '+line)
		exec(line,globals())
	print('# return ans')
	return ans

def _handle_shell(self,cfg_file,*args,**options):
        """Command 'supervisord shell' runs the interactive command shell."""
        args = ("--interactive",) + args
        return supervisorctl.main(("-c",cfg_file) + args)

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

def _round_half_hour(record):
    """
    Round a time DOWN to half nearest half-hour.
    """
    k = record.datetime + timedelta(minutes=-(record.datetime.minute % 30))
    return datetime(k.year, k.month, k.day, k.hour, k.minute, 0)

def arguments_as_dict(cls, *args, **kwargs):
        """
        Generate the arguments dictionary provided to :py:meth:`generate_name` and :py:meth:`calculate_total_steps`.

        This makes it possible to fetch arguments by name regardless of
        whether they were passed as positional or keyword arguments.  Unnamed
        positional arguments are provided as a tuple under the key ``pos``.
        """
        all_args = (None, ) + args
        return inspect.getcallargs(cls.run, *all_args, **kwargs)

def _is_override(meta, method):
        """Checks whether given class or instance method has been marked
        with the ``@override`` decorator.
        """
        from taipan.objective.modifiers import _OverriddenMethod
        return isinstance(method, _OverriddenMethod)

def find_nearest_index(arr, value):
    """For a given value, the function finds the nearest value
    in the array and returns its index."""
    arr = np.array(arr)
    index = (abs(arr-value)).argmin()
    return index

def column(self):
        """
        Returns a zero-based column number of the beginning of this range.
        """
        line, column = self.source_buffer.decompose_position(self.begin_pos)
        return column

def _pip_exists(self):
        """Returns True if pip exists inside the virtual environment. Can be
        used as a naive way to verify that the environment is installed."""
        return os.path.isfile(os.path.join(self.path, 'bin', 'pip'))

def get_count(self, query):
        """
        Returns a number of query results. This is faster than .count() on the query
        """
        count_q = query.statement.with_only_columns(
            [func.count()]).order_by(None)
        count = query.session.execute(count_q).scalar()
        return count

def draw(graph, fname):
    """Draw a graph and save it into a file"""
    ag = networkx.nx_agraph.to_agraph(graph)
    ag.draw(fname, prog='dot')

def argsort_indices(a, axis=-1):
    """Like argsort, but returns an index suitable for sorting the
    the original array even if that array is multidimensional
    """
    a = np.asarray(a)
    ind = list(np.ix_(*[np.arange(d) for d in a.shape]))
    ind[axis] = a.argsort(axis)
    return tuple(ind)

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

def get_gzipped_contents(input_file):
    """
    Returns a gzipped version of a previously opened file's buffer.
    """
    zbuf = StringIO()
    zfile = GzipFile(mode="wb", compresslevel=6, fileobj=zbuf)
    zfile.write(input_file.read())
    zfile.close()
    return ContentFile(zbuf.getvalue())

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

def is_callable_tag(tag):
    """ Determine whether :tag: is a valid callable string tag.

    String is assumed to be valid callable if it starts with '{{'
    and ends with '}}'.

    :param tag: String name of tag.
    """
    return (isinstance(tag, six.string_types) and
            tag.strip().startswith('{{') and
            tag.strip().endswith('}}'))

def cric__lasso():
    """ Lasso Regression
    """
    model = sklearn.linear_model.LogisticRegression(penalty="l1", C=0.002)

    # we want to explain the raw probability outputs of the trees
    model.predict = lambda X: model.predict_proba(X)[:,1]
    
    return model

def get_order(self):
        """
        Return a list of dicionaries. See `set_order`.
        """
        return [dict(reverse=r[0], key=r[1]) for r in self.get_model()]

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

def remove_last_entry(self):
        """Remove the last NoteContainer in the Bar."""
        self.current_beat -= 1.0 / self.bar[-1][1]
        self.bar = self.bar[:-1]
        return self.current_beat

def vowels(self):
        """
        Return a new IPAString, containing only the vowels in the current string.

        :rtype: IPAString
        """
        return IPAString(ipa_chars=[c for c in self.ipa_chars if c.is_vowel])

def _ws_on_close(self, ws: websocket.WebSocketApp):
        """Callback for closing the websocket connection

        Args:
            ws: websocket connection (now closed)
        """
        self.connected = False
        self.logger.error('Websocket closed')
        self._reconnect_websocket()

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

def set_constraint_bound(self, name, value):
        """Set the upper bound of a constraint."""
        index = self._get_constraint_index(name)
        self.upper_bounds[index] = value
        self._reset_solution()

def poke_array(self, store, name, elemtype, elements, container, visited, _stack):
        """abstract method"""
        raise NotImplementedError

def input_yn(conf_mess):
    """Print Confirmation Message and Get Y/N response from user."""
    ui_erase_ln()
    ui_print(conf_mess)
    with term.cbreak():
        input_flush()
        val = input_by_key()
    return bool(val.lower() == 'y')

def _delete_local(self, filename):
        """Deletes the specified file from the local filesystem."""

        if os.path.exists(filename):
            os.remove(filename)

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

def loads(string):
  """
  Deserializes Java objects and primitive data serialized by ObjectOutputStream
  from a string.
  """
  f = StringIO.StringIO(string)
  marshaller = JavaObjectUnmarshaller(f)
  marshaller.add_transformer(DefaultObjectTransformer())
  return marshaller.readObject()

def insort_no_dup(lst, item):
    """
    If item is not in lst, add item to list at its sorted position
    """
    import bisect
    ix = bisect.bisect_left(lst, item)
    if lst[ix] != item: 
        lst[ix:ix] = [item]

async def fetchall(self) -> Iterable[sqlite3.Row]:
        """Fetch all remaining rows."""
        return await self._execute(self._cursor.fetchall)

def init():
    """
    Execute init tasks for all components (virtualenv, pip).
    """
    print(yellow("# Setting up environment...\n", True))
    virtualenv.init()
    virtualenv.update_requirements()
    print(green("\n# DONE.", True))
    print(green("Type ") + green("activate", True) + green(" to enable your virtual environment."))

def extent_count(self):
        """
        Returns the volume group extent count.
        """
        self.open()
        count = lvm_vg_get_extent_count(self.handle)
        self.close()
        return count

def calculate_size(name, count):
    """ Calculates the request payload size"""
    data_size = 0
    data_size += calculate_size_str(name)
    data_size += INT_SIZE_IN_BYTES
    return data_size

def logout(cache):
    """
    Logs out the current session by removing it from the cache. This is
    expected to only occur when a session has
    """
    cache.set(flask.session['auth0_key'], None)
    flask.session.clear()
    return True

def polyline(self, arr):
        """Draw a set of lines"""
        for i in range(0, len(arr) - 1):
            self.line(arr[i][0], arr[i][1], arr[i + 1][0], arr[i + 1][1])

def form_valid(self, form):
        """Security check complete. Log the user in."""
        auth_login(self.request, form.get_user())
        return HttpResponseRedirect(self.get_success_url())

def array(self):
        """
        return the underlying numpy array
        """
        return np.arange(self.start, self.stop, self.step)

def unique_deps(deps):
    """Remove duplicities from deps list of the lists"""
    deps.sort()
    return list(k for k, _ in itertools.groupby(deps))

def read32(bytestream):
  """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]

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

def contains(self, token: str) -> bool:
        """Return if the token is in the list or not."""
        self._validate_token(token)
        return token in self

def camel_to_(s):
    """
    Convert CamelCase to camel_case
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def generate_uuid():
    """Generate a UUID."""
    r_uuid = base64.urlsafe_b64encode(uuid.uuid4().bytes)
    return r_uuid.decode().replace('=', '')

def group(data, num):
    """ Split data into chunks of num chars each """
    return [data[i:i+num] for i in range(0, len(data), num)]

def json_to_initkwargs(self, json_data, kwargs):
        """Subclassing hook to specialize how JSON data is converted
        to keyword arguments"""
        if isinstance(json_data, basestring):
            json_data = json.loads(json_data)
        return json_to_initkwargs(self, json_data, kwargs)

def check_player_collision(self):
        """Check to see if we are colliding with the player."""
        player_tiles = r.TileMapManager.active_map.grab_collisions(self.char.coords)
        enemy_tiles = r.TileMapManager.active_map.grab_collisions(self.coords)

        #Check to see if any of the tiles are the same. If so, there is a collision.
        for ptile in player_tiles:
            for etile in enemy_tiles:
                if r.TileMapManager.active_map.pixels_to_tiles(ptile.coords) == r.TileMapManager.active_map.pixels_to_tiles(etile.coords):
                    return True

        return False

def delete(self, key_name):
        """Delete the key and return true if the key was deleted, else false
        """
        self.db.remove(Query().name == key_name)
        return self.get(key_name) == {}

def from_file(cls, file_path, validate=True):
        """ Creates a Python object from a XML file

        :param file_path: Path to the XML file
        :param validate: XML should be validated against the embedded XSD definition
        :type validate: Boolean
        :returns: the Python object
        """
        return xmlmap.load_xmlobject_from_file(file_path, xmlclass=cls, validate=validate)

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

def write_login(collector, image, **kwargs):
    """Login to a docker registry with write permissions"""
    docker_api = collector.configuration["harpoon"].docker_api
    collector.configuration["authentication"].login(docker_api, image, is_pushing=True, global_docker=True)

def multidict_to_dict(d):
    """
    Turns a werkzeug.MultiDict or django.MultiValueDict into a dict with
    list values
    :param d: a MultiDict or MultiValueDict instance
    :return: a dict instance
    """
    return dict((k, v[0] if len(v) == 1 else v) for k, v in iterlists(d))

def raw_connection_from(engine_or_conn):
    """Extract a raw_connection and determine if it should be automatically closed.

    Only connections opened by this package will be closed automatically.
    """
    if hasattr(engine_or_conn, 'cursor'):
        return engine_or_conn, False
    if hasattr(engine_or_conn, 'connection'):
        return engine_or_conn.connection, False
    return engine_or_conn.raw_connection(), True

def compute_centroid(points):
    """ Computes the centroid of set of points

    Args:
        points (:obj:`list` of :obj:`Point`)
    Returns:
        :obj:`Point`
    """
    lats = [p[1] for p in points]
    lons = [p[0] for p in points]
    return Point(np.mean(lats), np.mean(lons), None)

def is_adb_detectable(self):
        """Checks if USB is on and device is ready by verifying adb devices."""
        serials = list_adb_devices()
        if self.serial in serials:
            self.log.debug('Is now adb detectable.')
            return True
        return False

def call_and_exit(self, cmd, shell=True):
        """Run the *cmd* and exit with the proper exit code."""
        sys.exit(subprocess.call(cmd, shell=shell))

def _text_to_graphiz(self, text):
        """create a graphviz graph from text"""
        dot = Source(text, format='svg')
        return dot.pipe().decode('utf-8')

def parse(self, data, lexer=None, *args, **kwargs):
        """Parse the input JSON data string into a python data structure.
        Args:
          data: An input data string
          lexer:  An optional ply.lex instance that overrides the default lexer.
        Returns:
          A python dict or list representing the input JSON data.
        """
        if lexer is None:
            lexer = self.lexer
        return self.parser.parse(data, lexer=lexer, *args, **kwargs)

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

def _merge_meta(model1, model2):
    """Simple merge of samplesets."""
    w1 = _get_meta(model1)
    w2 = _get_meta(model2)
    return metadata.merge(w1, w2, metadata_conflicts='silent')

def wrap_key(self, key):
        """Translate the key into the central cell

           This method is only applicable in case of a periodic system.
        """
        return tuple(np.round(
            self.integer_cell.shortest_vector(key)
        ).astype(int))

def close(self):
        """Close child subprocess"""
        if self._subprocess is not None:
            os.killpg(self._subprocess.pid, signal.SIGTERM)
            self._subprocess = None

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

def _brief_print_list(lst, limit=7):
    """Print at most `limit` elements of list."""
    lst = list(lst)
    if len(lst) > limit:
        return _brief_print_list(lst[:limit//2], limit) + ', ..., ' + \
            _brief_print_list(lst[-limit//2:], limit)
    return ', '.join(["'%s'"%str(i) for i in lst])

def wait(self, timeout=None):
    """
    Block until all jobs in the ThreadPool are finished. Beware that this can
    make the program run into a deadlock if another thread adds new jobs to the
    pool!

    # Raises
    Timeout: If the timeout is exceeded.
    """

    if not self.__running:
      raise RuntimeError("ThreadPool ain't running")
    self.__queue.wait(timeout)

def fuzzy_get_tuple(dict_obj, approximate_key, dict_keys=None, key_and_value=False, similarity=0.6, default=None):
    """Find the closest matching key and/or value in a dictionary (must have all string keys!)"""
    return fuzzy_get(dict(('|'.join(str(k2) for k2 in k), v) for (k, v) in viewitems(dict_obj)),
                     '|'.join(str(k) for k in approximate_key), dict_keys=dict_keys,
                     key_and_value=key_and_value, similarity=similarity, default=default)

def combinations(l):
    """Pure-Python implementation of itertools.combinations(l, 2)."""
    result = []
    for x in xrange(len(l) - 1):
        ls = l[x + 1:]
        for y in ls:
            result.append((l[x], y))
    return result

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

def get_hash(self, handle):
        """Return the hash."""
        fpath = self._fpath_from_handle(handle)
        return DiskStorageBroker.hasher(fpath)

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

def remove_hop_by_hop_headers(headers):
    """Remove all HTTP/1.1 "Hop-by-Hop" headers from a list or
    :class:`Headers` object.  This operation works in-place.

    .. versionadded:: 0.5

    :param headers: a list or :class:`Headers` object.
    """
    headers[:] = [
        (key, value) for key, value in headers if not is_hop_by_hop_header(key)
    ]

def match_aspect_to_viewport(self):
        """Updates Camera.aspect to match the viewport's aspect ratio."""
        viewport = self.viewport
        self.aspect = float(viewport.width) / viewport.height

def raise_figure_window(f=0):
    """
    Raises the supplied figure number or figure window.
    """
    if _fun.is_a_number(f): f = _pylab.figure(f)
    f.canvas.manager.window.raise_()

def main():
    """Ideally we shouldn't lose the first second of events"""
    time.sleep(1)
    with Input() as input_generator:
        for e in input_generator:
            print(repr(e))

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

def _GetValue(self, name):
    """Returns the TextFSMValue object natching the requested name."""
    for value in self.values:
      if value.name == name:
        return value

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

def findMin(arr):
    """
    in comparison to argrelmax() more simple and  reliable peak finder
    """
    out = np.zeros(shape=arr.shape, dtype=bool)
    _calcMin(arr, out)
    return out

def duration_expired(start_time, duration_seconds):
    """
    Return True if ``duration_seconds`` have expired since ``start_time``
    """

    if duration_seconds is not None:
        delta_seconds = datetime_delta_to_seconds(dt.datetime.now() - start_time)

        if delta_seconds >= duration_seconds:
            return True

    return False

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

def do(self):
        """
        Set a restore point (copy the object), then call the method.
        :return: obj.do_method(*args)
        """
        self.restore_point = self.obj.copy()
        return self.do_method(self.obj, *self.args)

def split_comment(cls, code):
        """ Removes comments (#...) from python code. """
        if '#' not in code: return code
        #: Remove comments only (leave quoted strings as they are)
        subf = lambda m: '' if m.group(0)[0]=='#' else m.group(0)
        return re.sub(cls.re_pytokens, subf, code)

def reopen(self):
        """Reopen the tough connection.

        It will not complain if the connection cannot be reopened.

        """
        try:
            self._con.reopen()
        except Exception:
            if self._transcation:
                self._transaction = False
                try:
                    self._con.query('rollback')
                except Exception:
                    pass
        else:
            self._transaction = False
            self._closed = False
            self._setsession()
            self._usage = 0

def size(self):
        """The size of this parameter, equivalent to self.value.size"""
        return np.multiply.reduce(self.shape, dtype=np.int32)

def pretty_describe(object, nestedness=0, indent=2):
    """Maintain dict ordering - but make string version prettier"""
    if not isinstance(object, dict):
        return str(object)
    sep = f'\n{" " * nestedness * indent}'
    out = sep.join((f'{k}: {pretty_describe(v, nestedness + 1)}' for k, v in object.items()))
    if nestedness > 0 and out:
        return f'{sep}{out}'
    return out

def unzoom_all(self, event=None):
        """ zoom out full data range """
        if len(self.conf.zoom_lims) > 0:
            self.conf.zoom_lims = [self.conf.zoom_lims[0]]
        self.unzoom(event)

def _clean_name(self, prefix, obj):
        """Create a C variable name with the given prefix based on the name of obj."""
        return '{}{}_{}'.format(prefix, self._uid(), ''.join(c for c in obj.name if c.isalnum()))

def getScreenDims(self):
        """returns a tuple that contains (screen_width,screen_height)
        """
        width = ale_lib.getScreenWidth(self.obj)
        height = ale_lib.getScreenHeight(self.obj)
        return (width,height)

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

def _multiline_width(multiline_s, line_width_fn=len):
    """Visible width of a potentially multiline content."""
    return max(map(line_width_fn, re.split("[\r\n]", multiline_s)))

def _mean_dict(dict_list):
    """Compute the mean value across a list of dictionaries
    """
    return {k: np.array([d[k] for d in dict_list]).mean()
            for k in dict_list[0].keys()}

def xor_bytes(a, b):
    """
    Calculate the byte wise exclusive of of two :class:`bytes` objects
    of the same length.
    """
    assert len(a) == len(b)
    return bytes(map(operator.xor, a, b))

def set_basic_auth(self, username, password):
        """
        Set authenatication.
        """
        from requests.auth import HTTPBasicAuth
        self.auth = HTTPBasicAuth(username, password)
        return self

def from_file(file_path) -> dict:
        """ Load JSON file """
        with io.open(file_path, 'r', encoding='utf-8') as json_stream:
            return Json.parse(json_stream, True)

def new(self, size, fill):
        """Return a new Image instance filled with a color."""
        return Image(PIL.Image.new("RGB", size, fill))

def typename(obj):
    """Returns the type of obj as a string. More descriptive and specific than
    type(obj), and safe for any object, unlike __class__."""
    if hasattr(obj, '__class__'):
        return getattr(obj, '__class__').__name__
    else:
        return type(obj).__name__

def rm(venv_name):
    """ Removes the venv by name """
    inenv = InenvManager()
    venv = inenv.get_venv(venv_name)
    click.confirm("Delete dir {}".format(venv.path))
    shutil.rmtree(venv.path)

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

def _position():
    """Returns the current xy coordinates of the mouse cursor as a two-integer
    tuple by calling the GetCursorPos() win32 function.

    Returns:
      (x, y) tuple of the current xy coordinates of the mouse cursor.
    """

    cursor = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(cursor))
    return (cursor.x, cursor.y)

def print_log(text, *colors):
    """Print a log message to standard error."""
    sys.stderr.write(sprint("{}: {}".format(script_name, text), *colors) + "\n")

def input(prompt=""):
	"""input([prompt]) -> value

Equivalent to eval(raw_input(prompt))."""
	
	string = stdin_decode(raw_input(prompt))
	
	caller_frame = sys._getframe(1)
	globals = caller_frame.f_globals
	locals = caller_frame.f_locals
	
	return eval(string, globals, locals)

def bbox(self):
        """BBox"""
        return self.left, self.top, self.right, self.bottom

def previous_quarter(d):
    """
    Retrieve the previous quarter for dt
    """
    from django_toolkit.datetime_util import quarter as datetime_quarter
    return quarter( (datetime_quarter(datetime(d.year, d.month, d.day))[0] + timedelta(days=-1)).date() )

def server(self):
        """Returns the size of remote files
        """
        try:
            tar = urllib2.urlopen(self.registry)
            meta = tar.info()
            return int(meta.getheaders("Content-Length")[0])
        except (urllib2.URLError, IndexError):
            return " "

def log_request(self, code='-', size='-'):
        """Selectively log an accepted request."""

        if self.server.logRequests:
            BaseHTTPServer.BaseHTTPRequestHandler.log_request(self, code, size)

def similarity(word1: str, word2: str) -> float:
    """
    Get cosine similarity between two words.
    If a word is not in the vocabulary, KeyError will be raised.

    :param string word1: first word
    :param string word2: second word
    :return: the cosine similarity between the two word vectors
    """
    return _MODEL.similarity(word1, word2)

def get_obj_cols(df):
    """
    Returns names of 'object' columns in the DataFrame.
    """
    obj_cols = []
    for idx, dt in enumerate(df.dtypes):
        if dt == 'object' or is_category(dt):
            obj_cols.append(df.columns.values[idx])

    return obj_cols

def cor(y_true, y_pred):
    """Compute Pearson correlation coefficient.
    """
    y_true, y_pred = _mask_nan(y_true, y_pred)
    return np.corrcoef(y_true, y_pred)[0, 1]

def get_long_description():
    """Convert the README file into the long description.
    """
    with open(path.join(root_path, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
    return long_description

def filter_float(n: Node, query: str) -> float:
    """
    Filter and ensure that the returned value is of type int.
    """
    return _scalariter2item(n, query, float)

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

def on_property_change(self, name, old_value, new_value):
        """
        Called by the instance manager when a component property is modified

        :param name: The changed property name
        :param old_value: The previous property value
        :param new_value: The new property value
        """
        if self._registration is not None:
            # use the registration to trigger the service event
            self._registration.set_properties({name: new_value})

def RadiusGrid(gridSize):
    """
    Return a square grid with values of the distance from the centre 
    of the grid to each gridpoint
    """
    x,y=np.mgrid[0:gridSize,0:gridSize]
    x = x-(gridSize-1.0)/2.0
    y = y-(gridSize-1.0)/2.0
    return np.abs(x+1j*y)

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

def squash(self, a, b):
        """
        Returns a generator that squashes two iterables into one.

        ```
        ['this', 'that'], [[' and', ' or']] => ['this and', 'this or', 'that and', 'that or']
        ```
        """

        return ((''.join(x) if isinstance(x, tuple) else x) for x in itertools.product(a, b))

def start(self, timeout=None):
        """
        Startup of the node.
        :param join: optionally wait for the process to end (default : True)
        :return: None
        """

        assert super(PyrosBase, self).start(timeout=timeout)
        # Because we currently use this to setup connection
        return self.name

def blocking(func, *args, **kwargs):
    """Run a function that uses blocking IO.

    The function is run in the IO thread pool.
    """
    pool = get_io_pool()
    fut = pool.submit(func, *args, **kwargs)
    return fut.result()

def abort(self):
        """ ensure the master exit from Barrier """
        self.mutex.release()
        self.turnstile.release()
        self.mutex.release()
        self.turnstile2.release()

def get_top(self, *args, **kwargs):
        """Return a get_content generator for top submissions.

        Corresponds to the submissions provided by
        ``https://www.reddit.com/top/`` for the session.

        The additional parameters are passed directly into
        :meth:`.get_content`. Note: the `url` parameter cannot be altered.

        """
        return self.get_content(self.config['top'], *args, **kwargs)

def escape(s):
    """Escape a URL including any /."""
    if not isinstance(s, bytes):
        s = s.encode('utf-8')
    return quote(s, safe='~')

def has_edge(self, p_from, p_to):
        """ Returns True when the graph has the given edge. """
        return p_from in self._edges and p_to in self._edges[p_from]

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

def _prtstr(self, obj, dashes):
        """Print object information using a namedtuple and a format pattern."""
        self.prt.write('{DASHES:{N}}'.format(
            DASHES=self.fmt_dashes.format(DASHES=dashes, ID=obj.item_id),
            N=self.dash_len))
        self.prt.write("{INFO}\n".format(INFO=str(obj)))

def split_on(s, sep=" "):
    """Split s by sep, unless it's inside a quote."""
    pattern = '''((?:[^%s"']|"[^"]*"|'[^']*')+)''' % sep

    return [_strip_speechmarks(t) for t in re.split(pattern, s)[1::2]]

def __call__(self, kind: Optional[str] = None, **kwargs):
        """Use the plotter as callable."""
        return plot(self.histogram, kind=kind, **kwargs)

def remove_from_lib(self, name):
        """ Remove an object from the bin folder. """
        self.__remove_path(os.path.join(self.root_dir, "lib", name))

def str2bytes(x):
  """Convert input argument to bytes"""
  if type(x) is bytes:
    return x
  elif type(x) is str:
    return bytes([ ord(i) for i in x ])
  else:
    return str2bytes(str(x))

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

def monthly(date=datetime.date.today()):
    """
    Take a date object and return the first day of the month.
    """
    return datetime.date(date.year, date.month, 1)

def dictify(a_named_tuple):
    """Transform a named tuple into a dictionary"""
    return dict((s, getattr(a_named_tuple, s)) for s in a_named_tuple._fields)

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

def stop(self, dummy_signum=None, dummy_frame=None):
        """ Shutdown process (this method is also a signal handler) """
        logging.info('Shutting down ...')
        self.socket.close()
        sys.exit(0)

def classify_clusters(points, n=10):
    """
    Return an array of K-Means cluster classes for an array of `shapely.geometry.Point` objects.
    """
    arr = [[p.x, p.y] for p in points.values]
    clf = KMeans(n_clusters=n)
    clf.fit(arr)
    classes = clf.predict(arr)
    return classes

def method(func):
    """Wrap a function as a method."""
    attr = abc.abstractmethod(func)
    attr.__imethod__ = True
    return attr

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

def PrintIndented(self, file, ident, code):
        """Takes an array, add indentation to each entry and prints it."""
        for entry in code:
            print >>file, '%s%s' % (ident, entry)

def accel_prev(self, *args):
        """Callback to go to the previous tab. Called by the accel key.
        """
        if self.get_notebook().get_current_page() == 0:
            self.get_notebook().set_current_page(self.get_notebook().get_n_pages() - 1)
        else:
            self.get_notebook().prev_page()
        return True

def send_notice(self, text):
        """Send a notice (from bot) message to the room."""
        return self.client.api.send_notice(self.room_id, text)

def __repr__(self):
        """Return list-lookalike of representation string of objects"""
        strings = []
        for currItem in self:
            strings.append("%s" % currItem)
        return "(%s)" % (", ".join(strings))

def _rm_name_match(s1, s2):
  """
  determine whether two sequence names from a repeatmasker alignment match.

  :return: True if they are the same string, or if one forms a substring of the
           other, else False
  """
  m_len = min(len(s1), len(s2))
  return s1[:m_len] == s2[:m_len]

def is_identifier(string):
    """Check if string could be a valid python identifier

    :param string: string to be tested
    :returns: True if string can be a python identifier, False otherwise
    :rtype: bool
    """
    matched = PYTHON_IDENTIFIER_RE.match(string)
    return bool(matched) and not keyword.iskeyword(string)

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

def read_bytes(fo, writer_schema=None, reader_schema=None):
    """Bytes are encoded as a long followed by that many bytes of data."""
    size = read_long(fo)
    return fo.read(size)

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

def get_window_dim():
    """ gets the dimensions depending on python version and os"""
    version = sys.version_info

    if version >= (3, 3):
        return _size_36()
    if platform.system() == 'Windows':
        return _size_windows()
    return _size_27()

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

def sort_key(val):
    """Sort key for sorting keys in grevlex order."""
    return numpy.sum((max(val)+1)**numpy.arange(len(val)-1, -1, -1)*val)

def empirical(X):
    """Compute empirical covariance as baseline estimator.
    """
    print("Empirical")
    cov = np.dot(X.T, X) / n_samples
    return cov, np.linalg.inv(cov)

def set_scrollregion(self, event=None):
        """ Set the scroll region on the canvas"""
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

def screen_cv2(self):
        """cv2 Image of current window screen"""
        pil_image = self.screen.convert('RGB')
        cv2_image = np.array(pil_image)
        pil_image.close()
        # Convert RGB to BGR 
        cv2_image = cv2_image[:, :, ::-1]
        return cv2_image

def onLeftDown(self, event=None):
        """ left button down: report x,y coords, start zooming mode"""
        if event is None:
            return
        self.cursor_mode_action('leftdown', event=event)
        self.ForwardEvent(event=event.guiEvent)

def input_dir(self):
        """
        :returns: absolute path to where the job.ini is
        """
        return os.path.abspath(os.path.dirname(self.inputs['job_ini']))

def remove_empty_text(utterances: List[Utterance]) -> List[Utterance]:
    """Remove empty utterances from a list of utterances
    Args:
        utterances: The list of utterance we are processing
    """
    return [utter for utter in utterances if utter.text.strip() != ""]

def user_return(self, frame, return_value):
        """This function is called when a return trap is set here."""
        pdb.Pdb.user_return(self, frame, return_value)

def remove_falsy_values(counter: Mapping[Any, int]) -> Mapping[Any, int]:
    """Remove all values that are zero."""
    return {
        label: count
        for label, count in counter.items()
        if count
    }

def autopage(self):
        """Iterate through results from all pages.

        :return: all results
        :rtype: generator
        """
        while self.items:
            yield from self.items
            self.items = self.fetch_next()

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

def make_file_readable (filename):
    """Make file user readable if it is not a link."""
    if not os.path.islink(filename):
        util.set_mode(filename, stat.S_IRUSR)

def matshow(*args, **kwargs):
    """
    imshow without interpolation like as matshow
    :param args:
    :param kwargs:
    :return:
    """
    kwargs['interpolation'] = kwargs.pop('interpolation', 'none')
    return plt.imshow(*args, **kwargs)

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

def rnormal(mu, tau, size=None):
    """
    Random normal variates.
    """
    return np.random.normal(mu, 1. / np.sqrt(tau), size)

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

def indented_show(text, howmany=1):
        """Print a formatted indented text.
        """
        print(StrTemplate.pad_indent(text=text, howmany=howmany))

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

def _cast_to_type(self, value):
        """ Convert the value to its string representation"""
        if isinstance(value, str) or value is None:
            return value
        return str(value)

def distance(vec1, vec2):
        """Calculate the distance between two Vectors"""
        if isinstance(vec1, Vector2) \
                and isinstance(vec2, Vector2):
            dist_vec = vec2 - vec1
            return dist_vec.length()
        else:
            raise TypeError("vec1 and vec2 must be Vector2's")

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

def clean(ctx, text):
    """
    Removes all non-printable characters from a text string
    """
    text = conversions.to_string(text, ctx)
    return ''.join([c for c in text if ord(c) >= 32])

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

def has_edit_permission(self, request):
        """ Can edit this object """
        return request.user.is_authenticated and request.user.is_active and request.user.is_staff

def ave_list_v3(vec_list):
    """Return the average vector of a list of vectors."""

    vec = Vec3(0, 0, 0)
    for v in vec_list:
        vec += v
    num_vecs = float(len(vec_list))
    vec = Vec3(vec.x / num_vecs, vec.y / num_vecs, vec.z / num_vecs)
    return vec

def _get_session():
    """Return (and memoize) a database session"""
    session = getattr(g, '_session', None)
    if session is None:
        session = g._session = db.session()
    return session

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

def ensure_us_time_resolution(val):
    """Convert val out of numpy time, for use in to_dict.
    Needed because of numpy bug GH#7619"""
    if np.issubdtype(val.dtype, np.datetime64):
        val = val.astype('datetime64[us]')
    elif np.issubdtype(val.dtype, np.timedelta64):
        val = val.astype('timedelta64[us]')
    return val

def round_to_x_digits(number, digits):
    """
    Returns 'number' rounded to 'digits' digits.
    """
    return round(number * math.pow(10, digits)) / math.pow(10, digits)

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

def _dt_to_epoch(dt):
        """Convert datetime to epoch seconds."""
        try:
            epoch = dt.timestamp()
        except AttributeError:  # py2
            epoch = (dt - datetime(1970, 1, 1)).total_seconds()
        return epoch

def interpolate_slice(slice_rows, slice_cols, interpolator):
    """Interpolate the given slice of the larger array."""
    fine_rows = np.arange(slice_rows.start, slice_rows.stop, slice_rows.step)
    fine_cols = np.arange(slice_cols.start, slice_cols.stop, slice_cols.step)
    return interpolator(fine_cols, fine_rows)

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

def check_by_selector(self, selector):
    """Check the checkbox matching the CSS selector."""
    elem = find_element_by_jquery(world.browser, selector)
    if not elem.is_selected():
        elem.click()

def code(self):
    """Returns the code object for this BUILD file."""
    return compile(self.source(), self.full_path, 'exec', flags=0, dont_inherit=True)

def _sum_cycles_from_tokens(self, tokens: List[str]) -> int:
        """Sum the total number of cycles over a list of tokens."""
        return sum((int(self._nonnumber_pattern.sub('', t)) for t in tokens))

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

def owner(self):
        """
        Username of document creator
        """
        if self._owner:
            return self._owner
        elif not self.abstract:
            return self.read_meta()._owner

        raise EmptyDocumentException()

def issubset(self, other):
        """Report whether another set contains this RangeSet."""
        self._binary_sanity_check(other)
        return set.issubset(self, other)

def rand_elem(seq, n=None):
    """returns a random element from seq n times. If n is None, it continues indefinitly"""
    return map(random.choice, repeat(seq, n) if n is not None else repeat(seq))

def vars_class(cls):
    """Return a dict of vars for the given class, including all ancestors.

    This differs from the usual behaviour of `vars` which returns attributes
    belonging to the given class and not its ancestors.
    """
    return dict(chain.from_iterable(
        vars(cls).items() for cls in reversed(cls.__mro__)))

def make_file_read_only(file_path):
    """
    Removes the write permissions for the given file for owner, groups and others.

    :param file_path: The file whose privileges are revoked.
    :raise FileNotFoundError: If the given file does not exist.
    """
    old_permissions = os.stat(file_path).st_mode
    os.chmod(file_path, old_permissions & ~WRITE_PERMISSIONS)

def from_json(cls, json_doc):
        """Parse a JSON string and build an entity."""
        try:
            d = json.load(json_doc)
        except AttributeError:  # catch the read() error
            d = json.loads(json_doc)

        return cls.from_dict(d)

def get_cov(config):
    """Returns the coverage object of pytest-cov."""

    # Check with hasplugin to avoid getplugin exception in older pytest.
    if config.pluginmanager.hasplugin('_cov'):
        plugin = config.pluginmanager.getplugin('_cov')
        if plugin.cov_controller:
            return plugin.cov_controller.cov
    return None

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

def to_index(self, index_type, index_name, includes=None):
        """ Create an index field from this field """
        return IndexField(self.name, self.data_type, index_type, index_name, includes)

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

def update(self, **kwargs):
    """Creates or updates a property for the instance for each parameter."""
    for key, value in kwargs.items():
      setattr(self, key, value)

def json_decode(data):
    """
    Decodes the given JSON as primitives
    """
    if isinstance(data, six.binary_type):
        data = data.decode('utf-8')

    return json.loads(data)

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

def dt2jd(dt):
    """Convert datetime to julian date
    """
    a = (14 - dt.month)//12
    y = dt.year + 4800 - a
    m = dt.month + 12*a - 3
    return dt.day + ((153*m + 2)//5) + 365*y + y//4 - y//100 + y//400 - 32045

def test_SVD(pca):
    """
    Function to test the validity of singular
    value decomposition by reconstructing original
    data.
    """
    _ = pca
    rec = N.dot(_.U,N.dot(_.sigma,_.V))
    assert N.allclose(_.arr,rec)

def to_json(self) -> Mapping:
        """Return the properties of this :class:`Sample` as JSON serializable.

        """
        return {str(x): str(y) for x, y in self.items()}

def get_line_flux(line_wave, wave, flux, **kwargs):
    """Interpolated flux at a given wavelength (calls np.interp)."""
    return np.interp(line_wave, wave, flux, **kwargs)

def get_tokens(line: str) -> Iterator[str]:
    """
    Yields tokens from input string.

    :param line: Input string.
    :return: Iterator over tokens.
    """
    for token in line.rstrip().split():
        if len(token) > 0:
            yield token

def get_type_len(self):
        """Retrieve the type and length for a data record."""
        # Check types and set type/len
        self.get_sql()
        return self.type, self.len, self.len_decimal

def setValue(self, p_float):
        """Override method to set a value to show it as 0 to 100.

        :param p_float: The float number that want to be set.
        :type p_float: float
        """
        p_float = p_float * 100

        super(PercentageSpinBox, self).setValue(p_float)

def _delete_keys(dct, keys):
    """Returns a copy of dct without `keys` keys
    """
    c = deepcopy(dct)
    assert isinstance(keys, list)
    for k in keys:
        c.pop(k)
    return c

def close_session(self):
        """ Close tensorflow session. Exposes for memory management. """
        with self._graph.as_default():
            self._sess.close()
            self._sess = None

def impute_data(self,x):
        """Imputes data set containing Nan values"""
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        return imp.fit_transform(x)

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

def uniq(seq):
    """ Return a copy of seq without duplicates. """
    seen = set()
    return [x for x in seq if str(x) not in seen and not seen.add(str(x))]

def find_largest_contig(contig_lengths_dict):
    """
    Determine the largest contig for each strain
    :param contig_lengths_dict: dictionary of strain name: reverse-sorted list of all contig lengths
    :return: longest_contig_dict: dictionary of strain name: longest contig
    """
    # Initialise the dictionary
    longest_contig_dict = dict()
    for file_name, contig_lengths in contig_lengths_dict.items():
        # As the list is sorted in descending order, the largest contig is the first entry in the list
        longest_contig_dict[file_name] = contig_lengths[0]
    return longest_contig_dict

def get_memory(self, mode):
        """Return a smt bit vector that represents a memory location.
        """
        mem = {
            "pre": self._translator.get_memory_init(),
            "post": self._translator.get_memory_curr(),
        }

        return mem[mode]

def extend_with(func):
    """Extends with class or function"""
    if not func.__name__ in ArgParseInator._plugins:
        ArgParseInator._plugins[func.__name__] = func

def gen_random_string(str_len):
    """ generate random string with specified length
    """
    return ''.join(
        random.choice(string.ascii_letters + string.digits) for _ in range(str_len))

def get(key, default=None):
    """ return the key from the request
    """
    data = get_form() or get_query_string()
    return data.get(key, default)

def afx_small():
  """Small transformer model with small batch size for fast step times."""
  hparams = transformer.transformer_tpu()
  hparams.filter_size = 1024
  hparams.num_heads = 4
  hparams.num_hidden_layers = 3
  hparams.batch_size = 512
  return hparams

def snake_case(a_string):
    """Returns a snake cased version of a string.

    :param a_string: any :class:`str` object.

    Usage:
        >>> snake_case('FooBar')
        "foo_bar"
    """

    partial = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', a_string)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', partial).lower()

def list_to_csv(value):
    """
    Converts list to string with comma separated values. For string is no-op.
    """
    if isinstance(value, (list, tuple, set)):
        value = ",".join(value)
    return value

def get_range(self, start=None, stop=None):
		"""Return a RangeMap for the range start to stop.

		Returns:
			A RangeMap
		"""
		return self.from_iterable(self.ranges(start, stop))

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

def finditer(self, string, pos=0, endpos=sys.maxint):
        """Return a list of all non-overlapping matches of pattern in string."""
        scanner = self.scanner(string, pos, endpos)
        return iter(scanner.search, None)

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

def props(cls):
    """
    Class method that returns all defined arguments within the class.
    
    Returns:
      A dictionary containing all action defined arguments (if any).
    """
    return {k:v for (k, v) in inspect.getmembers(cls) if type(v) is Argument}

def _width_is_big_enough(image, width):
    """Check that the image width is superior to `width`"""
    if width > image.size[0]:
        raise ImageSizeError(image.size[0], width)

def to_str(obj):
    """Attempts to convert given object to a string object
    """
    if not isinstance(obj, str) and PY3 and isinstance(obj, bytes):
        obj = obj.decode('utf-8')
    return obj if isinstance(obj, string_types) else str(obj)

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

def is_readable_dir(path):
  """Returns whether a path names an existing directory we can list and read files from."""
  return os.path.isdir(path) and os.access(path, os.R_OK) and os.access(path, os.X_OK)

def param (self, param, kwargs, default_value=False):
        """gets a param from kwargs, or uses a default_value. if found, it's
        removed from kwargs"""
        if param in kwargs:
            value= kwargs[param]
            del kwargs[param]
        else:
            value= default_value
        setattr (self, param, value)

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

def _rectangular(n):
    """Checks to see if a 2D list is a valid 2D matrix"""
    for i in n:
        if len(i) != len(n[0]):
            return False
    return True

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

def str_to_boolean(input_str):
    """ a conversion function for boolean
    """
    if not isinstance(input_str, six.string_types):
        raise ValueError(input_str)
    input_str = str_quote_stripper(input_str)
    return input_str.lower() in ("true", "t", "1", "y", "yes")

def toggle_pause(self):
        """Toggle pause mode"""
        self.controller.playing = not self.controller.playing
        self.music.toggle_pause()

def get_db_version(session):
    """
    :param session: actually it is a sqlalchemy session
    :return: version number
    """
    value = session.query(ProgramInformation.value).filter(ProgramInformation.name == "db_version").scalar()
    return int(value)

def set_subparsers_args(self, *args, **kwargs):
        """
        Sets args and kwargs that are passed when creating a subparsers group
        in an argparse.ArgumentParser i.e. when calling
        argparser.ArgumentParser.add_subparsers
        """
        self.subparsers_args = args
        self.subparsers_kwargs = kwargs

def extend(self, iterable):
        """Extend the list by appending all the items in the given list."""
        return super(Collection, self).extend(
            self._ensure_iterable_is_valid(iterable))

def unapostrophe(text):
    """Strip apostrophe and 's' from the end of a string."""
    text = re.sub(r'[%s]s?$' % ''.join(APOSTROPHES), '', text)
    return text

def ResetConsoleColor() -> bool:
    """
    Reset to the default text color on console window.
    Return bool, True if succeed otherwise False.
    """
    if sys.stdout:
        sys.stdout.flush()
    bool(ctypes.windll.kernel32.SetConsoleTextAttribute(_ConsoleOutputHandle, _DefaultConsoleColor))

def is_unix_like(platform=None):
    """Returns whether the given platform is a Unix-like platform with the usual
    Unix filesystem. When the parameter is omitted, it defaults to ``sys.platform``
    """
    platform = platform or sys.platform
    platform = platform.lower()
    return platform.startswith("linux") or platform.startswith("darwin") or \
            platform.startswith("cygwin")

def save_dict_to_file(filename, dictionary):
  """Saves dictionary as CSV file."""
  with open(filename, 'w') as f:
    writer = csv.writer(f)
    for k, v in iteritems(dictionary):
      writer.writerow([str(k), str(v)])

def filesavebox(msg=None, title=None, argInitialFile=None):
    """Original doc: A file to get the name of a file to save.
        Returns the name of a file, or None if user chose to cancel.

        if argInitialFile contains a valid filename, the dialog will
        be positioned at that file when it appears.
        """
    return psidialogs.ask_file(message=msg, title=title, default=argInitialFile, save=True)

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

def qth_pw(self, q):
        """
        returns the qth most probable element in the dawg.
        """
        return heapq.nlargest(q + 2, self._T.iteritems(),
                              key=operator.itemgetter(1))[-1]

def calculate_dimensions(image, long_side, short_side):
    """Returns the thumbnail dimensions depending on the images format."""
    if image.width >= image.height:
        return '{0}x{1}'.format(long_side, short_side)
    return '{0}x{1}'.format(short_side, long_side)

def find_task_by_id(self, id, session=None):
        """
        Find task with the given record ID.
        """
        with self._session(session) as session:
            return session.query(TaskRecord).get(id)

def chi_square_calc(classes, table, TOP, P, POP):
    """
    Calculate chi-squared.

    :param classes: confusion matrix classes
    :type classes : list
    :param table: confusion matrix table
    :type table : dict
    :param TOP: test outcome positive
    :type TOP : dict
    :param P: condition positive
    :type P : dict
    :param POP: population
    :type POP : dict
    :return: chi-squared as float
    """
    try:
        result = 0
        for i in classes:
            for index, j in enumerate(classes):
                expected = (TOP[j] * P[i]) / (POP[i])
                result += ((table[i][j] - expected)**2) / expected
        return result
    except Exception:
        return "None"

def match_files(files, pattern: Pattern):
    """Yields file name if matches a regular expression pattern."""

    for name in files:
        if re.match(pattern, name):
            yield name

def _load_mod_ui_libraries(self, path):
        """
        :param Path path:
        """
        path = path / Path('mod')
        sys.path.append(str(path))

def rate_limited(max_per_hour: int, *args: Any) -> Callable[..., Any]:
    """Rate limit a function."""
    return util.rate_limited(max_per_hour, *args)

def vector_distance(a, b):
    """The Euclidean distance between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

def url_to_image(url):
    """
    Fetch an image from url and convert it into a Pillow Image object
    """
    r = requests.get(url)
    image = StringIO(r.content)
    return image

def _gzip(self, response):
        """Apply gzip compression to a response."""
        bytesio = six.BytesIO()
        with gzip.GzipFile(fileobj=bytesio, mode='w') as gz:
            gz.write(response)
        return bytesio.getvalue()

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

def get_max(qs, field):
    """
    get max for queryset.

    qs: queryset
    field: The field name to max.
    """
    max_field = '%s__max' % field
    num = qs.aggregate(Max(field))[max_field]
    return num if num else 0

def _is_root():
    """Checks if the user is rooted."""
    import os
    import ctypes
    try:
        return os.geteuid() == 0
    except AttributeError:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    return False

def do_next(self, args):
        """Step over the next statement
        """
        self._do_print_from_last_cmd = True
        self._interp.step_over()
        return True

def fillScreen(self, color=None):
        """Fill the matrix with the given RGB color"""
        md.fill_rect(self.set, 0, 0, self.width, self.height, color)

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

def _skip_section(self):
        """Skip a section"""
        self._last = self._f.readline()
        while len(self._last) > 0 and len(self._last[0].strip()) == 0:
            self._last = self._f.readline()

def calc_base64(s):
    """Return base64 encoded binarystring."""
    s = compat.to_bytes(s)
    s = compat.base64_encodebytes(s).strip()  # return bytestring
    return compat.to_native(s)

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

def __as_list(value: List[JsonObjTypes]) -> List[JsonTypes]:
        """ Return a json array as a list

        :param value: array
        :return: array with JsonObj instances removed
        """
        return [e._as_dict if isinstance(e, JsonObj) else e for e in value]

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

def timeit(func, *args, **kwargs):
    """
    Time execution of function. Returns (res, seconds).

    >>> res, timing = timeit(time.sleep, 1)
    """
    start_time = time.time()
    res = func(*args, **kwargs)
    timing = time.time() - start_time
    return res, timing

def popup(self, title, callfn, initialdir=None):
        """Let user select a directory."""
        super(DirectorySelection, self).popup(title, callfn, initialdir)

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

def access(self, accessor, timeout=None):
        """Return a result from an asyncio future."""
        if self.loop.is_running():
            raise RuntimeError("Loop is already running")
        coro = asyncio.wait_for(accessor, timeout, loop=self.loop)
        return self.loop.run_until_complete(coro)

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

def _baseattrs(self):
        """A dict of members expressed in literals"""

        result = super()._baseattrs
        result["params"] = ", ".join(self.parameters)
        return result

def _get_node_path(self, node):
        """Return the path from the root to ``node`` as a list of node names."""
        path = []
        while node.up:
            path.append(node.name)
            node = node.up
        return list(reversed(path))

def indent(s, spaces=4):
    """
    Inserts `spaces` after each string of new lines in `s`
    and before the start of the string.
    """
    new = re.sub('(\n+)', '\\1%s' % (' ' * spaces), s)
    return (' ' * spaces) + new.strip()

def __delitem__ (self, key):
        """Remove key from dict."""
        self._keys.remove(key)
        super(ListDict, self).__delitem__(key)

def intround(value):
    """Given a float returns a rounded int. Should give the same result on
    both Py2/3
    """

    return int(decimal.Decimal.from_float(
        value).to_integral_value(decimal.ROUND_HALF_EVEN))

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

def doc_parser():
    """Utility function to allow getting the arguments for a single command, for Sphinx documentation"""

    parser = argparse.ArgumentParser(
        prog='ambry',
        description='Ambry {}. Management interface for ambry, libraries '
                    'and repositories. '.format(ambry._meta.__version__))

    return parser

def wireshark(pktlist, *args):
    """Run wireshark on a list of packets"""
    fname = get_temp_file()
    wrpcap(fname, pktlist)
    subprocess.Popen([conf.prog.wireshark, "-r", fname] + list(args))

def find_lt(a, x):
    """Find rightmost value less than x."""
    i = bs.bisect_left(a, x)
    if i: return i - 1
    raise ValueError

def index():
    """ Display productpage with normal user and test user buttons"""
    global productpage

    table = json2html.convert(json = json.dumps(productpage),
                              table_attributes="class=\"table table-condensed table-bordered table-hover\"")

    return render_template('index.html', serviceTable=table)

def ma(self):
        """Represent data as a masked array.

        The array is returned with column-first indexing, i.e. for a data file with
        columns X Y1 Y2 Y3 ... the array a will be a[0] = X, a[1] = Y1, ... .

        inf and nan are filtered via :func:`numpy.isfinite`.
        """
        a = self.array
        return numpy.ma.MaskedArray(a, mask=numpy.logical_not(numpy.isfinite(a)))

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

def postprocessor(prediction):
    """Map prediction tensor to labels."""
    prediction = prediction.data.numpy()[0]
    top_predictions = prediction.argsort()[-3:][::-1]
    return [labels[prediction] for prediction in top_predictions]

def unzip_file_to_dir(path_to_zip, output_directory):
    """
    Extract a ZIP archive to a directory
    """
    z = ZipFile(path_to_zip, 'r')
    z.extractall(output_directory)
    z.close()

def ms_panset(self, viewer, event, data_x, data_y,
                  msg=True):
        """An interactive way to set the pan position.  The location
        (data_x, data_y) will be centered in the window.
        """
        if self.canpan and (event.state == 'down'):
            self._panset(viewer, data_x, data_y, msg=msg)
        return True

def me(self):
        """Similar to :attr:`.Guild.me` except it may return the :class:`.ClientUser` in private message contexts."""
        return self.guild.me if self.guild is not None else self.bot.user

def expired(self):
        """Boolean property if this action has expired
        """
        if self.timeout is None:
            return False

        return monotonic() - self.start_time > self.timeout

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

def ignored_regions(source):
    """Return ignored regions like strings and comments in `source` """
    return [(match.start(), match.end()) for match in _str.finditer(source)]

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

def _manhattan_distance(vec_a, vec_b):
    """Return manhattan distance between two lists of numbers."""
    if len(vec_a) != len(vec_b):
        raise ValueError('len(vec_a) must equal len(vec_b)')
    return sum(map(lambda a, b: abs(a - b), vec_a, vec_b))

def set_input_value(self, selector, value):
        """Set the value of the input matched by given selector."""
        script = 'document.querySelector("%s").setAttribute("value", "%s")'
        script = script % (selector, value)
        self.evaluate(script)

def __delitem__(self, resource):
        """Remove resource instance from internal cache"""
        self.__caches[type(resource)].pop(resource.get_cache_internal_key(), None)

def parse_date(s):
    """Fast %Y-%m-%d parsing."""
    try:
        return datetime.date(int(s[:4]), int(s[5:7]), int(s[8:10]))
    except ValueError:  # other accepted format used in one-day data set
        return datetime.datetime.strptime(s, '%d %B %Y').date()

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

def scopes_as(self, new_scopes):
        """Replace my :attr:`scopes` for the duration of the with block.

        My global scope is not replaced.

        Args:
            new_scopes (list of dict-likes): The new :attr:`scopes` to use.
        """
        old_scopes, self.scopes = self.scopes, new_scopes
        yield
        self.scopes = old_scopes

def __clear_buffers(self):
        """Clears the input and output buffers"""
        try:
            self._port.reset_input_buffer()
            self._port.reset_output_buffer()
        except AttributeError:
            #pySerial 2.7
            self._port.flushInput()
            self._port.flushOutput()

def initialize_api(flask_app):
    """Initialize an API."""
    if not flask_restplus:
        return

    api = flask_restplus.Api(version="1.0", title="My Example API")
    api.add_resource(HelloWorld, "/hello")

    blueprint = flask.Blueprint("api", __name__, url_prefix="/api")
    api.init_app(blueprint)
    flask_app.register_blueprint(blueprint)

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

def _parse_config(config_file_path):
    """ Parse Config File from yaml file. """
    config_file = open(config_file_path, 'r')
    config = yaml.load(config_file)
    config_file.close()
    return config

def test_security(self):
        """ Run security.py -- demonstrate the SECURITY extension """
        self.assertEqual(run_example(examples_folder + "security.py --generate"), 0)
        self.assertEqual(run_example(examples_folder + "security.py --revoke"), 0)

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

def iteritems(data, **kwargs):
    """Iterate over dict items."""
    return iter(data.items(**kwargs)) if IS_PY3 else data.iteritems(**kwargs)

def _get_token(self, oauth_request, token_type='access'):
        """Try to find the token for the provided request token key."""
        token_field = oauth_request.get_parameter('oauth_token')
        token = self.data_store.lookup_token(token_type, token_field)
        if not token:
            raise OAuthError('Invalid %s token: %s' % (token_type, token_field))
        return token

def _divide(self, x1, x2, out):
        """Raw pointwise multiplication of two elements."""
        self.tspace._divide(x1.tensor, x2.tensor, out.tensor)

def _unjsonify(x, isattributes=False):
    """Convert JSON string to an ordered defaultdict."""
    if isattributes:
        obj = json.loads(x)
        return dict_class(obj)
    return json.loads(x)

def _startswith(expr, pat):
    """
    Return boolean sequence or scalar indicating whether each string in the sequence or scalar
    starts with passed pattern. Equivalent to str.startswith().

    :param expr:
    :param pat: Character sequence
    :return: sequence or scalar
    """

    return _string_op(expr, Startswith, output_type=types.boolean, _pat=pat)

def is_serializable(obj):
    """Return `True` if the given object conforms to the Serializable protocol.

    :rtype: bool
    """
    if inspect.isclass(obj):
      return Serializable.is_serializable_type(obj)
    return isinstance(obj, Serializable) or hasattr(obj, '_asdict')

def cli_command_quit(self, msg):
        """\
        kills the child and exits
        """
        if self.state == State.RUNNING and self.sprocess and self.sprocess.proc:
            self.sprocess.proc.kill()
        else:
            sys.exit(0)

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

def include_raw_constructor(self, loader, node):
        """
        Called when PyYaml encounters '!include-raw'
        """

        path = convert_path(node.value)

        with open(path, 'r') as f:
            config = f.read()

            config = self.inject_include_info(path, config, include_type='include-raw')

            self.add_file(path, config)

            return config

def barray(iterlines):
    """
    Array of bytes
    """
    lst = [line.encode('utf-8') for line in iterlines]
    arr = numpy.array(lst)
    return arr

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

def _increment_numeric_suffix(s):
        """Increment (or add) numeric suffix to identifier."""
        if re.match(r".*\d+$", s):
            return re.sub(r"\d+$", lambda n: str(int(n.group(0)) + 1), s)
        return s + "_2"

def move_up(lines=1, file=sys.stdout):
    """ Move the cursor up a number of lines.

        Esc[ValueA:
        Moves the cursor up by the specified number of lines without changing
        columns. If the cursor is already on the top line, ANSI.SYS ignores
        this sequence.
    """
    move.up(lines).write(file=file)

def end_index(self):
        """Return the 1-based index of the last item on this page."""
        paginator = self.paginator
        # Special case for the last page because there can be orphans.
        if self.number == paginator.num_pages:
            return paginator.count
        return (self.number - 1) * paginator.per_page + paginator.first_page

def load_from_file(cls, filename_prefix):
    """Extracts list of subwords from file."""
    filename = cls._filename(filename_prefix)
    lines, _ = cls._read_lines_from_file(filename)
    # Strip wrapping single quotes
    vocab_list = [line[1:-1] for line in lines]
    return cls(vocab_list=vocab_list)

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

def replaceNewlines(string, newlineChar):
	"""There's probably a way to do this with string functions but I was lazy.
		Replace all instances of \r or \n in a string with something else."""
	if newlineChar in string:
		segments = string.split(newlineChar)
		string = ""
		for segment in segments:
			string += segment
	return string

def bool_str(string):
    """Returns a boolean from a string imput of 'true' or 'false'"""
    if string not in BOOL_STRS:
        raise ValueError('Invalid boolean string: "{}"'.format(string))
    return True if string == 'true' else False

def create_widget(self):
        """ Create the toolkit widget for the proxy object.
        """
        d = self.declaration
        button_type = UIButton.UIButtonTypeSystem if d.flat else UIButton.UIButtonTypeRoundedRect
        self.widget = UIButton(buttonWithType=button_type)

def _connect(self, servers):
        """ connect to the given server, e.g.: \\connect localhost:4200 """
        self._do_connect(servers.split(' '))
        self._verify_connection(verbose=True)

def setLib(self, lib):
        """ Copy the lib items into our font. """
        for name, item in lib.items():
            self.font.lib[name] = item

def getBuffer(x):
    """
    Copy @x into a (modifiable) ctypes byte array
    """
    b = bytes(x)
    return (c_ubyte * len(b)).from_buffer_copy(bytes(x))

def clean(self, text):
        """Remove all unwanted characters from text."""
        return ''.join([c for c in text if c in self.alphabet])

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

def movingaverage(arr, window):
    """
    Calculates the moving average ("rolling mean") of an array
    of a certain window size.
    """
    m = np.ones(int(window)) / int(window)
    return scipy.ndimage.convolve1d(arr, m, axis=0, mode='reflect')

def point8_to_box(points):
    """
    Args:
        points: (nx4)x2
    Returns:
        nx4 boxes (x1y1x2y2)
    """
    p = points.reshape((-1, 4, 2))
    minxy = p.min(axis=1)   # nx2
    maxxy = p.max(axis=1)   # nx2
    return np.concatenate((minxy, maxxy), axis=1)

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

def matches_glob_list(path, glob_list):
    """
    Given a list of glob patterns, returns a boolean
    indicating if a path matches any glob in the list
    """
    for glob in glob_list:
        try:
            if PurePath(path).match(glob):
                return True
        except TypeError:
            pass
    return False

def url_to_image(url, flag=cv2.IMREAD_COLOR):
    """ download the image, convert it to a NumPy array, and then read
    it into OpenCV format """
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, flag)
    return image

def write_tsv_line_from_list(linelist, outfp):
    """Utility method to convert list to tsv line with carriage return"""
    line = '\t'.join(linelist)
    outfp.write(line)
    outfp.write('\n')

def make_coord_dict(coord):
    """helper function to make a dict from a coordinate for logging"""
    return dict(
        z=int_if_exact(coord.zoom),
        x=int_if_exact(coord.column),
        y=int_if_exact(coord.row),
    )

def __absolute__(self, uri):
        """ Get the absolute uri for a file

        :param uri: URI of the resource to be retrieved
        :return: Absolute Path
        """
        return op.abspath(op.join(self.__path__, uri))

def datetime_from_timestamp(timestamp, content):
    """
    Helper function to add timezone information to datetime,
    so that datetime is comparable to other datetime objects in recent versions
    that now also have timezone information.
    """
    return set_date_tzinfo(
        datetime.fromtimestamp(timestamp),
        tz_name=content.settings.get('TIMEZONE', None))

def dimensions(path):
    """Get width and height of a PDF"""
    pdf = PdfFileReader(path)
    size = pdf.getPage(0).mediaBox
    return {'w': float(size[2]), 'h': float(size[3])}

def _intermediary_to_dot(tables, relationships):
    """ Returns the dot source representing the database in a string. """
    t = '\n'.join(t.to_dot() for t in tables)
    r = '\n'.join(r.to_dot() for r in relationships)
    return '{}\n{}\n{}\n}}'.format(GRAPH_BEGINNING, t, r)

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

def _check_valid(key, val, valid):
    """Helper to check valid options"""
    if val not in valid:
        raise ValueError('%s must be one of %s, not "%s"'
                         % (key, valid, val))

def apply_caching(response):
    """Applies the configuration's http headers to all responses"""
    for k, v in config.get('HTTP_HEADERS').items():
        response.headers[k] = v
    return response

def is_equal_strings_ignore_case(first, second):
    """The function compares strings ignoring case"""
    if first and second:
        return first.upper() == second.upper()
    else:
        return not (first or second)

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

def find_le(a, x):
    """Find rightmost value less than or equal to x."""
    i = bs.bisect_right(a, x)
    if i: return i - 1
    raise ValueError

def rlognormal(mu, tau, size=None):
    """
    Return random lognormal variates.
    """

    return np.random.lognormal(mu, np.sqrt(1. / tau), size)

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

def dedupe_list(seq):
    """
    Utility function to remove duplicates from a list
    :param seq: The sequence (list) to deduplicate
    :return: A list with original duplicates removed
    """
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]

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

def list(self):
        """position in 3d space"""
        return [self._pos3d.x, self._pos3d.y, self._pos3d.z]

def _truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Truncates a colormap to use.
    Code originall from http://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    """
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(numpy.linspace(minval, maxval, n))
    )
    return new_cmap

def to_tree(self):
        """ returns a TreeLib tree """
        tree = TreeLibTree()
        for node in self:
            tree.create_node(node, node.node_id, parent=node.parent)
        return tree

def get_size(self, m):
        """
        Return the 2-D size of a Jacobian matrix in tuple
        """
        nrow, ncol = 0, 0
        if m[0] == 'F':
            nrow = self.n
        elif m[0] == 'G':
            nrow = self.m

        if m[1] == 'x':
            ncol = self.n
        elif m[1] == 'y':
            ncol = self.m

        return nrow, ncol

def load_yaml_file(file_path: str):
    """Load a YAML file from path"""
    with codecs.open(file_path, 'r') as f:
        return yaml.safe_load(f)

def scroll_up(self, locator):
        """Scrolls up to element"""
        driver = self._current_application()
        element = self._element_find(locator, True, True)
        driver.execute_script("mobile: scroll", {"direction": 'up', 'element': element.id})

def clean_url(url):
        """URL Validation function"""
        if not url.startswith(('http://', 'https://')):
            url = f'http://{url}'

        if not URL_RE.match(url):
            raise BadURLException(f'{url} is not valid')

        return url

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

def _validate(data, schema, ac_schema_safe=True, **options):
    """
    See the descritpion of :func:`validate` for more details of parameters and
    return value.

    Validate target object 'data' with given schema object.
    """
    try:
        jsonschema.validate(data, schema, **options)

    except (jsonschema.ValidationError, jsonschema.SchemaError,
            Exception) as exc:
        if ac_schema_safe:
            return (False, str(exc))  # Validation was failed.
        raise

    return (True, '')

def translation(language):
    """
    Return a translation object in the default 'django' domain.
    """
    global _translations
    if language not in _translations:
        _translations[language] = Translations(language)
    return _translations[language]

def print_item_with_children(ac, classes, level):
    """ Print the given item and all children items """
    print_row(ac.id, ac.name, f"{ac.allocation:,.2f}", level)
    print_children_recursively(classes, ac, level + 1)

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

def __to_localdatetime(val):
    """Convert val into a local datetime for tz Europe/Amsterdam."""
    try:
        dt = datetime.strptime(val, __DATE_FORMAT)
        dt = pytz.timezone(__TIMEZONE).localize(dt)
        return dt
    except (ValueError, TypeError):
        return None

def inline_inputs(self):
        """Inline all input latex files references by this document. The
        inlining is accomplished recursively. The document is modified
        in place.
        """
        self.text = texutils.inline(self.text,
                                    os.path.dirname(self._filepath))
        # Remove children
        self._children = {}

def safe_quotes(text, escape_single_quotes=False):
    """htmlify string"""
    if isinstance(text, str):
        safe_text = text.replace('"', "&quot;")
        if escape_single_quotes:
            safe_text = safe_text.replace("'", "&#92;'")
        return safe_text.replace('True', 'true')
    return text

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

def is_int_vector(l):
    r"""Checks if l is a numpy array of integers

    """
    if isinstance(l, np.ndarray):
        if l.ndim == 1 and (l.dtype.kind == 'i' or l.dtype.kind == 'u'):
            return True
    return False

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

def abs_img(img):
    """ Return an image with the binarised version of the data of `img`."""
    bool_img = np.abs(read_img(img).get_data())
    return bool_img.astype(int)

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

def vars_(self):
        """ Returns symbol instances corresponding to variables
        of the current scope.
        """
        return [x for x in self[self.current_scope].values() if x.class_ == CLASS.var]

def _Members(self, group):
    """Unify members of a group and accounts with the group as primary gid."""
    group.members = set(group.members).union(self.gids.get(group.gid, []))
    return group

def __init__(self, usb):
    """Constructs a FastbootCommands instance.

    Arguments:
      usb: UsbHandle instance.
    """
    self._usb = usb
    self._protocol = self.protocol_handler(usb)

def get_distance_matrix(x):
    """Get distance matrix given a matrix. Used in testing."""
    square = nd.sum(x ** 2.0, axis=1, keepdims=True)
    distance_square = square + square.transpose() - (2.0 * nd.dot(x, x.transpose()))
    return nd.sqrt(distance_square)

def exit(self):
        """
        Closes the connection
        """
        self.pubsub.unsubscribe()
        self.client.connection_pool.disconnect()

        logger.info("Connection to Redis closed")

def is_gzipped_fastq(file_name):
    """
    Determine whether indicated file appears to be a gzipped FASTQ.

    :param str file_name: Name/path of file to check as gzipped FASTQ.
    :return bool: Whether indicated file appears to be in gzipped FASTQ format.
    """
    _, ext = os.path.splitext(file_name)
    return file_name.endswith(".fastq.gz") or file_name.endswith(".fq.gz")

def stackplot(marray, seconds=None, start_time=None, ylabels=None):
    """
    will plot a stack of traces one above the other assuming
    marray.shape = numRows, numSamples
    """
    tarray = np.transpose(marray)
    stackplot_t(tarray, seconds=seconds, start_time=start_time, ylabels=ylabels)
    plt.show()

def get_builder_toplevel(self, builder):
        """Get the toplevel widget from a gtk.Builder file.

        The main view implementation first searches for the widget named as
        self.toplevel_name (which defaults to "main". If this is missing, or not
        a gtk.Window, the first toplevel window found in the gtk.Builder is
        used.
        """
        toplevel = builder.get_object(self.toplevel_name)
        if not gobject.type_is_a(toplevel, gtk.Window):
            toplevel = None
        if toplevel is None:
            toplevel = get_first_builder_window(builder)
        return toplevel

def coords_from_query(query):
    """Transform a query line into a (lng, lat) pair of coordinates."""
    try:
        coords = json.loads(query)
    except ValueError:
        vals = re.split(r'[,\s]+', query.strip())
        coords = [float(v) for v in vals]
    return tuple(coords[:2])

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

def sort_key(x):
    """
    >>> sort_key(('name', ('ROUTE', 'URL')))
    -3
    """
    name, (r, u) = x
    return - len(u) + u.count('}') * 100

def managepy(cmd, extra=None):
    """Run manage.py using this component's specific Django settings"""

    extra = extra.split() if extra else []
    run_django_cli(['invoke', cmd] + extra)

def dot(a, b):
    """Take arrays `a` and `b` and form the dot product between the last axis
    of `a` and the first of `b`.
    """
    b = numpy.asarray(b)
    return numpy.dot(a, b.reshape(b.shape[0], -1)).reshape(a.shape[:-1] + b.shape[1:])

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

def render_template(template_name, **context):
    """Render a template into a response."""
    tmpl = jinja_env.get_template(template_name)
    context["url_for"] = url_for
    return Response(tmpl.render(context), mimetype="text/html")

def _run_cmd_get_output(cmd):
    """Runs a shell command, returns console output.

    Mimics python3's subprocess.getoutput
    """
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    out, err = process.communicate()
    return out or err

def underscore(text):
    """Converts text that may be camelcased into an underscored format"""
    return UNDERSCORE[1].sub(r'\1_\2', UNDERSCORE[0].sub(r'\1_\2', text)).lower()

def OnDoubleClick(self, event):
        """Double click on a given square in the map"""
        node = HotMapNavigator.findNodeAtPosition(self.hot_map, event.GetPosition())
        if node:
            wx.PostEvent( self, SquareActivationEvent( node=node, point=event.GetPosition(), map=self ) )

def test(nose_argsuments):
    """ Run application tests """
    from nose import run

    params = ['__main__', '-c', 'nose.ini']
    params.extend(nose_argsuments)
    run(argv=params)

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

def replacing_symlink(source, link_name):
    """Create symlink that overwrites any existing target.
    """
    with make_tmp_name(link_name) as tmp_link_name:
        os.symlink(source, tmp_link_name)
        replace_file_or_dir(link_name, tmp_link_name)

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

def getScriptLocation():
	"""Helper function to get the location of a Python file."""
	location = os.path.abspath("./")
	if __file__.rfind("/") != -1:
		location = __file__[:__file__.rfind("/")]
	return location

def make_bintree(levels):
    """Make a symmetrical binary tree with @levels"""
    G = nx.DiGraph()
    root = '0'
    G.add_node(root)
    add_children(G, root, levels, 2)
    return G

def is_complex(dtype):
  """Returns whether this is a complex floating point type."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'is_complex'):
    return dtype.is_complex
  return np.issubdtype(np.dtype(dtype), np.complex)

def to_list(self):
        """Convert this confusion matrix into a 2x2 plain list of values."""
        return [[int(self.table.cell_values[0][1]), int(self.table.cell_values[0][2])],
                [int(self.table.cell_values[1][1]), int(self.table.cell_values[1][2])]]

def _hue(color, **kwargs):
    """ Get hue value of HSL color.
    """
    h = colorsys.rgb_to_hls(*[x / 255.0 for x in color.value[:3]])[0]
    return NumberValue(h * 360.0)

def finished(self):
        """
        Must be called to print final progress label.
        """
        self.progress_bar.set_state(ProgressBar.STATE_DONE)
        self.progress_bar.show()

def clean(self):
        """Return a copy of this Text instance with invalid characters removed."""
        return Text(self.__text_cleaner.clean(self[TEXT]), **self.__kwargs)

def parse(text, showToc=True):
	"""Returns HTML from MediaWiki markup"""
	p = Parser(show_toc=showToc)
	return p.parse(text)

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

def resize(self, size):
        """Return a new Image instance with the given size."""
        return Image(self.pil_image.resize(size, PIL.Image.ANTIALIAS))

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

def exp_fit_fun(x, a, tau, c):
    """Function used to fit the exponential decay."""
    # pylint: disable=invalid-name
    return a * np.exp(-x / tau) + c

def _to_java_object_rdd(rdd):
    """ Return an JavaRDD of Object by unpickling

    It will convert each Python object into Java object by Pyrolite, whenever the
    RDD is serialized in batch or not.
    """
    rdd = rdd._reserialize(AutoBatchedSerializer(PickleSerializer()))
    return rdd.ctx._jvm.org.apache.spark.ml.python.MLSerDe.pythonToJava(rdd._jrdd, True)

def getCursor(self):
		"""
		Get a Dictionary Cursor for executing queries
		"""
		if self.connection is None:
			self.Connect()
			
		return self.connection.cursor(MySQLdb.cursors.DictCursor)

def from_json_list(cls, api_client, data):
        """Convert a list of JSON values to a list of models
        """
        return [cls.from_json(api_client, item) for item in data]

def _reshuffle(mat, shape):
    """Reshuffle the indicies of a bipartite matrix A[ij,kl] -> A[lj,ki]."""
    return np.reshape(
        np.transpose(np.reshape(mat, shape), (3, 1, 2, 0)),
        (shape[3] * shape[1], shape[0] * shape[2]))

def size(self):
        """Return the viewable size of the Table as @tuple (x,y)"""
        width = max(
            map(lambda x: x.size()[0], self.sections.itervalues()))

        height = sum(
            map(lambda x: x.size()[1], self.sections.itervalues()))

        return width, height

def _string_hash(s):
    """String hash (djb2) with consistency between py2/py3 and persistency between runs (unlike `hash`)."""
    h = 5381
    for c in s:
        h = h * 33 + ord(c)
    return h

def plot3d_init(fignum):
    """
    initializes 3D plot
    """
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(fignum)
    ax = fig.add_subplot(111, projection='3d')
    return ax

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

def removeFromRegistery(obj) :
	"""Removes an object/rabalist from registery. This is useful if you want to allow the garbage collector to free the memory
	taken by the objects you've already loaded. Be careful might cause some discrepenties in your scripts. For objects,
	cascades to free the registeries of related rabalists also"""
	
	if isRabaObject(obj) :
		_unregisterRabaObjectInstance(obj)
	elif isRabaList(obj) :
		_unregisterRabaListInstance(obj)

def set_mem_per_proc(self, mem_mb):
        """Set the memory per process in megabytes"""
        super().set_mem_per_proc(mem_mb)
        self.qparams["mem_per_cpu"] = self.mem_per_proc

def on_source_directory_chooser_clicked(self):
        """Autoconnect slot activated when tbSourceDir is clicked."""

        title = self.tr('Set the source directory for script and scenario')
        self.choose_directory(self.source_directory, title)

def set_proxy(proxy_url, transport_proxy=None):
    """Create the proxy to PyPI XML-RPC Server"""
    global proxy, PYPI_URL
    PYPI_URL = proxy_url
    proxy = xmlrpc.ServerProxy(
        proxy_url,
        transport=RequestsTransport(proxy_url.startswith('https://')),
        allow_none=True)

def __add__(self, other):
        """Merges two with identical columns."""

        new_table = copy.copy(self)
        for row in other:
            new_table.Append(row)

        return new_table

def _zeep_to_dict(cls, obj):
        """Convert a zeep object to a dictionary."""
        res = serialize_object(obj)
        res = cls._get_non_empty_dict(res)
        return res

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

def _ndarray_representer(dumper, data):
    """

    :param dumper:
    :param data:
    :type data: :class:`numpy.ndarray`
    :return:
    """
    mapping = [('object', data.tolist()), ('dtype', data.dtype.name)]
    return dumper.represent_mapping(_NUMPY_ARRAY_TAG, mapping)

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

def get_idx_rect(index_list):
    """Extract the boundaries from a list of indexes"""
    rows, cols = list(zip(*[(i.row(), i.column()) for i in index_list]))
    return ( min(rows), max(rows), min(cols), max(cols) )

def in_transaction(self):
        """Check if this database is in a transactional context."""
        if not hasattr(self.local, 'tx'):
            return False
        return len(self.local.tx) > 0

def to_json(value, **kwargs):
        """Return a copy of the tuple as a list

        If the tuple contains HasProperties instances, they are serialized.
        """
        serial_list = [
            val.serialize(**kwargs) if isinstance(val, HasProperties)
            else val for val in value
        ]
        return serial_list

def nonull_dict(self):
        """Like dict, but does not hold any null values.

        :return:

        """
        return {k: v for k, v in six.iteritems(self.dict) if v and k != '_codes'}

def generate(env):
    """Add Builders and construction variables for SGI MIPS C++ to an Environment."""

    cplusplus.generate(env)

    env['CXX']         = 'CC'
    env['CXXFLAGS']    = SCons.Util.CLVar('-LANG:std')
    env['SHCXX']       = '$CXX'
    env['SHOBJSUFFIX'] = '.o'
    env['STATIC_AND_SHARED_OBJECTS_ARE_THE_SAME'] = 1

def _clip(sid, prefix):
    """Clips a prefix from the beginning of a string if it exists."""
    return sid[len(prefix):] if sid.startswith(prefix) else sid

def _select_features(example, feature_list=None):
  """Select a subset of features from the example dict."""
  feature_list = feature_list or ["inputs", "targets"]
  return {f: example[f] for f in feature_list}