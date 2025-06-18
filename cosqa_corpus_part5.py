def percent_cb(name, complete, total):
    """ Callback for updating target progress """
    logger.debug(
        "{}: {} transferred out of {}".format(
            name, sizeof_fmt(complete), sizeof_fmt(total)
        )
    )
    progress.update_target(name, complete, total)

def unfolding(tens, i):
    """Compute the i-th unfolding of a tensor."""
    return reshape(tens.full(), (np.prod(tens.n[0:(i+1)]), -1))

def stop(self, timeout=None):
        """ Initiates a graceful stop of the processes """

        self.stopping = True

        for process in list(self.processes):
            self.stop_process(process, timeout=timeout)

def check_auth(email, password):
    """Check if a username/password combination is valid.
    """
    try:
        user = User.get(User.email == email)
    except User.DoesNotExist:
        return False
    return password == user.password

def natural_sort(list, key=lambda s:s):
    """
    Sort the list into natural alphanumeric order.
    """
    def get_alphanum_key_func(key):
        convert = lambda text: int(text) if text.isdigit() else text
        return lambda s: [convert(c) for c in re.split('([0-9]+)', key(s))]
    sort_key = get_alphanum_key_func(key)
    list.sort(key=sort_key)

def parse_form(self, req, name, field):
        """Pull a form value from the request."""
        return get_value(req.body_arguments, name, field)

def __iter__(self):
        """Define a generator function and return it"""
        def generator():
            for i, obj in enumerate(self._sequence):
                if i >= self._limit:
                    break
                yield obj
            raise StopIteration
        return generator

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

def should_skip_logging(func):
    """
    Should we skip logging for this handler?

    """
    disabled = strtobool(request.headers.get("x-request-nolog", "false"))
    return disabled or getattr(func, SKIP_LOGGING, False)

def osx_clipboard_get():
    """ Get the clipboard's text on OS X.
    """
    p = subprocess.Popen(['pbpaste', '-Prefer', 'ascii'],
        stdout=subprocess.PIPE)
    text, stderr = p.communicate()
    # Text comes in with old Mac \r line endings. Change them to \n.
    text = text.replace('\r', '\n')
    return text

def batch_split_sentences(self, texts: List[str]) -> List[List[str]]:
        """
        This method lets you take advantage of spacy's batch processing.
        Default implementation is to just iterate over the texts and call ``split_sentences``.
        """
        return [self.split_sentences(text) for text in texts]

def to_dataframe(products):
        """Return the products from a query response as a Pandas DataFrame
        with the values in their appropriate Python types.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("to_dataframe requires the optional dependency Pandas.")

        return pd.DataFrame.from_dict(products, orient='index')

def save_excel(self, fd):
        """ Saves the case as an Excel spreadsheet.
        """
        from pylon.io.excel import ExcelWriter
        ExcelWriter(self).write(fd)

def median(self):
        """Computes the median of a log-normal distribution built with the stats data."""
        mu = self.mean()
        ret_val = math.exp(mu)
        if math.isnan(ret_val):
            ret_val = float("inf")
        return ret_val

def draw_tree(t, df, size=10, ratio=0.6, precision=0):
    """ Draws a representation of a random forest in IPython.
    Parameters:
    -----------
    t: The tree you wish to draw
    df: The data used to train the tree. This is used to get the names of the features.
    """
    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True,
                      special_characters=True, rotate=True, precision=precision)
    IPython.display.display(graphviz.Source(re.sub('Tree {',
       f'Tree {{ size={size}; ratio={ratio}', s)))

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

def shutdown(self):
        """close socket, immediately."""
        if self.sock:
            self.sock.close()
            self.sock = None
            self.connected = False

def ln_norm(x, mu, sigma=1.0):
    """ Natural log of scipy norm function truncated at zero """
    return np.log(stats.norm(loc=mu, scale=sigma).pdf(x))

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

def figsize(x=8, y=7., aspect=1.):
    """ manually set the default figure size of plots
    ::Arguments::
        x (float): x-axis size
        y (float): y-axis size
        aspect (float): aspect ratio scalar
    """
    # update rcparams with adjusted figsize params
    mpl.rcParams.update({'figure.figsize': (x*aspect, y)})

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

def get_decimal_quantum(precision):
    """Return minimal quantum of a number, as defined by precision."""
    assert isinstance(precision, (int, decimal.Decimal))
    return decimal.Decimal(10) ** (-precision)

def set_header(self, name, value):
        """ Create a new response header, replacing any previously defined
            headers with the same name. """
        self._headers[_hkey(name)] = [_hval(value)]

def basic_word_sim(word1, word2):
    """
    Simple measure of similarity: Number of letters in common / max length
    """
    return sum([1 for c in word1 if c in word2]) / max(len(word1), len(word2))

def _set_widget_background_color(widget, color):
        """
        Changes the base color of a widget (background).
        :param widget: widget to modify
        :param color: the color to apply
        """
        pal = widget.palette()
        pal.setColor(pal.Base, color)
        widget.setPalette(pal)

def iterate_chunks(file, chunk_size):
    """
    Iterate chunks of size chunk_size from a file-like object
    """
    chunk = file.read(chunk_size)
    while chunk:
        yield chunk
        chunk = file.read(chunk_size)

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def read_from_file(file_path, encoding="utf-8"):
    """
    Read helper method

    :type file_path: str|unicode
    :type encoding: str|unicode
    :rtype: str|unicode
    """
    with codecs.open(file_path, "r", encoding) as f:
        return f.read()

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

def from_rotation_vector(rot):
    """Convert input 3-vector in axis-angle representation to unit quaternion

    Parameters
    ----------
    rot: (Nx3) float array
        Each vector represents the axis of the rotation, with norm
        proportional to the angle of the rotation in radians.

    Returns
    -------
    q: array of quaternions
        Unit quaternions resulting in rotations corresponding to input
        rotations.  Output shape is rot.shape[:-1].

    """
    rot = np.array(rot, copy=False)
    quats = np.zeros(rot.shape[:-1]+(4,))
    quats[..., 1:] = rot[...]/2
    quats = as_quat_array(quats)
    return np.exp(quats)

def wipe(self):
        """ Wipe the store
        """
        query = "DELETE FROM {}".format(self.__tablename__)
        connection = sqlite3.connect(self.sqlite_file)
        cursor = connection.cursor()
        cursor.execute(query)
        connection.commit()

def print_matrix(X, decimals=1):
    """Pretty printing for numpy matrix X"""
    for row in np.round(X, decimals=decimals):
        print(row)

def restore_image_options(cli, image, options):
    """ Restores CMD and ENTRYPOINT values of the image

    This is needed because we force the overwrite of ENTRYPOINT and CMD in the
    `run_code_in_container` function, to be able to run the code in the
    container, through /bin/bash.
    """
    dockerfile = io.StringIO()

    dockerfile.write(u'FROM {image}\nCMD {cmd}'.format(
        image=image, cmd=json.dumps(options['cmd'])))

    if options['entrypoint']:
        dockerfile.write(
            '\nENTRYPOINT {}'.format(json.dumps(options['entrypoint'])))

    cli.build(tag=image, fileobj=dockerfile)

def get_size(path):
    """ Returns the size in bytes if `path` is a file,
        or the size of all files in `path` if it's a directory.
        Analogous to `du -s`.
    """
    if os.path.isfile(path):
        return os.path.getsize(path)
    return sum(get_size(os.path.join(path, f)) for f in os.listdir(path))

def LogBinomialCoef(n, k):
    """Computes the log of the binomial coefficient.

    http://math.stackexchange.com/questions/64716/
    approximating-the-logarithm-of-the-binomial-coefficient

    n: number of trials
    k: number of successes

    Returns: float
    """
    return n * log(n) - k * log(k) - (n - k) * log(n - k)

def load_yaml(filepath):
    """Convenience function for loading yaml-encoded data from disk."""
    with open(filepath) as f:
        txt = f.read()
    return yaml.load(txt)

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

def prevPlot(self):
        """Moves the displayed plot to the previous one"""
        if self.stacker.currentIndex() > 0:
            self.stacker.setCurrentIndex(self.stacker.currentIndex()-1)

def _rindex(mylist: Sequence[T], x: T) -> int:
    """Index of the last occurrence of x in the sequence."""
    return len(mylist) - mylist[::-1].index(x) - 1

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

def __init__(self, filename, mode, encoding=None):
        """Use the specified filename for streamed logging."""
        FileHandler.__init__(self, filename, mode, encoding)
        self.mode = mode
        self.encoding = encoding

def rewrap(s, width=COLS):
    """ Join all lines from input string and wrap it at specified width """
    s = ' '.join([l.strip() for l in s.strip().split('\n')])
    return '\n'.join(textwrap.wrap(s, width))

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

def not_matching_list(self):
        """
        Return a list of string which don't match the
        given regex.
        """

        pre_result = comp(self.regex)

        return [x for x in self.data if not pre_result.search(str(x))]

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

def __gt__(self, other):
        """Test for greater than."""
        if isinstance(other, Address):
            return str(self) > str(other)
        raise TypeError

def launched():
    """Test whether the current python environment is the correct lore env.

    :return:  :any:`True` if the environment is launched
    :rtype: bool
    """
    if not PREFIX:
        return False

    return os.path.realpath(sys.prefix) == os.path.realpath(PREFIX)

def find_all(self, string, callback):
		"""
		Wrapper on iter method, callback gets an iterator result
		"""
		for index, output in self.iter(string):
			callback(index, output)

def get_python():
    """Determine the path to the virtualenv python"""
    if sys.platform == 'win32':
        python = path.join(VE_ROOT, 'Scripts', 'python.exe')
    else:
        python = path.join(VE_ROOT, 'bin', 'python')
    return python

def mpl_outside_legend(ax, **kwargs):
    """ Places a legend box outside a matplotlib Axes instance. """
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), **kwargs)

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

def __iand__(self, other):
        """Intersect this flag with ``other`` in-place.
        """
        self.known &= other.known
        self.active &= other.active
        return self

def is_bool_matrix(l):
    r"""Checks if l is a 2D numpy array of bools

    """
    if isinstance(l, np.ndarray):
        if l.ndim == 2 and (l.dtype == bool):
            return True
    return False

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

def init_checks_registry():
    """Register all globally visible functions.

    The first argument name is either 'physical_line' or 'logical_line'.
    """
    mod = inspect.getmodule(register_check)
    for (name, function) in inspect.getmembers(mod, inspect.isfunction):
        register_check(function)

def is_real_floating_dtype(dtype):
    """Return ``True`` if ``dtype`` is a real floating point type."""
    dtype = np.dtype(dtype)
    return np.issubsctype(getattr(dtype, 'base', None), np.floating)

def register_type(cls, name):
    """Register `name` as a type to validate as an instance of class `cls`."""
    x = TypeDefinition(name, (cls,), ())
    Validator.types_mapping[name] = x

def _write_json(obj, path):  # type: (object, str) -> None
    """Writes a serializeable object as a JSON file"""
    with open(path, 'w') as f:
        json.dump(obj, f)

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

def main(pargs):
    """This should only be used for testing. The primary mode of operation is
    as an imported library.
    """
    input_file = sys.argv[1]
    fp = ParseFileLineByLine(input_file)
    for i in fp:
        print(i)

def _renamer(self, tre):
        """ renames newick from numbers to sample names"""
        ## get the tre with numbered tree tip labels
        names = tre.get_leaves()

        ## replace numbered names with snames
        for name in names:
            name.name = self.samples[int(name.name)]

        ## return with only topology and leaf labels
        return tre.write(format=9)

def translate_v3(vec, amount):
    """Return a new Vec3 that is translated version of vec."""

    return Vec3(vec.x+amount, vec.y+amount, vec.z+amount)

def __enter__(self):
        """Acquire a lock on the output file, prevents collisions between multiple runs."""
        self.fd = open(self.filename, 'a')
        fcntl.lockf(self.fd, fcntl.LOCK_EX)
        return self.fd

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

def get_power(self):
        """Check if the device is on."""
        power = (yield from self.handle_int(self.API.get('power')))
        return bool(power)

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

def get_cache(self, decorated_function, *args, **kwargs):
		""" :meth:`WCacheStorage.get_cache` method implementation
		"""
		has_value = self.has(decorated_function, *args, **kwargs)
		cached_value = None
		if has_value is True:
			cached_value = self.get_result(decorated_function, *args, **kwargs)
		return WCacheStorage.CacheEntry(has_value=has_value, cached_value=cached_value)

def purge_dict(idict):
    """Remove null items from a dictionary """
    odict = {}
    for key, val in idict.items():
        if is_null(val):
            continue
        odict[key] = val
    return odict

def open(self, flag="c"):
        """Open handle

        set protocol=2 to fix python3

        .. versionadded:: 1.3.1
        """
        return shelve.open(os.path.join(gettempdir(), self.index), flag=flag, protocol=2)

def generate_device_id(steamid):
    """Generate Android device id

    :param steamid: Steam ID
    :type steamid: :class:`.SteamID`, :class:`int`
    :return: android device id
    :rtype: str
    """
    h = hexlify(sha1_hash(str(steamid).encode('ascii'))).decode('ascii')
    return "android:%s-%s-%s-%s-%s" % (h[:8], h[8:12], h[12:16], h[16:20], h[20:32])

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

def createArgumentParser(description):
    """
    Create an argument parser
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=SortedHelpFormatter)
    return parser

def get_screen_resolution(self):
        """Return the screen resolution of the primary screen."""
        widget = QDesktopWidget()
        geometry = widget.availableGeometry(widget.primaryScreen())
        return geometry.width(), geometry.height()

def _idx_col2rowm(d):
    """Generate indexes to change from col-major to row-major ordering"""
    if 0 == len(d):
        return 1
    if 1 == len(d):
        return np.arange(d[0])
    # order='F' indicates column-major ordering
    idx = np.array(np.arange(np.prod(d))).reshape(d, order='F').T
    return idx.flatten(order='F')

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

def lint(args):
    """Run lint checks using flake8."""
    application = get_current_application()
    if not args:
        args = [application.name, 'tests']
    args = ['flake8'] + list(args)
    run.main(args, standalone_mode=False)

def save_as_png(self, filename, width=300, height=250, render_time=1):
        """Open saved html file in an virtual browser and save a screen shot to PNG format."""
        self.driver.set_window_size(width, height)
        self.driver.get('file://{path}/{filename}'.format(
            path=os.getcwd(), filename=filename + ".html"))
        time.sleep(render_time)
        self.driver.save_screenshot(filename + ".png")

def url(self):
        """ The url of this window """
        with switch_window(self._browser, self.name):
            return self._browser.url

def js_classnameify(s):
  """
  Makes a classname.
  """
  if not '_' in s:
    return s
  return ''.join(w[0].upper() + w[1:].lower() for w in s.split('_'))

def on_IOError(self, e):
        """ Handle an IOError exception. """

        sys.stderr.write("Error: %s: \"%s\"\n" % (e.strerror, e.filename))

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

def executemany(self, sql, *params):
        """Prepare a database query or command and then execute it against
        all parameter sequences  found in the sequence seq_of_params.

        :param sql: the SQL statement to execute with optional ? parameters
        :param params: sequence parameters for the markers in the SQL.
        """
        fut = self._run_operation(self._impl.executemany, sql, *params)
        return fut

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

def tofile(self, fileobj):
		"""
		write a cache object to the fileobj as a lal cache file
		"""
		for entry in self:
			print >>fileobj, str(entry)
		fileobj.close()

def prepend_line(filepath, line):
    """Rewrite a file adding a line to its beginning.
    """
    with open(filepath) as f:
        lines = f.readlines()

    lines.insert(0, line)

    with open(filepath, 'w') as f:
        f.writelines(lines)

def ffmpeg_works():
  """Tries to encode images with ffmpeg to check if it works."""
  images = np.zeros((2, 32, 32, 3), dtype=np.uint8)
  try:
    _encode_gif(images, 2)
    return True
  except (IOError, OSError):
    return False

def _handle_chat_name(self, data):
        """Handle user name changes"""

        self.room.user.nick = data
        self.conn.enqueue_data("user", self.room.user)

def replace_keys(record: Mapping, key_map: Mapping) -> dict:
    """New record with renamed keys including keys only found in key_map."""

    return {key_map[k]: v for k, v in record.items() if k in key_map}

def _get_parsing_plan_for_multifile_children(self, obj_on_fs: PersistedObject, desired_type: Type[Any],
                                                 logger: Logger) -> Dict[str, Any]:
        """
        Implementation of AnyParser API
        """
        raise Exception('This should never happen, since this parser relies on underlying parsers')

def __init__(self):
    """Initializes an attribute container identifier."""
    super(AttributeContainerIdentifier, self).__init__()
    self._identifier = id(self)

def chunk_list(l, n):
    """Return `n` size lists from a given list `l`"""
    return [l[i:i + n] for i in range(0, len(l), n)]

def setHSV(self, pixel, hsv):
        """Set single pixel to HSV tuple"""
        color = conversions.hsv2rgb(hsv)
        self._set_base(pixel, color)

def _set_property(self, val, *args):
        """Private method that sets the value currently of the property"""
        val = UserClassAdapter._set_property(self, val, *args)
        if val:
            Adapter._set_property(self, val, *args)
        return val

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

def get_day_name(self) -> str:
        """ Returns the day name """
        weekday = self.value.isoweekday() - 1
        return calendar.day_name[weekday]

def _collection_literal_to_py_ast(
    ctx: GeneratorContext, form: Iterable[LispForm]
) -> Iterable[GeneratedPyAST]:
    """Turn a quoted collection literal of Lisp forms into Python AST nodes.

    This function can only handle constant values. It does not call back into
    the generic AST generators, so only constant values will be generated down
    this path."""
    yield from map(partial(_const_val_to_py_ast, ctx), form)

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

def is_iterable(etype) -> bool:
    """ Determine whether etype is a List or other iterable """
    return type(etype) is GenericMeta and issubclass(etype.__extra__, Iterable)

def datetime_to_timezone(date, tz="UTC"):
    """ convert naive datetime to timezone-aware datetime """
    if not date.tzinfo:
        date = date.replace(tzinfo=timezone(get_timezone()))
    return date.astimezone(timezone(tz))

def chunked(l, n):
    """Chunk one big list into few small lists."""
    return [l[i:i + n] for i in range(0, len(l), n)]

def dict_to_numpy_array(d):
    """
    Convert a dict of 1d array to a numpy recarray
    """
    return fromarrays(d.values(), np.dtype([(str(k), v.dtype) for k, v in d.items()]))

def generate_id():
    """Generate new UUID"""
    # TODO: Use six.string_type to Py3 compat
    try:
        return unicode(uuid1()).replace(u"-", u"")
    except NameError:
        return str(uuid1()).replace(u"-", u"")

def snake_to_camel(s: str) -> str:
    """Convert string from snake case to camel case."""

    fragments = s.split('_')

    return fragments[0] + ''.join(x.title() for x in fragments[1:])

def array_size(x, axis):
  """Calculate the size of `x` along `axis` dimensions only."""
  axis_shape = x.shape if axis is None else tuple(x.shape[a] for a in axis)
  return max(numpy.prod(axis_shape), 1)

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def path(self):
        """Return the project path (aka project root)

        If ``package.__file__`` is ``/foo/foo/__init__.py``, then project.path
        should be ``/foo``.
        """
        return pathlib.Path(self.package.__file__).resolve().parent.parent

def get_file_name(url):
  """Returns file name of file at given url."""
  return os.path.basename(urllib.parse.urlparse(url).path) or 'unknown_name'

def _enter_plotting(self, fontsize=9):
        """assumes that a figure is open """
        # interactive_status = matplotlib.is_interactive()
        self.original_fontsize = pyplot.rcParams['font.size']
        pyplot.rcParams['font.size'] = fontsize
        pyplot.hold(False)  # opens a figure window, if non exists
        pyplot.ioff()

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

def new_iteration(self, prefix):
        """When inside a loop logger, created a new iteration
        """
        # Flush data for the current iteration
        self.flush()

        # Fix prefix
        self.prefix[-1] = prefix
        self.reset_formatter()

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

def solr_to_date(d):
    """ converts YYYY-MM-DDT00:00:00Z to DD-MM-YYYY """
    return "{day}:{m}:{y}".format(y=d[:4], m=d[5:7], day=d[8:10]) if d else d

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

def _get_ipv4_from_binary(self, bin_addr):
        """Converts binary address to Ipv4 format."""

        return socket.inet_ntop(socket.AF_INET, struct.pack("!L", bin_addr))

def _increment(self, *args):
        """Move the slider only by increment given by resolution."""
        value = self._var.get()
        if self._resolution:
            value = self._start + int(round((value - self._start) / self._resolution)) * self._resolution
            self._var.set(value)
        self.display_value(value)

def set_icon(self, bmp):
        """Sets main window icon to given wx.Bitmap"""

        _icon = wx.EmptyIcon()
        _icon.CopyFromBitmap(bmp)
        self.SetIcon(_icon)

def _check_key(self, key):
        """
        Ensures well-formedness of a key.
        """
        if not len(key) == 2:
            raise TypeError('invalid key: %r' % key)
        elif key[1] not in TYPES:
            raise TypeError('invalid datatype: %s' % key[1])

def validate(self):
        """Validate the configuration file."""
        validator = Draft4Validator(self.SCHEMA)
        if not validator.is_valid(self.config):
            for err in validator.iter_errors(self.config):
                LOGGER.error(str(err.message))
            validator.validate(self.config)

def user_exists(username):
    """Check if a user exists"""
    try:
        pwd.getpwnam(username)
        user_exists = True
    except KeyError:
        user_exists = False
    return user_exists

def range(self, chromosome, start, stop, exact=False):
        """
        Shortcut to do range filters on genomic datasets.
        """
        return self._clone(
            filters=[GenomicFilter(chromosome, start, stop, exact)])

def __init__(self, response):
        """Initialize a ResponseException instance.

        :param response: A requests.response instance.

        """
        self.response = response
        super(ResponseException, self).__init__(
            "received {} HTTP response".format(response.status_code)
        )

def print_runs(query):
    """ Print all rows in this result query. """

    if query is None:
        return

    for tup in query:
        print(("{0} @ {1} - {2} id: {3} group: {4}".format(
            tup.end, tup.experiment_name, tup.project_name,
            tup.experiment_group, tup.run_group)))

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

def acquire_node(self, node):
        """
        acquire a single redis node
        """
        try:
            return node.set(self.resource, self.lock_key, nx=True, px=self.ttl)
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
            return False

def check(self, var):
        """Check whether the provided value is a valid enum constant."""
        if not isinstance(var, _str_type): return False
        return _enum_mangle(var) in self._consts

def static_url(path, absolute=False):
    """ Shorthand for returning a URL for the requested static file.

    Arguments:

    path -- the path to the file (relative to the static files directory)
    absolute -- whether the link should be absolute or relative
    """

    if os.sep != '/':
        path = '/'.join(path.split(os.sep))

    return flask.url_for('static', filename=path, _external=absolute)

def fixed(ctx, number, decimals=2, no_commas=False):
    """
    Formats the given number in decimal format using a period and commas
    """
    value = _round(ctx, number, decimals)
    format_str = '{:f}' if no_commas else '{:,f}'
    return format_str.format(value)

def stop_containers(self):
        """ Stops all containers used by this instance of the backend.
        """
        while len(self._containers):
            container = self._containers.pop()
            try:
                container.kill(signal.SIGKILL)
            except docker.errors.APIError:  # probably doesn't exist anymore
                pass

def last_commit(self) -> Tuple:
        """Returns a tuple (hash, and commit object)"""
        from libs.repos import git

        return git.get_last_commit(repo_path=self.path)

def authenticate(self, transport, account_name, password=None):
        """
        Authenticates account using soap method.
        """
        Authenticator.authenticate(self, transport, account_name, password)

        if password == None:
            return self.pre_auth(transport, account_name)
        else:
            return self.auth(transport, account_name, password)

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

def format_screen(strng):
    """Format a string for screen printing.

    This removes some latex-type format codes."""
    # Paragraph continue
    par_re = re.compile(r'\\$',re.MULTILINE)
    strng = par_re.sub('',strng)
    return strng

def is_up_to_date(outfile, basedatetime):
        # type: (AnyStr, datetime) -> bool
        """Return true if outfile exists and is no older than base datetime."""
        if os.path.exists(outfile):
            if os.path.getmtime(outfile) >= basedatetime:
                return True
        return False

def schunk(string, size):
    """Splits string into n sized chunks."""
    return [string[i:i+size] for i in range(0, len(string), size)]

def after_epoch(self, **_) -> None:
        """Save/override the latest model after every epoch."""
        SaveEvery.save_model(model=self._model, name_suffix=self._OUTPUT_NAME, on_failure=self._on_save_failure)

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

def save_to_16bit_wave_file(fname, sig, rate):
  """
  Save a given signal ``sig`` to file ``fname`` as a 16-bit one-channel wave
  with the given ``rate`` sample rate.
  """
  with closing(wave.open(fname, "wb")) as wave_file:
    wave_file.setnchannels(1)
    wave_file.setsampwidth(2)
    wave_file.setframerate(rate)
    for chunk in chunks((clip(sig) * 2 ** 15).map(int), dfmt="h", padval=0):
      wave_file.writeframes(chunk)

def split_multiline(value):
    """Split a multiline string into a list, excluding blank lines."""
    return [element for element in (line.strip() for line in value.split('\n'))
            if element]

def filter(self, func):
        """Returns a SndRcv list filtered by a truth function"""
        return self.__class__( [ i for i in self.res if func(*i) ], name='filtered %s'%self.listname)

def normalize(x, min_value, max_value):
    """Normalize value between min and max values.
    It also clips the values, so that you cannot have values higher or lower
    than 0 - 1."""
    x = (x - min_value) / (max_value - min_value)
    return clip(x, 0, 1)

def datetime_from_str(string):
    """

    Args:
        string: string of the form YYMMDD-HH_MM_SS, e.g 160930-18_43_01

    Returns: a datetime object

    """


    return datetime.datetime(year=2000+int(string[0:2]), month=int(string[2:4]), day=int(string[4:6]), hour=int(string[7:9]), minute=int(string[10:12]),second=int(string[13:15]))

def simulate(self):
        """Generates a random integer in the available range."""
        min_ = (-sys.maxsize - 1) if self._min is None else self._min
        max_ = sys.maxsize if self._max is None else self._max
        return random.randint(min_, max_)

def config_parser_to_dict(config_parser):
    """
    Convert a ConfigParser to a dictionary.
    """
    response = {}

    for section in config_parser.sections():
        for option in config_parser.options(section):
            response.setdefault(section, {})[option] = config_parser.get(section, option)

    return response

def teardown(self):
        """Cleanup cache tables."""
        for table_spec in reversed(self._table_specs):
            with self._conn:
                table_spec.teardown(self._conn)

def print_verbose(*args, **kwargs):
    """Utility to print something only if verbose=True is given
    """
    if kwargs.pop('verbose', False) is True:
        gprint(*args, **kwargs)

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

def build_output(self, fout):
        """Squash self.out into string.

        Join every line in self.out with a new line and write the
        result to the output file.
        """
        fout.write('\n'.join([s for s in self.out]))

def coverage(ctx, opts=""):
    """
    Execute all tests (normal and slow) with coverage enabled.
    """
    return test(ctx, coverage=True, include_slow=True, opts=opts)

def getSize(self):
        """
        Returns the size of the layer, with the border size already subtracted.
        """
        return self.widget.size[0]-self.border[0]*2,self.widget.size[1]-self.border[1]*2

def dump(self, *args, **kwargs):
        """Dumps a representation of the Model on standard output."""
        lxml.etree.dump(self._obj, *args, **kwargs)

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

def LinSpace(start, stop, num):
    """
    Linspace op.
    """
    return np.linspace(start, stop, num=num, dtype=np.float32),

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

def _correct_args(func, kwargs):
    """
        Convert a dictionary of arguments including __argv into a list
        for passing to the function.
    """
    args = inspect.getargspec(func)[0]
    return [kwargs[arg] for arg in args] + kwargs['__args']

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

def wait_on_rate_limit(self, value):
        """Enable or disable automatic rate-limit handling."""
        check_type(value, bool, may_be_none=False)
        self._wait_on_rate_limit = value

def clone_with_copy(src_path, dest_path):
    """Clone a directory try by copying it.

   Args:
        src_path: The directory to be copied.
        dest_path: The location to copy the directory to.
    """
    log.info('Cloning directory tree %s to %s', src_path, dest_path)
    shutil.copytree(src_path, dest_path)

def plot_kde(data, ax, title=None, color='r', fill_bt=True):
    """
    Plot a smoothed (by kernel density estimate) histogram.
    :type data: numpy array
    :param data: An array containing the data to be plotted

    :type ax: matplotlib.Axes
    :param ax: The Axes object to draw to

    :type title: str
    :param title: The plot title

    :type color: str
    :param color: The color of the histogram line and fill. Note that the fill
                  will be plotted with an alpha of 0.35.

    :type fill_bt: bool
    :param fill_bt: Specify whether to fill the area beneath the histogram line
    """
    if isinstance(data, list):
        data = np.asarray(data)
    e = kde.KDEUnivariate(data.astype(np.float))
    e.fit()
    ax.plot(e.support, e.density, color=color, alpha=0.9, linewidth=2.25)
    if fill_bt:
        ax.fill_between(e.support, e.density, alpha=.35, zorder=1,
                        antialiased=True, color=color)
    if title is not None:
        t = ax.set_title(title)
        t.set_y(1.05)

def xpathEvalExpression(self, str):
        """Evaluate the XPath expression in the given context. """
        ret = libxml2mod.xmlXPathEvalExpression(str, self._o)
        if ret is None:raise xpathError('xmlXPathEvalExpression() failed')
        return xpathObjectRet(ret)

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

def is_alive(self):
        """
        Will test whether the ACS service is up and alive.
        """
        response = self.get_monitoring_heartbeat()
        if response.status_code == 200 and response.content == 'alive':
            return True

        return False

def libpath(self):
        """Returns the full path to the shared *wrapper* library created for the
        module.
        """
        from os import path
        return path.join(self.dirpath, self.libname)

def datetime_iso_format(date):
    """
    Return an ISO-8601 representation of a datetime object.
    """
    return "{0:0>4}-{1:0>2}-{2:0>2}T{3:0>2}:{4:0>2}:{5:0>2}Z".format(
        date.year, date.month, date.day, date.hour,
        date.minute, date.second)

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

def _bindingsToDict(self, bindings):
        """
        Given a binding from the sparql query result,
        create a dict of plain text
        """
        myDict = {}
        for key, val in bindings.iteritems():
            myDict[key.toPython().replace('?', '')] = val.toPython()
        return myDict

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

def token_list_to_text(tokenlist):
    """
    Concatenate all the text parts again.
    """
    ZeroWidthEscape = Token.ZeroWidthEscape
    return ''.join(item[1] for item in tokenlist if item[0] != ZeroWidthEscape)

def validate(self, *args, **kwargs): # pylint: disable=arguments-differ
        """
        Validate a parameter dict against a parameter schema from an ocrd-tool.json

        Args:
            obj (dict):
            schema (dict):
        """
        return super(ParameterValidator, self)._validate(*args, **kwargs)

def normalize_time(timestamp):
    """Normalize time in arbitrary timezone to UTC naive object."""
    offset = timestamp.utcoffset()
    if offset is None:
        return timestamp
    return timestamp.replace(tzinfo=None) - offset

def closest(xarr, val):
    """ Return the index of the closest in xarr to value val """
    idx_closest = np.argmin(np.abs(np.array(xarr) - val))
    return idx_closest

def rex_assert(self, rex, byte=False):
        """
        If `rex` expression is not found then raise `DataNotFound` exception.
        """

        self.rex_search(rex, byte=byte)

def import_js(path, lib_name, globals):
    """Imports from javascript source file.
      globals is your globals()"""
    with codecs.open(path_as_local(path), "r", "utf-8") as f:
        js = f.read()
    e = EvalJs()
    e.execute(js)
    var = e.context['var']
    globals[lib_name] = var.to_python()

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

def from_json(s):
    """Given a JSON-encoded message, build an object.

    """
    d = json.loads(s)
    sbp = SBP.from_json_dict(d)
    return sbp

def clean_markdown(text):
    """
    Parse markdown sintaxt to html.
    """
    result = text

    if isinstance(text, str):
        result = ''.join(
            BeautifulSoup(markdown(text), 'lxml').findAll(text=True))

    return result

def mongoqs_to_json(qs, fields=None):
    """
    transform mongoengine.QuerySet to json
    """

    l = list(qs.as_pymongo())

    for element in l:
        element.pop('_cls')

    # use DjangoJSONEncoder for transform date data type to datetime
    json_qs = json.dumps(l, indent=2, ensure_ascii=False, cls=DjangoJSONEncoder)
    return json_qs

def is_type(value):
        """Determine if value is an instance or subclass of the class Type."""
        if isinstance(value, type):
            return issubclass(value, Type)
        return isinstance(value, Type)

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

def label_saves(name):
    """Labels plots and saves file"""
    plt.legend(loc=0)
    plt.ylim([0, 1.025])
    plt.xlabel('$U/D$', fontsize=20)
    plt.ylabel('$Z$', fontsize=20)
    plt.savefig(name, dpi=300, format='png',
            transparent=False, bbox_inches='tight', pad_inches=0.05)

def handle_m2m(self, sender, instance, **kwargs):
    """ Handle many to many relationships """
    self.handle_save(instance.__class__, instance)

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

def copy(obj):
    def copy(self):
        """
        Copy self to a new object.
        """
        from copy import deepcopy

        return deepcopy(self)
    obj.copy = copy
    return obj

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

def strip_line(line, sep=os.linesep):
    """
    Removes occurrence of character (sep) from a line of text
    """

    try:
        return line.strip(sep)
    except TypeError:
        return line.decode('utf-8').strip(sep)

def getCenter(self):
        """ Return the ``Location`` of the center of this region """
        return Location(self.x+(self.w/2), self.y+(self.h/2))

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

def get_month_start_date(self):
        """Returns the first day of the current month"""
        now = timezone.now()
        return timezone.datetime(day=1, month=now.month, year=now.year, tzinfo=now.tzinfo)

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

def load_yaml(yaml_file: str) -> Any:
    """
    Load YAML from file.

    :param yaml_file: path to YAML file
    :return: content of the YAML as dict/list
    """
    with open(yaml_file, 'r') as file:
        return ruamel.yaml.load(file, ruamel.yaml.RoundTripLoader)

def b2u(string):
    """ bytes to unicode """
    if (isinstance(string, bytes) or
        (PY2 and isinstance(string, str))):
        return string.decode('utf-8')
    return string

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

def memsize(self):
        """ Total array cell + indexes size
        """
        return self.size + 1 + TYPE.size(gl.BOUND_TYPE) * len(self.bounds)

def __call__(self, _):
        """Print the current iteration."""
        if self.iter % self.step == 0:
            print(self.fmt.format(self.iter), **self.kwargs)

        self.iter += 1

def get(self, key):
        """Get a value from the cache.

        Returns None if the key is not in the cache.
        """
        value = redis_conn.get(key)

        if value is not None:
            value = pickle.loads(value)

        return value

def f(x, a, c):
    """ Objective function (sum of squared residuals) """
    v = g(x, a, c)
    return v.dot(v)

def set_context(self, data):
        """Load Context with data"""
        for key in data:
            setattr(self.local_context, key, data[key])

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

def _name_exists(self, name):
        """
        Checks if we already have an opened tab with the same name.
        """
        for i in range(self.count()):
            if self.tabText(i) == name:
                return True
        return False

def log_y_cb(self, w, val):
        """Toggle linear/log scale for Y-axis."""
        self.tab_plot.logy = val
        self.plot_two_columns()

def lambda_failure_response(*args):
        """
        Helper function to create a Lambda Failure Response

        :return: A Flask Response
        """
        response_data = jsonify(ServiceErrorResponses._LAMBDA_FAILURE)
        return make_response(response_data, ServiceErrorResponses.HTTP_STATUS_CODE_502)

def join_field(path):
    """
    RETURN field SEQUENCE AS STRING
    """
    output = ".".join([f.replace(".", "\\.") for f in path if f != None])
    return output if output else "."

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

def time_string(seconds):
    """Returns time in seconds as a string formatted HHHH:MM:SS."""
    s = int(round(seconds))  # round to nearest second
    h, s = divmod(s, 3600)  # get hours and remainder
    m, s = divmod(s, 60)  # split remainder into minutes and seconds
    return "%2i:%02i:%02i" % (h, m, s)

def setupLogFile(self):
		"""Set up the logging file for a new session- include date and some whitespace"""
		self.logWrite("\n###############################################")
		self.logWrite("calcpkg.py log from " + str(datetime.datetime.now()))
		self.changeLogging(True)

def strip_accents(s):
    """
    Strip accents to prepare for slugification.
    """
    nfkd = unicodedata.normalize('NFKD', unicode(s))
    return u''.join(ch for ch in nfkd if not unicodedata.combining(ch))

def log_stop(logger):
    """log stop"""

    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)

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

def truncate(self, table):
        """Empty a table by deleting all of its rows."""
        if isinstance(table, (list, set, tuple)):
            for t in table:
                self._truncate(t)
        else:
            self._truncate(table)

def is_valid_row(cls, row):
        """Indicates whether or not the given row contains valid data."""
        for k in row.keys():
            if row[k] is None:
                return False
        return True

def size_on_disk(self):
        """
        :return: size of the entire schema in bytes
        """
        return int(self.connection.query(
            """
            SELECT SUM(data_length + index_length)
            FROM information_schema.tables WHERE table_schema='{db}'
            """.format(db=self.database)).fetchone()[0])

def random_filename(path=None):
    """Make a UUID-based file name which is extremely unlikely
    to exist already."""
    filename = uuid4().hex
    if path is not None:
        filename = os.path.join(path, filename)
    return filename

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

def to_networkx(graph):
    """ Convert a Mapper 1-complex to a networkx graph.

    Parameters
    -----------

    graph: dictionary, graph object returned from `kmapper.map`

    Returns
    --------

    g: graph as networkx.Graph() object

    """

    # import here so networkx is not always required.
    import networkx as nx

    nodes = graph["nodes"].keys()
    edges = [[start, end] for start, ends in graph["links"].items() for end in ends]

    g = nx.Graph()
    g.add_nodes_from(nodes)
    nx.set_node_attributes(g, dict(graph["nodes"]), "membership")

    g.add_edges_from(edges)

    return g

def get_account_id_by_fullname(self, fullname: str) -> str:
        """ Locates the account by fullname """
        account = self.get_by_fullname(fullname)
        return account.guid

def jsonify(resource):
    """Return a Flask ``Response`` object containing a
    JSON representation of *resource*.

    :param resource: The resource to act as the basis of the response
    """

    response = flask.jsonify(resource.to_dict())
    response = add_link_headers(response, resource.links())
    return response

def fgrad_y(self, y, return_precalc=False):
        """
        gradient of f w.r.t to y ([N x 1])

        :returns: Nx1 vector of derivatives, unless return_precalc is true, 
        then it also returns the precomputed stuff
        """
        d = self.d
        mpsi = self.psi

        # vectorized version
        S = (mpsi[:,1] * (y[:,:,None] + mpsi[:,2])).T
        R = np.tanh(S)
        D = 1 - (R ** 2)

        GRAD = (d + (mpsi[:,0:1][:,:,None] * mpsi[:,1:2][:,:,None] * D).sum(axis=0)).T

        if return_precalc:
            return GRAD, S, R, D

        return GRAD

def _debug_log(self, msg):
        """Debug log messages if debug=True"""
        if not self.debug:
            return
        sys.stderr.write('{}\n'.format(msg))

def parallel(processes, threads):
    """
    execute jobs in processes using N threads
    """
    pool = multithread(threads)
    pool.map(run_process, processes)
    pool.close()
    pool.join()

def quote(self, s):
        """Return a shell-escaped version of the string s."""

        if six.PY2:
            from pipes import quote
        else:
            from shlex import quote

        return quote(s)

def timestamp_to_datetime(timestamp):
    """Convert an ARF timestamp to a datetime.datetime object (naive local time)"""
    from datetime import datetime, timedelta
    obj = datetime.fromtimestamp(timestamp[0])
    return obj + timedelta(microseconds=int(timestamp[1]))

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

def price_rounding(price, decimals=2):
    """Takes a decimal price and rounds to a number of decimal places"""
    try:
        exponent = D('.' + decimals * '0')
    except InvalidOperation:
        # Currencies with no decimal places, ex. JPY, HUF
        exponent = D()
    return price.quantize(exponent, rounding=ROUND_UP)

def forceupdate(self, *args, **kw):
        """Like a bulk :meth:`forceput`."""
        self._update(False, self._ON_DUP_OVERWRITE, *args, **kw)

def reset_params(self):
        """Reset all parameters to their default values."""
        self.__params = dict([p, None] for p in self.param_names)
        self.set_params(self.param_defaults)

def build_docs(directory):
    """Builds sphinx docs from a given directory."""
    os.chdir(directory)
    process = subprocess.Popen(["make", "html"], cwd=directory)
    process.communicate()

def _histplot_op(ax, data, **kwargs):
    """Add a histogram for the data to the axes."""
    bins = get_bins(data)
    ax.hist(data, bins=bins, align="left", density=True, **kwargs)
    return ax

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

def strids2ids(tokens: Iterable[str]) -> List[int]:
    """
    Returns sequence of integer ids given a sequence of string ids.

    :param tokens: List of integer tokens.
    :return: List of word ids.
    """
    return list(map(int, tokens))

def format(x, format):
    """Uses http://www.cplusplus.com/reference/string/to_string/ for formatting"""
    # don't change the dtype, otherwise for each block the dtype may be different (string length)
    sl = vaex.strings.format(x, format)
    return column.ColumnStringArrow(sl.bytes, sl.indices, sl.length, sl.offset, string_sequence=sl)

def get_user_id_from_email(self, email):
        """ Uses the get-all-user-accounts Portals API to retrieve the
        user-id by supplying an email. """
        accts = self.get_all_user_accounts()

        for acct in accts:
            if acct['email'] == email:
                return acct['id']
        return None

def downsample(array, k):
    """Choose k random elements of array."""
    length = array.shape[0]
    indices = random.sample(xrange(length), k)
    return array[indices]

def zoomed_scaled_array_around_mask(self, mask, buffer=1):
        """Extract the 2D region of an array corresponding to the rectangle encompassing all unmasked values.

        This is used to extract and visualize only the region of an image that is used in an analysis.

        Parameters
        ----------
        mask : mask.Mask
            The mask around which the scaled array is extracted.
        buffer : int
            The buffer of pixels around the extraction.
        """
        return self.new_with_array(array=array_util.extracted_array_2d_from_array_2d_and_coordinates(
            array_2d=self,  y0=mask.zoom_region[0]-buffer, y1=mask.zoom_region[1]+buffer,
            x0=mask.zoom_region[2]-buffer, x1=mask.zoom_region[3]+buffer))

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

def metadata(self):
        """google.protobuf.Message: the current operation metadata."""
        if not self._operation.HasField("metadata"):
            return None

        return protobuf_helpers.from_any_pb(
            self._metadata_type, self._operation.metadata
        )

async def vc_check(ctx: commands.Context):  # pylint: disable=unused-argument
    """
    Check for whether VC is available in this bot.
    """

    if not discord.voice_client.has_nacl:
        raise commands.CheckFailure("voice cannot be used because PyNaCl is not loaded")

    if not discord.opus.is_loaded():
        raise commands.CheckFailure("voice cannot be used because libopus is not loaded")

    return True

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

def MatrixSolve(a, rhs, adj):
    """
    Matrix solve op.
    """
    return np.linalg.solve(a if not adj else _adjoint(a), rhs),

def angle_to_cartesian(lon, lat):
    """Convert spherical coordinates to cartesian unit vectors."""
    theta = np.array(np.pi / 2. - lat)
    return np.vstack((np.sin(theta) * np.cos(lon),
                      np.sin(theta) * np.sin(lon),
                      np.cos(theta))).T

def fail(message=None, exit_status=None):
    """Prints the specified message and exits the program with the specified
    exit status.

    """
    print('Error:', message, file=sys.stderr)
    sys.exit(exit_status or 1)

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

def disable_busy_cursor():
    """Disable the hourglass cursor and listen for layer changes."""
    while QgsApplication.instance().overrideCursor() is not None and \
            QgsApplication.instance().overrideCursor().shape() == \
            QtCore.Qt.WaitCursor:
        QgsApplication.instance().restoreOverrideCursor()

def ranks(self, key, value):
    """Populate the ``ranks`` key."""
    return [normalize_rank(el) for el in force_list(value.get('a'))]

def strip_comments(string, comment_symbols=frozenset(('#', '//'))):
    """Strip comments from json string.

    :param string: A string containing json with comments started by comment_symbols.
    :param comment_symbols: Iterable of symbols that start a line comment (default # or //).
    :return: The string with the comments removed.
    """
    lines = string.splitlines()
    for k in range(len(lines)):
        for symbol in comment_symbols:
            lines[k] = strip_comment_line_with_symbol(lines[k], start=symbol)
    return '\n'.join(lines)

def _save_file(self, filename, contents):
        """write the html file contents to disk"""
        with open(filename, 'w') as f:
            f.write(contents)

def r_num(obj):
    """Read list of numbers."""
    if isinstance(obj, (list, tuple)):
        it = iter
    else:
        it = LinesIterator
    dataset = Dataset([Dataset.FLOAT])
    return dataset.load(it(obj))

def input_validate_str(string, name, max_len=None, exact_len=None):
    """ Input validation for strings. """
    if type(string) is not str:
        raise pyhsm.exception.YHSM_WrongInputType(name, str, type(string))
    if max_len != None and len(string) > max_len:
        raise pyhsm.exception.YHSM_InputTooLong(name, max_len, len(string))
    if exact_len != None and len(string) != exact_len:
        raise pyhsm.exception.YHSM_WrongInputSize(name, exact_len, len(string))
    return string

def _stdin_(p):
    """Takes input from user. Works for Python 2 and 3."""
    _v = sys.version[0]
    return input(p) if _v is '3' else raw_input(p)

def total_seconds(td):
  """convert a timedelta to seconds.

  This is patterned after timedelta.total_seconds, which is only
  available in python 27.

  Args:
    td: a timedelta object.

  Returns:
    total seconds within a timedelta. Rounded up to seconds.
  """
  secs = td.seconds + td.days * 24 * 3600
  if td.microseconds:
    secs += 1
  return secs

def validate(self, xml_input):
        """
        This method validate the parsing and schema, return a boolean
        """
        parsed_xml = etree.parse(self._handle_xml(xml_input))
        try:
            return self.xmlschema.validate(parsed_xml)
        except AttributeError:
            raise CannotValidate('Set XSD to validate the XML')

def set_pivot_keys(self, foreign_key, other_key):
        """
        Set the key names for the pivot model instance
        """
        self.__foreign_key = foreign_key
        self.__other_key = other_key

        return self

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

def output_dir(self, *args) -> str:
        """ Directory where to store output """
        return os.path.join(self.project_dir, 'output', *args)

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

def maxDepth(self, currentDepth=0):
        """Compute the depth of the longest branch of the tree"""
        if not any((self.left, self.right)):
            return currentDepth
        result = 0
        for child in (self.left, self.right):
            if child:
                result = max(result, child.maxDepth(currentDepth + 1))
        return result

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

def __init__(self, collection, index_type_obj):
        """
            Constructs wrapper for general index creation and deletion

            :param collection Collection
            :param index_type_obj BaseIndex Object of a index sub-class
        """

        self.collection = collection
        self.index_type_obj = index_type_obj

def _read_date_from_string(str1):
    """
    Reads the date from a string in the format YYYY/MM/DD and returns
    :class: datetime.date
    """
    full_date = [int(x) for x in str1.split('/')]
    return datetime.date(full_date[0], full_date[1], full_date[2])

def is_a_sequence(var, allow_none=False):
    """ Returns True if var is a list or a tuple (but not a string!)
    """
    return isinstance(var, (list, tuple)) or (var is None and allow_none)

async def executemany(self, sql: str, parameters: Iterable[Iterable[Any]]) -> None:
        """Execute the given multiquery."""
        await self._execute(self._cursor.executemany, sql, parameters)

def printComparison(results, class_or_prop):
	"""
	print(out the results of the comparison using a nice table)
	"""

	data = []

	Row = namedtuple('Row',[class_or_prop,'VALIDATED'])

	for k,v in sorted(results.items(), key=lambda x: x[1]):
		data += [Row(k, str(v))]

	pprinttable(data)

def execute(self, cmd, *args, **kwargs):
        """ Execute the SQL command and return the data rows as tuples
        """
        self.cursor.execute(cmd, *args, **kwargs)

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

def log_magnitude_spectrum(frames):
    """Compute the log of the magnitude spectrum of frames"""
    return N.log(N.abs(N.fft.rfft(frames)).clip(1e-5, N.inf))

def changed(self):
        """Returns dict of fields that changed since save (with old values)"""
        return dict(
            (field, self.previous(field))
            for field in self.fields
            if self.has_changed(field)
        )

def tanimoto_set_similarity(x: Iterable[X], y: Iterable[X]) -> float:
    """Calculate the tanimoto set similarity."""
    a, b = set(x), set(y)
    union = a | b

    if not union:
        return 0.0

    return len(a & b) / len(union)

def read_string(buff, byteorder='big'):
    """Read a string from a file-like object."""
    length = read_numeric(USHORT, buff, byteorder)
    return buff.read(length).decode('utf-8')

def method_double_for(self, method_name):
        """Returns the method double for the provided method name, creating one if necessary.

        :param str method_name: The name of the method to retrieve a method double for.
        :return: The mapped ``MethodDouble``.
        :rtype: MethodDouble
        """

        if method_name not in self._method_doubles:
            self._method_doubles[method_name] = MethodDouble(method_name, self._target)

        return self._method_doubles[method_name]

def list_of_lists_to_dict(l):
    """ Convert list of key,value lists to dict

    [['id', 1], ['id', 2], ['id', 3], ['foo': 4]]
    {'id': [1, 2, 3], 'foo': [4]}
    """
    d = {}
    for key, val in l:
        d.setdefault(key, []).append(val)
    return d

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

def file_uptodate(fname, cmp_fname):
    """Check if a file exists, is non-empty and is more recent than cmp_fname.
    """
    try:
        return (file_exists(fname) and file_exists(cmp_fname) and
                getmtime(fname) >= getmtime(cmp_fname))
    except OSError:
        return False

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

def get_list_from_file(file_name):
    """read the lines from a file into a list"""
    with open(file_name, mode='r', encoding='utf-8') as f1:
        lst = f1.readlines()
    return lst

def kill_process(process):
    """Kill the process group associated with the given process. (posix)"""
    logger = logging.getLogger('xenon')
    logger.info('Terminating Xenon-GRPC server.')
    os.kill(process.pid, signal.SIGINT)
    process.wait()

def normalize_enum_constant(s):
    """Return enum constant `s` converted to a canonical snake-case."""
    if s.islower(): return s
    if s.isupper(): return s.lower()
    return "".join(ch if ch.islower() else "_" + ch.lower() for ch in s).strip("_")

def mag(z):
    """Get the magnitude of a vector."""
    if isinstance(z[0], np.ndarray):
        return np.array(list(map(np.linalg.norm, z)))
    else:
        return np.linalg.norm(z)

def filter_query(s):
    """
    Filters given query with the below regex
    and returns lists of quoted and unquoted strings
    """
    matches = re.findall(r'(?:"([^"]*)")|([^"]*)', s)
    result_quoted = [t[0].strip() for t in matches if t[0]]
    result_unquoted = [t[1].strip() for t in matches if t[1]]
    return result_quoted, result_unquoted

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

def get_colors(img):
    """
    Returns a list of all the image's colors.
    """
    w, h = img.size
    return [color[:3] for count, color in img.convert('RGB').getcolors(w * h)]

def _if(ctx, logical_test, value_if_true=0, value_if_false=False):
    """
    Returns one value if the condition evaluates to TRUE, and another value if it evaluates to FALSE
    """
    return value_if_true if conversions.to_boolean(logical_test, ctx) else value_if_false

def debug_on_error(type, value, tb):
    """Code due to Thomas Heller - published in Python Cookbook (O'Reilley)"""
    traceback.print_exc(type, value, tb)
    print()
    pdb.pm()

def was_into_check(self) -> bool:
        """
        Checks if the king of the other side is attacked. Such a position is not
        valid and could only be reached by an illegal move.
        """
        king = self.king(not self.turn)
        return king is not None and self.is_attacked_by(self.turn, king)

def option2tuple(opt):
    """Return a tuple of option, taking possible presence of level into account"""

    if isinstance(opt[0], int):
        tup = opt[1], opt[2:]
    else:
        tup = opt[0], opt[1:]

    return tup

def _check_for_errors(etree: ET.ElementTree):
    """Check AniDB response XML tree for errors."""
    if etree.getroot().tag == 'error':
        raise APIError(etree.getroot().text)

def calc_volume(self, sample: np.ndarray):
        """Find the RMS of the audio"""
        return sqrt(np.mean(np.square(sample)))

def list_all(dev: Device):
    """List all available API calls."""
    for name, service in dev.services.items():
        click.echo(click.style("\nService %s" % name, bold=True))
        for method in service.methods:
            click.echo("  %s" % method.name)

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

def getAllTriples(self):
        """Returns:

        list of tuples : Each tuple holds a subject, predicate, object triple

        """
        return [(str(s), str(p), str(o)) for s, p, o in self]

def wrap(text, indent='    '):
    """Wrap text to terminal width with default indentation"""
    wrapper = textwrap.TextWrapper(
        width=int(os.environ.get('COLUMNS', 80)),
        initial_indent=indent,
        subsequent_indent=indent
    )
    return '\n'.join(wrapper.wrap(text))

def cpp_checker(code, working_directory):
    """Return checker."""
    return gcc_checker(code, '.cpp',
                       [os.getenv('CXX', 'g++'), '-std=c++0x'] + INCLUDE_FLAGS,
                       working_directory=working_directory)

def add_chart(self, chart, row, col):
        """
        Adds a chart to the worksheet at (row, col).

        :param xltable.Chart Chart: chart to add to the workbook.
        :param int row: Row to add the chart at.
        """
        self.__charts.append((chart, (row, col)))

def norm(x, mu, sigma=1.0):
    """ Scipy norm function """
    return stats.norm(loc=mu, scale=sigma).pdf(x)

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

def filter_contour(imageFile, opFile):
    """ convert an image by applying a contour """
    im = Image.open(imageFile)
    im1 = im.filter(ImageFilter.CONTOUR)
    im1.save(opFile)

def is_float(value):
    """must be a float"""
    return isinstance(value, float) or isinstance(value, int) or isinstance(value, np.float64), float(value)

def __del__(self):
    """Cleans up the file entry."""
    # __del__ can be invoked before __init__ has completed.
    if hasattr(self, '_encoded_stream'):
      self._encoded_stream.close()
      self._encoded_stream = None

    super(EncodedStreamFileEntry, self).__del__()

def printc(cls, txt, color=colors.red):
        """Print in color."""
        print(cls.color_txt(txt, color))

def utcfromtimestamp(cls, timestamp):
    """Returns a datetime object of a given timestamp (in UTC)."""
    obj = datetime.datetime.utcfromtimestamp(timestamp)
    obj = pytz.utc.localize(obj)
    return cls(obj)

def fromtimestamp(cls, timestamp):
    """Returns a datetime object of a given timestamp (in local tz)."""
    d = cls.utcfromtimestamp(timestamp)
    return d.astimezone(localtz())

def stop_button_click_handler(self):
        """Method to handle what to do when the stop button is pressed"""
        self.stop_button.setDisabled(True)
        # Interrupt computations or stop debugging
        if not self.shellwidget._reading:
            self.interrupt_kernel()
        else:
            self.shellwidget.write_to_stdin('exit')

def get_known_read_position(fp, buffered=True):
    """ 
    Return a position in a file which is known to be read & handled.
    It assumes a buffered file and streaming processing. 
    """
    buffer_size = io.DEFAULT_BUFFER_SIZE if buffered else 0
    return max(fp.tell() - buffer_size, 0)

def sav_to_pandas_rpy2(input_file):
    """
    SPSS .sav files to Pandas DataFrame through Rpy2

    :param input_file: string

    :return:
    """
    import pandas.rpy.common as com

    w = com.robj.r('foreign::read.spss("%s", to.data.frame=TRUE)' % input_file)
    return com.convert_robj(w)

def version_jar(self):
		"""
		Special case of version() when the executable is a JAR file.
		"""
		cmd = config.get_command('java')
		cmd.append('-jar')
		cmd += self.cmd
		self.version(cmd=cmd, path=self.cmd[0])

def paren_change(inputstring, opens=opens, closes=closes):
    """Determine the parenthetical change of level (num closes - num opens)."""
    count = 0
    for c in inputstring:
        if c in opens:  # open parens/brackets/braces
            count -= 1
        elif c in closes:  # close parens/brackets/braces
            count += 1
    return count

def bfx(value, msb, lsb):
    """! @brief Extract a value from a bitfield."""
    mask = bitmask((msb, lsb))
    return (value & mask) >> lsb

def lognorm(x, mu, sigma=1.0):
    """ Log-normal function from scipy """
    return stats.lognorm(sigma, scale=mu).pdf(x)

def format_docstring(*args, **kwargs):
    """
    Decorator for clean docstring formatting
    """
    def decorator(func):
        func.__doc__ = getdoc(func).format(*args, **kwargs)
        return func
    return decorator

def has_multiline_items(maybe_list: Optional[Sequence[str]]):
    """Check whether one of the items in the list has multiple lines."""
    return maybe_list and any(is_multiline(item) for item in maybe_list)

def zeros(self, name, **kwargs):
        """Create an array. Keyword arguments as per
        :func:`zarr.creation.zeros`."""
        return self._write_op(self._zeros_nosync, name, **kwargs)

def append_table(self, name, **kwargs):
        """Create a new table."""
        self.stack.append(Table(name, **kwargs))

def dotproduct(X, Y):
    """Return the sum of the element-wise product of vectors x and y.
    >>> dotproduct([1, 2, 3], [1000, 100, 10])
    1230
    """
    return sum([x * y for x, y in zip(X, Y)])

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

def remove_stopped_threads (self):
        """Remove the stopped threads from the internal thread list."""
        self.threads = [t for t in self.threads if t.is_alive()]

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

def to_dict(self):
        """Converts the table to a dict."""
        return {"name": self.table_name, "kind": self.table_kind, "data": [r.to_dict() for r in self]}

def get_property(self):
        """Establishes access of GettableProperty values"""

        scope = self

        def fget(self):
            """Call the HasProperties _get method"""
            return self._get(scope.name)

        return property(fget=fget, doc=scope.sphinx())

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

def cross_list(*sequences):
  """
  From: http://book.opensourceproject.org.cn/lamp/python/pythoncook2/opensource/0596007973/pythoncook2-chp-19-sect-9.html
  """
  result = [[ ]]
  for seq in sequences:
    result = [sublist+[item] for sublist in result for item in seq]
  return result

def make_prefixed_stack_name(prefix, template_path):
    """

    :param prefix:
    :param template_path:
    """
    parts = os.path.basename(template_path).split('-')
    parts = parts if len(parts) == 1 else parts[:-1]
    return ('%s-%s' % (prefix, '-'.join(parts))).split('.')[0]

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

def page_title(step, title):
    """
    Check that the page title matches the given one.
    """

    with AssertContextManager(step):
        assert_equals(world.browser.title, title)

def _set_lastpage(self):
        """Calculate value of class attribute ``last_page``."""
        self.last_page = (len(self._page_data) - 1) // self.screen.page_size

def __delitem__(self, key):
		"""Remove item with given key from the mapping.

		Runs in O(n), unless removing last item, then in O(1).
		"""
		index, value = self._dict.pop(key)
		key2, value2 = self._list.pop(index)
		assert key == key2
		assert value is value2

		self._fix_indices_after_delete(index)

def _default(self, obj):
        """ return a serialized version of obj or raise a TypeError

        :param obj:
        :return: Serialized version of obj
        """
        return obj.__dict__ if isinstance(obj, JsonObj) else json.JSONDecoder().decode(obj)

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

def uninstall(cls):
        """Remove the package manager from the system."""
        if os.path.exists(cls.home):
            shutil.rmtree(cls.home)

def normcdf(x, log=False):
    """Normal cumulative density function."""
    y = np.atleast_1d(x).copy()
    flib.normcdf(y)
    if log:
        if (y>0).all():
            return np.log(y)
        return -np.inf
    return y

def main(argv, version=DEFAULT_VERSION):
    """Install or upgrade setuptools and EasyInstall"""
    tarball = download_setuptools()
    _install(tarball, _build_install_args(argv))

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

def default_parser() -> argparse.ArgumentParser:
    """Create a parser for CLI arguments and options."""
    parser = argparse.ArgumentParser(
        prog=CONSOLE_SCRIPT,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    build_parser(parser)
    return parser

def extract(self, destination):
        """Extract the archive."""
        with zipfile.ZipFile(self.archive, 'r') as zip_ref:
            zip_ref.extractall(destination)

def get_func_name(func):
    """Return a name which includes the module name and function name."""
    func_name = getattr(func, '__name__', func.__class__.__name__)
    module_name = func.__module__

    if module_name is not None:
        module_name = func.__module__
        return '{}.{}'.format(module_name, func_name)

    return func_name

def _array2cstr(arr):
    """ Serializes a numpy array to a compressed base64 string """
    out = StringIO()
    np.save(out, arr)
    return b64encode(out.getvalue())

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

def read(self):
        """Iterate over all JSON input (Generator)"""

        for line in self.io.read():
            with self.parse_line(line) as j:
                yield j

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

def search_for_tweets_about(user_id, params):
    """ Search twitter API """
    url = "https://api.twitter.com/1.1/search/tweets.json"
    response = make_twitter_request(url, user_id, params)
    return process_tweets(response.json()["statuses"])

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

def main(idle):
    """Any normal python logic which runs a loop. Can take arguments."""
    while True:

        LOG.debug("Sleeping for {0} seconds.".format(idle))
        time.sleep(idle)

def check_color(cls, raw_image):
        """
        Just check if raw_image is completely white.
        http://stackoverflow.com/questions/14041562/python-pil-detect-if-an-image-is-completely-black-or-white
        """
        # sum(img.convert("L").getextrema()) in (0, 2)
        extrema = raw_image.convert("L").getextrema()
        if extrema == (255, 255): # all white
            raise cls.MonoImageException

def _increment(arr, indices):
    """Increment some indices in a 1D vector of non-negative integers.
    Repeated indices are taken into account."""
    arr = _as_array(arr)
    indices = _as_array(indices)
    bbins = np.bincount(indices)
    arr[:len(bbins)] += bbins
    return arr

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

def itervalues(d, **kw):
    """Return an iterator over the values of a dictionary."""
    if not PY2:
        return iter(d.values(**kw))
    return d.itervalues(**kw)

def read_credentials(fname):
    """
    read a simple text file from a private location to get
    username and password
    """
    with open(fname, 'r') as f:
        username = f.readline().strip('\n')
        password = f.readline().strip('\n')
    return username, password

def compose(func_list):
    """
    composion of preprocessing functions
    """

    def f(G, bim):
        for func in func_list:
            G, bim = func(G, bim)
        return G, bim

    return f

def expect_all(a, b):
    """\
    Asserts that two iterables contain the same values.
    """
    assert all(_a == _b for _a, _b in zip_longest(a, b))

def _validate_pos(df):
    """Validates the returned positional object
    """
    assert isinstance(df, pd.DataFrame)
    assert ["seqname", "position", "strand"] == df.columns.tolist()
    assert df.position.dtype == np.dtype("int64")
    assert df.strand.dtype == np.dtype("O")
    assert df.seqname.dtype == np.dtype("O")
    return df

def positive_int(val):
    """Parse `val` into a positive integer."""
    if isinstance(val, float):
        raise ValueError('"{}" must not be a float'.format(val))
    val = int(val)
    if val >= 0:
        return val
    raise ValueError('"{}" must be positive'.format(val))

def GetIndentLevel(line):
  """Return the number of leading spaces in line.

  Args:
    line: A string to check.

  Returns:
    An integer count of leading spaces, possibly zero.
  """
  indent = Match(r'^( *)\S', line)
  if indent:
    return len(indent.group(1))
  else:
    return 0

def AsPrimitiveProto(self):
    """Return an old style protocol buffer object."""
    if self.protobuf:
      result = self.protobuf()
      result.ParseFromString(self.SerializeToString())
      return result

def _go_to_line(editor, line):
    """
    Move cursor to this line in the current buffer.
    """
    b = editor.application.current_buffer
    b.cursor_position = b.document.translate_row_col_to_index(max(0, int(line) - 1), 0)

def is_client(self):
        """Return True if Glances is running in client mode."""
        return (self.args.client or self.args.browser) and not self.args.server

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

def load_search_freq(fp=SEARCH_FREQ_JSON):
    """
    Load the search_freq from JSON file
    """
    try:
        with open(fp) as f:
            return Counter(json.load(f))
    except FileNotFoundError:
        return Counter()

def dereference_url(url):
    """
    Makes a HEAD request to find the final destination of a URL after
    following any redirects
    """
    res = open_url(url, method='HEAD')
    res.close()
    return res.url

def signal_handler(signal_name, frame):
    """Quit signal handler."""
    sys.stdout.flush()
    print("\nSIGINT in frame signal received. Quitting...")
    sys.stdout.flush()
    sys.exit(0)

def compress(data, **kwargs):
    """zlib.compress(data, **kwargs)
    
    """ + zopfli.__COMPRESSOR_DOCSTRING__  + """
    Returns:
      String containing a zlib container
    """
    kwargs['gzip_mode'] = 0
    return zopfli.zopfli.compress(data, **kwargs)

def __grid_widgets(self):
        """Places all the child widgets in the appropriate positions."""
        scrollbar_column = 0 if self.__compound is tk.LEFT else 2
        self._canvas.grid(row=0, column=1, sticky="nswe")
        self._scrollbar.grid(row=0, column=scrollbar_column, sticky="ns")

def print_ldamodel_topic_words(topic_word_distrib, vocab, n_top=10, row_labels=DEFAULT_TOPIC_NAME_FMT):
    """Print `n_top` values from a LDA model's topic-word distributions."""
    print_ldamodel_distribution(topic_word_distrib, row_labels=row_labels, val_labels=vocab,
                                top_n=n_top)

def parse_domain(url):
    """ parse the domain from the url """
    domain_match = lib.DOMAIN_REGEX.match(url)
    if domain_match:
        return domain_match.group()

def _safe_db(num, den):
    """Properly handle the potential +Inf db SIR instead of raising a
    RuntimeWarning.
    """
    if den == 0:
        return np.inf
    return 10 * np.log10(num / den)

def _or(ctx, *logical):
    """
    Returns TRUE if any argument is TRUE
    """
    for arg in logical:
        if conversions.to_boolean(arg, ctx):
            return True
    return False

def keys(self):
        """Return a list of all keys in the dictionary.

        Returns:
            list of str: [key1,key2,...,keyN]
        """
        all_keys = [k.decode('utf-8') for k,v in self.rdb.hgetall(self.session_hash).items()]
        return all_keys

def _create_statusicon(self):
        """Return a new Gtk.StatusIcon."""
        statusicon = Gtk.StatusIcon()
        statusicon.set_from_gicon(self._icons.get_gicon('media'))
        statusicon.set_tooltip_text(_("udiskie"))
        return statusicon

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

def win32_refresh_window(cls):
        """
        Call win32 API to refresh the whole Window.

        This is sometimes necessary when the application paints background
        for completion menus. When the menu disappears, it leaves traces due
        to a bug in the Windows Console. Sending a repaint request solves it.
        """
        # Get console handle
        handle = windll.kernel32.GetConsoleWindow()

        RDW_INVALIDATE = 0x0001
        windll.user32.RedrawWindow(handle, None, None, c_uint(RDW_INVALIDATE))

def copy_default_data_file(filename, module=None):
    """Copies file from default data directory to local directory."""
    if module is None:
        module = __get_filetypes_module()
    fullpath = get_default_data_path(filename, module=module)
    shutil.copy(fullpath, ".")

def register_extension_class(ext, base, *args, **kwargs):
    """Instantiate the given extension class and register as a public attribute of the given base.

    README: The expected protocol here is to instantiate the given extension and pass the base
    object as the first positional argument, then unpack args and kwargs as additional arguments to
    the extension's constructor.
    """
    ext_instance = ext.plugin(base, *args, **kwargs)
    setattr(base, ext.name.lstrip('_'), ext_instance)

def cleanup_storage(*args):
    """Clean up processes after SIGTERM or SIGINT is received."""
    ShardedClusters().cleanup()
    ReplicaSets().cleanup()
    Servers().cleanup()
    sys.exit(0)

def discard(self, element):
        """Remove element from the RangeSet if it is a member.

        If the element is not a member, do nothing.
        """
        try:
            i = int(element)
            set.discard(self, i)
        except ValueError:
            pass

def print_bintree(tree, indent='  '):
    """print a binary tree"""
    for n in sorted(tree.keys()):
        print "%s%s" % (indent * depth(n,tree), n)

def _interval_to_bound_points(array):
    """
    Helper function which returns an array
    with the Intervals' boundaries.
    """

    array_boundaries = np.array([x.left for x in array])
    array_boundaries = np.concatenate(
        (array_boundaries, np.array([array[-1].right])))

    return array_boundaries

def get_as_string(self, s3_path, encoding='utf-8'):
        """
        Get the contents of an object stored in S3 as string.

        :param s3_path: URL for target S3 location
        :param encoding: Encoding to decode bytes to string
        :return: File contents as a string
        """
        content = self.get_as_bytes(s3_path)
        return content.decode(encoding)

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

def __contains__ (self, key):
        """Check lowercase key item."""
        assert isinstance(key, basestring)
        return dict.__contains__(self, key.lower())

def get_data_table(filename):
  """Returns a DataTable instance built from either the filename, or STDIN if filename is None."""
  with get_file_object(filename, "r") as rf:
    return DataTable(list(csv.reader(rf)))

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

def myreplace(astr, thefind, thereplace):
    """in string astr replace all occurences of thefind with thereplace"""
    alist = astr.split(thefind)
    new_s = alist.split(thereplace)
    return new_s

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

def Cinv(self):
        """Inverse of the noise covariance."""
        try:
            return np.linalg.inv(self.c)
        except np.linalg.linalg.LinAlgError:
            print('Warning: non-invertible noise covariance matrix c.')
            return np.eye(self.c.shape[0])

def put_text(self, key, text):
        """Put the text into the storage associated with the key."""
        with open(key, "w") as fh:
            fh.write(text)

def strip_tweet(text, remove_url=True):
    """Strip tweet message.

    This method removes mentions strings and urls(optional).

    :param text: tweet message
    :type text: :class:`str`

    :param remove_url: Remove urls. default :const:`True`.
    :type remove_url: :class:`boolean`

    :returns: Striped tweet message
    :rtype: :class:`str`

    """
    if remove_url:
        text = url_pattern.sub('', text)
    else:
        text = expand_url(text)
    text = mention_pattern.sub('', text)
    text = html_parser.unescape(text)
    text = text.strip()
    return text

def algo_exp(x, m, t, b):
    """mono-exponential curve."""
    return m*np.exp(-t*x)+b

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

def do_rewind(self, line):
        """
        rewind
        """
        self.print_response("Rewinding from frame %s to 0" % self.bot._frame)
        self.bot._frame = 0

def to_unix(cls, timestamp):
        """ Wrapper over time module to produce Unix epoch time as a float """
        if not isinstance(timestamp, datetime.datetime):
            raise TypeError('Time.milliseconds expects a datetime object')
        base = time.mktime(timestamp.timetuple())
        return base

def close_database_session(session):
    """Close connection with the database"""

    try:
        session.close()
    except OperationalError as e:
        raise DatabaseError(error=e.orig.args[1], code=e.orig.args[0])

def __enter__(self):
        """Enable the download log filter."""
        self.logger = logging.getLogger('pip.download')
        self.logger.addFilter(self)

def clean(s):
  """Removes trailing whitespace on each line."""
  lines = [l.rstrip() for l in s.split('\n')]
  return '\n'.join(lines)

def roc_auc(y_true, y_score):
    """
    Returns are under the ROC curve
    """
    notnull = ~np.isnan(y_true)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true[notnull], y_score[notnull])
    return sklearn.metrics.auc(fpr, tpr)

def _clear(self):
        """Resets all assigned data for the current message."""
        self._finished = False
        self._measurement = None
        self._message = None
        self._message_body = None

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

def multi_replace(instr, search_list=[], repl_list=None):
    """
    Does a string replace with a list of search and replacements

    TODO: rename
    """
    repl_list = [''] * len(search_list) if repl_list is None else repl_list
    for ser, repl in zip(search_list, repl_list):
        instr = instr.replace(ser, repl)
    return instr

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

def __init__(self, find, subcon):
        """Initialize."""
        Subconstruct.__init__(self, subcon)
        self.find = find

def count_rows_with_nans(X):
    """Count the number of rows in 2D arrays that contain any nan values."""
    if X.ndim == 2:
        return np.where(np.isnan(X).sum(axis=1) != 0, 1, 0).sum()

def is_full_slice(obj, l):
    """
    We have a full length slice.
    """
    return (isinstance(obj, slice) and obj.start == 0 and obj.stop == l and
            obj.step is None)

def _wait_for_response(self):
		"""
		Wait until the user accepted or rejected the request
		"""
		while not self.server.response_code:
			time.sleep(2)
		time.sleep(5)
		self.server.shutdown()

def safe_format(s, **kwargs):
  """
  :type s str
  """
  return string.Formatter().vformat(s, (), defaultdict(str, **kwargs))

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

def urljoin(*args):
    """
    Joins given arguments into a url, removing duplicate slashes
    Thanks http://stackoverflow.com/a/11326230/1267398

    >>> urljoin('/lol', '///lol', '/lol//')
    '/lol/lol/lol'
    """
    value = "/".join(map(lambda x: str(x).strip('/'), args))
    return "/{}".format(value)

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

def args_update(self):
        """Update the argparser namespace with any data from configuration file."""
        for key, value in self._config_data.items():
            setattr(self._default_args, key, value)

def generate_id(self, obj):
        """Generate unique document id for ElasticSearch."""
        object_type = type(obj).__name__.lower()
        return '{}_{}'.format(object_type, self.get_object_id(obj))

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

def save(self, fname: str):
        """
        Saves this training state to fname.
        """
        with open(fname, "wb") as fp:
            pickle.dump(self, fp)

def encode_to_shape(inputs, shape, scope):
  """Encode the given tensor to given image shape."""
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    w, h = shape[1], shape[2]
    x = inputs
    x = tfl.flatten(x)
    x = tfl.dense(x, w * h, activation=None, name="enc_dense")
    x = tf.reshape(x, (-1, w, h, 1))
    return x

def set_header(self, key, value):
    """ Sets a HTTP header for future requests. """
    self.conn.issue_command("Header", _normalize_header(key), value)

def get_default_preds():
    """dynamically build autocomplete options based on an external file"""
    g = ontospy.Ontospy(rdfsschema, text=True, verbose=False, hide_base_schemas=False)
    classes = [(x.qname, x.bestDescription()) for x in g.all_classes]
    properties = [(x.qname, x.bestDescription()) for x in g.all_properties]
    commands = [('exit', 'exits the terminal'), ('show', 'show current buffer')]
    return rdfschema + owlschema + classes + properties + commands

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

def is_power_of_2(num):
    """Return whether `num` is a power of two"""
    log = math.log2(num)
    return int(log) == float(log)

def cio_close(cio):
    """Wraps openjpeg library function cio_close.
    """
    OPENJPEG.opj_cio_close.argtypes = [ctypes.POINTER(CioType)]
    OPENJPEG.opj_cio_close(cio)

def get_site_name(request):
    """Return the domain:port part of the URL without scheme.
    Eg: facebook.com, 127.0.0.1:8080, etc.
    """
    urlparts = request.urlparts
    return ':'.join([urlparts.hostname, str(urlparts.port)])

def ruler_line(self, widths, linetype='-'):
        """Generates a ruler line for separating rows from each other"""
        cells = []
        for w in widths:
            cells.append(linetype * (w+2))
        return '+' + '+'.join(cells) + '+'

def round_sig(x, sig):
    """Round the number to the specified number of significant figures"""
    return round(x, sig - int(floor(log10(abs(x)))) - 1)

def get_dates_link(url):
    """ download the dates file from the internet and parse it as a dates file"""
    urllib.request.urlretrieve(url, "temp.txt")
    dates = get_dates_file("temp.txt")
    os.remove("temp.txt")
    return dates

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

def n_choose_k(n, k):
    """ get the number of quartets as n-choose-k. This is used
    in equal splits to decide whether a split should be exhaustively sampled
    or randomly sampled. Edges near tips can be exhaustive while highly nested
    edges probably have too many quartets
    """
    return int(reduce(MUL, (Fraction(n-i, i+1) for i in range(k)), 1))

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

def get_pid_list():
    """Returns a list of PIDs currently running on the system."""
    pids = [int(x) for x in os.listdir('/proc') if x.isdigit()]
    return pids

def selectnotnone(table, field, complement=False):
    """Select rows where the given field is not `None`."""

    return select(table, field, lambda v: v is not None,
                  complement=complement)

def _set_scroll_v(self, *args):
        """Scroll both categories Canvas and scrolling container"""
        self._canvas_categories.yview(*args)
        self._canvas_scroll.yview(*args)

def get_flat_size(self):
        """Returns the total length of all of the flattened variables.

        Returns:
            The length of all flattened variables concatenated.
        """
        return sum(
            np.prod(v.get_shape().as_list()) for v in self.variables.values())

def count_rows(self, table, cols='*'):
        """Get the number of rows in a particular table."""
        query = 'SELECT COUNT({0}) FROM {1}'.format(join_cols(cols), wrap(table))
        result = self.fetch(query)
        return result if result is not None else 0

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

def update(self, other_dict):
        """update() extends rather than replaces existing key lists."""
        for key, value in iter_multi_items(other_dict):
            MultiDict.add(self, key, value)

def test_kwargs_are_optional(self):
        """kwarg values always have defaults"""
        with patch("sys.exit") as mock_exit:
            cli = MicroCLITestCase.T("script_name f3".split()).run()
            # kwargs are optional
            mock_exit.assert_called_with(4)

def dict_to_html_attrs(dict_):
    """
    Banana banana
    """
    res = ' '.join('%s="%s"' % (k, v) for k, v in dict_.items())
    return res

def machine_info():
    """Retrieve core and memory information for the current machine.
    """
    import psutil
    BYTES_IN_GIG = 1073741824.0
    free_bytes = psutil.virtual_memory().total
    return [{"memory": float("%.1f" % (free_bytes / BYTES_IN_GIG)), "cores": multiprocessing.cpu_count(),
             "name": socket.gethostname()}]

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

def _store_helper(model: Action, session: Optional[Session] = None) -> None:
    """Help store an action."""
    if session is None:
        session = _make_session()

    session.add(model)
    session.commit()
    session.close()

def datetime_created(self):
        """Returns file group's create aware *datetime* in UTC format."""
        if self.info().get('datetime_created'):
            return dateutil.parser.parse(self.info()['datetime_created'])

def isin(value, values):
    """ Check that value is in values """
    for i, v in enumerate(value):
        if v not in np.array(values)[:, i]:
            return False
    return True

def inFocus(self):
        """Set GUI on-top flag"""
        previous_flags = self.window.flags()
        self.window.setFlags(previous_flags |
                             QtCore.Qt.WindowStaysOnTopHint)

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

def kick(self, channel, nick, comment=""):
        """Send a KICK command."""
        self.send_items('KICK', channel, nick, comment and ':' + comment)

def generate_unique_host_id():
    """Generate a unique ID, that is somewhat guaranteed to be unique among all
    instances running at the same time."""
    host = ".".join(reversed(socket.gethostname().split(".")))
    pid = os.getpid()
    return "%s.%d" % (host, pid)

def model_field_attr(model, model_field, attr):
    """
    Returns the specified attribute for the specified field on the model class.
    """
    fields = dict([(field.name, field) for field in model._meta.fields])
    return getattr(fields[model_field], attr)

def entropy(string):
    """Compute entropy on the string"""
    p, lns = Counter(string), float(len(string))
    return -sum(count/lns * math.log(count/lns, 2) for count in p.values())

def _get_data(self):
        """
        Extracts the session data from cookie.
        """
        cookie = self.adapter.cookies.get(self.name)
        return self._deserialize(cookie) if cookie else {}

def load(path):
    """Loads a pushdb maintained in a properties file at the given path."""
    with open(path, 'r') as props:
      properties = Properties.load(props)
      return PushDb(properties)

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

def __gt__(self, other):
    """Greater than ordering."""
    if not isinstance(other, Key):
      return NotImplemented
    return self.__tuple() > other.__tuple()

def check_key(self, key: str) -> bool:
        """
        Checks if key exists in datastore. True if yes, False if no.

        :param: SHA512 hash key

        :return: whether or key not exists in datastore
        """
        keys = self.get_keys()
        return key in keys

def is_sqlatype_numeric(coltype: Union[TypeEngine, VisitableType]) -> bool:
    """
    Is the SQLAlchemy column type one that inherits from :class:`Numeric`,
    such as :class:`Float`, :class:`Decimal`?
    """
    coltype = _coltype_to_typeengine(coltype)
    return isinstance(coltype, sqltypes.Numeric)

def move_to_start(self, column_label):
        """Move a column to the first in order."""
        self._columns.move_to_end(column_label, last=False)
        return self

def get_available_gpus():
  """
  Returns a list of string names of all available GPUs
  """
  local_device_protos = device_lib.list_local_devices()
  return [x.name for x in local_device_protos if x.device_type == 'GPU']

def disable_insecure_request_warning():
    """Suppress warning about untrusted SSL certificate."""
    import requests
    from requests.packages.urllib3.exceptions import InsecureRequestWarning
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

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

def is_sqlatype_text_over_one_char(
        coltype: Union[TypeEngine, VisitableType]) -> bool:
    """
    Is the SQLAlchemy column type a string type that's more than one character
    long?
    """
    coltype = _coltype_to_typeengine(coltype)
    return is_sqlatype_text_of_length_at_least(coltype, 2)

def as_dict(self):
        """Return all child objects in nested dict."""
        dicts = [x.as_dict for x in self.children]
        return {'{0} {1}'.format(self.name, self.value): dicts}

def uniqued(iterable):
    """Return unique list of ``iterable`` items preserving order.

    >>> uniqued('spameggs')
    ['s', 'p', 'a', 'm', 'e', 'g']
    """
    seen = set()
    return [item for item in iterable if item not in seen and not seen.add(item)]

def is_numeric_dtype(dtype):
    """Return ``True`` if ``dtype`` is a numeric type."""
    dtype = np.dtype(dtype)
    return np.issubsctype(getattr(dtype, 'base', None), np.number)

def gauss_box_model(x, amplitude=1.0, mean=0.0, stddev=1.0, hpix=0.5):
    """Integrate a Gaussian profile."""
    z = (x - mean) / stddev
    z2 = z + hpix / stddev
    z1 = z - hpix / stddev
    return amplitude * (norm.cdf(z2) - norm.cdf(z1))

def cleanup_nodes(doc):
    """
    Remove text nodes containing only whitespace
    """
    for node in doc.documentElement.childNodes:
        if node.nodeType == Node.TEXT_NODE and node.nodeValue.isspace():
            doc.documentElement.removeChild(node)
    return doc

def getCachedDataKey(engineVersionHash, key):
		"""
		Retrieves the cached data value for the specified engine version hash and dictionary key
		"""
		cacheFile = CachedDataManager._cacheFileForHash(engineVersionHash)
		return JsonDataManager(cacheFile).getKey(key)

def get_height_for_line(self, lineno):
        """
        Return the height of the given line.
        (The height that it would take, if this line became visible.)
        """
        if self.wrap_lines:
            return self.ui_content.get_height_for_line(lineno, self.window_width)
        else:
            return 1

def do_last(environment, seq):
    """Return the last item of a sequence."""
    try:
        return next(iter(reversed(seq)))
    except StopIteration:
        return environment.undefined('No last item, sequence was empty.')

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

def is_any_type_set(sett: Set[Type]) -> bool:
    """
    Helper method to check if a set of types is the {AnyObject} singleton

    :param sett:
    :return:
    """
    return len(sett) == 1 and is_any_type(min(sett))

def kernel(self, spread=1):
        """ This will return whatever kind of kernel we want to use.
            Must have signature (ndarray size NxM, ndarray size 1xM) -> ndarray size Nx1
        """
        # TODO: use self.kernel_type to choose function

        def gaussian(data, pixel):
            return mvn.pdf(data, mean=pixel, cov=spread)

        return gaussian

def apply_filters(df, filters):
        """Basic filtering for a dataframe."""
        idx = pd.Series([True]*df.shape[0])
        for k, v in list(filters.items()):
            if k not in df.columns:
                continue
            idx &= (df[k] == v)

        return df.loc[idx]

def build_service_class(metadata):
    """Generate a service class for the service contained in the specified metadata class."""
    i = importlib.import_module(metadata)
    service = i.service
    env = get_jinja_env()
    service_template = env.get_template('service.py.jinja2')
    with open(api_path(service.name.lower()), 'w') as t:
        t.write(service_template.render(service_md=service))

def object_to_json(obj, indent=2):
    """
        transform object to json
    """
    instance_json = json.dumps(obj, indent=indent, ensure_ascii=False, cls=DjangoJSONEncoder)
    return instance_json

def maybeparens(lparen, item, rparen):
    """Wrap an item in optional parentheses, only applying them if necessary."""
    return item | lparen.suppress() + item + rparen.suppress()

def stop_capture(self):
        """Stop listening for output from the stenotype machine."""
        super(Treal, self).stop_capture()
        if self._machine:
            self._machine.close()
        self._stopped()

def dir_path(dir):
    """with dir_path(path) to change into a directory."""
    old_dir = os.getcwd()
    os.chdir(dir)
    yield
    os.chdir(old_dir)

def exit_if_missing_graphviz(self):
        """
        Detect the presence of the dot utility to make a png graph.
        """
        (out, err) = utils.capture_shell("which dot")

        if "dot" not in out:
            ui.error(c.MESSAGES["dot_missing"])

def blueprint_name_to_url(name):
        """ remove the last . in the string it it ends with a .
            for the url structure must follow the flask routing format
            it should be /model/method instead of /model/method/
        """
        if name[-1:] == ".":
            name = name[:-1]
        name = str(name).replace(".", "/")
        return name

def prt_nts(data_nts, prtfmt=None, prt=sys.stdout, nt_fields=None, **kws):
    """Print list of namedtuples into a table using prtfmt."""
    prt_txt(prt, data_nts, prtfmt, nt_fields, **kws)

def update_table_row(self, table, row_idx):
        """Add this instance as a row on a `astropy.table.Table` """
        try:
            table[row_idx]['timestamp'] = self.timestamp
            table[row_idx]['status'] = self.status
        except IndexError:
            print("Index error", len(table), row_idx)

def remove_non_magic_cols(self):
        """
        Remove all non-MagIC columns from all tables.
        """
        for table_name in self.tables:
            table = self.tables[table_name]
            table.remove_non_magic_cols_from_table()

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

def get_column_names(engine: Engine, tablename: str) -> List[str]:
    """
    Get all the database column names for the specified table.
    """
    return [info.name for info in gen_columns_info(engine, tablename)]

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

def load(self):
		"""Load the noise texture data into the current texture unit"""
		glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE16_ALPHA16, 
			self.width, self.width, self.width, 0, GL_LUMINANCE_ALPHA, 
			GL_UNSIGNED_SHORT, ctypes.byref(self.data))

def hidden_cursor():
    """Temporarily hide the terminal cursor."""
    if sys.stdout.isatty():
        _LOGGER.debug('Hiding cursor.')
        print('\x1B[?25l', end='')
        sys.stdout.flush()
    try:
        yield
    finally:
        if sys.stdout.isatty():
            _LOGGER.debug('Showing cursor.')
            print('\n\x1B[?25h', end='')
            sys.stdout.flush()

def after_third_friday(day=None):
    """ check if day is after month's 3rd friday """
    day = day if day is not None else datetime.datetime.now()
    now = day.replace(day=1, hour=16, minute=0, second=0, microsecond=0)
    now += relativedelta.relativedelta(weeks=2, weekday=relativedelta.FR)
    return day > now

def json_get_data(filename):
    """Get data from json file
    """
    with open(filename) as fp:
        json_data = json.load(fp)
        return json_data

    return False

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

def get_var(self, name):
        """ Returns the variable set with the given name.
        """
        for var in self.vars:
            if var.name == name:
                return var
        else:
            raise ValueError

def getheader(self, name, default=None):
        """Returns a given response header."""
        return self.aiohttp_response.headers.get(name, default)

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

def _ram_buffer(self):
        """Setup the RAM buffer from the C++ code."""
        # get the address of the RAM
        address = _LIB.Memory(self._env)
        # create a buffer from the contents of the address location
        buffer_ = ctypes.cast(address, ctypes.POINTER(RAM_VECTOR)).contents
        # create a NumPy array from the buffer
        return np.frombuffer(buffer_, dtype='uint8')

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

def read_sphinx_environment(pth):
    """Read the sphinx environment.pickle file at path `pth`."""

    with open(pth, 'rb') as fo:
        env = pickle.load(fo)
    return env

def make_writeable(filename):
    """
    Make sure that the file is writeable.
    Useful if our source is read-only.
    """
    if not os.access(filename, os.W_OK):
        st = os.stat(filename)
        new_permissions = stat.S_IMODE(st.st_mode) | stat.S_IWUSR
        os.chmod(filename, new_permissions)

def uint32_to_uint8(cls, img):
        """
        Cast uint32 RGB image to 4 uint8 channels.
        """
        return np.flipud(img.view(dtype=np.uint8).reshape(img.shape + (4,)))

def _make_sql_params(self,kw):
        """Make a list of strings to pass to an SQL statement
        from the dictionary kw with Python types"""
        return ['%s=?' %k for k in kw.keys() ]
        for k,v in kw.iteritems():
            vals.append('%s=?' %k)
        return vals

def contains_all(self, array):
        """Test if `array` is an array of real numbers."""
        dtype = getattr(array, 'dtype', None)
        if dtype is None:
            dtype = np.result_type(*array)
        return is_real_dtype(dtype)

def inverse(d):
    """
    reverse the k:v pairs
    """
    output = {}
    for k, v in unwrap(d).items():
        output[v] = output.get(v, [])
        output[v].append(k)
    return output

def flatten(lis):
    """Given a list, possibly nested to any level, return it flattened."""
    new_lis = []
    for item in lis:
        if isinstance(item, collections.Sequence) and not isinstance(item, basestring):
            new_lis.extend(flatten(item))
        else:
            new_lis.append(item)
    return new_lis

def border(self):
        """Region formed by taking border elements.

        :returns: :class:`jicimagelib.region.Region`
        """

        border_array = self.bitmap - self.inner.bitmap
        return Region(border_array)

def index(m, val):
    """
    Return the indices of all the ``val`` in ``m``
    """
    mm = np.array(m)
    idx_tuple = np.where(mm == val)
    idx = idx_tuple[0].tolist()

    return idx

def read_data(file, endian, num=1):
    """
    Read a given number of 32-bits unsigned integers from the given file
    with the given endianness.
    """
    res = struct.unpack(endian + 'L' * num, file.read(num * 4))
    if len(res) == 1:
        return res[0]
    return res

def from_url(url, db=None, **kwargs):
    """
    Returns an active Redis client generated from the given database URL.

    Will attempt to extract the database id from the path url fragment, if
    none is provided.
    """
    from redis.client import Redis
    return Redis.from_url(url, db, **kwargs)

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

def list(self,table, **kparams):
        """
        get a collection of records by table name.
        returns a dict (the json map) for python 3.4
        """
        result = self.table_api_get(table, **kparams)
        return self.to_records(result, table)

def list_rds(region, filter_by_kwargs):
    """List all RDS thingys."""
    conn = boto.rds.connect_to_region(region)
    instances = conn.get_all_dbinstances()
    return lookup(instances, filter_by=filter_by_kwargs)

def _read_stream_for_size(stream, buf_size=65536):
    """Reads a stream discarding the data read and returns its size."""
    size = 0
    while True:
        buf = stream.read(buf_size)
        size += len(buf)
        if not buf:
            break
    return size

def log_leave(event, nick, channel):
	"""
	Log a quit or part event.
	"""
	if channel not in pmxbot.config.log_channels:
		return
	ParticipantLogger.store.log(nick, channel, event.type)

def loads(s, model=None, parser=None):
    """Deserialize s (a str) to a Python object."""
    with StringIO(s) as f:
        return load(f, model=model, parser=parser)

def quaternion_imag(quaternion):
    """Return imaginary part of quaternion.

    >>> quaternion_imag([3, 0, 1, 2])
    array([0., 1., 2.])

    """
    return np.array(quaternion[1:4], dtype=np.float64, copy=True)

def Pyramid(pos=(0, 0, 0), s=1, height=1, axis=(0, 0, 1), c="dg", alpha=1):
    """
    Build a pyramid of specified base size `s` and `height`, centered at `pos`.
    """
    return Cone(pos, s, height, axis, c, alpha, 4)

async def cursor(self) -> Cursor:
        """Create an aiosqlite cursor wrapping a sqlite3 cursor object."""
        return Cursor(self, await self._execute(self._conn.cursor))

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

def get_average_color(colors):
    """Calculate the average color from the list of colors, where each color
    is a 3-tuple of (r, g, b) values.
    """
    c = reduce(color_reducer, colors)
    total = len(colors)
    return tuple(v / total for v in c)

def prin(*args, **kwargs):
    r"""Like ``print``, but a function. I.e. prints out all arguments as
    ``print`` would do. Specify output stream like this::

      print('ERROR', `out="sys.stderr"``).

    """
    print >> kwargs.get('out',None), " ".join([str(arg) for arg in args])

def sort_data(data, cols):
    """Sort `data` rows and order columns"""
    return data.sort_values(cols)[cols + ['value']].reset_index(drop=True)

def read(fname):
    """Quick way to read a file content."""
    content = None
    with open(os.path.join(here, fname)) as f:
        content = f.read()
    return content

def enable_proxy(self, host, port):
        """Enable a default web proxy"""

        self.proxy = [host, _number(port)]
        self.proxy_enabled = True

def tree(string, token=[WORD, POS, CHUNK, PNP, REL, ANCHOR, LEMMA]):
    """ Transforms the output of parse() into a Text object.
        The token parameter lists the order of tags in each token in the input string.
    """
    return Text(string, token)

def calculate_size(name, data_list):
    """ Calculates the request payload size"""
    data_size = 0
    data_size += calculate_size_str(name)
    data_size += INT_SIZE_IN_BYTES
    for data_list_item in data_list:
        data_size += calculate_size_data(data_list_item)
    return data_size

def normal_noise(points):
    """Init a noise variable."""
    return np.random.rand(1) * np.random.randn(points, 1) \
        + random.sample([2, -2], 1)

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

def chop(seq, size):
    """Chop a sequence into chunks of the given size."""
    chunk = lambda i: seq[i:i+size]
    return map(chunk,xrange(0,len(seq),size))

def copy_without_prompts(self):
        """Copy text to clipboard without prompts"""
        text = self.get_selected_text()
        lines = text.split(os.linesep)
        for index, line in enumerate(lines):
            if line.startswith('>>> ') or line.startswith('... '):
                lines[index] = line[4:]
        text = os.linesep.join(lines)
        QApplication.clipboard().setText(text)

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

def insert_slash(string, every=2):
    """insert_slash insert / every 2 char"""
    return os.path.join(string[i:i+every] for i in xrange(0, len(string), every))

def getTypeStr(_type):
  r"""Gets the string representation of the given type.
  """
  if isinstance(_type, CustomType):
    return str(_type)

  if hasattr(_type, '__name__'):
    return _type.__name__

  return ''

def is_running(self):
        """Returns a bool determining if the process is in a running state or
        not

        :rtype: bool

        """
        return self.state in [self.STATE_IDLE, self.STATE_ACTIVE,
                              self.STATE_SLEEPING]

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

def sine_wave(i, frequency=FREQUENCY, framerate=FRAMERATE, amplitude=AMPLITUDE):
    """
    Returns value of a sine wave at a given frequency and framerate
    for a given sample i
    """
    omega = 2.0 * pi * float(frequency)
    sine = sin(omega * (float(i) / float(framerate)))
    return float(amplitude) * sine

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

def _group_dict_set(iterator):
    """Make a dict that accumulates the values for each key in an iterator of doubles.

    :param iter[tuple[A,B]] iterator: An iterator
    :rtype: dict[A,set[B]]
    """
    d = defaultdict(set)
    for key, value in iterator:
        d[key].add(value)
    return dict(d)

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

def search(self, filterstr, attrlist):
        """Query the configured LDAP server."""
        return self._paged_search_ext_s(self.settings.BASE, ldap.SCOPE_SUBTREE, filterstr=filterstr,
                                        attrlist=attrlist, page_size=self.settings.PAGE_SIZE)

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

def get_last_row(dbconn, tablename, n=1, uuid=None):
    """
    Returns the last `n` rows in the table
    """
    return fetch(dbconn, tablename, n, uuid, end=True)

def ratio_and_percentage(current, total, time_remaining):
    """Returns the progress ratio and percentage."""
    return "{} / {} ({}% completed)".format(current, total, int(current / total * 100))

def flatten(nested):
    """ Return a flatten version of the nested argument """
    flat_return = list()

    def __inner_flat(nested,flat):
        for i in nested:
            __inner_flat(i, flat) if isinstance(i, list) else flat.append(i)
        return flat

    __inner_flat(nested,flat_return)

    return flat_return

def set(self, f):
        """Call a function after a delay, unless another function is set
        in the meantime."""
        self.stop()
        self._create_timer(f)
        self.start()

def update_cursor_position(self, line, index):
        """Update cursor position."""
        value = 'Line {}, Col {}'.format(line + 1, index + 1)
        self.set_value(value)

def def_linear(fun):
    """Flags that a function is linear wrt all args"""
    defjvp_argnum(fun, lambda argnum, g, ans, args, kwargs:
                  fun(*subval(args, argnum, g), **kwargs))

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

def read(filename):
    """Read and return `filename` in root dir of project and return string"""
    return codecs.open(os.path.join(__DIR__, filename), 'r').read()

def read_json(location):
    """Open and load JSON from file.

    location (Path): Path to JSON file.
    RETURNS (dict): Loaded JSON content.
    """
    location = ensure_path(location)
    with location.open('r', encoding='utf8') as f:
        return ujson.load(f)

def on_modified(self, event):
        """Function called everytime a new file is modified.

        Args:
            event: Event to process.
        """
        self._logger.debug('Detected modify event on watched path: %s', event.src_path)

        self._process_event(event)

def pull_stream(image):
    """
    Return generator of pull status objects
    """
    return (json.loads(s) for s in _get_docker().pull(image, stream=True))

def hook_focus_events(self):
        """ Install the hooks for focus events.

        This method may be overridden by subclasses as needed.

        """
        widget = self.widget
        widget.focusInEvent = self.focusInEvent
        widget.focusOutEvent = self.focusOutEvent

def extract_log_level_from_environment(k, default):
    """Gets the log level from the environment variable."""
    return LOG_LEVELS.get(os.environ.get(k)) or int(os.environ.get(k, default))

def normalise_string(string):
    """ Strips trailing whitespace from string, lowercases it and replaces
        spaces with underscores
    """
    string = (string.strip()).lower()
    return re.sub(r'\W+', '_', string)

def get_url_args(url):
    """ Returns a dictionary from a URL params """
    url_data = urllib.parse.urlparse(url)
    arg_dict = urllib.parse.parse_qs(url_data.query)
    return arg_dict

def add_index_alias(es, index_name, alias_name):
    """Add index alias to index_name"""

    es.indices.put_alias(index=index_name, name=terms_alias)

def lazy_reverse_binmap(f, xs):
    """
    Same as lazy_binmap, except the parameters are flipped for the binary function
    """
    return (f(y, x) for x, y in zip(xs, xs[1:]))

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

def get_2D_samples_gauss(n, m, sigma, random_state=None):
    """ Deprecated see  make_2D_samples_gauss   """
    return make_2D_samples_gauss(n, m, sigma, random_state=None)

def delistify(x):
    """ A basic slug version of a given parameter list. """
    if isinstance(x, list):
        x = [e.replace("'", "") for e in x]
        return '-'.join(sorted(x))
    return x

def get_package_info(package):
    """Gets the PyPI information for a given package."""
    url = 'https://pypi.python.org/pypi/{}/json'.format(package)
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

def convert_2_utc(self, datetime_, timezone):
        """convert to datetime to UTC offset."""

        datetime_ = self.tz_mapper[timezone].localize(datetime_)
        return datetime_.astimezone(pytz.UTC)

def text_to_bool(value: str) -> bool:
    """
    Tries to convert a text value to a bool. If unsuccessful returns if value is None or not

    :param value: Value to check
    """
    try:
        return bool(strtobool(value))
    except (ValueError, AttributeError):
        return value is not None

def _make_sentence(txt):
    """Make a sentence from a piece of text."""
    #Make sure first letter is capitalized
    txt = txt.strip(' ')
    txt = txt[0].upper() + txt[1:] + '.'
    return txt

def _add(self, codeobj):
        """Add a child (variable) to this object."""
        assert isinstance(codeobj, CodeVariable)
        self.variables.append(codeobj)

def open01(x, limit=1.e-6):
    """Constrain numbers to (0,1) interval"""
    try:
        return np.array([min(max(y, limit), 1. - limit) for y in x])
    except TypeError:
        return min(max(x, limit), 1. - limit)

def top_class(self):
        """reference to a parent class, which contains this class and defined
        within a namespace

        if this class is defined under a namespace, self will be returned"""
        curr = self
        parent = self.parent
        while isinstance(parent, class_t):
            curr = parent
            parent = parent.parent
        return curr

def run_migration(connection, queries, engine):
    """ Apply a migration to the SQL server """

    # Execute query
    with connection.cursor() as cursorMig:
        # Parse statements
        queries = parse_statements(queries, engine)

        for query in queries:
            cursorMig.execute(query)
        connection.commit()

    return True

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

def is_valid(cls, arg):
        """Return True if arg is valid value for the class.  If the string
        value is wrong for the enumeration, the encoding will fail.
        """
        return (isinstance(arg, (int, long)) and (arg >= 0)) or \
            isinstance(arg, basestring)

def _power(ctx, number, power):
    """
    Returns the result of a number raised to a power
    """
    return decimal_pow(conversions.to_decimal(number, ctx), conversions.to_decimal(power, ctx))

def assert_looks_like(first, second, msg=None):
    """ Compare two strings if all contiguous whitespace is coalesced. """
    first = _re.sub("\s+", " ", first.strip())
    second = _re.sub("\s+", " ", second.strip())
    if first != second:
        raise AssertionError(msg or "%r does not look like %r" % (first, second))

def neo(graph: BELGraph, connection: str, password: str):
    """Upload to neo4j."""
    import py2neo
    neo_graph = py2neo.Graph(connection, password=password)
    to_neo4j(graph, neo_graph)

def cli(env, identifier):
    """Delete an image."""

    image_mgr = SoftLayer.ImageManager(env.client)
    image_id = helpers.resolve_id(image_mgr.resolve_ids, identifier, 'image')

    image_mgr.delete_image(image_id)

def hex_color_to_tuple(hex):
    """ convent hex color to tuple
    "#ffffff"   ->  (255, 255, 255)
    "#ffff00ff" ->  (255, 255, 0, 255)
    """
    hex = hex[1:]
    length = len(hex) // 2
    return tuple(int(hex[i*2:i*2+2], 16) for i in range(length))

def family_directory(fonts):
  """Get the path of font project directory."""
  if fonts:
    dirname = os.path.dirname(fonts[0])
    if dirname == '':
      dirname = '.'
    return dirname

def http_request_json(*args, **kwargs):
    """

    See: http_request
    """
    ret, status = http_request(*args, **kwargs)
    return json.loads(ret), status

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

def sprint(text, *colors):
    """Format text with color or other effects into ANSI escaped string."""
    return "\33[{}m{content}\33[{}m".format(";".join([str(color) for color in colors]), RESET, content=text) if IS_ANSI_TERMINAL and colors else text

def are_in_interval(s, l, r, border = 'included'):
        """
        Checks whether all number in the sequence s lie inside the interval formed by
        l and r.
        """
        return numpy.all([IntensityRangeStandardization.is_in_interval(x, l, r, border) for x in s])

def _nth(arr, n):
    """
    Return the nth value of array

    If it is missing return NaN
    """
    try:
        return arr.iloc[n]
    except (KeyError, IndexError):
        return np.nan

def last_modified_date(filename):
    """Last modified timestamp as a UTC datetime"""
    mtime = os.path.getmtime(filename)
    dt = datetime.datetime.utcfromtimestamp(mtime)
    return dt.replace(tzinfo=pytz.utc)

def median(ls):
    """
    Takes a list and returns the median.
    """
    ls = sorted(ls)
    return ls[int(floor(len(ls)/2.0))]

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

def set(cls, color):
        """
        Sets the terminal to the passed color.
        :param color: one of the availabe colors.
        """
        sys.stdout.write(cls.colors.get(color, cls.colors['RESET']))

def stop(self):
        """Stop stream."""
        if self.stream and self.stream.session.state != STATE_STOPPED:
            self.stream.stop()

def _root(self):
        """Attribute referencing the root node of the tree.

        :returns: the root node of the tree containing this instance.
        :rtype: Node
        """
        _n = self
        while _n.parent:
            _n = _n.parent
        return _n

def get(self, queue_get):
        """
        to get states from multiprocessing.queue
        """
        if isinstance(queue_get, (tuple, list)):
            self.result.extend(queue_get)

def comments(tag, limit=0, flags=0, **kwargs):
    """Get comments only."""

    return [comment for comment in cm.CommentsMatch(tag).get_comments(limit)]

def SampleSum(dists, n):
    """Draws a sample of sums from a list of distributions.

    dists: sequence of Pmf or Cdf objects
    n: sample size

    returns: new Pmf of sums
    """
    pmf = MakePmfFromList(RandomSum(dists) for i in xrange(n))
    return pmf

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

def oplot(self, x, y, **kw):
        """generic plotting method, overplotting any existing plot """
        self.panel.oplot(x, y, **kw)

def __delitem__(self, key):
        """Remove a variable from this dataset.
        """
        del self._variables[key]
        self._coord_names.discard(key)

def get_column_definition(self, table, column):
        """Retrieve the column definition statement for a column from a table."""
        # Parse column definitions for match
        for col in self.get_column_definition_all(table):
            if col.strip('`').startswith(column):
                return col.strip(',')

def find(command, on):
    """Find the command usage."""
    output_lines = parse_man_page(command, on)
    click.echo(''.join(output_lines))

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

def set_mlimits(self, row, column, min=None, max=None):
        """Set limits for the point meta (colormap).

        Point meta values outside this range will be clipped.

        :param min: value for start of the colormap.
        :param max: value for end of the colormap.

        """
        subplot = self.get_subplot_at(row, column)
        subplot.set_mlimits(min, max)

def get_code(module):
    """
    Compile and return a Module's code object.
    """
    fp = open(module.path)
    try:
        return compile(fp.read(), str(module.name), 'exec')
    finally:
        fp.close()

def _is_date_data(self, data_type):
        """Private method for determining if a data record is of type DATE."""
        dt = DATA_TYPES[data_type]
        if isinstance(self.data, dt['type']):
            self.type = data_type.upper()
            self.len = None
            return True

def get_property(self, filename):
        """Opens the file and reads the value"""

        with open(self.filepath(filename)) as f:
            return f.read().strip()

def with_defaults(method, nparams, defaults=None):
  """Call method with nparams positional parameters, all non-specified defaults are passed None.

  :method: the method to call
  :nparams: the number of parameters the function expects
  :defaults: the default values to pass in for the last len(defaults) params
  """
  args = [None] * nparams if not defaults else defaults + max(nparams - len(defaults), 0) * [None]
  return method(*args)

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

def get_month_start_end_day():
    """
    Get the month start date a nd end date
    """
    t = date.today()
    n = mdays[t.month]
    return (date(t.year, t.month, 1), date(t.year, t.month, n))

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

def url(self, action, **kwargs):
        """ Construct and return the URL for a specific API service. """
        # TODO : should be static method ?
        return self.URLS['BASE'] % self.URLS[action] % kwargs

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

def isin(self, column, compare_list):
        """
        Returns a boolean list where each elements is whether that element in the column is in the compare_list.

        :param column: single column name, does not work for multiple columns
        :param compare_list: list of items to compare to
        :return: list of booleans
        """
        return [x in compare_list for x in self._data[self._columns.index(column)]]

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

def unique(iterable):
    """Filter out duplicate items from an iterable"""
    seen = set()
    for item in iterable:
        if item not in seen:
            seen.add(item)
            yield item

def save_form(self, request, form, change):
        """
        Super class ordering is important here - user must get saved first.
        """
        OwnableAdmin.save_form(self, request, form, change)
        return DisplayableAdmin.save_form(self, request, form, change)

def stop_at(iterable, idx):
    """Stops iterating before yielding the specified idx."""
    for i, item in enumerate(iterable):
        if i == idx: return
        yield item

def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return F.pad(img, self.padding, self.fill, self.padding_mode)

def valid_uuid(value):
    """ Check if value is a valid UUID. """

    try:
        uuid.UUID(value, version=4)
        return True
    except (TypeError, ValueError, AttributeError):
        return False

def _include_yaml(loader, node):
    """Load another YAML file and embeds it using the !include tag.

    Example:
        device_tracker: !include device_tracker.yaml
    """
    return load_yaml(os.path.join(os.path.dirname(loader.name), node.value))

async def login(
        username: str, password: str, brand: str,
        websession: ClientSession = None) -> API:
    """Log in to the API."""
    api = API(brand, websession)
    await api.authenticate(username, password)
    return api

def index_exists(self, table: str, indexname: str) -> bool:
        """Does an index exist? (Specific to MySQL.)"""
        # MySQL:
        sql = ("SELECT COUNT(*) FROM information_schema.statistics"
               " WHERE table_name=? AND index_name=?")
        row = self.fetchone(sql, table, indexname)
        return True if row[0] >= 1 else False

def get_base_dir():
    """
    Return the base directory
    """
    return os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]

def pytest_runtest_logreport(self, report):
        """Store all test reports for evaluation on finish"""
        rep = report
        res = self.config.hook.pytest_report_teststatus(report=rep)
        cat, letter, word = res
        self.stats.setdefault(cat, []).append(rep)

def get_unique_indices(df, axis=1):
    """

    :param df:
    :param axis:
    :return:
    """
    return dict(zip(df.columns.names, dif.columns.levels))

def c_str(string):
    """"Convert a python string to C string."""
    if not isinstance(string, str):
        string = string.decode('ascii')
    return ctypes.c_char_p(string.encode('utf-8'))

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

def get_least_distinct_words(vocab, topic_word_distrib, doc_topic_distrib, doc_lengths, n=None):
    """
    Order the words from `vocab` by "distinctiveness score" (Chuang et al. 2012) from least to most distinctive.
    Optionally only return the `n` least distinctive words.

    J. Chuang, C. Manning, J. Heer 2012: "Termite: Visualization Techniques for Assessing Textual Topic Models"
    """
    return _words_by_distinctiveness_score(vocab, topic_word_distrib, doc_topic_distrib, doc_lengths, n,
                                           least_to_most=True)

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

def move_page_bottom(self):
        """
        Move the cursor to the last item on the page.
        """
        self.nav.page_index = self.content.range[1]
        self.nav.cursor_index = 0
        self.nav.inverted = True

def glog(x,l = 2):
    """
    Generalised logarithm

    :param x: number
    :param p: number added befor logarithm 

    """
    return np.log((x+np.sqrt(x**2+l**2))/2)/np.log(l)

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

def _tf_squared_euclidean(X, Y):
        """Squared Euclidean distance between the rows of `X` and `Y`.
        """
        return tf.reduce_sum(tf.pow(tf.subtract(X, Y), 2), axis=1)

def circ_permutation(items):
    """Calculate the circular permutation for a given list of items."""
    permutations = []
    for i in range(len(items)):
        permutations.append(items[i:] + items[:i])
    return permutations

def lower_camel_case_from_underscores(string):
    """generate a lower-cased camelCase string from an underscore_string.
    For example: my_variable_name -> myVariableName"""
    components = string.split('_')
    string = components[0]
    for component in components[1:]:
        string += component[0].upper() + component[1:]
    return string

def uniquify_list(L):
    """Same order unique list using only a list compression."""
    return [e for i, e in enumerate(L) if L.index(e) == i]

def replace(s, replace):
    """Replace multiple values in a string"""
    for r in replace:
        s = s.replace(*r)
    return s

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

def write(self, text):
        """Write text. An additional attribute terminator with a value of
           None is added to the logging record to indicate that StreamHandler
           should not add a newline."""
        self.logger.log(self.loglevel, text, extra={'terminator': None})

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

def uniform_noise(points):
    """Init a uniform noise variable."""
    return np.random.rand(1) * np.random.uniform(points, 1) \
        + random.sample([2, -2], 1)

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

def find_model_by_table_name(name):
    """Find a model reference by its table name"""

    for model in ModelBase._decl_class_registry.values():
        if hasattr(model, '__table__') and model.__table__.fullname == name:
            return model
    return None

def compute_capture(args):
    x, y, w, h, params = args
    """Callable function for the multiprocessing pool."""
    return x, y, mandelbrot_capture(x, y, w, h, params)

def main(argv=None):
  """Run a Tensorflow model on the Iris dataset."""
  args = parse_arguments(sys.argv if argv is None else argv)

  tf.logging.set_verbosity(tf.logging.INFO)
  learn_runner.run(
      experiment_fn=get_experiment_fn(args),
      output_dir=args.job_dir)

def sort_by_name(self):
        """Sort list elements by name."""
        super(JSSObjectList, self).sort(key=lambda k: k.name)

def _kw(keywords):
    """Turn list of keywords into dictionary."""
    r = {}
    for k, v in keywords:
        r[k] = v
    return r

def hdf5_to_dict(filepath, group='/'):
    """load the content of an hdf5 file to a dict.

    # TODO: how to split domain_type_dev : parameter : value ?
    """
    if not h5py.is_hdf5(filepath):
        raise RuntimeError(filepath, 'is not a valid HDF5 file.')

    with h5py.File(filepath, 'r') as handler:
        dic = walk_hdf5_to_dict(handler[group])
    return dic

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

def name(self):
        """A unique name for this scraper."""
        return ''.join('_%s' % c if c.isupper() else c for c in self.__class__.__name__).strip('_').lower()

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

def getElementsBy(self, cond: Callable[[Element], bool]) -> NodeList:
        """Get elements in this document which matches condition."""
        return getElementsBy(self, cond)

def _valid_other_type(x, types):
    """
    Do all elements of x have a type from types?
    """
    return all(any(isinstance(el, t) for t in types) for el in np.ravel(x))

def write(self):
        """Write content back to file."""
        with open(self.path, 'w') as file_:
            file_.write(self.content)

def get_body_size(params, boundary):
    """Returns the number of bytes that the multipart/form-data encoding
    of ``params`` will be."""
    size = sum(p.get_size(boundary) for p in MultipartParam.from_params(params))
    return size + len(boundary) + 6

def upcaseTokens(s,l,t):
    """Helper parse action to convert tokens to upper case."""
    return [ tt.upper() for tt in map(_ustr,t) ]

def clear_global(self):
        """Clear only any cached global data.

        """
        vname = self.varname
        logger.debug(f'global clearning {vname}')
        if vname in globals():
            logger.debug('removing global instance var: {}'.format(vname))
            del globals()[vname]

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

def read_large_int(self, bits, signed=True):
        """Reads a n-bits long integer value."""
        return int.from_bytes(
            self.read(bits // 8), byteorder='little', signed=signed)

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

def qsize(self):
        """Return the approximate size of the queue (not reliable!)."""
        self.mutex.acquire()
        n = self._qsize()
        self.mutex.release()
        return n

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

def assign_parent(node: astroid.node_classes.NodeNG) -> astroid.node_classes.NodeNG:
    """return the higher parent which is not an AssignName, Tuple or List node
    """
    while node and isinstance(node, (astroid.AssignName, astroid.Tuple, astroid.List)):
        node = node.parent
    return node

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

def underline(self, msg):
        """Underline the input"""
        return click.style(msg, underline=True) if self.colorize else msg

def map_keys_deep(f, dct):
    """
    Implementation of map that recurses. This tests the same keys at every level of dict and in lists
    :param f: 2-ary function expecting a key and value and returns a modified key
    :param dct: Dict for deep processing
    :return: Modified dct with matching props mapped
    """
    return _map_deep(lambda k, v: [f(k, v), v], dct)

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

def area(x,y):
    """
    Calculate the area of a polygon given as x(...),y(...)
    Implementation of Shoelace formula
    """
    # http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

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

def _release(self):
        """Destroy self since closures cannot be called again."""
        del self.funcs
        del self.variables
        del self.variable_values
        del self.satisfied

def searchlast(self,n=10):
        """Return the last n results (or possibly less if not found). Note that the last results are not necessarily the best ones! Depending on the search type."""            
        solutions = deque([], n)
        for solution in self:
            solutions.append(solution)
        return solutions

def get_resource_attribute(self, attr):
        """Gets the resource attribute if available

        :param attr: Name of the attribute
        :return: Value of the attribute, if set in the resource. None otherwise
        """
        if attr not in self.resource_attributes:
            raise KeyError("%s is not in resource attributes" % attr)

        return self.resource_attributes[attr]

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

def _not_none(items):
    """Whether the item is a placeholder or contains a placeholder."""
    if not isinstance(items, (tuple, list)):
        items = (items,)
    return all(item is not _none for item in items)

def get_file_md5sum(path):
    """Calculate the MD5 hash for a file."""
    with open(path, 'rb') as fh:
        h = str(hashlib.md5(fh.read()).hexdigest())
    return h

def read(self, count=0):
        """ Read """
        return self.f.read(count) if count > 0 else self.f.read()

def _ignore_comments(lines_enum):
    """
    Strips comments and filter empty lines.
    """
    for line_number, line in lines_enum:
        line = COMMENT_RE.sub('', line)
        line = line.strip()
        if line:
            yield line_number, line

def on_train_end(self, logs):
        """ Print training time at end of training """
        duration = timeit.default_timer() - self.train_start
        print('done, took {:.3f} seconds'.format(duration))

def _attrprint(d, delimiter=', '):
    """Print a dictionary of attributes in the DOT format"""
    return delimiter.join(('"%s"="%s"' % item) for item in sorted(d.items()))

def _remove_from_index(index, obj):
    """Removes object ``obj`` from the ``index``."""
    try:
        index.value_map[indexed_value(index, obj)].remove(obj.id)
    except KeyError:
        pass

def __complex__(self):
        """Called to implement the built-in function complex()."""
        if self._t != 99 or self.key != ['re', 'im']:
            return complex(float(self))
        return complex(float(self.re), float(self.im))

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

def watched_extension(extension):
    """Return True if the given extension is one of the watched extensions"""
    for ext in hamlpy.VALID_EXTENSIONS:
        if extension.endswith('.' + ext):
            return True
    return False

def from_years_range(start_year, end_year):
        """Transform a range of years (two ints) to a DateRange object."""
        start = datetime.date(start_year, 1 , 1)
        end = datetime.date(end_year, 12 , 31)
        return DateRange(start, end)

def OnTogglePlay(self, event):
        """Toggles the video status between play and hold"""

        if self.player.get_state() == vlc.State.Playing:
            self.player.pause()
        else:
            self.player.play()

        event.Skip()

def with_args(self, *args, **kwargs):
        """Declares that the double can only be called with the provided arguments.

        :param args: Any positional arguments required for invocation.
        :param kwargs: Any keyword arguments required for invocation.
        """

        self.args = args
        self.kwargs = kwargs
        self.verify_arguments()
        return self

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

def set_font_size(self, size):
        """Convenience method for just changing font size."""
        if self.font.font_size == size:
            pass
        else:
            self.font._set_size(size)

def python_utc_datetime_to_sqlite_strftime_string(
        value: datetime.datetime) -> str:
    """
    Converts a Python datetime to a string literal compatible with SQLite,
    including the millisecond field.
    """
    millisec_str = str(round(value.microsecond / 1000)).zfill(3)
    return value.strftime("%Y-%m-%d %H:%M:%S") + "." + millisec_str

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

def filter_by_ids(original_list, ids_to_filter):
    """Filter a list of dicts by IDs using an id key on each dict."""
    if not ids_to_filter:
        return original_list

    return [i for i in original_list if i['id'] in ids_to_filter]

def get_keys_from_class(cc):
    """Return list of the key property names for a class """
    return [prop.name for prop in cc.properties.values() \
            if 'key' in prop.qualifiers]

def normalize_pattern(pattern):
    """Converts backslashes in path patterns to forward slashes.

    Doesn't normalize regular expressions - they may contain escapes.
    """
    if not (pattern.startswith('RE:') or pattern.startswith('!RE:')):
        pattern = _slashes.sub('/', pattern)
    if len(pattern) > 1:
        pattern = pattern.rstrip('/')
    return pattern

def _session_set(self, key, value):
        """
        Saves a value to session.
        """

        self.session[self._session_key(key)] = value

def find_one(cls, *args, **kwargs):
        """Run a find_one on this model's collection.  The arguments to
        ``Model.find_one`` are the same as to ``pymongo.Collection.find_one``."""
        database, collection = cls._collection_key.split('.')
        return current()[database][collection].find_one(*args, **kwargs)

async def create_websocket_server(sock, filter=None):  # pylint: disable=W0622
    """
    A more low-level form of open_websocket_server.
    You are responsible for closing this websocket.
    """
    ws = Websocket()
    await ws.start_server(sock, filter=filter)
    return ws

def _format_title_string(self, title_string):
        """ format mpv's title """
        return self._title_string_format_text_tag(title_string.replace(self.icy_tokkens[0], self.icy_title_prefix))

def string_to_float_list(string_var):
        """Pull comma separated string values out of a text file and converts them to float list"""
        try:
            return [float(s) for s in string_var.strip('[').strip(']').split(', ')]
        except:
            return [float(s) for s in string_var.strip('[').strip(']').split(',')]

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

def __init__(self):
    """Initializes a filter object."""
    super(FilterObject, self).__init__()
    self._filter_expression = None
    self._matcher = None

def display(self):
        """ Get screen width and height """
        w, h = self.session.window_size()
        return Display(w*self.scale, h*self.scale)

def _replace_none(self, aDict):
        """ Replace all None values in a dict with 'none' """
        for k, v in aDict.items():
            if v is None:
                aDict[k] = 'none'

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

def are_token_parallel(sequences: Sequence[Sized]) -> bool:
    """
    Returns True if all sequences in the list have the same length.
    """
    if not sequences or len(sequences) == 1:
        return True
    return all(len(s) == len(sequences[0]) for s in sequences)

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

def destroy(self):
        """ Destroy the SQLStepQueue tables in the database """
        with self._db_conn() as conn:
            for table_name in self._tables:
                conn.execute('DROP TABLE IF EXISTS %s' % table_name)
        return self

def compute_y(self, coefficients, num_x):
        """ Return calculated y-values for the domain of x-values in [1, num_x]. """
        y_vals = []

        for x in range(1, num_x + 1):
            y = sum([c * x ** i for i, c in enumerate(coefficients[::-1])])
            y_vals.append(y)

        return y_vals

def get_property_as_float(self, name: str) -> float:
        """Return the value of a float property.

        :return: The property value (float).

        Raises exception if property with name doesn't exist.

        .. versionadded:: 1.0

        Scriptable: Yes
        """
        return float(self.__instrument.get_property(name))

def _IsRetryable(error):
  """Returns whether error is likely to be retryable."""
  if not isinstance(error, MySQLdb.OperationalError):
    return False
  if not error.args:
    return False
  code = error.args[0]
  return code in _RETRYABLE_ERRORS

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

def get_value(key, obj, default=missing):
    """Helper for pulling a keyed value off various types of objects"""
    if isinstance(key, int):
        return _get_value_for_key(key, obj, default)
    return _get_value_for_keys(key.split('.'), obj, default)

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

def unit_key_from_name(name):
  """Return a legal python name for the given name for use as a unit key."""
  result = name

  for old, new in six.iteritems(UNIT_KEY_REPLACEMENTS):
    result = result.replace(old, new)

  # Collapse redundant underscores and convert to uppercase.
  result = re.sub(r'_+', '_', result.upper())

  return result

def desc(self):
        """Get a short description of the device."""
        return '{0} (ID: {1}) - {2} - {3}'.format(
            self.name, self.device_id, self.type, self.status)

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

def forget_coords(self):
        """Forget all loaded coordinates."""
        self.w.ntotal.set_text('0')
        self.coords_dict.clear()
        self.redo()

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

def _normalize_abmn(abmn):
    """return a normalized version of abmn
    """
    abmn_2d = np.atleast_2d(abmn)
    abmn_normalized = np.hstack((
        np.sort(abmn_2d[:, 0:2], axis=1),
        np.sort(abmn_2d[:, 2:4], axis=1),
    ))
    return abmn_normalized

def clean(dry_run='n'):
    """Wipes compiled and cached python files. To simulate: pynt clean[dry_run=y]"""
    file_patterns = ['*.pyc', '*.pyo', '*~']
    dir_patterns = ['__pycache__']
    recursive_pattern_delete(project_paths.root, file_patterns, dir_patterns, dry_run=bool(dry_run.lower() == 'y'))

def get_http_method(self, method):
        """Gets the http method that will be called from the requests library"""
        return self.http_methods[method](self.url, **self.http_method_args)

def set_locale(request):
    """Return locale from GET lang param or automatically."""
    return request.query.get('lang', app.ps.babel.select_locale_by_request(request))

def datetime_delta_to_ms(delta):
    """
    Given a datetime.timedelta object, return the delta in milliseconds
    """
    delta_ms = delta.days * 24 * 60 * 60 * 1000
    delta_ms += delta.seconds * 1000
    delta_ms += delta.microseconds / 1000
    delta_ms = int(delta_ms)
    return delta_ms

def ensure_hbounds(self):
        """Ensure the cursor is within horizontal screen bounds."""
        self.cursor.x = min(max(0, self.cursor.x), self.columns - 1)

def we_are_in_lyon():
    """Check if we are on a Lyon machine"""
    import socket
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        return False
    return ip.startswith("134.158.")

def flatten(nested, containers=(list, tuple)):
    """ Flatten a nested list by yielding its scalar items.
    """
    for item in nested:
        if hasattr(item, "next") or isinstance(item, containers):
            for subitem in flatten(item):
                yield subitem
        else:
            yield item

def chunked_list(_list, _chunk_size=50):
    """
    Break lists into small lists for processing:w
    """
    for i in range(0, len(_list), _chunk_size):
        yield _list[i:i + _chunk_size]

def get_file_size(filename):
    """
    Get the file size of a given file

    :param filename: string: pathname of a file
    :return: human readable filesize
    """
    if os.path.isfile(filename):
        return convert_size(os.path.getsize(filename))
    return None

def cache_page(page_cache, page_hash, cache_size):
    """Add a page to the page cache."""
    page_cache.append(page_hash)
    if len(page_cache) > cache_size:
        page_cache.pop(0)

def find_root(self):
        """ Traverse parent refs to top. """
        cmd = self
        while cmd.parent:
            cmd = cmd.parent
        return cmd

def prune(self, n):
        """prune all but the first (=best) n items"""
        if self.minimize:
            self.data = self.data[:n]
        else:
            self.data = self.data[-1 * n:]

def lmx_h1k_f64k():
  """HParams for training languagemodel_lm1b32k_packed.  880M Params."""
  hparams = lmx_base()
  hparams.hidden_size = 1024
  hparams.filter_size = 65536
  hparams.batch_size = 2048
  return hparams

def runcoro(async_function):
    """
    Runs an asynchronous function without needing to use await - useful for lambda

    Args:
        async_function (Coroutine): The asynchronous function to run
    """

    future = _asyncio.run_coroutine_threadsafe(async_function, client.loop)
    result = future.result()
    return result

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

def decode_bytes(string):
    """ Decodes a given base64 string into bytes.

    :param str string: The string to decode
    :return: The decoded bytes
    :rtype: bytes
    """

    if is_string_type(type(string)):
        string = bytes(string, "utf-8")
    return base64.decodebytes(string)

def create_cursor(self, name=None):
        """
        Returns an active connection cursor to the database.
        """
        return Cursor(self.client_connection, self.connection, self.djongo_connection)

def doc_to_html(doc, doc_format="ROBOT"):
    """Convert documentation to HTML"""
    from robot.libdocpkg.htmlwriter import DocToHtml
    return DocToHtml(doc_format)(doc)

def _close_websocket(self):
        """Closes the websocket connection."""
        close_method = getattr(self._websocket, "close", None)
        if callable(close_method):
            asyncio.ensure_future(close_method(), loop=self._event_loop)
        self._websocket = None
        self._dispatch_event(event="close")

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

def safe_mkdir_for(path, clean=False):
  """Ensure that the parent directory for a file is present.

  If it's not there, create it. If it is, no-op.
  """
  safe_mkdir(os.path.dirname(path), clean=clean)

def v_normalize(v):
    """
    Normalizes the given vector.
    
    The vector given may have any number of dimensions.
    """
    vmag = v_magnitude(v)
    return [ v[i]/vmag  for i in range(len(v)) ]

def tokenize(self, s):
        """Return a list of token strings from the given sentence.

        :param string s: The sentence string to tokenize.
        :rtype: iter(str)
        """
        return [s[start:end] for start, end in self.span_tokenize(s)]

def _updateItemComboBoxIndex(self, item, column, num):
        """Callback for comboboxes: notifies us that a combobox for the given item and column has changed"""
        item._combobox_current_index[column] = num
        item._combobox_current_value[column] = item._combobox_option_list[column][num][0]

def _aws_get_instance_by_tag(region, name, tag, raw):
    """Get all instances matching a tag."""
    client = boto3.session.Session().client('ec2', region)
    matching_reservations = client.describe_instances(Filters=[{'Name': tag, 'Values': [name]}]).get('Reservations', [])
    instances = []
    [[instances.append(_aws_instance_from_dict(region, instance, raw))  # pylint: disable=expression-not-assigned
      for instance in reservation.get('Instances')] for reservation in matching_reservations if reservation]
    return instances

def restore_scrollbar_position(self):
        """Restoring scrollbar position after main window is visible"""
        scrollbar_pos = self.get_option('scrollbar_position', None)
        if scrollbar_pos is not None:
            self.explorer.treewidget.set_scrollbar_position(scrollbar_pos)

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

def ask_bool(question: str, default: bool = True) -> bool:
    """Asks a question yes no style"""
    default_q = "Y/n" if default else "y/N"
    answer = input("{0} [{1}]: ".format(question, default_q))
    lower = answer.lower()
    if not lower:
        return default
    return lower == "y"

def parse_scale(x):
    """Splits a "%s:%d" string and returns the string and number.

    :return: A ``(string, int)`` pair extracted from ``x``.

    :raise ValueError: the string ``x`` does not respect the input format.
    """
    match = re.match(r'^(.+?):(\d+)$', x)
    if not match:
        raise ValueError('Invalid scale "%s".' % x)
    return match.group(1), int(match.group(2))

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

def select(self, cmd, *args, **kwargs):
        """ Execute the SQL command and return the data rows as tuples
        """
        self.cursor.execute(cmd, *args, **kwargs)
        return self.cursor.fetchall()

def redirect(view=None, url=None, **kwargs):
    """Redirects to the specified view or url
    """
    if view:
        if url:
            kwargs["url"] = url
        url = flask.url_for(view, **kwargs)
    current_context.exit(flask.redirect(url))

def euclidean(c1, c2):
    """Square of the euclidean distance"""
    diffs = ((i - j) for i, j in zip(c1, c2))
    return sum(x * x for x in diffs)

def _clear(self):
        """
        Helper that clears the composition.
        """
        draw = ImageDraw.Draw(self._background_image)
        draw.rectangle(self._device.bounding_box,
                       fill="black")
        del draw

def imapchain(*a, **kwa):
    """ Like map but also chains the results. """

    imap_results = map( *a, **kwa )
    return itertools.chain( *imap_results )

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

def inject_into_urllib3():
    """
    Monkey-patch urllib3 with SecureTransport-backed SSL-support.
    """
    util.ssl_.SSLContext = SecureTransportContext
    util.HAS_SNI = HAS_SNI
    util.ssl_.HAS_SNI = HAS_SNI
    util.IS_SECURETRANSPORT = True
    util.ssl_.IS_SECURETRANSPORT = True

def runiform(lower, upper, size=None):
    """
    Random uniform variates.
    """
    return np.random.uniform(lower, upper, size)

def _top(self):
        """ g """
        # Goto top of the list
        self.top.body.focus_position = 2 if self.compact is False else 0
        self.top.keypress(self.size, "")

def selectin(table, field, value, complement=False):
    """Select rows where the given field is a member of the given value."""

    return select(table, field, lambda v: v in value,
                  complement=complement)

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

def generate_header(headerfields, oldheader, group_by_field):
    """Returns a header as a list, ready to write to TSV file"""
    fieldtypes = ['peptidefdr', 'peptidepep', 'nopsms', 'proteindata',
                  'precursorquant', 'isoquant']
    return generate_general_header(headerfields, fieldtypes,
                                   peptabledata.HEADER_PEPTIDE, oldheader,
                                   group_by_field)

def update_token_tempfile(token):
    """
    Example of function for token update
    """
    with open(tmp, 'w') as f:
        f.write(json.dumps(token, indent=4))

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

def extract_words(lines):
    """
    Extract from the given iterable of lines the list of words.

    :param lines: an iterable of lines;
    :return: a generator of words of lines.
    """
    for line in lines:
        for word in re.findall(r"\w+", line):
            yield word

def is_number(obj):
    """Check if obj is number."""
    return isinstance(obj, (int, float, np.int_, np.float_))

def apply(f, obj, *args, **kwargs):
    """Apply a function in parallel to each element of the input"""
    return vectorize(f)(obj, *args, **kwargs)

def series_table_row_offset(self, series):
        """
        Return the number of rows preceding the data table for *series* in
        the Excel worksheet.
        """
        title_and_spacer_rows = series.index * 2
        data_point_rows = series.data_point_offset
        return title_and_spacer_rows + data_point_rows

def api_home(request, key=None, hproPk=None):
    """Show the home page for the API with all methods"""

    if not check_api_key(request, key, hproPk):
        return HttpResponseForbidden

    return render_to_response('plugIt/api.html', {}, context_instance=RequestContext(request))

def round_to_float(number, precision):
    """Round a float to a precision"""
    rounded = Decimal(str(floor((number + precision / 2) // precision))
                      ) * Decimal(str(precision))
    return float(rounded)

def wget(url):
    """
    Download the page into a string
    """
    import urllib.parse
    request = urllib.request.urlopen(url)
    filestring = request.read()
    return filestring

def fit_gaussian(x, y, yerr, p0):
    """ Fit a Gaussian to the data """
    try:
        popt, pcov = curve_fit(gaussian, x, y, sigma=yerr, p0=p0, absolute_sigma=True)
    except RuntimeError:
        return [0],[0]
    return popt, pcov

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

def stop(self):
        """Stop the resolver threads.
        """
        with self.lock:
            for dummy in self.threads:
                self.queue.put(None)

def create(self, ami, count, config=None):
        """Create an instance using the launcher."""
        return self.Launcher(config=config).launch(ami, count)

def random_int(self, min=0, max=9999, step=1):
        """
        Returns a random integer between two values.

        :param min: lower bound value (inclusive; default=0)
        :param max: upper bound value (inclusive; default=9999)
        :param step: range step (default=1)
        :returns: random integer between min and max
        """
        return self.generator.random.randrange(min, max + 1, step)

def truncate_table(self, tablename):
        """
        SQLite3 doesn't support direct truncate, so we just use delete here
        """
        self.get(tablename).remove()
        self.db.commit()

def NeuralNetLearner(dataset, sizes):
   """Layered feed-forward network."""

   activations = map(lambda n: [0.0 for i in range(n)], sizes)
   weights = []

   def predict(example):
      unimplemented()

   return predict

def json_iter (path):
    """
    iterator for JSON-per-line in a file pattern
    """
    with open(path, 'r') as f:
        for line in f.readlines():
            yield json.loads(line)

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

def __exit__(self, *args):
        """
        Cleanup any necessary opened files
        """

        if self._output_file_handle:
            self._output_file_handle.close()
            self._output_file_handle = None

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

def _show(self, message, indent=0, enable_verbose=True):  # pragma: no cover
        """Message printer.
        """
        if enable_verbose:
            print("    " * indent + message)

def get_object_or_none(model, *args, **kwargs):
    """
    Like get_object_or_404, but doesn't throw an exception.

    Allows querying for an object that might not exist without triggering
    an exception.
    """
    try:
        return model._default_manager.get(*args, **kwargs)
    except model.DoesNotExist:
        return None

def SGT(self, a, b):
        """Signed greater-than comparison"""
        # http://gavwood.com/paper.pdf
        s0, s1 = to_signed(a), to_signed(b)
        return Operators.ITEBV(256, s0 > s1, 1, 0)

def get_week_start_end_day():
    """
    Get the week start date and end date
    """
    t = date.today()
    wd = t.weekday()
    return (t - timedelta(wd), t + timedelta(6 - wd))

def pformat(object, indent=1, width=80, depth=None):
    """Format a Python object into a pretty-printed representation."""
    return PrettyPrinter(indent=indent, width=width, depth=depth).pformat(object)

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

def abbreviate_dashed(s):
    """Abbreviates each part of string that is delimited by a '-'."""
    r = []
    for part in s.split('-'):
        r.append(abbreviate(part))
    return '-'.join(r)

def is_valid_email(email):
    """
    Check if email is valid
    """
    pattern = re.compile(r'[\w\.-]+@[\w\.-]+[.]\w+')
    return bool(pattern.match(email))

def callable_validator(v: Any) -> AnyCallable:
    """
    Perform a simple check if the value is callable.

    Note: complete matching of argument type hints and return types is not performed
    """
    if callable(v):
        return v

    raise errors.CallableError(value=v)