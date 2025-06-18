def SetCursorPos(x: int, y: int) -> bool:
    """
    SetCursorPos from Win32.
    Set mouse cursor to point x, y.
    x: int.
    y: int.
    Return bool, True if succeed otherwise False.
    """
    return bool(ctypes.windll.user32.SetCursorPos(x, y))

def callJavaFunc(func, *args):
    """ Call Java Function """
    gateway = _get_gateway()
    args = [_py2java(gateway, a) for a in args]
    result = func(*args)
    return _java2py(gateway, result)

def walk_tree(root):
    """Pre-order depth-first"""
    yield root

    for child in root.children:
        for el in walk_tree(child):
            yield el

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

def overlap(intv1, intv2):
    """Overlaping of two intervals"""
    return max(0, min(intv1[1], intv2[1]) - max(intv1[0], intv2[0]))

async def handle(self, record):
        """
        Call the handlers for the specified record.

        This method is used for unpickled records received from a socket, as
        well as those created locally. Logger-level filtering is applied.
        """
        if (not self.disabled) and self.filter(record):
            await self.callHandlers(record)

def _close_socket(self):
        """Shutdown and close the Socket.

        :return:
        """
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except (OSError, socket.error):
            pass
        self.socket.close()

def is_done(self):
        """True if the last two moves were Pass or if the position is at a move
        greater than the max depth."""
        return self.position.is_game_over() or self.position.n >= FLAGS.max_game_length

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

def stop_process(self):
        """
        Stops the child process.
        """
        self._process.terminate()
        if not self._process.waitForFinished(100):
            self._process.kill()

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

def similarity_transformation(rot, mat):
    """ R x M x R^-1 """
    return np.dot(rot, np.dot(mat, np.linalg.inv(rot)))

def transformer(data, label):
    """ data preparation """
    data = mx.image.imresize(data, IMAGE_SIZE, IMAGE_SIZE)
    data = mx.nd.transpose(data, (2, 0, 1))
    data = data.astype(np.float32) / 128.0 - 1
    return data, label

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

def Print(self):
        """Prints the values and freqs/probs in ascending order."""
        for val, prob in sorted(self.d.iteritems()):
            print(val, prob)

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

def slugify(s, delimiter='-'):
    """
    Normalize `s` into ASCII and replace non-word characters with `delimiter`.
    """
    s = unicodedata.normalize('NFKD', to_unicode(s)).encode('ascii', 'ignore').decode('ascii')
    return RE_SLUG.sub(delimiter, s).strip(delimiter).lower()

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

def parse_text_to_dict(self, txt):
        """ 
        takes a string and parses via NLP, ready for mapping
        """
        op = {}
        print('TODO - import NLP, split into verbs / nouns')
        op['nouns'] = txt
        op['verbs'] = txt
        
        return op

def copy(string, **kwargs):
    """Copy given string into system clipboard."""
    window = Tk()
    window.withdraw()
    window.clipboard_clear()
    window.clipboard_append(string)
    window.destroy()
    return

def remove(self, document_id, namespace, timestamp):
        """Removes documents from Solr

        The input is a python dictionary that represents a mongo document.
        """
        self.solr.delete(id=u(document_id),
                         commit=(self.auto_commit_interval == 0))

def generate_user_token(self, user, salt=None):
        """Generates a unique token associated to the user
        """
        return self.token_serializer.dumps(str(user.id), salt=salt)

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

def find_lt(a, x):
    """Find rightmost value less than x"""
    i = bisect.bisect_left(a, x)
    if i:
        return a[i-1]
    raise ValueError

def free(self):
        """Free the underlying C array"""
        if self._ptr is None:
            return
        Gauged.array_free(self.ptr)
        FloatArray.ALLOCATIONS -= 1
        self._ptr = None

def dictmerge(x, y):
    """
    merge two dictionaries
    """
    z = x.copy()
    z.update(y)
    return z

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

def _get_url(url):
    """Retrieve requested URL"""
    try:
        data = HTTP_SESSION.get(url, stream=True)
        data.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise FetcherException(exc)

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

def connect(host, port, username, password):
        """Connect and login to an FTP server and return ftplib.FTP object."""
        # Instantiate ftplib client
        session = ftplib.FTP()

        # Connect to host without auth
        session.connect(host, port)

        # Authenticate connection
        session.login(username, password)
        return session

def seq_include(records, filter_regex):
    """
    Filter any sequences who's seq does not match the filter. Ignore case.
    """
    regex = re.compile(filter_regex)
    for record in records:
        if regex.search(str(record.seq)):
            yield record

def one_hot(x, size, dtype=np.float32):
  """Make a n+1 dim one-hot array from n dim int-categorical array."""
  return np.array(x[..., np.newaxis] == np.arange(size), dtype)

def get_grid_spatial_dimensions(self, variable):
        """Returns (width, height) for the given variable"""

        data = self.open_dataset(self.service).variables[variable.variable]
        dimensions = list(data.dimensions)
        return data.shape[dimensions.index(variable.x_dimension)], data.shape[dimensions.index(variable.y_dimension)]

def focusInEvent(self, event):
        """Reimplement Qt method to send focus change notification"""
        self.focus_changed.emit()
        return super(ControlWidget, self).focusInEvent(event)

def b64_decode(data: bytes) -> bytes:
    """
    :param data: Base 64 encoded data to decode.
    :type data: bytes
    :return: Base 64 decoded data.
    :rtype: bytes
    """
    missing_padding = len(data) % 4
    if missing_padding != 0:
        data += b'=' * (4 - missing_padding)
    return urlsafe_b64decode(data)

def init_app(self, app):
        """Initialize Flask application."""
        app.config.from_pyfile('{0}.cfg'.format(app.name), silent=True)

def to_dict(self):
        """Serialize representation of the table for local caching."""
        return {'schema': self.schema, 'name': self.name, 'columns': [col.to_dict() for col in self._columns],
                'foreign_keys': self.foreign_keys.to_dict(), 'ref_keys': self.ref_keys.to_dict()}

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

def url_read_text(url, verbose=True):
    r"""
    Directly reads text data from url
    """
    data = url_read(url, verbose)
    text = data.decode('utf8')
    return text

def dumped(text, level, indent=2):
    """Put curly brackets round an indented text"""
    return indented("{\n%s\n}" % indented(text, level + 1, indent) or "None", level, indent) + "\n"

def translate_index_to_position(self, index):
        """
        Given an index for the text, return the corresponding (row, col) tuple.
        (0-based. Returns (0, 0) for index=0.)
        """
        # Find start of this line.
        row, row_index = self._find_line_start_index(index)
        col = index - row_index

        return row, col

def WritePythonFile(file_descriptor, package, version, printer):
    """Write the given extended file descriptor to out."""
    _WriteFile(file_descriptor, package, version,
               _ProtoRpcPrinter(printer))

def input_validate_yubikey_secret(data, name='data'):
    """ Input validation for YHSM_YubiKeySecret or string. """
    if isinstance(data, pyhsm.aead_cmd.YHSM_YubiKeySecret):
        data = data.pack()
    return input_validate_str(data, name)

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

def isetdiff_flags(list1, list2):
    """
    move to util_iter
    """
    set2 = set(list2)
    return (item not in set2 for item in list1)

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

def pad_image(arr, max_size=400):
    """Pads an image to a square then resamples to max_size"""
    dim = np.max(arr.shape)
    img = np.zeros((dim, dim, 3), dtype=arr.dtype)
    xl = (dim - arr.shape[0]) // 2
    yl = (dim - arr.shape[1]) // 2
    img[xl:arr.shape[0]+xl, yl:arr.shape[1]+yl, :] = arr
    return resample_image(img, max_size=max_size)

def int_to_date(date):
    """
    Convert an int of form yyyymmdd to a python date object.
    """

    year = date // 10**4
    month = date % 10**4 // 10**2
    day = date % 10**2

    return datetime.date(year, month, day)

def get_ntobj(self):
        """Create namedtuple object with GOEA fields."""
        if self.nts:
            return cx.namedtuple("ntgoea", " ".join(vars(next(iter(self.nts))).keys()))

def change_bgcolor_enable(self, state):
        """
        This is implementet so column min/max is only active when bgcolor is
        """
        self.dataModel.bgcolor(state)
        self.bgcolor_global.setEnabled(not self.is_series and state > 0)

def bytes_base64(x):
    """Turn bytes into base64"""
    if six.PY2:
        return base64.encodestring(x).replace('\n', '')
    return base64.encodebytes(bytes_encode(x)).replace(b'\n', b'')

def makedirs(directory):
    """ Resursively create a named directory. """
    parent = os.path.dirname(os.path.abspath(directory))
    if not os.path.exists(parent):
        makedirs(parent)
    os.mkdir(directory)

def update_dict(obj, dict, attributes):
    """Update dict with fields from obj.attributes.

    :param obj: the object updated into dict
    :param dict: the result dictionary
    :param attributes: a list of attributes belonging to obj
    """
    for attribute in attributes:
        if hasattr(obj, attribute) and getattr(obj, attribute) is not None:
            dict[attribute] = getattr(obj, attribute)

def send(self, topic, *args, **kwargs):
        """
        Appends the prefix to the topic before sendingf
        """
        prefix_topic = self.heroku_kafka.prefix_topic(topic)
        return super(HerokuKafkaProducer, self).send(prefix_topic, *args, **kwargs)

def apply_color_map(name: str, mat: np.ndarray = None):
    """returns an RGB matrix scaled by a matplotlib color map"""
    def apply_map(mat):
        return (cm.get_cmap(name)(_normalize(mat))[:, :, :3] * 255).astype(np.uint8)
        
    return apply_map if mat is None else apply_map(mat)

def validate_type(self, type_):
        """Take an str/unicode `type_` and raise a ValueError if it's not 
        a valid type for the object.
        
        A valid type for a field is a value from the types_set attribute of 
        that field's class. 
        
        """
        if type_ is not None and type_ not in self.types_set:
            raise ValueError('Invalid type for %s:%s' % (self.__class__, type_))

def image_to_texture(image):
    """Converts ``vtkImageData`` to a ``vtkTexture``"""
    vtex = vtk.vtkTexture()
    vtex.SetInputDataObject(image)
    vtex.Update()
    return vtex

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

def delete(self):
        """Remove this object."""
        self._client.remove_object(self._instance, self._bucket, self.name)

def _lookup_parent(self, cls):
        """Lookup a transitive parent object that is an instance
            of a given class."""
        codeobj = self.parent
        while codeobj is not None and not isinstance(codeobj, cls):
            codeobj = codeobj.parent
        return codeobj

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

def check_oneof(**kwargs):
    """Raise ValueError if more than one keyword argument is not none.

    Args:
        kwargs (dict): The keyword arguments sent to the function.

    Returns: None

    Raises:
        ValueError: If more than one entry in kwargs is not none.
    """
    # Sanity check: If no keyword arguments were sent, this is fine.
    if not kwargs:
        return None

    not_nones = [val for val in kwargs.values() if val is not None]
    if len(not_nones) > 1:
        raise ValueError('Only one of {fields} should be set.'.format(
            fields=', '.join(sorted(kwargs.keys())),
        ))

def chunks(iterable, chunk):
    """Yield successive n-sized chunks from an iterable."""
    for i in range(0, len(iterable), chunk):
        yield iterable[i:i + chunk]

def check_hash_key(query_on, key):
    """Only allows == against query_on.hash_key"""
    return (
        isinstance(key, BaseCondition) and
        (key.operation == "==") and
        (key.column is query_on.hash_key)
    )

def imdecode(image_path):
    """Return BGR image read by opencv"""
    import os
    assert os.path.exists(image_path), image_path + ' not found'
    im = cv2.imread(image_path)
    return im

def softmax(attrs, inputs, proto_obj):
    """Softmax function."""
    if 'axis' not in attrs:
        attrs = translation_utils._add_extra_attributes(attrs, {'axis': 1})
    return 'softmax', attrs, inputs

def get_value(self, context):
        """Run python eval on the input string."""
        if self.value:
            return expressions.eval_string(self.value, context)
        else:
            # Empty input raises cryptic EOF syntax err, this more human
            # friendly
            raise ValueError('!py string expression is empty. It must be a '
                             'valid python expression instead.')

def print_fatal_results(results, level=0):
    """Print fatal errors that occurred during validation runs.
    """
    print_level(logger.critical, _RED + "[X] Fatal Error: %s", level, results.error)

def asyncStarCmap(asyncCallable, iterable):
    """itertools.starmap for deferred callables using cooperative multitasking
    """
    results = []
    yield coopStar(asyncCallable, results.append, iterable)
    returnValue(results)

def transformer_ae_a3():
  """Set of hyperparameters."""
  hparams = transformer_ae_base()
  hparams.batch_size = 4096
  hparams.layer_prepostprocess_dropout = 0.3
  hparams.optimizer = "Adafactor"
  hparams.learning_rate = 0.25
  hparams.learning_rate_warmup_steps = 10000
  return hparams

def delete_item(self, item):
        """Delete an object in DynamoDB.

        :param item: Unpacked into kwargs for :func:`boto3.DynamoDB.Client.delete_item`.
        :raises bloop.exceptions.ConstraintViolation: if the condition (or atomic) is not met.
        """
        try:
            self.dynamodb_client.delete_item(**item)
        except botocore.exceptions.ClientError as error:
            handle_constraint_violation(error)

def stats(self):
        """ shotcut to pull out useful info for interactive use """
        printDebug("Classes.....: %d" % len(self.all_classes))
        printDebug("Properties..: %d" % len(self.all_properties))

def camel_to_snake(s: str) -> str:
    """Convert string from camel case to snake case."""

    return CAMEL_CASE_RE.sub(r'_\1', s).strip().lower()

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

def _replace(self, data, replacements):
        """
        Given a list of 2-tuples (find, repl) this function performs all
        replacements on the input and returns the result.
        """
        for find, repl in replacements:
            data = data.replace(find, repl)
        return data

def filtany(entities, **kw):
  """Filter a set of entities based on method return. Use keyword arguments.
  
  Example:
    filtmeth(entities, id='123')
    filtmeth(entities, name='bart')

  Multiple filters are 'OR'.
  """
  ret = set()
  for k,v in kw.items():
    for entity in entities:
      if getattr(entity, k)() == v:
        ret.add(entity)
  return ret

def document(schema):
    """Print a documented teleport version of the schema."""
    teleport_schema = from_val(schema)
    return json.dumps(teleport_schema, sort_keys=True, indent=2)

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

def im2mat(I):
    """Converts and image to matrix (one pixel per line)"""
    return I.reshape((I.shape[0] * I.shape[1], I.shape[2]))

def _get_user_agent(self):
        """Retrieve the request's User-Agent, if available.

        Taken from Flask Login utils.py.
        """
        user_agent = request.headers.get('User-Agent')
        if user_agent:
            user_agent = user_agent.encode('utf-8')
        return user_agent or ''

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

def _check_color_dim(val):
    """Ensure val is Nx(n_col), usually Nx3"""
    val = np.atleast_2d(val)
    if val.shape[1] not in (3, 4):
        raise RuntimeError('Value must have second dimension of size 3 or 4')
    return val, val.shape[1]

def _short_repr(obj):
  """Helper function returns a truncated repr() of an object."""
  stringified = pprint.saferepr(obj)
  if len(stringified) > 200:
    return '%s... (%d bytes)' % (stringified[:200], len(stringified))
  return stringified

def represented_args(args, separator=" "):
    """
    Args:
        args (list | tuple | None): Arguments to represent
        separator (str | unicode): Separator to use

    Returns:
        (str): Quoted as needed textual representation
    """
    result = []
    if args:
        for text in args:
            result.append(quoted(short(text)))
    return separator.join(result)

def setAsApplication(myappid):
    """
    Tells Windows this is an independent application with an unique icon on task bar.

    id is an unique string to identify this application, like: 'mycompany.myproduct.subproduct.version'
    """

    if os.name == 'nt':
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

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

def serialize(obj):
    """Takes a object and produces a dict-like representation

    :param obj: the object to serialize
    """
    if isinstance(obj, list):
        return [serialize(o) for o in obj]
    return GenericSerializer(ModelProviderImpl()).serialize(obj)

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

def int2str(num, radix=10, alphabet=BASE85):
    """helper function for quick base conversions from integers to strings"""
    return NumConv(radix, alphabet).int2str(num)

def get_args(method_or_func):
    """Returns method or function arguments."""
    try:
        # Python 3.0+
        args = list(inspect.signature(method_or_func).parameters.keys())
    except AttributeError:
        # Python 2.7
        args = inspect.getargspec(method_or_func).args
    return args

def rgamma(alpha, beta, size=None):
    """
    Random gamma variates.
    """

    return np.random.gamma(shape=alpha, scale=1. / beta, size=size)

def HttpResponse401(request, template=KEY_AUTH_401_TEMPLATE,
content=KEY_AUTH_401_CONTENT, content_type=KEY_AUTH_401_CONTENT_TYPE):
    """
    HTTP response for not-authorized access (status code 403)
    """
    return AccessFailedResponse(request, template, content, content_type, status=401)

def triangle_normal(a, b, c):
    """Return a vector orthogonal to the given triangle

       Arguments:
         a, b, c  --  three 3D numpy vectors
    """
    normal = np.cross(a - c, b - c)
    norm = np.linalg.norm(normal)
    return normal/norm

def _file_and_exists(val, input_files):
    """Check if an input is a file and exists.

    Checks both locally (staged) and from input files (re-passed but never localized).
    """
    return ((os.path.exists(val) and os.path.isfile(val)) or
            val in input_files)

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

def list_autoscaling_group(region, filter_by_kwargs):
    """List all Auto Scaling Groups."""
    conn = boto.ec2.autoscale.connect_to_region(region)
    groups = conn.get_all_groups()
    return lookup(groups, filter_by=filter_by_kwargs)

def send(self, data):
        """
        Send data to the child process through.
        """
        self.stdin.write(data)
        self.stdin.flush()

def _is_initialized(self, entity):
    """Internal helper to ask if the entity has a value for this Property.

    This returns False if a value is stored but it is None.
    """
    return (not self._required or
            ((self._has_value(entity) or self._default is not None) and
             self._get_value(entity) is not None))

def src2ast(src: str) -> Expression:
    """Return ast.Expression created from source code given in `src`."""
    try:
        return ast.parse(src, mode='eval')
    except SyntaxError:
        raise ValueError("Not a valid expression.") from None

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

def if_(*args):
    """Implements the 'if' operator with support for multiple elseif-s."""
    for i in range(0, len(args) - 1, 2):
        if args[i]:
            return args[i + 1]
    if len(args) % 2:
        return args[-1]
    else:
        return None

def safe_unicode(string):
    """If Python 2, replace non-ascii characters and return encoded string."""
    if not PY3:
        uni = string.replace(u'\u2019', "'")
        return uni.encode('utf-8')
        
    return string

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

def connected_socket(address, timeout=3):
    """ yields a connected socket """
    sock = socket.create_connection(address, timeout)
    yield sock
    sock.close()

def debug(ftn, txt):
    """Used for debugging."""
    if debug_p:
        sys.stdout.write("{0}.{1}:{2}\n".format(modname, ftn, txt))
        sys.stdout.flush()

def _is_target_a_directory(link, rel_target):
	"""
	If creating a symlink from link to a target, determine if target
	is a directory (relative to dirname(link)).
	"""
	target = os.path.join(os.path.dirname(link), rel_target)
	return os.path.isdir(target)

def install_rpm_py():
    """Install RPM Python binding."""
    python_path = sys.executable
    cmd = '{0} install.py'.format(python_path)
    exit_status = os.system(cmd)
    if exit_status != 0:
        raise Exception('Command failed: {0}'.format(cmd))

def reset():
    """Delete the session and clear temporary directories

    """
    shutil.rmtree(session['img_input_dir'])
    shutil.rmtree(session['img_output_dir'])
    session.clear()
    return jsonify(ok='true')

def min_or_none(val1, val2):
    """Returns min(val1, val2) returning None only if both values are None"""
    return min(val1, val2, key=lambda x: sys.maxint if x is None else x)

def sem(inlist):
    """
Returns the estimated standard error of the mean (sx-bar) of the
values in the passed list.  sem = stdev / sqrt(n)

Usage:   lsem(inlist)
"""
    sd = stdev(inlist)
    n = len(inlist)
    return sd / math.sqrt(n)

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

def get_table_width(table):
    """
    Gets the width of the table that would be printed.
    :rtype: ``int``
    """
    columns = transpose_table(prepare_rows(table))
    widths = [max(len(cell) for cell in column) for column in columns]
    return len('+' + '|'.join('-' * (w + 2) for w in widths) + '+')

def handle_m2m_user(self, sender, instance, **kwargs):
    """ Handle many to many relationships for user field """
    self.handle_save(instance.user.__class__, instance.user)

def _convert_to_array(array_like, dtype):
        """
        Convert Matrix attributes which are array-like or buffer to array.
        """
        if isinstance(array_like, bytes):
            return np.frombuffer(array_like, dtype=dtype)
        return np.asarray(array_like, dtype=dtype)

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

def extract_all(zipfile, dest_folder):
    """
    reads the zip file, determines compression
    and unzips recursively until source files 
    are extracted 
    """
    z = ZipFile(zipfile)
    print(z)
    z.extract(dest_folder)

def average_price(quantity_1, price_1, quantity_2, price_2):
    """Calculates the average price between two asset states."""
    return (quantity_1 * price_1 + quantity_2 * price_2) / \
            (quantity_1 + quantity_2)

def aws_to_unix_id(aws_key_id):
    """Converts a AWS Key ID into a UID"""
    uid_bytes = hashlib.sha256(aws_key_id.encode()).digest()[-2:]
    if USING_PYTHON2:
        return 2000 + int(from_bytes(uid_bytes) // 2)
    else:
        return 2000 + (int.from_bytes(uid_bytes, byteorder=sys.byteorder) // 2)

def _environment_variables() -> Dict[str, str]:
    """
    Wraps `os.environ` to filter out non-encodable values.
    """
    return {key: value
            for key, value in os.environ.items()
            if _is_encodable(value)}

def _log_multivariate_normal_density_tied(X, means, covars):
    """Compute Gaussian log-density at X for a tied model."""
    cv = np.tile(covars, (means.shape[0], 1, 1))
    return _log_multivariate_normal_density_full(X, means, cv)

def del_object_from_parent(self):
        """ Delete object from parent object. """
        if self.parent:
            self.parent.objects.pop(self.ref)

def _normalize_numpy_indices(i):
    """Normalize the index in case it is a numpy integer or boolean
    array."""
    if isinstance(i, np.ndarray):
        if i.dtype == bool:
            i = tuple(j.tolist() for j in i.nonzero())
        elif i.dtype == int:
            i = i.tolist()
    return i

def bulk_query(self, query, *multiparams):
        """Bulk insert or update."""

        with self.get_connection() as conn:
            conn.bulk_query(query, *multiparams)

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

def euclidean(x, y):
    """Standard euclidean distance.

    ..math::
        D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2
    return np.sqrt(result)

def pairwise_indices(self):
        """ndarray containing tuples of pairwise indices."""
        return np.array([sig.pairwise_indices for sig in self.values]).T

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

def norm_slash(name):
    """Normalize path slashes."""

    if isinstance(name, str):
        return name.replace('/', "\\") if not is_case_sensitive() else name
    else:
        return name.replace(b'/', b"\\") if not is_case_sensitive() else name

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

def fit_select_best(X, y):
    """
    Selects the best fit of the estimators already implemented by choosing the
    model with the smallest mean square error metric for the trained values.
    """
    models = [fit(X,y) for fit in [fit_linear, fit_quadratic]]
    errors = map(lambda model: mse(y, model.predict(X)), models)

    return min(zip(models, errors), key=itemgetter(1))[0]

def _check_fpos(self, fp_, fpos, offset, block):
        """Check file position matches blocksize"""
        if (fp_.tell() + offset != fpos):
            warnings.warn("Actual "+block+" header size does not match expected")
        return

def __iadd__(self, other_model):
        """Incrementally add the content of another model to this model (+=).

        Copies of all the reactions in the other model are added to this
        model. The objective is the sum of the objective expressions for the
        two models.
        """
        warn('use model.merge instead', DeprecationWarning)
        return self.merge(other_model, objective='sum', inplace=True)

def handle_data(self, data):
        """
        Djeffify data between tags
        """
        if data.strip():
            data = djeffify_string(data)
        self.djhtml += data

def _module_name_from_previous_frame(num_frames_back):
    """
    Returns the module name associated with a frame `num_frames_back` in the
    call stack. This function adds 1 to account for itself, so `num_frames_back`
    should be given relative to the caller.
    """
    frm = inspect.stack()[num_frames_back + 1]
    return inspect.getmodule(frm[0]).__name__

def colorbar(height, length, colormap):
    """Return the channels of a colorbar.
    """
    cbar = np.tile(np.arange(length) * 1.0 / (length - 1), (height, 1))
    cbar = (cbar * (colormap.values.max() - colormap.values.min())
            + colormap.values.min())

    return colormap.colorize(cbar)

def cos_sin_deg(deg):
    """Return the cosine and sin for the given angle
    in degrees, with special-case handling of multiples
    of 90 for perfect right angles
    """
    deg = deg % 360.0
    if deg == 90.0:
        return 0.0, 1.0
    elif deg == 180.0:
        return -1.0, 0
    elif deg == 270.0:
        return 0, -1.0
    rad = math.radians(deg)
    return math.cos(rad), math.sin(rad)

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

def _string_width(self, s):
        """Get width of a string in the current font"""
        s = str(s)
        w = 0
        for i in s:
            w += self.character_widths[i]
        return w * self.font_size / 1000.0

def setup_environment():
    """Set up neccessary environment variables

    This appends all path of sys.path to the python path
    so mayapy will find all installed modules.
    We have to make sure, that we use maya libs instead of
    libs of the virtual env. So we insert all the libs for mayapy
    first.

    :returns: None
    :rtype: None
    :raises: None
    """
    osinter = ostool.get_interface()
    pypath = osinter.get_maya_envpath()
    for p in sys.path:
        pypath = os.pathsep.join((pypath, p))
    os.environ['PYTHONPATH'] = pypath

def dot(self, w):
        """Return the dotproduct between self and another vector."""

        return sum([x * y for x, y in zip(self, w)])

def show(self, imgs, ax=None):
        """ Visualize the persistence image

        """

        ax = ax or plt.gca()

        if type(imgs) is not list:
            imgs = [imgs]

        for i, img in enumerate(imgs):
            ax.imshow(img, cmap=plt.get_cmap("plasma"))
            ax.axis("off")

def feed_eof(self):
        """Send a potentially "ragged" EOF.

        This method will raise an SSL_ERROR_EOF exception if the EOF is
        unexpected.
        """
        self._incoming.write_eof()
        ssldata, appdata = self.feed_ssldata(b'')
        assert appdata == [] or appdata == [b'']

def camelcase2list(s, lower=False):
    """Converts a camelcase string to a list."""
    s = re.findall(r'([A-Z][a-z0-9]+)', s)
    return [w.lower() for w in s] if lower else s

def require(executable: str, explanation: str = "") -> None:
    """
    Ensures that the external tool is available.
    Asserts upon failure.
    """
    assert shutil.which(executable), "Need {!r} on the PATH.{}".format(
        executable, "\n" + explanation if explanation else "")

def copy_user_agent_from_driver(self):
        """ Updates requests' session user-agent with the driver's user agent

        This method will start the browser process if its not already running.
        """
        selenium_user_agent = self.driver.execute_script("return navigator.userAgent;")
        self.headers.update({"user-agent": selenium_user_agent})

def tuple(self, var, cast=None, default=NOTSET):
        """
        :rtype: tuple
        """
        return self.get_value(var, cast=tuple if not cast else (cast,), default=default)

def yesno(prompt):
    """Returns True if user answers 'y' """
    prompt += " [y/n]"
    a = ""
    while a not in ["y", "n"]:
        a = input(prompt).lower()

    return a == "y"

def process_docstring(app, what, name, obj, options, lines):
    """React to a docstring event and append contracts to it."""
    # pylint: disable=unused-argument
    # pylint: disable=too-many-arguments
    lines.extend(_format_contracts(what=what, obj=obj))

def hash_file(fileobj):
    """
    :param fileobj: a file object
    :return: a hash of the file content
    """
    hasher = hashlib.md5()
    buf = fileobj.read(65536)
    while len(buf) > 0:
        hasher.update(buf)
        buf = fileobj.read(65536)
    return hasher.hexdigest()

def dfromdm(dm):
    """Returns distance given distance modulus.
    """
    if np.size(dm)>1:
        dm = np.atleast_1d(dm)
    return 10**(1+dm/5)

def _relative_frequency(self, word):
		"""Computes the log relative frequency for a word form"""

		count = self.type_counts.get(word, 0)
		return math.log(count/len(self.type_counts)) if count > 0 else 0

def information(filename):
    """Returns the file exif"""
    check_if_this_file_exist(filename)
    filename = os.path.abspath(filename)
    result = get_json(filename)
    result = result[0]
    return result

def _resizeColumnsToContents(self, header, data, limit_ms):
        """Resize all the colummns to its contents."""
        max_col = data.model().columnCount()
        if limit_ms is None:
            max_col_ms = None
        else:
            max_col_ms = limit_ms / max(1, max_col)
        for col in range(max_col):
            self._resizeColumnToContents(header, data, col, max_col_ms)

def stylize(text, styles, reset=True):
    """conveniently styles your text as and resets ANSI codes at its end."""
    terminator = attr("reset") if reset else ""
    return "{}{}{}".format("".join(styles), text, terminator)

def _read_json_file(self, json_file):
        """ Helper function to read JSON file as OrderedDict """

        self.log.debug("Reading '%s' JSON file..." % json_file)

        with open(json_file, 'r') as f:
            return json.load(f, object_pairs_hook=OrderedDict)

def _replace_nan(a, val):
    """
    replace nan in a by val, and returns the replaced array and the nan
    position
    """
    mask = isnull(a)
    return where_method(val, mask, a), mask

def format_float(value): # not used
    """Modified form of the 'g' format specifier.
    """
    string = "{:g}".format(value).replace("e+", "e")
    string = re.sub("e(-?)0*(\d+)", r"e\1\2", string)
    return string

def parent(self, index):
        """Return the index of the parent for a given index of the
        child. Unfortunately, the name of the method has to be parent,
        even though a more verbose name like parentIndex, would avoid
        confusion about what parent actually is - an index or an item.
        """
        childItem = self.item(index)
        parentItem = childItem.parent

        if parentItem == self.rootItem:
            parentIndex = QModelIndex()
        else:
            parentIndex = self.createIndex(parentItem.row(), 0, parentItem)

        return parentIndex

def build_parser():
    """Build the script's argument parser."""

    parser = argparse.ArgumentParser(description="The IOTile task supervisor")
    parser.add_argument('-c', '--config', help="config json with options")
    parser.add_argument('-v', '--verbose', action="count", default=0, help="Increase logging verbosity")

    return parser

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

def sed(match, replacement, path, modifiers=""):
    """
    Perform sed text substitution.
    """
    cmd = "sed -r -i 's/%s/%s/%s' %s" % (match, replacement, modifiers, path)

    process = Subprocess(cmd, shell=True)
    ret, out, err = process.run(timeout=60)
    if ret:
        raise SubprocessError("Sed command failed!")

def _remove_dict_keys_with_value(dict_, val):
  """Removes `dict` keys which have have `self` as value."""
  return {k: v for k, v in dict_.items() if v is not val}

def _fill(self):
    """Advance the iterator without returning the old head."""
    try:
      self._head = self._iterable.next()
    except StopIteration:
      self._head = None

def get_cube(name):
    """ Load the named cube from the current registered ``CubeManager``. """
    manager = get_manager()
    if not manager.has_cube(name):
        raise NotFound('No such cube: %r' % name)
    return manager.get_cube(name)

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

def itemlist(item, sep, suppress_trailing=True):
    """Create a list of items seperated by seps."""
    return condense(item + ZeroOrMore(addspace(sep + item)) + Optional(sep.suppress() if suppress_trailing else sep))

def clean_all_buckets(self):
        """
        Removes all buckets from all hashes and their content.
        """
        bucket_keys = self.redis_object.keys(pattern='nearpy_*')
        if len(bucket_keys) > 0:
            self.redis_object.delete(*bucket_keys)

def compile(expr, params=None):
    """
    Force compilation of expression for the SQLite target
    """
    from ibis.sql.alchemy import to_sqlalchemy

    return to_sqlalchemy(expr, dialect.make_context(params=params))

def replace_one(self, replacement):
        """Replace one entire document matching the selector criteria.

        :Parameters:
          - `replacement` (dict): the replacement document
        """
        self.__bulk.add_replace(self.__selector, replacement, upsert=True,
                                collation=self.__collation)

def _add_default_arguments(parser):
    """Add the default arguments to the parser.

    :param argparse.ArgumentParser parser: The argument parser

    """
    parser.add_argument('-c', '--config', action='store', dest='config',
                        help='Path to the configuration file')
    parser.add_argument('-f', '--foreground', action='store_true', dest='foreground',
                        help='Run the application interactively')

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

def _read_section(self):
        """Read and return an entire section"""
        lines = [self._last[self._last.find(":")+1:]]
        self._last = self._f.readline()
        while len(self._last) > 0 and len(self._last[0].strip()) == 0:
            lines.append(self._last)
            self._last = self._f.readline()
        return lines

def set_empty(self, row, column):
        """Keep one of the subplots completely empty.

        :param row,column: specify the subplot.

        """
        subplot = self.get_subplot_at(row, column)
        subplot.set_empty()

def process_kill(pid, sig=None):
    """Send signal to process.
    """
    sig = sig or signal.SIGTERM
    os.kill(pid, sig)

def check_dependencies_remote(args):
    """
    Invoke this command on a remote Python.
    """
    cmd = [args.python, '-m', 'depends', args.requirement]
    env = dict(PYTHONPATH=os.path.dirname(__file__))
    return subprocess.check_call(cmd, env=env)

def read(*args):
    """Reads complete file contents."""
    return io.open(os.path.join(HERE, *args), encoding="utf-8").read()

def iprotate(l, steps=1):
    r"""Like rotate, but modifies `l` in-place.

    >>> l = [1,2,3]
    >>> iprotate(l) is l
    True
    >>> l
    [2, 3, 1]
    >>> iprotate(iprotate(l, 2), -3)
    [1, 2, 3]

    """
    if len(l):
        steps %= len(l)
        if steps:
            firstPart = l[:steps]
            del l[:steps]
            l.extend(firstPart)
    return l

def p_if_statement_2(self, p):
        """if_statement : IF LPAREN expr RPAREN statement ELSE statement"""
        p[0] = ast.If(predicate=p[3], consequent=p[5], alternative=p[7])

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

def getEventTypeNameFromEnum(self, eType):
        """returns the name of an EVREvent enum value"""

        fn = self.function_table.getEventTypeNameFromEnum
        result = fn(eType)
        return result

def indent(text: str, num: int = 2) -> str:
    """Indent a piece of text."""
    lines = text.splitlines()
    return "\n".join(indent_iterable(lines, num=num))

def get_input(input_func, input_str):
    """
    Get input from the user given an input function and an input string
    """
    val = input_func("Please enter your {0}: ".format(input_str))
    while not val or not len(val.strip()):
        val = input_func("You didn't enter a valid {0}, please try again: ".format(input_str))
    return val

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

def isbinary(*args):
    """Checks if value can be part of binary/bitwise operations."""
    return all(map(lambda c: isnumber(c) or isbool(c), args))

def get_element_with_id(self, id):
        """Return the element with the specified ID."""
        # Should we maintain a hashmap of ids to make this more efficient? Probably overkill.
        # TODO: Elements can contain nested elements (captions, footnotes, table cells, etc.)
        return next((el for el in self.elements if el.id == id), None)

def teardown_test(self, context):
        """
        Tears down the Django test
        """
        context.test.tearDownClass()
        context.test._post_teardown(run=True)
        del context.test

def timer():
    """
    Timer used for calculate time elapsed
    """
    if sys.platform == "win32":
        default_timer = time.clock
    else:
        default_timer = time.time

    return default_timer()

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

def read_string_from_file(path, encoding="utf8"):
  """
  Read entire contents of file into a string.
  """
  with codecs.open(path, "rb", encoding=encoding) as f:
    value = f.read()
  return value

def add_arrow(self, x1, y1, x2, y2, **kws):
        """add arrow to plot"""
        self.panel.add_arrow(x1, y1, x2, y2, **kws)

def hms_to_seconds(time_string):
    """
    Converts string 'hh:mm:ss.ssssss' as a float
    """
    s = time_string.split(':')
    hours = int(s[0])
    minutes = int(s[1])
    secs = float(s[2])
    return hours * 3600 + minutes * 60 + secs

def remove(self, entry):
        """Removes an entry"""
        try:
            list = self.cache[entry.key]
            list.remove(entry)
        except:
            pass

def isString(s):
    """Convenience method that works with all 2.x versions of Python
    to determine whether or not something is stringlike."""
    try:
        return isinstance(s, unicode) or isinstance(s, basestring)
    except NameError:
        return isinstance(s, str)

def get_date_field(datetimes, field):
    """Adapted from pandas.tslib.get_date_field"""
    return np.array([getattr(date, field) for date in datetimes])

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

def growthfromrange(rangegrowth, startdate, enddate):
    """
    Annual growth given growth from start date to end date.
    """
    _yrs = (pd.Timestamp(enddate) - pd.Timestamp(startdate)).total_seconds() /\
            dt.timedelta(365.25).total_seconds()
    return yrlygrowth(rangegrowth, _yrs)

def as_list(self):
        """Return all child objects in nested lists of strings."""
        return [self.name, self.value, [x.as_list for x in self.children]]

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

def success_response(**data):
    """Return a generic success response."""
    data_out = {}
    data_out["status"] = "success"
    data_out.update(data)
    js = dumps(data_out, default=date_handler)
    return Response(js, status=200, mimetype="application/json")

def log_all(self, file):
        """Log all data received from RFLink to file."""
        global rflink_log
        if file == None:
            rflink_log = None
        else:
            log.debug('logging to: %s', file)
            rflink_log = open(file, 'a')

def _split_python(python):
    """Split Python source into chunks.

    Chunks are separated by at least two return lines. The break must not
    be followed by a space. Also, long Python strings spanning several lines
    are not splitted.

    """
    python = _preprocess(python)
    if not python:
        return []
    lexer = PythonSplitLexer()
    lexer.read(python)
    return lexer.chunks

def _load_data(filepath):
  """Loads the images and latent values into Numpy arrays."""
  with h5py.File(filepath, "r") as h5dataset:
    image_array = np.array(h5dataset["images"])
    # The 'label' data set in the hdf5 file actually contains the float values
    # and not the class labels.
    values_array = np.array(h5dataset["labels"])
  return image_array, values_array

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

def returned(n):
	"""Generate a random walk and return True if the walker has returned to
	the origin after taking `n` steps.
	"""
	## `takei` yield lazily so we can short-circuit and avoid computing the rest of the walk
	for pos in randwalk() >> drop(1) >> takei(xrange(n-1)):
		if pos == Origin:
			return True
	return False

def intToBin(i):
    """ Integer to two bytes """
    # divide in two parts (bytes)
    i1 = i % 256
    i2 = int(i / 256)
    # make string (little endian)
    return i.to_bytes(2, byteorder='little')

def read_uint(data, start, length):
    """Extract a uint from a position in a sequence."""
    return int.from_bytes(data[start:start+length], byteorder='big')

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

def define_macro(self, name, themacro):
        """Define a new macro

        Parameters
        ----------
        name : str
            The name of the macro.
        themacro : str or Macro
            The action to do upon invoking the macro.  If a string, a new
            Macro object is created by passing the string to it.
        """

        from IPython.core import macro

        if isinstance(themacro, basestring):
            themacro = macro.Macro(themacro)
        if not isinstance(themacro, macro.Macro):
            raise ValueError('A macro must be a string or a Macro instance.')
        self.user_ns[name] = themacro

def get_all_attributes(klass_or_instance):
    """Get all attribute members (attribute, property style method).
    """
    pairs = list()
    for attr, value in inspect.getmembers(
            klass_or_instance, lambda x: not inspect.isroutine(x)):
        if not (attr.startswith("__") or attr.endswith("__")):
            pairs.append((attr, value))
    return pairs

def cast_bytes(s, encoding=None):
    """Source: https://github.com/ipython/ipython_genutils"""
    if not isinstance(s, bytes):
        return encode(s, encoding)
    return s

def matches(self, s):
    """Whether the pattern matches anywhere in the string s."""
    regex_matches = self.compiled_regex.search(s) is not None
    return not regex_matches if self.inverted else regex_matches

def relpath(self):
        """ Return this path as a relative path,
        based from the current working directory.
        """
        cwd = self.__class__(os.getcwdu())
        return cwd.relpathto(self)

def remove_node(self, node):
        """ Remove a node from this network. """
        if _debug: Network._debug("remove_node %r", node)

        self.nodes.remove(node)
        node.lan = None

def _generate(self):
        """Parses a file or directory of files into a set of ``Document`` objects."""
        doc_count = 0
        for fp in self.all_files:
            for doc in self._get_docs_for_path(fp):
                yield doc
                doc_count += 1
                if doc_count >= self.max_docs:
                    return

def endline_semicolon_check(self, original, loc, tokens):
        """Check for semicolons at the end of lines."""
        return self.check_strict("semicolon at end of line", original, loc, tokens)

def Stop(self):
    """Stops the process status RPC server."""
    self._Close()

    if self._rpc_thread.isAlive():
      self._rpc_thread.join()
    self._rpc_thread = None

def _log_response(response):
    """Log out information about a ``Request`` object.

    After calling ``requests.request`` or one of its convenience methods, the
    object returned can be passed to this method. If done, information about
    the object returned is logged.

    :return: Nothing is returned.

    """
    message = u'Received HTTP {0} response: {1}'.format(
        response.status_code,
        response.text
    )
    if response.status_code >= 400:  # pragma: no cover
        logger.warning(message)
    else:
        logger.debug(message)

def tokenize(string):
    """Match and yield all the tokens of the input string."""
    for match in TOKENS_REGEX.finditer(string):
        yield Token(match.lastgroup, match.group().strip(), match.span())

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

def test_value(self, value):
        """Test if value is an instance of float."""
        if not isinstance(value, float):
            raise ValueError('expected float value: ' + str(type(value)))

def tsv_escape(x: Any) -> str:
    """
    Escape data for tab-separated value (TSV) format.
    """
    if x is None:
        return ""
    x = str(x)
    return x.replace("\t", "\\t").replace("\n", "\\n")

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

def validate(self, value, model_instance, **kwargs):
        """This follows the validate rules for choices_form_class field used.
        """
        self.get_choices_form_class().validate(value, model_instance, **kwargs)

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

def assert_error(text, check, n=1):
    """Assert that text has n errors of type check."""
    assert_error.description = "No {} error for '{}'".format(check, text)
    assert(check in [error[0] for error in lint(text)])

def last_day(year=_year, month=_month):
    """
    get the current month's last day
    :param year:  default to current year
    :param month:  default to current month
    :return: month's last day
    """
    last_day = calendar.monthrange(year, month)[1]
    return datetime.date(year=year, month=month, day=last_day)

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

def simple_eq(one: Instance, two: Instance, attrs: List[str]) -> bool:
    """
    Test if two objects are equal, based on a comparison of the specified
    attributes ``attrs``.
    """
    return all(getattr(one, a) == getattr(two, a) for a in attrs)

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

def getOffset(self, loc):
        """ Returns the offset between the given point and this point """
        return Location(loc.x - self.x, loc.y - self.y)

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

def combine(self, a, b):
        """A generator that combines two iterables."""

        for l in (a, b):
            for x in l:
                yield x

def read_text_from_file(path: str) -> str:
    """ Reads text file contents """
    with open(path) as text_file:
        content = text_file.read()

    return content

def visit_Name(self, node):
        """ Get range for parameters for examples or false branching. """
        return self.add(node, self.result[node.id])

def check_create_folder(filename):
    """Check if the folder exisits. If not, create the folder"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

def compare(self, first, second):
        """
        Case in-sensitive comparison of two strings.
        Required arguments:
        * first - The first string to compare.
        * second - The second string to compare.
        """
        if first.lower() == second.lower():
            return True
        else:
            return False

def _initialize_id(self):
        """Initializes the id of the instance."""
        self.id = str(self.db.incr(self._key['id']))

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

def _add_pos1(token):
    """
    Adds a 'pos1' element to a frog token.
    """
    result = token.copy()
    result['pos1'] = _POSMAP[token['pos'].split("(")[0]]
    return result

def setwinsize(self, rows, cols):
        """Set the terminal window size of the child tty.
        """
        self._winsize = (rows, cols)
        self.pty.set_size(cols, rows)

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

def csvtolist(inputstr):
    """ converts a csv string into a list """
    reader = csv.reader([inputstr], skipinitialspace=True)
    output = []
    for r in reader:
        output += r
    return output

def colorize(string, color, *args, **kwargs):
    """
    Implements string formatting along with color specified in colorama.Fore
    """
    string = string.format(*args, **kwargs)
    return color + string + colorama.Fore.RESET

def urlencoded(body, charset='ascii', **kwargs):
    """Converts query strings into native Python objects"""
    return parse_query_string(text(body, charset=charset), False)

def availability_pdf() -> bool:
    """
    Is a PDF-to-text tool available?
    """
    pdftotext = tools['pdftotext']
    if pdftotext:
        return True
    elif pdfminer:
        log.warning("PDF conversion: pdftotext missing; "
                    "using pdfminer (less efficient)")
        return True
    else:
        return False

def dict_hash(dct):
    """Return a hash of the contents of a dictionary"""
    dct_s = json.dumps(dct, sort_keys=True)

    try:
        m = md5(dct_s)
    except TypeError:
        m = md5(dct_s.encode())

    return m.hexdigest()

def _clone_properties(self):
    """Internal helper to clone self._properties if necessary."""
    cls = self.__class__
    if self._properties is cls._properties:
      self._properties = dict(cls._properties)

def find(self, *args, **kwargs):
        """Same as :meth:`pymongo.collection.Collection.find`, except
        it returns the right document class.
        """
        return Cursor(self, *args, wrap=self.document_class, **kwargs)

def close(self):
        """Closes this response."""
        if self._connection:
            self._connection.close()
        self._response.close()

def __init__(self, filename, formatting_info=False, handle_ambiguous_date=None):
    """Initialize the ExcelWorkbook instance."""
    super().__init__(filename)
    self.workbook = xlrd.open_workbook(self.filename, formatting_info=formatting_info)
    self.handle_ambiguous_date = handle_ambiguous_date

def fast_exit(code):
    """Exit without garbage collection, this speeds up exit by about 10ms for
    things like bash completion.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)

def delayed_close(self):
        """ Delayed close - won't close immediately, but on the next reactor
        loop. """
        self.state = SESSION_STATE.CLOSING
        reactor.callLater(0, self.close)

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

def accel_next(self, *args):
        """Callback to go to the next tab. Called by the accel key.
        """
        if self.get_notebook().get_current_page() + 1 == self.get_notebook().get_n_pages():
            self.get_notebook().set_current_page(0)
        else:
            self.get_notebook().next_page()
        return True

def minus(*args):
    """Also, converts either to ints or to floats."""
    if len(args) == 1:
        return -to_numeric(args[0])
    return to_numeric(args[0]) - to_numeric(args[1])

def _is_valid_url(self, url):
        """Callback for is_valid_url."""
        try:
            r = requests.head(url, proxies=self.proxy_servers)
            value = r.status_code in [200]
        except Exception as error:
            logger.error(str(error))
            value = False

        return value

def get_previous(self):
        """Get the billing cycle prior to this one. May return None"""
        return BillingCycle.objects.filter(date_range__lt=self.date_range).order_by('date_range').last()

def recursively_update(d, d2):
  """dict.update but which merges child dicts (dict2 takes precedence where there's conflict)."""
  for k, v in d2.items():
    if k in d:
      if isinstance(v, dict):
        recursively_update(d[k], v)
        continue
    d[k] = v

def query_proc_row(procname, args=(), factory=None):
    """
    Execute a stored procedure. Returns the first row of the result set,
    or `None`.
    """
    for row in query_proc(procname, args, factory):
        return row
    return None

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

def get_cantons(self):
        """
        Return the list of unique cantons, sorted by name.
        """
        return sorted(list(set([
            location.canton for location in self.get_locations().values()
        ])))

def _get_closest_week(self, metric_date):
        """
        Gets the closest monday to the date provided.
        """
        #find the offset to the closest monday
        days_after_monday = metric_date.isoweekday() - 1

        return metric_date - datetime.timedelta(days=days_after_monday)

def get_element_attribute_or_empty(element, attribute_name):
    """

    Args:
        element (element): The xib's element.
        attribute_name (str): The desired attribute's name.

    Returns:
        The attribute's value, or an empty str if none exists.

    """
    return element.attributes[attribute_name].value if element.hasAttribute(attribute_name) else ""

def match_paren(self, tokens, item):
        """Matches a paren."""
        match, = tokens
        return self.match(match, item)

def calculate_size(name, replace_existing_values):
    """ Calculates the request payload size"""
    data_size = 0
    data_size += calculate_size_str(name)
    data_size += BOOLEAN_SIZE_IN_BYTES
    return data_size

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

def isdir(s):
    """Return true if the pathname refers to an existing directory."""
    try:
        st = os.stat(s)
    except os.error:
        return False
    return stat.S_ISDIR(st.st_mode)

def extract_table_names(query):
    """ Extract table names from an SQL query. """
    # a good old fashioned regex. turns out this worked better than actually parsing the code
    tables_blocks = re.findall(r'(?:FROM|JOIN)\s+(\w+(?:\s*,\s*\w+)*)', query, re.IGNORECASE)
    tables = [tbl
              for block in tables_blocks
              for tbl in re.findall(r'\w+', block)]
    return set(tables)

def returns(self):
        """The return type for this method in a JSON-compatible format.

        This handles the special case of ``None`` which allows ``type(None)`` also.

        :rtype: str | None
        """
        return_type = self.signature.return_type
        none_type = type(None)
        if return_type is not None and return_type is not none_type:
            return return_type.__name__

def timestamp_to_datetime(cls, dt, dt_format=DATETIME_FORMAT):
        """Convert unix timestamp to human readable date/time string"""
        return cls.convert_datetime(cls.get_datetime(dt), dt_format=dt_format)

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

def cover(session):
    """Run the final coverage report.
    This outputs the coverage report aggregating coverage from the unit
    test runs (not system test runs), and then erases coverage data.
    """
    session.interpreter = 'python3.6'
    session.install('coverage', 'pytest-cov')
    session.run('coverage', 'report', '--show-missing', '--fail-under=100')
    session.run('coverage', 'erase')

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

def as_tuple(self, value):
        """Utility function which converts lists to tuples."""
        if isinstance(value, list):
            value = tuple(value)
        return value

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

def close(self):
        """Close port."""
        os.close(self.in_d)
        os.close(self.out_d)

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

def parse(self):
        """
        Parses format string looking for substitutions

        This method is responsible for returning a list of fields (as strings)
        to include in all log messages.
        """
        standard_formatters = re.compile(r'\((.+?)\)', re.IGNORECASE)
        return standard_formatters.findall(self._fmt)

def string_presenter(self, dumper, data):
    """Presenter to force yaml.dump to use multi-line string style."""
    if '\n' in data:
      return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    else:
      return dumper.represent_scalar('tag:yaml.org,2002:str', data)

def newest_file(file_iterable):
  """
  Returns the name of the newest file given an iterable of file names.

  """
  return max(file_iterable, key=lambda fname: os.path.getmtime(fname))

def _get_name(self, key):
        """ get display name for a key, or mangle for display """
        if key in self.display_names:
            return self.display_names[key]

        return key.capitalize()

def generator_to_list(fn):
    """This decorator is for flat_list function.
    It converts returned generator to list.
    """
    def wrapper(*args, **kw):
        return list(fn(*args, **kw))
    return wrapper

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

def email_user(self, subject, message, from_email=None):
        """ Send an email to this User."""
        send_mail(subject, message, from_email, [self.email])

def stringc(text, color):
    """
    Return a string with terminal colors.
    """
    if has_colors:
        text = str(text)

        return "\033["+codeCodes[color]+"m"+text+"\033[0m"
    else:
        return text

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

def get_random_id(length):
    """Generate a random, alpha-numerical id."""
    alphabet = string.ascii_uppercase + string.ascii_lowercase + string.digits
    return ''.join(random.choice(alphabet) for _ in range(length))

def table_width(self):
        """Return the width of the table including padding and borders."""
        outer_widths = max_dimensions(self.table_data, self.padding_left, self.padding_right)[2]
        outer_border = 2 if self.outer_border else 0
        inner_border = 1 if self.inner_column_border else 0
        return table_width(outer_widths, outer_border, inner_border)

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

def plot_pauli_transfer_matrix(self, ax):
        """
        Plot the elements of the Pauli transfer matrix.

        :param matplotlib.Axes ax: A matplotlib Axes object to plot into.
        """
        title = "Estimated process"
        ut.plot_pauli_transfer_matrix(self.r_est, ax, self.pauli_basis.labels, title)

def __init__(self, encoding='utf-8'):
    """Initializes an stdin input reader.

    Args:
      encoding (Optional[str]): input encoding.
    """
    super(StdinInputReader, self).__init__(sys.stdin, encoding=encoding)

def _list_available_rest_versions(self):
        """Return a list of the REST API versions supported by the array"""
        url = "https://{0}/api/api_version".format(self._target)

        data = self._request("GET", url, reestablish_session=False)
        return data["version"]

def exists(self, path):
        """
        Returns true if the path exists and false otherwise.
        """
        import hdfs
        try:
            self.client.status(path)
            return True
        except hdfs.util.HdfsError as e:
            if str(e).startswith('File does not exist: '):
                return False
            else:
                raise e

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

def async_comp_check(self, original, loc, tokens):
        """Check for Python 3.6 async comprehension."""
        return self.check_py("36", "async comprehension", original, loc, tokens)

def rpc_fix_code(self, source, directory):
        """Formats Python code to conform to the PEP 8 style guide.

        """
        source = get_source(source)
        return fix_code(source, directory)

def setLoggerAll(self, mthd):
        """ Sends all messages to ``logger.[mthd]()`` for handling """
        for key in self._logger_methods:
            self._logger_methods[key] = mthd

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

def end_index(self):
        """
        Returns the 1-based index of the last object on this page,
        relative to total objects found (hits).
        """
        return ((self.number - 1) * self.paginator.per_page +
            len(self.object_list))

def list_blobs(self, prefix=''):
    """Lists names of all blobs by their prefix."""
    return [b.name for b in self.bucket.list_blobs(prefix=prefix)]

def top_level(url, fix_protocol=True):
    """Extract the top level domain from an URL."""
    ext = tld.get_tld(url, fix_protocol=fix_protocol)
    toplevel = '.'.join(urlparse(url).netloc.split('.')[-2:]).split(
        ext)[0] + ext
    return toplevel

def get_current_item(self):
        """Returns (first) selected item or None"""
        l = self.selectedIndexes()
        if len(l) > 0:
            return self.model().get_item(l[0])

def commits_with_message(message):
    """All commits with that message (in current branch)"""
    output = log("--grep '%s'" % message, oneline=True, quiet=True)
    lines = output.splitlines()
    return [l.split(' ', 1)[0] for l in lines]

def _get_str_columns(sf):
    """
    Returns a list of names of columns that are string type.
    """
    return [name for name in sf.column_names() if sf[name].dtype == str]

def normalize(self):
        """ Normalize data. """

        if self.preprocessed_data.empty:
            data = self.original_data
        else:
            data = self.preprocessed_data

        data = pd.DataFrame(preprocessing.normalize(data), columns=data.columns, index=data.index)
        self.preprocessed_data = data

def native_conn(self):
        """Native connection object."""
        if self.__native is None:
            self.__native = self._get_connection()

        return self.__native

def validate(key):
    """Check that the key is a string or bytestring.

    That's the only valid type of key.
    """
    if not isinstance(key, (str, bytes)):
        raise KeyError('Key must be of type str or bytes, found type {}'.format(type(key)))

def _rows_sort(self, rows):
        """
        Returns a list of rows sorted by start and end date.

        :param list[dict[str,T]] rows: The list of rows.

        :rtype: list[dict[str,T]]
        """
        return sorted(rows, key=lambda row: (row[self._key_start_date], row[self._key_end_date]))

def test():
    """Run the unit tests."""
    import unittest
    tests = unittest.TestLoader().discover('tests')
    unittest.TextTestRunner(verbosity=2).run(tests)

def lines(self):
        """

        :return:
        """
        if self._lines is None:
            self._lines = self.obj.content.splitlines()
        return self._lines

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

def _update_texttuple(self, x, y, s, cs, d):
        """Update the text tuple at `x` and `y` with the given `s` and `d`"""
        pos = (x, y, cs)
        for i, (old_x, old_y, old_s, old_cs, old_d) in enumerate(self.value):
            if (old_x, old_y, old_cs) == pos:
                self.value[i] = (old_x, old_y, s, old_cs, d)
                return
        raise ValueError("No text tuple found at {0}!".format(pos))

def go_to_background():
    """ Daemonize the running process. """
    try:
        if os.fork():
            sys.exit()
    except OSError as errmsg:
        LOGGER.error('Fork failed: {0}'.format(errmsg))
        sys.exit('Fork failed')

def is_collection(obj):
    """Tests if an object is a collection."""

    col = getattr(obj, '__getitem__', False)
    val = False if (not col) else True

    if isinstance(obj, basestring):
        val = False

    return val

def _normalize_instancemethod(instance_method):
    """
    wraps(instancemethod) returns a function, not an instancemethod so its repr() is all messed up;
    we want the original repr to show up in the logs, therefore we do this trick
    """
    if not hasattr(instance_method, 'im_self'):
        return instance_method

    def _func(*args, **kwargs):
        return instance_method(*args, **kwargs)

    _func.__name__ = repr(instance_method)
    return _func

def confusion_matrix(links_true, links_pred, total=None):
    """Compute the confusion matrix.

    The confusion matrix is of the following form:

    +----------------------+-----------------------+----------------------+
    |                      |  Predicted Positives  | Predicted Negatives  |
    +======================+=======================+======================+
    | **True Positives**   | True Positives (TP)   | False Negatives (FN) |
    +----------------------+-----------------------+----------------------+
    | **True Negatives**   | False Positives (FP)  | True Negatives (TN)  |
    +----------------------+-----------------------+----------------------+

    The confusion matrix is an informative way to analyse a prediction. The
    matrix can used to compute measures like precision and recall. The count
    of true prositives is [0,0], false negatives is [0,1], true negatives
    is [1,1] and false positives is [1,0].

    Parameters
    ----------
    links_true: pandas.MultiIndex, pandas.DataFrame, pandas.Series
        The true (or actual) links.
    links_pred: pandas.MultiIndex, pandas.DataFrame, pandas.Series
        The predicted links.
    total: int, pandas.MultiIndex
        The count of all record pairs (both links and non-links). When the
        argument is a pandas.MultiIndex, the length of the index is used. If
        the total is None, the number of True Negatives is not computed.
        Default None.

    Returns
    -------
    numpy.array
        The confusion matrix with TP, TN, FN, FP values.

    Note
    ----
    The number of True Negatives is computed based on the total argument.
    This argument is the number of record pairs of the entire matrix.

    """

    links_true = _get_multiindex(links_true)
    links_pred = _get_multiindex(links_pred)

    tp = true_positives(links_true, links_pred)
    fp = false_positives(links_true, links_pred)
    fn = false_negatives(links_true, links_pred)

    if total is None:
        tn = numpy.nan
    else:
        tn = true_negatives(links_true, links_pred, total)

    return numpy.array([[tp, fn], [fp, tn]])

def cli_parse(parser):
    """Add method specific options to CLI parser.

    Parameters
    ----------
    parser : argparse object

    Returns
    ----------
    Updated argparse object
    """
    parser.add_argument('-n', '--samples', type=int, required=True,
                        help='Number of Samples')
    return parser

def get_by(self, name):
    """get element by name"""
    return next((item for item in self if item.name == name), None)

def __repr__(self):
    """Returns a stringified representation of this object."""
    return str({'name': self._name, 'watts': self._watts,
                'type': self._output_type, 'id': self._integration_id})

def load(cls,filename):
        """Load from stored files"""
        filename = cls.correct_file_extension(filename)
        with open(filename,'rb') as f:
            return pickle.load(f)

def parsed_args():
    parser = argparse.ArgumentParser(description="""python runtime functions""", epilog="")
    parser.add_argument('command',nargs='*',
        help="Name of the function to run with arguments")
    args = parser.parse_args()
    return (args, parser)

def get_env_default(self, variable, default):
        """
        Fetch environment variables, returning a default if not found
        """
        if variable in os.environ:
            env_var = os.environ[variable]
        else:
            env_var = default
        return env_var

def timestamp_filename(basename, ext=None):
    """
    Return a string of the form [basename-TIMESTAMP.ext]
    where TIMESTAMP is of the form YYYYMMDD-HHMMSS-MILSEC
    """
    dt = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    if ext:
        return '%s-%s.%s' % (basename, dt, ext)
    return '%s-%s' % (basename, dt)

def _stdout_raw(self, s):
        """Writes the string to stdout"""
        print(s, end='', file=sys.stdout)
        sys.stdout.flush()

def smallest_signed_angle(source, target):
    """Find the smallest angle going from angle `source` to angle `target`."""
    dth = target - source
    dth = (dth + np.pi) % (2.0 * np.pi) - np.pi
    return dth

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

def debug(self, text):
		""" Ajout d'un message de log de type DEBUG """
		self.logger.debug("{}{}".format(self.message_prefix, text))

def Sum(a, axis, keep_dims):
    """
    Sum reduction op.
    """
    return np.sum(a, axis=axis if not isinstance(axis, np.ndarray) else tuple(axis),
                  keepdims=keep_dims),

def kindex(matrix, k):
    """ Returns indices to select the kth nearest neighbour"""

    ix = (np.arange(len(matrix)), matrix.argsort(axis=0)[k])
    return ix

def getRowCurrentIndex(self):
        """ Returns the index of column 0 of the current item in the underlying model.
            See also the notes at the top of this module on current item vs selected item(s).
        """
        curIndex = self.currentIndex()
        col0Index = curIndex.sibling(curIndex.row(), 0)
        return col0Index

def is_quoted(arg: str) -> bool:
    """
    Checks if a string is quoted
    :param arg: the string being checked for quotes
    :return: True if a string is quoted
    """
    return len(arg) > 1 and arg[0] == arg[-1] and arg[0] in constants.QUOTES

def percent_d(data, period):
    """
    %D.

    Formula:
    %D = SMA(%K, 3)
    """
    p_k = percent_k(data, period)
    percent_d = sma(p_k, 3)
    return percent_d

def elmo_loss2ppl(losses: List[np.ndarray]) -> float:
    """ Calculates perplexity by loss

    Args:
        losses: list of numpy arrays of model losses

    Returns:
        perplexity : float
    """
    avg_loss = np.mean(losses)
    return float(np.exp(avg_loss))

def intty(cls):
        """ Check if we are in a tty. """
        # XXX: temporary hack until we can detect if we are in a pipe or not
        return True

        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            return True

        return False

def camelize(key):
    """Convert a python_style_variable_name to lowerCamelCase.

    Examples
    --------
    >>> camelize('variable_name')
    'variableName'
    >>> camelize('variableName')
    'variableName'
    """
    return ''.join(x.capitalize() if i > 0 else x
                   for i, x in enumerate(key.split('_')))

def close(*args, **kwargs):
    r"""Close last created figure, alias to ``plt.close()``."""
    _, plt, _ = _import_plt()
    plt.close(*args, **kwargs)

def cleanup(self, app):
        """Close all connections."""
        if hasattr(self.database.obj, 'close_all'):
            self.database.close_all()

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

def _shape(self):
        """Return the tensor shape of the matrix operator"""
        return tuple(reversed(self.output_dims())) + tuple(
            reversed(self.input_dims()))

def class_name(obj):
    """
    Get the name of an object, including the module name if available.
    """

    name = obj.__name__
    module = getattr(obj, '__module__')

    if module:
        name = f'{module}.{name}'
    return name

def to_utc(self, dt):
        """Convert any timestamp to UTC (with tzinfo)."""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=self.utc)
        return dt.astimezone(self.utc)

def to_datetime(value):
    """Converts a string to a datetime."""
    if value is None:
        return None

    if isinstance(value, six.integer_types):
        return parser.parse(value)
    return parser.isoparse(value)

def __getitem__(self, key):
        """Returns a new PRDD of elements from that key."""
        return self.from_rdd(self._rdd.map(lambda x: x[key]))

def content_type(self, data):
        """The Content-Type header value for this request."""
        self._content_type = str(data)
        self.add_header('Content-Type', str(data))

def get_single_item(d):
    """Get an item from a dict which contains just one item."""
    assert len(d) == 1, 'Single-item dict must have just one item, not %d.' % len(d)
    return next(six.iteritems(d))

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

def integer_partition(size: int, nparts: int) -> Iterator[List[List[int]]]:
    """ Partition a list of integers into a list of partitions """
    for part in algorithm_u(range(size), nparts):
        yield part

def bytes_to_c_array(data):
    """
    Make a C array using the given string.
    """
    chars = [
        "'{}'".format(encode_escape(i))
        for i in decode_escape(data)
    ]
    return ', '.join(chars) + ', 0'

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

def filter_query_string(query):
    """
        Return a version of the query string with the _e, _k and _s values
        removed.
    """
    return '&'.join([q for q in query.split('&')
        if not (q.startswith('_k=') or q.startswith('_e=') or q.startswith('_s'))])

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

def list_formatter(handler, item, value):
    """Format list."""
    return u', '.join(str(v) for v in value)

def check_if_branch_exist(db, root_hash, key_prefix):
    """
    Given a key prefix, return whether this prefix is
    the prefix of an existing key in the trie.
    """
    validate_is_bytes(key_prefix)

    return _check_if_branch_exist(db, root_hash, encode_to_bin(key_prefix))

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

def create_opengl_object(gl_gen_function, n=1):
    """Returns int pointing to an OpenGL texture"""
    handle = gl.GLuint(1)
    gl_gen_function(n, byref(handle))  # Create n Empty Objects
    if n > 1:
        return [handle.value + el for el in range(n)]  # Return list of handle values
    else:
        return handle.value

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

def print_images(self, *printable_images):
        """
        This method allows printing several images in one shot. This is useful if the client code does not want the
        printer to make pause during printing
        """
        printable_image = reduce(lambda x, y: x.append(y), list(printable_images))
        self.print_image(printable_image)

def __call__(self, actual_value, expect):
        """Main entry point for assertions (called by the wrapper).
        expect is a function the wrapper class uses to assert a given match.
        """
        self._expect = expect
        if self.expected_value is NO_ARG:
            return self.asserts(actual_value)
        return self.asserts(actual_value, self.expected_value)

def subsystem(s):
    """Validate a |Subsystem|.

    Checks its state and cut.
    """
    node_states(s.state)
    cut(s.cut, s.cut_indices)
    if config.VALIDATE_SUBSYSTEM_STATES:
        state_reachable(s)
    return True

def assert_called(_mock_self):
        """assert that the mock was called at least once
        """
        self = _mock_self
        if self.call_count == 0:
            msg = ("Expected '%s' to have been called." %
                   self._mock_name or 'mock')
            raise AssertionError(msg)

def _cho_factor(A, lower=True, check_finite=True):
    """Implementaton of :func:`scipy.linalg.cho_factor` using
    a function supported in cupy."""

    return cp.linalg.cholesky(A), True

def unique_everseen(seq):
    """Solution found here : http://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-python-whilst-preserving-order"""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def _eq(self, other):
        """Compare two nodes for equality."""
        return (self.type, self.value) == (other.type, other.value)

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

def sync(self, recursive=False):
        """
        Syncs the information from this item to the tree and view.
        """
        self.syncTree(recursive=recursive)
        self.syncView(recursive=recursive)

def quit(self):
        """ Quits the application (called when the last window is closed)
        """
        logger.debug("ArgosApplication.quit called")
        assert len(self.mainWindows) == 0, \
            "Bug: still {} windows present at application quit!".format(len(self.mainWindows))
        self.qApplication.quit()

def as_dict(self):
        """
        Json-serializable dict representation of PhononDos.
        """
        return {"@module": self.__class__.__module__,
                "@class": self.__class__.__name__,
                "frequencies": list(self.frequencies),
                "densities": list(self.densities)}

def validate(request: Union[Dict, List], schema: dict) -> Union[Dict, List]:
    """
    Wraps jsonschema.validate, returning the same object passed in.

    Args:
        request: The deserialized-from-json request.
        schema: The jsonschema schema to validate against.

    Raises:
        jsonschema.ValidationError
    """
    jsonschema_validate(request, schema)
    return request

def _days_in_month(date):
    """The number of days in the month of the given date"""
    if date.month == 12:
        reference = type(date)(date.year + 1, 1, 1)
    else:
        reference = type(date)(date.year, date.month + 1, 1)
    return (reference - timedelta(days=1)).day

def add_column(connection, column):
    """
    Add a column to the current table.
    """
    stmt = alembic.ddl.base.AddColumn(_State.table.name, column)
    connection.execute(stmt)
    _State.reflect_metadata()

def naturalsortkey(s):
    """Natural sort order"""
    return [int(part) if part.isdigit() else part
            for part in re.split('([0-9]+)', s)]

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

def _remove_none_values(dictionary):
    """ Remove dictionary keys whose value is None """
    return list(map(dictionary.pop,
                    [i for i in dictionary if dictionary[i] is None]))

def filter_dict_by_key(d, keys):
    """Filter the dict *d* to remove keys not in *keys*."""
    return {k: v for k, v in d.items() if k in keys}

def type_converter(text):
    """ I convert strings into integers, floats, and strings! """
    if text.isdigit():
        return int(text), int

    try:
        return float(text), float
    except ValueError:
        return text, STRING_TYPE

def deprecate(func):
  """ A deprecation warning emmiter as a decorator. """
  @wraps(func)
  def wrapper(*args, **kwargs):
    warn("Deprecated, this will be removed in the future", DeprecationWarning)
    return func(*args, **kwargs)
  wrapper.__doc__ = "Deprecated.\n" + (wrapper.__doc__ or "")
  return wrapper

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

def extract_zip(zip_path, target_folder):
    """
    Extract the content of the zip-file at `zip_path` into `target_folder`.
    """
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target_folder)

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

def chars(string: any) -> str:
        """Return all (and only) the chars in the given string."""
        return ''.join([c if c.isalpha() else '' for c in str(string)])

def calculate_embedding(self, batch_image_bytes):
    """Get the embeddings for a given JPEG image.

    Args:
      batch_image_bytes: As if returned from [ff.read() for ff in file_list].

    Returns:
      The Inception embeddings (bottleneck layer output)
    """
    return self.tf_session.run(
        self.embedding, feed_dict={self.input_jpeg: batch_image_bytes})

def levenshtein_distance_metric(a, b):
    """ 1 - farthest apart (same number of words, all diff). 0 - same"""
    return (levenshtein_distance(a, b) / (2.0 * max(len(a), len(b), 1)))

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

def __deepcopy__(self, memo):
        """Create a deep copy of the node"""
        # noinspection PyArgumentList
        return self.__class__(
            **{key: deepcopy(getattr(self, key), memo) for key in self.keys}
        )

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

def fast_median(a):
    """Fast median operation for masked array using 50th-percentile
    """
    a = checkma(a)
    #return scoreatpercentile(a.compressed(), 50)
    if a.count() > 0:
        out = np.percentile(a.compressed(), 50)
    else:
        out = np.ma.masked
    return out

def _do_remove_prefix(name):
    """Strip the possible prefix 'Table: ' from a table name."""
    res = name
    if isinstance(res, str):
        if (res.find('Table: ') == 0):
            res = res.replace('Table: ', '', 1)
    return res

def _izip(*iterables):
    """ Iterate through multiple lists or arrays of equal size """
    # This izip routine is from itertools
    # izip('ABCD', 'xy') --> Ax By

    iterators = map(iter, iterables)
    while iterators:
        yield tuple(map(next, iterators))

def stop(self):
        """Stop the progress bar."""
        if self._progressing:
            self._progressing = False
            self._thread.join()

def partition_items(count, bin_size):
	"""
	Given the total number of items, determine the number of items that
	can be added to each bin with a limit on the bin size.

	So if you want to partition 11 items into groups of 3, you'll want
	three of three and one of two.

	>>> partition_items(11, 3)
	[3, 3, 3, 2]

	But if you only have ten items, you'll have two groups of three and
	two of two.

	>>> partition_items(10, 3)
	[3, 3, 2, 2]
	"""
	num_bins = int(math.ceil(count / float(bin_size)))
	bins = [0] * num_bins
	for i in range(count):
		bins[i % num_bins] += 1
	return bins

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

def decode_example(self, example):
    """Reconstruct the image from the tf example."""
    img = tf.image.decode_image(
        example, channels=self._shape[-1], dtype=tf.uint8)
    img.set_shape(self._shape)
    return img

def _platform_is_windows(platform=sys.platform):
        """Is the current OS a Windows?"""
        matched = platform in ('cygwin', 'win32', 'win64')
        if matched:
            error_msg = "Windows isn't supported yet"
            raise OSError(error_msg)
        return matched

def _rgbtomask(self, obj):
        """Convert RGB arrays from mask canvas object back to boolean mask."""
        dat = obj.get_image().get_data()  # RGB arrays
        return dat.sum(axis=2).astype(np.bool)

def gen_api_key(username):
    """
    Create a random API key for a user
    :param username:
    :return: Hex encoded SHA512 random string
    """
    salt = str(os.urandom(64)).encode('utf-8')
    return hash_password(username, salt)

def _split_str(s, n):
    """
    split string into list of strings by specified number.
    """
    length = len(s)
    return [s[i:i + n] for i in range(0, length, n)]

def Min(a, axis, keep_dims):
    """
    Min reduction op.
    """
    return np.amin(a, axis=axis if not isinstance(axis, np.ndarray) else tuple(axis),
                   keepdims=keep_dims),

def render_none(self, context, result):
		"""Render empty responses."""
		context.response.body = b''
		del context.response.content_length
		return True

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

def _parse_canonical_int64(doc):
    """Decode a JSON int64 to bson.int64.Int64."""
    l_str = doc['$numberLong']
    if len(doc) != 1:
        raise TypeError('Bad $numberLong, extra field(s): %s' % (doc,))
    return Int64(l_str)

def clean_dataframe(df):
    """Fill NaNs with the previous value, the next value or if all are NaN then 1.0"""
    df = df.fillna(method='ffill')
    df = df.fillna(0.0)
    return df

def flatten_dict_join_keys(dct, join_symbol=" "):
    """ Flatten dict with defined key join symbol.

    :param dct: dict to flatten
    :param join_symbol: default value is " "
    :return:
    """
    return dict( flatten_dict(dct, join=lambda a,b:a+join_symbol+b) )

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

def try_instance_init(self, instance, late_start=False):
        """Try to "initialize" the given module instance.

        :param instance: instance to init
        :type instance: object
        :param late_start: If late_start, don't look for last_init_try
        :type late_start: bool
        :return: True on successful init. False if instance init method raised any Exception.
        :rtype: bool
        """
        try:
            instance.init_try += 1
            # Maybe it's a retry
            if not late_start and instance.init_try > 1:
                # Do not try until too frequently, or it's too loopy
                if instance.last_init_try > time.time() - MODULE_INIT_PERIOD:
                    logger.info("Too early to retry initialization, retry period is %d seconds",
                                MODULE_INIT_PERIOD)
                    # logger.info("%s / %s", instance.last_init_try, time.time())
                    return False
            instance.last_init_try = time.time()

            logger.info("Trying to initialize module: %s", instance.name)

            # If it's an external module, create/update Queues()
            if instance.is_external:
                instance.create_queues(self.daemon.sync_manager)

            # The module instance init function says if initialization is ok
            if not instance.init():
                logger.warning("Module %s initialisation failed.", instance.name)
                return False
            logger.info("Module %s is initialized.", instance.name)
        except Exception as exp:  # pylint: disable=broad-except
            # pragma: no cover, simple protection
            msg = "The module instance %s raised an exception " \
                  "on initialization: %s, I remove it!" % (instance.name, str(exp))
            self.configuration_errors.append(msg)
            logger.error(msg)
            logger.exception(exp)
            return False

        return True

def do_EOF(self, args):
        """Exit on system end of file character"""
        if _debug: ConsoleCmd._debug("do_EOF %r", args)
        return self.do_exit(args)

def serve(application, host='127.0.0.1', port=8080, threads=4, **kw):
	"""The recommended development HTTP server.
	
	Note that this server performs additional buffering and will not honour chunked encoding breaks.
	"""
	
	# Bind and start the server; this is a blocking process.
	serve_(application, host=host, port=int(port), threads=int(threads), **kw)

def check_valid(number, input_base=10):
    """
    Checks if there is an invalid digit in the input number.

    Args:
        number: An number in the following form:
            (int, int, int, ... , '.' , int, int, int)
            (iterable container) containing positive integers of the input base
        input_base(int): The base of the input number.

    Returns:
        bool, True if all digits valid, else False.

    Examples:
        >>> check_valid((1,9,6,'.',5,1,6), 12)
        True
        >>> check_valid((8,1,15,9), 15)
        False
    """
    for n in number:
        if n in (".", "[", "]"):
            continue
        elif n >= input_base:
            if n == 1 and input_base == 1:
                continue
            else:
                return False
    return True

def end_block(self):
        """Ends an indentation block, leaving an empty line afterwards"""
        self.current_indent -= 1

        # If we did not add a new line automatically yet, now it's the time!
        if not self.auto_added_line:
            self.writeln()
            self.auto_added_line = True

def sortBy(self, keyfunc, ascending=True, numPartitions=None):
        """
        Sorts this RDD by the given keyfunc

        >>> tmp = [('a', 1), ('b', 2), ('1', 3), ('d', 4), ('2', 5)]
        >>> sc.parallelize(tmp).sortBy(lambda x: x[0]).collect()
        [('1', 3), ('2', 5), ('a', 1), ('b', 2), ('d', 4)]
        >>> sc.parallelize(tmp).sortBy(lambda x: x[1]).collect()
        [('a', 1), ('b', 2), ('1', 3), ('d', 4), ('2', 5)]
        """
        return self.keyBy(keyfunc).sortByKey(ascending, numPartitions).values()

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

def _numpy_bytes_to_char(arr):
    """Like netCDF4.stringtochar, but faster and more flexible.
    """
    # ensure the array is contiguous
    arr = np.array(arr, copy=False, order='C', dtype=np.string_)
    return arr.reshape(arr.shape + (1,)).view('S1')

def json_pretty_dump(obj, filename):
    """
    Serialize obj as a JSON formatted stream to the given filename (
    pretty printing version)
    """
    with open(filename, "wt") as fh:
        json.dump(obj, fh, indent=4, sort_keys=4)

def drop_column(self, tablename: str, fieldname: str) -> int:
        """Drops (deletes) a column from an existing table."""
        sql = "ALTER TABLE {} DROP COLUMN {}".format(tablename, fieldname)
        log.info(sql)
        return self.db_exec_literal(sql)

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

def subscribe_to_quorum_channel(self):
        """In case the experiment enforces a quorum, listen for notifications
        before creating Partipant objects.
        """
        from dallinger.experiment_server.sockets import chat_backend

        self.log("Bot subscribing to quorum channel.")
        chat_backend.subscribe(self, "quorum")

def disable_stdout_buffering():
    """This turns off stdout buffering so that outputs are immediately
    materialized and log messages show up before the program exits"""
    stdout_orig = sys.stdout
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
    # NOTE(brandyn): This removes the original stdout
    return stdout_orig

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

async def stdout(self) -> AsyncGenerator[str, None]:
        """Asynchronous generator for lines from subprocess stdout."""
        await self.wait_running()
        async for line in self._subprocess.stdout:  # type: ignore
            yield line

def interp(x, xp, *args, **kwargs):
    """Wrap interpolate_1d for deprecated interp."""
    return interpolate_1d(x, xp, *args, **kwargs)

def chunks(iterable, size=1):
    """Splits iterator in chunks."""
    iterator = iter(iterable)

    for element in iterator:
        yield chain([element], islice(iterator, size - 1))

def open_usb_handle(self, port_num):
    """open usb port

    Args:
      port_num: port number on the Cambrionix unit

    Return:
      usb handle
    """
    serial = self.get_usb_serial(port_num)
    return local_usb.LibUsbHandle.open(serial_number=serial)

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

def execfile(fname, variables):
    """ This is builtin in python2, but we have to roll our own on py3. """
    with open(fname) as f:
        code = compile(f.read(), fname, 'exec')
        exec(code, variables)

def split_into_sentences(s):
  """Split text into list of sentences."""
  s = re.sub(r"\s+", " ", s)
  s = re.sub(r"[\\.\\?\\!]", "\n", s)
  return s.split("\n")

def is_valid_url(url):
    """Checks if a given string is an url"""
    pieces = urlparse(url)
    return all([pieces.scheme, pieces.netloc])

def is_admin(self):
        """Is the user a system administrator"""
        return self.role == self.roles.administrator.value and self.state == State.approved

def _dump_spec(spec):
    """Dump bel specification dictionary using YAML

    Formats this with an extra indentation for lists to make it easier to
    use cold folding on the YAML version of the spec dictionary.
    """
    with open("spec.yaml", "w") as f:
        yaml.dump(spec, f, Dumper=MyDumper, default_flow_style=False)

def version_triple(tag):
    """
    returns: a triple of integers from a version tag
    """
    groups = re.match(r'v?(\d+)\.(\d+)\.(\d+)', tag).groups()
    return tuple(int(n) for n in groups)

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

def translate_dict(cls, val):
        """Translate dicts to scala Maps"""
        escaped = ', '.join(
            ["{} -> {}".format(cls.translate_str(k), cls.translate(v)) for k, v in val.items()]
        )
        return 'Map({})'.format(escaped)

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

def batch_tensor(self, name):
        """ A buffer of a given value in a 'flat' (minibatch-indexed) format """
        if name in self.transition_tensors:
            return tensor_util.merge_first_two_dims(self.transition_tensors[name])
        else:
            return self.rollout_tensors[name]

def _make_proxy_property(bind_attr, attr_name):
    def proxy_property(self):
        """
        proxy
        """
        bind = getattr(self, bind_attr)
        return getattr(bind, attr_name)
    return property(proxy_property)

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

def _is_retryable_exception(e):
    """Returns True if the exception is always safe to retry.

    This is True if the client was never able to establish a connection
    to the server (for example, name resolution failed or the connection
    could otherwise not be initialized).

    Conservatively, if we can't tell whether a network connection could
    have been established, we return False.

    """
    if isinstance(e, urllib3.exceptions.ProtocolError):
        e = e.args[1]
    if isinstance(e, (socket.gaierror, socket.herror)):
        return True
    if isinstance(e, socket.error) and e.errno in _RETRYABLE_SOCKET_ERRORS:
        return True
    if isinstance(e, urllib3.exceptions.NewConnectionError):
        return True
    return False

def get_average_length_of_string(strings):
    """Computes average length of words

    :param strings: list of words
    :return: Average length of word on list
    """
    if not strings:
        return 0

    return sum(len(word) for word in strings) / len(strings)

def get_filesize(self, pdf):
        """Compute the filesize of the PDF
        """
        try:
            filesize = float(pdf.get_size())
            return filesize / 1024
        except (POSKeyError, TypeError):
            return 0

def lognormcdf(x, mu, tau):
    """Log-normal cumulative density function"""
    x = np.atleast_1d(x)
    return np.array(
        [0.5 * (1 - flib.derf(-(np.sqrt(tau / 2)) * (np.log(y) - mu))) for y in x])

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

def get_time(filename):
	"""
	Get the modified time for a file as a datetime instance
	"""
	ts = os.stat(filename).st_mtime
	return datetime.datetime.utcfromtimestamp(ts)

def _normalize(mat: np.ndarray):
    """rescales a numpy array, so that min is 0 and max is 255"""
    return ((mat - mat.min()) * (255 / mat.max())).astype(np.uint8)

def TextWidget(*args, **kw):
    """Forces a parameter value to be text"""
    kw['value'] = str(kw['value'])
    kw.pop('options', None)
    return TextInput(*args,**kw)

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

def clean_time(time_string):
    """Return a datetime from the Amazon-provided datetime string"""
    # Get a timezone-aware datetime object from the string
    time = dateutil.parser.parse(time_string)
    if not settings.USE_TZ:
        # If timezone support is not active, convert the time to UTC and
        # remove the timezone field
        time = time.astimezone(timezone.utc).replace(tzinfo=None)
    return time

def pout(msg, log=None):
    """Print 'msg' to stdout, and option 'log' at info level."""
    _print(msg, sys.stdout, log_func=log.info if log else None)

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

def _openResources(self):
        """ Uses numpy.load to open the underlying file
        """
        arr = np.load(self._fileName, allow_pickle=ALLOW_PICKLE)
        check_is_an_array(arr)
        self._array = arr

def assert_is_instance(value, types, message=None, extra=None):
    """Raises an AssertionError if value is not an instance of type(s)."""
    assert isinstance(value, types), _assert_fail_message(
        message, value, types, "is not an instance of", extra
    )

def test_for_image(self, cat, img):
        """Tests if image img in category cat exists"""
        return self.test_for_category(cat) and img in self.items[cat]

def bbox(self):
        """
        The bounding box ``(ymin, xmin, ymax, xmax)`` of the minimal
        rectangular region containing the source segment.
        """

        # (stop - 1) to return the max pixel location, not the slice index
        return (self._slice[0].start, self._slice[1].start,
                self._slice[0].stop - 1, self._slice[1].stop - 1) * u.pix

def samefile(a: str, b: str) -> bool:
    """Check if two pathes represent the same file."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normpath(a) == os.path.normpath(b)

def ner_chunk(args):
  """Chunk named entities."""
  chunker = NEChunker(lang=args.lang)
  tag(chunker, args)

def is_listish(obj):
    """Check if something quacks like a list."""
    if isinstance(obj, (list, tuple, set)):
        return True
    return is_sequence(obj)

def get_variables(args):
  """
  Return a dictionary of variables specified at CLI
  :param: args: Command Line Arguments namespace
  """
  variables_dict = {}
  if args.variables:
    for var in args.variables:
      words = var.split('=')
      variables_dict[words[0]] = words[1]
  return variables_dict

def _column_resized(self, col, old_width, new_width):
        """Update the column width."""
        self.dataTable.setColumnWidth(col, new_width)
        self._update_layout()

def __del__(self):
        """Frees all resources.
        """
        if hasattr(self, '_Api'):
            self._Api.close()

        self._Logger.info('object destroyed')

def binary(length):
    """
        returns a a random string that represent a binary representation

    :param length: number of bits
    """
    num = randint(1, 999999)
    mask = '0' * length
    return (mask + ''.join([str(num >> i & 1) for i in range(7, -1, -1)]))[-length:]

def dict_from_object(obj: object):
    """Convert a object into dictionary with all of its readable attributes."""

    # If object is a dict instance, no need to convert.
    return (obj if isinstance(obj, dict)
            else {attr: getattr(obj, attr)
                  for attr in dir(obj) if not attr.startswith('_')})

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

def cols_strip(df,col_list, dest = False):
    """ Performs str.strip() a column of a DataFrame
    Parameters:
    df - DataFrame
        DataFrame to operate on
    col_list - list of strings
        names of columns to strip
    dest - bool, default False
        Whether to apply the result to the DataFrame or return it.
        True is apply, False is return.
    """
    if not dest:
        return _pd.DataFrame({col_name:col_strip(df,col_name) for col_name in col_list})
    for col_name in col_list:
        col_strip(df,col_name,dest)

def stop(self):
        """stop server"""
        try:
            self.shutdown()
        except (PyMongoError, ServersError) as exc:
            logger.info("Killing %s with signal, shutdown command failed: %r",
                        self.name, exc)
            return process.kill_mprocess(self.proc)

def ungzip_data(input_data):
    """Return a string of data after gzip decoding

    :param the input gziped data
    :return  the gzip decoded data"""
    buf = StringIO(input_data)
    f = gzip.GzipFile(fileobj=buf)
    return f

def diff(file_, imports):
    """Display the difference between modules in a file and imported modules."""
    modules_not_imported = compare_modules(file_, imports)

    logging.info("The following modules are in {} but do not seem to be imported: "
                 "{}".format(file_, ", ".join(x for x in modules_not_imported)))

def remove_series(self, series):
        """Removes a :py:class:`.Series` from the chart.

        :param Series series: The :py:class:`.Series` to remove.
        :raises ValueError: if you try to remove the last\
        :py:class:`.Series`."""

        if len(self.all_series()) == 1:
            raise ValueError("Cannot remove last series from %s" % str(self))
        self._all_series.remove(series)
        series._chart = None

def u2b(string):
    """ unicode to bytes"""
    if ((PY2 and isinstance(string, unicode)) or
        ((not PY2) and isinstance(string, str))):
        return string.encode('utf-8')
    return string

def __init__(self, enumtype, index, key):
        """ Set up a new instance. """
        self._enumtype = enumtype
        self._index = index
        self._key = key

def polygon_from_points(points):
    """
    Constructs a numpy-compatible polygon from a page representation.
    """
    polygon = []
    for pair in points.split(" "):
        x_y = pair.split(",")
        polygon.append([float(x_y[0]), float(x_y[1])])
    return polygon

def percentile_index(a, q):
    """
    Returns the index of the value at the Qth percentile in array a.
    """
    return np.where(
        a==np.percentile(a, q, interpolation='nearest')
    )[0][0]

def write(self, value):
        """
        Write value to the target
        """
        self.get_collection().update_one(
            {'_id': self._document_id},
            {'$set': {self._path: value}},
            upsert=True
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

def compress(self, data_list):
        """
        Return the cleaned_data of the form, everything should already be valid
        """
        data = {}
        if data_list:
            return dict(
                (f.name, data_list[i]) for i, f in enumerate(self.form))
        return data

def image_set_aspect(aspect=1.0, axes="gca"):
    """
    sets the aspect ratio of the current zoom level of the imshow image
    """
    if axes is "gca": axes = _pylab.gca()

    e = axes.get_images()[0].get_extent()
    axes.set_aspect(abs((e[1]-e[0])/(e[3]-e[2]))/aspect)

def delete_all_from_db():
    """Clear the database.

    Used for testing and debugging.

    """
    # The models.CASCADE property is set on all ForeignKey fields, so tables can
    # be deleted in any order without breaking constraints.
    for model in django.apps.apps.get_models():
        model.objects.all().delete()

def is_static(*p):
    """ A static value (does not change at runtime)
    which is known at compile time
    """
    return all(is_CONST(x) or
               is_number(x) or
               is_const(x)
               for x in p)

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

def on_close(self, evt):
    """
    Pop-up menu and wx.EVT_CLOSE closing event
    """
    self.stop() # DoseWatcher
    if evt.EventObject is not self: # Avoid deadlocks
      self.Close() # wx.Frame
    evt.Skip()

def get_cursor(self):
        """Returns current grid cursor cell (row, col, tab)"""

        return self.grid.GetGridCursorRow(), self.grid.GetGridCursorCol(), \
            self.grid.current_table

def base64ToImage(imgData, out_path, out_file):
        """ converts a base64 string to a file """
        fh = open(os.path.join(out_path, out_file), "wb")
        fh.write(imgData.decode('base64'))
        fh.close()
        del fh
        return os.path.join(out_path, out_file)

def get_files(dir_name):
    """Simple directory walker"""
    return [(os.path.join('.', d), [os.path.join(d, f) for f in files]) for d, _, files in os.walk(dir_name)]

def calculate_size(name, function):
    """ Calculates the request payload size"""
    data_size = 0
    data_size += calculate_size_str(name)
    data_size += calculate_size_data(function)
    return data_size

def write_fits(self, fitsfile):
        """Write the ROI model to a FITS file."""

        tab = self.create_table()
        hdu_data = fits.table_to_hdu(tab)
        hdus = [fits.PrimaryHDU(), hdu_data]
        fits_utils.write_hdus(hdus, fitsfile)

def consecutive(data, stepsize=1):
    """Converts array into chunks with consecutive elements of given step size.
    http://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-from-an-array-in-numpy
    """
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)

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

def replace_all(text, dic):
    """Takes a string and dictionary. replaces all occurrences of i with j"""

    for i, j in dic.iteritems():
        text = text.replace(i, j)
    return text

def list_files(directory):
    """Returns all files in a given directory
    """
    return [f for f in pathlib.Path(directory).iterdir() if f.is_file() and not f.name.startswith('.')]

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

def _tostr(self,obj):
        """ converts a object to list, if object is a list, it creates a
            comma seperated string.
        """
        if not obj:
            return ''
        if isinstance(obj, list):
            return ', '.join(map(self._tostr, obj))
        return str(obj)

def toListInt(value):
        """
        Convert a value to list of ints, if possible.
        """
        if TypeConverters._can_convert_to_list(value):
            value = TypeConverters.toList(value)
            if all(map(lambda v: TypeConverters._is_integer(v), value)):
                return [int(v) for v in value]
        raise TypeError("Could not convert %s to list of ints" % value)

def get_api_url(self, lambda_name, stage_name):
        """
        Given a lambda_name and stage_name, return a valid API URL.
        """
        api_id = self.get_api_id(lambda_name)
        if api_id:
            return "https://{}.execute-api.{}.amazonaws.com/{}".format(api_id, self.boto_session.region_name, stage_name)
        else:
            return None

def _multilingual(function, *args, **kwargs):
    """ Returns the value from the function with the given name in the given language module.
        By default, language="en".
    """
    return getattr(_module(kwargs.pop("language", "en")), function)(*args, **kwargs)

def execute_cast_simple_literal_to_timestamp(op, data, type, **kwargs):
    """Cast integer and strings to timestamps"""
    return pd.Timestamp(data, tz=type.timezone)

def _try_join_cancelled_thread(thread):
    """Join a thread, but if the thread doesn't terminate for some time, ignore it
    instead of waiting infinitely."""
    thread.join(10)
    if thread.is_alive():
        logging.warning("Thread %s did not terminate within grace period after cancellation",
                        thread.name)

def timeit(output):
    """
    If output is string, then print the string and also time used
    """
    b = time.time()
    yield
    print output, 'time used: %.3fs' % (time.time()-b)

def ranges_to_set(lst):
    """
    Convert a list of ranges to a set of numbers::

    >>> ranges = [(1,3), (5,6)]
    >>> sorted(list(ranges_to_set(ranges)))
    [1, 2, 3, 5, 6]

    """
    return set(itertools.chain(*(range(x[0], x[1]+1) for x in lst)))

def lengths_offsets(value):
    """Split the given comma separated value to multiple integer values. """
    values = []
    for item in value.split(','):
        item = int(item)
        values.append(item)
    return values

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

def load(cls, fp, **kwargs):
    """wrapper for :py:func:`json.load`"""
    json_obj = json.load(fp, **kwargs)
    return parse(cls, json_obj)

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

def pretty_dict_str(d, indent=2):
    """shows JSON indented representation of d"""
    b = StringIO()
    write_pretty_dict_str(b, d, indent=indent)
    return b.getvalue()

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

def enbw(wnd):
  """ Equivalent Noise Bandwidth in bins (Processing Gain reciprocal). """
  return sum(el ** 2 for el in wnd) / sum(wnd) ** 2 * len(wnd)

def file_empty(fp):
    """Determine if a file is empty or not."""
    # for python 2 we need to use a homemade peek()
    if six.PY2:
        contents = fp.read()
        fp.seek(0)
        return not bool(contents)

    else:
        return not fp.peek()

def run(context, port):
    """ Run the Webserver/SocketIO and app
    """
    global ctx
    ctx = context
    app.run(port=port)

def smartSum(x,key,value):
    """ create a new page in x if key is not a page of x
        otherwise add value to x[key] """
    if key not in list(x.keys()):
        x[key] = value
    else:   x[key]+=value

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

def _escape(s):
    """ Helper method that escapes parameters to a SQL query. """
    e = s
    e = e.replace('\\', '\\\\')
    e = e.replace('\n', '\\n')
    e = e.replace('\r', '\\r')
    e = e.replace("'", "\\'")
    e = e.replace('"', '\\"')
    return e

def python(string: str):
        """
            :param string: String can be type, resource or python case
        """
        return underscore(singularize(string) if Naming._pluralize(string) else string)

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

def add_queue_handler(queue):
    """Add a queue log handler to the global logger."""
    handler = QueueLogHandler(queue)
    handler.setFormatter(QueueFormatter())
    handler.setLevel(DEBUG)
    GLOBAL_LOGGER.addHandler(handler)

def rms(x):
    """"Root Mean Square"

    Arguments:
        x (seq of float): A sequence of numerical values

    Returns:
        The square root of the average of the squares of the values

        math.sqrt(sum(x_i**2 for x_i in x) / len(x))

        or

        return (np.array(x) ** 2).mean() ** 0.5

    >>> rms([0, 2, 4, 4])
    3.0
    """
    try:
        return (np.array(x) ** 2).mean() ** 0.5
    except:
        x = np.array(dropna(x))
        invN = 1.0 / len(x)
        return (sum(invN * (x_i ** 2) for x_i in x)) ** .5

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

def set_trace():
    """Start a Pdb instance at the calling frame, with stdout routed to sys.__stdout__."""
    # https://github.com/nose-devs/nose/blob/master/nose/tools/nontrivial.py
    pdb.Pdb(stdout=sys.__stdout__).set_trace(sys._getframe().f_back)

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

def zrank(self, name, value):
        """
        Returns the rank of the element.

        :param name: str     the name of the redis key
        :param value: the element in the sorted set
        """
        with self.pipe as pipe:
            value = self.valueparse.encode(value)
            return pipe.zrank(self.redis_key(name), value)

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

def _config_session():
        """
        Configure session for particular device

        Returns:
            tensorflow.Session
        """
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = '0'
        return tf.Session(config=config)

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

def is_seq(obj):
    """ Returns True if object is not a string but is iterable """
    if not hasattr(obj, '__iter__'):
        return False
    if isinstance(obj, basestring):
        return False
    return True

def finish():
    """Print warning about interrupt and empty the job queue."""
    out.warn("Interrupted!")
    for t in threads:
        t.stop()
    jobs.clear()
    out.warn("Waiting for download threads to finish.")

def as_dict(self):
        """Package up the public attributes as a dict."""
        attrs = vars(self)
        return {key: attrs[key] for key in attrs if not key.startswith('_')}

def save_session_to_file(self, sessionfile):
        """Not meant to be used directly, use :meth:`Instaloader.save_session_to_file`."""
        pickle.dump(requests.utils.dict_from_cookiejar(self._session.cookies), sessionfile)

def generate_hash(filepath):
    """Public function that reads a local file and generates a SHA256 hash digest for it"""
    fr = FileReader(filepath)
    data = fr.read_bin()
    return _calculate_sha256(data)

def getbyteslice(self, start, end):
        """Direct access to byte data."""
        c = self._rawarray[start:end]
        return c

def reindent(s, numspaces):
    """ reinidents a string (s) by the given number of spaces (numspaces) """
    leading_space = numspaces * ' '
    lines = [leading_space + line.strip()for line in s.splitlines()]
    return '\n'.join(lines)

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

def _get_context(argspec, kwargs):
    """Prepare a context for the serialization.

    :param argspec: The argspec of the serialization function.
    :param kwargs: Dict with context
    :return: Keywords arguments that function can accept.
    """
    if argspec.keywords is not None:
        return kwargs
    return dict((arg, kwargs[arg]) for arg in argspec.args if arg in kwargs)

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

def _convert_dict_to_json(array):
    """ Converts array to a json string """
    return json.dumps(
        array,
        skipkeys=False,
        allow_nan=False,
        indent=None,
        separators=(",", ":"),
        sort_keys=True,
        default=lambda o: o.__dict__,
    )

def close_all_but_this(self):
        """Close all files but the current one"""
        self.close_all_right()
        for i in range(0, self.get_stack_count()-1  ):
            self.close_file(0)

def _check_model(obj, models=None):
    """Checks object if it's a peewee model and unique."""
    return isinstance(obj, type) and issubclass(obj, pw.Model) and hasattr(obj, '_meta')

def _skip_frame(self):
        """Skip a single frame from the trajectory"""
        size = self.read_size()
        for i in range(size+1):
            line = self._f.readline()
            if len(line) == 0:
                raise StopIteration

def get_key(self, key, bucket_name=None):
        """
        Returns a boto3.s3.Object

        :param key: the path to the key
        :type key: str
        :param bucket_name: the name of the bucket
        :type bucket_name: str
        """
        if not bucket_name:
            (bucket_name, key) = self.parse_s3_url(key)

        obj = self.get_resource_type('s3').Object(bucket_name, key)
        obj.load()
        return obj

def connect_mysql(host, port, user, password, database):
    """Connect to MySQL with retries."""
    return pymysql.connect(
        host=host, port=port,
        user=user, passwd=password,
        db=database
    )

def excepthook(self, except_type, exception, traceback):
    """Not Used: Custom exception hook to replace sys.excepthook

    This is for CPython's default shell. IPython does not use sys.exepthook.

    https://stackoverflow.com/questions/27674602/hide-traceback-unless-a-debug-flag-is-set
    """
    if except_type is DeepReferenceError:
        print(exception.msg)
    else:
        self.default_excepthook(except_type, exception, traceback)

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

def _cdf(self, xloc, dist, cache):
        """Cumulative distribution function."""
        return evaluation.evaluate_forward(dist, numpy.e**xloc, cache=cache)

def has_permission(user, permission_name):
    """Check if a user has a given permission."""
    if user and user.is_superuser:
        return True

    return permission_name in available_perm_names(user)

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

def from_file(cls, path, encoding, dialect, fields, converters, field_index):
        """Read delimited text from a text file."""

        return cls(open(path, 'r', encoding=encoding), dialect, fields, converters, field_index)

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

def build_parser():
    """Build argument parsers."""

    parser = argparse.ArgumentParser("Release packages to pypi")
    parser.add_argument('--check', '-c', action="store_true", help="Do a dry run without uploading")
    parser.add_argument('component', help="The component to release as component-version")
    return parser

def get_stripped_file_lines(filename):
    """
    Return lines of a file with whitespace removed
    """
    try:
        lines = open(filename).readlines()
    except FileNotFoundError:
        fatal("Could not open file: {!r}".format(filename))

    return [line.strip() for line in lines]

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

def chunks(iterable, n):
    """Yield successive n-sized chunks from iterable object. https://stackoverflow.com/a/312464 """
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]

def _log_disconnect(self):
        """ Decrement connection count """
        if self.logged:
            self.server.stats.connectionClosed()
            self.logged = False

def _mid(pt1, pt2):
    """
    (Point, Point) -> Point
    Return the point that lies in between the two input points.
    """
    (x0, y0), (x1, y1) = pt1, pt2
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)

def pythonise(id, encoding='ascii'):
    """Return a Python-friendly id"""
    replace = {'-': '_', ':': '_', '/': '_'}
    func = lambda id, pair: id.replace(pair[0], pair[1])
    id = reduce(func, replace.iteritems(), id)
    id = '_%s' % id if id[0] in string.digits else id
    return id.encode(encoding)

def hidden_cursor(self):
        """Return a context manager that hides the cursor while inside it and
        makes it visible on leaving."""
        self.stream.write(self.hide_cursor)
        try:
            yield
        finally:
            self.stream.write(self.normal_cursor)

def infer_format(filename:str) -> str:
    """Return extension identifying format of given filename"""
    _, ext = os.path.splitext(filename)
    return ext

def is_parameter(self):
        """Whether this is a function parameter."""
        return (isinstance(self.scope, CodeFunction)
                and self in self.scope.parameters)

def parse_path(path):
    """Parse path string."""
    version, project = path[1:].split('/')
    return dict(version=int(version), project=project)

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

def assert_or_raise(stmt: bool, exception: Exception,
                    *exception_args, **exception_kwargs) -> None:
  """
  If the statement is false, raise the given exception.
  """
  if not stmt:
    raise exception(*exception_args, **exception_kwargs)

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

def csv_to_dicts(file, header=None):
    """Reads a csv and returns a List of Dicts with keys given by header row."""
    with open(file) as csvfile:
        return [row for row in csv.DictReader(csvfile, fieldnames=header)]

def dict_jsonp(param):
    """Convert the parameter into a dictionary before calling jsonp, if it's not already one"""
    if not isinstance(param, dict):
        param = dict(param)
    return jsonp(param)

def backward_delete_word(self, e): # (Control-Rubout)
        u"""Delete the character behind the cursor. A numeric argument means
        to kill the characters instead of deleting them."""
        self.l_buffer.backward_delete_word(self.argument_reset)
        self.finalize()

def get_param_names(cls):
        """Returns a list of plottable CBC parameter variables"""
        return [m[0] for m in inspect.getmembers(cls) \
            if type(m[1]) == property]

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

def dispatch(self, request, *args, **kwargs):
        """Dispatch all HTTP methods to the proxy."""
        self.request = DownstreamRequest(request)
        self.args = args
        self.kwargs = kwargs

        self._verify_config()

        self.middleware = MiddlewareSet(self.proxy_middleware)

        return self.proxy()

def _cached_search_compile(pattern, re_verbose, re_version, pattern_type):
    """Cached search compile."""

    return _bregex_parse._SearchParser(pattern, re_verbose, re_version).parse()

def from_bytes(cls, b):
		"""Create :class:`PNG` from raw bytes.
		
		:arg bytes b: The raw bytes of the PNG file.
		:rtype: :class:`PNG`
		"""
		im = cls()
		im.chunks = list(parse_chunks(b))
		im.init()
		return im

def _get_mtime():
    """
    Get the modified time of the RPM Database.

    Returns:
        Unix ticks
    """
    return os.path.exists(RPM_PATH) and int(os.path.getmtime(RPM_PATH)) or 0

def write_enum(fo, datum, schema):
    """An enum is encoded by a int, representing the zero-based position of
    the symbol in the schema."""
    index = schema['symbols'].index(datum)
    write_int(fo, index)

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

def dir_exists(self):
        """
        Makes a ``HEAD`` requests to the URI.

        :returns: ``True`` if status code is 2xx.
        """

        r = requests.request(self.method if self.method else 'HEAD', self.url, **self.storage_args)
        try: r.raise_for_status()
        except Exception: return False

        return True

def cancel(self, event=None):
        """Function called when Cancel-button clicked.

        This method returns focus to parent, and destroys the dialog.
        """

        if self.parent != None:
            self.parent.focus_set()

        self.destroy()

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

def flush(self):
        """
        Flush all unwritten data to disk.
        """
        if self._cache_modified_count > 0:
            self.storage.write(self.cache)
            self._cache_modified_count = 0

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

def elliot_function( signal, derivative=False ):
    """ A fast approximation of sigmoid """
    s = 1 # steepness
    
    abs_signal = (1 + np.abs(signal * s))
    if derivative:
        return 0.5 * s / abs_signal**2
    else:
        # Return the activation signal
        return 0.5*(signal * s) / abs_signal + 0.5

def test_python_java_rt():
    """ Run Python test cases against Java runtime classes. """
    sub_env = {'PYTHONPATH': _build_dir()}

    log.info('Executing Python unit tests (against Java runtime classes)...')
    return jpyutil._execute_python_scripts(python_java_rt_tests,
                                           env=sub_env)

def ceil_nearest(x, dx=1):
    """
    ceil a number to within a given rounding accuracy
    """
    precision = get_sig_digits(dx)
    return round(math.ceil(float(x) / dx) * dx, precision)

def parse_timestamp(timestamp):
    """Parse ISO8601 timestamps given by github API."""
    dt = dateutil.parser.parse(timestamp)
    return dt.astimezone(dateutil.tz.tzutc())

def do_help(self, arg):
        """
        Show help on all commands.
        """
        print(self.response_prompt, file=self.stdout)
        return cmd.Cmd.do_help(self, arg)

def remove_parenthesis_around_tz(cls, timestr):
        """get rid of parenthesis around timezone: (GMT) => GMT

        :return: the new string if parenthesis were found, `None` otherwise
        """
        parenthesis = cls.TIMEZONE_PARENTHESIS.match(timestr)
        if parenthesis is not None:
            return parenthesis.group(1)

def _skip_frame(self):
        """Skip the next time frame"""
        for line in self._f:
            if line == 'ITEM: ATOMS\n':
                break
        for i in range(self.num_atoms):
            next(self._f)

def indices_to_labels(self, indices: Sequence[int]) -> List[str]:
        """ Converts a sequence of indices into their corresponding labels."""

        return [(self.INDEX_TO_LABEL[index]) for index in indices]

def get_ram(self, format_ = "nl"):
		"""
			return a string representations of the ram
		"""
		ram = [self.ram.read(i) for i in range(self.ram.size)]
		return self._format_mem(ram, format_)

def _sourced_dict(self, source=None, **kwargs):
        """Like ``dict(**kwargs)``, but where the ``source`` key is special.
        """
        if source:
            kwargs['source'] = source
        elif self.source:
            kwargs['source'] = self.source
        return kwargs

async def send_files_preconf(filepaths, config_path=CONFIG_PATH):
    """Send files using the config.ini settings.

    Args:
        filepaths (list(str)): A list of filepaths.
    """
    config = read_config(config_path)
    subject = "PDF files from pdfebc"
    message = ""
    await send_with_attachments(subject, message, filepaths, config)

def negate(self):
        """Reverse the range"""
        self.from_value, self.to_value = self.to_value, self.from_value
        self.include_lower, self.include_upper = self.include_upper, self.include_lower

def get_point_hash(self, point):
        """
        return geohash for given point with self.precision
        :param point: GeoPoint instance
        :return: string
        """
        return geohash.encode(point.latitude, point.longitude, self.precision)

def data(self, data):
        """Store a copy of the data."""
        self._data = {det: d.copy() for (det, d) in data.items()}

def _return_result(self, done):
        """Called set the returned future's state that of the future
        we yielded, and set the current future for the iterator.
        """
        chain_future(done, self._running_future)

        self.current_future = done
        self.current_index = self._unfinished.pop(done)

def rAsciiLine(ifile):
    """Returns the next non-blank line in an ASCII file."""

    _line = ifile.readline().strip()
    while len(_line) == 0:
        _line = ifile.readline().strip()
    return _line

def list_move_to_front(l,value='other'):
    """if the value is in the list, move it to the front and return it."""
    l=list(l)
    if value in l:
        l.remove(value)
        l.insert(0,value)
    return l

def calculate_single_tanimoto_set_distances(target: Iterable[X], dict_of_sets: Mapping[Y, Set[X]]) -> Mapping[Y, float]:
    """Return a dictionary of distances keyed by the keys in the given dict.

    Distances are calculated based on pairwise tanimoto similarity of the sets contained

    :param set target: A set
    :param dict_of_sets: A dict of {x: set of y}
    :type dict_of_sets: dict
    :return: A similarity dicationary based on the set overlap (tanimoto) score between the target set and the sets in
            dos
    :rtype: dict
    """
    target_set = set(target)

    return {
        k: tanimoto_set_similarity(target_set, s)
        for k, s in dict_of_sets.items()
    }

def cint8_array_to_numpy(cptr, length):
    """Convert a ctypes int pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_int8)):
        return np.fromiter(cptr, dtype=np.int8, count=length)
    else:
        raise RuntimeError('Expected int pointer')

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

def strip_spaces(s):
    """ Strip excess spaces from a string """
    return u" ".join([c for c in s.split(u' ') if c])

def create_response(self, request, content, content_type):
        """Returns a response object for the request. Can be overridden to return different responses."""

        return HttpResponse(content=content, content_type=content_type)

def _send_cmd(self, cmd):
        """Write command to remote process
        """
        self._process.stdin.write("{}\n".format(cmd).encode("utf-8"))
        self._process.stdin.flush()

def get_subplot_at(self, row, column):
        """Return the subplot at row, column position.

        :param row,column: specify the subplot.

        """
        idx = row * self.columns + column
        return self.subplots[idx]

def from_iso_time(timestring, use_dateutil=True):
    """Parse an ISO8601-formatted datetime string and return a datetime.time
    object.
    """
    if not _iso8601_time_re.match(timestring):
        raise ValueError('Not a valid ISO8601-formatted time string')
    if dateutil_available and use_dateutil:
        return parser.parse(timestring).time()
    else:
        if len(timestring) > 8:  # has microseconds
            fmt = '%H:%M:%S.%f'
        else:
            fmt = '%H:%M:%S'
        return datetime.datetime.strptime(timestring, fmt).time()

def reject(self):
        """
        Rejects the snapshot and closes the widget.
        """
        if self.hideWindow():
            self.hideWindow().show()
            
        self.close()
        self.deleteLater()

def _lookup_enum_in_ns(namespace, value):
    """Return the attribute of namespace corresponding to value."""
    for attribute in dir(namespace):
        if getattr(namespace, attribute) == value:
            return attribute

def batchify(data, batch_size):
    """Reshape data into (num_example, batch_size)"""
    nbatch = data.shape[0] // batch_size
    data = data[:nbatch * batch_size]
    data = data.reshape((batch_size, nbatch)).T
    return data

def clear_all(self):
        """Delete all Labels."""
        logger.info("Clearing ALL Labels and LabelKeys.")
        self.session.query(Label).delete(synchronize_session="fetch")
        self.session.query(LabelKey).delete(synchronize_session="fetch")

def _iterate_flattened_values(value):
  """Provides an iterator over all values in a nested structure."""
  if isinstance(value, six.string_types):
    yield value
    return

  if isinstance(value, collections.Mapping):
    value = collections.ValuesView(value)

  if isinstance(value, collections.Iterable):
    for nested_value in value:
      for nested_nested_value in _iterate_flattened_values(nested_value):
        yield nested_nested_value

  yield value

def convertDatetime(t):
    """
    Converts the specified datetime object into its appropriate protocol
    value. This is the number of milliseconds from the epoch.
    """
    epoch = datetime.datetime.utcfromtimestamp(0)
    delta = t - epoch
    millis = delta.total_seconds() * 1000
    return int(millis)

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

def remove_duplicates(seq):
    """
    Return unique elements from list while preserving order.
    From https://stackoverflow.com/a/480227/2589328
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def timespan(start_time):
    """Return time in milliseconds from start_time"""

    timespan = datetime.datetime.now() - start_time
    timespan_ms = timespan.total_seconds() * 1000
    return timespan_ms

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

def info(txt):
    """Print, emphasized 'neutral', the given 'txt' message"""

    print("%s# %s%s%s" % (PR_EMPH_CC, get_time_stamp(), txt, PR_NC))
    sys.stdout.flush()

def decode_unicode_string(string):
    """
    Decode string encoded by `unicode_string`
    """
    if string.startswith('[BASE64-DATA]') and string.endswith('[/BASE64-DATA]'):
        return base64.b64decode(string[len('[BASE64-DATA]'):-len('[/BASE64-DATA]')])
    return string

def file_exists(fname):
    """Check if a file exists and is non-empty.
    """
    try:
        return fname and os.path.exists(fname) and os.path.getsize(fname) > 0
    except OSError:
        return False

def get_file_size(fileobj):
    """
    Returns the size of a file-like object.
    """
    currpos = fileobj.tell()
    fileobj.seek(0, 2)
    total_size = fileobj.tell()
    fileobj.seek(currpos)
    return total_size

def _strip_namespace(self, xml):
        """strips any namespaces from an xml string"""
        p = re.compile(b"xmlns=*[\"\"][^\"\"]*[\"\"]")
        allmatches = p.finditer(xml)
        for match in allmatches:
            xml = xml.replace(match.group(), b"")
        return xml

def caller_locals():
    """Get the local variables in the caller's frame."""
    import inspect
    frame = inspect.currentframe()
    try:
        return frame.f_back.f_back.f_locals
    finally:
        del frame

def barv(d, plt, title=None, rotation='vertical'):
    """A convenience function for plotting a vertical bar plot from a Counter"""
    labels = sorted(d, key=d.get, reverse=True)
    index = range(len(labels))
    plt.xticks(index, labels, rotation=rotation)
    plt.bar(index, [d[v] for v in labels])

    if title is not None:
        plt.title(title)

def prepare(doc):
    """Sets the caption_found and plot_found variables to False."""
    doc.caption_found = False
    doc.plot_found = False
    doc.listings_counter = 0

def val_to_bin(edges, x):
    """Convert axis coordinate to bin index."""
    ibin = np.digitize(np.array(x, ndmin=1), edges) - 1
    return ibin

def save_cache(data, filename):
    """Save cookies to a file."""
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle)

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

def format_line(data, linestyle):
    """Formats a list of elements using the given line style"""
    return linestyle.begin + linestyle.sep.join(data) + linestyle.end

def __init__(self, xmin=0, ymin=0, xmax=1, ymax=1):
        """
        Create the chart bounds with min max horizontal
        and vertical values
        """
        self._xmin = xmin
        self._ymin = ymin
        self._xmax = xmax
        self._ymax = ymax

def read_numpy(fd, byte_order, dtype, count):
    """Read tag data from file and return as numpy array."""
    return numpy.fromfile(fd, byte_order+dtype[-1], count)

def tree_render(request, upy_context, vars_dictionary):
    """
    It renders template defined in upy_context's page passed in arguments
    """
    page = upy_context['PAGE']
    return render_to_response(page.template.file_name, vars_dictionary, context_instance=RequestContext(request))

def byte2int(s, index=0):
    """Get the ASCII int value of a character in a string.

    :param s: a string
    :param index: the position of desired character

    :return: ASCII int value
    """
    if six.PY2:
        return ord(s[index])
    return s[index]

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

def gcd_float(numbers, tol=1e-8):
    """
    Returns the greatest common divisor for a sequence of numbers.
    Uses a numerical tolerance, so can be used on floats

    Args:
        numbers: Sequence of numbers.
        tol: Numerical tolerance

    Returns:
        (int) Greatest common divisor of numbers.
    """

    def pair_gcd_tol(a, b):
        """Calculate the Greatest Common Divisor of a and b.

        Unless b==0, the result will have the same sign as b (so that when
        b is divided by it, the result comes out positive).
        """
        while b > tol:
            a, b = b, a % b
        return a

    n = numbers[0]
    for i in numbers:
        n = pair_gcd_tol(n, i)
    return n

def rank(self):
        """how high in sorted list each key is. inverse permutation of sorter, such that sorted[rank]==keys"""
        r = np.empty(self.size, np.int)
        r[self.sorter] = np.arange(self.size)
        return r

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

def set_position(self, x, y, width, height):
        """Set window top-left corner position and size"""
        SetWindowPos(self._hwnd, None, x, y, width, height, ctypes.c_uint(0))

def remove_legend(ax=None):
    """Remove legend for axes or gca.

    See http://osdir.com/ml/python.matplotlib.general/2005-07/msg00285.html
    """
    from pylab import gca, draw
    if ax is None:
        ax = gca()
    ax.legend_ = None
    draw()

def get_ips(self, instance_id):
        """Retrieves all IP addresses associated to a given instance.

        :return: tuple (IPs)
        """
        instance = self._load_instance(instance_id)
        IPs = sum(instance.networks.values(), [])
        return IPs

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

def save_config_value(request, response, key, value):
    """Sets value of key `key` to `value` in both session and cookies."""
    request.session[key] = value
    response.set_cookie(key, value, expires=one_year_from_now())
    return response

def get_content_type (headers):
    """
    Get the MIME type from the Content-Type header value, or
    'application/octet-stream' if not found.

    @return: MIME type
    @rtype: string
    """
    ptype = headers.get('Content-Type', 'application/octet-stream')
    if ";" in ptype:
        # split off not needed extension info
        ptype = ptype.split(';')[0]
    return ptype.strip().lower()

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

def this_quarter():
        """ Return start and end date of this quarter. """
        since = TODAY + delta(day=1)
        while since.month % 3 != 0:
            since -= delta(months=1)
        until = since + delta(months=3)
        return Date(since), Date(until)

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

def _chunks(l, n):
    """ Yield successive n-sized chunks from l.

    http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def recursively_get_files_from_directory(directory):
    """
    Return all filenames under recursively found in a directory
    """
    return [
        os.path.join(root, filename)
        for root, directories, filenames in os.walk(directory)
        for filename in filenames
    ]

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

def connection_lost(self, exc):
        """Called when asyncio.Protocol loses the network connection."""
        if exc is None:
            self.log.warning('eof from receiver?')
        else:
            self.log.warning('Lost connection to receiver: %s', exc)

        self.transport = None

        if self._connection_lost_callback:
            self._loop.call_soon(self._connection_lost_callback)

def _value_to_color(value, cmap):
    """Convert a value in the range [0,1] to an RGB tuple using a colormap."""
    cm = plt.get_cmap(cmap)
    rgba = cm(value)
    return [int(round(255*v)) for v in rgba[0:3]]

def printcsv(csvdiffs):
    """print the csv"""
    for row in csvdiffs:
        print(','.join([str(cell) for cell in row]))

def all_strings(arr):
        """
        Ensures that the argument is a list that either is empty or contains only strings
        :param arr: list
        :return:
        """
        if not isinstance([], list):
            raise TypeError("non-list value found where list is expected")
        return all(isinstance(x, str) for x in arr)

def copy_to_temp(object):
    """
    Copy file-like object to temp file and return
    path.
    """
    temp_file = NamedTemporaryFile(delete=False)
    _copy_and_close(object, temp_file)
    return temp_file.name

def  make_html_code( self, lines ):
        """ convert a code sequence to HTML """
        line = code_header + '\n'
        for l in lines:
            line = line + html_quote( l ) + '\n'

        return line + code_footer

def screen_to_client(self, x, y):
        """
        Translates window screen coordinates to client coordinates.

        @note: This is a simplified interface to some of the functionality of
            the L{win32.Point} class.

        @see: {win32.Point.screen_to_client}

        @type  x: int
        @param x: Horizontal coordinate.
        @type  y: int
        @param y: Vertical coordinate.

        @rtype:  tuple( int, int )
        @return: Translated coordinates in a tuple (x, y).

        @raise WindowsError: An error occured while processing this request.
        """
        return tuple( win32.ScreenToClient( self.get_handle(), (x, y) ) )

def delaunay_2d(self, tol=1e-05, alpha=0.0, offset=1.0, bound=False):
        """Apply a delaunay 2D filter along the best fitting plane. This
        extracts the grid's points and perfoms the triangulation on those alone.
        """
        return PolyData(self.points).delaunay_2d(tol=tol, alpha=alpha, offset=offset, bound=bound)

def open_hdf5(filename, mode='r'):
    """Wrapper to open a :class:`h5py.File` from disk, gracefully
    handling a few corner cases
    """
    if isinstance(filename, (h5py.Group, h5py.Dataset)):
        return filename
    if isinstance(filename, FILE_LIKE):
        return h5py.File(filename.name, mode)
    return h5py.File(filename, mode)

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

def _using_stdout(self):
        """
        Return whether the handler is using sys.stdout.
        """
        if WINDOWS and colorama:
            # Then self.stream is an AnsiToWin32 object.
            return self.stream.wrapped is sys.stdout

        return self.stream is sys.stdout

def token_list_len(tokenlist):
    """
    Return the amount of characters in this token list.

    :param tokenlist: List of (token, text) or (token, text, mouse_handler)
                      tuples.
    """
    ZeroWidthEscape = Token.ZeroWidthEscape
    return sum(len(item[1]) for item in tokenlist if item[0] != ZeroWidthEscape)

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

def rowlenselect(table, n, complement=False):
    """Select rows of length `n`."""

    where = lambda row: len(row) == n
    return select(table, where, complement=complement)

def show_correlation_matrix(self, correlation_matrix):
        """Shows the given correlation matrix as image

        :param correlation_matrix: Correlation matrix of features
        """
        cr_plot.create_correlation_matrix_plot(
            correlation_matrix, self.title, self.headers_to_test
        )
        pyplot.show()

def search_index_file():
    """Return the default local index file, from the download cache"""
    from metapack import Downloader
    from os import environ

    return environ.get('METAPACK_SEARCH_INDEX',
                       Downloader.get_instance().cache.getsyspath('index.json'))

def jac(x,a):
    """ Jacobian matrix given Christophe's suggestion of f """
    return (x-a) / np.sqrt(((x-a)**2).sum(1))[:,np.newaxis]

def distinct_permutations(iterable):
    """Yield successive distinct permutations of the elements in *iterable*.

        >>> sorted(distinct_permutations([1, 0, 1]))
        [(0, 1, 1), (1, 0, 1), (1, 1, 0)]

    Equivalent to ``set(permutations(iterable))``, except duplicates are not
    generated and thrown away. For larger input sequences this is much more
    efficient.

    Duplicate permutations arise when there are duplicated elements in the
    input iterable. The number of items returned is
    `n! / (x_1! * x_2! * ... * x_n!)`, where `n` is the total number of
    items input, and each `x_i` is the count of a distinct item in the input
    sequence.

    """
    def make_new_permutations(permutations, e):
        """Internal helper function.
        The output permutations are built up by adding element *e* to the
        current *permutations* at every possible position.
        The key idea is to keep repeated elements (reverse) ordered:
        if e1 == e2 and e1 is before e2 in the iterable, then all permutations
        with e1 before e2 are ignored.

        """
        for permutation in permutations:
            for j in range(len(permutation)):
                yield permutation[:j] + [e] + permutation[j:]
                if permutation[j] == e:
                    break
            else:
                yield permutation + [e]

    permutations = [[]]
    for e in iterable:
        permutations = make_new_permutations(permutations, e)

    return (tuple(t) for t in permutations)

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

def reduce_fn(x):
    """
    Aggregation function to get the first non-zero value.
    """
    values = x.values if pd and isinstance(x, pd.Series) else x
    for v in values:
        if not is_nan(v):
            return v
    return np.NaN

def sorted_product_set(array_a, array_b):
  """Compute the product set of array_a and array_b and sort it."""
  return np.sort(
      np.concatenate(
          [array_a[i] * array_b for i in xrange(len(array_a))], axis=0)
  )[::-1]

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

def round_float(f, digits, rounding=ROUND_HALF_UP):
    """
    Accurate float rounding from http://stackoverflow.com/a/15398691.
    """
    return Decimal(str(f)).quantize(Decimal(10) ** (-1 * digits),
                                    rounding=rounding)

def replaceStrs(s, *args):
    r"""Replace all ``(frm, to)`` tuples in `args` in string `s`.

    >>> replaceStrs("nothing is better than warm beer",
    ...             ('nothing','warm beer'), ('warm beer','nothing'))
    'warm beer is better than nothing'

    """
    if args == (): return s
    mapping = dict((frm, to) for frm, to in args)
    return re.sub("|".join(map(re.escape, mapping.keys())),
                  lambda match:mapping[match.group(0)], s)

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

def __add_method(m: lmap.Map, key: T, method: Method) -> lmap.Map:
        """Swap the methods atom to include method with key."""
        return m.assoc(key, method)

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

def to_dicts(recarray):
    """convert record array to a dictionaries"""
    for rec in recarray:
        yield dict(zip(recarray.dtype.names, rec.tolist()))

def package_in_pypi(package):
    """Check whether the package is registered on pypi"""
    url = 'http://pypi.python.org/simple/%s' % package
    try:
        urllib.request.urlopen(url)
        return True
    except urllib.error.HTTPError as e:
        logger.debug("Package not found on pypi: %s", e)
        return False

def socket_close(self):
        """Close our socket."""
        if self.sock != NC.INVALID_SOCKET:
            self.sock.close()
        self.sock = NC.INVALID_SOCKET

def run_tests(self):
		"""
		Invoke pytest, replacing argv. Return result code.
		"""
		with _save_argv(_sys.argv[:1] + self.addopts):
			result_code = __import__('pytest').main()
			if result_code:
				raise SystemExit(result_code)

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

def instance_name(string):
    """Check for valid instance name
    """
    invalid = ':/@'
    if set(string).intersection(invalid):
        msg = 'Invalid instance name {}'.format(string)
        raise argparse.ArgumentTypeError(msg)
    return string

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

def export(defn):
    """Decorator to explicitly mark functions that are exposed in a lib."""
    globals()[defn.__name__] = defn
    __all__.append(defn.__name__)
    return defn

def __set_token_expired(self, value):
        """Internal helper for oauth code"""
        self._token_expired = datetime.datetime.now() + datetime.timedelta(seconds=value)
        return

def to_capitalized_camel_case(snake_case_string):
    """
    Convert a string from snake case to camel case with the first letter capitalized. For example, "some_var"
    would become "SomeVar".

    :param snake_case_string: Snake-cased string to convert to camel case.
    :returns: Camel-cased version of snake_case_string.
    """
    parts = snake_case_string.split('_')
    return ''.join([i.title() for i in parts])

def __str__(self):
        """Executes self.function to convert LazyString instance to a real
        str."""
        if not hasattr(self, '_str'):
            self._str=self.function(*self.args, **self.kwargs)
        return self._str

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

def get_hline():
    """ gets a horiztonal line """
    return Window(
        width=LayoutDimension.exact(1),
        height=LayoutDimension.exact(1),
        content=FillControl('-', token=Token.Line))

def mixedcase(path):
    """Removes underscores and capitalizes the neighbouring character"""
    words = path.split('_')
    return words[0] + ''.join(word.title() for word in words[1:])

def attribute(func):
    """Wrap a function as an attribute."""
    attr = abc.abstractmethod(func)
    attr.__iattribute__ = True
    attr = _property(attr)
    return attr

def create_app():
    """Create a Qt application."""
    global QT_APP
    QT_APP = QApplication.instance()
    if QT_APP is None:  # pragma: no cover
        QT_APP = QApplication(sys.argv)
    return QT_APP

def _check_whitespace(string):
    """
    Make sure thre is no whitespace in the given string. Will raise a
    ValueError if whitespace is detected
    """
    if string.count(' ') + string.count('\t') + string.count('\n') > 0:
        raise ValueError(INSTRUCTION_HAS_WHITESPACE)

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

def rollapply(data, window, fn):
    """
    Apply a function fn over a rolling window of size window.

    Args:
        * data (Series or DataFrame): Series or DataFrame
        * window (int): Window size
        * fn (function): Function to apply over the rolling window.
            For a series, the return value is expected to be a single
            number. For a DataFrame, it shuold return a new row.

    Returns:
        * Object of same dimensions as data
    """
    res = data.copy()
    res[:] = np.nan
    n = len(data)

    if window > n:
        return res

    for i in range(window - 1, n):
        res.iloc[i] = fn(data.iloc[i - window + 1:i + 1])

    return res

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

def getWindowPID(self, hwnd):
        """ Gets the process ID that the specified window belongs to """
        pid = ctypes.c_ulong()
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return int(pid.value)

def input(self,pin):
        """Read the specified pin and return HIGH/true if the pin is pulled high,
        or LOW/false if pulled low.
        """
        return self.mraa_gpio.Gpio.read(self.mraa_gpio.Gpio(pin))

def check_python_version():
    """Check if the currently running Python version is new enough."""
    # Required due to multiple with statements on one line
    req_version = (2, 7)
    cur_version = sys.version_info
    if cur_version >= req_version:
        print("Python version... %sOK%s (found %s, requires %s)" %
              (Bcolors.OKGREEN, Bcolors.ENDC, str(platform.python_version()),
               str(req_version[0]) + "." + str(req_version[1])))
    else:
        print("Python version... %sFAIL%s (found %s, requires %s)" %
              (Bcolors.FAIL, Bcolors.ENDC, str(cur_version),
               str(req_version)))

def get_memory_usage():
    """Gets RAM memory usage

    :return: MB of memory used by this process
    """
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss
    return mem / (1024 * 1024)

def remove_property(self, key=None, value=None):
        """Remove all properties matching both key and value.

        :param str key: Key of the property.
        :param str value: Value of the property.
        """
        for k, v in self.properties[:]:
            if (key is None or key == k) and (value is None or value == v):
                del(self.properties[self.properties.index((k, v))])

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

def search_overlap(self, point_list):
        """
        Returns all intervals that overlap the point_list.
        """
        result = set()
        for j in point_list:
            self.search_point(j, result)
        return result

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

def __init__(self, function):
		"""function: to be called with each stream element as its
		only argument
		"""
		super(filter, self).__init__()
		self.function = function

def __next__(self):
        """

        :return: a pair (1-based line number in the input, row)
        """
        # Retrieve the row, thereby incrementing the line number:
        row = super(UnicodeReaderWithLineNumber, self).__next__()
        return self.lineno + 1, row

def SegmentMax(a, ids):
    """
    Segmented max op.
    """
    func = lambda idxs: np.amax(a[idxs], axis=0)
    return seg_map(func, a, ids),

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

def clear_timeline(self):
        """
        Clear the contents of the TimeLine Canvas

        Does not modify the actual markers dictionary and thus after
        redrawing all markers are visible again.
        """
        self._timeline.delete(tk.ALL)
        self._canvas_ticks.delete(tk.ALL)

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

def drop_all_tables(self):
        """Drop all tables in the database"""
        for table_name in self.table_names():
            self.execute_sql("DROP TABLE %s" % table_name)
        self.connection.commit()

def add_to_toolbar(self, toolbar, widget):
        """Add widget actions to toolbar"""
        actions = widget.toolbar_actions
        if actions is not None:
            add_actions(toolbar, actions)

def _to_java_object_rdd(rdd):
    """ Return a JavaRDD of Object by unpickling

    It will convert each Python object into Java object by Pyrolite, whenever the
    RDD is serialized in batch or not.
    """
    rdd = rdd._reserialize(AutoBatchedSerializer(PickleSerializer()))
    return rdd.ctx._jvm.org.apache.spark.mllib.api.python.SerDe.pythonToJava(rdd._jrdd, True)

def check_str(obj):
        """ Returns a string for various input types """
        if isinstance(obj, str):
            return obj
        if isinstance(obj, float):
            return str(int(obj))
        else:
            return str(obj)

def get_the_node_dict(G, name):
    """
    Helper function that returns the node data
    of the node with the name supplied
    """
    for node in G.nodes(data=True):
        if node[0] == name:
            return node[1]

def offsets(self):
        """ Returns the offsets values of x, y, z as a numpy array
        """
        return np.array([self.x_offset, self.y_offset, self.z_offset])

def expect_comment_end(self):
        """Expect a comment end and return the match object.
        """
        match = self._expect_match('#}', COMMENT_END_PATTERN)
        self.advance(match.end())

def find_mapping(es_url, index):
    """ Find the mapping given an index """

    mapping = None

    backend = find_perceval_backend(es_url, index)

    if backend:
        mapping = backend.get_elastic_mappings()

    if mapping:
        logging.debug("MAPPING FOUND:\n%s", json.dumps(json.loads(mapping['items']), indent=True))
    return mapping

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

def method_name(func):
    """Method wrapper that adds the name of the method being called to its arguments list in Pascal case

    """
    @wraps(func)
    def _method_name(*args, **kwargs):
        name = to_pascal_case(func.__name__)
        return func(name=name, *args, **kwargs)
    return _method_name

def hex_escape(bin_str):
  """
  Hex encode a binary string
  """
  printable = string.ascii_letters + string.digits + string.punctuation + ' '
  return ''.join(ch if ch in printable else r'0x{0:02x}'.format(ord(ch)) for ch in bin_str)

def getfield(f):
    """convert values from cgi.Field objects to plain values."""
    if isinstance(f, list):
        return [getfield(x) for x in f]
    else:
        return f.value

def to_camel_case(text):
    """Convert to camel case.

    :param str text:
    :rtype: str
    :return:
    """
    split = text.split('_')
    return split[0] + "".join(x.title() for x in split[1:])

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

def _get_line_no_from_comments(py_line):
    """Return the line number parsed from the comment or 0."""
    matched = LINECOL_COMMENT_RE.match(py_line)
    if matched:
        return int(matched.group(1))
    else:
        return 0

def pop_row(self, idr=None, tags=False):
        """Pops a row, default the last"""
        idr = idr if idr is not None else len(self.body) - 1
        row = self.body.pop(idr)
        return row if tags else [cell.childs[0] for cell in row]

def flatten4d3d(x):
  """Flatten a 4d-tensor into a 3d-tensor by joining width and height."""
  xshape = shape_list(x)
  result = tf.reshape(x, [xshape[0], xshape[1] * xshape[2], xshape[3]])
  return result

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

def _deserialize_datetime(self, data):
        """Take any values coming in as a datetime and deserialize them

        """
        for key in data:
            if isinstance(data[key], dict):
                if data[key].get('type') == 'datetime':
                    data[key] = \
                        datetime.datetime.fromtimestamp(data[key]['value'])
        return data

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

def OnCellBackgroundColor(self, event):
        """Cell background color event handler"""

        with undo.group(_("Background color")):
            self.grid.actions.set_attr("bgcolor", event.color)

        self.grid.ForceRefresh()

        self.grid.update_attribute_toolbar()

        event.Skip()

def create(parallel):
    """Create a queue based on the provided parallel arguments.

    TODO Startup/tear-down. Currently using default queue for testing
    """
    queue = {k: v for k, v in parallel.items() if k in ["queue", "cores_per_job", "mem"]}
    yield queue

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