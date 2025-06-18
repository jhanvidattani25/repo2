def is_datetime_like(dtype):
    """Check if a dtype is a subclass of the numpy datetime types
    """
    return (np.issubdtype(dtype, np.datetime64) or
            np.issubdtype(dtype, np.timedelta64))

def notin(arg, values):
    """
    Like isin, but checks whether this expression's value(s) are not
    contained in the passed values. See isin docs for full usage.
    """
    op = ops.NotContains(arg, values)
    return op.to_expr()

def rgba_bytes_tuple(self, x):
        """Provides the color corresponding to value `x` in the
        form of a tuple (R,G,B,A) with int values between 0 and 255.
        """
        return tuple(int(u*255.9999) for u in self.rgba_floats_tuple(x))

def validate(raw_schema, target=None, **kwargs):
    """
    Given the python representation of a JSONschema as defined in the swagger
    spec, validate that the schema complies to spec.  If `target` is provided,
    that target will be validated against the provided schema.
    """
    schema = schema_validator(raw_schema, **kwargs)
    if target is not None:
        validate_object(target, schema=schema, **kwargs)

def Diag(a):
    """
    Diag op.
    """
    r = np.zeros(2 * a.shape, dtype=a.dtype)
    for idx, v in np.ndenumerate(a):
        r[2 * idx] = v
    return r,

def register(self, target):
        """Registers url_rules on the blueprint
        """
        for rule, options in self.url_rules:
            target.add_url_rule(rule, self.name, self.dispatch_request, **options)

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

def _stop_instance(self):
        """Stop the instance."""
        instance = self._get_instance()
        instance.stop()
        self._wait_on_instance('stopped', self.timeout)

def clean_int(x) -> int:
    """
    Returns its parameter as an integer, or raises
    ``django.forms.ValidationError``.
    """
    try:
        return int(x)
    except ValueError:
        raise forms.ValidationError(
            "Cannot convert to integer: {}".format(repr(x)))

def find_one_by_id(self, _id):
        """
        Find a single document by id

        :param str _id: BSON string repreentation of the Id
        :return: a signle object
        :rtype: dict

        """
        document = (yield self.collection.find_one({"_id": ObjectId(_id)}))
        raise Return(self._obj_cursor_to_dictionary(document))

def get_function(function_name):
    """
    Given a Python function name, return the function it refers to.
    """
    module, basename = str(function_name).rsplit('.', 1)
    try:
        return getattr(__import__(module, fromlist=[basename]), basename)
    except (ImportError, AttributeError):
        raise FunctionNotFound(function_name)

def println(msg):
    """
    Convenience function to print messages on a single line in the terminal
    """
    sys.stdout.write(msg)
    sys.stdout.flush()
    sys.stdout.write('\x08' * len(msg))
    sys.stdout.flush()

def prep_regex(patterns):
    """Compile regex patterns."""

    flags = 0 if Config.options.case_sensitive else re.I

    return [re.compile(pattern, flags) for pattern in patterns]

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

def aandb(a, b):
    """Return a matrix of logic comparison of A or B"""
    return matrix(np.logical_and(a, b).astype('float'), a.size)

def has_table(self, name):
        """Return ``True`` if the table *name* exists in the database."""
        return len(self.sql("SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                            parameters=(name,), asrecarray=False, cache=False)) > 0

def open_as_pillow(filename):
    """ This way can delete file immediately """
    with __sys_open(filename, 'rb') as f:
        data = BytesIO(f.read())
        return Image.open(data)

def unlock(self):
    """Closes the session to the database."""
    if not hasattr(self, 'session'):
      raise RuntimeError('Error detected! The session that you want to close does not exist any more!')
    logger.debug("Closed database session of '%s'" % self._database)
    self.session.close()
    del self.session

def mean_date(dt_list):
    """Calcuate mean datetime from datetime list
    """
    dt_list_sort = sorted(dt_list)
    dt_list_sort_rel = [dt - dt_list_sort[0] for dt in dt_list_sort]
    avg_timedelta = sum(dt_list_sort_rel, timedelta())/len(dt_list_sort_rel)
    return dt_list_sort[0] + avg_timedelta

def redirect_output(fileobj):
    """Redirect standard out to file."""
    old = sys.stdout
    sys.stdout = fileobj
    try:
        yield fileobj
    finally:
        sys.stdout = old

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

def register_plugin(self):
        """Register plugin in Spyder's main window"""
        self.main.restore_scrollbar_position.connect(
                                               self.restore_scrollbar_position)
        self.main.add_dockwidget(self)

def _distance(coord1, coord2):
    """
    Return the distance between two points, `coord1` and `coord2`. These
    parameters are assumed to be (x, y) tuples.
    """
    xdist = coord1[0] - coord2[0]
    ydist = coord1[1] - coord2[1]
    return sqrt(xdist*xdist + ydist*ydist)

def listify(a):
    """
    Convert a scalar ``a`` to a list and all iterables to list as well.

    Examples
    --------
    >>> listify(0)
    [0]

    >>> listify([1,2,3])
    [1, 2, 3]

    >>> listify('a')
    ['a']

    >>> listify(np.array([1,2,3]))
    [1, 2, 3]

    >>> listify('string')
    ['string']
    """
    if a is None:
        return []
    elif not isinstance(a, (tuple, list, np.ndarray)):
        return [a]
    return list(a)

def identify_request(request: RequestType) -> bool:
    """
    Try to identify whether this is an ActivityPub request.
    """
    # noinspection PyBroadException
    try:
        data = json.loads(decode_if_bytes(request.body))
        if "@context" in data:
            return True
    except Exception:
        pass
    return False

def hsv2rgb_spectrum(hsv):
    """Generates RGB values from HSV values in line with a typical light
    spectrum."""
    h, s, v = hsv
    return hsv2rgb_raw(((h * 192) >> 8, s, v))

def distance_matrix(trains1, trains2, cos, tau):
    """
    Return the *bipartite* (rectangular) distance matrix between the observations in the first and the second list.

    Convenience function; equivalent to ``dissimilarity_matrix(trains1, trains2, cos, tau, "distance")``. Refer to :func:`pymuvr.dissimilarity_matrix` for full documentation.
    """
    return dissimilarity_matrix(trains1, trains2, cos, tau, "distance")

def __getLogger(cls):
    """ Get the logger for this object.

    :returns: (Logger) A Logger object.
    """
    if cls.__logger is None:
      cls.__logger = opf_utils.initLogger(cls)
    return cls.__logger

def submit_the_only_form(self):
    """
    Look for a form on the page and submit it.

    Asserts if more than one form exists.
    """
    form = ElementSelector(world.browser, str('//form'))
    assert form, "Cannot find a form on the page."
    form.submit()

def try_cast_int(s):
    """(str) -> int
    All the digits in a given string are concatenated and converted into a single number.
    """
    try:
        temp = re.findall('\d', str(s))
        temp = ''.join(temp)
        return int(temp)
    except:
        return s

def rotate_img(im, deg, mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_AREA):
    """ Rotates an image by deg degrees

    Arguments:
        deg (float): degree to rotate.
    """
    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c//2,r//2),deg,1)
    return cv2.warpAffine(im,M,(c,r), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS+interpolation)

def mtf_unitransformer_all_layers_tiny():
  """Test out all the layers on local CPU."""
  hparams = mtf_unitransformer_tiny()
  hparams.moe_num_experts = 4
  hparams.moe_expert_x = 4
  hparams.moe_expert_y = 4
  hparams.moe_hidden_size = 512
  hparams.layers = ["self_att", "local_self_att", "moe_1d", "moe_2d", "drd"]
  return hparams

def assert_valid_name(name: str) -> str:
    """Uphold the spec rules about naming."""
    error = is_valid_name_error(name)
    if error:
        raise error
    return name

def get_pylint_options(config_dir='.'):
    # type: (str) -> List[str]
    """Checks for local config overrides for `pylint`
    and add them in the correct `pylint` `options` format.

    :param config_dir:
    :return: List [str]
    """
    if PYLINT_CONFIG_NAME in os.listdir(config_dir):
        pylint_config_path = PYLINT_CONFIG_NAME
    else:
        pylint_config_path = DEFAULT_PYLINT_CONFIG_PATH

    return ['--rcfile={}'.format(pylint_config_path)]

def _closeResources(self):
        """ Closes the root Dataset.
        """
        logger.info("Closing: {}".format(self._fileName))
        self._h5Group.close()
        self._h5Group = None

def _to_hours_mins_secs(time_taken):
    """Convert seconds to hours, mins, and seconds."""
    mins, secs = divmod(time_taken, 60)
    hours, mins = divmod(mins, 60)
    return hours, mins, secs

def ttl(self):
        """how long you should cache results for cacheable queries"""
        ret = 3600
        cn = self.get_process()
        if "ttl" in cn:
            ret = cn["ttl"]
        return ret

def _keys_to_camel_case(self, obj):
        """
        Make a copy of a dictionary with all keys converted to camel case. This is just calls to_camel_case on each of the keys in the dictionary and returns a new dictionary.

        :param obj: Dictionary to convert keys to camel case.
        :return: Dictionary with the input values and all keys in camel case
        """
        return dict((to_camel_case(key), value) for (key, value) in obj.items())

def layer_with(self, sample: np.ndarray, value: int) -> np.ndarray:
        """Create an identical 2d array where the second row is filled with value"""
        b = np.full((2, len(sample)), value, dtype=float)
        b[0] = sample
        return b

def unpack2D(_x):
    """
        Helper function for splitting 2D data into x and y component to make
        equations simpler
    """
    _x = np.atleast_2d(_x)
    x = _x[:, 0]
    y = _x[:, 1]
    return x, y

def getChildElementsByTagName(self, tagName):
    """ Return child elements of type tagName if found, else [] """
    result = []
    for child in self.childNodes:
        if isinstance(child, Element):
            if child.tagName == tagName:
                result.append(child)
    return result

def obj_in_list_always(target_list, obj):
    """
    >>> l = [1,1,1]
    >>> obj_in_list_always(l, 1)
    True
    >>> l.append(2)
    >>> obj_in_list_always(l, 1)
    False
    """
    for item in set(target_list):
        if item is not obj:
            return False
    return True

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

def qr(self,text):
        """ Print QR Code for the provided string """
        qr_code = qrcode.QRCode(version=4, box_size=4, border=1)
        qr_code.add_data(text)
        qr_code.make(fit=True)
        qr_img = qr_code.make_image()
        im = qr_img._img.convert("RGB")
        # Convert the RGB image in printable image
        self._convert_image(im)

def min(self):
        """
        :returns the minimum of the column
        """
        res = self._qexec("min(%s)" % self._name)
        if len(res) > 0:
            self._min = res[0][0]
        return self._min

def safe_exit(output):
    """exit without breaking pipes."""
    try:
        sys.stdout.write(output)
        sys.stdout.flush()
    except IOError:
        pass

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

def lengths( self ):
        """
        The cell lengths.

        Args:
            None

        Returns:
            (np.array(a,b,c)): The cell lengths.
        """
        return( np.array( [ math.sqrt( sum( row**2 ) ) for row in self.matrix ] ) )

def _get_tuple(self, fields):
    """
    :param fields: a list which contains either 0,1,or 2 values
    :return: a tuple with default values of '';
    """
    v1 = ''
    v2 = ''
    if len(fields) > 0:
      v1 = fields[0]
    if len(fields) > 1:
      v2 = fields[1]
    return v1, v2

def export_to_dot(self, filename: str = 'output') -> None:
        """ Export the graph to the dot file "filename.dot". """
        with open(filename + '.dot', 'w') as output:
            output.write(self.as_dot())

def stderr(a):
    """
    Calculate the standard error of a.
    """
    return np.nanstd(a) / np.sqrt(sum(np.isfinite(a)))

def should_be_hidden_as_cause(exc):
    """ Used everywhere to decide if some exception type should be displayed or hidden as the casue of an error """
    # reduced traceback in case of HasWrongType (instance_of checks)
    from valid8.validation_lib.types import HasWrongType, IsWrongType
    return isinstance(exc, (HasWrongType, IsWrongType))

def makeBiDirectional(d):
    """
    Helper for generating tagNameConverter
    Makes dict that maps from key to value and back
    """
    dTmp = d.copy()
    for k in d:
        dTmp[d[k]] = k
    return dTmp

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

def set_cache_max(self, cache_name, maxsize, **kwargs):
        """
        Sets the maxsize attribute of the named cache
        """
        cache = self._get_cache(cache_name)
        cache.set_maxsize(maxsize, **kwargs)

def update_hash_from_str(hsh, str_input):
    """
    Convert a str to object supporting buffer API and update a hash with it.
    """
    byte_input = str(str_input).encode("UTF-8")
    hsh.update(byte_input)

def add_element_to_doc(doc, tag, value):
    """Set text value of an etree.Element of tag, appending a new element with given tag if it doesn't exist."""
    element = doc.find(".//%s" % tag)
    if element is None:
        element = etree.SubElement(doc, tag)
    element.text = value

def out_shape_from_array(arr):
    """Get the output shape from an array."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.shape
    else:
        return (arr.shape[1],)

def parse_form(self, req, name, field):
        """Pull a form value from the request."""
        return core.get_value(req.POST, name, field)

def format_exc(limit=None):
    """Like print_exc() but return a string. Backport for Python 2.3."""
    try:
        etype, value, tb = sys.exc_info()
        return ''.join(traceback.format_exception(etype, value, tb, limit))
    finally:
        etype = value = tb = None

def get_attributes(var):
    """
    Given a varaible, return the list of attributes that are available inside
    of a template
    """
    is_valid = partial(is_valid_in_template, var)
    return list(filter(is_valid, dir(var)))

def set_slug(apps, schema_editor, class_name):
    """
    Create a slug for each Work already in the DB.
    """
    Cls = apps.get_model('spectator_events', class_name)

    for obj in Cls.objects.all():
        obj.slug = generate_slug(obj.pk)
        obj.save(update_fields=['slug'])

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

def get_window(self): 
        """
        Returns the object's parent window. Returns None if no window found.
        """
        x = self
        while not x._parent == None and \
              not isinstance(x._parent, Window): 
                  x = x._parent
        return x._parent

def stop(self):
        """Stops the background synchronization thread"""
        with self.synclock:
            if self.syncthread is not None:
                self.syncthread.cancel()
                self.syncthread = None

def GetValueByName(self, name):
    """Retrieves a value by name.

    Value names are not unique and pyregf provides first match for the value.

    Args:
      name (str): name of the value or an empty string for the default value.

    Returns:
      WinRegistryValue: Windows Registry value if a corresponding value was
          found or None if not.
    """
    pyregf_value = self._pyregf_key.get_value_by_name(name)
    if not pyregf_value:
      return None

    return REGFWinRegistryValue(pyregf_value)

def gettext(self, string, domain=None, **variables):
        """Translate a string with the current locale."""
        t = self.get_translations(domain)
        return t.ugettext(string) % variables

def _basic_field_data(field, obj):
    """Returns ``obj.field`` data as a dict"""
    value = field.value_from_object(obj)
    return {Field.TYPE: FieldType.VAL, Field.VALUE: value}

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

def email_type(arg):
	"""An argparse type representing an email address."""
	if not is_valid_email_address(arg):
		raise argparse.ArgumentTypeError("{0} is not a valid email address".format(repr(arg)))
	return arg

def area (self):
    """area() -> number

    Returns the area of this Polygon.
    """
    area = 0.0
    
    for segment in self.segments():
      area += ((segment.p.x * segment.q.y) - (segment.q.x * segment.p.y))/2

    return area

def construct_from_string(cls, string):
        """
        Construction from a string, raise a TypeError if not
        possible
        """
        if string == cls.name:
            return cls()
        raise TypeError("Cannot construct a '{}' from "
                        "'{}'".format(cls, string))

def _plot(self):
        """Plot stacked serie lines and stacked secondary lines"""
        for serie in self.series[::-1 if self.stack_from_top else 1]:
            self.line(serie)
        for serie in self.secondary_series[::-1 if self.stack_from_top else 1]:
            self.line(serie, True)

def _gcd_array(X):
    """
    Return the largest real value h such that all elements in x are integer
    multiples of h.
    """
    greatest_common_divisor = 0.0
    for x in X:
        greatest_common_divisor = _gcd(greatest_common_divisor, x)

    return greatest_common_divisor

def dot_product(self, other):
        """ Return the dot product of the given vectors. """
        return self.x * other.x + self.y * other.y

def parameter_vector(self):
        """An array of all parameters (including frozen parameters)"""
        return np.array([getattr(self, k) for k in self.parameter_names])

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

def pieces(array, chunk_size):
        """Yield successive chunks from array/list/string.
        Final chunk may be truncated if array is not evenly divisible by chunk_size."""
        for i in range(0, len(array), chunk_size): yield array[i:i+chunk_size]

def toggle_word_wrap(self):
        """
        Toggles document word wrap.

        :return: Method success.
        :rtype: bool
        """

        self.setWordWrapMode(not self.wordWrapMode() and QTextOption.WordWrap or QTextOption.NoWrap)
        return True

def quoted_or_list(items: List[str]) -> Optional[str]:
    """Given [A, B, C] return "'A', 'B', or 'C'".

    Note: We use single quotes here, since these are also used by repr().
    """
    return or_list([f"'{item}'" for item in items])

def safe_pow(base, exp):
    """safe version of pow"""
    if exp > MAX_EXPONENT:
        raise RuntimeError("Invalid exponent, max exponent is {}".format(MAX_EXPONENT))
    return base ** exp

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

def _tableExists(self, tableName):
        cursor=_conn.execute("""
            SELECT * FROM sqlite_master WHERE name ='{0}' and type='table';
        """.format(tableName))
        exists = cursor.fetchone() is not None
        cursor.close()
        return exists

def retrieve_asset(filename):
    """ Retrieves a non-image asset associated with an entry """

    record = model.Image.get(asset_name=filename)
    if not record:
        raise http_error.NotFound("File not found")
    if not record.is_asset:
        raise http_error.Forbidden()

    return flask.send_file(record.file_path, conditional=True)

def is_prime(n):
    """
    Check if n is a prime number
    """
    if n % 2 == 0 and n > 2:
        return False
    return all(n % i for i in range(3, int(math.sqrt(n)) + 1, 2))

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

def uniqify(cls, seq):
        """Returns a unique list of seq"""
        seen = set()
        seen_add = seen.add
        return [ x for x in seq if x not in seen and not seen_add(x)]

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

def _vector_or_scalar(x, type='row'):
    """Convert an object to either a scalar or a row or column vector."""
    if isinstance(x, (list, tuple)):
        x = np.array(x)
    if isinstance(x, np.ndarray):
        assert x.ndim == 1
        if type == 'column':
            x = x[:, None]
    return x

def have_pyrex():
    """
    Return True if Cython or Pyrex can be imported.
    """
    pyrex_impls = 'Cython.Distutils.build_ext', 'Pyrex.Distutils.build_ext'
    for pyrex_impl in pyrex_impls:
        try:
            # from (pyrex_impl) import build_ext
            __import__(pyrex_impl, fromlist=['build_ext']).build_ext
            return True
        except Exception:
            pass
    return False

def _pad(self):
    """Pads the output with an amount of indentation appropriate for the number of open element.

    This method does nothing if the indent value passed to the constructor is falsy.
    """
    if self._indent:
      self.whitespace(self._indent * len(self._open_elements))

def memory_full():
    """Check if the memory is too full for further caching."""
    current_process = psutil.Process(os.getpid())
    return (current_process.memory_percent() >
            config.MAXIMUM_CACHE_MEMORY_PERCENTAGE)

def vec_angle(a, b):
    """
    Calculate angle between two vectors
    """
    cosang = np.dot(a, b)
    sinang = fast_norm(np.cross(a, b))
    return np.arctan2(sinang, cosang)

def update_scale(self, value):
        """ updates the scale of all actors in the plotter """
        self.plotter.set_scale(self.x_slider_group.value,
                               self.y_slider_group.value,
                               self.z_slider_group.value)

def checksum(path):
    """Calculcate checksum for a file."""
    hasher = hashlib.sha1()
    with open(path, 'rb') as stream:
        buf = stream.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = stream.read(BLOCKSIZE)
    return hasher.hexdigest()

def filter_out(queryset, setting_name):
  """
  Remove unwanted results from queryset
  """
  kwargs = helpers.get_settings().get(setting_name, {}).get('FILTER_OUT', {})
  queryset = queryset.exclude(**kwargs)
  return queryset

def irecarray_to_py(a):
    """Slow conversion of a recarray into a list of records with python types.

    Get the field names from :attr:`a.dtype.names`.

    :Returns: iterator so that one can handle big input arrays
    """
    pytypes = [pyify(typestr) for name,typestr in a.dtype.descr]
    def convert_record(r):
        return tuple([converter(value) for converter, value in zip(pytypes,r)])
    return (convert_record(r) for r in a)

def _is_path(s):
    """Return whether an object is a path."""
    if isinstance(s, string_types):
        try:
            return op.exists(s)
        except (OSError, ValueError):
            return False
    else:
        return False

def get_free_memory_win():
    """Return current free memory on the machine for windows.

    Warning : this script is really not robust
    Return in MB unit
    """
    stat = MEMORYSTATUSEX()
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return int(stat.ullAvailPhys / 1024 / 1024)

def get_func_posargs_name(f):
    """Returns the name of the function f's keyword argument parameter if it exists, otherwise None"""
    sigparams = inspect.signature(f).parameters
    for p in sigparams:
        if sigparams[p].kind == inspect.Parameter.VAR_POSITIONAL:
            return sigparams[p].name
    return None

def array_bytes(array):
    """ Estimates the memory of the supplied array in bytes """
    return np.product(array.shape)*np.dtype(array.dtype).itemsize

def bash(filename):
    """Runs a bash script in the local directory"""
    sys.stdout.flush()
    subprocess.call("bash {}".format(filename), shell=True)

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

def cache(self):
        """Memoize access to the cache backend."""
        if self._cache is None:
            self._cache = django_cache.get_cache(self.cache_name)
        return self._cache

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

def _get_column_by_db_name(cls, name):
        """
        Returns the column, mapped by db_field name
        """
        return cls._columns.get(cls._db_map.get(name, name))

def position(self) -> Position:
        """The current position of the cursor."""
        return Position(self._index, self._lineno, self._col_offset)

def next(self):
        """Retrieve the next row."""
        # I'm pretty sure this is the completely wrong way to go about this, but
        # oh well, this works.
        if not hasattr(self, '_iter'):
            self._iter = self.readrow_as_dict()
        return self._iter.next()

def build(ctx):
    """Build documentation as HTML.

    The build HTML site is located in the ``doc/_build/html`` directory
    of the package.
    """
    return_code = run_sphinx(ctx.obj['root_dir'])
    if return_code > 0:
        sys.exit(return_code)

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

def _merge_args_with_kwargs(args_dict, kwargs_dict):
    """Merge args with kwargs."""
    ret = args_dict.copy()
    ret.update(kwargs_dict)
    return ret

def from_json(cls, s):
        """
        Restores the object from the given JSON.

        :param s: the JSON string to parse
        :type s: str
        :return: the
        """
        d = json.loads(s)
        return get_dict_handler(d["type"])(d)

def drop_post(self):
        """Remove .postXXXX postfix from version"""
        post_index = self.version.find('.post')
        if post_index >= 0:
            self.version = self.version[:post_index]

def skip_connection_distance(a, b):
    """The distance between two skip-connections."""
    if a[2] != b[2]:
        return 1.0
    len_a = abs(a[1] - a[0])
    len_b = abs(b[1] - b[0])
    return (abs(a[0] - b[0]) + abs(len_a - len_b)) / (max(a[0], b[0]) + max(len_a, len_b))

def _clean_up_name(self, name):
        """
        Cleans up the name according to the rules specified in this exact
        function. Uses self.naughty, a list of naughty characters.
        """
        for n in self.naughty: name = name.replace(n, '_')
        return name

def dimension_size(x, axis):
  """Returns the size of a specific dimension."""
  # Since tf.gather isn't "constant-in, constant-out", we must first check the
  # static shape or fallback to dynamic shape.
  s = tf.compat.dimension_value(
      tensorshape_util.with_rank_at_least(x.shape, np.abs(axis))[axis])
  if s is not None:
    return s
  return tf.shape(input=x)[axis]

def _openpyxl_read_xl(xl_path: str):
    """ Use openpyxl to read an Excel file. """
    try:
        wb = load_workbook(filename=xl_path, read_only=True)
    except:
        raise
    else:
        return wb

def set_parent_path(self, value):
        """
        Set the parent path and the path from the new parent path.

        :param value: The path to the object's parent
        """

        self._parent_path = value
        self.path = value + r'/' + self.name
        self._update_childrens_parent_path()

def _pdf_at_peak(self):
    """Pdf evaluated at the peak."""
    return (self.peak - self.low) / (self.high - self.low)

def directory_files(path):
    """Yield directory file names."""

    for entry in os.scandir(path):
        if not entry.name.startswith('.') and entry.is_file():
            yield entry.name

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

def _norm(self, x):
    """Compute the safe norm."""
    return tf.sqrt(tf.reduce_sum(tf.square(x), keepdims=True, axis=-1) + 1e-7)

def add_xlabel(self, text=None):
        """
        Add a label to the x-axis.
        """
        x = self.fit.meta['independent']
        if not text:
            text = '$' + x['tex_symbol'] + r'$ $(\si{' + x['siunitx'] +  r'})$'
        self.plt.set_xlabel(text)

def _deserialize(cls, key, value, fields):
        """ Marshal incoming data into Python objects."""
        converter = cls._get_converter_for_field(key, None, fields)
        return converter.deserialize(value)

def get_month_start(day=None):
    """Returns the first day of the given month."""
    day = add_timezone(day or datetime.date.today())
    return day.replace(day=1)

def display_len(text):
    """
    Get the display length of a string. This can differ from the character
    length if the string contains wide characters.
    """
    text = unicodedata.normalize('NFD', text)
    return sum(char_width(char) for char in text)

def resize(self):
        """
        Get target size for a cropped image and do the resizing if we got
        anything usable.
        """
        resized_size = self.get_resized_size()
        if not resized_size:
            return

        self.image = self.image.resize(resized_size, Image.ANTIALIAS)

def _parse_date(string: str) -> datetime.date:
    """Parse an ISO format date (YYYY-mm-dd).

    >>> _parse_date('1990-01-02')
    datetime.date(1990, 1, 2)
    """
    return datetime.datetime.strptime(string, '%Y-%m-%d').date()

def __add__(self, other):
        """Left addition."""
        return chaospy.poly.collection.arithmetics.add(self, other)

def _grammatical_join_filter(l, arg=None):
    """
    :param l: List of strings to join
    :param arg: A pipe-separated list of final_join (" and ") and
    initial_join (", ") strings. For example
    :return: A string that grammatically concatenates the items in the list.
    """
    if not arg:
        arg = " and |, "
    try:
        final_join, initial_joins = arg.split("|")
    except ValueError:
        final_join = arg
        initial_joins = ", "
    return grammatical_join(l, initial_joins, final_join)

def __getitem__(self, name):
        """
        A pymongo-like behavior for dynamically obtaining a collection of documents
        """
        if name not in self._collections:
            self._collections[name] = Collection(self.db, name)
        return self._collections[name]

def copy(self):
        """
        Creates a copy of model
        """
        return self.__class__(field_type=self.get_field_type(), data=self.export_data())

def create_db(app, appbuilder):
    """
        Create all your database objects (SQLAlchemy specific).
    """
    from flask_appbuilder.models.sqla import Base

    _appbuilder = import_application(app, appbuilder)
    engine = _appbuilder.get_session.get_bind(mapper=None, clause=None)
    Base.metadata.create_all(engine)
    click.echo(click.style("DB objects created", fg="green"))

def wait_send(self, timeout = None):
		"""Wait until all queued messages are sent."""
		self._send_queue_cleared.clear()
		self._send_queue_cleared.wait(timeout = timeout)

def pmon(month):
	"""
	P the month

	>>> print(pmon('2012-08'))
	August, 2012
	"""
	year, month = month.split('-')
	return '{month_name}, {year}'.format(
		month_name=calendar.month_name[int(month)],
		year=year,
	)

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

def convert_value(bind, value):
    """ Type casting. """
    type_name = get_type(bind)
    try:
        return typecast.cast(type_name, value)
    except typecast.ConverterError:
        return value

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

def scale_min(im, targ, interpolation=cv2.INTER_AREA):
    """ Scale the image so that the smallest axis is of size targ.

    Arguments:
        im (array): image
        targ (int): target size
    """
    r,c,*_ = im.shape
    ratio = targ/min(r,c)
    sz = (scale_to(c, ratio, targ), scale_to(r, ratio, targ))
    return cv2.resize(im, sz, interpolation=interpolation)

def describe_unique_1d(series):
    """Compute summary statistics of a unique (`S_TYPE_UNIQUE`) variable (a Series).

    Parameters
    ----------
    series : Series
        The variable to describe.

    Returns
    -------
    Series
        The description of the variable as a Series with index being stats keys.
    """
    return pd.Series([base.S_TYPE_UNIQUE], index=['type'], name=series.name)

def from_file(filename):
    """
    load an nparray object from a json filename

    @parameter str filename: path to the file
    """
    f = open(filename, 'r')
    j = json.load(f)
    f.close()

    return from_dict(j)

def fmt_sz(intval):
    """ Format a byte sized value.
    """
    try:
        return fmt.human_size(intval)
    except (ValueError, TypeError):
        return "N/A".rjust(len(fmt.human_size(0)))

def del_label(self, name):
        """Delete a label by name."""
        labels_tag = self.root[0]
        labels_tag.remove(self._find_label(name))

def destroy_webdriver(driver):
    """
    Destroy a driver
    """

    # This is some very flaky code in selenium. Hence the retries
    # and catch-all exceptions
    try:
        retry_call(driver.close, tries=2)
    except Exception:
        pass
    try:
        driver.quit()
    except Exception:
        pass

def __eq__(self, other):
        """Determine if two objects are equal."""
        return isinstance(other, self.__class__) \
            and self._freeze() == other._freeze()

def split_long_sentence(sentence, words_per_line):
    """Takes a sentence and adds a newline every "words_per_line" words.

    Parameters
    ----------
    sentence: str
        Sentene to split
    words_per_line: double
        Add a newline every this many words
    """
    words = sentence.split(' ')
    split_sentence = ''
    for i in range(len(words)):
        split_sentence = split_sentence + words[i]
        if (i+1) % words_per_line == 0:
            split_sentence = split_sentence + '\n'
        elif i != len(words) - 1:
            split_sentence = split_sentence + " "
    return split_sentence

def sort_filenames(filenames):
    """
    sort a list of files by filename only, ignoring the directory names
    """
    basenames = [os.path.basename(x) for x in filenames]
    indexes = [i[0] for i in sorted(enumerate(basenames), key=lambda x:x[1])]
    return [filenames[x] for x in indexes]

def file_found(filename,force):
    """Check if a file exists"""
    if os.path.exists(filename) and not force:
        logger.info("Found %s; skipping..."%filename)
        return True
    else:
        return False

def gcall(func, *args, **kwargs):
    """
    Calls a function, with the given arguments inside Gtk's main loop.
    Example::
        gcall(lbl.set_text, "foo")

    If this call would be made in a thread there could be problems, using
    it inside Gtk's main loop makes it thread safe.
    """
    def idle():
        with gdk.lock:
            return bool(func(*args, **kwargs))
    return gobject.idle_add(idle)

def open_store_variable(self, name, var):
        """Turn CDMRemote variable into something like a numpy.ndarray."""
        data = indexing.LazilyOuterIndexedArray(CDMArrayWrapper(name, self))
        return Variable(var.dimensions, data, {a: getattr(var, a) for a in var.ncattrs()})

def is_valid_image_extension(file_path):
    """is_valid_image_extension."""
    valid_extensions = ['.jpeg', '.jpg', '.gif', '.png']
    _, extension = os.path.splitext(file_path)
    return extension.lower() in valid_extensions

def zoom_cv(x,z):
    """ Zoom the center of image x by a factor of z+1 while retaining the original image size and proportion. """
    if z==0: return x
    r,c,*_ = x.shape
    M = cv2.getRotationMatrix2D((c/2,r/2),0,z+1.)
    return cv2.warpAffine(x,M,(c,r))

def get_key_by_value(dictionary, search_value):
    """
    searchs a value in a dicionary and returns the key of the first occurrence

    :param dictionary: dictionary to search in
    :param search_value: value to search for
    """
    for key, value in dictionary.iteritems():
        if value == search_value:
            return ugettext(key)

def list_view_changed(self, widget, event, data=None):
        """
        Function shows last rows.
        """
        adj = self.scrolled_window.get_vadjustment()
        adj.set_value(adj.get_upper() - adj.get_page_size())

def list_of(cls):
    """
    Returns a function that checks that each element in a
    list is of a specific type.
    """
    return lambda l: isinstance(l, list) and all(isinstance(x, cls) for x in l)

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

def batch(input_iter, batch_size=32):
  """Batches data from an iterator that returns single items at a time."""
  input_iter = iter(input_iter)
  next_ = list(itertools.islice(input_iter, batch_size))
  while next_:
    yield next_
    next_ = list(itertools.islice(input_iter, batch_size))

def pool_args(function, sequence, kwargs):
    """Return a single iterator of n elements of lists of length 3, given a sequence of len n."""
    return zip(itertools.repeat(function), sequence, itertools.repeat(kwargs))

async def result_processor(tasks):
    """An async result aggregator that combines all the results
       This gets executed in unsync.loop and unsync.thread"""
    output = {}
    for task in tasks:
        num, res = await task
        output[num] = res
    return output

def _try_lookup(table, value, default = ""):
    """ try to get a string from the lookup table, return "" instead of key
    error
    """
    try:
        string = table[ value ]
    except KeyError:
        string = default
    return string

def _get_set(self, key, operation, create=False):
        """
        Get (and maybe create) a set by name.
        """
        return self._get_by_type(key, operation, create, b'set', set())

def session(self):
        """A context manager for this client's session.

        This function closes the current session when this client goes out of
        scope.
        """
        self._session = requests.session()
        yield
        self._session.close()
        self._session = None

def _find_value(key, *args):
    """Find a value for 'key' in any of the objects given as 'args'"""
    for arg in args:
        v = _get_value(arg, key)
        if v is not None:
            return v

async def packets_from_tshark(self, packet_callback, packet_count=None, close_tshark=True):
        """
        A coroutine which creates a tshark process, runs the given callback on each packet that is received from it and
        closes the process when it is done.

        Do not use interactively. Can be used in order to insert packets into your own eventloop.
        """
        tshark_process = await self._get_tshark_process(packet_count=packet_count)
        try:
            await self._go_through_packets_from_fd(tshark_process.stdout, packet_callback, packet_count=packet_count)
        except StopCapture:
            pass
        finally:
            if close_tshark:
                await self._close_async()

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

def _extract_value(self, value):
        """If the value is true/false/null replace with Python equivalent."""
        return ModelEndpoint._value_map.get(smart_str(value).lower(), value)

def cli_run():
    """docstring for argparse"""
    parser = argparse.ArgumentParser(description='Stupidly simple code answers from StackOverflow')
    parser.add_argument('query', help="What's the problem ?", type=str, nargs='+')
    parser.add_argument('-t','--tags', help='semicolon separated tags -> python;lambda')
    args = parser.parse_args()
    main(args)

def ignore_comments(iterator):
    """
    Strips and filters empty or commented lines.
    """
    for line in iterator:
        line = COMMENT_RE.sub('', line)
        line = line.strip()
        if line:
            yield line

def parse_json_date(value):
    """
    Parses an ISO8601 formatted datetime from a string value
    """
    if not value:
        return None

    return datetime.datetime.strptime(value, JSON_DATETIME_FORMAT).replace(tzinfo=pytz.UTC)

def __str__(self):
        """Return human readable string."""
        return ", ".join("{:02x}{:02x}={:02x}{:02x}".format(c[0][0], c[0][1], c[1][0], c[1][1]) for c in self.alias_array_)

def guess_media_type(filepath):
    """Returns the media-type of the file at the given ``filepath``"""
    o = subprocess.check_output(['file', '--mime-type', '-Lb', filepath])
    o = o.strip()
    return o

def _longest_val_in_column(self, col):
        """
        get size of longest value in specific column

        :param col: str, column name
        :return int
        """
        try:
            # +2 is for implicit separator
            return max([len(x[col]) for x in self.table if x[col]]) + 2
        except KeyError:
            logger.error("there is no column %r", col)
            raise

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

def has_key(cls, *args):
        """
        Check whether flyweight object with specified key has already been created.

        Returns:
            bool: True if already created, False if not
        """
        key = args if len(args) > 1 else args[0]
        return key in cls._instances

def left_zero_pad(s, blocksize):
    """
    Left padding with zero bytes to a given block size

    :param s:
    :param blocksize:
    :return:
    """
    if blocksize > 0 and len(s) % blocksize:
        s = (blocksize - len(s) % blocksize) * b('\000') + s
    return s

def hline(self, x, y, width, color):
        """Draw a horizontal line up to a given length."""
        self.rect(x, y, width, 1, color, fill=True)

def flush(self):
        """
        Ensure all logging output has been flushed
        """
        if len(self._buffer) > 0:
            self.logger.log(self.level, self._buffer)
            self._buffer = str()

def clip_to_seconds(m: Union[int, pd.Series]) -> Union[int, pd.Series]:
        """Clips UTC datetime in nanoseconds to seconds."""
        return m // pd.Timedelta(1, unit='s').value

def is_set(self, key):
        """Return True if variable is a set"""
        data = self.model.get_data()
        return isinstance(data[key], set)

def read_byte_data(self, addr, cmd):
        """read_byte_data(addr, cmd) -> result

        Perform SMBus Read Byte Data transaction.
        """
        self._set_addr(addr)
        res = SMBUS.i2c_smbus_read_byte_data(self._fd, ffi.cast("__u8", cmd))
        if res == -1:
            raise IOError(ffi.errno)
        return res

def long_substr(data):
    """Return the longest common substring in a list of strings.
    
    Credit: http://stackoverflow.com/questions/2892931/longest-common-substring-from-more-than-two-strings-python
    """
    substr = ''
    if len(data) > 1 and len(data[0]) > 0:
        for i in range(len(data[0])):
            for j in range(len(data[0])-i+1):
                if j > len(substr) and all(data[0][i:i+j] in x for x in data):
                    substr = data[0][i:i+j]
    elif len(data) == 1:
        substr = data[0]
    return substr

def GetAllPixelColors(self) -> ctypes.Array:
        """
        Return `ctypes.Array`, an iterable array of int values in argb.
        """
        return self.GetPixelColorsOfRect(0, 0, self.Width, self.Height)

def validate_string_list(lst):
    """Validate that the input is a list of strings.

    Raises ValueError if not."""
    if not isinstance(lst, list):
        raise ValueError('input %r must be a list' % lst)
    for x in lst:
        if not isinstance(x, basestring):
            raise ValueError('element %r in list must be a string' % x)

def save_dot(self, fd):
        """ Saves a representation of the case in the Graphviz DOT language.
        """
        from pylon.io import DotWriter
        DotWriter(self).write(fd)

def from_timestamp(microsecond_timestamp):
    """Convert a microsecond timestamp to a UTC datetime instance."""
    # Create datetime without losing precision from floating point (yes, this
    # is actually needed):
    return datetime.datetime.fromtimestamp(
        microsecond_timestamp // 1000000, datetime.timezone.utc
    ).replace(microsecond=(microsecond_timestamp % 1000000))

def is_connected(self):
        """
        Return true if the socket managed by this connection is connected

        :rtype: bool
        """
        try:
            return self.socket is not None and self.socket.getsockname()[1] != 0 and BaseTransport.is_connected(self)
        except socket.error:
            return False

def move_to(x, y):
    """Moves the brush to a particular position.

    Arguments:
        x - a number between -250 and 250.
        y - a number between -180 and 180.
    """
    _make_cnc_request("coord/{0}/{1}".format(x, y))
    state['turtle'].goto(x, y)

def call_alias(self, alias, rest=''):
        """Call an alias given its name and the rest of the line."""
        cmd = self.transform_alias(alias, rest)
        try:
            self.shell.system(cmd)
        except:
            self.shell.showtraceback()

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

def is_end_of_month(self) -> bool:
        """ Checks if the date is at the end of the month """
        end_of_month = Datum()
        # get_end_of_month(value)
        end_of_month.end_of_month()
        return self.value == end_of_month.value

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

def _strptime(self, time_str):
        """Convert an ISO 8601 formatted string in UTC into a
        timezone-aware datetime object."""
        if time_str:
            # Parse UTC string into naive datetime, then add timezone
            dt = datetime.strptime(time_str, __timeformat__)
            return dt.replace(tzinfo=UTC())
        return None

def gday_of_year(self):
        """Return the number of days since January 1 of the given year."""
        return (self.date - dt.date(self.date.year, 1, 1)).days

def _run_parallel_process_with_profiling(self, start_path, stop_path, queue, filename):
        """
        wrapper for usage of profiling
        """
        runctx('Engine._run_parallel_process(self,  start_path, stop_path, queue)', globals(), locals(), filename)

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

def lpush(self, key, *args):
        """Emulate lpush."""
        redis_list = self._get_list(key, 'LPUSH', create=True)

        # Creates the list at this key if it doesn't exist, and appends args to its beginning
        args_reversed = [self._encode(arg) for arg in args]
        args_reversed.reverse()
        updated_list = args_reversed + redis_list
        self.redis[self._encode(key)] = updated_list

        # Return the length of the list after the push operation
        return len(updated_list)

def get_creation_datetime(filepath):
    """
    Get the date that a file was created.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    creation_datetime : datetime.datetime or None
    """
    if platform.system() == 'Windows':
        return datetime.fromtimestamp(os.path.getctime(filepath))
    else:
        stat = os.stat(filepath)
        try:
            return datetime.fromtimestamp(stat.st_birthtime)
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return None

def on_success(self, fn, *args, **kwargs):
        """
        Call the given callback if or when the connected deferred succeeds.

        """

        self._callbacks.append((fn, args, kwargs))

        result = self._resulted_in
        if result is not _NOTHING_YET:
            self._succeed(result=result)

def stretch(iterable, n=2):
    r"""Repeat each item in `iterable` `n` times.

    Example:

    >>> list(stretch(range(3), 2))
    [0, 0, 1, 1, 2, 2]
    """
    times = range(n)
    for item in iterable:
        for i in times: yield item

def isBlockComment(self, line, column):
        """Check if text at given position is a block comment.

        If language is not known, or text is not parsed yet, ``False`` is returned
        """
        return self._highlighter is not None and \
               self._highlighter.isBlockComment(self.document().findBlockByNumber(line), column)

def ansi(color, text):
    """Wrap text in an ansi escape sequence"""
    code = COLOR_CODES[color]
    return '\033[1;{0}m{1}{2}'.format(code, text, RESET_TERM)

def stopwatch_now():
    """Get a timevalue for interval comparisons

    When possible it is a monotonic clock to prevent backwards time issues.
    """
    if six.PY2:
        now = time.time()
    else:
        now = time.monotonic()
    return now

def __add_namespaceinfo(self, ni):
        """Internal method to directly add a _NamespaceInfo object to this
        set.  No sanity checks are done (e.g. checking for prefix conflicts),
        so be sure to do it yourself before calling this."""
        self.__ns_uri_map[ni.uri] = ni
        for prefix in ni.prefixes:
            self.__prefix_map[prefix] = ni

def range(*args, interval=0):
    """Generate a given range of numbers.

    It supports the same arguments as the builtin function.
    An optional interval can be given to space the values out.
    """
    agen = from_iterable.raw(builtins.range(*args))
    return time.spaceout.raw(agen, interval) if interval else agen

def is_orthogonal(
        matrix: np.ndarray,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8) -> bool:
    """Determines if a matrix is approximately orthogonal.

    A matrix is orthogonal if it's square and real and its transpose is its
    inverse.

    Args:
        matrix: The matrix to check.
        rtol: The per-matrix-entry relative tolerance on equality.
        atol: The per-matrix-entry absolute tolerance on equality.

    Returns:
        Whether the matrix is orthogonal within the given tolerance.
    """
    return (matrix.shape[0] == matrix.shape[1] and
            np.all(np.imag(matrix) == 0) and
            np.allclose(matrix.dot(matrix.T), np.eye(matrix.shape[0]),
                        rtol=rtol,
                        atol=atol))

def _duplicates(list_):
    """Return dict mapping item -> indices."""
    item_indices = {}
    for i, item in enumerate(list_):
        try:
            item_indices[item].append(i)
        except KeyError:  # First time seen
            item_indices[item] = [i]
    return item_indices

def vadd(v1, v2):
    """ Add two 3 dimensional vectors.
    http://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/vadd_c.html

    :param v1: First vector to be added. 
    :type v1: 3-Element Array of floats
    :param v2: Second vector to be added. 
    :type v2: 3-Element Array of floats
    :return: v1+v2
    :rtype: 3-Element Array of floats
    """
    v1 = stypes.toDoubleVector(v1)
    v2 = stypes.toDoubleVector(v2)
    vout = stypes.emptyDoubleVector(3)
    libspice.vadd_c(v1, v2, vout)
    return stypes.cVectorToPython(vout)

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

def server(request):
    """
    Respond to requests for the server's primary web page.
    """
    return direct_to_template(
        request,
        'server/index.html',
        {'user_url': getViewURL(request, idPage),
         'server_xrds_url': getViewURL(request, idpXrds),
         })

def unique_list(lst):
    """Make a list unique, retaining order of initial appearance."""
    uniq = []
    for item in lst:
        if item not in uniq:
            uniq.append(item)
    return uniq

def remove_prefix(text, prefix):
	"""
	Remove the prefix from the text if it exists.

	>>> remove_prefix('underwhelming performance', 'underwhelming ')
	'performance'

	>>> remove_prefix('something special', 'sample')
	'something special'
	"""
	null, prefix, rest = text.rpartition(prefix)
	return rest

def clean_dict_keys(d):
    """Convert all keys of the dict 'd' to (ascii-)strings.

    :Raises: UnicodeEncodeError
    """
    new_d = {}
    for (k, v) in d.iteritems():
        new_d[str(k)] = v
    return new_d

def _is_one_arg_pos_call(call):
    """Is this a call with exactly 1 argument,
    where that argument is positional?
    """
    return isinstance(call, astroid.Call) and len(call.args) == 1 and not call.keywords

def autoconvert(string):
    """Try to convert variables into datatypes."""
    for fn in (boolify, int, float):
        try:
            return fn(string)
        except ValueError:
            pass
    return string

def from_rectangle(box):
        """ Create a vector randomly within the given rectangle. """
        x = box.left + box.width * random.uniform(0, 1)
        y = box.bottom + box.height * random.uniform(0, 1)
        return Vector(x, y)

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

def _validate_input_data(self, data, request):
        """ Validate input data.

        :param request: the HTTP request
        :param data: the parsed data
        :return: if validation is performed and succeeds the data is converted
                 into whatever format the validation uses (by default Django's
                 Forms) If not, the data is returned unchanged.
        :raises: HttpStatusCodeError if data is not valid
        """

        validator = self._get_input_validator(request)
        if isinstance(data, (list, tuple)):
            return map(validator.validate, data)
        else:
            return validator.validate(data)

def flatten_multidict(multidict):
    """Return flattened dictionary from ``MultiDict``."""
    return dict([(key, value if len(value) > 1 else value[0])
                 for (key, value) in multidict.iterlists()])

def wait_run_in_executor(func, *args, **kwargs):
    """
    Run blocking code in a different thread and wait
    for the result.

    :param func: Run this function in a different thread
    :param args: Parameters of the function
    :param kwargs: Keyword parameters of the function
    :returns: Return the result of the function
    """

    loop = asyncio.get_event_loop()
    future = loop.run_in_executor(None, functools.partial(func, *args, **kwargs))
    yield from asyncio.wait([future])
    return future.result()

def iter_finds(regex_obj, s):
    """Generate all matches found within a string for a regex and yield each match as a string"""
    if isinstance(regex_obj, str):
        for m in re.finditer(regex_obj, s):
            yield m.group()
    else:
        for m in regex_obj.finditer(s):
            yield m.group()

def contextMenuEvent(self, event):
        """Override Qt method"""
        self.update_menu()
        self.menu.popup(event.globalPos())

def get_focused_window_sane(self):
        """
        Like xdo_get_focused_window, but return the first ancestor-or-self
        window * having a property of WM_CLASS. This allows you to get
        the "real" or top-level-ish window having focus rather than something
        you may not expect to be the window having focused.

        :param window_ret:
            Pointer to a window where the currently-focused window
            will be stored.
        """
        window_ret = window_t(0)
        _libxdo.xdo_get_focused_window_sane(
            self._xdo, ctypes.byref(window_ret))
        return window_ret.value

def cookies(self) -> Dict[str, str]:
        """The parsed cookies attached to this request."""
        cookies = SimpleCookie()
        cookies.load(self.headers.get('Cookie', ''))
        return {key: cookie.value for key, cookie in cookies.items()}

def do_files_exist(filenames):
  """Whether any of the filenames exist."""
  preexisting = [tf.io.gfile.exists(f) for f in filenames]
  return any(preexisting)

def writeCSV(data, headers, csvFile):
  """Write data with column headers to a CSV."""
  with open(csvFile, "wb") as f:
    writer = csv.writer(f, delimiter=",")
    writer.writerow(headers)
    writer.writerows(data)

def sine_wave(frequency):
  """Emit a sine wave at the given frequency."""
  xs = tf.reshape(tf.range(_samples(), dtype=tf.float32), [1, _samples(), 1])
  ts = xs / FLAGS.sample_rate
  return tf.sin(2 * math.pi * frequency * ts)

def set_verbosity(verbosity):
        """Banana banana
        """
        Logger._verbosity = min(max(0, WARNING - verbosity), 2)
        debug("Verbosity set to %d" % (WARNING - Logger._verbosity), 'logging')

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

def logger(message, level=10):
    """Handle logging."""
    logging.getLogger(__name__).log(level, str(message))

def read_set_from_file(filename: str) -> Set[str]:
    """
    Extract a de-duped collection (set) of text from a file.
    Expected file format is one item per line.
    """
    collection = set()
    with open(filename, 'r') as file_:
        for line in file_:
            collection.add(line.rstrip())
    return collection

def log(self, level, msg=None, *args, **kwargs):
        """Writes log out at any arbitray level."""

        return self._log(level, msg, args, kwargs)

def scale_dtype(arr, dtype):
    """Convert an array from 0..1 to dtype, scaling up linearly
    """
    max_int = np.iinfo(dtype).max
    return (arr * max_int).astype(dtype)

def validate(self, obj):
        """ Raises django.core.exceptions.ValidationError if any validation error exists """

        if not isinstance(obj, self.model_class):
            raise ValidationError('Invalid object(%s) for service %s' % (type(obj), type(self)))
        LOG.debug(u'Object %s state: %s', self.model_class, obj.__dict__)
        obj.full_clean()

def get_feature_order(dataset, features):
    """ Returns a list with the order that features requested appear in
    dataset """
    all_features = dataset.get_feature_names()

    i = [all_features.index(f) for f in features]

    return i

def acknowledge_time(self):
        """
        Processor time when the alarm was acknowledged.

        :type: :class:`~datetime.datetime`
        """
        if (self.is_acknowledged and
                self._proto.acknowledgeInfo.HasField('acknowledgeTime')):
            return parse_isostring(self._proto.acknowledgeInfo.acknowledgeTime)
        return None

def selectgt(table, field, value, complement=False):
    """Select rows where the given field is greater than the given value."""

    value = Comparable(value)
    return selectop(table, field, value, operator.gt, complement=complement)

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

def group_exists(groupname):
    """Check if a group exists"""
    try:
        grp.getgrnam(groupname)
        group_exists = True
    except KeyError:
        group_exists = False
    return group_exists

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

def closeEvent(self, event):
        """closeEvent reimplementation"""
        if self.closing(True):
            event.accept()
        else:
            event.ignore()

def close( self ):
        """
        Close the db and release memory
        """
        if self.db is not None:
            self.db.commit()
            self.db.close()
            self.db = None

        return

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

def speedtest(func, *args, **kwargs):
    """ Test the speed of a function. """
    n = 100
    start = time.time()
    for i in range(n): func(*args,**kwargs)
    end = time.time()
    return (end-start)/n

def _get_col_index(name):
    """Convert column name to index."""

    index = string.ascii_uppercase.index
    col = 0
    for c in name.upper():
        col = col * 26 + index(c) + 1
    return col

def __call__(self, _):
        """Update the progressbar."""
        if self.iter % self.step == 0:
            self.pbar.update(self.step)

        self.iter += 1

def returns(self) -> T.Optional[DocstringReturns]:
        """Return return information indicated in docstring."""
        try:
            return next(
                DocstringReturns.from_meta(meta)
                for meta in self.meta
                if meta.args[0] in {"return", "returns", "yield", "yields"}
            )
        except StopIteration:
            return None

def _maybe_pandas_data(data, feature_names, feature_types):
    """ Extract internal data from pd.DataFrame for DMatrix data """

    if not isinstance(data, DataFrame):
        return data, feature_names, feature_types

    data_dtypes = data.dtypes
    if not all(dtype.name in PANDAS_DTYPE_MAPPER for dtype in data_dtypes):
        bad_fields = [data.columns[i] for i, dtype in
                      enumerate(data_dtypes) if dtype.name not in PANDAS_DTYPE_MAPPER]

        msg = """DataFrame.dtypes for data must be int, float or bool.
                Did not expect the data types in fields """
        raise ValueError(msg + ', '.join(bad_fields))

    if feature_names is None:
        if isinstance(data.columns, MultiIndex):
            feature_names = [
                ' '.join([str(x) for x in i])
                for i in data.columns
            ]
        else:
            feature_names = data.columns.format()

    if feature_types is None:
        feature_types = [PANDAS_DTYPE_MAPPER[dtype.name] for dtype in data_dtypes]

    data = data.values.astype('float')

    return data, feature_names, feature_types

def __connect():
    """
    Connect to a redis instance.
    """
    global redis_instance
    if use_tcp_socket:
        redis_instance = redis.StrictRedis(host=hostname, port=port)
    else:
        redis_instance = redis.StrictRedis(unix_socket_path=unix_socket)

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

def to_identifier(s):
  """
  Convert snake_case to camel_case.
  """
  if s.startswith('GPS'):
      s = 'Gps' + s[3:]
  return ''.join([i.capitalize() for i in s.split('_')]) if '_' in s else s

def get_enum_from_name(self, enum_name):
        """
            Return an enum from a name
        Args:
            enum_name (str): name of the enum
        Returns:
            Enum
        """
        return next((e for e in self.enums if e.name == enum_name), None)

def get_var_type(self, name):
        """
        Return type string, compatible with numpy.
        """
        name = create_string_buffer(name)
        type_ = create_string_buffer(MAXSTRLEN)
        self.library.get_var_type.argtypes = [c_char_p, c_char_p]
        self.library.get_var_type(name, type_)
        return type_.value

def _type_bool(label,default=False):
    """Shortcut fot boolean like fields"""
    return label, abstractSearch.nothing, abstractRender.boolen, default

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

def display_iframe_url(target, **kwargs):
    """Display the contents of a URL in an IPython notebook.
    
    :param target: the target url.
    :type target: string

    .. seealso:: `iframe_url()` for additional arguments."""

    txt = iframe_url(target, **kwargs)
    display(HTML(txt))

def show_tip(self, tip=""):
        """Show tip"""
        QToolTip.showText(self.mapToGlobal(self.pos()), tip, self)

def bundle_dir():
    """Handle resource management within an executable file."""
    if frozen():
        directory = sys._MEIPASS
    else:
        directory = os.path.dirname(os.path.abspath(stack()[1][1]))
    if os.path.exists(directory):
        return directory

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

def ancestors(self, node):
        """Returns set of the ancestors of a node as DAGNodes."""
        if isinstance(node, int):
            warnings.warn('Calling ancestors() with a node id is deprecated,'
                          ' use a DAGNode instead',
                          DeprecationWarning, 2)
            node = self._id_to_node[node]

        return nx.ancestors(self._multi_graph, node)

def exit(self):
        """Stop the simple WSGI server running the appliation."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

def multivariate_normal_tril(x,
                             dims,
                             layer_fn=tf.compat.v1.layers.dense,
                             loc_fn=lambda x: x,
                             scale_fn=tril_with_diag_softplus_and_shift,
                             name=None):
  """Constructs a trainable `tfd.MultivariateNormalTriL` distribution.

  This function creates a MultivariateNormal (MVN) with lower-triangular scale
  matrix. By default the MVN is parameterized via affine transformation of input
  tensor `x`. Using default args, this function is mathematically equivalent to:

  ```none
  Y = MVN(loc=matmul(W, x) + b,
          scale_tril=f(reshape_tril(matmul(M, x) + c)))

  where,
    W in R^[d, n]
    M in R^[d*(d+1)/2, n]
    b in R^d
    c in R^d
    f(S) = set_diag(S, softplus(matrix_diag_part(S)) + 1e-5)
  ```

  Observe that `f` makes the diagonal of the triangular-lower scale matrix be
  positive and no smaller than `1e-5`.

  #### Examples

  ```python
  # This example fits a multilinear regression loss.
  import tensorflow as tf
  import tensorflow_probability as tfp

  # Create fictitious training data.
  dtype = np.float32
  n = 3000    # number of samples
  x_size = 4  # size of single x
  y_size = 2  # size of single y
  def make_training_data():
    np.random.seed(142)
    x = np.random.randn(n, x_size).astype(dtype)
    w = np.random.randn(x_size, y_size).astype(dtype)
    b = np.random.randn(1, y_size).astype(dtype)
    true_mean = np.tensordot(x, w, axes=[[-1], [0]]) + b
    noise = np.random.randn(n, y_size).astype(dtype)
    y = true_mean + noise
    return y, x
  y, x = make_training_data()

  # Build TF graph for fitting MVNTriL maximum likelihood estimator.
  mvn = tfp.trainable_distributions.multivariate_normal_tril(x, dims=y_size)
  loss = -tf.reduce_mean(mvn.log_prob(y))
  train_op = tf.train.AdamOptimizer(learning_rate=2.**-3).minimize(loss)
  mse = tf.reduce_mean(tf.squared_difference(y, mvn.mean()))
  init_op = tf.global_variables_initializer()

  # Run graph 1000 times.
  num_steps = 1000
  loss_ = np.zeros(num_steps)   # Style: `_` to indicate sess.run result.
  mse_ = np.zeros(num_steps)
  with tf.Session() as sess:
    sess.run(init_op)
    for it in xrange(loss_.size):
      _, loss_[it], mse_[it] = sess.run([train_op, loss, mse])
      if it % 200 == 0 or it == loss_.size - 1:
        print("iteration:{}  loss:{}  mse:{}".format(it, loss_[it], mse_[it]))

  # ==> iteration:0    loss:38.2020797729  mse:4.17175960541
  #     iteration:200  loss:2.90179634094  mse:0.990987896919
  #     iteration:400  loss:2.82727336884  mse:0.990926623344
  #     iteration:600  loss:2.82726788521  mse:0.990926682949
  #     iteration:800  loss:2.82726788521  mse:0.990926682949
  #     iteration:999  loss:2.82726788521  mse:0.990926682949
  ```

  Args:
    x: `Tensor` with floating type. Must have statically defined rank and
      statically known right-most dimension.
    dims: Scalar, `int`, `Tensor` indicated the MVN event size, i.e., the
      created MVN will be distribution over length-`dims` vectors.
    layer_fn: Python `callable` which takes input `x` and `int` scalar `d` and
      returns a transformation of `x` with shape
      `tf.concat([tf.shape(x)[:-1], [d]], axis=0)`.
      Default value: `tf.layers.dense`.
    loc_fn: Python `callable` which transforms the `loc` parameter. Takes a
      (batch of) length-`dims` vectors and returns a `Tensor` of same shape and
      `dtype`.
      Default value: `lambda x: x`.
    scale_fn: Python `callable` which transforms the `scale` parameters. Takes a
      (batch of) length-`dims * (dims + 1) / 2` vectors and returns a
      lower-triangular `Tensor` of same batch shape with rightmost dimensions
      having shape `[dims, dims]`.
      Default value: `tril_with_diag_softplus_and_shift`.
    name: A `name_scope` name for operations created by this function.
      Default value: `None` (i.e., "multivariate_normal_tril").

  Returns:
    mvntril: An instance of `tfd.MultivariateNormalTriL`.
  """
  with tf.compat.v1.name_scope(name, 'multivariate_normal_tril', [x, dims]):
    x = tf.convert_to_tensor(value=x, name='x')
    x = layer_fn(x, dims + dims * (dims + 1) // 2)
    return tfd.MultivariateNormalTriL(
        loc=loc_fn(x[..., :dims]),
        scale_tril=scale_fn(x[..., dims:]))

def unique(transactions):
    """ Remove any duplicate entries. """
    seen = set()
    # TODO: Handle comments
    return [x for x in transactions if not (x in seen or seen.add(x))]

def from_pb(cls, pb):
        """Instantiate the object from a protocol buffer.

        Args:
            pb (protobuf)

        Save a reference to the protocol buffer on the object.
        """
        obj = cls._from_pb(pb)
        obj._pb = pb
        return obj

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

def _parse_array(self, tensor_proto):
        """Grab data in TensorProto and convert to numpy array."""
        try:
            from onnx.numpy_helper import to_array
        except ImportError as e:
            raise ImportError("Unable to import onnx which is required {}".format(e))
        np_array = to_array(tensor_proto).reshape(tuple(tensor_proto.dims))
        return mx.nd.array(np_array)

def union_overlapping(intervals):
    """Union any overlapping intervals in the given set."""
    disjoint_intervals = []

    for interval in intervals:
        if disjoint_intervals and disjoint_intervals[-1].overlaps(interval):
            disjoint_intervals[-1] = disjoint_intervals[-1].union(interval)
        else:
            disjoint_intervals.append(interval)

    return disjoint_intervals

def _get_non_empty_list(cls, iter):
        """Return a list of the input, excluding all ``None`` values."""
        res = []
        for value in iter:
            if hasattr(value, 'items'):
                value = cls._get_non_empty_dict(value) or None
            if value is not None:
                res.append(value)
        return res

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

def auto_up(self, count=1, go_to_start_of_line_if_history_changes=False):
        """
        If we're not on the first line (of a multiline input) go a line up,
        otherwise go back in history. (If nothing is selected.)
        """
        if self.complete_state:
            self.complete_previous(count=count)
        elif self.document.cursor_position_row > 0:
            self.cursor_up(count=count)
        elif not self.selection_state:
            self.history_backward(count=count)

            # Go to the start of the line?
            if go_to_start_of_line_if_history_changes:
                self.cursor_position += self.document.get_start_of_line_position()

def terminate(self):
        """Terminate all workers and threads."""
        for t in self._threads:
            t.quit()
        self._thread = []
        self._workers = []

def to_jupyter(graph: BELGraph, chart: Optional[str] = None) -> Javascript:
    """Render the graph as JavaScript in a Jupyter Notebook."""
    with open(os.path.join(HERE, 'render_with_javascript.js'), 'rt') as f:
        js_template = Template(f.read())

    return Javascript(js_template.render(**_get_context(graph, chart=chart)))

def average(iterator):
    """Iterative mean."""
    count = 0
    total = 0
    for num in iterator:
        count += 1
        total += num
    return float(total)/count

def _uptime_syllable():
    """Returns uptime in seconds or None, on Syllable."""
    global __boottime
    try:
        __boottime = os.stat('/dev/pty/mst/pty0').st_mtime
        return time.time() - __boottime
    except (NameError, OSError):
        return None

def columns_equal(a: Column, b: Column) -> bool:
    """
    Are two SQLAlchemy columns are equal? Checks based on:

    - column ``name``
    - column ``type`` (see :func:`column_types_equal`)
    - ``nullable``
    """
    return (
        a.name == b.name and
        column_types_equal(a.type, b.type) and
        a.nullable == b.nullable
    )

def json_dumps(self, obj):
        """Serializer for consistency"""
        return json.dumps(obj, sort_keys=True, indent=4, separators=(',', ': '))

def download_file(save_path, file_url):
    """ Download file from http url link """

    r = requests.get(file_url)  # create HTTP response object

    with open(save_path, 'wb') as f:
        f.write(r.content)

    return save_path

def update(self, params):
        """Update the dev_info data from a dictionary.

        Only updates if it already exists in the device.
        """
        dev_info = self.json_state.get('deviceInfo')
        dev_info.update({k: params[k] for k in params if dev_info.get(k)})

def settimeout(self, timeout):
        """
        Set the timeout to the websocket.

        timeout: timeout time(second).
        """
        self.sock_opt.timeout = timeout
        if self.sock:
            self.sock.settimeout(timeout)

def add_mark_at(string, index, mark):
    """
    Add mark to the index-th character of the given string. Return the new string after applying change.
    Notice: index > 0
    """
    if index == -1:
        return string
    # Python can handle the case which index is out of range of given string
    return string[:index] + add_mark_char(string[index], mark) + string[index+1:]

def round_to_int(number, precision):
    """Round a number to a precision"""
    precision = int(precision)
    rounded = (int(number) + precision / 2) // precision * precision
    return rounded

def is_blankspace(self, char):
        """
        Test if a character is a blankspace.

        Parameters
        ----------
        char : str
            The character to test.

        Returns
        -------
        ret : bool
            True if character is a blankspace, False otherwise.

        """
        if len(char) > 1:
            raise TypeError("Expected a char.")
        if char in self.blankspaces:
            return True
        return False

def re_raise(self):
        """ Raise this exception with the original traceback """
        if self.exc_info is not None:
            six.reraise(type(self), self, self.exc_info[2])
        else:
            raise self

def highpass(cutoff):
  """
  This strategy uses an exponential approximation for cut-off frequency
  calculation, found by matching the one-pole Laplace lowpass filter
  and mirroring the resulting filter to get a highpass.
  """
  R = thub(exp(cutoff - pi), 2)
  return (1 - R) / (1 + R * z ** -1)

def is_relative_url(url):
    """ simple method to determine if a url is relative or absolute """
    if url.startswith("#"):
        return None
    if url.find("://") > 0 or url.startswith("//"):
        # either 'http(s)://...' or '//cdn...' and therefore absolute
        return False
    return True

def optional(self, value = None):
		"""Optional

		Getter/Setter method for optional flag

		Args:
			value (bool): If set, the method is a setter

		Returns:
			bool | None
		"""

		# If there's no value, this is a getter
		if value is None:
			return this._optional

		# Else, set the flag
		else:
			this._optional = value and True or False

def getTuple(self):
        """ Returns the shape of the region as (x, y, w, h) """
        return (self.x, self.y, self.w, self.h)

def _listify(collection):
        """This is a workaround where Collections are no longer iterable
        when using JPype."""
        new_list = []
        for index in range(len(collection)):
            new_list.append(collection[index])
        return new_list

def Flush(self):
    """Flush all items from cache."""
    while self._age:
      node = self._age.PopLeft()
      self.KillObject(node.data)

    self._hash = dict()

def is_timestamp(obj):
    """
    Yaml either have automatically converted it to a datetime object
    or it is a string that will be validated later.
    """
    return isinstance(obj, datetime.datetime) or is_string(obj) or is_int(obj) or is_float(obj)

def has_jongsung(letter):
    """Check whether this letter contains Jongsung"""
    if len(letter) != 1:
        raise Exception('The target string must be one letter.')
    if not is_hangul(letter):
        raise NotHangulException('The target string must be Hangul')

    code = lt.hangul_index(letter)
    return code % NUM_JONG > 0

def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.

    Checks whether all objects in arrays have the same shape or length.

    Parameters
    ----------
    arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """

    uniques = np.unique([_num_samples(X) for X in arrays if X is not None])
    if len(uniques) > 1:
        raise ValueError("Found arrays with inconsistent numbers of samples: %s"
                         % str(uniques))

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

def clean_tmpdir(path):
    """Invoked atexit, this removes our tmpdir"""
    if os.path.exists(path) and \
       os.path.isdir(path):
        rmtree(path)

def _tool_to_dict(tool):
    """Parse a tool definition into a cwl2wdl style dictionary.
    """
    out = {"name": _id_to_name(tool.tool["id"]),
           "baseCommand": " ".join(tool.tool["baseCommand"]),
           "arguments": [],
           "inputs": [_input_to_dict(i) for i in tool.tool["inputs"]],
           "outputs": [_output_to_dict(o) for o in tool.tool["outputs"]],
           "requirements": _requirements_to_dict(tool.requirements + tool.hints),
           "stdin": None, "stdout": None}
    return out

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

def reverse_mapping(mapping):
	"""
	For every key, value pair, return the mapping for the
	equivalent value, key pair

	>>> reverse_mapping({'a': 'b'}) == {'b': 'a'}
	True
	"""
	keys, values = zip(*mapping.items())
	return dict(zip(values, keys))

def get_coordinates_by_full_name(self, name):
        """Retrieves a person's coordinates by full name"""
        person = self.get_person_by_full_name(name)
        if not person:
            return '', ''
        return person.latitude, person.longitude

def __init__(self):
        """Initialize the state of the object"""
        self.state = self.STATE_INITIALIZING
        self.state_start = time.time()

def color_to_hex(color):
    """Convert matplotlib color code to hex color code"""
    if color is None or colorConverter.to_rgba(color)[3] == 0:
        return 'none'
    else:
        rgb = colorConverter.to_rgb(color)
        return '#{0:02X}{1:02X}{2:02X}'.format(*(int(255 * c) for c in rgb))

def ReadManyFromPath(filepath):
  """Reads a Python object stored in a specified YAML file.

  Args:
    filepath: A filepath to the YAML file.

  Returns:
    A Python data structure corresponding to the YAML in the given file.
  """
  with io.open(filepath, mode="r", encoding="utf-8") as filedesc:
    return ReadManyFromFile(filedesc)

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

def get_unique_links(self):
        """ Get all unique links in the html of the page source.
            Page links include those obtained from:
            "a"->"href", "img"->"src", "link"->"href", and "script"->"src". """
        page_url = self.get_current_url()
        soup = self.get_beautiful_soup(self.get_page_source())
        links = page_utils._get_unique_links(page_url, soup)
        return links

def cli(ctx, project_dir):
    """Clean the previous generated files."""
    exit_code = SCons(project_dir).clean()
    ctx.exit(exit_code)

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

def to_python(self, value):
        """
        Convert a string from a form into an Enum value.
        """
        if value is None:
            return value
        if isinstance(value, self.enum):
            return value
        return self.enum[value]

def Timestamp(year, month, day, hour, minute, second):
    """Constructs an object holding a datetime/timestamp value."""
    return datetime.datetime(year, month, day, hour, minute, second)

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

def init_mq(self):
        """Init connection and consumer with openstack mq."""
        mq = self.init_connection()
        self.init_consumer(mq)
        return mq.connection

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

def _digits(minval, maxval):
    """Digits needed to comforatbly display values in [minval, maxval]"""
    if minval == maxval:
        return 3
    else:
        return min(10, max(2, int(1 + abs(np.log10(maxval - minval)))))

def value_for_key(membersuite_object_data, key):
    """Return the value for `key` of membersuite_object_data.
    """
    key_value_dicts = {
        d['Key']: d['Value'] for d
        in membersuite_object_data["Fields"]["KeyValueOfstringanyType"]}
    return key_value_dicts[key]

def __ror__(self, other):
		"""The main machinery of the Pipe, calling the chosen callable with the recorded arguments."""
		
		return self.callable(*(self.args + (other, )), **self.kwargs)

def matrix_at_check(self, original, loc, tokens):
        """Check for Python 3.5 matrix multiplication."""
        return self.check_py("35", "matrix multiplication", original, loc, tokens)

def truncate_string(value, max_width=None):
    """Truncate string values."""
    if isinstance(value, text_type) and max_width is not None and len(value) > max_width:
        return value[:max_width]
    return value

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

def Unlock(fd, path):
  """Release the lock on the file.

  Args:
    fd: int, the file descriptor of the file to unlock.
    path: string, the name of the file to lock.

  Raises:
    IOError, raised from flock while attempting to release a file lock.
  """
  try:
    fcntl.flock(fd, fcntl.LOCK_UN | fcntl.LOCK_NB)
  except IOError as e:
    if e.errno == errno.EWOULDBLOCK:
      raise IOError('Exception unlocking %s. Locked by another process.' % path)
    else:
      raise IOError('Exception unlocking %s. %s.' % (path, str(e)))

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

def _lower(string):
    """Custom lower string function.
    Examples:
        FooBar -> foo_bar
    """
    if not string:
        return ""

    new_string = [string[0].lower()]
    for char in string[1:]:
        if char.isupper():
            new_string.append("_")
        new_string.append(char.lower())

    return "".join(new_string)

def start(self):
        """
        Starts the loop. Calling a running loop is an error.
        """
        assert not self.has_started(), "called start() on an active GeventLoop"
        self._stop_event = Event()
        # note that we don't use safe_greenlets.spawn because we take care of it in _loop by ourselves
        self._greenlet = gevent.spawn(self._loop)

def mask_nonfinite(self):
        """Extend the mask with the image elements where the intensity is NaN."""
        self.mask = np.logical_and(self.mask, (np.isfinite(self.intensity)))

def action(self):
        """
        This class overrides this method
        """
        self.return_value = self.function(*self.args, **self.kwargs)

def get_soup(page=''):
    """
    Returns a bs4 object of the page requested
    """
    content = requests.get('%s/%s' % (BASE_URL, page)).text
    return BeautifulSoup(content)

def getcolslice(self, blc, trc, inc=[], startrow=0, nrow=-1, rowincr=1):
        """Get a slice from a table column holding arrays.
        (see :func:`table.getcolslice`)"""
        return self._table.getcolslice(self._column, blc, trc, inc, startrow, nrow, rowincr)

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

def _check_limit(self):
        """Intenal method: check if current cache size exceeds maximum cache
           size and pop the oldest item in this case"""

        # First compress
        self._compress()

        # Then check the max size
        if len(self._store) >= self._max_size:
            self._store.popitem(last=False)

def on_windows ():
    """ Returns true if running on windows, whether in cygwin or not.
    """
    if bjam.variable("NT"):
        return True

    elif bjam.variable("UNIX"):

        uname = bjam.variable("JAMUNAME")
        if uname and uname[0].startswith("CYGWIN"):
            return True

    return False

def rel_path(filename):
    """
    Function that gets relative path to the filename
    """
    return os.path.join(os.getcwd(), os.path.dirname(__file__), filename)

def strip_columns(tab):
    """Strip whitespace from string columns."""
    for colname in tab.colnames:
        if tab[colname].dtype.kind in ['S', 'U']:
            tab[colname] = np.core.defchararray.strip(tab[colname])

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

def has_common(self, other):
        """Return set of common words between two word sets."""
        if not isinstance(other, WordSet):
            raise ValueError('Can compare only WordSets')
        return self.term_set & other.term_set

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

def help_for_command(command):
    """Get the help text (signature + docstring) for a command (function)."""
    help_text = pydoc.text.document(command)
    # remove backspaces
    return re.subn('.\\x08', '', help_text)[0]

def pop_all(self):
        """
        NON-BLOCKING POP ALL IN QUEUE, IF ANY
        """
        with self.lock:
            output = list(self.queue)
            self.queue.clear()

        return output

def is_iter_non_string(obj):
    """test if object is a list or tuple"""
    if isinstance(obj, list) or isinstance(obj, tuple):
        return True
    return False

def append(self, item):
        """ append item and print it to stdout """
        print(item)
        super(MyList, self).append(item)

def __init__(self, form_post_data=None, *args, **kwargs):
        """
        Overriding init so we can set the post vars like a normal form and generate
        the form the same way Django does.
        """
        kwargs.update({'form_post_data': form_post_data})
        super(MongoModelForm, self).__init__(*args, **kwargs)

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

def coerce(self, value):
        """Convert from whatever is given to a list of scalars for the lookup_field."""
        if isinstance(value, dict):
            value = [value]
        if not isiterable_notstring(value):
            value = [value]
        return [coerce_single_instance(self.lookup_field, v) for v in value]

def send_request(self, *args, **kwargs):
        """Wrapper for session.request
        Handle connection reset error even from pyopenssl
        """
        try:
            return self.session.request(*args, **kwargs)
        except ConnectionError:
            self.session.close()
            return self.session.request(*args, **kwargs)

def normal_log_q(self,z):
        """
        The unnormalized log posterior components for mean-field normal family (the quantity we want to approximate)
        RAO-BLACKWELLIZED!
        """             
        means, scale = self.get_means_and_scales()
        return ss.norm.logpdf(z,loc=means,scale=scale)

def update(self, iterable):
        """
        Return a new PSet with elements in iterable added

        >>> s1 = s(1, 2)
        >>> s1.update([3, 4, 4])
        pset([1, 2, 3, 4])
        """
        e = self.evolver()
        for element in iterable:
            e.add(element)

        return e.persistent()

def _from_list_dict(cls, list_dic):
        """Takes a list of dict like objects and uses `champ_id` field as Id"""
        return cls({_convert_id(dic[cls.CHAMP_ID]): dict(dic) for dic in list_dic})

def delete(args):
    """
    Delete a river by name
    """
    m = RiverManager(args.hosts)
    m.delete(args.name)

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

def ibatch(iterable, size):
    """Yield a series of batches from iterable, each size elements long."""
    source = iter(iterable)
    while True:
        batch = itertools.islice(source, size)
        yield itertools.chain([next(batch)], batch)

def path_distance(points):
    """
    Compute the path distance from given set of points
    """
    vecs = np.diff(points, axis=0)[:, :3]
    d2 = [np.dot(p, p) for p in vecs]
    return np.sum(np.sqrt(d2))

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

def random_numbers(n):
    """
    Generate a random string from 0-9
    :param n: length of the string
    :return: the random string
    """
    return ''.join(random.SystemRandom().choice(string.digits) for _ in range(n))

def _updateTabStopWidth(self):
        """Update tabstop width after font or indentation changed
        """
        self.setTabStopWidth(self.fontMetrics().width(' ' * self._indenter.width))

def get_codes(s: Union[str, 'ChainedBase']) -> List[str]:
    """ Grab all escape codes from a string.
        Returns a list of all escape codes.
    """
    return codegrabpat.findall(str(s))

def get_table_columns(dbconn, tablename):
    """
    Return a list of tuples specifying the column name and type
    """
    cur = dbconn.cursor()
    cur.execute("PRAGMA table_info('%s');" % tablename)
    info = cur.fetchall()
    cols = [(i[1], i[2]) for i in info]
    return cols

def remove_file_from_s3(awsclient, bucket, key):
    """Remove a file from an AWS S3 bucket.

    :param awsclient:
    :param bucket:
    :param key:
    :return:
    """
    client_s3 = awsclient.get_client('s3')
    response = client_s3.delete_object(Bucket=bucket, Key=key)

def is_changed():
    """ Checks if current project has any noncommited changes. """
    executed, changed_lines = execute_git('status --porcelain', output=False)
    merge_not_finished = mod_path.exists('.git/MERGE_HEAD')
    return changed_lines.strip() or merge_not_finished

def delete(self, endpoint: str, **kwargs) -> dict:
        """HTTP DELETE operation to API endpoint."""

        return self._request('DELETE', endpoint, **kwargs)

def make_table_map(table, headers):
    """Create a function to map from rows with the structure of the headers to the structure of the table."""

    header_parts = {}
    for i, h in enumerate(headers):
        header_parts[h] = 'row[{}]'.format(i)

    body_code = 'lambda row: [{}]'.format(','.join(header_parts.get(c.name, 'None') for c in table.columns))
    header_code = 'lambda row: [{}]'.format(
        ','.join(header_parts.get(c.name, "'{}'".format(c.name)) for c in table.columns))

    return eval(header_code), eval(body_code)

def record_diff(old, new):
    """Return a JSON-compatible structure capable turn the `new` record back
    into the `old` record. The parameters must be structures compatible with
    json.dumps *or* strings compatible with json.loads. Note that by design,
    `old == record_patch(new, record_diff(old, new))`"""
    old, new = _norm_json_params(old, new)
    return json_delta.diff(new, old, verbose=False)

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

def delete_lines(self):
        """
        Deletes the document lines under cursor.

        :return: Method success.
        :rtype: bool
        """

        cursor = self.textCursor()
        self.__select_text_under_cursor_blocks(cursor)
        cursor.removeSelectedText()
        cursor.deleteChar()
        return True

def _check_samples_nodups(fnames):
    """Ensure a set of input VCFs do not have duplicate samples.
    """
    counts = defaultdict(int)
    for f in fnames:
        for s in get_samples(f):
            counts[s] += 1
    duplicates = [s for s, c in counts.items() if c > 1]
    if duplicates:
        raise ValueError("Duplicate samples found in inputs %s: %s" % (duplicates, fnames))

def is_subdir(a, b):
    """
    Return true if a is a subdirectory of b
    """
    a, b = map(os.path.abspath, [a, b])

    return os.path.commonpath([a, b]) == b

def date_to_datetime(d):
    """
    >>> date_to_datetime(date(2000, 1, 2))
    datetime.datetime(2000, 1, 2, 0, 0)
    >>> date_to_datetime(datetime(2000, 1, 2, 3, 4, 5))
    datetime.datetime(2000, 1, 2, 3, 4, 5)
    """
    if not isinstance(d, datetime):
        d = datetime.combine(d, datetime.min.time())
    return d

def read_key(suppress=False):
    """
    Blocks until a keyboard event happens, then returns that event's name or,
    if missing, its scan code.
    """
    event = read_event(suppress)
    return event.name or event.scan_code

def release_lock():
    """Release lock on compilation directory."""
    get_lock.n_lock -= 1
    assert get_lock.n_lock >= 0
    # Only really release lock once all lock requests have ended.
    if get_lock.lock_is_enabled and get_lock.n_lock == 0:
        get_lock.start_time = None
        get_lock.unlocker.unlock()

def indent(self, message):
        """
        Sets the indent for standardized output
        :param message: (str)
        :return: (str)
        """
        indent = self.indent_char * self.indent_size
        return indent + message

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

def dump_parent(self, obj):
        """Dump the parent of a PID."""
        if not self._is_parent(obj):
            return self._dump_relative(obj.pid)
        return None

def to_snake_case(name):
    """ Given a name in camelCase return in snake_case """
    s1 = FIRST_CAP_REGEX.sub(r'\1_\2', name)
    return ALL_CAP_REGEX.sub(r'\1_\2', s1).lower()

def pprint(o, stream=None, indent=1, width=80, depth=None):
    """Pretty-print a Python o to a stream [default is sys.stdout]."""
    printer = PrettyPrinter(
        stream=stream, indent=indent, width=width, depth=depth)
    printer.pprint(o)

def test_value(self, value):
        """Test if value is an instance of int."""
        if not isinstance(value, int):
            raise ValueError('expected int value: ' + str(type(value)))

def parse_cookies(self, req, name, field):
        """Pull the value from the cookiejar."""
        return core.get_value(req.COOKIES, name, field)

def _remove_nonascii(self, df):
    """Make copy and remove non-ascii characters from it."""

    df_copy = df.copy(deep=True)
    for col in df_copy.columns:
      if (df_copy[col].dtype == np.dtype('O')):
        df_copy[col] = df[col].apply(
          lambda x: re.sub(r'[^\x00-\x7f]', r'', x) if isinstance(x, six.string_types) else x)

    return df_copy

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

def flush(self):
        """ Force commit changes to the file and stdout """
        if not self.nostdout:
            self.stdout.flush()
        if self.file is not None:
            self.file.flush()

def cluster_kmeans(data, n_clusters, **kwargs):
    """
    Identify clusters using K - Means algorithm.

    Parameters
    ----------
    data : array_like
        array of size [n_samples, n_features].
    n_clusters : int
        The number of clusters expected in the data.

    Returns
    -------
    dict
        boolean array for each identified cluster.
    """
    km = cl.KMeans(n_clusters, **kwargs)
    kmf = km.fit(data)

    labels = kmf.labels_

    return labels, [np.nan]

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

def fail_print(error):
    """Print an error in red text.
    Parameters
        error (HTTPError)
            Error object to print.
    """
    print(COLORS.fail, error.message, COLORS.end)
    print(COLORS.fail, error.errors, COLORS.end)

def step_next_line(self):
        """Sets cursor as beginning of next line."""
        self._eol.append(self.position)
        self._lineno += 1
        self._col_offset = 0

def pdf(x, mu, std):
    """Probability density function (normal distribution)"""
    return (1.0 / (std * sqrt(2 * pi))) * np.exp(-(x - mu) ** 2 / (2 * std ** 2))

def write_to_file(file_path, contents, encoding="utf-8"):
    """
    Write helper method

    :type file_path: str|unicode
    :type contents: str|unicode
    :type encoding: str|unicode
    """
    with codecs.open(file_path, "w", encoding) as f:
        f.write(contents)

def print_yaml(o):
    """Pretty print an object as YAML."""
    print(yaml.dump(o, default_flow_style=False, indent=4, encoding='utf-8'))

def startEdit( self ):
        """
        Rebuilds the pathing based on the parts.
        """
        self._originalText = self.text()
        self.scrollWidget().hide()
        self.setFocus()
        self.selectAll()

def run(*tasks: Awaitable, loop: asyncio.AbstractEventLoop=asyncio.get_event_loop()):
    """Helper to run tasks in the event loop

    :param tasks: Tasks to run in the event loop.
    :param loop: The event loop.
    """
    futures = [asyncio.ensure_future(task, loop=loop) for task in tasks]
    return loop.run_until_complete(asyncio.gather(*futures))

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

def findfirst(f, coll):
    """Return first occurrence matching f, otherwise None"""
    result = list(dropwhile(f, coll))
    return result[0] if result else None

def yview(self, *args):
        """Update inplace widgets position when doing vertical scroll"""
        self.after_idle(self.__updateWnds)
        ttk.Treeview.yview(self, *args)

def remove_na_arraylike(arr):
    """
    Return array-like containing only true/non-NaN values, possibly empty.
    """
    if is_extension_array_dtype(arr):
        return arr[notna(arr)]
    else:
        return arr[notna(lib.values_from_object(arr))]

def _write_json(file, contents):
    """Write a dict to a JSON file."""
    with open(file, 'w') as f:
        return json.dump(contents, f, indent=2, sort_keys=True)

def _generate_plane(normal, origin):
    """ Returns a vtk.vtkPlane """
    plane = vtk.vtkPlane()
    plane.SetNormal(normal[0], normal[1], normal[2])
    plane.SetOrigin(origin[0], origin[1], origin[2])
    return plane

def cross_v2(vec1, vec2):
    """Return the crossproduct of the two vectors as a Vec2.
    Cross product doesn't really make sense in 2D, but return the Z component
    of the 3d result.
    """

    return vec1.y * vec2.x - vec1.x * vec2.y

def us2mc(string):
    """Transform an underscore_case string to a mixedCase string"""
    return re.sub(r'_([a-z])', lambda m: (m.group(1).upper()), string)

def lsem (inlist):
    """
Returns the estimated standard error of the mean (sx-bar) of the
values in the passed list.  sem = stdev / sqrt(n)

Usage:   lsem(inlist)
"""
    sd = stdev(inlist)
    n = len(inlist)
    return sd/math.sqrt(n)

def fetchvalue(self, sql: str, *args) -> Optional[Any]:
        """Executes SQL; returns the first value of the first row, or None."""
        row = self.fetchone(sql, *args)
        if row is None:
            return None
        return row[0]

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

def unique_(self, col):
        """
        Returns unique values in a column
        """
        try:
            df = self.df.drop_duplicates(subset=[col], inplace=False)
            return list(df[col])
        except Exception as e:
            self.err(e, "Can not select unique data")

def count(self, elem):
        """
        Return the number of elements equal to elem present in the queue

        >>> pdeque([1, 2, 1]).count(1)
        2
        """
        return self._left_list.count(elem) + self._right_list.count(elem)

def arcball_map_to_sphere(point, center, radius):
    """Return unit sphere coordinates from window coordinates."""
    v0 = (point[0] - center[0]) / radius
    v1 = (center[1] - point[1]) / radius
    n = v0*v0 + v1*v1
    if n > 1.0:
        # position outside of sphere
        n = math.sqrt(n)
        return numpy.array([v0/n, v1/n, 0.0])
    else:
        return numpy.array([v0, v1, math.sqrt(1.0 - n)])

def xyz2lonlat(x, y, z):
    """Convert cartesian to lon lat."""
    lon = xu.rad2deg(xu.arctan2(y, x))
    lat = xu.rad2deg(xu.arctan2(z, xu.sqrt(x**2 + y**2)))
    return lon, lat

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

def ss_tot(self):
        """Total sum of squares."""
        return np.sum(np.square(self.y - self.ybar), axis=0)

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

def de_duplicate(items):
    """Remove any duplicate item, preserving order

    >>> de_duplicate([1, 2, 1, 2])
    [1, 2]
    """
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return result

def remove_bad(string):
    """
    remove problem characters from string
    """
    remove = [':', ',', '(', ')', ' ', '|', ';', '\'']
    for c in remove:
        string = string.replace(c, '_')
    return string

def select_default(self):
        """
        Resets the combo box to the original "selected" value from the
        constructor (or the first value if no selected value was specified).
        """
        if self._default is None:
            if not self._set_option_by_index(0):
                utils.error_format(self.description + "\n" +
                "Unable to select default option as the Combo is empty")

        else:
            if not self._set_option(self._default):
                utils.error_format( self.description + "\n" +
                "Unable to select default option as it doesnt exist in the Combo")

def type(self):
        """Returns type of the data for the given FeatureType."""
        if self is FeatureType.TIMESTAMP:
            return list
        if self is FeatureType.BBOX:
            return BBox
        return dict

def is_exe(fpath):
    """
    Path references an executable file.
    """
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

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

def impose_legend_limit(limit=30, axes="gca", **kwargs):
    """
    This will erase all but, say, 30 of the legend entries and remake the legend.
    You'll probably have to move it back into your favorite position at this point.
    """
    if axes=="gca": axes = _pylab.gca()

    # make these axes current
    _pylab.axes(axes)

    # loop over all the lines_pylab.
    for n in range(0,len(axes.lines)):
        if n >  limit-1 and not n==len(axes.lines)-1: axes.lines[n].set_label("_nolegend_")
        if n == limit-1 and not n==len(axes.lines)-1: axes.lines[n].set_label("...")

    _pylab.legend(**kwargs)

def main(output=None, error=None, verbose=False):
    """ The main (cli) interface for the pylint runner. """
    runner = Runner(args=["--verbose"] if verbose is not False else None)
    runner.run(output, error)

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

def text_coords(string, position):
    r"""
    Transform a simple index into a human-readable position in a string.

    This function accepts a string and an index, and will return a triple of
    `(lineno, columnno, line)` representing the position through the text. It's
    useful for displaying a string index in a human-readable way::

        >>> s = "abcdef\nghijkl\nmnopqr\nstuvwx\nyz"
        >>> text_coords(s, 0)
        (0, 0, 'abcdef')
        >>> text_coords(s, 4)
        (0, 4, 'abcdef')
        >>> text_coords(s, 6)
        (0, 6, 'abcdef')
        >>> text_coords(s, 7)
        (1, 0, 'ghijkl')
        >>> text_coords(s, 11)
        (1, 4, 'ghijkl')
        >>> text_coords(s, 15)
        (2, 1, 'mnopqr')
    """
    line_start = string.rfind('\n', 0, position) + 1
    line_end = string.find('\n', position)
    lineno = string.count('\n', 0, position)
    columnno = position - line_start
    line = string[line_start:line_end]
    return (lineno, columnno, line)

def Max(a, axis, keep_dims):
    """
    Max reduction op.
    """
    return np.amax(a, axis=axis if not isinstance(axis, np.ndarray) else tuple(axis),
                   keepdims=keep_dims),

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

def filehash(path):
    """Make an MD5 hash of a file, ignoring any differences in line
    ending characters."""
    with open(path, "rU") as f:
        return md5(py3compat.str_to_bytes(f.read())).hexdigest()

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

def is_defined(self, objtxt, force_import=False):
        """Return True if object is defined"""
        return self.interpreter.is_defined(objtxt, force_import)

def synth_hangul(string):
    """Convert jamo characters in a string into hcj as much as possible."""
    raise NotImplementedError
    return ''.join([''.join(''.join(jamo_to_hcj(_)) for _ in string)])

def find_column(token):
    """ Compute column:
            input is the input text string
            token is a token instance
    """
    i = token.lexpos
    input = token.lexer.lexdata

    while i > 0:
        if input[i - 1] == '\n':
            break
        i -= 1

    column = token.lexpos - i + 1

    return column

def fmt_subst(regex, subst):
    """Replace regex with string."""
    return lambda text: re.sub(regex, subst, text) if text else text

def clear():
    """Clears the console."""
    if sys.platform.startswith("win"):
        call("cls", shell=True)
    else:
        call("clear", shell=True)

def serialize(self, value, **kwargs):
        """Serialize every item of the list."""
        return [self.item_type.serialize(val, **kwargs) for val in value]

def good(txt):
    """Print, emphasized 'good', the given 'txt' message"""

    print("%s# %s%s%s" % (PR_GOOD_CC, get_time_stamp(), txt, PR_NC))
    sys.stdout.flush()

def get_table_list(dbconn):
    """
    Get a list of tables that exist in dbconn
    :param dbconn: database connection
    :return: List of table names
    """
    cur = dbconn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    try:
        return [item[0] for item in cur.fetchall()]
    except IndexError:
        return get_table_list(dbconn)

def pprint(self, stream=None, indent=1, width=80, depth=None):
    """
    Pretty print the underlying literal Python object
    """
    pp.pprint(to_literal(self), stream, indent, width, depth)

def mark(self, lineno, count=1):
        """Mark a given source line as executed count times.

        Multiple calls to mark for the same lineno add up.
        """
        self.sourcelines[lineno] = self.sourcelines.get(lineno, 0) + count

def _sort_tensor(tensor):
  """Use `top_k` to sort a `Tensor` along the last dimension."""
  sorted_, _ = tf.nn.top_k(tensor, k=tf.shape(input=tensor)[-1])
  sorted_.set_shape(tensor.shape)
  return sorted_

def calculate_bounding_box(data):
    """
    Returns a 2 x m array indicating the min and max along each
    dimension.
    """
    mins = data.min(0)
    maxes = data.max(0)
    return mins, maxes

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

def on_close(self, ws):
        """ Called when websocket connection is closed
        """
        log.debug("Closing WebSocket connection with {}".format(self.url))
        if self.keepalive and self.keepalive.is_alive():
            self.keepalive.do_run = False
            self.keepalive.join()

def blk_coverage_1d(blk, size):
    """Return the part of a 1d array covered by a block.

    :param blk: size of the 1d block
    :param size: size of the 1d a image
    :return: a tuple of size covered and remaining size

    Example:

        >>> blk_coverage_1d(7, 100)
        (98, 2)

    """
    rem = size % blk
    maxpix = size - rem
    return maxpix, rem

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

def add_widgets(self, *widgets_or_spacings):
        """Add widgets/spacing to dialog vertical layout"""
        layout = self.layout()
        for widget_or_spacing in widgets_or_spacings:
            if isinstance(widget_or_spacing, int):
                layout.addSpacing(widget_or_spacing)
            else:
                layout.addWidget(widget_or_spacing)

def validate(self, val):
        """
        Validates that the val is in the list of values for this Enum.

        Returns two element tuple: (bool, string)

        - `bool` - True if valid, False if not
        - `string` - Description of validation error, or None if valid

        :Parameters:
          val
            Value to validate.  Should be a string.
        """
        if val in self.values:
            return True, None
        else:
            return False, "'%s' is not in enum: %s" % (val, str(self.values))

def __pop_top_frame(self):
        """Pops the top frame off the frame stack."""
        popped = self.__stack.pop()
        if self.__stack:
            self.__stack[-1].process_subframe(popped)

def items(self, section_name):
        """:return: list((option, value), ...) pairs of all items in the given section"""
        return [(k, v) for k, v in super(GitConfigParser, self).items(section_name) if k != '__name__']

def str_to_time(time_str: str) -> datetime.datetime:
    """
    Convert human readable string to datetime.datetime.
    """
    pieces: Any = [int(piece) for piece in time_str.split('-')]
    return datetime.datetime(*pieces)

def __contains__(self, key):
        """
        Invoked when determining whether a specific key is in the dictionary
        using `key in d`.

        The key is looked up case-insensitively.
        """
        k = self._real_key(key)
        return k in self._data

def unpackbools(integers, dtype='L'):
    """Yield booleans unpacking integers of dtype bit-length.

    >>> list(unpackbools([42], 'B'))
    [False, True, False, True, False, True, False, False]
    """
    atoms = ATOMS[dtype]

    for chunk in integers:
        for a in atoms:
            yield not not chunk & a

def lowercase_chars(string: any) -> str:
        """Return all (and only) the lowercase chars in the given string."""
        return ''.join([c if c.islower() else '' for c in str(string)])

def info(self, text):
		""" Ajout d'un message de log de type INFO """
		self.logger.info("{}{}".format(self.message_prefix, text))

def get_field_by_name(self, name):
        """
        the field member matching name, or None if no such field is found
        """

        for f in self.fields:
            if f.get_name() == name:
                return f
        return None

def get_active_window(self):
        """
        The current active :class:`.Window`.
        """
        app = get_app()

        try:
            return self._active_window_for_cli[app]
        except KeyError:
            self._active_window_for_cli[app] = self._last_active_window or self.windows[0]
            return self.windows[0]

def tmpfile(prefix, direc):
    """Returns the path to a newly created temporary file."""
    return tempfile.mktemp(prefix=prefix, suffix='.pdb', dir=direc)

def build_and_start(query, directory):
    """This function will create and then start a new Async task with the
    default callbacks argument defined in the decorator."""

    Async(target=grep, args=[query, directory]).start()

def eval_Rf(self, Vf):
        """Evaluate smooth term in Vf."""

        return sl.inner(self.Df, Vf, axis=self.cri.axisM) - self.Sf

def unescape_all(string):
    """Resolve all html entities to their corresponding unicode character"""
    def escape_single(matchobj):
        return _unicode_for_entity_with_name(matchobj.group(1))
    return entities.sub(escape_single, string)

def copy_no_perm(src, dst):
    """
    Copies a file from *src* to *dst* including meta data except for permission bits.
    """
    shutil.copy(src, dst)
    perm = os.stat(dst).st_mode
    shutil.copystat(src, dst)
    os.chmod(dst, perm)

def GetRootKey(self):
    """Retrieves the root key.

    Returns:
      WinRegistryKey: Windows Registry root key or None if not available.
    """
    regf_key = self._regf_file.get_root_key()
    if not regf_key:
      return None

    return REGFWinRegistryKey(regf_key, key_path=self._key_path_prefix)

def on_pause(self):
        """Sync the database with the current state of the game."""
        self.engine.commit()
        self.strings.save()
        self.funcs.save()
        self.config.write()

def is_scalar(value):
    """Test if the given value is a scalar.

    This function also works with memory mapped array values, in contrast to the numpy is_scalar method.

    Args:
        value: the value to test for being a scalar value

    Returns:
        boolean: if the given value is a scalar or not
    """
    return np.isscalar(value) or (isinstance(value, np.ndarray) and (len(np.squeeze(value).shape) == 0))

def _from_bytes(bytes, byteorder="big", signed=False):
    """This is the same functionality as ``int.from_bytes`` in python 3"""
    return int.from_bytes(bytes, byteorder=byteorder, signed=signed)

def do_exit(self, arg):
        """Exit the shell session."""

        if self.current:
            self.current.close()
        self.resource_manager.close()
        del self.resource_manager
        return True

def fromDict(cls, _dict):
        """ Builds instance from dictionary of properties. """
        obj = cls()
        obj.__dict__.update(_dict)
        return obj

def to_topojson(self):
        """Adds points and converts to topojson string."""
        topojson = self.topojson
        topojson["objects"]["points"] = {
            "type": "GeometryCollection",
            "geometries": [point.to_topojson() for point in self.points.all()],
        }
        return json.dumps(topojson)

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

def unit_vector(x):
    """Return a unit vector in the same direction as x."""
    y = np.array(x, dtype='float')
    return y/norm(y)

def empty_wav(wav_path: Union[Path, str]) -> bool:
    """Check if a wav contains data"""
    with wave.open(str(wav_path), 'rb') as wav_f:
        return wav_f.getnframes() == 0

def delimited(items, character='|'):
    """Returns a character delimited version of the provided list as a Python string"""
    return '|'.join(items) if type(items) in (list, tuple, set) else items

def str_upper(x):
    """Converts all strings in a column to uppercase.

    :returns: an expression containing the converted strings.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.


    >>> df.text.str.upper()
    Expression = str_upper(text)
    Length: 5 dtype: str (expression)
    ---------------------------------
    0    SOMETHING
    1  VERY PRETTY
    2    IS COMING
    3          OUR
    4         WAY.

    """
    sl = _to_string_sequence(x).upper()
    return column.ColumnStringArrow(sl.bytes, sl.indices, sl.length, sl.offset, string_sequence=sl)

def _disable_venv(self, env):
        """
        Disable virtualenv and venv in the environment.
        """
        venv = env.pop('VIRTUAL_ENV', None)
        if venv:
            venv_path, sep, env['PATH'] = env['PATH'].partition(os.pathsep)

def _frombuffer(ptr, frames, channels, dtype):
    """Create NumPy array from a pointer to some memory."""
    framesize = channels * dtype.itemsize
    data = np.frombuffer(ffi.buffer(ptr, frames * framesize), dtype=dtype)
    data.shape = -1, channels
    return data

def _on_select(self, *args):
        """
        Function bound to event of selection in the Combobox, calls callback if callable
        
        :param args: Tkinter event
        """
        if callable(self.__callback):
            self.__callback(self.selection)

def __add__(self, other):
        """Handle the `+` operator."""
        return self._handle_type(other)(self.value + other.value)

def min_max_normalize(img):
    """Centre and normalize a given array.

    Parameters:
    ----------
    img: np.ndarray

    """

    min_img = img.min()
    max_img = img.max()

    return (img - min_img) / (max_img - min_img)

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

def _makes_clone(_func, *args, **kw):
    """
    A decorator that returns a clone of the current object so that
    we can re-use the object for similar requests.
    """
    self = args[0]._clone()
    _func(self, *args[1:], **kw)
    return self

def camel_to_under(name):
    """
    Converts camel-case string to lowercase string separated by underscores.

    Written by epost (http://stackoverflow.com/questions/1175208).

    :param name: String to be converted
    :return: new String with camel-case converted to lowercase, underscored
    """
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

def isdir(path, **kwargs):
    """Check if *path* is a directory"""
    import os.path
    return os.path.isdir(path, **kwargs)

def scipy_sparse_to_spmatrix(A):
    """Efficient conversion from scipy sparse matrix to cvxopt sparse matrix"""
    coo = A.tocoo()
    SP = spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=A.shape)
    return SP

def money(min=0, max=10):
    """Return a str of decimal with two digits after a decimal mark."""
    value = random.choice(range(min * 100, max * 100))
    return "%1.2f" % (float(value) / 100)

def split_comma_argument(comma_sep_str):
    """Split a comma separated option into a list."""
    terms = []
    for term in comma_sep_str.split(','):
        if term:
            terms.append(term)
    return terms

def polyline(*points):
    """Converts a list of points to a Path composed of lines connecting those 
    points (i.e. a linear spline or polyline).  See also `polygon()`."""
    return Path(*[Line(points[i], points[i+1])
                  for i in range(len(points) - 1)])

def make_temp(text):
        """
        Creates a temprorary file and writes the `text` into it
        """
        import tempfile
        (handle, path) = tempfile.mkstemp(text=True)
        os.close(handle)
        afile = File(path)
        afile.write(text)
        return afile

def is_clicked(self, MouseStateType):
        """
        Did the user depress and release the button to signify a click?
        MouseStateType is the button to query. Values found under StateTypes.py
        """
        return self.previous_mouse_state.query_state(MouseStateType) and (
        not self.current_mouse_state.query_state(MouseStateType))

def percentile(sorted_list, percent, key=lambda x: x):
    """Find the percentile of a sorted list of values.

    Arguments
    ---------
    sorted_list : list
        A sorted (ascending) list of values.
    percent : float
        A float value from 0.0 to 1.0.
    key : function, optional
        An optional function to compute a value from each element of N.

    Returns
    -------
    float
        The desired percentile of the value list.

    Examples
    --------
    >>> sorted_list = [4,6,8,9,11]
    >>> percentile(sorted_list, 0.4)
    7.0
    >>> percentile(sorted_list, 0.44)
    8.0
    >>> percentile(sorted_list, 0.6)
    8.5
    >>> percentile(sorted_list, 0.99)
    11.0
    >>> percentile(sorted_list, 1)
    11.0
    >>> percentile(sorted_list, 0)
    4.0
    """
    if not sorted_list:
        return None
    if percent == 1:
        return float(sorted_list[-1])
    if percent == 0:
        return float(sorted_list[0])
    n = len(sorted_list)
    i = percent * n
    if ceil(i) == i:
        i = int(i)
        return (sorted_list[i-1] + sorted_list[i]) / 2
    return float(sorted_list[ceil(i)-1])

def _root_mean_square_error(y, y_pred, w):
    """Calculate the root mean square error."""
    return np.sqrt(np.average(((y_pred - y) ** 2), weights=w))

def get_input_nodes(G: nx.DiGraph) -> List[str]:
    """ Get all input nodes from a network. """
    return [n for n, d in G.in_degree() if d == 0]

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

def print_error(msg):
    """ Print an error message """
    if IS_POSIX:
        print(u"%s[ERRO] %s%s" % (ANSI_ERROR, msg, ANSI_END))
    else:
        print(u"[ERRO] %s" % (msg))

def value_left(self, other):
    """
    Returns the value of the other type instance to use in an
    operator method, namely when the method's instance is on the
    left side of the expression.
    """
    return other.value if isinstance(other, self.__class__) else other

def Any(a, axis, keep_dims):
    """
    Any reduction op.
    """
    return np.any(a, axis=axis if not isinstance(axis, np.ndarray) else tuple(axis),
                  keepdims=keep_dims),

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

def _unique_rows_numpy(a):
    """return unique rows"""
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def main(args=sys.argv):
    """
    main entry point for the jardiff CLI
    """

    parser = create_optparser(args[0])
    return cli(parser.parse_args(args[1:]))

def __get_registry_key(self, key):
        """ Read currency from windows registry """
        import winreg

        root = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, r'SOFTWARE\GSettings\org\gnucash\general', 0, winreg.KEY_READ)
        [pathname, regtype] = (winreg.QueryValueEx(root, key))
        winreg.CloseKey(root)
        return pathname

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

def _regex_span(_regex, _str, case_insensitive=True):
    """Return all matches in an input string.
    :rtype : regex.match.span
    :param _regex: A regular expression pattern.
    :param _str: Text on which to run the pattern.
    """
    if case_insensitive:
        flags = regex.IGNORECASE | regex.FULLCASE | regex.VERSION1
    else:
        flags = regex.VERSION1
    comp = regex.compile(_regex, flags=flags)
    matches = comp.finditer(_str)
    for match in matches:
        yield match

def intersect(self, other):
        """ Return a new :class:`DataFrame` containing rows only in
        both this frame and another frame.

        This is equivalent to `INTERSECT` in SQL.
        """
        return DataFrame(self._jdf.intersect(other._jdf), self.sql_ctx)

def setup(app):
  """
  Just connects the docstring pre_processor and should_skip functions to be
  applied on all docstrings.

  """
  app.connect('autodoc-process-docstring',
              lambda *args: pre_processor(*args, namer=audiolazy_namer))
  app.connect('autodoc-skip-member', should_skip)

def h5ToDict(h5, readH5pyDataset=True):
    """ Read a hdf5 file into a dictionary """
    h = h5py.File(h5, "r")
    ret = unwrapArray(h, recursive=True, readH5pyDataset=readH5pyDataset)
    if readH5pyDataset: h.close()
    return ret

def _windowsLdmodTargets(target, source, env, for_signature):
    """Get targets for loadable modules."""
    return _dllTargets(target, source, env, for_signature, 'LDMODULE')

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

def _file_chunks(self, data, chunk_size):
        """ Yield compressed chunks from a data array"""
        for i in xrange(0, len(data), chunk_size):
            yield self.compressor(data[i:i+chunk_size])

def mkhead(repo, path):
    """:return: New branch/head instance"""
    return git.Head(repo, git.Head.to_full_path(path))

def is_float_array(l):
    r"""Checks if l is a numpy array of floats (any dimension

    """
    if isinstance(l, np.ndarray):
        if l.dtype.kind == 'f':
            return True
    return False

def find(self, node, path):
        """Wrapper for lxml`s find."""

        return node.find(path, namespaces=self.namespaces)

def _prtfmt(self, item_id, dashes):
        """Print object information using a namedtuple and a format pattern."""
        ntprt = self.id2nt[item_id]
        dct = ntprt._asdict()
        self.prt.write('{DASHES:{N}}'.format(
            DASHES=self.fmt_dashes.format(DASHES=dashes, ID=self.nm2prtfmt['ID'].format(**dct)),
            N=self.dash_len))
        self.prt.write("{INFO}\n".format(INFO=self.nm2prtfmt['ITEM'].format(**dct)))

def get_tweepy_auth(twitter_api_key,
                    twitter_api_secret,
                    twitter_access_token,
                    twitter_access_token_secret):
    """Make a tweepy auth object"""
    auth = tweepy.OAuthHandler(twitter_api_key, twitter_api_secret)
    auth.set_access_token(twitter_access_token, twitter_access_token_secret)
    return auth

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

def round_corner(radius, fill):
    """Draw a round corner"""
    corner = Image.new('L', (radius, radius), 0)  # (0, 0, 0, 0))
    draw = ImageDraw.Draw(corner)
    draw.pieslice((0, 0, radius * 2, radius * 2), 180, 270, fill=fill)
    return corner

def contained_in(filename, directory):
    """Test if a file is located within the given directory."""
    filename = os.path.normcase(os.path.abspath(filename))
    directory = os.path.normcase(os.path.abspath(directory))
    return os.path.commonprefix([filename, directory]) == directory

def clean_text_by_sentences(text, language="english", additional_stopwords=None):
    """ Tokenizes a given text into sentences, applying filters and lemmatizing them.
    Returns a SyntacticUnit list. """
    init_textcleanner(language, additional_stopwords)
    original_sentences = split_sentences(text)
    filtered_sentences = filter_words(original_sentences)

    return merge_syntactic_units(original_sentences, filtered_sentences)

def paste(xsel=False):
    """Returns system clipboard contents."""
    selection = "primary" if xsel else "clipboard"
    try:
        return subprocess.Popen(["xclip", "-selection", selection, "-o"], stdout=subprocess.PIPE).communicate()[0].decode("utf-8")
    except OSError as why:
        raise XclipNotFound

def without(seq1, seq2):
    r"""Return a list with all elements in `seq2` removed from `seq1`, order
    preserved.

    Examples:

    >>> without([1,2,3,1,2], [1])
    [2, 3, 2]
    """
    if isSet(seq2): d2 = seq2
    else: d2 = set(seq2)
    return [elt for elt in seq1 if elt not in d2]

def strip_querystring(url):
    """Remove the querystring from the end of a URL."""
    p = six.moves.urllib.parse.urlparse(url)
    return p.scheme + "://" + p.netloc + p.path

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

def get_numbers(s):
    """Extracts all integers from a string an return them in a list"""

    result = map(int, re.findall(r'[0-9]+', unicode(s)))
    return result + [1] * (2 - len(result))

def comma_converter(float_string):
    """Convert numbers to floats whether the decimal point is '.' or ','"""
    trans_table = maketrans(b',', b'.')
    return float(float_string.translate(trans_table))

def parse_datetime(dt_str, format):
    """Create a timezone-aware datetime object from a datetime string."""
    t = time.strptime(dt_str, format)
    return datetime(t[0], t[1], t[2], t[3], t[4], t[5], t[6], pytz.UTC)

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

def __init__(self, name, contained_key):
        """Instantiate an anonymous file-based Bucket around a single key.
        """
        self.name = name
        self.contained_key = contained_key

def _parse_single_response(cls, response_data):
        """de-serialize a JSON-RPC Response/error

        :Returns: | [result, id] for Responses
        :Raises:  | RPCFault+derivates for error-packages/faults, RPCParseError, RPCInvalidRPC
        """

        if not isinstance(response_data, dict):
            raise errors.RPCInvalidRequest("No valid RPC-package.")

        if "id" not in response_data:
            raise errors.RPCInvalidRequest("""Invalid Response, "id" missing.""")

        request_id = response_data['id']

        if "jsonrpc" not in response_data:
            raise errors.RPCInvalidRequest("""Invalid Response, "jsonrpc" missing.""", request_id)
        if not isinstance(response_data["jsonrpc"], (str, unicode)):
            raise errors.RPCInvalidRequest("""Invalid Response, "jsonrpc" must be a string.""")
        if response_data["jsonrpc"] != "2.0":
            raise errors.RPCInvalidRequest("""Invalid jsonrpc version.""", request_id)

        error = response_data.get('error', None)
        result = response_data.get('result', None)

        if error and result:
            raise errors.RPCInvalidRequest("""Invalid Response, only "result" OR "error" allowed.""", request_id)

        if error:
            if not isinstance(error, dict):
                raise errors.RPCInvalidRequest("Invalid Response, invalid error-object.", request_id)

            if not ("code" in error and "message" in error):
                raise errors.RPCInvalidRequest("Invalid Response, invalid error-object.", request_id)

            error_data = error.get("data", None)

            if error['code'] in errors.ERROR_CODE_CLASS_MAP:
                raise errors.ERROR_CODE_CLASS_MAP[error['code']](error_data, request_id)
            else:
                error_object = errors.RPCFault(error_data, request_id)
                error_object.error_code = error['code']
                error_object.message = error['message']
                raise error_object

        return result, request_id

def dag_longest_path(graph, source, target):
    """
    Finds the longest path in a dag between two nodes
    """
    if source == target:
        return [source]
    allpaths = nx.all_simple_paths(graph, source, target)
    longest_path = []
    for l in allpaths:
        if len(l) > len(longest_path):
            longest_path = l
    return longest_path

def connect_to_database_odbc_access(self,
                                        dsn: str,
                                        autocommit: bool = True) -> None:
        """Connects to an Access database via ODBC, with the DSN
        prespecified."""
        self.connect(engine=ENGINE_ACCESS, interface=INTERFACE_ODBC,
                     dsn=dsn, autocommit=autocommit)

def xml(cls, res, *args, **kwargs):
        """Parses XML from a response."""
        return parse_xml(res.text, *args, **kwargs)

def format_repr(obj, attributes) -> str:
    """Format an object's repr method with specific attributes."""

    attribute_repr = ', '.join(('{}={}'.format(attr, repr(getattr(obj, attr)))
                                for attr in attributes))
    return "{0}({1})".format(obj.__class__.__qualname__, attribute_repr)

def beta_pdf(x, a, b):
  """Beta distirbution probability density function."""
  bc = 1 / beta(a, b)
  fc = x ** (a - 1)
  sc = (1 - x) ** (b - 1)
  return bc * fc * sc

def getAttributeData(self, name, channel=None):
        """ Returns a attribut """
        return self._getNodeData(name, self._ATTRIBUTENODE, channel)

def np2str(value):
    """Convert an `numpy.string_` to str.

    Args:
        value (ndarray): scalar or 1-element numpy array to convert

    Raises:
        ValueError: if value is array larger than 1-element or it is not of
                    type `numpy.string_` or it is not a numpy array

    """
    if hasattr(value, 'dtype') and \
            issubclass(value.dtype.type, (np.string_, np.object_)) and value.size == 1:
        value = np.asscalar(value)
        if not isinstance(value, str):
            # python 3 - was scalar numpy array of bytes
            # otherwise python 2 - scalar numpy array of 'str'
            value = value.decode()
        return value
    else:
        raise ValueError("Array is not a string type or is larger than 1")

def node__name__(self):
        """Return the name of this node or its class name."""

        return self.node.__name__ \
            if self.node.__name__ is not None else self.node.__class__.__name__

def trim(self):
        """Clear not used counters"""
        for key, value in list(iteritems(self.counters)):
            if value.empty():
                del self.counters[key]

def name2rgb(hue):
    """Originally used to calculate color based on module name.
    """
    r, g, b = colorsys.hsv_to_rgb(hue / 360.0, .8, .7)
    return tuple(int(x * 256) for x in [r, g, b])

def _load_ngram(name):
    """Dynamically import the python module with the ngram defined as a dictionary.
    Since bigger ngrams are large files its wasteful to always statically import them if they're not used.
    """
    module = importlib.import_module('lantern.analysis.english_ngrams.{}'.format(name))
    return getattr(module, name)

def fetch_event(urls):
    """
    This parallel fetcher uses gevent one uses gevent
    """
    rs = (grequests.get(u) for u in urls)
    return [content.json() for content in grequests.map(rs)]

def pformat(o, indent=1, width=80, depth=None):
    """Format a Python o into a pretty-printed representation."""
    return PrettyPrinter(indent=indent, width=width, depth=depth).pformat(o)

def sometimesish(fn):
    """
    Has a 50/50 chance of calling a function
    """
    def wrapped(*args, **kwargs):
        if random.randint(1, 2) == 1:
            return fn(*args, **kwargs)

    return wrapped

def x_values_ref(self, series):
        """
        The Excel worksheet reference to the X values for this chart (not
        including the column label).
        """
        top_row = self.series_table_row_offset(series) + 2
        bottom_row = top_row + len(series) - 1
        return "Sheet1!$A$%d:$A$%d" % (top_row, bottom_row)

def __init__(self, function):
		"""function: to be called with each stream element as its
		only argument
		"""
		super(takewhile, self).__init__()
		self.function = function

def apply(self, func, args=(), kwds=dict()):
        """Equivalent of the apply() builtin function. It blocks till
        the result is ready."""
        return self.apply_async(func, args, kwds).get()

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

def decorator(func):
  r"""Makes the passed decorators to support optional args.
  """
  def wrapper(__decorated__=None, *Args, **KwArgs):
    if __decorated__ is None: # the decorator has some optional arguments.
      return lambda _func: func(_func, *Args, **KwArgs)

    else:
      return func(__decorated__, *Args, **KwArgs)

  return wrap(wrapper, func)

def erase(self):
        """White out the progress bar."""
        with self._at_last_line():
            self.stream.write(self._term.clear_eol)
        self.stream.flush()

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

def get_static_url():
    """Return a base static url, always ending with a /"""
    path = getattr(settings, 'STATIC_URL', None)
    if not path:
        path = getattr(settings, 'MEDIA_URL', None)
    if not path:
        path = '/'
    return path

def pprint(j, no_pretty):
    """
    Prints as formatted JSON
    """
    if not no_pretty:
        click.echo(
            json.dumps(j, cls=PotionJSONEncoder, sort_keys=True, indent=4, separators=(",", ": "))
        )
    else:
        click.echo(j)

def set_scrollbars_cb(self, w, tf):
        """This callback is invoked when the user checks the 'Use Scrollbars'
        box in the preferences pane."""
        scrollbars = 'on' if tf else 'off'
        self.t_.set(scrollbars=scrollbars)

def reverse_transform(self, col):
        """Converts data back into original format.

        Args:
            col(pandas.DataFrame): Data to transform.

        Returns:
            pandas.DataFrame
        """

        output = pd.DataFrame()
        output[self.col_name] = self.get_category(col[self.col_name])

        return output

def get_action_methods(self):
        """
        return a list of methods on this class for executing actions.
        methods are return as a list of (name, func) tuples
        """
        return [(name, getattr(self, name))
                for name, _ in Action.get_command_types()]

def set_xlimits(self, min=None, max=None):
        """Set limits for the x-axis.

        :param min: minimum value to be displayed.  If None, it will be
            calculated.
        :param max: maximum value to be displayed.  If None, it will be
            calculated.

        """
        self.limits['xmin'] = min
        self.limits['xmax'] = max

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

def registered_filters_list(self):
        """
        Return the list of registered filters (as a list of strings).

        The list **only** includes registered filters (**not** the predefined :program:`Jinja2` filters).

        """
        return [filter_name for filter_name in self.__jinja2_environment.filters.keys() if filter_name not in self.__jinja2_predefined_filters ]

def get_width():
    """Get terminal width"""
    # Get terminal size
    ws = struct.pack("HHHH", 0, 0, 0, 0)
    ws = fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, ws)
    lines, columns, x, y = struct.unpack("HHHH", ws)
    width = min(columns * 39 // 40, columns - 2)
    return width

def _check_methods(self, methods):
        """ @type methods: tuple """
        for method in methods:
            if method not in self.ALLOWED_METHODS:
                raise Exception('Invalid \'%s\' method' % method)

def detach_index(self, name):
        """

        :param name:
        :return:
        """
        assert type(name) == str

        if name in self._indexes:
            del self._indexes[name]

def to_bipartite_matrix(A):
    """Returns the adjacency matrix of a bipartite graph whose biadjacency
    matrix is `A`.

    `A` must be a NumPy array.

    If `A` has **m** rows and **n** columns, then the returned matrix has **m +
    n** rows and columns.

    """
    m, n = A.shape
    return four_blocks(zeros(m, m), A, A.T, zeros(n, n))

def aug_sysargv(cmdstr):
    """ DEBUG FUNC modify argv to look like you ran a command """
    import shlex
    argv = shlex.split(cmdstr)
    sys.argv.extend(argv)

def wait_and_join(self, task):
        """ Given a task, waits for it until it finishes
        :param task: Task
        :return:
        """
        while not task.has_started:
            time.sleep(self._polling_time)
        task.thread.join()

def apply(filter):
    """Manufacture decorator that filters return value with given function.

    ``filter``:
      Callable that takes a single parameter.
    """
    def decorator(callable):
        return lambda *args, **kwargs: filter(callable(*args, **kwargs))
    return decorator

def set_attached_console_visible(state):
    """Show/hide system console window attached to current process.
       Return it's previous state.

       Availability: Windows"""
    flag = {True: SW_SHOW, False: SW_HIDE}
    return bool(ShowWindow(console_window_handle, flag[state]))

def length(self):
        """Array of vector lengths"""
        return np.sqrt(np.sum(self**2, axis=1)).view(np.ndarray)

def unpickle(pickle_file):
    """Unpickle a python object from the given path."""
    pickle = None
    with open(pickle_file, "rb") as pickle_f:
        pickle = dill.load(pickle_f)
    if not pickle:
        LOG.error("Could not load python object from file")
    return pickle

def close(self):
    """Flush the buffer and finalize the file.

    When this returns the new file is available for reading.
    """
    if not self.closed:
      self.closed = True
      self._flush(finish=True)
      self._buffer = None

def aux_insertTree(childTree, parentTree):
	"""This a private (You shouldn't have to call it) recursive function that inserts a child tree into a parent tree."""
	if childTree.x1 != None and childTree.x2 != None :
		parentTree.insert(childTree.x1, childTree.x2, childTree.name, childTree.referedObject)

	for c in childTree.children:
		aux_insertTree(c, parentTree)

def minify_js(input_files, output_file):
    """
    Minifies the input javascript files to the output file.

    Output file may be same as input to minify in place.

    In debug mode this function just concatenates the files
    without minifying.
    """
    from .modules import minify, utils

    if not isinstance(input_files, (list, tuple)):
        raise RuntimeError('JS minifier takes a list of input files.')

    return {
        'dependencies_fn': utils.no_dependencies,
        'compiler_fn': minify.minify_js,
        'input': input_files,
        'output': output_file,
        'kwargs': {},
    }

def do_history(self, line):
        """history Display a list of commands that have been entered."""
        self._split_args(line, 0, 0)
        for idx, item in enumerate(self._history):
            d1_cli.impl.util.print_info("{0: 3d} {1}".format(idx, item))

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

def _end_del(self):
        """ Deletes the line content after the cursor  """
        text = self.edit_text[:self.edit_pos]
        self.set_edit_text(text)

def mmap(func, iterable):
    """Wrapper to make map() behave the same on Py2 and Py3."""

    if sys.version_info[0] > 2:
        return [i for i in map(func, iterable)]
    else:
        return map(func, iterable)

def send_text(self, text):
        """Send a plain text message to the room."""
        return self.client.api.send_message(self.room_id, text)

def valid_substitution(strlen, index):
    """
    skip performing substitutions that are outside the bounds of the string
    """
    values = index[0]
    return all([strlen > i for i in values])

def _axes(self):
        """Set the _force_vertical flag when rendering axes"""
        self.view._force_vertical = True
        super(HorizontalGraph, self)._axes()
        self.view._force_vertical = False

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

def get_callable_documentation(the_callable):
    """Return a string with the callable signature and its docstring.

    :param the_callable: the callable to be analyzed.
    :type the_callable: function/callable.
    :return: the signature.
    """
    return wrap_text_in_a_box(
        title=get_callable_signature_as_string(the_callable),
        body=(getattr(the_callable, '__doc__') or 'No documentation').replace(
            '\n', '\n\n'),
        style='ascii_double')

def stack_template_url(bucket_name, blueprint, endpoint):
    """Produces an s3 url for a given blueprint.

    Args:
        bucket_name (string): The name of the S3 bucket where the resulting
            templates are stored.
        blueprint (:class:`stacker.blueprints.base.Blueprint`): The blueprint
            object to create the URL to.
        endpoint (string): The s3 endpoint used for the bucket.

    Returns:
        string: S3 URL.
    """
    key_name = stack_template_key_name(blueprint)
    return "%s/%s/%s" % (endpoint, bucket_name, key_name)

def get_rounded(self, digits):
        """ Return a vector with the elements rounded to the given number of digits. """
        result = self.copy()
        result.round(digits)
        return result

def flatten( iterables ):
    """ Flatten an iterable, except for string elements. """
    for it in iterables:
        if isinstance(it, str):
            yield it
        else:
            for element in it:
                yield element

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

def is_not_null(df: DataFrame, col_name: str) -> bool:
    """
    Return ``True`` if the given DataFrame has a column of the given
    name (string), and there exists at least one non-NaN value in that
    column; return ``False`` otherwise.
    """
    if (
        isinstance(df, pd.DataFrame)
        and col_name in df.columns
        and df[col_name].notnull().any()
    ):
        return True
    else:
        return False

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

def fopen(name, mode='r', buffering=-1):
    """Similar to Python's built-in `open()` function."""
    f = _fopen(name, mode, buffering)
    return _FileObjectThreadWithContext(f, mode, buffering)

def equal(obj1, obj2):
    """Calculate equality between two (Comparable) objects."""
    Comparable.log(obj1, obj2, '==')
    equality = obj1.equality(obj2)
    Comparable.log(obj1, obj2, '==', result=equality)
    return equality

def distinct(xs):
    """Get the list of distinct values with preserving order."""
    # don't use collections.OrderedDict because we do support Python 2.6
    seen = set()
    return [x for x in xs if x not in seen and not seen.add(x)]

def quote(s, unsafe='/'):
    """Pass in a dictionary that has unsafe characters as the keys, and the percent
    encoded value as the value."""
    res = s.replace('%', '%25')
    for c in unsafe:
        res = res.replace(c, '%' + (hex(ord(c)).upper())[2:])
    return res

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

def _ipv4_text_to_int(self, ip_text):
        """convert ip v4 string to integer."""
        if ip_text is None:
            return None
        assert isinstance(ip_text, str)
        return struct.unpack('!I', addrconv.ipv4.text_to_bin(ip_text))[0]

def isnumber(*args):
    """Checks if value is an integer, long integer or float.

    NOTE: Treats booleans as numbers, where True=1 and False=0.
    """
    return all(map(lambda c: isinstance(c, int) or isinstance(c, float), args))

def calculate_bbox_area(bbox, rows, cols):
    """Calculate the area of a bounding box in pixels."""
    bbox = denormalize_bbox(bbox, rows, cols)
    x_min, y_min, x_max, y_max = bbox[:4]
    area = (x_max - x_min) * (y_max - y_min)
    return area

def add_text(text, x=0.01, y=0.01, axes="gca", draw=True, **kwargs):
    """
    Adds text to the axes at the specified position.

    **kwargs go to the axes.text() function.
    """
    if axes=="gca": axes = _pylab.gca()
    axes.text(x, y, text, transform=axes.transAxes, **kwargs)
    if draw: _pylab.draw()

def _tab(content):
    """
    Helper funcation that converts text-based get response
    to tab separated values for additional manipulation.
    """
    response = _data_frame(content).to_csv(index=False,sep='\t')
    return response

def get_randomized_guid_sample(self, item_count):
        """ Fetch a subset of randomzied GUIDs from the whitelist """
        dataset = self.get_whitelist()
        random.shuffle(dataset)
        return dataset[:item_count]

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

def ToDatetime(self):
    """Converts Timestamp to datetime."""
    return datetime.utcfromtimestamp(
        self.seconds + self.nanos / float(_NANOS_PER_SECOND))

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

def show(config):
    """Show revision list"""
    with open(config, 'r'):
        main.show(yaml.load(open(config)))

def fix_dashes(string):
    """Fix bad Unicode special dashes in string."""
    string = string.replace(u'\u05BE', '-')
    string = string.replace(u'\u1806', '-')
    string = string.replace(u'\u2E3A', '-')
    string = string.replace(u'\u2E3B', '-')
    string = unidecode(string)
    return re.sub(r'--+', '-', string)

def multi_split(s, split):
    # type: (S, Iterable[S]) -> List[S]
    """Splits on multiple given separators."""
    for r in split:
        s = s.replace(r, "|")
    return [i for i in s.split("|") if len(i) > 0]

def get_index(self, bucket, index, startkey, endkey=None,
                  return_terms=None, max_results=None, continuation=None,
                  timeout=None, term_regex=None):
        """
        Performs a secondary index query.
        """
        raise NotImplementedError

def pascal_row(n):
    """ Returns n-th row of Pascal's triangle
    """
    result = [1]
    x, numerator = 1, n
    for denominator in range(1, n // 2 + 1):
        x *= numerator
        x /= denominator
        result.append(x)
        numerator -= 1
    if n & 1 == 0:
        result.extend(reversed(result[:-1]))
    else:
        result.extend(reversed(result))
    return result

def to_snake_case(text):
    """Convert to snake case.

    :param str text:
    :rtype: str
    :return:
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def sarea_(self, col, x=None, y=None, rsum=None, rmean=None):
		"""
		Get an stacked area chart
		"""
		try:
			charts = self._multiseries(col, x, y, "area", rsum, rmean)
			return hv.Area.stack(charts)
		except Exception as e:
			self.err(e, self.sarea_, "Can not draw stacked area chart")

def const_rand(size, seed=23980):
    """ Generate a random array with a fixed seed.
    """
    old_seed = np.random.seed()
    np.random.seed(seed)
    out = np.random.rand(size)
    np.random.seed(old_seed)
    return out

def dim_axis_label(dimensions, separator=', '):
    """
    Returns an axis label for one or more dimensions.
    """
    if not isinstance(dimensions, list): dimensions = [dimensions]
    return separator.join([d.pprint_label for d in dimensions])

def ylim(self, low, high):
        """Set yaxis limits

        Parameters
        ----------
        low : number
        high : number
        index : int, optional

        Returns
        -------
        Chart

        """
        self.chart['yAxis'][0]['min'] = low
        self.chart['yAxis'][0]['max'] = high
        return self

def draw_header(self, stream, header):
        """Draw header with underline"""
        stream.writeln('=' * (len(header) + 4))
        stream.writeln('| ' + header + ' |')
        stream.writeln('=' * (len(header) + 4))
        stream.writeln()

def _handle_authentication_error(self):
        """
        Return an authentication error.
        """
        response = make_response('Access Denied')
        response.headers['WWW-Authenticate'] = self.auth.get_authenticate_header()
        response.status_code = 401
        return response

def get_longest_line_length(text):
    """Get the length longest line in a paragraph"""
    lines = text.split("\n")
    length = 0

    for i in range(len(lines)):
        if len(lines[i]) > length:
            length = len(lines[i])

    return length

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

def get_area(self):
        """Calculate area of bounding box."""
        return (self.p2.x-self.p1.x)*(self.p2.y-self.p1.y)

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

def get_var(name, factory=None):
    """Gets a global variable given its name.

    If factory is not None and the variable is not set, factory
    is a callable that will set the variable.

    If not set, returns None.
    """
    if name not in _VARS and factory is not None:
        _VARS[name] = factory()
    return _VARS.get(name)

def add_option(self, *args, **kwargs):
        """Add optparse or argparse option depending on CmdHelper initialization."""
        if self.parseTool == 'argparse':
            if args and args[0] == '':   # no short option
                args = args[1:]
            return self.parser.add_argument(*args, **kwargs)
        else:
            return self.parser.add_option(*args, **kwargs)

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

def set_left_to_right(self):
        """Set text direction left to right."""
        self.displaymode |= LCD_ENTRYLEFT
        self.write8(LCD_ENTRYMODESET | self.displaymode)

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

def typescript_compile(source):
    """Compiles the given ``source`` from TypeScript to ES5 using TypescriptServices.js"""
    with open(TS_COMPILER, 'r') as tsservices_js:
        return evaljs(
            (tsservices_js.read(),
             'ts.transpile(dukpy.tscode, {options});'.format(options=TSC_OPTIONS)),
            tscode=source
        )

def pause():
	"""Tell iTunes to pause"""

	if not settings.platformCompatible():
		return False

	(output, error) = subprocess.Popen(["osascript", "-e", PAUSE], stdout=subprocess.PIPE).communicate()

def _modify(item, func):
    """
    Modifies each item.keys() string based on the func passed in.
    Often used with inflection's camelize or underscore methods.

    :param item: dictionary representing item to be modified
    :param func: function to run on each key string
    :return: dictionary where each key has been modified by func.
    """
    result = dict()
    for key in item:
        result[func(key)] = item[key]
    return result

def __exit__(self, type, value, traceback):
        """When the `with` statement ends."""

        if not self.asarfile:
            return

        self.asarfile.close()
        self.asarfile = None

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

def file_matches(filename, patterns):
    """Does this filename match any of the patterns?"""
    return any(fnmatch.fnmatch(filename, pat) for pat in patterns)

def none(self):
        """
        Returns an empty QuerySet.
        """
        return EmptyQuerySet(model=self.model, using=self._using, connection=self._connection)

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

def resizeEvent(self, event):
        """Reimplement Qt method"""
        if not self.isMaximized() and not self.fullscreen_flag:
            self.window_size = self.size()
        QMainWindow.resizeEvent(self, event)

        # To be used by the tour to be able to resize
        self.sig_resized.emit(event)

def get_property_by_name(pif, name):
    """Get a property by name"""
    return next((x for x in pif.properties if x.name == name), None)

def numeric_part(s):
    """Returns the leading numeric part of a string.

    >>> numeric_part("20-alpha")
    20
    >>> numeric_part("foo")
    >>> numeric_part("16b")
    16
    """

    m = re_numeric_part.match(s)
    if m:
        return int(m.group(1))
    return None

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")

def executable_exists(executable):
    """Test if an executable is available on the system."""
    for directory in os.getenv("PATH").split(":"):
        if os.path.exists(os.path.join(directory, executable)):
            return True
    return False

def get_model_index_properties(instance, index):
    """Return the list of properties specified for a model in an index."""
    mapping = get_index_mapping(index)
    doc_type = instance._meta.model_name.lower()
    return list(mapping["mappings"][doc_type]["properties"].keys())

def last_modified_time(path):
    """
    Get the last modified time of path as a Timestamp.
    """
    return pd.Timestamp(os.path.getmtime(path), unit='s', tz='UTC')

def make_dep_graph(depender):
	"""Returns a digraph string fragment based on the passed-in module
	"""
	shutit_global.shutit_global_object.yield_to_draw()
	digraph = ''
	for dependee_id in depender.depends_on:
		digraph = (digraph + '"' + depender.module_id + '"->"' + dependee_id + '";\n')
	return digraph

def GetAttributeNs(self, localName, namespaceURI):
        """Provides the value of the specified attribute """
        ret = libxml2mod.xmlTextReaderGetAttributeNs(self._o, localName, namespaceURI)
        return ret

def _ensure_element(tup, elem):
    """
    Create a tuple containing all elements of tup, plus elem.

    Returns the new tuple and the index of elem in the new tuple.
    """
    try:
        return tup, tup.index(elem)
    except ValueError:
        return tuple(chain(tup, (elem,))), len(tup)

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

def shutdown(self):
        """
        shutdown: to be run by atexit handler. All open connection are closed.
        """
        self.run_clean_thread = False
        self.cleanup(True)
        if self.cleaner_thread.isAlive():
            self.cleaner_thread.join()

def serialize(self):
        """Serialize the query to a structure using the query DSL."""
        data = {'doc': self.doc}
        if isinstance(self.query, Query):
            data['query'] = self.query.serialize()
        return data

def ncores_reserved(self):
        """
        Returns the number of cores reserved in this moment.
        A core is reserved if it's still not running but
        we have submitted the task to the queue manager.
        """
        return sum(task.manager.num_cores for task in self if task.status == task.S_SUB)

def pick_unused_port(self):
    """ Pick an unused port. There is a slight chance that this wont work. """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', 0))
    _, port = s.getsockname()
    s.close()
    return port

def str_from_file(path):
    """
    Return file contents as string.

    """
    with open(path) as f:
        s = f.read().strip()
    return s

def fval(self, instance):
        """return the raw value that this property is holding internally for instance"""
        try:
            val = instance.__dict__[self.instance_field_name]
        except KeyError as e:
            #raise AttributeError(str(e))
            val = None

        return val

def do_striptags(value):
    """Strip SGML/XML tags and replace adjacent whitespace by one space.
    """
    if hasattr(value, '__html__'):
        value = value.__html__()
    return Markup(unicode(value)).striptags()

def segment_intersection(start0, end0, start1, end1):
    r"""Determine the intersection of two line segments.

    Assumes each line is parametric

    .. math::

       \begin{alignat*}{2}
        L_0(s) &= S_0 (1 - s) + E_0 s &&= S_0 + s \Delta_0 \\
        L_1(t) &= S_1 (1 - t) + E_1 t &&= S_1 + t \Delta_1.
       \end{alignat*}

    To solve :math:`S_0 + s \Delta_0 = S_1 + t \Delta_1`, we use the
    cross product:

    .. math::

       \left(S_0 + s \Delta_0\right) \times \Delta_1 =
           \left(S_1 + t \Delta_1\right) \times \Delta_1 \Longrightarrow
       s \left(\Delta_0 \times \Delta_1\right) =
           \left(S_1 - S_0\right) \times \Delta_1.

    Similarly

    .. math::

       \Delta_0 \times \left(S_0 + s \Delta_0\right) =
           \Delta_0 \times \left(S_1 + t \Delta_1\right) \Longrightarrow
       \left(S_1 - S_0\right) \times \Delta_0 =
           \Delta_0 \times \left(S_0 - S_1\right) =
           t \left(\Delta_0 \times \Delta_1\right).

    .. note::

       Since our points are in :math:`\mathbf{R}^2`, the "traditional"
       cross product in :math:`\mathbf{R}^3` will always point in the
       :math:`z` direction, so in the above we mean the :math:`z`
       component of the cross product, rather than the entire vector.

    For example, the diagonal lines

    .. math::

       \begin{align*}
        L_0(s) &= \left[\begin{array}{c} 0 \\ 0 \end{array}\right] (1 - s) +
                  \left[\begin{array}{c} 2 \\ 2 \end{array}\right] s \\
        L_1(t) &= \left[\begin{array}{c} -1 \\ 2 \end{array}\right] (1 - t) +
                  \left[\begin{array}{c} 1 \\ 0 \end{array}\right] t
       \end{align*}

    intersect at :math:`L_0\left(\frac{1}{4}\right) =
    L_1\left(\frac{3}{4}\right) =
    \frac{1}{2} \left[\begin{array}{c} 1 \\ 1 \end{array}\right]`.

    .. image:: ../images/segment_intersection1.png
       :align: center

    .. testsetup:: segment-intersection1, segment-intersection2

       import numpy as np
       from bezier._geometric_intersection import segment_intersection

    .. doctest:: segment-intersection1
       :options: +NORMALIZE_WHITESPACE

       >>> start0 = np.asfortranarray([0.0, 0.0])
       >>> end0 = np.asfortranarray([2.0, 2.0])
       >>> start1 = np.asfortranarray([-1.0, 2.0])
       >>> end1 = np.asfortranarray([1.0, 0.0])
       >>> s, t, _ = segment_intersection(start0, end0, start1, end1)
       >>> s
       0.25
       >>> t
       0.75

    .. testcleanup:: segment-intersection1

       import make_images
       make_images.segment_intersection1(start0, end0, start1, end1, s)

    Taking the parallel (but different) lines

    .. math::

       \begin{align*}
        L_0(s) &= \left[\begin{array}{c} 1 \\ 0 \end{array}\right] (1 - s) +
                  \left[\begin{array}{c} 0 \\ 1 \end{array}\right] s \\
        L_1(t) &= \left[\begin{array}{c} -1 \\ 3 \end{array}\right] (1 - t) +
                  \left[\begin{array}{c} 3 \\ -1 \end{array}\right] t
       \end{align*}

    we should be able to determine that the lines don't intersect, but
    this function is not meant for that check:

    .. image:: ../images/segment_intersection2.png
       :align: center

    .. doctest:: segment-intersection2
       :options: +NORMALIZE_WHITESPACE

       >>> start0 = np.asfortranarray([1.0, 0.0])
       >>> end0 = np.asfortranarray([0.0, 1.0])
       >>> start1 = np.asfortranarray([-1.0, 3.0])
       >>> end1 = np.asfortranarray([3.0, -1.0])
       >>> _, _, success = segment_intersection(start0, end0, start1, end1)
       >>> success
       False

    .. testcleanup:: segment-intersection2

       import make_images
       make_images.segment_intersection2(start0, end0, start1, end1)

    Instead, we use :func:`parallel_lines_parameters`:

    .. testsetup:: segment-intersection2-continued

       import numpy as np
       from bezier._geometric_intersection import parallel_lines_parameters

       start0 = np.asfortranarray([1.0, 0.0])
       end0 = np.asfortranarray([0.0, 1.0])
       start1 = np.asfortranarray([-1.0, 3.0])
       end1 = np.asfortranarray([3.0, -1.0])

    .. doctest:: segment-intersection2-continued

       >>> disjoint, _ = parallel_lines_parameters(start0, end0, start1, end1)
       >>> disjoint
       True

    .. note::

       There is also a Fortran implementation of this function, which
       will be used if it can be built.

    Args:
        start0 (numpy.ndarray): A 1D NumPy ``2``-array that is the start
            vector :math:`S_0` of the parametric line :math:`L_0(s)`.
        end0 (numpy.ndarray): A 1D NumPy ``2``-array that is the end
            vector :math:`E_0` of the parametric line :math:`L_0(s)`.
        start1 (numpy.ndarray): A 1D NumPy ``2``-array that is the start
            vector :math:`S_1` of the parametric line :math:`L_1(s)`.
        end1 (numpy.ndarray): A 1D NumPy ``2``-array that is the end
            vector :math:`E_1` of the parametric line :math:`L_1(s)`.

    Returns:
        Tuple[float, float, bool]: Pair of :math:`s_{\ast}` and
        :math:`t_{\ast}` such that the lines intersect:
        :math:`L_0\left(s_{\ast}\right) = L_1\left(t_{\ast}\right)` and then
        a boolean indicating if an intersection was found (i.e. if the lines
        aren't parallel).
    """
    delta0 = end0 - start0
    delta1 = end1 - start1
    cross_d0_d1 = _helpers.cross_product(delta0, delta1)
    if cross_d0_d1 == 0.0:
        return None, None, False

    else:
        start_delta = start1 - start0
        s = _helpers.cross_product(start_delta, delta1) / cross_d0_d1
        t = _helpers.cross_product(start_delta, delta0) / cross_d0_d1
        return s, t, True

def reseed_random(seed):
    """Reseed factory.fuzzy's random generator."""
    r = random.Random(seed)
    random_internal_state = r.getstate()
    set_random_state(random_internal_state)

def is_square_matrix(mat):
    """Test if an array is a square matrix."""
    mat = np.array(mat)
    if mat.ndim != 2:
        return False
    shape = mat.shape
    return shape[0] == shape[1]

def register_logging_factories(loader):
    """
    Registers default factories for logging standard package.

    :param loader: Loader where you want register default logging factories
    """
    loader.register_factory(logging.Logger, LoggerFactory)
    loader.register_factory(logging.Handler, LoggingHandlerFactory)

def pop(h):
    """Pop the heap value from the heap."""
    n = h.size() - 1
    h.swap(0, n)
    down(h, 0, n)
    return h.pop()

def _get_env(self, env_var):
        """Helper to read an environment variable
        """
        value = os.environ.get(env_var)
        if not value:
            raise ValueError('Missing environment variable:%s' % env_var)
        return value

def stop(self) -> None:
        """Stops the analysis as soon as possible."""
        if self._stop and not self._posted_kork:
            self._stop()
            self._stop = None

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

def _one_exists(input_files):
    """
    at least one file must exist for multiqc to run properly
    """
    for f in input_files:
        if os.path.exists(f):
            return True
    return False

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

def static_method(cls, f):
        """Decorator which dynamically binds static methods to the model for later use."""
        setattr(cls, f.__name__, staticmethod(f))
        return f

def export_all(self):
		query = """
			SELECT quote, library, logid
			from quotes
			left outer join quote_log on quotes.quoteid = quote_log.quoteid
			"""
		fields = 'text', 'library', 'log_id'
		return (dict(zip(fields, res)) for res in self.db.execute(query))

def _infer_interval_breaks(coord):
    """
    >>> _infer_interval_breaks(np.arange(5))
    array([-0.5,  0.5,  1.5,  2.5,  3.5,  4.5])

    Taken from xarray.plotting.plot module
    """
    coord = np.asarray(coord)
    deltas = 0.5 * (coord[1:] - coord[:-1])
    first = coord[0] - deltas[0]
    last = coord[-1] + deltas[-1]
    return np.r_[[first], coord[:-1] + deltas, [last]]

def filter_dict(d, keys):
    """
    Creates a new dict from an existing dict that only has the given keys
    """
    return {k: v for k, v in d.items() if k in keys}

def detokenize(s):
    """ Detokenize a string by removing spaces before punctuation."""
    print(s)
    s = re.sub("\s+([;:,\.\?!])", "\\1", s)
    s = re.sub("\s+(n't)", "\\1", s)
    return s

def filtered_image(self, im):
        """Returns a filtered image after applying the Fourier-space filters"""
        q = np.fft.fftn(im)
        for k,v in self.filters:
            q[k] -= v
        return np.real(np.fft.ifftn(q))

def get_object_attrs(obj):
    """
    Get the attributes of an object using dir.

    This filters protected attributes
    """
    attrs = [k for k in dir(obj) if not k.startswith('__')]
    if not attrs:
        attrs = dir(obj)
    return attrs

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

def numpy_to_yaml(representer: Representer, data: np.ndarray) -> Sequence[Any]:
    """ Write a numpy array to YAML.

    It registers the array under the tag ``!numpy_array``.

    Use with:

    .. code-block:: python

        >>> yaml = ruamel.yaml.YAML()
        >>> yaml.representer.add_representer(np.ndarray, yaml.numpy_to_yaml)

    Note:
        We cannot use ``yaml.register_class`` because it won't register the proper type.
        (It would register the type of the class, rather than of `numpy.ndarray`). Instead,
        we use the above approach to register this method explicitly with the representer.
    """
    return representer.represent_sequence(
        "!numpy_array",
        data.tolist()
    )

def empty_tree(input_list):
    """Recursively iterate through values in nested lists."""
    for item in input_list:
        if not isinstance(item, list) or not empty_tree(item):
            return False
    return True

def cli(yamlfile, format, context):
    """ Generate JSONLD file from biolink schema """
    print(JSONLDGenerator(yamlfile, format).serialize(context=context))

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

def getLinesFromLogFile(stream):
    """
    Returns all lines written to the passed in stream
    """
    stream.flush()
    stream.seek(0)
    lines = stream.readlines()
    return lines

def register():
    """
    Calls the shots, based on signals
    """
    signals.article_generator_finalized.connect(link_source_files)
    signals.page_generator_finalized.connect(link_source_files)
    signals.page_writer_finalized.connect(write_source_files)

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

def MatrixInverse(a, adj):
    """
    Matrix inversion op.
    """
    return np.linalg.inv(a if not adj else _adjoint(a)),

def rank(idx, dim):
    """Calculate the index rank according to Bertran's notation."""
    idxm = multi_index(idx, dim)
    out = 0
    while idxm[-1:] == (0,):
        out += 1
        idxm = idxm[:-1]
    return out

def encode(strs):
    """Encodes a list of strings to a single string.
    :type strs: List[str]
    :rtype: str
    """
    res = ''
    for string in strs.split():
        res += str(len(string)) + ":" + string
    return res

def get_duckduckgo_links(limit, params, headers):
	"""
	function to fetch links equal to limit

	duckduckgo pagination is not static, so there is a limit on
	maximum number of links that can be scraped
	"""
	resp = s.get('https://duckduckgo.com/html', params = params, headers = headers)
	links = scrape_links(resp.content, engine = 'd')
	return links[:limit]

def print_out(self, *lst):
      """ Print list of strings to the predefined stdout. """
      self.print2file(self.stdout, True, True, *lst)

def prevmonday(num):
    """
    Return unix SECOND timestamp of "num" mondays ago
    """
    today = get_today()
    lastmonday = today - timedelta(days=today.weekday(), weeks=num)
    return lastmonday

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

def get_view_selection(self):
        """Get actual tree selection object and all respective models of selected rows"""
        if not self.MODEL_STORAGE_ID:
            return None, None

        # avoid selection requests on empty tree views -> case warnings in gtk3
        if len(self.store) == 0:
            paths = []
        else:
            model, paths = self._tree_selection.get_selected_rows()

        # get all related models for selection from respective tree store field
        selected_model_list = []
        for path in paths:
            model = self.store[path][self.MODEL_STORAGE_ID]
            selected_model_list.append(model)
        return self._tree_selection, selected_model_list

def get_key_goids(self, goids):
        """Given GO IDs, return key GO IDs."""
        go2obj = self.go2obj
        return set(go2obj[go].id for go in goids)

def map_tree(visitor, tree):
    """Apply function to nodes"""
    newn = [map_tree(visitor, node) for node in tree.nodes]
    return visitor(tree, newn)

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

def set_default(self_,param_name,value):
        """
        Set the default value of param_name.

        Equivalent to setting param_name on the class.
        """
        cls = self_.cls
        setattr(cls,param_name,value)

def get_subject(self, msg):
        """Extracts the subject line from an EmailMessage object."""

        text, encoding = decode_header(msg['subject'])[-1]

        try:
            text = text.decode(encoding)

        # If it's already decoded, ignore error
        except AttributeError:
            pass

        return text

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

def to_str(s):
    """
    Convert bytes and non-string into Python 3 str
    """
    if isinstance(s, bytes):
        s = s.decode('utf-8')
    elif not isinstance(s, str):
        s = str(s)
    return s

def normalize_path(path):
    """
    Convert a path to its canonical, case-normalized, absolute version.

    """
    return os.path.normcase(os.path.realpath(os.path.expanduser(path)))

async def send(self, data):
        """ Add data to send queue. """
        self.writer.write(data)
        await self.writer.drain()

def download_url(url, filename, headers):
    """Download a file from `url` to `filename`."""
    ensure_dirs(filename)
    response = requests.get(url, headers=headers, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(16 * 1024):
                f.write(chunk)

def process_result_value(self, value, dialect):
        """convert value from json to a python object"""
        if value is not None:
            value = simplejson.loads(value)
        return value

def standardize(table, with_std=True):
    """
    Perform Z-Normalization on each numeric column of the given table.

    Parameters
    ----------
    table : pandas.DataFrame or numpy.ndarray
        Data to standardize.

    with_std : bool, optional, default: True
        If ``False`` data is only centered and not converted to unit variance.

    Returns
    -------
    normalized : pandas.DataFrame
        Table with numeric columns normalized.
        Categorical columns in the input table remain unchanged.
    """
    if isinstance(table, pandas.DataFrame):
        cat_columns = table.select_dtypes(include=['category']).columns
    else:
        cat_columns = []

    new_frame = _apply_along_column(table, standardize_column, with_std=with_std)

    # work around for apply converting category dtype to object
    # https://github.com/pydata/pandas/issues/9573
    for col in cat_columns:
        new_frame[col] = table[col].copy()

    return new_frame

def delete(self, mutagen_file):
        """Remove all images from the file.
        """
        for cover_tag in self.TAG_NAMES.values():
            try:
                del mutagen_file[cover_tag]
            except KeyError:
                pass

def _calculate_similarity(c):
    """Get a similarity matrix of % of shared sequence

    :param c: cluster object

    :return ma: similarity matrix
    """
    ma = {}
    for idc in c:
        set1 = _get_seqs(c[idc])
        [ma.update({(idc, idc2): _common(set1, _get_seqs(c[idc2]), idc, idc2)}) for idc2 in c if idc != idc2 and (idc2, idc) not in ma]
    # logger.debug("_calculate_similarity_ %s" % ma)
    return ma

def get_geoip(ip):
    """Lookup country for IP address."""
    reader = geolite2.reader()
    ip_data = reader.get(ip) or {}
    return ip_data.get('country', {}).get('iso_code')

def check_no_element_by_selector(self, selector):
    """Assert an element does not exist matching the given selector."""
    elems = find_elements_by_jquery(world.browser, selector)
    if elems:
        raise AssertionError("Expected no matching elements, found {}.".format(
            len(elems)))

def as_tree(context):
    """Return info about an object's members as JSON"""

    tree = _build_tree(context, 2, 1)
    if type(tree) == dict:
        tree = [tree] 
    
    return Response(content_type='application/json', body=json.dumps(tree))

def Value(self, name):
    """Returns the value coresponding to the given enum name."""
    if name in self._enum_type.values_by_name:
      return self._enum_type.values_by_name[name].number
    raise ValueError('Enum %s has no value defined for name %s' % (
        self._enum_type.name, name))

def camel_case_from_underscores(string):
    """generate a CamelCase string from an underscore_string."""
    components = string.split('_')
    string = ''
    for component in components:
        string += component[0].upper() + component[1:]
    return string

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

def str_time_to_day_seconds(time):
    """
    Converts time strings to integer seconds
    :param time: %H:%M:%S string
    :return: integer seconds
    """
    t = str(time).split(':')
    seconds = int(t[0]) * 3600 + int(t[1]) * 60 + int(t[2])
    return seconds

def singularize(word):
    """
    Return the singular form of a word, the reverse of :func:`pluralize`.

    Examples::

        >>> singularize("posts")
        "post"
        >>> singularize("octopi")
        "octopus"
        >>> singularize("sheep")
        "sheep"
        >>> singularize("word")
        "word"
        >>> singularize("CamelOctopi")
        "CamelOctopus"

    """
    for inflection in UNCOUNTABLES:
        if re.search(r'(?i)\b(%s)\Z' % inflection, word):
            return word

    for rule, replacement in SINGULARS:
        if re.search(rule, word):
            return re.sub(rule, replacement, word)
    return word

def cleanLines(source, lineSep=os.linesep):
    """
    :param source: some iterable source (list, file, etc)
    :param lineSep: string of separators (chars) that must be removed
    :return: list of non empty lines with removed separators
    """
    stripped = (line.strip(lineSep) for line in source)
    return (line for line in stripped if len(line) != 0)

def to_distribution_values(self, values):
        """
        Returns numpy array of natural logarithms of ``values``.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # avoid RuntimeWarning: divide by zero encountered in log
            return numpy.log(values)

def yvals(self):
        """All y values"""
        return [
            val[1] for serie in self.series for val in serie.values
            if val[1] is not None
        ]

def sha1(s):
    """ Returns a sha1 of the given string
    """
    h = hashlib.new('sha1')
    h.update(s)
    return h.hexdigest()

def get_readonly_fields(self, request, obj=None):
        """Set all fields readonly."""
        return list(self.readonly_fields) + [field.name for field in obj._meta.fields]

def indexes_equal(a: Index, b: Index) -> bool:
    """
    Are two indexes equal? Checks by comparing ``str()`` versions of them.
    (AM UNSURE IF THIS IS ENOUGH.)
    """
    return str(a) == str(b)

def is_file(path):
    """Determine if a Path or string is a file on the file system."""
    try:
        return path.expanduser().absolute().is_file()
    except AttributeError:
        return os.path.isfile(os.path.abspath(os.path.expanduser(str(path))))

def get_selected_values(self, selection):
        """Return a list of values for the given selection."""
        return [v for b, v in self._choices if b & selection]

def str_check(*args, func=None):
    """Check if arguments are str type."""
    func = func or inspect.stack()[2][3]
    for var in args:
        if not isinstance(var, (str, collections.UserString, collections.abc.Sequence)):
            name = type(var).__name__
            raise StringError(
                f'Function {func} expected str, {name} got instead.')

def cached_query(qs, timeout=None):
    """ Auto cached queryset and generate results.
    """
    cache_key = generate_cache_key(qs)
    return get_cached(cache_key, list, args=(qs,), timeout=None)

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

def sort_fn_list(fn_list):
    """Sort input filename list by datetime
    """
    dt_list = get_dt_list(fn_list)
    fn_list_sort = [fn for (dt,fn) in sorted(zip(dt_list,fn_list))]
    return fn_list_sort

def normalize(self, string):
        """Normalize the string according to normalization list"""
        return ''.join([self._normalize.get(x, x) for x in nfd(string)])

def _last_index(x, default_dim):
  """Returns the last dimension's index or default_dim if x has no shape."""
  if x.get_shape().ndims is not None:
    return len(x.get_shape()) - 1
  else:
    return default_dim

def pid_exists(pid):
    """ Determines if a system process identifer exists in process table.
        """
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    else:
        return True

def get_next_scheduled_time(cron_string):
    """Calculate the next scheduled time by creating a crontab object
    with a cron string"""
    itr = croniter.croniter(cron_string, datetime.utcnow())
    return itr.get_next(datetime)

def get_value(self) -> Decimal:
        """ Returns the current value of stocks """
        quantity = self.get_quantity()
        price = self.get_last_available_price()
        if not price:
            # raise ValueError("no price found for", self.full_symbol)
            return Decimal(0)

        value = quantity * price.value
        return value

def split(s):
  """Uses dynamic programming to infer the location of spaces in a string without spaces."""
  l = [_split(x) for x in _SPLIT_RE.split(s)]
  return [item for sublist in l for item in sublist]

def fields(self):
        """Returns the list of field names of the model."""
        return (self.attributes.values() + self.lists.values()
                + self.references.values())

def _delete_whitespace(self):
        """Delete all whitespace from the end of the line."""
        while isinstance(self._lines[-1], (self._Space, self._LineBreak,
                                           self._Indent)):
            del self._lines[-1]

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

def is_iterable_but_not_string(obj):
    """
    Determine whether or not obj is iterable but not a string (eg, a list, set, tuple etc).
    """
    return hasattr(obj, '__iter__') and not isinstance(obj, str) and not isinstance(obj, bytes)

def fetch_header(self):
        """Make a header request to the endpoint."""
        query = self.query().add_query_parameter(req='header')
        return self._parse_messages(self.get_query(query).content)[0]

def dot_v2(vec1, vec2):
    """Return the dot product of two vectors"""

    return vec1.x * vec2.x + vec1.y * vec2.y

def print_env_info(key, out=sys.stderr):
    """If given environment key is defined, print it out."""
    value = os.getenv(key)
    if value is not None:
        print(key, "=", repr(value), file=out)

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

def detach_all(self):
        """
        Detach from all tracked classes and objects.
        Restore the original constructors and cleanse the tracking lists.
        """
        self.detach_all_classes()
        self.objects.clear()
        self.index.clear()
        self._keepalive[:] = []

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

def argsort_k_smallest(x, k):
    """ Return no more than ``k`` indices of smallest values. """
    if k == 0:
        return np.array([], dtype=np.intp)
    if k is None or k >= len(x):
        return np.argsort(x)
    indices = np.argpartition(x, k)[:k]
    values = x[indices]
    return indices[np.argsort(values)]

def horz_dpi(self):
        """
        Integer dots per inch for the width of this image. Defaults to 72
        when not present in the file, as is often the case.
        """
        pHYs = self._chunks.pHYs
        if pHYs is None:
            return 72
        return self._dpi(pHYs.units_specifier, pHYs.horz_px_per_unit)

def compose_all(tups):
  """Compose all given tuples together."""
  from . import ast  # I weep for humanity
  return functools.reduce(lambda x, y: x.compose(y), map(ast.make_tuple, tups), ast.make_tuple({}))

def tree_predict(x, root, proba=False, regression=False):
    """Predicts a probabilities/value/label for the sample x.
    """

    if isinstance(root, Leaf):
        if proba:
            return root.probabilities
        elif regression:
            return root.mean
        else:
            return root.most_frequent

    if root.question.match(x):
        return tree_predict(x, root.true_branch, proba=proba, regression=regression)
    else:
        return tree_predict(x, root.false_branch, proba=proba, regression=regression)

def _to_diagonally_dominant(mat):
    """Make matrix unweighted diagonally dominant using the Laplacian."""
    mat += np.diag(np.sum(mat != 0, axis=1) + 0.01)
    return mat

def imagemagick(color_count, img, magick_command):
    """Call Imagemagick to generate a scheme."""
    flags = ["-resize", "25%", "-colors", str(color_count),
             "-unique-colors", "txt:-"]
    img += "[0]"

    return subprocess.check_output([*magick_command, img, *flags]).splitlines()

def enrich_complexes(graph: BELGraph) -> None:
    """Add all of the members of the complex abundances to the graph."""
    nodes = list(get_nodes_by_function(graph, COMPLEX))
    for u in nodes:
        for v in u.members:
            graph.add_has_component(u, v)

def split_string(text, chars_per_string):
    """
    Splits one string into multiple strings, with a maximum amount of `chars_per_string` characters per string.
    This is very useful for splitting one giant message into multiples.

    :param text: The text to split
    :param chars_per_string: The number of characters per line the text is split into.
    :return: The splitted text as a list of strings.
    """
    return [text[i:i + chars_per_string] for i in range(0, len(text), chars_per_string)]

def comment (self, s, **args):
        """Write GML comment."""
        self.writeln(s=u'comment "%s"' % s, **args)

def cat_acc(y_true, y_pred):
    """Categorical accuracy
    """
    return np.mean(y_true.argmax(axis=1) == y_pred.argmax(axis=1))

def multi_pop(d, *args):
    """ pops multiple keys off a dict like object """
    retval = {}
    for key in args:
        if key in d:
            retval[key] = d.pop(key)
    return retval

def __is__(cls, s):
        """Test if string matches this argument's format."""
        return s.startswith(cls.delims()[0]) and s.endswith(cls.delims()[1])

def keyReleaseEvent(self, event):
        """
        Pyqt specific key release callback function.
        Translates and forwards events to :py:func:`keyboard_event`.
        """
        self.keyboard_event(event.key(), self.keys.ACTION_RELEASE, 0)

def _to_bstr(l):
    """Convert to byte string."""

    if isinstance(l, str):
        l = l.encode('ascii', 'backslashreplace')
    elif not isinstance(l, bytes):
        l = str(l).encode('ascii', 'backslashreplace')
    return l

def daterange(start, end, delta=timedelta(days=1), lower=Interval.CLOSED, upper=Interval.OPEN):
    """Returns a generator which creates the next value in the range on demand"""
    date_interval = Interval(lower=lower, lower_value=start, upper_value=end, upper=upper)
    current = start if start in date_interval else start + delta
    while current in date_interval:
        yield current
        current = current + delta

def close(self, wait=False):
        """Close session, shutdown pool."""
        self.session.close()
        self.pool.shutdown(wait=wait)

def strip_sdist_extras(filelist):
    """Strip generated files that are only present in source distributions.

    We also strip files that are ignored for other reasons, like
    command line arguments, setup.cfg rules or MANIFEST.in rules.
    """
    return [name for name in filelist
            if not file_matches(name, IGNORE)
            and not file_matches_regexps(name, IGNORE_REGEXPS)]

def remove(parent, idx):
  """Remove a value from a dict."""
  if isinstance(parent, dict):
    del parent[idx]
  elif isinstance(parent, list):
    del parent[int(idx)]
  else:
    raise JSONPathError("Invalid path for operation")

def split_every(iterable, n):  # TODO: Remove this, or make it return a generator.
    """
    A generator of n-length chunks of an input iterable
    """
    i = iter(iterable)
    piece = list(islice(i, n))
    while piece:
        yield piece
        piece = list(islice(i, n))

def str_ripper(self, text):
        """Got this code from here:
        http://stackoverflow.com/questions/6116978/python-replace-multiple-strings

        This method takes a set of strings, A, and removes all whole
        elements of set A from string B.

        Input: text string to strip based on instance attribute self.censor
        Output: a stripped (censored) text string
        """
        return self.pattern.sub(lambda m: self.rep[re.escape(m.group(0))], text)

def putkeyword(self, keyword, value, makesubrecord=False):
        """Put the value of a column keyword.
        (see :func:`table.putcolkeyword`)"""
        return self._table.putcolkeyword(self._column, keyword, value, makesubrecord)

def not0(a):
    """Return u if u!= 0, return 1 if u == 0"""
    return matrix(list(map(lambda x: 1 if x == 0 else x, a)), a.size)

def disconnect(self):
        """
        Closes the connection.
        """
        self.logger.debug('Close connection...')

        self.auto_reconnect = False

        if self.websocket is not None:
            self.websocket.close()

def get_jsonparsed_data(url):
    """Receive the content of ``url``, parse it as JSON and return the
       object.
    """
    response = urlopen(url)
    data = response.read().decode('utf-8')
    return json.loads(data)

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

def generate_write_yaml_to_file(file_name):
    """ generate a method to write the configuration in yaml to the method desired """
    def write_yaml(config):
        with open(file_name, 'w+') as fh:
            fh.write(yaml.dump(config))
    return write_yaml

def release(self):
        """
        Releases this resource back to the pool it came from.
        """
        if self.errored:
            self.pool.delete_resource(self)
        else:
            self.pool.release(self)

def _print_memory(self, memory):
        """Print memory.
        """
        for addr, value in memory.items():
            print("    0x%08x : 0x%08x (%d)" % (addr, value, value))

def put(self, endpoint: str, **kwargs) -> dict:
        """HTTP PUT operation to API endpoint."""

        return self._request('PUT', endpoint, **kwargs)

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

def is_sqlatype_integer(coltype: Union[TypeEngine, VisitableType]) -> bool:
    """
    Is the SQLAlchemy column type an integer type?
    """
    coltype = _coltype_to_typeengine(coltype)
    return isinstance(coltype, sqltypes.Integer)

def _get_wow64():
    """
    Determines if the current process is running in Windows-On-Windows 64 bits.

    @rtype:  bool
    @return: C{True} of the current process is a 32 bit program running in a
        64 bit version of Windows, C{False} if it's either a 32 bit program
        in a 32 bit Windows or a 64 bit program in a 64 bit Windows.
    """
    # Try to determine if the debugger itself is running on WOW64.
    # On error assume False.
    if bits == 64:
        wow64 = False
    else:
        try:
            wow64 = IsWow64Process( GetCurrentProcess() )
        except Exception:
            wow64 = False
    return wow64

def recClearTag(element):
    """Applies maspy.xml.clearTag() to the tag attribute of the "element" and
    recursively to all child elements.

    :param element: an :instance:`xml.etree.Element`
    """
    children = element.getchildren()
    if len(children) > 0:
        for child in children:
            recClearTag(child)
    element.tag = clearTag(element.tag)

def RecurseKeys(self):
    """Recurses the subkeys starting with the key.

    Yields:
      WinRegistryKey: Windows Registry key.
    """
    yield self
    for subkey in self.GetSubkeys():
      for key in subkey.RecurseKeys():
        yield key

def assert_in(obj, seq, message=None, extra=None):
    """Raises an AssertionError if obj is not in seq."""
    assert obj in seq, _assert_fail_message(message, obj, seq, "is not in", extra)

def check_precomputed_distance_matrix(X):
    """Perform check_array(X) after removing infinite values (numpy.inf) from the given distance matrix.
    """
    tmp = X.copy()
    tmp[np.isinf(tmp)] = 1
    check_array(tmp)

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

def get_dt_list(fn_list):
    """Get list of datetime objects, extracted from a filename
    """
    dt_list = np.array([fn_getdatetime(fn) for fn in fn_list])
    return dt_list

def any_contains_any(strings, candidates):
    """Whether any of the strings contains any of the candidates."""
    for string in strings:
        for c in candidates:
            if c in string:
                return True

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

def scroll_element_into_view(self):
        """Scroll element into view

        :returns: page element instance
        """
        x = self.web_element.location['x']
        y = self.web_element.location['y']
        self.driver.execute_script('window.scrollTo({0}, {1})'.format(x, y))
        return self

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

def done(self, result):
        """save the geometry before dialog is close to restore it later"""
        self._geometry = self.geometry()
        QtWidgets.QDialog.done(self, result)

def is_primary(self):
        """``True`` if this is a primary key; ``False`` if this is a subkey"""
        return isinstance(self._key, Primary) and not isinstance(self._key, Sub)

def _transform_triple_numpy(x):
    """Transform triple index into a 1-D numpy array."""
    return np.array([x.head, x.relation, x.tail], dtype=np.int64)

def raise_os_error(_errno, path=None):
    """
    Helper for raising the correct exception under Python 3 while still
    being able to raise the same common exception class in Python 2.7.
    """

    msg = "%s: '%s'" % (strerror(_errno), path) if path else strerror(_errno)
    raise OSError(_errno, msg)

async def delete(self):
        """
        Delete this message

        :return: bool
        """
        return await self.bot.delete_message(self.chat.id, self.message_id)

def key_to_metric(self, key):
        """Replace all non-letter characters with underscores"""
        return ''.join(l if l in string.letters else '_' for l in key)

def configure_relation(graph, ns, mappings):
    """
    Register relation endpoint(s) between two resources.

    """
    convention = RelationConvention(graph)
    convention.configure(ns, mappings)

def datetime_local_to_utc(local):
    """
    Simple function to convert naive :std:`datetime.datetime` object containing
    local time to a naive :std:`datetime.datetime` object with UTC time.
    """
    timestamp = time.mktime(local.timetuple())
    return datetime.datetime.utcfromtimestamp(timestamp)

def inner(tensor0: BKTensor, tensor1: BKTensor) -> BKTensor:
    """Return the inner product between two tensors"""
    # Note: Relying on fact that vdot flattens arrays
    return np.vdot(tensor0, tensor1)

def list_apis(awsclient):
    """List APIs in account."""
    client_api = awsclient.get_client('apigateway')

    apis = client_api.get_rest_apis()['items']

    for api in apis:
        print(json2table(api))

def validate(schema, data, owner=None):
    """Validate input data with input schema.

    :param Schema schema: schema able to validate input data.
    :param data: data to validate.
    :param Schema owner: input schema parent schema.
    :raises: Exception if the data is not validated.
    """
    schema._validate(data=data, owner=owner)

def reset(self):
        """Reset analyzer state
        """
        self.prevframe = None
        self.wasmoving = False
        self.t0 = 0
        self.ismoving = False

def _get_triplet_value_list(self, graph, identity, rdf_type):
        """
        Get a list of values from RDF triples when more than one may be present
        """
        values = []
        for elem in graph.objects(identity, rdf_type):
            values.append(elem.toPython())
        return values

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

async def async_run(self) -> None:
        """
        Asynchronously run the worker, does not close connections. Useful when testing.
        """
        self.main_task = self.loop.create_task(self.main())
        await self.main_task