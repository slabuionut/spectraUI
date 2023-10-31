from contextlib import contextmanager
import sys
import warnings
import re
import functools
import os

__all__ = ['all_warnings', 'expected_warnings', 'warn']


warn = functools.partial(warnings.warn, stacklevel=2)


@contextmanager
def all_warnings():
    import inspect
    frame = inspect.currentframe()
    if frame:
        for f in inspect.getouterframes(frame):
            f[0].f_locals['__warningregistry__'] = {}
    del frame

    for mod_name, mod in list(sys.modules.items()):
        try:
            mod.__warningregistry__.clear()
        except AttributeError:
            pass

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        yield w


@contextmanager
def expected_warnings(matching):
    if isinstance(matching, str):
        raise ValueError('``matching`` should be a list of strings and not '
                         'a string itself.')

    if matching is None:
        yield None
        return

    strict_warnings = os.environ.get('PACKAGE_TEST_STRICT_WARNINGS', '1')
    if strict_warnings.lower() == 'true':
        strict_warnings = True
    elif strict_warnings.lower() == 'false':
        strict_warnings = False
    else:
        strict_warnings = bool(int(strict_warnings))

    with all_warnings() as w:

        yield w
        while None in matching:
            matching.remove(None)
        remaining = [m for m in matching if r'\A\Z' not in m.split('|')]
        for warn in w:
            found = False
            for match in matching:
                if re.search(match, str(warn.message)) is not None:
                    found = True
                    if match in remaining:
                        remaining.remove(match)
            if strict_warnings and not found:
                raise ValueError(f'Unexpected warning: {str(warn.message)}')
        if strict_warnings and (len(remaining) > 0):
            newline = "\n"
            msg = f"No warning raised matching:{newline}{newline.join(remaining)}"
            raise ValueError(msg)
