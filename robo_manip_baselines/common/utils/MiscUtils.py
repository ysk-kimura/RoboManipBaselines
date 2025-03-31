def remove_prefix(s, prefix):
    """Support Python 3.8 and below."""
    if s.startswith(prefix):
        return s[len(prefix) :]
    else:
        return s


def remove_suffix(s, suffix):
    """Support Python 3.8 and below."""
    if s.endswith(suffix):
        return s[: -len(suffix)]
    else:
        return s
