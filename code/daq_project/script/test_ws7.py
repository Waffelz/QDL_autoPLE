from ctypes import c_long, byref

def get_result_mode(lib):
    """Return mode int if supported, else None."""
    if not hasattr(lib, "GetResultMode"):
        return None

    # Many WLM DLLs use: long GetResultMode(long dummy)
    for arg in (0, c_long(0)):
        try:
            return int(lib.GetResultMode(arg))
        except Exception:
            pass

    # Some variants use pointer-style: long GetResultMode(long* mode)
    try:
        m = c_long()
        lib.GetResultMode(byref(m))
        return int(m.value)
    except Exception:
        return None


def set_result_mode(lib, mode: int) -> bool:
    """Try to set mode; return True if call succeeds."""
    if not hasattr(lib, "SetResultMode"):
        return False

    for arg in (int(mode), c_long(int(mode))):
        try:
            lib.SetResultMode(arg)
            return True
        except Exception:
            pass

    return False
