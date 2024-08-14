from typing import Any, Callable, Sequence

def select_output(f: Callable[..., Sequence[Any], argnum: int]) -> Callable[..., Any]:
    def f_selected(*args, **kwargs):
        return f(*args, **kwargs)[argnum]

    return f_selected