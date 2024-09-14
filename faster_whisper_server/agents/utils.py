from collections.abc import Callable
from inspect import Parameter, signature
from typing import Any


def filter_function_params(
    func: Callable, *args: tuple[Any], **kwargs: dict[str, Any]
) -> tuple[tuple[Any], dict[str, Any]]:
    """Filters the provided args and kwargs to match the signature of the provided function.

    Args:
    ----
        func: The callable function whose parameters need to be matched.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
    -------
        A tuple containing the filtered args and kwargs.

    """
    # Get the function's signature
    sig = signature(func)
    params = sig.parameters

    # Prepare filtered args and kwargs
    filtered_args: list[Any] = []
    filtered_kwargs: dict[str, Any] = {}

    # Keep track of how many positional arguments we've processed
    arg_index = 0

    # Handle positional arguments first
    for param_name, param in params.items():
        if param.kind in {Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD}:
            # Check if we have an argument for this position
            if arg_index < len(args):
                arg_value = args[arg_index]

                # Check the type hint, if it exists, and match the type
                if param.annotation != Parameter.empty and not isinstance(arg_value, param.annotation):
                    continue
                    # raise TypeError(f"Argument '{param_name}' does not match the expected type {param.annotation}")

                filtered_args.append(arg_value)
                arg_index += 1
            elif param.default != Parameter.empty:
                # Use the default value if no argument is provided
                filtered_args.append(param.default)
            else:
                raise TypeError(f"Missing required positional argument: '{param_name}'")
        elif param.kind == Parameter.KEYWORD_ONLY and param_name in kwargs:
            # Handle keyword-only arguments
            kwarg_value = kwargs.pop(param_name)

            # Type checking for keyword-only arguments
            if param.annotation != Parameter.empty and not isinstance(kwarg_value, param.annotation):
                raise TypeError(f"Keyword argument '{param_name}' does not match the expected type {param.annotation}")

            filtered_kwargs[param_name] = kwarg_value
        elif param.kind == Parameter.VAR_POSITIONAL:
            # Handle *args (variadic positional arguments)
            filtered_args.extend(args[arg_index:])
            arg_index = len(args)  # All positional arguments are consumed
        elif param.kind == Parameter.VAR_KEYWORD:
            # Handle **kwargs (variadic keyword arguments)
            filtered_kwargs.update(kwargs)

    # Ensure no required keyword-only arguments are missing
    for param_name, param in params.items():
        if param.kind == Parameter.KEYWORD_ONLY and param_name not in filtered_kwargs:
            if param.default != Parameter.empty:
                filtered_kwargs[param_name] = param.default
            else:
                raise TypeError(f"Missing required keyword argument: '{param_name}'")
    return tuple(filtered_args), filtered_kwargs
