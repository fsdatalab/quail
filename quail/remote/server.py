"""The `quail-server` program: Quail Server, which the Rust module runs."""

import sys

import _quail_native


def main() -> None:
    """Run Quail Server with the command line's arguments until it is stopped."""
    try:
        _quail_native.serve(sys.argv[1:])
    except ValueError as error:
        sys.exit(str(error))
    except KeyboardInterrupt:
        # Python also saw the Ctrl-C that stopped the server
        pass


if __name__ == "__main__":
    main()
