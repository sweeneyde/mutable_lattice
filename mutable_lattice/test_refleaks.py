import sys
try:
    rc = sys.gettotalrefcount()
except AttributeError as e:
    raise RuntimeError("hunt_refleaks must be run on a debug build of Python") from e

def main():
    import mutable_lattice.test
    sys.modules["test.test_mutable_lattice_dynamic"] = mutable_lattice.test
    from test.libregrtest.main import main as _main
    _main(["test_mutable_lattice_dynamic"], huntrleaks=(3,3,"reflog.txt"))

if __name__ == "__main__":
    main()
