
import cffi
ffibuilder = cffi.FFI()
ffibuilder.cdef("""void fibo(int *, int *);""")
ffibuilder.set_source("fibonacci._API_fibo", r"""
  void fibo(int *a, int *b)
  {
    int next;
    next = *a + *b;
    *b = *a;
    *a = next;
  }
  """)


if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
