"""Trava de processo liberada pelo sistema ao encerrar a execução."""
import os
from pathlib import Path
import time


def held(path):
    """Consulta a trava real, sem confiar no estado persistido de uma execução."""
    path = Path(path)
    if not path.exists():
        return False
    with path.open('r+b') as stream:
        if os.fstat(stream.fileno()).st_size == 0:
            return False
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                return True
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return False


def acquire(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    stream=os.fdopen(os.open(path,os.O_RDWR|os.O_CREAT), 'r+b')
    if os.fstat(stream.fileno()).st_size==0:
        stream.write(b'0');stream.flush()
    while True:
        stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            return stream
        except OSError:
            time.sleep(10)
