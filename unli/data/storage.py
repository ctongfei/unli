from bsddb3 import db
import numpy as np
from typing import TypeVar, Generic, MutableMapping, Iterator, Tuple
from abc import abstractmethod

TK = TypeVar('TK')
TV = TypeVar('TV')


class KeyValueStorage(Generic[TK, TV], MutableMapping[TK, TV]):
    """
    A high-performance key-value storage on disk, powered by BerkeleyDB underneath.
    Generic abstract class.
    """
    def __init__(self, kvs):
        self.kvs = kvs

    @abstractmethod
    def encode_key(self, k: TK) -> bytes:
        pass

    @abstractmethod
    def decode_key(self, k: bytes) -> TK:
        pass

    @abstractmethod
    def encode_value(self, v: TV) -> bytes:
        pass

    @abstractmethod
    def decode_value(self, v: bytes) -> TV:
        pass

    def __setitem__(self, k: TK, v: TV) -> None:
        self.kvs.put(self.encode_key(k), self.encode_value(v))

    def __delitem__(self, k: TK) -> None:
        self.kvs.delete(self.encode_key(k))

    def __getitem__(self, k: TK) -> TV:
        return self.decode_value(self.kvs.get(self.encode_key(k)))

    def __len__(self) -> int:
        return self.kvs.stat()["ndata"]

    def __iter__(self) -> Iterator[TV]:
        for k, v in self.items():
            yield v

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def items(self) -> Iterator[Tuple[TK, TV]]:
        cursor = self.kvs.cursor()
        entry = cursor.first()
        while entry:
            raw_key, raw_value = entry
            yield self.decode_key(raw_key), self.decode_value(raw_value)
            entry = cursor.next()

    def close(self) -> None:
        self.kvs.close()


class StringVectorStorage(KeyValueStorage[str, np.ndarray]):

    def __init__(self, kvs, dtype=np.float32):
        super(StringVectorStorage, self).__init__(kvs)
        self.dtype = dtype

    def encode_key(self, k: str) -> bytes:
        return k.encode()

    def decode_key(self, k: bytes) -> str:
        return k.decode()

    def encode_value(self, v: np.ndarray) -> bytes:
        return v.astype(self.dtype).tobytes()

    def decode_value(self, v: bytes) -> np.ndarray:
        return np.frombuffer(v, dtype=self.dtype)

    @classmethod
    def open(cls, file: str, dtype=np.float32, db_kind=db.DB_BTREE, mode: str = 'r'):
        kvs = db.DB()
        db_mode = {
            'r': db.DB_DIRTY_READ,
            'w': db.DB_CREATE
        }[mode]
        kvs.open(file, None, db_kind, db_mode)
        return cls(kvs, dtype=dtype)


class StringStringStorage(KeyValueStorage[str, str]):

    def __init__(self, kvs):
        super(StringStringStorage, self).__init__(kvs)

    def encode_key(self, k: str) -> bytes:
        return k.encode()

    def decode_key(self, k: bytes) -> str:
        return k.decode()

    def encode_value(self, k: str) -> bytes:
        return k.encode()

    def decode_value(self, k: bytes) -> str:
        return k.decode()

    @classmethod
    def open(cls, file: str, db_kind=db.DB_BTREE, mode: str = 'r'):
        kvs = db.DB()
        db_mode = {
            'r': db.DB_DIRTY_READ,
            'w': db.DB_CREATE
        }[mode]
        kvs.open(file, None, db_kind, db_mode)
        return cls(kvs)
