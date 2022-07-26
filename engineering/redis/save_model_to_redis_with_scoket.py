# !/usr/bin/env python
# -*- coding: utf-8 -*-
import datetime
import socket
import time


class Encoder(object):
    "Encode strings to bytes-like and decode bytes-like to strings"

    def __init__(self, encoding, encoding_errors, decode_responses):
        self.encoding = encoding
        self.encoding_errors = encoding_errors
        self.decode_responses = decode_responses

        self.next = next
        self.unichr = chr
        self.imap = map
        self.izip = zip
        self.xrange = range
        self.basestring = str
        self.unicode = str
        self.long = int

    def encode(self, value):
        "Return a bytestring or bytes-like representation of the value"
        if isinstance(value, (bytes, memoryview)):
            return value
        elif isinstance(value, bool):
            # special case bool since it is a subclass of int
            raise ValueError("Invalid input of type: 'bool'. Convert to a "
                             "bytes, string, int or float first.")
        elif isinstance(value, float):
            value = repr(value).encode()
        elif isinstance(value, (int, self.long)):
            # python 2 repr() on longs is '123L', so use str() instead
            value = str(value).encode()
        elif not isinstance(value, self.basestring):
            # a value we don't know how to deal with. throw an error
            typename = type(value).__name__
            raise ValueError("Invalid input of type: '%s'. Convert to a "
                             "bytes, string, int or float first." % typename)
        if isinstance(value, self.unicode):
            value = value.encode(self.encoding, self.encoding_errors)
        return value

    def decode(self, value, force=False):
        "Return a unicode string from the bytes-like representation"
        if self.decode_responses or force:
            if isinstance(value, memoryview):
                value = value.tobytes()
            if isinstance(value, bytes):
                value = value.decode(self.encoding, self.encoding_errors)
        return value


class NaiveRedisSentinelClient(object):
    def __init__(self, sentinels, master_name, password, user_name=None, timeout=1):
        self.SYM_STAR = b'*'
        self.SYM_DOLLAR = b'$'
        self.SYM_CRLF = b'\r\n'
        self.SYM_EMPTY = b''

        self.BlockingIOError = BlockingIOError

        self._buffer_cutoff = 6000

        self.encoder = Encoder("utf-8", "strict", False)

        self.master_name = master_name

        self.sentinels = sentinels
        self.sentinel_sockets = []

        self.password = password
        self.user_name = user_name

        self.redis_sock = None
        self.timeout = timeout
        self.init_sentinels()
        self.init_redis()

    def init_sentinels(self):
        for _sent_host_port in self.sentinels:
            _sentinel_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            _sentinel_sock.connect((_sent_host_port[0], _sent_host_port[1]))
            _sentinel_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            self.sentinel_sockets.append(_sentinel_sock)

    def master_for(self, master_name):
        for sent in self.sentinel_sockets:
            sent.sendall('SENTINEL get-master-addr-by-name {}\r\n'.format(master_name).encode())
            resp = sent.recv(self._buffer_cutoff)
            _splits = str(resp).split("\\r\\n")
            _h, _p = _splits[2], int(_splits[4])
            return _h, _p

    def init_redis(self):
        self.redis_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _h, _p = self.master_for(self.master_name)
        self.redis_sock.connect((_h, _p))
        self.redis_sock.settimeout(self.timeout)
        self.redis_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if self.user_name is not None:
            self.redis_sock.sendall("auth {} {}\r\n".format(self.user_name, self.password).encode())
        else:
            self.redis_sock.sendall("auth {}\r\n".format(self.password).encode())

    def pack_command(self, *args):
        "Pack a series of arguments into the Redis protocol"
        output = []
        if isinstance(args[0], self.encoder.unicode):
            args = tuple(args[0].encode().split()) + args[1:]
        elif b' ' in args[0]:
            args = tuple(args[0].split()) + args[1:]

        buff = self.SYM_EMPTY.join((self.SYM_STAR, str(len(args)).encode(), self.SYM_CRLF))

        buffer_cutoff = self._buffer_cutoff
        for arg in self.encoder.imap(self.encoder.encode, args):
            arg_length = len(arg)
            if (len(buff) > buffer_cutoff or arg_length > buffer_cutoff
                    or isinstance(arg, memoryview)):
                buff = self.SYM_EMPTY.join(
                    (buff, self.SYM_DOLLAR, str(arg_length).encode(), self.SYM_CRLF))
                output.append(buff)
                output.append(arg)
                buff = self.SYM_CRLF
            else:
                buff = self.SYM_EMPTY.join(
                    (buff, self.SYM_DOLLAR, str(arg_length).encode(),
                     self.SYM_CRLF, arg, self.SYM_CRLF))
        output.append(buff)
        return output

    def set(self, name, value, ex):
        pieces = [b"set", name]
        if isinstance(value, str):
            pieces.append(value.encode())
        else:
            pieces.append(value)

        if ex is not None:
            pieces.append('EX')
            if isinstance(ex, datetime.timedelta):
                ex = int(ex.total_seconds())
            pieces.append(ex)

        _command = self.pack_command(*pieces)

        if self.redis_sock is None:
            self.init_sentinels()
            self.init_redis()

        try:
            for item in _command:
                self.redis_sock.sendall(item)
            _resp = self.redis_sock.recv(2048).decode("utf8").strip()[1:]
        except:
            time.sleep(10)
            try:
                for item in _command:
                    self.redis_sock.sendall(item)
                _resp = self.redis_sock.recv(2048).decode("utf8").strip()[1:]
            except Exception as e:
                raise Exception("Save model to Redis failed: {} => {}".format(self.master_name, e))

        assert _resp == "OK", "Save model to Redis failed: {}".format(self.master_name)


if __name__ == '__main__':
    # model
    model_path = '/Users/jeff/Documents/tensorflow-inference/src/main/resources/fm/model.pb'
    model_key = "fm_v3"
    expire_secs = 120

    with open(model_path, 'rb') as f:
        model = f.read()

    # redis
    sentinels = [("127.0.0.1", 26379)]
    master_name = "ml"
    password = "123456"

    naive_redis_client = NaiveRedisSentinelClient(sentinels, master_name, password)
    naive_redis_client.set(model_key, model, expire_secs)

