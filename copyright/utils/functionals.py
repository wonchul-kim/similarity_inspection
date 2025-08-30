import os, re, io, sys, time, json, math, base64, hashlib

def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()