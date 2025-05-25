# src/feature_extraction.py
#
# ▸ здесь лежит всё, что «добывает» числа (признаки) из URL
# ▸ каждая функция должна вернуть 0 или 1  (или целое число, если логично)

import re, ipaddress
from urllib.parse import urlparse


# ---------- базовые признаки из диплома ----------

def has_ip(url: str) -> int:
    """1, если домен ­— голый IP-адрес"""
    host = urlparse(url if "://" in url else "http://" + url).netloc.split(":")[0]
    try:
        ipaddress.ip_address(host)
        return 1
    except ValueError:
        return 0


def has_at(url: str) -> int:
    return int("@" in url)


def url_length(url: str) -> int:
    return len(url)


def url_depth(url: str) -> int:
    return urlparse(url).path.count("/")


def redirect_dslash(url: str) -> int:
    return int(url.rfind("//") > 7)


def http_token_in_domain(url: str) -> int:
    return int("https" in urlparse(url).netloc.lower())


def tinyurl(url: str) -> int:
    services = r"(bit\.ly|goo\.gl|t\.co|tinyurl|is\.gd|ow\.ly)"
    return int(bool(re.search(services, url)))


def prefix_suffix(url: str) -> int:
    return int("-" in urlparse(url).netloc)


# ---------- новые «дешёвые» признаки ----------

def has_login_word(url: str) -> int:
    return int(bool(re.search(r"login|signin|account|verify", url, re.I)))


def count_digits(url: str) -> int:
    return sum(c.isdigit() for c in url)


def has_https_token(url: str) -> int:
    """1, если в ПУТИ (а не в протоколе) намешано 'https'"""
    path_plus_query = urlparse(url).path + urlparse(url).query
    return int("https" in path_plus_query.lower())


# ---------- регистрируем все функции ----------

FEATURE_FUNCS = [
    has_ip,
    has_at,
    url_length,
    url_depth,
    redirect_dslash,
    http_token_in_domain,
    tinyurl,
    prefix_suffix,
    has_login_word,      # новые
    count_digits,        # новые
    has_https_token,     # новые
]

FEATURE_NAMES = [f.__name__ for f in FEATURE_FUNCS]

def extract(url: str) -> list:
    """Вернёт список чисел – признаки для одного URL"""
    return [f(url) for f in FEATURE_FUNCS]
