"""Shared helpers: polite HTTP, URL canonicalisation, small text utilities."""

from driftfeed.util.http import HttpClient, HttpError, RateLimiter
from driftfeed.util.urls import canonicalize, domain_of

__all__ = ["HttpClient", "HttpError", "RateLimiter", "canonicalize", "domain_of"]
