"""Run pip with IPv4 DNS results only; TLS certificate verification stays enabled."""
import runpy
import socket
import json
from datetime import datetime, timezone
from pathlib import Path

original_getaddrinfo = socket.getaddrinfo
cache_path = Path.home() / "DriveWAM_Lightweight_Action_Readout/locks/network-dns-cache.json"
cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
fresh = cache and (datetime.now(timezone.utc) - datetime.fromisoformat(cache["captured_at_utc"])).total_seconds() < 21600
cached = cache.get("addresses", {}) if fresh else {}

def ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    # Use this session's verified public CDN addresses only for explicitly cached hosts.
    # The original hostname is still used for HTTPS SNI and certificate validation.
    if host in cached:
        return [record for address in cached[host] for record in
                original_getaddrinfo(address, port, socket.AF_INET, type, proto, flags)]
    return original_getaddrinfo(host, port, socket.AF_INET if family == socket.AF_UNSPEC else family,
                                type, proto, flags)

socket.getaddrinfo = ipv4_getaddrinfo
runpy.run_module("pip", run_name="__main__")
