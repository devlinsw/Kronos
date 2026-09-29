#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$script_dir/.."

if ! command -v tailscale >/dev/null 2>&1; then
    printf '%s\n' "Tailscale CLI is required for a tailnet-bound launch." >&2
    exit 2
fi

tailscale_ip=$(tailscale ip -4) || {
    printf '%s\n' "Could not read the host's Tailscale IPv4 address." >&2
    exit 2
}
if ! python3 -c 'import ipaddress,sys; ip=ipaddress.ip_address(sys.argv[1].strip()); sys.exit(0 if ip.version == 4 and ip in ipaddress.ip_network("100.64.0.0/10") else 1)' "$tailscale_ip"; then
    printf '%s\n' "Tailscale returned no valid Tailscale IPv4 address." >&2
    exit 2
fi

if [ -n "${KRONOS_TAILSCALE_IP:-}" ] && [ "$KRONOS_TAILSCALE_IP" != "$tailscale_ip" ]; then
    printf '%s\n' "KRONOS_TAILSCALE_IP does not match the host's actual Tailscale IPv4 address." >&2
    exit 2
fi

export KRONOS_TAILSCALE_IP="$tailscale_ip"
exec docker compose "$@"
