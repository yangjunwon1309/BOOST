#!/usr/bin/env bash

# Source this file and call ensure_televuer_tls before starting Televuer.
ensure_televuer_tls() {
  local default_dir="${HOME}/.config/xr_teleoperate"
  local default_cert="$default_dir/cert.pem"
  local default_key="$default_dir/key.pem"
  local hostname_value san_entries ip tmp_dir old_umask

  # Keep a working explicit pair, including one supplied by an existing
  # Televuer checkout or deployment configuration.
  if [[ -n "${PATH_TELEVUER_CERT:-}" && -n "${PATH_TELEVUER_KEY:-}" \
    && -r "$PATH_TELEVUER_CERT" && -r "$PATH_TELEVUER_KEY" ]]; then
    export PATH_TELEVUER_CERT PATH_TELEVUER_KEY
    return 0
  fi

  # Respect Televuer's native environment variables if they point to a pair.
  if [[ -n "${XR_TELEOP_CERT:-}" && -n "${XR_TELEOP_KEY:-}" \
    && -r "$XR_TELEOP_CERT" && -r "$XR_TELEOP_KEY" ]]; then
    export PATH_TELEVUER_CERT="$XR_TELEOP_CERT"
    export PATH_TELEVUER_KEY="$XR_TELEOP_KEY"
    return 0
  fi

  # A source checkout stores the generated pair in its root by default.
  if [[ -n "${PATH_TELEVUER_ROOT:-}" \
    && -r "$PATH_TELEVUER_ROOT/cert.pem" && -r "$PATH_TELEVUER_ROOT/key.pem" ]]; then
    export PATH_TELEVUER_CERT="$PATH_TELEVUER_ROOT/cert.pem"
    export PATH_TELEVUER_KEY="$PATH_TELEVUER_ROOT/key.pem"
    return 0
  fi

  # Fall back to Televuer's shared per-user configuration directory.
  if [[ -r "$default_cert" && -r "$default_key" ]]; then
    export PATH_TELEVUER_CERT="$default_cert"
    export PATH_TELEVUER_KEY="$default_key"
    return 0
  fi

  if [[ -n "${PATH_TELEVUER_CERT:-}" || -n "${PATH_TELEVUER_KEY:-}" ]]; then
    echo "[Televuer] configured TLS files are missing or unreadable; using $default_dir" >&2
  fi
  if ! command -v openssl >/dev/null 2>&1; then
    echo "[Televuer] openssl is required to create the default TLS certificate pair." >&2
    return 2
  fi

  mkdir -p "$default_dir"
  chmod 700 "$default_dir"
  tmp_dir="$(mktemp -d "$default_dir/.tls.XXXXXX")"
  hostname_value="$(hostname 2>/dev/null || printf 'localhost')"
  san_entries="DNS:localhost,DNS:$hostname_value,IP:127.0.0.1"
  for ip in $(hostname -I 2>/dev/null || true); do
    san_entries+=",IP:$ip"
  done

  old_umask="$(umask)"
  umask 077
  if ! openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout "$tmp_dir/key.pem" \
    -out "$tmp_dir/cert.pem" \
    -subj "/CN=$hostname_value" \
    -addext "subjectAltName=$san_entries" >/dev/null 2>&1; then
    umask "$old_umask"
    rm -rf "$tmp_dir"
    echo "[Televuer] failed to generate TLS files under $default_dir." >&2
    return 2
  fi
  umask "$old_umask"

  chmod 600 "$tmp_dir/key.pem"
  chmod 644 "$tmp_dir/cert.pem"
  mv "$tmp_dir/key.pem" "$default_key"
  mv "$tmp_dir/cert.pem" "$default_cert"
  rmdir "$tmp_dir"

  export PATH_TELEVUER_CERT="$default_cert"
  export PATH_TELEVUER_KEY="$default_key"
  echo "[Televuer] created TLS certificate and key at $default_dir."
  echo "[Televuer] if the headset rejects the connection, trust/import cert.pem on the headset."
}
