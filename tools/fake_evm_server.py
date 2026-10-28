"""
fake_evm_server.py - pretend to be the AMS EVM GUI's TCP port, for testing without the sensor.

    python tools/fake_evm_server.py path/to/recording.csv [--port 39998] [--loop]

Answers every '(m0)' with the next record of the recording (the lines from one #ITT row to the next), so the
live tab behaves as it would with the real sensor. Any other command is acknowledged and ignored.
"""
import argparse
import socket
import time


def records(path):
    recs, cur = [], []
    for line in open(path, errors="ignore"):
        line = line.rstrip("\r\n")
        if not line.strip() or line.startswith("sep="):
            continue
        if line.startswith("#ITT") and line.split(";")[1].strip().isdigit() and cur:
            recs.append(cur); cur = []
        cur.append(line)
    if cur:
        recs.append(cur)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file"); ap.add_argument("--port", type=int, default=39998); ap.add_argument("--loop", action="store_true")
    ap.add_argument("--period", type=float, default=0.23, help="seconds per measurement (sensor speed)")
    a = ap.parse_args()
    recs = records(a.file)
    print(f"{len(recs)} records from {a.file}; listening on 127.0.0.1:{a.port}")
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", a.port)); srv.listen(1)
    while True:
        conn, _ = srv.accept()
        print("client connected", flush=True)
        conn.sendall(b"fake EVM ready\n")
        i, buf = 0, b""
        try:
            while True:
                d = conn.recv(1024)
                if not d:
                    break
                buf += d
                while b")" in buf:
                    cmd, buf = buf.split(b")", 1)
                    cmd = cmd.strip().lstrip(b"(").lower()   # EVM command codes are case-insensitive
                    if cmd == b"m0":
                        if i >= len(recs):
                            if not a.loop:
                                continue
                            i = 0
                        time.sleep(a.period)
                        conn.sendall(("\n".join(recs[i]) + "\n").encode()); i += 1
                    else:
                        print("command", cmd.decode(), flush=True); conn.sendall(f"ok {cmd.decode()}\n".encode())
        except (ConnectionResetError, BrokenPipeError):
            pass
        print("client left")


if __name__ == "__main__":
    main()
