
import subprocess
import sys
import time
import os

SERVER_PORT = "8002"
os.environ["HF_HOME"] = "H:/gen_ai/llm"

def main():
    procs = []
    
    # 1. Server
    print("Starting Server...")
    procs.append(subprocess.Popen([sys.executable, "-m", "uvicorn", "server.coordinator:app", "--port", SERVER_PORT], cwd="g:\\python\\hdgAtHome"))
    time.sleep(3)
    
    # 2. Worker 1
    print("Starting Worker 1...")
    procs.append(subprocess.Popen([sys.executable, "-u", "client/worker.py"], cwd="g:\\python\\hdgAtHome"))
    
    # 3. Worker 2
    print("Starting Worker 2...")
    procs.append(subprocess.Popen([sys.executable, "-u", "client/worker.py"], cwd="g:\\python\\hdgAtHome"))
    
    print("Cluster Running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
            # Check if alive
            if procs[0].poll() is not None:
                print("Server died!")
                break
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        for p in procs:
            p.terminate()

if __name__ == "__main__":
    main()
