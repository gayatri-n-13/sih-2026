import requests
import time
import os
from tests.synthetic.generator import SyntheticLunarDataGenerator

def test_integration():
    # 1. Setup data
    gen = SyntheticLunarDataGenerator()
    data = gen.create_dataset('int_test_api', theta=0.1, scale=1.05, tx=10, ty=-5)

    # 2. Trigger match
    payload = {
        "job_id": "int_test_api",
        "pyramid_source_ref": data["pyramid_source"],
        "pyramid_reference_ref": data["pyramid_ref"],
        "invariant_channels_source_ref": data["invariant_channels_source"],
        "invariant_channels_reference_ref": data["invariant_channels_ref"],
    }
    print("Sending match request...")
    resp = requests.post("http://localhost:8001/match", json=payload)
    print(f"Response: {resp.json()}")

    # 3. Poll status
    for _ in range(10):
        time.sleep(2)
        status_resp = requests.get("http://localhost:8001/status/int_test_api")
        status = status_resp.json()
        print(f"Status: {status}")
        if status["status"] == "COMPLETED":
            print("Success!")
            return
        if status["status"] == "FAILED":
            print(f"Failed: {status['error']}")
            return

    print("Timed out")

if __name__ == "__main__":
    test_integration()
