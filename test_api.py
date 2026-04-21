"""
Test script for AI Insights Dashboard.
Run this to test the API without the web interface.
"""
import requests
import time
import sys
import os


API_BASE = "http://localhost:8000"


def create_sample_csv():
    """Create a sample CSV file for testing."""
    csv_content = """name,age,salary,department,years_experience
John Smith,32,75000,Engineering,8
Jane Doe,28,68000,Marketing,5
Bob Johnson,45,92000,Engineering,15
Alice Brown,36,81000,Sales,10
Charlie Wilson,29,65000,Marketing,4
Diana Lee,41,88000,Engineering,12
Eva Martinez,33,72000,Sales,7
Frank Taylor,38,79000,Engineering,9
Grace Kim,27,62000,Marketing,3
Henry Chen,44,95000,Sales,18
Ivy Wang,31,70000,Engineering,6
Jack Robinson,35,76000,Sales,8
Karen White,30,67000,Marketing,5
Leo Garcia,42,90000,Engineering,14
Mia Thompson,26,58000,Sales,2
"""
    with open("sample_data.csv", "w") as f:
        f.write(csv_content)
    return "sample_data.csv"


def test_health():
    """Test the health endpoint."""
    print("\n[1] Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE}/health")
        print(f"    Status: {response.status_code}")
        print(f"    Response: {response.json()}")
        return True
    except Exception as e:
        print(f"    Error: {e}")
        return False


def test_upload(file_path):
    """Test file upload."""
    print(f"\n[2] Uploading file: {file_path}")

    with open(file_path, "rb") as f:
        response = requests.post(
            f"{API_BASE}/upload",
            files={"file": (os.path.basename(file_path), f)}
        )

    print(f"    Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"    Job ID: {data['job_id']}")
        print(f"    Status: {data['status']}")
        return data['job_id']
    else:
        print(f"    Error: {response.text}")
        return None


def test_status(job_id, poll_interval=3):
    """Poll for job status until complete."""
    print(f"\n[3] Polling job status: {job_id}")

    while True:
        response = requests.get(f"{API_BASE}/status/{job_id}")
        data = response.json()

        print(f"    Progress: {data['progress']}% - {data['message']}")

        if data['status'] == 'completed':
            print("    ✓ Analysis complete!")
            return True
        elif data['status'] == 'failed':
            print(f"    ✗ Job failed: {data['message']}")
            return False

        time.sleep(poll_interval)


def test_result(job_id):
    """Get analysis result."""
    print(f"\n[4] Fetching analysis result...")

    response = requests.get(f"{API_BASE}/result/{job_id}")
    data = response.json()

    print(f"    Summary: {data.get('summary', 'N/A')[:200]}...")
    print(f"    Insights: {len(data.get('insights', []))} items")
    print(f"    Recommendations: {len(data.get('recommendations', []))} items")

    return data


def test_download(job_id, output_path="test_report.pdf"):
    """Download PDF report."""
    print(f"\n[5] Downloading PDF report...")

    response = requests.get(f"{API_BASE}/download/{job_id}")

    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"    ✓ Saved to: {output_path}")
        return True
    else:
        print(f"    ✗ Error: {response.text}")
        return False


def main():
    print("=" * 60)
    print("  AI Insights Dashboard - API Test Script")
    print("=" * 60)

    # Create sample file if not provided
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        print("\nNo file provided, creating sample CSV...")
        test_file = create_sample_csv()

    # Run tests
    if not test_health():
        print("\n! Server not running. Start with: python main.py")
        return

    job_id = test_upload(test_file)
    if not job_id:
        return

    if not test_status(job_id):
        return

    result = test_result(job_id)
    test_download(job_id)

    print("\n" + "=" * 60)
    print("  Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()