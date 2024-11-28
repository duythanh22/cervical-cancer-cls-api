import argparse
import time
import asyncio
import aiohttp
import json
import os
from config.config import config

API_URL = config.API_URL
IMAGE_DIR = "test/image_test"
TIMEOUT_THRESHOLD = 1

async def send_request(session, image_path, request_id):
    """
    Sends a single image request to the API.

    Args:
        session (aiohttp.ClientSession): The client session to use.
        image_path (str): Path to the image file.
        request_id (int): ID of the request for tracking.

    Returns:
        dict: The response or error details.
    """
    headers = {"Authorization": f"Bearer {config.API_AUTH_TOKEN}"}
    try:
        with open(image_path, "rb") as image_file:
            data = aiohttp.FormData()
            data.add_field('request', image_file, filename=os.path.basename(image_path), content_type='image/png')

            start_time = time.time()
            async with session.post(API_URL, headers=headers, data=data, ssl=False, timeout=TIMEOUT_THRESHOLD) as response:
                end_time = time.time()
                response_time = end_time - start_time

                if response.status == 200:
                    json_response = await response.json()
                    return {"id": request_id, "result": json_response, "response_time": response_time}
                else:
                    return {
                        "id": request_id,
                        "error": response.status,
                        "detail": await response.text(),
                        "response_time": response_time
                    }
    except asyncio.TimeoutError:
        return {"id": request_id, "error": "Request timed out", "response_time": None}
    except Exception as e:
        return {"id": request_id, "error": str(e), "response_time": None}

async def send_batch_requests(image_paths):
    """
    Sends a batch of image requests asynchronously.

    Args:
        image_paths (list): List of image file paths.

    Returns:
        list: List of results for each request.
    """
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, path, request_id=i+1) for i, path in enumerate(image_paths)]
        return await asyncio.gather(*tasks)

def load_image_paths(image_dir, num_requests):
    """
    Loads image file paths from the specified directory.

    Args:
        image_dir (str): Directory containing image files.
        num_requests (int): Number of requests to prepare.

    Returns:
        list: List of image file paths.
    """
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)
                   if img.lower().endswith(('png', 'jpg', 'jpeg'))]

    if len(image_paths) < num_requests:
        image_paths *= (num_requests // len(image_paths)) + 1

    return image_paths[:num_requests]

def summarize_results(results, start_time, end_time):
    """
    Summarizes the results of the API requests.

    Args:
        results (list): List of result dictionaries.
        start_time (float): Start time of the tests.
        end_time (float): End time of the tests.
    """
    successful_requests = sum(1 for res in results if "error" not in res)
    failed_requests = len(results) - successful_requests
    timeout_requests = sum(1 for res in results if res.get("error") == "Request timed out")
    total_response_time = sum(res["response_time"] for res in results if res.get("response_time") is not None)
    first_timeout_request = next((res["id"] for res in results if res.get("error") == "Request timed out"), None)

    print(f"Total requests sent: {len(results)}")
    print(f"Successful requests: {successful_requests}")
    print(f"Failed requests: {failed_requests}")
    print(f"Timed out requests: {timeout_requests}")
    if successful_requests:
        print(f"Average response time: {total_response_time / successful_requests:.2f} seconds")
    print(f"Total test duration: {end_time - start_time:.2f} seconds")
    if first_timeout_request:
        print(f"First request that timed out: Request {first_timeout_request}")
    else:
        print("No requests timed out.")

def main():
    parser = argparse.ArgumentParser(description="Stress test Cervical Cell Classifier API")
    parser.add_argument("--image_dir", type=str, default=IMAGE_DIR, help="Directory containing the images")
    parser.add_argument("--num-requests", type=int, default=10, help="Number of requests to send")
    args = parser.parse_args()

    image_paths = load_image_paths(args.image_dir, args.num_requests)

    print(f"Sending {len(image_paths)} requests to {API_URL}...")
    loop = asyncio.get_event_loop()

    start_time = time.time()
    results = loop.run_until_complete(send_batch_requests(image_paths))
    end_time = time.time()

    for result in results:
        print(f"Result for request {result['id']}:")
        print(json.dumps(result, indent=2))
        if result["response_time"]:
            print(f"Response time: {result['response_time']:.2f} seconds")
        else:
            print("No response time recorded")
        print()

    summarize_results(results, start_time, end_time)

if __name__ == "__main__":
    main()
