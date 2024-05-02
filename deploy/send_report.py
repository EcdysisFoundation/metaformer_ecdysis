import argparse
import json
import logging
import requests

def upload_json(json_file: str, url:str):
    """ Send the JSON data to the url and validate the response
    Args:
        json_file: Path to pregenerated report data
        url: Url of the endpoint
    Raises:
        ValueError: if there added count is different from expected, or status_code != 200
        JSONDecodeError: if the response isn't a valid JSON
     """
    # Load JSON data from file
    with open(json_file) as file:
        json_data = json.load(file)
    item_count = len(json_data["data"])
    if not url.startswith("http://"):
        url = "http://" + url
    logging.warning(f"Sending {item_count} elements...")
    # Make the POST request to the REST endpoint
    response = requests.post(url, json=json_data)

    # Validate the response
    if response.status_code == 200:
        try:
            response_data = response.json()
            added_items = response_data.get('added')
            # validate response
            if added_items is not None:
                if added_items == item_count:
                    logging.warning(f"{response.status_code}: {item_count} entries added (OK).")
                else:
                    raise ValueError(f"{response.status_code}: Received {added_items}, expected {item_count}")
            else:
                raise ValueError(f"{response.status_code}: No 'added' key in reply:\n{response_data}")
        except json.JSONDecodeError:
            raise ValueError(f"{response.status_code}: Invalid response JSON:\n{response.text}")
    else:
        raise ValueError(f"Request failed with status code {response.status_code}:{response.text}")

def parse_arguments():
    """ Parse command line arguments """
    parser = argparse.ArgumentParser(description="Sends the report JSON data to the endpoint")
    parser.add_argument('json', help='Path to the JSON file')
    parser.add_argument('url', help='URL of the REST endpoint')
    return parser.parse_args()


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()
    try:
        # Upload JSON and validate response
        upload_json(args.json, args.url)
    except ValueError as err:
        logging.error(err)
        exit(1)