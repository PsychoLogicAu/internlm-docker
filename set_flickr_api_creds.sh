#!/usr/bin/env bash

# This script updates two fields in the file /flickr_scraper/flickr_scraper.py to set the API key and secret.
# These are sourced from the environment variables, FLICKR_API_KEY and FLICKR_API_SECRET in the file /data/flickr_scraper.env

# Check if the environment file exists
if [ ! -f /data/flickr_scraper.env ]; then
    echo "The file /data/flickr_scraper.env does not exist. Please create it and set the environment variables FLICKR_API_KEY and FLICKR_API_SECRET"
    exit 1
fi

# Source the environment variables
source /data/flickr_scraper.env

# Check if the environment variables are set
if [ -z "$FLICKR_API_KEY" ]; then
    echo "FLICKR_API_KEY is not set. Please set it in the file /data/flickr_scraper.env"
    exit 1
fi

if [ -z "$FLICKR_API_SECRET" ]; then
    echo "FLICKR_API_SECRET is not set. Please set it in the file /data/flickr_scraper.env"
    exit 1
fi

echo "Setting the API key to $FLICKR_API_KEY and the API secret to $FLICKR_API_SECRET"

# Fields to update:
# key = ""  # Flickr API key https://www.flickr.com/services/apps/create/apply
# secret = ""

# Update the file /flickr_scraper/flickr_scraper.py, replace YOUR_API_KEY and YOUR_API_SECRET with the environment variable values
sed -i "s/key = \"\"/key = \"$FLICKR_API_KEY\"/g" /flickr_scraper/flickr_scraper.py
sed -i "s/secret = \"\"/secret = \"$FLICKR_API_SECRET\"/g" /flickr_scraper/flickr_scraper.py
