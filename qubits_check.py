import boto3

# Create a Braket client
braket_client = boto3.client('braket')

# List quantum devices using filters parameter
response = braket_client.search_devices(
    filters=[]  # Pass an empty list instead of a dictionary
)

# Print the devices and their details
for device in response.get('devices', []):
    print(f"Device: {device}")
    # Safely access device attributes, use .get() method to avoid KeyError
    print(f"Device ARN: {device.get('deviceArn', 'N/A')}")
    print(f"Device type: {device.get('deviceType', 'N/A')}")
    print(f"Device status: {device.get('deviceStatus', 'N/A')}")
    print("-" * 30)
