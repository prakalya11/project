import paho.mqtt.client as mqtt

DEVICE_ID = "abc123"

def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe(f"devices/{DEVICE_ID}/control")
def on_message(client, userdata, msg):
    command = msg.payload.decode()
    print(f"Received command: {command}")
    if command == "on":
        print("💡 Light is now ON")
    elif command == "off":
        print("💡 Light is now OFF")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect("your-iot-endpoint.iot.us-west-2.amazonaws.com", 1883, 60)
client.loop_forever()
