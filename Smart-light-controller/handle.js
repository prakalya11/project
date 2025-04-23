// handler.js
const AWS = require('aws-sdk');

// Replace with your IoT Core endpoint
const iot = new AWS.IotData({ endpoint: 'your-iot-endpoint.iot.us-west-2.amazonaws.com' });

exports.handler = async (event) => {
    const deviceId = event.queryStringParameters?.deviceId;
    const state = event.queryStringParameters?.state; // "on" or "off"

    if (!deviceId || !['on', 'off'].includes(state)) {
        return {
            statusCode: 400,
            body: JSON.stringify({ error: 'Missing or invalid parameters' })
        };
    }

    const payload = JSON.stringify({ state });
    const params = {
        topic: `devices/${deviceId}/control`,
        payload,
        qos: 0
    };

    try {
        await iot.publish(params).promise();
        console.log(`Published state '${state}' to device ${deviceId}`);
        return {
            statusCode: 200,
            body: JSON.stringify({ message: `Device ${deviceId} turned ${state}` })
        };
    } catch (err) {
        console.error('Publish failed:', err);
        return {
            statusCode: 500,
            body: JSON.stringify({ error: 'Failed to publish to device' })
        };
    }
};

