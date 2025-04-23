const { handler } = require('../handler');

test('Returns 400 for missing params', async () => {
    const res = await handler({ queryStringParameters: {} });
    expect(res.statusCode).toBe(400);
});

test('Returns 200 for valid request', async () => {
    // Mock AWS SDK
    require('aws-sdk').IotData.prototype.publish = jest.fn(() => ({
        promise: () => Promise.resolve()
    }));

    const res = await handler({
        queryStringParameters: {
            deviceId: 'abc123',
            state: 'on'
        }
    });

    expect(res.statusCode).toBe(200);
    expect(JSON.parse(res.body).message).toContain('turned on');
});
