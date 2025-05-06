import random

def lambda_handler(event, context):
    # if the user doesn't provide num_points to simulate, assume 10k
    num_points = event.get('num_points', 10000)
    points_inside_circle = 0

    for _ in range(num_points):
        x, y = random.random(), random.random()
        if x**2 + y**2 <= 1:
            points_inside_circle += 1

    pi_estimate = 4 * points_inside_circle / num_points
    return {
        'statusCode': 200,
        'body': {
            'pi_estimate': pi_estimate,
            'num_points': num_points
        }
    }