import boto3
import json

dynamodb = boto3.resource('dynamodb')

def lambda_handler(event, context):
    source = None
    if 'Records' in event:
        if 'Sns' in event['Records'][0]:
            # If data coming from SNS, grab Message data
            # Also get TopicArn for Demo purposes
            source = event['Records'][0]['Sns']['TopicArn']
            event = json.loads(event['Records'][0]['Sns']['Message'])
        else:
            # If data being pulled from SQS queue, need to grab "body"
            # of message form SQS output
            # For demo, also collect eventSourceARN
            source = event['Records'][0]['eventSourceARN']
            event = json.loads(event['Records'][0]['body'])

    # enter data into a DB
    # 'id' is concatenation of location/sensor/timestamp for sake of demo
    # in real DB, 'timestamp' would be a good sorting key
    table = dynamodb.Table('ed_db')
    table.put_item(
       Item={
            'id': event['location'] + event['sensor'] + event['timestamp'],
            'db': event['db'],
            'source': source
        }
    )

    return {'StatusCode': 200}
