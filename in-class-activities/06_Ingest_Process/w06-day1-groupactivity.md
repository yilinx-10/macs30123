## 6M Group Activity
Today, we're going to practice designing streaming data solutions that solve the following problem.

In groups, you should address the following questions. When you are done drafting your answers as a group, post them below with the names of your group members up at the top of the post.

1. You have been tasked by your city government with designing a scalable AWS system that can do all of the following. How would you architect such a solution? What storage, compute, and data movement resources would you employ?
   * (1) Collect streaming data from Twitter and Instagram related to public responses to a particular government policy (for the sake of this exercise, assume that everyone uses a common hashtag related to the policy in their posts).
   * (2) Archive the streaming data in its raw form for later analysis by city data scientists.
   * (3) Process the streaming (text) data to identify real-time sentiment surrounding the policy, using this processed data to power an interactive dashboard for policy makers. Don't worry about the mechanics of the interactive dashboard. Just assume that you have an EC2 instance that acts as a server for the dashboard.
   * (4) Automatically send a weekly report to policy makers (via email) each Monday morning that includes a plot of the previous week's sentiment time series on a day-by-day basis.

**Instructor Solution:**
1. Cluster of EC2 instances connected to streaming Twitter/Instagram APIs for that specific hashtag. These producer instances write data into a Kinesis stream.
2. Lambda/EC2 consumer instances reads data from Kinesis stream as it comes in and writes it in batches (i.e. 100 records, 1 hr of data, etc.) to S3 for later analysis. Note: could also use Kinesis Firehose (which is a serverless approach that you might have seen in the documentation that will allow you to do this same thing -- automatically sending data in some form into storage like S3)
3. Lambda/EC2 reads data from Kinesis stream and pre-processes it (computing sentiment -- could use Comprehend here, or PySpark code here, along with any other needed metadata) and sends it to a Redshift Cluster in batch for data storage and for powering an interactive dashboard (from EC2 instance as stated in the prompt, AWS' built-in Quicksight, etc.). Could also draw on data in S3 bucket from (2) in setting up a serverless QuickSight dashboard (via Athena, etc.) if you didn't think there would be too much interactive traffic, but this would not be as scalable as having a devoted Redshift cluster for making a lot of concurrent analytical queries on the data on a regular basis (say you have a large team of data scientists that are all working on the data at the same time).
4. Every week: Have a Lambda/EC2 instance query Redshift cluster for previous week's sentiment time series and then publish that data to an SNS topic that has the relevant policy makers' emails subscribed to it -- the process you practiced in the Boto3 DataCamp tutorial last week.

2. After this workflow has been implemented, your research assistant notices that it is performing slowly and identifies the Kinesis stream as the problem. Note that your solution requires 10 MB/s throughput. Which of the following would you do to solve this problem? Why?
   * (a) Use AWS Lambda to preprocess the data and transform the records into a simple format, such as CSV or JSON.
   * (b) Run the MergeShard command to reduce the number of shards that the consumer can more easily process.
   * (c ) Run the UpdateShardCount command to increase the number of shards in the stream.

**Instructor Solution:**
(c ) You probably have too few Kinesis shards in your stream. Each shard can only handle 1 MB/s write and 2 MB/s read, so you would likely need at least 10 shards and then need to monitor the bandwidth used to scale up the number of shards as necessary.

Preprocessing, as in (a), might help, but will probably give you diminishing returns with a format like Twitter text data -- this might be a good strategy if you're working with big image data, though.
