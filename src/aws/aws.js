require("dotenv").config({ path: `${__dirname}/../../.env` });
const fs = require('fs')
const AWS = require('aws-sdk');

const {
  AmazonWSAccessKeyId,
  AmazonWSSecretKey,
} = process.env

AWS.config.update({
  accessKeyId: AmazonWSAccessKeyId,
  secretAccessKey: AmazonWSSecretKey,
  region: 'us-east-1'
});

const s3Client = new AWS.S3()

const streamFromS3 = (key, bucket, dest) => {
  return new Promise((resolve, reject) => {
    const params = { Bucket: bucket, Key: key }
    const s3Stream = s3Client.getObject(params).createReadStream();
    const fileStream = fs.createWriteStream(dest);
    s3Stream.on('error', reject);
    fileStream.on('error', reject);
    fileStream.on('close', () => { resolve(dest); });
    s3Stream.pipe(fileStream);
  });
}

const uploadToS3 = (Key, Bucket, Body, ContentType, Metadata, ACL) => {
  const params = {
    Key,
    Bucket,
    Body,
    ContentType,
    Metadata,
    ACL,
  }

  return new Promise((resolve, reject) => {
    s3Client.upload(params, (err, data) => {
      if (err) {
        return reject(err)
      } else {
        return resolve(data)
      }
    })
  })
}

module.exports = {
  streamFromS3,
  uploadToS3,
  s3Client,
}