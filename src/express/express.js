require('dotenv').config()
const express = require('express')
const bodyParser = require('body-parser');
const CORS = require('cors');

const app = express()

app.use(CORS());
app.use(bodyParser.json({ strict: false, limit: '50mb' }));

app.get('/', (req, res) => {
  res.send('Hello world!')
})

app.listen(3020, async () => {
  try {
    console.log('listening on 3020')
  } catch (error) {
    console.error('app.listen error:', error)
    console.log(error)
  }
})
